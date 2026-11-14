"""Turn-level pipeline: N intents in one utterance.

Two layers of decompose (do not merge them):
- This module: ask / plan / modify / execute / client (git, ops)
- planner.py: multiple *ask* questions (two APIs, 流程+字段) inside the ask batch

Splitting strategy (LLM primary, heuristic fallback):
- ``llm_turn_plan`` asks the LLM to split a **compound long utterance** into
  ordered intent stages (ask/plan/modify/execute/client). It only fires when
  the heuristic produced a single stage AND the text looks genuinely compound
  (multiple intent signals without explicit 先/然后/再 connectives) — so we do
  NOT burn an LLM call on every short/simple turn.
- Story guards (add-api / change-plan / compose-code) run BEFORE the LLM: those
  utterances are ONE ask with orchestration and must never be split
  (desktop-chat-fa14438a45aa / ed1e15401152 / b58bb8b77eea regressions).
- Any LLM failure / non-JSON / invalid layers falls back to
  ``heuristic_turn_plan`` unchanged (source="heuristic" or "heuristic_fallback").
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("app.agent.runtime.turn_plan")

_ORDER_SPLIT_RE = re.compile(
    r"(?:然后|看完后|看完再|接着|之后|做完后|最后|再帮我|再给我|然后再|再去|，再)"
)
_NUM_SPLIT_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\s*[\.\)、]|[（(]\d+[）)]|[一二三四五六七八九十]\s*[、.])\s*"
)
_GIT_RE = re.compile(
    r"(?i)(?:提交并推送|推送代码|提交代码|帮我提交|帮我推送|"
    r"git\s*push|git\s*commit|commit\s+and\s+push|"
    r"(?:提交|推送).{0,8}(?:远程|仓库|github|gitlab)|"
    r"(?:push|commit).{0,12}(?:code|代码))"
)
_ASK_RE = re.compile(
    r"看看|看一下|看下|做了什么|做什么|是什么|有没有|流程|校验|"
    r"怎么走|主链|源码|接口|字段|状态|哪块有问题|日志"
)
_MODIFY_RE = re.compile(
    r"直接改代码|落地补丁|帮我改代码|帮我实现|apply\s*patch|写入代码|按方案实现"
)
_PLAN_RE = re.compile(
    r"改哪些|修改哪些|要改什么|改动点|改动清单|影响面|影响范围|DDL|都要修改|"
    r"几种方案|对比利弊|方案对比|给我方案"
)
_EXECUTE_RE = re.compile(
    r"编译一下|帮我编译|跑一下测试|自测|本机编译|编译验证|改完自测|改完编译|"
    r"修一下报错|修编译错误"
)

# A single "add_api story": 新增/生成/实现 接口 + business-step orchestration
# (先…然后…再…最后…). This is ONE ask with service-body steps, NOT several
# independent questions — splitting it re-runs the whole scaffold per segment
# (desktop-chat-fa14438a45aa: 4× duplicate GoodsQuery scaffold).
_ADD_API_STORY_RE = re.compile(
    r"(?:新增|生成|实现|开发|创建|写)(?:一个|个|一)?(?:接口|API|功能|方法|endpoint)"
    r".{0,60}(?:知识库|文档|接口文档)"
)
_ADD_API_STORY_STEPS_RE = re.compile(r"先|然后|再|最后|接着|随后")

# A single "change-plan story": 加字段/入参 + (落库/改动清单/透传/DDL) is ONE ask
# even when phrased with 然后/再 — splitting loses the field name in the second
# segment and re-runs a contradictory placeholder plan
# (desktop-chat-ed1e15401152: N1 labor vs N2 newField duplicate lists).
_CHANGE_PLAN_STORY_RE = re.compile(
    r"(?:加个|加一|增加|新增|添加)(?:字段|入参|参数|字段名)"
)
_CHANGE_PLAN_STORY_TAIL_RE = re.compile(
    r"落库|入库|持久化|改动清单|修改哪些|改哪些|要改什么|透传|DDL|改动"
)

# A single "compose-code story": 给/写/生成 一段代码 + 多步查询编排
# (先查A → 查到后再查B → …). This is ONE compose_code ask whose lookup_steps
# describe the query pipeline — splitting on 然后/再 turns it into two
# independent code tasks that each re-run the full PEV + LLM generation
# (desktop-chat-b58bb8b77eea: 2× compose PEV, 249s, placeholder LoanCoupon).
# Two-part check: code verb up front (给我一段/写一段/生成一个 …) AND a code
# noun anywhere in the utterance (代码/方法/程序). 「再给我改动清单」matches
# neither part, so mixed-intent turns still split normally.
_COMPOSE_STORY_VERB_RE = re.compile(
    r"(?:给我一段|给我一个|给我写|帮我写|帮我生成|写一段|写个|生成一段|生成一个|"
    r"来一段|来一个|给我一段代码|给我一个方法)"
)
_COMPOSE_STORY_CODE_NOUN_RE = re.compile(r"代码|方法|Java|java|程序|逻辑")
_COMPOSE_STORY_STEPS_RE = re.compile(r"先|然后|再|查到|之后|接着|随后|根据结果|按结果")


def _is_add_api_story(question: str) -> bool:
    """True when the utterance is one add-API request with step orchestration."""
    q = (question or "").strip()
    if not _ADD_API_STORY_RE.search(q):
        return False
    # Steps (先/然后/再…) after the API intent are service-body orchestration of
    # the SAME ask — do not split them into independent tasks.
    return bool(_ADD_API_STORY_STEPS_RE.search(q))


def _is_change_plan_story(question: str) -> bool:
    """True when the utterance is one add-field/persist change plan request.

    Splitting such an ask on 然后/再 drops the field name from later segments
    (they fall back to the newField placeholder) and duplicates the whole plan.
    """
    q = (question or "").strip()
    return bool(_CHANGE_PLAN_STORY_RE.search(q) and _CHANGE_PLAN_STORY_TAIL_RE.search(q))


def _is_compose_code_story(question: str) -> bool:
    """True when the utterance is one code-composition ask with query steps.

    「给我一段代码 通过X查A，查到结果后再查B和C」是单个 compose_code 任务
    （lookup_steps 描述查询流水线）。按 然后/再 拆分会把一步查询拆成两个独立
    “查代码”任务，各自重跑完整 PEV + LLM 生成（desktop-chat-b58bb8b77eea：
    两次 compose PEV、耗时 249s、产出 LoanCoupon/LoanApproval 占位符）。
    """
    q = (question or "").strip()
    if not _COMPOSE_STORY_VERB_RE.search(q):
        return False
    if not _COMPOSE_STORY_CODE_NOUN_RE.search(q):
        return False
    return bool(_COMPOSE_STORY_STEPS_RE.search(q))

QA_LAYERS = frozenset({"ask", "plan"})
POST_LAYERS = frozenset({"modify", "execute", "client"})

# Backward-compatible aliases used by older call sites / tests.
_LAYER_ALIASES = {
    "impact": "plan",
    "options": "plan",
    "implement": "modify",
    "verify": "execute",
    "repair": "execute",
}


@dataclass
class TurnNode:
    id: str
    layer: str  # ask | plan | modify | execute | client
    title: str
    question: str
    depends_on: list[str] = field(default_factory=list)
    client_mode: str = ""  # direct | code
    needs_confirm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer,
            "title": self.title,
            "question": self.question,
            "depends_on": list(self.depends_on),
            "client_mode": self.client_mode,
            "needs_confirm": self.needs_confirm,
        }


@dataclass
class TurnPlan:
    nodes: list[TurnNode]
    source: str = "heuristic"

    @property
    def qa_nodes(self) -> list[TurnNode]:
        return [n for n in self.nodes if n.layer in QA_LAYERS]

    @property
    def post_qa_nodes(self) -> list[TurnNode]:
        return [n for n in self.nodes if n.layer in POST_LAYERS]

    @property
    def qa_text(self) -> str:
        return " ".join(n.question for n in self.qa_nodes).strip()

    @property
    def ask_text(self) -> str:
        return self.qa_text

    @property
    def has_ask(self) -> bool:
        return any(n.layer == "ask" for n in self.nodes) or bool(self.qa_nodes)

    @property
    def has_qa(self) -> bool:
        return bool(self.qa_nodes)

    @property
    def has_post_qa(self) -> bool:
        return bool(self.post_qa_nodes)

    @property
    def follow_on_goal(self) -> str:
        if not self.has_qa:
            return ""
        clients = [n for n in self.post_qa_nodes if n.layer == "client"]
        return " ".join(n.question for n in clients).strip()

    @property
    def follow_on_client_mode(self) -> str:
        if not self.has_qa:
            return ""
        if any(n.layer == "modify" for n in self.post_qa_nodes):
            return "code"
        if any(n.layer == "client" for n in self.post_qa_nodes):
            return "direct"
        if any(n.layer == "execute" for n in self.post_qa_nodes):
            return "code"
        return ""

    @property
    def intent_layers(self) -> list[str]:
        return [n.layer for n in self.nodes]

    @property
    def needs_sequential_qa(self) -> bool:
        """Two+ QA stages must run in utterance order, not concatenated."""
        return len(self.qa_nodes) >= 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "source": self.source,
            "ask_text": self.ask_text,
            "qa_text": self.qa_text,
            "follow_on_goal": self.follow_on_goal,
            "follow_on_client_mode": self.follow_on_client_mode,
            "has_ask": self.has_ask,
            "intent_layers": self.intent_layers,
        }


_LAYER_TITLE = {
    "ask": "查代码",
    "plan": "改动清单/方案",
    "modify": "改代码（需确认）",
    "execute": "编译/自测（需确认）",
    "client": "本机操作（需确认）",
}


def normalize_layer(layer: str) -> str:
    raw = (layer or "").strip().lower()
    return _LAYER_ALIASES.get(raw, raw)


def classify_layer(text: str) -> str:
    """Structural ops/QA layer for turn stages.

    Detects git / modify / execute / plan by *ops shape* (not product-domain
    nouns, not Goal kinds). Default Q&A stage → ask.
    """
    t = (text or "").strip()
    if not t:
        return ""
    if _GIT_RE.search(t):
        return "client"
    if _EXECUTE_RE.search(t):
        return "execute"
    if _MODIFY_RE.search(t):
        return "modify"
    if _PLAN_RE.search(t):
        return "plan"
    return "ask"


def split_turn_stages(question: str) -> list[str]:
    q = (question or "").strip()
    if not q:
        return []
    # add-api / change-plan / compose-code stories stay whole (one ask with
    # service-body / query-pipeline orchestration, not independent questions).
    if (
        _is_add_api_story(q)
        or _is_change_plan_story(q)
        or _is_compose_code_story(q)
    ):
        return [q]
    numbered = [p.strip() for p in _NUM_SPLIT_RE.split(q) if p and p.strip()]
    if len(numbered) >= 2:
        return numbered[:8]
    parts = [p.strip(" ，,;；") for p in _ORDER_SPLIT_RE.split(q) if p and p.strip()]
    if len(parts) >= 2:
        return parts[:8]
    return [q]


def heuristic_turn_plan(question: str) -> TurnPlan:
    """Split ordered stages; classify each by ops/QA shape (fallback for mixed turns)."""
    q = (question or "").strip()
    if not q:
        return TurnPlan(nodes=[], source="heuristic")
    stages = split_turn_stages(q)
    if len(stages) < 2:
        stages = [q]
    nodes: list[TurnNode] = []
    prev_id = ""
    for st in stages[:8]:
        layer = normalize_layer(classify_layer(st) or "ask")
        nid = f"N{len(nodes) + 1}_{layer}"
        needs_confirm = layer in POST_LAYERS
        client_mode = ""
        if layer == "client":
            client_mode = "direct"
        elif layer in {"modify", "execute"}:
            client_mode = "code"
        nodes.append(
            TurnNode(
                id=nid,
                layer=layer,
                title=_LAYER_TITLE.get(layer, layer),
                question=st,
                depends_on=[prev_id] if prev_id else [],
                client_mode=client_mode,
                needs_confirm=needs_confirm,
            )
        )
        prev_id = nid
    if not nodes:
        nodes.append(
            TurnNode(id="N1_ask", layer="ask", title="查代码", question=q)
        )
    return TurnPlan(nodes=nodes, source="heuristic")


def is_mixed_intent_turn(question: str) -> bool:
    """True when one utterance has QA plus a later non-QA intent."""
    plan = heuristic_turn_plan(question)
    return bool(plan.has_qa and plan.has_post_qa)


# ─────────────────────────────────────────────────────────────────────────────
# LLM-driven turn splitting (primary judge for compound long utterances).
# ─────────────────────────────────────────────────────────────────────────────

# Compound markers the heuristic regex does NOT split on — evidence the
# utterance carries multiple intents without explicit 先/然后/再 connectives.
_COMPOUND_RE = re.compile(
    r"顺便|同时|还有|另外|以及|加上|帮我(?:再|接着|顺便|同时)?|"
    r"再看看|顺便看|顺带|然后顺带|接着帮我|再帮我|此外|除此之外|"
    r"再看|再看下|帮我查|帮我找|帮我列|帮我改|帮我提交|帮我推送"
)
# Threshold: below this length a single intent is overwhelmingly likely —
# do not spend an LLM call on it (cost guard). NOTE: this is a MINIMUM —
# punctuation-less compact sentences (「看看放款校验做了什么顺便分析改动点」)
# can be short yet carry multiple intents, so the trigger also fires when
# ≥2 intent-signal clusters are present (see _intent_signal_count).
_COMPOUND_MIN_LEN = 18

# Intent-signal regexes reused for the "multiple intents" trigger. A no-space /
# no-punctuation long utterance like 「看看放款流程帮我改下代码」 hits both
# the ask signal (看看) and an ops cluster (帮我改…), so we must not skip the
# LLM split just because the string is short.
_INTENT_SIGNAL_RES: tuple[re.Pattern[str], ...] = (
    _ASK_RE,
    _PLAN_RE,
    _MODIFY_RE,
    _EXECUTE_RE,
    _GIT_RE,
)
_OPS_PREFIX_RE = re.compile(
    r"帮我(?:查|找|列|改|提交|推送|实现|写)|改一下|改下|提交下|推送下|编译下|测一下"
)


def _intent_signal_count(text: str) -> int:
    """Number of distinct intent-signal clusters in the utterance.

    Each of the five intent regexes counts one signal; a 帮我+verb ops prefix
    counts an extra one (the ask regex misses 「帮我改…」 because it has no
    ask keyword). Used to decide whether a short compact sentence needs the
    LLM split: ≥2 signals → yes.
    """
    t = text or ""
    n = sum(1 for r in _INTENT_SIGNAL_RES if r.search(t))
    if _OPS_PREFIX_RE.search(t):
        n += 1
    return n

_TURN_SPLIT_PROMPT = """你是回合意图拆分器。把用户一句话按意图拆成有序的步骤序列。

用户原话：{question}

严格要求：
1. 只输出 JSON，不要任何解释文字：
{"stages":[{"layer":"ask|plan|modify|execute|client","text":"步骤原话（保留原文，不改写、不丢对象名/字段名）"}, ...]}
2. layer 只能取：ask（查代码/理解）、plan（改动清单/影响面/方案）、modify（改代码）、execute（编译/测试）、client（git 提交/推送/本机操作）。
3. 按原话顺序排列，最多 8 步；如果整句只有一个意图，只输出 1 步。
4. 每个步骤的 text 必须是原话的连续片段，保持原来的对象名、类名、字段、接口名；
   不能漏掉任何请求（例如「顺便把改动清单给我」里的改动清单请求）。
5. 「新增接口/加字段落库 + 先查A再查B」这类单任务编排是同一个 ask，
   绝对不要拆成多个独立步骤。
6. 不要臆造原话里没有的步骤。"""


async def llm_turn_plan(question: str) -> TurnPlan:
    """LLM-driven intent split with heuristic fallback.

    Fires only when (a) the heuristic cannot split (single stage) AND
    (b) the text looks compound (multiple intent signals without explicit
    connectives). Stories are kept whole BEFORE the LLM. Any failure falls
    back to ``heuristic_turn_plan`` so the pipeline never regresses.
    """
    q = (question or "").strip()
    if not q:
        return TurnPlan(nodes=[], source="heuristic")
    # Story guards first: one ask with orchestration → never split
    # (desktop-chat-fa14438a45aa / ed1e15401152 / b58bb8b77eea).
    if _is_add_api_story(q) or _is_change_plan_story(q) or _is_compose_code_story(q):
        return heuristic_turn_plan(q)
    base = heuristic_turn_plan(q)
    if len(base.nodes) >= 2:
        return base  # heuristic already found structure — no LLM call needed
    if not _COMPOUND_RE.search(q):
        return base  # single intent — no LLM call (cost guard)
    if len(q) < _COMPOUND_MIN_LEN and _intent_signal_count(q) < 2:
        return base  # short AND single signal — no LLM call (cost guard)

    parts: list[dict[str, Any]] | None = None
    try:
        from app.llm.factory import get_llm_provider
        from app.core.utils import truncate

        llm = get_llm_provider()
        raw = await llm.chat(
            [
                {"role": "system", "content": _TURN_SPLIT_PROMPT},
                {"role": "user", "content": truncate(q, 600)},
            ]
        )
        text = str(raw or "").strip()
        # strip code fences, then first {...} block
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
            text = re.sub(r"\s*```$", "", text).strip()
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            data = json.loads(m.group(0))
            cand = data.get("stages") or []
            if cand and isinstance(cand, list):
                parts = cand[:8]
    except Exception as exc:  # provider error / json error / shape error → fallback
        logger.debug("llm_turn_plan failed (%s); heuristic fallback", exc)

    if not parts:
        return TurnPlan(nodes=base.nodes, source="heuristic_fallback")
    valid = {"ask", "plan", "modify", "execute", "client"}
    nodes: list[TurnNode] = []
    prev_id = ""
    for i, st in enumerate(parts, 1):
        text = (str(st.get("text") or "")).strip() or q
        # Hybrid layer: the deterministic classifier owns known strong signals
        # (CN git/modify/execute/plan keywords — consistent with the rest of the
        # pipeline; EN forms that the regexes miss fall through to the LLM tag).
        heur = classify_layer(text)
        llm_layer = normalize_layer(str(st.get("layer") or ""))
        layer = (heur or llm_layer or "ask") if (heur or llm_layer) in valid else (heur or "ask")
        if layer not in valid:
            layer = "ask"
        nid = f"N{i}_{layer}"
        nodes.append(
            TurnNode(
                id=nid,
                layer=layer,
                title=_LAYER_TITLE.get(layer, layer),
                question=text,
                depends_on=[prev_id] if prev_id else [],
                client_mode="direct" if layer == "client" else ("code" if layer in {"modify", "execute"} else ""),
                needs_confirm=layer in POST_LAYERS,
            )
        )
        prev_id = nid
    if not nodes:
        return TurnPlan(nodes=base.nodes, source="heuristic_fallback")
    return TurnPlan(nodes=nodes, source="llm")


def is_mixed_ask_then_ops(question: str) -> bool:
    """Back-compat alias: ask/plan + modify/execute/git in one turn."""
    return is_mixed_intent_turn(question)


def compose_ordered_qa_reply(parts: list[tuple[TurnNode, str]]) -> str:
    """Join per-stage answers in the same order the user asked them."""
    cleaned = [(n, (ans or "").strip()) for n, ans in parts if n]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0][1]
    lines = ["## 按你的提问顺序作答", ""]
    for i, (node, ans) in enumerate(cleaned, 1):
        q = (node.question or "").strip()
        if len(q) > 80:
            q = q[:80] + "…"
        lines.append(f"### {i}. {node.title}：{q}")
        lines.append("")
        lines.append(ans or "（本步证据不足）")
        lines.append("")
    return "\n".join(lines).strip()
