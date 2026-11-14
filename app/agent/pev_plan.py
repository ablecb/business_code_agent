"""PEV plan layer extracted from pev_agent.py.

Goal extraction / task-planning / repo-inventory / scaffold tools that run
before execution: heuristic_goal, extract_goal_contract, build_task_plan,
verify_contract callers, repo-file inventory and the _exec_* plan tools.
pev_agent re-exports every name for import compatibility.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Callable

from app.core.utils import truncate
from app.agent.prompts import load_prompt
from app.agent.runtime.turn_plan import (
    _COMPOSE_STORY_STEPS_RE,
    _is_compose_code_story,
)

logger = logging.getLogger("app.agent.pev_plan")

ProgressCb = Callable[[dict[str, Any]], Any] | None

# --- Test tags (visible in progress + answer footer) ---
TAG = "[PEV]"


def _tag(*parts: str) -> str:
    body = "][".join(p for p in parts if p)
    return f"{TAG}[{body}]" if body else TAG


def _live_pev_plan(
    tasks: list[dict[str, Any]],
    *,
    phase: dict[str, str],
    notes: dict[str, str] | None = None,
    entry: str | None = None,
    spine_source: str | None = None,
) -> dict[str, Any]:
    """Plan payload for desktop SSE — high-level phases + per-task rows."""
    notes = notes or {}
    steps: list[dict[str, Any]] = [
        {
            "id": "goal",
            "title": "理解问题",
            "status": phase.get("goal") or "pending",
            "note": notes.get("goal"),
        },
        {
            "id": "plan",
            "title": "规划检索",
            "status": phase.get("plan") or "pending",
            "note": notes.get("plan"),
        },
        {
            "id": "exec",
            "title": "检索代码",
            "status": phase.get("exec") or "pending",
            "note": notes.get("exec"),
        },
    ]
    for t in tasks or []:
        steps.append(
            {
                "id": str(t.get("id") or ""),
                "title": str(t.get("title") or t.get("id") or "task"),
                "status": str(t.get("status") or "pending"),
                "note": t.get("note")
                or (f"tool={t.get('tool')}" if t.get("tool") else None),
            }
        )
    steps.append(
        {
            "id": "verify",
            "title": "核对回答",
            "status": phase.get("verify") or "pending",
            "note": notes.get("verify"),
        }
    )
    for s in steps:
        if not s.get("note"):
            s.pop("note", None)
    plan: dict[str, Any] = {
        "kind": "pev",
        "title": "分析中",
        "steps": steps,
    }
    if entry:
        plan["entry"] = entry
    if spine_source:
        plan["spine_source"] = spine_source
    return plan


GOAL_TYPES = frozenset(
    {
        "explain_flow",
        "change_plan",
        "add_api",
        "compound",
        "show_code",
        "compose_code",
        "inventory",
        "diagnose",
        "field_value",
        "api_io",
        "general",
    }
)


async def _emit(on_progress: ProgressCb, event: dict[str, Any]) -> None:
    if not on_progress:
        return
    try:
        maybe = on_progress(event)
        if hasattr(maybe, "__await__"):
            await maybe  # type: ignore[misc]
    except Exception:
        logger.debug("pev progress emit failed", exc_info=True)


def _parse_json_obj(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def extract_field_label_zh(question: str) -> str | None:
    """businessLabel（业务标签）→ 业务标签."""
    m = re.search(
        r"[A-Za-z][A-Za-z0-9_]{1,64}\s*[（(]\s*([\u4e00-\u9fff][^）)]{0,32})\s*[）)]",
        question or "",
    )
    if m:
        return m.group(1).strip() or None
    return None


def extract_named_tables(question: str) -> list[str]:
    from app.agent.change_impact.discover import extract_named_tables as _ext

    return _ext(question)


def wants_repo_file_inventory(question: str) -> bool:
    """True for「项目里几个 pom / 有哪些 yml」— filesystem inventory, not API/flow."""
    q = question or ""
    if not re.search(r"几个|多少|有哪些|列出|清单|都有哪些", q):
        return False
    # API inventory owns 「几个接口」
    if re.search(r"接口|Controller|endpoint|API", q, re.I):
        return False
    # Architecture / tech-stack surveys are not bare file inventories.
    try:
        from app.agent.codebase_survey import wants_codebase_architecture

        if wants_codebase_architecture(q):
            return False
    except Exception:
        pass
    # Avoid \\b — Chinese adjacency still counts as \\w, so 「几个pom」has no boundary.
    return bool(
        re.search(
            r"(?i)(?<![A-Za-z0-9_])pom(?:\.xml)?(?![A-Za-z0-9_])|"
            r"\.xml|\.ya?ml|\.properties|"
            r"Dockerfile|mapper\.xml|配置文件|模块",
            q,
        )
    )


def extract_repo_file_globs(question: str) -> list[str]:
    """Map question tokens → repo-relative globs (generic extensions / build files)."""
    q = question or ""
    globs: list[str] = []
    if re.search(r"(?i)(?<![A-Za-z0-9_])pom(?:\.xml)?(?![A-Za-z0-9_])", q):
        globs.append("**/pom.xml")
    if re.search(r"(?i)Dockerfile", q):
        globs.append("**/Dockerfile")
    if re.search(r"(?i)\.ya?ml|yml配置|yaml", q):
        globs.extend(["**/*.yml", "**/*.yaml"])
    if re.search(r"(?i)\.properties", q):
        globs.append("**/*.properties")
    if re.search(r"(?i)mapper\.xml", q):
        globs.append("**/*Mapper.xml")
    # Bare「配置文件」without a sharper token
    if not globs and re.search(r"配置文件", q):
        globs.extend(
            [
                "**/application*.yml",
                "**/application*.yaml",
                "**/application*.properties",
            ]
        )
    # De-dupe
    out: list[str] = []
    for g in globs:
        if g not in out:
            out.append(g)
    return out or ["**/pom.xml"]


def wants_add_api_scaffold(question: str) -> bool:
    """True when user asks to implement a new HTTP API from KB/docs.

    Covers both full-API (Controller + VO + Service) and entity-only (VO/DTO)
    code generation from knowledge-base documents.  The entity-only pattern is
    needed because users may say 「给我对应的java 实体类代码」 without the
    explicit 新增/实现/生成/写/落地 action verbs the original regex required;
    those requests still need the add_api goal (with entity_only=True), not
    explain_flow (which would fail with empty spine).
    """
    q = question or ""
    return bool(
        re.search(
            r"(?i)"
            r"(?:新增|新建|实现).{0,16}(?:接口|API)|"
            r"(?:按|根据).{0,12}(?:知识库|接口文档).{0,48}"
            r"(?:新增|实现|生成|写|落地)|"
            r"(?:按|根据).{0,12}(?:知识库|接口文档).{0,60}"
            r"(?:实体类|实体|Entity|entity|VO|Vo|vo|Dto|DTO|JavaBean|javabean|类名)|"
            r"新接口",
            q,
        )
    )


def wants_compound_change(question: str) -> bool:
    """New API scaffold + existing-API field/persist change in one ask.

    Capability intersection only — no product-domain nouns; LLM decides goal.
    """
    from app.agent.question_intent import wants_add_field_change_plan

    return bool(wants_add_api_scaffold(question) and wants_add_field_change_plan(question))


def heuristic_goal(question: str) -> dict[str, Any]:
    """Deterministic fallback when Goal LLM is unavailable.

    Not a product-noun cascade — uses structural ask shapes (流程/接口/源码/改法/
    表字段) so the degraded path still routes to a sensible goal instead of a
    generic search that drifts to an unrelated entry (desktop-chat-0c3000119343:
    「结清证明申请接口 主要流程是什么」 was mis-routed to a QuotaRisk processor
    when the Goal LLM was down).
    """
    q = question or ""
    goal = "general"
    must = [
        {"id": "search", "desc": "检索相关符号"},
        {"id": "answer", "desc": "基于证据回答"},
    ]
    ask_scope = None
    entry_hint = ""
    view_kind = None
    inventory_kind = None
    lookup_key = None
    lookup_table = None
    lookup_steps: list[dict[str, Any]] = []
    file_globs: list[str] = []
    also: list[str] = []
    deliverable = "accurate_evidence_backed_answer"

    try:
        from app.agent.explore_intent import (
            _wants_api_inventory,
            _wants_deep_flow_facts,
            _wants_endpoint_lookup,
            _wants_full_flow,
        )
        from app.agent.explore_source import _extract_explicit_class_method

        explicit = _extract_explicit_class_method(q)
        # Single-class asks ("XxxService 实现了什么逻辑") are NOT full-flow —
        # keep general so symbol_probe / LLM handles them.
        single_class_ask = bool(
            re.search(
                r"(?i)[A-Z][A-Za-z0-9_]*(?:Service|Listener|Client|Controller|Job|Util)"
                r"\s*(?:这个类|该类)?\s*"
                r"(?:实现|消费|处理|做了|是干|是干嘛|干什么|负责|提供)"
                r"[^。？?]{0,12}(?:逻辑|功能|什么)?[。？?]?$",
                q,
            )
        )
        flow_shape = (
            _wants_deep_flow_facts(q)
            or _wants_full_flow(q)
            or (
                "流程" in q
                and re.search(r"什么样|什么流程|怎么走|怎样|如何|描述|介绍|主要|做", q)
            )
        )
        # 1) change-plan / add-field asks (structure-first, no product nouns).
        from app.agent.question_intent import wants_add_field_change_plan

        # 1.5) 知识库生成代码/实体类：按知识库文档生成 Java 实体类/代码
        # 不走 explain_flow（不需要 Neo4j 调用链），直接走 add_api 加载知识库文档
        has_kb = "知识库" in q or "接口文档" in q or "接口设计文档" in q
        has_gen_code = re.search(
            r"生成|给我|写[一|段|个|出]|创建|构建|实现",
            q,
        )
        has_entity = re.search(
            r"实体类|实体|Entity|entity|VO|Vo|vo|Dto|DTO|JavaBean|javabean|类名|代码",
            q,
        )
        # change-plan must be FIRST in the chain: 增加字段并落库 also matches
        # flow_shape (_wants_deep_flow_facts), and as a standalone if it was
        # overwritten to explain_flow by the elif below (heuristic degradation
        # of 「放款接口增加一个字段 businessLabel 并落库」).
        if wants_add_field_change_plan(q) or re.search(
            r"改哪些|修改哪些|都要修改|改动点|改动清单|落库.*改|加字段|新增字段", q
        ):
            goal = "change_plan"
            entry_hint = explicit or _extract_entry_phrase(q)
            must = [
                {"id": "entry", "desc": "定位业务入口"},
                {"id": "discover", "desc": "发现改动角色与链路"},
                {"id": "render", "desc": "输出改动清单与代码"},
            ]
            deliverable = "程序修改清单和 DDL"
        elif has_kb and has_gen_code and has_entity:
            goal = "add_api"
            must = [
                {"id": "kb_spec", "desc": "读取知识库接口文档"},
                {"id": "scaffold", "desc": "生成实体类代码"},
            ]
            deliverable = "按知识库文档生成 Java 实体类代码"
            entry_hint = explicit or _extract_entry_phrase(q)
        # 1.6) compose-code asks: 给我一段代码 + 多步查询编排
        # （先查A → 查到后再查B → 支用）。确定性 story 判定优先于 LLM：
        # 否则 Goal LLM 常把这类问法降级成 explain_flow（展开调用链），
        # 而调用链证据不足时整轮拒绝（desktop-chat-ec81c911b4d8 /
        # desktop-chat-1b5c244b4a5e / desktop-chat-b58bb8b77eea）。
        elif _is_compose_code_story(q):
            goal = "compose_code"
            view_kind = "db_pipeline" if _COMPOSE_STORY_STEPS_RE.search(q) else "db_lookup"
            must = [
                {"id": "evidence", "desc": "检索相关表/Mapper/Service 证据"},
                {"id": "synthesize", "desc": "按证据生成可放入 Service 的代码"},
            ]
            deliverable = "证据支持的代码片段"
            entry_hint = explicit or _extract_entry_phrase(q)
        # 2) show-code asks: 贴/看 + 代码/源码 (explicit Class#method or topic).
        elif re.search(r"源码|源代码", q) or (
            "代码" in q
            and re.search(r"贴|看|发|show|源码|代码是|怎么写的|长什么样", q)
        ):
            goal = "show_code"
            view_kind = "method"
            entry_hint = explicit or _extract_entry_phrase(q)
            must = [{"id": "locate", "desc": "定位要展示的源码"}]
            deliverable = "展示方法/文件源码"
        # 2.5) 接口清单 asks（有哪些接口 / 接口列表）→ deterministic api_inventory.
        # Goal LLM 不可用时 heuristic 也必须路由到 inventory（否则退化到 general，
        # 只答一个入口主链漏掉其余端点 —— desktop-chat-5b263372926e）。
        elif _wants_api_inventory(q) and not _is_entry_or_downstream_ask(q):
            goal = "inventory"
            inventory_kind = "api"
            must = [
                {"id": "list", "desc": "列出相关 HTTP 接口"},
            ]
            deliverable = "列出相关 HTTP 接口（Controller 端点清单）"
            entry_hint = explicit or _extract_entry_phrase(q)
        # 3) full-flow asks (接口+流程 shape).
        elif flow_shape and not _wants_endpoint_lookup(q) and not single_class_ask:
            goal = "explain_flow"
            ask_scope = "full_flow"
            entry_hint = explicit or _extract_entry_phrase(q)
            must = [
                {"id": "entry", "desc": "定位业务 HTTP 主入口"},
                {"id": "spine", "desc": "展开入口主链/调用"},
                {"id": "summarize", "desc": "按证据汇总流程"},
            ]
            deliverable = "按执行序说明完整业务流程"
        elif "接口" in q and re.search(r"请求参数|响应参数|入参|出参", q):
            goal = "api_io"
            entry_hint = explicit or _extract_entry_phrase(q)
            must = [
                {"id": "entry", "desc": "解析系统入口"},
                {"id": "io", "desc": "抽取请求/响应参数"},
            ]
            deliverable = "列出接口入参出参"
        elif re.search(r"涉及哪些表|用到哪些表|都涉及|哪些表$", q):
            goal = "explain_flow"
            ask_scope = "full_flow"
            entry_hint = explicit or _extract_entry_phrase(q)
            must = [
                {"id": "entry", "desc": "定位接口入口"},
                {"id": "tables", "desc": "列出涉及的表"},
            ]
            deliverable = "接口涉及的表清单"
    except Exception:
        logger.debug("heuristic_goal enrichment failed", exc_info=True)

    base = {
        "goal": goal,
        "also": also,
        "must": must,
        "field_name": None,
        "field_label_zh": extract_field_label_zh(q),
        "tables": extract_named_tables(q),
        "inventory_kind": inventory_kind,
        "view_kind": view_kind,
        "lookup_key": lookup_key,
        "lookup_table": lookup_table,
        "lookup_steps": lookup_steps,
        "file_globs": file_globs,
        "deliverable": deliverable,
        "source": "heuristic",
    }
    if ask_scope:
        base["ask_scope"] = ask_scope
    if entry_hint:
        base["entry_hint"] = entry_hint
        base["entry_anchor"] = entry_hint
    return base


def wants_field_value_ask_safe(question: str) -> bool:
    try:
        from app.agent.field_evidence import wants_field_value_ask

        return bool(wants_field_value_ask(question))
    except Exception:
        return False


_GOAL_PROMPT = load_prompt("goal_prompt")
_BAD_MUST_IDS = frozenset({"短id", "id", "xxx", "todo", "must", "clause"})


# —— business_steps 确定性兜底（LLM 未填或填坏时从原话按步骤提取）——
# 用户常写「先参数校验，然后查询放款表，然后调用acc-service的接口，打印，返回」，
# 也可能用顿号/逗号直接列动作（「先校验、查放款表、调acc-service、打印、返回」）。
# 这里按「先/然后/再/最后」连接词 + 动作短语边界双路切分，每段识别 op 类别。
_BSTEP_SPLIT_RE = re.compile(r"(?:首先|先|然后|再|最后|接下来|其次|接着|并且|同时)\s*")
# 动作边界：识别「动作短语」的开头动词，顿号/逗号/「，然后」「再」之后都是新步骤起点。
_BSTEP_ACTION_EDGE_RE = re.compile(
    r"(?:，|、|,|;|；|然后|再|接着|之后|随后)\s*"
    r"(?=(?:进行|执行|做)?(?:参数)?(?:校验|验证|必填|非空|断言|检查参数|参数检查|assert|"
    r"查询|查一下|查\w*表|读取|select|查库|获取|调用|调|请求|调起|feign|rpc|远程|client|"
    r"打印|日志|log|输出|记录|返回|return|组装|拼装))"
)
_BSTEP_OP_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"校验|必填|非空|验证|断言|检查参数|参数检查|assert"), "validate"),
    (re.compile(r"查询|查一下|读取|select|查库|获取(?:数据|记录)|查.*表"), "query"),
    (re.compile(r"调用|调\s*\w+|请求|调起|feign|rpc|远程|client"), "call"),
    (re.compile(r"打印|日志|log|输出.*结果|记录.*结果"), "log"),
    (re.compile(r"返回|return|组装.*返回|拼装.*返回"), "return"),
]


def _classify_bstep(segment: str) -> str:
    for pat, kind in _BSTEP_OP_MAP:
        if pat.search(segment or ""):
            return kind
    return "other"


def _extract_business_steps_fallback(question: str) -> list[dict[str, Any]]:
    """Split question on step connectors / action edges; classify each → step."""
    q = (question or "").strip()
    if not q:
        return []
    # 先按连接词切分（先/然后/再/最后…），再在每个分句内按动作边界切分
    # （顿号/逗号分隔的动作序列：校验、查表、调服务、打印、返回）。
    parts: list[str] = []
    for chunk in _BSTEP_SPLIT_RE.split(q):
        chunk = chunk.strip()
        if not chunk:
            continue
        sub = [s.strip() for s in _BSTEP_ACTION_EDGE_RE.split(chunk) if s.strip()]
        parts.extend(sub or [chunk])
    if not parts:
        parts = [q]
    # 引导句（根据…文档生成接口/新增接口/生成代码）不是 service 步骤，跳过。
    _LEADING_RE = re.compile(
        r"^(?:根据|按|按照|基于).{0,24}(?:文档|知识库|接口文档).{0,24}"
        r"(?:生成|新增|实现|写).{0,12}(?:接口|代码|API)|"
        r"^生成.{0,12}(?:接口|代码|API)"
    )
    steps: list[dict[str, Any]] = []
    for p in parts:
        if len(p) < 2:
            continue
        # 引导句直接跳过
        if _LEADING_RE.match(p):
            continue
        # 「在service中 / service 里」这类位置短语无动作，跳过
        if re.fullmatch(r"在?\w*(?:service|Service|方法|实现|里|中|内|里面|里 面)[的里中]*", p.strip("，。；、：: ")):
            continue
        op = _classify_bstep(p)
        if op == "other":
            # 无动作的引导残留也跳过
            if re.search(r"生成.*(?:接口|代码)|根据.*文档|新增.*接口", p):
                continue
        target = p.strip("，。；、：: \n")[:60]
        steps.append({"op": op, "target": target or op})
    # 最后一段通常是「返回」——若分类为 other 但含返回语义，补 return
    if steps and steps[-1]["op"] == "other" and re.search(r"返回|给用户|输出", steps[-1]["target"]):
        steps[-1]["op"] = "return"
    return steps[:12]

_SPINE_NOISE_RE = re.compile(
    r"(?i)(?:^|\.)(DateUtil|LogUtils|Objects|StringUtils|JSON|JSONUtil|"
    r"BeanUtils|BeanUtil|ObjectUtil|CollectionUtil)#|"
    r"#(format\w*|toString|equals|hashCode|mqSend|mqSended)\b"
)

# Code-generation goals that a later segment of the same utterance continues.
_CONTINUATION_CODE_GOALS = frozenset(
    {"compose_code", "add_api", "change_plan", "show_code"}
)
# Segment words that mean the user switched to a flow-narration ask.
_FLOW_SWITCH_RE = re.compile(
    r"流程|主链|链路|全链路|从入口|串一下|串起来|怎么走|执行序|入口(?:是|在)?什么"
)


def _segment_changed_topic(question: str) -> bool:
    """True when a continuation segment names a new entry/anchor (topic switch)."""
    q = (question or "").strip()
    if re.search(r"(?:Controller|Service|Client|Mapper|Listener)\b", q):
        return True
    return bool(
        re.search(
            r"[A-Za-z][A-Za-z0-9_]*(?:接口|入口)|"
            r"另(?:外|一个)|换一个|不要(?:上面|刚才)|改成",
            q,
        )
    )


def _inherit_continuation_goal(
    base: dict[str, Any], question: str, prior_goal: str
) -> dict[str, Any]:
    """Keep a code-gen goal for a continuation segment of the same ask.

    Explicit flow vocabulary (流程/主链/链路…) or a fresh entry anchor means the
    user really switched asks — then let the normal goal judgement run.
    """
    q = (question or "").strip()
    if _FLOW_SWITCH_RE.search(q) and not re.search(
        r"代码|生成|写(?:一段|个|给我)|给我一段", q
    ):
        return base
    out = dict(base)
    out["goal"] = prior_goal
    # Preserve lookup slots for compose_code continuations (query steps may be
    # split across segments; base heuristics keep whatever was extractable).
    if prior_goal == "compose_code" and not out.get("lookup_table"):
        out["view_kind"] = out.get("view_kind") or "db_lookup"
    return out


async def extract_goal_contract(
    question: str,
    *,
    history_text: str = "",
    prior_goal: str | None = None,
) -> dict[str, Any]:
    base = heuristic_goal(question)
    # D: multi-segment continuation — a later segment of the same "give me code"
    # ask inherits the code-generation goal instead of being re-judged as a flow
    # question (desktop-chat-1b5c244b4a5e: segment 2 misjudged explain_flow).
    if prior_goal in _CONTINUATION_CODE_GOALS:
        base = _inherit_continuation_goal(base, question, prior_goal)
    data: dict[str, Any] = {}
    # Always call Goal LLM for ask_scope / dig-in vs full_flow. Heuristic is seed + fallback.
    try:
        from app.agent.question_intent import assumed_new_field_name
        from app.llm.factory import get_llm_provider

        fname = assumed_new_field_name(question)
        if fname and fname != "newField":
            base["field_name"] = fname
        llm = get_llm_provider()
        prior_note = ""
        if prior_goal in _CONTINUATION_CODE_GOALS:
            prior_note = (
                f"\n（前序意图={prior_goal}：这是同一句「生成代码」需求的后续步骤；"
                f"除非本段明显换了话题/新入口，否则 goal 应保持 {prior_goal}，"
                f"不要把延续步骤改判成 explain_flow。）"
            )
        content = await asyncio.wait_for(
            llm.chat(
                [
                    {"role": "system", "content": _GOAL_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"历史:\n{truncate(history_text or '(空)', 1200)}"
                            f"{prior_note}\n\n"
                            f"问题:\n{question}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=500,
            ),
            timeout=15.0,
        )
        data = _parse_json_obj(content)
        g = str(data.get("goal") or "").strip()
        # Prefer LLM goal when valid; heuristic seed is fallback only.
        if g in GOAL_TYPES:
            # Continuation safety: code-gen segments keep their goal unless the
            # LLM explicitly moves to another code-gen/flow goal with a new anchor.
            if prior_goal in _CONTINUATION_CODE_GOALS and base.get("goal") == prior_goal:
                if g not in _CONTINUATION_CODE_GOALS and _segment_changed_topic(question):
                    base["goal"] = g
                else:
                    base["goal"] = prior_goal
            else:
                # Guard: heuristic detected KB code-gen (add_api) or compose-code
                # (多步查询生成代码) pattern — don't let LLM override to
                # explain_flow which would fail on empty spine.
                if base.get("goal") == "add_api" and g == "explain_flow":
                    logger.debug(
                        "heuristic KB code-gen goal=%s kept over LLM goal=%s",
                        base["goal"], g,
                    )
                elif base.get("goal") == "compose_code" and g == "explain_flow":
                    # 「给我一段代码 先查A再查B…」 must stay compose_code:
                    # explain_flow expands a call spine that has no evidence for
                    # a synthesized snippet, so the whole turn gets refused
                    # (desktop-chat-ec81c911b4d8 / 1b5c244b4a5e / b58bb8b77eea).
                    logger.debug(
                        "heuristic compose-code goal=%s kept over LLM goal=%s",
                        base["goal"], g,
                    )
                else:
                    base["goal"] = g
        if isinstance(data.get("must"), list) and data["must"]:
            must = []
            for m in data["must"][:12]:
                if not isinstance(m, dict) or not m.get("id"):
                    continue
                mid = str(m["id"]).strip()[:40]
                if not mid or mid.lower() in _BAD_MUST_IDS or mid in _BAD_MUST_IDS:
                    continue
                must.append(
                    {
                        "id": mid,
                        "desc": str(m.get("desc") or "")[:200],
                    }
                )
            if must:
                base["must"] = must
        if data.get("field_name"):
            base["field_name"] = str(data["field_name"]).strip()
        if data.get("field_label_zh"):
            base["field_label_zh"] = str(data["field_label_zh"]).strip()
        elif base.get("field_label_zh") is None:
            base["field_label_zh"] = extract_field_label_zh(question)
        if isinstance(data.get("tables"), list) and data["tables"]:
            merged = [str(t) for t in data["tables"] if t][:8]
            physical = [
                t.lower()
                for t in merged
                if re.match(r"(?i)^demo_[a-z0-9_]+$", str(t).strip())
            ]
            base["tables"] = physical or list(base.get("tables") or [])
        ik = str(data.get("inventory_kind") or "").strip()
        if ik in {
            "api",
            "repo_files",
            "architecture",
            "symbol_probe",
            "topic_probe",
        } and base.get("goal") == "inventory":
            base["inventory_kind"] = ik
            if ik == "repo_files" and not base.get("file_globs"):
                base["file_globs"] = extract_repo_file_globs(question)
        if base.get("goal") == "change_plan":
            base["inventory_kind"] = None
            base["file_globs"] = []
        # view_kind / lookup_* — Goal LLM only (no regex post-guards).
        vk = str(data.get("view_kind") or "").strip().lower()
        if vk in {"method", "file", "config", "db_lookup", "db_pipeline"}:
            if vk.startswith("db_"):
                # Generate-from-evidence path (not raw paste).
                if str(base.get("goal") or "") not in {"compose_code", "show_code"}:
                    base["goal"] = "compose_code"
                elif str(base.get("goal") or "") == "show_code":
                    # LLM said show_code + db_* → prefer compose for synthesize asks.
                    base["goal"] = "compose_code"
                base["view_kind"] = vk
            elif str(base.get("goal") or "") in {"show_code", "compose_code"}:
                base["view_kind"] = vk
        if data.get("lookup_key"):
            base["lookup_key"] = str(data["lookup_key"]).strip()[:120]
        if data.get("lookup_table"):
            base["lookup_table"] = str(data["lookup_table"]).strip()[:120]
        if isinstance(data.get("lookup_steps"), list) and data["lookup_steps"]:
            steps_out: list[dict[str, Any]] = []
            for raw in data["lookup_steps"][:12]:
                if not isinstance(raw, dict):
                    continue
                op = str(raw.get("op") or "").strip().lower()[:40]
                if not op:
                    continue
                step: dict[str, Any] = {"op": op}
                for k in ("table", "key", "key_from"):
                    val = raw.get(k)
                    if val is None:
                        continue
                    s = str(val).strip()
                    if not s or s.lower() == "null":
                        continue
                    step[k] = s[:120]
                for k in ("fields", "agg"):
                    if isinstance(raw.get(k), list) and raw[k]:
                        step[k] = [
                            str(x).strip()[:80]
                            for x in raw[k]
                            if x and str(x).strip().lower() != "null"
                        ][:8]
                steps_out.append(step)
            if steps_out:
                base["lookup_steps"] = steps_out
                if not base.get("view_kind") and str(base.get("goal") or "") in {
                    "show_code",
                    "compose_code",
                    "general",
                }:
                    base["view_kind"] = (
                        "db_pipeline" if len(steps_out) > 1 else "db_lookup"
                    )
                    if str(base.get("goal") or "") != "show_code":
                        base["goal"] = "compose_code"
        if (
            str(base.get("view_kind") or "") in {"db_lookup", "db_pipeline"}
            and str(base.get("goal") or "") == "show_code"
        ):
            base["goal"] = "compose_code"
        if (
            str(base.get("view_kind") or "") == "db_lookup"
            and not base.get("lookup_table")
            and isinstance(base.get("tables"), list)
            and base["tables"]
        ):
            base["lookup_table"] = str(base["tables"][0])[:120]
        # No regex post-guards — Goal LLM owns goal / inventory_kind / view_kind.
        if isinstance(data.get("also"), list) and data["also"]:
            also = [
                str(x).strip()
                for x in data["also"]
                if str(x).strip() in {"add_api", "change_plan", "explain_flow"}
            ]
            if also:
                base["also"] = also[:4]
        if data.get("deliverable"):
            base["deliverable"] = str(data["deliverable"])[:200]
        # ask_scope from LLM only when present; default explain_flow → full_flow.
        # Dig-in ask_scope is not forced from structural heuristics (session gate).
        ask_scope = str(data.get("ask_scope") or "").strip().lower()
        if ask_scope in {"full_flow", "entry_scoped_dig_in", "dig_in"}:
            if ask_scope == "dig_in":
                ask_scope = "entry_scoped_dig_in"
            base["ask_scope"] = ask_scope
        elif str(base.get("goal") or "") == "explain_flow":
            base["ask_scope"] = "full_flow"
        if data.get("focus_topic"):
            base["focus_topic"] = str(data["focus_topic"]).strip()[:120]
        # entry_anchor + method_topic: LLM split for method-first dig-in.
        if data.get("entry_anchor"):
            ea = str(data["entry_anchor"]).strip()[:120]
            if ea:
                base["entry_anchor"] = ea
                if not base.get("entry_hint"):
                    base["entry_hint"] = ea
        if data.get("method_topic"):
            mt = str(data["method_topic"]).strip()[:120]
            if mt:
                base["method_topic"] = mt
                if not base.get("focus_topic"):
                    base["focus_topic"] = mt
        elif base.get("focus_topic") and not base.get("method_topic"):
            base["method_topic"] = str(base["focus_topic"])
        if base.get("ask_scope") == "entry_scoped_dig_in":
            base["dig_in_focus"] = True
            if str(base.get("goal") or "") in {"explain_flow", "general", ""}:
                base["goal"] = "explain_flow"
                base["must"] = [
                    {"id": "entry", "desc": "定位业务 HTTP 主入口"},
                    {"id": "spine", "desc": "展开入口主链/调用"},
                    {
                        "id": "summarize",
                        "desc": "按子话题先锁方法再校验入口链并只答该方法描述",
                    },
                ]
        base["source"] = "llm" if g in GOAL_TYPES else "llm+heuristic"
    except asyncio.TimeoutError:
        logger.warning("extract_goal_contract LLM timed out after 15s; heuristic only")
        base["source"] = "heuristic"
    except Exception:
        logger.debug("extract_goal_contract LLM failed; heuristic only", exc_info=True)
        base["source"] = "heuristic"
    # —— 知识库接口脚手架兜底（确定性，防 LLM 把 add_api 误判成 compose_code）——
    # 「根据知识库/接口文档生成/实现 Java 接口代码」需要的是新增接口骨架
    # （Controller + 请求/响应 VO + Service + Impl），compose_code 只产出单段方法
    # 片段，给不出实体类/DTO，无法满足「生成接口代码」。LLM 常因「生成代码」
    # 措辞误判为 compose_code（或把新增接口误判成 modify/改现有代码），
    # 这里用 wants_add_api_scaffold 确定性兜底。
    # 实体类场景（「按知识库文档生成实体类」）不含「接口/API/新增」等词，
    # 用 _wants_entity_only 补充兜底（desktop-chat-75fe902deb0b）。
    if (
        wants_add_api_scaffold(question)
        and (
            re.search(r"(?i)接口|API|Controller|endpoint|新增|新建", question)
            or _wants_entity_only(question)
        )
        and str(base.get("goal") or "") in {
            "compose_code",
            "show_code",
            "general",
            "modify",
        }
    ):
        base["goal"] = "add_api"
        base["view_kind"] = None
        base["lookup_key"] = None
        base["lookup_table"] = None
        base["lookup_steps"] = []
        base["must"] = [
            {"id": "kb_spec", "desc": "读取知识库接口文档"},
            {"id": "style", "desc": "对齐项目 Controller 风格"},
            {"id": "scaffold", "desc": "生成新增接口清单与代码骨架"},
        ]
    # —— add_api 业务步骤（service 方法体内执行顺序）——
    # 「先校验 → 查表 → 调 acc-service → 打印 → 返回」这类编排由 Goal LLM 填
    # business_steps；这里做结构清洗 + 确定性归一（不依赖产品词表）。
    _biz_steps = data.get("business_steps")
    if isinstance(_biz_steps, list) and _biz_steps and str(base.get("goal") or "") in {
        "add_api",
        "compound",
    }:
        cleaned: list[dict[str, Any]] = []
        for raw in _biz_steps[:12]:
            if not isinstance(raw, dict):
                continue
            op = str(raw.get("op") or "").strip().lower()[:24]
            if op not in {"validate", "query", "call", "log", "return", "other"}:
                op = "other"
            target = str(raw.get("target") or raw.get("desc") or "").strip()[:120]
            if not op and not target:
                continue
            cleaned.append({"op": op, "target": target or op})
        if cleaned:
            base["business_steps"] = cleaned
    # 确定性兜底：LLM 未填 business_steps，或填的明显残缺/把多个动作塞进一步时，
    # 从用户原话按「先/然后/再」+ 动作边界（顿号/逗号分隔）分句提取，覆盖 LLM 结果。
    # 判定「填坏了」：任一 target 含 2+ 动作词（校验/查询/调用/打印/返回…）。
    _llm_steps = base.get("business_steps") or []
    _llm_bad = any(
        sum(1 for _pat, _k in _BSTEP_OP_MAP if _pat.search(str(s.get("target") or "")))
        >= 2
        for s in _llm_steps
    ) or (
        len(_llm_steps) == 1
        and re.search(r"(?:校验|查询|调用|打印|返回).{0,20}(?:查询|调用|打印|返回)", str(_llm_steps[0].get("target") or ""))
    )
    if (
        str(base.get("goal") or "") in {"add_api", "compound"}
        and (
            not _llm_steps
            or _llm_bad
            or re.search(
                r"(?:校验|查询|调用|打印|返回).{0,12}(?:、|，|然后|再)(?:校验|查询|调用|打印|返回)",
                question,
            )
        )
    ):
        fallback = _extract_business_steps_fallback(question)
        if fallback and len(fallback) >= len(_llm_steps):
            base["business_steps"] = fallback
    # —— 入口/下游问法护栏（确定性，防 LLM 误判成 inventory）——
    # 「XX入口在哪个 Controller / 入口地址 / 调用了哪些下游接口 / 调用链」这类问题
    # 必须解析入口并给出主链（含下游），inventory（接口清单）答不了"入口是哪个"。
    # entry_route 已能解析这些问法（实测 放款接口入口→WithdrawalsController#withdrawalsAndQuota）。
    if _is_entry_or_downstream_ask(question):
        base["goal"] = "explain_flow"
        base["ask_scope"] = "full_flow"
        base["inventory_kind"] = None
        base["view_kind"] = None
        base["dig_in_focus"] = False
        if not base.get("entry_anchor"):
            # 从问法里抠入口短语：取「XX接口/XX入口/XX流程」的核心话题
            ea = _extract_entry_phrase(question)
            if ea:
                base["entry_anchor"] = ea
                base["entry_hint"] = ea
        base["source"] = f"{base.get('source') or 'llm'}+entry_guard"
    # Default explain_flow without LLM ask_scope → full_flow (session dig-in only).
    if (
        str(base.get("goal") or "") == "explain_flow"
        and not base.get("ask_scope")
    ):
        base["ask_scope"] = "full_flow"
    return base


_ENTRY_OR_DOWNSTREAM_RE = re.compile(
    r"(?:"
    r"入口\s*(?:在|是|在哪|是哪个|地址|在哪|是什么)"
    r"|哪个\s*(?:Controller|controller|类|类名)"
    r"|入口地址|入口在哪|入口是什么"
    r"|调用了哪些|调用链|下游接口|下游有哪些|调了哪些|都调用了|调用过哪些"
    r"|下游调用了"
    r")"
)


def _is_entry_or_downstream_ask(question: str) -> bool:
    q = question or ""
    if _ENTRY_OR_DOWNSTREAM_RE.search(q):
        return True
    try:
        from app.agent.explore_intent import _wants_flow_downstream_apis

        return bool(_wants_flow_downstream_apis(q))
    except Exception:
        return False


def _extract_entry_phrase(question: str) -> str:
    """从「XX入口在哪个 Controller」里抠出 XX（放款接口 → 放款接口）。"""
    q = question or ""
    m = re.search(r"([\u4e00-\u9fffA-Za-z0-9_#\.]+?(?:接口|入口|流程|Controller))\s*(?:入口|在哪个|地址|调用|在哪)", q)
    if m:
        return m.group(1).strip()
    m = re.search(r"([\u4e00-\u9fffA-Za-z0-9_#\.]{2,16}?(?:接口|流程))\s*(?:的)?入口", q)
    if m:
        return m.group(1).strip()
    # 兜底：去掉疑问尾缀后的最长中文话题
    cleaned = re.sub(r"(入口在哪个 Controller|入口地址|入口在哪|调用了哪些下游接口|怎么关联的|是什么)$", "", q)
    toks = re.findall(r"[\u4e00-\u9fff]{2,12}", cleaned)
    if toks:
        return max(toks, key=len)
    return ""


def build_task_plan(goal: dict[str, Any]) -> list[dict[str, Any]]:
    """Map goal → executable tasks (generic tools, no product Class lists)."""
    g = str(goal.get("goal") or "general")
    tasks: list[dict[str, Any]] = []

    def _t(tid: str, title: str, tool: str) -> None:
        tasks.append({"id": tid, "title": title, "tool": tool, "status": "pending"})

    if g == "explain_flow":
        _t("T1_entry", "解析系统入口", "resolve_entry")
        _t("T2_spine", "展开 FOLLOWS/执行序主链", "expand_calls")
        _t("T3_enrich", "MySQL 摘要补齐", "mysql_enrich")
        _t("T4_answer", "按执行序汇总流程", "summarize_flow")
    elif g == "compound":
        _t("T1_kb", "读取知识库接口文档", "load_kb_spec")
        _t("T2_style", "对齐项目 Controller 风格", "find_api_style")
        _t("T3_scaffold", "生成新增接口清单与代码", "render_new_api")
        _t("T4_entry", "解析已有接口系统入口", "resolve_entry")
        _t("T5_discover", "发现已有接口改动触点", "discover_change")
        _t("T6_outbound", "检索外送/关卡触点", "find_outbound")
        _t("T7_change", "渲染已有接口改动清单与 DDL", "render_change")
    elif g == "change_plan":
        _t("T1_entry", "解析系统入口", "resolve_entry")
        _t("T2_discover", "发现改动触点", "discover_change")
        _t("T3_outbound", "检索外送/关卡触点", "find_outbound")
        _t("T4_answer", "渲染改动清单与 DDL", "render_change")
    elif g == "add_api":
        _t("T1_kb", "读取知识库接口文档", "load_kb_spec")
        _t("T2_style", "对齐项目 Controller 风格", "find_api_style")
        _t("T3_scaffold", "生成新增接口清单与代码", "render_new_api")
    elif g == "show_code":
        _t("T1_locate", "定位要看的代码", "source_view")
    elif g == "compose_code":
        _t("T1_compose", "检索证据并生成代码", "compose_code")
    elif g == "inventory":
        ik = str(goal.get("inventory_kind") or "")
        if ik == "repo_files" or goal.get("file_globs"):
            _t("T1_files", "扫描仓库文件并计数", "repo_files")
        elif ik == "architecture":
            _t("T1_arch", "读项目笔记本并按问题作答", "codebase_architecture")
        elif ik == "symbol_probe":
            _t("T1_probe", "核查类型是否存在于源码/索引", "symbol_probe")
        elif ik == "topic_probe":
            _t("T1_topic", "按话题检索源码/索引确认是否存在", "topic_existence")
        else:
            _t("T1_list", "列出相关入口", "api_inventory")
    elif g == "diagnose":
        _t("T1_signal", "提取问题/日志信号", "parse_signal")
        _t("T2_map", "映射到代码符号", "map_and_read")
        _t("T3_answer", "诊断结论", "diagnose_answer")
    elif g == "field_value":
        _t("T1_entry", "解析系统入口", "resolve_entry")
        _t("T2_spine", "展开 FOLLOWS/执行序主链", "expand_calls")
        _t("T3_field", "主链字段写点证据", "collect_field_evidence")
        _t("T4_answer", "按字段证据作答", "render_field_answer")
    elif g == "api_io":
        _t("T1_entry", "解析系统入口", "resolve_entry")
        _t("T2_spine", "展开入口辅助方法", "expand_calls")
        _t("T3_io", "抽取请求/响应参数证据", "collect_api_io")
        _t("T4_answer", "列出接口入参出参", "render_api_io")
    else:
        _t("T1_entry", "尝试解析入口", "resolve_entry")
        _t("T2_search", "检索相关符号", "search_hits")
        _t("T3_answer", "综合回答", "general_answer")
    from app.agent.runtime.registry import filter_task_tools

    return filter_task_tools(tasks)


def _footer_tags(
    goal: dict[str, Any],
    tasks: list[dict[str, Any]],
    verify: dict[str, Any],
    evidence: dict[str, Any],
) -> str:
    done = sum(1 for t in tasks if t.get("status") == "done")
    fail = sum(1 for t in tasks if t.get("status") == "failed")
    lines = [
        "",
        "---",
        f"{_tag('TEST')} goal={goal.get('goal')} source={goal.get('source')}",
        f"{_tag('TEST')} tasks_done={done}/{len(tasks)} failed={fail}",
        f"{_tag('TEST')} entry={evidence.get('entry') or '(none)'}",
        f"{_tag('TEST')} spine_source={evidence.get('spine_source') or '(none)'}",
        f"{_tag('TEST')} verify_ok={verify.get('ok')} gaps={len(verify.get('gaps') or [])}",
    ]
    if goal.get("field_label_zh"):
        lines.append(f"{_tag('TEST')} field_zh={goal.get('field_label_zh')}")
    if goal.get("tables"):
        lines.append(f"{_tag('TEST')} tables={','.join(goal.get('tables') or [])}")
    if evidence.get("intersect_cms"):
        lines.append(
            f"{_tag('TEST')} outbound_hits="
            + ",".join(list(evidence.get("intersect_cms") or [])[:5])
        )
    if verify.get("gaps"):
        lines.append(
            f"{_tag('TEST')} gap_ids="
            + ",".join(str(g.get("id") or g) for g in (verify.get("gaps") or [])[:8])
        )
    return "\n".join(lines)


def _project_root_for(project_id: str):
    from pathlib import Path

    from app.core.config import get_settings

    try:
        from app.services.projects import get_project

        meta = get_project(project_id) if project_id else None
        if meta and meta.get("root_path"):
            root = Path(meta["root_path"])
            if root.exists():
                return root.resolve()
    except Exception:
        logger.debug("get_project root failed", exc_info=True)
    # Fallback: settings.project_root (often java_project/<id>)
    settings = get_settings()
    root = settings.project_root_path
    if project_id and root.name != project_id:
        alt = root.parent / project_id
        if alt.is_dir():
            return alt.resolve()
    return root


def list_repo_files(
    project_id: str,
    globs: list[str],
    *,
    limit: int = 80,
) -> dict[str, Any]:
    """Glob files under the project root; return relative paths + count."""
    root = _project_root_for(project_id)
    paths: list[str] = []
    seen: set[str] = set()
    for pattern in globs or []:
        try:
            for p in root.glob(pattern):
                if not p.is_file():
                    continue
                try:
                    rel = p.resolve().relative_to(root).as_posix()
                except ValueError:
                    rel = p.as_posix()
                if rel in seen:
                    continue
                seen.add(rel)
                paths.append(rel)
        except Exception:
            logger.debug("glob failed: %s", pattern, exc_info=True)
    paths.sort()
    truncated = len(paths) > limit
    return {
        "ok": True,
        "root": str(root),
        "globs": list(globs or []),
        "count": len(paths),
        "paths": paths[:limit],
        "truncated": truncated,
    }


def format_repo_files_answer(scan: dict[str, Any], question: str) -> str:
    count = int(scan.get("count") or 0)
    paths = list(scan.get("paths") or [])
    globs = list(scan.get("globs") or [])
    root = scan.get("root") or ""
    topic = "、".join(f"`{g}`" for g in globs) or "目标文件"
    lines = [
        f"## 仓库文件清单（{topic}）",
        "",
        f"共 **{count}** 个" + ("（已截断展示）" if scan.get("truncated") else "") + "。",
        f"扫描根目录：`{root}`",
        "",
    ]
    if not paths:
        lines.append("未匹配到文件；可换更具体的文件名/后缀再问。")
        return "\n".join(lines)
    lines.append("| # | 相对路径 |")
    lines.append("|---|---|")
    for i, p in enumerate(paths, 1):
        lines.append(f"| {i} | `{p}` |")
    if question and "几个" in question:
        lines.extend(["", f"结论：共 {count} 个。"])
    return "\n".join(lines)


async def _exec_repo_files(
    project_id: str, question: str, goal: dict[str, Any], evidence: dict[str, Any]
) -> str:
    globs = list(goal.get("file_globs") or []) or extract_repo_file_globs(question)
    scan = list_repo_files(project_id, globs)
    evidence["repo_file_scan"] = scan
    evidence["repo_file_count"] = scan.get("count")
    return format_repo_files_answer(scan, question)


_ARCH_ANSWER_PROMPT = load_prompt("arch_answer_prompt")
async def _exec_codebase_architecture(
    project_id: str,
    question: str,
    evidence: dict[str, Any],
) -> str:
    from app.agent.codebase_survey import run_architecture_survey

    result = run_architecture_survey(project_id, question)
    evidence["architecture_ok"] = bool(result.get("ok"))
    evidence["architecture_meta"] = dict(result.get("evidence") or {})
    brief = str(result.get("brief") or result.get("answer") or "").strip()
    evidence["architecture_brief"] = brief[:12000]
    mode = str((result.get("evidence") or {}).get("mode") or "")
    evidence["notebook_mode"] = mode
    if not brief:
        return str(result.get("answer") or "项目笔记本/结构扫描未得到可用证据。")
    # Framework-shape inventories (@XxlJob / Feign / Listener…) are already a
    # deterministic scan list — LLM rewrite truncates and remaps rows. Ship as-is.
    if mode == "focused_scan":
        return brief
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        raw = await llm.chat(
            [
                {"role": "system", "content": _ARCH_ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：\n{question}\n\n"
                        f"项目笔记本证据：\n{brief[:14000]}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=2200,
        )
        text = str(raw or "").strip()
        if text:
            return text
    except Exception:
        logger.debug("architecture LLM answer failed", exc_info=True)
    return brief


async def _exec_symbol_probe(
    project_id: str,
    question: str,
    evidence: dict[str, Any],
) -> str:
    from app.agent.codebase_survey import run_symbol_probe

    result = run_symbol_probe(project_id, question)
    evidence["symbol_probe_ok"] = bool(result.get("ok"))
    evidence["symbol_probe_meta"] = dict(result.get("evidence") or {})
    return str(result.get("answer") or "")


_TOPIC_ANSWER_PROMPT = load_prompt("topic_answer_prompt")
def _topic_answer_too_thin(text: str) -> bool:
    """True when LLM collapsed a rich probe into a bare yes/no."""
    t = (text or "").strip()
    if not t:
        return True
    # Strip trivial punctuation / whitespace for length check.
    compact = re.sub(r"[\s。.!！?？、,，;；:：\-—]+", "", t)
    if len(compact) <= 4 and compact in {
        "有",
        "没有",
        "无",
        "存在",
        "不存在",
        "是",
        "否",
        "有的",
        "没有的",
    }:
        return True
    if len(t) < 40 and ("`" not in t) and ("Controller" not in t) and ("Service" not in t):
        return True
    return False


async def _exec_topic_existence(
    project_id: str,
    question: str,
    evidence: dict[str, Any],
) -> str:
    from app.agent.codebase_survey import run_topic_existence_probe

    result = run_topic_existence_probe(project_id, question)
    evidence["topic_probe_ok"] = bool(result.get("ok"))
    evidence["topic_probe_exists"] = bool(result.get("exists"))
    evidence["topic_probe_phrase"] = str(result.get("phrase") or "")
    evidence["topic_probe_meta"] = {
        "source_hits": len(result.get("source_hits") or []),
        "index_hits": len(result.get("index_hits") or []),
    }
    brief = str(result.get("brief") or result.get("answer") or "").strip()
    evidence["topic_probe_brief"] = brief[:12000]
    if not brief:
        return "未检索到该话题相关证据。"
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        raw = await llm.chat(
            [
                {"role": "system", "content": _TOPIC_ANSWER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：\n{question}\n\n"
                        f"话题存在性证据：\n{brief[:14000]}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=1600,
        )
        text = str(raw or "").strip()
        if text and not _topic_answer_too_thin(text):
            return text
        if text:
            logger.info("topic existence LLM answer too thin; using brief")
    except Exception:
        logger.debug("topic existence LLM answer failed", exc_info=True)
    return brief


async def _exec_load_kb_spec(question: str, evidence: dict[str, Any]) -> None:
    from app.knowledge.add_api import load_kb_spec_for_question

    kb = load_kb_spec_for_question(question)
    evidence["kb_spec"] = kb
    evidence["kb_ok"] = bool(kb.get("full_text") or kb.get("hits"))
    evidence["kb_title"] = kb.get("title") or kb.get("filename") or ""


async def _exec_find_api_style(project_id: str, evidence: dict[str, Any]) -> None:
    from app.knowledge.add_api import find_style_exemplar

    style = find_style_exemplar(project_id)
    evidence["api_style"] = style
    evidence["api_style_ok"] = bool(style.get("ok"))


async def _exec_render_new_api(question: str, evidence: dict[str, Any]) -> str:
    from app.knowledge.add_api import render_new_api_plan

    kb = evidence.get("kb_spec") or {}
    style = evidence.get("api_style") or {}
    if not kb.get("full_text") and not kb.get("hits"):
        return (
            "未从知识库读到接口文档。请先在左侧「知识库」上传 md/txt/pdf，"
            "再说明要用哪份文档（例如：按知识库的登录接口文档新增接口）。"
        )
    return render_new_api_plan(
        question,
        kb=kb,
        style=style,
        business_steps=evidence.get("business_steps") or None,
        project_id=evidence.get("project_id") or "",
        entity_only=_wants_entity_only(question),
    )


def _wants_entity_only(question: str) -> bool:
    """True when the user asks for entity/VO/DTO classes only, not the full API.

    结构性判断（desktop-chat-d641bff7c836）：「根据知识库 商品查询 帮我生成
    对应的 java 实体类」→ 只要实体类；「新增一个登录接口」→ 完整接口。
    """
    q = (question or "").strip()
    if not q:
        return False
    has_entity = bool(
        re.search(r"实体类|实体|Entity|entity|VO|Vo|vo|Dto|DTO|JavaBean|javabean|类名", q)
    )
    if not has_entity:
        return False
    # 完整接口意图会同时出现 接口/Controller/方法/完整/整个 等词 → 非 entity_only。
    wants_full = bool(
        re.search(
            r"接口|Controller|endpoint|API|完整|整个|全套|代码块|方法",
            q,
            re.I,
        )
    )
    return not wants_full


def _merge_pev_answers(existing: str, part: str, *, heading: str) -> str:
    """Append a section so compound goals keep both deliverables."""
    part = (part or "").strip()
    if not part:
        return existing or ""
    block = f"{heading}\n\n{part}".strip()
    prev = (existing or "").strip()
    if not prev:
        return block
    if part in prev:
        return prev
    return prev.rstrip() + "\n\n---\n\n" + block


