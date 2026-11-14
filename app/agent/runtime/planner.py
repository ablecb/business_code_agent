"""LLM-first task DAG planner (regex only as fallback)."""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.runtime.skills import skill_title
from app.agent.runtime.types import SKILL_NAMES, SOLO_SKILLS, TaskDAG, TaskNode

logger = logging.getLogger("app.agent.runtime.planner")

SUBTASK_KINDS = SKILL_NAMES

_API_ANCHOR_RE = re.compile(
    r"((?:(?:(?![和与及、,，])[\u4e00-\u9fff]){2,8}|[A-Za-z][A-Za-z0-9_]{1,15})接口)"
)
_NUM_ITEM_RE = re.compile(
    r"(?:^|\n)\s*(?:\d+\s*[\.\)、]|[（(]\d+[）)]|[一二三四五六七八九十]\s*[、.])\s*"
)
_CONNECTOR_SPLIT_RE = re.compile(
    r"(?:另外|同时|并且|以及|还有|再问|再讲|再说|分别[：:]?)"
)
_MULTI_CONNECTOR_RE = re.compile(
    r"并且|同时|另外|以及|还有|再(问|说|讲)|分别|"
    r"[，,].{0,8}(并|还|另|同时)|"
    r"(流程|主链).{0,24}(状态|字段|失败|成功)|"
    r"(状态|字段).{0,24}(流程|怎么走)"
)


def _wants_flow_part(q: str) -> bool:
    return bool(
        re.search(
            r"(?:流程|怎么走|怎样走|主链|执行序|代码流程|完整说明|"
            r"接口什么流程|什么样的流程)",
            q,
        )
    )


def _wants_field_part(q: str) -> bool:
    try:
        from app.agent.field_evidence import wants_field_value_ask

        return bool(wants_field_value_ask(q))
    except Exception:
        return bool(re.search(r"状态|字段|落库|表.*什么|怎么更新|写成什么", q))


def _wants_change_part(q: str) -> bool:
    return bool(
        re.search(
            r"要改什么|改哪些|怎么改|改动点|影响哪些|加字段|新增字段|落库到",
            q,
        )
    )


def _wants_code_part(q: str) -> bool:
    return bool(re.search(r"源码|看代码|贴代码|怎么写的|代码是什么", q))


def _wants_diagnose_part(q: str) -> bool:
    return bool(
        re.search(
            r"日志|报错|异常|堆栈|失败原因|哪块有问题|error|exception",
            q,
            re.I,
        )
    )


def _wants_inventory_part(q: str) -> bool:
    return bool(
        re.search(
            r"(?:有哪些|有几个|哪些|几个|列出).{0,8}(?:接口|入口)|"
            r"(?:接口|入口).{0,8}(?:有哪些|有几个|列表)|"
            r"几个pom|有哪些yml|哪些配置",
            q,
        )
    )


def _wants_api_io_part(q: str) -> bool:
    return bool(
        re.search(r"请求参数|响应参数|入参出参|入参和出参|出参是什么", q)
    )


def _has_multi_connector(q: str) -> bool:
    return bool(_MULTI_CONNECTOR_RE.search(q))


def extract_entry_hint(question: str) -> str:
    q = (question or "").strip()
    if not q:
        return ""
    m = _API_ANCHOR_RE.search(q)
    if m:
        return m.group(1)
    m = re.search(
        r"([A-Za-z][\w]*(?:Controller|ServiceImpl)#[A-Za-z_]\w*)"
        r"|([A-Za-z][\w]*(?:Controller|ServiceImpl))",
        q,
    )
    if m:
        return next((g for g in m.groups() if g), "") or ""
    return ""


def distinct_api_anchors(question: str) -> list[str]:
    found = _API_ANCHOR_RE.findall(question or "")
    return list(dict.fromkeys(found))


def classify_skill(question: str) -> str:
    """Fallback skill when DAG LLM is unavailable — no keyword→skill cascade."""
    _ = (question or "").strip()
    return "general"


def _looks_like_question_piece(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 4:
        return False
    # Structural length / punctuation only — not a goal keyword table.
    if "?" in t or "？" in t:
        return True
    return len(t) >= 8


def split_question_segments(question: str) -> list[str]:
    """Split a compound ask into independent question pieces."""
    q = (question or "").strip()
    if not q:
        return []
    numbered = [p.strip() for p in _NUM_ITEM_RE.split(q) if p and p.strip()]
    if len(numbered) >= 2 and sum(1 for p in numbered if _looks_like_question_piece(p)) >= 2:
        return [p for p in numbered if _looks_like_question_piece(p)][:6]
    qmark = [p.strip(" ，,;；") for p in re.split(r"[？?]\s*", q) if p.strip()]
    qmark = [p for p in qmark if _looks_like_question_piece(p)]
    if len(qmark) >= 2:
        return qmark[:6]
    conn = [p.strip(" ，,;；") for p in _CONNECTOR_SPLIT_RE.split(q) if p.strip()]
    conn = [p for p in conn if _looks_like_question_piece(p)]
    if len(conn) >= 2:
        return conn[:6]
    return [q]


def wants_decompose(question: str) -> bool:
    """True when the ask clearly has ≥2 structural segments or API anchors.

    Skill/kind routing is LLM-owned; this only detects multi-question shape.
    """
    q = (question or "").strip()
    if not q or len(q) < 12:
        return False
    segs = split_question_segments(q)
    if len(segs) >= 2:
        return True
    anchors = distinct_api_anchors(q)
    if len(anchors) >= 2 and (
        _has_multi_connector(q) or "？" in q or "?" in q or "分别" in q
    ):
        return True
    return False


def _should_share_entry(nodes: list[TaskNode]) -> bool:
    if len(nodes) <= 1:
        return True
    skills = {n.skill for n in nodes}
    if skills & SOLO_SKILLS and len(skills) > 1:
        return False
    hints = [n.entry_hint for n in nodes]
    nonempty = [h for h in hints if h]
    if len(set(nonempty)) > 1:
        return False
    return True


def heuristic_dag(question: str) -> TaskDAG | None:
    """Structural multi-segment fallback — each node skill=general (LLM owns kinds)."""
    q = (question or "").strip()
    if not wants_decompose(q):
        return None
    segs = split_question_segments(q)
    if len(segs) < 2:
        return None
    nodes: list[TaskNode] = []
    seen_pair: set[tuple[str, str]] = set()
    for seg in segs:
        skill = "general"
        hint = extract_entry_hint(seg)
        key = (skill, hint or seg[:24])
        if key in seen_pair:
            continue
        seen_pair.add(key)
        nodes.append(
            TaskNode(
                id=f"S{len(nodes) + 1}",
                skill=skill,
                title=skill_title(skill),
                question=seg.strip() or q,
                entry_hint=hint,
            )
        )
    if len(nodes) < 2:
        return None
    uniq = nodes[:6]
    for i, n in enumerate(uniq):
        n.id = f"S{i + 1}"
        if i > 0 and not n.depends_on:
            n.depends_on = [uniq[i - 1].id]
    share = _should_share_entry(uniq)
    compose = (
        "按分节输出；入口相同则共享主链，入口不同则分别锁定；"
        "禁止编造未出现在分节证据中的结论"
    )
    return TaskDAG(
        nodes=uniq,
        source="heuristic",
        compose=compose,
        share_entry=share,
    )


def heuristic_decompose(question: str) -> dict[str, Any] | None:
    dag = heuristic_dag(question)
    return dag.to_legacy() if dag else None


def compose_subtask_answers(
    parts: list[dict[str, Any]],
    *,
    entry: str = "",
    compose_note: str = "",
) -> str:
    """Join subtask answers without inventing new facts."""
    lines = [
        "## 综合回答（复杂问题已分解）",
        "",
    ]
    entries = []
    if entry:
        entries.append(entry)
    for p in parts:
        e = str(p.get("entry") or "").strip()
        if e and e not in entries:
            entries.append(e)
    if len(entries) == 1:
        lines.append(f"共享业务入口：`{entries[0]}`")
        lines.append("")
    elif entries:
        lines.append("各节业务入口：" + "、".join(f"`{e}`" for e in entries))
        lines.append("")
    if compose_note:
        lines.append(f"> 合成说明：{compose_note}")
        lines.append("")
    for p in parts:
        title = p.get("title") or p.get("kind") or "分节"
        kind = p.get("kind") or p.get("skill") or ""
        ans = (p.get("answer") or "").strip()
        lines.append(f"### {title}" + (f"（`{kind}`）" if kind else ""))
        lines.append("")
        if ans:
            lines.append(ans)
        else:
            lines.append("（本分节证据不足，未生成正文）")
        lines.append("")
    lines.append(
        "说明：以上各节均基于代码/主链证据分别作答；"
        "若某节标明证据不足，请补充入口、关卡或日志后再问。"
    )
    return "\n".join(lines)


async def plan_task_dag(
    question: str,
    *,
    history_text: str = "",
) -> TaskDAG | None:
    """LLM-first DAG; structural heuristic only as fallback."""
    base = heuristic_dag(question)
    try:
        from app.core.utils import truncate
        from app.llm.factory import get_llm_provider

        def _parse_obj(content: str) -> dict[str, Any]:
            import json

            text = (content or "").strip()
            if not text:
                return {}
            try:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                data = json.loads(m.group(0) if m else text)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        llm = get_llm_provider()
        draft = base.to_legacy() if base else {"tasks": [], "note": "no heuristic draft"}
        raw = await llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是任务分解器。把复杂代码问答拆成 2～6 个原子任务，形成 DAG。"
                        "只输出 JSON：{\"share_entry\":true,\"compose\":\"\","
                        "\"tasks\":[{\"kind\":\"explain_flow|field_value|change_plan|"
                        "show_code|compose_code|diagnose|inventory|add_api|api_io|general\","
                        "\"title\":\"\",\"question\":\"子问题\","
                        "\"entry_hint\":\"xx接口或空\",\"depends_on\":[]}]}"
                        "kind 必须是给定枚举。子问题要短、可独立用代码证据回答。"
                        "必须按用户原话的语义先后排 tasks；后一任务 depends_on 前一任务。"
                        "只有用户明确说「同时」才允许无依赖。"
                        "不同业务接口必须给不同 entry_hint，且 share_entry=false。"
                        "同一接口的流程+字段：share_entry=true，entry_hint 相同。"
                        "不要写死具体 Class#method；不要依赖行业词表。"
                        "若问题其实是单一意图，输出少于 2 个 tasks（表示不分解）。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"历史:\n{truncate(history_text or '(空)', 800)}\n\n"
                        f"问题:\n{question}\n\n"
                        f"结构草案(可忽略):\n{draft}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=800,
        )
        data = _parse_obj(raw or "")
        tasks_in = data.get("tasks") if isinstance(data, dict) else None
        if not isinstance(tasks_in, list) or len(tasks_in) < 2:
            return base
        out_nodes: list[TaskNode] = []
        for i, t in enumerate(tasks_in[:6]):
            if not isinstance(t, dict):
                continue
            kind = str(t.get("kind") or t.get("skill") or "").strip()
            if kind not in SKILL_NAMES:
                continue
            title = str(t.get("title") or skill_title(kind))[:80]
            sub_q = str(t.get("question") or question).strip()[:400]
            hint = str(t.get("entry_hint") or extract_entry_hint(sub_q) or "")[:80]
            deps = t.get("depends_on") if isinstance(t.get("depends_on"), list) else []
            out_nodes.append(
                TaskNode(
                    id=f"S{i + 1}",
                    skill=kind,
                    title=title,
                    question=sub_q or question,
                    entry_hint=hint,
                    depends_on=[str(d) for d in deps if d],
                )
            )
        if len(out_nodes) < 2:
            return base
        idset = {n.id for n in out_nodes}
        for i, n in enumerate(out_nodes):
            n.depends_on = [d for d in n.depends_on if d in idset and d != n.id]
            if i > 0 and not n.depends_on:
                n.depends_on = [out_nodes[i - 1].id]
        share = data.get("share_entry")
        if share is None:
            share = _should_share_entry(out_nodes)
        return TaskDAG(
            nodes=out_nodes,
            compose=str(
                data.get("compose") or (base.compose if base else "") or ""
            )[:300],
            source="llm",
            share_entry=bool(share) and _should_share_entry(out_nodes),
        )
    except Exception:
        logger.debug("DAG planner LLM failed; heuristic only", exc_info=True)
        return base


async def decompose_question(
    question: str,
    *,
    history_text: str = "",
) -> dict[str, Any] | None:
    dag = await plan_task_dag(question, history_text=history_text)
    return dag.to_legacy() if dag else None
