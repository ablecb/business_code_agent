"""Layered explore agent: LLM picks tools until it can answer (ask_flow / 校验类问题)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.compaction import maybe_compact as _maybe_compact_messages
from app.agent.explore_add_field import (
    _ADD_FIELD_MIN_READS,
    _add_field_answer_has_offtopic_credit_noise,
    _add_field_answer_has_synth_junk,
    _add_field_answer_is_persist_gate_narrative,
    _add_field_answer_missing_scannable_format,
    _add_field_answer_wrong_apply_persist,
    _add_field_evidence_gaps,
    _add_field_role_of_hit,
    _add_field_seed_queries,
    _fallback_add_field_plan_from_evidence,
    _partial_add_field_plan_from_evidence,
    _sanitize_add_field_change_plan,
    _sanitize_protocol_sync_plan,
)
from app.agent.explore_answers import (
    _CLASS_METHOD_ANCHOR,
    _evidence_brief,
    _facts_map_evidence_text,
    _full_flow_answer_too_shallow,
    _is_messy_full_flow_answer,
    _sanitize_answer,
    _with_review_footnote,
)
from app.agent.explore_intent import (
    _ATOMIC_CN_FOR_PICK,
    _CN_IDENTITY_STEMS_PICK,
    _assemble_api_inventory_answer,
    _lexical_main_entry,
    _method_matches_gate_themes,
    _question_match_tokens,
    _token_hard_identity_for_pick,
    _user_question_core,
    _wants_api_inventory,
    _wants_callback_section,
    _wants_deep_flow_facts,
    _wants_endpoint_lookup,
    _wants_flow_downstream_apis,
    _wants_full_flow,
    _wants_gate_detail,
    _wants_notice_entry,
    _wants_source_paste,
    _wants_sync_entry_first,
    _wants_tech_detail,
    _wants_validation_focus,
)
from app.agent.explore_keys import (
    _CONCRETE_FLOW_MARKERS,
    _CONTROLLER_CM_RE,
    _GATE_PRIMARY_METHODS,
    _GATE_THEME_METHOD_STEMS,
    _PEDANTIC_NOTE_RE,
    _QUERYISH_METHOD,
    _RESULT_ENTRY_MARKERS,
)
from app.agent.explore_shared import (
    _class_method_from_hint,
    _collect_entry_calls_closure,
    _controller_stem,
    _hit_class_method,
    _hit_fqcn,
    _hit_method_name,
    _hit_text,
    _method_from_class_method,
    parse_numbered_answer_steps,
    _read_count,
    _simple_class_method,
)
from app.agent.explore_source import (
    _assemble_source_code_answer,
    _normalize_show_code_question,
)
from app.agent.explore_tools import dispatch_tool, tool_catalog_text
from app.agent.prompts import load_prompt
from app.agent.question_intent import (
    add_field_explore_guidance,
    assumed_new_field_name,
    composite_change_explore_guidance,
    protocol_sync_explore_guidance,
    wants_add_field_change_plan,
    wants_extend_field_to_protocol_sync,
    wants_show_gate_code,
)
from app.core.utils import truncate
from app.llm.factory import get_llm_provider
from app.llm.tracing import traceable
from app.tagging.search import hybrid_search

logger = logging.getLogger("app.agent.explore")

SYSTEM = load_prompt("explore_system") + "\n\n" + tool_catalog_text()


# Caps: keep wall-clock / chat rounds in check for typical flow Qs.
_DEFAULT_MAX_STEPS = 12
_FULL_FLOW_MAX_STEPS = 14
_ENDPOINT_MAX_STEPS = 8
_VALIDATION_MAX_STEPS = 10
# Add-field needs role coverage (VO + bind + assemble + entity) — more rounds than validation.
_ADD_FIELD_MAX_STEPS = 14
_OBS_CHARS = 3200
_MAX_BATCH_TOOLS = 3
_MAX_STYLE_FINISH_REJECTS = 1
_MAX_DEPTH_FINISH_REJECTS = 2
_MAX_REVIEW_REJECTS = 3  # answer-reviewer can send the agent back to dig deeper
_MSG_TAIL_KEEP = 8  # recent role messages after system+first user


# Explicit full-chain asks — includes colloquial「从入口到…串一下」, not bare「xx流程».


# Structural role atoms: user says 通知/回调/监听 → match those English name stems.
# Not a product Class#method map.


def _is_flow_doc_hit(h: Any) -> bool:
    if isinstance(h, dict):
        kind = h.get("kind") or ""
        sid = h.get("symbol_id") or ""
    else:
        kind = getattr(h, "kind", "") or ""
        sid = getattr(h, "symbol_id", "") or ""
    return kind in {"FlowDoc", "Overview"} or sid.startswith(("flow:", "overview:"))


def _hit_kind(h: Any) -> str:
    if isinstance(h, dict):
        return str(h.get("kind") or "")
    return str(getattr(h, "kind", "") or "")


def _is_controller_sibling_hit(h: Any, prefer_cm: str) -> bool:
    """True when hit is another Controller method on the same class as prefer_cm.

    Used so a generic flow offline corpus does not become a Controller
    endpoint catalog after entry_route locked WithdrawalsController#withdrawals.
    """
    prefer = (prefer_cm or "").strip()
    if "#" not in prefer:
        return False
    prefer_simple, prefer_method = prefer.split("#", 1)
    prefer_simple = prefer_simple.split(".")[-1]
    prefer_method = prefer_method.split("(", 1)[0].strip()
    if not prefer_simple or not prefer_method:
        return False

    cm = _hit_class_method(h)
    if cm and "#" in cm:
        c_simple, c_method = cm.split("#", 1)
        c_simple = c_simple.split(".")[-1]
        c_method = c_method.split("(", 1)[0].strip()
    else:
        fqcn = _hit_fqcn(h)
        c_simple = fqcn.split(".")[-1] if fqcn else ""
        c_method = (_hit_method_name(h) or "").split("(", 1)[0].strip()
    if not c_simple or not c_method:
        return False
    if c_simple != prefer_simple:
        return False
    if c_method == prefer_method:
        return False

    kind = _hit_kind(h)
    fqcn = _hit_fqcn(h)
    if _is_flow_doc_hit(h):
        # Wrong-entry FlowDoc on same class — treat as sibling noise for sync flow.
        return True
    return kind in {"Controller", "HTTP入口"} or "Controller" in fqcn


def _is_foreign_controller_hit(h: Any, prefer_cm: str) -> bool:
    """True when hit is a Controller not equal to the locked Class#method's class."""
    prefer = (prefer_cm or "").strip()
    if "#" not in prefer:
        return False
    prefer_simple = prefer.split("#", 1)[0].split(".")[-1]
    if not prefer_simple:
        return False
    kind = _hit_kind(h)
    fqcn = _hit_fqcn(h)
    cm = _hit_class_method(h)
    simple = ""
    if cm and "#" in cm:
        simple = cm.split("#", 1)[0].split(".")[-1]
    elif fqcn:
        simple = fqcn.split(".")[-1]
    if not simple or "Controller" not in simple:
        # Also treat HTTP入口 kind without Controller name via fqcn.
        if kind in {"Controller", "HTTP入口"} and fqcn and "Controller" in fqcn:
            simple = fqcn.split(".")[-1]
        else:
            return False
    return simple != prefer_simple


def _scope_prefetch_to_entry(
    prefetch: list[Any],
    *,
    prefer_cm: str,
    question: str,
    keep_unrelated: int = 0,
) -> list[Any]:
    """Keep materials for the locked Class#method; drop same-Controller siblings on flow Qs."""
    prefer = (prefer_cm or "").strip()
    if not prefer or not prefetch:
        return prefetch
    simple = prefer.split("#", 1)[0]
    method = prefer.split("#", 1)[1] if "#" in prefer else ""
    prefer_l = prefer.lower()
    # Downstream「调用了哪些接口」also demotes sibling Controllers.
    sync_flow = (
        (_wants_sync_entry_first(question) or _wants_flow_downstream_apis(question))
        and not _wants_api_inventory(question)
    )
    downstream = _wants_flow_downstream_apis(question) and not _wants_api_inventory(
        question
    )
    scoped: list[Any] = []
    other: list[Any] = []
    for h in prefetch:
        if sync_flow and _is_controller_sibling_hit(h, prefer):
            other.append(h)
            continue
        # Downstream asks: drop AmountController / EntrustedRefundController etc.
        if downstream and _is_foreign_controller_hit(h, prefer):
            other.append(h)
            continue
        blob = _hit_text(h).lower()
        cm = _hit_class_method(h).lower()
        # Sibling Controllers already filtered; class-name match keeps ServiceImpl / nodes.
        if (
            prefer_l in blob
            or prefer_l == cm
            or (simple and simple.lower() in blob)
            or (method and method.lower() in blob and simple.lower() in blob)
        ):
            scoped.append(h)
        else:
            other.append(h)
    if not scoped:
        # Locked entry has no materials — do NOT fall back to unrelated FlowDocs
        # (that caused notice-API → unrelated offline narrative).
        if keep_unrelated <= 0:
            return []
        return other[:keep_unrelated]
    return scoped + (other[:keep_unrelated] if keep_unrelated else [])


def _looks_like_result_entry(
    *,
    class_method: str,
    paths: list[str],
    title: str,
    flow_role: str,
) -> bool:
    """True if this flow_doc is a result callback/notice entry (not sync business entry)."""
    role = (flow_role or "").lower()
    if role in {"callback", "async"}:
        return True
    if role == "notice":
        return True
    blob = f"{class_method} {title} {' '.join(paths)}"
    return bool(_RESULT_ENTRY_MARKERS.search(blob))


def _offline_blob_aligned(blob: str, question: str) -> bool:
    """True when offline corpus identity looks related to question content nouns."""
    text = blob or ""
    q = question or ""
    if not text or not q:
        return True
    # Extract Class#method-like anchors from corpus.
    cms = re.findall(r"\b([A-Z][A-Za-z0-9_]+(?:Controller|ServiceImpl|Listener)#\w+)", text)
    if not cms:
        # No structural anchors — require at least one content token in blob.
        toks = [
            t
            for t in _question_match_tokens(q)
            if t not in {"接口", "流程", "业务", "描述", "一下", "什么样"}
        ]
        return any(t in text for t in toks[:6]) if toks else True
    return any(_entry_aligned_with_question(cm, q) for cm in cms[:12])


def _entry_aligned_with_question(class_method: str, question: str) -> bool:
    """False when locked entry clearly mismatches question *entry-role* tokens.

    Only entry-defining role cues are checked (e.g. 通知接口 → notice/notification).
    Trailing「…到回调」is a section ask, not an entry-class requirement.
    No product CN→EN map.
    """
    cm = (class_method or "").strip()
    q = question or ""
    if not cm or not q:
        return True
    lexical = _lexical_main_entry(q)
    if lexical and lexical.lower() == cm.lower():
        return True
    toks = _question_match_tokens(q)
    content = [
        t
        for t in toks
        if t
        not in {
            "接口",
            "流程",
            "业务",
            "描述",
            "一下",
            "什么样",
            "怎么走",
            "怎样",
            "相关",
        }
        and len(t) >= 2
    ]
    if not content:
        return True
    id_l = cm.lower()
    # Entry-role tokens only (not trailing 回调 section, not product nouns).
    strong: list[str] = []
    wants_notice = _wants_notice_entry(q)
    callback_as_entry = bool(
        re.search(r"(?i)回调接口|回调流程|CallbackController|onCallback", q)
    ) and not _wants_callback_section(q)
    for t in content:
        if t == "通知" or t in _CN_IDENTITY_STEMS_PICK.get("通知", ()):
            if wants_notice or re.search(r"通知接口|通知流程|通知入口", q):
                strong.append("通知")
            continue
        if t == "回调" or t in _CN_IDENTITY_STEMS_PICK.get("回调", ()):
            if callback_as_entry:
                strong.append("回调")
            continue
        if t in _CN_IDENTITY_STEMS_PICK or t in _ATOMIC_CN_FOR_PICK:
            strong.append(t)
    if not strong:
        return True
    for tok in strong:
        if tok.lower() in id_l:
            return True
        for stem in _CN_IDENTITY_STEMS_PICK.get(tok) or ():
            if stem and stem.lower() in id_l:
                return True
    return False


def _flow_question_hit_boost(h: Any, question: str) -> float:
    """Re-rank retrieval for「流程」Qs: prefer sync action entries over query/callback."""
    if not _wants_sync_entry_first(question):
        return 0.0
    boost = 0.0
    cm = _hit_class_method(h)
    method = _hit_method_name(h)
    fqcn = _hit_fqcn(h)
    kind = (h.get("kind") if isinstance(h, dict) else getattr(h, "kind", None)) or ""
    text = _hit_text(h)

    if _looks_like_result_entry(class_method=cm, paths=[], title=text[:80], flow_role=""):
        # Notice/callback flow — notice/callback naming IS the entry, do not demote.
        if not _wants_notice_entry(question):
            boost -= 25.0
        else:
            boost += 20.0
    if _QUERYISH_METHOD.match(method or ""):
        boost -= 35.0
    stem = _controller_stem(fqcn or cm)
    if stem and method and stem in method.lower():
        # RepayController#repayment / OrderController#confirm won't match; repay⊂repayment will.
        boost += 30.0
    if kind in {"Controller", "HTTP入口"} or "Controller" in fqcn:
        if method and not _QUERYISH_METHOD.match(method):
            boost += 12.0
    # Prefer true data-source entry stubs / flow entries over deep side nodes when asking 流程
    if "数据源头" in text or "HTTP入口" in text or "流程入口" in text:
        if method and not _QUERYISH_METHOD.match(method):
            boost += 8.0
    return boost


def _rerank_hits_for_question(hits: list[Any], question: str) -> list[Any]:
    if not hits:
        return hits
    scored: list[tuple[float, int, Any]] = []
    for i, h in enumerate(hits):
        base = float(
            (h.get("score") if isinstance(h, dict) else getattr(h, "score", None)) or 0.0
        )
        scored.append((base + _flow_question_hit_boost(h, question), -i, h))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [h for _, _, h in scored]


def _resolve_flow_entry_anchor(project_id: str, question: str) -> dict[str, Any] | None:
    """Search entry_route collection first for flow questions → Class#method anchor."""
    if not project_id or not (question or "").strip():
        return None
    # Only for flow-style / callback-result asks; endpoint lookups keep hybrid alone.
    if not (
        _wants_sync_entry_first(question)
        or _wants_full_flow(question)
        or _wants_flow_downstream_apis(question)
        or _wants_callback_section(question)
        or "流程" in (question or "")
        or re.search(r"回调|callback", question or "", re.I)
    ):
        return None
    lexical = _lexical_main_entry(question)
    try:
        from app.analysis.entry_route import resolve_entry_anchor

        hit = resolve_entry_anchor(project_id, question, top_k=8)
    except Exception:
        hit = None
    # Lexical topic from the question wins over embedding near-miss / empty Milvus.
    if lexical:
        route_cm = str((hit or {}).get("class_method") or "").strip()
        if not route_cm or route_cm.split("#", 1)[0].split(".")[-1] != lexical.split(
            "#", 1
        )[0]:
            return {
                "class_method": lexical,
                "flow_role": "main",
                "api_doc": "",
                "score": 999.0,
                "source": "lexical",
            }
        # Same Controller family — keep vector hit method if it matches lexical method.
        if route_cm.split("#", 1)[-1].split("(", 1)[0] == lexical.split("#", 1)[-1]:
            return hit
        return {
            "class_method": lexical,
            "flow_role": "main",
            "api_doc": "",
            "score": 999.0,
            "source": "lexical",
        }
    return hit


def _load_flow_doc_by_class_method(
    project_id: str, class_method: str
) -> dict[str, Any] | None:
    """Load MySQL FlowDoc by exact Class#method (ignores question soft-match)."""
    prefer = (class_method or "").strip()
    if not project_id or not prefer or "#" not in prefer:
        return None
    prefer_l = prefer.lower()
    prefer_simple = prefer.split("#", 1)[0].split(".")[-1].lower()
    prefer_method = prefer.split("#", 1)[1].split("(", 1)[0].strip().lower()
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session
        from app.models.orm import SymbolDescORM
    except Exception:
        return None
    best: dict[str, Any] | None = None
    best_n = -1
    try:
        with get_session() as session:
            rows = session.scalars(
                select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
            ).all()
            for d in rows:
                ej = dict(d.extra_json or {})
                fd = ej.get("flow_doc")
                if not isinstance(fd, dict):
                    continue
                stages = fd.get("stages") or []
                if not isinstance(stages, list) or len(stages) < 2:
                    continue
                cm = str(ej.get("class_method") or "").strip()
                if not cm or "#" not in cm:
                    continue
                cm_l = cm.lower()
                c_simple = cm.split("#", 1)[0].split(".")[-1].lower()
                c_method = cm.split("#", 1)[1].split("(", 1)[0].strip().lower()
                if cm_l != prefer_l and not (
                    c_simple == prefer_simple and c_method == prefer_method
                ):
                    continue
                n = len(stages)
                if n <= best_n:
                    continue
                best_n = n
                paths = [str(p) for p in (ej.get("http_paths") or []) if p]
                roles = ej.get("flow_roles") or []
                flow_role = str(
                    fd.get("flow_role")
                    or (roles[0] if isinstance(roles, list) and roles else "")
                    or ""
                )
                fk = str(fd.get("flow_key") or "")
                if not fk:
                    fks = ej.get("flow_keys") or []
                    if isinstance(fks, list) and fks:
                        fk = str(fks[0])
                best = {
                    "summary": d.summary or "",
                    "class_method": cm,
                    "http_paths": paths,
                    "flow_key": fk,
                    "flow_role": flow_role,
                    "is_result_entry": _looks_like_result_entry(
                        class_method=cm,
                        paths=paths,
                        title=str(fd.get("title") or ""),
                        flow_role=flow_role,
                    ),
                    "cm_match": True,
                    "hard_hits": 1,
                    "soft_hits": 0,
                    "flow_doc": fd,
                }
    except Exception:
        return None
    return best


def _overlay_follows_on_flow(
    project_id: str,
    flow: dict[str, Any] | None,
    *,
    entry_cm: str | None = None,
    flow_key: str | None = None,
) -> dict[str, Any] | None:
    """Replace FlowDoc stages with Neo4j FOLLOWS spine (+ MySQL descriptions).

    Lookup order: Milvus/entry lock → Neo4j FOLLOWS → stage titles/descs.
    Falls back to the input ``flow`` when FOLLOWS is unavailable.
    """
    if not project_id:
        return flow
    entry = (entry_cm or "").strip()
    if not entry and flow:
        entry = str(flow.get("class_method") or "").strip()
    if not entry or "#" not in entry:
        return flow
    fk = (flow_key or "").strip()
    if not fk and flow:
        fk = str(flow.get("flow_key") or "").strip()
        fd0 = flow.get("flow_doc") if isinstance(flow.get("flow_doc"), dict) else {}
        if not fk and fd0:
            fk = str(fd0.get("flow_key") or "").strip()
    try:
        from app.analysis.flow_neo4j_sync import get_follows_path

        follows = get_follows_path(project_id, entry, fk or "")
    except Exception:
        logger.debug("overlay FOLLOWS failed", exc_info=True)
        return flow
    if not follows.get("ok") or len(follows.get("steps") or []) < 2:
        return flow

    import copy

    base: dict[str, Any] = copy.deepcopy(flow) if flow else {}
    stages: list[dict[str, Any]] = []
    for st in follows.get("steps") or []:
        if not isinstance(st, dict):
            continue
        cm = str(st.get("class_method") or "").strip()
        if not cm:
            continue
        method = cm.split("#", 1)[-1] if "#" in cm else cm
        raw_title = str(st.get("title") or "").strip()
        desc_b = str(st.get("desc_business") or st.get("summary") or "").strip()
        # Prefer graph/MySQL titles; never invent product-specific Chinese labels.
        if raw_title and raw_title != method and raw_title != desc_b and len(raw_title) <= 40:
            title = raw_title
        else:
            title = raw_title if raw_title and len(raw_title) <= 40 else method
        stages.append(
            {
                "step": st.get("step"),
                "class_method": cm,
                "title": title,
                "desc_business": desc_b,
                "desc_dev": st.get("desc_dev") or "",
                "desc": st.get("desc") or desc_b or "",
                "status_line": "",
                "data_flow": st.get("data_flow") or "",
            }
        )
    fd = dict(base.get("flow_doc") or {})
    if follows.get("title"):
        fd["title"] = follows["title"]
    if follows.get("narrative"):
        fd["narrative"] = follows["narrative"]
        fd.setdefault("narrative_business_clear", follows["narrative"])
    fd["stages"] = stages
    fd["execution_spine"] = list(follows.get("ordered_path") or [])
    fd["stage_order_source"] = str(follows.get("source") or "neo4j_follows")
    fd["flow_key"] = follows.get("flow_key") or fk or fd.get("flow_key") or ""
    base["flow_doc"] = fd
    base["class_method"] = follows.get("entry") or entry
    base["flow_key"] = fd.get("flow_key") or base.get("flow_key") or ""
    if follows.get("http_paths"):
        base["http_paths"] = list(follows["http_paths"])
    elif not base.get("http_paths"):
        base["http_paths"] = []
    base["follows_source"] = follows.get("source")
    base["cm_match"] = True
    return base


def _downstream_answer_foreign_controllers(
    answer: str, flow: dict[str, Any] | None
) -> list[str]:
    """Controller#method names in answer that are not the locked entry."""
    if not flow:
        return []
    entry = str(flow.get("class_method") or "").strip()
    entry_simple = entry.split("#", 1)[0].split(".")[-1] if "#" in entry else ""
    entry_method = (
        entry.split("#", 1)[1].split("(", 1)[0].strip() if "#" in entry else ""
    )
    allowed = {f"{entry_simple}#{entry_method}".lower()} if entry_simple else set()
    foreign: list[str] = []
    seen: set[str] = set()
    for m in _CONTROLLER_CM_RE.finditer(answer or ""):
        simple = m.group(1).split(".")[-1]
        method = m.group(2)
        key = f"{simple}#{method}".lower()
        if key in allowed or key in seen:
            continue
        seen.add(key)
        foreign.append(f"{simple}#{method}")
    return foreign


def _pick_structured_flow_doc(
    project_id: str,
    question: str,
    *,
    prefer_class_method: str | None = None,
) -> dict[str, Any] | None:
    """Load best MySQL flow_doc (with stages) by overlap with the user question.

    When the user asks「从入口到回调」/「xx流程」, prefer a sync business entry
    FlowDoc and heavily demote Callback/Notice/Listener and query/get side paths
    (even if their narrative mentions the same business nouns).

    If prefer_class_method is set (from entry_route), that entry wins when a
    matching FlowDoc exists.
    """
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session
        from app.models.orm import SymbolDescORM
    except Exception:
        return None

    prefer_tokens = _question_match_tokens(question)
    sync_first = _wants_sync_entry_first(question)
    notice_entry = _wants_notice_entry(question)
    wants_cb = _wants_callback_section(question) or bool(
        re.search(r"回调|callback", question or "", re.I)
    )
    prefer_cm = (prefer_class_method or "").strip()
    prefer_cm_l = prefer_cm.lower()
    best: dict[str, Any] | None = None
    best_score = -10_000
    try:
        with get_session() as session:
            rows = session.scalars(
                select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
            ).all()
            for d in rows:
                ej = dict(d.extra_json or {})
                fd = ej.get("flow_doc")
                if not isinstance(fd, dict):
                    continue
                stages = fd.get("stages") or []
                if not isinstance(stages, list) or len(stages) < 3:
                    continue
                paths = [str(p) for p in (ej.get("http_paths") or []) if p]
                cm = str(ej.get("class_method") or "")
                method = _method_from_class_method(cm)
                title = str(fd.get("title") or "")
                api_doc = str(ej.get("api_doc") or "")
                roles = ej.get("flow_roles") or []
                flow_role = str(
                    fd.get("flow_role")
                    or (roles[0] if isinstance(roles, list) and roles else "")
                    or ""
                )
                narrative = str(fd.get("narrative") or "")
                summary = str(d.summary or "")
                # Strong identity: class#method / paths / Swagger api_doc.
                # Title alone is weaker (narratives often reuse shared business nouns).
                strong_id = f"{cm} {' '.join(paths)} {api_doc}".lower()
                title_l = title.lower()
                soft = f"{narrative} {summary}".lower()

                # Cap stage-count so a long callback BFS cannot outrank the real entry.
                score = min(len(stages), 10)
                if ej.get("entry") or ej.get("is_entry") or ej.get("important"):
                    score += 6
                if paths:
                    score += 18
                if flow_role == "main":
                    score += 12

                is_result = _looks_like_result_entry(
                    class_method=cm, paths=paths, title=title, flow_role=flow_role
                )
                is_queryish = bool(_QUERYISH_METHOD.match(method or ""))
                stem = _controller_stem(cm)
                stem_in_method = bool(stem and method and stem in method.lower())
                cm_match = bool(prefer_cm_l and cm.lower() == prefer_cm_l)
                if cm_match:
                    # entry_route already chose this Class#method — lock FlowDoc to it.
                    score += 200

                if notice_entry and is_result:
                    # Notice/callback flow — notice entry IS the subject.
                    score += 40
                elif sync_first and is_result and not cm_match:
                    score -= 100
                elif wants_cb and is_result:
                    score += 28
                elif wants_cb and flow_role == "main" and not cm_match:
                    score -= 18
                elif is_result:
                    score -= 15
                if notice_entry and flow_role == "main" and not cm_match:
                    # Demote an unrelated main when asking a notice API.
                    score -= 80
                if prefer_cm_l and not cm_match and not _entry_aligned_with_question(
                    cm, question
                ):
                    score -= 120
                if sync_first and is_queryish and not cm_match:
                    # 「还款流程」must not lock onto queryRepayBank / queryXxx side APIs.
                    score -= 55
                if sync_first and stem_in_method and not is_queryish:
                    score += 22

                hard_hits = 0
                soft_hits = 0
                for tok in prefer_tokens:
                    tl = tok.lower()
                    if not tl:
                        continue
                    if tl in strong_id or _token_hard_identity_for_pick(
                        tok, cm=cm, paths=paths, api_doc=api_doc
                    ):
                        hard_hits += 1
                        if api_doc and tl == api_doc.strip().lower():
                            score += 40
                        else:
                            score += 28
                    elif tl in title_l:
                        soft_hits += 1
                        score += 8
                    elif tl in soft:
                        soft_hits += 1
                        score += 3

                # Soft-only overlap is not enough to pick a FlowDoc for sync「流程」Qs.
                if sync_first and prefer_tokens and hard_hits == 0 and soft_hits > 0:
                    if is_queryish or is_result:
                        score -= 40
                    else:
                        score -= 12
                if prefer_tokens and hard_hits == 0 and soft_hits == 0 and not cm_match:
                    score -= 90

                if score > best_score:
                    best_score = score
                    fk = str(fd.get("flow_key") or "")
                    if not fk:
                        fks = ej.get("flow_keys") or []
                        if isinstance(fks, list) and fks:
                            fk = str(fks[0])
                    best = {
                        "summary": summary,
                        "class_method": cm,
                        "http_paths": paths,
                        "flow_key": fk,
                        "flow_role": flow_role,
                        "is_result_entry": is_result,
                        "is_queryish": is_queryish,
                        "stem_in_method": stem_in_method,
                        "hard_hits": hard_hits,
                        "soft_hits": soft_hits,
                        "cm_match": cm_match,
                        "flow_doc": fd,
                    }
    except Exception:
        return None
    if not best:
        return None
    # Prefer-anchor hit is always accepted.
    if best.get("cm_match"):
        return best
    # Refuse to lock the answer onto a soft-matched query/callback / unrelated FlowDoc.
    if sync_first and prefer_tokens:
        hard = int(best.get("hard_hits") or 0)
        soft = int(best.get("soft_hits") or 0)
        if hard == 0 and soft == 0:
            return None
        if hard == 0 and not best.get("stem_in_method"):
            # Title/narrative-only hits (shared nouns) without Class↔method stem alignment.
            return None
        if best.get("is_queryish") and hard == 0:
            return None
        if best.get("is_result_entry") and hard == 0:
            return None
    return best


# Internal correction asides baked into FlowDoc descs — hide in business presentation.


def _load_deep_facts_by_class_methods(
    project_id: str,
    class_methods: list[str],
) -> dict[str, dict[str, Any]]:
    """Map Class#method → deep-parse facts from MySQL symbol_desc."""
    from app.core.utils import truncate

    wanted = [cm for cm in class_methods if cm and "#" in cm]
    if not wanted or not project_id:
        return {}
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session
        from app.models.orm import SymbolDescORM, SymbolORM
    except Exception:
        return {}

    out: dict[str, dict[str, Any]] = {}
    try:
        with get_session() as session:
            for cm in wanted:
                simple, method = cm.split("#", 1)
                simple = simple.split(".")[-1]
                method = method.split("(", 1)[0].strip()
                syms = list(
                    session.scalars(
                        select(SymbolORM).where(
                            SymbolORM.project_id == project_id,
                            SymbolORM.simple_name == simple,
                            SymbolORM.method_name == method,
                        )
                    )
                )
                for sym in syms[:2]:
                    desc = session.scalar(
                        select(SymbolDescORM).where(SymbolDescORM.symbol_id == sym.id)
                    )
                    if not desc:
                        continue
                    extra = desc.extra_json if isinstance(desc.extra_json, dict) else {}
                    status = extra.get("status_transitions") or []
                    if not isinstance(status, list):
                        status = []
                    db_ops = []
                    iv = extra.get("impl_view") if isinstance(extra.get("impl_view"), dict) else {}
                    for op in iv.get("db_ops") or []:
                        if not isinstance(op, dict):
                            continue
                        fields = op.get("field_sources") or {}
                        status_fields = [
                            f"{k}={v}"
                            for k, v in (fields.items() if isinstance(fields, dict) else [])
                            if re.search(r"status|state|Status|State", str(k))
                        ]
                        if status_fields or re.search(
                            r"status|状态", str(op.get("persist_changes") or ""), re.I
                        ):
                            db_ops.append(
                                f"{op.get('op')}:{op.get('table_or_entity') or ''} "
                                f"({','.join(status_fields) or op.get('persist_changes') or ''})"
                                f" via {op.get('via_call') or ''}"
                            )
                    status_line = "；".join(str(x) for x in status[:8] if x)
                    if not status_line and db_ops:
                        status_line = "；".join(db_ops[:4])
                    paths = extra.get("http_paths") or []
                    if not isinstance(paths, list):
                        paths = []
                    bv = (
                        extra.get("business_view")
                        if isinstance(extra.get("business_view"), dict)
                        else {}
                    )
                    rules_raw = bv.get("rules") if isinstance(bv, dict) else None
                    if not isinstance(rules_raw, list):
                        rules_raw = []
                    vp_raw = extra.get("validation_points") or []
                    if not isinstance(vp_raw, list):
                        vp_raw = []
                    facts = {
                        "class_method": f"{simple}#{method}",
                        "biz_clear": str(desc.summary_business_clear or "").strip(),
                        "biz_detail": str(desc.summary_business_detail or "").strip(),
                        "dev_clear": str(desc.summary_dev_clear or "").strip(),
                        "summary": str(desc.summary or "").strip(),
                        "http_paths": [str(p) for p in paths if p],
                        "status_line": truncate(status_line, 360),
                        "data_flow": truncate(str(extra.get("data_flow_narrative") or ""), 220),
                        "capability": truncate(str(bv.get("capability") or ""), 220),
                        "business_rules": [
                            str(x).strip() for x in rules_raw if str(x).strip()
                        ][:16],
                        "validation_points": [
                            str(x).strip() for x in vp_raw if str(x).strip()
                        ][:20],
                    }
                    if any(
                        facts.get(k)
                        for k in ("biz_clear", "biz_detail", "dev_clear", "summary", "status_line")
                    ):
                        out[f"{simple}#{method}"] = facts
                        out[cm] = facts
                    break
    except Exception:
        logger.debug("load deep facts failed", exc_info=True)
    return out


_SEGMENT_MARK_RE = re.compile(r"\[段\d+\]\s*")
_SEGMENT_NARRATION_RE = re.compile(
    r"(?:本段代码(?:首先|主要|接着|随后)?|"
    r"本段(?:首先|主要|接着|随后)?(?:代码)?(?:处理|完成|用于|为)?|"
    r"代码首先|该方法用于|该方法用于将)"
)
_FIELD_NULL_RULE_RE = re.compile(
    r"(?:非空(?:校验|检查)?|不能为空|不得为空|是否为空|为空(?:时|则|校验)?|"
    r"字段(?:缺失|为空)|缺失或格式错误|格式必须为\s*\d+\s*位)"
)
_BIZ_GATE_HINT_RE = re.compile(
    r"(?:开关|决策|重复|加锁|落库|路由|审批|"
    r"日限|签约|拦截|异步|回调|占用|注销|资金|新老流程|版本|"
    r"[\u4e00-\u9fff]{2,8}校验)"
)
# Higher score = prefer when assembling「关键业务关卡」bullets.
_BIZ_GATE_PRIORITY_RE = re.compile(
    r"(?:审批|资金路由|开关|拦截|重复请求|加锁|"
    r"新老流程|流程版本|日限|签约|注销|占用|"
    r"[\u4e00-\u9fff]{2,8}校验)"
)
# No product-specific checklist / plumbing / spine label maps in agent code.
_FIELD_CHECKLIST_METHODS: frozenset[str] = frozenset()
_PLUMBING_METHODS: frozenset[str] = frozenset()
_SPINE_BIZ_LABEL: dict[str, str] = {}
_NESTED_FLOW_GATE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = ()


def _clean_stored_biz_text(text: str) -> str:
    """Light cleanup of treesitter dual-summary text; do not rewrite meaning."""
    t = _SEGMENT_MARK_RE.sub("", text or "")
    t = t.replace("；[段", "；")  # leftover if mark already stripped oddly
    t = re.sub(r"[；;]{2,}", "；", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip(" ；;\n")


def _strip_segment_narration(text: str) -> str:
    """Drop treesitter '本段代码首先…' wrappers; keep the business clause."""
    t = _clean_stored_biz_text(text)
    if not t:
        return ""
    parts: list[str] = []
    for raw in re.split(r"[\n]+", t):
        chunk = raw.strip(" ；;")
        if not chunk:
            continue
        chunk = _SEGMENT_NARRATION_RE.sub("", chunk).strip(" ，,；;。")
        if not chunk:
            continue
        # Drop pure call-chain dumps ("调用 generateReqParam … 调用 loan …")
        if re.fullmatch(
            r"(?:调用\s*[\w.#]+\s*(?:整合|持久化|处理|方法)?[^。]*){2,}",
            chunk,
        ):
            continue
        if not chunk.endswith(("。", "！", "？", "…")):
            chunk += "。"
        parts.append(chunk)
    # Merge into fluent prose (one paragraph).
    out = "".join(parts)
    out = re.sub(r"。{2,}", "。", out)
    return out.strip()


def _is_field_null_rule(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if _BIZ_GATE_HINT_RE.search(t) and not re.search(
        r"(?:非空|不能为空|是否为空)", t
    ):
        return False
    return bool(_FIELD_NULL_RULE_RE.search(t)) and not (
        _BIZ_GATE_HINT_RE.search(t)
        and re.search(r"(?:校验|开关|拦截|签约)", t)
    )


def _first_biz_sentence(text: str, *, max_len: int = 120) -> str:
    t = _strip_segment_narration(text) or _clean_stored_biz_text(text)
    if not t:
        return ""
    sent = re.split(r"[。！？\n]", t, maxsplit=1)[0].strip(" ；;")
    if not sent:
        return ""
    if len(sent) > max_len:
        sent = sent[: max_len - 1].rstrip("，,、；; ") + "…"
    return sent if sent.endswith(("。", "…")) else sent + "。"


def _coerce_biz_prose(raw: str) -> str:
    """Normalize LLM-stored list/JSON blobs into readable Chinese prose."""
    import ast
    import json

    t = (raw or "").strip()
    if not t:
        return ""
    if (t.startswith("[") and t.endswith("]")) or (t.startswith("{") and "capability" in t[:80]):
        data: Any = None
        try:
            data = json.loads(t)
        except Exception:
            try:
                data = ast.literal_eval(t)
            except Exception:
                data = None
        if isinstance(data, list):
            parts = [str(x).strip().rstrip("。；;") for x in data if str(x).strip()]
            return "。".join(parts) + ("。" if parts else "")
        if isinstance(data, dict):
            for k in ("capability", "summary", "text", "desc"):
                if data.get(k):
                    return _coerce_biz_prose(str(data.get(k)))
    t = re.sub(r"【业务[·・.]?精简】\s*", "", t)
    t = re.sub(r"【研发[·・.].*?】\s*", "", t)
    return t.strip()


def _entry_business_overview(entry: dict[str, Any]) -> str:
    """Prefer clear capability prose; fall back to cleaned detail without segment junk."""
    clear_raw = _coerce_biz_prose(str(entry.get("biz_clear") or entry.get("capability") or ""))
    # biz_clear often joins segments with "；[段2] …" — keep first clause only.
    clear_first = re.split(r"[；;]\s*", _clean_stored_biz_text(clear_raw), maxsplit=1)[
        0
    ]
    clear = _strip_segment_narration(clear_first) or _strip_segment_narration(clear_raw)
    detail = _strip_segment_narration(
        _coerce_biz_prose(str(entry.get("biz_detail") or ""))
    )
    # Prefer clear when detail is long call-chain narration; else keep detail's
    # first business paragraph (already narration-stripped).
    if clear and (
        not detail
        or len(detail) > 280
        or detail.count("调用") >= 3
        or re.search(r"generateReqParam|insertLoanData", detail)
        or detail.startswith("[")
    ):
        body = clear
        # Only append detail when clear is thin AND not already covering 受理/校验.
        if (
            detail
            and len(clear) < 60
            and not detail.startswith("[")
            and not re.search(r"(?:接收|校验|受理|加锁|决策)", clear)
        ):
            extra = _first_biz_sentence(detail, max_len=140)
            if extra and extra not in body and not re.search(
                r"generateReqParam|insertLoanData", extra
            ):
                body = body.rstrip("。") + "。" + extra
        return body
    if detail and not detail.startswith("["):
        # Cap verbose multi-clause detail to ~2 sentences for overview.
        sents = [s.strip() for s in re.split(r"(?<=[。！？])", detail) if s.strip()]
        kept: list[str] = []
        for s in sents:
            if re.search(r"generateReqParam|insertLoanData", s) and "调用" in s:
                continue
            kept.append(s)
            if len(kept) >= 2:
                break
        return "".join(kept) or clear or detail
    return clear or _clean_stored_biz_text(
        _coerce_biz_prose(str(entry.get("summary") or ""))
    )


def _gate_theme(text: str) -> str:
    """Coarse theme key so we diversify gate bullets (from the text itself)."""
    t = text or ""
    m = re.search(r"[\u4e00-\u9fff]{2,8}校验", t)
    if m:
        return m.group(0)
    for key, pat in (
        ("审批", r"审批"),
        ("开关", r"开关|拦截"),
        ("路由", r"资金路由|路由"),
        ("流程", r"流程版本|loanFlow|新老流程|异步"),
        ("重复", r"重复|加锁"),
        ("签约", r"签约"),
    ):
        if re.search(pat, t):
            return key
    return "其他"


def _gate_priority_score(text: str) -> int:
    """Rank stored rules: business gates first, generic persist/fail last."""
    t = text or ""
    if _is_field_null_rule(t):
        return -5
    score = 0
    if _BIZ_GATE_PRIORITY_RE.search(t):
        score += 8
    if re.search(r"(?:校验|审批)", t):
        score += 4
    if re.search(r"(?:开关|拦截|重复|加锁|流程版本|资金路由)", t):
        score += 3
    # Prefer decisive gate wording over side-branch narration.
    if re.search(r"(?:失败则|不足|拦截|阻断|必须|否则返回)", t):
        score += 2
    if re.search(r"(?:设备信息|联系人|地址信息|保存失败|获取失败|再次提交|信息比对)", t):
        score -= 3
    if re.search(r"(?:写入数据库|表示流程被阻断)$", t.rstrip("。")):
        score -= 2  # outcome side-effect, weaker than the gate itself
    if re.search(r"(?:调用\s*\w+|封装为 Response|写库|插入)", t) and not _BIZ_GATE_PRIORITY_RE.search(
        t
    ):
        score -= 2
    if re.search(r"^否则根据", t):
        score -= 4  # incomplete fragment from segmented rules
    return score


def _pick_gate_bullets(
    facts: dict[str, Any],
    *,
    max_n: int = 4,
    allow_field_null: bool = False,
) -> list[str]:
    """Select business-critical rules / validations from stored desc facts."""
    candidates: list[tuple[int, str]] = []
    seen: set[str] = set()

    def _consider(item: str, *, source_bonus: int = 0) -> None:
        raw = (item or "").strip()
        if not raw:
            return
        key = re.sub(r"\s+", "", raw.strip(" ；;。"))
        if not key or key in seen:
            return
        if _looks_like_code_status_dump(raw):
            return
        if not allow_field_null and _is_field_null_rule(raw):
            return
        if re.search(r"返回\s*false|返回\s*true", raw, re.I) and not re.search(
            r"(?:校验|审批|开关为)", raw
        ):
            return  # code-level return narration
        seen.add(key)
        s = raw.rstrip("。；;") + "。"
        candidates.append((_gate_priority_score(s) + source_bonus, s))

    for r in facts.get("business_rules") or []:
        _consider(str(r), source_bonus=2)
    for vp in facts.get("validation_points") or []:
        if allow_field_null or (
            _BIZ_GATE_HINT_RE.search(str(vp)) and not _is_field_null_rule(str(vp))
        ):
            _consider(str(vp), source_bonus=1)
    # Also mine cleaned biz_clear for gate themes missing from rules.
    clear = _strip_segment_narration(
        str(facts.get("biz_clear") or facts.get("capability") or "")
    )
    if clear:
        for sent in re.split(r"[。！？；;\n]", clear):
            if _BIZ_GATE_PRIORITY_RE.search(sent or ""):
                _consider(sent, source_bonus=0)

    candidates.sort(key=lambda x: (-x[0], -len(x[1])))
    strong = [(sc, s) for sc, s in candidates if sc > 0]
    pool = strong if strong else candidates

    picked: list[str] = []
    themes: set[str] = set()
    # Pass 1: diversified themes
    for sc, s in pool:
        theme = _gate_theme(s)
        if theme in themes and theme != "其他":
            continue
        themes.add(theme)
        picked.append(s)
        if len(picked) >= max_n:
            return picked
    # Pass 2: fill remaining slots by score
    for sc, s in pool:
        if s in picked:
            continue
        picked.append(s)
        if len(picked) >= max_n:
            break
    if not picked:
        sent = _first_biz_sentence(
            str(facts.get("biz_clear") or facts.get("capability") or ""),
            max_len=110,
        )
        if sent:
            picked = [sent]
    return picked[:max_n]


def _spine_method_name(cm: str) -> str:
    return cm.split("#", 1)[-1].split("(", 1)[0].strip() if "#" in cm else cm


def _pick_nested_gate_cms(closure: list[str]) -> list[tuple[str, str]]:
    """Map CALLS closure → ordered (label, class_method) for nested loan gates.

    Prefers exact stem matches (``quotaCheck`` over ``quotaCheckMIY``) so the
    sync-flow outline names the gate; dig-in Q&A can still expand product branches.
    """
    if not closure:
        return []
    by_meth: dict[str, str] = {}
    for cm in closure:
        if not cm or "#" not in cm:
            continue
        meth = _spine_method_name(cm)
        if not meth:
            continue
        key = meth.lower()
        # Keep first (shallower BFS) hit per method name.
        by_meth.setdefault(key, cm)

    picked: list[tuple[str, str]] = []
    seen_cm: set[str] = set()
    for label, stems in _NESTED_FLOW_GATE_SPECS:
        hit: str | None = None
        # Exact stem first, then prefix (e.g. quotaCheck* only if exact missing).
        for stem in stems:
            if stem in by_meth:
                hit = by_meth[stem]
                break
        if hit is None:
            for stem in stems:
                for meth_l, cm in by_meth.items():
                    if meth_l.startswith(stem) and meth_l != stem:
                        # Skip product dig-in variants when parent exists later —
                        # only accept prefix if no exact match in any stem.
                        hit = cm
                        break
                if hit:
                    break
        if not hit or hit in seen_cm:
            continue
        seen_cm.add(hit)
        picked.append((label, hit))
    return picked


def _format_gate_line(label: str, bullets: list[str], *, max_join: int = 180) -> str:
    if not bullets:
        return f"- {label}。"
    if len(bullets) == 1:
        return f"- {label}：{bullets[0]}"
    joined = "；".join(b.rstrip("。；;") for b in bullets) + "。"
    if len(joined) > max_join:
        joined = "；".join(b.rstrip("。；;") for b in bullets[:3]) + "。"
    return f"- {label}：{joined}"


def _assemble_answer_from_symbol_descs(
    project_id: str,
    entry_cm: str,
    *,
    full: bool = False,
    flow_key: str | None = None,
) -> str | None:
    """Direct business answer from MySQL symbol_desc (entry + FOLLOWS + nested CALLS).

    Entry overview from cleaned summary_business_*; along FOLLOWS highlight
    business gates from business_view.rules / validation_points / biz_clear.
    Also surfaces nested CALLS gates (query / check / route…) that live
    under umbrella ServiceImpl methods and are missing from the FOLLOWS L1 spine.
    Demotes pure DTO field-null checklists. No polish LLM, no status_line dumps.
    """
    entry_cm = (entry_cm or "").strip()
    if not project_id or not entry_cm or "#" not in entry_cm:
        return None
    facts = _load_deep_facts_by_class_methods(project_id, [entry_cm])
    entry = facts.get(entry_cm) or {}
    body = _entry_business_overview(entry)
    if len(body) < 24:
        return None

    paths = list(entry.get("http_paths") or [])
    lines: list[str] = []
    if paths:
        lines.append(f"从渠道请求 `{paths[0]}` 发起申请后，系统大致这样走：")
    else:
        lines.append("从本接口受理申请后，系统大致这样走：")
    lines.append("")

    overview = body.strip()
    # Soften dual-label noise for reading aloud.
    overview = re.sub(r"【业务[·・.]?精简】\s*", "", overview)
    overview = re.sub(r"【研发[·・.].*?】\s*", "", overview)
    if overview:
        lines.append(overview)
        lines.append("")

    # FOLLOWS-ordered spine + nested CALLS business gates.
    fk = (flow_key or "").strip()
    ordered: list[str] = []
    try:
        from app.analysis.flow_neo4j_sync import get_follows_path

        follows = get_follows_path(project_id, entry_cm, fk)
        if follows.get("ok"):
            ordered = [
                str(st.get("class_method") or "").strip()
                for st in (follows.get("steps") or [])
                if isinstance(st, dict)
            ]
            ordered = [c for c in ordered if c and "#" in c]
    except Exception:
        logger.debug("symbol-desc assemble FOLLOWS lookup failed", exc_info=True)
        ordered = []

    entry_simple = entry_cm.split("#", 1)[0].split(".")[-1]
    entry_method = entry_cm.split("#", 1)[1]
    downstream = [
        c
        for c in ordered
        if not (
            c.split("#", 1)[0].split(".")[-1] == entry_simple
            and c.split("#", 1)[1].split("(", 1)[0] == entry_method
        )
    ]

    # Nested gates under entry CALLS (check / query / route helpers).
    closure: list[str] = []
    try:
        closure = _collect_entry_calls_closure(project_id, entry_cm, depth=4, limit=160)
    except Exception:
        logger.debug("symbol-desc assemble CALLS closure failed", exc_info=True)
        closure = []
    nested_specs = _pick_nested_gate_cms(closure)
    nested_cms = [cm for _, cm in nested_specs]

    more: dict[str, dict[str, Any]] = {}
    load_cms = list(dict.fromkeys([*downstream, *nested_cms]))
    if load_cms:
        more = _load_deep_facts_by_class_methods(project_id, load_cms)

    gate_lines: list[str] = []
    field_note: str | None = None
    plumbing_lines: list[str] = []
    nested_inserted = False
    loan_summary: str | None = None

    def _emit_nested_gates() -> None:
        nonlocal nested_inserted
        if nested_inserted or not nested_specs:
            return
        nested_inserted = True
        for label, cm in nested_specs:
            d = more.get(cm) or more.get(_simple_class_method(cm)) or {}
            bullets = _pick_gate_bullets(d, max_n=2, allow_field_null=False)
            if not bullets:
                sent = _first_biz_sentence(
                    str(d.get("biz_clear") or d.get("capability") or ""),
                    max_len=100,
                )
                if sent:
                    bullets = [sent]
            if not bullets:
                # Evidence-safe fallback: method name from CALLS is the anchor.
                meth = _spine_method_name(cm)
                bullets = [f"主链调用 `{_simple_class_method(cm) or meth}` 完成该关卡"]
            gate_lines.append(_format_gate_line(label, bullets))

    if downstream or nested_specs:
        for cm in downstream:
            d = more.get(cm) or {}
            method = _spine_method_name(cm)
            label = _SPINE_BIZ_LABEL.get(cm) or _SPINE_BIZ_LABEL.get(
                _simple_class_method(cm) or ""
            ) or method
            if method in _FIELD_CHECKLIST_METHODS:
                if field_note is None:
                    field_note = (
                        f"- {label}：校验关键入参非空与格式"
                        f"（字段级清单从略，除非追问「校验了哪些字段」）。"
                    )
                continue
            if method == "loan":
                # Expand nested CALLS gates here instead of one thin loan line.
                _emit_nested_gates()
                bullets = _pick_gate_bullets(d, max_n=3, allow_field_null=False)
                # Drop bullets already covered by named nested gates.
                if nested_inserted and bullets:
                    covered = re.compile(r"(?:校验|签约|查询|产品)")
                    bullets = [b for b in bullets if not covered.search(b)]
                if bullets:
                    loan_summary = _format_gate_line("主处理", bullets[:2])
                elif not nested_inserted:
                    loan_summary = "- 主处理：编排校验/审批等关卡并推进流程。"
                continue
            max_n = 1 if method in _PLUMBING_METHODS else 2
            bullets = _pick_gate_bullets(d, max_n=max_n)
            if not bullets:
                sent = _first_biz_sentence(
                    str(d.get("biz_clear") or d.get("capability") or ""),
                    max_len=100,
                )
                if sent:
                    bullets = [sent]
            if not bullets:
                continue
            line = _format_gate_line(label, bullets)
            if method in _PLUMBING_METHODS:
                if len(plumbing_lines) < 2:
                    plumbing_lines.append(line)
                continue
            gate_lines.append(line)
            # After loanSwitch / apply path, still allow nested insert before loan.
            if method == "loanApply" and not nested_inserted and "loan" not in {
                _spine_method_name(c) for c in downstream
            }:
                _emit_nested_gates()
            if len(gate_lines) >= 10 and nested_inserted:
                break

        # FOLLOWS missing loan (or empty spine) — still surface nested CALLS gates.
        _emit_nested_gates()
        if loan_summary:
            gate_lines.append(loan_summary)
        gate_lines.extend(plumbing_lines)
        if field_note:
            gate_lines.append(field_note)
        if gate_lines:
            lines.append("关键业务关卡：")
            lines.extend(gate_lines)

    if full:
        lines.append("")
        lines.append(_default_business_callback_section())

    assembled = "\n".join(lines)
    # Evidence whitelist = symbol_desc + nested gate facts — no invented tables.
    ev_bag = _facts_map_evidence_text({**facts, **more})
    if nested_cms:
        ev_bag = (ev_bag + "\n" + "\n".join(nested_cms)).strip()
    return _sanitize_answer(assembled, full_flow=full, evidence_text=ev_bag) or assembled


def _load_flow_status_nodes(project_id: str, flow_key: str, *, limit: int = 12) -> list[dict[str, Any]]:
    """Important deep-parse nodes under flow_key that carry status / db status writes."""
    from app.core.utils import truncate

    if not project_id or not flow_key:
        return []
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session
        from app.models.orm import SymbolDescORM
    except Exception:
        return []
    rows_out: list[dict[str, Any]] = []
    try:
        with get_session() as session:
            descs = session.scalars(
                select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
            ).all()
            for d in descs:
                extra = d.extra_json if isinstance(d.extra_json, dict) else {}
                keys = extra.get("flow_keys") or []
                if isinstance(keys, list) and flow_key not in [str(x) for x in keys]:
                    # also allow business_domain match
                    if str(d.business_domain or "") != flow_key:
                        continue
                cm = str(extra.get("class_method") or "")
                if not cm or "#" not in cm:
                    continue
                status = extra.get("status_transitions") or []
                if not isinstance(status, list) or not status:
                    iv = extra.get("impl_view") if isinstance(extra.get("impl_view"), dict) else {}
                    has_status_db = False
                    for op in iv.get("db_ops") or []:
                        if not isinstance(op, dict):
                            continue
                        fs = op.get("field_sources") or {}
                        if isinstance(fs, dict) and any(
                            re.search(r"status|state", str(k), re.I) for k in fs
                        ):
                            has_status_db = True
                            break
                    if not has_status_db:
                        continue
                    status = [
                        f"{k}={v}"
                        for op in (iv.get("db_ops") or [])
                        if isinstance(op, dict)
                        for k, v in ((op.get("field_sources") or {}).items() if isinstance(op.get("field_sources"), dict) else [])
                        if re.search(r"status|state", str(k), re.I)
                    ]
                rows_out.append(
                    {
                        "class_method": cm,
                        "status_line": truncate("；".join(str(x) for x in status[:8]), 360),
                        "dev_clear": truncate(str(d.summary_dev_clear or ""), 200),
                    }
                )
                if len(rows_out) >= limit:
                    break
    except Exception:
        logger.debug("load flow status nodes failed", exc_info=True)
    return rows_out


def _status_line_belongs_to_other_spine(
    status_line: str,
    cm: str,
    spine_cms: set[str],
) -> bool:
    """True when status text is clearly about another FOLLOWS-stage method.

    Entry WithdrawalsController#withdrawals often stores aggregated
    ``via withdrawalsService.insertLoanData`` facts that belong on later stages.
    """
    text = (status_line or "").strip()
    if not text:
        return False
    self_method = cm.split("#", 1)[-1] if "#" in cm else cm
    for other in spine_cms:
        if not other or other == cm:
            continue
        other_method = other.split("#", 1)[-1]
        if not other_method or other_method == self_method:
            continue
        if re.search(rf"(?i)\bvia\s+\w+\.{re.escape(other_method)}\b", text):
            return True
        if re.search(rf"(?i)\b{re.escape(other)}\b", text):
            return True
    return False


def _enrich_flow_doc_with_deep_nodes(project_id: str, flow: dict[str, Any]) -> dict[str, Any]:
    """Attach MySQL deep-parse facts onto FlowDoc stages for accurate flow answers."""
    import copy

    enriched = copy.deepcopy(flow)
    fd = enriched.get("flow_doc") if isinstance(enriched.get("flow_doc"), dict) else {}
    stages = list(fd.get("stages") or [])
    cms: list[str] = []
    entry = str(enriched.get("class_method") or "")
    if entry:
        cms.append(entry)
    for st in stages:
        if isinstance(st, dict) and st.get("class_method"):
            cms.append(str(st.get("class_method")))
    spine_cms = {str(c).strip() for c in cms if c and "#" in str(c)}
    facts = _load_deep_facts_by_class_methods(project_id, cms)
    new_stages: list[dict[str, Any]] = []
    for st in stages:
        if not isinstance(st, dict):
            continue
        row = dict(st)
        cm = str(row.get("class_method") or "")
        key = cm.split(".")[-1] if cm else ""
        deep = dict(facts.get(cm) or facts.get(key) or {})
        if deep:
            status = str(deep.get("status_line") or "").strip()
            if status and _status_line_belongs_to_other_spine(status, cm, spine_cms):
                deep = {**deep, "status_line": ""}
                status = ""
            row["deep"] = deep
            # Prefer dual-summary fields when stage text is thin/generic.
            if deep.get("biz_clear") and len(str(row.get("desc_business") or row.get("desc") or "")) < 48:
                row["desc_business"] = deep["biz_clear"]
            if deep.get("dev_clear"):
                row["desc_dev"] = deep["dev_clear"]
            if status:
                row["status_line"] = status
        new_stages.append(row)
    fd = dict(fd)
    fd["stages"] = new_stages
    enriched["flow_doc"] = fd
    fk = str(enriched.get("flow_key") or fd.get("flow_key") or "")
    status_nodes = _load_flow_status_nodes(project_id, fk) if fk else []
    # Drop nodes already covered as stages with status
    covered = {
        str(st.get("class_method") or "").split(".")[-1]
        for st in new_stages
        if st.get("status_line") or (st.get("deep") or {}).get("status_line")
    }
    # Prefer status nodes on/near the FOLLOWS spine; drop foreign Controllers.
    spine_simple = {c.split("#", 1)[0].split(".")[-1].lower() for c in spine_cms}
    spine_toks: set[str] = set()
    for c in spine_cms:
        cls = str(c).split("#", 1)[0].split(".")[-1]
        spine_toks.update(
            p.lower()
            for p in re.findall(r"[A-Z][a-z]{3,}|[A-Z]{2,}(?![a-z])", cls)
        )
    filtered_nodes: list[dict[str, Any]] = []
    for n in status_nodes:
        cm_n = str(n.get("class_method") or "").split(".")[-1]
        if cm_n in covered:
            continue
        simple = cm_n.split("#", 1)[0].lower() if "#" in cm_n else ""
        if spine_simple and simple and simple not in spine_simple:
            cand_toks = {
                p.lower()
                for p in re.findall(r"[A-Z][a-z]{3,}|[A-Z]{2,}(?![a-z])", cm_n)
            }
            if not (spine_toks & cand_toks):
                continue
        filtered_nodes.append(n)
    enriched["deep_status_nodes"] = filtered_nodes[:10]
    return enriched


def _strip_pedantic_notes(text: str) -> str:
    """Drop pedantic correction asides (e.g. 「注意：不是 updateLoanInfo」)."""
    out = text or ""
    for _ in range(4):
        nxt = _PEDANTIC_NOTE_RE.sub("", out)
        if nxt == out:
            break
        out = nxt
    out = re.sub(r"[；;]{2,}", "；", out)
    out = re.sub(r"。{2,}", "。", out)
    return out.strip(" ；;。")


# FOLLOWS spine: keep FlowDoc stage order/titles; do not merge by product method names.
_BIZ_MERGE_GROUPS: list[tuple[tuple[str, ...], str, str]] = []

_CODE_STATUS_SMELL = re.compile(
    r"(?:[A-Za-z][\w]*\.){1,}[A-Za-z_][\w]*|"  # map.RESPONSE_CODE / dmo.loanStatus
    r"\bset[A-Z]\w+\b|"
    r"(?:Common|Response|Route|Ab)?Constants?\b|"
    r"\b\w+Enum\b|"
    r"null\s*→|"
    r"['\"]F['\"]|"
    r"\bvia\s+\w+\.|"
    r"checker|false\s*→\s*true|"
    r"EXPIRE_WHOLE|FROZEN_WHOLE|useStatus|executionStatus|"
    r"RESPONSE_CODE|NO_PUSH|SUCC_CODE",
    re.I,
)


def _stage_method_name(st: dict[str, Any]) -> str:
    cm = str(st.get("class_method") or "").strip()
    if "#" in cm:
        return cm.split("#", 1)[1].split("(", 1)[0].strip()
    return cm.split("(", 1)[0].strip()


def _looks_like_code_status_dump(text: str) -> bool:
    return bool(_CODE_STATUS_SMELL.search(text or ""))


def _biz_group_for_method(method: str) -> tuple[int, str, str] | None:
    m = (method or "").strip()
    if not m:
        return None
    for gi, (methods, title, sentence) in enumerate(_BIZ_MERGE_GROUPS):
        if m in methods:
            return gi, title, sentence
    return None


def _condense_business_stages(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge FOLLOWS stages into 4～7 fluent business buckets (order preserved)."""
    raw = [st for st in stages if isinstance(st, dict)]
    if not raw:
        return []
    buckets: list[dict[str, Any]] = []
    for st in raw:
        method = _stage_method_name(st)
        title = str(st.get("title") or "").strip()
        desc = _strip_pedantic_notes(
            str(st.get("desc_business") or st.get("desc") or "").strip()
        )
        tables = _stage_table_facts(f"{title} {desc}")
        ginfo = _biz_group_for_method(method)
        if ginfo and buckets and buckets[-1].get("gid") == ginfo[0]:
            buckets[-1]["methods"].append(method)
            for t in tables:
                if t not in buckets[-1]["tables"]:
                    buckets[-1]["tables"].append(t)
            continue
        if ginfo:
            gid, gtitle, gsentence = ginfo
            buckets.append(
                {
                    "gid": gid,
                    "title": gtitle,
                    "sentence": gsentence,
                    "methods": [method],
                    "tables": list(tables),
                }
            )
            continue
        # Unknown stage: keep as its own bucket (title-first).
        sentence = desc
        if not sentence or _looks_like_code_status_dump(sentence) or sentence == title:
            sentence = f"{title}。" if title else "继续主链处理。"
        elif len(sentence) > 72:
            sentence = sentence[:70].rstrip("，,、；; ") + "。"
        elif not sentence.endswith(("。", "！", "？")):
            sentence = sentence.rstrip("；;，,") + "。"
        buckets.append(
            {
                "gid": None,
                "title": title or method or "业务环节",
                "sentence": sentence,
                "methods": [method] if method else [],
                "tables": list(tables),
            }
        )

    # If still too long, merge adjacent unknown/thin buckets until ≤7.
    while len(buckets) > 7:
        # Merge the two shortest adjacent pair by title length.
        best_i = 0
        best_score = 10**9
        for i in range(len(buckets) - 1):
            score = len(buckets[i]["title"]) + len(buckets[i + 1]["title"])
            if score < best_score:
                best_score = score
                best_i = i
        a, b = buckets[best_i], buckets[best_i + 1]
        merged = {
            "gid": a.get("gid"),
            "title": f"{a['title']}与{b['title']}",
            "sentence": f"{a['sentence'].rstrip('。')}；{b['sentence']}",
            "methods": list(a.get("methods") or []) + list(b.get("methods") or []),
            "tables": list(
                dict.fromkeys(list(a.get("tables") or []) + list(b.get("tables") or []))
            ),
        }
        buckets = buckets[:best_i] + [merged] + buckets[best_i + 2 :]
    return buckets


def _default_business_callback_section() -> str:
    return (
        "结果回调/通知：\n"
        "以下为主入口同步处理之后的下游回传，不是本接口当场返回给渠道的最终结果。\n"
        "- 主处理完成后，由结果通知把成功或失败回传给上游渠道；\n"
        "- 失败时会回写主表失败状态，并按配置触发相应通知；\n"
        "- 占用/支用等随主流程成败一并落账，不在同步主链逐步展开。"
    )


def _callback_is_code_laundry(text: str) -> bool:
    """True when 结果回调 section dumps deep-parse failure branches / code facts."""
    body = text or ""
    smells = len(_CODE_STATUS_SMELL.findall(body))
    bullets = len(re.findall(r"(?m)^\s*[-•·]", body))
    return smells >= 3 or (bullets >= 6 and smells >= 1) or "状态变更要点" in body


def _render_business_flow_narrative(flow: dict[str, Any]) -> str:
    """User-facing business answer: overall → condensed 4～7 points (FOLLOWS-grounded).

    Does NOT dump raw status_line / Class#method / deep-parse laundry lists.
    """
    fd = flow.get("flow_doc") or {}
    stages = [st for st in (fd.get("stages") or []) if isinstance(st, dict)]
    paths = flow.get("http_paths") or []
    path = str(paths[0]).strip() if paths else ""
    is_result = bool(flow.get("is_result_entry"))
    buckets = _condense_business_stages(stages)
    if not buckets:
        return ""

    entry_bit = f"HTTP `{path}`" if path else "本业务入口"
    if is_result:
        overview = (
            f"该入口（{entry_bit}）本身是结果回调/通知侧，用于接收下游回传并落最终结果；"
            f"若要问同步申请主链，请换用申请/提交类入口。"
        )
    else:
        arc = " → ".join(str(b["title"]) for b in buckets)
        overview = (
            f"本接口（{entry_bit}）同步受理渠道发起的申请："
            f"从入口受理，经校验与开关、流程决策、申请落库，进入主处理，"
            f"并在事后通过结果通知把成功或失败回传给上游。"
            f"本接口的同步返回不等于最终处理完成；最终结果以结果回调/通知为准。\n"
            f"业务主线可概括为：{arc}。"
        )

    lines: list[str] = ["总体说明：", overview, "", "精简要点："]
    for i, b in enumerate(buckets, 1):
        title = str(b.get("title") or f"环节{i}")
        sentence = str(b.get("sentence") or "").strip()
        tables = [t for t in (b.get("tables") or []) if t and t not in title]
        if tables:
            title = f"{title}（{'/'.join(tables)}）"
        if sentence:
            lines.append(f"{i}. {title} — {sentence}")
        else:
            lines.append(f"{i}. {title}")
    return "\n".join(lines)


def _render_ordered_main_chain(
    flow: dict[str, Any],
    *,
    tech: bool = True,
    deep: bool = False,
) -> str:
    """Deterministic main-chain text from FlowDoc stages.

    tech=True: keep Class#method + fuller desc (only when user asks tech detail).
    tech=False: prefer overall→condensed business narrative (not forced 1..N FOLLOWS dump).
    deep=True: attach MySQL status facts under steps in tech mode only.
    """
    # Business mode: overall summary + condensed buckets; FOLLOWS used only for grounding.
    if not tech:
        return _render_business_flow_narrative(flow)

    fd = flow.get("flow_doc") or {}
    stages = fd.get("stages") or []
    paths = flow.get("http_paths") or []
    path = paths[0] if paths else ""
    entry = flow.get("class_method") or ""
    is_result = bool(flow.get("is_result_entry"))
    lines: list[str] = []
    if path or entry:
        head = "业务入口："
        if path:
            head += f"HTTP `{path}`"
        if entry:
            head += f"（{entry}）" if path else entry
        if is_result:
            head += " — 结果回调/通知入口（若你要的是同步业务主链，请换用申请/提交类入口再问）。"
        else:
            head += " — 本题同步业务主链起点；结果回调见文末单独一节。"
        lines.append(head)
    lines.append("主链（按入口代码调用顺序）：")
    for i, st in enumerate(stages, 1):
        if not isinstance(st, dict):
            continue
        step = st.get("step") or i
        cm = str(st.get("class_method") or "").strip()
        title = str(st.get("title") or "").strip()
        desc = str(
            st.get("desc_business") or st.get("desc") or st.get("desc_dev") or ""
        ).strip()
        status_line = str(
            st.get("status_line") or (st.get("deep") or {}).get("status_line") or ""
        ).strip()
        dev_clear = str(
            st.get("desc_dev") or (st.get("deep") or {}).get("dev_clear") or ""
        ).strip()
        # tech: keep Class#method + concrete facts
        if desc.startswith(cm + " ") or desc.startswith(cm + "：") or desc.startswith(cm + ":"):
            body = desc
        elif cm and title and title not in cm:
            body = f"{cm} — {title}" + (f"：{desc}" if desc and desc != title else "")
        elif cm:
            body = f"{cm}" + (f"：{desc}" if desc else "")
        else:
            body = (title or f"步骤{step}") + (f"：{desc}" if desc else "")
        lines.append(f"{step}. {body}")
        extras: list[str] = []
        if deep and status_line:
            extras.append(f"状态：{status_line}")
        if deep and dev_clear and dev_clear not in body:
            extras.append(f"研发：{truncate(dev_clear, 160)}")
        for ex in extras:
            lines.append(f"   · {ex}")

    # Extra status nodes from deep-parse — tech mode only (business hides laundry lists).
    extra_nodes = flow.get("deep_status_nodes") or []
    if deep and extra_nodes:
        lines.append("状态变更要点（来自深析节点）：")
        for n in extra_nodes[:10]:
            if not isinstance(n, dict):
                continue
            cm = str(n.get("class_method") or "")
            st_line = str(n.get("status_line") or "")
            if not st_line:
                continue
            if cm:
                lines.append(f"- {cm}：{st_line}")
            else:
                lines.append(f"- {st_line}")
    return "\n".join(lines)


def _collect_flow_downstream_apis(
    flow: dict[str, Any],
) -> tuple[str, list[tuple[str, str]]]:
    """Return (entry Class#method, unique downstream Class#method + note) from FlowDoc stages."""
    fd = flow.get("flow_doc") or {}
    stages = fd.get("stages") or []
    entry = str(flow.get("class_method") or "").strip()
    entry_simple = ""
    entry_method = ""
    if "#" in entry:
        entry_simple, entry_method = entry.split("#", 1)
        entry_simple = entry_simple.split(".")[-1]
        entry_method = entry_method.split("(", 1)[0].strip()

    items: list[tuple[str, str]] = []
    seen: set[str] = set()
    for st in stages:
        if not isinstance(st, dict):
            continue
        cm = str(st.get("class_method") or "").strip()
        if not cm or "#" not in cm:
            continue
        c_simple, c_method = cm.split("#", 1)
        c_simple = c_simple.split(".")[-1]
        c_method = c_method.split("(", 1)[0].strip()
        if not c_simple or not c_method:
            continue
        # Skip the locked entry itself.
        if entry_simple and c_simple == entry_simple and c_method == entry_method:
            continue
        # Never treat other HTTP Controller/Resource entries as "downstream"
        # (siblings OR unrelated Amount/EntrustedRefund Controllers).
        if "Controller" in c_simple or c_simple.endswith("Resource"):
            continue
        key = f"{c_simple}#{c_method}".lower()
        if key in seen:
            continue
        seen.add(key)
        title = str(st.get("title") or "").strip()
        desc = str(st.get("desc") or "").strip()
        note = title if title and title != cm and title != c_method else (desc[:80] if desc else "")
        if note.startswith(cm):
            note = note[len(cm) :].lstrip(" ：:-—")
        items.append((f"{c_simple}#{c_method}", note))
    return entry, items


def _render_flow_downstream_apis(flow: dict[str, Any]) -> str:
    """List unique downstream Class#method from locked FlowDoc stages (not sibling Controllers).

    Count in the header always matches the number of listed items.
    """
    entry, items = _collect_flow_downstream_apis(flow)
    n = len(items)
    lines: list[str] = []
    if entry:
        lines.append(f"主链入口：{entry}")
    lines.append(f"过程中调用的下游接口/服务（共 {n} 个）：")
    if not items:
        lines.append("（主链阶段中未见独立下游 Class#method；可继续追问具体 Feign/MQ。）")
        return "\n".join(lines)
    for i, (cm, note) in enumerate(items, 1):
        if note:
            lines.append(f"{i}. {cm} — {note}")
        else:
            lines.append(f"{i}. {cm}")
    return "\n".join(lines)


def _flow_doc_stage_count(flow: dict[str, Any] | None) -> int:
    if not flow:
        return 0
    stages = (flow.get("flow_doc") or {}).get("stages") or []
    return sum(1 for st in stages if isinstance(st, dict))


def _count_numbered_steps(text: str) -> int:
    """Count top-level numbered steps (1. / 1、 / 1))."""
    return len(re.findall(r"(?m)^\s*\d+\s*[\.、\)]\s*\S", text or ""))


def _answer_collapsed_main_chain(answer: str, stage_count: int) -> bool:
    """True when LLM paraphrase dropped most FlowDoc stages (e.g. 8 → 4)."""
    if stage_count < 3:
        return False
    steps = _count_numbered_steps(answer)
    # Allow missing at most one stage; 4-of-8 style summaries are collapsed.
    return steps < max(3, stage_count - 1)


# Field-checklist drift: LLM expands checkApplyReqParam into the whole「主链」.
_VALIDATION_CHECKLIST_MARKERS = re.compile(
    r"必填|不能为空|入参校验|字段校验|参数校验|校验(?:了)?(?:哪些)?字段|"
    r"校验[\u4e00-\u9fff]{2,12}"
)
# Real sync-entry FlowDoc stages mention apply/persist/switch — not only fields.
_FLOW_BUSINESS_STAGE_MARKERS = re.compile(
    r"loanApply|insertLoanData|updateReult|insertOrUpdate|loanSwitch|"
    r"ProcessConfig|loanDecision|写(?:入|库)|落库|主表|申请表|扩展表|"
    r"(?!x)x",
    re.I,
)


def _flow_doc_stages_blob(flow: dict[str, Any] | None) -> str:
    if not flow:
        return ""
    fd = flow.get("flow_doc") or {}
    parts: list[str] = [str(fd.get("narrative") or ""), str(fd.get("title") or "")]
    for st in fd.get("stages") or []:
        if isinstance(st, dict):
            parts.append(
                f"{st.get('class_method') or ''} {st.get('title') or ''} {st.get('desc') or ''}"
            )
    return "\n".join(parts)


def _answer_is_validation_only_collapse(
    answer: str, flow: dict[str, Any] | None
) -> bool:
    """True when answer is a field-validation checklist but FlowDoc has business stages."""
    stages_blob = _flow_doc_stages_blob(flow)
    if not stages_blob or not _FLOW_BUSINESS_STAGE_MARKERS.search(stages_blob):
        return False
    text = answer or ""
    # Kept real stage anchors → not a validation-only rewrite.
    if _FLOW_BUSINESS_STAGE_MARKERS.search(text):
        return False
    val_hits = len(_VALIDATION_CHECKLIST_MARKERS.findall(text))
    return val_hits >= 3


def _flow_answer_collapsed(
    answer: str, flow: dict[str, Any] | None, stage_count: int
) -> bool:
    """Count collapse OR validation-checklist rewrite of a business FlowDoc."""
    if _answer_collapsed_main_chain(answer, stage_count):
        return True
    return _answer_is_validation_only_collapse(answer, flow)


def _append_tech_anchor(text: str, flow: dict[str, Any] | None) -> str:
    if not flow or "技术锚点" in (text or ""):
        return text
    fk = flow.get("flow_key") or ""
    paths = flow.get("http_paths") or []
    cm = flow.get("class_method") or ""
    bits: list[str] = []
    if paths:
        bits.append(f"入口 `{paths[0]}`")
    elif cm:
        bits.append(f"入口 {cm}")
    if fk:
        bits.append(f"flow_key={fk}")
    if not bits:
        return text
    return (text or "").rstrip() + "\n\n技术锚点：" + "；".join(bits)


# Structural table-role phrases (not domain nouns) — polish must keep those present in source.
_TABLE_ROLE_FACTS = re.compile(r"申请表|扩展表|主表|从表|明细表|配置表")


def _stage_table_facts(text: str) -> list[str]:
    seen: list[str] = []
    for m in _TABLE_ROLE_FACTS.finditer(text or ""):
        if m.group(0) not in seen:
            seen.append(m.group(0))
    return seen


def _polish_dropped_table_facts(source: str, polished: str) -> bool:
    """True when polished text dropped a table-role fact that appeared in locked stages."""
    for fact in _stage_table_facts(source):
        if fact not in (polished or ""):
            return True
    return False


def _entry_simple_and_method(entry: str) -> tuple[str, str]:
    entry = (entry or "").strip()
    if not entry:
        return "", ""
    simple = entry.split("#", 1)[0].split(".")[-1]
    method = (
        entry.split("#", 1)[1].split("(", 1)[0].strip() if "#" in entry else ""
    )
    return simple, method


def _flow_answer_wrong_entry(
    answer: str, flow: dict[str, Any] | None, *, tech: bool = True
) -> bool:
    """True when polish drifted off the locked sync entry (path / Class#method / callback-as-entry)."""
    if not flow:
        return False
    text = answer or ""
    paths = flow.get("http_paths") or []
    path = str(paths[0]).strip() if paths else ""
    entry = str(flow.get("class_method") or "").strip()
    simple, method = _entry_simple_and_method(entry)
    if path and path not in text:
        # Business polish may soft-word the entry line; accept if sync entry is still anchored.
        if tech or not (simple in text or (method and method in text)):
            return True
    if not path and entry:
        if simple and simple not in text:
            return True
        if method and method not in text:
            return True
    if not flow.get("is_result_entry"):
        m = re.search(r"业务入口[：:].{0,160}", text)
        if m and _RESULT_ENTRY_MARKERS.search(m.group(0)):
            return True
    return False


def _strip_callback_section(text: str) -> str:
    """Polish stage must not emit 结果回调; drop if model appended one."""
    m = re.search(r"\n\s*(?:结果回调|回调/通知|下游回传)", text or "")
    if m:
        return (text or "")[: m.start()].rstrip()
    return text or ""


def _flow_doc_stage_class_methods(flow: dict[str, Any] | None) -> list[str]:
    """Ordered Class#method anchors from FlowDoc stages (simple Class#method form)."""
    if not flow:
        return []
    fd = flow.get("flow_doc") or {}
    out: list[str] = []
    for st in fd.get("stages") or []:
        if not isinstance(st, dict):
            continue
        cm = str(st.get("class_method") or "").strip()
        if not cm or "#" not in cm:
            continue
        simple = cm.split("#", 1)[0].split(".")[-1]
        method = cm.split("#", 1)[1].split("(", 1)[0].strip()
        if simple and method:
            out.append(f"{simple}#{method}")
    return out


def _polish_dropped_class_methods(
    polished: str, flow: dict[str, Any] | None, *, stage_count: int
) -> bool:
    """True when tech/deep polish lost most FlowDoc Class#method anchors."""
    wanted = _flow_doc_stage_class_methods(flow)
    if len(wanted) < 2:
        return False
    text = polished or ""
    kept = 0
    for cm in wanted:
        simple, method = cm.split("#", 1)
        if cm in text:
            kept += 1
            continue
        # Accept same simple Class#method even if polish dropped package prefix.
        if re.search(
            rf"\b{re.escape(simple)}#{re.escape(method)}\b",
            text,
        ):
            kept += 1
    # Require nearly all stage anchors; generic title rewrites keep ~0.
    min_keep = max(2, len(wanted) - 1) if stage_count >= 3 else 1
    return kept < min_keep


def _polish_generic_stage_rewrite(
    polished: str, flow: dict[str, Any] | None, *, stage_count: int
) -> bool:
    """True when numbered steps look like industry boilerplate, not FlowDoc titles."""
    if not flow or stage_count < 3:
        return False
    fd = flow.get("flow_doc") or {}
    titles = [
        str(st.get("title") or "").strip()
        for st in (fd.get("stages") or [])
        if isinstance(st, dict) and str(st.get("title") or "").strip()
    ]
    if len(titles) < 3:
        return False
    text = polished or ""
    # If Class#methods are present, titles may be lightly reworded — not generic.
    if len(_CLASS_METHOD_ANCHOR.findall(text)) >= max(2, stage_count - 2):
        return False
    title_hits = 0
    for t in titles:
        # Accept partial overlap on core title tokens (len>=2 Chinese / alnum).
        core = re.sub(r"[\s（）()【】\[\]：:、，,。]+", "", t)
        if len(core) >= 4 and core[:4] in text:
            title_hits += 1
        elif len(core) >= 2 and core in text:
            title_hits += 1
    return title_hits < max(2, (len(titles) + 1) // 2)


def _reject_polished_main_reason(
    polished: str,
    *,
    ordered_main: str,
    flow: dict[str, Any] | None,
    stage_count: int,
    tech: bool = True,
    deep: bool = False,
) -> str | None:
    """Return reject reason, or None if polish is acceptable."""
    text = (polished or "").strip()
    if not text:
        return "empty"
    steps = _count_numbered_steps(text)
    if tech:
        # Tech: must not drop FOLLOWS stages; allow at most one extra numbered line.
        if stage_count >= 3 and steps < stage_count:
            return f"stage_count_drop:{steps}<{stage_count}"
        if stage_count >= 3 and steps > stage_count + 1:
            return f"stage_count_inflate:{steps}>{stage_count}"
        if _answer_collapsed_main_chain(text, stage_count):
            return f"collapsed_steps:{steps}/{stage_count}"
        if _polish_dropped_class_methods(text, flow, stage_count=stage_count):
            return "dropped_class_methods"
        if deep and "状态" in (ordered_main or "") and "状态" not in text:
            return "dropped_status_facts"
    else:
        # Business narrative: overall → 4～7 condensed points (not forced 1..N spine).
        if steps and (steps < 3 or steps > 8):
            return f"business_points_out_of_range:{steps}"
        if _looks_like_code_status_dump(text):
            return "code_status_dump"
        if "总体" not in text and "精简" not in text and "要点" not in text:
            # Accept lightly polished variants that still keep numbered points + entry path.
            if steps < 3:
                return "missing_business_structure"
    if _answer_is_validation_only_collapse(text, flow):
        return "validation_only_collapse"
    if _polish_dropped_table_facts(ordered_main, text):
        return "dropped_table_facts"
    if _flow_answer_wrong_entry(text, flow, tech=tech):
        return "wrong_entry"
    # Tech/business: reject industry-boilerplate stage titles unrelated to FlowDoc.
    if tech and _polish_generic_stage_rewrite(text, flow, stage_count=stage_count):
        return "generic_stage_rewrite"
    return None


def _reject_polished_main(
    polished: str,
    *,
    ordered_main: str,
    flow: dict[str, Any] | None,
    stage_count: int,
    tech: bool = True,
    deep: bool = False,
) -> bool:
    """True → discard polish and keep deterministic ordered_main."""
    return (
        _reject_polished_main_reason(
            polished,
            ordered_main=ordered_main,
            flow=flow,
            stage_count=stage_count,
            tech=tech,
            deep=deep,
        )
        is not None
    )


def _main_chain_polish_prompts(
    question: str,
    ordered_main: str,
    *,
    stage_count: int,
    tech: bool,
    soft: bool = False,
) -> tuple[str, str]:
    """System/user prompts for wording-only polish (ordered_main is the sole source)."""
    if tech:
        style = (
            "风格：业务可读，但每步须保留 Class#method"
            "（格式「N. Class#method — 业务说明」）。"
        )
        hard = (
            "你是业务流程文案润色助手。准确优先。只能改写【已定稿主链】的措辞，"
            "不得增删阶段、不得改入口、不得编造事实。\n"
            "硬性规则：\n"
            f"1. 主链编号步骤数量必须仍为 {stage_count} 步（与原文一致，一步都不少）。\n"
            "2. 不得把主链改写成入参字段校验清单；不得引入原文没有的校验字段列表。\n"
            "3. 原文出现的表角色事实（如申请表/扩展表/主表等）必须原词保留，不得丢掉。\n"
            "4. 业务入口行须保留原文 HTTP path（若有）；不得把回调/通知写成主链第 1 步。\n"
            "5. 只用原文已有事实改写；禁止编造表名/字段/接口/外部 URL；禁止行业套话与臆造步骤。\n"
            f"6. {style}\n"
            "7. 保持「业务入口：…」+「主链：」结构；不要输出结果回调节。\n"
            "8. 不要输出思考过程。"
        )
        if soft:
            hard += (
                "\n9. 【重试】上一次润色被拒。请逐字对照原文步骤数与表角色词；"
                "只做轻量通顺改写，宁可不改句子也要保住步骤数与申请表/扩展表/主表。"
            )
        user = (
            f"用户问题: {question}\n\n"
            f"【已定稿主链——唯一事实来源】\n{ordered_main}\n\n"
            "请输出润色后的主链全文（含业务入口与编号主链）。"
        )
        return hard, user

    # Business: overall → condensed points (not forced FOLLOWS 1..N).
    hard = (
        "你是面向业务同学的流程解读润色助手。准确优先。只能改写【已定稿业务说明】的措辞，"
        "不得编造事实、不得改入口 path、不得新增原文没有的表名/字段/接口。\n"
        "硬性规则：\n"
        "1. 结构保持「总体说明」+「精简要点」（约 4～7 条），不要扩成十步源码主链。\n"
        "2. 通顺业务中文，像业务同事口述；一句一事，禁止翻译腔与机械罗列。\n"
        "3. 禁止把变量名/常量名/字段赋值堆进正文"
        "（如 map.RESPONSE_CODE、reqVo.setXxx、checker false→true、null→'F'）。\n"
        "4. 原文出现的表角色词（申请表/扩展表/主表等）须保留；未见具体表名不要举例臆造。\n"
        "5. 不得改写成入参字段校验清单；不得把回调写成第 1 要点。\n"
        "6. 正文不要出现 Class#method，不要写 flow_key=…；不要输出结果回调节。\n"
        "7. 不要输出思考过程。"
    )
    if soft:
        hard += (
            "\n8. 【重试】上一次润色被拒。请只做轻量通顺改写，保住 path、表角色词与要点条数区间。"
        )
    user = (
        f"用户问题: {question}\n\n"
        f"【已定稿业务说明——唯一事实来源】\n{ordered_main}\n\n"
        "请输出润色后的全文（总体说明 + 精简要点）。"
    )
    return hard, user


async def _polish_ordered_main_chain(
    question: str,
    ordered_main: str,
    flow: dict[str, Any] | None,
    *,
    deep: bool = False,
) -> str:
    """Second-stage wording rewrite of locked FlowDoc main chain; fallback on drift."""
    if not flow:
        return ordered_main
    stage_count = _flow_doc_stage_count(flow)
    if stage_count < 1:
        return ordered_main or ""
    tech = _wants_tech_detail(question)
    # Mode-matched source: tech keeps full FOLLOWS + status; business = narrative.
    source = _render_ordered_main_chain(flow, tech=tech, deep=deep and tech) or ordered_main
    if not source:
        return ordered_main or ""
    fallback = source
    llm = None
    for attempt, soft in enumerate((False, True)):
        sys, user = _main_chain_polish_prompts(
            question, source, stage_count=stage_count, tech=tech, soft=soft
        )
        if deep and tech:
            sys += (
                "\n额外硬性要求：必须保留每步中的 Class#method，以及「状态：」/「状态变更要点」行；"
                "禁止删掉状态字段或写成空泛「更新贷款状态」。"
            )
        try:
            if llm is None:
                llm = get_llm_provider()
            raw = await llm.chat(
                [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user},
                ],
                temperature=0.1 if not soft else 0.0,
                max_tokens=2200,
            )
        except Exception as e:
            logger.warning(
                "main-chain polish LLM failed (attempt=%s tech=%s): %s",
                attempt + 1,
                tech,
                e,
            )
            return fallback
        text = _strip_callback_section((raw or "").strip())
        if not tech:
            text = _strip_pedantic_notes(text)
            text = re.sub(
                r"(?m)^(\s*\d+\s*[\.、\)]\s*)[A-Za-z][\w.]*#[\w]+\s*[—\-–:：]\s*",
                r"\1",
                text,
            )
        reason = _reject_polished_main_reason(
            text,
            ordered_main=source,
            flow=flow,
            stage_count=stage_count,
            tech=tech,
            deep=deep and tech,
        )
        if reason is None:
            return text
        logger.info(
            "main-chain polish rejected (attempt=%s tech=%s deep=%s stages=%s): %s",
            attempt + 1,
            tech,
            deep,
            stage_count,
            reason,
        )
    logger.info(
        "main-chain polish fallback to deterministic render (tech=%s deep=%s stages=%s)",
        tech,
        deep,
        stage_count,
    )
    return fallback


def _offline_corpus_quality(hits: list[Any]) -> dict[str, Any]:
    """Score prefetch/retrieved offline materials for fast-path eligibility.

    After ingest, FlowDoc + important nodes (含回调/Listener) already live in
    Milvus/MySQL — full-chain Qs should answer from that corpus in one LLM call.
    """
    flow_chars = 0
    node_hits = 0
    concrete = 0
    has_callback = False
    flow_bits: list[str] = []
    node_bits: list[str] = []
    callback_bits: list[str] = []
    seen: set[str] = set()
    for h in hits:
        text = _hit_text(h)
        key = (text[:120] if text else "") + str(
            h.get("symbol_id") if isinstance(h, dict) else getattr(h, "symbol_id", "")
        )
        if key in seen:
            continue
        seen.add(key)
        is_cb = bool(
            re.search(
                r"回调|通知|Listener|Listenner|Strategy|Callback|Notice|Notify|"
                r"@RabbitListener|MQ|Consumer",
                text,
                re.I,
            )
        )
        if _is_flow_doc_hit(h):
            flow_chars += len(text.strip())
            flow_bits.append(text)
        elif is_cb:
            has_callback = True
            callback_bits.append(text)
        elif text.strip():
            node_hits += 1
            node_bits.append(text)
        if _CONCRETE_FLOW_MARKERS.search(text):
            concrete += 1
        if is_cb:
            has_callback = True
    parts: list[str] = []
    if flow_bits:
        parts.append("【主流程文档】\n" + "\n".join(flow_bits[:4]))
    if node_bits:
        parts.append("【主链重要节点】\n" + "\n".join(node_bits[:18]))
    if callback_bits:
        parts.append(
            "【下游回调/通知（主入口之后，勿插入主链编号）】\n"
            + "\n".join(callback_bits[:10])
        )
    blob = "\n\n".join(parts)
    usable = len(blob) >= 400 and concrete >= 2 and (flow_chars >= 180 or node_hits >= 3)
    rich = usable and (
        (flow_chars >= 250 and concrete >= 3)
        or (node_hits >= 5 and concrete >= 4)
        or flow_chars >= 500
    )
    return {
        "rich": rich,
        "usable": usable,
        "flow_chars": flow_chars,
        "node_hits": node_hits,
        "concrete": concrete,
        "has_callback": has_callback,
        "blob": blob[:16000],
    }


def _retrieved_has_rich_offline(retrieved: list[dict[str, Any]]) -> bool:
    q = _offline_corpus_quality(retrieved)
    return bool(q.get("rich") or q.get("usable"))


def _eligible_follows_spine_fast_path(
    question: str,
    *,
    prefer_cm: str | None,
    endpoint: bool = False,
    api_inventory: bool = False,
    downstream_apis: bool = False,
    validation: bool = False,
    tech: bool = False,
    notice_entry: bool = False,
    status_update: bool = False,
) -> bool:
    """True when Neo4j FOLLOWS / MySQL FlowDoc offline answer should be preferred.

    Independent of EXPLORE_BACKEND / SEARCH_USE_MILVUS: source-only mode must
    still use the locked spine when graph data exists, otherwise the tool-loop
    LLM collapses ~10 FOLLOWS stages into a shallow 4–5 step rewrite.
    """
    if not prefer_cm or "#" not in prefer_cm:
        return False
    if endpoint or api_inventory or downstream_apis or validation or tech:
        return False
    if notice_entry or status_update:
        return False
    # Developer change-plan asks must read VO/bind/Ext — never spine narrative.
    if wants_add_field_change_plan(question):
        return False
    if not (
        _wants_full_flow(question)
        or (
            _wants_sync_entry_first(question)
            and not _wants_api_inventory(question)
        )
    ):
        return False
    return _entry_aligned_with_question(prefer_cm, question)


def _load_follows_spine_flow(
    project_id: str, prefer_cm: str
) -> dict[str, Any] | None:
    """Load MySQL FlowDoc for ``prefer_cm`` and overlay Neo4j FOLLOWS spine."""
    if not project_id or not prefer_cm or "#" not in prefer_cm:
        return None
    locked = _load_flow_doc_by_class_method(project_id, prefer_cm)
    overlaid = _overlay_follows_on_flow(
        project_id,
        locked,
        entry_cm=prefer_cm,
        flow_key=(locked or {}).get("flow_key") if locked else None,
    )
    flow = overlaid or locked
    if not flow or _flow_doc_stage_count(flow) < 3:
        return None
    flow_cm = str(flow.get("class_method") or prefer_cm)
    prefer_simple = prefer_cm.split("#", 1)[0].split(".")[-1].lower()
    flow_simple = flow_cm.split("#", 1)[0].split(".")[-1].lower()
    if prefer_simple and flow_simple and prefer_simple != flow_simple:
        return None
    return flow


async def _answer_from_offline_corpus(
    question: str,
    *,
    corpus: str,
    full: bool,
    project_id: str | None = None,
    prefer_class_method: str | None = None,
) -> str:
    """One-shot answer from offline materials (no tool loop).

    Business (non-tech) sync/full-flow: prefer locked entry's MySQL ``symbol_desc``
    dual business summaries (+ brief FOLLOWS callee one-liners). No polish essay.
    Tech: keep FOLLOWS numbered chain with Class#method / status facts.
    """
    prefer_cm = prefer_class_method
    if project_id and not prefer_cm:
        anchor = _resolve_flow_entry_anchor(project_id, question)
        prefer_cm = (anchor or {}).get("class_method") if anchor else None
    tech_q = _wants_tech_detail(question)
    sync_flow = _wants_sync_entry_first(question) and not _wants_api_inventory(question)

    # --- Business path: direct symbol_desc (Milvus/entry lock → MySQL summaries) ---
    if project_id and prefer_cm and not tech_q and (full or sync_flow):
        fk = None
        locked = _load_flow_doc_by_class_method(project_id, prefer_cm)
        if locked:
            fk = str(locked.get("flow_key") or "") or None
        direct = _assemble_answer_from_symbol_descs(
            project_id, prefer_cm, full=full, flow_key=fk
        )
        if direct:
            return _sanitize_answer(direct, full_flow=True) or direct

    structured = (
        _pick_structured_flow_doc(
            project_id, question, prefer_class_method=prefer_cm
        )
        if project_id
        else None
    )
    # Prefer Neo4j FOLLOWS spine over raw FlowDoc stage catalog for tech order.
    if project_id:
        entry_for_follows = prefer_cm or (
            str(structured.get("class_method") or "") if structured else ""
        )
        overlaid = _overlay_follows_on_flow(
            project_id,
            structured,
            entry_cm=entry_for_follows or None,
            flow_key=(structured or {}).get("flow_key") if structured else None,
        )
        if overlaid:
            structured = overlaid
    if project_id and structured:
        structured = _enrich_flow_doc_with_deep_nodes(project_id, structured)
    deep_q = (_wants_deep_flow_facts(question) or full) and tech_q
    ordered_main = (
        _render_ordered_main_chain(structured, tech=tech_q, deep=deep_q)
        if structured
        else ""
    )
    stage_count = _flow_doc_stage_count(structured)

    if ordered_main and (full or sync_flow) and tech_q:
        main = await _polish_ordered_main_chain(
            question, ordered_main, structured, deep=deep_q
        )
        if not full:
            text = _append_tech_anchor(main, structured)
            return _sanitize_answer(text, full_flow=True) or text

        llm = get_llm_provider()
        cb_rules = (
            "你是业务流程解读助手。主链已按代码顺序写好，你只能补「结果回调/通知」一节。"
            "规则：\n"
            "1. 不要改写、不要重排主链；不要把回调步骤插进主链编号。\n"
            "2. 回调节开头写明：以下为主入口处理之后的下游回传，不是本接口同步返回。\n"
            "3. 每条回调写「Class#method — 做什么」；材料没有的不要编造。\n"
            "4. 用词只来自用户问题与材料事实，禁止套用行业领域套话。\n"
            "5. 末尾一行技术锚点（可含 flow_key / 入口 path）。"
        )
        user = (
            f"用户问题: {question}\n\n"
            f"【已定稿正文——请原样保留在回答开头】\n{main}\n\n"
            f"【离线材料（仅用于写回调节）】\n{corpus}\n\n"
            "请输出：先原样给出正文，再写「结果回调/通知：」分节，最后技术锚点。"
        )
        raw = await llm.chat(
            [
                {"role": "system", "content": cb_rules},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=2200,
        )
        text = (raw or "").strip()
        drifted = (
            main[:40] not in text
            and "主链" not in text[:120]
            and "业务入口" not in text[:120]
        )
        collapsed = _flow_answer_collapsed(text, structured, stage_count)
        if drifted or collapsed:
            cb = ""
            m = re.search(r"(结果回调|回调/通知|下游回传).*$", text, re.S)
            if m and not _answer_is_validation_only_collapse(m.group(0), structured):
                cb = "\n\n" + m.group(0).strip()
            text = main + cb
        text = _append_tech_anchor(text, structured)
        return _sanitize_answer(text, full_flow=True) or text

    if full:
        if tech_q:
            sys = (
                "你是业务流程解读助手。只能根据【离线材料】整理回答，禁止编造材料没有的步骤。\n"
                "输出版式：\n"
                "业务入口：写清 HTTP path 与 Controller#method"
                "（必须是本题同步业务入口，禁止 Notice/Callback 当入口）\n"
                "主链：\n"
                "1. Class#method — 做什么（按调用顺序，不要乱序拼凑）\n"
                "2. …\n"
                "结果回调/通知：\n"
                "- Class#method — …（主入口之后的下游回传，单独成节）\n"
                "每步写具体事实（校验项/写库/状态）；禁止空模板与空的 **** 占位；"
                "禁止套用任何行业领域套话，用词只来自问题与材料。"
            )
        else:
            sys = (
                "你是业务流程解读助手。只能根据【离线材料】整理回答，禁止编造。\n"
                "优先直接采用材料里入口方法的业务摘要（summary_business_detail/clear），"
                "可补少量下游方法业务一句话；不要改写成十步源码主链，不要堆变量名/status。\n"
                "完整题末尾加「结果回调/通知」业务简述。"
            )
    else:
        sys = (
            "你是业务流程解读助手。只能根据【离线材料】整理回答，禁止编造。"
            + (
                "关键步骤可写 Class#method；"
                if tech_q
                else "优先复述入口方法已有业务摘要；默认不要主动给出类名/方法名；"
            )
            + "禁止套用行业领域套话。"
            "若材料里出现同 Controller 多个 HTTP 方法，只围绕与问题匹配的主入口展开，"
            "禁止写成接口清单。"
        )
    llm = get_llm_provider()
    raw = await llm.chat(
        [
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": (
                    f"用户问题: {question}\n\n【离线材料】\n{corpus}\n\n"
                    "请直接给出业务可读的答案。"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2200,
    )
    return _sanitize_answer(raw, full_flow=full) or raw.strip() or "离线材料不足，未能整理流程。"


def _gate_methods_from_mysql(
    project_id: str,
    themes: list[str],
    *,
    limit: int = 24,
) -> list[str]:
    """Fallback when Neo4j CALLS is empty: theme-matched methods from MySQL."""
    if not project_id or not themes:
        return []
    stems: list[str] = []
    for theme in themes:
        for s in _GATE_THEME_METHOD_STEMS.get(theme) or ():
            if s and s not in stems:
                stems.append(s)
    if not stems:
        return []
    try:
        from sqlalchemy import or_, select

        from app.db.mysql import get_session
        from app.models.orm import SymbolORM
    except Exception:
        return []
    like_clauses = [SymbolORM.method_name.ilike(f"%{s}%") for s in stems[:8]]
    out: list[str] = []
    seen: set[str] = set()
    try:
        with get_session() as session:
            rows = list(
                session.scalars(
                    select(SymbolORM)
                    .where(
                        SymbolORM.project_id == project_id,
                        or_(*like_clauses),
                    )
                    .limit(max(limit * 3, 40))
                )
            )
        for sym in rows:
            simple = (sym.simple_name or "").strip() or (sym.fqcn or "").split(".")[-1]
            meth = (sym.method_name or "").strip()
            if not simple or not meth:
                continue
            cm = f"{simple}#{meth}"
            if cm in seen:
                continue
            if not _method_matches_gate_themes(cm, themes):
                continue
            seen.add(cm)
            out.append(cm)
            if len(out) >= limit:
                break
    except Exception:
        logger.debug("MySQL gate-method fallback failed", exc_info=True)
        return []
    return out


def _rank_gate_callees(
    cms: list[str],
    themes: list[str],
    *,
    question: str = "",
) -> list[str]:
    """Prefer theme-matched primary gate methods over deep plumbing.

    Ranking uses CALLS closure evidence + theme stems only — no product-token
    invent (MIY→quotaCheckMIY) and no hard-coded class allowlist.
    """
    del question  # reserved for future user-token ranking; keep signature stable
    primary_l = {m.lower() for m in _GATE_PRIMARY_METHODS}

    def _score(cm: str) -> tuple[int, int, int, str]:
        meth = cm.split("#", 1)[-1].split("(", 1)[0]
        meth_l = meth.lower()
        # Exact primary name first; then prefix (quotaCheckCKHB01 under CALLS).
        if meth_l in primary_l:
            primary = 0
        elif any(meth_l.startswith(p) for p in primary_l):
            primary = 1
        else:
            primary = 2
        cls = cm.split("#", 1)[0].split(".")[-1].lower()
        # Prefer any ServiceImpl in the locked-entry CALLS closure.
        if cls.endswith("serviceimpl"):
            impl_bonus = 0
        elif "client" in cls or "mapper" in cls or "util" in cls or "mq" in cls:
            impl_bonus = 3
        else:
            impl_bonus = 1
        depth_hint = len(meth)
        return (primary, impl_bonus, depth_hint, cm)

    matched = [cm for cm in cms if _method_matches_gate_themes(cm, themes)]
    if not matched:
        matched = [cm for cm in cms if cm and "#" in str(cm)]
    matched.sort(key=_score)
    out: list[str] = []
    seen: set[str] = set()
    for cm in matched:
        if cm in seen:
            continue
        seen.add(cm)
        out.append(cm)
    return out


def _code_bullets_for_gate_method(project_id: str, cm: str, *, max_n: int = 4) -> list[str]:
    """Light facts from raw_code when symbol_desc bullets are thin."""
    if not project_id or not cm or "#" not in cm:
        return []
    simple, meth = cm.split("#", 1)
    simple = simple.split(".")[-1]
    meth = meth.split("(", 1)[0]
    code = ""
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session
        from app.models.orm import SymbolORM

        with get_session() as session:
            row = session.scalars(
                select(SymbolORM)
                .where(
                    SymbolORM.project_id == project_id,
                    SymbolORM.simple_name == simple,
                    SymbolORM.method_name == meth,
                )
                .limit(1)
            ).first()
            if row and row.raw_code:
                code = str(row.raw_code)
    except Exception:
        code = ""
    if not code:
        try:
            from app.agent.explore_tools import get_symbol_code

            got = get_symbol_code(
                project_id,
                fqcn=simple,
                method_name=meth,
                max_chars=8000,
            )
            if got.get("ok"):
                code = str(got.get("raw_code") or "")
        except Exception:
            return []
    if not code or len(code) < 40:
        return []
    bullets: list[str] = []
    # External client / service calls — quote identifiers only (no product narrative).
    for m in re.finditer(
        r"\b([A-Za-z]\w*(?:Client|Service))\s*\.\s*([A-Za-z_]\w*)\s*\(",
        code,
    ):
        line = f"调用 `{m.group(1)}.{m.group(2)}`"
        if line not in bullets:
            bullets.append(line)
        if len(bullets) >= max_n:
            break
    # Failure / business messages present in source.
    for m in re.finditer(
        r'(?:setResponseMsg|getInnerMsg)\([^)]*["\']([^"\']{6,60})["\']',
        code,
    ):
        msg = m.group(1).strip()
        if msg:
            line = f"失败时提示：{msg}"
            if line not in bullets:
                bullets.append(line)
        if len(bullets) >= max_n:
            break
    # Field accessors literally present in the method body (no product field list).
    for m in re.finditer(
        r"\b(?:get|set|is)([A-Z][A-Za-z0-9_]{2,40})\s*\(",
        code,
    ):
        prop = m.group(1)
        label = prop[:1].lower() + prop[1:]
        line = f"源码中使用 `{label}`"
        if line not in bullets:
            bullets.append(line)
        if len(bullets) >= max_n:
            break
    out: list[str] = []
    seen: set[str] = set()
    for b in bullets:
        key = re.sub(r"\s+", "", b)
        if key in seen:
            continue
        seen.add(key)
        out.append(b)
        if len(out) >= max_n:
            break
    return out
def _assemble_gate_detail_from_calls(
    project_id: str,
    entry_cm: str,
    question: str,
) -> tuple[str | None, list[str]]:
    """Removed business-gate assemble short-circuit — use explore tool loop."""
    del project_id, entry_cm, question
    return None, []
    closure = _collect_entry_calls_closure(project_id, entry, depth=4, limit=160)
    if not closure:
        closure = _gate_methods_from_mysql(project_id, themes)
    ranked = _rank_gate_callees(closure, themes, question=question)
    if not ranked:
        return None, []
    # Cap to top theme-matched methods from CALLS ranking (no product reordering).
    top = ranked[:8]
    symbols = [_simple_class_method(cm) for cm in top if cm and "#" in cm]
    facts_map = _load_deep_facts_by_class_methods(project_id, top)
    evidence_bag_parts: list[str] = [_facts_map_evidence_text(facts_map)]
    theme_label = "/".join(themes[:3])
    qcore = _user_question_core(question) or (question or "")
    existence_ask = bool(
        re.search(r"有没有|有无|做不做|是否做|有没有做", qcore)
    )
    lines: list[str] = []
    if existence_ask:
        named = ", ".join(f"`{_simple_class_method(cm)}`" for cm in top[:4])
        lines.append(
            f"有。入口 `{entry}` 的 CALLS 闭包内存在与「{theme_label}」相关的方法"
            + (f"：{named}" if named else "")
            + "。"
        )
        lines.append("")
    lines.extend(
        [
            f"在入口 `{entry}` 主链里，与「{theme_label}」相关的处理如下"
            f"（依据该入口 CALLS 闭包与已入库业务摘要）：",
            "",
        ]
    )
    used = 0
    for cm in top:
        facts = facts_map.get(cm) or facts_map.get(_simple_class_method(cm)) or {}
        meth = cm.split("#", 1)[-1]
        label = _SPINE_BIZ_LABEL.get(cm) or _SPINE_BIZ_LABEL.get(
            _simple_class_method(cm)
        ) or meth
        bullets = _pick_gate_bullets(facts, max_n=3, allow_field_null=False)
        if not bullets:
            # Fall back to first business sentence from stored desc — quote, don't invent.
            sent = _first_biz_sentence(
                str(facts.get("biz_clear") or facts.get("capability") or ""),
                max_len=120,
            )
            if sent:
                bullets = [sent]
        # Thin narrative → enrich from raw_code identifiers only.
        if len(bullets) < 2:
            code_bits = _code_bullets_for_gate_method(project_id, cm, max_n=4)
            for b in code_bits:
                evidence_bag_parts.append(b)
                if b not in bullets:
                    bullets.append(b)
                if len(bullets) >= 4:
                    break
        if not bullets and not facts:
            continue
        used += 1
        lines.append(f"{used}. `{_simple_class_method(cm)}`（{label}）")
        if bullets:
            for b in bullets[:4]:
                lines.append(f"   - {b.rstrip('。；;')}。")
        else:
            lines.append(
                "   - （已入库摘要较少；未在已读材料中看到更多规则细节，方法名即关卡锚点。）"
            )
        if used >= 5:
            break
    if used < 1:
        # List theme-matched CALLS methods — no invented primary-name fallback.
        lines = [
            f"在入口 `{entry}` 主链 CALLS 闭包内，与「{theme_label}」直接相关的方法：",
            "",
        ]
        for i, cm in enumerate(top[:5], 1):
            lines.append(f"{i}. `{_simple_class_method(cm)}`")
        return "\n".join(lines), symbols
    assembled = "\n".join(lines)
    ev_bag = "\n".join(p for p in evidence_bag_parts if p)
    return (_sanitize_answer(assembled, evidence_text=ev_bag) or assembled), symbols


def _adaptive_max_steps(
    question: str,
    *,
    requested: int | None = None,
) -> int:
    """Clamp explore LLM turns; full-flow no longer inflates to 28+."""
    if _wants_endpoint_lookup(question) or _wants_api_inventory(question):
        cap = _ENDPOINT_MAX_STEPS
    elif wants_add_field_change_plan(question):
        # Need VO + bind + Ext entity + Mapper/MP confirm (multi-file, not 2-read shortcut).
        cap = _ADD_FIELD_MAX_STEPS
    elif _wants_validation_focus(question):
        cap = _VALIDATION_MAX_STEPS
    elif _wants_full_flow(question):
        cap = _FULL_FLOW_MAX_STEPS
    else:
        cap = _DEFAULT_MAX_STEPS
    if requested is None:
        return cap
    # Honor explicit lower budgets; never raise above adaptive cap.
    return max(1, min(int(requested), cap))


_VALIDATION_METHOD = re.compile(
    r"(?i)(check|valid|verify|generate\w*param|fill\w*param|build\w*param|parse\w*param)",
)
# Drop lines that are only a Symbol ID label; strip inline Symbol ID crumbs.
_EXT_CALL_MARKERS = re.compile(
    r"@FeignClient|FeignClient|RestTemplate|WebClient|"
    r"RabbitTemplate|@RabbitListener|@KafkaListener|RocketMQ|"
    r"\b[A-Za-z_]\w*Client\.\w+\(",
    re.I,
)
_MQ_PUBLISH_MARKERS = re.compile(
    r"RabbitTemplate|rabbitMqProducer|\.send\(|convertAndSend|RocketMQTemplate|kafkaTemplate",
    re.I,
)
_LISTENER_READ_MARKERS = re.compile(r"Listener|Listenner|Consumer|@RabbitListener", re.I)
_CLIENT_READ_MARKERS = re.compile(r"Client|Feign", re.I)
_IMPL_MARKERS = re.compile(r"ServiceImpl|ClientService", re.I)
_NOTICE_NAME = re.compile(r"Notice|Callback|Notify", re.I)
_STRATEGY_OR_LISTENER = re.compile(r"Listener|Listenner|Strategy|Consumer", re.I)
_FEIGN_CODE = re.compile(
    r"@FeignClient|(?:name|url|path)\s*=\s*\"[^\"]+\""
    r"|@(?:Post|Get|Put|Delete)Mapping",
    re.I,
)

_ALLOWED_TOOLS = {
    "search_symbols",
    "get_symbol_code",
    "list_callees",
    "list_callers",
    "resolve_impl",
    "discover_add_field_plan",
    "finish",
}


def _normalize_tool_dict(data: dict[str, Any]) -> dict[str, Any] | None:
    if "tool" in data:
        tool = data["tool"]
        if tool not in _ALLOWED_TOOLS:
            return None
        return {"tool": tool, "args": data.get("args") or {}}
    if "name" in data and data["name"] in _ALLOWED_TOOLS:
        return {
            "tool": data["name"],
            "args": data.get("arguments") or data.get("args") or {},
        }
    return None


def _extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse one tool, a tools[] batch, or a top-level JSON array of tools."""
    if not text:
        return []
    m = re.search(r"\{[\s\S]*\}|\[[\s\S]*\]", text)
    if not m:
        return []
    raw = m.group(0)
    try:
        data = json.loads(raw)
    except Exception:
        data = None
        for m2 in re.finditer(r"\{[\s\S]*?\}", text):
            try:
                data = json.loads(m2.group(0))
            except Exception:
                continue
            else:
                break
        if data is None:
            return []

    calls: list[dict[str, Any]] = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                c = _normalize_tool_dict(item)
                if c:
                    calls.append(c)
    elif isinstance(data, dict):
        if isinstance(data.get("tools"), list):
            for item in data["tools"]:
                if isinstance(item, dict):
                    c = _normalize_tool_dict(item)
                    if c:
                        calls.append(c)
        else:
            c = _normalize_tool_dict(data)
            if c:
                calls.append(c)

    # finish must be alone; drop siblings if mixed
    if any(c["tool"] == "finish" for c in calls):
        finish = next(c for c in calls if c["tool"] == "finish")
        return [finish]
    return calls[:_MAX_BATCH_TOOLS]


def _extract_tool_call(text: str) -> dict[str, Any] | None:
    """Back-compat: first tool call only."""
    calls = _extract_tool_calls(text)
    return calls[0] if calls else None


def _compact_observation(obs: dict[str, Any]) -> str:
    cleaned = obs
    if isinstance(obs, dict) and obs.get("plan_markdown"):
        cleaned = {
            "ok": obs.get("ok"),
            "entry": obs.get("entry"),
            "roles": list((obs.get("roles") or {}).keys()),
            "pass_through": obs.get("pass_through"),
            "note": obs.get("note")
            or "finish.answer 请原样采用 plan_markdown",
            "plan_markdown": truncate(str(obs.get("plan_markdown") or ""), 4000),
        }
    elif isinstance(obs, dict) and "hits" in obs:
        cleaned = dict(obs)
        hits = []
        for h in (obs.get("hits") or [])[:6]:
            if not isinstance(h, dict):
                hits.append(h)
                continue
            hits.append(
                {
                    "symbol_id": h.get("symbol_id"),
                    "kind": h.get("kind"),
                    "fqcn": h.get("fqcn"),
                    "method_sig": h.get("method_sig") or h.get("method_name"),
                    "file_path": h.get("file_path"),
                    "summary": truncate(h.get("summary") or "", 120) if h.get("summary") else None,
                    "tags": (h.get("tags") or [])[:4],
                    "score": h.get("score"),
                }
            )
        cleaned["hits"] = hits
        cleaned["note"] = (
            "用 fqcn+method 回答用户；symbol_id 仅作工具参数。"
            "核对入口是否与用户问题字面贴近，再 resolve_impl / 读远程 Client。"
        )
    elif isinstance(obs, dict) and obs.get("raw_code"):
        cleaned = dict(obs)
        cleaned["raw_code"] = truncate(str(obs.get("raw_code") or ""), 2400)
        if isinstance(cleaned.get("neighbors"), list):
            cleaned["neighbors"] = cleaned["neighbors"][:10]
        if isinstance(cleaned.get("impls"), list):
            cleaned["impls"] = cleaned["impls"][:6]
    text = json.dumps(cleaned, ensure_ascii=False, default=str)
    return truncate(text, _OBS_CHARS)


def _trim_messages(messages: list[dict[str, str]]) -> None:
    """In-place: keep system + first user + recent tail to limit context growth."""
    if len(messages) <= 2 + _MSG_TAIL_KEEP:
        return
    head = messages[:2]
    tail = messages[-( _MSG_TAIL_KEEP ):]
    messages[:] = head + tail


def _has_read_code(steps: list[dict[str, Any]]) -> bool:
    return any(s.get("tool") == "get_symbol_code" and s.get("ok") for s in steps)
def _joined_code(retrieved: list[dict[str, Any]]) -> str:
    parts = []
    for h in retrieved:
        if not h:
            continue
        parts.append(h.get("raw_code") or "")
    return "\n".join(parts)


def _joined_fqcns(retrieved: list[dict[str, Any]]) -> str:
    return " ".join(
        f"{(h or {}).get('fqcn') or ''} {(h or {}).get('file_path') or ''}" for h in retrieved
    )


def _read_method_names(retrieved: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for h in retrieved:
        if not h:
            continue
        n = h.get("method_name")
        if n:
            names.add(str(n))
        sig = h.get("method_sig") or ""
        if "#" in sig:
            names.add(sig.split("#")[-1].split("(")[0])
        elif sig:
            names.add(sig.split("(")[0].strip())
    return {n for n in names if n}


def _has_validation_method_read(retrieved: list[dict[str, Any]]) -> bool:
    for h in retrieved:
        if not h:
            continue
        name = f"{h.get('method_name') or ''} {h.get('method_sig') or ''}"
        if _VALIDATION_METHOD.search(name):
            return True
        code = h.get("raw_code") or ""
        # private param assemblers often return false on missing fields
        if code.count("return false") >= 3 and (
            "isNull" in code or "StringUtil" in code or "respMap" in code
        ):
            return True
    return False


def _pending_same_class_helpers(retrieved: list[dict[str, Any]]) -> list[str]:
    """Helpers mentioned in read code but not yet opened."""
    read_names = {n.lower() for n in _read_method_names(retrieved)}
    pending: list[str] = []
    for h in retrieved:
        code = (h or {}).get("raw_code") or ""
        if not code:
            continue
        from app.agent.explore_tools import extract_same_class_calls

        for name in extract_same_class_calls(code, current_method=h.get("method_name")):
            if name.lower() not in read_names and name not in pending:
                pending.append(name)
    return pending[:8]


def _has_feign_facts(retrieved: list[dict[str, Any]]) -> bool:
    code = _joined_code(retrieved)
    if not _FEIGN_CODE.search(code):
        return False
    # Prefer real service coordinates
    return bool(
        re.search(r"name\s*=\s*\"[^\"]+\"", code)
        or re.search(r"url\s*=\s*\"[^\"]+\"", code)
        or re.search(r"path\s*=\s*\"[^\"]+\"", code)
        or "@FeignClient" in code
    )


def _core_chain_ready(
    steps: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    *,
    question: str = "",
) -> bool:
    """True when controller/entry + ServiceImpl (+ async/client for full-flow) exists."""
    if _wants_api_inventory(question):
        # Enough once we have ≥1 Controller method hit for the topic.
        return any(
            (h.get("kind") or "") == "Controller" and (h.get("fqcn") or "")
            for h in retrieved
        ) or any(
            s.get("tool") == "search_symbols" and s.get("ok") and (s.get("hit_count") or 0) > 0
            for s in steps
        )
    if _wants_endpoint_lookup(question):
        return _has_feign_facts(retrieved)
    if wants_extend_field_to_protocol_sync(question):
        return any(
            s.get("tool") == "discover_protocol_sync_plan" and s.get("ok")
            for s in steps
        )
    if wants_add_field_change_plan(question):
        if any(
            s.get("tool") == "discover_add_field_plan" and s.get("ok") for s in steps
        ):
            return True
        return _add_field_evidence_gaps(steps, retrieved) is None
    if _wants_validation_focus(question):
        return _read_count(steps) >= 2 and _has_validation_method_read(retrieved)
    fqcns = _joined_fqcns(retrieved)
    has_impl = bool(_IMPL_MARKERS.search(fqcns))
    if not has_impl or _read_count(steps) < 2:
        return False
    drilled = any(
        s.get("tool") in {"list_callees", "resolve_impl"} and s.get("ok") for s in steps
    )
    if not drilled:
        return False
    if not _wants_full_flow(question):
        return True
    has_async = bool(_STRATEGY_OR_LISTENER.search(fqcns))
    has_client = bool(_CLIENT_READ_MARKERS.search(fqcns))
    code = _joined_code(retrieved)
    # If MQ publish seen, require Listener/Strategy before declaring ready.
    if _MQ_PUBLISH_MARKERS.search(code) and not has_async:
        return False
    # Full-flow early-stop: impl + (listener OR client), ≥3 reads
    return _read_count(steps) >= 3 and (has_async or has_client)


def _locked_entry_parts(prefer_cm: str) -> tuple[str, str]:
    left, _, right = (prefer_cm or "").partition("#")
    cls = left.split(".")[-1].strip()
    meth = right.split("(", 1)[0].strip()
    return cls, meth


def _answer_ignores_prefetch_or_invents_feign(
    answer: str, retrieved: list[dict[str, Any]]
) -> str | None:
    """Reject finish that ignores prefetched ServiceImpl or invents Feign facts."""
    text = answer or ""
    pref_down = [
        h
        for h in (retrieved or [])
        if str(h.get("summary") or "").startswith("预读下游")
        and (h.get("raw_code") or "").strip()
    ]
    for h in pref_down[:4]:
        fqcn = str(h.get("fqcn") or "")
        meth = str(h.get("method_name") or "")
        simple = fqcn.split(".")[-1]
        iface = simple.replace("Impl", "") if simple.endswith("Impl") else simple
        if "ServiceImpl" in simple or "Service" in simple:
            if (
                simple not in text
                and iface not in text
                and (not meth or meth not in text)
            ):
                return (
                    f"已预读下游 {simple}#{meth}，回答未引用该实现。"
                    f"请基于 {simple}#{meth} 源码重写（落库/查询/外部调用以代码为准）。"
                )
    # Invented Feign: answer claims Feign but no Feign facts in evidence
    if re.search(r"FeignClient|@FeignClient|\bFeign\b", text, re.I):
        blob = " ".join(
            f"{h.get('fqcn')} {h.get('raw_code')}" for h in (retrieved or []) if h
        )
        if not re.search(r"@FeignClient|FeignClient", blob, re.I):
            return (
                "回答出现 Feign/FeignClient，但已读代码中没有 @FeignClient。"
                "禁止编造远程调用；请改写已读 ServiceImpl/Mapper 中的真实逻辑。"
            )
    return None


def _answer_violates_locked_entry(answer: str, prefer_cm: str | None) -> str | None:
    """Reject finish text that replaces the locked Controller entry with another."""
    if not prefer_cm or "#" not in prefer_cm:
        return None
    cls, meth = _locked_entry_parts(prefer_cm)
    if not cls or not meth:
        return None
    text = answer or ""
    anchors = re.findall(r"\b([A-Z][\w]*Controller)#(\w+)\b", text)
    lock_tag = f"{cls}#{meth}"
    # Head of answer is where 业务入口 is usually stated.
    head = text[:500]
    head_anchors = re.findall(r"\b([A-Z][\w]*Controller)#(\w+)\b", head)
    for c, m in head_anchors[:3]:
        if c != cls:
            return (
                f"已锁定入口 {lock_tag}，回答入口写成了 {c}#{m}。"
                f"请以 {lock_tag} 为主入口、按已读源码重写，禁止改答其他 Controller。"
            )
    if anchors and lock_tag not in text and meth not in text:
        foreign = [f"{c}#{m}" for c, m in anchors if c != cls]
        if foreign:
            return (
                f"已锁定入口 {lock_tag}，回答未包含该入口却写了 {foreign[0]}。"
                f"请改回 {lock_tag} 并基于其调用链作答。"
            )
    if meth.lower() not in text.lower() and lock_tag not in text:
        return (
            f"已锁定入口 {lock_tag}，回答未点名该入口。"
            "请围绕锁定入口及其 CALLS 闭包作答。"
        )
    return None


_DOWNSTREAM_RECEIVER = re.compile(
    r"(?<![.\w])([a-z]\w{0,40}(?:Service|Facade|Client|Mapper|Fragment|Strategy))\s*\.\s*(\w+)\s*\(",
)


def _guess_types_for_receiver(receiver: str) -> list[str]:
    """Map field name fooService → FooService / FooServiceImpl simple names only."""
    name = (receiver or "").strip()
    if not name:
        return []
    stem = name[0].upper() + name[1:]
    out: list[str] = [stem]
    if stem.endswith("Service") and not stem.endswith("ServiceImpl"):
        out.append(f"{stem}Impl")
    elif stem.endswith("Facade"):
        out.append(f"{stem}Impl")
    return out


def _prefetch_downstream_from_code(
    project_id: str,
    raw_code: str,
    *,
    limit: int = 4,
) -> list[dict[str, Any]]:
    """Read first-level Service/Mapper/Client methods referenced by entry code."""
    from app.agent.explore_tools import get_symbol_code

    hits: list[dict[str, Any]] = []
    seen: set[str] = set()
    for recv, meth in _DOWNSTREAM_RECEIVER.findall(raw_code or ""):
        if meth in {"equals", "toString", "get", "put", "info", "error", "debug", "warn"}:
            continue
        key = f"{recv}#{meth}"
        if key in seen:
            continue
        seen.add(key)
        got = None
        for fqcn in _guess_types_for_receiver(recv):
            got = get_symbol_code(project_id, fqcn=fqcn, method_name=meth, max_chars=10000)
            if got.get("ok") and (got.get("raw_code") or "").strip():
                break
            got = None
        if not got:
            # Fallback: search by method name + receiver stem
            from app.agent.explore_tools import search_symbols

            stem = recv[0].upper() + recv[1:]
            sr = search_symbols(project_id, f"{stem} {meth}", top_k=6)
            for h in sr.get("hits") or []:
                if (h.get("method_name") or "").lower() != meth.lower():
                    continue
                got = get_symbol_code(
                    project_id,
                    symbol_id=h.get("symbol_id"),
                    fqcn=h.get("fqcn"),
                    method_name=h.get("method_name"),
                    max_chars=10000,
                )
                if got.get("ok"):
                    break
                got = None
        if not got or not got.get("ok"):
            continue
        sym = got.get("symbol") or {}
        hits.append(
            {
                "symbol_id": sym.get("symbol_id"),
                "fqcn": sym.get("fqcn"),
                "method_name": sym.get("method_name"),
                "kind": sym.get("kind") or "Service",
                "raw_code": got.get("raw_code"),
                "summary": f"预读下游 {recv}.{meth}",
                "file_path": sym.get("file_path") or "",
            }
        )
        if len(hits) >= limit:
            break
    return hits


def _finish_block_reason(
    steps: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    *,
    question: str = "",
    soft: bool = False,
) -> str | None:
    """Return Chinese reason to reject finish, or None if depth is enough.

    soft=True: after prior depth rejects / near budget, skip optional Client/MQ gates
    when core chain is already ready.
    """
    if _wants_endpoint_lookup(question):
        if _read_count(steps) < 1:
            return "请 search_symbols 找 *Client / Feign，并 get_symbol_code 阅读接口源码。"
        if not _has_feign_facts(retrieved):
            return (
                "尚未读到含 @FeignClient / name / url / path 的 Client 源码。"
                "请 search q=Client 或调用处类名，get_symbol_code 后再 finish；"
                "回答必须写出服务名与接口 path。"
            )
        return None

    if _wants_api_inventory(question):
        has_ctrl = any(
            (h.get("kind") or "") == "Controller" for h in retrieved
        )
        if has_ctrl:
            return None
        searched = any(s.get("tool") == "search_symbols" for s in steps)
        if not searched:
            return "请先 search_symbols(q=问题实词, kind=Controller) 列出相关 HTTP 入口。"
        return (
            "尚未命中 Controller 入口。请用问题实词（去掉「相关/接口/有几个」）重搜，"
            "或 search q=类名补全同 Controller 方法后再 finish。"
        )

    if wants_add_field_change_plan(question):
        return _add_field_evidence_gaps(steps, retrieved, soft=soft)

    if _wants_validation_focus(question):
        if _read_count(steps) < 2:
            return (
                "参数/字段校验问题：请再 get_symbol_code 读入口方法里的本类校验/组装方法"
                "（见 same_class_calls，如 generateReqParam/checkParam）。"
            )
        pending = _pending_same_class_helpers(retrieved)
        if pending and not _has_validation_method_read(retrieved):
            return (
                f"请先 get_symbol_code 阅读本类方法 {pending[0]}（不要 resolve_impl Controller，"
                "也不要跳到无关 Service），再归纳校验字段。"
            )
        if not _has_validation_method_read(retrieved):
            return (
                "尚未读到参数校验/组装方法。请对入口 raw_code 中的 same_class_calls "
                "get_symbol_code 后再 finish。"
            )
        return None

    full = _wants_full_flow(question)
    offline_rich = _retrieved_has_rich_offline(retrieved)
    # Fast finish: offline FlowDoc/重要节点已含具体业务事实
    if (
        offline_rich
        and not _wants_endpoint_lookup(question)
        and not _wants_api_inventory(question)
        and not _wants_validation_focus(question)
        and not wants_add_field_change_plan(question)
    ):
        if full and _wants_callback_section(question):
            blob = " ".join(str(h.get("summary") or "") for h in retrieved)
            if not re.search(
                r"回调|通知|Listener|Listenner|Strategy|MQ|异步|Callback|Notice",
                blob,
                re.I,
            ):
                return (
                    "离线材料缺回调/通知事实：请 search_symbols q=回调/Listener/Notice，"
                    "get_symbol_code 后再 finish。"
                )
        return None

    min_reads = 3 if full else 2
    evidence_reads = max(
        _read_count(steps),
        sum(
            1
            for h in retrieved
            if (h.get("raw_code") or "").strip()
            and str(h.get("summary") or "").startswith("预读")
        ),
    )
    if evidence_reads < min_reads:
        return (
            f"{'完整流程' if full else '流程'}问题：至少成功 get_symbol_code {min_reads} 次，"
            "覆盖入口、ServiceImpl"
            + ("、异步 Listener/策略或远程 Client" if full else "（及必要 Client）")
            + "。"
        )
    drilled = any(
        s.get("tool") in {"list_callees", "resolve_impl"} and s.get("ok") for s in steps
    ) or any(
        str(h.get("summary") or "").startswith("预读下游") for h in retrieved
    )
    if not drilled:
        return "请先 list_callees 或 resolve_impl 下钻，再 finish。"
    fqcns = _joined_fqcns(retrieved)
    if not _IMPL_MARKERS.search(fqcns):
        return (
            "尚未读到 *ServiceImpl / *ClientService。请 resolve_impl 后 "
            "get_symbol_code 读实现类（远程 Client/Feign 通常在实现里）。"
        )
    code = _joined_code(retrieved)
    if full:
        # Notice-only trap: post-hoc notify APIs are not the apply chain.
        controllers = [
            h
            for h in retrieved
            if h
            and (
                (h.get("kind") or "") == "Controller"
                or "Controller" in ((h.get("fqcn") or ""))
            )
        ]
        if controllers and all(_NOTICE_NAME.search(h.get("fqcn") or "") for h in controllers):
            return (
                "当前只读到 Notice/Callback 类入口，这通常是结果回传而非申请主流程。"
                "请按本题实词重搜申请/提交类 Controller，再下钻。"
            )
        has_listener = bool(_LISTENER_READ_MARKERS.search(fqcns)) or bool(
            _STRATEGY_OR_LISTENER.search(fqcns)
        )
        has_client = bool(_CLIENT_READ_MARKERS.search(fqcns))
        has_callback_read = bool(
            re.search(
                r"Callback|CallBack|Notice|Notify|Listener|Listenner",
                fqcns,
                re.I,
            )
        )
        if soft and _core_chain_ready(steps, retrieved, question=question):
            return None
        if _wants_callback_section(question) and not has_callback_read and not soft:
            return (
                "完整链路需覆盖下游回调/通知：请 search_symbols q=回调/Listener/Notice，"
                "get_symbol_code 读回调入口后再 finish。"
            )
        if _MQ_PUBLISH_MARKERS.search(code) and not has_listener:
            if soft and has_client:
                return None
            return (
                "代码已出现 MQ 发送：请 search_symbols/list_callees 找到对应 *Listener*，"
                "get_symbol_code 读消费方法（再到核算/远程 Client），再 finish。"
            )
        if not has_listener and not has_client and _MQ_PUBLISH_MARKERS.search(code):
            return "完整流程需读到异步 Listener 或回调 Strategy，请继续下钻。"
    if soft and _core_chain_ready(steps, retrieved, question=question):
        return None
    if _EXT_CALL_MARKERS.search(code) and not _CLIENT_READ_MARKERS.search(fqcns):
        # Soften: Listener already covers downstream for full-flow MQ paths
        if full and _STRATEGY_OR_LISTENER.search(fqcns):
            return None
        return (
            "已读代码出现 Feign/*Client/远程或消息调用，"
            "请继续 get_symbol_code 读对应 Client/远程接口后再 finish。"
        )
    return None


# Meta / think-out-loud lines that must not reach business readers.
def _absorb_tool_result(
    tool: str,
    obs: dict[str, Any],
    retrieved: list[dict[str, Any]],
) -> None:
    if tool == "search_symbols":
        for h in obs.get("hits") or []:
            retrieved.append(h)
    if tool == "resolve_impl" and obs.get("ok"):
        for h in obs.get("impls") or []:
            retrieved.append(h)
    if tool == "get_symbol_code" and obs.get("ok"):
        sym = dict(obs.get("symbol") or {})
        sym["raw_code"] = obs.get("raw_code") or ""
        retrieved.append(sym)
    if tool in {"discover_add_field_plan", "discover_protocol_sync_plan"} and obs.get(
        "ok"
    ):
        # Keep role hits for review/evidence; plan_markdown is the answer source.
        for role, hit in (obs.get("roles") or {}).items():
            if not isinstance(hit, dict):
                continue
            retrieved.append(
                {
                    "kind": role,
                    "fqcn": hit.get("fqcn") or "",
                    "method_name": hit.get("method_name") or "",
                    "file_path": hit.get("file_path") or "",
                    "summary": hit.get("why") or hit.get("evidence") or role,
                    "raw_code": "",
                    "from_change_impact": True,
                }
            )


def _critical_hint(
    tool: str,
    obs: dict[str, Any],
    *,
    endpoint: bool,
    validation: bool,
    full: bool,
    add_field: bool = False,
) -> str:
    """Short conditional hints only — steady-state guidance lives in SYSTEM."""
    if add_field and tool == "search_symbols" and obs.get("ok"):
        return (
            "已有候选：按角色并行 get_symbol_code——"
            "请求 VO、入参绑定/校验、落库组装、带 @TableName 的实体；"
            "不要只 search，不要背诵固定方法名。"
        )
    if add_field and tool == "get_symbol_code" and obs.get("ok"):
        roles = _add_field_role_of_hit(
            {
                **(obs.get("symbol") or {}),
                "raw_code": obs.get("raw_code") or "",
            }
        )
        if "vo" in roles or "bind" in roles:
            return (
                "继续沿本入口读：落库组装方法 + 持久化实体（@TableName/ORM）；勿提前 finish。"
            )
        if "assemble" in roles:
            return "下一步读持久化实体（@TableName）；确认是否 MP insert；读齐再 finish。"
        if "entity" in roles or "orm" in roles:
            return (
                "若 VO/绑定/组装仍缺请补齐；齐后 finish（## 结论 / ## 改动顺序 / "
                "## 从 → 到 / ## DDL；MP 一般不必改 Mapper.xml）。"
            )
    if endpoint and tool == "get_symbol_code" and obs.get("ok"):
        code = obs.get("raw_code") or ""
        if _FEIGN_CODE.search(code):
            return "已读到 Feign：请直接 finish 列出 name/path/Mapping。"
        if re.search(r"\w+Client\.\w+\(", code):
            return "请对调用的 *Client 接口 get_symbol_code 再 finish。"
    if validation and tool == "get_symbol_code":
        same = obs.get("same_class_calls") or []
        if same:
            fq = ((obs.get("symbol") or {}).get("fqcn")) or ""
            return f"下一步 get_symbol_code fqcn={fq} method_name={same[0]}，读完 finish。"
    if full and tool == "get_symbol_code" and obs.get("ok"):
        code = obs.get("raw_code") or ""
        fqcn = ((obs.get("symbol") or {}).get("fqcn")) or ""
        if _NOTICE_NAME.search(fqcn):
            return "这像结果通知入口，请重搜申请/提交 Controller；回调留到回答末节。"
        if _MQ_PUBLISH_MARKERS.search(code):
            return "已发 MQ：请并行 search/读 *Listener* 与下游 Client，然后 finish。"
    return ""


async def _summarize_from_evidence(
    question: str,
    *,
    retrieved: list[dict[str, Any]],
    steps: list[dict[str, Any]],
    endpoint: bool,
    tech: bool,
    full: bool,
    prefer_cm: str | None = None,
) -> str:
    evidence = _evidence_brief(retrieved)
    lock_line = ""
    if prefer_cm:
        lock_line = (
            f"【强制入口】必须按 {prefer_cm} 作答，禁止改答其他 Controller/方法"
            f"（尤其禁止把查询接口写成通知/防重链）。\n"
        )
    if endpoint:
        summary_sys = (
            "用户只要服务地址/服务名。根据已读 Client 源码直答："
            "Client 类、@FeignClient name、url、path、HTTP Mapping。"
            "准确优先；禁止业务流水账；未见硬编码 URL 则说明仅有服务名。"
            "禁止编造未读到的 path/URL；禁止 symbol_id。"
        )
        summary_ask = "请只输出服务地址相关事实。"
    elif wants_add_field_change_plan(question):
        field = assumed_new_field_name(question)
        summary_sys = (
            "用户要「新增入参字段并落库」的开发改动清单。只根据已读证据输出：\n"
            "## 结论\n\n"
            "一句话。\n\n"
            "## 改动顺序（按数据流：常量 → VO → 入参绑定 → 组装落库 → 实体/DDL）\n\n"
            "1. 文件路径或 Class#method\n"
            "   - 做什么：…\n"
            "   - 证据：…\n\n"
            "## 从 → 到（按文件，可复制）\n\n"
            f"- 具体改法（字段名优先 `{field}`）\n\n"
            "## DDL\n\n"
            "```sql\nALTER TABLE …;\n```\n\n"
            "## 说明（闭包/证据范围；缺口）\n\n"
            "节与节空一行。类名/方法名/表名/常量类必须来自已读证据，禁止套用其它接口的固定清单；"
            "若绑定用 *Constant*.FOO 作键则先补常量；"
            "按「常量 → 请求 VO → 绑定 → 落库组装 → 实体/@TableName → DDL」归纳；"
            "MyBatis-Plus insert 一般不必改 Mapper.xml；"
            "禁止业务状态流转/外部系统调用/评审提示；禁止编造；禁止 symbol_id。"
        )
        summary_ask = (
            "请按「## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明」输出；"
            "若已有 discover_add_field_plan 结果则原样采用。"
        )
    elif tech:
        summary_sys = (
            "根据已读符号证据总结流程。可写类名#方法、路径、关键错误码。"
            "准确优先：只写证据中出现的类/方法/字段/表/接口；"
            "证据不足写「未在已读材料中看到…」或「探索不足…」；禁止编造；禁止 symbol_id。"
        )
        summary_ask = "请总结（可含技术锚点）。"
    elif full:
        summary_sys = (
            "你在向业务同学讲解完整流程。准确优先。只根据已读证据，严格按版式输出：\n"
            "业务入口：HTTP path（不要主动写 Controller#method）\n"
            "主链：\n1. 业务动作 — …\n2. 业务动作 — …\n"
            "结果回调/通知：\n- 业务动作 — …（主入口之后的下游回传）\n"
            "按调用顺序；禁止拼凑；禁止空 ****；禁止把回调当主入口；"
            "禁止编造表名/字段/接口；未见则写「未在已读材料中看到…」；"
            "默认不要主动给出类名/方法名；禁止行业套话；禁止 symbol_id。"
        )
        summary_ask = "请直接给出「入口 + 主链 1…N + 结果回调」版式的完整流程说明。"
    else:
        summary_sys = (
            "你是业务解读助手。准确优先。只能根据已读证据总结。"
            "结构：业务入口、主流程、校验与规则、分支与结果、外部协同。"
            "用业务语言写清步骤与规则，默认不要主动给出类名/方法名；"
            "写清查哪张表/实体或返回字段——仅当证据中出现；否则写「未在已读材料中看到具体表名」；"
            "禁止编造 Feign/表名/字段/错误文案；禁止把查询写成通知流程。"
            "禁止 symbol_id；证据不足写「探索不足…」，不要用“可能”或“例如 t_xxx”补全。"
        )
        summary_ask = "请根据已读源码总结，入口必须与强制入口一致。"
    llm = get_llm_provider()
    summary = await llm.chat(
        [
            {"role": "system", "content": summary_sys},
            {
                "role": "user",
                "content": (
                    f"{lock_line}"
                    f"问题: {question}\n"
                    f"已见符号证据:\n{evidence}\n"
                    f"工具步骤: {truncate(json.dumps(steps, ensure_ascii=False), 1800)}\n"
                    f"{summary_ask}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=2500,
    )
    # Include retrieved evidence so sanitize keeps real tables/fields from code.
    ev_blob = evidence
    for h in retrieved or []:
        if not h:
            continue
        code = h.get("raw_code") or ""
        if code:
            ev_blob += "\n" + str(code)[:1200]
        if h.get("summary"):
            ev_blob += "\n" + str(h.get("summary"))
    out = (
        _sanitize_answer(summary, full_flow=full, evidence_text=ev_blob)
        or "探索步数用尽，未能完整回答。"
    )
    if wants_add_field_change_plan(question):
        out = _sanitize_add_field_change_plan(out, question=question) or out
        if _add_field_answer_missing_scannable_format(out):
            fb = _fallback_add_field_plan_from_evidence(question, retrieved)
            if fb:
                out = fb
    if prefer_cm and (
        _answer_violates_locked_entry(out, prefer_cm)
        or _answer_ignores_prefetch_or_invents_feign(out, retrieved)
    ):
        fb = _fallback_answer_from_prefetch(question, prefer_cm, retrieved)
        if fb:
            return fb
    return out


def _fallback_answer_from_prefetch(
    question: str,
    prefer_cm: str,
    retrieved: list[dict[str, Any]],
) -> str | None:
    """Deterministic answer from prefetched entry+downstream when LLM drifts."""
    cls, meth = _locked_entry_parts(prefer_cm)
    if not cls or not meth:
        return None
    downs = [
        h
        for h in (retrieved or [])
        if str(h.get("summary") or "").startswith("预读下游")
        and (h.get("raw_code") or "").strip()
    ]
    entry = next(
        (
            h
            for h in (retrieved or [])
            if str(h.get("summary") or "").startswith("预读锁定入口")
            or (h.get("method_name") or "").lower() == meth.lower()
        ),
        None,
    )
    lines = [
        f"业务入口：{prefer_cm}",
        "主流程：",
        f"1. {cls}#{meth} — 接收请求并调用下游服务（以已读源码为准）。",
    ]
    n = 2
    for h in downs:
        fq = h.get("fqcn") or ""
        m = h.get("method_name") or ""
        code = h.get("raw_code") or ""
        lines.append(f"{n}. {fq}#{m} — 下游实现。")
        fields = list(
            dict.fromkeys(re.findall(r'(?:req|reqMap|body)\.get\("(\w+)"\)', code))
        )[:10]
        if fields:
            lines.append("   入参/取值字段：" + "、".join(fields))
        if re.search(
            r"QueryWrapper|LambdaQueryWrapper|selectOne|selectList|Mapper\.|getOne\(",
            code,
        ):
            lines.append("   数据来源：本地库查询（Mapper/Wrapper）。")
        if re.search(r"@FeignClient|FeignClient", code):
            lines.append("   外部调用：源码含 Feign Client。")
        n += 1
    if entry and not downs:
        code = entry.get("raw_code") or ""
        calls = _DOWNSTREAM_RECEIVER.findall(code)
        if calls:
            lines.append(
                "入口内直接调用："
                + "、".join(f"{a}.{b}" for a, b in calls[:5])
            )
    lines.append("校验与规则：以已读方法内 assert/判空/分支为准；未见则写代码中未见额外校验。")
    lines.append("外部协同：仅列出已读代码中的 Client/Feign/MQ；未读到则写代码中未见。")
    return "\n".join(lines)


async def _finish_with_review(
    *,
    question: str,
    answer: str,
    steps: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    research_plan: dict[str, Any],
    progress: Any,
    prefer_cm: str | None = None,
) -> dict[str, Any]:
    """Attach completeness review and close progress (no further tool loop)."""
    from app.agent.answer_reviewer import evidence_from_retrieved, review_answer

    # Last-line defense: never ship a wrong-entry answer after tool budget ends.
    if prefer_cm and (
        _answer_violates_locked_entry(answer, prefer_cm)
        or _answer_ignores_prefetch_or_invents_feign(answer, retrieved)
    ):
        fb = _fallback_answer_from_prefetch(question, prefer_cm, retrieved)
        if fb:
            answer = fb

    ev_txt = evidence_from_retrieved(retrieved)
    answer = _sanitize_answer(answer, evidence_text=ev_txt) or answer
    if wants_add_field_change_plan(question):
        answer = _sanitize_add_field_change_plan(answer, question=question) or answer
        if _add_field_answer_missing_scannable_format(answer):
            fb = _fallback_add_field_plan_from_evidence(question, retrieved)
            if fb:
                answer = fb

    synth_id = progress.synthesize_step_id() if progress else None
    if synth_id:
        await progress.start(
            synth_id,
            "核对改动顺序…"
            if wants_add_field_change_plan(question)
            else "核对回答细节…",
        )
    if progress:
        await progress.emit(
            {
                "type": "review",
                "status": "running",
                "note": (
                    "核对结论 / 改动顺序 / 从→到 / DDL…"
                    if wants_add_field_change_plan(question)
                    else "核对落库 / 外部调用 / 主链是否写清…"
                ),
                "plan": research_plan,
            }
        )
    review = await review_answer(
        question,
        answer,
        evidence=ev_txt,
        plan_title=str(research_plan.get("title") or ""),
    )
    steps.append(
        {
            "step": (steps[-1].get("step") + 1) if steps else 0,
            "tool": "answer_review",
            "ok": bool(review.get("ok")),
            "review": {
                "ok": review.get("ok"),
                "score": review.get("score"),
                "summary": review.get("summary"),
                "gaps": review.get("gaps"),
            },
        }
    )
    if progress:
        await progress.emit(
            {
                "type": "review",
                "status": "done" if review.get("ok") else "rejected",
                "note": str(review.get("summary") or ""),
                "review": review,
                "plan": research_plan,
            }
        )
        if synth_id:
            await progress.done(
                synth_id,
                "细节已较完整" if review.get("ok") else "已整理当前最佳回答",
            )
        await progress.mark_remaining_done()
    return {
        "answer": _with_review_footnote(answer, review, question=question),
        "steps": steps,
        "retrieved": retrieved,
        "plan": research_plan,
        "review": review,
    }


@traceable(name="explore_agent", run_type="chain")
async def run_explore_agent(
    question: str,
    *,
    project_id: str,
    max_steps: int | None = None,
    entry_hint: str | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Returns {answer, steps, retrieved, plan}.

    on_progress: optional async/sync callback receiving SSE-style events
    ({type: plan|step, ...}) for the chat UI.
    """
    from app.agent.task_planner import (
        KIND_STATUS_UPDATE,
        PlanProgress,
        build_research_plan_async,
        plan_strategy_hint,
    )

    llm = get_llm_provider()
    steps: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []

    hint = f"\n入口提示: {entry_hint}" if entry_hint else ""
    tech = _wants_tech_detail(question)
    endpoint = _wants_endpoint_lookup(question)
    downstream_apis = _wants_flow_downstream_apis(question)
    api_inventory = _wants_api_inventory(question)
    validation = _wants_validation_focus(question)
    full = _wants_full_flow(question)
    add_field = wants_add_field_change_plan(question)
    protocol_sync = False  # domain router removed
    composite = False  # domain router removed

    # API inventory: deterministic Controller list (skip flow ask_retrieval / LLM essay).
    if api_inventory and project_id and not endpoint and not downstream_apis:
        try:
            inv_ans, inv_items = _assemble_api_inventory_answer(project_id, question)
            if inv_ans and inv_items:
                if on_progress:
                    try:
                        maybe = on_progress(
                            {
                                "type": "plan",
                                "plan": {
                                    "kind": "api_inventory",
                                    "title": "检索相关 HTTP 接口清单",
                                    "steps": [
                                        {
                                            "id": "search_entries",
                                            "title": "entry_route / Controller 检索",
                                            "status": "done",
                                        },
                                        {
                                            "id": "list",
                                            "title": "列出接口",
                                            "status": "done",
                                        },
                                    ],
                                },
                            }
                        )
                        if hasattr(maybe, "__await__"):
                            await maybe  # type: ignore[misc]
                    except Exception:
                        pass
                steps.append(
                    {
                        "step": 0,
                        "tool": "api_inventory",
                        "ok": True,
                        "count": len(inv_items),
                        "entries": [it.get("cm") for it in inv_items[:20]],
                    }
                )
                for it in inv_items[:20]:
                    retrieved.append(
                        {
                            "fqcn": str(it.get("cm") or "").split("#", 1)[0],
                            "method_name": (
                                str(it.get("cm") or "").split("#", 1)[-1]
                                if "#" in str(it.get("cm") or "")
                                else ""
                            ),
                            "kind": "Controller",
                            "summary": it.get("api_doc") or it.get("summary") or "",
                            "http_paths": it.get("http_paths") or "",
                        }
                    )
                return {
                    "answer": inv_ans,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": {
                        "kind": "api_inventory",
                        "title": "相关 HTTP 接口清单",
                        "steps": [],
                    },
                }
        except Exception:
            logger.debug("api inventory assemble failed; tool loop", exc_info=True)

    # Show-code / 源码: resolve Class#method + paste body BEFORE research-plan LLM.
    # Avoids picking mid-chain helpers (applyResultDealWith) and finishing without code.
    q_show = _normalize_show_code_question(question)
    source_paste_early = wants_show_gate_code(q_show) or _wants_source_paste(q_show)
    if (
        source_paste_early
        and project_id
        and not endpoint
        and not api_inventory
        and not downstream_apis
        and not add_field
    ):
        try:
            prefer_early = _class_method_from_hint(entry_hint) or _lexical_main_entry(
                q_show
            )
            src_ans, src_target = _assemble_source_code_answer(
                project_id, prefer_early, q_show
            )
            if src_ans and "```" in (src_ans or "") and src_target:
                if on_progress:
                    try:
                        maybe = on_progress(
                            {
                                "type": "plan",
                                "plan": {
                                    "kind": "show_gate_code",
                                    "title": f"贴出方法源码 {src_target}",
                                    "steps": [
                                        {
                                            "id": "resolve",
                                            "title": "定位 Class#method",
                                            "status": "done",
                                            "note": src_target,
                                        },
                                        {
                                            "id": "paste",
                                            "title": "读取并贴出源码",
                                            "status": "done",
                                        },
                                    ],
                                },
                            }
                        )
                        if hasattr(maybe, "__await__"):
                            await maybe  # type: ignore[misc]
                    except Exception:
                        pass
                steps.append(
                    {
                        "step": 0,
                        "tool": "source_paste",
                        "ok": True,
                        "target": src_target,
                        "entry_anchor": prefer_early,
                    }
                )
                retrieved.append(
                    {
                        "fqcn": src_target.split("#", 1)[0],
                        "method_name": src_target.split("#", 1)[-1],
                        "kind": "Controller"
                        if "Controller" in src_target
                        else "Method",
                        "summary": f"source paste {src_target}",
                    }
                )
                return {
                    "answer": src_ans,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": {
                        "kind": "show_gate_code",
                        "title": f"贴出方法源码 {src_target}",
                        "steps": [],
                    },
                    "review": {"ok": True, "source": "source_paste"},
                }
        except Exception:
            logger.debug("early source-paste failed; tool loop", exc_info=True)

    # Add-field: structured discover+render first — skip research-plan LLM and
    # free explore (those caused 1～2s stale「改动清单」or 70s LLM rewrite).
    if add_field and project_id:
        try:
            from app.agent.ask_retrieval import resolve_flow_entry_for_question
            from app.agent.explore_tools import discover_add_field_plan as _dap

            entry = _class_method_from_hint(entry_hint) or resolve_flow_entry_for_question(
                project_id, question, entry_hint=entry_hint
            )
            if on_progress:
                try:
                    maybe = on_progress(
                        {
                            "type": "plan",
                            "plan": {
                                "kind": "add_field_plan",
                                "title": "结构化发现新增字段改动点",
                                "steps": [
                                    {
                                        "id": "discover",
                                        "title": "发现改动触点",
                                        "status": "running",
                                    },
                                    {
                                        "id": "render",
                                        "title": "AI 梳理改动方案与 DDL",
                                        "status": "pending",
                                    },
                                ],
                            },
                        }
                    )
                    if hasattr(maybe, "__await__"):
                        await maybe  # type: ignore[misc]
                except Exception:
                    pass
            dap = _dap(
                project_id,
                question,
                entry_cm=entry,
                on_progress=on_progress,
            )
            steps.append(
                {
                    "step": 0,
                    "tool": "discover_add_field_plan",
                    "args": {"question": question, "entry_cm": entry},
                    "ok": bool(dap.get("ok")),
                    "roles": list((dap.get("roles") or {}).keys()),
                    "entry": dap.get("entry"),
                }
            )
            for st in dap.get("steps") or []:
                if isinstance(st, dict):
                    steps.append({**st, "via": "discover_add_field_plan"})
            plan_md = str(dap.get("plan_markdown") or dap.get("answer") or "")
            if dap.get("ok") and plan_md.strip():
                _absorb_tool_result("discover_add_field_plan", dap, retrieved)
                answer = (
                    _sanitize_add_field_change_plan(plan_md, question=question)
                    or plan_md
                )
                if on_progress:
                    try:
                        maybe = on_progress(
                            {
                                "type": "step",
                                "step_id": "*",
                                "status": "done",
                                "note": (
                                    "AI 梳理改动方案已生成"
                                    if dap.get("render_source") == "ai"
                                    else "改动方案已生成"
                                ),
                            }
                        )
                        if hasattr(maybe, "__await__"):
                            await maybe  # type: ignore[misc]
                    except Exception:
                        pass
                return {
                    "answer": answer,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": {
                        "kind": "add_field_plan",
                        "title": "结构化发现新增字段改动点",
                    },
                    "review": {"ok": True, "source": "change_impact"},
                }
        except Exception:
            logger.exception("add-field change_impact fast path failed; falling back")

    max_steps = _adaptive_max_steps(question, requested=max_steps)

    from app.core.config import get_settings as _get_settings

    source_only = (_get_settings().explore_backend or "source").strip().lower() == "source"

    # Dynamic plan via LLM (heuristic fallback inside); publish ASAP for UI.
    research_plan = await build_research_plan_async(question, project_id=project_id)
    progress = PlanProgress(research_plan, on_progress=on_progress)
    await progress.publish_plan()
    first_id = (research_plan.get("steps") or [{}])[0].get("id") or "search"
    await progress.start(
        first_id,
        "按研究计划在源码中查找…" if source_only else "开始按计划检索项目代码",
    )

    # Prefetch offline docs early — skipped entirely in source-only mode.
    offline_block = ""
    offline_quality: dict[str, Any] = {"rich": False, "usable": False}
    entry_anchor: dict[str, Any] | None = None
    mysql_flow: dict[str, Any] | None = None
    # Session / caller entry_hint with Class#method wins for follow-ups
    # (e.g. after a flow ask → the resolved Controller#method).
    prefer_cm: str | None = _class_method_from_hint(entry_hint)
    # Show-code: paste raw_code only when session injects Class#method symbols
    # or the question names Class#method explicitly — no business theme tables.
    lexical_cm = _lexical_main_entry(question)
    notice_entry = _wants_notice_entry(question)
    gate_detail = False
    source_paste = wants_show_gate_code(question) or _wants_source_paste(question)
    # Late paste retry (session symbols / after prefer_cm filled) — early path already tried.
    if (
        project_id
        and source_paste
        and not endpoint
        and not api_inventory
        and not downstream_apis
    ):
        try:
            src_ans, src_target = _assemble_source_code_answer(
                project_id, prefer_cm or lexical_cm, question
            )
            if src_ans and "```" in (src_ans or "") and src_target:
                await progress.done_at(0, f"定位方法源码 {src_target}")
                await progress.done_at(1, "raw_code / 工程 Java")
                synth_id = progress.synthesize_step_id()
                if synth_id:
                    await progress.start(synth_id, "贴出方法源码")
                steps.append(
                    {
                        "step": 0,
                        "tool": "source_paste",
                        "ok": True,
                        "entry_anchor": prefer_cm or lexical_cm,
                        "source_paste": True,
                        "target": src_target,
                        "symbols": [src_target],
                    }
                )
                if synth_id:
                    await progress.done(synth_id, "已贴出源码")
                await progress.mark_remaining_done("已由源码直出覆盖")
                return {
                    "answer": src_ans,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": research_plan,
                }
        except Exception:
            logger.debug("generic source-paste failed; tool loop", exc_info=True)

    if source_only:
        # Pure AI plan → tool loop on filesystem by default.
        # Keep lexical/API-path locks so we do not drift (postiveAdjust→quotaAdjust).
        prefer_cm = lexical_cm or prefer_cm
        # Exception: when Neo4j FOLLOWS / MySQL FlowDoc already exist for a locked
        # sync/full-flow entry, answer from that spine — same as index backend.
        # Otherwise source tool-loop LLM collapses ~10 stages into 4–5 steps.
        if _eligible_follows_spine_fast_path(
            question,
            prefer_cm=prefer_cm,
            endpoint=endpoint,
            api_inventory=api_inventory,
            downstream_apis=downstream_apis,
            validation=validation,
            tech=tech,
            notice_entry=notice_entry,
            status_update=research_plan.get("kind") == KIND_STATUS_UPDATE,
        ):
            try:
                spine_flow = _load_follows_spine_flow(project_id, prefer_cm or "")
                if spine_flow:
                    stage_n = _flow_doc_stage_count(spine_flow)
                    await progress.done_at(
                        0, f"入口 {prefer_cm}（用已入库业务摘要）"
                    )
                    await progress.done_at(1, f"FOLLOWS/FlowDoc 阶段数 {stage_n}")
                    synth_id = progress.synthesize_step_id()
                    if synth_id:
                        await progress.start(synth_id, "按入口业务摘要整理回答")
                    answer = await _answer_from_offline_corpus(
                        question,
                        corpus="",
                        full=full,
                        project_id=project_id,
                        prefer_class_method=prefer_cm,
                    )
                    steps.append(
                        {
                            "step": 0,
                            "tool": "offline_fast_path",
                            "ok": True,
                            "entry_anchor": prefer_cm,
                            "follows_source": spine_flow.get("follows_source"),
                            "stage_count": stage_n,
                            "source_backend_follows_bypass": True,
                        }
                    )
                    if synth_id:
                        await progress.done(synth_id, "已整理完成")
                    await progress.mark_remaining_done("已由入口业务摘要覆盖")
                    return {
                        "answer": answer,
                        "steps": steps,
                        "retrieved": retrieved,
                        "plan": research_plan,
                    }
            except Exception:
                logger.debug(
                    "source-backend FOLLOWS bypass failed; falling back to source tools",
                    exc_info=True,
                )
        hint = (
            "\n【纯源码模式】不要使用任何离线流程文档或记忆中的固定业务链路。"
            "必须 search_symbols(用问题实词/HTTP path) "
            "→ get_symbol_code 读入口 → list_callees/resolve_impl 下钻 → finish。"
            "禁止编造未读到的类/方法。"
        )
        if prefer_cm and "#" in prefer_cm:
            cls, meth = prefer_cm.split("#", 1)
            hint += (
                f"\n【已锁定入口】{prefer_cm}。"
                f"第一步 search_symbols({cls}) 或 get_symbol_code(fqcn={cls}, "
                f"method_name={meth})（用索引补全 fqcn），"
                f"禁止改答同 Controller 其他方法或其他业务入口。"
            )
            if re.search(r"Query|query", meth):
                hint += (
                    f"\n注意：{meth} 是查询接口，禁止写成通知/"
                    "重复申请/加锁防重；只写查询入参、查哪张表/实体、返回什么。"
                )
            # Prefetch locked entry code into evidence so the model cannot claim「未找到」.
            try:
                from app.agent.explore_tools import get_symbol_code

                fqcn_guess = (
                    prefer_cm
                    if prefer_cm.startswith("com.")
                    else f"{cls}"
                )
                # prefer_cm may already be Class#method without package.
                if "#" in fqcn_guess:
                    simple = fqcn_guess.split("#", 1)[0]
                    fqcn_guess = (
                        simple
                        if simple.startswith("com.")
                        else f"{simple}"
                    )
                pre = get_symbol_code(
                    project_id, fqcn=fqcn_guess, method_name=meth
                )
                if not pre.get("ok"):
                    # Retry via search then get.
                    from app.agent.explore_tools import search_symbols

                    sr = search_symbols(project_id, prefer_cm, kind="Controller", top_k=5)
                    for h in sr.get("hits") or []:
                        if (h.get("method_name") or "").lower() == meth.lower():
                            pre = get_symbol_code(
                                project_id,
                                symbol_id=h.get("symbol_id"),
                                fqcn=h.get("fqcn"),
                                method_name=h.get("method_name"),
                            )
                            if pre.get("ok"):
                                break
                if pre.get("ok"):
                    sym = pre.get("symbol") or {}
                    retrieved.append(
                        {
                            "symbol_id": sym.get("symbol_id"),
                            "fqcn": sym.get("fqcn"),
                            "method_name": sym.get("method_name"),
                            "kind": sym.get("kind"),
                            "raw_code": pre.get("raw_code"),
                            "summary": f"预读锁定入口 {prefer_cm}",
                            "file_path": (sym.get("file_path") or ""),
                        }
                    )
                    code_snip = truncate(str(pre.get("raw_code") or ""), 2800)
                    hint += (
                        f"\n【预读入口源码】{sym.get('fqcn')}#{sym.get('method_name')}:\n"
                        f"{code_snip}\n"
                        "以上源码已读到；禁止声称该接口不存在。继续下钻 ServiceImpl/落库/外部调用。"
                    )
                    # Auto-read first-level Service/Mapper/Client called by the entry.
                    down = _prefetch_downstream_from_code(
                        project_id, str(pre.get("raw_code") or ""), limit=4
                    )
                    if down:
                        retrieved.extend(down)
                        down_lines = []
                        for h in down:
                            down_lines.append(
                                f"- 已预读 {h.get('fqcn')}#{h.get('method_name')}\n"
                                f"{truncate(str(h.get('raw_code') or ''), 1200)}"
                            )
                        hint += (
                            "\n【预读下游实现】\n"
                            + "\n".join(down_lines)
                            + "\n请基于以上真实源码写清：校验字段、何时落库/写哪张表或实体、"
                            "外部 Client/Feign/MQ 怎么调；禁止空话「继续处理」。"
                        )
                    await progress.emit(
                        {
                            "type": "step",
                            "step_id": first_id,
                            "status": "running",
                            "note": (
                                f"已锁定并预读入口 {prefer_cm}"
                                + (f" +{len(down)} 下游" if down else "")
                            ),
                            "plan": research_plan,
                        }
                    )
            except Exception as e:
                logger.warning("source-only entry prefetch failed: %s", e)
        await progress.emit(
            {
                "type": "step",
                "step_id": first_id,
                "status": "running",
                "note": (
                    f"已跳过索引库；入口锁定 {prefer_cm}"
                    if prefer_cm
                    else "已跳过索引库，等待按计划 search 源码…"
                ),
                "plan": research_plan,
            }
        )
    else:
        await progress.emit(
            {
                "type": "step",
                "step_id": first_id,
                "status": "running",
                "note": "正在检索项目索引与流程材料…",
                "plan": research_plan,
            }
        )
    if not source_only:
        if lexical_cm and (
            _wants_flow_downstream_apis(question)
            or _wants_sync_entry_first(question)
            or notice_entry
        ):
            prefer_cm = lexical_cm
        try:
            # 1) entry_route classification → Class#method before raw FlowDoc noise.
            entry_anchor = _resolve_flow_entry_anchor(project_id, question)
            route_cm = (
                str(entry_anchor.get("class_method") or "").strip()
                if entry_anchor
                else None
            ) or None
            # Follow-up「调用了哪些接口」often has no new business noun — keep session lock.
            # Lexical lock already set prefer_cm; do not let empty/wrong route overwrite.
            if lexical_cm and (
                _wants_flow_downstream_apis(question)
                or _wants_sync_entry_first(question)
                or notice_entry
            ):
                prefer_cm = lexical_cm
            else:
                prefer_cm = prefer_cm or route_cm
            # Reject session/vector locks that clearly mismatch question nouns
            # (e.g. a notice API must not reuse an unrelated Controller).
            if prefer_cm and not _entry_aligned_with_question(prefer_cm, question):
                prefer_cm = lexical_cm  # may still be None
                if not prefer_cm:
                    entry_hint = None
                    hint = ""
            if prefer_cm:
                entry_hint = prefer_cm if not entry_hint else entry_hint
                hint = f"\n入口提示: {entry_hint}"
                if notice_entry:
                    hint += (
                        "\n注意：本题问的是通知类接口本身的主链，"
                        "禁止改答与本题无关的其它业务入口。"
                    )

            # 2) Load materials scoped to that entry (plus question tokens).
            search_q = question
            if prefer_cm:
                search_q = f"{prefer_cm} {question}"
            prefetch = list(hybrid_search(search_q, project_id, top_k=12) or [])
            # Full-flow: pull callback / listener / risk docs in the same offline pass
            if full or _wants_callback_section(question):
                extra_q = f"{question} 回调 Listener MQ Callback Notice Notify"
                try:
                    prefetch.extend(hybrid_search(extra_q, project_id, top_k=8) or [])
                except Exception:
                    pass
            if prefer_cm:
                try:
                    prefetch.extend(
                        hybrid_search(prefer_cm, project_id, top_k=8) or []
                    )
                except Exception:
                    pass
            prefetch = _rerank_hits_for_question(prefetch, question)
            # For「xx流程」Qs, drop query/get side paths so offline fast-path won't lock onto them.
            if _wants_sync_entry_first(question):
                filtered: list[Any] = []
                for h in prefetch:
                    cm = _hit_class_method(h)
                    method = _hit_method_name(h) or _method_from_class_method(cm)
                    # Drop query/get FlowDocs AND Controller node summaries that are queryish
                    # unless they are the routed entry itself.
                    if method and _QUERYISH_METHOD.match(method):
                        if not (prefer_cm and cm.lower() == prefer_cm.lower()):
                            continue
                    if (
                        _looks_like_result_entry(
                            class_method=cm,
                            paths=[],
                            title=_hit_text(h)[:80],
                            flow_role="",
                        )
                        and not _wants_callback_section(question)
                        and not _wants_notice_entry(question)
                    ):
                        # Keep for callback section later, but drop from main blob.
                        continue
                    filtered.append(h)
                # Only replace if we still have some signal; else keep reranked list.
                if filtered:
                    prefetch = filtered
            # Prefer locked entry; on「流程」Qs drop same-Controller sibling methods
            # so offline synthesis does not become an endpoint catalog.
            if prefer_cm and prefetch:
                prefetch = _scope_prefetch_to_entry(
                    prefetch,
                    prefer_cm=prefer_cm,
                    question=question,
                    keep_unrelated=4 if (full or _wants_callback_section(question)) else 0,
                )
            offline_quality = _offline_corpus_quality(prefetch)
            flow_bits: list[str] = []
            seen_sid: set[str] = set()
            for h in prefetch:
                sid = h.symbol_id or ""
                if sid and sid in seen_sid:
                    continue
                if sid:
                    seen_sid.add(sid)
                kind = h.kind or ""
                if kind in {"FlowDoc", "Overview"} or sid.startswith(("flow:", "overview:")):
                    flow_bits.append(f"- [{kind}] {h.fqcn}\n{truncate(h.summary or '', 2000)}")
                    retrieved.append(
                        {
                            "symbol_id": h.symbol_id,
                            "kind": h.kind,
                            "fqcn": h.fqcn,
                            "summary": truncate(h.summary or "", 1400),
                            "score": h.score,
                        }
                    )
                elif h.summary and (h.score or 0) >= 1.2:
                    flow_bits.append(
                        f"- [{kind}] {h.fqcn}#{(h.method_sig or '')[:80]}\n{truncate(h.summary, 700)}"
                    )
                    retrieved.append(
                        {
                            "symbol_id": h.symbol_id,
                            "kind": h.kind,
                            "fqcn": h.fqcn,
                            "summary": truncate(h.summary or "", 700),
                            "score": h.score,
                        }
                    )
            if flow_bits:
                offline_block = (
                    "\n\n【离线检索命中】（解析阶段已写入的流程/节点/回调；优先直接 finish 整理）\n"
                    + "\n".join(flow_bits[:14])
                )
            # MySQL FlowDoc stages for the routed/picked entry — answer even if Milvus is thin.
            mysql_flow = (
                _pick_structured_flow_doc(
                    project_id, question, prefer_class_method=prefer_cm
                )
                if project_id
                else None
            )
            # Hard lock: if pick drifted or refused soft-match, load exact Class#method FlowDoc.
            if prefer_cm and (
                not mysql_flow
                or str(mysql_flow.get("class_method") or "").split("#", 1)[0].split(".")[-1]
                != prefer_cm.split("#", 1)[0].split(".")[-1]
                or (
                    "#" in prefer_cm
                    and str(mysql_flow.get("class_method") or "")
                    .split("#", 1)[-1]
                    .split("(", 1)[0]
                    != prefer_cm.split("#", 1)[-1]
                )
            ):
                locked = _load_flow_doc_by_class_method(project_id, prefer_cm)
                if locked:
                    mysql_flow = locked
            # Milvus/entry lock → Neo4j FOLLOWS spine (not CALLS neighborhood / sibling catalog).
            if mysql_flow or prefer_cm:
                overlaid = _overlay_follows_on_flow(
                    project_id,
                    mysql_flow,
                    entry_cm=prefer_cm
                    or (str(mysql_flow.get("class_method") or "") if mysql_flow else None),
                    flow_key=(mysql_flow or {}).get("flow_key") if mysql_flow else None,
                )
                if overlaid:
                    mysql_flow = overlaid
            mysql_stage_count = _flow_doc_stage_count(mysql_flow)
            # Surface FOLLOWS spine in offline corpus so tool-loop answers ground on it.
            if mysql_flow and (mysql_flow.get("follows_source") or mysql_stage_count >= 2):
                spine_txt = _render_ordered_main_chain(
                    mysql_flow,
                    tech=_wants_tech_detail(question),
                    deep=False,
                )
                if spine_txt:
                    offline_block = (
                        (offline_block or "")
                        + "\n\n【Neo4j FOLLOWS 执行序】（业务主链顺序；禁止用 CALLS 邻域重排）\n"
                        + spine_txt
                    )
                    offline_quality = dict(offline_quality or {})
                    blob0 = str(offline_quality.get("blob") or "")
                    offline_quality["blob"] = (blob0 + "\n" + spine_txt).strip()
                    offline_quality["usable"] = True
                    if mysql_stage_count >= 3:
                        offline_quality["rich"] = True
                        offline_quality["flow_chars"] = max(
                            int(offline_quality.get("flow_chars") or 0),
                            len(spine_txt),
                        )
            # Downstream「调用了哪些」: deterministic list from locked FlowDoc stages.
            if (
                downstream_apis
                and mysql_flow
                and mysql_stage_count >= 2
                and not api_inventory
            ):
                _, down_items = _collect_flow_downstream_apis(mysql_flow)
                answer = _render_flow_downstream_apis(mysql_flow)
                steps.append(
                    {
                        "step": 0,
                        "tool": "offline_downstream_apis",
                        "ok": True,
                        "entry_anchor": prefer_cm or mysql_flow.get("class_method"),
                        "item_count": len(down_items),
                    }
                )
                await progress.done_at(
                    0, f"已锁定入口 {prefer_cm or mysql_flow.get('class_method')}"
                )
                await progress.done_at(1, f"FlowDoc 阶段数 {mysql_stage_count}")
                await progress.done_at(2, f"下游 {len(down_items)} 个")
                await progress.mark_remaining_done("已根据 FlowDoc 整理清单")
                return {
                    "answer": answer,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": research_plan,
                }

            sync_flow_q = _wants_sync_entry_first(question) and not api_inventory
            has_mysql_stages = bool(mysql_flow and mysql_stage_count >= 3 and sync_flow_q)
            # Fast path: usable Milvus corpus OR locked-entry MySQL FlowDoc stages.
            # Status-update asks need Entity→writers→callers; skip offline fast path.
            status_update = research_plan.get("kind") == KIND_STATUS_UPDATE
            # Misaligned entry vs FlowDoc must not use offline fast path.
            entry_ok = (not prefer_cm) or _entry_aligned_with_question(prefer_cm, question)
            flow_ok = (not mysql_flow) or _entry_aligned_with_question(
                str(mysql_flow.get("class_method") or ""), question
            )
            # Offline blob itself must match question nouns (not just prefer_cm label).
            blob_txt = str(offline_quality.get("blob") or "")
            corpus_ok = (not blob_txt) or _offline_blob_aligned(blob_txt, question)
            from app.core.config import get_settings as _get_settings

            # Without Milvus, do not trust thin/wrong offline FlowDoc narratives.
            milvus_on = bool(_get_settings().search_use_milvus)
            use_fast = (
                (
                    bool(offline_quality.get("usable") or offline_quality.get("rich"))
                    or has_mysql_stages
                )
                and not endpoint
                and not api_inventory
                and not downstream_apis
                and not validation
                and not tech
                and not add_field  # 新增字段改动清单必须读 VO/绑定/Ext，禁止离线套话
                and not status_update
                and not notice_entry  # 通知接口题必须读源码，禁止套无关离线文档
                and entry_ok
                and flow_ok
                and corpus_ok
                and (bool(offline_quality.get("blob")) or has_mysql_stages)
                and (milvus_on or has_mysql_stages)  # milvus off → only exact MySQL FlowDoc
            )
            if mysql_flow and not flow_ok:
                mysql_flow = None
                has_mysql_stages = False
            if not corpus_ok:
                offline_block = ""
                offline_quality = {"rich": False, "usable": False, "blob": ""}
            if use_fast:
                # Do not promote bare「流程/业务描述」to full-chain (callback) mode
                # just because FlowDoc text is long — that invited LLM rewrite drift.
                await progress.done_at(
                    0,
                    f"入口 {prefer_cm or (entry_anchor or {}).get('class_method') or '已定位'}",
                )
                await progress.done_at(
                    1,
                    f"离线材料可用（节点 {offline_quality.get('node_hits') or 0}）",
                )
                synth_id = progress.synthesize_step_id()
                if synth_id:
                    await progress.start(synth_id, "根据离线材料整理回答")
                answer = await _answer_from_offline_corpus(
                    question,
                    corpus=str(offline_quality.get("blob") or ""),
                    full=full,
                    project_id=project_id,
                    prefer_class_method=prefer_cm,
                )
                steps.append(
                    {
                        "step": 0,
                        "tool": "offline_fast_path",
                        "ok": True,
                        "flow_chars": offline_quality.get("flow_chars"),
                        "node_hits": offline_quality.get("node_hits"),
                        "has_callback": offline_quality.get("has_callback"),
                        "entry_anchor": prefer_cm,
                        "entry_role": (entry_anchor or {}).get("flow_role")
                        if entry_anchor
                        else None,
                        "follows_source": (mysql_flow or {}).get("follows_source")
                        if mysql_flow
                        else None,
                    }
                )
                if synth_id:
                    await progress.done(synth_id, "已整理完成")
                await progress.mark_remaining_done("已由离线材料覆盖")
                return {
                    "answer": answer,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": research_plan,
                }
        except Exception:
            offline_block = ""

    plan_hint = plan_strategy_hint(research_plan)

    if protocol_sync:
        style, plan = protocol_sync_explore_guidance(question)
    elif composite:
        style, plan = composite_change_explore_guidance(question)
    elif add_field:
        style, plan = add_field_explore_guidance(question)
    elif research_plan.get("kind") == KIND_STATUS_UPDATE:
        style = (
            "回答模式=状态更新链路：先写清对应哪张表/哪个实体、状态字段名；"
            "再写哪些方法对字段赋值/更新、触发条件与状态流转；"
            "最后用调用链说明从哪个业务入口触发。禁止空话，必须有代码证据。"
        )
        plan = (
            "请开始：search_symbols(q=表/实体实词, kind=Entity) → "
            "get_symbol_code 确认状态字段 → "
            "search_symbols(q=状态字段/update/setStatus/Mapper) → "
            "get_symbol_code 读赋值点 → list_callers 还原调用链 → finish。"
        )
    elif api_inventory:
        style = (
            "回答模式=接口清单：先给出命中数量，再逐条列出 "
            "Class#method、HTTP path（若有）、一句话用途（来自 @ApiOperation/注释/摘要）。"
            "只列与问题实词相关的 Controller 入口；不要讲业务流程长文；不要说「材料中未找到」除非 search 确实无 Controller 命中。"
            "声明的「共 N 个」必须与列表条数一致。"
        )
        if any((h.get("kind") or "") == "Controller" for h in retrieved):
            plan = (
                "请开始：【离线检索命中】已含 Controller 时可直接 finish 清点数量与清单；"
                "若缺 path/用途再 search_symbols(q=问题实词, kind=Controller) 补全。"
            )
        else:
            plan = (
                "请开始：search_symbols(q=问题实词, kind=Controller) → "
                "必要时再 search 同 Controller 类名补全同主题入口 → finish 给出数量+清单。"
            )
    elif downstream_apis:
        style = (
            "回答模式=主链下游调用清单：只列锁定入口主链上实际调用的 "
            "Service/ServiceImpl/Feign/*Client/MQ 等下游，"
            "禁止罗列同 Controller 上的兄弟 HTTP 入口方法。"
            "先写主链入口 Class#method，再写「共 N 个」且 N 必须等于列表条数；"
            "每条格式「Class#method — 做什么」。"
        )
        plan = (
            "请开始：优先用入口提示/FlowDoc stages；"
            "不足则 get_symbol_code 入口 → list_callees → resolve_impl → finish。"
            "不要 search 同 Controller 的其他 mapping 方法当答案。"
        )
        if prefer_cm:
            hint = f"\n入口提示: {prefer_cm}（已锁定，禁止换成同 Controller 其他方法）"
    elif endpoint:
        style = (
            "回答模式=直答服务地址：必须给出 Feign name、url（若有）、path、"
            "@PostMapping/@GetMapping 路径与 Client 类名；"
            "若只有注册中心服务名没有 http URL，明确说明。"
            "禁止只写业务流程空话。"
        )
        plan = (
            "请开始：可并行 search_symbols(q=服务实词+Client/Feign) → "
            "get_symbol_code 读 Client → finish 直答地址事实。"
        )
    elif validation:
        style = (
            "回答模式=字段校验清单：逐条列出校验了哪些业务字段、规则与失败时业务含义；"
            + ("可附类名#方法。" if tech else "默认不写类名/错误码字面量。")
            + "必须读入口方法的 same_class_calls 校验方法，禁止跑去无关 Service。"
        )
        plan = (
            "请开始：search_symbols → 同轮或下轮 get_symbol_code 入口与 same_class_calls → finish。"
        )
    elif full:
        if tech:
            style = (
                "回答模式=完整业务链路（有序阶段，要具体）：\n"
                "业务入口：HTTP path + Controller#method（本题同步入口）\n"
                "主链：\n"
                "1. Class#method — …（受理/开关/决策等，以代码为准）\n"
                "2. Class#method — …（校验/写库等具体事实）\n"
                "3. Class#method — …（继续主处理）\n"
                "结果回调/通知：\n"
                "- Class#method — …（主入口之后的下游回传；禁止当主链第1步）\n"
                "每步保留 Class#method，禁止输出空的 ****；按调用顺序，不要拼凑无关节点。"
                "禁止空模板与行业领域套话；用词只来自问题与代码/材料。"
            )
        else:
            style = (
                "回答模式=业务向完整链路：优先直接采用入口方法已入库的业务摘要"
                "（summary_business_detail/clear），可补少量下游方法一句话；"
                "末尾「结果回调/通知」业务简述。"
                "禁止强制十步源码主链；禁止变量名/常量名/status_line 摘抄；"
                "默认不要主动给出类名/方法名。"
            )
        if offline_quality.get("rich"):
            plan = (
                "请开始：【离线检索命中】已较完整时可直接 finish；"
                "主链按入口调用顺序；回调单独成节。"
            )
        else:
            plan = (
                "请开始：FlowDoc 仅作提纲；必须 search 相关 Controller → "
                "get_symbol_code 入口+resolve_impl/ServiceImpl；"
                "并行读校验相关方法；若问回调再 search 回调/Listener/Notice 并读码 → "
                "证据够再 finish（主链含具体校验+状态；回调节有事实）。"
            )
    else:
        style = (
            "回答模式=技术细节：可写类名#方法、路径、关键错误码。"
            if tech
            else (
                "回答模式=纯业务：finish.answer 以业务步骤与规则为主；"
                "须写清落库时机/对象/标准，以及外部系统如何协同（同步/异步、调谁、结果如何影响）；"
                "默认不要主动给出类名/方法名，也不写包名/错误码字符串，但不要省略落库与外部调用事实。"
            )
        )
        if offline_quality.get("rich") and not tech:
            plan = "请开始：离线材料已够时可直接 finish；不足再 search/读码。"
        else:
            plan = (
                "请开始：若有离线流程文档可作提纲，仍须 search/读码补具体规则后再 finish；"
                "可并行 search → get_symbol_code → list_callees/resolve_impl；"
                "读到 ServiceImpl 后写清校验、落库、外部调用，禁止「继续处理」收尾。"
            )

    flow_guard = ""
    if downstream_apis:
        flow_guard = (
            "\n注意：本题问的是主链过程中调用的下游接口/服务，"
            "不是同 Controller 上有哪些兄弟 HTTP 接口。"
        )
    elif _wants_sync_entry_first(question) and not _wants_api_inventory(question):
        flow_guard = (
            "\n注意：本题是业务流程/业务描述，不是「有哪几个接口」。"
            "只围绕入口主链（FlowDoc 阶段）作答，禁止罗列同 Controller 兄弟方法当步骤。"
        )

    from app.agent.explore_tools import tool_catalog_text as _tool_catalog_text

    system_content = SYSTEM
    if source_only:
        # Refresh tool catalog for source-only rules (SYSTEM may be stale at import).
        base = SYSTEM.split("可用工具")[0] if "可用工具" in SYSTEM else SYSTEM
        system_content = base + _tool_catalog_text()
        system_content += (
            "\n当前为纯源码探索：禁止引用离线 FlowDoc/主链记忆；"
            "只用 search_symbols → get_symbol_code → list_callees/resolve_impl 的真实返回作答。"
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        {
            "role": "user",
            "content": (
                f"项目 project_id={project_id}\n"
                f"用户问题: {question}{hint}\n"
                f"{style}\n"
                f"{plan_hint}\n"
                f"{plan}"
                f"{flow_guard}"
                f"{offline_block}"
            ),
        },
    ]

    style_rejects = 0
    depth_rejects = 0
    finish_nudges = 0
    review_rejects = 0
    # Structured plan from discover_add_field_plan tool — finish must pass through.
    change_impact_plan: str | None = None

    # Seed add-field via normal tool dispatch (not graph/HTTP short-circuit).
    if add_field and project_id:
        try:
            from app.agent.ask_retrieval import resolve_flow_entry_for_question
            from app.agent.explore_tools import discover_add_field_plan as _dap

            entry = prefer_cm or resolve_flow_entry_for_question(
                project_id, question, entry_hint=entry_hint
            )
            dap = _dap(
                project_id,
                question,
                entry_cm=entry,
                on_progress=on_progress,
            )
            steps.append(
                {
                    "step": 0,
                    "tool": "discover_add_field_plan",
                    "args": {"question": question, "entry_cm": entry},
                    "ok": bool(dap.get("ok")),
                    "prefetch": True,
                    "roles": list((dap.get("roles") or {}).keys()),
                    "entry": dap.get("entry"),
                }
            )
            if dap.get("ok") and (dap.get("plan_markdown") or dap.get("answer")):
                change_impact_plan = str(
                    dap.get("plan_markdown") or dap.get("answer") or ""
                )
                _absorb_tool_result("discover_add_field_plan", dap, retrieved)
                # Merge discovery sub-steps for UI transparency.
                for st in dap.get("steps") or []:
                    if isinstance(st, dict):
                        steps.append({**st, "via": "discover_add_field_plan"})
                # Return immediately: LLM must not rewrite into the legacy
                # 「改动清单 / 建议改动」摘要 (drops constants / java snippets).
                answer = (
                    _sanitize_add_field_change_plan(
                        change_impact_plan, question=question
                    )
                    or change_impact_plan
                )
                await progress.mark_remaining_done("结构化改动方案已生成")
                return {
                    "answer": answer,
                    "steps": steps,
                    "retrieved": retrieved,
                    "plan": {
                        "kind": "add_field_plan",
                        "title": "结构化发现新增字段改动点",
                    },
                    "review": {"ok": True, "source": "change_impact"},
                }
        except Exception:
            change_impact_plan = None
            # Fallback: light role seed via search (legacy path if tool fails).
            try:
                from app.agent.explore_tools import get_symbol_code, search_symbols

                seed_notes: list[str] = []
                for sq in _add_field_seed_queries(prefer_cm, question):
                    sr = search_symbols(project_id, sq, top_k=5)
                    steps.append(
                        {
                            "step": 0,
                            "tool": "search_symbols",
                            "args": {"q": sq},
                            "ok": bool(sr.get("ok")),
                            "hit_count": int(sr.get("count") or 0),
                            "prefetch": True,
                        }
                    )
                    if not sr.get("ok"):
                        continue
                    for h in (sr.get("hits") or [])[:3]:
                        if not h:
                            continue
                        retrieved.append(h)
                        sid = str(h.get("symbol_id") or "")
                        fq = str(h.get("fqcn") or "")
                        meth = str(h.get("method_name") or "")
                        code_args: dict[str, Any] = {}
                        if sid:
                            code_args["symbol_id"] = sid
                        else:
                            code_args["fqcn"] = fq
                            if meth:
                                code_args["method_name"] = meth
                        if not code_args.get("symbol_id") and not code_args.get("fqcn"):
                            continue
                        got = get_symbol_code(project_id, **code_args)
                        steps.append(
                            {
                                "step": 0,
                                "tool": "get_symbol_code",
                                "args": code_args,
                                "ok": bool(got.get("ok")),
                                "prefetch": True,
                            }
                        )
                        if got.get("ok"):
                            sym = dict(got.get("symbol") or h)
                            sym["raw_code"] = got.get("raw_code") or ""
                            retrieved.append(sym)
                            seed_notes.append(
                                f"{sym.get('fqcn') or fq}"
                                f"#{sym.get('method_name') or meth or ''}".rstrip("#")
                            )
                        if _read_count(steps) >= _ADD_FIELD_MIN_READS + 2:
                            break
                    if _add_field_evidence_gaps(steps, retrieved, soft=True) is None:
                        break
                if seed_notes:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "已按入口前缀预读候选（请继续补齐缺失角色后 finish）：\n- "
                                + "\n- ".join(seed_notes[:8])
                            ),
                        }
                    )
            except Exception:
                pass

    for step in range(1, max_steps + 1):
        if not await _maybe_compact_messages(messages):
            _trim_messages(messages)
        content = await llm.chat(messages, temperature=0.1, max_tokens=1600)
        calls = _extract_tool_calls(content)
        if not calls:
            messages.append({"role": "assistant", "content": content})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "请只输出 JSON 工具调用（可 tools 数组并行 1～3 个只读工具）。"
                        '例: {"tools":[{"tool":"resolve_impl","args":{"fqcn":"OrderService",'
                        '"method_name":"submit"}},{"tool":"get_symbol_code",'
                        '"args":{"symbol_id":"..."}}]} '
                        "深度够了再用单独的 finish。"
                    ),
                }
            )
            continue

        # ---- finish (single) ----
        if calls[0]["tool"] == "finish":
            call = calls[0]
            args = call.get("args") or {}
            # Pass through structured change_impact plan verbatim (light wrap only).
            if add_field and change_impact_plan:
                ans = change_impact_plan
                args = {**args, "answer": ans}
                call = {**call, "args": args}
                soft = True
                block = None
            else:
                soft = depth_rejects >= _MAX_DEPTH_FINISH_REJECTS or step >= max_steps - 1
                block = _finish_block_reason(
                    steps, retrieved, question=question, soft=soft
                )
                ans = str((args or {}).get("answer") or "")
                if not block and prefer_cm:
                    entry_block = _answer_violates_locked_entry(ans, prefer_cm)
                    if entry_block:
                        # Never soft-accept wrong-entry answers.
                        block = entry_block
                if not block:
                    invent_block = _answer_ignores_prefetch_or_invents_feign(ans, retrieved)
                    if invent_block:
                        # Wrong Feign / ignored ServiceImpl: keep rejecting until budget,
                        # but always reject at least twice.
                        if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 3 and soft:
                            block = None
                        else:
                            block = invent_block
                            style_rejects += 1
                if (
                    not block
                    and add_field
                    and _add_field_answer_is_persist_gate_narrative(ans)
                ):
                    if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 1:
                        block = None
                    else:
                        block = (
                            "本题是「新增字段要改哪些」，不是现有落库关卡解说。"
                            "请按已读证据重写：## 结论 → ## 改动顺序（常量 → VO → 绑定 → "
                            "组装 → 实体 → DDL）→ ## 从 → 到 → ## DDL → ## 说明；"
                            "节与节空一行；只写真实读到的类/方法；"
                            "MP insert 一般不必改 Mapper.xml；禁止只罗列 insert* 关卡。"
                        )
                        style_rejects += 1
                if (
                    not block
                    and add_field
                    and _add_field_answer_has_synth_junk(ans)
                ):
                    if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 1:
                        ans = _sanitize_add_field_change_plan(ans, question=question)
                        args = {**args, "answer": ans}
                        call = {**call, "args": args}
                        block = None
                    else:
                        block = (
                            "finish.answer 含流程评审残留（业务状态流转 / 外部系统调用 / "
                            "评审提示等）。请删除这些段落，只保留 "
                            "## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明"
                            "（节与节之间空一行）。"
                        )
                        style_rejects += 1
                if (
                    not block
                    and add_field
                    and _add_field_answer_has_offtopic_credit_noise(ans, question)
                ):
                    if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 1:
                        ans = _sanitize_add_field_change_plan(ans, question=question)
                        args = {**args, "answer": ans}
                        call = {**call, "args": args}
                        block = None
                    else:
                        block = (
                            "请只写问题所涉接口链路（已读的 VO/绑定/组装/实体 + DDL）。"
                            "不要把无关业务的 Controller（如 Credit*）混进改动清单。"
                        )
                        style_rejects += 1
                if (
                    not block
                    and add_field
                    and _add_field_answer_wrong_apply_persist(ans)
                ):
                    if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 1:
                        ans = _sanitize_add_field_change_plan(ans, question=question)
                        args = {**args, "answer": ans}
                        call = {**call, "args": args}
                        block = None
                    else:
                        block = (
                            "落库请按已读组装方法写入对应持久化实体（常为 Ext/@TableName + "
                            "MyBatis-Plus insert），不要臆造未在证据中的主表加列。"
                            "DDL 只 ALTER 已读 @TableName 对应表。"
                        )
                        style_rejects += 1
                if (
                    not block
                    and add_field
                    and _add_field_answer_missing_scannable_format(ans)
                ):
                    # Never soft-accept a thin DDL-only answer; prefer evidence fallback.
                    if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 3:
                        fb = _fallback_add_field_plan_from_evidence(question, retrieved)
                        if fb:
                            ans = fb
                            args = {**args, "answer": ans}
                            call = {**call, "args": args}
                            block = None
                        else:
                            block = (
                                "finish.answer 仍缺改动清单要点：须覆盖请求 VO、入参绑定、"
                                "落库组装、持久化实体与 DDL（均来自已读证据）；"
                                "请基于已读源码重写后再 finish。"
                            )
                            style_rejects += 1
                    else:
                        block = (
                            "finish.answer 版式不够直观：请严格按 "
                            "## 结论 → ## 改动顺序（做什么 + 证据）→ "
                            "## 从 → 到 → ## DDL → ## 说明 重写；"
                            "节与节之间空一行、每条单独一行；不要写业务流程解说；"
                            "类名/方法/表名/常量须来自已读证据；不要把 Mapper.xml 写成必改项。"
                        )
                        style_rejects += 1
            if not block and endpoint and not re.search(
                r"(/[A-Za-z]|-service|FeignClient|@FeignClient|name\s*=|url\s*=|path\s*=|Mapping)",
                ans,
            ):
                if style_rejects >= _MAX_STYLE_FINISH_REJECTS:
                    block = None  # accept; sanitize won't invent Feign facts
                else:
                    block = (
                        "回答未包含服务名/path/Feign 事实。请根据已读 Client 源码重写 finish.answer，"
                        "直接列出 name、path、Mapping，不要写业务流程。"
                    )
                    style_rejects += 1
            if not block and full and _is_messy_full_flow_answer(ans):
                if style_rejects >= _MAX_STYLE_FINISH_REJECTS:
                    # rewrite once via sanitize and accept — do not burn more rounds
                    ans = _sanitize_answer(ans, full_flow=True)
                    args = {**args, "answer": ans}
                    call = {**call, "args": args}
                    block = None
                else:
                    block = (
                        "finish.answer 不符合完整流程版式：请用「主链：1.2.3…」短阶段重写，"
                        "每步写出具体业务事实（校验项/状态变化）；「结果回调/通知」另起一节；"
                        "禁止独白与空模板。"
                    )
                    style_rejects += 1
            # Full-flow + ordinary 流程/通知 questions: reject vague tails before review.
            needs_depth = full or any(
                k in (question or "")
                for k in ("流程", "链路", "通知", "怎么走", "什么样", "回调")
            )
            if not block and needs_depth:
                shallow = _full_flow_answer_too_shallow(ans, question)
                if shallow:
                    if style_rejects >= _MAX_STYLE_FINISH_REJECTS + 1:
                        block = None
                    else:
                        block = shallow
                        style_rejects += 1
            # Downstream ask: reject answers that invent foreign Controllers.
            if not block and downstream_apis and mysql_flow:
                foreign = _downstream_answer_foreign_controllers(ans, mysql_flow)
                if foreign:
                    # Prefer deterministic FlowDoc stages over another LLM rewrite.
                    ans = _render_flow_downstream_apis(mysql_flow)
                    args = {**args, "answer": ans}
                    call = {**call, "args": args}
                    block = None
            if block:
                is_style_block = (
                    "版式" in block
                    or "Feign 事实" in block
                    or "不符合完整流程" in block
                    or "过空" in block
                    or "下游回调" in block
                    or "串到下游" in block
                    or "已锁定入口" in block
                    or "已预读下游" in block
                    or "编造" in block
                    or "FeignClient" in block
                )
                if not is_style_block:
                    depth_rejects += 1
                messages.append(
                    {"role": "assistant", "content": json.dumps(call, ensure_ascii=False)}
                )
                messages.append({"role": "user", "content": f"还不能 finish：{block}"})
                steps.append(
                    {
                        "step": step,
                        "tool": "finish",
                        "args": args,
                        "ok": False,
                        "rejected": True,
                    }
                )
                continue

            synth_id = progress.synthesize_step_id()
            if synth_id:
                await progress.start(synth_id, "正在整理最终回答")
            obs = dispatch_tool(project_id, "finish", args)
            from app.agent.answer_reviewer import (
                evidence_from_retrieved,
                review_answer,
                review_reject_message,
            )

            ev_txt = evidence_from_retrieved(retrieved)
            # Prefer structured tool plan over LLM rewrite / sanitize drift.
            if add_field and change_impact_plan:
                answer = change_impact_plan
            else:
                answer = _sanitize_answer(
                    (obs.get("answer") or "").strip(),
                    full_flow=full,
                    evidence_text=ev_txt,
                )
                if add_field:
                    answer = _sanitize_add_field_change_plan(answer, question=question)
                    if _add_field_answer_missing_scannable_format(answer):
                        fb_plan = (
                            change_impact_plan
                            or _fallback_add_field_plan_from_evidence(
                                question, retrieved
                            )
                        )
                        if fb_plan:
                            answer = fb_plan
            if not answer:
                answer = "未能根据代码得出结论，请换个问法或先完成项目解析/建图。"
            if (
                prefer_cm
                and not (add_field and change_impact_plan)
                and (
                    _answer_violates_locked_entry(answer, prefer_cm)
                    or _answer_ignores_prefetch_or_invents_feign(answer, retrieved)
                )
            ):
                fb = _fallback_answer_from_prefetch(question, prefer_cm, retrieved)
                if fb:
                    answer = _sanitize_answer(fb, full_flow=full, evidence_text=ev_txt) or fb
                    if add_field:
                        answer = _sanitize_add_field_change_plan(answer, question=question)
                        if _add_field_answer_missing_scannable_format(answer):
                            fb_plan = _fallback_add_field_plan_from_evidence(
                                question, retrieved
                            )
                            if fb_plan:
                                answer = fb_plan

            # Second AI: completeness review (落库/外部调用/下游展开等).
            # Add-field plans use heuristic-only review (no 状态/外部调用苛求).
            review_note = (
                "核对结论 / 改动顺序 / 从→到 / DDL…"
                if add_field
                else "核对落库 / 外部调用 / 主链是否写清…"
            )
            await progress.emit(
                {
                    "type": "review",
                    "status": "running",
                    "note": review_note,
                    "plan": research_plan,
                }
            )
            review = await review_answer(
                question,
                answer,
                evidence=ev_txt,
                plan_title=str(research_plan.get("title") or ""),
            )
            steps.append(
                {
                    "step": step,
                    "tool": "finish",
                    "args": {"answer": answer},
                    "ok": bool(review.get("ok")),
                    "review": {
                        "ok": review.get("ok"),
                        "score": review.get("score"),
                        "summary": review.get("summary"),
                        "gaps": review.get("gaps"),
                    },
                }
            )
            await progress.emit(
                {
                    "type": "review",
                    "status": "done" if review.get("ok") else "rejected",
                    "note": str(review.get("summary") or ""),
                    "review": review,
                    "plan": research_plan,
                }
            )
            if not review.get("ok") and review_rejects < (
                _MAX_REVIEW_REJECTS + (2 if add_field else 0)
            ):
                review_rejects += 1
                messages.append(
                    {"role": "assistant", "content": json.dumps(call, ensure_ascii=False)}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": review_reject_message(review, question=question),
                    }
                )
                if synth_id:
                    # Keep synthesize step open for rewrite after more tools.
                    await progress.start(synth_id, "补充更多细节，再查一层实现")
                continue

            if synth_id:
                await progress.done(
                    synth_id,
                    "细节已较完整" if review.get("ok") else "已整理当前最佳回答",
                )
            await progress.mark_remaining_done()
            return {
                "answer": _with_review_footnote(answer, review, question=question),
                "steps": steps,
                "retrieved": retrieved,
                "plan": research_plan,
                "review": review,
            }

        # ---- batch read-only tools ----
        batch_notes: list[str] = []
        for call in calls:
            tool = call["tool"]
            args = call.get("args") or {}
            await progress.tool_activity(tool, args=args)
            obs = dispatch_tool(project_id, tool, args)
            step_rec: dict[str, Any] = {
                "step": step,
                "tool": tool,
                "args": args,
                "ok": obs.get("ok"),
            }
            if tool == "get_symbol_code" and obs.get("ok"):
                sym = obs.get("symbol") or {}
                step_rec["fqcn"] = sym.get("fqcn")
                step_rec["method_name"] = sym.get("method_name")
                step_rec["file_path"] = sym.get("file_path")
            steps.append(step_rec)
            _absorb_tool_result(tool, obs, retrieved)
            if tool in {
                "discover_add_field_plan",
                "discover_protocol_sync_plan",
            } and obs.get("ok"):
                plan_md = str(obs.get("plan_markdown") or obs.get("answer") or "")
                if plan_md:
                    change_impact_plan = plan_md
            # Surface a short note for the matched plan step.
            brief = ""
            if tool == "search_symbols":
                hits = obs.get("hits") or []
                if hits:
                    top = hits[0]
                    brief = f"命中 {(top.get('fqcn') or top.get('kind') or '')}"
                    if len(hits) > 1:
                        brief += f" 等 {len(hits)} 条"
                else:
                    brief = "暂无命中，将换词继续"
            elif tool == "get_symbol_code" and obs.get("ok"):
                sym = obs.get("symbol") or {}
                brief = f"已读 {sym.get('fqcn') or ''}#{sym.get('method_name') or ''}".rstrip("#")
            elif tool in {"list_callees", "list_callers"} and obs.get("ok"):
                brief = f"{tool} 返回 {(obs.get('count') or len(obs.get('items') or []))}"
            elif tool == "resolve_impl" and obs.get("ok"):
                brief = "已解析实现类"
            elif tool in {
                "discover_add_field_plan",
                "discover_protocol_sync_plan",
            } and obs.get("ok"):
                brief = "已生成结构化改动方案（请 finish 原样采用）"
            if brief:
                await progress.tool_done(tool, args=args, note=brief)
            hint_line = _critical_hint(
                tool,
                obs,
                endpoint=endpoint,
                validation=validation,
                full=full,
                add_field=add_field,
            )
            note = f"### {tool}\n{_compact_observation(obs)}"
            if hint_line:
                note += f"\n{hint_line}"
            batch_notes.append(note)

        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {"tools": calls} if len(calls) > 1 else calls[0],
                    ensure_ascii=False,
                ),
            }
        )

        ready = _core_chain_ready(steps, retrieved, question=question)
        soft_ok = (
            _finish_block_reason(steps, retrieved, question=question, soft=True) is None
        )
        if ready or soft_ok:
            finish_nudges += 1
            footer = (
                "证据已够：下一步必须单独调用 finish（完整题用主链1…N+结果回调；"
                "不要再 search）。"
            )
            if protocol_sync:
                if change_impact_plan:
                    footer = (
                        "discover_protocol_sync_plan 已给出方案："
                        "下一步必须单独 finish，answer 原样采用 plan_markdown"
                        "（## 结论 / ## 改动清单）。"
                    )
                else:
                    footer = (
                        "请先 discover_protocol_sync_plan，再 finish 原样采用方案。"
                    )
            elif add_field:
                if change_impact_plan:
                    footer = (
                        "discover_add_field_plan 已给出方案："
                        "下一步必须单独 finish，answer 原样采用 plan_markdown"
                        "（## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明）。"
                    )
                else:
                    footer = (
                        "新增字段角色链路已齐（请求 VO / 绑定 / 组装 / 实体）："
                        "下一步必须单独 finish，或先 discover_add_field_plan 再 finish。"
                    )
            # If model ignored finish nudge twice and budget is tight, summarize now.
            if finish_nudges >= 2 and step >= min(max_steps - 2, 8):
                answer = await _summarize_from_evidence(
                    question,
                    retrieved=retrieved,
                    steps=steps,
                    endpoint=endpoint,
                    tech=tech,
                    full=full,
                    prefer_cm=prefer_cm,
                )
                return await _finish_with_review(
                    question=question,
                    answer=answer,
                    steps=steps,
                    retrieved=retrieved,
                    research_plan=research_plan,
                    progress=progress,
                    prefer_cm=prefer_cm,
                )
        else:
            footer = (
                "可继续并行下钻；证据够了立即 finish。"
                "流程题 finish 须写清：落库时机/表或实体、外部调用方式，禁止「继续处理」收尾。"
            )
            if protocol_sync:
                if change_impact_plan:
                    footer = (
                        "已有结构化方案：请立即 finish，answer=plan_markdown 原样粘贴。"
                    )
                else:
                    footer = (
                        "请先调用 discover_protocol_sync_plan；勿提前 finish。"
                    )
            elif add_field:
                if change_impact_plan:
                    footer = (
                        "已有结构化方案：请立即 finish，answer=plan_markdown 原样粘贴。"
                    )
                else:
                    gap = _add_field_evidence_gaps(steps, retrieved) or ""
                    footer = (
                        f"新增字段链路未齐：{gap} "
                        "请先调用 discover_add_field_plan，或继续 get_symbol_code；勿提前 finish。"
                    )

        messages.append(
            {
                "role": "user",
                "content": "工具结果:\n" + "\n\n".join(batch_notes) + f"\n{footer}",
            }
        )

    if protocol_sync and change_impact_plan:
        return await _finish_with_review(
            question=question,
            answer=_sanitize_protocol_sync_plan(
                change_impact_plan, question=question
            )
            or change_impact_plan,
            steps=steps,
            retrieved=retrieved,
            research_plan=research_plan,
            progress=progress,
            prefer_cm=prefer_cm,
        )

    if add_field:
        fb = change_impact_plan or _partial_add_field_plan_from_evidence(
            question,
            retrieved,
            gap_note=_add_field_evidence_gaps(steps, retrieved, soft=True),
        )
        return await _finish_with_review(
            question=question,
            answer=fb,
            steps=steps,
            retrieved=retrieved,
            research_plan=research_plan,
            progress=progress,
            prefer_cm=prefer_cm,
        )

    answer = await _summarize_from_evidence(
        question,
        retrieved=retrieved,
        steps=steps,
        endpoint=endpoint,
        tech=tech,
        full=full,
        prefer_cm=prefer_cm,
    )
    return await _finish_with_review(
        question=question,
        answer=answer,
        steps=steps,
        retrieved=retrieved,
        research_plan=research_plan,
        progress=progress,
        prefer_cm=prefer_cm,
    )
