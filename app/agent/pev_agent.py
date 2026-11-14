"""Planner–Executor–Verifier (PEV) main loop — goal contracts + generic tools.

Test-phase outputs carry ``[PEV][...]`` tags for easier log triage.
No business-intent routers: tasks are planned from a goal contract, then
executed with existing retrieval / discover / source tools.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from app.core.utils import truncate

from app.agent.prompts import load_prompt

logger = logging.getLogger("app.agent.pev")

ProgressCb = Callable[[dict[str, Any]], Any] | None

# Plan layer (goal contracts / task planning / repo inventory / scaffold tools)
# moved to pev_plan.py — re-exported here for import compatibility.
from app.agent.pev_plan import (
    GOAL_TYPES,
    TAG,
    _ARCH_ANSWER_PROMPT,
    _BAD_MUST_IDS,
    _GOAL_PROMPT,
    _SPINE_NOISE_RE,
    _TOPIC_ANSWER_PROMPT,
    _emit,
    _exec_codebase_architecture,
    _exec_find_api_style,
    _exec_load_kb_spec,
    _exec_render_new_api,
    _exec_repo_files,
    _exec_symbol_probe,
    _exec_topic_existence,
    _footer_tags,
    _live_pev_plan,
    _merge_pev_answers,
    _parse_json_obj,
    _project_root_for,
    _tag,
    _topic_answer_too_thin,
    build_task_plan,
    extract_field_label_zh,
    extract_goal_contract,
    extract_named_tables,
    extract_repo_file_globs,
    format_repo_files_answer,
    heuristic_goal,
    list_repo_files,
    wants_add_api_scaffold,
    wants_compound_change,
    wants_field_value_ask_safe,
    wants_repo_file_inventory,
)

# Change layer (discover_change / outbound / render_change / source_view /
# api inventory / field & api-io tools) moved to pev_change.py — re-exported
# here for import compatibility.
from app.agent.pev_change import (
    _OUTBOUND_DTO_SET_RE,
    _OUTBOUND_MAP_PUT_RE,
    _REQ_GET_RE,
    _RESP_PUT_RE,
    _RESP_SET_RE,
    _api_io_helper_cms,
    _apply_zh_comments,
    _exec_api_inventory,
    _exec_collect_api_io,
    _exec_collect_field_evidence,
    _exec_compose_code,
    _exec_diagnose_answer,
    _exec_discover_change,
    _exec_find_outbound,
    _exec_general_answer,
    _exec_map_and_read,
    _exec_parse_signal,
    _exec_render_api_io,
    _exec_render_change,
    _exec_render_field_answer,
    _exec_search_hits,
    _exec_source_view,
    _extract_request_param_keys,
    _extract_response_param_keys,
    _getter_name,
    _merge_outbound_into_answer,
    _outbound_focus_queries,
    _outbound_hit_score,
    _outbound_peer_excerpt,
    _rewrite_placeholder_field,
    suggest_outbound_param_code,
)

# Flow layer (entry resolution / spine / dig-in focus / flow rendering)
# moved to pev_flow.py — re-exported here for import compatibility.
from app.agent.pev_flow import (
    _DIG_IN_FOCUS_PROMPT,
    _PEV_FLOW_SUMMARIZE_PROMPT,
    _allows_entry_scoped_dig_in,
    _anchor_leaf_from_answer_steps,
    _apply_topic_ground_to_entries,
    _best_search_closure_topic_leaf,
    _cached_calls_closure_cms,
    _cjk_ngrams,
    _cjk_overlap_score,
    _cjk_runs,
    _cm_simple_class,
    _collect_topic_cms_on_chain,
    _entry_chain_cms_set,
    _entry_match_meta,
    _exec_expand_calls,
    _exec_resolve_entry,
    _exec_summarize_flow,
    _fallback_dig_in_focus_cms,
    _filter_spine_cms,
    _focus_cms_from_question,
    _focus_depth_label,
    _incidental_topic_umbrella,
    _is_method_dig_in_ask,
    _is_service_or_client_leaf,
    _is_spine_noise,
    _leaf_focus_bonus,
    _lock_dig_in_goal,
    _method_name_topic_bonus,
    _order_focus_shallow_to_deep,
    _pick_best_topic_leaf,
    _pinnable_controller_entry,
    _prior_answer_snippet,
    _promote_related_leaf_to_focus,
    _render_flow_answer_deterministic,
    _render_focus_method_answer,
    _render_multi_entry_flow_answer,
    _render_one_entry_flow_block,
    _resolve_flow_spine,
    _resolve_method_topic,
    _row_for_cm,
    _score_topic_hit,
    _seed_reuse_locked_spine,
    _should_compress_lock_dig_in_goal,
    _simple_cm,
    _spine_candidates_for_llm,
    _spine_cms_for_focus,
    _token_overlap_score,
    _topic_ground_entry_candidates,
    _topic_leaf_score,
    _topic_needles_from_question,
    _topic_peer_cms_for_enrich,
    _topic_phrase_in_text,
    _topic_summary_specificity,
    _try_method_first_on_chain_focus,
    _wants_dig_in_change,
    verify_contract,
)


# Monkeypatch-sensitive flow functions kept here (pev_flow uses lazy wrappers).
_COMPOSE_UNDEF_RE = re.compile(r"缺失[)）]\s*[:：]\s*([^。]+)")


def _compose_undef_class_names(desc: str) -> list[str]:
    """Extract undefined class names from a compose_undef_class gap description.

    desc format: 「生成代码引用了未定义的类（实体/DTO/Mapper 缺失）: LoanCoupon, Withdrawals。…」
    """
    m = _COMPOSE_UNDEF_RE.search(desc or "")
    if not m:
        return []
    return [t.strip() for t in re.split(r"[,，]", m.group(1)) if t.strip()]


def _hit_value(h: Any, key: str, default: str = "") -> str:
    """Read a key from dict-like or attribute-like search hits."""
    try:
        if isinstance(h, dict):
            return str(h.get(key) or default)
        return str(getattr(h, key, None) or default)
    except Exception:
        return default


def _find_real_symbol_class(project_id: str, name: str) -> str | None:
    """Search index/source for a real class matching an undefined class name."""
    from app.agent.explore_shared import _locate_java_file_for_class
    from app.agent.explore_tools import search_symbols

    # 0) Disk/MySQL class-file existence is the ground truth — constant/util
    #    classes (FieldNameConstants etc.) often miss the fuzzy symbol search
    #    that drowns in Controller hits, yet their .java file exists on disk.
    try:
        path = _locate_java_file_for_class(project_id, name)
        if path:
            return name
    except Exception:
        pass

    candidates: list[str] = [name]
    if len(name) > 2:
        candidates.append(name[:-1])  # plural-ish / suffix variants
    if len(name) > 4:
        candidates.append(name.lower())
    seen_fqcn: set[str] = set()
    for q in candidates:
        try:
            sr = search_symbols(project_id, q, top_k=8)
        except Exception:
            continue
        for h in sr.get("hits") or []:
            fqcn = _hit_value(h, "fqcn")
            if not fqcn:
                fqcn = _hit_value(h, "symbol_id")
            simple = str(fqcn).split(".")[-1].strip()
            if not simple or simple in seen_fqcn:
                continue
            seen_fqcn.add(simple)
            if simple == name:
                return simple
            if name.lower() in simple.lower() or simple.lower() in name.lower():
                if len(simple) >= 3:
                    return simple
    return None


async def _repair_compose_undefined(
    project_id: str,
    question: str,
    goal: dict[str, Any],
    evidence: dict[str, Any],
    answer: str,
    verify: dict[str, Any],
    hist: str = "",
) -> tuple[str, dict[str, Any]]:
    """compose_undef_class gap → re-search undefined classes and rewrite.

    Never ships fabricated code: when no real class is found for an undefined
    name, the answer becomes an honest gap note listing the searched keywords.
    Returns (answer, verify) with the compose_undef_class gap removed.
    """
    undef: list[str] = []
    for g in verify.get("gaps") or []:
        if g.get("id") == "compose_undef_class":
            undef.extend(_compose_undef_class_names(str(g.get("desc") or "")))
    undef = list(dict.fromkeys(u for u in undef if u))
    if not undef:
        return answer, verify

    mapping: dict[str, str] = {}
    searched: list[str] = []
    for name in undef:
        real = _find_real_symbol_class(project_id, name)
        if real:
            mapping[name] = real
        searched.append(name)

    if mapping:
        # Re-synthesize anchored on a real class so the code only references
        # entities/Mappers that exist in the repo.
        from app.agent.compose_code import run_compose_code

        real_anchor = next(iter(mapping.values()))
        try:
            out = await run_compose_code(
                project_id,
                question,
                history_text=hist,
                lookup_key=evidence.get("lookup_key"),
                lookup_table=real_anchor,
                lookup_steps=evidence.get("lookup_steps"),
                view_kind=evidence.get("view_kind"),
            )
        except Exception:
            logger.debug("compose repair re-synthesis failed", exc_info=True)
            out = {}
        new_answer = str(out.get("answer") or "") if isinstance(out, dict) else ""
        if new_answer and "```" in new_answer and "TODO" not in new_answer:
            evidence["compose_code"] = out
            evidence["compose_repaired"] = mapping
            verify2 = verify_contract(goal, evidence, new_answer)
            verify2["gaps"] = [
                g
                for g in (verify2.get("gaps") or [])
                if g.get("id") != "compose_undef_class"
            ]
            verify2["ok"] = len(verify2.get("gaps") or []) == 0
            return new_answer, verify2

    # Honest fallback — never ship fabricated classes as if they were real.
    lines = [
        "未能生成可运行的代码：仓库中未找到以下类/实体："
        + "、".join(f"`{u}`" for u in undef),
        f"已检索关键词：{'、'.join(searched) or '（无）'}。",
        "请核对表名/实体名后再问；或改用「改动方案」，让我给出已有接口的改造清单。",
    ]
    verify2 = dict(verify)
    verify2["gaps"] = [
        g
        for g in (verify.get("gaps") or [])
        if g.get("id") != "compose_undef_class"
    ]
    verify2["ok"] = len(verify2.get("gaps") or []) == 0
    return "\n".join(lines), verify2


def _pev_show_tags() -> bool:
    """True when internal [PEV][...] tags/gap notes may be embedded in the answer.

    Default off (config ``pev_test_tags_in_answer``): user replies stay clean,
    while the same debug info is still returned in ``result["pev_debug"]``.
    """
    try:
        from app.core.config import get_settings

        return bool(get_settings().pev_test_tags_in_answer)
    except Exception:
        return False


_USER_TOOL_NOTES: dict[str, tuple[str, str]] = {
    "resolve_entry": ("正在定位业务入口…", "已定位入口"),
    "expand_calls": ("正在展开调用链…", "已展开调用链"),
    "mysql_enrich": ("正在读取方法说明…", "已读取方法说明"),
    "summarize_flow": ("正在整理业务回答…", "已整理业务回答"),
    "discover_change": ("正在分析改动点…", "已分析改动点"),
    "render_change": ("正在生成改动清单…", "已生成改动清单"),
    "load_kb_spec": ("正在读取知识库文档…", "已读取知识库文档"),
    "render_new_api": ("正在生成接口骨架…", "已生成接口骨架"),
    "show_code": ("正在定位源码…", "已定位源码"),
    "compose_code": ("正在根据证据生成代码…", "已生成代码"),
    "repo_files": ("正在扫描仓库文件…", "已列出仓库文件"),
}


def _user_tool_note(
    tool: str,
    status: str,
    *,
    evidence: dict[str, Any] | None = None,
    title: str = "",
) -> str:
    """Desktop-facing step text — no internal [PEV] tags."""
    pair = _USER_TOOL_NOTES.get(str(tool or ""))
    ev = evidence or {}
    label = title or tool or "步骤"
    if status in {"running", "pending"}:
        return pair[0] if pair else f"正在执行{label}…"
    if status in {"error", "failed"}:
        return f"{label}未完成"
    if status == "skipped":
        return f"跳过{label}"
    entry = str(ev.get("entry") or "")
    if pair:
        if tool == "resolve_entry" and entry:
            return f"{pair[1]} `{entry}`"
        if tool == "expand_calls" and ev.get("calls_count"):
            return f"{pair[1]}（{ev.get('calls_count')} 步）"
        return pair[1]
    return f"完成{label}"


def _build_pev_debug(
    goal: dict[str, Any],
    tasks: list[dict[str, Any]],
    verify: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, Any]:
    """Collect internal PEV debug info (footer tags + verify gaps) for meta.

    Kept out of the user-visible answer by default; the caller decides whether
    to also embed them when ``pev_test_tags_in_answer`` is on.
    """
    return {
        "footer": _footer_tags(goal, tasks, verify, evidence),
        "gaps": list(verify.get("gaps") or []),
        "verify_ok": bool(verify.get("ok")),
    }


def _try_lock_dig_in_description_focus(
    question: str,
    evidence: dict[str, Any],
    *,
    project_id: str = "",
) -> tuple[list[str], dict[str, Any]] | None:
    """Description/CJK overlap on answer_steps + summaries → leaf focus.

    Runs *before* dig_in LLM. Returns ``(focus_cms, dig_decision)`` or None
    when no confident hit (ambiguous → LLM fallback).

    Allowed under session ``lock_entry`` *or* first-turn ``entry_scoped_dig_in``
    (dig_in_likely / ask_scope from Goal).

    When both entry and method_topic/focus_topic are present, prefer
    **method-first ∩ entry-chain** over spine-first summary scoring.
    """
    q = (question or "").strip()
    if not q or not _allows_entry_scoped_dig_in(evidence):
        return None
    pid = project_id or str(evidence.get("project_id") or "")

    # Invert: topic → method search, then require on entry spine/closure.
    method_first = _try_method_first_on_chain_focus(
        q, evidence, project_id=pid
    )
    if method_first:
        return method_first

    named = _focus_cms_from_question(q, evidence)
    if named:
        decision: dict[str, Any] = {
            "dig_in": True,
            "focus_method": named[0],
            "topic": str(evidence.get("focus_topic") or ""),
            "reason": "named_method_in_question",
            "source": "desc_match_named",
        }
        focus = _prefer_topic_focus_cms(
            named[:1],
            decision,
            evidence,
            project_id=pid,
            question=q,
        )[:1]
        if focus:
            decision["focus_method"] = focus[0]
            return focus, decision

    # Subtopic description match only when dig-in is expected (or change dig-in).
    # Avoid turning "再讲一遍全流程" into a false leaf dig-in via incidental overlap.
    if not (
        evidence.get("dig_in_likely")
        or _wants_dig_in_change(q)
        or evidence.get("dig_in_focus")
        or str(evidence.get("ask_scope") or "").strip().lower()
        in {"entry_scoped_dig_in", "dig_in"}
    ):
        return None

    topic = str(
        evidence.get("method_topic") or evidence.get("focus_topic") or ""
    )
    anchored = _anchor_leaf_from_answer_steps(
        q, evidence, topic=topic, project_id=pid
    )
    if anchored:
        decision = {
            "dig_in": True,
            "focus_method": anchored[0],
            "topic": topic,
            "reason": "description_match_answer_step",
            "source": "desc_match_answer_step",
        }
        return anchored[:1], decision

    # Score summary overlap only (not the full
    # 「入口 + 子话题」问句 — incidental 接口 overlap false-locks Clients).
    spine = _spine_cms_for_focus(evidence)
    entry = _simple_cm(str(evidence.get("entry") or ""))
    scored: list[tuple[int, str]] = []
    query_blob = (topic or q).strip()
    topic_has_long_run = any(len(r) >= 4 for r in _cjk_runs(topic))
    for cm in spine:
        if not cm or (entry and cm == entry):
            continue
        row = _row_for_cm(evidence, cm) or {}
        summary = str(
            row.get("summary") or row.get("neo4j_desc") or row.get("title") or ""
        )
        if not summary.strip() and "#" not in q:
            continue
        if topic_has_long_run and not _topic_phrase_in_text(summary, topic):
            continue
        sc = _token_overlap_score(f"{cm} {summary}", query_blob) + _cjk_overlap_score(
            summary, query_blob
        )
        sc += _topic_summary_specificity(summary, query_blob)
        sc += _method_name_topic_bonus(cm, query_blob)
        if sc >= 8:
            scored.append((sc, cm))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    # Ambiguous top-2 → defer to dig_in LLM.
    if len(scored) >= 2 and scored[0][0] - scored[1][0] < 3:
        return None
    best = scored[0][1]
    decision = {
        "dig_in": True,
        "focus_method": best,
        "topic": topic,
        "reason": "description_match_summary",
        "source": "desc_match_summary",
    }
    focus = _prefer_topic_focus_cms(
        [best],
        decision,
        evidence,
        project_id=pid,
        question=q,
    )[:1]
    if not focus:
        return None
    decision["focus_method"] = focus[0]
    return focus, decision


_DIG_IN_FOCUS_PROMPT = load_prompt("dig_in_focus_prompt")


async def _exec_mysql_enrich(
    project_id: str, evidence: dict[str, Any]
) -> None:
    from app.agent.ask_retrieval import _mysql_enrich_ordered

    entry = evidence.get("entry") or ""
    spines = evidence.get("entry_spines") if isinstance(evidence.get("entry_spines"), dict) else {}
    matches = [
        _simple_cm(str(x))
        for x in (evidence.get("entry_matches") or [])
        if x and "#" in str(x)
    ]
    if entry and _simple_cm(entry) not in matches:
        matches = [_simple_cm(entry), *matches]
    if not matches and entry:
        matches = [_simple_cm(entry)]

    def _rows_for(cm: str, calls: list[Any], spine_src: str) -> list[dict[str, Any]]:
        spine_rel = "FOLLOWS"
        src = str(spine_src or "")
        if src.startswith("neo4j_calls") or src.startswith("calls_"):
            spine_rel = "CALLS"
        nodes: list[dict[str, Any]] = []
        for i, call_cm in enumerate((calls or [])[:14], start=1):
            cm_s = _simple_cm(call_cm)
            if not cm_s or cm_s == _simple_cm(cm):
                continue
            nodes.append(
                {
                    "order": i,
                    "class_method": cm_s,
                    "rel": spine_rel,
                    "title": "",
                    "desc": "",
                }
            )
        for i, n in enumerate(nodes, start=1):
            n["order"] = i
        return _mysql_enrich_ordered(project_id, nodes, max_nodes=16) if nodes else []

    entry_mysql: dict[str, list[dict[str, Any]]] = {}
    for cm in matches[:4]:
        spine = spines.get(cm) if isinstance(spines, dict) else None
        if not isinstance(spine, dict):
            calls, src = _resolve_flow_spine(project_id, cm) if cm else ([], "")
            spine = {"calls": calls, "spine_source": src}
        entry_mysql[cm] = _rows_for(
            cm,
            list(spine.get("calls") or []),
            str(spine.get("spine_source") or ""),
        )
    evidence["entry_mysql_rows"] = entry_mysql
    primary = _simple_cm(entry) or (matches[0] if matches else "")
    evidence["mysql_rows"] = list(entry_mysql.get(primary) or [])






# Monkeypatch-sensitive flow functions kept here (pev_flow uses lazy wrappers).
def _prefer_topic_focus_cms(
    focus_cms: list[str],
    dig_decision: dict[str, Any],
    evidence: dict[str, Any],
    *,
    project_id: str = "",
    question: str = "",
) -> list[str]:
    """Prefer topic leaf method over parent umbrella when both resolve.

    Promotes leaves via summary CJK/Latin overlap + callee-closure depth even
    when the topic has no Latin method-name tokens. Defaults to one focus cm.
    """
    topic = str(dig_decision.get("topic") or "").strip()
    q = (question or str(evidence.get("question") or "")).strip()
    pid = project_id or str(evidence.get("project_id") or "")
    llm_focus = _simple_cm(str(dig_decision.get("focus_method") or ""))

    # Prior-answer step title/detail CJK overlap → leaf in that step's closure.
    anchored = _anchor_leaf_from_answer_steps(
        q, evidence, topic=topic, project_id=pid
    )
    if anchored:
        dig_decision["reason"] = (
            str(dig_decision.get("reason") or "") + "; answer_step_anchor"
        ).strip("; ")
        src0 = str(dig_decision.get("source") or "")
        if src0 and "answer_step" not in src0:
            dig_decision["source"] = f"{src0}+answer_step_anchor"
        elif not src0:
            dig_decision["source"] = "answer_step_anchor"
        # If anchor already found a leaf distinct from the LLM umbrella parent,
        # trust it. If anchor == LLM parent focus, do NOT short-circuit — keep
        # resolve + search promotion so a topic leaf can still win.
        if not (llm_focus and anchored[0] == llm_focus):
            return anchored[:1]
        focus_cms = list(dict.fromkeys([*anchored, *(focus_cms or [])]))

    if not topic and not dig_decision.get("focus_method"):
        return list(focus_cms or [])[:1]
    query = topic or str(dig_decision.get("focus_method") or "")
    if q and not topic:
        query = f"{query} {q}".strip()
    alt = _resolve_focus_method_symbol(
        query,
        evidence,
        project_id=pid,
        topic=topic or q,
    )
    if not alt:
        # Retrieval fallback when resolve is empty but we still hold an umbrella.
        fb = _fallback_dig_in_focus_cms(q or query, evidence, project_id=pid)
        if fb and (not focus_cms or fb[0] != focus_cms[0]):
            if _is_service_or_client_leaf(fb[0], entry=_simple_cm(str(evidence.get("entry") or ""))):
                return fb[:1]
        return list(focus_cms or [])[:1]
    entry = _simple_cm(str(evidence.get("entry") or ""))
    pick = alt[0]
    if entry and pick == entry:
        return (focus_cms or alt)[:1]
    spine_set = set(_spine_cms_for_focus(evidence))

    def _specificity(cm: str) -> int:
        score = 0
        if "ServiceImpl#" in cm or re.search(r"(?i)Client#", cm):
            score += 3
        if "Controller#" in cm:
            score -= 4
        if entry and cm != entry:
            score += 1
        row = _row_for_cm(evidence, cm) or {}
        summary = str(row.get("summary") or row.get("neo4j_desc") or "")
        score += _topic_summary_specificity(summary, query)
        score += _cjk_overlap_score(summary, query)
        score += _method_name_topic_bonus(cm, query)
        # Deeper leaf vs parent umbrella: tighter summary wins.
        if focus_cms and cm != focus_cms[0]:
            score += 2
            parent = focus_cms[0]
            prow = _row_for_cm(evidence, parent) or {}
            psum = str(prow.get("summary") or prow.get("neo4j_desc") or "")
            if _topic_summary_specificity(psum, query) < _topic_summary_specificity(
                summary, query
            ):
                score += 6
            if cm not in spine_set:
                score += 3
        return score

    # Build search-augmented pool up front so umbrella parents that tie with
    # resolve can still lose to a search∩closure leaf without mysql summary.
    pool = list(dict.fromkeys([*alt, *(focus_cms or []), *_spine_cms_for_focus(evidence)]))
    search_rank: dict[str, int] = {}
    chain = _entry_chain_cms_set(evidence, project_id=pid) if pid else set()
    if pid and (query or q):
        try:
            from app.agent.ask_retrieval import _search_hit_cms_for_question

            probe = (query or topic or q).strip()
            for i, h in enumerate(
                _search_hit_cms_for_question(pid, probe, top_k=12)
            ):
                cm = _simple_cm(h)
                if not cm or "#" not in cm:
                    continue
                if chain and cm not in chain:
                    continue
                if cm not in search_rank:
                    search_rank[cm] = i
                if cm not in pool:
                    pool.append(cm)
        except Exception:
            logger.debug("prefer-topic search pool failed", exc_info=True)

    closure_for_promote = chain or set(pool)

    def _promote_over_umbrella(cur: str) -> list[str] | None:
        """Hard rule: spine/LLM umbrella must yield to a topic-specific leaf."""
        if not cur:
            return None
        best = _pick_best_topic_leaf(
            pool,
            evidence,
            query,
            entry=entry,
            spine_set=spine_set,
            closure_set=closure_for_promote,
            search_rank=search_rank or None,
        )
        if best and best[0] != cur and _specificity(best[0]) > _specificity(cur):
            return best[:1]
        if search_rank and (
            cur == llm_focus or cur in spine_set
        ):
            promoted = _best_search_closure_topic_leaf(
                root=cur,
                closure_set=closure_for_promote,
                search_rank=search_rank,
                entry=entry,
                spine_set=spine_set,
            )
            if promoted and promoted != cur:
                return [promoted]
        if best and best[0] != cur and _is_service_or_client_leaf(best[0], entry=entry):
            if cur in spine_set and (
                best[0] not in spine_set or _specificity(best[0]) >= _specificity(cur)
            ):
                return best[:1]
        return None

    if not focus_cms or _specificity(pick) >= _specificity(focus_cms[0]):
        # Keep an existing non-spine / retrieval leaf over a spine umbrella parent
        # that only wins via incidental summary CJK overlap (no leaf mysql row yet).
        if focus_cms and pick != focus_cms[0]:
            cur = focus_cms[0]
            if (
                pick in spine_set
                and cur not in spine_set
                and entry
                and cur != entry
                and "Controller#" not in cur
            ):
                # Only replace leaf if pick is a clearly better method-name match.
                if _method_name_topic_bonus(pick, query) <= _method_name_topic_bonus(
                    cur, query
                ):
                    return [cur]
            return [pick]
        # pick == focus umbrella (or no focus yet): try leaf promotion first.
        cur = (focus_cms or [pick])[0]
        promoted = _promote_over_umbrella(cur)
        if promoted:
            return promoted
        return [pick]
    # Even if resolve kept the parent first, promote a tighter leaf from alt/pool.
    cur = focus_cms[0]
    promoted = _promote_over_umbrella(cur)
    if promoted:
        return promoted
    return list(focus_cms)[:1]




def _resolve_focus_method_symbol(
    focus_hint: str,
    evidence: dict[str, Any],
    *,
    project_id: str = "",
    topic: str = "",
) -> list[str]:
    """Resolve LLM focus_method/topic to the single most specific leaf method."""
    hint = (focus_hint or "").strip()
    topic_s = (topic or "").strip()
    if not hint and not topic_s:
        return []

    spine = _spine_cms_for_focus(evidence)
    candidates: list[str] = list(spine)
    seen: set[str] = set(candidates)
    closure_set: set[str] = set()
    entry = _simple_cm(str(evidence.get("entry") or ""))

    # One-hop / closure from locked entry (cached; do not fan out per spine root).
    pid = project_id or str(evidence.get("project_id") or "")
    if pid:
        chain = _entry_chain_cms_set(evidence, project_id=pid)
        for cm in chain:
            closure_set.add(cm)
            if cm not in seen:
                seen.add(cm)
                candidates.append(cm)
        try:
            from app.agent.ask_retrieval import _search_hit_cms_for_question

            q = " ".join(x for x in (hint, topic_s) if x).strip()
            for h in _search_hit_cms_for_question(pid, q, top_k=12):
                cm = _simple_cm(h)
                if not cm or "#" not in cm or cm in seen:
                    continue
                if chain and cm not in chain:
                    continue
                seen.add(cm)
                candidates.append(cm)
        except Exception:
            logger.debug("focus resolve search failed", exc_info=True)

    # Exact Class#method: accept immediately only when no competing topic,
    # or when the hint itself is already the tightest topic match.
    if hint and "#" in hint:
        want = _simple_cm(hint)
        if want in seen or want in spine:
            if not topic_s:
                return [want]
            # Topic present: still score pool so a tighter leaf can win.
            if want not in seen:
                seen.add(want)
                candidates.append(want)
    if hint:
        meth = hint.split("#", 1)[-1].split("(", 1)[0].strip()
        if len(meth) >= 3 and not topic_s:
            hits = [
                cm
                for cm in candidates
                if cm.split("#", 1)[-1].split("(", 1)[0].lower() == meth.lower()
            ]
            if not hits:
                hits = [
                    cm
                    for cm in spine
                    if cm.split("#", 1)[-1].split("(", 1)[0].lower() == meth.lower()
                ]
            if hits:
                # Prefer ServiceImpl/Client leaf over Controller when ambiguous.
                hits.sort(
                    key=lambda c: (
                        0 if "ServiceImpl#" in c or "Client#" in c else 1,
                        0 if c in closure_set else 1,
                        c,
                    )
                )
                return hits[:1]

    query = f"{hint} {topic_s}".strip()
    return _pick_best_topic_leaf(
        candidates,
        evidence,
        query,
        entry=entry,
        spine_set=set(spine),
        closure_set=closure_set or None,
        hint=hint if hint and "#" not in hint else "",
    )






# Monkeypatch-sensitive flow functions kept here so tests can
# monkeypatch app.agent.pev_agent symbols (pev_flow uses lazy wrappers).
async def decide_dig_in_focus(
    question: str,
    evidence: dict[str, Any],
    *,
    prior_answer: str = "",
    timeout: float = 12.0,
) -> dict[str, Any]:
    """LLM JSON: dig_in / focus_method / topic / reason (language-agnostic).

    Under lock dig-in, callers should try description match first and pass a
    short ``timeout`` so this never blocks the cheap path.
    """
    import asyncio

    default: dict[str, Any] = {
        "dig_in": False,
        "focus_method": "",
        "topic": "",
        "reason": "fallback",
        "source": "default",
    }
    cands = _spine_candidates_for_llm(evidence)
    if not (question or "").strip() or not cands:
        return default
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        wait_s = max(1.0, float(timeout or 12.0))

        async def _call() -> str:
            return await llm.chat(
                [
                    {"role": "system", "content": _DIG_IN_FOCUS_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Entry (resolved or locked): "
                            f"{evidence.get('entry') or ''}\n"
                            f"Ask scope: "
                            f"{evidence.get('ask_scope') or '(unspecified)'}\n"
                            f"Entry anchor hint: "
                            f"{evidence.get('entry_anchor') or evidence.get('entry_hint') or '(none)'}\n"
                            f"Method topic / focus topic: "
                            f"{evidence.get('method_topic') or evidence.get('focus_topic') or '(none)'}\n"
                            f"Prior answer snippet:\n"
                            f"{truncate(prior_answer or '(empty)', 900)}\n\n"
                            f"Question:\n{question}\n\n"
                            f"Spine candidates (Class#method + summary):\n"
                            f"{json.dumps(cands, ensure_ascii=False)}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=220,
            )

        raw = await asyncio.wait_for(_call(), timeout=wait_s)
        data = _parse_json_obj(raw or "")
        if not data:
            return {
                **default,
                "reason": "unparseable",
                "source": "error",
            }
        dig = data.get("dig_in")
        dig_in = dig is True or str(dig).strip().lower() in {"1", "true", "yes"}
        return {
            "dig_in": dig_in,
            "focus_method": str(data.get("focus_method") or "").strip()[:120],
            "topic": str(data.get("topic") or "").strip()[:120],
            "reason": str(data.get("reason") or "").strip()[:200],
            "source": "llm",
        }
    except asyncio.TimeoutError:
        logger.warning("decide_dig_in_focus timed out")
        return {
            "dig_in": False,
            "focus_method": "",
            "topic": "",
            "reason": "timeout",
            "source": "timeout",
        }
    except Exception:
        logger.debug("decide_dig_in_focus failed", exc_info=True)
        return {
            "dig_in": False,
            "focus_method": "",
            "topic": "",
            "reason": "error",
            "source": "error",
        }




def _related_cms_under_focus(
    project_id: str,
    focus_cms: list[str],
    question: str,
    evidence: dict[str, Any],
    *,
    topic: str = "",
) -> list[str]:
    """Find callees under focused method(s) that match the dig-in sub-topic."""
    needles = _topic_needles_from_question(question, focus_cms, topic=topic)
    if not project_id or not focus_cms:
        return []

    from app.agent.ask_retrieval import (
        _calls_closure_cms,
        _search_hit_cms_for_question,
    )

    focus_set = {_simple_cm(c) for c in focus_cms}
    closure: set[str] = set()
    for fc in focus_cms:
        for c in _calls_closure_cms(project_id, fc)[:80]:
            cm = _simple_cm(c)
            if cm and cm not in focus_set:
                closure.add(cm)
    # Same-class calls from source of focus methods.
    try:
        from app.agent.explore_source import _fetch_method_source_body
        from app.agent.explore_tools import extract_same_class_calls

        for fc in focus_cms[:2]:
            code, _path = _fetch_method_source_body(project_id, fc, max_chars=8000)
            if not code:
                continue
            cls = fc.split("#", 1)[0].split(".")[-1]
            meth = fc.split("#", 1)[1]
            for name in extract_same_class_calls(code, current_method=meth)[:30]:
                closure.add(_simple_cm(f"{cls}#{name}"))
    except Exception:
        logger.debug("dig-in same_class_calls failed", exc_info=True)

    # Search hits that land inside the focus closure win first.
    hits = _search_hit_cms_for_question(project_id, question, top_k=16)
    related: list[str] = []
    seen: set[str] = set()
    focus_classes = {
        fc.split("#", 1)[0].split(".")[-1].lower()
        for fc in focus_cms
        if "#" in fc
    }

    def _prefer_key(cm: str, score: int) -> tuple:
        cls = cm.split("#", 1)[0].split(".")[-1].lower()
        same = 0 if cls in focus_classes else 1
        return (-score, same, cm)

    hit_scored: list[tuple[int, str]] = []
    for h in hits:
        cm = _simple_cm(h)
        if not cm or cm in focus_set or cm not in closure:
            continue
        row = _row_for_cm(evidence, cm) or {}
        summary = str(row.get("summary") or row.get("neo4j_desc") or "")
        sc = max(1, _score_topic_hit(cm, summary, needles))
        hit_scored.append((sc, cm))
    for _sc, cm in sorted(hit_scored, key=lambda x: _prefer_key(x[1], x[0])):
        if cm not in seen:
            seen.add(cm)
            related.append(cm)

    # Score remaining callees only — never pull spine parents via summary overlap.
    scored: list[tuple[int, str]] = []
    for cm in list(closure):
        if not cm or cm in focus_set or cm in seen:
            continue
        row = _row_for_cm(evidence, cm) or {}
        summary = str(row.get("summary") or row.get("neo4j_desc") or "")
        sc = _score_topic_hit(cm, summary, needles)
        if sc <= 0 and needles:
            sc = _score_topic_hit(cm, "", needles)
        if sc > 0:
            scored.append((sc, cm))
    for _sc, cm in sorted(scored, key=lambda x: _prefer_key(x[1], x[0])):
        if cm not in seen:
            seen.add(cm)
            related.append(cm)
        if len(related) >= 4:
            break
    return related[:4]




def _enrich_focus_rows(
    project_id: str, evidence: dict[str, Any], cms: list[str]
) -> None:
    """Ensure mysql_rows contain summaries for focus + related methods."""
    if not project_id or not cms:
        return
    from app.agent.ask_retrieval import _mysql_enrich_ordered

    have = {
        _simple_cm(str(r.get("class_method") or ""))
        for r in (evidence.get("mysql_rows") or [])
        if isinstance(r, dict)
    }
    missing = [c for c in cms if _simple_cm(c) not in have]
    if not missing:
        return
    nodes = [
        {
            "order": i,
            "class_method": _simple_cm(c),
            "rel": "DIG_IN",
            "title": "",
            "desc": "",
        }
        for i, c in enumerate(missing, start=1)
    ]
    try:
        rows = _mysql_enrich_ordered(project_id, nodes, max_nodes=max(8, len(nodes)))
    except Exception:
        logger.debug("dig-in mysql enrich failed", exc_info=True)
        return
    existing = list(evidence.get("mysql_rows") or [])
    evidence["mysql_rows"] = existing + list(rows or [])




def _render_focus_source_notes(
    project_id: str, cm: str, *, max_lines: int = 12
) -> list[str]:
    """Condensed implementation notes from method source (branches / calls / returns)."""
    if not project_id or not cm:
        return []
    try:
        from app.agent.explore_source import _fetch_method_source_body

        code, path = _fetch_method_source_body(project_id, cm, max_chars=7000)
    except Exception:
        return []
    if not (code or "").strip():
        return []
    notes: list[str] = []
    if path:
        notes.append(f"  源码：`{path}`")
    # Client / Feign / service calls
    calls = re.findall(
        r"(?m)^\s*(?:return\s+)?(\w+)\.(\w+)\s*\(",
        code,
    )
    seen_calls: set[str] = set()
    for obj, meth in calls[:8]:
        if obj.lower() in {"if", "for", "while", "switch", "catch", "super", "this"}:
            continue
        if re.match(r"(?i)^(log|logger|string|objects|optional)$", obj):
            continue
        key = f"{obj}.{meth}"
        if key in seen_calls:
            continue
        seen_calls.add(key)
        notes.append(f"  调用：`{obj}.{meth}(...)`")
        if len(seen_calls) >= 4:
            break
    # Branch conditions (short)
    for m in re.finditer(r"(?m)^\s*if\s*\((.{8,120}?)\)\s*\{?", code):
        cond = re.sub(r"\s+", " ", m.group(1)).strip()
        if len(cond) > 100:
            cond = cond[:100] + "…"
        notes.append(f"  分支：if ({cond})")
        if sum(1 for n in notes if n.startswith("  分支")) >= 3:
            break
    # Failure / response returns
    for m in re.finditer(
        r"(?i)(return\s+[^;]{0,80}|setResponse\w*\s*\([^;]{0,60})",
        code,
    ):
        frag = re.sub(r"\s+", " ", m.group(0)).strip()
        if len(frag) > 90:
            frag = frag[:90] + "…"
        notes.append(f"  返回/响应：`{frag}`")
        if sum(1 for n in notes if n.startswith("  返回")) >= 2:
            break
    # Field / map assembly cues (language-agnostic symbols)
    for key in list(
        dict.fromkeys(
            re.findall(r'(?:req|reqMap|body)?\.?get\("(\w+)"\)', code)
            + re.findall(r'\.put\("(\w+)"', code)
        )
    )[:6]:
        notes.append(f"  键：`{key}`")
    return notes[:max_lines]






async def _exec_one_pev_tool(
    tool: str,
    *,
    project_id: str,
    user_q: str,
    evidence: dict[str, Any],
    goal: dict[str, Any],
    hist: str = "",
    entry_hint: str | None = None,
    answer_steps: list | None = None,
    flow_symbols: list | None = None,
    answer_so_far: str = "",
) -> str:
    """Execute a single PEV tool; return updated answer string."""
    from app.agent.runtime.registry import known_tool
    from app.agent.runtime.profile import get_runtime_profile

    if tool and not known_tool(tool):
        raise ValueError(f"unknown tool {tool}")
    prof = get_runtime_profile()
    if tool and not prof.tool_allowed(tool):
        raise ValueError(f"tool {tool} disabled in runtime_mode={prof.mode}")
    answer = answer_so_far
    if tool == "resolve_entry":
        await _exec_resolve_entry(project_id, user_q, evidence)
        if not evidence.get("entry") and entry_hint:
            from app.agent.explore_agent import _class_method_from_hint

            evidence["entry"] = _class_method_from_hint(entry_hint) or ""
    elif tool == "expand_calls":
        await _exec_expand_calls(project_id, evidence)
    elif tool == "mysql_enrich":
        await _exec_mysql_enrich(project_id, evidence)
    elif tool == "summarize_flow":
        answer = await _exec_summarize_flow(
            user_q,
            evidence,
            project_id=project_id,
            history_text=hist,
        )
    elif tool == "discover_change":
        await _exec_discover_change(project_id, user_q, evidence, goal=goal)
    elif tool == "find_outbound":
        await _exec_find_outbound(project_id, user_q, evidence)
    elif tool == "render_change":
        part = await _exec_render_change(user_q, goal, evidence)
        evidence["change_plan_answer"] = part
        answer = _merge_pev_answers(answer, part, heading="## 已有接口改动清单")
    elif tool == "load_kb_spec":
        await _exec_load_kb_spec(user_q, evidence)
    elif tool == "find_api_style":
        await _exec_find_api_style(project_id, evidence)
    elif tool == "render_new_api":
        part = await _exec_render_new_api(user_q, evidence)
        evidence["add_api_answer"] = part
        answer = _merge_pev_answers(answer, part, heading="## 新增接口（知识库脚手架）")
    elif tool == "source_view":
        if (
            not evidence.get("entry")
            and str(evidence.get("view_kind") or "")
            not in {"db_lookup", "db_pipeline"}
            and str(goal.get("goal") or "") != "compose_code"
        ):
            await _exec_resolve_entry(project_id, user_q, evidence)
        answer = await _exec_source_view(
            project_id,
            user_q,
            evidence,
            history_text=hist,
            answer_steps=answer_steps,
            flow_symbols=flow_symbols,
        )
    elif tool == "compose_code":
        answer = await _exec_compose_code(
            project_id, user_q, evidence, history_text=hist
        )
    elif tool == "api_inventory":
        answer = await _exec_api_inventory(project_id, user_q)
    elif tool == "repo_files":
        answer = await _exec_repo_files(project_id, user_q, goal, evidence)
    elif tool == "codebase_architecture":
        answer = await _exec_codebase_architecture(project_id, user_q, evidence)
    elif tool == "symbol_probe":
        answer = await _exec_symbol_probe(project_id, user_q, evidence)
    elif tool == "topic_existence":
        answer = await _exec_topic_existence(project_id, user_q, evidence)
    elif tool == "parse_signal":
        await _exec_parse_signal(user_q, evidence)
    elif tool == "map_and_read":
        await _exec_map_and_read(project_id, user_q, evidence)
    elif tool == "diagnose_answer":
        answer = await _exec_diagnose_answer(user_q, evidence)
    elif tool == "collect_field_evidence":
        await _exec_collect_field_evidence(project_id, user_q, evidence)
    elif tool == "render_field_answer":
        answer = _exec_render_field_answer(user_q, evidence)
    elif tool == "collect_api_io":
        await _exec_collect_api_io(project_id, evidence)
    elif tool == "render_api_io":
        answer = _exec_render_api_io(user_q, evidence)
    elif tool == "search_hits":
        await _exec_search_hits(project_id, user_q, evidence)
    elif tool == "general_answer":
        if evidence.get("entry"):
            if not evidence.get("calls"):
                await _exec_expand_calls(project_id, evidence)
            if not evidence.get("mysql_rows"):
                await _exec_mysql_enrich(project_id, evidence)
        else:
            await _exec_search_hits(project_id, user_q, evidence)
        answer = await _exec_general_answer(
            user_q, evidence, history_text=hist
        )
    else:
        raise ValueError(f"unknown tool {tool}")
    return answer


async def _run_subgoal(
    *,
    kind: str,
    sub_q: str,
    project_id: str,
    evidence: dict[str, Any],
    hist: str = "",
    entry_hint: str | None = None,
    answer_steps: list | None = None,
    flow_symbols: list | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Run one decomposed subgoal against shared evidence; return (answer, tasks)."""
    goal: dict[str, Any] = {
        "goal": kind,
        "must": [],
        "also": [],
        "source": "decompose",
        "tables": extract_named_tables(sub_q),
    }
    if kind == "field_value" or wants_field_value_ask_safe(sub_q):
        goal["goal"] = "field_value"
    tasks = build_task_plan(goal)
    # Reuse shared entry/spine — skip re-resolve when already locked.
    filtered: list[dict[str, Any]] = []
    for t in tasks:
        tool = t.get("tool")
        if tool == "resolve_entry" and evidence.get("entry"):
            t = {**t, "status": "done", "note": "reuse_shared_entry"}
            filtered.append(t)
            continue
        if tool == "expand_calls" and evidence.get("calls"):
            t = {**t, "status": "done", "note": "reuse_shared_spine"}
            filtered.append(t)
            continue
        filtered.append(t)

    answer = ""
    for t in filtered:
        if t.get("status") == "done":
            continue
        tool = str(t.get("tool") or "")
        if t.get("status") == "skipped":
            continue
        t["status"] = "running"
        try:
            answer = await _exec_one_pev_tool(
                tool,
                project_id=project_id,
                user_q=sub_q,
                evidence=evidence,
                goal=goal,
                hist=hist,
                entry_hint=entry_hint,
                answer_steps=answer_steps,
                flow_symbols=flow_symbols,
                answer_so_far=answer,
            )
            t["status"] = "done"
        except Exception as e:
            logger.exception("decompose subgoal tool failed %s", tool)
            t["status"] = "failed"
            t["error"] = str(e)
    return answer, filtered


async def _run_decomposed_pipeline(
    user_q: str,
    *,
    project_id: str,
    hist: str,
    entry_hint: str | None,
    lock_entry: bool,
    answer_steps: list | None,
    flow_symbols: list | None,
    on_progress: ProgressCb,
    decomp: dict[str, Any],
    dig_in_likely: bool = False,
) -> dict[str, Any]:
    """Shared-entry multi-subtask PEV for complex questions."""
    from app.agent.decompose import compose_subtask_answers

    steps: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    phase = {
        "goal": "done",
        "plan": "running",
        "exec": "pending",
        "verify": "pending",
    }
    notes: dict[str, str] = {
        "goal": f"{_tag('GOAL', 'decompose')} src={decomp.get('source')}",
    }
    evidence: dict[str, Any] = {
        "question": user_q,
        "entry": "",
        "prior_question": "",
        "entry_hint": (entry_hint or "").strip(),
        "lock_entry": bool(lock_entry),
        "dig_in_likely": bool(dig_in_likely),
        "project_id": project_id,
        "flow_symbols": list(flow_symbols or []),
        "answer_steps": list(answer_steps or []),
        "decompose": True,
    }

    async def _emit_live(
        *,
        etype: str,
        step_id: str | None = None,
        status: str | None = None,
        note: str | None = None,
    ) -> None:
        plan = _live_pev_plan(
            tasks,
            phase=phase,
            notes=notes,
            entry=evidence.get("entry") or None,
            spine_source=evidence.get("spine_source") or None,
        )
        plan["title"] = f"{_tag('TEST')} decompose"
        event: dict[str, Any] = {"type": etype, "plan": plan}
        if step_id is not None:
            event["step_id"] = step_id
        if status is not None:
            event["status"] = status
        if note is not None:
            event["note"] = note
        await _emit(on_progress, event)

    subtasks = list(decomp.get("tasks") or [])
    steps.append(
        {
            "step": 0,
            "tool": "pev_goal",
            "ok": True,
            "goal": "decompose",
            "subtasks": [
                {"id": t.get("id"), "kind": t.get("kind"), "title": t.get("title")}
                for t in subtasks
            ],
            "tag": _tag("GOAL", "decompose"),
        }
    )

    from app.agent.runtime.orchestrator import serial_groups

    share_entry = bool(decomp.get("share_entry", True))
    groups = serial_groups(subtasks, share_entry=share_entry)

    # Plan UI tasks: one spine lock per entry group
    if share_entry or len(groups) <= 1:
        tasks.append(
            {
                "id": "T0_shared",
                "title": "锁定共享入口与主链",
                "tool": "resolve_entry+expand_calls",
                "status": "pending",
            }
        )
    else:
        for gi, grp in enumerate(groups):
            hint = str((grp[0].get("entry_hint") if grp else "") or "")
            tasks.append(
                {
                    "id": f"T0_g{gi + 1}",
                    "title": f"锁定入口 {hint or gi + 1}",
                    "tool": "resolve_entry+expand_calls",
                    "status": "pending",
                }
            )
    for st in subtasks:
        tasks.append(
            {
                "id": str(st.get("id") or st.get("kind")),
                "title": str(st.get("title") or st.get("kind")),
                "tool": str(st.get("kind")),
                "status": "pending",
            }
        )
    phase["plan"] = "done"
    notes["plan"] = (
        f"{_tag('PLAN')} decompose → "
        + " → ".join(str(t.get("id")) for t in tasks)
    )
    await _emit_live(etype="step", step_id="plan", status="done", note=notes["plan"])
    steps.append(
        {
            "step": 1,
            "tool": "pev_plan",
            "ok": True,
            "tasks": [{"id": t["id"], "tool": t["tool"]} for t in tasks],
            "tag": _tag("PLAN"),
        }
    )

    phase["exec"] = "running"
    from app.agent.explore_agent import _class_method_from_hint

    def _fresh_evidence(hint: str) -> dict[str, Any]:
        return {
            "question": user_q,
            "entry": "",
            "prior_question": "",
            "entry_hint": (hint or entry_hint or "").strip(),
            "lock_entry": bool(lock_entry),
            "dig_in_likely": bool(dig_in_likely),
            "project_id": project_id,
            "flow_symbols": list(flow_symbols or []),
            "answer_steps": list(answer_steps or []),
            "decompose": True,
        }

    parts: list[dict[str, Any]] = []
    group_entries: list[str] = []
    step_i = 0
    hist_work = hist or ""
    for gi, group in enumerate(groups):
        g_hint = str((group[0].get("entry_hint") if group else "") or "") or (
            entry_hint or ""
        )
        ev = (
            evidence
            if (share_entry or len(groups) == 1)
            else _fresh_evidence(g_hint)
        )
        if g_hint and not ev.get("entry_hint"):
            ev["entry_hint"] = g_hint
        t0_id = (
            "T0_shared"
            if (share_entry or len(groups) <= 1)
            else f"T0_g{gi + 1}"
        )
        t0 = next((t for t in tasks if t.get("id") == t0_id), None)
        if t0:
            t0["status"] = "running"
        await _emit_live(
            etype="step",
            step_id=t0_id,
            status="running",
            note=f"resolve_entry + expand_calls hint={g_hint or '-'}",
        )
        seed_q = str((group[0].get("question") if group else "") or user_q)
        await _exec_resolve_entry(project_id, seed_q, ev)
        if not ev.get("entry") and (g_hint or entry_hint):
            ev["entry"] = _class_method_from_hint(g_hint or entry_hint) or ""
        if ev.get("entry"):
            await _exec_expand_calls(project_id, ev)
        if t0:
            t0["status"] = "done"
            t0["note"] = f"entry={ev.get('entry') or '-'}"
        await _emit_live(
            etype="step",
            step_id=t0_id,
            status="done",
            note=(t0 or {}).get("note") or "",
        )
        steps.append(
            {
                "step": 10 + gi,
                "tool": "pev_shared_entry" if gi == 0 else "pev_group_entry",
                "ok": bool(ev.get("entry")),
                "entry": ev.get("entry"),
                "symbols": list(ev.get("calls") or [])[:20],
                "tag": _tag("EXEC", t0_id, "OK"),
            }
        )
        group_entries.append(str(ev.get("entry") or ""))
        if gi == 0:
            evidence["entry"] = ev.get("entry") or evidence.get("entry") or ""
            evidence["calls"] = list(ev.get("calls") or evidence.get("calls") or [])
            evidence["calls_count"] = ev.get("calls_count") or evidence.get(
                "calls_count"
            )
            evidence["spine_source"] = ev.get("spine_source") or evidence.get(
                "spine_source"
            )

        for st in group:
            tid = str(st.get("id") or f"S{step_i + 1}")
            kind = str(st.get("kind") or st.get("skill") or "general")
            title = str(st.get("title") or kind)
            sub_q = str(st.get("question") or user_q)
            ui = next((t for t in tasks if t.get("id") == tid), None)
            if ui:
                ui["status"] = "running"
            await _emit_live(
                etype="step",
                step_id=tid,
                status="running",
                note=f"{kind}: {title}",
            )
            try:
                ans, sub_tools = await _run_subgoal(
                    kind=kind,
                    sub_q=sub_q,
                    project_id=project_id,
                    evidence=ev,
                    hist=hist_work,
                    entry_hint=g_hint or entry_hint,
                    answer_steps=answer_steps,
                    flow_symbols=flow_symbols,
                )
                parts.append(
                    {
                        "id": tid,
                        "kind": kind,
                        "title": title,
                        "question": sub_q,
                        "answer": ans,
                        "ok": bool((ans or "").strip()),
                        "entry": ev.get("entry") or "",
                    }
                )
                if (ans or "").strip():
                    hist_work = (
                        f"{hist_work}\nuser: {sub_q}\nassistant: {ans[:1500]}"
                    ).strip()
                if ui:
                    ui["status"] = "done"
                    ui["note"] = f"ok chars={len(ans or '')}"
                steps.append(
                    {
                        "step": 20 + step_i,
                        "tool": f"pev_sub_{kind}",
                        "ok": True,
                        "task_id": tid,
                        "entry": ev.get("entry"),
                        "tag": _tag("EXEC", tid, "OK"),
                    }
                )
                await _emit_live(
                    etype="step",
                    step_id=tid,
                    status="done",
                    note=(ui or {}).get("note") or "done",
                )
                del sub_tools
            except Exception as e:
                logger.exception("decompose subtask failed %s", tid)
                parts.append(
                    {
                        "id": tid,
                        "kind": kind,
                        "title": title,
                        "question": sub_q,
                        "answer": f"分节执行失败：{e}",
                        "ok": False,
                        "entry": ev.get("entry") or "",
                    }
                )
                if ui:
                    ui["status"] = "failed"
                    ui["error"] = str(e)
                await _emit_live(
                    etype="step",
                    step_id=tid,
                    status="error",
                    note=str(e)[:120],
                )
            step_i += 1

    phase["exec"] = "done"
    answer = compose_subtask_answers(
        parts,
        entry=str(evidence.get("entry") or ""),
        compose_note=str(decomp.get("compose") or ""),
    )
    goal = {
        "goal": "decompose",
        "must": [{"id": "compose", "desc": "分节合成"}],
        "also": [str(p.get("kind")) for p in parts],
        "source": decomp.get("source") or "heuristic",
        "share_entry": share_entry,
    }
    phase["verify"] = "running"
    gaps: list[dict[str, Any]] = []
    any_entry = any(e for e in group_entries) or bool(evidence.get("entry"))
    if share_entry and not any_entry:
        gaps.append({"id": "entry", "desc": "未锁定共享业务入口"})
    if sum(1 for p in parts if p.get("ok")) < 1:
        gaps.append({"id": "compose", "desc": "无成功分节"})
    if any_entry and "#" not in answer:
        gaps.append({"id": "method_anchors", "desc": "合成答未含 Class#method 锚点"})
    verify = {"ok": len(gaps) == 0, "gaps": gaps, "checked": ["entry", "compose"]}
    phase["verify"] = "done"
    notes["verify"] = (
        f"{_tag('VERIFY', 'OK' if verify['ok'] else 'GAP')} gaps={len(gaps)}"
    )
    await _emit_live(
        etype="step",
        step_id="verify",
        status="done",
        note=notes["verify"],
    )
    pev_debug = _build_pev_debug(goal, tasks, verify, evidence)
    if _pev_show_tags():
        if gaps:
            answer = (
                answer.rstrip()
                + "\n\n## [PEV][VERIFY][GAP] 契约缺口\n\n"
                + "\n".join(f"- `{g['id']}`：{g['desc']}" for g in gaps)
            )
        answer = answer.rstrip() + pev_debug["footer"]
        note = evidence.get("outbound_focus_note")
        if note:
            answer = f"{answer}\n\n{note}"
    else:
        # Internal debug tags/gaps never leak into the user-visible reply.
        answer = answer.rstrip()
        if evidence.get("outbound_focus_note"):
            pev_debug["outbound_focus_note"] = evidence["outbound_focus_note"]

    plan = _live_pev_plan(
        tasks,
        phase={
            "goal": "done",
            "plan": "done",
            "exec": "done",
            "verify": "done",
        },
        notes=notes,
        entry=evidence.get("entry") or None,
        spine_source=evidence.get("spine_source") or None,
    )
    plan["goal"] = goal
    plan["tasks"] = tasks
    plan["verify"] = verify
    plan["title"] = f"{_tag('TEST')} decompose"
    plan["decompose"] = {
        "tasks": subtasks,
        "source": decomp.get("source"),
    }

    retrieved = []
    if evidence.get("entry"):
        retrieved.append(
            {
                "fqcn": str(evidence["entry"]).split("#")[0],
                "kind": "Controller",
                "summary": f"PEV decompose entry {evidence['entry']}",
            }
        )
    return {
        "answer": answer,
        "steps": steps,
        "retrieved": retrieved,
        "plan": plan,
        "pev_debug": pev_debug,
        "review": {
            "ok": bool(verify.get("ok")),
            "source": "pev",
            "gaps": verify.get("gaps"),
        },
        "evidence": {
            "entry": evidence.get("entry"),
            "calls": list(evidence.get("calls") or [])[:20],
            "calls_count": evidence.get("calls_count"),
            "spine_source": evidence.get("spine_source"),
            "decompose": True,
            "lock_entry": bool(evidence.get("lock_entry")),
        },
    }


async def run_pev_pipeline(
    question: str,
    *,
    project_id: str,
    history_text: str | None = None,
    prior_question: str | None = None,
    prior_goal: str | None = None,
    entry_hint: str | None = None,
    lock_entry: bool = False,
    dig_in_likely: bool = False,
    answer_steps: list[dict[str, Any]] | None = None,
    flow_symbols: list[str] | None = None,
    prefer_cms: list[str] | None = None,
    on_progress: ProgressCb = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Run Goal → Plan → Execute → Verify → Answer (with [PEV][TEST] tags)."""
    steps: list[dict[str, Any]] = []
    user_q = (question or "").strip()
    hist = history_text or ""
    if not project_id or not user_q:
        return {
            "answer": "缺少 project_id 或问题为空。",
            "steps": steps,
            "plan": {"kind": "pev", "title": "PEV"},
            "review": {"ok": False, "source": "pev"},
        }

    # Complex multi-intent → LLM DAG (structural multi-segment gate).
    # Minimal runtime mode skips decompose for eval-stable single-goal runs.
    try:
        from app.agent.decompose import decompose_question, wants_decompose
        from app.agent.runtime.profile import get_runtime_profile

        if get_runtime_profile().allow_decompose and wants_decompose(user_q):
            decomp = await decompose_question(user_q, history_text=hist)
            if decomp and len(decomp.get("tasks") or []) >= 2:
                return await _run_decomposed_pipeline(
                    user_q,
                    project_id=project_id,
                    hist=hist,
                    entry_hint=entry_hint,
                    lock_entry=lock_entry,
                    dig_in_likely=dig_in_likely,
                    answer_steps=answer_steps,
                    flow_symbols=flow_symbols,
                    on_progress=on_progress,
                    decomp=decomp,
                )
    except Exception:
        logger.debug("decompose path skipped", exc_info=True)

    phase = {
        "goal": "running",
        "plan": "pending",
        "exec": "pending",
        "verify": "pending",
    }
    notes: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []
    prefer_list = [
        _simple_cm(str(x))
        for x in (prefer_cms or [])
        if x and "#" in str(x)
    ]
    evidence: dict[str, Any] = {
        "question": user_q,
        "entry": "",
        "prior_question": prior_question or "",
        "entry_hint": (entry_hint or "").strip(),
        "lock_entry": bool(lock_entry),
        "dig_in_likely": bool(dig_in_likely),
        "project_id": project_id,
        "flow_symbols": list(flow_symbols or []),
        "answer_steps": list(answer_steps or []),
        "prefer_cms": prefer_list,
    }

    async def _emit_live(
        *,
        etype: str,
        step_id: str | None = None,
        status: str | None = None,
        note: str | None = None,
    ) -> None:
        plan = _live_pev_plan(
            tasks,
            phase=phase,
            notes=notes,
            entry=evidence.get("entry") or None,
            spine_source=evidence.get("spine_source") or None,
        )
        event: dict[str, Any] = {"type": etype, "plan": plan}
        if step_id is not None:
            event["step_id"] = step_id
        if status is not None:
            event["status"] = status
        if note is not None:
            event["note"] = note
        await _emit(on_progress, event)

    await _emit_live(
        etype="plan",
        step_id="goal",
        status="running",
        note="正在分析问题目标（调用模型）…",
    )

    compress_dig_in = _should_compress_lock_dig_in_goal(
        user_q, lock_entry=lock_entry, dig_in_likely=dig_in_likely
    )
    if compress_dig_in:
        goal = _lock_dig_in_goal(entry_hint=entry_hint)
    else:
        goal = await extract_goal_contract(
            user_q, history_text=hist, prior_goal=prior_goal
        )
    if entry_hint and not goal.get("entry_hint"):
        goal["entry_hint"] = entry_hint
    if goal.get("entry_hint") and not evidence.get("entry_hint"):
        evidence["entry_hint"] = str(goal.get("entry_hint") or "").strip()
    # First-turn / Goal ask_scope → enable dig-in without prior session lock.
    ask_scope = str(goal.get("ask_scope") or "").strip().lower()
    if ask_scope == "dig_in":
        ask_scope = "entry_scoped_dig_in"
    if ask_scope in {"entry_scoped_dig_in", "full_flow"}:
        evidence["ask_scope"] = ask_scope
        goal["ask_scope"] = ask_scope
    if ask_scope == "entry_scoped_dig_in" or goal.get("dig_in_focus"):
        evidence["dig_in_likely"] = True
        evidence["dig_in_focus"] = True
        goal["dig_in_focus"] = True
        if not ask_scope:
            evidence["ask_scope"] = "entry_scoped_dig_in"
            goal["ask_scope"] = "entry_scoped_dig_in"
    if goal.get("focus_topic"):
        evidence["focus_topic"] = str(goal.get("focus_topic") or "").strip()[:120]
    if goal.get("method_topic"):
        evidence["method_topic"] = str(goal.get("method_topic") or "").strip()[:120]
        if not evidence.get("focus_topic"):
            evidence["focus_topic"] = evidence["method_topic"]
    elif evidence.get("focus_topic") and not evidence.get("method_topic"):
        evidence["method_topic"] = str(evidence["focus_topic"])
    if goal.get("entry_anchor"):
        evidence["entry_anchor"] = str(goal.get("entry_anchor") or "").strip()[:120]
        if not evidence.get("entry_hint"):
            evidence["entry_hint"] = evidence["entry_anchor"]
            goal["entry_hint"] = evidence["entry_anchor"]
    # Propagate Goal LLM view/lookup slots (no regex classification).
    for k in ("view_kind", "lookup_key", "lookup_table"):
        if goal.get(k):
            evidence[k] = goal[k]
    if goal.get("lookup_steps"):
        evidence["lookup_steps"] = list(goal.get("lookup_steps") or [])
    if goal.get("business_steps"):
        evidence["business_steps"] = list(goal.get("business_steps") or [])
    db_view = str(goal.get("view_kind") or "").strip().lower() in {
        "db_lookup",
        "db_pipeline",
    } or str(goal.get("goal") or "") == "compose_code"
    # show_code / dig-in: Class#method from prefer_cms is the view target —
    # keep session Controller lock separate via lock_entry / entry_hint.
    # Fresh full_flow on a new entry must not inherit a foreign prefer leaf as
    # view_hint (session dig-in pollution).
    # db_lookup/db_pipeline: never pin session Controller as prefer — free search.
    ask_scope0 = str(goal.get("ask_scope") or evidence.get("ask_scope") or "").strip().lower()
    if prefer_list and not db_view:
        evidence["prefer_cms"] = prefer_list
        carry_view = bool(
            evidence.get("lock_entry")
            or evidence.get("dig_in_likely")
            or evidence.get("dig_in_focus")
            or str(goal.get("goal") or "") == "show_code"
            or ask_scope0 in {"entry_scoped_dig_in", "dig_in"}
        )
        if carry_view:
            evidence["view_hint"] = prefer_list[0]
            goal["view_hint"] = prefer_list[0]
    elif prefer_list and db_view:
        # Keep prefer only if caller passed explicit dig-in focus; drop Controller pin.
        evidence["prefer_cms"] = []
        evidence.pop("view_hint", None)
        goal["view_hint"] = None
    else:
        from app.agent.explore_agent import _class_method_from_hint

        hint_cm = _class_method_from_hint(
            str(goal.get("entry_hint") or evidence.get("entry_hint") or "")
        )
        # Soft view_hint ONLY under lock/dig-in/show_code (not db_*).
        # Goal often invents the wrong sibling Controller (e.g. CreditQuota vs Credit);
        # promoting that to view_hint would pin resolve_entry and skip multi-entry search.
        allow_soft_view = bool(
            not db_view
            and (
                evidence.get("lock_entry")
                or evidence.get("dig_in_likely")
                or evidence.get("dig_in_focus")
                or str(goal.get("goal") or "") == "show_code"
                or ask_scope0 in {"entry_scoped_dig_in", "dig_in"}
            )
        )
        if hint_cm and "#" in hint_cm and allow_soft_view:
            evidence["view_hint"] = hint_cm
            goal["view_hint"] = hint_cm
            if (
                str(goal.get("goal") or "") == "show_code"
                and not evidence.get("lock_entry")
            ):
                evidence["prefer_cms"] = [hint_cm]
    # Unlocked full_flow: drop Goal-invented Controller view_hint so free search runs.
    if (
        not evidence.get("lock_entry")
        and not evidence.get("dig_in_likely")
        and not evidence.get("dig_in_focus")
        and str(goal.get("goal") or "") == "explain_flow"
        and ask_scope0 in {"", "full_flow"}
    ):
        vh = str(evidence.get("view_hint") or goal.get("view_hint") or "")
        if _pinnable_controller_entry(vh):
            evidence.pop("view_hint", None)
            goal["view_hint"] = None
        # entry_anchor as Class#method must not masquerade as a pin via entry_hint.
        ea = str(evidence.get("entry_anchor") or goal.get("entry_anchor") or "")
        if _pinnable_controller_entry(ea):
            # Keep natural-language hint from the user question when possible.
            q_hint = str(evidence.get("entry_hint_user") or "").strip()
            phrase = ""
            try:
                from app.analysis.entry_route import interface_ask_phrase

                phrase = interface_ask_phrase(user_q) or ""
            except Exception:
                phrase = ""
            natural = q_hint or phrase or ""
            # Only rewrite when entry_hint was the invented Class#method.
            eh = str(evidence.get("entry_hint") or "")
            if _pinnable_controller_entry(eh):
                evidence["entry_hint"] = natural if natural and "#" not in natural else (phrase or "")
                goal["entry_hint"] = evidence["entry_hint"]
    # Dig-in under lock + 「要改什么」: stay on locked spine (explain + focus).
    # Session lock only — not a keyword→goal table (LLM already chose goal).
    if (
        not compress_dig_in
        and lock_entry
        and _wants_dig_in_change(user_q)
        and str(goal.get("goal") or "") in {"general", "change_plan", "explain_flow", "compound"}
    ):
        goal["goal"] = "explain_flow"
        goal["must"] = [
            {"id": "entry", "desc": "沿用锁定主入口"},
            {"id": "spine", "desc": "展开锁定入口主链"},
            {"id": "summarize", "desc": "聚焦主链上命中的方法作答"},
        ]
        goal["also"] = []
        goal["dig_in_focus"] = True
        if goal.get("source") == "llm":
            goal["source"] = "llm+dig_in_lock"
        else:
            goal["source"] = "heuristic+dig_in_lock"
    # Locked dig-in: reuse session spine before plan/exec (skip T1–T3 when seeded).
    if lock_entry and (dig_in_likely or compress_dig_in or _wants_dig_in_change(user_q)):
        _seed_reuse_locked_spine(evidence)
    # Do NOT regex-force field_value over LLM goal (LLM owns routing).
    steps.append(
        {
            "step": 0,
            "tool": "pev_goal",
            "ok": True,
            "goal": goal.get("goal"),
            "must": goal.get("must"),
            "tag": _tag("GOAL", str(goal.get("goal"))),
        }
    )
    phase["goal"] = "done"
    notes["goal"] = (
        f"{_tag('GOAL', str(goal.get('goal')))} "
        f"must={len(goal.get('must') or [])} src={goal.get('source')}"
    )
    await _emit_live(
        etype="step",
        step_id="goal",
        status="done",
        note=f"已理解问题（{goal.get('goal') or 'ask'}）",
    )

    tasks = build_task_plan(goal)
    phase["plan"] = "done"
    notes["plan"] = f"{_tag('PLAN')} " + " → ".join(t["id"] for t in tasks)
    await _emit_live(
        etype="step",
        step_id="plan",
        status="done",
        note="已规划检索步骤：" + " → ".join(
            str(t.get("title") or t.get("id") or "") for t in tasks[:8]
        ),
    )
    steps.append(
        {
            "step": 1,
            "tool": "pev_plan",
            "ok": True,
            "tasks": [{"id": t["id"], "tool": t["tool"]} for t in tasks],
            "tag": _tag("PLAN"),
        }
    )

    answer = ""

    phase["exec"] = "running"
    notes["exec"] = f"{_tag('EXEC')} start"
    await _emit_live(
        etype="step",
        step_id="exec",
        status="running",
        note="正在检索代码入口与调用链…",
    )

    for i, task in enumerate(tasks):
        tid = task["id"]
        tool = task["tool"]
        if task.get("status") == "skipped":
            await _emit_live(
                etype="step",
                step_id=tid,
                status="skipped",
                note=str(task.get("note") or f"{_tag('EXEC', tid)} skipped"),
            )
            steps.append(
                {
                    "step": len(steps),
                    "tool": f"pev_{tool}",
                    "ok": True,
                    "skipped": True,
                    "note": task.get("note"),
                    "tag": _tag("EXEC", tid, "SKIP"),
                }
            )
            continue
        task["status"] = "running"
        task["note"] = f"{_tag('EXEC', tid)} {task.get('title')} tool={tool}"
        await _emit_live(
            etype="step",
            step_id=tid,
            status="running",
            note=_user_tool_note(
                tool, "running", evidence=evidence, title=str(task.get("title") or "")
            ),
        )
        try:
            reuse = bool(evidence.get("reuse_shared_spine"))
            if tool == "resolve_entry":
                if reuse and evidence.get("entry"):
                    task["note"] = (
                        f"{_tag('EXEC', tid, 'OK')} reuse_shared_entry"
                        f" entry={evidence.get('entry')}"
                    )
                else:
                    await _exec_resolve_entry(project_id, user_q, evidence)
                    if not evidence.get("entry") and entry_hint:
                        from app.agent.explore_agent import _class_method_from_hint

                        evidence["entry"] = _class_method_from_hint(entry_hint) or ""
            elif tool == "expand_calls":
                if reuse and (evidence.get("calls") or evidence.get("flow_symbols")):
                    if not evidence.get("calls") and evidence.get("flow_symbols"):
                        _seed_reuse_locked_spine(evidence)
                    task["note"] = (
                        f"{_tag('EXEC', tid, 'OK')} reuse_shared_spine"
                        f" calls={evidence.get('calls_count') or 0}"
                    )
                else:
                    await _exec_expand_calls(project_id, evidence, question=user_q)
            elif tool == "mysql_enrich":
                if reuse and (
                    evidence.get("mysql_rows")
                    or evidence.get("dig_in_likely")
                    or evidence.get("answer_steps")
                ):
                    # Dig-in light enrich happens inside summarize_flow / focus render.
                    task["note"] = (
                        f"{_tag('EXEC', tid, 'OK')} reuse_skip_mysql_enrich"
                    )
                else:
                    await _exec_mysql_enrich(project_id, evidence)
            elif tool == "summarize_flow":
                answer = await _exec_summarize_flow(
                    user_q,
                    evidence,
                    project_id=project_id,
                    history_text=hist,
                )
            elif tool == "discover_change":
                await _exec_discover_change(
                    project_id, user_q, evidence, goal=goal
                )
            elif tool == "find_outbound":
                await _exec_find_outbound(project_id, user_q, evidence)
            elif tool == "render_change":
                part = await _exec_render_change(user_q, goal, evidence)
                evidence["change_plan_answer"] = part
                answer = _merge_pev_answers(
                    answer, part, heading="## 已有接口改动清单"
                )
            elif tool == "load_kb_spec":
                await _exec_load_kb_spec(user_q, evidence)
            elif tool == "find_api_style":
                await _exec_find_api_style(project_id, evidence)
            elif tool == "render_new_api":
                part = await _exec_render_new_api(user_q, evidence)
                evidence["add_api_answer"] = part
                answer = _merge_pev_answers(
                    answer, part, heading="## 新增接口（知识库脚手架）"
                )
            elif tool == "source_view":
                # db_* paste: free search by Goal lookup slots — skip entry pin.
                if (
                    not evidence.get("entry")
                    and str(evidence.get("view_kind") or "")
                    not in {"db_lookup", "db_pipeline"}
                    and str(goal.get("goal") or "") != "compose_code"
                ):
                    await _exec_resolve_entry(project_id, user_q, evidence)
                answer = await _exec_source_view(
                    project_id,
                    user_q,
                    evidence,
                    history_text=hist,
                    answer_steps=answer_steps,
                    flow_symbols=flow_symbols,
                )
            elif tool == "compose_code":
                answer = await _exec_compose_code(
                    project_id, user_q, evidence, history_text=hist
                )
            elif tool == "api_inventory":
                answer = await _exec_api_inventory(project_id, user_q)
            elif tool == "repo_files":
                answer = await _exec_repo_files(project_id, user_q, goal, evidence)
            elif tool == "codebase_architecture":
                answer = await _exec_codebase_architecture(
                    project_id, user_q, evidence
                )
            elif tool == "symbol_probe":
                answer = await _exec_symbol_probe(project_id, user_q, evidence)
            elif tool == "topic_existence":
                answer = await _exec_topic_existence(project_id, user_q, evidence)
            elif tool == "parse_signal":
                await _exec_parse_signal(user_q, evidence)
            elif tool == "map_and_read":
                await _exec_map_and_read(project_id, user_q, evidence)
            elif tool == "diagnose_answer":
                answer = await _exec_diagnose_answer(user_q, evidence)
            elif tool == "collect_field_evidence":
                await _exec_collect_field_evidence(project_id, user_q, evidence)
            elif tool == "render_field_answer":
                answer = _exec_render_field_answer(user_q, evidence)
            elif tool == "collect_api_io":
                await _exec_collect_api_io(project_id, evidence)
            elif tool == "render_api_io":
                answer = _exec_render_api_io(user_q, evidence)
            elif tool == "search_hits":
                await _exec_search_hits(project_id, user_q, evidence)
            elif tool == "general_answer":
                if evidence.get("entry"):
                    await _exec_expand_calls(project_id, evidence)
                    await _exec_mysql_enrich(project_id, evidence)
                else:
                    await _exec_search_hits(project_id, user_q, evidence)
                answer = await _exec_general_answer(
                    user_q, evidence, history_text=hist
                )
            else:
                task["status"] = "failed"
                task["error"] = f"unknown tool {tool}"
                task["note"] = task["error"]
                continue
            if tool == "expand_calls" and not (evidence.get("calls_count") or 0):
                task["status"] = "failed"
                task["error"] = "empty_spine"
                task["note"] = (
                    f"{_tag('EXEC', tid, 'FAIL')} entry={evidence.get('entry') or '-'}"
                    f" spine={evidence.get('spine_source') or 'empty'}"
                    f" calls=0"
                )
                steps.append(
                    {
                        "step": 10 + i,
                        "tool": f"pev_{tool}",
                        "ok": False,
                        "task_id": tid,
                        "error": "empty_spine",
                        "tag": _tag("EXEC", tid, "FAIL"),
                        "entry": evidence.get("entry"),
                        "spine_source": evidence.get("spine_source"),
                    }
                )
                await _emit_live(
                    etype="step",
                    step_id=tid,
                    status="error",
                    note=_user_tool_note(tool, "error", title=str(task.get("title") or "")),
                )
                continue
            task["status"] = "done"
            if not task.get("note") or str(task.get("note") or "").startswith(
                f"{_tag('EXEC', tid)} "
            ):
                task["note"] = (
                    f"{_tag('EXEC', tid, 'OK')} entry={evidence.get('entry') or '-'}"
                    + (
                        f" spine={evidence.get('spine_source')}"
                        if evidence.get("spine_source")
                        else ""
                    )
                    + (
                        " reuse_shared_spine"
                        if evidence.get("reuse_shared_spine")
                        and tool
                        in {"resolve_entry", "expand_calls", "mysql_enrich"}
                        else ""
                    )
                )
            steps.append(
                {
                    "step": 10 + i,
                    "tool": f"pev_{tool}",
                    "ok": True,
                    "task_id": tid,
                    "tag": _tag("EXEC", tid, "OK"),
                    "entry": evidence.get("entry"),
                    "spine_source": evidence.get("spine_source"),
                    **(
                        {
                            "dig_in_decision": evidence.get("dig_in_decision") or {},
                            "dig_in_focus_cms": list(
                                evidence.get("dig_in_focus_cms") or []
                            )[:8],
                            "dig_in_likely": bool(evidence.get("dig_in_likely")),
                        }
                        if tool == "summarize_flow"
                        and (
                            evidence.get("dig_in_decision")
                            or evidence.get("dig_in_focus_cms")
                        )
                        else {}
                    ),
                }
            )
            await _emit_live(
                etype="step",
                step_id=tid,
                status="done",
                note=_user_tool_note(
                    tool, "done", evidence=evidence, title=str(task.get("title") or "")
                ),
            )
        except Exception as e:
            logger.exception("PEV task failed %s", tid)
            task["status"] = "failed"
            task["error"] = str(e)
            task["note"] = f"{_tag('EXEC', tid, 'FAIL')} {truncate(str(e), 120)}"
            steps.append(
                {
                    "step": 10 + i,
                    "tool": f"pev_{tool}",
                    "ok": False,
                    "task_id": tid,
                    "error": str(e),
                    "tag": _tag("EXEC", tid, "FAIL"),
                }
            )
            await _emit_live(
                etype="step",
                step_id=tid,
                status="error",
                note=_user_tool_note(tool, "error", title=str(task.get("title") or "")),
            )

    if not (answer or "").strip():
        answer = "未能根据已有代码证据生成回答。"

    phase["exec"] = "done"
    notes["exec"] = f"{_tag('EXEC')} done"
    phase["verify"] = "running"
    verify = verify_contract(goal, evidence, answer)

    # A: repair compose_code when generated code references undefined classes —
    # re-search the undefined names and rewrite anchored on real entities; never
    # ship fabricated code (falls back to an honest gap note).
    if str(goal.get("goal") or "") == "compose_code" and any(
        g.get("id") == "compose_undef_class" for g in (verify.get("gaps") or [])
    ):
        answer, verify = await _repair_compose_undefined(
            project_id, user_q, goal, evidence, answer, verify, hist
        )

    phase["verify"] = "done"
    notes["verify"] = (
        f"{_tag('VERIFY', 'OK' if verify.get('ok') else 'GAP')} "
        f"gaps={len(verify.get('gaps') or [])}"
    )
    await _emit_live(
        etype="step",
        step_id="verify",
        status="done",
        note="回答已核对",
    )
    steps.append(
        {
            "step": 90,
            "tool": "pev_verify",
            "ok": bool(verify.get("ok")),
            "gaps": verify.get("gaps"),
            "tag": _tag("VERIFY"),
        }
    )

    pev_debug = _build_pev_debug(goal, tasks, verify, evidence)
    if _pev_show_tags():
        if verify.get("gaps"):
            gap_lines = [
                "",
                f"## {_tag('VERIFY', 'GAP')} 契约缺口",
                "",
            ]
            for g in verify["gaps"]:
                gap_lines.append(f"- `{g.get('id')}`：{g.get('desc')}")
            # Don't duplicate if answer already refused thinly
            if "契约缺口" not in answer:
                answer = answer.rstrip() + "\n" + "\n".join(gap_lines)
        answer = answer.rstrip() + pev_debug["footer"]
        note = evidence.get("outbound_focus_note")
        if note:
            answer = f"{answer}\n\n{note}"
    else:
        # Internal debug tags/gaps never leak into the user-visible reply.
        answer = answer.rstrip()
        if evidence.get("outbound_focus_note"):
            pev_debug["outbound_focus_note"] = evidence["outbound_focus_note"]

    plan = _live_pev_plan(
        tasks,
        phase={
            "goal": "done",
            "plan": "done",
            "exec": "done",
            "verify": "done",
        },
        notes=notes,
        entry=evidence.get("entry") or None,
        spine_source=evidence.get("spine_source") or None,
    )
    plan["goal"] = goal
    plan["tasks"] = tasks
    plan["verify"] = verify
    plan["title"] = "分析完成"

    retrieved = []
    if evidence.get("entry"):
        retrieved.append(
            {
                "fqcn": str(evidence["entry"]).split("#")[0],
                "kind": "Controller",
                "summary": f"PEV entry {evidence['entry']}",
            }
        )

    return {
        "answer": answer,
        "steps": steps,
        "retrieved": retrieved,
        "plan": plan,
        "pev_debug": pev_debug,
        "review": {
            "ok": bool(verify.get("ok")),
            "source": "pev",
            "gaps": verify.get("gaps"),
        },
        "evidence": {
            "entry": evidence.get("entry"),
            "calls": list(evidence.get("calls") or [])[:20],
            "calls_count": evidence.get("calls_count"),
            "spine_source": evidence.get("spine_source"),
            "intersect_cms": evidence.get("intersect_cms"),
            "change_roles": evidence.get("change_roles"),
            "lock_entry": bool(evidence.get("lock_entry")),
            "ask_scope": str(evidence.get("ask_scope") or ""),
            "focus_topic": str(evidence.get("focus_topic") or ""),
            "dig_in_likely": bool(evidence.get("dig_in_likely")),
            "dig_in_focus": bool(evidence.get("dig_in_focus")),
            "dig_in_focus_cms": list(evidence.get("dig_in_focus_cms") or [])[:8],
            "dig_in_related": list(evidence.get("dig_in_related") or [])[:12],
            "dig_in_decision": evidence.get("dig_in_decision") or {},
            "prefer_cms": list(evidence.get("prefer_cms") or [])[:8],
            "view_hint": evidence.get("view_hint") or "",
            "source_view": evidence.get("source_view"),
            "reuse_shared_spine": bool(evidence.get("reuse_shared_spine")),
        },
    }
