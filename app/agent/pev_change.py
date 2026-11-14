"""PEV change layer extracted from pev_agent.py.

Change-plan execution tools: discover_change / find_outbound / render_change /
source_view / compose_code / api inventory / field & api-io tools. Depends on
pev_agent only via lazy in-function imports (\_simple_cm, \_exec_summarize_flow)
to avoid a module-level cycle; pev_agent re-exports every name.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

logger = logging.getLogger("app.agent.pev_change")


def _simple_cm(cm: str | None) -> str:
    """Lazy re-export from pev_agent to avoid a module-level import cycle."""
    from app.agent.pev_agent import _simple_cm as _impl

    return _impl(cm)


def _extract_field_label_zh(question: str) -> str:
    """Lazy re-export from pev_plan (avoids module-level import cycle)."""
    from app.agent.pev_plan import extract_field_label_zh as _impl

    return _impl(question)


async def _exec_summarize_flow(
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Lazy re-export from pev_agent to avoid a module-level import cycle."""
    from app.agent.pev_agent import _exec_summarize_flow as _impl

    return await _impl(*args, **kwargs)


async def _exec_discover_change(
    project_id: str,
    question: str,
    evidence: dict[str, Any],
    *,
    goal: dict[str, Any] | None = None,
) -> None:
    from app.agent.explore_tools import discover_add_field_plan
    from app.agent.question_intent import assumed_new_field_name

    # Inject explicit field so discover never drafts with placeholder newField
    # when goal/LLM already extracted businessLabel (or regex missed「入参 X」).
    q = question or ""
    field = str((goal or {}).get("field_name") or "").strip()
    if not field or field == "newField":
        field = assumed_new_field_name(q)
    if field and field != "newField" and f"字段名：{field}" not in q:
        q = f"{q}\n（字段名：{field}）"

    dap = discover_add_field_plan(
        project_id,
        q,
        entry_cm=evidence.get("entry") or None,
    )
    evidence["change_result"] = dap
    evidence["change_ok"] = bool(dap.get("ok"))
    if dap.get("entry"):
        evidence["entry"] = dap.get("entry")
    roles = dap.get("roles") or {}
    evidence["change_roles"] = list(roles.keys())
    persist_tables: list[str] = []
    for _rk, meta in roles.items():
        if not str(_rk).startswith("persist_entity"):
            continue
        if isinstance(meta, dict):
            tn = str(meta.get("table_name") or "").strip().lower()
            if tn and tn not in persist_tables:
                persist_tables.append(tn)
    evidence["persist_tables"] = persist_tables


def _outbound_focus_queries(question: str) -> list[str]:
    """Narrow queries for「调用X校验时传递」— full change-plan sentences bury the gate."""
    q = question or ""
    out: list[str] = []
    # 「…校验」phrases from the question.
    for m in re.finditer(r"[\u4e00-\u9fff]{2,12}校验", q):
        out.append(m.group(0))
    # Clause after 然后/并且/同时 … 调用
    m = re.search(
        r"(?:然后|并且|同时|再)[^。；\n]{0,40}调用(.{2,40}?)(?:的时候|时|，|,|需要|$)",
        q,
    )
    if m:
        clause = re.sub(
            r"将这个字段|作为入参|传递过去|传递|这个字段|需要改哪些地方",
            " ",
            m.group(1),
        )
        clause = re.sub(r"\s+", " ", clause).strip()
        if clause:
            out.append(clause)
    # Keep unique, short first
    uniq: list[str] = []
    for s in out:
        if s and s not in uniq:
            uniq.append(s)
    return uniq


def _outbound_hit_score(cm: str, question: str) -> int:
    """Prefer check/validate/client-style methods; demote persist updates."""
    meth = (cm or "").split("#", 1)[-1]
    sc = 0
    ml = meth.lower()
    ql = question or ""
    if re.search(r"(?i)check|validate|verify|client|feign", meth):
        sc += 40
    if re.search(r"(?i)^(update|insert|save|delete)", meth):
        sc -= 50
    # Token overlap with focus nouns in question (Latin already in method)
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", ql):
        if tok.lower() in ml:
            sc += 15
    if "ServiceImpl#" in (cm or ""):
        sc += 3
    return sc


async def _exec_find_outbound(
    project_id: str, question: str, evidence: dict[str, Any]
) -> None:
    from app.agent.ask_retrieval import _calls_closure_cms
    from app.agent.explore_shared import _method_name_from_hit
    from app.agent.explore_source import _fetch_method_source_body
    from app.agent.explore_tools import search_symbols

    entry = evidence.get("entry") or ""
    if not entry:
        return
    calls = set(_calls_closure_cms(project_id, entry))
    focus_qs = _outbound_focus_queries(question)
    evidence["outbound_focus"] = focus_qs
    hits: list[str] = []
    # 1) Focus queries first (gate phrase from the question)
    for fq in focus_qs or ["校验"]:
        try:
            sr = search_symbols(project_id, fq, top_k=10)
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                simple = str(h.get("fqcn") or "").split(".")[-1]
                mname = _method_name_from_hit(h)
                if simple and mname:
                    hits.append(f"{simple}#{mname}")
        except Exception:
            logger.debug("outbound focus search failed: %s", fq, exc_info=True)
    # 2) Fallback: dig-in core of full question (may be noisy)
    if not hits:
        from app.agent.ask_retrieval import _search_hit_cms_for_question

        hits = _search_hit_cms_for_question(project_id, question, top_k=12)

    inter = [c for c in hits if c in calls]
    inter = list(dict.fromkeys(inter))
    inter.sort(
        key=lambda cm: (-_outbound_hit_score(cm, question), cm),
    )
    # Drop strongly demoted persist methods if any better hit exists
    strong = [c for c in inter if _outbound_hit_score(c, question) >= 20]
    if strong:
        inter = strong + [c for c in inter if c not in strong]
    evidence["intersect_cms"] = inter[:12]
    snippets: list[dict[str, str]] = []
    # Read a few candidates; prefer bodies that already put/set outbound params.
    ranked_cms = list(inter[:8])
    loaded: list[tuple[int, dict[str, str]]] = []
    for cm in ranked_cms:
        if _outbound_hit_score(cm, question) < 0 and loaded:
            continue
        code, path = _fetch_method_source_body(project_id, cm, max_chars=12000)
        if not code:
            continue
        put_bonus = 0
        if _OUTBOUND_MAP_PUT_RE.search(code) or _OUTBOUND_DTO_SET_RE.search(code):
            put_bonus = 50
        score = _outbound_hit_score(cm, question) + put_bonus
        loaded.append(
            (
                score,
                {"class_method": cm, "path": path or "", "code": code},
            )
        )
    loaded.sort(key=lambda x: (-x[0], x[1]["class_method"]))
    snippets = [item for _, item in loaded[:3]]
    evidence["outbound_snippets"] = snippets
    if snippets:
        evidence["intersect_cms"] = [s["class_method"] for s in snippets] + [
            c for c in inter if c not in {s["class_method"] for s in snippets}
        ][:12]


def _apply_zh_comments(text: str, label_zh: str | None, field: str | None) -> str:
    if not text or not label_zh:
        return text
    out = text
    if field:
        out = out.replace(f"COMMENT '{field}'", f"COMMENT '{label_zh}'")
        out = out.replace(f"COMMENT \"{field}\"", f"COMMENT '{label_zh}'")
        out = re.sub(
            rf"/\*\*\s*\n\s*\*\s*{re.escape(field)}\s*\n\s*\*/",
            f"/**\n     * {label_zh}\n     */",
            out,
        )
        out = out.replace(f"// {field}", f"// {label_zh}")
    return out


def _rewrite_placeholder_field(text: str, field: str | None) -> str:
    """Replace discover drafts that still used newField with the real field name."""
    if not text or not field or field == "newField":
        return text
    if "newField" not in text and "NEW_FIELD" not in text and "new_field" not in text:
        return text
    from app.agent.change_impact.discover import _camel_to_snake, _field_to_const_name

    const = _field_to_const_name(field)
    snake = _camel_to_snake(field)
    pascal = field[:1].upper() + field[1:]
    out = text
    for a, b in (
        ("NEW_FIELD", const),
        ("new_field", snake),
        ("setNewField", f"set{pascal}"),
        ("getNewField", f"get{pascal}"),
        ("NewField", pascal),
        ("newField", field),
    ):
        out = out.replace(a, b)
    return out


# Peer: map.put(FieldNameConstants.FOO, reqVo.getFoo());
_OUTBOUND_MAP_PUT_RE = re.compile(
    r"(?m)^([ \t]*)"
    r"([A-Za-z_]\w*)\.put\s*\(\s*"
    r"((?:\w+\.)*\w*(?:Constant|Constants))\s*\.\s*([A-Z][A-Z0-9_]*)\s*,\s*"
    r"([A-Za-z_]\w*)\.(get[A-Z]\w*)\s*\(\s*\)\s*\)\s*;"
)
# Peer: loanSchemaReqVo.setFoo(reqVo.getFoo());
_OUTBOUND_DTO_SET_RE = re.compile(
    r"(?m)^([ \t]*)"
    r"([A-Za-z_]\w*)\.set([A-Z]\w*)\s*\(\s*"
    r"([A-Za-z_]\w*)\.get\3\s*\(\s*\)\s*\)\s*;"
)


def _getter_name(field: str) -> str:
    if not field:
        return "getField"
    return "get" + field[:1].upper() + field[1:]


def suggest_outbound_param_code(
    code: str,
    field: str | None,
    *,
    label_zh: str | None = None,
) -> str:
    """Clone map.put / DTO set peers to suggest passing the new field outbound."""
    from app.agent.change_impact.discover import _field_to_const_name, _setter_name

    if not field or not (code or "").strip():
        return ""
    const = _field_to_const_name(field)
    getter = _getter_name(field)
    setter = _setter_name(field)
    comment = label_zh or field
    lines: list[str] = []
    seen_maps: set[str] = set()
    for m in _OUTBOUND_MAP_PUT_RE.finditer(code):
        map_var, cref, vo = m.group(2), m.group(3), m.group(5)
        if map_var in seen_maps:
            continue
        if not re.search(r"(?i)req|vo|dto|param|apply", vo):
            continue
        seen_maps.add(map_var)
        ind = m.group(1) or "        "
        lines.append(f"{ind}// {comment}")
        lines.append(f"{ind}{map_var}.put({cref}.{const}, {vo}.{getter}());")
    seen_dto: set[str] = set()
    for m in _OUTBOUND_DTO_SET_RE.finditer(code):
        dst, src = m.group(2), m.group(4)
        if dst == src:
            continue
        if not re.search(r"(?i)req|vo|dto|param|apply", src):
            continue
        key = f"{dst}<-{src}"
        if key in seen_dto:
            continue
        seen_dto.add(key)
        ind = m.group(1) or "        "
        lines.append(f"{ind}// {comment}")
        lines.append(f"{ind}{dst}.{setter}({src}.{getter}());")
    return "\n".join(lines)


def _outbound_peer_excerpt(code: str, *, limit: int = 16) -> str:
    """Short peer region around first map.put / DTO set (not whole method)."""
    if not code:
        return ""
    lines = code.splitlines()
    idx = -1
    for i, ln in enumerate(lines):
        if _OUTBOUND_MAP_PUT_RE.search(ln) or _OUTBOUND_DTO_SET_RE.search(ln):
            idx = i
            break
    if idx < 0:
        chunk = lines[: min(limit, len(lines))]
        return "\n".join(chunk)
    start = max(0, idx - 2)
    end = min(len(lines), start + limit)
    return "\n".join(lines[start:end])


def _merge_outbound_into_answer(
    answer: str,
    evidence: dict[str, Any],
    *,
    field: str | None = None,
    label_zh: str | None = None,
) -> str:
    snippets = evidence.get("outbound_snippets") or []
    if not snippets:
        return answer

    # Build suggested-param sections (what user needs to paste).
    suggest_blocks: list[str] = []
    evidence_blocks: list[str] = []
    for s in snippets:
        cm = str(s.get("class_method") or "")
        path = str(s.get("path") or "")
        code = str(s.get("code") or "")
        suggested = suggest_outbound_param_code(code, field, label_zh=label_zh)
        if suggested:
            evidence["outbound_suggested"] = True
            block = [
                f"### 外送入参透传（`{cm}`）",
                "",
            ]
            if path:
                block.append(f"`{path}`（{cm}）")
                block.append("")
            block.extend(["```java", suggested.rstrip(), "```", ""])
            suggest_blocks.extend(block)
        excerpt = _outbound_peer_excerpt(code)
        if excerpt:
            eb = [f"### 证据摘录 `{cm}`", ""]
            if path:
                eb.append(f"文件：`{path}`")
                eb.append("")
            eb.extend(["```java", excerpt.rstrip(), "```", ""])
            evidence_blocks.extend(eb)

    ans = answer.rstrip()
    if suggest_blocks:
        insert = ["", *suggest_blocks]
        if "## DDL" in ans:
            ans = ans.replace("## DDL", "\n".join(insert) + "## DDL", 1)
        elif "## 改动清单" in ans:
            # Append after change list body
            ans = ans + "\n" + "\n".join(suggest_blocks)
        else:
            ans = ans + "\n\n## 外送入参改动\n\n" + "\n".join(suggest_blocks)

    if evidence_blocks:
        ans = (
            ans.rstrip()
            + "\n\n## 外送/关卡相关代码（证据摘录）\n\n"
            + "\n".join(evidence_blocks)
        )
    return ans


async def _exec_render_change(
    question: str, goal: dict[str, Any], evidence: dict[str, Any]
) -> str:
    dap = evidence.get("change_result") or {}
    ans = str(dap.get("plan_markdown") or dap.get("answer") or "").strip()
    if not ans:
        ans = "未能生成改动清单；证据不足。"
    label_zh = goal.get("field_label_zh") or _extract_field_label_zh(question)
    field = goal.get("field_name")
    if not field or field == "newField":
        try:
            from app.agent.question_intent import assumed_new_field_name

            guessed = assumed_new_field_name(question)
            if guessed and guessed != "newField":
                field = guessed
        except Exception:
            pass
    ans = _rewrite_placeholder_field(ans, field)
    ans = _apply_zh_comments(ans, label_zh, field)
    # Correct misleading「在 Controller 入口传递」when outbound is a gate method.
    # Internal review note goes to evidence (not the user answer); pev_agent
    # only embeds it when pev_test_tags_in_answer is enabled.
    outs = evidence.get("intersect_cms") or []
    if outs and re.search(r"Controller#", str(outs[0])) is None:
        evidence["outbound_focus_note"] = (
            f"> [PEV][TEST] 外送触点优先对照 `{outs[0]}`"
            f"（勿把 HTTP 入口本身当成关卡/外呼传递点）。"
        )
    # Ensure user-named physical tables appear in DDL section when missing
    tables = [
        t
        for t in (goal.get("tables") or [])
        if t and re.match(r"(?i)^demo_[a-z0-9_]+$", str(t).strip())
    ]
    if tables and "DDL" in ans:
        missing = [t for t in tables if t not in ans]
        if missing:
            col = re.sub(
                r"([A-Z])", r"_\1", str(field or "new_field")
            ).lower().lstrip("_")
            comment = label_zh or field or col
            extra = ["", "-- 用户点名表（发现层可能未绑定实体，请人工核对）"]
            for t in missing:
                extra.append(
                    f"ALTER TABLE `{t}`\n"
                    f"  ADD COLUMN `{col}` varchar(64) DEFAULT NULL COMMENT '{comment}';"
                )
            ans = ans.rstrip() + "\n" + "\n".join(extra) + "\n"
    ans = _merge_outbound_into_answer(
        ans, evidence, field=field, label_zh=label_zh
    )
    return ans


async def _exec_source_view(
    project_id: str,
    question: str,
    evidence: dict[str, Any],
    *,
    history_text: str = "",
    answer_steps: list | None = None,
    flow_symbols: list | None = None,
) -> str:
    from app.agent.explore_agent import _class_method_from_hint
    from app.agent.source_view import run_source_view

    view_kind = str(evidence.get("view_kind") or "").strip().lower()
    db_view = view_kind in {"db_lookup", "db_pipeline"}
    prefer: list[str] = []
    if not db_view:
        for raw in list(evidence.get("prefer_cms") or []) + list(
            evidence.get("dig_in_focus_cms") or []
        ):
            cm = _simple_cm(str(raw)) or _class_method_from_hint(str(raw)) or ""
            if cm and "#" in cm and cm not in prefer:
                prefer.append(cm)
        # Prefer explicit view_hint; avoid promoting locked Controller entry_hint.
        view_hint = _class_method_from_hint(
            str(evidence.get("view_hint") or "")
        ) or _simple_cm(str(evidence.get("view_hint") or ""))
        if not view_hint and not evidence.get("lock_entry"):
            view_hint = _class_method_from_hint(
                str(evidence.get("entry_hint") or "")
            )
        lock = _simple_cm(str(evidence.get("entry") or ""))
        if (
            view_hint
            and "#" in view_hint
            and view_hint != lock
            and view_hint not in prefer
        ):
            prefer.insert(0, view_hint)

        # If prefer is still empty and this is a show_code/dig-in with a
        # Chinese topic (method_topic / focus_topic), resolve that topic to
        # actual Class#method symbols via hybrid search.  This bridges the
        # semantic gap between the user's Chinese business term (e.g. "额度校验")
        # and the English code name (e.g. quotaCheck), preventing the fallback
        # in refine_source_view_plan from picking an unrelated search hit.
        if not prefer:
            topic = str(
                evidence.get("method_topic")
                or evidence.get("focus_topic")
                or ""
            ).strip()
            if topic:
                from app.agent.source_view import search_symbol_hits_for_question

                for cm in search_symbol_hits_for_question(project_id, topic, top_k=6):
                    if cm and "#" in cm and cm not in prefer:
                        prefer.append(cm)

    # Session-drift guard: when the session entry is an unrelated method (e.g. a
    # feign decorator from an earlier turn) but the current question explicitly
    # names a view target, show the view target — never the stale lock.
    session_entry = _simple_cm(str(evidence.get("entry") or ""))
    view_hint_cm = _class_method_from_hint(
        str(evidence.get("view_hint") or "")
    ) or _simple_cm(str(evidence.get("view_hint") or ""))
    if (
        not db_view
        and view_hint_cm
        and "#" in view_hint_cm
        and session_entry
        and view_hint_cm != session_entry
        and prefer
    ):
        session_entry = view_hint_cm

    out = await run_source_view(
        project_id,
        question,
        history_text=history_text,
        answer_steps=answer_steps,
        flow_symbols=flow_symbols,
        entry_cm=None if db_view else session_entry,
        preferred_cms=prefer or None,
        view_kind=view_kind or None,
        lookup_key=evidence.get("lookup_key"),
        lookup_table=evidence.get("lookup_table"),
        lookup_steps=evidence.get("lookup_steps"),
    )
    evidence["source_view"] = out
    if prefer:
        evidence["prefer_cms"] = prefer
    return str(out.get("answer") or "") if out.get("ok") else (
        out.get("answer") or "未能定位要展示的源码。"
    )


async def _exec_compose_code(
    project_id: str,
    question: str,
    evidence: dict[str, Any],
    *,
    history_text: str = "",
) -> str:
    from app.agent.compose_code import run_compose_code

    out = await run_compose_code(
        project_id,
        question,
        history_text=history_text,
        lookup_key=evidence.get("lookup_key"),
        lookup_table=evidence.get("lookup_table"),
        lookup_steps=evidence.get("lookup_steps"),
        view_kind=evidence.get("view_kind"),
    )
    evidence["compose_code"] = out
    return str(out.get("answer") or "") if out.get("ok") else (
        out.get("answer") or "未能根据仓库证据生成代码。"
    )


async def _exec_api_inventory(project_id: str, question: str) -> str:
    from app.agent.explore_agent import _assemble_api_inventory_answer

    ans, _syms = _assemble_api_inventory_answer(project_id, question)
    return ans or "未检索到相关 Controller 入口。"


async def _exec_parse_signal(question: str, evidence: dict[str, Any]) -> None:
    from app.agent.runtime.diagnose import ingest_signals

    ingest_signals(question, evidence)


async def _exec_map_and_read(
    project_id: str, question: str, evidence: dict[str, Any]
) -> None:
    from app.agent.runtime.diagnose import map_and_read

    map_and_read(project_id, question, evidence)


async def _exec_diagnose_answer(question: str, evidence: dict[str, Any]) -> str:
    from app.agent.runtime.diagnose import render_diagnose_answer

    return render_diagnose_answer(question, evidence)


async def _exec_collect_field_evidence(
    project_id: str, question: str, evidence: dict[str, Any]
) -> None:
    from app.agent.field_evidence import collect_field_evidence_pack

    entry = str(evidence.get("entry") or "")
    spine = list(evidence.get("calls") or [])
    pack = collect_field_evidence_pack(
        project_id, entry=entry, spine=spine, question=question
    )
    evidence["field_pack"] = pack


def _exec_render_field_answer(question: str, evidence: dict[str, Any]) -> str:
    from app.agent.field_evidence import render_field_evidence_answer

    pack = evidence.get("field_pack")
    if not isinstance(pack, dict) or not pack:
        return (
            "未收集到字段证据。请确认已锁定业务入口，或补充表名/字段名后再问。"
        )
    return render_field_evidence_answer(pack, question)


_REQ_GET_RE = re.compile(
    r'(?i)(?:req|reqMap|body|paramMap|params|map)\.get\s*\(\s*"(\w+)"\s*\)'
)
_RESP_PUT_RE = re.compile(
    r'(?i)(?:result|resp|response|respMap|out|data|map)\.put\s*\(\s*"(\w+)"'
)
_RESP_SET_RE = re.compile(
    r"(?i)\.set(ResponseCode|ResponseMsg|ResponseMessage|ResultCode|ResultMsg|"
    r"Code|Msg|Message|RetCode|RetMsg)\s*\("
)


def _extract_request_param_keys(code: str) -> list[str]:
    """Reuse explore_agent-style reqMap.get / body.get extraction."""
    keys = list(dict.fromkeys(_REQ_GET_RE.findall(code or "")))
    keys.extend(
        dict.fromkeys(
            re.findall(r'(?i)getParameter\s*\(\s*"(\w+)"\s*\)', code or "")
        )
    )
    return list(dict.fromkeys(keys))


def _extract_response_param_keys(code: str) -> list[str]:
    keys = list(dict.fromkeys(_RESP_PUT_RE.findall(code or "")))
    for m in _RESP_SET_RE.finditer(code or ""):
        name = m.group(1)
        # setResponseCode → responseCode
        camel = name[0].lower() + name[1:] if name else ""
        if camel:
            keys.append(camel)
    return list(dict.fromkeys(keys))


def _api_io_helper_cms(
    project_id: str, entry: str, evidence: dict[str, Any]
) -> list[str]:
    """Pick generateReqParam / checkParam / response builders near the entry."""
    out: list[str] = []
    seen: set[str] = set()
    entry_cm = _simple_cm(entry)
    if entry_cm:
        seen.add(entry_cm)

    def _add(raw: str | None) -> None:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or cm in seen:
            return
        meth = cm.split("#", 1)[-1].lower()
        if re.search(
            r"generatereq|reqparam|checkparam|buildresp|response|resultmap|param",
            meth,
        ):
            seen.add(cm)
            out.append(cm)

    for raw in list(evidence.get("calls") or [])[:20]:
        _add(str(raw))
    # same_class helpers from entry source
    try:
        from app.agent.explore_source import _fetch_method_source_body
        from app.agent.explore_tools import extract_same_class_calls

        code, _path = _fetch_method_source_body(project_id, entry_cm, max_chars=8000)
        if code and entry_cm and "#" in entry_cm:
            cls = entry_cm.split("#", 1)[0]
            meth = entry_cm.split("#", 1)[1]
            for name in extract_same_class_calls(code, current_method=meth)[:20]:
                _add(f"{cls}#{name}")
    except Exception:
        logger.debug("api_io same_class helpers failed", exc_info=True)
    return out[:8]


async def _exec_collect_api_io(
    project_id: str, evidence: dict[str, Any]
) -> None:
    from app.agent.explore_source import _fetch_method_source_body

    entry = _simple_cm(str(evidence.get("entry") or ""))
    cms: list[str] = []
    if entry:
        cms.append(entry)
    cms.extend(_api_io_helper_cms(project_id, entry, evidence))
    cms = list(dict.fromkeys(cms))

    req: list[str] = []
    resp: list[str] = []
    sources: list[dict[str, Any]] = []
    for cm in cms[:6]:
        code, path = _fetch_method_source_body(project_id, cm, max_chars=10000)
        if not code:
            continue
        rkeys = _extract_request_param_keys(code)
        skeys = _extract_response_param_keys(code)
        if rkeys or skeys:
            sources.append(
                {
                    "class_method": cm,
                    "path": path or "",
                    "request_params": rkeys,
                    "response_params": skeys,
                }
            )
        req.extend(rkeys)
        resp.extend(skeys)
    evidence["api_io_pack"] = {
        "entry": entry,
        "request_params": list(dict.fromkeys(req)),
        "response_params": list(dict.fromkeys(resp)),
        "sources": sources,
    }


def _exec_render_api_io(question: str, evidence: dict[str, Any]) -> str:
    pack = evidence.get("api_io_pack") if isinstance(
        evidence.get("api_io_pack"), dict
    ) else {}
    entry = _simple_cm(
        str((pack or {}).get("entry") or evidence.get("entry") or "")
    )
    req = list((pack or {}).get("request_params") or [])
    resp = list((pack or {}).get("response_params") or [])
    sources = list((pack or {}).get("sources") or [])
    if not entry and not req and not resp:
        return "未收集到接口入出参证据。请确认已锁定业务入口后再问。"
    lines = [
        f"业务入口：`{entry}`" if entry else "业务入口：",
        "",
        "## 请求参数",
        "",
    ]
    if req:
        for k in req:
            lines.append(f"- `{k}`")
    else:
        lines.append("- （入口及 generateReqParam/reqMap 辅助方法中未见明确 get 键）")
    lines.extend(["", "## 响应参数", ""])
    if resp:
        for k in resp:
            lines.append(f"- `{k}`")
    else:
        lines.append("- （未见明确 put/setResponse* 键；以入口返回对象字段为准）")
    if sources:
        lines.extend(["", "依据来源：", ""])
        for s in sources[:6]:
            cm = s.get("class_method") or ""
            n_req = len(s.get("request_params") or [])
            n_resp = len(s.get("response_params") or [])
            lines.append(f"- `{cm}`（请求键 {n_req} / 响应键 {n_resp}）")
    lines.append("")
    lines.append(
        f"依据：请求参数={len(req)}；响应参数={len(resp)}；"
        f"来源方法={len(sources)}"
    )
    return "\n".join(lines)


async def _exec_search_hits(
    project_id: str, question: str, evidence: dict[str, Any]
) -> None:
    from app.agent.ask_retrieval import _search_hit_cms_for_question

    evidence["search_hits"] = _search_hit_cms_for_question(
        project_id, question, top_k=12
    )


async def _exec_general_answer(
    question: str, evidence: dict[str, Any], *, history_text: str = ""
) -> str:
    if evidence.get("entry") and (evidence.get("mysql_rows") or evidence.get("calls")):
        return await _exec_summarize_flow(
            question,
            evidence,
            project_id=str(evidence.get("project_id") or ""),
            history_text=history_text,
        )
    hits = evidence.get("search_hits") or []
    if not hits and not evidence.get("entry"):
        return "检索未命中足够符号，证据不足，无法准确回答。"
    lines = [
        "## 回答（基于检索）",
        "",
        f"相关符号：{', '.join(f'`{h}`' for h in hits[:10])}",
        "",
        "可继续追问某一符号的流程或源码。",
    ]
    if evidence.get("entry"):
        lines.insert(2, f"建议入口：`{evidence['entry']}`")
    return "\n".join(lines)


