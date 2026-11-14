"""Field-value evidence: Neo4j spine + method_var_resolve + gate/branch status.

Primary truth: AST ``method_var_resolve`` + enclosing-method source branches.
Auxiliary: ``symbol_desc.extra_json.table_field_changes`` (AI summary) plus
deterministic Mapper-XML SQL column facts (``mapper_xml_sql`` source).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_TABLE_RE = re.compile(r"(?i)\b(demo_[a-z0-9_]+)\b")
_STATUS_FIELD_RE = re.compile(
    r"(?i)(loan_?status|pay_?status|push_?status|status|状态|支付状态)"
)
# Structural CN atoms → English method-name hints (not product Class#method locks).
_GATE_CN_STEMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("校验", ("check", "validate")),
    ("开关", ("switch",)),
)
_CONST_REF_RE = re.compile(
    r"(?P<cls>[A-Z][A-Za-z0-9_]*)\.(?P<name>[A-Z][A-Z0-9_]*)"
)
_STR_LIT_RE = re.compile(r'"([^"\\]{0,64})"')


def wants_field_value_ask(question: str) -> bool:
    """True when the user asks what a table/entity field or status becomes."""
    q = (question or "").strip()
    if not q:
        return False
    has_table = bool(_TABLE_RE.search(q)) or bool(
        re.search(r"[\u4e00-\u9fff]{2,12}表", q)
    )
    try:
        from app.agent.change_impact.discover import extract_named_tables

        has_table = has_table or bool(extract_named_tables(q))
    except Exception:
        pass
    fieldish = bool(
        re.search(
            r"(?:字段|列名|列值|取值|写成什么|是什么值|状态是什么|状态是啥|"
            r"状态会?变成|最终状态|落库.?什么|写入.?什么|表状态)",
            q,
        )
        or _STATUS_FIELD_RE.search(q)
    )
    gate_then = bool(
        re.search(r"校验后|判断后|失败后|通过后|成功后|拦截后", q)
        and re.search(r"状态|字段|表", q)
    )
    if has_table and (
        fieldish
        or gate_then
        or re.search(r"正常执行|执行完|成功后|跑完|完成后|失败", q)
    ):
        return True
    if fieldish and re.search(r"(?i)状态|字段|loanStatus|loan_status", q):
        return True
    if gate_then:
        return True
    return False


def detect_branch_intent(question: str) -> str:
    """Return fail | pass | both for status-after-gate asks."""
    q = question or ""
    want_fail = bool(
        re.search(r"失败|不通过|未通过|校验不过|拦截|拒绝|false", q, re.I)
    )
    want_pass = bool(
        re.search(
            r"通过|成功|正常执行|执行完|跑完|完成后|校验后(?!.*失败)|判断后(?!.*失败)",
            q,
        )
    )
    # 「校验后」alone → both paths
    if re.search(r"校验后|判断后", q) and not want_fail and not re.search(
        r"成功|通过|正常", q
    ):
        return "both"
    if want_fail and want_pass:
        return "both"
    if want_fail:
        return "fail"
    if want_pass:
        return "pass"
    if re.search(r"校验后|判断后|不同.*状态|状态流转", q):
        return "both"
    return "both"


def extract_field_anchors(question: str) -> dict[str, Any]:
    """Parse table / field / status needles from the question."""
    q = question or ""
    tables = list(_TABLE_RE.findall(q))
    try:
        from app.agent.change_impact.discover import extract_named_tables

        for t in extract_named_tables(q) or []:
            if t and t not in tables:
                tables.append(t)
    except Exception:
        pass
    fields: list[str] = []
    if re.search(r"loan_?status|表状态", q, re.I):
        fields.extend(
            ["loan_status", "loanStatus", "LOAN_STATUS", "presentResult", "present_result"]
        )
    if re.search(r"支付状态|pay_?status", q, re.I):
        fields.extend(["pay_status", "payStatus", "PAY_STATUS"])
    if re.search(r"push_?status|推送状态", q, re.I):
        fields.extend(["push_status", "pushStatus"])
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]{2,40})\b", q):
        tok = m.group(1)
        if re.search(r"(?i)status|flag|amount|code|type|result", tok):
            fields.append(tok)
            fields.append(_snake(tok))
            fields.append(_camel(tok))
    if "状态" in q:
        fields.extend(["status", "Status", "Result"])
    fields = list(dict.fromkeys(f for f in fields if f))
    return {
        "tables": tables,
        "fields": fields,
        "question": q,
        "branch": detect_branch_intent(q),
    }


def _snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name).lower().lstrip("_")


def _camel(name: str) -> str:
    parts = re.split(r"[_\s]+", name)
    if not parts:
        return name
    return parts[0].lower() + "".join(
        p[:1].upper() + p[1:].lower() for p in parts[1:] if p
    )


def _simple_cm(cm: str | None) -> str:
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    return f"{left.split('.')[-1].strip()}#{right.split('(', 1)[0].strip()}"


def _field_match(token: str, needles: list[str]) -> bool:
    t = (token or "").lower().replace("_", "")
    if not t:
        return False
    for n in needles:
        nn = (n or "").lower().replace("_", "")
        if not nn:
            continue
        if nn == t or nn in t or t in nn:
            return True
    return False


def _score_gate_cm(
    cm: str,
    question: str,
    *,
    on_spine: bool = False,
    summary: str = "",
) -> int:
    meth = cm.split("#", 1)[-1].lower() if "#" in cm else cm.lower()
    q = question or ""
    score = 0
    if on_spine:
        score += 10
    # Direct method token in question
    if re.search(
        rf"(?i)(^|[^A-Za-z0-9_]){re.escape(meth)}([^A-Za-z0-9_]|$|方法)",
        q,
    ):
        score += 20
    for cn, stems in _GATE_CN_STEMS:
        if cn in q:
            for st in stems:
                if st in meth:
                    score += 8
    if re.search(r"(?i)check|validate|verify|switch", meth):
        score += 4
    if "Controller#" in (cm or ""):
        score -= 4
    # Evidence overlap: question CJK runs vs method summary/source (no glossary).
    blob = f"{cm} {summary}"
    for run in re.findall(r"[\u4e00-\u9fff]{2,8}", q):
        if run and run in blob:
            score += min(12, len(run))
    return score


def resolve_gate_method(
    question: str,
    spine: list[str],
    *,
    bags: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    """Pick the judgment/gate Class#method on the locked spine."""
    spine_cms = [_simple_cm(c) for c in spine if c and "#" in str(c)]
    spine_set = set(spine_cms)
    candidates = list(spine_cms)
    # Also include check/validate callees mentioned in bags under spine methods
    for cm, payload in (bags or {}).items():
        for site in payload.get("call_sites") or []:
            raw = str(site.get("raw_call") or "")
            name = raw.rsplit(".", 1)[-1].split("(", 1)[0]
            if re.search(r"(?i)check|validate|verify|switch", name):
                cls = _simple_cm(cm).split("#", 1)[0]
                candidates.append(f"{cls}#{name}")
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    for cm in candidates:
        if not cm or cm in seen:
            continue
        seen.add(cm)
        payload = (bags or {}).get(cm) or {}
        summary = str(
            payload.get("summary")
            or payload.get("neo4j_desc")
            or payload.get("snippet")
            or ""
        )
        sc = _score_gate_cm(
            cm, question, on_spine=cm in spine_set, summary=summary
        )
        if sc > 0:
            scored.append((sc, cm))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[0][1]


def _load_table_field_changes(
    project_id: str, class_methods: list[str]
) -> dict[str, list[dict[str, Any]]]:
    if not project_id or not class_methods:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session, init_db
        from app.models.orm import SymbolDescORM, SymbolORM

        init_db()
        want = {_simple_cm(c) for c in class_methods if c and "#" in c}
        with get_session() as session:
            rows = session.execute(
                select(SymbolORM, SymbolDescORM)
                .join(SymbolDescORM, SymbolDescORM.symbol_id == SymbolORM.id)
                .where(SymbolORM.project_id == project_id)
            ).all()
            for sym, desc in rows:
                cm = _simple_cm(f"{(sym.fqcn or '').split('.')[-1]}#{sym.method_name}")
                if cm not in want:
                    continue
                extra = desc.extra_json if isinstance(desc.extra_json, dict) else {}
                tfc = extra.get("table_field_changes") or []
                if isinstance(tfc, list) and tfc:
                    out[cm] = [x for x in tfc if isinstance(x, dict)]
    except Exception:
        logger.debug("load table_field_changes failed", exc_info=True)
    return out


def _load_mapper_sql_changes(
    project_id: str, class_methods: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic table_field_changes from Mapper XML SQL bodies.

    Parses the project's Mapper XML files (XML only — no full Java AST) and maps
    each statement's write columns onto ``table_field_changes``-shaped rows.
    Complements the AI summary with hard SQL evidence.
    """
    if not project_id or not class_methods:
        return {}
    want = {_simple_cm(c) for c in class_methods if c and "#" in c}
    if not want:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    try:
        from app.analysis.global_ast import _parse_one_mapper_xml
        from app.core.config import get_settings
        from app.parser.scanner import scan_mapper_xml_files

        settings = get_settings()
        root = settings.project_root_path
        if root.name != project_id:
            alt = root.parent / project_id
            if alt.is_dir():
                root = alt
        if not root.is_dir():
            return {}
        for fp in scan_mapper_xml_files(root):
            try:
                t, _errs = _parse_one_mapper_xml(root, fp)
            except Exception:
                continue
            if t is None:
                continue
            for cm, meta in (t.method_sql_meta or {}).items():
                cm_simple = _simple_cm(cm)
                if cm_simple not in want:
                    continue
                rows: list[dict[str, Any]] = []
                for tbl in meta.get("tables") or []:
                    for col in meta.get("write_cols") or []:
                        rows.append(
                            {
                                "table_or_entity": tbl,
                                "field": col,
                                "from_value": "",
                                "to_value": "",
                                "op": str(meta.get("op") or "write"),
                                "when": f"{cm_simple} Mapper SQL",
                                "source": "mapper_xml_sql",
                            }
                        )
                if rows:
                    out.setdefault(cm_simple, []).extend(rows)
    except Exception:
        logger.debug("load mapper sql changes failed", exc_info=True)
    return out


def _resolve_java_constant(project_id: str, const_ref: str) -> str | None:
    """Resolve ClassName.CONST_NAME → string/number literal when found on disk."""
    m = _CONST_REF_RE.search(const_ref or "")
    if not m:
        return None
    cls, name = m.group("cls"), m.group("name")
    try:
        from app.core.config import get_settings

        root = get_settings().project_root_path
        if project_id and root.name != project_id:
            alt = root.parent / project_id
            if alt.is_dir():
                root = alt
        hits = list(Path(root).rglob(f"{cls}.java"))[:5]
        pat = re.compile(
            rf"(?m)(?:public|private|protected)?\s*static\s+final\s+\w+\s+{re.escape(name)}\s*=\s*([^;]+);"
        )
        for path in hits:
            text = path.read_text(encoding="utf-8", errors="ignore")
            mm = pat.search(text)
            if not mm:
                continue
            rhs = mm.group(1).strip()
            lit = _STR_LIT_RE.search(rhs)
            if lit:
                return lit.group(1)
            return rhs
    except Exception:
        logger.debug("resolve constant failed %s", const_ref, exc_info=True)
    return None


def _fetch_source(project_id: str, cm: str) -> str:
    try:
        from app.agent.explore_source import _fetch_method_source_body

        code, _path = _fetch_method_source_body(project_id, cm, max_chars=16000)
        return code or ""
    except Exception:
        logger.debug("fetch source failed %s", cm, exc_info=True)
        return ""


def _find_callers_of_gate(
    gate_cm: str,
    spine: list[str],
    bags: dict[str, dict[str, Any]],
) -> list[str]:
    gate_meth = gate_cm.split("#", 1)[-1]
    out: list[str] = []
    for cm in spine:
        cm_s = _simple_cm(cm)
        payload = bags.get(cm_s) or {}
        for site in payload.get("call_sites") or []:
            raw = str(site.get("raw_call") or "")
            if gate_meth and gate_meth in raw:
                out.append(cm_s)
                break
    if not out:
        # Same-class default: Prefer ServiceImpl methods on spine that aren't the gate
        gcls = gate_cm.split("#", 1)[0]
        for cm in spine:
            cm_s = _simple_cm(cm)
            if cm_s.split("#", 1)[0] == gcls and cm_s != gate_cm:
                out.append(cm_s)
    return list(dict.fromkeys(out))


def _extract_if_block_after(text: str, cond_start: int) -> tuple[str, int]:
    """Return (then_block_body, end_index) for if (...) { ... } starting at cond."""
    brace = text.find("{", cond_start)
    if brace < 0:
        return "", cond_start
    depth = 0
    for j in range(brace, len(text)):
        ch = text[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1 : j], j + 1
    return "", cond_start


def analyze_gate_branches(
    project_id: str,
    *,
    gate_cm: str,
    callers: list[str],
) -> dict[str, Any]:
    """Parse fail/pass aftermath around gate calls in caller source."""
    gate_meth = gate_cm.split("#", 1)[-1]
    fail_actions: list[dict[str, Any]] = []
    pass_notes: list[str] = []
    snippets: list[dict[str, Any]] = []

    for caller in callers[:4]:
        src = _fetch_source(project_id, caller)
        if not src or gate_meth not in src:
            continue
        # Negated check: if (!gate(...)) { fail }
        for m in re.finditer(
            rf"if\s*\(\s*!\s*{re.escape(gate_meth)}\s*\(",
            src,
        ):
            block, end = _extract_if_block_after(src, m.start())
            snippets.append(
                {
                    "caller": caller,
                    "branch": "fail",
                    "cond": f"!{gate_meth}(...)",
                    "snippet": block.strip()[:500],
                }
            )
            for call in re.finditer(
                r"(?:this\.)?([A-Za-z_][\w]*)\s*\(([^;]{0,200})\)",
                block,
            ):
                callees = call.group(1)
                args = call.group(2)
                if callees in {"if", "for", "while", "switch", "log", "info", "error"}:
                    continue
                const_m = _CONST_REF_RE.search(args)
                const_ref = const_m.group(0) if const_m else ""
                lit = _STR_LIT_RE.search(args)
                resolved = (
                    _resolve_java_constant(project_id, const_ref) if const_ref else None
                )
                if lit and not resolved:
                    resolved = lit.group(1)
                fail_actions.append(
                    {
                        "caller": caller,
                        "call": f"{callees}(...)",
                        "raw_args": args.strip()[:160],
                        "const_ref": const_ref,
                        "resolved_value": resolved,
                        "returns": "return" in block,
                    }
                )
            # Pass path: next non-empty lines after if-block
            after = src[end : end + 240].strip().splitlines()
            after = [ln.strip() for ln in after if ln.strip() and not ln.strip().startswith("//")]
            if after:
                pass_notes.append(
                    f"`{caller}` 校验通过后继续：`{after[0][:120]}`"
                )

        # Positive check used as failure (e.g. if (someCheck(...)) { fail })
        for m in re.finditer(
            rf"if\s*\(\s*{re.escape(gate_meth)}\s*\(",
            src,
        ):
            # skip already matched negated forms
            prev = src[max(0, m.start() - 3) : m.start()]
            if "!" in prev:
                continue
            block, end = _extract_if_block_after(src, m.start())
            if not re.search(
                r"(?i)FAILED|fail|return|updateWithdrawals|update\w*Status",
                block,
            ):
                continue
            snippets.append(
                {
                    "caller": caller,
                    "branch": "fail",
                    "cond": f"{gate_meth}(...)",
                    "snippet": block.strip()[:500],
                }
            )
            for call in re.finditer(
                r"(?:this\.)?([A-Za-z_][\w]*)\s*\(([^;]{0,200})\)",
                block,
            ):
                callees = call.group(1)
                args = call.group(2)
                if callees in {"if", "for", "while", "switch", "log", "info", "error"}:
                    continue
                const_m = _CONST_REF_RE.search(args)
                const_ref = const_m.group(0) if const_m else ""
                resolved = (
                    _resolve_java_constant(project_id, const_ref) if const_ref else None
                )
                fail_actions.append(
                    {
                        "caller": caller,
                        "call": f"{callees}(...)",
                        "raw_args": args.strip()[:160],
                        "const_ref": const_ref,
                        "resolved_value": resolved,
                        "returns": "return" in block,
                    }
                )

    # Dig into update* helpers for concrete field writes (loanStatus="F" etc.)
    status_writes: list[dict[str, Any]] = []
    for act in fail_actions:
        call_name = str(act.get("call") or "").split("(", 1)[0]
        if not re.search(r"(?i)update", call_name):
            continue
        # Prefer same-class helper on caller
        caller_cls = str(act.get("caller") or "").split("#", 1)[0]
        helper_cm = f"{caller_cls}#{call_name}"
        body = _fetch_source(project_id, helper_cm)
        if not body:
            continue
        for sm in re.finditer(
            r"\.set([A-Z][A-Za-z0-9_]*)\s*\(\s*([^)]*)\s*\)",
            body,
        ):
            prop, arg = sm.group(1), sm.group(2).strip()
            if not re.search(r"(?i)status|result|fail|present", prop):
                continue
            lit = _STR_LIT_RE.search(arg)
            const_m = _CONST_REF_RE.search(arg)
            val = lit.group(1) if lit else None
            cref = const_m.group(0) if const_m else ""
            if cref and not val:
                val = _resolve_java_constant(project_id, cref)
            # Parameter passthrough: presentResult → use fail action resolved_value
            if not val and re.search(r"(?i)presentResult|loanStatus|status", arg):
                val = act.get("resolved_value")
                if not cref:
                    cref = act.get("const_ref") or arg
            status_writes.append(
                {
                    "via": helper_cm,
                    "setter": f"set{prop}",
                    "field": prop[:1].lower() + prop[1:],
                    "arg": arg[:120],
                    "value": val,
                    "const_ref": cref
                    or (
                        act.get("const_ref")
                        if not lit and re.search(r"(?i)presentResult", arg)
                        else ""
                    ),
                    "branch": "fail",
                }
            )

    return {
        "gate": gate_cm,
        "callers": callers,
        "fail_actions": fail_actions[:12],
        "pass_notes": pass_notes[:6],
        "status_writes": status_writes[:16],
        "snippets": snippets[:4],
    }


def collect_field_evidence_pack(
    project_id: str,
    *,
    entry: str,
    spine: list[str],
    question: str,
) -> dict[str, Any]:
    """Build field evidence along a locked Neo4j spine (+ gate/branch when present)."""
    anchors = extract_field_anchors(question)
    entry_cm = _simple_cm(entry)
    cms: list[str] = []
    seen: set[str] = set()
    for raw in [entry_cm] + list(spine or []):
        cm = _simple_cm(raw)
        if cm and "#" in cm and cm not in seen:
            seen.add(cm)
            cms.append(cm)

    try:
        from app.analysis.var_resolve import load_method_var_resolves

        bags_raw = load_method_var_resolves(project_id, cms[:60])
        by_simple = {_simple_cm(k): v for k, v in bags_raw.items()}
    except Exception:
        logger.debug("load_method_var_resolves failed", exc_info=True)
        by_simple = {}

    field_needles = list(anchors.get("fields") or [])
    table_needles = [t.lower() for t in (anchors.get("tables") or [])]

    set_hits: list[dict[str, Any]] = []
    for cm in cms:
        payload = by_simple.get(cm) or {}
        for site in payload.get("call_sites") or []:
            raw = str(site.get("raw_call") or "")
            meth = raw.rsplit(".", 1)[-1] if raw else ""
            if not meth.startswith("set") or len(meth) < 4 or not meth[3:4].isupper():
                continue
            prop = meth[3:]
            prop_l = prop[:1].lower() + prop[1:] if prop else ""
            if field_needles and not (
                _field_match(prop, field_needles)
                or _field_match(prop_l, field_needles)
                or _field_match(_snake(prop), field_needles)
            ):
                if not (
                    "状态" in question
                    and re.search(r"(?i)status|Status|Result", meth)
                ):
                    continue
            set_hits.append(
                {
                    "class_method": cm,
                    "raw_call": raw,
                    "setter": meth,
                    "receiver": site.get("receiver"),
                    "receiver_type": site.get("resolved_type") or "",
                    "resolve_how": site.get("resolve_how") or "",
                    "source": "method_var_resolve",
                }
            )

    tfc_by_cm = _load_table_field_changes(project_id, cms)
    # Deterministic Mapper-XML SQL facts — merge into the same evidence stream.
    sql_by_cm = _load_mapper_sql_changes(project_id, cms)
    merged_tfc: dict[str, list[dict[str, Any]]] = dict(tfc_by_cm)
    for cm, rows in sql_by_cm.items():
        existing = merged_tfc.setdefault(cm, [])
        existing_keys = {
            (str(r.get("table_or_entity") or ""), str(r.get("field") or ""))
            for r in existing
        }
        for row in rows:
            key = (str(row.get("table_or_entity") or ""), str(row.get("field") or ""))
            if key not in existing_keys:
                existing_keys.add(key)
                existing.append(row)
    tfc_hits: list[dict[str, Any]] = []
    for cm in cms:
        for i, row in enumerate(merged_tfc.get(cm) or []):
            field = str(row.get("field") or "")
            table = str(row.get("table_or_entity") or "")
            if table_needles and not any(
                tn in table.lower() or table.lower() in tn for tn in table_needles
            ):
                if not (field_needles and _field_match(field, field_needles)):
                    continue
            elif field_needles and not _field_match(field, field_needles):
                if not (
                    table_needles
                    and any(tn in table.lower() for tn in table_needles)
                    and ("状态" in question or "status" in field.lower())
                ):
                    continue
            tfc_hits.append(
                {
                    "class_method": cm,
                    "order": i + 1,
                    "table_or_entity": table,
                    "field": field,
                    "from_value": row.get("from_value"),
                    "to_value": row.get("to_value"),
                    "op": row.get("op"),
                    "source": row.get("source") or "table_field_changes",
                }
            )

    gate_cm = resolve_gate_method(question, cms, bags=by_simple)
    gate_info: dict[str, Any] = {}
    if gate_cm:
        # Ensure gate bag loaded for caller detection
        if gate_cm not in by_simple:
            try:
                from app.analysis.var_resolve import load_method_var_resolves

                extra = load_method_var_resolves(project_id, [gate_cm])
                by_simple.update({_simple_cm(k): v for k, v in extra.items()})
            except Exception:
                pass
        callers = _find_callers_of_gate(gate_cm, cms, by_simple)
        # Always try loan-like callers on same class if empty
        if not callers:
            gcls = gate_cm.split("#", 1)[0]
            for cm in cms:
                if cm.startswith(gcls + "#") and cm != gate_cm:
                    callers.append(cm)
        gate_info = analyze_gate_branches(
            project_id, gate_cm=gate_cm, callers=callers or cms[:3]
        )

    return {
        "entry": entry_cm,
        "anchors": anchors,
        "spine": cms,
        "bags_loaded": len(by_simple),
        "set_hits": set_hits[:40],
        "tfc_hits": tfc_hits[:40],
        "gate": gate_cm,
        "gate_info": gate_info,
        "branch_intent": anchors.get("branch") or "both",
    }


def render_field_evidence_answer(pack: dict[str, Any], question: str) -> str:
    """Deterministic answer from field evidence pack (gate/branch aware)."""
    entry = pack.get("entry") or ""
    anchors = pack.get("anchors") or {}
    set_hits = pack.get("set_hits") or []
    tfc_hits = pack.get("tfc_hits") or []
    gate = pack.get("gate") or ""
    gate_info = pack.get("gate_info") or {}
    branch = pack.get("branch_intent") or detect_branch_intent(question)

    lines = [
        f"业务入口：`{entry}`" if entry else "业务入口：",
        "字段/状态证据（主链锁定；关卡分支 + `method_var_resolve` 为主，"
        "`table_field_changes` 为辅）：",
        "",
    ]
    if anchors.get("tables") or anchors.get("fields"):
        bits = []
        if anchors.get("tables"):
            bits.append("表=" + ", ".join(f"`{t}`" for t in anchors["tables"][:4]))
        if anchors.get("fields"):
            bits.append("字段≈" + ", ".join(f"`{f}`" for f in anchors["fields"][:6]))
        lines.append("锚点：" + "；".join(bits))
        lines.append("")

    if gate:
        lines.append(f"关卡：`{gate}`（分支意图：`{branch}`）")
        lines.append("")

    fail_actions = gate_info.get("fail_actions") or []
    status_writes = gate_info.get("status_writes") or []
    pass_notes = gate_info.get("pass_notes") or []

    if gate and branch in {"fail", "both"} and (fail_actions or status_writes):
        lines.append("### 校验失败分支")
        lines.append("")
        for act in fail_actions[:6]:
            bits = [f"`{act.get('caller')}` → `{act.get('call')}`"]
            if act.get("const_ref"):
                bits.append(f"常量 `{act['const_ref']}`")
            if act.get("resolved_value") is not None:
                bits.append(f"值 **`{act['resolved_value']}`**")
            if act.get("returns"):
                bits.append("随后 return（中断主链）")
            lines.append("- " + "；".join(bits))
        if status_writes:
            lines.append("")
            lines.append("失败后落库/状态写点：")
            for w in status_writes[:8]:
                val = w.get("value")
                lines.append(
                    f"- `{w.get('via')}` → `{w.get('setter')}` "
                    f"字段 `{w.get('field')}` = **`{val if val is not None else w.get('arg')}`**"
                    + (f"（{w.get('const_ref')}）" if w.get("const_ref") else "")
                )
        lines.append("")

    if gate and branch in {"pass", "both"}:
        lines.append("### 校验通过分支")
        lines.append("")
        if pass_notes:
            for n in pass_notes[:4]:
                lines.append(f"- {n}")
            lines.append(
                "- 通过时通常**不在关卡处立即改失败态**；主表最终状态取决于后续主链写点。"
            )
        else:
            lines.append(
                "- 源码侧未解析到显式「通过后立刻改状态」；通过后继续主链后续步骤。"
            )
        lines.append("")

    if set_hits and not (gate and status_writes):
        lines.append("### 精确写点（method_var_resolve set*）")
        lines.append("")
        for h in set_hits[:10]:
            lines.append(
                f"- `{h.get('class_method')}` → `{h.get('raw_call')}`"
                + (
                    f"（receiver `{h.get('receiver_type')}`）"
                    if h.get("receiver_type")
                    else ""
                )
            )
        lines.append("")

    if tfc_hits:
        ai_rows = [h for h in tfc_hits if h.get("source") != "mapper_xml_sql"]
        sql_rows = [h for h in tfc_hits if h.get("source") == "mapper_xml_sql"]
        if sql_rows:
            lines.append("### 字段变动摘要（Mapper SQL 确定性证据）")
            lines.append("")
            for h in sql_rows[:10]:
                lines.append(
                    f"- `{h.get('class_method')}` → SQL 写入 "
                    f"`{h.get('table_or_entity')}.{h.get('field')}`"
                    f"（{h.get('op') or 'write'}）"
                )
            lines.append("")
        if ai_rows:
            lines.append("### 字段变动摘要（table_field_changes，AI 总结，需对照）")
            lines.append("")
            for h in ai_rows[:10]:
                lines.append(
                    f"- `{h.get('class_method')}` #{h.get('order')} "
                    f"`{h.get('table_or_entity')}.{h.get('field')}` "
                    f"{h.get('from_value') or '?'} → **`{h.get('to_value')}`** "
                    f"（{h.get('op') or 'op'}）"
                )
            lines.append("")

    # Concise conclusion
    if gate and status_writes and branch in {"fail", "both"}:
        # Prefer loanStatus / presentResult
        prefer = sorted(
            status_writes,
            key=lambda w: (
                0
                if re.search(r"(?i)loanstatus|loan_status", str(w.get("field") or ""))
                else 1
            ),
        )
        top = prefer[0]
        lines.append(
            f"**结论（失败）**：关卡 `{gate}` 失败后，经由 `{top.get('via')}` 写入 "
            f"`{top.get('field')}` = **`{top.get('value')}`**"
            + (
                f"（常量 `{top.get('const_ref')}`）"
                if top.get("const_ref")
                else ""
            )
            + "。"
        )
        # Also mention presentResult if distinct
        for w in prefer[1:4]:
            if w.get("field") != top.get("field") and w.get("value") is not None:
                lines.append(
                    f"- 同时 `{w.get('field')}` = **`{w.get('value')}`**"
                    + (f"（`{w.get('const_ref')}`）" if w.get("const_ref") else "")
                )
        lines.append("")
    elif gate and branch == "pass":
        lines.append(
            f"**结论（通过）**：关卡 `{gate}` 通过后不在该处写失败态；需结合后续主链看最终业务状态。"
        )
        lines.append("")

    if not set_hits and not tfc_hits and not fail_actions and not status_writes:
        lines.append(
            "主链闭包内未命中足够字段写点/关卡分支。可补充字段英文名或关卡方法名后再问。"
        )
        lines.append("")

    lines.append(
        f"依据：spine={len(pack.get('spine') or [])}；"
        f"var_bags={pack.get('bags_loaded') or 0}；"
        f"gate={gate or '-'}；"
        f"set_hits={len(set_hits)}；tfc_hits={len(tfc_hits)}；"
        f"fail_actions={len(fail_actions)}；status_writes={len(status_writes)}"
    )
    return "\n".join(lines)
