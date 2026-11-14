"""Role-based touchpoint discovery from entry CALLS + source reads."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

ProgressCb = Callable[[dict[str, Any]], Any] | None

_VO_NAME_RE = re.compile(
    r"(?i)(?:ReqVo|Request|ReqDto|RequestDto|ParamVo|ApplyVo)\b"
)
_TABLE_NAME_RE = re.compile(
    r'@TableName\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']'
)
_SETTER_FROM_MAP_RE = re.compile(
    r"(?i)(?:reqMap|paramMap|map)\.get\s*\(\s*[\"'](\w+)[\"']\s*\)"
)
# Peer fields often use ClassConstant.FOO in reqMap.get / put (not string literals).
_SETTER_FROM_CONST_GET_RE = re.compile(
    r"(?i)(?:reqMap|paramMap|map)\.get\s*\(\s*"
    r"((?:\w+\.)*\w*(?:Constant|Constants))\s*\.\s*([A-Z][A-Z0-9_]*)\s*\)"
)
_CONST_CLASS_SIMPLE_RE = re.compile(r"(?i)^\w*(?:Constant|Constants)$")
_CONST_REF_RE = re.compile(
    r"\b((?:\w+\.)*\w*(?:Constant|Constants))\s*\.\s*([A-Z][A-Z0-9_]*)\b"
)
_CONST_IMPORT_RE = re.compile(
    r"import\s+([\w.]+\.(\w*(?:Constant|Constants)))\s*;",
    re.I,
)
_CONST_STRING_DEF_RE = re.compile(
    r"public\s+static\s+final\s+String\s+([A-Z][A-Z0-9_]*)\s*=\s*[\"']([^\"']+)[\"']"
)
_BIND_METHOD_RE = re.compile(r"(?i)check\w*Param|validate\w*|bind\w*|parse\w*Param")
_ASSEMBLE_METHOD_RE = re.compile(
    r"(?i)(?:2Ext|2Dmo|2Entity|ToExt|ToDmo|reqVo2|vo2|"
    r"insert\w*Param|build\w*(?:Dmo|Entity|Po)|assemble\w+)"
)
_INSERT_MP_RE = re.compile(
    r"(?i)(?:Ext|Dmo|Entity)?\w*Mapper\.insert\s*\(|BaseMapper"
)
# Outbound 协议同步 / syncSign* map assembly (generic name/code shapes).
_PROTOCOL_SYNC_METHOD_RE = re.compile(
    r"(?i)(?:syncSign|SignInfo|syncAgreement|syncProtocol|协议同步)"
)
_PROTOCOL_SYNC_CODE_RE = re.compile(
    r"(?i)(?:syncSignMap|signMap|协议同步|客户扣款协议)"
    r"|(?:\w*Map)\.put\s*\(\s*\w*(?:Constant|Constants)\.\w+"
)
_SYNC_MAP_PUT_RE = re.compile(
    r"(?i)([a-z]\w*Map)\s*\.\s*put\s*\(\s*"
    r"((?:\w+\.)*\w*(?:Constant|Constants))\s*\.\s*([A-Z][A-Z0-9_]*)\s*,\s*"
    r"([a-z]\w*)\.get([A-Z]\w*)\s*\(\s*\)\s*\)"
)
_SYNC_PARAM_VO_RE = re.compile(
    r"(?i)\b([A-Z][\w]*(?:ReqVo|Request|ReqDto|RequestDto))\b\s+\w+"
)
_FEIGN_SYNC_CALL_RE = re.compile(
    r"(?i)(\w*(?:Client|Feign|Service))\s*\.\s*(syncSign\w*|syncAgreement\w*)\s*\("
)


@dataclass
class RoleHit:
    role: str
    fqcn: str = ""
    method_name: str = ""
    file_path: str = ""
    raw_code: str = ""
    why: str = ""
    table_name: str | None = None
    analog_fields: list[str] = field(default_factory=list)
    # Provenance for UX: calls_closure | var_resolve | prefix_type | search_fallback
    via: str = ""
    evidence: str = ""
    # For role=constant: proposed `NAME = "field"` and peer defs from same class.
    proposed_const: str = ""
    peer_const_examples: list[str] = field(default_factory=list)

    @property
    def class_method(self) -> str:
        simple = (self.fqcn or "").split(".")[-1]
        if self.method_name:
            return f"{simple}#{self.method_name}"
        return simple or ""

    @property
    def label(self) -> str:
        fp = (self.file_path or "").strip()
        cm = self.class_method
        if fp and cm:
            return f"`{fp}`（{cm}）"
        if fp:
            return f"`{fp}`"
        return cm or "(未知符号)"


# External pass — code-shape only (Client/Feign/Map.put). No product gate nouns.
_ORG_GATE_METHOD_RE = re.compile(r"(?!x)x")  # disabled product method enum
_ORG_GATE_CODE_RE = re.compile(r"(?!x)x")
_SINGLE_BATCH_CODE_RE = re.compile(r"(?!x)x")
_EXTERNAL_CALL_RE = re.compile(
    r"(?i)(?:\w*(?:Client|Feign)\s*\.\s*\w+\s*\(|"
    r"(?:\w*Map)\s*\.\s*put\s*\(|setBody\s*\()"
)


@dataclass
class Touchpoints:
    capability: str
    entry: str
    field_name: str
    roles: dict[str, RoleHit] = field(default_factory=dict)
    need_mapper_xml: bool = False
    gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    # Precise set* sites from method_var_resolve (when available).
    set_sites: list[dict[str, Any]] = field(default_factory=list)
    # Discovery stats for answer footer (so identical names ≠ template).
    closure_size: int = 0
    bag_count: int = 0
    elapsed_ms: float = 0.0
    # User-named tables from the question (e.g. demo_b_withdrawals).
    named_tables: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        r = self.roles
        if self.capability in {"protocol_sync", "extend_field_to_sync"}:
            # Sync map assembly is enough; VO/constant enrich when found.
            return bool(r.get("protocol_sync"))
        if self.capability == "composite_field":
            # Need persist path + at least one org/single-batch external touchpoint.
            base = bool(
                (r.get("request_vo") or r.get("bind"))
                and (r.get("assemble") or r.get("persist_entity"))
            )
            gate = bool(
                r.get("org_check_gate")
                or r.get("single_batch_external")
                or r.get("external_pass")
            )
            return base and gate
        # Enough to draft a useful plan: VO or bind + (assemble or entity)
        return bool(
            (r.get("request_vo") or r.get("bind"))
            and (r.get("assemble") or r.get("persist_entity"))
        )

    @property
    def table_name(self) -> str | None:
        ent = self.roles.get("persist_entity")
        return (ent.table_name if ent else None) or None

    @property
    def persist_table_names(self) -> list[str]:
        """All @TableName values from persist_entity / persist_entity_N roles."""
        out: list[str] = []
        for key, hit in (self.roles or {}).items():
            if not key.startswith("persist_entity"):
                continue
            t = (hit.table_name or "").strip()
            if t and t not in out:
                out.append(t)
        for t in self.named_tables or []:
            if t and t not in out:
                out.append(t)
        return out


def extract_named_tables(question: str) -> list[str]:
    """Pull ``demo_*`` table names from the question (order-preserving, unique)."""
    found = re.findall(r"(demo_[a-z0-9_]+)", question or "", re.I)
    out: list[str] = []
    for t in found:
        tl = t.lower()
        if tl not in out:
            out.append(tl)
    return out


def _table_stem_to_pascal(table: str) -> str:
    """demo_b_loan_assistance → LoanAssistance; demo_b_withdrawals → Withdrawals."""
    stem = re.sub(r"(?i)^demo_[a-z]_", "", (table or "").strip())
    if not stem:
        return ""
    return "".join(p.capitalize() for p in stem.split("_") if p)


def _score_entity_for_table(simple: str, table: str) -> int:
    """Rank entity class names against a user-named table (generic, no product list)."""
    simple = (simple or "").strip()
    table_l = (table or "").lower()
    if not simple or not table_l:
        return -100
    s = 0
    expected = _table_stem_to_pascal(table_l)
    if expected and simple == expected:
        s += 40
    elif expected and simple.startswith(expected):
        s += 10
    # Ext/His/Apply siblings must not steal an exact base table.
    if simple.endswith("Ext") and not table_l.endswith("_ext"):
        s -= 30
    if simple.endswith("Apply") and not table_l.endswith("_apply"):
        s -= 20
    if re.search(r"(?i)(?:His|History|Sub|Plan|Approval)$", simple) and not re.search(
        r"(?i)_(?:his|history|sub|plan|approval)$", table_l
    ):
        s -= 15
    if expected and expected.lower() in simple.lower():
        s += 5
    return s


def _find_entity_sym_for_table(
    project_id: str,
    table: str,
    *,
    steps: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Resolve a Java type whose @TableName exactly matches ``table``."""
    table_l = (table or "").strip().lower()
    if not project_id or not table_l:
        return None
    expected = _table_stem_to_pascal(table_l)
    queries = [table_l]
    if expected:
        queries.append(expected)
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for q in queries:
        for sym in _search_and_read(project_id, q, steps=steps, top_k=6):
            code = str(sym.get("raw_code") or "")
            tm = _TABLE_NAME_RE.search(code)
            if not tm or tm.group(1).lower() != table_l:
                continue
            fqcn = str(sym.get("fqcn") or "")
            if fqcn in seen:
                continue
            seen.add(fqcn)
            candidates.append(sym)
    if not candidates:
        return None
    candidates.sort(
        key=lambda s: -_score_entity_for_table(
            str(s.get("fqcn") or "").split(".")[-1], table_l
        )
    )
    best = candidates[0]
    steps.append(
        {
            "tool": "resolve_entity_by_table",
            "args": {"table": table_l},
            "ok": True,
            "fqcn": str(best.get("fqcn") or ""),
        }
    )
    return best


def _simple_cm(cm: str | None) -> str:
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    return f"{left.split('.')[-1].strip()}#{right.split('(', 1)[0].strip()}"


def _entry_prefix(entry_cm: str | None) -> str | None:
    cm = _simple_cm(entry_cm)
    if not cm or "#" not in cm:
        return None
    simple = cm.split("#", 1)[0]
    for suf in ("Controller", "ServiceImpl", "Service", "Facade", "Api"):
        if simple.endswith(suf) and len(simple) > len(suf):
            return simple[: -len(suf)]
    return simple or None


def _camel_to_snake(name: str) -> str:
    return re.sub(r"([A-Z])", r"_\1", name or "").lower().lstrip("_")


def _setter_name(field: str) -> str:
    if not field:
        return "setField"
    return "set" + field[:1].upper() + field[1:]


def _field_to_const_name(field: str) -> str:
    """camelCase / PascalCase field → SCREAMING_SNAKE (e.g. businessLabel → BUSINESS_LABEL)."""
    return _camel_to_snake(field or "").upper() or "NEW_FIELD"


def _const_class_simple(ref: str) -> str:
    """`com.x.FooConstants` or `FooConstants` → simple name."""
    return (ref or "").rsplit(".", 1)[-1].strip()


def _collect_const_get_refs(code: str) -> list[tuple[str, str]]:
    """(ClassRef, CONST_NAME) from reqMap/paramMap.get(SomeConstant.FOO)."""
    return [
        (m.group(1), m.group(2))
        for m in _SETTER_FROM_CONST_GET_RE.finditer(code or "")
    ]


def _peer_const_defs(
    const_code: str,
    *,
    peer_fields: list[str],
    limit: int = 4,
) -> list[str]:
    """Return `NAME = \"value\"` defs whose value matches peer camelCase fields.

    Prefer peer_fields order (VO analogs first) over source-file declaration order.
    """
    want = [f for f in peer_fields if f]
    if not want:
        return []
    by_val: dict[str, str] = {}
    for m in _CONST_STRING_DEF_RE.finditer(const_code or ""):
        name, val = m.group(1), m.group(2)
        if val not in by_val:
            by_val[val] = name
    out: list[str] = []
    for val in want:
        name = by_val.get(val)
        if name:
            out.append(f'{name} = "{val}"')
            if len(out) >= limit:
                break
    return out


def _emit(on_progress: ProgressCb, note: str, **extra: Any) -> None:
    if not on_progress:
        return
    try:
        on_progress({"type": "explore_step", "note": note, **extra})
    except Exception:
        pass


def _read_symbol(
    project_id: str,
    *,
    fqcn: str | None = None,
    method_name: str | None = None,
    symbol_id: str | None = None,
    max_chars: int = 100_000,
) -> dict[str, Any] | None:
    """Read symbol body for change-impact (large methods need full style peers)."""
    from app.agent.explore_tools import get_symbol_code

    got = get_symbol_code(
        project_id,
        symbol_id=symbol_id,
        fqcn=fqcn,
        method_name=method_name or None,
        max_chars=max_chars,
    )
    if not got.get("ok"):
        return None
    sym = dict(got.get("symbol") or {})
    sym["raw_code"] = str(got.get("raw_code") or "")
    if not sym.get("fqcn") and fqcn:
        sym["fqcn"] = fqcn
    if method_name and not sym.get("method_name"):
        sym["method_name"] = method_name
    if got.get("file_path") and not sym.get("file_path"):
        sym["file_path"] = got.get("file_path")
    return sym


def _classify_roles(
    sym: dict[str, Any],
    *,
    preferred_entity_names: set[str] | None = None,
) -> list[tuple[str, str]]:
    """Return [(role, why), ...] from code shape — no fixed class names."""
    code = str(sym.get("raw_code") or "")
    if not code.strip():
        return []
    fqcn = str(sym.get("fqcn") or "")
    method = str(sym.get("method_name") or "")
    simple = fqcn.split(".")[-1]
    out: list[tuple[str, str]] = []
    preferred = {n for n in (preferred_entity_names or set()) if n}

    if _VO_NAME_RE.search(simple) or (
        not method
        and re.search(r"(?i)\bprivate\s+String\s+\w+", code)
        and "@TableName" not in code
        and not re.search(r"(?i)(?:Controller|ServiceImpl|Mapper)\b", simple)
    ):
        out.append(("request_vo", "请求 VO/DTO（类名或字段形态）"))

    if (
        _SETTER_FROM_MAP_RE.search(code)
        or _SETTER_FROM_CONST_GET_RE.search(code)
        or _BIND_METHOD_RE.search(method)
        or (
            re.search(r"(?i)\.get\s*\(\s*[\"']\w+[\"']\s*\)", code)
            and re.search(r"(?i)set[A-Z]\w*\s*\(", code)
            and re.search(r"(?i)reqMap|paramMap|Request|HttpServlet", code)
        )
    ):
        out.append(("bind", "入参绑定/校验（reqMap.get→set* 或 check*Param）"))

    # Assemble = VO → persist object. Require name shape and/or clear Ext/Entity
    # receivers — bare locals named `dmo` + BeanUtils(dmo, …) are too common in
    # unrelated service methods (false "assemble").
    strong_assemble_name = bool(_ASSEMBLE_METHOD_RE.search(method))
    ext_receiver_set = bool(
        re.search(
            r"(?i)\b(?:\w*(?:Ext|Entity)|extDmo|ext)\s*\.\s*set[A-Z]\w*\s*\(",
            code,
        )
        and re.search(r"(?i)(?:reqVo|applyReq|request)\.", code)
    )
    bean_to_ext = bool(
        re.search(
            r"(?i)BeanUtils\.copyProperties\s*\([^)]*\b\w*(?:Ext|Entity)\b",
            code,
        )
    )
    named_entity_assemble = False
    if preferred and re.search(r"(?i)(?:reqVo|applyReq|request)\.get[A-Z]", code):
        for en in preferred:
            if re.search(
                rf"\b{re.escape(en)}\b\s+\w+\s*=\s*new\s+{re.escape(en)}\b",
                code,
            ) and re.search(r"(?i)\w+\.set[A-Z]\w*\s*\(", code):
                named_entity_assemble = True
                break
            if re.search(
                rf"(?i)\b(?:insertDmo|dmo|{re.escape(en)})\s*\.\s*set[A-Z]\w*\s*\(\s*"
                r"(?:reqVo|applyReq|request)\.",
                code,
            ) and (
                f"new {en}" in code
                or re.search(rf"\b{re.escape(en)}\b\s+\w+", code)
            ):
                named_entity_assemble = True
                break
    if strong_assemble_name or ext_receiver_set or bean_to_ext or named_entity_assemble:
        why_a = "VO→持久化对象组装"
        if named_entity_assemble and preferred:
            why_a = f"VO→用户点名表实体组装（{', '.join(sorted(preferred)[:3])}）"
        out.append(("assemble", why_a))

    table = None
    m = _TABLE_NAME_RE.search(code)
    if m:
        table = m.group(1)
    if table or (
        re.search(r"(?i)(?:Ext|Dmo|DO|Entity|Po)\b", simple)
        and "@TableName" in code
    ):
        out.append(("persist_entity", f"持久化实体{f' @TableName={table}' if table else ''}"))

    if _INSERT_MP_RE.search(code) or "@TableName" in code:
        out.append(("orm", "MyBatis-Plus / @TableName 写库形态"))

    # Field-name / key constants class (e.g. *Constants with FOO = "foo").
    if (
        not method
        and _CONST_CLASS_SIMPLE_RE.search(simple)
        and _CONST_STRING_DEF_RE.search(code)
    ):
        out.append(("constant", "字段名/入参键常量类（*Constant*）"))

    return out


def _analog_fields_from_vo(code: str, *, limit: int = 8) -> list[str]:
    names = re.findall(r"(?i)\bprivate\s+String\s+(\w+)\s*;", code or "")
    # Prefer label-like / business string fields as analogs.
    scored: list[tuple[int, str]] = []
    for n in names:
        score = 0
        low = n.lower()
        if any(k in low for k in ("label", "scene", "code", "tag", "type", "channel")):
            score += 2
        if not low.startswith(("id", "no", "msg", "remark")):
            score += 1
        scored.append((score, n))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [n for _, n in scored[:limit]]


def _calls_closure(project_id: str, entry_cm: str) -> list[str]:
    try:
        from app.agent.explore_agent import _collect_entry_calls_closure

        return _collect_entry_calls_closure(project_id, entry_cm, depth=4, limit=120)
    except Exception:
        logger.debug("CALLS closure unavailable", exc_info=True)
        return []


def _var_resolve_role_boost(cm: str, bags: dict[str, dict[str, Any]]) -> int:
    """Score Class#method from var-resolve call_sites (set* on VO/Ext receivers)."""
    payload = bags.get(cm) or {}
    if not payload or payload.get("error"):
        return 0
    score = 0
    meth = cm.split("#", 1)[-1] if "#" in cm else ""
    if _BIND_METHOD_RE.search(meth):
        score += 3
    if _ASSEMBLE_METHOD_RE.search(meth):
        score += 3
    for site in payload.get("call_sites") or []:
        raw = str(site.get("raw_call") or "")
        setter = raw.rsplit(".", 1)[-1] if raw else ""
        if not (
            setter.startswith("set")
            and len(setter) > 3
            and setter[3:4].isupper()
        ):
            continue
        rtype = str(site.get("resolved_type") or "")
        how = str(site.get("resolve_how") or "")
        if re.search(r"(?i)(?:ReqVo|Request|ReqDto)$", rtype):
            score += 4
        if re.search(r"(?i)(?:Ext|Dmo|Entity|DO)$", rtype):
            score += 4
        if how in {"local_decl", "new", "param", "cast", "field_map"}:
            score += 1
        if site.get("improved_by_locals"):
            score += 1
    # Locals typed as VO/Ext also hint bind/assemble methods
    for loc in payload.get("locals") or []:
        tn = str(loc.get("type_name") or "")
        if re.search(r"(?i)(?:ReqVo|Request)$", tn):
            score += 2
        if re.search(r"(?i)(?:Ext|Dmo)$", tn):
            score += 2
    return score


def _apply_var_resolve_hints(
    project_id: str,
    *,
    entry: str,
    field: str,
    closure: list[str],
    roles: dict[str, RoleHit],
    steps: list[dict[str, Any]],
    notes: list[str],
    ingest: Callable[..., None],
) -> list[dict[str, Any]]:
    """Prefer MySQL var-resolve bags for set-site precision under entry closure."""
    try:
        from app.analysis.var_resolve import (
            load_method_var_resolves,
            setter_sites_from_bags,
        )
    except Exception:
        return []

    cms = [entry] + [c for c in closure if c and c != entry]
    bags = load_method_var_resolves(project_id, cms[:80])
    steps.append(
        {
            "tool": "load_method_var_resolves",
            "args": {"entry": entry, "n": len(cms[:80])},
            "ok": bool(bags),
            "hit_count": len(bags),
        }
    )
    if not bags:
        return []

    notes.append(f"已加载 method_var_resolve：{len(bags)} 个方法（入口闭包）")
    set_sites = setter_sites_from_bags(bags, field_name=field, limit=30)

    # Re-prioritize: read high-scoring methods; upgrade weak bind/assemble too
    ranked = sorted(
        bags.keys(),
        key=lambda cm: -_var_resolve_role_boost(cm, bags),
    )
    weak_assemble = bool(roles.get("assemble")) and not _ASSEMBLE_METHOD_RE.search(
        (roles["assemble"].method_name or "")
    )
    missing = [
        r
        for r in ("bind", "assemble", "request_vo", "persist_entity")
        if r not in roles
    ]
    if missing or weak_assemble:
        for cm in ranked[:25]:
            boost = _var_resolve_role_boost(cm, bags)
            if boost < 3:
                continue
            if "#" not in cm:
                continue
            cls, _, meth = cm.partition("#")
            sym = _read_symbol(project_id, fqcn=cls, method_name=meth)
            steps.append(
                {
                    "tool": "get_symbol_code",
                    "args": {"class_method": cm, "via": "var_resolve"},
                    "ok": bool(sym),
                    "boost": boost,
                }
            )
            ingest(sym, via="var_resolve")
            if (
                roles.get("bind")
                and roles.get("assemble")
                and roles.get("request_vo")
                and roles.get("persist_entity")
                and _ASSEMBLE_METHOD_RE.search(
                    (roles.get("assemble").method_name or "")  # type: ignore[union-attr]
                )
            ):
                break

    # Annotate bind/assemble why + evidence with precise set* when available
    for role in ("bind", "assemble"):
        hit = roles.get(role)
        if not hit:
            continue
        cm = hit.class_method
        bag = bags.get(cm) or {}
        sites = [
            s
            for s in (bag.get("call_sites") or [])
            if str(s.get("raw_call") or "").rsplit(".", 1)[-1].startswith("set")
        ]
        if sites:
            sample = sites[0]
            how = sample.get("resolve_how") or "?"
            rtype = sample.get("resolved_type") or "?"
            raw = sample.get("raw_call") or "?"
            extra = f"；var-resolve: `{raw}`→{rtype}({how})"
            if extra not in (hit.why or ""):
                hit.why = (hit.why or "") + extra
            vr_bit = f"var-resolve `{raw}`→{rtype}({how})"
            if vr_bit not in (hit.evidence or ""):
                hit.evidence = (
                    f"{hit.evidence} · {vr_bit}" if hit.evidence else vr_bit
                )
            # Keep via as calls_closure if already; else mark var_resolve
            if not hit.via:
                hit.via = "var_resolve"

    exact = [s for s in set_sites if s.get("exact_field")]
    if exact:
        notes.append(
            "已有同名字段 setter 证据："
            + "；".join(
                f"`{s['class_method']}` `{s['raw_call']}`" for s in exact[:3]
            )
        )
    elif set_sites:
        notes.append(
            "闭包内 set* 落点（对照新增字段）："
            + "；".join(
                f"`{s['class_method']}` `{s['raw_call']}`({s.get('receiver_type') or '?'})"
                for s in set_sites[:4]
            )
        )
    return set_sites


def _seed_search_queries(entry_cm: str | None) -> list[str]:
    prefix = _entry_prefix(entry_cm)
    if not prefix:
        return []
    return [
        f"{prefix}ApplyReqVo",
        f"{prefix}ReqVo",
        f"{prefix}Request",
        f"{prefix}Ext",
        f"{prefix}Apply",
    ]


def _read_prefix_java_types(
    project_id: str,
    entry_cm: str | None,
    *,
    steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Filesystem fallback: `{Prefix}ApplyReqVo.java` / `{Prefix}Ext.java` near project root."""
    prefix = _entry_prefix(entry_cm)
    if not prefix:
        return []
    try:
        from app.core.config import get_settings

        root = get_settings().project_root_path
    except Exception:
        return []
    if not root:
        return []
    out: list[dict[str, Any]] = []
    for name in (f"{prefix}ApplyReqVo", f"{prefix}ReqVo", f"{prefix}Ext", f"{prefix}Request"):
        found = False
        try:
            paths = list(root.glob(f"**/{name}.java"))
        except Exception:
            paths = []
        for p in paths:
            if "target" in p.parts:
                continue
            try:
                rel = str(p.relative_to(root)).replace("\\", "/")
            except Exception:
                rel = str(p)
            sym = _read_symbol(project_id, fqcn=name, method_name=None)
            if not sym:
                # Force source: path read via get_symbol_code source: id
                from app.agent.explore_tools import get_symbol_code

                got = get_symbol_code(
                    project_id, symbol_id=f"source:{rel}", fqcn=name
                )
                steps.append(
                    {
                        "tool": "get_symbol_code",
                        "args": {"symbol_id": f"source:{rel}"},
                        "ok": bool(got.get("ok")),
                    }
                )
                if got.get("ok"):
                    sym = dict(got.get("symbol") or {})
                    sym["raw_code"] = str(got.get("raw_code") or "")
                    sym.setdefault("fqcn", name)
                    sym.setdefault("file_path", rel)
            else:
                steps.append(
                    {
                        "tool": "get_symbol_code",
                        "args": {"fqcn": name},
                        "ok": True,
                    }
                )
            if sym and str(sym.get("raw_code") or "").strip():
                out.append(sym)
                found = True
                break
        if found and len(out) >= 3:
            break
    return out


def _search_and_read(
    project_id: str,
    q: str,
    *,
    steps: list[dict[str, Any]],
    top_k: int = 4,
) -> list[dict[str, Any]]:
    from app.agent.explore_tools import search_symbols

    sr = search_symbols(project_id, q, top_k=top_k)
    steps.append(
        {
            "tool": "search_symbols",
            "args": {"q": q},
            "ok": bool(sr.get("ok")),
            "hit_count": int(sr.get("count") or 0),
        }
    )
    out: list[dict[str, Any]] = []
    if not sr.get("ok"):
        return out
    for h in (sr.get("hits") or [])[:top_k]:
        if not h:
            continue
        sid = str(h.get("symbol_id") or "") or None
        fq = str(h.get("fqcn") or "") or None
        meth = str(h.get("method_name") or "") or None
        # Prefer type-level read for VO/Ext (empty method).
        simple = (fq or "").split(".")[-1]
        read_meth = meth
        if _VO_NAME_RE.search(simple) or re.search(r"(?i)Ext\b", simple):
            read_meth = None
        sym = _read_symbol(
            project_id, symbol_id=sid, fqcn=fq, method_name=read_meth
        )
        steps.append(
            {
                "tool": "get_symbol_code",
                "args": {"q": q, "fqcn": fq, "method_name": read_meth},
                "ok": bool(sym),
            }
        )
        if sym:
            out.append(sym)
    return out


def _evidence_snip(sym: dict[str, Any], *, role: str, via: str, entry: str) -> str:
    """Short provenance line for render — not tool jargon dumps."""
    parts: list[str] = []
    if via == "calls_closure" and entry:
        parts.append(f"CALLS←{entry}")
    elif via == "var_resolve":
        parts.append("via=var_resolve")
    elif via == "prefix_type":
        parts.append("via=prefix_type（入口前缀定位类型）")
    elif via == "search_fallback":
        parts.append("via=search_fallback（闭包外补漏）")
    elif via:
        parts.append(f"via={via}")
    code = str(sym.get("raw_code") or "")
    if role == "bind":
        cm = _SETTER_FROM_CONST_GET_RE.search(code)
        if cm:
            parts.append(f"reqMap.get({cm.group(1)}.{cm.group(2)})→set*")
        else:
            m = _SETTER_FROM_MAP_RE.search(code)
            if m:
                parts.append(f'reqMap.get("{m.group(1)}")→set*')
            else:
                sm = re.search(r"(?i)(\w+\.set[A-Z]\w*)\s*\(", code)
                if sm:
                    parts.append(f"`{sm.group(1)}`")
    elif role == "assemble":
        sm = re.search(
            r"(?i)((?:\w*(?:Ext|Entity)|extDmo|ext)\s*\.\s*set[A-Z]\w*)\s*\(",
            code,
        )
        if sm:
            cleaned = re.sub(r"\s+", "", sm.group(1))
            parts.append(f"`{cleaned}`")
        else:
            sm2 = re.search(r"(?i)(\w+\.set[A-Z]\w*)\s*\(", code)
            if sm2:
                parts.append(f"`{sm2.group(1)}`")
    elif role == "request_vo":
        fields = re.findall(r"(?i)\bprivate\s+String\s+(\w+)\s*;", code)
        if fields:
            parts.append("对照字段 " + "/".join(fields[:2]))
    elif role == "persist_entity":
        tm = _TABLE_NAME_RE.search(code)
        if tm:
            parts.append(f'@TableName="{tm.group(1)}"')
    elif role == "constant":
        defs = list(_CONST_STRING_DEF_RE.finditer(code))
        if defs:
            sample = defs[0]
            parts.append(f'对照 `{sample.group(1)} = "{sample.group(2)}"`')
    return " · ".join(parts)


def _merge_role(
    roles: dict[str, RoleHit],
    role: str,
    sym: dict[str, Any],
    why: str,
    *,
    entry_prefix: str | None = None,
    closure_cms: set[str] | None = None,
    via: str = "",
    entry: str = "",
    proposed_const: str = "",
    peer_const_examples: list[str] | None = None,
    preferred_tables: set[str] | None = None,
    preferred_entity_names: set[str] | None = None,
) -> None:
    code = str(sym.get("raw_code") or "")
    table = None
    m = _TABLE_NAME_RE.search(code)
    if m:
        table = m.group(1)
    analogs: list[str] = []
    if role == "request_vo":
        analogs = _analog_fields_from_vo(code)
    evid = _evidence_snip(sym, role=role, via=via, entry=entry)
    hit = RoleHit(
        role=role,
        fqcn=str(sym.get("fqcn") or ""),
        method_name=str(sym.get("method_name") or ""),
        file_path=str(sym.get("file_path") or ""),
        raw_code=code,
        why=why,
        table_name=table,
        analog_fields=analogs,
        via=via,
        evidence=evid,
        proposed_const=proposed_const or "",
        peer_const_examples=list(peer_const_examples or []),
    )
    prev = roles.get(role)
    pref_tables = {t.lower() for t in (preferred_tables or set()) if t}
    pref_ents = {n for n in (preferred_entity_names or set()) if n}

    def _score(h: RoleHit) -> int:
        simple = (h.fqcn or "").split(".")[-1]
        meth = h.method_name or ""
        cm = h.class_method
        s = (1 if h.file_path else 0) + (2 if h.table_name else 0)
        s += min(len(h.raw_code), 8000) // 800
        # Prefer symbols on the entry CALLS closure when available (evidence > name luck).
        if closure_cms and cm and cm in closure_cms:
            s += 10
        if role == "request_vo":
            # Prefer *ApplyReqVo over bare *ReqVo (specificity, not a fixed class).
            if re.search(r"(?i)ApplyReqVo$", simple):
                s += 8
            elif re.search(r"(?i)(?:ReqVo|Request(?:Dto)?)$", simple):
                s += 4
            if entry_prefix and entry_prefix in simple:
                s += 3
            if h.analog_fields:
                s += min(len(h.analog_fields), 4)
        elif role == "bind":
            # Generic bind shapes only — no fixed method names (e.g. checkApply*).
            if _BIND_METHOD_RE.search(meth):
                s += 4
            if entry_prefix and entry_prefix in simple:
                s += 2
            if _SETTER_FROM_MAP_RE.search(h.raw_code):
                s += 3
            if _SETTER_FROM_CONST_GET_RE.search(h.raw_code):
                s += 4
            if re.search(r"(?i)set[A-Z]\w*\s*\(", h.raw_code) and re.search(
                r"(?i)\.get\s*\(\s*[\"']\w+[\"']\s*\)", h.raw_code
            ):
                s += 2
        elif role == "assemble":
            # Naming convention (*2Ext / vo2* / insert*Param) is OK; not a fixed Class#method list.
            if _ASSEMBLE_METHOD_RE.search(meth):
                s += 8
            if re.search(
                r"(?i)\b(?:\w*(?:Ext|Entity)|extDmo|ext)\s*\.\s*set[A-Z]\w*\s*\(",
                h.raw_code,
            ):
                # Ext assemble is demoted when user named non-_ext tables.
                if pref_tables and not any(t.endswith("_ext") for t in pref_tables):
                    s -= 6
                else:
                    s += 3
            if pref_ents:
                touches = sum(1 for en in pref_ents if en in (h.raw_code or ""))
                s += min(touches * 8, 24)
                if touches == 0 and re.search(r"(?i)Ext", meth + (h.raw_code or "")[:200]):
                    s -= 8
            if entry_prefix and entry_prefix in simple:
                s += 2
        elif role == "persist_entity":
            tlow = (h.table_name or "").lower()
            if pref_tables:
                if tlow and tlow in pref_tables:
                    s += 40
                elif tlow:
                    s -= 25
                s += _score_entity_for_table(simple, tlow or next(iter(pref_tables)))
            else:
                if re.search(r"(?i)(?:Ext|Entity)$", simple):
                    s += 6
                elif re.search(r"(?i)(?:Dmo|DO|Po)$", simple):
                    s += 3
            if entry_prefix and entry_prefix in simple:
                s += 3
        elif role == "constant":
            if re.search(r"(?i)FieldName", simple):
                s += 6
            if _CONST_CLASS_SIMPLE_RE.search(simple):
                s += 4
            if h.peer_const_examples:
                s += min(len(h.peer_const_examples) * 3, 9)
            if h.proposed_const:
                s += 2
            s += min(len(h.raw_code), 20000) // 2000
        elif role == "protocol_sync":
            s += _protocol_sync_score(cm, h.raw_code)
            if entry_prefix and entry_prefix in simple:
                s += 3
            if _SYNC_MAP_PUT_RE.search(h.raw_code or ""):
                s += 6
        # Sync-path VO: prefer bare *ReqVo (sync method param) over *ApplyReqVo.
        if role == "request_vo" and "sync" in (h.via or ""):
            if re.search(r"(?i)ApplyReqVo$", simple):
                s -= 6
            elif re.search(r"(?i)ReqVo$", simple):
                s += 8
        return s

    if prev is None or _score(hit) >= _score(prev):
        roles[role] = hit


def _attach_constant_role(
    project_id: str,
    *,
    field: str,
    roles: dict[str, RoleHit],
    steps: list[dict[str, Any]],
    notes: list[str],
    ingest: Callable[..., None],
) -> None:
    """Discover field-name constant class when bind/assemble peers use *Constant*."""
    if roles.get("constant"):
        return
    seed_codes: list[str] = []
    for r in ("bind", "assemble", "entry"):
        hit = roles.get(r) if r != "entry" else None
        if hit and hit.raw_code:
            seed_codes.append(hit.raw_code)
    bind = roles.get("bind")
    if bind and bind.raw_code:
        seed_codes.insert(0, bind.raw_code)

    # Class refs: prefer reqMap.get(Const.X); also imports / any Const.X usage.
    class_scores: dict[str, int] = {}
    for code in seed_codes:
        for cref, _cname in _collect_const_get_refs(code):
            simple = _const_class_simple(cref)
            class_scores[simple] = class_scores.get(simple, 0) + 5
        for m in _CONST_IMPORT_RE.finditer(code):
            simple = m.group(2)
            class_scores[simple] = class_scores.get(simple, 0) + 3
        for m in _CONST_REF_RE.finditer(code):
            simple = _const_class_simple(m.group(1))
            class_scores[simple] = class_scores.get(simple, 0) + 1

    if not class_scores:
        # Optional note only — do not invent a constant touchpoint.
        if bind and bind.raw_code and _SETTER_FROM_MAP_RE.search(bind.raw_code):
            notes.append(
                "入参绑定使用字符串字面量 get，未发现 *Constant* 字段名常量类；"
                "无需新增常量（除非同模块惯例要求）。"
            )
        return

    ranked = sorted(class_scores.keys(), key=lambda k: -class_scores[k])
    peer_fields: list[str] = []
    vo = roles.get("request_vo")
    if vo and vo.analog_fields:
        peer_fields.extend(vo.analog_fields)
    # Also harvest camelCase values from bind: setAccCardType / get("accCardType")
    if bind and bind.raw_code:
        peer_fields.extend(
            re.findall(r"(?i)\.get\s*\(\s*[\"'](\w+)[\"']\s*\)", bind.raw_code)
        )
        for m in re.finditer(r"(?i)\.set([A-Z]\w*)\s*\(", bind.raw_code):
            name = m.group(1)
            peer_fields.append(name[:1].lower() + name[1:])
    # Dedup preserve order
    seen_pf: set[str] = set()
    peers: list[str] = []
    for p in peer_fields:
        if p and p not in seen_pf:
            seen_pf.add(p)
            peers.append(p)

    for simple in ranked[:4]:
        # Prefer type-level read of the constants class.
        sym = _read_symbol(project_id, fqcn=simple, method_name=None)
        steps.append(
            {
                "tool": "get_symbol_code",
                "args": {"fqcn": simple, "via": "constant_from_bind"},
                "ok": bool(sym and str(sym.get("raw_code") or "").strip()),
            }
        )
        if not sym or not str(sym.get("raw_code") or "").strip():
            # Search by simple name (may resolve FQCN).
            for got in _search_and_read(
                project_id, simple, steps=steps, top_k=2
            ):
                gsimple = str(got.get("fqcn") or "").split(".")[-1]
                if gsimple == simple or _CONST_CLASS_SIMPLE_RE.search(gsimple):
                    sym = got
                    break
        if not sym:
            continue
        code = str(sym.get("raw_code") or "")
        # Large *Constants files are often truncated in the symbol store — prefer disk.
        try:
            from app.core.config import get_settings

            root = get_settings().project_root_path
            if root:
                for p in root.glob(f"**/{simple}.java"):
                    if "target" in p.parts:
                        continue
                    disk = p.read_text(encoding="utf-8", errors="ignore")
                    if _CONST_STRING_DEF_RE.search(disk) and len(disk) > len(code):
                        code = disk
                        sym = dict(sym)
                        sym["raw_code"] = disk
                        try:
                            sym["file_path"] = str(p.relative_to(root)).replace(
                                "\\", "/"
                            )
                        except Exception:
                            sym["file_path"] = str(p).replace("\\", "/")
                    break
        except Exception:
            logger.debug("constant disk fill failed", exc_info=True)
        if not _CONST_STRING_DEF_RE.search(code):
            continue
        vo_peers = [
            f
            for f in (
                (roles.get("request_vo").analog_fields if roles.get("request_vo") else None)
                or []
            )
            if f
        ]
        # Prefer VO-analog peers (accCardType …) when present in full source.
        peers_hit = _peer_const_defs(
            code, peer_fields=vo_peers or peers, limit=4
        )
        if not peers_hit:
            peers_hit = _peer_const_defs(code, peer_fields=peers, limit=4)
        # If no peer field match yet, still accept when bind clearly uses this class.
        if not peers_hit and class_scores.get(simple, 0) < 5:
            continue
        if not peers_hit:
            # Fall back: any 1–2 string defs as naming convention samples.
            for m in list(_CONST_STRING_DEF_RE.finditer(code))[:2]:
                peers_hit.append(f'{m.group(1)} = "{m.group(2)}"')
        proposed = f'{_field_to_const_name(field)} = "{field}"'
        why = (
            f"绑定侧使用 `{simple}.…` 作入参键；"
            f"按同类命名约定新增常量"
        )
        evid_bits = []
        if bind:
            evid_bits.append(f"bind=`{bind.class_method}`")
        get_refs = _collect_const_get_refs(bind.raw_code if bind else "")
        if get_refs:
            cref, cname = get_refs[0]
            evid_bits.append(f"reqMap.get({_const_class_simple(cref)}.{cname})")
        if peers_hit:
            evid_bits.append("对照 " + " / ".join(peers_hit[:2]))
        _merge_role(
            roles,
            "constant",
            sym,
            why,
            via="bind_const_ref",
            entry="",
            proposed_const=proposed,
            peer_const_examples=peers_hit,
        )
        hit = roles.get("constant")
        if hit:
            hit.proposed_const = proposed
            hit.peer_const_examples = peers_hit
            hit.evidence = " · ".join(evid_bits) if evid_bits else hit.evidence
            hit.via = "bind_const_ref"
        notes.append(
            f"发现字段名常量类 `{simple}`"
            + (f"；建议新增 `{proposed}`" if proposed else "")
        )
        # Also classify via ingest for consistency (type role).
        ingest(sym, allow_roles=frozenset({"constant"}), via="bind_const_ref")
        # Re-apply proposal after ingest merge may have replaced hit.
        hit2 = roles.get("constant")
        if hit2:
            hit2.proposed_const = proposed
            hit2.peer_const_examples = peers_hit
            if evid_bits:
                hit2.evidence = " · ".join(evid_bits)
        return

    notes.append(
        "绑定代码引用了 *Constant* 类，但未能读到含字符串字段名定义的源码；"
        "常量改动点未列入清单。"
    )


def discover_add_field_touchpoints(
    project_id: str,
    question: str,
    *,
    entry_cm: str | None = None,
    on_progress: ProgressCb = None,
) -> Touchpoints:
    """Discover add-field roles under a business entry (generic across APIs)."""
    import time

    from app.agent.question_intent import assumed_new_field_name

    t0 = time.perf_counter()
    field = assumed_new_field_name(question)
    steps: list[dict[str, Any]] = []
    notes: list[str] = []

    # Resolve entry: explicit → entry_route + prefer_flow_entry (no lexicon).
    entry = _simple_cm(entry_cm) if entry_cm else ""
    if not entry or "#" not in entry:
        try:
            from app.agent.ask_retrieval import resolve_flow_entry_for_question

            entry = _simple_cm(
                resolve_flow_entry_for_question(
                    project_id,
                    question,
                    entry_hint=str(
                        (entry_cm or "") if entry_cm and "#" not in str(entry_cm) else ""
                    )
                    or None,
                )
                or ""
            ) or ""
        except Exception:
            logger.debug("resolve_flow_entry_for_question failed", exc_info=True)
            entry = ""

    tp = Touchpoints(
        capability="add_field",
        entry=entry or "(未定位入口)",
        field_name=field,
        steps=steps,
    )
    if not project_id:
        tp.gaps.append("缺少 project_id")
        tp.elapsed_ms = (time.perf_counter() - t0) * 1000
        return tp
    if not entry or "#" not in entry:
        tp.gaps.append("未能从问题定位业务入口（Controller#method）")
        notes.append("请在问题中带上业务入口名或先问清入口流程。")
        tp.notes = notes
        tp.elapsed_ms = (time.perf_counter() - t0) * 1000
        return tp

    _emit(on_progress, f"定位入口 {entry}，开始结构化发现改动点…")
    notes.append(f"入口：`{entry}`")
    q_low = (question or "").lower()
    named_in_q = bool(
        field
        and field != "newField"
        and (field in (question or "") or _camel_to_snake(field) in q_low)
    )
    notes.append(
        f"字段：`{field}`"
        + ("（问题未指名，按此假设）" if not named_in_q else "")
    )

    named_tables = extract_named_tables(question)
    tp.named_tables = list(named_tables)
    preferred_tables: set[str] = set(named_tables)
    preferred_entity_names: set[str] = set()
    named_entity_syms: list[dict[str, Any]] = []
    if named_tables:
        notes.append("用户点名表：" + "、".join(f"`{t}`" for t in named_tables))
        _emit(on_progress, f"按用户点名表解析实体：{', '.join(named_tables)}")
        for tname in named_tables:
            esym = _find_entity_sym_for_table(project_id, tname, steps=steps)
            if not esym:
                tp.gaps.append(f"未找到 @TableName={tname} 的实体类")
                continue
            named_entity_syms.append(esym)
            simple_e = str(esym.get("fqcn") or "").split(".")[-1]
            if simple_e:
                preferred_entity_names.add(simple_e)

    # Ensure MySQL var-resolve bags (scoped sync if missing) — no manual CLI.
    try:
        from app.analysis.var_resolve import ensure_entry_var_resolve

        _emit(on_progress, "检查 method_var_resolve（缺失则按入口闭包补齐）…")
        ens = ensure_entry_var_resolve(project_id, entry)
        steps.append(
            {
                "tool": "ensure_entry_var_resolve",
                "args": {"entry": entry},
                "ok": True,
                "skipped": bool(ens.get("skipped")),
                "reason": ens.get("reason"),
                "written": (ens.get("mysql") or {}).get("written"),
            }
        )
        if not ens.get("skipped"):
            notes.append(
                f"已自动补齐 method_var_resolve（入口闭包，written="
                f"{(ens.get('mysql') or {}).get('written')}）"
            )
            _emit(on_progress, "method_var_resolve 已补齐")
    except Exception as exc:
        steps.append(
            {
                "tool": "ensure_entry_var_resolve",
                "args": {"entry": entry},
                "ok": False,
                "error": str(exc)[:200],
            }
        )
        logger.debug("ensure_entry_var_resolve failed", exc_info=True)

    roles: dict[str, RoleHit] = {}
    seen_keys: set[str] = set()
    prefix = _entry_prefix(entry)
    closure_cms: set[str] = set()

    # Type-level roles only during early filesystem probes (VO/Ext not on CALLS).
    _TYPE_ROLES = frozenset({"request_vo", "persist_entity", "orm", "constant"})
    # Method roles must come from CALLS / var-resolve when closure exists.
    _METHOD_ROLES = frozenset({"bind", "assemble"})

    def _ingest(
        sym: dict[str, Any] | None,
        *,
        allow_roles: frozenset[str] | None = None,
        via: str = "",
    ) -> None:
        if not sym:
            return
        key = f"{sym.get('fqcn') or ''}#{sym.get('method_name') or ''}"
        if key in seen_keys and not str(sym.get("raw_code") or "").strip():
            return
        seen_keys.add(key)
        for role, why in _classify_roles(
            sym, preferred_entity_names=preferred_entity_names or None
        ):
            if allow_roles is not None and role not in allow_roles:
                continue
            # With a non-empty CALLS closure, do not accept bind/assemble from
            # off-closure search popularity (prefix *ReqVo hits often surface
            # the same summary methods without CALLS membership).
            if (
                role in _METHOD_ROLES
                and closure_cms
                and allow_roles is None
            ):
                cm = ""
                simple = str(sym.get("fqcn") or "").split(".")[-1]
                meth = str(sym.get("method_name") or "")
                if simple and meth:
                    cm = f"{simple}#{meth}"
                if cm and cm not in closure_cms:
                    continue
            role_via = via
            if not role_via:
                role_via = "calls_closure" if role in _METHOD_ROLES else "read"
            _merge_role(
                roles,
                role,
                sym,
                why,
                entry_prefix=prefix,
                closure_cms=closure_cms or None,
                via=role_via,
                entry=entry,
                preferred_tables=preferred_tables or None,
                preferred_entity_names=preferred_entity_names or None,
            )

    # 0) Force-bind entities for user-named tables (wins over Ext siblings).
    for esym in named_entity_syms:
        _ingest(
            esym,
            allow_roles=frozenset({"persist_entity", "orm"}),
            via="named_table",
        )

    # 1) Read entry method
    entry_cls, _, entry_meth = entry.partition("#")
    entry_sym = _read_symbol(project_id, fqcn=entry_cls, method_name=entry_meth)
    steps.append(
        {
            "tool": "get_symbol_code",
            "args": {"fqcn": entry_cls, "method_name": entry_meth},
            "ok": bool(entry_sym),
            "role": "entry",
        }
    )
    _ingest(entry_sym, via="entry")
    if entry_sym:
        _emit(on_progress, f"已读入口 {entry}")

    # 1b) Filesystem type probe ONLY (VO/Ext). Do NOT search_symbols yet —
    #    usage hits for *ReqVo often short-circuit bind/assemble off CALLS.
    _emit(on_progress, "按入口前缀定位请求 VO / 扩展实体（仅类型）…")
    for sym in _read_prefix_java_types(project_id, entry, steps=steps):
        _ingest(sym, allow_roles=_TYPE_ROLES, via="prefix_type")

    # 2) CALLS closure first — bind/assemble must be grounded here when present
    closure = _calls_closure(project_id, entry)
    closure_cms = {_simple_cm(c) for c in closure if c and "#" in c}
    if entry and "#" in entry:
        closure_cms.add(_simple_cm(entry))
    tp.closure_size = len(closure)
    _emit(on_progress, f"入口 CALLS 闭包 {len(closure)} 个候选")

    # Preload var-resolve bags early so priority can use them
    early_bags: dict[str, dict[str, Any]] = {}
    try:
        from app.analysis.var_resolve import load_method_var_resolves

        early_bags = load_method_var_resolves(
            project_id, [entry] + list(closure[:80])
        )
        tp.bag_count = len(early_bags)
    except Exception:
        early_bags = {}

    def _priority(cm: str) -> int:
        low = cm.lower()
        meth = cm.split("#", 1)[-1] if "#" in cm else cm
        score = 0
        if "reqvo" in low or "request" in low:
            score += 5
        if "ext" in low and "mapper" not in low:
            score += 4
        if "param" in low or "check" in low or "valid" in low or "bind" in low:
            score += 3
        # Strong assemble name shapes should be read early (avoid scanning 30+ gates).
        if _ASSEMBLE_METHOD_RE.search(meth):
            score += 10
        elif "2ext" in low or "2dmo" in low or "assemble" in low or "vo2" in low:
            score += 4
        # When user named concrete tables, prefer insert*Param / save* over Ext mappers.
        if preferred_tables and "insert" in low and "param" in low:
            score += 12
        elif "insert" in low and "param" in low:
            score -= 2
        if preferred_entity_names and any(
            en.lower() in low for en in preferred_entity_names
        ):
            score += 6
        if (
            preferred_tables
            and not any(t.endswith("_ext") for t in preferred_tables)
            and "ext" in low
            and "2ext" in low
        ):
            score -= 6
        if early_bags:
            score += _var_resolve_role_boost(cm, early_bags)
        return -score

    ordered = sorted(closure, key=_priority)
    for cm in ordered[:40]:
        if "#" not in cm:
            continue
        cls, _, meth = cm.partition("#")
        sym = _read_symbol(project_id, fqcn=cls, method_name=meth)
        steps.append(
            {
                "tool": "get_symbol_code",
                "args": {"class_method": cm, "via": "calls_closure"},
                "ok": bool(sym),
            }
        )
        _ingest(sym, via="calls_closure")
        assemble_hit = roles.get("assemble")
        assemble_strong = bool(
            assemble_hit
            and _ASSEMBLE_METHOD_RE.search(assemble_hit.method_name or "")
        )
        if (
            roles.get("request_vo")
            and roles.get("bind")
            and assemble_strong
            and roles.get("persist_entity")
        ):
            break

    # 2b) Var-resolve set-site precision (MySQL bags + enriched CALLS facts)
    _emit(on_progress, "用 method_var_resolve 精化 set* / 绑定落点…")
    tp.set_sites = _apply_var_resolve_hints(
        project_id,
        entry=entry,
        field=field,
        closure=closure,
        roles=roles,
        steps=steps,
        notes=notes,
        ingest=_ingest,
    )
    # Prefer bag count from load step if early was empty
    for st in steps:
        if st.get("tool") == "load_method_var_resolves" and st.get("hit_count"):
            tp.bag_count = max(tp.bag_count, int(st["hit_count"]))

    # 3) Prefix / symbol search only for still-missing roles (fallback)
    missing_core = [
        r
        for r in ("request_vo", "bind", "assemble", "persist_entity")
        if r not in roles
    ]
    if missing_core:
        _emit(on_progress, "补搜缺失角色…")
        # Types first via filesystem again if VO/entity missing
        if "request_vo" in missing_core or "persist_entity" in missing_core:
            for sym in _read_prefix_java_types(project_id, entry, steps=steps):
                _ingest(sym, allow_roles=_TYPE_ROLES, via="prefix_type")
        for q in _seed_search_queries(entry):
            for sym in _search_and_read(project_id, q, steps=steps, top_k=3):
                # Off-closure search may fill gaps when CALLS lacked evidence;
                # temporarily clear closure gate by allowing all roles via
                # explicit fallback ingest when method roles still missing.
                if not closure_cms:
                    _ingest(sym, via="search_fallback")
                else:
                    # Type roles always OK; method roles only if in closure OR
                    # still missing after CALLS (recorded as weaker fallback).
                    meth = str(sym.get("method_name") or "")
                    simple = str(sym.get("fqcn") or "").split(".")[-1]
                    cm = f"{simple}#{meth}" if simple and meth else ""
                    if cm and cm in closure_cms:
                        _ingest(sym, via="calls_closure")
                    else:
                        # Fallback: allow type roles; for method roles only if
                        # that role is still missing (and note weakness).
                        allow = set(_TYPE_ROLES)
                        for mr in _METHOD_ROLES:
                            if mr not in roles:
                                allow.add(mr)
                        before = set(roles.keys())
                        _ingest(
                            sym,
                            allow_roles=frozenset(allow),
                            via="search_fallback",
                        )
                        newly = set(roles.keys()) - before
                        if newly & _METHOD_ROLES:
                            notes.append(
                                "部分绑定/组装来自前缀检索补漏（不在 CALLS 闭包内），"
                                "请以读码为准核对。"
                            )
            missing_core = [
                r
                for r in ("request_vo", "bind", "assemble", "persist_entity")
                if r not in roles
            ]
            if not missing_core:
                break

    # 4) If we have assemble but no entity, try to find Ext type from assemble code
    if roles.get("assemble") and not roles.get("persist_entity"):
        acode = roles["assemble"].raw_code
        for m in re.finditer(r"(?i)\b([A-Z][\w]*(?:Ext|Dmo|Entity|DO))\b", acode):
            tname = m.group(1)
            if tname in {"String", "Integer", "Boolean", "List", "Map"}:
                continue
            for sym in _search_and_read(project_id, tname, steps=steps, top_k=2):
                _ingest(sym, allow_roles=_TYPE_ROLES, via="search_fallback")
            if roles.get("persist_entity"):
                break

    # 5) If bind missing but VO found, search bind helpers under same prefix
    if roles.get("request_vo") and not roles.get("bind"):
        prefix = _entry_prefix(entry) or ""
        for q in (f"{prefix}Param", f"{prefix} check", f"{prefix}Controller"):
            if not q.strip():
                continue
            for sym in _search_and_read(project_id, q.strip(), steps=steps, top_k=3):
                meth = str(sym.get("method_name") or "")
                simple = str(sym.get("fqcn") or "").split(".")[-1]
                cm = f"{simple}#{meth}" if simple and meth else ""
                if closure_cms and cm and cm not in closure_cms:
                    # Still allow if bind missing — last resort
                    before = "bind" in roles
                    _ingest(
                        sym,
                        allow_roles=frozenset({"bind"}),
                        via="search_fallback",
                    )
                    if not before and "bind" in roles:
                        notes.append(
                            "入参绑定来自前缀补搜（不在 CALLS 闭包），请核对。"
                        )
                else:
                    _ingest(
                        sym,
                        via="calls_closure" if cm in closure_cms else "search_fallback",
                    )
                if roles.get("bind"):
                    break
            if roles.get("bind"):
                break

    # 6) Field-name constant class when bind peers use *Constant*.FOO keys
    if roles.get("bind") or roles.get("assemble"):
        _emit(on_progress, "检查绑定侧是否使用字段名常量类…")
        _attach_constant_role(
            project_id,
            field=field,
            roles=roles,
            steps=steps,
            notes=notes,
            ingest=_ingest,
        )

    # 7) User-named tables win: lock persist_entity(+N) to exact @TableName matches.
    if named_entity_syms:
        prev_ent = roles.get("persist_entity")
        for i, esym in enumerate(named_entity_syms):
            code = str(esym.get("raw_code") or "")
            tm = _TABLE_NAME_RE.search(code)
            table = tm.group(1) if tm else None
            role_key = "persist_entity" if i == 0 else f"persist_entity_{i + 1}"
            roles[role_key] = RoleHit(
                role=role_key,
                fqcn=str(esym.get("fqcn") or ""),
                method_name="",
                file_path=str(esym.get("file_path") or ""),
                raw_code=code,
                why=f"用户点名表 @TableName={table}" if table else "用户点名表实体",
                table_name=table,
                via="named_table",
                evidence=_evidence_snip(esym, role="persist_entity", via="named_table", entry=entry),
            )
        if (
            prev_ent
            and prev_ent.table_name
            and prev_ent.table_name.lower() not in preferred_tables
        ):
            notes.append(
                f"组装路径曾指向 `{prev_ent.table_name}`（{prev_ent.class_method}），"
                f"与用户点名表不一致；已改绑为点名表实体。"
            )
        # Prefer assemble that actually builds named-table entities (not Ext-only).
        if preferred_entity_names and closure_cms:
            cur_a = roles.get("assemble")
            cur_code = (cur_a.raw_code if cur_a else "") or ""
            touches = sum(1 for en in preferred_entity_names if en in cur_code)
            if touches == 0:
                for cm in sorted(closure_cms, key=_priority, reverse=True)[:40]:
                    if "#" not in cm:
                        continue
                    cls, _, meth = cm.partition("#")
                    asym = _read_symbol(project_id, fqcn=cls, method_name=meth)
                    if not asym:
                        continue
                    before = roles.get("assemble")
                    _ingest(asym, allow_roles=frozenset({"assemble"}), via="named_table_assemble")
                    after = roles.get("assemble")
                    if after and after is not before:
                        acode = after.raw_code or ""
                        if any(en in acode for en in preferred_entity_names):
                            notes.append(
                                f"落库组装已改绑到触及点名表实体的方法：`{after.class_method}`"
                            )
                            break

    tp.roles = roles
    # MP insert → usually no XML
    tp.need_mapper_xml = False
    if roles.get("orm") or (
        roles.get("persist_entity") and roles["persist_entity"].table_name
    ):
        tp.need_mapper_xml = False
        notes.append("写库形态像 MyBatis-Plus（@TableName / Mapper.insert）：一般不必改 Mapper.xml")

    for role, label in (
        ("request_vo", "请求 VO/DTO"),
        ("bind", "入参绑定/校验"),
        ("assemble", "落库组装"),
        ("persist_entity", "持久化实体"),
    ):
        if role not in roles:
            tp.gaps.append(label)

    if not tp.ok:
        notes.append(
            "入口闭包内未凑齐「请求侧 + 落库侧」证据；下列方案仅含已发现部分，缺口已标明。"
        )
    tp.notes = notes
    tp.elapsed_ms = (time.perf_counter() - t0) * 1000
    _emit(
        on_progress,
        f"角色发现完成：已定位 {', '.join(roles.keys()) or '无'}；缺口 {len(tp.gaps)}",
    )
    return tp


def _protocol_sync_score(cm: str, code: str = "") -> int:
    """Rank Class#method for outbound协议同步 relevance (no fixed class names)."""
    low = (cm or "").lower()
    meth = cm.split("#", 1)[-1] if "#" in cm else cm
    score = 0
    if _PROTOCOL_SYNC_METHOD_RE.search(meth):
        score += 12
    if "syncsign" in low or "signinfo" in low:
        score += 8
    body = code or ""
    if _PROTOCOL_SYNC_CODE_RE.search(body):
        score += 10
    if _SYNC_MAP_PUT_RE.search(body):
        score += 8
    if _FEIGN_SYNC_CALL_RE.search(body):
        score += 4
    if "协议" in body:
        score += 3
    return score


def _seed_protocol_sync_queries(entry_cm: str | None) -> list[str]:
    prefix = _entry_prefix(entry_cm) or ""
    qs = [
        "syncSignInfo",
        "syncSign",
        "协议同步",
        "syncSignMap",
    ]
    if prefix:
        qs.extend([f"{prefix} syncSign", f"{prefix} 协议"])
    return qs


def discover_protocol_sync_touchpoints(
    project_id: str,
    question: str,
    *,
    entry_cm: str | None = None,
    on_progress: ProgressCb = None,
) -> Touchpoints:
    """Discover roles to pass a field into 协议同步 outbound map/params.

    Evidence-based via entry CALLS + sync* code shapes — no hardcoded VO/method list.
    """
    import time

    from app.agent.question_intent import assumed_new_field_name

    t0 = time.perf_counter()
    field = assumed_new_field_name(question)
    steps: list[dict[str, Any]] = []
    notes: list[str] = []

    entry = _simple_cm(entry_cm) if entry_cm else ""
    if not entry or "#" not in entry:
        try:
            from app.agent.ask_retrieval import resolve_flow_entry_for_question

            hint = (
                str(entry_cm).strip()
                if entry_cm and "#" not in str(entry_cm)
                else None
            )
            entry = _simple_cm(
                resolve_flow_entry_for_question(
                    project_id, question, entry_hint=hint
                )
                or ""
            ) or ""
        except Exception:
            logger.debug(
                "protocol_sync resolve_flow_entry_for_question failed",
                exc_info=True,
            )
            entry = ""

    tp = Touchpoints(
        capability="protocol_sync",
        entry=entry or "(未定位入口)",
        field_name=field,
        steps=steps,
    )
    if not project_id:
        tp.gaps.append("缺少 project_id")
        tp.elapsed_ms = (time.perf_counter() - t0) * 1000
        return tp
    if not entry or "#" not in entry:
        tp.gaps.append("未能从问题定位业务入口（Controller#method）")
        notes.append("请补充业务接口名或 Controller#method 后再问。")
        tp.notes = notes
        tp.elapsed_ms = (time.perf_counter() - t0) * 1000
        return tp

    _emit(on_progress, f"定位入口 {entry}，发现协议同步透传触点…")
    notes.append(f"入口：`{entry}`")
    q_low = (question or "").lower()
    named_in_q = bool(
        field
        and field != "newField"
        and (field in (question or "") or _camel_to_snake(field) in q_low)
    )
    notes.append(
        f"字段：`{field}`"
        + ("（问题未指名，按此假设）" if not named_in_q else "")
    )

    roles: dict[str, RoleHit] = {}
    prefix = _entry_prefix(entry)
    closure = _calls_closure(project_id, entry)
    closure_cms = {_simple_cm(c) for c in closure if c and "#" in c}
    if entry and "#" in entry:
        closure_cms.add(_simple_cm(entry))
    tp.closure_size = len(closure)
    _emit(on_progress, f"入口 CALLS 闭包 {len(closure)} 个候选")

    def _ingest_sync(sym: dict[str, Any] | None, *, via: str = "") -> None:
        if not sym:
            return
        code = str(sym.get("raw_code") or "")
        meth = str(sym.get("method_name") or "")
        cm = ""
        simple = str(sym.get("fqcn") or "").split(".")[-1]
        if simple and meth:
            cm = f"{simple}#{meth}"
        score = _protocol_sync_score(cm or meth, code)
        if score < 8 and not (_PROTOCOL_SYNC_METHOD_RE.search(meth) and code):
            return
        why = "协议同步组装（*Map.put / syncSign* / 协议同步）"
        _merge_role(
            roles,
            "protocol_sync",
            sym,
            why,
            entry_prefix=prefix,
            closure_cms=closure_cms or None,
            via=via or "calls_closure",
            entry=entry,
        )
        hit = roles.get("protocol_sync")
        if hit:
            puts = list(_SYNC_MAP_PUT_RE.finditer(code))
            if puts:
                p0 = puts[0]
                hit.evidence = (
                    f"via={via or 'calls_closure'} · "
                    f"{p0.group(1)}.put({_const_class_simple(p0.group(2))}."
                    f"{p0.group(3)}, {p0.group(4)}.get…)"
                )
            elif _FEIGN_SYNC_CALL_RE.search(code):
                fm = _FEIGN_SYNC_CALL_RE.search(code)
                hit.evidence = (
                    f"via={via or 'calls_closure'} · "
                    f"{fm.group(1)}.{fm.group(2)}(…)"  # type: ignore[union-attr]
                )

    # Prefer closure members whose names look like sync*, then score by body.
    ranked = sorted(
        [c for c in closure if c and "#" in c],
        key=lambda cm: -_protocol_sync_score(cm),
    )
    # Always try name-matched methods first even if score from name alone is mid.
    name_hits = [c for c in ranked if _PROTOCOL_SYNC_METHOD_RE.search(c.split("#", 1)[-1])]
    ordered = name_hits + [c for c in ranked if c not in name_hits]
    for cm in ordered[:35]:
        cls, _, meth = cm.partition("#")
        sym = _read_symbol(project_id, fqcn=cls, method_name=meth)
        steps.append(
            {
                "tool": "get_symbol_code",
                "args": {"class_method": cm, "via": "calls_closure"},
                "ok": bool(sym),
                "sync_score": _protocol_sync_score(cm, str((sym or {}).get("raw_code") or "")),
            }
        )
        _ingest_sync(sym, via="calls_closure")
        if roles.get("protocol_sync") and _SYNC_MAP_PUT_RE.search(
            roles["protocol_sync"].raw_code or ""
        ):
            break

    # Fallback search when CALLS missed private sync helpers.
    if not roles.get("protocol_sync"):
        _emit(on_progress, "闭包未命中协议同步，补搜 sync*/协议同步…")
        for q in _seed_protocol_sync_queries(entry):
            for sym in _search_and_read(project_id, q, steps=steps, top_k=4):
                _ingest_sync(sym, via="search_fallback")
                if roles.get("protocol_sync"):
                    notes.append(
                        "协议同步方法来自符号检索补漏（请以读码为准核对是否属本入口主链）。"
                    )
                    break
            if roles.get("protocol_sync"):
                break

    sync = roles.get("protocol_sync")
    # Sync inbound VO from method params / get* receivers in put lines.
    if sync:
        code = sync.raw_code or ""
        vo_names: list[str] = []
        for m in _SYNC_PARAM_VO_RE.finditer(code.split("{", 1)[0] if "{" in code else code[:400]):
            name = m.group(1)
            if name not in vo_names:
                vo_names.append(name)
        if not vo_names:
            for m in _SYNC_MAP_PUT_RE.finditer(code):
                # Infer from getter receiver only — type still unknown; search *ReqVo under prefix.
                pass
            if prefix:
                vo_names.append(f"{prefix}ReqVo")
        for vname in vo_names[:3]:
            for sym in _search_and_read(project_id, vname, steps=steps, top_k=2):
                simple = str(sym.get("fqcn") or "").split(".")[-1]
                if not _VO_NAME_RE.search(simple):
                    continue
                # Prefer type-level read (full class) for field decl peers.
                if sym.get("method_name"):
                    full = _read_symbol(
                        project_id, fqcn=str(sym.get("fqcn") or simple)
                    )
                    if full:
                        sym = full
                _merge_role(
                    roles,
                    "request_vo",
                    sym,
                    "协议同步入参 VO（sync 方法参数/前缀 ReqVo）",
                    entry_prefix=prefix,
                    via="sync_param",
                    entry=entry,
                )
                vo_hit = roles.get("request_vo")
                if vo_hit:
                    vo_hit.analog_fields = _analog_fields_from_vo(vo_hit.raw_code or "")
                if roles.get("request_vo"):
                    break
            if roles.get("request_vo"):
                break

    # Constant class from syncSignMap.put(SomeConstants.FOO, …)
    if sync:
        code = sync.raw_code or ""
        const_refs = [
            (m.group(2), m.group(3)) for m in _SYNC_MAP_PUT_RE.finditer(code)
        ]
        if const_refs:
            class_scores: dict[str, int] = {}
            for cref, _cname in const_refs:
                class_scores[_const_class_simple(cref)] = (
                    class_scores.get(_const_class_simple(cref), 0) + 1
                )
            best_simple = max(class_scores, key=class_scores.get) if class_scores else ""
            if best_simple:
                for sym in _search_and_read(
                    project_id, best_simple, steps=steps, top_k=2
                ):
                    simple = str(sym.get("fqcn") or "").split(".")[-1]
                    if simple != best_simple and not simple.endswith(best_simple):
                        continue
                    if sym.get("method_name"):
                        full = _read_symbol(
                            project_id, fqcn=str(sym.get("fqcn") or simple)
                        )
                        if full:
                            sym = full
                    code_c = str(sym.get("raw_code") or "")
                    if not _CONST_STRING_DEF_RE.search(code_c):
                        continue
                    vo = roles.get("request_vo")
                    peers = list((vo.analog_fields if vo else None) or [])
                    peers_hit = _peer_const_defs(code_c, peer_fields=peers, limit=4)
                    if not peers_hit:
                        for m in list(_CONST_STRING_DEF_RE.finditer(code_c))[:2]:
                            peers_hit.append(f'{m.group(1)} = "{m.group(2)}"')
                    proposed = f'{_field_to_const_name(field)} = "{field}"'
                    _merge_role(
                        roles,
                        "constant",
                        sym,
                        f"同步 Map 键使用 `{best_simple}.…`",
                        via="sync_const_ref",
                        entry=entry,
                        proposed_const=proposed,
                        peer_const_examples=peers_hit,
                    )
                    hit = roles.get("constant")
                    if hit:
                        hit.proposed_const = proposed
                        hit.peer_const_examples = peers_hit
                        hit.evidence = (
                            f"sync put 使用 {best_simple}.* · 建议新增 `{proposed}`"
                        )
                    notes.append(
                        f"发现同步键常量类 `{best_simple}`；建议新增 `{proposed}`"
                    )
                    break

    tp.roles = roles
    tp.need_mapper_xml = False
    for role, label in (
        ("protocol_sync", "协议同步组装"),
        ("request_vo", "同步入参 VO"),
    ):
        if role not in roles:
            tp.gaps.append(label)
    if not tp.ok:
        notes.append(
            "未定位到协议同步 *Map.put / syncSign* 触点；请确认入口或补充「协议同步」实词。"
        )
    else:
        notes.append(
            "下游 Feign/Client 通常接收 Map/Request body，一般不必改方法签名。"
        )
    tp.notes = notes
    tp.elapsed_ms = (time.perf_counter() - t0) * 1000
    _emit(
        on_progress,
        f"协议同步触点：已定位 {', '.join(roles.keys()) or '无'}；缺口 {len(tp.gaps)}",
    )
    return tp


def _org_external_score(cm: str, code: str = "") -> int:
    """Score a Class#method for external-pass relevance (Client/Feign/Map.put shapes)."""
    del cm  # method name not used for product enums
    score = 0
    if _EXTERNAL_CALL_RE.search(code):
        score += 8
    return score


def discover_composite_field_touchpoints(
    project_id: str,
    question: str,
    *,
    entry_cm: str | None = None,
    on_progress: ProgressCb = None,
) -> Touchpoints:
    """Add-field roles + check-gate / external-pass sites (evidence-based).

    Reuses add_field discovery; then scans entry CALLS for org-quota / single-batch
    external call shapes — no hardcoded Controller/method checklist.
    """
    import time

    t0 = time.perf_counter()
    base = discover_add_field_touchpoints(
        project_id,
        question,
        entry_cm=entry_cm,
        on_progress=on_progress,
    )
    tp = Touchpoints(
        capability="composite_field",
        entry=base.entry,
        field_name=base.field_name,
        roles=dict(base.roles),
        need_mapper_xml=base.need_mapper_xml,
        gaps=list(base.gaps),
        notes=list(base.notes),
        steps=list(base.steps),
        set_sites=list(base.set_sites or []),
        closure_size=base.closure_size,
        bag_count=base.bag_count,
    )
    entry = _simple_cm(tp.entry) if tp.entry and "#" in tp.entry else ""
    if not project_id or not entry:
        tp.elapsed_ms = (time.perf_counter() - t0) * 1000
        return tp

    _emit(on_progress, "在入口 CALLS 上发现关卡校验 / 外部透传触点…")
    prefix = _entry_prefix(entry)
    closure = _calls_closure(project_id, entry)
    tp.closure_size = max(tp.closure_size, len(closure))
    closure_cms = {_simple_cm(c) for c in closure if c and "#" in c}
    if entry:
        closure_cms.add(entry)

    ranked = sorted(
        [c for c in closure if c and "#" in c],
        key=lambda cm: -_org_external_score(cm),
    )
    # Prefer name-matched org/amount/productVersion methods first.
    name_hits = [
        c for c in ranked if _ORG_GATE_METHOD_RE.search(c.split("#", 1)[-1])
    ]
    ordered = name_hits + [c for c in ranked if c not in name_hits]

    def _ingest_org(sym: dict[str, Any] | None, *, via: str = "") -> None:
        if not sym:
            return
        code = str(sym.get("raw_code") or "")
        meth = str(sym.get("method_name") or "")
        simple = str(sym.get("fqcn") or "").split(".")[-1]
        cm = f"{simple}#{meth}" if simple and meth else meth
        score = _org_external_score(cm, code)
        if score < 10:
            return
        is_single = bool(_SINGLE_BATCH_CODE_RE.search(code))
        is_org = bool(
            _ORG_GATE_METHOD_RE.search(meth) or _ORG_GATE_CODE_RE.search(code)
        )
        has_ext = bool(_EXTERNAL_CALL_RE.search(code))
        if is_org:
            _merge_role(
                tp.roles,
                "org_check_gate",
                sym,
                "org/limit check gate",
                entry_prefix=prefix,
                closure_cms=closure_cms or None,
                via=via or "calls_closure",
                entry=entry,
            )
            hit = tp.roles.get("org_check_gate")
            if hit:
                hit.evidence = f"via={via or 'calls_closure'} · score={score}"
        if is_single and has_ext:
            _merge_role(
                tp.roles,
                "single_batch_external",
                sym,
                "分支上的外部调用 / Map 组装",
                entry_prefix=prefix,
                closure_cms=closure_cms or None,
                via=via or "calls_closure",
                entry=entry,
            )
            hit = tp.roles.get("single_batch_external")
            if hit:
                hit.evidence = f"via={via or 'calls_closure'} · 分支+外部调用"
        elif is_single or (is_org and has_ext):
            _merge_role(
                tp.roles,
                "external_pass",
                sym,
                "关卡路径上可透传字段的外部调用点",
                entry_prefix=prefix,
                closure_cms=closure_cms or None,
                via=via or "calls_closure",
                entry=entry,
            )

    for cm in ordered[:40]:
        cls, _, meth = cm.partition("#")
        sym = _read_symbol(project_id, fqcn=cls, method_name=meth)
        tp.steps.append(
            {
                "tool": "get_symbol_code",
                "args": {"class_method": cm, "via": "org_external_scan"},
                "ok": bool(sym),
                "org_score": _org_external_score(
                    cm, str((sym or {}).get("raw_code") or "")
                ),
            }
        )
        _ingest_org(sym, via="calls_closure")
        if tp.roles.get("org_check_gate") and (
            tp.roles.get("single_batch_external") or tp.roles.get("external_pass")
        ):
            break

    if not (
        tp.roles.get("org_check_gate")
        or tp.roles.get("single_batch_external")
        or tp.roles.get("external_pass")
    ):
        # Fallback: search with user question tokens + generic external-call shapes.
        _emit(on_progress, "闭包未命中外部透传触点，按问题词补搜…")
        q_tokens = [
            t
            for t in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,40}|[\u4e00-\u9fff]{2,8}", question or "")
            if t.lower() not in {"field", "null", "string", "改动", "清单", "哪些"}
        ][:6]
        for q in (
            " ".join(q_tokens[:4]) if q_tokens else "",
            f"{prefix} Client put" if prefix else "Client put Map",
            f"{prefix} Feign" if prefix else "Feign",
        ):
            if not q.strip():
                continue
            for sym in _search_and_read(project_id, q, steps=tp.steps, top_k=4):
                _ingest_org(sym, via="search_fallback")
            if tp.roles.get("external_pass"):
                tp.notes.append(
                    "外部透传触点来自符号检索补漏（请以读码核对是否属本入口主链）。"
                )
                break

    for role, label in (
        ("org_check_gate", "关卡校验"),
        ("single_batch_external", "分支外部透传"),
    ):
        if role not in tp.roles and role == "org_check_gate":
            if not (
                tp.roles.get("external_pass")
                or tp.roles.get("single_batch_external")
            ):
                tp.gaps.append(label)
        if role == "single_batch_external" and role not in tp.roles:
            if not tp.roles.get("external_pass"):
                tp.gaps.append(label)

    if tp.ok:
        tp.notes.append(
            "复合方案：先补齐入参→落库，再在关卡/外部调用分支把字段带入外部请求。"
        )
    else:
        tp.notes.append(
            "未同时定位到落库链路与关卡/外部透传触点；请确认入口或补充关卡实词。"
        )
    tp.elapsed_ms = (time.perf_counter() - t0) * 1000
    _emit(
        on_progress,
        f"复合触点：已定位 {', '.join(tp.roles.keys()) or '无'}；缺口 {len(tp.gaps)}",
    )
    return tp
