"""Deterministic render of add-field change plans from Touchpoints evidence."""
from __future__ import annotations

import re

from app.agent.change_impact.discover import (
    RoleHit,
    Touchpoints,
    _camel_to_snake,
    _field_to_const_name,
    _setter_name,
)

_ROLE_ORDER = (
    ("constant", "常量"),
    ("request_vo", "请求 VO"),
    ("bind", "入参绑定"),
    ("assemble", "组装落库"),
    ("persist_entity", "实体"),
)


def _persist_role_keys(tp: Touchpoints) -> list[str]:
    keys = [
        k
        for k in (tp.roles or {})
        if k == "persist_entity" or re.match(r"^persist_entity_\d+$", k)
    ]

    def _ord(k: str) -> int:
        if k == "persist_entity":
            return 1
        m = re.match(r"^persist_entity_(\d+)$", k)
        return int(m.group(1)) if m else 99

    return sorted(keys, key=_ord)


def _role_iter(tp: Touchpoints) -> list[tuple[str, str]]:
    """Ordered (role, label) including persist_entity_N for multi-table plans."""
    out: list[tuple[str, str]] = []
    for role, label in _ROLE_ORDER:
        if role == "persist_entity":
            for pk in _persist_role_keys(tp):
                hit = tp.roles.get(pk)
                table = (hit.table_name if hit else None) or ""
                simple = ((hit.fqcn if hit else "") or "").split(".")[-1]
                lab = "实体"
                if table or simple:
                    lab = f"实体 {simple or ''}@{table}".strip()
                out.append((pk, lab))
        else:
            out.append((role, label))
    return out

_VO_VAR_RE = re.compile(r"\b([a-z]\w*)\.(?:set[A-Z]\w*)\s*\(", re.M)
_EXT_SET_RE = re.compile(
    r"\b([a-z]\w*)\.set[A-Z]\w*\s*\(\s*([a-z]\w*)\.get[A-Z]\w*\s*\("
)
_BIND_PEER_RE = re.compile(
    r"(?m)^([ \t]*)(?://[^\n]*\n\1)?"
    r"String\s+(\w+)\s*=\s*\(String\)\s*"
    r"(reqMap|paramMap|map)\.get\s*\(([^;]+?)\)\s*;\s*\n"
    r"\1([a-z]\w*)\.set([A-Z]\w*)\s*\(\s*\2\s*\)\s*;"
)
_ASSEMBLE_PEER_RE = re.compile(
    r"(?m)^([ \t]*)(?://[ \t]*([^\n]*)\n\1)?"
    r"([a-z]\w*)\.set([A-Z]\w*)\s*\(\s*"
    r"([a-z]\w*)\.get\4\s*\(\s*\)\s*\)\s*;"
)
_CONST_PEER_RE = re.compile(
    r"(?m)^([ \t]*)(/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n\1)?"
    r"public\s+static\s+final\s+String\s+([A-Z][A-Z0-9_]*)\s*=\s*[\"']([^\"']+)[\"']\s*;"
)
# Single String field only — never span multiple declarations.
_FIELD_PEER_RE = re.compile(
    r"(?m)^([ \t]*)(/\*\*[^*]*\*+(?:[^/*][^*]*\*+)*/\s*\n\1)?"
    r"((?:@[A-Za-z][\w.]*(?:\([^;\n]*\))?\s*\n\1)*)"
    r"private\s+String\s+(\w+)\s*;"
)
_PEER_LABELISH_RE = re.compile(
    r"(?i)(?:Label|Source|Scene|Channel|Tag|Flag|Remark|Desc)$|"
    r"(?:Label|Scene|Channel|Source)"
)
_PEER_CORE_RE = re.compile(
    r"(?i)amount|rate|term|period|custid|userid|serial|product|orderid|trans|"
    r"interest|quota|mobile|bankno|cardid"
)


def _getter_name(field: str) -> str:
    if not field:
        return "getField"
    return "get" + field[:1].upper() + field[1:]


def _const_simple(hit: RoleHit) -> str:
    return (hit.fqcn or "").split(".")[-1] or "Constants"


def _path_label(hit: RoleHit) -> str:
    fp = (hit.file_path or "").strip()
    cm = hit.class_method
    if fp and cm and "#" in cm:
        return f"`{fp}` · `{cm}`"
    if fp:
        return f"`{fp}`"
    return f"`{cm}`" if cm else "`(未知)`"


def _indent_of(code: str, default: str = "        ") -> str:
    for line in (code or "").splitlines():
        m = re.match(r"^([ \t]+)\S", line)
        if m:
            return m.group(1)
    return default


def _bind_get_expr(tp: Touchpoints, field: str) -> str:
    const = tp.roles.get("constant")
    if const and const.proposed_const:
        name = const.proposed_const.split("=", 1)[0].strip()
        return f"{_const_simple(const)}.{name}"
    bind = tp.roles.get("bind")
    code = (bind.raw_code if bind else "") or ""
    m = re.search(
        r"(?i)(?:reqMap|paramMap|map)\.get\s*\(\s*"
        r"((?:\w+\.)*\w*(?:Constant|Constants))\s*\.\s*[A-Z][A-Z0-9_]*\s*\)",
        code,
    )
    if m:
        simple = m.group(1).rsplit(".", 1)[-1]
        return f"{simple}.{_field_to_const_name(field)}"
    return f'"{field}"'


def _infer_vo_receiver(code: str) -> str:
    for m in _VO_VAR_RE.finditer(code or ""):
        name = m.group(1)
        if any(k in name.lower() for k in ("req", "vo", "apply", "request", "param")):
            return name
    m = _VO_VAR_RE.search(code or "")
    return m.group(1) if m else "reqVo"


def _infer_assemble_vars(code: str) -> tuple[str, str]:
    m = _EXT_SET_RE.search(code or "")
    if m:
        return m.group(1), m.group(2)
    ext = "extDmo"
    for m2 in re.finditer(r"\b([a-z]\w*)\.set[A-Z]\w*\s*\(", code or ""):
        name = m2.group(1)
        if any(k in name.lower() for k in ("ext", "dmo", "entity", "po", "do")):
            ext = name
            break
    return ext, _infer_vo_receiver(code)


def _score_peer(
    peer: dict[str, str],
    *,
    index: int,
    total: int,
    analog_fields: list[str] | None = None,
) -> int:
    score = index
    local = peer.get("local") or ""
    if not local and peer.get("setter_tail"):
        st = peer["setter_tail"]
        local = st[:1].lower() + st[1:] if st else ""
    analogs = {a.lower() for a in (analog_fields or [])}
    if local.lower() in analogs:
        score += 40
    if peer.get("comment"):
        score += 12
    if _PEER_LABELISH_RE.search(local):
        score += 18
    if _PEER_CORE_RE.search(local):
        score -= 20
    if total > 0 and index >= max(0, total - max(3, total // 3)):
        score += 10
    return score


def _pick_bind_peer(
    code: str,
    *,
    prefer_const: bool,
    analog_fields: list[str] | None = None,
) -> dict[str, str] | None:
    peers: list[dict[str, str]] = []
    for m in _BIND_PEER_RE.finditer(code or ""):
        indent, local, map_var, get_arg, vo, setter_tail = (
            m.group(1),
            m.group(2),
            m.group(3),
            m.group(4).strip(),
            m.group(5),
            m.group(6),
        )
        full = m.group(0)
        comment = ""
        cm = re.match(r"[ \t]*(//[^\n]*)\n", full)
        if cm:
            comment = cm.group(1)
        uses_const = bool(re.search(r"(?i)Constant|Constants", get_arg))
        peers.append(
            {
                "indent": indent,
                "comment": comment,
                "local": local,
                "map_var": map_var,
                "vo": vo,
                "uses_const": "1" if uses_const else "0",
            }
        )
    if not peers:
        return None
    pool = peers
    if prefer_const:
        const_peers = [p for p in peers if p["uses_const"] == "1"]
        if const_peers:
            pool = const_peers
    return max(
        enumerate(pool),
        key=lambda it: _score_peer(
            it[1], index=it[0], total=len(pool), analog_fields=analog_fields
        ),
    )[1]


def _pick_assemble_peer(
    code: str,
    *,
    analog_fields: list[str] | None = None,
) -> dict[str, str] | None:
    peers: list[dict[str, str]] = []
    for m in _ASSEMBLE_PEER_RE.finditer(code or ""):
        indent, comment_text, ext, setter_tail, vo = (
            m.group(1),
            (m.group(2) or "").strip(),
            m.group(3),
            m.group(4),
            m.group(5),
        )
        local = setter_tail[:1].lower() + setter_tail[1:] if setter_tail else ""
        peers.append(
            {
                "indent": indent,
                "comment": comment_text,
                "ext": ext,
                "vo": vo,
                "setter_tail": setter_tail,
                "local": local,
            }
        )
    if not peers:
        return None
    return max(
        enumerate(peers),
        key=lambda it: _score_peer(
            it[1], index=it[0], total=len(peers), analog_fields=analog_fields
        ),
    )[1]


def _snippet_const(hit: RoleHit, field: str) -> str:
    const_name = _field_to_const_name(field)
    peers = list(_CONST_PEER_RE.finditer(hit.raw_code or ""))
    indent = "    "
    eq_sp = " = "
    has_jd = True
    if peers:
        want = set()
        for ex in hit.peer_const_examples or []:
            m = re.search(r'=\s*"([^"]+)"', ex)
            if m:
                want.add(m.group(1))
        chosen = peers[-1]
        for p in peers:
            if p.group(4) in want:
                chosen = p
                break
        indent = chosen.group(1) or "    "
        has_jd = bool(chosen.group(2))
        if "  =" in chosen.group(0):
            eq_sp = "  = "
    lines: list[str] = []
    if has_jd:
        lines.append(f"{indent}/**")
        lines.append(f"{indent} * {field}")
        lines.append(f"{indent} */")
    lines.append(
        f"{indent}public static final String {const_name}{eq_sp}\"{field}\";"
    )
    return "\n".join(lines)


def _snippet_field_decl(
    code: str,
    field: str,
    *,
    analog_fields: list[str] | None = None,
) -> str:
    """Emit exactly one field declaration, cloned from a single peer."""
    peers = list(_FIELD_PEER_RE.finditer(code or ""))
    indent = "    "
    has_jd = True
    has_api = False
    if peers:
        want = set(analog_fields or [])
        chosen = peers[-1]
        for p in peers:
            if p.group(4) in want:
                chosen = p
                break
        # Prefer label-ish String peers near end.
        scored = []
        for i, p in enumerate(peers):
            scored.append(
                (
                    _score_peer(
                        {"local": p.group(4), "comment": p.group(2) or ""},
                        index=i,
                        total=len(peers),
                        analog_fields=analog_fields,
                    ),
                    p,
                )
            )
        scored.sort(key=lambda x: x[0])
        chosen = scored[-1][1]
        indent = chosen.group(1) or "    "
        has_jd = bool(chosen.group(2))
        annos = chosen.group(3) or ""
        has_api = "@ApiModelProperty" in annos
    lines: list[str] = []
    if has_jd:
        lines.append(f"{indent}/**")
        lines.append(f"{indent} * {field}")
        lines.append(f"{indent} */")
    if has_api:
        lines.append(f'{indent}@ApiModelProperty("{field}")')
    lines.append(f"{indent}private String {field};")
    return "\n".join(lines)


def _snippet_bind(tp: Touchpoints, hit: RoleHit, field: str, setter: str) -> str:
    get_expr = _bind_get_expr(tp, field)
    vo_hit = tp.roles.get("request_vo")
    analogs = list(vo_hit.analog_fields) if vo_hit else []
    prefer_const = bool(tp.roles.get("constant")) or "Constant" in get_expr
    peer = _pick_bind_peer(
        hit.raw_code or "", prefer_const=prefer_const, analog_fields=analogs
    )
    ind = (peer or {}).get("indent") or _indent_of(hit.raw_code or "")
    map_var = (peer or {}).get("map_var") or "reqMap"
    vo = (peer or {}).get("vo") or _infer_vo_receiver(hit.raw_code or "")
    lines: list[str] = []
    if peer and peer.get("comment"):
        lines.append(f"{ind}// {field}")
    lines.append(f"{ind}String {field} = (String) {map_var}.get({get_expr});")
    lines.append(f"{ind}{vo}.{setter}({field});")
    return "\n".join(lines)


def _snippet_assemble(
    hit: RoleHit,
    field: str,
    setter: str,
    *,
    analog_fields: list[str] | None = None,
) -> str:
    peer = _pick_assemble_peer(hit.raw_code or "", analog_fields=analog_fields)
    ext, vo = _infer_assemble_vars(hit.raw_code or "")
    ind = (peer or {}).get("indent") or _indent_of(hit.raw_code or "")
    if peer:
        ext, vo = peer.get("ext") or ext, peer.get("vo") or vo
    getter = _getter_name(field)
    lines: list[str] = []
    if peer and peer.get("comment"):
        lines.append(f"{ind}// {field}")
    lines.append(f"{ind}{ext}.{setter}({vo}.{getter}());")
    return "\n".join(lines)


def _java_block(code: str) -> list[str]:
    return ["```java", code.rstrip(), "```"]


def _section(n: int, title: str, hit: RoleHit, code: str) -> list[str]:
    return [
        f"### {n}. {title}",
        "",
        _path_label(hit),
        "",
        *_java_block(code),
        "",
    ]


def _peer_style_excerpt(code: str, *, limit: int = 12) -> str:
    """Short peer excerpt for AI style alignment (not dumped to user)."""
    lines = [ln for ln in (code or "").splitlines() if ln.strip()]
    if not lines:
        return ""
    # Prefer trailing optional-field region.
    chunk = lines[-limit:] if len(lines) > limit else lines
    return "\n".join(chunk)


def build_add_field_evidence_pack(tp: Touchpoints) -> dict:
    """Structured evidence + suggested snippets for AI / deterministic render."""
    field = tp.field_name or "newField"
    setter = _setter_name(field)
    col = _camel_to_snake(field)
    table = tp.table_name
    vo = tp.roles.get("request_vo")
    analogs = list((vo.analog_fields if vo else None) or [])

    items: list[dict] = []
    for role, label in _role_iter(tp):
        hit = tp.roles.get(role)
        if not hit:
            continue
        if role == "constant":
            suggested = _snippet_const(hit, field)
            peer = _peer_style_excerpt(hit.raw_code or "", limit=8)
        elif role == "request_vo":
            suggested = _snippet_field_decl(
                hit.raw_code or "", field, analog_fields=analogs
            )
            peer = _peer_style_excerpt(hit.raw_code or "", limit=10)
        elif role == "bind":
            suggested = _snippet_bind(tp, hit, field, setter)
            peer = _peer_style_excerpt(hit.raw_code or "", limit=14)
        elif role == "assemble":
            suggested = _snippet_assemble(
                hit, field, setter, analog_fields=analogs
            )
            peer = _peer_style_excerpt(hit.raw_code or "", limit=14)
        else:  # persist_entity / persist_entity_N
            suggested = _snippet_field_decl(hit.raw_code or "", field)
            peer = _peer_style_excerpt(hit.raw_code or "", limit=10)
        items.append(
            {
                "role": role,
                "label": label,
                "file_path": hit.file_path or "",
                "class_method": hit.class_method,
                "fqcn": hit.fqcn or "",
                "table_name": hit.table_name,
                "analog_fields": list(hit.analog_fields or [])[:6],
                "proposed_const": hit.proposed_const or "",
                "suggested_code": suggested,
                "peer_style": peer,
            }
        )

    tables = list(tp.persist_table_names or [])
    if not tables and table:
        tables = [table]
    if tables:
        ddl_parts = [
            (
                f"ALTER TABLE `{t}`\n"
                f"  ADD COLUMN `{col}` varchar(64) DEFAULT NULL COMMENT '{field}';"
            )
            for t in tables
        ]
        ddl = "\n\n".join(ddl_parts)
    else:
        ddl = f"-- 未读到 @TableName，表名待补；建议列名 `{col}`"

    return {
        "capability": "add_field",
        "question_field": field,
        "entry": tp.entry,
        "table_name": tables[0] if tables else table,
        "table_names": tables,
        "column_name": col,
        "need_mapper_xml": bool(tp.need_mapper_xml),
        "gaps": list(tp.gaps or []),
        "notes": [
            n
            for n in (tp.notes or [])
            if "假设" in n or "未指名" in n or "常量" in n
        ][:4],
        "change_items": items,
        "suggested_ddl": ddl,
    }


def render_add_field_plan(tp: Touchpoints, *, question: str = "") -> str:
    """Compact professional plan: 结论 → 改动清单(可粘贴) → DDL."""
    pack = build_add_field_evidence_pack(tp)
    field = pack["question_field"]
    table = pack["table_name"]
    tables = list(pack.get("table_names") or ([] if not table else [table]))
    items = pack["change_items"]

    roles_present = [it["label"] for it in items]
    if roles_present:
        table_part = ""
        if tables:
            table_part = "至 " + "、".join(f"`{t}`" for t in tables)
        elif table:
            table_part = f"至 `{table}`"
        conclusion = (
            f"入口 `{tp.entry}` 新增入参 `{field}` 并落库"
            + table_part
            + "，按 "
            + " → ".join(roles_present)
            + " 补齐。"
        )
    else:
        conclusion = (
            f"入口 `{tp.entry}` 下尚未定位到可改动触点，请补充业务名或入口后再问。"
        )

    assumed = ""
    if any("未指名" in n or "假设" in n for n in (tp.notes or [])):
        assumed = f"字段按 `{field}` 假设。"
    head = (assumed + " " if assumed else "") + conclusion

    sections: list[str] = []
    for i, it in enumerate(items, 1):
        hit = tp.roles.get(it["role"])
        if not hit:
            continue
        sections.extend(
            _section(i, it["label"], hit, it.get("suggested_code") or "")
        )

    if not sections:
        sections = ["（尚未定位到可改动的常量 / VO / 绑定 / 组装 / 实体。）", ""]

    xml_note = ""
    if tp.roles.get("persist_entity") and not tp.need_mapper_xml:
        xml_note = "MyBatis-Plus insert，一般不必改 Mapper.xml。"

    lines = [
        "## 结论",
        "",
        head.strip(),
        "",
        "## 改动清单",
        "",
        *sections,
        "## DDL",
        "",
        "```sql",
        pack["suggested_ddl"],
        "```",
    ]
    if xml_note:
        lines.extend(["", xml_note])
    if tp.gaps:
        lines.extend(["", f"缺口：{'、'.join(tp.gaps)}。"])

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return _append_changed_method_section(text, tp, question=question)


_PROTOCOL_ROLE_ORDER = (
    ("constant", "常量"),
    ("request_vo", "同步入参 VO"),
    ("protocol_sync", "协议同步组装"),
)

_SYNC_PUT_PEER_RE = re.compile(
    r"(?m)^([ \t]*)(?://[^\n]*\n\1)?"
    r"([a-z]\w*Map)\s*\.\s*put\s*\(\s*"
    r"((?:\w+\.)*\w*(?:Constant|Constants))\s*\.\s*([A-Z][A-Z0-9_]*)\s*,\s*"
    r"([a-z]\w*)\.get([A-Z]\w*)\s*\(\s*\)\s*\)\s*;"
)


def _snippet_protocol_put(tp: Touchpoints, hit: RoleHit, field: str) -> str:
    """Emit map.put(CONST, reqVo.getField()) cloned from peer put style."""
    code = hit.raw_code or ""
    const = tp.roles.get("constant")
    const_name = _field_to_const_name(field)
    if const and const.proposed_const:
        const_name = const.proposed_const.split("=", 1)[0].strip() or const_name
    const_simple = _const_simple(const) if const else ""

    peers = list(_SYNC_PUT_PEER_RE.finditer(code))
    peer = peers[-1] if peers else None
    if peer:
        ind = peer.group(1) or "        "
        map_var = peer.group(2)
        cref = const_simple or peer.group(3).rsplit(".", 1)[-1]
        vo = peer.group(5)
        lines: list[str] = []
        if re.match(r"[ \t]*//", peer.group(0)):
            lines.append(f"{ind}// {field}")
        lines.append(
            f"{ind}{map_var}.put({cref}.{const_name}, {vo}.{_getter_name(field)}());"
        )
        return "\n".join(lines)

    # No peer put found — still avoid inventing a project-specific constant class.
    ind = _indent_of(code)
    cref = const_simple or "Constants"
    return (
        f"{ind}// {field}\n"
        f"{ind}syncSignMap.put({cref}.{const_name}, reqVo.{_getter_name(field)}());"
    )


def build_protocol_sync_evidence_pack(tp: Touchpoints) -> dict:
    """Evidence pack for extending a field into 协议同步 outbound params."""
    field = tp.field_name or "newField"
    vo = tp.roles.get("request_vo")
    analogs = list((vo.analog_fields if vo else None) or [])
    items: list[dict] = []
    for role, label in _PROTOCOL_ROLE_ORDER:
        hit = tp.roles.get(role)
        if not hit:
            continue
        if role == "constant":
            suggested = _snippet_const(hit, field)
            peer = _peer_style_excerpt(hit.raw_code or "", limit=8)
        elif role == "request_vo":
            suggested = _snippet_field_decl(
                hit.raw_code or "", field, analog_fields=analogs
            )
            peer = _peer_style_excerpt(hit.raw_code or "", limit=10)
        else:
            suggested = _snippet_protocol_put(tp, hit, field)
            peer = _peer_style_excerpt(hit.raw_code or "", limit=16)
        items.append(
            {
                "role": role,
                "label": label,
                "file_path": hit.file_path or "",
                "class_method": hit.class_method,
                "fqcn": hit.fqcn or "",
                "table_name": hit.table_name,
                "analog_fields": list(hit.analog_fields or [])[:6],
                "proposed_const": hit.proposed_const or "",
                "suggested_code": suggested,
                "peer_style": peer,
            }
        )
    return {
        "capability": "protocol_sync",
        "question_field": field,
        "entry": tp.entry,
        "table_name": None,
        "column_name": _camel_to_snake(field),
        "need_mapper_xml": False,
        "need_ddl": False,
        "gaps": list(tp.gaps or []),
        "notes": [
            n
            for n in (tp.notes or [])
            if any(k in n for k in ("假设", "未指名", "常量", "Feign", "Client"))
        ][:4],
        "change_items": items,
        "suggested_ddl": "",
    }


def render_protocol_sync_plan(tp: Touchpoints) -> str:
    """结论 → 改动清单（透传到协议同步）；无 DDL。"""
    pack = build_protocol_sync_evidence_pack(tp)
    field = pack["question_field"]
    items = pack["change_items"]
    roles_present = [it["label"] for it in items]
    if roles_present:
        conclusion = (
            f"入口 `{tp.entry}` 将字段 `{field}` 在协议同步时作为入参透传，按 "
            + " → ".join(roles_present)
            + " 补齐；下游 Feign 一般不必改签名。"
        )
    else:
        conclusion = (
            f"入口 `{tp.entry}` 下尚未定位到协议同步组装触点，"
            f"请补充业务名或「协议同步/syncSign」实词后再问。"
        )
    assumed = ""
    if any("未指名" in n or "假设" in n for n in (tp.notes or [])):
        assumed = f"字段按 `{field}` 假设。"
    head = (assumed + " " if assumed else "") + conclusion

    sections: list[str] = []
    for i, it in enumerate(items, 1):
        hit = tp.roles.get(it["role"])
        if not hit:
            continue
        sections.extend(
            _section(i, it["label"], hit, it.get("suggested_code") or "")
        )
    if not sections:
        sections = ["（尚未定位到协议同步 VO / 常量 / *Map.put 触点。）", ""]

    lines = [
        "## 结论",
        "",
        head.strip(),
        "",
        "## 改动清单",
        "",
        *sections,
    ]
    if tp.gaps:
        lines.extend(["", f"缺口：{'、'.join(tp.gaps)}。"])
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


_COMPOSITE_EXTRA_ROLES = (
    ("org_check_gate", "关卡校验"),
    ("single_batch_external", "分支外部透传"),
    ("external_pass", "外部透传触点"),
)


def _suggest_external_pass_code(tp: Touchpoints, hit: RoleHit) -> str:
    field = tp.field_name or "newField"
    getter = _getter_name(field)
    setter = _setter_name(field)
    code = hit.raw_code or ""
    # Prefer aligning with an existing DTO set* before Client call.
    m_set = re.search(
        r"(?m)^([ \t]*)([a-z]\w*)\.set([A-Z]\w*)\s*\([^;]+;\s*$",
        code,
    )
    if m_set:
        ind, recv = m_set.group(1), m_set.group(2)
        return (
            f"{ind}// 关卡校验 / 外部调用时透传 {field}\n"
            f"{ind}{recv}.{setter}(reqVo.{getter}());"
        )
    m_put = re.search(
        r"(?m)^([ \t]*)([a-z]\w*Map)\.put\s*\(\s*([^,]+),\s*([^)]+)\)\s*;",
        code,
    )
    if m_put:
        ind, map_name, key = m_put.group(1), m_put.group(2), m_put.group(3)
        const_name = _field_to_const_name(field)
        return (
            f"{ind}// 外部调用时透传 {field}\n"
            f"{ind}{map_name}.put({const_name}, reqVo.{getter}());"
            f"  // 键名对齐已有 Constants，示例键={key.strip()}"
        )
    return (
        f"// 在 `{hit.class_method}` 的关卡/外部调用分支，\n"
        f"// 将 reqVo.{getter}() 写入外部请求 DTO / Map 后再发起调用。"
    )


def build_composite_field_evidence_pack(tp: Touchpoints) -> dict:
    """Evidence pack = add_field items + org/single-batch external roles."""
    pack = build_add_field_evidence_pack(tp)
    pack["capability"] = "composite_field"
    pack["need_ddl"] = True
    items = list(pack.get("change_items") or [])
    field = tp.field_name or "newField"
    for role, label in _COMPOSITE_EXTRA_ROLES:
        hit = tp.roles.get(role)
        if not hit:
            continue
        items.append(
            {
                "role": role,
                "label": label,
                "file_path": hit.file_path or "",
                "class_method": hit.class_method,
                "why": hit.why,
                "evidence": hit.evidence,
                "suggested_code": _suggest_external_pass_code(tp, hit),
                "peer_style": "",
            }
        )
    pack["change_items"] = items
    pack["notes"] = list(tp.notes or [])[:6]
    pack["question_field"] = field
    return pack


def render_composite_field_plan(tp: Touchpoints, *, question: str = "") -> str:
    """结论 → 改动清单（入参+落库+关卡/外部透传）→ DDL."""
    pack = build_composite_field_evidence_pack(tp)
    field = pack["question_field"]
    table = pack.get("table_name")
    items = pack["change_items"]
    roles_present = [it["label"] for it in items]
    if roles_present:
        conclusion = (
            f"入口 `{tp.entry}` 新增入参 `{field}` 并落库"
            + (f"至 `{table}`" if table else "")
            + "，并在关卡校验/外部调用路径透传给外部系统；按 "
            + " → ".join(roles_present)
            + " 补齐。"
        )
    else:
        conclusion = (
            f"入口 `{tp.entry}` 下尚未定位到复合改动触点，请补充业务名或入口后再问。"
        )
    assumed = ""
    if any("未指名" in n or "假设" in n for n in (tp.notes or [])):
        assumed = f"字段按 `{field}` 假设。"
    head = (assumed + " " if assumed else "") + conclusion

    sections: list[str] = []
    for i, it in enumerate(items, 1):
        hit = tp.roles.get(it["role"])
        if not hit:
            continue
        sections.extend(
            _section(i, it["label"], hit, it.get("suggested_code") or "")
        )
    if not sections:
        sections = ["（尚未定位到入参/落库或外部透传触点。）", ""]

    ddl = pack.get("suggested_ddl") or ""
    lines = [
        "## 结论",
        "",
        head.strip(),
        "",
        "## 改动清单",
        "",
        *sections,
    ]
    if ddl:
        lines.extend(["", "## DDL", "", "```sql", ddl.strip(), "```"])
    elif pack.get("need_ddl"):
        col = pack.get("column_name") or _camel_to_snake(field)
        tbl = table or "<table>"
        lines.extend(
            [
                "",
                "## DDL",
                "",
                "```sql",
                f"ALTER TABLE {tbl} ADD COLUMN {col} VARCHAR(64) NULL "
                f"COMMENT '{field}';",
                "```",
            ]
        )
    if tp.roles.get("persist_entity") and not tp.need_mapper_xml:
        lines.extend(["", "MyBatis-Plus insert，一般不必改 Mapper.xml。"])
    if tp.gaps:
        lines.extend(["", f"缺口：{'、'.join(tp.gaps)}。"])
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return _append_changed_method_section(text, tp, question=question)


# —— 阶段2：改动后完整方法体 ——
# 业务问答从「改动清单」升级为「可直接粘贴的改动后方法」：
# 以 assemble/bind 角色的 raw_code 为骨架，标注新增字段声明/绑定/落库插入点。
_RESP_VAR_RE = re.compile(r"Response<[^>]*>\s+([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*[a-zA-Z_][\w.]*\.\w+\s*\(")
_ENTITY_VAR_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9_]*)\s*\.\s*set[A-Z]\w*\s*\(")


def _extract_response_field(question: str) -> str | None:
    """从「其响应中 bbb 字段 / 返回的 bbb / 响应 bbb / 返回结果里的bizNo」提取响应字段名。

    只在出现「响应/返回」语义且不在「入参X和Y」并列上下文时提取，避免把
    「增加入参 aaa 和 bbb 两个字段」里的第二个字段误判为响应字段。
    """
    q = question or ""
    # 先排除纯入参并列：含「入参…和…字段」时该语境不产生响应字段
    if re.search(r"入参[^，。；]{0,20}(?:和|、|与)[^，。；]{0,20}字段", q):
        # 但若同时有点名「响应」，仍按响应语义提取
        if not re.search(r"其?响应|返回", q):
            return None
    patterns = [
        # 其响应中bbb字段 / 响应中的bbb / 响应bbb
        r"(?:其?响应|返回报文|返回参数|出参)(?:中|里|的|里边的|中边的)?\s*([A-Za-z][A-Za-z0-9_]{1,40})",
        # 返回结果里的bizNo / 返回的bizNo / 返回bizNo / 返回数据中total
        r"返回(?:结果|值|的数据|数据)?(?:中|里|里面|里的|中 的)?\s*([A-Za-z][A-Za-z0-9_]{1,40})",
        # bbb字段落库 / bbb字段写入
        r"([A-Za-z][A-Za-z0-9_]{1,40})\s*字段\s*(?:落|写入|保存|存|赋值)",
    ]
    for pat in patterns:
        m = re.search(pat, q)
        if m:
            name = m.group(1)
            if name.lower() not in {"json", "string", "int", "long", "data", "body", "code", "msg"}:
                return name
    return None


def _find_response_var(code: str) -> str:
    """方法体内第一个 `Response<...> xxx = client.yyy(...)` 的响应变量名。"""
    m = _RESP_VAR_RE.search(code or "")
    return m.group(1) if m else ""


def _find_entity_var(code: str) -> str:
    """方法体内落库实体接收者（最靠后的 set* 调用方，通常为 entity/map）。"""
    hits = list(_ENTITY_VAR_RE.finditer(code or ""))
    if not hits:
        return ""
    # 取出现最多的接收者（最可能是落库实体）
    recv = hits[-1].group(1)
    counts: dict[str, int] = {}
    for h in hits:
        counts[h.group(1)] = counts.get(h.group(1), 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0] if counts else recv


def build_changed_method_body(
    tp: Touchpoints, *, question: str = "", max_chars: int = 6000
) -> str:
    """Return a paste-ready method body with insert markers for the new field.

    Picks the best anchor role (assemble > bind > persist_entity) whose raw_code
    looks like a method/block; shows the method signature + body with
    ``// [新增]`` markers at the recommended insertion spots. Never edits
    the original — it annotates. Empty string when no usable anchor found.

    除入参落库外，若问题点名「响应中 bbb 字段落到放款表」，且方法体里有
    ``Response<...> resp = client.xxx(...)`` 外部调用，则追加响应字段→落库列映射段。
    """
    field = tp.field_name or "newField"
    setter = _setter_name(field)
    col = _camel_to_snake(field)
    anchor: RoleHit | None = None
    for role in ("assemble", "bind", "persist_entity", "persist_entity_1"):
        hit = tp.roles.get(role)
        if hit and (hit.raw_code or "").strip():
            anchor = hit
            break
    if not anchor:
        return ""
    code = (anchor.raw_code or "").strip()
    if len(code) > max_chars:
        code = code[:max_chars]
    head: list[str] = []
    body = code
    # 若 raw_code 是「void methodName(...) { ... }」形式，拆出签名与体
    m = re.match(r"([ \t]*)((?:public|protected|private)[^\n]*?)\{\s*\n(.*)\n\s*\}", code, re.S)
    if m:
        ind = m.group(1)
        sig = m.group(2).strip()
        inner = m.group(3)
        head = [f"{ind}{sig} {{", ""]
        body = inner
    else:
        head = [f"// 目标方法/代码块：`{anchor.class_method}`（{anchor.file_path or '-'}）", ""]

    setter_line = (
        f"        {setter}(req.get{field[:1].upper()}{field[1:]}()); // [新增] 字段赋值（值来自入参 req）"
        if field and re.match(r"^[A-Za-z]", field)
        else f"        {setter}({field}); // [新增] 字段赋值"
    )
    insert_hint = (
        f"        // [新增] 落库列 `{col}`：实体/Map 需补 {setter}(...) 或 put(\"{col}\", ...)"
    )
    # 响应字段→落库列映射（用户说「响应中 bbb 字段落到放款表」时）
    resp_field = _extract_response_field(question)
    resp_var = _find_response_var(body)
    entity_var = _find_entity_var(body)
    resp_lines: list[str] = []
    if resp_field and resp_var and entity_var:
        rsetter = _setter_name(resp_field)
        resp_lines = [
            f"        // [新增] 响应字段 {resp_field} → 落库列 {_camel_to_snake(resp_field)}",
            f"        if ({resp_var}.getBody() != null) {{",
            f"            {entity_var}.{rsetter}({resp_var}.getBody().get{resp_field[:1].upper()}{resp_field[1:]}());",
            "        }",
        ]
    # 在体末尾 return 前插入；无 return 则末尾追加
    lines_out: list[str] = []
    body_lines = (body or "").splitlines()
    inserted = False
    for i, ln in enumerate(body_lines):
        if re.match(r"^\s*return\b", ln) and not inserted:
            lines_out.append(setter_line)
            lines_out.append(insert_hint)
            if resp_lines:
                lines_out.append("")
                lines_out.extend(resp_lines)
            inserted = True
        lines_out.append(ln)
    if not inserted:
        lines_out.append(setter_line)
        lines_out.append(insert_hint)
        if resp_lines:
            lines_out.append("")
            lines_out.extend(resp_lines)
    body = "\n".join(lines_out)
    return "\n".join([*head, body, "}"] if m else [*head, body]).strip()


def _append_changed_method_section(text: str, tp: Touchpoints, question: str = "") -> str:
    """Append「改动后完整方法体」section after DDL when an anchor exists."""
    body = build_changed_method_body(tp, question=question)
    if not body:
        return text
    block = [
        "",
        "## 改动后完整方法体（可直接粘贴，`// [新增]` 处按上下文微调）",
        "",
        "```java",
        body,
        "```",
    ]
    return (text.rstrip() + "\n" + "\n".join(block)).strip()
