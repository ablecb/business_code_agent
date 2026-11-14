"""Add-field change-plan helpers extracted from explore_agent.py.

Deterministic「新增字段并落库」change plans: role-tagging of read symbols
(VO / bind / assemble / entity), evidence-gap gating, fallback plan rendering
and answer sanitizing. Depends only on explore_intent / explore_keys /
explore_shared / question_intent at import time; explore_agent re-exports
every name for compatibility.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.explore_intent import _lexical_main_entry
from app.agent.explore_keys import _REVIEW_FOOTER_RE
from app.agent.explore_shared import _read_count
from app.agent.question_intent import assumed_new_field_name

logger = logging.getLogger("app.agent.explore_add_field")

# Minimum successful get_symbol_code reads before finishing an add-field plan.
_ADD_FIELD_MIN_READS = 4


def _add_field_entry_prefix(prefer_cm: str | None, question: str) -> str | None:
    """Derive a type-name prefix from locked/lexical entry (e.g. WithdrawalsController → Withdrawals)."""
    cm = (prefer_cm or "").strip()
    if not cm or "#" not in cm:
        cm = _lexical_main_entry(question) or ""
    if not cm or "#" not in cm:
        return None
    simple = cm.split("#", 1)[0].split(".")[-1]
    for suf in ("Controller", "ServiceImpl", "Service", "Facade", "Api"):
        if simple.endswith(suf) and len(simple) > len(suf):
            return simple[: -len(suf)]
    return simple or None


def _add_field_seed_queries(prefer_cm: str | None, question: str) -> list[str]:
    """Dynamic search seeds from entry prefix — not a fixed method checklist."""
    prefix = _add_field_entry_prefix(prefer_cm, question)
    qs: list[str] = []
    if prefix:
        qs.extend(
            [
                f"{prefix}ReqVo",
                f"{prefix}ApplyReqVo",
                f"{prefix}Ext",
                f"{prefix}Request",
            ]
        )
    # Question nouns as soft seeds (strip boilerplate).
    q = re.sub(
        r"(?:我想|请|帮我|都要修改哪些|改哪些|并落库|增加一个字段|新增字段|字段)",
        " ",
        question or "",
    )
    q = re.sub(r"\s+", " ", q).strip()
    if q and len(q) >= 2:
        qs.append(q[:40])
    # de-dupe preserve order
    out: list[str] = []
    for s in qs:
        if s and s not in out:
            out.append(s)
    return out[:6]


def _add_field_hit_blob(h: dict[str, Any]) -> str:
    """Join identity + body for one retrieved symbol (finish-gate matching)."""
    if not h:
        return ""
    return " ".join(
        str(h.get(k) or "")
        for k in (
            "fqcn",
            "method_name",
            "method_sig",
            "file_path",
            "kind",
            "summary",
            "raw_code",
        )
    )


def _add_field_role_of_hit(h: dict[str, Any]) -> set[str]:
    """Infer generic roles from code patterns — no fixed Class#method names."""
    roles: set[str] = set()
    if not h:
        return roles
    code = str(h.get("raw_code") or "").strip()
    if not code:
        return roles
    blob = _add_field_hit_blob(h)
    fqcn = str(h.get("fqcn") or "")
    method = str(h.get("method_name") or "")
    simple = fqcn.split(".")[-1]
    kind = str(h.get("kind") or "").lower()

    # Request VO / DTO (name heuristic OR field-rich type without @TableName).
    if re.search(
        r"(?i)(?:ReqVo|Request|ReqDto|RequestDto|ParamVo|ApplyVo)\b",
        simple,
    ) or kind in {"vo", "dto", "request"}:
        roles.add("vo")
    elif (
        re.search(r"(?i)\bprivate\s+String\s+\w+", code)
        and "@TableName" not in code
        and not re.search(r"(?i)(?:Controller|ServiceImpl|Mapper)\b", simple)
        and not method
    ):
        roles.add("vo")

    # Bind / validate from HTTP map or request body.
    if (
        re.search(r"(?i)reqMap\.get\s*\(|request\.getParameter\s*\(", code)
        or re.search(r"(?i)\.get\s*\(\s*[\"']\w+[\"']\s*\).{0,40}set[A-Z]", code)
        or re.search(r"(?i)check\w*Param|validate\w*|bind\w*", method)
    ):
        roles.add("bind")

    # Assemble VO → persist object (set on Ext/Dmo/Entity, *2Ext*, copyProperties).
    if (
        re.search(r"(?i)(?:2Ext|2Dmo|2Entity|ToExt|ToDmo)\b", method)
        or re.search(
            r"(?i)(?:ext|dmo|entity)\w*\.set[A-Z]\w*\s*\([^;]{0,80}"
            r"(?:get[A-Z]|reqVo\.|req\.)",
            code,
        )
        or re.search(
            r"(?i)BeanUtils\.copyProperties\s*\([^)]+(?:Ext|Dmo|Entity)",
            code,
        )
    ):
        roles.add("assemble")

    # Persist entity with table mapping.
    if "@TableName" in code or re.search(
        r"(?i)@Table\s*\(|@Entity\b",
        code,
    ):
        roles.add("entity")
        roles.add("orm")
    elif re.search(r"(?i)(?:Ext|Dmo|DO|Entity|Po)\b", simple) and re.search(
        r"(?i)\bprivate\s+\w+",
        code,
    ):
        roles.add("entity")

    if re.search(r"(?i)extends\s+BaseMapper\b|InterfaceBaseMapper", blob):
        roles.add("orm")

    # Analog: other business fields already on a request VO (pattern reference).
    if "vo" in roles and len(re.findall(r"(?i)\bprivate\s+String\s+\w+", code)) >= 2:
        roles.add("analog")

    return roles


def _add_field_evidence_coverage(
    retrieved: list[dict[str, Any]],
) -> dict[str, bool]:
    """Role coverage from actually-read bodies (search hits without raw_code ignored)."""
    found: set[str] = set()
    for h in retrieved or []:
        found |= _add_field_role_of_hit(h)
    return {
        "vo": "vo" in found,
        "bind": "bind" in found,
        "assemble": "assemble" in found,
        "ext": "entity" in found,  # key name kept for callers; means persist entity
        "mp": "orm" in found,
        "analog": "analog" in found,
    }


def _add_field_pick_by_role(
    retrieved: list[dict[str, Any]], role: str
) -> dict[str, Any] | None:
    for h in retrieved or []:
        if h and role in _add_field_role_of_hit(h):
            return h
    return None


def _add_field_evidence_gaps(
    steps: list[dict[str, Any]],
    retrieved: list[dict[str, Any]],
    *,
    soft: bool = False,
) -> str | None:
    """Block finish until role coverage is enough.

    soft=True (near budget end / exhaust): only need min reads + 2 core roles
    so we can still produce a partial plan instead of dumping an internal error.
    """
    # Structured discover tool already covered roles + rendered plan.
    if any(
        s.get("tool") == "discover_add_field_plan" and s.get("ok") for s in steps
    ):
        return None
    reads = _read_count(steps)
    searched = any(s.get("tool") == "search_symbols" and s.get("ok") for s in steps)
    if not searched and reads < 1:
        return (
            "新增字段改动方案：请先调用 discover_add_field_plan，"
            "或按问题中的接口/业务词 search_symbols 定位入口与请求 VO。"
        )
    min_reads = 3 if soft else _ADD_FIELD_MIN_READS
    if reads < min_reads:
        return (
            f"新增字段改动方案：至少成功 get_symbol_code {min_reads} 次"
            f"（当前 {reads}），覆盖角色：请求 VO、入参绑定/校验、落库组装、"
            "持久化实体（@TableName/ORM）后再 finish。"
        )
    cov = _add_field_evidence_coverage(retrieved)
    if soft:
        # Accept if we have request-side + persist-side signal.
        ok_req = cov["vo"] or cov["bind"]
        ok_persist = cov["assemble"] or cov["ext"]
        if ok_req and ok_persist and reads >= min_reads:
            return None
        missing_soft: list[str] = []
        if not ok_req:
            missing_soft.append("请求 VO 或入参绑定")
        if not ok_persist:
            missing_soft.append("落库组装或持久化实体")
        return "尚缺：" + "、".join(missing_soft) if missing_soft else None

    missing: list[str] = []
    if not cov["vo"]:
        missing.append("请求 VO/DTO（*ReqVo/*Request 等，以实际为准）")
    if not cov["bind"]:
        missing.append("入参绑定/校验方法（reqMap.get→set* 等）")
    if not cov["assemble"]:
        missing.append("VO→持久化对象的组装方法")
    if not cov["ext"]:
        missing.append("持久化实体（含 @TableName 或等价）")
    # ORM is soft when entity already has @TableName (counted in ext+mp together).
    if not cov["mp"] and cov["ext"]:
        pass  # entity without orm tag is rare; don't hard-block
    elif not cov["mp"] and not cov["ext"]:
        missing.append("ORM 证据（@TableName 或 BaseMapper）")
    if missing:
        return (
            "尚未读全新增字段角色链路："
            + "、".join(missing)
            + "。请继续按入口真实调用读码后再 finish（禁止只读 1～2 个符号；"
            "禁止背诵固定方法名清单）。"
        )
    return None


def _partial_add_field_plan_from_evidence(
    question: str,
    retrieved: list[dict[str, Any]],
    *,
    gap_note: str | None = None,
) -> str:
    """User-facing plan from whatever roles were read — never dump internal tool gates."""
    field = assumed_new_field_name(question)
    cov = _add_field_evidence_coverage(retrieved)
    full = _fallback_add_field_plan_from_evidence(question, retrieved)
    if full:
        return full

    setter = f"set{field[:1].upper() + field[1:]}"
    vo = _add_field_pick_by_role(retrieved, "vo")
    bind = _add_field_pick_by_role(retrieved, "bind")
    assemble = _add_field_pick_by_role(retrieved, "assemble")
    entity = _add_field_pick_by_role(retrieved, "entity")

    def _label(h: dict[str, Any] | None, fallback: str) -> str:
        if not h:
            return fallback
        fp = str(h.get("file_path") or "").strip()
        fqcn = str(h.get("fqcn") or "")
        method = str(h.get("method_name") or "")
        if fp and method:
            return f"{fp}（{fqcn.split('.')[-1]}#{method}）"
        if fp:
            return fp
        if method:
            return f"{fqcn}#{method}" if fqcn else method
        return fqcn or fallback

    table = ""
    if entity:
        m = re.search(
            r'@TableName\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
            str(entity.get("raw_code") or ""),
        )
        if m:
            table = m.group(1)
    col = re.sub(r"([A-Z])", r"_\1", field).lower().lstrip("_")

    items: list[str] = []
    n = 1
    if vo:
        items.append(f"{n}. 请求 VO — {_label(vo, '请求 VO')}")
        items.append(f"   - 文件: `{vo.get('file_path') or _label(vo, '请求 VO')}`")
        items.append(f"   - 做什么: 新增 `private String {field};`")
        items.append("")
        n += 1
    if bind:
        items.append(f"{n}. 入参绑定 — {_label(bind, '入参绑定')}")
        items.append(f"   - 文件: `{bind.get('file_path') or _label(bind, '入参绑定')}`")
        items.append(
            f"   - 做什么: 按已读方式绑定（如 reqMap.get → {setter}）"
        )
        items.append("")
        n += 1
    if assemble:
        items.append(f"{n}. 组装落库 — {_label(assemble, '落库组装')}")
        items.append(
            f"   - 文件: `{assemble.get('file_path') or _label(assemble, '落库组装')}`"
        )
        items.append(f"   - 做什么: `{setter}(…)` 写入持久化对象")
        items.append("")
        n += 1
    if entity:
        ent = _label(entity, "持久化实体")
        extra = f"；@TableName={table}" if table else ""
        items.append(f"{n}. 实体/DDL — {ent}")
        items.append(f"   - 文件: `{entity.get('file_path') or ent}`")
        items.append(
            f"   - 做什么: 增加属性（必要时 @TableField）{extra}；"
            "若 MyBatis-Plus insert，一般不必改 Mapper.xml"
        )
        items.append("")
        n += 1
    while items and items[-1] == "":
        items.pop()
    if not items:
        items.append("（本轮尚未读到可改动的 VO/绑定/组装/实体。）")

    suggest: list[str] = []
    if vo:
        suggest.append(f"- VO：新增 `private String {field};`")
        suggest.append("")
    if bind:
        suggest.append(f"- 绑定：按入口已有字段同样方式增加 `{field}`")
        suggest.append("")
    if assemble or entity:
        suggest.append("- 组装/实体：写入对应持久化对象并加实体属性")
        suggest.append("")
    while suggest and suggest[-1] == "":
        suggest.pop()
    if not suggest:
        suggest.append("- （尚无足够证据给出具体改法；见下方说明）")

    ddl = (
        f"ALTER TABLE `{table}` ADD COLUMN `{col}` varchar(64) DEFAULT NULL COMMENT '{field}';"
        if table
        else f"-- 未读到 @TableName，暂不能确定表名；列建议 `{col}`（字段 {field}）"
    )
    lines = [
        f"假设新增字段 `{field}`（基于本轮已读证据；仅列已发现部分）。",
        "",
        "## 结论",
        "",
        f"本轮已定位部分触点；按数据流补齐 `{field}` 后落库。",
        "",
        "## 改动顺序（按数据流：常量 → VO → 入参绑定 → 组装落库 → 实体/DDL）",
        "",
        *items,
        "",
        "## 从 → 到（按文件，可复制）",
        "",
        *suggest,
        "",
        "## DDL",
        "",
        "```sql",
        ddl,
        "```",
    ]
    if gap_note:
        lines.extend(
            [
                "",
                "## 说明（闭包/证据范围；缺口）",
                "",
                str(gap_note).strip(),
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## 说明（闭包/证据范围；缺口）",
                "",
                "上列为已读证据归纳（非写死模板）。",
            ]
        )
    return "\n".join(lines)


def _add_field_answer_is_persist_gate_narrative(answer: str) -> bool:
    """True when answer only lists existing insert* gates (wrong mode)."""
    a = answer or ""
    has_insert_list = bool(
        re.search(
            r"(?i)insertLoanData|insertOrUpdateLoanInfo|loanApply\b|"
            r"写入主表|回写主表|申请落库",
            a,
        )
    )
    has_change_plan = bool(
        re.search(
            r"(?i)ReqVo|Request\b|reqMap\.get|ALTER\s+TABLE|@TableName|"
            r"改动清单|改动顺序|增加字段|新增字段|建议改动|从\s*→\s*到|"
            r"private\s+String",
            a,
        )
    )
    return has_insert_list and not has_change_plan


def _add_field_answer_has_synth_junk(answer: str) -> bool:
    """True when answer contains flow-review/synth leftovers after a change plan."""
    a = answer or ""
    if not a:
        return False
    if re.search(r"业务状态流转|当.{0,8}状态为\s*F\b", a):
        return True
    if re.search(r"外部系统调用[：:].{0,60}未在已读", a):
        return True
    if re.search(r"[（(]评审提示|可追问「|补充说明[：:]", a):
        return True
    if re.search(r"(?m)^#{1,6}\s*技术锚点\b", a):
        return True
    # Bare ## 说明 tip blocks (not discovery stats titled with 闭包/证据/缺口).
    if re.search(r"(?m)^#{1,6}\s*说明\b(?![^\n]*(?:闭包|证据|CALLS|缺口))", a):
        return True
    return False


def _add_field_answer_wrong_apply_persist(answer: str) -> bool:
    """True when plan invents Apply-entity persist without grounding (ext-table slip)."""
    a = answer or ""
    if not a:
        return False
    if re.search(
        r"WithdrawalsApply(?!ReqVo|RespVo|Service)"
        r".{0,80}(?:ADD\s+COLUMN|加列|新增列)",
        a,
    ):
        return True
    if re.search(
        r"(?i)(?:reqVo2ExtDmo|updateReult).{0,120}"
        r"WithdrawalsApply(?!ReqVo|RespVo|Service)",
        a,
    ):
        return True
    if re.search(
        r"复制到\s*WithdrawalsApply(?!ReqVo|RespVo|Service)",
        a,
    ):
        return True
    return False


def _add_field_answer_has_offtopic_credit_noise(answer: str, question: str) -> bool:
    """True when plan mixes a locked entry Controller with unrelated Controllers."""
    del question
    entry = _infer_plan_entry_controller(answer or "")
    if not entry:
        return False
    others = [
        c
        for c in re.findall(r"\b([A-Z][A-Za-z0-9_]*Controller)\b", answer or "")
        if c != entry
    ]
    return bool(others)


def _infer_plan_entry_controller(answer: str) -> str | None:
    """Best-effort locked Controller from 改动清单 / 入口 lines."""
    a = answer or ""
    m = re.search(
        r"(?:入口|主入口)[^\n`]*`([A-Z][A-Za-z0-9_]*Controller)#[A-Za-z_]\w*`",
        a,
    )
    if m:
        return m.group(1)
    m = re.search(
        r"##\s*改动清单[\s\S]{0,1200}?([A-Z][A-Za-z0-9_]*Controller)#[A-Za-z_]\w*",
        a,
    )
    if m:
        return m.group(1)
    m = re.search(r"\b([A-Z][A-Za-z0-9_]*Controller)#[A-Za-z_]\w*", a)
    return m.group(1) if m else None


def _drop_offtopic_controller_lines(text: str, *, entry_class: str | None) -> str:
    """Drop plan lines that only mention Controllers outside the locked entry family."""
    if not text or not entry_class:
        return text
    out: list[str] = []
    for line in text.splitlines():
        ctrls = re.findall(r"\b([A-Z][A-Za-z0-9_]*Controller)\b", line)
        if ctrls and all(c != entry_class for c in ctrls):
            continue
        # Entity/DTO noise tied to another domain when entry is locked.
        if entry_class and re.search(
            rf"(?i)\b(?!{re.escape(entry_class)})Credit(?:Notification|Quota)\b",
            line,
        ):
            # Keep if entry itself is Credit*
            if not entry_class.startswith("Credit"):
                continue
        out.append(line)
    return "\n".join(out)


def _add_field_answer_missing_scannable_format(answer: str) -> bool:
    """True when substance may exist but scannable plan/DDL headings are absent.

    Honest partial plans (found roles + gap note) count as scannable — do not
    require a full VO+bind checklist that invents missing steps.
    """
    a = answer or ""
    if not a:
        return True
    has_heading = bool(
        re.search(r"改动清单|改动顺序|从\s*→\s*到|建议改动|##\s*结论", a)
    )
    has_ddl = bool(
        re.search(r"(?i)##\s*DDL|ALTER\s+TABLE|不能确定表名|表名待补", a)
    )
    honest_gap = bool(
        re.search(r"未定位到|未读全|尚未读到|证据不足|缺口|仅含已发现", a)
    )
    has_bind = bool(
        re.search(
            r"(?i)reqMap\.get|RequestBody|绑定|set[A-Z]\w*\s*\(|check\w*Param",
            a,
        )
    )
    has_vo = bool(
        re.search(r"(?i)ReqVo|Request\b|Dto\b|入参|VO\b|private\s+String", a)
    )
    substance = (has_bind and has_vo) or honest_gap
    return not (has_heading and has_ddl and substance)


def _fallback_add_field_plan_from_evidence(
    question: str,
    retrieved: list[dict[str, Any]],
) -> str | None:
    """Deterministic change plan from role-tagged reads when LLM finish is thin."""
    field = assumed_new_field_name(question)
    cov = _add_field_evidence_coverage(retrieved)
    if not (cov["vo"] and cov["bind"] and cov["assemble"] and cov["ext"]):
        return None

    vo = _add_field_pick_by_role(retrieved, "vo")
    bind = _add_field_pick_by_role(retrieved, "bind")
    assemble = _add_field_pick_by_role(retrieved, "assemble")
    entity = _add_field_pick_by_role(retrieved, "entity")

    def _label(h: dict[str, Any] | None, fallback: str) -> str:
        if not h:
            return fallback
        fp = str(h.get("file_path") or "").strip()
        fqcn = str(h.get("fqcn") or "")
        method = str(h.get("method_name") or "")
        if fp and method:
            return f"{fp}（{fqcn.split('.')[-1]}#{method}）"
        if fp:
            return fp
        if method:
            return f"{fqcn}#{method}"
        return fqcn or fallback

    setter = f"set{field[:1].upper() + field[1:]}"
    table = ""
    entity_name = "持久化实体"
    if entity:
        entity_name = str(entity.get("fqcn") or "").split(".")[-1] or entity_name
        m = re.search(
            r'@TableName\s*\(\s*(?:value\s*=\s*)?["\']([^"\']+)["\']',
            str(entity.get("raw_code") or ""),
        )
        if m:
            table = m.group(1)
    col = re.sub(r"([A-Z])", r"_\1", field).lower().lstrip("_")
    ddl = (
        f"ALTER TABLE {table} ADD COLUMN {col} VARCHAR(64) NULL COMMENT '{field}';"
        if table
        else f"-- 未在已读实体中看到 @TableName；请补读落库实体后再写 ALTER（字段 {field} / 列 {col}）"
    )

    lines = [
        f"假设新增字段 `{field}`（对照本入口已读到的类似入参路径，非固定模板类名）。",
        "",
        "## 结论",
        "",
        f"按「请求 VO → 入参绑定 → 组装落库 → 实体/DDL」为 `{field}` 补齐即可落库"
        + (f"（表 `{table}`）" if table else "")
        + "。",
        "",
        "## 改动顺序（按数据流：常量 → VO → 入参绑定 → 组装落库 → 实体/DDL）",
        "",
        f"1. {_label(vo, '请求 VO/DTO')}",
        f"   - 做什么: 新增 `private String {field};`",
        f"2. {_label(bind, '入参绑定/校验')}",
        f"   - 做什么: 按已读方式绑定（如 reqMap.get → {setter}）",
        f"3. {_label(assemble, '落库组装')}",
        f"   - 做什么: 写入持久化对象 `{setter}(…)`",
        f"4. {_label(entity, entity_name)}",
        "   - 做什么: 实体属性"
        + (f"（@TableName={table}）" if table else "")
        + "；若 MyBatis-Plus insert，一般不必改 Mapper.xml",
        "",
        "## 从 → 到（按文件，可复制）",
        "",
        f"- VO：新增 `private String {field};`",
        f"- 绑定：按已读绑定方法增加 `{field}`",
        f"- 组装：在已读组装方法中 `{setter}(…)`",
        f"- 实体：`{entity_name}` 增加同名属性（必要时 `@TableField`）",
        "",
        "## DDL",
        "",
        "```sql",
        ddl,
        "```",
        "",
        "## 说明（闭包/证据范围；缺口）",
        "",
        "上列为已读证据归纳（非写死模板）。",
    ]
    return "\n".join(lines)


_ADD_FIELD_JUNK_SECTION_RE = re.compile(
    r"(?ms)^(?:#{1,6}\s*)?(?:\*\*)?(?:"
    r"业务状态流转|外部系统调用|外部协同|状态流转|"
    r"评审提示|可追问|补充说明|注意事项|技术锚点总结|技术锚点"
    r")\b[：:]?.*?(?=^#{1,6}\s|\Z)"
)
_ADD_FIELD_JUNK_INLINE_RE = re.compile(
    r"(?:业务状态流转|外部系统调用|外部协同)[：:][^\n]*(?:\n(?![#\d])[^\n]*)*"
)
_ADD_FIELD_TRAILING_NOTES_RE = re.compile(
    r"(?ms)^(?:#{1,6}\s*)?(?:\*\*)?(?:注意事项|技术锚点总结|技术锚点)\b[：:]?.*\Z"
)


_ADD_FIELD_CREDIT_NOISE_LINE_RE = re.compile(
    r"(?im)^[ \t]*[-*•]?\s*(?:\d+\s*[\.、\)]\s*)?.*?"
    r"(?:CreditController|CreditQuotaController|CreditNotification)\b.*$"
)

# Lines that wrongly treat *Mapper.xml as required for MyBatis-Plus insert 落库.
# Do NOT match「一般不必改 / 不用改 Mapper.xml」(those are correct guidance).
_ADD_FIELD_XML_REQUIRED_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*•]\s*|\d+\s*[\.、\)]\s*)?.*?"
    r"(?<![不未无])(?:必(?:须|改|需)|需要|要改|修改|更新|同步|新增到|加到).{0,40}"
    r"\w*Mapper\.xml"
    r".*$"
)
_ADD_FIELD_APPLY_TABLE_COL_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*•]\s*|\d+\s*[\.、\)]\s*)?.*?"
    r"WithdrawalsApply(?!ReqVo|RespVo|Service)"
    r".{0,80}(?:ADD\s+COLUMN|加列|新增列|新增字段|加字段|ALTER\s+TABLE)"
    r".*$"
)
_ADD_FIELD_APPLY_COPY_LINE_RE = re.compile(
    r"(?im)^[ \t]*(?:[-*•]\s*|\d+\s*[\.、\)]\s*)?.*?"
    r"(?:把|将).{0,40}WithdrawalsApplyReqVo.{0,40}(?:复制|拷贝|同步).{0,40}"
    r"WithdrawalsApply(?!ReqVo|RespVo|Service)"
    r".*$"
)


def _normalize_add_field_plan_spacing(text: str) -> str:
    """Ensure scannable markdown: blank lines around ## sections; one item per line."""
    if not text:
        return text
    # Split smashed "## 改动清单 1. …" onto heading + body.
    text = re.sub(
        r"(##\s*(?:改动清单|建议改动|DDL)[^\n]*?)\s+(?=(?:\d+\s*[\.、\)]|[-*•])\s+\S)",
        r"\1\n\n",
        text,
    )
    # "## DDL ```sql …" → heading + fence on following lines.
    text = re.sub(
        r"(##\s*DDL)\s*```(\w*)\s*",
        r"\1\n\n```\2\n",
        text,
    )
    text = re.sub(
        r"(##\s*DDL)\s+(?=ALTER\s+TABLE)",
        r"\1\n\n```sql\n",
        text,
        flags=re.I,
    )
    # Soft-break smashed numbered items: "2. Foo 3. Bar" → separate lines.
    text = re.sub(
        r"(?<!\n)(?<!#)\s+(?=(?:\d+)\s*[\.、\)]\s+\S)",
        "\n",
        text,
    )
    # Soft-break smashed bullets after a sentence end.
    text = re.sub(r"(?<=[。；;])\s*(?=[-*•]\s+\S)", "\n", text)
    # Ensure ## section heads start on their own line — never split ### into # + ##.
    text = re.sub(r"(?<!\n)(?<!#)(##\s+)(?!#)", r"\n\n\1", text)
    lines = text.split("\n")
    out: list[str] = []
    _section_re = re.compile(
        r"^##\s+(?:改动清单|改动顺序|建议改动|从\s*→\s*到|从→到|结论|DDL|说明)"
    )

    def _is_section_heading(line: str) -> bool:
        return bool(_section_re.match(line.strip()))

    in_fence = False
    for line in lines:
        # Preserve indentation inside fenced code (method-style snippets).
        if line.strip().startswith("```"):
            in_fence = not in_fence
            out.append(line.rstrip())
            continue
        if in_fence:
            out.append(line.rstrip("\n"))
            continue
        stripped = line.strip()
        if not stripped:
            if out and out[-1] != "":
                out.append("")
            continue
        prev = out[-1] if out else None
        if _is_section_heading(stripped):
            if out and prev != "":
                out.append("")
            # Keep ## titles; do not collapse ### file subsections into ##.
            out.append(stripped)
            out.append("")
            continue
        out.append(stripped)
    cleaned: list[str] = []
    for line in out:
        if line == "":
            if cleaned and cleaned[-1] == "":
                continue
            cleaned.append("")
        else:
            cleaned.append(line)
    text = "\n".join(cleaned).strip()
    for head in ("改动清单", "建议改动", "DDL"):
        text = re.sub(
            rf"(^|\n)(##\s*{re.escape(head)}[^\n]*)\n(?!\n)",
            rf"\1\2\n\n",
            text,
        )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_add_field_change_plan(
    answer: str, *, question: str = "", entry_cm: str = ""
) -> str:
    """Strip flow-synth leftovers; keep scannable change-plan body only."""
    text = (answer or "").strip()
    if not text:
        return text
    # Never leak internal explore gates to the user.
    text = re.sub(
        r"(?m)^.*(?:探索步数用尽|请继续\s*get_symbol_code|禁止仅凭\s*1～2\s*个符号|"
        r"尚未读全新增字段).*$",
        "",
        text,
    )
    text = _ADD_FIELD_JUNK_SECTION_RE.sub("", text)
    text = _ADD_FIELD_JUNK_INLINE_RE.sub("", text)
    text = _ADD_FIELD_TRAILING_NOTES_RE.sub("", text)
    # Soft tip ## 说明 (not discovery stats titled with 闭包/证据/缺口).
    text = re.sub(
        r"(?ms)^(?:#{1,6}\s*)说明\b(?![^\n]*(?:闭包|证据|CALLS|缺口)).*?(?=^#{1,6}\s|\Z)",
        "",
        text,
    )
    text = _REVIEW_FOOTER_RE.sub("", text)
    text = re.sub(r"[（(]?评审提示[：:][^）)\n]*[）)]?", "", text)
    text = re.sub(r"可追问「[^」]*」[。.]?", "", text)
    # Drop "must edit ExtMapper.xml" bullets; MP insert does not need XML.
    text = _ADD_FIELD_XML_REQUIRED_LINE_RE.sub("", text)
    # Drop invented WithdrawalsApply ADD COLUMN / copy-to-Apply lines (ext persist only).
    text = _ADD_FIELD_APPLY_TABLE_COL_LINE_RE.sub("", text)
    text = _ADD_FIELD_APPLY_COPY_LINE_RE.sub("", text)
    # Soft-fix smashed inline items after "— 做什么 - `file` - `Class#method`"
    text = re.sub(
        r"(?<=[;；])\s*-\s*(?=`[^`]+`\s*—)",
        "\n",
        text,
    )
    text = re.sub(
        r"(?<=[。；;])\s*-\s*(?=在\s*`)",
        "\n- ",
        text,
    )
    # Split "1. A - `path` — … - `Class#method` — …" into numbered lines when smashed.
    text = re.sub(
        r"\s+-\s+(`[^`]+(?:\.java)?`\s*—)",
        r"\n- \1",
        text,
    )
    entry_class = ""
    if entry_cm and "#" in entry_cm:
        entry_class = entry_cm.split("#", 1)[0].split(".")[-1].strip()
    if not entry_class:
        entry_class = _infer_plan_entry_controller(text) or ""
    if entry_class:
        text = _drop_offtopic_controller_lines(text, entry_class=entry_class)
        if not entry_class.startswith("Credit"):
            text = _ADD_FIELD_CREDIT_NOISE_LINE_RE.sub("", text)
            text = re.sub(
                r"(?i)[；;。]?\s*[-–—]?\s*`?[^`\n]*Credit(?:Controller|QuotaController|Notification)[^。\n]*[。]?",
                "",
                text,
            )
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = _normalize_add_field_plan_spacing(text)
    # Drop legacy「建议改动 / 从→到」sections (content already in 改动清单).
    text = re.sub(
        r"(?ms)^#{1,6}\s*建议改动(?:（从\s*→\s*到）|（从→到）)?\b.*?(?=^#{1,6}\s|\Z)",
        "",
        text,
    )
    text = re.sub(
        r"(?ms)^#{1,6}\s*从\s*→\s*到(?:（[^）\n]*）)?\b.*?(?=^#{1,6}\s|\Z)",
        "",
        text,
    )
    # Legacy long title → compact.
    text = re.sub(
        r"(?m)^(#{1,6}\s*)改动顺序(?:（[^）\n]*）)?\s*$",
        r"\1改动清单",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _sanitize_protocol_sync_plan(answer: str, *, question: str = "") -> str:
    """Keep protocol-sync pass-through plans scannable; strip validation/gate junk."""
    text = _sanitize_add_field_change_plan(answer, question=question)
    if not text:
        return text
    # Protocol sync is not a persist ask — drop invented DDL sections.
    text = re.sub(
        r"(?ms)^#{1,6}\s*DDL\b.*?(?=^#{1,6}\s|\Z)",
        "",
        text,
    )
    # Off-topic validation / gate dig-in leftovers.
    text = re.sub(
        r"(?im)^.*(?:queryProductGuarateeName|productCode\s*[/／]\s*merchantNo).*$",
        "",
        text,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
