"""Generic question-intent detectors (no business-domain keyword routers).

Only cross-project capabilities live here: add-field change plans and
show-source asks. Domain nouns must be resolved via
search/CALLS/session evidence — never by hardcoded product enums.
"""
from __future__ import annotations

import re

# 「增加/新增一个字段并落库，都要修改哪些」— developer change analysis.
_ADD_FIELD_ACTION_RE = re.compile(r"(?:增加|新增|加一个|加个|添加|加一)")
_ADD_FIELD_CHANGE_ASK_RE = re.compile(
    r"(?:改哪些|修改哪些|都要修改|要改什么|哪些类|哪些文件|修改方案|"
    r"改动点|改动清单|改动哪些|DDL|脚本)"
)
# 「保存到业务表 / 存到主表」 equally means persist — not only「保存到库」.
_ADD_FIELD_PERSIST_RE = re.compile(
    r"(?:落库|入库|持久化|写库|保存到库|"
    r"保存到.{0,16}表|存到.{0,16}表|写到.{0,16}表|落到.{0,16}表|落表)"
)
# 「送给/传给外部系统」= outbound pass-through (change plan, not flow narration).
_ADD_FIELD_OUTBOUND_RE = re.compile(
    r"(?:送给|传给|同步给|透传给|带给).{0,24}(?:外部|外系统|第三方|下游)|"
    r"(?:送给|传给|同步给|透传).{0,12}系统|"
    r"(?:外送|透传|同步到外部)"
)
_FIELD_NAME_RE = re.compile(
    r"(?:"
    r"(?:字段|field)\s*[「『\"'`]?([A-Za-z][A-Za-z0-9_]{1,63})"
    r"|"
    r"(?:增加|新增|加一个|加个|添加)\s*[「『\"'`]?([A-Za-z][A-Za-z0-9_]{1,63})\s*字段"
    r"|"
    # 「加 businessLabel（业务标签）」— name then Chinese gloss, no 字段 suffix
    r"(?:增加|新增|加一个|加个|添加|加)\s*[「『\"'`]?([A-Za-z][A-Za-z0-9_]{1,63})\s*[（(]"
    r"|"
    # 「增加一个入参 businessLabel(业务标签)」— 入参/字段 + name + optional gloss
    r"(?:入参|字段)\s*[「『\"'`]?([A-Za-z][A-Za-z0-9_]{1,63})\s*[（(]"
    r"|"
    r"(?:增加|新增|加一个|加个|添加).{0,20}(?:入参|字段)\s*[「『\"'`]?([A-Za-z][A-Za-z0-9_]{1,63})"
    r"|"
    r"[「『\"'`]([A-Za-z][A-Za-z0-9_]{1,63})[」』\"'`]\s*字段"
    r"|"
    r"\b([A-Za-z][A-Za-z0-9_]{1,63})\s*字段"
    r"|"
    r"（字段名[：:]\s*([A-Za-z][A-Za-z0-9_]{1,63})\s*）"
    r")"
)
_DEFAULT_NEW_FIELD = "newField"

# Generic「看源码 / 对应的代码」— no gate/theme vocabulary.
_SHOW_CODE_PHRASES = (
    "源码是什么",
    "源代码是什么",
    "实现代码是什么",
    "把代码贴出来",
    "把源码贴出来",
    "贴出代码",
    "贴出源码",
    "贴一下代码",
    "贴一下源码",
    "代码贴出来",
    "源码贴出来",
    "完整源码",
    "方法实现代码",
    "看下源码",
    "看看源码",
    "看下代码",
    "看看代码",
    "对应的代码",
    "对应代码",
    "发下源码",
    "发一下源码",
    "部分代码",
    "代码什么样",
    "代码是什么",
)
_SHOW_CODE_RE = re.compile(
    r"(?i)(?:源码|源代码|实现代码|部分代码|代码)\s*(?:是什么|什么样|怎么样|如何|在哪|给我)|"
    r"(?:把|将)?(?:代码|源码).{0,8}(?:贴|发|给|看)|"
    r"(?:贴|发|看).{0,12}(?:代码|源码)|"
    r"我想看.{0,8}(?:代码|源码)|"
    r"(?:的)?代码\s*$"
)
# Explicit Class#method or camelCase method token + code ask.
_METHOD_TOKEN_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]{2,80}(?:#[A-Za-z][A-Za-z0-9_]{1,80})?(?![A-Za-z0-9_])"
)


def _core_text(question: str) -> str:
    text = (question or "").strip()
    if not text:
        return ""
    text = re.sub(
        r"[（(](?:追问|新问题|需求复述)[：:][^）)]*[）)]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def _extract_field_name_from_text(text: str) -> str | None:
    if not text:
        return None
    m = _FIELD_NAME_RE.search(text)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g
    return None


def wants_add_field_change_plan(question: str) -> bool:
    """True when user asks what to change to ADD a request field/param (+ persist).

    Generic across interfaces — not tied to any fixed Controller/method names.
    """
    text = _core_text(question)
    if not text:
        return False
    if "字段" not in text and "入参" not in text:
        return False
    if not _ADD_FIELD_ACTION_RE.search(text) and not re.search(
        r"(?:增加|新增).{0,8}(?:入参|字段)", text
    ):
        return False
    # Relaxed: any "字段" + "新增/增加/加" triggers change plan
    # without requiring persist/change/outbound signals.
    return True


def wants_show_gate_code(question: str) -> bool:
    """True when user asks to see actual method source (generic show-code).

    Name kept for call-site compatibility; no business-gate vocabulary.
    """
    text = _core_text(question)
    if not text:
        return False
    if wants_add_field_change_plan(text):
        return False
    if any(p in text for p in _SHOW_CODE_PHRASES):
        return True
    if _SHOW_CODE_RE.search(text):
        return True
    # Method/token + 代码/源码 (any project), without domain theme tables.
    if re.search(r"源码|源代码|(?:部分)?代码", text) and _METHOD_TOKEN_RE.search(text):
        return True
    return False


def assumed_new_field_name(question: str) -> str:
    """Extract explicit field name, else neutral ``newField``."""
    text = _core_text(question)
    hit = _extract_field_name_from_text(text)
    if hit:
        return hit
    raw = question or ""
    m_inj = re.search(
        r"[（(]字段名[：:]\s*([A-Za-z][A-Za-z0-9_]{1,63})\s*[）)]",
        raw,
    )
    if m_inj:
        return m_inj.group(1)
    for block in re.findall(
        r"[（(](?:追问|需求复述)[：:]([^）)]*)[）)]",
        raw,
    ):
        hit = _extract_field_name_from_text(block)
        if hit:
            return hit
    return _DEFAULT_NEW_FIELD


def add_field_explore_guidance(question: str) -> tuple[str, str]:
    """Return (style, plan) for explore — role-based, no hardcoded Class#method list."""
    field = assumed_new_field_name(question)
    setter = f"set{field[:1].upper() + field[1:]}"
    const_name = re.sub(r"([A-Z])", r"_\1", field).upper().lstrip("_")
    style = (
        f"回答模式=新增入参字段改动清单（开发改造方案，非现有落库流程解说）：\n"
        f"优先调用工具 discover_add_field_plan（结构化发现常量/VO/绑定/组装/实体），"
        f"拿到 plan_markdown 后 finish.answer **原样采用**该 Markdown，"
        f"禁止改写版式、禁止增删步骤、禁止改成旧的「改动清单/建议改动」标题。\n"
        f"用户未指定字段名时按假设字段 `{field}` 作答。\n"
        f"入口由问题中的业务名词 / 会话锁定入口定位（任意接口均可，禁止写死某一套方法名）。\n"
        f"通用链路角色：常量（若绑定用 *Constant*.FOO）→ 请求 VO → 入参绑定 → "
        f"组装落库 → 实体/@TableName → DDL；MyBatis-Plus insert 一般不必改 Mapper.xml。\n"
        f"常量类名须来自证据（如 FieldNameConstants），禁止臆造 CommonConstant。"
        f"若有常量角色，绑定写 `reqMap.get(FieldNameConstants.{const_name})` → `{setter}`，"
        f"不要用裸字符串 `\"{field}\"`。\n"
        f"若工具未返回方案，finish 才允许手写，且必须用："
        f"## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明；"
        f"节与节空一行；每步含文件、做什么、证据；"
        f"「从 → 到」须含可粘贴 ```java 代码块（字段定义 / get→set / 组装 set），"
        f"不要只写一句话摘要。\n"
        f"禁止：业务状态流转、外部系统调用、评审提示、固定方法清单、必改 Mapper.xml。"
    )
    plan = (
        f"请开始（结构化工具优先，禁止背诵固定方法清单）：\n"
        f"1) 立刻调用 discover_add_field_plan"
        f"（args.question=用户原问；若有入口提示可传 entry_cm）\n"
        f"2) 工具返回后：finish，answer=工具结果中的 plan_markdown（原样粘贴）\n"
        f"3) 仅当工具失败/无方案时，再 search_symbols → get_symbol_code 补读角色后手写同版式\n"
        f"禁止输出旧标题「## 改动清单」「## 建议改动」。"
    )
    return style, plan


# --- Compatibility stubs: business-domain routers removed (always False / empty) ---


def wants_composite_change(question: str) -> bool:
    """Removed: no domain composite (gate/external-pass) router."""
    del question
    return False


def wants_extend_field_to_protocol_sync(question: str) -> bool:
    """Removed: no 协议同步 / syncSign keyword router."""
    del question
    return False


def wants_drill_gate_logic(question: str) -> bool:
    """Removed: no business-gate theme dig-in router.

    Multi-turn dig-in uses session last_entry/last_symbols + normal explore.
    """
    del question
    return False


def gate_themes_in_question(question: str) -> list[str]:
    """Removed: no business theme enum. Opaque topics come from session/evidence."""
    del question
    return []


def protocol_sync_explore_guidance(question: str) -> tuple[str, str]:
    """Removed domain guidance — fall back to generic add-field style if called."""
    return add_field_explore_guidance(question)


def composite_change_explore_guidance(question: str) -> tuple[str, str]:
    """Removed domain guidance — fall back to generic add-field style if called."""
    return add_field_explore_guidance(question)
