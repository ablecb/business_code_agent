"""LLM polish for change-impact plans (evidence-grounded, capability-oriented)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.agent.change_impact.discover import Touchpoints
from app.agent.change_impact.render import (
    build_add_field_evidence_pack,
    build_composite_field_evidence_pack,
    build_protocol_sync_evidence_pack,
    render_add_field_plan,
    render_composite_field_plan,
    render_protocol_sync_plan,
)

logger = logging.getLogger(__name__)

_CAPABILITY_HINTS = {
    "add_field": "新增入参字段并落库",
    "composite_field": "新增入参并落库，且在关卡校验/外部调用时透传",
    "protocol_sync": "将字段在协议同步时作为入参透传",
    "extend_field_to_sync": "将字段在协议同步时作为入参透传",
    "modify_field": "修改已有字段（含义/类型/校验/落库）",
    "add_method": "新增方法/接口能力",
    "modify_method": "修改已有方法逻辑",
}

_SYSTEM_ADD_FIELD = """你是资深 Java/Spring 改造方案编辑。根据「证据包」输出开发可读的 Markdown 改动方案。

硬性规则：
1. 只能使用证据包中的 file_path、class_method、表名、suggested_code；禁止编造类/方法/表/字段。
2. 代码必须对齐 peer_style / suggested_code 的写法（缩进、注释、reqMap.get、set*）。
3. 版式必须整齐、专业、好扫读，严格如下（节与节空一行）：

## 结论

一句话说明改什么、落哪。

## 改动清单

### 1. 角色名（如 常量 / 请求 VO / 入参绑定 / 组装落库 / 持久化实体）

`文件路径` · `Class#method`（无方法则只写路径）

```java
// 可粘贴的新增/修改代码（只含本角色必要行）
```

### 2. …

## DDL

```sql
ALTER …
```

4. 禁止输出：证据行、CALLS/var-resolve、对照大段原文、评审提示、状态流转、外部系统、注意事项、技术锚点。
5. 不要重复罗列「做什么/文件/证据」 bullet；改动清单里路径 + 代码即可。
6. gaps 非空时，只在结论末尾加半句「缺口：…」。
7. 若 need_mapper_xml=false，可在 DDL 后单行注明「MyBatis-Plus insert，一般不必改 Mapper.xml」。
8. 只输出 Markdown 正文，不要包裹 ```markdown。"""

_SYSTEM_COMPOSITE = """你是资深 Java/Spring 改造方案编辑。根据「证据包」输出「新增入参+落库+关卡校验/外部透传」的 Markdown 改动方案。

硬性规则：
1. 只能使用证据包中的 file_path、class_method、表名、suggested_code；禁止编造类/方法/表/字段。
2. 版式严格如下（节与节空一行）：

## 结论

一句话：新增哪个入参、落哪、在哪条关卡/外部调用路径透传。

## 改动清单

### 1. 角色名（常量 / 请求 VO / 入参绑定 / 组装落库 / 实体 / 关卡校验 / 外部透传）

`文件路径` · `Class#method`

```java
// 可粘贴代码
```

### 2. …

## DDL

```sql
ALTER …
```

3. 关卡校验 / 外部透传相关步骤必须来自证据包 change_items，禁止背诵固定方法清单。
4. 禁止：协议同步 checklist、评审提示、CALLS/证据行、编造 Feign 签名。
5. gaps 非空时结论末加「缺口：…」。
6. 只输出 Markdown 正文。"""

_SYSTEM_PROTOCOL_SYNC = """你是资深 Java/Spring 改造方案编辑。根据「证据包」输出「字段透传到协议同步」的 Markdown 改动方案。

硬性规则：
1. 只能使用证据包中的 file_path、class_method、suggested_code；禁止编造类/方法/字段。
2. 代码必须对齐 peer_style / suggested_code（*Map.put(CONST, reqVo.get*)）。
3. 版式严格如下（节与节空一行）；**不要写 DDL**（本题不是落库）：

## 结论

一句话：把哪个字段在协议同步时作为入参透传。

## 改动清单

### 1. 角色名（常量 / 同步入参 VO / 协议同步组装）

`文件路径` · `Class#method`

```java
// 可粘贴代码
```

### 2. …

4. 禁止：校验字段清单、关卡清单、状态流转、评审提示、CALLS/证据行。
5. 可在结论末注明「下游 Feign 一般不必改签名」。
6. gaps 非空时结论末加「缺口：…」。
7. 只输出 Markdown 正文。"""


def _validate_ai_plan(
    text: str,
    pack: dict[str, Any],
    *,
    capability: str = "add_field",
) -> bool:
    if not text or "## 结论" not in text or "## 改动清单" not in text:
        return False
    need_ddl = pack.get(
        "need_ddl", capability in {"add_field", "composite_field"}
    )
    if need_ddl and "## DDL" not in text and "ALTER" not in text.upper():
        return False
    if not need_ddl and "## DDL" in text:
        # Soft: protocol sync plans should not invent DDL.
        return False
    # Must not invent file paths outside evidence.
    allowed = {
        (it.get("file_path") or "").replace("\\", "/").strip()
        for it in pack.get("change_items") or []
        if it.get("file_path")
    }
    for m in re.finditer(r"`([^`]+\.java)`", text):
        path = m.group(1).replace("\\", "/").strip()
        if allowed and path not in allowed:
            # Allow path without module prefix mismatch soft-fail only if suffix matches.
            if not any(path.endswith(a) or a.endswith(path) for a in allowed):
                return False
    junk = (
        "CALLS",
        "var-resolve",
        "证据:",
        "对照本方法",
        "评审提示",
        "业务状态流转",
        "queryProductGuarateeName",
    )
    if any(j in text for j in junk):
        return False
    return True


def render_change_plan_with_ai(
    question: str,
    tp: Touchpoints,
    *,
    capability: str = "add_field",
) -> tuple[str, str]:
    """Return (markdown, source) where source is ``ai`` or ``deterministic``."""
    cap = capability or getattr(tp, "capability", None) or "add_field"
    if cap in {"protocol_sync", "extend_field_to_sync"}:
        pack = build_protocol_sync_evidence_pack(tp)
        draft = render_protocol_sync_plan(tp)
        system = _SYSTEM_PROTOCOL_SYNC
    elif cap == "composite_field":
        pack = build_composite_field_evidence_pack(tp)
        draft = render_composite_field_plan(tp, question=question)
        system = _SYSTEM_COMPOSITE
    else:
        pack = build_add_field_evidence_pack(tp)
        draft = render_add_field_plan(tp, question=question)
        system = _SYSTEM_ADD_FIELD
    pack["capability"] = cap
    pack["capability_hint"] = _CAPABILITY_HINTS.get(cap, cap)
    pack["user_question"] = question or ""

    if not pack.get("change_items"):
        return draft, "deterministic"

    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        user = (
            f"改造类型：{pack['capability_hint']}（capability={cap}）\n"
            f"用户问题：{question}\n\n"
            "证据包 JSON（唯一事实来源）：\n"
            f"{json.dumps(pack, ensure_ascii=False, indent=2)}\n\n"
            "下面是程序草稿（可参考结构，但请按规则重排得更清晰专业；"
            "代码以证据包 suggested_code 为准，可按 peer_style 微调格式）：\n\n"
            f"{draft}"
        )
        out = llm.chat_sync(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2500,
        )
        text = (out or "").strip()
        text = re.sub(r"^```(?:markdown|md)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if _validate_ai_plan(text, pack, capability=cap):
            return text, "ai"
        logger.info("AI change plan failed validation; using deterministic draft")
    except Exception:
        logger.exception("AI change plan render failed; using deterministic draft")
    return draft, "deterministic"
