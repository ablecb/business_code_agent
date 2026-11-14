"""Generate multiple implementable change options for the options intent."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.utils import truncate
from app.llm.factory import get_llm_provider

logger = logging.getLogger("app.agent.options")


async def build_change_options(
    *,
    question: str,
    explore_answer: str = "",
    scope: list[dict[str, Any]] | None = None,
    explore_steps: list[dict[str, Any]] | None = None,
    count: int = 3,
) -> list[dict[str, Any]]:
    """Return 2–3 structured options the user can pick."""
    n = max(2, min(int(count or 3), 3))
    llm = get_llm_provider()
    scope = scope or []
    steps_brief = []
    for s in (explore_steps or [])[:12]:
        if not isinstance(s, dict):
            continue
        steps_brief.append(
            {
                "tool": s.get("tool"),
                "ok": s.get("ok"),
                "fqcn": s.get("fqcn"),
                "method_name": s.get("method_name"),
            }
        )
    prompt = f"""你是资深 Java 改造顾问。根据需求与已探索材料，给出 {n} 种可落地的实现方案供用户选择。
不要编造未在材料中出现的表名/接口。材料不足时诚实写清假设。

用户需求:
{question}

探索回答摘要:
{truncate(explore_answer or '', 3500)}

已知修改范围:
{json.dumps(scope[:20], ensure_ascii=False)}

探索步骤摘要:
{json.dumps(steps_brief, ensure_ascii=False)}

输出严格 JSON 数组，长度 {n}，每项字段：
{{
  "id": "A",
  "title": "短标题",
  "summary": "做什么、怎么做（2～4句）",
  "tradeoffs": "代价/风险/适用场景",
  "scope_preview": [{{"file":"相对路径","action":"add_field|persist|downstream|new_endpoint|validate|other","reason":"..."}}]
}}
id 依次用 A、B、C。
"""
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": "只输出 JSON 数组"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2500,
        )
        options = _parse_options_json(content)
        if options:
            return _normalize_options(options, n=n, fallback_scope=scope)
    except Exception:
        logger.exception("build_change_options LLM failed")
    return _fallback_options(question, scope, n=n)


def render_options_reply(options: list[dict[str, Any]], question: str = "") -> str:
    lines = [
        "## 可选实现方案",
        "",
        "请选择一个方案继续（回复「选方案A」或点击卡片按钮）。",
        "",
    ]
    if question:
        lines.append(f"针对：{truncate(question, 120)}")
        lines.append("")
    for o in options:
        oid = o.get("id") or "?"
        title = o.get("title") or "方案"
        lines.append(f"### 方案 {oid}：{title}")
        lines.append(str(o.get("summary") or "").strip() or "（无摘要）")
        trade = str(o.get("tradeoffs") or "").strip()
        if trade:
            lines.append(f"- 利弊：{trade}")
        preview = o.get("scope_preview") or []
        if isinstance(preview, list) and preview:
            lines.append("- 涉及：")
            for item in preview[:8]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"  - `{item.get('file')}` — {item.get('action')}: {item.get('reason')}"
                )
        lines.append("")
    return "\n".join(lines).strip()


def _parse_options_json(content: str) -> list[dict[str, Any]]:
    text = content or ""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    data = json.loads(m.group(0) if m else text)
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _normalize_options(
    options: list[dict[str, Any]],
    *,
    n: int,
    fallback_scope: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    letters = "ABCDEFGH"
    out: list[dict[str, Any]] = []
    for i, raw in enumerate(options[:n]):
        oid = str(raw.get("id") or letters[i]).strip().upper()[:8] or letters[i]
        preview = raw.get("scope_preview")
        if not isinstance(preview, list) or not preview:
            preview = [
                {
                    "file": s.get("file"),
                    "action": s.get("action") or "other",
                    "reason": s.get("reason") or "",
                }
                for s in fallback_scope[:6]
                if isinstance(s, dict) and s.get("file")
            ]
        out.append(
            {
                "id": oid if oid else letters[i],
                "title": str(raw.get("title") or f"方案{letters[i]}")[:80],
                "summary": str(raw.get("summary") or "").strip()[:1200],
                "tradeoffs": str(raw.get("tradeoffs") or "").strip()[:800],
                "scope_preview": preview,
            }
        )
    while len(out) < min(n, 2):
        out.extend(_fallback_options("", fallback_scope, n=min(n, 2) - len(out)))
    # Ensure unique ids
    seen: set[str] = set()
    for i, o in enumerate(out):
        oid = str(o["id"])
        if oid in seen:
            o["id"] = letters[i]
        seen.add(str(o["id"]))
    return out[:n]


def _fallback_options(
    question: str, scope: list[dict[str, Any]], *, n: int
) -> list[dict[str, Any]]:
    base_scope = [
        {
            "file": s.get("file"),
            "action": s.get("action") or "other",
            "reason": s.get("reason") or "",
        }
        for s in (scope or [])[:6]
        if isinstance(s, dict) and s.get("file")
    ]
    templates = [
        {
            "id": "A",
            "title": "最小改动（沿现有链路扩展）",
            "summary": (
                f"在现有入口与服务链路上做最小必要修改以满足需求"
                f"{('：' + truncate(question, 80)) if question else ''}。"
            ),
            "tradeoffs": "改动面小、回归风险低；扩展性一般。",
            "scope_preview": base_scope,
        },
        {
            "id": "B",
            "title": "分层完整改造（VO/Service/持久化一并对齐）",
            "summary": "从入参、业务服务到落库/下游透传完整对齐，保证字段链路闭环。",
            "tradeoffs": "改动文件更多，一致性更好，联调成本更高。",
            "scope_preview": base_scope,
        },
        {
            "id": "C",
            "title": "兼容扩展（开关/可选字段）",
            "summary": "以可选字段或配置开关方式接入，降低对存量调用方的破坏。",
            "tradeoffs": "兼容性好；分支逻辑与测试矩阵更复杂。",
            "scope_preview": base_scope,
        },
    ]
    return templates[: max(2, min(n, 3))]
