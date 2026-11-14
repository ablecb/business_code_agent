"""Answer assembly & sanitizing helpers extracted from explore_agent.py.

Deterministic post-processing for generated answers: strip monologue / symbol-id
crumbs / fake table names / contradictory disclaimers, keep scannable structure,
append review footnotes. explore_agent re-exports every name for compatibility.
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("app.agent.explore_answers")

from app.agent.explore_keys import (
    _FAKE_IMPL_DISCLAIMER_RE,
    _HOLLOW_TABLE_PHRASE_RE,
    _KNOWN_FAKE_TABLES,
    _REVIEW_FOOTER_RE,
    _TABLE_NAME_RE,
)
from app.agent.explore_intent import (
    _gate_themes_in_question,
    _wants_callback_section,
)
from app.agent.explore_add_field import _sanitize_add_field_change_plan

_HEX_ID = re.compile(r"\b[a-f0-9]{32,64}\b", re.I)

# Debug / index footers that belong in logs, not the desktop reply.
_EVIDENCE_FOOTER_RE = re.compile(
    r"(?m)^\s*依据：[^\n]*(?:节点数|聚焦方法数|主链来源)[^\n]*\s*$"
)
_FOLLOWS_HEADER_RE = re.compile(
    r"(?m)^主链（按执行序[^）\n]*）\s*：?\s*$"
)


def strip_answer_chrome(answer: str) -> str:
    """Drop retrieval/debug scaffolding from a user-visible reply."""
    text = answer or ""
    if not text:
        return text
    text = _EVIDENCE_FOOTER_RE.sub("", text)
    text = _FOLLOWS_HEADER_RE.sub("主链：", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


_SYMBOL_ID_ONLY_LINE = re.compile(
    r"^\s*[-*•]?\s*\**\s*symbol[_\s-]?id\s*\**\s*[:：].*$",
    re.I | re.M,
)


_SYMBOL_ID_INLINE = re.compile(
    r"[（(]?\s*symbol[_\s-]?id\s*[:：]?\s*`?[a-f0-9]{32,64}`?\s*[）)]?",
    re.I,
)
# Framework-level remote/messaging clues only — no project/domain names.


_MONOLOGUE_LINE = re.compile(
    r"^\s*(?:[-*•]\s*|#{1,6}\s*)?(?:"
    r"我需要|我得先|我先|让我|首先我|接下来我|然后我|"
    r"根据(?:已读)?证据|从证据看|证据表明|探索不足|证据不足|"
    r"工具(?:结果)?显示|我注意到|看起来还|似乎还|可能需要继续|"
    r"综上所述我|思考过程|心路|分析一下我|我来梳理"
    r").*$",
    re.M,
)


_MONOLOGUE_INLINE = re.compile(
    r"(?:根据(?:已读)?证据[，,]?|探索不足[，,]?|我需要先(?:理解|确认)[，,]?)"
)


_CLASS_METHOD_ANCHOR = re.compile(
    r"\b[A-Z][\w.]*?(?:Controller|ServiceImpl|Listener|Listenner|Client)#[\w]+\b"
)




def _strip_monologue(text: str) -> str:
    cleaned = _MONOLOGUE_LINE.sub("", text)
    cleaned = _MONOLOGUE_INLINE.sub("", cleaned)
    return cleaned




def _full_flow_answer_too_shallow(answer: str, question: str) -> str | None:
    """Reject template-thin full-flow answers that lack concrete business facts."""
    text = (answer or "").strip()
    if len(text) < 40:
        return "回答过短：请按已读代码写出具体阶段与规则。"

    # Generic "做校验" without any concrete check nouns
    has_vague_check = bool(
        re.search(r"(验证|校验).{0,24}(合法性|完整性|有效性)", text)
    ) or bool(re.search(r"(?m)^\s*\d+[\.、)].{0,8}校验[：:]", text))
    concrete = any(
        k in text
        for k in (
            "状态",
            "枚举",
            "必填",
            "回调",
            "Listener",
            "MQ",
            "队列",
            "落库",
            "开关",
            "校验",
            "Controller",
            "ServiceImpl",
            "insert",
            "update",
        )
    )
    if has_vague_check and not concrete:
        return (
            "回答过空：勿套空模板。请根据已读代码写清具体校验项、状态如何变化；"
            "不足则再 get_symbol_code 读校验方法。"
        )

    # User asked to string through callback but answer waved it away
    if _wants_callback_section(question) and re.search(
        r"结果回调|通知", text
    ):
        if re.search(r"(代码中)?未见.{0,12}(回调|通知|独立回调)", text) or re.search(
            r"未见独立回调", text
        ):
            return (
                "用户要串到下游回调/通知：请 search_symbols q=回调/Listener/Notice，"
                "get_symbol_code 读回调入口后再写「结果回调/通知」节，不要轻易写未见。"
            )

    # Flow / notice answers that stop at "构造对象并继续处理"
    flowish = any(k in (question or "") for k in ("流程", "链路", "通知", "怎么走", "什么样"))
    if flowish:
        if re.search(
            r"(构造.{0,12}对象.{0,12}继续|进行后续|继续处理|后续业务处理)",
            text,
        ):
            return (
                "结尾过粗：不要停在「构造对象/继续处理」。请 get_symbol_code 下钻 ServiceImpl，"
                "写清落库时机/表或实体/成功失败标准，以及外部系统如何调用（Feign/Client/MQ）。"
            )
        has_persist = bool(
            re.search(
                r"落库|写入|insert|update|save|持久化|保存|写(入)?表|"
                r"代码中未见.{0,8}(落库|写库|持久化)",
                text,
                re.I,
            )
        )
        has_external = bool(
            re.search(
                r"Feign|Client|MQ|Rabbit|Kafka|远程|外部系统|"
                r"调用.{0,12}(服务|系统)|同步通知|"
                r"代码中未见.{0,12}(外部|远程|Feign|Client|MQ)",
                text,
                re.I,
            )
        )
        if not has_persist or not has_external:
            return (
                "流程回答须同时写清：何时落库与落库标准，以及外部系统如何调用；"
                "请继续读 ServiceImpl / Client / MQ 发送代码后再 finish。"
            )
    return None




def _is_messy_full_flow_answer(answer: str) -> bool:
    """True if finish text looks like monologue / class dump / unnumbered ramble."""
    text = (answer or "").strip()
    if len(text) < 24:
        return True
    if _MONOLOGUE_LINE.search(text) or _MONOLOGUE_INLINE.search(text):
        return True
    if any(
        k in text
        for k in (
            "我需要理解",
            "我需要先",
            "根据证据",
            "探索不足",
            "让我先",
            "思考过程",
        )
    ):
        return True
    # Numbered Class#method steps are desired for full-flow — not "messy".
    has_numbered = bool(re.search(r"(?m)^\s*1\s*[\.、\)]\s+\S", text))
    if not has_numbered:
        return True
    # Bare class dump without numbering (many anchors, no step list structure)
    anchors = _CLASS_METHOD_ANCHOR.findall(text)
    numbered_anchors = len(
        re.findall(
            r"(?m)^\s*\d+\s*[\.、\)]\s+[A-Z][\w.]*?(?:Controller|ServiceImpl|Listener|Client)#[\w]+",
            text,
        )
    )
    if len(anchors) >= 8 and numbered_anchors == 0:
        return True
    return False




def _sanitize_answer(
    answer: str,
    *,
    full_flow: bool = False,
    evidence_text: str = "",
) -> str:
    text = _SYMBOL_ID_ONLY_LINE.sub("", answer)
    text = _SYMBOL_ID_INLINE.sub("", text)
    text = _HEX_ID.sub("", text)
    text = _strip_monologue(text)
    # Never strip Class#method from full-flow answers — that produced empty "****" steps.
    # Only collapse accidental blank bold markers left by older sanitization.
    text = re.sub(r"\*{4,}", "", text)
    text = _strip_contradictory_impl_disclaimers(text)
    text = _strip_unverified_table_names(text, evidence_text=evidence_text)
    text = _REVIEW_FOOTER_RE.sub("", text)
    text = strip_answer_chrome(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    if full_flow and not re.search(r"(?m)^\s*1\s*[\.、\)]", text):
        # keep as-is; caller may reject via style gate
        pass
    return text.strip()




def _strip_contradictory_impl_disclaimers(answer: str) -> str:
    """Drop「未获取到某某实现代码」when the answer already cites that method."""
    text = answer or ""
    if not text:
        return text
    # Collect Class#method / bare method names already present as positive citations.
    cited = {
        m.group(1).lower()
        for m in re.finditer(
            r"`?([A-Za-z][\w]*(?:ServiceImpl|Controller|Client)?#[A-Za-z_]\w*)`?",
            text,
        )
    }
    cited_methods = {
        m.group(1).lower()
        for m in re.finditer(r"(?i)(?:^|[\s`（(])([A-Za-z_]\w*Check\w*)", text)
    }

    def _keep_line(line: str) -> bool:
        if not _FAKE_IMPL_DISCLAIMER_RE.search(line):
            return True
        # Disclaimer names a method already described → drop.
        named = re.findall(r"`([^`]+)`|([A-Za-z_]\w*#?[A-Za-z_]\w*)", line)
        for a, b in named:
            token = (a or b or "").lower()
            if not token:
                continue
            if token in cited or token in cited_methods:
                return False
            if "#" in token and token.split("#", 1)[-1] in cited_methods:
                return False
            meth = token.split("#", 1)[-1]
            if meth and any(meth in c or c.endswith(meth) for c in cited | cited_methods):
                return False
        # Generic「未获取到具体实现」with no method but body already has gate methods.
        if cited or cited_methods:
            return False
        return True

    lines = [ln for ln in text.splitlines() if _keep_line(ln)]
    text = "\n".join(lines)
    # Also drop inline trailing disclaimer sentences.
    text = re.sub(
        r"(?:当前)?(?:未获取到|未读到).{0,40}(?:具体)?(?:实现)?(?:代码|源码)[，,].{0,40}(?:无法|不能)进一步确认[^。\n]*[。]?",
        "",
        text,
    )
    return text




def _strip_unverified_table_names(answer: str, *, evidence_text: str = "") -> str:
    """Remove invented t_* table citations not present in evidence."""
    text = answer or ""
    if not text:
        return text
    ev = (evidence_text or "").lower()
    stripped_any = False
    for fake in _KNOWN_FAKE_TABLES:
        if fake.lower() in ev:
            continue
        if fake not in text and fake.lower() not in text.lower():
            continue
        stripped_any = True
        # Drop parenthetical「（如 `t_xxx`）」or bare `t_xxx` mentions.
        text = re.sub(
            rf"[（(]?\s*如?\s*[`\"']?{re.escape(fake)}[`\"']?\s*[）)]?",
            "",
            text,
            flags=re.I,
        )
    # Broader: any t_* not in evidence → replace with soft unknown note once.
    found_unverified: list[str] = []
    for m in _TABLE_NAME_RE.finditer(text):
        name = m.group(1)
        if name.lower() in ev:
            continue
        if name.lower() in {f.lower() for f in _KNOWN_FAKE_TABLES}:
            continue  # already stripped
        found_unverified.append(name)
    for name in dict.fromkeys(found_unverified):
        stripped_any = True
        text = re.sub(
            rf"[（(]?\s*如?\s*[`\"']?{re.escape(name)}[`\"']?\s*[）)]?",
            "",
            text,
            flags=re.I,
        )
    # Clean debris left by stripping examples: （如 ）、和 。、双空格.
    text = re.sub(r"[（(]\s*如\s*[）)]", "", text)
    text = re.sub(r"[（(]\s*[）)]", "", text)
    text = re.sub(r"(?:和|与|及)\s*(?=[，,。．\n]|$)", "", text)
    text = re.sub(r"[，,]\s*(?=[，,。．\n])", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    # Hollow「查询用户…表」after table tokens gone → honest gap, not fake completeness.
    if stripped_any or found_unverified:
        if _HOLLOW_TABLE_PHRASE_RE.search(text) and not _TABLE_NAME_RE.search(text):
            text = _HOLLOW_TABLE_PHRASE_RE.sub("未在已读材料中看到具体表名", text)
            text = re.sub(r"未在已读材料中看到具体表名(?:\s*[，,]\s*)+", "未在已读材料中看到具体表名", text)
    return text




def _facts_map_evidence_text(facts_map: dict[str, dict[str, Any]]) -> str:
    """Flatten symbol_desc facts into an evidence bag for sanitize / grounding."""
    parts: list[str] = []
    for cm, facts in (facts_map or {}).items():
        if not isinstance(facts, dict):
            continue
        parts.append(str(cm or ""))
        for k in (
            "biz_clear",
            "biz_detail",
            "dev_clear",
            "summary",
            "capability",
            "status_line",
            "data_flow",
        ):
            v = facts.get(k)
            if v:
                parts.append(str(v))
        for key in ("business_rules", "validation_points", "http_paths"):
            for item in facts.get(key) or []:
                if item:
                    parts.append(str(item))
    return "\n".join(p for p in parts if p)




def _answer_has_gate_substance(answer: str, question: str = "") -> bool:
    """True when answer already names a gate method with concrete rule bullets."""
    a = answer or ""
    if len(a.strip()) < 40:
        return False
    if re.search(
        r"(?i)#\w*Check\w*|ServiceImpl#\w+|Client#\w+",
        a,
    ):
        if re.search(r"校验|可用|冻结|失效|开关|调用", a):
            return True
    themes = _gate_themes_in_question(question) if question else []
    if themes and any(t in a for t in themes):
        if re.search(r"[-*•]|\d+\.", a) and len(a) >= 80:
            return True
    return False




def _filter_footnote_gaps(gaps: list[Any], answer: str) -> list[str]:
    """Drop review gaps that demand source text when narrative already covers it."""
    a = answer or ""
    out: list[str] = []
    for g in gaps or []:
        s = str(g).strip()
        if not s:
            continue
        if re.search(r"具体实现代码|未提供.*源码|未读到.*实现|未查看.*实现", s):
            if _answer_has_gate_substance(a) or re.search(
                r"(?i)ServiceImpl#|#\w*Check\w*", a
            ):
                continue
        out.append(s)
    return out




def _with_review_footnote(
    answer: str,
    review: dict[str, Any],
    *,
    question: str = "",
) -> str:
    """Never surface review gaps as user-visible footnotes.

    Internal review / retry / SSE thinking may still use ``review``; chat answer
    body must stay free of「评审提示」「补充说明」footers. Also strip any leaked
    footer the model may have embedded in the prose.
    """
    from app.agent.question_intent import wants_add_field_change_plan

    _ = review  # kept for call-site compatibility
    text = answer or ""
    text = _REVIEW_FOOTER_RE.sub("", text)
    # Mid-body / non-paren leaks
    text = re.sub(
        r"[（(]?评审提示[：:][^）)\n]*[）)]?",
        "",
        text,
    )
    if wants_add_field_change_plan(question):
        text = _sanitize_add_field_change_plan(text, question=question)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text




def _evidence_brief(retrieved: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for h in retrieved:
        if not h:
            continue
        fqcn = h.get("fqcn") or ""
        sig = h.get("method_sig") or h.get("method_name") or ""
        key = f"{fqcn}#{sig}"
        if not fqcn or key in seen:
            continue
        seen.add(key)
        path = h.get("file_path") or ""
        lines.append(f"- {h.get('kind') or '?'} {fqcn}#{sig} @ {path}")
        if len(lines) >= 16:
            break
    return "\n".join(lines) if lines else "(无有效符号证据)"




