"""Question-intent classifiers extracted from explore_agent.py.

Pure intent routing: every function here depends only on explore_keys
vocabulary, explore_shared helpers, question_intent proxies, or external
modules — never on explore_agent internals (avoids circular imports).
explore_agent re-exports all names to keep existing imports working.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.explore_keys import (
    _API_INVENTORY_KEYS,
    _API_INVENTORY_STOP,
    _API_INVENTORY_STOP_EN,
    _CONTROLLER_HELPER_METHOD_RE,
    _ENDPOINT_KEYS,
    _FLOW_DOWNSTREAM_API_KEYS,
    _FULL_FLOW_KEYS,
    _GATE_DETAIL_METHOD_RE,
    _GATE_DETAIL_THEME_KEYS,
    _GATE_THEME_METHOD_STEMS,
    _QUERY_STOP_FOR_FLOW,
    _SOURCE_PASTE_PHRASES,
    _SOURCE_PASTE_RE,
    _TECH_DETAIL_KEYS,
    _VALIDATION_KEYS,
)
from app.agent.explore_shared import _simple_class_method
from app.agent.question_intent import (
    gate_themes_in_question as _qi_gate_themes,
    wants_add_field_change_plan,
    wants_composite_change,
    wants_extend_field_to_protocol_sync,
    wants_show_gate_code,
)
from app.core.utils import truncate

logger = logging.getLogger("app.agent.explore_intent")


def _wants_endpoint_lookup(question: str) -> bool:
    """User wants concrete remote service name / URL / Feign path — not a business essay."""
    text = question or ""
    return any(k in text for k in _ENDPOINT_KEYS)


def _wants_flow_downstream_apis(question: str) -> bool:
    """User asks which interfaces/services were called *along a flow*, not sibling Controllers."""
    text = question or ""
    if not text:
        return False
    if any(k in text for k in _FLOW_DOWNSTREAM_API_KEYS):
        return True
    # 「…过程中…哪些接口/服务」without explicit 调用
    if re.search(r"过程中.{0,12}(哪些|什么).{0,6}(接口|服务|客户端|Feign|Client)", text):
        return True
    return False


def _wants_api_inventory(question: str) -> bool:
    """User wants a count/list of related HTTP APIs (Controller endpoints)."""
    text = question or ""
    # 「调用了哪些接口」must NOT become a same-Controller endpoint catalog.
    if _wants_flow_downstream_apis(text):
        return False
    # Case-insensitive for the English keys ("Which APIs", "API list", ...).
    lower = text.lower()
    if any(k in lower for k in _API_INVENTORY_KEYS):
        return True
    # 「某接口有哪些 / 帮我看看XX接口有哪些 / XX接口都有哪些」— 接口与「有哪些」中间
    # 可插入「都」等口语词（desktop-chat-cb2aaea937da: 结清证明申请接口都有哪些）。
    if re.search(r"接口\s*(?:都)?\s*有哪些|有哪些\s*(?:都)?\s*接口", text):
        return True
    # English inventory shapes: "which APIs exist", "what APIs are there",
    # "list the endpoints", "APIs available for X". The which/what/list branch
    # MUST be followed by the API noun — bare "which"/"what" must not trigger
    # (e.g. "what is API design best practice" stays general).
    if re.search(
        r"(?:which|what)\s+(?:apis?|endpoints?|interfaces|controllers)"
        r"|list(?:\s+of)?\s+(?:the\s+)?(?:apis?|endpoints?|interfaces|controllers)"
        r"|(?:apis?|endpoints?|interfaces)\s+(?:are\s+)?(?:there|available|exist|exists)",
        lower,
    ):
        return True
    return False


def _api_inventory_query_core(question: str) -> str:
    """Strip inventory glue words; keep topic nouns for search (from the question)."""
    text = (question or "").strip()
    for stop in sorted(_API_INVENTORY_STOP, key=len, reverse=True):
        text = text.replace(stop, " ")
    # English glue is stripped whole-word only (substring "in"/"the"/"is"
    # would damage "include"/"other"/"this"). Word-boundary, case-insensitive.
    for stop in _API_INVENTORY_STOP_EN:
        text = re.sub(rf"(?i)\b{re.escape(stop)}\b", " ", text)
    text = re.sub(r"[？?！!。．，,、：:；;…]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _format_api_inventory_answer(
    items: list[dict[str, Any]], *, topic: str = ""
) -> str:
    """Render Controller endpoint list; N must match bullet count."""
    topic_label = f"「{topic}」" if topic else "相关"
    lines = [f"与{topic_label}主题相关的 HTTP 接口共 {len(items)} 个：", ""]
    for i, it in enumerate(items, 1):
        cm = str(it.get("cm") or "").strip()
        path = str(it.get("http_paths") or "").strip()
        doc = str(it.get("api_doc") or it.get("summary") or "").strip()
        bit = f"{i}. `{cm}`"
        if path:
            path_show = path.split(",")[0].strip()
            if path_show:
                bit += f"  path=`{path_show}`"
        if doc:
            bit += f"  — {truncate(doc, 80)}"
        lines.append(bit)
    return "\n".join(lines)


def _collect_api_inventory_items(
    project_id: str, question: str, *, top_k: int = 20
) -> list[dict[str, Any]]:
    """Gather Controller HTTP entries from entry_route + symbol search."""
    q_core = _api_inventory_query_core(question) or (question or "").strip()
    by_cm: dict[str, dict[str, Any]] = {}

    def _add(
        cm: str,
        *,
        http_paths: str = "",
        api_doc: str = "",
        summary: str = "",
        kind: str = "Controller",
        score: float = 0.0,
    ) -> None:
        c = _simple_class_method(cm) if cm else ""
        if not c or "#" not in c:
            return
        cls, meth = c.split("#", 1)
        meth = meth.split("(", 1)[0]
        if "Controller" not in cls and "Resource" not in cls:
            return
        if _CONTROLLER_HELPER_METHOD_RE.match(meth):
            return
        prev = by_cm.get(c)
        row = {
            "cm": c,
            "http_paths": http_paths or "",
            "api_doc": api_doc or "",
            "summary": summary or "",
            "kind": kind or "Controller",
            "score": float(score or 0),
        }
        if prev is None:
            by_cm[c] = row
            return
        # Prefer richer path/doc
        def _rich(x: dict[str, Any]) -> float:
            return (
                (3.0 if x.get("http_paths") else 0.0)
                + (2.0 if x.get("api_doc") else 0.0)
                + float(x.get("score") or 0)
            )

        if _rich(row) >= _rich(prev):
            by_cm[c] = row

    # 1) entry_route (dedicated system-entry index)
    try:
        from app.analysis.entry_route import search_entry_routes

        for h in search_entry_routes(q_core, project_id, top_k=top_k) or []:
            kind = str(h.get("kind") or "")
            if kind and kind.lower() not in {"controller", "http入口", ""}:
                # Keep Controllers; skip Job/Listener for「接口有哪些」unless no Controllers later
                if "controller" not in kind.lower():
                    continue
            paths = h.get("http_paths") or ""
            if isinstance(paths, list):
                paths = ",".join(str(p) for p in paths if p)
            _add(
                str(h.get("class_method") or ""),
                http_paths=str(paths or ""),
                api_doc=str(h.get("api_doc") or ""),
                kind=kind or "Controller",
                score=float(h.get("score") or 0),
            )
    except Exception:
        logger.debug("api inventory entry_route failed", exc_info=True)

    # 2) symbol hybrid search (Controller kind)
    try:
        from app.agent.explore_tools import search_symbols

        sr = search_symbols(project_id, q_core, kind="Controller", top_k=top_k)
        for h in sr.get("hits") or []:
            if not isinstance(h, dict):
                continue
            fqcn = str(h.get("fqcn") or "")
            meth = str(h.get("method_name") or "").strip()
            simple = fqcn.split(".")[-1] if fqcn else ""
            if not simple or not meth:
                continue
            extra = h.get("extra") if isinstance(h.get("extra"), dict) else {}
            paths = ""
            if isinstance(extra, dict):
                hp = extra.get("http_paths") or []
                if isinstance(hp, list):
                    paths = ",".join(str(p) for p in hp if p)
                elif hp:
                    paths = str(hp)
            _add(
                f"{simple}#{meth}",
                http_paths=paths,
                api_doc=str(h.get("summary") or ""),
                summary=str(h.get("summary") or ""),
                kind=str(h.get("kind") or "Controller"),
                score=float(h.get("score") or 0),
            )
    except Exception:
        logger.debug("api inventory search_symbols failed", exc_info=True)

    items = list(by_cm.values())
    # Soft relevance: if topic tokens exist, prefer items that mention them
    toks = [t for t in re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_]{2,}", q_core)]
    if toks:
        def _tok_score(it: dict[str, Any]) -> float:
            blob = f"{it.get('cm')} {it.get('api_doc')} {it.get('http_paths')} {it.get('summary')}".lower()
            hit = sum(1 for t in toks if t.lower() in blob)
            return hit * 10.0 + float(it.get("score") or 0)

        items.sort(key=_tok_score, reverse=True)
        # Drop zero-token matches only when we still keep ≥1 hit
        kept = [it for it in items if _tok_score(it) >= 10.0]
        if kept:
            items = kept
    else:
        items.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return items[:top_k]


def _assemble_api_inventory_answer(
    project_id: str, question: str
) -> tuple[str | None, list[dict[str, Any]]]:
    """Deterministic Controller inventory — no flow narration."""
    if not project_id or not _wants_api_inventory(question):
        return None, []
    items = _collect_api_inventory_items(project_id, question, top_k=20)
    if not items:
        return None, []
    topic = _api_inventory_query_core(question)
    answer = _format_api_inventory_answer(items, topic=topic)
    return answer, items


def _wants_tech_detail(question: str) -> bool:
    text = question or ""
    if _wants_endpoint_lookup(text) or _wants_api_inventory(text):
        return True
    # Change-plan asks need Class#method / file / DDL names.
    if (
        wants_composite_change(text)
        or wants_add_field_change_plan(text)
        or wants_extend_field_to_protocol_sync(text)
    ):
        return True
    return any(k in text for k in _TECH_DETAIL_KEYS)


def _wants_full_flow(question: str) -> bool:
    text = question or ""
    if _wants_endpoint_lookup(text):
        return False
    return any(k in text for k in _FULL_FLOW_KEYS)


def _wants_notice_entry(question: str) -> bool:
    """Deprecated product/notice keyword gate — always False.

    Topic switch / entry pick is LLM session-gate + Milvus/entry_route resolve.
    Kept as a no-op stub so call sites do not special-case 通知/notice nouns.
    """
    del question
    return False


def _lexical_main_entry(question: str) -> str | None:
    """No hardcoded business lexicon in agent (always None).

    Entry routing is LLM + evidence (entry_route / Milvus / FlowDoc / symbol
    search), never product-noun → Class#method keyword maps. Kept as a stub so
    call sites share one point and the no-domain-nouns architecture holds.
    """
    del question
    return None

# Structural role atoms: user says 通知/回调/监听 → match those English name stems.
# Not a product Class#method map.
_ATOMIC_CN_FOR_PICK: tuple[str, ...] = (
    "回调",
    "通知",
    "监听",
)
_CN_IDENTITY_STEMS_PICK: dict[str, tuple[str, ...]] = {
    "通知": ("notice", "notify", "notification"),
    "回调": ("callback", "callBack"),
    "监听": ("listen", "listener", "listenner"),
}


def _expand_pick_tokens(toks: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if not re.search(r"[\u4e00-\u9fff]", t):
            continue
        for atom in _ATOMIC_CN_FOR_PICK:
            if atom != t and atom in t and atom not in seen:
                seen.add(atom)
                out.append(atom)
    return out


def _question_match_tokens(question: str) -> list[str]:
    """Tokens from the user question only — used to pick the right FlowDoc."""
    text = question or ""
    toks: list[str] = []
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_./-]{1,}", text):
        t = m.group(0).lower()
        if t not in {"http", "api", "url", "mq"}:
            toks.append(t)
    # Strip question glue, then keep short Chinese content words.
    cleaned = text
    for stop in sorted(_QUERY_STOP_FOR_FLOW, key=len, reverse=True):
        cleaned = cleaned.replace(stop, " ")
    cleaned = re.sub(r"[？?！!。．，,、：:；;…]+", " ", cleaned)
    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", cleaned):
        toks.append(m.group(0))
    for m in re.finditer(r"[A-Za-z][A-Za-z0-9_./-]{1,}", cleaned):
        t = m.group(0).lower()
        if t not in {"http", "api", "url", "mq"}:
            toks.append(t)
    seen: set[str] = set()
    out: list[str] = []
    for t in toks:
        if t not in seen and t not in _QUERY_STOP_FOR_FLOW:
            seen.add(t)
            out.append(t)
    return _expand_pick_tokens(out)


def _token_hard_identity_for_pick(
    tok: str, *, cm: str, paths: list[str], api_doc: str
) -> bool:
    """Hard identity hit: api_doc / Class#method / paths — never FlowDoc narrative.

    No CN→EN business stem map; product pick is LLM / weak text overlap.
    """
    tl = (tok or "").strip().lower()
    if not tl:
        return False
    api_l = (api_doc or "").strip().lower()
    id_l = f"{cm} {' '.join(paths)} {api_doc}".lower()
    if api_l and (tl == api_l or tl in api_l or api_l in tl):
        return True
    if tl in id_l:
        return True
    return False


def _wants_sync_entry_first(question: str) -> bool:
    """User asks to start from the sync business entry (not a query/callback side path)."""
    q = question or ""
    # Notice/callback API flow — notice API is the entry; do not force another main.
    if _wants_notice_entry(q):
        return True
    if _wants_flow_downstream_apis(q):
        return True
    if re.search(r"从入口|入口到|主入口|主链|串一下|完整|全链路|端到端", q):
        return True
    # 「xx流程是什么样」通常要主业务入口，而不是 query/卡号查询之类旁路。
    if "流程" in q and not re.search(r"查询流程|怎么查|如何查|查一下", q):
        return True
    return _wants_full_flow(q)


def _wants_deep_flow_facts(question: str) -> bool:
    """True when the user asks for concrete logic / status / end-to-end chain.

    Bare「xx流程什么样 / 业务描述」also opts into deep FlowDoc render
    (status lines + concrete facts; Class#method only when tech detail is asked).
    """
    q = question or ""
    # Address / API-list questions are not flow-stage deep renders.
    if _wants_endpoint_lookup(q) or _wants_api_inventory(q):
        return False
    if re.search(
        r"具体|细节|状态|怎么变|如何更新|串一下|完整|全链路|从入口|下游|回调|逻辑",
        q,
    ):
        return True
    if _wants_tech_detail(q):
        return True
    # Sync flow-shape asks: 「某流程什么样？」「某接口什么流程」「…业务描述下」
    if "业务描述" in q:
        return True
    if "流程" in q and not re.search(r"查询流程|怎么查|如何查|查一下", q):
        if re.search(
            r"什么样|什么流程|怎么走|怎样|如何|描述|介绍|主链|调用链|链路|走一遍|走一趟",
            q,
        ):
            return True
        # 「某接口什么流程」「这个接口的流程」— 接口+流程 = deep sync chain
        if "接口" in q:
            return True
        if _wants_full_flow(q):
            return True
    return False


def _wants_validation_focus(question: str) -> bool:
    """Question asks about request-param / field checks (not full business flow)."""
    text = question or ""
    if _wants_endpoint_lookup(text) or _wants_api_inventory(text) or _wants_full_flow(text):
        return False
    # 「增加字段并落库改哪些」is a change plan, not a DTO validation checklist.
    if wants_composite_change(text) or wants_add_field_change_plan(text):
        return False
    # 「协议同步传字段」is outbound pass-through, not validation checklist.
    if wants_extend_field_to_protocol_sync(text):
        return False
    # Gate dig-ins (how a check step works) are not bare DTO field checklists.
    if _wants_gate_detail(text) or _wants_source_paste(text):
        return False
    return any(k in text for k in _VALIDATION_KEYS)


def _user_question_core(question: str) -> str:
    """Strip graph_agent session hints so gate-theme matching uses the user ask only."""
    text = (question or "").strip()
    if not text:
        return ""
    # Drop trailing （追问：…） / （新问题：…） / （需求复述：…） / （步骤序号：…） blocks.
    text = re.sub(
        r"[（(](?:追问|新问题|需求复述|步骤序号|关卡|符号|入口|候选符号|步骤标题)[：:][^）)]*[）)]",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip()


def extract_step_code_index(question: str) -> int | None:
    """Parse『第6步代码』/『6中对应代码』→ 6."""
    text = _user_question_core(question) or (question or "")
    if not text:
        return None
    for pat in (
        r"第\s*(\d+)\s*步",
        r"步骤\s*(\d+)",
        r"(?<!\d)(\d+)\s*中(?:对应)?(?:的)?(?:代码|源码|实现)",
        r"(?<!\d)(\d+)\s*[、.．]\s*(?:对应)?(?:的)?(?:代码|源码)",
    ):
        m = re.search(pat, text)
        if m:
            n = int(m.group(1))
            if 1 <= n <= 50:
                return n
    # Injected hint: （步骤序号：6；…）
    m = re.search(r"[（(]步骤序号[：:]\s*(\d+)", question or "")
    if m:
        n = int(m.group(1))
        if 1 <= n <= 50:
            return n
    return None


def _wants_add_field_change_plan(question: str) -> bool:
    """Proxy for question_intent — add-field / 落库 / 改哪些 developer plan."""
    return wants_add_field_change_plan(question)


def _wants_gate_detail(question: str) -> bool:
    """Removed: no business-gate theme router (always False)."""
    del question
    return False


def _wants_source_paste(question: str) -> bool:
    """User asks to see the actual Java method body (not a narrative summary)."""
    text = _user_question_core(question) or (question or "")
    if not text:
        return False
    if wants_composite_change(text) or wants_add_field_change_plan(text):
        return False
    if wants_show_gate_code(text):
        return True
    if any(p in text for p in _SOURCE_PASTE_PHRASES):
        return True
    if _SOURCE_PASTE_RE.search(text):
        return True
    # Method-token or theme + code ask (source paste).
    if re.search(r"源码|源代码|(?:部分)?代码", text) and (
        _GATE_DETAIL_METHOD_RE.search(text)
        or any(k in text for k in _GATE_DETAIL_THEME_KEYS)
    ):
        return True
    return False


def _session_gate_hint_from_question(question: str) -> tuple[str | None, list[str]]:
    """Parse graph_agent injections: （关卡：…；符号：Foo#bar,…）."""
    raw = question or ""
    gate = None
    symbols: list[str] = []
    m_gate = re.search(r"[（(]关卡[：:]\s*([^；;）)]+)", raw)
    if m_gate:
        gate = m_gate.group(1).strip() or None
    m_syms = re.search(r"符号[：:]\s*([^）)]+)[）)]", raw)
    if m_syms:
        for part in re.split(r"[,，、\s]+", m_syms.group(1)):
            p = part.strip()
            if p and "#" in p:
                symbols.append(_simple_class_method(p))
    return gate, symbols


def _gate_themes_in_question(question: str) -> list[str]:
    text = _user_question_core(question)
    themes: list[str] = []
    for key in _GATE_DETAIL_THEME_KEYS:
        if key in text and key not in themes:
            themes.append(key)
    # Also accept session injection「关卡：…」→ theme from the injected label.
    inj_gate, _ = _session_gate_hint_from_question(question)
    if inj_gate:
        for key in _GATE_DETAIL_THEME_KEYS:
            if key in inj_gate and key not in themes:
                themes.append(key)
        # Full injected gate label may itself be the theme.
        if not themes:
            for key in _GATE_DETAIL_THEME_KEYS:
                if inj_gate.startswith(key) or key in inj_gate:
                    themes.append(key)
                    break
    if _GATE_DETAIL_METHOD_RE.search(text):
        m = _GATE_DETAIL_METHOD_RE.search(text)
        name = (m.group(0) if m else "").lower()
        for theme, stems in _GATE_THEME_METHOD_STEMS.items():
            if any(stem in name for stem in stems) and theme not in themes:
                themes.append(theme)
    # Align with question_intent theme table when explore keys miss a cue.
    for t in _qi_gate_themes(question):
        if t not in themes:
            themes.append(t)
    return themes


def _method_matches_gate_themes(class_method: str, themes: list[str]) -> bool:
    cm = (class_method or "").strip()
    if not cm or "#" not in cm or not themes:
        return False
    meth = cm.split("#", 1)[-1].split("(", 1)[0].lower()
    full = cm.lower()
    # Skip unrelated Acc / limit-center clients even if stem overlaps "quota".
    if re.search(r"(?i)AccClient|LimitClient|QuotaController|AmountController", cm):
        return False
    for theme in themes:
        stems = _GATE_THEME_METHOD_STEMS.get(theme) or ()
        for stem in stems:
            if stem in meth or stem in full.replace("#", "").lower():
                return True
    return False


def _wants_callback_section(question: str) -> bool:
    """True when user wants a trailing 回调/通知 section of another main flow."""
    text = question or ""
    # Notice-API flow asks about the notice API itself — not another entry's callback.
    if _wants_notice_entry(text):
        return False
    return any(k in text for k in ("回调", "通知", "异步落地", "下游结果", "串一下", "全链路", "完整"))
