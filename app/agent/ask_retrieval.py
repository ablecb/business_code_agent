"""Ask retrieval: Milvus → LLM plans Neo4j (templates) → MySQL → LLM summarize.

No business-domain routers. Intent is decided upstream; this module only runs
when intent == ask.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.utils import truncate
from app.llm.factory import get_llm_provider

logger = logging.getLogger("app.agent.ask_retrieval")

TEMPLATE_IDS = frozenset(
    {"follows_spine", "calls_callees", "calls_callers", "feign_neighbors"}
)

_ENTRY_PICK_PROMPT = """你是业务入口判定器。根据用户问题，从候选系统入口里选唯一最贴切的 Class#method。
只输出 JSON：{"entry":"XxxController#xxx","reason":"一句中文"}

规则：
- entry 必须原样选自候选列表，禁止编造
- 问某业务接口/流程/加参落库：选该业务的 HTTP 主入口（flow_role=main），不要选无关通知/回调
- 明确问通知/回调/查询/定时时再选对应 role
- 方法名通常等于去掉 Controller 后的类名（例如 FooController#foo）
- 两个候选方法名相同时：选 api_doc 能在问题里整段对上的那个，不要选只是同类名的兄弟 Controller
"""

_ENTRY_FILTER_PROMPT = """你是业务入口过滤器。根据用户问题，从候选系统入口里保留所有相关的 Class#method（可多个）。
只输出 JSON：{"entries":["XxxController#a","YyyController#a"],"reason":"一句中文"}

规则：
- entries 必须原样选自候选列表，禁止编造；至少 1 个，相关则尽量保留多个
- 同名方法、不同 Controller（兄弟入口）若都与问题业务相关，必须都保留（例如两个 #creditNotification）
- 明显无关的入口（别的业务主链）去掉
- 不要为了「唯一答案」强行只留一个
"""


def _parse_json_obj_loose(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _candidate_brief_rows(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    from app.agent.explore_agent import _class_method_from_hint

    brief: list[dict[str, Any]] = []
    allow: set[str] = set()
    for c in candidates[:8]:
        if not isinstance(c, dict):
            continue
        cm = _class_method_from_hint(str(c.get("cm") or c.get("class_method") or ""))
        if not cm or "#" not in cm or cm in allow:
            continue
        allow.add(cm)
        brief.append(
            {
                "cm": cm,
                "flow_role": str(c.get("flow_role") or "main"),
                "api_doc": truncate(str(c.get("api_doc") or ""), 80),
                "http_paths": truncate(str(c.get("http_paths") or ""), 80),
                "score": float(c.get("score") or 0.0),
            }
        )
    return brief, allow


def _llm_filter_flow_entries(
    question: str,
    candidates: list[dict[str, Any]],
) -> list[str]:
    """Ask LLM which candidate entries remain relevant (1+). Empty on failure."""
    from app.agent.explore_agent import _class_method_from_hint

    q = (question or "").strip()
    brief, allow = _candidate_brief_rows(candidates)
    if not q or not brief:
        return []
    if len(brief) == 1:
        return [brief[0]["cm"]]
    try:
        llm = get_llm_provider()
        text = llm.chat_sync(
            [
                {"role": "system", "content": _ENTRY_FILTER_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"问题：{truncate(q, 400)}\n"
                        f"候选：\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n"
                        "只输出 JSON。"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=280,
        )
        data = _parse_json_obj_loose(text) or {}
        raw_list = data.get("entries")
        if not isinstance(raw_list, list):
            one = _class_method_from_hint(str(data.get("entry") or ""))
            raw_list = [one] if one else []
        out: list[str] = []
        seen: set[str] = set()
        for item in raw_list:
            cm = _class_method_from_hint(str(item or ""))
            if cm and cm in allow and cm not in seen:
                seen.add(cm)
                out.append(cm)
        return out
    except Exception:
        logger.debug("LLM entry filter failed", exc_info=True)
        return []


def _llm_pick_flow_entry(
    question: str,
    candidates: list[dict[str, Any]],
) -> str:
    """Ask LLM to pick Class#method from a short candidate list. Empty on failure."""
    from app.agent.explore_agent import _class_method_from_hint

    filtered = _llm_filter_flow_entries(question, candidates)
    if filtered:
        return filtered[0]
    q = (question or "").strip()
    brief, allow = _candidate_brief_rows(candidates)
    if not q or not brief:
        return ""
    if len(brief) == 1:
        return brief[0]["cm"]
    try:
        llm = get_llm_provider()
        text = llm.chat_sync(
            [
                {"role": "system", "content": _ENTRY_PICK_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"问题：{truncate(q, 400)}\n"
                        f"候选：\n{json.dumps(brief, ensure_ascii=False, indent=2)}\n"
                        "只输出 JSON。"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=200,
        )
        picked = _class_method_from_hint(
            str((_parse_json_obj_loose(text) or {}).get("entry") or "")
        )
        if picked and picked in allow:
            return picked
    except Exception:
        logger.debug("LLM entry pick failed", exc_info=True)
    return ""


def _method_name_of_cm(cm: str) -> str:
    text = (cm or "").strip()
    if "#" not in text:
        return ""
    return text.split("#", 1)[-1].split("(", 1)[0].strip().lower()


def _same_method_siblings(cms: list[str]) -> list[str]:
    """Controllers that share a method name (e.g. two #creditNotification)."""
    by_method: dict[str, list[str]] = {}
    for cm in cms:
        m = _method_name_of_cm(cm)
        if not m:
            continue
        by_method.setdefault(m, []).append(cm)
    for group in by_method.values():
        if len(group) >= 2:
            return group
    return []


def _order_entries_primary_first(entries: list[str], primary: str) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for cm in [primary, *entries]:
        c = (cm or "").strip()
        if not c or c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


_NEO4J_PLAN_PROMPT = """你是 Neo4j 检索规划器（只读）。根据用户问题、对话历史、entry_route 与 Milvus 命中，选择一个固定模板并填参数。
只输出 JSON：
{"template_id":"follows_spine|calls_callees|calls_callers|feign_neighbors","params":{"entry":"XxxController#xxx","depth":4,"limit":20},"reason":"一句中文"}

模板：
- follows_spine: 主链/流程执行序
- calls_callees: 从 entry 出发的下游调用
- calls_callers: 谁调用了 entry
- feign_neighbors: Feign/外部调用邻域

规则：
- entry 只能是系统入口：Controller / Job(XXL-JOB) / Listener / MQ Consumer
  （来自 entry_route 或标注为入口的候选），禁止选 Service/ServiceImpl/普通工具方法
- 问「接口/流程/逻辑/怎么走」且未点名通知/回调时：优先 flow_role=main 的 HTTP 主入口，
  不要选 Notice/通知类入口（除非问题明确问通知/回调）
- 方法名通常等于去掉 Controller 后的类名（XxxController#xxx）；带 http_paths / flow_role=main 更优先
- 禁止把 Controller 内部辅助方法当流程入口（如 *DealWith、check*Param、generate*、insert*）
- 定时/调度问题优先 Job；消息/监听/回调优先 Listener/MQ
- 追问时结合历史主题与本轮问题选入口与模板
- 建议主入口可改选为「可选 entry 列表」中其它合法系统入口；无法判断时用建议主入口 + follows_spine
"""

_REFUSE_NO_ENTRY = (
    "未能定位到可靠的系统入口（Controller/Job/Listener），"
    "证据不足，无法准确回答。请换一种问法或点名接口/类名后再问。"
)
_REFUSE_THIN_EVIDENCE = (
    "已定位到入口，但下游调用图与检索交集证据不足，"
    "未在已读材料中展开到可核对的处理步骤，不会编造流程。"
    "可追问具体类名/方法名，或先问该接口的主流程。"
)

# Structural helper names on Controllers — not HTTP facades (no product nouns).
_CONTROLLER_HELPER_RE = re.compile(
    r"(?i)^(check\w*|validate\w*|generate\w*|insert\w*|update\w*|applyresult\w*|"
    r".*dealwith$|after\w*|before\w*|build\w*|convert\w*|to\w*vo$|from\w*)"
)

# System entry class stems (structural suffixes / roles — not business domains).
_ENTRY_KIND_OK = frozenset(
    {"controller", "job", "listener", "mq", "consumer", "entry", "handler"}
)
_ENTRY_CLASS_RE = re.compile(
    r"(?i)(Controller|Job|Listener|Listenner|Consumer|MqConsumer|MessageHandler)$"
)

_SUMMARIZE_PROMPT = """你是代码问答助手。严格依据下方证据回答，禁止编造表名/接口/URL/步骤。
用清楚的中文按步骤组织。

硬性规则：
- 每一步必须能对应到证据 steps 里某条 summary/desc/title；对不上就整段改答
  「未在已读材料中看到足够细节，证据不足，不会编造。」
- 证据 steps 少于 2 条时：禁止写 3 步及以上的完整流程，直接说明证据不足
- 禁止用无关命中拼凑通用校验故事
- 正文默认禁止 Class#method、Java 类名、方法名、包名、文件路径、symbol_id
- 用业务说法写；只有用户明确要求「类名/方法名/源码/代码」时才可写技术标识
"""

_CLASS_METHOD_RE = re.compile(
    r"`?[A-Za-z][\w]*(?:\.[A-Za-z][\w]*)*#[A-Za-z_][\w]*`?"
)
_JAVA_TYPE_RE = re.compile(
    r"\b(?:com|org|net)\.[A-Za-z][\w.]*(?:\.[A-Za-z][\w]*)+\b"
)


async def _emit(on_progress: Any, event: dict[str, Any]) -> None:
    if not on_progress:
        return
    try:
        maybe = on_progress(event)
        if hasattr(maybe, "__await__"):
            await maybe  # type: ignore[misc]
    except Exception:
        logger.debug("ask_retrieval progress emit failed", exc_info=True)


def _parse_json_obj(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _hit_to_cm(hit: Any) -> str | None:
    from app.agent.explore_agent import _class_method_from_hint

    if hit is None:
        return None
    if isinstance(hit, dict):
        for key in ("class_method", "entry", "fqcn"):
            cm = _class_method_from_hint(str(hit.get(key) or ""))
            if cm:
                return cm
        blob = f"{hit.get('summary') or ''} {' '.join(hit.get('tags') or [])}"
        cm = _class_method_from_hint(blob)
        if cm:
            return cm
        fqcn = str(hit.get("fqcn") or "")
        meth = str(hit.get("method_sig") or hit.get("method_name") or "")
        simple = fqcn.split(".")[-1] if fqcn else ""
        meth = meth.split("#")[-1].split("(")[0].strip()
        if simple and meth and re.match(r"^[A-Za-z_]\w*$", meth):
            return f"{simple}#{meth}"
        return None
    tags = getattr(hit, "tags", None) or []
    blob = f"{getattr(hit, 'summary', '') or ''} {' '.join(tags)}"
    cm = _class_method_from_hint(blob)
    if cm:
        return cm
    fqcn = getattr(hit, "fqcn", "") or ""
    simple = fqcn.split(".")[-1] if fqcn else ""
    sig = getattr(hit, "method_sig", None) or ""
    meth = str(sig).split("#")[-1].split("(")[0].strip()
    if simple and meth and re.match(r"^[A-Za-z_]\w*$", meth):
        return f"{simple}#{meth}"
    return None


def _is_placeholder_entry(cm: str | None) -> bool:
    """Reject prompt placeholders like Controller#method / Class#method."""
    text = (cm or "").strip()
    if not text or "#" not in text:
        return True
    cls, _, meth = text.partition("#")
    simple = cls.split(".")[-1].strip()
    meth = meth.split("(")[0].strip()
    if not simple or not meth:
        return True
    if simple.lower() in {"controller", "class", "xxx", "job", "listener"}:
        return True
    if meth.lower() in {"method", "xxx"}:
        return True
    return False


def _normalize_template_choice(
    raw: dict[str, Any], *, default_entry: str
) -> dict[str, Any]:
    from app.agent.explore_agent import _class_method_from_hint

    tid = str(raw.get("template_id") or "").strip().lower()
    if tid not in TEMPLATE_IDS:
        tid = "follows_spine"
    params = raw.get("params") if isinstance(raw.get("params"), dict) else {}
    entry = _class_method_from_hint(str(params.get("entry") or "")) or ""
    if _is_placeholder_entry(entry):
        entry = ""
    if not entry or _is_placeholder_entry(entry):
        entry = default_entry if not _is_placeholder_entry(default_entry) else ""
    try:
        depth = int(params.get("depth") or 4)
    except Exception:
        depth = 4
    try:
        limit = int(params.get("limit") or 20)
    except Exception:
        limit = 20
    depth = max(1, min(depth, 8))
    limit = max(1, min(limit, 40))
    return {
        "template_id": tid,
        "params": {"entry": entry, "depth": depth, "limit": limit},
        "reason": str(raw.get("reason") or "").strip()[:240],
    }


def controller_facade_cm(cm_or_class: str | None) -> str | None:
    """XxxController#any → XxxController#xxx (structural HTTP facade name)."""
    text = (cm_or_class or "").strip()
    if not text:
        return None
    simple = text.split("#", 1)[0].split(".")[-1]
    if not simple.endswith("Controller"):
        return None
    stem = re.sub(r"Controller$", "", simple, flags=re.I)
    if not stem or not re.match(r"^[A-Za-z_]\w*$", stem):
        return None
    meth = stem[0].lower() + stem[1:] if stem[0].isupper() else stem
    return f"{simple}#{meth}"


def synthesize_facades_from_hits(cms: list[str]) -> list[str]:
    """From helper hits like FooController#checkX, propose FooController#foo."""
    out: list[str] = []
    seen: set[str] = set()
    for cm in cms:
        facade = controller_facade_cm(cm)
        if not facade or facade in seen:
            continue
        seen.add(facade)
        out.append(facade)
    return out


def _class_simple(cm: str | None) -> str:
    text = (cm or "").strip()
    if "#" not in text:
        return text.split(".")[-1]
    return text.split("#", 1)[0].split(".")[-1]


def is_system_entry_point(
    cm: str | None, *, kind: str | None = None, fqcn: str | None = None
) -> bool:
    """True for Controller / Job / Listener / MQ entry classes (structural)."""
    kind_l = (kind or "").strip().lower()
    if kind_l in _ENTRY_KIND_OK:
        return True
    simple = _class_simple(cm)
    if simple and _ENTRY_CLASS_RE.search(simple):
        return True
    fq = (fqcn or "").strip().split(".")[-1]
    return bool(fq and _ENTRY_CLASS_RE.search(fq))


def is_controller_http_facade(cm: str | None) -> bool:
    """True when Class#method looks like XxxController#xxx (HTTP facade)."""
    text = (cm or "").strip()
    if not text or "#" not in text:
        return False
    cls, _, meth = text.partition("#")
    simple = cls.split(".")[-1]
    meth = meth.split("(")[0].strip()
    if not simple.endswith("Controller") or not meth:
        return False
    if _CONTROLLER_HELPER_RE.match(meth):
        return False
    stem = re.sub(r"Controller$", "", simple, flags=re.I).lower()
    return meth.lower() == stem or meth.lower() in {f"do{stem}", f"apply{stem}"}


def is_controller_helper_method(cm: str | None) -> bool:
    """Controller method that is unlikely to be the HTTP flow entry.

    Only structural helper naming (check*/validate*/build*/…). Do not treat
    every non-stem method as helper — creditNotification / creditQuery are
    real HTTP entries even when they are not CreditController#credit.
    """
    text = (cm or "").strip()
    if not text or "#" not in text:
        return False
    cls, _, meth = text.partition("#")
    simple = cls.split(".")[-1]
    meth = meth.split("(")[0].strip()
    if not simple.endswith("Controller"):
        return False
    if is_controller_http_facade(text):
        return False
    return bool(_CONTROLLER_HELPER_RE.match(meth))


def _milvus_brief(hits: list[Any], *, limit: int = 12) -> list[dict[str, Any]]:
    brief: list[dict[str, Any]] = []
    for h in hits[:limit]:
        d = h.model_dump() if hasattr(h, "model_dump") else (
            h if isinstance(h, dict) else {
                "symbol_id": getattr(h, "symbol_id", None),
                "fqcn": getattr(h, "fqcn", ""),
                "kind": getattr(h, "kind", ""),
                "summary": getattr(h, "summary", ""),
                "score": getattr(h, "score", None),
                "file_path": getattr(h, "file_path", ""),
                "method_sig": getattr(h, "method_sig", None),
                "tags": getattr(h, "tags", None) or [],
            }
        )
        cm = _hit_to_cm(h)
        kind = str(d.get("kind") or "")
        fqcn = str(d.get("fqcn") or "")
        brief.append(
            {
                "symbol_id": d.get("symbol_id"),
                "fqcn": fqcn,
                "kind": kind,
                "cm": cm,
                "summary": truncate(str(d.get("summary") or ""), 160),
                "score": d.get("score"),
                "is_system_entry": is_system_entry_point(cm, kind=kind, fqcn=fqcn),
                "is_http_facade": is_controller_http_facade(cm),
                "is_helper": is_controller_helper_method(cm),
            }
        )
    # Entry-point classes first; demote helpers / ServiceImpl.
    brief.sort(
        key=lambda b: (
            0 if b.get("is_http_facade") else 1,
            0 if b.get("is_system_entry") else 2,
            2 if b.get("is_helper") else 0,
            -(float(b.get("score") or 0.0)),
        )
    )
    return brief


def _entry_route_mysql_fallback(
    project_id: str, question: str, *, top_k: int = 8
) -> list[dict[str, Any]]:
    """When entry_route Milvus is empty: pick Controller/Job/Listener from MySQL.

    Scores by (a) Latin identity tokens (CreditUsageApply / useCredit …) AND
    (b) Chinese api_doc / summary overlap — 「用信申请接口」 must match an
    entry whose api_doc contains 用信申请 (creditUsageApply) instead of falling
    back to arbitrary Controller order (desktop-chat-09dfa4ec7ba7: 用信申请
    → WithdrawalsController#withdrawals).
    """
    if not project_id:
        return []
    q = (question or "").strip()
    try:
        from sqlalchemy import or_, select

        from app.db.mysql import get_session
        from app.models.orm import SymbolDescORM, SymbolORM
    except Exception:
        return []

    tokens = re.findall(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_]{2,}", q)
    # Keep Latin identity tokens only; product CN→EN mapping is LLM / entry_route.
    like_bits = [t for t in tokens if re.match(r"^[A-Za-z]", t)][:6]
    # Chinese business phrases (2-gram window) for api_doc overlap matching.
    cn_phrases: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]{2,16}", q)[:4]:
        if len(run) >= 2:
            cn_phrases.append(run)
        for i in range(len(run) - 1):
            g = run[i : i + 2]
            if g not in cn_phrases:
                cn_phrases.append(g)
    out: list[dict[str, Any]] = []
    try:
        with get_session() as session:
            rows = list(
                session.scalars(
                    select(SymbolORM)
                    .where(
                        SymbolORM.project_id == project_id,
                        or_(
                            SymbolORM.kind.in_(
                                ["Controller", "Job", "Listener", "Entry"]
                            ),
                            SymbolORM.simple_name.like("%Controller"),
                            SymbolORM.simple_name.like("%Job"),
                            SymbolORM.simple_name.like("%Listener"),
                            SymbolORM.simple_name.like("%Consumer"),
                        ),
                    )
                    .limit(200)
                )
            )
        # api_doc / summary per symbol (Chinese routing signal).
        api_by_cm: dict[str, str] = {}
        summary_by_cm: dict[str, str] = {}
        try:
            with get_session() as session:
                desc_rows = session.execute(
                    select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
                ).all()
                for d in desc_rows:
                    extra = d.extra_json if isinstance(d.extra_json, dict) else {}
                    api = str(extra.get("api_doc") or "").strip()
                    summary = str(d.summary or "").strip()
                    cm = f"{((d.symbol_id or '').split('.')[-1])}"
                    if not api:
                        m = re.search(r"([\u4e00-\u9fff]{2,12})接口|([\u4e00-\u9fff]{2,12})申请", summary)
                        if m:
                            api = m.group(1) or m.group(2) or ""
                    api_by_cm[d.symbol_id or ""] = api
                    summary_by_cm[d.symbol_id or ""] = summary
        except Exception:
            logger.debug("entry_route fallback desc read failed", exc_info=True)

        scored: list[tuple[float, str, SymbolORM]] = []
        for sym in rows:
            simple = (sym.simple_name or "").strip() or (sym.fqcn or "").split(".")[-1]
            meth = (sym.method_name or "").strip()
            if not simple or not meth:
                continue
            cm = f"{simple}#{meth}"
            if is_controller_helper_method(cm):
                continue
            score = 0.0
            id_l = f"{simple} {meth} {sym.fqcn or ''}".lower()
            if is_controller_http_facade(cm):
                score += 20.0
            if is_system_entry_point(cm, kind=sym.kind, fqcn=sym.fqcn):
                score += 5.0
            for t in like_bits:
                if t.lower() in id_l:
                    score += 12.0
            # Chinese api_doc / summary overlap — strong signal for 纯中文接口问法.
            api = api_by_cm.get(sym.id or "", "")
            summary = summary_by_cm.get(sym.id or "", "")
            blob_cn = f"{simple} {api} {summary[:200]}"
            if cn_phrases:
                hits = 0
                for ph in cn_phrases[:6]:
                    if ph and ph in blob_cn:
                        hits += 1
                if hits:
                    # 「用信申请」全短语命中 > 散落 bigram
                    full_hits = sum(
                        1 for ph in cn_phrases if len(ph) >= 4 and ph in blob_cn
                    )
                    score += 6.0 * hits + 4.0 * full_hits
            # Mild boost when question mentions 接口/流程 and this is a Controller facade
            if any(k in q for k in ("接口", "流程", "怎么走")) and simple.endswith(
                "Controller"
            ):
                score += 3.0
            scored.append((score, cm, sym))
        scored.sort(key=lambda x: x[0], reverse=True)
        for score, cm, sym in scored[:top_k]:
            if score <= 0 and not is_controller_http_facade(cm):
                continue
            out.append(
                {
                    "cm": cm,
                    "kind": sym.kind or "",
                    "flow_role": "main",
                    "http_paths": "",
                    "api_doc": api_by_cm.get(sym.id or "", ""),
                    "score": score,
                    "is_system_entry": True,
                    "is_http_facade": is_controller_http_facade(cm),
                    "is_helper": False,
                    "source": "mysql_entry_fallback",
                }
            )
    except Exception:
        logger.debug("entry_route mysql fallback failed", exc_info=True)
    return out


def _entry_route_candidates(
    project_id: str, question: str, *, top_k: int = 8
) -> list[dict[str, Any]]:
    """entry_route (Milvus) first; MySQL Controller/Job/Listener fallback."""
    out: list[dict[str, Any]] = []
    try:
        from app.analysis.entry_route import search_entry_routes

        for h in search_entry_routes(question, project_id, top_k=top_k) or []:
            cm = str(h.get("class_method") or "")
            if not cm or _is_placeholder_entry(cm):
                continue
            kind = str(h.get("kind") or "")
            out.append(
                {
                    "cm": cm,
                    "kind": kind,
                    "flow_role": h.get("flow_role"),
                    "http_paths": h.get("http_paths") or "",
                    "api_doc": truncate(str(h.get("api_doc") or ""), 120),
                    "routing_text": truncate(str(h.get("routing_text") or ""), 400),
                    "score": h.get("score"),
                    "is_system_entry": True,
                    "is_http_facade": is_controller_http_facade(cm),
                    "is_helper": is_controller_helper_method(cm),
                    "source": "entry_route",
                }
            )
    except Exception:
        logger.debug("entry_route search failed", exc_info=True)
    if not out:
        out = _entry_route_mysql_fallback(project_id, question, top_k=top_k)
    return out


def _infer_flow_role(cm: str, *, explicit: str = "") -> str:
    """Structural flow_role for an entry (reuse entry_route classifier)."""
    try:
        from app.analysis.entry_route import classify_flow_role

        simple, _, meth = (cm or "").partition("#")
        # Always pass identity into classifier so Notice* overrides stale offline main.
        return classify_flow_role(
            class_method=cm,
            method_name=meth.split("(")[0],
            fqcn=simple,
            kind="Controller" if "Controller" in (simple or "") else "",
            existing_role=(explicit or "").strip().lower(),
        )
    except Exception:
        role = (explicit or "").strip().lower()
        return role if role in {"main", "query", "notice", "callback", "job", "async"} else "main"


def _entry_topic_queries(question: str, *, entry_hint: str | None = None) -> list[str]:
    """Prefer short topic queries (e.g. 某接口) over long add-field sentences."""
    q = (question or "").strip()
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        text = re.sub(r"\s+", " ", (s or "").strip())
        if text and text not in seen:
            seen.add(text)
            out.append(text)

    hint = (entry_hint or "").strip()
    if hint and "#" not in hint:
        _add(hint)
    for m in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9_]{2,16}接口)", q):
        _add(m.group(1))

    core = re.sub(
        r"帮我在|帮我|请帮我|我想在|我想|请|一下|入参|增加一个|增加|新增|字段|"
        r"校验不能为空|不能为空|校验|并落库到|并落库|落库到|落库|这两张表|这张表|"
        r"都要改哪些地方|改哪些|哪些地方|businessLabel|业务标签|"
        r"[（(][^）)]*[）)]|"
        r"\bdemo_[a-z0-9_]+\b",
        " ",
        q,
        flags=re.I,
    )
    # Drop obvious new-field camelCase tokens (aaaName / businessLabel).
    core = re.sub(r"\b[a-z]+[A-Z][A-Za-z0-9]*\b", " ", core)
    core = re.sub(r"\s+", " ", core).strip(" ，,。.;；")
    if core:
        _add(core)
    if q:
        _add(q)
    return out


def _entry_lock_queries(question: str, *, entry_hint: str | None = None) -> list[str]:
    """Queries used to lock HTTP entry —「xx接口」only, no gate/field tails.

    Gate tokens (状态/校验) otherwise pull an unrelated Controller instead of
    the named API facade. Falls back to ``_entry_topic_queries`` when no
    API phrase is present.
    """
    q = (question or "").strip()
    out: list[str] = []
    seen: set[str] = set()

    def _add(s: str) -> None:
        text = re.sub(r"\s+", " ", (s or "").strip())
        if text and text not in seen:
            seen.add(text)
            out.append(text)

    hint = (entry_hint or "").strip()
    if hint and "#" not in hint:
        _add(hint)
    for m in re.finditer(r"([\u4e00-\u9fffA-Za-z0-9_]{2,16}接口)", q):
        _add(m.group(1))
    if out:
        return out
    return _entry_topic_queries(question, entry_hint=entry_hint)


def _search_controller_facade_routes(
    project_id: str, question: str, *, top_k: int = 10, entry_hint: str | None = None
) -> list[dict[str, Any]]:
    """When entry_route Milvus is empty: search_symbols → Controller#facade rows."""
    if not project_id or not (question or "").strip():
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        from app.agent.explore_shared import _method_name_from_hit
        from app.agent.explore_tools import search_symbols

        queries = _entry_lock_queries(question, entry_hint=entry_hint)
        for q in queries:
            sr = search_symbols(project_id, q, top_k=top_k)
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "")
                simple = fqcn.split(".")[-1]
                if not simple:
                    continue
                mname = _method_name_from_hit(h)
                cm = f"{simple}#{mname}" if mname else simple
                facade = controller_facade_cm(cm) or (
                    controller_facade_cm(simple) if simple.endswith("Controller") else None
                )
                if not facade or facade in seen:
                    continue
                if not is_controller_http_facade(facade):
                    continue
                seen.add(facade)
                out.append(
                    {
                        "cm": facade,
                        "kind": "Controller",
                        "flow_role": _infer_flow_role(facade),
                        "http_paths": "",
                        "api_doc": "",
                        "score": float(h.get("score") or 0.0) or 1.0,
                        "is_system_entry": True,
                        "is_http_facade": True,
                        "is_helper": False,
                        "source": "search_symbols_facade",
                        "routing_text": f"{facade} {q}",
                    }
                )
    except Exception:
        logger.debug("search_controller_facade_routes failed", exc_info=True)
    return out


def resolve_flow_entries_with_meta(
    project_id: str,
    question: str,
    *,
    entry_hint: str | None = None,
    prefer_hint: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Like resolve_flow_entries_for_question, plus route rows for matched cms."""
    from app.agent.explore_agent import _class_method_from_hint

    hint = _class_method_from_hint(entry_hint) or ""
    if not project_id or not (question or "").strip():
        if hint and is_controller_http_facade(hint):
            return [hint], []
        return [], []

    if prefer_hint and hint and "#" in hint:
        if not is_controller_helper_method(hint):
            return [hint], []

    lock_qs = _entry_lock_queries(question, entry_hint=entry_hint)
    entry_q = lock_qs[0] if lock_qs else (question or "").strip()
    routes: list[dict[str, Any]] = []
    seen_cm: set[str] = set()
    for tq in lock_qs[:4] or [entry_q]:
        extra = _entry_route_candidates(project_id, tq, top_k=8)
        for h in extra:
            cm = str(h.get("cm") or "")
            if not cm or cm in seen_cm:
                continue
            seen_cm.add(cm)
            routes.append(h)
    if not routes:
        routes = _search_controller_facade_routes(
            project_id, entry_q, top_k=10, entry_hint=entry_hint
        )
        seen_cm = {str(h.get("cm") or "") for h in routes}
    wants_http = bool(
        re.search(r"接口|Controller", question or "")
        and not re.search(r"(?i)定时|调度|\bjob\b|xxl|监听|mq|回调接口", question or "")
    )
    if wants_http and not any(
        is_controller_http_facade(str(h.get("cm") or "")) for h in routes
    ):
        for tq in lock_qs[:3] or [entry_q]:
            facades = _search_controller_facade_routes(
                project_id, tq, top_k=8, entry_hint=entry_hint
            )
            for h in facades:
                cm = str(h.get("cm") or "")
                if cm and cm not in seen_cm:
                    seen_cm.add(cm)
                    routes.append(h)
            if any(is_controller_http_facade(str(h.get("cm") or "")) for h in routes):
                break
    hit_cms = [str(h.get("cm")) for h in routes if h.get("cm")]
    for facade in synthesize_facades_from_hits(hit_cms):
        if not any(str(h.get("cm")) == facade for h in routes):
            routes.append(
                {
                    "cm": facade,
                    "kind": "Controller",
                    "flow_role": _infer_flow_role(facade),
                    "http_paths": "",
                    "api_doc": "",
                    "score": 1.0,
                    "is_system_entry": True,
                    "is_http_facade": True,
                    "is_helper": False,
                    "source": "synthesized_facade",
                }
            )
            hit_cms.append(facade)
    preferred = prefer_flow_entries(
        hit_cms,
        entry_routes=routes,
        entry_hint=hint or entry_hint,
        question=entry_q,
        prefer_hint=prefer_hint,
        project_id=project_id,
    )
    if not preferred:
        if hint and is_system_entry_point(hint) and not is_controller_helper_method(hint):
            preferred = [hint]
        else:
            facades = synthesize_facades_from_hits(hit_cms)
            preferred = facades[:1] if facades else []
    want = set(preferred)
    meta = [h for h in routes if str(h.get("cm") or "") in want]
    # Preserve prefer order for meta when possible.
    by_cm = {str(h.get("cm") or ""): h for h in meta}
    ordered_meta = [by_cm[cm] for cm in preferred if cm in by_cm]
    return preferred, ordered_meta


def resolve_flow_entries_for_question(
    project_id: str,
    question: str,
    *,
    entry_hint: str | None = None,
    prefer_hint: bool = False,
) -> list[str]:
    """Resolve one or more Controller#method entries (sibling matches kept)."""
    ents, _meta = resolve_flow_entries_with_meta(
        project_id,
        question,
        entry_hint=entry_hint,
        prefer_hint=prefer_hint,
    )
    return ents


def resolve_flow_entry_for_question(
    project_id: str,
    question: str,
    *,
    entry_hint: str | None = None,
    prefer_hint: bool = False,
) -> str:
    """Resolve Controller#method for ask/impact from entry_route (no lexicon).

    Used by ask_retrieval and add-field discover when session has no lock.
    Falls back to search_symbols→Controller facade when entry_route store is empty.
    Primary of :func:`resolve_flow_entries_for_question`.
    """
    ents = resolve_flow_entries_for_question(
        project_id,
        question,
        entry_hint=entry_hint,
        prefer_hint=prefer_hint,
    )
    return ents[0] if ents else ""


def _api_doc_in_question_score(
    api_doc: str, question: str, *, extra: str = ""
) -> int:
    """Higher when Swagger/api_doc matches the 接口问句 label.

    Distinguishes sibling Controllers that share a method name (e.g. both
    ``#creditNotification``) without a product-noun map: short ask label
    beats a longer mid-insert sibling; summary glue gaps stay acceptable.
    """
    from app.analysis.entry_route import interface_label_score

    return interface_label_score(api_doc or "", question or "", extra=extra or "")


def _facade_has_real_method(project_id: str | None, cm: str | None) -> bool:
    """True when an HTTP facade Class#method actually exists in the symbol index.

    Synthesized facades (FooController#foo from the class stem, with no real
    method of that name) must not crowd real entry_route candidates out of the
    candidate pool — otherwise the LLM arbiter only ever sees the phantom entry
    (desktop-chat-43c9b593e7e7: 结清证明申请接口 → SignatureController#signature
    → empty spine → "证据不足" refusal while the real signatureApply flow exists).
    Fail-open on None project_id / index errors: never break resolution.
    """
    if not project_id or "#" not in str(cm or ""):
        return True
    try:
        return bool(_resolve_symbol_id(str(project_id), str(cm)))
    except Exception:
        return True


def prefer_flow_entries(
    candidates: list[str],
    *,
    entry_routes: list[dict[str, Any]] | None = None,
    milvus_brief: list[dict[str, Any]] | None = None,
    entry_hint: str | None = None,
    question: str | None = None,
    prefer_hint: bool = False,
    project_id: str | None = None,
) -> list[str]:
    """Lock Neo4j entries to Controller/Job/Listener/MQ — never Service helpers.

    Returns one or more Class#method. Same-method sibling Controllers that survive
    AI filter are all kept (e.g. two ``#creditNotification``).
    """
    from app.agent.explore_agent import _class_method_from_hint

    meta: dict[str, dict[str, Any]] = {}
    ordered: list[str] = []

    def _add(cm: str | None, **flags: Any) -> None:
        c = _class_method_from_hint(str(cm or ""))
        if not c:
            return
        if c not in ordered:
            ordered.append(c)
        prev = meta.get(c) or {}
        merged = {**prev, **flags}
        # entry_route structural flags must not be overwritten by later candidate
        # heuristics (e.g. helper/non-facade mis-marks on HTTP methods).
        if prev.get("source") == "entry_route":
            for k in (
                "is_helper",
                "is_http_facade",
                "is_system_entry",
                "flow_role",
                "http_paths",
                "api_doc",
                "routing_text",
            ):
                if k in prev and prev[k] not in (None, ""):
                    merged[k] = prev[k]
            merged["source"] = "entry_route"
        if not merged.get("flow_role"):
            merged["flow_role"] = _infer_flow_role(
                c, explicit=str(flags.get("flow_role") or prev.get("flow_role") or "")
            )
        else:
            merged["flow_role"] = _infer_flow_role(
                c, explicit=str(merged.get("flow_role") or "")
            )
        meta[c] = merged

    routes = list(entry_routes or [])
    if question and routes:
        try:
            from app.analysis.entry_route import rerank_entry_routes_for_question

            as_hits = [
                {
                    "class_method": h.get("cm") or h.get("class_method"),
                    "flow_role": h.get("flow_role") or "main",
                    "api_doc": h.get("api_doc") or "",
                    "http_paths": h.get("http_paths") or "",
                    "routing_text": h.get("routing_text") or "",
                    "fqcn": h.get("fqcn") or "",
                    "method_name": (
                        str(h.get("cm") or "").split("#", 1)[-1]
                        if h.get("cm")
                        else ""
                    ),
                    "score": float(h.get("score") or 0.0),
                }
                for h in routes
            ]
            ranked = rerank_entry_routes_for_question(as_hits, question)
            cm_to_route = {
                str(h.get("cm") or h.get("class_method") or ""): h for h in routes
            }
            routes = []
            for hit in ranked:
                cm = str(hit.get("class_method") or "")
                base = dict(cm_to_route.get(cm) or {})
                base["cm"] = cm
                base["flow_role"] = hit.get("flow_role") or base.get("flow_role")
                base["score"] = hit.get("score")
                if hit.get("routing_text") and not base.get("routing_text"):
                    base["routing_text"] = hit.get("routing_text")
                routes.append(base)
        except Exception:
            logger.debug("prefer_flow_entry rerank failed", exc_info=True)

    for h in routes:
        _add(
            h.get("cm"),
            is_system_entry=True,
            is_http_facade=bool(
                h.get("is_http_facade")
                if "is_http_facade" in h
                else is_controller_http_facade(str(h.get("cm") or ""))
            ),
            is_helper=bool(h.get("is_helper")),
            flow_role=str(h.get("flow_role") or ""),
            http_paths=str(h.get("http_paths") or ""),
            api_doc=str(h.get("api_doc") or ""),
            routing_text=str(h.get("routing_text") or ""),
            source="entry_route",
        )
    for b in milvus_brief or []:
        _add(
            b.get("cm"),
            is_system_entry=bool(b.get("is_system_entry")),
            is_http_facade=bool(b.get("is_http_facade")),
            is_helper=bool(b.get("is_helper")),
            flow_role=str(b.get("flow_role") or ""),
            source="milvus",
        )
    hint = _class_method_from_hint(entry_hint)
    # Session lock already cleared by LLM when topic switches; keep hint as-is.
    if hint:
        _add(
            hint,
            is_system_entry=is_system_entry_point(hint),
            is_http_facade=is_controller_http_facade(hint),
            is_helper=is_controller_helper_method(hint),
            source="hint",
        )
    for c in candidates:
        _add(
            c,
            is_system_entry=is_system_entry_point(c),
            is_http_facade=is_controller_http_facade(c),
            is_helper=is_controller_helper_method(c),
            source="candidate",
        )

    # Only keep real system entries; drop ServiceImpl etc.
    entries = [
        cm
        for cm in ordered
        if meta.get(cm, {}).get("is_system_entry")
        and not meta.get(cm, {}).get("is_helper")
    ]
    if not entries:
        entries = [
            cm
            for cm in ordered
            if meta.get(cm, {}).get("source") == "entry_route"
            and not meta.get(cm, {}).get("is_helper")
        ]
    if not entries:
        return []

    # Synthesized facades (FooController#foo — no real method of that name)
    # must never crowd out real candidates: their empty api_doc plus routing
    # text can even win the doc_ranked primary pick below (desktop-chat-43c9
    # b593e7e7: SignatureController#signature scored 96 and beat the real
    # signatureApply flow).  Drop them whenever real entries exist; when the
    # facade is the *only* candidate the caller re-synthesizes / clears it
    # downstream (facade guard), so behavior there is unchanged.
    if project_id:
        real_entries = [
            cm
            for cm in entries
            if not (
                meta.get(cm, {}).get("is_http_facade")
                and not _facade_has_real_method(project_id, cm)
            )
        ]
        if real_entries:
            entries = real_entries

    # Dig-in / session lock: Class#method hint wins before LLM re-pick.
    if prefer_hint and hint and hint in entries and not is_controller_helper_method(hint):
        return [hint]

    wanted: set[str] = set()
    if question:
        try:
            from app.analysis.entry_route import _question_wants_role

            wanted = _question_wants_role(question)
        except Exception:
            wanted = set()

    facades = [
        cm
        for cm in entries
        if meta.get(cm, {}).get("is_http_facade")
        and _facade_has_real_method(project_id, cm)
    ]
    pool = facades or entries

    # Structural shortlist by role preference (not product noun maps).
    shortlist: list[str] = []
    if not wanted:
        mains = [
            cm
            for cm in pool
            if (meta.get(cm, {}).get("flow_role") or "main") == "main"
        ]
        # 「…接口」+ 关卡/状态 dig-in: keep HTTP main, drop job/async noise.
        if question and re.search(r"接口", question) and not re.search(
            r"(?i)定时|调度|\bjob\b|xxl", question
        ):
            http_mains = [
                cm
                for cm in mains
                if meta.get(cm, {}).get("is_http_facade")
                or (
                    str(cm).split("#", 1)[0].endswith("Controller")
                    and (meta.get(cm, {}).get("flow_role") or "")
                    not in {"job", "async", "notice", "callback"}
                )
            ]
            if http_mains:
                mains = http_mains
        shortlist = mains or [
            cm
            for cm in pool
            if (meta.get(cm, {}).get("flow_role") or "")
            not in {"notice", "callback"}
        ] or list(pool)
    else:
        # Role asks (notice/callback/…) must see non-stem Controller methods
        # (e.g. creditNotification) even when is_http_facade is stem-only False.
        role_pool = entries or pool
        role_hit = [
            cm
            for cm in role_pool
            if (meta.get(cm, {}).get("flow_role") or "") in wanted
        ]
        shortlist = role_hit or list(pool)

    if prefer_hint and hint and hint in shortlist:
        return [hint]

    route_by_cm = {
        str(h.get("cm") or h.get("class_method") or ""): h for h in routes
    }

    def _doc_of(cm: str) -> tuple[str, str]:
        m = meta.get(cm) or {}
        api = ""
        if m.get("api_doc"):
            api = str(m.get("api_doc") or "")
        else:
            api = str((route_by_cm.get(cm) or {}).get("api_doc") or "")
        extra = str(
            m.get("routing_text")
            or (route_by_cm.get(cm) or {}).get("routing_text")
            or ""
        )
        return api, extra

    def _cand_rows_for(cms: list[str]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for cm in cms[:8]:
            m = meta.get(cm) or {}
            base = dict(route_by_cm.get(cm) or {})
            rows.append(
                {
                    "cm": cm,
                    "flow_role": m.get("flow_role") or base.get("flow_role") or "main",
                    "api_doc": m.get("api_doc") or base.get("api_doc") or "",
                    "http_paths": m.get("http_paths")
                    or base.get("http_paths")
                    or "",
                    "routing_text": m.get("routing_text")
                    or base.get("routing_text")
                    or "",
                    "score": float(base.get("score") or 0.0),
                }
            )
        return rows

    def _filter_siblings(pool_cms: list[str], *, primary: str) -> list[str]:
        """AI-filter same-method siblings; keep all that remain (or all on fail)."""
        siblings = _same_method_siblings(pool_cms)
        if len(siblings) < 2:
            return [primary] if primary else list(pool_cms[:1])
        filtered = _llm_filter_flow_entries(str(question or ""), _cand_rows_for(siblings))
        if len(filtered) >= 2:
            return _order_entries_primary_first(filtered, primary)
        if len(filtered) == 1:
            # AI dropped siblings — still surface co-searched same-method Controllers
            # when the question looks like an interface/flow ask (user expects both).
            if question and re.search(r"接口|流程|入口", question):
                return _order_entries_primary_first(siblings, primary or filtered[0])
            return filtered
        # LLM fail: keep all co-searched siblings (do not silently hide one).
        return _order_entries_primary_first(siblings, primary)

    # LLM / sibling filter among shortlist when question is present and >1 candidate.
    if question and len(shortlist) >= 1:
        qn = str(question).strip()
        doc_ranked: list[tuple[int, str]] = []
        for cm in shortlist:
            api, extra = _doc_of(cm)
            doc_ranked.append(
                (_api_doc_in_question_score(api, qn, extra=extra), cm)
            )
        doc_ranked.sort(key=lambda x: (-x[0], x[1]))
        # Contiguous api⊂问句 (≥100) or clear fuzzy/summary winner (≥70 with margin).
        primary = ""
        if doc_ranked and doc_ranked[0][0] >= 70:
            second = doc_ranked[1][0] if len(doc_ranked) > 1 else -1
            margin = 8 if doc_ranked[0][0] >= 100 else 15
            if doc_ranked[0][0] - second >= margin:
                primary = doc_ranked[0][1]
        if primary:
            return _filter_siblings(shortlist, primary=primary)

        cand_rows = _cand_rows_for(shortlist)
        filtered = _llm_filter_flow_entries(qn, cand_rows)
        if len(filtered) >= 2:
            return filtered
        if len(filtered) == 1:
            return _filter_siblings(shortlist, primary=filtered[0])
        picked = _llm_pick_flow_entry(qn, cand_rows)
        if picked and picked in shortlist:
            return _filter_siblings(shortlist, primary=picked)

    # Deterministic fallback (structural only).
    if shortlist:
        return _filter_siblings(shortlist, primary=shortlist[0]) if question else [shortlist[0]]
    for cm in pool:
        if meta.get(cm, {}).get("is_http_facade"):
            return [cm]
    for cm in entries:
        m = meta.get(cm) or {}
        if m.get("flow_role") == "main" and m.get("http_paths"):
            return [cm]
    for cm in entries:
        if meta.get(cm, {}).get("source") == "entry_route":
            return [cm]
    return [entries[0]]


def prefer_flow_entry(
    candidates: list[str],
    *,
    entry_routes: list[dict[str, Any]] | None = None,
    milvus_brief: list[dict[str, Any]] | None = None,
    entry_hint: str | None = None,
    question: str | None = None,
    prefer_hint: bool = False,
    project_id: str | None = None,
) -> str:
    """Lock Neo4j entry to Controller/Job/Listener/MQ — never Service helpers.

    When the question does not ask for notice/callback/query/job/async, prefer
    flow_role=main HTTP facades over notice/callback (structural roles only).
    Primary of :func:`prefer_flow_entries`.
    """
    ents = prefer_flow_entries(
        candidates,
        entry_routes=entry_routes,
        milvus_brief=milvus_brief,
        entry_hint=entry_hint,
        question=question,
        prefer_hint=prefer_hint,
        project_id=project_id,
    )
    return ents[0] if ents else ""


def _sanitize_neo4j_choice(
    choice: dict[str, Any],
    *,
    preferred_entry: str,
    allowed_entries: list[str],
    question: str | None = None,
) -> dict[str, Any]:
    """Keep valid system-entry picks; rewrite helpers / notice-vs-main mismatches."""
    from app.agent.explore_agent import _class_method_from_hint

    entry = _class_method_from_hint(str((choice.get("params") or {}).get("entry") or ""))
    preferred = _class_method_from_hint(preferred_entry) or (preferred_entry or "")
    allow = {
        _class_method_from_hint(c) or c
        for c in allowed_entries
        if c and is_system_entry_point(c) and not is_controller_helper_method(c)
    }
    if preferred:
        allow.add(preferred)

    wanted: set[str] = set()
    if question:
        try:
            from app.analysis.entry_route import _question_wants_role

            wanted = _question_wants_role(question)
        except Exception:
            wanted = set()

    # No notice/callback ask → do not keep a notice entry when preferred is main.
    if (
        entry
        and preferred
        and not wanted
        and _infer_flow_role(entry) in {"notice", "callback"}
        and _infer_flow_role(preferred) == "main"
    ):
        return {
            **choice,
            "params": {**(choice.get("params") or {}), "entry": preferred},
            "reason": truncate(
                f"{choice.get('reason') or ''}; 非通知问法，入口由 notice 纠正为 {preferred}".strip(
                    "; "
                ),
                240,
            ),
        }

    # Valid system entry in the pool → keep (do not force back to preferred).
    if (
        entry
        and not _is_placeholder_entry(entry)
        and is_system_entry_point(entry)
        and not is_controller_helper_method(entry)
        and (not allow or entry in allow)
    ):
        return choice

    if preferred and not _is_placeholder_entry(preferred):
        return {
            **choice,
            "params": {**(choice.get("params") or {}), "entry": preferred},
            "reason": truncate(
                f"{choice.get('reason') or ''}; 入口纠正为系统入口 {preferred}".strip(
                    "; "
                ),
                240,
            ),
        }
    return choice


async def plan_neo4j_with_llm(
    question: str,
    *,
    milvus_brief: list[dict[str, Any]],
    history_text: str | None = None,
    prior_question: str | None = None,
    entry_hint: str | None = None,
    entry_routes: list[dict[str, Any]] | None = None,
    project_id: str | None = None,
) -> dict[str, Any]:
    """LLM chooses Neo4j template + entry from entry_route / entry-class hits."""
    from app.agent.explore_agent import _class_method_from_hint

    # entry_route rows carry project_id; use it when the caller did not pass one.
    pid = project_id
    if not pid:
        for h in (*(entry_routes or []), *(milvus_brief or [])):
            if isinstance(h, dict) and h.get("project_id"):
                pid = str(h["project_id"])
                break

    # Neo4j flow entry pool: entry_route ∪ Milvus system-entry classes only.
    entry_pool: list[str] = []
    for h in entry_routes or []:
        if h.get("cm") and not h.get("is_helper"):
            entry_pool.append(str(h["cm"]))
    for b in milvus_brief:
        if b.get("cm") and b.get("is_system_entry") and not b.get("is_helper"):
            entry_pool.append(str(b["cm"]))
    preferred = prefer_flow_entry(
        entry_pool,
        entry_routes=entry_routes,
        milvus_brief=milvus_brief,
        entry_hint=entry_hint,
        question=question,
        project_id=pid or None,
    )
    default_entry = preferred or _class_method_from_hint(entry_hint) or (
        entry_pool[0] if entry_pool else ""
    )
    default = _normalize_template_choice({}, default_entry=default_entry)
    # Only show entry-class rows to the LLM for entry selection.
    entry_brief = [
        b
        for b in milvus_brief
        if b.get("is_system_entry") and not b.get("is_helper")
    ][:12]
    llm = get_llm_provider()
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": _NEO4J_PLAN_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"对话历史:\n{truncate(history_text or '(空)', 2000)}\n\n"
                        f"上一问: {(prior_question or '(空)').strip()}\n"
                        f"本轮问题: {(question or '').strip()}\n"
                        f"建议主入口（可从候选改选，须同为系统入口）: {preferred or '(空)'}\n"
                        f"entry_route候选(仅系统入口):\n"
                        f"{truncate(json.dumps(entry_routes or [], ensure_ascii=False, indent=2), 3000)}\n"
                        f"Milvus系统入口命中:\n"
                        f"{truncate(json.dumps(entry_brief, ensure_ascii=False, indent=2), 5000)}\n"
                        f"可选 entry 列表: {json.dumps(list(dict.fromkeys(entry_pool))[:20], ensure_ascii=False)}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=400,
        )
        data = _parse_json_obj(content)
        if data:
            choice = _normalize_template_choice(data, default_entry=default_entry)
            return _sanitize_neo4j_choice(
                choice,
                preferred_entry=preferred,
                allowed_entries=entry_pool,
                question=question,
            )
    except Exception:
        logger.debug("plan_neo4j_with_llm failed", exc_info=True)
    return default


def _resolve_symbol_id(project_id: str, class_method: str) -> str | None:
    cm = (class_method or "").strip()
    if not project_id or "#" not in cm:
        return None
    simple, _, meth = cm.partition("#")
    simple = simple.split(".")[-1].strip()
    meth = meth.split("(")[0].strip()
    if not simple or not meth:
        return None
    try:
        from sqlalchemy import select

        from app.db.mysql import get_session
        from app.models.orm import SymbolORM

        with get_session() as session:
            rows = session.scalars(
                select(SymbolORM).where(
                    SymbolORM.project_id == project_id,
                    SymbolORM.method_name == meth,
                )
            ).all()
            for row in rows:
                if (row.simple_name or "").endswith(simple) or (
                    row.fqcn or ""
                ).endswith(simple):
                    return row.id
            # Strict: a Class#method lookup must match the class.  Returning a
            # same-named method from a *different* class (the old `rows[0]`
            # fallback) made synthesized facades like SignatureController#signature
            # (no such method exists) pass `_entry_is_real`, so the facade guard
            # never cleared them and the ask died with an empty spine.
            return None
    except Exception:
        logger.debug("resolve_symbol_id failed for %s", cm, exc_info=True)
    return None


def _run_neo4j_template(
    project_id: str, choice: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute a fixed read-only template; return (ordered nodes, raw meta)."""
    tid = choice["template_id"]
    params = choice["params"]
    entry = params["entry"]
    depth = int(params.get("depth") or 4)
    limit = int(params.get("limit") or 20)
    nodes: list[dict[str, Any]] = []
    meta: dict[str, Any] = {"template_id": tid, "params": params, "ok": False}

    if tid == "follows_spine":
        try:
            from app.analysis.flow_neo4j_sync import get_follows_path

            path = get_follows_path(project_id, entry)
            meta["raw_source"] = path.get("source")
            meta["ok"] = bool(path.get("ok"))
            ordered = list(path.get("ordered_path") or [])
            steps = list(path.get("steps") or [])
            step_by = {
                str(s.get("class_method") or ""): s
                for s in steps
                if isinstance(s, dict)
            }
            for i, cm in enumerate(ordered):
                st = step_by.get(str(cm)) or {}
                nodes.append(
                    {
                        "order": i + 1,
                        "class_method": str(cm),
                        "title": st.get("title") or "",
                        "rel": "FOLLOWS",
                        "desc": st.get("desc")
                        or st.get("desc_business")
                        or "",
                    }
                )
            if not nodes and entry:
                nodes = [
                    {
                        "order": 1,
                        "class_method": entry,
                        "title": "",
                        "rel": "ENTRY",
                        "desc": "",
                    }
                ]
                meta["ok"] = True
        except Exception as e:
            logger.warning("follows_spine failed: %s", e)
            meta["error"] = str(e)

    elif tid in {"calls_callees", "calls_callers", "feign_neighbors"}:
        try:
            from app.graph.builder import query_flow

            flow = query_flow(entry, project_id, depth=depth)
            meta["ok"] = True
            meta["raw_nodes"] = len(flow.get("nodes") or [])
            meta["raw_edges"] = len(flow.get("edges") or [])
            want_rel = {
                "calls_callees": {"CALLS", "ASYNC_CALLS", "TRIGGERS"},
                "calls_callers": {"CALLS", "ASYNC_CALLS", "TRIGGERS"},
                "feign_neighbors": {"FEIGN_CALLS", "PUBLISHES", "CONSUMES"},
            }[tid]
            edges = [
                e
                for e in (flow.get("edges") or [])
                if (e.get("type") or "") in want_rel
            ]
            nodes.append(
                {
                    "order": 1,
                    "class_method": entry,
                    "title": "",
                    "rel": "ENTRY",
                    "desc": "",
                }
            )
            seen = {entry}
            sid = _resolve_symbol_id(project_id, entry)
            if sid and tid in {"calls_callees", "calls_callers"}:
                from app.agent.explore_tools import _neo4j_neighbors

                direction = "out" if tid == "calls_callees" else "in"
                nb = _neo4j_neighbors(
                    project_id, sid, direction=direction, limit=limit
                )
                for n in nb.get("neighbors") or []:
                    fqcn = str(n.get("fqcn") or "")
                    name = str(n.get("name") or "")
                    simple = fqcn.split(".")[-1] if fqcn else ""
                    cm = f"{simple}#{name}" if simple and name else name or fqcn
                    if not cm or cm in seen:
                        continue
                    seen.add(cm)
                    nodes.append(
                        {
                            "order": len(nodes) + 1,
                            "class_method": cm,
                            "title": "",
                            "rel": n.get("rel") or "",
                            "desc": "",
                            "symbol_id": n.get("symbol_id"),
                        }
                    )
            for e in edges[:limit]:
                end = str(e.get("end") or "")
                start = str(e.get("start") or "")
                target = end if tid != "calls_callers" else start
                if not target or target in seen:
                    continue
                from app.agent.explore_agent import _class_method_from_hint

                cm = _class_method_from_hint(target) or target
                if cm in seen:
                    continue
                seen.add(cm)
                nodes.append(
                    {
                        "order": len(nodes) + 1,
                        "class_method": cm,
                        "title": "",
                        "rel": e.get("type") or "",
                        "desc": "",
                    }
                )
            if tid == "feign_neighbors" and len(nodes) <= 1:
                for n in flow.get("nodes") or []:
                    props = n.get("properties") or {}
                    blob = f"{n.get('id')} {props}"
                    if not re.search(r"(?i)feign|client", blob):
                        continue
                    cm = str(
                        props.get("class_method")
                        or props.get("fqcn")
                        or n.get("id")
                        or ""
                    )
                    from app.agent.explore_agent import _class_method_from_hint

                    cm2 = _class_method_from_hint(cm) or cm
                    if not cm2 or cm2 in seen:
                        continue
                    seen.add(cm2)
                    nodes.append(
                        {
                            "order": len(nodes) + 1,
                            "class_method": cm2,
                            "title": "",
                            "rel": "FEIGN",
                            "desc": "",
                        }
                    )
                    if len(nodes) >= limit:
                        break
        except Exception as e:
            logger.warning("%s failed: %s", tid, e)
            meta["error"] = str(e)
            meta["ok"] = False

    if not nodes and entry:
        nodes = [
            {
                "order": 1,
                "class_method": entry,
                "title": "",
                "rel": "ENTRY",
                "desc": "",
            }
        ]
    return nodes[:limit], meta


def _mysql_enrich_ordered(
    project_id: str,
    nodes: list[dict[str, Any]],
    *,
    max_nodes: int = 12,
    code_chars_each: int = 1200,
    code_budget: int = 6000,
) -> list[dict[str, Any]]:
    """Fill Symbol/SymbolDesc (+ optional truncated code) in Neo4j order."""
    from sqlalchemy import select

    from app.agent.explore_tools import get_symbol_code
    from app.db.mysql import get_session
    from app.models.orm import SymbolDescORM, SymbolORM

    out: list[dict[str, Any]] = []
    used_code = 0
    for n in nodes[:max_nodes]:
        cm = str(n.get("class_method") or "")
        row_info: dict[str, Any] = {
            "order": n.get("order"),
            "class_method": cm,
            "rel": n.get("rel"),
            "title": n.get("title") or "",
            "neo4j_desc": truncate(str(n.get("desc") or ""), 400),
        }
        sid = n.get("symbol_id") or _resolve_symbol_id(project_id, cm)
        try:
            with get_session() as session:
                sym: SymbolORM | None = None
                if sid:
                    sym = session.get(SymbolORM, sid)
                if sym is None and "#" in cm:
                    simple, _, meth = cm.partition("#")
                    simple = simple.split(".")[-1]
                    meth = meth.split("(")[0]
                    candidates = session.scalars(
                        select(SymbolORM).where(
                            SymbolORM.project_id == project_id,
                            SymbolORM.method_name == meth,
                        )
                    ).all()
                    for c in candidates:
                        if (c.simple_name or "") == simple or (
                            c.fqcn or ""
                        ).endswith("." + simple):
                            sym = c
                            break
                    if sym is None and candidates:
                        sym = candidates[0]
                if sym is not None:
                    sid = sym.id
                    row_info.update(
                        {
                            "symbol_id": sym.id,
                            "fqcn": sym.fqcn,
                            "kind": sym.kind,
                            "file_path": sym.file_path,
                            "method_name": sym.method_name,
                        }
                    )
                    desc = session.scalar(
                        select(SymbolDescORM).where(
                            SymbolDescORM.symbol_id == sym.id
                        )
                    )
                    if desc:
                        ej = dict(desc.extra_json or {})
                        fd = (
                            ej.get("flow_doc")
                            if isinstance(ej.get("flow_doc"), dict)
                            else None
                        )
                        row_info["summary"] = truncate(
                            str(
                                desc.summary_business_clear
                                or desc.summary
                                or ""
                            ),
                            800,
                        )
                        if fd:
                            row_info["flow_doc_title"] = fd.get("title")
                            stages = fd.get("stages") or []
                            row_info["flow_doc_stages"] = [
                                {
                                    "class_method": s.get("class_method"),
                                    "title": s.get("title"),
                                    "desc": truncate(
                                        str(
                                            s.get("desc")
                                            or s.get("desc_business")
                                            or ""
                                        ),
                                        200,
                                    ),
                                }
                                for s in stages[:12]
                                if isinstance(s, dict)
                            ]
        except Exception:
            logger.debug("mysql enrich failed for %s", cm, exc_info=True)

        if sid and used_code < code_budget:
            try:
                code_res = get_symbol_code(
                    project_id, symbol_id=str(sid), max_chars=code_chars_each
                )
                if code_res.get("ok") and code_res.get("raw_code"):
                    snippet = truncate(str(code_res["raw_code"]), code_chars_each)
                    row_info["code_snippet"] = snippet
                    used_code += len(snippet)
            except Exception:
                pass
        out.append(row_info)
    return out


def _wants_tech_names(question: str) -> bool:
    text = question or ""
    keys = ("类名", "方法名", "源码", "源代码", "代码", "Class#", "哪个类", "哪个方法", "fqcn")
    return any(k in text for k in keys)


def _strip_tech_names(text: str) -> str:
    """Remove Class#method / Java FQCN from user-facing answer."""
    out = _CLASS_METHOD_RE.sub("", text or "")
    out = _JAVA_TYPE_RE.sub("", out)
    # Clean leftover empty backticks / double spaces around punctuation
    out = re.sub(r"`\s*`", "", out)
    out = re.sub(r"[（(]\s*[）)]", "", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" *([，,。；;：:])", r"\1", out)
    return out.strip()


def _business_text(text: str | None) -> str:
    t = _strip_tech_names(str(text or "")).strip(" ：:-—")
    return t


async def _summarize_answer(
    question: str,
    *,
    neo4j_query: dict[str, Any],
    mysql_rows: list[dict[str, Any]],
    milvus_brief: list[dict[str, Any]],
    history_text: str | None = None,
    focus_rels: set[str] | None = None,
) -> str:
    llm = get_llm_provider()
    allow_tech = _wants_tech_names(question)
    focus = focus_rels or {"CALLS_INTERSECT"}
    # Do not feed Class#method into the summarizer unless user asked for tech names.
    evidence_rows = []
    for r in mysql_rows:
        row = {
            "order": r.get("order"),
            "rel": r.get("rel"),
            "title": _business_text(r.get("title")),
            "summary": _business_text(r.get("summary")),
            "desc": _business_text(r.get("neo4j_desc")),
            "priority": "dig_in" if (r.get("rel") or "") in focus else "context",
        }
        if allow_tech:
            row["class_method"] = r.get("class_method")
            row["fqcn"] = r.get("fqcn")
        evidence_rows.append(row)
    # Dig-in rows first for the model
    evidence_rows.sort(
        key=lambda r: (0 if r.get("priority") == "dig_in" else 1, r.get("order") or 0)
    )
    evidence = {
        "neo4j_query": {
            "template_id": neo4j_query.get("template_id"),
            "reason": neo4j_query.get("reason"),
        },
        "steps": evidence_rows,
        "hit_summaries": [
            _business_text(b.get("summary"))
            for b in milvus_brief[:6]
            if b.get("summary")
        ],
    }
    dig_hint = ""
    if any(r.get("priority") == "dig_in" for r in evidence_rows):
        dig_hint = (
            "用户在问主链上的「那一块」：请优先依据 priority=dig_in 的证据回答该关卡，"
            "不要改写成整条无关通知链路。\n"
        )
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": _SUMMARIZE_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"对话历史:\n{truncate(history_text or '(空)', 1200)}\n\n"
                        f"问题: {question}\n"
                        f"{dig_hint}"
                        f"允许写类名方法名: {'是' if allow_tech else '否'}\n\n"
                        f"证据(已脱敏，勿还原类名):\n"
                        f"{truncate(json.dumps(evidence, ensure_ascii=False, indent=2), 14000)}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=2500,
        )
        text = (content or "").strip()
        if text:
            return text if allow_tech else _strip_tech_names(text)
    except Exception:
        logger.exception("ask summarize LLM failed")

    lines = ["## 回答（基于检索）", ""]
    for r in mysql_rows:
        title = _business_text(
            r.get("title") or r.get("summary") or r.get("neo4j_desc")
        )
        if not title:
            title = "相关处理步骤"
        lines.append(f"{r.get('order')}. {truncate(title, 120)}")
    return "\n".join(lines)


def _simple_cm(cm: str | None) -> str:
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    return f"{left.split('.')[-1].strip()}#{right.split('(', 1)[0].strip()}"


def _dig_in_search_query(question: str) -> str:
    """Strip chat/flow glue so dig-in nouns from the question dominate search."""
    text = question or ""
    text = re.sub(
        r"我想看看|我想看|帮我看|看看|看下|请|一下|"
        r"那一块|这一块|那块|这块|对应的|流程中|怎么写的|怎么写|怎么做|"
        r"主要做什么|主要做了什么|主要是做|做了什么|做什么的|做了啥|做什么|"
        r"什么样|怎么走|是什么|如何|源码|源代码|代码",
        " ",
        text,
    )
    # Drop lone structural tails; keep topic nouns from the question.
    text = re.sub(r"(接口|流程|逻辑)(?=\s|$)", " ", text)
    return re.sub(r"\s+", " ", text).strip() or (question or "").strip()


_CLASS_METHOD_IN_TEXT = re.compile(
    r"class_method=([A-Za-z_][\w.]*#[A-Za-z_]\w*)"
)
_METHOD_EQ_IN_TEXT = re.compile(r"(?:^|\s)method=([A-Za-z_]\w*)")


def _vector_hit_to_cm(hit: dict[str, Any]) -> str:
    """Map a Milvus vector hit to Class#method (no SQL/literal search)."""
    if not isinstance(hit, dict):
        return ""
    text = str(hit.get("text") or "")
    m = _CLASS_METHOD_IN_TEXT.search(text)
    if m:
        return _simple_cm(m.group(1))
    cm = _hit_to_cm(
        {
            "fqcn": hit.get("fqcn"),
            "summary": text,
            "symbol_id": hit.get("symbol_id"),
            "tags": [t for t in str(hit.get("tags") or "").split(",") if t],
        }
    )
    if cm:
        return _simple_cm(cm)
    fqcn = str(hit.get("fqcn") or "")
    simple = fqcn.split(".")[-1]
    mm = _METHOD_EQ_IN_TEXT.search(text)
    if simple and mm:
        return f"{simple}#{mm.group(1)}"
    sid = str(hit.get("symbol_id") or "").strip()
    if sid and not sid.startswith(("flow:", "overview:", "status:")):
        try:
            from app.db.mysql import get_session
            from app.models.orm import SymbolORM

            with get_session() as session:
                sym = session.get(SymbolORM, sid)
            if sym and sym.method_name:
                cls = (sym.simple_name or (sym.fqcn or "").split(".")[-1] or "")
                meth = str(sym.method_name).split("(", 1)[0]
                if cls and meth:
                    return f"{cls}#{meth}"
        except Exception:
            logger.debug("vector hit symbol_id lookup failed", exc_info=True)
    return ""


def _vector_search_hit_cms_for_question(
    project_id: str, question: str, *, top_k: int = 12
) -> list[str]:
    """Milvus embedding similarity only — no hybrid SQL / tag / literal LIKE."""
    if not project_id or not (question or "").strip():
        return []
    try:
        from app.core.config import get_settings
        from app.db.milvus_client import get_milvus
        from app.llm.factory import get_embedding_provider

        if not bool(get_settings().search_use_milvus):
            return []
        milvus = get_milvus()
        # Skip in-memory fallback so unit tests do not hit a real embedder.
        if not milvus.verify():
            return []
        q = (question or "").strip()
        query = _dig_in_search_query(q) or q
        vec = get_embedding_provider().embed_one(query)
        hits = milvus.search(vec, project_id=project_id, top_k=max(int(top_k), 8) * 2)
    except Exception:
        logger.debug("vector method search failed", exc_info=True)
        return []
    out: list[str] = []
    seen: set[str] = set()
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        cm = _vector_hit_to_cm(h)
        if not cm or "#" not in cm or cm in seen:
            continue
        seen.add(cm)
        out.append(cm)
        if len(out) >= top_k:
            break
    return out


def _search_hit_cms_for_question(
    project_id: str, question: str, *, top_k: int = 12
) -> list[str]:
    """Hybrid/search_symbols → Class#method (no domain map).

    Runs raw question + dig-in core — a long interface+logic sentence often buries the gate noun.
    """
    if not project_id or not (question or "").strip():
        return []
    out: list[str] = []
    try:
        from app.agent.explore_shared import _method_name_from_hit
        from app.agent.explore_tools import search_symbols

        queries: list[str] = []
        core = _dig_in_search_query(question)
        if core:
            toks = core.split()
            # Narrower gate phrase first (later tokens before the full interface+gate phrase).
            if len(toks) >= 2:
                queries.append(" ".join(toks[1:]))
            queries.append(core)
        if question.strip() and question.strip() not in queries:
            queries.append(question.strip())
        for q in queries:
            sr = search_symbols(project_id, q, top_k=top_k)
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "")
                mname = _method_name_from_hit(h)
                simple = fqcn.split(".")[-1]
                if simple and mname:
                    out.append(f"{simple}#{mname}")
    except Exception:
        logger.debug("ask search_symbols enrich failed", exc_info=True)
    return list(dict.fromkeys(_simple_cm(s) for s in out if s))


def _calls_closure_cms(project_id: str, entry_cm: str) -> list[str]:
    try:
        from app.agent.explore_agent import _collect_entry_calls_closure

        raw = _collect_entry_calls_closure(
            project_id, entry_cm, depth=4, limit=160
        )
        return [_simple_cm(s) for s in raw if s and "#" in str(s)]
    except Exception:
        logger.debug("ask CALLS closure failed", exc_info=True)
        return []


def _merge_intersect_nodes(
    nodes: list[dict[str, Any]],
    intersect_cms: list[str],
    *,
    limit: int = 12,
) -> list[dict[str, Any]]:
    """Put CALLS∩search symbols right after the entry node (dig-in first)."""
    if not intersect_cms:
        return list(nodes)[:limit]
    head: list[dict[str, Any]] = []
    tail: list[dict[str, Any]] = []
    if nodes:
        head = [dict(nodes[0])]
        tail = [dict(n) for n in nodes[1:]]
    seen = {_simple_cm(str(n.get("class_method") or "")) for n in head}
    mid: list[dict[str, Any]] = []
    for cm in intersect_cms:
        c = _simple_cm(cm)
        if not c or c in seen:
            continue
        seen.add(c)
        mid.append(
            {
                "order": 0,
                "class_method": c,
                "title": "",
                "rel": "CALLS_INTERSECT",
                "desc": "",
            }
        )
    out = head + mid + [
        n
        for n in tail
        if _simple_cm(str(n.get("class_method") or "")) not in seen
    ]
    for i, n in enumerate(out[:limit], start=1):
        n["order"] = i
    return out[:limit]


def _mysql_evidence_row_count(mysql_rows: list[dict[str, Any]]) -> int:
    """Rows that carry usable summary/desc/code — not empty shells."""
    n = 0
    for r in mysql_rows or []:
        if (
            (r.get("summary") or "").strip()
            or (r.get("neo4j_desc") or "").strip()
            or (r.get("title") or "").strip()
            or (r.get("code_snippet") or "").strip()
            or (r.get("flow_doc_stages") or [])
        ):
            n += 1
    return n


def _count_answer_steps(answer: str) -> int:
    return len(re.findall(r"(?m)^\s*\d+\.\s+", answer or ""))


def _refuse_if_thin_evidence(
    *,
    preferred_entry: str,
    nodes: list[dict[str, Any]],
    intersect_cms: list[str],
    mysql_rows: list[dict[str, Any]],
) -> str | None:
    """Deterministic refuse (2A). None means OK to summarize."""
    if not preferred_entry or _is_placeholder_entry(preferred_entry):
        return _REFUSE_NO_ENTRY
    graph_thin = len(nodes) <= 1
    if graph_thin and not intersect_cms:
        return _REFUSE_THIN_EVIDENCE
    if _mysql_evidence_row_count(mysql_rows) < 2 and (
        graph_thin or len(mysql_rows) <= 1
    ):
        return _REFUSE_THIN_EVIDENCE
    return None


def _postcheck_summarize(
    answer: str, *, mysql_rows: list[dict[str, Any]], allow_tech: bool
) -> str:
    """Replace hallucinated multi-step answers when evidence rows are thin."""
    text = (answer or "").strip()
    if not text:
        return _REFUSE_THIN_EVIDENCE
    ev = _mysql_evidence_row_count(mysql_rows)
    steps = _count_answer_steps(text)
    if steps >= 3 and ev < 2:
        return _REFUSE_THIN_EVIDENCE
    if not allow_tech:
        text = _strip_tech_names(text)
    return text


async def run_ask_retrieval_pipeline(
    question: str,
    *,
    project_id: str,
    entry_hint: str | None = None,
    prior_question: str | None = None,
    history_text: str | None = None,
    on_progress: Any | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Milvus → LLM Neo4j plan → execute template → MySQL → LLM summarize.

    ``lock_entry`` / other legacy kwargs are ignored (kept for call-site compat).
    """
    steps: list[dict[str, Any]] = []
    retrieved: list[dict[str, Any]] = []
    user_q = (question or "").strip()
    if not project_id or not user_q:
        return {
            "answer": "",
            "steps": steps,
            "retrieved": retrieved,
            "plan": None,
            "neo4j_query": None,
            "mysql_rows": [],
        }

    plan = {
        "kind": "ask_retrieval",
        "title": "Ask 检索链路",
        "steps": [
            {"id": "milvus_search", "title": "Milvus 检索", "status": "pending"},
            {"id": "neo4j_plan", "title": "LLM 生成 Neo4j 查询", "status": "pending"},
            {"id": "mysql_enrich", "title": "按序查询 MySQL", "status": "pending"},
            {"id": "ask_summarize", "title": "LLM 汇总回答", "status": "pending"},
        ],
    }
    await _emit(on_progress, {"type": "plan", "plan": plan})

    # ---- 1. Milvus ----
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "milvus_search",
            "status": "running",
            "note": "Milvus 语义检索…",
            "plan": plan,
        },
    )
    hits: list[Any] = []
    try:
        from app.tagging.search import hybrid_search

        hits = hybrid_search(user_q, project_id, top_k=12)
        # If follow-up is short, also search with prior topic concatenated (generic).
        prior = (prior_question or "").strip()
        if prior and prior != user_q and len(user_q) <= 40:
            extra = hybrid_search(f"{prior} {user_q}", project_id, top_k=8)
            seen = {
                getattr(h, "symbol_id", None)
                or (h.get("symbol_id") if isinstance(h, dict) else None)
                for h in hits
            }
            for h in extra:
                sid = getattr(h, "symbol_id", None) or (
                    h.get("symbol_id") if isinstance(h, dict) else None
                )
                if sid and sid in seen:
                    continue
                hits.append(h)
                if sid:
                    seen.add(sid)
    except Exception:
        logger.debug("hybrid_search failed", exc_info=True)
        hits = []

    milvus_brief = _milvus_brief(hits)
    route_q = (prior_question or user_q).strip()
    entry_routes = _entry_route_candidates(project_id, route_q, top_k=8)
    hit_cms = [str(b.get("cm")) for b in milvus_brief if b.get("cm")]
    # Milvus often returns Controller helpers only — synthesize XxxController#xxx.
    for facade in synthesize_facades_from_hits(hit_cms):
        entry_routes.append(
            {
                "cm": facade,
                "kind": "Controller",
                "flow_role": "main",
                "http_paths": "",
                "api_doc": "",
                "score": 1.0,
                "is_system_entry": True,
                "is_http_facade": True,
                "is_helper": False,
                "source": "synthesized_facade",
            }
        )
    preferred_entry = prefer_flow_entry(
        hit_cms,
        entry_routes=entry_routes,
        milvus_brief=milvus_brief,
        entry_hint=entry_hint,
        question=user_q,
        project_id=project_id,
    )
    if not preferred_entry:
        facades = synthesize_facades_from_hits(hit_cms)
        preferred_entry = facades[0] if facades else ""
    for b in milvus_brief:
        if not b.get("symbol_id"):
            continue
        retrieved.append(
            {
                "symbol_id": b.get("symbol_id"),
                "project_id": project_id,
                "fqcn": b.get("fqcn") or "",
                "kind": b.get("kind") or "Class",
                "file_path": "",
                "summary": b.get("summary"),
                "tags": [],
                "score": b.get("score"),
            }
        )
    steps.append(
        {
            "step": 1,
            "tool": "milvus_search",
            "ok": bool(hits) or bool(entry_routes),
            "hit_count": len(hits),
            "candidates": [b.get("cm") for b in milvus_brief if b.get("cm")][:10],
            "entry_routes": [h.get("cm") for h in entry_routes][:6],
            "preferred_entry": preferred_entry,
        }
    )
    plan["steps"][0]["status"] = "done"
    plan["steps"][0]["note"] = (
        f"命中 {len(hits)}"
        + (f"；建议入口 {preferred_entry}" if preferred_entry else "")
    )
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "milvus_search",
            "status": "done",
            "note": plan["steps"][0]["note"],
            "plan": plan,
        },
    )

    if not hits and not entry_hint and not entry_routes:
        return {
            "answer": _REFUSE_NO_ENTRY,
            "steps": steps,
            "retrieved": retrieved,
            "plan": plan,
            "neo4j_query": None,
            "mysql_rows": [],
        }

    if not preferred_entry:
        plan["steps"][1]["status"] = "skipped"
        plan["steps"][2]["status"] = "skipped"
        plan["steps"][3]["status"] = "done"
        plan["steps"][3]["note"] = "无系统入口，拒答"
        return {
            "answer": _REFUSE_NO_ENTRY,
            "steps": steps,
            "retrieved": retrieved,
            "plan": plan,
            "neo4j_query": None,
            "mysql_rows": [],
        }

    # ---- 2. LLM → Neo4j ----
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "neo4j_plan",
            "status": "running",
            "note": "结合上下文让 LLM 生成 Neo4j 查询…",
            "plan": plan,
        },
    )
    choice = await plan_neo4j_with_llm(
        user_q,
        milvus_brief=milvus_brief,
        history_text=history_text,
        prior_question=prior_question,
        entry_hint=entry_hint or preferred_entry,
        entry_routes=entry_routes,
        project_id=project_id,
    )
    nodes, neo_meta = _run_neo4j_template(project_id, choice)
    entry_anchor = str((choice.get("params") or {}).get("entry") or preferred_entry)

    # CALLS ∩ search: dig-in evidence under locked entry (no domain map).
    calls_cms = _calls_closure_cms(project_id, entry_anchor)
    search_cms = _search_hit_cms_for_question(project_id, user_q, top_k=12)
    calls_set = set(calls_cms)
    intersect_cms = [c for c in search_cms if c in calls_set]
    # Also keep search hits that share the entry's simple class (same service)
    if not intersect_cms and calls_cms:
        # Prefer search hits that appear in CALLS; already empty — leave empty
        pass
    nodes = _merge_intersect_nodes(nodes, intersect_cms, limit=16)

    neo4j_query = {
        **choice,
        "exec": {
            k: neo_meta.get(k)
            for k in ("ok", "raw_source", "error", "raw_nodes", "raw_edges")
        },
        "calls_count": len(calls_cms),
        "intersect_cms": intersect_cms[:12],
    }
    steps.append(
        {
            "step": 2,
            "tool": "neo4j_plan",
            "ok": bool(nodes),
            "template_id": choice["template_id"],
            "params": choice["params"],
            "reason": choice.get("reason"),
            "node_count": len(nodes),
            "entry_anchor": entry_anchor,
            "calls_count": len(calls_cms),
            "intersect_count": len(intersect_cms),
        }
    )
    plan["steps"][1]["status"] = "done"
    plan["steps"][1]["note"] = (
        f"{choice['template_id']} @ {entry_anchor} → {len(nodes)} 节点"
        f"；CALLS∩检索={len(intersect_cms)}"
    )
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "neo4j_plan",
            "status": "done",
            "note": plan["steps"][1]["note"],
            "plan": plan,
        },
    )

    # ---- 3. MySQL ----
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "mysql_enrich",
            "status": "running",
            "note": "按 Neo4j/交集顺序查询 MySQL…",
            "plan": plan,
        },
    )
    mysql_rows = _mysql_enrich_ordered(project_id, nodes)
    steps.append(
        {
            "step": 3,
            "tool": "mysql_enrich",
            "ok": bool(mysql_rows),
            "row_count": len(mysql_rows),
            "evidence_rows": _mysql_evidence_row_count(mysql_rows),
            "symbols": [
                r.get("class_method")
                for r in mysql_rows
                if r.get("class_method")
            ][:8],
            "intersect_cms": intersect_cms[:8],
        }
    )
    plan["steps"][2]["status"] = "done"
    plan["steps"][2]["note"] = (
        f"MySQL 补齐 {len(mysql_rows)} 条"
        f"（有效摘要 {_mysql_evidence_row_count(mysql_rows)}）"
    )
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "mysql_enrich",
            "status": "done",
            "note": plan["steps"][2]["note"],
            "plan": plan,
        },
    )

    # ---- 4. Summarize or refuse (2A) ----
    refuse = _refuse_if_thin_evidence(
        preferred_entry=entry_anchor or preferred_entry,
        nodes=nodes,
        intersect_cms=intersect_cms,
        mysql_rows=mysql_rows,
    )
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "ask_summarize",
            "status": "running",
            "note": "证据不足，拒答…" if refuse else "LLM 汇总回答…",
            "plan": plan,
        },
    )
    if refuse:
        answer = refuse
    else:
        answer = await _summarize_answer(
            user_q,
            neo4j_query=neo4j_query,
            mysql_rows=mysql_rows,
            milvus_brief=milvus_brief,
            history_text=history_text,
            focus_rels={"CALLS_INTERSECT"},
        )
        answer = _postcheck_summarize(
            answer,
            mysql_rows=mysql_rows,
            allow_tech=_wants_tech_names(user_q),
        )
    steps.append(
        {
            "step": 4,
            "tool": "ask_summarize",
            "ok": bool((answer or "").strip()),
            "entry_anchor": entry_anchor,
            "refused": bool(refuse),
        }
    )
    plan["steps"][3]["status"] = "done"
    plan["steps"][3]["note"] = "证据不足拒答" if refuse else "已生成回答"
    await _emit(
        on_progress,
        {
            "type": "step",
            "step_id": "ask_summarize",
            "status": "done",
            "note": plan["steps"][3]["note"],
            "plan": plan,
        },
    )

    return {
        "answer": answer or "",
        "steps": steps,
        "retrieved": retrieved,
        "plan": plan,
        "neo4j_query": neo4j_query,
        "mysql_rows": mysql_rows,
    }


# Exported for unit tests
normalize_template_choice = _normalize_template_choice
