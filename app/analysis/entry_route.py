"""Entry routing / classification Milvus collection.

Offline: build `entry_route` rows from MySQL symbol_desc entries (Controllers /
Jobs / Listeners) with a deterministic flow_role and routing_text embedding.

Ask-time: search this collection first for「xx流程」questions → Class#method
anchor → then load FlowDoc / node materials for that entry only.

Rebuild (stop API first if Milvus Lite is locked)::

    python tests/rebuild_entry_route.py
    python tests/rebuild_entry_route.py --project-id demo-loan --recreate
"""

from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.core.utils import truncate
from app.db.milvus_client import get_entry_route_store, get_milvus
from app.db.mysql import get_session
from app.llm.factory import get_embedding_provider
from app.models.orm import SymbolDescORM, SymbolORM

logger = logging.getLogger("app.analysis.entry_route")

# Naming-only role signals — no industry/domain nouns.
_QUERYISH_METHOD = re.compile(
    r"(?i)^(query|get|list|find|select|search|count|sync|check|load|fetch|page)"
)
_CALLBACK_MARK = re.compile(r"(?i)callback|call_?back|回调")
_NOTICE_MARK = re.compile(r"(?i)notice|notify|通知")
_LISTENER_MARK = re.compile(r"(?i)listener|listenner|consumer|mq入口")
_JOB_MARK = re.compile(r"(?i)(^|[^a-z])job([^a-z]|$)|xxl|scheduled|job入口")


def classify_flow_role(
    *,
    kind: str = "",
    class_method: str = "",
    method_name: str = "",
    fqcn: str = "",
    api_doc: str = "",
    summary: str = "",
    entry_role: str = "",
    existing_role: str = "",
) -> str:
    """Deterministic flow_role: main | query | notice | callback | job | async."""
    role_hint = (existing_role or "").strip().lower()
    identity = f"{class_method} {fqcn}"
    # Offline role=main is often wrong for Notice*/Notify* Controllers — name wins.
    if role_hint == "main" and (
        _NOTICE_MARK.search(class_method or "") or _NOTICE_MARK.search(fqcn or "")
    ):
        role_hint = ""
    if role_hint in {"main", "query", "notice", "callback", "job", "async"}:
        # Trust explicit offline role only when not contradicted by queryish naming
        # on Controllers (queryRepayBank tagged main by mistake).
        if role_hint == "main" and _QUERYISH_METHOD.match(method_name or ""):
            return "query"
        # Risk* MQ listeners tagged async offline are still result callbacks.
        if role_hint == "async" and _is_risk_result_listener(
            kind=kind, class_method=class_method, fqcn=fqcn, blob=""
        ):
            return "callback"
        if role_hint:
            return role_hint

    kind_l = (kind or "").lower()
    blob = f"{class_method} {fqcn} {method_name} {api_doc} {summary} {entry_role}"

    if kind_l == "job" or _JOB_MARK.search(blob) or "job入口" in (entry_role or "").lower():
        return "job"
    if _CALLBACK_MARK.search(blob):
        return "callback"
    if _NOTICE_MARK.search(class_method) or _NOTICE_MARK.search(fqcn) or (
        _NOTICE_MARK.search(blob) and kind_l in {"controller", "listener", ""}
    ):
        # Prefer notice over generic listener when Notice* naming is present.
        if kind_l == "listener" and not _NOTICE_MARK.search(identity):
            pass
        else:
            return "notice"
    if kind_l == "listener" or _LISTENER_MARK.search(fqcn) or _LISTENER_MARK.search(
        class_method
    ):
        # Risk* result consumers are callback entries (MQ notice into the system).
        if _is_risk_result_listener(
            kind=kind, class_method=class_method, fqcn=fqcn, blob=blob
        ):
            return "callback"
        return "async"
    if "mq入口" in (entry_role or "").lower():
        return "async"
    if _QUERYISH_METHOD.match(method_name or ""):
        return "query"
    return "main"


def _is_risk_result_listener(
    *,
    kind: str = "",
    class_method: str = "",
    fqcn: str = "",
    blob: str = "",
) -> bool:
    """True for Risk* Listener / MQ consumer entries (class-stem, not domain nouns)."""
    kind_l = (kind or "").lower()
    identity = f"{class_method} {fqcn}"
    if not re.search(r"(?i)risk", identity):
        return False
    if kind_l == "listener" or _LISTENER_MARK.search(identity):
        return True
    # FQCN/class already Risk* + consumer naming in blob.
    return bool(_LISTENER_MARK.search(blob or identity))


def build_routing_text(
    *,
    api_doc: str = "",
    class_method: str = "",
    http_paths: list[str] | str | None = None,
    flow_role: str = "main",
    kind: str = "",
    summary: str = "",
    method_name: str = "",
    fqcn: str = "",
    labels: str = "",
) -> str:
    """Compact text for embedding — what natural-language questions should hit."""
    if isinstance(http_paths, list):
        paths = " ".join(str(p) for p in http_paths if p)
    else:
        paths = str(http_paths or "")
    # Brief summary only — drop long stage narratives so routing stays entry-centric.
    brief = truncate((summary or "").replace("\n", " ").strip(), 280)
    parts = [
        api_doc.strip() if api_doc else "",
        class_method,
        method_name,
        fqcn.split(".")[-1] if fqcn else "",
        f"path={paths}" if paths else "",
        f"role={flow_role}",
        kind,
        labels,
        brief,
    ]
    return truncate(" ".join(p for p in parts if p), 1200)


def _is_entry_desc(
    extra: dict[str, Any],
    domain: str | None,
    summary: str,
    *,
    kind: str = "",
    class_method: str = "",
) -> bool:
    if extra.get("entry") or extra.get("data_source") or extra.get("is_entry"):
        return True
    if domain == "数据源头":
        return True
    if "数据源头" in (summary or "") or "HTTP入口" in (summary or ""):
        return True
    paths = extra.get("http_paths") or []
    if isinstance(paths, list) and paths and extra.get("class_method"):
        return True
    # topo-priority LLM summaries often omit is_entry / http_paths; still index
    # real system entry kinds so entry_route rebuild is not empty after full refresh.
    cm = (class_method or str(extra.get("class_method") or "")).strip()
    if not cm or "#" not in cm:
        return False
    kind_l = (kind or str(extra.get("kind") or "")).lower()
    simple = cm.split("#", 1)[0]
    if kind_l in {"controller", "job", "listener", "entry", "consumer"}:
        return True
    if (
        simple.endswith("Controller")
        or simple.endswith("Job")
        or "Listener" in simple
        or simple.endswith("Consumer")
    ):
        return True
    return False


def collect_entry_route_rows(project_id: str) -> list[dict[str, Any]]:
    """Build entry_route row dicts (no embeddings yet) from MySQL symbol_desc."""
    rows: list[dict[str, Any]] = []
    with get_session() as session:
        descs = session.scalars(
            select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
        ).all()
        for d in descs:
            extra = dict(d.extra_json or {})
            summary = d.summary or ""
            # Skip pseudo FlowDoc / overview rows — routing is about real entries.
            sid = d.symbol_id or ""
            if sid.startswith(("flow:", "overview:", "status:")):
                continue
            sym = session.get(SymbolORM, sid)
            fqcn = (sym.fqcn if sym else "") or str(extra.get("fqcn") or "")
            if fqcn.startswith("__"):
                continue
            method_name = (sym.method_name if sym else None) or str(
                extra.get("method_name") or ""
            )
            if not method_name and "#" in str(extra.get("class_method") or ""):
                method_name = str(extra["class_method"]).split("#", 1)[1]
            simple = (sym.simple_name if sym else "") or (
                fqcn.split(".")[-1] if fqcn else ""
            )
            class_method = str(extra.get("class_method") or "")
            if not class_method and simple and method_name:
                class_method = f"{simple}#{method_name}"
            if not class_method or "#" not in class_method:
                continue

            kind = (sym.kind if sym else "") or str(extra.get("kind") or "")
            if not kind:
                sn = simple.lower()
                if "controller" in sn:
                    kind = "Controller"
                elif "listener" in sn or "listenner" in sn:
                    kind = "Listener"
                elif sn.endswith("job"):
                    kind = "Job"
                else:
                    kind = str(extra.get("entry_role") or "Entry")

            if not _is_entry_desc(
                extra,
                d.business_domain,
                summary,
                kind=kind,
                class_method=class_method,
            ):
                continue

            http_paths = extra.get("http_paths") or []
            if not isinstance(http_paths, list):
                http_paths = [str(http_paths)] if http_paths else []
            path_str = ",".join(str(p) for p in http_paths if p)[:1000]
            api_doc = str(extra.get("api_doc") or "").strip()
            if not api_doc:
                # Discover/FlowDoc often writes「xx入口」; lift that as routing label.
                m = re.search(r"([\u4e00-\u9fff]{2,12})入口", summary)
                if m:
                    api_doc = m.group(1)
            # topo summaries: lift short business_domain as api_doc when Chinese.
            if not api_doc and d.business_domain and re.search(
                r"[\u4e00-\u9fff]", d.business_domain or ""
            ):
                api_doc = str(d.business_domain).strip()[:40]
            entry_role = str(extra.get("entry_role") or "")
            roles = extra.get("flow_roles") or []
            existing_role = ""
            if isinstance(roles, list) and roles:
                existing_role = str(roles[0])
            fd = extra.get("flow_doc")
            if isinstance(fd, dict) and fd.get("flow_role"):
                existing_role = str(fd.get("flow_role")) or existing_role

            flow_role = classify_flow_role(
                kind=kind,
                class_method=class_method,
                method_name=method_name,
                fqcn=fqcn,
                api_doc=api_doc,
                summary=summary,
                entry_role=entry_role,
                existing_role=existing_role,
            )
            labels = ",".join(
                x
                for x in [
                    kind,
                    flow_role,
                    entry_role,
                    "entry",
                    "数据源头" if d.business_domain == "数据源头" else "",
                ]
                if x
            )[:500]
            routing_text = build_routing_text(
                api_doc=api_doc,
                class_method=class_method,
                http_paths=http_paths,
                flow_role=flow_role,
                kind=kind,
                summary=summary,
                method_name=method_name,
                fqcn=fqcn,
                labels=labels,
            )
            rows.append(
                {
                    "id": sid[:64],
                    "project_id": project_id,
                    "symbol_id": sid[:64],
                    "kind": kind[:64],
                    "class_method": class_method[:256],
                    "fqcn": fqcn[:500],
                    "method_name": method_name[:256],
                    "http_paths": path_str,
                    "mq_topic": str(extra.get("mq_topic") or "")[:500],
                    "flow_role": flow_role[:32],
                    "api_doc": api_doc[:500],
                    "labels": labels,
                    "routing_text": routing_text,
                }
            )
    # Dedupe by class_method — prefer richer routing signal (api_doc / paths / summary).
    by_cm: dict[str, dict[str, Any]] = {}
    for r in rows:
        cm = r["class_method"]
        prev = by_cm.get(cm)
        if prev is None:
            by_cm[cm] = r
            continue

        def _rich(x: dict[str, Any]) -> int:
            return (
                (4 if x.get("api_doc") else 0)
                + (3 if x.get("http_paths") else 0)
                + min(len(str(x.get("routing_text") or "")), 400) // 40
            )

        if _rich(r) >= _rich(prev):
            by_cm[cm] = r
    return list(by_cm.values())


def upsert_entry_route_rows(rows: list[dict[str, Any]]) -> int:
    """Embed routing_text and upsert into entry_route collection."""
    if not rows:
        return 0
    store = get_entry_route_store()
    embedder = get_embedding_provider()
    texts = [str(r.get("routing_text") or " ").strip() or " " for r in rows]
    embeddings: list[list[float]] = []
    batch_size = 8
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            embeddings.extend(embedder.embed([t[:1200] for t in chunk]))
        except Exception:
            logger.exception("entry_route embed batch failed at %s; retry one-by-one", i)
            for t in chunk:
                try:
                    embeddings.extend(embedder.embed([t[:800]]))
                except Exception:
                    logger.exception("entry_route embed skip")
                    embeddings.append([0.0] * (store.dim or 1024))
    if len(embeddings) != len(rows):
        logger.error(
            "entry_route embed count mismatch %s vs %s", len(embeddings), len(rows)
        )
        return 0
    store.dim = len(embeddings[0]) if embeddings else store.dim
    store.upsert(rows, embeddings)
    return len(rows)


def rebuild_entry_route(
    project_id: str | None = None,
    *,
    recreate: bool = False,
) -> dict[str, Any]:
    """Rebuild entry_route for a project. Returns {count, collection, roles}."""
    settings = get_settings()
    pid = (project_id or settings.project_id).strip()
    # Ensure primary milvus client is up (shares Lite lock / URI).
    milvus = get_milvus()
    if not milvus.verify():
        raise RuntimeError(
            "Milvus unavailable (Lite lock?). Stop the API process and retry."
        )
    store = get_entry_route_store()
    if recreate:
        store.recreate_collection(milvus.dim or settings.embedding_dim)
    else:
        store.clear_project(pid)
    rows = collect_entry_route_rows(pid)
    n = upsert_entry_route_rows(rows)
    role_counts: dict[str, int] = {}
    for r in rows:
        role_counts[r["flow_role"]] = role_counts.get(r["flow_role"], 0) + 1
    logger.info(
        "entry_route rebuilt project=%s rows=%s roles=%s collection=%s",
        pid,
        n,
        role_counts,
        store.collection_name,
    )
    return {
        "project_id": pid,
        "count": n,
        "collection": store.collection_name,
        "roles": role_counts,
        "sample": [
            {
                "class_method": r["class_method"],
                "flow_role": r["flow_role"],
                "api_doc": r.get("api_doc") or "",
                "http_paths": r.get("http_paths") or "",
            }
            for r in rows[:12]
        ],
    }


# Atomic Chinese nouns for structural role splitting only (回调/通知/监听…).
# No product-domain nouns — entry/topic pick is LLM + entry_route embeddings.
_ATOMIC_CN_NOUNS: tuple[str, ...] = (
    "回调",
    "通知",
    "监听",
)

# CN → EN identity stems intentionally empty: product entry pick is LLM / weak
# text overlap on api_doc & Class#method, not hardcoded business glossaries.
_CN_IDENTITY_STEMS: dict[str, tuple[str, ...]] = {}


def _question_wants_role(question: str) -> set[str]:
    """Which flow_roles the user is asking for (else prefer main)."""
    q = question or ""
    wanted: set[str] = set()
    if re.search(r"查询|怎么查|如何查|查一下|列表|详情", q):
        wanted.add("query")
    if re.search(r"通知|notice|notify", q, re.I):
        wanted.add("notice")
    if re.search(r"回调|callback", q, re.I):
        wanted.add("callback")
    if re.search(r"定时|调度|job|xxl", q, re.I):
        wanted.add("job")
    if re.search(r"监听|listener|mq|消息|异步", q, re.I):
        wanted.add("async")
    return wanted


# 「某某通知接口」→ 某某通知. Structural glue only — no product noun map.
_INTERFACE_ASK_RE = re.compile(
    r"([\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9]{1,23}?)"
    r"(?:接口|流程|入口)"
)
_LABEL_GAP_GLUE_CHARS = set("的了吗呢是在与及和或后前中对将把被从到向于则即已未可会能要等之其此该各每")
_LABEL_GAP_GLUE_WORDS = (
    "同步",
    "结果",
    "处理",
    "系统",
    "信息",
    "业务",
    "数据",
    "相关",
    "进行",
    "调用",
    "返回",
    "完成",
)


def interface_ask_phrase(question: str) -> str:
    """Core label before 接口/流程/入口 (e.g. 某某通知接口 → 某某通知)."""
    q = (question or "").strip()
    if not q:
        return ""
    m = _INTERFACE_ASK_RE.search(q)
    return (m.group(1).strip() if m else "") or ""


def _min_subsequence_window(phrase: str, text: str) -> str | None:
    """Shortest substring of text that contains phrase as an ordered subsequence."""
    if not phrase or not text:
        return None
    n = len(text)
    m = len(phrase)
    best: str | None = None
    for start in range(n):
        if text[start] != phrase[0]:
            continue
        j = 0
        for i in range(start, n):
            if text[i] == phrase[j]:
                j += 1
                if j == m:
                    span = text[start : i + 1]
                    if best is None or len(span) < len(best):
                        best = span
                    break
    return best


def _foreign_gap_penalty(gap: str, question: str) -> int:
    """Penalize foreign mid-inserts between phrase endpoints (not in the ask)."""
    g = gap or ""
    for w in _LABEL_GAP_GLUE_WORDS:
        g = g.replace(w, "")
    g = "".join(ch for ch in g if ch not in _LABEL_GAP_GLUE_CHARS)
    leftover = "".join(ch for ch in g if ("\u4e00" <= ch <= "\u9fff") and ch not in (question or ""))
    return 35 * len(leftover)


def _fuzzy_phrase_score(phrase: str, text: str, question: str, *, base: int) -> int:
    window = _min_subsequence_window(phrase, text)
    if not window:
        return 0
    gap_len = max(0, len(window) - len(phrase))
    # Drop matched phrase chars from window to isolate inserts.
    gap_chars: list[str] = []
    j = 0
    for ch in window:
        if j < len(phrase) and ch == phrase[j]:
            j += 1
        else:
            gap_chars.append(ch)
    gap = "".join(gap_chars)
    prox = 40 - 5 * gap_len
    return max(0, base + prox - _foreign_gap_penalty(gap, question))


def interface_label_score(
    label: str, question: str, *, extra: str = ""
) -> int:
    """Higher when Swagger/api_doc (or summary extra) matches an 接口问句 label.

    Sibling Controllers sharing ``#method`` (e.g. both ``#creditNotification``)
    are separated without a product lexicon: the short ask label beats a longer
    mid-insert sibling; summary glue gaps stay acceptable.
    """
    api = (label or "").strip()
    q = (question or "").strip()
    if not q:
        return 0
    blob = f"{api} {(extra or '').strip()}".strip()

    if api and len(api) >= 2:
        if api == q:
            return 200 + len(api)
        if api in q:
            return 100 + len(api)
        if q in api and len(q) >= 4:
            return 60 + len(q)

    phrase = interface_ask_phrase(q)
    if not phrase or len(phrase) < 2:
        return 0

    if phrase in api:
        return 100 + len(phrase)
    if api and len(api) >= 4 and api in phrase:
        return 80 + len(api)
    if phrase in blob:
        return 90 + len(phrase)

    best = 0
    if api:
        best = max(best, _fuzzy_phrase_score(phrase, api, q, base=80))
    if extra and extra.strip():
        # Prefer the Chinese head of routing/summary — avoid long English noise.
        head = truncate(
            "".join(
                ch
                for ch in (extra or "")
                if "\u4e00" <= ch <= "\u9fff" or ch in "，。、：；"
            ),
            160,
        )
        if head:
            best = max(best, _fuzzy_phrase_score(phrase, head, q, base=70))
        elif blob != api:
            best = max(best, _fuzzy_phrase_score(phrase, blob, q, base=65))
    return best


def _expand_atomic_cn_tokens(toks: list[str]) -> list[str]:
    """Split compounds so「xxx回调」also yields structural atom 回调."""
    out: list[str] = []
    seen: set[str] = set()
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
        if not re.search(r"[\u4e00-\u9fff]", t):
            continue
        for atom in _ATOMIC_CN_NOUNS:
            if atom != t and atom in t and atom not in seen:
                seen.add(atom)
                out.append(atom)
    return out


def _question_content_tokens(question: str) -> list[str]:
    """Content tokens from a flow question (strip glue words)."""
    text = question or ""
    for stop in (
        "流程",
        "什么样",
        "怎么走",
        "怎样",
        "怎么",
        "如何",
        "介绍",
        "一下",
        "完整",
        "全链路",
        "端到端",
        "从入口到",
        "调用链",
        "链路",
        "串一下",
        "的",
        "是",
        "了",
        "吗",
        "呢",
    ):
        text = text.replace(stop, " ")
    text = re.sub(r"[？?！!。．，,、：:；;…\s]+", " ", text).strip()
    toks: list[str] = []
    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}|[A-Za-z][A-Za-z0-9_]{2,}", text):
        t = m.group(0)
        if t not in toks:
            toks.append(t)
        # Also keep 2-char slices of longer CJK compounds (user-written atoms).
        if re.fullmatch(r"[\u4e00-\u9fff]{4,}", t):
            for i in range(len(t) - 1):
                bit = t[i : i + 2]
                if bit not in toks:
                    toks.append(bit)
    return _expand_atomic_cn_tokens(toks)


def _identity_blob(hit: dict[str, Any]) -> str:
    """Class#method / api_doc / paths only — never stage narrative."""
    return " ".join(
        str(hit.get(k) or "")
        for k in ("class_method", "fqcn", "method_name", "api_doc", "http_paths")
    )


def _token_hard_identity_hit(tok: str, identity: str, api_doc: str = "") -> bool:
    """True when tok hard-matches entry identity (api_doc / class text), not narrative.

    No CN→EN business stem map — product entry pick is LLM / weak overlap downstream.
    """
    tl = (tok or "").strip().lower()
    if not tl:
        return False
    id_l = (identity or "").lower()
    api_l = (api_doc or "").strip().lower()
    if api_l and (tl == api_l or tl in api_l or api_l in tl):
        return True
    if tl in id_l:
        return True
    return False


def rerank_entry_routes_for_question(
    hits: list[dict[str, Any]], question: str
) -> list[dict[str, Any]]:
    """Demote query/notice/callback/async unless the question asks for them."""
    if not hits:
        return hits
    wanted = _question_wants_role(question)
    # Sync main-chain asks: 「流程」plus cold-start「接口/逻辑/那一块…」
    # (role demotion only applies when wanted roles are empty — notice/回调仍走 wanted 分支)
    sync_flow = bool(
        "流程" in (question or "")
        or "过程中" in (question or "")
        or re.search(
            r"从入口|主链|完整|全链路|端到端|串一下|调用了哪些|调了哪些|"
            r"接口|逻辑|那一块|那块|怎么做|做了啥|什么样|怎么走",
            question or "",
        )
    )
    q_toks = _question_content_tokens(question)
    cn_toks = [t for t in q_toks if re.search(r"[\u4e00-\u9fff]", t)]
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for i, h in enumerate(hits):
        base = float(h.get("score") or 0.0)
        role = str(h.get("flow_role") or "main").lower()
        boost = 0.0
        if sync_flow and not wanted:
            if role == "main":
                boost += 8.0
            elif role == "query":
                boost -= 20.0
            elif role in {"notice", "callback"}:
                boost -= 18.0
            elif role in {"async", "job"}:
                boost -= 10.0
        elif wanted:
            if role in wanted:
                boost += 10.0
                if role in {"notice", "callback"} and wanted & {"notice", "callback"}:
                    boost += 12.0
            elif role == "main" and not wanted & {"query", "notice", "callback"}:
                boost += 2.0
            else:
                boost -= 18.0 if wanted & {"notice", "callback"} else 8.0

        api = str(h.get("api_doc") or "").strip()
        paths = str(h.get("http_paths") or "")
        routing = str(h.get("routing_text") or "")
        identity = _identity_blob(h)
        id_l = identity.lower()
        soft_blob = f"{routing}".lower()
        hard = 0
        # Interface-label affinity (sibling #method titles) — stronger than raw
        # token overlap which otherwise prefers a longer mid-insert sibling title.
        label_sc = interface_label_score(api, question or "", extra=routing)
        if label_sc >= 100:
            hard += 1
            boost += 28.0
        elif label_sc >= 70:
            hard += 1
            boost += 18.0
        elif label_sc >= 40:
            hard += 1
            boost += 8.0
        elif label_sc > 0:
            boost += 3.0
        # Sibling titles with foreign mid-inserts still get raw token hits —
        # cut that noise when the interface phrase only matches api_doc
        # through a non-glue gap.
        phrase = interface_ask_phrase(question or "")
        if phrase and api and label_sc < 70:
            win = _min_subsequence_window(phrase, api)
            if win:
                gap_chars: list[str] = []
                j = 0
                for ch in win:
                    if j < len(phrase) and ch == phrase[j]:
                        j += 1
                    else:
                        gap_chars.append(ch)
                if _foreign_gap_penalty("".join(gap_chars), question or "") >= 35:
                    boost -= 22.0
        # api_doc phrase ⊂ question (label ⊂ label+接口) — stronger than token equality
        # which often keeps the whole「…接口」compound as one token.
        if api and len(api) >= 4 and api in (question or ""):
            hard += 1
            boost += 24.0
        for tok in q_toks:
            tl = tok.lower()
            if not tl:
                continue
            # Weak overlap only (api_doc / identity / soft routing) — no CN→EN stem boost.
            if api and (tl == api.lower() or api.lower() == tl):
                hard += 1
                boost += 28.0
            elif api and tl in api.lower():
                hard += 1
                boost += 18.0
            elif _token_hard_identity_hit(tok, identity, api):
                hard += 1
                boost += 12.0
            elif tl in soft_blob or tl in id_l:
                hard += 1
                boost += 8.0 if len(tl) >= 2 else 3.0

        # Concrete Chinese Swagger title that does NOT match the question → demote.
        # Skip demotion when interface_label_score already found a fuzzy/summary hit.
        _SUB_OP_PREFIX = re.compile(r"^(累加|查询|新增|更新|删除|修改|缓存|获取|批量)")
        if cn_toks and api and re.search(r"[\u4e00-\u9fff]", api) and label_sc < 40:
            api_hit = any(
                t == api or t in api or api in t for t in cn_toks
            )
            # Sub-operation prefix mismatch: api_doc starts with a sub-op verb
            # (累加/查询/新增/…) but the question token that matched it is merely
            # a business-domain word (放款) rather than the same verb — the match
            # is a false positive (e.g. 累加放款额 vs 放款接口主要做了什么).
            # When this happens, treat it as a miss *even when* the bare domain
            # token appears inside the api_doc string.
            sub_op_m = _SUB_OP_PREFIX.match(api)
            if sub_op_m and not re.search(re.escape(sub_op_m.group(1)), question or ""):
                boost -= 18.0
            elif not api_hit:
                boost -= 16.0

        if sync_flow and role == "main" and paths:
            boost += 3.0

        scored.append((base + boost, -i, h))
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    out: list[dict[str, Any]] = []
    for combined, _, h in scored:
        row = dict(h)
        row["score"] = combined
        out.append(row)
    return out


def search_entry_routes(
    question: str,
    project_id: str,
    *,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Vector-search entry_route then apply role-aware rerank.

    When SEARCH_USE_MILVUS=false, skip vector entry_route entirely (lexical/SQL
    locks in explore_agent handle notice/flow questions instead).
    """
    q = (question or "").strip()
    if not q:
        return []
    if not get_settings().search_use_milvus:
        return []
    try:
        store = get_entry_route_store()
        if not store.verify():
            return []
        embedder = get_embedding_provider()
        emb = embedder.embed([q[:800]])[0]
        hits = store.search(emb, project_id=project_id, top_k=top_k)
        return rerank_entry_routes_for_question(hits, q)
    except Exception:
        logger.exception("search_entry_routes failed")
        return []


def resolve_entry_anchor(
    project_id: str,
    question: str,
    *,
    top_k: int = 8,
) -> dict[str, Any] | None:
    """Best entry Class#method for a flow-style question, or None."""
    hits = search_entry_routes(question, project_id, top_k=top_k)
    if not hits:
        return None
    top = hits[0]
    if not top.get("class_method"):
        return None
    return top
