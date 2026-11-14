from __future__ import annotations

import logging
import re

from sqlalchemy import String, cast, or_, select

from app.db.milvus_client import get_milvus
from app.db.mysql import get_session
from app.llm.factory import get_embedding_provider
from app.models.orm import SymbolDescORM, SymbolORM, SymbolTagORM, TagDictORM
from app.models.schemas import SymbolHit

logger = logging.getLogger("app.search")

# Universal question glue — not industry/business terms.
_QUERY_STOP = {
    "流程",
    "什么样",
    "怎么走",
    "怎样",
    "怎么",
    "如何",
    "介绍",
    "一下",
    "什么",
    "哪些",
    "有哪些",
    "调用链",
    "链路",
    "串一下",
    "说说",
    "讲讲",
    "看下",
    "看一下",
    "相关的",
    "相关",
    "接口",
    "有几个",
    "几个",
    "多少个",
    "多少",
    "一共",
    "总共",
    "列出",
    "全部",
    "所有",
    "的",
    "是",
    "了",
    "吗",
    "呢",
    "啊",
    "吧",
    "请",
    "帮我",
    "我想",
    "了解",
}


def extract_api_path_tokens(q: str) -> list[str]:
    """Extract HTTP path / camelCase API tokens from a question."""
    text = q or ""
    out: list[str] = []
    seen: set[str] = set()

    def _add(tok: str) -> None:
        t = (tok or "").strip().strip("/")
        if len(t) < 4 or not re.search(r"[A-Za-z]", t):
            return
        if t.lower() in {"http", "https", "post", "controller", "service"}:
            return
        if t not in seen:
            seen.add(t)
            out.append(t)

    for m in re.finditer(
        r"(?i)(?:^|[\s`\"'(\\[])(/?[A-Za-z][\w./-]{3,}(?:/[A-Za-z][\w.-]+)*)",
        text,
    ):
        path = m.group(1).strip().strip("`\"'")
        parts = [p for p in path.strip("/").split("/") if p]
        skip = {"api", "v1", "v2", "loan", "demo"}
        for p in parts:
            if p.lower() in skip:
                continue
            _add(p)
    # Bare camelCase / long Latin identifiers (mapping methods — no product prefix list).
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]{5,80})\b", text):
        tok = m.group(1)
        if tok.lower() in {"controller", "service", "https"}:
            continue
        if re.search(r"[A-Z]", tok[1:]) or len(tok) >= 8:
            _add(tok)
    return out[:8]


def extract_query_literals(q: str) -> list[str]:
    """Pull literal search terms from the user question (language-agnostic).

    Prefer longer remaining phrases after stripping question glue words, so
    "某通知流程是什么样的" → ["某通知"] and matches code comments containing
    that exact phrase — without any industry synonym table.
    """
    text = (q or "").strip()
    if not text:
        return []
    for stop in sorted(_QUERY_STOP, key=len, reverse=True):
        text = text.replace(stop, " ")
    text = re.sub(r"[\s，。？?！!、；;：:]+", " ", text).strip()

    terms: list[str] = []
    # HTTP path / API tokens first (highest precision for Controller methods).
    for tok in extract_api_path_tokens(q):
        if tok not in terms:
            terms.append(tok)
    # Whole leftover chunks (CN or identifiers)
    for part in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]{1,}", text):
        if part not in terms:
            terms.append(part)
    # Also keep contiguous Chinese left in text after cleanup as one phrase
    joined_cn = "".join(re.findall(r"[\u4e00-\u9fff]+", text))
    if len(joined_cn) >= 2 and joined_cn not in terms:
        terms.insert(0, joined_cn)

    # Longer first — better for LIKE precision
    terms.sort(key=len, reverse=True)
    return terms[:8]


def _is_pseudo_doc(symbol_id: str | None, fqcn: str | None) -> bool:
    sid = symbol_id or ""
    fq = fqcn or ""
    return (
        sid.startswith("overview:")
        or sid.startswith("flow:")
        or sid.startswith("status:")
        or fq == "__project_overview__"
        or fq.startswith("__flow__:")
        or fq.startswith("__entity_status__:")
    )


def _pseudo_boost(symbol_id: str | None, fqcn: str | None, q: str) -> float:
    """Boost flow/overview/status docs for process / tech-stack questions."""
    sid = symbol_id or ""
    fq = fqcn or ""
    q_l = (q or "").lower()
    boost = 0.0
    if sid.startswith("status:") or fq.startswith("__entity_status__:"):
        boost += 2.8
        if any(k in q for k in ("状态", "流转", "status", "怎么变", "变成")):
            boost += 2.5
    if sid.startswith("flow:") or fq.startswith("__flow__:"):
        boost += 2.5
        if any(k in q for k in ("流程", "链路", "怎么走", "完整")):
            boost += 2.0
    if sid.startswith("overview:") or fq == "__project_overview__":
        boost += 1.5
        if any(k in q_l for k in ("技术栈", "版本", "用了什么", "overview", "依赖", "框架")):
            boost += 2.0
    return boost


def _hit_from_vector(project_id: str, hit: dict, q: str) -> SymbolHit:
    sid = hit.get("symbol_id") or ""
    fqcn = hit.get("fqcn") or ""
    tags_raw = hit.get("tags") or ""
    tags = [t for t in tags_raw.split(",") if t] if isinstance(tags_raw, str) else []
    score = float(hit.get("score") or 0) + _pseudo_boost(sid, fqcn, q)
    kind = "FlowDoc" if sid.startswith("flow:") or (fqcn or "").startswith("__flow__:") else "Overview"
    if sid.startswith("status:") or (fqcn or "").startswith("__entity_status__:"):
        kind = "StatusDoc"
    if not _is_pseudo_doc(sid, fqcn):
        kind = "Unknown"
    return SymbolHit(
        symbol_id=sid,
        project_id=project_id,
        fqcn=fqcn,
        method_sig=None,
        kind=kind,
        file_path="",
        summary=hit.get("text"),
        tags=tags,
        business_domain=(
            "status"
            if kind == "StatusDoc"
            else ("flow" if kind == "FlowDoc" else "overview")
        ),
        score=score,
    )


def _embed_with_cache(embedder: Any, q: str) -> list[float] | None:
    """embed_one + 进程内 LRU 缓存（best-effort；缓存键含 embedding provider/model）。"""
    from app.core.config import get_settings

    settings = get_settings()
    enabled = bool(settings.search_cache_enabled)
    provider = settings.embedding_provider or "embed"
    model = settings.embedding_model or "embed"
    if enabled:
        try:
            from app.tagging import search_cache

            hit = search_cache.cached_embed(
                provider,
                model,
                q,
                maxsize=int(settings.search_cache_max_embed or 512),
            )
            if hit is not None:
                return hit
        except Exception:
            logger.debug("embed cache get failed", exc_info=True)
    vec = embedder.embed_one(q)
    if vec and enabled:
        try:
            from app.tagging import search_cache

            search_cache.put_embed(
                provider,
                model,
                q,
                vec,
                maxsize=int(settings.search_cache_max_embed or 512),
            )
        except Exception:
            logger.debug("embed cache put failed", exc_info=True)
    return vec


def hybrid_search(
    q: str,
    project_id: str,
    *,
    top_k: int = 12,
    domain: str | None = None,
) -> list[SymbolHit]:
    """混合检索（Milvus 向量 + MySQL 标签 + 字面关键词），带进程内 LRU 缓存。

    缓存键 (project_id, q, top_k, domain)：命中直接返回重建结果，追问/重复问
    复用同一批证据（快 + 口径稳定）。ingest 完成后由 app.services.ingest 调用
    search_cache.clear() 清空，避免复用旧索引证据。
    """
    from app.core.config import get_settings

    settings = get_settings()
    if bool(settings.search_cache_enabled):
        try:
            from app.tagging import search_cache

            hit = search_cache.get_search(
                project_id,
                q,
                top_k=top_k,
                domain=domain,
                maxsize=int(settings.search_cache_max_search or 256),
            )
            if hit is not None:
                return hit
        except Exception:
            logger.debug("search cache get failed", exc_info=True)

    out = _hybrid_search_impl(q, project_id, top_k=top_k, domain=domain)

    if bool(settings.search_cache_enabled):
        try:
            from app.tagging import search_cache

            search_cache.put_search(
                project_id,
                q,
                top_k=top_k,
                domain=domain,
                hits=out,
                maxsize=int(settings.search_cache_max_search or 256),
            )
        except Exception:
            logger.debug("search cache put failed", exc_info=True)
    return out


def _hybrid_search_impl(
    q: str,
    project_id: str,
    *,
    top_k: int = 12,
    domain: str | None = None,
) -> list[SymbolHit]:
    from app.core.config import get_settings

    vector_hits: list[dict] = []
    use_milvus = bool(get_settings().search_use_milvus)
    if use_milvus:
        try:
            embedder = get_embedding_provider()
            milvus = get_milvus()
            vec = _embed_with_cache(embedder, q)
            vector_hits = milvus.search(vec, project_id=project_id, top_k=top_k * 2)
        except Exception:
            # Vector path is optional — fall back to MySQL/tag search so chat still works.
            logger.exception("vector search failed; continuing with SQL/tag search only")
    else:
        logger.debug("search_use_milvus=false; SQL/literal search only")

    by_id: dict[str, SymbolHit] = {}
    literals = extract_query_literals(q)

    try:
        with get_session() as session:
            like = f"%{q}%"
            tag_symbol_ids: list[str] = []
            tag_rows = session.scalars(
                select(TagDictORM).where(
                    TagDictORM.project_id == project_id,
                    TagDictORM.tag.like(like),
                )
            )
            tag_ids = [t.id for t in tag_rows]
            if tag_ids:
                tag_symbol_ids = list(
                    session.scalars(
                        select(SymbolTagORM.symbol_id).where(SymbolTagORM.tag_id.in_(tag_ids))
                    )
                )

            stmt = select(SymbolORM).where(SymbolORM.project_id == project_id)
            stmt = stmt.where(
                or_(
                    SymbolORM.fqcn.like(like),
                    SymbolORM.simple_name.like(like),
                    SymbolORM.method_name.like(like),
                    SymbolORM.method_sig.like(like),
                    SymbolORM.id.in_(tag_symbol_ids) if tag_symbol_ids else SymbolORM.fqcn.like(like),
                )
            )
            if domain:
                desc_ids = list(
                    session.scalars(
                        select(SymbolDescORM.symbol_id).where(
                            SymbolDescORM.project_id == project_id,
                            SymbolDescORM.business_domain == domain,
                        )
                    )
                )
                if desc_ids:
                    stmt = stmt.where(SymbolORM.id.in_(desc_ids))
            sql_symbols = list(session.scalars(stmt.limit(top_k * 2)))

            def enrich(sym: SymbolORM, score: float | None = None) -> SymbolHit:
                desc = session.scalar(
                    select(SymbolDescORM).where(SymbolDescORM.symbol_id == sym.id)
                )
                tags = list(
                    session.scalars(
                        select(TagDictORM.tag)
                        .join(SymbolTagORM, SymbolTagORM.tag_id == TagDictORM.id)
                        .where(SymbolTagORM.symbol_id == sym.id)
                    )
                )
                # Prefer flow narrative from entry extra_json when present
                summary = desc.summary if desc else None
                extra_boost = 0.0
                if desc and isinstance(desc.extra_json, dict):
                    flow_doc = desc.extra_json.get("flow_doc")
                    if isinstance(flow_doc, dict) and flow_doc.get("narrative"):
                        if any(k in q for k in ("流程", "链路", "怎么走", "完整")):
                            summary = str(flow_doc.get("narrative"))
                            extra_boost += 2.0
                    if desc.extra_json.get("important"):
                        extra_boost += 0.8
                return SymbolHit(
                    symbol_id=sym.id,
                    project_id=sym.project_id,
                    fqcn=sym.fqcn,
                    method_sig=sym.method_sig,
                    kind=sym.kind,
                    file_path=sym.file_path,
                    summary=summary,
                    tags=tags,
                    business_domain=desc.business_domain if desc else None,
                    score=(score or 0) + extra_boost,
                )

            for sym in sql_symbols:
                by_id[sym.id] = enrich(sym, score=0.5)

            for hit in vector_hits:
                sid = hit.get("symbol_id") or ""
                fqcn = hit.get("fqcn") or ""
                if _is_pseudo_doc(sid, fqcn):
                    item = _hit_from_vector(project_id, hit, q)
                    key = sid or item.symbol_id
                    if key in by_id:
                        by_id[key].score = max(by_id[key].score or 0, item.score or 0) + 0.3
                        if item.summary and (
                            not by_id[key].summary or len(item.summary) > len(by_id[key].summary or "")
                        ):
                            by_id[key].summary = item.summary
                    else:
                        by_id[key] = item
                    continue

                sym = session.get(SymbolORM, sid)
                if not sym:
                    # Keep vector-only hit as soft result
                    item = _hit_from_vector(project_id, hit, q)
                    item.kind = "Unknown"
                    by_id[sid or item.symbol_id] = item
                    continue
                item = enrich(sym, score=hit.get("score"))
                item.score = (item.score or 0) + _pseudo_boost(sid, fqcn, q)
                if sid in by_id:
                    prev = by_id[sid]
                    prev.score = max(prev.score or 0, item.score or 0) + 0.2
                    if item.summary and (not prev.summary or len(item.summary) > len(prev.summary or "")):
                        prev.summary = item.summary
                else:
                    by_id[sid] = item

            # Literal match in source/summary/tags — project-agnostic, uses user words only.
            for term in literals:
                if len(term) < 2:
                    continue
                like_t = f"%{term}%"
                desc_ids = list(
                    session.scalars(
                        select(SymbolDescORM.symbol_id).where(
                            SymbolDescORM.project_id == project_id,
                            SymbolDescORM.summary.like(like_t),
                        ).limit(top_k * 3)
                    )
                )
                # Also match flow narratives stored in extra_json via summary already;
                # boost important / flow_doc entries found by literal in summary
                tag_ids_t = list(
                    session.scalars(
                        select(TagDictORM.id).where(
                            TagDictORM.project_id == project_id,
                            TagDictORM.tag.like(like_t),
                        )
                    )
                )
                tag_sids: list[str] = []
                if tag_ids_t:
                    tag_sids = list(
                        session.scalars(
                            select(SymbolTagORM.symbol_id).where(
                                SymbolTagORM.tag_id.in_(tag_ids_t)
                            ).limit(top_k * 3)
                        )
                    )
                lit_conds = [
                    SymbolORM.raw_code.like(like_t),
                    SymbolORM.method_name.like(like_t),
                    SymbolORM.simple_name.like(like_t),
                    SymbolORM.fqcn.like(like_t),
                    # Swagger titles live in annotations JSON even when summary lacks them
                    cast(SymbolORM.annotations, String).like(like_t),
                ]
                if desc_ids:
                    lit_conds.append(SymbolORM.id.in_(desc_ids))
                if tag_sids:
                    lit_conds.append(SymbolORM.id.in_(tag_sids))
                # Also match api_doc / paths stashed on symbol_desc.extra_json
                extra_ids = list(
                    session.scalars(
                        select(SymbolDescORM.symbol_id).where(
                            SymbolDescORM.project_id == project_id,
                            cast(SymbolDescORM.extra_json, String).like(like_t),
                        ).limit(top_k * 3)
                    )
                )
                if extra_ids:
                    lit_conds.append(SymbolORM.id.in_(extra_ids))
                lit_syms = list(
                    session.scalars(
                        select(SymbolORM)
                        .where(SymbolORM.project_id == project_id, or_(*lit_conds))
                        .limit(top_k * 3)
                    )
                )
                # score: longer term ⇒ stronger boost (exact phrase beats single char noise)
                boost = 1.5 + min(len(term), 12) * 0.15
                for sym in lit_syms:
                    if sym.id in by_id:
                        by_id[sym.id].score = (by_id[sym.id].score or 0) + boost
                    else:
                        by_id[sym.id] = enrich(sym, score=boost)
                    # Extra for method-level Controller with term in body/javadoc
                    if (sym.kind or "") == "Controller" and sym.method_name:
                        by_id[sym.id].score = (by_id[sym.id].score or 0) + 1.0
    except Exception:
        return [
            _hit_from_vector(project_id, h, q)
            for h in vector_hits[:top_k]
        ]

    ranked = sorted(by_id.values(), key=lambda x: x.score or 0, reverse=True)
    return ranked[:top_k]


def hybrid_search_multi(
    q: str,
    project_ids: list[str],
    *,
    top_k: int = 12,
    domain: str | None = None,
) -> list[SymbolHit]:
    """Search across multiple projects and merge by score."""
    ids = [p for p in (project_ids or []) if p]
    if not ids:
        return []
    if len(ids) == 1:
        return hybrid_search(q, ids[0], top_k=top_k, domain=domain)
    per = max(4, (top_k + len(ids) - 1) // len(ids) + 2)
    merged: list[SymbolHit] = []
    seen: set[str] = set()
    for pid in ids:
        for hit in hybrid_search(q, pid, top_k=per, domain=domain):
            key = f"{hit.project_id}:{hit.symbol_id}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(hit)
    merged.sort(key=lambda x: x.score or 0, reverse=True)
    return merged[:top_k]
