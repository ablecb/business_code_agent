"""Retrieve global knowledge-base chunks for chat context."""
from __future__ import annotations

import logging
import math
from typing import Any

from app.db.mysql import get_session
from app.models.orm import KbChunkORM, KbDocumentORM

logger = logging.getLogger("app.knowledge")


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        fx = float(x)
        fy = float(y)
        dot += fx * fy
        na += fx * fx
        nb += fy * fy
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def search_knowledge(query: str, *, top_k: int = 5) -> list[dict[str, Any]]:
    """Embed query and rank ready chunks (global, no project filter)."""
    q = (query or "").strip()
    if not q:
        return []
    try:
        from app.llm.factory import get_embedding_provider

        qvec = get_embedding_provider().embed_one(q)
    except Exception:
        logger.debug("kb embed query failed", exc_info=True)
        return _keyword_fallback(q, top_k=top_k)

    if not qvec:
        return _keyword_fallback(q, top_k=top_k)

    scored: list[tuple[float, dict[str, Any]]] = []
    with get_session() as session:
        ready_docs = (
            session.query(KbDocumentORM)
            .filter(KbDocumentORM.status == "ready")
            .all()
        )
        ready_ids = {d.id for d in ready_docs}
        if not ready_ids:
            return []
        rows = (
            session.query(KbChunkORM, KbDocumentORM)
            .join(KbDocumentORM, KbDocumentORM.id == KbChunkORM.document_id)
            .filter(KbDocumentORM.status == "ready")
            .all()
        )
        for chunk, doc in rows:
            emb = chunk.embedding_json
            if not isinstance(emb, list) or not emb:
                continue
            sc = _cosine(list(qvec), [float(x) for x in emb])
            if sc < 0.15:
                continue
            scored.append(
                (
                    sc,
                    {
                        "document_id": doc.id,
                        "title": doc.title,
                        "filename": doc.filename,
                        "chunk_index": chunk.chunk_index,
                        "score": round(sc, 4),
                        "text": (chunk.text or "")[:1800],
                    },
                )
            )
    scored.sort(key=lambda x: -x[0])
    return [item for _, item in scored[: max(1, top_k)]]


def _keyword_fallback(query: str, *, top_k: int) -> list[dict[str, Any]]:
    tokens = [t for t in re_split_tokens(query) if len(t) >= 2][:8]
    if not tokens:
        return []
    out: list[dict[str, Any]] = []
    with get_session() as session:
        rows = (
            session.query(KbChunkORM, KbDocumentORM)
            .join(KbDocumentORM, KbDocumentORM.id == KbChunkORM.document_id)
            .filter(KbDocumentORM.status == "ready")
            .limit(400)
            .all()
        )
        for chunk, doc in rows:
            text = chunk.text or ""
            hit = sum(1 for t in tokens if t.lower() in text.lower())
            if hit <= 0:
                continue
            out.append(
                {
                    "document_id": doc.id,
                    "title": doc.title,
                    "filename": doc.filename,
                    "chunk_index": chunk.chunk_index,
                    "score": float(hit),
                    "text": text[:1800],
                }
            )
    out.sort(key=lambda x: -float(x.get("score") or 0))
    return out[: max(1, top_k)]


def re_split_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]{2,}", text or "")


def format_kb_context(hits: list[dict[str, Any]], *, max_chars: int = 3500) -> str:
    if not hits:
        return ""
    lines = ["[知识库检索]", "以下摘录来自用户上传的全局文档，回答改动/接口题时可引用：", ""]
    used = 0
    for i, h in enumerate(hits, start=1):
        block = (
            f"### KB-{i} 《{h.get('title') or h.get('filename')}》 "
            f"(score={h.get('score')})\n{h.get('text') or ''}\n"
        )
        if used + len(block) > max_chars:
            break
        lines.append(block)
        used += len(block)
    return "\n".join(lines).strip()
