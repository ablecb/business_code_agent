"""Persist and index global knowledge-base documents."""
from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from app.core.config import ROOT_DIR
from app.db.mysql import get_session
from app.knowledge.chunker import chunk_text, extract_text_from_bytes
from app.models.orm import KbChunkORM, KbDocumentORM

logger = logging.getLogger("app.knowledge")

KB_DIR = ROOT_DIR / "data" / "knowledge"
_ALLOWED = {".md", ".txt", ".pdf", ".markdown"}


def _kind_from_name(filename: str) -> str:
    suf = Path(filename or "").suffix.lower()
    if suf in {".md", ".markdown"}:
        return "md"
    if suf == ".pdf":
        return "pdf"
    return "txt"


def save_upload(
    *,
    filename: str,
    data: bytes,
    title: str | None = None,
) -> dict[str, Any]:
    name = Path(filename or "upload.txt").name
    suf = Path(name).suffix.lower()
    if suf not in _ALLOWED:
        raise ValueError(f"仅支持 md/txt/pdf，收到: {suf or '(无扩展名)'}")
    if not data:
        raise ValueError("空文件")
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("文件过大（上限 20MB）")

    doc_id = uuid.uuid4().hex[:16]
    kind = _kind_from_name(name)
    doc_dir = KB_DIR / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    path = doc_dir / name
    path.write_bytes(data)

    title_s = (title or Path(name).stem or doc_id).strip()[:256]
    with get_session() as session:
        session.add(
            KbDocumentORM(
                id=doc_id,
                title=title_s,
                filename=name,
                kind=kind,
                storage_path=str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
                status="queued",
                char_count=0,
                chunk_count=0,
            )
        )
    index_document(doc_id)
    return get_document(doc_id) or {"id": doc_id, "status": "failed"}


def index_document(doc_id: str) -> None:
    with get_session() as session:
        doc = session.get(KbDocumentORM, doc_id)
        if not doc:
            return
        doc.status = "indexing"
        doc.error = None
        storage = doc.storage_path
        kind = doc.kind
        filename = doc.filename

    path = ROOT_DIR / storage
    try:
        raw = path.read_bytes()
        text = extract_text_from_bytes(raw, filename=filename, kind=kind)
        chunks = chunk_text(text)
        if not chunks:
            raise ValueError("文档无可用文本")

        from app.llm.factory import get_embedding_provider

        embedder = get_embedding_provider()
        vectors = embedder.embed(chunks)
        if len(vectors) != len(chunks):
            raise ValueError("embedding 数量与分块不一致")

        with get_session() as session:
            session.query(KbChunkORM).filter(KbChunkORM.document_id == doc_id).delete()
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                session.add(
                    KbChunkORM(
                        id=f"{doc_id}_{i:04d}",
                        document_id=doc_id,
                        chunk_index=i,
                        text=chunk[:12000],
                        embedding_json=list(vec) if vec is not None else None,
                    )
                )
            doc = session.get(KbDocumentORM, doc_id)
            if doc:
                doc.status = "ready"
                doc.char_count = len(text)
                doc.chunk_count = len(chunks)
                doc.error = None
                doc.updated_at = datetime.utcnow()
    except Exception as e:
        logger.exception("index kb doc %s failed", doc_id)
        with get_session() as session:
            doc = session.get(KbDocumentORM, doc_id)
            if doc:
                doc.status = "failed"
                doc.error = str(e)[:500]
                doc.updated_at = datetime.utcnow()


def list_documents() -> list[dict[str, Any]]:
    with get_session() as session:
        rows = (
            session.query(KbDocumentORM)
            .order_by(KbDocumentORM.created_at.desc())
            .limit(200)
            .all()
        )
        return [_doc_dict(r) for r in rows]


def get_document(doc_id: str) -> dict[str, Any] | None:
    with get_session() as session:
        doc = session.get(KbDocumentORM, doc_id)
        return _doc_dict(doc) if doc else None


def delete_document(doc_id: str) -> bool:
    with get_session() as session:
        doc = session.get(KbDocumentORM, doc_id)
        if not doc:
            return False
        storage = doc.storage_path
        session.query(KbChunkORM).filter(KbChunkORM.document_id == doc_id).delete()
        session.delete(doc)
    try:
        path = ROOT_DIR / storage
        if path.is_file():
            path.unlink()
        parent = path.parent
        if parent.is_dir() and parent != KB_DIR and not any(parent.iterdir()):
            parent.rmdir()
    except Exception:
        logger.debug("kb file cleanup failed", exc_info=True)
    return True


def _doc_dict(doc: KbDocumentORM) -> dict[str, Any]:
    return {
        "id": doc.id,
        "title": doc.title,
        "filename": doc.filename,
        "kind": doc.kind,
        "status": doc.status,
        "char_count": doc.char_count,
        "chunk_count": doc.chunk_count,
        "error": doc.error,
        "created_at": doc.created_at.isoformat() + "Z" if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() + "Z" if doc.updated_at else None,
    }


def guess_title_from_text(text: str, fallback: str) -> str:
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^#+\s+(.+)$", line)
        if m:
            return m.group(1).strip()[:256]
        return line[:256]
    return fallback
