"""Global knowledge-base API — upload / list / delete docs used in chat RAG."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.knowledge.service import (
    delete_document,
    get_document,
    index_document,
    list_documents,
    save_upload,
)

router = APIRouter(tags=["knowledge"])


class KbDocumentOut(BaseModel):
    id: str
    title: str
    filename: str
    kind: str
    status: str
    char_count: int = 0
    chunk_count: int = 0
    error: str | None = None
    created_at: str | None = None
    updated_at: str | None = None


class KbListResponse(BaseModel):
    items: list[KbDocumentOut] = Field(default_factory=list)


@router.get("/knowledge", response_model=KbListResponse)
def api_list_knowledge() -> KbListResponse:
    items = [KbDocumentOut(**d) for d in list_documents()]
    return KbListResponse(items=items)


@router.post("/knowledge/upload", response_model=KbDocumentOut)
async def api_upload_knowledge(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
) -> KbDocumentOut:
    data = await file.read()
    try:
        # Index synchronously for small docs so UI shows ready quickly;
        # save_upload already calls index_document.
        doc = save_upload(
            filename=file.filename or "upload.txt",
            data=data,
            title=title,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}") from e
    return KbDocumentOut(**doc)


@router.post("/knowledge/{doc_id}/reindex", response_model=KbDocumentOut)
def api_reindex_knowledge(doc_id: str, background: BackgroundTasks) -> KbDocumentOut:
    if not get_document(doc_id):
        raise HTTPException(status_code=404, detail="文档不存在")
    background.add_task(index_document, doc_id)
    doc = get_document(doc_id) or {"id": doc_id, "status": "indexing"}
    # fill required fields for response
    return KbDocumentOut(
        id=str(doc.get("id") or doc_id),
        title=str(doc.get("title") or ""),
        filename=str(doc.get("filename") or ""),
        kind=str(doc.get("kind") or "txt"),
        status="indexing",
        char_count=int(doc.get("char_count") or 0),
        chunk_count=int(doc.get("chunk_count") or 0),
        error=doc.get("error"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
    )


@router.delete("/knowledge/{doc_id}")
def api_delete_knowledge(doc_id: str) -> dict[str, Any]:
    ok = delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"ok": True, "id": doc_id}
