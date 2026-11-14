"""Global knowledge base — upload docs, chunk/embed, retrieve into chat."""

from app.knowledge.retrieve import format_kb_context, search_knowledge
from app.knowledge.service import (
    delete_document,
    index_document,
    list_documents,
    save_upload,
)

__all__ = [
    "delete_document",
    "format_kb_context",
    "index_document",
    "list_documents",
    "save_upload",
    "search_knowledge",
]
