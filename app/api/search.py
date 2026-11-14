from __future__ import annotations

from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.models.schemas import SymbolHit
from app.tagging.search import hybrid_search

router = APIRouter(tags=["search"])


@router.get("/search", response_model=list[SymbolHit])
def search(
    q: str = Query(..., min_length=1),
    project_id: str | None = None,
    top_k: int = 12,
    domain: str | None = None,
) -> list[SymbolHit]:
    settings = get_settings()
    return hybrid_search(
        q,
        project_id or settings.project_id,
        top_k=top_k,
        domain=domain,
    )
