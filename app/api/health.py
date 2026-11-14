from __future__ import annotations

from fastapi import APIRouter

from app import __version__
from app.core.config import get_settings
from app.db.milvus_client import get_milvus
from app.db.mysql import get_engine
from app.db.neo4j_client import get_neo4j
from app.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    checks: dict[str, str] = {}
    try:
        from sqlalchemy import text

        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        checks["mysql"] = "ok"
    except Exception as e:
        checks["mysql"] = f"error: {e}"
    try:
        checks["neo4j"] = "ok" if get_neo4j().verify() else "error"
    except Exception as e:
        checks["neo4j"] = f"error: {e}"
    try:
        milvus = get_milvus()
        if milvus.verify():
            checks["milvus"] = (
                f"ok ({settings.milvus_mode}:"
                f"{settings.milvus_lite_path if settings.milvus_mode == 'lite' else settings.milvus_host})"
            )
        else:
            checks["milvus"] = "memory_fallback"
    except Exception as e:
        checks["milvus"] = f"error: {e}"

    status = "ok" if checks.get("mysql") == "ok" else "degraded"
    return HealthResponse(
        status=status,
        version=__version__,
        llm_provider=settings.llm_provider,
        checks=checks,
    )
