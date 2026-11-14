from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import api_router
from app.core.config import ROOT_DIR, get_settings
from app.core.logging_setup import setup_logging
from app.db.mysql import init_db
from app.llm.tracing import setup_langsmith

WEB_DIR = ROOT_DIR / "web"
logger = logging.getLogger("uvicorn.error")
setup_logging()
setup_langsmith()


def _pin_caches_to_project() -> None:
    """Keep HF/Torch caches on project disk (D:) instead of filling C:."""
    settings = get_settings()
    mapping = {
        "HF_HOME": settings.hf_home,
        "TRANSFORMERS_CACHE": settings.transformers_cache,
        "TORCH_HOME": settings.torch_home,
    }
    for key, value in mapping.items():
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)


@asynccontextmanager
async def lifespan(_: FastAPI):
    _pin_caches_to_project()
    # init_db can block on MySQL metadata locks; never stall process startup forever.
    logger.info("Initializing database (timeout 20s)…")
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(init_db)
            fut.result(timeout=20)
        logger.info("Database ready")
    except FuturesTimeout:
        logger.error(
            "init_db timed out after 20s — MySQL may be stuck on metadata locks. "
            "App will start anyway; check docker logs / SHOW PROCESSLIST."
        )
    except Exception:
        logger.exception("init_db failed; app will start without DB readiness")
    _orphan_ingest_jobs_cleanup()
    try:
        from app.analysis.var_resolve import maybe_startup_var_resolve_ensure

        maybe_startup_var_resolve_ensure()
    except Exception:
        logger.exception("startup var-resolve ensure schedule failed")
    yield


def _pid_alive(pid: int | None) -> bool:
    """Windows-safe liveness check (os.kill(pid, 0) would TERMINATE on Windows)."""
    if not pid or int(pid) <= 0:
        return False
    try:
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid)
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    except Exception:
        return False


def _orphan_ingest_jobs_cleanup() -> None:
    """Mark ingest jobs failed after a backend restart — only true orphans.

    BackgroundTasks die with the process, so a job that was running inside the
    previous backend is orphaned after a restart (its worker is gone) and would
    otherwise show ``running`` forever in the UI.

    Safety: only jobs whose recorded ``pid`` is dead (or missing) are marked.
    A CLI parse running in a *separate* process (pid alive) is left untouched —
    restarting the API server must not kill a CLI ingest.
    """
    try:
        from datetime import datetime

        from sqlalchemy import select

        from app.db.mysql import get_session, init_db
        from app.models.orm import IngestJobORM

        init_db()
        orphans: list[str] = []
        with get_session() as session:
            rows = session.scalars(
                select(IngestJobORM).where(
                    IngestJobORM.status.in_(("queued", "running"))
                )
            ).all()
            for row in rows:
                if not _pid_alive(row.pid):
                    orphans.append(row.id)
                    row.status = "failed"
                    row.stage = "cancelled"
                    row.message = "backend restarted, job orphaned"
                    row.error = "orphaned: owning process no longer exists"
                    row.updated_at = datetime.utcnow()
        if orphans:
            logger.warning(
                "orphaned %s ingest job(s) marked failed on startup: %s",
                len(orphans),
                orphans,
            )
    except Exception:
        logger.exception("orphan ingest-job cleanup failed (non-fatal)")


app = FastAPI(
    title="Business Code Agent",
    version=__version__,
    description="Java/SpringCloud code intelligence + AI coding agent",
    lifespan=lifespan,
)
app.include_router(api_router)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
def root():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    settings = get_settings()
    return {
        "name": "Business Code Agent",
        "version": __version__,
        "project_id": settings.project_id,
        "llm_provider": settings.llm_provider,
        "docs": "/docs",
    }


@app.get("/ui")
def ui():
    index = WEB_DIR / "index.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


@app.get("/logdiag")
def logdiag_page():
    index = WEB_DIR / "logdiag.html"
    if index.exists():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


@app.get("/projects")
def projects_page():
    page = WEB_DIR / "projects.html"
    if page.exists():
        return FileResponse(page)
    return RedirectResponse(url="/")


def run() -> None:
    import uvicorn

    setup_logging()
    setup_langsmith()
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
