from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.config import ROOT_DIR, get_settings
from app.core.utils import new_id
from app.db.mysql import get_session, init_db
from app.models.orm import IngestJobORM
from app.models.schemas import IngestJobResponse, IngestRequest
from app.services.ingest import get_job, run_ingest_job

router = APIRouter(tags=["ingest"])


def _to_resp(job: IngestJobORM) -> IngestJobResponse:
    return IngestJobResponse(
        id=job.id,
        project_id=job.project_id,
        status=job.status,
        stage=job.stage,
        total_files=job.total_files or 0,
        processed=job.processed or 0,
        message=job.message,
        error=job.error,
    )


@router.post("/projects/ingest", response_model=IngestJobResponse)
async def ingest_project(req: IngestRequest, background: BackgroundTasks) -> IngestJobResponse:
    settings = get_settings()
    init_db()
    project_id = req.project_id or settings.project_id
    root = settings.project_root_path
    if req.path:
        from pathlib import Path

        root = Path(req.path)
        if not root.is_absolute():
            root = (ROOT_DIR / root).resolve()
    tagging = settings.tagging_enabled if req.tagging is None else req.tagging
    job_id = new_id("job_")
    with get_session() as session:
        job = IngestJobORM(
            id=job_id,
            project_id=project_id,
            status="queued",
            stage="queued",
            message=f"queued for {root}",
        )
        session.add(job)
        session.flush()
        resp = _to_resp(job)

    background.add_task(
        run_ingest_job,
        job_id,
        project_id,
        root,
        tagging=tagging,
        full_refresh=req.full_refresh,
    )
    return resp


@router.get("/jobs/{job_id}", response_model=IngestJobResponse)
def job_status(job_id: str) -> IngestJobResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return _to_resp(job)
