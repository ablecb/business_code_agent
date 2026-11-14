from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from sqlalchemy import select

from app.core.config import ROOT_DIR, get_settings
from app.core.utils import new_id
from app.db.mysql import get_session, init_db
from app.models.orm import IngestJobORM, ProjectORM
from app.models.schemas import (
    IngestJobResponse,
    ProjectCreateRequest,
    ProjectParseRequest,
    ProjectResponse,
)
from app.services.ingest import run_ingest_job
from app.services.projects import (
    get_project,
    list_projects,
    register_project,
    save_uploaded_zip,
)

router = APIRouter(tags=["projects"])
logger = logging.getLogger("app.ingest")


def _job_resp(job: IngestJobORM) -> IngestJobResponse:
    # Copy scalars immediately — do not keep a live ORM instance across Session close.
    return IngestJobResponse(
        id=str(job.id),
        project_id=str(job.project_id),
        status=str(job.status),
        stage=job.stage,
        total_files=int(job.total_files or 0),
        processed=int(job.processed or 0),
        message=job.message,
        error=job.error,
    )


@router.get("/projects", response_model=list[ProjectResponse])
def api_list_projects() -> list[ProjectResponse]:
    try:
        return [ProjectResponse(**p) for p in list_projects()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"无法读取项目列表: {e}") from e


@router.post("/projects", response_model=ProjectResponse)
def api_create_project(req: ProjectCreateRequest) -> ProjectResponse:
    settings = get_settings()
    path = req.path or str(settings.project_root_path)
    try:
        data = register_project(
            project_id=req.project_id,
            name=req.name,
            root_path=path,
            description=req.description,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return ProjectResponse(**data)


@router.post("/projects/upload", response_model=ProjectResponse)
async def api_upload_project(
    file: UploadFile = File(...),
    project_id: str | None = Form(default=None),
    name: str | None = Form(default=None),
    description: str | None = Form(default=None),
) -> ProjectResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件为空")
    try:
        data = save_uploaded_zip(
            filename=file.filename or "project.zip",
            content=content,
            project_id=project_id,
            name=name,
            description=description,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上传失败: {e}") from e
    return ProjectResponse(**data)


@router.post("/projects/{project_id}/parse", response_model=IngestJobResponse)
async def api_parse_project(
    project_id: str,
    background: BackgroundTasks,
    req: ProjectParseRequest | None = None,
) -> IngestJobResponse:
    body = req or ProjectParseRequest()
    init_db()
    settings = get_settings()
    with get_session() as session:
        proj = session.get(ProjectORM, project_id)
        if not proj:
            raise HTTPException(status_code=404, detail="项目不存在")
        root = Path(proj.root_path)
        if not root.is_absolute():
            root = (ROOT_DIR / root).resolve()
        if not root.exists():
            raise HTTPException(status_code=400, detail=f"项目路径不存在: {root}")

        latest = session.scalar(
            select(IngestJobORM)
            .where(IngestJobORM.project_id == project_id)
            .order_by(IngestJobORM.created_at.desc())
            .limit(1)
        )
        if latest and latest.status in {"queued", "running"}:
            if not body.force:
                logger.info(
                    "parse request ignored; existing job %s status=%s stage=%s",
                    latest.id,
                    latest.status,
                    latest.stage,
                )
                return _job_resp(latest)
            logger.warning(
                "force re-parse: cancelling job %s (was %s/%s)",
                latest.id,
                latest.status,
                latest.stage,
            )
            latest.status = "failed"
            latest.stage = "cancelled"
            latest.message = "cancelled by force re-parse"
            latest.error = "superseded by a new parse request"

        job_id = new_id("job_")
        job = IngestJobORM(
            id=job_id,
            project_id=project_id,
            status="queued",
            stage="queued",
            message=f"queued parse for {root}",
        )
        session.add(job)
        session.flush()
        resp = _job_resp(job)

    tagging = settings.tagging_enabled if body.tagging is None else body.tagging
    logger.info(
        "queue parse job=%s project=%s root=%s force=%s tagging=%s full_refresh=%s",
        job_id,
        project_id,
        root,
        body.force,
        tagging,
        body.full_refresh,
    )
    background.add_task(
        run_ingest_job,
        job_id,
        project_id,
        root,
        tagging=tagging,
        full_refresh=body.full_refresh,
    )
    return resp


@router.get("/projects/{project_id}", response_model=ProjectResponse)
def api_get_project(project_id: str) -> ProjectResponse:
    data = get_project(project_id)
    if not data:
        raise HTTPException(status_code=404, detail="项目不存在")
    return ProjectResponse(**data)
