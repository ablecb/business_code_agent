from __future__ import annotations

import logging
import re
import shutil
import zipfile
from pathlib import Path

from sqlalchemy import func, select

from app.core.config import ROOT_DIR, get_settings
from app.core.utils import new_id
from app.db.mysql import get_session, init_db
from app.models.orm import IngestJobORM, ProjectORM, SymbolORM
from app.parser.scanner import scan_java_files

PROJECTS_DIR = ROOT_DIR / "projects"
logger = logging.getLogger("app.projects")

# Avoid rescanning thousands of Java files on every UI poll.
_java_count_cache: dict[str, tuple[float, int]] = {}


def _safe_project_id(name: str) -> str:
    raw = re.sub(r"[^a-zA-Z0-9_\-]+", "-", name.strip()).strip("-").lower()
    return (raw or new_id("proj"))[:64]


def _count_java(root: Path) -> int:
    if not root.exists():
        return 0
    try:
        return len(scan_java_files(root))
    except Exception:
        return 0


def _count_java_cached(root: Path) -> int:
    key = str(root)
    try:
        mtime = root.stat().st_mtime
    except OSError:
        return 0
    hit = _java_count_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    n = _count_java(root)
    _java_count_cache[key] = (mtime, n)
    return n


def _latest_job(session, project_id: str) -> IngestJobORM | None:
    return session.scalar(
        select(IngestJobORM)
        .where(IngestJobORM.project_id == project_id)
        .order_by(IngestJobORM.created_at.desc(), IngestJobORM.id.desc())
        .limit(1)
    )


def _symbol_count(session, project_id: str) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(SymbolORM).where(SymbolORM.project_id == project_id)
        )
        or 0
    )


def project_to_dict(row: ProjectORM, job: IngestJobORM | None, symbol_count: int, java_files: int) -> dict:
    status = "ready"
    if job:
        if job.status in {"queued", "running"}:
            status = "parsing"
        elif job.status == "failed":
            status = "failed"
        elif job.status == "succeeded":
            status = "parsed"
    elif symbol_count > 0:
        status = "parsed"
    elif java_files == 0:
        status = "empty"

    latest_job = None
    if job:
        elapsed_sec = None
        if (
            job.status in {"succeeded", "failed", "cancelled"}
            and job.created_at
            and job.updated_at
        ):
            elapsed_sec = round(
                max(0.0, (job.updated_at - job.created_at).total_seconds()),
                2,
            )
        latest_job = {
            "id": job.id,
            "status": job.status,
            "stage": job.stage,
            "total_files": job.total_files or 0,
            "processed": job.processed or 0,
            "message": job.message,
            # Truncate — full traceback can make list API huge/slow for the UI.
            "error": (job.error[:500] + "…") if job.error and len(job.error) > 500 else job.error,
            "elapsed_sec": elapsed_sec,
        }

    return {
        "id": row.id,
        "name": row.name,
        "root_path": row.root_path,
        "description": row.description,
        "status": status,
        "java_files": java_files,
        "symbol_count": symbol_count,
        "latest_job": latest_job,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def ensure_default_project() -> None:
    """Register bundled sample project if present and not yet recorded."""
    settings = get_settings()
    sample = settings.project_root_path
    if not sample.exists():
        return
    init_db()
    with get_session() as session:
        if session.get(ProjectORM, settings.project_id):
            return
        session.add(
            ProjectORM(
                id=settings.project_id,
                name=settings.project_id,
                root_path=str(sample),
                description="默认样本工程",
            )
        )


def list_projects() -> list[dict]:
    # Keep this endpoint cheap: UI polls every ~2.5s during parse.
    ensure_default_project()
    with get_session() as session:
        rows = list(session.scalars(select(ProjectORM).order_by(ProjectORM.updated_at.desc())))
        result = []
        for row in rows:
            root = Path(row.root_path)
            if not root.is_absolute():
                root = (ROOT_DIR / root).resolve()
            job = _latest_job(session, row.id)
            # While ingest is writing symbols, COUNT(*) can stall; prefer job counters.
            if job and job.status in {"queued", "running"}:
                symbol_count = int(job.processed or 0)
            else:
                try:
                    symbol_count = _symbol_count(session, row.id)
                except Exception as e:
                    logger.warning("symbol_count failed for %s: %s", row.id, e)
                    symbol_count = 0
            result.append(
                project_to_dict(
                    row,
                    job,
                    symbol_count,
                    _count_java_cached(root),
                )
            )
        return result


def get_project(project_id: str) -> dict | None:
    init_db()
    with get_session() as session:
        row = session.get(ProjectORM, project_id)
        if not row:
            return None
        root = Path(row.root_path)
        if not root.is_absolute():
            root = (ROOT_DIR / root).resolve()
        return project_to_dict(
            row,
            _latest_job(session, row.id),
            _symbol_count(session, row.id),
            _count_java_cached(root),
        )


def register_project(
    *,
    project_id: str | None,
    name: str,
    root_path: str | Path,
    description: str | None = None,
) -> dict:
    init_db()
    pid = _safe_project_id(project_id or name)
    root = Path(root_path)
    if not root.is_absolute():
        root = (ROOT_DIR / root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"项目路径不存在: {root}")

    with get_session() as session:
        row = session.get(ProjectORM, pid)
        if row is None:
            row = ProjectORM(
                id=pid,
                name=name or pid,
                root_path=str(root),
                description=description,
            )
            session.add(row)
        else:
            row.name = name or row.name
            row.root_path = str(root)
            if description is not None:
                row.description = description
        session.flush()
        job = _latest_job(session, pid)
        return project_to_dict(row, job, _symbol_count(session, pid), _count_java_cached(root))


def _unwrap_single_root(extract_dir: Path) -> Path:
    """If zip has one top-level folder, use that as project root."""
    entries = [p for p in extract_dir.iterdir() if p.name not in {".DS_Store", "__MACOSX"}]
    dirs = [p for p in entries if p.is_dir()]
    files = [p for p in entries if p.is_file()]
    if len(dirs) == 1 and not files:
        return dirs[0]
    return extract_dir


def save_uploaded_zip(
    *,
    filename: str,
    content: bytes,
    project_id: str | None = None,
    name: str | None = None,
    description: str | None = None,
) -> dict:
    if not filename.lower().endswith(".zip"):
        raise ValueError("仅支持上传 .zip 压缩包")

    stem = Path(filename).stem
    pid = _safe_project_id(project_id or stem)
    display = name or stem
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
    target = PROJECTS_DIR / pid
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)

    zip_path = target / "_upload.zip"
    zip_path.write_bytes(content)
    extract_dir = target / "src"
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as e:
        shutil.rmtree(target, ignore_errors=True)
        raise ValueError("无效的 zip 文件") from e
    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)

    root = _unwrap_single_root(extract_dir)
    return register_project(
        project_id=pid,
        name=display,
        root_path=root,
        description=description or f"上传自 {filename}",
    )
