from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.utils import new_id
from app.db.milvus_client import get_milvus
from app.db.mysql import get_session, init_db
from app.db.neo4j_client import get_neo4j
from app.models.orm import IngestJobORM
from app.parser.symbol_store import clear_symbols, upsert_project

logger = logging.getLogger("app.ingest")


def _format_duration_zh(seconds: float) -> str:
    """Human-readable duration for job messages / UI (e.g. 12分3秒)."""
    sec = max(0, int(round(float(seconds or 0))))
    if sec < 60:
        return f"{sec}秒"
    hours, rem = divmod(sec, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        parts = [f"{hours}小时"]
        if minutes:
            parts.append(f"{minutes}分")
        if secs and not minutes:
            parts.append(f"{secs}秒")
        elif secs and minutes:
            parts.append(f"{secs}秒")
        return "".join(parts)
    if secs:
        return f"{minutes}分{secs}秒"
    return f"{minutes}分"


def _job_elapsed_sec(job_id: str) -> float | None:
    with get_session() as session:
        row = session.get(IngestJobORM, job_id)
        if not row or not row.created_at:
            return None
        return max(0.0, (datetime.utcnow() - row.created_at).total_seconds())


def _with_elapsed(job_id: str, message: str) -> str:
    elapsed = _job_elapsed_sec(job_id)
    if elapsed is None:
        return message
    suffix = f"耗时 {_format_duration_zh(elapsed)}"
    base = (message or "").rstrip()
    return f"{base} · {suffix}" if base else suffix


def _update_job(job_id: str, **kwargs) -> None:
    with get_session() as session:
        row = session.get(IngestJobORM, job_id)
        if not row:
            return
        for k, v in kwargs.items():
            setattr(row, k, v)
        # Ensure terminal updates bump updated_at even if ORM onupdate is skipped.
        if "status" in kwargs and kwargs["status"] in {"succeeded", "failed", "cancelled"}:
            row.updated_at = datetime.utcnow()


def _job_is_running(job_id: str) -> bool:
    with get_session() as session:
        row = session.get(IngestJobORM, job_id)
        return bool(row and row.status == "running")


def _prepare_stores(project_id: str, project_root: Path, *, full_refresh: bool) -> None:
    init_db()
    with get_session() as session:
        upsert_project(session, project_id, project_id, str(project_root))
        if full_refresh:
            clear_symbols(session, project_id)
    neo = get_neo4j()
    if not neo.verify():
        raise RuntimeError("Neo4j unavailable")
    if full_refresh:
        from app.analysis.neo4j_store import clear_project_graph

        clear_project_graph(project_id)
        try:
            get_milvus().clear_project(project_id)
        except Exception:
            logger.exception("milvus clear_project failed")
        try:
            from app.db.milvus_client import get_entry_route_store

            get_entry_route_store().clear_project(project_id)
        except Exception:
            logger.exception("entry_route clear_project failed")
    # 索引变更（含失败前的清空）→ 问答检索缓存必须同步清空，防止复用旧索引证据。
    try:
        from app.tagging import search_cache

        search_cache.clear()
    except Exception:
        logger.debug("search cache clear failed", exc_info=True)


def _package_roots_from_settings() -> list[str] | None:
    settings = get_settings()
    raw = (settings.topo_package_roots or "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return parts or None


async def run_ingest_job(
    job_id: str,
    project_id: str,
    project_root: Path,
    *,
    tagging: bool = True,
    full_refresh: bool = True,
) -> None:
    """Parse-project pipeline.

    Default (topo_ast_pipeline_enabled):
      prepare → global Tree-sitter AST (once) → topo+priority AI
      → MySQL / Milvus / Neo4j CALLS → entry_route

    Legacy fallback (topo disabled):
      overview → entries → recursive flow → entry_route
    """
    settings = get_settings()
    use_topo = bool(getattr(settings, "topo_ast_pipeline_enabled", True))
    logger.info(
        "[%s] START project=%s root=%s concurrency=%s topo=%s",
        job_id,
        project_id,
        project_root,
        settings.parse_concurrency,
        use_topo,
    )
    try:
        _update_job(
            job_id,
            status="running",
            stage="prepare",
            message="preparing stores",
            processed=0,
            total_files=1,
            pid=os.getpid(),
        )
        await asyncio.to_thread(
            _prepare_stores, project_id, project_root, full_refresh=full_refresh
        )

        use_llm = bool(tagging and settings.tagging_enabled)
        if not _job_is_running(job_id):
            return

        if use_topo:
            await _run_topo_path(job_id, project_id, project_root, use_llm=use_llm)
        else:
            await _run_legacy_path(job_id, project_id, project_root, use_llm=use_llm)

        # 索引更新完成 → 清空问答检索缓存（incremental 也需使旧证据失效）。
        try:
            from app.tagging import search_cache

            search_cache.clear()
        except Exception:
            logger.debug("search cache clear failed", exc_info=True)

    except Exception as e:
        logger.exception("[%s] FAILED: %s", job_id, e)
        if not _job_is_running(job_id):
            return
        _update_job(
            job_id,
            status="failed",
            error=f"{e}\n{traceback.format_exc()}",
            message=_with_elapsed(job_id, "ingest failed"),
        )


async def _run_topo_path(
    job_id: str,
    project_id: str,
    project_root: Path,
    *,
    use_llm: bool,
) -> None:
    from app.analysis.topo_pipeline import run_topo_ast_pipeline

    settings = get_settings()
    package_roots = _package_roots_from_settings()
    max_methods = int(getattr(settings, "topo_max_methods", 0) or 0) or None

    _update_job(
        job_id,
        stage="global_ast",
        message="tree-sitter global AST + topo AI",
        processed=0,
        total_files=5,
    )

    # Throttle high-frequency LLM node ticks so list/poll stays cheap.
    prog_state = {"t": 0.0, "stage": "", "done": -1}

    def prog(done: int, total: int, msg: str) -> None:
        if not _job_is_running(job_id):
            return
        logger.info("[%s] topo %s/%s %s", job_id, done, total, msg)
        text = (msg or "").strip()
        low = text.lower()
        if "building global" in low or "global ast" in low or "ast parsed" in low or "ast ready" in low:
            stage = "global_ast"
        elif "persist symbol" in low:
            stage = "persist"
        elif "milvus" in low:
            stage = "milvus"
        elif "neo4j" in low or "calls from ast" in low:
            stage = "neo4j_calls"
        elif low.startswith("done in"):
            stage = "done"
        elif (
            "topo ai" in low
            or low.startswith("done ")
            or low.startswith("cached ")
            or low.startswith("failed ")
            or low.startswith("llm ")
            or "schedule nodes" in low
        ):
            stage = "topo_ai"
        else:
            stage = "topo_ai" if done >= 2 else "global_ast"

        now = time.monotonic()
        last = float(prog_state["t"] or 0.0)
        force = (
            stage != prog_state["stage"]
            or done >= total
            or done == 0
            or stage in {"persist", "milvus", "neo4j_calls", "done"}
        )
        # LLM ticks: ~1s; AST file ticks: ~0.6s.
        min_gap = 0.6 if stage == "global_ast" else 1.0
        if not force and (now - last) < min_gap and done == prog_state["done"]:
            return
        if not force and (now - last) < min_gap and stage == "topo_ai":
            # Still advance processed occasionally even within gap when done moves.
            if (now - last) < 0.85:
                return

        prog_state["t"] = now
        prog_state["stage"] = stage
        prog_state["done"] = done
        _update_job(
            job_id,
            stage=stage,
            processed=done,
            total_files=max(total, 1),
            message=text[:500],
        )

    if not use_llm:
        logger.warning("[%s] tagging disabled — AST persist only (dry AI)", job_id)

    stats = await run_topo_ast_pipeline(
        project_id,
        project_root,
        package_roots=package_roots,
        concurrency=settings.parse_concurrency,
        max_retries=int(getattr(settings, "topo_max_retries", 2)),
        max_methods=max_methods,
        use_llm=use_llm,
        persist_graph_calls=True,
        progress=prog,
        # Cooperative cancel: force re-parse / API cancel flips the DB row to
        # non-running, which aborts the scheduler / milvus / neo4j loops quickly
        # instead of letting the old task run for hours after "cancellation".
        is_cancelled=lambda: not _job_is_running(job_id),
    )
    logger.info("[%s] topo pipeline done %s", job_id, stats)
    if not _job_is_running(job_id):
        return

    route_stats: dict = {}
    try:
        _update_job(
            job_id,
            stage="entry_route",
            message="building entry_route classification",
            processed=0,
            total_files=1,
        )
        from app.analysis.entry_route import rebuild_entry_route

        route_stats = await asyncio.to_thread(rebuild_entry_route, project_id)
        logger.info("[%s] stage=entry_route done %s", job_id, route_stats)
    except Exception:
        logger.exception("[%s] stage=entry_route failed (non-fatal)", job_id)
        route_stats = {"error": "rebuild_failed"}

    if not _job_is_running(job_id):
        return

    done_msg = _with_elapsed(job_id, f"ok topo={stats} entry_route={route_stats}")
    _update_job(
        job_id,
        status="succeeded",
        stage="done",
        message=done_msg,
        processed=_job_processed(job_id),
    )
    logger.info("[%s] SUCCEEDED (topo) %s", job_id, done_msg)
    try:
        from app.agent.project_notebook import build_and_save_notebook

        await asyncio.to_thread(build_and_save_notebook, project_id, project_root)
    except Exception:
        logger.exception("[%s] project notebook build failed (non-fatal)", job_id)


async def _run_legacy_path(
    job_id: str,
    project_id: str,
    project_root: Path,
    *,
    use_llm: bool,
) -> None:
    from app.analysis.common import list_java_index
    from app.analysis.discover import discover_and_index_entries
    from app.analysis.overview import build_project_overview
    from app.analysis.recursive_flow import recursive_flow_from_entries

    settings = get_settings()

    _update_job(job_id, stage="overview", message="AI project overview", processed=0, total_files=3)

    def oprog(done: int, total: int, msg: str) -> None:
        if not _job_is_running(job_id):
            return
        logger.info("[%s] stage=overview %s/%s %s", job_id, done, total, msg)
        _update_job(job_id, stage="overview", processed=done, total_files=total, message=msg)

    overview_stats = await build_project_overview(
        project_id,
        project_root,
        use_llm=use_llm,
        progress=oprog,
    )
    logger.info("[%s] stage=overview done %s", job_id, overview_stats)
    if not _job_is_running(job_id):
        return

    workers = max(1, min(int(settings.parse_concurrency or 4), 16))
    _update_job(
        job_id,
        stage="entries",
        message=f"marking data-source entries concurrency={workers}",
        processed=0,
        total_files=1,
    )

    def enprog(done: int, total: int, msg: str) -> None:
        if not _job_is_running(job_id):
            return
        logger.info("[%s] stage=entries %s/%s %s", job_id, done, total, msg)
        _update_job(job_id, stage="entries", processed=done, total_files=total, message=msg)

    index = list_java_index(project_root)
    entry_stats = await discover_and_index_entries(
        project_id,
        project_root,
        index=index,
        progress=enprog,
        concurrency=workers,
    )
    entry_methods = entry_stats.pop("_entry_methods", []) or []
    index = entry_stats.pop("_index", None) or index
    logger.info(
        "[%s] stage=entries done files=%s methods=%s",
        job_id,
        entry_stats.get("entry_files"),
        entry_stats.get("entry_methods"),
    )
    if not _job_is_running(job_id):
        return

    flow_stats: dict = {}
    if settings.flow_analysis_enabled and use_llm and entry_methods:
        _update_job(
            job_id,
            stage="flow",
            message=f"recursive flow concurrency={settings.parse_concurrency}",
            processed=0,
            total_files=max(len(entry_methods), 1),
        )

        def fprog(done: int, total: int, msg: str) -> None:
            if not _job_is_running(job_id):
                return
            logger.info("[%s] stage=flow %s/%s %s", job_id, done, total, msg)
            _update_job(job_id, stage="flow", processed=done, total_files=total, message=msg)

        flow_stats = await recursive_flow_from_entries(
            project_id,
            project_root,
            entry_methods,
            index=index,
            progress=fprog,
            concurrency=settings.parse_concurrency,
        )
        logger.info("[%s] stage=flow done %s", job_id, flow_stats)
    else:
        reason = "no entries" if not entry_methods else "disabled"
        logger.info("[%s] stage=flow skipped (%s)", job_id, reason)
        flow_stats = {"skipped": reason}

    if not _job_is_running(job_id):
        return

    route_stats: dict = {}
    try:
        _update_job(
            job_id,
            stage="entry_route",
            message="building entry_route classification",
            processed=0,
            total_files=1,
        )
        from app.analysis.entry_route import rebuild_entry_route

        route_stats = await asyncio.to_thread(rebuild_entry_route, project_id)
        logger.info("[%s] stage=entry_route done %s", job_id, route_stats)
    except Exception:
        logger.exception("[%s] stage=entry_route failed (non-fatal)", job_id)
        route_stats = {"error": "rebuild_failed"}

    if not _job_is_running(job_id):
        return

    done_msg = _with_elapsed(
        job_id,
        f"ok overview={overview_stats} entries={entry_stats} "
        f"flow={flow_stats} entry_route={route_stats}",
    )
    _update_job(
        job_id,
        status="succeeded",
        stage="done",
        message=done_msg,
        processed=_job_processed(job_id),
    )
    logger.info("[%s] SUCCEEDED (legacy) %s", job_id, done_msg)
    try:
        from app.agent.project_notebook import build_and_save_notebook

        await asyncio.to_thread(build_and_save_notebook, project_id, project_root)
    except Exception:
        logger.exception("[%s] project notebook build failed (non-fatal)", job_id)


def _job_processed(job_id: str) -> int:
    with get_session() as session:
        row = session.get(IngestJobORM, job_id)
        return row.processed if row else 0


def create_ingest_job(
    project_id: str | None = None,
    path: str | None = None,
    tagging: bool | None = None,
    full_refresh: bool = False,
) -> IngestJobORM:
    settings = get_settings()
    init_db()
    project_id = project_id or settings.project_id
    root = Path(path) if path else settings.project_root_path
    if not root.is_absolute():
        from app.core.config import ROOT_DIR

        root = (ROOT_DIR / root).resolve()
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
        result = IngestJobORM(
            id=job.id,
            project_id=job.project_id,
            status=job.status,
            stage=job.stage,
            total_files=job.total_files,
            processed=job.processed,
            message=job.message,
            error=job.error,
        )

    do_tag = settings.tagging_enabled if tagging is None else tagging
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(
                run_ingest_job(
                    job_id,
                    project_id,
                    root,
                    tagging=do_tag,
                    full_refresh=full_refresh,
                )
            )
        else:
            loop.run_until_complete(
                run_ingest_job(
                    job_id,
                    project_id,
                    root,
                    tagging=do_tag,
                    full_refresh=full_refresh,
                )
            )
    except RuntimeError:
        asyncio.run(
            run_ingest_job(
                job_id,
                project_id,
                root,
                tagging=do_tag,
                full_refresh=full_refresh,
            )
        )
    return result


def _snapshot_job(row: IngestJobORM) -> IngestJobORM:
    return IngestJobORM(
        id=row.id,
        project_id=row.project_id,
        status=row.status,
        stage=row.stage,
        total_files=row.total_files or 0,
        processed=row.processed or 0,
        message=row.message,
        error=row.error,
    )


def get_job(job_id: str) -> IngestJobORM | None:
    with get_session() as session:
        row = session.get(IngestJobORM, job_id)
        if not row:
            return None
        return _snapshot_job(row)
