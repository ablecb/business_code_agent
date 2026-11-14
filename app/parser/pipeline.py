from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from sqlalchemy import text

from app.core.config import get_settings
from app.db.mysql import get_session
from app.db.retry import run_with_deadlock_retry
from app.parser.java_parser import parse_java_file
from app.parser.mapper_xml_parser import parse_mapper_xml_file
from app.parser.models import ParseResult, ParsedType
from app.parser.scanner import scan_source_files
from app.parser.symbol_store import clear_symbols, persist_types, upsert_project

logger = logging.getLogger("app.parser")


class ParseCancelled(RuntimeError):
    """Raised when an ingest job was superseded / cancelled mid-parse."""


def _parse_one(fp: Path, project_root: Path) -> ParseResult:
    if fp.suffix.lower() == ".xml":
        return parse_mapper_xml_file(fp, project_root)
    return parse_java_file(fp, project_root)


def _persist_parsed(project_id: str, types: list[ParsedType]) -> None:
    if not types:
        return

    def _write() -> None:
        with get_session() as session:
            try:
                persist_types(session, project_id, types)
            except Exception:
                session.rollback()
                raise

    run_with_deadlock_retry(_write)


def parse_project(
    project_id: str,
    project_root: Path,
    *,
    full_refresh: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    concurrency: int | None = None,
) -> tuple[list[ParsedType], list[str]]:
    settings = get_settings()
    # AI parse is LLM/API-bound — keep concurrency modest.
    workers = max(1, int(concurrency or settings.parse_concurrency or 4))
    workers = max(1, min(workers, 16))

    t0 = time.time()
    logger.info("scan sources under %s …", project_root)
    files = scan_source_files(project_root)
    logger.info("scan done files=%s (%.1fs)", len(files), time.time() - t0)
    all_types: list[ParsedType] = []
    errors: list[str] = []
    lock_key = f"bca_ingest_{project_id}"

    # Keep GET_LOCK on one connection for the whole job; writers use other sessions.
    with get_session() as lock_session:
        logger.info("acquiring ingest lock %s (wait up to 120s)…", lock_key)
        got = lock_session.execute(
            text("SELECT GET_LOCK(:k, 120)"), {"k": lock_key}
        ).scalar()
        if not got:
            raise RuntimeError(
                f"获取项目解析锁超时（可能仍有旧任务在写库）: {project_id}"
            )
        logger.info("ingest lock acquired")
        try:
            upsert_project(lock_session, project_id, project_id, str(project_root))
            lock_session.commit()

            if full_refresh:
                if should_continue and not should_continue():
                    raise ParseCancelled(project_id)
                logger.info("full_refresh: clearing old symbols/tags…")
                t_clear = time.time()
                deleted = clear_symbols(
                    lock_session,
                    project_id,
                    should_continue=should_continue,
                )
                logger.info("cleared symbols=%s (%.1fs)", deleted, time.time() - t_clear)
                if should_continue and not should_continue():
                    raise ParseCancelled(project_id)

            total = len(files)
            logger.info(
                "AI-parsing %s files with concurrency=%s (no javalang) …",
                total,
                workers,
            )
            done = 0

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_parse_one, fp, project_root): fp for fp in files
                }
                for fut in as_completed(futures):
                    if should_continue and not should_continue():
                        for pending in futures:
                            pending.cancel()
                        raise ParseCancelled(project_id)

                    fp = futures[fut]
                    try:
                        parsed = fut.result()
                    except Exception as e:
                        errors.append(f"{fp}: {e}")
                        done += 1
                        continue

                    errors.extend(parsed.errors)
                    if parsed.types:
                        all_types.extend(parsed.types)
                        _persist_parsed(project_id, parsed.types)

                    done += 1
                    if progress and (done % 25 == 0 or done == total):
                        progress(done, total, f"parsed {fp.name}")
                    elif done % 100 == 0:
                        logger.info("parse progress %s/%s %s", done, total, fp.name)
        finally:
            try:
                lock_session.execute(text("SELECT RELEASE_LOCK(:k)"), {"k": lock_key})
                lock_session.commit()
                logger.info("ingest lock released")
            except Exception:
                lock_session.rollback()

    logger.info(
        "parse_project finished types=%s errors=%s concurrency=%s elapsed=%.1fs",
        len(all_types),
        len(errors),
        workers,
        time.time() - t0,
    )
    return all_types, errors
