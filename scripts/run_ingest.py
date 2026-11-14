#!/usr/bin/env python
"""CLI: python -m scripts.run_ingest [--no-tagging] [--full-refresh]"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings
from app.core.utils import new_id
from app.db.mysql import get_session, init_db
from app.models.orm import IngestJobORM
from app.services.ingest import get_job, run_ingest_job


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Java project into MySQL/Neo4j/Milvus")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--path", default=None)
    parser.add_argument("--no-tagging", action="store_true")
    parser.add_argument("--full-refresh", action="store_true")
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    project_id = args.project_id or settings.project_id
    root = Path(args.path) if args.path else settings.project_root_path
    if not root.is_absolute():
        root = (ROOT / root).resolve()

    job_id = new_id("job_")
    with get_session() as session:
        session.add(
            IngestJobORM(
                id=job_id,
                project_id=project_id,
                status="queued",
                stage="queued",
                message=f"cli ingest {root}",
            )
        )

    print(f"job={job_id} project={project_id} root={root}")
    await run_ingest_job(
        job_id,
        project_id,
        root,
        tagging=not args.no_tagging,
        full_refresh=args.full_refresh,
    )
    job = get_job(job_id)
    print(f"status={job.status} stage={job.stage} message={job.message}")
    if job.error:
        print(job.error)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
