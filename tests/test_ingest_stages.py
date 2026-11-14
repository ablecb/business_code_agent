"""离线测试 parse / graph / tagging。PyCharm 里直接运行本文件的 main。"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("EMBEDDING_PROVIDER", "hash")

# 先用小数量确认能跑通；确认没问题再改大或改成 None（全量）
LIMIT = None


def log(msg: str) -> None:
    print(msg, flush=True)


def _project() -> tuple[str, Path]:
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    settings = get_settings()
    root = settings.project_root_path
    log(f"[init] mysql={settings.mysql_host}:{settings.mysql_port}/{settings.mysql_database}")
    log(f"[init] project_root={root}")
    log("[init] 连接 MySQL / 建表中…（连不上会卡几秒后报错）")
    t0 = time.time()
    init_db()
    log(f"[init] MySQL OK ({time.time() - t0:.1f}s)")
    return settings.project_id, root


def run_parse():
    from app.db.mysql import get_session
    from app.parser.java_parser import parse_java_file
    from app.parser.mapper_xml_parser import parse_mapper_xml_file
    from app.parser.scanner import scan_java_files, scan_mapper_xml_files
    from app.parser.symbol_store import (
        clear_symbols,
        count_symbols,
        persist_types,
        upsert_project,
    )

    project_id, root = _project()

    log("[parse] 扫描 .java / Mapper XML…")
    t0 = time.time()
    java_files = scan_java_files(root)
    xml_files = scan_mapper_xml_files(root)
    log(
        f"[parse] 扫描完成 java={len(java_files)} mapperXml={len(xml_files)} "
        f"({time.time() - t0:.1f}s)"
    )
    files = java_files + xml_files
    if LIMIT is not None:
        # keep some xml even when limiting java
        java_part = java_files[:LIMIT]
        files = java_part + xml_files
        log(f"[parse] LIMIT={LIMIT} java，本次解析 {len(files)} 个文件（含全部 mapper xml）")

    types = []
    with get_session() as session:
        log("[parse] upsert project…")
        upsert_project(session, project_id, project_id, str(root))
        session.commit()

        n = count_symbols(session, project_id)
        log(f"[parse] 库里已有 symbol={n}，开始分批删除…")
        t1 = time.time()
        deleted = clear_symbols(session, project_id)
        session.commit()
        log(f"[parse] 清理完成 deleted={deleted} ({time.time() - t1:.1f}s)")

        for i, fp in enumerate(files, start=1):
            log(f"[parse] ({i}/{len(files)}) 开始解析 {fp}")
            t2 = time.time()
            try:
                if fp.suffix.lower() == ".xml":
                    parsed = parse_mapper_xml_file(fp, root)
                else:
                    parsed = parse_java_file(fp, root)
            except Exception as e:
                log(f"[parse] 失败 {fp.name}: {e}")
                continue
            if parsed.types:
                persist_types(session, project_id, parsed.types)
                types.extend(parsed.types)
            session.commit()
            log(
                f"[parse] ({i}/{len(files)}) 完成 "
                f"types+={len(parsed.types)} total_types={len(types)} "
                f"{time.time() - t2:.2f}s  {fp.name}"
            )

    log(f"[parse] done types={len(types)}")
    return project_id, root, types


def run_graph(project_id=None, root=None, types=None):
    from app.db.neo4j_client import get_neo4j
    from app.graph.builder import build_call_graph

    if types is None:
        project_id, root, types = run_parse()

    log("[graph] 检查 Neo4j…")
    if not get_neo4j().verify():
        raise RuntimeError("Neo4j 不可用，先 docker compose up -d")

    log(f"[graph] building types={len(types)}")
    stats = build_call_graph(project_id, types, project_root=root, full_refresh=True)
    log(f"[graph] done {stats}")
    return stats


def run_tagging(project_id=None):
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider
    from app.tagging.worker import tag_symbols

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    if project_id is None:
        project_id, _, _ = run_parse()

    settings = get_settings()
    log(f"[tagging] project={project_id} provider={settings.llm_provider}")
    stats = asyncio.run(
        tag_symbols(
            project_id,
            batch_size=settings.ingest_batch_size,
            use_llm=True,
        )
    )
    log(f"[tagging] done {stats}")
    return stats


def main():
    project_id, root, types = run_parse()
    run_graph(project_id, root, types)
    run_tagging(project_id)


if __name__ == "__main__":
    main()
