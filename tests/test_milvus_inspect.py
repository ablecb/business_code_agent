"""Dump ALL Milvus rows for the current project (full text, no truncation).

Run from repo root — 先停掉占用 milvus db 的 API/解析进程（Lite 单进程锁）:

  python tests/test_milvus_inspect.py --all
  python tests/test_milvus_inspect.py --all --out data/milvus_all.txt
  python tests/test_milvus_inspect.py --all --json data/milvus_all.json

Default without --all still dumps a short sample (limit=30).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _short(text: str, n: int = 240) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def _fetch_all_rows(milvus, project_id: str, log) -> list[dict[str, Any]]:
    client = milvus._client
    coll = milvus.collection_name
    filt = f'project_id == "{project_id}"'
    rows: list[dict[str, Any]] = []
    milvus._ensure_loaded()
    offset = 0
    page = 200
    while True:
        batch = client.query(
            collection_name=coll,
            filter=filt,
            output_fields=["id", "project_id", "symbol_id", "fqcn", "tags", "text"],
            limit=page,
            offset=offset,
        )
        if not batch:
            break
        rows.extend(batch)
        log(f"  fetched {len(rows)} …")
        if len(batch) < page:
            break
        offset += page
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect / dump Milvus rows")
    parser.add_argument("--project-id", default="", help="override PROJECT_ID")
    parser.add_argument(
        "--all",
        action="store_true",
        help="dump ALL rows with full text (recommended)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help="max rows when not using --all (0 = all)",
    )
    parser.add_argument("--keyword", default="", help="filter text/fqcn/tags contains")
    parser.add_argument("--q", default="", help="semantic search query")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--out", default="data/milvus_all.txt", help="text report path")
    parser.add_argument(
        "--json",
        default="data/milvus_all.json",
        help="JSON dump path (with --all, or empty to skip)",
    )
    parser.add_argument(
        "--no-json",
        action="store_true",
        help="do not write JSON",
    )
    args = parser.parse_args()

    from app.core.config import get_settings
    from app.db.milvus_client import get_milvus, reset_milvus

    get_settings.cache_clear()
    reset_milvus()
    settings = get_settings()
    project_id = (args.project_id or settings.project_id).strip()
    milvus = get_milvus()

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg, flush=True)
        lines.append(msg)

    dump_all = bool(args.all) or args.limit == 0
    limit = None if dump_all else max(1, args.limit)

    log("=" * 72)
    log("MILVUS DUMP" + (" (ALL)" if dump_all else f" (limit={limit})"))
    log("=" * 72)
    log(f"mode: {milvus.mode}")
    log(f"uri/path: {milvus._resolve_uri()}")
    log(f"collection: {milvus.collection_name}")
    log(f"config dim: {milvus.dim}")
    log(f"project_id: {project_id}")
    log(f"verify_ok: {milvus.verify()}")

    if milvus._use_memory or milvus._client is None:
        log("\n! 打不开 Milvus Lite 文件（多半是 API/解析进程占用锁）。")
        log("  请先停止服务，再执行：python tests/test_milvus_inspect.py --all")
        _write(args.out, lines)
        return 2

    try:
        info = milvus._client.describe_collection(milvus.collection_name)
        log("\n--- SCHEMA ---")
        fields = info.get("fields") or info.get("schema", {}).get("fields") or []
        for f in fields:
            name = f.get("name") or f.get("field_name")
            params = f.get("params") or f.get("type_params") or {}
            dim = params.get("dim") or f.get("dim")
            dtype = f.get("type") or f.get("dtype") or f.get("data_type")
            log(f"  - {name}: {dtype}" + (f" dim={dim}" if dim else ""))
        log(f"collection embedding dim: {milvus._collection_dim()}")
    except Exception as e:
        log(f"! describe failed: {e}")

    log("\n--- FETCH ---")
    try:
        rows = _fetch_all_rows(milvus, project_id, log)
    except Exception as e:
        log(f"! query failed: {e}")
        rows = []
    log(f"total rows: {len(rows)}")

    tag_counts: dict[str, int] = {}
    flow_n = overview_n = 0
    for r in rows:
        for t in [
            x.strip()
            for x in (r.get("tags") or "").replace("，", ",").split(",")
            if x.strip()
        ]:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        fq, sid = r.get("fqcn") or "", r.get("symbol_id") or ""
        if fq.startswith("__flow__") or sid.startswith("flow:"):
            flow_n += 1
        if "overview" in sid.lower() or "Overview" in (r.get("tags") or ""):
            overview_n += 1
    log(f"flow-like: {flow_n}  overview-like: {overview_n}")
    if tag_counts:
        log("top tags:")
        for t, c in sorted(tag_counts.items(), key=lambda x: -x[1])[:30]:
            log(f"  {c:4d}  {t}")

    dump = rows
    kw = (args.keyword or "").strip().lower()
    if kw:
        dump = [
            r
            for r in rows
            if kw in (r.get("text") or "").lower()
            or kw in (r.get("fqcn") or "").lower()
            or kw in (r.get("tags") or "").lower()
            or kw in (r.get("symbol_id") or "").lower()
        ]
        log(f"keyword={kw!r} matched: {len(dump)}")

    if limit is not None:
        dump = dump[:limit]

    log(f"\n--- ROWS ({len(dump)}) ---")
    # Console: short preview; file: full text
    file_only: list[str] = []
    for i, r in enumerate(dump, 1):
        block = [
            f"\n[{i}/{len(dump)}] id={r.get('id')}",
            f"symbol_id: {r.get('symbol_id')}",
            f"fqcn: {r.get('fqcn')}",
            f"tags: {r.get('tags')}",
            "text:",
            str(r.get("text") or ""),
            "-" * 40,
        ]
        file_only.extend(block)
        # stdout preview
        print(
            f"[{i}/{len(dump)}] {r.get('fqcn')} | {_short(r.get('text') or '', 120)}",
            flush=True,
        )

    lines.extend(file_only)

    q = (args.q or "").strip()
    if q:
        log(f"\n--- SEMANTIC SEARCH q={q!r} ---")
        try:
            from app.llm.factory import get_embedding_provider, reset_provider_caches

            reset_provider_caches()
            emb = get_embedding_provider().embed([q])[0]
            hits = milvus.search(emb, project_id=project_id, top_k=args.top_k)
            for i, h in enumerate(hits, 1):
                log(f"\n[{i}] score={h.get('score')}")
                log(f"  symbol_id={h.get('symbol_id')}")
                log(f"  fqcn={h.get('fqcn')}")
                log(f"  tags={h.get('tags')}")
                log(f"  text:\n{h.get('text') or ''}")
        except Exception as e:
            log(f"! semantic search failed: {e}")

    out_path = _write(args.out, lines)
    log(f"\nwrote text: {out_path}")

    if dump_all and not args.no_json and args.json:
        json_path = Path(args.json)
        if not json_path.is_absolute():
            json_path = ROOT / json_path
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "project_id": project_id,
            "collection": milvus.collection_name,
            "count": len(dump),
            "rows": [
                {
                    "id": r.get("id"),
                    "project_id": r.get("project_id"),
                    "symbol_id": r.get("symbol_id"),
                    "fqcn": r.get("fqcn"),
                    "tags": r.get("tags"),
                    "text": r.get("text"),
                }
                for r in dump
            ],
        }
        json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log(f"wrote json: {json_path}")

    return 0


def _write(rel: str, lines: list[str]) -> Path:
    path = Path(rel)
    if not path.is_absolute():
        path = ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
