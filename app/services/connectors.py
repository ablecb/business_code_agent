"""Settings / ad-hoc DB connectors for desktop client_agent actions."""
from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text

from app.agent.remote_protocol import MAX_LOG_CHARS
from app.core.config import get_settings

logger = logging.getLogger("app.connectors")

MAX_ROWS = 200
DEFAULT_OUTPUT_FIELDS = [
    "id",
    "project_id",
    "symbol_id",
    "fqcn",
    "module_name",
    "tags",
    "text",
]


def _truncate_text(s: str, n: int = MAX_LOG_CHARS) -> str:
    t = str(s or "")
    if len(t) <= n:
        return t
    half = n // 2
    return t[:half] + "\n…[truncated]…\n" + t[-half:]


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


def _format_rows(rows: list[dict[str, Any]], *, row_count: int | None = None) -> str:
    n = row_count if row_count is not None else len(rows)
    payload = {"row_count": n, "rows": rows}
    return _truncate_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2))


def _mysql_url(cfg: dict[str, Any]) -> str:
    settings = get_settings()
    host = str(cfg.get("host") or settings.mysql_host)
    port = int(cfg.get("port") or settings.mysql_port or 3306)
    user = quote_plus(str(cfg.get("user") or settings.mysql_user))
    password = quote_plus(str(cfg.get("password") if cfg.get("password") is not None else settings.mysql_password))
    database = str(cfg.get("database") or settings.mysql_database)
    return (
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    )


def execute_mysql(
    sql: str,
    *,
    mysql: dict[str, Any] | None = None,
    max_rows: int = MAX_ROWS,
) -> dict[str, Any]:
    """Run one SQL statement. Settings when mysql omitted; else temporary connection."""
    raw = (sql or "").strip()
    if not raw:
        return {"ok": False, "error": "empty sql", "row_count": 0, "rows": [], "stdout": ""}

    # Single statement only (reject multi to avoid injection chaining).
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if len(parts) > 1:
        return {
            "ok": False,
            "error": "仅允许单条 SQL（去掉多余分号）",
            "row_count": 0,
            "rows": [],
            "stdout": "",
        }
    stmt = parts[0]

    use_temp = bool(mysql)
    engine = None
    try:
        if use_temp:
            engine = create_engine(
                _mysql_url(mysql or {}),
                pool_pre_ping=True,
                future=True,
                connect_args={
                    "connect_timeout": 5,
                    "read_timeout": 30,
                    "write_timeout": 30,
                },
            )
            conn_cm = engine.connect()
        else:
            from app.db.mysql import get_engine

            conn_cm = get_engine().connect()

        with conn_cm as conn:
            result = conn.execute(text(stmt))
            if result.returns_rows:
                keys = list(result.keys())
                fetched = result.fetchmany(max_rows + 1)
                truncated = len(fetched) > max_rows
                fetched = fetched[:max_rows]
                rows = [_json_safe(dict(zip(keys, row))) for row in fetched]
                if truncated:
                    rows.append({"_note": f"truncated to {max_rows} rows"})
                conn.commit()
                stdout = _format_rows(rows, row_count=len(rows) - (1 if truncated else 0))
                return {
                    "ok": True,
                    "error": None,
                    "row_count": len(rows) - (1 if truncated else 0),
                    "rows": rows,
                    "stdout": stdout,
                    "source": "temp_mysql" if use_temp else "settings",
                }
            # DML / DDL
            try:
                conn.commit()
            except Exception:
                pass
            affected = result.rowcount if result.rowcount is not None else -1
            stdout = f"ok affected_rows={affected}"
            return {
                "ok": True,
                "error": None,
                "row_count": affected if affected >= 0 else 0,
                "rows": [],
                "stdout": stdout,
                "affected_rows": affected,
                "source": "temp_mysql" if use_temp else "settings",
            }
    except Exception as e:
        logger.exception("mysql connector failed")
        err = str(e)
        return {
            "ok": False,
            "error": err,
            "row_count": 0,
            "rows": [],
            "stdout": "",
            "stderr": err,
        }
    finally:
        if engine is not None:
            engine.dispose()


def execute_cypher(
    query: str,
    *,
    params: dict[str, Any] | None = None,
    max_rows: int = MAX_ROWS,
) -> dict[str, Any]:
    raw = (query or "").strip()
    if not raw:
        return {"ok": False, "error": "empty query", "row_count": 0, "rows": [], "stdout": ""}
    try:
        from app.db.neo4j_client import get_neo4j

        client = get_neo4j()
        all_rows = client.run(raw, params or {})
        truncated = len(all_rows) > max_rows
        rows = [_json_safe(r) for r in all_rows[:max_rows]]
        if truncated:
            rows.append({"_note": f"truncated to {max_rows} rows"})
        stdout = _format_rows(rows, row_count=len(all_rows[:max_rows]))
        return {
            "ok": True,
            "error": None,
            "row_count": len(all_rows[:max_rows]),
            "rows": rows,
            "stdout": stdout,
            "source": "settings",
        }
    except Exception as e:
        logger.exception("neo4j connector failed")
        err = str(e)
        return {
            "ok": False,
            "error": err,
            "row_count": 0,
            "rows": [],
            "stdout": "",
            "stderr": err,
        }


def execute_milvus(
    *,
    op: str = "search",
    collection: str | None = None,
    filter: str | None = None,
    expr: str | None = None,
    limit: int | None = None,
    rows: list[dict[str, Any]] | None = None,
    embeddings: list[list[float]] | None = None,
    ids: list[str] | None = None,
    max_rows: int = MAX_ROWS,
) -> dict[str, Any]:
    """search (query/filter) or mutate (upsert/delete) against Settings Milvus."""
    try:
        from app.db.milvus_client import get_milvus

        store = get_milvus()
        settings = get_settings()
        coll = (collection or store.collection_name or settings.milvus_collection).strip()
        lim = int(limit or max_rows)
        lim = max(1, min(lim, max_rows))
        filt = (filter or expr or "").strip()
        op_l = (op or "search").strip().lower()

        if op_l in {"search", "query", "list"}:
            if store._use_memory or store._client is None:
                mem = list(store._memory)
                if filt and "project_id" in filt:
                    # best-effort in-memory filter for project_id == "x"
                    import re

                    m = re.search(r'project_id\s*==\s*"([^"]+)"', filt)
                    if m:
                        pid = m.group(1)
                        mem = [r for r in mem if str(r.get("project_id")) == pid]
                out_rows = []
                for r in mem[:lim]:
                    out_rows.append(
                        {
                            k: r.get(k)
                            for k in DEFAULT_OUTPUT_FIELDS
                            if k in r or k == "id"
                        }
                    )
                stdout = _format_rows(out_rows)
                return {
                    "ok": True,
                    "error": None,
                    "row_count": len(out_rows),
                    "rows": out_rows,
                    "stdout": stdout,
                    "source": "memory_fallback",
                    "collection": coll,
                }

            store._ensure_loaded()
            # Switch collection temporarily if requested
            prev = store.collection_name
            store.collection_name = coll
            try:
                q_filter = filt or 'id != ""'
                raw = store._client.query(
                    collection_name=coll,
                    filter=q_filter,
                    limit=lim,
                    output_fields=[f for f in DEFAULT_OUTPUT_FIELDS if f != "id"]
                    + (["id"] if "id" not in DEFAULT_OUTPUT_FIELDS else []),
                )
            finally:
                store.collection_name = prev
            out_rows = [_json_safe(dict(r)) for r in (raw or [])[:lim]]
            stdout = _format_rows(out_rows)
            return {
                "ok": True,
                "error": None,
                "row_count": len(out_rows),
                "rows": out_rows,
                "stdout": stdout,
                "source": "settings",
                "collection": coll,
            }

        if op_l == "upsert":
            data_rows = list(rows or [])
            embs = list(embeddings or [])
            if not data_rows:
                return {
                    "ok": False,
                    "error": "upsert requires rows",
                    "row_count": 0,
                    "rows": [],
                    "stdout": "",
                }
            if len(embs) != len(data_rows):
                return {
                    "ok": False,
                    "error": f"embeddings length {len(embs)} != rows {len(data_rows)}",
                    "row_count": 0,
                    "rows": [],
                    "stdout": "",
                }
            prev = store.collection_name
            store.collection_name = coll
            try:
                store.upsert(data_rows, embs)
            finally:
                store.collection_name = prev
            msg = f"upserted {len(data_rows)} rows into {coll}"
            return {
                "ok": True,
                "error": None,
                "row_count": len(data_rows),
                "rows": [],
                "stdout": msg,
                "source": "settings",
                "collection": coll,
            }

        if op_l == "delete":
            id_list = [str(x) for x in (ids or []) if x]
            if id_list:
                # Milvus delete by ids
                if store._use_memory or store._client is None:
                    before = len(store._memory)
                    id_set = set(id_list)
                    store._memory = [r for r in store._memory if str(r.get("id")) not in id_set]
                    deleted = before - len(store._memory)
                else:
                    store._client.delete(collection_name=coll, ids=id_list)
                    deleted = len(id_list)
                msg = f"deleted ids={len(id_list)} collection={coll} (reported={deleted})"
            elif filt:
                if store._use_memory or store._client is None:
                    return {
                        "ok": False,
                        "error": "memory fallback only supports delete by ids",
                        "row_count": 0,
                        "rows": [],
                        "stdout": "",
                    }
                store._client.delete(collection_name=coll, filter=filt)
                msg = f"deleted by filter={filt!r} collection={coll}"
            else:
                return {
                    "ok": False,
                    "error": "delete requires ids or filter",
                    "row_count": 0,
                    "rows": [],
                    "stdout": "",
                }
            return {
                "ok": True,
                "error": None,
                "row_count": 0,
                "rows": [],
                "stdout": msg,
                "source": "settings",
                "collection": coll,
            }

        return {
            "ok": False,
            "error": f"unsupported milvus op: {op}",
            "row_count": 0,
            "rows": [],
            "stdout": "",
        }
    except Exception as e:
        logger.exception("milvus connector failed")
        err = str(e)
        return {
            "ok": False,
            "error": err,
            "row_count": 0,
            "rows": [],
            "stdout": "",
            "stderr": err,
        }
