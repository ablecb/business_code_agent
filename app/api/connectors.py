"""HTTP connectors for desktop whitelist actions (MySQL / Neo4j / Milvus)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.connectors import execute_cypher, execute_milvus, execute_mysql

router = APIRouter(prefix="/connectors", tags=["connectors"])


class MysqlConnectorRequest(BaseModel):
    sql: str = ""
    mysql: dict[str, Any] | None = None
    max_rows: int | None = None


class Neo4jConnectorRequest(BaseModel):
    query: str = ""
    params: dict[str, Any] | None = None
    max_rows: int | None = None


class MilvusConnectorRequest(BaseModel):
    op: str = "search"  # search | upsert | delete
    collection: str | None = None
    filter: str | None = None
    expr: str | None = None
    limit: int | None = None
    rows: list[dict[str, Any]] = Field(default_factory=list)
    embeddings: list[list[float]] = Field(default_factory=list)
    ids: list[str] = Field(default_factory=list)
    max_rows: int | None = None


class ConnectorResponse(BaseModel):
    ok: bool
    error: str | None = None
    row_count: int = 0
    rows: list[dict[str, Any]] = Field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    meta: dict[str, Any] = Field(default_factory=dict)


def _to_response(result: dict[str, Any]) -> ConnectorResponse:
    meta = {
        k: v
        for k, v in result.items()
        if k
        not in {
            "ok",
            "error",
            "row_count",
            "rows",
            "stdout",
            "stderr",
        }
    }
    return ConnectorResponse(
        ok=bool(result.get("ok")),
        error=result.get("error"),
        row_count=int(result.get("row_count") or 0),
        rows=list(result.get("rows") or []),
        stdout=str(result.get("stdout") or ""),
        stderr=str(result.get("stderr") or result.get("error") or ""),
        meta=meta,
    )


@router.post("/mysql", response_model=ConnectorResponse)
def connectors_mysql(body: MysqlConnectorRequest) -> ConnectorResponse:
    kwargs: dict[str, Any] = {"mysql": body.mysql}
    if body.max_rows is not None:
        kwargs["max_rows"] = int(body.max_rows)
    return _to_response(execute_mysql(body.sql, **kwargs))


@router.post("/neo4j", response_model=ConnectorResponse)
def connectors_neo4j(body: Neo4jConnectorRequest) -> ConnectorResponse:
    kwargs: dict[str, Any] = {"params": body.params}
    if body.max_rows is not None:
        kwargs["max_rows"] = int(body.max_rows)
    return _to_response(execute_cypher(body.query, **kwargs))


@router.post("/milvus", response_model=ConnectorResponse)
def connectors_milvus(body: MilvusConnectorRequest) -> ConnectorResponse:
    kwargs: dict[str, Any] = {
        "op": body.op,
        "collection": body.collection,
        "filter": body.filter,
        "expr": body.expr,
        "limit": body.limit,
        "rows": body.rows,
        "embeddings": body.embeddings,
        "ids": body.ids,
    }
    if body.max_rows is not None:
        kwargs["max_rows"] = int(body.max_rows)
    return _to_response(execute_milvus(**kwargs))
