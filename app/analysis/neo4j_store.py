"""Neo4j writers for AI exploration: Entity / Class / Method / entries / edges."""

from __future__ import annotations

import logging
from typing import Any, Callable

from app.core.utils import stable_id
from app.db.neo4j_client import get_neo4j
from app.parser.java_parser import extract_publish_queues, extract_queue_from_annotation
from app.parser.models import ParsedMethod, ParsedType
from app.parser.symbol_store import symbol_id_for

logger = logging.getLogger("app.analysis.neo4j")

_KIND_LABEL = {
    "Entity": "Entity",
    "Controller": "Controller",
    "Listener": "Listener",
    "Job": "JobEntry",
    "Feign": "Feign",
    "Service": "Service",
    "Mapper": "Mapper",
}


def _neo4j_string_list(values: Any, *, limit: int = 20) -> list[str]:
    """Neo4j property arrays may only contain primitives — coerce dict/list to JSON."""
    import json

    out: list[str] = []
    for v in list(values or [])[:limit]:
        if isinstance(v, (dict, list)):
            out.append(json.dumps(v, ensure_ascii=False)[:800])
        elif v is None:
            continue
        else:
            out.append(str(v)[:800])
    return out


def clear_project_graph(project_id: str) -> None:
    client = get_neo4j()
    client.ensure_constraints()
    client.clear_project(project_id)


def clear_all_graph() -> int:
    """Delete every node/relationship in Neo4j. Returns prior node count."""
    client = get_neo4j()
    return client.clear_all()


def upsert_class_node(project_id: str, t: ParsedType) -> str:
    client = get_neo4j()
    cid = symbol_id_for(project_id, t.fqcn, None)
    kind = t.symbol_kind or "Class"
    client.write_tx(
        """
        MERGE (c:Class {id: $id})
        SET c.project_id=$project_id, c.fqcn=$fqcn, c.simple_name=$simple_name,
            c.kind=$kind, c.file_path=$file_path, c.module_name=$module_name
        """,
        {
            "id": cid,
            "project_id": project_id,
            "fqcn": t.fqcn,
            "simple_name": t.simple_name,
            "kind": kind,
            "file_path": t.file_path,
            "module_name": t.module_name,
        },
    )
    extra = _KIND_LABEL.get(kind)
    if extra:
        try:
            client.write_tx(f"MATCH (c:Class {{id: $id}}) SET c:{extra}", {"id": cid})
        except Exception:
            logger.debug("label %s failed for %s", extra, t.fqcn, exc_info=True)
    return cid


def upsert_method_node(
    project_id: str,
    t: ParsedType,
    m: ParsedMethod,
    *,
    important: bool | None = None,
    important_types: list[str] | None = None,
    flow_keys: list[str] | None = None,
    is_entry: bool | None = None,
    summary: str | None = None,
    data_flow_narrative: str | None = None,
    business_tables: list[str] | None = None,
    table_field_changes: list[dict[str, Any]] | str | None = None,
    status_transitions: list[str] | None = None,
    bfs_depth: int | None = None,
    store_narratives: bool | None = None,
) -> str:
    """Upsert Method + DECLARES. Narratives (summary/data_flow/…) default off — keep in MySQL."""
    from app.core.config import get_settings

    client = get_neo4j()
    mid = symbol_id_for(project_id, t.fqcn, m.signature)
    cid = symbol_id_for(project_id, t.fqcn, None)
    write_narratives = (
        bool(store_narratives)
        if store_narratives is not None
        else bool(get_settings().neo4j_store_narratives)
    )
    props: dict[str, Any] = {
        "project_id": project_id,
        "fqcn": t.fqcn,
        "simple_name": t.simple_name,
        "name": m.name,
        "signature": m.signature,
        "class_method": f"{t.simple_name}#{m.name}",
        "file_path": t.file_path,
        "kind": t.symbol_kind,
        "module_name": t.module_name or "",
        "start_line": m.start_line,
        "end_line": m.end_line,
    }
    if important is not None:
        props["important"] = important
    if important_types is not None:
        props["important_types"] = important_types
    if flow_keys is not None:
        props["flow_keys"] = flow_keys
    if is_entry is not None:
        props["is_entry"] = is_entry
    if write_narratives:
        if summary is not None:
            props["summary"] = summary[:1500]
        if data_flow_narrative is not None:
            props["data_flow"] = data_flow_narrative[:2000]
        if business_tables is not None:
            props["business_tables"] = _neo4j_string_list(business_tables, limit=30)
        if status_transitions is not None:
            # LLM may emit dicts; Neo4j props must be primitives / primitive arrays.
            props["status_transitions"] = _neo4j_string_list(status_transitions, limit=20)
        if table_field_changes is not None:
            import json

            if isinstance(table_field_changes, str):
                props["table_field_changes"] = table_field_changes[:3000]
            else:
                props["table_field_changes"] = json.dumps(
                    table_field_changes[:20], ensure_ascii=False
                )[:3000]
    if bfs_depth is not None:
        props["bfs_depth"] = int(bfs_depth)
    client.write_tx(
        """
        MERGE (m:Method {id: $id})
        SET m += $props
        WITH m
        MERGE (c:Class {id: $class_id})
        ON CREATE SET c.project_id=$project_id, c.fqcn=$fqcn, c.simple_name=$simple_name
        MERGE (c)-[:DECLARES]->(m)
        """,
        {
            "id": mid,
            "props": props,
            "class_id": cid,
            "project_id": project_id,
            "fqcn": t.fqcn,
            "simple_name": t.simple_name,
        },
    )
    return mid


def batch_merge_calls(
    project_id: str,
    edges: list[tuple[str, str, int | None, Any]],
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> int:
    """Batch merge CALLS edges in one UNWIND transaction per chunk.

    Avoids the O(N)-transaction overhead of calling merge_calls in a loop,
    which was extremely slow for thousands of CALLS edges
    (desktop-chat-95272f967734: stuck 3h at neo4j_calls stage).
    ``is_cancelled`` is checked between chunks so a cancelled ingest does
    not keep hammering Neo4j.
    """
    client = get_neo4j()
    rows: list[dict[str, Any]] = []
    for a, b, order, e in edges:
        props: dict[str, Any] = {"project_id": project_id}
        if order and int(order) > 0:
            props["order"] = int(order)
        src = getattr(e, "resolve_how", None) or "global-ast"
        props["source"] = src
        if getattr(e, "receiver", None):
            props["receiver"] = str(e.receiver)[:256]
        if getattr(e, "receiver_type", None):
            props["receiver_type"] = str(e.receiver_type)[:256]
        if getattr(e, "raw_call", None):
            props["raw_call"] = str(e.raw_call)[:512]
        if getattr(e, "resolve_how", None):
            props["resolve_how"] = str(e.resolve_how)[:64]
        rows.append({"a": a, "b": b, "props": props})

    if not rows:
        return 0

    chunk_size = 500
    total = 0
    for i in range(0, len(rows), chunk_size):
        if is_cancelled is not None and is_cancelled():
            logger.warning(
                "batch_merge_calls cancelled at %s/%s edges — aborting",
                i,
                len(rows),
            )
            break
        chunk = rows[i : i + chunk_size]
        client.write_tx(
            """
            UNWIND $rows AS row
            MATCH (a:Method {id: row.a})
            MATCH (b:Method {id: row.b})
            MERGE (a)-[r:CALLS]->(b)
            SET r += row.props
            """,
            {"rows": chunk},
        )
        total += len(chunk)
    return total


def merge_calls(
    project_id: str,
    caller_method_id: str,
    callee_method_id: str,
    *,
    order: int | None = None,
    source: str | None = None,
    flow_key: str | None = None,
    data_flow: str | None = None,
    receiver: str | None = None,
    receiver_type: str | None = None,
    raw_call: str | None = None,
    resolve_how: str | None = None,
) -> None:
    """Upsert CALLS. Optional order/source + AST resolve facts (receiver/how)."""
    client = get_neo4j()
    props: dict[str, Any] = {"project_id": project_id}
    if order is not None:
        props["order"] = int(order)
    if source:
        props["source"] = source
    if flow_key:
        props["flow_key"] = flow_key
    if data_flow:
        props["data_flow"] = data_flow[:1500]
    if receiver is not None:
        props["receiver"] = str(receiver)[:256]
    if receiver_type is not None:
        props["receiver_type"] = str(receiver_type)[:256]
    if raw_call is not None:
        props["raw_call"] = str(raw_call)[:512]
    if resolve_how is not None:
        props["resolve_how"] = str(resolve_how)[:64]
    client.write_tx(
        """
        MATCH (a:Method {id: $a})
        MATCH (b:Method {id: $b})
        MERGE (a)-[r:CALLS]->(b)
        SET r += $props
        """,
        {"a": caller_method_id, "b": callee_method_id, "props": props},
    )


def merge_follows(
    project_id: str,
    from_method_id: str,
    to_method_id: str,
    *,
    step: int,
    flow_key: str,
    title: str = "",
    data_flow: str = "",
) -> None:
    """Upsert FOLLOWS (dispatch / stage order) with step property."""
    client = get_neo4j()
    client.write_tx(
        """
        MATCH (a:Method {id: $a})
        MATCH (b:Method {id: $b})
        MERGE (a)-[r:FOLLOWS]->(b)
        SET r.project_id = $project_id,
            r.step = $step,
            r.flow_key = $flow_key,
            r.title = $title,
            r.data_flow = $data_flow
        """,
        {
            "a": from_method_id,
            "b": to_method_id,
            "project_id": project_id,
            "step": int(step),
            "flow_key": flow_key,
            "title": title or "",
            "data_flow": (data_flow or "")[:1500],
        },
    )


def upsert_endpoint(project_id: str, method_id: str, http_method: str, path: str) -> None:
    client = get_neo4j()
    eid = stable_id(project_id, "endpoint", http_method, path, method_id)
    client.write_tx(
        """
        MERGE (e:Endpoint {id: $id})
        SET e.project_id=$project_id, e.http_method=$http_method, e.path=$path
        WITH e
        MATCH (m:Method {id: $mid})
        MERGE (e)-[:IMPLEMENTS_API]->(m)
        """,
        {
            "id": eid,
            "project_id": project_id,
            "http_method": http_method,
            "path": path,
            "mid": method_id,
        },
    )


def upsert_queue(project_id: str, queue_name: str, *, method_id: str, rel: str) -> None:
    client = get_neo4j()
    qid = stable_id(project_id, "queue", queue_name)
    client.write_tx(
        f"""
        MERGE (q:Queue {{id: $id}})
        SET q.project_id=$project_id, q.name=$name
        WITH q
        MATCH (m:Method {{id: $mid}})
        MERGE (m)-[r:{rel}]->(q)
        SET r.project_id=$project_id
        """,
        {"id": qid, "project_id": project_id, "name": queue_name, "mid": method_id},
    )


def upsert_job(project_id: str, job_name: str, method_id: str) -> None:
    client = get_neo4j()
    jid = stable_id(project_id, "job", job_name)
    client.write_tx(
        """
        MERGE (j:Job {id: $id})
        SET j.project_id=$project_id, j.name=$name
        WITH j
        MATCH (m:Method {id: $mid})
        MERGE (j)-[:SCHEDULES]->(m)
        """,
        {"id": jid, "project_id": project_id, "name": job_name, "mid": method_id},
    )


def write_parsed_type_graph(project_id: str, t: ParsedType) -> dict[str, int]:
    """Upsert class + methods + endpoint/job/queue annotations into Neo4j."""
    upsert_class_node(project_id, t)
    methods = 0
    endpoints = 0
    jobs = 0
    queues = 0
    for m in t.methods:
        mid = upsert_method_node(project_id, t, m)
        methods += 1
        for hm in m.http_mappings or []:
            upsert_endpoint(
                project_id,
                mid,
                hm.get("http_method") or "REQUEST",
                hm.get("path") or "",
            )
            endpoints += 1
        for a in m.annotations:
            if a.name == "XxlJob":
                upsert_job(project_id, a.values.get("value") or m.name, mid)
                jobs += 1
            if a.name == "RabbitListener":
                for q in extract_queue_from_annotation(a):
                    upsert_queue(project_id, q, method_id=mid, rel="CONSUMES")
                    queues += 1
        for q in extract_publish_queues(m.raw_code or ""):
            upsert_queue(project_id, q, method_id=mid, rel="PUBLISHES")
            queues += 1
    return {"methods": methods, "endpoints": endpoints, "jobs": jobs, "queues": queues}
