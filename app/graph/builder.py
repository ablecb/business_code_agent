from __future__ import annotations

import logging
import time
from pathlib import Path

from app.db.neo4j_client import get_neo4j
from app.graph.async_builder import build_async_graph
from app.graph.sync_builder import build_sync_graph
from app.parser.models import ParsedType

logger = logging.getLogger("app.graph")


def build_call_graph(
    project_id: str,
    types: list[ParsedType],
    *,
    project_root: Path | None = None,
    full_refresh: bool = True,
) -> dict[str, dict[str, int]]:
    t0 = time.time()
    client = get_neo4j()
    client.ensure_constraints()
    if full_refresh:
        logger.info("clearing neo4j project=%s …", project_id)
        client.clear_project(project_id)
    logger.info("building sync call graph types=%s …", len(types))
    sync_stats = build_sync_graph(client, project_id, types)
    logger.info("sync graph done %s", sync_stats)
    logger.info("building async (mq/feign) graph …")
    async_stats = build_async_graph(client, project_id, types, project_root=project_root)
    logger.info("async graph done %s (%.1fs)", async_stats, time.time() - t0)
    return {"sync": sync_stats, "async": async_stats}


def query_flow(entry: str, project_id: str, depth: int = 5) -> dict:
    try:
        client = get_neo4j()
        if not client.verify():
            return {"entry": entry, "nodes": [], "edges": []}
    except Exception:
        return {"entry": entry, "nodes": [], "edges": []}
    cypher = """
    MATCH (start)
    WHERE start.project_id = $project_id
      AND (
        start.fqcn CONTAINS $entry
        OR start.name = $entry
        OR start.signature CONTAINS $entry
        OR start.path CONTAINS $entry
        OR start.simple_name = $entry
      )
    WITH start LIMIT 5
    CALL {
      WITH start
      MATCH path = (start)-[:CALLS|ASYNC_CALLS|TRIGGERS|FEIGN_CALLS|IMPLEMENTS_API|SCHEDULES|PUBLISHES|CONSUMES|DECLARES*1..%d]-(n)
      RETURN path
      LIMIT 200
    }
    WITH start, collect(path) AS paths
    WITH start,
         reduce(ns = [start], p IN paths | ns + nodes(p)) AS all_nodes,
         reduce(rs = [], p IN paths | rs + relationships(p)) AS all_rels
    UNWIND all_nodes AS node
    WITH start, collect(DISTINCT node) AS nodes, all_rels
    UNWIND all_rels AS rel
    WITH nodes, collect(DISTINCT {
      type: type(rel),
      start: startNode(rel).id,
      end: endNode(rel).id,
      properties: properties(rel)
    }) AS edges
    RETURN nodes, edges
    """ % max(1, min(depth, 8))

    rows = client.run(cypher, {"project_id": project_id, "entry": entry})
    if not rows:
        # fallback: return matching nodes only
        rows = client.run(
            """
            MATCH (n)
            WHERE n.project_id=$project_id AND (
              n.fqcn CONTAINS $entry OR n.name CONTAINS $entry OR n.simple_name CONTAINS $entry
            )
            RETURN collect(n) AS nodes, [] AS edges
            LIMIT 1
            """,
            {"project_id": project_id, "entry": entry},
        )
    if not rows:
        return {"entry": entry, "nodes": [], "edges": []}
    row = rows[0]
    nodes = []
    for n in row.get("nodes") or []:
        nodes.append(
            {
                "id": n.get("id"),
                "labels": list(getattr(n, "labels", []) or n.get("labels", []) or []),
                "properties": dict(n) if not hasattr(n, "items") else {k: v for k, v in dict(n).items()},
            }
        )
    # neo4j Node objects
    from neo4j.graph import Node

    clean_nodes = []
    for n in row.get("nodes") or []:
        if isinstance(n, Node):
            clean_nodes.append(
                {
                    "id": n.get("id"),
                    "labels": list(n.labels),
                    "properties": dict(n),
                }
            )
        elif isinstance(n, dict):
            clean_nodes.append(
                {
                    "id": n.get("id"),
                    "labels": n.get("labels", []),
                    "properties": n,
                }
            )
    edges = row.get("edges") or []
    return {"entry": entry, "nodes": clean_nodes, "edges": edges}
