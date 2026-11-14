from __future__ import annotations

from typing import Any

from app.db.neo4j_client import Neo4jClient


def upsert_method(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (m:Method {id: $id})
        SET m += $props
        """,
        {"id": props["id"], "props": props},
    )


def upsert_class(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (c:Class {id: $id})
        SET c += $props
        """,
        {"id": props["id"], "props": props},
    )


def upsert_queue(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (q:Queue {id: $id})
        SET q += $props
        """,
        {"id": props["id"], "props": props},
    )


def upsert_endpoint(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (e:Endpoint {id: $id})
        SET e += $props
        """,
        {"id": props["id"], "props": props},
    )


def upsert_feign(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (f:FeignAPI {id: $id})
        SET f += $props
        """,
        {"id": props["id"], "props": props},
    )


def upsert_job(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (j:Job {id: $id})
        SET j += $props
        """,
        {"id": props["id"], "props": props},
    )


def upsert_external(client: Neo4jClient, props: dict[str, Any]) -> None:
    client.write_tx(
        """
        MERGE (x:ExternalService {id: $id})
        SET x += $props
        """,
        {"id": props["id"], "props": props},
    )


def merge_rel(
    client: Neo4jClient,
    start_label: str,
    start_id: str,
    end_label: str,
    end_id: str,
    rel_type: str,
    props: dict[str, Any] | None = None,
) -> None:
    props = props or {}
    cypher = f"""
    MERGE (a:{start_label} {{id: $start_id}})
    MERGE (b:{end_label} {{id: $end_id}})
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += $props
    """
    client.write_tx(cypher, {"start_id": start_id, "end_id": end_id, "props": props})


def batch_write(client: Neo4jClient, statements: list[tuple[str, dict[str, Any]]]) -> None:
    with client._driver.session() as session:
        def _work(tx):
            for cypher, params in statements:
                tx.run(cypher, params)

        session.execute_write(_work)
