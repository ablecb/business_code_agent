from __future__ import annotations

from typing import Any

from neo4j import GraphDatabase, Driver

from app.core.config import get_settings


class Neo4jClient:
    def __init__(self) -> None:
        settings = get_settings()
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            connection_timeout=3,
        )

    def close(self) -> None:
        self._driver.close()

    def verify(self) -> bool:
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def run(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    def write_tx(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        with self._driver.session() as session:
            session.execute_write(lambda tx: tx.run(cypher, params or {}))

    def clear_project(self, project_id: str) -> None:
        self.write_tx(
            """
            MATCH (n {project_id: $project_id})
            DETACH DELETE n
            """,
            {"project_id": project_id},
        )

    def clear_all(self) -> int:
        """Full wipe: detach-delete every node. Returns deleted node count (best-effort)."""
        before = self.run("MATCH (n) RETURN count(n) AS n")
        n = int((before[0] or {}).get("n") or 0) if before else 0
        self.write_tx("MATCH (n) DETACH DELETE n")
        return n

    def ensure_constraints(self) -> None:
        statements = [
            "CREATE CONSTRAINT method_id IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT class_id IF NOT EXISTS FOR (c:Class) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT queue_id IF NOT EXISTS FOR (q:Queue) REQUIRE q.id IS UNIQUE",
            "CREATE CONSTRAINT endpoint_id IF NOT EXISTS FOR (e:Endpoint) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT feign_id IF NOT EXISTS FOR (f:FeignAPI) REQUIRE f.id IS UNIQUE",
            "CREATE CONSTRAINT job_id IF NOT EXISTS FOR (j:Job) REQUIRE j.id IS UNIQUE",
            "CREATE CONSTRAINT external_id IF NOT EXISTS FOR (x:ExternalService) REQUIRE x.id IS UNIQUE",
        ]
        for stmt in statements:
            try:
                self.write_tx(stmt)
            except Exception:
                pass


_neo4j: Neo4jClient | None = None


def get_neo4j() -> Neo4jClient:
    global _neo4j
    if _neo4j is None:
        _neo4j = Neo4jClient()
    return _neo4j
