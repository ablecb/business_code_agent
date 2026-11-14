from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from app.core.utils import resolve_spring_placeholder, stable_id
from app.db.neo4j_client import Neo4jClient
from app.graph.writer import batch_write
from app.parser.java_parser import extract_publish_queues, extract_queue_from_annotation
from app.parser.models import ParsedType
from app.parser.symbol_store import symbol_id_for

CONST_ASSIGN_RE = re.compile(
    r"(?:public|private|protected|static|final|\s)+String\s+([A-Za-z_]\w*)\s*=\s*\"([^\"]+)\""
)


def _load_queue_constants(project_root: Path) -> dict[str, str]:
    """Map ConstantName / FQCN.Constant -> queue string from Java constants."""
    mapping: dict[str, str] = {}
    if not project_root.exists():
        return mapping
    for path in project_root.rglob("*.java"):
        if "target" in path.parts:
            continue
        name = path.name.lower()
        if "constant" not in name and "queue" not in name and "rabbit" not in name:
            # still scan for string constants used as queues
            pass
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for m in CONST_ASSIGN_RE.finditer(text):
            key, val = m.group(1), m.group(2)
            mapping[key] = val
            mapping[f"{path.stem}.{key}"] = val
    return mapping


def build_async_graph(
    client: Neo4jClient,
    project_id: str,
    types: list[ParsedType],
    project_root: Path | None = None,
) -> dict[str, int]:
    const_map = _load_queue_constants(project_root) if project_root else {}
    publishes: list[tuple[str, str]] = []  # method_id, queue
    consumes: list[tuple[str, str]] = []
    stmts: list[tuple[str, dict]] = []
    stats = {"queues": 0, "publishes": 0, "consumes": 0, "triggers": 0}

    queue_ids: set[str] = set()

    def ensure_queue(queue_name: str) -> str:
        qn = resolve_spring_placeholder(queue_name)
        # resolve constant reference
        if qn in const_map:
            qn = const_map[qn]
        elif qn.split(".")[-1] in const_map:
            qn = const_map[qn.split(".")[-1]]
        qid = stable_id(project_id, "queue", qn)
        if qid not in queue_ids:
            stmts.append(
                (
                    """
                    MERGE (q:Queue {id: $id})
                    SET q.project_id=$project_id, q.name=$name
                    """,
                    {"id": qid, "project_id": project_id, "name": qn},
                )
            )
            queue_ids.add(qid)
            stats["queues"] += 1
        return qid, qn

    for t in types:
        for m in t.methods:
            mid = symbol_id_for(project_id, t.fqcn, m.signature)
            # CONSUMES via @RabbitListener
            for ann in list(m.annotations) + list(t.annotations):
                for q in extract_queue_from_annotation(ann):
                    qid, qn = ensure_queue(q)
                    consumes.append((mid, qn))
                    stmts.append(
                        (
                            """
                            MATCH (m:Method {id: $mid})
                            MATCH (q:Queue {id: $qid})
                            MERGE (m)-[r:CONSUMES]->(q)
                            SET r.project_id=$project_id
                            """,
                            {"mid": mid, "qid": qid, "project_id": project_id},
                        )
                    )
                    stats["consumes"] += 1

            # PUBLISHES via convertAndSend
            for q in extract_publish_queues(m.raw_code or ""):
                resolved = const_map.get(q) or const_map.get(q.split(".")[-1]) or q
                # skip obvious non-queue identifiers
                if resolved.endswith("Template") or resolved in {"routingKey", "exchange"}:
                    continue
                qid, qn = ensure_queue(resolved)
                publishes.append((mid, qn))
                stmts.append(
                    (
                        """
                        MATCH (m:Method {id: $mid})
                        MATCH (q:Queue {id: $qid})
                        MERGE (m)-[r:PUBLISHES]->(q)
                        SET r.project_id=$project_id
                        """,
                        {"mid": mid, "qid": qid, "project_id": project_id},
                    )
                )
                stats["publishes"] += 1

            # @Async marker edges already in sync; ensure property
            if any(a.name == "Async" for a in m.annotations):
                stmts.append(
                    (
                        """
                        MATCH (m:Method {id: $mid})
                        SET m.async = true
                        """,
                        {"mid": mid},
                    )
                )

    # TRIGGERS: publisher -> consumer via same queue name
    pub_by_q: dict[str, list[str]] = defaultdict(list)
    con_by_q: dict[str, list[str]] = defaultdict(list)
    for mid, qn in publishes:
        pub_by_q[qn].append(mid)
    for mid, qn in consumes:
        con_by_q[qn].append(mid)

    for qn, pubs in pub_by_q.items():
        for cons in con_by_q.get(qn, []):
            for p in pubs:
                if p == cons:
                    continue
                stmts.append(
                    (
                        """
                        MATCH (a:Method {id: $src})
                        MATCH (b:Method {id: $dst})
                        MERGE (a)-[r:TRIGGERS]->(b)
                        SET r.project_id=$project_id, r.via=$via, r.async=true
                        """,
                        {
                            "src": p,
                            "dst": cons,
                            "project_id": project_id,
                            "via": qn,
                        },
                    )
                )
                stats["triggers"] += 1

    # Also connect consumers of same queue as siblings is not needed.
    # When only consumers exist (external publisher), still keep Queue+CONSUMES.

    chunk = 80
    for i in range(0, len(stmts), chunk):
        batch_write(client, stmts[i : i + chunk])
    return stats
