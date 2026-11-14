from __future__ import annotations

from collections import defaultdict

from app.core.utils import stable_id
from app.db.neo4j_client import Neo4jClient
from app.graph.writer import batch_write
from app.parser.models import ParsedType
from app.parser.symbol_store import symbol_id_for


def build_sync_graph(
    client: Neo4jClient,
    project_id: str,
    types: list[ParsedType],
) -> dict[str, int]:
    # indexes for resolution
    methods_by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)  # name -> [(fqcn, sig)]
    fqcn_set = {t.fqcn for t in types}
    field_type_by_class: dict[str, dict[str, str]] = defaultdict(dict)  # fqcn -> field->type
    simple_to_fqcn: dict[str, str] = {}

    for t in types:
        simple_to_fqcn[t.simple_name] = t.fqcn
        for f in t.fields:
            field_type_by_class[t.fqcn][f.name] = f.type_name
        for m in t.methods:
            methods_by_name[m.name].append((t.fqcn, m.signature))

    stmts: list[tuple[str, dict]] = []
    stats = {
        "classes": 0,
        "methods": 0,
        "calls": 0,
        "endpoints": 0,
        "feign": 0,
        "jobs": 0,
    }

    for t in types:
        class_id = symbol_id_for(project_id, t.fqcn, None)
        stmts.append(
            (
                """
                MERGE (c:Class {id: $id})
                SET c.project_id=$project_id, c.fqcn=$fqcn, c.simple_name=$simple_name,
                    c.kind=$kind, c.file_path=$file_path, c.module_name=$module_name
                """,
                {
                    "id": class_id,
                    "project_id": project_id,
                    "fqcn": t.fqcn,
                    "simple_name": t.simple_name,
                    "kind": t.symbol_kind,
                    "file_path": t.file_path,
                    "module_name": t.module_name,
                },
            )
        )
        stats["classes"] += 1

        class_base = t.extra.get("request_mapping") or ""
        for m in t.methods:
            mid = symbol_id_for(project_id, t.fqcn, m.signature)
            ann_names = {a.name for a in m.annotations}
            is_async = "Async" in ann_names
            stmts.append(
                (
                    """
                    MERGE (m:Method {id: $id})
                    SET m.project_id=$project_id, m.fqcn=$fqcn, m.name=$name, m.signature=$signature,
                        m.file_path=$file_path, m.kind=$kind, m.async=$async,
                        m.start_line=$start_line, m.end_line=$end_line
                    WITH m
                    MATCH (c:Class {id: $class_id})
                    MERGE (c)-[:DECLARES]->(m)
                    """,
                    {
                        "id": mid,
                        "project_id": project_id,
                        "fqcn": t.fqcn,
                        "name": m.name,
                        "signature": m.signature,
                        "file_path": t.file_path,
                        "kind": t.symbol_kind,
                        "async": is_async,
                        "start_line": m.start_line,
                        "end_line": m.end_line,
                        "class_id": class_id,
                    },
                )
            )
            stats["methods"] += 1

            # Endpoints
            for hm in m.http_mappings:
                path = (str(class_base).rstrip("/") + "/" + (hm.get("path") or "").lstrip("/")).strip("/")
                ep_id = stable_id(project_id, "endpoint", hm.get("http_method", ""), path, mid)
                stmts.append(
                    (
                        """
                        MERGE (e:Endpoint {id: $id})
                        SET e.project_id=$project_id, e.http_method=$http_method, e.path=$path
                        WITH e
                        MATCH (m:Method {id: $mid})
                        MERGE (e)-[:IMPLEMENTS_API]->(m)
                        """,
                        {
                            "id": ep_id,
                            "project_id": project_id,
                            "http_method": hm.get("http_method", "REQUEST"),
                            "path": path,
                            "mid": mid,
                        },
                    )
                )
                stats["endpoints"] += 1

            # XxlJob
            for a in m.annotations:
                if a.name == "XxlJob":
                    job_name = a.values.get("value") or m.name
                    job_id = stable_id(project_id, "job", job_name)
                    stmts.append(
                        (
                            """
                            MERGE (j:Job {id: $id})
                            SET j.project_id=$project_id, j.name=$name
                            WITH j
                            MATCH (m:Method {id: $mid})
                            MERGE (j)-[:SCHEDULES]->(m)
                            """,
                            {
                                "id": job_id,
                                "project_id": project_id,
                                "name": job_name,
                                "mid": mid,
                            },
                        )
                    )
                    stats["jobs"] += 1

            # CALLS resolution (injection-aware)
            for call in m.calls:
                target_mid = _resolve_call(
                    call,
                    caller_fqcn=t.fqcn,
                    field_types=field_type_by_class.get(t.fqcn, {}),
                    methods_by_name=methods_by_name,
                    simple_to_fqcn=simple_to_fqcn,
                    fqcn_set=fqcn_set,
                    project_id=project_id,
                )
                if not target_mid or target_mid == mid:
                    continue
                rel = "ASYNC_CALLS" if is_async else "CALLS"
                stmts.append(
                    (
                        f"""
                        MATCH (a:Method {{id: $src}})
                        MATCH (b:Method {{id: $dst}})
                        MERGE (a)-[r:{rel}]->(b)
                        SET r.project_id=$project_id, r.async=$async
                        """,
                        {
                            "src": mid,
                            "dst": target_mid,
                            "project_id": project_id,
                            "async": is_async or rel == "ASYNC_CALLS",
                        },
                    )
                )
                stats["calls"] += 1

        # Feign client type -> ExternalService / FeignAPI (after methods exist)
        if t.symbol_kind == "Feign" or t.feign_meta:
            svc_name = t.feign_meta.get("name") or t.simple_name
            ext_id = stable_id(project_id, "external", svc_name)
            stmts.append(
                (
                    """
                    MERGE (x:ExternalService {id: $id})
                    SET x.project_id=$project_id, x.name=$name
                    """,
                    {"id": ext_id, "project_id": project_id, "name": svc_name},
                )
            )
            base_path = t.feign_meta.get("path") or ""
            for m in t.methods:
                mid = symbol_id_for(project_id, t.fqcn, m.signature)
                mappings = m.http_mappings or [{"http_method": "REQUEST", "path": ""}]
                for hm in mappings:
                    path = (base_path.rstrip("/") + "/" + (hm.get("path") or "").lstrip("/")).rstrip("/")
                    feign_id = stable_id(
                        project_id, "feign", svc_name, hm.get("http_method", ""), path, m.name
                    )
                    stmts.append(
                        (
                            """
                            MERGE (f:FeignAPI {id: $id})
                            SET f.project_id=$project_id, f.service=$service, f.path=$path,
                                f.http_method=$http_method, f.method_name=$method_name
                            WITH f
                            MATCH (m:Method {id: $mid})
                            MERGE (m)-[r:FEIGN_CALLS]->(f)
                            SET r.project_id=$project_id
                            WITH f
                            MATCH (x:ExternalService {id: $ext_id})
                            MERGE (f)-[:BELONGS_TO]->(x)
                            """,
                            {
                                "id": feign_id,
                                "project_id": project_id,
                                "service": svc_name,
                                "path": path or m.name,
                                "http_method": hm.get("http_method", "REQUEST"),
                                "method_name": m.name,
                                "mid": mid,
                                "ext_id": ext_id,
                            },
                        )
                    )
                    stats["feign"] += 1

    # flush in chunks
    chunk = 80
    for i in range(0, len(stmts), chunk):
        batch_write(client, stmts[i : i + chunk])
    return stats


def _resolve_call(
    call: str,
    *,
    caller_fqcn: str,
    field_types: dict[str, str],
    methods_by_name: dict[str, list[tuple[str, str]]],
    simple_to_fqcn: dict[str, str],
    fqcn_set: set[str],
    project_id: str,
) -> str | None:
    if "." in call:
        obj, method_name = call.split(".", 1)
        type_name = field_types.get(obj) or obj
        target_fqcn = simple_to_fqcn.get(type_name)
        if not target_fqcn:
            for sn, fq in simple_to_fqcn.items():
                if sn.endswith(type_name) or type_name.endswith(sn):
                    target_fqcn = fq
                    break
        if target_fqcn:
            cands = [sig for fq, sig in methods_by_name.get(method_name, []) if fq == target_fqcn]
            if cands:
                return symbol_id_for(project_id, target_fqcn, cands[0])
        # same-class this.xxx already handled if obj is this
        if obj in {"this", "super"}:
            cands = [sig for fq, sig in methods_by_name.get(method_name, []) if fq == caller_fqcn]
            if cands:
                return symbol_id_for(project_id, caller_fqcn, cands[0])
        return None

    method_name = call
    # prefer same class
    cands = [sig for fq, sig in methods_by_name.get(method_name, []) if fq == caller_fqcn]
    if cands:
        return symbol_id_for(project_id, caller_fqcn, cands[0])
    # unique global
    all_cands = methods_by_name.get(method_name, [])
    if len(all_cands) == 1:
        fq, sig = all_cands[0]
        return symbol_id_for(project_id, fq, sig)
    return None
