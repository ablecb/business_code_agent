"""Sync a MySQL FlowDoc / deep-parse flow subgraph into Neo4j.

Writes (execution-order first):
  - Method / Endpoint / Class nodes (upsert)
  - FOLLOWS edges = method execution order along the locked entry call spine
    (BFS/AST order from YAML preferred; FlowDoc stages for titles / fallback)
  - Start / end markers on Method nodes (``is_entry`` / ``is_end`` / ``role``)
  - Optional CALLS edges = reference graph (off by default)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.analysis.neo4j_store import (
    clear_all_graph,
    merge_calls,
    merge_follows,
    upsert_endpoint,
)
from app.core.utils import stable_id, truncate
from app.db.mysql import get_session
from app.db.neo4j_client import get_neo4j
from app.models.orm import SymbolDescORM, SymbolORM

logger = logging.getLogger("app.analysis.flow_neo4j_sync")

DEFAULT_YAML = Path(__file__).resolve().parents[2] / "data" / "demo_treesitter_withdrawals_deep.yaml"

# Title-only hints for branch terminals (avoid matching stage *desc* body).
_TITLE_EXCEPTION = re.compile(r"异常|catch", re.IGNORECASE)
_TITLE_CALLBACK = re.compile(r"回调|callback|listener", re.IGNORECASE)


def _class_method_key(simple: str, method: str) -> str:
    return f"{simple}#{method}"


def load_flow_descs(
    project_id: str,
    *,
    flow_key: str = "loan_disburse",
    entry_class_method: str = "WithdrawalsController#withdrawals",
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    """Return (class_method -> node payload, entry_desc_extra or None)."""
    by_cm: dict[str, dict[str, Any]] = {}
    entry_extra: dict[str, Any] | None = None

    with get_session() as session:
        descs = list(
            session.scalars(
                select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
            )
        )
        sym_by_id: dict[str, SymbolORM] = {
            s.id: s
            for s in session.scalars(
                select(SymbolORM).where(SymbolORM.project_id == project_id)
            )
        }

        for d in descs:
            ej = d.extra_json if isinstance(d.extra_json, dict) else {}
            domain = d.business_domain or ""
            flow_keys = list(ej.get("flow_keys") or [])
            keep = (
                domain == flow_key
                or flow_key in flow_keys
                or (
                    isinstance(ej.get("flow_doc"), dict)
                    and (ej.get("flow_doc") or {}).get("flow_key") == flow_key
                )
            )
            if not keep:
                continue

            cm = str(ej.get("class_method") or "").strip()
            sym = sym_by_id.get(d.symbol_id)
            if not cm and sym and sym.method_name:
                cm = _class_method_key(sym.simple_name, sym.method_name)
            if not cm or "#" not in cm:
                continue

            simple, method = cm.split("#", 1)
            payload = {
                "symbol_id": d.symbol_id,
                "class_method": cm,
                "fqcn": str(ej.get("fqcn") or (sym.fqcn if sym else "") or ""),
                "simple_name": simple,
                "method_name": method,
                "method_sig": str(
                    ej.get("method_sig") or (sym.method_sig if sym else "") or ""
                ),
                "summary": truncate(d.summary or "", 800),
                "important": bool(ej.get("important") or ej.get("entry")),
                "is_entry": bool(ej.get("entry") or cm == entry_class_method),
                "flow_keys": flow_keys or [flow_key],
                "http_paths": list(ej.get("http_paths") or []),
                "bfs_depth": ej.get("bfs_depth"),
                "external_calls": list(ej.get("external_calls") or []),
                "business_tables": list(ej.get("business_tables") or []),
                "table_field_changes": list(ej.get("table_field_changes") or []),
                "status_transitions": list(ej.get("status_transitions") or []),
                "data_flow_narrative": truncate(
                    str(ej.get("data_flow_narrative") or ""), 1500
                ),
                "module_name": str(
                    ej.get("module_name") or (sym.module_name if sym else "") or ""
                ),
                "file_path": str(sym.file_path if sym else "") or "",
                "kind": str(sym.kind if sym else "") or "",
                "flow_doc": ej.get("flow_doc")
                if isinstance(ej.get("flow_doc"), dict)
                else None,
            }
            prev = by_cm.get(cm)
            if prev is None or (payload["is_entry"] and not prev.get("is_entry")):
                by_cm[cm] = payload
            if payload["is_entry"] or cm == entry_class_method:
                entry_extra = {**ej, "symbol_id": d.symbol_id, "class_method": cm}

    return by_cm, entry_extra


def build_callee_map_from_yaml(yaml_path: Path | None) -> dict[str, list[str]]:
    """Ordered caller -> callees from tree-sitter BFS YAML (tree / edges)."""
    if not yaml_path or not yaml_path.exists():
        return {}
    try:
        import yaml

        from app.analysis.treesitter_deep_flow import _callee_map_from_bfs

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return _callee_map_from_bfs(data)
    except Exception:
        logger.exception("failed loading callee map from %s", yaml_path)
        return {}


def load_bfs_yaml(yaml_path: Path | None) -> dict[str, Any]:
    if not yaml_path or not yaml_path.exists():
        return {}
    try:
        import yaml

        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("failed loading BFS YAML %s", yaml_path)
        return {}


def entry_call_spine_from_yaml(
    yaml_path: Path | None,
    entry_class_method: str,
    *,
    known_cms: set[str] | None = None,
    entry_calls: list[str] | None = None,
) -> list[str]:
    """BFS/AST ordered direct callees of the locked entry (FOLLOWS spine)."""
    data = load_bfs_yaml(yaml_path)
    if not data:
        return []
    from app.analysis.treesitter_deep_flow import _entry_call_spine

    return _entry_call_spine(
        data,
        entry_class_method,
        entry_calls=entry_calls,
        known_cms=known_cms,
    )


def _filter_stage_cms(
    cms: list[str],
    *,
    entry_class_method: str,
    yaml_path: Path | None = None,
) -> list[str]:
    """Drop off-chain Controller sibling HTTP methods from a stage/FOLLOWS list."""
    data = load_bfs_yaml(yaml_path)
    reachable: set[str] = set()
    if data:
        from app.analysis.treesitter_deep_flow import (
            _is_off_chain_controller_sibling,
            _reachable_cms,
        )

        reachable = _reachable_cms(data, entry_class_method)
        out: list[str] = []
        for cm in cms:
            if not cm or "#" not in cm:
                continue
            if _is_off_chain_controller_sibling(cm, entry_class_method, reachable | set(cms)):
                continue
            if cm not in out:
                out.append(cm)
        return out
    # No YAML: still drop same-Controller methods other than entry unless already listed
    # after entry as helpers that share the controller simple name *and* look like
    # private helpers (kept). Sibling HTTP methods usually never appear without YAML.
    return list(dict.fromkeys(cm for cm in cms if cm and "#" in cm))


def ordered_execution_cms(
    entry_extra: dict[str, Any] | None,
    *,
    entry_class_method: str,
    yaml_path: Path | None = None,
    prefer_bfs_spine: bool = True,
    known_cms: set[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (ordered class_methods including entry, stages used for titles).

    When ``prefer_bfs_spine`` and YAML are available, FOLLOWS follows the entry
    call spine (Controller→Service1→…→Service2), not LLM-invented stage catalogs.
    FlowDoc stages still supply titles/descriptions matched by class_method.
    """
    stages: list[dict[str, Any]] = []
    if entry_extra and isinstance(entry_extra.get("flow_doc"), dict):
        stages = list((entry_extra.get("flow_doc") or {}).get("stages") or [])

    flow_cms: list[str] = []
    if entry_class_method:
        flow_cms.append(entry_class_method)
    for st in stages:
        if not isinstance(st, dict):
            continue
        cm = str(st.get("class_method") or "").strip()
        if cm and "#" in cm and cm not in flow_cms:
            flow_cms.append(cm)

    spine = entry_call_spine_from_yaml(
        yaml_path,
        entry_class_method,
        known_cms=known_cms,
        entry_calls=list((entry_extra or {}).get("entry_calls") or [])
        if entry_extra
        else None,
    )
    # Prefer spine stored on flow_doc from deep-parse / rebuild when richer.
    if entry_extra and isinstance(entry_extra.get("flow_doc"), dict):
        stored = [
            str(x)
            for x in ((entry_extra.get("flow_doc") or {}).get("execution_spine") or [])
            if x and "#" in str(x)
        ]
        if stored and (not spine or len(stored) >= len(spine)):
            spine = stored
        elif stored:
            # Merge: keep YAML/AST order, append missing stored steps.
            seen = set(spine)
            for cm in stored:
                if cm not in seen:
                    spine.append(cm)
                    seen.add(cm)

    if prefer_bfs_spine and spine:
        cms = [entry_class_method] if entry_class_method else []
        for cm in spine:
            if cm and "#" in cm and cm not in cms:
                cms.append(cm)
        cms = _filter_stage_cms(
            cms, entry_class_method=entry_class_method, yaml_path=yaml_path
        )
        # Rebuild stage list in spine order; keep LLM/FlowDoc titles when present.
        by_cm_stage = {
            str(st.get("class_method") or ""): st
            for st in stages
            if isinstance(st, dict) and st.get("class_method")
        }
        ordered_stages: list[dict[str, Any]] = []
        for i, cm in enumerate(cms):
            if cm == entry_class_method:
                continue
            prev = by_cm_stage.get(cm) or {}
            ordered_stages.append(
                {
                    "step": i,
                    "title": str(prev.get("title") or cm.split("#", 1)[-1])[:120],
                    "class_method": cm,
                    "desc_business": str(prev.get("desc_business") or ""),
                    "desc_dev": str(prev.get("desc_dev") or ""),
                    "desc": str(prev.get("desc") or ""),
                }
            )
        return cms, ordered_stages

    cms = _filter_stage_cms(
        flow_cms, entry_class_method=entry_class_method, yaml_path=yaml_path
    )
    return cms, stages


def build_follows_edges(
    entry_extra: dict[str, Any] | None,
    by_cm: dict[str, dict[str, Any]],
    *,
    flow_key: str,
    entry_class_method: str,
    yaml_path: Path | None = None,
    prefer_bfs_spine: bool = True,
) -> list[dict[str, Any]]:
    """FOLLOWS chain: entry call spine (BFS) preferred over raw FlowDoc stage order."""
    cms, stages = ordered_execution_cms(
        entry_extra,
        entry_class_method=entry_class_method,
        yaml_path=yaml_path,
        prefer_bfs_spine=prefer_bfs_spine,
        known_cms=set(by_cm.keys()),
    )

    stage_by_cm = {
        str(st.get("class_method") or ""): st
        for st in stages
        if isinstance(st, dict) and st.get("class_method")
    }

    out: list[dict[str, Any]] = []
    for i in range(len(cms) - 1):
        to_cm = cms[i + 1]
        title = ""
        st = stage_by_cm.get(to_cm)
        if st:
            title = str(st.get("title") or "")[:120]
        out.append(
            {
                "from": cms[i],
                "to": to_cm,
                "step": i + 1,
                "flow_key": flow_key,
                "rel": "FOLLOWS",
                "title": title,
            }
        )
    return out


def end_class_methods(
    entry_extra: dict[str, Any] | None,
    *,
    entry_class_method: str,
    yaml_path: Path | None = None,
    prefer_bfs_spine: bool = True,
) -> dict[str, str]:
    """Map class_method -> end role (``end`` / ``exception_end`` / ``callback_end``).

    Always marks the last execution-order stage. Extra terminals only when the *title*
    looks like an exception/callback branch (not the stage description body).
    """
    cms, stages = ordered_execution_cms(
        entry_extra,
        entry_class_method=entry_class_method,
        yaml_path=yaml_path,
        prefer_bfs_spine=prefer_bfs_spine,
    )
    ends: dict[str, str] = {}
    for st in stages:
        if not isinstance(st, dict):
            continue
        cm = str(st.get("class_method") or "").strip()
        if not cm or "#" not in cm:
            continue
        title = str(st.get("title") or "")
        if _TITLE_CALLBACK.search(title):
            ends[cm] = "callback_end"
        elif _TITLE_EXCEPTION.search(title):
            ends[cm] = "exception_end"
    if cms:
        last = cms[-1]
        ends.setdefault(last, "end")
        # Prefer exception/callback label already set from title.
    return ends


def _guess_cm_from_external_call(call: str, known_cms: set[str]) -> str | None:
    """Best-effort map 'withdrawalsService.loan' -> 'WithdrawalsServiceImpl#loan'."""
    raw = (call or "").strip().split("(")[0].strip()
    if not raw or "." not in raw:
        return None
    obj, method = raw.rsplit(".", 1)
    if not method or method in {"getLock", "buildSuccess", "buildFailure"}:
        return None
    if "#" in raw:
        return raw if raw in known_cms else None

    obj_l = obj.lower()
    candidates: list[str] = []
    for cm in known_cms:
        simple, m = cm.split("#", 1)
        if m != method:
            continue
        sl = simple.lower()
        if obj_l in sl or sl.replace("impl", "") in obj_l:
            candidates.append(cm)
    if len(candidates) == 1:
        return candidates[0]
    impls = [c for c in candidates if c.split("#", 1)[0].endswith("Impl")]
    if len(impls) == 1:
        return impls[0]
    return None


def build_reference_edges(
    by_cm: dict[str, dict[str, Any]],
    callee_map: dict[str, list[str]],
    *,
    flow_key: str,
    entry_class_method: str,
) -> list[dict[str, Any]]:
    """CALLS reference edges with deterministic ``order``."""
    known = set(by_cm.keys())
    seed = set(known)
    if entry_class_method in callee_map or entry_class_method in known:
        seed.add(entry_class_method)
    # Grow seed: any YAML edge touching a known flow node
    changed = True
    while changed:
        changed = False
        for frm, tos in callee_map.items():
            if frm in seed:
                for to in tos:
                    if to not in seed:
                        seed.add(to)
                        changed = True
            elif any(to in seed for to in tos):
                if frm not in seed:
                    seed.add(frm)
                    changed = True

    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def add(frm: str, to: str, order: int, source: str) -> None:
        if "#" not in frm or "#" not in to:
            return
        if frm not in seed:
            return
        key = (frm, to)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "from": frm,
                "to": to,
                "order": order,
                "source": source,
                "flow_key": flow_key,
                "rel": "CALLS",
            }
        )

    for frm, tos in callee_map.items():
        if frm not in seed:
            continue
        for i, to in enumerate(tos):
            add(frm, to, i + 1, "treesitter-bfs")

    for cm, payload in by_cm.items():
        existing_tos = {e["to"] for e in edges if e["from"] == cm}
        order_base = len(existing_tos)
        for j, call in enumerate(payload.get("external_calls") or []):
            if isinstance(call, dict):
                call_s = str(call.get("target") or "")
            else:
                call_s = str(call)
            to = _guess_cm_from_external_call(call_s, known | seed)
            if not to or to in existing_tos:
                continue
            add(cm, to, order_base + j + 1, "mysql-external_calls")
            existing_tos.add(to)

    return edges


def _upsert_method_from_payload(project_id: str, payload: dict[str, Any]) -> str:
    from app.core.config import get_settings
    from app.parser.symbol_store import symbol_id_for

    client = get_neo4j()
    mid = payload["symbol_id"]
    fqcn = payload.get("fqcn") or payload.get("simple_name") or ""
    simple = payload.get("simple_name") or (fqcn.split(".")[-1] if fqcn else "")
    class_id = symbol_id_for(project_id, fqcn, None) if fqcn else stable_id(project_id, simple)
    write_narratives = bool(get_settings().neo4j_store_narratives)

    props: dict[str, Any] = {
        "project_id": project_id,
        "fqcn": fqcn,
        "simple_name": simple,
        "name": payload.get("method_name") or "",
        "signature": payload.get("method_sig") or "",
        "class_method": payload.get("class_method") or "",
        "file_path": payload.get("file_path") or "",
        "kind": payload.get("kind") or "",
        "module_name": payload.get("module_name") or "",
        "important": bool(payload.get("important")),
        "is_entry": bool(payload.get("is_entry")),
        "is_end": bool(payload.get("is_end")),
        "flow_keys": list(payload.get("flow_keys") or []),
    }
    if payload.get("role"):
        props["role"] = str(payload["role"])
    if payload.get("bfs_depth") is not None:
        props["bfs_depth"] = payload["bfs_depth"]
    if payload.get("http_paths"):
        props["http_path"] = str(payload["http_paths"][0])
    if write_narratives:
        if payload.get("summary"):
            props["summary"] = payload["summary"]
        if payload.get("data_flow_narrative"):
            props["data_flow"] = truncate(str(payload["data_flow_narrative"]), 2000)
        if payload.get("business_tables"):
            from app.analysis.neo4j_store import _neo4j_string_list

            props["business_tables"] = _neo4j_string_list(
                payload["business_tables"], limit=30
            )
        if payload.get("status_transitions"):
            from app.analysis.neo4j_store import _neo4j_string_list

            props["status_transitions"] = _neo4j_string_list(
                payload["status_transitions"], limit=20
            )
        if payload.get("table_field_changes"):
            import json

            props["table_field_changes"] = truncate(
                json.dumps(payload["table_field_changes"][:20], ensure_ascii=False),
                3000,
            )

    client.write_tx(
        """
        MERGE (m:Method {id: $id})
        SET m += $props
        WITH m
        MERGE (c:Class {id: $class_id})
        ON CREATE SET c.project_id=$project_id, c.fqcn=$fqcn, c.simple_name=$simple_name
        SET c.project_id = coalesce(c.project_id, $project_id),
            c.fqcn = coalesce(c.fqcn, $fqcn),
            c.simple_name = coalesce(c.simple_name, $simple_name)
        MERGE (c)-[:DECLARES]->(m)
        """,
        {
            "id": mid,
            "props": props,
            "class_id": class_id,
            "project_id": project_id,
            "fqcn": fqcn,
            "simple_name": simple,
        },
    )
    return mid


def _ensure_stub_method(
    project_id: str,
    class_method: str,
    *,
    flow_key: str,
    id_by_cm: dict[str, str],
) -> str | None:
    if class_method in id_by_cm:
        return id_by_cm[class_method]
    if "#" not in class_method:
        return None
    simple, method = class_method.split("#", 1)
    with get_session() as session:
        sym = session.scalar(
            select(SymbolORM).where(
                SymbolORM.project_id == project_id,
                SymbolORM.simple_name == simple,
                SymbolORM.method_name == method,
            )
        )
        if sym is None:
            mid = None
            fqcn = simple
            method_sig = ""
            file_path = ""
            kind = ""
            module_name = ""
        else:
            mid = sym.id
            fqcn = sym.fqcn or simple
            method_sig = sym.method_sig or ""
            file_path = sym.file_path or ""
            kind = sym.kind or ""
            module_name = sym.module_name or ""
    if mid:
        payload = {
            "symbol_id": mid,
            "class_method": class_method,
            "fqcn": fqcn,
            "simple_name": simple,
            "method_name": method,
            "method_sig": method_sig,
            "file_path": file_path,
            "kind": kind,
            "module_name": module_name,
            "important": False,
            "is_entry": False,
            "flow_keys": [flow_key],
            "http_paths": [],
        }
        _upsert_method_from_payload(project_id, payload)
        id_by_cm[class_method] = mid
        return mid

    mid = stable_id(project_id, simple, method)
    payload = {
        "symbol_id": mid,
        "class_method": class_method,
        "fqcn": simple,
        "simple_name": simple,
        "method_name": method,
        "method_sig": "",
        "file_path": "",
        "kind": "",
        "module_name": "",
        "important": False,
        "is_entry": False,
        "flow_keys": [flow_key],
        "http_paths": [],
    }
    _upsert_method_from_payload(project_id, payload)
    id_by_cm[class_method] = mid
    return mid


def clear_flow_edges(project_id: str, flow_key: str) -> dict[str, int]:
    """Delete scoped CALLS/FOLLOWS for this flow_key only."""
    client = get_neo4j()
    before_calls = client.run(
        """
        MATCH (a:Method)-[r:CALLS]->(b:Method)
        WHERE a.project_id = $pid AND r.flow_key = $fk
        RETURN count(r) AS n
        """,
        {"pid": project_id, "fk": flow_key},
    )
    before_follows = client.run(
        """
        MATCH (a:Method)-[r:FOLLOWS]->(b:Method)
        WHERE a.project_id = $pid AND r.flow_key = $fk
        RETURN count(r) AS n
        """,
        {"pid": project_id, "fk": flow_key},
    )
    client.write_tx(
        """
        MATCH (a:Method)-[r:CALLS]->(b:Method)
        WHERE a.project_id = $pid AND r.flow_key = $fk
        DELETE r
        """,
        {"pid": project_id, "fk": flow_key},
    )
    client.write_tx(
        """
        MATCH (a:Method)-[r:FOLLOWS]->(b:Method)
        WHERE a.project_id = $pid AND r.flow_key = $fk
        DELETE r
        """,
        {"pid": project_id, "fk": flow_key},
    )
    return {
        "cleared_calls": int((before_calls[0] or {}).get("n") or 0) if before_calls else 0,
        "cleared_follows": int((before_follows[0] or {}).get("n") or 0)
        if before_follows
        else 0,
    }


def _mark_start_end(
    *,
    project_id: str,
    flow_key: str,
    entry_class_method: str,
    entry_id: str | None,
    http_path: str,
    id_by_cm: dict[str, str],
    ends: dict[str, str],
    entry_extra: dict[str, Any] | None,
) -> dict[str, Any]:
    neo = get_neo4j()
    # Clear prior start/end flags on methods tagged with this flow_key.
    neo.write_tx(
        """
        MATCH (m:Method)
        WHERE m.project_id = $pid AND $fk IN coalesce(m.flow_keys, [])
        SET m.is_entry = false,
            m.is_end = false,
            m.role = CASE
              WHEN m.role IN ['start', 'end', 'exception_end', 'callback_end']
              THEN null
              ELSE m.role
            END
        """,
        {"pid": project_id, "fk": flow_key},
    )

    if entry_id:
        neo.write_tx(
            """
            MATCH (m:Method {id: $id})
            SET m.is_entry = true,
                m.is_end = false,
                m.role = 'start',
                m.important = true,
                m.flow_keys = $flow_keys,
                m.http_path = $http_path,
                m.class_method = $cm
            """,
            {
                "id": entry_id,
                "flow_keys": [flow_key],
                "http_path": http_path,
                "cm": entry_class_method,
            },
        )
        fd = (entry_extra or {}).get("flow_doc") if entry_extra else None
        if isinstance(fd, dict):
            neo.write_tx(
                """
                MATCH (m:Method {id: $id})
                SET m.flow_title = $title,
                    m.flow_narrative = $narrative
                """,
                {
                    "id": entry_id,
                    "title": truncate(str(fd.get("title") or ""), 200),
                    "narrative": truncate(str(fd.get("narrative") or ""), 1500),
                },
            )

    marked_ends: list[dict[str, str]] = []
    for cm, role in ends.items():
        mid = id_by_cm.get(cm)
        if not mid or mid == entry_id:
            continue
        neo.write_tx(
            """
            MATCH (m:Method {id: $id})
            SET m.is_end = true,
                m.role = $role,
                m.important = true,
                m.flow_keys = CASE
                  WHEN m.flow_keys IS NULL THEN [$fk]
                  WHEN $fk IN m.flow_keys THEN m.flow_keys
                  ELSE m.flow_keys + [$fk]
                END
            """,
            {"id": mid, "role": role, "fk": flow_key},
        )
        marked_ends.append({"class_method": cm, "role": role})

    # Any FOLLOWS terminal for this flow with no outgoing FOLLOWS is also an end.
    terminals = neo.run(
        """
        MATCH (a:Method)-[r:FOLLOWS]->(b:Method)
        WHERE a.project_id = $pid AND r.flow_key = $fk
        WITH collect(DISTINCT a.id) AS srcs, collect(DISTINCT b) AS targets
        UNWIND targets AS t
        WITH t, srcs
        WHERE NOT t.id IN srcs
        RETURN t.id AS id, t.class_method AS cm
        """,
        {"pid": project_id, "fk": flow_key},
    )
    for row in terminals or []:
        tid = row.get("id")
        cm = str(row.get("cm") or "")
        if not tid or tid == entry_id:
            continue
        if any(e.get("class_method") == cm for e in marked_ends):
            continue
        neo.write_tx(
            """
            MATCH (m:Method {id: $id})
            SET m.is_end = true,
                m.role = coalesce(m.role, 'end')
            """,
            {"id": tid},
        )
        marked_ends.append({"class_method": cm, "role": "end"})

    return {
        "start": entry_class_method if entry_id else None,
        "ends": marked_ends,
    }


def verify_execution_graph(
    project_id: str,
    *,
    flow_key: str,
    entry_class_method: str,
) -> dict[str, Any]:
    """Counts + ordered FOLLOWS chain from start."""
    neo = get_neo4j()
    counts = neo.run(
        """
        MATCH (n)
        WITH count(n) AS nodes
        OPTIONAL MATCH ()-[f:FOLLOWS]->()
        WITH nodes, count(f) AS follows
        OPTIONAL MATCH ()-[c:CALLS]->()
        WITH nodes, follows, count(c) AS calls
        OPTIONAL MATCH (s:Method)
        WHERE s.project_id = $pid AND coalesce(s.is_entry, false) = true
        WITH nodes, follows, calls, count(s) AS starts
        OPTIONAL MATCH (e:Method)
        WHERE e.project_id = $pid AND coalesce(e.is_end, false) = true
        RETURN nodes, follows, calls, starts, count(e) AS ends
        """,
        {"pid": project_id},
    )
    chain = neo.run(
        """
        MATCH (start:Method)
        WHERE start.project_id = $pid
          AND start.class_method = $cm
        MATCH path = (start)-[:FOLLOWS*0..30]->(tail:Method)
        WHERE ALL(r IN relationships(path) WHERE r.flow_key = $fk)
          AND NOT (tail)-[:FOLLOWS {flow_key: $fk}]->()
        WITH path
        ORDER BY length(path) DESC
        LIMIT 1
        RETURN [n IN nodes(path) | n.class_method] AS ordered,
               [n IN nodes(path) | {
                 class_method: n.class_method,
                 role: n.role,
                 is_entry: coalesce(n.is_entry, false),
                 is_end: coalesce(n.is_end, false)
               }] AS marked
        """,
        {"pid": project_id, "cm": entry_class_method, "fk": flow_key},
    )
    starts = neo.run(
        """
        MATCH (m:Method)
        WHERE m.project_id = $pid AND coalesce(m.is_entry, false) = true
        OPTIONAL MATCH (ep:Endpoint)-[:IMPLEMENTS_API]->(m)
        RETURN m.class_method AS class_method, m.role AS role,
               collect(DISTINCT ep.path) AS paths
        """,
        {"pid": project_id},
    )
    ends = neo.run(
        """
        MATCH (m:Method)
        WHERE m.project_id = $pid AND coalesce(m.is_end, false) = true
        RETURN m.class_method AS class_method, m.role AS role
        ORDER BY m.class_method
        """,
        {"pid": project_id},
    )
    return {
        "counts": (counts[0] if counts else {}),
        "ordered_path": (chain[0].get("ordered") if chain else []) or [],
        "marked_path": (chain[0].get("marked") if chain else []) or [],
        "starts": starts or [],
        "ends": ends or [],
    }


def _first_business_sentence(text: str, *, max_len: int = 80) -> str:
    """Strip dual-perspective markers; keep first usable business sentence."""
    raw = (text or "").strip()
    if not raw:
        return ""
    m = re.search(r"【业务[·・.]?精简】\s*(.+?)(?:\n|【|$)", raw, re.S)
    if m:
        raw = m.group(1).strip()
    # Drop leading Class#method label if present.
    raw = re.sub(r"^[A-Za-z_][\w.]*#[A-Za-z_][\w]*\s*", "", raw).strip()
    raw = re.sub(r"；?\s*\[段\d+\].*$", "", raw).strip()
    for sep in ("。", "；", ";", "\n"):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
            break
    return truncate(raw, max_len)


def _ordered_calls_l1_spine(project_id: str, entry_class_method: str) -> list[str]:
    """Entry + L1 CALLS ordered by ``r.order`` (topo AST spine when FOLLOWS empty)."""
    entry = (entry_class_method or "").strip()
    if not project_id or not entry or "#" not in entry:
        return []
    try:
        client = get_neo4j()
        rows = client.run(
            """
            MATCH (a:Method {project_id: $pid})-[r:CALLS]->(b:Method)
            WHERE a.class_method = $cm
            RETURN b.class_method AS cm, r.order AS ord
            ORDER BY coalesce(r.order, 9999), b.class_method
            """,
            {"pid": project_id, "cm": entry},
        )
    except Exception:
        logger.debug("ordered CALLS L1 lookup failed", exc_info=True)
        return []
    out = [entry]
    seen = {entry}
    for row in rows or []:
        cm = str((row or {}).get("cm") or "").strip()
        if not cm or "#" not in cm or cm in seen:
            continue
        seen.add(cm)
        out.append(cm)
    return out if len(out) >= 2 else []


def get_follows_path(
    project_id: str,
    entry_class_method: str,
    flow_key: str = "loan_disburse",
) -> dict[str, Any]:
    """Ordered FOLLOWS chain from locked entry + MySQL symbol facts for Q&A.

    Prefer Neo4j ``FOLLOWS`` (execution spine). When empty: MySQL spine, then
    ordered CALLS L1 from AST topo (``neo4j_calls``).

    Returns keys: ok, source, flow_key, entry, ordered_path, steps, http_paths,
    title, narrative.
    """
    entry = (entry_class_method or "").strip()
    fk = (flow_key or "").strip() or "loan_disburse"
    empty: dict[str, Any] = {
        "ok": False,
        "source": "",
        "flow_key": fk,
        "entry": entry,
        "ordered_path": [],
        "steps": [],
        "http_paths": [],
        "title": "",
        "narrative": "",
    }
    if not project_id or not entry or "#" not in entry:
        return empty

    ordered: list[str] = []
    source = ""
    try:
        verify = verify_execution_graph(
            project_id, flow_key=fk, entry_class_method=entry
        )
        ordered = [
            str(x) for x in (verify.get("ordered_path") or []) if x and "#" in str(x)
        ]
        if len(ordered) >= 2:
            source = "neo4j_follows"
    except Exception:
        logger.debug("get_follows_path neo4j failed", exc_info=True)
        ordered = []

    by_cm, entry_extra = load_flow_descs(
        project_id, flow_key=fk, entry_class_method=entry
    )
    fd = (
        (entry_extra or {}).get("flow_doc")
        if entry_extra and isinstance((entry_extra or {}).get("flow_doc"), dict)
        else {}
    ) or {}
    if len(ordered) < 2:
        cms, _stages = ordered_execution_cms(
            entry_extra,
            entry_class_method=entry,
            prefer_bfs_spine=True,
            known_cms=set(by_cm.keys()),
        )
        ordered = [str(x) for x in cms if x and "#" in str(x)]
        if len(ordered) >= 2:
            source = "mysql_spine"

    # Topo pilot may only write CALLS{order} (no FOLLOWS). Use L1 CALLS as spine.
    if len(ordered) < 2:
        ordered = _ordered_calls_l1_spine(project_id, entry)
        if len(ordered) >= 2:
            source = "neo4j_calls"

    if len(ordered) < 2:
        return empty

    stage_by_cm: dict[str, dict[str, Any]] = {}
    for st in fd.get("stages") or []:
        if isinstance(st, dict) and st.get("class_method"):
            stage_by_cm[str(st["class_method"])] = st

    # Prefer dual-perspective columns when present on SymbolDescORM.
    biz_clear_by_cm: dict[str, str] = {}
    try:
        with get_session() as session:
            for cm, payload in by_cm.items():
                sid = str(payload.get("symbol_id") or "")
                if not sid:
                    continue
                d = session.scalar(
                    select(SymbolDescORM).where(SymbolDescORM.symbol_id == sid)
                )
                if d and getattr(d, "summary_business_clear", None):
                    biz_clear_by_cm[cm] = str(d.summary_business_clear or "")
    except Exception:
        logger.debug("get_follows_path biz_clear load failed", exc_info=True)

    steps: list[dict[str, Any]] = []
    for i, cm in enumerate(ordered):
        payload = by_cm.get(cm) or {}
        st = stage_by_cm.get(cm) or {}
        method = cm.split("#", 1)[-1]
        biz = (
            _first_business_sentence(biz_clear_by_cm.get(cm) or "")
            or _first_business_sentence(str(st.get("desc_business") or ""))
            or _first_business_sentence(str(payload.get("summary") or ""))
        )
        title = str(st.get("title") or "").strip()
        if not title or title == method or title == cm:
            title = biz or method
        steps.append(
            {
                "step": i + 1,
                "class_method": cm,
                "title": truncate(title, 120),
                "desc_business": truncate(
                    str(st.get("desc_business") or biz or ""), 400
                ),
                "desc_dev": truncate(str(st.get("desc_dev") or ""), 400),
                "desc": truncate(
                    str(st.get("desc") or biz or payload.get("summary") or ""), 400
                ),
                "summary": truncate(str(payload.get("summary") or ""), 600),
                "data_flow": truncate(
                    str(payload.get("data_flow_narrative") or ""), 400
                ),
                "business_tables": list(payload.get("business_tables") or []),
                "status_transitions": list(payload.get("status_transitions") or []),
            }
        )

    paths = list((entry_extra or {}).get("http_paths") or []) if entry_extra else []
    return {
        "ok": True,
        "source": source,
        "flow_key": fk,
        "entry": entry,
        "ordered_path": ordered,
        "steps": steps,
        "http_paths": [str(p) for p in paths if p],
        "title": str(fd.get("title") or ""),
        "narrative": str(
            fd.get("narrative_business_clear") or fd.get("narrative") or ""
        ),
    }


def sync_flow_to_neo4j(
    project_id: str,
    *,
    flow_key: str = "loan_disburse",
    entry_class_method: str = "WithdrawalsController#withdrawals",
    http_path: str = "withdrawals/withdrawals",
    http_method: str = "POST",
    yaml_path: Path | None = DEFAULT_YAML,
    dry_run: bool = False,
    clear_all: bool = False,
    include_calls: bool = False,
    prefer_bfs_spine: bool = True,
) -> dict[str, Any]:
    """Read MySQL FlowDoc and upsert execution-order graph (FOLLOWS + start/end).

    FOLLOWS prefers the BFS/AST entry call spine from ``yaml_path`` when
    ``prefer_bfs_spine`` is true (avoids LLM-invented stage catalogs).
    Set ``include_calls=True`` to also write CALLS from YAML / external_calls.
    ``clear_all`` wipes the entire DB.
    """
    by_cm, entry_extra = load_flow_descs(
        project_id, flow_key=flow_key, entry_class_method=entry_class_method
    )
    exec_cms, stages = ordered_execution_cms(
        entry_extra,
        entry_class_method=entry_class_method,
        yaml_path=yaml_path,
        prefer_bfs_spine=prefer_bfs_spine,
        known_cms=set(by_cm.keys()),
    )
    ends = end_class_methods(
        entry_extra,
        entry_class_method=entry_class_method,
        yaml_path=yaml_path,
        prefer_bfs_spine=prefer_bfs_spine,
    )
    follows = build_follows_edges(
        entry_extra,
        by_cm,
        flow_key=flow_key,
        entry_class_method=entry_class_method,
        yaml_path=yaml_path,
        prefer_bfs_spine=prefer_bfs_spine,
    )

    ref_edges: list[dict[str, Any]] = []
    callee_map: dict[str, list[str]] = {}
    if include_calls:
        callee_map = build_callee_map_from_yaml(yaml_path)
        ref_edges = build_reference_edges(
            by_cm,
            callee_map,
            flow_key=flow_key,
            entry_class_method=entry_class_method,
        )

    # Ensure stage methods appear even if not yet in by_cm.
    for cm in exec_cms:
        if cm not in by_cm and "#" in cm:
            simple, method = cm.split("#", 1)
            by_cm[cm] = {
                "symbol_id": "",
                "class_method": cm,
                "fqcn": simple,
                "simple_name": simple,
                "method_name": method,
                "method_sig": "",
                "summary": "",
                "important": cm == entry_class_method or cm in ends,
                "is_entry": cm == entry_class_method,
                "is_end": cm in ends,
                "role": (
                    "start"
                    if cm == entry_class_method
                    else ends.get(cm, "")
                ),
                "flow_keys": [flow_key],
                "http_paths": [http_path] if cm == entry_class_method else [],
                "external_calls": [],
                "module_name": "",
                "file_path": "",
                "kind": "",
                "flow_doc": None,
            }

    for cm, payload in by_cm.items():
        if cm == entry_class_method:
            payload["is_entry"] = True
            payload["role"] = "start"
            payload["important"] = True
            payload["is_end"] = False
        elif cm in ends:
            payload["is_end"] = True
            payload["role"] = ends[cm]
            payload["important"] = True

    report: dict[str, Any] = {
        "project_id": project_id,
        "flow_key": flow_key,
        "entry": entry_class_method,
        "http_path": http_path,
        "mysql_nodes": len(by_cm),
        "execution_cms": exec_cms,
        "stage_count": len(stages),
        "follows_edges": len(follows),
        "reference_edges": len(ref_edges),
        "include_calls": include_calls,
        "prefer_bfs_spine": prefer_bfs_spine,
        "yaml_path": str(yaml_path) if yaml_path else None,
        "clear_all": clear_all,
        "dry_run": dry_run,
        "ends": ends,
        "sample_follows": follows[:12],
        "sample_calls": ref_edges[:8],
    }

    if dry_run:
        return report

    neo = get_neo4j()
    if not neo.verify():
        raise RuntimeError("Neo4j unavailable")
    neo.ensure_constraints()

    if clear_all:
        deleted = clear_all_graph()
        report["cleared_all_nodes"] = deleted
        neo.ensure_constraints()
    else:
        cleared = clear_flow_edges(project_id, flow_key)
        report["cleared"] = cleared

    id_by_cm: dict[str, str] = {}
    for cm, p in by_cm.items():
        sid = p.get("symbol_id")
        if sid:
            id_by_cm[cm] = str(sid)

    nodes_written = 0
    for payload in by_cm.values():
        if not payload.get("symbol_id"):
            continue
        _upsert_method_from_payload(project_id, payload)
        nodes_written += 1

    needed_cms = {e["from"] for e in follows} | {e["to"] for e in follows}
    if include_calls:
        needed_cms |= {e["from"] for e in ref_edges} | {e["to"] for e in ref_edges}
    stubs = 0
    for cm in sorted(needed_cms):
        if cm not in id_by_cm:
            mid = _ensure_stub_method(
                project_id, cm, flow_key=flow_key, id_by_cm=id_by_cm
            )
            if mid:
                stubs += 1

    # Edge narratives (title/data_flow) stay in MySQL unless neo4j_store_narratives.
    from app.core.config import get_settings

    write_edge_narratives = bool(get_settings().neo4j_store_narratives)

    follows_written = 0
    for e in follows:
        a = id_by_cm.get(e["from"])
        b = id_by_cm.get(e["to"])
        if not a or not b:
            continue
        to_payload = by_cm.get(e["to"]) or {}
        merge_follows(
            project_id,
            a,
            b,
            step=int(e.get("step") or 0),
            flow_key=flow_key,
            title=str(e.get("title") or "") if write_edge_narratives else "",
            data_flow=(
                str(to_payload.get("data_flow_narrative") or "")
                if write_edge_narratives
                else ""
            ),
        )
        follows_written += 1

    calls_written = 0
    if include_calls:
        for e in ref_edges:
            a = id_by_cm.get(e["from"])
            b = id_by_cm.get(e["to"])
            if not a or not b:
                continue
            to_payload = by_cm.get(e["to"]) or {}
            merge_calls(
                project_id,
                a,
                b,
                order=int(e.get("order") or 0) or None,
                source=str(e.get("source") or ""),
                flow_key=flow_key,
                data_flow=(
                    str(to_payload.get("data_flow_narrative") or "")
                    if write_edge_narratives
                    else None
                ),
            )
            calls_written += 1

    entry_id = id_by_cm.get(entry_class_method)
    if entry_id and http_path:
        upsert_endpoint(project_id, entry_id, http_method, http_path)

    markers = _mark_start_end(
        project_id=project_id,
        flow_key=flow_key,
        entry_class_method=entry_class_method,
        entry_id=entry_id,
        http_path=http_path,
        id_by_cm=id_by_cm,
        ends=ends,
        entry_extra=entry_extra,
    )

    verify = verify_execution_graph(
        project_id,
        flow_key=flow_key,
        entry_class_method=entry_class_method,
    )

    report.update(
        {
            "nodes_written": nodes_written,
            "stubs_written": stubs,
            "calls_written": calls_written,
            "follows_written": follows_written,
            "markers": markers,
            "verify": verify,
        }
    )
    return report
