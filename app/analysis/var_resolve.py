"""Per-method AST var-resolve bags → MySQL + enriched Neo4j CALLS props.

Production source of truth for precise receiver typing / setXxx discovery.
File dumps under ``data/`` are debug exports of the same payload shape.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any, Callable, Iterable

from sqlalchemy import func, select

from app.analysis.global_ast import (
    AstType,
    CallEdge,
    GlobalAst,
    ScopedName,
    extract_method_locals,
    find_method_node,
    resolve_call_detailed,
    _parser,
)
from app.db.mysql import get_session, init_db
from app.models.orm import MethodVarResolveORM, SymbolORM

logger = logging.getLogger("app.analysis.var_resolve")

RAW_CODE_PREVIEW_MAX = 1200

_ensure_meta_lock = threading.Lock()
_ensure_locks: dict[str, threading.Lock] = {}
_ensured_ok: set[str] = set()


def _truncate(s: str, n: int = RAW_CODE_PREVIEW_MAX) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return s[:n] + f"\n... [truncated {len(s) - n} chars]"


def type_for_class_method(ast: GlobalAst, cm: str) -> AstType | None:
    m = ast.methods.get(cm)
    if not m:
        return None
    for t in ast.types.values():
        if t.simple_name == m.simple_name and any(
            x.class_method == cm for x in t.methods
        ):
            return t
    for t in ast.types.values():
        if t.simple_name == m.simple_name:
            return t
    return None


def build_method_var_resolve_payload(
    ast: GlobalAst,
    cm: str,
    *,
    include_preview: bool = True,
    parse_cache: dict[bytes, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical var-resolve bag for one Class#method.

    ``parse_cache`` (source_bytes → tree root) lets a batch caller parse each
    class source exactly once instead of once per method — on giant files
    (e.g. 440KB service classes with hundreds of methods) the re-parse loop
    was O(methods × file size) and could stall the ingest for hours.
    """
    m = ast.methods.get(cm)
    if not m:
        return {"class_method": cm, "error": "method_not_in_ast"}
    t = type_for_class_method(ast, cm)
    if t is None or not t.source_bytes:
        return {
            "class_method": cm,
            "fqcn": m.fqcn,
            "file_path": m.file_path,
            "start_line": m.start_line,
            "end_line": m.end_line,
            "error": "no_source_bytes",
        }

    source = t.source_bytes
    try:
        if parse_cache is not None and source in parse_cache:
            root = parse_cache[source]
        else:
            root = _parser().parse(source).root_node
            if parse_cache is not None:
                parse_cache[source] = root
    except Exception as exc:
        return {
            "class_method": cm,
            "fqcn": m.fqcn,
            "error": f"parse_failed:{exc}",
        }
    node = find_method_node(
        source, root, method_name=m.method_name, start_line=m.start_line
    )
    locals_map: dict[str, ScopedName] = {}
    if node is not None:
        # Enrich var/lambda typing with project-wide return types + receiver generics.
        returns_map = ast.returns_for()
        generic_map: dict[str, list[str]] = {}
        for c in m.calls:
            if "." in c:
                recv = c.rsplit(".", 1)[0].split(".")[-1]
                if recv not in generic_map:
                    generic_map[recv] = ast.generic_args_for(recv)
        locals_map = extract_method_locals(
            source, node, returns_by_name=returns_map, generic_params=generic_map
        )

    field_scoped = {
        name: ScopedName(name=name, type_name=tn, how="field_map")
        for name, tn in t.field_map.items()
    }
    declared = {x.method_name for x in t.methods}
    call_sites: list[dict[str, Any]] = []
    receivers_used: dict[str, dict[str, Any]] = {}

    for call in m.calls:
        detail = resolve_call_detailed(
            call,
            field_map=t.field_map,
            locals_map=locals_map,
            by_simple=ast.by_simple,
            current_class=t.simple_name,
            declared_methods=declared,
        )
        field_only = resolve_call_detailed(
            call,
            field_map=t.field_map,
            locals_map={},
            by_simple=ast.by_simple,
            current_class=t.simple_name,
            declared_methods=declared,
        )
        site = {
            "raw_call": detail["raw_call"],
            "receiver": detail["receiver"],
            "recv_lookup": detail["recv_lookup"],
            "resolved_type": detail["type_name"],
            "resolve_how": detail["how"],
            "resolved": detail["resolved"],
            "status": detail["status"],
            "skip_reason": detail["skip_reason"],
            "improved_by_locals": (
                (
                    field_only["status"] == "skipped"
                    and detail["status"] in {"ok_local", "no_local_source"}
                )
                or (
                    field_only.get("resolved") != detail.get("resolved")
                    and detail["status"] != "skipped"
                )
                or (
                    not field_only.get("type_name")
                    and bool(detail.get("type_name"))
                    and detail.get("how")
                    in {"local_decl", "new", "cast", "param", "foreach"}
                )
            ),
        }
        call_sites.append(site)

        rkey = detail["recv_lookup"]
        if rkey and rkey not in {"this", "super"}:
            if rkey in locals_map:
                sn = locals_map[rkey]
                receivers_used[rkey] = {
                    "name": rkey,
                    "type_name": sn.type_name,
                    "how": sn.how,
                }
            elif rkey in field_scoped:
                sn = field_scoped[rkey]
                receivers_used[rkey] = {
                    "name": rkey,
                    "type_name": sn.type_name,
                    "how": sn.how,
                }
            elif detail["type_name"]:
                receivers_used[rkey] = {
                    "name": rkey,
                    "type_name": detail["type_name"],
                    "how": detail["how"],
                }
            else:
                receivers_used.setdefault(
                    rkey,
                    {"name": rkey, "type_name": None, "how": "unknown"},
                )
        elif rkey in {"this", "super"}:
            receivers_used[rkey] = {
                "name": rkey,
                "type_name": t.simple_name,
                "how": "this",
            }

    locals_table = [
        {
            "name": sn.name,
            "type_name": sn.type_name,
            "how": sn.how,
            "type_args": list(sn.type_args or []),
            "reassigned_to": sn.reassigned_to,
        }
        for sn in sorted(locals_map.values(), key=lambda x: x.name)
    ]
    fields_table = [
        {"name": n, "type_name": tn, "how": "field_map"}
        for n, tn in sorted(t.field_map.items())
    ]
    out: dict[str, Any] = {
        "class_method": cm,
        "fqcn": m.fqcn,
        "method_name": m.method_name,
        "signature": m.signature,
        "file_path": m.file_path,
        "start_line": m.start_line,
        "end_line": m.end_line,
        "symbol_kind": m.symbol_kind,
        "locals": locals_table,
        "fields": fields_table,
        "receivers_used": sorted(receivers_used.values(), key=lambda x: x["name"]),
        "call_sites": call_sites,
        "resolved_callees": list(m.resolved_callees),
        "stats": {
            "locals": len(locals_table),
            "locals_typed": sum(1 for x in locals_table if x["type_name"]),
            "call_sites": len(call_sites),
            "ok_local": sum(1 for x in call_sites if x["status"] == "ok_local"),
            "no_local_source": sum(
                1 for x in call_sites if x["status"] == "no_local_source"
            ),
            "skipped": sum(1 for x in call_sites if x["status"] == "skipped"),
            "improved_by_locals": sum(
                1 for x in call_sites if x.get("improved_by_locals")
            ),
            "unknown_receivers": sum(
                1
                for x in call_sites
                if x.get("skip_reason") == "unknown_receiver"
            ),
        },
    }
    if include_preview:
        out["raw_code_preview"] = _truncate(m.raw_code)
    return out


def calls_edge_props(edge: CallEdge) -> dict[str, Any]:
    """Slim Neo4j CALLS property bag from a CallEdge."""
    props: dict[str, Any] = {
        "raw_call": edge.raw_call or "",
        "receiver": edge.receiver or "",
        "resolve_how": edge.resolve_how or "unknown",
    }
    if edge.receiver_type:
        props["receiver_type"] = edge.receiver_type
    return props


def upsert_method_var_resolve(
    project_id: str,
    class_method: str,
    payload: dict[str, Any],
    *,
    symbol_id: str | None = None,
) -> None:
    """Idempotent upsert of one method_var_resolve row (project_id scoped)."""
    if not project_id or not class_method:
        return
    init_db()
    now = datetime.utcnow()
    with get_session() as session:
        if symbol_id is None:
            simple, _, meth = class_method.partition("#")
            if simple and meth:
                row = session.scalars(
                    select(SymbolORM)
                    .where(
                        SymbolORM.project_id == project_id,
                        SymbolORM.method_name == meth,
                    )
                    .limit(40)
                ).all()
                for sym in row:
                    if (sym.fqcn or "").split(".")[-1] == simple:
                        symbol_id = sym.id
                        break
        existing = session.scalars(
            select(MethodVarResolveORM).where(
                MethodVarResolveORM.project_id == project_id,
                MethodVarResolveORM.class_method == class_method,
            )
        ).first()
        if existing:
            existing.payload_json = payload
            existing.symbol_id = symbol_id or existing.symbol_id
            existing.updated_at = now
        else:
            session.add(
                MethodVarResolveORM(
                    project_id=project_id,
                    class_method=class_method,
                    symbol_id=symbol_id,
                    payload_json=payload,
                    created_at=now,
                    updated_at=now,
                )
            )


def load_method_var_resolves(
    project_id: str,
    class_methods: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load payload bags keyed by Class#method. Empty dict on miss / DB error."""
    if not project_id:
        return {}
    try:
        init_db()
        with get_session() as session:
            q = select(MethodVarResolveORM).where(
                MethodVarResolveORM.project_id == project_id
            )
            cms = [c for c in (class_methods or []) if c]
            if cms:
                q = q.where(MethodVarResolveORM.class_method.in_(cms))
            rows = session.scalars(q).all()
            out: dict[str, dict[str, Any]] = {}
            for r in rows:
                payload = r.payload_json if isinstance(r.payload_json, dict) else {}
                out[r.class_method] = payload
            return out
    except Exception:
        logger.debug("load_method_var_resolves failed", exc_info=True)
        return {}


_METHOD_TIMEOUT_SEC = 30.0


def persist_ast_var_resolve(
    project_id: str,
    ast: GlobalAst,
    class_methods: Iterable[str] | None = None,
    *,
    include_preview: bool = False,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build + upsert bags for the given methods (default: all with bodies).

    - ``parse_cache`` reuses one tree-sitter parse per class source.
    - per-method timeout skips pathological files instead of stalling the
      whole ingest on a single method.
    - ``is_cancelled`` checked between methods so cancellation takes effect
      promptly (the API only flips the DB row; the loop must look).
    """
    if class_methods is None:
        targets = [
            cm
            for cm, m in ast.methods.items()
            if m.has_body and m.method_name
        ]
    else:
        targets = [c for c in class_methods if c]
    written = 0
    errors = 0
    timed_out = 0
    parse_cache: dict[bytes, Any] = {}
    for idx, cm in enumerate(targets):
        if is_cancelled is not None and is_cancelled():
            logger.warning(
                "persist_ast_var_resolve cancelled at %s/%s methods — aborting",
                idx,
                len(targets),
            )
            break
        holder: dict[str, Any] = {}

        def _run() -> None:
            holder["p"] = build_method_var_resolve_payload(
                ast,
                cm,
                include_preview=include_preview,
                parse_cache=parse_cache,
            )

        worker = threading.Thread(target=_run, daemon=True)
        worker.start()
        worker.join(timeout=_METHOD_TIMEOUT_SEC)
        if worker.is_alive():
            timed_out += 1
            errors += 1
            logger.warning(
                "var-resolve payload timed out after %.0fs for %s — skipping",
                _METHOD_TIMEOUT_SEC,
                cm,
            )
            continue
        payload = holder.get("p") or {}
        if payload.get("error"):
            errors += 1
            continue
        try:
            upsert_method_var_resolve(project_id, cm, payload)
            written += 1
        except Exception:
            errors += 1
            logger.debug("upsert var_resolve failed %s", cm, exc_info=True)
    return {
        "written": written,
        "errors": errors,
        "timed_out": timed_out,
        "targets": len(targets),
    }


def patch_neo4j_calls_from_edges(
    project_id: str,
    ast: GlobalAst,
    *,
    id_by_cm: dict[str, str] | None = None,
    keep_cms: set[str] | None = None,
    source: str = "global-ast",
) -> int:
    """Write/patch Neo4j CALLS with resolve edge props for ok_local edges."""
    from app.analysis.neo4j_store import merge_calls

    if id_by_cm is None:
        id_by_cm = {}
        init_db()
        with get_session() as session:
            for sym in session.scalars(
                select(SymbolORM).where(
                    SymbolORM.project_id == project_id,
                    SymbolORM.method_name.is_not(None),
                )
            ):
                simple = (sym.fqcn or "").split(".")[-1]
                if sym.method_name:
                    id_by_cm[f"{simple}#{sym.method_name}"] = sym.id

    n = 0
    seen: set[tuple[str, str]] = set()
    for e in ast.edges:
        if e.status != "ok_local":
            continue
        if keep_cms is not None and (
            e.caller not in keep_cms or e.callee not in keep_cms
        ):
            continue
        a = id_by_cm.get(e.caller)
        b = id_by_cm.get(e.callee)
        if not a or not b or a == b:
            continue
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        props = calls_edge_props(e)
        try:
            merge_calls(
                project_id,
                a,
                b,
                order=int(e.order) if e.order else None,
                source=source,
                receiver=props.get("receiver"),
                receiver_type=props.get("receiver_type"),
                raw_call=props.get("raw_call"),
                resolve_how=props.get("resolve_how"),
            )
            n += 1
        except Exception:
            logger.debug("merge_calls enrich skip %s -> %s", e.caller, e.callee, exc_info=True)
    return n


def collect_entry_targets(
    ast: GlobalAst,
    entry: str,
    *,
    hop: int = 2,
) -> list[str]:
    """Entry + same-class helpers + hop-N resolved callees (for scoped sync)."""
    from app.analysis.topo_pipeline import normalize_entry_class_method

    entry_cm = normalize_entry_class_method(entry)
    ordered: list[str] = []
    seen: set[str] = set()

    def add(cm: str) -> None:
        if cm in seen or cm not in ast.methods:
            return
        seen.add(cm)
        ordered.append(cm)

    add(entry_cm)
    entry_m = ast.methods.get(entry_cm)
    if not entry_m:
        return ordered
    same = entry_m.simple_name
    for c in entry_m.resolved_callees:
        if c.startswith(f"{same}#"):
            add(c)
    for call in entry_m.calls:
        if "." not in call:
            add(f"{same}#{call}")
        elif call.startswith("this."):
            add(f"{same}#{call.split('.', 1)[1]}")

    frontier = list(ordered)
    for _ in range(max(0, hop)):
        nxt: list[str] = []
        for cm in frontier:
            m = ast.methods.get(cm)
            if not m:
                continue
            for c in m.resolved_callees:
                if c not in seen:
                    add(c)
                    nxt.append(c)
        frontier = nxt
    return ordered


def sync_entry_var_resolve(
    project_id: str,
    entry: str,
    *,
    project_root: Any = None,
    package_roots: list[str] | None = None,
    hop: int = 2,
    parse_workers: int = 8,
    patch_neo4j: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Scoped rebuild: AST → MySQL bags + optional Neo4j CALLS prop patch."""
    from pathlib import Path

    from app.analysis.global_ast import build_global_ast
    from app.core.config import get_settings

    settings = get_settings()
    pid = (project_id or settings.project_id).strip()
    root = Path(project_root) if project_root else settings.project_root_path
    pkg = package_roots or _default_package_roots(settings)

    if progress:
        progress(0, 3, f"AST parse under {root}")
    ast = build_global_ast(
        project_id=pid,
        project_root=root,
        package_roots=pkg,
        parse_workers=parse_workers,
        progress=progress,
    )
    targets = collect_entry_targets(ast, entry, hop=hop)
    if progress:
        progress(1, 3, f"upsert method_var_resolve n={len(targets)}")
    mysql_stats = persist_ast_var_resolve(
        pid, ast, targets, include_preview=False
    )
    neo_n = 0
    if patch_neo4j:
        if progress:
            progress(2, 3, "patch Neo4j CALLS resolve props")
        keep = set(targets)
        # Also keep callees of targets that appear on edges
        for e in ast.edges:
            if e.caller in keep:
                keep.add(e.callee)
        neo_n = patch_neo4j_calls_from_edges(
            pid, ast, keep_cms=keep, source="var-resolve-sync"
        )
    return {
        "project_id": pid,
        "entry": entry,
        "targets": targets,
        "mysql": mysql_stats,
        "neo4j_calls_patched": neo_n,
        "ast": ast.stats() if hasattr(ast, "stats") else {},
    }


def _default_package_roots(settings: Any | None = None) -> list[str]:
    from app.core.config import get_settings

    s = settings or get_settings()
    raw = (getattr(s, "var_resolve_package_roots", None) or "").strip()
    if not raw:
        return ["loan-service/src/main/java"]
    return [p.strip() for p in raw.split(",") if p.strip()]


def count_method_var_resolves(project_id: str) -> int:
    """Return row count for project; 0 on miss / DB error."""
    if not project_id:
        return 0
    try:
        init_db()
        with get_session() as session:
            n = session.scalar(
                select(func.count())
                .select_from(MethodVarResolveORM)
                .where(MethodVarResolveORM.project_id == project_id)
            )
            return int(n or 0)
    except Exception:
        logger.debug("count_method_var_resolves failed", exc_info=True)
        return 0


def entry_var_resolve_ready(
    project_id: str,
    entry: str,
    *,
    min_bags: int | None = None,
) -> bool:
    """True when entry bag exists (no error) and project has enough rows."""
    from app.analysis.topo_pipeline import normalize_entry_class_method
    from app.core.config import get_settings

    if not project_id or not entry:
        return False
    settings = get_settings()
    need = int(min_bags if min_bags is not None else settings.var_resolve_min_bags)
    entry_cm = normalize_entry_class_method(entry)
    bags = load_method_var_resolves(project_id, [entry_cm])
    payload = bags.get(entry_cm) or {}
    if not payload or payload.get("error"):
        return False
    return count_method_var_resolves(project_id) >= max(1, need)


def _ensure_key(project_id: str, entry: str) -> str:
    from app.analysis.topo_pipeline import normalize_entry_class_method

    return f"{project_id}::{normalize_entry_class_method(entry)}"


def ensure_entry_var_resolve(
    project_id: str,
    entry: str,
    *,
    force: bool = False,
    project_root: Any = None,
    package_roots: list[str] | None = None,
    hop: int | None = None,
    parse_workers: int = 8,
    patch_neo4j: bool = True,
    progress: Any = None,
) -> dict[str, Any]:
    """Idempotent scoped sync when bags missing. Safe to call from request path."""
    from app.analysis.topo_pipeline import normalize_entry_class_method
    from app.core.config import get_settings

    settings = get_settings()
    pid = (project_id or settings.project_id or "").strip()
    entry_cm = normalize_entry_class_method((entry or "").strip())
    if not pid or not entry_cm or "#" not in entry_cm:
        return {"skipped": True, "reason": "bad_args", "project_id": pid, "entry": entry_cm}
    if not settings.var_resolve_auto_ensure and not force:
        return {"skipped": True, "reason": "disabled", "project_id": pid, "entry": entry_cm}

    key = _ensure_key(pid, entry_cm)
    if key in _ensured_ok and not force:
        return {"skipped": True, "reason": "cached", "project_id": pid, "entry": entry_cm}

    with _ensure_meta_lock:
        lock = _ensure_locks.setdefault(key, threading.Lock())

    with lock:
        if key in _ensured_ok and not force:
            return {"skipped": True, "reason": "cached", "project_id": pid, "entry": entry_cm}
        if not force and entry_var_resolve_ready(pid, entry_cm):
            _ensured_ok.add(key)
            return {
                "skipped": True,
                "reason": "already_ready",
                "project_id": pid,
                "entry": entry_cm,
                "bag_count": count_method_var_resolves(pid),
            }

        hop_n = int(hop if hop is not None else settings.var_resolve_hop)
        logger.info(
            "ensuring var-resolve entry=%s project=%s hop=%s force=%s",
            entry_cm,
            pid,
            hop_n,
            force,
        )
        result = sync_entry_var_resolve(
            pid,
            entry_cm,
            project_root=project_root,
            package_roots=package_roots or _default_package_roots(settings),
            hop=hop_n,
            parse_workers=parse_workers,
            patch_neo4j=patch_neo4j,
            progress=progress,
        )
        if entry_var_resolve_ready(pid, entry_cm) or int(
            (result.get("mysql") or {}).get("written") or 0
        ):
            _ensured_ok.add(key)
        return {
            "skipped": False,
            "reason": "synced",
            **result,
        }


def maybe_startup_var_resolve_ensure() -> None:
    """Non-blocking: if default entry bags empty, scoped sync in a daemon thread."""
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.var_resolve_auto_ensure or not settings.var_resolve_startup_ensure:
        return
    pid = (settings.project_id or "").strip()
    entry = (settings.var_resolve_default_entry or "").strip()
    if not pid or not entry or "#" not in entry:
        return
    try:
        if entry_var_resolve_ready(pid, entry):
            logger.info(
                "var-resolve ready project=%s entry=%s bags=%s",
                pid,
                entry,
                count_method_var_resolves(pid),
            )
            _ensured_ok.add(_ensure_key(pid, entry))
            return
    except Exception:
        logger.debug("startup var-resolve readiness check failed", exc_info=True)

    def _run() -> None:
        try:
            logger.info(
                "startup scoped var-resolve sync project=%s entry=%s",
                pid,
                entry,
            )
            out = ensure_entry_var_resolve(pid, entry)
            logger.info(
                "startup var-resolve done skipped=%s reason=%s written=%s",
                out.get("skipped"),
                out.get("reason"),
                (out.get("mysql") or {}).get("written"),
            )
        except Exception:
            logger.exception("startup var-resolve ensure failed")

    threading.Thread(
        target=_run, name="var-resolve-startup", daemon=True
    ).start()


def setter_sites_from_bags(
    bags: dict[str, dict[str, Any]],
    *,
    field_name: str | None = None,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Extract set* call sites (optionally matching a field) for change_impact."""
    want = ""
    if field_name:
        want = "set" + field_name[:1].upper() + field_name[1:]
    out: list[dict[str, Any]] = []
    for cm, payload in bags.items():
        if payload.get("error"):
            continue
        for site in payload.get("call_sites") or []:
            raw = str(site.get("raw_call") or "")
            meth = ""
            if "." in raw:
                meth = raw.rsplit(".", 1)[-1]
            else:
                meth = raw
            if not meth.startswith("set") or len(meth) < 4 or not meth[3:4].isupper():
                continue
            if want and meth != want and field_name:
                # Keep analogs (any set*) when looking for placement evidence
                pass
            recv_type = site.get("resolved_type") or ""
            how = site.get("resolve_how") or "unknown"
            out.append(
                {
                    "class_method": cm,
                    "raw_call": raw,
                    "setter": meth,
                    "receiver": site.get("receiver"),
                    "receiver_type": recv_type,
                    "resolve_how": how,
                    "exact_field": bool(want and meth == want),
                }
            )
            if len(out) >= limit:
                return out
    return out
