from __future__ import annotations

from datetime import datetime
from typing import Callable

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.mysql import insert as mysql_insert
from sqlalchemy.orm import Session

from app.core.utils import stable_id
from app.db.retry import run_with_deadlock_retry
from app.models.orm import ProjectORM, SymbolDescORM, SymbolORM, SymbolTagORM, TagDictORM
from app.parser.java_parser import is_accessor_method
from app.parser.models import ParsedType


def upsert_project(session: Session, project_id: str, name: str, root_path: str) -> None:
    row = session.get(ProjectORM, project_id)
    if row is None:
        session.add(
            ProjectORM(id=project_id, name=name, root_path=root_path, description=name)
        )
    else:
        row.name = name
        row.root_path = root_path


def clear_symbols(
    session: Session,
    project_id: str,
    *,
    batch_size: int = 500,
    should_continue: Callable[[], bool] | None = None,
) -> int:
    """Delete symbols (and related tag/desc rows) for a project in batches.

    A single huge DELETE on `symbol` (with MEDIUMTEXT raw_code) can look like a hang
    and also block other connections; batching + commit keeps each lock short.
    """
    session.execute(delete(SymbolTagORM).where(SymbolTagORM.project_id == project_id))
    session.execute(delete(SymbolDescORM).where(SymbolDescORM.project_id == project_id))
    session.execute(delete(TagDictORM).where(TagDictORM.project_id == project_id))
    session.commit()

    total = 0
    while True:
        if should_continue and not should_continue():
            break
        ids = list(
            session.scalars(
                select(SymbolORM.id).where(SymbolORM.project_id == project_id).limit(batch_size)
            )
        )
        if not ids:
            break
        batch_ids = list(ids)

        def _delete_batch() -> None:
            try:
                session.execute(delete(SymbolORM).where(SymbolORM.id.in_(batch_ids)))
                session.commit()
            except Exception:
                session.rollback()
                raise

        run_with_deadlock_retry(_delete_batch)
        total += len(batch_ids)
    return total


def count_symbols(session: Session, project_id: str) -> int:
    return int(
        session.scalar(
            select(func.count()).select_from(SymbolORM).where(SymbolORM.project_id == project_id)
        )
        or 0
    )


def _extract_method_name(method_sig: str | None) -> str | None:
    """Pull method name from either `fqcn#name(` or a raw Java declaration."""
    if not method_sig:
        return None
    s = method_sig.strip()
    if "#" in s and "(" in s:
        # com.foo.Bar#baz(Request x)
        mid = s.split("#", 1)[1]
        return mid.split("(", 1)[0].strip() or None
    if "(" in s:
        # public Response foo(@RequestBody Request x)
        before = s.split("(", 1)[0].strip()
        tok = before.split()[-1] if before.split() else ""
        tok = tok.split(".")[-1]
        return tok or None
    return None


_METHOD_ID_CACHE: dict[tuple[str, str, str], str] = {}


def find_existing_method_symbol_id(
    session: Session, project_id: str, fqcn: str, method_name: str
) -> str | None:
    """Reuse discover-time symbol id when the same fqcn#method already exists."""
    rows = list(
        session.scalars(
            select(SymbolORM).where(
                SymbolORM.project_id == project_id,
                SymbolORM.fqcn == fqcn,
                SymbolORM.method_name == method_name,
            )
        )
    )
    if not rows:
        return None
    if len(rows) == 1:
        return rows[0].id
    # Prefer discover-style signature: fqcn#method(...)
    for r in rows:
        sig = r.method_sig or ""
        if f"#{method_name}(" in sig:
            return r.id
    ids = [r.id for r in rows]
    described = list(
        session.scalars(select(SymbolDescORM.symbol_id).where(SymbolDescORM.symbol_id.in_(ids)))
    )
    if described:
        return described[0]
    return rows[0].id


def symbol_id_for(
    project_id: str,
    fqcn: str,
    method_sig: str | None,
    *,
    method_name: str | None = None,
    reuse_existing: bool = True,
) -> str:
    """Stable symbol id. Prefer existing MySQL row for same fqcn#method_name.

    Avoids duplicate ids when source_reader emits a declaration-style signature
    while AI/discover used ``fqcn#method(params)``.
    """
    name = method_name or _extract_method_name(method_sig)
    if reuse_existing and name:
        cache_key = (project_id, fqcn, name)
        cached = _METHOD_ID_CACHE.get(cache_key)
        if cached:
            return cached
        try:
            from app.db.mysql import get_session

            with get_session() as session:
                existing = find_existing_method_symbol_id(session, project_id, fqcn, name)
            if existing:
                _METHOD_ID_CACHE[cache_key] = existing
                return existing
        except Exception:
            pass
    if name:
        return stable_id(project_id, fqcn, name)
    return stable_id(project_id, fqcn, method_sig or "")


def clear_method_id_cache() -> None:
    _METHOD_ID_CACHE.clear()


def _symbol_rows(project_id: str, types: list[ParsedType]) -> list[dict]:
    by_id: dict[str, dict] = {}
    now = datetime.utcnow()
    for t in types:
        class_id = symbol_id_for(project_id, t.fqcn, None)
        ann_list: list[dict] = [{"name": a.name, "values": a.values} for a in t.annotations]
        field_names = sorted({f.name for f in t.fields if f.name})
        if field_names:
            ann_list.append({"name": "_fields", "values": {"names": ",".join(field_names)}})
        class_raw = None
        if t.symbol_kind == "MapperXml":
            class_raw = t.extra.get("raw_code") or None
            if class_raw:
                class_raw = class_raw[:16_000_000]
        by_id[class_id] = {
            "id": class_id,
            "project_id": project_id,
            "module_name": t.module_name,
            "kind": t.symbol_kind,
            "fqcn": t.fqcn,
            "simple_name": t.simple_name,
            "method_name": None,
            "method_sig": "",
            "file_path": t.file_path,
            "start_line": t.start_line,
            "end_line": t.end_line,
            "annotations": ann_list,
            "modifiers": t.modifiers,
            "parent_fqcn": None,
            "raw_code": class_raw,
            "created_at": now,
            "updated_at": now,
        }
        for m in t.methods:
            param_count = 0
            if "(" in m.signature and m.signature.endswith(")"):
                inner = m.signature[m.signature.rfind("(") + 1 : -1].strip()
                param_count = 0 if not inner else inner.count(",") + 1
            if is_accessor_method(m.name, param_count):
                continue
            mid = symbol_id_for(project_id, t.fqcn, m.signature)
            kind = t.symbol_kind
            ann_names = {a.name for a in m.annotations}
            if "RabbitListener" in ann_names:
                kind = "Listener"
            elif "XxlJob" in ann_names:
                kind = "Job"
            by_id[mid] = {
                "id": mid,
                "project_id": project_id,
                "module_name": t.module_name,
                "kind": kind,
                "fqcn": t.fqcn,
                "simple_name": t.simple_name,
                "method_name": m.name,
                "method_sig": m.signature,
                "file_path": t.file_path,
                "start_line": m.start_line,
                "end_line": m.end_line,
                "annotations": [{"name": a.name, "values": a.values} for a in m.annotations],
                "modifiers": m.modifiers,
                "parent_fqcn": t.fqcn,
                "raw_code": m.raw_code[:16_000_000] if m.raw_code else None,
                "created_at": now,
                "updated_at": now,
            }
    return list(by_id.values())


def persist_types(session: Session, project_id: str, types: list[ParsedType]) -> int:
    """Upsert symbols via MySQL ON DUPLICATE KEY UPDATE (avoids merge SELECT races)."""
    rows = _symbol_rows(project_id, types)
    if not rows:
        return 0

    table = SymbolORM.__table__
    update_cols = [
        c.name
        for c in table.columns
        if c.name not in {"id", "created_at"}
    ]

    chunk_size = 40
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        stmt = mysql_insert(table).values(chunk)
        stmt = stmt.on_duplicate_key_update(
            **{name: stmt.inserted[name] for name in update_cols}
        )
        session.execute(stmt)
    return len(rows)


def list_symbols(session: Session, project_id: str, kind: str | None = None) -> list[SymbolORM]:
    stmt = select(SymbolORM).where(SymbolORM.project_id == project_id)
    if kind:
        stmt = stmt.where(SymbolORM.kind == kind)
    return list(session.scalars(stmt))
