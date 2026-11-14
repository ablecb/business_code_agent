"""Shared helpers for AI exploration pipeline: file index, persist, desc/milvus."""

from __future__ import annotations

import logging
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError, OperationalError

from app.core.utils import truncate
from app.db.milvus_client import get_milvus
from app.db.mysql import get_session
from app.llm.factory import get_embedding_provider
from app.models.orm import SymbolDescORM, SymbolORM, SymbolTagORM, TagDictORM
from app.parser.java_parser import parse_java_file
from app.parser.models import ParsedType
from app.parser.scanner import scan_java_files
from app.parser.symbol_store import persist_types, symbol_id_for

logger = logging.getLogger("app.analysis.common")

# Serialize tag_dict inserts within this process (concurrent preload often races on
# the same tags like DTO/VO → MySQL deadlock 1213).
_tag_create_lock = threading.Lock()


def _is_mysql_deadlock(exc: BaseException) -> bool:
    orig = getattr(exc, "orig", None)
    args = getattr(orig, "args", None) or getattr(exc, "args", ())
    return bool(args) and args[0] == 1213


def ensure_tag(session, project_id: str, tag: str, category: str | None = None) -> int:
    """Resolve tag id; create in a short locked transaction if missing.

    Uses a separate commit for new tags so concurrent upsert_description sessions
    do not interleave INSERT into tag_dict (avoids deadlock on uk_tag).
    """
    tag = (tag or "").strip()
    if not tag:
        raise ValueError("empty tag")
    row = session.scalar(
        select(TagDictORM).where(TagDictORM.project_id == project_id, TagDictORM.tag == tag)
    )
    if row:
        return int(row.id)

    with _tag_create_lock:
        # Dedicated short session so the unique row is visible to other writers ASAP.
        for attempt in range(5):
            try:
                with get_session() as tag_session:
                    existing = tag_session.scalar(
                        select(TagDictORM).where(
                            TagDictORM.project_id == project_id, TagDictORM.tag == tag
                        )
                    )
                    if existing:
                        tid = int(existing.id)
                        break
                    tag_session.add(
                        TagDictORM(project_id=project_id, tag=tag, category=category)
                    )
                    tag_session.flush()
                    existing = tag_session.scalar(
                        select(TagDictORM).where(
                            TagDictORM.project_id == project_id, TagDictORM.tag == tag
                        )
                    )
                    tid = int(existing.id) if existing else 0
                if tid:
                    break
            except IntegrityError:
                # Another writer inserted the same tag — re-read.
                with get_session() as tag_session:
                    existing = tag_session.scalar(
                        select(TagDictORM).where(
                            TagDictORM.project_id == project_id, TagDictORM.tag == tag
                        )
                    )
                    if existing:
                        tid = int(existing.id)
                        break
                tid = 0
            except OperationalError as e:
                if not _is_mysql_deadlock(e) or attempt >= 4:
                    raise
                time.sleep(0.05 * (attempt + 1))
                tid = 0
        else:
            tid = 0

    if not tid:
        # Final read on caller's session (may see committed row).
        row = session.scalar(
            select(TagDictORM).where(
                TagDictORM.project_id == project_id, TagDictORM.tag == tag
            )
        )
        if not row:
            raise RuntimeError(f"failed to ensure tag {tag!r} for project {project_id}")
        return int(row.id)

    # Refresh caller's view if identity map is stale.
    row = session.get(TagDictORM, tid)
    if row is None:
        row = session.scalar(
            select(TagDictORM).where(
                TagDictORM.project_id == project_id, TagDictORM.tag == tag
            )
        )
    return int(row.id) if row is not None else tid


def list_java_index(project_root: Path) -> list[dict[str, str]]:
    """Lightweight file index for AI discovery (path + simple name hints)."""
    root = Path(project_root).resolve()
    files = scan_java_files(root)
    index: list[dict[str, str]] = []
    for fp in files:
        try:
            rel = fp.relative_to(root).as_posix()
        except ValueError:
            rel = str(fp)
        simple = fp.stem
        index.append({"path": rel, "simple_name": simple, "abs": str(fp)})
    return index


def read_file_snippet(project_root: Path, rel_path: str, *, limit: int = 6000) -> str:
    path = Path(project_root) / rel_path
    if not path.exists():
        # try abs
        path = Path(rel_path)
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def guess_role_from_name(simple_name: str, path: str) -> str:
    n = (simple_name or "").lower()
    p = (path or "").lower()
    if "entity" in n or "/entity/" in p or "\\entity\\" in p or n.endswith("do") or n.endswith("po"):
        return "Entity"
    if "controller" in n:
        return "Controller"
    if "listener" in n or "listenner" in n:
        return "Listener"
    if n.endswith("job") or "xxl" in n or "/job/" in p:
        return "Job"
    if "feign" in n or n.endswith("client"):
        return "Feign"
    if n.endswith("mapper"):
        return "Mapper"
    if "service" in n:
        return "Service"
    return "Class"


def parse_and_persist(project_id: str, project_root: Path, rel_or_abs: str | Path) -> list[ParsedType]:
    root = Path(project_root).resolve()
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = root / path
    if not path.exists():
        logger.warning("parse skip missing file %s", path)
        return []
    result = parse_java_file(path, root)
    if result.errors:
        for e in result.errors[:3]:
            logger.warning("parse error: %s", e)
    if not result.types:
        return []
    with get_session() as session:
        persist_types(session, project_id, result.types)
    return result.types


def upsert_description(
    project_id: str,
    symbol_id: str,
    *,
    summary: str,
    business_domain: str | None = None,
    tags: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write MySQL symbol_desc (+ tags) and return a Milvus row dict.

    Upserts by unique symbol_id (not autoincrement id). Safe under concurrent writers.
    Retries MySQL deadlocks (1213) from concurrent tag/desc writes.
    """
    from datetime import datetime

    tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
    # Stable order reduces lock inversion across concurrent symbol_tag inserts.
    tags = list(dict.fromkeys(tags))
    extra = dict(extra or {})
    domain = business_domain or "其他"

    fqcn = ""
    method_sig = ""
    method_name = ""
    simple_name = ""
    module_name = str(extra.get("module_name") or "").strip()

    def _apply(session, existing: SymbolDescORM | None, ordered_tags: list[str]) -> str:
        nonlocal domain
        merged_extra = (
            dict(existing.extra_json) if existing and isinstance(existing.extra_json, dict) else {}
        )
        # merge list-valued flow keys instead of clobbering
        for list_key in ("flow_keys", "flow_roles", "important_types", "business_tables"):
            old = merged_extra.get(list_key)
            new = extra.get(list_key)
            if isinstance(old, list) and isinstance(new, list):
                merged = list(old)
                for x in new:
                    if x not in merged:
                        merged.append(x)
                merged_extra[list_key] = merged
            elif list_key in extra:
                merged_extra[list_key] = extra[list_key]
        for k, v in extra.items():
            if k in ("flow_keys", "flow_roles", "important_types", "business_tables"):
                continue
            if k == "flow_doc" and isinstance(v, dict) and isinstance(merged_extra.get("flow_doc"), dict):
                merged_extra["flow_doc"] = {**merged_extra["flow_doc"], **v}
            else:
                merged_extra[k] = v

        domain = business_domain or (existing.business_domain if existing else None) or "其他"
        now = datetime.utcnow()
        biz_c = str(extra.get("summary_business_clear") or "").strip() or None
        biz_d = str(extra.get("summary_business_detail") or "").strip() or None
        dev_c = str(extra.get("summary_dev_clear") or "").strip() or None
        dev_d = str(extra.get("summary_dev_detail") or "").strip() or None
        if existing:
            existing.summary = summary
            if biz_c is not None:
                existing.summary_business_clear = biz_c
            if biz_d is not None:
                existing.summary_business_detail = biz_d
            if dev_c is not None:
                existing.summary_dev_clear = dev_c
            if dev_d is not None:
                existing.summary_dev_detail = dev_d
            existing.business_domain = domain
            existing.extra_json = merged_extra
            existing.updated_at = now
        else:
            session.add(
                SymbolDescORM(
                    project_id=project_id,
                    symbol_id=symbol_id,
                    summary=summary,
                    summary_business_clear=biz_c,
                    summary_business_detail=biz_d,
                    summary_dev_clear=dev_c,
                    summary_dev_detail=dev_d,
                    business_domain=domain,
                    extra_json=merged_extra,
                    created_at=now,
                    updated_at=now,
                )
            )
        if ordered_tags:
            session.execute(delete(SymbolTagORM).where(SymbolTagORM.symbol_id == symbol_id))
            for tag in ordered_tags:
                tid = ensure_tag(session, project_id, tag, category=domain)
                session.add(SymbolTagORM(project_id=project_id, symbol_id=symbol_id, tag_id=tid))
        return domain

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            with get_session() as session:
                # Resolve module_name from SymbolORM before desc write (MySQL + Milvus).
                sym = session.get(SymbolORM, symbol_id)
                fqcn = sym.fqcn if sym else ""
                method_sig = sym.method_sig if sym else ""
                method_name = sym.method_name if sym else ""
                simple_name = sym.simple_name if sym else ""
                module_name = str(
                    extra.get("module_name") or (sym.module_name if sym else "") or ""
                ).strip()
                work_tags = list(tags)
                if module_name:
                    extra["module_name"] = module_name
                    if module_name not in work_tags:
                        work_tags.append(module_name)
                ordered_tags = sorted(dict.fromkeys(work_tags))
                tags = ordered_tags

                existing = session.scalar(
                    select(SymbolDescORM).where(SymbolDescORM.symbol_id == symbol_id)
                )
                try:
                    domain = _apply(session, existing, ordered_tags)
                    session.flush()
                except IntegrityError:
                    # Concurrent insert won the race — reload and update.
                    session.rollback()
                    existing = session.scalar(
                        select(SymbolDescORM).where(SymbolDescORM.symbol_id == symbol_id)
                    )
                    if not existing:
                        raise
                    domain = _apply(session, existing, ordered_tags)
            break
        except OperationalError as e:
            last_err = e
            if not _is_mysql_deadlock(e) or attempt >= 5:
                raise
            logger.warning(
                "mysql deadlock on upsert_description symbol=%s attempt=%s; retry",
                symbol_id,
                attempt + 1,
            )
            time.sleep(0.05 * (attempt + 1) + 0.02 * (hash(symbol_id) % 7))
        except IntegrityError as e:
            last_err = e
            if attempt >= 5:
                raise
            logger.warning(
                "integrity race on upsert_description symbol=%s attempt=%s; retry",
                symbol_id,
                attempt + 1,
            )
            time.sleep(0.05 * (attempt + 1))
    else:
        if last_err:
            raise last_err

    class_method = str(extra.get("class_method") or "")
    if not class_method and simple_name and method_name:
        class_method = f"{simple_name}#{method_name}"
    http_paths = extra.get("http_paths") or []
    if isinstance(http_paths, list):
        path_blob = " ".join(str(p) for p in http_paths if p)
    else:
        path_blob = str(http_paths or "")
    biz_c = str(extra.get("summary_business_clear") or "").strip()
    dev_c = str(extra.get("summary_dev_clear") or "").strip()
    milvus_row = {
        "id": symbol_id,
        "project_id": project_id,
        "symbol_id": symbol_id,
        "fqcn": (fqcn or "")[:500],
        "module_name": (module_name or "")[:128],
        "tags": ",".join(tags)[:1000],
        "text": truncate(
            f"{domain} {' '.join(tags)} {summary} "
            f"module={module_name} class_method={class_method} fqcn={fqcn} "
            f"method={method_name or ''} http_path={path_blob} {method_sig or ''}\n"
            f"业务精简: {biz_c}\n研发精简: {dev_c}",
            2000,
        ),
    }
    return milvus_row


def upsert_milvus_rows(
    rows: list[dict[str, Any]],
    *,
    is_cancelled: Callable[[], bool] | None = None,
) -> None:
    """Embed + upsert rows into Milvus. ``is_cancelled`` checked between
    embedding batches so a cancelled ingest stops promptly instead of
    grinding through thousands of embedding HTTP calls."""
    if not rows:
        return
    by_id = {r["id"]: r for r in rows}
    uniq = list(by_id.values())
    # SiliconFlow / BGE-zh reject long inputs (~512 tokens); keep embed text short.
    for r in uniq:
        t = str(r.get("text") or "").strip() or " "
        t = t.replace("\x00", " ")
        if len(t) > 450:
            r["text"] = t[:450]
        else:
            r["text"] = t
    try:
        embedder = get_embedding_provider()
        milvus = get_milvus()
        embeddings: list[list[float]] = []
        # Smaller batches reduce 400s on proxy / payload limits.
        batch_size = 4
        for i in range(0, len(uniq), batch_size):
            if is_cancelled is not None and is_cancelled():
                logger.warning(
                    "milvus embed cancelled at %s/%s rows — aborting upsert",
                    i,
                    len(uniq),
                )
                return
            chunk = uniq[i : i + batch_size]
            try:
                embeddings.extend(embedder.embed([r["text"] for r in chunk]))
            except Exception:
                logger.exception("milvus embed batch failed at %s; retry one-by-one", i)
                for r in chunk:
                    try:
                        embeddings.extend(embedder.embed([str(r.get("text") or " ")[:300]]))
                    except Exception:
                        logger.exception("embed skip id=%s", r.get("id"))
                        embeddings.append([0.0] * (milvus.dim or 1024))
        if len(embeddings) != len(uniq):
            logger.error(
                "embed count mismatch %s vs %s; abort milvus upsert",
                len(embeddings),
                len(uniq),
            )
            return
        milvus.dim = len(embeddings[0]) if embeddings else milvus.dim
        milvus.upsert(uniq, embeddings)
    except Exception:
        logger.exception("milvus upsert failed; MySQL descriptions kept")


def find_files_by_simple_name(index: list[dict[str, str]], simple: str) -> list[dict[str, str]]:
    simple_l = simple.lower()
    exact = [x for x in index if x["simple_name"].lower() == simple_l]
    if exact:
        return exact
    # FooService → FooServiceImpl
    impl = [x for x in index if x["simple_name"].lower() in {f"{simple_l}impl", f"{simple_l}serviceimpl"}]
    if impl:
        return impl
    return [x for x in index if simple_l in x["simple_name"].lower()][:5]


def resolve_call_targets(
    *,
    caller: ParsedType,
    call: str,
    index: list[dict[str, str]],
) -> list[tuple[str, str]]:
    """Map rough call 'fooService.bar' → [(fqcn_hint_or_simple, method)]."""
    call = (call or "").strip()
    if not call or "(" in call:
        call = call.split("(")[0].strip()
    if "." in call:
        obj, method = call.rsplit(".", 1)
    else:
        obj, method = "", call
    if not method or method in {"if", "for", "while", "super", "this", "new"}:
        return []

    # this.foo / super.foo → same-class
    if not obj or obj in {"this", "super"}:
        return [(caller.simple_name, method)]

    # field type lookup
    field_types = {f.name: f.type_name for f in caller.fields}
    type_hint = field_types.get(obj) or field_types.get(obj[0].lower() + obj[1:] if obj else "")
    if not type_hint and obj:
        # FooService style variable → FooService
        type_hint = obj[0].upper() + obj[1:] if obj else ""
    simple = re.sub(r"<.*?>", "", type_hint or "").split(".")[-1].strip()
    if not simple:
        return [(caller.simple_name, method)]  # same-class call
    matches = find_files_by_simple_name(index, simple)
    if not matches:
        return [(simple, method)]
    return [(m["simple_name"], method) for m in matches[:2]]


# Generic noise: do not descend (not business-specific keywords)
_NOISE_TYPE_SUFFIX = (
    "Util",
    "Utils",
    "Tool",
    "Tools",
    "Helper",
    "Builder",
    "Constants",
    "Constant",
    "Mapper",
    "Wrapper",
)
_NOISE_TYPE_EXACT = {
    "String",
    "Objects",
    "Arrays",
    "Collections",
    "CollectionUtils",
    "CollectionUtil",
    "ObjectUtil",
    "JSON",
    "JSONObject",
    "JSONArray",
    "JSONTool",
    "JSONUtil",
    "log",
    "Logger",
    "Log",
    "System",
    "Map",
    "HashMap",
    "List",
    "ArrayList",
    "Set",
    "Optional",
    "ResponseBuilder",
    "ResponseStateUtil",
    "ParamSdkUtil",
    "ParamSdkConfig",
    "ParamQueryUtil",
    "Wrappers",
    "BeanUtils",
    "NumberUtil",
    "DateUtil",
    "LogUtils",
}
_NOISE_METHODS = {
    "info",
    "debug",
    "warn",
    "error",
    "trace",
    "put",
    "get",
    "format",
    "trim",
    "valueOf",
    "toString",
    "equals",
    "hashCode",
    "equalsIgnoreCase",
    "toJSONString",
    "toJsonStr",
    "getInnerCode",
    "getInnerMsg",
    "builder",
    "build",
    "buildSuccess",
    "buildFailure",
}


def is_noise_callee(simple: str, method: str) -> bool:
    """True if callee is framework/util noise and should not consume BFS budget."""
    simple = (simple or "").split(".")[-1]
    method = (method or "").strip()
    if not method:
        return True
    if method in _NOISE_METHODS:
        return True
    # JavaBean getX/setX on plain types — keep on Service/Controller/business types
    if re.match(r"^(get|set)[A-Z]", method):
        keep_suffix = (
            "Service",
            "ServiceImpl",
            "Controller",
            "Fragment",
            "Processor",
            "Handler",
            "Manager",
            "Facade",
            "Client",
            "Decorator",
        )
        if any(simple.endswith(s) for s in keep_suffix):
            return False
        return True
    if simple in _NOISE_TYPE_EXACT:
        return True
    if any(simple.endswith(sfx) for sfx in _NOISE_TYPE_SUFFIX):
        return True
    if simple.endswith("Enum") and method.startswith("get"):
        return True
    return False
