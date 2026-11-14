"""End-to-end: Global AST → topo+priority AI → MySQL / Milvus / Neo4j.

Triggered by parse-project (ingest) or CLI ``tests/run_topo_ast_pipeline.py``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import select

from app.analysis.common import upsert_description, upsert_milvus_rows
from app.analysis.discover import describe_entity
from app.analysis.global_ast import AstMethod, AstType, CallEdge, GlobalAst, build_global_ast
from app.analysis.neo4j_store import merge_calls, upsert_method_node, write_parsed_type_graph
from app.analysis.topo_scheduler import (
    ScheduleNode,
    SchedulerContext,
    TopoPriorityScheduler,
    build_schedule_nodes_from_ast,
)
from app.core.config import get_settings
from app.core.utils import truncate
from app.db.mysql import get_session, init_db
from app.llm.factory import get_parse_llm_provider
from app.llm.tracing import traceable
from app.models.orm import SymbolDescORM, SymbolORM
from app.parser.models import ParsedMethod, ParsedType
from app.parser.symbol_store import persist_types, symbol_id_for, upsert_project

logger = logging.getLogger("app.analysis.topo_pipeline")

ProgressFn = Callable[[int, int, str], None]

# Optional flow_key → Class#method aliases for scoped pilots (CLI / config only).
# Keep empty: pass explicit --entry Class#method; no product-domain defaults here.
KNOWN_FLOW_KEY_ENTRIES: dict[str, list[str]] = {}


def resolve_flow_key_entries(flow_key: str) -> list[str]:
    fk = (flow_key or "").strip()
    if not fk:
        return []
    return list(KNOWN_FLOW_KEY_ENTRIES.get(fk) or KNOWN_FLOW_KEY_ENTRIES.get(fk.lower()) or [])


def normalize_entry_class_method(entry: str) -> str:
    """Accept 'Class#method' or 'fqcn.Class#method' → 'Class#method'."""
    raw = (entry or "").strip()
    if not raw or "#" not in raw:
        return raw
    left, right = raw.rsplit("#", 1)
    simple = left.split(".")[-1].strip()
    return f"{simple}#{right.strip()}"


def call_closure_methods(
    ast: GlobalAst,
    entries: list[str],
    *,
    skip: Callable[[AstMethod], bool] | None = None,
) -> tuple[list[AstMethod], list[str]]:
    """BFS from entry Class#method(s) over ``resolved_callees`` (call closure).

    Returns (methods in discovery order, unresolved entry keys).
    """
    skip_fn = skip or (lambda _m: False)
    unresolved: list[str] = []
    seeds: list[str] = []
    for raw in entries:
        cm = normalize_entry_class_method(raw)
        if not cm:
            continue
        if cm in ast.methods:
            seeds.append(cm)
            continue
        # Interface → Impl alias (e.g. WithdrawalsService#loan)
        if "#" in cm:
            cls, meth = cm.split("#", 1)
            impl = f"{cls}Impl#{meth}"
            if impl in ast.methods:
                seeds.append(impl)
                continue
        unresolved.append(raw)

    ordered: list[AstMethod] = []
    seen: set[str] = set()
    queue: list[str] = list(dict.fromkeys(seeds))
    while queue:
        cm = queue.pop(0)
        if cm in seen:
            continue
        m = ast.methods.get(cm)
        if m is None:
            continue
        seen.add(cm)
        if not skip_fn(m):
            ordered.append(m)
        for callee in m.resolved_callees:
            if callee not in seen and callee in ast.methods:
                queue.append(callee)
    return ordered, unresolved


DEEP_NODE_PROMPT = """你是资深 Java/Spring「业务+研发」双视角代码分析专家。必须严格按源码做深度解析，禁止空泛套话。

输出严格 JSON（不要 markdown）：
{{
  "important": true/false,
  "important_types": ["validation|status_transition|external_call|mq_publish|mq_consume|db_write|branch|other_important"],
  "class_method": "类名#方法名",
  "business_domain": "短中文分类名",
  "summary_business_clear": "【业务·精简】2～4句",
  "summary_business_detail": "【业务·详细】分点",
  "summary_dev_clear": "【研发·精简】入参→校验→调用→写库→返回",
  "summary_dev_detail": "【研发·详细】完整数据流",
  "summary": "检索用短摘要，必须含「类名#方法名」",
  "business_tables": ["表/实体"],
  "table_field_changes": [{{"table_or_entity":"","field":"","from_value":"","to_value":"","op":""}}],
  "validation_points": ["具体校验"],
  "status_transitions": ["状态从A到B"],
  "external_calls": ["外呼"],
  "data_flow_narrative": "数据流叙述"
}}
若提供了 callee_ai_results：本方法只做编排串联，下游细节直接引用，禁止臆造。
若有 _callee_failed / _upstream_fail_flag：在摘要中注明下游分析不完整。
"""

DTO_PROMPT = """你是资深 Java 建模专家。根据 Model/DTO/VO/Entity 源码输出严格 JSON：
{{
  "tags":["DTO|Entity","入参|出参|持久化"],
  "summary":"一句话角色与关键字段",
  "business_domain":"短中文分类名",
  "table_name":"",
  "key_fields":["核心字段"]
}}
不要编造源码中不存在的字段。
"""


def _loads_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    return json.loads(text)


async def _llm_json(system: str, user: str, *, max_tokens: int = 3500) -> dict[str, Any]:
    llm = get_parse_llm_provider()
    content = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    content = content if isinstance(content, str) else str(content)
    try:
        data = _loads_json(content)
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("LLM JSON parse failed; heuristic fallback")
    return {
        "summary": truncate(content, 500),
        "summary_business_clear": truncate(content, 300),
        "summary_dev_clear": "",
        "important": False,
        "important_types": [],
    }


def _ast_type_to_parsed(t: AstType) -> ParsedType:
    return ParsedType(
        kind=t.kind,
        package=t.package,
        simple_name=t.simple_name,
        fqcn=t.fqcn,
        file_path=t.file_path,
        module_name=t.module_name,
        start_line=t.start_line,
        end_line=t.end_line,
        annotations=list(t.annotations),
        fields=list(t.fields),
        methods=[
            ParsedMethod(
                name=m.method_name,
                signature=m.signature,
                start_line=m.start_line,
                end_line=m.end_line,
                annotations=list(m.annotations),
                calls=list(m.calls),
                raw_code=m.raw_code,
            )
            for m in t.methods
        ],
        symbol_kind=t.symbol_kind,
    )


def _ast_method_to_parsed(m: AstMethod, t: AstType | None) -> tuple[ParsedType, ParsedMethod]:
    pm = ParsedMethod(
        name=m.method_name,
        signature=m.signature,
        start_line=m.start_line,
        end_line=m.end_line,
        annotations=list(m.annotations),
        calls=list(m.calls),
        raw_code=m.raw_code,
    )
    if t is None:
        t_fake = ParsedType(
            kind="class",
            package="",
            simple_name=m.simple_name,
            fqcn=m.fqcn,
            file_path=m.file_path,
            module_name=m.module_name,
            start_line=m.start_line,
            end_line=m.end_line,
            methods=[pm],
            symbol_kind=m.symbol_kind,
        )
        return t_fake, pm
    slim = ParsedType(
        kind=t.kind,
        package=t.package,
        simple_name=t.simple_name,
        fqcn=t.fqcn,
        file_path=t.file_path,
        module_name=t.module_name,
        start_line=t.start_line,
        end_line=t.end_line,
        annotations=list(t.annotations),
        fields=list(t.fields),
        methods=[pm],
        symbol_kind=t.symbol_kind or m.symbol_kind,
    )
    return slim, pm


def _entity_node_id(fqcn: str) -> str:
    return f"type:{fqcn}"


def _persist_symbols_from_ast(project_id: str, ast: GlobalAst, *, batch: int = 40) -> int:
    types = ast.to_parsed_types()
    written = 0
    for i in range(0, len(types), batch):
        chunk = types[i : i + batch]
        with get_session() as session:
            written += persist_types(session, project_id, chunk)
        for t in chunk:
            try:
                write_parsed_type_graph(project_id, t)
            except Exception:
                logger.debug("neo4j type skip %s", t.fqcn, exc_info=True)
    return written


def _mysql_has_desc(project_id: str, symbol_id: str) -> dict[str, Any] | None:
    with get_session() as session:
        row = session.scalar(
            select(SymbolDescORM).where(SymbolDescORM.symbol_id == symbol_id)
        )
        if not row or not (row.summary or "").strip():
            return None
        extra = dict(row.extra_json) if isinstance(row.extra_json, dict) else {}
        return {
            "summary": row.summary or "",
            "summary_business_clear": row.summary_business_clear or "",
            "summary_business_detail": row.summary_business_detail or "",
            "summary_dev_clear": row.summary_dev_clear or "",
            "summary_dev_detail": row.summary_dev_detail or "",
            "business_domain": row.business_domain or "",
            "symbol_id": symbol_id,
            **{k: extra[k] for k in (
                "important",
                "important_types",
                "business_tables",
                "table_field_changes",
                "data_flow_narrative",
                "class_method",
            ) if k in extra},
        }


def _format_callee_blob(callee_results: dict[str, dict[str, Any]], *, max_n: int = 12) -> str:
    if not callee_results:
        return ""
    lines = ["callee_ai_results (bottom-up; cite, do not re-invent):"]
    for i, (cm, data) in enumerate(callee_results.items()):
        if i >= max_n:
            lines.append(f"... +{len(callee_results) - max_n} more")
            break
        summary = (
            data.get("summary_business_clear")
            or data.get("summary_dev_clear")
            or data.get("summary")
            or ""
        )
        flag = ""
        if data.get("_callee_failed") or data.get("fail_flag"):
            flag = " [FAILED]"
        elif data.get("_from_cache"):
            flag = " [cached]"
        lines.append(f"- {cm}{flag}: {truncate(str(summary), 220)}")
    return "\n".join(lines)


@traceable(name="topo_ast_pipeline", run_type="chain")
async def run_topo_ast_pipeline(
    project_id: str,
    project_root: Path,
    *,
    package_roots: list[str] | None = None,
    concurrency: int | None = None,
    max_retries: int | None = None,
    max_methods: int | None = None,
    entry_methods: list[str] | None = None,
    flow_key: str | None = None,
    use_llm: bool = True,
    persist_graph_calls: bool = True,
    progress: ProgressFn | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    ast: GlobalAst | None = None,
) -> dict[str, Any]:
    """Full re-parse path: global AST once → topo AI → stores.

    When ``entry_methods`` / ``flow_key`` is set, AI schedule is limited to the
    BFS call-closure from those entries (still topo-ordered within the subgraph).
    AST parse remains global (or package_roots-scoped) so call edges resolve.
    """
    settings = get_settings()
    workers = max(1, min(int(concurrency or settings.parse_concurrency or 8), 32))
    retries = int(max_retries if max_retries is not None else getattr(settings, "topo_max_retries", 2))
    t0 = time.time()
    # Dry-run only needs AST + schedule; avoid MySQL/Neo4j side effects.
    if use_llm or persist_graph_calls:
        init_db()
    root = Path(project_root).resolve()

    entries: list[str] = []
    for e in entry_methods or []:
        e = (e or "").strip()
        if e:
            entries.append(e)
    if flow_key:
        mapped = resolve_flow_key_entries(flow_key)
        if not mapped and not entries:
            raise ValueError(
                f"unknown --flow-key={flow_key!r}; known={sorted(KNOWN_FLOW_KEY_ENTRIES)}"
            )
        for e in mapped:
            if e not in entries:
                entries.append(e)

    if use_llm or persist_graph_calls:
        with get_session() as session:
            upsert_project(session, project_id, project_id, str(root))

    # --- 1. Global AST (once) ---
    if progress:
        progress(0, 5, "building global AST (tree-sitter, once)")
    if ast is None:
        ast = await asyncio.to_thread(
            build_global_ast,
            project_id,
            root,
            package_roots=package_roots,
            parse_workers=min(workers, 16),
            progress=progress,
        )
    ast_stats = ast.stats()
    logger.info("AST stats %s", ast_stats)

    # --- 2. Persist structure symbols (skip on --dry-run) ---
    sym_count = 0
    if use_llm or persist_graph_calls:
        if progress:
            progress(
                1,
                5,
                f"persist symbols types={ast_stats['types']} methods={ast_stats['methods']}",
            )
        sym_count = await asyncio.to_thread(_persist_symbols_from_ast, project_id, ast)
        logger.info("persisted symbols written≈%s", sym_count)
    elif progress:
        progress(1, 5, "dry-run: skip symbol persist")

    # --- 3. Build schedule ---
    from app.parser.java_parser import is_accessor_method

    def _skip_method(m: AstMethod) -> bool:
        # Skip JavaBean accessors and MyBatis SQL fragments (not business methods).
        if is_accessor_method(m.method_name, len(m.parameter_types)):
            return True
        if m.method_name in {"toString", "hashCode", "equals", "builder", "build"}:
            return True
        if m.symbol_kind == "MapperXml" and (
            m.method_name.endswith("_Column_List")
            or m.method_name.startswith("Base_")
            or m.method_name in {"BaseResultMap", "ResultMapWithBLOBs"}
        ):
            return True
        return False

    scope_mode = "all"
    unresolved_entries: list[str] = []
    if entries:
        methods, unresolved_entries = call_closure_methods(
            ast, entries, skip=_skip_method
        )
        scope_mode = "entry_closure"
        if unresolved_entries:
            logger.warning("entry not found in AST: %s", unresolved_entries)
        if not methods:
            raise ValueError(
                f"entry call-closure empty; entries={entries} unresolved={unresolved_entries}"
            )
        logger.info(
            "entry-scoped schedule entries=%s methods=%s unresolved=%s",
            entries,
            len(methods),
            unresolved_entries,
        )
    else:
        methods = [m for m in ast.methods.values() if not _skip_method(m)]

    if max_methods and max_methods > 0 and len(methods) > max_methods:
        if scope_mode == "entry_closure":
            # Keep BFS discovery order (entry first, then depth) so the subgraph
            # stays coherent rather than reshuffling by symbol_kind.
            methods = methods[:max_methods]
            logger.warning(
                "entry closure truncated to max_methods=%s (was larger)",
                max_methods,
            )
        else:
            # Prefer DB/service over controllers for capped pilot runs.
            methods = sorted(
                methods,
                key=lambda m: (
                    0 if m.symbol_kind in {"Mapper", "MapperXml", "Entity"} else
                    1 if m.symbol_kind in {"Service", "Util", "Class"} else 2,
                    m.class_method,
                ),
            )[:max_methods]
            keep = {m.class_method for m in methods}
            # One-hop callees only (still capped soft by not exploding further).
            extra: list[AstMethod] = []
            for m in list(methods):
                for c in m.resolved_callees:
                    if c in keep or c not in ast.methods:
                        continue
                    cm = ast.methods[c]
                    if _skip_method(cm):
                        continue
                    keep.add(c)
                    extra.append(cm)
            methods = methods + extra
        scope_mode = f"{scope_mode}+max_methods"

    scoped = scope_mode != "all"

    # Entity/DTO/Mapper type nodes: when scoped/capped, only those touched by selected methods.
    touched_types: set[str] = {m.fqcn for m in methods}
    for m in methods:
        for c in m.resolved_callees:
            if c in ast.methods:
                touched_types.add(ast.methods[c].fqcn)

    entity_ids: list[str] = []
    entity_meta: dict[str, dict[str, Any]] = {}
    for fqcn, t in ast.types.items():
        is_db = t.symbol_kind in {"Entity", "DTO", "Mapper", "MapperXml"} or t.simple_name.endswith(
            ("Entity", "DO", "PO", "Dmo", "Mapper")
        )
        if not is_db:
            continue
        if scoped and fqcn not in touched_types:
            continue
        eid = _entity_node_id(fqcn)
        entity_ids.append(eid)
        entity_meta[eid] = {
            "kind": t.symbol_kind.lower() if t.symbol_kind else "entity",
            "fqcn": fqcn,
            "simple_name": t.simple_name,
            "file_path": t.file_path,
            "symbol_kind": t.symbol_kind,
        }

    method_meta: dict[str, dict[str, Any]] = {}
    method_callees: dict[str, list[str]] = {}
    keep_cms = {m.class_method for m in methods}
    for m in methods:
        method_meta[m.class_method] = {
            "kind": "method",
            "symbol_kind": m.symbol_kind,
            "fqcn": m.fqcn,
            "simple_name": m.simple_name,
            "method_name": m.method_name,
            "file_path": m.file_path,
            "signature": m.signature,
        }
        method_callees[m.class_method] = [
            c for c in m.resolved_callees if c in keep_cms
        ]

    nodes = build_schedule_nodes_from_ast(
        method_callees=method_callees,
        method_meta=method_meta,
        entity_ids=entity_ids,
        entity_meta=entity_meta,
    )
    logger.info(
        "schedule nodes=%s entities=%s methods=%s edges≈%s",
        len(nodes),
        len(entity_ids),
        len(method_meta),
        sum(len(v) for v in method_callees.values()),
    )

    milvus_rows: list[dict] = []
    milvus_lock = asyncio.Lock()

    async def cache_lookup(nid: str) -> dict[str, Any] | None:
        node = nodes[nid]
        if nid.startswith("type:"):
            fqcn = nid[5:]
            sid = symbol_id_for(project_id, fqcn, None)
        else:
            meta = node.meta
            sid = symbol_id_for(
                project_id,
                str(meta.get("fqcn") or ""),
                str(meta.get("signature") or ""),
                method_name=str(meta.get("method_name") or ""),
            )
        return await asyncio.to_thread(_mysql_has_desc, project_id, sid)

    async def analyze(node: ScheduleNode, ctx: SchedulerContext) -> dict[str, Any]:
        if not use_llm:
            return {
                "summary": f"(dry-run) {node.id}",
                "summary_business_clear": f"dry-run {node.id}",
                "important": False,
            }

        if node.id.startswith("type:"):
            fqcn = node.id[5:]
            t = ast.types.get(fqcn)
            if not t:
                return {"summary": f"missing type {fqcn}", "important": False}
            parsed = _ast_type_to_parsed(t)
            # Prefer existing describe_entity for persist entities
            if t.symbol_kind == "Entity" or t.simple_name.endswith(("Entity", "DO", "PO")):
                desc = await describe_entity(project_id, parsed, project_root=root)
                if desc:
                    async with milvus_lock:
                        milvus_rows.append(desc)
                    return {
                        "summary": desc.get("summary") or desc.get("text") or f"Entity {fqcn}",
                        "summary_business_clear": desc.get("summary") or "",
                        "symbol_id": desc.get("symbol_id"),
                        "business_domain": desc.get("business_domain") or "",
                    }
            # DTO / Mapper type shell
            snippet = ""
            if t.file_path:
                try:
                    snippet = (root / t.file_path).read_text(encoding="utf-8", errors="ignore")[:4000]
                except OSError:
                    snippet = ""
            if t.symbol_kind == "MapperXml" and t.methods:
                snippet = "\n".join(
                    f"-- {m.method_name}\n{m.raw_code[:400]}" for m in t.methods[:8]
                )
            data = await _llm_json(
                DTO_PROMPT,
                f"kind={t.symbol_kind}\nfqcn={fqcn}\nfields={[f.name for f in t.fields][:40]}\n"
                f"tables={t.table_hints[:20]}\nsql_ids={t.sql_ids[:30]}\ncode:\n{snippet}",
                max_tokens=900,
            )
            sid = symbol_id_for(project_id, fqcn, None)
            row = await asyncio.to_thread(
                upsert_description,
                project_id,
                sid,
                summary=str(data.get("summary") or f"{t.symbol_kind} {fqcn}"),
                business_domain=str(data.get("business_domain") or "其他"),
                tags=list(data.get("tags") or [t.symbol_kind or "Type"]) + ["topo-ast"],
                extra={
                    "module_name": t.module_name,
                    "table_name": data.get("table_name"),
                    "key_fields": data.get("key_fields"),
                    "table_hints": t.table_hints,
                    "parse_engine": "tree-sitter-global-ast",
                },
            )
            async with milvus_lock:
                milvus_rows.append(row)
            return {
                "summary": row.get("summary") or data.get("summary"),
                "summary_business_clear": data.get("summary") or "",
                "symbol_id": sid,
                "business_domain": data.get("business_domain") or "",
            }

        # Method node
        cm = node.id
        m = ast.methods.get(cm)
        if not m:
            return {"summary": f"missing method {cm}", "important": False}
        t = ast.types.get(m.fqcn)
        if t is None:
            # try by simple
            for fq in ast.by_simple.get(m.simple_name) or []:
                t = ast.types.get(fq)
                if t:
                    break
        parsed_t, parsed_m = _ast_method_to_parsed(m, t)

        def _persist() -> str:
            with get_session() as session:
                persist_types(session, project_id, [parsed_t])
            return symbol_id_for(
                project_id, parsed_t.fqcn, parsed_m.signature, method_name=parsed_m.name
            )

        mid = await asyncio.to_thread(_persist)
        callee_blob = _format_callee_blob(ctx.callee_results(cm))
        code = m.raw_code or ""
        code_limit = 5500 if m.symbol_kind == "Controller" or len(code) > 2500 else 3800
        fail_note = ""
        if node.fail_flag:
            fail_note = "\nNOTE: lower-level callee analysis failed — mark incompleteness.\n"
        if node.in_cycle:
            fail_note += "\nNOTE: method is in a call cycle — partial callee context only.\n"

        data = await _llm_json(
            DEEP_NODE_PROMPT,
            f"kind={m.symbol_kind}\nfqcn={m.fqcn}\nclass_method={cm}\n"
            f"known_calls={list(m.calls)[:20]}\nresolved_callees={m.resolved_callees[:20]}\n"
            f"{fail_note}{callee_blob}\ncode:\n{truncate(code, code_limit)}",
            max_tokens=4200,
        )
        data["class_method"] = cm
        summary = str(
            data.get("summary")
            or f"{cm}: {data.get('summary_business_clear') or data.get('summary_dev_clear') or ''}"
        )
        tags = ["topo-ast", "双向深度解析", m.symbol_kind]
        if data.get("important"):
            tags = ["重要节点"] + tags + list(data.get("important_types") or [])
        if node.fail_flag:
            tags.append("downstream_fail")
        if node.in_cycle:
            tags.append("call_cycle")

        row = await asyncio.to_thread(
            upsert_description,
            project_id,
            mid,
            summary=summary,
            business_domain=str(data.get("business_domain") or "其他"),
            tags=tags,
            extra={
                "important": bool(data.get("important")),
                "important_types": data.get("important_types") or [],
                "class_method": cm,
                "fqcn": m.fqcn,
                "module_name": m.module_name,
                "method_name": m.method_name,
                "method_sig": m.signature,
                "business_tables": data.get("business_tables") or [],
                "table_field_changes": data.get("table_field_changes") or [],
                "status_transitions": data.get("status_transitions") or [],
                "external_calls": data.get("external_calls") or [],
                "validation_points": data.get("validation_points") or [],
                "data_flow_narrative": data.get("data_flow_narrative") or "",
                "summary_business_clear": data.get("summary_business_clear"),
                "summary_business_detail": data.get("summary_business_detail"),
                "summary_dev_clear": data.get("summary_dev_clear"),
                "summary_dev_detail": data.get("summary_dev_detail"),
                "parse_engine": "tree-sitter-global-ast",
                "parse_mode": "topo-priority",
                "fail_flag": node.fail_flag,
                "in_cycle": node.in_cycle,
                "priority": int(node.priority),
            },
        )
        try:
            await asyncio.to_thread(
                upsert_method_node,
                project_id,
                parsed_t,
                parsed_m,
                important=bool(data.get("important")),
                important_types=list(data.get("important_types") or []),
                # Narratives stay in MySQL; Neo4j keeps id + structural props only.
                store_narratives=False,
            )
        except Exception:
            logger.debug("upsert_method_node failed %s", cm, exc_info=True)

        async with milvus_lock:
            milvus_rows.append(row)
        data["summary"] = summary
        data["symbol_id"] = mid
        return data

    if progress:
        progress(
            2,
            5,
            f"topo AI schedule nodes={len(nodes)} concurrency={workers}",
        )

    scheduler = TopoPriorityScheduler(
        nodes,
        project_id=project_id,
        concurrency=workers,
        max_retries=retries,
        analyze=analyze,
        cache_lookup=cache_lookup if use_llm else None,
        progress=progress,
        is_cancelled=is_cancelled,
    )
    sched_stats = await scheduler.run()

    # --- 4. Milvus batch ---
    if is_cancelled is not None and is_cancelled():
        logger.warning("topo pipeline cancelled after scheduler — skipping milvus/neo4j writes")
        return {
            "project_id": project_id,
            "ast": ast_stats,
            "symbols_persisted": sym_count,
            "schedule": sched_stats.as_dict(),
            "milvus_rows": 0,
            "neo4j_calls": 0,
            "method_var_resolve": {},
            "elapsed_sec": round(time.time() - t0, 2),
            "concurrency": workers,
            "max_retries": retries,
            "package_roots": list(package_roots or []),
            "scope_mode": scope_mode,
            "entry_methods": list(entries),
            "flow_key": flow_key or "",
            "unresolved_entries": list(unresolved_entries),
            "scheduled_methods": len(method_meta),
            "scheduled_entities": len(entity_ids),
            "cancelled": True,
        }

    if progress:
        progress(3, 5, f"milvus upsert rows={len(milvus_rows)}")
    milvus_ok = 0
    if milvus_rows and use_llm:
        try:
            await asyncio.to_thread(
                upsert_milvus_rows, milvus_rows, is_cancelled=is_cancelled
            )
            milvus_ok = len(milvus_rows)
        except Exception:
            logger.exception("milvus upsert failed")

    # --- 5. Neo4j CALLS from AST edges ---
    calls_written = 0
    if persist_graph_calls and not (is_cancelled is not None and is_cancelled()):
        if progress:
            progress(4, 5, "neo4j CALLS from AST edges")
        id_by_cm: dict[str, str] = {}
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

        edge_triples: list[tuple[str, str, int, CallEdge]] = []
        for e in ast.edges:
            if e.status != "ok_local":
                continue
            if scoped and (e.caller not in keep_cms or e.callee not in keep_cms):
                continue
            a = id_by_cm.get(e.caller)
            b = id_by_cm.get(e.callee)
            if a and b and a != b:
                edge_triples.append((a, b, int(e.order or 0), e))
        # Dedupe by (a,b) keep first order
        seen_ab: set[tuple[str, str]] = set()
        uniq_triples: list[tuple[str, str, int, Any]] = []
        for a, b, ord_, e in edge_triples:
            key = (a, b)
            if key in seen_ab:
                continue
            seen_ab.add(key)
            uniq_triples.append((a, b, ord_, e))

        def _write_calls(triples: list[tuple[str, str, int, Any]]) -> int:
            from app.analysis.neo4j_store import batch_merge_calls

            return batch_merge_calls(project_id, triples, is_cancelled=is_cancelled)

        calls_written = await asyncio.to_thread(_write_calls, uniq_triples)

        # Persist per-method var-resolve bags (MySQL) for methods in this write set.
        try:
            from app.analysis.var_resolve import persist_ast_var_resolve

            mvr_cms: list[str] | None
            if scoped:
                mvr_cms = sorted(keep_cms)
            else:
                # Full graph: only methods that appear as callers on written edges
                mvr_cms = sorted({e.caller for _, _, _, e in uniq_triples})
            if progress and not (is_cancelled is not None and is_cancelled()):
                progress(4, 5, f"var-resolve bags n={len(mvr_cms)}")
            mvr_stats = await asyncio.to_thread(
                persist_ast_var_resolve,
                project_id,
                ast,
                mvr_cms,
                include_preview=False,
                is_cancelled=is_cancelled,
            )
            logger.info("method_var_resolve upsert %s", mvr_stats)
        except Exception:
            logger.debug("method_var_resolve upsert failed", exc_info=True)
            mvr_stats = {}
    else:
        mvr_stats = {}

    elapsed = time.time() - t0
    if progress:
        progress(5, 5, f"done in {elapsed:.1f}s")

    result = {
        "project_id": project_id,
        "ast": ast_stats,
        "symbols_persisted": sym_count,
        "schedule": sched_stats.as_dict(),
        "milvus_rows": milvus_ok,
        "neo4j_calls": calls_written,
        "method_var_resolve": mvr_stats if isinstance(mvr_stats, dict) else {},
        "elapsed_sec": round(elapsed, 2),
        "concurrency": workers,
        "max_retries": retries,
        "package_roots": list(package_roots or []),
        "scope_mode": scope_mode,
        "entry_methods": list(entries),
        "flow_key": flow_key or "",
        "unresolved_entries": list(unresolved_entries),
        "scheduled_methods": len(method_meta),
        "scheduled_entities": len(entity_ids),
    }
    logger.info("topo_ast_pipeline finished %s", result)
    return result
