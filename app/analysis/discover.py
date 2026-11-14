"""Discover entities and data-source entries via AI + file index (no bulk AST dump)."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from app.analysis.common import (
    guess_role_from_name,
    list_java_index,
    parse_and_persist,
    upsert_description,
    upsert_milvus_rows,
)
from app.analysis.neo4j_store import write_parsed_type_graph
from app.analysis.pilot_config import load_pilot_entries
from app.core.utils import truncate
from app.llm.factory import get_parse_llm_provider
from app.llm.tracing import traceable
from app.parser.models import ParsedMethod, ParsedType

logger = logging.getLogger("app.analysis.discover")

# Annotations that mark the true start of data/business logic.
ENTRY_ANNOTATIONS = frozenset(
    {
        "RequestMapping",
        "GetMapping",
        "PostMapping",
        "PutMapping",
        "DeleteMapping",
        "PatchMapping",
        "RabbitListener",
        "KafkaListener",
        "RocketMQMessageListener",
        "JmsListener",
        "XxlJob",
        "Scheduled",
        "JobHandler",
        "StreamListener",
    }
)

CLASSIFY_PROMPT = """你是 Java/Spring 项目结构分析专家。根据文件名列表，挑选数据库实体与数据入口文件。
输出严格 JSON（不要 markdown）：
{
  "entities": [{"path":"相对路径","reason":"为何是实体"}],
  "controllers": [{"path":"...","reason":"..."}],
  "listeners": [{"path":"...","reason":"MQ消费入口"}],
  "jobs": [{"path":"...","reason":"定时任务/XxlJob"}],
  "other_entries": [{"path":"...","role":"Feign|其他","reason":"..."}]
}
规则：只从给定列表选 path；实体优先 *Entity/*DO/*PO 或 entity 包；入口优先 Controller/Listener/Job（数据逻辑最开始处）。
每类最多返回 40 条。
"""


def is_data_entry_method(m: ParsedMethod, class_kind: str) -> bool:
    """True if this method is a data-logic entry (HTTP / MQ / Job / schedule)."""
    if _is_scaffold_crud_method(m):
        return False
    if m.http_mappings:
        return True
    ann_names = {a.name for a in (m.annotations or [])}
    if ann_names & ENTRY_ANNOTATIONS:
        return True
    if class_kind == "Controller" and any(n.endswith("Mapping") for n in ann_names):
        return True
    if class_kind == "Listener" and any("Listener" in n for n in ann_names):
        return True
    if class_kind == "Job" and (
        "XxlJob" in ann_names or "Scheduled" in ann_names or m.name in {"execute", "run", "handle"}
    ):
        return True
    return False


# Generic code-gen / MyBatis-Plus style scaffolding — not useful as flow roots.
_SCAFFOLD_CRUD_NAMES = frozenset(
    {
        "list",
        "listpage",
        "insert",
        "updatebyid",
        "getbyid",
        "deletebyid",
        "deletebyids",
        "selectbyid",
        "page",
    }
)


def _mapping_path_hints(m: ParsedMethod) -> list[str]:
    paths: list[str] = []
    for hm in m.http_mappings or []:
        paths.append(str(hm.get("path") or ""))
    for a in m.annotations or []:
        if a.name.endswith("Mapping") or a.name in ENTRY_ANNOTATIONS:
            paths.append(str(a.values.get("value") or a.values.get("path") or ""))
    return paths


def _class_mapping_prefix(t: ParsedType) -> str:
    """Class-level @RequestMapping prefix, e.g. api/v1/."""
    for a in t.annotations or []:
        if a.name in {"RequestMapping", "GetMapping", "PostMapping"} or a.name.endswith("Mapping"):
            raw = str(a.values.get("value") or a.values.get("path") or "").strip().strip('"').strip("'")
            if raw:
                return raw.strip("/")
    return ""


def _resolved_http_paths(t: ParsedType, m: ParsedMethod) -> list[str]:
    """Full HTTP paths for recall (class prefix + method mapping)."""
    prefix = _class_mapping_prefix(t)
    out: list[str] = []
    seen: set[str] = set()
    for raw in _mapping_path_hints(m):
        p = str(raw or "").strip().strip('"').strip("'").strip("/")
        if not p and not prefix:
            continue
        full = f"{prefix}/{p}".strip("/") if prefix else p
        if full and full not in seen:
            seen.add(full)
            out.append(full)
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    if prefix and prefix not in seen and not out:
        out.append(prefix)
    return out[:6]


_DOC_ANN_NAMES = frozenset({"Api", "ApiOperation", "Tag", "Tags", "Operation", "Schema"})
_CN_HINT_RE = re.compile(r"[\u4e00-\u9fff]{2,}")


def _ann_doc_bits(annotations: list[Any] | None) -> list[str]:
    """Extract Swagger/OpenAPI title strings from annotation objects or dicts."""
    bits: list[str] = []
    for a in annotations or []:
        if isinstance(a, dict):
            name = str(a.get("name") or "")
            values = a.get("values") if isinstance(a.get("values"), dict) else {}
        else:
            name = getattr(a, "name", None) or ""
            values = getattr(a, "values", None) or {}
        if name not in _DOC_ANN_NAMES:
            continue
        if not isinstance(values, dict):
            continue
        for key in ("value", "summary", "description", "name", "title", "tags", "notes"):
            raw = str(values.get(key) or "").strip().strip('"').strip("'")
            if raw and raw not in bits:
                bits.append(raw)
    return bits


def leading_comment_hints(
    file_path: str | Path | None,
    start_line: int | None,
    *,
    project_root: Path | None = None,
    method_name: str | None = None,
    max_chars: int = 120,
) -> str:
    """Chinese text from javadoc / // comments immediately above a method."""
    if not file_path:
        return ""
    path = Path(str(file_path))
    if not path.is_file() and project_root is not None:
        cand = Path(project_root) / str(file_path)
        if cand.is_file():
            path = cand
    if not path.is_file():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return ""

    def _from_index(i: int) -> str:
        while i >= 0 and (not lines[i].strip() or lines[i].lstrip().startswith("@")):
            i -= 1
        chunk: list[str] = []
        if i >= 0 and "*/" in lines[i]:
            end = i
            while i >= 0 and "/*" not in lines[i]:
                i -= 1
            if i >= 0:
                for line in lines[i : end + 1]:
                    s = re.sub(r"^\s*/?\*+/?\s?", "", line).strip()
                    if not s or s.startswith("@"):
                        continue
                    if _CN_HINT_RE.search(s):
                        chunk.append(s)
        elif i >= 0 and lines[i].lstrip().startswith("//"):
            while i >= 0 and lines[i].lstrip().startswith("//"):
                s = lines[i].lstrip()[2:].strip()
                if s and _CN_HINT_RE.search(s):
                    chunk.insert(0, s)
                i -= 1
        return truncate(" ".join(chunk), max_chars)

    # Prefer locating the method signature by name (AI start_line is often wrong).
    if method_name:
        pat = re.compile(
            rf"^\s*(?:public|protected|private|default)?\s*[\w.<>,\[\]\s]+\s+{re.escape(method_name)}\s*\("
        )
        for idx, line in enumerate(lines):
            if pat.search(line):
                hint = _from_index(idx - 1)
                if hint:
                    return hint

    if not start_line or start_line < 2:
        return ""
    return _from_index(min(start_line, len(lines)) - 2)


def _entry_doc_hints(t: ParsedType, m: ParsedMethod) -> str:
    """Pull Swagger titles + leading comments so natural-language recall works.

    Framework-only — no industry vocabulary; uses whatever the project wrote on
    @Api / @ApiOperation / @Tag / @Operation / javadoc.
    """
    bits = _ann_doc_bits(list(t.annotations or []) + list(m.annotations or []))
    comment = leading_comment_hints(t.file_path, m.start_line, method_name=m.name)
    if comment:
        for part in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_/\-]{2,}", comment):
            if part not in bits:
                bits.append(part)
    return " ".join(bits[:8])


def _is_scaffold_crud_method(m: ParsedMethod) -> bool:
    """Drop generic CRUD scaffolding so entry budget goes to real API mappings."""
    name = (m.name or "").lower()
    if name not in _SCAFFOLD_CRUD_NAMES:
        return False
    paths = [p.lower().replace('"', "").replace("'", "").strip().strip("/") for p in _mapping_path_hints(m)]
    if not paths:
        return True
    for p in paths:
        head = (p.split("/")[0] if p else "") or name
        if head.replace("{id}", "").strip("/") in _SCAFFOLD_CRUD_NAMES or name in p:
            return True
    return False


def _entry_role_tag(m: ParsedMethod, class_kind: str) -> str:
    ann_names = {a.name for a in (m.annotations or [])}
    if m.http_mappings or any(n.endswith("Mapping") for n in ann_names):
        return "HTTP入口"
    if ann_names & {"RabbitListener", "KafkaListener", "RocketMQMessageListener", "JmsListener", "StreamListener"}:
        return "MQ入口"
    if ann_names & {"XxlJob", "Scheduled", "JobHandler"} or class_kind == "Job":
        return "Job入口"
    return f"{class_kind}入口"

ENTITY_DESC_PROMPT = """你是资深 Java 持久层专家。根据 Entity 源码输出严格 JSON：
{{
  "tags":["实体","落库"],
  "summary":"一句话：表名、关键字段、状态字段及可能取值",
  "business_domain":"短中文分类名",
  "domain_action":"reuse|create",
  "table_name":"表名或空",
  "status_fields":["status等状态/枚举字段名"],
  "status_values":{{"status":["可选枚举值或常量名"]}},
  "key_fields":["id", "..."]
}}
重点找出能表示状态流转的字段（status/state 等）及源码中出现的枚举/常量取值。不要编造源码中不存在的字段。
{domain_rule}
"""


def _kw_match(text: str, keywords: tuple[str, ...]) -> bool:
    """Empty keywords = match any (kind filter already applied)."""
    if not keywords:
        return True
    blob = text or ""
    blob_l = blob.lower()
    for kw in keywords:
        if not kw:
            continue
        if kw.isascii():
            if kw.lower() in blob_l:
                return True
        elif kw in blob:
            return True
    return False


@traceable(name="classify_files_llm", run_type="llm")
async def _llm_classify(index: list[dict[str, str]]) -> dict[str, Any]:
    llm = get_parse_llm_provider()
    # send compact list
    lines = [f"{x['path']}\t{x['simple_name']}" for x in index[:800]]
    user = "文件列表(path\\tsimple_name):\n" + "\n".join(lines)
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": CLASSIFY_PROMPT},
                {"role": "user", "content": truncate(user, 14000)},
            ],
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
        if not isinstance(data, dict):
            raise ValueError("bad classify json")
        return data
    except Exception:
        logger.exception("classify LLM failed; using name heuristics")
        return _heuristic_classify(index)


def _heuristic_classify(index: list[dict[str, str]]) -> dict[str, Any]:
    out: dict[str, list] = {
        "entities": [],
        "controllers": [],
        "listeners": [],
        "jobs": [],
        "other_entries": [],
    }
    for x in index:
        role = guess_role_from_name(x["simple_name"], x["path"])
        item = {"path": x["path"], "reason": f"name hint {role}"}
        if role == "Entity":
            out["entities"].append(item)
        elif role == "Controller":
            out["controllers"].append(item)
        elif role == "Listener":
            out["listeners"].append(item)
        elif role == "Job":
            out["jobs"].append(item)
        elif role in {"Feign"}:
            out["other_entries"].append({**item, "role": role})
    for k in out:
        out[k] = out[k][:40]
    return out


def _paths_from(data: dict[str, Any], key: str) -> list[str]:
    items = data.get(key) or []
    paths: list[str] = []
    for it in items:
        if isinstance(it, dict) and it.get("path"):
            paths.append(str(it["path"]).replace("\\", "/"))
        elif isinstance(it, str):
            paths.append(it.replace("\\", "/"))
    # unique preserve order
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


async def describe_entity(
    project_id: str,
    t: ParsedType,
    *,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    from app.analysis.common import read_file_snippet
    from app.analysis.domain_catalog import domain_instruction, list_known_domains, resolve_business_domain
    from app.parser.symbol_store import symbol_id_for

    llm = get_parse_llm_provider()
    snippet = ""
    if project_root and t.file_path:
        snippet = read_file_snippet(project_root, t.file_path, limit=5000)
    # heuristic status fields from ParsedField names
    field_names = [f.name for f in t.fields]
    heuristic_status = [
        n for n in field_names if "status" in n.lower() or "state" in n.lower()
    ]
    known = list_known_domains(project_id)
    system = ENTITY_DESC_PROMPT.format(domain_rule=domain_instruction(known))
    user = (
        f"fqcn={t.fqcn}\nsimple_name={t.simple_name}\n"
        f"annotations={[a.name for a in t.annotations]}\n"
        f"fields={field_names}\n"
        f"code:\n{truncate(snippet, 4500)}"
    )
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=900,
            response_format={"type": "json_object"},
        )
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
    except Exception:
        data = {
            "tags": ["实体", "落库", "Entity"],
            "summary": f"实体类 {t.simple_name}",
            "business_domain": "",
            "domain_action": "create",
            "status_fields": heuristic_status,
        }
    if not data.get("status_fields") and heuristic_status:
        data["status_fields"] = heuristic_status
    domain = resolve_business_domain(
        project_id,
        str(data.get("business_domain") or ""),
        domain_action=str(data.get("domain_action") or ""),
        known=known,
    )
    sid = symbol_id_for(project_id, t.fqcn, None)
    return await asyncio.to_thread(
        upsert_description,
        project_id,
        sid,
        summary=str(data.get("summary") or f"实体 {t.fqcn}"),
        business_domain=domain,
        tags=[str(x) for x in (data.get("tags") or ["实体", "Entity"])],
        extra={
            "module_name": t.module_name or "",
            "domain_action": data.get("domain_action"),
            **{
                k: data.get(k)
                for k in ("table_name", "status_fields", "status_values", "key_fields")
                if k in data
            },
        },
    )


@traceable(name="discover_entities", run_type="chain")
async def discover_and_index_entities(
    project_id: str,
    project_root: Path,
    *,
    classify: dict[str, Any] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    index = list_java_index(project_root)
    if classify is None:
        if progress:
            progress(0, 3, "AI classifying entities/entries from file list")
        classify = await _llm_classify(index)
    entity_paths = _paths_from(classify, "entities")
    # also add heuristic entity names missing from AI
    for x in index:
        if guess_role_from_name(x["simple_name"], x["path"]) == "Entity" and x["path"] not in entity_paths:
            entity_paths.append(x["path"])
    entity_paths = entity_paths[:60]

    milvus_rows: list[dict] = []
    parsed_n = 0
    for i, rel in enumerate(entity_paths):
        if progress:
            progress(i, max(len(entity_paths), 1), f"entity {rel}")
        # enrich snippet into AI desc — parse file with AI
        types = await asyncio.to_thread(parse_and_persist, project_id, project_root, rel)
        for t in types:
            t.symbol_kind = "Entity"
            # ensure MySQL kind reflects Entity
            from app.db.mysql import get_session
            from app.models.orm import SymbolORM
            from app.parser.symbol_store import symbol_id_for

            with get_session() as session:
                cid = symbol_id_for(project_id, t.fqcn, None)
                row = session.get(SymbolORM, cid)
                if row:
                    row.kind = "Entity"
            await asyncio.to_thread(write_parsed_type_graph, project_id, t)
            row = await describe_entity(project_id, t, project_root=project_root)
            if row:
                milvus_rows.append(row)
            parsed_n += 1

    upsert_milvus_rows(milvus_rows)
    if progress:
        progress(len(entity_paths), max(len(entity_paths), 1), f"entities done={parsed_n}")
    return {
        "entity_files": len(entity_paths),
        "entities_parsed": parsed_n,
        "classify": {k: len(_paths_from(classify, k)) for k in ("entities", "controllers", "listeners", "jobs")},
        "_classify_raw": classify,
        "_index": index,
    }


@traceable(name="discover_entries", run_type="chain")
async def discover_and_index_entries(
    project_id: str,
    project_root: Path,
    *,
    classify: dict[str, Any] | None = None,
    index: list[dict[str, str]] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    max_entry_methods: int = 120,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Parse Controller / Listener / Job entry files → mark data-source methods.

    Persists Java class + method symbols to MySQL, Endpoint/Job/Queue to Neo4j,
    and entry descriptions to MySQL + Milvus (always with 类#方法).
    Concurrent via PARSE_CONCURRENCY (async semaphore + thread parse).

    Note: every discovered entry is written to Milvus. max_entry_methods only
    limits how many become recursive-flow roots (BFS), not Milvus coverage.
    """
    from app.analysis.neo4j_store import upsert_method_node
    from app.core.config import get_settings
    from app.db.mysql import get_session
    from app.models.orm import SymbolORM
    from app.parser.symbol_store import symbol_id_for

    settings = get_settings()
    workers = max(1, int(concurrency or settings.parse_concurrency or 4))
    workers = max(1, min(workers, 16))

    index = index or list_java_index(project_root)
    if classify is None:
        if progress:
            progress(0, 3, "AI classifying data-source entries from file list")
        classify = await _llm_classify(index)
    paths: list[tuple[str, str]] = []
    for key, role in (
        ("controllers", "Controller"),
        ("listeners", "Listener"),
        ("jobs", "Job"),
    ):
        for p in _paths_from(classify, key):
            paths.append((p, role))
    # heuristics fill
    for x in index:
        role = guess_role_from_name(x["simple_name"], x["path"])
        if role in {"Controller", "Listener", "Job"} and not any(p == x["path"] for p, _ in paths):
            paths.append((x["path"], role))

    # Per-role budgets so Listeners/Jobs cannot push Controllers out of the window.
    # (Previous flat uniq[:80] dropped late Controllers after classify filled with listeners.)
    role_caps = {"Controller": 60, "Listener": 40, "Job": 30}
    by_role: dict[str, list[tuple[str, str]]] = {k: [] for k in role_caps}
    seen: set[str] = set()
    for p, role in paths:
        if p in seen or role not in by_role:
            continue
        seen.add(p)
        by_role[role].append((p, role))
    uniq: list[tuple[str, str]] = []
    for role, cap in role_caps.items():
        uniq.extend(by_role[role][:cap])

    lock = asyncio.Lock()
    entry_methods: list[dict[str, Any]] = []
    milvus_rows: list[dict] = []
    done_files = 0
    total_files = max(len(uniq), 1)
    sem = asyncio.Semaphore(workers)

    logger.info(
        "discover_entries files=%s concurrency=%s (PARSE_CONCURRENCY) max_methods=%s",
        len(uniq),
        workers,
        max_entry_methods,
    )

    def _index_one_type(t: ParsedType, role: str) -> list[dict[str, Any]]:
        """Sync work for one parsed type → entry method dicts + side effects."""
        if t.symbol_kind == "Class":
            t.symbol_kind = role
        with get_session() as session:
            cid = symbol_id_for(project_id, t.fqcn, None)
            row = session.get(SymbolORM, cid)
            if row:
                row.kind = t.symbol_kind
        write_parsed_type_graph(project_id, t)

        candidates = [m for m in t.methods if is_data_entry_method(m, t.symbol_kind)]
        if not candidates:
            candidates = [
                m
                for m in t.methods
                if m.name
                and not m.name.startswith(("get", "set", "is", "toString", "hashCode", "equals"))
            ][:3]

        local: list[dict[str, Any]] = []
        local_milvus: list[dict] = []
        for m in candidates:
            sid = symbol_id_for(project_id, t.fqcn, m.signature)
            class_method = f"{t.simple_name}#{m.name}"
            role_tag = _entry_role_tag(m, t.symbol_kind)
            http_paths = _resolved_http_paths(t, m)
            path_txt = (" path=" + ",".join(http_paths)) if http_paths else ""
            doc_hints = _entry_doc_hints(t, m)
            doc_txt = f" {doc_hints}" if doc_hints else ""
            flow_key = {
                "Controller": "http_entry",
                "Listener": "mq_entry",
                "Job": "job_entry",
            }.get(t.symbol_kind, "data_entry")
            summary = (
                f"【数据源头·{role_tag}】{class_method}（{t.fqcn}）{path_txt}{doc_txt}"
            ).strip()
            upsert_method_node(
                project_id,
                t,
                m,
                important=True,
                important_types=["other_important"],
                flow_keys=[flow_key],
                is_entry=True,
            )
            desc = upsert_description(
                project_id,
                sid,
                summary=summary,
                business_domain="数据源头",
                tags=[t.symbol_kind, "数据源头", "入口", role_tag, class_method]
                + ([f"path:{p}" for p in http_paths[:3]]),
                extra={
                    "entry": True,
                    "data_source": True,
                    "entry_role": role_tag,
                    "class_method": class_method,
                    "fqcn": t.fqcn,
                    "method_name": m.name,
                    "method_sig": m.signature,
                    "flow_key": flow_key,
                    "http_paths": http_paths,
                    "api_doc": doc_hints or None,
                },
            )
            local_milvus.append(desc)
            with get_session() as session:
                mrow = session.get(SymbolORM, sid)
                if mrow and t.symbol_kind in {"Controller", "Listener", "Job"}:
                    mrow.kind = t.symbol_kind
            local.append(
                {
                    "symbol_id": sid,
                    "fqcn": t.fqcn,
                    "simple_name": t.simple_name,
                    "method_name": m.name,
                    "method_sig": m.signature,
                    "kind": t.symbol_kind,
                    "file_path": t.file_path,
                    "raw_code": m.raw_code,
                    "calls": list(m.calls or []),
                    "annotations": [{"name": a.name, "values": a.values} for a in m.annotations],
                    "class_method": class_method,
                    "flow_key": flow_key,
                    "flow_role": "main",
                    "pilot_id": f"{t.simple_name}_{m.name}",
                    "entry_role": role_tag,
                    "_milvus": desc,
                }
            )
        return local

    async def _worker(rel: str, role: str) -> None:
        nonlocal done_files
        async with sem:
            logger.info("entry parse start [%s] %s", role, rel)
            try:
                types = await asyncio.to_thread(parse_and_persist, project_id, project_root, rel)
                batch: list[dict[str, Any]] = []
                for t in types:
                    batch.extend(await asyncio.to_thread(_index_one_type, t, role))
                async with lock:
                    # Always index every entry to Milvus/MySQL-linked vectors.
                    # max_entry_methods only caps which ones become recursive-flow roots.
                    for item in batch:
                        mv = item.pop("_milvus", None)
                        if mv:
                            milvus_rows.append(mv)
                        if len(entry_methods) < max_entry_methods:
                            entry_methods.append(item)
                        else:
                            logger.debug(
                                "entry method over flow-root budget, still milvus-indexed: %s#%s",
                                item.get("simple_name"),
                                item.get("method_name"),
                            )
            except Exception:
                logger.exception("entry parse failed %s", rel)
            finally:
                async with lock:
                    done_files += 1
                    done = done_files
                logger.info("entry parse done %s/%s %s", done, total_files, rel)
                if progress:
                    progress(done, total_files, f"entry {role} {rel} (workers={workers})")

    await asyncio.gather(*[_worker(rel, role) for rel, role in uniq])

    upsert_milvus_rows(milvus_rows)
    if progress:
        progress(total_files, total_files, f"entries methods={len(entry_methods)} concurrency={workers}")
    return {
        "entry_files": len(uniq),
        "entry_methods": len(entry_methods),
        "concurrency": workers,
        "_entry_methods": entry_methods,
        "_index": index,
    }


def select_pilot_entries(entry_methods: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Pick configured pilot entries; return (selected, missing_ids).

    Empty-keyword main pilots match every Controller — take one HTTP entry method
    per Controller class (not only hits[0]), otherwise sibling APIs never get FlowDocs.
    """
    pilots = load_pilot_entries()
    selected: list[dict[str, Any]] = []
    missing: list[str] = []
    seen_cm: set[str] = set()
    max_per_pilot = 40
    for pilot in pilots:
        hits = []
        for em in entry_methods:
            if pilot.kinds and em.get("kind") not in pilot.kinds:
                continue
            blob = " ".join(
                [
                    em.get("fqcn") or "",
                    em.get("simple_name") or "",
                    em.get("method_name") or "",
                    em.get("file_path") or "",
                    json.dumps(em.get("annotations") or {}, ensure_ascii=False),
                    (em.get("raw_code") or "")[:800],
                ]
            )
            if _kw_match(blob, pilot.keywords):
                hits.append(em)
        if not hits:
            missing.append(pilot.id)
            continue
        # Prefer mapping-like public entry methods over private helpers when many hit.
        def _entry_rank(em: dict[str, Any]) -> tuple:
            name = (em.get("method_name") or "").lower()
            helper = 1 if re.match(
                r"^(is|check|valid|generate|build|fill|parse|convert|to)", name
            ) else 0
            return (helper, name)

        hits_sorted = sorted(hits, key=_entry_rank)
        take_multi = (not pilot.keywords) or pilot.flow_role in {
            "main",
            "callback",
            "notice",
            "async",
        }
        picked = 0
        for em in hits_sorted:
            fqcn = (em.get("fqcn") or "").strip()
            method = (em.get("method_name") or "").strip()
            cm = f"{fqcn.split('.')[-1]}#{method}" if method else fqcn
            if not cm or cm in seen_cm:
                continue
            # Empty-keyword main: one method per Controller class.
            if not pilot.keywords and pilot.flow_role == "main":
                class_key = f"main:{fqcn}"
                if class_key in seen_cm:
                    continue
                seen_cm.add(class_key)
            seen_cm.add(cm)
            chosen = dict(em)
            chosen["pilot_id"] = (
                f"{pilot.id}_{method}" if take_multi and method else pilot.id
            )
            chosen["flow_key"] = pilot.flow_key
            chosen["flow_role"] = pilot.flow_role
            selected.append(chosen)
            picked += 1
            if not take_multi or picked >= max_per_pilot:
                break
        if picked == 0:
            missing.append(pilot.id)
    return selected, missing
