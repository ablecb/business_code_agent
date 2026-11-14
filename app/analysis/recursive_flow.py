"""Recursive AI flow analysis from data-source entries → MySQL / Milvus / Neo4j."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from app.analysis.common import (
    guess_role_from_name,
    is_noise_callee,
    parse_and_persist,
    resolve_call_targets,
    upsert_description,
    upsert_milvus_rows,
)
from app.analysis.discover import describe_entity
from app.analysis.neo4j_store import (
    merge_calls,
    upsert_method_node,
    write_parsed_type_graph,
)
from app.analysis.source_reader import load_method, load_type_shell, merge_method, resolve_type_file
from app.core.config import get_settings
from app.core.utils import stable_id, truncate
from app.db.mysql import get_session
from app.llm.factory import get_parse_llm_provider
from app.llm.tracing import traceable
from app.models.orm import SymbolORM
from app.parser.models import ParsedMethod, ParsedType
from app.parser.symbol_store import persist_types, symbol_id_for

logger = logging.getLogger("app.analysis.recursive")

IMPORTANT_TYPES = (
    "validation",
    "status_transition",
    "external_call",
    "mq_publish",
    "mq_consume",
    "db_write",
    "branch",
    "other_important",
)

NODE_PROMPT = """你是资深 Java/Spring 代码分析专家。分析该方法在流程中的作用。
输出严格 JSON（不要 markdown）：
{
  "important": true/false,
  "important_types": ["validation|status_transition|external_call|mq_publish|mq_consume|db_write|branch|other_important"],
  "summary": "必须包含「类名#方法名」的中文描述，写清具体规则而非空泛「做校验」",
  "status_transitions": ["字段/状态从A到B", "..."],
  "external_calls": ["Feign/MQ/外部系统", "..."],
  "validation_points": ["具体校验项：以源码字段与条件为准", "..."],
  "business_tables": ["触达的表/实体简单类名（仅源码出现的）"],
  "next_calls": ["obj.method 或 Class.method，从代码真实调用中提取，最多15个"]
}
重要节点优先：参数校验、状态流转、写库、发/收MQ、调用外部系统、关键分支。透传 getter 标 important=false。
类名或方法名含 risk / asyn / async 且确有外呼/写库时通常 important=true。
validation_points / status_transitions 必须尽量从代码字面提取，不要只写「校验合法性」。
"""

FLOW_PROMPT = """你是面向业务同学的流程解读助手。根据入口与重要节点，输出严格 JSON：
{
  "title": "流程短标题",
  "narrative": "有序中文阶段1…N；必须点出具体校验（以节点为准）、状态从A到B、发MQ/调下游；回调/通知另起一节；用业务动词叙述，正文不要堆砌类#方法",
  "stages": [{"step":1,"title":"","class_method":"","desc":"含具体规则，禁止空泛「做校验」"}]
}
禁止编造未提供的节点。节点若含 validation_points / status_transitions 必须写进对应阶段。
class_method 字段必须填写真实类#方法；narrative/desc 面向业务，不要主动堆砌类名方法名。
类名/方法名含 risk、asyn、async 的重要节点必须单独成 stage（即使是条件调用），不可只写在父方法 desc 里。
"""

_ENTITY_NAME_RE = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:Entity|DO|PO))\b")
# Method/class names that typically gate secondary risk / async risk triggers.
_RISK_ASYNC_NAME_RE = re.compile(r"(?i)(risk|asyn|async)")


def _is_risk_async_node(class_method: str, method_name: str = "") -> bool:
    """True when the *method* name looks like a risk/async trigger (not every Risk* helper)."""
    method_part = (method_name or "").strip()
    if not method_part and "#" in (class_method or ""):
        method_part = (class_method or "").split("#", 1)[-1].strip()
    if method_part:
        return bool(_RISK_ASYNC_NAME_RE.search(method_part))
    return bool(_RISK_ASYNC_NAME_RE.search(class_method or ""))


def _boost_risk_async_importance(data: dict[str, Any], t: ParsedType, m: ParsedMethod) -> None:
    """Force-important for risk/async-named callees so they enter important_payloads → stages."""
    cm = str(data.get("class_method") or f"{t.simple_name}#{m.name}")
    if not _is_risk_async_node(cm, m.name):
        return
    data["important"] = True
    types = [x for x in (data.get("important_types") or []) if x in IMPORTANT_TYPES]
    if "external_call" not in types:
        types.append("external_call")
    if not types:
        types = ["other_important"]
    data["important_types"] = types[:6]
    data["class_method"] = cm


def _ensure_stages_include_risk_async(
    stages: list[Any],
    important_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """LLM stages often fold conditional risk into the parent; reinject as own steps."""
    out: list[dict[str, Any]] = [s for s in (stages or []) if isinstance(s, dict)]
    existing = {
        str(s.get("class_method") or "").strip().lower()
        for s in out
        if str(s.get("class_method") or "").strip()
    }
    payload_cms = [
        str(n.get("class_method") or "").strip()
        for n in important_payloads
        if str(n.get("class_method") or "").strip()
    ]
    for n in important_payloads:
        cm = str(n.get("class_method") or "").strip()
        if not cm or cm.lower() in existing:
            continue
        if not _is_risk_async_node(cm):
            continue
        new_st = {
            "step": 0,
            "title": cm.split("#")[-1] if "#" in cm else cm,
            "class_method": cm,
            "desc": str(n.get("summary") or f"{cm}：risk/async 触发"),
        }
        # Insert after the nearest preceding important node already present in stages.
        try:
            idx_payload = payload_cms.index(cm)
        except ValueError:
            idx_payload = -1
        predecessors = {c.lower() for c in payload_cms[:idx_payload] if c}
        insert_at = len(out)
        for i, st in enumerate(out):
            scm = str(st.get("class_method") or "").strip().lower()
            if scm in predecessors:
                insert_at = i + 1
        out.insert(insert_at, new_st)
        existing.add(cm.lower())
    for i, st in enumerate(out, 1):
        st["step"] = i
    return out


def _type_cache_key(simple_or_fqcn: str) -> str:
    return (simple_or_fqcn or "").split(".")[-1].lower()


def _looks_like_entity(simple: str, path: str = "", kind: str = "") -> bool:
    if kind == "Entity":
        return True
    return guess_role_from_name(simple or "", path or "") == "Entity"


def _entity_name_hints(t: ParsedType, m: ParsedMethod, data: dict[str, Any]) -> list[str]:
    hints: list[str] = []
    for x in data.get("business_tables") or []:
        if x:
            hints.append(str(x).split(".")[-1].strip())
    if _looks_like_entity(t.simple_name, t.file_path or "", t.symbol_kind or ""):
        hints.append(t.simple_name)
    for f in t.fields or []:
        tn = re.sub(r"<.*?>", "", f.type_name or "").split(".")[-1].strip()
        if tn and _looks_like_entity(tn):
            hints.append(tn)
    simple = t.simple_name or ""
    if simple.endswith("Mapper"):
        base = re.sub(r"Mapper$", "", simple)
        hints.extend([f"{base}Entity", f"{base}DO", f"{base}PO", base])
    for name in _ENTITY_NAME_RE.findall(m.raw_code or ""):
        hints.append(name)
    seen: set[str] = set()
    out: list[str] = []
    for h in hints:
        key = h.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(h)
    return out[:12]


async def _llm_json(system: str, user: str, *, max_tokens: int = 1200) -> dict[str, Any]:
    llm = get_parse_llm_provider()
    content = await llm.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    m = re.search(r"\{.*\}", content, re.DOTALL)
    data = json.loads(m.group(0) if m else content)
    if not isinstance(data, dict):
        raise ValueError("not object")
    return data


def _heuristic_node(t: ParsedType, m: ParsedMethod) -> dict[str, Any]:
    code = (m.raw_code or "").lower()
    types: list[str] = []
    if any(k in code for k in ("valid", "check", "assert", "校验")):
        types.append("validation")
    if any(k in code for k in ("status", "state", "状态")):
        types.append("status_transition")
    if any(k in code for k in ("feign", "resttemplate", "webclient")):
        types.append("external_call")
    if "convertandsend" in code or "kafka" in code:
        types.append("mq_publish")
    if "rabbitlistener" in code:
        types.append("mq_consume")
    if any(k in code for k in ("insert", "update", "save", "mapper")):
        types.append("db_write")
    cm = f"{t.simple_name}#{m.name}"
    important = bool(types) or t.symbol_kind in {"Controller", "Listener", "Job", "Feign"}
    if important and not types:
        types = ["other_important"]
    data = {
        "important": important,
        "important_types": types[:6],
        "summary": f"{cm}：{t.symbol_kind} 方法",
        "status_transitions": [],
        "external_calls": [],
        "validation_points": [],
        "business_tables": _entity_name_hints(t, m, {}),
        "next_calls": list(m.calls or [])[:15],
        "class_method": cm,
    }
    _boost_risk_async_importance(data, t, m)
    return data


@traceable(name="analyze_node_llm", run_type="llm")
async def _analyze_node(t: ParsedType, m: ParsedMethod) -> dict[str, Any]:
    cm = f"{t.simple_name}#{m.name}"
    try:
        data = await _llm_json(
            NODE_PROMPT,
            f"kind={t.symbol_kind}\nfqcn={t.fqcn}\nclass_method={cm}\n"
            f"known_calls={m.calls[:20]}\ncode:\n{truncate(m.raw_code or '', 2800)}",
        )
        data["class_method"] = cm
        types = [x for x in (data.get("important_types") or []) if x in IMPORTANT_TYPES]
        data["important_types"] = types
        summary = str(data.get("summary") or "")
        if cm not in summary:
            data["summary"] = f"{cm}：{summary}" if summary else f"{cm}：节点"
        if not data.get("next_calls"):
            data["next_calls"] = list(m.calls or [])[:15]
        if not data.get("business_tables"):
            data["business_tables"] = _entity_name_hints(t, m, data)
        _boost_risk_async_importance(data, t, m)
        return data
    except Exception:
        return _heuristic_node(t, m)


def _find_method(types: list[ParsedType], method_name: str) -> tuple[ParsedType, ParsedMethod] | None:
    for t in types:
        for m in t.methods:
            if m.name == method_name:
                return t, m
    return None


class _FlowSharedState:
    """Shared caches across concurrent entry workers."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.type_by_simple: dict[str, ParsedType] = {}
        self.entity_done: set[str] = set()
        # method symbol_id → AI node analysis (global dedupe across entries)
        self.node_cache: dict[str, dict[str, Any]] = {}
        self.node_inflight: dict[str, asyncio.Future] = {}
        self.milvus_rows: list[dict] = []
        self.entities_described = 0
        self.nodes_ai = 0
        self.nodes_cache_hit = 0
        self.total_important = 0
        self.flow_docs = 0
        self.missing_detail: list[str] = []
        self.done_entries = 0


async def _analyze_node_cached(
    state: _FlowSharedState,
    t: ParsedType,
    m: ParsedMethod,
    mid: str,
) -> dict[str, Any]:
    """AI-analyze a method once globally; concurrent waiters share the same Future."""
    async with state.lock:
        cached = state.node_cache.get(mid)
        if cached is not None:
            state.nodes_cache_hit += 1
            return dict(cached)
        inflight = state.node_inflight.get(mid)
        if inflight is not None:
            fut = inflight
            owner = False
        else:
            fut = asyncio.get_running_loop().create_future()
            state.node_inflight[mid] = fut
            owner = True

    if not owner:
        data = await fut
        async with state.lock:
            state.nodes_cache_hit += 1
        return dict(data)

    try:
        data = await _analyze_node(t, m)
        async with state.lock:
            state.node_cache[mid] = data
            state.nodes_ai += 1
            state.node_inflight.pop(mid, None)
            if not fut.done():
                fut.set_result(data)
        return dict(data)
    except Exception as e:
        async with state.lock:
            state.node_inflight.pop(mid, None)
            if not fut.done():
                fut.set_exception(e)
        raise


async def _ensure_type(
    state: _FlowSharedState,
    project_id: str,
    project_root: Path,
    index: list[dict[str, str]],
    simple: str,
) -> ParsedType | None:
    """Load class shell from disk (no full AI structural parse)."""
    key = _type_cache_key(simple)
    if not key:
        return None
    async with state.lock:
        if key in state.type_by_simple:
            return state.type_by_simple[key]
    hit = resolve_type_file(index, simple, prefer_impl=True)
    if not hit:
        return None
    t = await asyncio.to_thread(load_type_shell, project_root, hit["path"])
    async with state.lock:
        state.type_by_simple[_type_cache_key(t.simple_name)] = t
        # also index requested key if different (Service → ServiceImpl)
        state.type_by_simple[key] = t
    try:
        await asyncio.to_thread(write_parsed_type_graph, project_id, t)
    except Exception:
        logger.debug("neo4j shell write skipped for %s", t.fqcn, exc_info=True)
    return t


async def _ensure_method(
    state: _FlowSharedState,
    project_id: str,
    project_root: Path,
    index: list[dict[str, str]],
    simple: str,
    method_name: str,
) -> tuple[ParsedType, ParsedMethod] | None:
    """Resolve class file and extract method body by name (layer-by-layer read)."""
    if not method_name:
        return None
    t = await _ensure_type(state, project_id, project_root, index, simple)
    if not t:
        return None
    # already have method with body?
    for m in t.methods:
        if m.name == method_name and (m.raw_code or "").strip():
            return t, m
    # extract from source
    if not t.file_path:
        hit = resolve_type_file(index, t.simple_name, prefer_impl=True)
        if not hit:
            return None
        t.file_path = hit["path"]
    loaded = await asyncio.to_thread(load_method, project_root, t.file_path, method_name)
    if not loaded:
        # try Impl explicitly
        if not str(simple).endswith("Impl"):
            return await _ensure_method(
                state, project_id, project_root, index, f"{simple}Impl", method_name
            )
        return None
    t2, method = loaded
    async with state.lock:
        cached = state.type_by_simple.get(_type_cache_key(t2.simple_name))
        if cached:
            merge_method(cached, method)
            t2 = cached
            # refresh method ref from cache
            for m in t2.methods:
                if m.name == method_name and (m.raw_code or "").strip():
                    method = m
                    break
        state.type_by_simple[_type_cache_key(t2.simple_name)] = t2
        state.type_by_simple[_type_cache_key(simple)] = t2
    try:
        await asyncio.to_thread(_persist_one, project_id, t2, method)
    except Exception:
        logger.debug("persist method skipped %s#%s", t2.simple_name, method_name, exc_info=True)
    return t2, method


def _persist_one(project_id: str, t: ParsedType, method: ParsedMethod) -> None:
    slim = ParsedType(
        kind=t.kind,
        package=t.package,
        simple_name=t.simple_name,
        fqcn=t.fqcn,
        file_path=t.file_path,
        module_name=t.module_name,
        start_line=t.start_line,
        end_line=t.end_line,
        fields=list(t.fields),
        methods=[method],
        annotations=list(t.annotations),
        symbol_kind=t.symbol_kind,
    )
    with get_session() as session:
        persist_types(session, project_id, [slim])
    write_parsed_type_graph(project_id, slim)


async def _maybe_describe_entity(
    state: _FlowSharedState,
    project_id: str,
    project_root: Path,
    t: ParsedType,
) -> None:
    if not _looks_like_entity(t.simple_name, t.file_path or "", t.symbol_kind or ""):
        return
    async with state.lock:
        if t.fqcn in state.entity_done:
            return
        state.entity_done.add(t.fqcn)

    t.symbol_kind = "Entity"
    with get_session() as session:
        cid = symbol_id_for(project_id, t.fqcn, None)
        row = session.get(SymbolORM, cid)
        if row:
            row.kind = "Entity"
    await asyncio.to_thread(write_parsed_type_graph, project_id, t)
    desc = await describe_entity(project_id, t, project_root=project_root)
    if desc:
        async with state.lock:
            state.milvus_rows.append(desc)
            state.entities_described += 1
        logger.info("inline entity described %s", t.fqcn)


async def _touch_business_tables(
    state: _FlowSharedState,
    project_id: str,
    project_root: Path,
    index: list[dict[str, str]],
    hints: list[str],
) -> None:
    for hint in hints:
        et = await _ensure_type(state, project_id, project_root, index, hint)
        if et:
            await _maybe_describe_entity(state, project_id, project_root, et)


async def _analyze_one_entry(
    state: _FlowSharedState,
    project_id: str,
    project_root: Path,
    index: list[dict[str, str]],
    entry: dict[str, Any],
    *,
    max_nodes: int,
) -> None:
    flow_key = entry.get("flow_key") or "business_flow"
    flow_role = entry.get("flow_role") or "main"
    pilot_id = entry.get("pilot_id") or entry.get("method_name") or "entry"

    method_name = entry.get("method_name") or ""
    found = await _ensure_method(
        state,
        project_id,
        project_root,
        index,
        entry.get("simple_name") or "",
        method_name,
    )
    # Fallback: entry may only have file_path (AI structural parse of annotated methods)
    if not found and entry.get("file_path"):
        types = await asyncio.to_thread(
            parse_and_persist, project_id, project_root, entry["file_path"]
        )
        async with state.lock:
            for t in types:
                state.type_by_simple[_type_cache_key(t.simple_name)] = t
        found = _find_method(types, method_name)
        if not found and types and types[0].methods:
            # last resort: extract from the entry file by name
            loaded = await asyncio.to_thread(
                load_method, project_root, entry["file_path"], method_name
            )
            if loaded:
                found = loaded
                async with state.lock:
                    state.type_by_simple[_type_cache_key(loaded[0].simple_name)] = loaded[0]
            elif types[0].methods:
                found = (types[0], types[0].methods[0])
        for t in types:
            await asyncio.to_thread(write_parsed_type_graph, project_id, t)
    if not found:
        async with state.lock:
            state.missing_detail.append(f"{pilot_id}: method not found")
        return

    start_t, start_m = found
    queue: list[tuple[ParsedType, ParsedMethod]] = [(start_t, start_m)]
    visited: set[str] = set()
    important_payloads: list[dict[str, Any]] = []
    local_milvus: list[dict] = []

    while queue and len(visited) < max_nodes:
        t, m = queue.pop(0)
        mid = symbol_id_for(project_id, t.fqcn, m.signature)
        if mid in visited:
            continue
        visited.add(mid)

        # Encountered entity class itself while walking
        await _maybe_describe_entity(state, project_id, project_root, t)

        data = await _analyze_node_cached(state, t, m, mid)
        is_entry = mid == symbol_id_for(project_id, start_t.fqcn, start_m.signature)
        if is_entry:
            data["important"] = True
            if not data.get("important_types"):
                data["important_types"] = ["other_important"]

        await asyncio.to_thread(
            upsert_method_node,
            project_id,
            t,
            m,
            important=bool(data.get("important")),
            important_types=list(data.get("important_types") or []),
            flow_keys=[flow_key],
            is_entry=is_entry,
        )

        # Side-path: describe business tables touched by this node
        await _touch_business_tables(
            state,
            project_id,
            project_root,
            index,
            _entity_name_hints(t, m, data),
        )

        if data.get("important"):
            row = upsert_description(
                project_id,
                mid,
                summary=str(data.get("summary") or ""),
                business_domain=flow_key,
                tags=["重要节点", flow_key, flow_role] + list(data.get("important_types") or []),
                extra={
                    "important": True,
                    "important_types": data.get("important_types") or [],
                    "flow_keys": [flow_key],
                    "flow_roles": [flow_role],
                    "pilot_entry_id": pilot_id,
                    "class_method": data.get("class_method"),
                    "fqcn": t.fqcn,
                    "module_name": t.module_name or "",
                    "method_name": m.name,
                    "method_sig": m.signature,
                    "business_tables": data.get("business_tables") or [],
                    "status_transitions": data.get("status_transitions") or [],
                    "external_calls": data.get("external_calls") or [],
                    "validation_points": data.get("validation_points") or [],
                },
            )
            local_milvus.append(row)
            important_payloads.append(
                {
                    "class_method": data.get("class_method"),
                    "summary": data.get("summary"),
                    "important_types": data.get("important_types") or [],
                    "validation_points": data.get("validation_points") or [],
                    "status_transitions": data.get("status_transitions") or [],
                    "external_calls": data.get("external_calls") or [],
                    "business_tables": data.get("business_tables") or [],
                }
            )

        seen_calls: set[str] = set()
        next_calls: list[str] = []
        # Prefer heuristic source order for coverage, then AI hints (AI often truncates late calls like risk/quota)
        for c in list(m.calls or []) + list(data.get("next_calls") or []):
            key = str(c).split("(")[0].strip()
            if key and key not in seen_calls:
                seen_calls.add(key)
                next_calls.append(str(c))
        # Resolve + drop noise first, then take budget (avoid util crowding out business)
        resolved: list[tuple[str, str]] = []
        seen_target: set[tuple[str, str]] = set()
        for call in next_calls:
            targets = resolve_call_targets(caller=t, call=str(call), index=index)
            for simple, callee_method in targets:
                if is_noise_callee(simple, callee_method):
                    continue
                tk = (simple, callee_method)
                if tk in seen_target:
                    continue
                seen_target.add(tk)
                resolved.append(tk)
            if len(resolved) >= 28:
                break
        for simple, callee_method in resolved:
            pair = await _ensure_method(
                state, project_id, project_root, index, simple, callee_method
            )
            if not pair:
                continue
            ct, cmth = pair
            await _maybe_describe_entity(state, project_id, project_root, ct)
            callee_id = symbol_id_for(project_id, ct.fqcn, cmth.signature)
            await asyncio.to_thread(merge_calls, project_id, mid, callee_id)
            if callee_id not in visited:
                queue.append((ct, cmth))

    try:
        narrative = await _llm_json(
            FLOW_PROMPT,
            f"flow_key={flow_key}\nflow_role={flow_role}\n"
            f"entry={start_t.fqcn}#{start_m.name}\n"
            f"important_nodes={json.dumps(important_payloads[:40], ensure_ascii=False)}",
            max_tokens=2000,
        )
    except Exception:
        lines = [f"入口 {start_t.simple_name}#{start_m.name}（{flow_role}）"]
        stages = []
        for i, n in enumerate(important_payloads[:20], 1):
            lines.append(f"{i}. {n.get('summary')}")
            stages.append(
                {
                    "step": i,
                    "title": n.get("class_method") or "",
                    "class_method": n.get("class_method") or "",
                    "desc": n.get("summary") or "",
                }
            )
        narrative = {
            "title": f"{flow_key}:{flow_role}",
            "narrative": "\n".join(lines),
            "stages": stages,
        }

    stages = _ensure_stages_include_risk_async(
        list(narrative.get("stages") or []),
        important_payloads,
    )
    narrative["stages"] = stages

    narrative_text = str(narrative.get("narrative") or "")
    title = str(narrative.get("title") or f"{flow_key}:{pilot_id}")
    entry_sid = symbol_id_for(project_id, start_t.fqcn, start_m.signature)
    entry_cm = f"{start_t.simple_name}#{start_m.name}"
    # Keep AI node summary if present; do not clobber with a stub label.
    rich_entry = next(
        (
            str(n.get("summary") or "")
            for n in important_payloads
            if (n.get("class_method") or "") == entry_cm and n.get("summary")
        ),
        "",
    )
    entry_summary = rich_entry or f"流程入口 {entry_cm}"
    http_paths = entry.get("http_paths") or []
    if not http_paths and entry.get("http_path"):
        http_paths = [entry.get("http_path")]
    entry_row = upsert_description(
        project_id,
        entry_sid,
        summary=entry_summary,
        business_domain=flow_key,
        tags=["流程入口", "重要节点", flow_key, flow_role, entry_cm],
        extra={
            "entry": True,
            "important": True,
            "class_method": entry_cm,
            "fqcn": start_t.fqcn,
            "module_name": start_t.module_name or "",
            "method_name": start_m.name,
            "http_paths": http_paths,
            "pilot_entry_id": pilot_id,
            "flow_keys": [flow_key],
            "flow_roles": [flow_role],
            "flow_doc": {
                "flow_key": flow_key,
                "flow_role": flow_role,
                "pilot_entry_id": pilot_id,
                "title": title,
                "narrative": narrative_text,
                "stages": stages,
            },
        },
    )
    # Refresh entry vector so flow questions can hit the method itself.
    entry_row["text"] = truncate(
        f"{entry_row.get('text', '')}\n流程叙事: {narrative_text}",
        3500,
    )
    local_milvus.append(entry_row)
    doc_id = stable_id("flow", project_id, flow_key, pilot_id, entry_sid)
    path_blob = " ".join(str(p) for p in http_paths if p)
    entry_module = start_t.module_name or ""
    local_milvus.append(
        {
            "id": doc_id,
            "project_id": project_id,
            "symbol_id": f"flow:{flow_key}:{pilot_id}"[:64],
            "fqcn": f"__flow__:{flow_key}:{flow_role}",
            "module_name": entry_module,
            "tags": f"业务流程,{flow_key},{flow_role},flow,{entry_cm},{entry_module}",
            "text": truncate(
                f"业务流程 {title} flow_key={flow_key} role={flow_role} "
                f"module={entry_module} 入口 {start_t.fqcn}#{start_m.name} path={path_blob}\n{narrative_text}",
                3500,
            ),
        }
    )
    async with state.lock:
        state.milvus_rows.extend(local_milvus)
        state.total_important += len(important_payloads)
        state.flow_docs += 1
    logger.info(
        "flow %s done visited=%s important=%s",
        pilot_id,
        len(visited),
        len(important_payloads),
    )


@traceable(name="recursive_flow_from_entries", run_type="chain")
async def recursive_flow_from_entries(
    project_id: str,
    project_root: Path,
    entries: list[dict[str, Any]],
    *,
    index: list[dict[str, str]],
    progress: Callable[[int, int, str], None] | None = None,
    max_nodes_per_entry: int | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Concurrent BFS from entry methods; describe business tables when touched."""
    settings = get_settings()
    max_nodes = max_nodes_per_entry or max(20, settings.graph_max_depth * 8)
    workers = max(1, int(concurrency or settings.parse_concurrency or 4))
    workers = max(1, min(workers, 16))
    sem = asyncio.Semaphore(workers)
    state = _FlowSharedState()
    total = max(len(entries), 1)

    logger.info(
        "recursive_flow entries=%s concurrency=%s (PARSE_CONCURRENCY) max_nodes=%s",
        len(entries),
        workers,
        max_nodes,
    )

    async def _worker(entry: dict[str, Any]) -> None:
        async with sem:
            try:
                await _analyze_one_entry(
                    state,
                    project_id,
                    project_root,
                    index,
                    entry,
                    max_nodes=max_nodes,
                )
            except Exception:
                pilot = entry.get("pilot_id") or entry.get("method_name") or "?"
                logger.exception("flow entry failed %s", pilot)
                async with state.lock:
                    state.missing_detail.append(f"{pilot}: exception")
            finally:
                async with state.lock:
                    state.done_entries += 1
                    done = state.done_entries
                if progress:
                    cm = f"{entry.get('simple_name')}#{entry.get('method_name')}"
                    progress(done, total, f"flow {done}/{total} {cm}")

    await asyncio.gather(*[_worker(e) for e in entries])

    upsert_milvus_rows(state.milvus_rows)
    if progress:
        progress(
            total,
            total,
            f"flows done docs={state.flow_docs} important={state.total_important} "
            f"entities={state.entities_described} ai={state.nodes_ai} cache_hit={state.nodes_cache_hit}",
        )
    return {
        "entries_analyzed": len(entries),
        "important_nodes": state.total_important,
        "flow_docs": state.flow_docs,
        "entities_described": state.entities_described,
        "nodes_ai": state.nodes_ai,
        "nodes_cache_hit": state.nodes_cache_hit,
        "concurrency": workers,
        "missing_detail": state.missing_detail,
    }

