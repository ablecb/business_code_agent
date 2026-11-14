"""Tree-sitter BFS deep flow for withdrawals-style APIs.

Follows ``tests/demo_treesitter_withdrawals.py`` traversal policy:
  - BFS until local ``.java`` exhausted; JDK/deps are leaves
  - skip fluent/builder & noise receivers
  - cycle-safe via seen class_method set

Parse order (hard constraint):
  1) Collect related Model/DTO/Entity/VO from the call graph
  2) Describe each entity type **once** (cache) — entities first
  3) Dual-perspective deep-analyze business methods (what it does,
     related tables, field value changes after write, data flow)
  4) Persist MySQL symbol_desc + Milvus rows for every analyzed method
  5) Wire Neo4j CALLS from BFS; FOLLOWS synced after via flow_neo4j_sync
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.common import (  # noqa: E402
    list_java_index,
    read_file_snippet,
    upsert_description,
    upsert_milvus_rows,
)
from app.analysis.discover import describe_entity  # noqa: E402
from app.analysis.domain_catalog import (  # noqa: E402
    domain_instruction,
    list_known_domains,
    resolve_business_domain,
)
from app.analysis.neo4j_store import merge_calls, upsert_method_node, write_parsed_type_graph  # noqa: E402
from app.analysis.source_reader import load_method, load_type_shell, resolve_type_file  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.core.utils import stable_id, truncate  # noqa: E402
from app.db.mysql import get_session  # noqa: E402
from app.llm.factory import get_parse_llm_provider  # noqa: E402
from app.llm.tracing import traceable  # noqa: E402
from app.models.orm import SymbolORM  # noqa: E402
from app.parser.models import ParsedMethod, ParsedType  # noqa: E402
from app.parser.symbol_store import persist_types, symbol_id_for  # noqa: E402

logger = logging.getLogger("app.analysis.treesitter_deep")

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

_SKIP_TYPE_NAMES = frozenset(
    {
        "String",
        "Integer",
        "Long",
        "Boolean",
        "BigDecimal",
        "Object",
        "Map",
        "List",
        "Set",
        "Collection",
        "Optional",
        "Date",
        "LocalDate",
        "LocalDateTime",
        "Void",
        "void",
        "Class",
        "T",
        "R",
        "Response",
        "Request",
        "HttpResponse",
        "Executable",
        "Pattern",
        "Matcher",
        "Exception",
        "Throwable",
        "JSONObject",
        "JSONArray",
    }
)

_ENTITY_DTO_SUFFIX = (
    "Entity",
    "DO",
    "PO",
    "DTO",
    "Dto",
    "VO",
    "Vo",
    "Req",
    "Resp",
    "Request",
    "Response",
    "Body",
    "Param",
    "Model",
    "Bean",
)

DTO_DESC_PROMPT = """你是资深 Java 建模专家。根据 Model/DTO/VO 源码输出严格 JSON：
{{
  "tags":["DTO","入参|出参|报文"],
  "summary":"一句话：该对象在流程中的角色与关键字段含义（仅依据源码）",
  "business_domain":"短中文分类名",
  "domain_action":"reuse|create",
  "table_name":"",
  "status_fields":["状态类字段名"],
  "status_values":{{}},
  "key_fields":["核心字段"]
}}
不要编造源码中不存在的字段。
{domain_rule}
"""

DEEP_NODE_PROMPT = """你是资深 Java/Spring「业务+研发」双视角代码分析专家。必须严格按源码做深度解析，禁止空泛套话。

输出严格 JSON（不要 markdown）：
{{
  "important": true/false,
  "important_types": ["validation|status_transition|external_call|mq_publish|mq_consume|db_write|branch|other_important"],
  "class_method": "类名#方法名",
  "business_domain": "短中文分类名",
  "domain_action": "reuse|create",

  "summary_business_clear": "【业务·精简】2～4句，非技术同学也能懂：做什么、关键规则、成功/失败意味着什么",
  "summary_business_detail": "【业务·详细】分点写清：能力、规则、边界、影响；点出关键字段/状态含义（勿堆砌类名）",
  "summary_dev_clear": "【研发·精简】3～6句：入参从哪来→关键校验/分支→调谁→写什么→返回什么（必须出现真实变量名/方法名）",
  "summary_dev_detail": "【研发·详细】完整数据流：每个关键对象如何创建、字段如何赋值/改写、分支差异、外呼报文取值、返回如何组装",

  "summary": "检索用短摘要：业务精简一句 + 研发精简一句，必须含「类名#方法名」",

  "business_view": {{
    "capability": "业务能力",
    "rules": ["具体规则，勿写「做校验」这类空话"],
    "boundaries": ["前置条件/不适用场景/幂等与开关边界"],
    "impacts": ["对上游/下游/持久化数据的影响（以源码为准）"]
  }},

  "impl_view": {{
    "params": [
      {{
        "name": "变量名",
        "type": "类型",
        "source": "从哪来（如 request.getBody() / header.getXxx()）",
        "validation": "校验逻辑或「无」",
        "transforms": ["赋值或改写步骤"],
        "passed_to": ["传给哪些调用"]
      }}
    ],
    "param_flow": [
      {{
        "step": 1,
        "from": "来源对象.字段",
        "change": "如何变化/拷贝/put",
        "to": "目标对象.字段或下游方法参数"
      }}
    ],
    "branches": [
      {{"condition": "源码条件", "path": "走哪条路径", "diff": "返回/写库/外呼差异", "early_return": true}}
    ],
    "memory_objects": [
      {{"object": "对象名", "type": "类型", "reads": ["读哪些属性"], "writes": ["写哪些属性"], "state_changes": ["状态如何变"]}}
    ],
    "db_ops": [
      {{
        "op": "select|insert|update|delete",
        "table_or_entity": "实体/表",
        "field_sources": {{"字段": "赋值来源"}},
        "persist_changes": "持久化变更说明",
        "via_call": "哪个方法触发"
      }}
    ],
    "external_calls": [
      {{
        "target": "服务/Feign/MQ/锁",
        "request_assembly": "请求如何从本地对象组装",
        "response_parse": "响应如何解析",
        "local_effect": "写回哪些本地变量/字段"
      }}
    ],
    "return_value": {{
      "assembly": "返回值如何组装（列出 key 与取值来源）",
      "composition": "最终返回字段构成",
      "success_vs_fail": "成功与失败返回差异"
    }}
  }},

  "data_flow_narrative": "必须写成连贯链条，显式写出变量名：入参 → 中间对象 → 下游调用 → 写库/返回。禁止只写「做校验/调服务」。",

  "validation_points": ["具体校验项（字段级）"],
  "status_transitions": ["实体.字段：原值 → 新值（含触发条件）"],
  "external_calls": ["简述外呼"],
  "business_tables": ["本方法触达的实体/表简单类名；无写库则空数组"],
  "table_field_changes": [
    {{
      "table_or_entity": "实体或表名",
      "field": "字段名",
      "from_value": "修改前（常量/枚举/入参/未知）",
      "to_value": "修改后的值或表达式",
      "op": "insert|update|delete|assign",
      "when": "触发条件或分支"
    }}
  ],
  "next_calls": ["obj.method，最多15个"]
}}

硬性规则：
1) 只依据给定源码，禁止编造字段/调用；
2) params / param_flow / branches / return_value.assembly 必须具体到变量名与赋值；
3) 四类摘要（business/dev × clear/detail）都必填且互不重复；summary_business_clear 必须说清「这个方法做了啥」；
4) 有 insert/update/delete/setStatus 等写库时：business_tables 必填，table_field_changes 必须写出字段修改后的值；无写库则两者为空数组；
5) getter/透传 important=false；名称或调用含 risk/asyn/async 且确有外呼/写库时 important=true。
{domain_rule}
"""

# Overlong methods: segment → per-chunk deep parse → merge.
DEEP_CHUNK_CHAR_LIMIT = 4000
DEEP_CHUNK_SOFT_MAX = 5500  # wait for a clean boundary up to this size
DEEP_CHUNK_TRIGGER = 5500  # raw_code longer than this → segmented analysis
DEEP_CHUNK_MAX = 10
LLM_LOG_REQ_CHARS = 2500
LLM_LOG_RESP_CHARS = 2500

DEEP_CHUNK_PROMPT = """你是资深 Java/Spring「业务+研发」双视角代码分析专家。
这是**超长方法的第 {chunk_idx}/{chunk_total} 段**源码（已在完整语句/代码块边界切开），只分析本段，禁止编造本段未出现的内容。
输出严格 JSON：
{{
  "chunk_index": {chunk_idx},
  "summary_business_clear": "本段业务精简（1～3句）",
  "summary_business_detail": "本段业务详细",
  "summary_dev_clear": "本段研发精简（必须含真实变量名）",
  "summary_dev_detail": "本段研发详细数据流",
  "business_view": {{"capability":"","rules":[],"boundaries":[],"impacts":[]}},
  "impl_view": {{
    "params": [{{"name":"","type":"","source":"","validation":"","transforms":[],"passed_to":[]}}],
    "param_flow": [{{"step":1,"from":"","change":"","to":""}}],
    "branches": [{{"condition":"","path":"","diff":"","early_return":false}}],
    "memory_objects": [{{"object":"","type":"","reads":[],"writes":[],"state_changes":[]}}],
    "db_ops": [{{"op":"","table_or_entity":"","field_sources":{{}},"persist_changes":"","via_call":""}}],
    "external_calls": [{{"target":"","request_assembly":"","response_parse":"","local_effect":""}}],
    "return_value": {{"assembly":"","composition":"","success_vs_fail":""}}
  }},
  "data_flow_narrative": "本段：变量从哪来→如何改→写哪里/调谁",
  "validation_points": [],
  "status_transitions": [],
  "external_calls": [],
  "business_tables": [],
  "table_field_changes": [{{"table_or_entity":"","field":"","from_value":"","to_value":"","op":"","when":""}}],
  "next_calls": []
}}
列表每项最多 10 条；有写库时必须填 business_tables + table_field_changes（含修改后的值）；字符串避免未转义换行。
"""

DEEP_MERGE_PROMPT = """你是资深 Java/Spring 双视角汇总专家。下面是同一方法分段深析结果，请汇总为**完整方法**级 JSON（不要 markdown）。
输出字段与完整深析一致：
important, important_types, class_method,
summary_business_clear, summary_business_detail, summary_dev_clear, summary_dev_detail, summary,
business_view, impl_view(含 params/param_flow/branches/memory_objects/db_ops/external_calls/return_value),
data_flow_narrative, validation_points, status_transitions, external_calls, business_tables, table_field_changes, next_calls

硬性要求（正确且完整）：
1) 四类摘要覆盖全方法，按分段执行顺序连贯，禁止空泛、禁止丢掉任一分段要点；
2) param_flow / branches / memory_objects / db_ops / external_calls 必须合并全部分段并去重，按出现顺序；
3) data_flow_narrative 写成一条完整链条（显式变量名：从入参到返回）；
4) return_value 取最后有效分段，并写清成功/失败差异；
5) 只依据分段结果，禁止编造未出现的字段/调用；
6) 若分段之间有衔接（上段写入的对象下段继续用），必须在汇总里写清。
"""

FLOW_PROMPT = """你是面向业务+研发同学的流程解读助手。根据入口与「按调用执行序」排列的重要节点，输出严格 JSON：
{
  "title": "流程短标题",
  "narrative_business_clear": "业务精简版流程（非技术同学可读；不要堆砌类#方法）",
  "narrative_business_detail": "业务详细版：阶段规则/边界/影响；不要主动给出类名方法名",
  "narrative_dev_clear": "研发精简版：按执行序 关键入参→分支→外呼→写库→返回",
  "narrative_dev_detail": "研发详细版：点出类#方法与关键变量数据流（须与 stages 同序）",
  "narrative": "默认对外叙述=业务精简（勿堆砌类#方法）；研发细节放 narrative_dev_*",
  "stages": [{"step":1,"title":"","class_method":"","desc_business":"","desc_dev":"","desc":"综合"}]
}
硬性约束：
1) stages 必须严格按输入 important_nodes / execution_spine 的顺序（入口方法内的真实调用序：Controller→Service1→回到入口→Service2…），禁止按「业务故事」重排；
2) 禁止把同 Controller 上未在本链调用的其它 HTTP 接口（兄弟 endpoint）写进 stages；
3) 禁止编造未提供的 class_method；节点若含 param_flow / data_flow_narrative / summary_* 必须写入对应阶段；
4) class_method 保留技术锚点；业务叙述字段不要主动堆砌类名方法名。
"""


def _dto_system_prompt(known_domains: list[str]) -> str:
    return DTO_DESC_PROMPT.format(domain_rule=domain_instruction(known_domains))


def _deep_node_system_prompt(known_domains: list[str]) -> str:
    return DEEP_NODE_PROMPT.format(domain_rule=domain_instruction(known_domains))


def _type_cache_key(simple_or_fqcn: str) -> str:
    return (simple_or_fqcn or "").split(".")[-1].lower()


def _strip_generics(type_text: str) -> list[str]:
    raw = re.sub(r"[<>\[\],?]", " ", type_text or "")
    out: list[str] = []
    for part in raw.split():
        name = part.split(".")[-1].strip()
        if not name or name in _SKIP_TYPE_NAMES:
            continue
        if name[0].isupper():
            out.append(name)
    return out


def _looks_like_model_or_dto(simple: str, path: str = "") -> bool:
    if not simple or simple in _SKIP_TYPE_NAMES:
        return False
    n = simple
    p = (path or "").replace("\\", "/").lower()
    if any(seg in p for seg in ("/entity/", "/dto/", "/vo/", "/model/", "/request/", "/response/")):
        return True
    if n.endswith(("Entity", "DO", "PO", "DTO", "Dto", "VO", "Vo", "ReqVo", "RespVo", "Request", "Response", "Body")):
        return True
    if any(n.endswith(suf) for suf in _ENTITY_DTO_SUFFIX):
        return True
    # domain aggregates often bare names under entity package already covered;
    # also accept bare names without suffix when path hints entity
    if "/entity/" in p:
        return True
    return False


def _is_persist_entity(simple: str, path: str = "") -> bool:
    n = simple or ""
    p = (path or "").replace("\\", "/").lower()
    if "/entity/" in p or n.endswith(("Entity", "DO", "PO")):
        return True
    # bare table-mapped names commonly live under entity/
    if "/entity/" in p:
        return True
    return False


def _strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_balanced_object(text: str) -> str | None:
    """Return first top-level {...} with brace balance (string-aware)."""
    s = _strip_code_fences(text)
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    # Truncated: return from start so caller can repair/close.
    if depth > 0:
        return s[start:]
    return None


def _repair_json_text(raw: str) -> str:
    """Best-effort fix for truncated / trailing-comma JSON objects."""
    s = (raw or "").strip()
    if not s:
        return s
    # Drop trailing incomplete , "key or , "key":
    s = re.sub(r",\s*\"[^\"]*$", "", s)
    s = re.sub(r",\s*$", "", s)
    # Close an open string if truncated mid-value
    # Count unescaped quotes
    quote_n = 0
    esc = False
    for ch in s:
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            quote_n += 1
    if quote_n % 2 == 1:
        s += '"'
    # Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)
    # Balance braces / brackets (outside strings already roughly handled)
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in s:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
    while stack:
        s += stack.pop()
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s


def _loads_llm_json(content: str) -> dict[str, Any]:
    """Parse LLM JSON with repair for truncation / trailing commas."""
    raw = _strip_code_fences(content)
    attempts: list[str] = []
    extracted = _extract_balanced_object(raw)
    if extracted:
        attempts.append(extracted)
        attempts.append(_repair_json_text(extracted))
    attempts.append(raw)
    attempts.append(_repair_json_text(raw))
    # greedy fallback (legacy)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        attempts.append(m.group(0))
        attempts.append(_repair_json_text(m.group(0)))

    seen: set[str] = set()
    last_err: Exception | None = None
    for cand in attempts:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            data = json.loads(cand)
            if isinstance(data, dict):
                return data
            last_err = ValueError("JSON root must be object")
        except Exception as e:
            last_err = e
            continue
    raise ValueError(f"invalid LLM JSON: {last_err}") from last_err


def _log_llm_io(kind: str, *, system: str, user: str, response: str | None = None) -> None:
    """Log AI request / response (truncated) for deep-parse debugging."""
    if response is None:
        sys_prev = truncate(system.replace("\n", "\\n"), 400)
        user_prev = truncate(user, LLM_LOG_REQ_CHARS)
        logger.info("[ai-request] kind=%s system=%s", kind, sys_prev)
        logger.info("[ai-request] kind=%s user_chars=%s user=\n%s", kind, len(user or ""), user_prev)
        return
    logger.info(
        "[ai-response] kind=%s resp_chars=%s resp=\n%s",
        kind,
        len(response or ""),
        truncate(response or "", LLM_LOG_RESP_CHARS),
    )


async def _llm_json(
    system: str,
    user: str,
    *,
    max_tokens: int = 2200,
    kind: str = "json",
) -> dict[str, Any]:
    llm = get_parse_llm_provider()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    _log_llm_io(kind, system=system, user=user)
    content = await llm.chat(
        messages,
        temperature=0.1,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    _log_llm_io(kind, system=system, user=user, response=content)
    try:
        return _loads_llm_json(content)
    except Exception as first_err:
        logger.warning("llm json parse failed kind=%s (%s); retry compact JSON", kind, first_err)
        retry_user = (
            f"{user}\n\n"
            "上次输出 JSON 非法/被截断。请重新输出一个更短但仍完整的合法 JSON 对象："
            "优先填满四类摘要(clear/detail)+params+param_flow+branches+return_value；"
            "列表字段各最多 8 条；字符串避免未转义换行；不要 markdown。"
        )
        _log_llm_io(f"{kind}-retry", system=system, user=retry_user)
        content2 = await llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": retry_user},
            ],
            temperature=0.0,
            max_tokens=max(max_tokens, 4500),
            response_format={"type": "json_object"},
        )
        _log_llm_io(f"{kind}-retry", system=system, user=retry_user, response=content2)
        return _loads_llm_json(content2)


def _import_demo():
    from tests import demo_treesitter_withdrawals as demo

    return demo


def collect_model_dto_names(bfs_result: dict[str, Any]) -> list[str]:
    """Collect Model/DTO/Entity simple names from tree-sitter BFS graph (once each)."""
    names: list[str] = []
    seen: set[str] = set()

    def _add(type_text: str, path: str = "") -> None:
        for simple in _strip_generics(type_text):
            if not _looks_like_model_or_dto(simple, path):
                continue
            key = simple.lower()
            if key in seen:
                continue
            seen.add(key)
            names.append(simple)

    def walk(node: dict[str, Any]) -> None:
        path = str(node.get("file") or "")
        for p in node.get("parameters") or []:
            if isinstance(p, dict):
                _add(str(p.get("type") or ""), path)
        _add(str(node.get("return_type") or ""), path)
        for sig in (node.get("signature"),):
            if sig:
                # crude: pull CapWord tokens that look like DTOs
                for m in re.finditer(r"\b([A-Z][A-Za-z0-9]+)\b", str(sig)):
                    _add(m.group(1), path)
        for c in node.get("children") or []:
            if isinstance(c, dict):
                walk(c)

    tree = bfs_result.get("tree")
    if isinstance(tree, dict):
        walk(tree)
    # also by_depth
    by_depth = bfs_result.get("by_depth") or {}
    if isinstance(by_depth, dict):
        for bucket in by_depth.values():
            for n in bucket or []:
                if isinstance(n, dict):
                    path = str(n.get("file") or "")
                    for p in n.get("parameters") or []:
                        if isinstance(p, dict):
                            _add(str(p.get("type") or ""), path)
                    _add(str(n.get("return_type") or ""), path)
    return names


class _DeepState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.type_by_simple: dict[str, ParsedType] = {}
        self.entity_done: set[str] = set()  # fqcn lower → described once
        self.node_cache: dict[str, dict[str, Any]] = {}
        self.node_inflight: dict[str, asyncio.Future] = {}
        self.milvus_rows: list[dict] = []
        self.known_domains: list[str] = []
        self.entities_described = 0
        self.dtos_described = 0
        self.nodes_ai = 0
        self.nodes_started = 0
        self.nodes_finished = 0
        self.nodes_inflight = 0
        self.nodes_cache_hit = 0
        self.total_important = 0
        self.flow_docs = 0
        self.missing: list[str] = []

    def snapshot_domains(self) -> list[str]:
        return list(self.known_domains)

    def note_domain(
        self,
        project_id: str,
        proposed: str | None,
        *,
        domain_action: str | None = None,
    ) -> str:
        resolved = resolve_business_domain(
            project_id,
            proposed,
            domain_action=domain_action,
            known=self.known_domains,
        )
        if resolved and not any(x.casefold() == resolved.casefold() for x in self.known_domains):
            self.known_domains.append(resolved)
        return resolved


async def _ensure_type(
    state: _DeepState,
    project_id: str,
    project_root: Path,
    index: list[dict[str, str]],
    simple: str,
) -> ParsedType | None:
    key = _type_cache_key(simple)
    if not key:
        return None
    async with state.lock:
        if key in state.type_by_simple:
            return state.type_by_simple[key]
    hit = resolve_type_file(index, simple, prefer_impl=False)
    if not hit:
        return None
    t = await asyncio.to_thread(load_type_shell, project_root, hit["path"])
    async with state.lock:
        state.type_by_simple[_type_cache_key(t.simple_name)] = t
        state.type_by_simple[key] = t
    try:
        await asyncio.to_thread(write_parsed_type_graph, project_id, t)
    except Exception:
        logger.debug("neo4j shell skipped %s", t.fqcn, exc_info=True)
    return t


async def _describe_model_or_dto(
    state: _DeepState,
    project_id: str,
    project_root: Path,
    t: ParsedType,
) -> None:
    """Describe Entity or DTO/VO once; cache by fqcn."""
    fqcn = t.fqcn or t.simple_name
    async with state.lock:
        if fqcn in state.entity_done:
            return
        state.entity_done.add(fqcn)

    is_entity = _is_persist_entity(t.simple_name, t.file_path or "")
    # Persist type shell so SymbolORM exists before description upsert.
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
        methods=[],
        annotations=list(t.annotations),
        symbol_kind="Entity" if is_entity else "DTO",
    )

    def _persist_shell() -> None:
        with get_session() as session:
            persist_types(session, project_id, [slim])
            cid = symbol_id_for(project_id, t.fqcn, None)
            row = session.get(SymbolORM, cid)
            if row:
                row.kind = "Entity" if is_entity else "DTO"

    # Keep event loop free so other concurrent LLM tasks can progress.
    await asyncio.to_thread(_persist_shell)

    if is_entity:
        t.symbol_kind = "Entity"
        try:
            await asyncio.to_thread(write_parsed_type_graph, project_id, t)
        except Exception:
            pass
        desc = await describe_entity(project_id, t, project_root=project_root)
        if desc:
            async with state.lock:
                state.milvus_rows.append(desc)
                state.entities_described += 1
        return

    # DTO / VO / Req / Model
    t.symbol_kind = "DTO"
    snippet = ""
    if t.file_path:
        snippet = await asyncio.to_thread(
            read_file_snippet, project_root, t.file_path, limit=4500
        )
    field_names = [f.name for f in t.fields]
    async with state.lock:
        known = state.snapshot_domains()
    try:
        data = await _llm_json(
            _dto_system_prompt(known),
            f"fqcn={t.fqcn}\nsimple_name={t.simple_name}\nfields={field_names}\n"
            f"code:\n{truncate(snippet, 4000)}",
            max_tokens=900,
            kind=f"dto-desc-{t.simple_name}",
        )
    except Exception:
        data = {
            "tags": ["DTO", "报文"],
            "summary": f"传输对象 {t.simple_name}",
            "business_domain": "",
            "domain_action": "create",
            "key_fields": field_names[:12],
        }
    async with state.lock:
        domain = state.note_domain(
            project_id,
            str(data.get("business_domain") or ""),
            domain_action=str(data.get("domain_action") or ""),
        )
    sid = symbol_id_for(project_id, t.fqcn, None)
    desc = await asyncio.to_thread(
        upsert_description,
        project_id,
        sid,
        summary=str(data.get("summary") or f"DTO {t.fqcn}"),
        business_domain=domain,
        tags=[str(x) for x in (data.get("tags") or ["DTO"])] + ["预加载实体"],
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
    async with state.lock:
        state.milvus_rows.append(desc)
        state.dtos_described += 1
    logger.info("preloaded model/dto %s", t.fqcn)


async def preload_entities(
    state: _DeepState,
    project_id: str,
    project_root: Path,
    index: list[dict[str, str]],
    type_names: list[str],
    *,
    concurrency: int = 8,
    progress: Callable[[int, int, str], None] | None = None,
) -> list[str]:
    """Preload all Model/DTO/Entity types — each described at most once (concurrent LLM)."""
    loaded: list[str] = []
    total = max(len(type_names), 1)
    started_n = 0
    done_n = 0
    inflight = 0
    sem = asyncio.Semaphore(max(1, concurrency))

    async def _one(name: str) -> str | None:
        nonlocal started_n, done_n, inflight
        async with sem:
            async with state.lock:
                started_n += 1
                inflight += 1
                started = started_n
                cur_in = inflight
            if progress:
                progress(
                    1,
                    4,
                    f"start preload {started}/{total} inflight={cur_in} {name}",
                )
            try:
                t = await _ensure_type(state, project_id, project_root, index, name)
                if not t:
                    async with state.lock:
                        state.missing.append(f"entity_missing:{name}")
                        done_n += 1
                        cur = done_n
                        inflight -= 1
                    if progress:
                        progress(1, 4, f"done preload miss {cur}/{total} {name}")
                    return None
                await _describe_model_or_dto(state, project_id, project_root, t)
                async with state.lock:
                    done_n += 1
                    cur = done_n
                    inflight -= 1
                if progress:
                    progress(1, 4, f"done preload {cur}/{total} {name}")
                return t.fqcn
            except Exception:
                async with state.lock:
                    done_n += 1
                    cur = done_n
                    inflight -= 1
                if progress:
                    progress(1, 4, f"done preload error {cur}/{total} {name}")
                raise

    results = await asyncio.gather(*[_one(n) for n in type_names])
    loaded = [x for x in results if x]
    if progress:
        progress(1, 4, f"entities preloaded={len(loaded)}")
    return loaded


def _compose_dual_summaries(cm: str, data: dict[str, Any]) -> dict[str, str]:
    """Normalize four summary flavors + MySQL/Milvus primary summary."""
    biz_clear = str(data.get("summary_business_clear") or "").strip()
    biz_detail = str(data.get("summary_business_detail") or "").strip()
    dev_clear = str(data.get("summary_dev_clear") or "").strip()
    dev_detail = str(data.get("summary_dev_detail") or "").strip()
    bv = data.get("business_view") if isinstance(data.get("business_view"), dict) else {}
    iv = data.get("impl_view") if isinstance(data.get("impl_view"), dict) else {}
    flow = str(data.get("data_flow_narrative") or "").strip()

    if not biz_clear:
        biz_clear = str(bv.get("capability") or data.get("summary") or f"{cm} 业务节点").strip()
    if not biz_detail:
        bits = [biz_clear]
        if bv.get("rules"):
            bits.append("规则：" + "；".join(str(x) for x in bv.get("rules")[:6]))
        if bv.get("boundaries"):
            bits.append("边界：" + "；".join(str(x) for x in bv.get("boundaries")[:4]))
        if bv.get("impacts"):
            bits.append("影响：" + "；".join(str(x) for x in bv.get("impacts")[:4]))
        biz_detail = "\n".join(bits)
    if not dev_clear:
        dev_clear = flow[:400] if flow else str(data.get("summary") or f"{cm} 实现节点")
    if not dev_detail:
        parts = [dev_clear]
        if flow:
            parts.append(f"数据流：{flow}")
        params = iv.get("params") if isinstance(iv.get("params"), list) else []
        if params:
            parts.append(
                "入参："
                + "；".join(
                    f"{p.get('name')}←{p.get('source')}" for p in params[:8] if isinstance(p, dict)
                )
            )
        pf = iv.get("param_flow") if isinstance(iv.get("param_flow"), list) else []
        if pf:
            parts.append(
                "参数变化："
                + " → ".join(
                    f"{p.get('from')}[{p.get('change')}]→{p.get('to')}"
                    for p in pf[:10]
                    if isinstance(p, dict)
                )
            )
        ret = iv.get("return_value") if isinstance(iv.get("return_value"), dict) else {}
        if ret.get("assembly"):
            parts.append(f"返回组装：{ret.get('assembly')}")
        dev_detail = "\n".join(parts)

    # Ensure class_method anchors
    for label, val in (
        ("summary_business_clear", biz_clear),
        ("summary_business_detail", biz_detail),
        ("summary_dev_clear", dev_clear),
        ("summary_dev_detail", dev_detail),
    ):
        if cm not in val:
            if label.endswith("clear"):
                # keep clear short — prefix once
                pass
    primary = str(data.get("summary") or "").strip()
    if not primary or len(primary) < 40 or "业务能力:" in primary and "reqMap" not in primary:
        primary = (
            f"{cm}\n"
            f"【业务精简】{biz_clear}\n"
            f"【研发精简】{dev_clear}"
        )
    elif cm not in primary:
        primary = f"{cm}：{primary}"

    return {
        "summary_business_clear": biz_clear,
        "summary_business_detail": biz_detail,
        "summary_dev_clear": dev_clear,
        "summary_dev_detail": dev_detail,
        "summary": truncate(primary, 2500),
        # MySQL summary stores both clear versions for quick read
        "summary_mysql": truncate(
            f"{cm}\n【业务·精简】{biz_clear}\n【研发·精简】{dev_clear}\n"
            f"【业务·详细】{biz_detail}\n【研发·详细】{dev_detail}",
            8000,
        ),
    }


def _heuristic_deep(t: ParsedType, m: ParsedMethod) -> dict[str, Any]:
    cm = f"{t.simple_name}#{m.name}"
    code = (m.raw_code or "").lower()
    types: list[str] = []
    if any(k in code for k in ("valid", "check", "assert", "校验")):
        types.append("validation")
    if any(k in code for k in ("status", "state", "状态")):
        types.append("status_transition")
    if any(k in code for k in ("feign", "resttemplate", "convertandsend", "kafka")):
        types.append("external_call")
    if any(k in code for k in ("insert", "update", "save", "mapper")):
        types.append("db_write")
    important = bool(types) or t.symbol_kind in {"Controller", "Listener", "Job", "Feign"}

    # Javadoc / line comment above the method → professional business summary.
    from app.analysis.comment_meta import (
        business_summary_from_comment,
        clean_comment,
        param_flow_from_javadoc,
    )

    raw_comment = getattr(m, "doc_comment", "") or ""
    biz_text = business_summary_from_comment(raw_comment, fallback=f"{cm} 业务节点")
    param_flow_doc = param_flow_from_javadoc(raw_comment)
    comment_clean = clean_comment(raw_comment)

    data = {
        "important": important or bool(types),
        "important_types": types[:6] or (["other_important"] if important else []),
        "class_method": cm,
        "summary": f"{cm}：{biz_text}",
        "summary_business_clear": biz_text,
        "summary_business_detail": (
            comment_clean[:300] if comment_clean else f"{cm} 业务节点（启发式）"
        ),
        "summary_dev_clear": f"{cm} 实现节点，见源码调用",
        "summary_dev_detail": f"{cm} 实现细节需重跑 LLM 深析",
        "business_view": {
            "capability": biz_text,
            "rules": [],
            "boundaries": [],
            "impacts": [],
        },
        "impl_view": {
            "params": param_flow_doc,
            "param_flow": [],
            "branches": [],
            "memory_objects": [],
            "db_ops": [],
            "external_calls": [],
            "return_value": {"assembly": "", "composition": "", "success_vs_fail": ""},
        },
        "data_flow_narrative": f"{cm}：见源码调用链",
        "validation_points": [],
        "status_transitions": [],
        "external_calls": [],
        "business_tables": [],
        "table_field_changes": [],
        "next_calls": list(m.calls or [])[:15],
    }
    data.update(_compose_dual_summaries(cm, data))
    return data


def _line_brace_delta(line: str) -> int:
    """Net `{`−`}` outside strings/chars for one Java source line (best-effort)."""
    depth = 0
    in_str = False
    in_char = False
    esc = False
    i = 0
    code = line
    while i < len(code):
        ch = code[i]
        if in_str or in_char:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif (in_str and ch == '"') or (in_char and ch == "'"):
                in_str = False
                in_char = False
            i += 1
            continue
        if ch == "/" and i + 1 < len(code) and code[i + 1] == "/":
            break
        if ch == '"':
            in_str = True
        elif ch == "'":
            in_char = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    return depth


def _is_clean_split_boundary(line: str, depth_after: int, *, base_depth: int) -> bool:
    """True when ending a chunk after this line keeps statements/blocks intact."""
    stripped = line.strip()
    if not stripped:
        return depth_after <= base_depth + 1
    if depth_after > base_depth + 1:
        return False
    if stripped == "}" or stripped.endswith("}") or stripped.endswith(";"):
        return depth_after <= base_depth + 1
    return False


def _split_method_code_chunks(
    code: str,
    *,
    chunk_size: int = DEEP_CHUNK_CHAR_LIMIT,
    soft_max: int = DEEP_CHUNK_SOFT_MAX,
    max_chunks: int = DEEP_CHUNK_MAX,
) -> list[str]:
    """Split on complete statement/block boundaries — avoid mid-statement hard cuts.

    Strategy:
      1) Track brace depth (string-aware)
      2) After reaching ``chunk_size``, cut only at a clean boundary
      3) Wait until ``soft_max`` for a clean ``}`` / ``;`` / blank at low depth
      4) Never split mid-line; character hard-cut only for rare overlong lines
    """
    text = code or ""
    if len(text) <= chunk_size:
        return [text]
    lines = text.splitlines(keepends=True)
    if not lines:
        return [text]

    base_depth = 1  # typical method-body depth after opening '{'
    depth = 0
    chunks: list[str] = []
    buf: list[str] = []
    cleans: list[bool] = []
    buf_len = 0

    def _flush_upto(end_inclusive: int) -> None:
        nonlocal buf, cleans, buf_len
        if end_inclusive < 0:
            return
        piece = "".join(buf[: end_inclusive + 1])
        if piece.strip():
            chunks.append(piece)
        buf = buf[end_inclusive + 1 :]
        cleans = cleans[end_inclusive + 1 :]
        buf_len = sum(len(x) for x in buf)

    def _last_clean_idx() -> int:
        for j in range(len(cleans) - 1, -1, -1):
            if cleans[j]:
                return j
        return -1

    def _pack_rest_and_return(rest: str) -> list[str]:
        if rest.strip():
            chunks.append(rest)
        out = [c for c in chunks if c.strip()]
        logger.info("code split chunks=%s sizes=%s", len(out), [len(c) for c in out])
        return out or [text]

    i = 0
    while i < len(lines):
        line = lines[i]
        # Extremely long single line: last-resort hard split (still logged)
        if len(line) > soft_max:
            if buf:
                cut = _last_clean_idx()
                if cut >= 0:
                    _flush_upto(cut)
                if buf:
                    chunks.append("".join(buf))
                    buf, cleans, buf_len = [], [], 0
            logger.warning("hard-split overlong line chars=%s (last resort)", len(line))
            for j in range(0, len(line), chunk_size):
                part = line[j : j + chunk_size]
                if part.strip():
                    chunks.append(part)
                if len(chunks) >= max_chunks - 1:
                    return _pack_rest_and_return(line[j + chunk_size :] + "".join(lines[i + 1 :]))
            i += 1
            continue

        depth += _line_brace_delta(line)
        clean = _is_clean_split_boundary(line, depth, base_depth=base_depth)
        buf.append(line)
        cleans.append(clean)
        buf_len += len(line)

        if len(chunks) >= max_chunks - 1:
            return _pack_rest_and_return("".join(buf) + "".join(lines[i + 1 :]))

        # Prefer waiting for brace depth to drop; only force-cut at hard_cap.
        hard_cap = max(soft_max * 2, soft_max + 3000)
        if buf_len >= soft_max:
            cut = _last_clean_idx()
            if cut >= 0:
                _flush_upto(cut)
            elif buf_len >= hard_cap:
                logger.warning(
                    "hard-cap without clean brace boundary; cut at line end depth=%s buf_chars=%s",
                    depth,
                    buf_len,
                )
                _flush_upto(len(buf) - 1)
            # else: keep accumulating until depth drops or hard_cap
        elif buf_len >= chunk_size and clean:
            _flush_upto(len(buf) - 1)
        i += 1

    if buf:
        chunks.append("".join(buf))
    out = [c for c in chunks if c.strip()]
    joined = "".join(out)
    if joined != text:
        logger.warning(
            "chunk join mismatch orig=%s joined=%s; fallback line-pack",
            len(text),
            len(joined),
        )
        out = []
        buf2: list[str] = []
        n2 = 0
        for line in lines:
            if buf2 and n2 + len(line) > soft_max:
                out.append("".join(buf2))
                buf2, n2 = [line], len(line)
            else:
                buf2.append(line)
                n2 += len(line)
        if buf2:
            out.append("".join(buf2))
        out = [c for c in out if c.strip()]
    logger.info("code split chunks=%s sizes=%s", len(out), [len(c) for c in out])
    return out or [text]


def _dedupe_dict_list(items: list[Any], *, key_fields: tuple[str, ...]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        key = "|".join(str(it.get(k) or "") for k in key_fields)
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def _merge_segment_results_heuristic(
    cm: str,
    segments: list[dict[str, Any]],
    *,
    known_calls: list[str],
) -> dict[str, Any]:
    """Merge chunk analyses without another LLM call (fallback)."""
    biz_rules: list[Any] = []
    biz_bounds: list[Any] = []
    biz_impacts: list[Any] = []
    caps: list[str] = []
    params: list[Any] = []
    param_flow: list[Any] = []
    branches: list[Any] = []
    memory: list[Any] = []
    db_ops: list[Any] = []
    ext_iv: list[Any] = []
    validations: list[Any] = []
    status: list[Any] = []
    ext: list[Any] = []
    tables: list[Any] = []
    table_changes: list[Any] = []
    next_calls: list[Any] = []
    flows: list[str] = []
    biz_clears: list[str] = []
    biz_details: list[str] = []
    dev_clears: list[str] = []
    dev_details: list[str] = []
    ret: dict[str, Any] = {}
    types: list[str] = []
    important = False

    for i, seg in enumerate(segments, 1):
        if not isinstance(seg, dict):
            continue
        if seg.get("important"):
            important = True
        for t in seg.get("important_types") or []:
            if t in IMPORTANT_TYPES and t not in types:
                types.append(t)
        biz_clears.append(f"[段{i}] {seg.get('summary_business_clear') or ''}".strip())
        biz_details.append(f"[段{i}] {seg.get('summary_business_detail') or ''}".strip())
        dev_clears.append(f"[段{i}] {seg.get('summary_dev_clear') or ''}".strip())
        dev_details.append(f"[段{i}] {seg.get('summary_dev_detail') or ''}".strip())
        flow = str(seg.get("data_flow_narrative") or "").strip()
        if flow:
            flows.append(f"[段{i}] {flow}")
        bv = seg.get("business_view") if isinstance(seg.get("business_view"), dict) else {}
        if bv.get("capability"):
            caps.append(str(bv.get("capability")))
        biz_rules.extend(bv.get("rules") or [])
        biz_bounds.extend(bv.get("boundaries") or [])
        biz_impacts.extend(bv.get("impacts") or [])
        iv = seg.get("impl_view") if isinstance(seg.get("impl_view"), dict) else {}
        params.extend(iv.get("params") or [])
        for pf in iv.get("param_flow") or []:
            if isinstance(pf, dict):
                pf = dict(pf)
                pf["step"] = len(param_flow) + 1
                param_flow.append(pf)
        branches.extend(iv.get("branches") or [])
        memory.extend(iv.get("memory_objects") or [])
        db_ops.extend(iv.get("db_ops") or [])
        ext_iv.extend(iv.get("external_calls") or [])
        if isinstance(iv.get("return_value"), dict) and any(iv["return_value"].values()):
            ret = iv["return_value"]
        validations.extend(seg.get("validation_points") or [])
        status.extend(seg.get("status_transitions") or [])
        ext.extend(seg.get("external_calls") or [])
        tables.extend(seg.get("business_tables") or [])
        table_changes.extend(seg.get("table_field_changes") or [])
        next_calls.extend(seg.get("next_calls") or [])

    # unique string lists preserving order
    def _uniq(xs: list[Any]) -> list[Any]:
        seen: set[str] = set()
        out: list[Any] = []
        for x in xs:
            k = str(x)
            if k in seen or not k.strip():
                continue
            seen.add(k)
            out.append(x)
        return out

    data = {
        "important": important or True,
        "important_types": types[:6] or ["other_important"],
        "class_method": cm,
        "summary_business_clear": "；".join(x.replace("[段1] ", "", 1) if i == 0 else x for i, x in enumerate(biz_clears[:4]) if x),
        "summary_business_detail": "\n".join(x for x in biz_details if x),
        "summary_dev_clear": " → ".join(x for x in dev_clears if x)[:1200],
        "summary_dev_detail": "\n".join(x for x in dev_details if x),
        "business_view": {
            "capability": "；".join(_uniq(caps)[:4]) or cm,
            "rules": _uniq(biz_rules)[:12],
            "boundaries": _uniq(biz_bounds)[:8],
            "impacts": _uniq(biz_impacts)[:8],
        },
        "impl_view": {
            "params": _dedupe_dict_list(params, key_fields=("name", "source"))[:20],
            "param_flow": param_flow[:40],
            "branches": _dedupe_dict_list(branches, key_fields=("condition", "path"))[:20],
            "memory_objects": _dedupe_dict_list(memory, key_fields=("object",))[:20],
            "db_ops": _dedupe_dict_list(db_ops, key_fields=("op", "table_or_entity", "via_call"))[:20],
            "external_calls": _dedupe_dict_list(ext_iv, key_fields=("target",))[:20],
            "return_value": ret
            or {"assembly": "", "composition": "", "success_vs_fail": ""},
        },
        "data_flow_narrative": "\n".join(flows),
        "validation_points": _uniq(validations)[:20],
        "status_transitions": _uniq(status)[:20],
        "external_calls": _uniq(ext)[:20],
        "business_tables": _uniq(tables)[:20],
        "table_field_changes": _dedupe_dict_list(
            table_changes, key_fields=("table_or_entity", "field", "to_value", "op")
        )[:30],
        "next_calls": _uniq(list(known_calls) + [str(x) for x in next_calls])[:15],
        "segmented": True,
        "segment_count": len(segments),
    }
    data.update(_compose_dual_summaries(cm, data))
    return data


async def _analyze_deep_node_segmented(
    t: ParsedType,
    m: ParsedMethod,
    *,
    entity_summaries: dict[str, str] | None = None,
    downstream_blob: str = "",
) -> dict[str, Any]:
    """Chunk overlong method → per-segment LLM deep parse → merge."""
    cm = f"{t.simple_name}#{m.name}"
    code = m.raw_code or ""
    chunks = _split_method_code_chunks(code)
    logger.info(
        "segmented deep analyze %s chunks=%s chars=%s",
        cm,
        len(chunks),
        len(code),
    )
    ent_blob = ""
    if entity_summaries:
        bits = [f"{k}: {v}" for k, v in list(entity_summaries.items())[:12]]
        ent_blob = "\npreloaded_entities:\n" + "\n".join(bits)
    if downstream_blob:
        ent_blob = f"{ent_blob}\n{downstream_blob}".strip()

    # Signature header shared by all chunks (helps param continuity)
    header_lines = []
    for line in code.splitlines()[:8]:
        header_lines.append(line)
        if "{" in line:
            break
    header = "\n".join(header_lines)

    segment_results: list[dict[str, Any]] = []

    async def _one_chunk(i: int, chunk: str) -> dict[str, Any] | None:
        logger.info(
            "segmented chunk start %s %s/%s chars=%s",
            cm,
            i,
            len(chunks),
            len(chunk),
        )
        prompt = DEEP_CHUNK_PROMPT.format(chunk_idx=i, chunk_total=len(chunks))
        try:
            seg = await _llm_json(
                prompt,
                f"kind={t.symbol_kind}\nfqcn={t.fqcn}\nmodule={t.module_name}\n"
                f"class_method={cm}\nchunk={i}/{len(chunks)}\n"
                f"method_header:\n{header}\n{ent_blob}\n"
                f"本段源码:\n{chunk}",
                max_tokens=2200,
                kind=f"deep-chunk-{cm}-{i}/{len(chunks)}",
            )
            if not isinstance(seg, dict):
                return None
            seg["chunk_index"] = i
            return seg
        except Exception:
            logger.exception("chunk deep analyze failed %s#%s", cm, i)
            return None

    # Parallelize segment LLM calls — biggest win on long methods.
    chunk_outs = await asyncio.gather(
        *[_one_chunk(i, chunk) for i, chunk in enumerate(chunks, 1)]
    )
    segment_results = [s for s in chunk_outs if isinstance(s, dict)]
    segment_results.sort(key=lambda s: int(s.get("chunk_index") or 0))

    if not segment_results:
        return _heuristic_deep(t, m)

    # Prefer LLM merge for coherent dual summaries; fallback to heuristic merge.
    compact_segments = []
    for seg in segment_results:
        compact_segments.append(
            {
                "chunk_index": seg.get("chunk_index"),
                "summary_business_clear": seg.get("summary_business_clear"),
                "summary_business_detail": truncate(str(seg.get("summary_business_detail") or ""), 600),
                "summary_dev_clear": seg.get("summary_dev_clear"),
                "summary_dev_detail": truncate(str(seg.get("summary_dev_detail") or ""), 800),
                "business_view": seg.get("business_view") or {},
                "impl_view": seg.get("impl_view") or {},
                "data_flow_narrative": truncate(str(seg.get("data_flow_narrative") or ""), 800),
                "validation_points": (seg.get("validation_points") or [])[:8],
                "status_transitions": (seg.get("status_transitions") or [])[:8],
                "external_calls": (seg.get("external_calls") or [])[:8],
                "business_tables": (seg.get("business_tables") or [])[:8],
                "table_field_changes": (seg.get("table_field_changes") or [])[:8],
                "next_calls": (seg.get("next_calls") or [])[:8],
            }
        )
    # Fast merge by default (skip extra LLM merge call); set TS_FAST_MERGE=0 to force LLM merge.
    _fm = (os.environ.get("TS_FAST_MERGE") or "1").strip().lower()
    use_fast_merge = _fm not in ("0", "false", "no", "off")
    if use_fast_merge:
        merged = _merge_segment_results_heuristic(
            cm, segment_results, known_calls=list(m.calls or [])
        )
        merged["segmented"] = True
        merged["segment_count"] = len(chunks)
        merged["fast_merge"] = True
        return merged
    try:
        merged = await _llm_json(
            DEEP_MERGE_PROMPT,
            f"class_method={cm}\nfqcn={t.fqcn}\nmodule={t.module_name}\n"
            f"known_calls={list(m.calls or [])[:20]}\n"
            f"segments={json.dumps(compact_segments, ensure_ascii=False)}",
            max_tokens=3500,
            kind=f"deep-merge-{cm}",
        )
        if not isinstance(merged, dict):
            raise ValueError("merge not object")
        # Ensure completeness if merge truncated — backfill empty fields from heuristic.
        heur = _merge_segment_results_heuristic(
            cm, segment_results, known_calls=list(m.calls or [])
        )
        iv_m = merged.get("impl_view") if isinstance(merged.get("impl_view"), dict) else {}
        iv_h = heur.get("impl_view") if isinstance(heur.get("impl_view"), dict) else {}
        for k in (
            "params",
            "param_flow",
            "branches",
            "memory_objects",
            "db_ops",
            "external_calls",
        ):
            if not iv_m.get(k) and iv_h.get(k):
                iv_m[k] = iv_h[k]
        ret_m = iv_m.get("return_value") if isinstance(iv_m.get("return_value"), dict) else {}
        ret_h = iv_h.get("return_value") if isinstance(iv_h.get("return_value"), dict) else {}
        if not any(str(ret_m.get(x) or "").strip() for x in ("assembly", "composition", "success_vs_fail")):
            if ret_h:
                iv_m["return_value"] = ret_h
        merged["impl_view"] = iv_m
        for k in (
            "data_flow_narrative",
            "summary_business_clear",
            "summary_business_detail",
            "summary_dev_clear",
            "summary_dev_detail",
            "validation_points",
            "status_transitions",
            "external_calls",
            "business_tables",
            "table_field_changes",
            "next_calls",
        ):
            if not merged.get(k) and heur.get(k):
                merged[k] = heur[k]
        bv_m = merged.get("business_view") if isinstance(merged.get("business_view"), dict) else {}
        bv_h = heur.get("business_view") if isinstance(heur.get("business_view"), dict) else {}
        if bv_h:
            for k in ("capability", "rules", "boundaries", "impacts"):
                if not bv_m.get(k) and bv_h.get(k):
                    bv_m[k] = bv_h[k]
            merged["business_view"] = bv_m
        merged["class_method"] = cm
        merged["segmented"] = True
        merged["segment_count"] = len(segment_results)
    except Exception:
        logger.exception("segment merge LLM failed %s; heuristic merge", cm)
        merged = _merge_segment_results_heuristic(
            cm, segment_results, known_calls=list(m.calls or [])
        )

    types = [x for x in (merged.get("important_types") or []) if x in IMPORTANT_TYPES]
    merged["important_types"] = types or ["other_important"]
    merged["important"] = True if merged.get("important") is None else bool(merged.get("important"))
    if not merged.get("next_calls"):
        merged["next_calls"] = list(m.calls or [])[:15]
    merged.update(_compose_dual_summaries(cm, merged))
    if re.search(r"(?i)(risk|asyn|async)", m.name or ""):
        merged["important"] = True
        if "external_call" not in merged["important_types"]:
            merged["important_types"] = (merged["important_types"] or []) + ["external_call"]
    return merged


@traceable(name="treesitter_deep_node", run_type="llm")
async def _analyze_deep_node(
    t: ParsedType,
    m: ParsedMethod,
    *,
    entity_summaries: dict[str, str] | None = None,
    known_domains: list[str] | None = None,
    project_id: str | None = None,
    downstream_blob: str = "",
) -> dict[str, Any]:
    cm = f"{t.simple_name}#{m.name}"
    code = m.raw_code or ""
    domains = list(known_domains or [])
    # Overlong method: segment → analyze → merge (full coverage, avoid JSON truncate)
    if len(code) > DEEP_CHUNK_TRIGGER:
        try:
            return await _analyze_deep_node_segmented(
                t,
                m,
                entity_summaries=entity_summaries,
                downstream_blob=downstream_blob,
            )
        except Exception:
            logger.exception("segmented deep analyze failed %s; fallback single", cm)

    ent_blob = ""
    if entity_summaries:
        bits = [f"{k}: {v}" for k, v in list(entity_summaries.items())[:20]]
        ent_blob = "\npreloaded_entities:\n" + "\n".join(bits)
    if downstream_blob:
        ent_blob = f"{ent_blob}\n{downstream_blob}".strip()
    code_limit = 5500 if (t.symbol_kind == "Controller" or len(code) > 2500) else 3800
    try:
        data = await _llm_json(
            _deep_node_system_prompt(domains),
            f"kind={t.symbol_kind}\nfqcn={t.fqcn}\nmodule={t.module_name}\n"
            f"class_method={cm}\nknown_calls={list(m.calls or [])[:20]}\n{ent_blob}\n"
            f"要求：必须写清每个关键变量从哪来、如何改写、传给谁；四类摘要都要写满。"
            f"若提供了 mysql_downstream_descs：本方法只做编排/调用串联，下游内部细节直接引用该描述，"
            f"禁止臆造或重复展开下游源码。\n"
            f"code:\n{truncate(code, code_limit)}",
            max_tokens=4500,
            kind=f"deep-node-{cm}",
        )
        data["class_method"] = cm
        if project_id:
            data["business_domain"] = resolve_business_domain(
                project_id,
                str(data.get("business_domain") or ""),
                domain_action=str(data.get("domain_action") or ""),
                known=domains,
            )
        types = [x for x in (data.get("important_types") or []) if x in IMPORTANT_TYPES]
        data["important_types"] = types
        if not data.get("next_calls"):
            data["next_calls"] = list(m.calls or [])[:15]
        composed = _compose_dual_summaries(cm, data)
        data.update(composed)
        if re.search(r"(?i)(risk|asyn|async)", m.name or ""):
            data["important"] = True
            if "external_call" not in data["important_types"]:
                data["important_types"] = (data["important_types"] or []) + ["external_call"]
        return data
    except Exception:
        logger.exception("deep node analyze failed %s", cm)
        return _heuristic_deep(t, m)


async def _analyze_node_cached(
    state: _DeepState,
    t: ParsedType,
    m: ParsedMethod,
    mid: str,
    entity_summaries: dict[str, str],
    *,
    project_id: str,
    downstream_blob: str = "",
) -> dict[str, Any]:
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
        known = state.snapshot_domains()
    if not owner:
        data = await fut
        async with state.lock:
            state.nodes_cache_hit += 1
        return dict(data)
    try:
        data = await _analyze_deep_node(
            t,
            m,
            entity_summaries=entity_summaries,
            known_domains=known,
            project_id=project_id,
            downstream_blob=downstream_blob,
        )
        async with state.lock:
            if data.get("business_domain"):
                state.note_domain(
                    project_id,
                    str(data.get("business_domain") or ""),
                    domain_action=str(data.get("domain_action") or "reuse"),
                )
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


def _persist_method(project_id: str, t: ParsedType, method: ParsedMethod) -> None:
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
    try:
        write_parsed_type_graph(project_id, slim)
    except Exception:
        pass


def _iter_local_methods(bfs_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten local-source methods from by_depth / tree (dedupe by class_method)."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_depth = bfs_result.get("by_depth") or {}
    if isinstance(by_depth, dict):
        for d in sorted(by_depth.keys(), key=lambda x: int(str(x).split("_")[-1] or 0)):
            depth_n = int(str(d).split("_")[-1] or 0)
            for n in by_depth.get(d) or []:
                if not isinstance(n, dict):
                    continue
                cm = str(n.get("class_method") or "")
                if not cm or cm in seen:
                    continue
                if not n.get("file"):
                    continue
                seen.add(cm)
                row = dict(n)
                row.setdefault("depth", depth_n)
                out.append(row)
    if out:
        return out

    def walk(node: dict[str, Any]) -> None:
        cm = str(node.get("class_method") or "")
        if cm and cm not in seen and node.get("file") and not node.get("leaf_reason"):
            seen.add(cm)
            out.append(node)
        for c in node.get("children") or []:
            if isinstance(c, dict):
                walk(c)

    tree = bfs_result.get("tree")
    if isinstance(tree, dict):
        walk(tree)
    return out


def _layers_deep_to_shallow(nodes: list[dict[str, Any]]) -> list[tuple[int, list[dict[str, Any]]]]:
    """Group methods by BFS depth; deepest layer first (Service before Controller)."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        d = int(n.get("depth") or 0)
        buckets.setdefault(d, []).append(n)
    return [(d, buckets[d]) for d in sorted(buckets.keys(), reverse=True)]


def _callee_map_from_bfs(bfs_result: dict[str, Any]) -> dict[str, list[str]]:
    """from class_method → ordered unique callee class_methods (local edges)."""
    out: dict[str, list[str]] = {}
    for edge in bfs_result.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        if edge.get("status") not in (None, "ok_local", "ok"):
            # keep ok_local; also allow missing status from YAML reuse
            if edge.get("status") and "local" not in str(edge.get("status")):
                continue
        frm = str(edge.get("from") or "")
        to = str(edge.get("to") or "")
        if "#" not in frm or "#" not in to:
            continue
        bucket = out.setdefault(frm, [])
        if to not in bucket:
            bucket.append(to)
    # Tree children as fallback
    def walk(node: dict[str, Any]) -> None:
        cm = str(node.get("class_method") or "")
        for c in node.get("children") or []:
            if not isinstance(c, dict):
                continue
            child = str(c.get("class_method") or "")
            if cm and child and "#" in child:
                bucket = out.setdefault(cm, [])
                if child not in bucket:
                    bucket.append(child)
            walk(c)

    tree = bfs_result.get("tree")
    if isinstance(tree, dict):
        walk(tree)
    return out


def _bfs_discovery_order(bfs_result: dict[str, Any]) -> list[str]:
    """Entry-locked local methods in BFS discovery order (by_depth then tree)."""
    return [
        str(n.get("class_method") or "")
        for n in _iter_local_methods(bfs_result)
        if n.get("class_method") and "#" in str(n.get("class_method") or "")
    ]


def _bfs_order_index(bfs_result: dict[str, Any]) -> dict[str, int]:
    return {cm: i for i, cm in enumerate(_bfs_discovery_order(bfs_result))}


def _reachable_cms(bfs_result: dict[str, Any], entry_cm: str) -> set[str]:
    """All local class_methods reachable from entry via BFS call edges/tree."""
    if not entry_cm or "#" not in entry_cm:
        return set()
    callee_map = _callee_map_from_bfs(bfs_result)
    seen: set[str] = set()
    stack = [entry_cm]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        for to in callee_map.get(cur) or []:
            if to not in seen:
                stack.append(to)
    return seen


def _is_off_chain_controller_sibling(
    cm: str,
    entry_cm: str,
    reachable: set[str],
) -> bool:
    """Same Controller type, different method, not on this entry's call chain."""
    if not cm or "#" not in cm or not entry_cm or "#" not in entry_cm:
        return False
    c_simple, c_method = cm.split("#", 1)
    e_simple, e_method = entry_cm.split("#", 1)
    if c_simple != e_simple or c_method == e_method:
        return False
    if not c_simple.endswith("Controller"):
        return False
    return cm not in reachable


def _match_call_to_cm(call: str, known_cms: set[str]) -> str | None:
    """Map rough AST call ``obj.method`` / ``method`` to a known Class#method."""
    raw = (call or "").strip().split("(")[0].strip()
    if not raw:
        return None
    if "#" in raw:
        return raw if raw in known_cms else None
    if "." in raw:
        obj, method = raw.rsplit(".", 1)
    else:
        obj, method = "", raw
    method = (method or "").strip()
    obj = (obj or "").strip()
    if not method or method in {
        "getLock",
        "tryLock",
        "buildSuccess",
        "buildFailure",
        "info",
        "error",
        "debug",
        "toJSONString",
        "put",
        "get",
        "builder",
        "build",
        "isLocked",
        "isHeldByCurrentThread",
        "unlock",
        "isNull",
        "getKey",
        "getCode",
        "getMessage",
        "getInnerCode",
        "getInnerMsg",
    }:
        return None
    hits = [cm for cm in known_cms if cm.endswith(f"#{method}")]
    if not hits:
        return None
    if len(hits) == 1:
        return hits[0]

    obj_l = obj.lower().replace("this", "")
    if obj_l:
        scored: list[tuple[int, str]] = []
        for cm in hits:
            simple = cm.split("#", 1)[0]
            sl = simple.lower()
            score = 0
            if obj_l in sl or sl.replace("impl", "") in obj_l:
                score += 10
            # withdrawalsApplyService → prefer *ApplyService*
            if "apply" in obj_l and "apply" in sl:
                score += 5
            if obj_l.startswith("withdrawals") and sl.startswith("withdrawals"):
                score += 2
            if simple.endswith("Impl"):
                score += 1
            scored.append((score, cm))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if scored and scored[0][0] > 0:
            return scored[0][1]

    impls = [cm for cm in hits if cm.split("#", 1)[0].endswith("Impl")]
    if len(impls) == 1:
        return impls[0]
    preferred = [
        cm
        for cm in hits
        if any(
            x in cm.split("#", 1)[0]
            for x in ("ApplyService", "ServiceImpl", "Service")
        )
    ]
    if len(preferred) == 1:
        return preferred[0]
    return None


def _entry_call_spine(
    bfs_result: dict[str, Any],
    entry_cm: str,
    *,
    entry_calls: list[str] | None = None,
    known_cms: set[str] | None = None,
) -> list[str]:
    """Ordered direct execution steps from the locked entry.

    Prefer AST call sites on the entry method (Controller→Service1→…→Service2).
    When AST spine is rich enough, it wins over tree-child append order so helpers
    like generateReqParam are not shoved to the end. Tree children fill gaps only
    when AST calls are sparse.
    """
    reachable = _reachable_cms(bfs_result, entry_cm)
    known = set(known_cms or ()) | reachable | {entry_cm}
    callee_map = _callee_map_from_bfs(bfs_result)
    tree_spine = [
        cm
        for cm in (callee_map.get(entry_cm) or [])
        if cm and "#" in cm and cm != entry_cm
    ]

    call_spine: list[str] = []
    seen: set[str] = set()
    for call in entry_calls or []:
        hit = _match_call_to_cm(str(call), known)
        if not hit or hit == entry_cm or hit in seen:
            continue
        if _is_off_chain_controller_sibling(
            hit, entry_cm, reachable | set(tree_spine) | set(call_spine) | {hit}
        ):
            if hit not in reachable and hit not in tree_spine:
                continue
        seen.add(hit)
        call_spine.append(hit)

    # Also map unqualified local calls (checkApplyReqParam) via tree / known Controller methods.
    entry_simple = entry_cm.split("#", 1)[0] if "#" in entry_cm else ""
    for call in entry_calls or []:
        raw = str(call).strip().split("(")[0].strip()
        if "." in raw or "#" in raw:
            continue
        cand = f"{entry_simple}#{raw}"
        if cand in known and cand not in seen and cand != entry_cm:
            # Insert keeping relative order: append if not already placed via this.
            seen.add(cand)
            call_spine.append(cand)

    if len(call_spine) >= 3:
        # AST order is authoritative; optionally weave in on-chain Controller helpers
        # from tree that sit between known neighbors (by tree order).
        tree_only = [
            cm
            for cm in tree_spine
            if cm not in seen
            and not _is_off_chain_controller_sibling(
                cm, entry_cm, reachable | set(call_spine)
            )
        ]
        if not tree_only:
            return call_spine
        # Insert tree-only Controller helpers after the nearest preceding spine item
        # that appears before them in tree_spine order.
        tree_idx = {cm: i for i, cm in enumerate(tree_spine)}
        merged = list(call_spine)
        for cm in tree_only:
            # Only weave private Controller helpers; skip extra service callees.
            if not cm.startswith(entry_simple + "#"):
                continue
            ti = tree_idx.get(cm, 10_000)
            insert_at = len(merged)
            for j, existing in enumerate(merged):
                if tree_idx.get(existing, -1) < ti:
                    insert_at = j + 1
            merged.insert(insert_at, cm)
            seen.add(cm)
        return merged

    if call_spine:
        for cm in tree_spine:
            if cm not in seen and not _is_off_chain_controller_sibling(
                cm, entry_cm, reachable | set(call_spine)
            ):
                seen.add(cm)
                call_spine.append(cm)
        return call_spine

    return [
        cm
        for cm in tree_spine
        if not _is_off_chain_controller_sibling(cm, entry_cm, reachable)
    ]


def _sort_important_by_bfs(
    payloads: list[dict[str, Any]],
    order_index: dict[str, int],
) -> list[dict[str, Any]]:
    def key(p: dict[str, Any]) -> tuple[int, int, str]:
        cm = str(p.get("class_method") or "")
        depth = int(p.get("bfs_depth") if p.get("bfs_depth") is not None else 10_000)
        idx = order_index.get(cm, 10_000 + depth)
        return (idx, depth, cm)

    return sorted(payloads, key=key)


def _build_execution_stages(
    *,
    entry_cm: str,
    bfs_result: dict[str, Any],
    important_payloads: list[dict[str, Any]],
    llm_stages: list[Any] | None,
    entry_calls: list[str] | None = None,
    max_stages: int = 24,
) -> list[dict[str, Any]]:
    """Authoritative FlowDoc stages = entry call spine (BFS/AST execution order)."""
    order_index = _bfs_order_index(bfs_result)
    reachable = _reachable_cms(bfs_result, entry_cm)
    payload_by_cm = {
        str(p.get("class_method") or ""): p
        for p in important_payloads
        if p.get("class_method")
    }
    known = set(payload_by_cm) | set(order_index) | reachable
    spine = _entry_call_spine(
        bfs_result,
        entry_cm,
        entry_calls=entry_calls,
        known_cms=known,
    )
    # Keep spine methods; drop Controller siblings not on chain.
    stage_cms = [
        cm
        for cm in spine
        if cm
        and cm != entry_cm
        and not _is_off_chain_controller_sibling(cm, entry_cm, reachable | set(spine))
    ][:max_stages]

    llm_by_cm: dict[str, dict[str, Any]] = {}
    for st in llm_stages or []:
        if not isinstance(st, dict):
            continue
        cm = str(st.get("class_method") or "").strip()
        if not cm or "#" not in cm:
            continue
        if _is_off_chain_controller_sibling(cm, entry_cm, reachable | set(stage_cms)):
            continue
        llm_by_cm[cm] = st

    stages: list[dict[str, Any]] = []
    for i, cm in enumerate(stage_cms, 1):
        p = payload_by_cm.get(cm) or {}
        llm = llm_by_cm.get(cm) or {}
        biz = str(
            llm.get("desc_business")
            or p.get("summary_business_clear")
            or p.get("summary")
            or ""
        )
        dev = str(
            llm.get("desc_dev")
            or p.get("summary_dev_clear")
            or p.get("data_flow_narrative")
            or ""
        )
        title = str(llm.get("title") or "").strip() or (
            cm.split("#", 1)[-1] if "#" in cm else cm
        )
        stages.append(
            {
                "step": i,
                "title": title[:120],
                "class_method": cm,
                "desc_business": biz,
                "desc_dev": dev,
                "desc": str(llm.get("desc") or biz or dev or cm),
                "bfs_order": order_index.get(cm),
                "bfs_depth": p.get("bfs_depth"),
            }
        )
    return stages


def _load_mysql_downstream_descs(
    project_id: str,
    class_methods: list[str],
    *,
    limit: int = 24,
) -> str:
    """Load already-persisted deeper method descriptions from MySQL for shallow orchestration."""
    from sqlalchemy import select

    from app.models.orm import SymbolDescORM, SymbolORM

    wanted = [cm for cm in class_methods if cm and "#" in cm][:limit]
    if not wanted:
        return ""

    lines: list[str] = []
    with get_session() as session:
        for cm in wanted:
            simple, method = cm.split("#", 1)
            syms = list(
                session.scalars(
                    select(SymbolORM).where(
                        SymbolORM.project_id == project_id,
                        SymbolORM.simple_name == simple,
                        SymbolORM.method_name == method,
                    )
                )
            )
            if not syms:
                continue
            for sym in syms[:2]:
                desc = session.scalar(
                    select(SymbolDescORM).where(SymbolDescORM.symbol_id == sym.id)
                )
                if not desc:
                    continue
                biz = truncate(str(desc.summary_business_clear or ""), 280)
                dev = truncate(str(desc.summary_dev_clear or ""), 360)
                extra = desc.extra_json if isinstance(desc.extra_json, dict) else {}
                flow = truncate(str(extra.get("data_flow_narrative") or ""), 280)
                if not biz and not dev:
                    biz = truncate(str(desc.summary or ""), 360)
                if not (biz or dev or flow):
                    continue
                chunk = f"- {cm}"
                if biz:
                    chunk += f"\n  业务: {biz}"
                if dev:
                    chunk += f"\n  研发: {dev}"
                if flow:
                    chunk += f"\n  数据流: {flow}"
                lines.append(chunk)
                break
    if not lines:
        return ""
    return (
        "mysql_downstream_descs（下游已入库描述，浅层只编排、勿重写下游内部细节）:\n"
        + "\n".join(lines)
    )


def _collect_downstream_cms(
    cm: str,
    *,
    callee_map: dict[str, list[str]],
    depth_by_cm: dict[str, int],
    self_depth: int,
    max_n: int = 20,
) -> list[str]:
    """Callees of cm that are strictly deeper (or unknown depth)."""
    out: list[str] = []
    seen: set[str] = set()
    for to in callee_map.get(cm) or []:
        if to in seen or to == cm:
            continue
        td = depth_by_cm.get(to)
        if td is not None and td <= self_depth:
            continue
        seen.add(to)
        out.append(to)
        if len(out) >= max_n:
            break
    return out


@traceable(name="treesitter_deep_flow", run_type="chain")
async def run_treesitter_deep_flow(
    project_id: str,
    project_root: Path,
    *,
    entry_file: Path | None = None,
    entry_method: str = "withdrawals",
    flow_key: str = "loan_disburse",
    flow_role: str = "main",
    pilot_id: str = "loan_api",
    http_path: str = "withdrawals/withdrawals",
    max_depth: int = 0,
    max_nodes: int = 120,
    concurrency: int = 10,
    progress: Callable[[int, int, str], None] | None = None,
    reuse_bfs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full pipeline: tree-sitter BFS → entity preload → dual deep analyze → persist."""
    demo = _import_demo()
    settings = get_settings()
    java_file = entry_file or demo.JAVA_FILE
    index = list_java_index(project_root)
    state = _DeepState()
    state.known_domains = list_known_domains(project_id, refresh=True)
    if progress and state.known_domains:
        progress(
            0,
            4,
            f"known business domains={len(state.known_domains)}: "
            + ",".join(state.known_domains[:12]),
        )

    if progress:
        progress(0, 4, "tree-sitter deep BFS")
    if reuse_bfs and isinstance(reuse_bfs.get("tree"), dict):
        bfs = reuse_bfs
    else:
        bfs = await asyncio.to_thread(
            demo.deep_bfs,
            java_file,
            entry_method,
            max_depth=max_depth,
            max_nodes=max(max_nodes, 500),
        )
    if bfs.get("error"):
        return {"error": bfs["error"]}

    # --- Phase 1: preload Model/DTO/Entity (each once) ---
    type_names = collect_model_dto_names(bfs)
    if progress:
        progress(1, 4, f"preload {len(type_names)} model/DTO entities")
    preloaded = await preload_entities(
        state,
        project_id,
        project_root,
        index,
        type_names,
        concurrency=concurrency,
        progress=progress,
    )
    # Build entity summary map for method prompts
    entity_summaries: dict[str, str] = {}
    from sqlalchemy import select

    from app.models.orm import SymbolDescORM

    with get_session() as session:
        for fqcn in preloaded:
            sid = symbol_id_for(project_id, fqcn, None)
            d = session.scalar(select(SymbolDescORM).where(SymbolDescORM.symbol_id == sid))
            if d and d.summary:
                entity_summaries[fqcn.split(".")[-1]] = truncate(d.summary, 180)

    # --- Phase 2: dual-perspective method analysis (deep → shallow layers) ---
    # Selection keeps entry-first BFS list + max_nodes cap; execution is by depth
    # deepest→shallowest. Same depth runs concurrently; Controller waits for Service.
    local_methods = _iter_local_methods(bfs)[:max_nodes]
    layers = _layers_deep_to_shallow(local_methods)
    callee_map = _callee_map_from_bfs(bfs)
    depth_by_cm = {
        str(n.get("class_method") or ""): int(n.get("depth") or 0)
        for n in local_methods
        if n.get("class_method")
    }
    if progress:
        depth_bits = ",".join(f"d{d}:{len(ns)}" for d, ns in layers)
        progress(
            2,
            4,
            f"deep-analyze {len(local_methods)} methods deep→shallow [{depth_bits}]",
        )

    sem = asyncio.Semaphore(max(1, concurrency))
    important_payloads: list[dict[str, Any]] = []
    local_milvus: list[dict] = []
    entry_cm = f"{Path(str(java_file)).stem}#{entry_method}"
    # Prefer controller simple name from bfs
    entry_cm = str((bfs.get("entry") or entry_cm))
    entry_t: ParsedType | None = None
    entry_m: ParsedMethod | None = None
    entry_sid: str | None = None

    async def _one(node: dict[str, Any], *, depth: int) -> None:
        nonlocal entry_t, entry_m, entry_sid
        cm = str(node.get("class_method") or "")
        if "#" not in cm:
            return
        simple, method_name = cm.split("#", 1)
        async with sem:
            hit = resolve_type_file(index, simple, prefer_impl=True)
            if not hit:
                async with state.lock:
                    state.missing.append(f"method_file_missing:{cm}")
                    state.nodes_finished += 1
                return
            loaded = await asyncio.to_thread(
                load_method, project_root, hit["path"], method_name
            )
            if not loaded:
                async with state.lock:
                    state.missing.append(f"method_body_missing:{cm}")
                    state.nodes_finished += 1
                return
            t, m = loaded
            # Shallow layers: read this method only + MySQL descs of deeper callees
            callee_cms = _collect_downstream_cms(
                cm,
                callee_map=callee_map,
                depth_by_cm=depth_by_cm,
                self_depth=depth,
            )
            downstream_blob = ""
            if callee_cms:
                downstream_blob = await asyncio.to_thread(
                    _load_mysql_downstream_descs, project_id, callee_cms
                )
            async with state.lock:
                state.type_by_simple[_type_cache_key(t.simple_name)] = t
                state.nodes_started += 1
                state.nodes_inflight += 1
                started = state.nodes_started
                inflight = state.nodes_inflight
            code_chars = len(m.raw_code or "")
            if progress:
                progress(
                    2,
                    4,
                    f"start d{depth} {started}/{len(local_methods)} "
                    f"inflight={inflight} {cm} chars={code_chars} "
                    f"mysql_down={len(callee_cms) if downstream_blob else 0}"
                    + (" [segmented]" if code_chars > DEEP_CHUNK_TRIGGER else ""),
                )
            try:
                await asyncio.to_thread(_persist_method, project_id, t, m)
                mid = symbol_id_for(project_id, t.fqcn, m.signature)
                data = await _analyze_node_cached(
                    state,
                    t,
                    m,
                    mid,
                    entity_summaries,
                    project_id=project_id,
                    downstream_blob=downstream_blob,
                )
                is_entry = cm == entry_cm or (
                    t.simple_name.endswith("Controller") and m.name == entry_method
                )
                if is_entry:
                    data["important"] = True
                    if not data.get("important_types"):
                        data["important_types"] = ["other_important"]
                    entry_t, entry_m = t, m
                    entry_sid = mid

                try:
                    await asyncio.to_thread(
                        upsert_method_node,
                        project_id,
                        t,
                        m,
                        important=bool(data.get("important")),
                        important_types=list(data.get("important_types") or []),
                        flow_keys=[flow_key],
                        is_entry=is_entry,
                        summary=str(data.get("summary_mysql") or data.get("summary") or "")[:1500],
                        data_flow_narrative=str(data.get("data_flow_narrative") or ""),
                        business_tables=list(data.get("business_tables") or []),
                        table_field_changes=list(data.get("table_field_changes") or []),
                        status_transitions=list(data.get("status_transitions") or []),
                        bfs_depth=depth,
                    )
                except Exception:
                    logger.warning(
                        "upsert_method_node failed for %s (continuing)",
                        cm,
                        exc_info=True,
                    )

                summary_pack = {
                    "summary_business_clear": str(data.get("summary_business_clear") or ""),
                    "summary_business_detail": str(data.get("summary_business_detail") or ""),
                    "summary_dev_clear": str(data.get("summary_dev_clear") or ""),
                    "summary_dev_detail": str(data.get("summary_dev_detail") or ""),
                }
                summary = str(data.get("summary_mysql") or data.get("summary") or "")
                flow_bits = str(data.get("data_flow_narrative") or "")
                iv = data.get("impl_view") if isinstance(data.get("impl_view"), dict) else {}
                table_changes = list(data.get("table_field_changes") or [])
                # Derive table_field_changes from db_ops / status when LLM omitted them.
                if not table_changes and isinstance(iv, dict):
                    for op in iv.get("db_ops") or []:
                        if not isinstance(op, dict):
                            continue
                        fs = op.get("field_sources") if isinstance(op.get("field_sources"), dict) else {}
                        for field, src in (fs or {}).items():
                            table_changes.append(
                                {
                                    "table_or_entity": op.get("table_or_entity") or "",
                                    "field": field,
                                    "from_value": "未知",
                                    "to_value": str(src),
                                    "op": op.get("op") or "update",
                                    "when": op.get("persist_changes") or op.get("via_call") or "",
                                }
                            )
                if not table_changes:
                    for st in data.get("status_transitions") or []:
                        table_changes.append(
                            {
                                "table_or_entity": "",
                                "field": "",
                                "from_value": "",
                                "to_value": str(st),
                                "op": "update",
                                "when": "",
                            }
                        )

                # Persist every analyzed method (+ description) so Milvus/Neo4j can impact-trace.
                tags = ["双向深度解析", flow_key, flow_role]
                if data.get("important") or is_entry:
                    tags = ["重要节点"] + tags + list(data.get("important_types") or [])
                if is_entry:
                    tags = ["流程入口"] + tags
                row = await asyncio.to_thread(
                    upsert_description,
                    project_id,
                    mid,
                    summary=summary,
                    business_domain=flow_key,
                    tags=tags,
                    extra={
                        "important": bool(data.get("important") or is_entry),
                        "important_types": data.get("important_types") or [],
                        "flow_keys": [flow_key],
                        "flow_roles": [flow_role],
                        "pilot_entry_id": pilot_id,
                        "class_method": cm,
                        "fqcn": t.fqcn,
                        "module_name": t.module_name or "",
                        "method_name": m.name,
                        "method_sig": m.signature,
                        "bfs_depth": depth,
                        "business_tables": data.get("business_tables") or [],
                        "table_field_changes": table_changes,
                        "status_transitions": data.get("status_transitions") or [],
                        "external_calls": data.get("external_calls") or [],
                        "validation_points": data.get("validation_points") or [],
                        "business_view": data.get("business_view") or {},
                        "impl_view": iv,
                        "data_flow_narrative": flow_bits,
                        **summary_pack,
                        "parse_engine": "tree-sitter-java",
                        "parse_mode": "deep-bfs-dual-perspective",
                        "segmented": bool(data.get("segmented")),
                        "segment_count": data.get("segment_count") or 1,
                    },
                )
                tables_txt = ",".join(str(x) for x in (data.get("business_tables") or [])[:12])
                changes_txt = json.dumps(table_changes[:8], ensure_ascii=False)[:600]
                row["class_method"] = cm
                row["bfs_depth"] = depth
                row["text"] = truncate(
                    f"{row.get('text', '')}\n"
                    f"做什么: {summary_pack['summary_business_clear']}\n"
                    f"研发精简: {summary_pack['summary_dev_clear']}\n"
                    f"相关表: {tables_txt}\n"
                    f"字段变更: {changes_txt}\n"
                    f"数据流: {flow_bits}\n"
                    f"参数流: {json.dumps(iv.get('param_flow') or [], ensure_ascii=False)[:800]}",
                    3500,
                )
                async with state.lock:
                    local_milvus.append(row)
                    if data.get("important") or is_entry:
                        important_payloads.append(
                            {
                                "class_method": cm,
                                "bfs_depth": depth,
                                "summary": summary_pack["summary_business_clear"],
                                "summary_business_clear": summary_pack["summary_business_clear"],
                                "summary_dev_clear": summary_pack["summary_dev_clear"],
                                "important_types": data.get("important_types") or [],
                                "validation_points": data.get("validation_points") or [],
                                "status_transitions": data.get("status_transitions") or [],
                                "external_calls": data.get("external_calls") or [],
                                "business_tables": data.get("business_tables") or [],
                                "table_field_changes": table_changes,
                                "business_view": data.get("business_view") or {},
                                "param_flow": iv.get("param_flow") or [],
                                "data_flow_narrative": flow_bits,
                            }
                        )
                        state.total_important += 1

                if progress:
                    async with state.lock:
                        state.nodes_finished += 1
                        state.nodes_inflight = max(0, state.nodes_inflight - 1)
                        finished = state.nodes_finished
                        inflight_left = state.nodes_inflight
                    progress(
                        2,
                        4,
                        f"done d{depth} {finished}/{len(local_methods)} "
                        f"inflight={inflight_left} {cm}"
                        + (" important" if data.get("important") else ""),
                    )
                else:
                    async with state.lock:
                        state.nodes_finished += 1
                        state.nodes_inflight = max(0, state.nodes_inflight - 1)
            except Exception:
                async with state.lock:
                    state.nodes_finished += 1
                    state.nodes_inflight = max(0, state.nodes_inflight - 1)
                raise

    for depth, layer_nodes in layers:
        if progress:
            progress(
                2,
                4,
                f"layer depth={depth} size={len(layer_nodes)} "
                f"(shallow reads MySQL descs of deeper callees)",
            )
        await asyncio.gather(*[_one(n, depth=depth) for n in layer_nodes])

    # important_payloads were appended in deep→shallow completion order — reorder to
    # BFS discovery / entry call order and drop off-chain Controller sibling HTTP methods.
    order_index = _bfs_order_index(bfs)
    reachable = _reachable_cms(bfs, entry_cm)
    important_payloads = [
        p
        for p in important_payloads
        if not _is_off_chain_controller_sibling(
            str(p.get("class_method") or ""), entry_cm, reachable
        )
    ]
    important_payloads = _sort_important_by_bfs(important_payloads, order_index)
    entry_calls = list(entry_m.calls or []) if entry_m else []
    known_for_spine = (
        {str(p.get("class_method") or "") for p in important_payloads}
        | set(order_index)
        | reachable
    )
    # Interface callees (e.g. withdrawalsApplyService.loanApply) may be missing from BFS.
    if any("loanApply" in str(c) for c in entry_calls):
        if not any(
            cm.endswith("#loanApply") and "ApplyService" in cm for cm in known_for_spine
        ):
            known_for_spine.add("WithdrawalsApplyServiceImpl#loanApply")
    execution_spine = _entry_call_spine(
        bfs,
        entry_cm,
        entry_calls=entry_calls,
        known_cms=known_for_spine,
    )

    # Wire CALLS edges from tree-sitter BFS (execution call order for impact analysis).
    calls_written = 0
    id_by_cm: dict[str, str] = {}
    with get_session() as session:
        from app.models.orm import SymbolORM

        for sym in session.scalars(
            select(SymbolORM).where(
                SymbolORM.project_id == project_id,
                SymbolORM.method_name.is_not(None),
            )
        ):
            if not sym.method_name:
                continue
            cm_key = f"{sym.simple_name}#{sym.method_name}"
            id_by_cm.setdefault(cm_key, sym.id)
    # order = appearance order within each parent method body (not global edge index)
    order_by_from: dict[str, int] = {}
    for edge in bfs.get("edges") or []:
        if not isinstance(edge, dict) or edge.get("status") != "ok_local":
            continue
        frm = str(edge.get("from") or "")
        to = str(edge.get("to") or "")
        if "#" not in frm or "#" not in to:
            continue
        a = id_by_cm.get(frm)
        b = id_by_cm.get(to)
        if not a or not b:
            continue
        order_by_from[frm] = order_by_from.get(frm, 0) + 1
        order = order_by_from[frm]
        try:
            await asyncio.to_thread(
                merge_calls,
                project_id,
                a,
                b,
                order=order,
                source="treesitter-bfs",
                flow_key=flow_key,
            )
            calls_written += 1
        except Exception:
            logger.debug("merge_calls failed %s -> %s", frm, to, exc_info=True)

    # --- Phase 3: flow narrative (business/dev × clear/detail) ---
    if progress:
        progress(3, 4, "flow narrative")
    spine_payloads = []
    by_cm_imp = {
        str(p.get("class_method") or ""): p for p in important_payloads if p.get("class_method")
    }
    for cm in execution_spine[:40]:
        if cm in by_cm_imp:
            spine_payloads.append(by_cm_imp[cm])
        else:
            spine_payloads.append({"class_method": cm, "summary_business_clear": cm})
    # Fill remaining important nodes (still BFS-ordered) for LLM context, not stage order.
    seen_spine = {str(p.get("class_method") or "") for p in spine_payloads}
    for p in important_payloads:
        cm = str(p.get("class_method") or "")
        if cm and cm not in seen_spine:
            spine_payloads.append(p)
            seen_spine.add(cm)
    try:
        narrative = await _llm_json(
            FLOW_PROMPT,
            f"flow_key={flow_key}\nflow_role={flow_role}\n"
            f"entry={entry_cm}\nhttp_path={http_path}\n"
            f"execution_spine={json.dumps(execution_spine[:30], ensure_ascii=False)}\n"
            f"important_nodes={json.dumps(spine_payloads[:40], ensure_ascii=False)}",
            max_tokens=3200,
            kind=f"flow-{flow_key}",
        )
    except Exception:
        lines = [f"入口 {entry_cm}（{flow_role}）"]
        stages = []
        for i, n in enumerate(spine_payloads[:20], 1):
            lines.append(f"{i}. {n.get('summary_business_clear') or n.get('summary')}")
            stages.append(
                {
                    "step": i,
                    "title": n.get("class_method") or "",
                    "class_method": n.get("class_method") or "",
                    "desc_business": n.get("summary_business_clear") or "",
                    "desc_dev": n.get("summary_dev_clear") or "",
                    "desc": n.get("summary") or "",
                }
            )
        narrative = {
            "title": f"{flow_key}:{flow_role}",
            "narrative_business_clear": "\n".join(lines),
            "narrative_business_detail": "\n".join(lines),
            "narrative_dev_clear": "\n".join(
                f"{i}. {n.get('summary_dev_clear') or n.get('summary')}"
                for i, n in enumerate(spine_payloads[:20], 1)
            ),
            "narrative_dev_detail": "\n".join(
                f"{i}. {n.get('data_flow_narrative') or n.get('summary_dev_clear') or ''}"
                for i, n in enumerate(spine_payloads[:20], 1)
            ),
            "narrative": "\n".join(lines),
            "stages": stages,
        }

    nb_clear = str(narrative.get("narrative_business_clear") or "").strip()
    nb_detail = str(narrative.get("narrative_business_detail") or "").strip()
    nd_clear = str(narrative.get("narrative_dev_clear") or "").strip()
    nd_detail = str(narrative.get("narrative_dev_detail") or "").strip()
    narrative_text = str(narrative.get("narrative") or "").strip()
    if not narrative_text:
        narrative_text = "\n".join(
            x for x in (nb_clear, nd_clear) if x
        ) or f"流程 {entry_cm}"
    if not nb_clear:
        nb_clear = narrative_text
    if not nd_clear:
        nd_clear = narrative_text
    if not nb_detail:
        nb_detail = nb_clear
    if not nd_detail:
        nd_detail = nd_clear

    title = str(narrative.get("title") or f"{flow_key}:{pilot_id}")
    # Prefer BFS/AST entry call spine over LLM-invented stage membership/order.
    stages = _build_execution_stages(
        entry_cm=entry_cm,
        bfs_result=bfs,
        important_payloads=important_payloads,
        llm_stages=list(narrative.get("stages") or []),
        entry_calls=entry_calls,
    )
    chain_pos = {str(st.get("class_method") or ""): int(st.get("step") or 0) for st in stages}

    if entry_t and entry_m and entry_sid:
        entry_hit = next(
            (n for n in important_payloads if (n.get("class_method") or "") == entry_cm),
            {},
        )
        # Keep dual summaries from method deep-parse; do not replace with thin one-liner.
        biz_c = str(entry_hit.get("summary_business_clear") or "").strip()
        dev_c = str(entry_hit.get("summary_dev_clear") or "").strip()
        entry_summary = truncate(
            f"{entry_cm}\n【业务·精简】{biz_c or '流程入口'}\n【研发·精简】{dev_c or entry_hit.get('data_flow_narrative') or ''}\n"
            f"【流程·业务精简】{nb_clear}\n【流程·研发精简】{nd_clear}",
            8000,
        )
        entry_row = upsert_description(
            project_id,
            entry_sid,
            summary=entry_summary,
            business_domain=flow_key,
            tags=["流程入口", "重要节点", "双向深度解析", flow_key, flow_role, entry_cm],
            extra={
                "entry": True,
                "important": True,
                "class_method": entry_cm,
                "fqcn": entry_t.fqcn,
                "module_name": entry_t.module_name or "",
                "method_name": entry_m.name,
                "http_paths": [http_path],
                "pilot_entry_id": pilot_id,
                "flow_keys": [flow_key],
                "flow_roles": [flow_role],
                "parse_engine": "tree-sitter-java",
                "parse_mode": "deep-bfs-dual-perspective",
                "preloaded_entities": preloaded[:80],
                "summary_business_clear": biz_c,
                "summary_dev_clear": dev_c,
                "data_flow_narrative": str(entry_hit.get("data_flow_narrative") or ""),
                "flow_doc": {
                    "flow_key": flow_key,
                    "flow_role": flow_role,
                    "pilot_entry_id": pilot_id,
                    "title": title,
                    "narrative": narrative_text,
                    "narrative_business_clear": nb_clear,
                    "narrative_business_detail": nb_detail,
                    "narrative_dev_clear": nd_clear,
                    "narrative_dev_detail": nd_detail,
                    "stages": stages,
                    "execution_spine": execution_spine,
                    "stage_order_source": "bfs_entry_call_spine",
                },
            },
        )
        stage_lines = " → ".join(
            str(st.get("class_method") or "") for st in stages[:16]
        )
        entry_row["text"] = truncate(
            f"{entry_row.get('text', '')}\n"
            f"链上位置: 入口 step=0\n"
            f"执行序: {entry_cm} → {stage_lines}\n"
            f"做什么: {biz_c}\n"
            f"业务精简流程: {nb_clear}\n研发精简流程: {nd_clear}\n"
            f"业务详细流程: {nb_detail}\n研发详细流程: {nd_detail}",
            3500,
        )
        local_milvus.append(entry_row)

    entry_module = (entry_t.module_name if entry_t else "") or ""
    doc_id = stable_id("flow", project_id, flow_key, pilot_id, entry_sid or entry_cm)
    stage_blob = "\n".join(
        f"{st.get('step')}. [{st.get('class_method')}] {st.get('title')} — "
        f"做什么: {truncate(str(st.get('desc_business') or st.get('desc') or ''), 160)}"
        for st in stages[:20]
    )
    local_milvus.append(
        {
            "id": doc_id,
            "project_id": project_id,
            "symbol_id": f"flow:{flow_key}:{pilot_id}"[:64],
            "fqcn": f"__flow__:{flow_key}:{flow_role}",
            "module_name": entry_module,
            "tags": (
                f"业务流程,{flow_key},{flow_role},flow,{entry_cm},tree-sitter,"
                f"双向深度解析,{entry_module}"
            ).strip(","),
            "text": truncate(
                f"业务流程 {title} flow_key={flow_key} role={flow_role} "
                f"module={entry_module} 入口 {entry_cm} path={http_path}\n"
                f"执行序(入口调用链): {entry_cm} → "
                + " → ".join(str(st.get("class_method") or "") for st in stages[:16])
                + "\n"
                f"【业务精简】{nb_clear}\n【研发精简】{nd_clear}\n"
                f"【业务详细】{nb_detail}\n【研发详细】{nd_detail}\n"
                f"【阶段·做什么+链上位置+表字段】\n{stage_blob}",
                3500,
            ),
        }
    )

    # Patch method Milvus texts with chain position (stages known only after flow phase).
    for row in local_milvus:
        if not isinstance(row, dict):
            continue
        cm_hit = str(row.get("class_method") or "")
        if not cm_hit or cm_hit not in chain_pos:
            continue
        step = chain_pos[cm_hit]
        text0 = str(row.get("text") or "")
        marker = f"链上位置: 入口执行序 step={step}/{len(stages)}"
        if marker not in text0:
            row["text"] = truncate(f"{text0}\n{marker}", 3500)

    async with state.lock:
        state.milvus_rows.extend(local_milvus)
        state.flow_docs += 1
        all_rows = list(state.milvus_rows)

    if progress:
        progress(4, 4, f"upsert milvus rows={len(all_rows)}")
    upsert_milvus_rows(all_rows)

    return {
        "engine": "tree-sitter-java",
        "mode": "deep-bfs-dual-perspective",
        "entry": entry_cm,
        "http_path": http_path,
        "flow_key": flow_key,
        "bfs_nodes": bfs.get("nodes_count"),
        "bfs_edges": bfs.get("edges_count"),
        "calls_written": calls_written,
        "model_dto_candidates": len(type_names),
        "entities_preloaded": len(preloaded),
        "entities_described": state.entities_described,
        "dtos_described": state.dtos_described,
        "methods_analyzed": state.nodes_ai,
        "nodes_cache_hit": state.nodes_cache_hit,
        "important_nodes": state.total_important,
        "flow_docs": state.flow_docs,
        "flow_title": title,
        "flow_narrative_preview": truncate(narrative_text, 500),
        "execution_spine": execution_spine[:30],
        "stage_cms": [str(st.get("class_method") or "") for st in stages],
        "milvus_rows": len(all_rows),
        "missing": state.missing[:40],
        "preloaded_fqcns": preloaded[:60],
        "bfs": bfs,
    }
