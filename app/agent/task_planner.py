"""Structured research plans for project Q&A (shown live in the chat UI)."""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Awaitable, Callable

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None] | None]

logger = logging.getLogger("app.agent.planner")

# Question kinds that drive plan templates + explore strategy hints.
KIND_STATUS_UPDATE = "status_update"
KIND_VALIDATION = "validation"
KIND_ENDPOINT = "endpoint"
KIND_DOWNSTREAM = "downstream_apis"
KIND_API_INVENTORY = "api_inventory"
KIND_FULL_FLOW = "full_flow"
KIND_FLOW = "flow"
KIND_ADD_FIELD = "add_field_plan"
KIND_SHOW_GATE_CODE = "show_gate_code"
KIND_GENERAL = "general"
# Removed domain kinds (protocol_sync / composite / gate_detail) — do not schedule
# business short-circuit tools in research plans.

_VALID_KINDS = {
    KIND_STATUS_UPDATE,
    KIND_VALIDATION,
    KIND_ENDPOINT,
    KIND_DOWNSTREAM,
    KIND_API_INVENTORY,
    KIND_FULL_FLOW,
    KIND_FLOW,
    KIND_ADD_FIELD,
    KIND_SHOW_GATE_CODE,
    KIND_GENERAL,
}

# Keep aliases so old session/plan payloads do not crash.
KIND_PROTOCOL_SYNC = "protocol_sync_plan"  # deprecated alias
KIND_COMPOSITE_FIELD = "composite_field_plan"  # deprecated alias
KIND_GATE_DETAIL = "gate_detail"  # deprecated alias

_STATUS_UPDATE_KEYS = (
    "状态怎么更新",
    "状态如何更新",
    "状态是怎么更新",
    "状态更新",
    "怎么更新的",
    "如何更新的",
    "谁更新",
    "谁改的",
    "怎么改的",
    "字段怎么赋值",
    "setStatus",
    "更新状态",
    "状态变更",
    "状态流转",
    "状态写入",
    "写状态",
)
_TABLE_KEYS = ("表", "实体", "Entity", "落库", "哪张表", "哪个表")
_STATUS_KEYS = ("状态", "status", "Status", "state", "State")

_PLAN_PROMPT = """你是 Java/Spring 项目代码研究规划助手。根据用户问题，输出「如何查代码」的步骤计划。
只输出严格 JSON（不要 markdown）：
{
  "kind": "status_update|validation|endpoint|downstream_apis|api_inventory|full_flow|flow|add_field_plan|show_gate_code|general",
  "title": "一句话研究目标（中文，必须贴合用户原话）",
  "steps": [
    {
      "id": "snake_case",
      "title": "短标题（中文）",
      "detail": "这一步具体查什么、怎么判断（中文，含问题里的业务词）",
      "tools": ["search_symbols"]
    }
  ]
}

规则：
1. 步骤 3～6 个，按真实调研顺序；最后一步必须是整理作答，id 用 analyze_synthesize，tools 含 finish。
2. 针对本题定制，禁止空泛套话；title/detail 要出现用户问题中的业务词（如表名、状态、流程名）。
3. id 用英文 snake_case，且在本计划内唯一。
4. tools 可选，取值仅限：search_symbols, get_symbol_code, list_callees, list_callers, resolve_impl, discover_add_field_plan, finish。
5. kind 选最贴切的一类；状态/表字段如何更新 → status_update；流程/什么流程 → flow/full_flow；校验字段 → validation；新增字段落库改哪些 → add_field_plan。
6. 可用工具能力：语义搜索符号、读源码、查调用方/被调方、接口找实现类、新增字段改动发现。按这些能力规划，不要写做不到的步骤。
7. 流程/链路类问题禁止规划「协议同步」「syncSign」「透传字段」「关卡外部透传」等改动发现步骤；那些不是流程解读。
"""


def classify_question_kind(question: str) -> str:
    """Fallback research kind — Goal/LLM own routing; no keyword cascade."""
    _ = question or ""
    return KIND_GENERAL


def _is_status_update_question(text: str) -> bool:
    if any(k in text for k in _STATUS_UPDATE_KEYS):
        return True
    has_table = any(k in text for k in _TABLE_KEYS)
    has_status = any(k in text for k in _STATUS_KEYS)
    has_how = any(
        k in text for k in ("怎么", "如何", "更新", "赋值", "写入", "修改", "变更")
    )
    return bool(has_table and has_status and has_how) or bool(
        has_status and has_how and ("更新" in text or "赋值" in text or "写入" in text)
    )


def _step(
    step_id: str,
    title: str,
    detail: str = "",
    *,
    status: str = "pending",
    tools: list[str] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "id": step_id,
        "title": title,
        "detail": detail,
        "status": status,  # pending | running | done | skipped | error
        "note": "",
    }
    if tools:
        out["tools"] = tools
    return out


def build_research_plan(question: str, *, project_id: str | None = None) -> dict[str, Any]:
    """Heuristic fallback plan (sync). Prefer build_research_plan_async for LLM plans."""
    kind = classify_question_kind(question)
    noun = _extract_topic_hint(question)

    if kind == KIND_STATUS_UPDATE:
        title = f"分析「{noun or '业务'}」表/实体状态如何更新"
        steps = [
            _step(
                "locate_table",
                "定位相关表/实体",
                f"按问题实词检索 Entity/表名（如 {noun or '业务词'} 相关实体）",
                tools=["search_symbols"],
            ),
            _step(
                "identify_fields",
                "确认状态字段",
                "从实体源码/摘要中识别 status/state 等字段及可能取值",
                tools=["get_symbol_code"],
            ),
            _step(
                "find_writers",
                "查找字段赋值点",
                "搜索 setStatus / update / Mapper 等对状态字段写值的代码",
                tools=["search_symbols", "get_symbol_code"],
            ),
            _step(
                "trace_call_chains",
                "还原调用链",
                "用 list_callers / list_callees 找出触发更新的入口与下游",
                tools=["list_callers", "list_callees", "resolve_impl"],
            ),
            _step(
                "analyze_synthesize",
                "整理分析并作答",
                "归纳何时、由谁、如何更新状态，给出业务可读结论",
                tools=["finish"],
            ),
        ]
    elif kind == KIND_VALIDATION:
        title = f"梳理「{noun or '入口'}」字段/参数校验"
        steps = [
            _step("locate_entry", "定位业务入口", "搜索相关 Controller/入口方法", tools=["search_symbols"]),
            _step("read_entry", "阅读入口源码", "读取入口方法与 same_class_calls", tools=["get_symbol_code"]),
            _step("read_validators", "下钻校验方法", "逐个阅读校验/组装相关私有方法", tools=["get_symbol_code"]),
            _step("analyze_synthesize", "整理校验清单", "逐条归纳字段、规则与失败含义", tools=["finish"]),
        ]
    elif kind == KIND_ADD_FIELD:
        title = f"分析「{noun or '接口'}」新增入参字段并落库要改哪些"
        steps = [
            _step(
                "discover_roles",
                "结构化发现改动触点",
                "调用 discover_add_field_plan：入口 CALLS 闭包发现常量/VO/绑定/组装/实体并渲染方案",
                tools=["discover_add_field_plan"],
            ),
            _step(
                "analyze_synthesize",
                "输出改动方案",
                "finish 原样采用 plan_markdown（## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明）",
                tools=["finish"],
            ),
        ]
    elif kind == KIND_SHOW_GATE_CODE:
        title = f"贴出「{noun or '关卡'}」对应源码"
        steps = [
            _step(
                "locate_method",
                "定位关卡方法",
                "按会话 last_gate / last_symbols 或题中方法名取 Class#method",
                tools=["get_symbol_code"],
            ),
            _step(
                "analyze_synthesize",
                "贴出方法源码",
                "原样输出 Java 方法体，禁止编造",
                tools=["finish"],
            ),
        ]
    elif kind == KIND_ENDPOINT:
        title = f"查找「{noun or '下游'}」服务地址"
        steps = [
            _step("search_client", "搜索 Feign/Client", "按服务实词检索 *Client / Feign", tools=["search_symbols"]),
            _step("read_client", "阅读 Client 注解", "提取 name/url/path/Mapping", tools=["get_symbol_code"]),
            _step("analyze_synthesize", "直答地址事实", "汇总服务名与路径", tools=["finish"]),
        ]
    elif kind == KIND_DOWNSTREAM:
        title = f"列出「{noun or '主链'}」过程中调用的下游"
        steps = [
            _step("lock_entry", "锁定主链入口", "确定本题对应的 Controller#method", tools=["search_symbols"]),
            _step("gather_stages", "收集主链阶段", "优先用 FlowDoc stages / 调用图", tools=["get_symbol_code"]),
            _step(
                "list_downstream",
                "枚举下游调用",
                "Service/Feign/MQ 等，排除兄弟 HTTP 入口",
                tools=["list_callees", "resolve_impl"],
            ),
            _step("analyze_synthesize", "输出下游清单", "入口 + 共 N 条下游", tools=["finish"]),
        ]
    elif kind == KIND_API_INVENTORY:
        title = f"清点「{noun or '主题'}」相关接口"
        steps = [
            _step("search_controllers", "搜索 Controller 入口", "按问题实词检索接口", tools=["search_symbols"]),
            _step("collect_endpoints", "汇总 path/用途", "补全 HTTP path 与一句话说明", tools=["get_symbol_code"]),
            _step("analyze_synthesize", "输出接口清单", "数量与列表一致", tools=["finish"]),
        ]
    elif kind == KIND_FULL_FLOW:
        title = f"梳理「{noun or '业务'}」完整链路（含回调）"
        steps = [
            _step("locate_entry", "定位同步业务入口", "找到本题 HTTP/业务入口", tools=["search_symbols"]),
            _step("gather_materials", "收集离线流程材料", "FlowDoc / Overview / 节点摘要", tools=["search_symbols"]),
            _step("read_core", "阅读入口与实现", "Controller → ServiceImpl / 校验与写库", tools=["get_symbol_code", "resolve_impl"]),
            _step("trace_callback", "查找结果回调/通知", "Listener / Callback / Notice", tools=["search_symbols", "get_symbol_code"]),
            _step("analyze_synthesize", "整理主链+回调", "有序阶段 + 结果回调分节", tools=["finish"]),
        ]
    elif kind == KIND_FLOW:
        title = f"梳理「{noun or '业务'}」流程"
        steps = [
            _step("locate_entry", "定位业务入口", "按问题实词锁定 Controller#method", tools=["search_symbols"]),
            _step("gather_materials", "收集流程材料", "检索 FlowDoc / 节点摘要", tools=["search_symbols"]),
            _step("read_core", "阅读核心实现", "下钻 ServiceImpl 与关键校验/状态", tools=["get_symbol_code", "resolve_impl"]),
            _step("follow_chain", "沿调用链核实", "必要时 list_callees / resolve_impl", tools=["list_callees", "resolve_impl"]),
            _step("analyze_synthesize", "整理业务步骤", "用业务语言说明流程", tools=["finish"]),
        ]
    else:
        title = "检索项目代码并作答"
        steps = [
            _step("search", "检索相关符号", "按问题实词搜索代码与摘要", tools=["search_symbols"]),
            _step("read_code", "阅读关键源码", "打开命中方法核实事实", tools=["get_symbol_code"]),
            _step("analyze_synthesize", "整理回答", "基于证据给出结论", tools=["finish"]),
        ]

    return {
        "kind": kind,
        "title": title,
        "question": question,
        "project_id": project_id or "",
        "steps": steps,
        "source": "heuristic",
    }


def _scrub_banned_plan_steps(plan: dict[str, Any]) -> dict[str, Any]:
    """Remove protocol-sync / composite short-circuit steps from any plan."""
    q = str(plan.get("question") or "")
    steps = [
        s
        for s in (plan.get("steps") or [])
        if isinstance(s, dict)
        and not _BANNED_PLAN_STEP_RE.search(
            f"{s.get('title') or ''}\n{s.get('detail') or ''}\n"
            f"{' '.join(s.get('tools') or [])}"
        )
    ]
    if steps:
        plan["steps"] = steps
    title = str(plan.get("title") or "")
    if _BANNED_PLAN_STEP_RE.search(title):
        plan["title"] = f"梳理「{_extract_topic_hint(q) or '业务'}」流程"
    kind = str(plan.get("kind") or "")
    if kind in {KIND_PROTOCOL_SYNC, KIND_COMPOSITE_FIELD, KIND_GATE_DETAIL}:
        plan["kind"] = (
            KIND_FLOW if ("流程" in q or "链路" in q) else KIND_GENERAL
        )
    return plan


async def build_research_plan_async(
    question: str,
    *,
    project_id: str | None = None,
) -> dict[str, Any]:
    """LLM-generated research plan tailored to the question; heuristic on failure."""
    fallback = _scrub_banned_plan_steps(
        build_research_plan(question, project_id=project_id)
    )
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        content = await llm.chat(
            [
                {"role": "system", "content": _PLAN_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"project_id={project_id or ''}\n"
                        f"用户问题：{question}\n"
                        f"启发式 kind 参考（可调整）：{fallback.get('kind')}"
                    ),
                },
            ],
            temperature=0.2,
            max_tokens=700,
        )
        plan = _parse_llm_plan(content, question=question, project_id=project_id)
        if plan:
            plan["source"] = "llm"
            return _scrub_banned_plan_steps(plan)
    except Exception as e:
        logger.warning("dynamic research plan failed (%s); using heuristic", e)
    return fallback


def _parse_llm_plan(
    content: str,
    *,
    question: str,
    project_id: str | None,
) -> dict[str, Any] | None:
    if not content:
        return None
    m = re.search(r"\{.*\}", content, re.DOTALL)
    raw = m.group(0) if m else content
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    steps_in = data.get("steps")
    if not isinstance(steps_in, list) or not steps_in:
        return None

    kind = str(data.get("kind") or "").strip()
    if kind not in _VALID_KINDS:
        kind = classify_question_kind(question)

    title = str(data.get("title") or "").strip()
    if not title:
        title = f"研究：{_extract_topic_hint(question) or '项目问题'}"

    steps: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(steps_in[:8]):
        if not isinstance(item, dict):
            continue
        sid = _normalize_step_id(str(item.get("id") or f"step_{i+1}"), seen)
        title_s = str(item.get("title") or "").strip() or f"步骤{i+1}"
        detail = str(item.get("detail") or item.get("desc") or "").strip()
        tools = _normalize_tools(item.get("tools"))
        # Drop domain short-circuit steps the LLM sometimes invents on flow asks.
        blob = f"{title_s}\n{detail}\n{' '.join(tools)}"
        if _BANNED_PLAN_STEP_RE.search(blob):
            continue
        steps.append(_step(sid, title_s, detail, tools=tools or None))

    if len(steps) < 2:
        return None

    # Flow/overview kinds must never keep a protocol-sync-shaped title.
    if kind in {KIND_FLOW, KIND_FULL_FLOW, KIND_GENERAL, KIND_DOWNSTREAM}:
        if _BANNED_PLAN_STEP_RE.search(title):
            title = f"梳理「{_extract_topic_hint(question) or '业务'}」流程"

    # Normalize final step id so progress / finish mapping stay stable.
    last = steps[-1]
    last_id = str(last.get("id") or "")
    looks_final = (
        last_id == "analyze_synthesize"
        or "finish" in (last.get("tools") or [])
        or any(k in str(last.get("title") or "") for k in ("整理", "作答", "汇总", "结论"))
    )
    if not looks_final:
        steps.append(
            _step(
                "analyze_synthesize",
                "整理分析并作答",
                "基于已收集证据给出结论",
                tools=["finish"],
            )
        )
        last = steps[-1]
        last_id = "analyze_synthesize"
    if last_id != "analyze_synthesize":
        if last_id in seen:
            seen.discard(last_id)
        last["id"] = "analyze_synthesize"
        seen.add("analyze_synthesize")
    tools = list(last.get("tools") or [])
    if "finish" not in tools:
        tools.append("finish")
        last["tools"] = tools

    return {
        "kind": kind,
        "title": title,
        "question": question,
        "project_id": project_id or "",
        "steps": steps,
        "source": "llm",
    }


def _normalize_step_id(raw: str, seen: set[str]) -> str:
    sid = re.sub(r"[^a-zA-Z0-9_]+", "_", (raw or "").strip()).strip("_").lower()
    if not sid:
        sid = "step"
    if not re.match(r"^[a-z]", sid):
        sid = f"s_{sid}"
    base = sid
    n = 2
    while sid in seen:
        sid = f"{base}_{n}"
        n += 1
    seen.add(sid)
    return sid


_BANNED_PLAN_STEP_RE = re.compile(
    r"协议同步|syncSign|discover_protocol_sync|discover_composite_field|"
    r"外部透传|关卡校验.*透传|字段透传"
)


def _normalize_tools(raw: Any) -> list[str]:
    allowed = {
        "search_symbols",
        "get_symbol_code",
        "list_callees",
        "list_callers",
        "resolve_impl",
        "discover_add_field_plan",
        "finish",
    }
    banned = {
        "discover_protocol_sync_plan",
        "discover_composite_field_plan",
    }
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for t in raw:
        name = str(t or "").strip()
        if name in banned:
            continue
        if name in allowed and name not in out:
            out.append(name)
    return out


def plan_strategy_hint(plan: dict[str, Any]) -> str:
    """Inject into explore LLM user message so tools follow the visible plan."""
    kind = plan.get("kind") or KIND_GENERAL
    lines = ["【研究计划】请按下列步骤推进（可用工具落地，证据够再 finish）："]
    for i, s in enumerate(plan.get("steps") or [], 1):
        tools = s.get("tools") or []
        tool_bit = f"（可用 {'/'.join(tools)}）" if tools else ""
        lines.append(f"{i}. {s.get('title')} — {s.get('detail') or ''}{tool_bit}")
    if kind == KIND_STATUS_UPDATE:
        lines.append(
            "策略重点：先 search_symbols(kind=Entity 或 q=表/实体实词) 定位表与状态字段；"
            "再 search 赋值/update/Mapper；用 list_callers 还原谁触发更新；"
            "回答写清「何时、由谁、如何改状态」。"
        )
    elif kind == KIND_ADD_FIELD:
        lines.append(
            "策略重点：优先 discover_add_field_plan 结构化发现触点并渲染方案；"
            "finish 原样采用 plan_markdown（## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明）；"
            "禁止背诵固定清单；禁止旧标题「改动清单/建议改动」；禁止 insert* 关卡罗列。"
        )
    elif kind == KIND_SHOW_GATE_CODE:
        lines.append(
            "策略重点：按会话 last_gate/last_symbols 贴出真实方法源码，禁止编造。"
        )
    return "\n".join(lines)


def map_tool_to_step_id(
    tool: str,
    plan_kind: str,
    *,
    args: dict[str, Any] | None = None,
    plan_steps: list[dict[str, Any]] | None = None,
) -> str | None:
    """Best-effort map a tool call onto a plan step id (heuristic ids or tools hints)."""
    args = args or {}
    kind_arg = str(args.get("kind") or "").lower()
    q = str(args.get("q") or "")

    # Dynamic plans: prefer a step that declared this tool.
    if plan_steps:
        for s in plan_steps:
            if s.get("status") == "done":
                continue
            tools = s.get("tools") or []
            if tool in tools:
                return str(s.get("id") or "") or None

    if plan_kind == KIND_STATUS_UPDATE:
        if tool == "search_symbols":
            if kind_arg == "entity" or any(
                k in q for k in ("Entity", "实体", "表", "DO", "PO")
            ):
                return "locate_table"
            if any(
                k in q.lower()
                for k in ("status", "update", "setstatus", "mapper", "状态", "更新")
            ):
                return "find_writers"
            return "locate_table"
        if tool == "get_symbol_code":
            return "identify_fields"
        if tool in {"list_callers", "list_callees", "resolve_impl"}:
            return "trace_call_chains"
        if tool == "finish":
            return "analyze_synthesize"
    if plan_kind == KIND_VALIDATION:
        return {
            "search_symbols": "locate_entry",
            "get_symbol_code": "read_validators",
            "list_callees": "read_validators",
            "resolve_impl": "read_entry",
            "finish": "analyze_synthesize",
        }.get(tool)
    if plan_kind == KIND_ADD_FIELD:
        if tool == "discover_add_field_plan":
            return "discover_roles"
        if tool == "search_symbols":
            return "discover_roles"
        if tool == "get_symbol_code":
            return "discover_roles"
        if tool == "resolve_impl":
            return "discover_roles"
        if tool == "finish":
            return "analyze_synthesize"
        return None
    if plan_kind == KIND_SHOW_GATE_CODE:
        if tool in {"get_symbol_code", "search_symbols"}:
            return "locate_method"
        if tool == "finish":
            return "analyze_synthesize"
        return None
    if plan_kind == KIND_ENDPOINT:
        return {
            "search_symbols": "search_client",
            "get_symbol_code": "read_client",
            "finish": "analyze_synthesize",
        }.get(tool)
    if plan_kind == KIND_DOWNSTREAM:
        return {
            "search_symbols": "lock_entry",
            "get_symbol_code": "gather_stages",
            "list_callees": "list_downstream",
            "resolve_impl": "list_downstream",
            "list_callers": "list_downstream",
            "finish": "analyze_synthesize",
        }.get(tool)
    if plan_kind == KIND_API_INVENTORY:
        return {
            "search_symbols": "search_controllers",
            "get_symbol_code": "collect_endpoints",
            "finish": "analyze_synthesize",
        }.get(tool)
    if plan_kind in {KIND_FULL_FLOW, KIND_FLOW}:
        if tool == "search_symbols":
            q_l = q.lower()
            if any(k in q_l for k in ("callback", "listener", "notice", "回调", "通知")):
                return "trace_callback" if plan_kind == KIND_FULL_FLOW else "follow_chain"
            return "locate_entry"
        if tool == "get_symbol_code":
            return "read_core"
        if tool in {"list_callees", "resolve_impl", "list_callers"}:
            return "follow_chain" if plan_kind == KIND_FLOW else "read_core"
        if tool == "finish":
            return "analyze_synthesize"
    return {
        "search_symbols": "search",
        "get_symbol_code": "read_code",
        "list_callees": "read_code",
        "list_callers": "read_code",
        "resolve_impl": "read_code",
        "finish": "analyze_synthesize",
    }.get(tool)


def _extract_topic_hint(question: str) -> str:
    text = question or ""
    m = re.search(r"([\u4e00-\u9fff]{2,8})表", text)
    if m:
        return m.group(1)
    m = re.search(r"([\u4e00-\u9fff]{2,8})状态", text)
    if m:
        return m.group(1)
    m = re.search(r"([\u4e00-\u9fff]{2,8})(?:流程|接口|相关)", text)
    if m:
        return m.group(1)
    stops = {"怎么", "如何", "什么", "哪些", "请问", "一下", "这个", "那个", "是否"}
    for m in re.finditer(r"[\u4e00-\u9fff]{2,6}", text):
        w = m.group(0)
        if w not in stops and not any(s in w for s in ("怎么", "如何", "什么")):
            return w
    return ""


class PlanProgress:
    """Mutable plan + async/sync progress fan-out for SSE UI."""

    def __init__(
        self,
        plan: dict[str, Any],
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.plan = plan
        self._on_progress = on_progress
        self._step_index = {s["id"]: s for s in plan.get("steps") or []}

    def rebind(self, plan: dict[str, Any]) -> None:
        """Replace plan (e.g. after LLM refinement) and rebuild index."""
        self.plan = plan
        self._step_index = {s["id"]: s for s in plan.get("steps") or []}

    def step_ids(self) -> list[str]:
        return [str(s.get("id")) for s in (self.plan.get("steps") or []) if s.get("id")]

    def has_step(self, step_id: str) -> bool:
        return step_id in self._step_index

    def synthesize_step_id(self) -> str | None:
        if "analyze_synthesize" in self._step_index:
            return "analyze_synthesize"
        ids = self.step_ids()
        return ids[-1] if ids else None

    async def emit(self, event: dict[str, Any]) -> None:
        if not self._on_progress:
            return
        try:
            result = self._on_progress(event)
            if hasattr(result, "__await__"):
                await result  # type: ignore[misc]
        except Exception:
            pass

    async def publish_plan(self) -> None:
        await self.emit({"type": "plan", "plan": self.plan})

    async def start(self, step_id: str, note: str = "") -> None:
        step = self._step_index.get(step_id)
        if not step:
            return
        if step.get("status") == "done":
            return
        step["status"] = "running"
        if note:
            step["note"] = note
        await self.emit(
            {
                "type": "step",
                "step_id": step_id,
                "status": "running",
                "note": note or step.get("detail") or "",
                "plan": self.plan,
            }
        )

    async def done(self, step_id: str, note: str = "") -> None:
        step = self._step_index.get(step_id)
        if not step:
            return
        step["status"] = "done"
        if note:
            step["note"] = note
        await self.emit(
            {
                "type": "step",
                "step_id": step_id,
                "status": "done",
                "note": note or step.get("note") or "",
                "plan": self.plan,
            }
        )

    async def done_at(self, index: int, note: str = "") -> None:
        ids = self.step_ids()
        if 0 <= index < len(ids):
            await self.done(ids[index], note)

    async def mark_remaining_done(self, note: str = "已完成") -> None:
        for s in self.plan.get("steps") or []:
            if s.get("status") in {"pending", "running"}:
                s["status"] = "done"
                if note and not s.get("note"):
                    s["note"] = note
        await self.emit({"type": "step", "step_id": "*", "status": "done", "plan": self.plan})

    def _status(self, step_id: str) -> str:
        step = self._step_index.get(step_id)
        return str(step.get("status") or "pending") if step else "pending"

    def _resolve_step_for_tool(
        self, tool: str, *, args: dict[str, Any] | None = None
    ) -> str | None:
        kind = self.plan.get("kind") or ""
        steps = list(self.plan.get("steps") or [])

        # 1) tools hint on unfinished steps
        for s in steps:
            if s.get("status") == "done":
                continue
            if tool in (s.get("tools") or []):
                return str(s.get("id") or "") or None

        # 2) heuristic id if it exists in this plan
        mapped = map_tool_to_step_id(tool, kind, args=args, plan_steps=steps)
        if mapped and mapped in self._step_index and self._status(mapped) != "done":
            return mapped

        # Status-update special case only when those ids exist.
        if kind == KIND_STATUS_UPDATE and tool == "get_symbol_code":
            if "identify_fields" in self._step_index and self._status("identify_fields") != "done":
                if self._status("locate_table") in {"done", "running", "pending"}:
                    return "identify_fields"
            if "find_writers" in self._step_index and self._status("identify_fields") == "done":
                return "find_writers"

        # 3) keep current running step
        for s in steps:
            if s.get("status") == "running":
                return str(s.get("id") or "") or None

        # 4) first pending
        for s in steps:
            if s.get("status") == "pending":
                return str(s.get("id") or "") or None
        return None

    async def tool_activity(
        self, tool: str, *, args: dict[str, Any] | None = None, note: str = ""
    ) -> None:
        step_id = self._resolve_step_for_tool(tool, args=args)
        if not step_id:
            return
        await self.start(step_id, note or f"执行 {tool}")

    async def tool_done(
        self, tool: str, *, args: dict[str, Any] | None = None, note: str = ""
    ) -> None:
        step_id = self._resolve_step_for_tool(tool, args=args)
        # finish always closes synthesize if present
        if tool == "finish" and "analyze_synthesize" in self._step_index:
            step_id = "analyze_synthesize"
        if step_id:
            await self.done(step_id, note)
