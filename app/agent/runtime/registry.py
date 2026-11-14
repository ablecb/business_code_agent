"""Plugin-style tool catalog for PEV (dispatch still lives in pev_agent).

Skills declare tool lists in skills.SKILL_SPECS; this module is the single place
to list tool metadata and resolve which tools exist for a runtime mode.
"""
from __future__ import annotations

from typing import Any

from app.agent.runtime.profile import RuntimeProfile, get_runtime_profile
from app.agent.runtime.skills import SKILL_SPECS

# name → short description (handlers remain in pev_agent._exec_one_pev_tool).
TOOL_CATALOG: dict[str, dict[str, Any]] = {
    "resolve_entry": {"title": "锁定入口", "layer": "ask"},
    "expand_calls": {"title": "展开调用链", "layer": "ask"},
    "mysql_enrich": {"title": "符号/标签补强", "layer": "ask"},
    "summarize_flow": {"title": "流程摘要", "layer": "ask"},
    "discover_change": {"title": "发现改动点", "layer": "plan"},
    "find_outbound": {"title": "外送触点", "layer": "plan"},
    "render_change": {"title": "渲染改动清单", "layer": "plan"},
    "load_kb_spec": {"title": "加载知识库规格", "layer": "plan"},
    "find_api_style": {"title": "对齐接口风格", "layer": "plan"},
    "render_new_api": {"title": "渲染新接口脚手架", "layer": "plan"},
    "source_view": {"title": "源码定位", "layer": "ask"},
    "compose_code": {"title": "证据生成代码", "layer": "ask"},
    "api_inventory": {"title": "接口清单", "layer": "ask"},
    "repo_files": {"title": "仓库文件检索", "layer": "ask"},
    "codebase_architecture": {"title": "架构概览", "layer": "ask"},
    "symbol_probe": {"title": "符号探测", "layer": "ask"},
    "topic_existence": {"title": "主题存在性", "layer": "ask"},
    "parse_signal": {"title": "解析日志信号", "layer": "ask"},
    "map_and_read": {"title": "映射并读源码", "layer": "ask"},
    "diagnose_answer": {"title": "诊断作答", "layer": "ask"},
    "collect_field_evidence": {"title": "收集字段证据", "layer": "ask"},
    "render_field_answer": {"title": "渲染字段答案", "layer": "ask"},
    "collect_api_io": {"title": "收集入参出参", "layer": "ask"},
    "render_api_io": {"title": "渲染接口 IO", "layer": "ask"},
    "search_hits": {"title": "检索命中", "layer": "ask"},
    "general_answer": {"title": "综合作答", "layer": "ask"},
}


def known_tool(name: str) -> bool:
    return name in TOOL_CATALOG


def list_tools(*, profile: RuntimeProfile | None = None) -> list[dict[str, Any]]:
    prof = profile or get_runtime_profile()
    out: list[dict[str, Any]] = []
    for name, meta in TOOL_CATALOG.items():
        if not prof.tool_allowed(name):
            continue
        out.append({"name": name, **meta})
    return out


def tools_for_skill(skill: str, *, profile: RuntimeProfile | None = None) -> list[str]:
    prof = profile or get_runtime_profile()
    if not prof.skill_allowed(skill):
        return []
    spec = SKILL_SPECS.get(skill) or {}
    tools = [str(t) for t in (spec.get("tools") or []) if t]
    return [t for t in tools if known_tool(t) and prof.tool_allowed(t)]


def filter_task_tools(
    tasks: list[dict[str, Any]],
    *,
    profile: RuntimeProfile | None = None,
) -> list[dict[str, Any]]:
    """Drop / mark tasks whose tool is outside the active runtime profile."""
    prof = profile or get_runtime_profile()
    if prof.allowed_tools is None:
        return list(tasks)
    out: list[dict[str, Any]] = []
    for t in tasks:
        tool = str(t.get("tool") or "")
        if tool and not prof.tool_allowed(tool):
            out.append(
                {
                    **t,
                    "status": "skipped",
                    "note": f"runtime_mode={prof.mode}: tool {tool} not allowed",
                }
            )
            continue
        out.append(t)
    return out


# ---------------------------------------------------------------------------
# Explore-agent tool specs (single source of truth; explore_tools re-exports).
# These are the LLM-facing descriptions used by explore_agent's tool catalog
# prompt — keep them in sync with dispatch_tool handlers in explore_tools.py.
# ---------------------------------------------------------------------------
EXPLORE_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "search_symbols",
        "desc": "按关键词/语义搜索符号（优先找 Controller/Service）。返回简要列表，不含完整代码。",
        "args": {
            "q": "查询词",
            "kind": "可选 Controller|Service|Listener|Mapper|MapperXml|Entity",
            "top_k": "默认8",
        },
    },
    {
        "name": "get_symbol_code",
        "desc": "按 symbol_id 或 fqcn+method_name 读取方法/类的 raw_code（完整源码片段）。",
        "args": {"symbol_id": "优先", "fqcn": "可选", "method_name": "可选"},
    },
    {
        "name": "list_callees",
        "desc": (
            "从调用图查出该方法直接下游（CALLS/ASYNC_CALLS/TRIGGERS/FEIGN_CALLS）。"
            "若指向接口且无边，配合 resolve_impl 找 *ServiceImpl。"
        ),
        "args": {"symbol_id": "方法 symbol id", "limit": "默认15"},
    },
    {
        "name": "list_callers",
        "desc": "查出谁调用了该方法（直接上游）。",
        "args": {"symbol_id": "方法 symbol id", "limit": "默认10"},
    },
    {
        "name": "resolve_impl",
        "desc": (
            "仅用于 XxxService 等接口→*ServiceImpl。"
            "不要对 Controller 调用；本类私有校验方法请直接 get_symbol_code(fqcn, method_name)。"
        ),
        "args": {
            "symbol_id": "可选，接口方法 id",
            "fqcn": "可选，Service 接口 fqcn",
            "method_name": "可选，方法名",
        },
    },
    {
        "name": "discover_add_field_plan",
        "desc": (
            "新增入参字段并落库：结构化发现改动触点（常量/VO/绑定/组装/实体）并渲染改动方案。"
            "仅用于「增加字段+落库/改哪些」类问题；返回完整 Markdown 方案，finish 时应原样采用。"
        ),
        "args": {
            "question": "用户原问（可省略，默认用会话问题）",
            "entry_cm": "可选 Class#method 入口提示",
        },
    },
    {
        "name": "finish",
        "desc": (
            "信息已足够时调用。业务向流程题：优先直接采用入口方法已有业务摘要"
            "（可补少量下游一句话 + 结果回调简述）；禁止强制十步源码主链与 status 摘抄。"
            "技术细节题才用主链1…N+类名。地址/类名题才给技术事实。"
            "禁止 symbol_id 与探索独白。"
        ),
        "args": {"answer": "入口业务摘要(+下游要点)+结果回调；默认少类名"},
    },
]
