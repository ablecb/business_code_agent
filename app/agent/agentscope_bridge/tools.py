"""AgentScope 工具注册：把项目现有只读工具包装成 Toolkit 工具函数。

学习对照（AgentScope 官方教程 task_tool）：
- 工具函数 = 普通 Python 函数，返回 `ToolResponse(content=[TextBlock(...)])`；
- docstring 的 `Args:` 段落会被 Toolkit 自动解析成工具 JSON Schema；
- `register_tool_function(fn, preset_kwargs={...})` 可预置参数（如 project_id），
  预置后该参数不会出现在工具的 JSON Schema 里；
- 也可以直接传 `func_name` / `func_description` / `json_schema` 显式给工具
  起名/描述/参数（本文件即采用这种方式，与 `runtime.registry.EXPLORE_TOOL_SPECS`
  的描述表保持一致 --- 项目里的工具名/描述/参数单一来源）；
- Agent（ReActAgent）拿到 JSON Schema → LLM 生成 ToolUseBlock → Toolkit 执行 →
  ToolResultBlock 回填，全部由框架完成（替代原 explore_agent 手写
  _extract_tool_calls / _absorb_tool_result 循环）。

本项目工具实现（dispatch 逻辑）仍在 app/agent/explore_tools.py 与
app/agent/runtime/registry.py，这里只做薄包装，不复制业务逻辑。
"""
from __future__ import annotations

from typing import Any

from app.agent.runtime.registry import EXPLORE_TOOL_SPECS as _SPECS

try:  # agentscope 可选依赖
    from agentscope.message import TextBlock
    from agentscope.tool import Toolkit, ToolResponse
except Exception:  # pragma: no cover - depends on optional dep
    Toolkit = None  # type: ignore[assignment,misc]
    ToolResponse = None  # type: ignore[assignment,misc]
    TextBlock = None  # type: ignore[assignment,misc]


def _resp(data: dict[str, Any]) -> Any:
    """dict 结果 → ToolResponse（文本化，供 LLM 观察）。"""
    if ToolResponse is None or TextBlock is None:  # pragma: no cover
        raise RuntimeError("agentscope 未安装")
    if data.get("ok") is False:
        return ToolResponse(
            content=[TextBlock(type="text", text=f"错误: {data.get('error') or 'unknown' }")]
        )
    finish_answer = data.get("answer")
    if finish_answer:  # finish 工具：把答案原样返回给 ReAct 循环
        return ToolResponse(content=[TextBlock(type="text", text=str(finish_answer))])
    import json

    from app.core.utils import truncate

    text = truncate(json.dumps(data, ensure_ascii=False), 3200)
    return ToolResponse(content=[TextBlock(type="text", text=text)])


def _make_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """按 EXPLORE_TOOL_SPECS 描述手动构造工具 JSON Schema。"""
    props: dict[str, Any] = {}
    required: list[str] = []
    for arg_name, arg_desc in (spec.get("args") or {}).items():
        props[arg_name] = {
            "type": "string",
            "description": str(arg_desc),
        }
        # 描述里带「可选」或「默认」的字段不强制
        if "可选" not in str(arg_desc) and "默认" not in str(arg_desc):
            required.append(arg_name)
    return {
        "type": "function",
        "function": {
            "name": spec["name"],
            "description": str(spec.get("desc") or ""),
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


def _make(project_id: str, name: str) -> Any:
    """生成执行 dispatch_tool 的工具函数（闭包，名称/描述由注册时显式指定）。"""

    def _tool(**kwargs: Any) -> Any:
        from app.agent.explore_tools import dispatch_tool

        return _resp(dispatch_tool(project_id, name, kwargs))

    return _tool


def build_toolkit(
    project_id: str | None = None,
    *,
    tool_names: list[str] | None = None,
) -> Any:
    """构建 AgentScope Toolkit，注册本项目探索工具。

    :param project_id: 预置到所有工具的项目 id（从 .env 默认取）。
    :param tool_names: 只注册这些工具（None = 注册 EXPLORE_TOOL_SPECS 全部）。
    """
    if Toolkit is None:  # pragma: no cover
        raise RuntimeError("agentscope 未安装")
    from app.core.config import get_settings

    pid = project_id or get_settings().project_id or "local"
    names = tool_names or [str(s.get("name")) for s in _SPECS]
    by_name = {str(s.get("name")): s for s in _SPECS}

    toolkit = Toolkit()
    for name in names:
        spec = by_name.get(name)
        if not spec:
            continue
        # preset 内嵌 project_id：参数不进 schema，调用时自动注入
        toolkit.register_tool_function(
            _make(pid, name),
            preset_kwargs={"project_id": pid},
            func_name=name,
            json_schema=_make_schema(spec),
        )
    return toolkit


def tool_catalog_text() -> str:
    """AgentScope 版工具目录提示文本（供 sys_prompt 使用）。

    只输出工具名+描述（与 EXPLORE_TOOL_SPECS 单一来源保持一致），
    不再包含旧版「输出格式（严格 JSON）」协议段 —— AgentScope 下
    工具 JSON Schema 由 Toolkit 注入给模型，模型直接发起 tool_use 调用，
    不需要手写 JSON 文本协议。
    """
    lines = ["可用工具（需要检索/读码/查调用关系时直接调用；信息足够时调用 finish）："]
    for t in _SPECS:
        lines.append(f"- {t['name']}: {t['desc']} args={t['args']}")
    return "\n".join(lines)