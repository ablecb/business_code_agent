"""探索 Agent（AgentScope ReActAgent 标准写法）：对应原 app/agent/explore_agent.py。

学习对照（AgentScope 官方教程 quickstart_agent「创建 ReAct 智能体」/ task_tool）：
- ``ReActAgent`` 是开箱即用的思考-行动循环：LLM 每轮生成 ``ToolUseBlock``
  （或直接文本回答）→ Toolkit 执行 → ``ToolResultBlock`` 回填 → 继续，直到
  不再调用工具；``max_iters`` 限制最大轮数。
- 工具注册进 ``Toolkit`` 后自动生成 JSON Schema 给 LLM（无需手写解析/执行，
  替代原 explore_agent 手写的 ``_extract_tool_calls`` / ``_absorb_tool_result``
  与评审循环）。
- ``finish`` 工具原样返回答案文本，ReActAgent 收到后结束循环并输出最终 ``Msg``。

差异说明：
- 原实现做了大量离线质量增强/评审（深挖关卡、证据链校验等）；迁移骨架先
  保持 ReActAgent 原生循环 + 现有工具 + 现有 explore_system prompt，行为细节
  不追求 1:1（按用户要求暂不评测）。
"""
from __future__ import annotations

from typing import Any

try:  # agentscope 可选依赖
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.message import Msg
except Exception:  # pragma: no cover - depends on optional dep
    ReActAgent = object  # type: ignore[assignment,misc]
    OpenAIChatFormatter = None  # type: ignore[assignment,misc]
    InMemoryMemory = None  # type: ignore[assignment,misc]
    Msg = None  # type: ignore[assignment,misc]

# 与 explore_agent 相同的步数上限（按问题类型可放宽，骨架统一用默认）
_DEFAULT_MAX_STEPS = 12


def build_explore_agent(
    *,
    project_id: str | None = None,
    max_iters: int = _DEFAULT_MAX_STEPS,
    tool_names: list[str] | None = None,
) -> Any:
    """构建 ReActAgent（检索/读码/查调用图工具循环）。

    :param project_id: 预置给所有工具的项目 id（默认取 .env）。
    :param max_iters: ReAct 最大迭代轮数（对应原 explore_agent._DEFAULT_MAX_STEPS）。
    :param tool_names: 只注册这些工具；None = EXPLORE_TOOL_SPECS 全部。
    """
    from app.agent.agentscope_bridge.model import get_chat_model
    from app.agent.agentscope_bridge.tools import build_toolkit

    sys_prompt = _explore_system_prompt()
    return ReActAgent(
        name="ExploreAgent",
        sys_prompt=sys_prompt,
        model=get_chat_model(),
        formatter=OpenAIChatFormatter(),
        toolkit=build_toolkit(project_id=project_id, tool_names=tool_names),
        memory=InMemoryMemory(),
        max_iters=max_iters,
    )


def _explore_system_prompt() -> str:
    """AgentScope 版系统提示：复用现有 explore_system 主提示，但去掉手写 JSON 协议。

    差异说明（学习对照）：原 explore_agent 让模型输出
    `{"tool": "...", "args": {...}}` JSON 文本，由自研代码解析执行；
    AgentScope 的 ReActAgent 原生支持工具调用（tool_use），模型看到
    Toolkit 的 JSON Schema 后直接发起调用，框架负责执行与回填。
    因此这里不再注入「输出格式（严格 JSON）」指令，改为说明工具调用语义。
    """
    from app.agent.agentscope_bridge.tools import tool_catalog_text
    from app.agent.prompts import load_prompt

    base = load_prompt("explore_system")
    # 去掉旧协议里的输出格式段（「输出格式（严格 JSON…」起至结尾）
    marker = "输出格式（严格 JSON"
    if marker in base:
        base = base.split(marker)[0].rstrip()
    return (
        base
        + "\n\n"
        + "工具使用说明（AgentScope 框架自动执行工具调用）：\n"
        + "1) 需要检索/读代码/查调用关系时，直接调用对应的工具函数，一次可调用 1~3 个；\n"
        + "2) 信息足够时，**必须**调用 finish 工具并传入 answer 参数提交最终回答 —— "
        "不要只思考不输出，不要直接用对话文本回答；\n"
        + "3) 工具返回错误时不要编造数据，换关键词或换工具重试，仍无法获取则如实说明。\n\n"
        + tool_catalog_text()
    )


async def run_explore_agent_scope(
    question: str,
    *,
    project_id: str | None = None,
    max_steps: int | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """AgentScope 版探索：返回 {answer, steps, plan, retrieved}。

    兼容原 run_explore_agent 的返回形态（steps/plan/retrieved 现有调用方
    可能读取；骨架先尽力填充，不完整处标注）。
    """
    from app.agent.agentscope_bridge.model import get_text

    agent = build_explore_agent(project_id=project_id, max_iters=max_steps or _DEFAULT_MAX_STEPS)
    if on_progress:
        try:
            maybe = on_progress(
                {
                    "type": "step",
                    "status": "running",
                    "step_id": "explore",
                    "note": "AgentScope ReActAgent 探索中…",
                }
            )
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            pass
    reply = await agent(Msg("user", question, "user"))
    answer = get_text(reply)
    return {
        "answer": answer,
        "steps": [],  # AgentScope 骨架：工具轨迹未逐条展开（可挂 middleware/hook 补齐）
        "retrieved": [],
        "plan": {"kind": "explore", "title": "AgentScope ReAct 探索", "steps": []},
    }