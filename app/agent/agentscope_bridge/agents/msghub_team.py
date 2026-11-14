"""多智能体协作样板（MsgHub）：探索 Agent + 评审 Agent。

学习对照（ACP 课程 3_3「多 Agent 团队协作」 + AgentScope 官方教程 workflow_conversation）：
- ``MsgHub`` 是异步上下文管理器：同一 hub 内的参与者通过 ``reply`` 返回的消息
  会自动广播/路由给其他参与者（无需手动转发），``announcement`` 是入场公告；
- 参与者用 ``ReActAgent``（课程 3_1 同构写法），多智能体场景的 formatter 用
  ``OpenAIMultiAgentFormatter``（课程用 DashScopeMultiAgentFormatter，协议对应）；
- 原 explore_agent 的评审循环（_finish_with_review，最多 3 轮打回深挖）在这里
  变成「探索 Agent 出草稿 → 评审 Agent 给意见 → 探索 Agent 观察意见后修订/收尾」
  的消息协作，不需要手写轮次控制 —— 每轮 `await` 一个参与者即一轮发言。

代码结构（对照课程 3_3 原文）：
    create_expert_agent(name, sys_prompt) -> ReActAgent   <- 课程 3_3 的工厂函数
    async with MsgHub(participants=[...], announcement=Msg(...)) as hub:
        await explorer()      # 出草稿
        await reviewer()      # 评审
        await explorer()      # 修订（observe 到评审意见后自动改进）
"""
from __future__ import annotations

from typing import Any

try:  # agentscope 可选依赖
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIMultiAgentFormatter
    from agentscope.pipeline import MsgHub
except Exception:  # pragma: no cover - depends on optional dep
    ReActAgent = object  # type: ignore[assignment,misc]
    OpenAIMultiAgentFormatter = None  # type: ignore[assignment,misc]
    MsgHub = None  # type: ignore[assignment,misc]

_OPENAI_MULTI_FORMATTER = None


def _multi_formatter() -> Any:
    """多智能体 formatter（懒加载创建，1.x 名称 OpenAIMultiAgentFormatter）。"""
    global _OPENAI_MULTI_FORMATTER
    if _OPENAI_MULTI_FORMATTER is None:
        from agentscope.formatter import OpenAIMultiAgentFormatter

        _OPENAI_MULTI_FORMATTER = OpenAIMultiAgentFormatter()
    return _OPENAI_MULTI_FORMATTER


def create_expert_agent(
    name: str,
    sys_prompt: str,
    *,
    toolkit: Any | None = None,
    max_iters: int = 10,
) -> Any:
    """创建参与者 ReActAgent（对应课程 3_3 的 create_expert_agent 工厂）。

    :param toolkit: 探索/检索工具（评审 Agent 不传 = 纯文本评审）。
    """
    from app.agent.agentscope_bridge.model import get_chat_model

    return ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=get_chat_model(),
        formatter=_multi_formatter(),
        toolkit=toolkit,
        max_iters=max_iters,
    )


_REVIEWER_SYSTEM = (
    "你是代码问答的评审员。你收到探索 Agent 的探索过程与回答草稿，请只输出一行评审结论：\n"
    "1) 若探索 Agent 已通过 finish 提交最终答案，且回答有代码依据、结构清晰、"
    "直接回答了用户问题 → 输出：APPROVED\n"
    "2) 若探索 Agent 尚未提交最终答案（只有思考和工具调用）→ 输出：REVISE + 一句"
    "「请调用 finish 提交最终答案」\n"
    "3) 若答案缺关键证据（没读代码/没查调用链/凭猜）→ 输出：REVISE + 一句具体修改要求\n"
    "不要输出其他内容。"
)


async def run_explore_team(
    question: str,
    *,
    project_id: str | None = None,
    max_iters: int = 12,
    rounds: int = 2,
    tool_names: list[str] | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """MsgHub 多智能体探索：探索 Agent 草稿 → 评审 Agent 意见 → 修订收尾。

    :param rounds: hub 内轮流发言的轮数（默认 2 轮：探索→评审→修订→评审）。
    :return: {"answer": 最终回答, "review": 最后评审意见, "transcript": 全部发言摘要}
    """
    from agentscope.message import Msg

    from app.agent.agentscope_bridge.agents.explore_agent import _explore_system_prompt
    from app.agent.agentscope_bridge.model import get_text
    from app.agent.agentscope_bridge.tools import build_toolkit

    explorer = create_expert_agent(
        "ExploreAgent",
        _explore_system_prompt(),
        toolkit=build_toolkit(project_id=project_id, tool_names=tool_names),
        max_iters=max_iters,
    )
    reviewer = create_expert_agent("ReviewerAgent", _REVIEWER_SYSTEM)

    if on_progress:
        try:
            maybe = on_progress(
                {
                    "type": "step",
                    "status": "running",
                    "step_id": "msghub",
                    "note": "MsgHub 多智能体协作：探索 → 评审 → 修订…",
                }
            )
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            pass

    async with MsgHub(
        participants=[explorer, reviewer],
        announcement=Msg("system", f"协作目标：回答用户问题。\n用户问题：{question}", "system"),
    ):
        for _ in range(max(1, rounds)):
            await explorer()  # 草稿 / 修订稿
            await reviewer()  # 评审意见（会自动广播给 explorer）

    # 取最终回答：探索 Agent 的 memory 里最后一条 ExploreAgent 发言。
    # 注意 Msg 的 get_text_content() 只取 text 块；若最后一条只有 thinking（
    # 模型思考后直接收尾），则回退到该消息的完整 JSON 内容，避免回答为空。
    answer = ""
    review = ""
    transcript: list[str] = []
    explorer_msgs = [m for m in await explorer.memory.get_memory() if getattr(m, "name", "") == "ExploreAgent"]
    for msg in explorer_msgs:
        text = get_text(msg)
        if text:
            answer = text
    if not answer and explorer_msgs:
        import json

        last = explorer_msgs[-1]
        text = json.dumps(last.content, ensure_ascii=False)[:4000]
        # 多智能体对话下模型常只输出 thinking：回退到思考内容（保留检索依据），
        # 并注明这是「思考轨迹」，避免回答为空（学习笔记：可后续让探索 Agent\n        # 以结构化输出强制 finish 提交答案）。
        answer = f"（探索思考轨迹）{text}"
    for msg in await explorer.memory.get_memory():
        text = get_text(msg)
        if not text:
            continue
        transcript.append(f"{msg.name}: {text[:200]}")
        if msg.name == "ReviewerAgent":
            review = text  # 取最后一条评审意见（REVISE → APPROVED 循环的终态）

    if on_progress:
        try:
            maybe = on_progress(
                {
                    "type": "step",
                    "status": "done",
                    "step_id": "msghub",
                    "note": "MsgHub 协作完成" + (f"（评审：{review[:40]}）" if review else ""),
                }
            )
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            pass

    return {
        "answer": answer,
        "review": review,
        "transcript": transcript,
        "steps": [],
        "retrieved": [],
        "plan": {"kind": "msghub_team", "title": "MsgHub 探索+评审", "steps": []},
    }