"""通用问答 Agent（AgentScope 标准写法）：对应原 app/agent/general_chat.py。

学习对照（AgentScope 官方教程 quickstart_agent「从零开始创建」）：
- 继承 ``AgentBase``，实现三个抽象方法：``reply`` / ``observe`` / ``handle_interrupt``；
- ``reply(msg) -> Msg``：把消息存入 memory → formatter 组装 system+历史 prompt →
  调用 ``self.model(prompt)`` → 生成 ``Msg(name=self.name, content=..., role="assistant")``
  → 记入 memory → ``await self.print(msg)`` 输出；
- 消息统一载体是 ``Msg(name, content, role)``，role ∈ {system, assistant, user}。

差异说明（为什么这里不用 ReActAgent）：通用问答没有工具调用，一次性直答即可，
正好演示最朴素的 AgentBase 写法 —— 这也是课程里第一个自定义 Agent 的标准模板。
"""
from __future__ import annotations

from typing import Any

from app.agent.general_chat import GENERAL_SYSTEM

try:  # agentscope 可选依赖
    from agentscope.agent import AgentBase
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.message import Msg
except Exception:  # pragma: no cover - depends on optional dep
    AgentBase = object  # type: ignore[assignment,misc]
    OpenAIChatFormatter = None  # type: ignore[assignment,misc]
    InMemoryMemory = None  # type: ignore[assignment,misc]
    Msg = None  # type: ignore[assignment,misc]


class GeneralAgent(AgentBase):  # type: ignore[misc]
    """通用问答 Agent：无工具、一次性直答（对应 general_chat_reply）。"""

    def __init__(self, model: Any, *, name: str = "GeneralAgent") -> None:
        super().__init__()
        self.name = name
        self.sys_prompt = GENERAL_SYSTEM
        self.model = model
        self.formatter = OpenAIChatFormatter()
        self.memory = InMemoryMemory()

    async def reply(self, msg: Msg | list[Msg] | None) -> Msg:  # type: ignore[name-defined]
        """直接把用户问题交给 LLM 回答（系统提示 + 历史，无工具）。"""
        await self.memory.add(msg)
        prompt = await self.formatter.format(
            [
                Msg("system", self.sys_prompt, "system"),
                *await self.memory.get_memory(),
            ]
        )
        response = await self.model(prompt)
        out = Msg(name=self.name, content=response.content, role="assistant")
        await self.memory.add(out)
        await self.print(out)
        return out

    async def observe(self, msg: Msg | list[Msg] | None) -> None:  # type: ignore[name-defined]
        # msghub 中观察其他智能体的消息（本 Agent 单聊时通常无观察对象）
        await self.memory.add(msg)

    async def handle_interrupt(self) -> Msg:  # type: ignore[name-defined]
        return Msg(name=self.name, content="我注意到您打断了我的回复，我能为你做些什么？", role="assistant")


async def general_chat_reply_scope(message: str, history_text: str = "") -> str:
    """AgentScope 版通用对话：返回纯文本回答（对应 general_chat.general_chat_reply）。"""
    from app.agent.agentscope_bridge.model import get_chat_model, get_text

    history_part = f"\n历史对话:\n{history_text or '(空)'}\n\n" if history_text else "\n"
    agent = GeneralAgent(get_chat_model())
    reply = await agent(Msg("user", f"{history_part}用户问题:\n{message}", "user"))
    return get_text(reply)