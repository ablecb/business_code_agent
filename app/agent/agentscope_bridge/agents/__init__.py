"""AgentScope Agent 实现（迁移层）。

- general_agent.py —— AgentBase 直答（对应 general_chat.py）
- explore_agent.py —— ReActAgent 工具循环（对应 explore_agent.py）
- msghub_team.py —— MsgHub 多智能体协作：探索 + 评审（对应 _finish_with_review）
"""
from __future__ import annotations

from app.agent.agentscope_bridge.agents.general_agent import GeneralAgent
from app.agent.agentscope_bridge.agents.msghub_team import run_explore_team

__all__ = ["GeneralAgent", "run_explore_team"]