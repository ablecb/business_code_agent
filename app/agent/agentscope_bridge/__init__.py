"""AgentScope 迁移层（学习用）：把项目编排重写为 AgentScope 标准写法。

对照 ACP 课程学习时看这里：
- model.py    —— 模型接入（OpenAIChatModel + LLM_* 配置）
- tools.py    —— 工具注册（Toolkit / @agentscope.tool 风格）
- agents/     —— Agent 实现（AgentBase / ReActAgent / msghub）
- adapter.py  —— run_chat 兼容适配（保持 API 外层签名不变）
- orchestration.py —— 意图路由（显式/隐式 Routing）

原实现（graph_agent.py / explore_agent.py）保留，随时可回退：
- AGENT_RUNTIME_MODE=standard|minimal → 原编排
- AGENT_RUNTIME_MODE=agentscope     → 本迁移层
"""
from __future__ import annotations

from app.agent.agentscope_bridge.model import agentscope_available
from app.agent.agentscope_bridge.adapter import run_chat_scope

__all__ = ["agentscope_available", "run_chat_scope"]