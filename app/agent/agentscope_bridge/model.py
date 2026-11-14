"""AgentScope 模型接入层：把项目现有 LLM_* 配置映射成 AgentScope 模型对象。

学习对照（AgentScope 官方教程 quickstart_agent / task_model）：
- 模型是对象：`model(messages, tools, tool_choice)` 统一接口，返回 `ChatResponse`。
- OpenAI 兼容 API（本项目 siliconflow / deepseek / ollama 全是 OpenAI 兼容形态）
  用 `OpenAIChatModel`，通过 `client_kwargs={"base_url": ...}` 指定端点，
  `generate_kwargs` 设定 temperature / max_tokens。
- **不替换** `app/llm/` 现有层：解析管线（ingest/tagging）仍走原有 LLMProvider，
  这里只给 AgentScope 编排层提供模型。
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import get_settings

# agentscope 是迁移层必需的依赖；原项目 requirements.txt 未包含，安装后此处可导入。
# 若未安装，保持原有实现可运行（agentscope_bridge 仅在 agent_runtime_mode=agentscope 时启用）。
_AGENTSCOPE_AVAILABLE = False
try:
    from agentscope.model import OpenAIChatModel  # noqa: F401  (v1.x 标准导入)

    _AGENTSCOPE_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional dep
    _AGENTSCOPE_AVAILABLE = False


def agentscope_available() -> bool:
    """True = agentscope 已安装，可启用 agentscope 编排模式。"""
    return _AGENTSCOPE_AVAILABLE


def build_chat_model(**overrides: Any) -> Any:
    """按项目 `.env` 的 LLM_* 配置构建 AgentScope 聊天模型。

    对应官方用法：
        OpenAIChatModel(
            model_name="Qwen/Qwen3-32b",
            api_key="sk-...",
            client_kwargs={"base_url": "https://api.siliconflow.cn/v1"},
            generate_kwargs={"temperature": 0.2, "max_tokens": 4096},
        )
    """
    if not _AGENTSCOPE_AVAILABLE:
        raise RuntimeError(
            "agentscope 未安装：请先 `pip install agentscope`，"
            "或在 .env 关闭 AGENT_RUNTIME_MODE=agentscope"
        )
    settings = get_settings()
    provider = (settings.llm_provider or "siliconflow").lower().strip()

    model_name = overrides.pop("model_name", None) or settings.llm_model
    api_key = overrides.pop("api_key", None) or settings.llm_api_key or "EMPTY"
    base_url = overrides.pop("base_url", None) or settings.llm_base_url
    temperature = overrides.pop("temperature", None) or settings.llm_temperature or 0.2
    max_tokens = overrides.pop("max_tokens", None) or settings.llm_max_tokens or 4096

    # ollama 特殊：agentscope 也有 OllamaChatModel，但统一走 OpenAI 兼容更简单
    # （ollama 提供 /v1/chat/completions）。
    kwargs: dict[str, Any] = {
        "model_name": model_name,
        "api_key": api_key,
        # OpenAIChatModel 默认 stream=True；项目现有 chat() 均为非流式，保持一致。
        "stream": False,
        "client_kwargs": {"base_url": base_url},
        "generate_kwargs": {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        },
    }
    kwargs.update(overrides)
    return OpenAIChatModel(**kwargs)


@lru_cache(maxsize=4)
def get_chat_model() -> Any:
    """进程级缓存的默认聊天模型（与 app.llm.factory.get_llm_provider 对齐）。"""
    return build_chat_model()


def get_text(response: Any) -> str:
    """从 AgentScope 响应（ChatResponse dict / Msg / ChatResponse 对象）取纯文本。

    - ChatResponse 是 dict 子类：内容在 `response["content"]`（block 列表，
      每块 {type: text|thinking|tool_use|tool_result, ...}）。
    - Msg（ReActAgent 的返回值）有 `get_text_content()`。
    """
    if response is None:
        return ""
    # ChatResponse 是 dict 子类且重载了 __getattr__（属性→键转发），
    # 所以必须先 isinstance 判断再 getattr，否则 getattr 会触发 KeyError。
    if isinstance(response, dict):
        content = response.get("content")
    else:
        fn = getattr(response, "get_text_content", None)
        if callable(fn):
            try:
                return str(fn() or "")
            except Exception:
                pass
        content = getattr(response, "content", None)
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(b.get("text") or "") for b in content if isinstance(b, dict) and b.get("text")
        ]
        joined = "\n".join(p for p in parts if p).strip()
        if joined:
            return joined
        # 无 text block：回退到最后一个 tool_use / 原始 JSON
        for b in reversed(content):
            if isinstance(b, dict) and b.get("type") == "tool_use":
                import json

                return json.dumps(b, ensure_ascii=False)
        return ""
    return str(content).strip()