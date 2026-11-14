"""未选择项目 → 普通 AI 问答（带会话上下文记忆）；同一会话可混合项目问答与普通问答。

行为契约：
- 显式传 project_ids=[]（桌面端未勾选任何项目）→ 普通问答，不回退默认项目；
- 同一 session_id 内可来回切换「选中项目问答」与「未选项目普通问答」，历史不断；
- 桌面端 client_agent 开启时，OS/本机类需求（如 C 盘清理）仍走本机执行闭环。
"""
from __future__ import annotations

import asyncio
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"


def _reset():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    session_store._MEMORY_SESSIONS.clear()
    session_store._MEMORY_MESSAGES.clear()
    session_store._MEMORY_FEEDBACK.clear()
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()


def test_no_project_plain_chat_has_memory():
    from app.agent.graph_agent import run_chat

    _reset()
    first = asyncio.run(run_chat("什么是数据库事务", project_ids=[]))
    assert first.project_ids == []
    assert first.intent == "ask"
    assert "什么是数据库事务" in (first.reply or "")

    # 同一 session 第二轮：上下文记忆应包含第一轮的提问
    second = asyncio.run(
        run_chat("再展开讲一下", session_id=first.session_id, project_ids=[])
    )
    assert second.session_id == first.session_id
    assert second.project_ids == []
    assert second.intent == "ask"
    assert "什么是数据库事务" in (second.reply or "")
    assert "历史对话" in (second.reply or "")


def test_no_project_does_not_fallback_default_project():
    from app.agent.graph_agent import run_chat

    _reset()
    resp = asyncio.run(run_chat("写一首关于秋天的诗", project_ids=[]))
    assert resp.project_ids == []
    assert resp.reply


def test_same_session_mix_project_and_plain():
    from app.agent.graph_agent import run_chat

    _reset()
    # 选中项目 → 项目问答
    code = asyncio.run(run_chat("订单流程什么样", project_id="p1", project_ids=["p1"]))
    assert code.project_ids == ["p1"]
    assert code.reply

    # 同一会话取消勾选 → 普通问答，且记得上一轮的提问
    plain = asyncio.run(
        run_chat("你刚才提到了什么", session_id=code.session_id, project_ids=[])
    )
    assert plain.session_id == code.session_id
    assert plain.project_ids == []
    assert plain.intent == "ask"
    assert "订单流程什么样" in (plain.reply or "")

    # 再勾选回去 → 又是项目问答，会话不变
    again = asyncio.run(
        run_chat("放款接口的校验逻辑是什么", session_id=code.session_id, project_id="p1", project_ids=["p1"])
    )
    assert again.session_id == code.session_id
    assert again.project_ids == ["p1"]


def test_no_project_os_assist_still_direct():
    from app.agent.graph_agent import run_chat

    _reset()
    resp = asyncio.run(
        run_chat("C盘满了，帮我看看怎么清理", client_agent=True, project_ids=[])
    )
    assert resp.project_ids == []
    assert resp.intent == "client_agent"
    assert resp.phase == "client_running"
    assert resp.client_run_id


def test_http_explicit_empty_project_ids_plain_chat():
    """HTTP 层：显式传 project_ids=[] 与「未传」区分开（走普通问答，不回退默认项目）。"""
    from fastapi.testclient import TestClient

    from app.main import app

    _reset()
    client = TestClient(app)
    # 显式空数组 → 普通问答
    resp = client.post(
        "/api/v1/chat",
        json={"message": "什么是数据库事务", "project_ids": []},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["project_ids"] == []
    assert body["intent"] == "ask"
    assert body["reply"]