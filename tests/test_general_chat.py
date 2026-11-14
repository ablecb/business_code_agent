"""Tests for the general-chat fallback (回答所有问题：非代码域直答)."""
from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"

import asyncio

from app.agent.general_chat import (
    GENERAL_SYSTEM,
    general_chat_reply,
    is_general_chat_question,
)
from app.llm.mock import MockLLMProvider


def test_general_question_true():
    assert is_general_chat_question("帮我解释一下什么是多线程") is True
    assert is_general_chat_question("写一首关于秋天的诗") is True
    assert is_general_chat_question("今天天气怎么样") is True
    assert is_general_chat_question("") is True
    # 回归：通用计算机概念不得被代码锚点劫持
    assert is_general_chat_question("什么是数据库事务") is True
    assert is_general_chat_question("什么是线程锁") is True
    assert is_general_chat_question("什么是消息队列") is True
    # 回归：用某语言写代码/算法 → 通用直答
    assert is_general_chat_question("用 Java 写一个快速排序") is True
    assert is_general_chat_question("写一个 Python 爬虫") is True


def test_code_question_false():
    # 领域关键词
    assert is_general_chat_question("放款流程什么样") is False
    assert is_general_chat_question("这个接口的校验逻辑是什么") is False
    assert is_general_chat_question("帮我加一个字段并落库") is False
    # 代码标识符
    assert is_general_chat_question("WithdrawalsController#withdrawals 是做什么的") is False
    assert is_general_chat_question("com.demo.loan.controller.LoanController 在哪") is False
    assert is_general_chat_question("看下 LoanService.java 的代码") is False
    # 项目代码生成仍走代码域（写 Controller / 写接口）
    assert is_general_chat_question("帮我写一个 LoanController") is False
    assert is_general_chat_question("帮我写一个新增接口") is False


def test_wants_verify_action():
    from app.agent.general_chat import wants_verify_action

    assert wants_verify_action("把改的代码编译一下") is True
    assert wants_verify_action("跑一下测试看看") is True
    assert wants_verify_action("验证一下这个方案") is True
    assert wants_verify_action("用 Java 写一个快速排序") is False
    assert wants_verify_action("放款流程什么样") is False


def test_general_chat_reply_returns_string():
    async def run():
        return await general_chat_reply("解释一下多线程", "", MockLLMProvider())

    reply = asyncio.run(run())
    assert isinstance(reply, str)
    assert reply


def test_general_system_prompt_present():
    assert "中文" in GENERAL_SYSTEM
    assert "禁止编造" in GENERAL_SYSTEM


def test_dispatch_code_domain_not_short_circuited_by_general_capability(monkeypatch):
    """代码域问法被意图模型误判成 capability=general 时不得短路进闲聊兜底。

    回归：desktop-chat-997ffe53135d —— 「根据知识库的商品查询文档生成 request/
    response 实体类」被判成 intent=execute + capability=general，execute 因无
    编译/验证词降级 ask 后，capability=general 单独就把它短路进 general_chat_reply，
    7.5 分钟只回「这需要查询项目代码，可以换个更具体的问法」，知识库→实体类管线
    根本没跑。修复后：代码域锚点复查把 capability 纠正为 code_qa，交回 _explore。
    """
    import asyncio

    from app.agent import session_store
    from app.agent.graph_agent import _dispatch_by_intent
    from app.llm.factory import get_llm_provider
    from unittest.mock import AsyncMock

    get_llm_provider.cache_clear()
    session_store._USE_MEMORY = True
    session_store._MEMORY_SESSIONS.clear()
    session_store._MEMORY_MESSAGES.clear()

    def fake_try(project_id, question, *, min_score=None):
        del project_id, question, min_score
        return None  # 黄金库未命中

    monkeypatch.setattr("app.analysis.golden_qa.try_golden_answer", fake_try)

    general_reply = AsyncMock(return_value="闲聊兜底回答")
    explore = AsyncMock(
        return_value={
            "intent": "ask",
            "capability": "code_qa",
            "message": "根据我知识库的商品查询文档，生成对应的request和response实体类",
            "reply": "## 请求/响应实体类\n（知识库 add-api 管线产出）",
            "phase": "idle",
            "options": [],
        }
    )
    monkeypatch.setattr("app.agent.graph_agent.general_chat_reply", general_reply)
    monkeypatch.setattr("app.agent.graph_agent._explore", explore)

    state = {
        "message": "根据我知识库的商品查询文档，生成对应的request和response实体类",
        "project_id": "demo",
        "project_ids": ["demo"],
        "session_id": "sess-gen-cap",
        "intent": "execute",
        "capability": "general",
        "want_apply": False,
        "want_verify": False,
        "need_options": False,
        "detail": "根据知识库中的商品查询文档，生成对应的 request 和 response 实体类。",
        "keywords": ["商品查询", "request", "response", "实体类"],
        "phase": "idle",
        "options": [],
    }
    out = asyncio.run(_dispatch_by_intent(state))

    general_reply.assert_not_awaited()  # 严禁被 general 兜底抢答
    explore.assert_awaited_once()  # 必须交回代码问答/PEV 管线
    assert out.get("capability") == "code_qa"
    assert out.get("intent") == "ask"
    assert "实体类" in (out.get("reply") or "")


def test_dispatch_general_question_direct_answer(monkeypatch):
    """非代码域问题 → 不进 explore 工具循环，直接由 LLM 回答。"""
    import asyncio

    from app.agent import session_store
    from app.agent.graph_agent import _dispatch_by_intent
    from app.llm.factory import get_llm_provider

    get_llm_provider.cache_clear()
    session_store._USE_MEMORY = True
    session_store._MEMORY_SESSIONS.clear()
    session_store._MEMORY_MESSAGES.clear()

    def fake_try(project_id, question, *, min_score=None):
        del project_id, question, min_score
        return None  # 黄金库未命中

    monkeypatch.setattr("app.analysis.golden_qa.try_golden_answer", fake_try)

    state = {
        "message": "帮我写一首关于秋天的诗",
        "project_id": "p1",
        "project_ids": ["p1"],
        "session_id": "sess-gen",
        "intent": "ask",
        "phase": "idle",
    }
    out = asyncio.run(_dispatch_by_intent(state))
    assert isinstance(out.get("reply"), str)
    assert out["reply"]
    assert out["phase"] == "idle"
