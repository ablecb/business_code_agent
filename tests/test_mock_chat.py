from __future__ import annotations

import asyncio
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"


def test_chat_ask_flow():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    resp = asyncio.run(run_chat("订单流程什么样"))
    assert resp.session_id
    assert resp.reply
    assert resp.intent in {
        "ask",
        "plan",
        "modify",
        "execute",
        "meta",
    }
    assert isinstance(resp.scope, list)
    assert resp.phase in {
        None,
        "idle",
        "analyzing",
        "awaiting_option",
        "implementing",
        "verifying",
        "repairing",
        "done",
        "failed",
    }


def test_meta_recalls_previous_question():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    first = asyncio.run(run_chat("订单流程什么样"))
    second = asyncio.run(
        run_chat("我刚才问你什么了", session_id=first.session_id)
    )
    assert second.intent == "meta"
    assert "订单流程什么样" in second.reply
    assert "修改范围" not in second.reply
    assert "Spring Cloud" not in second.reply


def test_plan_options_intent_sets_awaiting_phase():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    resp = asyncio.run(
        run_chat("给我三种实现方案对比利弊，放款接口加字段")
    )
    assert resp.intent == "plan"
    assert resp.phase == "awaiting_option"
    assert isinstance(resp.options, list)
    assert len(resp.options) >= 2
    assert all(o.get("id") for o in resp.options)


def test_select_option_action_enters_modify():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    first = asyncio.run(
        run_chat("给我两种实现方案对比利弊")
    )
    assert first.phase == "awaiting_option"
    oid = first.options[0]["id"]
    second = asyncio.run(
        run_chat(
            f"选方案{oid}",
            session_id=first.session_id,
            action="select_option",
            option_id=oid,
        )
    )
    assert second.intent == "modify"
    assert second.selected_option_id == oid
