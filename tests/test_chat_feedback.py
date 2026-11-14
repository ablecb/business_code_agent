from __future__ import annotations

import asyncio
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"


def test_feedback_upsert_and_list():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    session_store._MEMORY_SESSIONS.clear()
    session_store._MEMORY_MESSAGES.clear()
    session_store._MEMORY_FEEDBACK.clear()
    session_store._MEMORY_FEEDBACK_SEQ = 0
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    resp = asyncio.run(run_chat("订单流程什么样"))
    assert resp.assistant_message_id is not None

    row = session_store.upsert_feedback(
        session_id=resp.session_id,
        assistant_message_id=resp.assistant_message_id,
        vote="up",
        project_id="demo",
    )
    assert row["vote"] == "up"
    assert row["updated"] is False
    assert row["id"]

    row2 = session_store.upsert_feedback(
        session_id=resp.session_id,
        assistant_message_id=resp.assistant_message_id,
        vote="down",
    )
    assert row2["vote"] == "down"
    assert row2["updated"] is True
    assert row2["id"] == row["id"]

    msgs = session_store.list_messages(resp.session_id)
    assistant = next(m for m in msgs if m["id"] == resp.assistant_message_id)
    assert (assistant.get("meta") or {}).get("vote") == "down"

    items, total = session_store.list_feedback(session_id=resp.session_id)
    assert total == 1
    assert items[0]["vote"] == "down"
    assert items[0]["question_excerpt"]
    assert items[0]["reply_excerpt"]


def test_chat_response_includes_message_id():
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    resp = asyncio.run(run_chat("你好"))
    assert isinstance(resp.assistant_message_id, int)
    assert resp.assistant_message_id > 0
