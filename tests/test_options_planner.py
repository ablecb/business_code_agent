from __future__ import annotations

import asyncio
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"


def test_build_change_options_returns_ab():
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.options_planner import build_change_options, render_options_reply

    opts = asyncio.run(
        build_change_options(
            question="放款接口增加字段",
            explore_answer="入口 WithdrawalsController#withdrawals",
            scope=[{"file": "a.java", "action": "add_field", "reason": "VO"}],
            count=3,
        )
    )
    assert len(opts) >= 2
    assert opts[0]["id"]
    text = render_options_reply(opts, question="放款接口增加字段")
    assert "可选实现方案" in text
    assert "方案" in text
