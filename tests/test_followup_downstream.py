"""Follow-up / content probes are structural-only; LLM session gate decides lock."""

from __future__ import annotations

import asyncio

from app.agent.graph_agent import (
    _is_followup,
    _question_has_content_noun,
    resolve_session_ctx_gate,
)


def test_keyword_followup_detector_disabled():
    """No CN keyword follow-up gates in production code."""
    assert not _is_followup("回答过程中都调用了哪些接口")
    assert not _is_followup("那机构是怎么校验的")
    assert not _is_followup("放款掉了哪些接口")


def test_content_noun_is_structural_not_stopword_list():
    """Any CJK/Latin token counts; no business stopword stripping."""
    assert _question_has_content_noun("放款掉了哪些接口")
    assert _question_has_content_noun("调用了哪些接口")
    assert _question_has_content_noun("loan")


def test_downstream_followup_via_llm(monkeypatch):
    """Deictic downstream under lock → LLM related keeps session."""
    from app.llm.base import LLMProvider

    class _Related(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":true,"continue":true,"dig_in":true,'
                '"ask_kind":"followup","reason":"deictic under same entry"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _Related()
    )
    ctx = {
        "last_class_method": "WithdrawalsController#withdrawals",
        "last_flow_symbols": ["WithdrawalsServiceImpl#loan"],
        "last_explore_question": "放款接口什么流程",
    }
    gate = asyncio.run(
        resolve_session_ctx_gate("回答过程中都调用了哪些接口", ctx, session_id="")
    )
    assert gate["use_session_ctx"] is True
    assert gate["new_topic"] is False
