from __future__ import annotations

import asyncio
import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"


def _reset_env() -> None:
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


# ---------------------------------------------------------------------------
# Unit: 纯重问句识别（desktop-chat-cb2aaea937da 的 bug 形态）
# ---------------------------------------------------------------------------


def test_is_reanswer_request_accepts_bare_reasks():
    from app.agent.graph_agent import _is_reanswer_request

    for text in (
        "现在再回答一下",
        "再回答一下",
        "再回答一遍",
        "重新回答一下",
        "重新回答",
        "回答一下",
        "请再回答一下",
        "回答这个问题",
        "重答一下",
        "现在重新回答",
    ):
        assert _is_reanswer_request(text), text


def test_is_reanswer_request_rejects_meta_recall_and_new_questions():
    from app.agent.graph_agent import _is_reanswer_request

    for text in (
        "我刚才问你什么了",  # meta recall (taxonomy gate first)
        "再说一遍",  # meta recall
        "你听到了什么",  # meta recall
        "回答一下放款流程",  # new code question — has 「流程」 anchor
        "回答一下SignatureController的方法",  # new code question
        "什么是多线程",  # fresh general question
        "[CORRECTION] 上一答不对",  # protocol marker → correction path
    ):
        assert not _is_reanswer_request(text), text


def test_is_reanswer_request_accepts_english_reasks():
    """English re-answers must resolve the same way (language-agnostic guard)."""
    from app.agent.graph_agent import _is_reanswer_request

    for text in (
        "answer again",
        "please answer it again",
        "answer it",
        "answer this",
        "answer that",
        "answer me",
        "re-answer",
        "reanswer",
        "now answer",
        "give me the answer",
    ):
        assert _is_reanswer_request(text), text

    # English 再说一遍/重复一下 equivalents are META (conversation recall, same
    # as the CN taxonomy: 刚才问/再说一遍/重复一下 → meta), so they are NOT re-asks.
    # Bare verbs / fragments and real content stay OUT too.
    for text in (
        "say it again",  # == 再说一遍 → meta, not re-ask
        "repeat it",  # == 重复一下 → meta, not re-ask
        "repeat that",  # meta recall (English)
        "answer",  # bare verb — input fragment
        "again",  # fragment
        "repeat",  # ambiguous bare verb
        "what did I ask",  # meta recall (English)
        "say it again louder",  # not a pure re-ask
        "answer the apply flow",  # new code question
    ):
        assert not _is_reanswer_request(text), text


# ---------------------------------------------------------------------------
# 端到端：选择项目后「现在再回答一下」应重跑上一问的代码管线
# ---------------------------------------------------------------------------


def test_reanswer_reruns_previous_code_question_after_selecting_project():
    _reset_env()

    from app.agent.graph_agent import run_chat

    # Turn 1: 用户问「结清证明申请接口都有哪些」并选中 demo-loan
    first = asyncio.run(
        run_chat(
            "结清证明申请接口都有哪些",
            project_id="demo-loan",
            project_ids=["demo-loan"],
        )
    )
    assert first.session_id
    assert first.intent == "ask"

    # Turn 2: 未带新代码内容，仅重问上一问 → 必须重跑代码管线（ask/code_qa），
    # 而不是被意图模型判成 meta/general 后用通用问答敷衍（旧 bug 行为）。
    second = asyncio.run(
        run_chat(
            "现在再回答一下",
            session_id=first.session_id,
            project_id="demo-loan",
            project_ids=["demo-loan"],
        )
    )
    assert second.intent == "ask"
    assert "需要查询项目代码" not in second.reply
    # 已经重跑代码管线（mock 检索返回 Controller 清单），而不是通用敷衍
    assert "Controller" in second.reply


def test_reanswer_english_reruns_previous_code_question():
    """English re-ask after selecting the project must re-run the code pipeline,
    exactly like the CN form (language-agnostic guard)."""
    _reset_env()

    from app.agent.graph_agent import run_chat

    # Turn 1: 英文接口清单问（命中 API-inventory 规则）
    first = asyncio.run(
        run_chat(
            "which settlement certificate apply APIs exist",
            project_id="demo-loan",
            project_ids=["demo-loan"],
        )
    )
    assert first.intent == "ask"
    assert "需要查询项目代码" not in first.reply

    # Turn 2: 英文重问 → 解析回上一问重新作答，而不是通用敷衍
    second = asyncio.run(
        run_chat(
            "answer again",
            session_id=first.session_id,
            project_id="demo-loan",
            project_ids=["demo-loan"],
        )
    )
    assert second.intent == "ask"
    assert "需要查询项目代码" not in second.reply
    # 重问内容承接上一问的英文问题（历史可见上一问）
    assert "settlement certificate" in second.reply or "APIs" in second.reply


def test_reanswer_with_code_content_kept_as_new_question():
    _reset_env()

    from app.agent.graph_agent import run_chat

    first = asyncio.run(
        run_chat(
            "订单流程什么样",
            project_id="demo-loan",
            project_ids=["demo-loan"],
        )
    )
    # 「回答一下放款流程」含自身代码锚点 → 不被解析回上一问，仍按新问题处理
    third = asyncio.run(
        run_chat(
            "回答一下放款流程",
            session_id=first.session_id,
            project_id="demo-loan",
            project_ids=["demo-loan"],
        )
    )
    assert third.intent == "ask"
    assert third.reply and "放款流程" in third.reply


# ---------------------------------------------------------------------------
# 无项目时：代码域问题应提示勾选项目（turn-1 UX）
# ---------------------------------------------------------------------------


def test_english_api_inventory_routing_and_topic():
    """English inventory asks route to the API-inventory path with a clean topic."""
    from app.agent.explore_intent import _api_inventory_query_core, _wants_api_inventory

    # Positive: English inventory shapes.
    for text in (
        "which settlement certificate apply APIs exist",
        "list the endpoints for settlement",
        "what interfaces are there",
        "which apis does this module include",
        "apis available for the apply flow",
    ):
        assert _wants_api_inventory(text), text

    # Negative: bare which/what or general API-concepts stay general.
    for text in (
        "what is API design best practice",
        "who called the api yesterday",
        "how does the api handle errors",
    ):
        assert not _wants_api_inventory(text), text

    # Topic core strips English glue whole-word, keeps topic nouns.
    assert _api_inventory_query_core(
        "which settlement certificate apply APIs exist"
    ) == "settlement certificate apply"
    assert (
        _api_inventory_query_core("list the endpoints for settlement certificate")
        == "settlement certificate"
    )


def test_no_project_code_domain_question_hints_select_project():
    _reset_env()

    from app.agent.graph_agent import run_chat

    resp = asyncio.run(run_chat("结清证明申请接口都有哪些", project_ids=[]))
    assert resp.project_ids == []
    assert "请先在左侧勾选项目" in resp.reply