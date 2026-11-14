from __future__ import annotations

from app.agent.intent_taxonomy import (
    heuristic_intent,
    normalize_intent,
    parse_option_selection,
    wants_implement_and_verify,
)


def test_normalize_legacy_intents():
    assert normalize_intent("ask_flow", "订单流程什么样") == "ask"
    # Valid/legacy labels win; do not keyword-override ask→plan.
    assert normalize_intent("ask_flow", "都要修改哪些类") == "ask"
    assert normalize_intent("modify_field", "直接改代码") == "modify"
    assert normalize_intent("new_api", "新增接口") == "modify"
    assert normalize_intent("impact", "改哪些") == "plan"
    assert normalize_intent("options", "三种方案") == "plan"
    assert normalize_intent("implement", "直接改") == "modify"
    assert normalize_intent("verify", "编译一下") == "execute"
    assert normalize_intent("repair", "修编译错误") == "execute"
    assert normalize_intent("general", "随便问问") == "ask"
    assert heuristic_intent("都要修改哪些类")["intent"] == "ask"


def test_heuristic_options_and_implement():
    assert heuristic_intent("给我三种实现方案对比利弊")["intent"] == "ask"
    assert heuristic_intent("直接改代码落地补丁")["intent"] == "ask"
    assert heuristic_intent("放款流程什么样")["intent"] == "ask"
    assert heuristic_intent("都要修改哪些")["intent"] == "ask"


def test_add_field_persist_outbound_is_plan():
    """陈述改造需求（加字段+落库/外送）即使没写「改哪些」也是 plan（unknown→fallback）。"""
    q = (
        "我想放款接口增加一个字段labelBusiness，并且将其保存到放款表，"
        "然后再调用额度校验时候送给外部系统"
    )
    assert heuristic_intent(q)["intent"] == "ask"
    # Prefer model label when already valid (LLM path); heuristic covers fallback.
    assert normalize_intent("ask", q) == "ask"
    assert normalize_intent("unknown_label", q) == "plan"
    # 纯理解落库流程仍是 ask
    assert heuristic_intent("放款落库是怎么做的")["intent"] == "ask"


def test_parse_option_selection():
    opts = [
        {"id": "A", "title": "最小改动"},
        {"id": "B", "title": "完整改造"},
    ]
    assert parse_option_selection("选方案B", opts) == "B"
    assert parse_option_selection("用第2个", opts) == "B"
    assert parse_option_selection("A", opts) == "A"
    assert parse_option_selection("选方案A", opts) == "A"
    assert parse_option_selection("无关内容", opts) is None


def test_negated_direct_change_not_want_apply():
    q = "先只给我要改的类/方法清单，不要直接改代码"
    h = heuristic_intent(q)
    assert h["want_apply"] is False
    assert wants_implement_and_verify(q) is False
    assert heuristic_intent("直接改代码落地补丁")["want_apply"] is False
