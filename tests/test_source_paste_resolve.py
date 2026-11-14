"""Show-code / source-paste target resolution (Controller withdraw → withdrawals)."""
from __future__ import annotations

from app.agent.explore_intent import _wants_source_paste
from app.agent.explore_source import (
    _extract_source_paste_method_token,
    _method_name_fuzzy_score,
    _pick_best_method_match,
    _wants_controller_source,
)
from app.agent.question_intent import wants_show_gate_code


def test_withdraw_controller_source_ask_signals():
    q = "我想看放款接口 controller withdraw方法的源码"
    assert wants_show_gate_code(q)
    assert _wants_source_paste(q)
    assert _wants_controller_source(q)
    assert _extract_source_paste_method_token(q) == "withdraw"


def test_method_token_skips_word_controller():
    q = "看看 controller 源码"
    # no method token like withdraw — may return None or unrelated
    tok = _extract_source_paste_method_token(q)
    assert tok is None or tok.lower() != "controller"


def test_fuzzy_withdraw_to_withdrawals_prefers_facade():
    assert _method_name_fuzzy_score("withdrawals", "withdraw") == 80
    picked = _pick_best_method_match(
        [
            "WithdrawalsController#applyResultDealWith",
            "WithdrawalsController#checkApplyReqParam",
            "WithdrawalsController#withdrawals",
            "AmountController#dayQuota",
        ],
        "withdraw",
        prefer_controller=True,
    )
    assert picked == "WithdrawalsController#withdrawals"


def test_helpers_never_win_exact_method():
    picked = _pick_best_method_match(
        [
            "WithdrawalsController#applyResultDealWith",
            "WithdrawalsController#checkApplyReqParam",
        ],
        "withdraw",
        prefer_controller=True,
    )
    assert picked is None


def test_controller_typo_whole_file_ask():
    from app.agent.explore_source import (
        _normalize_show_code_question,
        _wants_controller_source,
    )

    q = "放款接口controoler源码"
    assert "controller" in _normalize_show_code_question(q).lower()
    assert _wants_controller_source(q)
    assert _extract_source_paste_method_token(q) is None
