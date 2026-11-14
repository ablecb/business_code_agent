"""「放款风控回调」must lock Risk* callback/listener, not AboutFinanceGuarantee#loanCallback."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

Q = "放款风控回调loan后 loan都做了啥"


def test_tokens_expand_risk() -> None:
    from app.analysis.entry_route import _question_content_tokens
    from app.agent.explore_agent import _question_match_tokens, _lexical_main_entry

    toks = _question_content_tokens(Q)
    assert "风控" in toks
    assert "回调" in toks
    assert "放款" in toks

    pick_toks = _question_match_tokens(Q)
    # explore pick stems are role-oriented (回调/通知…), not full business CN map.
    assert "回调" in pick_toks

    # Must not force WithdrawalsController when asking risk callback.
    assert _lexical_main_entry(Q) is None
    assert _lexical_main_entry("放款流程是什么样的") is None
    assert _lexical_main_entry("放款从入口到回调串一下") is None


def test_rerank_prefers_risk_not_guarantee() -> None:
    from app.analysis.entry_route import (
        classify_flow_role,
        rerank_entry_routes_for_question,
    )

    assert (
        classify_flow_role(
            kind="Listener",
            class_method="RiskControlRabbitMQListener#consume",
            method_name="consume",
            fqcn="com.demo.loan.listener.RiskControlRabbitMQListener",
        )
        == "callback"
    )

    hits = [
        {
            "class_method": "AboutFinanceGuaranteeController#loanCallback",
            "flow_role": "callback",
            "api_doc": "融担放款担保回调",
            "http_paths": "/loan/callback/v1/loan",
            "fqcn": "com.demo.loan.controller.AboutFinanceGuaranteeController",
            "method_name": "loanCallback",
            "routing_text": (
                "融担放款担保回调 AboutFinanceGuaranteeController#loanCallback "
                "role=callback 放款担保回调"
            ),
            "score": 0.92,
        },
        {
            "class_method": "RiskControlRabbitMQListener#consume",
            "flow_role": "callback",
            "api_doc": "风控",
            "http_paths": "",
            "fqcn": "com.demo.loan.listener.RiskControlRabbitMQListener",
            "method_name": "consume",
            "routing_text": (
                "风控 RiskControlRabbitMQListener#consume role=callback MQ决策结果"
            ),
            "score": 0.55,
        },
        {
            "class_method": "RiskDemoServiceImpl#callbackRisk",
            "flow_role": "callback",
            "api_doc": "风控回调",
            "http_paths": "",
            "fqcn": "com.demo.loan.service.impl.RiskDemoServiceImpl",
            "method_name": "callbackRisk",
            "routing_text": "风控回调 RiskDemoServiceImpl#callbackRisk role=callback",
            "score": 0.5,
        },
        {
            "class_method": "WithdrawalsController#withdrawals",
            "flow_role": "main",
            "api_doc": "放款",
            "http_paths": "/withdrawals/withdrawals",
            "fqcn": "com.demo.loan.controller.WithdrawalsController",
            "method_name": "withdrawals",
            "routing_text": "放款 WithdrawalsController#withdrawals role=main",
            "score": 0.8,
        },
    ]
    ranked = rerank_entry_routes_for_question(hits, Q)
    top = ranked[0]["class_method"]
    assert "Risk" in top, f"expected Risk* top, got {top}"
    assert "AboutFinanceGuarantee" not in top
    assert "loanCallback" not in top or "Risk" in top
    # Guarantee must rank below any Risk* hit.
    cms = [h["class_method"] for h in ranked]
    risk_idx = min(i for i, c in enumerate(cms) if "Risk" in c)
    guar_idx = cms.index("AboutFinanceGuaranteeController#loanCallback")
    assert risk_idx < guar_idx, cms


def test_guarantee_still_wins_when_asked() -> None:
    from app.analysis.entry_route import rerank_entry_routes_for_question

    hits = [
        {
            "class_method": "AboutFinanceGuaranteeController#loanCallback",
            "flow_role": "callback",
            "api_doc": "融担放款担保回调",
            "http_paths": "/loan/callback/v1/loan",
            "fqcn": "com.demo.loan.controller.AboutFinanceGuaranteeController",
            "method_name": "loanCallback",
            "routing_text": "融担放款担保回调 AboutFinanceGuaranteeController#loanCallback",
            "score": 0.7,
        },
        {
            "class_method": "RiskControlRabbitMQListener#consume",
            "flow_role": "callback",
            "api_doc": "风控",
            "http_paths": "",
            "fqcn": "com.demo.loan.listener.RiskControlRabbitMQListener",
            "method_name": "consume",
            "routing_text": "风控 RiskControlRabbitMQListener#consume",
            "score": 0.7,
        },
    ]
    top = rerank_entry_routes_for_question(hits, "融担放款担保回调做了什么")[0]
    assert top["class_method"] == "AboutFinanceGuaranteeController#loanCallback"


def main() -> None:
    test_tokens_expand_risk()
    print("OK tokens + lexical")
    test_rerank_prefers_risk_not_guarantee()
    print("OK risk callback rerank")
    test_guarantee_still_wins_when_asked()
    print("OK guarantee when asked")


if __name__ == "__main__":
    main()
