"""Unit + light integration: entry_route classification and flow pick wiring.

Run::

    python tests/test_entry_route.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_classify_flow_role() -> None:
    from app.analysis.entry_route import classify_flow_role, build_routing_text
    from app.analysis.entry_route import rerank_entry_routes_for_question

    assert (
        classify_flow_role(
            kind="Controller",
            class_method="RepayController#repayment",
            method_name="repayment",
            fqcn="com.example.RepayController",
            api_doc="还款",
        )
        == "main"
    )
    assert (
        classify_flow_role(
            kind="Controller",
            class_method="BankcardController#queryRepayBank",
            method_name="queryRepayBank",
            fqcn="com.example.BankcardController",
            api_doc="查询还款卡",
        )
        == "query"
    )
    assert (
        classify_flow_role(
            kind="Controller",
            class_method="LoanNoticeController#notice",
            method_name="notice",
            fqcn="com.example.LoanNoticeController",
        )
        == "notice"
    )
    assert (
        classify_flow_role(
            kind="Controller",
            class_method="PayCallbackController#callback",
            method_name="callback",
            fqcn="com.example.PayCallbackController",
        )
        == "callback"
    )
    assert (
        classify_flow_role(
            kind="Listener",
            class_method="OrderResultListener#onMessage",
            method_name="onMessage",
            fqcn="com.example.OrderResultListener",
        )
        == "async"
    )
    # Risk* MQ result listener → callback (class stem), not generic async.
    assert (
        classify_flow_role(
            kind="Listener",
            class_method="RiskControlRabbitMQListener#consume",
            method_name="consume",
            fqcn="com.demo.loan.listener.RiskControlRabbitMQListener",
            api_doc="风控",
            summary="决策结果MQ",
        )
        == "callback"
    )
    assert (
        classify_flow_role(
            kind="Listener",
            class_method="RiskControlRabbitMQListener#consume",
            method_name="consume",
            fqcn="com.demo.loan.listener.RiskControlRabbitMQListener",
            existing_role="async",
        )
        == "callback"
    )
    assert (
        classify_flow_role(
            kind="Job",
            class_method="SettleJob#execute",
            method_name="execute",
            fqcn="com.example.SettleJob",
        )
        == "job"
    )
    # existing main + queryish method → query
    assert (
        classify_flow_role(
            kind="Controller",
            class_method="X#queryFoo",
            method_name="queryFoo",
            existing_role="main",
        )
        == "query"
    )

    text = build_routing_text(
        api_doc="还款",
        class_method="RepayController#repayment",
        http_paths=["/api/repay"],
        flow_role="main",
        kind="Controller",
        summary="【数据源头·HTTP入口】RepayController#repayment",
    )
    assert "还款" in text and "RepayController#repayment" in text and "role=main" in text

    hits = [
        {
            "class_method": "BankcardController#queryRepayBank",
            "flow_role": "query",
            "api_doc": "查询还款卡",
            "http_paths": "/bank/query",
            "routing_text": "查询还款卡 BankcardController#queryRepayBank role=query",
            "score": 0.9,
        },
        {
            "class_method": "RepayController#repayment",
            "flow_role": "main",
            "api_doc": "还款",
            "http_paths": "/api/repay",
            "routing_text": "还款 RepayController#repayment role=main",
            "score": 0.5,
        },
        {
            "class_method": "WithdrawalsController#withdrawals",
            "flow_role": "main",
            "api_doc": "",
            "http_paths": "withdrawals/withdrawals",
            "routing_text": "WithdrawalsController#withdrawals 放款入口 role=main",
            "score": 0.55,
        },
    ]
    top = rerank_entry_routes_for_question(hits, "还款流程是什么样的")[0]
    assert top["class_method"] == "RepayController#repayment", top
    top2 = rerank_entry_routes_for_question(hits, "放款流程是什么样的")[0]
    assert top2["class_method"] == "WithdrawalsController#withdrawals", top2


def test_rerank_demotes_notice_for_interface_logic_ask() -> None:
    """「接口+逻辑」无通知意图时，main 压过 notice（冷启动 ask）。"""
    from app.analysis.entry_route import rerank_entry_routes_for_question

    hits = [
        {
            "class_method": "LoanNoticeController#loanNotice",
            "flow_role": "notice",
            "api_doc": "放款通知",
            "http_paths": "/loan/notice",
            "routing_text": "LoanNoticeController#loanNotice role=notice 放款",
            "score": 0.95,
        },
        {
            "class_method": "WithdrawalsController#withdrawals",
            "flow_role": "main",
            "api_doc": "放款",
            "http_paths": "/withdrawals/withdrawals",
            "routing_text": "WithdrawalsController#withdrawals 放款入口 role=main",
            "score": 0.55,
        },
    ]
    q = "我想看看放款接口 额度校验那一块的逻辑"
    ranked = rerank_entry_routes_for_question(hits, q)
    assert ranked[0]["class_method"] == "WithdrawalsController#withdrawals", ranked[0]


def test_pick_with_prefer_anchor() -> None:
    from app.agent.explore_agent import _pick_structured_flow_doc
    from app.core.config import get_settings

    pid = get_settings().project_id
    # Without DB FlowDocs this may be None — only assert when present.
    picked = _pick_structured_flow_doc(
        pid, "还款流程是什么样的", prefer_class_method="RepayController#repayment"
    )
    if picked is None:
        print("SKIP pick (no FlowDoc in MySQL for RepayController#repayment)")
        return
    assert picked.get("class_method") == "RepayController#repayment", picked.get(
        "class_method"
    )
    assert not picked.get("is_queryish")


def test_live_entry_route_search() -> None:
    """Optional: requires rebuilt entry_route + embedding; skips if empty."""
    from app.analysis.entry_route import search_entry_routes
    from app.core.config import get_settings
    from app.db.milvus_client import get_entry_route_store, reset_milvus

    reset_milvus()
    store = get_entry_route_store()
    if not store.verify():
        print("SKIP live search (Milvus unavailable)")
        return
    pid = get_settings().project_id
    hits = search_entry_routes("还款流程是什么样的", pid, top_k=5)
    if not hits:
        print("SKIP live search (entry_route empty — run tests/rebuild_entry_route.py)")
        return
    top = hits[0]
    cm = top.get("class_method") or ""
    role = top.get("flow_role") or ""
    print(f"live top={cm} role={role} score={top.get('score')}")
    # Prefer non-query for bare「xx流程」
    assert role != "query" or "query" in cm.lower()
    # Soft expectation when repay materials exist
    if any("Repay" in str(h.get("class_method") or "") for h in hits):
        assert "queryRepay" not in cm or role == "query"


def main() -> None:
    test_classify_flow_role()
    print("OK classify + rerank")
    test_pick_with_prefer_anchor()
    print("OK prefer-anchor pick")
    test_live_entry_route_search()
    print("OK (live search may have skipped)")


if __name__ == "__main__":
    main()
