"""Unit tests for flow Neo4j sync payload builders (no Neo4j required)."""
from __future__ import annotations

from app.analysis.flow_neo4j_sync import build_follows_edges, build_reference_edges
from app.analysis.treesitter_deep_flow import (
    _build_execution_stages,
    _entry_call_spine,
    _is_off_chain_controller_sibling,
)


def test_build_reference_edges_orders_bfs_and_dedupes():
    by_cm = {
        "WithdrawalsController#withdrawals": {
            "symbol_id": "a",
            "external_calls": ["withdrawalsService.loan", "ResponseBuilder.buildSuccess"],
        },
        "WithdrawalsServiceImpl#loan": {"symbol_id": "b", "external_calls": []},
        "WithdrawalsServiceImpl#loanSwitch": {"symbol_id": "c", "external_calls": []},
    }
    callee_map = {
        "WithdrawalsController#withdrawals": [
            "WithdrawalsServiceImpl#loanSwitch",
            "WithdrawalsServiceImpl#loan",
        ]
    }
    edges = build_reference_edges(
        by_cm,
        callee_map,
        flow_key="loan_disburse",
        entry_class_method="WithdrawalsController#withdrawals",
    )
    assert edges[0]["from"] == "WithdrawalsController#withdrawals"
    assert edges[0]["to"] == "WithdrawalsServiceImpl#loanSwitch"
    assert edges[0]["order"] == 1
    assert edges[0]["rel"] == "CALLS"
    assert edges[0]["source"] == "treesitter-bfs"
    tos = [e["to"] for e in edges if e["from"] == "WithdrawalsController#withdrawals"]
    assert tos.count("WithdrawalsServiceImpl#loan") == 1


def test_build_follows_edges_from_flow_doc_stages_without_yaml():
    entry_extra = {
        "flow_doc": {
            "flow_key": "loan_disburse",
            "stages": [
                {"step": 1, "class_method": "A#a", "title": "t1"},
                {"step": 2, "class_method": "B#b", "title": "t2"},
                {"step": 3, "class_method": "C#c", "title": "t3"},
            ],
        }
    }
    follows = build_follows_edges(
        entry_extra,
        {},
        flow_key="loan_disburse",
        entry_class_method="WithdrawalsController#withdrawals",
        yaml_path=None,
        prefer_bfs_spine=False,
    )
    assert len(follows) == 3
    assert follows[0]["from"] == "WithdrawalsController#withdrawals"
    assert follows[0]["to"] == "A#a"
    assert follows[0]["step"] == 1
    assert follows[0]["rel"] == "FOLLOWS"
    assert follows[1]["from"] == "A#a"
    assert follows[2]["to"] == "C#c"


def test_entry_call_spine_prefers_ast_order_and_drops_siblings():
    bfs = {
        "tree": {
            "class_method": "WithdrawalsController#withdrawals",
            "children": [
                {"class_method": "WithdrawalsController#checkApplyReqParam", "file": "a.java"},
                {"class_method": "WithdrawalsServiceImpl#loanSwitch", "file": "b.java"},
                {"class_method": "WithdrawalsServiceImpl#loan", "file": "c.java"},
            ],
        },
        "edges": [],
    }
    known = {
        "WithdrawalsController#withdrawals",
        "WithdrawalsController#checkApplyReqParam",
        "WithdrawalsController#withDrawalCalculate",
        "WithdrawalsController#creditUsageApply",
        "WithdrawalsServiceImpl#loanSwitch",
        "WithdrawalsApplyServiceImpl#loanApply",
        "WithdrawalsServiceImpl#loan",
    }
    spine = _entry_call_spine(
        bfs,
        "WithdrawalsController#withdrawals",
        entry_calls=[
            "this.checkApplyReqParam",
            "withdrawalsService.loanSwitch",
            "withdrawalsApplyService.loanApply",
            "withdrawalsService.loan",
        ],
        known_cms=known,
    )
    assert spine[0] == "WithdrawalsController#checkApplyReqParam"
    assert "WithdrawalsServiceImpl#loanSwitch" in spine
    assert "WithdrawalsApplyServiceImpl#loanApply" in spine
    assert "WithdrawalsController#withDrawalCalculate" not in spine
    assert "WithdrawalsController#creditUsageApply" not in spine
    reachable = {
        "WithdrawalsController#withdrawals",
        "WithdrawalsController#checkApplyReqParam",
        "WithdrawalsServiceImpl#loanSwitch",
        "WithdrawalsServiceImpl#loan",
    }
    assert _is_off_chain_controller_sibling(
        "WithdrawalsController#creditUsageApply",
        "WithdrawalsController#withdrawals",
        reachable,
    )
    stages = _build_execution_stages(
        entry_cm="WithdrawalsController#withdrawals",
        bfs_result=bfs,
        important_payloads=[
            {"class_method": cm, "summary_business_clear": cm, "bfs_depth": 1}
            for cm in spine
        ],
        llm_stages=[
            {
                "step": 1,
                "class_method": "LimitAmountServiceImpl#agencyLimitRepayment",
                "title": "wrong",
            }
        ],
        entry_calls=[
            "this.checkApplyReqParam",
            "withdrawalsService.loanSwitch",
            "withdrawalsApplyService.loanApply",
            "withdrawalsService.loan",
        ],
    )
    assert stages[0]["class_method"] == "WithdrawalsController#checkApplyReqParam"
    assert all(
        st["class_method"] != "LimitAmountServiceImpl#agencyLimitRepayment"
        for st in stages
    )
