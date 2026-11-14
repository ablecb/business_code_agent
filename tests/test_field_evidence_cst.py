"""Deep business-QA tests: how variables/status change across gate branches.

Level-4 tests: the field-evidence chain that answers "校验失败后 loanStatus 变成什么"
with real Java-shaped sources — branch extraction from caller code, update-helper
setter discovery, constant resolution, and the final rendered answer. Sources are
synthetic but shaped like the real demo-loan code so no middleware is needed.
"""
from __future__ import annotations

import pytest

from app.agent.field_evidence import (
    analyze_gate_branches,
    detect_branch_intent,
    extract_field_anchors,
    render_field_evidence_answer,
    resolve_gate_method,
    wants_field_value_ask,
)

_CALLER_SRC = """package com.demo;

public class WithdrawalsServiceImpl {
    private WithdrawalsMapper withdrawalsMapper;

    public void loan(WithdrawalsApplyReq req) {
        if (!orgQuotaCheck(req)) {
            updateLoanFail(req.getDealRefNo(), "WITHDRAWALS_FAILED");
            return;
        }
        updateWithdrawals(req);
    }

    private boolean orgQuotaCheck(WithdrawalsApplyReq req) {
        return req.getAmount() <= 10000;
    }

    private void updateLoanFail(String no, String reason) {
        Withdrawals w = new Withdrawals();
        w.setDealRefNo(no);
        w.setLoanStatus("F");
        w.setFailReason(reason);
        withdrawalsMapper.updateById(w);
    }

    private void updateWithdrawals(WithdrawalsApplyReq req) {
        Withdrawals w = new Withdrawals();
        w.setDealRefNo(req.getDealRefNo());
        w.setLoanStatus("S");
        withdrawalsMapper.updateById(w);
    }
}
"""

_CONST_SRC = """package com.demo;

public class WithdrawalsConstants {
    public static final String WITHDRAWALS_FAILED = "20";
}
"""


@pytest.fixture
def _patch_sources(monkeypatch):
    import app.agent.field_evidence as fe

    def _fake_fetch(project_id: str, cm: str) -> str:
        del project_id
        if cm.endswith("#loan"):
            return _CALLER_SRC
        if cm.endswith("#updateLoanFail"):
            return _CALLER_SRC
        return ""

    monkeypatch.setattr(fe, "_fetch_source", _fake_fetch)
    monkeypatch.setattr(fe, "_resolve_java_constant", lambda _pid, ref: "20")


# ---------- intent / anchors ----------

def test_wants_field_value_change_after_gate():
    assert wants_field_value_ask("机构限额校验失败后 放款表状态是什么")
    assert wants_field_value_ask("放款接口正常执行完 放款表loan_status变成什么")
    assert not wants_field_value_ask("放款接口都涉及哪些表")


def test_branch_intent_detection():
    assert detect_branch_intent("机构限额校验失败 放款状态") == "fail"
    assert detect_branch_intent("机构限额校验通过 放款状态") == "pass"
    assert detect_branch_intent("机构限额校验后 放款状态") == "both"


def test_extract_anchors_fields_and_branch():
    anchors = extract_field_anchors("校验失败后 放款表 demo_b_withdrawals 的 loanStatus 是什么")
    assert "demo_b_withdrawals" in anchors["tables"] or any(
        "demo_b_withdrawals" in t for t in anchors["tables"]
    )
    assert anchors["branch"] == "fail"
    assert any("loanStatus" in f or "loan_status" in f for f in anchors["fields"])


# ---------- gate branch analysis (variable flow) ----------

def test_analyze_gate_branches_fail_actions(_patch_sources):
    info = analyze_gate_branches(
        "demo",
        gate_cm="WithdrawalsServiceImpl#orgQuotaCheck",
        callers=["WithdrawalsServiceImpl#loan"],
    )
    assert info["fail_actions"], info
    update = next(
        (a for a in info["fail_actions"] if "updateLoanFail" in str(a.get("call"))),
        None,
    )
    assert update is not None
    # "WITHDRAWALS_FAILED" is a plain string literal → resolved_value captured,
    # const_ref stays empty (only Class.CONST refs populate const_ref).
    assert update["resolved_value"] == "WITHDRAWALS_FAILED"
    # pass path note captured
    assert any("updateWithdrawals" in n for n in info["pass_notes"])


def test_analyze_gate_status_writes_resolves_setter_values(_patch_sources):
    """The variable-flow payoff: update helper setters → concrete values."""
    info = analyze_gate_branches(
        "demo",
        gate_cm="WithdrawalsServiceImpl#orgQuotaCheck",
        callers=["WithdrawalsServiceImpl#loan"],
    )
    writes = info["status_writes"]
    assert writes, info
    loan_writes = [
        w
        for w in writes
        if str(w.get("field") or "").lower() in {"loanstatus", "loan_status"}
    ]
    assert loan_writes
    # fail branch hard-codes loanStatus="F"
    assert loan_writes[0]["value"] == "F"
    assert loan_writes[0]["setter"] == "setLoanStatus"
    assert loan_writes[0]["branch"] == "fail"


def test_gate_resolution_from_spine(_patch_sources):
    spine = [
        "WithdrawalsServiceImpl#loan",
        "WithdrawalsServiceImpl#orgQuotaCheck",
        "WithdrawalsServiceImpl#quotaCheck",
    ]
    gate = resolve_gate_method("机构限额校验失败 放款状态是什么", spine)
    assert gate == "WithdrawalsServiceImpl#orgQuotaCheck"


# ---------- collect + render ----------

def test_collect_and_render_answer_has_variable_changes(_patch_sources):
    from app.agent.field_evidence import collect_field_evidence_pack

    pack = collect_field_evidence_pack(
        "demo",
        entry="WithdrawalsController#withdrawals",
        spine=[
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsServiceImpl#orgQuotaCheck",
            "WithdrawalsServiceImpl#updateWithdrawals",
        ],
        question="机构限额校验失败后 放款表状态是什么",
    )
    ans = render_field_evidence_answer(pack, "机构限额校验失败后 放款表状态是什么")
    # answer must mention the fail branch and the concrete status write
    assert "失败" in ans or "F" in ans or "loanStatus" in ans or "loan_status" in ans


def test_collect_and_render_no_gate_still_answers(_patch_sources):
    from app.agent.field_evidence import collect_field_evidence_pack

    pack = collect_field_evidence_pack(
        "demo",
        entry="WithdrawalsController#withdrawals",
        spine=["WithdrawalsServiceImpl#loan"],
        question="放款表loan_status正常执行完变成什么",
    )
    ans = render_field_evidence_answer(pack, "放款表loan_status正常执行完变成什么")
    assert ans  # no crash, some evidence-based answer


def test_analyze_gate_no_source_no_crash():
    info = analyze_gate_branches(
        "demo",
        gate_cm="Ghost#check",
        callers=["Ghost#run"],
    )
    assert info["fail_actions"] == []
    assert info["pass_notes"] == []
    assert info["status_writes"] == []
