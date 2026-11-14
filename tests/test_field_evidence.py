"""Field-value evidence under locked Neo4j spine (+ gate/branch)."""

from app.agent.field_evidence import (
    analyze_gate_branches,
    collect_field_evidence_pack,
    detect_branch_intent,
    render_field_evidence_answer,
    resolve_gate_method,
    wants_field_value_ask,
)
from app.agent.graph_agent import is_dig_in_under_lock, is_new_topic_ask
from app.agent.pev_agent import heuristic_goal


def test_wants_field_value_status_on_table():
    q = "那放款接口正常执行完 放款表demo_b_withdrawals的放款状态是什么"
    assert wants_field_value_ask(q)
    # Enhanced heuristic routes flow-shaped asks to explain_flow (structural);
    # field_value routing is still owned by the Goal LLM.
    assert heuristic_goal(q)["goal"] in {"explain_flow", "general"}


def test_wants_field_after_gate():
    q = "机构限额校验后 放款表状态"
    assert wants_field_value_ask(q)
    assert detect_branch_intent(q) == "both"
    assert detect_branch_intent("机构限额校验失败 放款状态是什么") == "fail"


def test_dig_in_field_ask_under_lock(monkeypatch):
    """Field dig-in under lock: LLM related+dig_in (no CN auto-lock)."""
    import asyncio

    from app.llm.base import LLMProvider

    class _Related(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":true,"continue":true,"dig_in":true,'
                '"ask_kind":"dig_in","reason":"field dig-in under same entry"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _Related()
    )
    locked = "WithdrawalsController#withdrawals"
    ctx = {
        "last_class_method": locked,
        "last_flow_symbols": ["WithdrawalsServiceImpl#loan"],
        "last_explore_question": "放款接口什么流程",
    }
    q = "那放款接口正常执行完 放款表demo_b_withdrawals的放款状态是什么"
    # Structural probe: table name Latin token may or may not hit spine.
    assert not is_new_topic_ask(q, locked, ctx)
    from app.agent.graph_agent import resolve_session_ctx_gate

    gate = asyncio.run(resolve_session_ctx_gate(q, ctx, session_id=""))
    assert gate["use_session_ctx"] is True
    assert gate["dig_in"] is True



def test_resolve_gate_org_quota():
    spine = [
        "WithdrawalsServiceImpl#loan",
        "WithdrawalsServiceImpl#orgQuotaCheck",
        "WithdrawalsServiceImpl#quotaCheck",
    ]
    gate = resolve_gate_method("机构限额校验失败 放款状态是什么", spine)
    assert gate == "WithdrawalsServiceImpl#orgQuotaCheck"


def test_gate_fail_branch_writes_loan_status():
    pid = "demo-loan"
    info = analyze_gate_branches(
        pid,
        gate_cm="WithdrawalsServiceImpl#orgQuotaCheck",
        callers=["WithdrawalsServiceImpl#loan"],
    )
    assert info["fail_actions"], info
    assert any(
        "WITHDRAWALS_FAILED" in str(a.get("const_ref") or "")
        or a.get("resolved_value") == "20"
        for a in info["fail_actions"]
    ), info["fail_actions"]
    assert info["status_writes"], info
    fields = {str(w.get("field") or "").lower() for w in info["status_writes"]}
    assert "loanstatus" in fields or any(
        "loan" in f and "status" in f for f in fields
    ), fields
    # loanStatus hard-coded "F" in updateWithdrawals
    loan_writes = [
        w
        for w in info["status_writes"]
        if "loanstatus" == str(w.get("field") or "").lower()
    ]
    assert loan_writes and loan_writes[0].get("value") == "F", loan_writes


def test_collect_and_render_gate_both():
    pid = "demo-loan"
    entry = "WithdrawalsController#withdrawals"
    spine = [
        "WithdrawalsServiceImpl#loan",
        "WithdrawalsServiceImpl#orgQuotaCheck",
        "WithdrawalsServiceImpl#updateWithdrawals",
    ]
    q = "机构限额校验后 放款表状态"
    pack = collect_field_evidence_pack(pid, entry=entry, spine=spine, question=q)
    assert pack.get("gate") == "WithdrawalsServiceImpl#orgQuotaCheck"
    ans = render_field_evidence_answer(pack, q)
    assert "orgQuotaCheck" in ans
    assert "失败" in ans and "通过" in ans
    assert "loanStatus" in ans or "loan_status" in ans or "`F`" in ans or "F" in ans


def test_collect_field_pack_hits_loan_status():
    pid = "demo-loan"
    entry = "WithdrawalsController#withdrawals"
    spine = [
        "WithdrawalsServiceImpl#loan",
        "WithdrawalsServiceImpl#updateWithdrawals",
        "WithdrawalsServiceImpl#insertOrUpdateLoanInfo",
    ]
    q = "放款表demo_b_withdrawals的放款状态是什么 正常执行完"
    pack = collect_field_evidence_pack(pid, entry=entry, spine=spine, question=q)
    assert pack["bags_loaded"] >= 1
    ans = render_field_evidence_answer(pack, q)
    assert "WithdrawalsController#withdrawals" in ans
    assert "字段" in ans or "状态" in ans
