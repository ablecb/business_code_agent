"""Session dig-in lock vs new-topic re-resolve (LLM context gate)."""

import asyncio

from app.agent.ask_retrieval import prefer_flow_entry
from app.agent.graph_agent import (
    decide_continue_locked_entry,
    is_dig_in_under_lock,
    is_new_topic_ask,
    resolve_session_ctx_gate,
)
from app.agent.pev_agent import (
    _exec_resolve_entry,
    _extract_request_param_keys,
    _extract_response_param_keys,
    _focus_cms_from_question,
    _prefer_topic_focus_cms,
    _render_focus_method_answer,
    _wants_dig_in_change,
    build_task_plan,
    extract_goal_contract,
    verify_contract,
)


def test_dig_in_loan_change_keeps_lock():
    """Structural Latin spine method id → dig-in without NL keyword lists."""
    ctx = {
        "last_class_method": "WithdrawalsController#withdrawals",
        "last_flow_symbols": [
            "WithdrawalsController#withdrawals",
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsServiceImpl#loanApply",
        ],
    }
    locked = "WithdrawalsController#withdrawals"
    q = "loan 方法里客户信息同步要改什么"
    assert not is_new_topic_ask(q, locked, ctx)
    assert is_dig_in_under_lock(q, locked, ctx)
    assert _wants_dig_in_change(q)


def test_notice_topic_clears_withdrawals_lock(monkeypatch):
    """LLM unrelated → unlock (no CN notice keyword gate in production)."""
    from app.llm.base import LLMProvider

    class _Unrelated(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":false,"continue":false,"dig_in":false,'
                '"ask_kind":"new_topic","reason":"different API"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _Unrelated()
    )
    ctx = {
        "last_class_method": "WithdrawalsController#withdrawals",
        "last_flow_symbols": ["WithdrawalsServiceImpl#loan"],
        "last_explore_question": "放款接口什么流程",
        "last_dig_in_focus_cms": [
            "WithdrawalsServiceImpl#quotaCheck",
            "WithdrawalsServiceImpl#quotaCheckMIY",
        ],
    }
    q = "授信通知接口逻辑是什么"
    # Structural probe alone must not claim new_topic via CN keywords.
    assert not is_new_topic_ask(q, "WithdrawalsController#withdrawals", ctx)
    assert not is_dig_in_under_lock(q, "WithdrawalsController#withdrawals", ctx)
    gate = asyncio.run(resolve_session_ctx_gate(q, ctx, session_id=""))
    assert gate["use_session_ctx"] is False
    assert gate["new_topic"] is True
    assert gate["dig_in"] is False
    # Policy: unlocked / new_topic must not carry prior dig-in into prefer.
    carry_prior_dig_in = bool(gate["use_session_ctx"] or gate["dig_in"])
    assert carry_prior_dig_in is False


def test_full_flow_goal_skips_foreign_prefer_view_hint():
    """desktop-chat-d667b9c1da0d: full_flow must not stamp prior dig-in as view_hint."""
    from app.agent import pev_agent as pev

    goal = {
        "goal": "explain_flow",
        "ask_scope": "full_flow",
        "entry_hint": "授信通知接口",
        "must": [{"id": "entry", "desc": "x"}],
        "also": [],
        "deliverable": "desc",
        "source": "test",
    }
    evidence: dict = {"lock_entry": False}
    prefer = ["WithdrawalsServiceImpl#quotaCheck"]
    # Mirror run_pev_pipeline prefer→view_hint gate.
    ask_scope0 = str(goal.get("ask_scope") or "").strip().lower()
    carry_view = bool(
        evidence.get("lock_entry")
        or evidence.get("dig_in_likely")
        or evidence.get("dig_in_focus")
        or str(goal.get("goal") or "") == "show_code"
        or ask_scope0 in {"entry_scoped_dig_in", "dig_in"}
    )
    if prefer and carry_view:
        evidence["view_hint"] = prefer[0]
        goal["view_hint"] = prefer[0]
    assert "view_hint" not in evidence
    assert "view_hint" not in goal
    # Dig-in under lock still carries.
    evidence2 = {"lock_entry": True, "dig_in_likely": True}
    goal2 = dict(goal)
    goal2["ask_scope"] = "entry_scoped_dig_in"
    ask2 = str(goal2.get("ask_scope") or "").strip().lower()
    carry2 = bool(
        evidence2.get("lock_entry")
        or evidence2.get("dig_in_likely")
        or ask2 in {"entry_scoped_dig_in", "dig_in"}
    )
    if prefer and carry2:
        evidence2["view_hint"] = prefer[0]
    assert evidence2.get("view_hint") == "WithdrawalsServiceImpl#quotaCheck"
    del pev  # imported for module stability / future hook


def test_prefer_hint_wins_on_dig_in():
    picked = prefer_flow_entry(
        [
            "LoanAssistanceExtController#loanAssistanceExt",
            "WithdrawalsController#withdrawals",
        ],
        entry_routes=[
            {
                "cm": "LoanAssistanceExtController#loanAssistanceExt",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "score": 9.0,
            },
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "score": 1.0,
            },
        ],
        entry_hint="WithdrawalsController#withdrawals",
        question="loan 方法里客户信息同步要改什么",
        prefer_hint=True,
    )
    assert picked == "WithdrawalsController#withdrawals"


def test_unlocked_notice_picks_by_score_not_stale_hint():
    """After LLM unlock, prefer_hint=False → higher-score notice entry wins."""
    picked = prefer_flow_entry(
        [
            "WithdrawalsController#withdrawals",
            "CreditController#creditNotification",
        ],
        entry_routes=[
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "score": 5.0,
            },
            {
                "cm": "CreditController#creditNotification",
                "flow_role": "notice",
                "is_http_facade": True,
                "is_helper": False,
                "score": 8.0,
            },
        ],
        entry_hint="",
        question="授信通知接口逻辑是什么",
        prefer_hint=False,
    )
    assert picked == "CreditController#creditNotification"


def test_focus_render_uses_spine_loan():
    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "spine_source": "neo4j_calls",
        "calls": [
            "WithdrawalsServiceImpl#loanApply",
            "WithdrawalsServiceImpl#loan",
        ],
        "mysql_rows": [
            {
                "class_method": "WithdrawalsServiceImpl#loan",
                "summary": "放款核心：同步客户信息并调用下游",
            },
            {
                "class_method": "WithdrawalsServiceImpl#orgQuotaCheck",
                "summary": "校验机构限额状态",
            },
        ],
        "lock_entry": True,
    }
    q = "loan 方法里客户信息同步要改什么"
    focus = _focus_cms_from_question(q, evidence)
    assert any(cm.endswith("#loan") for cm in focus)
    ans = _render_focus_method_answer(evidence, focus, question=q)
    assert "WithdrawalsController#withdrawals" in ans
    assert "WithdrawalsServiceImpl#loan" in ans
    assert "LoanAssistance" not in ans


def test_topic_score_prefers_org_quota():
    from app.agent.pev_agent import _score_topic_hit, _topic_needles_from_question

    q = "loan方法中机构限额校验是什么 具体做了什么"
    needles = _topic_needles_from_question(q, ["WithdrawalsServiceImpl#loan"])
    assert any("限额" in n or "机构" in n for n in needles)
    assert _score_topic_hit(
        "WithdrawalsServiceImpl#orgQuotaCheck", "校验机构限额", needles
    ) > _score_topic_hit(
        "WithdrawalsServiceImpl#insertLoanData", "插入放款数据", needles
    )


def test_credit_io_ask_not_structural_new_topic():
    """IO ask has no foreign Class#method → structural new_topic=false; LLM decides."""
    ctx = {
        "last_class_method": "CreditController#creditNotification",
        "last_flow_symbols": [
            "CreditController#creditNotification",
            "CreditController#generateReqParam",
            "CreditServiceImpl#synchroMemberPrice",
        ],
    }
    locked = "CreditController#creditNotification"
    q = "请求参数有哪些 响应参数有哪些"
    assert not is_new_topic_ask(q, locked, ctx)


def test_continue_locked_entry_llm_true(monkeypatch):
    from app.llm.base import LLMProvider

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":true,"continue":true,"dig_in":false,'
                '"ask_kind":"followup","reason":"same API IO follow-up"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _FakeLLM()
    )
    out = asyncio.run(
        decide_continue_locked_entry(
            "请求参数有哪些 响应参数有哪些",
            "CreditController#creditNotification",
            prior_context="prior_question: 授信通知接口什么流程",
        )
    )
    assert out["continue"] is True
    assert out["source"] == "llm"


def test_intent_hint_matching_lock_does_not_skip_relatedness_llm(monkeypatch):
    """Intent copying prior Class#method must NOT force session keep.

    Regression: 用信申请 after 放款 — intent entry_hint == locked withdrawals
    used to short-circuit via intent_cm_matches_lock and reuse wrong entry.
    """
    from app.llm.base import LLMProvider

    calls: list[str] = []

    class _NewTopicLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            calls.append("chat")
            return (
                '{"related":false,"continue":false,"dig_in":false,'
                '"ask_kind":"new_topic","reason":"用信申请 vs 放款"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _NewTopicLLM()
    )
    locked = "WithdrawalsController#withdrawals"
    ctx = {
        "last_class_method": locked,
        "last_entry": locked,
        "last_explore_question": "放款接口什么流程",
        "last_flow_symbols": [
            "WithdrawalsController#checkApplyReqParam",
            "WithdrawalsServiceImpl#loan",
        ],
    }
    gate = asyncio.run(
        resolve_session_ctx_gate(
            "用信申请接口 主要做了什么",
            ctx,
            session_id="",
            entry_hint=locked,  # polluted intent hint
        )
    )
    assert calls, "relatedness LLM must run"
    assert gate["use_session_ctx"] is False
    assert gate["new_topic"] is True
    assert gate["dig_in"] is False
    assert (gate.get("continue_decision") or {}).get("ask_kind") == "new_topic"


def test_continue_locked_entry_fast_path_intent_match():
    """Same-API follow-up text is not treated as new_topic by heuristic alone."""
    locked = "CreditController#creditNotification"
    intent_cm = "CreditController#creditNotification"
    assert intent_cm.lower() == locked.lower()
    q = "请求参数有哪些 响应参数有哪些"
    assert not is_new_topic_ask(q, locked, {"last_class_method": locked})


def test_resolve_entry_does_not_soft_pin_class_method_hint(monkeypatch):
    """Unlocked turn: Class#method entry_hint alone must not skip free search."""

    def _fake_resolve(project_id, question, **kwargs):  # noqa: ARG001
        return (
            ["CreditController#creditNotification"],
            [
                {
                    "cm": "CreditController#creditNotification",
                    "api_doc": "授信通知",
                }
            ],
        )

    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta", _fake_resolve
    )
    evidence = {
        "entry_hint": "CreditQuotaController#creditNotification",
        "view_hint": "",
        "prefer_cms": [],
        "lock_entry": False,
    }
    asyncio.run(
        _exec_resolve_entry("demo-loan", "授信通知接口做了什么", evidence)
    )
    assert evidence["entry"] == "CreditController#creditNotification"
    assert evidence["entry_matches"] == ["CreditController#creditNotification"]


def test_resolve_entry_pins_view_hint(monkeypatch):
    """B: view_hint Controller#method pins; free-search must not overwrite."""

    def _boom(*_a, **_k):
        raise AssertionError("resolve_flow_entries_with_meta should not run")

    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta", _boom
    )
    evidence = {
        "entry_hint": "",
        "view_hint": "CreditController#creditNotification",
        "prefer_cms": [],
        "lock_entry": False,
    }
    asyncio.run(
        _exec_resolve_entry("demo-loan", "请求参数有哪些", evidence)
    )
    assert evidence["entry"] == "CreditController#creditNotification"


def test_resolve_entry_pins_prefer_cms_controller(monkeypatch):
    def _boom(*_a, **_k):
        raise AssertionError("free search should not run")

    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta", _boom
    )
    evidence = {
        "entry_hint": "ProductSyncController#productSync",
        "view_hint": "",
        "prefer_cms": ["CreditController#creditNotification"],
        "lock_entry": False,
    }
    asyncio.run(_exec_resolve_entry("demo-loan", "参数", evidence))
    assert evidence["entry"] == "CreditController#creditNotification"


def test_resolve_entry_pins_explicit_class_method_ask(monkeypatch):
    """User names Class#method (WithdrawalsServiceImpl#loan 做了什么) → pin it directly.

    Regression: desktop-chat-0f3cb1d6b410 asked about WithdrawalsServiceImpl#loan
    but the answer described the WithdrawalsController#withdrawals chain — the
    flow-entries resolver returns only Controller facades (never Service helpers)
    and the fuzzy symbol ranking dropped the exact method. Method-level asks must
    answer the named symbol itself.
    """

    def _boom(*_a, **_k):
        raise AssertionError("free search should not run for an explicit anchor")

    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta", _boom
    )
    monkeypatch.setattr(
        "app.agent.pev_flow._entry_is_real",
        lambda _pid, cm: str(cm) == "WithdrawalsServiceImpl#loan",
    )
    evidence = {
        "entry_hint": "",
        "view_hint": "",
        "prefer_cms": [],
        "lock_entry": False,
    }
    asyncio.run(
        _exec_resolve_entry(
            "demo-loan", "WithdrawalsServiceImpl#loan 这个方法主要做了什么", evidence
        )
    )
    assert evidence["entry"] == "WithdrawalsServiceImpl#loan"
    assert evidence["entry_matches"] == ["WithdrawalsServiceImpl#loan"]


def test_resolve_entry_flow_words_keep_controller_entry(monkeypatch):
    """Explicit anchor + flow words (调用链) must keep the Controller entry."""

    def _fake_resolve(project_id, question, **kwargs):  # noqa: ARG001
        return (
            ["WithdrawalsController#withdrawals"],
            [{"cm": "WithdrawalsController#withdrawals"}],
        )

    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta", _fake_resolve
    )
    monkeypatch.setattr("app.agent.pev_flow._entry_is_real", lambda _p, _c: True)
    monkeypatch.setattr(
        "app.agent.pev_flow._resolve_entry_via_symbol_search", lambda _p, _q: None
    )
    monkeypatch.setattr(
        "app.agent.pev_flow._topic_ground_entry_candidates",
        lambda _p, _q: {"phrase": "", "controllers": [], "primary": []},
    )
    evidence = {
        "entry_hint": "",
        "view_hint": "",
        "prefer_cms": [],
        "lock_entry": False,
    }
    asyncio.run(
        _exec_resolve_entry(
            "demo-loan", "WithdrawalsServiceImpl#loan 的调用链是什么", evidence
        )
    )
    assert evidence["entry"] == "WithdrawalsController#withdrawals"


def test_api_io_goal_keeps_llm_judgement(monkeypatch):
    """C: Goal LLM owns goal (no regex post-guards) — api_io stays; plan uses collect_api_io."""
    from app.llm.base import LLMProvider

    q = "请求参数有哪些 响应参数有哪些"

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"api_io","must":[{"id":"entry","desc":"x"}],'
                '"tables":[],"deliverable":"列出参数"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.llm.factory.get_llm_provider", lambda: _FakeLLM()
    )
    out = asyncio.run(extract_goal_contract(q))
    assert out["goal"] == "api_io"
    tools = [t["tool"] for t in build_task_plan(out)]
    assert "collect_api_io" in tools
    assert "collect_field_evidence" not in tools


def test_api_io_verify_wants_param_evidence_not_field_pack():
    goal = {
        "goal": "api_io",
        "must": [
            {"id": "entry", "desc": "入口"},
            {"id": "io_ev", "desc": "参数"},
            {"id": "answer", "desc": "作答"},
        ],
    }
    evidence = {
        "entry": "CreditController#creditNotification",
        "api_io_pack": {
            "request_params": ["orgCode", "memberId"],
            "response_params": ["responseCode", "responseMsg"],
            "sources": [{"class_method": "CreditController#generateReqParam"}],
        },
    }
    ans = (
        "业务入口：`CreditController#creditNotification`\n\n"
        "## 请求参数\n\n- `orgCode`\n- `memberId`\n\n"
        "## 响应参数\n\n- `responseCode`\n- `responseMsg`\n\n"
        "依据：请求参数=2；响应参数=2；来源方法=1"
    )
    v = verify_contract(goal, evidence, ans)
    assert v["ok"] is True
    assert not any(g.get("id") == "field_ev" for g in (v.get("gaps") or []))


def test_api_io_extractors_reqmap_and_response():
    code = """
    reqMap.get("orgCode");
    reqMap.get("memberId");
    result.put("responseCode", "0000");
    result.put("responseMsg", "ok");
    """
    assert "orgCode" in _extract_request_param_keys(code)
    assert "memberId" in _extract_request_param_keys(code)
    assert "responseCode" in _extract_response_param_keys(code)


def test_prefer_topic_focus_synchro_member_price():
    evidence = {
        "entry": "CreditController#creditNotification",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
        ],
    }
    focus = _prefer_topic_focus_cms(
        ["CreditServiceImpl#creditNotification"],
        {
            "dig_in": True,
            "focus_method": "CreditServiceImpl#creditNotification",
            "topic": "synchroMemberPrice 同步会员价格",
        },
        evidence,
    )
    assert focus[0].endswith("#synchroMemberPrice")
    assert len(focus) == 1


def test_prefer_topic_focus_chinese_only_promotes_leaf():
    """Chinese-only topic (no Latin method id) still promotes the leaf."""
    evidence = {
        "entry": "CreditController#creditNotification",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，含同步会员价格信息等步骤",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
        ],
    }
    focus = _prefer_topic_focus_cms(
        ["CreditServiceImpl#creditNotification"],
        {
            "dig_in": True,
            "focus_method": "CreditServiceImpl#creditNotification",
            "topic": "同步会员价格信息",
        },
        evidence,
        question="同步会员价格信息 是做啥呢",
    )
    assert focus == ["CreditServiceImpl#synchroMemberPrice"]


def _credit_notification_session_ctx() -> dict:
    """Session after T1 explain_flow (fixtures may use product nouns; code must not)."""
    return {
        "last_class_method": "CreditController#creditNotification",
        "last_entry": "CreditController#creditNotification",
        "last_explore_question": "授信通知接口什么流程",
        "last_flow_symbols": [
            "CreditController#creditNotification",
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        ],
        "last_answer_steps": [
            {
                "n": 1,
                "title": "授信通知主流程",
                "detail": "CreditServiceImpl#creditNotification 处理授信通知",
                "class_method": "CreditServiceImpl#creditNotification",
            },
            {
                "n": 2,
                "title": "同步会员价格信息",
                "detail": "CreditServiceImpl#synchroMemberPrice 同步会员价格",
                "class_method": "CreditServiceImpl#synchroMemberPrice",
            },
        ],
    }


def test_synchro_member_price_dig_in_via_llm(monkeypatch):
    """LLM related+dig_in → keep credit lock (no CN overlap auto-lock)."""
    from app.llm.base import LLMProvider

    class _RelatedDigIn(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":true,"continue":true,"dig_in":true,'
                '"ask_kind":"dig_in","reason":"same entry step dig-in"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _RelatedDigIn()
    )
    ctx = _credit_notification_session_ctx()
    q = "同步会员价格信息这个主要是做什么"
    # No Latin method id → structural dig-in false; LLM is authority.
    assert not is_dig_in_under_lock(q, "CreditController#creditNotification", ctx)
    gate = asyncio.run(resolve_session_ctx_gate(q, ctx, session_id=""))
    assert gate["use_session_ctx"] is True
    assert gate["dig_in"] is True
    assert gate["new_topic"] is False
    assert (gate.get("continue_decision") or {}).get("source") == "llm"


def test_loan_flow_after_credit_llm_unlocks(monkeypatch):
    """Fixture utterance 放款 after 授信: LLM unrelated → unlock; no structural dig-in."""
    from app.llm.base import LLMProvider

    class _Unrelated(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":false,"continue":false,"dig_in":false,'
                '"ask_kind":"new_topic","reason":"different API"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _Unrelated()
    )
    ctx = _credit_notification_session_ctx()
    locked = "CreditController#creditNotification"
    q = "放款接口什么流程"
    assert not is_dig_in_under_lock(q, locked, ctx)
    assert not is_new_topic_ask(q, locked, ctx)  # structural-only
    gate = asyncio.run(resolve_session_ctx_gate(q, ctx, session_id=""))
    assert gate["use_session_ctx"] is False
    assert gate["dig_in"] is False
    assert gate["new_topic"] is True


def test_entry_level_continue_false_unlocks_credit_for_loan(monkeypatch):
    """LLM continue=false after credit session + 放款接口什么流程 → clear lock."""
    from app.llm.base import LLMProvider

    class _UnrelatedLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"related":false,"continue":false,"dig_in":false,'
                '"ask_kind":"new_topic","reason":"different API / loan flow"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _UnrelatedLLM()
    )
    ctx = _credit_notification_session_ctx()
    q = "放款接口什么流程"
    out = asyncio.run(
        decide_continue_locked_entry(
            q,
            "CreditController#creditNotification",
            prior_context="prior_question: 授信通知接口什么流程",
            session_ctx=ctx,
        )
    )
    assert out["continue"] is False
    assert out["source"] == "llm"

    gate = asyncio.run(resolve_session_ctx_gate(q, ctx, session_id=""))
    assert gate["use_session_ctx"] is False
    assert gate["dig_in"] is False
    assert gate["new_topic"] is True


def test_uncertain_llm_fail_open_unlocks(monkeypatch):
    """LLM empty/uncertain → unlock (do not keep wrong lock)."""
    from app.llm.base import LLMProvider

    class _EmptyLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return "not-json"

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _EmptyLLM()
    )
    ctx = _credit_notification_session_ctx()
    out = asyncio.run(
        decide_continue_locked_entry(
            "放款接口什么流程",
            "CreditController#creditNotification",
            prior_context="prior_question: 授信通知接口什么流程",
            session_ctx=ctx,
        )
    )
    assert out["continue"] is False
    assert out["source"] == "default_unlock"


def test_synchro_latin_structural_dig_in_skips_llm(monkeypatch):
    """Literal Latin spine method → structural keep; LLM must not run."""
    from app.llm.base import LLMProvider

    class _BoomLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            raise AssertionError("LLM must not run for structural Latin dig-in")

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.agent.graph_agent.get_llm_provider", lambda: _BoomLLM()
    )
    ctx = _credit_notification_session_ctx()
    gate = asyncio.run(
        resolve_session_ctx_gate(
            "synchroMemberPrice 这一块做什么",
            ctx,
            session_id="",
        )
    )
    assert gate["use_session_ctx"] is True
    assert gate["dig_in"] is True
    assert gate["new_topic"] is False
    assert (gate.get("continue_decision") or {}).get("source") == "structural"


def test_unlocked_loan_flow_resolves_withdrawals_no_session_reuse(monkeypatch):
    """After unlock: lock_entry=False → resolve_entry to withdrawals; no session_reuse."""
    from app.agent import pev_agent as pev

    q = "放款接口什么流程"
    resolve_calls: list[str] = []

    async def _fake_resolve(_pid, _question, evidence):
        resolve_calls.append("resolve")
        evidence["entry"] = "WithdrawalsController#withdrawals"
        evidence["spine_source"] = "milvus_resolve"
        return evidence

    async def _fake_expand(_pid, evidence):
        evidence["calls"] = [
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsServiceImpl#loanApply",
        ]
        evidence["calls_count"] = 2
        evidence["spine_source"] = "neo4j_calls"
        return evidence

    async def _fake_mysql(_pid, evidence):
        evidence["mysql_rows"] = [
            {
                "class_method": "WithdrawalsController#withdrawals",
                "summary": "放款接口入口",
            },
            {
                "class_method": "WithdrawalsServiceImpl#loan",
                "summary": "放款核心处理",
            },
        ]
        return evidence

    async def _fake_summarize(question, evidence, **_k):  # noqa: ARG001
        entry = evidence.get("entry") or ""
        return f"业务入口：`{entry}`\n1. 放款入口\n2. 放款核心"

    async def _goal(q, **_k):  # noqa: ARG001
        return {
            "goal": "explain_flow",
            "also": [],
            "must": [
                {"id": "entry", "desc": "入口"},
                {"id": "spine", "desc": "主链"},
                {"id": "summarize", "desc": "作答"},
            ],
            "deliverable": "flow",
            "source": "test",
        }

    monkeypatch.setattr(pev, "_exec_resolve_entry", _fake_resolve)
    monkeypatch.setattr(pev, "_exec_expand_calls", _fake_expand)
    monkeypatch.setattr(pev, "_exec_mysql_enrich", _fake_mysql)
    monkeypatch.setattr(pev, "_exec_summarize_flow", _fake_summarize)
    monkeypatch.setattr(pev, "extract_goal_contract", _goal)
    monkeypatch.setattr(
        "app.agent.decompose.wants_decompose", lambda *_a, **_k: False
    )

    out = asyncio.run(
        pev.run_pev_pipeline(
            q,
            project_id="demo-loan",
            entry_hint="",
            lock_entry=False,
            dig_in_likely=False,
            flow_symbols=[
                "CreditController#creditNotification",
                "CreditServiceImpl#synchroMemberPrice",
            ],
            answer_steps=[
                {
                    "title": "授信通知主流程",
                    "detail": "credit",
                    "class_method": "CreditServiceImpl#creditNotification",
                }
            ],
        )
    )
    ev = out.get("evidence") or {}
    assert resolve_calls, "must run resolve_entry on unlocked path"
    assert ev.get("entry") == "WithdrawalsController#withdrawals"
    assert ev.get("reuse_shared_spine") is not True
    assert ev.get("spine_source") != "session_reuse"
    assert "CreditController" not in str(ev.get("entry") or "")
    ans = out.get("answer") or ""
    assert "WithdrawalsController#withdrawals" in ans


def test_explore_gate_followup_defined_no_nameerror(monkeypatch):
    """Regression: dig-in under lock must not NameError on gate_followup."""
    import uuid

    from app.agent import session_store
    from app.agent.graph_agent import _explore
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    sid = session_store.get_or_create_session(
        "demo", f"sess_gate_fu_{uuid.uuid4().hex[:8]}", title="t"
    )
    prior = "授信通知接口什么流程"
    session_store.update_context(
        sid,
        {
            "last_class_method": "CreditController#creditNotification",
            "last_entry": "CreditController#creditNotification",
            "last_flow_symbols": [
                "CreditController#creditNotification",
                "CreditServiceImpl#synchroMemberPrice",
            ],
            "last_explore_question": prior,
        },
    )
    session_store.append_message(sid, "user", prior)

    async def _fake_gate(*_a, **_k):
        return {
            "use_session_ctx": True,
            "dig_in": True,
            "new_topic": False,
            "continue_decision": {
                "continue": True,
                "related": True,
                "dig_in": True,
                "ask_kind": "dig_in",
                "source": "test",
            },
            "locked_cm": "CreditController#creditNotification",
            "entry_level_ask": False,
        }

    async def _skip_pev(*_a, **_k):
        return {"answer": "", "steps": [], "plan": {}}

    captured: dict = {}

    async def _fake_ask(*_a, **_k):
        return {"answer": "", "steps": []}

    async def _fake_explore(question, **_k):
        captured["question"] = question
        return {"answer": "dig-in-ok", "steps": [], "retrieved": []}

    monkeypatch.setattr(
        "app.agent.graph_agent.resolve_session_ctx_gate", _fake_gate
    )
    monkeypatch.setattr("app.agent.pev_agent.run_pev_pipeline", _skip_pev)
    monkeypatch.setattr(
        "app.agent.ask_retrieval.run_ask_retrieval_pipeline", _fake_ask
    )
    monkeypatch.setattr(
        "app.agent.graph_agent.run_explore_agent", _fake_explore
    )

    state = {
        "message": "synchroMemberPrice 这一块做什么",
        "project_id": "demo",
        "session_id": sid,
        "intent": "ask",
        "detail": "",
        "entry_hint": "",
    }
    out = asyncio.run(_explore(state))
    assert "dig-in-ok" in (out.get("reply") or "")
    q = str(captured.get("question") or "")
    assert "主链上的业务关卡" in q


def test_explore_new_topic_unlock_no_nameerror(monkeypatch):
    """New-topic unlock path must also avoid gate_followup NameError."""
    import uuid

    from app.agent import session_store
    from app.agent.graph_agent import _explore
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    sid = session_store.get_or_create_session(
        "demo", f"sess_new_topic_{uuid.uuid4().hex[:8]}", title="t"
    )
    session_store.update_context(
        sid,
        {
            "last_class_method": "CreditController#creditNotification",
            "last_explore_question": "授信通知接口什么流程",
        },
    )
    session_store.append_message(sid, "user", "授信通知接口什么流程")

    async def _fake_gate(*_a, **_k):
        return {
            "use_session_ctx": False,
            "dig_in": False,
            "new_topic": True,
            "continue_decision": {
                "continue": False,
                "related": False,
                "dig_in": False,
                "ask_kind": "new_topic",
                "source": "test",
            },
            "locked_cm": "CreditController#creditNotification",
            "entry_level_ask": True,
        }

    async def _skip_pev(*_a, **_k):
        return {"answer": "", "steps": [], "plan": {}}

    captured: dict = {}

    async def _fake_ask(*_a, **_k):
        return {"answer": "", "steps": []}

    async def _fake_explore(question, **_k):
        captured["question"] = question
        return {"answer": "new-topic-ok", "steps": [], "retrieved": []}

    monkeypatch.setattr(
        "app.agent.graph_agent.resolve_session_ctx_gate", _fake_gate
    )
    monkeypatch.setattr("app.agent.pev_agent.run_pev_pipeline", _skip_pev)
    monkeypatch.setattr(
        "app.agent.ask_retrieval.run_ask_retrieval_pipeline", _fake_ask
    )
    monkeypatch.setattr(
        "app.agent.graph_agent.run_explore_agent", _fake_explore
    )

    state = {
        "message": "放款接口什么流程",
        "project_id": "demo",
        "session_id": sid,
        "intent": "ask",
        "detail": "",
        "entry_hint": "",
    }
    out = asyncio.run(_explore(state))
    assert "new-topic-ok" in (out.get("reply") or "")
    q = str(captured.get("question") or "")
    assert "不要沿用会话里上一业务入口" in q
