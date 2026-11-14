from __future__ import annotations

import asyncio
import os
from typing import Any

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"


def test_normalize_template_choice_invalid_falls_back():
    from app.agent.ask_retrieval import normalize_template_choice

    out = normalize_template_choice(
        {"template_id": "DROP_DB", "params": {"entry": "FooController#bar"}},
        default_entry="DemoController#demo",
    )
    assert out["template_id"] == "follows_spine"
    assert out["params"]["entry"] == "FooController#bar"

    out2 = normalize_template_choice(
        {"template_id": "calls_callees", "params": {}},
        default_entry="DemoController#demo",
    )
    assert out2["template_id"] == "calls_callees"
    assert out2["params"]["entry"] == "DemoController#demo"


def test_strip_tech_names_from_answer():
    from app.agent.ask_retrieval import _strip_tech_names, _wants_tech_names

    raw = (
        "1. 通过 `WithdrawalsController#withdrawals` 受理请求。\n"
        "2. 再进入 WithdrawalsServiceImpl#loan 做放款。"
    )
    cleaned = _strip_tech_names(raw)
    assert "WithdrawalsController#withdrawals" not in cleaned
    assert "WithdrawalsServiceImpl#loan" not in cleaned
    assert not _wants_tech_names("放款接口什么流程")
    assert _wants_tech_names("把类名方法名也写出来")


def test_synthesize_facade_and_reject_placeholder():
    from app.agent.ask_retrieval import (
        _is_placeholder_entry,
        controller_facade_cm,
        normalize_template_choice,
        synthesize_facades_from_hits,
    )

    assert controller_facade_cm("WithdrawalsController#applyResultDealWith") == (
        "WithdrawalsController#withdrawals"
    )
    assert synthesize_facades_from_hits(
        ["WithdrawalsController#applyResultDealWith", "AmountController#addDayQuota"]
    ) == ["WithdrawalsController#withdrawals", "AmountController#amount"]
    assert _is_placeholder_entry("Controller#method")
    out = normalize_template_choice(
        {
            "template_id": "follows_spine",
            "params": {"entry": "Controller#method"},
        },
        default_entry="WithdrawalsController#withdrawals",
    )
    assert out["params"]["entry"] == "WithdrawalsController#withdrawals"


def test_prefer_controller_facade_over_helper():
    from app.agent.ask_retrieval import (
        is_controller_helper_method,
        is_controller_http_facade,
        is_system_entry_point,
        prefer_flow_entry,
        _sanitize_neo4j_choice,
    )

    assert is_system_entry_point("WithdrawalsController#withdrawals", kind="Controller")
    assert is_system_entry_point("FooJob#execute", kind="Job")
    assert is_system_entry_point("BarListener#onMessage", kind="Listener")
    assert not is_system_entry_point("WithdrawalsServiceImpl#quotaCheck", kind="Method")
    assert is_controller_http_facade("WithdrawalsController#withdrawals")
    assert is_controller_helper_method("WithdrawalsController#applyResultDealWith")
    assert prefer_flow_entry(
        [
            "WithdrawalsServiceImpl#quotaCheck",
            "WithdrawalsController#applyResultDealWith",
            "WithdrawalsController#withdrawals",
        ],
        milvus_brief=[
            {
                "cm": "WithdrawalsServiceImpl#quotaCheck",
                "is_system_entry": False,
                "is_helper": False,
            },
            {
                "cm": "WithdrawalsController#applyResultDealWith",
                "is_system_entry": True,
                "is_helper": True,
            },
            {
                "cm": "WithdrawalsController#withdrawals",
                "is_system_entry": True,
                "is_http_facade": True,
                "is_helper": False,
            },
        ],
    ) == "WithdrawalsController#withdrawals"

    fixed = _sanitize_neo4j_choice(
        {
            "template_id": "follows_spine",
            "params": {
                "entry": "WithdrawalsServiceImpl#loan",
                "depth": 4,
                "limit": 20,
            },
            "reason": "llm",
        },
        preferred_entry="WithdrawalsController#withdrawals",
        allowed_entries=[
            "WithdrawalsController#applyResultDealWith",
            "WithdrawalsController#withdrawals",
        ],
    )
    assert fixed["params"]["entry"] == "WithdrawalsController#withdrawals"


def test_ask_retrieval_pipeline_steps(monkeypatch):
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent import ask_retrieval

    class _Hit:
        def __init__(self):
            self.symbol_id = "sym1"
            self.project_id = "p"
            self.fqcn = "com.demo.DemoController"
            self.method_sig = "demo()"
            self.kind = "Controller"
            self.file_path = "a.java"
            self.summary = "DemoController#demo 入口"
            self.tags = ["DemoController#demo"]
            self.score = 1.0

        def model_dump(self):
            return {
                "symbol_id": self.symbol_id,
                "project_id": self.project_id,
                "fqcn": self.fqcn,
                "method_sig": self.method_sig,
                "kind": self.kind,
                "file_path": self.file_path,
                "summary": self.summary,
                "tags": self.tags,
                "score": self.score,
            }

    monkeypatch.setattr(
        "app.tagging.search.hybrid_search",
        lambda *a, **k: [_Hit()],
    )

    def _fake_neo4j(project_id: str, choice: dict[str, Any]):
        entry = choice["params"]["entry"]
        return (
            [
                {
                    "order": 1,
                    "class_method": entry,
                    "title": "入口",
                    "rel": "FOLLOWS",
                    "desc": "入口",
                },
                {
                    "order": 2,
                    "class_method": "DemoService#run",
                    "title": "处理",
                    "rel": "FOLLOWS",
                    "desc": "处理",
                },
            ],
            {"ok": True, "raw_source": "mock"},
        )

    monkeypatch.setattr(ask_retrieval, "_run_neo4j_template", _fake_neo4j)
    monkeypatch.setattr(
        ask_retrieval,
        "_mysql_enrich_ordered",
        lambda project_id, nodes, **kw: [
            {
                "order": n["order"],
                "class_method": n["class_method"],
                "rel": n.get("rel"),
                "title": n.get("title"),
                "summary": n.get("desc"),
                "fqcn": f"com.demo.{n['class_method'].split('#')[0]}",
                "file_path": "x.java",
            }
            for n in nodes
        ],
    )

    result = asyncio.run(
        ask_retrieval.run_ask_retrieval_pipeline(
            "订单流程什么样",
            project_id="demo",
            history_text="user: hi",
        )
    )
    assert (result.get("answer") or "").strip()
    tools = [s.get("tool") for s in result.get("steps") or []]
    assert tools == ["milvus_search", "neo4j_plan", "mysql_enrich", "ask_summarize"]
    assert result.get("neo4j_query", {}).get("template_id") in ask_retrieval.TEMPLATE_IDS


def test_ask_retrieval_uses_history_for_neo4j_plan(monkeypatch):
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent import ask_retrieval

    seen: dict[str, Any] = {}

    class _Hit:
        symbol_id = "s1"
        fqcn = "com.demo.DemoController"
        method_sig = "demo()"
        kind = "Controller"
        summary = "DemoController#demo"
        tags = ["DemoController#demo"]
        score = 1.0

        def model_dump(self):
            return {
                "symbol_id": self.symbol_id,
                "fqcn": self.fqcn,
                "method_sig": self.method_sig,
                "kind": self.kind,
                "summary": self.summary,
                "tags": self.tags,
                "score": self.score,
            }

    async def _fake_plan(question, **kwargs):
        seen["question"] = question
        seen["history"] = kwargs.get("history_text")
        seen["prior"] = kwargs.get("prior_question")
        seen["brief"] = kwargs.get("milvus_brief")
        return {
            "template_id": "follows_spine",
            "params": {"entry": "DemoController#demo", "depth": 4, "limit": 20},
            "reason": "test",
        }

    monkeypatch.setattr(
        "app.tagging.search.hybrid_search", lambda *a, **k: [_Hit()]
    )
    monkeypatch.setattr(ask_retrieval, "plan_neo4j_with_llm", _fake_plan)
    monkeypatch.setattr(
        ask_retrieval,
        "_run_neo4j_template",
        lambda project_id, choice: (
            [{"order": 1, "class_method": "DemoController#demo", "rel": "ENTRY", "desc": "x"}],
            {"ok": True},
        ),
    )
    monkeypatch.setattr(
        ask_retrieval,
        "_mysql_enrich_ordered",
        lambda *a, **k: [
            {
                "order": 1,
                "class_method": "DemoController#demo",
                "summary": "demo",
            }
        ],
    )

    result = asyncio.run(
        ask_retrieval.run_ask_retrieval_pipeline(
            "那一步怎么校验",
            project_id="demo",
            prior_question="订单流程什么样",
            history_text="user: 订单流程什么样\nassistant: …",
        )
    )
    assert seen.get("prior") == "订单流程什么样"
    assert "订单流程" in (seen.get("history") or "")
    assert seen.get("brief")
    assert (result.get("answer") or "").strip()


def test_explore_ask_uses_ask_retrieval(monkeypatch):
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    async def _skip_pev(*_a, **_k):
        return {"answer": "", "steps": [], "plan": {}}

    monkeypatch.setattr("app.agent.pev_agent.run_pev_pipeline", _skip_pev)

    called: dict[str, Any] = {"ask": 0, "explore": 0}

    async def _fake_ask(*a, **k):
        called["ask"] += 1
        called["kwargs"] = k
        return {
            "answer": "ask-pipeline-hit",
            "steps": [
                {"step": 1, "tool": "milvus_search", "ok": True},
                {"step": 2, "tool": "neo4j_plan", "ok": True, "entry_anchor": "A#a"},
                {"step": 3, "tool": "mysql_enrich", "ok": True, "symbols": ["A#a"]},
                {"step": 4, "tool": "ask_summarize", "ok": True},
            ],
            "retrieved": [],
            "plan": {"kind": "ask_retrieval"},
            "neo4j_query": {
                "template_id": "follows_spine",
                "params": {"entry": "A#a"},
            },
            "mysql_rows": [],
        }

    async def _fake_explore(*a, **k):
        called["explore"] += 1
        return {"answer": "explore-hit", "steps": [], "retrieved": []}

    monkeypatch.setattr(
        "app.agent.ask_retrieval.run_ask_retrieval_pipeline", _fake_ask
    )
    monkeypatch.setattr(
        "app.agent.graph_agent.run_explore_agent", _fake_explore
    )

    from app.agent.graph_agent import run_chat

    resp = asyncio.run(run_chat("订单流程什么样"))
    assert called["ask"] == 1
    assert called["explore"] == 0
    assert "ask-pipeline-hit" in (resp.reply or "")


def test_resolve_flow_entry_for_add_field_question(monkeypatch):
    """Add-field「放款接口入参增加…」→ Withdrawals, not empty / LoanNotice."""
    from app.agent import ask_retrieval

    q = "我想在放款接口入参增加一个 businessLabel（业务标签），都要改哪些地方"
    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", lambda *a, **k: "")
    monkeypatch.setattr(
        ask_retrieval,
        "_entry_route_candidates",
        lambda project_id, question, top_k=8: [
            {
                "cm": "LoanNoticeController#loanNotice",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "放款通知",
                "score": 0.99,
            },
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "放款",
                "routing_text": "WithdrawalsController#withdrawals 放款入口 role=main",
                "http_paths": "/withdrawals/withdrawals",
                "score": 0.55,
            },
        ],
    )
    picked = ask_retrieval.resolve_flow_entry_for_question("demo-loan", q)
    assert picked == "WithdrawalsController#withdrawals", picked


def test_resolve_flow_entry_search_fallback_when_entry_route_empty(monkeypatch):
    from app.agent import ask_retrieval

    q = "我想在放款接口入参增加一个 businessLabel（业务标签），都要改哪些地方"
    monkeypatch.setattr(
        ask_retrieval, "_entry_route_candidates", lambda *a, **k: []
    )
    monkeypatch.setattr(
        ask_retrieval,
        "_search_controller_facade_routes",
        lambda *a, **k: [
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "score": 1.0,
                "source": "search_symbols_facade",
            }
        ],
    )
    picked = ask_retrieval.resolve_flow_entry_for_question("demo-loan", q)
    assert picked == "WithdrawalsController#withdrawals", picked


def test_entry_topic_queries_prefers_api_phrase():
    from app.agent.ask_retrieval import _entry_topic_queries

    q = (
        "帮我在放款接口 增加一个入参 aaaName，校验不能为空 "
        "并落库到demo_b_withdrawals，demo_b_loan_assistance这两张表"
    )
    qs = _entry_topic_queries(q, entry_hint="放款接口")
    assert qs[0] == "放款接口"
    assert "放款接口" in qs
    assert "demo_b_withdrawals" not in qs[0]


def test_entry_lock_queries_drops_gate_tails():
    from app.agent.ask_retrieval import _entry_lock_queries

    q = "放款接口中 机构限额校验具体做了什么"
    qs = _entry_lock_queries(q)
    assert qs == ["放款接口"], qs
    assert all("限额" not in x for x in qs)


def test_resolve_flow_entry_quota_dig_in_not_amount_controller(monkeypatch):
    """「放款接口中 机构限额…」must lock Withdrawals, not AmountController."""
    from app.agent import ask_retrieval

    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", lambda *a, **k: "")
    seen: list[str] = []

    def _cands(project_id, question, top_k=8):  # noqa: ARG001
        seen.append(question)
        if question == "放款接口":
            return [
                {
                    "cm": "WithdrawalsController#withdrawals",
                    "flow_role": "main",
                    "is_http_facade": True,
                    "is_helper": False,
                    "api_doc": "放款",
                    "http_paths": "/withdrawals/withdrawals",
                    "score": 0.4,
                }
            ]
        return [
            {
                "cm": "AmountController#amount",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "限额",
                "score": 0.99,
            },
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "放款",
                "score": 0.4,
            },
        ]

    monkeypatch.setattr(ask_retrieval, "_entry_route_candidates", _cands)
    q = "放款接口中 机构限额校验具体做了什么"
    picked = ask_retrieval.resolve_flow_entry_for_question("demo-loan", q)
    assert picked == "WithdrawalsController#withdrawals", picked
    assert seen, seen
    assert seen[0] == "放款接口"
    assert all("限额" not in s for s in seen)


def test_prefer_flow_entry_api_phrase_beats_quota_controller(monkeypatch):
    from app.agent import ask_retrieval
    from app.agent.ask_retrieval import prefer_flow_entry

    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", lambda *a, **k: "")
    picked = prefer_flow_entry(
        [],
        entry_routes=[
            {
                "cm": "AmountController#amount",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "限额",
                "score": 0.99,
            },
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "放款",
                "http_paths": "/withdrawals/withdrawals",
                "score": 0.4,
            },
        ],
        question="放款接口",
    )
    assert picked == "WithdrawalsController#withdrawals", picked


def test_resolve_flow_entry_long_add_field_sentence(monkeypatch):
    """Long 放款加参句 must still resolve via topic「放款接口」search."""
    from app.agent import ask_retrieval

    q = (
        "帮我在放款接口 增加一个入参 aaaName，校验不能为空 "
        "并落库到demo_b_withdrawals，demo_b_loan_assistance这两张表"
    )
    monkeypatch.setattr(ask_retrieval, "_entry_route_candidates", lambda *a, **k: [])

    def _fake_search(project_id, question, top_k=10, entry_hint=None):
        # Real impl searches topic「放款接口」first; fake honors entry_hint/topic.
        if (entry_hint == "放款接口") or (
            "放款接口" in (question or "") and "aaaName" not in (question or "")
        ):
            return [
                {
                    "cm": "WithdrawalsController#withdrawals",
                    "flow_role": "main",
                    "is_http_facade": True,
                    "is_helper": False,
                    "score": 3.6,
                }
            ]
        return [
            {
                "cm": "RepayController#repay",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "score": 1.9,
            }
        ]

    monkeypatch.setattr(
        ask_retrieval, "_search_controller_facade_routes", _fake_search
    )
    picked = ask_retrieval.resolve_flow_entry_for_question(
        "demo-loan", q, entry_hint="放款接口"
    )
    assert picked == "WithdrawalsController#withdrawals", picked


def test_notice_name_overrides_stale_offline_main():
    from app.agent.ask_retrieval import _infer_flow_role
    from app.analysis.entry_route import classify_flow_role

    assert (
        classify_flow_role(
            kind="Controller",
            class_method="LoanNoticeController#loanNotice",
            method_name="loanNotice",
            existing_role="main",
        )
        == "notice"
    )
    assert (
        _infer_flow_role("LoanNoticeController#loanNotice", explicit="main")
        == "notice"
    )


def test_dig_in_search_query_keeps_gate_nouns():
    from app.agent.ask_retrieval import _dig_in_search_query

    q = "我想看看放款接口 额度校验那一块的逻辑"
    core = _dig_in_search_query(q)
    assert "额度校验" in core
    assert "我想看看" not in core
    assert "那一块" not in core


def test_prefer_main_over_notice_for_interface_logic(monkeypatch):
    """Cold-start「接口+逻辑」→ main facade, not LoanNotice (even if offline role=main)."""
    from app.agent import ask_retrieval
    from app.agent.ask_retrieval import prefer_flow_entry

    # Structural fallback after rerank; skip live LLM in unit test.
    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", lambda *a, **k: "")

    q = "我想看看放款接口 额度校验那一块的逻辑"
    entry_routes = [
        {
            "cm": "LoanNoticeController#loanNotice",
            "flow_role": "main",  # stale offline — Notice* name must demote
            "is_http_facade": True,
            "is_helper": False,
            "http_paths": "/loan/notice",
            "api_doc": "放款通知",
            "score": 0.99,
        },
        {
            "cm": "SignatureController#signature",
            "flow_role": "main",
            "is_http_facade": True,
            "is_helper": False,
            "http_paths": "/sign",
            "score": 0.4,
        },
        {
            "cm": "WithdrawalsController#withdrawals",
            "flow_role": "main",
            "is_http_facade": True,
            "is_helper": False,
            "http_paths": "/withdrawals/withdrawals",
            "api_doc": "放款",
            "routing_text": "WithdrawalsController#withdrawals 放款入口 role=main",
            "score": 0.55,
        },
    ]
    picked = prefer_flow_entry(
        [],
        entry_routes=entry_routes,
        question=q,
    )
    assert picked == "WithdrawalsController#withdrawals", picked
    assert picked != "LoanNoticeController#loanNotice"


def test_prefer_notice_when_question_asks_notice(monkeypatch):
    from app.agent import ask_retrieval
    from app.agent.ask_retrieval import prefer_flow_entry

    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", lambda *a, **k: "")

    picked = prefer_flow_entry(
        [],
        entry_routes=[
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "is_http_facade": True,
                "is_helper": False,
                "score": 0.9,
            },
            {
                "cm": "LoanNoticeController#loanNotice",
                "flow_role": "notice",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "放款通知",
                "score": 0.5,
            },
        ],
        question="放款通知接口做了什么",
    )
    assert picked == "LoanNoticeController#loanNotice", picked


def test_llm_pick_flow_entry_selects_from_candidates(monkeypatch):
    from app.agent import ask_retrieval
    from app.llm.base import LLMProvider

    class _Fake(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return '{"entry":"CreditController#creditNotification","reason":"授信通知"}'

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 4 for _ in texts]

    monkeypatch.setattr(ask_retrieval, "get_llm_provider", lambda: _Fake())
    picked = ask_retrieval._llm_pick_flow_entry(
        "授信通知接口做了什么",
        [
            {
                "cm": "WithdrawalsController#withdrawals",
                "flow_role": "main",
                "api_doc": "放款",
                "score": 0.9,
            },
            {
                "cm": "CreditController#creditNotification",
                "flow_role": "notice",
                "api_doc": "授信通知",
                "score": 0.4,
            },
        ],
    )
    assert picked == "CreditController#creditNotification"


def test_prefer_flow_entry_api_doc_in_question_beats_quota_llm(monkeypatch):
    """Sibling #creditNotification: api_doc⊂问句 wins; LLM must not flip to Quota."""
    from app.agent import ask_retrieval
    from app.agent.ask_retrieval import prefer_flow_entry, prefer_flow_entries

    def _bad_llm(*_a, **_k):
        return "CreditQuotaController#creditNotification"

    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", _bad_llm)
    monkeypatch.setattr(ask_retrieval, "_llm_filter_flow_entries", lambda *_a, **_k: [])
    routes = [
        {
            "cm": "CreditQuotaController#creditNotification",
            "flow_role": "notice",
            "is_http_facade": True,
            "is_helper": False,
            "api_doc": "授信额度通知",
            "score": 0.99,
        },
        {
            "cm": "CreditController#creditNotification",
            "flow_role": "notice",
            "is_http_facade": True,
            "is_helper": False,
            "api_doc": "授信通知",
            "score": 0.4,
        },
    ]
    picked = prefer_flow_entry([], entry_routes=routes, question="授信通知接口")
    assert picked == "CreditController#creditNotification", picked
    # Same-method siblings: both stay in the reply set (primary first).
    both = prefer_flow_entries([], entry_routes=routes, question="授信通知接口")
    assert both[0] == "CreditController#creditNotification"
    assert "CreditQuotaController#creditNotification" in both
    assert len(both) == 2


def test_prefer_flow_entries_keeps_ai_filtered_siblings(monkeypatch):
    """AI filter may keep both #creditNotification Controllers for 授信通知接口."""
    from app.agent import ask_retrieval
    from app.agent.ask_retrieval import prefer_flow_entries

    monkeypatch.setattr(
        ask_retrieval,
        "_llm_filter_flow_entries",
        lambda *_a, **_k: [
            "CreditController#creditNotification",
            "CreditQuotaController#creditNotification",
        ],
    )
    both = prefer_flow_entries(
        [],
        entry_routes=[
            {
                "cm": "CreditQuotaController#creditNotification",
                "flow_role": "notice",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "授信额度通知",
                "score": 0.99,
            },
            {
                "cm": "CreditController#creditNotification",
                "flow_role": "notice",
                "is_http_facade": True,
                "is_helper": False,
                "api_doc": "授信通知",
                "score": 0.4,
            },
        ],
        question="授信通知接口流程是什么",
    )
    assert both == [
        "CreditController#creditNotification",
        "CreditQuotaController#creditNotification",
    ]


def test_prefer_flow_entry_credit_notification_beats_quota_domain(monkeypatch):
    """Real domains: 信用审批通知同步 + summary vs 授信额度通知 (desktop-chat-d667)."""
    from app.agent import ask_retrieval
    from app.agent.ask_retrieval import prefer_flow_entry, prefer_flow_entries

    def _bad_llm(*_a, **_k):
        return "CreditQuotaController#creditNotification"

    monkeypatch.setattr(ask_retrieval, "_llm_pick_flow_entry", _bad_llm)
    monkeypatch.setattr(ask_retrieval, "_llm_filter_flow_entries", lambda *_a, **_k: [])
    credit_rt = (
        "CreditController#creditNotification：处理合作方系统授信后同步通知新系统的授信结果，"
        "包括参数校验、重复授信判断、客户信息校验及调用授信服务。"
    )
    quota_rt = (
        "CreditQuotaController#creditNotification 处理授信额度通知的业务逻辑，"
        "包括参数校验、客户信息检查和信用额度共享操作。"
    )
    routes = [
        {
            "cm": "CreditQuotaController#creditNotification",
            "flow_role": "notice",
            "is_http_facade": True,
            "is_helper": False,
            "api_doc": "授信额度通知",
            "routing_text": quota_rt,
            "score": 58.7,
        },
        {
            "cm": "CreditController#creditNotification",
            "flow_role": "notice",
            "is_http_facade": True,
            "is_helper": False,
            "api_doc": "信用审批通知同步",
            "routing_text": credit_rt,
            "score": 48.7,
        },
    ]
    picked = prefer_flow_entry([], entry_routes=routes, question="授信通知接口做了什么")
    assert picked == "CreditController#creditNotification", picked
    both = prefer_flow_entries([], entry_routes=routes, question="授信通知接口做了什么")
    assert "CreditController#creditNotification" in both
    assert "CreditQuotaController#creditNotification" in both


def test_render_multi_entry_flow_lists_both_credit_notifications():
    from app.agent.pev_agent import _render_multi_entry_flow_answer

    text = _render_multi_entry_flow_answer(
        {
            "entry": "CreditController#creditNotification",
            "entry_matches": [
                "CreditController#creditNotification",
                "CreditQuotaController#creditNotification",
            ],
            "entry_match_meta": [
                {
                    "cm": "CreditController#creditNotification",
                    "api_doc": "授信通知",
                    "http_paths": "/creditNotification",
                },
                {
                    "cm": "CreditQuotaController#creditNotification",
                    "api_doc": "授信额度通知",
                    "http_paths": "/creditQuota/creditNotification",
                },
            ],
            "entry_spines": {
                "CreditController#creditNotification": {
                    "calls": ["CreditServiceImpl#creditNotification"],
                    "spine_source": "neo4j_calls_l1",
                },
                "CreditQuotaController#creditNotification": {
                    "calls": ["CreditQuotaServiceImpl#creditNotification"],
                    "spine_source": "neo4j_calls_l1",
                },
            },
            "entry_mysql_rows": {
                "CreditController#creditNotification": [
                    {
                        "class_method": "CreditServiceImpl#creditNotification",
                        "summary": "授信结果同步主流程",
                    }
                ],
                "CreditQuotaController#creditNotification": [
                    {
                        "class_method": "CreditQuotaServiceImpl#creditNotification",
                        "summary": "额度共享通知主流程",
                    }
                ],
            },
        }
    )
    assert "2 个相关系统入口" in text
    assert "CreditController#creditNotification" in text
    assert "CreditQuotaController#creditNotification" in text
    assert "授信通知" in text
    assert "授信额度通知" in text
    assert "授信结果同步主流程" in text
    assert "额度共享通知主流程" in text


def test_interface_label_score_penalizes_quota_mid_insert():
    from app.analysis.entry_route import interface_label_score

    q = "授信通知接口"
    credit = interface_label_score(
        "信用审批通知同步",
        q,
        extra="处理合作方系统授信后同步通知新系统的授信结果",
    )
    quota = interface_label_score("授信额度通知", q, extra="处理授信额度通知")
    assert credit > quota, (credit, quota)
    assert credit >= 70
    assert interface_label_score("授信通知", q) >= 100


def test_sanitize_keeps_valid_main_not_forced_to_preferred():
    from app.agent.ask_retrieval import _sanitize_neo4j_choice

    out = _sanitize_neo4j_choice(
        {
            "template_id": "follows_spine",
            "params": {
                "entry": "WithdrawalsController#withdrawals",
                "depth": 4,
                "limit": 20,
            },
            "reason": "llm chose main",
        },
        preferred_entry="LoanNoticeController#loanNotice",
        allowed_entries=[
            "LoanNoticeController#loanNotice",
            "WithdrawalsController#withdrawals",
        ],
        question="放款接口额度校验那一块的逻辑",
    )
    assert out["params"]["entry"] == "WithdrawalsController#withdrawals"


def test_sanitize_rewrites_notice_when_question_wants_main():
    from app.agent.ask_retrieval import _sanitize_neo4j_choice

    out = _sanitize_neo4j_choice(
        {
            "template_id": "follows_spine",
            "params": {
                "entry": "LoanNoticeController#loanNotice",
                "depth": 4,
                "limit": 20,
            },
            "reason": "llm wrongly picked notice",
        },
        preferred_entry="WithdrawalsController#withdrawals",
        allowed_entries=[
            "LoanNoticeController#loanNotice",
            "WithdrawalsController#withdrawals",
        ],
        question="放款接口额度校验那一块的逻辑",
    )
    assert out["params"]["entry"] == "WithdrawalsController#withdrawals"


def test_refuse_thin_evidence_gate():
    from app.agent.ask_retrieval import (
        _REFUSE_NO_ENTRY,
        _REFUSE_THIN_EVIDENCE,
        _postcheck_summarize,
        _refuse_if_thin_evidence,
    )

    assert (
        _refuse_if_thin_evidence(
            preferred_entry="",
            nodes=[],
            intersect_cms=[],
            mysql_rows=[],
        )
        == _REFUSE_NO_ENTRY
    )
    assert (
        _refuse_if_thin_evidence(
            preferred_entry="LoanNoticeController#loanNotice",
            nodes=[
                {
                    "order": 1,
                    "class_method": "LoanNoticeController#loanNotice",
                }
            ],
            intersect_cms=[],
            mysql_rows=[
                {
                    "order": 1,
                    "class_method": "LoanNoticeController#loanNotice",
                    "summary": "通知入口",
                }
            ],
        )
        == _REFUSE_THIN_EVIDENCE
    )
    assert (
        _refuse_if_thin_evidence(
            preferred_entry="WithdrawalsController#withdrawals",
            nodes=[
                {"order": 1, "class_method": "WithdrawalsController#withdrawals"},
                {"order": 2, "class_method": "WithdrawalsServiceImpl#loan"},
            ],
            intersect_cms=["WithdrawalsServiceImpl#quotaCheck"],
            mysql_rows=[
                {"order": 1, "summary": "受理"},
                {"order": 2, "summary": "额度校验"},
            ],
        )
        is None
    )

    hallucinated = (
        "1. 受理请求\n2. 参数校验\n3. 开关检查\n4. 执行校验\n5. 返回结果\n"
        "依据：节点数=1"
    )
    assert (
        _postcheck_summarize(hallucinated, mysql_rows=[{"summary": "x"}], allow_tech=False)
        == _REFUSE_THIN_EVIDENCE
    )


def test_ask_pipeline_refuses_when_graph_empty(monkeypatch):
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent import ask_retrieval

    class _Hit:
        symbol_id = "s1"
        fqcn = "com.demo.LoanNoticeController"
        method_sig = "loanNotice()"
        kind = "Controller"
        summary = "LoanNoticeController#loanNotice"
        tags = ["LoanNoticeController#loanNotice"]
        score = 1.0

        def model_dump(self):
            return {
                "symbol_id": self.symbol_id,
                "fqcn": self.fqcn,
                "method_sig": self.method_sig,
                "kind": self.kind,
                "summary": self.summary,
                "tags": self.tags,
                "score": self.score,
            }

    monkeypatch.setattr(
        "app.tagging.search.hybrid_search", lambda *a, **k: [_Hit()]
    )
    monkeypatch.setattr(
        ask_retrieval,
        "_entry_route_candidates",
        lambda *a, **k: [
            {
                "cm": "LoanNoticeController#loanNotice",
                "flow_role": "notice",
                "is_http_facade": True,
                "is_helper": False,
                "score": 1.0,
            }
        ],
    )

    async def _fake_plan(question, **kwargs):
        return {
            "template_id": "follows_spine",
            "params": {
                "entry": "LoanNoticeController#loanNotice",
                "depth": 4,
                "limit": 20,
            },
            "reason": "test",
        }

    monkeypatch.setattr(ask_retrieval, "plan_neo4j_with_llm", _fake_plan)
    monkeypatch.setattr(
        ask_retrieval,
        "_run_neo4j_template",
        lambda project_id, choice: (
            [
                {
                    "order": 1,
                    "class_method": "LoanNoticeController#loanNotice",
                    "rel": "ENTRY",
                    "desc": "",
                }
            ],
            {"ok": True},
        ),
    )
    monkeypatch.setattr(ask_retrieval, "_calls_closure_cms", lambda *a, **k: [])
    monkeypatch.setattr(
        ask_retrieval, "_search_hit_cms_for_question", lambda *a, **k: []
    )
    monkeypatch.setattr(
        ask_retrieval,
        "_mysql_enrich_ordered",
        lambda *a, **k: [
            {
                "order": 1,
                "class_method": "LoanNoticeController#loanNotice",
                "summary": "通知",
            }
        ],
    )

    result = asyncio.run(
        ask_retrieval.run_ask_retrieval_pipeline(
            "我想看看放款接口 额度校验那一块的逻辑",
            project_id="demo",
        )
    )
    assert "证据不足" in (result.get("answer") or "")
    assert (result.get("steps") or [])[-1].get("refused") is True


def test_vector_hit_to_cm_from_indexed_text():
    from app.agent.ask_retrieval import _vector_hit_to_cm

    cm = _vector_hit_to_cm(
        {
            "fqcn": "com.demo.loan.service.impl.CreditServiceImpl",
            "text": (
                "credit 同步会员价格信息 class_method="
                "CreditServiceImpl#synchroMemberPrice method=synchroMemberPrice"
            ),
        }
    )
    assert cm == "CreditServiceImpl#synchroMemberPrice"

    cm2 = _vector_hit_to_cm(
        {
            "fqcn": "com.demo.loan.plugin.CusCbsServiceClient",
            "text": "同步会员价格 method=synchroMemberPriceInfo",
        }
    )
    assert cm2 == "CusCbsServiceClient#synchroMemberPriceInfo"


def test_vector_search_skips_unverified_milvus(monkeypatch):
    from app.agent import ask_retrieval

    class _MemMilvus:
        def verify(self):
            return False

        def search(self, *_a, **_k):
            raise AssertionError("must not search when milvus.verify is false")

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: type("S", (), {"search_use_milvus": True})(),
    )
    monkeypatch.setattr("app.db.milvus_client.get_milvus", lambda: _MemMilvus())
    out = ask_retrieval._vector_search_hit_cms_for_question(
        "demo-loan", "同步会员价格信息", top_k=8
    )
    assert out == []


def test_vector_search_returns_ranked_cms(monkeypatch):
    from app.agent import ask_retrieval

    class _OkMilvus:
        def verify(self):
            return True

        def search(self, *_a, **_k):
            return [
                {
                    "fqcn": "com.demo.loan.service.impl.CreditServiceImpl",
                    "text": "class_method=CreditServiceImpl#synchroMemberPrice",
                    "score": 0.91,
                },
                {
                    "fqcn": "com.other.Unrelated",
                    "text": "class_method=Unrelated#foo",
                    "score": 0.4,
                },
            ]

    class _Emb:
        def embed_one(self, text):
            assert "同步会员价格信息" in text
            return [0.1, 0.2]

    monkeypatch.setattr(
        "app.core.config.get_settings",
        lambda: type("S", (), {"search_use_milvus": True})(),
    )
    monkeypatch.setattr("app.db.milvus_client.get_milvus", lambda: _OkMilvus())
    monkeypatch.setattr("app.llm.factory.get_embedding_provider", lambda: _Emb())
    out = ask_retrieval._vector_search_hit_cms_for_question(
        "demo-loan", "同步会员价格信息", top_k=8
    )
    assert out[0] == "CreditServiceImpl#synchroMemberPrice"
    assert "Unrelated#foo" in out


def test_entry_mysql_fallback_ranks_chinese_api_doc(monkeypatch):
    """纯中文接口问法（用信申请接口）应按 api_doc 中文匹配排序。

    回归 desktop-chat-09dfa4ec7ba7：无拉丁词元时 fallback 按任意 Controller
    顺序返回（用信申请 → WithdrawalsController#withdrawals），
    api_doc 含「用信申请」的 creditUsageApply 反而排不到前面。
    """
    from app.agent import ask_retrieval as ar

    class _Sym:
        def __init__(self, sid, simple, meth, fqcn, kind):
            self.id, self.simple_name, self.method_name = sid, simple, meth
            self.fqcn, self.kind = fqcn, kind

    rows = [
        _Sym("s-withdrawals", "WithdrawalsController", "withdrawals",
             "com.x.WithdrawalsController", "Controller"),
        _Sym("s-creditusage", "WithdrawalsController", "creditUsageApply",
             "com.x.WithdrawalsController", "Controller"),
    ]

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def scalars(self, q):
            return _FakeScalar(rows)

        def execute(self, q):
            return []

    class _FakeScalar:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

        def __iter__(self):
            return iter(self._rows)

    class _FakeDesc:
        def __init__(self, sid, api):
            self.symbol_id = sid
            self.extra_json = {"api_doc": api}
            self.summary = ""

    # session 第一次 scalars 返回 symbol 行；第二次 execute 返回 desc 行
    desc_rows = [
        _FakeDesc("s-creditusage", "用信申请"),
        _FakeDesc("s-withdrawals", "放款提现申请"),
    ]

    calls = {"n": 0}

    def _fake_session():
        calls["n"] += 1
        if calls["n"] == 1:
            return _FakeSession()
        return _DescSession(desc_rows)

    class _DescSession:
        def __init__(self, rows):
            self._rows = rows

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def execute(self, q):
            return _DescScalar(self._rows)

    class _DescScalar:
        def __init__(self, rows):
            self._rows = rows

        def all(self):
            return self._rows

    monkeypatch.setattr("app.db.mysql.get_session", _fake_session)
    monkeypatch.setattr(ar, "is_controller_http_facade", lambda cm: True)
    monkeypatch.setattr(ar, "is_controller_helper_method", lambda cm: False)
    monkeypatch.setattr(ar, "is_system_entry_point", lambda *a, **k: False)

    cands = ar._entry_route_mysql_fallback("demo-loan", "用信申请接口", top_k=5)
    assert cands, cands
    # creditUsageApply（api_doc=用信申请）必须排第一
    assert cands[0]["cm"] == "WithdrawalsController#creditUsageApply", [
        (c["cm"], c["score"]) for c in cands
    ]
