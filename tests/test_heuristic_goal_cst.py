"""Regression: degraded path (Goal LLM down) must still route flow/change asks.

Bug from desktop-chat-0c3000119343: with the LLM unavailable the Goal LLM
fell back to heuristic_goal which was always ``general``, so
「结清证明申请接口 主要流程是什么」 was mis-routed to an unrelated
QuotaRisk processor and the answer said "无法回答". heuristic_goal now uses
structural ask shapes (流程/接口/源码/改法/表字段) to seed a sensible goal.
"""
from __future__ import annotations

import asyncio

from app.agent.pev_plan import heuristic_goal


def test_flow_ask_routes_explain_flow():
    g = heuristic_goal("结清证明申请接口 主要流程是什么")
    assert g["goal"] == "explain_flow"
    assert g.get("ask_scope") == "full_flow"
    assert g.get("entry_hint")
    ids = [m.get("id") for m in g.get("must") or []]
    assert "entry" in ids and "spine" in ids and "summarize" in ids


def test_flow_ask_no_entry_hint_for_bare_topic():
    g = heuristic_goal("某业务接口具体做了什么")
    assert g["goal"] == "explain_flow"
    assert g.get("ask_scope") == "full_flow"


def test_add_field_change_routes_change_plan():
    g = heuristic_goal(
        "放款接口增加一个入参 businessLabel(业务标签) 并落库到放款表 给我修改清单和ddl"
    )
    assert g["goal"] == "change_plan"
    assert any(m.get("id") == "render" for m in g.get("must") or [])


def test_show_code_routes_show_code():
    g = heuristic_goal("我想看看 CreditServiceImpl#creditNotification 的代码")
    assert g["goal"] == "show_code"
    assert g.get("view_kind") == "method"
    assert g.get("entry_hint") == "CreditServiceImpl#creditNotification"


def test_api_io_ask_routes_api_io():
    g = heuristic_goal("该接口请求参数有哪些 响应参数有哪些")
    assert g["goal"] == "api_io"


def test_api_inventory_ask_routes_inventory():
    """「XX有哪些接口」必须确定性路由到 inventory(api)，LLM 挂了也一样
    （desktop-chat-5b263372926e：Goal LLM 超时后退化成 general，只答了
    preapproveApply 一个入口，漏掉 SignatureController 其余端点）。"""
    g = heuristic_goal("结清证明申请有哪些接口")
    assert g["goal"] == "inventory", g
    assert g.get("inventory_kind") == "api"
    assert any(m.get("id") == "list" for m in g.get("must") or [])


def test_api_inventory_variants_routes_inventory():
    for q in (
        "放款接口列表",
        "有哪些接口",
        "相关接口有几个",
        "帮我看看结清证明接口有哪些",
    ):
        g = heuristic_goal(q)
        assert g["goal"] == "inventory", (q, g)


def test_downstream_apis_not_inventory():
    """「调用了哪些下游接口」是链路问题，不是 Controller 端点清单。"""
    g = heuristic_goal("放款流程中调用了哪些下游接口")
    assert g["goal"] != "inventory", g
    # 链路问法会被入口/下游护栏转成 explain_flow
    assert g["goal"] == "explain_flow", g


def test_table_list_routes_explain_flow():
    g = heuristic_goal("放款接口都涉及哪些表")
    assert g["goal"] == "explain_flow"
    assert any(m.get("id") == "tables" for m in g.get("must") or [])


def test_non_code_general_stays_general():
    g = heuristic_goal("什么是数据库事务")
    assert g["goal"] == "general"


def test_degraded_extract_goal_contract_uses_heuristic(monkeypatch):
    """Goal LLM down → extract_goal_contract falls back to enhanced heuristic."""
    from app.llm.base import LLMProvider

    class _DownLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            raise RuntimeError("LLM service unavailable (simulated)")

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.llm.factory.get_llm_provider", lambda: _DownLLM()
    )
    from app.agent.pev_plan import extract_goal_contract

    out = asyncio.run(extract_goal_contract("结清证明申请接口 主要流程是什么"))
    assert out["goal"] == "explain_flow", out
    assert out.get("source") == "heuristic"
    assert out.get("ask_scope") == "full_flow"
    assert out.get("entry_hint")
