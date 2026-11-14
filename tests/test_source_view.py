"""Generic AI source-view: plan target → fetch file/method."""
from __future__ import annotations

import asyncio
import re

from app.agent.source_view import (
    build_source_view_context,
    fetch_source_view,
)


def test_build_context_includes_steps_and_symbols():
    ctx = build_source_view_context(
        user_question="我想看看 6中对应代码怎么写的",
        history_text="user: 放款接口都做了啥",
        answer_steps=[
            {"index": 6, "title": "放款处理", "detail": "含风控"},
        ],
        flow_symbols=["WithdrawalsController#withdrawals", "WithdrawalsServiceImpl#loan"],
        entry_cm="WithdrawalsController#withdrawals",
        search_hits=["WithdrawalsServiceImpl#loan"],
    )
    assert "放款处理" in ctx
    assert "WithdrawalsServiceImpl#loan" in ctx
    assert "6中对应代码" in ctx
    assert "检索命中" in ctx


def test_fetch_method_plan_reads_file():
    plan = {
        "view_kind": "method",
        "class_method": "WithdrawalsServiceImpl#loan",
        "reason": "unit-test",
    }
    out = fetch_source_view("demo-loan", plan)
    assert out["ok"] is True
    assert out["target"] == "WithdrawalsServiceImpl#loan"
    assert "```java" in out["answer"]


def test_fetch_file_plan_reads_controller():
    plan = {
        "view_kind": "file",
        "fqcn": "WithdrawalsController",
        "reason": "unit-test",
    }
    out = fetch_source_view("demo-loan", plan)
    assert out["ok"] is True
    assert out["target"] == "WithdrawalsController"
    assert "```java" in out["answer"]
    assert len(out["answer"]) > 5000


def test_refine_prefers_search_hit_over_wrong_llm_pick():
    from app.agent.source_view import refine_source_view_plan

    plan = {
        "view_kind": "method",
        "class_method": "WithdrawalsController#checkApplyReqParam",
        "reason": "llm drift",
        "ok": True,
    }
    symbols = [
        "WithdrawalsController#checkApplyReqParam",
        "WithdrawalsController#withdrawals",
        "WithdrawalsServiceImpl#quotaCheck",
        "WithdrawalsServiceImpl#orgQuotaCheck",
    ]
    hits = [
        "WithdrawalsServiceImpl#quotaCheck",
        "WithdrawalsServiceImpl#orgQuotaCheck",
    ]
    out = refine_source_view_plan(
        "我想看看 放款接口流程中 额度校验那一块的代码",
        plan,
        symbols,
        search_hits=hits,
    )
    assert out["class_method"] == "WithdrawalsServiceImpl#quotaCheck"
    assert out.get("refined") is True


def test_refine_keeps_intent_hint_over_unrelated_search_top1():
    """Dig-in / intent Class#method must not be replaced by QueueHelper#submit-style hits."""
    from app.agent.source_view import refine_source_view_plan

    plan = {
        "view_kind": "method",
        "class_method": "CreditServiceImpl#synchroMemberPrice",
        "reason": "intent dig-in",
        "ok": True,
    }
    symbols = [
        "CreditController#creditNotification",
        "CreditServiceImpl#synchroMemberPrice",
        "QueueHelper#submit",
        "CreditServiceImpl#saveCredit",
    ]
    hits = [
        "QueueHelper#submit",
        "SomeUnrelated#foo",
        "CreditServiceImpl#synchroMemberPrice",
    ]
    out = refine_source_view_plan(
        "这个方法仔细介绍一下 或者代码发出来也可以",
        plan,
        symbols,
        search_hits=hits,
        preferred_cms=["CreditServiceImpl#synchroMemberPrice"],
    )
    assert out["class_method"] == "CreditServiceImpl#synchroMemberPrice"
    assert out.get("refined") is not True


def test_refine_keeps_latin_method_token_without_preferred():
    from app.agent.source_view import refine_source_view_plan

    plan = {
        "view_kind": "method",
        "class_method": "CreditServiceImpl#synchroMemberPrice",
        "reason": "llm",
        "ok": True,
    }
    out = refine_source_view_plan(
        "please paste synchroMemberPrice source",
        plan,
        ["CreditServiceImpl#synchroMemberPrice", "QueueHelper#submit"],
        search_hits=["QueueHelper#submit"],
    )
    assert out["class_method"] == "CreditServiceImpl#synchroMemberPrice"
    assert out.get("refined") is not True


def test_refine_restores_preferred_when_llm_picks_unrelated_hit():
    from app.agent.source_view import refine_source_view_plan

    plan = {
        "view_kind": "method",
        "class_method": "QueueHelper#submit",
        "reason": "search drift",
        "ok": True,
    }
    out = refine_source_view_plan(
        "这个方法仔细介绍一下 或者代码发出来也可以",
        plan,
        [
            "CreditServiceImpl#synchroMemberPrice",
            "QueueHelper#submit",
            "CreditController#creditNotification",
        ],
        search_hits=["QueueHelper#submit", "CreditServiceImpl#synchroMemberPrice"],
        preferred_cms=["CreditServiceImpl#synchroMemberPrice"],
    )
    assert out["class_method"] == "CreditServiceImpl#synchroMemberPrice"
    assert out.get("refined") == "prefer_focus"


def test_prioritize_puts_preferred_before_search_hits():
    from app.agent.source_view import prioritize_symbols_for_question

    ranked = prioritize_symbols_for_question(
        "这个方法代码发出来",
        [
            "CreditController#creditNotification",
            "QueueHelper#submit",
            "CreditServiceImpl#synchroMemberPrice",
        ],
        search_hits=["QueueHelper#submit"],
        preferred_cms=["CreditServiceImpl#synchroMemberPrice"],
    )
    assert ranked[0] == "CreditServiceImpl#synchroMemberPrice"


def test_prioritize_puts_search_hits_first():
    from app.agent.source_view import prioritize_symbols_for_question

    ranked = prioritize_symbols_for_question(
        "额度校验那一块的代码",
        [
            "WithdrawalsController#checkApplyReqParam",
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsServiceImpl#quotaCheck",
        ],
        search_hits=["WithdrawalsServiceImpl#quotaCheck"],
    )
    assert ranked[0] == "WithdrawalsServiceImpl#quotaCheck"


def test_run_source_view_respects_preferred_over_search(monkeypatch):
    """preferred dig-in target wins even when search top1 is unrelated."""
    import asyncio

    from app.agent import source_view as sv

    async def _fake_plan(**kwargs):
        return {
            "view_kind": "method",
            "class_method": "QueueHelper#submit",
            "ok": True,
            "reason": "search top1",
        }

    monkeypatch.setattr(sv, "plan_source_view_target", _fake_plan)
    monkeypatch.setattr(
        sv,
        "search_symbol_hits_for_question",
        lambda *a, **k: ["QueueHelper#submit"],
    )
    monkeypatch.setattr(sv, "_expand_symbol_candidates", lambda *a, **k: [
        "CreditServiceImpl#synchroMemberPrice",
        "QueueHelper#submit",
    ])

    def _fake_fetch(project_id, plan, **kwargs):
        cm = plan.get("class_method") or ""
        if "synchroMemberPrice" in str(cm):
            return {
                "ok": True,
                "answer": f"`{cm}` 方法源码如下：\n\n```java\nvoid synchroMemberPrice(){{}}\n```",
                "target": cm,
                "view_kind": "method",
            }
        return {"ok": False, "answer": "no", "target": cm}

    monkeypatch.setattr(sv, "fetch_source_view", _fake_fetch)
    out = asyncio.run(
        sv.run_source_view(
            "demo-loan",
            "这个方法仔细介绍一下 或者代码发出来也可以",
            flow_symbols=["CreditServiceImpl#synchroMemberPrice"],
            entry_cm="CreditController#creditNotification",
            preferred_cms=["CreditServiceImpl#synchroMemberPrice"],
        )
    )
    assert out["ok"] is True
    assert out["target"] == "CreditServiceImpl#synchroMemberPrice"
    assert "synchroMemberPrice" in out["answer"]
    assert "QueueHelper" not in (out.get("target") or "")


def test_run_source_view_fallback_step_without_llm_ok(monkeypatch):
    from app.agent import source_view as sv

    async def _fake_plan(**kwargs):
        return {"view_kind": "unknown", "ok": False, "reason": "fake"}

    monkeypatch.setattr(sv, "plan_source_view_target", _fake_plan)
    monkeypatch.setattr(sv, "search_symbol_hits_for_question", lambda *a, **k: [])
    out = asyncio.run(
        sv.run_source_view(
            "demo-loan",
            "我想看看 6中对应代码怎么写的",
            answer_steps=[
                {
                    "index": 6,
                    "title": "放款处理",
                    "detail": "",
                    "class_method": "WithdrawalsServiceImpl#loan",
                }
            ],
            flow_symbols=[
                "WithdrawalsController#withdrawals",
                "WithdrawalsServiceImpl#loan",
            ],
            entry_cm="WithdrawalsController#withdrawals",
        )
    )
    assert out["ok"] is True
    assert out["target"] == "WithdrawalsServiceImpl#loan"


def test_lookup_search_queries_from_llm_slots_only():
    from app.agent.source_view import _lookup_search_queries

    qs = _lookup_search_queries(
        question="给我一段通过借据号查询放款表的java代码",
        lookup_key="借据号",
        lookup_table="放款表",
    )
    assert qs[0] == "放款表 借据号"
    assert "放款表" in qs
    assert "借据号" in qs


def test_build_db_lookup_notes_includes_steps():
    from app.agent.source_view import build_db_lookup_notes

    notes = build_db_lookup_notes(
        view_kind="db_pipeline",
        lookup_steps=[
            {"op": "select", "table": "放款表", "key": "借据号"},
            {"op": "group_by", "fields": ["产品"]},
        ],
    )
    assert "db_pipeline" in notes
    assert "放款表" in notes
    assert "group_by" in notes


def test_run_source_view_db_lookup_uses_slot_search_not_session_prefer(monkeypatch):
    """db_lookup must search by Goal slots and ignore session Controller prefer."""
    from app.agent import source_view as sv

    async def _fake_plan(**kwargs):
        hits = kwargs.get("search_hits") or []
        assert "WithdrawalsServiceImpl#queryProductByDealRefNO" in hits
        assert not kwargs.get("preferred_cms")
        return {
            "view_kind": "method",
            "ask_view_kind": "db_lookup",
            "class_method": "WithdrawalsServiceImpl#queryProductByDealRefNO",
            "reason": "db slot hit",
            "ok": True,
        }

    monkeypatch.setattr(sv, "plan_source_view_target", _fake_plan)
    monkeypatch.setattr(
        sv,
        "search_db_lookup_hits",
        lambda *a, **k: ["WithdrawalsServiceImpl#queryProductByDealRefNO"],
    )
    monkeypatch.setattr(
        sv,
        "search_symbol_hits_for_question",
        lambda *a, **k: ["RepayController#queryCancelWithdrawals"],
    )
    out = asyncio.run(
        sv.run_source_view(
            "demo-loan",
            "给我一段通过借据号查询放款表的java代码",
            entry_cm="WithdrawalsController#withdrawals",
            preferred_cms=None,
            view_kind="db_lookup",
            lookup_key="借据号",
            lookup_table="放款表",
            flow_symbols=["RepayController#queryCancelWithdrawals"],
        )
    )
    assert out["ok"] is True
    assert out["target"] == "WithdrawalsServiceImpl#queryProductByDealRefNO"
    assert "queryProductByDealRefNO" in out["answer"]


# ---------- compose_code evidence quality (desktop-chat-b58bb8b77eea) ----------

def test_augment_with_table_dict_maps_lookup_steps():
    """Multi-step lookup pipeline: each step's Chinese table → real entity/Service.

    回归 desktop-chat-b58bb8b77eea：只有主表映射，子表（放款优惠券表）没映射 →
    LLM 编造 LoanCoupon 占位符。现在 lookup_steps 的每个 table 都走表字典。
    """
    from app.agent.source_view import _augment_with_table_dict

    steps = [
        {"op": "select", "table": "放款表", "key": "order_id"},
        {"op": "join", "table": "放款优惠券表", "key_from": "loan_id"},
    ]
    qs = _augment_with_table_dict(
        "demo-loan",
        ["放款表 order_id"],
        lookup_table="放款表",
        lookup_steps=steps,
        question="通过order_id查询放款表 再查询放款优惠券表",
    )
    joined = " ".join(qs)
    # CouponInfo / CouponInfoService / 物理表名 must be injected for the step table
    assert "CouponInfo" in joined, qs
    assert "demo_b_coupon_info" in joined, qs


def test_search_db_lookup_hits_prefers_implementations():
    """compose 证据检索优先 ServiceImpl/Mapper — Controller 适配器沉底。

    回归 desktop-chat-b58bb8b77eea：命中的全是 Controller HTTP 适配方法
    （checkApplyReqParam），LLM 没有实体证据只能编占位符。
    """
    from app.agent.source_view import search_db_lookup_hits

    hits = search_db_lookup_hits(
        "demo-loan",
        "给我一段 通过order_id查询放款表，查到结果后，再查询放款优惠券表和放款审批表的代码",
        lookup_key="order_id",
        lookup_table="放款表",
        lookup_steps=[
            {"op": "select", "table": "放款表", "key": "order_id"},
            {"op": "join", "table": "放款优惠券表", "key_from": "loan_id"},
        ],
        top_k=20,
    )
    assert hits, hits
    top = hits[0]
    # implementation leaf (ServiceImpl/Mapper/Service) should be first
    assert re.search(
        r"(?i)(ServiceImpl|Mapper|Service|Repository|Dao|Client|Listener|Job|Handler|Processor)",
        top.split("#", 1)[0],
    ), (top, hits[:5])
