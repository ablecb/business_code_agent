"""Regression tests for quality optimizations A–E:

A. compose_undef_class gap triggers re-search/rewrite (no fabricated code ships)
B. [PEV][TEST] internal tags never leak into the user-visible answer
C. compose retrieval augments queries via table_dict Chinese-table→entity mapping
D. multi-segment continuation inherits the code-generation goal
E. retrieval-fallback focus with zero topic overlap degrades to an honest note
"""
from __future__ import annotations

import asyncio


# ---------------- A: compose_undef_class repair ----------------

def _mk_verify(gap_id: str = "compose_undef_class", desc: str = "") -> dict:
    desc = desc or (
        "生成代码引用了未定义的类（实体/DTO/Mapper 缺失）: LoanCoupon, Withdrawals。"
        "若为知识库接口，应走 add_api 脚手架…"
    )
    return {"ok": False, "gaps": [{"id": gap_id, "desc": desc}]}


def test_pev_compose_pipeline_no_fabricated_code(monkeypatch):
    """End-to-end: compose pipeline must not ship fabricated classes or [PEV] tags.

    Mirrors desktop-chat-1b5c244b4a5e: generated code referenced LoanCoupon
    (not in repo) — the honest repair path must replace it with a gap note and
    the answer must stay free of internal debug tags.
    """
    import asyncio

    from app.agent import pev_agent as pev

    async def _fake_goal(q, *, history_text="", prior_goal=None):  # noqa: ARG001
        return {
            "goal": "compose_code",
            "also": [],
            "must": [{"id": "db_lookup", "desc": "查表"}],
            "view_kind": "db_lookup",
            "lookup_key": "借据号",
            "lookup_table": "放款表",
            "source": "llm",
        }

    async def _fake_compose(project_id, question, evidence, **kw):  # noqa: ARG001
        return "```java\nLoanCoupon c = loanCouponService.list(wrapper);\n```"

    monkeypatch.setattr(pev, "extract_goal_contract", _fake_goal)
    monkeypatch.setattr(pev, "_exec_compose_code", _fake_compose)
    monkeypatch.setattr(pev, "_find_real_symbol_class", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "app.agent.decompose.wants_decompose", lambda *_a, **_k: False
    )

    out = asyncio.run(
        pev.run_pev_pipeline("根据借据号查放款表", project_id="demo-loan")
    )
    ans = out.get("answer") or ""
    assert "LoanCoupon" in ans  # honest note names the missing class
    assert "未能生成可运行的代码" in ans
    assert "```" not in ans  # no fabricated code block ships
    assert "[PEV][TEST]" not in ans
    assert "契约缺口" not in ans
    assert "[PEV][VERIFY]" not in ans
    assert (out.get("pev_debug") or {}).get("footer", "").startswith("\n---\n[PEV][TEST]")


def test_compose_undef_class_names_parser():
    from app.agent.pev_agent import _compose_undef_class_names

    names = _compose_undef_class_names(
        "生成代码引用了未定义的类（实体/DTO/Mapper 缺失）: LoanCoupon, Withdrawals。请重写"
    )
    assert names == ["LoanCoupon", "Withdrawals"]


def test_compose_repair_honest_note_when_no_real_class(monkeypatch):
    """No real class found → honest gap note, never fabricated code."""
    from app.agent import pev_agent as pev

    monkeypatch.setattr(pev, "_find_real_symbol_class", lambda *_a, **_k: None)
    answer, verify = asyncio.run(
        pev._repair_compose_undefined(
            "demo-loan",
            "根据借据号查放款表再查券表",
            {"goal": "compose_code"},
            {},
            "```java\nLoanCoupon c = loanCouponService.list(...);\n```",
            _mk_verify(),
            hist="",
        )
    )
    assert "LoanCoupon" in answer
    assert "未能生成可运行的代码" in answer
    assert "已检索关键词" in answer
    assert "```" not in answer  # no fabricated code block ships
    assert all(g["id"] != "compose_undef_class" for g in verify["gaps"])


def test_compose_repair_rewrites_with_real_class(monkeypatch):
    """Real class found → re-synthesize anchored on it."""
    from app.agent import pev_agent as pev

    monkeypatch.setattr(
        pev, "_find_real_symbol_class", lambda _pid, name: "CouponInfo"
    )
    calls = {}

    async def _fake_compose(project_id, question, **kw):
        calls["lookup_table"] = kw.get("lookup_table")
        return {
            "ok": True,
            "answer": (
                "```java\nCouponInfo c = couponInfoService.list(wrapper);\n```"
            ),
            "snippets": [],
            "refs": ["CouponInfoService#x"],
        }

    monkeypatch.setattr("app.agent.compose_code.run_compose_code", _fake_compose)
    answer, verify = asyncio.run(
        pev._repair_compose_undefined(
            "demo-loan",
            "根据借据号查券表",
            {"goal": "compose_code"},
            {},
            "```java\nLoanCoupon c = loanCouponService.list(...);\n```",
            _mk_verify(),
            hist="",
        )
    )
    assert calls.get("lookup_table") == "CouponInfo"
    assert "CouponInfo" in answer
    assert verify["ok"] is True


# ---------------- C: table_dict query augmentation ----------------

def test_augment_with_table_dict_adds_entity_seeds(monkeypatch):
    from app.agent import source_view as sv

    rows = [
        {
            "table_name": "demo_b_coupon_info",
            "entity_class": "CouponInfo",
            "name_cn": "优惠券信息表",
            "entity_fqcn": "com.demo.loan.entity.CouponInfo",
            "file_path": "x.java",
            "fields": [],
        }
    ]
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict", lambda _pid: rows
    )
    out = sv._augment_with_table_dict(
        "demo-loan",
        ["放款优惠券表"],
        lookup_table="放款优惠券表",
        question="",
    )
    assert "CouponInfo" in out
    assert "demo_b_coupon_info" in out
    assert "CouponInfoService" in out


def test_augment_with_table_dict_no_phrase_noop(monkeypatch):
    from app.agent import source_view as sv

    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict", lambda _pid: []
    )
    out = sv._augment_with_table_dict("demo-loan", ["foo"], lookup_table="", question="")
    assert out == ["foo"]


# ---------------- D: continuation goal inheritance ----------------

def test_inherit_continuation_goal_keeps_compose():
    from app.agent.pev_plan import _inherit_continuation_goal

    base = {"goal": "general"}
    out = _inherit_continuation_goal(
        base, "用放款表的记录查券表，查到后调用外部接口", "compose_code"
    )
    assert out["goal"] == "compose_code"


def test_inherit_continuation_goal_flow_switch_not_inherited():
    from app.agent.pev_plan import _inherit_continuation_goal

    base = {"goal": "explain_flow"}
    out = _inherit_continuation_goal(base, "放款完整流程是什么", "compose_code")
    assert out["goal"] == "explain_flow"


# ---------------- E: zero-overlap fallback guard ----------------

def test_focus_zero_overlap_client_with_summary_demoted():
    from app.agent.pev_flow import _focus_zero_overlap

    evidence = {
        "mysql_rows": [
            {
                "class_method": "ApiOrgServiceClient#queryProductInfos",
                "summary": "查询产品信息列表并返回产品数据",
            }
        ]
    }
    assert _focus_zero_overlap(
        "ApiOrgServiceClient#queryProductInfos",
        "调用外部接口对券进行支用",
        evidence,
    ) is True


def test_focus_zero_overlap_matching_summary_kept():
    from app.agent.pev_flow import _focus_zero_overlap

    evidence = {
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "同步会员价格信息",
            }
        ]
    }
    assert _focus_zero_overlap(
        "CreditServiceImpl#synchroMemberPrice",
        "同步会员价格信息 怎么实现",
        evidence,
    ) is False


def test_focus_zero_overlap_summaryless_leaf_kept():
    """Summary-less ServiceImpl leaf is read before rendering — do not demote."""
    from app.agent.pev_flow import _focus_zero_overlap

    evidence = {"mysql_rows": []}
    assert _focus_zero_overlap(
        "CreditServiceImpl#synchroMemberPrice",
        "同步会员价格信息 怎么实现",
        evidence,
    ) is False


# ---------------- add-api story: single ask, not split ----------------

def test_add_api_story_not_split_into_steps():
    """新增接口 + 业务步骤编排 = 一个整体需求（防止 4 段重复脚手架）。"""
    from app.agent.runtime.planner import wants_decompose
    from app.agent.runtime.turn_plan import heuristic_turn_plan, split_turn_stages

    q = (
        "我想新增一个接口，接口具体文档如 知识库商品查询，"
        "service中要查询放款表，然后再查询优惠券表，再调用acc-service，然后组装代码返回"
    )
    assert split_turn_stages(q) == [q]
    assert len(heuristic_turn_plan(q).nodes) == 1
    assert wants_decompose(q) is False


def test_add_api_story_still_splits_truly_multi_question():
    """真正的多问题（编号换行列表）不受影响，仍拆分。"""
    from app.agent.runtime.turn_plan import split_turn_stages

    q = "1. 新增一个查询接口\n2. 还款流程是什么"
    stages = split_turn_stages(q)
    assert len(stages) == 2


def test_change_plan_story_not_split_loses_field_name():
    """加字段 + 落库/改动清单 = 单需求（防止 N2 退化成 newField 占位方案）。"""
    from app.agent.runtime.turn_plan import heuristic_turn_plan, split_turn_stages

    q = "我想在放款接口加个字段 labor 然后并落到放款表 给我对应的代码改动清单"
    assert split_turn_stages(q) == [q]
    assert len(heuristic_turn_plan(q).nodes) == 1


# ---------------- entry resolution: symbol-search fallback ----------------

def test_resolve_entry_via_symbol_search_loanswitch(monkeypatch):
    """Bare method-token ask (loanSwitch干嘛的) resolves via symbol search."""
    from app.agent import pev_flow

    hits = [
        {
            "fqcn": "com.demo.loan.service.impl.WithdrawalsServiceImpl",
            "method_name": "loanSwitch",
            "method_sig": "WithdrawalsServiceImpl#loanSwitch",
            "summary": "放款开关判断是否进入异步流程",
        }
    ]

    def _fake_search(project_id, q, **kw):
        return {"ok": True, "hits": hits}

    monkeypatch.setattr("app.agent.explore_tools.search_symbols", _fake_search)
    best = pev_flow._resolve_entry_via_symbol_search("demo-loan", "loanSwitch干嘛的")
    assert best is not None
    cm, score = best
    assert cm == "WithdrawalsServiceImpl#loanSwitch"
    assert score >= 4.0


def test_resolve_entry_via_symbol_search_unrelated_returns_none(monkeypatch):
    """No literal/topic match → None (honest empty stays)."""
    from app.agent import pev_flow

    monkeypatch.setattr(
        "app.agent.explore_tools.search_symbols",
        lambda *a, **k: {"ok": True, "hits": []},
    )
    best = pev_flow._resolve_entry_via_symbol_search("demo-loan", "随便聊聊")
    assert best is None


# ---------------- session gate: conservative keep on timeout ----------------

def test_decide_continue_locked_entry_timeout_conservative_keep(monkeypatch):
    """Timeout with high prior-answer overlap keeps the lock (no drift)."""
    import asyncio

    from app.agent import graph_agent as ga

    async def _timeout_llm(*a, **k):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(ga, "get_llm_provider", lambda: type(
        "F", (), {"chat": _timeout_llm})())
    ctx = {
        "last_class_method": "WithdrawalsController#withdrawals",
        "last_answer_steps": [
            {"title": "放款落库", "detail": "写入放款表并更新状态", "class_method": "WithdrawalsServiceImpl#insertLoanData"},
            {"title": "状态流转", "detail": "放款成功状态变更", "class_method": "WithdrawalsController#applyResultDealWith"},
        ],
    }
    decision = asyncio.run(
        ga.decide_continue_locked_entry("放款落库写哪", "WithdrawalsController#withdrawals", session_ctx=ctx)
    )
    assert decision.get("continue") is True
    assert "conservative" in str(decision.get("source") or "")


def test_decide_continue_locked_entry_timeout_unlocks_when_no_overlap(monkeypatch):
    """Timeout with zero overlap still unlocks (fail-open safety preserved)."""
    import asyncio

    from app.agent import graph_agent as ga

    async def _timeout_llm(*a, **k):
        raise asyncio.TimeoutError()

    monkeypatch.setattr(ga, "get_llm_provider", lambda: type(
        "F", (), {"chat": _timeout_llm})())
    ctx = {"last_class_method": "WithdrawalsController#withdrawals", "last_answer_steps": []}
    decision = asyncio.run(
        ga.decide_continue_locked_entry("今天天气怎么样", "WithdrawalsController#withdrawals", session_ctx=ctx)
    )
    assert decision.get("continue") is False


def test_strip_answer_chrome_drops_index_footer():
    from app.agent.explore_answers import strip_answer_chrome

    raw = (
        "业务入口：`WithdrawalsController#withdrawals`\n"
        "主链（按执行序 / FOLLOWS 或有序 CALLS）：\n\n"
        "1. `WithdrawalsServiceImpl#loan` — 落库\n\n"
        "依据：节点数=3；主链来源=neo4j_follows\n"
    )
    out = strip_answer_chrome(raw)
    assert "依据：" not in out
    assert "FOLLOWS" not in out
    assert "主链：" in out
    assert "WithdrawalsServiceImpl#loan" in out
