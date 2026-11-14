"""Regression test: show_code / dig-in view_hint must pin the display target
even when it is a non-Controller (ServiceImpl) method.

Bug from desktop-chat-e47d3b34dba4 TURN 3/4: user asked to see
``if (!synchroMemberPrice(creditNotification, errorRespMap)) { return ... }``
method internals; the view_hint (CreditServiceImpl#creditNotification /
#synchroMemberPrice) was NOT pinned because _pinnable_controller_entry only
accepts *Controller facades, so free search drifted to an unrelated
``ProductServiceImpl#queryProductModuleDetail`` (product feign decorator) and
the answer showed the wrong entry.
"""
from __future__ import annotations

import asyncio

from app.agent.pev_flow import _exec_resolve_entry


def _evidence(**kw) -> dict:
    base = {
        "entry_hint": "",
        "view_hint": "",
        "prefer_cms": [],
        "lock_entry": False,
        "dig_in_focus": False,
        "dig_in_likely": False,
        "entry": "",
        "entry_matches": [],
        "entry_match_meta": [],
    }
    base.update(kw)
    return base


def test_show_code_view_hint_pins_service_impl_entry():
    """view_hint=CreditServiceImpl#creditNotification (ServiceImpl, not Controller)
    must be pinned as the display target under show_code/dig-in."""
    ev = _evidence(
        entry_hint="",
        view_hint="CreditServiceImpl#creditNotification",
        lock_entry=False,
        dig_in_focus=True,
    )
    asyncio.run(
        _exec_resolve_entry("demo-loan", "我想看看这一块代码", ev)
    )
    assert ev["entry"] == "CreditServiceImpl#creditNotification", ev
    assert ev["entry_matches"] == ["CreditServiceImpl#creditNotification"]


def test_show_code_view_hint_synchro_member_price():
    """TURN 4 shape: view_hint points at the leaf method being dug into."""
    ev = _evidence(
        entry_hint="CreditServiceImpl#synchroMemberPrice",
        view_hint="CreditServiceImpl#synchroMemberPrice",
        dig_in_likely=True,
    )
    asyncio.run(
        _exec_resolve_entry(
            "demo-loan",
            "if (!synchroMemberPrice(creditNotification, errorRespMap)) { return errorRespMap; } 这一块代码 方法里面是做什么的",
            ev,
        )
    )
    assert ev["entry"] == "CreditServiceImpl#synchroMemberPrice", ev


def test_controller_view_hint_still_pins_controller():
    """A Controller view_hint still goes through the Controller pin path."""
    ev = _evidence(
        view_hint="CreditController#creditNotification",
        lock_entry=True,
    )
    asyncio.run(
        _exec_resolve_entry("demo-loan", "授信通知接口什么流程", ev)
    )
    assert ev["entry"] == "CreditController#creditNotification", ev


def test_no_view_hint_free_search_no_pin():
    """No view_hint → no forced pin; entry resolved via search (may be empty
    when nothing matches, but must NOT crash or pin a ghost)."""
    ev = _evidence(entry_hint="", view_hint="")
    asyncio.run(
        _exec_resolve_entry(
            "demo-loan", "授信通知接口什么流程", ev
        )
    )
    # free search should land on a real Credit entry, not Product feign decorator
    entry = ev.get("entry") or ""
    assert entry != "ProductServiceImpl#queryProductModuleDetail", ev
    assert "Product" not in entry, ev


# ---------- _exec_source_view session-drift guard ----------

def test_source_view_uses_view_hint_over_stale_session_entry(monkeypatch):
    """Session entry is a stale unrelated method (Product feign decorator) but
    the question names a view target — run_source_view must receive the view
    target as entry_cm, not the stale lock."""
    import app.agent.pev_change as pc

    captured: dict[str, object] = {}

    async def _fake_run_source_view(project_id, question, **kwargs):
        captured["entry_cm"] = kwargs.get("entry_cm")
        captured["preferred"] = kwargs.get("preferred_cms")
        return {"ok": True, "answer": "fake source"}

    monkeypatch.setattr(
        "app.agent.source_view.run_source_view", _fake_run_source_view
    )
    ev = {
        "entry": "ProductServiceImpl#queryProductModuleDetail",  # stale lock
        "view_hint": "CreditServiceImpl#creditNotification",
        "prefer_cms": [],
        "view_kind": "method",
        "lock_entry": False,
        "dig_in_focus": True,
        "lookup_key": None,
        "lookup_table": None,
        "lookup_steps": None,
    }
    ans = asyncio.run(
        pc._exec_source_view(
            "demo-loan",
            "我想看看这一块代码",
            ev,
        )
    )
    assert ans == "fake source"
    # stale Product entry must NOT leak into run_source_view
    assert captured["entry_cm"] == "CreditServiceImpl#creditNotification", captured
    assert "ProductServiceImpl" not in str(captured["entry_cm"])


def test_source_view_keeps_session_entry_when_no_view_hint(monkeypatch):
    """No view_hint → session entry stays (free search context)."""
    import app.agent.pev_change as pc

    captured: dict[str, object] = {}

    async def _fake_run_source_view(project_id, question, **kwargs):
        captured["entry_cm"] = kwargs.get("entry_cm")
        return {"ok": True, "answer": "fake"}

    monkeypatch.setattr(
        "app.agent.source_view.run_source_view", _fake_run_source_view
    )
    ev = {
        "entry": "CreditController#creditNotification",
        "view_hint": "",
        "prefer_cms": [],
        "view_kind": "method",
        "lock_entry": False,
        "lookup_key": None,
        "lookup_table": None,
        "lookup_steps": None,
    }
    asyncio.run(pc._exec_source_view("demo-loan", "看看代码", ev))
    assert captured["entry_cm"] == "CreditController#creditNotification", captured


# ---------- _seed_reuse_locked_spine session-drift guard ----------

def test_seed_reuse_prefers_view_hint_over_stale_lock():
    """TURN 4 shape: locked entry is a stale Product feign decorator but the
    dig-in names synchroMemberPrice — spine reuse must target the view_hint."""
    from app.agent.pev_flow import _seed_reuse_locked_spine

    ev = {
        "lock_entry": True,
        "entry": "ProductServiceImpl#queryProductModuleDetail",
        "entry_hint": "ProductServiceImpl#queryProductModuleDetail",
        "view_hint": "CreditServiceImpl#synchroMemberPrice",
        "flow_symbols": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#assemParam",
        ],
        "answer_steps": [],
        "calls": [],
        "mysql_rows": [],
    }
    assert _seed_reuse_locked_spine(ev) is True
    assert ev["entry"] == "CreditServiceImpl#synchroMemberPrice", ev
    # view target must be first on the spine candidates
    calls = ev.get("calls") or []
    assert calls[0] == "CreditServiceImpl#synchroMemberPrice", calls
    assert "ProductServiceImpl" not in str(ev.get("entry"))


def test_seed_reuse_keeps_lock_without_view_hint():
    from app.agent.pev_flow import _seed_reuse_locked_spine

    ev = {
        "lock_entry": True,
        "entry": "CreditServiceImpl#creditNotification",
        "entry_hint": "CreditServiceImpl#creditNotification",
        "view_hint": "",
        "flow_symbols": [
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#assemParam",
        ],
        "answer_steps": [],
        "calls": [],
        "mysql_rows": [],
    }
    assert _seed_reuse_locked_spine(ev) is True
    assert ev["entry"] == "CreditServiceImpl#creditNotification", ev
