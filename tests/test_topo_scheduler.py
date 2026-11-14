"""Unit tests for topological priority scheduler (no LLM / DB)."""

from __future__ import annotations

import asyncio

from app.analysis.topo_scheduler import (
    NodeStatus,
    Priority,
    ScheduleNode,
    SchedulerContext,
    TopoPriorityScheduler,
    break_cycles,
    build_schedule_nodes_from_ast,
    classify_priority,
)


def test_classify_priority():
    assert classify_priority(kind="Entity", simple_name="LoanDO") == Priority.HIGH
    assert classify_priority(kind="Mapper", simple_name="OrderMapper") == Priority.HIGH
    assert classify_priority(kind="Util", simple_name="StringUtil") == Priority.NORMAL
    assert classify_priority(kind="Controller", simple_name="FooController") == Priority.LOW


def test_cycle_break():
    nodes = {
        "A": ScheduleNode(id="A", kind="method", priority=Priority.NORMAL, deps={"B"}),
        "B": ScheduleNode(id="B", kind="method", priority=Priority.NORMAL, deps={"A"}),
        "C": ScheduleNode(id="C", kind="method", priority=Priority.LOW, deps={"A"}),
    }
    nodes["A"].callers.add("B")
    nodes["A"].callers.add("C")
    nodes["B"].callers.add("A")
    warns: list[str] = []
    n = break_cycles(nodes, warnings=warns)
    assert n == 1
    assert nodes["A"].in_cycle and nodes["B"].in_cycle
    assert "B" in nodes["A"].soft_deps or "A" in nodes["B"].soft_deps
    assert "A" in nodes["C"].deps  # C outside cycle keeps hard dep


def test_topo_order_and_priority():
    """Leaves first; among ready nodes HIGH before LOW; callee results passed up."""
    order: list[str] = []

    async def analyze(node: ScheduleNode, ctx: SchedulerContext):
        order.append(node.id)
        callees = ctx.callee_results(node.id)
        return {"summary": node.id, "callee_keys": sorted(callees.keys())}

    nodes = build_schedule_nodes_from_ast(
        method_callees={
            "Ctrl#entry": ["Svc#work"],
            "Svc#work": ["Util#help"],
            "Util#help": [],
        },
        method_meta={
            "Ctrl#entry": {
                "symbol_kind": "Controller",
                "simple_name": "Ctrl",
                "method_name": "entry",
            },
            "Svc#work": {
                "symbol_kind": "Service",
                "simple_name": "Svc",
                "method_name": "work",
            },
            "Util#help": {
                "symbol_kind": "Util",
                "simple_name": "Util",
                "method_name": "help",
            },
        },
        entity_ids=["type:LoanDO"],
        entity_meta={
            "type:LoanDO": {"kind": "entity", "simple_name": "LoanDO"},
        },
    )

    async def run():
        sched = TopoPriorityScheduler(
            nodes,
            project_id="test",
            concurrency=2,
            max_retries=0,
            analyze=analyze,
            cache_lookup=None,
        )
        return await sched.run()

    stats = asyncio.run(run())
    assert stats.done == 4
    assert stats.failed == 0
    # Entity (HIGH, no deps) and Util#help ready together — entity should not be after Ctrl
    assert order.index("type:LoanDO") < order.index("Ctrl#entry")
    assert order.index("Util#help") < order.index("Svc#work")
    assert order.index("Svc#work") < order.index("Ctrl#entry")
    # Bottom-up context
    assert nodes["Svc#work"].result["callee_keys"] == ["Util#help"]
    assert nodes["Ctrl#entry"].result["callee_keys"] == ["Svc#work"]


def test_failure_propagates_and_retry():
    attempts = {"leaf": 0}

    async def analyze(node: ScheduleNode, ctx: SchedulerContext):
        if node.id == "leaf":
            attempts["leaf"] += 1
            if attempts["leaf"] <= 2:
                raise RuntimeError("boom")
            return {"summary": "ok-after-retry"}
        return {"summary": node.id, "up": node.fail_flag}

    nodes = {
        "leaf": ScheduleNode(id="leaf", kind="method", priority=Priority.HIGH),
        "mid": ScheduleNode(
            id="mid", kind="method", priority=Priority.NORMAL, deps={"leaf"}
        ),
        "top": ScheduleNode(
            id="top", kind="method", priority=Priority.LOW, deps={"mid"}
        ),
    }
    nodes["leaf"].callers.add("mid")
    nodes["mid"].callers.add("top")

    async def run_fail():
        # max_retries=0 → fail immediately, propagate
        nodes_f = {
            "leaf": ScheduleNode(id="leaf", kind="method", priority=Priority.HIGH),
            "mid": ScheduleNode(
                id="mid", kind="method", priority=Priority.NORMAL, deps={"leaf"}
            ),
            "top": ScheduleNode(
                id="top", kind="method", priority=Priority.LOW, deps={"mid"}
            ),
        }
        nodes_f["leaf"].callers.add("mid")
        nodes_f["mid"].callers.add("top")

        async def always_fail(node: ScheduleNode, ctx: SchedulerContext):
            if node.id == "leaf":
                raise RuntimeError("fail")
            return {"summary": node.id, "up": node.fail_flag}

        sched = TopoPriorityScheduler(
            nodes_f,
            project_id="test",
            concurrency=2,
            max_retries=0,
            analyze=always_fail,
            retry_backoff_sec=0,
        )
        await sched.run()
        assert nodes_f["leaf"].status == NodeStatus.FAILED
        assert nodes_f["mid"].fail_flag is True
        assert nodes_f["top"].fail_flag is True
        return nodes_f

    asyncio.run(run_fail())

    async def run_retry():
        sched = TopoPriorityScheduler(
            nodes,
            project_id="test",
            concurrency=1,
            max_retries=2,
            analyze=analyze,
            retry_backoff_sec=0,
        )
        return await sched.run()

    stats = asyncio.run(run_retry())
    assert attempts["leaf"] == 3
    assert stats.done == 3
    assert nodes["leaf"].status == NodeStatus.DONE


def test_mysql_cache_skip():
    ai_calls = []

    async def analyze(node: ScheduleNode, ctx: SchedulerContext):
        ai_calls.append(node.id)
        return {"summary": "fresh"}

    async def cache_lookup(nid: str):
        if nid == "cached":
            return {"summary": "from-mysql"}
        return None

    nodes = {
        "cached": ScheduleNode(id="cached", kind="method", priority=Priority.NORMAL),
        "fresh": ScheduleNode(id="fresh", kind="method", priority=Priority.NORMAL),
    }

    async def run():
        sched = TopoPriorityScheduler(
            nodes,
            project_id="test",
            concurrency=2,
            max_retries=0,
            analyze=analyze,
            cache_lookup=cache_lookup,
        )
        return await sched.run()

    stats = asyncio.run(run())
    assert stats.cached == 1
    assert "cached" not in ai_calls
    assert "fresh" in ai_calls
    assert nodes["cached"].result["summary"] == "from-mysql"
