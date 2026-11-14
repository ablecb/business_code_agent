"""Tests for research plan classification, LLM parse, and step mapping."""
from __future__ import annotations

from app.agent.task_planner import (
    KIND_STATUS_UPDATE,
    KIND_FLOW,
    KIND_VALIDATION,
    KIND_GENERAL,
    _parse_llm_plan,
    build_research_plan,
    classify_question_kind,
    map_tool_to_step_id,
)


def test_status_update_question_kind():
    q = "放款表放款状态是怎么更新的"
    assert classify_question_kind(q) == KIND_GENERAL
    plan = build_research_plan(q, project_id="demo")
    ids = [s["id"] for s in plan["steps"]]
    # Heuristic kind is always general; specialized steps come from LLM plan.
    assert ids == [
        "search",
        "read_code",
        "analyze_synthesize",
    ]
    assert plan["kind"] == KIND_GENERAL


def test_flow_and_validation_kinds():
    assert classify_question_kind("放款流程什么样") == KIND_GENERAL
    assert classify_question_kind("入口参数校验了哪些字段") == KIND_GENERAL


def test_map_tool_for_status_update():
    assert (
        map_tool_to_step_id(
            "search_symbols",
            KIND_STATUS_UPDATE,
            args={"q": "放款", "kind": "Entity"},
        )
        == "locate_table"
    )
    assert (
        map_tool_to_step_id(
            "search_symbols",
            KIND_STATUS_UPDATE,
            args={"q": "status update setStatus"},
        )
        == "find_writers"
    )
    assert map_tool_to_step_id("list_callers", KIND_STATUS_UPDATE) == "trace_call_chains"
    assert map_tool_to_step_id("finish", KIND_STATUS_UPDATE) == "analyze_synthesize"


def test_parse_llm_dynamic_plan():
    raw = """
    {
      "kind": "status_update",
      "title": "查清放款表状态如何被更新",
      "steps": [
        {"id": "find_loan_table", "title": "找到放款表实体", "detail": "搜放款相关 Entity", "tools": ["search_symbols"]},
        {"id": "check_status_field", "title": "确认状态字段", "detail": "读实体字段", "tools": ["get_symbol_code"]},
        {"id": "who_writes", "title": "谁在写状态", "detail": "搜 setStatus/update", "tools": ["search_symbols", "list_callers"]},
        {"id": "wrap_up", "title": "汇总结论", "detail": "说明更新链路", "tools": ["finish"]}
      ]
    }
    """
    plan = _parse_llm_plan(raw, question="放款表放款状态是怎么更新的", project_id="p1")
    assert plan is not None
    assert plan["source"] == "llm"
    assert plan["kind"] == KIND_STATUS_UPDATE
    assert "放款" in plan["title"]
    ids = [s["id"] for s in plan["steps"]]
    assert ids[0] == "find_loan_table"
    assert ids[-1] == "analyze_synthesize"
    # tools hint mapping
    assert (
        map_tool_to_step_id(
            "search_symbols",
            plan["kind"],
            plan_steps=plan["steps"],
        )
        == "find_loan_table"
    )
