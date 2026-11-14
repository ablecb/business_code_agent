"""Intents + routing for multi-turn gate drill-down and composite change.

Domain routers (composite / protocol_sync / gate_detail / drill_gate) are stubs —
business attribution is LLM + explore. Keep render + denylist regressions.
"""
from __future__ import annotations

from pathlib import Path

from app.agent.change_impact.discover import RoleHit, Touchpoints
from app.agent.change_impact.render import render_composite_field_plan
from app.agent.explore_agent import (
    _wants_gate_detail,
    _wants_source_paste,
    _wants_validation_focus,
)
from app.agent.question_intent import (
    assumed_new_field_name,
    gate_themes_in_question,
    wants_add_field_change_plan,
    wants_composite_change,
    wants_drill_gate_logic,
    wants_extend_field_to_protocol_sync,
    wants_show_gate_code,
)
from app.agent.task_planner import (
    KIND_GENERAL,
    classify_question_kind,
)

_CHANGE_IMPACT_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "agent" / "change_impact"
)
_BANNED_COMPOSITE = (
    "WithdrawalsController#checkApplyReqParam",
    "WithdrawalsApplyServiceImpl#reqVo2ExtDmo",
    "WithdrawalsServiceImpl#orgQuotaCheck",
    "WithdrawalsServiceImpl#quotaCheck",
    "demo_b_withdrawals_ext",
)


def test_drill_gate_logic_router_removed():
    q = "那额度校验这一块逻辑什么样的"
    assert not wants_drill_gate_logic(q)
    assert not _wants_gate_detail(q)
    assert not wants_show_gate_code(q)
    assert gate_themes_in_question(q) == []


def test_drill_gate_org_query_router_removed():
    q = "机构查询这一块逻辑什么样的"
    assert not wants_drill_gate_logic(q)
    assert gate_themes_in_question(q) == []


def test_show_gate_code_short_followup():
    q = "我想看看对应的代码"
    assert wants_show_gate_code(q)
    assert _wants_source_paste(q)
    assert not wants_drill_gate_logic(q)
    assert classify_question_kind(q) == KIND_GENERAL


def test_cold_start_quota_code_show_without_last_gate():
    """Entry+theme+code in one ask → show-code (no prior last_gate needed)."""
    for q in (
        "放款接口中额度校验部分代码什么样",
        "放款接口里额度校验的代码",
        "额度校验部分代码",
        "放款接口中 额度校验部分代码什么样",
    ):
        assert wants_show_gate_code(q), q
        assert _wants_source_paste(q), q
        assert not wants_drill_gate_logic(q), q
        assert not _wants_gate_detail(q), q
        assert classify_question_kind(q) == KIND_GENERAL, q
        assert gate_themes_in_question(q) == [], q


def test_cold_start_quota_logic_without_domain_router():
    """Logic dig-in no longer forced via keyword gate router."""
    q = "放款接口中额度校验逻辑什么样"
    assert not wants_drill_gate_logic(q)
    assert not _wants_gate_detail(q)
    assert not wants_show_gate_code(q)
    assert gate_themes_in_question(q) == []


def test_composite_change_router_removed_falls_to_add_field():
    q = (
        "放款接口我想增加新入参，并落库，并在机构校验时判断是单笔单批时"
        "传递给外部系统，需要改动哪些"
    )
    assert not wants_composite_change(q)
    assert wants_add_field_change_plan(q)
    assert not wants_extend_field_to_protocol_sync(q)
    assert not _wants_gate_detail(q)
    assert not _wants_validation_focus(q)
    assert classify_question_kind(q) == KIND_GENERAL
    assert assumed_new_field_name(q) == "newField"


def test_composite_with_explicit_field_falls_to_add_field():
    q = (
        "放款接口增加 businessLabel 字段并落库，"
        "机构校验单笔单批时传给外部系统，改哪些"
    )
    assert not wants_composite_change(q)
    assert wants_add_field_change_plan(q)
    assert assumed_new_field_name(q) == "businessLabel"
    assert classify_question_kind(q) == KIND_GENERAL


def test_plain_add_field_not_composite():
    q = "放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    assert wants_add_field_change_plan(q)
    assert not wants_composite_change(q)


def test_protocol_sync_router_removed():
    q = "然后我还要把这个字段在协议同步时作为入参传过去"
    assert not wants_extend_field_to_protocol_sync(q)
    assert not wants_composite_change(q)


def test_change_impact_has_no_hardcoded_withdrawals_composite_checklist():
    blob = "\n".join(
        p.read_text(encoding="utf-8") for p in _CHANGE_IMPACT_DIR.glob("*.py")
    )
    for token in _BANNED_COMPOSITE:
        assert token not in blob, f"hardcoded checklist token leaked: {token}"


def test_render_composite_field_plan_scannable():
    tp = Touchpoints(
        capability="composite_field",
        entry="FooController#submit",
        field_name="businessLabel",
        roles={
            "request_vo": RoleHit(
                role="request_vo",
                fqcn="com.example.FooReqVo",
                file_path="vo/FooReqVo.java",
                raw_code="private String productCode;",
                analog_fields=["productCode"],
            ),
            "persist": RoleHit(
                role="persist",
                fqcn="com.example.FooExt",
                file_path="entity/FooExt.java",
                raw_code="private String productCode;",
                analog_fields=["productCode"],
                table_name="t_foo_ext",
            ),
        },
    )
    md = render_composite_field_plan(tp)
    assert "businessLabel" in md
    assert "FooReqVo" in md
    assert "## 结论" in md
    assert "## 改动清单" in md
