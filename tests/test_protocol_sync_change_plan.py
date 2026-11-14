"""Intent + render tests for protocol-sync field pass-through follow-ups."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from app.agent.change_impact.discover import RoleHit, Touchpoints
from app.agent.change_impact.render import (
    build_protocol_sync_evidence_pack,
    render_protocol_sync_plan,
)
from app.agent.explore_agent import (
    _sanitize_protocol_sync_plan,
    _wants_validation_focus,
)
from app.agent.question_intent import (
    assumed_new_field_name,
    wants_add_field_change_plan,
    wants_extend_field_to_protocol_sync,
)
from app.agent.task_planner import KIND_PROTOCOL_SYNC, classify_question_kind

_CHANGE_IMPACT_DIR = (
    Path(__file__).resolve().parents[1] / "app" / "agent" / "change_impact"
)
_BANNED = (
    "checkApplyReqParam",
    "reqVo2ExtDmo",
    "WithdrawalsApplyReqVo",
    "WithdrawalsExt",
    "demo_b_withdrawals_ext",
    "WithdrawalsController#checkApplyReqParam",
    "WithdrawalsApplyServiceImpl#reqVo2ExtDmo",
    "WithdrawalsServiceImpl#syncSignInfo",
    "WithdrawalsReqVo",
    "FieldNameConstants",
    "CommonConstant",
)


def test_change_impact_has_no_withdrawals_sync_checklist():
    blob = "\n".join(
        p.read_text(encoding="utf-8")
        for p in _CHANGE_IMPACT_DIR.glob("*.py")
    )
    for token in _BANNED:
        assert token not in blob, f"hardcoded checklist token leaked: {token}"


def test_wants_protocol_sync_followup():
    """Domain protocol-sync keyword router removed — falls through to LLM/explore."""
    q = "然后我还要把这个字段在协议同步时作为入参传过去"
    assert not wants_extend_field_to_protocol_sync(q)
    assert not wants_add_field_change_plan(q)
    assert classify_question_kind(q) != KIND_PROTOCOL_SYNC


def test_wants_protocol_sync_with_explicit_field():
    q = "把 businessLabel 字段在协议同步时作为入参传过去"
    assert not wants_extend_field_to_protocol_sync(q)
    assert assumed_new_field_name(q) == "businessLabel"


def test_field_name_from_prior_追问_block():
    q = (
        "然后我还要把这个字段在协议同步时作为入参传过去\n"
        "（追问：请承接上一问主题「放款接口增加一个 businessLabel 字段并落库，改哪些」；"
        "入口优先 WithdrawalsController#withdrawals）"
    )
    assert not wants_extend_field_to_protocol_sync(q)
    assert assumed_new_field_name(q) == "businessLabel"


def test_field_name_from_injection():
    q = "这个字段在协议同步时传过去\n（字段名：businessLabel）"
    assert assumed_new_field_name(q) == "businessLabel"


def test_add_field_still_not_protocol_sync():
    q = "放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    assert wants_add_field_change_plan(q)
    assert not wants_extend_field_to_protocol_sync(q)


def test_render_protocol_sync_plan_scannable():
    sync_code = """
    private boolean syncSignInfo(FooReqVo reqVo, FooRespVo respVo) {
        Map<String, Object> syncSignMap = new HashMap<>();
        // 产品类型
        syncSignMap.put(FooConstants.LOAN_TYPE, reqVo.getProductCode());
        // 渠道号
        syncSignMap.put(FooConstants.CHANNEL, reqVo.getMerchantNo());
        requestPara.setBody(syncSignMap);
        cbsServiceClient.syncSignInfo(requestPara);
        return true;
    }
    """
    tp = Touchpoints(
        capability="protocol_sync",
        entry="FooController#submit",
        field_name="businessLabel",
        roles={
            "constant": RoleHit(
                role="constant",
                fqcn="com.example.FooConstants",
                file_path="const/FooConstants.java",
                raw_code='public static final String LOAN_TYPE = "loanType";',
                proposed_const='BUSINESS_LABEL = "businessLabel"',
                peer_const_examples=['LOAN_TYPE = "loanType"'],
            ),
            "request_vo": RoleHit(
                role="request_vo",
                fqcn="com.example.FooReqVo",
                file_path="vo/FooReqVo.java",
                raw_code="private String productCode;\nprivate String merchantNo;",
                analog_fields=["productCode", "merchantNo"],
                via="sync_param",
            ),
            "protocol_sync": RoleHit(
                role="protocol_sync",
                fqcn="com.example.FooServiceImpl",
                method_name="syncSignInfo",
                file_path="service/FooServiceImpl.java",
                raw_code=sync_code,
                via="calls_closure",
            ),
        },
    )
    text = render_protocol_sync_plan(tp)
    assert "## 结论" in text
    assert "## 改动清单" in text
    assert "## DDL" not in text
    assert "businessLabel" in text
    assert "syncSignMap.put" in text
    assert "getBusinessLabel" in text
    assert "FooConstants.BUSINESS_LABEL" in text
    pack = build_protocol_sync_evidence_pack(tp)
    assert pack["need_ddl"] is False
    assert pack["capability"] == "protocol_sync"
    assert len(pack["change_items"]) == 3


def test_sanitize_strips_validation_noise():
    junk = (
        "## 结论\n\n透传 businessLabel。\n\n"
        "## 改动清单\n\n"
        "### 1. 协议同步组装\n\n"
        "`a.java` · `Foo#syncSignInfo`\n\n"
        "```java\nsyncSignMap.put(C.BUSINESS_LABEL, reqVo.getBusinessLabel());\n```\n\n"
        "## DDL\n\n```sql\nALTER TABLE t_x ADD COLUMN c;\n```\n\n"
        "queryProductGuarateeName 要校验 productCode/merchantNo。\n"
    )
    out = _sanitize_protocol_sync_plan(junk, question="协议同步传字段")
    assert "## DDL" not in out
    assert "queryProductGuarateeName" not in out
    assert "## 结论" in out
    assert "改动清单" in out


def test_discover_protocol_sync_plan_tool_removed_from_catalog():
    from app.agent.explore_tools import TOOL_SPECS, dispatch_tool

    names = {t["name"] for t in TOOL_SPECS}
    assert "discover_protocol_sync_plan" not in names
    routed = dispatch_tool(
        "p",
        "discover_protocol_sync_plan",
        {"question": "把字段在协议同步时传过去"},
    )
    assert routed.get("ok") is False
    assert "removed" in str(routed.get("error") or "").lower()


def test_ai_render_protocol_sync_skips_ddl_requirement():
    from app.agent.change_impact.ai_render import _validate_ai_plan

    tp = Touchpoints(
        capability="protocol_sync",
        entry="Foo#bar",
        field_name="businessLabel",
        roles={
            "protocol_sync": RoleHit(
                role="protocol_sync",
                fqcn="com.example.FooService",
                method_name="syncSignInfo",
                file_path="svc/FooService.java",
                raw_code="syncSignMap.put(C.X, reqVo.getX());",
            ),
        },
    )
    pack = build_protocol_sync_evidence_pack(tp)
    draft = render_protocol_sync_plan(tp)
    assert _validate_ai_plan(draft, pack, capability="protocol_sync")
    with_ddl = draft + "\n\n## DDL\n\n```sql\nALTER TABLE t;\n```"
    assert not _validate_ai_plan(with_ddl, pack, capability="protocol_sync")
