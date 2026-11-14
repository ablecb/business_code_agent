"""Unit tests: risk/async callees must land in FlowDoc stages."""
from __future__ import annotations

from app.analysis.recursive_flow import (
    _boost_risk_async_importance,
    _ensure_stages_include_risk_async,
    _heuristic_node,
    _is_risk_async_node,
)
from app.parser.models import ParsedMethod, ParsedType


def _type(simple: str = "RiskServiceImpl", kind: str = "Service") -> ParsedType:
    return ParsedType(
        kind="class",
        package="com.demo",
        simple_name=simple,
        fqcn=f"com.demo.{simple}",
        file_path=f"{simple}.java",
        module_name="demo",
        start_line=1,
        end_line=10,
        symbol_kind=kind,
    )


def _method(name: str, code: str = "// noop") -> ParsedMethod:
    return ParsedMethod(
        name=name,
        signature=f"{name}()",
        start_line=1,
        end_line=5,
        raw_code=code,
    )


def test_is_risk_async_node_method_name():
    assert _is_risk_async_node("RiskServiceImpl#asynRisk")
    assert _is_risk_async_node("RiskServiceImpl#loanRiskV2")
    assert _is_risk_async_node("Foo#doRiskCheck")
    assert not _is_risk_async_node("RiskServiceImpl#getConfig")
    assert not _is_risk_async_node("WithdrawalsServiceImpl#loan")
    assert not _is_risk_async_node("WithdrawalsApplyServiceImpl#loanApply")


def test_boost_marks_asyn_risk_important():
    t = _type()
    m = _method("asynRisk", "return true;")
    data = {
        "important": False,
        "important_types": [],
        "class_method": "RiskServiceImpl#asynRisk",
    }
    _boost_risk_async_importance(data, t, m)
    assert data["important"] is True
    assert "external_call" in data["important_types"]


def test_heuristic_boosts_risk_method():
    data = _heuristic_node(_type(), _method("asynRisk"))
    assert data["important"] is True
    assert "external_call" in data["important_types"]


def test_ensure_stages_injects_asyn_risk_after_loan():
    stages = [
        {
            "step": 1,
            "title": "入口",
            "class_method": "WithdrawalsController#withdrawals",
            "desc": "入口",
        },
        {
            "step": 2,
            "title": "放款主流程",
            "class_method": "WithdrawalsServiceImpl#loan",
            "desc": "含条件风控",
        },
    ]
    payloads = [
        {"class_method": "WithdrawalsController#withdrawals", "summary": "入口"},
        {"class_method": "WithdrawalsServiceImpl#loan", "summary": "主流程"},
        {
            "class_method": "RiskServiceImpl#asynRisk",
            "summary": "RiskServiceImpl#asynRisk：发起风控",
        },
        {"class_method": "WithdrawalsServiceImpl#insertOrUpdateLoanInfo", "summary": "异常补表"},
    ]
    out = _ensure_stages_include_risk_async(stages, payloads)
    cms = [s["class_method"] for s in out]
    assert "RiskServiceImpl#asynRisk" in cms
    assert cms.index("RiskServiceImpl#asynRisk") == cms.index("WithdrawalsServiceImpl#loan") + 1
    assert [s["step"] for s in out] == list(range(1, len(out) + 1))


def test_ensure_stages_idempotent_when_already_present():
    stages = [
        {
            "step": 1,
            "title": "风控",
            "class_method": "RiskServiceImpl#asynRisk",
            "desc": "已有",
        }
    ]
    payloads = [
        {
            "class_method": "RiskServiceImpl#asynRisk",
            "summary": "RiskServiceImpl#asynRisk：发起风控",
        }
    ]
    out = _ensure_stages_include_risk_async(stages, payloads)
    assert len(out) == 1
    assert out[0]["desc"] == "已有"
