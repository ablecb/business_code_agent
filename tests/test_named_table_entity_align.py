"""Named-table → @TableName entity alignment (no product class hardcoding)."""
from __future__ import annotations

from app.agent.change_impact.discover import (
    RoleHit,
    Touchpoints,
    _classify_roles,
    _merge_role,
    _score_entity_for_table,
    _table_stem_to_pascal,
    extract_named_tables,
)
from app.agent.change_impact.render import build_add_field_evidence_pack, render_add_field_plan


def test_extract_named_tables_order_unique():
    q = "落库到 demo_b_withdrawals 和 demo_b_loan_assistance，以及 demo_b_withdrawals"
    assert extract_named_tables(q) == [
        "demo_b_withdrawals",
        "demo_b_loan_assistance",
    ]


def test_table_stem_to_pascal():
    assert _table_stem_to_pascal("demo_b_withdrawals") == "Withdrawals"
    assert _table_stem_to_pascal("demo_b_loan_assistance") == "LoanAssistance"
    assert _table_stem_to_pascal("demo_b_withdrawals_ext") == "WithdrawalsExt"


def test_score_prefers_exact_entity_over_ext_sibling():
    table = "demo_b_withdrawals"
    assert _score_entity_for_table("Withdrawals", table) > _score_entity_for_table(
        "WithdrawalsExt", table
    )
    assert _score_entity_for_table("LoanAssistance", "demo_b_loan_assistance") > (
        _score_entity_for_table("LoanAssistanceExt", "demo_b_loan_assistance")
    )


def test_merge_persist_prefers_named_table_over_ext():
    roles: dict[str, RoleHit] = {}
    pref = {"demo_b_withdrawals"}
    ext = {
        "fqcn": "com.example.FooExt",
        "raw_code": '@TableName(value = "demo_b_withdrawals_ext")\nclass FooExt {}',
        "file_path": "FooExt.java",
    }
    base = {
        "fqcn": "com.example.Foo",
        "raw_code": '@TableName(value = "demo_b_withdrawals")\nclass Foo {}',
        "file_path": "Foo.java",
    }
    _merge_role(
        roles, "persist_entity", ext, "ext", preferred_tables=pref, via="prefix_type"
    )
    _merge_role(
        roles, "persist_entity", base, "base", preferred_tables=pref, via="named_table"
    )
    assert roles["persist_entity"].table_name == "demo_b_withdrawals"
    assert roles["persist_entity"].fqcn.endswith(".Foo")


def test_classify_assemble_for_named_entity_insert_param_shape():
    sym = {
        "fqcn": "com.example.FooServiceImpl",
        "method_name": "insertParam",
        "raw_code": (
            "private Foo insertParam(FooReqVo reqVo) {\n"
            "  Foo insertDmo = new Foo();\n"
            "  insertDmo.setOrderId(reqVo.getOrderId());\n"
            "  return insertDmo;\n"
            "}"
        ),
    }
    roles = _classify_roles(sym, preferred_entity_names={"Foo"})
    assert any(r == "assemble" for r, _ in roles)


def test_render_multi_table_ddl_and_entities():
    tp = Touchpoints(
        capability="add_field",
        entry="FooController#foo",
        field_name="businessLabel",
        named_tables=["demo_b_withdrawals", "demo_b_loan_assistance"],
        roles={
            "request_vo": RoleHit(
                role="request_vo",
                fqcn="com.x.FooReqVo",
                file_path="FooReqVo.java",
                raw_code="private String orderId;",
            ),
            "assemble": RoleHit(
                role="assemble",
                fqcn="com.x.FooServiceImpl",
                method_name="insertParam",
                file_path="FooServiceImpl.java",
                raw_code="insertDmo.setOrderId(reqVo.getOrderId());",
            ),
            "persist_entity": RoleHit(
                role="persist_entity",
                fqcn="com.x.Withdrawals",
                file_path="Withdrawals.java",
                table_name="demo_b_withdrawals",
                raw_code='@TableName("demo_b_withdrawals")\nprivate String orderId;',
            ),
            "persist_entity_2": RoleHit(
                role="persist_entity_2",
                fqcn="com.x.LoanAssistance",
                file_path="LoanAssistance.java",
                table_name="demo_b_loan_assistance",
                raw_code='@TableName("demo_b_loan_assistance")\nprivate String orderId;',
            ),
        },
    )
    pack = build_add_field_evidence_pack(tp)
    assert "demo_b_withdrawals" in pack["table_names"]
    assert "demo_b_loan_assistance" in pack["table_names"]
    assert pack["suggested_ddl"].count("ALTER TABLE") == 2
    md = render_add_field_plan(tp)
    assert "demo_b_withdrawals" in md
    assert "demo_b_loan_assistance" in md
    assert "Withdrawals" in md
    assert "LoanAssistance" in md
