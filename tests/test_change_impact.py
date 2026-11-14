"""Unit tests for change_impact role discovery + render (no hardcoded withdrawals methods)."""
from __future__ import annotations

import re
from pathlib import Path

from app.agent.change_impact.discover import (
    RoleHit,
    Touchpoints,
    _classify_roles,
    _entry_prefix,
    _field_to_const_name,
    _merge_role,
    _seed_search_queries,
)
from app.agent.change_impact.render import (
    _extract_response_field,
    build_changed_method_body,
    render_add_field_plan,
    render_composite_field_plan,
)
from app.agent.question_intent import wants_add_field_change_plan

_DISCOVER_SRC = Path(__file__).resolve().parents[1] / "app" / "agent" / "change_impact" / "discover.py"
_RENDER_SRC = Path(__file__).resolve().parents[1] / "app" / "agent" / "change_impact" / "render.py"
_AI_RENDER_SRC = Path(__file__).resolve().parents[1] / "app" / "agent" / "change_impact" / "ai_render.py"
_INIT_SRC = Path(__file__).resolve().parents[1] / "app" / "agent" / "change_impact" / "__init__.py"

# Literal checklist tokens that must never be re-hardcoded into change_impact.
_BANNED_CHECKLIST = (
    "checkApplyReqParam",
    "reqVo2ExtDmo",
    "WithdrawalsApplyReqVo",
    "WithdrawalsExt",
    "demo_b_withdrawals_ext",
    "WithdrawalsController#checkApplyReqParam",
    "WithdrawalsApplyServiceImpl#reqVo2ExtDmo",
    # Do not hardcode a specific CommonConstant#field for businessLabel.
    "CommonConstant.BUSINESS_LABEL",
    "CommonConstant#BUSINESS_LABEL",
)


def test_change_impact_modules_have_no_withdrawals_checklist():
    blob = "\n".join(
        p.read_text(encoding="utf-8")
        for p in (_DISCOVER_SRC, _RENDER_SRC, _AI_RENDER_SRC, _INIT_SRC)
    )
    for token in _BANNED_CHECKLIST:
        assert token not in blob, f"hardcoded checklist token leaked: {token}"
    # Must not pin a specific constant class name as the answer target.
    assert "FieldNameConstants" not in blob
    assert "CommonConstant" not in blob


def test_field_to_const_name_convention():
    assert _field_to_const_name("businessLabel") == "BUSINESS_LABEL"
    assert _field_to_const_name("accCardType") == "ACC_CARD_TYPE"


def test_entry_prefix_generic():
    assert _entry_prefix("FooController#bar") == "Foo"
    assert _entry_prefix("WithdrawalsController#withdrawals") == "Withdrawals"
    assert _entry_prefix("com.x.RepayServiceImpl#repay") == "Repay"


def test_seed_queries_from_prefix_not_fixed_methods():
    qs = _seed_search_queries("WithdrawalsController#withdrawals")
    assert any("ReqVo" in q for q in qs)
    assert any(q.startswith("Withdrawals") for q in qs)
    joined = " ".join(qs)
    assert "checkApplyReqParam" not in joined
    assert "reqVo2ExtDmo" not in joined


def test_merge_role_does_not_prefer_checkApply_name():
    """Bind scoring must be code-shape / closure based — not checkApply* favoritism."""
    roles: dict[str, RoleHit] = {}
    closure = {"FooController#checkFooParam"}
    apply_shaped = {
        "fqcn": "com.example.FooController",
        "method_name": "checkApplyReqParam",
        "raw_code": 'reqVo.setX((String) reqMap.get("x"));',
        "file_path": "a.java",
    }
    generic = {
        "fqcn": "com.example.FooController",
        "method_name": "checkFooParam",
        "raw_code": 'reqVo.setX((String) reqMap.get("x"));',
        "file_path": "a.java",
    }
    _merge_role(
        roles, "bind", apply_shaped, "bind", entry_prefix="Foo", closure_cms=closure
    )
    _merge_role(
        roles, "bind", generic, "bind", entry_prefix="Foo", closure_cms=closure
    )
    assert roles["bind"].method_name == "checkFooParam"


def test_merge_role_prefers_calls_closure_member():
    roles: dict[str, RoleHit] = {}
    closure = {"FooService#reqVo2ExtDmo"}
    off = {
        "fqcn": "com.example.OtherService",
        "method_name": "reqVo2ExtDmo",
        "raw_code": "ext.setX(reqVo.getX());",
        "file_path": "other.java",
    }
    on = {
        "fqcn": "com.example.FooService",
        "method_name": "reqVo2ExtDmo",
        "raw_code": "ext.setX(reqVo.getX());",
        "file_path": "foo.java",
    }
    _merge_role(roles, "assemble", off, "assemble", entry_prefix="Foo", closure_cms=closure)
    _merge_role(roles, "assemble", on, "assemble", entry_prefix="Foo", closure_cms=closure)
    assert roles["assemble"].fqcn.endswith("FooService")


def test_classify_roles_by_code_shape():
    vo = {
        "fqcn": "com.example.FooApplyReqVo",
        "method_name": "",
        "raw_code": "private String alpha;\nprivate String sceneCode;\n",
    }
    roles = dict(_classify_roles(vo))
    assert "request_vo" in roles

    bind = {
        "fqcn": "com.example.FooController",
        "method_name": "checkFooParam",
        "raw_code": 'reqVo.setAlpha((String) reqMap.get("alpha"));',
    }
    assert "bind" in dict(_classify_roles(bind))

    bind_const = {
        "fqcn": "com.example.FooController",
        "method_name": "checkFooParam",
        "raw_code": (
            "String t = (String) reqMap.get(FooFieldConstants.ACC_CARD_TYPE);\n"
            "reqVo.setAccCardType(t);\n"
        ),
    }
    assert "bind" in dict(_classify_roles(bind_const))

    assemble = {
        "fqcn": "com.example.FooServiceImpl",
        "method_name": "reqVo2ExtDmo",
        "raw_code": "extDmo.setAlpha(reqVo.getAlpha());",
    }
    assert "assemble" in dict(_classify_roles(assemble))

    # Unrelated service with local `dmo` + BeanUtils must NOT look like assemble.
    false_assemble = {
        "fqcn": "com.example.FooServiceImpl",
        "method_name": "loanSwitch",
        "raw_code": (
            "BeanUtils.copyProperties(dmo, loanAssistance);\n"
            "dmo.setPresentResult(reqVo.getFromChannel());\n"
        ),
    }
    assert "assemble" not in dict(_classify_roles(false_assemble))

    entity = {
        "fqcn": "com.example.FooExt",
        "method_name": "",
        "raw_code": '@TableName("t_foo_ext")\npublic class FooExt { private String alpha; }',
    }
    cr = dict(_classify_roles(entity))
    assert "persist_entity" in cr
    assert "orm" in cr

    const = {
        "fqcn": "com.example.FooFieldConstants",
        "method_name": "",
        "raw_code": 'public static final String ACC_CARD_TYPE = "accCardType";\n',
    }
    assert "constant" in dict(_classify_roles(const))


def test_render_add_field_plan_scannable():
    tp = Touchpoints(
        capability="add_field",
        entry="FooController#submit",
        field_name="businessLabel",
        roles={
            "constant": RoleHit(
                role="constant",
                fqcn="com.example.FooFieldConstants",
                file_path="common/FooFieldConstants.java",
                raw_code='public static final String ACC_CARD_TYPE = "accCardType";',
                why="const",
                via="bind_const_ref",
                evidence='bind=`FooController#checkFooParam` · 对照 ACC_CARD_TYPE = "accCardType"',
                proposed_const='BUSINESS_LABEL = "businessLabel"',
                peer_const_examples=['ACC_CARD_TYPE = "accCardType"'],
            ),
            "request_vo": RoleHit(
                role="request_vo",
                fqcn="com.example.FooApplyReqVo",
                file_path="vo/FooApplyReqVo.java",
                raw_code="private String sceneCode;",
                why="vo",
                analog_fields=["sceneCode", "accCardType"],
                via="prefix_type",
                evidence="via=prefix_type（入口前缀定位类型） · 对照字段 sceneCode",
            ),
            "bind": RoleHit(
                role="bind",
                fqcn="com.example.FooController",
                method_name="checkFooParam",
                file_path="controller/FooController.java",
                raw_code=(
                    "        // 场景码\n"
                    "        String sceneCode = (String) reqMap.get(FooFieldConstants.SCENE_CODE);\n"
                    "        reqVo.setSceneCode(sceneCode);\n"
                    "        String accCardType = (String) reqMap.get(FooFieldConstants.ACC_CARD_TYPE);\n"
                    "        reqVo.setAccCardType(accCardType);\n"
                ),
                why="bind",
                via="calls_closure",
                evidence="CALLS←FooController#submit · reqMap.get(FooFieldConstants.ACC_CARD_TYPE)→set*",
            ),
            "assemble": RoleHit(
                role="assemble",
                fqcn="com.example.FooServiceImpl",
                method_name="reqVo2ExtDmo",
                file_path="service/FooServiceImpl.java",
                raw_code=(
                    "        // 场景码\n"
                    "        extDmo.setSceneCode(reqVo.getSceneCode());\n"
                    "        extDmo.setMarketingType(reqVo.getMarketingType());\n"
                ),
                why="assemble",
                via="calls_closure",
                evidence="CALLS←FooController#submit · `extDmo.setSceneCode`",
            ),
            "persist_entity": RoleHit(
                role="persist_entity",
                fqcn="com.example.FooExt",
                file_path="entity/FooExt.java",
                raw_code=(
                    '@TableName("t_foo_ext")\n'
                    "    /**\n"
                    "     * 场景码\n"
                    "     */\n"
                    "    private String sceneCode;\n"
                ),
                why="entity",
                table_name="t_foo_ext",
                via="prefix_type",
                evidence='via=prefix_type · @TableName="t_foo_ext"',
            ),
        },
        need_mapper_xml=False,
        closure_size=42,
        bag_count=12,
        elapsed_ms=1700,
    )
    text = render_add_field_plan(tp)
    assert "## 结论" in text
    assert "## 改动清单" in text
    assert "## DDL" in text
    assert "t_foo_ext" in text
    assert "business_label" in text
    assert "FooApplyReqVo" in text
    assert "FooFieldConstants" in text
    assert "BUSINESS_LABEL" in text
    assert "Mapper.xml" not in text or "不必改" in text
    assert "get_symbol_code" not in text
    assert "探索步数用尽" not in text
    assert "\n\n## " in text
    # Paste-ready Java only — no peer对照 / 证据废话.
    assert "```java" in text
    assert "private String businessLabel;" in text
    assert "setBusinessLabel" in text
    assert "对照本方法" not in text
    assert "证据:" not in text and "证据：" not in text
    assert "CALLS 闭包" not in text
    assert "var-resolve" not in text
    assert "String businessLabel = (String) reqMap.get(FooFieldConstants.BUSINESS_LABEL);" in text
    assert "extDmo.setBusinessLabel(reqVo.getBusinessLabel());" in text
    assert text.index("FooFieldConstants") < text.index("FooApplyReqVo")
    assert "FooFieldConstants.BUSINESS_LABEL" in text
    assert 'reqMap.get("businessLabel")' not in text
    # Single-field VO snippet — never paste peer neighbors.
    vo_block = text.split("### 2.")[1].split("### 3.")[0]
    assert "private String businessLabel;" in vo_block
    assert "partnerQuota" not in vo_block
    assert vo_block.count("private ") == 1


def test_render_partial_roles_no_fake_checklist():
    """Missing roles must be gap notes — never invent 4 placeholder steps."""
    tp = Touchpoints(
        capability="add_field",
        entry="FooController#submit",
        field_name="alpha",
        roles={
            "request_vo": RoleHit(
                role="request_vo",
                fqcn="com.example.FooApplyReqVo",
                file_path="vo/FooApplyReqVo.java",
                raw_code="private String sceneCode;",
                why="vo",
                via="prefix_type",
                evidence="via=prefix_type · 对照字段 sceneCode",
            ),
        },
        gaps=["入参绑定/校验", "落库组装", "持久化实体"],
        notes=["入口：`FooController#submit`", "字段：`alpha`"],
        closure_size=10,
        bag_count=3,
        elapsed_ms=500,
    )
    text = render_add_field_plan(tp)
    assert "FooApplyReqVo" in text
    assert "未定位到具体" not in text
    assert "入参绑定/校验 — 在入口绑定方法中增加" not in text
    assert "落库组装 — 将 VO" not in text
    assert "持久化实体 — 增加属性 + DDL（未定位" not in text
    assert "缺口" in text and "入参绑定/校验" in text
    # Only one checklist section for the found VO
    assert text.count("### 1.") == 1
    assert "### 2." not in text
    assert "表名待补" in text or "未读到 @TableName" in text
    assert "证据:" not in text
    assert "对照本方法" not in text


def test_var_resolve_role_boost_prefers_set_on_vo():
    from app.agent.change_impact.discover import _var_resolve_role_boost

    bags = {
        "Foo#checkParam": {
            "locals": [{"name": "reqVo", "type_name": "FooApplyReqVo", "how": "new"}],
            "call_sites": [
                {
                    "raw_call": "reqVo.setAlpha",
                    "resolved_type": "FooApplyReqVo",
                    "resolve_how": "local_decl",
                    "improved_by_locals": True,
                }
            ],
        },
        "Foo#unrelated": {"locals": [], "call_sites": []},
    }
    assert _var_resolve_role_boost("Foo#checkParam", bags) > _var_resolve_role_boost(
        "Foo#unrelated", bags
    )


def test_intent_still_routes_add_field():
    assert wants_add_field_change_plan(
        "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    )


def test_discover_add_field_plan_tool_in_catalog():
    from unittest.mock import patch

    from app.agent.explore_tools import TOOL_SPECS, dispatch_tool

    names = {t["name"] for t in TOOL_SPECS}
    assert "discover_add_field_plan" in names
    # Avoid Neo4j/MySQL: stub change_impact runner used by the tool.
    with patch(
        "app.agent.change_impact.run_add_field_plan",
        return_value={
            "ok": True,
            "answer": "## 结论\n\nok\n",
            "steps": [],
            "entry": "Foo#bar",
            "touchpoints": None,
            "render_source": "ai",
        },
    ):
        routed = dispatch_tool(
            "p",
            "discover_add_field_plan",
            {"question": "增加一个字段 foo 并落库，改哪些"},
        )
    assert routed.get("ok") is True
    assert routed.get("plan_markdown") or routed.get("answer")
    assert routed.get("pass_through") is True
    assert routed.get("render_source") == "ai"
    assert "AI 梳理" in (routed.get("note") or "")


def test_ai_render_validates_and_falls_back():
    from unittest.mock import MagicMock, patch

    from app.agent.change_impact.ai_render import (
        _validate_ai_plan,
        render_change_plan_with_ai,
    )
    from app.agent.change_impact.render import build_add_field_evidence_pack

    tp = Touchpoints(
        capability="add_field",
        entry="FooController#submit",
        field_name="businessLabel",
        roles={
            "request_vo": RoleHit(
                role="request_vo",
                fqcn="com.example.FooApplyReqVo",
                file_path="vo/FooApplyReqVo.java",
                raw_code="private String sceneCode;",
            ),
            "persist_entity": RoleHit(
                role="persist_entity",
                fqcn="com.example.FooExt",
                file_path="entity/FooExt.java",
                raw_code='@TableName("t_foo_ext")\nprivate String sceneCode;',
                table_name="t_foo_ext",
            ),
        },
    )
    pack = build_add_field_evidence_pack(tp)
    assert pack["change_items"]
    draft = render_add_field_plan(tp)
    assert _validate_ai_plan(draft, pack)

    bad = "## 结论\n\nx\n\n## 改动清单\n\n`evil/Hack.java`\n\n## DDL\n\n```sql\nALTER\n```"
    assert not _validate_ai_plan(bad, pack)

    junk = draft + "\nCALLS 闭包证明\n"
    assert not _validate_ai_plan(junk, pack)

    polished = (
        "## 结论\n\n"
        "入口补 `businessLabel` 并落库。\n\n"
        "## 改动清单\n\n"
        "### 1. 请求 VO\n\n"
        "`vo/FooApplyReqVo.java`\n\n"
        "```java\nprivate String businessLabel;\n```\n\n"
        "### 2. 实体\n\n"
        "`entity/FooExt.java`\n\n"
        "```java\nprivate String businessLabel;\n```\n\n"
        "## DDL\n\n"
        "```sql\nALTER TABLE `t_foo_ext` ADD COLUMN `business_label` varchar(64);\n```"
    )
    mock_llm = MagicMock()
    mock_llm.chat_sync.return_value = polished
    with patch("app.llm.factory.get_llm_provider", return_value=mock_llm):
        text, source = render_change_plan_with_ai("加字段 businessLabel", tp)
    assert source == "ai"
    assert "## 改动清单" in text
    assert "businessLabel" in text

    mock_llm.chat_sync.side_effect = RuntimeError("llm down")
    with patch("app.llm.factory.get_llm_provider", return_value=mock_llm):
        text2, source2 = render_change_plan_with_ai("加字段 businessLabel", tp)
    assert source2 == "deterministic"
    assert "## 结论" in text2


def test_extract_response_field_generic():
    """响应字段提取：只在「响应/返回」语义且非入参并列时命中。"""
    assert _extract_response_field("增加入参aaa，然后将其响应中bbb字段落到放款表") == "bbb"
    assert _extract_response_field("把返回结果里的bizNo存到放款表") == "bizNo"
    assert _extract_response_field("响应中的total字段写入库") == "total"
    # 无响应语义 / 纯入参并列 → 不提取
    assert _extract_response_field("入参加个aaa然后落库") is None
    assert _extract_response_field("增加入参aaa和bbb两个字段并落库") is None
    assert _extract_response_field("查询产品配置时加参数aaa") is None


def _mk_tp_for_method_body(*, field_name: str = "aaa") -> Touchpoints:
    return Touchpoints(
        capability="composite_field",
        entry="DemoController#demo",
        field_name=field_name,
        roles={
            "assemble": RoleHit(
                role="assemble",
                fqcn="com.demo.loan.service.impl.DemoServiceImpl",
                method_name="demo",
                file_path="loan-service/.../DemoServiceImpl.java",
                raw_code=(
                    "    public void demo(DemoReqVo req) {\n"
                    "        DemoEntity entity = new DemoEntity();\n"
                    "        entity.setName(req.getName());\n"
                    "        Response<ProdResp> prodResp = prodClient.queryProd(request);\n"
                    "        entity.setProdCode(prodResp.getBody().getProdCode());\n"
                    "        demoMapper.insert(entity);\n"
                    "        return;\n"
                    "    }"
                ),
                why="assemble",
            ),
            "persist_entity": RoleHit(
                role="persist_entity",
                fqcn="com.demo.loan.entity.DemoEntity",
                raw_code="public class DemoEntity {\n    private String name;\n}",
                why="persist",
                table_name="demo_b_demo",
            ),
        },
        need_mapper_xml=False,
        gaps=[],
        notes=[],
    )


def test_changed_method_body_injects_param_and_response_field():
    """改动后方法体：入参落库 + 响应字段→落库列映射（不编造、不改原文）。"""
    tp = _mk_tp_for_method_body()
    q = "增加入参aaa，然后将其响应中bbb字段落到放款表"
    body = build_changed_method_body(tp, question=q)
    assert "setAaa" in body  # 入参落库
    assert "setBbb" in body  # 响应字段落库
    assert "prodResp.getBody().getBbb" in body  # 从真实响应变量取字段
    assert "prodResp.getBody() != null" in body  # 判空保护
    # 不编造：响应变量来自证据代码本身
    assert "prodClient.queryProd" in body
    # 原文保留
    assert "entity.setName(req.getName())" in body
    assert "demoMapper.insert(entity)" in body


def test_changed_method_body_no_response_field_when_absent():
    """问题没有点名响应字段时，不追加响应映射段。"""
    tp = _mk_tp_for_method_body()
    body = build_changed_method_body(tp, question="增加入参aaa然后落库")
    assert "setAaa" in body
    assert "setBbb" not in body
    assert "响应字段" not in body


def test_render_composite_appends_changed_method_section():
    """composite 渲染含「改动后完整方法体」小节且含 [新增] 标注。"""
    tp = _mk_tp_for_method_body()
    out = render_composite_field_plan(tp, question="增加入参aaa，然后将其响应中bbb字段落到放款表")
    assert "改动后完整方法体" in out
    assert "[新增]" in out
    assert "setBbb" in out
