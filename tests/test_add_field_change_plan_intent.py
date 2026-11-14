"""Unit tests for add-field change-plan intent (not 落库 gate dig-in)."""
from __future__ import annotations

import re

from app.agent.explore_agent import (
    _add_field_answer_has_synth_junk,
    _add_field_answer_is_persist_gate_narrative,
    _add_field_answer_missing_scannable_format,
    _sanitize_add_field_change_plan,
    _wants_gate_detail,
    _wants_tech_detail,
    _wants_validation_focus,
)
from app.agent.explore_intent import _wants_add_field_change_plan
from app.agent.question_intent import assumed_new_field_name, wants_add_field_change_plan
from app.agent.task_planner import KIND_GENERAL, classify_question_kind

def test_wants_add_field_change_plan_canonical_ask():
    q = "我想放款接口增加一个字段并落库，都要修改哪些"
    assert wants_add_field_change_plan(q)
    assert _wants_add_field_change_plan(q)
    assert not _wants_gate_detail(q)
    assert not _wants_validation_focus(q)
    assert _wants_tech_detail(q)
    assert classify_question_kind(q) == KIND_GENERAL
    assert assumed_new_field_name(q) == "newField"


def test_wants_add_field_with_explicit_name():
    q = "放款接口增加一个 businessLabel 字段并落库，改哪些"
    assert wants_add_field_change_plan(q)
    assert assumed_new_field_name(q) == "businessLabel"


def test_assumed_field_from_param_gloss():
    """Desktop phrasing: 增加一个入参 businessLabel(业务标签)."""
    q = (
        "我想放款接口增加一个入参 businessLabel(业务标签) 并落库到放款表，"
        "同时在额度校验时将该参数作为入参传递给外部系统  "
        "给我对应的程序修改清单和ddl"
    )
    assert wants_add_field_change_plan(q)
    assert assumed_new_field_name(q) == "businessLabel"


def test_wants_add_field_save_to_table_and_outbound():
    """保存到××表 / 送给外部系统 也算改动方案信号，不必写「改哪些」。"""
    q = (
        "我想放款接口增加一个字段labelBusiness，并且将其保存到放款表，"
        "然后再调用额度校验时候送给外部系统"
    )
    assert wants_add_field_change_plan(q)
    assert assumed_new_field_name(q) == "labelBusiness"


def test_gate_detail_still_for_persist_dig_in():
    """「落库怎么做」without 增加字段 → not add-field plan (gate router removed)."""
    q = "放款落库是怎么做的"
    assert not wants_add_field_change_plan(q)
    assert _wants_gate_detail(q) is False


def test_quota_gate_unaffected():
    q = "额度校验是怎么校验的"
    assert not wants_add_field_change_plan(q)
    assert _wants_gate_detail(q) is False


def test_persist_gate_narrative_detector():
    bad = (
        "1. loanApply 申请落库\n"
        "2. insertLoanData 写入放款主表\n"
        "3. insertOrUpdateLoanInfo 更新\n"
    )
    assert _add_field_answer_is_persist_gate_narrative(bad)
    good = (
        "假设字段 businessLabel。\n"
        "1. WithdrawalsApplyReqVo 增加属性\n"
        "2. checkApplyReqParam 绑定\n"
        "3. reqVo2ExtDmo set 到 WithdrawalsExt\n"
        "4. Mapper / ALTER demo_b_withdrawals_ext\n"
    )
    assert not _add_field_answer_is_persist_gate_narrative(good)


def test_synth_junk_detector_and_sanitize():
    from app.agent.explore_agent import _add_field_answer_has_offtopic_credit_noise

    junk = (
        "## 改动清单\n"
        "1. WithdrawalsApplyReqVo — 新增 businessLabel\n"
        "2. WithdrawalsController#checkApplyReqParam — reqMap.get → setBusinessLabel\n"
        "3. CreditController#generateReqParam — 赋值到 CreditNotification\n"
        "4. 必须修改 WithdrawalsExtMapper.xml 增加列映射\n"
        "5. ALTER TABLE WithdrawalsApply 加列 business_label\n"
        "## DDL\n"
        "```sql\nALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(64);\n```\n"
        "业务状态流转：在放款流程中，当放款状态为 F 时更新主表。\n"
        "外部系统调用：未在已读材料中看到明确的 Feign 调用。\n"
        "## 技术锚点\n"
        "- WithdrawalsApplyReqVo\n"
        "## 说明\n"
        "- 若字段仅在 VO 使用可省略其他 DTO\n"
        "（评审提示：可追问「落库怎么写的」。）\n"
    )
    assert _add_field_answer_has_synth_junk(junk)
    q = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    assert _add_field_answer_has_offtopic_credit_noise(junk, q)
    clean = _sanitize_add_field_change_plan(junk, question=q)
    assert "业务状态流转" not in clean
    assert "外部系统调用" not in clean
    assert "评审提示" not in clean
    assert "CreditController" not in clean
    assert "技术锚点" not in clean
    assert "## 说明" not in clean
    assert "ExtMapper.xml" not in clean
    assert "WithdrawalsApply 加列" not in clean
    # Legacy heading migrated to new layout.
    assert "## 改动清单" in clean
    assert "checkApplyReqParam" in clean
    assert "ALTER TABLE" in clean
    assert not _add_field_answer_missing_scannable_format(clean)
    # Sections separated by blank lines for UI readability.
    assert re.search(r"##\s*改动清单[^\n]*\n\n", clean)
    assert re.search(r"\n\n##\s*DDL", clean)


def test_missing_scannable_format():
    prose = (
        "1. WithdrawalsApplyReqVo 增加属性\n"
        "2. checkApplyReqParam 绑定\n"
        "ALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(64);\n"
    )
    assert _add_field_answer_missing_scannable_format(prose)
    good = (
        "## 结论\n"
        "入口补齐 businessLabel。\n"
        "## 改动清单\n"
        "1. WithdrawalsApplyReqVo — 新增字段\n"
        "2. WithdrawalsController#checkApplyReqParam — 绑定\n"
        "## DDL\n```sql\nALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(64);\n```\n"
    )
    assert not _add_field_answer_missing_scannable_format(good)
    no_bind = (
        "## 改动清单\n1. WithdrawalsApplyReqVo — 新增字段\n"
        "## DDL\n```sql\nALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(64);\n```\n"
    )
    assert _add_field_answer_missing_scannable_format(no_bind)


def test_add_field_guidance_is_role_based_not_hardcoded():
    from app.agent.question_intent import add_field_explore_guidance

    style, plan = add_field_explore_guidance(
        "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    )
    assert "discover_add_field_plan" in style or "discover_add_field_plan" in plan
    assert "MyBatis-Plus" in style or "Mapper.xml" in style or "Mapper.xml" in plan
    # Must NOT force a fixed withdrawals method checklist.
    assert "WithdrawalsApplyReqVo" not in style + plan
    assert "checkApplyReqParam" not in style + plan
    assert "reqVo2ExtDmo" not in style + plan
    assert "改动清单" not in style + plan or "禁止" in style
    assert "FieldNameConstants" in style or "BUSINESS_LABEL" in style
    assert "原样" in style or "原样" in plan


def test_add_field_roles_detect_generic_patterns():
    from app.agent.explore_add_field import _add_field_evidence_coverage
    from app.agent.explore_agent import (
        _add_field_evidence_gaps,
        _add_field_role_of_hit,
    )

    vo = {
        "fqcn": "com.example.FooApplyReqVo",
        "method_name": "",
        "raw_code": "private String alpha;\nprivate String beta;\n",
    }
    bind = {
        "fqcn": "com.example.FooController",
        "method_name": "checkFooParam",
        "raw_code": 'reqVo.setAlpha((String) reqMap.get("alpha"));',
    }
    assemble = {
        "fqcn": "com.example.FooServiceImpl",
        "method_name": "reqVo2ExtDmo",
        "raw_code": "ext.setAlpha(reqVo.getAlpha());",
    }
    entity = {
        "fqcn": "com.example.FooExt",
        "method_name": "",
        "raw_code": '@TableName("t_foo_ext")\npublic class FooExt { private String alpha; }',
    }
    assert "vo" in _add_field_role_of_hit(vo)
    assert "bind" in _add_field_role_of_hit(bind)
    assert "assemble" in _add_field_role_of_hit(assemble)
    assert "entity" in _add_field_role_of_hit(entity)
    cov = _add_field_evidence_coverage([vo, bind, assemble, entity])
    assert cov["vo"] and cov["bind"] and cov["assemble"] and cov["ext"] and cov["mp"]
    steps = [
        {"tool": "search_symbols", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    assert _add_field_evidence_gaps(steps, [vo, bind, assemble, entity]) is None
    # Gaps must not name Withdrawals* specifically.
    gap = _add_field_evidence_gaps(steps, [vo, bind])
    assert gap
    assert "WithdrawalsApplyReqVo" not in gap
    assert "角色" in gap or "请求 VO" in gap


def test_normalize_smashed_sections_spacing():
    smashed = (
        "假设字段 businessLabel。"
        "## 改动清单 1. WithdrawalsApplyReqVo — 新增 businessLabel "
        "2. checkApplyReqParam — 绑定 "
        "3. reqVo2ExtDmo — setBusinessLabel "
        "4. WithdrawalsExt — 实体字段"
        "## 建议改动（从 → 到）- 新增 private String businessLabel;"
        "## DDL\n```sql\nALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(64);\n```"
    )
    clean = _sanitize_add_field_change_plan(
        smashed,
        question="我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
    )
    assert "## 改动清单" in clean
    assert "## 建议改动" not in clean
    assert "## DDL" in clean
    assert "\n\n## " in clean
    assert "ExtMapper.xml" not in clean


def test_wrong_apply_persist_sanitized():
    from app.agent.explore_agent import _add_field_answer_wrong_apply_persist

    bad = (
        "## 改动清单\n\n"
        "1. WithdrawalsApplyReqVo — 新增 businessLabel\n"
        "2. reqVo2ExtDmo — 将 businessLabel 从 WithdrawalsApplyReqVo 复制到 WithdrawalsApply 实体\n"
        "3. updateReult — 将 businessLabel 从 WithdrawalsApply 实体复制到 WithdrawalsApplyRespVo\n\n"
        "## 建议改动（从 → 到）\n\n"
        "- 复制到 WithdrawalsApply 实体\n\n"
        "## DDL\n\n"
        "```sql\nALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(64);\n```\n"
    )
    assert _add_field_answer_wrong_apply_persist(bad)
    clean = _sanitize_add_field_change_plan(
        bad,
        question="我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
    )
    assert "复制到 WithdrawalsApply 实体" not in clean
    assert "demo_b_withdrawals_ext" in clean
    assert "## 改动清单" in clean


def test_ddl_heading_split_from_sql_fence():
    smashed = (
        "## 改动清单\n\n"
        "1. WithdrawalsApplyReqVo — 新增字段\n\n"
        "## 建议改动（从 → 到）\n\n"
        "- 新增 private String businessLabel;\n\n"
        "## DDL ```sql ALTER TABLE demo_b_withdrawals_ext ADD COLUMN business_label VARCHAR(255);\n"
    )
    clean = _sanitize_add_field_change_plan(
        smashed,
        question="我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
    )
    assert re.search(r"## DDL\n\n```sql\nALTER TABLE", clean)
    assert "ExtMapper.xml" not in clean
    assert "## 改动清单" in clean


def test_fallback_add_field_plan_from_evidence():
    from app.agent.explore_agent import _fallback_add_field_plan_from_evidence

    q = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    retrieved = [
        {
            "fqcn": "com.demo.loan.dto.WithdrawalsApplyReqVo",
            "method_name": "",
            "raw_code": "private String marketingLabel;\nprivate String sceneCode;",
            "file_path": "vo/WithdrawalsApplyReqVo.java",
        },
        {
            "fqcn": "com.demo.loan.controller.WithdrawalsController",
            "method_name": "checkApplyReqParam",
            "raw_code": 'reqVo.setMarketingLabel((String) reqMap.get("marketingLabel"));',
            "file_path": "controller/WithdrawalsController.java",
        },
        {
            "fqcn": "com.demo.loan.service.impl.WithdrawalsApplyServiceImpl",
            "method_name": "reqVo2ExtDmo",
            "raw_code": "ext.setMarketingLabel(reqVo.getMarketingLabel());",
            "file_path": "service/impl/WithdrawalsApplyServiceImpl.java",
        },
        {
            "fqcn": "com.demo.loan.entity.WithdrawalsExt",
            "method_name": "",
            "raw_code": '@TableName(value = "demo_b_withdrawals_ext")\n'
            "public class WithdrawalsExt {\nprivate String marketingLabel;\n}",
            "file_path": "entity/WithdrawalsExt.java",
        },
    ]
    plan = _fallback_add_field_plan_from_evidence(q, retrieved)
    assert plan
    assert "## 改动清单" in plan or "## 改动顺序" in plan
    assert "## 改动清单" not in plan
    assert "checkApplyReqParam" in plan
    assert "reqVo2ExtDmo" in plan
    assert "WithdrawalsExt" in plan
    assert "ALTER TABLE demo_b_withdrawals_ext" in plan
    assert "ExtMapper.xml" not in plan or "不必改" in plan
    assert "\n\n## " in plan


def test_partial_add_field_plan_never_dumps_internal_gate():
    from app.agent.explore_agent import _partial_add_field_plan_from_evidence

    q = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    plan = _partial_add_field_plan_from_evidence(
        q,
        [
            {
                "fqcn": "com.example.FooApplyReqVo",
                "method_name": "",
                "raw_code": "private String a;\nprivate String b;\n",
                "file_path": "vo/FooApplyReqVo.java",
            }
        ],
        gap_note="尚缺：入参绑定、落库组装",
    )
    assert "## 改动清单" in plan or "## 改动顺序" in plan
    assert "## 改动清单" not in plan
    assert "探索步数用尽" not in plan
    assert "get_symbol_code" not in plan
    assert "businessLabel" in plan
    assert "说明" in plan
    # Partial: only found VO — no fake bind/assemble/entity placeholders
    assert "证据不足，类名待补" not in plan
    assert "尚未读到具体方法" not in plan
    assert "尚未读到 @TableName 实体" not in plan
    assert "FooApplyReqVo" in plan
    assert "\n2. " not in plan


def test_add_field_seed_queries_from_entry_prefix():
    from app.agent.explore_agent import _add_field_seed_queries

    qs = _add_field_seed_queries(
        "WithdrawalsController#withdrawals",
        "我想放款接口增加一个字段并落库，都要修改哪些",
    )
    assert any("ReqVo" in x for x in qs)
    assert any(x.startswith("Withdrawals") for x in qs)


def test_add_field_finish_requires_multi_file_reads():
    """Shallow 1–2 reads must not pass finish; full analog path must."""
    from app.agent.explore_agent import (
        _ADD_FIELD_MIN_READS,
        _adaptive_max_steps,
        _add_field_evidence_gaps,
        _core_chain_ready,
        _finish_block_reason,
    )

    q = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    assert _adaptive_max_steps(q) >= 12
    assert _ADD_FIELD_MIN_READS >= 4

    shallow_steps = [
        {"tool": "search_symbols", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    shallow_retrieved = [
        {
            "fqcn": "com.demo.loan.dto.WithdrawalsApplyReqVo",
            "method_name": "",
            "raw_code": "private String marketingLabel;\nprivate String sceneCode;",
            "file_path": "…/WithdrawalsApplyReqVo.java",
        },
        {
            "fqcn": "com.demo.loan.controller.WithdrawalsController",
            "method_name": "checkApplyReqParam",
            "raw_code": 'reqMap.get("marketingLabel");',
            "file_path": "…/WithdrawalsController.java",
        },
    ]
    block = _finish_block_reason(shallow_steps, shallow_retrieved, question=q)
    assert block is not None
    assert "至少成功 get_symbol_code" in block or "角色" in block or "请求 VO" in block
    assert not _core_chain_ready(shallow_steps, shallow_retrieved, question=q)

    full_steps = [
        {"tool": "search_symbols", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    full_retrieved = [
        {
            "fqcn": "com.demo.loan.dto.WithdrawalsApplyReqVo",
            "method_name": "",
            "raw_code": "private String marketingLabel;\nprivate String sceneCode;",
            "file_path": "vo/WithdrawalsApplyReqVo.java",
        },
        {
            "fqcn": "com.demo.loan.controller.WithdrawalsController",
            "method_name": "checkApplyReqParam",
            "raw_code": 'reqVo.setMarketingLabel((String) reqMap.get("marketingLabel"));',
            "file_path": "controller/WithdrawalsController.java",
        },
        {
            "fqcn": "com.demo.loan.service.impl.WithdrawalsApplyServiceImpl",
            "method_name": "reqVo2ExtDmo",
            "raw_code": "ext.setMarketingLabel(reqVo.getMarketingLabel());",
            "file_path": "service/impl/WithdrawalsApplyServiceImpl.java",
        },
        {
            "fqcn": "com.demo.loan.entity.WithdrawalsExt",
            "method_name": "",
            "raw_code": '@TableName(value = "demo_b_withdrawals_ext")\n'
            "public class WithdrawalsExt {\nprivate String marketingLabel;\n}",
            "file_path": "entity/WithdrawalsExt.java",
        },
    ]
    assert _add_field_evidence_gaps(full_steps, full_retrieved) is None
    assert _finish_block_reason(full_steps, full_retrieved, question=q) is None
    assert _core_chain_ready(full_steps, full_retrieved, question=q)

    # Near budget: soft gate accepts request-side + persist-side.
    soft_ok = _finish_block_reason(
        full_steps[:4],
        full_retrieved[:3],
        question=q,
        soft=True,
    )
    # 3 reads with vo+bind+assemble should soft-pass
    assert soft_ok is None or "尚缺" not in (soft_ok or "")


def test_discover_add_field_plan_tool_satisfies_finish_gate():
    """Successful discover_add_field_plan step is enough — no multi get_symbol_code required."""
    from app.agent.explore_agent import (
        _add_field_evidence_gaps,
        _core_chain_ready,
        _finish_block_reason,
    )

    q = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    steps = [
        {
            "tool": "discover_add_field_plan",
            "ok": True,
            "roles": ["constant", "request_vo", "bind", "assemble", "persist_entity"],
        }
    ]
    assert _add_field_evidence_gaps(steps, []) is None
    assert _finish_block_reason(steps, [], question=q) is None
    assert _core_chain_ready(steps, [], question=q)


def test_graph_agent_has_no_add_field_early_return():
    """graph_agent must not short-circuit add-field before run_explore_agent."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "agent" / "graph_agent.py"
    text = src.read_text(encoding="utf-8")
    assert "run_add_field_plan" not in text
    # Must still route into explore for add-field (lexical lock ok).
    assert "run_explore_agent" in text
    assert "wants_add_field_change_plan" in text


def test_explore_agent_add_field_uses_discover_fast_path():
    """explore_agent should prefer discover_add_field_plan (not free LLM rewrite)."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "app" / "agent" / "explore_agent.py"
    text = src.read_text(encoding="utf-8")
    assert "from app.agent.change_impact import run_add_field_plan" not in text
    assert "discover_add_field_plan" in text
    # Fast path returns structured plan; must not invent hardcoded Withdrawals* list.
    assert "checkApplyReqParam" not in text or "discover_add_field_plan" in text
    assert '"source": "change_impact"' in text or "结构化改动方案已生成" in text
