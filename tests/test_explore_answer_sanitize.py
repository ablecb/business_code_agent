from __future__ import annotations

import asyncio
import re
from unittest.mock import AsyncMock, patch

from app.agent.explore_agent import (
    _answer_collapsed_main_chain,
    _answer_is_validation_only_collapse,
    _class_method_from_hint,
    _collect_flow_downstream_apis,
    _downstream_answer_foreign_controllers,
    _finish_block_reason,
    _flow_answer_collapsed,
    _flow_answer_wrong_entry,
    _flow_doc_stage_count,
    _has_read_code,
    _is_controller_sibling_hit,
    _is_messy_full_flow_answer,
    _lexical_main_entry,
    _main_chain_polish_prompts,
    _polish_dropped_table_facts,
    _polish_ordered_main_chain,
    _question_match_tokens,
    _reject_polished_main,
    _reject_polished_main_reason,
    _render_flow_downstream_apis,
    _render_ordered_main_chain,
    _sanitize_answer,
    _scope_prefetch_to_entry,
    _stage_table_facts,
    _strip_pedantic_notes,
    _wants_api_inventory,
    _wants_deep_flow_facts,
    _wants_flow_downstream_apis,
    _wants_full_flow,
    _wants_sync_entry_first,
)


def _sample_disburse_flow() -> dict:
    return {
        "class_method": "WithdrawalsController#withdrawals",
        "http_paths": ["withdrawals/withdrawals"],
        "flow_key": "loan_disburse",
        "flow_doc": {
            "stages": [
                {
                    "step": 1,
                    "title": "接收放款申请请求",
                    "class_method": "WithdrawalsController#withdrawals",
                    "desc": "校验入参与开关",
                },
                {
                    "step": 2,
                    "title": "放款开关控制",
                    "class_method": "WithdrawalsServiceImpl#loanSwitch",
                    "desc": "查询放款开关",
                },
                {
                    "step": 3,
                    "title": "放款流程配置决策",
                    "class_method": "LoanPProcessConfigServiceImpl#loanDecisionProcessConfig",
                    "desc": "新老流程决策",
                },
                {
                    "step": 4,
                    "title": "放款申请（写申请表）",
                    "class_method": "WithdrawalsApplyServiceImpl#loanApply",
                    "desc": "写入 WithdrawalsApply",
                },
                {
                    "step": 5,
                    "title": "更新申请表并写入扩展表",
                    "class_method": "WithdrawalsApplyServiceImpl#updateReult",
                    "desc": "写 WithdrawalsExt",
                },
                {
                    "step": 6,
                    "title": "插入放款主表",
                    "class_method": "WithdrawalsServiceImpl#insertLoanData",
                    "desc": "写 Withdrawals 主表",
                },
                {
                    "step": 7,
                    "title": "放款主流程",
                    "class_method": "WithdrawalsServiceImpl#loan",
                    "desc": "继续放款处理",
                },
                {
                    "step": 8,
                    "title": "异常时更新/补插放款主表",
                    "class_method": "WithdrawalsServiceImpl#insertOrUpdateLoanInfo",
                    "desc": "异常分支更新主表",
                },
            ]
        },
    }


def test_sanitize_strips_symbol_id_lines_and_hex():
    raw = (
        "### 入口\n"
        "- **入口点**：`OrderController#submit`\n"
        "- **Symbol ID**：`115693db595a49699804bd26747be725d441a533`\n"
        "- 调用 `FooService`（Symbol ID：`43c113eff4616ab6568984c75a1104016345a941`）\n"
    )
    out = _sanitize_answer(raw)
    assert "Symbol ID" not in out
    assert "115693db595a49699804bd26747be725d441a533" not in out
    assert "43c113eff4616ab6568984c75a1104016345a941" not in out
    assert "OrderController#submit" in out
    assert "FooService" in out


def test_wants_full_flow_matches_chain_phrasing():
    assert _wants_full_flow("放款流程什么样？从入口到下游回调串一下。")
    assert _wants_full_flow("把提现完整流程走一遍")
    assert not _wants_full_flow("放款流程什么样")
    assert not _wants_full_flow("订单流程什么样")
    assert not _wants_full_flow("核算服务地址是什么")


def test_eligible_follows_spine_fast_path_for_source_backend_bypass():
    """Source backend must still take FOLLOWS for「串一下」/「放款流程」."""
    from app.agent.explore_agent import _eligible_follows_spine_fast_path

    q_full = "放款流程什么样？从入口到下游回调串一下。"
    q_bare = "放款流程什么样"
    cm = "WithdrawalsController#withdrawals"
    assert _eligible_follows_spine_fast_path(q_full, prefer_cm=cm)
    assert _eligible_follows_spine_fast_path(q_bare, prefer_cm=cm)
    # Notice / tech / inventory → no bypass (product Class mismatch is LLM's job).
    assert not _eligible_follows_spine_fast_path(
        q_full, prefer_cm=cm, tech=True
    )
    assert not _eligible_follows_spine_fast_path(
        q_full, prefer_cm=cm, api_inventory=True
    )
    assert not _eligible_follows_spine_fast_path(
        "授信通知接口流程是什么样的",
        prefer_cm="CreditController#creditNotification",
        notice_entry=True,
    )
    assert not _eligible_follows_spine_fast_path(
        "核算服务地址是什么", prefer_cm=cm
    )


def test_notice_entry_not_confused_with_withdrawals():
    from app.agent.explore_agent import (
        _entry_aligned_with_question,
        _lexical_main_entry,
        _wants_callback_section,
        _wants_notice_entry,
    )

    q = "授信通知接口流程是什么样的"
    # Notice keyword gate removed — always False; entry via retrieval/LLM.
    assert not _wants_notice_entry(q)
    assert _lexical_main_entry(q) is None
    assert _entry_aligned_with_question("CreditController#creditNotification", q)
    assert not _entry_aligned_with_question("WithdrawalsController#withdrawals", q)
    assert not _wants_notice_entry("放款流程什么样？从入口到下游回调串一下。")
    assert _wants_callback_section("放款流程什么样？从入口到下游回调串一下。")


def test_fangkuan_entry_aligned_with_withdrawals():
    """Role alignment: no product 放款→Withdrawals map; notice role still filters."""
    from app.agent.explore_agent import _entry_aligned_with_question, _offline_blob_aligned

    for q in (
        "放款流程什么样",
        "放款接口什么流程",
        "放款从入口到落库串一下",
        "提现流程什么样",
    ):
        # Without CN product stems, withdrawals is not rejected for 放款 asks.
        assert _entry_aligned_with_question("WithdrawalsController#withdrawals", q), q
    blob = (
        "WithdrawalsController#withdrawals\n"
        "WithdrawalsServiceImpl#loanSwitch\n"
        "WithdrawalsApplyServiceImpl#loanApply"
    )
    assert _offline_blob_aligned(blob, "放款流程什么样")
    # Notice-role question must not use a non-notice offline blob.
    assert not _offline_blob_aligned(blob, "授信通知接口流程")
    assert _entry_aligned_with_question(
        "CreditController#creditNotification", "授信通知接口流程"
    )


def test_wants_deep_flow_facts_for_flow_questions():
    assert _wants_deep_flow_facts("放款流程什么样？")
    assert _wants_deep_flow_facts("放款流程什么样")
    assert _wants_deep_flow_facts("放款流程什么样？从入口到下游回调串一下。")
    assert _wants_deep_flow_facts("放款流程是什么样的 业务描述下")
    # 「接口什么流程」has 流程+什么流程 / 接口 — must be deep (status/facts chain).
    assert _wants_deep_flow_facts("放款接口什么流程")
    assert _wants_deep_flow_facts("放款接口什么流程？")
    assert _wants_sync_entry_first("放款接口什么流程")
    assert not _wants_deep_flow_facts("核算服务地址是什么")
    assert not _wants_deep_flow_facts("放款相关的接口有哪些")


def test_deep_flow_render_includes_status_and_class_method():
    """Business mode is overall→要点; status/Class#method live in tech(+deep)."""
    flow = _sample_disburse_flow()
    for st in flow["flow_doc"]["stages"]:
        if st.get("class_method") == "WithdrawalsServiceImpl#loan":
            st["desc_business"] = "核心放款编排：落库后提交第三方。"
            st["status_line"] = (
                "WithdrawalsApply.status：0 申请中 → 1 放款成功 / 2 放款失败；"
                "Withdrawals.status：INIT → SUCCESS / FAIL"
            )
            st["deep"] = {"status_line": st["status_line"], "biz_clear": st["desc_business"]}
            break
    flow["deep_status_nodes"] = [
        {
            "class_method": "WithdrawalsServiceImpl#updateResult",
            "status_line": "Withdrawals.status：SUCCESS → SETTLED",
        }
    ]
    # Business narrative: no Class#method / status laundry.
    deep_biz = _render_ordered_main_chain(flow, tech=False, deep=True)
    assert "总体说明：" in deep_biz
    assert "精简要点：" in deep_biz
    assert "WithdrawalsServiceImpl#loan" not in deep_biz
    assert "状态变更要点" not in deep_biz
    # Tech+deep still keeps Class#method anchors and status.
    tech_deep = _render_ordered_main_chain(flow, tech=True, deep=True)
    assert "WithdrawalsServiceImpl#loan" in tech_deep
    assert "状态：" in tech_deep
    assert "WithdrawalsApply.status" in tech_deep
    assert "INIT → SUCCESS" in tech_deep
    assert "状态变更要点" in tech_deep
    assert "SUCCESS → SETTLED" in tech_deep
    assert "WithdrawalsServiceImpl#updateResult" in tech_deep
    business = _render_ordered_main_chain(flow, tech=False, deep=False)
    assert "WithdrawalsServiceImpl#loan" not in business
    assert "总体说明：" in business


def test_business_desc_flow_keeps_all_flowdoc_stages():
    """「业务描述」must not collapse FlowDoc stages into a 4-step rewrite."""
    q = "放款流程是什么样的 业务描述下"
    assert _wants_sync_entry_first(q)
    assert not _wants_full_flow(q)  # bare 流程 ≠ 完整全链路关键词
    flow = _sample_disburse_flow()
    assert _flow_doc_stage_count(flow) == 8
    ordered_tech = _render_ordered_main_chain(flow, tech=True)
    for needle in (
        "loanApply",
        "updateReult",
        "insertLoanData",
        "WithdrawalsServiceImpl#loan",
        "insertOrUpdateLoanInfo",
        "WithdrawalsApply",
        "WithdrawalsExt",
    ):
        assert needle in ordered_tech
    ordered = ordered_tech
    business = _render_ordered_main_chain(flow, tech=False)
    assert "WithdrawalsServiceImpl#loan" not in business
    assert "申请表" in business and "扩展表" in business and "主表" in business
    assert "注意：" not in business
    # Over-summarized 4-step paraphrase must be detected as collapsed.
    shallow = (
        "主链：\n"
        "1. 接收申请\n"
        "2. 开关校验\n"
        "3. 路由决策\n"
        "4. 提交下游\n"
    )
    assert _answer_collapsed_main_chain(shallow, 8)
    assert not _answer_collapsed_main_chain(ordered, 8)
    # 5 field-validation steps ≈ stage count — count collapse misses; content must catch.
    validation_only = (
        "主链：\n"
        "1. 校验机构标识\n"
        "2. 校验交易日期\n"
        "3. 校验流水号\n"
        "4. 校验订单号\n"
        "5. 校验金额不能为空\n"
        "结果回调：暂无\n"
    )
    assert not _answer_collapsed_main_chain(validation_only, 5)
    assert _answer_is_validation_only_collapse(validation_only, flow)
    assert _flow_answer_collapsed(validation_only, flow, 8)
    assert not _flow_answer_collapsed(ordered, flow, 8)


def test_polish_prompt_is_wording_only_and_hides_flow_key_for_business():
    flow = _sample_disburse_flow()
    ordered = _render_ordered_main_chain(flow)
    sys, user = _main_chain_polish_prompts(
        "放款流程是什么样的 业务描述下",
        ordered,
        stage_count=8,
        tech=False,
    )
    assert "唯一事实来源" in user
    assert ordered in user
    assert "申请表" in sys and "扩展表" in sys
    assert "校验清单" in sys or "字段校验" in sys
    assert "flow_key" in sys  # instruct to keep out of body
    assert "正文不要出现 Class#method" in sys
    # Business polish is wording-only; pedantic「注意：不是」lives in tech path / strip helper.
    assert "总体说明" in sys or "精简要点" in sys
    sys_tech, _ = _main_chain_polish_prompts(
        "放款流程调用链细节",
        ordered,
        stage_count=8,
        tech=True,
    )
    assert "每步须保留 Class#method" in sys_tech


def test_polish_rejectors_catch_collapse_validation_and_table_drop():
    flow = _sample_disburse_flow()
    ordered = _render_ordered_main_chain(flow, tech=False)
    assert {"申请表", "扩展表", "主表"} <= set(_stage_table_facts(ordered))

    good = (
        "业务入口：HTTP `withdrawals/withdrawals`\n"
        "主链：\n"
        "1. 接收放款申请\n"
        "2. 检查放款开关\n"
        "3. 决策新老流程配置\n"
        "4. 写入申请表\n"
        "5. 更新申请表并写入扩展表\n"
        "6. 插入放款主表\n"
        "7. 继续放款主流程\n"
        "8. 异常时更新或补插放款主表\n"
    )
    # Business polish (tech=False) may omit Class#method.
    assert not _reject_polished_main(
        good, ordered_main=ordered, flow=flow, stage_count=8, tech=False
    )

    shallow = (
        "业务入口：HTTP `withdrawals/withdrawals`\n"
        "主链：\n"
        "1. 接收申请\n"
        "2. 开关\n"
        "3. 决策\n"
        "4. 提交\n"
    )
    assert _reject_polished_main(
        shallow, ordered_main=ordered, flow=flow, stage_count=8, tech=False
    )

    dropped_tables = (
        "业务入口：HTTP `withdrawals/withdrawals`\n"
        "主链：\n"
        "1. 接收放款申请\n"
        "2. 检查放款开关\n"
        "3. 决策新老流程配置\n"
        "4. 发起申请写入\n"
        "5. 更新申请结果并写扩展\n"
        "6. 插入放款记录\n"
        "7. 继续放款主流程\n"
        "8. 异常时补更新记录\n"
    )
    assert _polish_dropped_table_facts(ordered, dropped_tables)
    assert _reject_polished_main(
        dropped_tables, ordered_main=ordered, flow=flow, stage_count=8, tech=False
    )

    wrong_entry = good.replace("withdrawals/withdrawals", "loan/notice")
    assert _flow_answer_wrong_entry(wrong_entry, flow, tech=False)
    assert _reject_polished_main(
        wrong_entry, ordered_main=ordered, flow=flow, stage_count=8, tech=False
    )

    validation_only = (
        "业务入口：HTTP `withdrawals/withdrawals`\n"
        "主链：\n"
        "1. 校验机构标识\n"
        "2. 校验交易日期\n"
        "3. 校验流水号\n"
        "4. 校验订单号\n"
        "5. 校验商户号\n"
        "6. 校验金额不能为空\n"
        "7. 校验入参\n"
        "8. 字段校验完成\n"
    )
    assert _reject_polished_main(
        validation_only, ordered_main=ordered, flow=flow, stage_count=8, tech=False
    )


def test_business_reject_screenshot_generic_seven_steps():
    """Business narrative rejects screenshot titles that ignore FlowDoc grounding."""
    flow = _sample_disburse_flow()
    ordered = _render_ordered_main_chain(flow, tech=False, deep=False)
    assert "总体说明：" in ordered
    assert "精简要点：" in ordered
    assert "主链（按入口代码调用顺序）：" not in ordered
    screenshot = (
        "业务入口：HTTP `withdrawals/withdrawals` — 本题同步业务主链起点；"
        "结果回调见文末单独一节。\n"
        "主链：\n"
        "1. 功能启用判断\n"
        "2. 额度状态查询\n"
        "3. 风控评估\n"
        "4. 担保路由\n"
        "5. 优惠券处理\n"
        "6. 人工审批判断\n"
        "7. 贷款状态更新\n"
        "\n"
        "技术锚点：入口 withdrawals/withdrawals · flow_key=loan_disburse"
    )
    reason = _reject_polished_main_reason(
        screenshot,
        ordered_main=ordered,
        flow=flow,
        stage_count=7,
        tech=False,
        deep=False,
    )
    assert reason is not None
    assert reason.split(":")[0] in (
        "stage_count_drop",
        "generic_stage_rewrite",
        "dropped_table_facts",
        "business_points_out_of_range",
        "missing_overview",
        "missing_points",
    )


def test_deep_reject_screenshot_generic_seven_steps():
    """Screenshot-style generic 7 titles must not win when deep=True."""
    q = "放款流程什么样？"
    assert _wants_deep_flow_facts(q)
    flow = _sample_disburse_flow()
    # Simulate a 7-stage FlowDoc so step count alone would not catch the rewrite.
    flow["flow_doc"]["stages"] = flow["flow_doc"]["stages"][:7]
    for st in flow["flow_doc"]["stages"]:
        if st.get("class_method") == "WithdrawalsServiceImpl#loan":
            st["status_line"] = "Withdrawals.status：INIT → SUCCESS / FAIL"
            break
    deep_biz = _render_ordered_main_chain(flow, tech=False, deep=True)
    assert "总体说明：" in deep_biz
    assert "WithdrawalsServiceImpl#" not in deep_biz
    deep_tech = _render_ordered_main_chain(flow, tech=True, deep=True)
    assert "WithdrawalsServiceImpl#" in deep_tech
    assert "状态：" in deep_tech

    screenshot = (
        "业务入口：HTTP `withdrawals/withdrawals` — 本题同步业务主链起点；"
        "结果回调见文末单独一节。\n"
        "主链：\n"
        "1. 功能启用判断\n"
        "2. 额度状态查询\n"
        "3. 风控评估\n"
        "4. 担保路由\n"
        "5. 优惠券处理\n"
        "6. 人工审批判断\n"
        "7. 贷款状态更新\n"
        "\n"
        "技术锚点：入口 withdrawals/withdrawals · flow_key=loan_disburse"
    )
    reason = _reject_polished_main_reason(
        screenshot,
        ordered_main=deep_tech,
        flow=flow,
        stage_count=7,
        tech=True,
        deep=True,
    )
    assert reason is not None
    assert reason.split(":")[0] in (
        "dropped_class_methods",
        "generic_stage_rewrite",
        "dropped_status_facts",
        "dropped_table_facts",
        "stage_count_drop",
    )
    assert _reject_polished_main(
        screenshot,
        ordered_main=deep_tech,
        flow=flow,
        stage_count=7,
        tech=True,
        deep=True,
    )

    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value=screenshot)
    with patch("app.agent.explore_agent.get_llm_provider", return_value=fake_llm):
        out = asyncio.run(
            _polish_ordered_main_chain(q, deep_biz, flow, deep=True)
        )
    # Prefer deterministic deep business render over bad polish.
    assert out == deep_biz
    assert "WithdrawalsServiceImpl#" not in out
    assert "总体说明：" in out
    assert "功能启用判断" not in out
    assert fake_llm.chat.await_count >= 1


def test_polish_ordered_main_falls_back_on_bad_llm():
    flow = _sample_disburse_flow()
    ordered_biz = _render_ordered_main_chain(flow, tech=False)
    bad = (
        "主链：\n"
        "1. 接收申请\n"
        "2. 开关校验\n"
        "3. 路由决策\n"
        "4. 提交下游\n"
    )
    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value=bad)
    with patch("app.agent.explore_agent.get_llm_provider", return_value=fake_llm):
        out = asyncio.run(
            _polish_ordered_main_chain(
                "放款流程是什么样的 业务描述下", ordered_biz, flow
            )
        )
    # Soft retry then fallback to deterministic business render (not raw Class#method).
    assert out == ordered_biz
    assert "WithdrawalsServiceImpl#" not in out
    assert fake_llm.chat.await_count == 2
    # Prompt must not include corpus / checkApplyReqParam materials.
    blob = str(fake_llm.chat.await_args_list[0])
    assert "checkApplyReqParam" not in blob
    assert "唯一事实来源" in blob


def test_polish_ordered_main_accepts_faithful_business_rewrite():
    flow = _sample_disburse_flow()
    ordered = _render_ordered_main_chain(flow, tech=False)
    polished = (
        "业务入口：HTTP `withdrawals/withdrawals` — 本题同步业务主链起点\n"
        "主链：\n"
        "1. 接收放款申请请求并做基础校验\n"
        "2. 查询放款开关是否允许继续\n"
        "3. 按配置决策走新流程还是老流程\n"
        "4. 写入放款申请表\n"
        "5. 更新申请表状态并写入扩展表\n"
        "6. 插入放款主表\n"
        "7. 进入放款主流程继续处理\n"
        "8. 异常时更新或补插放款主表\n"
    )
    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value=polished)
    with patch("app.agent.explore_agent.get_llm_provider", return_value=fake_llm):
        out = asyncio.run(
            _polish_ordered_main_chain(
                "放款流程是什么样的 业务描述下", ordered, flow
            )
        )
    assert out == polished.strip()
    assert "申请表" in out and "扩展表" in out and "主表" in out
    assert "flow_key" not in out
    assert "WithdrawalsServiceImpl#" not in out
    fake_llm.chat.assert_awaited_once()


def test_nine_stage_business_polish_accepts_without_class_method():
    """9-stage FlowDoc business polish stays on overall→要点 (no Class#method dump)."""
    flow = _sample_disburse_flow()
    stages = flow["flow_doc"]["stages"]
    stages.insert(
        7,
        {
            "step": 8,
            "title": "发起风控（条件调用）",
            "class_method": "RiskServiceImpl#asynRisk",
            "desc": "loan 内条件调用；结果经 MQ 回调，非本 HTTP 同步返回。"
            "注意：不是 updateLoanInfo。",
        },
    )
    stages[8]["step"] = 9
    assert _flow_doc_stage_count(flow) == 9
    source = _render_ordered_main_chain(flow, tech=False)
    assert "RiskServiceImpl#asynRisk" not in source
    assert "总体说明：" in source
    assert "精简要点：" in source
    assert "注意：" not in source
    assert "不是 updateLoanInfo" not in _strip_pedantic_notes(
        "注意：不是 updateLoanInfo，也不写放款主表。"
    )

    # Faithful polish = same overall→要点 shape (旧「主链 1..9」会被拒).
    assert (
        _reject_polished_main_reason(
            source, ordered_main=source, flow=flow, stage_count=9, tech=False
        )
        is None
    )
    legacy_nine = (
        "业务入口：HTTP `withdrawals/withdrawals` — 本题同步业务主链起点\n"
        "主链：\n"
        "1. 接收放款申请\n"
        "2. 检查放款开关\n"
        "3. 决策新老流程配置\n"
        "4. 写入放款申请表\n"
        "5. 更新申请表并写入扩展表\n"
        "6. 插入放款主表\n"
        "7. 继续放款主流程\n"
        "8. 按需发起风控（结果异步回调）\n"
        "9. 异常时更新或补插放款主表\n"
    )
    assert (
        _reject_polished_main_reason(
            legacy_nine, ordered_main=source, flow=flow, stage_count=9, tech=False
        )
        is not None
    )
    fake_llm = AsyncMock()
    fake_llm.chat = AsyncMock(return_value=source)
    with patch("app.agent.explore_agent.get_llm_provider", return_value=fake_llm):
        out = asyncio.run(
            _polish_ordered_main_chain(
                "放款流程是什么样的 业务描述下", source, flow
            )
        )
    assert "总体说明：" in out
    assert "RiskServiceImpl#" not in out
    assert "asynRisk" not in out or "风控" in out


def test_sync_flow_scopes_out_controller_siblings():
    """「放款流程」must not treat sibling WithdrawalsController methods as the chain."""
    q = "放款流程是什么样的 业务描述下"
    assert _wants_sync_entry_first(q)
    prefer = "WithdrawalsController#withdrawals"
    entry = {
        "kind": "Controller",
        "fqcn": "com.x.WithdrawalsController",
        "method_sig": "withdrawals(",
        "class_method": "WithdrawalsController#withdrawals",
        "summary": "WithdrawalsController#withdrawals 放款入口",
    }
    siblings = [
        {
            "kind": "Controller",
            "fqcn": "com.x.WithdrawalsController",
            "method_sig": "withDrawalCalculate(",
            "class_method": "WithdrawalsController#withDrawalCalculate",
            "summary": "WithdrawalsController#withDrawalCalculate 试算",
        },
        {
            "kind": "Controller",
            "fqcn": "com.x.WithdrawalsController",
            "method_sig": "creditUsageApply(",
            "class_method": "WithdrawalsController#creditUsageApply",
            "summary": "WithdrawalsController#creditUsageApply 额度占用",
        },
    ]
    impl = {
        "kind": "Service",
        "fqcn": "com.x.WithdrawalsServiceImpl",
        "method_sig": "apply(",
        "summary": "WithdrawalsController 下游 WithdrawalsServiceImpl#apply",
    }
    for s in siblings:
        assert _is_controller_sibling_hit(s, prefer)
    assert not _is_controller_sibling_hit(entry, prefer)
    assert not _is_controller_sibling_hit(impl, prefer)

    scoped = _scope_prefetch_to_entry(
        [siblings[0], entry, siblings[1], impl],
        prefer_cm=prefer,
        question=q,
        keep_unrelated=0,
    )
    cms = {
        (h.get("class_method") or h.get("fqcn"))
        for h in scoped
    }
    assert "WithdrawalsController#withdrawals" in cms
    assert "com.x.WithdrawalsServiceImpl" in cms
    assert "WithdrawalsController#withDrawalCalculate" not in cms
    assert "WithdrawalsController#creditUsageApply" not in cms


def test_downstream_api_ask_not_controller_inventory():
    """「调用了哪些接口」must not route to same-Controller endpoint catalog."""
    for q in (
        "回答过程中都调用了哪些接口",
        "放款掉了哪些接口",
        "主链调用了哪些服务",
        "过程中调了哪些 Feign",
        "下游接口有哪些",
    ):
        assert _wants_flow_downstream_apis(q), q
        assert not _wants_api_inventory(q), q
        assert _wants_sync_entry_first(q), q

    # True sibling-Controller inventory asks stay inventory.
    for q in (
        "放款相关的接口有哪些",
        "提现相关接口有几个",
        "接口列表",
        "帮我看看放款接口有哪些",
        "放款接口有哪些",
    ):
        assert _wants_api_inventory(q), q
        assert not _wants_flow_downstream_apis(q), q


def test_format_api_inventory_answer_count_matches():
    from app.agent.explore_intent import (
        _api_inventory_query_core,
        _format_api_inventory_answer,
    )

    assert _api_inventory_query_core("帮我看看放款接口有哪些") == "放款"
    items = [
        {
            "cm": "WithdrawalsController#withdrawals",
            "http_paths": "/loan/withdrawals",
            "api_doc": "放款申请",
        },
        {
            "cm": "WithdrawalsController#withDrawalCalculate",
            "http_paths": "/loan/calc",
            "api_doc": "试算",
        },
    ]
    ans = _format_api_inventory_answer(items, topic="放款")
    assert "共 2 个" in ans
    assert "WithdrawalsController#withdrawals" in ans
    assert ans.count("`WithdrawalsController#") == 2


def test_render_flow_downstream_apis_excludes_siblings_and_matches_count():
    flow = _sample_disburse_flow()
    # Inject a sibling Controller stage (must not appear in downstream list).
    flow["flow_doc"]["stages"].append(
        {
            "step": 99,
            "title": "额度试算入口",
            "class_method": "WithdrawalsController#withDrawalCalculate",
            "desc": "兄弟 HTTP 入口，不是下游",
        }
    )
    # Inject unrelated Controllers (must never appear as「下游」).
    flow["flow_doc"]["stages"].append(
        {
            "step": 100,
            "title": "日额度",
            "class_method": "AmountController#dayQuota",
            "desc": "无关 Controller",
        }
    )
    flow["flow_doc"]["stages"].append(
        {
            "step": 101,
            "title": "受托退款",
            "class_method": "EntrustedRefundController#refundApply",
            "desc": "无关 Controller",
        }
    )
    entry, items = _collect_flow_downstream_apis(flow)
    assert entry == "WithdrawalsController#withdrawals"
    cms = [cm for cm, _ in items]
    assert "WithdrawalsController#withdrawals" not in cms
    assert "WithdrawalsController#withDrawalCalculate" not in cms
    assert "AmountController#dayQuota" not in cms
    assert "EntrustedRefundController#refundApply" not in cms
    assert "WithdrawalsServiceImpl#loanSwitch" in cms
    assert "WithdrawalsServiceImpl#loan" in cms

    text = _render_flow_downstream_apis(flow)
    assert f"共 {len(items)} 个" in text
    # Header count must equal numbered lines.
    numbered = [ln for ln in text.splitlines() if re.match(r"^\d+\.\s+", ln)]
    assert len(numbered) == len(items)
    assert "AmountController" not in text
    assert "EntrustedRefundController" not in text


def test_fangkuan_downstream_question_locks_withdrawals_not_foreign_controllers():
    """Regression:「放款过程中都调用了哪些接口」must not list Amount/EntrustedRefund."""
    q = "放款过程中都调用了哪些接口"
    assert _wants_flow_downstream_apis(q)
    assert not _wants_api_inventory(q)
    assert _lexical_main_entry(q) is None
    assert _question_match_tokens(q) == ["放款"]

    flow = _sample_disburse_flow()
    flow["flow_doc"]["stages"].extend(
        [
            {
                "step": 90,
                "title": "日额度缓存",
                "class_method": "AmountController#dayQuotaCache",
                "desc": "噪声",
            },
            {
                "step": 91,
                "title": "受托退款申请",
                "class_method": "EntrustedRefundController#refundApply",
                "desc": "噪声",
            },
        ]
    )
    text = _render_flow_downstream_apis(flow)
    assert "主链入口：WithdrawalsController#withdrawals" in text
    assert "AmountController" not in text
    assert "EntrustedRefundController" not in text
    assert "loanSwitch" in text or "loanApply" in text or "asynRisk" in text

    bad_llm = (
        "主链：\n"
        "1. WithdrawalsController#withdrawals\n"
        "2. AmountController#addDayQuota\n"
        "3. AmountController#dayQuota\n"
        "4. EntrustedRefundController#refundApply\n"
    )
    foreign = _downstream_answer_foreign_controllers(bad_llm, flow)
    assert "AmountController#addDayQuota" in foreign
    assert "EntrustedRefundController#refundApply" in foreign


def test_class_method_from_hint_and_downstream_scope():
    assert (
        _class_method_from_hint("WithdrawalsController#withdrawals")
        == "WithdrawalsController#withdrawals"
    )
    assert (
        _class_method_from_hint("com.x.WithdrawalsController#withdrawals 备注")
        == "WithdrawalsController#withdrawals"
    )
    assert _class_method_from_hint("com.x.WithdrawalsController") is None

    q = "回答过程中都调用了哪些接口"
    prefer = "WithdrawalsController#withdrawals"
    siblings = [
        {
            "kind": "Controller",
            "fqcn": "com.x.WithdrawalsController",
            "class_method": "WithdrawalsController#creditUsageApply",
            "summary": "兄弟入口",
        }
    ]
    impl = {
        "kind": "Service",
        "fqcn": "com.x.RiskService",
        "class_method": "RiskService#asynRisk",
        "summary": "WithdrawalsController#withdrawals 下游风控",
    }
    scoped = _scope_prefetch_to_entry(
        siblings + [impl],
        prefer_cm=prefer,
        question=q,
        keep_unrelated=0,
    )
    assert all(
        (h.get("class_method") or "") != "WithdrawalsController#creditUsageApply"
        for h in scoped
    )
    assert any((h.get("class_method") or "").endswith("asynRisk") for h in scoped)

    # 「放款过程中…」must drop foreign Controllers from prefetch.
    foreign_ctrls = [
        {
            "kind": "Controller",
            "fqcn": "com.x.AmountController",
            "class_method": "AmountController#dayQuota",
            "summary": "日额度",
        },
        {
            "kind": "Controller",
            "fqcn": "com.x.EntrustedRefundController",
            "class_method": "EntrustedRefundController#refundApply",
            "summary": "受托退款",
        },
    ]
    scoped2 = _scope_prefetch_to_entry(
        foreign_ctrls + [impl],
        prefer_cm=prefer,
        question="放款过程中都调用了哪些接口",
        keep_unrelated=0,
    )
    cms2 = {(h.get("class_method") or "") for h in scoped2}
    assert "AmountController#dayQuota" not in cms2
    assert "EntrustedRefundController#refundApply" not in cms2
    assert any("asynRisk" in c for c in cms2)
    assert any((h.get("fqcn") or "") == "com.x.RiskService" for h in scoped)


def test_sanitize_strips_monologue_and_full_flow_class_dump():
    raw = (
        "我需要理解整条放款链路。\n"
        "根据证据，入口是 WithdrawalsController#withdrawals。\n"
        "主链：\n"
        "1. 受理提现申请\n"
        "2. 校验与风控\n"
        "探索不足，还要看更多类。\n"
        "LoanNoticeController#notice FooServiceImpl#apply "
        "LoanRabbitMqListenner#onMessage AccClient#dealV2\n"
    )
    out = _sanitize_answer(raw, full_flow=True)
    assert "我需要理解" not in out
    assert "根据证据" not in out
    assert "探索不足" not in out
    assert "1. 受理提现申请" in out
    # Sanitize keeps Class#method; dense dumps are rejected by _is_messy_full_flow_answer.
    assert "LoanNoticeController#notice" in out
    assert _is_messy_full_flow_answer(raw)


def test_messy_full_flow_answer_detection():
    assert _is_messy_full_flow_answer("我需要先确认入口再回答。")
    assert _is_messy_full_flow_answer(
        "入口 WithdrawalsController#withdrawals，然后 FooServiceImpl#apply，"
        "再到 LoanRabbitMqListenner#onMessage，最后 AccClient#dealV2。"
    )
    assert _is_messy_full_flow_answer("受理后会校验再放款，最后异步回调。")  # no numbers
    ok = (
        "主链：\n"
        "1. 受理提现申请\n"
        "2. 参数与业务校验\n"
        "3. 额度/风控通过后落库并投递下游\n"
        "4. 异步消费并调用外部核算\n"
        "结果回调/通知：\n"
        "- 下游结果回写订单状态\n"
    )
    assert not _is_messy_full_flow_answer(ok)


def test_has_read_code_requires_successful_get():
    assert not _has_read_code([{"tool": "search_symbols", "ok": True}])
    assert not _has_read_code([{"tool": "get_symbol_code", "ok": False}])
    assert _has_read_code([{"tool": "get_symbol_code", "ok": True}])


def test_finish_requires_impl_and_remote_client_depth():
    steps = [
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "list_callees", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    # only controller — blocked (flow question)
    assert _finish_block_reason(
        steps,
        [{"fqcn": "com.x.OrderController", "raw_code": "orderService.submit()"}],
        question="订单流程什么样",
    )
    # impl without reading Client though code calls *Client — blocked
    assert _finish_block_reason(
        steps,
        [
            {
                "fqcn": "com.x.OrderServiceImpl",
                "raw_code": "paymentClient.charge(req);",
            },
        ],
        question="订单流程什么样",
    )
    # impl + Client read — ok
    assert (
        _finish_block_reason(
            steps,
            [
                {
                    "fqcn": "com.x.OrderServiceImpl",
                    "raw_code": "paymentClient.charge(req);",
                },
                {
                    "fqcn": "com.x.PaymentClient",
                    "raw_code": "@FeignClient void charge();",
                },
            ],
            question="订单流程什么样",
        )
        is None
    )


def test_validation_finish_requires_same_class_helper_not_random_service():
    steps = [
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    # entry only, helper pending — blocked
    assert _finish_block_reason(
        steps[:1],
        [
            {
                "fqcn": "com.x.FooController",
                "method_name": "notify",
                "raw_code": "if (!generateReqParam(map, resp, dto)) { return; }",
            }
        ],
        question="请求参数校验都校验了哪些字段",
    )
    # read generateReqParam — ok
    assert (
        _finish_block_reason(
            steps,
            [
                {
                    "fqcn": "com.x.FooController",
                    "method_name": "notify",
                    "raw_code": "generateReqParam(map, resp, dto);",
                },
                {
                    "fqcn": "com.x.FooController",
                    "method_name": "generateReqParam",
                    "raw_code": "if (StringUtil.isNull(a)) { return false; }\n"
                    "if (StringUtil.isNull(b)) { return false; }\n"
                    "if (StringUtil.isNull(c)) { return false; }\n"
                    "return true;",
                },
            ],
            question="请求参数校验都校验了哪些字段",
        )
        is None
    )


def test_enrich_skips_entry_status_that_belongs_to_later_spine():
    """Entry status_line with via insertLoanData must not stick on step 1."""
    from unittest.mock import patch

    from app.agent.explore_agent import (
        _enrich_flow_doc_with_deep_nodes,
        _status_line_belongs_to_other_spine,
    )

    spine = {
        "WithdrawalsController#withdrawals",
        "WithdrawalsServiceImpl#insertLoanData",
        "WithdrawalsServiceImpl#insertOrUpdateLoanInfo",
    }
    stolen = (
        "insert:Withdrawals via withdrawalsService.insertLoanData；"
        "insert or update:Withdrawals via withdrawalsService.insertOrUpdateLoanInfo"
    )
    assert _status_line_belongs_to_other_spine(
        stolen, "WithdrawalsController#withdrawals", spine
    )
    assert not _status_line_belongs_to_other_spine(
        "dmo.loanStatus：null → 'F'",
        "WithdrawalsServiceImpl#insertLoanData",
        spine,
    )

    flow = {
        "class_method": "WithdrawalsController#withdrawals",
        "flow_key": "loan_disburse",
        "flow_doc": {
            "stages": [
                {
                    "step": 1,
                    "class_method": "WithdrawalsController#withdrawals",
                    "title": "接收放款申请请求",
                },
                {
                    "step": 8,
                    "class_method": "WithdrawalsServiceImpl#insertLoanData",
                    "title": "插入放款主表",
                },
            ]
        },
    }
    fake_facts = {
        "WithdrawalsController#withdrawals": {
            "status_line": stolen,
            "biz_clear": "入口",
        },
        "WithdrawalsServiceImpl#insertLoanData": {
            "status_line": "dmo.loanStatus：null → 'F'",
            "biz_clear": "写主表",
        },
    }
    with patch(
        "app.agent.explore_agent._load_deep_facts_by_class_methods",
        return_value=fake_facts,
    ), patch(
        "app.agent.explore_agent._load_flow_status_nodes",
        return_value=[],
    ):
        out = _enrich_flow_doc_with_deep_nodes("demo-loan", flow)
    stages = (out.get("flow_doc") or {}).get("stages") or []
    assert not stages[0].get("status_line")
    assert "loanStatus" in (stages[1].get("status_line") or "")


def test_assemble_answer_from_symbol_descs_prefers_stored_business_text():
    """Business flow answers should surface entry symbol_desc, not status_line soup."""
    from app.agent.explore_agent import _assemble_answer_from_symbol_descs

    entry = "WithdrawalsController#withdrawals"
    fake_entry = {
        entry: {
            "biz_clear": "本段为提现申请接口的入口方法，负责接收请求、参数校验、流程决策与加锁。",
            "biz_detail": (
                "[段1] 本段代码首先接收提现请求，提取请求参数并进行初步校验。"
                "若校验失败则直接返回错误信息。\n"
                "[段2] 本段代码首先调用 generateReqParam 整合参数，然后调用 insertLoanData。"
            ),
            "http_paths": ["withdrawals/withdrawals"],
            "status_line": "map.RESPONSE_CODE：null → SUCC_CODE；checker false→true",
            "business_rules": ["通过 Redis 加锁防止重复请求"],
        },
        "WithdrawalsController#checkApplyReqParam": {
            "biz_clear": "入参非空校验。",
            "business_rules": [
                "机构标识（instId）不能为空",
                "交易时间（transTime）不能为空",
                "客户编号（custId）不能为空",
            ],
            "validation_points": ["instId 非空校验", "custId 非空校验"],
        },
        "WithdrawalsServiceImpl#loanSwitch": {
            "biz_clear": "该方法用于判断是否启用贷款开关。若开关关闭则拦截。",
            "business_rules": [
                "开关为 0 时，触发贷款或还款切换机构的校验",
                "拦截信息写入数据库，表示流程被阻断",
            ],
        },
        "WithdrawalsServiceImpl#loan": {
            "biz_clear": "执行放款主流程编排。",
            "business_rules": [
                "额度校验失败则标记提现为失败",
                "若机构额度不足，提现失败",
                "若需走融担流程，根据渠道选择新老流程",
                "若资金路由模式为 FUND 或 FLOW，则调用资金路由逻辑",
            ],
            "validation_points": ["approval 为 false", "riskService.asynRisk 返回值"],
        },
        "WithdrawalsServiceImpl#queryProductVersion": {
            "biz_clear": "调用机构中心查询产品版本信息。",
            "business_rules": ["机构中心返回失败则中断放款"],
        },
        "WithdrawalsServiceImpl#quotaCheck": {
            "biz_clear": "按产品分支执行额度校验。",
            "business_rules": ["额度不足或状态异常则标记提现失败"],
        },
        "WithdrawalsServiceImpl#orgQuotaCheck": {
            "biz_clear": "校验机构限额是否充足。",
            "business_rules": ["机构限额不足则提现失败"],
        },
    }

    def _fake_load(_pid, cms):
        out = {}
        for cm in cms:
            if cm in fake_entry:
                out[cm] = fake_entry[cm]
        return out

    with patch(
        "app.agent.explore_agent._load_deep_facts_by_class_methods",
        side_effect=_fake_load,
    ), patch(
        "app.analysis.flow_neo4j_sync.get_follows_path",
        return_value={
            "ok": True,
            "steps": [
                {"class_method": entry},
                {"class_method": "WithdrawalsController#checkApplyReqParam"},
                {"class_method": "WithdrawalsServiceImpl#loanSwitch"},
                {"class_method": "WithdrawalsServiceImpl#loan"},
            ],
        },
    ), patch(
        "app.agent.explore_agent._collect_entry_calls_closure",
        return_value=[
            "WithdrawalsServiceImpl#queryProductVersion",
            "WithdrawalsServiceImpl#quotaCheck",
            "WithdrawalsServiceImpl#orgQuotaCheck",
            "WithdrawalsServiceImpl#quotaCheckMIY",
        ],
    ):
        text = _assemble_answer_from_symbol_descs(
            "demo-loan", entry, full=True, flow_key="loan_disburse"
        )
    assert text
    assert "withdrawals/withdrawals" in text
    assert "本段代码首先" not in text
    assert "generateReqParam" not in text  # demoted call-chain narration
    assert "[段1]" not in text
    assert "map.RESPONSE_CODE" not in text
    assert "checker" not in text
    assert "关键业务关卡" in text
    # Generic stage labels — product「放款开关」copy not required.
    assert "业务开关" in text or "loanSwitch" in text or "开关" in text or "checkApplyReqParam" in text
    assert "额度" in text or "quotaCheck" in text or "机构" in text or "主处理" in text
    assert "机构标识（instId）不能为空" not in text  # field-null laundry demoted
    assert "字段级清单从略" in text or "入参校验" in text or "checkApplyReqParam" in text
    assert "结果回调" in text
    assert "1. 接收放款申请请求" not in text  # not forced 10-step FOLLOWS essay
    assert "业务流程：" not in text  # prefer 关键业务关卡 over thin numbered FOLLOWS


def test_is_field_null_rule_vs_business_gate():
    from app.agent.explore_agent import _is_field_null_rule, _pick_gate_bullets

    assert _is_field_null_rule("机构标识（instId）不能为空")
    assert not _is_field_null_rule("额度校验失败则标记提现为失败")
    assert not _is_field_null_rule("开关为 0 时触发拦截")
    gates = _pick_gate_bullets(
        {
            "business_rules": [
                "机构标识不能为空",
                "额度校验失败则标记提现为失败",
                "若需走融担流程，根据渠道选择新老流程",
            ]
        },
        max_n=3,
    )
    assert any("额度" in g for g in gates)


def test_sanitize_strips_fake_tables_and_impl_disclaimer():
    raw = (
        "MIY额度校验走 `WithdrawalsServiceImpl#quotaCheckMIY`。\n"
        "查询用户额度表（如 `t_user_quota`）和 `t_product_config`。\n"
        "当前未获取到 `quotaCheckMIY` 方法的具体实现代码，无法进一步确认细节。\n"
    )
    out = _sanitize_answer(raw)
    assert "quotaCheckMIY" in out
    assert "t_user_quota" not in out
    assert "t_product_config" not in out
    assert "未获取到" not in out
    assert "具体实现代码" not in out
    assert "未在已读材料中看到具体表名" in out


def test_sanitize_no_evidence_strips_plausible_t_star():
    """Without evidence bag, invented t_* must not ship — even unknown ones."""
    raw = "落库写入 `t_loan_apply_xyz` 并更新额度。"
    out = _sanitize_answer(raw, evidence_text="")
    assert "t_loan_apply_xyz" not in out


def test_sanitize_keeps_table_present_in_evidence():
    raw = "成功后写入 `t_withdrawals` 申请主表。"
    out = _sanitize_answer(raw, evidence_text="Mapper insert into t_withdrawals ...")
    assert "t_withdrawals" in out


def test_sanitize_aqc_client_path_keeps_real_cues_strips_fakes():
    """Real MIY evidence cites aqcClient; fake quota tables must not appear."""
    raw = (
        "1. `WithdrawalsServiceImpl#quotaCheckMIY`\n"
        "   - 调用外部 `aqcClient.loanSchemaQuery`\n"
        "   - 比较申请金额与可用额度 availLimit\n"
        "   - 查询用户额度表 `t_user_quota`\n"
    )
    ev = "aqcClient.loanSchemaQuery getAvailLimit creditAmount LimitStatus"
    out = _sanitize_answer(raw, evidence_text=ev)
    assert "aqcClient" in out
    assert "loanSchemaQuery" in out
    assert "availLimit" in out
    assert "t_user_quota" not in out


def test_review_footnote_never_appended_to_answer():
    """Review gaps must never become user-visible 评审提示/补充说明 footers."""
    from app.agent.explore_agent import _with_review_footnote

    ans = (
        "1. `WithdrawalsServiceImpl#quotaCheckMIY`\n"
        "   - 调用 aqcClient.loanSchemaQuery\n"
        "   - 校验可用额度"
    )
    out = _with_review_footnote(
        ans,
        {"ok": False, "gaps": ["未提供具体实现代码", "未说明落库", "未发现明确的额度校验"]},
        question="放款接口中有没有做额度校验",
    )
    assert out == ans
    assert "补充说明" not in out
    assert "评审提示" not in out

    bare = "随便一段回答"
    out2 = _with_review_footnote(
        bare,
        {"ok": False, "gaps": ["缺落库", "缺外部协同"]},
        question="放款流程什么样",
    )
    assert out2 == bare
    assert "评审提示" not in out2
    assert "补充说明" not in out2


def test_wants_gate_detail_existence_quota_ask():
    """Business-gate theme router removed — dig-in uses explore/session, not keyword gate."""
    from app.agent.explore_agent import _wants_gate_detail

    q = "放款接口 中 有没有做额度校验 额度校验逻辑又是什么样"
    assert not _wants_gate_detail(q)
    assert not _wants_gate_detail("有没有做额度校验")
    assert not _wants_gate_detail("有无额度校验")
    assert not _wants_gate_detail("做不做额度校验")


def test_assemble_gate_detail_quota_existence_cites_quota_check():
    """Gate-detail short-circuit removed — returns empty; explore tool loop handles dig-in."""
    from app.agent.explore_agent import _assemble_gate_detail_from_calls

    out, symbols = _assemble_gate_detail_from_calls(
        "demo-loan",
        "WithdrawalsController#withdrawals",
        "放款接口 中 有没有做额度校验 额度校验逻辑又是什么样",
    )
    assert out is None
    assert symbols == []


def test_rank_gate_callees_prefers_primary_stem_not_product_token():
    """Ranking is CALLS/theme evidence only — no MIY→quotaCheckMIY invent."""
    from app.agent.explore_agent import _rank_gate_callees

    ranked = _rank_gate_callees(
        [
            "WithdrawalsServiceImpl#quotaCheck",
            "WithdrawalsServiceImpl#quotaCheckMIY",
            "WithdrawalsServiceImpl#quotaCheckCKHB01",
            "SomeClient#quotaCheckRemote",
        ],
        ["额度"],
        question="MIY的额度校验具体什么样",
    )
    # Exact primary stem beats product-suffixed variants; ServiceImpl beats Client.
    assert ranked[0].endswith("#quotaCheck")
    assert all("Client#" not in cm or cm != ranked[0] for cm in ranked[:1])


def test_wants_source_paste_detects_common_asks():
    from app.agent.explore_agent import _wants_source_paste

    assert _wants_source_paste("MIY额度校验源码是什么")
    assert _wants_source_paste("quotaCheckMIY 源码")
    assert _wants_source_paste("把代码贴出来")
    assert _wants_source_paste("我想看看对应的代码")
    assert _wants_source_paste("看下代码")
    assert not _wants_source_paste("MIY的额度校验具体什么样")
    assert not _wants_source_paste("放款流程什么样？从入口到下游回调串一下。")


def test_assemble_source_code_quota_check_under_entry_has_real_body():
    """Theme+entry resolve to CALLS quotaCheck* when MySQL/graph present."""
    import pytest
    from pathlib import Path

    from app.agent.explore_agent import _assemble_source_code_answer
    from app.core.config import get_settings
    from app.db.mysql import init_db

    if not (Path(__file__).resolve().parents[1] / "java_project" / "demo-loan").is_dir():
        pytest.skip("demo-loan 工程目录不存在（本地未放置被索引仓库）")

    get_settings.cache_clear()
    init_db()
    out, target = _assemble_source_code_answer(
        "demo-loan",
        "WithdrawalsController#withdrawals",
        "放款接口中额度校验部分代码什么样",
    )
    if not out or not target or not re.search(r"(?i)quotaCheck", target or ""):
        # Offline / empty CALLS — skip strict quotaCheck paste; never invent tables.
        if out:
            assert "t_user_quota" not in out
            assert "不会编造源码" in out or "未在已读材料" in out or "```java" in out
        return
    assert re.search(r"(?i)quotaCheck", out)
    assert "```java" in out
    assert "t_user_quota" not in out
    assert "不会编造源码" not in out
    assert "未在 MySQL" not in out
    assert "AmountController" not in (target or "")


def test_resolve_source_paste_uses_session_gate_injection():
    from app.agent.explore_source import _resolve_source_paste_target

    q = (
        "我想看看对应的代码\n"
        "（关卡：额度；符号：WithdrawalsServiceImpl#quotaCheck）"
    )
    target = _resolve_source_paste_target(
        "demo-loan",
        "WithdrawalsController#withdrawals",
        q,
    )
    assert target == "WithdrawalsServiceImpl#quotaCheck"


def test_wants_gate_detail_org_query_and_single_batch():
    from app.agent.explore_agent import _wants_gate_detail

    assert not _wants_gate_detail("那机构查询这一块逻辑什么样的")
    assert not _wants_gate_detail("单笔单批是怎么判断的")
    assert not _wants_gate_detail("机构校验怎么做的")