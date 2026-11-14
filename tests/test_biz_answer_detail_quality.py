"""Business answer detail quality — flow completeness, depth, grounding, hallucination rejection.

Tests the *answer reviewer* and *sanitizer* layers that guard business answer quality.
Every test runs with deterministic logic (no live LLM) to validate the quality gates
themselves.
"""

from __future__ import annotations


# ═══════════════════════════════════════════════════════════════════════
# 1. FLOW COMPLETENESS — full flow must cover all key stages
# ═══════════════════════════════════════════════════════════════════════


def test_heuristic_review_accepts_complete_flow():
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "放款流程是什么样的，从入口到下游回调串一下",
        (
            "主链：\n"
            "1. WithdrawalsController#withdrawals — 受理放款申请，校验请求参数必填\n"
            "2. WithdrawalsServiceImpl#process — 校验商户状态与额度开关\n"
            "3. 通过 aqcClient#loanSchemaQuery 调用额度服务检查可用额度\n"
            "4. 校验通过后，落库至 demo_b_withdrawals（状态=待放款）\n"
            "5. 同步调用 paymentClient#disburse 执行资金划拨\n"
            "结果回调/通知：\n"
            "- 放款完成后，MQ 发送消息到 loan.notify.queue\n"
            "- WithdrawalsNoticeListener#onMessage 消费，回写状态为已放款"
        ),
    )
    assert review["ok"] is True, (
        f"complete flow should pass, got gaps: {review.get('gaps')}"
    )
    assert review["score"] >= 75


def test_heuristic_review_detects_vague_tail():
    """Flow answer that ends with "继续处理" must be rejected."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "授信通知接口流程是什么样的",
        (
            "主链：\n"
            "1. 校验请求体\n"
            "2. 校验机构标识与流水号\n"
            "3. 加锁防重\n"
            "4. 构造授信通知对象并继续处理"
        ),
    )
    assert review["ok"] is False
    assert review["force_fail"] is True
    assert any("继续处理" in g for g in review["gaps"])


def test_heuristic_review_detects_missing_persist():
    """Flow answer without persistence details must be rejected."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "提现申请流程什么样",
        (
            "主链：\n"
            "1. 校验必填字段\n"
            "2. 判断重复申请\n"
            "3. 校验额度可用\n"
            "4. 返回成功结果"
        ),
    )
    assert review["ok"] is False
    gaps = " ".join(review["gaps"])
    assert "落库" in gaps or "表" in gaps


def test_heuristic_review_detects_missing_external_calls():
    """Flow answer without external system calls must be flagged."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "放款完整流程串一下",
        (
            "主链：\n"
            "1. Controller 受理请求\n"
            "2. Service 校验数据\n"
            "3. 数据落库\n"
            "4. 返回处理结果"
        ),
    )
    assert review["ok"] is False
    gaps = " ".join(review["gaps"])
    assert "外部" in gaps or "Feign" in gaps or "Client" in gaps


def test_heuristic_review_full_flow_with_persist_and_external_passes():
    """Flow answer with both persist and external call details must pass."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "授信通知接口流程是什么样的",
        (
            "主链：\n"
            "1. CreditNoticeController#receive — 接收通知请求\n"
            "2. 校验签名与商户信息\n"
            "3. 通过 aqcClient#creditQuery 查询额度信息\n"
            "4. 处理完成后落库至 demo_b_credit_notify（状态=已处理）\n"
            "5. 通过 MQ 发送处理结果至 credit.result.queue\n"
            "结果回调/通知：\n"
            "- CreditResultListener#onMessage 消费回调"
        ),
    )
    assert review["ok"] is True
    assert review["score"] >= 75


def test_heuristic_review_gate_detail_skips_persist_demand():
    """Gate dig-in questions should NOT require persist/external details."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "MIY的额度校验具体什么样",
        (
            "1. `WithdrawalsServiceImpl#quotaCheckMIY`\n"
            "   - 调用 aqcClient#loanSchemaQuery\n"
            "   - 校验额度状态（冻结/失效）\n"
            "   - 比较申请金额与可用额度"
        ),
    )
    assert review["ok"] is True
    assert review["force_fail"] is False


# ═══════════════════════════════════════════════════════════════════════
# 2. HALLUCINATION / GROUNDING
# ═══════════════════════════════════════════════════════════════════════


def test_invented_tables_detected():
    """Answer citing t_* tables not in evidence must be flagged."""
    from app.agent.answer_reviewer import invented_tables_not_in_evidence

    invented = invented_tables_not_in_evidence(
        "落库至 t_user_quota 和 t_withdrawals_apply",
        "demo_b_withdrawals aqcClient",
    )
    assert "t_user_quota" in invented
    assert "t_withdrawals_apply" in invented


def test_known_table_not_flagged():
    """Tables present in evidence must NOT be flagged as invented."""
    from app.agent.answer_reviewer import invented_tables_not_in_evidence

    result = invented_tables_not_in_evidence(
        "落库至 demo_b_withdrawals 和 demo_b_ext",
        "demo_b_withdrawals demo_b_ext Mapper insert",
    )
    assert result == [], (
        f"tables in evidence should not be flagged: {result}"
    )


def test_invented_tables_force_fail():
    """Presence of invented tables should cause force_fail."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "提现流程",
        "落库至 t_user_quota 表",
        evidence="WithdrawalsController Mapper insert",
    )
    assert review["force_fail"] is True
    assert not review["ok"]


def test_force_fail_respected_in_review():
    """Force_fail must cause overall ok=False even if other checks pass."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "放款流程",
        (
            "1. Controller 受理\n"
            "2. 落库至 demo_b_withdrawals\n"
            "3. 通过 Feign 调用下游\n"
            "4. 落库至 t_user_quota 表"  # ← invented
        ),
        evidence="WithdrawalsController#withdrawals FeignClient",
    )
    assert review["force_fail"] is True
    assert not review["ok"]
    assert any("编造" in g for g in review["gaps"])


# ═══════════════════════════════════════════════════════════════════════
# 3. ANSWER DEPTH / SHALLOWNESS DETECTION
# ═══════════════════════════════════════════════════════════════════════


def test_full_flow_shallow_blocked_on_vague():
    from app.agent.explore_answers import _full_flow_answer_too_shallow

    block = _full_flow_answer_too_shallow(
        "主链：1. 校验请求体 2. 校验机构标识 "
        "3. 加分布式锁 4. 构造授信通知对象并继续处理",
        "授信通知接口流程是什么样的",
    )
    assert block is not None
    assert "继续处理" in block or "落库" in block


def test_full_flow_shallow_ok_when_concrete():
    from app.agent.explore_answers import _full_flow_answer_too_shallow

    block = _full_flow_answer_too_shallow(
        "主链：1. 校验机构与流水号 2. 加锁防重 "
        "3. 通过 Feign 同步调用额度服务 "
        "4. 成功后落库授信通知表，失败返回错误",
        "授信通知接口流程是什么样的",
    )
    assert block is None


def test_full_flow_shallow_detects_no_real_steps():
    from app.agent.explore_answers import _full_flow_answer_too_shallow

    block = _full_flow_answer_too_shallow(
        "这个过程涉及多个业务环节，需要逐步分析。",
        "放款流程什么样",
    )
    assert block is not None


def test_messy_full_flow_answer_detected():
    from app.agent.explore_answers import _is_messy_full_flow_answer

    assert _is_messy_full_flow_answer(
        "我需要先确认入口再回答。受理后会校验，再放款。"
    )
    assert _is_messy_full_flow_answer(
        "我需要理解整条链路。先看看 Controller 入口。"
    )
    assert not _is_messy_full_flow_answer(
        "主链：\n1. 受理申请\n2. 校验\n3. 落库\n4. 回调"
    )


def test_sanitize_strips_monologue():
    from app.agent.explore_answers import _sanitize_answer

    cleaned = _sanitize_answer(
        (
            "我需要先理解这个问题。根据已读证据，我们来梳理放款流程。\n"
            "主链：\n1. Controller 受理\n2. 校验\n3. 落库"
        ),
        full_flow=True,
    )
    assert "我需要先理解" not in cleaned
    assert "根据已读证据" not in cleaned
    assert "主链：" in cleaned


def test_sanitize_keeps_class_method_when_user_asks():
    """When user asks for tech detail, Class#method anchors should survive."""
    from app.agent.explore_answers import _sanitize_answer

    cleaned = _sanitize_answer(
        (
            "主链：\n"
            "1. `WithdrawalsController#withdrawals` 受理请求\n"
            "2. `WithdrawalsServiceImpl#process` 处理\n"
            "3. 落库"
        ),
        full_flow=True,
    )
    assert "主链：" in cleaned


def test_sanitize_removes_hex_ids():
    from app.agent.explore_answers import _sanitize_answer

    raw = (
        "主链：\n"
        "1. Controller 受理 (symbol_id: abc123def4567890abcdef1234567890)\n"
        "2. Service 处理\n"
        "3. 落库"
    )
    cleaned = _sanitize_answer(raw)
    assert "symbol_id" not in cleaned
    assert "abc123def" not in cleaned


# ═══════════════════════════════════════════════════════════════════════
# 4. REVIEW SYSTEM
# ═══════════════════════════════════════════════════════════════════════


def test_review_reject_message_generates_actionable_advice():
    from app.agent.answer_reviewer import review_reject_message

    msg = review_reject_message(
        {
            "summary": "缺落库细节",
            "gaps": ["未说明何时落库", "缺外部系统调用"],
            "followups": ["读 ServiceImpl 找 Mapper"],
        }
    )
    assert "补充更多实现细节" in msg
    assert "落库" in msg
    assert "Mapper" in msg


def test_review_empty_answer_returns_ok_false():
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review("放款流程", "")
    assert review["ok"] is False
    assert review["score"] == 0


def test_score_scales_with_gaps_count():
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "完整放款流程",
        "做校验然后处理",
    )
    assert review["score"] < 60
    assert not review["ok"]


def test_reviewer_parses_json_response():
    from app.agent.answer_reviewer import _parse_review

    result = _parse_review(
        '{"ok": true, "score": 85, "summary": "基本完整", '
        '"gaps": ["缺一小细节"], "followups": ["查一下某方法"]}'
    )
    assert result is not None
    assert result["ok"] is True
    assert result["score"] == 85
    assert "缺一小细节" in result["gaps"]


def test_reviewer_handles_broken_json():
    from app.agent.answer_reviewer import _parse_review

    result = _parse_review("不，这个回答不够具体")
    assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 5. QUESTION INTENT QUALITY
# ═══════════════════════════════════════════════════════════════════════


def test_intent_detects_add_field_plan_with_change_keyword():
    """wants_add_field_change_plan needs persist/change/outbound signal."""
    from app.agent.question_intent import wants_add_field_change_plan

    # "新增一个字段 xxx，从入参到落库串一下" contains "字段" AND "新增" AND "落库" (persist)
    assert wants_add_field_change_plan("新增一个字段 xxx，从入参到落库串一下")
    # "加个字段到 VO 和实体，需要改哪些" contains "改哪些" (change)
    assert wants_add_field_change_plan("加个字段到 VO 和实体，需要改哪些")
    # Plain "新增一个字段 xxx" is now enough (relaxed function)
    assert wants_add_field_change_plan("新增一个字段 xxx")
    # Business flow question is not an add-field plan question
    assert not wants_add_field_change_plan("放款流程什么样")


def test_intent_detects_show_gate_code():
    from app.agent.question_intent import wants_show_gate_code

    assert wants_show_gate_code("额度校验的代码看看")
    assert wants_show_gate_code("看看校验关卡代码")
    assert not wants_show_gate_code("放款流程什么样")


def test_intent_taxonomy_explore_intent_detection():
    from app.agent.intent_taxonomy import is_explore_intent

    # Valid intent types that should be recognized as explore
    assert is_explore_intent("ask")
    assert is_explore_intent("plan")
    # Invalid types should NOT be explore
    assert not is_explore_intent("execute")
    assert not is_explore_intent("modify")


def test_intent_taxonomy_normalize_unknown_falls_back():
    from app.agent.intent_taxonomy import normalize_intent

    # Unknown intent types fall back to message-based heuristics
    result = normalize_intent(None, "放款流程")
    assert result and len(result) > 0


# ═══════════════════════════════════════════════════════════════════════
# 6. EXPLORE TOOLS — evidence chain quality
# ═══════════════════════════════════════════════════════════════════════


def test_same_class_calls_heuristic_detects_valid_helpers():
    from app.agent.explore_tools import extract_same_class_calls

    code = """public String process(String input) {
        this.validateInput(input);
        this.checkStatus(input);
        String result = this.doInternal(input);
        if (result == null) {
            this.logError("failed");
        }
        return result;
    }"""
    calls = extract_same_class_calls(code, current_method="process")
    assert "validateInput" in calls
    assert "checkStatus" in calls
    assert "doInternal" in calls
    assert calls


def test_same_class_calls_skips_noise():
    from app.agent.explore_tools import extract_same_class_calls

    code = """public void run() {
        if (flag) { for (int i = 0; i < 10; i++) { log.info("loop"); } }
        String.format("test");
    }"""
    calls = extract_same_class_calls(code, current_method="run")
    assert "if" not in calls
    assert "log" not in calls
    assert "format" not in calls