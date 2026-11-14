"""Heuristic completeness checks for the answer reviewer."""
from app.agent.answer_reviewer import (
    _heuristic_review,
    _soften_source_code_gaps,
    invented_tables_not_in_evidence,
    review_reject_message,
)
from app.agent.explore_agent import _full_flow_answer_too_shallow


def test_heuristic_rejects_vague_continue_tail():
    review = _heuristic_review(
        "授信通知接口流程是什么样的",
        "1. 校验请求体\n2. 校验机构标识与流水号\n3. 加锁防重\n4. 构造授信通知对象并继续处理",
    )
    assert review["ok"] is False
    assert review["force_fail"] is True
    assert any("继续处理" in g or "下游" in g for g in review["gaps"])


def test_heuristic_rejects_missing_persist_and_external():
    review = _heuristic_review(
        "授信通知流程怎么走",
        "主链：1. 校验必填字段 2. 判断重复授信 3. 返回成功",
    )
    assert review["ok"] is False
    joined = " ".join(review["gaps"])
    assert "落库" in joined
    assert "外部" in joined


def test_heuristic_ok_when_persist_and_external_present():
    review = _heuristic_review(
        "授信通知流程是什么样的",
        (
            "主链：1. 校验机构与流水号 2. Redisson 加锁防重 "
            "3. 通过 Feign Client 同步调用额度服务 "
            "4. 成功后落库到授信通知表，失败回滚并返回错误"
        ),
    )
    assert review["ok"] is True
    assert review["score"] >= 70


def test_heuristic_gate_detail_ok_without_persist():
    """MIY/额度 dig-in should not demand 落库 when quotaCheckMIY rules are present."""
    review = _heuristic_review(
        "MIY的额度校验具体什么样",
        (
            "1. `WithdrawalsServiceImpl#quotaCheckMIY`\n"
            "   - 调用外部 aqcClient.loanSchemaQuery\n"
            "   - 校验额度状态冻结/失效\n"
            "   - 比较申请金额与可用额度"
        ),
    )
    assert review["ok"] is True
    assert review["force_fail"] is False
    assert not any("落库" in g for g in review["gaps"])


def test_heuristic_gate_fails_on_invented_tables():
    """Gate dig-in that invents t_user_quota must force_fail even with method name."""
    review = _heuristic_review(
        "MIY的额度校验具体什么样",
        (
            "1. `WithdrawalsServiceImpl#quotaCheckMIY`\n"
            "   - 查询 `t_user_quota` 获取可用额度\n"
            "   - 校验额度状态"
        ),
        evidence="aqcClient.loanSchemaQuery availLimit",
    )
    assert review["ok"] is False
    assert review["force_fail"] is True
    assert any("编造" in g or "表名" in g for g in review["gaps"])


def test_invented_tables_helper_respects_evidence():
    ans = "写入 t_withdrawals 与 t_user_quota"
    assert "t_user_quota" in invented_tables_not_in_evidence(
        ans, "insert t_withdrawals"
    )
    assert invented_tables_not_in_evidence(
        ans, "t_withdrawals t_user_quota Mapper"
    ) == []


def test_soften_keeps_grounding_gap_drops_source_demand():
    gaps = _soften_source_code_gaps(
        [
            "未提供具体实现代码",
            "编造未在证据中出现的表名（t_user_quota）",
            "未说明落库",
        ],
        "MIY的额度校验具体什么样",
        "quotaCheckMIY 调用 aqcClient.loanSchemaQuery 校验可用额度",
        "aqcClient loanSchemaQuery",
    )
    assert not any("实现代码" in g for g in gaps)
    assert any("编造" in g for g in gaps)
    assert not any("落库" in g for g in gaps)


def test_shallow_blocks_continue_tail_on_notice_flow():
    block = _full_flow_answer_too_shallow(
        "主链：1. 校验请求体 2. 校验机构标识 3. 加分布式锁 4. 构造授信通知对象并继续处理",
        "授信通知接口流程是什么样的",
    )
    assert block is not None
    assert "继续处理" in block or "落库" in block


def test_shallow_ok_when_persist_and_feign_present():
    block = _full_flow_answer_too_shallow(
        "主链：1. 校验机构与流水号 2. 加锁防重 3. 通过 Feign 同步调用额度服务 "
        "4. 成功后落库授信通知表，失败返回错误",
        "授信通知接口流程是什么样的",
    )
    assert block is None


def test_review_reject_message_mentions_followups():
    msg = review_reject_message(
        {
            "summary": "细节不足",
            "gaps": ["未说明落库"],
            "followups": ["读 ServiceImpl 找 Mapper"],
        }
    )
    assert "补充更多实现细节" in msg
    assert "落库" in msg
    assert "Mapper" in msg


def test_review_reject_mentions_no_fake_tables():
    msg = review_reject_message(
        {"summary": "细节不足", "gaps": ["未说明落库"], "followups": []}
    )
    assert "t_user_quota" in msg or "编造" in msg
