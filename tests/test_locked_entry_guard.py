from app.agent.explore_agent import (
    _answer_violates_locked_entry,
    _prefetch_downstream_from_code,
)


def test_reject_wrong_controller_entry():
    msg = _answer_violates_locked_entry(
        "业务入口：CreditQuotaController#creditNotification — 授信查询",
        "CreditController#creditQuery",
    )
    assert msg is not None
    assert "CreditController#creditQuery" in msg


def test_reject_credit_query_narrated_as_notification():
    msg = _answer_violates_locked_entry(
        "业务入口：CreditController 提供授信查询。主链：1.接收授信通知 2.校验重复授信 3.Redisson加锁",
        "CreditController#creditQuery",
    )
    assert msg is not None
    assert "creditQuery" in msg


def test_fallback_answer_uses_prefetched_service():
    from app.agent.explore_agent import _fallback_answer_from_prefetch

    retrieved = [
        {
            "summary": "预读锁定入口 CreditController#creditQuery",
            "fqcn": "com.demo.loan.controller.CreditController",
            "method_name": "creditQuery",
            "raw_code": "result = creditService.creditQuery(reqMap);",
        },
        {
            "summary": "预读下游 creditService.creditQuery",
            "fqcn": "com.demo.loan.service.impl.CreditServiceImpl",
            "method_name": "creditQuery",
            "raw_code": 'String outLoanId= (String) req.get("outLoanId"); QueryWrapper q;',
        },
    ]
    ans = _fallback_answer_from_prefetch(
        "credit/creditQuery",
        "CreditController#creditQuery",
        retrieved,
    )
    assert ans is not None
    assert "CreditController#creditQuery" in ans
    assert "CreditServiceImpl#creditQuery" in ans
    assert "creditNotification" not in ans

    code = """
    public Response foo() {
        if (!generateReqParam(reqMap, result, creditNotification)) {
            return response;
        }
        result = creditService.creditNotification(creditNotification);
        return response;
    }
    """
    hits = _prefetch_downstream_from_code("demo-loan", code, limit=2)
    assert hits, "should resolve creditService.creditNotification"
    joined = " ".join(f"{h.get('fqcn')}#{h.get('method_name')}" for h in hits)
    assert "creditNotification" in joined
