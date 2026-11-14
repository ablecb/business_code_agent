from __future__ import annotations

from app.agent.explore_agent import _finish_block_reason, _wants_endpoint_lookup, _wants_tech_detail


def test_endpoint_question_detected():
    assert _wants_endpoint_lookup("发下 调用的交易中心地址")
    assert _wants_tech_detail("调用交易中心服务 哪个服务了 服务地址发下")


def test_finish_requires_feign_source():
    assert _finish_block_reason([], [], question="服务地址发下")
    code = '@FeignClient(name="pre-service", path="/credit")\n@PostMapping("/granting")'
    assert (
        _finish_block_reason(
            [{"tool": "get_symbol_code", "ok": True}],
            [{"fqcn": "com.x.CreditServiceClient", "raw_code": code}],
            question="服务地址发下",
        )
        is None
    )
