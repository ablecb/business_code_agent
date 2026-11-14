"""Caps / early-finish / batch parsing for explore_agent round-trip reduction."""
from __future__ import annotations

from app.agent.explore_agent import (
    _adaptive_max_steps,
    _core_chain_ready,
    _extract_tool_calls,
    _finish_block_reason,
    _is_messy_full_flow_answer,
    _sanitize_answer,
)


def test_adaptive_max_steps_clamps_full_flow_and_endpoint():
    q_full = "放款流程什么样？从入口到下游回调串一下"
    assert _adaptive_max_steps(q_full) == 14
    assert _adaptive_max_steps(q_full, requested=28) == 14
    assert _adaptive_max_steps(q_full, requested=8) == 8

    q_ep = "核算服务地址发下"
    assert _adaptive_max_steps(q_ep) == 8
    assert _adaptive_max_steps(q_ep, requested=20) == 8

    q_flow = "订单流程什么样"
    assert _adaptive_max_steps(q_flow) == 12
    assert _adaptive_max_steps(q_flow, requested=40) == 12


def test_extract_tool_calls_batch_and_finish_alone():
    single = _extract_tool_calls(
        '思考一下\n{"tool":"search_symbols","args":{"q":"放款","kind":"Controller"}}'
    )
    assert len(single) == 1
    assert single[0]["tool"] == "search_symbols"

    batch = _extract_tool_calls(
        '{"tools":['
        '{"tool":"get_symbol_code","args":{"symbol_id":"a"}},'
        '{"tool":"list_callees","args":{"symbol_id":"a"}},'
        '{"tool":"resolve_impl","args":{"fqcn":"FooService"}}'
        "]}"
    )
    assert len(batch) == 3
    assert {c["tool"] for c in batch} == {
        "get_symbol_code",
        "list_callees",
        "resolve_impl",
    }

    # finish drops siblings
    mixed = _extract_tool_calls(
        '{"tools":[{"tool":"get_symbol_code","args":{"symbol_id":"a"}},'
        '{"tool":"finish","args":{"answer":"主链：\\n1. 受理"}}]}'
    )
    assert mixed == [{"tool": "finish", "args": {"answer": "主链：\n1. 受理"}}]

    # cap at 3
    many = _extract_tool_calls(
        '{"tools":['
        + ",".join(
            f'{{"tool":"get_symbol_code","args":{{"symbol_id":"{i}"}}}}' for i in range(5)
        )
        + "]}"
    )
    assert len(many) == 3


def test_core_chain_ready_full_flow_needs_listener_when_mq():
    steps = [
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "resolve_impl", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    q = "放款完整流程从入口到下游回调串一下"
    impl_only = [
        {
            "fqcn": "com.x.LoanController",
            "kind": "Controller",
            "raw_code": "service.apply();",
        },
        {
            "fqcn": "com.x.LoanServiceImpl",
            "raw_code": 'rabbitTemplate.convertAndSend("q", msg);',
        },
    ]
    assert not _core_chain_ready(steps, impl_only, question=q)

    with_listener = impl_only + [
        {
            "fqcn": "com.x.LoanRabbitMqListenner",
            "raw_code": "@RabbitListener void onMessage(){ accClient.deal(); }",
        }
    ]
    assert _core_chain_ready(steps, with_listener, question=q)


def test_finish_soft_skips_optional_client_when_listener_ready():
    steps = [
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "list_callees", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    retrieved = [
        {
            "fqcn": "com.x.LoanController",
            "kind": "Controller",
            "raw_code": "loanService.apply();",
        },
        {
            "fqcn": "com.x.LoanServiceImpl",
            "raw_code": 'rabbitTemplate.convertAndSend("loan.q", body); paymentClient.x();',
        },
        {
            "fqcn": "com.x.LoanRabbitMqListenner",
            "raw_code": "@RabbitListener void onMessage() {}",
        },
    ]
    q = "放款流程从入口到下游回调串一下"
    # Hard gate may still ask for Client; soft accepts core chain.
    hard = _finish_block_reason(steps, retrieved, question=q, soft=False)
    soft = _finish_block_reason(steps, retrieved, question=q, soft=True)
    # With listener present, Client gate is already softened in hard path too —
    # either way soft must allow finish.
    assert soft is None
    if hard is not None:
        assert "Client" in hard or "Feign" in hard


def test_messy_answer_sanitize_then_accept_pattern():
    """Style reject should be collapsible via sanitize (used after 1 retry)."""
    messy = (
        "我需要理解整条链路。根据证据入口是 FooController#a，"
        "然后 BarServiceImpl#b，再到 BazListener#c。"
    )
    assert _is_messy_full_flow_answer(messy)
    cleaned = _sanitize_answer(
        "主链：\n1. 受理申请\n2. 校验与风控\n3. 投递下游\n结果回调/通知：\n- 回写状态\n"
        + messy,
        full_flow=True,
    )
    assert "我需要理解" not in cleaned
    assert "1. 受理申请" in cleaned


def test_full_flow_min_reads_is_three():
    steps = [
        {"tool": "get_symbol_code", "ok": True},
        {"tool": "resolve_impl", "ok": True},
        {"tool": "get_symbol_code", "ok": True},
    ]
    # Avoid「完整」so callback-section gate does not require Listener/Notice.
    q = "业务流程走一遍"
    # only 2 successful reads → still blocked
    assert _finish_block_reason(
        steps[:2] + [{"tool": "resolve_impl", "ok": True}],
        [{"fqcn": "com.x.FooServiceImpl", "raw_code": "x();"}],
        question=q,
    )
    # 3 reads + impl + client → ok
    steps3 = steps + [{"tool": "get_symbol_code", "ok": True}]
    assert (
        _finish_block_reason(
            steps3,
            [
                {"fqcn": "com.x.FooController", "kind": "Controller", "raw_code": "svc.a();"},
                {"fqcn": "com.x.FooServiceImpl", "raw_code": "remoteClient.call();"},
                {"fqcn": "com.x.RemoteClient", "raw_code": "@FeignClient void call();"},
            ],
            question=q,
        )
        is None
    )
