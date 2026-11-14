"""Turn-level mixed intent: ask then git/ops."""
from app.agent.runtime.turn_plan import (
    compose_ordered_qa_reply,
    heuristic_turn_plan,
    is_mixed_ask_then_ops,
    split_turn_stages,
)


def test_mixed_look_then_commit_push():
    q = "先帮我看看放款接口 机构限额校验 做了什么，然后看完后帮我提交并推送代码"
    assert is_mixed_ask_then_ops(q)
    plan = heuristic_turn_plan(q)
    assert plan.has_ask
    assert plan.follow_on_client_mode == "direct"
    assert "提交" in plan.follow_on_goal or "推送" in plan.follow_on_goal
    assert "放款" in plan.ask_text
    assert "机构限额" in plan.ask_text
    assert "推送" not in plan.ask_text
    layers = [n.layer for n in plan.nodes]
    assert layers[0] == "ask"
    assert "client" in layers
    git = next(n for n in plan.nodes if n.layer == "client")
    assert git.needs_confirm
    assert git.depends_on
    assert any("ask" in d for d in git.depends_on)


def test_ask_only_not_mixed():
    q = "放款接口里机构限额校验做了什么"
    assert not is_mixed_ask_then_ops(q)
    plan = heuristic_turn_plan(q)
    assert plan.has_ask
    assert not plan.follow_on_goal


def test_git_only_not_mixed_ask():
    q = "帮我提交并推送代码"
    assert not is_mixed_ask_then_ops(q)
    plan = heuristic_turn_plan(q)
    assert not plan.has_ask


def test_stage_split():
    parts = split_turn_stages(
        "先看放款流程，然后看完后帮我提交并推送代码"
    )
    assert len(parts) >= 2


def test_compose_code_story_not_split():
    """「给我一段代码 + 先查A再查B」是单任务多步查询编排，绝不拆成两个查代码。

    回归 desktop-chat-b58bb8b77eea：拆成两个独立 PEV 导致 249s 耗时 + 占位符代码。
    """
    from app.agent.runtime.turn_plan import _is_compose_code_story

    q = "给我一段 通过order_id查询放款表，查到结果后，再查询放款优惠券表和放款审批表的代码"
    assert _is_compose_code_story(q)
    assert split_turn_stages(q) == [q]


def test_compose_code_story_variants():
    from app.agent.runtime.turn_plan import _is_compose_code_story, split_turn_stages

    assert _is_compose_code_story("帮我写一段代码 先查订单表 再根据结果查明细表")
    assert _is_compose_code_story("生成一个方法 根据订单号查主表 然后查子表")
    # 单步代码请求无步骤连接词 → story 标记 False，但也不会被误拆（split=1）
    assert not _is_compose_code_story("写一段代码 查用户表")
    assert split_turn_stages("写一段代码 查用户表") == ["写一段代码 查用户表"]


def test_mixed_ask_then_code_still_splits():
    """非代码故事的混合意图（问流程 + 再看代码）仍正常拆分。"""
    from app.agent.runtime.turn_plan import _is_compose_code_story

    q = "看看放款接口的流程 然后帮我看下代码"
    assert not _is_compose_code_story(q)
    assert len(split_turn_stages(q)) >= 2


def test_three_asks_stay_ask_layer():
    q = "先看放款校验做什么，然后看授信通知有没有，接着看额度字段从哪来"
    assert not is_mixed_ask_then_ops(q)
    plan = heuristic_turn_plan(q)
    assert plan.intent_layers == ["ask", "ask", "ask"]
    assert not plan.has_post_qa
    assert plan.needs_sequential_qa
    for i, n in enumerate(plan.nodes[1:], 1):
        assert n.depends_on == [plan.nodes[i - 1].id]


def test_two_api_ask_still_ask_only():
    q = "放款接口中机构限额校验是什么 以及 授信通知接口有没有这个校验"
    assert not is_mixed_ask_then_ops(q)
    plan = heuristic_turn_plan(q)
    assert plan.has_ask
    assert not plan.follow_on_goal
    assert plan.intent_layers == ["ask"]


def test_three_plus_intents_ordered_dag():
    q = (
        "先看放款接口机构限额校验做了什么，"
        "然后看授信通知接口有没有这个校验，"
        "再给我改动清单，"
        "最后帮我提交并推送代码"
    )
    plan = heuristic_turn_plan(q)
    assert plan.intent_layers == ["ask", "ask", "plan", "client"]
    assert plan.has_qa and plan.has_post_qa
    assert "放款" in plan.qa_text and "授信通知" in plan.qa_text
    assert "改动清单" in plan.qa_text
    assert "推送" not in plan.qa_text
    assert plan.follow_on_client_mode == "direct"
    git = next(n for n in plan.nodes if n.layer == "client")
    assert git.needs_confirm
    assert git.depends_on
    for i, n in enumerate(plan.nodes):
        if i == 0:
            assert n.depends_on == []
        else:
            assert n.depends_on == [plan.nodes[i - 1].id]
    assert plan.needs_sequential_qa


def test_ask_then_implement_then_git_is_code_mode():
    q = "先看下放款校验做什么，然后帮我改代码，再提交并推送"
    plan = heuristic_turn_plan(q)
    assert plan.intent_layers == ["ask", "modify", "client"]
    assert plan.follow_on_client_mode == "code"


def _pin_mock(monkeypatch) -> None:
    """Pin LLM/embedding to the deterministic mock and clear provider caches.

    Required so LLM-driven turn splitting is hermetic in the full suite
    (a previous test may have left a real provider cached / env leaked).
    """
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()


def test_llm_turn_plan_splits_compound_long_sentence(monkeypatch):
    """Long utterance with 顺便/还有 (no standard connectives) → LLM splits by intent.

    启发式只分出 1 段（且误判成 plan），LLM 按语义拆成 ask/ask/plan 三段顺序执行。
    """
    import asyncio

    _pin_mock(monkeypatch)

    from app.agent.runtime.turn_plan import llm_turn_plan, split_turn_stages

    q = (
        "帮我看看放款接口机构限额校验做了什么，"
        "顺便查一下授信通知有没有这个校验，"
        "还有把改动清单给我"
    )
    # 前置：启发式确实拆不出（这正是 LLM 介入的触发条件）
    assert len(split_turn_stages(q)) == 1

    plan = asyncio.run(llm_turn_plan(q))
    assert plan.source == "llm"
    assert [n.layer for n in plan.nodes] == ["ask", "ask", "plan"]
    assert plan.needs_sequential_qa  # 两个 QA 段按原话顺序执行
    assert "放款" in plan.nodes[0].question
    assert "授信通知" in plan.nodes[1].question
    assert "改动清单" in plan.nodes[2].question
    assert plan.nodes[1].depends_on == [plan.nodes[0].id]
    assert plan.nodes[2].depends_on == [plan.nodes[1].id]


def test_llm_turn_plan_compact_no_punctuation_splits(monkeypatch):
    """无标点/无空格紧凑句含多意图 → 也触发 LLM 拆分。

    实测缺口：紧凑句字符数少（如 16 字），原长度护栏(≥18)会跳过 LLM，
    导致「顺便」等无标准连接词的多意图被压成单段。修复：短句只要
    意图信号簇 ≥2 也触发。mock 拆分需连接词（再看看/顺便），verb 拼接
    （帮我改下代码）由真实 LLM 语义拆分，mock 保持整句。
    """
    import asyncio

    _pin_mock(monkeypatch)

    from app.agent.runtime.turn_plan import llm_turn_plan, _intent_signal_count

    # 16 字、无标点、含「顺便」→ sig=2 触发 LLM，拆成 ask/plan
    compact = "看看放款校验做了什么顺便分析改动点"
    assert _intent_signal_count(compact) >= 2
    plan = asyncio.run(llm_turn_plan(compact))
    assert plan.source == "llm"
    assert [n.layer for n in plan.nodes] == ["ask", "plan"]

    # 无标点「再看看」连接词 → 同样拆分
    p2 = asyncio.run(llm_turn_plan("看看放款流程再看看改动点"))
    assert p2.source == "llm"
    assert [n.layer for n in p2.nodes] == ["ask", "plan"]

    # 单意图短句 → 仍走启发式（费用护栏）
    single = asyncio.run(llm_turn_plan("放款流程什么样"))
    assert single.source in ("heuristic", "heuristic_fallback")


def test_llm_turn_plan_short_or_simple_stays_heuristic(monkeypatch):
    """费用护栏：短句/单意图/启发式已拆出多段时，不调用 LLM。"""
    import asyncio

    _pin_mock(monkeypatch)

    from app.agent.runtime.turn_plan import llm_turn_plan

    # 短单意图：不触发 LLM（启发式返回）
    short = asyncio.run(llm_turn_plan("放款流程什么样"))
    assert short.source in ("heuristic", "heuristic_fallback")
    assert [n.layer for n in short.nodes] == ["ask"]

    # 启发式已拆出多段（先…然后…）：直接用启发式结果，不再调用 LLM
    mixed = asyncio.run(
        llm_turn_plan("先看放款校验做什么，然后帮我提交并推送代码")
    )
    assert [n.layer for n in mixed.nodes] == ["ask", "client"]
    assert "提交" in mixed.nodes[1].question


def test_llm_turn_plan_falls_back_on_bad_json(monkeypatch):
    """LLM 返回非法 JSON → 回退启发式，管线不回归。"""
    import asyncio
    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()

    from app.agent.runtime.turn_plan import llm_turn_plan, heuristic_turn_plan

    async def _bad_chat(messages, **kwargs):
        return "这不是 JSON"

    monkeypatch.setattr(
        "app.llm.factory.get_llm_provider",
        lambda: type("Bad", (), {"chat": _bad_chat})(),
    )

    q = (
        "帮我看看放款接口做了什么，"
        "顺便查一下授信通知有没有这个校验，"
        "还有把改动清单给我"
    )
    plan = asyncio.run(llm_turn_plan(q))
    base = heuristic_turn_plan(q)
    assert plan.source == "heuristic_fallback"
    assert [n.layer for n in plan.nodes] == [n.layer for n in base.nodes]


def test_llm_turn_plan_keeps_story_whole(monkeypatch):
    """单任务编排句（改字段+落库 / 加接口+步骤）LLM 前先被故事守卫整块保留。"""
    import asyncio

    _pin_mock(monkeypatch)

    from app.agent.runtime.turn_plan import llm_turn_plan

    story = (
        "在CouponInfo加个字段保存到业务表落库，"
        "顺便把改动清单给我"
    )
    plan = asyncio.run(llm_turn_plan(story))
    assert len(plan.nodes) == 1  # 故事不拆分，即便含「顺便」

    code_story = (
        "给我一段 通过order_id查询放款表，查到结果后，"
        "再查询放款优惠券表和放款审批表的代码"
    )
    p2 = asyncio.run(llm_turn_plan(code_story))
    assert len(p2.nodes) == 1


def test_mixed_chat_runs_qa_then_queues_git(monkeypatch):
    import asyncio

    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    async def _fake_direct(*_a, **_k):
        return {
            "mode": "direct",
            "goal": "should-not-short-circuit",
            "reply_preview": "git-only-wrong",
        }

    async def _skip_pev(*_a, **_k):
        return {
            "answer": "机构限额校验在放款主链上由 quotaCheck 执行。",
            "steps": [],
            "plan": {"kind": "pev"},
        }

    monkeypatch.setattr("app.agent.remote_loop.route_client_need", _fake_direct)
    monkeypatch.setattr("app.agent.pev_agent.run_pev_pipeline", _skip_pev)

    from app.agent.graph_agent import run_chat

    q = "先帮我看看放款接口 机构限额校验 做了什么，然后看完后帮我提交并推送代码"
    resp = asyncio.run(run_chat(q, client_agent=True))
    assert "git-only-wrong" not in (resp.reply or "")
    assert "quotaCheck" in (resp.reply or "")
    assert "问答已完成" in (resp.reply or "")
    assert resp.client_run_id


def test_multi_intent_runs_qa_in_semantic_order(monkeypatch):
    import asyncio

    from app.agent import session_store
    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    session_store._USE_MEMORY = True
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    calls: list[tuple[str, str]] = []

    async def _fake_direct(*_a, **_k):
        return {"mode": "direct", "goal": "no", "reply_preview": "git-only-wrong"}

    async def _skip_pev(q, **k):
        hist = str(k.get("history_text") or "")
        calls.append((q, hist))
        return {
            "answer": f"答：{q[:24]}",
            "steps": [],
            "plan": {"kind": "pev"},
        }

    monkeypatch.setattr("app.agent.remote_loop.route_client_need", _fake_direct)
    monkeypatch.setattr("app.agent.pev_agent.run_pev_pipeline", _skip_pev)

    from app.agent.graph_agent import run_chat

    q = (
        "先看放款接口机构限额校验做了什么，"
        "然后看授信通知接口有没有这个校验，"
        "再给我改动清单，"
        "最后帮我提交并推送代码"
    )
    resp = asyncio.run(run_chat(q, client_agent=True))
    assert len(calls) == 3
    assert "放款" in calls[0][0] and "授信" not in calls[0][0]
    assert "授信通知" in calls[1][0]
    assert "改动清单" in calls[2][0]
    assert "答：" in calls[1][1] and "放款" in calls[1][1]
    assert "按你的提问顺序作答" in (resp.reply or "")
    assert resp.client_run_id


def test_compose_ordered_qa_reply_keeps_stage_order():
    plan = heuristic_turn_plan(
        "先看放款校验做什么，然后看授信通知有没有，再给我改动清单"
    )
    text = compose_ordered_qa_reply(
        [
            (plan.qa_nodes[0], "A1"),
            (plan.qa_nodes[1], "A2"),
            (plan.qa_nodes[2], "A3"),
        ]
    )
    assert text.index("A1") < text.index("A2") < text.index("A3")
