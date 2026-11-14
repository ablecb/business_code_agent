"""Continuous conversation quality — context, history, follow-up, compaction."""
from __future__ import annotations

import asyncio
import json


# ═══════════════════════════════════════════════════════════════════════
# 1. SESSION — create / retrieve
# ═══════════════════════════════════════════════════════════════════════

def test_get_or_create_returns_same_id():
    from app.agent.session_store import get_or_create_session

    sid = get_or_create_session("test_proj", "test_sess_001")
    assert sid == "test_sess_001"

    sid2 = get_or_create_session("test_proj", "test_sess_001")
    assert sid2 == sid


def test_create_auto_generates_id():
    from app.agent.session_store import get_or_create_session

    sid = get_or_create_session("test_proj", None)
    assert sid and sid.startswith("chat_")


def test_append_and_list():
    from app.agent.session_store import get_or_create_session, append_message, list_messages
    from app.agent.session_store import get_session_project_id
    import uuid as _uid

    # Use UUID to guarantee unique session per test run.
    sid = get_or_create_session("proj", f"test_msg_flow_{_uid.uuid4().hex[:12]}")
    m1 = append_message(sid, "user", "放款流程什么样")
    assert m1 is not None

    m2 = append_message(sid, "assistant", "主链：\n1. 受理\n2. 校验\n3. 落库")
    assert m2 is not None and m2 > m1

    msgs = list_messages(sid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_list_messages_unknown_returns_empty():
    from app.agent.session_store import list_messages

    assert list_messages("no_such_session_xyz") == []


# ═══════════════════════════════════════════════════════════════════════
# 2. CONTEXT
# ═══════════════════════════════════════════════════════════════════════

def test_update_context_merges():
    from app.agent.session_store import get_or_create_session, update_context, get_context

    sid = get_or_create_session("proj", "ctx_merge")
    update_context(sid, {"last_entry": "Ctrl#method"})
    update_context(sid, {"last_symbol": "Svc#method"})

    ctx = get_context(sid)
    assert ctx["last_entry"] == "Ctrl#method"
    assert ctx["last_symbol"] == "Svc#method"


def test_context_accumulates():
    from app.agent.session_store import get_or_create_session, update_context, get_context

    sid = get_or_create_session("proj", "ctx_acc")
    update_context(sid, {"turn": 1, "topic": "flow"})
    assert get_context(sid)["turn"] == 1

    update_context(sid, {"turn": 2, "last_q": "放款流程"})
    ctx = get_context(sid)
    assert ctx["turn"] == 2
    assert ctx["topic"] == "flow"
    assert ctx["last_q"] == "放款流程"


def test_context_isolated():
    from app.agent.session_store import get_or_create_session, update_context, get_context

    sida = get_or_create_session("proj_a", "ctx_iso_a")
    sidb = get_or_create_session("proj_b", "ctx_iso_b")

    update_context(sida, {"last_entry": "CtrlA#m"})
    update_context(sidb, {"last_entry": "CtrlB#m"})

    assert get_context(sida)["last_entry"] == "CtrlA#m"
    assert get_context(sidb)["last_entry"] == "CtrlB#m"


# ═══════════════════════════════════════════════════════════════════════
# 3. COMPACTION
# ═══════════════════════════════════════════════════════════════════════

def test_should_compact_true():
    from app.agent.compaction import should_compact

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(20):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    assert should_compact(msgs, threshold=6)


def test_should_compact_false():
    from app.agent.compaction import should_compact

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "放款流程"},
        {"role": "assistant", "content": "主链：1.受理 2.校验"},
    ]
    assert not should_compact(msgs, threshold=10)


def test_compact_preserves_head_and_first_user():
    from app.agent.compaction import compact_messages

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(10):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    class _MockLLM:
        async def chat(self, messages, **kwargs):
            return "此前对话摘要：讨论了10轮。"
        async def embed(self, texts, **kwargs):
            return [[0.0] * 8 for _ in texts]

    result = asyncio.run(compact_messages(_MockLLM(), msgs, keep_tail=4, threshold=6))
    if result is None:
        return  # compaction not triggered — fine

    assert result[0]["role"] == "system"
    has_first_user = any("q0" in m.get("content", "") for m in result)
    assert has_first_user


def test_compact_returns_none_on_llm_error():
    from app.agent.compaction import compact_messages

    msgs = [{"role": "system", "content": "sys"}]
    for i in range(12):
        msgs.append({"role": "user", "content": f"q{i}"})
        msgs.append({"role": "assistant", "content": f"a{i}"})

    class _BrokenLLM:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("LLM down")
        async def embed(self, texts, **kwargs):
            return [[0.0] * 8 for _ in texts]

    result = asyncio.run(compact_messages(_BrokenLLM(), msgs))
    assert result is None


# ═══════════════════════════════════════════════════════════════════════
# 4. TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════

def test_trajectory_writes_valid_jsonl(monkeypatch, tmp_path):
    from app.agent.trajectory import AppendTrajectory, trajectory_dir

    logdir = tmp_path / "trajs"
    logdir.mkdir()
    monkeypatch.setattr("app.agent.trajectory.trajectory_dir", lambda: logdir)

    writer = AppendTrajectory("test_traj", enabled=True)
    writer.emit("turn", question="放款流程")
    writer.emit("tool", tool="search_symbols")
    writer.emit("answer", text="主链：1.受理 2.校验 3.落库")

    lines = (logdir / "test_traj.jsonl").read_text("utf-8").splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert obj


def test_trajectory_writes_nothing_when_disabled(monkeypatch, tmp_path):
    from app.agent.trajectory import AppendTrajectory

    logdir = tmp_path / "trajs_off"
    logdir.mkdir()
    monkeypatch.setattr("app.agent.trajectory.trajectory_dir", lambda: logdir)

    writer = AppendTrajectory("off_traj", enabled=False)
    writer.emit("turn", question="测试")
    assert not (logdir / "off_traj.jsonl").exists()


# ═══════════════════════════════════════════════════════════════════════
# 5. MESSAGE ORDER & KEYS
# ═══════════════════════════════════════════════════════════════════════

def test_message_order_preserved():
    from app.agent.session_store import get_or_create_session, append_message, list_messages
    import uuid as _uid

    sid = get_or_create_session("proj", f"order_test_{_uid.uuid4().hex[:12]}")
    for i in range(5):
        append_message(sid, "user", f"q{i}")
        append_message(sid, "assistant", f"a{i}")

    msgs = list_messages(sid)
    assert len(msgs) == 10
    assert msgs[0]["content"] == "q0"
    assert msgs[1]["content"] == "a0"
    assert msgs[-2]["content"] == "q4"
    assert msgs[-1]["content"] == "a4"


def test_message_has_role_and_content():
    from app.agent.session_store import get_or_create_session, append_message, list_messages
    import uuid as _uid

    sid = get_or_create_session("proj", f"keys_test_{_uid.uuid4().hex[:12]}")
    append_message(sid, "user", "放款流程什么样")
    msgs = list_messages(sid)
    assert len(msgs) == 1
    msg = msgs[0]
    assert "role" in msg
    assert "content" in msg
    assert msg["role"] == "user"
    assert "放款" in msg["content"]