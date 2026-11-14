"""Full pipeline integration quality — parse → graph → search → answer.

Tests the *integration* of pipeline stages. Most use mock LLM/embedding
so they remain deterministic and fast, but they exercise the real orchestration paths.
"""

from __future__ import annotations

import asyncio


# ═══════════════════════════════════════════════════════════════════════
# 1. TOOL DISPATCH
# ═══════════════════════════════════════════════════════════════════════


def test_dispatch_finish_marks_finished():
    from app.agent.explore_tools import dispatch_tool

    out = dispatch_tool(
        "demo",
        "finish",
        {"answer": "主链：\n1. 受理\n2. 校验\n3. 落库"},
    )
    assert out.get("finished") is True
    assert "主链：" in out.get("answer", "")


def test_dispatch_unknown_returns_ok_false():
    from app.agent.explore_tools import dispatch_tool

    out = dispatch_tool("demo", "no_such_tool_xyz123", {})
    assert out.get("ok") is False


# ═══════════════════════════════════════════════════════════════════════
# 2. TASK PLANNER
# ═══════════════════════════════════════════════════════════════════════


def test_task_planner_plan_has_required_keys():
    from app.agent.task_planner import build_research_plan

    plan = build_research_plan("放款流程什么样")
    assert plan is not None
    assert "steps" in plan or "tasks" in plan or "plan" in plan


def test_task_planner_classify_kind():
    from app.agent.task_planner import classify_question_kind

    # classify_question_kind always returns "general" per architecture (keyword routing removed)
    assert classify_question_kind("放款流程什么样") in {"general", "flow", "process", "api", "other"}
    assert classify_question_kind("你好") in {"general", "greeting", "other"}


# ═══════════════════════════════════════════════════════════════════════
# 3. RUNTIME PROFILE
# ═══════════════════════════════════════════════════════════════════════


def test_runtime_profile_allows_common_skills():
    from app.agent.runtime.profile import get_runtime_profile

    prof = get_runtime_profile()
    assert prof is not None
    assert hasattr(prof, "skill_allowed")
    assert prof.skill_allowed("explain_flow")
    assert prof.skill_allowed("show_code")
    assert prof.skill_allowed("compose_code")
    assert prof.skill_allowed("general")


def test_runtime_profile_minimal_mode_restricts_repairs(monkeypatch):
    from app.core.config import get_settings
    from app.agent.runtime.profile import get_runtime_profile, clear_runtime_profile_cache

    settings = get_settings()
    monkeypatch.setattr(settings, "agent_runtime_mode", "minimal")
    monkeypatch.setattr(settings, "agent_minimal_max_repairs", 0)

    clear_runtime_profile_cache()
    prof = get_runtime_profile()
    assert prof.mode == "minimal"
    assert prof.max_repairs == 0


# ═══════════════════════════════════════════════════════════════════════
# 4. ORCHESTRATOR — task grouping
# ═══════════════════════════════════════════════════════════════════════


def test_orchestrator_groups_by_entry_hint():
    from app.agent.runtime.orchestrator import group_subtasks

    tasks = [
        {"id": "t1", "kind": "explain_flow", "entry_hint": "Ctrl#a"},
        {"id": "t2", "kind": "show_code", "entry_hint": "Ctrl#a"},
        {"id": "t3", "kind": "explain_flow", "entry_hint": "Svc#b"},
    ]
    groups = group_subtasks(tasks, share_entry=False)
    assert len(groups) == 2
    assert any("Ctrl#a" in str(g) for g in groups)
    assert any("Svc#b" in str(g) for g in groups)


def test_orchestrator_serial_order_respects_deps():
    from app.agent.runtime.orchestrator import serial_task_order

    tasks = [
        {"id": "t2", "depends_on": ["t1"]},
        {"id": "t1", "depends_on": []},
        {"id": "t3", "depends_on": ["t2"]},
    ]
    ordered = serial_task_order(tasks)
    ids = [t["id"] for t in ordered]
    assert ids.index("t1") < ids.index("t2") < ids.index("t3")


# ═══════════════════════════════════════════════════════════════════════
# 5. COMPOSE CODE — evidence pack keys
# ═══════════════════════════════════════════════════════════════════════


def test_compose_evidence_pack_has_expected_keys(monkeypatch):
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: ["Svc#method"],
    )
    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body",
        lambda *a, **k: ("public void method() {}", "x.java"),
    )
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict", lambda *a, **k: []
    )

    pack = cc.gather_compose_evidence("demo-loan", "查某表", lookup_key="id", lookup_table="某表")

    assert "hits" in pack
    assert "snippets" in pack
    assert "lookup_key" in pack
    assert "lookup_table" in pack
    assert "lookup_steps" in pack


# ═══════════════════════════════════════════════════════════════════════
# 6. KNOWLEDGE — chunker
# ═══════════════════════════════════════════════════════════════════════


def test_knowledge_chunker_splits_on_boundary():
    from app.knowledge.chunker import chunk_text

    text = "\n\n".join([f"# 第{i}章\n\n这是第{i}章内容。" for i in range(5)])
    chunks = chunk_text(text, size=30, overlap=5)

    assert len(chunks) >= 2
    assert all(len(c) > 0 for c in chunks)


def test_knowledge_chunker_preserves_all_content():
    """All original paragraphs must be recoverable from joined chunks."""
    from app.knowledge.chunker import chunk_text

    text = "".join(f"段落{i}。\n" for i in range(50))
    chunks = chunk_text(text, size=120, overlap=0)

    assert len(chunks) >= 1
    joined = "".join(chunks)
    for i in range(50):
        para = f"段落{i}。"
        assert para in joined, f"paragraph {i} missing from reconstructed output"


def test_knowledge_service_validates_file_type():
    import pytest
    from app.knowledge.service import save_upload

    with pytest.raises(ValueError, match="仅支持"):
        save_upload(filename="bad.exe", data=b"fake binary")


# ═══════════════════════════════════════════════════════════════════════
# 7. CHAT FEEDBACK (via ORM helper — not a dedicated module)
# ═══════════════════════════════════════════════════════════════════════


def test_feedback_upsert_stores_rating():
    from app.agent.session_store import (
        get_or_create_session,
        append_message,
        upsert_feedback,
    )
    from app.core.utils import new_id

    sid = get_or_create_session("proj", new_id("test_fb_"))
    aid = append_message(sid, "assistant", "这是一个测试回答")
    assert aid is not None, "assistant message must have an id"

    result = upsert_feedback(
        session_id=sid,
        assistant_message_id=aid,
        vote="up",
        project_id="proj",
        comment="回答完整",
    )
    assert result is not None
    assert "session_id" in result
    assert result.get("vote") == "up"