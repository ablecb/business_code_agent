"""Tests for harness-style trajectory / runtime profile / tool registry."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agent.runtime.profile import (
    MINIMAL_TOOLS,
    clear_runtime_profile_cache,
    get_runtime_profile,
)
from app.agent.runtime.registry import filter_task_tools, known_tool, list_tools, tools_for_skill
from app.agent.trajectory import AppendTrajectory, read_trajectory, wrap_progress
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _clear_caches(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path / "traj"))
    monkeypatch.setenv("AGENT_TRAJECTORY_ENABLED", "true")
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "standard")
    get_settings.cache_clear()
    clear_runtime_profile_cache()
    yield
    get_settings.cache_clear()
    clear_runtime_profile_cache()


def test_trajectory_append_and_read(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path / "traj"))
    get_settings.cache_clear()
    traj = AppendTrajectory.for_session("sess-demo", turn_id="turn1", compact=False)
    traj.emit("user_turn", type="user_turn", message="hello")
    traj.emit("progress", type="step", step_id="goal", status="done", note="ok")
    traj.emit("final", type="final", ok=True, intent="ask")
    rows = read_trajectory("sess-demo")
    assert len(rows) == 3
    assert rows[0]["kind"] == "user_turn"
    assert rows[0]["turn_id"] == "turn1"
    assert rows[-1]["kind"] == "final"
    assert traj.path.exists()
    line = traj.path.read_text(encoding="utf-8").strip().splitlines()[0]
    assert json.loads(line)["session_id"] == "sess-demo"


def test_wrap_progress_forwards_and_records(tmp_path: Path, monkeypatch):
    import asyncio

    monkeypatch.setenv("AGENT_TRAJECTORY_DIR", str(tmp_path / "traj"))
    get_settings.cache_clear()
    seen: list[dict] = []

    async def _cb(ev: dict) -> None:
        seen.append(ev)

    traj = AppendTrajectory.for_session("sess-wrap", compact=True)
    wrapped = wrap_progress(_cb, traj)
    assert wrapped is not None

    async def _run() -> None:
        await wrapped({"type": "step", "step_id": "T1", "status": "running", "note": "x" * 400})

    asyncio.run(_run())
    assert len(seen) == 1
    rows = read_trajectory("sess-wrap")
    assert len(rows) == 1
    assert rows[0]["kind"] == "progress"
    assert rows[0]["step_id"] == "T1"
    assert len(str(rows[0].get("note") or "")) <= 240


def test_runtime_profile_minimal(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "minimal")
    monkeypatch.setenv("AGENT_MINIMAL_MAX_REPAIRS", "0")
    get_settings.cache_clear()
    clear_runtime_profile_cache()
    prof = get_runtime_profile()
    assert prof.mode == "minimal"
    assert prof.allow_decompose is False
    assert prof.carry_dig_in_pins is False
    assert prof.max_repairs == 0
    assert prof.tool_allowed("resolve_entry")
    assert not prof.tool_allowed("compose_code")
    tools = {t["name"] for t in list_tools(profile=prof)}
    assert tools <= set(MINIMAL_TOOLS)
    assert "compose_code" not in tools
    assert tools_for_skill("compose_code", profile=prof) == []
    assert "resolve_entry" in tools_for_skill("explain_flow", profile=prof)


def test_filter_task_tools_marks_skipped(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "minimal")
    get_settings.cache_clear()
    clear_runtime_profile_cache()
    tasks = [
        {"id": "T1", "tool": "resolve_entry", "status": "pending"},
        {"id": "T2", "tool": "compose_code", "status": "pending"},
    ]
    out = filter_task_tools(tasks)
    assert out[0]["status"] == "pending"
    assert out[1]["status"] == "skipped"
    assert known_tool("resolve_entry")
    assert known_tool("compose_code")


def test_build_task_plan_respects_minimal(monkeypatch):
    monkeypatch.setenv("AGENT_RUNTIME_MODE", "minimal")
    get_settings.cache_clear()
    clear_runtime_profile_cache()
    from app.agent.pev_agent import build_task_plan

    tasks = build_task_plan({"goal": "compose_code", "source": "test"})
    assert any(t.get("status") == "skipped" for t in tasks)
    assert all(
        t.get("status") == "skipped" or t.get("tool") in MINIMAL_TOOLS for t in tasks
    )
