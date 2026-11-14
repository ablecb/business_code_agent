"""Complex code-gen: patch → compile-fail → LLM repair → recompile → success.

Level-6 tests for the client patch verify loop (server-side orchestration of
"apply patch, compile, judge, repair, recompile"). The LLM judge/repair steps
are faked to force every branch deterministically:
  - compile fail → repair actions → recompile → success
  - apply_patch fail → repair budget exhaustion → failed
  - write_file action path
  - repair JSON parsing (fenced / raw / bad)
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.client_patch_loop import (
    _mk,
    _parse_repair_actions,
    _truncate,
    create_patch_verify_run,
    run_patch_verify_session,
    stop_patch_run,
    submit_patch_result,
)
from app.agent.remote_protocol import ActionResult, ClientAction


def test_compile_fail_then_repair_then_success(monkeypatch):
    """Fail compile → LLM returns repair patch → recompile → done."""
    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}],
        max_repairs=1,
    )

    async def _fake_after_compile(run_, log):
        if "BUILD SUCCESS" in (log or ""):
            return [ClientAction(id="act_done", type="done", note="编译通过")]
        return [
            ClientAction(
                id="act_repair",
                type="apply_patch",
                path="f.java",
                replacements=[{"old": "y", "new": "z"}],
            ),
            ClientAction(id="act_recompile", type="maven_compile", note="修复后重新编译验证"),
        ]

    monkeypatch.setattr(
        "app.agent.client_patch_loop._llm_after_compile", _fake_after_compile
    )

    compile_count = 0

    async def emit(msg):
        nonlocal compile_count
        if msg.get("type") == "action":
            action = msg["action"]
            if action["type"] == "maven_compile":
                compile_count += 1
                is_repair = compile_count > 1
                stdout = "BUILD SUCCESS" if is_repair else "ERROR: cannot find symbol"
                ok = is_repair
            else:
                stdout = "ok"
                ok = True
            submit_patch_result(
                run,
                ActionResult(
                    action_id=action["id"],
                    type=action["type"],
                    ok=ok,
                    exit_code=0 if "SUCCESS" in stdout else 1,
                    stdout=stdout,
                ),
            )

    asyncio.run(asyncio.wait_for(run_patch_verify_session(run, emit), timeout=5))
    assert run.status == "succeeded", run.message
    assert compile_count >= 2
    # the repair patch was applied before the successful compile
    types = [s.action.type for s in run.steps]
    assert types.count("apply_patch") >= 2


def test_patch_failure_exhausts_repairs():
    # max_repairs=0 → first apply_patch failure terminates the run
    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}],
        max_repairs=0,
    )

    async def emit(msg):
        if msg.get("type") == "action":
            action = msg["action"]
            submit_patch_result(
                run,
                ActionResult(
                    action_id=action["id"],
                    type=action["type"],
                    ok=False,
                    error="file not found",
                ),
            )

    asyncio.run(asyncio.wait_for(run_patch_verify_session(run, emit), timeout=5))
    assert run.status == "failed"
    assert "补丁应用失败" in run.message


def test_write_file_action_success():
    run = create_patch_verify_run(
        patches=[{"file": "new/Thing.java", "content": "class Thing {}"}],
        max_repairs=1,
    )
    assert run.action_queue[0].type == "write_file"

    async def emit(msg):
        if msg.get("type") == "action":
            action = msg["action"]
            if action["type"] == "maven_compile":
                ok, stdout = True, "BUILD SUCCESS"
            else:
                ok, stdout = True, "ok"
            submit_patch_result(
                run,
                ActionResult(
                    action_id=action["id"],
                    type=action["type"],
                    ok=ok,
                    exit_code=0,
                    stdout=stdout,
                ),
            )

    asyncio.run(asyncio.wait_for(run_patch_verify_session(run, emit), timeout=5))
    assert run.status == "succeeded"


def test_llm_judge_fail_action_finishes_failed(monkeypatch):
    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}],
        max_repairs=1,
    )

    async def _fake_after_compile(run_, log):
        return [ClientAction(id="act_fail", type="fail", note="无法判定编译结果")]

    monkeypatch.setattr(
        "app.agent.client_patch_loop._llm_after_compile", _fake_after_compile
    )

    async def emit(msg):
        if msg.get("type") == "action":
            action = msg["action"]
            submit_patch_result(
                run,
                ActionResult(
                    action_id=action["id"],
                    type=action["type"],
                    ok=True,
                    exit_code=0,
                    stdout="BUILD SUCCESS",
                ),
            )

    asyncio.run(asyncio.wait_for(run_patch_verify_session(run, emit), timeout=5))
    assert run.status == "failed"


# ---------- repair action parsing ----------

def test_parse_repair_actions_json():
    acts = _parse_repair_actions(
        '{"actions":[{"type":"apply_patch","path":"a.java",'
        '"replacements":[{"old":"x","new":"y"}]},{"type":"maven_compile"}]}'
    )
    assert len(acts) == 2
    assert acts[0].type == "apply_patch"
    assert dict(acts[0].replacements[0]) == {"old": "x", "new": "y"}
    assert acts[1].type == "maven_compile"


def test_parse_repair_actions_fenced():
    acts = _parse_repair_actions(
        '```json\n{"actions":[{"type":"done","note":"编译通过"}]}\n```'
    )
    assert len(acts) == 1
    assert acts[0].type == "done"


def test_parse_repair_actions_garbage():
    assert _parse_repair_actions("抱歉，无法修复。") == []
    assert _parse_repair_actions("") == []


def test_truncate_middle_ellipsis():
    t = _truncate("a" * 100 + "b" * 100, n=30)
    assert "…" in t
    assert len(t) < 200


def test_repair_actions_unknown_type_dropped():
    acts = _parse_repair_actions(
        '{"actions":[{"type":"hack_the_planet","note":"no"}]}'
    )
    assert acts == []


# ---------- misc ----------

def test_patch_run_to_response_shape():
    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}],
        max_repairs=1,
    )
    resp = run.to_response()
    assert resp.id == run.id
    assert resp.max_repairs == 1
    assert resp.step_count == 0


def test_stop_patch_run_marks_stopped():
    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}]
    )
    stopped = stop_patch_run(run.id)
    assert stopped is not None
    assert stopped.status == "stopped"


def test_get_patch_run_and_list():
    from app.agent.client_patch_loop import get_patch_run, list_patch_runs

    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}]
    )
    assert get_patch_run(run.id) is run
    assert any(r.id == run.id for r in list_patch_runs(limit=50))
