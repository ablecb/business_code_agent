from __future__ import annotations

import asyncio

from app.agent.client_patch_loop import (
    create_patch_verify_run,
    run_patch_verify_session,
    submit_patch_result,
)
from app.agent.remote_protocol import ActionResult
from app.tagging.search import hybrid_search_multi


def test_patches_to_actions_queue():
    run = create_patch_verify_run(
        patches=[
            {
                "file": "src/A.java",
                "replacements": [{"old": "a", "new": "b"}],
            }
        ],
        project_id="demo",
    )
    types = [a.type for a in run.action_queue]
    assert types[0] == "apply_patch"
    assert types[-1] == "maven_compile"
    assert run.id.startswith("pvr_")


def test_patch_verify_succeeds_on_compile_ok():
    run = create_patch_verify_run(
        patches=[{"file": "f.java", "replacements": [{"old": "x", "new": "y"}]}],
        max_repairs=1,
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
    assert run.status == "succeeded"


def test_hybrid_search_multi_empty():
    assert hybrid_search_multi("x", []) == []
