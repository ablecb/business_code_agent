"""Smoke: unified LLM-routed client agent (no keyword product gate)."""
from __future__ import annotations

import asyncio
import json
import os

os.environ["LLM_PROVIDER"] = "mock"

from app.agent.remote_loop import (
    create_agent_run,
    create_cleanup_agent_run,
    create_patch_agent_run,
    route_client_need,
    run_agent_session,
    submit_action_result,
)
from app.agent.remote_protocol import ActionResult
from app.llm.factory import get_llm_provider


def _use_mock_llm():
    os.environ["LLM_PROVIDER"] = "mock"
    get_llm_provider.cache_clear()


def test_route_timeout_fails_closed_to_none(monkeypatch):
    """路由 LLM 慢端点必须被 LLM_ROUTE_TIMEOUT 兜住，快速 fail-closed none。

    回归：desktop-chat-997ffe53135d —— 路由判定调用拖满 240s LLM_TOTAL_TIMEOUT
    才回退 none，用户 UI 空白 4 分钟。修复后 route_client_need 内 30s 硬上限，
    超时即返回 mode=none（安全方向，不启动任何本机动作）。
    """
    import time

    from app.agent import remote_loop

    class _SlowLLM:
        async def chat(
            self, messages, *, temperature=None, max_tokens=None, response_format=None
        ):
            del messages, temperature, max_tokens, response_format
            await asyncio.sleep(60)
            return "{}"

    monkeypatch.setattr("app.agent.remote_loop.get_llm_provider", lambda: _SlowLLM())

    class _Cfg:
        llm_route_timeout = 1.0

    monkeypatch.setattr("app.core.config.get_settings", lambda: _Cfg())

    t0 = time.time()
    r = asyncio.run(remote_loop.route_client_need("帮我看看 C 盘垃圾"))
    elapsed = time.time() - t0
    assert r["mode"] == "none"
    assert r["needs_client"] is False
    assert elapsed < 5.0, f"expected fast fail-closed none, took {elapsed:.1f}s"


def test_route_disk_direct_not_none():
    _use_mock_llm()
    r = asyncio.run(route_client_need("帮我看看 C 盘垃圾"))
    assert r["mode"] == "direct"
    assert r["goal"]


def test_route_qa_is_none():
    _use_mock_llm()
    r = asyncio.run(route_client_need("放款流程什么样"))
    assert r["mode"] == "none"


def test_route_code_change():
    _use_mock_llm()
    r = asyncio.run(route_client_need("放款接口加字段并自测"))
    assert r["mode"] == "code"


def test_route_local_git_check_is_direct():
    _use_mock_llm()
    r = asyncio.run(route_client_need("帮我看看我本地是否按照了git以及git路径"))
    assert r["mode"] == "direct"
    assert r["goal"]


def test_route_english_java_check_is_direct():
    _use_mock_llm()
    r = asyncio.run(route_client_need("Is Java installed on my machine and what is the path?"))
    assert r["mode"] == "direct"


def test_route_install_start_direct():
    _use_mock_llm()
    r = asyncio.run(route_client_need("把依赖装好并启动"))
    assert r["mode"] == "direct"


def test_disk_scan_then_ask_user_options():
    _use_mock_llm()
    run = create_agent_run(
        goal="分析磁盘占用并给出可清理项",
        allow_write=False,
        allow_install=False,
        allow_process=False,
        allow_docker=False,
        allow_sys=True,
        max_steps=8,
    )
    scan_payload = {
        "cleanup_candidates": [
            {
                "id": "user_temp",
                "label": "用户临时文件",
                "size_human": "1.2 GB",
                "safe": True,
                "kind": "temp",
                "path": "C:\\Users\\x\\AppData\\Local\\Temp",
            }
        ]
    }

    async def emit(msg):
        if msg.get("type") != "action":
            return
        action = msg["action"]
        content = None
        stdout = "ok"
        if action["type"] == "sys_disk_overview":
            stdout = "C: free=20G total=100G"
        elif action["type"] == "sys_scan_heavy":
            content = json.dumps(scan_payload, ensure_ascii=False)
            stdout = "scan done"
        submit_action_result(
            run,
            ActionResult(
                action_id=action["id"],
                type=action["type"],
                ok=True,
                stdout=stdout,
                content=content,
                meta=scan_payload if action["type"] == "sys_scan_heavy" else {},
            ),
        )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=10))
    assert run.status == "succeeded"
    assert run.final_phase == "awaiting_cleanup"
    assert any(o.get("id") == "user_temp" for o in run.final_options)
    user_temp = next(o for o in run.final_options if o.get("id") == "user_temp")
    assert "Temp" in str(user_temp.get("path") or "")
    assert "路径" in str(user_temp.get("summary") or "") or "Temp" in str(
        user_temp.get("path") or ""
    )


def test_cleanup_run_sys_cleanup():
    run = create_cleanup_agent_run(
        goal="按选择清理",
        option_ids=["user_temp"],
        catalog=[{"id": "user_temp", "title": "用户临时文件"}],
    )
    seen = []

    async def emit(msg):
        if msg.get("type") == "action":
            seen.append(msg["action"]["type"])
            submit_action_result(
                run,
                ActionResult(
                    action_id=msg["action"]["id"],
                    type=msg["action"]["type"],
                    ok=True,
                    stdout="已清理 user_temp",
                ),
            )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=5))
    assert "sys_cleanup" in seen
    assert run.status == "succeeded"
    assert "清理" in (run.final_reply or run.message or "")


def test_patch_apply_only_skips_compile():
    run = create_patch_agent_run(
        patches=[{"file": "A.java", "replacements": [{"old": "x", "new": "y"}]}],
        goal="只应用补丁",
        with_compile=False,
    )
    types = [a.type for a in run.action_queue]
    assert "apply_patch" in types
    assert "maven_compile" not in types
    assert types[-1] == "done"


def test_patch_apply_with_followup_has_no_early_done():
    run = create_patch_agent_run(
        patches=[{"file": "A.java", "content": "class A {}", "new_file": True}],
        goal="写入清单文件后新建分支 feature/x 再提交并推送",
        with_compile=False,
        continue_after_write=True,
    )
    types = [a.type for a in run.action_queue]
    assert types == ["write_file"]
    assert "done" not in types
    assert "git_commit" not in types  # LLM decides follow-up
    assert run.write_allowlist
    assert any("分支" in s or "提交" in s or "推送" in s for s in run.plan_steps)
    assert "全部完成" in (run.success_criteria or "") or "用户目标" in (
        run.success_criteria or ""
    )


def test_patch_compile_then_followup_no_hardcoded_git():
    run = create_patch_agent_run(
        patches=[{"file": "A.java", "replacements": [{"old": "x", "new": "y"}]}],
        goal="改代码并 push",
        with_compile=True,
        with_git=True,
    )
    types = [a.type for a in run.action_queue]
    assert types == ["apply_patch", "maven_compile"]
    assert "git_push" not in types


def test_filter_blocks_write_outside_allowlist():
    from app.agent.remote_loop import RemoteRun, _filter_actions, _mk_action

    run = RemoteRun(id="al", goal="x")
    run.write_allowlist = ["src/A.java"]
    kept = _filter_actions(
        run,
        [
            _mk_action("write_file", path="src/A.java", content="ok"),
            _mk_action("write_file", path="src/Evil.java", content="no"),
            _mk_action("git_status"),
        ],
    )
    assert [a.type + ":" + (a.path or "") for a in kept] == [
        "write_file:src/A.java",
        "git_status:",
    ]


def test_patch_compile_succeeds():
    run = create_patch_agent_run(
        patches=[{"file": "A.java", "replacements": [{"old": "x", "new": "y"}]}],
        goal="应用补丁并编译自测",
        max_repairs=1,
        with_compile=True,
    )

    async def emit(msg):
        if msg.get("type") == "action":
            a = msg["action"]
            submit_action_result(
                run,
                ActionResult(
                    action_id=a["id"],
                    type=a["type"],
                    ok=True,
                    exit_code=0,
                    stdout="BUILD SUCCESS",
                ),
            )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=5))
    assert run.status == "succeeded"
    types = [s.action.type for s in run.steps]
    assert "apply_patch" in types
    assert "maven_compile" in types


def test_install_start_queue_from_llm():
    _use_mock_llm()
    run = create_agent_run(goal="把依赖装好并启动", max_steps=10)
    seen = []

    async def emit(msg):
        if msg.get("type") != "action":
            return
        a = msg["action"]
        seen.append(a["type"])
        meta = {}
        stdout = "ok"
        if a["type"] == "start_app":
            stdout = "Started DemoApplication in 1.2 seconds"
            meta = {"alive": True}
        submit_action_result(
            run,
            ActionResult(
                action_id=a["id"],
                type=a["type"],
                ok=True,
                exit_code=0,
                stdout=stdout,
                meta=meta,
            ),
        )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=10))
    assert "maven_install" in seen
    assert "start_app" in seen
    assert run.status == "succeeded"
