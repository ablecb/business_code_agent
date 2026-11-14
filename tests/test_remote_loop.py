from __future__ import annotations

import asyncio

from app.agent.remote_loop import (
    _parse_llm_actions,
    create_run,
    get_run,
    run_agent_session,
    submit_action_result,
)
from app.agent.remote_protocol import ActionResult, ClientAction, RemoteRunCreate


def _use_mock_llm() -> None:
    import os

    os.environ["LLM_PROVIDER"] = "mock"
    from app.core.config import get_settings
    from app.llm.factory import get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()


def test_parse_llm_actions_json():
    text = """
    {
      "actions": [
        {"type": "list_dir", "path": "."},
        {"type": "maven_install"},
        {"type": "done", "note": "ok"}
      ],
      "reasoning": "x"
    }
    """
    acts = _parse_llm_actions(text)
    assert [a.type for a in acts] == ["list_dir", "maven_install", "done"]


def test_parse_llm_actions_fenced():
    text = """```json
    {"actions":[{"type":"read_file","path":"pom.xml"}]}
    ```"""
    acts = _parse_llm_actions(text)
    assert len(acts) == 1
    assert acts[0].type == "read_file"
    assert acts[0].path == "pom.xml"


def test_post_tool_summary_enriches_thin_done():
    """After shell results, thin done.reply must become a user-facing conclusion."""
    _use_mock_llm()
    from app.agent.remote_loop import create_agent_run, run_agent_session, submit_action_result
    from app.agent.remote_protocol import ActionResult

    run = create_agent_run(
        goal="帮我看看我本地git有没有安装以及对应的路径",
        session_id="sess_git_summary_test",
        max_steps=8,
    )
    finished = []

    async def emit(msg):
        if msg.get("type") == "run_finished":
            finished.append(msg)
        if msg.get("type") != "action":
            return
        a = msg["action"]
        stdout = "ok"
        if a["type"] in {"run_powershell", "run_shell"}:
            stdout = (
                "C:\\Program Files\\Git\\cmd\\git.exe\n"
                "git version 2.45.1.windows.1\n"
            )
        submit_action_result(
            run,
            ActionResult(
                action_id=a["id"],
                type=a["type"],
                ok=True,
                exit_code=0,
                stdout=stdout,
            ),
        )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=15))
    assert run.status == "succeeded"
    assert finished
    reply = finished[-1].get("reply") or run.final_reply or ""
    assert "Git" in reply or "git" in reply.lower()
    assert "见上方" not in reply
    assert "路径" in reply or "Program Files" in reply or "version" in reply.lower()


def test_reply_needs_enrichment_flags_thin_text():
    from app.agent.remote_loop import _reply_needs_enrichment

    assert _reply_needs_enrichment("")
    assert _reply_needs_enrichment("已根据本机命令输出汇总：见上方结果。")
    assert not _reply_needs_enrichment(
        "本地已安装 Git。\n路径：C:\\Program Files\\Git\\cmd\\git.exe\n版本：git version 2.45.1"
    )


def test_evaluate_shell_repeat_policy():
    """Hard anti-repeat rules must be deterministic (customer-facing stability)."""
    from app.agent.remote_loop import (
        RemoteRun,
        _mk_action,
        evaluate_shell_repeat,
    )
    from app.agent.remote_protocol import RunStep

    run = RemoteRun(id="pol", goal="看看java有没有安装")

    def add(cmd: str, stdout: str = "", stderr: str = "", *, rejected: bool = False):
        act = _mk_action("run_shell", command=cmd)
        meta = {"rejected": True, "server_guard": True} if rejected else {}
        run.steps.append(
            RunStep(
                action=act,
                result=ActionResult(
                    action_id=act.id,
                    type="run_shell",
                    ok=not rejected,
                    exit_code=0 if not rejected else None,
                    stdout=stdout,
                    stderr=stderr,
                    error="guard" if rejected else None,
                    meta=meta,
                ),
                ts=float(len(run.steps) + 1),
            )
        )

    # Fresh run: where allowed.
    assert evaluate_shell_repeat(run, "where.exe java")[0] == "allow"

    add(
        "where.exe java",
        stdout="D:\\java\\java17\\bin\\java.exe\n",
    )
    # Exact duplicate → finish (enough path evidence).
    assert evaluate_shell_repeat(run, "where.exe java")[0] == "finish"
    # First -version still allowed (no version text yet).
    assert evaluate_shell_repeat(run, "java -version")[0] == "allow"

    add("java -version", stderr='java version "17.0.10"\n')
    # Path+version present → any more java probe finishes.
    assert evaluate_shell_repeat(run, "java -version")[0] == "finish"
    assert evaluate_shell_repeat(run, "where.exe java")[0] == "finish"
    # Unrelated env cmd still allowed once.
    assert evaluate_shell_repeat(run, "Get-ChildItem Env:JAVA_HOME")[0] == "allow"
    add("Get-ChildItem Env:JAVA_HOME", stdout="JAVA_HOME D:\\java\\java17\n")
    assert evaluate_shell_repeat(run, "Get-ChildItem Env:JAVA_HOME")[0] == "finish"

    # printenv failure then same cmd without path evidence → reject (not finish).
    run2 = RemoteRun(id="pol2", goal="环境变量")
    act = _mk_action("run_shell", command="printenv")
    run2.steps.append(
        RunStep(
            action=act,
            result=ActionResult(
                action_id=act.id,
                type="run_shell",
                ok=False,
                exit_code=1,
                stderr="CommandNotFoundException",
            ),
            ts=1.0,
        )
    )
    assert evaluate_shell_repeat(run2, "printenv")[0] == "reject"
    assert evaluate_shell_repeat(run2, "Get-ChildItem Env:PATH")[0] == "allow"


def test_soft_ok_and_reject_repeated_java_probe():
    """Crash-y java -version with path evidence must soft-ok; repeats must stop."""
    from app.agent.remote_loop import (
        RemoteRun,
        _count_tool_probes,
        _mk_action,
        _probe_tool_name,
        _run_has_probe_evidence,
        _soft_ok_probe_result,
        evaluate_shell_repeat,
    )
    from app.agent.remote_protocol import RunStep

    act = _mk_action(
        "run_powershell",
        command="where.exe java; java -version",
    )
    result = ActionResult(
        action_id=act.id,
        type="run_powershell",
        ok=False,
        exit_code=-1073741819,
        stdout="C:\\Program Files\\Java\\bin\\java.exe\nD:\\java\\java17\\bin\\java.exe\n",
        stderr='java version "17.0.10"\n',
        error="exit -1073741819",
    )
    soft = _soft_ok_probe_result(act, result)
    assert soft.ok is True
    assert soft.meta.get("soft_ok_server") is True

    run = RemoteRun(id="java_loop", goal="看看java有没有安装")
    run.steps.append(RunStep(action=act, result=soft, ts=1.0))
    assert _run_has_probe_evidence(run)
    assert _probe_tool_name(act.command) == "java"
    assert _count_tool_probes(run, "java") == 1
    # Path+version already in one step → further java probes finish.
    assert evaluate_shell_repeat(run, "java -version")[0] == "finish"
    assert evaluate_shell_repeat(run, "where.exe java")[0] == "finish"


def test_java_probe_loop_stops_in_session():
    """If the model keeps asking java -version, the loop must cut it off and finish."""
    _use_mock_llm()
    from app.agent import remote_loop as rl

    run = rl.create_agent_run(goal="帮我看看java有没有安装", max_steps=10)
    calls = {"n": 0}
    orig_decide = rl._llm_decide

    async def looping_decide(r):
        calls["n"] += 1
        if calls["n"] <= 6:
            return [
                rl._mk_action(
                    "run_powershell",
                    command="java -version",
                    cwd=".",
                )
            ]
        return await orig_decide(r)

    rl._llm_decide = looping_decide  # type: ignore[assignment]
    try:
        shell_hits = {"n": 0}

        async def emit(msg):
            if msg.get("type") != "action":
                return
            a = msg["action"]
            if a["type"] in {"run_powershell", "run_shell"}:
                shell_hits["n"] += 1
                ok = False
                stdout = ""
                stderr = ""
                code = -1073741819
                if shell_hits["n"] == 1:
                    stdout = "C:\\Program Files\\Common Files\\Oracle\\Java\\javapath\\java.exe\n"
                    stderr = 'java version "1.8.0_281"\n'
                submit_action_result(
                    run,
                    ActionResult(
                        action_id=a["id"],
                        type=a["type"],
                        ok=ok,
                        exit_code=code,
                        stdout=stdout,
                        stderr=stderr,
                        error=f"exit {code}",
                    ),
                )
            else:
                submit_action_result(
                    run,
                    ActionResult(
                        action_id=a["id"],
                        type=a["type"],
                        ok=True,
                        exit_code=0,
                        stdout="ok",
                    ),
                )

        asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=15))
    finally:
        rl._llm_decide = orig_decide  # type: ignore[assignment]

    assert shell_hits["n"] == 1  # second+ probes finish server-side without re-exec
    assert run.status == "succeeded"


def test_where_then_version_allowed_once():
    """where success must still allow one java -version if version text missing."""
    from app.agent.remote_loop import RemoteRun, _mk_action, evaluate_shell_repeat
    from app.agent.remote_protocol import RunStep

    run = RemoteRun(id="wv", goal="检查java")
    act = _mk_action("run_shell", command="where.exe java")
    run.steps.append(
        RunStep(
            action=act,
            result=ActionResult(
                action_id=act.id,
                type="run_shell",
                ok=True,
                exit_code=0,
                stdout="D:\\java\\java17\\bin\\java.exe\n",
            ),
            ts=1.0,
        )
    )
    assert evaluate_shell_repeat(run, "java -version")[0] == "allow"
    assert evaluate_shell_repeat(run, "where.exe java")[0] == "finish"


def test_compound_goal_check_then_pull():
    """Compound goal: check install then pull — must not stop after probe alone."""
    _use_mock_llm()
    from app.agent.remote_loop import create_agent_run

    run = create_agent_run(
        goal="帮我看看git是否安装，如果安装，帮我拉取一下最新的代码",
        max_steps=12,
    )
    seen = []

    async def emit(msg):
        if msg.get("type") != "action":
            return
        a = msg["action"]
        seen.append(a["type"])
        stdout = "ok"
        if a["type"] in {"run_powershell", "run_shell"}:
            stdout = "D:\\Git\\cmd\\git.exe\ngit version 2.45.2.windows.1\n"
        elif a["type"] == "git_pull":
            stdout = "Already up to date."
        submit_action_result(
            run,
            ActionResult(
                action_id=a["id"],
                type=a["type"],
                ok=True,
                exit_code=0,
                stdout=stdout,
            ),
        )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=15))
    assert run.status == "succeeded"
    assert any(t in {"run_powershell", "run_shell"} for t in seen)
    assert "git_pull" in seen
    assert len(run.plan_steps) >= 2


def test_run_session_done_queue():
    _use_mock_llm()
    run = create_run(
        RemoteRunCreate(
            goal="test",
            allow_docker=False,
            allow_install=False,
            allow_process=False,
            allow_write=False,
            max_steps=5,
        )
    )
    run.bootstrap_done = True
    run.action_queue = [ClientAction(id="a1", type="done", note="already ok")]

    emitted = []

    async def emit(msg):
        emitted.append(msg)

    asyncio.run(run_agent_session(run, emit))
    assert run.status == "succeeded"
    assert any(m.get("type") == "run_finished" for m in emitted)


def test_action_result_unblocks():
    _use_mock_llm()
    run = create_run(
        RemoteRunCreate(
            goal="probe",
            allow_docker=False,
            allow_install=False,
            allow_process=False,
            max_steps=3,
        )
    )
    run.bootstrap_done = True
    run.action_queue = [
        ClientAction(id="list1", type="list_dir", path="."),
        ClientAction(id="done1", type="done", note="ok"),
    ]

    async def emit(msg):
        if msg.get("type") == "action":
            action = msg["action"]
            submit_action_result(
                run,
                ActionResult(
                    action_id=action["id"],
                    type=action["type"],
                    ok=True,
                    content="f\tpom.xml",
                    stdout="ok",
                ),
            )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=5))
    assert run.status == "succeeded"
    assert len(run.steps) >= 1


def test_task_plan_emitted_before_actions():
    _use_mock_llm()

    from app.agent.remote_loop import create_agent_run

    run = create_agent_run(goal="帮我启动一下项目", max_steps=10)
    events = []

    async def emit(msg):
        events.append(msg)
        if msg.get("type") != "action":
            return
        a = msg["action"]
        stdout = "ok"
        meta = {}
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
    plan_ev = next(
        (e for e in events if e.get("event") == "task_plan"),
        None,
    )
    assert plan_ev is not None
    assert run.plan_done
    assert len(run.plan_steps) >= 2
    assert run.success_criteria
    assert run.status == "succeeded"


def test_premature_fail_becomes_ask_user():
    from app.agent.remote_loop import _guard_terminal_action, _mk_action, RemoteRun

    run = RemoteRun(id="r1", goal="启动", max_repairs=5, repair_count=0)
    guarded = _guard_terminal_action(
        run, _mk_action("fail", note="算了", reply="启动失败")
    )
    assert guarded is not None
    assert guarded.type == "ask_user"
    ids = {o.get("id") for o in (guarded.options or [])}
    assert "continue_repair" in ids
    assert "abandon" in ids

    run.repair_count = 5
    allowed = _guard_terminal_action(run, _mk_action("fail", note="用尽"))
    assert allowed is not None
    assert allowed.type == "fail"


def test_reject_done_after_critical_failure():
    from app.agent.remote_loop import RemoteRun, _guard_terminal_action, _mk_action
    from app.agent.remote_protocol import RunStep

    run = RemoteRun(id="r2", goal="启动")
    act = _mk_action("start_app")
    run.steps.append(
        RunStep(
            action=act,
            result=ActionResult(
                action_id=act.id, type="start_app", ok=False, stderr="port in use"
            ),
            ts=1.0,
        )
    )
    assert _guard_terminal_action(run, _mk_action("done", note="假装成功")) is None


def test_start_app_failure_increments_repair():
    _use_mock_llm()
    from app.agent.remote_loop import create_agent_run

    run = create_agent_run(goal="帮我启动项目", max_steps=12, max_repairs=5)
    start_attempts = {"n": 0}

    async def emit(msg):
        if msg.get("type") != "action":
            return
        a = msg["action"]
        ok = True
        stdout = "ok"
        stderr = ""
        if a["type"] == "start_app":
            start_attempts["n"] += 1
            if start_attempts["n"] == 1:
                ok = False
                stderr = "APPLICATION FAILED TO START\nPort 8080 already in use"
                stdout = ""
            else:
                stdout = "Started DemoApplication in 2.0 seconds"
        submit_action_result(
            run,
            ActionResult(
                action_id=a["id"],
                type=a["type"],
                ok=ok,
                exit_code=0 if ok else 1,
                stdout=stdout,
                stderr=stderr,
            ),
        )

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=15))
    assert start_attempts["n"] >= 2
    assert run.repair_count >= 1
    assert run.status == "succeeded"


def test_action_fingerprint_drops_identical_retry():
    from app.agent.remote_loop import (
        RemoteRun,
        _action_fingerprint,
        _drop_tried_fingerprints,
        _filter_actions,
        _mk_action,
        _record_tried_method,
        _same_action_fingerprint,
        _tried_methods_prompt,
    )

    run = RemoteRun(id="fp1", goal="push")
    a1 = _mk_action("git_push", remote="origin", ref="main")
    a2 = _mk_action("git_push", remote="origin", ref="main")
    a3 = _mk_action("git_push", remote="origin", ref="develop")
    assert _same_action_fingerprint(a1, a2)
    assert not _same_action_fingerprint(a1, a3)

    _record_tried_method(
        run,
        a1,
        ActionResult(
            action_id=a1.id,
            type="git_push",
            ok=False,
            exit_code=1,
            error="authentication failed",
            stderr="AUTHENTICATION failed for origin",
        ),
    )
    assert run.tried_methods
    assert "禁止原样重试" in _tried_methods_prompt(run)

    kept = _drop_tried_fingerprints(run, [a2, a3])
    assert [x.type + (x.ref or "") for x in kept] == ["git_pushdevelop"]

    shell_fail = _mk_action("run_shell", command="git push origin main")
    _record_tried_method(
        run,
        shell_fail,
        ActionResult(
            action_id=shell_fail.id,
            type="run_shell",
            ok=False,
            exit_code=1,
            stderr="rejected non-fast-forward",
        ),
    )
    dup = _mk_action("run_shell", command="git  push   origin main")
    assert _action_fingerprint(shell_fail) == _action_fingerprint(dup)
    filtered = _filter_actions(run, [dup, _mk_action("git_status")])
    assert [x.type for x in filtered] == ["git_status"]


def test_failure_digest_three_failures_and_hints():
    from app.agent.remote_loop import RemoteRun, _failure_digest, _mk_action
    from app.agent.remote_protocol import RunStep

    run = RemoteRun(id="dig", goal="compile")
    cases = [
        ("run_shell", "earlier failure should be dropped when only last 3 kept"),
        ("maven_compile", "COMPILATION ERROR\nFoo.java:[10,1] cannot find symbol"),
        ("git_push", "AUTHENTICATION failed\nremote rejected"),
        ("run_shell", "BUILD FAILURE\nCould not find artifact com.x:y:1.0"),
    ]
    for i, (atype, stderr) in enumerate(cases, start=1):
        if atype == "git_push":
            act = _mk_action("git_push", remote="origin", ref="main")
        elif atype == "maven_compile":
            act = _mk_action("maven_compile")
        else:
            act = _mk_action("run_shell", command=f"cmd-{i}")
        run.steps.append(
            RunStep(
                action=act,
                result=ActionResult(
                    action_id=act.id,
                    type=atype,
                    ok=False,
                    exit_code=1,
                    stderr=stderr,
                ),
                ts=float(i),
            )
        )
    digest = _failure_digest(run)
    # Last 3 failures only (oldest of the four omitted).
    assert "earlier failure should be dropped" not in digest
    assert "COMPILATION ERROR" in digest or "cannot find symbol" in digest
    assert "AUTHENTICATION" in digest or "authentication" in digest.lower()
    assert "Could not find artifact" in digest
    assert "建议方向" in digest


def test_repair_budget_exhausted_asks_user_not_silent_fail():
    """When repairs are exhausted, loop asks the user instead of silent fail."""
    _use_mock_llm()
    from app.agent.remote_loop import create_agent_run

    run = create_agent_run(
        goal="跑一条必然失败的 shell",
        max_steps=20,
        max_repairs=2,
    )
    run.repair_count = 3
    run.plan_done = True
    run.plan_steps = ["run failing command", "fix"]
    run.success_criteria = "command succeeds"
    finished = []

    async def emit(msg):
        if msg.get("type") == "run_finished":
            finished.append(msg)

    asyncio.run(asyncio.wait_for(run_agent_session(run, emit), timeout=15))
    assert finished
    assert run.repair_exhausted_asked is True
    assert run.final_phase == "awaiting_option"
    assert finished[-1].get("phase") == "awaiting_option"
    blob = (finished[-1].get("reply") or "") + (run.final_reply or "")
    assert "换方法" in blob or "修复" in blob
    ids = {
        str(o.get("id"))
        for o in (finished[-1].get("options") or [])
        if isinstance(o, dict)
    }
    assert "continue_repair" in ids
    assert "abandon" in ids
    # Must not hard-fail on first exhaustion.
    assert finished[-1].get("status") == "succeeded"


def test_stuck_reply_explains_progress():
    from app.agent.remote_loop import (
        RemoteRun,
        _mk_action,
        _user_facing_stuck_reply,
    )
    from app.agent.remote_protocol import ActionResult, RunStep

    run = RemoteRun(
        id="stuck1",
        goal="写入后新建分支并提交推送",
        allow_write=True,
        allow_git=True,
    )
    run.steps.extend(
        [
            RunStep(
                action=(
                    w := _mk_action(
                        "write_file",
                        path="a/Foo.java",
                        content="class Foo {}",
                    )
                ),
                result=ActionResult(
                    action_id=w.id, type="write_file", ok=True, stdout="wrote"
                ),
            ),
            RunStep(
                action=(gs := _mk_action("git_status")),
                result=ActionResult(
                    action_id=gs.id,
                    type="git_status",
                    ok=True,
                    stdout="?? a/Foo.java\nOn branch main",
                ),
            ),
            RunStep(
                action=(gc := _mk_action("git_checkout", branch="main")),
                result=ActionResult(
                    action_id=gc.id,
                    type="git_checkout",
                    ok=True,
                    stdout="Already on main",
                ),
            ),
        ]
    )
    reply = _user_facing_stuck_reply(run)
    assert "当前目标" in reply
    assert "刚才实际做了什么" in reply
    assert "write_file" in reply
    assert "还缺什么" in reply
    assert "commit" in reply.lower() or "提交" in reply
    assert "push" in reply.lower() or "推送" in reply
    assert "请你直接回复下一步" in reply


def test_drop_noop_git_checkout_when_already_on_branch():
    from app.agent.remote_loop import (
        RemoteRun,
        _drop_redundant_git_actions,
        _mk_action,
    )
    from app.agent.remote_protocol import ActionResult, RunStep

    run = RemoteRun(id="g1", goal="提交并推送", allow_git=True)
    st = _mk_action("git_status")
    run.steps.append(
        RunStep(
            action=st,
            result=ActionResult(
                action_id=st.id,
                type="git_status",
                ok=True,
                stdout="## main...origin/main\n?? a.java",
            ),
        )
    )
    kept = _drop_redundant_git_actions(
        run,
        [
            _mk_action("git_checkout", branch="main"),
            _mk_action("git_commit", message="feat: x"),
        ],
    )
    types = [a.type for a in kept]
    assert "git_checkout" not in types
    assert "git_commit" in types


def test_after_commit_push_decide_finishes_not_checkout_loop(monkeypatch):
    import asyncio
    import json

    from app.agent import remote_loop as rl
    from app.agent.remote_loop import RemoteRun, _mk_action
    from app.agent.remote_protocol import ActionResult, RunStep

    run = RemoteRun(id="g2", goal="提交并推送到远程", allow_git=True)
    c = _mk_action("git_commit", message="提交信息")
    p = _mk_action("git_push", remote="origin", branch="main")
    run.steps.extend(
        [
            RunStep(
                action=c,
                result=ActionResult(
                    action_id=c.id, type="git_commit", ok=True, stdout="[main abc] ok"
                ),
            ),
            RunStep(
                action=p,
                result=ActionResult(
                    action_id=p.id, type="git_push", ok=True, stdout="main -> main"
                ),
            ),
        ]
    )

    class FakeLLM:
        async def chat(self, *args, **kwargs):
            return json.dumps(
                {
                    "actions": [
                        {"type": "git_checkout", "branch": "main"},
                        {"type": "git_checkout", "branch": "main"},
                    ],
                    "reasoning": "再确认一下分支",
                }
            )

    monkeypatch.setattr(rl, "get_llm_provider", lambda: FakeLLM())
    actions = asyncio.run(rl._llm_decide(run))
    assert len(actions) == 1
    assert actions[0].type == "done"
