"""Unified server brain: LLM decides whitelist client actions (code/ops/OS)."""
from __future__ import annotations

import asyncio
import json
import logging
import platform
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent.action_policy import annotate_action_dict
from app.agent.remote_protocol import (
    ALLOWED_ACTIONS,
    TERMINAL_STATUSES,
    ActionResult,
    ClientAction,
    RemoteRunCreate,
    RemoteRunResponse,
    RunStep,
    RunStatus,
)
from app.core.utils import new_id
from app.llm.factory import get_llm_provider
from app.agent.prompts import load_prompt

logger = logging.getLogger("app.agent.remote_loop")

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]

SYSTEM_PROMPT = load_prompt("system_prompt")
SUMMARIZE_SYSTEM = load_prompt("summarize_system")
PLAN_SYSTEM = load_prompt("plan_system")
_REPAIRABLE_TYPES = frozenset(
    {
        "maven_compile",
        "maven_install",
        "run_sql",
        "run_cypher",
        "milvus_search",
        "milvus_mutate",
        "open_browser",
        "start_app",
        "http_smoke",
        "docker_compose_up",
        "apply_patch",
        "write_file",
        "run_shell",
        "run_powershell",
        "sys_cleanup",
        "git_merge",
        "git_pull",
        "git_push",
        "git_commit",
        "git_checkout",
    }
)

_CRITICAL_TYPES = frozenset(
    {
        "start_app",
        "maven_compile",
        "maven_install",
        "http_smoke",
        "sys_cleanup",
        "docker_compose_up",
        "apply_patch",
        "write_file",
        "run_sql",
        "run_cypher",
        "milvus_mutate",
    }
)

ROUTE_PROMPT = load_prompt("route_prompt")


@dataclass
class RemoteRun:
    id: str
    goal: str
    project_id: str | None = None
    session_id: str | None = None
    local_root_hint: str | None = None
    status: RunStatus = "idle"
    message: str | None = None
    max_steps: int = 20
    max_repairs: int = 5
    repair_count: int = 0
    success_pattern: str = r"Started \S+Application"
    allow_write: bool = True
    allow_install: bool = True
    allow_process: bool = True
    allow_docker: bool = True
    allow_sys: bool = True
    allow_shell: bool = True
    allow_git: bool = True
    steps: list[RunStep] = field(default_factory=list)
    pending_action: ClientAction | None = None
    action_queue: list[ClientAction] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    bootstrap_done: bool = True  # default: LLM decides first step (no forced maven)
    plan_done: bool = False
    plan_steps: list[str] = field(default_factory=list)
    plan_text: str = ""
    success_criteria: str = ""
    # User already confirmed cleanup ids in a prior turn (skip re-scan guard).
    cleanup_preapproved: bool = False
    final_reply: str = ""
    final_options: list[dict[str, Any]] = field(default_factory=list)
    final_phase: str = "done"
    # Structured ledger of failed repairable methods (fingerprint → ban identical retry).
    tried_methods: list[dict[str, Any]] = field(default_factory=list)
    last_decide_reasoning: str = ""
    repair_exhausted_asked: bool = False
    # When set, write_file/apply_patch outside this list are filtered out.
    write_allowlist: list[str] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _result_event: asyncio.Event = field(default_factory=asyncio.Event)
    _last_result: ActionResult | None = None
    _log_buf: str = ""

    def touch(self, status: RunStatus | None = None, message: str | None = None) -> None:
        self.updated_at = time.time()
        if status is not None:
            self.status = status
        if message is not None:
            self.message = message

    def to_response(self) -> RemoteRunResponse:
        return RemoteRunResponse(
            id=self.id,
            status=self.status,
            goal=self.goal,
            project_id=self.project_id,
            local_root_hint=self.local_root_hint,
            step_count=len(self.steps),
            repair_count=self.repair_count,
            max_steps=self.max_steps,
            max_repairs=self.max_repairs,
            message=self.message,
            steps=list(self.steps),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


_runs: dict[str, RemoteRun] = {}


def get_run(run_id: str) -> RemoteRun | None:
    return _runs.get(run_id)


def list_runs(limit: int = 50) -> list[RemoteRunResponse]:
    items = sorted(_runs.values(), key=lambda r: r.created_at, reverse=True)
    return [r.to_response() for r in items[:limit]]


def create_run(req: RemoteRunCreate) -> RemoteRun:
    rid = new_id("run_")
    run = RemoteRun(
        id=rid,
        goal=(req.goal or "").strip() or "完成本机任务",
        project_id=req.project_id,
        local_root_hint=req.local_root_hint,
        max_steps=max(1, min(int(req.max_steps or 20), 80)),
        max_repairs=max(0, min(int(req.max_repairs or 5), 20)),
        success_pattern=req.success_pattern or r"Started \S+Application",
        allow_write=bool(req.allow_write),
        allow_install=bool(req.allow_install),
        allow_process=bool(req.allow_process),
        allow_docker=bool(req.allow_docker),
        allow_shell=bool(getattr(req, "allow_shell", True)),
        allow_git=bool(getattr(req, "allow_git", True)),
        status="idle",
        message="已创建，等待客户端连接",
        bootstrap_done=True,
    )
    _runs[rid] = run
    return run


def create_agent_run(
    *,
    goal: str,
    project_id: str | None = None,
    session_id: str | None = None,
    local_root_hint: str | None = None,
    initial_actions: list[ClientAction] | None = None,
    max_steps: int = 40,
    max_repairs: int = 8,
    allow_write: bool = True,
    allow_install: bool = True,
    allow_process: bool = True,
    allow_docker: bool = True,
    allow_sys: bool = True,
    allow_shell: bool = True,
    allow_git: bool = True,
) -> RemoteRun:
    rid = new_id("run_")
    run = RemoteRun(
        id=rid,
        goal=(goal or "").strip() or "完成本机任务",
        project_id=project_id,
        session_id=session_id,
        local_root_hint=local_root_hint,
        max_steps=max(1, min(int(max_steps), 80)),
        max_repairs=max(0, min(int(max_repairs), 20)),
        allow_write=allow_write,
        allow_install=allow_install,
        allow_process=allow_process,
        allow_docker=allow_docker,
        allow_sys=allow_sys,
        allow_shell=allow_shell,
        allow_git=allow_git,
        status="idle",
        message="已创建，等待客户端连接",
        bootstrap_done=True,
        action_queue=list(initial_actions or []),
        # Pre-seeded queues already encode the plan (patch/cleanup/tests).
        plan_done=bool(initial_actions),
    )
    _runs[rid] = run
    return run


def _goal_needs_followup(goal: str) -> bool:
    """True when user goal has work beyond writing the file list."""
    t = goal or ""
    if re.search(
        r"(?i)分支|branch|提交|推送|commit|push|远程|"
        r"编译|compile|maven|启动|start[_ ]?app|自测|验证|smoke",
        t,
    ):
        return True
    if len(t) > 60 and re.search(r"并|然后|之后|同时|再", t):
        return True
    return False


def _norm_allow_path(path: str | None) -> str:
    return str(path or "").replace("\\", "/").strip().lstrip("./").lower()


def _path_in_allowlist(path: str | None, allow: list[str]) -> bool:
    if not allow:
        return True
    p = _norm_allow_path(path)
    if not p:
        return False
    for raw in allow:
        a = _norm_allow_path(raw)
        if not a:
            continue
        if p == a or p.endswith("/" + a) or a.endswith("/" + p) or p.endswith(a):
            return True
    return False


def create_patch_agent_run(
    *,
    patches: list[dict[str, Any]],
    goal: str = "在本机应用补丁并编译通过",
    project_id: str | None = None,
    session_id: str | None = None,
    max_repairs: int = 3,
    path_hints: list[str] | None = None,
    with_compile: bool = True,
    with_git: bool = False,
    commit_message: str | None = None,
    continue_after_write: bool | None = None,
) -> RemoteRun:
    """Queue patch writes; optional compile; remaining goal parts via LLM decide.

    ``with_git`` / ``commit_message`` are kept for API compat but no longer
    hard-code commit/push into the queue — follow-up work is decided dynamically.
    """
    from app.agent.patcher import normalize_patch_paths

    del commit_message  # API compat; LLM chooses commit message dynamically

    acts: list[ClientAction] = []
    normalized = normalize_patch_paths(
        [p for p in (patches or []) if isinstance(p, dict)],
        hints=path_hints,
        project_id=project_id,
    )
    allow: list[str] = []
    for p in normalized:
        path = p.get("path") or p.get("file")
        if not path:
            continue
        allow.append(str(path))
        reps = p.get("replacements") or []
        if reps:
            acts.append(
                _mk_action(
                    "apply_patch",
                    path=str(path),
                    replacements=[
                        {
                            "old": r.get("old") or r.get("old_string") or "",
                            "new": r.get("new") or r.get("new_string") or "",
                        }
                        for r in reps
                        if isinstance(r, dict)
                    ],
                    note="应用补丁",
                )
            )
        elif p.get("content") is not None:
            acts.append(
                _mk_action(
                    "write_file",
                    path=str(path),
                    content=str(p.get("content") or ""),
                    note="写入文件",
                )
            )
    if path_hints:
        for h in path_hints:
            if h and str(h) not in allow:
                allow.append(str(h))

    if with_compile:
        acts.append(_mk_action("maven_compile", note="编译验证"))

    follow = continue_after_write
    if follow is None:
        follow = bool(with_git) or _goal_needs_followup(goal)

    if not follow and not with_compile:
        acts.append(
            _mk_action(
                "done",
                note="补丁已应用（未编译自测）",
                reply="代码已按补丁写入本机，未执行编译。",
            )
        )

    run = create_agent_run(
        goal=goal,
        project_id=project_id,
        session_id=session_id,
        initial_actions=acts,
        max_repairs=max(max_repairs, 5 if follow else max_repairs),
        allow_sys=False,
        allow_git=True,
        max_steps=48 if follow else 40,
    )
    run.write_allowlist = list(allow)
    if follow:
        run.plan_steps = [
            "写入清单内文件（禁止改清单外）",
            "按用户完整目标继续：如新建分支 / 提交 / 推送 / 编译 / 启动等",
            "全部完成后 done；未达成功标准禁止提前结束",
        ]
        run.success_criteria = (
            "用户目标全部完成（含分支/提交/推送/编译等剩余部分）；"
            "仅清单内文件被修改"
        )
        run.plan_text = " → ".join(run.plan_steps)
        run.plan_done = True
    elif with_compile:
        run.plan_steps = ["写入清单内文件", "编译验证", "确认成功后结束"]
        run.success_criteria = "清单文件已写入且编译成功"
        run.plan_text = " → ".join(run.plan_steps)
        run.plan_done = True
    return run


def create_cleanup_agent_run(
    *,
    goal: str,
    option_ids: list[str],
    catalog: list[dict[str, Any]] | None = None,
    session_id: str | None = None,
) -> RemoteRun:
    act = _mk_action(
        "sys_cleanup",
        note="清理用户勾选项",
        ids=[str(x) for x in (option_ids or []) if x],
        catalog_json=json.dumps(catalog or [], ensure_ascii=False),
    )
    run = create_agent_run(
        goal=goal or "按选择清理",
        session_id=session_id,
        initial_actions=[act],
        allow_write=False,
        allow_install=False,
        allow_process=False,
        allow_docker=False,
        allow_sys=True,
        max_steps=8,
    )
    run.cleanup_preapproved = True
    return run


def stop_run(run_id: str) -> RemoteRun | None:
    run = _runs.get(run_id)
    if not run:
        return None
    run.action_queue.clear()
    run.pending_action = None
    run.touch("stopped", "用户停止")
    run._result_event.set()
    return run


def _mk_action(atype: str, **kwargs: Any) -> ClientAction:
    payload = {"id": new_id("act_"), "type": atype, **kwargs}
    try:
        return ClientAction.model_validate(payload)
    except Exception:
        return ClientAction.model_construct(**payload)


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit // 2] + "\n…[truncated]…\n" + text[-(limit // 2) :]


def _normalize_drive(action: ClientAction) -> ClientAction:
    """Prefer drive=C over path=C:\\ for sys_* actions."""
    if not str(action.type or "").startswith("sys_"):
        return action
    drive = (action.drive or "").strip()
    path = (action.path or "").strip()
    if not drive and path:
        m = re.match(r"^([A-Za-z])[:\\/]?", path)
        if m:
            drive = m.group(1).upper()
    if not drive:
        drive = "C"
    drive = drive[:1].upper()
    try:
        return action.model_copy(update={"drive": drive, "path": None})
    except Exception:
        action.drive = drive
        action.path = None
        return action


def _has_successful_scan(run: RemoteRun) -> bool:
    for step in run.steps:
        if step.action.type == "sys_scan_heavy" and step.result and step.result.ok:
            return True
    return False


def _validate_sys_cleanup(run: RemoteRun, action: ClientAction) -> str | None:
    """Return error message if cleanup must be rejected; else None."""
    ids = [str(x) for x in (action.ids or []) if x]
    if not ids:
        return (
            "拒绝 sys_cleanup：缺少 ids。"
            "请先 sys_scan_heavy，再用 ask_user 让用户勾选可清理项，"
            "最后 sys_cleanup {ids:[已确认的id...]}。"
        )
    # Reject obvious whole-disk attempts
    bad = {"C", "C:", "C:\\", "C:/", "/", "\\"}
    if any(str(i).strip() in bad for i in ids):
        return "拒绝 sys_cleanup：不可把整盘路径当作清理 id。"
    if run.cleanup_preapproved:
        return None
    if not _has_successful_scan(run):
        return (
            "拒绝 sys_cleanup：尚未成功执行 sys_scan_heavy。"
            "顺序必须是 overview → scan_heavy → ask_user → cleanup。"
        )
    return None


def _append_session_client_log(run: RemoteRun, msg: dict[str, Any]) -> None:
    """Mirror client-agent events into data/desktop_chat/<session>.log."""
    if not run.session_id:
        return
    try:
        from app.agent.chat_trace import get_session_chat_logger, log_client_run_event

        log_client_run_event(get_session_chat_logger(run.session_id), msg)
    except Exception:
        logger.debug("session client log failed", exc_info=True)


def _inject_rejected_result(run: RemoteRun, action: ClientAction, error: str) -> None:
    """Record a synthetic failed result so the LLM can re-plan (no client execute)."""
    result = ActionResult(
        action_id=action.id,
        type=action.type,
        ok=False,
        error=error,
        stderr=error,
        stdout="",
        meta={"rejected": True, "server_guard": True},
    )
    run.steps.append(RunStep(action=action, result=result, ts=time.time()))
    run._last_result = result
    run.pending_action = None
    run.touch("observing", f"服务端拒绝: {action.type}")
    _append_session_client_log(
        run,
        {
            "type": "tool_event",
            "event": "tool_result",
            "run_id": run.id,
            "action": action.model_dump(),
            "result": result.model_dump(),
        },
    )


def _action_fingerprint(action: ClientAction) -> str:
    """Stable id for 'same method' — type + command/path/remote/ref."""
    t = str(action.type or "")
    if t in {"run_shell", "run_powershell"}:
        return f"{t}|{_norm_shell_cmd(getattr(action, 'command', None))}"
    if t.startswith("git_"):
        remote = str(getattr(action, "remote", None) or "").strip().lower()
        ref = str(
            getattr(action, "ref", None)
            or getattr(action, "branch", None)
            or ""
        ).strip().lower()
        msg = str(getattr(action, "message", None) or "").strip().lower()[:80]
        return f"{t}|{remote}|{ref}|{msg}"
    path = str(getattr(action, "path", None) or "").strip().lower().replace("\\", "/")
    urls = getattr(action, "urls", None) or []
    url = str(getattr(action, "url", None) or "")
    if urls and not url:
        url = ",".join(str(u) for u in urls[:4])
    if t == "http_smoke":
        return f"{t}|{url.strip().lower()}"
    return f"{t}|{path}"


def _same_action_fingerprint(a: ClientAction, b: ClientAction) -> bool:
    return _action_fingerprint(a) == _action_fingerprint(b)


def _record_tried_method(
    run: RemoteRun, action: ClientAction, result: ActionResult
) -> None:
    err = (result.error or "").strip()
    if not err and result.stderr:
        lines = [ln.strip() for ln in result.stderr.splitlines() if ln.strip()]
        err = lines[-1] if lines else ""
    if not err:
        err = (result.stdout or "")[:160]
    entry = {
        "type": action.type,
        "command": getattr(action, "command", None),
        "path": getattr(action, "path", None),
        "remote": getattr(action, "remote", None),
        "ref": getattr(action, "ref", None) or getattr(action, "branch", None),
        "error": _truncate(err, 200),
        "exit_code": result.exit_code,
        "fingerprint": _action_fingerprint(action),
    }
    run.tried_methods.append(entry)


def _tried_methods_prompt(run: RemoteRun) -> str:
    if not run.tried_methods:
        return ""
    # Deduplicate by fingerprint, keep last error text.
    by_fp: dict[str, dict[str, Any]] = {}
    for m in run.tried_methods:
        fp = str(m.get("fingerprint") or "")
        if fp:
            by_fp[fp] = m
    lines = ["已失败方法（禁止原样重试）："]
    for i, m in enumerate(by_fp.values(), 1):
        label = (
            m.get("command")
            or m.get("path")
            or " ".join(
                x
                for x in (m.get("remote"), m.get("ref"))
                if x
            )
            or ""
        )
        lines.append(
            f"{i}. {m.get('type')} {label} — exit={m.get('exit_code')} — "
            f"{_truncate(str(m.get('error') or ''), 120)}"
        )
    lines.append(
        "请换另一种可行做法（例如换参数/远程/命令、先读日志再 patch、ask_user 要凭据），"
        "禁止重复列表中的组合。"
    )
    return "\n".join(lines)


def _drop_tried_fingerprints(
    run: RemoteRun, actions: list[ClientAction]
) -> list[ClientAction]:
    """
    Drop identical retries.

    - shell / mutating git: ban any fingerprint already in tried_methods
    - other types (start_app, maven_*, …): only ban matching the *latest* failed
      step so a later verify retry after a different fix step remains allowed
    """
    strong_banned = {
        str(m.get("fingerprint") or "")
        for m in run.tried_methods
        if m.get("fingerprint")
        and (
            str(m.get("type") or "") in {"run_shell", "run_powershell"}
            or (
                str(m.get("type") or "").startswith("git_")
                and str(m.get("type") or "")
                not in {"git_status", "git_branch"}
            )
        )
    }
    last_fp: str | None = None
    for step in reversed(run.steps):
        if step.result is None:
            continue
        if step.result.ok:
            break
        last_fp = _action_fingerprint(step.action)
        break

    if not strong_banned and not last_fp:
        return actions
    out: list[ClientAction] = []
    for a in actions:
        if a.type in {"ask_user", "done", "fail"}:
            out.append(a)
            continue
        fp = _action_fingerprint(a)
        ban = fp in strong_banned or (last_fp is not None and fp == last_fp)
        if ban:
            logger.info(
                "drop duplicate failed method run=%s fingerprint=%s",
                run.id,
                fp,
            )
            continue
        out.append(a)
    return out


def _infer_current_branch(run: RemoteRun) -> str | None:
    """Best-effort current branch from recent git observations."""
    for step in reversed(run.steps):
        if not step.result or not step.result.ok:
            continue
        blob = "\n".join(
            x
            for x in (step.result.stdout or "", step.result.stderr or "")
            if x
        )
        if step.action.type == "git_status":
            m = re.search(r"(?m)^##\s+([^\s\.\n]+)", blob)
            if m:
                return m.group(1).strip()
        if step.action.type == "git_checkout":
            m = re.search(r"Already on ['\"]([^'\"]+)['\"]", blob, re.I)
            if m:
                return m.group(1).strip()
            m = re.search(r"Switched to (?:a new )?branch ['\"]([^'\"]+)['\"]", blob, re.I)
            if m:
                return m.group(1).strip()
            br = str(
                getattr(step.action, "branch", None)
                or getattr(step.action, "ref", None)
                or ""
            ).strip()
            if br:
                return br
    return None


def _successful_action_fingerprints(
    run: RemoteRun, types: set[str] | frozenset[str]
) -> set[str]:
    out: set[str] = set()
    for step in run.steps:
        if step.result and step.result.ok and step.action.type in types:
            out.add(_action_fingerprint(step.action))
    return out


def _git_commit_push_goal_met(run: RemoteRun) -> bool:
    """True when user asked commit+push and both already succeeded in this run."""
    goal = run.goal or ""
    wants_commit = bool(re.search(r"提交|commit", goal, re.I))
    wants_push = bool(re.search(r"推送|push", goal, re.I))
    if not (wants_commit and wants_push):
        return False
    ok = {
        s.action.type
        for s in run.steps
        if s.result and s.result.ok
    }
    return "git_commit" in ok and "git_push" in ok


def _drop_redundant_git_actions(
    run: RemoteRun, actions: list[ClientAction]
) -> list[ClientAction]:
    """
    Drop noop / already-done git mutations that only spam confirm dialogs.
    e.g. git_checkout main when already on main, or repeating a successful checkout.
    """
    cur = _infer_current_branch(run)
    succeeded = _successful_action_fingerprints(
        run,
        frozenset(
            {
                "git_checkout",
                "git_commit",
                "git_push",
                "git_pull",
                "git_merge",
            }
        ),
    )
    out: list[ClientAction] = []
    for a in actions:
        if a.type in {"ask_user", "done", "fail"}:
            out.append(a)
            continue
        fp = _action_fingerprint(a)
        if a.type in {
            "git_checkout",
            "git_push",
            "git_pull",
            "git_merge",
        } and fp in succeeded:
            logger.info(
                "drop already-succeeded git action run=%s fingerprint=%s",
                run.id,
                fp,
            )
            continue
        if a.type == "git_checkout":
            br = str(
                getattr(a, "branch", None) or getattr(a, "ref", None) or ""
            ).strip()
            create = bool(
                getattr(a, "create", None)
                or getattr(a, "new_branch", None)
                or getattr(a, "create_branch", None)
            )
            if (
                not create
                and cur
                and br
                and br.lower() == cur.lower()
            ):
                logger.info(
                    "drop noop git_checkout run=%s already on %s",
                    run.id,
                    cur,
                )
                continue
        out.append(a)
    return out


def _repair_direction_hint(action_type: str, blob: str) -> str:
    """Soft hint only — not a hard-coded path."""
    low = (blob or "").lower()
    t = action_type or ""
    if t.startswith("git_push") or "authentication" in low or "auth" in low and "fail" in low:
        return "建议方向：检查凭据/远程权限，可先 git_status，必要时 ask_user 要凭据；勿原样再 push。"
    if "non-fast-forward" in low or ("rejected" in low and "git" in t):
        return "建议方向：先 pull/对齐远程再推，确认分支；禁止 force push。"
    if "compilation error" in low or "编译" in blob:
        return "建议方向：根据报错文件 read_file，再 apply_patch 后重新 compile。"
    if "could not find artifact" in low or "dependency" in low and "fail" in low:
        return "建议方向：核对 pom/依赖坐标与仓库，必要时换依赖版本或先读相关配置。"
    if t in {"run_shell", "run_powershell"}:
        return "建议方向：换等价命令/参数，或按本机 OS 换工具；禁止同一 command 原样重试。"
    if t in {"apply_patch", "write_file"}:
        return "建议方向：根据编译/运行报错调整补丁内容，勿重复写入相同错误内容。"
    return "建议方向：根据失败摘要换参数或工具，勿原样重试。"


def _filter_actions(run: RemoteRun, actions: list[ClientAction]) -> list[ClientAction]:
    out: list[ClientAction] = []
    for a in actions:
        if a.type not in ALLOWED_ACTIONS:
            continue
        if a.type in {"write_file", "apply_patch"} and not run.allow_write:
            continue
        if a.type in {"write_file", "apply_patch"} and run.write_allowlist:
            if not _path_in_allowlist(getattr(a, "path", None), run.write_allowlist):
                logger.info(
                    "drop write outside allowlist run=%s path=%s",
                    run.id,
                    getattr(a, "path", None),
                )
                continue
        if a.type in {"maven_install", "maven_compile", "run_sql"} and not run.allow_install:
            continue
        if a.type in {"start_app", "stop_app"} and not run.allow_process:
            continue
        if a.type == "docker_compose_up" and not run.allow_docker:
            continue
        if a.type.startswith("sys_") and not run.allow_sys:
            continue
        if a.type in {"run_shell", "run_powershell"} and not run.allow_shell:
            continue
        if a.type.startswith("git_") and not run.allow_git:
            continue
        if not a.id:
            a = a.model_copy(update={"id": new_id("act_")})
        # Annotate confirm / UI preview for desktop interleaved timeline
        try:
            annotated = annotate_action_dict(a.model_dump())
            a = ClientAction.model_validate(annotated)
        except Exception:
            pass
        out.append(a)
    return _drop_redundant_git_actions(run, _drop_tried_fingerprints(run, out))


def _parse_json_obj(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return {}
        try:
            data = json.loads(m.group(0))
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}


def _parse_llm_actions(text: str) -> list[ClientAction]:
    data = _parse_json_obj(text)
    items = data.get("actions") if data else None
    if items is None:
        # whole payload might be actions-only parse attempt
        try:
            raw = json.loads((text or "").strip())
            items = raw if isinstance(raw, list) else None
        except Exception:
            items = None
    if not isinstance(items, list):
        return []
    out: list[ClientAction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("type") or item.get("action")
        if t not in ALLOWED_ACTIONS:
            continue
        try:
            payload = {**item, "type": t, "id": item.get("id") or new_id("act_")}
            reps = payload.get("replacements") or []
            norm = []
            for r in reps:
                if not isinstance(r, dict):
                    continue
                norm.append(
                    {
                        "old": r.get("old") or r.get("old_string") or "",
                        "new": r.get("new") or r.get("new_string") or "",
                    }
                )
            payload["replacements"] = norm
            opts = payload.get("options")
            if opts is not None and not isinstance(opts, list):
                payload["options"] = []
            out.append(ClientAction.model_validate(payload))
        except Exception:
            try:
                out.append(ClientAction.model_construct(**{**item, "type": t, "id": new_id("act_")}))
            except Exception:
                logger.exception("skip bad action %s", item)
    return out


def _observation_blob(run: RemoteRun, limit: int = 12_000) -> str:
    parts: list[str] = []
    for step in run.steps[-8:]:
        a = step.action
        r = step.result
        parts.append(f"### ACTION {a.type} id={a.id} path={a.path or ''}")
        if a.note:
            parts.append(f"note: {a.note}")
        if a.type == "http_smoke":
            if a.url:
                parts.append(f"smoke_url: {a.url}")
            if a.urls:
                parts.append("smoke_urls: " + ", ".join(str(u) for u in a.urls[:8]))
        if not r:
            parts.append("(no result yet — still running or waiting)")
            continue
        parts.append(f"ok={r.ok} exit={r.exit_code} error={r.error or ''}")
        if r.meta:
            parts.append("meta: " + _truncate(json.dumps(r.meta, ensure_ascii=False), 1500))
        if r.content:
            parts.append("content:\n" + _truncate(r.content, 4000))
        parts.append("stdout:\n" + _truncate(r.stdout, 3000))
        parts.append("stderr:\n" + _truncate(r.stderr, 3000))
    if run._log_buf:
        parts.append("### LIVE LOGS\n" + _truncate(run._log_buf, 4000))
    digest = _failure_digest(run)
    if digest:
        parts.append("\n### 失败摘要（优先据此修复）\n" + digest)
    parts.append(
        "\n### 分析要求\n"
        "请你根据以上原始输出自行判断成败与下一步；"
        "失败则进入修复，不要轻易 fail；"
        "不要假设已有固定规则判定启动成功或冒烟通过。"
    )
    return _truncate("\n".join(parts), limit)


def _failure_digest(run: RemoteRun, limit: int = 2500) -> str:
    """Compress recent failed step logs into actionable lines for the LLM."""
    chunks: list[str] = []
    for step in reversed(run.steps[-8:]):
        r = step.result
        if not r or r.ok:
            continue
        a = step.action
        header = f"[{a.type}] error={r.error or ''} exit={r.exit_code}"
        blob = "\n".join(
            x
            for x in (
                r.stderr or "",
                r.stdout or "",
                r.content or "",
                json.dumps(r.meta, ensure_ascii=False) if r.meta else "",
            )
            if x
        )
        lines = []
        for line in blob.splitlines():
            s = line.strip()
            if not s:
                continue
            if re.search(
                r"(?i)error|exception|failed|failure|fatal|cannot|unable|"
                r"拒绝|失败|错误|找不到|not found|denied|refused|timeout|BUILD FAILURE|"
                r"rejected|non-fast-forward|AUTHENTICATION|authentication|"
                r"Could not find artifact|COMPILATION ERROR|Compilation failure",
                s,
            ):
                lines.append(s)
            if len(lines) >= 12:
                break
        if not lines:
            lines = [s for s in (x.strip() for x in blob.splitlines()) if s][-8:]
        hint = _repair_direction_hint(str(a.type or ""), blob)
        chunks.append(header + "\n" + "\n".join(lines) + "\n" + hint)
        if len(chunks) >= 3:
            break
    return _truncate("\n\n".join(chunks), limit)


def _last_finished_step(run: RemoteRun) -> RunStep | None:
    for step in reversed(run.steps):
        if step.result is not None:
            return step
    return None


def _may_terminal_fail(run: RemoteRun) -> bool:
    """Allow fail only when repair budget / step budget is exhausted."""
    if run.repair_count >= run.max_repairs:
        return True
    if len(run.steps) >= max(1, run.max_steps - 1):
        return True
    return False


def _norm_shell_cmd(command: str | None) -> str:
    return re.sub(r"\s+", " ", (command or "").strip().lower())


def _is_env_probe_cmd(command: str | None) -> bool:
    c = command or ""
    return bool(
        re.search(
            r"(?i)\bwhere(?:\.exe)?\b|\bget-command\b|--version|-version|\bversion\b|"
            r"Get-ChildItem\s+Env:|\$env:|echo\s+%|printenv\b",
            c,
        )
    )


_PROBE_EVIDENCE_RE = re.compile(
    r"(?i)(?:git version|java version|openjdk|node v|Python \d|Apache Maven|"
    r"[A-Za-z]:\\[^\r\n]+?\.(?:exe|bat|cmd))"
)

_PROBE_VERSION_RE = re.compile(
    r"(?i)(?:git version|java version|openjdk|node v|Python \d|Apache Maven)"
)


def _blob_has_probe_evidence(blob: str) -> bool:
    return bool(_PROBE_EVIDENCE_RE.search(blob or ""))


def _blob_has_version_evidence(blob: str) -> bool:
    return bool(_PROBE_VERSION_RE.search(blob or ""))


def _run_shell_blobs(run: RemoteRun) -> str:
    parts: list[str] = []
    for step in run.steps:
        if step.action.type not in {"run_shell", "run_powershell"} or not step.result:
            continue
        r = step.result
        parts.extend(
            x for x in (r.stdout or "", r.stderr or "", r.content or "") if x
        )
    return "\n".join(parts)


def _run_has_probe_evidence(run: RemoteRun) -> bool:
    return _blob_has_probe_evidence(_run_shell_blobs(run))


def _run_has_version_evidence(run: RemoteRun) -> bool:
    return _blob_has_version_evidence(_run_shell_blobs(run))


def _is_version_probe_cmd(command: str | None) -> bool:
    return bool(re.search(r"(?i)(?:--version|-version|(?<![A-Za-z])-v)\b", command or ""))


def _is_path_lookup_cmd(command: str | None) -> bool:
    return bool(re.search(r"(?i)\bwhere(?:\.exe)?\b|\bget-command\b", command or ""))


def _count_same_shell_cmd(run: RemoteRun, command: str | None) -> int:
    """Count prior attempts of the exact command (including server rejects)."""
    norm = _norm_shell_cmd(command)
    if not norm:
        return 0
    n = 0
    for step in run.steps:
        if step.action.type not in {"run_shell", "run_powershell"}:
            continue
        if step.result is None:
            continue
        if _norm_shell_cmd(getattr(step.action, "command", None)) == norm:
            n += 1
    return n


def _probe_tool_name(command: str | None) -> str | None:
    """Best-effort tool name from where/version probes (java, git, …)."""
    c = command or ""
    m = re.search(r"(?i)\bwhere(?:\.exe)?\s+([A-Za-z][A-Za-z0-9_.-]*)", c)
    if m:
        name = m.group(1).lower()
        return "maven" if name == "mvn" else name
    m = re.search(
        r"(?i)\b(?:cmd\s+/c\s+[\"']?)?([A-Za-z][A-Za-z0-9_.-]*)\s+"
        r"(?:-version|--version|-v)\b",
        c,
    )
    if m:
        name = m.group(1).lower()
        if name in {"cmd", "powershell", "pwsh"}:
            return None
        return "maven" if name == "mvn" else name
    return None


def _count_tool_probes(run: RemoteRun, tool: str | None) -> int:
    """Count non-rejected where/version probes for a tool."""
    if not tool:
        return 0
    n = 0
    for step in run.steps:
        if step.action.type not in {"run_shell", "run_powershell"}:
            continue
        if step.result is None:
            continue
        meta = step.result.meta or {}
        if meta.get("rejected") or meta.get("server_guard"):
            continue
        if _probe_tool_name(getattr(step.action, "command", None)) == tool:
            n += 1
    return n


def evaluate_shell_repeat(run: RemoteRun, command: str | None) -> tuple[str, str]:
    """
    Stable anti-repeat policy for shell/PowerShell.

    Returns (decision, reason):
      - allow: execute
      - finish: enough install evidence — end run (caller may downgrade if
        a non-probe follow-up is already queued, e.g. git_pull)
      - reject: do not execute; redecide with a different command

    Hard rules (order matters):
      1) Exact same command never runs twice in one run.
      2) For a tool (java/git/…): at most one path lookup + one version probe.
      3) If path+version already exist, further tool probes → finish.
      4) Duplicate where when path already known → finish.
      5) Env-var / printenv style cmds: only exact-cmd rule (#1).
    """
    cmd = (command or "").strip()
    if not cmd:
        return "allow", ""

    same = _count_same_shell_cmd(run, cmd)
    has_path = _run_has_probe_evidence(run)
    has_ver = _run_has_version_evidence(run)
    tool = _probe_tool_name(cmd)
    same_tool = _count_tool_probes(run, tool) if tool else 0
    wants_ver = _is_version_probe_cmd(cmd)
    wants_path = _is_path_lookup_cmd(cmd)

    # Rule 1: exact duplicate.
    if same >= 1:
        if has_path:
            return (
                "finish",
                "相同命令已执行过，且观察中已有安装路径/版本证据，直接汇总结束。",
            )
        return (
            "reject",
            "相同 shell 命令已执行过。请换一条尚未执行的命令，或基于观察 done；禁止原样重试。",
        )

    # Rule 2–4: tool probe budget (where / -version only).
    if tool and (wants_path or wants_ver):
        if has_path and has_ver:
            return (
                "finish",
                "该工具路径与版本均已在观察中，禁止继续探测，直接汇总结束。",
            )
        if wants_path and has_path:
            return (
                "finish",
                "可执行路径已确认，禁止重复 where/Get-Command，直接汇总结束。",
            )
        if wants_ver and has_path and not has_ver and same_tool < 2:
            return "allow", ""
        if same_tool >= 2 or (same_tool >= 1 and has_path and not wants_ver):
            return (
                "finish",
                "该工具探测次数已用尽或信息已足够，直接汇总结束。",
            )

    return "allow", ""


def _queue_has_non_probe_followup(run: RemoteRun) -> bool:
    """True if queued actions still need to run after skipping a redundant probe."""
    skip = {"run_shell", "run_powershell", "done", "fail", "ask_user"}
    return any(a.type not in skip for a in run.action_queue)


def _soft_ok_probe_result(action: ClientAction, result: ActionResult) -> ActionResult:
    """If a probe already printed path/version, do not treat it as failure."""
    if action.type not in {"run_shell", "run_powershell"}:
        return result
    if result.ok:
        return result
    cmd = str(getattr(action, "command", None) or "")
    if not _is_env_probe_cmd(cmd):
        return result
    blob = "\n".join(
        x for x in (result.stdout or "", result.stderr or "", result.content or "") if x
    )
    if not _blob_has_probe_evidence(blob):
        return result
    result.ok = True
    result.error = None
    meta = dict(result.meta or {})
    meta["soft_ok_server"] = True
    result.meta = meta
    return result


def _guard_terminal_action(run: RemoteRun, action: ClientAction) -> ClientAction | None:
    """
    Return replacement action, or None to reject and re-plan, or action unchanged.
    Blocks premature fail/done so the loop keeps repairing toward success.
    """
    if action.type == "fail":
        if _may_terminal_fail(run):
            return action
        # Soft stop: ask user instead of ending as failed.
        note = (action.note or action.reply or "").strip()
        reply = (
            (action.reply or "").strip()
            or "当前步骤尚未成功。要继续自动修复，还是先停止？"
        )
        if note and note not in reply:
            reply = f"{reply}\n\n（上次判断：{note}）"
        return _mk_action(
            "ask_user",
            reply=reply,
            note="未达成功标准，需用户确认是否放弃",
            options=[
                {
                    "id": "continue_repair",
                    "title": "继续修复",
                    "summary": "根据错误日志继续尝试直到成功",
                },
                {
                    "id": "abandon",
                    "title": "停止任务",
                    "summary": "先停下来，稍后再说",
                },
            ],
        )
    if action.type == "done":
        last = _last_finished_step(run)
        if (
            last
            and last.action.type in _CRITICAL_TYPES
            and last.result
            and not last.result.ok
        ):
            return None  # reject — force re-decide with failure context
        return action
    return action


def _host_os_hint() -> str:
    """Tell the LLM which shell dialect to use on this machine."""
    sysname = platform.system()
    if sysname == "Windows":
        return (
            "本机 OS=Windows。请使用 PowerShell/cmd 命令："
            "where.exe <tool>；Get-ChildItem Env:JAVA_HOME,Env:PATH,Env:MAVEN_HOME；"
            "echo %JAVA_HOME%；java -version。"
            "禁止 printenv / which / uname / apt / brew 等非 Windows 命令。"
        )
    if sysname == "Darwin":
        return "本机 OS=macOS。可用 which / printenv / brew；注意权限。"
    return f"本机 OS={sysname or 'Linux'}。可用 which / printenv / apt 或发行版包管理器。"


async def _llm_plan(run: RemoteRun) -> dict[str, Any]:
    """First-turn task decomposition before any client action."""
    llm = get_llm_provider()
    user = (
        f"用户目标: {run.goal}\n"
        f"{_host_os_hint()}\n"
        f"权限: write={run.allow_write} install={run.allow_install} "
        f"process={run.allow_process} docker={run.allow_docker} sys={run.allow_sys} "
        f"shell={run.allow_shell} git={run.allow_git}\n"
        f"本地根目录提示: {run.local_root_hint or '(未提供)'}\n"
        "请输出任务分解 JSON。"
    )
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        data = _parse_json_obj(text)
        steps_raw = data.get("plan_steps") or data.get("plan") or []
        steps: list[str] = []
        if isinstance(steps_raw, list):
            steps = [str(s).strip() for s in steps_raw if str(s).strip()]
        elif isinstance(steps_raw, str) and steps_raw.strip():
            steps = [steps_raw.strip()]
        criteria = str(data.get("success_criteria") or "").strip()
        reply = str(data.get("reply") or "").strip()
        if steps:
            return {
                "plan_steps": steps[:10],
                "success_criteria": criteria
                or "关键动作成功且满足用户目标",
                "reply": reply or "已完成任务分解，开始执行。",
            }
    except Exception:
        logger.exception("LLM plan failed for run %s", run.id)

    return {
        "plan_steps": [
            "了解当前环境与目标相关现状",
            f"推进完成：{run.goal}",
            "验证是否达到成功标准；失败则诊断并修复后重试",
        ],
        "success_criteria": "用户目标完成，且最近关键动作成功",
        "reply": "我先按「了解 → 执行 → 验证/修复」来推进，直到成功或需要你确认。",
    }


def _extract_branch_name(message: str) -> str:
    """从用户原话提取分支名：分支名/分支叫/分支为 xxx，或「创建xxx分支」。

    例：「从main分支创建新分支 分支名 aaa_07」→ aaa_07；
    「帮我建一个 feature/login 分支」→ feature/login。
    提取失败返回空串。
    """
    text = (message or "").strip()
    if not text:
        return ""
    # 1) 分支名 xxx / 分支叫 xxx / 分支为 xxx / 分支名称 xxx
    m = re.search(
        r"(?is)(?:分支|branch)(?:名|名字|名称|叫|为|是)?[:：\s]*(?:\"|'|「|『)?"
        r"([A-Za-z0-9][A-Za-z0-9_\-\./]{0,79})",
        text,
    )
    if m:
        return m.group(1)
    # 2) 创建/新建 xxx 分支
    m = re.search(
        r"(?is)(?:创建|新建|建|切到|切至)(?:一个|新)?\s*"
        r"([A-Za-z0-9][A-Za-z0-9_\-\./]{0,79})\s*(?:分支|branch)",
        text,
    )
    if m:
        return m.group(1)
    return ""


async def route_client_need(message: str) -> dict[str, Any]:
    """LLM router: none | direct | code — intent only, no keyword lists."""
    from app.core.config import get_settings

    llm = get_llm_provider()
    text_msg = (message or "").strip()
    try:
        timeout = float(get_settings().llm_route_timeout or 30.0)
        # 路由只是 tiny 分类调用：不在慢端点上烧满 LLM_TIMEOUT/总预算。
        # 超时即 fail-closed none（安全方向），并记录告警便于排查
        # （desktop-chat-997ffe53135d：路由调用拖满 240s 总预算后回退 none）。
        try:
            text = await asyncio.wait_for(
                llm.chat(
                    [
                        {"role": "system", "content": ROUTE_PROMPT},
                        {"role": "user", "content": f"User message:\n{text_msg}"},
                    ],
                    temperature=0.1,
                    max_tokens=300,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "route_client_need LLM timeout after %.0fs — fail closed to none",
                timeout,
            )
            return {
                "mode": "none",
                "needs_client": False,
                "goal": text_msg or "",
                "reply_preview": "",
            }
        data = _parse_json_obj(text)
        mode = str(data.get("mode") or "none").strip().lower()
        if mode not in {"none", "direct", "code"}:
            mode = "direct" if data.get("needs_client") else "none"
        goal = str(data.get("goal") or text_msg or "").strip() or (text_msg or "")
        # Never keep the router prompt wrapper inside the run goal.
        if re.match(r"(?is)^\s*user\s*message\s*:", goal):
            goal = re.sub(r"(?is)^\s*user\s*message\s*:\s*", "", goal).strip() or text_msg
        # 分支名保真：从用户原话提取「分支名 xxx / 分支叫 xxx / 创建xxx分支」，
        # 拼进 goal，避免下游 agent 自创分支名（如 aaa_07 变成 feature/new-branch）。
        branch_name = _extract_branch_name(text_msg)
        if branch_name and re.search(r"(?i)分支|branch", text_msg):
            if "分支名" in goal or "分支叫" in goal or branch_name in goal:
                pass  # goal 已带分支名
            else:
                goal = f"{goal}（分支名必须是: {branch_name}，禁止自创其他分支名）"
        preview = str(data.get("reply_preview") or "").strip()
        if mode == "direct" and not preview:
            preview = "好的，我在本机执行相关操作。"
        return {
            "mode": mode,
            "needs_client": mode in {"direct", "code"},
            "goal": goal,
            "reply_preview": preview,
        }
    except Exception:
        logger.exception("route_client_need failed")
        # Fail closed to knowledge Q&A; do not keyword-guess local ops.
        return {
            "mode": "none",
            "needs_client": False,
            "goal": text_msg or "",
            "reply_preview": "",
        }


def _step_one_liner(step: RunStep) -> str:
    a = step.action
    r = step.result
    label = a.type or "?"
    detail = (
        getattr(a, "command", None)
        or getattr(a, "path", None)
        or getattr(a, "branch", None)
        or getattr(a, "ref", None)
        or ""
    )
    detail = _truncate(str(detail).strip(), 80)
    if not r:
        return f"- {label} {detail}（尚无结果）".strip()
    if r.ok:
        extra = ""
        if a.type == "git_status" and (r.stdout or "").strip():
            lines = [ln.strip() for ln in (r.stdout or "").splitlines() if ln.strip()][:4]
            if lines:
                extra = " → " + "；".join(lines)
        elif a.type in {"write_file", "apply_patch"} and a.path:
            extra = f" → 已写入 {a.path}"
        elif a.type == "git_checkout":
            br = getattr(a, "branch", None) or getattr(a, "ref", None) or ""
            create = bool(
                getattr(a, "create", None)
                or getattr(a, "new_branch", None)
                or getattr(a, "create_branch", None)
            )
            extra = f" → {'新建并切换' if create else '切换到'} {br}".rstrip()
        return f"- [成功] {label}" + (f" {detail}" if detail else "") + extra
    err_src = (r.error or "").strip()
    if not err_src and (r.stderr or "").strip():
        err_src = (r.stderr or "").strip().splitlines()[-1]
    err = _truncate(err_src or "失败", 120)
    return f"- [失败] {label}" + (f" {detail}" if detail else "") + f" — {err}"


def _goal_remaining_hints(run: RemoteRun) -> list[str]:
    """Soft hints about unfinished goal parts based on steps + goal text."""
    goal = (run.goal or "").lower()
    types_ok = {
        s.action.type
        for s in run.steps
        if s.result and s.result.ok
    }
    types_fail = {
        s.action.type
        for s in run.steps
        if s.result and not s.result.ok
    }
    hints: list[str] = []
    wants_branch = bool(
        re.search(r"分支|branch|feature/", run.goal or "", re.I)
    )
    wants_commit = bool(re.search(r"提交|commit", run.goal or "", re.I))
    wants_push = bool(re.search(r"推送|push|远程", run.goal or "", re.I))
    wants_compile = bool(
        re.search(r"编译|compile|自测|验证", run.goal or "", re.I)
    ) or any(s.action.type == "maven_compile" for s in run.steps)

    wrote = bool(types_ok & {"write_file", "apply_patch"})
    if wrote:
        hints.append("清单内文件似乎已写入本机")
    if wants_compile and "maven_compile" in types_fail:
        hints.append("编译未通过（常见原因：父 POM/私服依赖找不到）")
    if wants_branch and "git_checkout" not in types_ok:
        hints.append("尚未新建/切换到你要的功能分支")
    elif wants_branch:
        # checkout happened but maybe only switched to main
        for s in reversed(run.steps):
            if s.action.type == "git_checkout" and s.result and s.result.ok:
                br = str(
                    getattr(s.action, "branch", None)
                    or getattr(s.action, "ref", None)
                    or ""
                )
                create = bool(
                    getattr(s.action, "create", None)
                    or getattr(s.action, "new_branch", None)
                )
                if br in {"main", "master"} and not create:
                    hints.append(
                        f"当前只切到了 `{br}`，还没有按目标新建功能分支"
                    )
                break
    if wants_commit and "git_commit" not in types_ok:
        hints.append("还没有执行 git commit")
    if wants_push and "git_push" not in types_ok:
        hints.append("还没有执行 git push")
    if not hints and goal:
        hints.append("目标尚未全部完成，自动决策这一步没有给出可用动作")
    return hints


_READONLY_GOAL_RE = re.compile(
    r"(?is)(?:列出|查看|查询|获取|看看|看下|显示|展示|告诉我).{0,8}"
    r"(?:分支|状态|日志|版本|路径|文件|目录|列表|仓库|提交|commit|branch|status|log)"
)
_READONLY_ACTION_TYPES = frozenset(
    {"git_branch", "git_status", "git_log", "list_dir", "read_file", "open_file"}
)


def _readonly_goal_loop_hint(run: RemoteRun) -> bool:
    """目标为只读查询，且对应只读命令已成功执行 >= 3 次 → 死循环，强制收敛。"""
    if not _READONLY_GOAL_RE.search(run.goal or ""):
        return False
    ok_readonly = [
        s for s in run.steps
        if s.action.type in _READONLY_ACTION_TYPES and s.result and s.result.ok
    ]
    return len(ok_readonly) >= 3


def _readonly_goal_reply(run: RemoteRun) -> str:
    """把已观察到的只读结果拼成最终 reply（覆盖查询目标全部要点）。"""
    parts: list[str] = []
    for s in run.steps[-8:]:
        if s.action.type not in _READONLY_ACTION_TYPES or not s.result:
            continue
        out = (s.result.stdout or "").strip()
        err = (s.result.stderr or "").strip()
        if out:
            parts.append(out)
        elif err:
            parts.append(err)
    body = "\n".join(parts).strip() or "已完成查询（见上方执行记录）。"
    return f"已按要求完成查询：\n\n{body}"


def _user_facing_stuck_reply(run: RemoteRun) -> str:
    """Explain why we paused — plain language for non-technical users."""
    lines = [
        "本机任务暂时停住了，需要你确认下一步。",
        "",
        "【当前目标】",
        _truncate((run.goal or "").strip() or "（未写明）", 300),
        "",
        "【刚才实际做了什么】",
    ]
    recent = [s for s in run.steps[-8:] if s.result is not None]
    if not recent:
        lines.append("- 还没有任何本机动作结果")
    else:
        for s in recent:
            lines.append(_step_one_liner(s))

    remain = _goal_remaining_hints(run)
    if remain:
        lines.append("")
        lines.append("【对照目标，还缺什么】")
        for h in remain:
            lines.append(f"- {h}")

    if run.repair_count:
        lines.append("")
        lines.append(
            f"【修复尝试】已换方法重试 {run.repair_count}/{run.max_repairs} 次"
        )

    if run.last_decide_reasoning:
        lines.append("")
        lines.append("【系统内部判断】")
        lines.append(_truncate(run.last_decide_reasoning, 200))

    lines.extend(
        [
            "",
            "【请你直接回复下一步】例如：",
            "- 新建分支 feature/xxx，把改动 commit 并 push（先不要编译）",
            "- 跳过编译，只提交推送",
            "- 先停在这里，我自己处理",
        ]
    )
    return "\n".join(lines)


async def _llm_decide(run: RemoteRun) -> list[ClientAction]:
    llm = get_llm_provider()
    plan_block = ""
    if run.plan_steps or run.success_criteria:
        plan_block = (
            "任务计划:\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(run.plan_steps))
            + f"\n成功标准: {run.success_criteria or '(未写明，按用户目标判断)'}\n\n"
        )
    tried_block = _tried_methods_prompt(run)
    if run.repair_count > 0:
        phase_hint = (
            "当前处于修复阶段（换方法重试）：\n"
            "1. 必须先读 stderr/stdout 与「失败摘要」；\n"
            "2. 禁止对「已失败方法」列表中的 fingerprint 原样重试；\n"
            "3. 换方法优先级：修参数 → 换工具/命令 → 读日志再 patch → 仅信息不足时 ask_user；\n"
            f"4. 修复预算未用尽（{run.repair_count}/{run.max_repairs}）时禁止 fail；"
            "done 仅当成功标准已满足。"
        )
    else:
        phase_hint = (
            "按计划推进完整目标：对照计划步骤与观察，只做下一步；"
            "若目标含多个部分，前序成功后必须继续后续步骤，禁止提前 done；"
            "禁止重复执行已经跑过的相同命令；失败后必须换成另一条命令。"
            "若目标含新建分支/提交/推送：写文件只是第一步，应按 checkout(create)→commit→push 继续。"
            "只读查询类目标（列出分支/查看状态/查看日志/查看版本/查看路径等）："
            "只要对应只读命令已成功执行且结果已出现在「最近观察」中，目标即已完成，"
            "下一步必须直接 done（把结果写进 done.reply），禁止再重复执行同一条只读命令"
            "或反复 git_branch/git_status 空转。"
        )
    allow_block = ""
    if run.write_allowlist:
        allow_block = (
            "写文件白名单（禁止清单外 write/patch）：\n"
            + "\n".join(f"- {p}" for p in run.write_allowlist[:40])
            + "\n\n"
        )
    user = (
        f"目标: {run.goal}\n"
        f"{_host_os_hint()}\n"
        f"{plan_block}"
        f"{allow_block}"
        f"当前状态: {run.status}\n"
        f"已执行步数: {len(run.steps)}/{run.max_steps}\n"
        f"修复次数: {run.repair_count}/{run.max_repairs}\n"
        f"权限: write={run.allow_write} install={run.allow_install} "
        f"process={run.allow_process} docker={run.allow_docker} sys={run.allow_sys} "
        f"shell={run.allow_shell} git={run.allow_git}\n\n"
        f"最近观察:\n{_observation_blob(run) or '(尚无观察，请根据目标与计划选第一步)'}\n\n"
        + (f"{tried_block}\n\n" if tried_block else "")
        + f"{phase_hint}\n"
        "请根据目标语义与观察自行判断下一步（不要套固定工具脚本），输出 JSON。"
        "服务端不会替你判定业务成功；但会拒绝过早的 fail、原样重试，以及关键动作失败后的虚假 done。"
    )
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2500,
        )
        parsed = _parse_json_obj(text)
        reasoning = str((parsed or {}).get("reasoning") or "").strip()
        if reasoning:
            run.last_decide_reasoning = _truncate(reasoning, 240)
        actions = _filter_actions(run, _parse_llm_actions(text))
        if actions:
            # If goal already met, prefer finishing over leftover noop actions.
            if _git_commit_push_goal_met(run) and all(
                a.type not in {"done", "fail", "ask_user"} for a in actions
            ):
                non_git_noise = [
                    a
                    for a in actions
                    if a.type
                    not in {
                        "git_status",
                        "git_branch",
                        "git_checkout",
                        "run_shell",
                        "run_powershell",
                    }
                ]
                if not non_git_noise:
                    return [
                        _mk_action(
                            "done",
                            note="提交与推送已完成",
                            reply="代码已提交并推送到远程，任务完成。",
                        )
                    ]
            return actions
    except Exception:
        logger.exception("LLM decide failed for run %s", run.id)

    if _git_commit_push_goal_met(run):
        return [
            _mk_action(
                "done",
                note="提交与推送已完成",
                reply="代码已提交并推送到远程，任务完成。",
            )
        ]

    # Model returned nothing usable — explain status clearly, never invent a plan.
    reply = _user_facing_stuck_reply(run)
    return [
        _mk_action(
            "ask_user",
            reply=reply,
            note="等待用户补充（已说明当前进度）",
            options=[
                {
                    "id": "continue_git",
                    "title": "继续：建分支并提交推送",
                    "summary": "跳过编译，新建功能分支后 commit + push",
                },
                {
                    "id": "stop_here",
                    "title": "先停在这里",
                    "summary": "我自己处理剩余步骤",
                },
            ],
        )
    ]

async def _emit_status(run: RemoteRun, emit: EmitFn, message: str | None = None) -> None:
    if message is not None:
        run.message = message
    run.updated_at = time.time()
    await emit(
        {
            "type": "run_status",
            "run_id": run.id,
            "status": run.status,
            "message": run.message,
            "run": run.to_response().model_dump(),
        }
    )


async def _send_action(run: RemoteRun, action: ClientAction, emit: EmitFn) -> None:
    try:
        annotated = annotate_action_dict(action.model_dump())
        action = ClientAction.model_validate(annotated)
    except Exception:
        pass
    run.pending_action = action
    run.steps.append(RunStep(action=action, result=None, ts=time.time()))
    status: RunStatus = "waiting_confirm" if action.needs_confirm else "waiting_client"
    run.touch(status, f"等待客户端执行: {action.type}")
    payload = action.model_dump()
    await emit(
        {
            "type": "tool_event",
            "run_id": run.id,
            "event": "tool_call",
            "status": run.status,
            "message": run.message,
            "action": payload,
        }
    )
    await emit(
        {
            "type": "action",
            "run_id": run.id,
            "status": run.status,
            "message": run.message,
            "action": payload,
        }
    )


async def _finish(
    run: RemoteRun,
    emit: EmitFn,
    status: RunStatus,
    message: str,
    *,
    reply: str | None = None,
    options: list[dict[str, Any]] | None = None,
    phase: str | None = None,
) -> None:
    run.pending_action = None
    run.action_queue.clear()
    run.touch(status, message)
    run.final_options = list(options or [])
    if phase:
        run.final_phase = phase
    elif run.final_options:
        run.final_phase = "awaiting_cleanup"
    elif status == "succeeded":
        run.final_phase = "done"
    else:
        run.final_phase = "failed"

    final_reply = (reply or message or "").strip()
    # After tools ran: ensure a human conclusion (not just tool cards).
    if (
        status in {"succeeded", "failed"}
        and not str(run.final_phase or "").startswith("awaiting")
        and any(s.result is not None for s in run.steps)
    ):
        final_reply = await _enrich_final_reply(run, final_reply, status=status)

    run.final_reply = final_reply
    _persist_run_reply(run)

    await emit(
        {
            "type": "run_finished",
            "run_id": run.id,
            "status": status,
            "message": message,
            "reply": run.final_reply,
            "options": run.final_options,
            "phase": run.final_phase,
            "catalog": run.final_options,
            "run": run.to_response().model_dump(),
        }
    )


def _reply_needs_enrichment(reply: str) -> bool:
    text = (reply or "").strip()
    if not text:
        return True
    thin = (
        "见上方",
        "见执行结果",
        "见工具",
        "根据本机命令输出汇总",
        "根据命令输出",
        "任务完成",
        "本机任务完成",
        "已完成本机环境检查",
    )
    if any(m in text for m in thin):
        return True
    if len(text) < 28:
        return True
    return False


def _fallback_summary_from_steps(run: RemoteRun, status: RunStatus) -> str:
    """Deterministic short summary when LLM summarize is unavailable."""
    change_bullets: list[str] = []
    lines: list[str] = []
    for step in run.steps:
        r = step.result
        if not r:
            continue
        a = step.action
        if a.type in {"apply_patch", "write_file"}:
            meta = r.meta if isinstance(getattr(r, "meta", None), dict) else {}
            summary = meta.get("line_summary") if isinstance(meta, dict) else None
            path = a.path or ""
            if summary:
                mark = "[ok]" if r.ok else "[fail]"
                change_bullets.append(f"- {mark} {summary}")
            elif path:
                mark = "[ok]" if r.ok else "[fail]"
                change_bullets.append(
                    f"- {mark} `{path}`" + (f" — {r.error}" if (not r.ok and r.error) else "")
                )
    for step in run.steps[-6:]:
        r = step.result
        if not r:
            continue
        a = step.action
        if a.type in {"apply_patch", "write_file"}:
            continue
        blob = "\n".join(
            x for x in (r.stdout or "", r.stderr or "", r.content or "", r.error or "") if x
        ).strip()
        if not blob:
            continue
        snippet = _truncate(blob, 600)
        lines.append(f"【{a.type}】{'成功' if r.ok else '失败'}\n{snippet}")
    parts: list[str] = []
    if change_bullets:
        parts.append("### 修改文件与行号\n" + "\n".join(change_bullets))
    if lines:
        head = "本机检查结果如下：" if status == "succeeded" else "本机任务未完全成功，结果如下："
        parts.append(head + "\n\n" + "\n\n".join(lines))
    if not parts:
        return "本机任务已结束，但没有可展示的执行输出。"
    return "\n\n".join(parts)


async def _enrich_final_reply(run: RemoteRun, reply: str, *, status: RunStatus) -> str:
    change_block = ""
    change_bullets: list[str] = []
    for step in run.steps:
        r = step.result
        if not r:
            continue
        a = step.action
        if a.type not in {"apply_patch", "write_file"}:
            continue
        meta = r.meta if isinstance(getattr(r, "meta", None), dict) else {}
        summary = meta.get("line_summary") if isinstance(meta, dict) else None
        path = a.path or ""
        if summary:
            mark = "[ok]" if r.ok else "[fail]"
            change_bullets.append(f"- {mark} {summary}")
        elif path:
            mark = "[ok]" if r.ok else "[fail]"
            change_bullets.append(
                f"- {mark} `{path}`" + (f" — {r.error}" if (not r.ok and r.error) else "")
            )
    if change_bullets:
        change_block = "\n\n### 修改文件与行号\n" + "\n".join(change_bullets)

    def _with_changes(text: str) -> str:
        base = (text or "").rstrip()
        if change_block and "修改文件与行号" not in base:
            return base + change_block
        return base or change_block.lstrip()

    if not _reply_needs_enrichment(reply):
        return _with_changes(reply)
    obs = _observation_blob(run, limit=8_000)
    if not obs or obs.startswith("(尚无"):
        return _with_changes(reply or _fallback_summary_from_steps(run, status))
    llm = get_llm_provider()
    user = (
        f"用户目标: {run.goal}\n"
        f"运行状态: {status}\n"
        f"成功标准: {run.success_criteria or '(无)'}\n"
        f"模型草稿回复（可能太空，请重写）: {reply or '(空)'}\n\n"
        f"本机观察:\n{obs}\n\n"
        "请写出给用户的最终中文结论。"
        + (
            "\n若有代码改动，结论中保留「修改文件与行号」清单（路径+行号）。"
            if change_bullets
            else ""
        )
    )
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": SUMMARIZE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=800,
        )
        text = (text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:\w+)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text).strip()
        if text and not _reply_needs_enrichment(text):
            return _with_changes(text)
        if text and len(text) > len(reply or ""):
            return _with_changes(text)
    except Exception:
        logger.exception("enrich final reply failed for run %s", run.id)
    return _with_changes(reply or _fallback_summary_from_steps(run, status))


def _persist_run_reply(run: RemoteRun) -> None:
    """Write the post-tool AI conclusion into the chat session history."""
    if not run.session_id or not (run.final_reply or "").strip():
        return
    try:
        from app.agent import session_store

        session_store.append_message(
            run.session_id,
            "assistant",
            run.final_reply,
            meta={
                "intent": "client_agent",
                "phase": run.final_phase,
                "client_run_id": run.id,
                "options": list(run.final_options or []),
                "post_tool_summary": True,
            },
        )
        awaiting = str(run.final_phase or "").startswith("awaiting")
        session_store.update_context(
            run.session_id,
            {
                "phase": run.final_phase if awaiting else "idle",
                "client_run_id": run.id if awaiting else None,
                "pending_options": list(run.final_options or []) if awaiting else [],
            },
        )
    except Exception:
        logger.debug("persist run reply failed", exc_info=True)


def _display_fs_path(path: Any) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    if p.upper() in {"RECYCLE_BIN", "RECYCLEBIN"}:
        return "系统回收站"
    return p


def _options_from_scan(result: ActionResult | None) -> list[dict[str, Any]]:
    if not result:
        return []
    raw: dict[str, Any] = {}
    if result.content:
        try:
            raw = json.loads(result.content)
        except json.JSONDecodeError:
            raw = result.meta or {}
    else:
        raw = result.meta or {}
    cands = raw.get("cleanup_candidates") or []
    out = []
    for c in cands:
        if not isinstance(c, dict) or not c.get("id"):
            continue
        if not c.get("safe", True):
            continue
        path = c.get("path")
        path_disp = _display_fs_path(path)
        size = str(c.get("size_human") or "").strip()
        summary_bits = []
        if size:
            summary_bits.append(size)
        if path_disp:
            summary_bits.append(f"路径: {path_disp}")
        summary_bits.append("可安全清理")
        out.append(
            {
                "id": c["id"],
                "title": c.get("label") or c["id"],
                "summary": " · ".join(summary_bits),
                "path": path_disp or path,
                "size_human": c.get("size_human"),
                "safe": True,
                "kind": c.get("kind"),
            }
        )
    return out


def _enrich_options_with_scan_paths(
    options: list[dict[str, Any]],
    scan_opts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill missing path/summary from scan catalog so UI always shows targets."""
    by_id = {
        str(o.get("id")): o
        for o in scan_opts
        if isinstance(o, dict) and o.get("id")
    }
    out: list[dict[str, Any]] = []
    for opt in options:
        if not isinstance(opt, dict):
            continue
        item = dict(opt)
        sid = str(item.get("id") or "")
        src = by_id.get(sid) or {}
        if not item.get("path") and src.get("path"):
            item["path"] = src.get("path")
        path_disp = _display_fs_path(item.get("path"))
        if path_disp:
            item["path"] = path_disp
        if path_disp and "路径" not in str(item.get("summary") or ""):
            base = str(item.get("summary") or "").strip()
            item["summary"] = (
                f"{base} · 路径: {path_disp}".strip(" ·")
                if base
                else f"路径: {path_disp}"
            )
        if not item.get("title") and src.get("title"):
            item["title"] = src.get("title")
        if item.get("size_human") is None and src.get("size_human") is not None:
            item["size_human"] = src.get("size_human")
        out.append(item)
    return out


def ingest_log_chunk(run: RemoteRun, chunk: str, stream: str | None = None) -> None:
    prefix = f"[{stream}] " if stream else ""
    run._log_buf = _truncate(run._log_buf + prefix + (chunk or ""), 40_000)
    run.updated_at = time.time()


def submit_action_result(run: RemoteRun, result: ActionResult) -> None:
    result.stdout = _truncate(result.stdout, 24_000)
    result.stderr = _truncate(result.stderr, 24_000)
    if result.content is not None:
        result.content = _truncate(result.content, 80_000)
    for step in reversed(run.steps):
        if step.action.id == result.action_id or (
            run.pending_action and step.action.id == run.pending_action.id
        ):
            step.result = result
            break
    else:
        if run.pending_action:
            run.steps.append(
                RunStep(action=run.pending_action, result=result, ts=time.time())
            )
    run._last_result = result
    run.pending_action = None
    run._result_event.set()


async def run_agent_session(run: RemoteRun, emit: EmitFn) -> None:
    """Drive the loop until terminal status. Call after client_ready."""

    async def emit_logged(msg: dict[str, Any]) -> None:
        _append_session_client_log(run, msg)
        await emit(msg)

    async with run._lock:
        if run.status in TERMINAL_STATUSES:
            await _emit_status(run, emit_logged)
            return
        _append_session_client_log(
            run,
            {
                "type": "hello",
                "run_id": run.id,
                "status": run.status,
                "message": f"开始本机执行: {run.goal}",
            },
        )
        run.touch("planning", "开始规划")
        await _emit_status(run, emit_logged)

        while run.status not in TERMINAL_STATUSES:
            if len(run.steps) >= run.max_steps:
                await _finish(run, emit_logged, "failed", f"超过最大步数 {run.max_steps}")
                return
            # 只读查询死循环硬保险：目标只是查/列/看，且对应只读命令已成功
            # 执行 >= 3 次 → 结果已在观察中，直接 done，不再空转。
            if _readonly_goal_loop_hint(run):
                reply = _readonly_goal_reply(run)
                await _finish(run, emit_logged, "succeeded", reply)
                return
            if run.repair_count > run.max_repairs:
                if not run.repair_exhausted_asked:
                    run.repair_exhausted_asked = True
                    stuck = _user_facing_stuck_reply(run)
                    run.action_queue = [
                        _mk_action(
                            "ask_user",
                            reply=(
                                f"已尝试 {run.max_repairs} 次根据控制台失败换方法修复，仍未成功。\n"
                                "要继续自动尝试，还是先停止？\n\n"
                                + stuck
                            ),
                            note="修复预算用尽，征求用户是否继续",
                            options=[
                                {
                                    "id": "continue_repair",
                                    "title": "继续换方法",
                                    "summary": "再给几次自动修复机会",
                                },
                                {
                                    "id": "abandon",
                                    "title": "先停止",
                                    "summary": "结束本机任务",
                                },
                            ],
                        )
                    ]
                else:
                    await _finish(
                        run,
                        emit_logged,
                        "failed",
                        f"超过最大修复次数 {run.max_repairs}",
                    )
                    return

            # Explicit task decomposition before first decide (unless pre-seeded).
            if not run.plan_done:
                if run.action_queue:
                    run.plan_done = True
                    run.plan_text = "(预置动作队列)"
                    run.success_criteria = run.success_criteria or "预置动作全部成功完成"
                else:
                    run.touch("planning", "任务分解中…")
                    await _emit_status(run, emit_logged)
                    planned = await _llm_plan(run)
                    run.plan_done = True
                    run.plan_steps = list(planned.get("plan_steps") or [])
                    run.success_criteria = str(planned.get("success_criteria") or "")
                    run.plan_text = " → ".join(run.plan_steps)
                    reply = str(planned.get("reply") or "").strip()
                    run.message = reply or f"计划：{run.plan_text}"
                    await emit_logged(
                        {
                            "type": "tool_event",
                            "event": "task_plan",
                            "run_id": run.id,
                            "message": run.message,
                            "plan_steps": run.plan_steps,
                            "success_criteria": run.success_criteria,
                            "plan_text": run.plan_text,
                        }
                    )
                    await _emit_status(run, emit_logged, run.message)

            if not run.action_queue:
                run.touch(
                    "planning" if run.repair_count == 0 else "repairing",
                    (
                        f"进入修复 {run.repair_count}/{run.max_repairs}：根据控制台失败换方法…"
                        if run.repair_count > 0
                        else "LLM 决策中…"
                    ),
                )
                await _emit_status(run, emit_logged)
                run.action_queue = await _llm_decide(run)
                if run.repair_count > 0 and run.last_decide_reasoning:
                    short = _truncate(run.last_decide_reasoning, 120)
                    run.touch(
                        "repairing",
                        f"进入修复 {run.repair_count}/{run.max_repairs}：换方法 — {short}",
                    )
                    await _emit_status(run, emit_logged)

            if not run.action_queue:
                await _finish(run, emit_logged, "failed", "无可用动作")
                return

            action = run.action_queue.pop(0)

            # Stable shell anti-repeat (see evaluate_shell_repeat docstring).
            if action.type in {"run_shell", "run_powershell"}:
                cmd = str(getattr(action, "command", None) or "")
                decision, reason = evaluate_shell_repeat(run, cmd)
                if decision in {"finish", "reject"}:
                    # Preserve queued non-probe work (e.g. git_pull after check).
                    if _queue_has_non_probe_followup(run):
                        _inject_rejected_result(run, action, reason)
                        continue
                    if decision == "finish":
                        _inject_rejected_result(run, action, reason)
                        run.action_queue.clear()
                        await _finish(
                            run,
                            emit_logged,
                            "succeeded",
                            "探测信息已足够",
                            reply=await _enrich_final_reply(
                                run,
                                "本机检查已完成。",
                                status="succeeded",
                            ),
                        )
                        return
                    _inject_rejected_result(run, action, reason)
                    run.action_queue.clear()
                    await _emit_status(run, emit_logged)
                    continue

            # User denied a confirm — do not spam the same mutating action.
            if action.type not in {"ask_user", "done", "fail"}:
                denied_same = 0
                for step in run.steps:
                    if step.action.type != action.type or not step.result:
                        continue
                    err = str(step.result.error or "") + str(step.result.stderr or "")
                    if "user_denied" in err or "用户拒绝" in err:
                        denied_same += 1
                if denied_same >= 1:
                    _inject_rejected_result(
                        run,
                        action,
                        "用户已拒绝同类操作。请 ask_user 确认是否换方式继续，或 done/fail 结束；"
                        "禁止再次下发相同确认动作。",
                    )
                    run.action_queue.clear()
                    await _emit_status(run, emit_logged)
                    continue

            guarded = _guard_terminal_action(run, action)
            if guarded is None:
                _inject_rejected_result(
                    run,
                    action,
                    "拒绝提前结束：上一步关键动作失败，请根据失败摘要继续修复后再验证。",
                )
                run.action_queue.clear()
                await _emit_status(run, emit_logged)
                continue
            action = guarded

            if action.type == "ask_user":
                opts = list(action.options or [])
                scan_opts: list[dict[str, Any]] = []
                for step in reversed(run.steps):
                    if step.action.type == "sys_scan_heavy":
                        scan_opts = _options_from_scan(step.result)
                        break
                # Fill empty options from last scan; otherwise merge missing paths.
                if not opts:
                    opts = list(scan_opts)
                elif scan_opts:
                    opts = _enrich_options_with_scan_paths(
                        [o for o in opts if isinstance(o, dict)],
                        scan_opts,
                    )
                reply = (action.reply or action.note or "请确认下一步。").strip()
                # UI mode: concrete options → cleanup multi-select; abandon/continue stays general.
                phase = "awaiting_cleanup" if opts else "awaiting_option"
                opt_ids = {str(o.get("id") or "") for o in opts if isinstance(o, dict)}
                if opt_ids & {
                    "continue_repair",
                    "abandon",
                    "continue",
                    "stop",
                    "yes",
                    "no",
                    # 本机任务失败/中断后的通用引导选项：不是可清理项
                    "open_folder",
                    "continue_git",
                    "stop_here",
                } or any(
                    str(o.get("id") or "").startswith(("continue", "stop", "open"))
                    for o in opts if isinstance(o, dict)
                ):
                    phase = "awaiting_option"
                # Ensure cleanup reply lists concrete paths for non-technical users.
                if phase == "awaiting_cleanup" and opts and "路径" not in reply:
                    path_lines = []
                    for o in opts:
                        if not isinstance(o, dict):
                            continue
                        title = o.get("title") or o.get("id")
                        path = o.get("path")
                        if path:
                            path_lines.append(f"- {title}\n  路径: {path}")
                    if path_lines:
                        reply = (
                            reply.rstrip()
                            + "\n\n以下为可清理项及完整路径，请核对后再勾选：\n"
                            + "\n".join(path_lines)
                        )
                await _finish(
                    run,
                    emit_logged,
                    "succeeded",
                    action.note or "等待用户确认",
                    reply=reply,
                    options=opts,
                    phase=phase,
                )
                return

            if action.type == "done":
                await _finish(
                    run,
                    emit_logged,
                    "succeeded",
                    action.note or "完成",
                    reply=action.reply or action.note or "本机任务完成",
                    options=list(action.options or []),
                )
                return
            if action.type == "fail":
                await _finish(
                    run,
                    emit_logged,
                    "failed",
                    action.note or "失败",
                    reply=action.reply or action.note or "本机任务失败",
                )
                return

            if str(action.type or "").startswith("sys_"):
                action = _normalize_drive(action)

            if action.type == "sys_cleanup":
                err = _validate_sys_cleanup(run, action)
                if err:
                    _inject_rejected_result(run, action, err)
                    run.action_queue.clear()
                    await _emit_status(run, emit_logged)
                    continue

            if action.type == "sys_disk_overview" and not any(
                s.action.type == "sys_scan_heavy" for s in run.steps
            ):
                # Nudge queue: after overview, ensure scan is next if model forgot.
                # Only prepend if queue doesn't already have scan/ask/cleanup.
                upcoming = {a.type for a in run.action_queue}
                if "sys_scan_heavy" not in upcoming and "ask_user" not in upcoming:
                    run.action_queue.insert(
                        0,
                        _mk_action(
                            "sys_scan_heavy",
                            drive=action.drive or "C",
                            note="overview 之后必须扫描可清理项",
                        ),
                    )

            run._result_event.clear()
            await _send_action(run, action, emit_logged)

            try:
                await asyncio.wait_for(run._result_event.wait(), timeout=600.0)
            except asyncio.TimeoutError:
                await _finish(run, emit_logged, "failed", f"客户端执行超时: {action.type}")
                return

            if run.status == "stopped":
                await _finish(run, emit_logged, "stopped", "已停止", reply="已停止。")
                return

            result = run._last_result
            if result and run.pending_action is None:
                # Prefer the action we just ran (last step).
                last = run.steps[-1] if run.steps else None
                if last and last.result is result:
                    result = _soft_ok_probe_result(last.action, result)
                    last.result = result
                    run._last_result = result

            if result is not None:
                _append_session_client_log(
                    run,
                    {
                        "type": "tool_event",
                        "event": "tool_result",
                        "run_id": run.id,
                        "action": action.model_dump(),
                        "result": result.model_dump(),
                    },
                )

            run.touch("observing", f"已收到结果: {action.type}")
            await _emit_status(run, emit_logged)

            # Optional hint for LLM only — never auto ask_user / finish / smoke.
            if action.type == "sys_scan_heavy" and result and result.ok:
                opts = _options_from_scan(result)
                if opts:
                    try:
                        result.meta = dict(result.meta or {})
                        result.meta["cleanup_options_hint"] = opts[:12]
                    except Exception:
                        pass

            # Any failed repairable action clears the queue and enters repair.
            # Env probes that already yielded path/version evidence do not burn repair budget.
            if (
                result
                and not result.ok
                and action.type in _REPAIRABLE_TYPES
            ):
                cmd = str(getattr(action, "command", None) or "")
                blob = "\n".join(
                    x
                    for x in (
                        result.stdout or "",
                        result.stderr or "",
                        result.content or "",
                    )
                    if x
                )
                if _is_env_probe_cmd(cmd) and _blob_has_probe_evidence(blob):
                    run.action_queue.clear()
                else:
                    _record_tried_method(run, action, result)
                    run.repair_count += 1
                    run.action_queue.clear()
                    run.touch(
                        "repairing",
                        f"进入修复 {run.repair_count}/{run.max_repairs}：根据控制台失败换方法…",
                    )
                    await _emit_status(run, emit_logged)

            # Return to LLM decide with full observations (no hardcoded verdicts).
            continue
