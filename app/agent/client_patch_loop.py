"""Client-side patch → compile-verify → repair loop (server orchestrates)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from app.agent.remote_protocol import (
    TERMINAL_STATUSES,
    ActionResult,
    ClientAction,
    RemoteRunResponse,
    RunStep,
    RunStatus,
)
from app.core.utils import new_id
from app.llm.factory import get_llm_provider
from app.agent.prompts import load_prompt

logger = logging.getLogger("app.agent.client_patch_loop")

EmitFn = Callable[[dict[str, Any]], Awaitable[None]]

REPAIR_SYSTEM = load_prompt("repair_system")
POST_COMPILE_SYSTEM = load_prompt("post_compile_system")
async def _llm_after_compile(run: "PatchVerifyRun", compile_log: str) -> list[ClientAction]:
    llm = get_llm_provider()
    user = (
        f"目标: {run.goal}\n项目: {run.project_id}\n\n"
        f"编译日志:\n{_truncate(compile_log, 10000)}\n"
    )
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": POST_COMPILE_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=1200,
        )
        acts = _parse_repair_actions(text)
        if acts:
            return acts
    except Exception:
        logger.exception("post-compile LLM failed")
    return [
        _mk(
            "fail",
            note="无法判定编译结果",
            reply="编译已执行，但未能从日志自动得出结论，请人工核对。",
        )
    ]


@dataclass
class PatchVerifyRun:
    id: str
    session_id: str | None
    project_id: str | None
    goal: str
    status: RunStatus = "idle"
    message: str | None = None
    max_repairs: int = 3
    repair_count: int = 0
    steps: list[RunStep] = field(default_factory=list)
    pending_action: ClientAction | None = None
    action_queue: list[ClientAction] = field(default_factory=list)
    initial_patches: list[dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _result_event: asyncio.Event = field(default_factory=asyncio.Event)
    _last_result: ActionResult | None = None

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
            step_count=len(self.steps),
            repair_count=self.repair_count,
            max_steps=50,
            max_repairs=self.max_repairs,
            message=self.message,
            steps=list(self.steps),
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


_runs: dict[str, PatchVerifyRun] = {}


def get_patch_run(run_id: str) -> PatchVerifyRun | None:
    return _runs.get(run_id)


def list_patch_runs(limit: int = 50) -> list[RemoteRunResponse]:
    items = sorted(_runs.values(), key=lambda r: r.created_at, reverse=True)
    return [r.to_response() for r in items[:limit]]


def _mk(atype: str, **kwargs: Any) -> ClientAction:
    return ClientAction(id=new_id("act_"), type=atype, **kwargs)  # type: ignore[arg-type]


def patches_to_actions(
    patches: list[dict[str, Any]],
    *,
    path_hints: list[str] | None = None,
    project_id: str | None = None,
) -> list[ClientAction]:
    from app.agent.patcher import normalize_patch_paths

    acts: list[ClientAction] = []
    for p in normalize_patch_paths(
        [x for x in (patches or []) if isinstance(x, dict)],
        hints=path_hints,
        project_id=project_id,
    ):
        path = p.get("file") or p.get("path")
        if not path:
            continue
        if p.get("new_file") or p.get("content"):
            acts.append(
                _mk(
                    "write_file",
                    path=str(path),
                    content=p.get("content") or "",
                    note="写入新文件/全文",
                )
            )
        else:
            reps = []
            for r in p.get("replacements") or []:
                reps.append(
                    {
                        "old": r.get("old") or r.get("old_string") or "",
                        "new": r.get("new") or r.get("new_string") or "",
                    }
                )
            acts.append(
                _mk(
                    "apply_patch",
                    path=str(path),
                    replacements=reps,
                    note="应用补丁",
                )
            )
    return acts


def create_patch_verify_run(
    *,
    patches: list[dict[str, Any]],
    session_id: str | None = None,
    project_id: str | None = None,
    goal: str = "在本机应用补丁并编译通过",
    max_repairs: int = 3,
) -> PatchVerifyRun:
    rid = new_id("pvr_")
    run = PatchVerifyRun(
        id=rid,
        session_id=session_id,
        project_id=project_id,
        goal=goal,
        max_repairs=max(0, min(int(max_repairs), 10)),
        initial_patches=list(patches or []),
        message="等待客户端连接并执行改码验证",
        status="idle",
    )
    run.action_queue = patches_to_actions(patches)
    run.action_queue.append(_mk("maven_compile", note="编译验证 mvn -DskipTests compile"))
    _runs[rid] = run
    return run


def stop_patch_run(run_id: str) -> PatchVerifyRun | None:
    run = _runs.get(run_id)
    if not run:
        return None
    run.action_queue.clear()
    run.pending_action = None
    run.touch("stopped", "用户停止")
    run._result_event.set()
    return run


def submit_patch_result(run: PatchVerifyRun, result: ActionResult) -> None:
    for step in reversed(run.steps):
        if step.action.id == result.action_id:
            step.result = result
            break
    else:
        if run.pending_action:
            run.steps.append(RunStep(action=run.pending_action, result=result, ts=time.time()))
    run._last_result = result
    run.pending_action = None
    run._result_event.set()


def _truncate(text: str | None, n: int = 8000) -> str:
    t = text or ""
    if len(t) <= n:
        return t
    return t[: n // 2] + "\n…\n" + t[-(n // 2) :]


def _parse_repair_actions(text: str) -> list[ClientAction]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = data.get("actions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[ClientAction] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("type")
        if t not in {"apply_patch", "write_file", "fail", "done", "maven_compile"}:
            continue
        try:
            payload = {**item, "id": item.get("id") or new_id("act_")}
            reps = []
            for r in payload.get("replacements") or []:
                if isinstance(r, dict):
                    reps.append(
                        {
                            "old": r.get("old") or r.get("old_string") or "",
                            "new": r.get("new") or r.get("new_string") or "",
                        }
                    )
            payload["replacements"] = reps
            out.append(ClientAction.model_validate(payload))
        except Exception:
            logger.exception("bad repair action")
    return out


async def _llm_repair(run: PatchVerifyRun, compile_log: str) -> list[ClientAction]:
    llm = get_llm_provider()
    user = (
        f"目标: {run.goal}\n项目: {run.project_id}\n"
        f"修复次数: {run.repair_count}/{run.max_repairs}\n\n"
        f"编译日志:\n{_truncate(compile_log, 10000)}\n"
    )
    try:
        text = await llm.chat(
            [
                {"role": "system", "content": REPAIR_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2500,
        )
        acts = _parse_repair_actions(text)
        if acts:
            return acts
    except Exception:
        logger.exception("repair LLM failed")
    return [_mk("fail", note="无法根据编译日志生成修复补丁")]


async def _emit(run: PatchVerifyRun, emit: EmitFn, msg: dict[str, Any]) -> None:
    await emit(msg)


def _finish_reply(status: RunStatus, message: str, *, repairs: int = 0, run: PatchVerifyRun | None = None) -> str:
    change_block = ""
    if run is not None:
        bullets: list[str] = []
        for step in run.steps:
            a = step.action
            r = step.result
            if not r or a.type not in {"apply_patch", "write_file"}:
                continue
            meta = r.meta if isinstance(getattr(r, "meta", None), dict) else {}
            summary = meta.get("line_summary") if isinstance(meta, dict) else None
            path = a.path or (meta.get("path") if isinstance(meta, dict) else None) or ""
            if summary:
                mark = "[ok]" if r.ok else "[fail]"
                bullets.append(f"- {mark} {summary}")
            elif path:
                mark = "[ok]" if r.ok else "[fail]"
                bullets.append(f"- {mark} `{path}`" + (f" — {r.error}" if not r.ok and r.error else ""))
        if bullets:
            change_block = "\n\n### 修改文件与行号\n" + "\n".join(bullets)
    if status == "succeeded":
        extra = f"（期间自动修复 {repairs} 次）" if repairs else ""
        return (
            f"本机自测通过：{message}{extra}\n"
            "已在本机应用补丁并完成 `mvn compile`。"
            f"{change_block}"
        )
    if status == "stopped":
        return f"本机改码验证已停止：{message}{change_block}"
    return (
        f"本机自测未通过：{message}\n可查看顶部状态或编译日志后继续让我修复。"
        f"{change_block}"
    )


async def _finish(run: PatchVerifyRun, emit: EmitFn, status: RunStatus, message: str) -> None:
    run.pending_action = None
    run.action_queue.clear()
    run.touch(status, message)
    await _emit(
        run,
        emit,
        {
            "type": "run_finished",
            "run_id": run.id,
            "status": status,
            "message": message,
            "reply": _finish_reply(status, message, repairs=run.repair_count, run=run),
            "phase": "done" if status == "succeeded" else "failed",
            "run": run.to_response().model_dump(),
        },
    )


async def run_patch_verify_session(run: PatchVerifyRun, emit: EmitFn) -> None:
    async with run._lock:
        if run.status in TERMINAL_STATUSES:
            return
        run.touch("planning", "开始本机改码验证")
        await _emit(
            run,
            emit,
            {
                "type": "run_status",
                "run_id": run.id,
                "status": run.status,
                "message": run.message,
                "run": run.to_response().model_dump(),
            },
        )

        while run.status not in TERMINAL_STATUSES:
            if not run.action_queue:
                await _finish(run, emit, "failed", "无待执行动作")
                return

            action = run.action_queue.pop(0)
            if action.type == "done":
                await _finish(run, emit, "succeeded", action.note or "验证通过")
                return
            if action.type == "fail":
                await _finish(run, emit, "failed", action.note or "验证失败")
                return

            # Map maven_install → client compile verify (executor treats as compile)
            run.pending_action = action
            run.steps.append(RunStep(action=action, result=None, ts=time.time()))
            run.touch("waiting_client", f"等待客户端: {action.type}")
            run._result_event.clear()
            await _emit(
                run,
                emit,
                {
                    "type": "action",
                    "run_id": run.id,
                    "status": run.status,
                    "message": run.message,
                    "action": action.model_dump(),
                },
            )

            try:
                await asyncio.wait_for(run._result_event.wait(), timeout=900.0)
            except asyncio.TimeoutError:
                await _finish(run, emit, "failed", f"客户端超时: {action.type}")
                return

            if run.status == "stopped":
                await _finish(run, emit, "stopped", "已停止")
                return

            result = run._last_result
            run.touch("observing", f"收到结果: {action.type}")
            await _emit(
                run,
                emit,
                {
                    "type": "run_status",
                    "run_id": run.id,
                    "status": run.status,
                    "message": run.message,
                    "run": run.to_response().model_dump(),
                },
            )

            if action.type in {"apply_patch", "write_file"}:
                if result and not result.ok:
                    run.repair_count += 1
                    if run.repair_count > run.max_repairs:
                        await _finish(run, emit, "failed", f"补丁应用失败: {result.error}")
                        return
                    # continue queue; may still compile
                continue

            if action.type == "maven_compile":
                log = ((result.stdout if result else "") + "\n" + (result.stderr if result else ""))
                # Always let LLM judge compile logs (do not hard-succeed on exit_code).
                run.touch("planning", "分析编译日志…")
                await _emit(
                    run,
                    emit,
                    {
                        "type": "run_status",
                        "run_id": run.id,
                        "status": run.status,
                        "message": run.message,
                        "run": run.to_response().model_dump(),
                    },
                )
                decided = await _llm_after_compile(run, log)
                run.action_queue = []
                for a in decided:
                    if a.type in {"fail", "done"}:
                        run.action_queue.append(a)
                        break
                    if a.type in {"apply_patch", "write_file", "maven_compile"}:
                        run.action_queue.append(a)
                if not run.action_queue:
                    # fallback only if model returned nothing parseable
                    if result and result.ok:
                        run.action_queue.append(
                            _mk("done", note="编译通过", reply="本机编译验证通过。")
                        )
                    else:
                        run.repair_count += 1
                        if run.repair_count > run.max_repairs:
                            await _finish(run, emit, "failed", "编译失败且超过修复次数")
                            return
                        repairs = await _llm_repair(run, log)
                        for a in repairs:
                            if a.type in {"fail", "done"}:
                                run.action_queue.append(a)
                                break
                            if a.type in {"apply_patch", "write_file"}:
                                run.action_queue.append(a)
                        if not run.action_queue:
                            run.action_queue.append(_mk("fail", note="未生成有效修复补丁"))
                        elif run.action_queue[-1].type not in {"fail", "done"}:
                            run.action_queue.append(
                                _mk("maven_compile", note="修复后重新编译验证")
                            )
                continue
