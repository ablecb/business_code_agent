"""HTTP + WebSocket for unified client agent runs."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from app.agent.remote_loop import (
    create_agent_run,
    create_cleanup_agent_run,
    create_patch_agent_run,
    create_run,
    get_run,
    list_runs,
    run_agent_session,
    stop_run,
    submit_action_result,
)
from app.agent.remote_protocol import ActionResult, RemoteRunCreate, RemoteRunResponse, WsClientMessage

router = APIRouter(tags=["agent-remote"])
logger = logging.getLogger("app.api.agent_remote")


class PatchVerifyCreate(BaseModel):
    patches: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str | None = None
    project_id: str | None = None
    goal: str = "在本机应用补丁并编译通过"
    max_repairs: int = 3


class LocalAnalyzeCreate(BaseModel):
    goal: str = "分析磁盘占用并给出清理建议"
    drive: str = "C"


class LocalCleanupCreate(BaseModel):
    goal: str = "按用户选择清理"
    option_ids: list[str] = Field(default_factory=list)
    catalog: list[dict[str, Any]] = Field(default_factory=list)


class AgentRunCreate(BaseModel):
    goal: str
    project_id: str | None = None
    session_id: str | None = None
    local_root_hint: str | None = None
    max_steps: int = 20
    max_repairs: int = 5


class StopResponse(BaseModel):
    ok: bool
    run: RemoteRunResponse | None = None


@router.post("/agent/runs", response_model=RemoteRunResponse)
async def api_create_agent_run(req: AgentRunCreate) -> RemoteRunResponse:
    run = create_agent_run(
        goal=req.goal,
        project_id=req.project_id,
        session_id=req.session_id,
        local_root_hint=req.local_root_hint,
        max_steps=req.max_steps,
        max_repairs=req.max_repairs,
    )
    return run.to_response()


@router.post("/agent/remote-runs", response_model=RemoteRunResponse)
async def api_create_remote_run_legacy(req: RemoteRunCreate) -> RemoteRunResponse:
    return create_run(req).to_response()


@router.get("/agent/runs", response_model=list[RemoteRunResponse])
async def api_list_runs(limit: int = Query(default=50, ge=1, le=200)) -> list[RemoteRunResponse]:
    return list_runs(limit=limit)


@router.get("/agent/runs/{run_id}", response_model=RemoteRunResponse)
async def api_get_run(run_id: str) -> RemoteRunResponse:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run 不存在")
    return run.to_response()


@router.post("/agent/runs/{run_id}/stop", response_model=StopResponse)
@router.post("/agent/patch-runs/{run_id}/stop", response_model=StopResponse)
async def api_stop_run(run_id: str) -> StopResponse:
    run = stop_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run 不存在")
    return StopResponse(ok=True, run=run.to_response())


@router.post("/agent/patch-runs", response_model=RemoteRunResponse)
async def api_create_patch_run(req: PatchVerifyCreate) -> RemoteRunResponse:
    if not req.patches:
        raise HTTPException(status_code=400, detail="patches 不能为空")
    run = create_patch_agent_run(
        patches=req.patches,
        session_id=req.session_id,
        project_id=req.project_id,
        goal=req.goal,
        max_repairs=req.max_repairs,
    )
    return run.to_response()


@router.get("/agent/patch-runs", response_model=list[RemoteRunResponse])
async def api_list_patch_runs(limit: int = Query(default=50, ge=1, le=200)) -> list[RemoteRunResponse]:
    return list_runs(limit=limit)


@router.get("/agent/patch-runs/{run_id}", response_model=RemoteRunResponse)
async def api_get_patch_run(run_id: str) -> RemoteRunResponse:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run 不存在")
    return run.to_response()


@router.post("/agent/local-runs", response_model=RemoteRunResponse)
async def api_create_local_analyze(req: LocalAnalyzeCreate) -> RemoteRunResponse:
    drive = (req.drive or "C")[:1].upper()
    run = create_agent_run(
        goal=req.goal or f"分析 {drive}: 盘占用并给出清理建议",
        allow_write=False,
        allow_install=False,
        allow_process=False,
        allow_docker=False,
        allow_sys=True,
    )
    return run.to_response()


@router.post("/agent/local-cleanup-runs", response_model=RemoteRunResponse)
async def api_create_local_cleanup(req: LocalCleanupCreate) -> RemoteRunResponse:
    if not req.option_ids:
        raise HTTPException(status_code=400, detail="请至少选择一项清理")
    run = create_cleanup_agent_run(
        goal=req.goal,
        option_ids=req.option_ids,
        catalog=req.catalog,
    )
    return run.to_response()


@router.get("/agent/local-runs/{run_id}", response_model=RemoteRunResponse)
async def api_get_local_run(run_id: str) -> RemoteRunResponse:
    run = get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="run 不存在")
    return run.to_response()


@router.websocket("/agent/ws")
async def agent_ws(websocket: WebSocket, run_id: str = Query(...)) -> None:
    run = get_run(run_id)
    if not run:
        await websocket.close(code=4404)
        return

    await websocket.accept()
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    session_task: asyncio.Task[None] | None = None

    async def emit(msg: dict[str, Any]) -> None:
        await queue.put(msg)

    async def sender() -> None:
        while True:
            msg = await queue.get()
            if msg is None:
                break
            try:
                await websocket.send_json(msg)
            except Exception:
                break

    send_task = asyncio.create_task(sender())

    try:
        await emit(
            {
                "type": "hello",
                "run_id": run.id,
                "status": run.status,
                "message": "已连接，请发送 client_ready",
                "run": run.to_response().model_dump(),
            }
        )
        while True:
            raw = await websocket.receive_json()
            try:
                msg = WsClientMessage.model_validate(raw)
            except Exception as e:
                await emit({"type": "error", "run_id": run.id, "message": f"非法消息: {e}"})
                continue

            if msg.type == "stop":
                stop_run(run.id)
                await emit(
                    {
                        "type": "run_finished",
                        "run_id": run.id,
                        "status": "stopped",
                        "message": "客户端请求停止",
                        "reply": "已停止。",
                        "options": [],
                        "run": run.to_response().model_dump(),
                    }
                )
                break

            if msg.type == "log_chunk":
                continue

            if msg.type == "action_result" and msg.result is not None:
                result = msg.result
                if isinstance(result, dict):
                    result = ActionResult.model_validate(result)
                submit_action_result(run, result)
                continue

            if msg.type == "client_ready":
                if session_task is None or session_task.done():
                    session_task = asyncio.create_task(run_agent_session(run, emit))
                continue
    except WebSocketDisconnect:
        logger.info("agent ws disconnected run_id=%s", run_id)
        if run.status not in {"succeeded", "failed", "stopped"}:
            stop_run(run.id)
    except Exception:
        logger.exception("agent ws error run_id=%s", run_id)
    finally:
        if session_task and not session_task.done():
            session_task.cancel()
            try:
                await session_task
            except Exception:
                pass
        await queue.put(None)
        try:
            await send_task
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass
