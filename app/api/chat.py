from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger("app.api.chat")

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.agent.chat_trace import ApiChatTurnTrace
from app.agent.graph_agent import run_chat
from app.agent.runtime.registry import list_tools
from app.agent.session_store import delete_session, list_feedback, list_messages, upsert_feedback
from app.agent.trajectory import read_trajectory, trajectory_path
from app.models.schemas import (
    ChatFeedbackItem,
    ChatFeedbackListResponse,
    ChatFeedbackRequest,
    ChatFeedbackResponse,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
)

router = APIRouter(tags=["chat"])


def _sse(event: dict[str, Any]) -> str:
    payload = json.dumps(event, ensure_ascii=False)
    # Comment padding so Electron / proxies flush each event instead of holding
    # the thinking stream until the turn finishes.
    return f"data: {payload}\n:{' ' * 1024}\n\n"


def _trace_extra(req: ChatRequest) -> dict[str, Any]:
    return {
        "project_ids": req.project_ids or None,
        "client_apply": req.client_apply,
        "local_assist": req.local_assist,
        "action": req.action,
        "option_id": req.option_id,
        "option_ids": req.option_ids or None,
        "drive": req.drive,
    }


async def _run_chat_with_trace(
    req: ChatRequest,
    *,
    on_progress: Any | None = None,
    source: str = "desktop-api",
) -> ChatResponse:
    """run_chat + same console/file log format as local_chat_interactive.LocalChatTester.

    Desktop: append turns into data/desktop_chat/<session_id>.log (one file per conversation).
    """
    trace = ApiChatTurnTrace(
        source=source,
        session_id=req.session_id,
        project_id=req.project_id,
    )
    trace.begin(
        req.message,
        project_id=req.project_id,
        session_id=req.session_id,
        client_agent=bool(req.client_agent or req.local_assist),
        extra=_trace_extra(req),
    )

    async def _progress(event: dict[str, Any]) -> None:
        await trace.on_progress(event)
        if on_progress:
            maybe = on_progress(event)
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]

    # 区分「未传 project_ids」（→ 回退默认项目，兼容旧客户端/API）与
    # 「显式传了空数组」（→ 未选择项目＝普通问答，桌面端勾选状态）。
    explicit_pids = "project_ids" in req.model_fields_set
    pids = req.project_ids if explicit_pids else None

    try:
        result = await run_chat(
            req.message,
            project_id=req.project_id,
            project_ids=pids,
            session_id=req.session_id,
            client_apply=req.client_apply,
            client_agent=req.client_agent,
            local_assist=req.local_assist,
            drive=req.drive,
            on_progress=_progress,
            action=req.action,
            option_id=req.option_id,
            option_ids=req.option_ids or None,
            cleanup_catalog=req.cleanup_catalog or None,
        )
        try:
            result.log_file = str(trace.logger.path)
        except Exception:
            pass
        trace.finish(result)
        return result
    except Exception as e:
        trace.fail(e)
        raise


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    return await _run_chat_with_trace(req, source="desktop-api/chat")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    """SSE stream: plan → step updates → final ChatResponse payload."""

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def on_progress(event: dict[str, Any]) -> None:
        await queue.put(event)

    async def runner() -> None:
        try:
            # Announce append log path as soon as the turn starts (desktop UI).
            from app.agent.chat_trace import get_session_chat_logger

            log = get_session_chat_logger(req.session_id, project_id=req.project_id)
            await queue.put(
                {
                    "type": "log",
                    "log_file": str(log.path),
                    "session_id": req.session_id,
                    "note": "对话日志追加写入（与 local_chat_interactive 同款格式）",
                }
            )
            result = await _run_chat_with_trace(
                req,
                on_progress=on_progress,
                source="desktop-api/chat/stream",
            )
            payload = result.model_dump() if hasattr(result, "model_dump") else dict(result)
            await queue.put({"type": "final", **payload})
        except Exception as e:
            await queue.put(
                {
                    "type": "error",
                    "message": f"{type(e).__name__}: {e}",
                }
            )
        finally:
            await queue.put(None)

    async def event_gen() -> AsyncIterator[str]:
        task = asyncio.create_task(runner())
        try:
            yield _sse({"type": "started"})
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=45.0)
                except asyncio.TimeoutError:
                    # Keep proxies / browsers from closing idle streams.
                    yield _sse({"type": "heartbeat"})
                    continue
                if event is None:
                    break
                yield _sse(event)
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/chat/feedback", response_model=ChatFeedbackResponse)
def chat_feedback(req: ChatFeedbackRequest) -> ChatFeedbackResponse:
    try:
        row = upsert_feedback(
            session_id=req.session_id,
            assistant_message_id=req.assistant_message_id,
            vote=req.vote,
            project_id=req.project_id,
            comment=req.comment,
        )
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    # 双赢：用户点「有用」→ 该 (问, 答) 对自动回填黄金问答库（source=manual_feedback），
    # 命中即免 8~16 次 LLM 长链；护栏/异常一律 best-effort，绝不阻断反馈本身。
    if req.vote == "up":
        try:
            from app.analysis.golden_qa import record_confirmed_qa

            msgs = list_messages(req.session_id)
            question = None
            answer = None
            for i, m in enumerate(msgs):
                if m.get("id") == req.assistant_message_id and m.get("role") == "assistant":
                    answer = m.get("content")
                    for j in range(i - 1, -1, -1):
                        if msgs[j].get("role") == "user":
                            question = msgs[j].get("content")
                            break
                    break
            if question is None:
                for m in reversed(msgs):
                    if m.get("role") == "user":
                        question = m.get("content")
                        break
            if question and answer:
                record_confirmed_qa(req.project_id, question, answer)
        except Exception:
            logger.debug("golden_qa backfill failed", exc_info=True)
    return ChatFeedbackResponse(
        id=row["id"],
        session_id=row["session_id"],
        assistant_message_id=row["assistant_message_id"],
        vote=row["vote"],
        created_at=row.get("created_at"),
        updated=bool(row.get("updated")),
    )


@router.get("/chat/feedback", response_model=ChatFeedbackListResponse)
def chat_feedback_list(
    session_id: str | None = None,
    project_id: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ChatFeedbackListResponse:
    items, total = list_feedback(
        session_id=session_id,
        project_id=project_id,
        limit=limit,
        offset=offset,
    )
    return ChatFeedbackListResponse(
        items=[ChatFeedbackItem(**x) for x in items],
        total=total,
    )


@router.get("/chat/{session_id}", response_model=ChatHistoryResponse)
def chat_history(session_id: str) -> ChatHistoryResponse:
    messages = list_messages(session_id)
    if not messages:
        raise HTTPException(status_code=404, detail="session not found or empty")
    return ChatHistoryResponse(session_id=session_id, messages=messages)


@router.get("/chat/{session_id}/trajectory")
def chat_trajectory(
    session_id: str,
    turn_id: str | None = None,
    limit: int = Query(default=500, ge=1, le=5000),
    offset: int = Query(default=0, ge=0),
) -> dict:
    """Append-only harness trajectory (JSONL) for eval / replay tooling."""
    rows = read_trajectory(session_id, turn_id=turn_id, limit=limit, offset=offset)
    path = trajectory_path(session_id)
    return {
        "session_id": session_id,
        "path": str(path) if path.exists() else None,
        "count": len(rows),
        "events": rows,
    }


@router.get("/agent/tools")
def agent_tools() -> dict:
    """List tools enabled for the active runtime mode (plugin-style catalog)."""
    from app.agent.runtime.profile import get_runtime_profile

    prof = get_runtime_profile()
    return {
        "runtime_mode": prof.mode,
        "tools": list_tools(profile=prof),
    }


@router.delete("/chat/{session_id}")
def chat_delete(session_id: str) -> dict:
    ok = delete_session(session_id)
    if not ok:
        # Still 200 for idempotent client cleanup of local-only drafts with server id stale
        return {"ok": True, "deleted": False, "session_id": session_id}
    return {"ok": True, "deleted": True, "session_id": session_id}
