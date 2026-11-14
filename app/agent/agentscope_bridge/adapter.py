"""AgentScope 版主对话入口：保持 `run_chat` 签名与 `ChatResponse` 契约不变。

学习对照（AgentScope 官方教程 + ACP 课程 3_x）：
- 本文件是「编排层」，负责会话管理、意图路由、最终回复组装；
- 真正的 Agent 能力在 agents/ 下：GeneralAgent（最简 AgentBase 直答）与
  ExploreAgent（ReActAgent 工具循环）；
- 与 graph_agent.py 的差异：那里用「状态字典 + 节点函数 + LLM 意图 JSON」，
  这里用「Agent 对象 + Msg 消息 + 工具调用」，意图路由简化为确定性规则
  （通用问题直答 / 代码问题进 ReAct），把复杂分支留给模型在工具循环里决策。

当前阶段（阶段一骨架）：实现 general 直答 + ReAct 检索问答；
改码（modify/apply/verify）等高级链路暂回退到原实现的提示说明（TODO 阶段二/三）。
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("app.agent.agentscope_bridge")


async def run_chat_scope(
    message: str,
    *,
    project_id: str | None = None,
    project_ids: list[str] | None = None,
    session_id: str | None = None,
    client_apply: bool = False,
    client_agent: bool = False,
    local_assist: bool = False,
    drive: str | None = None,
    on_progress: Any | None = None,
    action: str | None = None,
    option_id: str | None = None,
    option_ids: list[str] | None = None,
    cleanup_catalog: list[dict[str, Any]] | None = None,
) -> Any:
    """AgentScope 版 run_chat（签名与原 graph_agent.run_chat 保持一致）。"""
    from app.agent import session_store
    from app.agent.general_chat import is_general_chat_question
    from app.agent.runtime.profile import get_runtime_profile
    from app.models.schemas import ChatResponse

    t0 = time.perf_counter()
    settings = get_settings()

    # —— 会话与项目路由：与原实现相同的语义 ——
    ids = [p for p in (project_ids or []) if p]
    if project_id and project_id not in ids:
        ids.insert(0, project_id)
    explicit_selection = project_ids is not None
    if not ids and not explicit_selection:
        ids = [settings.project_id or "local"]
    project_id = ids[0] if ids else None
    sid = session_store.get_or_create_session(project_id, session_id, title=message[:40])
    session_store.append_message(sid, "user", message)

    if on_progress:
        try:
            maybe = on_progress({"type": "intent", "status": "running", "note": "AgentScope 路由…"})
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            logger.debug("progress emit failed", exc_info=True)

    elapsed = max(0, int((time.perf_counter() - t0) * 1000))

    # —— 未选择项目：普通问答（带历史）——
    if not ids:
        reply = await _general_reply_scope(message, session_id=sid)
        mid = session_store.append_message(
            sid, "assistant", reply, meta={"intent": "ask", "phase": "done", "elapsed_ms": elapsed}
        )
        session_store.update_context(sid, {"last_intent": "ask", "phase": "done", "last_explore_question": message})
        return ChatResponse(
            session_id=sid,
            reply=reply,
            intent="ask",
            phase="done",
            assistant_message_id=mid,
            project_ids=ids,
            elapsed_ms=elapsed,
            runtime_mode=get_runtime_profile().mode,
        )

    # —— 意图分诊（确定性规则；复杂分支留给 ReAct 循环）——
    # —— client_agent / remote_loop（客户端命令循环）暂未迁移 ——
    # 原 run_chat 在 client_agent=True 时走 remote_loop（本机编译/git/运行）。
    # AgentScope 模式下可把命令执行封装为工具（对应课程内置 execute_shell_command），
    # 本轮先给出明确提示，避免静默丢行为；TODO 阶段三剩余项。
    if client_agent or local_assist or client_apply:
        reply = (
            "当前为 AgentScope 编排模式：客户端命令循环（编译/运行/git）尚未迁移，"
            "请关闭 client_agent 后重试，或切换 AGENT_RUNTIME_MODE=standard 使用原实现。"
        )
        mid = session_store.append_message(
            sid, "assistant", reply, meta={"intent": "client", "phase": "idle", "elapsed_ms": elapsed}
        )
        session_store.update_context(sid, {"last_intent": "client", "phase": "idle"})
        return ChatResponse(
            session_id=sid,
            reply=reply,
            intent="client",
            phase="idle",
            assistant_message_id=mid,
            project_ids=ids,
            elapsed_ms=elapsed,
            runtime_mode=get_runtime_profile().mode,
        )

    act = (action or "").strip().lower()
    if act == "meta" or _wants_meta(message):
        reply = _meta_reply_scope(message, session_id=sid)
        mid = session_store.append_message(
            sid, "assistant", reply, meta={"intent": "meta", "phase": "idle", "elapsed_ms": elapsed}
        )
        session_store.update_context(sid, {"last_intent": "meta", "phase": "idle"})
        return ChatResponse(
            session_id=sid,
            reply=reply,
            intent="meta",
            phase="idle",
            assistant_message_id=mid,
            project_ids=ids,
            elapsed_ms=elapsed,
            runtime_mode=get_runtime_profile().mode,
        )

    # —— 意图分诊（确定性规则；复杂分支留给 ReAct 循环）——
    # 日志驱动问题定位（log_diagnose 意图）→ AgentScope 模型版诊断管线
    if _wants_log_diagnose(message):
        from app.agent.agentscope_bridge.log_diagnosis_scope import run_log_diagnosis_scope

        result = await run_log_diagnosis_scope(
            message, project_id=project_id, on_progress=on_progress
        )
        reply = str(result.get("reply") or "（诊断未生成）")
        intent = "log_diagnose"
        phase = "done"
        explore_steps_scope = list(result.get("steps") or [])
        patches_scope = []
    # 改码意图（"直接改代码/落地补丁/帮我改"等）→ ModifyAgent（ReAct + workspace 写工具）
    elif _wants_modify(message):
        from app.agent.agentscope_bridge.agents.modify_agent import run_modify_agent_scope

        result = await run_modify_agent_scope(
            message, project_id=project_id, on_progress=on_progress
        )
        reply = str(result.get("answer") or "（改码未产生最终说明）")
        intent = "modify"
        phase = "done"
        explore_steps_scope = list(result.get("steps") or [])
        patches_scope = [
            p for p in (result.get("patches") or []) if isinstance(p, dict) and p.get("file")
        ]
    # 通用闲聊/知识问答（无代码锚点）→ GeneralAgent 直答（不进工具循环）
    elif is_general_chat_question(message):
        reply = await _general_reply_scope(message, session_id=sid)
        intent = "ask"
        phase = "idle"
        explore_steps_scope = []
        patches_scope = []
    # 代码域问题 → ReActAgent 工具循环（检索/读码/查调用图）
    else:
        intent = "ask"
        phase = "done"
        explore_steps_scope = []
        patches_scope = []
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "running",
                        "step_id": "explore",
                        "note": "AgentScope ReActAgent 探索中…",
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("progress emit failed", exc_info=True)
        # AGENTSCOPE_TEAM=msghub 时用 MsgHub 多智能体协作（探索+评审），
        # 默认单 ReActAgent —— 两组都是 ACP 课程标准写法，可对照切换。
        team = str(get_settings().agentscope_team or "").strip().lower()
        if team == "msghub":
            from app.agent.agentscope_bridge.agents.msghub_team import run_explore_team

            result = await run_explore_team(
                message, project_id=project_id, on_progress=on_progress
            )
        else:
            from app.agent.agentscope_bridge.agents.explore_agent import (
                run_explore_agent_scope,
            )

            result = await run_explore_agent_scope(
                message, project_id=project_id, on_progress=on_progress
            )
        reply = str(result.get("answer") or "（未能生成回答）")
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "done",
                        "step_id": "explore",
                        "note": "AgentScope ReAct 探索完成",
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("progress emit failed", exc_info=True)

    mid = session_store.append_message(
        sid, "assistant", reply, meta={"intent": intent, "phase": phase, "elapsed_ms": elapsed}
    )
    session_store.update_context(
        sid, {"last_intent": intent, "phase": phase, "last_explore_question": message}
    )
    steps_out = [s for s in explore_steps_scope if isinstance(s, dict) and s.get("step_id")]
    return ChatResponse(
        session_id=sid,
        reply=reply,
        intent=intent,
        phase=phase,
        assistant_message_id=mid,
        project_ids=ids,
        elapsed_ms=elapsed,
        explore_steps=steps_out,
        patches=[p for p in patches_scope if isinstance(p, dict)],
        runtime_mode=get_runtime_profile().mode,
    )


async def _general_reply_scope(message: str, *, session_id: str) -> str:
    """GeneralAgent 直答（带会话历史）。"""
    from app.agent import session_store
    from app.agent.general_chat import format_history_text
    from app.agent.agentscope_bridge.agents.general_agent import general_chat_reply_scope

    history = session_store.list_messages(session_id)
    history_text = format_history_text(history)
    return await general_chat_reply_scope(message, history_text=history_text)


def _wants_log_diagnose(message: str) -> bool:
    """复用原 log_diagnosis 的意图闸门（日志/报错/异常/排查等信号词）。

    排除改码语境：「加日志/写日志/插入日志」是改码意图（modify），不是日志诊断。
    """
    from app.agent.log_diagnosis import wants_log_diagnose

    text = (message or "").strip()
    if any(k in text for k in ("加日志", "写日志", "插入日志", "加一行日志", "添加日志", "增加日志")):
        return False
    return wants_log_diagnose(text)


def _wants_modify(message: str) -> bool:
    """改码意图判定：复用原 intent_taxonomy 的启发式（改/实现/落地补丁等）。

    注意「先只给我清单/影响分析」属于 plan（analysis_only），不算改码。
    """
    from app.agent.intent_taxonomy import is_analysis_only_ask, normalize_intent

    if is_analysis_only_ask(message):
        return False
    try:
        return normalize_intent(None, message) == "modify"
    except Exception:
        return False


def _wants_meta(message: str) -> bool:
    """简化 meta 判定：问上一句/刚才问了什么/我刚才说什么。"""
    m = (message or "").strip()
    return any(k in m for k in ("我刚才", "刚才问", "上一句", "上一步", "之前问", "我问了什么", "刚才说了什么"))


def _meta_reply_scope(message: str, *, session_id: str) -> str:
    """找「当前这条之前的」最近一条 user 消息。

    注意：run_chat 入口已把当前 message 写入 session_store，
    所以要从倒数第二条 user 消息开始找（跳过当前这条）。
    """
    from app.agent import session_store

    msgs = session_store.list_messages(session_id)
    current = (message or "").strip()
    prev = None
    for m in reversed(list(msgs or [])):
        if m.get("role") != "user":
            continue
        content = str(m.get("content") or "").strip()
        if content == current and prev is None:
            # 第一条遇到的 user 消息就是当前这条：跳过，继续向前找
            continue
        prev = content
        break
    if prev and prev != current:
        return f"你刚才问的是：{prev}"
    return "这是会话的开始，还没有更早的问题。"