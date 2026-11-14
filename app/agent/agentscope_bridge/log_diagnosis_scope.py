"""AgentScope 版日志驱动问题定位（对应 log_diagnosis.run_log_diagnosis）。

学习对照（ACP 课程 3_2「规划与执行」）：
- 原实现：确定性流水线（锚点提取 → 符号检索 → 证据收集），只有最后一步
  「LLM 诊断」调用自研 LLMProvider（messages 列表 + llm.chat）。
- 本迁移版：前 3 步的检索/取证逻辑是**业务确定性函数**（复用，不动），
  第 4 步改走 AgentScope 模型层：OpenAIChatModel + Msg + OpenAIChatFormatter
  显式 format（与 ReActAgent 内部同一套消息管线），返回 ChatResponse-dict，
  用 get_text() 取文本。这是「编排层使用框架、业务函数复用」的典型切分。

返回形态与 run_log_diagnosis 一致（{reply, anchors, candidates,
evidence_parts, steps, branch_hit}），adapter 无需感知差异。
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.core.utils import truncate

logger = logging.getLogger("app.agent.agentscope_bridge")


async def run_log_diagnosis_scope(
    message: str,
    *,
    project_id: str,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """AgentScope 版日志诊断：锚点 → 候选 → 证据 → AgentScope 模型诊断。

    :param message: 用户贴的日志/报错描述（原 run_log_diagnosis 同名参数）。
    :param project_id: 项目 id（检索与源码取证作用域）。
    :param on_progress: 进度事件（保持原 on_progress 协议）。
    :return: 与原 run_log_diagnosis 同形状的 dict。
    """
    from app.agent.explore_tools import get_symbol_code
    from app.agent.log_diagnosis import (
        _resolve_candidates,
        _grep_source_evidence,
        branch_extra_queries,
        extract_anchors,
        has_branch_signal,
    )
    from app.agent.prompts import load_prompt
    from app.agent.agentscope_bridge.model import get_chat_model, get_text

    async def _emit(event: dict[str, Any]) -> None:
        if not on_progress:
            return
        try:
            maybe = on_progress(event)
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            logger.debug("progress emit failed", exc_info=True)

    # ---- 1. anchors（确定性步骤，复用原实现）----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_anchors",
         "note": "日志锚点提取（类/方法/异常/开关/失败文案）…"}
    )
    anchors = extract_anchors(message)
    await _emit(
        {"type": "step", "status": "done", "step_id": "log_anchors",
         "note": f"锚点: 类{len(anchors['classes'])} 方法{len(anchors['methods'])} "
                 f"异常{len(anchors['exceptions'])} 开关{len(anchors['config_keys'])} "
                 f"失败文案{len(anchors['fail_texts'])}"}
    )

    # ---- 2. candidates（符号检索，复用原实现）----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_symbol_resolve",
         "note": "锚点→符号检索…"}
    )
    extra_queries = branch_extra_queries(message)
    candidates = _resolve_candidates(project_id, anchors, extra_queries=extra_queries)
    cm_labels = [
        f"{c['fqcn'] or ''}#{(c['method_sig'] or '').replace(c['fqcn'] or '', '')}"
        for c in candidates
    ]

    # ---- 3. evidence（源码取证，复用原实现）----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_evidence",
         "note": "读取候选符号源码 + 源码精确检索…"}
    )
    branch_flag = has_branch_signal(message)
    sym_parts: list[str] = []
    src_parts: list[str] = []
    parts: list[str] = []
    _seen_ev: set[str] = set()

    def _add_ev(target: list[str], part: str) -> None:
        key = part.split("\n", 1)[0][:100]
        if key in _seen_ev:
            return
        _seen_ev.add(key)
        target.append(part)

    for c in candidates[:3]:
        sig = str(c.get("method_sig") or "")
        meth = ""
        if "#" in sig:
            meth = sig.split("#", 1)[-1].split("(", 1)[0].strip()
        elif sig.split("(", 1)[0].split():
            meth = sig.split("(", 1)[0].split()[-1]
        code = None
        try:
            code = get_symbol_code(
                project_id,
                symbol_id=str(c.get("symbol_id") or ""),
                fqcn=str(c.get("fqcn") or ""),
                method_name=meth or None,
                max_chars=2000,
            )
        except Exception:
            logger.debug("get_symbol_code failed", exc_info=True)
        if code and code.get("ok"):
            fp = str(code.get("file_path") or "")
            sl = code.get("start_line")
            el = code.get("end_line")
            loc = f"{fp}:{sl}-{el}" if sl else fp
            _add_ev(
                sym_parts,
                f"### {code.get('fqcn') or c.get('fqcn')}#{meth or '?'}（{loc}）\n"
                f"{code.get('raw_code') or '(no raw_code)'}"
            )
    src_ev = _grep_source_evidence(project_id, anchors, extra_needles=extra_queries)

    for ev in sorted(src_ev, key=lambda e: (0, -int(e.get("line") or 0)))[:12]:
        _add_ev(
            src_parts,
            f"### 源码定位 {ev['file']}:{ev['line']}（命中: {ev['needle']}）\n{ev['region']}"
        )
    parts_raw = (src_parts + sym_parts) if branch_flag else (sym_parts + src_parts)
    budget = 10000
    used = 0
    for p in parts_raw:
        if used + len(p) > budget:
            break
        parts.append(p)
        used += len(p)
    evidence_block = "\n\n".join(parts) if parts else "（未检索到代码证据）"

    # ---- 4. LLM diagnosis（AgentScope 模型层：Model + Msg + Formatter）----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_diagnose",
         "note": "生成结构化诊断（现象→定位→根因→为什么→修复→证据链）…"}
    )
    log_excerpt = truncate(message, 7000)
    branch_hit = has_branch_signal(message, log_excerpt)
    system = load_prompt("log_diagnose_system")
    if branch_hit:
        system += (
            "\n\n# 补充要求（分支错位场景命中）\n"
            "本问题疑似「分支走错」（配置与实测分支不一致）。必须在「为什么」一节输出五段证据链：\n"
            "1) 配置项（键名/来源） 2) 读取点（类/方法读配置） 3) 兜底/默认值（空值默认哪个方向）\n"
            "4) 分支条件（if 判据原文） 5) 观测日志（哪一行证明实际分支）。\n"
            "重点核查：空值默认 Y / 开关取值语义（1 vs Y/N）/ 注释表名与注解表名不一致 / 查不到配置行静默回退。"
        )
    user_prompt = (
        "【用户描述/日志】\n"
        f"{log_excerpt}\n\n"
        "【已提取日志锚点】\n"
        f"类: {anchors['classes']}  方法: {anchors['methods']}\n"
        f"异常: {anchors['exceptions']}  配置开关: {anchors['config_keys']}  失败文案: {anchors['fail_texts']}\n\n"
        "【代码证据（只允许引用以下真实代码）】\n"
        f"{evidence_block}\n\n"
        "请按系统要求输出结构化诊断（现象/定位/根因/为什么/修复建议/证据链与置信度）。"
    )

    # AgentScope 写法：模型对象 + Msg 构造 + formatter 显式转换（与 ReActAgent 同管线）。
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.message import Msg

    model = get_chat_model()
    formatter = OpenAIChatFormatter()
    msgs = [
        Msg(name="system", content=system, role="system"),
        Msg(name="user", content=user_prompt, role="user"),
    ]
    try:
        # 1.x 的 formatter.format() 与 model.__call__() 均为 async（需 await）。
        formatted = await formatter.format(msgs)
        response = await asyncio.wait_for(
            model(formatted),
            timeout=300,
        )
        reply = get_text(response) or "**诊断未生成**"
    except (asyncio.TimeoutError, TimeoutError):
        reply = "**诊断超时**：模型响应超过 300s。可重试，或缩小日志/证据范围后再试。"
    except Exception as e:  # noqa: BLE001
        logger.exception("log diagnosis LLM failed (agentscope)")
        reply = f"**诊断失败**：模型调用异常（{type(e).__name__}）。请稍后重试。"

    steps = [
        {"step_id": "log_anchors", "note": f"锚点: {anchors}", "status": "done"},
        {"step_id": "log_symbol_resolve", "note": f"候选: {cm_labels}", "status": "done"},
        {"step_id": "log_evidence", "note": f"代码证据 {len(parts)} 段", "status": "done"},
        {"step_id": "log_diagnose", "note": "诊断完成" if not reply.startswith("**诊断") else reply,
         "status": "done"},
    ]
    return {
        "reply": reply,
        "anchors": anchors,
        "candidates": candidates,
        "evidence_parts": parts,
        "steps": steps,
        "branch_hit": branch_hit,
    }