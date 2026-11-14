"""AgentScope 版改码智能体（对应 graph_agent 的 modify 链路：探索→补丁→落地）。

学习对照（ACP 课程 3_1 工具调用 + 本项目改码链路）：
- 原实现：explore 检索 → scope 清单 → LLM 生成 patch（parse_patches_from_llm）→
  apply_patch_list 落地到 workspace（copy-on-write，不碰原代码）。
- 本迁移版：ModifyAgent = ReActAgent，工具集 = 只读探索工具 + 两个「写」工具
  （apply_replacements 原文替换 / write_new_file 新建文件，均走 patcher 的
  workspace copy-on-write 安全机制），由模型在 ReAct 循环里自主完成
  「定位→读码→替换→提交」，不再手写 scope/patch 状态机。

安全红线（与原有实现一致）：
- 写操作永远落在 .agent_workspace（agent_write_mode=workspace 默认值），
  原项目代码只读；agent_write_mode=inplace 时才直接写原文件（原语义保留）。
- 工具描述里明确：必须先用 read 工具确认原文再替换，禁止臆造代码。

返回形态兼容 run_explore_agent_scope：{answer, patches, steps, plan,
retrieved}，patches 从 ReAct 循环里 apply_replacements / write_new_file 的
工具结果中收集（真实落地记录，供 ChatResponse.patches 展示）。
"""
from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import get_settings
from app.core.utils import truncate

logger = logging.getLogger("app.agent.agentscope_bridge")

try:  # agentscope 可选依赖
    from agentscope.agent import ReActAgent
    from agentscope.tool import Toolkit
except Exception:  # pragma: no cover - depends on optional dep
    ReActAgent = object  # type: ignore[assignment,misc]
    Toolkit = None  # type: ignore[assignment,misc]

# 改码工具注册表（名称/描述/参数与 tools.py 的显式 schema 风格一致）。
_MODIFY_TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "read_workspace_file",
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": (
                "读取 .agent_workspace 中该文件的当前内容（若尚无 workspace 副本则读项目源码）。"
                "修改落地失败、需要核对当前内容时使用；返回内容截断到 4000 字符。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {
                        "type": "string",
                        "description": "项目内相对路径，如 loan-service/src/main/java/com/demo/loan/controller/XxxController.java",
                    },
                },
                "required": ["rel_path"],
            },
        },
    },
    {
        "name": "apply_replacements",
        "type": "function",
        "function": {
            "name": "apply_replacements",
            "description": (
                "把指定 Java 文件内的原文片段替换为新文本（写入 .agent_workspace "
                "副本，不修改原项目代码）。必须先 get_symbol_code 读到原文再调用，"
                "替换片段必须是文件里的真实原文（含缩进），禁止臆造。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {
                        "type": "string",
                        "description": "项目内相对路径，如 loan-service/src/main/java/com/demo/loan/controller/XxxController.java",
                    },
                    "replacements": {
                        "type": "string",
                        "description": "JSON 数组字符串，每个元素 {old: 原文片段, new: 新片段}，如 [{\"old\":\"int a = 1;\",\"new\":\"int a = 2;\"}]",
                    },
                },
                "required": ["rel_path", "replacements"],
            },
        },
    },
    {
        "name": "write_new_file",
        "type": "function",
        "function": {
            "name": "write_new_file",
            "description": (
                "新建一个文件并写入内容（放入 .agent_workspace 对应路径；"
                "若项目已有同名原文件则先拷贝原文件再整体覆写）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rel_path": {
                        "type": "string",
                        "description": "项目内相对路径（新文件），如 loan-service/src/main/java/com/demo/loan/controller/XxxController.java",
                    },
                    "content": {"type": "string", "description": "文件完整内容（UTF-8）"},
                },
                "required": ["rel_path", "content"],
            },
        },
    },
]

_MODIFY_EXPLORE_TOOLS = [
    "search_symbols",
    "get_symbol_code",
    "list_callees",
    "list_callers",
    "resolve_impl",
]

_MODIFY_SYSTEM_EXTRA = (
    "你是会真正动手改代码的智能体。工作流程：\n"
    "1) 先用搜索/读码工具定位要改的类与方法，读取原文，确认字段/逻辑现状；\n"
    "2) 读完原文后，**必须**调用 generate_response 提交结构化补丁（不要只描述计划，\n"
    "   不要用对话文本总结代替补丁）。补丁字段：\n"
    "   - patches[]: 每个补丁 {file: 项目内相对路径, old: 被替换的真实原文\n"
    "     （含缩进，须与读到的代码逐字一致）, new: 替换后的新文本}；\n"
    "     新建文件用 {file, new_file: true, content: 完整内容}；\n"
    "   - summary: 一句话说明改动意图与待办（编译/测试验证请在桌面端执行）。\n"
    "红线：old 必须是读到的真实原文；读不到原文就继续检索，不要臆造。\n"
)


def _write_tool_fn(name: str) -> Any:
    """生成执行写操作的闭包（名称/描述/schema 由注册时显式指定）。"""

    def _fn(**kwargs: Any) -> Any:
        from agentscope.message import TextBlock
        from agentscope.tool import ToolResponse

        from app.agent.patcher import apply_replacements, ensure_workspace_file, write_new_file

        rel_path = str(kwargs.get("rel_path") or "").strip()
        try:
            if name == "read_workspace_file":
                p = ensure_workspace_file(rel_path)
                text_content = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
                result = {
                    "file": rel_path,
                    "ok": bool(text_content),
                    "read_path": str(p),
                    "content": truncate(text_content, 4000),
                }
            elif name == "write_new_file":
                result = write_new_file(rel_path, str(kwargs.get("content") or ""))
            else:
                raw = kwargs.get("replacements")
                if isinstance(raw, str):
                    reps = json.loads(raw or "[]")
                else:
                    reps = list(raw or [])
                result = apply_replacements(rel_path, reps)
        except Exception as e:  # noqa: BLE001
            logger.exception("modify write tool failed")
            result = {"file": rel_path, "ok": False, "error": str(e)}
        text = truncate(json.dumps(result, ensure_ascii=False), 2000)
        return ToolResponse(content=[TextBlock(type="text", text=text)])

    return _fn


def _modify_system_prompt() -> str:
    """改码系统提示：探索主提示（去 JSON 协议）+ 改码流程说明 + 工具目录。"""
    from app.agent.agentscope_bridge.agents.explore_agent import _explore_system_prompt

    base = _explore_system_prompt()
    # 探索提示的收尾段（「信息足够时调用 finish」）仍合适，追加改码流程说明。
    return base + "\n\n" + _MODIFY_SYSTEM_EXTRA


def build_modify_agent(
    project_id: str | None = None,
    *,
    max_iters: int = 16,
    tool_names: list[str] | None = None,
) -> Any:
    """构建 ModifyAgent（ReActAgent）：探索只读工具 + workspace 写工具 + finish。"""
    from app.agent.agentscope_bridge.model import get_chat_model
    from app.agent.agentscope_bridge.tools import build_toolkit
    from agentscope.formatter import OpenAIChatFormatter
    from agentscope.memory import InMemoryMemory

    from app.core.config import get_settings

    pid = project_id or get_settings().project_id or "local"
    # 只读探索工具（finish 由 EXPLORE_TOOL_SPECS 注册）
    explore_names = tool_names or _MODIFY_EXPLORE_TOOLS + ["finish"]
    toolkit = build_toolkit(project_id=pid, tool_names=explore_names)
    # 追加写工具
    for spec in _MODIFY_TOOL_SPECS:
        toolkit.register_tool_function(
            _write_tool_fn(spec["name"]),
            func_name=spec["name"],
            json_schema=spec,
        )
    return ReActAgent(
        name="ModifyAgent",
        sys_prompt=_modify_system_prompt(),
        model=get_chat_model(),
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=max_iters,
    )


async def _land_patches(patch_items: list[Any]) -> list[dict[str, Any]]:
    """把结构化补丁项（dict / PatchItem）经 apply_patch_list 落地到 workspace。

    :param patch_items: 补丁项列表（元素为 dict 或 pydantic 模型）。
    :return: 落地结果列表 [{file, ok, applied, write_path, changes, error, ...}]。
    """
    if not patch_items:
        return []
    from app.agent.patcher import apply_patch_list

    mapped: list[dict[str, Any]] = []
    for item in patch_items:
        if isinstance(item, dict):
            if item.get("new_file"):
                mapped.append(
                    {"file": item["file"], "new_file": True, "content": item.get("content") or ""}
                )
            else:
                mapped.append(
                    {
                        "file": item["file"],
                        "replacements": [
                            {"old": item.get("old") or "", "new": item.get("new") or ""}
                        ],
                    }
                )
        else:  # PatchItem 实例
            if item.new_file:
                mapped.append({"file": item.file, "new_file": True, "content": item.content or ""})
            else:
                mapped.append(
                    {
                        "file": item.file,
                        "replacements": [{"old": item.old or "", "new": item.new or ""}],
                    }
                )
    results = apply_patch_list(mapped)
    return [
        {
            "file": r.get("file"),
            "ok": bool(r.get("ok")),
            "applied": r.get("applied"),
            "new_file": bool(r.get("new_file")),
            "write_path": r.get("write_path"),
            "changes": r.get("changes") or [],
            "error": r.get("error"),
        }
        for r in results
    ]


async def _collect_patches(agent: Any) -> list[dict[str, Any]]:
    """从 ReAct 循环的 memory 里收集写工具的真实结果（patches 展示用）。

    注意：1.x 的 `memory.get_memory()` 是 async（返回 coroutine），需 await。
    """
    patches: list[dict[str, Any]] = []
    try:
        msgs = await agent.memory.get_memory()
    except Exception:
        return patches
    for m in msgs:
        name = getattr(m, "name", "") or ""
        if name not in {"apply_replacements", "write_new_file"}:
            continue
        try:
            content = getattr(m, "content", None)
            if isinstance(content, list):
                text = ""
                for b in content:
                    if isinstance(b, dict) and b.get("text"):
                        text = str(b.get("text"))
                        break
            else:
                text = str(content or "")
            data = json.loads(text)
            if isinstance(data, dict):
                patches.append(
                    {
                        "file": data.get("file"),
                        "ok": bool(data.get("ok")),
                        "applied": data.get("applied"),
                        "new_file": bool(data.get("new_file")),
                        "write_path": data.get("write_path"),
                        "changes": data.get("changes") or [],
                        "error": data.get("error"),
                    }
                )
        except Exception:
            logger.debug("patch collect failed for %s", name, exc_info=True)
    return patches


async def run_modify_agent_scope(
    question: str,
    *,
    project_id: str | None = None,
    max_steps: int | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """AgentScope 版改码入口：{answer, patches, steps, plan} 兼容形态。

    流程（结构化输出，对应课程 3_1「结构化输出」/ ReActAgent 的
    `structured_model` 机制）：
    1. ModifyAgent（ReActAgent）探索：检索→读原文（工具循环）；
    2. 带 `structured_model=PatchPlan` 调用：框架强制 tool_choice=required，
       模型每轮必须调用工具（杜绝「只思考不落地」），最终必须调用
       generate_response 提交结构化补丁（PatchPlan）——这本身就是 1.x
       官方「ReAct 收尾」的标准做法，替代手工解析文本补丁；
    3. 结构化结果经 `apply_patch_list` 落地到 workspace
       （copy-on-write，安全红线与原有实现一致）。
    """
    from pydantic import BaseModel, Field

    from agentscope.message import Msg

    from app.agent.agentscope_bridge.model import get_text

    class PatchItem(BaseModel):
        file: str = Field(description="项目内相对路径，如 loan-service/src/main/java/.../XxxController.java")
        old: str | None = Field(default=None, description="被替换的真实原文（含缩进，须与读到的代码逐字一致）")
        new: str | None = Field(default=None, description="替换后的新文本（含缩进）")
        new_file: bool = Field(default=False, description="True=新建文件（此时用 content 字段）")
        content: str | None = Field(default=None, description="新建文件的完整内容（new_file=True 时必填）")

    class PatchPlan(BaseModel):
        summary: str = Field(description="一句话说明改动意图与待办")
        patches: list[PatchItem] = Field(default_factory=list, description="补丁列表")

    agent = build_modify_agent(project_id=project_id, max_iters=max_steps or 16)
    steps: list[dict[str, Any]] = []
    if on_progress:
        try:
            maybe = on_progress(
                {
                    "type": "step",
                    "status": "running",
                    "step_id": "modify_explore",
                    "note": "ModifyAgent 探索中（定位→读码→提交结构化补丁）…",
                }
            )
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            logger.debug("progress emit failed", exc_info=True)
    reply = await agent(Msg("user", question, "user"), structured_model=PatchPlan)
    answer_text = get_text(reply) or ""
    steps.append(
        {"step_id": "modify_explore", "note": truncate(answer_text or "（结构化补丁已提交）", 300), "status": "done"}
    )

    # —— 结构化补丁 → 确定性落地（失败反馈重试闭环，最多 3 轮）——
    # 1.x 中 reply.metadata 直接就是 structured_output 对象本身（PatchPlan
    # 实例 / 与之同构的 dict，见 ReActAgent.reply 退出分支），不是嵌套键。
    # 落地失败的常见原因：模型生成的 OLD 与文件当前内容有细微偏差（缩进/转义），
    # 与原 graph_agent 的 verify/repair 重试同理；这里把失败原因反馈给模型，
    # 并用 read_workspace_file 让模型核对 workspace 当前内容后重新提交补丁。
    patches: list[dict[str, Any]] = []
    plan = getattr(reply, "metadata", None)
    patch_items = []
    if isinstance(plan, PatchPlan):
        patch_items = plan.patches
    elif isinstance(plan, dict):
        patch_items = list(plan.get("patches") or [])
    patches = await _land_patches(patch_items)
    retry = 0
    max_retry = max(0, int(getattr(get_settings(), "agentscope_modify_retries", 2) or 0))
    while patches and any(not p.get("ok") for p in patches) and retry < max_retry:
        retry += 1
        failed = [p for p in patches if not p.get("ok")]
        feedback = (
            "以下补丁落地失败（OLD 片段与文件当前内容不匹配或文件不存在），"
            "请先用 read_workspace_file 读取对应文件确认当前内容，"
            "再重新调用 generate_response 提交修正后的补丁：\n"
            + "\n".join(
                f"- {p.get('file')}: {p.get('error') or 'no replacement matched'}" for p in failed
            )
        )
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "running",
                        "step_id": f"modify_retry_{retry}",
                        "note": f"反馈修订第 {retry} 轮（{len(failed)} 个补丁失败）…",
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("progress emit failed", exc_info=True)
        reply2 = await agent(Msg("user", feedback, "user"), structured_model=PatchPlan)
        plan2 = getattr(reply2, "metadata", None)
        items2 = []
        if isinstance(plan2, PatchPlan):
            items2 = plan2.patches
        elif isinstance(plan2, dict):
            items2 = list(plan2.get("patches") or [])
        patches = await _land_patches(items2)
        steps.append(
            {
                "step_id": f"modify_retry_{retry}",
                "note": truncate(get_text(reply2) or "（修订补丁已提交）", 300),
                "status": "done",
            }
        )
    if on_progress:
        try:
            maybe = on_progress(
                {
                    "type": "step",
                    "status": "done",
                    "step_id": "modify_landed",
                    "note": f"落地 {len(patches)} 个文件（workspace）",
                }
            )
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            pass
    return {
        "answer": answer_text or (plan.summary if isinstance(plan, PatchPlan) else "") or "（改码未产生最终说明）",
        "patches": patches,
        "steps": steps,
        "plan": {
            "kind": "modify",
            "title": "AgentScope ModifyAgent（ReAct 探索 + 结构化补丁落地）",
        },
        "retrieved": [],
    }