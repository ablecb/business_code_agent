# 迁移方案：把编排层重写为 AgentScope 标准写法（学习用）

> 目的：把本项目的 Agent 编排层（自研状态机 + 手工 ReAct 循环 + LangGraph 薄封装）
> 重写为阿里 AgentScope 框架的标准写法，作为学习阿里云 ACP 认证课程的代码样板。
> 要求：**迁移后业务功能保持可用**（对话 / 检索 / 改码 / 日志诊断），按用户要求**暂不跑评测**。

## 1. 现状盘点（2025-06 核实）

| 模块 | 现状 | 对应 AgentScope 概念 |
|------|------|---------------------|
| `app/agent/graph_agent.py`（4241 行） | `AgentState` TypedDict + `_understand/_retrieve/_explore/_scope_plan/_patch_fixed/_verify/_reply` 节点函数 + `_dispatch_by_intent` 意图分发 | `Agent` 子类 + 消息驱动（`Msg`），或 `msghub` 多 Agent 协作 |
| `app/agent/explore_agent.py`（5762 行） | 自研 ReAct 循环：`_extract_tool_calls` → `dispatch_tool` → `_absorb_tool_result` → 评审循环 | `ReActAgent` + `@agentscope.tool` 工具系统（原生 ReAct 循环） |
| `app/agent/explore_tools.py`（1203 行） | `dispatch_tool` + `EXPLORE_TOOL_SPECS`（name/desc/args 描述表） | `@agentscope.tool` 装饰器注册（自动生成工具描述） |
| `app/agent/runtime/registry.py` | `TOOL_CATALOG` 插件式工具清单 + `list_tools` | 工具注册中心 |
| `app/agent/general_chat.py` | 无工具直答（LLM 一次调用） | 最简 `Agent._reply` 一次性回答 |
| `app/llm/factory.py` 等 | 自研 `LLMProvider`（OpenAI 兼容 / SiliconFlow / DeepSeek / Ollama / Mock） | AgentScope 模型层（OpenAI 兼容配置 `base_url/api_key/model`） |
| `app/agent/remote_loop.py`（2247 行） | 本机命令循环（client_apply / compile / git 操作） | 可封装为工具 + Agent 循环 |
| `app/agent/log_diagnosis.py` | 日志诊断管线（锚点→检索→证据→诊断） | 多 Agent 流水线 |

## 2. 目标架构（已按此落地 ✅）

```
app/agent/agentscope_bridge/          # 新增：AgentScope 迁移层（标准写法，已实现）
├── model.py        # 模型工厂：settings → OpenAIChatModel（OpenAI 兼容，stream=False）
│                   #   + get_text()：ChatResponse-dict / Msg 文本提取
├── tools.py        # 工具注册：EXPLORE_TOOL_SPECS → Toolkit.register_tool_function
│                   #   （显式 func_name + json_schema，preset_kwargs 内嵌 project_id）
├── log_diagnosis_scope.py  # 日志诊断：锚点→候选→证据（复用确定性函数）→
│                   #   AgentScope 模型诊断（Model + Msg + Formatter.format）
├── agents/
│   ├── general_agent.py   # AgentBase 直答（对应 general_chat）
│   ├── explore_agent.py   # ReActAgent：检索/读码/查图工具循环（对应 explore_agent）
│   ├── modify_agent.py    # ModifyAgent：ReAct 探索 + structured_model 结构化补丁
│   │                      #   → apply_patch_list 落地 workspace（对应 modify 链路）
│   └── msghub_team.py     # MsgHub 多智能体：探索 + 评审（对应 _finish_with_review）
└── adapter.py      # run_chat 兼容层：意图路由（client/meta/log_diagnose/modify/
                    #   general/code）+ 会话管理 + ChatResponse 组装
```

切换开关（已实现）：`agent_runtime_mode` 扩展为 `standard | minimal | agentscope`
（`.env` / 环境变量配置），`run_chat` 在 agentscope 模式时走 `adapter.py`；
`AGENTSCOPE_TEAM=msghub` 可把代码问答从单 ReActAgent 切到 MsgHub 多智能体协作。
`runtime/profile.py` 的 `RuntimeMode` 同步扩展（`get_runtime_profile().mode=agentscope`）。

## 3. 分阶段实施（附进度）

1. **阶段一（骨架 + 最小链路）**：安装 AgentScope → `model.py` → 最简 `general_agent`
   跑通 `/api/v1/chat` 无项目问答；注册 2~3 个核心只读工具（search_symbols /
   get_symbol_code / finish），用 `ReActAgent` 跑通"检索回答"链路。
   ✅ 已完成：agentscope 1.0.18 安装验证；model.py（OpenAIChatModel+硅基流动）；
   tools.py（Toolkit 注册全部 EXPLORE_TOOL_SPECS 工具，显式 json_schema）；
   general_agent.py（AgentBase 直答）；explore_agent.py（ReActAgent 检索循环）。
2. **阶段二（explore 多智能体）**：把 `run_explore_agent` 的完整工具循环迁到
   `ReActAgent`，评审循环迁到 `msghub`（explore + reviewer 协作），保 `on_progress`
   事件兼容。
   ✅ 已完成：msghub_team.py（MsgHub 探索+评审协作，REVISE→修订→APPROVED 循环
   实测跑通）；adapter 支持 `agentscope_team=msghub` 切换。
3. **阶段三（日志诊断 + 改码）**：`log_diagnosis` 迁到 AgentScope 模型版诊断管线
   （log_diagnosis_scope.py），改码链路迁到 ModifyAgent（ReAct 探索 + structured_model
   结构化补丁 → apply_patch_list 落地 workspace，失败反馈重试闭环）。✅ 已完成并实测。
   `remote_loop`（客户端命令循环）：adapter 中给出明确提示（未迁移，TODO 剩余项）。
4. **收尾**：`run_chat` 默认仍走原实现（保底），`agent_runtime_mode=agentscope` 走新实现；
   冒烟验证 /api/v1/chat 正常回复（不跑评测）。✅ 已通过服务级冒烟
   （通用问答/代码问答/meta 追问/日志诊断/改码落地，全部路由与回复正常）。

## 4. 保持不变的边界（兼容层红线）

- `run_chat(message, project_id, session_id, on_progress, ...)` 签名与返回 `ChatResponse` **不变**。
- `app/api/chat.py`、SSE `/chat/stream`、`session_store`、`on_progress` 事件协议 **不变**。
- `app/llm/`、`app/tagging/`、`app/graph/`、`app/db/` 等基础设施 **不动**（tools.py 直接调用）。
- 原 `graph_agent.py` / `explore_agent.py` **保留**，便于对照学习与随时回退。

## 5. 学习对照表（原实现 → AgentScope）

| 原实现 | AgentScope 写法 | 意义 |
|--------|----------------|------|
| 手写 messages 列表 + LLM.chat() | `Model` 对象 + `agent(msg)` | 模型即对象，`Msg` 是统一消息载体 |
| `_extract_tool_calls` 手写 JSON 解析 | `ReActAgent` + `@agentscope.tool` | 工具调用/解析/执行由框架完成 |
| `_dispatch_by_intent` if/else 链 | 多 Agent + `msghub` 路由 | 智能体间用消息协作，不写死分支 |
| 手写评审循环（reviewer 打分） | `msghub` 中 reviewer Agent 监听并反馈 | 观察者模式多智能体协作 |
| LangGraph StateGraph（仅 1 处，可选） | 原生消息流 | 无额外图依赖 |

## 6. 风险与对策

- **版本差异**：AgentScope 1.x 与 2.x API 有差异（2.0 是 breaking release：ReActAgent
  合并为统一 Agent、`__call__`→`reply_stream/reply`、ToolUseBlock→ToolCallBlock、
  hook→middleware、memory 模块废弃）。官方主站 doc.agentscope.io 默认 **v1.0 Stable**，
  先确认 ACP 课程用哪个版本（默认按 1.0 写，因为官方教程与示例都是 1.0 风格）。
- **行为差异**：ReActAgent 的循环策略与自研循环不同 → 保留原实现可回退，`on_progress`
  只映射关键事件（plan/step），不追求 1:1。
- **同步/异步**：AgentScope 支持同步为主；本项目是 async 管线 → adapter 里用
  `asyncio.to_thread` / run-in-thread 桥接，避免阻塞事件循环。

## 7. 已核实的 AgentScope 1.x 关键 API（2026-05 官方中文文档）

来源：doc.agentscope.io（v1.0 Stable 教程）

| 用途 | 写法 |
|------|------|
| 消息 | `from agentscope.message import Msg, TextBlock, ToolUseBlock, ToolResultBlock`<br>`Msg(name, content, role)`，role ∈ `system\|assistant\|user`，`metadata` 存结构化输出 |
| ReAct 智能体 | `from agentscope.agent import ReActAgent, AgentBase, UserAgent`<br>`ReActAgent(name=..., sys_prompt=..., model=..., formatter=..., toolkit=..., memory=..., max_iters=...)`；`await agent(msg)` |
| 自定义智能体 | 继承 `AgentBase`，实现 `reply(msg)->Msg`、`observe(msg)`、`handle_interrupt()`；`self.model(formatter.format(msgs))` |
| 工具 | `from agentscope.tool import Toolkit, ToolResponse`<br>`toolkit.register_tool_function(fn, preset_kwargs=...)`；函数返回 `ToolResponse(content=[TextBlock(...)])`，docstring 自动生成 JSON Schema |
| 模型（OpenAI 兼容） | `from agentscope.model import OpenAIChatModel`<br>`OpenAIChatModel(model_name=..., api_key=..., client_kwargs={"base_url": ...}, generate_kwargs={"temperature":..., "max_tokens":...})`；也可 `OllamaChatModel` / `DeepSeekChatModel` / `DashScopeChatModel` |
| 格式化器 | `from agentscope.formatter import OpenAIChatFormatter, OpenAIMultiAgentFormatter`（ReAct 工具序列转换） |
| 记忆 | `from agentscope.memory import InMemoryMemory` |
| msghub | `from agentscope.pipeline import MsgHub`<br>`async with MsgHub([alice, bob], announcement=Msg("system", ..., "system")): await alice()`（同 hub 的智能体自动互收 reply 消息） |
| 显式 Routing | `await router(msg, structured_model=RoutingChoice)` → 结构化输出存 `msg.metadata` |
| 隐式 Routing | 下游智能体包装成工具函数注册进 router 的 Toolkit，router 用工具调用转发 |
| 状态/会话管理 | `agentscope` 1.x 提供 `State` / `Session`（`task_state.html`），可挂接会话 |
| 安装 | Python ≥3.10，`pip install agentscope`（项目 conda 环境 Python 3.12 ✓） |

迁移骨架代码（第一阶段，ReAct 检索回答）：

```python
from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

toolkit = Toolkit()
toolkit.register_tool_function(search_symbols, preset_kwargs={"project_id": pid})
toolkit.register_tool_function(get_symbol_code, preset_kwargs={"project_id": pid})
toolkit.register_tool_function(finish)

agent = ReActAgent(
    name="ExploreAgent",
    sys_prompt=SYSTEM,                     # 现有 explore_system prompt + 工具说明
    model=OpenAIChatModel(
        model_name=settings.llm_model,
        api_key=settings.llm_api_key,
        client_kwargs={"base_url": settings.llm_base_url},
        generate_kwargs={"temperature": 0.2, "max_tokens": 4096},
    ),
    formatter=OpenAIChatFormatter(),
    toolkit=toolkit,
    memory=InMemoryMemory(),
    max_iters=12,                          # 对应 _DEFAULT_MAX_STEPS
)

reply = await agent(Msg("user", question, "user"))
answer = reply.get_text_content()          # 取最终文本回答
```

## 8. 实测踩坑笔记（1.0.18 真机验证，2026-01）

以下全部在本项目 conda 环境 + SiliconFlow（Qwen/Qwen3-32b）真机验证过：

| 坑 | 现象 | 结论/修法 |
|---|---|---|
| mcp 版本冲突 | `import agentscope` 报 `cannot import name 'streamablehttp_client'` | agentscope 1.x 依赖旧 mcp 命名，`pip install "mcp>=1.13,<2"`（1.29.1 ✓） |
| OpenAIChatModel 默认流式 | 响应是 async_generator，`.content` 属性报错 | 构造时显式 `stream=False` |
| ChatResponse 是 dict 子类 | 用 `response.get_text_content()` 报 KeyError | 取文本走 `response.get("content")`（block 列表）→ 各 block 的 `text` 字段；`Msg` 才有 `get_text_content()` |
| ChatResponse 重载 `__getattr__` | `getattr(res, "get_text_content", None)` 也触发 KeyError | 先 `isinstance(res, dict)` 分支再 `getattr`（见 `model.get_text`） |
| formatter 不可直接调用 | `OpenAIChatFormatter()(msgs)` 报 TypeError（1.0.18 中无 `__call__`） | 用 `await formatter.format(msgs)`（1.x 该方法是 async） |
| model 也是 async | `model(messages)` 返回 coroutine | `await model(...)`（`tracing` 包装成 async_wrapper） |
| Toolkit 闭包工具 schema 全空 | `register_tool_function(闭包)` 只生成 1 个 `{"name":"_tool", properties:{}}` | 闭包无 docstring 时 docstring-parser 解析不出参数 → 改用显式 `func_name=` + `json_schema=_make_schema(spec)`（见 `tools.py`） |
| ReActAgent 只思考不输出 | 模型（Qwen3 思考型）收尾只产生 thinking 块，无 text → 返回空 | 系统提示强调「必须调用 finish 工具提交答案，不要只思考不输出」；仍可能发生则回退取 thinking JSON（见 `msghub_team.py`） |
| 多智能体 formatter 名称 | `OpenAIChatMultiAgentFormatter` 不存在 | 1.0.18 中的标准名是 `OpenAIMultiAgentFormatter`（还有 DashScopeMultiAgentFormatter 等按厂商分） |
| MsgHub 协作终态 | 评审 Agent 会因历史里有摘要而误 APPROVED | 评审提示词明确「仅在看到 finish 提交的答案后 APPROVED，否则 REVISE 要求调用 finish」（见 `msghub_team.py`） |
| 模型只思考不落地（改码场景） | Qwen3 思考型读完原文后只输出 thinking/计划，不发起写工具、不出文本 → 补丁 0 | 用 `structured_model`（1.x 官方 ReAct 收尾机制）：框架自动 `tool_choice="required"`，模型每轮必须调工具，最终必须调 `generate_response` 提交结构化补丁（见 `modify_agent.py`，实测落地成功） |
| 结构化输出位置 | `reply.metadata` 直接就是 structured_output 对象本身（PatchPlan 实例），不是嵌套 `{"structured_output": ...}` | 直接 `getattr(reply, "metadata", None)` 判 `isinstance(plan, PatchPlan)` / `isinstance(plan, dict)` |
| `memory.get_memory()` 是 async | 直接 `get_memory()` 返回 coroutine，遍历报 TypeError | `await agent.memory.get_memory()`（1.x 中 Memory 接口多为 async） |
| 控制台中文乱码 | PowerShell 下中文显示为 mojibake | 仅显示问题，内容为正确 UTF-8；重定向到文件后可正常查看 |