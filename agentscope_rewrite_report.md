# AgentScope 深度调研报告：把 FastAPI 自研 Agent 重写为 AgentScope 标准写法（ACP 认证课程代码样板）

> 调研对象：阿里云 ACP（智能体开发）认证课程仓库 `AlibabaCloudDocs/aliyun_acp_learning`、AgentScope 1.x / 2.x 官方源码与示例、官方文档站点。
> 调研方式：下载并通读了两份 AgentScope 发布版源码（`agentscope==1.0.18`、`agentscope==2.0.7.post1`，均从 GitHub tag 拉取）、ACP 课程仓库 main 分支全文、官方文档仓库 `agentscope-ai/docs` 的版本化文档（`versions/2.0.7/en|zh` 等），并核对 PyPI 版本与日期。**以下每条结论均指向可核查的源码文件或 URL（附录"证据与来源"）。**

---

## 0. TL;DR（结论速览）

| 问题 | 结论 |
|---|---|
| ACP 课程用哪个版本？ | **AgentScope 1.x**。课程 `requirements.txt` 锁定 `agentscope==1.0.18`（并注明 `mcp<2.0`、`mem0ai<2.0` 是 1.x 的依赖约束） |
| 课程教哪些概念？ | `ReActAgent`、`Toolkit` + `register_tool_function`、内置工具 `execute_python_code/execute_shell_command`、`Msg`、formatter（`DashScopeChatFormatter` / `DashScopeMultiAgentFormatter`）、`MsgHub`（多智能体协作）、memory（`InMemoryMemory`/`Mem0LongTermMemory`）、Skill、`PlanNotebook`、pipeline、`agentscope.init` 追踪。**不教** `@agent` 装饰器、不教 `AgentBase` 手写 `_reply` |
| 1.x 与 2.x 最大差异？ | 2.x 是 **breaking 重构**（2026-05-25 v2.0.0 发布）：`ReActAgent`→统一 `Agent`；`__call__`→`reply()/reply_stream()`；`MsgHub/msghub` **删除**；`ToolUseBlock`→`ToolCallBlock`；工具由 `Toolkit.register_tool_function`→`Toolkit(tools=[FunctionTool(...)])`；模型必须配 `Credential`；hooks→middleware；`agentscope.init` 移除 |
| 该装哪个版本对照学习？ | **做 ACP 课程与代码样板：装 1.x（推荐 `1.0.18`，或最新 `1.0.21`）**，课程代码可原样运行。2.x（现稳定版 `2.0.7.post1`）是当前主线，建议另起环境学习；**不要**用 2.x 跑课程 notebook |
| 2.x 是否稳定 / PyPI？ | 是。PyPI 包名不变 `agentscope`，Python ≥3.11；2.0.0（2026-05-25）→ 2.0.7.post1（2026-08-28），月级发版节奏，官方 change-log 明确 2.0 为 breaking release |

**对你的 FastAPI 项目（Business Code Agent，Spring Cloud/Maven Java 检索与改码助手，OpenAI 兼容 `base_url+api_key+model`，含 MySQL 符号表 / Milvus 向量 / Neo4j 调用图）的落地建议：**
1. 先按 **1.x** 把"自研编排循环"替换成 `ReActAgent`（框架自带 think–act–observe 循环、工具调用、错误重试），与 ACP 课程 3.1–3.6 完全同构 —— 这是本报告的推荐主样板；
2. 工具层把"查符号表 / 查调用图 / 向量检索 / 读文件 / 生成补丁"实现为普通 Python 函数后 `Toolkit.register_tool_function(...)` 注册；
3. FastAPI 路由直接 `await agent(msg)`，需要流式时用 `stream_printing_messages(...)`（1.x）或事件流 `agent.reply_stream(...)`（2.x）；
4. 若你后续想上 2.x：模型配置换成 `Credential + model=`，工具换成 `Toolkit(tools=[FunctionTool(...), Read(), Edit(), ...])`，多智能体协作从 `MsgHub` 迁移到事件流 / `GoalPipeline` / 官方 FastAPI Agent Service（2.x `agentscope.app.create_app` 本身就内置 FastAPI 后端 + Web UI，可替代你手写的大部分服务层）。

---

## 1. 调研目标 1：ACP 课程用的 AgentScope 版本与所教概念

### 1.1 版本：1.x，锁定 1.0.18

仓库 `requirements.txt`（main 分支，与 README 同级）：

```text
agentscope==1.0.18
mcp<2.0  # agentscope 1.x 依赖旧 streamablehttp_client 名称，mcp v2 已重命名
mem0ai<2.0  # agentscope 1.x 使用旧版 search() API，mem0 v2 改为 filters 传参
```

来源：
- 文件：`https://github.com/AlibabaCloudDocs/aliyun_acp_learning/blob/main/requirements.txt`
- 时间线佐证：`agentscope` PyPI 上 `1.0.18` 于 2026-03-26 发布，`2.0.0` 于 2026-05-25 才发布，而课程当前 main 分支仍锁 1.0.18 —— 说明**课程内容与 2.x 无关**。
- 仓库里的课程主体为 `大模型ACP认证教程/`（中文）与 `LLM_ACP_EN/`（英文同内容镜像），AgentScope 出现在 **C3“构建 Agent 系统”章（3_1~3_6）** 与英文版 `2_6_Building_an_Agent_to_Accomplish_Complex_Tasks.ipynb`。

### 1.2 课程教的 AgentScope 概念（按章节映射）

以下 import 全部逐条在课程 notebook 源码中核实：

| 课程章节 | 教的内容 | 课程源码里出现的 import（= 标准 1.x 写法） |
|---|---|---|
| 3_0 / 3_1 Agent 基础与工具调用 | ReAct 循环思想（think–act–observe）、从“手写工具路由/结构化输出”演进到框架用法、工具定义、MCP 工具接入 | `from agentscope.agent import ReActAgent`；`from agentscope.tool import Toolkit, ToolResponse`；`from agentscope.model import DashScopeChatModel`；`from agentscope.message import Msg, TextBlock`；`from agentscope.formatter import DashScopeChatFormatter`；`from agentscope.mcp import StdIOStatefulClient, HttpStatelessClient` |
| 3_2 规划与执行 | ReAct 多步执行、Python 代码执行工具、规划（PlanNotebook）、pipeline（sequential/fanout） | `from agentscope.plan import PlanNotebook`；`from agentscope.pipeline import sequential_pipeline, fanout_pipeline`；`from agentscope.tool import execute_python_code`；`from agentscope.agent import UserAgent` |
| 3_3 多 Agent 团队协作 | **MsgHub** 多智能体协作、MultiAgent formatter、用 memory 汇总会议记录 | `from agentscope.pipeline import MsgHub`；`from agentscope.formatter import DashScopeMultiAgentFormatter` |
| 3_4 Memory | 短期/长期记忆 | `from agentscope.memory import InMemoryMemory, Mem0LongTermMemory`；`from agentscope.embedding import DashScopeTextEmbedding`；`from agentscope.token import OpenAITokenCounter` |
| 3_5 Skill | 把能力固化为可复用流程（Skill 机制） | 复用 ReActAgent/Toolkit；扩展阅读指向 `doc.agentscope.io/tutorial/task_agent_skill.html` |
| 3_6 评测驱动开发 | `agentscope.evaluate`（Task、Metric、SolutionOutput）、`agentscope.init(studio_url=…, tracing_url=…)` 追踪 | `from agentscope.evaluate import Task, MetricBase/MetricResult/MetricType, SolutionOutput` |
| EN 2_6（英文镜像） | 同上（ReAct Agent、工具、MCP、多 Agent、memory、评测） | 同 3_1~3_6 |

**关于你问题里的几个猜测**：课程 **没有**出现 `@agent` 装饰器，**没有**让学员继承 `AgentBase` 重写 `_reply`，也没有出现“2.x 的 ToolUse 体系”——课程的教学模型就是：**实例化 `ReActAgent` + `Toolkit` 注册工具 + `Msg` 收发**，全程 async。课程用模型为百炼 DashScope（`DashScopeChatModel` + `DashScopeChatFormatter`），但**同样先教了 OpenAI 兼容协议**（`base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"` + openai SDK），这与"OpenAI 兼容 base_url+api_key+model"的诉求一致。

课程 3_1 的“最小 AgentScope 示例”（节选，原文出自 `大模型ACP认证教程/C3_构建Agent系统/3_1_Agent基础与工具调用.ipynb`）：

```python
import asyncio, os
from agentscope.agent import ReActAgent
from agentscope.tool import Toolkit, ToolResponse
from agentscope.model import DashScopeChatModel
from agentscope.message import Msg, TextBlock
from agentscope.formatter import DashScopeChatFormatter

def search_arxiv_paper(query: str) -> ToolResponse:
    """在 Arxiv.org 上搜索学术论文。
    Args:
        query (str): 搜索关键词
    """
    paper_url = "https://arxiv.org/abs/1706.03762"
    return ToolResponse(content=[TextBlock(type="text",
        text=f"已成功找到论文 '{query}'：{paper_url}")])

async def run_agentscope_example():
    toolkit = Toolkit()
    toolkit.register_tool_function(search_arxiv_paper)   # 普通函数即工具，docstring 生成 schema

    agent = ReActAgent(
        name="Course Research Agent",
        sys_prompt="你是一个课程研究助理，擅长帮人搜集和整理学习资料。",
        model=DashScopeChatModel(model_name="qwen-plus",
                                 api_key=os.environ.get("DASHSCOPE_API_KEY")),
        toolkit=toolkit,
        formatter=DashScopeChatFormatter(),
    )
    msg = Msg(name="user", content="帮我找那篇 Transformer 论文 'Attention Is All You Need'", role="user")
    await agent(msg)          # ReAct 循环在框架内自动完成

await run_agentscope_example()   # Jupyter 内直接 await；脚本里用 asyncio.run(...)
```

3_3 的多智能体协作（MsgHub 用法的课程原文摘录）：

```python
from agentscope.agent import ReActAgent
from agentscope.formatter import DashScopeMultiAgentFormatter
from agentscope.message import Msg
from agentscope.model import DashScopeChatModel
from agentscope.pipeline import MsgHub

def create_expert_agent(name: str, sys_prompt: str) -> ReActAgent:
    return ReActAgent(name=name, sys_prompt=sys_prompt,
                      model=DashScopeChatModel(model_name="qwen-plus",
                                               api_key=os.environ.get("DASHSCOPE_API_KEY")),
                      formatter=DashScopeMultiAgentFormatter())

async def main() -> None:
    designer_li = create_expert_agent("李老师 (教学设计师)", "...")
    scientist_wang = create_expert_agent("王工 (数据科学家)", "...")
    writer_zhang = create_expert_agent("小张 (内容编写者)", "...")

    async with MsgHub(
        participants=[designer_li, scientist_wang, writer_zhang],
        announcement=Msg("system", "…目标…", "system"),
    ) as hub:
        for i in range(2):
            await designer_li()      # hub 内发言会被路由给其他参与者
            await scientist_wang()
            await writer_zhang()

    # 会议秘书 Agent 读取 memory 汇总
    transcript = await designer_li.memory.get_memory()
    ...
    final_output_msg = await secretary_agent(Msg("user", prompt, "user"))
    print(final_output_msg.content)

await main()
```

---

## 2. 调研目标 2：1.x 与 2.x 各自的标准写法（官方示例 + 来源）

### 2.1 AgentScope 1.x（课程与官方 README/示例一致）

1.x 的文档站点是 **https://doc.agentscope.io/**（教程源码在 agentscope 仓库 `docs/tutorial/`，例如 `docs/tutorial/en/src/quickstart_agent.py`、`task_agent.py`、`task_tool.py`…），示例代码在仓库 `examples/`。注意：repo `main` 分支 2026-05-25 之后已是 2.x 内容，**看 1.x 官方示例请用 tag**：`https://github.com/agentscope-ai/agentscope/blob/v1.0.18/examples/...`。

（a）**跑一个开箱即用的 ReActAgent**（官方 `examples/agent/react_agent/main.py` @ v1.0.18 + README Quickstart）：

```python
import asyncio, os
from agentscope.agent import ReActAgent, UserAgent
from agentscope.formatter import DashScopeChatFormatter
from agentscope.memory import InMemoryMemory
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, execute_python_code, execute_shell_command, view_text_file

async def main() -> None:
    toolkit = Toolkit()
    toolkit.register_tool_function(execute_shell_command)   # 内置工具也是普通函数
    toolkit.register_tool_function(execute_python_code)
    toolkit.register_tool_function(view_text_file)

    agent = ReActAgent(
        name="Friday",
        sys_prompt="You are a helpful assistant named Friday.",
        model=DashScopeChatModel(api_key=os.environ.get("DASHSCOPE_API_KEY"),
                                 model_name="qwen-max", enable_thinking=False, stream=True),
        formatter=DashScopeChatFormatter(),   # 必须与 model 对应
        toolkit=toolkit,
        memory=InMemoryMemory(),
    )
    user = UserAgent("User")
    msg = None
    while True:                                  # 终端多轮对话
        msg = await user(msg)                    # UserAgent 从终端收集输入
        if msg.get_text_content() == "exit":
            break
        msg = await agent(msg)                   # await agent(msg) == 一次 reply
asyncio.run(main())
```

（b）**OpenAI 兼容 endpoint（base_url + api_key + model）**：1.x 的 `OpenAIChatModel` 没有单独的 `base_url` 参数，`base_url` 放在 `client_kwargs`（透传给 `openai.AsyncClient`）；生成参数放 `generate_kwargs`。对应 formatter 用 `OpenAIChatFormatter`（`from agentscope.formatter import OpenAIChatFormatter`）。源码：`agentscope/model/_openai_model.py`（v1.0.18）：

```python
from agentscope.model import OpenAIChatModel
from agentscope.formatter import OpenAIChatFormatter

model = OpenAIChatModel(
    model_name=os.environ["LLM_MODEL"],                      # 例如 "Qwen/Qwen3-32b"
    api_key=os.environ["LLM_API_KEY"],                       # 例如 SiliconFlow sk-xxx
    stream=True,
    client_kwargs={"base_url": os.environ["LLM_BASE_URL"]},  # 例如 https://api.siliconflow.cn/v1
    generate_kwargs={"temperature": 0.2},
)
formatter = OpenAIChatFormatter()                            # OpenAI chat-completions 协议
# 也可不传 formatter 直接给 OpenAIChatFormatter 于 ReActAgent(...)
```

（c）**如何定义工具（1.x）**：普通 Python 函数/async 函数，docstring + 类型注解即 schema 来源；注册进 `Toolkit`：

```python
from agentscope.tool import Toolkit

def search_symbol(symbol_name: str, project_id: str = "default") -> str:
    """在符号字典中检索 Java 类/方法。
    Args:
        symbol_name: 类名或方法名关键字
        project_id: 项目 ID
    """
    rows = ...  # 你的 MySQL 查询
    return "\n".join(f"{r['fqn']}:{r['summary']}" for r in rows)

toolkit = Toolkit()
toolkit.register_tool_function(search_symbol)      # 1.x 标准注册方式
# toolkit.register_mcp_client(mcp_client)          # MCP 客户端整体注册（1.0.18 支持）
# toolkit.register_agent_skill(skill_dir)          # Anthropic Agent Skill
```

返回类型可自由：`str`、`ToolResponse(content=[TextBlock(...)])`（课程常用、可携带多媒体）、任意 JSON 对象。

（d）**消息与打印/流式**：
- 构造消息：`Msg(name, content, role)`（或 `Msg(name="user", content="...", role="user")`），内容缺省是纯文本；多模态用 `TextBlock/ImageBlock/AudioBlock/VideoBlock/ToolUseBlock/ToolResultBlock`（`from agentscope.message import ...`）。
- ReActAgent 内置"控制台打印"（agent 默认把 thinking/工具过程打印到终端）；关闭它用 `agent.set_console_output_enabled(False)`（课程 3_5 / 官方 `stream_printing_messages` 示例都用到）。
- 程序内拿流式打印消息用官方 `stream_printing_messages`（`examples/functionality/stream_printing_messages/single_agent.py`）：

```python
from agentscope.pipeline import stream_printing_messages
from agentscope.message import Msg

agent.set_console_output_enabled(False)
async for msg, last in stream_printing_messages(agents=[agent],
                                                coroutine_task=agent(user_msg)):
    print(msg, last)
```

（e）**多智能体 / msghub（1.0.18 正式 API）**：1.0.18 把消息中心收敛到 `agentscope.pipeline.MsgHub`（源码 `agentscope/pipeline/_msghub.py`），配合 `sequential_pipeline / fanout_pipeline / ChatRoom`。官方 README（v1.0.18）示例：

```python
from agentscope.pipeline import MsgHub, sequential_pipeline
from agentscope.message import Msg

async def multi_agent_conversation():
    async with MsgHub(
        participants=[agent1, agent2, agent3],
        announcement=Msg("Host", "Introduce yourselves.", "assistant"),
    ) as hub:
        await sequential_pipeline([agent1, agent2, agent3])  # 轮流发言
        hub.add(agent4); hub.delete(agent3)                  # 动态管理参与者
        await hub.broadcast(Msg("Host", "Goodbye!", "assistant"))
asyncio.run(multi_agent_conversation())
```

（f）**RAG/记忆等**：1.x 内置 `agentscope.rag`、`InMemoryMemory`、`Mem0LongTermMemory`、`memory.get_memory()`（课程 3_3 用它取会议记录）；评测/追踪用 `agentscope.evaluate` 与 `agentscope.init(studio_url=…)`。

### 2.2 AgentScope 2.x（breaking 重构后的标准写法）

2.x 文档站点是 **https://docs.agentscope.io/**（版本化源码在 `agentscope-ai/docs` 仓库 `versions/2.0.7/en|zh/`，例如 `quickstart.mdx`、`building-blocks/agent/overview.mdx`、`building-blocks/tool/python-tool.mdx`、`building-blocks/model/llm.mdx`）。GitHub 示例在 `agentscope-ai/agentscope` 的 `examples/`（console、agent_service、pipeline、long_term_memory、rag…），README Quickstart 即 2.x 用法。

（a）**Agent 与 ReAct 循环**：官方 changelog 写明 “Refactor `ReActAgent` into a new unified `Agent` class implementation” —— **2.x 没有 ReActAgent 这个名字**，统一 `Agent` 内部自带 reasoning–acting 循环（`ReActConfig(max_iters=50…)` 控制），没有 `@agent` 装饰器，也不靠继承重写 `_reply` 来开发（`Agent._reply` 仍是内部 async 方法，但不建议覆写；扩展点是 middleware）。标准用法（官方 2.0.7 quickstart.mdx）：

```python
import asyncio, os
from agentscope.agent import Agent
from agentscope.credential import DashScopeCredential
from agentscope.event import EventType
from agentscope.message import UserMsg
from agentscope.model import DashScopeChatModel
from agentscope.tool import Toolkit, Bash, Read, Write, Edit

async def main() -> None:
    agent = Agent(
        name="Friday",
        system_prompt="You are a helpful assistant named Friday.",
        model=DashScopeChatModel(
            credential=DashScopeCredential(api_key=os.getenv("DASHSCOPE_API_KEY")),
            model="qwen-plus",
        ),
        toolkit=Toolkit(tools=[Bash(), Read(), Write(), Edit()]),
    )
    user_msg = UserMsg(name="user", content="Hello, who are you?")

    # 方式一：等最终消息（Agent 是"纯生产者"，不再自己往终端打印）
    reply_msg = await agent.reply(user_msg)          # -> AssistantMsg，content 是 block 列表

    # 方式二：事件流（文本增量/工具调用/thinking…）
    async for event in agent.reply_stream(user_msg):
        match event.type:
            case EventType.TEXT_BLOCK_DELTA:   # event.delta
                ...
            case EventType.TOOL_CALL_START:    # event.tool_call_name
                ...
            case _:
                ...

asyncio.run(main())
```

`Agent` 构造签名（源码 `agentscope/agent/_agent.py`，v2.0.7.post1）：

```python
Agent(
    name: str,
    system_prompt: str,
    model: ChatModelBase,
    toolkit: Toolkit | None = None,
    middlewares: list[MiddlewareBase] | None = None,   # hooks → middleware
    state: AgentState | None = None,
    offloader: Offloader | None = None,                # 上下文压缩/超长工具结果卸载
    model_config: ModelConfig | None = None,           # 含 fallback_model/max_retries
    context_config: ContextConfig | None = None,
    react_config: ReActConfig | None = None,           # max_iters 等
    injection_config: InjectionConfig | None = None,   # 时间/任务状态注入
)
```

公开方法：`await agent.reply(inputs, structured_schema=None) -> Msg`；`async for … in agent.reply_stream(inputs, structured_schema=None, yield_final_msg=False)`（产出 `AgentEvent` 或最终 `Msg`）；`await agent.observe(msgs)`（把别的 agent 的消息注入上下文，不触发回复）；`await agent.compress_context(...)`。

（b）**模型配置（OpenAI 兼容 base_url+api_key+model）**：2.x 把凭证从模型类中解耦为 `agentscope.credential`，`base_url` 是 `OpenAICredential` 的字段（“Can be used for OpenAI-compatible endpoints”），formatter 自动按 provider 选择（OpenAI 系默认 `OpenAIChatFormatter`），可用 `parameters=` 传 provider 参数（`OpenAIChatModel.Parameters`：`temperature/top_p/max_tokens/parallel_tool_calls…`）。官方 `building-blocks/model/llm.mdx` + 源码 `agentscope/model/_openai_chat/_model.py`：

```python
from agentscope.model import OpenAIChatModel
from agentscope.credential import OpenAICredential

model = OpenAIChatModel(
    credential=OpenAICredential(
        api_key=os.environ["LLM_API_KEY"],                 # 例如 SiliconFlow sk-xxx
        base_url=os.environ["LLM_BASE_URL"],               # 例如 https://api.siliconflow.cn/v1
    ),
    model=os.environ["LLM_MODEL"],                         # 例如 Qwen/Qwen3-32b
    stream=True,
    # parameters=OpenAIChatModel.Parameters(temperature=0.2),
)
```

其它：`DashScopeChatModel(credential=DashScopeCredential(api_key=…), model="qwen-plus")`、`DeepSeekChatModel/AnthropicChatModel/GeminiChatModel/OllamaChatModel/MoonshotChatModel/XAIChatModel/OpenAIResponseModel`（`agentscope.model`），每种都有 `Credential` 对应物（`agentscope.credential`）。

（c）**如何定义工具（2.x）**：官方文档 `building-blocks/tool/python-tool.mdx` 明确两条路：

1. 轻量：普通函数包 `FunctionTool`（自动从 `__name__`/docstring/类型注解提取 name/description/schema；返回值支持 `str`、JSON 对象、`ToolChunk`；需要 enum/范围/嵌套时传 Pydantic `input_schema`）：

```python
from agentscope.tool import FunctionTool, Toolkit

def get_weather(city: str, unit: str = "celsius") -> str:
    """Get the current weather for a city.
    Args:
        city: The city name to look up.
        unit: Temperature unit, either "celsius" or "fahrenheit".
    """
    return f"The weather in {city} is 22°{unit[0].upper()}"

toolkit = Toolkit(tools=[FunctionTool(get_weather)])
```

2. 进阶：继承 `ToolBase`（声明 `name/description/input_schema`，实现 `check_permissions()` 与 `async call(**kwargs)`），或继承 `ToolMiddlewareBase` 做洋葱式工具中间件。内置工具（实例化后放进 `tools=`）：`Bash / PowerShell / Read / Write / Edit / Glob / Grep / TaskCreate / TaskGet / TaskList / TaskUpdate`。`Toolkit(tools=[...], skills_or_loaders=[...], mcps=[...], tool_groups=[...])`；2.x MCP 收敛为单一 `MCPClient` + `StdioMCPConfig/HttpMCPConfig`（`agentscope.mcp`）。

（d）**消息（2.x）**：`Msg` 改为 pydantic 模型，`content` 是 **ContentBlock 列表**（文本增量/工具调用/思考块/多媒体统一用块表达），用工厂函数写消息最省事：

```python
from agentscope.message import UserMsg, AssistantMsg, SystemMsg, Msg
from agentscope.message import TextBlock, ThinkingBlock, HintBlock, ToolCallBlock, ToolResultBlock, DataBlock

user_msg = UserMsg(name="user", content="把 xxx 重构为 AgentScope 标准写法")   # str 自动包 TextBlock
```

块命名变化：1.x `ToolUseBlock/ToolResultBlock/ImageBlock/AudioBlock/VideoBlock` → 2.x `ToolCallBlock/ToolResultBlock/DataBlock(media_type=…)`；新增 `ThinkingBlock/HintBlock`；块与 Msg 都带 `id/created_at/…`（官方 change-log：Message 一节）。

（e）**事件系统 / 权限 / 中间件 / 控制台 / 服务（2.x 新增面）**：
- 事件：`agentscope.event` 定义 `EventType`（`REPLY_END / TEXT_BLOCK_DELTA / THINKING_BLOCK_DELTA / TOOL_CALL_START / TOOL_CALL_DELTA / TOOL_CALL_END / TOOL_RESULT_* / PERMISSION_REQUIRED …`），供流式 UI / HITL。
- 权限：`agentscope.permission`（`PermissionMode/ PermissionDecision/ PermissionBehavior`…）做工具与资源分级放行。
- 中间件：`agentscope.middleware.MiddlewareBase`（on_reply/on_reasoning/on_acting/on_model_call/on_check_permission/on_system_prompt/on_compress_context）。
- 控制台：`agentscope.console.launch_console(agent, verbosity=...)` 一行起终端交互。
- Agent Service：`agentscope.app.create_app(...)` —— **官方自带 FastAPI 后端 + Web UI**（`examples/agent_service`），支持多租户/会话/RAG/渠道；多智能体可上 "Agent Team"（leader–worker，靠内置 team 工具 + message bus）或 `agentscope.pipeline.GoalPipeline`（按固定逻辑串多个 agent 的事件流）。**2.x 里没有 msghub 的多智能体广播原语了**。

### 2.3 官方示例索引（含 URL，按你需要的主题）

| 主题 | 1.x（在 v1.0.18 tag 上） | 2.x（v2.0.7.post1 / docs 2.0.7） |
|---|---|---|
| 开箱即用 ReAct/Agent + 工具 | `examples/agent/react_agent/main.py`；README “Hello AgentScope!” | README Quickstart；docs `quickstart.mdx`；`examples/console/main.py` |
| 自定义 Python 工具 | 课程 3_1 + `docs/tutorial/en/src/task_tool.py`（Toolkit.register_tool_function） | docs `building-blocks/tool/python-tool.mdx`（FunctionTool / ToolBase）；`tests/toolkit_test.py` |
| 消息/流式打印 | `examples/functionality/stream_printing_messages/single_agent.py` | docs `building-blocks/agent/run-agent.mdx`（reply / reply_stream / observe） |
| 多智能体 | README MsgHub 段落；`examples/workflows/multiagent_*` | docs `deploy/agent-team.mdx`、`examples/pipeline/goal/goal_pipeline.py` |
| 模型配置 | docs `tutorial/en/src/task_model.py`；源码 `agentscope/model/_openai_model.py` | docs `building-blocks/model/llm.mdx`；`agentscope/model/_openai_chat/_model.py` |
| MCP | `examples/functionality/mcp/main.py`（`StdIOStatefulClient/HttpStatelessClient` + `register_mcp_client`） | docs `building-blocks/tool/mcp.mdx`（MCPClient + Stdio/HttpMCPConfig） |
| Skill | `examples/functionality/agent_skill/main.py`；doc.agentscope.io `task_agent_skill.html` | docs `building-blocks/tool/skill.mdx` |
| Memory | `examples/functionality/long_term_memory/*`；doc `task_memory.html` | docs `building-blocks/long-term-memory.mdx`（2.x memory 模块已弃用，走 middleware） |
| 评测/追踪 | doc `task_eval.html`；`agentscope.init(...)` | 2.x：`agentscope.evaluate` 迁移中；追踪用 `TracingMiddleware` |

---

## 3. 调研目标 3：最小可运行骨架（带工具调用的“检索/改码 Agent”）

> 场景与你的项目一致：OpenAI 兼容 endpoint（`base_url + api_key + model`，如 SiliconFlow/DeepSeek/任意兼容服务）；工具 = 代码库检索 + 读取 + 生成/落补丁；主流程 = ReAct 循环 + 消息收发（终端打印或后续接入 msghub / SSE）。
> 两个骨架都只依赖 agentscope 官方 API，可直接替换 `# 你的实现` 部分。

### 3.1 版本 A（推荐，1.x —— 与 ACP 课程代码逐行同构）

`pip install "agentscope==1.0.18"`（依赖约束同课程：`mcp<2.0`、`mem0ai<2.0` 如需这两项）。

```python
# code_refactor_agent_v1.py
# -*- coding: utf-8 -*-
"""检索/改码 ReActAgent（AgentScope 1.x 标准写法，对照 ACP C3 课程）。"""
import asyncio
import os

from agentscope.agent import ReActAgent
from agentscope.formatter import OpenAIChatFormatter   # OpenAI 兼容 endpoint
from agentscope.memory import InMemoryMemory
from agentscope.message import Msg, TextBlock
from agentscope.model import OpenAIChatModel
from agentscope.tool import Toolkit, ToolResponse

# ---------- 1) 工具：普通函数 + docstring，register_tool_function 自动生成 JSON Schema ----------

def search_symbols(symbol: str, project_id: str = "demo-loan") -> str:
    """在代码符号字典中搜索 Java 类/方法（MySQL）。
    Args:
        symbol: 类名/方法名关键字，如 'OrderService'
        project_id: 项目 ID
    """
    # TODO 你的实现：SQLAlchemy 查 mysql symbol table
    return "OrderService#createOrder(int,CreateOrderReq): 订单创建入口\nOrderService#cancelOrder(...)"

def search_callers(fqn: str, project_id: str = "demo-loan") -> str:
    """查询某个方法被谁调用（Neo4j 调用图，含异步链路标注）。
    Args:
        fqn: 方法全限定名，如 com.demo.order.service.OrderService#createOrder
        project_id: 项目 ID
    """
    # TODO 你的实现：Cypher 查 neo4j
    return "OrderController#create <- Feign OrderApi#createOrder <- Job:createOrderJob"

def search_similar_code(query: str, top_k: int = 5) -> str:
    """按语义相似度检索相似代码片段（Milvus 向量库）。
    Args:
        query: 描述或代码片段
        top_k: 返回条数
    """
    # TODO 你的实现：embedding + pymilvus 检索
    return "[score=0.92] OrderService.java#L120-L145 同类代码 …"

def read_file_range(file_path: str, start: int = 1, end: int = 80) -> ToolResponse:
    """读取源码文件的指定行区间，返回带行号文本。
    Args:
        file_path: 仓库内相对路径，如 src/main/java/com/demo/order/service/OrderService.java
        start: 起始行号（含）
        end: 结束行号（含）
    """
    try:  # TODO 你的实现：按 java_project 工作区/库表读取
        with open(file_path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = [f"# TODO 未找到 {file_path}，以下为演示内容",
                 "public Result createOrder(CreateOrderReq req) { ... }"]
    shown = [f"{i:>5}  {lines[i - 1]}"
             for i in range(max(1, start), min(end, len(lines)) + 1)]
    return ToolResponse(content=[TextBlock(type="text", text="\n".join(shown))])

def write_patch_file(file_path: str, patch: str) -> str:
    """把生成的改造补丁（diff/或按原风格改写后的完整文件）写入工作区。
    Args:
        file_path: 目标文件路径
        patch: 补丁或新文件内容
    """
    # TODO 你的实现：根据 AGENT_WRITE_MODE=workspace|inplace 落盘（含 read-before-write 校验）
    return f"已写入 {file_path}（{len(patch)} chars）"

# ---------- 2) 模型：OpenAI 兼容 base_url + api_key + model ----------
def build_model() -> OpenAIChatModel:
    return OpenAIChatModel(
        model_name=os.environ["LLM_MODEL"],                       # Qwen/Qwen3-32b
        api_key=os.environ["LLM_API_KEY"],
        stream=True,
        client_kwargs={"base_url": os.environ["LLM_BASE_URL"]},   # 例如 https://api.siliconflow.cn/v1
        generate_kwargs={"temperature": 0.2},
    )

# ---------- 3) 组装 ReActAgent + 消息循环 ----------
async def main() -> None:
    toolkit = Toolkit()
    for fn in (search_symbols, search_callers, search_similar_code,
               read_file_range, write_patch_file):
        toolkit.register_tool_function(fn)

    agent = ReActAgent(
        name="Code-Refactor-Agent",
        sys_prompt=("你是 Java 代码改造助手。先查符号表与调用图确认影响范围，"
                    "必要时向量检索相似实现；结论必须引用文件路径与行号；"
                    "给出补丁前先 read_file_range 原代码。"),
        model=build_model(),
        formatter=OpenAIChatFormatter(),
        toolkit=toolkit,
        memory=InMemoryMemory(),
        max_iters=15,               # ReAct 循环上限（默认 10）
        parallel_tool_calls=False,
    )

    user_request = ("把 OrderService 里 Feign 同步调用改造成 MQ 异步，"
                    "先画出影响面再给 OrderService.java 的改造补丁。")
    msg = Msg(name="user", content=user_request, role="user")
    print(f"[user] {user_request}\n")
    final = await agent(msg)        # think-act-observe 循环自动跑完
    print(f"\n[agent] {final.get_text_content() or final.content}")

asyncio.run(main())
```

> 跑多轮/接 Web 时，把循环换成 `UserAgent` + `await agent(msg)`（官方 react_agent 示例），或在 FastAPI 里 `await agent(msg)` 后按 `agent` 输出包装 SSE；需要流式过程消息用 `from agentscope.pipeline import stream_printing_messages`（见 2.1(d)）。若你的 endpoint 本身就是百炼 DashScope，把 `DashScopeChatModel + DashScopeChatFormatter` 与上面对换即可（与课程 3_1 完全一致）。

### 3.2 版本 B（2.x —— 当前主线写法）

`pip install agentscope`（=2.0.7.post1，Python ≥ 3.11）。

```python
# code_refactor_agent_v2.py
# -*- coding: utf-8 -*-
"""检索/改码 Agent（AgentScope 2.0 标准写法）。"""
import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.event import EventType
from agentscope.message import UserMsg, AssistantMsg
from agentscope.model import OpenAIChatModel
from agentscope.tool import FunctionTool, Toolkit, Read, Write, Edit, Grep, Glob

# ---------- 1) 工具：普通函数 + FunctionTool；内置文件工具直接用 ----------
def search_symbols(symbol: str, project_id: str = "demo-loan") -> str:
    """在代码符号字典中搜索 Java 类/方法（MySQL）。
    Args:
        symbol: 类名/方法名关键字
        project_id: 项目 ID
    """
    return "OrderService#createOrder(int,CreateOrderReq): 订单创建入口\n..."

def search_callers(fqn: str, project_id: str = "demo-loan") -> str:
    """查询某方法被谁调用（Neo4j 调用图）。
    Args:
        fqn: 方法全限定名
        project_id: 项目 ID
    """
    return "OrderController#create <- Feign OrderApi#createOrder <- Job:createOrderJob"

def search_similar_code(query: str, top_k: int = 5) -> str:
    """按语义相似度检索相似代码（Milvus）。
    Args:
        query: 描述或代码片段
        top_k: 返回条数
    """
    return "[0.92] OrderService.java#L120-L145 ..."

tools = [FunctionTool(search_symbols), FunctionTool(search_callers),
         FunctionTool(search_similar_code),
         Read(), Write(), Edit(), Grep(), Glob()]   # 2.x 内置文件/检索工具

# ---------- 2) 模型 + 凭证 ----------
model = OpenAIChatModel(
    credential=OpenAICredential(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],        # OpenAI 兼容 endpoint
    ),
    model=os.environ["LLM_MODEL"],
    stream=True,
)

# ---------- 3) Agent（自带 ReAct 循环）+ 事件流收发 ----------
async def main() -> None:
    agent = Agent(
        name="Code-Refactor-Agent",
        system_prompt=("你是 Java 代码改造助手。先用工具确认影响面并引用"
                       "文件路径/行号；写文件前必须先 Read 原文件。"),
        model=model,
        toolkit=Toolkit(tools=tools),
        # middlewares=[TracingMiddleware(...)]      # 可选：OTel 追踪
    )

    user_msg = UserMsg(name="user", content=(
        "把 OrderService 里 Feign 同步调用改造成 MQ 异步，"
        "先画影响面再给出改造补丁。"))

    # 事件流：逐字打印模型输出，同时能看到工具调用
    async for event in agent.reply_stream(user_msg, yield_final_msg=True):
        if isinstance(event, AssistantMsg):        # 最终消息
            text = "\n".join(b.text for b in event.content if hasattr(b, "text"))
            print(f"\n[final] {text}")
        elif event.type == EventType.TEXT_BLOCK_DELTA:
            print(event.delta, end="", flush=True)
        elif event.type == EventType.TOOL_CALL_START:
            print(f"\n[tool] {event.tool_call_name} …")

asyncio.run(main())
```

> 2.x 里工具默认要过权限系统（自定义函数工具默认 ASK，即需要用户确认；`FunctionTool` 只读时设 `is_read_only=True`；自动跑可配置权限模式 BYPASS/EXPLORE，见官方文档 building-blocks/permission-system/permission-mode）。若要在 FastAPI 里托管，2.x 官方直接提供 `agentscope.app.create_app(...)` FastAPI 服务（`examples/agent_service/main.py`），内部已带会话、事件流中间件与 Web UI，可优先评估是否直接替换你的服务壳。交互式调试用 `from agentscope.console import launch_console; await launch_console(agent)`。

---

## 4. 调研目标 4：1.x vs 2.x 关键 API 差异 + 2.x 发布状态

### 4.1 差异总表（依据：v2.0.7.post1 官方 change-log + 两侧源码逐一比对）

| 维度 | AgentScope 1.x（课程版本，≤1.0.21） | AgentScope 2.x（2.0.0+，当前 2.0.7.post1） |
|---|---|---|
| ReActAgent | `from agentscope.agent import ReActAgent`（也有 `ReActAgentBase`、`AgentBase`、`UserAgent`、`RealtimeAgent`、`A2AAgent`） | **改名/合并为 `Agent`**：`from agentscope.agent import Agent`；ReAct 循环内置于 Agent，配置走 `ReActConfig`（`max_iters` 默认 50） |
| 调用方式 | `msg = await agent(msg)`（`__call__`） | `__call__` 移除；`await agent.reply(...) -> Msg`；`async for e in agent.reply_stream(...)`（事件流） |
| `_reply` 方法 | `AgentBase` 是抽象基类：自定义 agent 继承并实现 `reply/observe/handle_interrupt`（`ReActAgentBase` 再加 `_reasoning/_acting`） | `Agent` 是**具体类**（配置式使用），内部仍有 `async def _reply`（事件发生器）供框架调用，但不作为扩展点；扩展点是 `MiddlewareBase` |
| `@agent` 装饰器 | 无（1.x 无此装饰器，勿混淆其他框架） | 无（2.x 无此装饰器；顶层 `agentscope` 只剩 `logger/setup_logger/set_id_factory/set_timestamp_factory/__version__`） |
| msghub / MsgHub | `from agentscope.pipeline import MsgHub`（另 `sequential_pipeline/fanout_pipeline/ChatRoom`） | **删除**（2.x pipeline 只剩 `PipelineProtocol`、`GoalPipeline`）；多智能体改走：事件流互发 + `observe()` 注入、Agent Service 的 Agent Team（leader–worker + team 工具）、`GoalPipeline` |
| Agent 定义 | `ReActAgent(name, sys_prompt, model, formatter, toolkit=…, memory=…, …)`；自定义 = 继承 AgentBase | `Agent(name, system_prompt, model, toolkit=…, middlewares=…, react_config=…, …)`；不再要求 formatter（模型自带默认 formatter） |
| 工具定义 | `Toolkit()` + `toolkit.register_tool_function(fn)` / `register_mcp_client` / `register_agent_skill`；内置 `execute_python_code/execute_shell_command/view_text_file/write_text_file/insert_text_file` | `Toolkit(tools=[FunctionTool(fn), Read(), Write(), Edit(), Bash(), Grep(), Glob(), TaskCreate()…])`；`FunctionTool`/`MCPTool` 适配器；`ToolBase` + `check_permissions`/`call`；**没有 `@agentscope.tool` 装饰器、没有 ToolUse 类名** |
| 消息块 | `ToolUseBlock / ToolResultBlock / ImageBlock / AudioBlock / VideoBlock / TextBlock / ThinkingBlock` | `ToolUseBlock → ToolCallBlock`；`Image/Audio/Video → DataBlock(media_type=…)`；新增 `HintBlock`；Msg 带 `id/created_at/finished_at/usage/structured_output`；新增工厂 `UserMsg/AssistantMsg/SystemMsg` |
| Msg 内容 | 通常 str 文本（`Msg("user","你好","user")`） | `content: list[ContentBlock]`（pydantic 强校验）；工厂函数自动把 str 包成 `TextBlock` |
| 模型配置 | `OpenAIChatModel(model_name=…, api_key=…, client_kwargs={"base_url":…})`、`DashScopeChatModel(model_name=…, api_key=…)`；formatter 单独传 | `OpenAIChatModel(credential=OpenAICredential(api_key=…, base_url=…), model=…, parameters=…)`；凭证/模型解耦（`agentscope.credential`），provider 内建 formatter，支持 `ModelCard`/`list_models` |
| formatter | 必传且手动对齐 provider（`DashScopeChatFormatter/OpenAIChatFormatter/…MultiAgentFormatter`） | 融入模型类（默认按 provider 生成）；`agentscope.formatter` 仍有 `FormatterBase/OpenAIChatFormatter` 等可覆盖 |
| 记忆 | `agentscope.memory`：`InMemoryMemory`、`Mem0LongTermMemory`、`memory.get_memory()`；RAG 独立 `agentscope.rag` | **memory 模块弃用**（change-log：Deprecate the memory module in 2.0）；长期记忆 = `AgenticMemoryMiddleware` + ReMe/Mem0（`agentscope.middleware`）；RAG 在 2.0 架构上重建 |
| hooks/OTel | hooks（reply/observe/print/_reasoning/_acting…）、`agentscope.init(studio_url=…, tracing_url=…)`、`agent.set_console_output_enabled(...)` | **hooks → middleware**（`MiddlewareBase`）；`agentscope.init` **移除**；追踪 = `TracingMiddleware`；agent 变"纯生产者"（不再自带终端打印），控制台由 `launch_console`/事件流承担 |
| 权限/HITL | 无独立权限体系（打断恢复走 realtime steering） | 新增 `agentscope.permission`（工具/资源分级、BYPASS/EXPLORE/ACCEPT_EDITS/DONT_ASK 模式）、`RequireExternalExecutionEvent` 等 HITL 事件 |
| 事件系统 | 无（消息打印用 `stream_printing_messages`） | 新增 `agentscope.event`：`EventType.{REPLY_END,TEXT_BLOCK_DELTA,THINKING_BLOCK_DELTA,TOOL_CALL_START,TOOL_CALL_DELTA,TOOL_CALL_END,TOOL_RESULT_*,…}` |
| 服务化 | 无内置服务层（自己 FastAPI 包） | `agentscope.app.create_app()`：官方 FastAPI Agent Service + Web UI、多租户/会话/RAG/渠道/Agent Team（`examples/agent_service`） |
| Python | ≥3.10 | ≥3.11 |
| 文档站点 | https://doc.agentscope.io/ | https://docs.agentscope.io/（含 /versions/2.0.7 等快照；main 分支/`latest` 指向 2.x） |

### 4.2 2.x 发布状态与 PyPI

- 包名不变：**`agentscope`**（PyPI: https://pypi.org/project/agentscope/ ），要求 `python >=3.11`。
- 版本时间线（PyPI 上传日期核实）：`1.0.0` 2025-08-15 → `1.0.18` 2026-03-26（课程锁定）→ `1.0.21` 2026-05-25（1.x 末版）→ **`2.0.0` 2026-05-25**（breaking）→ `2.0.2/2.0.4/2.0.6/2.0.7`（6–8 月月更）→ **当前最新 `2.0.7.post1`（2026-08-28）**；GitHub release 与 PyPI 同步。
- 成熟度判断：2.x 处于**活跃稳定发布期**（官方 README/NEWS 2026-06~08 新增 pipeline、console、channels、Agent Team、workspace 沙箱等），但其 change-log 明确是 breaking release，且部分 1.x 模块（memory、部分 evaluate/RAG）在 2.0 仍在迁移或重构 —— **对照 ACP 课程学 1.x、项目选型两者皆可但代码不互通**。

### 4.3 版本建议（回到你的问题）

1. **为了对照 ACP 课程学习并把课程代码当样板：安装 1.x**。最稳妥 `pip install "agentscope==1.0.18"`（与课程逐字节一致）；追求 1.x 最新修复可用 `agentscope==1.0.21`（API 与 1.0.18 一致，上述 import 全部成立）。
2. 若要跟上 2.x 主线（尤其是官方 FastAPI Agent Service、事件流、权限系统），**另建一个 Python ≥3.11 的 venv**，`pip install agentscope`，并把"检索/改码 Agent"用 3.2 的 2.x 骨架再实现一份 —— 两版都保留，恰好就是最好的 1.x→2.x 迁移教学素材。
3. 不要在 2.x 上直接跑 ACP notebook（`ReActAgent/MsgHub/agentscope.init` 等全部不兼容）。

---

## 附录 A：关键 API 签名清单（import 路径，均对照源码核实）

### A.1 AgentScope 1.x（`agentscope==1.0.18`）

```python
from agentscope.agent import AgentBase, ReActAgentBase, ReActAgent, UserAgent, A2AAgent
ReActAgent(name: str, sys_prompt: str, model: ChatModelBase, formatter: FormatterBase,
           toolkit: Toolkit | None = None, memory: MemoryBase | None = None,
           long_term_memory: LongTermMemoryBase | None = None,
           long_term_memory_mode="both", enable_meta_tool=False,
           parallel_tool_calls=False, knowledge=None, enable_rewrite_query=True,
           plan_notebook=None, print_hint_msg=False, max_iters=10,
           tts_model=None, compression_config=None)

from agentscope.model import DashScopeChatModel, OpenAIChatModel, AnthropicChatModel, GeminiChatModel, OllamaChatModel, TrinityChatModel
OpenAIChatModel(model_name, api_key=None, stream=True, reasoning_effort=None,
                organization=None, stream_tool_parsing=True,
                client_type="openai", client_kwargs=None, generate_kwargs=None)   # base_url 放 client_kwargs

from agentscope.formatter import (DashScopeChatFormatter, DashScopeMultiAgentFormatter,
                                  OpenAIChatFormatter, OpenAIMultiAgentFormatter,
                                  AnthropicChatFormatter, ...)
from agentscope.message import Msg, TextBlock, ThinkingBlock, ToolUseBlock, ToolResultBlock, ImageBlock, AudioBlock, VideoBlock, ContentBlock
Msg(name, content, role)          # role in {"user","assistant","system",...}
from agentscope.memory import InMemoryMemory, Mem0LongTermMemory, MemoryBase
from agentscope.tool import Toolkit, ToolResponse, execute_python_code, execute_shell_command, view_text_file, write_text_file, insert_text_file
Toolkit().register_tool_function(func); Toolkit().register_mcp_client(client); Toolkit().register_agent_skill(dir)
from agentscope.pipeline import MsgHub, SequentialPipeline, FanoutPipeline, sequential_pipeline, fanout_pipeline, stream_printing_messages, ChatRoom
MsgHub(participants=[...], announcement=Msg(...))    # async with MsgHub(...) as hub: hub.add/hub.delete/hub.broadcast
from agentscope.mcp import StdIOStatefulClient, HttpStatelessClient, HttpStatefulClient
from agentscope.plan import PlanNotebook
from agentscope.evaluate import Task, MetricBase, MetricResult, MetricType, SolutionOutput
import agentscope; agentscope.init(project=None, name=None, run_id=None, logging_path=None,
                                   logging_level="INFO", studio_url=None, tracing_url=None)
```

### A.2 AgentScope 2.x（`agentscope==2.0.7.post1`）

```python
from agentscope.agent import Agent, ContextConfig, InjectionConfig, ModelConfig, ReActConfig
Agent(name: str, system_prompt: str, model: ChatModelBase, toolkit: Toolkit | None = None,
      middlewares: list[MiddlewareBase] | None = None, state: AgentState | None = None,
      offloader: Offloader | None = None, model_config: ModelConfig | None = None,
      context_config: ContextConfig | None = None, react_config: ReActConfig | None = None,
      injection_config: InjectionConfig | None = None)
# Agent 公开方法：reply / reply_stream / observe / compress_context

from agentscope.credential import OpenAICredential, DashScopeCredential, AnthropicCredential, DeepSeekCredential, GeminiCredential, MoonshotCredential, OllamaCredential, XAICredential
OpenAICredential(api_key: SecretStr, base_url: str | None = None, organization: str | None = None)

from agentscope.model import OpenAIChatModel, OpenAIResponseModel, DashScopeChatModel, AnthropicChatModel, DeepSeekChatModel, GeminiChatModel, MoonshotChatModel, OllamaChatModel, XAIChatModel, ModelCard, ChatResponse, StructuredResponse, ChatUsage
OpenAIChatModel(credential: OpenAICredential, model: str,
                parameters: OpenAIChatModel.Parameters | None = None,
                stream: bool = True, max_retries=3, retry_delay=1.0,
                context_size=128000, formatter=None, client_kwargs=None, extra_body=None)

from agentscope.message import Msg, UserMsg, AssistantMsg, SystemMsg, Usage
from agentscope.message import TextBlock, ThinkingBlock, HintBlock, ToolCallBlock, ToolResultBlock, DataBlock, Base64Source, URLSource, ContentBlock
UserMsg(name=..., content="str 或 blocks"); AssistantMsg(...); SystemMsg(...)

from agentscope.tool import Toolkit, ToolBase, ToolChunk, ToolResponse, FunctionTool, MCPTool, ToolGroup, Function, RegisteredTool, ToolChoice
from agentscope.tool import Bash, PowerShell, Read, Write, Edit, Glob, Grep, TaskCreate, TaskGet, TaskList, TaskUpdate, ResetTools
Toolkit(tools=[...], skills_or_loaders=[...], mcps=[...], tool_groups=[...])
FunctionTool(func, name=None, description=None, input_schema=None, is_concurrency_safe=True,
             is_read_only=False, is_state_injected=False, middlewares=None)
# ToolBase 子类：name/description/input_schema + async check_permissions(tool_input, context) + async call(**kwargs)

from agentscope.event import EventType, AgentEvent, TextBlockDeltaEvent, ToolCallStartEvent, ReplyEndEvent, ...
from agentscope.permission import PermissionMode, PermissionDecision, PermissionBehavior, PermissionContext
from agentscope.middleware import MiddlewareBase, AgenticMemoryMiddleware, TracingMiddleware
from agentscope.console import launch_console
from agentscope.pipeline import PipelineProtocol, GoalPipeline
from agentscope.app import create_app, SubAgentTemplate
from agentscope.mcp import MCPClient, StdioMCPConfig, HttpMCPConfig
from agentscope.workspace import LocalWorkspace, DockerWorkspace, E2BWorkspace
```

---

## 附录 B：证据与来源（全部可点开复核）

**ACP 认证课程仓库（= AgentScope 1.x 内容）**
- 仓库首页/目录：https://github.com/AlibabaCloudDocs/aliyun_acp_learning
- `requirements.txt`（`agentscope==1.0.18`、`mcp<2.0`、`mem0ai<2.0`）：https://github.com/AlibabaCloudDocs/aliyun_acp_learning/blob/main/requirements.txt
- 3_1 Agent 基础与工具调用（含课程版 ReActAgent/Toolkit 示例与 MCP）：https://github.com/AlibabaCloudDocs/aliyun_acp_learning/blob/main/大模型ACP认证教程/C3_构建Agent系统/3_1_Agent基础与工具调用.ipynb
- 3_2 让 Agent 学会规划与执行（PlanNotebook/pipeline）：https://github.com/AlibabaCloudDocs/aliyun_acp_learning/blob/main/大模型ACP认证教程/C3_构建Agent系统/3_2_让Agent学会规划与执行.ipynb
- 3_3 用多 Agent 实现团队协作（MsgHub/MultiAgentFormatter）：https://github.com/AlibabaCloudDocs/aliyun_acp_learning/blob/main/大模型ACP认证教程/C3_构建Agent系统/3_3_用多Agent实现团队协作.ipynb
- 3_4 用 Memory 让 Agent 积累经验：…/C3_构建Agent系统/3_4_用Memory让Agent积累经验.ipynb（同目录）
- 3_5 用 Skill 将能力固化为可复用流程：…/3_5_用Skill将能力固化为可复用流程.ipynb；3_6 用评测驱动 Agent 开发：…/3_6_用评测驱动Agent开发.ipynb
- 英文镜像 2_6：https://github.com/AlibabaCloudDocs/aliyun_acp_learning/blob/main/LLM_ACP_EN/p2_Build%20LLM%20Q%26A%20System/2_6_Building_an_Agent_to_Accomplish_Complex_Tasks.ipynb

**AgentScope 官方（GitHub / 源码 / 示例）**
- 主仓库：https://github.com/agentscope-ai/agentscope
- 1.x ReActAgent 官方示例：https://github.com/agentscope-ai/agentscope/blob/v1.0.18/examples/agent/react_agent/main.py
- 1.x 流式打印官方示例：https://github.com/agentscope-ai/agentscope/blob/v1.0.18/examples/functionality/stream_printing_messages/single_agent.py
- 1.x 源码 `agent/_react_agent.py`、`tool/_toolkit.py`、`model/_openai_model.py`、`pipeline/_msghub.py`、`pipeline/__init__.py`：…/blob/v1.0.18/src/agentscope/…（同 tag）
- 1.x 官方教程源码（doc.agentscope.io 页面来源，含 quickstart_agent/task_tool/task_agent_skill 等）：…/blob/v1.0.18/docs/tutorial/en/src/quickstart_agent.py
- 2.x README/Quickstart（Agent+Toolkit 内置工具+launch_console）：https://github.com/agentscope-ai/agentscope/blob/v2.0.7.post1/README.md
- 2.x console 示例（Agent + Workspace + AgenticMemoryMiddleware）：…/v2.0.7.post1/examples/console/main.py
- 2.x Agent Service 示例（FastAPI create_app/多租户）：…/v2.0.7.post1/examples/agent_service/main.py
- 2.x 源码：`src/agentscope/agent/_agent.py`、`agent/_config.py`（ReActConfig/ModelConfig/ContextConfig/InjectionConfig）、`tool/_adapters.py`（FunctionTool）、`tool/_toolkit.py`、`model/_openai_chat/_model.py`、`credential/_openai.py`、`message/_base.py`、`event/_event.py`（…/v2.0.7.post1/src/agentscope/…）
- 2.x 官方 change-log（"Refactor ReActAgent into a new unified Agent class implementation." 等逐条差异）：https://github.com/agentscope-ai/agentscope/blob/v2.0.7.post1/docs/NEWS.md ；版本化文档镜像：https://docs.agentscope.io/versions/2.0.7/en/others/change-log （文档源码 https://github.com/agentscope-ai/agentscope-docs…，即 org 仓库 `agentscope-ai/docs`，路径 `versions/2.0.7/en/…`）

**官方文档站点 / 文档仓库**
- 1.x 文档站（doc.agentscope.io，教程/FAQ/API）：https://doc.agentscope.io/ ；课程引用的 `task_agent_skill.html`、`task_memory.html`、`task_eval.html` 均在 doc.agentscope.io/tutorial/ 下
- 2.x 文档站（含版本快照与 zh 中文）：https://docs.agentscope.io/ ；2.0.7 英文 quickstart：https://docs.agentscope.io/latest/en/quickstart （镜像文件 `agentscope-ai/docs` `versions/2.0.7/en/quickstart.mdx`）
- 2.x building-blocks 英文页：agent overview/run-agent（reply/reply_stream/observe）、tool python-tool（FunctionTool/ToolBase）、tool mcp、model llm、console、permission-system、middleware、pipeline —— 均在 https://docs.agentscope.io/latest/en/building-blocks/…（仓库镜像 `agentscope-ai/docs` `versions/2.0.7/en/building-blocks/…`）

**版本与发布状态**
- PyPI 包页：https://pypi.org/project/agentscope/ （json API：https://pypi.org/pypi/agentscope/json —— 全版本+日期；2.0.7.post1 为最新，requires_python >=3.11）
- GitHub Releases：https://github.com/agentscope-ai/agentscope/releases （v1.0.18 2026-03-26；v2.0.0 2026-05-25；v2.0.7.post1 2026-08-28）

**参考论文（可选背景）**
- AgentScope 1.0 论文：https://arxiv.org/abs/2508.16279 ；AgentScope 平台论文：https://arxiv.org/abs/2402.14034
