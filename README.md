# Business Code Agent

面向 Spring Cloud / Maven 多模块 Java 项目的代码智能与改码助手：

1. 解析 Java 类/方法 → MySQL 符号与标签字典  
2. AI 打标签/描述 → Milvus 向量检索  
3. 同步 + 异步（MQ / `@Async` / XXL-Job / Feign）调用图 → Neo4j  
4. 多轮对话：读图定范围 → 按原风格给出改造方案/补丁  

当前默认索引项目：`java_project/demo-loan`。表结构与图模型均带 `project_id`，便于后续扩展多项目与跨仓调用。

## 快速开始

### 1. 启动中间件

先启动 Docker Desktop，再执行：

```bash
docker compose up -d
```

包含 MySQL、Neo4j、Milvus（etcd/minio）。未启动中间件时，API 仍可启动；Milvus/会话会走内存回退，但完整 ingest 需要 MySQL + Neo4j。

### 2. 配置环境

```bash
copy .env.example .env
# 编辑 .env，填入硅基流动 LLM_API_KEY
```

关键配置（见 `.env.example`）：

| 变量 | 说明 |
|------|------|
| `LLM_PROVIDER` | `siliconflow`（默认）/ `ollama` / `deepseek` / `mock` |
| `LLM_MODEL` / `LLM_BASE_URL` / `LLM_API_KEY` | 硅基流动例见下 |
| `EMBEDDING_*` | 与 LLM 独立：`EMBEDDING_PROVIDER` / `EMBEDDING_API_KEY` / `EMBEDDING_BASE_URL`；DeepSeek 无 embeddings，需硅基流动或 `hash` |
| `MYSQL_*` / `NEO4J_*` | 本地 MySQL / Neo4j |
| `MILVUS_MODE=lite` | 本地文件库 `MILVUS_LITE_PATH=data/milvus_demo.db`，无需 Docker |
| `AGENT_WRITE_MODE` | `workspace` 或 `inplace` |
| `AGENT_RUNTIME_MODE` | 编排层运行模式：`standard`（默认，自研 LangGraph 链路）/ `minimal` / `agentscope`（走 AgentScope 迁移层） |
| `AGENTSCOPE_TEAM` | 仅 agentscope 模式：空=单 ReActAgent；`msghub`=MsgHub 多智能体协作（探索+评审） |

换模型示例：

```env
# 硅基流动（默认）：对话 + 向量同一 API Key
LLM_PROVIDER=siliconflow
LLM_MODEL=Qwen/Qwen3-32b
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_API_KEY=sk-xxx
EMBEDDING_PROVIDER=siliconflow
EMBEDDING_MODEL=BAAI/bge-large-zh-v1.5
EMBEDDING_DIM=1024

# 本地 Ollama
# LLM_PROVIDER=ollama
# LLM_MODEL=qwen3:32b
# LLM_BASE_URL=http://127.0.0.1:11434

# DeepSeek 对话 + 硅基流动向量（必须分开配）
# LLM_PROVIDER=deepseek
# LLM_MODEL=deepseek-chat
# LLM_BASE_URL=https://api.deepseek.com
# LLM_API_KEY=sk-deepseek-xxx
# EMBEDDING_PROVIDER=siliconflow
# EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
# EMBEDDING_API_KEY=sk-siliconflow-xxx
```

切换 embedding 维度后（例如 512→1024），请删除旧的 `MILVUS_LITE_PATH` 文件再重新 ingest。

### 3. 安装依赖并启动

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
# 等价：python -m app.main   或已安装时 bca-server
```

启动时会自动 `init_db`（含 `method_var_resolve` 等表）；默认入口的 var-resolve bags 在缺失时后台/懒加载补齐，日常问答与「加字段并落库」无需再跑同步脚本。

- 对话页：http://127.0.0.1:8000/ （或 `/ui`）
- 项目管理：http://127.0.0.1:8000/projects （上传 zip、查看状态、一键解析）
- API 文档：http://127.0.0.1:8000/docs
- Windows 客户端（Cursor 风格，连接同一后端）：见 [`desktop/README.md`](desktop/README.md)
  - 多项目勾选对话；修代码时可本机落补丁 + 编译验证（`client_apply` + `/api/v1/agent/ws`）
  - 未勾选项目＝普通 AI 问答（带会话上下文记忆）；同一会话内可随时勾选/取消项目切换问答模式

对话页支持连续多轮（前端自动携带 `session_id`），侧边栏可切换/新建会话。  
项目页点击「解析」会触发 ingest：Java 解析 → MySQL 标签/描述 → Neo4j 调用图 → Milvus 向量。

### 4. 摄入代码

```bash
# API
curl -X POST http://127.0.0.1:8000/api/v1/projects/ingest -H "Content-Type: application/json" -d "{\"full_refresh\":true}"

# 或 CLI
python scripts/run_ingest.py --full-refresh
```

无 DeepSeek Key 时可先：

```bash
set LLM_PROVIDER=mock
set EMBEDDING_PROVIDER=hash
python scripts/run_ingest.py --full-refresh
```

### 5. 检索 / 流程 / 对话

```bash
# 语义+标签检索
curl "http://127.0.0.1:8000/api/v1/search?q=Controller"

# 调用/异步流程
curl "http://127.0.0.1:8000/api/v1/graph/flow?entry=DemoController"

# 多轮对话
curl -X POST http://127.0.0.1:8000/api/v1/chat -H "Content-Type: application/json" -d "{\"message\":\"放款接口增加一个字段，并落库，然后再透传给下游\"}"
```

本地解析冒烟（不依赖中间件）：

```bash
python scripts/e2e_demo.py
python scripts/e2e_demo.py --chat "放款流程什么样"
```

## 架构要点

- **LLM 可插拔**：`app/llm/base.py` + `factory.py`，业务代码不绑定 DeepSeek。  
- **异步串链**：`@RabbitListener` → `Queue` ← `convertAndSend`，并推导 `TRIGGERS` 边。  
- **改码写入**：默认写入 `.agent_workspace`，避免直接改样本工程；需要时可设 `AGENT_WRITE_MODE=inplace`。
- **黄金问答库（内网 qwen3 专用）**：离线用思考模式 + 自一致性投票把项目咀嚼成
  高置信问答对（MySQL `golden_qa` 表）；运行时 ask 类问题命中即答，未命中走
  原管线；非代码问题走通用兜底直答 —— 保证「任何问题都有回复」且准确率不依赖
  qwen3 在线长链推理。生成：`python scripts/generate_golden_qa.py`（详见
  `docs/内网部署.md` 第 7 节）。

### AgentScope 迁移层（学习阿里云 ACP 课程）

编排层在保留原自研链路（`app/agent/graph_agent.py` 等，可随时回退）的同时，新增了
**AgentScope 标准写法迁移层** `app/agent/agentscope_bridge/`，作为学习阿里云 ACP 认证课程的代码样板：

- 框架版本对齐课程：`agentscope==1.0.18`（见 `requirements.txt`，含 mcp<2 依赖约束）
- 切换：`.env` 设 `AGENT_RUNTIME_MODE=agentscope`（路由入口在 `app/agent/graph_agent.py` → `adapter.py`）
- 覆盖课程核心知识点，可逐章对照：
  - ReActAgent + Toolkit 工具系统（`agents/explore_agent.py`，对应课程 3_1）
  - AgentBase / Msg / Formatter / InMemoryMemory（`agents/general_agent.py`、`model.py`）
  - MsgHub 多智能体协作（`agents/msghub_team.py`，对应课程 3_3，`AGENTSCOPE_TEAM=msghub` 启用）
  - 结构化输出 structured_model（`agents/modify_agent.py` 改码链路：ReAct 探索 → 结构化补丁 → workspace 落地）
  - 日志诊断迁移（`log_diagnosis_scope.py`：Model + Msg + Formatter.format 直连）
- 兼容红线：`/api/v1/chat`、SSE、会话、写工具安全（`.agent_workspace` copy-on-write）全部不变，
  原实现 `AGENT_RUNTIME_MODE=standard` 随时回退
- 完整迁移方案与真机踩坑笔记：`docs/agentscope迁移方案.md`

## API 一览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/health` | 健康检查 |
| POST | `/api/v1/projects/ingest` | 摄入项目 |
| GET | `/api/v1/jobs/{id}` | 任务进度 |
| GET | `/api/v1/search?q=` | 混合检索 |
| GET | `/api/v1/graph/flow?entry=` | 流程子图 |
| POST | `/api/v1/chat` | 多轮对话改码 |
| GET | `/api/v1/chat/{session_id}` | 会话历史 |

## 测试

```bash
pip install pytest
set EMBEDDING_PROVIDER=hash
set LLM_PROVIDER=mock
pytest -q
```

## 日志查看 / 问题定位（已内置 ✅）

BCA 已内置「日志 → 代码问题定位」管线：`/api/v1/chat` 输入含日志的问题自动走 `log_diagnose`
意图（锚点提取 → 锚点→符号检索 → 真实源码行级证据 → 结构化诊断：现象/定位/根因/为什么/修复/证据链与置信度，
分支走错场景强制「配置项→读取点→兜底默认→分支条件→观测日志」五段证据链）；另有日志浏览/诊断 API
（`/api/v1/logs/meta`、`/logs/tail`、`/logs/diagnose`）与 Web 页 `/logdiag`。

15 场景全量回归（`python scripts/eval_log_diagnosis.py`）：基线 0/15 有效诊断 → 落地后 **12/15（80%）**，
无套主链兜底、S11 正常停机正确识别为「非故障」。分支+配置场景若无法定论，回答会输出
「嫌疑代码块与不确定性」清单（读取点/兜底默认/分支条件/路由点逐条列嫌疑 + 还差什么信息才能定论），
不假装确定。评测与实现细节：

- 评测场景集：`data/log_diag_battery/scenarios.json`（15 个真实代码+真实日志构造的诊断场景，
  覆盖 XXL-Job 参数缺失/校验盲区、RabbitMQ 解析失败裸签收/脏数据静默丢、Feign 服务发现、
  SQL 表结构漂移、Controller @NotBlank、NPE 未判空、利率校验 NumberFormatException、正常停机防误报、
  **分支走错（branch-routing）**：配置不走风控却实际走了新风控引擎（newEngineLoan 空值默认 Y）、
  调额 isRisk 开关取值语义错位导致分支走错、新老流程配置表名注释/注解不一致导致静默走老流程）
- 评测脚本：`python scripts/eval_log_diagnosis.py`（跑 `/api/v1/chat` 实测，输出 `data/log_diag_battery/results.md`）
- 评测报告：`docs/日志定位能力评测报告.md`（基线 15/15 被拒、0 有效诊断 → 落地后 12/15 有效、分支五段证据链达成）
- 增强方案：`docs/日志查看与问题定位增强方案.md`（实现落点：`app/agent/log_diagnosis.py`、`app/api/logs.py`、
  `web/logdiag.html`/`web/logdiag.js`、`configs/prompts/log_diagnose_system.txt`；另附评测中发现的 10 个 Java 侧可修代码问题）
