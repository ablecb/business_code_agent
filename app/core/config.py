from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_host: str = "0.0.0.0"
    app_port: int = 8000

    project_id: str = "demo-loan"
    project_root: str = "java_project/demo-loan"
    agent_write_mode: Literal["inplace", "workspace"] = "workspace"
    workspace_dir: str = ".agent_workspace"

    llm_provider: str = "siliconflow"
    llm_model: str = "Qwen/Qwen3-32b"
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_api_key: str = ""
    llm_temperature: float = 0.2
    llm_max_tokens: int = 16384
    # Comma-separated extra models (same base_url/api_key) tried in order when
    # the primary model fails — failover until one succeeds. Empty = no failover.
    llm_failover_models: str = ""
    # ===== Parse-only LLM (ingest/topo/tagging/flow), independent of chat LLM =====
    # Empty values fall back to the main LLM_* config (e.g. prod: parse with qwen3-32b).
    # Dev: point these at a cheap host (sensenova) while chat stays on the main LLM.
    parse_llm_provider: str = ""
    parse_llm_model: str = ""
    parse_llm_base_url: str = ""
    parse_llm_api_key: str = ""
    parse_llm_failover_models: str = ""
    # Comma-separated multiple parse API keys — rotated round-robin across
    # calls; if every model fails on one key, the next key is tried.
    parse_llm_api_keys: str = ""
    # HTTP read timeout for LLM chat (seconds). Slow remote models need headroom.
    llm_timeout: float = 300.0
    # 本机路由判定（route_client_need）的 LLM 调用超时上限（秒）。
    # 路由只是 tiny 分类调用，慢端点不值得烧满 LLM_TIMEOUT/总预算——
    # 超时即 fail-closed mode=none（安全方向，不启动任何本机动作），
    # 避免 desktop-chat-997ffe53135d 这类整 240s 白等（desktop 一次路由判定
    # 烧满 LLM_TOTAL_TIMEOUT 后回退 none，用户 UI 空白 4 分钟）。
    llm_route_timeout: float = 30.0
    # 单次 LLM 调用总预算（秒，0=不限制）。覆盖整个 failover / 多 key 链，
    # 防止端点卡死时「超时 × 重试 × 换 key × 换模型」把一轮问答拖到几十分钟~几小时。
    llm_total_timeout: float = 90.0
    # True 时问答 LLM 也走 PARSE_LLM_*（sensenova + 多 key 轮询）。
    # 默认 False：问答走主 LLM_* 单模型，避免多 key failover 链拖垮单轮问答。
    chat_llm_use_parse: bool = False
    # ===== 问答 LLM 熔断（fail-fast circuit breaker）=====
    # 端点故障（连读超时/传输错误）时，连续失败达到阈值后 chat() 立即快速失败，
    # 让启发式/确定性兜底接管，而不是每次调用都烧满 LLM_TOTAL_TIMEOUT（90s）×重试。
    # desktop-chat-5b263372926e 就是这类故障把一轮问答拖到 7.3 分钟。
    llm_circuit_fail_threshold: int = 2     # 连续失败多少次后打开熔断
    llm_circuit_open_secs: float = 45.0     # 打开后持续多久（之后进入 half-open 探测）
    llm_circuit_probe_timeout: float = 15.0  # half-open 探测请求的单次超时（秒）
    # False: skip TLS verify (proxy/hostname-mismatch). Prefer fixing certs in prod.
    http_verify_ssl: bool = True
    # True: log chat prompt/response bodies to console (app.llm).
    llm_log_prompts: bool = True
    llm_log_max_chars: int = 8000
    # Qwen3 等推理模型默认会 think，打标签很慢；关闭后只返回最终 content。
    llm_enable_thinking: bool = False
    # Conversation compaction (DSH-style): when messages exceed the threshold,
    # middle turns are LLM-condensed into a summary instead of being dropped.
    compaction_enabled: bool = True
    compaction_threshold: int = 50  # total messages before compaction triggers
    compaction_keep_tail: int = 16  # most recent messages kept verbatim

    embedding_provider: str = "siliconflow"
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_dim: int = 1024
    # Prefer dedicated keys. Empty key/url may reuse LLM_* only when LLM host
    # supports embeddings (not DeepSeek chat).
    embedding_api_key: str = ""
    embedding_base_url: str = ""

    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_user: str = "root"
    mysql_password: str = "123456"
    mysql_database: str = "business_code_agent"
    # True 时 SQLAlchemy 在控制台打印执行的 SQL（调试用）
    mysql_echo: bool = False

    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""  # 通过环境变量 NEO4J_PASSWORD 配置

    # lite = local file via MilvusClient(uri=*.db); server = docker compose milvus
    # NOTE: do not name env MILVUS_URI — pymilvus reserves that for server HTTP URIs
    # Start: docker compose up -d etcd minio milvus
    milvus_mode: str = "server"
    milvus_lite_path: str = "data/milvus_demo.db"
    milvus_host: str = "127.0.0.1"
    milvus_port: int = 19530
    milvus_collection: str = "method_embeddings"
    # Separate collection: entry routing / classification (Controller#method anchors).
    milvus_entry_route_collection: str = "entry_route"
    # When false: chat/explore skip Milvus vector + entry_route.
    search_use_milvus: bool = True
    # Explore retrieval backend:
    #   source = AI plan + filesystem tools (FOLLOWS/FlowDoc still used for sync/full-flow Qs)
    #   index  = MySQL (+ optional Milvus when search_use_milvus=true)
    explore_backend: str = "index"
    # ===== 问答双赢优化开关（默认开，单开关可一键回滚）=====
    # 进程内 LRU 缓存 hybrid_search/embedding 结果：追问/相似问题直接命中，
    # 跳过 Milvus + MySQL，同时保证同一问题检索口径一致。
    search_cache_enabled: bool = True
    search_cache_max_embed: int = 512
    search_cache_max_search: int = 256
    # 会话证据继承：追问（use_session_ctx/dig_in）轮复用上一轮已读证据摘要，
    # 提示"禁止重复 get_symbol_code"，省 3~6 轮工具循环且口径不漂移。
    agent_evidence_reuse: bool = True
    evidence_reuse_max_chars: int = 2400
    # 用户正向反馈（vote=up）自动回填 golden_qa（source=manual_feedback），
    # 命中即免长链；重复问题不覆盖离线高置信（consensus/votes>=0.7）答案。
    golden_qa_backfill_enabled: bool = True

    # True: chat uses Planner–Executor–Verifier loop (test tags [PEV][...]).
    agent_use_pev: bool = True
    # When PEV fails/empty, fall back to ask_retrieval / explore_agent.
    # Keep True so empty-PEV still has a safety net; set False to force PEV-only.
    agent_pev_fallback: bool = True
    # True: append [PEV][...]/契约缺口 internal debug tags into the user-visible
    # answer (dev/testing only). Default False keeps user replies clean; the
    # tags/gaps are still returned in result["pev_debug"].
    pev_test_tags_in_answer: bool = False
    # Harness-style runtime: standard = full skills; minimal = eval-friendly subset;
    # agentscope = AgentScope 迁移层（agentscope_bridge，学习样板，原实现保留可回退）。
    agent_runtime_mode: Literal["standard", "minimal", "agentscope"] = "standard"
    # agentscope 模式下的协作方式：空=单 ReActAgent；msghub=MsgHub 多智能体（探索+评审）。
    agentscope_team: str = ""
    # agentscope 改码落地的失败反馈重试轮数（0=不重试；OLD 不匹配时模型修订重提）。
    agentscope_modify_retries: int = 2
    # Append-only JSONL trajectory (tool/progress/verify) under agent_trajectory_dir.
    agent_trajectory_enabled: bool = True
    agent_trajectory_dir: str = "data/trajectories"
    # Verify/repair retries when agent_runtime_mode=minimal (0 = verify once, no repair).
    agent_minimal_max_repairs: int = 0

    # ===== 黄金问答库（方案一：离线预咀嚼，运行时检索直接作答）=====
    # True 时 chat 对 ask 类问题先查 golden_qa 表，命中高分直接返回存储答案。
    golden_qa_enabled: bool = True
    # 命中阈值（0~1，语义+关键词加权分），低于阈值不拦截，走正常管线。
    golden_qa_min_score: float = 0.55
    # 检索返回条数上限。
    golden_qa_top_k: int = 3
    # 离线生成时每个问题采样的答案数（自一致性投票）。
    golden_qa_n_samples: int = 3
    # 离线生成时最多生成多少条候选问答（0=自动，按索引规模）。
    golden_qa_max_candidates: int = 300
    # 候选问答条数上限（每条问题字数上限，避免喂给弱模型超长上下文）。
    golden_qa_context_max_chars: int = 2600
    # 通用兜底：检索无命中（非代码问题）时直接用 LLM 回答，保证任何问题都有回复。
    general_fallback_enabled: bool = True

    ingest_batch_size: int = 20
    # Concurrent LLM tag calls (async semaphore). API-bound — not OS threads.
    tagging_concurrency: int = 8
    # Concurrent AI parse workers (LLM-bound; keep modest to avoid rate limits).
    parse_concurrency: int = 10
    # Topo AST pipeline (global tree-sitter + priority topo AI). Default on.
    topo_ast_pipeline_enabled: bool = True
    # AI analyze retries per node before fail + upward propagate.
    topo_max_retries: int = 2
    # Optional package roots under project_root (comma-separated). Empty = whole project.
    topo_package_roots: str = ""
    # Cap methods for pilot runs (0 = no cap).
    topo_max_methods: int = 0
    # When false: Method nodes skip summary/data_flow/table_field_* (keep in MySQL).
    neo4j_store_narratives: bool = False
    graph_max_depth: int = 6
    tagging_enabled: bool = True
    # Scoped var-resolve bags (method_var_resolve + Neo4j CALLS props) for change_impact.
    # Lazy ensure on first change_impact/explore; optional background sync at startup if empty.
    var_resolve_auto_ensure: bool = True
    var_resolve_startup_ensure: bool = True
    var_resolve_default_entry: str = ""
    # Comma-separated package roots under project_root (entry closure parse scope).
    var_resolve_package_roots: str = "loan-service/src/main/java"
    var_resolve_hop: int = 2
    # Entry bag present + at least this many project rows ⇒ skip re-sync.
    var_resolve_min_bags: int = 5
    # Offline flow analysis after graph (pilot entries → important nodes + flow docs).
    flow_analysis_enabled: bool = True
    # JSON list of pilot entry matchers; empty → no built-in product pilots.
    # Each item: {id, flow_key, flow_role, kinds, keywords}
    flow_pilot_entries_json: str = ""
    # Max chars of a Java file sent to LLM for structure parse.
    parse_ai_max_chars: int = 22000

    # LangSmith: set LANGSMITH_TRACING=true + LANGSMITH_API_KEY in .env
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "business-code-agent"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    hf_home: str = Field(default=str(ROOT_DIR / ".cache" / "huggingface"))
    transformers_cache: str = Field(default=str(ROOT_DIR / ".cache" / "huggingface"))
    torch_home: str = Field(default=str(ROOT_DIR / ".cache" / "torch"))

    config_yaml: str = Field(default=str(ROOT_DIR / "configs" / "default.yaml"))

    @property
    def project_root_path(self) -> Path:
        path = Path(self.project_root)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path.resolve()

    @property
    def workspace_path(self) -> Path:
        path = Path(self.workspace_dir)
        if not path.is_absolute():
            path = ROOT_DIR / path
        return path.resolve()

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    def load_yaml(self) -> dict:
        cfg_path = Path(self.config_yaml)
        if not cfg_path.exists():
            return {}
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}


@lru_cache
def get_settings() -> Settings:
    return Settings()
