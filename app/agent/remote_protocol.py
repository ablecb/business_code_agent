"""WebSocket protocol for client-side remote agent execution."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Structured whitelist + general shell (shell always UI-confirmed).
ActionType = Literal[
    "read_file",
    "list_dir",
    "write_file",
    "apply_patch",
    "open_file",
    "maven_install",
    "maven_compile",
    "run_sql",
    "run_cypher",
    "milvus_search",
    "milvus_mutate",
    "open_browser",
    "docker_compose_up",
    "start_app",
    "stop_app",
    "http_smoke",
    "sys_disk_overview",
    "sys_scan_heavy",
    "sys_cleanup",
    "run_shell",
    "run_powershell",
    "git_status",
    "git_branch",
    "git_checkout",
    "git_pull",
    "git_merge",
    "git_commit",
    "git_push",
    "ask_user",
    "done",
    "fail",
]

ALLOWED_ACTIONS: frozenset[str] = frozenset(
    {
        "read_file",
        "list_dir",
        "write_file",
        "apply_patch",
        "open_file",
        "maven_install",
        "maven_compile",
        "run_sql",
        "run_cypher",
        "milvus_search",
        "milvus_mutate",
        "open_browser",
        "docker_compose_up",
        "start_app",
        "stop_app",
        "http_smoke",
        "sys_disk_overview",
        "sys_scan_heavy",
        "sys_cleanup",
        "run_shell",
        "run_powershell",
        "git_status",
        "git_branch",
        "git_checkout",
        "git_pull",
        "git_merge",
        "git_commit",
        "git_push",
        "ask_user",
        "done",
        "fail",
    }
)

RunStatus = Literal[
    "idle",
    "planning",
    "waiting_client",
    "waiting_confirm",
    "observing",
    "repairing",
    "succeeded",
    "failed",
    "stopped",
]

TERMINAL_STATUSES = frozenset({"succeeded", "failed", "stopped"})

DEFAULT_SUCCESS_PATTERN = r"Started \S+Application"
DEFAULT_MAX_STEPS = 40
DEFAULT_MAX_REPAIRS = 8
MAX_LOG_CHARS = 24_000
MAX_FILE_CHARS = 80_000


class PatchReplacement(BaseModel):
    old: str = ""
    new: str = ""


class ClientAction(BaseModel):
    model_config = {"extra": "allow"}

    id: str
    type: ActionType
    path: str | None = None
    content: str | None = None
    replacements: list[PatchReplacement] = Field(default_factory=list)
    sql: str | None = None
    mysql: dict[str, Any] | None = None
    # Neo4j / Milvus connectors
    query: str | None = None
    params: dict[str, Any] | None = None
    collection: str | None = None
    filter: str | None = None
    expr: str | None = None
    limit: int | None = None
    op: str | None = None  # milvus_mutate: upsert | delete
    rows: list[dict[str, Any]] = Field(default_factory=list)
    embeddings: list[list[float]] = Field(default_factory=list)
    # open_browser
    browser: str | None = None  # chrome | default
    note: str | None = None
    wait_ms: int | None = None
    # http_smoke / open_browser
    url: str | None = None
    urls: list[str] = Field(default_factory=list)
    method: str | None = None
    expect_status: int | None = None
    expect_body_regex: str | None = None
    headers: dict[str, Any] = Field(default_factory=dict)
    body: str | None = None
    drive: str | None = None
    ids: list[str] = Field(default_factory=list)  # sys_cleanup / milvus delete ids
    catalog_json: str | None = None
    options: list[dict[str, Any]] = Field(default_factory=list)
    reply: str | None = None
    # General command execution
    command: str | None = None
    argv: list[str] = Field(default_factory=list)
    cwd: str | None = None  # relative to project root, or "." 
    timeout_ms: int | None = None
    # Git structured fields (executor builds safe argv)
    branch: str | None = None
    ref: str | None = None
    message: str | None = None
    remote: str | None = None
    force: bool = False
    # Policy / UI (server annotates; client enforces confirm)
    needs_confirm: bool | None = None
    policy_tier: str | None = None
    ui_title: str | None = None
    ui_preview: str | None = None


class ActionResult(BaseModel):
    action_id: str
    type: ActionType | str
    ok: bool
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    content: str | None = None
    path: str | None = None
    elapsed_ms: int | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RunStep(BaseModel):
    action: ClientAction
    result: ActionResult | None = None
    ts: float | None = None


class RemoteRunCreate(BaseModel):
    goal: str = "把项目依赖装好并启动成功"
    project_id: str | None = None
    local_root_hint: str | None = None
    max_steps: int = DEFAULT_MAX_STEPS
    max_repairs: int = DEFAULT_MAX_REPAIRS
    success_pattern: str = DEFAULT_SUCCESS_PATTERN
    allow_write: bool = True
    allow_install: bool = True
    allow_process: bool = True
    allow_docker: bool = True
    allow_shell: bool = True
    allow_git: bool = True


class RemoteRunResponse(BaseModel):
    id: str
    status: RunStatus
    goal: str
    project_id: str | None = None
    local_root_hint: str | None = None
    step_count: int = 0
    repair_count: int = 0
    max_steps: int = DEFAULT_MAX_STEPS
    max_repairs: int = DEFAULT_MAX_REPAIRS
    message: str | None = None
    steps: list[RunStep] = Field(default_factory=list)
    created_at: float | None = None
    updated_at: float | None = None


class WsServerMessage(BaseModel):
    type: Literal[
        "hello",
        "run_status",
        "action",
        "run_finished",
        "error",
        "log_ack",
        "tool_event",
    ]
    run_id: str
    status: RunStatus | None = None
    message: str | None = None
    action: ClientAction | None = None
    run: RemoteRunResponse | None = None
    # tool_event payload for interleaved chat UI
    event: str | None = None  # tool_call | tool_log | tool_result
    result: ActionResult | None = None
    chunk: str | None = None


class WsClientMessage(BaseModel):
    type: Literal["action_result", "log_chunk", "client_ready", "stop"]
    run_id: str | None = None
    result: ActionResult | None = None
    chunk: str | None = None
    stream: Literal["stdout", "stderr"] | None = None
    action_id: str | None = None
