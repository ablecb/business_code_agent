from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    project_id: str | None = None
    path: str | None = None
    tagging: bool | None = None
    full_refresh: bool = False


class IngestJobResponse(BaseModel):
    id: str
    project_id: str
    status: str
    stage: str | None = None
    total_files: int = 0
    processed: int = 0
    message: str | None = None
    error: str | None = None


class SearchRequest(BaseModel):
    q: str
    project_id: str | None = None
    top_k: int = 12
    domain: str | None = None


class SymbolHit(BaseModel):
    symbol_id: str
    project_id: str
    fqcn: str
    method_sig: str | None = None
    kind: str
    file_path: str
    summary: str | None = None
    tags: list[str] = Field(default_factory=list)
    business_domain: str | None = None
    score: float | None = None


class GraphFlowRequest(BaseModel):
    entry: str
    project_id: str | None = None
    depth: int | None = None


class GraphNode(BaseModel):
    id: str
    labels: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    type: str
    start: str
    end: str
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphFlowResponse(BaseModel):
    entry: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    project_id: str | None = None
    # Cross-project Q&A: when set, retrieval merges these projects (primary = first or project_id).
    project_ids: list[str] = Field(default_factory=list)
    # When True with patches: do not write server workspace; create client patch-verify run instead.
    client_apply: bool = False
    # Desktop: allow server to command client via unified agent WS loop (LLM-routed).
    client_agent: bool = False
    # Deprecated: treated as client_agent=True.
    local_assist: bool = False
    drive: str | None = None
    # Structured client actions (option pick / verify / rollback / cleanup).
    action: Literal["select_option", "verify", "rollback", "cleanup_selected"] | None = None
    option_id: str | None = None
    option_ids: list[str] = Field(default_factory=list)
    cleanup_catalog: list[dict[str, Any]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    intent: str | None = None
    scope: list[dict[str, Any]] = Field(default_factory=list)
    patches: list[dict[str, Any]] = Field(default_factory=list)
    retrieved: list[SymbolHit] = Field(default_factory=list)
    # Server-side wall time from request accepted to reply ready (ms).
    elapsed_ms: int | None = None
    # DB id of the persisted assistant message (for feedback).
    assistant_message_id: int | None = None
    # Structured research plan (steps + status) for project explore Q&A.
    plan: dict[str, Any] | None = None
    # Completeness review from the second-pass reviewer AI.
    review: dict[str, Any] | None = None
    # Explore tool trail (search/get_symbol_code/finish) for smoke/debug.
    explore_steps: list[dict[str, Any]] = Field(default_factory=list)
    # Change-workflow session phase (idle/awaiting_option/verifying/...).
    phase: str | None = None
    # Multi-scheme options for user pick (options intent / awaiting_option).
    options: list[dict[str, Any]] = Field(default_factory=list)
    selected_option_id: str | None = None
    # Compile/test report from verify_loop.
    verify_report: dict[str, Any] | None = None
    # Projects used for this answer (may be multiple).
    project_ids: list[str] = Field(default_factory=list)
    # Desktop client should connect WS /agent/ws?run_id= to apply+verify locally.
    client_run_id: str | None = None
    # Server-side append log for this conversation (data/desktop_chat/<session>.log).
    log_file: str | None = None
    # Append-only JSONL trajectory (data/trajectories/<session>.jsonl).
    trajectory_file: str | None = None
    # Active harness runtime mode for this turn.
    runtime_mode: str | None = None


class ChatHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict[str, Any]]


class ChatFeedbackRequest(BaseModel):
    session_id: str
    assistant_message_id: int
    vote: Literal["up", "down"]
    project_id: str | None = None
    comment: str | None = None


class ChatFeedbackResponse(BaseModel):
    id: int
    session_id: str
    assistant_message_id: int
    vote: Literal["up", "down"]
    created_at: str | None = None
    updated: bool = False


class ChatFeedbackItem(BaseModel):
    id: int
    project_id: str
    session_id: str
    assistant_message_id: int
    vote: str
    comment: str | None = None
    question_excerpt: str | None = None
    reply_excerpt: str | None = None
    created_at: str | None = None


class ChatFeedbackListResponse(BaseModel):
    items: list[ChatFeedbackItem] = Field(default_factory=list)
    total: int = 0


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    llm_provider: str
    checks: dict[str, str] = Field(default_factory=dict)


class ProjectJobBrief(BaseModel):
    id: str
    status: str
    stage: str | None = None
    total_files: int = 0
    processed: int = 0
    message: str | None = None
    error: str | None = None
    # Wall-clock seconds from job created_at → updated_at (terminal jobs only).
    elapsed_sec: float | None = None


class ProjectResponse(BaseModel):
    id: str
    name: str
    root_path: str
    description: str | None = None
    status: str
    java_files: int = 0
    symbol_count: int = 0
    latest_job: ProjectJobBrief | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ProjectCreateRequest(BaseModel):
    name: str
    project_id: str | None = None
    path: str | None = None
    description: str | None = None


class ProjectParseRequest(BaseModel):
    tagging: bool | None = None
    full_refresh: bool = True
    # True: cancel any queued/running job and start a new parse (e.g. after server restart).
    force: bool = False

