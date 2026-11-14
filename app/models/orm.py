from __future__ import annotations

from datetime import datetime

from sqlalchemy import BigInteger, DateTime, Float, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import MEDIUMTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ProjectORM(Base):
    __tablename__ = "project"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    root_path: Mapped[str] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ProjectMetaORM(Base):
    """AI project overview / tech-stack metadata (one row per project)."""

    __tablename__ = "project_meta"

    project_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tech_stack: Mapped[str | None] = mapped_column(Text, nullable=True)
    overview_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class IngestJobORM(Base):
    __tablename__ = "ingest_job"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_files: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Owning OS process id; lets the backend tell "orphaned on restart"
    # (pid dead) from "CLI parse still running in another process" (pid alive).
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class SymbolORM(Base):
    __tablename__ = "symbol"
    __table_args__ = (
        # MySQL utf8mb4 unique key length limit — use prefix lengths
        Index(
            "uk_symbol",
            "project_id",
            "fqcn",
            "method_sig",
            unique=True,
            mysql_length={"fqcn": 180, "method_sig": 180},
        ),
        Index("idx_symbol_fqcn", "fqcn", mysql_length=180),
        Index("idx_symbol_module", "module_name"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    module_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    kind: Mapped[str] = mapped_column(String(64), index=True)
    fqcn: Mapped[str] = mapped_column(String(512))
    simple_name: Mapped[str] = mapped_column(String(256))
    method_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    method_sig: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    file_path: Mapped[str] = mapped_column(String(1024))
    start_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    end_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    annotations: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    modifiers: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    parent_fqcn: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # full method source (javadoc+ann+body); TEXT(64KB) is too small for large methods
    raw_code: Mapped[str | None] = mapped_column(
        MEDIUMTEXT().with_variant(Text, "sqlite"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TagDictORM(Base):
    __tablename__ = "tag_dict"
    __table_args__ = (UniqueConstraint("project_id", "tag", name="uk_tag"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    tag: Mapped[str] = mapped_column(String(128))
    category: Mapped[str | None] = mapped_column(String(64), nullable=True)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SymbolTagORM(Base):
    __tablename__ = "symbol_tag"
    __table_args__ = (UniqueConstraint("symbol_id", "tag_id", name="uk_symbol_tag"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol_id: Mapped[str] = mapped_column(String(64))
    tag_id: Mapped[int] = mapped_column(BigInteger, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class SymbolDescORM(Base):
    __tablename__ = "symbol_desc"
    __table_args__ = (UniqueConstraint("symbol_id", name="uk_symbol_desc"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    symbol_id: Mapped[str] = mapped_column(String(64))
    # Combined short summary for legacy/search; dual views live in dedicated columns.
    summary: Mapped[str] = mapped_column(MEDIUMTEXT().with_variant(Text, "sqlite"))
    summary_business_clear: Mapped[str | None] = mapped_column(
        MEDIUMTEXT().with_variant(Text, "sqlite"), nullable=True
    )
    summary_business_detail: Mapped[str | None] = mapped_column(
        MEDIUMTEXT().with_variant(Text, "sqlite"), nullable=True
    )
    summary_dev_clear: Mapped[str | None] = mapped_column(
        MEDIUMTEXT().with_variant(Text, "sqlite"), nullable=True
    )
    summary_dev_detail: Mapped[str | None] = mapped_column(
        MEDIUMTEXT().with_variant(Text, "sqlite"), nullable=True
    )
    business_domain: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    extra_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class MethodVarResolveORM(Base):
    """Per-method AST var-resolve bag (locals + call_sites + receivers).

    Production source of truth for precise setXxx / receiver typing; Neo4j
    CALLS edges carry a slim subset of the same facts as relationship props.
    """

    __tablename__ = "method_var_resolve"
    __table_args__ = (
        UniqueConstraint("project_id", "class_method", name="uk_mvr_project_cm"),
        Index("idx_mvr_project", "project_id"),
        Index("idx_mvr_symbol", "symbol_id"),
        Index(
            "idx_mvr_class_method",
            "class_method",
            mysql_length={"class_method": 180},
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64))
    class_method: Mapped[str] = mapped_column(String(512))
    symbol_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ChatSessionORM(Base):
    __tablename__ = "chat_session"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    project_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    context_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class ChatMessageORM(Base):
    __tablename__ = "chat_message"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(32))
    content: Mapped[str] = mapped_column(Text)
    meta_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class KbDocumentORM(Base):
    """Global knowledge-base document (API specs, notes, …) — not project-scoped."""

    __tablename__ = "kb_document"
    __table_args__ = (Index("idx_kb_doc_status", "status"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(256))
    filename: Mapped[str] = mapped_column(String(256))
    kind: Mapped[str] = mapped_column(String(16))  # md|txt|pdf
    storage_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="queued")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class KbChunkORM(Base):
    """Embedded text chunk for global KB retrieval (embedding JSON in MySQL)."""

    __tablename__ = "kb_chunk"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uk_kb_chunk"),
        Index("idx_kb_chunk_doc", "document_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64))
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)
    text: Mapped[str] = mapped_column(MEDIUMTEXT)
    embedding_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChatFeedbackORM(Base):
    """User like/dislike on an assistant reply, with full conversation snapshot."""

    __tablename__ = "chat_feedback"
    __table_args__ = (
        UniqueConstraint("session_id", "assistant_message_id", name="uk_feedback_msg"),
        Index("idx_cf_project", "project_id"),
        Index("idx_cf_session", "session_id"),
        Index("idx_cf_created", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    project_id: Mapped[str] = mapped_column(String(64))
    session_id: Mapped[str] = mapped_column(String(64))
    assistant_message_id: Mapped[int] = mapped_column(BigInteger)
    vote: Mapped[str] = mapped_column(String(16))  # up | down
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    question_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    reply_excerpt: Mapped[str | None] = mapped_column(Text, nullable=True)
    conversation_json: Mapped[list | dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class GoldenQAORM(Base):
    """离线黄金问答库：ingest 期用（内网最强的）模型把项目咀嚼成高置信问答对。

    运行时「命中即答」：检索问题与存储问题的语义+关键词相似度，高于阈值直接
    返回存储答案，避免让 qwen3-32b 在长链推理里丢准确率（方案一：重推理离线化）。
    """

    __tablename__ = "golden_qa"
    __table_args__ = (
        Index("idx_gq_project", "project_id"),
        Index("idx_gq_entry", "entry_cm", mysql_length={"entry_cm": 180}),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    project_id: Mapped[str] = mapped_column(String(64))
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(MEDIUMTEXT().with_variant(Text, "sqlite"))
    entry_cm: Mapped[str | None] = mapped_column(String(512), nullable=True)
    # 来源: consensus（自一致性投票）| flow_doc（直接转写）| manual |
    #       manual_feedback（人工正向反馈自动回填）
    source: Mapped[str] = mapped_column(String(32), default="consensus")
    # 自一致性投票得分（0~1，多份答案两两相似度均值）
    votes: Mapped[float] = mapped_column(Float, default=0.0)
    tags_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    samples_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    # 问题句向量（与 EMBEDDING_* 同一模型），运行时语义检索用
    question_embedding_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class TableDictORM(Base):
    """表结构字典：离线从 Entity（@TableName + 字段注释）解析，供 SQL 生成与表关联问答。"""

    __tablename__ = "table_dict"
    __table_args__ = (
        UniqueConstraint("project_id", "table_name", name="uk_td_project_table"),
        Index("idx_td_project", "project_id"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    project_id: Mapped[str] = mapped_column(String(64))
    table_name: Mapped[str] = mapped_column(String(128))
    entity_fqcn: Mapped[str | None] = mapped_column(String(512), nullable=True)
    entity_class: Mapped[str | None] = mapped_column(String(256), nullable=True)
    # 中文名（@ApiModel / 类 javadoc / Controller @Api tag 推导），可能为空
    name_cn: Mapped[str | None] = mapped_column(String(256), nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # 字段列表: [{name, type, comment}]
    fields_json: Mapped[list | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
