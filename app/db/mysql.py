from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Generator
from urllib.parse import quote_plus

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings
from app.models.orm import Base

_engine = None
_SessionLocal = None
_init_lock = threading.Lock()
_db_ready = False


def _ensure_database() -> None:
    settings = get_settings()
    user = quote_plus(settings.mysql_user)
    password = quote_plus(settings.mysql_password)
    server_url = (
        f"mysql+pymysql://{user}:{password}"
        f"@{settings.mysql_host}:{settings.mysql_port}/?charset=utf8mb4"
    )
    tmp = create_engine(server_url, future=True, connect_args={"connect_timeout": 3})
    try:
        with tmp.connect() as conn:
            conn.execute(
                text(
                    f"CREATE DATABASE IF NOT EXISTS `{settings.mysql_database}` "
                    "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
                )
            )
            conn.commit()
    finally:
        tmp.dispose()


def get_engine():
    global _engine, _SessionLocal
    if _engine is None:
        settings = get_settings()
        try:
            _ensure_database()
        except Exception:
            # may already exist / permission limited; create_all will surface errors
            pass
        _engine = create_engine(
            settings.mysql_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            future=True,
            echo=settings.mysql_echo,
            # Avoid hanging forever when MySQL is blocked on metadata locks.
            connect_args={
                "connect_timeout": 5,
                "read_timeout": 30,
                "write_timeout": 30,
            },
        )
        _SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False)
    return _engine


def _raw_code_needs_widen(conn) -> bool:
    """Return True only when symbol.raw_code exists and is narrower than MEDIUMTEXT."""
    row = conn.execute(
        text(
            """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'symbol'
              AND COLUMN_NAME = 'raw_code'
            """
        )
    ).first()
    if not row:
        return False
    data_type = (row[0] or "").lower()
    # MEDIUMTEXT / LONGTEXT are already wide enough.
    if data_type in {"mediumtext", "longtext"}:
        return False
    return True


def _ensure_symbol_raw_code_width(conn) -> None:
    if not _raw_code_needs_widen(conn):
        return
    # Only run DDL when actually needed — repeated ALTER during UI poll causes MySQL 1684.
    conn.execute(text("ALTER TABLE symbol MODIFY COLUMN raw_code MEDIUMTEXT NULL"))
    conn.commit()


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table
              AND COLUMN_NAME = :column
            LIMIT 1
            """
        ),
        {"table": table, "column": column},
    ).first()
    return bool(row)


def _ensure_ingest_job_pid(conn) -> None:
    """Add ingest_job.pid for orphan detection (existing DBs lack the column)."""
    if _column_exists(conn, "ingest_job", "pid"):
        return
    try:
        conn.execute(text("ALTER TABLE ingest_job ADD COLUMN pid INT NULL"))
        conn.commit()
    except Exception:
        conn.rollback()


def _index_exists(conn, table: str, index_name: str) -> bool:
    row = conn.execute(
        text(
            """
            SELECT 1
            FROM information_schema.STATISTICS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = :table
              AND INDEX_NAME = :index_name
            LIMIT 1
            """
        ),
        {"table": table, "index_name": index_name},
    ).first()
    return bool(row)


def _ensure_symbol_desc_dual_summary_columns(conn) -> None:
    """Add dual-perspective summary columns + widen summary if needed."""
    # Widen legacy TEXT summary so four-view payloads fit.
    row = conn.execute(
        text(
            """
            SELECT DATA_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = 'symbol_desc'
              AND COLUMN_NAME = 'summary'
            """
        )
    ).first()
    if row and (row[0] or "").lower() == "text":
        conn.execute(text("ALTER TABLE symbol_desc MODIFY COLUMN summary MEDIUMTEXT NOT NULL"))
        conn.commit()

    dual_cols = (
        "summary_business_clear",
        "summary_business_detail",
        "summary_dev_clear",
        "summary_dev_detail",
    )
    prev = "summary"
    for col in dual_cols:
        if _column_exists(conn, "symbol_desc", col):
            prev = col
            continue
        conn.execute(
            text(
                f"ALTER TABLE symbol_desc ADD COLUMN {col} MEDIUMTEXT NULL AFTER {prev}"
            )
        )
        conn.commit()
        prev = col

    if not _index_exists(conn, "symbol", "idx_symbol_module"):
        try:
            conn.execute(text("ALTER TABLE symbol ADD INDEX idx_symbol_module (module_name)"))
            conn.commit()
        except Exception:
            conn.rollback()


def init_db(*, force: bool = False) -> None:
    """Create tables once per process. Skip repeated DDL on every API poll."""
    global _db_ready
    if _db_ready and not force:
        return
    with _init_lock:
        if _db_ready and not force:
            return
        # Register all ORM tables on Base.metadata (incl. global KB).
        from app.models import orm as _orm  # noqa: F401

        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            try:
                _ensure_symbol_raw_code_width(conn)
            except Exception:
                conn.rollback()
            try:
                _ensure_symbol_desc_dual_summary_columns(conn)
            except Exception:
                conn.rollback()
            try:
                _ensure_ingest_job_pid(conn)
            except Exception:
                conn.rollback()
        _db_ready = True


@contextmanager
def get_session() -> Generator[Session, None, None]:
    get_engine()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
