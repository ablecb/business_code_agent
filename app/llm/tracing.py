from __future__ import annotations

import logging
import os
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

logger = logging.getLogger("app.llm")

F = TypeVar("F", bound=Callable[..., Any])


def setup_langsmith() -> None:
    """Configure LangSmith from app settings (safe no-op when disabled)."""
    from app.core.config import get_settings

    settings = get_settings()
    if not settings.langsmith_tracing:
        os.environ.pop("LANGSMITH_TRACING", None)
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        logger.info("LangSmith tracing disabled")
        return

    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    if settings.langsmith_endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint
    if settings.langsmith_project:
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project

    key_ok = bool(os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY"))
    if not key_ok:
        logger.warning(
            "LangSmith tracing enabled but LANGSMITH_API_KEY is empty — traces will not upload"
        )
    else:
        logger.info(
            "LangSmith tracing ON project=%s",
            settings.langsmith_project or "(default)",
        )


def traceable(name: str, *, run_type: str = "chain", **kwargs: Any) -> Callable[[F], F]:
    """LangSmith @traceable with graceful fallback when package/tracing unavailable."""

    def decorator(fn: F) -> F:
        try:
            from langsmith import traceable as ls_traceable

            return ls_traceable(name=name, run_type=run_type, **kwargs)(fn)  # type: ignore[return-value]
        except Exception:
            return fn

    return decorator
