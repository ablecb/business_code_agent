from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings
from app.core.utils import truncate

logger = logging.getLogger("app.llm")


def _enabled() -> bool:
    try:
        return bool(get_settings().llm_log_prompts)
    except Exception:
        return False


def log_chat_request(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> None:
    if not _enabled():
        return
    limit = get_settings().llm_log_max_chars
    logger.info(
        "LLM REQUEST provider=%s model=%s temp=%s max_tokens=%s messages=%s",
        provider,
        model,
        temperature,
        max_tokens,
        len(messages),
    )
    for i, m in enumerate(messages):
        role = m.get("role", "?")
        content = truncate(str(m.get("content") or ""), limit)
        logger.info("LLM MSG[%s] role=%s\n%s", i, role, content)


def log_chat_waiting(provider: str, model: str) -> None:
    if not _enabled():
        return
    logger.info("LLM WAITING provider=%s model=%s (HTTP in flight…)", provider, model)


def log_chat_response(
    provider: str,
    model: str,
    content: str,
    *,
    elapsed_s: float | None = None,
) -> None:
    if not _enabled():
        return
    limit = get_settings().llm_log_max_chars
    timing = f" elapsed={elapsed_s:.1f}s" if elapsed_s is not None else ""
    body = truncate(content or "", limit) or "(empty content)"
    logger.info(
        "LLM RESPONSE provider=%s model=%s%s\n%s",
        provider,
        model,
        timing,
        body,
    )
