"""Conversation compaction — condense old turns into a summary instead of dropping them.

Borrowed from DeepSeek Harness's compaction idea and adapted to this codebase:
when a conversation grows past a threshold, the middle turns are handed to the
LLM to be condensed into one system-note summary; the system prompt, the first
user message and the most recent turns stay verbatim (keeps answerable context
without unbounded token growth — important for local 32B models with limited
context windows).

Message shape matches the rest of the agent: ``[{"role": ..., "content": ...}]``.
"""
from __future__ import annotations

import logging
from typing import Any

from app.agent.prompts import load_prompt
from app.core.config import get_settings

logger = logging.getLogger("app.agent.compaction")

_COMPACTION_PROMPT = load_prompt("compaction")


def should_compact(messages: list[dict[str, Any]], *, threshold: int | None = None) -> bool:
    """True when the conversation is long enough to warrant compaction.

    Requires at least 3 messages (system + user + assistant) beyond the tail
    that would otherwise be dropped.
    """
    n = len(messages or [])
    thr = threshold if threshold is not None else get_settings().compaction_threshold
    if n <= thr:
        return False
    keep_tail = get_settings().compaction_keep_tail
    # head = system + first user (2), tail = keep_tail → need >2+keep_tail to compress.
    return n > 2 + keep_tail


def _role_line(m: dict[str, Any]) -> str:
    role = str(m.get("role") or "user")
    content = str(m.get("content") or "").strip()
    if not content:
        return ""
    return f"{role}: {content}"


async def compact_messages(
    llm: Any,
    messages: list[dict[str, Any]],
    *,
    keep_tail: int | None = None,
    threshold: int | None = None,
) -> list[dict[str, Any]] | None:
    """Return a compacted message list, or None when compaction is not applicable.

    Layout of the returned list::

        [system, {role: system, content: "此前对话摘要：…"}, first_user, *tail]

    The middle turns (everything after the first user message and before the
    tail) are summarized. On any LLM failure we return None so the caller can
    fall back to plain truncation — compaction must never break the agent loop.
    """
    if not messages:
        return None
    if not should_compact(messages, threshold=threshold):
        return None
    keep_tail = keep_tail if keep_tail is not None else get_settings().compaction_keep_tail
    keep_tail = max(2, int(keep_tail))

    # head = system + first user message (keep verbatim for grounding)
    head = messages[:2]
    if not head or head[0].get("role") != "system":
        head = []
    # tail = most recent messages (verbatim)
    tail = messages[-keep_tail:]
    # middle = everything between head and tail (the compressible history)
    mid_start = len(head)
    mid_end = max(mid_start, len(messages) - keep_tail)
    middle = messages[mid_start:mid_end]
    if not middle:
        return None

    block = "\n".join(line for line in (_role_line(m) for m in middle) if line)
    if not block.strip():
        return None

    user_msg = _COMPACTION_PROMPT.replace("{block}", block)
    try:
        summary = await llm.chat(
            [
                {"role": "system", "content": "你是会话历史压缩器。"},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=600,
        )
    except Exception:
        logger.warning("compaction LLM call failed; falling back to truncation", exc_info=True)
        return None
    summary = (summary or "").strip()
    if not summary:
        return None

    note = {"role": "system", "content": f"此前对话摘要：{summary}"}
    return [*head, note, *tail]


async def maybe_compact(
    llm: Any,
    messages: list[dict[str, Any]],
) -> bool:
    """In-place compaction when enabled and applicable. Returns True if compacted."""
    try:
        enabled = bool(get_settings().compaction_enabled)
    except Exception:
        enabled = False
    if not enabled:
        return False
    compacted = await compact_messages(llm, messages)
    if not compacted:
        return False
    messages[:] = compacted
    logger.info("compaction applied: %d messages -> %d", len(messages), len(compacted))
    return True
