"""Prompt loading: system prompts live as plain-text files under configs/prompts/.

Edit the .txt files to tune behavior without touching code. Keep prompts
UTF-8; trailing whitespace is stripped on load. Dynamic parts (e.g. the
tool catalog text) are appended by the caller, never stored here.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from app.core.config import ROOT_DIR

logger = logging.getLogger("app.agent.prompts")

PROMPTS_DIR = ROOT_DIR / "configs" / "prompts"


@lru_cache(maxsize=64)
def load_prompt(name: str) -> str:
    """Load a prompt template by file name (no extension needed).

    Raises FileNotFoundError with a helpful message when the file is
    missing so a broken deployment fails loudly instead of silently
    degrading prompts.
    """
    path = _prompt_path(name)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Prompt file missing: {path}. "
            f"Restore it from the repo or re-run install (configs/prompts)."
        ) from None
    if not text.strip():
        logger.warning("Prompt file %s is empty; returning empty prompt", path)
    return text.strip()


def prompt_path(name: str) -> Path:
    """Resolve the prompt file path for a prompt name (no extension needed)."""
    return _prompt_path(name)


def _prompt_path(name: str) -> Path:
    if name.endswith(".txt"):
        path = PROMPTS_DIR / name
    else:
        path = PROMPTS_DIR / f"{name}.txt"
    return path
