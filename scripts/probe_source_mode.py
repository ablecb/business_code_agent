#!/usr/bin/env python
"""Probe: run_chat in middleware-free mode (memory session + EXPLORE_BACKEND=source).

No MySQL/Neo4j/Milvus needed — the agent explores the Java project via
filesystem tools directly (source backend). Keeps real LLM from .env.

Usage:
    python scripts/probe_source_mode.py "放款流程什么样？从入口到下游回调串一下"
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must be set before any app module reads settings.
os.environ["EXPLORE_BACKEND"] = "source"
os.environ.setdefault("LLM_LOG_PROMPTS", "false")


def _setup() -> None:
    from app.agent import session_store

    session_store._USE_MEMORY = True  # noqa: SLF001 — memory sessions, no MySQL

    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    s = get_settings()
    print(
        f"[setup] llm={s.llm_provider}/{s.llm_model} "
        f"embedding={s.embedding_provider} explore_backend={s.explore_backend} "
        f"pev={s.agent_use_pev} fallback={s.agent_pev_fallback}",
        flush=True,
    )


async def main() -> int:
    q = sys.argv[1] if len(sys.argv) > 1 else "放款流程什么样？从入口到下游回调串一下"
    _setup()
    from app.agent.graph_agent import run_chat

    import time

    t0 = time.time()
    try:
        result = await run_chat(q, project_id="demo-loan", session_id="probe-source-1")
    except Exception as e:  # noqa: BLE001
        print(f"[EXC] {type(e).__name__}: {e}", flush=True)
        return 1
    dt = time.time() - t0
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    answer = str(data.get("answer") or data.get("reply") or "")
    print(f"\n[elapsed] {dt:.1f}s  intent={data.get('intent') or data.get('goal') or ''}", flush=True)
    print(f"[answer_head]\n{answer[:1600]}", flush=True)
    return 0 if answer.strip() else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))