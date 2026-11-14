#!/usr/bin/env python
"""End-to-end smoke: parse sample file locally + optional API chat."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def demo_parse() -> None:
    from app.core.config import get_settings
    from app.parser.java_parser import parse_java_file
    from app.parser.scanner import scan_java_files

    settings = get_settings()
    root = settings.project_root_path
    files = scan_java_files(root)
    print(f"java files: {len(files)}")
    sample = None
    for f in files:
        if "Controller" in f.name or "Listener" in f.name:
            sample = f
            break
    sample = sample or (files[0] if files else None)
    if not sample:
        print("no java files found")
        return
    result = parse_java_file(sample, root)
    print(f"parsed {sample.name}: types={len(result.types)}")
    for t in result.types[:3]:
        print(f"  - {t.symbol_kind} {t.fqcn} methods={len(t.methods)}")
        for m in t.methods[:5]:
            anns = [a.name for a in m.annotations]
            print(f"      #{m.name} anns={anns} calls={len(m.calls)}")


async def demo_chat(message: str) -> None:
    import os

    os.environ["LLM_PROVIDER"] = "mock"
    os.environ["EMBEDDING_PROVIDER"] = "hash"
    # force memory session (no mysql required)
    from app.agent import session_store

    session_store._USE_MEMORY = True

    from app.core.config import get_settings
    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()

    from app.agent.graph_agent import run_chat

    resp = await run_chat(message)
    print(json.dumps(resp.model_dump(), ensure_ascii=False, indent=2)[:4000])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat", default="")
    parser.add_argument("--skip-parse", action="store_true")
    args = parser.parse_args()
    if not args.skip_parse:
        demo_parse()
    if args.chat:
        asyncio.run(demo_chat(args.chat))


if __name__ == "__main__":
    main()
