#!/usr/bin/env python
"""Probe: how PEV entry resolution behaves for the 5 business domains.

Runs the same entry-resolution internals as the live pipeline and prints the
intermediate entry candidates / symbol-search results / facade guard outcome.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.db.mysql import init_db  # noqa: E402


def probe(question: str) -> None:
    from app.agent.pev_flow import (
        _apply_topic_ground_to_entries,
        _entry_is_real,
        _resolve_entry_via_symbol_search,
        _topic_ground_entry_candidates,
    )
    from app.agent.ask_retrieval import resolve_flow_entries_with_meta

    pid = get_settings().project_id
    print(f"\n===== {question} =====")
    grounded = _topic_ground_entry_candidates(pid, question)
    print("topic_ground phrase:", grounded.get("phrase"))
    print("topic_ground controllers:", grounded.get("controllers"))
    matches, meta = resolve_flow_entries_with_meta(pid, question)
    print("resolve_flow_entries matches:", [m for m in (matches or [])][:8])
    f, m2 = _apply_topic_ground_to_entries(
        pid, question, matches=list(matches or []), meta=list(meta or []), grounded=grounded
    )
    print("after topic_ground entries:", [x for x in (f or [])][:8])
    sym = _resolve_entry_via_symbol_search(pid, question)
    print("symbol_search result:", sym)
    for cm in [c for c in (f or [])[:3]]:
        print(f"  _entry_is_real({cm}) = {_entry_is_real(pid, cm)}")


def main() -> None:
    init_db()
    get_settings.cache_clear()
    for q in [
        "还款流程是什么样的",
        "放款流程什么样？从入口到下游回调串一下",
        "额度调整有哪些类型和策略",
        "结清证明申请接口 主要流程是什么",
        "授信通知怎么实现的",
        "授信通知接口的地址是什么",
        "放款接口的地址是什么",
        "还款结果怎么通知下游/渠道",
        "调额接口增加一个字段 batchNo 并透传给渠道，要改哪些",
        "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
    ]:
        try:
            probe(q)
        except Exception:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()