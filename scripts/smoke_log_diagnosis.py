#!/usr/bin/env python
"""In-process smoke for the log-diagnosis pipeline (no server restart needed).

Usage: python scripts/smoke_log_diagnosis.py S04 S13 [S15 ...]
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())


def load_scenario(sid: str) -> dict:
    with open("data/log_diag_battery/scenarios.json", encoding="utf-8") as f:
        data = json.load(f)
    for s in data["scenarios"]:
        if s["id"] == sid:
            return s
    raise SystemExit(f"scenario {sid} not found")


async def main() -> None:
    from app.agent.log_diagnosis import (
        extract_anchors,
        has_branch_signal,
        run_log_diagnosis,
    )

    sids = sys.argv[1:] or ["S13"]
    for sid in sids:
        sc = load_scenario(sid)
        msg = str(sc["log"])
        anchors = extract_anchors(msg)
        t0 = time.perf_counter()
        result = await run_log_diagnosis(msg, project_id="demo-loan")
        wall = round(time.perf_counter() - t0, 1)
        print("\n" + "=" * 70)
        print(f"[{sid}] {sc['title']}  wall={wall}s  branch={result['branch_hit']}")
        print(f"anchors: {anchors}")
        print(f"candidates: {[c.get('fqcn') for c in result.get('candidates') or []][:6]}")
        print(f"evidence_parts: {len(result.get('evidence_parts') or [])}")
        print("-" * 70)
        print((result.get("reply") or "")[:2500])
        print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())