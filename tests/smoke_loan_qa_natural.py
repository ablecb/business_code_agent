"""Quick smoke: 放款流程 natural business answer."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


async def main() -> int:
    from app.analysis.flow_neo4j_sync import get_follows_path
    from app.core.config import get_settings
    from app.db.mysql import init_db
    from app.agent.explore_agent import run_explore_agent

    get_settings.cache_clear()
    settings = get_settings()
    init_db()
    pid = settings.project_id
    q = "放款流程是什么样的，用业务语言说清楚"

    print("=== spine ===")
    fol = get_follows_path(pid, "WithdrawalsController#withdrawals", "loan_disburse")
    print("ok", fol.get("ok"), "source", fol.get("source"), "n", len(fol.get("steps") or []))
    for st in (fol.get("steps") or [])[:12]:
        print(f"  {st.get('step')}: {st.get('class_method')} | {(st.get('title') or '')[:40]}")

    print("\n=== answer ===")
    result = await run_explore_agent(q, project_id=pid)
    text = str(result.get("answer") or "")
    print(text)
    out = ROOT / "data" / "loan_qa_natural_smoke.txt"
    out.write_text(
        f"source={fol.get('source')}\nsteps={len(fol.get('steps') or [])}\n\n{text}\n",
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
