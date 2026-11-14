"""Smoke: cold-start quota code / logic without prior flow turn.

Usage:
  python tests/smoke_loan_cold_start_quota_code.py
"""
from __future__ import annotations

import asyncio
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT = ROOT / "data" / "loan_cold_start_quota_code_smoke.txt"
PROJECT_ID = "demo-loan"


async def _ask(q: str, sid: str) -> tuple[str, dict]:
    from app.agent import session_store
    from app.agent.graph_agent import run_chat

    result = await run_chat(
        q,
        project_id=PROJECT_ID,
        session_id=sid,
    )
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    answer = str(data.get("answer") or data.get("reply") or "")
    ctx = session_store.get_context(sid) or {}
    return answer, ctx


async def main() -> int:
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    lines: list[str] = [f"project_id={PROJECT_ID}", ""]
    all_ok = True

    # A) cold-start show-code
    sid_a = f"smoke-cold-code-{uuid.uuid4().hex[:10]}"
    q_code = "放款接口中额度校验部分代码什么样"
    ans_a, ctx_a = await _ask(q_code, sid_a)
    ok_a = bool(
        re.search(r"(?i)quotaCheck", ans_a)
        and ("```java" in ans_a or "boolean quotaCheck" in ans_a)
        and ctx_a.get("last_gate")
        and any("quotaCheck" in str(s) for s in (ctx_a.get("last_symbols") or []))
    )
    lines.append(f"A session={sid_a}")
    lines.append(f"Q: {q_code}")
    lines.append(
        f"ok={ok_a} last_gate={ctx_a.get('last_gate')} "
        f"last_symbols={ctx_a.get('last_symbols')} answer_len={len(ans_a)}"
    )
    lines.append(f"answer_head:\n{(ans_a or '')[:900]}\n")
    all_ok = all_ok and ok_a

    # Follow-up after cold-start code
    q_fu = "那机构查询这一块逻辑什么样的"
    ans_fu, ctx_fu = await _ask(q_fu, sid_a)
    ok_fu = bool(
        re.search(r"(?i)queryProductVersion|机构查询", ans_fu)
        and ctx_fu.get("last_gate")
    )
    lines.append(f"A2 follow-up: {q_fu}")
    lines.append(
        f"ok={ok_fu} last_gate={ctx_fu.get('last_gate')} answer_len={len(ans_fu)}"
    )
    lines.append(f"answer_head:\n{(ans_fu or '')[:600]}\n")
    all_ok = all_ok and ok_fu

    # B) cold-start logic-only
    sid_b = f"smoke-cold-logic-{uuid.uuid4().hex[:10]}"
    q_logic = "放款接口中额度校验逻辑什么样"
    ans_b, ctx_b = await _ask(q_logic, sid_b)
    ok_b = bool(
        re.search(r"(?i)quotaCheck", ans_b)
        and ctx_b.get("last_gate")
        and "```java" not in ans_b
    )
    q_code2 = "我想看看对应的代码"
    ans_b2, ctx_b2 = await _ask(q_code2, sid_b)
    ok_b2 = bool(
        re.search(r"(?i)quotaCheck", ans_b2)
        and ("```java" in ans_b2 or "boolean quotaCheck" in ans_b2)
    )
    lines.append(f"B session={sid_b}")
    lines.append(f"Q: {q_logic}")
    lines.append(
        f"ok={ok_b} last_gate={ctx_b.get('last_gate')} "
        f"last_symbols={ctx_b.get('last_symbols')} answer_len={len(ans_b)}"
    )
    lines.append(f"answer_head:\n{(ans_b or '')[:900]}\n")
    lines.append(f"B2: {q_code2}")
    lines.append(
        f"ok={ok_b2} last_gate={ctx_b2.get('last_gate')} answer_len={len(ans_b2)}"
    )
    lines.append(f"answer_head:\n{(ans_b2 or '')[:900]}\n")
    all_ok = all_ok and ok_b and ok_b2

    lines.append(f"ALL_OK={all_ok}")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {REPORT}")
    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
