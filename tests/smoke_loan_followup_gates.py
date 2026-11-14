"""Multi-turn smoke: 放款 overview then gate dig-ins (额度/风控/开关).

Expect Q2+ to stay on WithdrawalsController#withdrawals CALLS closure
(quotaCheck* / risk* / loanSwitch), not AccClient limit APIs.

Usage:
  python tests/smoke_loan_followup_gates.py
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

REPORT = ROOT / "data" / "loan_followup_gate_smoke.txt"
PROJECT_ID = "demo-loan"

QUESTIONS = [
    "放款流程是怎样的",
    "额度校验是怎么校验的",
    "风控是怎么做的",
    "loanSwitch干嘛的",
]

# Per-question expectations (substring / regex).
EXPECT = [
    {
        "any": ["放款", "withdrawals", "提现", "开关", "loanSwitch", "校验"],
        "forbid": ["AccClient"],
    },
    {
        "any": ["quotaCheck"],
        "forbid": ["AccClient#", "AmountController"],
    },
    {
        "any": ["risk", "Risk", "风控", "asynRisk"],
        "forbid": ["AccClient#"],
    },
    {
        "any": ["loanSwitch", "开关"],
        "forbid": ["AccClient#", "creditNotification"],
    },
]


def _judge(answer: str, spec: dict) -> tuple[bool, str]:
    a = answer or ""
    if not any(x in a for x in spec["any"]):
        return False, f"missing any of {spec['any']}"
    for bad in spec.get("forbid") or []:
        if bad in a:
            return False, f"forbidden hit: {bad}"
    if "评审提示" in a or re.search(r"[（(]补充说明[：:]", a):
        return False, "review footnote in answer body"
    return True, "ok"


async def main() -> int:
    from app.agent import session_store
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    sid = f"smoke-gate-{uuid.uuid4().hex[:10]}"
    lines: list[str] = [
        f"session_id={sid}",
        f"project_id={PROJECT_ID}",
        "",
    ]
    all_ok = True
    locked = None

    for i, q in enumerate(QUESTIONS):
        print(f"\n=== Q{i + 1}: {q} ===", flush=True)
        result = await run_chat(
            q,
            project_id=PROJECT_ID,
            session_id=sid,
        )
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        answer = str(data.get("answer") or data.get("reply") or "")
        ctx = session_store.get_context(sid)
        locked = ctx.get("last_class_method") or ctx.get("last_entry") or locked
        ok, reason = _judge(answer, EXPECT[i])
        # Q2+ must keep withdrawals lock
        lock_ok = True
        lock_note = ""
        if i >= 1:
            lock_ok = bool(
                locked
                and re.search(r"WithdrawalsController#withdrawals", str(locked), re.I)
            )
            if not lock_ok:
                lock_note = f" lock={locked!r}"
                ok = False
                reason = f"entry not locked to withdrawals;{lock_note} {reason}"
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] lock={locked} | {reason}", flush=True)
        print(answer[:800], flush=True)
        lines.append(f"=== Q{i + 1}: {q} ===")
        lines.append(f"status={status} reason={reason}")
        lines.append(f"last_class_method={locked}")
        lines.append(answer)
        lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REPORT}", flush=True)
    print("ALL_PASS" if all_ok else "ALL_FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
