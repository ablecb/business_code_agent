"""Four-turn smoke: flow → gate dig-in → show code → composite change plan.

Usage:
  python tests/smoke_loan_four_turn_drill.py
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

REPORT = ROOT / "data" / "loan_four_turn_drill_smoke.txt"
PROJECT_ID = "demo-loan"

QUESTIONS = [
    "放款流程是什么样的",
    "那额度校验这一块逻辑什么样的",
    "我想看看对应的代码",
    (
        "放款接口我想增加新入参，并落库，并在机构校验时判断是单笔单批时"
        "传递给外部系统，需要改动哪些"
    ),
]

EXPECT = [
    {
        "any": ["机构查询", "额度校验", "quotaCheck", "放款", "loanSwitch"],
        "forbid": ["AccClient#"],
        "ctx_has": ["last_class_method"],
    },
    {
        "any": ["quotaCheck"],
        "forbid": ["AccClient#", "AmountController"],
        "ctx_has": ["last_gate", "last_symbols"],
    },
    {
        "any": ["```java", "quotaCheck"],
        "forbid": ["不会编造源码"],
        "ctx_has": ["last_symbols"],
    },
    {
        "any": ["## 结论", "## 改动清单", "落库"],
        "forbid": ["协议同步组装（固定清单）"],
        "ctx_has": [],
    },
]


def _judge(answer: str, ctx: dict, spec: dict) -> tuple[bool, str]:
    a = answer or ""
    if not any(x in a for x in spec["any"]):
        return False, f"missing any of {spec['any']}"
    for bad in spec.get("forbid") or []:
        if bad in a:
            return False, f"forbidden hit: {bad}"
    for key in spec.get("ctx_has") or []:
        if not ctx.get(key):
            return False, f"missing session ctx.{key}"
    return True, "ok"


async def main() -> int:
    from app.agent import session_store
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    sid = f"smoke-4turn-{uuid.uuid4().hex[:10]}"
    lines: list[str] = [f"session_id={sid}", f"project_id={PROJECT_ID}", ""]
    all_ok = True

    for i, q in enumerate(QUESTIONS):
        print(f"\n=== Q{i + 1}: {q[:60]} ===", flush=True)
        result = await run_chat(
            q,
            project_id=PROJECT_ID,
            session_id=sid,
        )
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        answer = str(data.get("answer") or data.get("reply") or "")
        ctx = session_store.get_context(sid)
        ok, reason = _judge(answer, ctx, EXPECT[i])
        lock = ctx.get("last_class_method") or ctx.get("last_entry")
        lock_ok = True
        if i >= 1:
            lock_ok = bool(
                lock
                and re.search(r"WithdrawalsController#withdrawals", str(lock), re.I)
            )
        row_ok = ok and lock_ok
        all_ok = all_ok and row_ok
        lines.append(f"Q{i + 1}: {q}")
        lines.append(f"  ok={row_ok} reason={reason} lock_ok={lock_ok}")
        lines.append(f"  last_entry={lock}")
        lines.append(f"  last_gate={ctx.get('last_gate')}")
        lines.append(f"  last_symbols={ctx.get('last_symbols')}")
        lines.append(f"  answer_len={len(answer)}")
        lines.append(f"  answer_head={answer[:400].replace(chr(10), ' / ')}")
        lines.append("")
        print(
            f"  ok={row_ok} reason={reason} gate={ctx.get('last_gate')} "
            f"syms={ctx.get('last_symbols')}",
            flush=True,
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT}", flush=True)
    print("PASS" if all_ok else "FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
