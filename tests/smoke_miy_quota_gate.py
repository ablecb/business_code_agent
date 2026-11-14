"""Focused smoke: 放款 → MIY额度校验.

Checks:
  - Mentions quotaCheckMIY / real logic (aqcClient / 额度)
  - Does NOT invent fake tables
  - Does NOT end with「未获取到 quotaCheckMIY 实现代码」

Usage:
  python tests/smoke_miy_quota_gate.py
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

REPORT = ROOT / "data" / "miy_quota_gate_smoke.txt"
PROJECT_ID = "demo-loan"

FAKE_TABLES = (
    "t_user_quota",
    "t_user_credit_limit",
    "t_product_config",
    "t_credit_line",
)

QUESTIONS = [
    "放款流程是怎样的",
    "MIY的额度校验具体什么样",
]


def _judge_miy(answer: str) -> tuple[bool, list[str]]:
    a = answer or ""
    issues: list[str] = []
    if not re.search(r"(?i)quotaCheckMIY", a):
        issues.append("missing quotaCheckMIY")
    if not re.search(r"(?i)aqcClient|loanSchemaQuery|可用额度|额度状态|availLimit|额度树", a):
        issues.append("missing real MIY logic cues")
    for t in FAKE_TABLES:
        if t in a:
            issues.append(f"invented table: {t}")
    if re.search(r"未获取到.{0,30}quotaCheckMIY.{0,20}实现", a) or re.search(
        r"未获取到.{0,40}具体实现代码", a
    ):
        issues.append("contradictory「未获取到实现代码」footer")
    if "评审提示" in a or re.search(r"[（(]补充说明[：:]", a):
        issues.append("review footnote in answer body")
    return (not issues), issues


async def main() -> int:
    from app.agent import session_store
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    sid = f"smoke-miy-{uuid.uuid4().hex[:10]}"
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
        if i == 0:
            ok = bool(answer.strip()) and bool(
                locked
                and re.search(r"WithdrawalsController#withdrawals", str(locked), re.I)
            )
            reason = "ok" if ok else f"lock={locked!r} or empty answer"
        else:
            ok, issues = _judge_miy(answer)
            reason = "ok" if ok else "; ".join(issues)
            if not (
                locked
                and re.search(r"WithdrawalsController#withdrawals", str(locked), re.I)
            ):
                ok = False
                reason = f"entry not locked ({locked!r}); {reason}"
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] lock={locked} | {reason}", flush=True)
        print(answer[:1200], flush=True)
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
