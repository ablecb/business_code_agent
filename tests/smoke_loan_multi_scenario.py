"""Multi-turn smoke: 放款 flow → MIY dig-in → 源码 → sudden gate switches.

Session A checks:
  1. Full-flow overview locks WithdrawalsController#withdrawals
  2. MIY额度校验 cites real cues (quotaCheckMIY / aqcClient / loanSchemaQuery)
  3. 源码 ask pastes real Java body (loanSchemaQuery / availLimit / aqcClient)
  4–5. Sudden 风控 / 融担 / loanSwitch dig-ins stay on withdrawals CALLS
  6. Optional 落库 / 回调 still locked

Accuracy:
  - No invented t_user_quota / t_product_config / t_credit_line
  - No「未获取到 quotaCheckMIY 实现」when code exists

Usage:
  python tests/smoke_loan_multi_scenario.py
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

REPORT = ROOT / "data" / "loan_multi_scenario_smoke.txt"
PROJECT_ID = "demo-loan"

FAKE_TABLES = (
    "t_user_quota",
    "t_user_credit_limit",
    "t_product_config",
    "t_credit_line",
    "t_quota_info",
    "t_miy_quota",
)

QUESTIONS = [
    "放款流程什么样？从入口到下游回调串一下。",
    "MIY的额度校验具体什么样",
    "MIY额度校验源码是什么",
    "风控是怎么做的",
    "融担怎么判断的",
    "loanSwitch干嘛的",
    "放款落库写哪",
]


def _has_fake_tables(answer: str) -> list[str]:
    return [t for t in FAKE_TABLES if t in (answer or "")]


def _lock_ok(locked: str | None) -> bool:
    return bool(
        locked and re.search(r"WithdrawalsController#withdrawals", str(locked), re.I)
    )


def _judge(i: int, q: str, answer: str, locked: str | None) -> tuple[bool, str]:
    a = answer or ""
    issues: list[str] = []
    fakes = _has_fake_tables(a)
    if fakes:
        issues.append(f"fake tables: {fakes}")
    if re.search(r"未获取到.{0,30}quotaCheckMIY.{0,20}实现", a) or re.search(
        r"未获取到.{0,40}具体实现代码", a
    ):
        issues.append("contradictory「未获取到实现代码」")

    if i == 0:
        if not a.strip():
            issues.append("empty answer")
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(r"放款|withdrawals|提现|回调|校验", a, re.I):
            issues.append("missing flow cues")
    elif i == 1:
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(r"(?i)quotaCheckMIY", a):
            issues.append("missing quotaCheckMIY")
        if not re.search(
            r"(?i)aqcClient|loanSchemaQuery|可用额度|额度状态|availLimit|额度树",
            a,
        ):
            issues.append("missing real MIY logic cues")
    elif i == 2:
        # 源码 ask — must paste real body snippets
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(r"(?i)quotaCheckMIY", a):
            issues.append("missing quotaCheckMIY")
        if not re.search(r"(?i)loanSchemaQuery|aqcClient", a):
            issues.append("missing loanSchemaQuery/aqcClient in source answer")
        if not re.search(r"(?i)availLimit|getAvailLimit|creditAmount", a):
            issues.append("missing availLimit/creditAmount body snippet")
        if "```" not in a and "private boolean quotaCheckMIY" not in a:
            issues.append("missing java code fence / method signature")
        if "不会编造源码" in a or re.search(r"未在 MySQL.*也未在工程", a):
            issues.append("source body missing (honest miss)")
    elif i == 3:
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(r"(?i)risk|asynRisk|风控", a):
            issues.append("missing risk method cues")
    elif i == 4:
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(
            r"(?i)guara|FinanceGuara|guaranteeRoute|rongDan|融担",
            a,
        ):
            issues.append("missing guara/融担 method cues")
    elif i == 5:
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(r"(?i)loanSwitch|开关", a):
            issues.append("missing loanSwitch cues")
    elif i == 6:
        if not _lock_ok(locked):
            issues.append(f"entry not locked ({locked!r})")
        if not re.search(
            r"(?i)insert|落库|LoanApply|withdrawals|Mapper|write|写入",
            a,
        ):
            issues.append("missing 落库 cues")
        # Still must not invent AccClient drift
        if "AccClient#" in a:
            issues.append("forbidden AccClient# drift")

    # Shared: dig-ins after Q0 must not jump to AccClient APIs
    if i >= 1 and re.search(r"AccClient#|AmountController", a):
        issues.append("forbidden AccClient/AmountController drift")

    return (not issues), ("; ".join(issues) if issues else "ok")


async def main() -> int:
    from app.agent import session_store
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    sid = f"smoke-multi-{uuid.uuid4().hex[:10]}"
    lines: list[str] = [
        f"session_id={sid}",
        f"project_id={PROJECT_ID}",
        "",
    ]
    matrix: list[tuple[str, str, str]] = []
    all_ok = True
    locked = None
    source_sample = ""

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
        ok, reason = _judge(i, q, answer, locked)
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        matrix.append((f"Q{i + 1}", status, reason))
        print(f"[{status}] lock={locked} | {reason}", flush=True)
        print(answer[:1400], flush=True)
        if i == 2:
            source_sample = "\n".join((answer or "").splitlines()[:20])
        lines.append(f"=== Q{i + 1}: {q} ===")
        lines.append(f"status={status} reason={reason}")
        lines.append(f"last_class_method={locked}")
        lines.append(answer)
        lines.append("")

    lines.append("=== MATRIX ===")
    for qid, status, reason in matrix:
        lines.append(f"{qid}\t{status}\t{reason}")
    if source_sample:
        lines.append("")
        lines.append("=== SOURCE_SAMPLE (first ~20 lines of Q3) ===")
        lines.append(source_sample)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print("\n=== MATRIX ===", flush=True)
    for qid, status, reason in matrix:
        print(f"{qid}\t{status}\t{reason}", flush=True)
    print(f"\nwrote {REPORT}", flush=True)
    print("ALL_PASS" if all_ok else "ALL_FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
