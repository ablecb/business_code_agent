"""Smoke: 放款额度校验 existence dig-in (with/without prior 放款 turn).

Expect:
  - Mentions quotaCheck* (YES there is 额度校验)
  - No 评审提示 / 补充说明 in answer body
  - No fake t_* tables / 「未发现明确的额度校验」

Usage:
  python tests/smoke_loan_quota_existence.py
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

REPORT = ROOT / "data" / "loan_quota_existence_smoke.txt"
PROJECT_ID = "demo-loan"

Q = "放款接口 中 有没有做额度校验 额度校验逻辑又是什么样"

FAKE_TABLES = (
    "t_user_quota",
    "t_user_credit_limit",
    "t_product_config",
    "t_credit_line",
    "t_quota_info",
)


def _judge(answer: str) -> list[str]:
    a = answer or ""
    issues: list[str] = []
    if not re.search(r"(?i)quotaCheck", a):
        issues.append("missing quotaCheck*")
    if "未发现明确的额度校验" in a:
        issues.append("wrong negative conclusion")
    if "评审提示" in a:
        issues.append("评审提示 in answer body")
    if re.search(r"[（(]补充说明[：:]", a):
        issues.append("补充说明 footnote in answer body")
    if "可能在 checkApplyReqParam" in a:
        issues.append("speculative checkApplyReqParam")
    for t in FAKE_TABLES:
        if t in a:
            issues.append(f"invented table: {t}")
    return issues


async def _ask(sid: str, q: str) -> str:
    from app.agent.graph_agent import run_chat

    result = await run_chat(
        q,
        project_id=PROJECT_ID,
        session_id=sid,
    )
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    return str(data.get("answer") or data.get("reply") or "")


async def main() -> int:
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    lines: list[str] = [f"project_id={PROJECT_ID}", ""]
    all_ok = True

    # Case A: first message (lexical 放款, no prior turn)
    sid_a = f"smoke-quota-a-{uuid.uuid4().hex[:10]}"
    print(f"\n=== A first-turn: {Q} ===", flush=True)
    ans_a = await _ask(sid_a, Q)
    issues_a = _judge(ans_a)
    ok_a = not issues_a
    all_ok = all_ok and ok_a
    print(f"[{'PASS' if ok_a else 'FAIL'}] {issues_a or 'ok'}", flush=True)
    print(ans_a[:1200], flush=True)
    lines.append("=== A first-turn (no prior) ===")
    lines.append(f"status={'PASS' if ok_a else 'FAIL'} issues={issues_a}")
    lines.append(ans_a)
    lines.append("")

    # Case B: after 放款 overview (session lock)
    sid_b = f"smoke-quota-b-{uuid.uuid4().hex[:10]}"
    print("\n=== B1: 放款流程是怎样的 ===", flush=True)
    ans_b0 = await _ask(sid_b, "放款流程是怎样的")
    print(ans_b0[:400], flush=True)
    print(f"\n=== B2 follow-up: {Q} ===", flush=True)
    ans_b = await _ask(sid_b, Q)
    issues_b = _judge(ans_b)
    ok_b = not issues_b
    all_ok = all_ok and ok_b
    print(f"[{'PASS' if ok_b else 'FAIL'}] {issues_b or 'ok'}", flush=True)
    print(ans_b[:1200], flush=True)
    lines.append("=== B follow-up after 放款 ===")
    lines.append(f"status={'PASS' if ok_b else 'FAIL'} issues={issues_b}")
    lines.append(ans_b)
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {REPORT}", flush=True)
    print("ALL_PASS" if all_ok else "ALL_FAIL", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
