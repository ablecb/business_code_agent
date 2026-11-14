#!/usr/bin/env python
"""Quick mock-LLM verification: entry-resolution fix, scheduled-task & MQ coverage."""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["LLM_PROVIDER"] = "mock"
os.environ["HTTP_VERIFY_SSL"] = "false"

PID = "demo-loan"

async def _ask(q: str, sid: str) -> dict:
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    result = await run_chat(q, project_id=PID, session_id=sid)
    d = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    ev = d.get("evidence") or {}
    return {
        "entry": ev.get("entry") or d.get("entry") or "",
        "entry_source": ev.get("entry_source") or "",
        "intent": d.get("intent") or "",
        "answer": str(d.get("answer") or d.get("reply") or "")[:300],
        "calls": ev.get("calls") or [],
    }


def _judge(case: dict, resp: dict) -> bool:
    a = resp["answer"]
    entry = resp["entry"]
    issues = []
    for expect_token in case.get("expect_entry") or []:
        if expect_token not in entry:
            issues.append(f"entry missing {expect_token!r}")
    for forbid_token in case.get("forbid_entry") or []:
        if forbid_token in entry:
            issues.append(f"entry has forbidden {forbid_token!r}")
    for expect_token in case.get("expect") or []:
        if expect_token.lower() not in a.lower():
            issues.append(f"answer missing {expect_token!r}")
    for forbid_token in case.get("forbid") or []:
        if forbid_token.lower() in a.lower():
            issues.append(f"answer has forbidden {forbid_token!r}")
    ok = not issues
    print(f"    {'PASS' if ok else 'FAIL'}: entry={entry}; issues={issues}")
    return ok


CASES = [
    # === 入口修复测试（核心）===
    {
        "id": "fix-entry-fangkuan-interface",
        "q": "放款接口主要做了什么",
        "expect_entry": ["WithdrawalsController"],
        "forbid_entry": ["AmountController"],
        "expect": ["放款", "流程", "落库"],
        "forbid": ["累加放款额"],
        "note": "之前 AmountController#addDayQuota 误匹配修复",
    },
    {
        "id": "fix-entry-fangkuan-flow",
        "q": "放款流程什么样的",
        "expect_entry": ["WithdrawalsController"],
        "forbid_entry": ["AmountController"],
        "expect": ["放款", "流程"],
        "forbid": [],
        "note": "含流程关键词，应锚定放款主入口",
    },
    # === 定时任务测试 ===
    {
        "id": "job-overview",
        "q": "项目里有哪些 XXL-Job 定时任务",
        "expect_entry": [],
        "forbid_entry": [],
        "expect": ["XxlJob", "定时任务", "Job"],
        "forbid": ["未获取到"],
        "note": "定时任务总览",
    },
    {
        "id": "job-loan-apply",
        "q": "放款申请定时任务（LoanApplyJob）主要做了什么",
        "expect_entry": ["LoanApplyJob"],
        "forbid_entry": ["AmountController"],
        "expect": ["LoanApplyJob", "定时任务", "放款申请"],
        "forbid": ["未检索到"],
        "note": "放款申请定时任务",
    },
    {
        "id": "job-risk-result",
        "q": "获取风控结果定时任务是什么",
        "expect_entry": ["RiskResultJob"],
        "forbid_entry": [],
        "expect": ["风控", "定时任务", "risk"],
        "forbid": [],
        "note": "风控结果定时任务",
    },
    # === MQ/Listener 测试 ===
    {
        "id": "listener-overview",
        "q": "项目里有哪些 MQ 消息消费者（Listener）",
        "expect_entry": [],
        "forbid_entry": [],
        "expect": ["Listener", "MQ", "消费", "RabbitListener"],
        "forbid": ["未获取到"],
        "note": "MQ 消费者总览",
    },
    {
        "id": "listener-loan",
        "q": "放款结果 MQ 消费（LoanRabbitMqListenner）是什么",
        "expect_entry": ["LoanRabbitMqListenner"],
        "forbid_entry": ["AmountController"],
        "expect": ["LoanRabbitMqListenner", "MQ", "消费", "放款"],
        "forbid": [],
        "note": "放款结果 MQ 消费",
    },
    {
        "id": "listener-risk",
        "q": "风控决策结果 MQ 怎么消费的",
        "expect_entry": ["RiskControlRabbitMQListener"],
        "forbid_entry": [],
        "expect": ["RiskControlRabbitMQListener", "MQ", "消费", "风控"],
        "forbid": [],
        "note": "风控结果 MQ 消费",
    },
]


async def main() -> int:
    passed = total = 0
    for c in CASES:
        sid = f"verify-{c['id']}-{os.urandom(4).hex()}"
        print(f"[{c['id']}] {c['q'][:50]}")
        print(f"  note: {c['note']}")
        try:
            resp = await _ask(c["q"], sid)
            ok = _judge(c, resp)
        except Exception as e:
            print(f"    ERROR: {e}")
            ok = False
        if ok:
            passed += 1
        total += 1
        print()

    print(f"== 结果: {passed}/{total} passed ==")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))