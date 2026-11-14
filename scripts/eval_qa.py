#!/usr/bin/env python
"""Evaluation gate for chat quality — fixed question set + keyword judging + report.

Usage:
    python scripts/eval_qa.py                          # hit http://127.0.0.1:8000
    python scripts/eval_qa.py --api http://192.0.2.100:8000
    python scripts/eval_qa.py --count 3 --json out.json

Judging is deterministic keyword matching (expect-any / forbid-any) so it is
stable under weaker local models — run it after every change to the agent
loop, compaction, or retrieval to make sure nothing regressed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (question, expect_any, forbid_any) — expect: answer contains >=1; forbid: answer contains none
QUESTIONS: list[tuple[str, list[str], list[str]]] = [
    ("放款流程是什么样", ["放款", "流程", "状态", "落库"], []),
    ("提现接口的地址是什么", ["url", "path", "地址", "服务", "http"], ["未检索到"]),
    (
        "放款接口增加一个字段，并落库",
        ["字段", "VO", "改动", "落库", "ReqVo", "实体"],
        [],
    ),
    (
        "WithdrawalsController#withdrawals 的源码是什么",
        ["WithdrawalsController", "代码", "方法", "```"],
        ["未检索到"],
    ),
    ("放款流程涉及哪几张表", ["表", "t_", "实体", "TableName"], []),
]


def _ask(api: str, question: str, timeout: float = 240.0) -> dict:
    body = json.dumps(
        {"message": question, "session_id": None, "project_id": "demo-loan"}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api}/api/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    payload["_elapsed_ms"] = int((time.time() - t0) * 1000)
    return payload


def _judge(question: str, answer: str, expect: list[str], forbid: list[str]) -> dict:
    ans = answer or ""
    hits = [k for k in expect if k.lower() in ans.lower()]
    banned = [k for k in forbid if k.lower() in ans.lower()]
    ok = bool(hits) and not banned
    return {"pass": ok, "hits": hits, "banned": banned, "answer_len": len(ans)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Chat quality eval gate")
    ap.add_argument("--api", default="http://127.0.0.1:8000", help="backend base URL")
    ap.add_argument("--count", type=int, default=0, help="limit question count (0=all)")
    ap.add_argument("--json", default="", help="write report JSON to this path")
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    items = QUESTIONS[: args.count] if args.count > 0 else QUESTIONS
    results: list[dict] = []
    for q, expect, forbid in items:
        print(f"[..] {q}")
        try:
            payload = _ask(args.api, q, timeout=args.timeout)
            answer = str(
                payload.get("reply")
                or payload.get("answer")
                or payload.get("message")
                or ""
            )
            j = _judge(q, answer, expect, forbid)
        except Exception as e:  # noqa: BLE001
            j = {"pass": False, "hits": [], "banned": [], "answer_len": 0, "error": str(e)}
            answer = ""
        j.update(
            {
                "question": q,
                "elapsed_ms": payload.get("_elapsed_ms") if isinstance(payload, dict) else None,
                "answer_head": answer[:120],
            }
        )
        results.append(j)
        print(f"     {'PASS' if j['pass'] else 'FAIL'} ({j.get('elapsed_ms')}ms) hits={j['hits']} banned={j['banned']}")
        if not j["pass"]:
            print(f"     answer: {answer[:200]!r}")

    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"\n== eval gate: {passed}/{total} passed ==")
    if args.json:
        out = {
            "passed": passed,
            "total": total,
            "results": results,
        }
        Path(args.json).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report -> {args.json}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
