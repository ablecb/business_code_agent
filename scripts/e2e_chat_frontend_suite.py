#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""E2E suite against the real frontend chat API (SSE /api/v1/chat/stream).

Business questions are *test data only*. Assertions are structural
(length / steps / code fence / 改动清单) — not product keyword routers.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "http://127.0.0.1:8000"
DEFAULT_PROJECT = "demo-loan"
DEFAULT_REPORT = ROOT / "reports" / "e2e_chat_frontend_report.md"
DEFAULT_JSON = ROOT / "reports" / "e2e_chat_frontend_results.json"


@dataclass
class CaseResult:
    case_id: str
    name: str
    message: str
    session_id: str
    pass_: bool
    duration_s: float
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    reply_excerpt: str = ""
    reply_full: str = ""
    error: str | None = None
    elapsed_ms: int | None = None
    url: str = ""


def parse_sse(text: str) -> tuple[dict | None, list[dict]]:
    events: list[dict] = []
    final = None
    for block in re.split(r"\n\n+", text):
        data_lines = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines).strip()
        if not raw or raw == "[DONE]":
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        events.append(obj)
        if obj.get("type") == "final":
            final = obj
    return final, events


def chat_once(
    client: httpx.Client,
    *,
    base: str,
    message: str,
    session_id: str,
    project_id: str,
) -> tuple[str, dict[str, Any], str, int]:
    body = {
        "message": message,
        "session_id": session_id,
        "project_id": project_id,
    }
    url_stream = base.rstrip("/") + "/api/v1/chat/stream"
    t0 = time.perf_counter()
    try:
        with client.stream(
            "POST",
            url_stream,
            json=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        ) as resp:
            resp.raise_for_status()
            chunks: list[str] = []
            for chunk in resp.iter_text():
                chunks.append(chunk)
        wall_ms = int((time.perf_counter() - t0) * 1000)
        final, _ = parse_sse("".join(chunks))
        if final is None:
            raise RuntimeError("no final SSE event")
        reply = (
            final.get("reply")
            or final.get("answer")
            or final.get("content")
            or ""
        )
        if not reply and isinstance(final.get("data"), dict):
            reply = final["data"].get("reply") or final["data"].get("answer") or ""
        return str(reply), final, url_stream, wall_ms
    except Exception as stream_err:
        url = base.rstrip("/") + "/api/v1/chat"
        last_err: Exception | None = stream_err
        for attempt in range(3):
            t0 = time.perf_counter()
            try:
                r = client.post(
                    url, json=body, headers={"Content-Type": "application/json"}
                )
                r.raise_for_status()
                data = r.json()
                wall_ms = int((time.perf_counter() - t0) * 1000)
                reply = (
                    data.get("reply") or data.get("answer") or data.get("content") or ""
                )
                return str(reply), data, url, wall_ms
            except Exception as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
        raise last_err  # type: ignore[misc]


def _ok(cond: bool, detail: str) -> tuple[str, bool, str]:
    return (detail, cond, "pass" if cond else "fail")


def check_flow_structure(answer: str) -> list[tuple[str, bool, str]]:
    a = answer or ""
    numbered = len(re.findall(r"(?m)^\s*(?:\d+[\.\)]|[-*•])\s+\S+", a))
    has_cm = bool(re.search(r"[A-Za-z][A-Za-z0-9_]+#[A-Za-z][A-Za-z0-9_]+", a))
    return [
        _ok(len(a.strip()) >= 80, "answer length >= 80"),
        _ok(numbered >= 2 or a.count("\n") >= 4, "has multi-step / multi-line structure"),
        _ok(has_cm or bool(re.search(r"(?i)Controller|Service|流程|入口", a)), "cites entry/class or flow language"),
    ]


def check_dig_in_structure(answer: str) -> list[tuple[str, bool, str]]:
    a = answer or ""
    has_cm = bool(re.search(r"[A-Za-z][A-Za-z0-9_]+#[A-Za-z][A-Za-z0-9_]+", a))
    return [
        _ok(len(a.strip()) >= 60, "answer length >= 60"),
        _ok(has_cm, "cites at least one Class#method"),
        _ok(
            bool(re.search(r"校验|调用|判断|比较|逻辑|方法|入口|CALLS", a)),
            "has dig-in explanatory language",
        ),
    ]


def check_code_structure(answer: str) -> list[tuple[str, bool, str]]:
    a = answer or ""
    has_fence = "```" in a
    has_javaish = bool(
        re.search(r"(?i)```java|private |public |protected |return |if\s*\(", a)
    )
    return [
        _ok(len(a.strip()) >= 80, "answer length >= 80"),
        _ok(has_fence or has_javaish, "includes code fence or Java-like body"),
        _ok(
            bool(re.search(r"[A-Za-z][A-Za-z0-9_]+#[A-Za-z]|方法|源码|代码", a)),
            "names a method or code context",
        ),
    ]


def check_change_plan_structure(answer: str) -> list[tuple[str, bool, str]]:
    a = answer or ""
    return [
        _ok(len(a.strip()) >= 100, "answer length >= 100"),
        _ok(
            bool(re.search(r"改动清单|改动顺序|改动点|##\s*结论|建议改动", a)),
            "has change-plan heading language",
        ),
        _ok(
            bool(re.search(r"(?i)字段|VO|Dto|Mapper|DDL|ALTER|落库|实体|入参", a)),
            "mentions structural change roles",
        ),
        _ok(a.count("\n") >= 5, "multi-line plan"),
    ]


def check_alt_flow_structure(answer: str) -> list[tuple[str, bool, str]]:
    """Structural checks for non-loan angle questions (same shape as flow)."""
    return check_flow_structure(answer)


Judge = Callable[[str], list[tuple[str, bool, str]]]

CASES: list[dict[str, Any]] = [
    {
        "id": "T1_flow_overview",
        "name": "Flow overview (test data: 放款流程)",
        "message": "放款流程什么样？从入口到下游回调串一下。",
        "session": "loan_mt",
        "judge": check_flow_structure,
    },
    {
        "id": "T2_gate_dig_in",
        "name": "Dig-in follow-up (test data: 额度校验逻辑)",
        "message": "额度校验逻辑又是什么样？",
        "session": "loan_mt",
        "judge": check_dig_in_structure,
    },
    {
        "id": "T3_show_code_followup",
        "name": "Show code follow-up (test data: 对应的代码)",
        "message": "我想看看对应的代码",
        "session": "loan_mt",
        "judge": check_code_structure,
    },
    {
        "id": "T4_show_code_cold",
        "name": "Show code cold-start (test data: 额度校验部分代码)",
        "message": "放款接口中额度校验部分代码什么样",
        "session": "code_cold",
        "judge": check_code_structure,
    },
    {
        "id": "T5_composite_plan",
        "name": "Change plan (test data: 新入参+落库+透传)",
        "message": (
            "放款接口新增一个入参 businessLabel 并落库，"
            "同时在机构校验/单笔单批时传给外部，给出改动清单"
        ),
        "session": "composite",
        "judge": check_change_plan_structure,
    },
    {
        "id": "T6_repay_angle",
        "name": "Different angle (test data: 还款流程)",
        "message": "还款流程的入口和主要处理类是什么？",
        "session": "repay",
        "judge": check_alt_flow_structure,
    },
    {
        "id": "T7_notice_angle",
        "name": "Different angle (test data: 授信通知)",
        "message": "授信通知流程是怎么走的？入口是哪个？",
        "session": "notice",
        "judge": check_alt_flow_structure,
    },
]


def run_suite(
    *,
    base: str,
    project_id: str,
    timeout_s: float,
) -> tuple[list[CaseResult], dict[str, str]]:
    run_tag = uuid.uuid4().hex[:8]
    sessions = {
        "loan_mt": f"e2e-loan-mt-{run_tag}",
        "code_cold": f"e2e-code-cold-{run_tag}",
        "composite": f"e2e-composite-{run_tag}",
        "repay": f"e2e-repay-{run_tag}",
        "notice": f"e2e-notice-{run_tag}",
    }
    results: list[CaseResult] = []
    timeout = httpx.Timeout(timeout_s, connect=30.0)
    with httpx.Client(timeout=timeout) as client:
        health = client.get(base.rstrip("/") + "/api/v1/health")
        health.raise_for_status()
        health_json = health.json()
        for case in CASES:
            sid = sessions[case["session"]]
            msg = case["message"]
            judge: Judge = case["judge"]
            t0 = time.perf_counter()
            try:
                reply, final, url, wall_ms = chat_once(
                    client,
                    base=base,
                    message=msg,
                    session_id=sid,
                    project_id=project_id,
                )
                checks = judge(reply)
                passed = all(c[1] for c in checks) and bool(reply.strip())
                if not reply.strip():
                    checks = [_ok(False, "empty reply")] + checks
                    passed = False
                results.append(
                    CaseResult(
                        case_id=case["id"],
                        name=case["name"],
                        message=msg,
                        session_id=sid,
                        pass_=passed,
                        duration_s=round(time.perf_counter() - t0, 2),
                        checks=checks,
                        reply_excerpt=(reply or "")[:600],
                        reply_full=reply or "",
                        elapsed_ms=int(final.get("elapsed_ms") or wall_ms)
                        if isinstance(final, dict)
                        else wall_ms,
                        url=url,
                    )
                )
            except Exception as e:
                results.append(
                    CaseResult(
                        case_id=case["id"],
                        name=case["name"],
                        message=msg,
                        session_id=sid,
                        pass_=False,
                        duration_s=round(time.perf_counter() - t0, 2),
                        checks=[_ok(False, f"request error: {e}")],
                        error=str(e),
                        url=base,
                    )
                )
    meta = {
        "health": json.dumps(health_json, ensure_ascii=False),
        "llm_provider": str(health_json.get("llm_provider") or ""),
        "sessions": json.dumps(sessions, ensure_ascii=False),
    }
    return results, meta


def write_report(
    results: list[CaseResult],
    meta: dict[str, str],
    *,
    base: str,
    project_id: str,
    report_path: Path,
    json_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    passed = sum(1 for r in results if r.pass_)
    failed = len(results) - passed
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
    lines: list[str] = [
        "# E2E Frontend Chat Suite Report",
        "",
        f"- Generated: `{now}`",
        f"- API base: `{base}`",
        f"- Project: `{project_id}`",
        f"- LLM provider (health): `{meta.get('llm_provider') or 'unknown'}`",
        f"- Health payload: `{meta.get('health')}`",
        f"- Summary: **{passed} passed / {failed} failed / {len(results)} total**",
        "",
        "## Assertion policy",
        "",
        "Business phrases appear only as **request payloads** (test data).",
        "Checks are structural: length, multi-step layout, Class#method citation,",
        "code fence, or 改动清单 headings — not product-code keyword routers.",
        "",
        "## Case results",
        "",
    ]
    for r in results:
        status = "PASS" if r.pass_ else "FAIL"
        lines.extend(
            [
                f"### {r.case_id} — {status}",
                "",
                f"- Name: {r.name}",
                f"- Session: `{r.session_id}`",
                f"- Duration: **{r.duration_s}s** (server elapsed_ms={r.elapsed_ms})",
                f"- URL: `{r.url}`",
                f"- Request payload:",
                "",
                "```json",
                json.dumps(
                    {
                        "message": r.message,
                        "session_id": r.session_id,
                        "project_id": project_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
                "",
                "- Checks:",
                "",
            ]
        )
        for detail, ok, _ in r.checks:
            lines.append(f"  - {'✓' if ok else '✗'} {detail}")
        if r.error:
            lines.extend(["", f"- Error: `{r.error}`", ""])
        lines.extend(
            [
                "",
                "- Sample answer excerpt:",
                "",
                "```",
                (r.reply_excerpt or "(empty)")[:600],
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## How to re-run",
            "",
            "```bash",
            "python scripts/e2e_chat_frontend_suite.py --base http://127.0.0.1:8000 --project-id demo-loan",
            "```",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "generated": now,
                "base": base,
                "project_id": project_id,
                "meta": meta,
                "passed": passed,
                "failed": failed,
                "results": [
                    {
                        "case_id": r.case_id,
                        "name": r.name,
                        "pass": r.pass_,
                        "duration_s": r.duration_s,
                        "elapsed_ms": r.elapsed_ms,
                        "session_id": r.session_id,
                        "message": r.message,
                        "checks": [
                            {"detail": d, "ok": ok} for d, ok, _ in r.checks
                        ],
                        "reply_excerpt": r.reply_excerpt,
                        "error": r.error,
                    }
                    for r in results
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--project-id", default=DEFAULT_PROJECT)
    ap.add_argument("--timeout", type=float, default=660.0)
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--json-out", type=Path, default=DEFAULT_JSON)
    args = ap.parse_args()
    results, meta = run_suite(
        base=args.base, project_id=args.project_id, timeout_s=args.timeout
    )
    write_report(
        results,
        meta,
        base=args.base,
        project_id=args.project_id,
        report_path=args.report,
        json_path=args.json_out,
    )
    passed = sum(1 for r in results if r.pass_)
    failed = len(results) - passed
    print(f"E2E done: {passed} passed / {failed} failed / {len(results)} total")
    print(f"Report: {args.report}")
    for r in results:
        print(f"  [{'PASS' if r.pass_ else 'FAIL'}] {r.case_id} ({r.duration_s}s)")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
