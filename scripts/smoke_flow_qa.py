"""Live smoke: ask explore agent about core LOAN flows and score replies.

Usage (server already up on :8000):
  python scripts/smoke_flow_qa.py
  python scripts/smoke_flow_qa.py --base http://127.0.0.1:8000 --limit 6
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Case:
    id: str
    question: str
    # At least one of these should appear (business/tech anchors).
    expect_any: list[str] = field(default_factory=list)
    # Hard fail if any match.
    forbid_any: list[str] = field(default_factory=list)
    # Soft: flow answers should not end on vague continue.
    require_depth: bool = True


CASES: list[Case] = [
    Case(
        id="credit_notification",
        question="credit/creditNotification 授信通知接口流程是什么样的？请写清校验、防重、落库和外部调用。",
        expect_any=["授信", "creditNotification", "Credit", "流水", "机构"],
        forbid_any=["withdrawals/withdrawals", "放款申请主链", "WithdrawalsController#withdrawals"],
    ),
    Case(
        id="withdrawals",
        question="withdrawals/withdrawals 放款申请的完整流程是什么？从入参校验到落库再到调用外部系统。",
        expect_any=["放款", "withdrawals", "Withdrawals"],
        forbid_any=["creditNotification"],
    ),
    Case(
        id="credit_usage_apply_loan",
        question="creditUsageApplyLoan 用信申请放款统一接口流程是怎样的？和普通 withdrawals 有何区别？",
        expect_any=["用信", "creditUsage", "放款"],
    ),
    Case(
        id="repay",
        question="repay/repayment 还款接口主流程是怎样的？试算、扣款、入账分别怎么走？",
        expect_any=["还款", "repay", "Repay"],
        forbid_any=["creditNotification"],
    ),
    Case(
        id="repay_calculate",
        question="repay/repayCalculate 还款试算接口校验哪些字段？计算结果怎么返回？",
        expect_any=["试算", "repayCalculate", "还款"],
        require_depth=False,
    ),
    Case(
        id="quota_positive",
        question="quota/postiveAdjustApply 正向调额完整链路是什么？落库和外部回调怎么衔接？",
        expect_any=["调额", "postiveAdjust", "positive", "quota", "Quota"],
    ),
    Case(
        id="risk_callback",
        question="risk/callBack 外部风控回调后，放款或交易状态如何流转？",
        expect_any=["风控", "callBack", "Risk", "回调"],
    ),
    Case(
        id="loan_notice",
        question="loanNotice/deal 放款通知接口收到结果后怎么处理？会通知哪些下游？",
        expect_any=["放款通知", "loanNotice", "LoanNotice", "通知"],
    ),
    Case(
        id="guarantee_route",
        question="route/guaranteeRoute/callback 担保路由回调后放款流程如何继续？失败状态怎么变？",
        expect_any=["guaranteeRouteCallback", "RouteController", "担保", "Route"],
    ),
    Case(
        id="entrusted_refund",
        question="/entrusted/refund/apply 受托退货申请有哪些业务校验？涉及哪些表和外部接口？",
        expect_any=["退货", "refund", "受托", "Entrusted"],
    ),
    Case(
        id="approval_notify",
        question="withdrawals/approvalResultNotify 审批结果通知后借据状态如何变化？",
        expect_any=["approvalResultNotify", "WithdrawalsApproval", "审批", "processApprovalResult"],
    ),
    Case(
        id="credit_query",
        question="credit/creditQuery 授信查询接口做什么？数据从哪里取？",
        expect_any=["creditQuery", "CreditController", "授信查询", "查询授信"],
        forbid_any=[
            "CreditQuotaController#creditNotification",
            "CreditQuotaController",
            "creditNotification",
            "CreditNotification",
            "重复授信",
            "DEMO_CREDIT_LOCK",
        ],
        require_depth=False,
    ),
]


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_json(url: str, *, timeout: float = 15) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _score(case: Case, reply: str, review: dict[str, Any] | None) -> dict[str, Any]:
    text = reply or ""
    issues: list[str] = []
    ok = True

    if len(text.strip()) < 60:
        ok = False
        issues.append("回答过短")
    if re.search(r"未能根据代码得出|处理失败：|模型服务返回|探索步数用尽|模型服务响应超时", text):
        ok = False
        issues.append("失败/空探索话术")
    if case.expect_any and not any(k.lower() in text.lower() for k in case.expect_any):
        ok = False
        issues.append("未命中期望关键词: " + "/".join(case.expect_any[:4]))
    for bad in case.forbid_any:
        if bad and bad.lower() in text.lower():
            ok = False
            issues.append(f"串题/禁词: {bad}")
            break
    if case.require_depth:
        if re.search(r"(构造.{0,12}对象.{0,12}继续|继续处理|后续业务处理)", text):
            # Allow if also has persist+external nearby; otherwise fail.
            has_persist = bool(re.search(r"落库|写入|insert|update|save|持久化|保存", text, re.I))
            has_ext = bool(
                re.search(r"Feign|Client|MQ|远程|外部|同步|异步|调用", text, re.I)
            )
            if not (has_persist and has_ext):
                ok = False
                issues.append("空话收尾且缺落库/外部细节")
        else:
            has_persist = bool(re.search(r"落库|写入|insert|update|save|持久化|保存", text, re.I))
            has_ext = bool(
                re.search(
                    r"Feign|Client|MQ|Rabbit|Kafka|远程|外部|同步|异步|调用",
                    text,
                    re.I,
                )
            )
            if not has_persist:
                issues.append("软缺: 未见落库描述")
            if not has_ext:
                issues.append("软缺: 未见外部调用描述")
            # Soft misses do not fail unless review also fails.
            if (not has_persist or not has_ext) and review and review.get("ok") is False:
                ok = False
                issues.append("评审未通过且缺落库/外部")

    if review is not None and review.get("ok") is False:
        gaps = review.get("gaps") or []
        gap_txt = ";".join(str(g) for g in gaps[:2]) if gaps else str(review.get("summary") or "")
        grounded = bool(
            re.search(r"(Controller|ServiceImpl|Mapper|Facade)#\w+", text)
        ) or bool(re.search(r"(?m)^\s*\d+[\.、)]\s+\S+", text))
        if case.require_depth and not (grounded and len(text) > 180):
            ok = False
            issues.append("评审未通过: " + gap_txt)
        else:
            issues.append("评审未通过(软): " + gap_txt)

    return {"ok": ok, "issues": issues}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--project-id", default="demo-loan")
    ap.add_argument("--limit", type=int, default=0, help="0=all cases")
    ap.add_argument(
        "--ids",
        default="",
        help="comma-separated case ids to run (optional)",
    )
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument(
        "--out",
        default=str(Path("data") / "smoke_flow_qa_results.json"),
    )
    args = ap.parse_args()

    health_url = args.base.rstrip("/") + "/api/v1/health"
    try:
        _get_json(health_url, timeout=10)
    except Exception as e:
        # Fallback: OpenAPI docs means the app is up.
        try:
            with urllib.request.urlopen(args.base.rstrip("/") + "/docs", timeout=10) as resp:
                if resp.status >= 400:
                    raise RuntimeError(f"docs status {resp.status}")
        except Exception:
            print(f"FAIL: server not reachable at {args.base} ({e})", file=sys.stderr)
            return 2

    cases = CASES[: args.limit] if args.limit > 0 else list(CASES)
    if args.ids.strip():
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        cases = [c for c in cases if c.id in want]
    chat_url = args.base.rstrip("/") + "/api/v1/chat"
    results: list[dict[str, Any]] = []
    passed = 0

    for i, case in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}] {case.id}", flush=True)
        print(f"Q: {case.question}", flush=True)
        t0 = time.perf_counter()
        try:
            data = _post_json(
                chat_url,
                {
                    "message": case.question,
                    "project_id": args.project_id,
                },
                timeout=args.timeout,
            )
            err = None
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            data = {}
            err = f"HTTP {e.code}: {body[:300]}"
        except Exception as e:
            data = {}
            err = f"{type(e).__name__}: {e}"

        elapsed = int((time.perf_counter() - t0) * 1000)
        reply = str(data.get("reply") or "")
        review = data.get("review") if isinstance(data.get("review"), dict) else None
        if err:
            score = {"ok": False, "issues": [err]}
        else:
            score = _score(case, reply, review)
        if score["ok"]:
            passed += 1
            mark = "PASS"
        else:
            mark = "FAIL"
        print(f"{mark} ({elapsed}ms) issues={score['issues']}", flush=True)
        print("A:", (reply[:400] + ("…" if len(reply) > 400 else "")).replace("\n", " / "), flush=True)
        if review:
            print(
                f"review: ok={review.get('ok')} score={review.get('score')} "
                f"summary={review.get('summary')}",
                flush=True,
            )
        results.append(
            {
                "id": case.id,
                "question": case.question,
                "ok": score["ok"],
                "issues": score["issues"],
                "elapsed_ms": elapsed,
                "intent": data.get("intent"),
                "review": review,
                "reply": reply,
                "case": asdict(case),
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "passed": passed,
        "total": len(cases),
        "pass_rate": round(passed / max(len(cases), 1), 3),
        "results": results,
    }
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== {passed}/{len(cases)} passed → {out}", flush=True)
    return 0 if passed == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
