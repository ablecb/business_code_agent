"""Live smoke: 放款 Q&A vs Neo4j FOLLOWS spine.

Usage:
  set EXPLORE_BACKEND=index
  set SEARCH_USE_MILVUS=true
  python scripts/loan_qa_smoke.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force index path before settings load.
os.environ.setdefault("EXPLORE_BACKEND", "index")
os.environ.setdefault("SEARCH_USE_MILVUS", "true")

from app.agent.explore_agent import (  # noqa: E402
    _lexical_main_entry,
    run_explore_agent,
)
from app.analysis.flow_neo4j_sync import get_follows_path  # noqa: E402
from app.core.config import get_settings  # noqa: E402

REPORT = ROOT / "data" / "loan_qa_smoke_report.txt"

EXPECTED_SPINE = [
    "WithdrawalsController#withdrawals",
    "WithdrawalsController#checkApplyReqParam",
    "WithdrawalsServiceImpl#loanSwitch",
    "LoanPProcessConfigServiceImpl#loanDecisionProcessConfig",
    "WithdrawalsApplyServiceImpl#loanApply",
    "WithdrawalsController#applyResultDealWith",
    "WithdrawalsController#generateReqParam",
    "WithdrawalsServiceImpl#insertLoanData",
    "WithdrawalsServiceImpl#loan",
    "WithdrawalsServiceImpl#insertOrUpdateLoanInfo",
]

SIBLING_HTTP = [
    "withDrawalCalculate",
    "creditUsageApply",
    "WithdrawalsController#withDrawalCalculate",
    "WithdrawalsController#creditUsageApply",
]


@dataclass
class Case:
    id: str
    question: str
    lock_withdrawals: bool = False
    expect_order: bool = False  # loanSwitch before loanApply before loan
    forbid_sibling_http: bool = False
    business_mode: bool = False  # avoid aggressive Class#method dump
    expect_facts: bool = False  # tables/status
    trap_credit_notice: bool = False
    expect_any: list[str] = field(default_factory=list)
    forbid_any: list[str] = field(default_factory=list)


CASES: list[Case] = [
    Case(
        id="flow_shape",
        question="放款流程什么样",
        lock_withdrawals=True,
        expect_order=True,
        forbid_sibling_http=True,
        business_mode=True,
        expect_any=["放款", "提现", "申请"],
    ),
    Case(
        id="flow_api",
        question="放款接口什么流程",
        lock_withdrawals=True,
        expect_order=True,
        forbid_sibling_http=True,
        business_mode=True,
    ),
    Case(
        id="entry_to_db",
        question="放款从入口到落库串一下",
        lock_withdrawals=True,
        expect_order=True,
        forbid_sibling_http=True,
        expect_facts=True,
    ),
    Case(
        id="loan_switch",
        question="loanSwitch 做什么",
        expect_any=["开关", "loanSwitch", "放款"],
        forbid_any=["授信通知", "creditNotification"],
    ),
    Case(
        id="tables_status",
        question="放款写了哪些表/状态怎么变",
        lock_withdrawals=True,
        expect_facts=True,
        forbid_sibling_http=True,
    ),
    Case(
        id="trap_credit_notice",
        question="授信通知",
        trap_credit_notice=True,
        forbid_any=[
            "WithdrawalsController#withdrawals",
            "loanSwitch",
            "放款申请主链",
        ],
        expect_any=["授信", "creditNotification", "Credit"],
    ),
]


def _method_mentions(text: str) -> dict[str, int]:
    """Map short method names -> first char index in answer."""
    names = [
        "loanSwitch",
        "loanDecisionProcessConfig",
        "loanApply",
        "applyResultDealWith",
        "generateReqParam",
        "insertLoanData",
        "insertOrUpdateLoanInfo",
        "checkApplyReqParam",
    ]
    # Prefer whole Class#method, else bare method token.
    out: dict[str, int] = {}
    low = text
    for n in names:
        # bare method as word-ish
        m = re.search(rf"(?<![A-Za-z]){re.escape(n)}(?![A-Za-z])", low)
        if m:
            out[n] = m.start()
    # also loan as WithdrawalsServiceImpl#loan or 「放款主流程」proxies
    m_loan = re.search(
        r"WithdrawalsServiceImpl#loan(?![A-Za-z])|(?<![A-Za-z])#loan(?![A-Za-z])",
        low,
    )
    if m_loan:
        out["loan"] = m_loan.start()
    return out


def _order_ok(text: str) -> tuple[bool, str]:
    """loanSwitch before loanApply before loan (when present)."""
    pos = _method_mentions(text)
    # Chinese stage titles (business mode). Prefer specific phrases; avoid bare「放款」.
    zh_map = {
        "loanSwitch": r"放款开关控制|开关控制|放款开关|loanSwitch",
        "loanApply": r"放款申请（写申请表）|写申请表|loanApply",
        "loan": r"放款主流程|WithdrawalsServiceImpl#loan(?![A-Za-z])",
        "insertLoanData": r"插入放款主表|insertLoanData",
    }
    for k, pat in zh_map.items():
        if k not in pos:
            m = re.search(pat, text)
            if m:
                pos[k] = m.start()
    # Numbered spine: require switch before apply before (loan|insert主表).
    need = ["loanSwitch", "loanApply"]
    tail = "loan" if "loan" in pos else ("insertLoanData" if "insertLoanData" in pos else None)
    if tail:
        need = need + [tail]
    present = [k for k in need if k in pos]
    if len(present) < 2:
        return False, f"主链锚点不足: {present or '无'} (need loanSwitch/loanApply/loan)"
    ordered = sorted(present, key=lambda k: pos[k])
    expect = [k for k in need if k in present]
    if ordered != expect:
        return False, f"顺序错误: {ordered} expected {expect}"
    return True, "ok"


def _class_method_dump_score(text: str) -> int:
    return len(re.findall(r"\b[A-Z][\w]*#(?:[A-Za-z_][\w]*)\b", text or ""))


def _has_facts(text: str) -> bool:
    tables = bool(
        re.search(
            r"WithdrawalsApply|WithdrawalsExt|Withdrawals\b|申请表|主表|扩展表|"
            r"t_withdrawals|落库|写入|insert",
            text,
            re.I,
        )
    )
    status = bool(
        re.search(
            r"状态|status|SUCC|FAIL|处理中|LEND_|PROCESSING|成功|失败|更新",
            text,
            re.I,
        )
    )
    return tables or status


def score_case(case: Case, answer: str, meta: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    ok = True
    text = answer or ""

    if len(text.strip()) < 40:
        ok = False
        issues.append("回答过短")
    if re.search(r"未能根据代码得出|处理失败：|模型服务返回|探索步数用尽|模型服务响应超时", text):
        ok = False
        issues.append("失败/空探索话术")

    if case.lock_withdrawals:
        lex = _lexical_main_entry(case.question)
        if lex != "WithdrawalsController#withdrawals":
            issues.append(f"lexical锁异常: {lex}")
            ok = False
        # answer or meta should anchor withdrawals
        anchored = bool(
            re.search(
                r"WithdrawalsController#withdrawals|withdrawals/withdrawals|"
                r"贷款提现|自营放款|放款申请",
                text,
                re.I,
            )
            or (meta.get("entry_anchor") or "").endswith("#withdrawals")
        )
        if not anchored and "withdrawals" not in text.lower():
            # business mode may omit Class#method — require 放款主链语义
            if not re.search(r"放款|提现", text):
                ok = False
                issues.append("未锁定放款入口语义")

    if case.expect_order:
        good, msg = _order_ok(text)
        if not good:
            ok = False
            issues.append(msg)

    if case.forbid_sibling_http:
        for bad in SIBLING_HTTP:
            # only fail if presented as main-chain step-ish
            if bad.lower() in text.lower():
                # allow mention in forbid list context; hard fail if numbered near catalog
                if re.search(
                    rf"(?:主链|流程|步骤|①|1[\.、)]).{{0,40}}{re.escape(bad)}",
                    text,
                    re.I,
                ) or bad.startswith("WithdrawalsController#"):
                    ok = False
                    issues.append(f"兄弟HTTP进主链: {bad}")
                    break

    if case.business_mode:
        dump_n = _class_method_dump_score(text)
        if dump_n >= 6:
            ok = False
            issues.append(f"业务模式Class#method过密: {dump_n}")

    if case.expect_facts and not _has_facts(text):
        ok = False
        issues.append("缺少表/状态等具体事实")

    if case.trap_credit_notice:
        if re.search(r"WithdrawalsController#withdrawals|loanSwitch|insertLoanData", text):
            ok = False
            issues.append("授信通知串题到放款主链")
        if not any(k.lower() in text.lower() for k in case.expect_any):
            ok = False
            issues.append("授信通知未命中授信关键词")

    for bad in case.forbid_any:
        if bad and bad.lower() in text.lower():
            ok = False
            issues.append(f"禁词: {bad}")
            break

    if case.expect_any and not case.trap_credit_notice:
        if not any(k.lower() in text.lower() for k in case.expect_any):
            ok = False
            issues.append("未命中期望词: " + "/".join(case.expect_any[:4]))

    return {"ok": ok, "issues": issues, "dump_n": _class_method_dump_score(text)}


async def run_one(case: Case, project_id: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        result = await run_explore_agent(case.question, project_id=project_id)
    except Exception as e:
        return {
            "id": case.id,
            "question": case.question,
            "ok": False,
            "issues": [f"exception: {e}"],
            "answer": "",
            "elapsed": round(time.perf_counter() - t0, 1),
        }
    answer = str(result.get("answer") or "")
    entry = None
    for s in result.get("steps") or []:
        if s.get("entry_anchor"):
            entry = s.get("entry_anchor")
            break
    meta = {
        "entry_anchor": entry,
        "review": result.get("review"),
        "steps_n": len(result.get("steps") or []),
    }
    scored = score_case(case, answer, meta)
    return {
        "id": case.id,
        "question": case.question,
        "ok": scored["ok"],
        "issues": scored["issues"],
        "answer": answer,
        "entry_anchor": entry,
        "dump_n": scored["dump_n"],
        "elapsed": round(time.perf_counter() - t0, 1),
        "review": result.get("review"),
    }


async def main() -> int:
    get_settings.cache_clear()
    s = get_settings()
    print(
        f"backend={s.explore_backend} milvus={s.search_use_milvus} project={s.project_id}"
    )
    follows = get_follows_path(
        s.project_id, "WithdrawalsController#withdrawals", "loan_disburse"
    )
    ordered = follows.get("ordered_path") or [
        st.get("class_method") for st in (follows.get("steps") or [])
    ]
    spine_ok = list(ordered) == EXPECTED_SPINE
    print("FOLLOWS spine match:", spine_ok, "n=", len(ordered))
    if not spine_ok:
        print(" got:", ordered)
        print(" exp:", EXPECTED_SPINE)

    results: list[dict[str, Any]] = []
    for case in CASES:
        print(f"\n=== {case.id}: {case.question} ===")
        r = await run_one(case, s.project_id)
        results.append(r)
        status = "PASS" if r["ok"] else "FAIL"
        print(f"{status} ({r['elapsed']}s) issues={r['issues']}")
        print((r.get("answer") or "")[:500])
        print("---")

    lines: list[str] = []
    lines.append("loan_qa_smoke_report")
    lines.append(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(
        f"explore_backend={s.explore_backend} search_use_milvus={s.search_use_milvus}"
    )
    lines.append(f"FOLLOWS_spine_ok={spine_ok}")
    lines.append("FOLLOWS_ordered=" + " → ".join(str(x) for x in ordered))
    lines.append("")
    passed = sum(1 for r in results if r["ok"])
    failed = len(results) - passed
    lines.append(f"summary: {passed} passed / {failed} failed / {len(results)} total")
    lines.append("")
    for r in results:
        lines.append(f"## {r['id']} [{'PASS' if r['ok'] else 'FAIL'}] {r['elapsed']}s")
        lines.append(f"Q: {r['question']}")
        lines.append(f"entry_anchor: {r.get('entry_anchor')}")
        lines.append(f"issues: {r.get('issues')}")
        lines.append(f"dump_n: {r.get('dump_n')}")
        lines.append("A:")
        lines.append(r.get("answer") or "")
        lines.append("")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT}  passed={passed} failed={failed}")
    return 0 if failed == 0 and spine_ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
