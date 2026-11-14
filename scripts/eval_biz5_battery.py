#!/usr/bin/env python
"""Live 5-domain business battery: 放款 / 还款 / 调额 / 结清证明 / 授信.

Runs single-turn Q&A + codegen questions against the LIVE API (/api/v1/chat)
and measures answer quality (expect-any / forbid-any / honest-miss) and speed.

Usage:
    python scripts/eval_biz5_battery.py                     # all cases, sequential
    python scripts/eval_biz5_battery.py --api http://127.0.0.1:8000
    python scripts/eval_biz5_battery.py --concurrency 2
    python scripts/eval_biz5_battery.py --report data/eval_biz5.json
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FAKE_TABLES = [
    "t_user_quota",
    "t_user_credit_limit",
    "t_product_config",
    "t_credit_line",
    "t_quota_info",
    "t_miy_quota",
]

HONEST_MISS_RE = re.compile(
    r"未(?:在|能)(?:已读材料|已读|工程|代码库|索引|知识库)?中?找到|未检索到|证据不足|"
    r"不会编造|没有(?:找到|发现|检索到)|未找到(?:对应|相关)?(?:入口|实现|代码|方法)|"
    r"未在已读材料中看到|不存在于(?:工程|代码库)|无法给出(?:完整|准确)代码|只能在证据范围内",
    re.I,
)

# ==========================================================================
# 五域问题集（锚点均为工程内真实符号）
# ==========================================================================
DOMAINS: list[dict] = [
    {
        "domain": "放款",
        "entry": "WithdrawalsController#withdrawals",
        "cases": [
            {"id": "fk_flow", "q": "放款流程什么样？从入口到下游回调串一下",
             "expect": ["放款", "落库", "回调"], "forbid": FAKE_TABLES},
            {"id": "fk_status", "q": "放款接口返回结果里的 status 是怎么判断的",
             "expect": ["status", "判断", "枚举"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "fk_db", "q": "放款落库写哪张表、状态怎么变",
             "expect": ["落库", "demo_b_withdrawals", "状态"], "forbid": FAKE_TABLES},
            {"id": "fk_url", "q": "放款接口的地址是什么",
             "expect": ["url", "path", "地址", "http"], "forbid": ["未检索到"]},
            {"id": "fk_miy", "q": "MIY 的额度校验具体什么样",
             "expect": ["quotaCheckMIY", "额度", "校验"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "fk_addfield", "q": "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
             "expect": ["改动清单", "DDL", "WithdrawalsApplyReqVo", "落库"],
             "forbid": ["业务状态流转", FAKE_TABLES[0]]},
            {"id": "fk_code", "q": "给我一段代码：根据借据号查放款表，再按结果查助贷表",
             "expect": ["```java", "QueryWrapper", "deal_ref_no", "LoanAssistance"], "forbid": FAKE_TABLES},
        ],
    },
    {
        "domain": "还款",
        "entry": "RepayController#repayment",
        "cases": [
            {"id": "hk_flow", "q": "还款流程是什么样的",
             "expect": ["还款", "Repay"], "forbid": ["WithdrawalsController#withdrawals"]},
            {"id": "hk_notify", "q": "还款结果怎么通知下游/渠道",
             "expect": ["Listener", "MQ", "队列", "重推"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "hk_calc", "q": "还款试算 repayCalculate 主要做了什么",
             "expect": ["repayCalculate", "试算", "还款计划"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "hk_diff", "q": "主动还款和代扣还款有什么差别",
             "expect": ["代扣", "主动", "还款"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "hk_reverse", "q": "还款冲正/撤销流程",
             "expect": ["冲正", "撤销", "reverse"], "forbid": ["WithdrawalsController#revocation"],
             "honest_miss_ok": True},
            {"id": "hk_addfield", "q": "还款接口增加一个入参 newField，先透传给下游再落库，改动清单",
             "expect": ["改动清单", "透传", "落库", "Repay"],
             "forbid": ["WithdrawalsApplyReqVo", FAKE_TABLES[0]]},
            {"id": "hk_code", "q": "给我一段代码：还款成功回调里把结果重推给渠道（失败自动补推）",
             "expect": ["```java", "重推", "回调"], "forbid": FAKE_TABLES},
        ],
    },
    {
        "domain": "调额",
        "entry": "QuotaController#quotaAdjust / postiveAdjustApply / reverseAdjustApply",
        "cases": [
            {"id": "te_types", "q": "额度调整有哪些类型和策略",
             "expect": ["额度调整", "quotaAdjust", "正向", "反向", "策略"], "forbid": FAKE_TABLES},
            {"id": "te_positive", "q": "正向调额申请流程是什么",
             "expect": ["正向调额", "postiveAdjustApply", "额度"], "forbid": FAKE_TABLES,
             "honest_miss_ok": True},
            {"id": "te_reverse", "q": "反向调额流程是什么样的",
             "expect": ["反向调额", "reverseAdjust", "额度"], "forbid": FAKE_TABLES,
             "honest_miss_ok": True},
            {"id": "te_sync", "q": "调额结果怎么同步给渠道",
             "expect": ["同步", "渠道", "队列"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "te_tables", "q": "调额涉及哪些表",
             "expect": ["表", "quota_operation", "demo_b_"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "te_addfield", "q": "调额接口增加一个字段 batchNo 并透传给渠道，要改哪些",
             "expect": ["改动清单", "透传", "渠道", "调额"],
             "forbid": ["业务状态流转", FAKE_TABLES[0]], "honest_miss_ok": True},
        ],
    },
    {
        "domain": "结清证明",
        "entry": "SignatureController (结清证明申请/查询/发送)",
        "cases": [
            {"id": "jq_flow", "q": "结清证明申请接口 主要流程是什么",
             "expect": ["结清", "证明", "Signature"], "forbid": FAKE_TABLES},
            {"id": "jq_gen", "q": "结清证明是怎么生成和发送的",
             "expect": ["结清证明", "生成", "发送", "邮箱"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "jq_check", "q": "结清证明申请前要做哪些校验",
             "expect": ["校验", "结清", "未结清"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "jq_tables", "q": "结清证明申请涉及哪些表",
             "expect": ["表", "signature", "demo_b_"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "jq_addfield", "q": "结清证明申请接口增加一个字段 applySource 并落库，改动清单",
             "expect": ["改动清单", "落库", "Signature"], "forbid": ["业务状态流转", FAKE_TABLES[0]],
             "honest_miss_ok": True},
        ],
    },
    {
        "domain": "授信",
        "entry": "CreditController#creditNotification",
        "cases": [
            {"id": "sx_notify", "q": "授信通知怎么实现的",
             "expect": ["creditNotification", "授信", "Credit"],
             "forbid": ["WithdrawalsController#withdrawals", "loanSwitch"]},
            {"id": "sx_query", "q": "授信额度查询/共享是怎么做的",
             "expect": ["creditQuery", "共享", "额度"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "sx_url", "q": "授信通知接口的地址是什么",
             "expect": ["url", "path", "地址", "http"], "forbid": ["未检索到"]},
            {"id": "sx_usec", "q": "用信/授信使用（UseCredit）流程",
             "expect": ["用信", "UseCredit", "额度"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
            {"id": "sx_addfield", "q": "授信接口增加一个字段 channelCode 并落库，改动清单",
             "expect": ["改动清单", "落库", "Credit"], "forbid": ["业务状态流转", FAKE_TABLES[0]],
             "honest_miss_ok": True},
        ],
    },
]

# ==========================================================================
# 多轮（同 session）：入口锁定与切换
# ==========================================================================
MULTI_CASES: list[dict] = [
    {
        "id": "mt_lock", "turns": [
            {"q": "放款流程什么样？从入口到下游回调串一下", "expect": ["放款", "落库"], "forbid": FAKE_TABLES},
            {"q": "风控是怎么做的", "expect": ["风控", "risk"], "forbid": ["AmountController"]},
            {"q": "那 MIY 额度校验呢", "expect": ["MIY", "quotaCheckMIY"], "forbid": FAKE_TABLES},
            {"q": "loanSwitch 干嘛的", "expect": ["loanSwitch", "开关"], "forbid": ["creditNotification"]},
        ],
    },
    {
        "id": "mt_switch", "turns": [
            {"q": "还款流程是什么样的", "expect": ["还款", "Repay"], "forbid": ["WithdrawalsController#withdrawals"]},
            {"q": "那放款从入口到落库串一下", "expect": ["放款", "落库", "withdrawals"], "forbid": []},
        ],
    },
]


def _match_any(answer: str, items: list[str]) -> list[str]:
    a = answer or ""
    return [it for it in items if it.startswith("re:") and re.search(it[3:], a) or (not it.startswith("re:") and it in a)]


def _match_forbid(answer: str, items: list[str]) -> list[str]:
    a = answer or ""
    return [it for it in items if it.startswith("re:") and re.search(it[3:], a) or (not it.startswith("re:") and it in a)]


def _judge(c: dict, answer: str) -> dict:
    a = answer or ""
    hits = _match_any(a, c.get("expect") or [])
    banned = _match_forbid(a, c.get("forbid") or [])
    issues: list[str] = []
    if not hits:
        if c.get("honest_miss_ok") and HONEST_MISS_RE.search(a) and not banned:
            pass
        else:
            issues.append("no expect hit")
    if banned:
        issues.append(f"forbidden: {banned}")
    return {"pass": not issues, "issues": issues, "hits": hits, "banned": banned}


def _ask_http(api: str, question: str, session_id: str | None, timeout: float = 320.0) -> dict:
    body = json.dumps(
        {"message": question, "session_id": session_id, "project_id": "demo-loan"}
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{api}/api/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        payload["_ok"] = True
    except Exception as e:  # noqa: BLE001
        payload = {"_ok": False, "_error": f"{type(e).__name__}: {e}"}
    payload["_elapsed_s"] = round(time.time() - t0, 1)
    return payload


def _run_one(api: str, c: dict, session_id: str | None = None,
             inprocess: bool = False) -> dict:
    if inprocess:
        payload = _ask_inprocess(c["q"], session_id)
    else:
        payload = _ask_http(api, c["q"], session_id)
    if not payload.get("_ok"):
        return {"id": c["id"], "question": c["q"], "pass": False,
                "issues": [payload.get("_error", "request failed")], "elapsed_s": payload.get("_elapsed_s"),
                "answer": "", "intent": ""}
    reply = str(payload.get("reply") or payload.get("answer") or payload.get("message") or "")
    j = _judge(c, reply)
    return {"id": c["id"], "question": c["q"], "pass": j["pass"], "issues": j["issues"],
            "hits": j["hits"], "banned": j["banned"], "elapsed_s": payload.get("_elapsed_s"),
            "intent": str(payload.get("intent") or ""), "answer_length": len(reply),
            "answer_head": reply[:180]}


def _ask_inprocess(question: str, session_id: str | None, timeout: float = 320.0) -> dict:
    """In-process run_chat (current disk code, index backend)."""
    import asyncio
    import time
    import uuid

    from app.agent.graph_agent import run_chat

    t0 = time.time()
    sid = session_id or str(uuid.uuid4())

    async def _one() -> dict:
        return await run_chat(
            question,
            project_id="demo-loan",
            session_id=sid,
        )

    try:
        res = asyncio.run(asyncio.wait_for(_one(), timeout=timeout))
        payload = {"_ok": True}
        payload["reply"] = str(res.reply or "")
        payload["intent"] = str(res.intent or "")
    except Exception as e:  # noqa: BLE001
        payload = {"_ok": False, "_error": f"{type(e).__name__}: {e}"}
    payload["_elapsed_s"] = round(time.time() - t0, 1)
    payload["_session_id"] = sid
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description="5-domain live business battery")
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--report", default=str(ROOT / "data" / "eval_biz5.json"))
    ap.add_argument("--skip-multi", action="store_true")
    # --inprocess: run run_chat in-process (current disk code) instead of HTTP.
    ap.add_argument("--inprocess", action="store_true")
    args = ap.parse_args()

    def _ask(q: str, sid: str | None) -> dict:
        if args.inprocess:
            return _ask_inprocess(q, sid)
        return _ask_http(args.api, q, sid)

    flat: list[dict] = []
    for d in DOMAINS:
        for c in d["cases"]:
            flat.append({**c, "_domain": d["domain"], "_entry": d["entry"]})

    results: list[dict] = []
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {
            ex.submit(_run_one, args.api, c, None, args.inprocess): c for c in flat
        }
        for fut in cf.as_completed(futs):
            results.append(fut.result())

    report = {"project_id": "demo-loan", "api": args.api, "results": results,
              "multi": []}

    # multi-turn
    if not args.skip_multi:
        import uuid
        for mc in MULTI_CASES:
            sid = str(uuid.uuid4())
            turns = []
            for t in mc["turns"]:
                r = _run_one(args.api, {**t, "id": "turn"}, sid, args.inprocess)
                turns.append(r)
            report["multi"].append({"id": mc["id"], "turns": turns})

    # summary
    def _sum(items):
        ok = sum(1 for r in items if r["pass"])
        total = len(items)
        avg = round(sum(r.get("elapsed_s") or 0 for r in items) / total, 1) if total else 0
        return {"ok": ok, "total": total, "avg_elapsed_s": avg}

    summary = {}
    for d in DOMAINS:
        items = [r for r in results if r["id"] in {c["id"] for c in d["cases"]}]
        summary[d["domain"]] = _sum(items)
    all_single = [r for r in report["results"]]
    summary["_all_single"] = _sum(all_single)
    summary["_all_multi"] = _sum([t for m in report["multi"] for t in m["turns"]])
    report["summary"] = summary

    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n== 5-domain battery ({args.api}) ==")
    for d in DOMAINS:
        s = summary[d["domain"]]
        print(f"  {d['domain']}: {s['ok']}/{s['total']} pass, avg {s['avg_elapsed_s']}s")
    s = summary["_all_single"]; print(f"  ALL single: {s['ok']}/{s['total']} pass, avg {s['avg_elapsed_s']}s")
    s = summary["_all_multi"]; print(f"  ALL multi : {s['ok']}/{s['total']} pass, avg {s['avg_elapsed_s']}s")
    for r in results:
        if not r["pass"]:
            print(f"  FAIL {r['id']}: {r['issues']} ({r['elapsed_s']}s) q={r['question'][:40]}")
            print(f"        head={r['answer_head']!r}")
    print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())