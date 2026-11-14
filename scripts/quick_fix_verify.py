#!/usr/bin/env python
"""Quick fix-verification: run key questions across all categories with lexical fix.

Usage:
    python scripts/quick_fix_verify.py
    python scripts/quick_fix_verify.py --report data/quick_fix_verify.json
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["LLM_LOG_PROMPTS"] = "false"

PROJECT_ID = "demo-loan"

# (id, question, expect_any, forbid_any)
SINGLE_CASES: list[tuple[str, str, list[str], list[str]]] = [
    # --- Entry routing (lexical fix test) ---
    ("routing_fangkuan", "放款流程什么样？从入口到下游回调串一下",
     ["WithdrawalsController", "放款", "落库"], ["t_user_quota", "creditUsageDeposit"]),
    ("routing_huankuan", "还款流程是什么样的",
     ["RepayController#repayment"], ["WithdrawalsController#withdrawals", "t_user_quota"]),
    ("routing_shouxin", "授信通知怎么实现的",
     ["CreditController#creditNotification"], ["WithdrawalsController#withdrawals"]),
    ("routing_risk_callback", "放款风控回调（异步决策）入口在哪",
     ["RiskControlRabbitMQListener#consume"], ["AboutFinanceGuaranteeController#loanCallback 当入口"]),
    # --- Code generation ---
    ("codegen_add_field", "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
     ["WithdrawalsApplyReqVo", "reqVo2ExtDmo", "WithdrawalsExt", "DDL"],
     ["t_user_quota", "LoanNoticeController"]),
    ("codegen_compose", "给我一段代码：根据借据号查放款表，再按结果查助贷表",
     ["```java", "deal_ref_no", "LoanAssistance"], ["t_user_quota"]),
    ("codegen_kb_login", "按知识库商品查询文档新增一个查询接口",
     ["Controller", "GoodsQueryReqVo", "pageNum"], ["Request<Map"]),
    # --- Traps ---
    ("trap_quick_sort", "用 Java 写一个快速排序",
     ["```java", "sort"], ["mvn", "编译"]),
    ("trap_fake_table", "改一个不存在的表 t_user_quota 的字段",
     ["未在工程中找到", "不存在"], ["t_user_quota 存在"]),
    # --- Historical regressions ---
    ("hist_compose_occupy", "给我一段代码：根据借据号查放款表，然后用放款表的记录查放款优惠券表，查到券信息后，调用外部接口对券进行占用",
     ["```java", "Coupon", "占用"], ["不会编造", "证据不足", "t_user_quota"]),
    ("hist_risk_fork", "放款时如果风控校验不过会走什么分支",
     ["风控", "分支", "拒绝"], ["t_user_quota"]),
    ("hist_gate_detail", "给我一个方法：查机构中心的报文取字段（如 limitTree、额度树）",
     ["```java", "aqc", "queryLimitTreeBase"], ["编造类名"]),
]

MULTI_CASES: list[tuple[str, list[tuple[str, list[str], list[str]]]]] = [
    # MT1: 放款→风控追问→MIY→落库 (entry lock drill)
    ("multi_lock_drill", [
        ("放款流程什么样", ["放款", "WithdrawalsController"], ["t_user_quota"]),
        ("风控是怎么做的", ["风控"], ["WithdrawalsController#withdrawals 当入口"]),
        ("那 MIY 额度校验呢", ["MIY", "quotaCheckMIY"], ["AmountController"]),
        ("放款落库写哪", ["落库", "insert"], ["t_user_quota"]),
    ]),
    # MT2: 还款→放款切换
    ("multi_entry_switch", [
        ("还款流程是什么样的", ["RepayController#repayment"], ["WithdrawalsController#withdrawals"]),
        ("那放款从入口到落库串一下", ["WithdrawalsController", "放款", "落库"], []),
    ]),
]


async def run_single(question: str, session_id: str) -> tuple[str, float, str, list]:
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings

    get_settings.cache_clear()
    t0 = time.time()
    result = await run_chat(question, project_id=PROJECT_ID, session_id=session_id)
    elapsed = time.time() - t0
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    answer = str(data.get("answer") or data.get("reply") or "")
    intent = str(data.get("intent") or data.get("goal") or "")
    steps = data.get("explore_steps") or []
    return answer, elapsed, intent, steps


def judge(answer: str, expect: list[str], forbid: list[str]) -> tuple[bool, list[str], list[str]]:
    a = answer or ""
    hits = [k for k in expect if k in a]
    banned = [k for k in forbid if k in a]
    ok = bool(hits) and not banned
    return ok, hits, banned


async def main() -> int:
    from app.agent import session_store
    from app.core.config import get_settings
    from app.db.mysql import init_db

    try:
        init_db()
        print("[setup] MySQL init_db ok", flush=True)
    except Exception as e:
        session_store._USE_MEMORY = True
        print(f"[setup] MySQL fallback memory ({type(e).__name__})", flush=True)

    from app.llm.factory import get_embedding_provider, get_llm_provider
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    s = get_settings()
    print(f"[setup] llm={s.llm_provider}/{s.llm_model} backend={s.explore_backend}", flush=True)

    results: list[dict] = []
    total_ok = total = 0

    print(f"\n== Single-turn ({len(SINGLE_CASES)}) ==\n", flush=True)
    for case_id, q, expect, forbid in SINGLE_CASES:
        sid = f"quick-{uuid.uuid4().hex[:8]}"
        answer, elapsed, intent, steps = await run_single(q, sid)
        ok, hits, banned = judge(answer, expect, forbid)
        total_ok += int(ok); total += 1
        status = "PASS" if ok else "FAIL"
        print(f"[{total}/{len(SINGLE_CASES)}] {case_id}: {status} ({elapsed:.0f}s) hits={hits} banned={banned}", flush=True)
        results.append({"id": case_id, "kind": "single", "pass": ok, "elapsed_s": round(elapsed, 1),
                        "intent": intent, "hits": hits, "banned": banned, "answer_head": answer[:200]})

    print(f"\n== Multi-turn ({len(MULTI_CASES)}) ==\n", flush=True)
    for case_id, turns in MULTI_CASES:
        sid = f"quick-multi-{uuid.uuid4().hex[:8]}"
        turn_results: list[dict] = []
        all_ok = True
        for i, (q, expect, forbid) in enumerate(turns, 1):
            answer, elapsed, intent, steps = await run_single(q, sid)
            ok, hits, banned = judge(answer, expect, forbid)
            all_ok = all_ok and ok
            turn_results.append({"turn": i, "question": q, "pass": ok, "hits": hits, "banned": banned, "answer_head": answer[:200]})
            status = "PASS" if ok else "FAIL"
            print(f"  Turn {i}: {status} ({elapsed:.0f}s) hits={hits} banned={banned}", flush=True)
        total_ok += int(all_ok); total += 1
        case_pass = "PASS" if all_ok else "FAIL"
        print(f"  => {case_id}: {case_pass}\n", flush=True)
        results.append({"id": case_id, "kind": "multi_turn", "pass": all_ok, "turns": turn_results})

    print(f"\n== Result: {total_ok}/{total} passed ==", flush=True)

    report_path = Path(sys.argv[sys.argv.index("--report") + 1]) if "--report" in sys.argv else ROOT / "data" / "quick_fix_verify.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"total_ok": total_ok, "total": total, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {report_path}", flush=True)
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))