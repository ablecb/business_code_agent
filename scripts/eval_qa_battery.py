#!/usr/bin/env python
"""Comprehensive chat eval battery: business Q&A + code generation + multi-turn.

Covers the daily-use scenario matrix:
  A. 业务代码问答（单轮）：流程 / 细节追问 / 源码 / 字段取值 / 表关联 / 接口清单 / 存在性 / 地址
  B. 代码生成（单轮）：加字段改动方案 / 按意图合成代码 / 多步查表 / 知识库接口脚手架
  C. 多轮连续对话（同一会话）：流程→追问细节→再追问（入口锁定）
  D. 生成代码后按需求调整（同一会话）：用户指出修改点 → AI 按需求调整代码

Usage:
    python scripts/eval_qa_battery.py                # 全量（真实 LLM + 索引）
    python scripts/eval_qa_battery.py --smoke        # 每类 1 条，验证脚本/耗时
    python scripts/eval_qa_battery.py --only qa      # 只跑 qa 类
    python scripts/eval_qa_battery.py --report data/eval_battery.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROJECT_ID = "demo-loan"

# --------------------------------------------------------------------------
# A. 业务代码问答（单轮）
# --------------------------------------------------------------------------
QA_CASES: list[dict] = [
    # --- 流程类 ---
    {"id": "qa_flow_disburse", "q": "放款流程什么样？从入口到下游回调串一下",
     "expect": ["放款", "落库"], "forbid": ["t_user_quota", "t_product_config"]},
    {"id": "qa_flow_repay", "q": "还款流程什么样",
     "expect": ["还款", "Repay"], "forbid": ["WithdrawalsController#withdrawals"]},
    {"id": "qa_flow_withdraw_calc", "q": "提现试算流程是什么",
     "expect": ["试算", "withDrawalCalculate", "withdrawals", "Withdrawals"], "forbid": []},
    {"id": "qa_flow_quota_adjust", "q": "额度调整流程什么样",
     "expect": ["额度", "调整", "Quota", "quotaAdjust", "入口"], "forbid": ["[PEV]"]},
    {"id": "qa_flow_risk", "q": "风控是怎么做的",
     "expect": ["风控", "risk"], "forbid": ["t_user_quota"]},
    # --- 细节追问（首轮直接问细节）---
    {"id": "qa_detail_miy", "q": "MIY的额度校验具体什么样",
     "expect": ["quotaCheckMIY", "loanSchemaQuery"], "forbid": ["未获取到.*quotaCheckMIY"]},
    {"id": "qa_detail_guarantee", "q": "融担怎么判断的",
     "expect": ["guara", "融担", "guaranteeRoute"], "forbid": []},
    {"id": "qa_detail_switch", "q": "loanSwitch干嘛的",
     "expect": ["loanSwitch", "开关"], "forbid": ["creditNotification"]},
    # --- 源码类 ---
    {"id": "qa_source_miy", "q": "MIY额度校验源码是什么",
     "expect": ["quotaCheckMIY", "```"], "forbid": ["未获取到.*quotaCheckMIY"]},
    {"id": "qa_source_entry", "q": "WithdrawalsController#withdrawals 的源码是什么",
     "expect": ["WithdrawalsController", "```"], "forbid": ["未检索到"]},
    # --- 字段/取值/落库类 ---
    {"id": "qa_field_persist", "q": "放款落库写哪、状态怎么变",
     "expect": ["落库", "insert", "Mapper", "字段", "状态"], "forbid": ["t_user_quota"]},
    {"id": "qa_field_tables", "q": "放款流程涉及哪几张表",
     "expect": ["t_", "TableName"], "forbid": ["t_user_quota", "t_credit_line"]},
    {"id": "qa_field_value", "q": "放款成功之后 demo_b_withdrawals 的状态字段会变成什么",
     "expect": ["状态", "demo_b_withdrawals"], "forbid": []},
    # --- 表关联类 ---
    {"id": "qa_table_relation", "q": "放款表和还款表怎么关联的",
     "expect": ["关联", "="], "forbid": ["t_user_quota"]},
    # --- 接口清单 / 存在性类 ---
    {"id": "qa_inventory_api", "q": "放款相关的接口有哪些",
     "expect": ["withdrawals", "接口"], "forbid": ["未检索到"]},
    {"id": "qa_inventory_exists", "q": "项目里有没有放款撤回功能",
     "expect": ["有", "revocation", "撤回", "没有"], "forbid": []},
    {"id": "qa_arch", "q": "项目有哪些模块",
     "expect": ["模块", "loan-service"], "forbid": []},
    # --- 地址/服务类 ---
    {"id": "qa_addr_feign", "q": "机构中心服务的地址是什么",
     "expect": ["url", "path", "http", "路径", "地址"], "forbid": ["未检索到"]},
    # --- 通用兜底 ---
    # --- 入口类（修复 AmountController 误匹配）---
    {"id": "qa_entry_fangkuan", "q": "放款接口主要做了什么",
     "expect": ["放款", "WithdrawalsController", "落库", "流程"],
     "forbid": ["AmountController#addDayQuota", "累加放款额"]},
    {"id": "qa_entry_fangkuan_flow", "q": "放款流程什么样的",
     "expect": ["放款", "WithdrawalsController", "落库", "回调"],
     "forbid": ["AmountController", "t_user_quota"]},
    # --- 定时任务类 ---
    {"id": "qa_job_loan_apply", "q": "放款申请定时任务（LoanApplyJob）主要做了什么",
     "expect": ["LoanApplyJob", "定时任务", "放款申请"], "forbid": ["未检索到"]},
    {"id": "qa_job_risk_result", "q": "获取风控结果定时任务是什么",
     "expect": ["activeQueryRiskCallbackInfo", "风控", "定时任务"], "forbid": ["未检索到"]},
    {"id": "qa_job_loan_approval", "q": "放款审批定时任务有哪些",
     "expect": ["queryApprovalResult", "checkAndMarkApproval", "定时任务"],
     "forbid": ["未检索到"]},
    {"id": "qa_job_overview", "q": "项目里有哪些 XXL-Job 定时任务",
     "expect": ["XxlJob", "定时任务", "LoanApplyJob", "RiskResultJob"],
     "forbid": ["未获取到"]},
    {"id": "qa_job_route_result", "q": "资金路由结果查询定时任务是什么样的",
     "expect": ["loanFundRouteResultQueryJob", "资金路由", "定时任务"],
     "forbid": ["未检索到"]},
    # --- MQ / Listener 类 ---
    {"id": "qa_listener_loan", "q": "放款结果 MQ 消费（LoanRabbitMqListenner）是什么样的",
     "expect": ["LoanRabbitMqListenner", "onMessage", "MQ", "消费"],
     "forbid": ["未检索到"]},
    {"id": "qa_listener_risk", "q": "风控决策结果 MQ 怎么消费的",
     "expect": ["RiskControlRabbitMQListener", "MQ", "消费", "风控结果"],
     "forbid": ["未检索到"]},
    {"id": "qa_listener_repay", "q": "还款结果 MQ 消费是什么样的",
     "expect": ["RepaymentListener", "MQ", "消费", "还款"],
     "forbid": ["未检索到"]},
    {"id": "qa_listener_overview", "q": "项目里有哪些 MQ 消息消费者（Listener）",
     "expect": ["Listener", "MQ", "消费", "RabbitListener"],
     "forbid": ["未获取到"]},
    {"id": "qa_listener_cancel", "q": "放款撤销 MQ 消费是怎么处理的",
     "expect": ["CancellationListener", "放款撤销", "MQ"],
     "forbid": ["未检索到"]},
    # --- 通用兜底 ---
    {"id": "qa_general", "q": "什么是数据库事务",
     "expect": ["事务", "ACID", "原子"], "forbid": ["[PEV]"]},
    {"id": "qa_general_code", "q": "用 Java 写一个快速排序",
     "expect": ["```"], "forbid": ["mvn", "编译"]},
]

# --------------------------------------------------------------------------
# B. 代码生成（单轮）
# --------------------------------------------------------------------------
CODEGEN_CASES: list[dict] = [
    # --- 加字段改动方案 ---
    {"id": "cg_change_add_field", "q": "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
     "expect": ["改动顺序", "DDL", "WithdrawalsApplyReqVo"],
     "forbid": ["业务状态流转", "评审提示", "re:必须(?:改|要).{0,8}Mapper", "LoanCoupon", "[PEV]"]},
    {"id": "cg_change_add_field_zh", "q": "放款接口加一个入参 并落库，改动清单",
     "expect": ["改动清单", "DDL"], "forbid": ["[PEV][TEST]", "契约缺口"]},
    {"id": "cg_change_ext", "q": "放款接口增加字段并透传给下游，要改哪些地方",
     "expect": ["改动清单", "透传"], "forbid": ["[PEV][TEST]"]},
    # change-plan story must stay ONE ask (no N2 newField placeholder duplicate)
    # — desktop-chat-ed1e15401152 regression.
    {"id": "cg_change_story", "q": "我想在放款接口加个字段 labor 然后并落到放款表 给我对应的代码改动清单",
     "expect": ["改动清单", "DDL", "labor"], "forbid": ["newField", "按你的提问顺序作答", "[PEV]"]},
    # --- 按意图合成代码 ---
    {"id": "cg_compose_simple", "q": "给我一段代码：根据借据号查询放款表",
     "expect": ["```", "QueryWrapper", "deal_ref_no"], "forbid": ["TODO", "LoanCoupon", "[PEV]"]},
    {"id": "cg_compose_pipeline", "q": "给我一段代码 根据借据号查询放款表，然后用放款表的记录查放款优惠券表，查到券信息后，调用外部接口对券进行支用",
     "expect": ["```"], "forbid": ["LoanCoupon", "[PEV][TEST]", "契约缺口"]},
    {"id": "cg_compose_count", "q": "给我一段代码 统计某客户放款成功的笔数",
     "expect": ["```", "count"], "forbid": ["[PEV]"]},
    # --- 知识库接口脚手架 ---
    {"id": "cg_addapi", "q": "根据知识库文档生成一个登录接口",
     "expect": ["```", "Controller"], "forbid": ["[PEV][TEST]"]},
    # add-api + business-step orchestration must stay ONE ask (no 4× duplicate
    # scaffold) — desktop-chat-fa14438a45aa regression.
    {"id": "cg_addapi_story", "q": "我想新增一个接口，接口具体文档如 知识库商品查询，service中要查询放款表，然后再查询优惠券表，再调用acc-service，然后组装代码返回",
     "expect": ["知识库脚手架", "```", "GoodsQuery"], "forbid": ["[PEV][TEST]", "按你的提问顺序作答"]},
]

# --------------------------------------------------------------------------
# C/D. 多轮连续对话（同一 session）
# --------------------------------------------------------------------------
# 每项：turns=[...] 依次发问，逐轮判定
MULTI_TURN_CASES: list[dict] = [
    {
        "id": "multi_qa_lock",
        "turns": [
            {"q": "放款流程什么样？从入口到下游回调串一下",
             "expect": ["放款", "落库"], "forbid": ["t_user_quota"]},
            {"q": "风控是怎么做的",
             "expect": ["风控", "risk"], "forbid": ["AccClient#", "AmountController"]},
            {"q": "那 MIY 额度校验呢",
             "expect": ["MIY", "quotaCheckMIY", "额度"], "forbid": ["AccClient#"]},
            {"q": "放款落库写哪",
             "expect": ["落库", "insert", "Mapper", "字段", "写库"], "forbid": ["t_user_quota"]},
        ],
    },
    {
        "id": "multi_qa_switch_entry",
        "turns": [
            {"q": "还款流程什么样",
             "expect": ["还款", "Repay"], "forbid": ["WithdrawalsController#withdrawals"]},
            {"q": "那放款从入口到落库串一下",
             "expect": ["放款", "落库", "withdrawals"], "forbid": []},
        ],
    },
    {
        "id": "multi_code_adjust",  # 核心：生成代码 → 按用户需求逐步调整
        "turns": [
            {"q": "给我一段代码：根据借据号查询放款表",
             "expect": ["```", "QueryWrapper"], "forbid": ["TODO", "[PEV]"]},
            {"q": "不对，查询条件改成用实体字段的 LambdaQueryWrapper 方式，而且只取状态为成功的记录",
             "expect": ["LambdaQueryWrapper", "成功", "状态"], "forbid": ["[PEV][TEST]"]},
            {"q": "再改一下，如果没查到记录就抛业务异常，异常信息写清楚",
             "expect": ["异常", "throw", "Exception"], "forbid": ["[PEV][TEST]"]},
        ],
    },
    {
        "id": "multi_code_adjust_field",
        "turns": [
            {"q": "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
             "expect": ["改动顺序", "DDL"], "forbid": ["[PEV]"]},
            # 修改清单引用旧字段名（把 businessLabel 改为 marketLabel）是合理行为；
            # 判定点：AI 理解"字段名改掉"并重新给出方案，方案含新名且结构完整。
            {"q": "这个字段名不对，应该叫 marketLabel，重新给出改动方案",
             "expect": ["marketLabel", "改动"], "forbid": ["[PEV][TEST]", "契约缺口"]},
        ],
    },
    {
        "id": "multi_deep_dig",
        "turns": [
            {"q": "放款接口什么流程",
             "expect": ["放款", "流程"], "forbid": []},
            {"q": "流程里的校验具体校验了哪些字段",
             "expect": ["校验", "字段"], "forbid": ["未检索到"]},
            {"q": "这些字段校验失败返回什么错误码",
             "expect": ["错误码", "返回", "失败"], "forbid": []},
        ],
    },
]

# 归一化 expect/forbid：支持正则（前缀 re:）
def _match_list(answer: str, items: list[str]) -> tuple[list[str], list[str]]:
    hits, bans = [], []
    a = answer or ""
    for it in items:
        if it.startswith("re:"):
            pat = it[3:]
            if re.search(pat, a):
                hits.append(it)
        else:
            if it in a:
                hits.append(it)
    for it in items:
        if it.startswith("re:"):
            pat = it[3:]
            if re.search(pat, a):
                bans.append(it)
        else:
            if it in a:
                bans.append(it)
    return hits, bans


async def _ask(question: str, session_id: str) -> tuple[str, float, str]:
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings

    get_settings.cache_clear()
    t0 = time.time()
    result = await run_chat(question, project_id=PROJECT_ID, session_id=session_id)
    elapsed = time.time() - t0
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    answer = str(data.get("answer") or data.get("reply") or "")
    return answer, elapsed, data.get("intent") or ""


def _judge_case(c: dict, answer: str) -> dict:
    expect = c.get("expect") or []
    forbid = c.get("forbid") or []
    hits = [x for x in expect if x in (answer or "")]
    bans = [x for x in forbid if x in (answer or "")]
    ok = bool(hits) and not bans
    return {"pass": ok, "hits": hits, "banned": bans, "answer_head": (answer or "")[:180]}


async def _run_single(c: dict) -> dict:
    sid = f"eval-single-{uuid.uuid4().hex[:8]}"
    answer, elapsed, intent = await _ask(c["q"], sid)
    j = _judge_case(c, answer)
    return {
        "id": c["id"],
        "question": c["q"],
        "elapsed_s": round(elapsed, 1),
        "intent": intent,
        **j,
        "answer": answer,
    }


async def _run_multi(c: dict) -> dict:
    sid = f"eval-multi-{uuid.uuid4().hex[:8]}"
    turns_out: list[dict] = []
    for i, t in enumerate(c["turns"]):
        answer, elapsed, intent = await _ask(t["q"], sid)
        j = _judge_case(t, answer)
        turns_out.append(
            {
                "turn": i + 1,
                "question": t["q"],
                "elapsed_s": round(elapsed, 1),
                "intent": intent,
                **j,
                "answer": answer,
            }
        )
    ok_all = all(t["pass"] for t in turns_out)
    return {
        "id": c["id"],
        "kind": "multi_turn",
        "pass": ok_all,
        "turns": turns_out,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每类只跑 1 条")
    ap.add_argument("--only", default="", help="qa | codegen | multi")
    ap.add_argument("--ids", default="", help="只跑指定 case id（逗号分隔）")
    ap.add_argument("--report", default=str(ROOT / "data" / "eval_battery.json"))
    args = ap.parse_args()

    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()

    only = args.only
    ids = {x.strip() for x in args.ids.split(",") if x.strip()}
    results: dict[str, list] = {}
    total_ok = total = 0

    if only in {"", "qa"}:
        cases = QA_CASES[:1] if args.smoke else QA_CASES
        if ids:
            cases = [c for c in cases if c["id"] in ids]
        print(f"\n== A. 业务代码问答 ({len(cases)}) ==", flush=True)
        results["qa"] = []
        for i, c in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {c['id']}: {c['q'][:40]}", flush=True)
            r = await _run_single(c)
            results["qa"].append(r)
            total_ok += int(r["pass"]); total += 1
            print(f"    {'PASS' if r['pass'] else 'FAIL'} ({r['elapsed_s']}s) hits={r['hits']} banned={r['banned']}", flush=True)

    if only in {"", "codegen"}:
        cases = CODEGEN_CASES[:1] if args.smoke else CODEGEN_CASES
        if ids:
            cases = [c for c in cases if c["id"] in ids]
        print(f"\n== B. 代码生成 ({len(cases)}) ==", flush=True)
        results["codegen"] = []
        for i, c in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {c['id']}: {c['q'][:40]}", flush=True)
            r = await _run_single(c)
            results["codegen"].append(r)
            total_ok += int(r["pass"]); total += 1
            print(f"    {'PASS' if r['pass'] else 'FAIL'} ({r['elapsed_s']}s) hits={r['hits']} banned={r['banned']}", flush=True)

    if only in {"", "multi"}:
        cases = MULTI_TURN_CASES[:1] if args.smoke else MULTI_TURN_CASES
        if ids:
            cases = [c for c in cases if c["id"] in ids]
        print(f"\n== C/D. 多轮连续对话 ({len(cases)}) ==", flush=True)
        results["multi"] = []
        for i, c in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] {c['id']} ({len(c['turns'])} 轮)", flush=True)
            r = await _run_multi(c)
            results["multi"].append(r)
            total_ok += int(r["pass"]); total += 1
            for t in r["turns"]:
                print(f"    轮{t['turn']} {'PASS' if t['pass'] else 'FAIL'} ({t['elapsed_s']}s) hits={t['hits']} banned={t['banned']}", flush=True)
            print(f"    multi overall: {'PASS' if r['pass'] else 'FAIL'}", flush=True)

    print(f"\n== 汇总: {total_ok}/{total} ==", flush=True)
    report = {
        "project_id": PROJECT_ID,
        "total_ok": total_ok,
        "total": total,
        "results": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"report -> {args.report}", flush=True)
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
