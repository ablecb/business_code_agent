#!/usr/bin/env python
"""Expanded battery runner for docs/测试问题集-扩展版.md (source mode, no middleware).

Runs the 81-question set (A business Q&A / B codegen / C traps / D historical
regressions + multi-turn cases) in-process via app.agent.graph_agent.run_chat
with:
  - in-memory sessions (_USE_MEMORY=True) — no MySQL required
  - EXPLORE_BACKEND=source — filesystem tool loop instead of DB index
  - GOLDEN_QA_ENABLED=false — no golden_qa table
  - SEARCH_USE_MILVUS=false — no Milvus
  - real LLM from .env (siliconflow Qwen3-32b)

Judging: keyword expect-any / forbid-any + optional tool-evidence checks
(get_symbol_code count / facet reads) extracted from explore_steps.

Usage:
    python scripts/eval_expanded_battery.py                    # all cases
    python scripts/eval_expanded_battery.py --only single      # A+B+C+D single-turn only
    python scripts/eval_expanded_battery.py --only multi       # multi-turn only
    python scripts/eval_expanded_battery.py --ids qa_flow_disburse,cg_add_field
    python scripts/eval_expanded_battery.py --smoke            # 1 per section, quick sanity
    python scripts/eval_expanded_battery.py --concurrency 3
    python scripts/eval_expanded_battery.py --report data/eval_expanded.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---- env tunings (DO NOT override EXPLORE_BACKEND — let .env decide index vs source) ----
os.environ["LLM_LOG_PROMPTS"] = "false"
os.environ.setdefault("AGENT_WRITE_MODE", "workspace")

PROJECT_ID = "demo-loan"

FAKE_TABLES = [
    "t_user_quota",
    "t_user_credit_limit",
    "t_product_config",
    "t_credit_line",
    "t_quota_info",
    "t_miy_quota",
]


# ==========================================================================
# A. 业务细节问答（单轮，81 题中的 A 类）
# ==========================================================================
QA_CASES: list[dict] = [
    # ---- A1 放款/提现 ----
    {"id": "qa_adjustment_title", "q": "放款流程什么样？从入口到下游回调串一下",
     "expect": ["放款", "落库", "回调"], "forbid": FAKE_TABLES,
     "entry": "WithdrawalsController#withdrawals"},
    {"id": "A1.2", "q": "放款申请到放款成功，中间经过了哪些 HTTP 入口",
     "expect": ["申请", "withdrawals"], "forbid": FAKE_TABLES,
     "entry_hint": "WithdrawalsController"},
    {"id": "A1.3", "q": "放款失败时有哪些兜底逻辑，异常怎么处理",
     "expect": ["兜底", "重试", "RetryLoan", "异常"], "forbid": FAKE_TABLES,
     "honest_miss_ok": True},
    {"id": "A1.4", "q": "放款接口返回结果里的 status 是怎么判断的",
     "expect": ["status", "结果查询", "枚举", "queryWithdrawalsResult"], "forbid": FAKE_TABLES,
     "honest_miss_ok": True},
    {"id": "A1.5", "q": "放款时如果风控校验不过会走什么分支",
     "expect": ["风控", "拒绝", "挂起", "Risk"], "forbid": FAKE_TABLES,
     "honest_miss_ok": True},
    {"id": "A1.6", "q": "放款/提现试算（计算）接口返回哪些关键结果",
     "expect": ["试算", "计算", "Calculate", "金额", "费用"], "forbid": FAKE_TABLES},
    # ---- A2 还款/代扣/冲正 ----
    {"id": "A2.1", "q": "还款流程是什么样的",
     "expect": ["还款", "Repay"], "forbid": ["WithdrawalsController#withdrawals"],
     "entry": "RepayController#repayment"},
    {"id": "A2.2", "q": "还款结果怎么通知下游/渠道",
     "expect": ["Listener", "MQ", "队列", "重组", "repush"], "forbid": FAKE_TABLES,
     "entry_hint": "RepaymentListener"},
    {"id": "A2.3", "q": "主动还款和代扣还款有什么差别（还款来源）",
     "expect": ["代扣", "主动", "deduction"], "forbid": FAKE_TABLES,
     "entry_hint": "DeductionNoticeController"},
    {"id": "A2.4", "q": "还款试算 repayCalculate 主要做了什么",
     "expect": ["repayCalculate", "试算", "还款计划", "PaymentPlan"], "forbid": FAKE_TABLES},
    {"id": "A2.5", "q": "还款冲正/撤销流程",
     "expect": ["冲正", "撤销", "reverse", "Reverse"], "forbid": ["WithdrawalsController#revocation"],
     "entry_hint": "RepayReverseController"},
    {"id": "A2.6", "q": "逾期/催收还款（collection）怎么处理",
     "expect": ["催收", "CollectionRepay", "批量"], "forbid": FAKE_TABLES},
    # ---- A3 额度/授信 ----
    {"id": "A3.1", "q": "授信通知怎么实现的",
     "expect": ["creditNotification", "授信", "Credit"], "forbid": ["WithdrawalsController#withdrawals", "loanSwitch"],
     "entry": "CreditController#creditNotification"},
    {"id": "A3.2", "q": "授信额度查询/共享是怎么做的",
     "expect": ["creditQuery", "共享", "CreditQuotaShare"], "forbid": FAKE_TABLES},
    {"id": "A3.3", "q": "额度调整有哪些类型和策略",
     "expect": ["额度调整", "quotaAdjust", "正向", "反向", "策略"], "forbid": FAKE_TABLES,
     "entry_hint": "QuotaController"},
    {"id": "A3.4", "q": "FgAdjust/担保额度回调",
     "expect": ["FgAdjust", "担保", "回调", "额度"], "forbid": ["WithdrawalsController#withdrawals"]},
    {"id": "A3.5", "q": "额度调整结果怎么同步给渠道",
     "expect": ["同步", "通知", "渠道", "队列"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "A3.6", "q": "用信/授信使用（UseCredit）流程",
     "expect": ["用信", "UseCredit", "use credit"], "forbid": ["creditNotification 当主链"],
     "entry_hint": "UseCreditLoanListenner"},
    # ---- A4 借据/助贷/优惠券 ----
    {"id": "A4.1", "q": "借据与放款记录怎么关联",
     "expect": ["借据", "deal_ref_no", "withdrawals"], "forbid": FAKE_TABLES},
    {"id": "A4.2", "q": "助贷（撮合）申请流程",
     "expect": ["助贷", "LoanAssistance", "申请"], "forbid": FAKE_TABLES,
     "entry_hint": "LoanAssistanceApplyController"},
    {"id": "A4.3", "q": "助贷放款后给渠道的扩展信息怎么落",
     "expect": ["助贷扩展", "LoanAssistanceExt"], "forbid": FAKE_TABLES,
     "honest_miss_ok": True},
    {"id": "A4.4", "q": "放款优惠券怎么用的",
     "expect": ["优惠券", "Coupon", "占用", "核销"], "forbid": FAKE_TABLES,
     "honest_miss_ok": True},
    # ---- A5 路由/渠道/代偿/融担 ----
    {"id": "A5.1", "q": "产品路由/渠道路由怎么配的",
     "expect": ["路由", "route", "渠道", "product_route"], "forbid": FAKE_TABLES,
     "entry_hint": "RouteController"},
    {"id": "A5.2", "q": "融担担保回调怎么处理",
     "expect": ["担保", "guarantee", "loanCallback", "融担"], "forbid": ["WithdrawalsController#withdrawals"],
     "entry": "AboutFinanceGuaranteeController#loanCallback"},
    {"id": "A5.3", "q": "放款风控回调（异步决策）入口在哪",
     "expect": ["风控回调", "RiskControl", "Listener", "consume"], "forbid": ["AboutFinanceGuaranteeController#loanCallback"],
     "entry": "RiskControlRabbitMQListener#consume"},
    {"id": "A5.4", "q": "Hx/Union 渠道报文适配是怎么做的",
     "expect": ["CoreAdapter", "UnionAdapter", "报文", "渠道"], "forbid": ["编造类名"]},
    # ---- A6 黑名单/拦截 ----
    {"id": "A6.1", "q": "黑名单审批和操作流程",
     "expect": ["黑名单", "Blacklist", "审批"], "forbid": FAKE_TABLES},
    {"id": "A6.2", "q": "放款前有没有拦截校验",
     "expect": ["拦截", "Intercept"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    # ---- A7 退款/发票/签章/内部 ----
    {"id": "A7.1", "q": "退款流程（未入账/已入账）",
     "expect": ["退款", "refund", "未入账"], "forbid": FAKE_TABLES},
    {"id": "A7.2", "q": "发票开具流程",
     "expect": ["发票", "invoice", "开票", "红冲"], "forbid": FAKE_TABLES,
     "entry_hint": "InvoiceController"},
    {"id": "A7.3", "q": "电子签章（签名）怎么发起的",
     "expect": ["签章", "signature", "签约"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "A7.4", "q": "内部记账/开户接口",
     "expect": ["内部", "Internal", "generateAccount"], "forbid": FAKE_TABLES},
    {"id": "A7.5", "q": "结清证明申请接口主要流程",
     "expect": ["结清", "证明"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    # ---- A8 MQ/定时任务 ----
    {"id": "A8.1", "q": "放款申请是怎么异步进队列的",
     "expect": ["队列", "MQ", "create_order", "异步"], "forbid": FAKE_TABLES,
     "entry_hint": "LoanRabbitMqListenner"},
    {"id": "A8.2", "q": "还款回调失败怎么重推",
     "expect": ["重推", "repush", "定时", "失败重发"], "forbid": FAKE_TABLES,
     "entry_hint": "RepaymentCallbacksTime"},
    {"id": "A8.3", "q": "放款审批结果定时轮询",
     "expect": ["审批", "轮询", "queryApprovalResult", "定时"], "forbid": FAKE_TABLES,
     "entry_hint": "LoanApprovalJob"},
    {"id": "A8.4", "q": "融担扫描任务做什么的",
     "expect": ["融担扫描", "FinanceGuaranteeScan", "定时"], "forbid": FAKE_TABLES},
    {"id": "A8.5", "q": "代扣/限额等开关类定时任务有哪些",
     "expect": ["开关", "Switch", "定时", "日限额"], "forbid": ["编造任务名"], "honest_miss_ok": True},
]

# ==========================================================================
# B. 根据需求生成高质量代码（单轮）
# ==========================================================================
CODEGEN_CASES: list[dict] = [
    {"id": "B1", "q": "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
     "expect": ["## 改动顺序", "## 改动清单", "WithdrawalsApplyReqVo", "reqVo2ExtDmo", "WithdrawalsExt", "## DDL"],
     "forbid": ["业务状态流转", "评审提示", "必改 ExtMapper.xml", "t_user_quota"],
     "tool_min_get_symbol_code": 4,
     "facets": {
         "vo": re.compile(r"(?i)WithdrawalsApplyReqVo|ApplyReqVo"),
         "bind": re.compile(r"(?i)checkApplyReqParam"),
         "assemble": re.compile(r"(?i)reqVo2ExtDmo"),
         "ext": re.compile(r"(?i)WithdrawalsExt"),
     },
     "notes": "FieldNameConstants 常量风格：回答应出现 常量/FieldNameConstants，或至少不用裸字符串"},
    {"id": "B2", "q": "还款接口增加一个入参 newField，先透传给下游再落库，改动清单",
     "expect": ["改动清单", "改动顺序", "透传", "落库", "Repay"],
     "forbid": ["WithdrawalsApplyReqVo", "t_user_quota"]},
    {"id": "B3", "q": "放款接口加一个字段 batchNo 并透传给渠道，要改哪些",
     "expect": ["改动清单", "透传", "渠道", "同步"], "forbid": ["业务状态流转", "评审提示"]},
    {"id": "B4", "q": "给我一段代码：根据借据号查放款表，再按结果查助贷表",
     "expect": ["```java", "QueryWrapper", "deal_ref_no", "LoanAssistance"], "forbid": ["t_user_quota"]},
    {"id": "B5", "q": "给我一段代码：通过 order_id 查放款表，再查放款优惠券表和放款审批表",
     "expect": ["```java", "QueryWrapper", "order_id", "CouponInfo"],
     "forbid": ["t_user_quota", "t_credit_line"]},
    {"id": "B6", "q": "给我一个方法：查机构中心的报文取字段（如 limitTree、额度树）",
     "expect": ["```java", "aqc", "queryLimitTreeBase", "机构中心"], "forbid": ["编造类名"]},
    {"id": "B7", "q": "按知识库「商品查询」文档新增一个查询接口",
     "expect": ["Controller", "Request<GoodsQueryReqVo>", "pageNum", "total"],
     "forbid": ["Request<Map", "t_product_config"]},
    {"id": "B8", "q": "按知识库「登录」接口文档实现登录接口",
     "expect": ["Controller", "Service", "Request<LoginReqVo>", "Response<LoginRespVo>"],
     "forbid": ["Request<Map", "t_user_quota"]},
    {"id": "B9", "q": "根据知识库「商品查询」文档，只帮我生成对应的 Java 实体类",
     "expect": ["仅实体类", "GoodsQueryReqVo", "GoodsQueryRespVo"],
     "forbid": ["public class GoodsQueryController", "ServiceImpl"]},
    {"id": "B10", "q": "我想新增一个接口：入参是商户号和门店号，查询该机构当天可用额度",
     "expect": ["新增接口", "ReqVo", "额度", "org_day_quota"], "forbid": ["Request<Map", "t_user_quota"],
     "honest_miss_ok": True},
    {"id": "B11", "q": "帮我写一个 MQ 消费者骨架：消费放款申请创建订单消息，落 demo_b_withdrawals_apply",
     "expect": ["```java", "@RabbitListener", "create_order", "Mapper"],
     "forbid": ["t_user_quota", "编造队列名"]},
    {"id": "B12", "q": "帮我写一个 XXL-Job 定时任务：扫描放款审批结果",
     "expect": ["```java", "@XxlJob", "queryApprovalResult"], "forbid": ["编造任务名"]},
    {"id": "B13", "q": "给我一段代码：还款成功回调里把结果重推给渠道（失败自动补推）",
     "expect": ["```java", "重推", "repush", "回调"], "forbid": FAKE_TABLES},
    {"id": "B14", "q": "放款接口增加字段时顺便生成对应的 DDL 语句",
     "expect": ["```sql", "ALTER TABLE", "demo_b_withdrawals_ext", "加列"], "forbid": ["t_user_quota"]},
    {"id": "B15", "q": "帮我把 WithdrawalsController 里某段重复代码抽成一个公共方法",
     "expect": ["重构", "抽取", "公共方法"], "forbid": ["t_user_quota"], "honest_miss_ok": True},
    {"id": "B16", "q": "放款校验逻辑里增加一个「商户当天放款笔数上限」的规则校验",
     "expect": ["校验", "笔数上限", "count", "demo_b_withdrawals"], "forbid": ["t_user_quota"],
     "honest_miss_ok": True},
]

# ==========================================================================
# C. 反例/陷阱（单轮）
# ==========================================================================
TRAP_CASES: list[dict] = [
    {"id": "C1", "q": "放款流程写到了哪张表？", "expect": ["表", "落库"], "forbid": FAKE_TABLES},
    {"id": "C2", "q": "放款流程涉及哪些字段？", "expect": ["字段"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "C3", "q": "还款流程用到哪些表？", "expect": ["表", "demo_b_repay", "b_loan"], "forbid": ["t_user_quota", "demo_b_withdrawals 当主表"]},
    {"id": "C8", "q": "用 Java 写一个快速排序", "expect": ["```java", "sort"], "forbid": ["mvn", "编译", "[PEV]"]},
    {"id": "C9", "q": "什么是数据库事务", "expect": ["事务", "ACID", "原子"], "forbid": ["[PEV]"]},
    {"id": "C10", "q": "写个 Java 方法把字符串转大驼峰", "expect": ["```java", "toUpperCamel", "驼峰"], "forbid": ["mvn"]},
    {"id": "C11", "q": "改一个不存在的表 t_user_quota 的字段", "expect": ["未在工程中找到", "不存在"], "forbid": ["t_user_quota 存在"]},
    {"id": "C12", "q": "生成一个工程里不存在的类 XxxController 并要我改代码", "expect": ["未找到", "证据不足", "不存在"], "forbid": ["顺着编造"]},
    {"id": "C13", "q": "放款接口增加字段 businessLabel 并落库", "expect": ["改动", "落库"], "forbid": ["只解释现状"]},
    {"id": "C14", "q": "把放款流程讲清楚就行，不要改代码", "expect": ["放款", "流程"], "forbid": ["diff", "补丁", "修改代码"]},
    {"id": "C15", "q": "What is the loan disbursement flow?",
     "expect": ["WithdrawalsController", "放款", "flow", "disbursement"], "forbid": ["闲聊"]},
    {"id": "C16", "q": "Add a field businessLabel to the disbursement API and persist it — what needs to change?",
     "expect": ["businessLabel", "Change", "改动", "plan"], "forbid": ["只解释现状"]},
    {"id": "C17", "q": "放款接口地址是什么", "expect": ["url", "path", "地址", "http"], "forbid": ["未检索到"]},
    {"id": "C18", "q": "风控/审批是同步还是异步", "expect": ["同步", "异步", "MQ"], "forbid": FAKE_TABLES},
]

# ==========================================================================
# D. 真实历史失败回归（单轮）
# ==========================================================================
HIST_CASES: list[dict] = [
    {"id": "D1", "q": "给我一段代码：根据借据号查放款表，然后用放款表的记录查放款优惠券表，查到券信息后，调用外部接口对券进行占用",
     "expect": ["```java", "QueryWrapper", "占用", "Coupon"], "forbid": ["不会编造", "证据不足", "t_user_quota"]},
    {"id": "D2", "q": "根据知识库商品查询文档生成 Java 接口代码，先校验、查放款表、调 acc-service、打印、返回",
     "expect": ["Controller", "GoodsQuery", "demo_b_withdrawals", "aqc"], "forbid": ["不会编造", "证据不足"]},
    {"id": "D3", "q": "我想新增一个接口，接口文档如知识库商品查询，service 中要查询放款表，然后查优惠券表，再调 acc-service",
     "expect": ["Controller", "GoodsQuery", "demo_b_withdrawals", "CouponInfo"], "forbid": ["不会编造", "证据不足"]},
    {"id": "D4", "q": "放款失败时有哪些兜底逻辑，异常怎么处理",
     "expect": ["兜底", "重试", "异常"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "D5", "q": "放款接口返回结果里的 status 是怎么判断的",
     "expect": ["status", "判断"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "D6", "q": "放款时如果风控校验不过会走什么分支",
     "expect": ["风控", "分支"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "D7", "q": "结清证明申请接口 主要流程是什么",
     "expect": ["结清", "证明"], "forbid": FAKE_TABLES, "honest_miss_ok": True},
    {"id": "D8", "q": "根据知识库 商品查询 帮我生成对应的 java 实体类",
     "expect": ["仅实体类", "GoodsQueryReqVo", "GoodsQueryRespVo"], "forbid": ["ServiceImpl", "GoodsQueryController"]},
    {"id": "D9", "q": "给我一段代码：通过 order_id 查询放款表，查到结果后，再查询放款优惠券表和放款审批表",
     "expect": ["```java", "QueryWrapper", "order_id", "CouponInfo"], "forbid": ["未找到", "t_user_quota"]},
]

# ==========================================================================
# 多轮（同一 session）
# ==========================================================================
MULTI_CASES: list[dict] = [
    {
        "id": "MT1",  # C5: 放款主链入口锁定
        "turns": [
            {"q": "放款流程什么样？从入口到下游回调串一下", "expect": ["放款", "落库"], "forbid": FAKE_TABLES},
            {"q": "风控是怎么做的", "expect": ["风控", "risk"], "forbid": ["AmountController", "AccClient#"]},
            {"q": "那 MIY 额度校验呢", "expect": ["MIY", "quotaCheckMIY"], "forbid": FAKE_TABLES},
            {"q": "loanSwitch 干嘛的", "expect": ["loanSwitch", "开关"], "forbid": ["creditNotification"]},
        ],
        "keep_entry": "WithdrawalsController#withdrawals",
    },
    {
        "id": "MT2",  # C4: 还款 → 放款 入口切换
        "turns": [
            {"q": "还款流程是什么样的", "expect": ["还款", "Repay"], "forbid": ["WithdrawalsController#withdrawals"]},
            {"q": "那放款从入口到落库串一下", "expect": ["放款", "落库", "withdrawals"], "forbid": []},
        ],
        "switch_to": "WithdrawalsController#withdrawals",
    },
    {
        "id": "MT3",  # C6: 风控回调 → 融担担保回调 入口切题
        "turns": [
            {"q": "风控回调入口在哪", "expect": ["RiskControlRabbitMQListener", "风控回调"], "forbid": ["AboutFinanceGuaranteeController#loanCallback"]},
            {"q": "那融担担保回调呢", "expect": ["AboutFinanceGuaranteeController#loanCallback", "融担"], "forbid": ["RiskControlRabbitMQListener"]},
        ],
    },
    {
        "id": "MT4",  # 细节深挖
        "turns": [
            {"q": "放款接口什么流程", "expect": ["放款", "流程"], "forbid": []},
            {"q": "流程里的校验具体校验了哪些字段", "expect": ["校验", "字段"], "forbid": ["未检索到"]},
            {"q": "这些字段校验失败返回什么错误码", "expect": ["错误码", "返回", "失败"], "forbid": [], "honest_miss_ok": True},
        ],
    },
    {
        "id": "MT5",  # B1 → 改字段名 marketLabel
        "turns": [
            {"q": "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些",
             "expect": ["改动顺序", "DDL"], "forbid": ["[PEV]"]},
            {"q": "这个字段名不对，应该叫 marketLabel，重新给出改动方案",
             "expect": ["marketLabel", "改动"], "forbid": ["businessLabel 当主词"]},
        ],
    },
    {
        "id": "MT6",  # 历史高频 代码合成 + 调整
        "turns": [
            {"q": "给我一段代码：根据借据号查询放款表", "expect": ["```java", "QueryWrapper"], "forbid": ["TODO"]},
            {"q": "不对，查询条件改成用实体字段的 LambdaQueryWrapper 方式，而且只取状态为成功的记录",
             "expect": ["LambdaQueryWrapper", "成功", "状态"], "forbid": ["[PEV]"]},
        ],
    },
]

HONEST_MISS_RE = re.compile(
    r"未(?:在|能)(?:已读材料|已读|工程|代码库|索引|知识库)?中?找到|未检索到|证据不足|"
    r"不会编造|没有(?:找到|发现|检索到)|未找到(?:对应|相关)?(?:入口|实现|代码|方法)|"
    r"未在已读材料中看到|不存在于(?:工程|代码库)|无法给出(?:完整|准确)代码|只能在证据范围内",
    re.I,
)


def _match_any(answer: str, items: list[str]) -> list[str]:
    a = answer or ""
    hits: list[str] = []
    for it in items:
        if it.startswith("re:"):
            if re.search(it[3:], a):
                hits.append(it)
        elif it in a:
            hits.append(it)
    return hits


def _match_forbid(answer: str, items: list[str]) -> list[str]:
    a = answer or ""
    banned: list[str] = []
    for it in items:
        if it.startswith("re:"):
            if re.search(it[3:], a):
                banned.append(it)
        elif it in a:
            banned.append(it)
    return banned


def _judge_single(c: dict, answer: str, steps: list | None) -> dict:
    a = answer or ""
    expect = c.get("expect") or []
    forbid = c.get("forbid") or []
    hits = _match_any(a, expect)
    banned = _match_forbid(a, forbid)
    issues: list[str] = []
    if not hits:
        # 允许“诚实说没找到”作为通过（honest_miss_ok 且没有编造）
        if c.get("honest_miss_ok") and HONEST_MISS_RE.search(a) and not banned:
            pass
        else:
            issues.append("no expect hit")
    if banned:
        issues.append(f"forbidden: {banned}")

    # 工具证据（代码生成类）
    reads, facet_hit, code_reads = _collect_evidence(steps)
    min_code = c.get("tool_min_get_symbol_code") or 0
    if min_code and code_reads < min_code:
        issues.append(f"get_symbol_code {code_reads} < {min_code}")
    facets = c.get("facets")
    if facets:
        missing = [k for k, v in facets.items() if not v]
        if missing:
            issues.append(f"facets not read: {missing}")
    # 痕迹检查
    if "[PEV]" in a:
        issues.append("[PEV] tag leaked into answer")
    if c.get("notes") and c.get("notes_check", True):
        if "FieldNameConstants" in c.get("notes", ""):
            if not re.search(r"FieldNameConstants|常量类|常量", a):
                issues.append("bare field string instead of constants style")

    ok = not issues
    return {
        "pass": ok,
        "issues": issues,
        "hits": hits,
        "banned": banned,
        "code_reads": code_reads,
        "facet_hit": facet_hit,
        "reads": reads[:10],
    }


def _collect_evidence(steps: list | None) -> tuple[list[str], dict[str, bool], int]:
    reads: list[str] = []
    facet_hit: dict[str, bool] = {}
    code_reads = 0
    for s in (steps or []):
        if not isinstance(s, dict):
            continue
        tool = str(s.get("tool") or "")
        if tool != "get_symbol_code" or not s.get("ok"):
            continue
        code_reads += 1
        fqcn = str(s.get("fqcn") or "")
        method = str(s.get("method_name") or "")
        fpath = str(s.get("file_path") or "")
        blob = f"{fqcn} {method} {fpath} {s.get('args') or ''}"
        reads.append(f"{fqcn or '?'}#{method or '?'} {fpath}")
        for k, rx in (s.get("_facets") or {}).items():
            if rx.search(blob):
                facet_hit[k] = True
    return reads, facet_hit, code_reads


def _run_facets(c: dict, steps: list | None) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for k, rx in (c.get("facets") or {}).items():
        out[k] = False
        for s in (steps or []):
            if not isinstance(s, dict):
                continue
            if str(s.get("tool") or "") != "get_symbol_code" or not s.get("ok"):
                continue
            fqcn = str(s.get("fqcn") or "")
            method = str(s.get("method_name") or "")
            fpath = str(s.get("file_path") or "")
            args = s.get("args") or {}
            blob = f"{fqcn} {method} {fpath} {' '.join(str(v) for v in args.values())}"
            if rx.search(blob):
                out[k] = True
                break
    return out


async def _ask(question: str, session_id: str) -> tuple[str, float, str, list]:
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings

    get_settings.cache_clear()
    t0 = time.time()
    result = await run_chat(question, project_id=PROJECT_ID, session_id=session_id)
    elapsed = time.time() - t0
    data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
    answer = str(data.get("answer") or data.get("reply") or "")
    intent = str(data.get("intent") or data.get("goal") or "")
    steps = data.get("explore_steps") or None
    if isinstance(steps, list):
        for s in steps:
            if isinstance(s, dict) and "args" in s and isinstance(s["args"], dict):
                pass
    return answer, elapsed, intent, steps


async def _run_single(c: dict) -> dict:
    sid = f"eval-exp-{uuid.uuid4().hex[:8]}"
    answer, elapsed, intent, steps = await _ask(c["q"], sid)
    j = _judge_single(c, answer, steps)
    if c.get("facets"):
        j["facet_hit"] = _run_facets(c, steps)
    return {
        "id": c["id"],
        "question": c["q"],
        "elapsed_s": round(elapsed, 1),
        "intent": intent,
        **j,
        "answer": answer,
    }


async def _run_multi(c: dict) -> dict:
    sid = f"eval-exp-multi-{uuid.uuid4().hex[:8]}"
    turns_out: list[dict] = []
    for i, t in enumerate(c["turns"]):
        answer, elapsed, intent, steps = await _ask(t["q"], sid)
        j = _judge_single(t, answer, steps)
        turns_out.append({"turn": i + 1, "question": t["q"], "elapsed_s": round(elapsed, 1), "intent": intent, **j, "answer": answer})
    ok_all = all(t["pass"] for t in turns_out)
    return {"id": c["id"], "kind": "multi_turn", "pass": ok_all, "turns": turns_out}


def _collect_single_cases() -> list[dict]:
    out = list(QA_CASES) + list(CODEGEN_CASES) + list(TRAP_CASES) + list(HIST_CASES)
    return out


async def _run_cases(cases: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(c: dict) -> dict:
        async with sem:
            return await _run_single(c)

    return await asyncio.gather(*[guarded(c) for c in cases])


async def _run_multi_cases(cases: list[dict], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)

    async def guarded(c: dict) -> dict:
        async with sem:
            return await _run_multi(c)

    return await asyncio.gather(*[guarded(c) for c in cases])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="single | multi | codegen | traps | hist")
    ap.add_argument("--ids", default="", help="case ids (comma)")
    ap.add_argument("--smoke", action="store_true", help="1 per section quick sanity")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--report", default=str(ROOT / "data" / "eval_expanded.json"))
    args = ap.parse_args()

    from app.agent import session_store
    from app.core.config import get_settings
    from app.db.mysql import init_db

    # Try init_db (real MySQL); fall back to memory sessions if unavailable.
    try:
        init_db()
        print("[setup] MySQL init_db ok", flush=True)
    except Exception as e:
        session_store._USE_MEMORY = True  # noqa: SLF001
        print(f"[setup] MySQL unavailable ({type(e).__name__}), fallback memory session", flush=True)

    from app.llm.factory import get_embedding_provider, get_llm_provider

    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    s = get_settings()
    print(
        f"[setup] llm={s.llm_provider}/{s.llm_model} explore_backend={s.explore_backend} "
        f"golden_qa={s.golden_qa_enabled} milvus={s.search_use_milvus}"
        f" memory_session={getattr(session_store, '_USE_MEMORY', False)}",
        flush=True,
    )

    only = args.only
    ids = {x.strip() for x in args.ids.split(",") if x.strip()}

    def pick(cases: list[dict]) -> list[dict]:
        if args.smoke:
            return cases[:1]
        if ids:
            return [c for c in cases if c["id"] in ids]
        return cases

    results: dict[str, list] = {}
    total_ok = total = 0
    concurrency = max(1, args.concurrency)

    single_map = {
        "": _collect_single_cases(),
        "single": _collect_single_cases(),
        "codegen": CODEGEN_CASES,
        "traps": TRAP_CASES,
        "hist": HIST_CASES,
    }
    if only in single_map:
        cases = pick(single_map[only])
        if cases:
            print(f"\n== single-turn ({len(cases)}) ==", flush=True)
            results["single"] = []
            ran = asyncio.run(_run_cases(cases, concurrency))
            for i, r in enumerate(ran, 1):
                results["single"].append(r)
                total_ok += int(r["pass"]); total += 1
                print(f"[{i}/{len(ran)}] {r['id']} {'PASS' if r['pass'] else 'FAIL'} ({r['elapsed_s']}s) intent={r['intent']} issues={r['issues']} hits={r['hits']} banned={r['banned']}", flush=True)

    if only in {"", "multi"}:
        cases = pick(MULTI_CASES)
        if cases:
            print(f"\n== multi-turn ({len(cases)}) ==", flush=True)
            results["multi"] = []
            ran = asyncio.run(_run_multi_cases(cases, concurrency))
            for i, r in enumerate(ran, 1):
                results["multi"].append(r)
                total_ok += int(r["pass"]); total += 1
                for t in r["turns"]:
                    print(f"  轮{t['turn']} {'PASS' if t['pass'] else 'FAIL'} ({t['elapsed_s']}s) issues={t['issues']} banned={t['banned']}", flush=True)
                print(f"  multi {'PASS' if r['pass'] else 'FAIL'}", flush=True)

    print(f"\n== 汇总: {total_ok}/{total} ==", flush=True)
    report = {"project_id": PROJECT_ID, "mode": "source-memory", "total_ok": total_ok, "total": total, "results": results}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"report -> {args.report}", flush=True)
    return 0 if total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())