#!/usr/bin/env python
"""本管线测试电池：各种不同类型问题 → 走 run_chat 实测。

与 scripts/eval_qa.py（5 题、走 HTTP）不同，本脚本：
  * 进程内直调 run_chat（无需起服务）
  * 本管线：golden → explore/PEV → 通用兜底
  * 覆盖 流程/接口入口/表关联/表/字段方案/源码/下游/通用代码/通用知识/闲聊/意图边界
  * 确定性关键词判分（expect-any / forbid-any）+ 逐题回答摘要

用法:
    python scripts/test_native_battery.py                 # 全量（真模型，约 5-10 分钟）
    python scripts/test_native_battery.py --subset flow   # 只跑某类
    python scripts/test_native_battery.py --json out.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (类别, 问题, expect_any, forbid_any)
BATTERY: list[tuple[str, str, list[str], list[str]]] = [
    # —— 流程类 ——
    ("flow", "放款流程什么样", ["放款", "流程", "Withdrawals"], ["未在项目中找到", "Mock", "{"]) ,
    ("flow", "还款流程怎么走的", ["还款", "流程"], ["未在项目中找到", "Mock", "{"]),
    # —— 接口 / 入口 ——
    ("entry", "提现接口的地址是什么", ["url", "path", "地址", "http", "/"], ["未在项目中找到", "{"]),
    ("entry", "放款接口入口在哪个 Controller", ["WithdrawalsController"], ["未在项目中找到", "{"]),
    # —— 表关联 / 表 ——
    ("table", "放款表和放款优惠券表怎么关联的", ["关联", "order", "coupon"], ["未在项目中找到", '{"tool"', "Mock"]),
    ("table", "放款流程涉及哪几张表", ["表", "demo_", "实体", "TableName"], ["未在项目中找到", "{"]),
    # —— 字段 / 改码方案 ——
    ("field", "放款接口增加一个字段 businessLabel 并落库，需要改哪些地方", ["字段", "VO", "实体", "落库"], ["Mock", "{"]),
    ("field", "放款接口的入参校验有哪些", ["校验", "参数", "非空"], ["未在项目中找到", "Mock", "{"]),
    # —— 源码 ——
    # 注意：源码回答天然含 { }，forbid 不能放 '{'
    ("source", "WithdrawalsController#withdrawals 的源码是什么", ["WithdrawalsController", "代码", "```"], ["未在项目中找到", "Mock"]),
    # —— 下游 / 链路 ——
    ("downstream", "放款接口调用了哪些下游接口", ["调用", "Service", "Feign"], ["未在项目中找到", "{"]),
    # —— 通用代码生成（非项目代码） ——
    ("codegen", "用 Java 写一个快速排序", ["排序", "quick", "sort", "Java"], ["Mock"]),
    # —— 通用知识 ——
    ("general_knowledge", "什么是数据库事务", ["事务", "ACID", "提交", "回滚"], ["未在项目中找到"]),
    # —— 闲聊 ——
    ("chat", "你好", ["你好", "帮", "可以"], []),
    # —— 意图边界（改码 vs 问答 vs 通用） ——
    ("intent_boundary", "帮我把放款接口的代码优化一下", ["优化", "方案", "代码"], ["Mock"]),
]


async def _run_one(question: str, timeout: float = 150.0) -> dict:
    from app.agent.graph_agent import run_chat

    t0 = time.perf_counter()

    async def _inner() -> dict:
        resp = await run_chat(
            question,
            project_id="demo-loan",
        )
        return {
            "reply": resp.reply or "",
            "intent": resp.intent,
            "elapsed_ms": resp.elapsed_ms or int((time.perf_counter() - t0) * 1000),
            "options": len(resp.options or []),
        }

    return await asyncio.wait_for(_inner(), timeout=timeout)


def _judge(question: str, answer: str, expect: list[str], forbid: list[str]) -> dict:
    ans = (answer or "").lower()
    hits = [k for k in expect if k.lower() in ans]
    bad = [k for k in forbid if k.lower() in ans]
    ok = bool(hits) and not bad
    return {"ok": ok, "hits": hits, "bad": bad}


async def _run_battery(subset: str | None, json_path: str | None) -> int:
    questions = [(c, q, e, f) for c, q, e, f in BATTERY if not subset or c == subset]
    print(f"题目数={len(questions)}  LLM=真实(硅基流动 Qwen3-32B)\n")
    results = []
    for i, (cat, q, expect, forbid) in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] ({cat}) {q}", flush=True)
        t0 = time.time()
        try:
            r = await _run_one(q)
        except asyncio.TimeoutError:
            print(f"  !! 超时(>150s)，跳过", flush=True)
            results.append(
                {
                    "category": cat,
                    "question": q,
                    "error": "timeout>150s",
                    "ok": False,
                }
            )
            continue
        except Exception as e:
            print(f"  !! 异常: {type(e).__name__}: {e}", flush=True)
            results.append(
                {
                    "category": cat,
                    "question": q,
                    "error": f"{type(e).__name__}: {e}",
                    "ok": False,
                }
            )
            continue
        j = _judge(q, r["reply"], expect, forbid)
        elapsed = time.time() - t0
        snippet = (r["reply"] or "").replace("\n", " ")[:120]
        flag = "PASS" if j["ok"] else "FAIL"
        print(
            f"  [{flag}] intent={r['intent']} elapsed={elapsed:.1f}s\n      {snippet}",
            flush=True,
        )
        results.append(
            {
                "category": cat,
                "question": q,
                "intent": r["intent"],
                "elapsed_ms": r["elapsed_ms"],
                "ok": j["ok"],
                "expect_hits": j["hits"],
                "forbid_hits": j["bad"],
                "reply_len": len(r["reply"] or ""),
                "reply_head": snippet,
            }
        )

    passed = sum(1 for r in results if r.get("ok"))
    print("\n==== 汇总 ====")
    print(f"PASS {passed}/{len(results)}")
    for r in results:
        mark = "PASS" if r.get("ok") else "FAIL"
        err = r.get("error") or f"intent={r.get('intent')} hit={r.get('expect_hits')} bad={r.get('forbid_hits')}"
        print(f"  [{mark}] ({r.get('category')}) {r.get('question')}  ->  {err}")

    if json_path:
        Path(json_path).write_text(
            json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nJSON 已写: {json_path}")
    return 0 if passed == len(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="本管线测试电池")
    parser.add_argument("--subset", default=None, help="只跑某类: flow|entry|table|field|source|downstream|codegen|general_knowledge|chat|intent_boundary")
    parser.add_argument("--json", default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()
    return asyncio.run(_run_battery(args.subset, args.json))


if __name__ == "__main__":
    raise SystemExit(main())
