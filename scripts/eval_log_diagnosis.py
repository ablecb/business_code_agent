#!/usr/bin/env python
"""日志驱动的问题定位能力评测 — 把构造的测试日志喂给 BCA 对话接口，验证 AI 能否定位代码问题。

场景集：data/log_diag_battery/scenarios.json（每条含真实代码锚点 + 人工核验的 ground truth）。
对每个场景：
  1) 把日志片段按 question_template 拼成提问，POST /api/v1/chat（project_id=demo-loan）
  2) 保存原始回答（含 intent / retrieved / elapsed_ms）
  3) 关键词打分：回答命中 grade_keys 的比例
输出：
  data/log_diag_battery/results.json       原始结果
  data/log_diag_battery/results.md         评测报告（按场景打分 + 汇总）

用法：
  python scripts/eval_log_diagnosis.py              # 全量
  python scripts/eval_log_diagnosis.py --only S01   # 只跑指定场景（可多次 --only）
  python scripts/eval_log_diagnosis.py --host http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ROOT / "data" / "log_diag_battery" / "scenarios.json"
OUT_JSON = ROOT / "data" / "log_diag_battery" / "results.json"
OUT_MD = ROOT / "data" / "log_diag_battery" / "results.md"

REJECT_SIGNALS = ["证据不足", "未检索到", "未找到", "无法定位", "没有找到", "未能解析", "不会编造", "没有检索到"]


def load_scenarios() -> dict:
    data = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    return data


def build_message(meta: dict, sc: dict) -> str:
    # 真实用户输入 = 用户原始表述（症状/疑问）+ 附上的生产日志
    uq = str(sc.get("user_question") or "").strip()
    body = meta["question_template"].replace("{log}", sc["log"])
    return f"{uq}\n\n{body}" if uq else body


def call_chat(host: str, message: str, session_id: str, timeout=600) -> dict:
    payload = {
        "message": message,
        "session_id": session_id,
        "project_id": "demo-loan",
        "project_ids": ["demo-loan"],
        "client_apply": False,
    }
    req = urllib.request.Request(
        host.rstrip("/") + "/api/v1/chat",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    body["_wall_sec"] = round(time.time() - t0, 1)
    return body


def grade(sc: dict, reply: str) -> dict:
    keys = sc.get("grade_keys", [])
    hits = []
    for k in keys:
        hits.append(k in (reply or ""))
    matched = sum(hits)
    rate = matched / len(keys) if keys else 0.0
    if rate >= 1.0:
        verdict = "pass"
    elif rate >= 0.6:
        verdict = "partial"
    else:
        verdict = "fail"
    rejects = [s for s in REJECT_SIGNALS if s in (reply or "")]
    return {
        "keys": keys,
        "key_hits": [k for k, h in zip(keys, hits) if h],
        "key_missed": [k for k, h in zip(keys, hits) if not h],
        "matched": matched,
        "total": len(keys),
        "rate": round(rate, 2),
        "verdict": verdict if not rejects else "rejected",
        "reject_signals": rejects,
    }


def run_one(host: str, sc: dict, meta: dict) -> dict:
    message = build_message(meta, sc)
    sid = f"logdiag-{sc['id']}"
    try:
        resp = call_chat(host, message, sid)
        reply = resp.get("reply", "")
        return {
            "id": sc["id"],
            "title": sc["title"],
            "category": sc["category"],
            "session_id": sid,
            "elapsed_ms": resp.get("elapsed_ms"),
            "wall_sec": resp.get("_wall_sec"),
            "intent": resp.get("intent"),
            "phase": resp.get("phase"),
            "retrieved": [
                {
                    "fqcn": r.get("fqcn"),
                    "method_sig": r.get("method_sig"),
                    "score": r.get("score"),
                }
                for r in (resp.get("retrieved") or [])[:5]
            ],
            "explore_steps": (resp.get("explore_steps") or [])[:8],
            "grade": grade(sc, reply),
            "answer": reply,
            "error": None,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "id": sc["id"],
            "title": sc["title"],
            "category": sc["category"],
            "session_id": sid,
            "error": f"{type(e).__name__}: {e}",
            "grade": {"verdict": "error", "rate": 0.0, "matched": 0, "total": len(sc.get("grade_keys", []))},
            "answer": "",
        }


def render_report(results: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append("# 日志驱动的问题定位能力评测报告\n")
    lines.append(
        f"- 场景数: {len(results)}；评测时间: {time.strftime('%Y-%m-%d %H:%M:%S')}；"
        f"项目: {meta['project']}\n"
    )
    lines.append("- 评分口径: 回答原文命中 ground_truth 锚点关键词的比例（≥100% pass / ≥60% partial / <60% fail）。命中「证据不足」等拒绝信号判 rejected。\n")

    verdict_counts: dict[str, int] = {}
    for r in results:
        v = r["grade"]["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    lines.append("## 汇总\n")
    lines.append("| verdict | 数量 |")
    lines.append("|---|---|")
    for v in ["pass", "partial", "fail", "rejected", "error"]:
        n = verdict_counts.get(v, 0)
        if n:
            lines.append(f"| {v} | {n} |")
    lines.append("")

    lines.append("## 逐场景评分\n")
    lines.append("| ID | 场景 | 类别 | verdict | 命中关键词 | 缺失关键词 | 延迟(s) |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: int(x["id"][1:])):
        g = r["grade"]
        lines.append(
            f"| {r['id']} | {r['title']} | {r['category']} | **{g['verdict']}** "
            f"| {g['matched']}/{g['total']} {' '.join(g.get('key_hits') or [])} "
            f"| {' '.join(g.get('key_missed') or [])} | {r.get('wall_sec') or '-'} |"
        )
    lines.append("")

    for r in sorted(results, key=lambda x: int(x["id"][1:])):
        lines.append(f"\n### {r['id']} {r['title']}  [verdict: {r['grade']['verdict']}]\n")
        if r.get("error"):
            lines.append(f"- **调用失败**: {r['error']}\n")
            continue
        lines.append(f"- intent: {r['intent']} | elapsed: {r.get('elapsed_ms')}ms | wall: {r.get('wall_sec')}s")
        retr = "; ".join(f"{x.get('fqcn')}#{x.get('method_sig')}({x.get('score')})" for x in (r.get("retrieved") or []))
        lines.append(f"- 检索命中: {retr or '无'}")
        ans = (r.get("answer") or "").strip()
        lines.append(f"- 回答摘录:\n\n```\n{ans[:1200]}\n```\n")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="http://127.0.0.1:8000")
    ap.add_argument("--only", action="append", default=[], help="指定场景ID(可多次)")
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--merge", action="store_true",
                    help="--only 时把本次结果合并进已存在的 results.json（保留其它场景旧结果）")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    data = load_scenarios()
    meta = data["meta"]
    scenarios = data["scenarios"]
    if args.only:
        scenarios = [s for s in scenarios if s["id"] in args.only]
        if not scenarios:
            print(f"no scenarios matched --only {args.only}")
            return 1

    print(f"[eval_log_diagnosis] {len(scenarios)} scenarios -> {args.host}")
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futs = {pool.submit(run_one, args.host, s, meta): s for s in scenarios}
        for fut in as_completed(futs):
            sc = futs[fut]
            try:
                r = fut.result()
            except Exception as e:  # noqa: BLE001
                r = {"id": sc["id"], "title": sc["title"], "category": sc["category"],
                     "grade": {"verdict": "error", "rate": 0.0, "matched": 0, "total": 0},
                     "error": f"{type(e).__name__}: {e}", "answer": ""}
            results.append(r)
            v = r["grade"]["verdict"]
            print(f"  [{r['id']}] {v:>8}  wall={r.get('wall_sec')}s  matched={r['grade']['matched']}/{r['grade']['total']}  {r['title'][:40]}")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    if args.merge and OUT_JSON.exists():
        try:
            prev = json.loads(OUT_JSON.read_text(encoding="utf-8"))
            keep = [r for r in prev.get("results", []) if r["id"] not in {x["id"] for x in results}]
            results = keep + results
        except Exception:  # noqa: BLE001
            print("   [!] merge failed, writing fresh results")
    OUT_JSON.write_text(json.dumps({"meta": meta, "scenarios": [s["id"] for s in scenarios],
                                    "merged_with_previous": bool(args.merge), "results": results},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_report(results, meta), encoding="utf-8")
    print(f"\n[JSON] {OUT_JSON}\n[MD  ] {OUT_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())