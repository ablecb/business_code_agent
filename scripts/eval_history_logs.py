#!/usr/bin/env python
"""Parse desktop-chat history logs into a structured eval baseline.

Reads every *log under data/desktop_chat, extracts per-turn fields
(question / intent / phase / answer / elapsed_ms / golden hit / options),
then aggregates metrics: resolution proxy, refusal / low-evidence rate,
code-block delivery rate, golden-QA hit rate, and latency percentiles.

This is an OFFLINE baseline (no LLM calls) — run it before and after agent
changes to see whether the current system actually improved.

Usage:
    python scripts/eval_history_logs.py                      # all logs, console report
    python scripts/eval_history_logs.py --report data/eval_history_baseline.json
    python scripts/eval_history_logs.py --only desktop-chat-*.log
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "desktop_chat"

# ---------------------------------------------------------------------------
# refusal / low-evidence signals — the log lines that mark a failed turn
# ---------------------------------------------------------------------------
REJECT_SIGNALS: list[str] = [
    "不会编造",
    "证据不足",
    "未在已读材料",
    "未能解析出明确文件清单",
    "未检索到",
    "未获取到",
    "请确认知识库文档",
    "已拒绝改代码",
    "未能读取",
    "契约缺口",
    "无法定位",
    "未找到",
    "没有找到",
    "没有检索到",
]

# ---------------------------------------------------------------------------
# heuristic question type buckets (used for the by-type report only)
# ---------------------------------------------------------------------------
TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("codegen/改码", ["给我一段代码", "给我代码", "代码：", "帮我写", "生成", "加一个字段", "新增接口", "写一个", "改一下", "修改", "实现", "新增"]),
    ("流程/链路", ["流程", "调用链", "链路", "入口", "步骤", "怎么", "如何", "什么样", "过程", "顺序"]),
    ("表/字段", ["表", "字段", "落库", "列", "关联", "表结构"]),
    ("源码/定位", ["源码", "代码是什么", "在哪", "位置", "方法", "类", "接口有哪些", "有没有", "干嘛", "做什么"]),
    ("通用", []),
]


def bucket_type(question: str) -> str:
    for name, pats in TYPE_PATTERNS:
        if any(p in question for p in pats):
            return name
    return "通用"


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
TURN_RE = re.compile(r"\[[^\]]+\]\s*TURN (\d+)\s+USER QUESTION")
META_LINE_RE = re.compile(r"^intent=(\S+)\s+phase=(\S+)\s+selected_option_id=(\S+)\s+elapsed_ms=(\S+)(?:\s+assistant_message_id=(\d+))?")
DONE_RE = re.compile(r"^\[turn \d+ done\]")
META_OPEN_RE = re.compile(r"^\s*meta=")


def _balanced_json_lines(lines: list[str]) -> tuple[int, str]:
    """Collect lines until the outermost {} brace pair is balanced.
    Returns (end_index_exclusive, joined_text)."""
    depth = 0
    started = False
    for idx, ln in enumerate(lines):
        s = ln.strip()
        if not s:
            continue
        depth += s.count("{") - s.count("}")
        if s.count("{") > 0:
            started = True
        if started and depth <= 0:
            return idx + 1, "\n".join(lines[: idx + 1])
    return len(lines), "\n".join(lines)


def parse_file(path: Path) -> list[dict]:
    """Return list of turn dicts parsed from one log file."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        return [{"file": path.name, "parse_error": str(e)}]
    # skip binary / heavily mojibake files
    try:
        text = raw.decode("utf-8")
        if text.count("\ufffd") > max(5, len(text) // 500):
            return [{"file": path.name, "parse_error": "binary/mojibake"}]
    except UnicodeDecodeError:
        return [{"file": path.name, "parse_error": "not utf-8"}]
    lines = text.splitlines()

    turns: list[dict] = []
    cur: dict | None = None
    section: str | None = None  # "question" | "intent" | "answer" | "options"
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        m = TURN_RE.search(line)
        if m:
            if cur is not None:
                turns.append(cur)
            cur = {
                "file": path.name,
                "turn": int(m.group(1)),
                "question": "",
                "intent": None,
                "phase": None,
                "elapsed_ms": None,
                "answer": "",
                "meta": None,
                "options": [],
                "golden_hit": False,
            }
            section = "question"
            i += 1
            # collect question lines until session_id= / section divider / next turn
            while i < n:
                ln = lines[i]
                if re.match(r"^=+$", ln.strip()):
                    # the "====" banner sits between the TURN header and the question
                    i += 1
                    continue
                if (
                    ln.startswith("session_id=")
                    or re.match(r"^-{4,}", ln)
                    or TURN_RE.search(ln)
                    or DONE_RE.match(ln)
                ):
                    break
                if ln.strip():
                    cur["question"] = (cur["question"] + "\n" + ln).strip()
                i += 1
            continue

        if cur is None:
            i += 1
            continue

        if DONE_RE.match(line):
            section = None
            i += 1
            continue

        if "INTENT (recognized)" in line:
            section = "intent"
            i += 1
            continue

        if "AI REPLY" in line:
            section = "answer"
            i += 1
            continue

        if "AI STEPS / PROGRESS" in line:
            section = None
            i += 1
            continue

        if re.match(r"^-{4,}", line):
            # section divider: new subsection header follows (with timestamp)
            nxt = lines[i + 1] if i + 1 < n else ""
            if "OPTIONS" in nxt:
                section = "options"
            elif "SESSION CONTEXT" in nxt or "CONVERSATION HISTORY" in nxt or "ANSWER REVIEW" in nxt or "RESEARCH PLAN" in nxt or "SCOPE" in nxt or "EXPLORE STEPS" in nxt or "INTENT" in nxt:
                section = None
            # keep current section otherwise (e.g. answer text with dashes)
            i += 1
            continue

        if "命中黄金问答库" in line or "golden_hit" in stripped.lower():
            cur["golden_hit"] = True

        # single-line metadata:  intent=...  phase=...  selected_option_id=...  elapsed_ms=...
        m2 = META_LINE_RE.match(line)
        if m2:
            cur["intent"], cur["phase"] = m2.group(1), m2.group(2)
            cur["selected_option_id"] = m2.group(3)
            em = m2.group(4)
            cur["elapsed_ms"] = int(em) if em not in ("None", "") else None
            cur["assistant_message_id"] = m2.group(5)
            section = None
            i += 1
            continue

        # meta= JSON block (balance braces)
        if META_OPEN_RE.match(line):
            rest = lines[i + 1 :]
            used, blob = _balanced_json_lines(rest)
            try:
                parsed = json.loads(blob)
                cur["meta"] = parsed
                # meta often carries the authoritative intent/phase/elapsed
                if cur.get("intent") is None:
                    cur["intent"] = parsed.get("intent")
                if cur.get("phase") is None:
                    cur["phase"] = parsed.get("phase")
                if cur.get("elapsed_ms") is None:
                    em = parsed.get("elapsed_ms")
                    if isinstance(em, int) and em > 0:
                        cur["elapsed_ms"] = em
            except Exception:
                cur["meta_raw"] = blob[:2000]
            i += 1 + used
            section = None
            continue

        if section == "intent":
            if line.startswith("intent="):
                mm = re.match(r"^intent=(\S+)", line)
                if mm:
                    cur["intent"] = mm.group(1)
            i += 1
            continue

        if section == "answer":
            # stop at next section header / metadata line / done marker
            nxt_sec = re.match(r"^\[.*?\]\s+(AI REPLY|SESSION CONTEXT|CONVERSATION HISTORY|ANSWER REVIEW|RESEARCH PLAN|SCOPE|EXPLORE STEPS|INTENT|OPTIONS)", line)
            if nxt_sec or META_OPEN_RE.match(line) or re.match(r"^intent=.*phase=", line) or DONE_RE.match(line):
                section = None
                i += 1
                continue
            if stripped:
                cur["answer"] = (cur["answer"] + "\n" + line).strip()
            i += 1
            continue

        if section == "options":
            if stripped.startswith("["):
                # collect the full JSON array (multiline)
                buf = [stripped]
                j = i + 1
                while j < n and not stripped.endswith("]"):
                    stripped = lines[j].strip()
                    buf.append(stripped)
                    j += 1
                try:
                    cur["options"] = json.loads("\n".join(buf))
                except Exception:
                    pass
                section = None
                i = j
                continue
            i += 1
            continue

        i += 1

    if cur is not None:
        turns.append(cur)
    return turns


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def signal_hits(answer: str) -> list[str]:
    return [s for s in REJECT_SIGNALS if s in (answer or "")]


def eval_turns(turns: list[dict]) -> dict:
    """Compute per-turn verdicts + aggregate stats."""
    # dedupe: same file + same question recorded twice (log append artefacts)
    seen: set[tuple[str, str]] = set()
    uniq: list[dict] = []
    for t in turns:
        q = (t.get("question") or "").strip()
        # skip mojibake turns whose question is only '?' / whitespace
        if q and all(ch in "? \t" for ch in q):
            continue
        key = (t.get("file") or "", q)
        if not key[1] or key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    turns = uniq

    verdicts: list[dict] = []
    for t in turns:
        ans = t.get("answer") or ""
        sigs = signal_hits(ans)
        phase = t.get("phase")
        intent = t.get("intent")
        has_code = "```" in ans
        meta = t.get("meta") or {}
        golden = bool(t.get("golden_hit") or "黄金问答库" in ans or meta.get("golden_hit"))

        # system-level failure (ImportError etc.) — not an evidence problem
        is_sys_fail = phase == "failed" or "处理失败" in ans or "ImportError" in ans
        # client_agent = action/ops turn (git, table count...): short reply is OK
        min_len = 5 if intent == "client_agent" else 40
        if is_sys_fail:
            verdict = "system_failed"
        elif sigs:
            verdict = "rejected"
        elif phase == "error":
            verdict = "error"
        elif phase == "awaiting_option":
            verdict = "awaiting_option"
        elif len(ans) < min_len:
            verdict = "empty"
        else:
            verdict = "answered"

        verdicts.append(
            {
                "file": t.get("file"),
                "turn": t.get("turn"),
                "question": (t.get("question") or "")[:200],
                "intent": intent,
                "phase": phase,
                "elapsed_ms": t.get("elapsed_ms"),
                "answer_len": len(ans),
                "has_code": has_code,
                "golden_hit": golden,
                "verdict": verdict,
                "signals": sigs,
                "qtype": bucket_type(t.get("question") or ""),
                "options": t.get("options") or [],
            }
        )

    total = len(verdicts)
    vc = Counter(v["verdict"] for v in verdicts)
    ic = Counter(v["intent"] for v in verdicts)
    pc = Counter(v["phase"] for v in verdicts)
    qc = Counter(v["qtype"] for v in verdicts)

    lat = [v["elapsed_ms"] for v in verdicts if isinstance(v["elapsed_ms"], int) and v["elapsed_ms"] > 0]
    lat_by_intent: dict[str, list[int]] = defaultdict(list)
    for v in verdicts:
        em = v["elapsed_ms"]
        if isinstance(em, int) and em > 0:
            lat_by_intent[v["intent"] or "?"].append(em)

    golden_v = [v for v in verdicts if v["golden_hit"]]
    golden_lat = [v["elapsed_ms"] for v in golden_v if isinstance(v["elapsed_ms"], int) and v["elapsed_ms"] > 0]

    def pct(l: list[int]) -> dict:
        if not l:
            return {"n": 0, "mean_ms": None, "p50": None, "p90": None, "max_ms": None}
        return {
            "n": len(l),
            "mean_ms": round(statistics.mean(l)),
            "p50": round(statistics.median(l)),
            "p90": round(sorted(l)[max(int(len(l) * 0.9) - 1, 0)]),
            "max_ms": max(l),
        }

    return {
        "turns": total,
        "files": len({v["file"] for v in verdicts}),
        "verdicts": verdicts,
        "verdict_counts": dict(vc),
        "intent_counts": dict(ic),
        "phase_counts": dict(pc),
        "qtype_counts": dict(qc),
        "latency_all": pct(lat),
        "latency_by_intent": {k: pct(v) for k, v in sorted(lat_by_intent.items())},
        "golden_hit": {
            "n": len(golden_v),
            "rate": round(len(golden_v) / total, 4) if total else 0,
            "latency": pct(golden_lat),
        },
        "code_delivery": {
            "n": sum(1 for v in verdicts if v["has_code"]),
            "rate": round(sum(1 for v in verdicts if v["has_code"]) / total, 4) if total else 0,
        },
        "rejected": {
            "n": vc.get("rejected", 0),
            "rate": round(vc.get("rejected", 0) / total, 4) if total else 0,
        },
        "system_failed": {
            "n": vc.get("system_failed", 0),
            "rate": round(vc.get("system_failed", 0) / total, 4) if total else 0,
        },
        "resolved_approx": {
            "n": vc.get("answered", 0) + vc.get("awaiting_option", 0),
            "rate": round((vc.get("answered", 0) + vc.get("awaiting_option", 0)) / total, 4) if total else 0,
        },
    }


def render_report(r: dict) -> str:
    lines: list[str] = []
    lines.append("# 历史问答日志基线（desktop_chat）\n")
    lines.append(f"- 文件数: {r['files']}  轮次总数: {r['turns']}\n")
    lines.append("## 判定分布")
    lines.append("| verdict | 数量 | 占比 |")
    lines.append("|---|---|---|")
    for k, v in sorted(r["verdict_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"| {k} | {v} | {v / r['turns']:.1%} |")
    lines.append("")
    lines.append("## 意图分布")
    lines.append(f"`{json.dumps(r['intent_counts'], ensure_ascii=False)}`\n")
    lines.append("## phase 分布")
    lines.append(f"`{json.dumps(r['phase_counts'], ensure_ascii=False)}`\n")
    lines.append("## 问题类型分布")
    lines.append(f"`{json.dumps(r['qtype_counts'], ensure_ascii=False)}`\n")
    lines.append("## 关键指标")
    lat = r["latency_all"]
    sf = r.get("system_failed") or {"n": 0, "rate": 0}
    lines.append(
        f"- 拒绝/证据不足率: **{r['rejected']['rate']:.1%}** ({r['rejected']['n']}/{r['turns']})\n"
        f"- 系统故障率(ImportError等): {sf['rate']:.1%} ({sf['n']}/{r['turns']})\n"
        f"- 近似解决率(answered+awaiting_option): **{r['resolved_approx']['rate']:.1%}** ({r['resolved_approx']['n']}/{r['turns']})\n"
        f"- 代码块交付率: {r['code_delivery']['rate']:.1%}\n"
        f"- 黄金问答库命中率: {r['golden_hit']['rate']:.1%} ({r['golden_hit']['n']})\n"
        f"- 延迟(全部): n={lat['n']} mean={lat['mean_ms']}ms p50={lat['p50']}ms p90={lat['p90']}ms max={lat['max_ms']}ms"
    )
    lines.append("")
    lines.append("### 按意图延迟")
    for k, v in r["latency_by_intent"].items():
        lines.append(f"- `{k}`: n={v['n']} mean={v['mean_ms']}ms p50={v['p50']}ms p90={v['p90']}ms")
    gh = r["golden_hit"]
    lines.append(f"\n### 黄金问答库命中延迟: n={gh['latency']['n']} mean={gh['latency']['mean_ms']}ms p50={gh['latency']['p50']}ms")
    lines.append("\n## 拒绝/低证据轮次明细")
    lines.append("| 文件 | turn | intent | 问题 | 信号 |")
    lines.append("|---|---|---|---|---|")
    for v in r["verdicts"]:
        if v["verdict"] in ("rejected", "error", "empty"):
            lines.append(
                f"| {v['file']} | {v['turn']} | {v['intent']} | {v['question'][:60]} | {';'.join(v['signals'])} |"
            )
    lines.append("\n## 黄金命中轮次")
    lines.append("| 文件 | turn | intent | 问题 | 延迟 |")
    lines.append("|---|---|---|---|---|")
    for v in r["verdicts"]:
        if v["golden_hit"]:
            lines.append(
                f"| {v['file']} | {v['turn']} | {v['intent']} | {v['question'][:60]} | {v['elapsed_ms']}ms |"
            )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="*.log", help="glob filter (e.g. 'desktop-chat-*.log')")
    ap.add_argument("--report", default=str(ROOT / "data" / "eval_history_baseline.json"))
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    files = sorted(DATA.glob(args.only))
    if not files:
        print(f"no logs matched {args.only} under {DATA}")
        return 1

    all_turns: list[dict] = []
    parse_errors = 0
    for f in files:
        turns = parse_file(f)
        for t in turns:
            if "parse_error" in t:
                parse_errors += 1
                continue
            all_turns.append(t)

    report = eval_turns(all_turns)
    report["_meta"] = {
        "glob": args.only,
        "log_dir": str(DATA),
        "parse_errors": parse_errors,
        "note": "offline baseline; verdicts are keyword heuristics, not ground truth",
    }

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ROOT / "data" / "eval_history_baseline.md"
    md.write_text(render_report(report), encoding="utf-8")

    print(render_report(report))
    print(f"\n[JSON] {out}")
    print(f"[MD ] {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
