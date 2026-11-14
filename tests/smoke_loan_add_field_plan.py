"""Smoke: 放款接口增加字段并落库 → change plan (not insertLoanData gate list).

Usage:
  python tests/smoke_loan_add_field_plan.py
"""
from __future__ import annotations

import asyncio
import re
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPORT = ROOT / "data" / "loan_add_field_ai_plan_smoke.txt"
PROJECT_ID = "demo-loan"
QUESTION = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"

# Must-hit anchors for a real change plan (analog path).
NEED_ANY_OF = [
    ["WithdrawalsApplyReqVo", "ApplyReqVo"],
    ["checkApplyReqParam"],
    ["reqVo2ExtDmo", "WithdrawalsExt"],
    ["demo_b_withdrawals_ext", "WithdrawalsExtMapper", "Mapper", "@TableName"],
]
# Soft: DDL cue
DDL_CUES = ["ALTER", "DDL", "加列", "新增列", "businessLabel", "market_label"]

# Must NOT be primarily an insertLoanData business-gate list.
FORBID_IF_NO_PLAN = ["insertLoanData", "insertOrUpdateLoanInfo"]

# Flow-synth leftovers must not appear in add-field answers.
FORBID_ANY = [
    "业务状态流转",
    "外部系统调用：未在已读",
    "外部系统调用:未在已读",
    "评审提示",
]

# Evidence facets that must appear in get_symbol_code / retrieved trail.
READ_FACETS = {
    "vo": re.compile(r"(?i)WithdrawalsApplyReqVo|ApplyReqVo"),
    "bind": re.compile(r"(?i)checkApplyReqParam"),
    "assemble": re.compile(r"(?i)reqVo2ExtDmo"),
    "ext": re.compile(r"(?i)WithdrawalsExt"),
}


def _judge(answer: str) -> tuple[bool, list[str]]:
    a = answer or ""
    issues: list[str] = []
    if "评审提示" in a or re.search(r"[（(]补充说明[：:]", a):
        issues.append("review footnote in answer body")
    if "改动清单" not in a and "改动顺序" not in a:
        issues.append("missing 改动顺序/改动清单 structure")
    if not re.search(r"(?i)ALTER\s+TABLE|##\s*DDL", a):
        issues.append("missing DDL section/ALTER")
    for bad in FORBID_ANY:
        if bad in a:
            issues.append(f"forbidden leftover: {bad}")
    # MyBatis-Plus: must not push ExtMapper.xml as required.
    if re.search(
        r"(?i)(?:必须|必改|需要|要改).{0,24}WithdrawalsExtMapper\.xml",
        a,
    ):
        issues.append("wrongly requires ExtMapper.xml (MyBatis-Plus)")
    for group in NEED_ANY_OF:
        if not any(x in a for x in group):
            issues.append(f"missing any of {group}")
    has_ddl = any(x in a for x in DDL_CUES)
    if not has_ddl:
        issues.append("missing DDL/field-name cue")
    # Fail if insert* dominate without change-plan anchors.
    plan_ok = not any(i.startswith("missing") for i in issues)
    if not plan_ok:
        for bad in FORBID_IF_NO_PLAN:
            if bad in a:
                issues.append(f"persist-gate narrative hit: {bad}")
    # Even with some plan hits, reject if answer is mostly insertLoanData list.
    insert_hits = len(re.findall(r"(?i)insertLoanData|insertOrUpdateLoanInfo", a))
    plan_hits = sum(1 for g in NEED_ANY_OF if any(x in a for x in g))
    if insert_hits >= 2 and plan_hits < 3:
        issues.append("primarily insertLoanData/loanApply persist list")
    # Scannable spacing between main sections.
    if not re.search(r"##\s*(?:改动清单|改动顺序)[^\n]*\n\n", a):
        issues.append("改动顺序/改动清单 heading not followed by blank line")
    return (len(issues) == 0), issues


def _collect_tool_evidence(steps: list) -> tuple[list[str], list[str], dict[str, bool]]:
    tools: list[str] = []
    reads: list[str] = []
    facet_hit = {k: False for k in READ_FACETS}
    for s in steps or []:
        if not isinstance(s, dict):
            continue
        tool = str(s.get("tool") or "")
        if tool:
            tools.append(tool)
        if tool != "get_symbol_code" or not s.get("ok"):
            continue
        args = s.get("args") or {}
        fqcn = str(s.get("fqcn") or args.get("fqcn") or "")
        method = str(s.get("method_name") or args.get("method_name") or "")
        fpath = str(s.get("file_path") or args.get("file_path") or "")
        label = f"{fqcn}#{method} path={fpath}".strip()
        if not fqcn and not method:
            label = (
                str(args.get("fqcn") or "")
                + "#"
                + str(args.get("method_name") or "")
                + " "
                + str(args.get("symbol_id") or "")[:12]
            ).strip() or "get_symbol_code"
        reads.append(label)
        blob = " ".join([fqcn, method, fpath, " ".join(str(v) for v in args.values())])
        for name, rx in READ_FACETS.items():
            if rx.search(blob):
                facet_hit[name] = True
    return tools, reads, facet_hit


def _enrich_facets_from_retrieved(
    facet_hit: dict[str, bool], retrieved: list
) -> dict[str, bool]:
    """Do not invent facets from search stubs — steps already carry read identity."""
    _ = retrieved
    return dict(facet_hit)


async def main() -> int:
    from app.agent import session_store
    from app.agent.graph_agent import run_chat
    from app.core.config import get_settings
    from app.db.mysql import init_db

    get_settings.cache_clear()
    init_db()
    sid = f"smoke-addfield-{uuid.uuid4().hex[:10]}"
    lines: list[str] = [
        f"session_id={sid}",
        f"project_id={PROJECT_ID}",
        f"question={QUESTION}",
        "",
    ]
    try:
        session_store.delete_session(sid)
    except Exception:
        pass

    result = await run_chat(
        QUESTION,
        project_id=PROJECT_ID,
        session_id=sid,
    )
    answer = (result.reply if hasattr(result, "reply") else None) or (
        result.get("reply") if isinstance(result, dict) else ""
    ) or ""
    steps = getattr(result, "explore_steps", None) or (
        result.get("explore_steps") if isinstance(result, dict) else None
    )
    if not steps:
        ctx = session_store.get_context(sid) or {}
        steps = ctx.get("explore_steps") or []
    retrieved = getattr(result, "retrieved", None) or []
    elapsed_ms = getattr(result, "elapsed_ms", None)
    if elapsed_ms is None and isinstance(result, dict):
        elapsed_ms = result.get("elapsed_ms")

    tools, reads, facet_hit = _collect_tool_evidence(steps if isinstance(steps, list) else [])
    facet_hit = _enrich_facets_from_retrieved(facet_hit, list(retrieved or []))
    retrieved_labels: list[str] = []
    for h in list(retrieved or []):
        d = h.model_dump() if hasattr(h, "model_dump") else (h if isinstance(h, dict) else {})
        if not d:
            continue
        retrieved_labels.append(
            f"{d.get('fqcn') or ''}#{(d.get('method_sig') or '')} "
            f"path={d.get('file_path') or ''}".strip()
        )

    ok, issues = _judge(answer)
    if "gate_detail_fast_path" in tools or "offline_fast_path" in tools:
        ok = False
        issues.append("took offline/gate fast path (should use tool loop)")
    code_reads = sum(1 for t in tools if t == "get_symbol_code")
    if code_reads < 4:
        ok = False
        issues.append(f"too few get_symbol_code calls: {code_reads} (need >= 4)")
    missing_facets = [k for k, v in facet_hit.items() if not v]
    if missing_facets:
        ok = False
        issues.append(f"unread required facets: {missing_facets}")

    verdict = "PASS" if ok else "FAIL"
    lines.append(f"verdict={verdict}")
    lines.append(f"elapsed_ms={elapsed_ms}")
    lines.append(f"tools={tools}")
    lines.append(f"get_symbol_code_count={code_reads}")
    lines.append("reads:")
    for r in reads[:24]:
        lines.append(f"  - {r}")
    lines.append(f"facets={facet_hit}")
    lines.append("retrieved:")
    for lab in retrieved_labels[:24]:
        lines.append(f"  - {lab}")
    if issues:
        lines.append("issues:")
        for i in issues:
            lines.append(f"  - {i}")
    lines.append("")
    lines.append("--- answer ---")
    lines.append(answer.strip() or "(empty)")
    lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"{verdict} → {REPORT}")
    print(f"elapsed_ms={elapsed_ms} tools={tools[:16]} facets={facet_hit}")
    if not ok:
        print("issues:", issues)
        print(answer[:1200])
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
