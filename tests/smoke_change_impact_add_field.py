"""Smoke: structured add-field change plan for 放款 businessLabel."""
from __future__ import annotations

import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "loan_add_field_change_impact_smoke.txt"


def main() -> int:
    from app.agent.change_impact import run_add_field_plan
    from app.core.config import get_settings

    settings = get_settings()
    project_id = getattr(settings, "default_project_id", None) or getattr(
        settings, "project_id", None
    ) or "demo-loan"
    q = "我想放款接口增加一个字段 businessLabel 并落库，都要修改哪些"
    t0 = time.time()
    result = run_add_field_plan(
        project_id,
        q,
        entry_cm="WithdrawalsController#withdrawals",
    )
    elapsed = time.time() - t0
    answer = result.get("answer") or ""
    tp = result.get("touchpoints")
    roles = list((getattr(tp, "roles", None) or {}).keys())
    steps = result.get("steps") or []

    calls_reads = [
        st
        for st in steps
        if st.get("tool") == "get_symbol_code"
        and (st.get("args") or {}).get("via") == "calls_closure"
        and st.get("ok")
    ]
    # Bind/assemble must be grounded on CALLS (or var-resolve under closure),
    # not solely on early prefix search_symbols popularity.
    bind = (getattr(tp, "roles", None) or {}).get("bind")
    assemble = (getattr(tp, "roles", None) or {}).get("assemble")
    bind_cm = getattr(bind, "class_method", "") if bind else ""
    assemble_cm = getattr(assemble, "class_method", "") if assemble else ""
    calls_cms = {
        (st.get("args") or {}).get("class_method")
        for st in calls_reads
        if (st.get("args") or {}).get("class_method")
    }
    vr_reads = {
        (st.get("args") or {}).get("class_method")
        for st in steps
        if st.get("tool") == "get_symbol_code"
        and (st.get("args") or {}).get("via") == "var_resolve"
        and st.get("ok")
    }
    grounded = set(calls_cms) | set(vr_reads)

    checks = {
        "has_结论": "## 结论" in answer,
        "has_改动顺序": "## 改动顺序" in answer,
        "no_legacy_改动清单": "## 改动清单" not in answer,
        "has_从到": "## 从 → 到" in answer or "## 从→到" in answer,
        "no_legacy_建议改动": "## 建议改动" not in answer,
        "has_DDL": "## DDL" in answer or "ALTER TABLE" in answer,
        "no_gate_error": "探索步数用尽" not in answer and "get_symbol_code" not in answer,
        "has_roles_or_gap": bool(roles) or "未定位" in answer or "缺口" in answer,
        "fast_enough": elapsed < 120,
        "ok_or_partial": bool(result.get("ok")) or "## 改动顺序" in answer,
        "used_calls_closure": len(calls_reads) >= 1,
        "has_evidence_line": "证据:" in answer or "证据：" in answer,
        "has_file_line": "文件:" in answer or "文件：" in answer,
        "has_stats_footer": "非写死模板" in answer or "CALLS 闭包" in answer,
    }
    # When withdrawals bind uses FieldNameConstants, expect constant role.
    if "constant" in roles or "FieldNameConstants" in answer:
        checks["has_constant_role"] = "constant" in roles or "FieldNameConstants" in answer
        checks["has_proposed_const"] = (
            "BUSINESS_LABEL" in answer or "businessLabel" in answer
        )
        checks["no_hardcoded_common_constant"] = "CommonConstant.BUSINESS_LABEL" not in answer
    if bind_cm:
        checks["bind_from_calls_or_var_resolve"] = bind_cm in grounded
    if assemble_cm:
        checks["assemble_from_calls_or_var_resolve"] = assemble_cm in grounded
    # Prefer real ext table when discovery worked
    if result.get("ok"):
        checks["ext_table_or_entity"] = (
            "demo_b_withdrawals_ext" in answer
            or "WithdrawalsExt" in answer
            or "persist_entity" in roles
            or "@TableName" in answer
        )

    lines = [
        f"Q: {q}",
        f"project_id={project_id}",
        f"elapsed_s={elapsed:.2f}",
        f"roles={roles}",
        f"bind={bind_cm}",
        f"assemble={assemble_cm}",
        f"calls_reads={len(calls_reads)}",
        f"ok={result.get('ok')}",
        f"checks={checks}",
        "",
        "----- ANSWER -----",
        answer,
        "",
        "ALL_PASS" if all(checks.values()) else "FAIL",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"wrote {OUT}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
