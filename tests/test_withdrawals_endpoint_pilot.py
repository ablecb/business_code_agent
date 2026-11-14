"""Pilot: layer-by-layer read of withdrawals/withdrawals (no full-file AI parse).

Flow: Controller method → extract body → AI describe → next_calls →
extract callee body from disk → repeat.

Primary output:
  data/pilot_withdrawals_check.txt
  data/pilot_withdrawals_check.json

Run:
  python tests/test_withdrawals_endpoint_pilot.py
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT_TXT = ROOT / "data" / "pilot_withdrawals_check.txt"
OUT_JSON = ROOT / "data" / "pilot_withdrawals_check.json"
REL = "loan-service/src/main/java/com/demo/loan/controller/WithdrawalsController.java"
ENTRY_CLASS = "WithdrawalsController"
ENTRY_METHOD = "withdrawals"
MAX_NODES = 55


def log(msg: str) -> None:
    print(msg, flush=True)


def _write_txt(report: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PILOT CHECK: withdrawals/withdrawals")
    lines.append("=" * 72)
    lines.append(f"time: {report.get('started_at')}")
    lines.append(f"project_id: {report.get('project_id')}")
    lines.append(f"llm: {report.get('llm')}")
    lines.append(f"file: {report.get('file')}")
    lines.append(f"elapsed_sec: {report.get('elapsed_sec')}")
    lines.append(f"mode: {report.get('mode')}")
    lines.append("")

    entry = report.get("entry") or {}
    lines.append("--- ENTRY ---")
    lines.append(f"class_method: {entry.get('class_method')}")
    lines.append(f"fqcn: {entry.get('fqcn')}")
    lines.append(f"http_path: {entry.get('http_path')}")
    lines.append(f"body_chars: {entry.get('body_chars')}")
    lines.append(f"calls_sample: {entry.get('calls_sample')}")
    lines.append("")

    lines.append("--- NODE DESCRIPTIONS (AI) ---")
    for i, n in enumerate(report.get("nodes") or [], 1):
        lines.append(f"\n[{i}] {n.get('class_method')}  important={n.get('important')}")
        lines.append(f"    types: {n.get('important_types')}")
        lines.append(f"    summary: {n.get('summary')}")
        if n.get("validation_points"):
            lines.append(f"    validation_points: {n.get('validation_points')}")
        if n.get("status_transitions"):
            lines.append(f"    status_transitions: {n.get('status_transitions')}")
        if n.get("external_calls"):
            lines.append(f"    external_calls: {n.get('external_calls')}")
        if n.get("business_tables"):
            lines.append(f"    business_tables: {n.get('business_tables')}")
        if n.get("next_calls"):
            lines.append(f"    next_calls: {n.get('next_calls')}")

    lines.append("\n--- FLOW NARRATIVE ---")
    flow = report.get("flow") or {}
    lines.append(f"title: {flow.get('title')}")
    lines.append("narrative:")
    lines.append(str(flow.get("narrative") or "(empty)"))
    stages = flow.get("stages") or []
    if stages:
        lines.append("\nstages:")
        for st in stages:
            lines.append(
                f"  {st.get('step')}. [{st.get('class_method')}] {st.get('title')} — {st.get('desc')}"
            )

    lines.append("\n--- ENTITIES TOUCHED ---")
    ents = report.get("entities") or []
    if not ents:
        lines.append("(none)")
    for e in ents:
        lines.append(f"* {e.get('fqcn')}: {e.get('summary')}")
        if e.get("status_fields"):
            lines.append(f"  status_fields: {e.get('status_fields')}")
        if e.get("table_name"):
            lines.append(f"  table_name: {e.get('table_name')}")

    lines.append("\n--- MISSING CALLEES ---")
    missing = report.get("missing_callees") or []
    if not missing:
        lines.append("(none)")
    for m in missing[:40]:
        lines.append(f"- {m}")

    lines.append("\n--- QUALITY NOTES ---")
    for note in report.get("quality_notes") or []:
        lines.append(f"- {note}")

    if report.get("errors"):
        lines.append("\n--- ERRORS (non-fatal) ---")
        for err in report["errors"][:30]:
            lines.append(f"! {err}")

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _quality_notes(report: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    nodes = report.get("nodes") or []
    flow = report.get("flow") or {}
    narrative = str(flow.get("narrative") or "")
    if not nodes:
        notes.append("FAIL: no AI node descriptions")
        return notes
    notes.append(f"OK: analyzed {len(nodes)} nodes")
    notes.append(
        "OK: has validation_points"
        if any(n.get("validation_points") for n in nodes)
        else "WARN: no validation_points"
    )
    notes.append(
        "OK: has status_transitions"
        if any(n.get("status_transitions") for n in nodes)
        else "WARN: no status_transitions"
    )
    notes.append(
        "OK: has external_calls"
        if any(n.get("external_calls") for n in nodes)
        else "WARN: no external_calls"
    )
    notes.append(
        f"OK: flow narrative length={len(narrative)}"
        if len(narrative) > 80
        else "WARN: flow narrative too short / empty"
    )
    names = " ".join(str(n.get("class_method") or "") for n in nodes)
    notes.append(
        "OK: descended into same-class / helper methods"
        if "checkApplyReqParam" in names
        or any("Controller#" in str(n.get("class_method") or "") for n in nodes[1:6])
        else "WARN: no obvious helper like checkApplyReqParam"
    )
    if any("ServiceImpl" in str(n.get("class_method") or "") for n in nodes):
        notes.append("OK: reached *ServiceImpl")
    else:
        notes.append("WARN: did not reach *ServiceImpl")
    if any(
        str(n.get("class_method") or "").endswith("#loan")
        or "#loan " in str(n.get("class_method") or "")
        or n.get("method") == "loan"
        for n in nodes
    ):
        notes.append("OK: reached WithdrawalsService#loan")
    else:
        notes.append("WARN: missed Service#loan (quota/risk path)")
    if any(
        any(k in str(n.get("class_method") or "").lower() for k in ("risk", "quota", "limit", "approval"))
        for n in nodes
    ):
        notes.append("OK: touched risk/quota/approval-related nodes")
    else:
        notes.append("WARN: no risk/quota/approval nodes yet")
    # Prefer business depth over util noise
    utilish = sum(
        1
        for n in nodes
        if any(
            x in str(n.get("class_method") or "")
            for x in ("StringUtil", "JSONTool", "ResponseState", "ParamSdk")
        )
    )
    if utilish:
        notes.append(f"WARN: {utilish} util-like nodes still in report")
    else:
        notes.append("OK: no util-like nodes in report")
    if len(nodes) < 4:
        notes.append("WARN: fewer than 4 nodes — recursion may still be broken")
    return notes


async def run_pilot() -> dict[str, Any]:
    from datetime import datetime

    from app.analysis.common import is_noise_callee, list_java_index, resolve_call_targets
    from app.analysis.discover import describe_entity
    from app.analysis.recursive_flow import FLOW_PROMPT, _analyze_node, _llm_json
    from app.analysis.source_reader import load_method, merge_method, resolve_type_file
    from app.core.config import get_settings
    from app.parser.models import ParsedMethod, ParsedType

    get_settings.cache_clear()
    settings = get_settings()
    root = settings.project_root_path
    target = root / REL
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()

    report: dict[str, Any] = {
        "started_at": started,
        "project_id": settings.project_id,
        "llm": f"{settings.llm_provider}/{settings.llm_model}",
        "file": REL,
        "http_path": "withdrawals/withdrawals",
        "mode": "source_reader layer-by-layer (no full-file AI structural parse)",
        "entry": {},
        "nodes": [],
        "flow": {},
        "entities": [],
        "missing_callees": [],
        "errors": [],
        "quality_notes": [],
    }

    log(f"[init] llm={report['llm']} root={root}")
    log(f"[init] report → {OUT_TXT}")
    if not target.exists():
        raise FileNotFoundError(target)

    try:
        from app.db.mysql import get_session, init_db
        from app.parser.symbol_store import upsert_project

        init_db()
        with get_session() as session:
            upsert_project(session, settings.project_id, settings.project_id, str(root))
        report["mysql"] = "ok"
    except Exception as e:
        report["mysql"] = f"skipped: {e}"
        log(f"[init] mysql skipped: {e}")

    index = list_java_index(root)
    type_cache: dict[str, ParsedType] = {}

    def ensure_method(simple: str, method_name: str) -> tuple[ParsedType, ParsedMethod] | None:
        """Resolve class file + extract method body by name (disk only)."""
        if not method_name:
            return None
        key = (simple or "").split(".")[-1].lower()
        cached = type_cache.get(key)
        if cached:
            for m in cached.methods:
                if m.name == method_name and (m.raw_code or "").strip():
                    return cached, m
            if cached.file_path:
                loaded = load_method(root, cached.file_path, method_name, type_cache=type_cache)
                if loaded:
                    return loaded

        hit = resolve_type_file(index, simple, prefer_impl=True)
        if not hit:
            return None
        loaded = load_method(root, hit["path"], method_name, type_cache=type_cache)
        if loaded:
            t, m = loaded
            type_cache[t.simple_name.lower()] = t
            type_cache[key] = t
            return t, m
        # Service → ServiceImpl already preferred; try explicit Impl once more
        if not str(simple).endswith("Impl"):
            return ensure_method(f"{simple}Impl", method_name)
        return None

    log(f"[entry] extract {ENTRY_CLASS}#{ENTRY_METHOD} from disk …")
    found = ensure_method(ENTRY_CLASS, ENTRY_METHOD)
    if not found:
        # absolute path fallback
        loaded = load_method(root, REL, ENTRY_METHOD, type_cache=type_cache)
        found = loaded
    if not found:
        report["errors"].append("entry method not found via source_reader")
        report["elapsed_sec"] = round(time.time() - t0, 1)
        report["quality_notes"] = _quality_notes(report)
        _write_txt(report)
        raise RuntimeError("entry method not found — see " + str(OUT_TXT))

    entry_t, entry_m = found
    entry_t.symbol_kind = "Controller"
    report["entry"] = {
        "class_method": f"{entry_t.simple_name}#{entry_m.name}",
        "fqcn": entry_t.fqcn,
        "http_path": "withdrawals/withdrawals",
        "body_chars": len(entry_m.raw_code or ""),
        "calls_sample": list(entry_m.calls or [])[:20],
    }
    log(
        f"[entry] {report['entry']['class_method']} "
        f"body={report['entry']['body_chars']} chars "
        f"calls={len(entry_m.calls or [])}"
    )

    entity_done: set[str] = set()
    queue: list[tuple[ParsedType, ParsedMethod]] = [(entry_t, entry_m)]
    visited: set[str] = set()
    important_payloads: list[dict[str, Any]] = []

    while queue and len(visited) < MAX_NODES:
        t, m = queue.pop(0)
        mid = f"{t.fqcn}#{m.name}"
        if mid in visited:
            continue
        visited.add(mid)
        log(f"[node] {t.simple_name}#{m.name} ({len(visited)}/{MAX_NODES})")
        try:
            data = await _analyze_node(t, m)
        except Exception as e:
            report["errors"].append(f"analyze {t.simple_name}#{m.name}: {e}")
            continue
        if mid == f"{entry_t.fqcn}#{entry_m.name}":
            data["important"] = True
            data.setdefault("important_types", ["other_important"])

        node = {
            "class_method": data.get("class_method") or f"{t.simple_name}#{m.name}",
            "fqcn": t.fqcn,
            "method": m.name,
            "important": bool(data.get("important")),
            "important_types": data.get("important_types") or [],
            "summary": data.get("summary"),
            "validation_points": data.get("validation_points") or [],
            "status_transitions": data.get("status_transitions") or [],
            "external_calls": data.get("external_calls") or [],
            "business_tables": data.get("business_tables") or [],
            "next_calls": data.get("next_calls") or [],
        }
        report["nodes"].append(node)
        if node["important"]:
            important_payloads.append(
                {
                    "class_method": node["class_method"],
                    "summary": node["summary"],
                    "important_types": node["important_types"],
                    "validation_points": node["validation_points"],
                    "status_transitions": node["status_transitions"],
                    "external_calls": node["external_calls"],
                    "business_tables": node["business_tables"],
                }
            )

        for hint in (data.get("business_tables") or [])[:6]:
            simple = str(hint).split(".")[-1]
            if not re.search(r"(Entity|DO|PO)$", simple):
                continue
            hit = resolve_type_file(index, simple, prefer_impl=False)
            if not hit:
                continue
            from app.analysis.source_reader import load_type_shell

            et = load_type_shell(root, hit["path"])
            if et.fqcn in entity_done:
                continue
            entity_done.add(et.fqcn)
            try:
                await describe_entity(settings.project_id, et, project_root=root)
                ent_row: dict[str, Any] = {"fqcn": et.fqcn, "summary": f"entity {et.simple_name}"}
                try:
                    from sqlalchemy import select

                    from app.db.mysql import get_session
                    from app.models.orm import SymbolDescORM
                    from app.parser.symbol_store import symbol_id_for

                    sid = symbol_id_for(settings.project_id, et.fqcn, None)
                    with get_session() as session:
                        d = session.scalar(
                            select(SymbolDescORM).where(SymbolDescORM.symbol_id == sid)
                        )
                        if d:
                            ent_row["summary"] = d.summary
                            extra = d.extra_json or {}
                            ent_row["status_fields"] = extra.get("status_fields")
                            ent_row["table_name"] = extra.get("table_name")
                except Exception:
                    pass
                report["entities"].append(ent_row)
            except Exception as e:
                report["errors"].append(f"entity {simple}: {e}")

        seen_calls: set[str] = set()
        next_calls: list[str] = []
        # Prefer heuristic source order for coverage, then AI hints
        for c in list(m.calls or []) + list(data.get("next_calls") or []):
            key = str(c).split("(")[0].strip()
            if key and key not in seen_calls:
                seen_calls.add(key)
                next_calls.append(str(c))
        resolved: list[tuple[str, str]] = []
        seen_target: set[tuple[str, str]] = set()
        for call in next_calls:
            targets = resolve_call_targets(caller=t, call=str(call), index=index)
            if not targets:
                c = str(call).split("(")[0].strip()
                if "." not in c and c.isidentifier():
                    targets = [(t.simple_name, c)]
            for simple, method_name in targets:
                if is_noise_callee(simple, method_name):
                    continue
                tk = (simple, method_name)
                if tk in seen_target:
                    continue
                seen_target.add(tk)
                resolved.append(tk)
            if len(resolved) >= 28:
                break
        for simple, method_name in resolved:
            pair = ensure_method(simple, method_name)
            if not pair:
                report["missing_callees"].append(
                    f"{simple}#{method_name} from {t.simple_name}#{m.name}"
                )
                continue
            ct2, cmth = pair
            if ct2.simple_name == t.simple_name:
                merge_method(t, cmth)
            cid = f"{ct2.fqcn}#{cmth.name}"
            if cid not in visited:
                queue.append((ct2, cmth))

    try:
        narrative = await _llm_json(
            FLOW_PROMPT,
            f"flow_key=http_entry\nflow_role=main\n"
            f"entry={entry_t.fqcn}#{entry_m.name}\n"
            f"http_path=withdrawals/withdrawals\n"
            f"important_nodes={json.dumps(important_payloads[:40], ensure_ascii=False)}",
            max_tokens=2000,
        )
        report["flow"] = narrative
    except Exception as e:
        report["errors"].append(f"flow narrative: {e}")
        report["flow"] = {
            "title": "withdrawals/withdrawals",
            "narrative": "\n".join(
                f"{i}. {n.get('summary')}" for i, n in enumerate(important_payloads[:20], 1)
            ),
            "stages": [
                {
                    "step": i,
                    "class_method": n.get("class_method"),
                    "title": n.get("class_method"),
                    "desc": n.get("summary"),
                }
                for i, n in enumerate(important_payloads[:20], 1)
            ],
        }

    report["elapsed_sec"] = round(time.time() - t0, 1)
    report["quality_notes"] = _quality_notes(report)
    _write_txt(report)
    log(f"[done] wrote {OUT_TXT} nodes={len(report['nodes'])} missing={len(report['missing_callees'])}")
    for note in report["quality_notes"]:
        log(f"[quality] {note}")
    return report


def main() -> None:
    try:
        report = asyncio.run(run_pilot())
    except Exception:
        log(traceback.format_exc())
        raise
    if not (report.get("nodes") or []):
        raise SystemExit(2)
    log(f"\n请打开检查: {OUT_TXT}")


if __name__ == "__main__":
    main()
