"""Skill registry: name → title / PEV tools. Execution stays in pev_agent.

layer:
  ask  — read-only understanding (default Q&A)
  plan — read-only change analysis / options (top-level intent=plan)
"""
from __future__ import annotations

from typing import Any

from app.agent.runtime.types import SKILL_NAMES

SKILL_SPECS: dict[str, dict[str, Any]] = {
    "explain_flow": {
        "title": "主链流程",
        "layer": "ask",
        "tools": ["resolve_entry", "expand_calls", "mysql_enrich", "summarize_flow"],
    },
    "field_value": {
        "title": "字段/状态",
        "layer": "ask",
        "tools": [
            "resolve_entry",
            "expand_calls",
            "collect_field_evidence",
            "render_field_answer",
        ],
    },
    "change_plan": {
        "title": "改动清单",
        "layer": "plan",
        "tools": ["resolve_entry", "discover_change", "find_outbound", "render_change"],
    },
    "show_code": {
        "title": "源码定位",
        "layer": "ask",
        "tools": ["source_view"],
    },
    "compose_code": {
        "title": "证据生成代码",
        "layer": "ask",
        "tools": ["compose_code"],
    },
    "diagnose": {
        "title": "日志/异常诊断",
        "layer": "ask",
        "tools": ["parse_signal", "map_and_read", "diagnose_answer"],
    },
    "inventory": {
        "title": "入口/文件/架构清单",
        "layer": "ask",
        "tools": [
            "api_inventory",
            "repo_files",
            "codebase_architecture",
            "symbol_probe",
        ],
    },
    "add_api": {
        "title": "新增接口脚手架",
        "layer": "plan",
        "tools": ["load_kb_spec", "find_api_style", "render_new_api"],
    },
    "api_io": {
        "title": "接口入参出参",
        "layer": "ask",
        "tools": ["resolve_entry", "expand_calls", "collect_api_io", "render_api_io"],
    },
    "compound": {
        "title": "新接口+已有改动",
        "layer": "plan",
        "tools": [
            "load_kb_spec",
            "find_api_style",
            "render_new_api",
            "resolve_entry",
            "discover_change",
            "find_outbound",
            "render_change",
        ],
    },
    "general": {
        "title": "综合问答",
        "layer": "ask",
        "tools": ["resolve_entry", "search_hits", "general_answer"],
    },
}


def skill_title(name: str) -> str:
    spec = SKILL_SPECS.get(name) or {}
    return str(spec.get("title") or name)


def known_skill(name: str) -> bool:
    return name in SKILL_NAMES


def skill_layer(name: str) -> str:
    spec = SKILL_SPECS.get(name) or {}
    return str(spec.get("layer") or "ask")
