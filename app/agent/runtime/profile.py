"""Runtime profiles inspired by DeepSeek Harness modes (standard / minimal)."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

from app.core.config import get_settings

# agentscope = AgentScope 迁移层（agentscope_bridge），standard/minimal 仍是原编排。
RuntimeMode = Literal["standard", "minimal", "agentscope"]

# Minimal eval surface: entry + spine + read/answer tools (no scaffold/compose/diagnose).
MINIMAL_TOOLS: frozenset[str] = frozenset(
    {
        "resolve_entry",
        "expand_calls",
        "mysql_enrich",
        "summarize_flow",
        "source_view",
        "collect_field_evidence",
        "render_field_answer",
        "collect_api_io",
        "render_api_io",
        "search_hits",
        "general_answer",
        "discover_change",
        "find_outbound",
        "render_change",
    }
)

MINIMAL_SKILLS: frozenset[str] = frozenset(
    {
        "explain_flow",
        "field_value",
        "change_plan",
        "show_code",
        "api_io",
        "general",
    }
)


@dataclass(frozen=True)
class RuntimeProfile:
    mode: RuntimeMode = "standard"
    allow_decompose: bool = True
    carry_dig_in_pins: bool = True
    compact_trajectory: bool = False
    max_repairs: int = 3
    allowed_tools: frozenset[str] | None = None
    allowed_skills: frozenset[str] | None = None

    def tool_allowed(self, name: str) -> bool:
        if self.allowed_tools is None:
            return True
        return name in self.allowed_tools

    def skill_allowed(self, name: str) -> bool:
        if self.allowed_skills is None:
            return True
        return name in self.allowed_skills


@lru_cache
def get_runtime_profile() -> RuntimeProfile:
    settings = get_settings()
    raw = str(settings.agent_runtime_mode).strip().lower()
    # agentscope 模式走迁移层（agentscope_bridge），工具/技能约束与 standard 相同；
    # 仅标记 mode 供调用方识别（adapter 在 graph_agent.run_chat 入口切换）。
    mode: RuntimeMode = raw if raw in ("standard", "minimal", "agentscope") else "standard"
    if mode == "minimal":
        max_repairs = int(getattr(settings, "agent_minimal_max_repairs", 0) or 0)
        return RuntimeProfile(
            mode="minimal",
            allow_decompose=False,
            carry_dig_in_pins=False,
            compact_trajectory=True,
            max_repairs=max(0, max_repairs),
            allowed_tools=MINIMAL_TOOLS,
            allowed_skills=MINIMAL_SKILLS,
        )
    return RuntimeProfile(
        mode=mode,
        allow_decompose=True,
        carry_dig_in_pins=True,
        compact_trajectory=False,
        max_repairs=3,
        allowed_tools=None,
        allowed_skills=None,
    )


def clear_runtime_profile_cache() -> None:
    get_runtime_profile.cache_clear()
