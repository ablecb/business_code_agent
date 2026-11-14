"""Task DAG + skill names for the ask runtime."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SKILL_NAMES = frozenset(
    {
        "explain_flow",
        "field_value",
        "change_plan",
        "show_code",
        "compose_code",
        "diagnose",
        "inventory",
        "add_api",
        "api_io",
        "general",
        "compound",
    }
)

# Skills that must not be forced onto another node's Controller spine.
SOLO_SKILLS = frozenset({"inventory", "diagnose", "add_api", "repo_files", "compose_code"})


@dataclass
class TaskNode:
    id: str
    skill: str
    title: str
    question: str
    entry_hint: str = ""
    depends_on: list[str] = field(default_factory=list)
    inner_loop: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.skill,
            "skill": self.skill,
            "title": self.title,
            "question": self.question,
            "entry_hint": self.entry_hint,
            "depends_on": list(self.depends_on),
            "inner_loop": bool(self.inner_loop),
        }


@dataclass
class TaskDAG:
    nodes: list[TaskNode]
    source: str = "heuristic"
    compose: str = ""
    share_entry: bool = True

    def to_legacy(self) -> dict[str, Any]:
        """Shape expected by pev_agent._run_decomposed_pipeline."""
        return {
            "tasks": [n.to_dict() for n in self.nodes],
            "compose": self.compose,
            "source": self.source,
            "share_entry": bool(self.share_entry),
        }
