"""Pilot entry matchers for offline flow analysis.

Defaults are framework-oriented (Controller / Listener / Callback naming).
Override per project via configs/*.yaml `ingest.flow_pilot_entries` or
FLOW_PILOT_ENTRIES_JSON — put domain keywords there, not in code.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("app.analysis.pilot")

# Framework-only defaults — no industry/domain nouns.
DEFAULT_PILOT_ENTRIES: list[dict[str, Any]] = [
    {
        "id": "http_entry",
        "flow_key": "main_flow",
        "flow_role": "main",
        "kinds": ["Controller"],
        "keywords": [],
    },
    {
        "id": "callback_entry",
        "flow_key": "main_flow",
        "flow_role": "callback",
        "kinds": ["Controller", "Listener"],
        "keywords": ["Callback", "CallBack", "Notice", "Notify", "Listener"],
    },
    {
        "id": "mq_listener",
        "flow_key": "main_flow",
        "flow_role": "async",
        "kinds": ["Listener"],
        "keywords": ["RabbitListener", "KafkaListener", "Listener", "Consumer"],
    },
]


@dataclass(frozen=True)
class PilotEntry:
    id: str
    flow_key: str
    flow_role: str  # main | callback | async
    kinds: tuple[str, ...]
    keywords: tuple[str, ...]


def load_pilot_entries() -> list[PilotEntry]:
    settings = get_settings()
    raw: list[dict[str, Any]] | None = None

    env_json = (settings.flow_pilot_entries_json or "").strip()
    if env_json:
        try:
            parsed = json.loads(env_json)
            if isinstance(parsed, list) and parsed:
                raw = parsed
        except json.JSONDecodeError:
            logger.warning("invalid FLOW_PILOT_ENTRIES_JSON; falling back to yaml/defaults")

    if raw is None:
        yaml_cfg = settings.load_yaml()
        ingest = yaml_cfg.get("ingest") or {}
        entries = ingest.get("flow_pilot_entries")
        if isinstance(entries, list) and entries:
            raw = entries

    if raw is None:
        raw = DEFAULT_PILOT_ENTRIES

    result: list[PilotEntry] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id") or "").strip()
        if not eid:
            continue
        kinds = item.get("kinds") or []
        keywords = item.get("keywords") or []
        result.append(
            PilotEntry(
                id=eid,
                flow_key=str(item.get("flow_key") or "main_flow"),
                flow_role=str(item.get("flow_role") or "main"),
                kinds=tuple(str(k) for k in kinds) if kinds else ("Controller", "Listener", "Job"),
                keywords=tuple(str(k) for k in keywords if str(k).strip()),
            )
        )
    return result or [
        PilotEntry(
            id=e["id"],
            flow_key=e["flow_key"],
            flow_role=e["flow_role"],
            kinds=tuple(e["kinds"]),
            keywords=tuple(e["keywords"]),
        )
        for e in DEFAULT_PILOT_ENTRIES
    ]
