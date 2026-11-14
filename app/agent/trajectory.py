"""Append-only agent trajectory (DeepSeek-harness-style session log).

Records progress / tool / verify events as JSONL beside chat history so turns
can be inspected, eval'd, and (later) replayed without bloating context_json.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from app.core.config import ROOT_DIR, get_settings
from app.core.utils import truncate

logger = logging.getLogger("app.agent.trajectory")

ProgressCb = Callable[[dict[str, Any]], Any] | None

_lock = threading.Lock()


def trajectory_enabled() -> bool:
    try:
        return bool(get_settings().agent_trajectory_enabled)
    except Exception:
        return True


def trajectory_dir() -> Path:
    settings = get_settings()
    raw = (settings.agent_trajectory_dir or "data/trajectories").strip()
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT_DIR / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def trajectory_path(session_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (session_id or "anon"))
    return trajectory_dir() / f"{safe}.jsonl"


def _compact_payload(payload: dict[str, Any], *, compact: bool) -> dict[str, Any]:
    if not compact:
        out = dict(payload)
        # Drop huge nested plan trees from progress events unless asked.
        plan = out.get("plan")
        if isinstance(plan, dict) and "steps" in plan:
            steps = plan.get("steps") or []
            out["plan"] = {
                "kind": plan.get("kind"),
                "title": plan.get("title"),
                "step_count": len(steps) if isinstance(steps, list) else 0,
                "entry": plan.get("entry"),
            }
        return out
    keep = (
        "type",
        "kind",
        "step_id",
        "status",
        "note",
        "tool",
        "ok",
        "entry",
        "attempt",
        "intent",
        "phase",
        "elapsed_ms",
        "assistant_message_id",
        "error",
        "mode",
    )
    out: dict[str, Any] = {}
    for k in keep:
        if k in payload and payload[k] is not None:
            v = payload[k]
            if isinstance(v, str):
                out[k] = truncate(v, 240)
            else:
                out[k] = v
    return out


class AppendTrajectory:
    """Thread-safe JSONL writer for one chat session."""

    def __init__(
        self,
        session_id: str,
        *,
        turn_id: str | None = None,
        compact: bool | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.session_id = session_id or "anon"
        self.turn_id = turn_id or uuid.uuid4().hex[:12]
        self.path = trajectory_path(self.session_id)
        self._seq = 0
        self._enabled = trajectory_enabled() if enabled is None else bool(enabled)
        if compact is None:
            try:
                mode = str(get_settings().agent_runtime_mode or "standard")
            except Exception:
                mode = "standard"
            compact = mode == "minimal"
        self.compact = bool(compact)

    @classmethod
    def for_session(
        cls,
        session_id: str,
        *,
        turn_id: str | None = None,
        compact: bool | None = None,
    ) -> AppendTrajectory:
        return cls(session_id, turn_id=turn_id, compact=compact)

    def emit(self, kind: str, **payload: Any) -> dict[str, Any] | None:
        if not self._enabled:
            return None
        body = _compact_payload(payload, compact=self.compact)
        with _lock:
            self._seq += 1
            seq = self._seq
            record = {
                "ts": time.time(),
                "session_id": self.session_id,
                "turn_id": self.turn_id,
                "seq": seq,
                "kind": kind,
                **body,
            }
            try:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except Exception:
                logger.debug("trajectory write failed", exc_info=True)
                return None
        return record


def wrap_progress(
    on_progress: ProgressCb,
    traj: AppendTrajectory | None,
) -> ProgressCb:
    """Record progress events then forward to the original callback."""
    if traj is None or not traj._enabled:  # noqa: SLF001
        return on_progress

    async def _wrapped(event: dict[str, Any]) -> None:
        try:
            payload = dict(event or {})
            payload["type"] = str(payload.get("type") or "progress")
            traj.emit("progress", **payload)
        except Exception:
            logger.debug("trajectory progress record failed", exc_info=True)
        if not on_progress:
            return
        maybe = on_progress(event)
        if hasattr(maybe, "__await__"):
            await maybe  # type: ignore[misc]

    return _wrapped


def read_trajectory(
    session_id: str,
    *,
    turn_id: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict[str, Any]]:
    path = trajectory_path(session_id)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if turn_id and str(row.get("turn_id") or "") != turn_id:
                    continue
                rows.append(row)
    except Exception:
        logger.debug("trajectory read failed", exc_info=True)
        return []
    if offset:
        rows = rows[offset:]
    if limit and len(rows) > limit:
        rows = rows[-limit:]
    return rows
