"""Ask runtime: group DAG nodes and run multi-entry decompose."""
from __future__ import annotations

from typing import Any

from app.agent.runtime.types import SOLO_SKILLS


def group_subtasks(
    tasks: list[dict[str, Any]],
    *,
    share_entry: bool = True,
) -> list[list[dict[str, Any]]]:
    """Group DAG nodes for execution.

    Same entry_hint → one group (shared spine). Different APIs → separate groups.
    inventory/diagnose/add_api stay solo unless share_entry is forced.
    """
    if not tasks:
        return []
    if share_entry or len(tasks) == 1:
        return [list(tasks)]

    buckets: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    solos: list[dict[str, Any]] = []
    unhinted: list[dict[str, Any]] = []

    for t in tasks:
        kind = str(t.get("kind") or t.get("skill") or "")
        hint = str(t.get("entry_hint") or "").strip()
        if kind in SOLO_SKILLS:
            solos.append(t)
            continue
        if not hint:
            unhinted.append(t)
            continue
        if hint not in buckets:
            order.append(hint)
            buckets[hint] = []
        buckets[hint].append(t)

    groups = [buckets[h] for h in order]
    if unhinted:
        if groups:
            groups[0].extend(unhinted)
        else:
            groups.append(unhinted)
    for s in solos:
        groups.append([s])
    return [g for g in groups if g]


def serial_task_order(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep utterance / depends_on order. Never move a later node in front."""
    if not tasks:
        return []
    ids: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for i, t in enumerate(tasks):
        tid = str(t.get("id") or f"S{i + 1}")
        ids.append(tid)
        by_id[tid] = t
    deps: dict[str, list[str]] = {}
    has_dep = False
    idset = set(ids)
    for i, t in enumerate(tasks):
        tid = ids[i]
        raw = t.get("depends_on") if isinstance(t.get("depends_on"), list) else []
        ds = [str(d) for d in raw if str(d) in idset and str(d) != tid]
        deps[tid] = ds
        if ds:
            has_dep = True
    if not has_dep:
        return list(tasks)
    idx = {tid: i for i, tid in enumerate(ids)}
    remaining = set(ids)
    out: list[dict[str, Any]] = []
    while remaining:
        ready = [
            tid
            for tid in ids
            if tid in remaining and all(d not in remaining for d in deps[tid])
        ]
        if not ready:
            out.extend(by_id[tid] for tid in ids if tid in remaining)
            break
        ready.sort(key=lambda x: idx[x])
        pick = ready[0]
        remaining.remove(pick)
        out.append(by_id[pick])
    return out


def serial_groups(
    tasks: list[dict[str, Any]],
    *,
    share_entry: bool = True,
) -> list[list[dict[str, Any]]]:
    """Execute groups in semantic order: consecutive same entry_hint may share a spine.

    Unlike group_subtasks, an unhinted later node (改动清单) stays after earlier asks.
    """
    ordered = serial_task_order(tasks)
    if not ordered:
        return []
    if share_entry or len(ordered) == 1:
        return [list(ordered)]
    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    cur_key: str | None = None
    for t in ordered:
        key = str(t.get("entry_hint") or "").strip()
        if cur and key != cur_key:
            groups.append(cur)
            cur = []
        cur.append(t)
        cur_key = key
    if cur:
        groups.append(cur)
    return groups
