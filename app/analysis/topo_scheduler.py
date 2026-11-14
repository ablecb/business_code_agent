"""Topological + priority concurrent AI scheduler.

Rules:
  - If A calls B, C → wait until B and C finish before starting A.
  - Independent nodes run in parallel (async pool / semaphore).
  - Priority: HIGH (entity/DB) > NORMAL (utils/service) > LOW (controllers).
  - Cycles: detect SCCs, break wait-edges inside the cycle, warn, analyze
    with partial callee context.
  - Failure: retry; then mark node failed and propagate fail_flag upward.
  - MySQL cache: never re-analyze a method that already has a description.
"""

from __future__ import annotations

import asyncio
import heapq
import logging
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Awaitable, Callable, Iterable

logger = logging.getLogger("app.analysis.topo_scheduler")


class Priority(IntEnum):
    HIGH = 0  # entity / DO / Mapper / SQL
    NORMAL = 1  # utils / base service
    LOW = 2  # controllers / composite entries


class NodeStatus:
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CACHED = "cached"  # skipped AI — MySQL hit
    SKIPPED = "skipped"


AnalyzeFn = Callable[["ScheduleNode", "SchedulerContext"], Awaitable[dict[str, Any]]]
CacheLookupFn = Callable[[str], Awaitable[dict[str, Any] | None]]
ProgressFn = Callable[[int, int, str], None]
CancelledFn = Callable[[], bool]


@dataclass
class ScheduleNode:
    id: str
    kind: str  # entity | mapper | method | dto
    priority: Priority
    # Hard deps = callees that must complete (topo). Soft deps may be cycle-broken.
    deps: set[str] = field(default_factory=set)
    callers: set[str] = field(default_factory=set)
    meta: dict[str, Any] = field(default_factory=dict)
    status: str = NodeStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    fail_flag: bool = False
    retries: int = 0
    in_cycle: bool = False
    cycle_id: int | None = None
    soft_deps: set[str] = field(default_factory=set)  # cycle-broken deps


@dataclass
class SchedulerContext:
    """Shared view for analyze callbacks."""

    nodes: dict[str, ScheduleNode]
    project_id: str

    def callee_results(self, node_id: str, *, include_failed: bool = True) -> dict[str, dict[str, Any]]:
        node = self.nodes[node_id]
        out: dict[str, dict[str, Any]] = {}
        for dep in node.deps | node.soft_deps:
            d = self.nodes.get(dep)
            if not d:
                continue
            if d.result is not None:
                payload = dict(d.result)
                if d.fail_flag:
                    payload["_callee_failed"] = True
                out[dep] = payload
            elif include_failed and d.fail_flag:
                out[dep] = {"_callee_failed": True, "summary": f"(failed) {dep}"}
        return out


@dataclass
class SchedulerStats:
    total: int = 0
    done: int = 0
    cached: int = 0
    failed: int = 0
    retried: int = 0
    cycles: int = 0
    cycle_nodes: int = 0
    ai_calls: int = 0
    elapsed_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "done": self.done,
            "cached": self.cached,
            "failed": self.failed,
            "retried": self.retried,
            "cycles": self.cycles,
            "cycle_nodes": self.cycle_nodes,
            "ai_calls": self.ai_calls,
            "elapsed_sec": round(self.elapsed_sec, 2),
            "warnings": list(self.warnings[:50]),
        }


def classify_priority(
    *,
    kind: str,
    simple_name: str,
    file_path: str = "",
    method_name: str = "",
) -> Priority:
    """Highest: entity/DB; Normal: utils/base service; Low: controllers."""
    path_l = (file_path or "").replace("\\", "/").lower()
    sn = simple_name or ""
    kind_l = (kind or "").lower()

    if kind_l in {"entity", "mapper", "mapperxml", "dto"} or sn.endswith(
        ("Entity", "DO", "PO", "Dmo", "Mapper")
    ):
        return Priority.HIGH
    if "/entity/" in path_l or "/mapper/" in path_l or "/domain/" in path_l:
        return Priority.HIGH
    if kind_l == "controller" or sn.endswith("Controller"):
        return Priority.LOW
    if kind_l in {"listener", "job"} or sn.endswith(("Listener", "Job")):
        # entry-like → low (after callees)
        return Priority.LOW
    if (
        "Util" in sn
        or "Helper" in sn
        or "Common" in sn
        or "Base" in sn
        or "/util/" in path_l
        or "/common/" in path_l
    ):
        return Priority.NORMAL
    return Priority.NORMAL


def detect_sccs(nodes: dict[str, ScheduleNode]) -> list[list[str]]:
    """Tarjan SCC. Returns list of components (each a list of node ids)."""
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    components: list[list[str]] = []

    def strongconnect(v: str) -> None:
        nonlocal index
        indices[v] = index
        lowlink[v] = index
        index += 1
        stack.append(v)
        on_stack.add(v)

        node = nodes[v]
        for w in node.deps:
            if w not in nodes:
                continue
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                comp.append(w)
                if w == v:
                    break
            components.append(comp)

    for nid in nodes:
        if nid not in indices:
            strongconnect(nid)
    return components


def break_cycles(
    nodes: dict[str, ScheduleNode],
    *,
    warnings: list[str] | None = None,
) -> int:
    """Mark cyclic SCCs and convert intra-SCC deps to soft_deps (break deadlock).

    Returns number of cyclic components.
    """
    warns = warnings if warnings is not None else []
    components = detect_sccs(nodes)
    cycle_count = 0
    for i, comp in enumerate(components):
        is_cycle = len(comp) > 1
        if len(comp) == 1:
            n = nodes[comp[0]]
            if comp[0] in n.deps:
                is_cycle = True
        if not is_cycle:
            continue
        cycle_count += 1
        comp_set = set(comp)
        warns.append(
            f"call-graph cycle detected ({len(comp)} nodes): "
            + ",".join(sorted(comp)[:8])
            + ("…" if len(comp) > 8 else "")
        )
        logger.warning(warns[-1])
        for nid in comp:
            n = nodes[nid]
            n.in_cycle = True
            n.cycle_id = i
            hard = set()
            soft = set(n.soft_deps)
            for d in n.deps:
                if d in comp_set:
                    soft.add(d)
                else:
                    hard.add(d)
            # self-loop
            soft.discard(nid)
            hard.discard(nid)
            n.deps = hard
            n.soft_deps = soft
    return cycle_count


def _deps_satisfied(node: ScheduleNode, nodes: dict[str, ScheduleNode]) -> bool:
    for d in node.deps:
        dep = nodes.get(d)
        if dep is None:
            continue
        if dep.status not in {NodeStatus.DONE, NodeStatus.CACHED, NodeStatus.FAILED, NodeStatus.SKIPPED}:
            return False
    return True


def _propagate_failure(nodes: dict[str, ScheduleNode], start_id: str) -> None:
    """Mark callers (upward) with fail_flag when a lower-level node fails."""
    stack = [start_id]
    seen: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        node = nodes.get(cur)
        if not node:
            continue
        for caller_id in node.callers:
            c = nodes.get(caller_id)
            if not c:
                continue
            if not c.fail_flag:
                c.fail_flag = True
                logger.info("propagate fail_flag → %s (from %s)", caller_id, start_id)
            stack.append(caller_id)


class TopoPriorityScheduler:
    """Concurrent topological scheduler with priority queues."""

    def __init__(
        self,
        nodes: dict[str, ScheduleNode],
        *,
        project_id: str,
        concurrency: int = 8,
        max_retries: int = 2,
        retry_backoff_sec: float = 3.0,
        analyze: AnalyzeFn,
        cache_lookup: CacheLookupFn | None = None,
        progress: ProgressFn | None = None,
        is_cancelled: CancelledFn | None = None,
    ) -> None:
        self.nodes = nodes
        self.project_id = project_id
        self.concurrency = max(1, concurrency)
        self.max_retries = max(0, max_retries)
        self.retry_backoff_sec = max(0.0, retry_backoff_sec)
        self.analyze = analyze
        self.cache_lookup = cache_lookup
        self.progress = progress
        self.is_cancelled = is_cancelled
        self.stats = SchedulerStats(total=len(nodes))
        self.ctx = SchedulerContext(nodes=nodes, project_id=project_id)
        # heap entries: (priority, seq, node_id)
        self._heap: list[tuple[int, int, str]] = []
        self._seq = 0
        self._ready: set[str] = set()
        self._lock = asyncio.Lock()

    def _push_ready(self, nid: str) -> None:
        if nid in self._ready:
            return
        node = self.nodes[nid]
        if node.status not in {NodeStatus.PENDING, NodeStatus.READY}:
            return
        node.status = NodeStatus.READY
        self._ready.add(nid)
        heapq.heappush(self._heap, (int(node.priority), self._seq, nid))
        self._seq += 1

    def _bootstrap_ready(self) -> None:
        for nid, node in self.nodes.items():
            if _deps_satisfied(node, self.nodes):
                self._push_ready(nid)

    def _on_finished(self, nid: str) -> None:
        """After a node completes, unlock callers whose deps are now satisfied."""
        node = self.nodes[nid]
        for caller_id in node.callers:
            c = self.nodes.get(caller_id)
            if not c:
                continue
            if c.status != NodeStatus.PENDING:
                continue
            if _deps_satisfied(c, self.nodes):
                self._push_ready(caller_id)

    async def _pop_ready(self) -> str | None:
        async with self._lock:
            while self._heap:
                _p, _s, nid = heapq.heappop(self._heap)
                if nid not in self._ready:
                    continue
                self._ready.discard(nid)
                node = self.nodes[nid]
                if node.status != NodeStatus.READY:
                    continue
                # Re-check deps (should be true)
                if not _deps_satisfied(node, self.nodes):
                    node.status = NodeStatus.PENDING
                    continue
                node.status = NodeStatus.RUNNING
                return nid
            return None

    async def _finish(
        self,
        nid: str,
        *,
        status: str,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        async with self._lock:
            node = self.nodes[nid]
            node.status = status
            node.result = result
            node.error = error
            if status == NodeStatus.CACHED:
                self.stats.cached += 1
                self.stats.done += 1
            elif status == NodeStatus.DONE:
                self.stats.done += 1
            elif status == NodeStatus.FAILED:
                node.fail_flag = True
                self.stats.failed += 1
                self.stats.done += 1
                _propagate_failure(self.nodes, nid)
            elif status == NodeStatus.SKIPPED:
                self.stats.done += 1
            self._on_finished(nid)
            if self.progress:
                tag = getattr(status, "value", None) or str(status)
                # Compact line for desktop parse log (stage/counts shown separately).
                extra = ""
                if node.in_cycle:
                    extra += " cycle"
                if node.fail_flag and status != NodeStatus.FAILED:
                    extra += " fail↑"
                self.progress(
                    self.stats.done,
                    self.stats.total,
                    f"LLM {tag} {nid} ({self.stats.done}/{self.stats.total}){extra}",
                )

    async def _run_one(self, nid: str) -> None:
        node = self.nodes[nid]
        # MySQL / cache hit — never re-analyze
        if self.cache_lookup is not None:
            try:
                cached = await self.cache_lookup(nid)
            except Exception:
                logger.exception("cache lookup failed %s", nid)
                cached = None
            if cached is not None:
                cached = dict(cached)
                cached["_from_cache"] = True
                await self._finish(nid, status=NodeStatus.CACHED, result=cached)
                return

        attempt = 0
        last_err: Exception | None = None
        # Cap a single analyze attempt so a wedged LLM/proxy cannot stall the pool.
        analyze_timeout = 360.0
        while attempt <= self.max_retries:
            try:
                self.stats.ai_calls += 1
                result = await asyncio.wait_for(
                    self.analyze(node, self.ctx),
                    timeout=analyze_timeout,
                )
                if node.fail_flag:
                    # lower-level failure already propagated; still store result
                    result = dict(result or {})
                    result["_upstream_fail_flag"] = True
                if node.in_cycle:
                    result = dict(result or {})
                    result["_in_cycle"] = True
                    result["_cycle_id"] = node.cycle_id
                await self._finish(nid, status=NodeStatus.DONE, result=result)
                return
            except asyncio.TimeoutError as e:
                last_err = e
                attempt += 1
                node.retries = attempt
                self.stats.retried += 1
                logger.warning(
                    "analyze timeout %s attempt=%s/%s after %.0fs",
                    nid,
                    attempt,
                    self.max_retries,
                    analyze_timeout,
                )
                if attempt <= self.max_retries:
                    await asyncio.sleep(self.retry_backoff_sec * attempt)
            except Exception as e:
                last_err = e
                attempt += 1
                node.retries = attempt
                self.stats.retried += 1
                logger.warning(
                    "analyze failed %s attempt=%s/%s: %s",
                    nid,
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt <= self.max_retries:
                    await asyncio.sleep(self.retry_backoff_sec * attempt)

        await self._finish(
            nid,
            status=NodeStatus.FAILED,
            error=str(last_err) if last_err else "unknown",
            result={"error": str(last_err), "fail_flag": True},
        )

    async def run(self) -> SchedulerStats:
        t0 = time.time()
        # Cycle break before scheduling
        self.stats.cycles = break_cycles(self.nodes, warnings=self.stats.warnings)
        self.stats.cycle_nodes = sum(1 for n in self.nodes.values() if n.in_cycle)
        if self.stats.cycles:
            logger.warning(
                "broke %s cycle(s) affecting %s nodes — analyzing with partial context",
                self.stats.cycles,
                self.stats.cycle_nodes,
            )

        self._bootstrap_ready()
        if not self._heap and self.nodes:
            # All nodes have unsatisfied deps (shouldn't happen after cycle break)
            msg = "scheduler deadlock: no ready nodes after cycle break"
            self.stats.warnings.append(msg)
            logger.error(msg)
            # Force-push all pending as ready to avoid infinite hang
            for nid, n in self.nodes.items():
                if n.status == NodeStatus.PENDING:
                    n.deps.clear()
                    self._push_ready(nid)

        sem = asyncio.Semaphore(self.concurrency)
        inflight: set[asyncio.Task] = set()

        async def _guarded(nid: str) -> None:
            async with sem:
                await self._run_one(nid)

        while self.stats.done < self.stats.total:
            # Cooperative cancel: stop scheduling new work, let the pipeline
            # abort before the (slow) milvus / neo4j write stages.
            if self.is_cancelled is not None and self.is_cancelled():
                logger.warning(
                    "scheduler cancelled by caller; aborting %s remaining nodes",
                    self.stats.total - self.stats.done,
                )
                for t in inflight:
                    t.cancel()
                if inflight:
                    done_cancel, _ = await asyncio.wait(inflight)
                    for t in done_cancel:
                        try:
                            t.result()
                        except (asyncio.CancelledError, Exception):
                            pass
                for nid, n in self.nodes.items():
                    if n.status in {NodeStatus.PENDING, NodeStatus.READY, NodeStatus.RUNNING}:
                        n.status = NodeStatus.SKIPPED
                        n.error = "cancelled"
                self.stats.done = self.stats.total
                break
            # Fill worker pool
            while len(inflight) < self.concurrency:
                nid = await self._pop_ready()
                if nid is None:
                    break
                task = asyncio.create_task(_guarded(nid))
                inflight.add(task)

            if not inflight:
                # Nothing running and nothing ready — stuck
                pending = [
                    n.id
                    for n in self.nodes.values()
                    if n.status in {NodeStatus.PENDING, NodeStatus.READY}
                ]
                if not pending:
                    break
                msg = (
                    f"scheduler stall with {len(pending)} pending; "
                    "force-releasing deps to avoid deadlock"
                )
                self.stats.warnings.append(msg)
                logger.error(msg)
                for nid in pending:
                    n = self.nodes[nid]
                    n.soft_deps |= n.deps
                    n.deps.clear()
                    self._push_ready(nid)
                continue

            done, inflight = await asyncio.wait(
                inflight, return_when=asyncio.FIRST_COMPLETED
            )
            for t in done:
                try:
                    t.result()
                except Exception:
                    logger.exception("worker task crashed")

        self.stats.elapsed_sec = time.time() - t0
        return self.stats


def build_schedule_nodes_from_ast(
    *,
    method_callees: dict[str, list[str]],
    method_meta: dict[str, dict[str, Any]],
    entity_ids: Iterable[str] | None = None,
    entity_meta: dict[str, dict[str, Any]] | None = None,
) -> dict[str, ScheduleNode]:
    """Build ScheduleNode graph.

    method_callees: Class#method -> [callee Class#method]
    entity_ids: type-level nodes (Entity/DTO/Mapper) with no method deps;
      methods do not hard-wait on entities (priority already prefers them),
      but entity nodes are scheduled first via HIGH priority + empty deps.
    """
    nodes: dict[str, ScheduleNode] = {}
    entity_meta = entity_meta or {}

    for eid in entity_ids or []:
        meta = dict(entity_meta.get(eid) or {})
        nodes[eid] = ScheduleNode(
            id=eid,
            kind=str(meta.get("kind") or "entity"),
            priority=classify_priority(
                kind=str(meta.get("kind") or "entity"),
                simple_name=str(meta.get("simple_name") or eid.split(".")[-1]),
                file_path=str(meta.get("file_path") or ""),
            ),
            meta=meta,
        )

    # Ensure all methods exist
    all_methods = set(method_callees) | {
        c for callees in method_callees.values() for c in callees
    } | set(method_meta)
    for mid in all_methods:
        meta = dict(method_meta.get(mid) or {})
        simple = str(meta.get("simple_name") or (mid.split("#")[0] if "#" in mid else mid))
        nodes[mid] = ScheduleNode(
            id=mid,
            kind=str(meta.get("kind") or "method"),
            priority=classify_priority(
                kind=str(meta.get("symbol_kind") or meta.get("kind") or "method"),
                simple_name=simple,
                file_path=str(meta.get("file_path") or ""),
                method_name=str(meta.get("method_name") or (mid.split("#")[-1] if "#" in mid else "")),
            ),
            meta=meta,
        )

    # Wire deps: caller waits for callees
    for caller, callees in method_callees.items():
        if caller not in nodes:
            continue
        for callee in callees:
            if callee not in nodes or callee == caller:
                continue
            nodes[caller].deps.add(callee)
            nodes[callee].callers.add(caller)

    return nodes
