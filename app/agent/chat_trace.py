"""Chat turn tracing — same console + file format as tests/local_chat_interactive.py.

Used by LocalChatTester and by /api/v1/chat(+stream) so desktop and CLI share one log style.

Desktop: one log file per conversation (session_id), turns append within that file.
CLI LocalChatTester: still uses OverwriteFileLogger (full buffer rewrite each write).
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = ROOT / "data" / "local_chat_interactive.log"
DESKTOP_LOG_DIR = ROOT / "data" / "desktop_chat"

_lock = threading.RLock()
_api_logger: "OverwriteFileLogger | None" = None
_api_turn = 0
_session_loggers: dict[str, "AppendFileLogger"] = {}
_session_turns: dict[str, int] = {}

_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _safe_console_print(line: str) -> None:
    """Print to console without crashing on Windows GBK (e.g. ✓ in replies)."""
    text = "" if line is None else str(line)
    try:
        print(text, flush=True)
        return
    except UnicodeEncodeError:
        pass
    try:
        stream = getattr(sys.stdout, "buffer", None)
        if stream is not None:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            stream.write((text + "\n").encode(enc, errors="replace"))
            stream.flush()
            return
    except Exception:
        pass
    try:
        print(text.encode("ascii", errors="replace").decode("ascii"), flush=True)
    except Exception:
        pass


def pretty(obj: Any, *, max_chars: int = 120_000) -> str:
    try:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        text = repr(obj)
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... <truncated, total={len(text)} chars>"
    return text


def _safe_session_id(session_id: str) -> str:
    text = (session_id or "").strip() or f"pending-{uuid.uuid4().hex[:12]}"
    text = _SAFE_SESSION_RE.sub("_", text)
    return text[:120] or f"pending-{uuid.uuid4().hex[:12]}"


class ChatLogger(Protocol):
    path: Path

    def log(self, line: str = "") -> None: ...

    def section(self, title: str) -> None: ...

    def subsection(self, title: str) -> None: ...


class OverwriteFileLogger:
    """内存累积全部日志行；每次写入都覆盖指定文件；同步打印到控制台。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lines: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.reset(header=f"chat trace log started at {ts()}")

    def reset(self, header: str | None = None) -> None:
        with _lock:
            self._lines = []
            if header:
                self._lines.append(header)
                self._lines.append("")
            self._flush_unlocked()

    def log(self, line: str = "") -> None:
        with _lock:
            self._lines.append(line)
            _safe_console_print(line)
            self._flush_unlocked()

    def section(self, title: str) -> None:
        bar = "=" * 72
        self.log("")
        self.log(bar)
        self.log(f"[{ts()}] {title}")
        self.log(bar)

    def subsection(self, title: str) -> None:
        self.log("")
        self.log("-" * 56)
        self.log(f"[{ts()}] {title}")
        self.log("-" * 56)

    def _flush_unlocked(self) -> None:
        self.path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")


class AppendFileLogger:
    """Append-only file logger — multi-turn conversation keeps growing in one file."""

    def __init__(self, path: Path, *, header: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        exists = self.path.exists() and self.path.stat().st_size > 0
        if not exists:
            start = header or f"desktop chat log started at {ts()}"
            self._append_lines([*str(start).splitlines(), ""])

    def reset(self, header: str | None = None) -> None:
        with _lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            start = header or f"desktop chat log started at {ts()}"
            body = "\n".join(str(start).splitlines()) + "\n\n"
            self.path.write_text(body, encoding="utf-8")
            for line in str(start).splitlines():
                _safe_console_print(line)

    def log(self, line: str = "") -> None:
        with _lock:
            _safe_console_print(line)
            self._append_lines([line])

    def section(self, title: str) -> None:
        bar = "=" * 72
        self.log("")
        self.log(bar)
        self.log(f"[{ts()}] {title}")
        self.log(bar)

    def subsection(self, title: str) -> None:
        self.log("")
        self.log("-" * 56)
        self.log(f"[{ts()}] {title}")
        self.log("-" * 56)

    def _append_lines(self, lines: list[str]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")


def get_api_chat_logger(path: Path | None = None) -> OverwriteFileLogger:
    """Process-wide logger (legacy). Prefer get_session_chat_logger for desktop."""
    global _api_logger
    with _lock:
        if _api_logger is None:
            p = Path(path) if path else DEFAULT_LOG_PATH
            _api_logger = OverwriteFileLogger(p)
            _api_logger.log(f"[api] chat trace → {_api_logger.path}")
        return _api_logger


def get_session_chat_logger(
    session_id: str | None,
    *,
    project_id: str | None = None,
) -> AppendFileLogger:
    """One append log file per desktop conversation under data/desktop_chat/."""
    sid = _safe_session_id(session_id or "")
    with _lock:
        logger = _session_loggers.get(sid)
        if logger is None:
            path = DESKTOP_LOG_DIR / f"{sid}.log"
            header = (
                f"desktop chat log started at {ts()}\n"
                f"session_id={sid}\n"
                f"project_id={project_id or ''}\n"
                f"log_file={path}\n"
                f"(turns append within this conversation)"
            )
            logger = AppendFileLogger(path, header=header)
            _session_loggers[sid] = logger
            logger.log(f"[session] log append → {path}")
        return logger


def next_session_turn(session_id: str | None) -> int:
    sid = _safe_session_id(session_id or "")
    with _lock:
        _session_turns[sid] = int(_session_turns.get(sid) or 0) + 1
        return _session_turns[sid]


def _clip(text: Any, max_chars: int = 4_000) -> str:
    s = "" if text is None else str(text)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"\n... <truncated, total={len(s)} chars>"


def _action_preview(action: dict[str, Any] | None) -> list[str]:
    """Human-readable lines for a client action (command/path/ids…)."""
    if not isinstance(action, dict):
        return []
    lines: list[str] = []
    atype = action.get("type") or "?"
    aid = action.get("id") or ""
    lines.append(f"action_id={aid}  type={atype}")
    for key in (
        "command",
        "cwd",
        "path",
        "file",
        "drive",
        "ids",
        "remote",
        "ref",
        "branch",
        "note",
        "reply",
        "preview",
        "shell",
        "needs_confirm",
        "read_only",
        "risk",
    ):
        if key in action and action[key] not in (None, "", [], {}):
            val = action[key]
            if isinstance(val, (dict, list)):
                lines.append(f"{key}=")
                for line in pretty(val, max_chars=2_000).splitlines():
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}={_clip(val, 2_000)}")
    return lines


def log_client_run_event(
    logger: ChatLogger,
    msg: dict[str, Any],
    *,
    seq: int | None = None,
) -> None:
    """Append client-agent WS/loop events into the desktop conversation log."""
    if not msg:
        return
    mtype = str(msg.get("type") or "")
    event = str(msg.get("event") or "")
    run_id = msg.get("run_id") or ""
    prefix = f"  [{seq:03d}] " if seq is not None else "  "

    # Noise: skip high-frequency status without useful delta.
    if mtype == "run_status":
        status = msg.get("status")
        message = msg.get("message")
        if status in {"waiting_client", "waiting_confirm", "observing", "planning", "repairing"}:
            logger.log(
                f"{prefix}CLIENT status={status}"
                + (f"  run_id={run_id}" if run_id else "")
                + (f"  message={message}" if message else "")
            )
        return

    if mtype == "hello":
        logger.subsection(f"CLIENT RUN START  run_id={run_id}")
        logger.log(f"status={msg.get('status')}  message={msg.get('message') or ''}")
        return

    if mtype == "tool_event" and event == "task_plan":
        logger.subsection(f"CLIENT TASK PLAN  run_id={run_id}")
        steps = msg.get("plan_steps") or []
        for i, step in enumerate(steps, 1):
            logger.log(f"  {i}. {step}")
        crit = msg.get("success_criteria")
        if crit:
            logger.log(f"success_criteria={crit}")
        if msg.get("message"):
            logger.log(f"message={msg.get('message')}")
        return

    if mtype == "tool_event" and event == "tool_call":
        logger.subsection(f"CLIENT TOOL CALL  run_id={run_id}")
        for line in _action_preview(msg.get("action") if isinstance(msg.get("action"), dict) else None):
            logger.log(f"  {line}")
        return

    if mtype == "action":
        # Same payload as tool_call; avoid duplicate sections in the log file.
        return

    if mtype == "tool_event" and event == "tool_result":
        logger.subsection(f"CLIENT TOOL RESULT  run_id={run_id}")
        action = msg.get("action") if isinstance(msg.get("action"), dict) else None
        result = msg.get("result") if isinstance(msg.get("result"), dict) else None
        if action:
            logger.log(f"  action_id={action.get('id')}  type={action.get('type')}")
            cmd = action.get("command")
            if cmd:
                logger.log(f"  command={_clip(cmd, 2_000)}")
        if result:
            logger.log(f"  ok={result.get('ok')}  exit_code={result.get('exit_code')}")
            if result.get("error"):
                logger.log(f"  error={_clip(result.get('error'), 1_500)}")
            meta = result.get("meta")
            if isinstance(meta, dict) and meta:
                interesting = {
                    k: meta[k]
                    for k in (
                        "preview",
                        "soft_ok",
                        "soft_ok_server",
                        "rejected",
                        "server_guard",
                    )
                    if k in meta
                }
                if interesting:
                    logger.log(f"  meta={interesting}")
            for stream_name in ("stdout", "stderr", "content"):
                blob = result.get(stream_name)
                if blob:
                    logger.log(f"  --- {stream_name} ---")
                    for line in _clip(blob, 6_000).splitlines() or [""]:
                        logger.log(f"  {line}")
        return

    if mtype == "run_finished":
        logger.subsection(f"CLIENT RUN FINISHED  run_id={run_id}")
        logger.log(f"status={msg.get('status')}  phase={msg.get('phase')}")
        logger.log(f"message={msg.get('message') or ''}")
        reply = msg.get("reply")
        if reply:
            logger.log("--- reply ---")
            logger.log(str(reply))
        opts = msg.get("options") or []
        if opts:
            logger.log(f"options({len(opts)})=")
            for line in pretty(opts, max_chars=4_000).splitlines():
                logger.log(f"  {line}")
        logger.log(f"[client run done] log appended → {logger.path}")
        return

    if mtype == "error":
        logger.subsection(f"CLIENT RUN ERROR  run_id={run_id}")
        logger.log(str(msg.get("message") or msg))
        return


def log_client_action_result(
    logger: ChatLogger,
    *,
    run_id: str,
    action: dict[str, Any] | None,
    result: dict[str, Any] | None,
) -> None:
    """Convenience wrapper when only action+result are available (no WS envelope)."""
    log_client_run_event(
        logger,
        {
            "type": "tool_event",
            "event": "tool_result",
            "run_id": run_id,
            "action": action or {},
            "result": result or {},
        },
    )


def log_progress_event(
    logger: ChatLogger,
    event: dict[str, Any],
    step_n: int,
    *,
    at_ms: int | None = None,
    dt_ms: int | None = None,
) -> None:
    """Mirror LocalChatTester.on_progress line format.

    ``at_ms`` (距本轮开始) / ``dt_ms`` (距上一事件) 可选：传了就在首行打印耗时，
    便于事后秒级定位「时间花在哪个 step / 哪次 LLM 调用」（desktop-chat-5b263372926e
    整轮 438s 却只有末尾总耗时，无法定位卡点）。CLI LocalChatTester 不传则保持原格式。
    """
    etype = event.get("type") or "event"
    timing = ""
    if at_ms is not None:
        timing += f" at=+{at_ms}ms"
    if dt_ms is not None:
        timing += f" (Δ +{dt_ms}ms)"
    logger.log(f"  [{step_n:03d}] type={etype}{timing}")
    note = event.get("note")
    if note:
        logger.log(f"         note={note}")
    status = event.get("status")
    if status is not None:
        logger.log(f"         status={status}")
    step_id = event.get("step_id")
    if step_id is not None:
        logger.log(f"         step_id={step_id}")
    if etype == "intent":
        logger.log(f"         >>> INTENT = {event.get('intent')}")
        logger.log(f"         source={event.get('source')}")
        logger.log(f"         keywords={event.get('keywords')}")
        logger.log(f"         entry_hint={event.get('entry_hint')}")
        detail = event.get("detail")
        if detail:
            logger.log(f"         detail={detail}")
    for key in (
        # "plan" intentionally omitted — dumped once in RESEARCH PLAN section below
        "review",
        "tool",
        "args",
        "ok",
        "message",
        "options",
        "compile",
        "test",
        "attempt",
        "phase",
        "client_run_id",
        "goal",
        "plan_steps",
        "success_criteria",
    ):
        if key in event and event[key] is not None:
            logger.log(f"         {key}=")
            for line in pretty(event[key]).splitlines():
                logger.log(f"           {line}")
    known = {
        "type",
        "note",
        "status",
        "step_id",
        "plan",
        "review",
        "tool",
        "args",
        "ok",
        "message",
        "options",
        "compile",
        "test",
        "attempt",
        "phase",
        "intent",
        "keywords",
        "entry_hint",
        "detail",
        "need_options",
        "want_apply",
        "want_verify",
        "source",
        "client_run_id",
        "goal",
        "plan_steps",
        "success_criteria",
    }
    extra = {k: v for k, v in event.items() if k not in known}
    if extra:
        logger.log("         extra=")
        for line in pretty(extra).splitlines():
            logger.log(f"           {line}")


def log_chat_result(
    logger: ChatLogger,
    data: dict[str, Any],
    *,
    intent_event: dict[str, Any] | None = None,
    turn: int | None = None,
    append_mode: bool = False,
) -> None:
    """Mirror LocalChatTester post-run sections (INTENT / REPLY / PLAN / …)."""
    from app.agent import session_store

    logger.subsection("INTENT (recognized)")
    logger.log(f"intent={data.get('intent')}")
    if intent_event:
        logger.log(f"source={intent_event.get('source')}")
        logger.log(f"keywords={intent_event.get('keywords')}")
        logger.log(f"entry_hint={intent_event.get('entry_hint')}")
        if intent_event.get("detail"):
            logger.log(f"detail={intent_event.get('detail')}")
        logger.log(
            "flags="
            f"need_options={intent_event.get('need_options')} "
            f"want_apply={intent_event.get('want_apply')} "
            f"want_verify={intent_event.get('want_verify')}"
        )
    else:
        logger.log("(no live intent progress event; using ChatResponse.intent)")

    logger.subsection("AI REPLY")
    logger.log(str(data.get("reply") or ""))
    logger.log("")
    logger.log(
        f"intent={data.get('intent')}  phase={data.get('phase')}  "
        f"selected_option_id={data.get('selected_option_id')}  "
        f"elapsed_ms={data.get('elapsed_ms')}  "
        f"assistant_message_id={data.get('assistant_message_id')}"
    )

    if data.get("options"):
        logger.subsection(f"OPTIONS ({len(data['options'])})")
        for line in pretty(data["options"]).splitlines():
            logger.log(line)

    if data.get("verify_report"):
        logger.subsection("VERIFY REPORT")
        for line in pretty(data["verify_report"]).splitlines():
            logger.log(line)

    if data.get("plan"):
        logger.subsection("RESEARCH PLAN (final)")
        for line in pretty(data["plan"]).splitlines():
            logger.log(line)

    # EXPLORE STEPS omitted intentionally — each step was already printed
    # live in AI STEPS / PROGRESS above, and the final tree is in RESEARCH PLAN.

    if data.get("review"):
        logger.subsection("ANSWER REVIEW")
        for line in pretty(data["review"]).splitlines():
            logger.log(line)

    if data.get("scope"):
        logger.subsection("SCOPE")
        for line in pretty(data["scope"]).splitlines():
            logger.log(line)

    if data.get("retrieved"):
        logger.subsection(f"RETRIEVED ({len(data['retrieved'])})")
        for line in pretty(data["retrieved"]).splitlines():
            logger.log(line)

    client_run_id = data.get("client_run_id")
    if client_run_id and str(data.get("phase") or "") == "client_running":
        logger.subsection("CLIENT RUN (async continues below)")
        logger.log(f"client_run_id={client_run_id}")
        logger.log(
            "本机执行在 WebSocket 中继续；tool_call / stdout / 最终 reply "
            "会追加到本文件（CLIENT TOOL CALL / RESULT / RUN FINISHED）。"
        )

    sid = data.get("session_id")
    if sid:
        ctx = session_store.get_context(str(sid))
        # Keep only fields NOT already printed as dedicated sections above
        # (research_plan → RESEARCH PLAN, scope → SCOPE, review → ANSWER REVIEW,
        #  verify_report → VERIFY REPORT, explore_steps → EXPLORE STEPS removed)
        ctx_unique = {
            k: ctx[k] for k in ctx
            if k not in {"research_plan", "scope", "review", "verify_report", "explore_steps"}
        }
        if ctx_unique:
            logger.subsection("SESSION CONTEXT (unique)")
            for line in pretty(ctx_unique).splitlines():
                logger.log(line)

        history = session_store.list_messages(str(sid))
        logger.subsection(f"CONVERSATION HISTORY ({len(history)} messages)")
        for i, msg in enumerate(history, 1):
            role = msg.get("role") or "?"
            content = str(msg.get("content") or "")
            logger.log(f"--- [{i}] {role} ---")
            logger.log(content)
            meta = msg.get("meta")
            if meta:
                # Keep only meta fields NOT already printed as dedicated sections
                meta_unique = {
                    k: meta[k] for k in meta
                    if k not in {"research_plan", "scope", "review", "verify_report",
                                 "explore_steps", "options", "patches", "patch_results",
                                 "intent_event"}
                }
                if meta_unique:
                    logger.log("  meta=")
                    for line in pretty(meta_unique).splitlines():
                        logger.log(f"    {line}")
            logger.log("")

    mode = "appended" if append_mode else "overwritten"
    if turn is not None:
        logger.log(f"[turn {turn} done] log {mode} → {logger.path}")
    else:
        logger.log(f"[turn done] log {mode} → {logger.path}")


class ApiChatTurnTrace:
    """One HTTP chat turn with the same live progress logging as LocalChatTester.

    Desktop conversations: append to data/desktop_chat/<session_id>.log
    """

    def __init__(
        self,
        *,
        source: str = "desktop-api",
        session_id: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self.source = source
        self.step_n = 0
        self.intent_event: dict[str, Any] | None = None
        self.session_key = _safe_session_id(session_id or "")
        self.logger = get_session_chat_logger(
            self.session_key,
            project_id=project_id,
        )
        self.turn = next_session_turn(self.session_key)
        # Per-turn monotonic clock for at_ms / dt_ms progress timestamps (P2).
        self._t0 = time.perf_counter()
        self._last_ms = 0

    def begin(
        self,
        message: str,
        *,
        project_id: str | None,
        session_id: str | None,
        client_agent: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.logger.section(f"TURN {self.turn}  USER QUESTION  ({self.source})")
        self.logger.log(message)
        self.logger.log("")
        self.logger.log(f"session_id={session_id or self.session_key}")
        self.logger.log(f"project_id={project_id}")
        self.logger.log(f"client_agent={client_agent}")
        self.logger.log(f"log_file={self.logger.path} (append within conversation)")
        if extra:
            for k, v in extra.items():
                if v is not None and v != "" and v != [] and v is not False:
                    self.logger.log(f"{k}={v}")
        self.logger.subsection("AI STEPS / PROGRESS (live)")

    async def on_progress(self, event: dict[str, Any]) -> None:
        self.step_n += 1
        now_ms = int((time.perf_counter() - self._t0) * 1000)
        dt_ms = now_ms - self._last_ms
        self._last_ms = now_ms
        if (event.get("type") or "") == "intent":
            self.intent_event = dict(event)
        log_progress_event(
            self.logger,
            event,
            self.step_n,
            at_ms=now_ms,
            dt_ms=dt_ms,
        )

    def finish(self, result: Any) -> None:
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        # If server issued a new session_id, keep writing under the client key
        # (desktop always sends a stable desktop-chat-* id) but record mapping.
        result_sid = str(data.get("session_id") or "").strip()
        if result_sid and result_sid != self.session_key:
            self.logger.log(
                f"[session] server session_id={result_sid} "
                f"(file bound to {self.session_key})"
            )
        try:
            log_chat_result(
                self.logger,
                data,
                intent_event=self.intent_event,
                turn=self.turn,
                append_mode=True,
            )
        except Exception:
            # Never fail the HTTP/SSE response because console/file logging choked
            # (e.g. Windows GBK + unicode checkmarks in assistant reply).
            try:
                self.logger.subsection("ERROR")
                self.logger.log(
                    "log_chat_result failed (chat result still returned to client):\n"
                    + traceback.format_exc()
                )
                self.logger.log(
                    f"[turn {self.turn} log-partial] log appended → {self.logger.path}"
                )
            except Exception:
                pass

    def fail(self, exc: BaseException) -> None:
        self.logger.subsection("ERROR")
        self.logger.log("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        self.logger.log(f"[turn {self.turn} failed] log appended → {self.logger.path}")
