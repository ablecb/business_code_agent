"""本地交互式 AI 聊天测试（与页面同源：run_chat + on_progress + chat_trace 日志）。

用法:
  python tests/local_chat_interactive.py
  python tests/local_chat_interactive.py --project demo-loan
  python tests/local_chat_interactive.py --once "放款流程是什么样的"

日志固定写到 data/local_chat_interactive.log（每次写入整文件覆盖，控制台同步打印）。
桌面 /api/v1/chat(+stream) 使用同一套 app.agent.chat_trace 格式写入同一文件。
REPL 命令:
  直接输入问题 → 连续对话（同一 session_id）
  /history     → 打印当前历史对话
  /context     → 打印 session context
  /new         → 新开会话（清空内存历史，日志文件重新覆盖）
  /quit|/exit  → 退出
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.chat_trace import (  # noqa: E402
    DEFAULT_LOG_PATH,
    OverwriteFileLogger,
    log_chat_result,
    log_progress_event,
    pretty,
    ts,
)

DEFAULT_PROJECT_ID = "demo-loan"


class LocalChatTester:
    """像页面一样调用 AI：连续多轮 + 逐步进度回调 + 完整日志。"""

    def __init__(
        self,
        *,
        project_id: str = DEFAULT_PROJECT_ID,
        log_path: Path = DEFAULT_LOG_PATH,
        use_memory_session: bool = False,
    ) -> None:
        self.project_id = project_id
        self.use_memory_session = use_memory_session
        self.logger = OverwriteFileLogger(log_path)
        self.session_id: str | None = None
        self.turn = 0
        self._ready = False

    def setup(self) -> None:
        if self._ready:
            return
        from app.agent import session_store
        from app.core.config import get_settings
        from app.db.mysql import init_db
        from app.llm.factory import get_embedding_provider, get_llm_provider

        get_settings.cache_clear()
        get_llm_provider.cache_clear()
        get_embedding_provider.cache_clear()

        if self.use_memory_session:
            session_store._USE_MEMORY = True
            self.logger.log("[setup] session_store: in-memory")
        else:
            try:
                init_db()
                self.logger.log("[setup] MySQL init_db ok")
            except Exception as e:
                session_store._USE_MEMORY = True
                self.logger.log(
                    f"[setup] MySQL unavailable ({type(e).__name__}: {e}); "
                    "fallback to in-memory session"
                )

        settings = get_settings()
        self.logger.log(f"[setup] project_id={self.project_id}")
        self.logger.log(f"[setup] llm_provider={settings.llm_provider} model={settings.llm_model}")
        self.logger.log(f"[setup] explore_backend={settings.explore_backend}")
        self.logger.log(f"[setup] agent_use_pev={getattr(settings, 'agent_use_pev', None)}")
        self.logger.log(f"[setup] log_file={self.logger.path} (overwrite each write)")
        self._ready = True

    def new_session(self) -> None:
        self.session_id = f"local-chat-{uuid.uuid4().hex[:12]}"
        self.turn = 0
        self.logger.reset(
            header=(
                f"local chat log started at {ts()}\n"
                f"session_id={self.session_id}\n"
                f"project_id={self.project_id}\n"
                f"log_file={self.logger.path}"
            )
        )
        self.logger.log(f"[session] new session_id={self.session_id}")

    async def ask(self, message: str) -> Any:
        """发送一轮提问（连续对话复用 session_id）。"""
        from app.agent.graph_agent import run_chat

        if not self._ready:
            self.setup()
        if not self.session_id:
            self.new_session()

        self.turn += 1
        message = (message or "").strip()
        self.logger.section(f"TURN {self.turn}  USER QUESTION  (local-cli)")
        self.logger.log(message)
        self.logger.log("")
        self.logger.log(f"session_id={self.session_id}")
        self.logger.log(f"project_id={self.project_id}")
        self.logger.log("client_agent=False")
        self.logger.subsection("AI STEPS / PROGRESS (live)")

        step_n = 0
        intent_event: dict[str, Any] | None = None

        async def on_progress(event: dict[str, Any]) -> None:
            nonlocal step_n, intent_event
            step_n += 1
            if (event.get("type") or "") == "intent":
                intent_event = dict(event)
            log_progress_event(self.logger, event, step_n)

        try:
            result = await run_chat(
                message,
                project_id=self.project_id,
                session_id=self.session_id,
                on_progress=on_progress,
            )
        except Exception:
            self.logger.subsection("ERROR")
            import traceback

            self.logger.log(traceback.format_exc())
            raise

        self.session_id = result.session_id or self.session_id
        data = result.model_dump() if hasattr(result, "model_dump") else dict(result)
        log_chat_result(
            self.logger,
            data,
            intent_event=intent_event,
            turn=self.turn,
        )
        return result

    def dump_history(self) -> None:
        from app.agent import session_store

        if not self.session_id:
            self.logger.log("[history] no session yet")
            return
        history = session_store.list_messages(self.session_id)
        self.logger.subsection(f"CONVERSATION HISTORY ({len(history)} messages)")
        for i, msg in enumerate(history, 1):
            role = msg.get("role") or "?"
            content = str(msg.get("content") or "")
            self.logger.log(f"--- [{i}] {role} ---")
            self.logger.log(content)
            self.logger.log("")

    def dump_context(self) -> None:
        from app.agent import session_store

        if not self.session_id:
            self.logger.log("[context] no session yet")
            return
        ctx = session_store.get_context(self.session_id)
        self.logger.subsection("SESSION CONTEXT")
        for line in pretty(ctx).splitlines():
            self.logger.log(line)

    async def repl(self) -> None:
        self.setup()
        self.new_session()
        self.logger.log("")
        self.logger.log("进入交互模式。输入问题开始对话；/quit 退出。")
        self.logger.log(f"完整日志（覆盖写）: {self.logger.path}")
        while True:
            try:
                raw = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                self.logger.log("\n[exit] interrupted")
                break
            if not raw:
                continue
            low = raw.lower()
            if low in {"/quit", "/exit", "quit", "exit"}:
                self.logger.log("[exit] bye")
                break
            if low == "/new":
                self.new_session()
                continue
            if low == "/history":
                self.dump_history()
                continue
            if low == "/context":
                self.dump_context()
                continue
            try:
                result = await self.ask(raw)
                reply = result.reply if hasattr(result, "reply") else ""
                print("\n--- AI 回复预览 ---", flush=True)
                print(reply[:2000] + ("…" if len(reply) > 2000 else ""), flush=True)
                print(f"\n(完整日志已覆盖写入 {self.logger.path})", flush=True)
            except Exception as e:
                print(f"\n[error] {type(e).__name__}: {e}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local interactive AI chat tester")
    p.add_argument("--project", default=DEFAULT_PROJECT_ID, help="project_id")
    p.add_argument(
        "--log",
        default=str(DEFAULT_LOG_PATH),
        help="log file path (overwrite each write)",
    )
    p.add_argument(
        "--memory",
        action="store_true",
        help="force in-memory session (skip MySQL)",
    )
    p.add_argument(
        "--once",
        metavar="QUESTION",
        help="ask one question then exit (non-interactive)",
    )
    p.add_argument(
        "--questions",
        nargs="+",
        metavar="Q",
        help="ask multiple questions in one continuous session then exit",
    )
    return p


async def _amain() -> int:
    args = build_arg_parser().parse_args()
    tester = LocalChatTester(
        project_id=args.project,
        log_path=Path(args.log),
        use_memory_session=bool(args.memory),
    )
    tester.setup()
    if args.once:
        tester.new_session()
        await tester.ask(args.once)
        return 0
    if args.questions:
        tester.new_session()
        for q in args.questions:
            await tester.ask(q)
        return 0
    await tester.repl()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
