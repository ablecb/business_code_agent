from __future__ import annotations

import logging
import sys


def _reconfigure_stream_utf8() -> None:
    """Force UTF-8 + errors=replace on stdout/stderr.

    Chinese Windows consoles default to GBK; Java sources can contain
    characters outside GBK (e.g. CJK extension planes), which made every
    ``LLM_LOG_PROMPTS`` log line raise UnicodeEncodeError and spam full
    tracebacks (huge CPU cost, unreadable logs).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


class _QuietPollAccessFilter(logging.Filter):
    """Hide high-frequency UI poll access lines so ingest progress stays visible."""

    _skip = (
        "GET /api/v1/projects ",
        "GET /api/v1/health ",
        "GET /api/v1/jobs/",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._skip)


def setup_logging(level: int = logging.INFO, *, to_stdout: bool = True) -> None:
    """Ensure app.* loggers show in console (PyCharm / uvicorn)."""
    _reconfigure_stream_utf8()
    root = logging.getLogger()
    stream = sys.stdout if to_stdout else sys.stderr
    if not root.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
            stream=stream,
            force=True,
        )
    else:
        root.setLevel(level)
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) in (
                sys.stderr,
                sys.__stderr__,
            ):
                h.stream = stream

    for name in (
        "app",
        "app.ingest",
        "app.parser",
        "app.tagging",
        "app.graph",
        "app.llm",
        "app.analysis",
        "uvicorn.error",
    ):
        logging.getLogger(name).setLevel(level)
    # httpx only shows URL/status; prompt bodies are logged via app.llm
    logging.getLogger("httpx").setLevel(logging.WARNING)

    access = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, _QuietPollAccessFilter) for f in access.filters):
        access.addFilter(_QuietPollAccessFilter())
