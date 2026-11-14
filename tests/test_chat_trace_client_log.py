from __future__ import annotations

from pathlib import Path

from app.agent.chat_trace import (
    AppendFileLogger,
    log_chat_result,
    log_client_run_event,
)


def test_log_client_run_event_writes_command_and_result(tmp_path: Path):
    path = tmp_path / "desktop-chat-test.log"
    logger = AppendFileLogger(path, header="test")
    log_client_run_event(
        logger,
        {
            "type": "tool_event",
            "event": "tool_call",
            "run_id": "run_1",
            "action": {
                "id": "a1",
                "type": "run_powershell",
                "command": "where.exe java; java -version",
                "needs_confirm": False,
                "read_only": True,
            },
        },
    )
    log_client_run_event(
        logger,
        {
            "type": "tool_event",
            "event": "tool_result",
            "run_id": "run_1",
            "action": {
                "id": "a1",
                "type": "run_powershell",
                "command": "where.exe java; java -version",
            },
            "result": {
                "ok": True,
                "exit_code": -1073741819,
                "stdout": "D:\\java\\java17\\bin\\java.exe\n",
                "stderr": 'java version "17.0.10"\n',
                "meta": {"soft_ok": True},
            },
        },
    )
    log_client_run_event(
        logger,
        {
            "type": "run_finished",
            "run_id": "run_1",
            "status": "succeeded",
            "phase": "done",
            "message": "完成",
            "reply": "本机已安装 Java 17，路径 D:\\java\\java17\\bin\\java.exe",
        },
    )
    text = path.read_text(encoding="utf-8")
    assert "CLIENT TOOL CALL" in text
    assert "where.exe java; java -version" in text
    assert "CLIENT TOOL RESULT" in text
    assert "java version" in text
    assert "soft_ok" in text
    assert "CLIENT RUN FINISHED" in text
    assert "本机已安装 Java" in text


def test_log_chat_result_marks_async_client_run(tmp_path: Path):
    path = tmp_path / "turn.log"
    logger = AppendFileLogger(path, header="test")
    log_chat_result(
        logger,
        {
            "intent": "client_agent",
            "phase": "client_running",
            "reply": "正在检查…",
            "client_run_id": "run_abc",
            "session_id": None,
        },
        turn=1,
        append_mode=True,
    )
    text = path.read_text(encoding="utf-8")
    assert "CLIENT RUN (async continues below)" in text
    assert "run_abc" in text
