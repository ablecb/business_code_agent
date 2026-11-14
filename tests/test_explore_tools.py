from __future__ import annotations

from app.agent.explore_tools import dispatch_tool, tool_catalog_text


def test_tool_catalog_mentions_finish():
    text = tool_catalog_text()
    assert "search_symbols" in text
    assert "resolve_impl" in text
    assert "finish" in text
    assert "tools" in text
    assert "1～3" in text or "1~3" in text or "并行" in text


def test_dispatch_finish():
    out = dispatch_tool("demo", "finish", {"answer": "ok"})
    assert out["finished"] is True
    assert out["answer"] == "ok"


def test_dispatch_unknown():
    out = dispatch_tool("demo", "nope", {})
    assert out["ok"] is False
