from __future__ import annotations

from app.agent.patcher import (
    format_change_line_summary,
    normalize_patch_paths,
    resolve_patch_path,
    summarize_patches_with_lines,
)


def test_resolve_package_style_via_hint():
    hint = "loan-service/src/main/java/com/demo/loan/foo/Bar.java"
    got = resolve_patch_path(
        "com/demo/loan/foo/Bar.java",
        hints=[hint],
    )
    assert got == hint


def test_resolve_basename_via_hint():
    hint = "loan-service/src/main/java/com/demo/loan/foo/Bar.java"
    assert resolve_patch_path("Bar.java", hints=[hint]) == hint


def test_normalize_patch_paths_updates_file():
    hint = "loan-service/src/main/java/com/demo/loan/foo/Bar.java"
    patches = normalize_patch_paths(
        [
            {
                "file": "com/demo/loan/foo/Bar.java",
                "replacements": [{"old": "a", "new": "b"}],
            }
        ],
        hints=[hint],
    )
    assert patches[0]["file"] == hint


def test_format_and_summarize_new_file_lines():
    s = format_change_line_summary("a/B.java", new_file=True, end_line=12)
    assert "B.java" in s and "1–12" in s
    lines = summarize_patches_with_lines(
        [{"file": "a/B.java", "new_file": True, "content": "x\ny\nz\n"}]
    )
    assert lines and "B.java" in lines[0]
    assert "第 1–" in lines[0]


def test_apply_replacements_reports_lines(tmp_path, monkeypatch):
    from app.agent import patcher as patcher_mod
    from app.core.config import get_settings

    root = tmp_path / "proj"
    root.mkdir()
    rel = "src/Foo.java"
    (root / "src").mkdir()
    (root / rel).write_text("class Foo {\n  int a = 1;\n}\n", encoding="utf-8")
    # NOTE: get_settings() is lru_cached — mutating the cached instance then
    # calling cache_clear() rebuilds from env vars, so the env must be set too.
    monkeypatch.setenv("PROJECT_ROOT", str(root))
    monkeypatch.setenv("AGENT_WRITE_MODE", "inplace")
    get_settings.cache_clear()
    from app.agent.patcher import apply_replacements

    res = apply_replacements(rel, [{"old": "  int a = 1;", "new": "  int a = 2;\n  int b = 3;"}])
    assert res["ok"] is True
    assert res["changes"]
    assert res["changes"][0]["start_line"] == 2
    assert "line_summary" in res
    assert "第 2" in res["line_summary"]
