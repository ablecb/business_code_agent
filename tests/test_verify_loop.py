from __future__ import annotations

from pathlib import Path


def test_detect_build_root_finds_maven(tmp_path: Path, monkeypatch):
    from app.agent import verify_loop

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "pom.xml").write_text("<project/>", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()

    class _S:
        project_root_path = proj
        workspace_path = ws
        agent_write_mode = "workspace"

    monkeypatch.setattr(verify_loop, "get_settings", lambda: _S())
    root, tool = verify_loop.detect_build_root()
    assert tool == "mvn"
    assert root == proj or root == ws


def test_snapshot_and_rollback(tmp_path: Path, monkeypatch):
    from app.agent import verify_loop

    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "a.txt").write_text("v1", encoding="utf-8")
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "a.txt").write_text("v1", encoding="utf-8")
    snap_root = tmp_path / "snaps"
    snap_root.mkdir()

    class _S:
        project_root_path = proj
        workspace_path = ws
        agent_write_mode = "workspace"

    monkeypatch.setattr(verify_loop, "get_settings", lambda: _S())
    monkeypatch.setattr(verify_loop, "snapshot_dir", lambda: snap_root)

    meta = verify_loop.create_workspace_snapshot(label="t")
    assert meta.get("id")
    (ws / "a.txt").write_text("v2", encoding="utf-8")
    result = verify_loop.rollback_workspace(meta["id"])
    assert result.get("ok")
    assert (ws / "a.txt").read_text(encoding="utf-8") == "v1"


def test_render_verify_reply():
    from app.agent.verify_loop import render_verify_reply

    text = render_verify_reply(
        {
            "ok": False,
            "compile_ok": False,
            "test_ok": True,
            "test_skipped": True,
            "attempts": 2,
            "logs": "error: cannot find symbol",
        },
        repaired=True,
    )
    assert "验证结果" in text
    assert "编译" in text
