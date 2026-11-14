"""Client action policy + protocol coverage for shell/git."""
from __future__ import annotations


def test_needs_confirm_shell_and_git():
    from app.agent.action_policy import annotate_action_dict, needs_confirm

    assert needs_confirm("run_shell") is True
    assert needs_confirm("git_push") is True
    assert needs_confirm("git_merge") is True
    assert needs_confirm("maven_compile") is False
    assert needs_confirm("read_file") is False
    # force confirm cannot be disabled for push/cleanup
    assert needs_confirm("git_push", explicit=False) is True
    assert needs_confirm("sys_cleanup", explicit=False) is True
    # shell can be skipped when marked explicit false (read probes use annotate)
    assert needs_confirm("run_shell", explicit=False) is False

    a = annotate_action_dict(
        {"type": "run_shell", "command": "Remove-Item -Recurse tmp", "id": "a1"}
    )
    assert a["needs_confirm"] is True
    assert a["policy_tier"] == "shell"
    assert "Remove-Item" in a["ui_preview"]


def test_readonly_allows_java_version_redirect():
    from app.agent.action_policy import annotate_action_dict, is_readonly_shell_command

    cmd = 'where.exe java; cmd /c "java -version 2>&1"'
    assert is_readonly_shell_command(cmd)
    a = annotate_action_dict({"type": "run_powershell", "command": cmd, "id": "j1"})
    assert a["needs_confirm"] is False
    assert a["read_only"] is True


def test_readonly_shell_skips_confirm():
    from app.agent.action_policy import annotate_action_dict, is_readonly_shell_command

    assert is_readonly_shell_command("where.exe git; git --version")
    assert is_readonly_shell_command("Get-Command git")
    assert not is_readonly_shell_command("Remove-Item C:\\temp\\x")
    assert not is_readonly_shell_command("git push origin main")

    a = annotate_action_dict(
        {
            "type": "run_powershell",
            "command": "where.exe git; git --version",
            "id": "a2",
        }
    )
    assert a["needs_confirm"] is False
    assert a["read_only"] is True
    assert "无需确认" in a["ui_preview"] or a["ui_title"] == "查看本机信息"

    b = annotate_action_dict({"type": "git_status", "id": "a3"})
    assert b["needs_confirm"] is False
    assert b["read_only"] is True


def test_allowed_actions_include_shell_git():
    from app.agent.remote_protocol import ALLOWED_ACTIONS

    for t in (
        "run_shell",
        "run_powershell",
        "git_status",
        "git_checkout",
        "git_merge",
        "git_commit",
        "git_push",
        "http_smoke",
        "open_browser",
        "run_sql",
        "run_cypher",
        "milvus_search",
        "milvus_mutate",
    ):
        assert t in ALLOWED_ACTIONS


def test_sql_cypher_browser_milvus_policy():
    from app.agent.action_policy import (
        annotate_action_dict,
        is_readonly_cypher,
        is_readonly_sql,
        needs_confirm,
    )

    assert is_readonly_sql("SELECT 1")
    assert is_readonly_sql("SHOW TABLES")
    assert is_readonly_sql("EXPLAIN SELECT * FROM t")
    assert not is_readonly_sql("UPDATE t SET a=1")
    assert not is_readonly_sql("INSERT INTO t VALUES (1)")
    assert not is_readonly_sql("SELECT 1; DROP TABLE t")

    assert is_readonly_cypher("MATCH (n) RETURN n LIMIT 5")
    assert not is_readonly_cypher("MATCH (n) SET n.x = 1 RETURN n")
    assert not is_readonly_cypher("CREATE (n:Foo)")

    a = annotate_action_dict(
        {
            "type": "run_sql",
            "id": "s1",
            "sql": "SELECT id FROM symbol LIMIT 3",
            "mysql": {
                "host": "127.0.0.1",
                "user": "u",
                "password": "secret",
                "database": "db",
            },
        }
    )
    assert a["needs_confirm"] is False
    assert a["read_only"] is True
    assert a["mysql"]["password"] == "secret"
    assert "secret" not in a["ui_preview"]
    assert "SELECT" in a["ui_preview"]

    b = annotate_action_dict(
        {"type": "run_sql", "id": "s2", "sql": "DELETE FROM symbol WHERE id='x'"}
    )
    assert b["needs_confirm"] is True
    assert b["read_only"] is False

    c = annotate_action_dict(
        {"type": "run_cypher", "id": "c1", "query": "MATCH (n) RETURN count(n)"}
    )
    assert c["needs_confirm"] is False

    d = annotate_action_dict(
        {"type": "open_browser", "id": "b1", "url": "https://example.com"}
    )
    assert needs_confirm("open_browser") is True
    assert d["needs_confirm"] is True
    assert "example.com" in d["ui_preview"]

    e = annotate_action_dict(
        {"type": "milvus_mutate", "id": "m1", "op": "delete", "ids": ["a"]}
    )
    assert e["needs_confirm"] is True

    f = annotate_action_dict(
        {"type": "milvus_search", "id": "m2", "filter": 'project_id == "p"'}
    )
    assert f["needs_confirm"] is False
    assert f["read_only"] is True


def test_smoke_helpers_removed():
    """Success/smoke verdicts are LLM-owned; dead helpers must stay gone."""
    import app.agent.remote_loop as rl

    assert not hasattr(rl, "_goal_wants_smoke")
    assert not hasattr(rl, "_mk_smoke_action")
    assert not hasattr(rl, "_matches_success")
    assert not hasattr(rl, "_extract_app_port")


def test_reject_sys_cleanup_without_scan():
    from app.agent.remote_loop import (
        RemoteRun,
        _mk_action,
        _validate_sys_cleanup,
    )

    run = RemoteRun(id="r1", goal="清理C盘")
    err = _validate_sys_cleanup(run, _mk_action("sys_cleanup", ids=["user_temp"]))
    assert err and "sys_scan_heavy" in err

    err2 = _validate_sys_cleanup(run, _mk_action("sys_cleanup"))
    assert err2 and "ids" in err2


def test_filter_actions_respects_allow_shell_git():
    from app.agent.remote_loop import RemoteRun, _filter_actions, _mk_action

    run = RemoteRun(
        id="r1",
        goal="test",
        allow_shell=False,
        allow_git=False,
    )
    acts = _filter_actions(
        run,
        [
            _mk_action("run_shell", command="echo 1"),
            _mk_action("git_status"),
            _mk_action("maven_compile"),
        ],
    )
    types = {a.type for a in acts}
    assert "run_shell" not in types
    assert "git_status" not in types
    assert "maven_compile" in types


def test_cleanup_preview_includes_path():
    from app.agent.action_policy import annotate_action_dict
    import json

    catalog = [
        {
            "id": "user_temp",
            "title": "用户临时目录 (Temp)",
            "path": r"C:\Users\Administrator\AppData\Local\Temp",
            "size_human": "1.2 GB",
        }
    ]
    a = annotate_action_dict(
        {
            "type": "sys_cleanup",
            "id": "c1",
            "ids": ["user_temp"],
            "catalog_json": json.dumps(catalog, ensure_ascii=False),
        }
    )
    assert "C:\\Users\\Administrator\\AppData\\Local\\Temp" in a["ui_preview"]
    assert "路径" in a["ui_preview"]


def test_options_from_scan_include_path():
    from app.agent.remote_loop import _options_from_scan
    from app.agent.remote_protocol import ActionResult
    import json

    payload = {
        "cleanup_candidates": [
            {
                "id": "user_temp",
                "label": "用户临时目录 (Temp)",
                "path": r"C:\Users\x\AppData\Local\Temp",
                "size_human": "900 MB",
                "safe": True,
                "kind": "temp",
            }
        ]
    }
    opts = _options_from_scan(
        ActionResult(
            action_id="s1",
            type="sys_scan_heavy",
            ok=True,
            content=json.dumps(payload),
            meta=payload,
        )
    )
    assert opts and opts[0]["path"].endswith("Temp")
    assert "路径" in opts[0]["summary"]


def test_route_prompt_mentions_git_shell():
    from app.agent.remote_loop import SYSTEM_PROMPT

    assert "run_shell" in SYSTEM_PROMPT
    assert "git_merge" in SYSTEM_PROMPT
    assert "open_browser" in SYSTEM_PROMPT
    assert "run_cypher" in SYSTEM_PROMPT
    assert "milvus_mutate" in SYSTEM_PROMPT
