"""Unit tests for connector service helpers (mocked engines)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_execute_mysql_rejects_multi_statement():
    from app.services.connectors import execute_mysql

    r = execute_mysql("SELECT 1; DROP TABLE x")
    assert r["ok"] is False
    assert "单条" in (r.get("error") or "")


def test_execute_mysql_settings_select():
    from app.services.connectors import execute_mysql

    mock_result = MagicMock()
    mock_result.returns_rows = True
    mock_result.keys.return_value = ["id", "name"]
    mock_result.fetchmany.return_value = [("1", "a"), ("2", "b")]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    with patch("app.db.mysql.get_engine", return_value=mock_engine):
        r = execute_mysql("SELECT id, name FROM t LIMIT 2")
    assert r["ok"] is True
    assert r["row_count"] == 2
    assert r["rows"][0]["id"] == "1"
    assert "settings" in str(r.get("source"))
    assert "row_count" in r["stdout"]


def test_execute_mysql_temp_dml():
    from app.services.connectors import execute_mysql

    mock_result = MagicMock()
    mock_result.returns_rows = False
    mock_result.rowcount = 3

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    with patch("app.services.connectors.create_engine", return_value=mock_engine):
        r = execute_mysql(
            "UPDATE t SET x=1 WHERE id=2",
            mysql={
                "host": "127.0.0.1",
                "port": 3306,
                "user": "u",
                "password": "p",
                "database": "biz",
            },
        )
    assert r["ok"] is True
    assert r.get("affected_rows") == 3
    assert r.get("source") == "temp_mysql"
    mock_engine.dispose.assert_called()


def test_execute_mysql_truncates_rows():
    from app.services.connectors import execute_mysql

    mock_result = MagicMock()
    mock_result.returns_rows = True
    mock_result.keys.return_value = ["id"]
    mock_result.fetchmany.return_value = [(i,) for i in range(6)]

    mock_conn = MagicMock()
    mock_conn.__enter__.return_value = mock_conn
    mock_conn.__exit__.return_value = None
    mock_conn.execute.return_value = mock_result

    mock_engine = MagicMock()
    mock_engine.connect.return_value = mock_conn

    with patch("app.db.mysql.get_engine", return_value=mock_engine):
        r = execute_mysql("SELECT id FROM t", max_rows=5)
    assert r["ok"] is True
    assert any("_note" in row for row in r["rows"])


def test_execute_cypher_ok():
    from app.services.connectors import execute_cypher

    neo = MagicMock()
    neo.run.return_value = [{"n": 1}, {"n": 2}]
    with patch("app.db.neo4j_client.get_neo4j", return_value=neo):
        r = execute_cypher("MATCH (n) RETURN count(n) AS n")
    assert r["ok"] is True
    assert r["row_count"] == 2


def test_execute_milvus_search_memory():
    from app.services.connectors import execute_milvus

    store = MagicMock()
    store._use_memory = True
    store._client = None
    store.collection_name = "method_embeddings"
    store._memory = [
        {
            "id": "1",
            "project_id": "p1",
            "symbol_id": "s1",
            "fqcn": "A#m",
            "module_name": "m",
            "tags": "",
            "text": "hello",
        },
        {
            "id": "2",
            "project_id": "p2",
            "symbol_id": "s2",
            "fqcn": "B#n",
            "module_name": "m",
            "tags": "",
            "text": "world",
        },
    ]
    with patch("app.db.milvus_client.get_milvus", return_value=store):
        r = execute_milvus(op="search", filter='project_id == "p1"', limit=10)
    assert r["ok"] is True
    assert r["row_count"] == 1
    assert r["rows"][0]["project_id"] == "p1"


def test_connectors_api_mysql_route():
    from fastapi.testclient import TestClient

    from app.main import app

    with patch(
        "app.api.connectors.execute_mysql",
        return_value={
            "ok": True,
            "error": None,
            "row_count": 1,
            "rows": [{"x": 1}],
            "stdout": '{"row_count":1}',
            "source": "settings",
        },
    ):
        client = TestClient(app)
        resp = client.post("/api/v1/connectors/mysql", json={"sql": "SELECT 1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["row_count"] == 1
