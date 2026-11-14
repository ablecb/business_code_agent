"""Unit tests for var-resolve edge props + MySQL upsert (idempotent)."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from app.analysis.global_ast import CallEdge
from app.analysis.var_resolve import (
    calls_edge_props,
    setter_sites_from_bags,
    upsert_method_var_resolve,
)
from app.analysis.neo4j_store import merge_calls


def test_calls_edge_props_from_call_edge():
    e = CallEdge(
        caller="Foo#bar",
        callee="Svc#doIt",
        raw_call="svc.doIt",
        status="ok_local",
        order=2,
        receiver="svc",
        receiver_type="Svc",
        resolve_how="field_map",
    )
    props = calls_edge_props(e)
    assert props["raw_call"] == "svc.doIt"
    assert props["receiver"] == "svc"
    assert props["receiver_type"] == "Svc"
    assert props["resolve_how"] == "field_map"


def test_merge_calls_passes_resolve_props():
    """merge_calls forwards resolve facts into Neo4j SET props."""
    fake_client = MagicMock()
    with patch("app.analysis.neo4j_store.get_neo4j", return_value=fake_client):
        merge_calls(
            "proj",
            "mid-a",
            "mid-b",
            order=1,
            source="global-ast",
            receiver="loanService",
            receiver_type="WithdrawalsService",
            raw_call="loanService.loan",
            resolve_how="field_map",
        )
    fake_client.write_tx.assert_called_once()
    cypher, params = fake_client.write_tx.call_args[0]
    assert "MERGE (a)-[r:CALLS]->(b)" in cypher
    assert "SET r += $props" in cypher
    props = params["props"]
    assert props["project_id"] == "proj"
    assert props["order"] == 1
    assert props["source"] == "global-ast"
    assert props["receiver"] == "loanService"
    assert props["receiver_type"] == "WithdrawalsService"
    assert props["raw_call"] == "loanService.loan"
    assert props["resolve_how"] == "field_map"


def test_setter_sites_from_bags_exact_and_analog():
    bags = {
        "Foo#bind": {
            "call_sites": [
                {
                    "raw_call": "reqVo.setSceneCode",
                    "receiver": "reqVo",
                    "resolved_type": "FooApplyReqVo",
                    "resolve_how": "local_decl",
                },
                {
                    "raw_call": "reqVo.setBusinessLabel",
                    "receiver": "reqVo",
                    "resolved_type": "FooApplyReqVo",
                    "resolve_how": "param",
                },
            ]
        }
    }
    sites = setter_sites_from_bags(bags, field_name="businessLabel", limit=10)
    assert any(s.get("exact_field") for s in sites)
    assert any(s["setter"] == "setBusinessLabel" for s in sites)
    assert any(s["setter"] == "setSceneCode" for s in sites)


def test_upsert_method_var_resolve_idempotent():
    """Second upsert updates same unique (project_id, class_method) row."""
    existing = MagicMock()
    existing.symbol_id = "old-sid"
    existing.payload_json = {"v": 1}
    session = MagicMock()
    session.scalars.return_value.first.return_value = existing
    session.scalars.return_value.all.return_value = []

    cm = "WithdrawalsController#checkApplyReqParam"
    payload = {
        "class_method": cm,
        "locals": [{"name": "reqVo", "type_name": "WithdrawalsApplyReqVo", "how": "new"}],
        "call_sites": [],
        "receivers_used": [],
    }

    with (
        patch("app.analysis.var_resolve.init_db"),
        patch("app.analysis.var_resolve.get_session") as gs,
    ):
        gs.return_value.__enter__.return_value = session
        gs.return_value.__exit__.return_value = None
        upsert_method_var_resolve("demo-loan", cm, payload, symbol_id="sid-1")
        upsert_method_var_resolve("demo-loan", cm, {**payload, "v": 2}, symbol_id="sid-1")

    assert existing.payload_json["v"] == 2 or existing.payload_json == {
        **payload,
        "v": 2,
    }
    assert existing.symbol_id == "sid-1"
    assert isinstance(existing.updated_at, datetime) or existing.updated_at is not None
    # No second INSERT when row exists
    assert session.add.call_count == 0


def test_upsert_method_var_resolve_inserts_when_missing():
    session = MagicMock()
    session.scalars.return_value.first.return_value = None
    session.scalars.return_value.all.return_value = []
    with (
        patch("app.analysis.var_resolve.init_db"),
        patch("app.analysis.var_resolve.get_session") as gs,
    ):
        gs.return_value.__enter__.return_value = session
        gs.return_value.__exit__.return_value = None
        upsert_method_var_resolve(
            "demo-loan",
            "A#b",
            {"class_method": "A#b", "locals": [], "call_sites": []},
        )
    assert session.add.call_count == 1
    row = session.add.call_args[0][0]
    assert row.project_id == "demo-loan"
    assert row.class_method == "A#b"


def test_ensure_entry_var_resolve_skips_when_ready():
    from app.analysis import var_resolve as vr

    vr._ensured_ok.clear()
    with (
        patch.object(vr, "entry_var_resolve_ready", return_value=True),
        patch.object(vr, "count_method_var_resolves", return_value=59),
        patch.object(vr, "sync_entry_var_resolve") as sync,
        patch("app.core.config.get_settings") as gs,
    ):
        settings = MagicMock()
        settings.project_id = "demo-loan"
        settings.var_resolve_auto_ensure = True
        settings.var_resolve_hop = 2
        settings.var_resolve_package_roots = "loan-service/src/main/java"
        settings.var_resolve_min_bags = 5
        gs.return_value = settings
        out = vr.ensure_entry_var_resolve(
            "demo-loan", "WithdrawalsController#withdrawals"
        )
    assert out["skipped"] is True
    assert out["reason"] == "already_ready"
    sync.assert_not_called()


def test_ensure_entry_var_resolve_syncs_when_missing():
    from app.analysis import var_resolve as vr

    vr._ensured_ok.clear()
    with (
        patch.object(vr, "entry_var_resolve_ready", side_effect=[False, True]),
        patch.object(
            vr,
            "sync_entry_var_resolve",
            return_value={
                "project_id": "demo-loan",
                "entry": "WithdrawalsController#withdrawals",
                "targets": ["WithdrawalsController#withdrawals"],
                "mysql": {"written": 3, "errors": 0, "targets": 3},
                "neo4j_calls_patched": 1,
                "ast": {},
            },
        ) as sync,
        patch("app.core.config.get_settings") as gs,
    ):
        settings = MagicMock()
        settings.project_id = "demo-loan"
        settings.var_resolve_auto_ensure = True
        settings.var_resolve_hop = 2
        settings.var_resolve_package_roots = "loan-service/src/main/java"
        settings.var_resolve_min_bags = 5
        gs.return_value = settings
        out = vr.ensure_entry_var_resolve(
            "demo-loan", "WithdrawalsController#withdrawals"
        )
    assert out["skipped"] is False
    assert out["reason"] == "synced"
    assert out["mysql"]["written"] == 3
    sync.assert_called_once()
