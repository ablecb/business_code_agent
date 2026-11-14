"""Deterministic Mapper-XML SQL metadata + field-evidence integration.

Level-8 tests: SQL column/table/param extraction from MyBatis XML and its
consumption by field-evidence as hard (non-LLM) change evidence.

  1. parse_sql_meta: INSERT/UPDATE/SELECT tables, write/read cols, params
  2. Mapper XML → AstType.sql_table_fields / method_sql_meta (via global AST)
  3. field_evidence._load_mapper_sql_changes → table_field_changes rows
  4. collect_field_evidence_pack includes mapper_xml_sql evidence
  5. render distinguishes AI vs deterministic SQL sources
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.sql_meta import parse_sql_meta, sql_meta_to_table_changes

XML_MAPPER = r'''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.demo.ApplyMapper">
  <insert id="insertApply">
    INSERT INTO demo_b_apply (id, order_id, amount)
    VALUES (#{id}, #{orderId}, #{amount})
  </insert>
  <update id="updateStatus">
    UPDATE demo_b_apply
    SET status = #{status, jdbcType=VARCHAR}, fail_reason = #{failReason}
    WHERE order_id = #{orderId}
  </update>
  <select id="selectByOrder" resultType="com.demo.Apply">
    SELECT id, order_id, amount FROM demo_b_apply
    WHERE status = #{status} AND order_id = #{orderId}
  </select>
</mapper>
'''


def _write_mapper(tmp_path: Path) -> Path:
    xml = tmp_path / "src" / "main" / "resources" / "mapper"
    xml.mkdir(parents=True)
    fp = xml / "ApplyMapper.xml"
    fp.write_text(XML_MAPPER, encoding="utf-8")
    return fp


@pytest.fixture
def _tmp_project(monkeypatch, tmp_path):
    """Point the lru_cached settings at a temp project via env (cache-safe)."""
    _write_mapper(tmp_path)
    monkeypatch.setenv("PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("PROJECT_ID", "demo")
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


# ---------- 1. parse_sql_meta ----------

def test_parse_insert_meta():
    meta = parse_sql_meta(
        "INSERT INTO demo_b_apply (id, order_id, amount) VALUES (#{id}, #{orderId}, #{amount})",
        stmt_tag="insert",
    )
    assert meta["op"] == "insert"
    assert meta["tables"] == ["demo_b_apply"]
    assert meta["write_cols"] == ["id", "order_id", "amount"]
    assert "orderId" in meta["params"]
    assert meta["read_cols"] == []


def test_parse_update_meta_with_jdbc_type():
    meta = parse_sql_meta(
        "UPDATE demo_b_apply SET status = #{status, jdbcType=VARCHAR}, "
        "fail_reason = #{failReason} WHERE order_id = #{orderId}",
        stmt_tag="update",
    )
    assert meta["op"] == "update"
    assert meta["tables"] == ["demo_b_apply"]
    assert meta["write_cols"] == ["status", "fail_reason"]
    assert meta["read_cols"] == ["order_id"]
    # param ref cleans jdbcType annotation
    assert "status" in meta["params"]
    assert all("jdbcType" not in p for p in meta["params"])


def test_parse_select_meta_projection_and_where():
    meta = parse_sql_meta(
        "SELECT id, order_id, amount FROM demo_b_apply "
        "WHERE status = #{status} AND order_id = #{orderId}",
        stmt_tag="select",
    )
    assert meta["tables"] == ["demo_b_apply"]
    assert set(meta["read_cols"]) >= {"id", "order_id", "amount", "status"}
    assert meta["write_cols"] == []
    assert "orderId" in meta["params"]


def test_parse_join_tables():
    meta = parse_sql_meta(
        "SELECT a.id FROM demo_b_apply a JOIN demo_b_order o ON a.order_id = o.id WHERE o.type = 1",
        stmt_tag="select",
    )
    assert "demo_b_apply" in meta["tables"]
    assert "demo_b_order" in meta["tables"]


def test_sql_meta_to_table_changes_shape():
    meta = parse_sql_meta(
        "UPDATE demo_b_apply SET status = #{s} WHERE id = #{id}", stmt_tag="update"
    )
    rows = sql_meta_to_table_changes(meta, class_method="ApplyMapper#updateStatus")
    assert rows
    row = rows[0]
    assert row["table_or_entity"] == "demo_b_apply"
    assert row["field"] == "status"
    assert row["op"] == "update"
    assert row["source"] == "mapper_xml"
    assert row["when"] == "ApplyMapper#updateStatus Mapper SQL"


# ---------- 2. Mapper XML → AstType ----------

def test_parse_one_mapper_xml_sql_fields(tmp_path):
    from app.analysis.global_ast import _parse_one_mapper_xml

    fp = _write_mapper(tmp_path)
    t, errs = _parse_one_mapper_xml(tmp_path, fp)
    assert t is not None
    assert t.symbol_kind == "MapperXml"
    assert t.sql_table_fields.get("demo_b_apply") == ["id", "order_id", "amount", "status", "fail_reason"]
    meta = t.method_sql_meta.get("ApplyMapper#updateStatus")
    assert meta is not None
    assert meta["write_cols"] == ["status", "fail_reason"]


# ---------- 3/4. field_evidence integration ----------

def test_load_mapper_sql_changes(_tmp_project):
    from app.agent import field_evidence as fe

    out = fe._load_mapper_sql_changes("demo", ["ApplyMapper#updateStatus"])
    assert "ApplyMapper#updateStatus" in out
    rows = out["ApplyMapper#updateStatus"]
    fields = {r["field"] for r in rows}
    assert "status" in fields
    assert "fail_reason" in fields
    assert all(r["source"] == "mapper_xml_sql" for r in rows)
    assert all(r["table_or_entity"] == "demo_b_apply" for r in rows)


def test_collect_field_evidence_includes_mapper_sql(_tmp_project):
    from app.agent import field_evidence as fe

    pack = fe.collect_field_evidence_pack(
        "demo",
        entry="ApplyController#apply",
        spine=["ApplyMapper#updateStatus", "ApplyService#apply"],
        question="放款表 demo_b_apply 的 status 字段改成什么",
    )
    hits = pack.get("tfc_hits") or []
    sql_hits = [h for h in hits if h.get("source") == "mapper_xml_sql"]
    assert sql_hits, hits
    assert any(h["field"] == "status" for h in sql_hits)


def test_render_field_evidence_labels_sql_source(_tmp_project):
    from app.agent import field_evidence as fe

    pack = fe.collect_field_evidence_pack(
        "demo",
        entry="ApplyController#apply",
        spine=["ApplyMapper#updateStatus"],
        question="demo_b_apply 的 status 改成什么",
    )
    ans = fe.render_field_evidence_answer(pack, "demo_b_apply 的 status 改成什么")
    assert "Mapper SQL 确定性证据" in ans
    assert "demo_b_apply.status" in ans
