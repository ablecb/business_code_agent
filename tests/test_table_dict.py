"""Tests for table dictionary + table-relation goldens (Item 2)."""
from __future__ import annotations

import os
from pathlib import Path

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.llm.factory import get_embedding_provider, get_llm_provider
from app.models.orm import Base, GoldenQAORM, TableDictORM

_engine = create_engine("sqlite://")
Base.metadata.create_all(_engine, tables=[TableDictORM.__table__, GoldenQAORM.__table__])
_Session = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)


@contextmanager
def _fake_session():
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    with _Session() as s:
        s.query(TableDictORM).delete()
        s.query(GoldenQAORM).delete()
        s.commit()
    monkeypatch.setattr("app.analysis.table_dict.get_session", _fake_session)
    monkeypatch.setattr("app.analysis.golden_qa.get_session", _fake_session)
    yield


def _add_table(tid, table, cn, fields, entity="E"):
    with _Session() as s:
        s.add(
            TableDictORM(
                id=tid,
                project_id="p1",
                table_name=table,
                entity_class=entity,
                name_cn=cn,
                file_path=f"{table}.java",
                fields_json=fields,
            )
        )
        s.commit()


# ---------------------------------------------------------------------------
# 实体文件解析（纯文件，不依赖 DB）
# ---------------------------------------------------------------------------


def test_scan_entity_files_parses_table_and_fields(tmp_path):
    from app.analysis.table_dict import scan_entity_files

    src = tmp_path / "entity" / "CouponInfo.java"
    src.parent.mkdir(parents=True)
    src.write_text(
        """package com.x.entity;

import lombok.Data;

@Data
@ApiModel("优惠券表实体类")
@TableName(value = "demo_b_coupon_info", autoResultMap = true)
public class CouponInfo extends CommonEntity {
    /** 主键id */
    @TableField(updateStrategy = FieldStrategy.NEVER)
    private Long id;

    /** 放款id */
    @ApiModelProperty("放款id")
    private String orderId;

    @ApiModelProperty("券ID")
    private String couponId;
}
""",
        encoding="utf-8",
    )
    ents = scan_entity_files(tmp_path)
    assert len(ents) == 1
    e = ents[0]
    assert e["table_name"] == "demo_b_coupon_info"
    assert e["name_cn"] == "优惠券表"
    names = {f["name"]: f["comment"] for f in e["fields"]}
    assert names["orderId"] == "放款id"
    assert names["couponId"] == "券ID"


# ---------------------------------------------------------------------------
# 关联推导
# ---------------------------------------------------------------------------


def test_derive_relations_same_key_and_priority():
    from app.analysis.table_dict import derive_table_relations

    _add_table(1, "demo_b_withdrawals", "放款表", [{"name": "orderId", "comment": "订单号"}, {"name": "productCode", "comment": "产品代码"}])
    _add_table(2, "demo_b_coupon_info", "优惠券表", [{"name": "orderId", "comment": "放款id"}])
    _add_table(3, "b_demo_payment_info", "还款表", [{"name": "orderId", "comment": "还款订单号"}])
    rels = derive_table_relations("p1")
    pairs = {frozenset([r["table1"], r["table2"]]) | {r["key"]} for r in rels}
    assert frozenset(["demo_b_withdrawals", "demo_b_coupon_info", "orderId"]) in pairs
    assert frozenset(["b_demo_payment_info", "demo_b_coupon_info", "orderId"]) in pairs
    # 每对表最多 2 个键
    wc = [r for r in rels if {r["table1"], r["table2"]} == {"demo_b_withdrawals", "b_demo_payment_info"}]
    assert len(wc) <= 2


def test_relation_questions_and_answer():
    from app.analysis.table_dict import _relation_answer, _relation_questions

    r = {
        "table1": "demo_b_withdrawals", "cn1": "放款表", "entity1": "Withdrawals", "file1": "Withdrawals.java",
        "table2": "demo_b_coupon_info", "cn2": "优惠券表", "entity2": "CouponInfo", "file2": "CouponInfo.java",
        "key": "orderId", "comment1": "订单号", "comment2": "放款id",
    }
    qs = _relation_questions(r)
    assert "放款表和优惠券表怎么关联的" in qs
    assert "放款表和优惠券表通过什么字段关联" in qs
    ans = _relation_answer(r)
    assert "demo_b_withdrawals.orderId = demo_b_coupon_info.orderId" in ans
    assert "放款id" in ans


def test_key_priority():
    from app.analysis.table_dict import _key_priority

    assert _key_priority("orderId", "订单号") == 0
    assert _key_priority("someKey", "交易流水号") == 1
    assert _key_priority("someKey", "") == 2


# ---------------------------------------------------------------------------
# 黄金问答生成（去重先到先得 + 检索排序表名奖励）
# ---------------------------------------------------------------------------


def test_build_goldens_first_wins_on_duplicate_question():
    """同一问句被强/弱两个关联键生成时，先到（强键）者胜。"""
    from app.analysis.table_dict import build_table_relation_goldens
    from app.analysis.golden_qa import _tokens, _table_name_overlap

    _add_table(1, "demo_b_withdrawals", "放款表", [{"name": "orderId", "comment": "订单号"}])
    _add_table(2, "b_demo_payment_info", "还款表", [{"name": "orderId", "comment": "订单号"}])
    st = build_table_relation_goldens("p1", max_relations=10)
    assert st["created"] >= 1
    # 检索「还款表和放款表怎么关联的」应命中 orderId 答案
    from app.analysis.golden_qa import search_golden_qa

    hits = search_golden_qa("p1", "还款表和放款表怎么关联的", top_k=3)
    assert hits
    assert "orderId" in hits[0]["answer"]


def test_table_name_overlap_ranks_correct_relation():
    """「放款表…」应优先命中含「放款表」的黄金，而非同构的「还款表」行。"""
    from app.analysis.golden_qa import search_golden_qa
    from app.analysis.table_dict import build_table_relation_goldens

    _add_table(1, "demo_b_withdrawals", "放款表", [{"name": "orderId", "comment": "订单号"}])
    _add_table(2, "demo_b_coupon_info", "优惠券表", [{"name": "orderId", "comment": "放款id"}])
    _add_table(3, "b_demo_payment_info", "还款表", [{"name": "orderId", "comment": "订单号"}])
    build_table_relation_goldens("p1", max_relations=10)
    hits = search_golden_qa("p1", "放款表和放款优惠券表怎么关联的", top_k=5)
    assert hits
    assert "demo_b_withdrawals" in hits[0]["answer"], hits[0]["question"]
    # 表名同现奖励函数本身
    from app.analysis.golden_qa import _table_name_overlap

    assert _table_name_overlap(
        "放款表和放款优惠券表怎么关联的", "放款表和优惠券表怎么关联的"
    ) > _table_name_overlap("放款表和放款优惠券表怎么关联的", "还款表和优惠券表怎么关联的")
