"""Regression tests for Javadoc/comment business description extraction.

Level-9 tests: comment_meta (Javadoc parsing, business summary) and its wiring
into global_ast (AstMethod.doc_comment) and treesitter_deep_flow
(_heuristic_deep professional summaries).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.comment_meta import (
    business_summary_from_comment,
    clean_comment,
    param_flow_from_javadoc,
    split_javadoc,
)

pytestmark = pytest.mark.skipif(
    not hasattr(__import__("app.analysis.global_ast", fromlist=["_parser"]), "_parser"),
    reason="tree-sitter not installed",
)

_JAVADOC = """/**
 * 查询订单详情并校验商户额度
 * 按订单号查询主表，校验机构限额后返回详情
 * @param orderNo 订单号
 * @return 订单详情
 */
"""


# ---------- comment_meta unit ----------

def test_clean_comment_strips_delimiters():
    assert clean_comment(_JAVADOC) == "查询订单详情并校验商户额度 按订单号查询主表，校验机构限额后返回详情"


def test_split_javadoc_body_and_tags():
    body, tags = split_javadoc(_JAVADOC)
    assert any("查询订单详情" in b for b in body)
    assert "订单号" in tags.get("param", "")
    assert "订单详情" in tags.get("return", "")


def test_business_summary_prefers_body():
    s = business_summary_from_comment(_JAVADOC, fallback="FALLBACK")
    assert "查询订单详情" in s
    assert "FALLBACK" not in s


def test_business_summary_no_body_uses_tag():
    raw = "/** @see Foo#bar */"
    s = business_summary_from_comment(raw, fallback="F")
    assert s  # non-empty fallback or tag-derived


def test_param_flow_from_javadoc():
    params = param_flow_from_javadoc(_JAVADOC)
    assert params
    assert params[0]["name"] == "orderNo"
    assert "订单号" in params[0]["desc"]


# ---------- global_ast wiring ----------

def test_ast_method_carries_doc_comment():
    from app.analysis.global_ast import _parse_java_source, _parser

    src = """
package com.demo;

public class OrderService {
    /**
     * 查询订单详情
     * @param orderNo 订单号
     */
    public Order getOrder(String orderNo) {
        return repo.find(orderNo);
    }

    private void noComment() {
        noop();
    }
}
""".encode("utf-8")
    root = _parser().parse(src).root_node
    types = _parse_java_source(Path("proj"), Path("proj/OrderService.java"), src, root)
    t = types[0]
    by_name = {m.method_name: m for m in t.methods}
    assert "查询订单详情" in by_name["getOrder"].doc_comment
    assert by_name["noComment"].doc_comment == ""


def test_doc_comment_adjacency_requires_immediate_comment():
    """A comment separated by an annotation is still found (gap<=1)."""
    from app.analysis.global_ast import _parse_java_source, _parser

    src = """
package com.demo;

public class Svc {
    /** 业务注释 */
    @Transactional
    public void tx() {
        doIt();
    }
}
""".encode("utf-8")
    root = _parser().parse(src).root_node
    types = _parse_java_source(Path("proj"), Path("proj/Svc.java"), src, root)
    t = types[0]
    m = next(x for x in t.methods if x.method_name == "tx")
    assert "业务注释" in m.doc_comment


# ---------- _heuristic_deep integration ----------

def _fake_type(simple: str = "OrderService", symbol_kind: str = "Service"):
    from types import SimpleNamespace

    return SimpleNamespace(simple_name=simple, symbol_kind=symbol_kind)


def _fake_method(name: str = "getOrder", raw: str = "public Order getOrder() {}", doc: str = ""):
    from types import SimpleNamespace

    return SimpleNamespace(name=name, raw_code=raw, calls=["repo.find(x)"], doc_comment=doc)


def test_heuristic_deep_uses_comment_summary():
    from app.analysis.treesitter_deep_flow import _heuristic_deep

    data = _heuristic_deep(
        _fake_type(),
        _fake_method(doc="/** 查询订单详情并校验额度 */"),
    )
    assert "查询订单详情" in data["summary_business_clear"]
    assert "OrderService#getOrder 业务节点" not in data["summary_business_clear"]
    assert data["business_view"]["capability"] == "查询订单详情并校验额度"


def test_heuristic_deep_no_comment_fallback():
    from app.analysis.treesitter_deep_flow import _heuristic_deep

    data = _heuristic_deep(_fake_type(), _fake_method(doc=""))
    # falls back to class#method placeholder (unchanged behaviour)
    assert data["summary_business_clear"] == "OrderService#getOrder 业务节点"


def test_heuristic_deep_params_from_javadoc():
    from app.analysis.treesitter_deep_flow import _heuristic_deep

    data = _heuristic_deep(
        _fake_type(),
        _fake_method(doc="/** @param orderNo 订单号\n @return 详情 */"),
    )
    params = data["impl_view"]["params"]
    assert any(p.get("name") == "orderNo" for p in params)


# ---------- end-to-end: full project parse ----------

def test_full_parse_extracts_doc_comments(tmp_path):
    from app.analysis.global_ast import build_global_ast

    src = tmp_path / "src" / "main" / "java" / "com" / "demo"
    src.mkdir(parents=True)
    (src / "OrderService.java").write_text(
        """
package com.demo;
public class OrderService {
    /** 查询订单并校验额度 */
    public Order getOrder(String no) { return new Order(); }
}
""",
        encoding="utf-8",
    )
    ast = build_global_ast("proj", tmp_path)
    m = ast.methods.get("OrderService#getOrder")
    assert m is not None
    assert "查询订单并校验额度" in m.doc_comment
