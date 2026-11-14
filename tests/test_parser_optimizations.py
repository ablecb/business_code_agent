"""Regression tests for parser enrichments (parser data quality).

These pin the解析层优化 so future changes cannot regress:
  1. generic type args (List<OrderItem> → ["OrderItem"])
  2. var + method-return inference (var x = svc.find() → return type)
  3. lambda parameter typing from receiver generics (incl. fluent chains)
  4. method reference extraction (OrderItem::getCode → OrderItem.getCode)
  5. reassignment upgrades (Object target; target = repo.get(n) → concrete type)
"""
from __future__ import annotations

import pytest

from app.analysis.global_ast import (
    _method_calls_in_body,
    _parser,
    _type_node_args,
    _type_node_simple,
    build_global_ast,
    extract_method_locals,
    find_method_node,
)

pytestmark = pytest.mark.skipif(
    not hasattr(__import__("app.analysis.global_ast", fromlist=["_parser"]), "_parser"),
    reason="tree-sitter not installed",
)

_FLOW_SRC = r'''
package com.demo;

import java.util.List;

public class FlowService {
    private Repo repo;
    private List<OrderItem> items;

    public void run(List<OrderItem> rawItems, String name) {
        List<OrderItem> items2 = rawItems;
        Order o = buildFrom(name);
        var count = repo.count(o);
        items.forEach(item -> item.setStatus(1));
        items2.stream().filter(x -> x.getAmount() > 0)
              .map(OrderItem::getCode)
              .forEach(code -> log(code));
        Object target = o;
        target = repo.get(name);
    }

    public int count(Order o) { return 0; }
    public Order get(String name) { return null; }
    private Order buildFrom(String name) { return new Order(); }
    private void log(Object x) { }
}
'''


def _method(source: bytes, name: str):
    root = _parser().parse(source).root_node
    return find_method_node(source, root, method_name=name)


# ---------- 1. generic type args ----------

def test_type_node_args_list_of_item():
    src = b"class T { private List<OrderItem> items; }"
    root = _parser().parse(src).root_node
    for n in root.children:
        for c in n.children:  # class_body
            for f in getattr(c, "children", []):
                for x in getattr(f, "children", []):
                    if x.type == "generic_type":
                        assert _type_node_args(src, x) == ["OrderItem"]
                        assert _type_node_simple(src, x) == "List"
                        return
    pytest.fail("generic_type node not found")


def test_type_node_args_multiple():
    src = b"class T { private Map<String, OrderItem> m; }"
    root = _parser().parse(src).root_node
    for n in root.children:
        for c in n.children:
            for f in getattr(c, "children", []):
                for x in getattr(f, "children", []):
                    if x.type == "generic_type":
                        assert _type_node_args(src, x) == ["String", "OrderItem"]
                        return
    pytest.fail("generic_type node not found")


def test_type_node_args_non_generic_empty():
    src = b"class T { private OrderItem o; }"
    root = _parser().parse(src).root_node
    for n in root.children:
        for c in n.children:
            for f in getattr(c, "children", []):
                for x in getattr(f, "children", []):
                    if x.type == "type_identifier":
                        assert _type_node_args(src, x) == []
                        return
    pytest.fail("type_identifier not found")


# ---------- 2. var + method-return inference ----------

def test_var_typed_from_method_return():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    lm = extract_method_locals(
        src, node, returns_by_name={"count": "int", "get": "Order"}
    )
    assert lm["count"].type_name == "int"
    assert lm["count"].how == "return_type"


def test_explicit_decl_keeps_declared_type_even_with_returns():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    lm = extract_method_locals(
        src, node, returns_by_name={"count": "int", "get": "Order"}
    )
    # o declared `Order o = buildFrom(...)` — explicit type wins, but the
    # enrichment should not downgrade it even if returns map has buildFrom.
    assert lm["o"].type_name == "Order"
    assert lm["o"].how in {"local_decl", "new"}


# ---------- 3. lambda parameter typing ----------

def test_lambda_param_from_receiver_generics():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    lm = extract_method_locals(
        src,
        node,
        generic_params={"items": ["OrderItem"], "rawItems": ["OrderItem"]},
    )
    assert lm["item"].type_name == "OrderItem"
    assert lm["item"].how == "lambda"


def test_lambda_param_from_fluent_chain_receiver():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    lm = extract_method_locals(
        src,
        node,
        generic_params={"items2": ["OrderItem"], "items": ["OrderItem"]},
    )
    # x comes from items2.stream().filter(x -> …) — root receiver items2
    assert lm["x"].type_name == "OrderItem"
    assert lm["x"].how == "lambda"
    # code comes from …map(OrderItem::getCode).forEach(code -> …) — chain root
    assert lm["code"].type_name == "OrderItem"
    assert lm["code"].how == "lambda"


def test_lambda_param_without_generics_unknown_no_crash():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    lm = extract_method_locals(src, node, generic_params={})
    assert lm["item"].type_name is None
    assert lm["item"].how == "lambda"


# ---------- 4. method reference extraction ----------

def test_method_reference_extracted():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    calls = _method_calls_in_body(src, node)
    assert "OrderItem.getCode" in calls


def test_method_reference_this_filtered():
    src = b"class T { void m() { list.forEach(this::handle); } }"
    node = _method(src, "m")
    calls = _method_calls_in_body(src, node)
    assert not any("this." in c for c in calls)


# ---------- 5. reassignment upgrades ----------

def test_reassignment_upgrades_object_to_concrete():
    src = _FLOW_SRC.encode("utf-8")
    node = _method(src, "run")
    lm = extract_method_locals(
        src, node, returns_by_name={"count": "int", "get": "Order"}
    )
    assert lm["target"].type_name == "Order"
    assert lm["target"].reassigned_to == "Order"


def test_reassignment_does_not_downgrade_concrete():
    src = """
class T {
    void m(Repo repo) {
        Order o = new Order();
        o = repo.getOther();  // different type — first decl wins
    }
}
"""
    src_b = src.encode("utf-8")
    node = _method(src_b, "m")
    lm = extract_method_locals(src_b, node, returns_by_name={"getOther": "Other"})
    assert lm["o"].type_name == "Order"
    # reassigned_to records the later type but type_name stays first-decl
    assert lm["o"].reassigned_to == "Other"


# ---------- end-to-end: build_global_ast carries the tables ----------

def test_build_global_ast_populates_returns_and_generics(tmp_path):
    src = tmp_path / "src" / "main" / "java" / "com" / "demo"
    src.mkdir(parents=True)
    (src / "FlowService.java").write_text(
        _FLOW_SRC + "\n",
        encoding="utf-8",
    )
    ast = build_global_ast("proj", tmp_path)
    returns = ast.returns_for()
    assert returns.get("count") == "int"
    assert returns.get("get") == "Order"
    assert returns.get("buildFrom") == "Order"
    # generic receivers discovered from field declarations
    assert ast.generic_args_for("FlowService") == ["OrderItem"]


def test_var_resolve_payload_includes_type_args_and_reassign(monkeypatch, tmp_path):
    """The var-resolve bag now carries generic args + reassignment info."""
    import app.analysis.var_resolve as vr

    src = tmp_path / "src" / "main" / "java" / "com" / "demo"
    src.mkdir(parents=True)
    (src / "FlowService.java").write_text(
        _FLOW_SRC + "\n",
        encoding="utf-8",
    )
    from app.analysis.global_ast import build_global_ast

    ast = build_global_ast("proj", tmp_path)
    payload = vr.build_method_var_resolve_payload(ast, "FlowService#run")
    assert not payload.get("error")
    locals_by_name = {x["name"]: x for x in payload["locals"]}
    assert locals_by_name["items2"]["type_args"] == ["OrderItem"]
    assert locals_by_name["item"]["type_name"] == "OrderItem"
    assert locals_by_name["target"]["reassigned_to"] == "Order"
