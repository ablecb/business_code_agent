"""Shallow CST tests: tree-sitter Java parsing of variables & call resolution.

Level-1 unit tests for the CST layer (app.analysis.global_ast):
  - extract_method_locals: how variables get their types (param / new / cast /
    local_decl / foreach / catch / assignment re-fill)
  - resolve_call_detailed: receiver resolution via field_map / locals / this /
    static type, and skip rules (getters / setters / util noise)
  - _infer_rhs_type: RHS type inference (new T() / (Cast) / var)
  - build_global_ast: end-to-end single-project parse with call graph
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.global_ast import (
    _infer_rhs_type,
    _parse_java_source,
    _parser,
    _walk,
    build_global_ast,
    extract_method_locals,
    find_method_node,
    resolve_call_detailed,
)

pytestmark = pytest.mark.skipif(
    not hasattr(__import__("app.analysis.global_ast", fromlist=["_parser"]), "_parser"),
    reason="tree-sitter not installed",
)

_SIMPLE = b"""
package com.demo;

import java.util.List;

@RestController
public class LoanController {
    private LoanService loanService;
    private final String sceneCode = "SC01";

    @PostMapping("/loan")
    public Result apply(@RequestBody LoanReq req) {
        if (req.getAmount() == null) {
            throw new BizException("amount required");
        }
        String scene = req.getSceneCode();
        LoanApplyVO vo = new LoanApplyVO();
        vo.setAmount(req.getAmount());
        vo.setScene(scene);
        LoanDetail detail = loanService.loan(vo, req.getMemberId());
        return Result.ok(detail);
    }

    public void helper() {
        for (String tag : tags) {
            System.out.println(tag);
        }
        try {
            Thread.sleep(10L);
        } catch (InterruptedException e) {
            logger.error("boom", e);
        }
    }
}
"""

_VAR_CHANGE = b"""
package com.demo;

public class FlowService {
    private Repo repo;

    public void change() {
        Request req = new Request();
        String kind = req.getKind();       // local_decl from getter
        kind = "FIXED";                    // reassignment: first-decl wins
        Object raw = parse(kind);
        Money money = (Money) raw;         // cast
        var list = repo.query(kind);        // var + RHS unknown
        for (Item item : list) {
            item.setFlag(true);
        }
        Response resp = build(kind, money, list);
        send(resp);
    }
}
"""


def _parse_to_tree(source: bytes):
    return _parser().parse(source).root_node


def _method_node(source: bytes, name: str):
    root = _parse_to_tree(source)
    return find_method_node(source, root, method_name=name)


def test_parser_available():
    from app.analysis.global_ast import _TS_IMPORT_ERROR

    assert _TS_IMPORT_ERROR is None


def test_extract_locals_params_and_new():
    src = _SIMPLE
    node = _method_node(src, "apply")
    assert node is not None
    locals_map = extract_method_locals(src, node)
    names = {k: (v.type_name, v.how) for k, v in locals_map.items()}
    # formal param
    assert names["req"][0] == "LoanReq"
    assert names["req"][1] == "param"
    # local declared with explicit type + new RHS → how == "new"
    assert names["vo"][0] == "LoanApplyVO"
    assert names["vo"][1] == "new"
    # local declared with explicit type, non-new RHS → local_decl
    assert names["scene"][0] == "String"
    assert names["scene"][1] == "local_decl"
    assert names["detail"][0] == "LoanDetail"
    assert names["detail"][1] == "local_decl"


def test_extract_locals_foreach_and_catch():
    src = _SIMPLE
    node = _method_node(src, "helper")
    assert node is not None
    locals_map = extract_method_locals(src, node)
    names = {k: (v.type_name, v.how) for k, v in locals_map.items()}
    # enhanced for → foreach
    assert names["tag"][0] == "String"
    assert names["tag"][1] == "foreach"
    # catch param → param (type from catch_type)
    assert names["e"][0] == "InterruptedException"
    assert names["e"][1] == "param"


def test_extract_locals_cast_and_var_and_reassign():
    """Variable changes: first declaration wins; cast / var / assignment fills."""
    src = _VAR_CHANGE
    node = _method_node(src, "change")
    assert node is not None
    locals_map = extract_method_locals(src, node)
    names = {k: (v.type_name, v.how) for k, v in locals_map.items()}
    assert names["req"][0] == "Request"
    assert names["req"][1] == "new"
    # reassignment `kind = "FIXED"` must NOT change the declared type
    assert names["kind"][0] == "String"
    assert names["kind"][1] == "local_decl"
    # cast: `Money money = (Money) raw;` → cast
    assert names["money"][0] == "Money"
    assert names["money"][1] == "cast"
    # var with unknown RHS → type_name None, how unknown (no crash)
    assert names["list"][0] is None or names["list"][1] in {"unknown", "local_decl"}
    assert names["resp"][0] == "Response"


def test_infer_rhs_type_new():
    src = b"class T { void m() { LoanApplyVO vo = new LoanApplyVO(); } }"
    root = _parse_to_tree(src)
    declarator = next(
        c
        for c in next(
            n for n in _walk(root) if n.type == "local_variable_declaration"
        ).children
        if c.type == "variable_declarator"
    )
    # RHS is the last non-= child (skip identifier + "=")
    rhs = [c for c in declarator.children if c.type != "="][-1]
    tname, how = _infer_rhs_type(src, rhs)
    assert tname == "LoanApplyVO"
    assert how == "new"


def test_infer_rhs_type_cast():
    src = b"class T { void m() { Money money = (Money) raw; } }"
    root = _parse_to_tree(src)
    declarator = next(
        c
        for c in next(
            n for n in _walk(root) if n.type == "local_variable_declaration"
        ).children
        if c.type == "variable_declarator"
    )
    rhs = [c for c in declarator.children if c.type != "="][-1]
    tname, how = _infer_rhs_type(src, rhs)
    assert tname == "Money"
    assert how == "cast"


def test_find_method_node_by_line():
    src = _SIMPLE
    root = _parse_to_tree(src)
    # find "apply" by line number (1-based in source bytes)
    node = find_method_node(src, root, method_name="apply", start_line=16)
    assert node is not None
    assert node.type == "method_declaration"


def test_find_method_node_missing():
    src = _SIMPLE
    root = _parse_to_tree(src)
    assert find_method_node(src, root, method_name="nope") is None


def test_resolve_call_field_map():
    """this.field.method → resolved via field_map."""
    d = resolve_call_detailed(
        "loanService.loan",
        field_map={"loanService": "LoanService"},
        locals_map={},
        by_simple={"LoanServiceImpl": ["com.demo.LoanServiceImpl"]},
        current_class="LoanController",
        declared_methods=set(),
    )
    assert d["type_name"] == "LoanService"
    assert d["how"] == "field_map"
    assert d["resolved"] == "LoanServiceImpl#loan"
    assert d["status"] == "ok_local"


def test_resolve_call_local_decl():
    d = resolve_call_detailed(
        "vo.setAmount",
        field_map={},
        locals_map={"vo": type("S", (), {"type_name": "LoanApplyVO", "how": "new"})()},
        by_simple={"LoanApplyVO": ["com.demo.LoanApplyVO"]},
        current_class="LoanController",
        declared_methods=set(),
    )
    # setter is a skip candidate but receiver type is still attached
    assert d["skip_reason"] == "skipped_setter"
    assert d["type_name"] == "LoanApplyVO"
    assert d["how"] == "new"


def test_resolve_call_this_bare():
    d = resolve_call_detailed(
        "helper",
        field_map={},
        locals_map={},
        by_simple={"LoanController": ["com.demo.LoanController"]},
        current_class="LoanController",
        declared_methods={"helper"},
    )
    assert d["type_name"] == "LoanController"
    assert d["how"] == "this"
    assert d["status"] == "ok_local"
    assert d["resolved"] == "LoanController#helper"


def test_resolve_call_unknown_receiver():
    d = resolve_call_detailed(
        "mystery.doSomething",
        field_map={},
        locals_map={},
        by_simple={},
        current_class="LoanController",
        declared_methods=set(),
    )
    assert d["status"] == "skipped"
    assert d["skip_reason"] == "unknown_receiver"


def test_resolve_call_static_type():
    d = resolve_call_detailed(
        "ExcelUtil.export",
        field_map={},
        locals_map={},
        by_simple={"ExcelUtil": ["com.demo.ExcelUtil"]},
        current_class="LoanController",
        declared_methods=set(),
    )
    assert d["type_name"] == "ExcelUtil"
    assert d["how"] == "static_type"


def test_resolve_call_getter_skip():
    d = resolve_call_detailed(
        "req.getAmount",
        field_map={},
        locals_map={"req": type("S", (), {"type_name": "LoanReq", "how": "param"})()},
        by_simple={"LoanReq": ["com.demo.LoanReq"]},
        current_class="LoanController",
        declared_methods=set(),
    )
    assert d["skip_reason"] == "skipped_getter"
    assert d["type_name"] == "LoanReq"


def test_parse_java_source_basic():
    root = _parse_to_tree(_SIMPLE)
    types = _parse_java_source(Path("proj"), Path("proj/LoanController.java"), _SIMPLE, root)
    assert types
    t = types[0]
    assert t.simple_name == "LoanController"
    assert t.package == "com.demo"
    assert t.fqcn == "com.demo.LoanController"
    assert t.symbol_kind == "Controller"  # @RestController
    assert t.field_map.get("loanService") == "LoanService"
    names = {m.method_name for m in t.methods}
    assert "apply" in names
    assert "helper" in names
    apply = next(m for m in t.methods if m.method_name == "apply")
    # _formal_params returns type names only (no param names)
    assert apply.signature == "apply(LoanReq)"
    assert any("loanService.loan" in c for c in apply.calls)


def test_build_global_ast_small_project(tmp_path: Path):
    src = tmp_path / "src" / "main" / "java" / "com" / "demo"
    src.mkdir(parents=True)
    (src / "LoanController.java").write_bytes(_SIMPLE)
    (src / "LoanService.java").write_bytes(
        b"""
package com.demo;
public class LoanService {
    public LoanDetail loan(LoanApplyVO vo, String memberId) {
        return new LoanDetail();
    }
}
"""
    )
    ast = build_global_ast("proj", tmp_path)
    assert ast.java_file_count == 2
    assert "com.demo.LoanController" in ast.types
    assert "com.demo.LoanService" in ast.types
    # call edge from Controller#apply → LoanService#loan
    assert ast.edges
    edge = next(
        (e for e in ast.edges if e.raw_call == "loanService.loan"),
        None,
    )
    assert edge is not None
    assert edge.caller == "LoanController#apply"
    assert edge.callee == "LoanService#loan"
    assert edge.status == "ok_local"
    assert edge.resolve_how == "field_map"


def test_build_global_ast_missing_root_raises():
    from app.parser.scanner import scan_java_files

    with pytest.raises(FileNotFoundError):
        scan_java_files(Path("does-not-exist-xyz"))
