"""Mid-level CST tests: per-method var-resolve bags (variable flow).

Level-2 tests for app.analysis.var_resolve: build the canonical per-method
bag from real Java source — locals, receivers_used, call_sites, and the
stats that downstream change_impact / setter discovery rely on. Also covers
collect_entry_targets + setter_sites_from_bags end to end.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.analysis.global_ast import build_global_ast

pytestmark = pytest.mark.skipif(
    not hasattr(__import__("app.analysis.global_ast", fromlist=["_parser"]), "_parser"),
    reason="tree-sitter not installed",
)

_CTRL = b"""
package com.demo;

@RestController
public class ApplyController {
    private ApplyService applyService;

    @PostMapping("/apply")
    public Result apply(@RequestBody ApplyReq req) {
        if (req.getMemberId() == null) {
            throw new BizException("member required");
        }
        String scene = req.getSceneCode();
        ApplyVO vo = new ApplyVO();
        vo.setScene(scene);
        vo.setMemberId(req.getMemberId());
        ApplyDetail detail = applyService.doApply(vo);
        if (detail == null) {
            return Result.fail("no detail");
        }
        saveLog(scene, detail);
        return Result.ok(detail);
    }

    private void saveLog(String scene, ApplyDetail detail) {
        logger.info("scene={}", scene);
    }
}
"""

_SVC = b"""
package com.demo;

public class ApplyService {
    public ApplyDetail doApply(ApplyVO vo) {
        ApplyDetail detail = new ApplyDetail();
        detail.setAmount(vo.getAmount());
        return detail;
    }
}
"""

_VO = b"""
package com.demo;

public class ApplyVO {
    private String scene;
    private String memberId;
    public String getScene() { return scene; }
    public void setScene(String s) { this.scene = s; }
    public void setMemberId(String m) { this.memberId = m; }
}
"""


@pytest.fixture
def ast(tmp_path: Path):
    src = tmp_path / "src" / "main" / "java" / "com" / "demo"
    src.mkdir(parents=True)
    (src / "ApplyController.java").write_bytes(_CTRL)
    (src / "ApplyService.java").write_bytes(_SVC)
    (src / "ApplyVO.java").write_bytes(_VO)
    return build_global_ast("demo", tmp_path)


def test_ast_has_all_types(ast):
    assert ast.java_file_count == 3
    assert "com.demo.ApplyController" in ast.types
    assert "com.demo.ApplyService" in ast.types
    assert "com.demo.ApplyVO" in ast.types


def test_build_payload_locals_and_receivers(ast):
    from app.analysis.var_resolve import build_method_var_resolve_payload

    payload = build_method_var_resolve_payload(ast, "ApplyController#apply")
    assert payload["class_method"] == "ApplyController#apply"
    assert not payload.get("error")
    locals_by_name = {x["name"]: x for x in payload["locals"]}
    assert locals_by_name["req"]["type_name"] == "ApplyReq"
    assert locals_by_name["req"]["how"] == "param"
    assert locals_by_name["vo"]["type_name"] == "ApplyVO"
    assert locals_by_name["vo"]["how"] == "new"
    assert locals_by_name["scene"]["type_name"] == "String"
    assert locals_by_name["scene"]["how"] == "local_decl"
    assert locals_by_name["detail"]["type_name"] == "ApplyDetail"
    # receivers_used includes field + locals
    recv = {r["name"]: r for r in payload["receivers_used"]}
    assert recv["applyService"]["type_name"] == "ApplyService"
    assert recv["applyService"]["how"] == "field_map"
    assert recv["vo"]["how"] == "new"
    # stats sanity
    assert payload["stats"]["locals"] == 4
    assert payload["stats"]["locals_typed"] >= 4
    assert payload["stats"]["call_sites"] >= 4


def test_build_payload_call_sites_setters(ast):
    """Setters on a local VO resolve with how=new and are not dropped."""
    from app.analysis.var_resolve import build_method_var_resolve_payload

    payload = build_method_var_resolve_payload(ast, "ApplyController#apply")
    sites = payload["call_sites"]
    set_sites = [s for s in sites if s["raw_call"].startswith("vo.")]
    assert any(s["raw_call"] == "vo.setScene" for s in set_sites)
    scene_site = next(s for s in set_sites if s["raw_call"] == "vo.setScene")
    assert scene_site["resolved_type"] == "ApplyVO"
    assert scene_site["resolve_how"] == "new"
    assert scene_site["status"] == "skipped"  # setter skip but typed


def test_build_payload_field_call(ast):
    from app.analysis.var_resolve import build_method_var_resolve_payload

    payload = build_method_var_resolve_payload(ast, "ApplyController#apply")
    site = next(
        (s for s in payload["call_sites"] if s["raw_call"] == "applyService.doApply"),
        None,
    )
    assert site is not None
    assert site["resolved_type"] == "ApplyService"
    assert site["resolve_how"] == "field_map"
    assert site["status"] == "ok_local"
    assert site["resolved"] == "ApplyService#doApply"


def test_payload_missing_method(ast):
    from app.analysis.var_resolve import build_method_var_resolve_payload

    payload = build_method_var_resolve_payload(ast, "NoSuch#method")
    assert payload.get("error") == "method_not_in_ast"


def test_setter_sites_from_bags_exact_field(ast):
    from app.analysis.var_resolve import (
        build_method_var_resolve_payload,
        setter_sites_from_bags,
    )

    bags = {
        "ApplyController#apply": build_method_var_resolve_payload(
            ast, "ApplyController#apply"
        )
    }
    sites = setter_sites_from_bags(bags, field_name="scene", limit=20)
    exact = [s for s in sites if s["exact_field"]]
    assert any(s["setter"] == "setScene" for s in exact)
    assert all(s["setter"].startswith("set") for s in sites)


def test_collect_entry_targets_hop(ast):
    from app.analysis.var_resolve import collect_entry_targets

    targets = collect_entry_targets(ast, "ApplyController#apply", hop=1)
    assert "ApplyController#apply" in targets
    assert "ApplyService#doApply" in targets
    # same-class private helper included
    assert "ApplyController#saveLog" in targets


def test_calls_edge_props_from_real_ast(ast):
    from app.analysis.var_resolve import calls_edge_props

    edge = next(e for e in ast.edges if e.raw_call == "applyService.doApply")
    props = calls_edge_props(edge)
    assert props["raw_call"] == "applyService.doApply"
    assert props["receiver"] == "applyService"
    assert props["receiver_type"] == "ApplyService"
    assert props["resolve_how"] == "field_map"
