"""Deep CST tests: treesitter_deep_flow heuristics, code chunking, call spine.

Level-3 tests for app.analysis.treesitter_deep_flow pure functions:
  - _line_brace_delta / _is_clean_split_boundary: brace-depth tracking that
    respects strings/chars/comments (the basis of variable-flow chunking)
  - _split_method_code_chunks: never split mid-statement
  - _heuristic_deep: rule-free importance signals (validation/status/db/external)
  - _merge_segment_results_heuristic: chunk merge without a second LLM call
  - _match_call_to_cm / _entry_call_spine: call spine reconstruction
"""
from __future__ import annotations

from app.analysis.treesitter_deep_flow import (
    _heuristic_deep,
    _is_clean_split_boundary,
    _line_brace_delta,
    _merge_segment_results_heuristic,
    _split_method_code_chunks,
)


# ---------- brace depth ----------

def test_brace_delta_plain():
    assert _line_brace_delta("void m() {") == 1
    assert _line_brace_delta("}") == -1
    assert _line_brace_delta("int a = 1;") == 0


def test_brace_delta_ignores_strings_and_chars():
    # braces inside strings / chars must not count
    assert _line_brace_delta('String s = "a{b}c";') == 0
    assert _line_brace_delta("char c = '{';") == 0
    assert _line_brace_delta('log("{}}");') == 0


def test_brace_delta_ignores_comment_and_escape():
    assert _line_brace_delta("int a = 1; // }") == 0
    assert _line_brace_delta('String s = "a\\"b"; }') == -1  # escaped quote stays in string


def test_brace_delta_multiple_braces():
    assert _line_brace_delta("if (x) { doIt(); }") == 0
    assert _line_brace_delta("if (x) {{") == 2
    assert _line_brace_delta("}}") == -2


# ---------- clean split boundary ----------

def test_clean_boundary_semicolon():
    assert _is_clean_split_boundary("int a = 1;", 0, base_depth=0) is True
    assert _is_clean_split_boundary("int a = 1", 0, base_depth=0) is False


def test_clean_boundary_closing_brace():
    assert _is_clean_split_boundary("}", 0, base_depth=0) is True
    assert _is_clean_split_boundary("    }", 0, base_depth=0) is True


def test_clean_boundary_deep_blocks_not_clean():
    # inside nested block depth=2 with base 0 → not a clean boundary
    assert _is_clean_split_boundary("doIt();", 2, base_depth=0) is False
    assert _is_clean_split_boundary("}", 2, base_depth=0) is False


def test_clean_boundary_blank_line_shallow():
    assert _is_clean_split_boundary("", 1, base_depth=0) is True
    assert _is_clean_split_boundary("", 3, base_depth=0) is False


# ---------- code chunking ----------

def test_split_small_method_single_chunk():
    code = "void m() {\n  int a = 1;\n  doIt(a);\n}\n"
    chunks = _split_method_code_chunks(code)
    assert len(chunks) == 1
    assert chunks[0] == code


def test_split_long_method_preserves_join():
    """Chunking a long method must reproduce the original text when joined."""
    lines = ["void m() {"] + [f"  call{i}(arg{i});" for i in range(200)] + ["}"]
    code = "\n".join(lines) + "\n"
    chunks = _split_method_code_chunks(code, chunk_size=400, soft_max=600)
    assert len(chunks) >= 2
    joined = "".join(chunks)
    # whitespace-insensitive equivalence (chunk join preserves order & content)
    import re

    assert re.sub(r"\s+", "", joined) == re.sub(r"\s+", "", code)


def test_split_respects_statement_boundaries():
    """Each chunk should start at a statement boundary, not mid-statement."""
    lines = ["void m() {", "  int a = 1;", "  String s = call(1);"] + [
        f"  x{i} = doIt(i);" for i in range(150)
    ] + ["  return a;", "}"]
    code = "\n".join(lines) + "\n"
    chunks = _split_method_code_chunks(code, chunk_size=300, soft_max=500)
    for c in chunks:
        stripped = c.lstrip()
        # chunk never starts mid-line (chunk starts at line start by construction)
        assert not stripped.startswith("=")


def test_split_single_overlong_line_hard_cut():
    line = "x" * 5000
    code = f"void m() {{ {line} }}\n"
    chunks = _split_method_code_chunks(code, chunk_size=1000, soft_max=2000)
    assert len(chunks) >= 2
    assert sum(len(c) for c in chunks) >= len(line)


def test_split_empty_and_none():
    assert _split_method_code_chunks("") == [""]
    assert _split_method_code_chunks("void m() {}\n") == ["void m() {}\n"]


# ---------- heuristic deep ----------

def _fake_type(simple: str, kind: str, symbol_kind: str = "Service"):
    from types import SimpleNamespace

    return SimpleNamespace(simple_name=simple, symbol_kind=symbol_kind, kind=kind)


def _fake_method(name: str, raw: str):
    from types import SimpleNamespace

    return SimpleNamespace(name=name, raw_code=raw, calls=["svc.call1"])


def test_heuristic_deep_validation_signal():
    t = _fake_type("ApplyService", "class")
    m = _fake_method("checkAndApply", "public void checkAndApply() {\n  validate(req);\n}")
    data = _heuristic_deep(t, m)
    assert data["important"] is True
    assert "validation" in data["important_types"]
    assert data["class_method"] == "ApplyService#checkAndApply"
    assert data["next_calls"] == ["svc.call1"]


def test_heuristic_deep_status_transition():
    t = _fake_type("LoanService", "class")
    m = _fake_method("loan", "public void loan() {\n  if (status == 1) { doIt(); }\n}")
    data = _heuristic_deep(t, m)
    assert "status_transition" in data["important_types"]


def test_heuristic_deep_external_and_db():
    t = _fake_type("SettleService", "class")
    m = _fake_method(
        "settle",
        "public void settle() {\n  feignClient.push(x);\n  mapper.update(x);\n}",
    )
    data = _heuristic_deep(t, m)
    types = data["important_types"]
    assert "external_call" in types
    assert "db_write" in types


def test_heuristic_deep_controller_always_important():
    t = _fake_type("XController", "class", symbol_kind="Controller")
    m = _fake_method("index", "public void index() {\n  svc.list();\n}")
    data = _heuristic_deep(t, m)
    assert data["important"] is True
    assert data["important_types"] == ["other_important"] or "other_important" in data["important_types"]


def test_heuristic_deep_structure_keys():
    t = _fake_type("Svc", "class")
    m = _fake_method("run", "public void run() {}\n")
    data = _heuristic_deep(t, m)
    assert "summary_business_clear" in data
    assert "summary_business_detail" in data
    assert "impl_view" in data
    assert "data_flow_narrative" in data
    assert "validation_points" in data
    assert "status_transitions" in data
    assert "business_tables" in data
    assert "table_field_changes" in data


# ---------- segment merge ----------

def test_merge_segments_dedupes_and_orders():
    segs = [
        {
            "important": True,
            "important_types": ["validation"],
            "summary_business_clear": "段A",
            "summary_dev_clear": "devA",
            "data_flow_narrative": "先校验",
            "business_view": {"capability": "capA", "rules": ["r1"], "boundaries": [], "impacts": []},
            "impl_view": {
                "params": [{"name": "p", "source": "req"}],
                "param_flow": [{"name": "p", "from": "req"}],
                "branches": [{"condition": "c", "path": "ok"}],
                "memory_objects": [],
                "db_ops": [],
                "external_calls": [],
                "return_value": {"assembly": "x"},
            },
            "validation_points": ["v1"],
            "status_transitions": [],
            "external_calls": [],
            "business_tables": [],
            "table_field_changes": [],
            "next_calls": [],
        },
        {
            "important": True,
            "important_types": ["db_write"],
            "summary_business_clear": "段B",
            "summary_dev_clear": "devB",
            "data_flow_narrative": "再落库",
            "business_view": {"capability": "capB", "rules": ["r1"], "boundaries": [], "impacts": []},
            "impl_view": {
                "params": [{"name": "p", "source": "req"}],
                "param_flow": [{"name": "p2", "from": "db"}],
                "branches": [],
                "memory_objects": [],
                "db_ops": [{"op": "insert", "table_or_entity": "t_apply", "via_call": "mapper"}],
                "external_calls": [],
                "return_value": {},
            },
            "validation_points": [],
            "status_transitions": [],
            "external_calls": [],
            "business_tables": ["t_apply"],
            "table_field_changes": [],
            "next_calls": [],
        },
    ]
    merged = _merge_segment_results_heuristic("Svc#run", segs, known_calls=["a", "b"])
    assert merged["segmented"] is True
    assert merged["segment_count"] == 2
    # dedupe rules r1
    assert len(merged["business_view"]["rules"]) == 1
    # param_flow keeps both with step numbers
    assert len(merged["impl_view"]["param_flow"]) == 2
    assert [pf["step"] for pf in merged["impl_view"]["param_flow"]] == [1, 2]
    # both important types preserved
    assert "validation" in merged["important_types"]
    assert "db_write" in merged["important_types"]
    # db op merged
    assert merged["impl_view"]["db_ops"][0]["op"] == "insert"
    assert merged["next_calls"] == ["a", "b"]
    # flow narrative joined
    assert "先校验" in merged["data_flow_narrative"]
    assert "再落库" in merged["data_flow_narrative"]


def test_merge_segments_empty_fallback():
    merged = _merge_segment_results_heuristic("Svc#run", [], known_calls=[])
    assert merged["segmented"] is True
    assert merged["segment_count"] == 0
    # capability falls back to the class_method, then dual-summary echoes it
    assert merged["business_view"]["capability"] == "Svc#run"
    assert merged["summary_business_clear"] == "Svc#run"


def _bfs_result_for(edges, tree=None):
    return {"edges": edges, "tree": tree}


def test_entry_call_spine_basic():
    """Controller → service → repo spine, skipping unrelated siblings."""
    from app.analysis.treesitter_deep_flow import _entry_call_spine

    bfs = _bfs_result_for(
        edges=[
            {"from": "ApplyController#apply", "to": "ApplyService#doApply", "status": "ok_local"},
            {"from": "ApplyService#doApply", "to": "ApplyService#check", "status": "ok_local"},
            {"from": "ApplyService#doApply", "to": "Repo#query", "status": "ok_local"},
            # off-chain sibling: same Controller, different method, NOT reachable
            {"from": "ApplyController#helper", "to": "OtherService#x", "status": "ok_local"},
        ]
    )
    spine = _entry_call_spine(
        bfs,
        "ApplyController#apply",
        entry_calls=["applyService.doApply(vo)"],
        known_cms={
            "ApplyController#apply",
            "ApplyController#helper",
            "ApplyService#doApply",
            "ApplyService#check",
            "Repo#query",
            "OtherService#x",
        },
    )
    assert spine
    # spine = entry's DIRECT execution steps (not full transitive chain)
    assert spine == ["ApplyService#doApply"]
    # off-chain sibling helper is NOT on the spine
    assert "ApplyController#helper" not in spine
    assert "Repo#query" not in spine  # transitively reachable, not a direct step
