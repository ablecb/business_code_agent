"""Unit: LLM JSON repair for truncated deep-node responses."""
from __future__ import annotations

from app.analysis.treesitter_deep_flow import _loads_llm_json, _repair_json_text


def test_repair_truncated_string_and_trailing_comma():
    bad = '{"a":1,"b":{"c":"hello","d":[1,2,],"e":"wor'
    fixed = _repair_json_text(bad)
    assert fixed.endswith("}")
    obj = _loads_llm_json(bad)
    assert obj["a"] == 1
    assert obj["b"]["c"] == "hello"
    assert obj["b"]["d"] == [1, 2]
    assert obj["b"]["e"] == "wor"


def test_repair_truncated_mid_object():
    bad = '{"summary":"x","impl_view":{"params":[{"name":"a"}],'
    obj = _loads_llm_json(bad)
    assert obj["summary"] == "x"
    assert isinstance(obj["impl_view"], dict)
    assert obj["impl_view"]["params"][0]["name"] == "a"


def test_loads_with_fence():
    raw = '```json\n{"ok": true, "n": 1}\n```'
    obj = _loads_llm_json(raw)
    assert obj["ok"] is True
    assert obj["n"] == 1


def test_split_method_code_chunks_covers_all():
    from app.analysis.treesitter_deep_flow import _split_method_code_chunks

    code = "\n".join(f"line-{i}-xxxxxxxx;" for i in range(200))
    chunks = _split_method_code_chunks(code, chunk_size=200, soft_max=350, max_chunks=8)
    assert len(chunks) >= 2
    assert "".join(chunks) == code
    # Soft-max wait: non-final chunks stay near soft_max, never mid-line
    assert all(len(c) <= 400 or c is chunks[-1] for c in chunks)
    assert all(c.endswith("\n") or c.endswith(";") or c is chunks[-1] for c in chunks)


def test_split_prefers_block_boundary():
    from app.analysis.treesitter_deep_flow import _split_method_code_chunks

    body = ["public void foo() {\n"]
    for i in range(40):
        body.append(f"  if (flag{i}) {{\n")
        body.append(f"    doWork{i}(a, b);\n")
        body.append("  }\n")
    body.append("}\n")
    code = "".join(body)
    chunks = _split_method_code_chunks(code, chunk_size=180, soft_max=280, max_chunks=20)
    assert "".join(chunks) == code
    assert len(chunks) >= 2
    # Prefer ending on complete if-blocks, not mid-statement
    for c in chunks[:-1]:
        tail = c.rstrip()
        assert tail.endswith("}") or tail.endswith(";"), repr(tail[-40:])