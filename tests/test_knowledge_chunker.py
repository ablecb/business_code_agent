"""Knowledge-base chunker unit tests (no DB)."""
from __future__ import annotations


def test_chunk_text_splits_long():
    from app.knowledge.chunker import chunk_text

    text = ("段落A。\n\n" + ("很长内容" * 80) + "\n\n段落B。")
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) >= 2
    assert all(chunks)


def test_extract_txt_utf8():
    from app.knowledge.chunker import extract_text_from_bytes

    raw = "接口文档\nbusinessLabel".encode("utf-8")
    assert "businessLabel" in extract_text_from_bytes(
        raw, filename="api.md", kind="md"
    )
