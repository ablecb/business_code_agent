from __future__ import annotations

from app.tagging.search import extract_query_literals


def test_extract_keeps_business_phrase_drops_question_glue():
    terms = extract_query_literals("授信通知流程是什么样的")
    assert "授信通知" in terms
    assert "流程" not in terms
    assert "什么样" not in terms


def test_extract_english_identifier():
    terms = extract_query_literals("OrderController submit flow")
    assert "OrderController" in terms or "submit" in terms
