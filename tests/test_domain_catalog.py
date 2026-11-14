"""Unit: dynamic business-domain catalog (reuse then create)."""
from __future__ import annotations

from app.analysis.domain_catalog import (
    domain_instruction,
    remember_domain,
    resolve_business_domain,
)
from app.analysis.treesitter_deep_flow import _deep_node_system_prompt, _dto_system_prompt


def test_prompts_have_no_hardcoded_industry_words():
    for text in (_dto_system_prompt([]), _deep_node_system_prompt([])):
        for bad in ("放款", "授信", "风控", "额度"):
            assert bad not in text


def test_domain_instruction_lists_known():
    text = domain_instruction(["提现申请", "还款入账"])
    assert "提现申请" in text
    assert "reuse" in text
    assert "create" in text


def test_resolve_reuses_then_creates(monkeypatch):
    pid = "unit-domain-catalog"
    remember_domain(pid, "提现申请")
    assert (
        resolve_business_domain(pid, "提现申请", domain_action="reuse", known=["提现申请"])
        == "提现申请"
    )
    created = resolve_business_domain(
        pid, "担保路由", domain_action="create", known=["提现申请"]
    )
    assert created == "担保路由"
