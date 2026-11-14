"""Live MySQL regression:「放款过程中都调用了哪些接口」locks withdrawals FlowDoc.

Requires local MySQL with demo-loan FlowDocs (see tests/fix_withdrawals_flow_tables.py).
Skips cleanly when WithdrawalsController#withdrawals FlowDoc is missing.

Run:
    python tests/test_fangkuan_downstream_mysql.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.explore_agent import (
    _collect_flow_downstream_apis,
    _lexical_main_entry,
    _load_flow_doc_by_class_method,
    _pick_structured_flow_doc,
    _question_match_tokens,
    _render_flow_downstream_apis,
    _wants_api_inventory,
    _wants_flow_downstream_apis,
)
from app.agent.graph_agent import _is_followup, _question_has_content_noun
from app.core.config import get_settings
from app.db.mysql import init_db


def test_fangkuan_downstream_mysql_flowdoc() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    init_db()
    q = "放款过程中都调用了哪些接口"
    assert _wants_flow_downstream_apis(q)
    assert not _wants_api_inventory(q)
    assert not _is_followup(q)
    assert _question_has_content_noun(q)
    # Product CN→EN lexicon removed; entry comes from retrieval/LLM.
    assert _lexical_main_entry(q) is None
    assert "放款" in _question_match_tokens(q)
    assert "接口" not in _question_match_tokens(q)

    prefer = "WithdrawalsController#withdrawals"
    flow = _pick_structured_flow_doc(
        settings.project_id, q, prefer_class_method=prefer
    )
    if not flow or "Withdrawals" not in str(flow.get("class_method") or ""):
        flow = _load_flow_doc_by_class_method(
            settings.project_id, "WithdrawalsController#withdrawals"
        )
    if not flow:
        print("SKIP: no WithdrawalsController#withdrawals FlowDoc in MySQL")
        return

    assert "WithdrawalsController#withdrawals" in str(flow.get("class_method"))
    entry, items = _collect_flow_downstream_apis(flow)
    cms = [cm for cm, _ in items]
    text = _render_flow_downstream_apis(flow)
    assert "AmountController" not in text
    assert "EntrustedRefundController" not in text
    assert not any("AmountController" in c for c in cms)
    assert not any("EntrustedRefund" in c for c in cms)
    assert any(
        k in text for k in ("loanSwitch", "loanApply", "asynRisk", "loanDecision")
    ), text[:500]
    print("OK entry=", entry, "n=", len(cms))
    print(text[:600])


if __name__ == "__main__":
    test_fangkuan_downstream_mysql_flowdoc()
