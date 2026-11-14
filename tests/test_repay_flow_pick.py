"""Smoke:「还款流程」must pick RepayController#repayment, not queryRepayBank."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.agent.explore_agent import (
        _pick_structured_flow_doc,
        _rerank_hits_for_question,
        _flow_question_hit_boost,
    )
    from app.core.config import get_settings

    q = "还款流程是什么样的"
    pid = get_settings().project_id
    picked = _pick_structured_flow_doc(pid, q)
    assert picked is not None, "expected a FlowDoc for 还款"
    cm = picked.get("class_method") or ""
    assert cm == "RepayController#repayment", f"wrong entry: {cm}"
    assert not picked.get("is_queryish")

    hits = [
        {
            "fqcn": "com.demo.loan.controller.BankcardController",
            "method_sig": "public Response queryRepayBank(",
            "kind": "Controller",
            "summary": "query repay bank",
            "score": 5.6,
            "class_method": "BankcardController#queryRepayBank",
        },
        {
            "fqcn": "com.demo.loan.controller.RepayController",
            "method_sig": "com.demo.loan.controller.RepayController#repayment",
            "kind": "Controller",
            "summary": "数据源头 HTTP入口 RepayController#repayment 还款",
            "score": 2.8,
            "class_method": "RepayController#repayment",
        },
    ]
    assert _flow_question_hit_boost(hits[1], q) > _flow_question_hit_boost(hits[0], q)
    top = _rerank_hits_for_question(hits, q)[0]["class_method"]
    assert top == "RepayController#repayment", f"rerank top={top}"
    print("OK", cm)


if __name__ == "__main__":
    main()
