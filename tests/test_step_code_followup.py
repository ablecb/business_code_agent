"""Follow-up『第N步/N中对应代码』→ paste that step's method, not entry."""
from __future__ import annotations

from app.agent.explore_intent import extract_step_code_index
from app.agent.explore_shared import parse_numbered_answer_steps
from app.agent.explore_source import (
    _score_symbol_for_step_title,
    resolve_step_code_target,
)


def test_extract_step_code_index():
    assert extract_step_code_index("我想看看 6中对应代码怎么写的") == 6
    assert extract_step_code_index("第6步的源码") == 6
    assert extract_step_code_index("步骤 3 代码") == 3
    assert extract_step_code_index("放款接口controoler源码") is None


def test_parse_numbered_answer_steps():
    reply = (
        "放款接口的主要流程如下：\n\n"
        "1. **受理请求**：接收用户的放款请求。\n"
        "2. **参数校验**：校验关键字段。\n"
        "6. **放款处理**：处理贷款放款流程，含风控校验等。\n"
        "7. **结果回写**：判断是否成功。\n"
    )
    steps = parse_numbered_answer_steps(reply)
    by_i = {s["index"]: s["title"] for s in steps}
    assert by_i[1] == "受理请求"
    assert by_i[6] == "放款处理"


def test_score_prefers_search_hit():
    hits = ["WithdrawalsServiceImpl#loan"]
    assert _score_symbol_for_step_title(
        "WithdrawalsServiceImpl#loan", "放款处理", search_hits=hits
    ) > _score_symbol_for_step_title(
        "WithdrawalsController#withdrawals", "放款处理", search_hits=hits
    )


def test_resolve_uses_step_class_method_when_present():
    target = resolve_step_code_target(
        "demo-loan",
        step_n=6,
        answer_steps=[
            {
                "index": 6,
                "title": "放款处理",
                "detail": "",
                "class_method": "WithdrawalsServiceImpl#loan",
            },
        ],
        flow_symbols=[
            "WithdrawalsController#withdrawals",
            "WithdrawalsServiceImpl#loan",
        ],
        entry_cm="WithdrawalsController#withdrawals",
        step_title_hint="放款处理",
    )
    assert target == "WithdrawalsServiceImpl#loan"


def test_resolve_prefers_search_hit_in_candidates(monkeypatch):
    def _fake_search(project_id, query, top_k=12):
        return {
            "hits": [
                {
                    "fqcn": "com.demo.WithdrawalsServiceImpl",
                    "method_name": "loan",
                    "name": "loan",
                }
            ]
        }

    monkeypatch.setattr(
        "app.agent.explore_tools.search_symbols", _fake_search
    )
    target = resolve_step_code_target(
        "demo-loan",
        step_n=6,
        answer_steps=[
            {"index": 6, "title": "放款处理", "detail": ""},
        ],
        flow_symbols=[
            "WithdrawalsController#withdrawals",
            "WithdrawalsController#checkApplyReqParam",
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsController#applyResultDealWith",
        ],
        entry_cm="WithdrawalsController#withdrawals",
        step_title_hint="放款处理",
    )
    assert target == "WithdrawalsServiceImpl#loan"
