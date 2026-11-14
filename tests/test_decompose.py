"""Task DAG planner: multi-question / multi-entry decompose."""
from app.agent.decompose import heuristic_decompose, wants_decompose
from app.agent.runtime.orchestrator import group_subtasks, serial_groups, serial_task_order
from app.agent.runtime.planner import (
    classify_skill,
    distinct_api_anchors,
    extract_entry_hint,
    heuristic_dag,
    split_question_segments,
)


def test_simple_field_ask_not_decomposed():
    q = "放款接口 机构限额校验失败了 状态怎么更新"
    assert not wants_decompose(q)
    assert heuristic_decompose(q) is None


def test_flow_plus_status_decomposed():
    q = "放款接口什么流程，另外机构限额校验失败后放款状态怎么更新"
    assert wants_decompose(q)
    plan = heuristic_decompose(q)
    assert plan is not None
    kinds = [t["kind"] for t in plan["tasks"]]
    # Heuristic DAG no longer keyword-classifies skills; LLM owns kinds.
    assert kinds and all(k == "general" for k in kinds)
    assert plan.get("share_entry") is True


def test_numbered_two_apis_not_forced_share():
    q = "1. 放款接口什么流程\n2. 额度接口入口在哪"
    assert wants_decompose(q)
    segs = split_question_segments(q)
    assert len(segs) >= 2
    dag = heuristic_dag(q)
    assert dag is not None
    hints = {n.entry_hint for n in dag.nodes if n.entry_hint}
    assert "放款接口" in hints
    assert "额度接口" in hints
    assert dag.share_entry is False
    groups = group_subtasks(dag.to_legacy()["tasks"], share_entry=False)
    assert len(groups) >= 2
    assert dag.nodes[0].depends_on == []
    for i, n in enumerate(dag.nodes[1:], 1):
        assert n.depends_on == [dag.nodes[i - 1].id]
    serial = serial_groups(dag.to_legacy()["tasks"], share_entry=False)
    flat = [t["id"] for g in serial for t in g]
    assert flat == [n.id for n in dag.nodes]


def test_qmark_two_questions():
    q = "放款接口什么流程？额度接口入口在哪？"
    assert wants_decompose(q)
    dag = heuristic_dag(q)
    assert dag is not None
    assert dag.share_entry is False
    assert len(dag.nodes) >= 2


def test_compose_preserves_sections():
    from app.agent.decompose import compose_subtask_answers

    ans = compose_subtask_answers(
        [
            {
                "kind": "explain_flow",
                "title": "主链",
                "answer": "业务入口：`A#b`\n1. `X#y`",
            },
            {
                "kind": "field_value",
                "title": "状态",
                "answer": "关卡：`G#c`\nloanStatus=`F`",
            },
        ],
        entry="WithdrawalsController#withdrawals",
    )
    assert "综合回答" in ans
    assert "WithdrawalsController#withdrawals" in ans
    assert "主链" in ans and "状态" in ans
    assert "loanStatus" in ans


def test_compose_multiple_entries():
    from app.agent.decompose import compose_subtask_answers

    ans = compose_subtask_answers(
        [
            {
                "kind": "explain_flow",
                "title": "放款",
                "entry": "WithdrawalsController#withdrawals",
                "answer": "`A#a`",
            },
            {
                "kind": "general",
                "title": "额度",
                "entry": "AmountController#amount",
                "answer": "`B#b`",
            },
        ]
    )
    assert "各节业务入口" in ans
    assert "WithdrawalsController#withdrawals" in ans
    assert "AmountController#amount" in ans


def test_classify_and_anchors():
    assert classify_skill("放款接口什么流程") == "general"
    assert extract_entry_hint("放款接口什么流程") == "放款接口"
    assert distinct_api_anchors("放款接口和额度接口") == ["放款接口", "额度接口"]


def test_group_subtasks_share_vs_split():
    tasks = [
        {"id": "S1", "kind": "explain_flow", "entry_hint": "放款接口"},
        {"id": "S2", "kind": "field_value", "entry_hint": "放款接口"},
    ]
    assert len(group_subtasks(tasks, share_entry=True)) == 1
    split = [
        {"id": "S1", "kind": "explain_flow", "entry_hint": "放款接口"},
        {"id": "S2", "kind": "general", "entry_hint": "额度接口"},
    ]
    groups = group_subtasks(split, share_entry=False)
    assert len(groups) == 2


def test_serial_groups_keeps_later_unhinted_last():
    tasks = [
        {"id": "S1", "kind": "explain_flow", "entry_hint": "放款接口", "depends_on": []},
        {
            "id": "S2",
            "kind": "explain_flow",
            "entry_hint": "授信通知接口",
            "depends_on": ["S1"],
        },
        {"id": "S3", "kind": "change_plan", "entry_hint": "", "depends_on": ["S2"]},
    ]
    grouped = group_subtasks(tasks, share_entry=False)
    grouped_ids = [t["id"] for g in grouped for t in g]
    assert grouped_ids[0] == "S1"
    # bucket merge would pull S3 next to S1; serial must not.
    serial = [t["id"] for g in serial_groups(tasks, share_entry=False) for t in g]
    assert serial == ["S1", "S2", "S3"]
    assert [t["id"] for t in serial_task_order(tasks)] == ["S1", "S2", "S3"]
