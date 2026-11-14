"""PEV Planner–Executor–Verifier — goal/plan/verify + test tags."""
from __future__ import annotations

import asyncio


def test_topic_answer_rejects_bare_yes():
    from app.agent.pev_agent import _topic_answer_too_thin

    assert _topic_answer_too_thin("有。")
    assert _topic_answer_too_thin("没有")
    assert not _topic_answer_too_thin(
        "有。相关入口包括 `SignatureController`（结清证明申请）。"
    )


def test_heuristic_goal_change_plan_and_zh_label():
    from app.agent.pev_agent import extract_field_label_zh, heuristic_goal

    q = (
        "我想在放款接口增加一个 businessLabel（业务标签），"
        "并落库到demo_b_withdrawals 和demo_b_loan_assistance，"
        "然后调用额度校验的时候将这个字段作为入参传递过去 需要改哪些地方"
    )
    assert extract_field_label_zh(q) == "业务标签"
    g = heuristic_goal(q)
    # Enhanced heuristic: structural add-field+落库+改哪些 → change_plan
    # (previously always general; goal LLM owns routing, heuristic seeds better).
    assert g["goal"] == "change_plan"
    assert "demo_b_withdrawals" in (g.get("tables") or [])
    assert g.get("field_label_zh") == "业务标签"


def test_compound_goal_plan_includes_add_api_and_change_tools():
    from app.agent.pev_agent import build_task_plan, heuristic_goal, wants_compound_change

    q = (
        "按知识库商品查询文档新增接口，同时放款接口新增入参 businessLabel 并落库，"
        "给我完整修改清单"
    )
    assert wants_compound_change(q)
    g = heuristic_goal(q)
    # 按知识库新增接口 + 已有接口加字段落库 → compound/change-plan-ish heuristic.
    assert g["goal"] in {"change_plan", "compound", "general"}, g["goal"]
    g_llm = {
        "goal": "compound",
        "also": ["add_api", "change_plan"],
        "must": [],
    }
    tools = [t["tool"] for t in build_task_plan(g_llm)]
    assert "load_kb_spec" in tools
    assert "render_new_api" in tools
    assert "discover_change" in tools
    assert "render_change" in tools


def test_annotate_git_checkout_create_title():
    from app.agent.action_policy import annotate_action_dict

    a = annotate_action_dict(
        {"type": "git_checkout", "branch": "feature/goods-query", "create": True}
    )
    assert "新建" in (a.get("ui_title") or "")
    assert "feature/goods-query" in (a.get("ui_preview") or "")


def test_extract_goal_prefers_llm_over_heuristic_seed(monkeypatch):
    """Desktop log bug: LLM returned inventory/repo_files for add-field+DDL ask."""
    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = (
        "我想放款接口增加一个入参 businessLabel(业务标签) 并落库到放款表，"
        "同时在额度校验时将该参数作为入参传递给外部系统  "
        "给我对应的程序修改清单和ddl"
    )
    # Enhanced heuristic: add-field + 落库 + 修改清单 → change_plan (no longer general).
    assert pev.heuristic_goal(q)["goal"] == "change_plan"

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"inventory","must":[{"id":"entry","desc":"x"},'
                '{"id":"spine","desc":"y"}],'
                '"field_name":"businessLabel","field_label_zh":"业务标签",'
                '"tables":["放款表"],"inventory_kind":"repo_files",'
                '"deliverable":"程序修改清单和DDL"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr(
        "app.llm.factory.get_llm_provider", lambda: _FakeLLM()
    )
    out = asyncio.run(pev.extract_goal_contract(q))
    # LLM goal preferred over heuristic seed (no hard lock).
    assert out["goal"] == "inventory"
    assert out.get("source") == "llm"
    assert out.get("field_name") == "businessLabel"
    tools = [t["tool"] for t in pev.build_task_plan(out)]
    assert "repo_files" in tools or "api_inventory" in tools


def test_extract_goal_llm_change_plan_not_forced_compound(monkeypatch):
    """LLM change_plan must not be hard-overridden to compound by 放款 regex."""
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = (
        "按知识库商品查询文档新增接口，同时放款接口新增入参 businessLabel 并落库，"
        "给我完整修改清单"
    )

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"change_plan","also":[],'
                '"must":[{"id":"entry","desc":"改动清单"}],'
                '"field_name":"businessLabel","deliverable":"清单"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(pev.extract_goal_contract(q))
    assert out["goal"] == "change_plan"
    assert out.get("source") == "llm"


def test_heuristic_goal_explain_flow():
    from app.agent.pev_agent import heuristic_goal

    # Enhanced heuristic: structural flow shapes route to explain_flow even when
    # the Goal LLM is unavailable (desktop-chat-0c3000119343 degraded path).
    g = heuristic_goal("放款接口流程什么样")
    assert g["goal"] == "explain_flow"
    assert g.get("ask_scope") == "full_flow"
    assert heuristic_goal("某业务接口具体做了什么")["goal"] == "explain_flow"
    assert heuristic_goal("结清证明具体做了什么")["goal"] == "explain_flow"


def test_topic_entry_grounding_prefers_signature_controller(monkeypatch):
    """Bare topic flow ask must not lock an unrelated Controller."""
    import asyncio
    from pathlib import Path

    import app.agent.pev_agent as pev

    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.is_dir():
        return

    def _fake_resolve(project_id, question, *, entry_hint=None, prefer_hint=False):
        # Simulate weak lexical lock to repay (the bug).
        if entry_hint and "Signature" in str(entry_hint):
            return ["SignatureController#apply"], [{"cm": "SignatureController#apply"}]
        if "结清" in (question or "") or "Signature" in (question or ""):
            return (
                ["RepayDetailController#repayDetail"],
                [{"cm": "RepayDetailController#repayDetail"}],
            )
        return (
            ["RepayDetailController#repayDetail"],
            [{"cm": "RepayDetailController#repayDetail"}],
        )

    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta",
        _fake_resolve,
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.resolve_project_root",
        lambda _pid: root,
    )
    evidence: dict = {}
    asyncio.run(
        pev._exec_resolve_entry("demo-loan", "结清证明具体做了什么", evidence)
    )
    entry = str(evidence.get("entry") or "")
    assert "Signature" in entry or entry == "", evidence
    assert "RepayDetail" not in entry, evidence
    assert evidence.get("topic_ground_phrase") == "结清证明"
    assert any(
        str(c).endswith("Controller")
        for c in (evidence.get("topic_ground_controllers") or [])
    )


def test_explain_flow_goal_llm_full_flow(monkeypatch):
    """Whole-API flow asks go through Goal LLM → ask_scope=full_flow."""
    import asyncio

    import app.agent.pev_agent as pev
    from app.llm.mock import MockLLMProvider

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: MockLLMProvider())
    for q in (
        "某业务接口什么流程",
        "某业务接口具体做了什么",
        "通知接口什么流程",
    ):
        out = asyncio.run(pev.extract_goal_contract(q))
        assert out["goal"] == "explain_flow", q
        assert out.get("ask_scope") == "full_flow", q
        assert out.get("source") in {"llm", "llm+heuristic"}, q
        assert not out.get("dig_in_focus"), q
        tools = [t["tool"] for t in pev.build_task_plan(out)]
        assert tools[:3] == ["resolve_entry", "expand_calls", "mysql_enrich"], tools


def test_summarize_neo4j_calls_falls_back_when_polish_fails(monkeypatch):
    """neo4j_calls polish LLM fails → deterministic grounded fallback (accuracy kept)."""
    import app.agent.pev_agent as pev
    from app.llm.base import LLMProvider

    class _BoomLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            raise AssertionError("polish LLM down (simulated)")

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _BoomLLM())
    evidence = {
        "entry": "CreditController#creditNotification",
        "spine_source": "neo4j_calls",
        "calls_count": 3,
        "mysql_rows": [
            {
                "order": 1,
                "class_method": "CreditController#creditNotification",
                "rel": "ENTRY",
                "summary": "入口",
            },
            {
                "order": 2,
                "class_method": "CreditServiceImpl#creditNotification",
                "rel": "CALLS",
                "summary": "授信通知主流程",
            },
            {
                "order": 3,
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "rel": "CALLS",
                "summary": "同步会员价格信息",
            },
        ],
    }
    ans = asyncio.run(
        pev._exec_summarize_flow(
            "授信通知接口什么流程", evidence, project_id="", history_text=""
        )
    )
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "同步会员价格" in ans
    # Same fallback for calls_closure_filtered
    evidence2 = {**evidence, "spine_source": "calls_closure_filtered"}
    ans2 = asyncio.run(
        pev._exec_summarize_flow(
            "授信通知接口什么流程", evidence2, project_id="", history_text=""
        )
    )
    assert "CreditServiceImpl#creditNotification" in ans2


def test_summarize_flow_uses_business_narrative_polish(monkeypatch):
    """可靠主链也走 LLM 业务叙事润色（desktop-chat-75fe902deb0b 死板清单问题）。

    polish 输出必须是业务叙事（保留锚点），且「依据：」行自动补齐。
    """
    import app.agent.pev_agent as pev
    from app.llm.base import LLMProvider

    class _BizLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            system = messages[0].get("content", "") if messages else ""
            if "业务视角" in system or "业务入口" in system:
                return (
                    "这笔授信通知业务从受理到落库的完整链路如下：\n\n"
                    "业务入口：`CreditController#creditNotification`\n\n"
                    "1. `CreditServiceImpl#creditNotification` — 收到授信结果后先做主流程处理\n"
                    "2. `CreditServiceImpl#synchroMemberPrice` — 同步会员价格信息\n\n"
                    "**业务总结**：通知同步完成后更新授信状态并落库。"
                )
            return "fallback"

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _BizLLM())
    evidence = {
        "entry": "CreditController#creditNotification",
        "spine_source": "neo4j_calls",
        "calls_count": 2,
        "mysql_rows": [
            {
                "order": 1,
                "class_method": "CreditController#creditNotification",
                "rel": "ENTRY",
                "summary": "入口",
            },
            {
                "order": 2,
                "class_method": "CreditServiceImpl#creditNotification",
                "rel": "CALLS",
                "summary": "授信通知主流程",
            },
            {
                "order": 3,
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "rel": "CALLS",
                "summary": "同步会员价格信息",
            },
        ],
    }
    ans = asyncio.run(
        pev._exec_summarize_flow(
            "授信通知接口什么流程", evidence, project_id="", history_text=""
        )
    )
    # polish 的业务叙事保留锚点 + 业务总结，不再补「依据：节点数」脚注
    assert "授信通知业务" in ans
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "业务总结" in ans or "落库" in ans
    from app.agent.explore_answers import strip_answer_chrome

    assert "依据：节点数=" not in strip_answer_chrome(ans)


def test_heuristic_goal_repo_pom_inventory():
    from app.agent.pev_agent import (
        build_task_plan,
        extract_repo_file_globs,
        heuristic_goal,
        wants_repo_file_inventory,
    )

    q = "项目中几个pom文件"
    assert wants_repo_file_inventory(q)
    g = heuristic_goal(q)
    assert g["goal"] == "general"
    # LLM owns inventory routing; when it selects repo_files, globs + plan apply.
    globs = extract_repo_file_globs(q)
    assert "**/pom.xml" in globs
    tools = [
        t["tool"]
        for t in build_task_plan(
            {
                "goal": "inventory",
                "inventory_kind": "repo_files",
                "file_globs": globs,
                "must": [],
                "also": [],
            }
        )
    ]
    assert tools == ["repo_files"]
    assert "resolve_entry" not in tools


def test_list_repo_pom_files_demo():
    import pytest
    from pathlib import Path

    from app.agent.pev_agent import format_repo_files_answer, list_repo_files

    if not (Path(__file__).resolve().parents[1] / "java_project" / "demo-loan").is_dir():
        pytest.skip("demo-loan 工程目录不存在（本地未放置被索引仓库）")

    scan = list_repo_files("demo-loan", ["**/pom.xml"])
    assert scan["count"] >= 1
    assert any(p.endswith("pom.xml") for p in scan["paths"])
    ans = format_repo_files_answer(scan, "项目中几个pom文件")
    assert f"共 **{scan['count']}**" in ans
    assert "pom.xml" in ans


def test_build_task_plan_tools():
    from app.agent.pev_agent import build_task_plan

    tasks = build_task_plan({"goal": "change_plan"})
    tools = [t["tool"] for t in tasks]
    assert "resolve_entry" in tools
    assert "discover_change" in tools
    assert "find_outbound" in tools


def test_rewrite_placeholder_field_and_chinese_table_not_gap():
    from app.agent.pev_agent import _outbound_hit_score, _rewrite_placeholder_field, verify_contract

    raw = (
        "private String newField;\n"
        "public static final String NEW_FIELD = \"newField\";\n"
        "ADD COLUMN `new_field` ... COMMENT 'newField';\n"
        "reqVo.setNewField(newField);"
    )
    out = _rewrite_placeholder_field(raw, "businessLabel")
    assert "businessLabel" in out
    assert "BUSINESS_LABEL" in out
    assert "business_label" in out
    assert "setBusinessLabel" in out
    assert "newField" not in out

    q = "校验时传递"
    assert _outbound_hit_score("WithdrawalsServiceImpl#quotaCheck", q) > _outbound_hit_score(
        "WithdrawalsServiceImpl#updateWithdrawals", q
    )

    v = verify_contract(
        {
            "goal": "change_plan",
            "tables": ["放款表"],
            "must": [{"id": "entry", "desc": "x"}],
            "field_label_zh": "业务标签",
        },
        {
            "entry": "WithdrawalsController#withdrawals",
            "change_ok": True,
            "persist_tables": ["demo_b_withdrawals_ext"],
            "question": "落库",
            "intersect_cms": ["WithdrawalsServiceImpl#quotaCheck"],
        },
        "## 改动清单\n## DDL\nALTER TABLE `demo_b_withdrawals_ext`\n业务标签",
    )
    assert not any(g["id"] == "entity_table_mismatch" for g in v["gaps"])


def test_verify_and_footer_tags():
    from app.agent.pev_agent import _footer_tags, verify_contract

    goal = {
        "goal": "change_plan",
        "source": "heuristic",
        "field_label_zh": "业务标签",
        "tables": ["demo_b_withdrawals"],
        "must": [{"id": "entry", "desc": "x"}],
    }
    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "change_ok": True,
        "question": "传递 额度",
        "intersect_cms": [],
    }
    ans = "## 改动清单\nDDL\nALTER TABLE `demo_b_withdrawals`"
    v = verify_contract(goal, evidence, ans)
    assert v["ok"] is False
    assert any(g["id"] == "outbound" for g in v["gaps"])
    footer = _footer_tags(
        goal,
        [{"id": "T1", "status": "done"}],
        v,
        evidence,
    )
    assert "[PEV][TEST]" in footer
    assert "goal=change_plan" in footer


def test_apply_zh_comments():
    from app.agent.pev_agent import _apply_zh_comments

    raw = "ADD COLUMN `business_label` varchar(64) DEFAULT NULL COMMENT 'businessLabel';"
    out = _apply_zh_comments(raw, "业务标签", "businessLabel")
    assert "COMMENT '业务标签'" in out


def test_outbound_focus_prefers_quota_check_phrase():
    from app.agent.pev_agent import (
        _outbound_focus_queries,
        _outbound_hit_score,
    )

    q = (
        "我想在放款接口增加一个 businessLabel（业务标签），"
        "并落库到demo_b_withdrawals，然后调用额度校验的时候将这个字段作为入参传递过去"
    )
    focus = _outbound_focus_queries(q)
    assert any("额度校验" in f for f in focus), focus
    assert _outbound_hit_score(
        "WithdrawalsServiceImpl#quotaCheck", q
    ) > _outbound_hit_score("WithdrawalsServiceImpl#updateWithdrawals", q)


def test_suggest_outbound_param_from_map_put_peers():
    from app.agent.pev_agent import suggest_outbound_param_code

    code = """
        HashMap<String, Object> limitQueryMap = new HashMap<>();
        limitQueryMap.put(FieldNameConstants.CUST_ID, reqVo.getCustId());
        limitQueryMap.put(FieldNameConstants.PRODUCT_CODE, reqVo.getProductCode());
        map.put(FieldNameConstants.BIZ_TYPE, DEMO_TYPE);
"""
    out = suggest_outbound_param_code(code, "businessLabel", label_zh="业务标签")
    assert "业务标签" in out
    assert "limitQueryMap.put(FieldNameConstants.BUSINESS_LABEL, reqVo.getBusinessLabel())" in out
    # map.put peer whose value is not reqVo.get* is skipped
    assert "map.put" not in out


def test_merge_outbound_inserts_suggested_before_ddl():
    from app.agent.pev_agent import _merge_outbound_into_answer

    ans = "## 改动清单\n\n### 1. VO\n\n## DDL\n```sql\nALTER TABLE t;\n```\n"
    evidence = {
        "outbound_snippets": [
            {
                "class_method": "FooService#quotaCheck",
                "path": "Foo.java",
                "code": (
                    "        limitQueryMap.put(FieldNameConstants.CUST_ID, "
                    "reqVo.getCustId());\n"
                ),
            }
        ]
    }
    out = _merge_outbound_into_answer(
        ans, evidence, field="businessLabel", label_zh="业务标签"
    )
    assert out.index("外送入参透传") < out.index("## DDL")
    assert "getBusinessLabel()" in out
    assert "证据摘录" in out


def test_run_pev_pipeline_smoke_change(monkeypatch):
    from app.agent import pev_agent as pev

    async def _fake_goal(question, history_text="", prior_goal=None):  # noqa: ARG001
        return pev.heuristic_goal(question)

    monkeypatch.setattr(pev, "extract_goal_contract", _fake_goal)

    async def _resolve(project_id, question, evidence):
        evidence["entry"] = "WithdrawalsController#withdrawals"

    async def _discover(project_id, question, evidence):
        evidence["change_result"] = {
            "ok": True,
            "entry": "WithdrawalsController#withdrawals",
            "roles": {"request_vo": {}, "bind": {}},
            "plan_markdown": (
                "## 改动清单\n\n### 1. VO\n"
                "/**\n     * businessLabel\n     */\n"
                "## DDL\n```sql\n"
                "ALTER TABLE `demo_b_withdrawals_ext`\n"
                "  ADD COLUMN `business_label` varchar(64) "
                "DEFAULT NULL COMMENT 'businessLabel';\n```\n"
            ),
            "answer": "",
        }
        evidence["change_ok"] = True

    async def _outbound(project_id, question, evidence):
        evidence["intersect_cms"] = ["WithdrawalsServiceImpl#quotaCheck"]
        evidence["outbound_snippets"] = [
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheck",
                "path": "x.java",
                "code": (
                    "        limitQueryMap.put(FieldNameConstants.CUST_ID, "
                    "reqVo.getCustId());\n"
                ),
            }
        ]

    async def _goal_change(question, *, history_text="", prior_goal=None):  # noqa: ARG001
        return {
            "goal": "change_plan",
            "also": [],
            "must": [
                {"id": "entry", "desc": "入口"},
                {"id": "persist", "desc": "落库"},
                {"id": "deliverable", "desc": "改动清单"},
            ],
            "field_name": "businessLabel",
            "field_label_zh": "业务标签",
            "tables": ["demo_b_withdrawals"],
            "inventory_kind": None,
            "file_globs": [],
            "deliverable": "改动清单",
            "source": "llm",
        }

    monkeypatch.setattr(pev, "_exec_resolve_entry", _resolve)
    monkeypatch.setattr(pev, "_exec_discover_change", _discover)
    monkeypatch.setattr(pev, "_exec_find_outbound", _outbound)
    monkeypatch.setattr(pev, "extract_goal_contract", _goal_change)
    monkeypatch.setattr(
        "app.agent.decompose.wants_decompose", lambda *_a, **_k: False
    )

    q = (
        "我想在放款接口增加一个字段 businessLabel（业务标签），"
        "并落库到demo_b_withdrawals，然后传递过去，需要改哪些地方"
    )
    out = asyncio.run(pev.run_pev_pipeline(q, project_id="demo-loan"))
    assert out.get("plan", {}).get("kind") == "pev"
    assert out.get("plan", {}).get("goal", {}).get("goal") == "change_plan"
    # Internal [PEV][TEST] tags must NOT leak into the user-visible answer by
    # default; they are returned in result["pev_debug"]["footer"] instead.
    assert "[PEV][TEST]" not in (out.get("answer") or "")
    assert "[PEV][TEST]" in (out.get("pev_debug") or {}).get("footer", "")
    assert "业务标签" in (out.get("answer") or "")
    assert "quotaCheck" in (out.get("answer") or "")
    assert "getBusinessLabel()" in (out.get("answer") or "")
    assert any(s.get("tool") == "pev_goal" for s in out.get("steps") or [])


def test_build_task_plan_explain_flow_uses_follows_title():
    from app.agent.pev_agent import build_task_plan

    tasks = build_task_plan({"goal": "explain_flow"})
    titles = [t["title"] for t in tasks]
    assert any("FOLLOWS" in t or "执行序" in t for t in titles)


def test_filter_spine_drops_noise():
    from app.agent.pev_agent import _filter_spine_cms

    out = _filter_spine_cms(
        [
            "WithdrawalsController#checkApplyReqParam",
            "DateUtil#formatYYYYMMDDHHMMSSDate",
            "LogUtils#mqSend",
            "WithdrawalsServiceImpl#loan",
        ]
    )
    assert "WithdrawalsController#checkApplyReqParam" in out
    assert "WithdrawalsServiceImpl#loan" in out
    assert not any("DateUtil" in x for x in out)
    assert not any("LogUtils" in x for x in out)


def test_resolve_flow_spine_noise_only_is_empty(monkeypatch):
    from app.agent import pev_agent as pev

    monkeypatch.setattr(
        "app.analysis.flow_neo4j_sync.get_follows_path",
        lambda *_a, **_k: {"ordered_path": ["AmountController#amount"], "source": "follows"},
    )
    monkeypatch.setattr(
        "app.analysis.flow_neo4j_sync._ordered_calls_l1_spine",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda *_a, **_k: ["DateUtil#formatYYYYMMDD", "LogUtils#mqSend"],
    )
    calls, src = pev._resolve_flow_spine("demo-loan", "AmountController#amount")
    assert calls == []
    assert src == "empty"


def test_pipeline_empty_spine_marks_t2_failed(monkeypatch):
    """expand_calls with calls_count=0 must be FAIL, not OK calls_closure_filtered."""
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    class _BoomLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            raise AssertionError("no LLM")

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    async def _fake_resolve(project_id, question, evidence):  # noqa: ARG001
        evidence["entry"] = "AmountController#amount"

    async def _fake_expand(project_id, evidence, question=""):  # noqa: ARG001
        evidence["calls"] = []
        evidence["calls_count"] = 0
        evidence["spine_source"] = "empty"

    async def _fake_mysql(project_id, evidence):  # noqa: ARG001
        evidence["mysql_rows"] = []

    async def _goal_flow(question, *, history_text="", prior_goal=None):  # noqa: ARG001
        return {
            "goal": "explain_flow",
            "also": [],
            "must": [
                {"id": "entry", "desc": "入口"},
                {"id": "spine", "desc": "主链"},
                {"id": "summarize", "desc": "汇总"},
            ],
            "ask_scope": "full_flow",
            "field_name": None,
            "field_label_zh": None,
            "tables": [],
            "inventory_kind": None,
            "file_globs": [],
            "deliverable": "流程",
            "source": "llm",
        }

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _BoomLLM())
    monkeypatch.setattr(pev, "extract_goal_contract", _goal_flow)
    monkeypatch.setattr(pev, "_exec_resolve_entry", _fake_resolve)
    monkeypatch.setattr(pev, "_exec_expand_calls", _fake_expand)
    monkeypatch.setattr(pev, "_exec_mysql_enrich", _fake_mysql)
    monkeypatch.setattr(
        "app.agent.decompose.wants_decompose", lambda *_a, **_k: False
    )

    out = asyncio.run(
        pev.run_pev_pipeline(
            "某接口具体做了什么",
            project_id="demo-loan",
            lock_entry=False,
            dig_in_likely=False,
            history_text="",
        )
    )
    tasks = (out.get("plan") or {}).get("tasks") or []
    t2 = next(t for t in tasks if t.get("id") == "T2_spine")
    assert t2.get("status") == "failed", t2
    assert "FAIL" in str(t2.get("note") or "")
    assert "calls=0" in str(t2.get("note") or "")
    # [PEV][TEST] footer tags are kept out of the user answer; check pev_debug.
    assert "[PEV][TEST]" not in (out.get("answer") or "")
    footer = (out.get("pev_debug") or {}).get("footer", "")
    assert "failed=1" in footer or "tasks_done=3/4" in footer


def test_render_flow_deterministic_keeps_order_and_methods():
    from app.agent.pev_agent import _render_flow_answer_deterministic, verify_contract

    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "spine_source": "neo4j_follows",
        "calls_count": 3,
        "mysql_rows": [
            {
                "order": 1,
                "class_method": "WithdrawalsController#withdrawals",
                "summary": "入口受理",
            },
            {
                "order": 2,
                "class_method": "WithdrawalsController#checkApplyReqParam",
                "summary": "校验入参",
            },
            {
                "order": 3,
                "class_method": "WithdrawalsServiceImpl#loanSwitch",
                "summary": "放款开关",
            },
            {
                "order": 4,
                "class_method": "WithdrawalsServiceImpl#loan",
                "summary": "放款主处理",
            },
        ],
    }
    ans = _render_flow_answer_deterministic(evidence)
    assert "`WithdrawalsController#checkApplyReqParam`" in ans
    assert ans.index("checkApplyReqParam") < ans.index("loanSwitch")
    assert ans.index("`WithdrawalsServiceImpl#loanSwitch`") < ans.index(
        "`WithdrawalsServiceImpl#loan`"
    )
    v = verify_contract(
        {"goal": "explain_flow", "must": [{"id": "entry", "desc": "x"}]},
        evidence,
        ans,
    )
    assert v["ok"] is True, v


def test_verify_explain_flow_rejects_paraphrase_without_methods():
    from app.agent.pev_agent import verify_contract

    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "spine_source": "neo4j_follows",
        "calls_count": 4,
        "mysql_rows": [
            {"order": 1, "class_method": "WithdrawalsController#withdrawals", "summary": "a"},
            {
                "order": 2,
                "class_method": "WithdrawalsController#checkApplyReqParam",
                "summary": "b",
            },
            {
                "order": 3,
                "class_method": "WithdrawalsServiceImpl#loanSwitch",
                "summary": "c",
            },
            {"order": 4, "class_method": "WithdrawalsServiceImpl#loan", "summary": "d"},
        ],
    }
    vague = (
        "放款接口的流程如下：\n"
        "1. 首先校验参数。\n"
        "2. 根据开关配置判断。\n"
        "3. 处理贷款申请。\n"
        "4. 插入放款数据。\n"
        "依据：节点数=4"
    )
    v = verify_contract(
        {"goal": "explain_flow", "must": [{"id": "entry", "desc": "x"}]},
        evidence,
        vague,
    )
    assert v["ok"] is False
    assert any(g["id"] == "method_anchors" for g in v["gaps"])


def _credit_dig_in_evidence() -> dict:
    return {
        "entry": "CreditController#creditApply",
        "lock_entry": True,
        "spine_source": "neo4j_calls",
        "calls_count": 4,
        "project_id": "",
        "calls": [
            "CreditController#creditApply",
            "CreditServiceImpl#apply",
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#saveCredit",
        ],
        "mysql_rows": [
            {
                "order": 1,
                "class_method": "CreditController#creditApply",
                "summary": "授信申请入口",
            },
            {
                "order": 2,
                "class_method": "CreditServiceImpl#apply",
                "summary": "授信主处理",
            },
            {
                "order": 3,
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息到下游",
            },
            {
                "order": 4,
                "class_method": "CreditServiceImpl#saveCredit",
                "summary": "保存授信结果",
            },
        ],
    }


def test_summarize_lock_entry_llm_dig_in_focuses_method(monkeypatch):
    """Scheme 3: lock_entry + LLM dig_in → focused answer, not full chain."""
    import asyncio
    import re

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "这里面同步会员价格信息是做什么的"
    evidence = _credit_dig_in_evidence()

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"dig_in":true,"focus_method":"synchroMemberPrice",'
                '"topic":"member price sync","reason":"subtopic dig-in"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())

    ans = asyncio.run(
        pev._exec_summarize_flow(
            q,
            evidence,
            project_id="",
            history_text="assistant: 业务入口：`CreditController#creditApply`\n1. ...",
        )
    )
    assert evidence.get("dig_in_focus") is True
    assert any(
        str(c).endswith("#synchroMemberPrice")
        for c in (evidence.get("dig_in_focus_cms") or [])
    )
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "主链叶子" in ans or "由浅入深" in ans or "语义相似度" in ans
    # Must not rehash full numbered spine.
    assert not re.search(r"(?m)^\s*1\.\s+`", ans)
    assert "CreditServiceImpl#saveCredit" not in ans or "相关调用" in ans


def test_summarize_lock_entry_dig_in_false_full_chain(monkeypatch):
    """LLM dig_in=false keeps existing explain_flow / full-chain path."""
    import asyncio
    import re

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "把整个授信申请流程再讲一遍"
    evidence = _credit_dig_in_evidence()

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"dig_in":false,"focus_method":"","topic":"",'
                '"reason":"full flow re-ask"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())

    ans = asyncio.run(
        pev._exec_summarize_flow(q, evidence, project_id="", history_text="")
    )
    assert not evidence.get("dig_in_focus")
    assert "业务入口" in ans
    assert "主链" in ans or re.search(r"(?m)^\s*\d+\.\s+`", ans)
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "CreditServiceImpl#saveCredit" in ans


def test_verify_dig_in_fails_without_focus_method():
    """Scheme 4: dig-in verify requires focus Class#method coverage."""
    from app.agent.pev_agent import verify_contract

    evidence = _credit_dig_in_evidence()
    evidence["dig_in_focus"] = True
    evidence["dig_in_focus_cms"] = ["CreditServiceImpl#synchroMemberPrice"]
    vague = (
        "追问聚焦（锁定主链上的方法，未切换入口）：\n\n"
        "- 主链方法 `CreditServiceImpl#apply`\n"
        "  摘要：授信主处理\n"
    )
    v = verify_contract(
        {"goal": "explain_flow", "must": [{"id": "entry", "desc": "x"}]},
        evidence,
        vague,
    )
    assert v["ok"] is False
    assert any(g["id"] == "dig_in_focus" for g in v["gaps"])

    good = (
        "追问聚焦：\n"
        "- 主链方法 `CreditServiceImpl#synchroMemberPrice`\n"
        "  摘要：同步会员价格信息到下游\n"
        "依据：聚焦方法数=1\n"
    )
    v2 = verify_contract(
        {"goal": "explain_flow", "must": [{"id": "entry", "desc": "x"}]},
        evidence,
        good,
    )
    assert v2["ok"] is True, v2


def test_verify_show_code_requires_preferred_method_not_just_fence():
    """show_code verify: prefer_cms / dig-in focus must appear — not only ```."""
    from app.agent.pev_agent import verify_contract

    evidence = {
        "entry": "CreditController#creditNotification",
        "prefer_cms": ["CreditServiceImpl#synchroMemberPrice"],
        "source_view": {
            "ok": True,
            "target": "QueueHelper#submit",
            "plan": {"class_method": "QueueHelper#submit"},
        },
    }
    wrong = (
        "`QueueHelper#submit` 方法源码如下：\n\n"
        "```java\npublic void submit() {}\n```\n"
    )
    v = verify_contract(
        {"goal": "show_code", "must": [{"id": "paste", "desc": "x"}]},
        evidence,
        wrong,
    )
    assert v["ok"] is False
    assert any(g["id"] == "target_method" for g in v["gaps"])

    good = (
        "`CreditServiceImpl#synchroMemberPrice` 方法源码如下：\n\n"
        "```java\npublic void synchroMemberPrice() {}\n```\n"
    )
    v2 = verify_contract(
        {"goal": "show_code", "must": [{"id": "paste", "desc": "x"}]},
        evidence,
        good,
    )
    assert v2["ok"] is True, v2


def test_resolve_focus_method_by_name_and_summary():
    from app.agent.pev_agent import _resolve_focus_method_symbol

    evidence = _credit_dig_in_evidence()
    hit = _resolve_focus_method_symbol(
        "synchroMemberPrice", evidence, project_id="", topic="member price"
    )
    assert any(c.endswith("#synchroMemberPrice") for c in hit)


def test_summarize_lock_entry_dig_in_timeout_retrieval_fallback(monkeypatch):
    """LLM dig-in timeout → retrieval focus (not silent full-chain rehash)."""
    import asyncio
    import re

    from app.agent import pev_agent as pev

    q = "同步会员价格信息 具体怎么做的"
    # Mirror desktop log: L1 spine lacks the leaf; leaf is only in closure/search.
    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": True,
        "dig_in_likely": True,
        "spine_source": "neo4j_calls",
        "calls_count": 4,
        "project_id": "demo-loan",
        "calls": [
            "CreditController#generateReqParam",
            "CreditController#isRepeatCredit",
            "WithdrawalsApplyServiceImpl#checkMember",
            "CreditServiceImpl#creditNotification",
        ],
        "mysql_rows": [
            {
                "order": 1,
                "class_method": "CreditController#generateReqParam",
                "summary": "提取校验授信字段",
            },
            {
                "order": 2,
                "class_method": "CreditController#isRepeatCredit",
                "summary": "判断重复授信",
            },
            {
                "order": 3,
                "class_method": "WithdrawalsApplyServiceImpl#checkMember",
                "summary": "校验客户信息",
            },
            {
                "order": 4,
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "处理信用审批通知，包括同步会员价格信息",
            },
        ],
    }

    async def _timeout_decision(*_a, **_k):
        return {
            "dig_in": False,
            "focus_method": "",
            "topic": "",
            "reason": "timeout",
            "source": "timeout",
        }

    # Force LLM/timeout path (description match would otherwise win first).
    monkeypatch.setattr(
        pev, "_try_lock_dig_in_description_focus", lambda *_a, **_k: None
    )
    monkeypatch.setattr(pev, "decide_dig_in_focus", _timeout_decision)
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            [
                "CreditServiceImpl#synchroMemberPrice",
                "CusCbsServiceClient#synchroMemberPriceInfo",
                "CreditServiceImpl#assemParam",
            ]
            if "creditNotification" in str(root)
            else []
        ),
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda _pid, _q, top_k=16: [  # noqa: ARG005
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#creditNotification",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        ][:top_k],
    )
    monkeypatch.setattr(
        pev,
        "_related_cms_under_focus",
        lambda *_a, **_k: ["CusCbsServiceClient#synchroMemberPriceInfo"],
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])

    ans = asyncio.run(
        pev._exec_summarize_flow(
            q,
            evidence,
            project_id="demo-loan",
            history_text="assistant: 业务入口：`CreditController#creditNotification`",
        )
    )
    assert evidence.get("dig_in_focus") is True
    assert any(
        str(c).endswith("#synchroMemberPrice")
        for c in (evidence.get("dig_in_focus_cms") or [])
    )
    assert (evidence.get("dig_in_decision") or {}).get("source") == "retrieval_fallback"
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "主链叶子" in ans or "由浅入深" in ans or "语义相似度" in ans
    assert not re.search(r"(?m)^\s*1\.\s+`", ans)
    assert "WithdrawalsApplyServiceImpl#checkMember" not in ans


def test_fallback_dig_in_focus_prefers_search_closure_leaf(monkeypatch):
    from app.agent.pev_agent import _fallback_dig_in_focus_cms

    evidence = {
        "entry": "CreditController#creditNotification",
        "project_id": "demo-loan",
        "calls": [
            "CreditController#generateReqParam",
            "CreditServiceImpl#creditNotification",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "包括同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息到下游",
            },
        ],
    }
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            ["CreditServiceImpl#synchroMemberPrice"]
            if "creditNotification" in str(root)
            else []
        ),
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: [
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#creditNotification",
        ],
    )
    hits = _fallback_dig_in_focus_cms(
        "同步会员价格信息 具体怎么做的", evidence, project_id="demo-loan"
    )
    assert hits
    assert hits[0].endswith("#synchroMemberPrice")
    assert len(hits) == 1


def test_resolve_chinese_topic_prefers_leaf_over_parent_summary():
    """Parent+leaf both mention topic → lock only the more specific leaf."""
    from app.agent.pev_agent import _resolve_focus_method_symbol

    evidence = {
        "entry": "CreditController#creditNotification",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知：校验、同步会员价格信息、落库",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
        ],
    }
    hit = _resolve_focus_method_symbol(
        "",
        evidence,
        project_id="",
        topic="同步会员价格信息",
    )
    assert hit == ["CreditServiceImpl#synchroMemberPrice"]


def test_answer_step_cjk_anchor_locks_synchro_member_price(monkeypatch):
    """Prior step detail CJK overlap → leaf synchroMemberPrice, not parent alone.

    Production shape (desktop-chat-3dec6c): leaf has NO pre-seeded mysql summary;
    parent umbrella summary incidentally contains the topic phrase. Search∩closure
    must still promote the leaf as the single focus.
    """
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "同步会员价格信息 是做啥呢"
    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": True,
        "dig_in_likely": True,
        "spine_source": "neo4j_calls",
        "calls_count": 3,
        "project_id": "demo-loan",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#saveCredit",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，步骤含同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#saveCredit",
                "summary": "保存授信结果",
            },
            # Intentionally NO synchroMemberPrice row/summary (production dig-in
            # evidence often lacks the leaf until related enrich runs too late).
        ],
        "answer_steps": [
            {
                "index": 1,
                "title": "受理授信通知",
                "detail": "入口校验请求参数",
            },
            {
                "index": 2,
                "title": "同步会员价格信息",
                "detail": "将会员价格同步到下游系统",
                "class_method": "CreditServiceImpl#creditNotification",
            },
            {
                "index": 3,
                "title": "保存结果",
                "detail": "写库",
            },
        ],
    }

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            # LLM wrongly points at parent (desktop-chat-5109 / 3dec6c style).
            return (
                '{"dig_in":true,"focus_method":"CreditServiceImpl#creditNotification",'
                '"topic":"同步会员价格信息","reason":"subtopic"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    monkeypatch.setattr(
        pev,
        "_related_cms_under_focus",
        lambda *_a, **_k: [],
    )
    # Do not let MySQL enrich silently re-seed the leaf summary.
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)

    def _fake_closure(pid, root, *a, **k):  # noqa: ARG001
        if "creditNotification" in str(root):
            return [
                "CreditServiceImpl#synchroMemberPrice",
                "CreditServiceImpl#saveCredit",
            ]
        return []

    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        _fake_closure,
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: [
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#creditNotification",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        ],
    )

    ans = asyncio.run(
        pev._exec_summarize_flow(
            q,
            evidence,
            project_id="demo-loan",
            history_text=(
                "assistant: 业务入口：`CreditController#creditNotification`\n"
                "1. **受理授信通知**：入口校验\n"
                "2. **同步会员价格信息**：将会员价格同步到下游\n"
            ),
        )
    )
    focus = evidence.get("dig_in_focus_cms") or []
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    # Must not dual-lock parent as primary focus.
    assert focus[0].endswith("#synchroMemberPrice")
    decision = evidence.get("dig_in_decision") or {}
    assert decision.get("focus_cms") == focus
    assert decision.get("source")
    src = str(decision.get("source") or "")
    assert (
        "answer_step_anchor" in src
        or src.startswith("desc_match")
        or src.startswith("method_first")
    ), decision
    assert "主链叶子" in ans or "由浅入深" in ans or "语义相似度" in ans


def test_anchor_leaf_promotes_search_closure_without_leaf_summary(monkeypatch):
    """Unit: answer-step anchor alone — no leaf mysql summary, search∩closure wins."""
    from app.agent.pev_agent import _anchor_leaf_from_answer_steps

    evidence = {
        "entry": "CreditController#creditNotification",
        "project_id": "demo-loan",
        "calls": ["CreditServiceImpl#creditNotification"],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，步骤含同步会员价格信息",
            },
        ],
        "answer_steps": [
            {
                "title": "同步会员价格信息",
                "detail": "将会员价格同步到下游系统",
                "class_method": "CreditServiceImpl#creditNotification",
            },
        ],
    }
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            ["CreditServiceImpl#synchroMemberPrice"]
            if "creditNotification" in str(root)
            else []
        ),
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: [
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#creditNotification",
        ],
    )
    monkeypatch.setattr(
        "app.agent.pev_agent._enrich_focus_rows",
        lambda *_a, **_k: None,
    )
    hit = _anchor_leaf_from_answer_steps(
        "同步会员价格信息 具体做了什么",
        evidence,
        topic="同步会员价格信息",
        project_id="demo-loan",
    )
    assert hit == ["CreditServiceImpl#synchroMemberPrice"], hit


def test_prefer_topic_no_shortcircuit_when_anchor_equals_llm_parent(monkeypatch):
    """anchored == LLM parent must continue resolve+search promotion."""
    from app.agent.pev_agent import _prefer_topic_focus_cms

    evidence = {
        "entry": "CreditController#creditNotification",
        "project_id": "demo-loan",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，步骤含同步会员价格信息",
            },
        ],
    }
    # Force the old buggy behavior: answer-step anchor returns the umbrella parent.
    monkeypatch.setattr(
        "app.agent.pev_agent._anchor_leaf_from_answer_steps",
        lambda *_a, **_k: ["CreditServiceImpl#creditNotification"],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            ["CreditServiceImpl#synchroMemberPrice"]
            if "creditNotification" in str(root)
            else []
        ),
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: [
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#creditNotification",
        ],
    )
    monkeypatch.setattr(
        "app.agent.pev_agent._enrich_focus_rows",
        lambda *_a, **_k: None,
    )
    focus = _prefer_topic_focus_cms(
        ["CreditServiceImpl#creditNotification"],
        {
            "dig_in": True,
            "focus_method": "CreditServiceImpl#creditNotification",
            "topic": "同步会员价格信息",
            "source": "llm",
        },
        evidence,
        project_id="demo-loan",
        question="同步会员价格信息 具体做了什么",
    )
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus


def test_promote_related_leaf_over_umbrella_parent():
    """Unit: related already has synchroMemberPrice → promote over parent focus."""
    from app.agent.pev_agent import _promote_related_leaf_to_focus

    evidence = {
        "entry": "CreditController#creditNotification",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，步骤含同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
        ],
    }
    focus, related, demoted = _promote_related_leaf_to_focus(
        ["CreditServiceImpl#creditNotification"],
        [
            "CreditServiceImpl#synchroMemberPrice",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        ],
        evidence,
        question="同步会员价格信息 是做啥呢",
        topic="同步会员价格信息",
    )
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus
    assert demoted == "CreditServiceImpl#creditNotification"
    assert "CreditServiceImpl#synchroMemberPrice" not in related


def test_summarize_related_leaf_promote_fixes_022e6fc524a1(monkeypatch):
    """Production shape: llm+answer_step_anchor left parent; related had leaf.

    desktop-chat-022e6fc524a1: dig_in=true, focus=creditNotification,
    synchroMemberPrice only in related → final focus must be the leaf; render
    leads with leaf body, parent only as caller context.
    """
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "同步会员价格信息 是做啥呢"
    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": True,
        "dig_in_likely": True,
        "spine_source": "neo4j_calls",
        "calls_count": 3,
        "project_id": "demo-loan",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#saveCredit",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，步骤含同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
        ],
        "answer_steps": [
            {
                "title": "同步会员价格信息",
                "detail": "将会员价格同步到下游系统",
                "class_method": "CreditServiceImpl#creditNotification",
            },
        ],
    }

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"dig_in":true,"focus_method":"CreditServiceImpl#creditNotification",'
                '"topic":"同步会员价格信息","reason":"subtopic"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    # Keep related_leaf_promote path: skip description-first (would pick leaf alone).
    monkeypatch.setattr(
        pev, "_try_lock_dig_in_description_focus", lambda *_a, **_k: None
    )
    # Simulate prior prefer/anchor failing to promote (kept parent).
    monkeypatch.setattr(
        pev,
        "_prefer_topic_focus_cms",
        lambda focus_cms, *_a, **_k: (
            list(focus_cms)[:1] or ["CreditServiceImpl#creditNotification"]
        ),
    )
    monkeypatch.setattr(
        pev,
        "_resolve_focus_method_symbol",
        lambda *_a, **_k: ["CreditServiceImpl#creditNotification"],
    )
    # Related already found the leaf (production log shape).
    related_calls = {"n": 0}

    def _fake_related(pid, focus_cms, question, evidence, *, topic=""):  # noqa: ARG001
        related_calls["n"] += 1
        focus0 = (focus_cms or [""])[0]
        if "synchroMemberPrice" in str(focus0):
            return ["CusCbsServiceClient#synchroMemberPriceInfo"]
        return [
            "CreditServiceImpl#synchroMemberPrice",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        ]

    monkeypatch.setattr(pev, "_related_cms_under_focus", _fake_related)
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])

    ans = asyncio.run(
        pev._exec_summarize_flow(
            q,
            evidence,
            project_id="demo-loan",
            history_text=(
                "assistant: 业务入口：`CreditController#creditNotification`\n"
                "2. **同步会员价格信息**：将会员价格同步到下游\n"
            ),
        )
    )
    focus = evidence.get("dig_in_focus_cms") or []
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus
    decision = evidence.get("dig_in_decision") or {}
    assert "related_leaf_promote" in str(decision.get("source") or "")
    assert decision.get("demoted_parent") == "CreditServiceImpl#creditNotification"
    assert evidence.get("dig_in_caller") == ["CreditServiceImpl#creditNotification"]
    # Render: leaf as 主链叶子; parent only as 调用方 — never parent as 主链方法.
    assert "主链叶子 `CreditServiceImpl#synchroMemberPrice`" in ans
    assert "调用方（主链上层）：`CreditServiceImpl#creditNotification`" in ans
    assert "主链方法 `CreditServiceImpl#creditNotification`" not in ans
    # Leaf must appear before demoted parent in the answer body.
    assert ans.index("synchroMemberPrice") < ans.index("调用方（主链上层）")
    assert related_calls["n"] >= 2  # recompute under promoted leaf


def test_render_leaf_leads_parent_as_caller_only():
    from app.agent.pev_agent import _render_focus_method_answer

    evidence = {
        "entry": "CreditController#creditNotification",
        "spine_source": "neo4j_calls",
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程",
            },
            {
                "class_method": "CusCbsServiceClient#synchroMemberPriceInfo",
                "summary": "下游同步接口",
            },
        ],
    }
    ans = _render_focus_method_answer(
        evidence,
        ["CreditServiceImpl#synchroMemberPrice"],
        question="同步会员价格信息 是做啥呢",
        related_cms=["CusCbsServiceClient#synchroMemberPriceInfo"],
        caller_cms=["CreditServiceImpl#creditNotification"],
    )
    assert "主链叶子 `CreditServiceImpl#synchroMemberPrice`" in ans
    assert "调用方（主链上层）：`CreditServiceImpl#creditNotification`" in ans
    assert "主链方法" not in ans
    assert ans.index("主链叶子") < ans.index("调用方")
    # Dig-in answer describes the leaf only — no related-method dump.
    assert "相关调用" not in ans
    assert "CusCbsServiceClient#synchroMemberPriceInfo" not in ans


def test_verify_dig_in_rejects_numbered_full_chain():
    from app.agent.pev_agent import verify_contract

    evidence = _credit_dig_in_evidence()
    evidence["dig_in_focus"] = True
    evidence["dig_in_focus_cms"] = ["CreditServiceImpl#synchroMemberPrice"]
    full_chain = (
        "业务入口：`CreditController#creditApply`\n"
        "主链：\n"
        "1. `CreditController#creditApply` — 入口\n"
        "2. `CreditServiceImpl#apply` — 主处理\n"
        "3. `CreditServiceImpl#synchroMemberPrice` — 同步\n"
        "4. `CreditServiceImpl#saveCredit` — 保存\n"
    )
    v = verify_contract(
        {"goal": "explain_flow", "must": [{"id": "entry", "desc": "x"}]},
        evidence,
        full_chain,
    )
    assert v["ok"] is False
    assert any(g["id"] == "dig_in_full_chain" for g in v["gaps"])


def test_lock_entry_dig_in_fast_path_reuses_spine_no_expand(monkeypatch):
    """lock_entry dig-in: reuse spine + answer_step desc match → leaf; no expand/LLM wait."""
    import asyncio

    from app.agent import pev_agent as pev

    q = "同步会员价格信息 这一块主要是做什么呢"
    expand_calls: list[str] = []
    dig_llm_calls: list[str] = []
    goal_llm_calls: list[str] = []
    resolve_calls: list[str] = []
    mysql_calls: list[str] = []

    async def _boom_expand(*_a, **_k):
        expand_calls.append("expand")
        raise AssertionError("expand_calls must not run on dig-in reuse path")

    async def _boom_resolve(*_a, **_k):
        resolve_calls.append("resolve")
        raise AssertionError("resolve_entry must not run on dig-in reuse path")

    async def _boom_mysql(*_a, **_k):
        mysql_calls.append("mysql")
        raise AssertionError("mysql_enrich must not run on dig-in reuse path")

    async def _boom_dig_llm(*_a, **_k):
        dig_llm_calls.append("dig_llm")
        raise AssertionError("decide_dig_in_focus must not run when desc match hits")

    async def _boom_goal(*_a, **_k):
        goal_llm_calls.append("goal")
        raise AssertionError("extract_goal_contract must not run under lock dig-in compress")

    monkeypatch.setattr(pev, "_exec_expand_calls", _boom_expand)
    monkeypatch.setattr(pev, "_exec_resolve_entry", _boom_resolve)
    monkeypatch.setattr(pev, "_exec_mysql_enrich", _boom_mysql)
    monkeypatch.setattr(pev, "decide_dig_in_focus", _boom_dig_llm)
    monkeypatch.setattr(pev, "extract_goal_contract", _boom_goal)
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(pev, "_related_cms_under_focus", lambda *_a, **_k: [])
    monkeypatch.setattr(
        "app.agent.ask_retrieval._vector_search_hit_cms_for_question",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            [
                "CreditServiceImpl#synchroMemberPrice",
                "CusCbsServiceClient#synchroMemberPriceInfo",
            ]
            if "creditNotification" in str(root)
            else (
                ["CusCbsServiceClient#synchroMemberPriceInfo"]
                if "synchroMemberPrice" in str(root)
                else []
            )
        ),
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: [
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#creditNotification",
        ],
    )

    out = asyncio.run(
        pev.run_pev_pipeline(
            q,
            project_id="demo-loan",
            entry_hint="CreditController#creditNotification",
            lock_entry=True,
            dig_in_likely=True,
            flow_symbols=[
                "CreditController#creditNotification",
                "CreditServiceImpl#creditNotification",
                "CreditServiceImpl#synchroMemberPrice",
                "CreditServiceImpl#saveCredit",
            ],
            answer_steps=[
                {
                    "index": 1,
                    "title": "受理授信通知",
                    "detail": "入口校验",
                    "class_method": "CreditServiceImpl#creditNotification",
                },
                {
                    "index": 2,
                    "title": "同步会员价格信息",
                    "detail": "将会员价格同步到下游系统",
                    "class_method": "CreditServiceImpl#creditNotification",
                },
            ],
            history_text=(
                "assistant: 业务入口：`CreditController#creditNotification`\n"
                "2. **同步会员价格信息**：将会员价格同步到下游\n"
            ),
        )
    )
    ev = out.get("evidence") or {}
    assert ev.get("reuse_shared_spine") is True, ev
    assert ev.get("dig_in_focus") is True
    focus = ev.get("dig_in_focus_cms") or []
    # chain_topic may expand to shallow→deep family; leaf must be present.
    assert "CreditServiceImpl#synchroMemberPrice" in focus, focus
    decision = ev.get("dig_in_decision") or {}
    assert str(decision.get("source") or "").startswith(
        ("desc_match", "method_first")
    ) or "chain_topic" in str(decision.get("reason") or ""), decision
    assert "CreditServiceImpl#synchroMemberPrice" in (out.get("answer") or "")
    assert "主链叶子" in (out.get("answer") or "") or "由浅入深" in (out.get("answer") or "")
    assert not expand_calls, expand_calls
    assert not resolve_calls, resolve_calls
    assert not mysql_calls, mysql_calls
    assert not dig_llm_calls, dig_llm_calls
    assert not goal_llm_calls, goal_llm_calls
    # Task notes should mark reuse / skip.
    notes = " ".join(
        str(t.get("note") or "") for t in ((out.get("plan") or {}).get("tasks") or [])
    )
    assert "reuse_shared" in notes or "reuse_skip" in notes, notes


def test_summarize_first_turn_entry_scoped_dig_in_no_lock(monkeypatch):
    """First turn (no lock_entry): dig_in_likely/ask_scope → leaf, not full chain.

    desktop-chat-2ddd0073d94c shape: entry correct but answered full spine;
    with ask_scope=entry_scoped_dig_in dig-in helpers must run without session lock.
    """
    import asyncio
    import re

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "授信通知接口 同步会员价格信息 具体做了什么"
    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": False,
        "dig_in_likely": True,
        "ask_scope": "entry_scoped_dig_in",
        "focus_topic": "同步会员价格信息",
        "spine_source": "neo4j_calls",
        "calls_count": 3,
        "project_id": "demo-loan",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#saveCredit",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，步骤含同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#saveCredit",
                "summary": "保存授信结果",
            },
        ],
    }

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"dig_in":true,"focus_method":"CreditServiceImpl#synchroMemberPrice",'
                '"topic":"同步会员价格信息","reason":"first-turn entry+subtopic"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    # Force LLM dig_in path (skip desc-first) to prove gate no longer needs lock_entry.
    monkeypatch.setattr(
        pev, "_try_lock_dig_in_description_focus", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        pev,
        "_resolve_focus_method_symbol",
        lambda *_a, **_k: ["CreditServiceImpl#synchroMemberPrice"],
    )
    monkeypatch.setattr(
        pev,
        "_prefer_topic_focus_cms",
        lambda focus_cms, *_a, **_k: list(focus_cms)[:1],
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(pev, "_related_cms_under_focus", lambda *_a, **_k: [])

    ans = asyncio.run(
        pev._exec_summarize_flow(q, evidence, project_id="demo-loan", history_text="")
    )
    assert evidence.get("dig_in_focus") is True
    focus = evidence.get("dig_in_focus_cms") or []
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "主链叶子" in ans or "由浅入深" in ans or "语义相似度" in ans or "主链叶子" in ans
    assert not re.search(r"(?m)^\s*1\.\s+`", ans)
    assert "CreditServiceImpl#saveCredit" not in ans or "相关调用" in ans


def test_pipeline_first_turn_goal_ask_scope_dig_in_leaf(monkeypatch):
    """Goal LLM ask_scope=entry_scoped_dig_in → resolve+expand then leaf focus."""
    import asyncio
    import inspect
    import re

    from app.agent import pev_agent as pev

    q = "某接口 FooBar 子步骤 具体做了什么"
    # Ensure production allow-path has no Chinese product-noun gates.
    src = inspect.getsource(pev._allows_entry_scoped_dig_in)
    for banned in ("授信", "会员价格", "放款", "同步会员", "额度", "融担", "风控"):
        assert banned not in src, banned
    src_try = inspect.getsource(pev._try_lock_dig_in_description_focus)
    for banned in ("授信通知", "会员价格", "放款"):
        assert banned not in src_try, banned

    async def _fake_goal(question, *, history_text="", prior_goal=None):  # noqa: ARG001
        return {
            "goal": "explain_flow",
            "also": [],
            "must": [
                {"id": "entry", "desc": "定位入口"},
                {"id": "spine", "desc": "展开主链"},
                {"id": "summarize", "desc": "下钻叶子"},
            ],
            "ask_scope": "entry_scoped_dig_in",
            "focus_topic": "子步骤",
            "dig_in_focus": True,
            "source": "llm",
            "deliverable": "leaf method description",
        }

    async def _fake_resolve(project_id, question, evidence):  # noqa: ARG001
        evidence["entry"] = "CreditController#creditNotification"

    async def _fake_expand(project_id, evidence, question=""):  # noqa: ARG001
        evidence["calls"] = [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#saveCredit",
        ]
        evidence["calls_count"] = 3
        evidence["spine_source"] = "neo4j_calls"

    async def _fake_mysql(project_id, evidence):  # noqa: ARG001
        evidence["mysql_rows"] = [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "主流程含子步骤委托",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "子步骤实现",
            },
            {
                "class_method": "CreditServiceImpl#saveCredit",
                "summary": "保存结果",
            },
        ]

    async def _fake_dig(*_a, **_k):
        return {
            "dig_in": True,
            "focus_method": "CreditServiceImpl#synchroMemberPrice",
            "topic": "子步骤",
            "reason": "entry_scoped",
            "source": "llm",
        }

    monkeypatch.setattr(pev, "extract_goal_contract", _fake_goal)
    monkeypatch.setattr(pev, "_exec_resolve_entry", _fake_resolve)
    monkeypatch.setattr(pev, "_exec_expand_calls", _fake_expand)
    monkeypatch.setattr(pev, "_exec_mysql_enrich", _fake_mysql)
    monkeypatch.setattr(pev, "decide_dig_in_focus", _fake_dig)
    monkeypatch.setattr(
        pev, "_try_lock_dig_in_description_focus", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: ["CreditServiceImpl#synchroMemberPrice"],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            [
                "CreditServiceImpl#creditNotification",
                "CreditServiceImpl#synchroMemberPrice",
                "CreditServiceImpl#saveCredit",
            ]
            if "credit" in str(root).lower()
            else []
        ),
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(
        pev,
        "_related_cms_under_focus",
        lambda *_a, **_k: [],
    )
    # Do not force multi-intent decompose for this path.
    monkeypatch.setattr(
        "app.agent.decompose.wants_decompose", lambda *_a, **_k: False
    )

    out = asyncio.run(
        pev.run_pev_pipeline(
            q,
            project_id="demo-loan",
            lock_entry=False,
            dig_in_likely=False,
            history_text="",
        )
    )
    ev = out.get("evidence") or {}
    assert ev.get("lock_entry") is False
    assert ev.get("ask_scope") == "entry_scoped_dig_in"
    assert ev.get("dig_in_likely") is True
    assert ev.get("dig_in_focus") is True
    focus = ev.get("dig_in_focus_cms") or []
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus
    ans = out.get("answer") or ""
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert not re.search(r"(?m)^\s*1\.\s+`CreditServiceImpl#saveCredit`", ans)
    assert "主链叶子" in ans or "由浅入深" in ans or "语义相似度" in ans or "主链叶子" in ans


def test_method_first_on_chain_focuses_leaf(monkeypatch):
    """Method-first ∩ creditNotification closure → focus leaf, not full chain."""
    import asyncio
    import inspect
    import re

    from app.agent import pev_agent as pev

    q = "授信通知接口 同步会员价格信息 具体做了什么"
    # Production path must not hardcode business nouns.
    for fn in (
        pev._try_method_first_on_chain_focus,
        pev._resolve_method_topic,
        pev._entry_chain_cms_set,
        pev._allows_entry_scoped_dig_in,
    ):
        src = inspect.getsource(fn)
        for banned in ("授信通知", "会员价格", "同步会员", "放款", "额度", "融担"):
            assert banned not in src, f"{fn.__name__}: {banned}"

    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": False,
        "dig_in_likely": True,
        "ask_scope": "entry_scoped_dig_in",
        "method_topic": "同步会员价格信息",
        "focus_topic": "同步会员价格信息",
        "entry_anchor": "授信通知接口",
        "spine_source": "neo4j_calls",
        "calls_count": 3,
        "project_id": "demo-loan",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#saveCredit",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息到下游",
            },
            {
                "class_method": "CreditServiceImpl#saveCredit",
                "summary": "保存授信结果",
            },
        ],
    }

    monkeypatch.setattr(
        "app.agent.ask_retrieval._vector_search_hit_cms_for_question",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: [
            "CreditServiceImpl#synchroMemberPrice",
            "OtherService#unrelatedPrice",
        ],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            [
                "CreditServiceImpl#creditNotification",
                "CreditServiceImpl#synchroMemberPrice",
                "CreditServiceImpl#saveCredit",
            ]
            if "creditNotification" in str(root)
            else []
        ),
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(pev, "_related_cms_under_focus", lambda *_a, **_k: [])

    async def _boom_dig(*_a, **_k):
        raise AssertionError("decide_dig_in_focus must not run on method-first hit")

    monkeypatch.setattr(pev, "decide_dig_in_focus", _boom_dig)

    ans = asyncio.run(
        pev._exec_summarize_flow(q, evidence, project_id="demo-loan", history_text="")
    )
    assert evidence.get("dig_in_focus") is True
    focus = evidence.get("dig_in_focus_cms") or []
    assert focus == ["CreditServiceImpl#synchroMemberPrice"], focus
    decision = evidence.get("dig_in_decision") or {}
    assert decision.get("source") == "method_first_on_chain", decision
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "主链叶子" in ans or "由浅入深" in ans or "语义相似度" in ans
    assert not re.search(r"(?m)^\s*1\.\s+`", ans)
    assert "相关调用" not in ans
    assert "CreditServiceImpl#saveCredit" not in ans


def test_method_first_off_chain_no_false_lock(monkeypatch):
    """Method search hit outside entry closure must not lock that method."""
    from app.agent import pev_agent as pev

    q = "某接口 FooBar 子话题 具体做了什么"
    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": False,
        "dig_in_likely": True,
        "ask_scope": "entry_scoped_dig_in",
        "method_topic": "子话题",
        "focus_topic": "子话题",
        "spine_source": "neo4j_calls",
        "calls_count": 2,
        "project_id": "demo-loan",
        "calls": [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#saveCredit",
        ],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "主流程委托",
            },
            {
                "class_method": "CreditServiceImpl#saveCredit",
                "summary": "保存结果",
            },
        ],
    }

    monkeypatch.setattr(
        "app.agent.ask_retrieval._vector_search_hit_cms_for_question",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: ["UnrelatedService#totallyOffChain"],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#saveCredit",
        ],
    )

    miss = pev._try_method_first_on_chain_focus(
        q, evidence, project_id="demo-loan"
    )
    assert miss is None
    assert evidence.get("method_first_miss") is True
    assert "UnrelatedService#totallyOffChain" in (
        evidence.get("method_first_candidates") or []
    )
    assert not (evidence.get("method_first_on_chain") or [])


def test_extract_goal_llm_classifies_dig_in_vs_full_flow(monkeypatch):
    """Goal LLM (mock) splits entry+subtopic vs whole-API flow, including 主要做什么."""
    import asyncio

    import app.agent.pev_agent as pev
    from app.llm.mock import MockLLMProvider

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: MockLLMProvider())
    q = "某业务接口中 参数校验具体做了什么"
    out = asyncio.run(pev.extract_goal_contract(q))
    assert out["goal"] == "explain_flow"
    assert out.get("ask_scope") == "entry_scoped_dig_in"
    assert out.get("dig_in_focus") is True
    assert out.get("entry_anchor") == "某业务接口"
    assert "参数校验" in str(out.get("method_topic") or "")
    assert out.get("source") in {"llm", "llm+heuristic"}
    combined = "通知接口 同步价格信息 具体做了什么"
    out2 = asyncio.run(pev.extract_goal_contract(combined))
    assert out2.get("ask_scope") == "entry_scoped_dig_in"
    assert out2.get("entry_anchor") == "通知接口"
    assert "同步价格" in str(out2.get("method_topic") or "")
    colloquial = "某业务接口中  参数校验主要做什么"
    out3 = asyncio.run(pev.extract_goal_contract(colloquial))
    assert out3.get("ask_scope") == "entry_scoped_dig_in"
    assert out3.get("dig_in_focus") is True
    assert out3.get("entry_anchor") == "某业务接口"
    assert "参数校验" in str(out3.get("method_topic") or "")
    assert "主要" not in str(out3.get("method_topic") or "")
    assert out3.get("source") in {"llm", "llm+heuristic"}
    whole = "某业务接口都做了啥"
    out4 = asyncio.run(pev.extract_goal_contract(whole))
    assert out4.get("ask_scope") == "full_flow"
    assert not out4.get("dig_in_focus")


def test_pipeline_interface_flow_uses_goal_llm_full_flow(monkeypatch):
    """Plain interface flow ask: Goal LLM full_flow; resolve→expand→mysql→summarize."""
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    goal_llm_calls: list[str] = []
    path_calls: list[str] = []

    class _GoalLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            goal_llm_calls.append("goal")
            return (
                '{"goal":"explain_flow","also":[],'
                '"must":[{"id":"entry","desc":"e"},{"id":"spine","desc":"s"},'
                '{"id":"summarize","desc":"flow"}],'
                '"ask_scope":"full_flow",'
                '"entry_anchor":null,"method_topic":null,"focus_topic":null,'
                '"field_name":null,"field_label_zh":null,"tables":[],'
                '"inventory_kind":null,"deliverable":"flow"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    async def _fake_resolve(project_id, question, evidence):  # noqa: ARG001
        path_calls.append("resolve_entry")
        evidence["entry"] = "WithdrawalsController#withdrawals"

    async def _fake_expand(project_id, evidence, question=""):  # noqa: ARG001
        path_calls.append("expand_calls")
        evidence["calls"] = [
            "WithdrawalsServiceImpl#withdrawals",
            "WithdrawalsServiceImpl#checkQuota",
        ]
        evidence["calls_count"] = 2
        evidence["spine_source"] = "neo4j_calls"

    async def _fake_mysql(project_id, evidence):  # noqa: ARG001
        path_calls.append("mysql_enrich")
        evidence["mysql_rows"] = [
            {
                "order": 1,
                "class_method": "WithdrawalsController#withdrawals",
                "rel": "ENTRY",
                "summary": "入口",
            },
            {
                "order": 2,
                "class_method": "WithdrawalsServiceImpl#withdrawals",
                "rel": "CALLS",
                "summary": "主流程",
            },
        ]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _GoalLLM())
    monkeypatch.setattr(pev, "_exec_resolve_entry", _fake_resolve)
    monkeypatch.setattr(pev, "_exec_expand_calls", _fake_expand)
    monkeypatch.setattr(pev, "_exec_mysql_enrich", _fake_mysql)
    monkeypatch.setattr(
        "app.agent.decompose.wants_decompose", lambda *_a, **_k: False
    )

    q = "某接口具体做了什么"
    out = asyncio.run(
        pev.run_pev_pipeline(
            q,
            project_id="demo-loan",
            lock_entry=False,
            dig_in_likely=False,
            history_text="",
        )
    )
    assert goal_llm_calls, "Goal LLM must run for flow vs dig-in"
    assert path_calls == ["resolve_entry", "expand_calls", "mysql_enrich"], path_calls
    ev = out.get("evidence") or {}
    assert ev.get("ask_scope") == "full_flow"
    assert not ev.get("dig_in_focus")
    assert ev.get("entry") == "WithdrawalsController#withdrawals"
    assert ev.get("spine_source") == "neo4j_calls"
    assert (out.get("answer") or "").strip()
    # [PEV][TEST] footer tags are kept out of the user answer; check pev_debug.
    assert "[PEV][TEST]" not in (out.get("answer") or "")
    footer = (out.get("pev_debug") or {}).get("footer", "")
    assert "goal=explain_flow" in footer


def test_extract_goal_parses_ask_scope_entry_scoped(monkeypatch):
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"explain_flow","also":[],'
                '"must":[{"id":"entry","desc":"e"},{"id":"spine","desc":"s"},'
                '{"id":"summarize","desc":"leaf"}],'
                '"ask_scope":"entry_scoped_dig_in",'
                '"focus_topic":"同步会员价格信息",'
                '"field_name":null,"field_label_zh":null,"tables":[],'
                '"inventory_kind":null,"deliverable":"leaf desc"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())

    goal = asyncio.run(
        pev.extract_goal_contract(
            "某接口 子话题 具体做了什么",
            history_text="",
        )
    )
    assert goal.get("ask_scope") == "entry_scoped_dig_in"
    assert goal.get("dig_in_focus") is True
    assert goal.get("focus_topic") == "同步会员价格信息"
    assert goal.get("goal") == "explain_flow"
    # focus_topic alone also fills method_topic alias when LLM omits it.
    assert goal.get("method_topic") == "同步会员价格信息"


def test_extract_goal_parses_entry_anchor_method_topic(monkeypatch):
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"explain_flow","also":[],'
                '"must":[{"id":"entry","desc":"e"},{"id":"spine","desc":"s"},'
                '{"id":"summarize","desc":"leaf"}],'
                '"ask_scope":"entry_scoped_dig_in",'
                '"entry_anchor":"授信通知接口",'
                '"method_topic":"同步会员价格信息",'
                '"focus_topic":null,'
                '"field_name":null,"field_label_zh":null,"tables":[],'
                '"inventory_kind":null,"deliverable":"leaf desc"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())

    goal = asyncio.run(
        pev.extract_goal_contract(
            "授信通知接口 同步会员价格信息 具体做了什么",
            history_text="",
        )
    )
    assert goal.get("ask_scope") == "entry_scoped_dig_in"
    assert goal.get("entry_anchor") == "授信通知接口"
    assert goal.get("method_topic") == "同步会员价格信息"
    assert goal.get("focus_topic") == "同步会员价格信息"
    assert goal.get("dig_in_focus") is True


def test_method_first_uses_vector_search_not_hybrid(monkeypatch):
    """T4 method-first prefers Milvus vector hits over hybrid LIKE."""
    import asyncio

    from app.agent import pev_agent as pev

    q = "授信通知接口 同步会员价格信息 具体做了什么"
    evidence = {
        "entry": "CreditController#creditNotification",
        "lock_entry": False,
        "dig_in_likely": True,
        "ask_scope": "entry_scoped_dig_in",
        "method_topic": "同步会员价格信息",
        "focus_topic": "同步会员价格信息",
        "spine_source": "neo4j_calls",
        "calls_count": 2,
        "project_id": "demo-loan",
        "calls": ["CreditServiceImpl#creditNotification"],
        "mysql_rows": [
            {
                "class_method": "CreditServiceImpl#creditNotification",
                "summary": "授信通知主流程，包括同步会员价格信息",
            },
            {
                "class_method": "CreditServiceImpl#synchroMemberPrice",
                "summary": "同步会员价格信息到下游",
            },
        ],
    }

    monkeypatch.setattr(
        "app.agent.ask_retrieval._vector_search_hit_cms_for_question",
        lambda *_a, **_k: ["CreditServiceImpl#synchroMemberPrice"],
    )

    def _boom_hybrid(*_a, **_k):
        raise AssertionError("hybrid search must not run when vector hits exist")

    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        _boom_hybrid,
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: [
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
        ],
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(pev, "_related_cms_under_focus", lambda *_a, **_k: [])

    async def _boom_dig(*_a, **_k):
        raise AssertionError("decide_dig_in_focus must not run on vector method-first")

    monkeypatch.setattr(pev, "decide_dig_in_focus", _boom_dig)

    ans = asyncio.run(
        pev._exec_summarize_flow(q, evidence, project_id="demo-loan", history_text="")
    )
    assert evidence.get("dig_in_focus_cms") == [
        "CreditServiceImpl#synchroMemberPrice"
    ]
    decision = evidence.get("dig_in_decision") or {}
    assert decision.get("search") == "vector"
    assert evidence.get("method_first_backend") == "vector"
    assert "CreditServiceImpl#synchroMemberPrice" in ans
    assert "相关调用" not in ans
    assert "CreditServiceImpl#creditNotification" not in ans


def test_desc_match_does_not_lock_off_chain_client(monkeypatch):
    """desktop-chat-fbfa4fd75ba1: AccClient#deleteLimitV2 must not steal 同步会员价格."""
    import asyncio

    from app.agent import pev_agent as pev

    q = "授信通知接口 同步会员价格信息 具体做了什么"
    evidence = {
        "entry": "CreditQuotaController#creditNotification",
        "lock_entry": False,
        "dig_in_likely": True,
        "ask_scope": "entry_scoped_dig_in",
        "method_topic": "同步会员价格信息",
        "focus_topic": "同步会员价格信息",
        "spine_source": "neo4j_calls",
        "calls_count": 3,
        "project_id": "demo-loan",
        "calls": [
            "CreditQuotaController#generateReqParam",
            "WithdrawalsApplyServiceImpl#checkMember",
            "CreditQuotaShareServiceImpl#creditQuotaShare",
        ],
        "mysql_rows": [
            {
                "class_method": "WithdrawalsApplyServiceImpl#checkMember",
                "summary": "检查会员是否存在",
            },
            {
                "class_method": "CreditQuotaShareServiceImpl#creditQuotaShare",
                "summary": "额度分享",
            },
        ],
    }

    monkeypatch.setattr(
        "app.agent.ask_retrieval._vector_search_hit_cms_for_question",
        lambda *_a, **_k: [],
    )

    def _search(_pid, question, top_k=12):  # noqa: ARG001
        qn = str(question or "")
        if "同步会员价格" in qn:
            return ["CreditServiceImpl#synchroMemberPrice"]
        return ["AccClient#deleteLimitV2", "WithdrawalsApplyServiceImpl#checkMember"]

    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        _search,
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: [
            "CreditQuotaController#generateReqParam",
            "WithdrawalsApplyServiceImpl#checkMember",
            "CreditQuotaShareServiceImpl#creditQuotaShare",
            "AccClient#deleteLimitV2",
        ],
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(pev, "_related_cms_under_focus", lambda *_a, **_k: [])

    async def _no_dig(*_a, **_k):
        return {
            "dig_in": False,
            "focus_method": "",
            "topic": "同步会员价格信息",
            "reason": "no confident leaf",
            "source": "llm",
        }

    monkeypatch.setattr(pev, "decide_dig_in_focus", _no_dig)

    ans = asyncio.run(
        pev._exec_summarize_flow(q, evidence, project_id="demo-loan", history_text="")
    )
    focus = evidence.get("dig_in_focus_cms") or []
    assert "AccClient#deleteLimitV2" not in focus, focus
    assert "deleteLimitV2" not in (ans or "")
    decision = evidence.get("dig_in_decision") or {}
    assert "deleteLimitV2" not in str(decision.get("focus_method") or "")


def test_collect_topic_cms_on_neo4j_chain_not_same_file_prefix():
    """Topic phrase in on-chain summaries (any class), not same-file name prefix."""
    from app.agent.pev_agent import _collect_topic_cms_on_chain

    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "calls": [
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsServiceImpl#insertLoanData",
        ],
        "mysql_rows": [
            {
                "class_method": "WithdrawalsServiceImpl#loan",
                "summary": "该方法处理贷款申请的核心流程，包括担保信息查询、客户信息同步、设备信息保存、额度校验、机构限额检查等关键步骤，最终完成贷款数据的入库。",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheck",
                "summary": "按产品路由额度校验",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheckMIY",
                "summary": "MIY 产品额度校验",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheckCKHB01",
                "summary": "CKHB01 产品额度校验",
            },
            {
                "class_method": "LimitClient#queryQuota",
                "summary": "查询可用额度",
            },
            {
                "class_method": "WithdrawalsServiceImpl#checkApplyReqParam",
                "summary": "校验申请入参非空",
            },
            {
                "class_method": "AccServiceImpl#limitAdjustment",
                "summary": "信用额度冻结解冻调整",
            },
            {
                "class_method": "LimitAmountServiceImpl#amountLimitCheck",
                "summary": "机构和二级门店限额校验",
            },
        ],
    }
    chain = {
        "WithdrawalsServiceImpl#loan",
        "WithdrawalsServiceImpl#insertLoanData",
        "WithdrawalsServiceImpl#quotaCheck",
        "WithdrawalsServiceImpl#quotaCheckMIY",
        "WithdrawalsServiceImpl#quotaCheckCKHB01",
        "LimitClient#queryQuota",
        "WithdrawalsServiceImpl#checkApplyReqParam",
        "AccServiceImpl#limitAdjustment",
        "CreditServiceClient#limitAdjustment",
        "LimitAmountServiceImpl#amountLimitCheck",
        "LimitAmountServiceImpl#agencyLimitRepayment",
        "LimitQueryServiceClient#limitList",
    }
    hits = _collect_topic_cms_on_chain(
        "额度校验",
        chain,
        evidence,
        entry="WithdrawalsController#withdrawals",
        search_rank={"WithdrawalsServiceImpl#quotaCheckCKHB01": 0},
        question="放款接口中额度校验主要做什么",
    )
    assert "WithdrawalsServiceImpl#quotaCheck" in hits
    assert "WithdrawalsServiceImpl#quotaCheckMIY" in hits
    assert "WithdrawalsServiceImpl#quotaCheckCKHB01" in hits
    # Different class is kept when the summary is tightly about the topic.
    assert "AccServiceImpl#limitAdjustment" not in hits
    assert "CreditServiceClient#limitAdjustment" not in hits
    assert "LimitAmountServiceImpl#amountLimitCheck" not in hits
    assert "LimitAmountServiceImpl#agencyLimitRepayment" not in hits
    assert "LimitQueryServiceClient#limitList" not in hits
    assert "WithdrawalsServiceImpl#loan" not in hits
    assert "WithdrawalsServiceImpl#insertLoanData" not in hits
    assert "WithdrawalsServiceImpl#checkApplyReqParam" not in hits
    lone = _collect_topic_cms_on_chain(
        "同步会员价格信息",
        {
            "CreditServiceImpl#creditNotification",
            "CreditServiceImpl#synchroMemberPrice",
            "CreditServiceImpl#saveCredit",
            "CusCbsServiceClient#synchroMemberPriceInfo",
        },
        {
            "entry": "CreditController#creditNotification",
            "calls": ["CreditServiceImpl#creditNotification"],
            "mysql_rows": [
                {
                    "class_method": "CreditServiceImpl#creditNotification",
                    "summary": "授信通知主流程，步骤含同步会员价格信息",
                },
                {
                    "class_method": "CreditServiceImpl#synchroMemberPrice",
                    "summary": "同步会员价格信息到下游",
                },
                {
                    "class_method": "CreditServiceImpl#saveCredit",
                    "summary": "保存授信结果",
                },
            ],
        },
        entry="CreditController#creditNotification",
        search_rank={"CreditServiceImpl#synchroMemberPrice": 0},
        question="授信通知接口 同步会员价格信息 具体做了什么",
    )
    # CJK-only topic has no method-name mapping → do not multi-lock; single leaf stays.
    assert lone == []


def test_order_focus_shallow_to_deep_prefix_router_first():
    from app.agent.pev_agent import (
        _order_focus_shallow_to_deep,
        _render_focus_method_answer,
    )

    ordered = _order_focus_shallow_to_deep(
        [
            "WithdrawalsServiceImpl#quotaCheckMIY",
            "WithdrawalsServiceImpl#quotaCheckCKHB01",
            "WithdrawalsServiceImpl#quotaCheck",
        ]
    )
    assert ordered[0] == "WithdrawalsServiceImpl#quotaCheck"
    assert set(ordered[1:]) == {
        "WithdrawalsServiceImpl#quotaCheckMIY",
        "WithdrawalsServiceImpl#quotaCheckCKHB01",
    }
    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "spine_source": "neo4j_calls",
        "method_topic": "额度校验",
        "dig_in_decision": {"topic": "额度校验"},
        "mysql_rows": [
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheck",
                "summary": "按产品路由额度校验",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheckMIY",
                "summary": "MIY 产品额度校验",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheckCKHB01",
                "summary": "CKHB01 产品额度校验",
            },
        ],
    }
    ans = _render_focus_method_answer(
        evidence,
        [
            "WithdrawalsServiceImpl#quotaCheckMIY",
            "WithdrawalsServiceImpl#quotaCheckCKHB01",
            "WithdrawalsServiceImpl#quotaCheck",
        ],
        question="放款接口中额度校验主要做什么",
    )
    assert "由浅入深" in ans
    assert "语义相似度" in ans
    assert ans.index("quotaCheck`") < ans.index("quotaCheckMIY")
    assert "浅·路由总览" in ans
    assert "深·细分实现" in ans
    assert "聚焦方法数=3" in ans


def test_method_first_collects_quota_topic_on_neo4j_chain(monkeypatch):
    """desktop-chat-f163224596ff: 额度校验 = all on-chain topic methods, not one suffix."""
    import asyncio

    from app.agent import pev_agent as pev

    q = "放款接口中额度校验主要做什么"
    evidence = {
        "entry": "WithdrawalsController#withdrawals",
        "lock_entry": False,
        "dig_in_likely": True,
        "ask_scope": "entry_scoped_dig_in",
        "method_topic": "额度校验",
        "focus_topic": "额度校验",
        "entry_anchor": "放款接口",
        "spine_source": "neo4j_calls",
        "calls_count": 2,
        "project_id": "demo-loan",
        "calls": [
            "WithdrawalsServiceImpl#loan",
            "WithdrawalsServiceImpl#insertLoanData",
        ],
        "mysql_rows": [
            {
                "class_method": "WithdrawalsServiceImpl#loan",
                "summary": "该方法处理贷款申请的核心流程，包括担保信息查询、客户信息同步、设备信息保存、额度校验、机构限额检查等关键步骤，最终完成贷款数据的入库。",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheck",
                "summary": "按产品路由额度校验",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheckMIY",
                "summary": "MIY 产品额度校验",
            },
            {
                "class_method": "WithdrawalsServiceImpl#quotaCheckCKHB01",
                "summary": "CKHB01 产品额度校验",
            },
            {
                "class_method": "LimitClient#queryQuota",
                "summary": "查询可用额度",
            },
        ],
    }

    monkeypatch.setattr(
        "app.agent.ask_retrieval._vector_search_hit_cms_for_question",
        lambda *_a, **_k: ["WithdrawalsServiceImpl#quotaCheckCKHB01"],
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._search_hit_cms_for_question",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("hybrid must not run when vector hits exist")
        ),
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval._calls_closure_cms",
        lambda _pid, root: (
            [
                "WithdrawalsServiceImpl#loan",
                "WithdrawalsServiceImpl#quotaCheck",
                "WithdrawalsServiceImpl#quotaCheckMIY",
                "WithdrawalsServiceImpl#quotaCheckCKHB01",
                "LimitClient#queryQuota",
                "AccServiceImpl#limitAdjustment",
                "LimitAmountServiceImpl#amountLimitCheck",
                "LimitQueryServiceClient#limitList",
            ]
            if "withdrawals" in str(root).lower()
            else []
        ),
    )
    monkeypatch.setattr(pev, "_enrich_focus_rows", lambda *_a, **_k: None)
    monkeypatch.setattr(pev, "_render_focus_source_notes", lambda *_a, **_k: [])
    monkeypatch.setattr(pev, "_related_cms_under_focus", lambda *_a, **_k: [])

    async def _boom_dig(*_a, **_k):
        raise AssertionError("decide_dig_in_focus must not run on method-first hit")

    monkeypatch.setattr(pev, "decide_dig_in_focus", _boom_dig)

    ans = asyncio.run(
        pev._exec_summarize_flow(q, evidence, project_id="demo-loan", history_text="")
    )
    focus = evidence.get("dig_in_focus_cms") or []
    assert "WithdrawalsServiceImpl#quotaCheck" in focus
    assert "WithdrawalsServiceImpl#quotaCheckMIY" in focus
    assert "WithdrawalsServiceImpl#quotaCheckCKHB01" in focus
    # Callees whose summaries do not contain the topic phrase stay out
    # (no CJK→English product glossary).
    assert "LimitClient#queryQuota" not in focus
    assert "AccServiceImpl#limitAdjustment" not in focus
    assert "LimitAmountServiceImpl#amountLimitCheck" not in focus
    assert "LimitQueryServiceClient#limitList" not in focus
    assert "WithdrawalsServiceImpl#loan" not in focus
    decision = evidence.get("dig_in_decision") or {}
    assert "chain_topic" in str(decision.get("reason") or "")
    assert "stem_family" not in str(decision.get("reason") or "")
    assert "quotaCheck" in ans
    assert "quotaCheckMIY" in ans
    assert "quotaCheckCKHB01" in ans
    assert "queryQuota" not in ans
    assert "聚焦方法数=3" in ans
    assert "由浅入深" in ans
    assert focus[0] == "WithdrawalsServiceImpl#quotaCheck"


def test_extract_goal_parses_db_lookup_view_kind(monkeypatch):
    """Goal LLM owns db_lookup; maps to compose_code (synthesize, not paste)."""
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "给我一段通过借据号查询放款表的java代码"

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"compose_code","view_kind":"db_lookup",'
                '"lookup_key":"借据号","lookup_table":"放款表",'
                '"lookup_steps":[],'
                '"must":[{"id":"compose","desc":"生成查库代码"}],'
                '"deliverable":"按借据号查放款表的 Java 片段"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(pev.extract_goal_contract(q))
    assert out["goal"] == "compose_code"
    assert out["view_kind"] == "db_lookup"
    assert out["lookup_key"] == "借据号"
    assert out["lookup_table"] == "放款表"
    tools = [t["tool"] for t in pev.build_task_plan(out)]
    assert tools == ["compose_code"]


def test_extract_goal_parses_db_pipeline_steps(monkeypatch):
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "先查放款表再查还款计划再按产品分组的代码"

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"compose_code","view_kind":"db_pipeline",'
                '"lookup_key":null,"lookup_table":null,'
                '"lookup_steps":['
                '{"op":"select","table":"放款表","key":"借据号"},'
                '{"op":"select","table":"还款计划","key_from":"放款id"},'
                '{"op":"group_by","fields":["产品"],"agg":["sum金额"]}'
                "],"
                '"must":[{"id":"compose","desc":"生成多步查库"}],'
                '"deliverable":"多步查库代码"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(pev.extract_goal_contract(q))
    assert out["goal"] == "compose_code"
    assert out["view_kind"] == "db_pipeline"
    assert len(out.get("lookup_steps") or []) == 3
    assert out["lookup_steps"][0]["table"] == "放款表"
    assert out["lookup_steps"][2]["op"] == "group_by"


def test_extract_goal_show_code_plus_db_upgrades_to_compose(monkeypatch):
    """Legacy show_code+db_* from Goal → compose_code."""
    import asyncio

    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"show_code","view_kind":"db_pipeline",'
                '"lookup_key":"借据号","lookup_table":"放款表",'
                '"lookup_steps":[{"op":"select","table":"放款表","key":"借据号"},'
                '{"op":"select","table":"放款助贷表"}],'
                '"must":[{"id":"x","desc":"y"}],"deliverable":"代码"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(
        pev.extract_goal_contract(
            "给我一段代码 根据借据号查询放款表 再查助贷表"
        )
    )
    assert out["goal"] == "compose_code"
    assert out["view_kind"] == "db_pipeline"


def test_verify_show_code_db_lookup_skips_prefer_target():
    from app.agent.pev_agent import verify_contract

    goal = {
        "goal": "show_code",
        "view_kind": "db_lookup",
        "must": [{"id": "paste", "desc": "x"}],
    }
    evidence = {
        "prefer_cms": ["RepayController#queryCancelWithdrawals"],
        "view_hint": "RepayController#queryCancelWithdrawals",
        "source_view": {
            "view_kind": "db_lookup",
            "target": "WithdrawalsServiceImpl#queryProductByDealRefNO",
            "plan": {
                "class_method": "WithdrawalsServiceImpl#queryProductByDealRefNO"
            },
        },
    }
    ans = (
        "`WithdrawalsServiceImpl#queryProductByDealRefNO` 方法源码如下：\n"
        "```java\nreturn mapper.selectOne(...);\n```"
    )
    v = verify_contract(goal, evidence, ans)
    assert v["ok"] is True
    assert not any(g["id"] == "target_method" for g in v.get("gaps") or [])


def test_verify_compose_code_requires_fence():
    from app.agent.pev_agent import verify_contract

    goal = {
        "goal": "compose_code",
        "view_kind": "db_pipeline",
        "lookup_steps": [{"op": "select"}, {"op": "select"}],
        "must": [{"id": "compose", "desc": "x"}],
    }
    v = verify_contract(goal, {}, "只有说明没有代码")
    assert v["ok"] is False
    assert any(g["id"] == "compose_paste" for g in v.get("gaps") or [])
    v2 = verify_contract(
        goal,
        {"compose_code": {"refs": ["WithdrawalsServiceImpl#query"]}},
        (
            "基于证据 `WithdrawalsServiceImpl#query` 生成多步查库示意：\n"
            "步骤1查放款表，步骤2再查助贷表。\n"
            "```java\n"
            "Withdrawals w = service.getOne(qw.eq(\"deal_ref_no\", no));\n"
            "if (w != null) { assistance.list(qw2.eq(\"deal_ref_no\", no)); }\n"
            "```\n"
        ),
    )
    assert v2["ok"] is True


def test_business_steps_fallback_extracts_step_order():
    """LLM 未填 business_steps 时，从用户原话按「先/然后/再」确定性提取步骤。"""
    from app.agent.pev_plan import _extract_business_steps_fallback

    q = (
        "根据知识库商品查询文档生成 Java 接口代码，先进行参数校验，"
        "然后在service中先查询放款表，然后调用acc-service的接口，"
        "再将其结果都打印出来，然后返回"
    )
    steps = _extract_business_steps_fallback(q)
    ops = [s["op"] for s in steps]
    assert ops == ["validate", "query", "call", "log", "return"], ops
    # 引导句「根据…生成接口代码」与位置短语「在service中」不应成为独立步骤
    assert not any("生成" in s.get("target") for s in steps)
    # call 步骤 target 保留服务名（acc-service 是目标服务，不是位置短语）
    call_step = next(s for s in steps if s["op"] == "call")
    assert "acc-service" in call_step["target"]
    # 除 call 步骤外，其余步骤 target 不应是纯位置短语（如「在service中」）
    for s in steps:
        if s["op"] != "call":
            assert "service" not in s["target"].lower(), s


def test_business_steps_fallback_skips_when_no_flow():
    """无步骤编排的普通 add_api 提问不产生 business_steps。"""
    from app.agent.pev_plan import _extract_business_steps_fallback

    assert _extract_business_steps_fallback("根据知识库商品查询文档生成 Java 接口代码") == []
    assert _extract_business_steps_fallback("按知识库的登录文档新增接口") == []


# ────────────────────────────────────────────────────
# 以下为三维修复对应的新增测试
#   Fix#1 — _exec_expand_calls 同 Controller 类防护
#   Fix#2 — _topic_ground_entry_candidates index_hits 兜底
#   Fix#3 — explicit_anchor CJK 精度 + allow_override cur_score 低值例外
# ────────────────────────────────────────────────────


def test_topic_ground_falls_back_to_index_hits_when_primary_empty(monkeypatch):
    """Fix#2: _topic_ground_entry_candidates 当 primary_hits / source_hits
    都不含 Controller 类时，应回退到 index_hits 补充 Controller 候选。

    模拟话题探针返回不含 Controller 的 primary_hits 与 source_hits，
    但 index_hits 含有 WithdrawalsController + AmountController。
    """
    import app.agent.pev_flow as flow

    probe_called = []

    def _fake_probe(pid: str, q: str) -> dict:
        probe_called.append(q)
        return {
            "exists": True,
            "phrase": "放款",
            "primary_hits": [
                {"path": "amount/model/AddDayQuotaVO.java", "class": "AddDayQuotaVO",
                 "role": "data", "score": 70},
            ],
            "source_hits": [
                {"path": "amount/model/AddDayQuotaVO.java", "class": "AddDayQuotaVO",
                 "role": "data", "score": 70},
            ],
            "index_hits": [
                {"fqcn": "com.demo.controller.WithdrawalsController",
                 "class": "WithdrawalsController", "summary": "放款业务入口控制器",
                 "role": "impl", "score": 100},
                {"fqcn": "com.demo.controller.AmountController",
                 "class": "AmountController", "summary": "额度管理",
                 "role": "impl", "score": 100},
            ],
            "secondary_hits": [],
        }

    monkeypatch.setattr(
        "app.agent.codebase_survey.run_topic_existence_probe", _fake_probe
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.needs_topic_entry_grounding", lambda q: True
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.extract_flow_topic_phrase", lambda q: "放款"
    )

    result = flow._topic_ground_entry_candidates("demo-loan", "放款接口什么流程")

    assert "controllers" in result
    # primary_hits / source_hits 没有 Controller → 从 index_hits 回退
    assert "WithdrawalsController" in result["controllers"], (
        f"index_hits 中的 WithdrawalsController 应被加入 controllers，"
        f"实际得到: {result['controllers']}"
    )
    assert "AmountController" in result["controllers"], (
        f"index_hits 中的 AmountController 应被加入 controllers，"
        f"实际得到: {result['controllers']}"
    )
    assert len(result["controllers"]) >= 2
    assert probe_called == ["有放款业务吗"]


def test_topic_ground_returns_empty_when_no_hits_at_all(monkeypatch):
    """Fix#2 边界: primary_hits / source_hits / index_hits 全空时
    controllers 应为空列表（不发生崩溃也不会错误地加入无关类）。"""
    import app.agent.pev_flow as flow

    def _empty_probe(pid: str, q: str) -> dict:
        return {
            "exists": False, "phrase": "放款",
            "primary_hits": [], "source_hits": [],
            "index_hits": [], "secondary_hits": [],
        }

    monkeypatch.setattr(
        "app.agent.codebase_survey.run_topic_existence_probe", _empty_probe
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.needs_topic_entry_grounding", lambda q: True
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.extract_flow_topic_phrase", lambda q: "放款"
    )

    result = flow._topic_ground_entry_candidates("demo-loan", "放款接口什么流程")
    assert result["controllers"] == []
    assert result["phrase"] == "放款"


def test_expand_calls_same_controller_guard_blocks_cross_class_overwrite(
    monkeypatch,
):
    """Fix#1: _exec_expand_calls 中 spine 恢复遇到不同 Controller 类
    （AmountController → WithdrawalsController）时不应覆盖入口。"""
    import app.agent.pev_flow as flow

    evidence: dict = {
        "question": "放款接口什么流程",
        "entry": "AmountController#addDayQuota",
        "entry_matches": ["AmountController#addDayQuota"],
        "calls": ["DayQuotaServiceImpl#add"],
        "calls_count": 1,
        "project_id": "demo-loan",
    }

    calls_trace = []

    def _fake_resolve_spine(pid, cm):
        calls_trace.append(cm)
        if "WithdrawalsController" in cm:
            return (["WithdrawalsController#checkRetryLoan",
                     "WithdrawalsServiceImpl#retryLoan"], "neo4j_calls")
        if "AmountController" in cm:
            return (["DayQuotaServiceImpl#add"], "neo4j_calls")
        return ([], "")

    def _fake_symbol_search(pid, q, min_score=0.8):
        return ("WithdrawalsController#retryLoan", 2.5)

    monkeypatch.setattr(
        "app.agent.pev_flow._resolve_flow_spine", _fake_resolve_spine
    )
    monkeypatch.setattr(
        "app.agent.pev_flow._resolve_entry_via_symbol_search", _fake_symbol_search
    )

    import asyncio

    asyncio.run(flow._exec_expand_calls("demo-loan", evidence, "放款接口什么流程"))

    # 入口应保持为 AmountController#addDayQuota（不跨 Controller 覆盖）
    assert evidence["entry"] == "AmountController#addDayQuota", (
        f"入口不应被跨 Controller 覆盖，实际得到: {evidence['entry']}"
    )
    assert evidence["calls"] == ["DayQuotaServiceImpl#add"], (
        f"calls 应从 AmountController 的 spine 来，实际得到: {evidence['calls']}"
    )
    # _resolve_flow_spine 只被 AmountController 调用过
    assert all("AmountController" in cm for cm in calls_trace), (
        f"预期只解析 AmountController 的 spine，实际 trace: {calls_trace}"
    )


def test_expand_calls_same_controller_allows_same_class_overwrite(monkeypatch):
    """Fix#1 正向: 同一 Controller 类内方法（AmountController#addDayQuota →
    AmountController#quotaAdjust）允许 spine 恢复覆盖。"""
    import app.agent.pev_flow as flow

    evidence: dict = {
        "question": "放款接口什么流程",
        "entry": "AmountController#addDayQuota",
        "entry_matches": ["AmountController#addDayQuota"],
        "calls": ["DayQuotaServiceImpl#add"],
        "calls_count": 1,
        "project_id": "demo-loan",
    }

    def _fake_spine(pid, cm):
        if "AmountController#quotaAdjust" in cm:
            return (["AmountController#checkParam",
                     "QuotaServiceImpl#adjust",
                     "QuotaServiceImpl#save"], "neo4j_calls")
        if "AmountController#addDayQuota" in cm:
            return (["DayQuotaServiceImpl#add"], "neo4j_calls")
        return ([], "")

    def _fake_search(pid, q, min_score=0.8):
        # 同一 Controller 类：AmountController#quotaAdjust
        return ("AmountController#quotaAdjust", 2.5)

    monkeypatch.setattr(
        "app.agent.pev_flow._resolve_flow_spine", _fake_spine
    )
    monkeypatch.setattr(
        "app.agent.pev_flow._resolve_entry_via_symbol_search", _fake_search
    )

    import asyncio

    asyncio.run(flow._exec_expand_calls("demo-loan", evidence, "放款接口什么流程"))

    # 同一 Controller 类（AmountController）→ 允许覆盖
    assert evidence["entry"] == "AmountController#quotaAdjust", (
        f"同一 Controller 类应允许 spine 恢复覆盖，实际得到: {evidence['entry']}"
    )
    assert evidence["calls_count"] == 3, (
        f"预期 spine 恢复为 3 个节点，实际: {evidence['calls_count']}"
    )


def test_explicit_anchor_regex_not_triggered_by_cjk_prefixed_interface(monkeypatch):
    """Fix#3 正则: 「放款接口什么流程」中「接口」前是 CJK「放款」，
    不匹配 explicit_anchor 模式（(?<![\\u4e00-\\u9fff]) 负向后顾排除 CJK
    前缀的「接口」与「入口」）。"""
    import re

    pattern = re.compile(
        r"#"
        r"|[A-Za-z][A-Za-z0-9_]*(?:Controller|Service|Mapper|Client|Listener)\b"
        r"|(?:(?<=^)|(?<![\u4e00-\u9fff]))(?:接口|入口)(?![\u4e00-\u9fff])"
    )

    # CJK 前缀 → 不匹配
    assert not pattern.search("放款接口什么流程"), (
        "「放款接口什么流程」中「接口」前是 CJK「放款」，不应触发 explicit_anchor"
    )
    assert not pattern.search("放款业务流程入口在哪个Controller"), (
        "「入口」前是 CJK「流程」，不应触发 explicit_anchor"
    )

    # 拉丁标识符前缀 → 匹配
    assert pattern.search("AmountController接口"), (
        "「AmountController接口」中「接口」前是拉丁类名，应触发 explicit_anchor"
    )
    assert pattern.search("WithdrawalsController入口"), (
        "「WithdrawalsController入口」中「入口」前是拉丁类名，应触发 explicit_anchor"
    )

    # # / Controller/Service 类名原样匹配
    assert pattern.search("AmountController#addDayQuota")
    assert pattern.search("AmountController")


def test_resolve_entry_wrong_controller_overrides_despite_topic_grounding(
    monkeypatch,
):
    """Fix#4: 话题接地把「放款」锚到 AmountController（字面量命中「放款额」）
    时，符号搜索给出的其它 Controller 入口仍应跨类覆盖 —— 之前
    `if grounded.get("controllers"): sym, sym_score = "", 0.0` 无条件清空
    sym，导致 is_wrong_controller 修正分支永远不可达（desktop-chat-016f43043466）。"""
    import app.agent.pev_flow as flow

    evidence: dict = {
        "question": "放款接口主要做了什么",
        "entry_matches": ["AmountController#addDayQuota"],
        "entry_match_meta": [],
    }

    def _fake_topic_ground(pid, q):
        # 话题探针字面量命中「放款额」→ 只锚到 AmountController
        return {"phrase": "放款", "controllers": ["AmountController"], "primary": []}

    def _fake_resolve_entries(pid, q, entry_hint=None, prefer_hint=False):
        return (["AmountController#addDayQuota"], [{"cm": "AmountController#addDayQuota"}])

    def _fake_apply_topic_ground(pid, q, matches, meta, grounded):
        # 接地 Controller 过滤 → 只剩 AmountController
        return (matches, meta)

    def _fake_symbol_search(pid, q, min_score=1.2):
        # 符号搜索给出真实放款入口（即使分数不高也应被采用）
        return ("WithdrawalsController#withdrawals", 1.5)

    def _fake_entry_is_real(pid, cm):
        return True

    monkeypatch.setattr(
        "app.agent.pev_flow._topic_ground_entry_candidates", _fake_topic_ground
    )
    monkeypatch.setattr(
        "app.agent.ask_retrieval.resolve_flow_entries_with_meta", _fake_resolve_entries
    )
    monkeypatch.setattr(
        "app.agent.pev_flow._apply_topic_ground_to_entries", _fake_apply_topic_ground
    )
    monkeypatch.setattr(
        "app.agent.pev_flow._resolve_entry_via_symbol_search", _fake_symbol_search
    )
    monkeypatch.setattr("app.agent.pev_flow._entry_is_real", _fake_entry_is_real)

    import asyncio

    asyncio.run(flow._exec_resolve_entry("demo-loan", "放款接口主要做了什么", evidence))

    assert evidence["entry"] == "WithdrawalsController#withdrawals", (
        f"放款问题不应停留在 AmountController，实际得到: {evidence['entry']}"
    )
    assert evidence["entry_source"] == "symbol_search_wrong_controller_fix"
    assert evidence["entry_matches"] == ["WithdrawalsController#withdrawals"]
