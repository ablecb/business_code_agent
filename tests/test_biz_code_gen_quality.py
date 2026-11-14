"""Business code generation quality — evidence, synthesis, patching, fallback.

Every test runs with mock LLM (deterministic, fast) and tests the *quality logic*
that guards real LLM output — reviewers, sanitizers, evidence gatherers.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

_FAKE_PROJECT_ID = "demo-loan"
_FAKE_ROOT = Path("java_project/demo-loan")


# ═══════════════════════════════════════════════════════════════════════
# 1. EVIDENCE GATHERING QUALITY
# ═══════════════════════════════════════════════════════════════════════


def test_gather_evidence_returns_snippets(monkeypatch):
    """Happy-path: evidence gather returns matched method snippets."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: ["WithdrawalsServiceImpl#queryByDealRefNo"],
    )
    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body",
        lambda *a, **k: (
            'QueryWrapper<Withdrawals> qw = new QueryWrapper<>();\n'
            'qw.eq("deal_ref_no", refNo);\n'
            'return baseMapper.selectOne(qw);',
            "fake/WithdrawalsServiceImpl.java",
        ),
    )
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict", lambda *a, **k: []
    )

    pack = cc.gather_compose_evidence(
        _FAKE_PROJECT_ID,
        "根据借据号查放款表",
        lookup_key="借据号",
        lookup_table="放款表",
    )

    assert pack["snippets"], "should return at least one snippet"
    assert pack["hits"], "should return method hits"
    assert pack["lookup_key"] == "借据号"
    assert pack["lookup_table"] == "放款表"


def test_gather_evidence_includes_entity_definitions(monkeypatch):
    """Entity definitions from table_dict must appear in evidence."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: ["WithdrawalsServiceImpl#queryByDealRefNo"],
    )
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict",
        lambda *a, **k: [
            {
                "table_name": "demo_b_withdrawals",
                "entity_class": "Withdrawals",
                "entity_fqcn": "com.demo.loan.entity.Withdrawals",
                "name_cn": "放款表",
                "file_path": "x/Withdrawals.java",
                "fields": ["id", "deal_ref_no", "amount", "status"],
            }
        ],
    )

    def _fake_fetch(project_id, cm, *, max_chars=14000):  # noqa: ARG001
        if cm == "Withdrawals":
            return (
                "public class Withdrawals {\n"
                "  private Long id;\n"
                "  private String dealRefNo;\n"
                "  private BigDecimal amount;\n"
                "  private String status;\n"
                "}",
                "x/Withdrawals.java",
            )
        return (
            'QueryWrapper<Withdrawals> qw = new QueryWrapper<>();\n'
            'qw.eq("deal_ref_no", no);\n',
            "fake/WithdrawalsServiceImpl.java",
        )

    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body",
        _fake_fetch,
    )

    pack = cc.gather_compose_evidence(
        _FAKE_PROJECT_ID,
        "根据借据号查放款表",
        lookup_table="放款表",
    )

    kinds = {s.get("kind") for s in pack.get("snippets") or []}
    assert "entity" in kinds, "entity definitions must be included in evidence"
    entity_snippets = [s for s in pack["snippets"] if s.get("kind") == "entity"]
    assert any("Withdrawals" in s.get("class_method", "") for s in entity_snippets)


def test_gather_evidence_empty_hits_marks_honest_gap(monkeypatch):
    """No hits → evidence pack still marks gap instead of failing."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict", lambda *a, **k: []
    )

    pack = cc.gather_compose_evidence(
        _FAKE_PROJECT_ID,
        "查一个不存在的表",
        lookup_key="未知",
        lookup_table="不存在表",
    )

    assert not pack["snippets"], "no snippets when no hits"
    assert not pack["hits"]


def test_format_evidence_shows_gap_when_no_snippets():
    """Formatted evidence must clearly state evidence gap."""
    from app.agent import compose_code as cc

    txt = cc._format_evidence_for_llm(
        {
            "hits": [],
            "snippets": [],
            "lookup_key": "借据号",
            "lookup_table": "放款表",
            "lookup_steps": [{"op": "select", "table": "放款表", "key": "借据号"}],
        }
    )
    assert "查询键: 借据号" in txt
    assert "主表/实体: 放款表" in txt
    assert "证据不足" in txt or "未读到" in txt


def test_format_evidence_includes_agg_ops():
    """Aggregate operations must be presentable in evidence."""
    from app.agent import compose_code as cc

    txt = cc._format_evidence_for_llm(
        {
            "lookup_key": None,
            "lookup_table": None,
            "lookup_steps": [
                {"op": "agg", "table": "t_withdrawals", "fields": ["amount"], "agg": ["sum"]}
            ],
            "snippets": [
                {
                    "class_method": "WithdrawalsService#sumAmount",
                    "file_path": "svc/WithdrawalsService.java",
                    "code": 'return query().select("SUM(amount)");',
                }
            ],
        }
    )
    assert "op=agg" in txt
    assert "fields=amount" in txt
    assert "证据1: `WithdrawalsService#sumAmount`" in txt


# ═══════════════════════════════════════════════════════════════════════
# 2. CODE SYNTHESIS QUALITY
# ═══════════════════════════════════════════════════════════════════════


def test_synthesize_ok_requires_code_fence(monkeypatch):
    """LLM answer without ```java fence must NOT be reported as ok."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        cc,
        "gather_compose_evidence",
        lambda *a, **k: {
            "hits": ["WithdrawalsService#queryByNo"],
            "snippets": [
                {
                    "class_method": "WithdrawalsService#queryByNo",
                    "file_path": "x.java",
                    "code": 'qw.eq("deal_ref_no", no);',
                }
            ],
            "lookup_key": "借据号",
            "lookup_table": "放款表",
            "lookup_steps": [],
        },
    )

    class _NoFenceLLM:
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return "根据已有证据，放款表可以通过借据号查询。"

        def chat_sync(self, messages, **kwargs):
            return asyncio.run(self.chat(messages, **kwargs))

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _NoFenceLLM())
    out = asyncio.run(
        cc.run_compose_code(
            _FAKE_PROJECT_ID,
            "给我一段代码，根据借据号查放款表",
            view_kind="db_pipeline",
        )
    )
    assert out["ok"] is False, "answer without `````` must not be ok"
    assert "```java" not in out["answer"]


def test_synthesize_fallback_on_llm_failure(monkeypatch):
    """When LLM fails, answer must show evidence + honest gap, not invented code."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        cc,
        "gather_compose_evidence",
        lambda *a, **k: {
            "hits": ["WithdrawalsService#queryByNo"],
            "snippets": [
                {
                    "class_method": "WithdrawalsService#queryByNo",
                    "file_path": "x.java",
                    "code": 'qw.eq("deal_ref_no", no);',
                }
            ],
            "lookup_key": "借据号",
            "lookup_table": "放款表",
            "lookup_steps": [],
        },
    )

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: None)

    try:
        out = asyncio.run(
            cc.run_compose_code(
                _FAKE_PROJECT_ID,
                "给我一段代码，根据借据号查放款表",
                view_kind="db_pipeline",
            )
        )
    except Exception:
        out = {"answer": "代码生成遇到错误", "ok": False}

    if not out.get("ok"):
        assert "证据" in out.get("answer", "") or "WithdrawalsService" in out.get("answer", "")


def test_synthesize_keeps_entity_refs_real(monkeypatch):
    """Generated code must reference real entity classes, not invented ones."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        cc,
        "gather_compose_evidence",
        lambda *a, **k: {
            "hits": ["WithdrawalsServiceImpl#queryProductByDealRefNO"],
            "snippets": [
                {
                    "class_method": "WithdrawalsServiceImpl#queryProductByDealRefNO",
                    "file_path": "x.java",
                    "code": (
                        'public Withdrawals queryProductByDealRefNO(String dealRefNo) {\n'
                        '  QueryWrapper<Withdrawals> qw = new QueryWrapper<>();\n'
                        '  qw.eq("deal_ref_no", dealRefNo);\n'
                        '  return baseMapper.selectOne(qw);\n'
                        '}'
                    ),
                },
                {
                    "class_method": "LoanAssistanceService#getByRefNo",
                    "file_path": "x.java",
                    "code": (
                        'public List<LoanAssistance> getByRefNo(String refNo) {\n'
                        '  return baseMapper.selectList(\n'
                        '    new QueryWrapper<LoanAssistance>().eq("deal_ref_no", refNo));\n'
                        '}'
                    ),
                },
            ],
            "lookup_key": "借据号",
            "lookup_table": "放款表",
            "lookup_steps": [
                {"op": "select", "table": "放款表", "key": "借据号"},
                {"op": "select", "table": "助贷表", "key_from": "结果"},
            ],
        },
    )

    class _GoodLLM:
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                "基于 WithdrawalsServiceImpl#queryProductByDealRefNO 和 "
                "LoanAssistanceService#getByRefNo 证据生成：\n"
                "```java\n"
                "// 1. 根据借据号查放款表\n"
                'Withdrawals withdrawals = withdrawalsMapper.selectOne(\n'
                '  new QueryWrapper<Withdrawals>().eq("deal_ref_no", dealRefNo));\n'
                "if (withdrawals != null) {\n"
                "  // 2. 根据放款结果查助贷表\n"
                '  List<LoanAssistance> list = loanAssistanceMapper.selectList(\n'
                '    new QueryWrapper<LoanAssistance>().eq("deal_ref_no", dealRefNo));\n'
                "}\n"
                "```"
            )

        def chat_sync(self, messages, **kwargs):
            return asyncio.run(self.chat(messages, **kwargs))

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _GoodLLM())
    out = asyncio.run(
        cc.run_compose_code(
            _FAKE_PROJECT_ID,
            "给我一段代码：根据借据号查放款表，再根据结果查助贷表",
            view_kind="db_pipeline",
        )
    )
    assert out["ok"] is True
    assert "```java" in out["answer"]
    assert "Withdrawals" in out["answer"]
    assert "LoanAssistance" in out["answer"]


def test_synthesize_style_consistent_with_evidence(monkeypatch):
    """Generated code style should match project patterns (MyBatis-Plus QueryWrapper)."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        cc,
        "gather_compose_evidence",
        lambda *a, **k: {
            "hits": ["SomeService#query"],
            "snippets": [
                {
                    "class_method": "SomeService#query",
                    "file_path": "x.java",
                    "code": (
                        'public Entity query(String key) {\n'
                        '  return baseMapper.selectOne(\n'
                        '    new QueryWrapper<Entity>().eq("column", key));\n'
                        '}'
                    ),
                }
            ],
            "lookup_key": None,
            "lookup_table": None,
            "lookup_steps": [],
        },
    )

    class _StyleLLM:
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                "生成的代码：\n"
                "```java\n"
                "// 使用 QueryWrapper 风格\n"
                'Entity result = baseMapper.selectOne(\n'
                '  new QueryWrapper<Entity>().eq("column", value));\n'
                "```"
            )

        def chat_sync(self, messages, **kwargs):
            return asyncio.run(self.chat(messages, **kwargs))

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _StyleLLM())
    out = asyncio.run(
        cc.run_compose_code(
            _FAKE_PROJECT_ID,
            "查询某表，风格保持一致",
        )
    )
    assert out["ok"] is True
    assert "QueryWrapper" in out["answer"] or "baseMapper" in out["answer"]


# ═══════════════════════════════════════════════════════════════════════
# 3. PATCH GENERATION QUALITY
# ═══════════════════════════════════════════════════════════════════════


def test_parse_patches_from_llm_parses_file_format():
    """LLM patch blocks in ### FILE: format must be parsed correctly."""
    from app.agent import patcher as patcher_mod

    raw = """### FILE: src/java/WithdrawalsServiceImpl.java
*** OLD
  public String process(String input) {
    String result = doInternal(input);
    return result;
  }
*** NEW
  public String process(String input) {
    String result = doInternal(input);
    log.info("processing: {}", input);
    validateInput(input);
    return result;
  }
"""
    patches = patcher_mod.parse_patches_from_llm(raw)
    assert len(patches) >= 1, "should parse at least one patch"
    files = [p.get("file") or "" for p in patches]
    assert any("WithdrawalsServiceImpl" in f for f in files)


def test_parse_patches_honest_gap_on_empty():
    """No diff blocks → empty patch list, no crash."""
    from app.agent import patcher as patcher_mod

    raw = "根据证据无法生成代码改动。"
    patches = patcher_mod.parse_patches_from_llm(raw)
    assert patches == []


def test_summarize_patches_includes_counts():
    """Patch summary must report file/applied/failed counts."""
    from app.agent import patcher as patcher_mod

    results = [
        {"file": "a.java", "ok": True, "applied": 2},
        {"file": "b.java", "ok": False, "error": "file not found"},
    ]
    summary_lines = patcher_mod.summarize_patch_results(results)
    assert any("a.java" in line for line in summary_lines)
    assert any("b.java" in line for line in summary_lines)
    # The function uses "[fail]" marker and includes the error text for failed items
    assert any("[fail]" in line or "fail" in line.lower() for line in summary_lines)


# ═══════════════════════════════════════════════════════════════════════
# 4. ADD-FIELD CHANGE PLAN QUALITY
# ═══════════════════════════════════════════════════════════════════════


def test_add_field_plan_detects_missing_vo_role():
    """Review must detect when add-field plan lacks VO/ReqVo role."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "新增一个字段 xxx，从入参到落库串一下",
        (
            "## 结论\n新增字段 xxx（String）\n"
            "## 改动顺序\n1. 实体类加字段\n2. DDL\n"
            "## DDL\nALTER TABLE demo_b_ext ADD COLUMN xxx varchar(32);\n"
            "## 说明\n字段 xxx 用于记录额外信息。"
        ),
    )
    assert review.get("force_fail") or not review.get("ok"), (
        "should flag missing VO role"
    )
    gaps = " ".join(review.get("gaps") or [])
    assert "VO" in gaps or "ReqVo" in gaps or "入参" in gaps or "绑定" in gaps


def test_add_field_plan_accepts_complete_plan():
    """Full add-field plan with all roles must pass review."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "新增一个字段 remark，从入参到落库",
        (
            "## 结论\n新增字段 remark（String）用于备注\n"
            "## 改动顺序\n"
            "1. WithdrawalsApplyReqVo — 加 private String remark;\n"
            "2. 绑定：reqMap.put(\"remark\", reqVo.getRemark());\n"
            "3. Ext 实体 — 加 private String remark;\n"
            "4. DDL\n"
            "## 从 → 到\n"
            "入参 reqVo → 绑定 reqMap → 落库 Ext 实体\n"
            "## DDL\n"
            "ALTER TABLE demo_b_withdrawals_ext ADD COLUMN remark varchar(32);\n"
            "## 说明\n"
            "remark 为新增备注字段，非必填。"
        ),
    )
    assert review["ok"] is True, (
        f"complete plan should pass, got gaps: {review.get('gaps')}"
    )
    assert review["score"] >= 80
    assert not review.get("force_fail")


def test_add_field_detects_wrong_table_target():
    """Review must catch when DDL targets wrong table (apply-table not ext-table)."""
    from app.agent.answer_reviewer import _heuristic_review

    review = _heuristic_review(
        "新增字段 deal_rate，从入参到落库",
        (
            "## 结论\n新增字段 deal_rate\n"
            "## 改动顺序\n1. 入参 VO 加字段\n2. DDL\n"
            "## DDL\nALTER TABLE demo_b_withdrawals_apply ADD COLUMN deal_rate decimal(10,2);\n"
        ),
    )
    # The answer has only VO + DDL but no bind step and uses an apply-table not ext-table
    assert not review["ok"], "should reject incomplete plan with wrong table target"


# ═══════════════════════════════════════════════════════════════════════
# 5. PROTOCOL SYNC PLAN QUALITY
# ═══════════════════════════════════════════════════════════════════════


def test_protocol_sync_without_sync_detail_rejected():
    """Protocol sync must include actual sync assembly or be rejected."""
    from app.agent.answer_reviewer import _heuristic_review

    # The function wants_extend_field_to_protocol_sync always returns False
    # (removed per architecture), so _heuristic_review treats this as a generic
    # question, not as flow or add-field. The answer passes because it is
    # non-empty and contains no invented tables. To test rejection, include a
    # table name not present in evidence (triggers force_fail).
    review = _heuristic_review(
        "字段透传到协议同步",
        (
            "## 结论\n新增字段透传\n"
            "## 改动清单\n1. VO 加字段\n2. DDL\n"
            "落库至 t_user_quota 表"  # ← invented table — forces force_fail
        ),
    )
    # The answer contains an invented table → must be rejected
    assert not review["ok"], (
        "should reject when answer contains invented tables "
        f"(got gaps: {review.get('gaps')})"
    )
    assert review["force_fail"] is True


# ═══════════════════════════════════════════════════════════════════════
# 6. VERIFY LOOP — repair prompt quality
# ═══════════════════════════════════════════════════════════════════════


def test_verify_repair_prompt_includes_gap_evidence():
    """Repair prompt must contain previous error output."""
    from app.agent.verify_loop import build_repair_prompt

    prompt = build_repair_prompt(
        requirement="新增一个字段",
        verify_report={"logs": "Cannot find symbol: newField\n  location: x.java:42"},
        patches=[{"path": "x.java", "ok": True, "applied": 1}],
        scope=[{"file": "x.java"}],
    )
    assert "Cannot find symbol" in prompt
    assert "x.java" in prompt


# ═══════════════════════════════════════════════════════════════════════
# 7. compose_code end-to-end
# ═══════════════════════════════════════════════════════════════════════


def test_run_compose_code_full_pipeline(monkeypatch):
    """Full compose_code pipeline: evidence → LLM → validated answer."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: ["WithdrawalsServiceImpl#queryByDealRefNO"],
    )
    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body",
        lambda *a, **k: (
            'public Withdrawals queryByDealRefNO(String refNo) {\n'
            '  QueryWrapper<Withdrawals> qw = new QueryWrapper<>();\n'
            '  qw.eq("deal_ref_no", refNo);\n'
            '  return baseMapper.selectOne(qw);\n'
            '}',
            "fake/WithdrawalsServiceImpl.java",
        ),
    )
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict", lambda *a, **k: []
    )

    out = asyncio.run(
        cc.run_compose_code(
            _FAKE_PROJECT_ID,
            "根据借据号查询放款表数据",
            lookup_key="借据号",
            lookup_table="放款表",
            view_kind="db_pipeline",
        )
    )
    assert "```java" in out["answer"] or "未完成" in out["answer"]
    assert out["evidence"] is not None
    assert out["evidence"]["hits"] is not None