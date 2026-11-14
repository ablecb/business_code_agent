"""compose_code: evidence gather + synthesize (not raw paste)."""
from __future__ import annotations

import asyncio


def test_gather_compose_evidence_reads_snippets(monkeypatch):
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: ["WithdrawalsServiceImpl#queryProductByDealRefNO"],
    )
    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body",
        lambda *a, **k: (
            'QueryWrapper<Withdrawals> qw = new QueryWrapper<>();\nqw.eq("deal_ref_no", no);\n',
            "fake/WithdrawalsServiceImpl.java",
        ),
    )
    # entity augmentation hits the real table_dict DB; pin it empty so the
    # method snippet order stays deterministic in this unit test
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict",
        lambda *a, **k: [],
    )
    pack = cc.gather_compose_evidence(
        "demo-loan",
        "按借据号查放款表",
        lookup_key="借据号",
        lookup_table="放款表",
    )
    assert pack["snippets"]
    assert pack["snippets"][0]["class_method"].endswith("#queryProductByDealRefNO")
    assert "deal_ref_no" in pack["snippets"][0]["code"]


def test_gather_compose_evidence_includes_entity_defs(monkeypatch):
    """Chinese table names resolve to real Entity class definitions in evidence."""
    from app.agent import compose_code as cc

    monkeypatch.setattr(
        "app.agent.source_view.search_db_lookup_hits",
        lambda *a, **k: ["WithdrawalsServiceImpl#queryProductByDealRefNO"],
    )
    monkeypatch.setattr(
        "app.analysis.table_dict.load_table_dict",
        lambda *a, **k: [
            {
                "table_name": "demo_b_coupon_info",
                "entity_class": "CouponInfo",
                "entity_fqcn": "com.demo.loan.entity.CouponInfo",
                "name_cn": "优惠券表",
                "file_path": "x/CouponInfo.java",
                "fields": [],
            }
        ],
    )

    def _fake_fetch(project_id, cm, *, max_chars=14000):  # noqa: ARG001
        if cm == "CouponInfo":
            return "public class CouponInfo { private String couponId; }", "x/CouponInfo.java"
        return (
            'QueryWrapper<Withdrawals> qw = new QueryWrapper<>();\nqw.eq("deal_ref_no", no);\n',
            "fake/WithdrawalsServiceImpl.java",
        )

    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body",
        _fake_fetch,
    )
    pack = cc.gather_compose_evidence(
        "demo-loan",
        "根据借据号查放款表，再查放款优惠券表",
        lookup_table="放款优惠券表",
    )
    kinds = {s.get("kind") for s in pack.get("snippets") or []}
    assert "entity" in kinds
    assert any(s.get("class_method") == "CouponInfo" for s in pack["snippets"])
    assert any(
        s.get("class_method", "").endswith("#queryProductByDealRefNO")
        for s in pack["snippets"]
    )


def test_run_compose_code_uses_synth_not_raw_target(monkeypatch):
    from app.agent import compose_code as cc
    from app.llm.base import LLMProvider

    monkeypatch.setattr(
        cc,
        "gather_compose_evidence",
        lambda *a, **k: {
            "hits": ["CheckProcessingLoanUtils#checkUnProcessLoan"],
            "snippets": [
                {
                    "class_method": "WithdrawalsServiceImpl#queryProductByDealRefNO",
                    "file_path": "x.java",
                    "code": 'qw.eq("deal_ref_no", dealRefNo);',
                }
            ],
            "lookup_key": "借据号",
            "lookup_table": "放款表",
            "lookup_steps": [
                {"op": "select", "table": "放款表", "key": "借据号"},
                {"op": "select", "table": "放款助贷表", "key_from": "结果"},
            ],
        },
    )

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                "基于 `WithdrawalsServiceImpl#queryProductByDealRefNO` 证据生成：\n"
                "```java\n"
                "Withdrawals w = withdrawalsService.getOne(\n"
                "  new QueryWrapper<Withdrawals>().eq(\"deal_ref_no\", dealRefNo));\n"
                "if (w != null) {\n"
                "  List<LoanAssistance> list = loanAssistanceService.list(\n"
                "    new QueryWrapper<LoanAssistance>().eq(\"deal_ref_no\", dealRefNo));\n"
                "}\n"
                "```"
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(
        cc.run_compose_code(
            "demo-loan",
            "给我一段代码 根据借据号查询放款表 再查助贷表",
            lookup_key="借据号",
            lookup_table="放款表",
            view_kind="db_pipeline",
        )
    )
    assert out["ok"] is True
    assert "```java" in out["answer"]
    assert "deal_ref_no" in out["answer"]
    assert "CheckProcessingLoanUtils" not in out["answer"]


def test_format_evidence_no_snippets_marks_gap():
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
    assert "步骤计划" in txt
    assert "op=select" in txt
    assert "证据不足" in txt


def test_format_evidence_includes_agg_and_fields():
    from app.agent import compose_code as cc

    txt = cc._format_evidence_for_llm(
        {
            "lookup_key": None,
            "lookup_table": None,
            "lookup_steps": [
                {
                    "op": "agg",
                    "table": "t_apply",
                    "fields": ["amount"],
                    "agg": ["sum"],
                }
            ],
            "snippets": [
                {
                    "class_method": "ApplyService#sum",
                    "file_path": "svc/ApplyService.java",
                    "code": "return query().eq(\"k\", v);",
                }
            ],
        }
    )
    assert "op=agg" in txt
    assert "fields=amount" in txt
    assert "agg=sum" in txt
    assert "证据1: `ApplyService#sum`" in txt
    assert "svc/ApplyService.java" in txt


def test_synthesize_llm_failure_falls_back_to_evidence(monkeypatch):
    from app.agent import compose_code as cc
    from app.llm.base import LLMProvider

    class _BoomLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            raise RuntimeError("provider down")

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _BoomLLM())
    out = asyncio.run(
        cc.synthesize_compose_answer(
            "给我一段代码 根据借据号查放款表",
            {
                "snippets": [
                    {
                        "class_method": "Svc#q",
                        "file_path": "x.java",
                        "code": 'qw.eq("deal_ref_no", no);',
                    }
                ],
                "lookup_key": "借据号",
                "lookup_table": "放款表",
                "lookup_steps": [],
            },
        )
    )
    assert "未能完成代码生成" in out
    assert "Svc#q" in out
    assert "qw.eq" in out


def test_synthesize_no_snippets_honest_gap(monkeypatch):
    from app.agent import compose_code as cc
    from app.llm.base import LLMProvider

    class _BoomLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            raise RuntimeError("down")

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _BoomLLM())
    out = asyncio.run(
        cc.synthesize_compose_answer(
            "给我一段代码 按借据号查表",
            {"snippets": [], "lookup_key": None, "lookup_table": None, "lookup_steps": []},
        )
    )
    assert "未检索到可用的 Service/Mapper 证据" in out


def test_run_compose_code_ok_flag_requires_code_fence(monkeypatch):
    """LLM answer without a code fence must NOT be reported as ok."""
    from app.agent import compose_code as cc
    from app.llm.base import LLMProvider

    monkeypatch.setattr(
        cc,
        "gather_compose_evidence",
        lambda *a, **k: {
            "hits": [],
            "snippets": [],
            "lookup_key": None,
            "lookup_table": None,
            "lookup_steps": [],
        },
    )

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return "抱歉，证据不足无法生成代码。"

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(
        cc.run_compose_code("demo-loan", "给我一段代码", view_kind="db_lookup")
    )
    assert out["ok"] is False
    assert out["view_kind"] == "db_lookup"
