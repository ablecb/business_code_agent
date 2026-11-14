"""Tests for the golden QA library (方案一：离线黄金问答库 + 运行时命中即答)."""
from __future__ import annotations

import os

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_PROVIDER"] = "hash"

from contextlib import contextmanager

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.llm.embedding import HashEmbeddingProvider
from app.llm.factory import get_embedding_provider, get_llm_provider
from app.models.orm import Base, GoldenQAORM, SymbolDescORM, SymbolORM

_engine = create_engine("sqlite://")
# 只建本测试用到的表（全量 create_all 会在 kb_chunk 的裸 MEDIUMTEXT 上失败）
Base.metadata.create_all(
    _engine,
    tables=[
        GoldenQAORM.__table__,
        SymbolDescORM.__table__,
        SymbolORM.__table__,
    ],
)
_Session = sessionmaker(
    bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False
)


@contextmanager
def _fake_session():
    session = _Session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture(autouse=True)
def _reset_caches(monkeypatch):
    get_settings.cache_clear()
    get_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    with _Session() as s:
        s.query(GoldenQAORM).delete()
        s.query(SymbolDescORM).delete()
        s.query(SymbolORM).delete()
        s.commit()
    monkeypatch.setattr("app.analysis.golden_qa.get_session", _fake_session)
    yield


# sqlite 的 BIGINT 主键不会自增，测试里显式给 id
_seq = 0


def _next_id() -> int:
    global _seq
    _seq += 1
    return _seq


def _add_qa(question: str, answer: str, *, project_id: str = "p1", entry_cm: str | None = None):
    emb = HashEmbeddingProvider(dim=64)
    with _Session() as s:
        s.add(
            GoldenQAORM(
                id=_next_id(),
                project_id=project_id,
                question=question,
                answer=answer,
                entry_cm=entry_cm,
                votes=0.8,
                source="consensus",
                question_embedding_json=emb.embed([question])[0],
            )
        )
        s.commit()


def _add_qa_source(question: str, answer: str, source: str, *, project_id: str = "p1"):
    """Add a QA row with an explicit source (consensus / interface_tables / …)."""
    emb = HashEmbeddingProvider(dim=64)
    with _Session() as s:
        s.add(
            GoldenQAORM(
                id=_next_id(),
                project_id=project_id,
                question=question,
                answer=answer,
                entry_cm=None,
                votes=0.9,
                source=source,
                question_embedding_json=emb.embed([question])[0],
            )
        )
        s.commit()


def test_search_ranks_keyword_overlap_first(monkeypatch):
    from app.analysis.golden_qa import search_golden_qa

    _add_qa("放款流程怎么走", "放款流程答案")
    _add_qa("提现接口地址", "提现接口答案")
    hits = search_golden_qa("p1", "放款流程是什么样的", top_k=3)
    assert hits
    assert hits[0]["answer"] == "放款流程答案"
    assert hits[0]["score"] > 0


def test_search_scoped_by_project(monkeypatch):
    from app.analysis.golden_qa import search_golden_qa

    _add_qa("放款流程怎么走", "放款流程答案")
    _add_qa("放款流程怎么走", "别的项目答案", project_id="p2")
    hits = search_golden_qa("p2", "放款流程怎么走", top_k=3)
    assert len(hits) == 1
    assert hits[0]["answer"] == "别的项目答案"


def test_try_golden_answer_hit_above_threshold(monkeypatch):
    from app.analysis.golden_qa import try_golden_answer

    _add_qa("放款流程是什么样的", "离线生成的放款流程答案")
    hit = try_golden_answer("p1", "放款流程是什么样的", min_score=0.2)
    assert hit is not None
    assert "放款流程" in hit["answer"]
    assert hit["score"] >= 0.2


def test_try_golden_answer_below_threshold_none(monkeypatch):
    from app.analysis.golden_qa import try_golden_answer

    _add_qa("放款流程是什么样的", "答案")
    hit = try_golden_answer("p1", "帮我写一首关于春天的诗", min_score=0.9)
    assert hit is None


def test_purpose_ask_excludes_interface_tables_la():
    """「XX接口主要做了什么」是用途问法 — interface_tables 表清单答案必须被排除。

    回归 desktop-chat-264e7dfc904a: 「用信申请接口主要做了什么」以相似度 0.69
    命中了「涉及哪些表」黄金答案（答非所问）。「主要做了什么」缺「做」字后缀
    使 _QUESTION_PURPOSE_RE 未命中，导致硬排除失效。
    """
    from app.analysis.golden_qa import search_golden_qa, try_golden_answer

    _add_qa_source(
        "用信申请接口涉及哪些表",
        "用信申请涉及的数据库表（共 18 张）：…表清单…",
        source="interface_tables",
    )
    _add_qa_source(
        "用信申请接口主要做了什么",
        "用信申请主流程：受理→校验→调用授信→落库→返回",
        source="consensus",
    )
    q = "用信申请接口主要做了什么"
    hits = search_golden_qa("p1", q, top_k=5)
    # interface_tables 答案被硬排除 — 绝不排在第一
    assert hits, hits
    assert hits[0]["source"] != "interface_tables", hits
    assert hits[0]["answer"] != "用信申请涉及的数据库表（共 18 张）：…表清单…"
    # try_golden_answer 直接命中流程答案
    hit = try_golden_answer("p1", q, min_score=0.0)
    assert hit is not None
    assert hit["source"] == "consensus"
    assert "主流程" in hit["answer"]


def test_purpose_ask_excludes_interface_tables_le():
    """「具体做了什么 / 做了些什么」同样排除表清单答案。"""
    from app.analysis.golden_qa import search_golden_qa

    _add_qa_source(
        "结清证明申请接口涉及哪些表",
        "表清单答案",
        source="interface_tables",
    )
    _add_qa_source(
        "结清证明申请接口具体做了什么",
        "结清证明流程答案",
        source="consensus",
    )
    for q in ("结清证明申请接口具体做了什么", "授信通知接口做了些什么"):
        hits = search_golden_qa("p1", q, top_k=5)
        assert hits
        # interface_tables 不得命中这类用途问法（可能没有其他候选，允许为空但不得错排第一）
        top = hits[0]
        assert top["source"] != "interface_tables", (q, top)


def test_table_list_ask_still_allows_interface_tables():
    """表清单问法仍应命中 interface_tables（不误伤正向场景）。"""
    from app.analysis.golden_qa import search_golden_qa

    _add_qa_source(
        "用信申请接口涉及哪些表",
        "用信申请涉及的表（共 18 张）：…",
        source="interface_tables",
    )
    q = "用信申请接口涉及哪些表"
    hits = search_golden_qa("p1", q, top_k=5)
    assert hits
    assert hits[0]["source"] == "interface_tables", hits


def test_purpose_ask_does_not_exclude_consensus():
    """用途问法只排除 interface_tables，consensus 流程答案不受影响。"""
    from app.analysis.golden_qa import search_golden_qa

    _add_qa_source(
        "放款接口主要做了什么",
        "放款主流程答案",
        source="consensus",
    )
    hits = search_golden_qa("p1", "放款接口主要做了什么", top_k=3)
    assert hits
    assert hits[0]["source"] == "consensus"


def test_try_golden_answer_empty_store_none(monkeypatch):
    from app.analysis.golden_qa import try_golden_answer

    assert try_golden_answer("p1", "放款流程什么样") is None


def test_try_golden_answer_disabled_returns_none(monkeypatch):
    from app.analysis.golden_qa import try_golden_answer

    _add_qa("放款流程是什么样的", "答案")
    monkeypatch.setenv("GOLDEN_QA_ENABLED", "false")
    get_settings.cache_clear()
    assert try_golden_answer("p1", "放款流程是什么样的", min_score=0.0) is None
    get_settings.cache_clear()


def test_consensus_pick_returns_most_similar():
    from app.analysis.golden_qa import _consensus_pick

    # b 与 a、c 的相似度之和最高 → 「最居中」的样本胜出
    vecs = {"a": [1.0, 0.0, 0.0], "b": [0.9, 0.1, 0.0], "c": [0.0, 1.0, 0.0]}

    class _FakeEmbed:
        def embed(self, texts):
            return [vecs.get(t, [0.0, 0.0, 1.0]) for t in texts]

    best, votes, samples = _consensus_pick(["a", "b", "c"], _FakeEmbed())
    assert best == "b"
    assert votes > 0.45
    assert samples == ["a", "b", "c"]


def test_consensus_pick_single_answer():
    from app.analysis.golden_qa import _consensus_pick

    best, votes, _ = _consensus_pick(["only"], None)
    assert best == "only"
    assert votes == 1.0


def test_candidates_from_flow_doc_and_controller(monkeypatch):
    from app.analysis.golden_qa import candidate_questions_from_index

    with _Session() as s:
        s.add(
            SymbolDescORM(
                id=_next_id(),
                project_id="p1",
                symbol_id="s1",
                summary="放款主流程摘要",
                extra_json={
                    "class_method": "LoanController#loan",
                    "http_paths": ["/loan/apply"],
                    "flow_doc": {
                        "title": "放款流程",
                        "narrative": "从受理到放款",
                        "stages": [
                            {"class_method": "A#a", "title": "受理", "desc_business": "校验"},
                            {"class_method": "B#b", "title": "审批"},
                            {"class_method": "C#c", "title": "放款"},
                        ],
                    },
                },
            )
        )
        s.add(
            SymbolORM(
                id="s2",
                project_id="p1",
                kind="Controller",
                fqcn="com.x.LoanController",
                simple_name="LoanController",
                method_name="loan",
                file_path="LoanController.java",
            )
        )
        s.add(
            SymbolDescORM(
                id=_next_id(),
                project_id="p1",
                symbol_id="s2",
                summary="借款申请接口",
            )
        )
        s.commit()
    cands = candidate_questions_from_index("p1")
    kinds = {c["kind"] for c in cands}
    assert "flow" in kinds
    assert "controller" in kinds
    flow_qs = [c["question"] for c in cands if c["kind"] == "flow"]
    assert any("流程" in q for q in flow_qs)
    assert any("LoanController" in (c["entry_cm"] or "") for c in cands)


def test_stats_report():
    from app.analysis.golden_qa import golden_qa_stats

    _add_qa("放款流程是什么样的", "答案")
    stats = golden_qa_stats("p1")
    assert stats["ok"] is True
    assert stats["total"] == 1
    assert stats["by_source"].get("consensus") == 1


def test_try_golden_qa_reply_hit(monkeypatch):
    """黄金快路径命中 → 返回带 reply 的 state。"""
    import asyncio

    from app.agent.graph_agent import _try_golden_qa_reply

    def fake_try(project_id, question, *, min_score=None):
        del project_id, question, min_score
        return {
            "answer": "黄金答案：放款主流程是受理→审批→放款。",
            "score": 0.91,
            "source": "consensus",
            "votes": 0.8,
            "question": "放款流程",
        }

    monkeypatch.setattr("app.analysis.golden_qa.try_golden_answer", fake_try)

    state = {
        "message": "放款流程什么样",
        "project_id": "p1",
        "project_ids": ["p1"],
        "session_id": "sess-golden",
        "intent": "ask",
        "phase": "idle",
    }
    out = asyncio.run(_try_golden_qa_reply(state))
    assert out is not None
    assert "黄金答案" in out["reply"]
    assert "离线黄金问答库" in out["reply"]
    assert out["phase"] == "done"
    assert (out.get("golden_hit") or {}).get("score") == 0.91


def test_try_golden_qa_reply_miss_returns_none(monkeypatch):
    """黄金快路径未命中 → None（fail-open，走后续管线）。"""
    import asyncio

    from app.agent.graph_agent import _try_golden_qa_reply

    def fake_try(project_id, question, *, min_score=None):
        del project_id, question, min_score
        return None

    monkeypatch.setattr("app.analysis.golden_qa.try_golden_answer", fake_try)

    state = {
        "message": "放款流程什么样",
        "project_id": "p1",
        "project_ids": ["p1"],
        "session_id": "sess-miss",
        "intent": "ask",
        "phase": "idle",
    }
    out = asyncio.run(_try_golden_qa_reply(state))
    assert out is None


def test_purpose_ask_excludes_table_field():
    """用途/流程问法排除 table_field 字段清单答案。

    回归 desktop-chat-fb2229929b4f：「结清证明申请接口 主要流程是什么」以
    相似度 0.59 命中了「结清证明发送表 表结构/有哪些字段」（table_field 类型）
    的 27 个字段清单 — _EXCLUDED_SOURCES_FOR_PURPOSE 之前只排除
    interface_tables，未排除 table_field / table_relation。
    """
    from app.analysis.golden_qa import _hard_exclude, search_golden_qa

    q = "结清证明申请接口 主要流程是什么"
    # table_field / table_relation / interface_tables 全部被用途问法排除
    assert _hard_exclude(q, "table_field") is True
    assert _hard_exclude(q, "table_relation") is True
    assert _hard_exclude(q, "interface_tables") is True
    # consensus 流程答案不受影响
    assert _hard_exclude(q, "consensus") is False

    # 库里只有 table_field 答案 → 检索为空（宁缺毋滥，fail-open 走正常管线）
    _add_qa_source(
        "结清证明发送表的表结构是什么",
        "字段清单：id, channelId, signatureId …",
        source="table_field",
    )
    _add_qa_source(
        "结清证明发送表有哪些字段",
        "字段清单 v2",
        source="table_field",
    )
    hits = search_golden_qa("p1", q, top_k=5)
    assert hits == [], hits


def test_purpose_ask_table_field_excluded_but_consensus_wins():
    """用途问法排除字段清单，但 consensus 流程答案仍可命中。"""
    from app.analysis.golden_qa import search_golden_qa, try_golden_answer

    _add_qa_source(
        "结清证明发送表的表结构是什么",
        "字段清单：id, channelId …",
        source="table_field",
    )
    _add_qa_source(
        "结清证明申请接口 主要流程是什么",
        "结清证明申请主流程：受理→校验→开具→推送→返回",
        source="consensus",
    )
    q = "结清证明申请接口 主要流程是什么"
    hits = search_golden_qa("p1", q, top_k=5)
    assert hits, hits
    assert hits[0]["source"] == "consensus", hits
    hit = try_golden_answer("p1", q, min_score=0.0)
    assert hit is not None
    assert "主流程" in hit["answer"]


def test_api_enum_ask_excludes_interface_tables():
    """接口枚举问法「XX接口都有哪些」排除 interface_tables 表清单答案。

    回归 desktop-chat-e4472c4b89db：「结清证明申请接口都有哪些」以
    相似度 0.58 命中了「提现申请接口涉及哪些表」（interface_tables 类型）
    的 6 张表清单 — _EXCLUDED_SOURCES_FOR_API_TASK 之前只排除
    table_field / table_relation，未排除 interface_tables。
    """
    from app.analysis.golden_qa import _hard_exclude, search_golden_qa

    q = "结清证明申请接口都有哪些"
    # interface_tables / table_field / table_relation 全部被接口枚举问法排除
    assert _hard_exclude(q, "interface_tables") is True
    assert _hard_exclude(q, "table_field") is True
    assert _hard_exclude(q, "table_relation") is True
    # consensus 流程答案不受影响
    assert _hard_exclude(q, "consensus") is False

    # 库里只有 interface_tables 答案 → 检索为空（宁缺毋滥，fail-open 走正常管线）
    _add_qa_source(
        "提现申请接口涉及哪些表",
        "提现申请涉及的数据库表（共 6 张）：…表清单…",
        source="interface_tables",
    )
    hits = search_golden_qa("p1", q, top_k=5)
    assert hits == [], hits


def test_api_enum_excludes_interface_tables_but_consensus_wins():
    """接口枚举问法排除表清单，但 consensus 流程答案仍可命中。"""
    from app.analysis.golden_qa import search_golden_qa, try_golden_answer

    _add_qa_source(
        "提现申请接口涉及哪些表",
        "提现申请涉及的数据库表（共 6 张）：…表清单…",
        source="interface_tables",
    )
    _add_qa_source(
        "结清证明申请接口 主要流程是什么",
        "结清证明申请主流程：受理→校验→开具→推送→返回",
        source="consensus",
    )
    q = "结清证明申请接口都有哪些"
    hits = search_golden_qa("p1", q, top_k=5)
    assert hits, hits
    assert hits[0]["source"] == "consensus", hits
    hit = try_golden_answer("p1", q, min_score=0.0)
    assert hit is not None
    assert "主流程" in hit["answer"]
