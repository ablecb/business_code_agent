"""问答双赢优化（检索缓存 / 会话证据继承 / 黄金问答回填）的单元测试。

覆盖：
- search_cache 进程内 LRU 的存取/清空/embed 缓存；
- hybrid_search 包装层命中缓存即复用，不重复执行检索 impl；
- session_store.record_evidence / get_evidence_context 的去重与渲染；
- golden_qa.record_confirmed_qa 的护栏（长度/失败标记/已来自黄金/无项目）。
全部不依赖 MySQL/Milvus/LLM 网络。
"""
from __future__ import annotations

import pytest

from app.models.schemas import SymbolHit


class _Settings:
    """轻量 settings 替身，避开 .env / pydantic 加载。"""

    search_cache_enabled = True
    search_cache_max_embed = 512
    search_cache_max_search = 256
    search_use_milvus = False
    embedding_provider = "hash"
    embedding_model = "hash"
    golden_qa_enabled = True
    golden_qa_backfill_enabled = True


def _hit(symbol_id: str, score: float = 0.5) -> SymbolHit:
    return SymbolHit(
        symbol_id=symbol_id,
        project_id="p1",
        fqcn=f"com.x.{symbol_id}",
        method_sig=None,
        kind="Class",
        file_path="",
        summary=f"summary {symbol_id}",
        tags=[],
        business_domain=None,
        score=score,
    )


def test_search_cache_roundtrip_and_clear():
    from app.tagging import search_cache

    search_cache.clear()
    assert search_cache.get_search("p1", "放款流程", 12) is None
    search_cache.put_search("p1", "放款流程", 12, None, [_hit("A")])
    hits = search_cache.get_search("p1", "放款流程", 12)
    assert hits is not None and hits[0].symbol_id == "A"
    search_cache.clear()
    assert search_cache.get_search("p1", "放款流程", 12) is None


def test_search_cache_embed_roundtrip():
    from app.tagging import search_cache

    search_cache.clear()
    assert search_cache.cached_embed("hash", "hash", "放款接口") is None
    search_cache.put_embed("hash", "hash", "放款接口", [0.1, 0.2])
    assert search_cache.cached_embed("hash", "hash", "放款接口") == [0.1, 0.2]
    # 不同 provider/model 不应互相命中
    assert search_cache.cached_embed("other", "other", "放款接口") is None


def test_hybrid_search_uses_cache(monkeypatch):
    """同一 (project_id, q) 第二次调用直接命中缓存，不重复执行检索 impl。"""
    import app.tagging.search as search_mod
    from app.core import config as config_mod
    from app.tagging import search_cache

    monkeypatch.setattr(config_mod, "get_settings", lambda: _Settings())

    calls = {"n": 0}

    def fake_impl(q, project_id, *, top_k=12, domain=None):
        calls["n"] += 1
        return [_hit(f"R{calls['n']}", score=1.0)]

    monkeypatch.setattr(search_mod, "_hybrid_search_impl", fake_impl)
    search_cache.clear()

    first = search_mod.hybrid_search("放款流程", "p1", top_k=12)
    second = search_mod.hybrid_search("放款流程", "p1", top_k=12)
    assert calls["n"] == 1
    assert first[0].symbol_id == second[0].symbol_id == "R1"


def test_hybrid_search_cache_disabled_skips(monkeypatch):
    import app.tagging.search as search_mod
    from app.core import config as config_mod
    from app.tagging import search_cache

    class _Off(_Settings):
        search_cache_enabled = False

    monkeypatch.setattr(config_mod, "get_settings", lambda: _Off())

    calls = {"n": 0}

    def fake_impl(q, project_id, *, top_k=12, domain=None):
        calls["n"] += 1
        return [_hit(f"R{calls['n']}")]

    monkeypatch.setattr(search_mod, "_hybrid_search_impl", fake_impl)
    search_cache.clear()

    search_mod.hybrid_search("放款流程", "p1", top_k=12)
    search_mod.hybrid_search("放款流程", "p1", top_k=12)
    assert calls["n"] == 2


def test_record_evidence_dedup_and_context(monkeypatch):
    import app.agent.session_store as ss

    monkeypatch.setattr(ss, "_USE_MEMORY", True)
    monkeypatch.setattr(ss, "_MEMORY_SESSIONS", {})
    monkeypatch.setattr(ss, "_MEMORY_MESSAGES", {})

    sid = ss.get_or_create_session("p1", title="t")
    ss.record_evidence(
        sid,
        [
            {"cm": "WithdrawalsController#withdrawals", "kind": "Controller", "summary": "提现主流程"},
            {"fqcn": "com.x.WithdrawalsService", "kind": "Service", "summary": "服务层"},
        ],
    )
    # 重复写同一 cm（大小写不同）→ 去重，后者覆盖摘要
    ss.record_evidence(sid, [{"cm": "withdrawalscontroller#withdrawals", "summary": "重复"}])
    ctx = ss.get_context(sid)
    seen = ctx.get("evidence_seen") or []
    assert len(seen) == 2

    text = ss.get_evidence_context(sid)
    assert "WithdrawalsController#withdrawals" in text
    assert "com.x.WithdrawalsService" in text
    assert "禁止重复 get_symbol_code" in text
    assert ss.get_evidence_context("no-such-session") == ""


def test_record_confirmed_qa_guardrails(monkeypatch):
    from app.analysis import golden_qa

    monkeypatch.setattr(golden_qa, "get_settings", lambda: _Settings())

    assert (
        golden_qa.record_confirmed_qa("p1", "短的", "x" * 60)["reason"] == "question_length"
    )
    assert (
        golden_qa.record_confirmed_qa("p1", "放款流程是什么样的", "太短")["reason"]
        == "answer_length"
    )
    assert (
        golden_qa.record_confirmed_qa("p1", "放款流程是什么样的", "未能完成检索，请重试。")["reason"]
        == "failure_marker"
    )
    assert (
        golden_qa.record_confirmed_qa(
            "p1", "放款流程是什么样的", "（来源：离线黄金问答库）" + "x" * 80
        )["reason"]
        == "already_golden"
    )
    assert (
        golden_qa.record_confirmed_qa("", "放款流程是什么样的", "x" * 80)["reason"]
        == "no_project"
    )


def test_record_confirmed_qa_disabled(monkeypatch):
    from app.analysis import golden_qa

    class _Disabled(_Settings):
        golden_qa_enabled = False

    monkeypatch.setattr(golden_qa, "get_settings", lambda: _Disabled())
    assert (
        golden_qa.record_confirmed_qa("p1", "放款流程是什么样的", "x" * 80)["reason"] == "disabled"
    )