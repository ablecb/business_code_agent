from __future__ import annotations

import asyncio
from typing import Any

from app.llm.deepseek import (
    DeepSeekProvider,
    FailoverLLMProvider,
    MultiKeyFailoverLLMProvider,
)


class _HangingClient:
    """Fake httpx client whose POST hangs for hang_s seconds."""

    def __init__(self, hang_s: float):
        self.hang_s = hang_s

    async def __aenter__(self) -> "_HangingClient":
        return self

    async def __aexit__(self, *a: Any) -> bool:
        return False

    async def post(self, *a: Any, **kw: Any) -> Any:
        await asyncio.sleep(self.hang_s)
        raise AssertionError("post 不应在预算内返回")


def _make_hanging_provider(
    hang_s: float, *, total_budget: float | None = None
) -> DeepSeekProvider:
    return DeepSeekProvider(
        api_key="sk-test",
        base_url="http://127.0.0.1:9",
        model="test-sleep",
        timeout=300.0,
        total_budget=total_budget,
    )


def _patch_hanging_client(monkeypatch, hang_s: float) -> None:
    import app.llm.deepseek as d

    def _fake_make_async_client(timeout=None):
        return _HangingClient(hang_s)

    monkeypatch.setattr(d, "make_async_client", _fake_make_async_client)


def _elapsed_and_error(coro) -> tuple[float, Exception]:
    async def _run():
        t0 = asyncio.get_event_loop().time()
        try:
            await coro
        except Exception as e:  # noqa: BLE001
            return asyncio.get_event_loop().time() - t0, e
        return asyncio.get_event_loop().time() - t0, AssertionError("应抛出预算超时")

    return asyncio.run(_run())


def test_single_provider_budget_caps_total_time(monkeypatch):
    """单 provider：即便内部重试循环想跑 5×330s，总预算也应提前掐断。"""
    _patch_hanging_client(monkeypatch, hang_s=10)
    p = _make_hanging_provider(hang_s=10, total_budget=2)

    elapsed, err = _elapsed_and_error(
        p.chat([{"role": "user", "content": "hi"}])
    )
    assert "budget" in str(err).lower() or "timeout" in str(err).lower()
    assert elapsed < 8, f"预算应把单次调用限制在 ~2s，实际 {elapsed:.1f}s"


def test_failover_budget_caps_chain(monkeypatch):
    """failover 链：2 个都卡死的 provider，总预算应整体封顶而不是 2×各自超时。"""
    _patch_hanging_client(monkeypatch, hang_s=10)
    chain = FailoverLLMProvider(
        [
            _make_hanging_provider(hang_s=10),
            _make_hanging_provider(hang_s=10),
        ],
        total_budget=2,
    )

    elapsed, err = _elapsed_and_error(
        chain.chat([{"role": "user", "content": "hi"}])
    )
    assert "budget" in str(err).lower() or "timeout" in str(err).lower()
    assert elapsed < 8, f"failover 链应被总预算封顶，实际 {elapsed:.1f}s"


def test_multikey_budget_caps_rotation(monkeypatch):
    """多 key：2 key × 1 卡死模型，总预算应封顶整体轮询。"""
    _patch_hanging_client(monkeypatch, hang_s=10)
    mp = MultiKeyFailoverLLMProvider(
        keys=["k1", "k2"],
        providers_by_key=[
            [_make_hanging_provider(hang_s=10)],
            [_make_hanging_provider(hang_s=10)],
        ],
        total_budget=2,
    )

    elapsed, err = _elapsed_and_error(
        mp.chat([{"role": "user", "content": "hi"}])
    )
    assert "budget" in str(err).lower() or "timeout" in str(err).lower()
    assert elapsed < 8, f"多 key 轮询应被总预算封顶，实际 {elapsed:.1f}s"


def test_no_budget_allows_normal_path(monkeypatch):
    """未设总预算（None）时行为不变：走原重试循环并正常返回。"""
    import httpx

    _patch_hanging_client(monkeypatch, hang_s=0.01)

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ok"}}]}

    import app.llm.deepseek as d

    class _FastClient(_HangingClient):
        async def post(self, *a: Any, **kw: Any) -> Any:
            await asyncio.sleep(0)
            return _Resp()

    def _fake_make_async_client(timeout=None):
        return _FastClient(0.01)

    monkeypatch.setattr(d, "make_async_client", _fake_make_async_client)

    p = _make_hanging_provider(hang_s=0.01, total_budget=None)
    assert asyncio.run(p.chat([{"role": "user", "content": "hi"}])) == "ok"
