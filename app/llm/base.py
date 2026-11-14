from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Iterable


class LLMProvider(ABC):
    """Pluggable chat LLM interface. Swap providers via config only."""

    name: str = "base"

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        raise NotImplementedError

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        text = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        yield text

    def chat_sync(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        import asyncio
        import concurrent.futures

        async def _run() -> str:
            return await self.chat(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_run())

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, _run()).result()


class EmbeddingProvider(ABC):
    name: str = "base"
    dim: int = 512

    @abstractmethod
    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
