from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

import httpx

from app.llm.base import LLMProvider
from app.llm.debug_log import log_chat_request, log_chat_response
from app.llm.retry_policy import (
    DEFAULT_MAX_RETRIES,
    compute_delay,
    log_retry,
    should_retry,
)

logger = logging.getLogger("app.llm.ollama")


class OllamaProvider(LLMProvider):
    """Local Ollama models via OpenAI-compatible API (/v1) with /api/chat fallback."""

    name = "ollama"

    def __init__(
        self,
        model: str = "qwen3:32b",
        base_url: str = "http://127.0.0.1:11434",
        api_key: str = "ollama",
        temperature: float = 0.2,
        max_tokens: int = 16384,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "ollama"
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def _openai_base(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    def _native_base(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url[:-3]
        return self.base_url

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        temp = self.temperature if temperature is None else temperature
        tokens = self.max_tokens if max_tokens is None else max_tokens
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format

        log_chat_request(
            self.name,
            self.model,
            messages,
            temperature=temp,
            max_tokens=tokens,
        )
        max_retries = DEFAULT_MAX_RETRIES
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for attempt in range(1, max_retries + 1):
                try:
                    resp = await client.post(
                        f"{self._openai_base()}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    if should_retry(resp.status_code) and attempt < max_retries:
                        sleep_s = compute_delay(
                            resp.status_code,
                            attempt,
                            resp.headers.get("retry-after"),
                            max_delay_ms=40_000,
                        )
                        log_retry(
                            logger,
                            provider=self.name,
                            model=self.model,
                            attempt=attempt,
                            max_retries=max_retries,
                            status=resp.status_code,
                            sleep_s=sleep_s,
                            body=(resp.text or ""),
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    log_chat_response(self.name, self.model, content)
                    return content
                except (httpx.TransportError, httpx.TimeoutException) as e:
                    last_err = e
                    if attempt < max_retries:
                        sleep_s = compute_delay(None, attempt)
                        logger.warning(
                            "ollama transport error attempt=%s/%s sleep=%.1fs: %s — retry",
                            attempt,
                            max_retries,
                            sleep_s,
                            e,
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                except httpx.HTTPStatusError as e:
                    last_err = e
                    if e.response is not None and should_retry(e.response.status_code) and attempt < max_retries:
                        sleep_s = compute_delay(
                            e.response.status_code,
                            attempt,
                            e.response.headers.get("retry-after"),
                            max_delay_ms=40_000,
                        )
                        log_retry(
                            logger,
                            provider=self.name,
                            model=self.model,
                            attempt=attempt,
                            max_retries=max_retries,
                            status=e.response.status_code,
                            sleep_s=sleep_s,
                            body=e.response.text,
                        )
                        await asyncio.sleep(sleep_s)
                        continue
                    break
                except Exception:
                    # Non-retryable or exhausted — fall through to native API fallback.
                    last_err = None
                    break

            # Native Ollama API fallback (kept for robustness).
            native = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temp,
                    "num_predict": tokens,
                },
            }
            try:
                resp = await client.post(
                    f"{self._native_base()}/api/chat",
                    json=native,
                )
                resp.raise_for_status()
                data = resp.json()
                content = data.get("message", {}).get("content", "")
                log_chat_response(self.name, self.model, content)
                return content
            except Exception as e:
                if last_err is not None:
                    raise last_err from e
                raise

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        text = await self.chat(
            messages, temperature=temperature, max_tokens=max_tokens
        )
        yield text
