from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

import httpx

from app.core.config import get_settings
from app.llm.base import LLMProvider
from app.llm.debug_log import log_chat_request, log_chat_response, log_chat_waiting
from app.llm.http_client import make_async_client
from app.llm.tracing import traceable

logger = logging.getLogger("app.llm.deepseek")

# Transient / proxy-ish failures worth retrying (405 often from MITM/nginx when SSL verify off)
_RETRY_STATUS = {405, 408, 425, 429, 500, 502, 503, 504}
_MAX_RETRIES = 5


def _retry_sleep_sec(status_code: int, attempt: int) -> float:
    """Backoff for transient LLM errors; 429 needs longer waits (SiliconFlow RPM)."""
    if status_code == 429:
        # ~2s, 5s, 12s, 25s, 40s (capped)
        return min(40.0, 1.5 * (2.2**attempt))
    return min(8.0, 0.8 * attempt)


def _message_text(message: dict[str, Any]) -> str:
    """Prefer final content; fall back to reasoning_content for some Qwen3 replies."""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    if content is None:
        return ""
    return str(content)


class DeepSeekProvider(LLMProvider):
    """OpenAI-compatible DeepSeek chat provider."""

    name = "deepseek"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        total_budget: float | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 单次逻辑调用总预算（秒）；None = 不限制。只由问答 chat 构建时注入，
        # parse/ingest 管线保持不限（离线解析单次可能超过几十秒）。
        self.total_budget = total_budget

    def _v1_base(self) -> str:
        """Accept base with or without trailing /v1 (SiliconFlow docs use .../v1)."""
        if self.base_url.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    def _apply_thinking_flag(self, payload: dict[str, Any]) -> None:
        """Disable chain-of-thought for Qwen3-style models when configured."""
        try:
            enable = bool(get_settings().llm_enable_thinking)
        except Exception:
            enable = False
        # SiliconFlow / Qwen3 OpenAI-compatible knobs (ignored by other models).
        payload["enable_thinking"] = enable
        payload["chat_template_kwargs"] = {"enable_thinking": enable}

    @traceable(name="chat_completions", run_type="llm")
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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": tokens,
            "stream": False,
        }
        if response_format:
            payload["response_format"] = response_format
        self._apply_thinking_flag(payload)

        log_chat_request(
            self.name,
            self.model,
            messages,
            temperature=temp,
            max_tokens=tokens,
        )
        log_chat_waiting(self.name, self.model)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        t0 = time.time()
        timeout = httpx.Timeout(
            self.timeout,
            connect=min(30.0, self.timeout),
            read=self.timeout,
            write=min(60.0, self.timeout),
        )
        url = f"{self._v1_base()}/chat/completions"
        data: dict[str, Any] | None = None
        last_err: Exception | None = None
        # Hard ceiling so a stuck proxy/socket cannot block a worker forever
        # (httpx timeout alone can hang behind some local TLS proxies).
        hard_deadline = max(30.0, float(self.timeout) + 30.0)
        budget = self.total_budget
        deadline = time.monotonic() + budget if budget else None
        for attempt in range(1, _MAX_RETRIES + 1):
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "LLM total budget exceeded (%.0fs) provider=%s model=%s — give up",
                        budget,
                        self.name,
                        self.model,
                    )
                    raise TimeoutError(
                        f"LLM total budget exceeded ({budget:.0f}s) "
                        f"provider={self.name} model={self.model}"
                    )
                attempt_timeout = min(hard_deadline, remaining)
            else:
                attempt_timeout = hard_deadline
            try:
                async with make_async_client(timeout=timeout) as client:

                    async def _post_once() -> httpx.Response:
                        return await client.post(url, headers=headers, json=payload)

                    resp = await asyncio.wait_for(_post_once(), timeout=attempt_timeout)
                    if resp.status_code in _RETRY_STATUS and attempt < _MAX_RETRIES:
                        body = (resp.text or "")[:300]
                        sleep_s = _retry_sleep_sec(resp.status_code, attempt)
                        logger.warning(
                            "LLM HTTP %s attempt=%s/%s url=%s sleep=%.1fs body=%s — retry",
                            resp.status_code,
                            attempt,
                            _MAX_RETRIES,
                            url,
                            sleep_s,
                            body,
                        )
                        if deadline is not None:
                            sleep_s = min(sleep_s, max(0.0, deadline - time.monotonic()))
                        await asyncio.sleep(sleep_s)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
            except asyncio.TimeoutError as e:
                last_err = e
                logger.warning(
                    "LLM hard timeout attempt=%s/%s after %.0fs — retry",
                    attempt,
                    _MAX_RETRIES,
                    attempt_timeout,
                )
                if attempt >= _MAX_RETRIES or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    raise TimeoutError(
                        f"LLM hard timeout after {_MAX_RETRIES} attempts "
                        f"({attempt_timeout:.0f}s each)"
                    ) from e
                await asyncio.sleep(min(12.0, 1.5 * attempt))
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_err = e
                if attempt >= _MAX_RETRIES or (
                    deadline is not None and time.monotonic() >= deadline
                ):
                    raise
                sleep_s = min(8.0, 0.8 * attempt)
                logger.warning(
                    "LLM transport error attempt=%s/%s: %s — sleep=%.1fs retry",
                    attempt,
                    _MAX_RETRIES,
                    e,
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
        if data is None:
            if last_err:
                raise last_err
            raise RuntimeError(f"LLM empty response after {_MAX_RETRIES} attempts")
        content = _message_text(data["choices"][0].get("message") or {})
        log_chat_response(self.name, self.model, content, elapsed_s=time.time() - t0)
        return content

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        async with make_async_client(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self._v1_base()}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    if data == "[DONE]":
                        break
                    import json

                    chunk = json.loads(data)
                    delta = chunk["choices"][0]["delta"].get("content")
                    if delta:
                        yield delta


class OpenAICompatibleProvider(DeepSeekProvider):
    """Generic OpenAI-compatible provider (swap base_url/model via config)."""

    name = "openai_compatible"


class SiliconFlowProvider(OpenAICompatibleProvider):
    """SiliconFlow (硅基流动) OpenAI-compatible chat."""

    name = "siliconflow"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "Qwen/Qwen3-32b",
        temperature: float = 0.2,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        total_budget: float | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            total_budget=total_budget,
        )


class FailoverLLMProvider(LLMProvider):
    """Try a list of chat providers in order; on failure move to the next.

    Useful when one model name on a host is flaky/rate-limited — the caller
    still gets a successful completion as long as ANY configured model works.
    Empty/whitespace-only replies are treated as failure too (some reasoning
    models burn all max_tokens on thinking and never emit final content).
    """

    name = "failover"

    def __init__(self, providers: list[LLMProvider], *, total_budget: float | None = None) -> None:
        if not providers:
            raise ValueError("FailoverLLMProvider needs at least one provider")
        self.providers = providers
        self.model = "|".join(p.model for p in providers)
        self.total_budget = total_budget

    @traceable(name="chat_failover", run_type="chain")
    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        budget = self.total_budget

        async def _run() -> str:
            errors: list[str] = []
            for idx, provider in enumerate(self.providers):
                try:
                    text = await provider.chat(
                        messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )
                    if not (text or "").strip():
                        raise RuntimeError("empty completion (reasoning may have consumed max_tokens)")
                    return text
                except Exception as e:
                    errors.append(f"[{idx}] {provider.model}: {e}")
                    logger.warning(
                        "failover provider %s failed (%s); %s remaining — switch",
                        provider.model,
                        e,
                        len(self.providers) - idx - 1,
                    )
            raise RuntimeError(
                f"all {len(self.providers)} failover providers failed: {'; '.join(errors)}"
            )

        if budget:
            try:
                return await asyncio.wait_for(_run(), timeout=budget)
            except asyncio.TimeoutError as e:
                logger.warning(
                    "LLM failover total budget exceeded (%.0fs) models=%s",
                    budget,
                    self.model,
                )
                raise TimeoutError(
                    f"LLM failover total budget exceeded ({budget:.0f}s) models={self.model}"
                ) from e
        return await _run()


class MultiKeyFailoverLLMProvider(LLMProvider):
    """Rotate multiple API keys round-robin; inside each key try a model chain.

    Call flow per request:
      1. Pick next key in rotation (key1 -> key2 -> key1 ...).
      2. Try that key's model chain in order; first non-empty reply wins.
      3. If every model on the current key failed, fall back to the next key
         (its full model chain), until all keys are exhausted.
    """

    name = "multi_key_failover"

    def __init__(
        self,
        keys: list[str],
        providers_by_key: list[list[LLMProvider]],
        *,
        excluded_model_substr: tuple[str, ...] = ("deepseek",),
        total_budget: float | None = None,
    ) -> None:
        if len(keys) != len(providers_by_key) or not keys:
            raise ValueError("keys and providers_by_key must be same non-empty length")
        self.keys = keys
        self.providers_by_key = providers_by_key
        self.excluded_model_substr = excluded_model_substr
        self.total_budget = total_budget
        self._rotor = 0
        self.model = "|".join(p.model for chain in providers_by_key for p in chain)

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        budget = self.total_budget

        async def _run() -> str:
            n = len(self.keys)
            start = self._rotor % n
            self._rotor += 1
            all_errors: list[str] = []
            for step in range(n):
                key_idx = (start + step) % n
                key = self.keys[key_idx]
                chain = self.providers_by_key[key_idx]
                key_errors: list[str] = []
                for p in chain:
                    try:
                        text = await p.chat(
                            messages,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            response_format=response_format,
                        )
                        if (text or "").strip():
                            return text
                        raise RuntimeError("empty completion (reasoning may have consumed max_tokens)")
                    except Exception as e:
                        key_errors.append(f"{p.model}: {e}")
                        logger.warning(
                            "multikey key#%s model %s failed (%s); %s models left on key",
                            key_idx,
                            p.model,
                            e,
                            len(chain) - len(key_errors),
                        )
                err = f"key#{key_idx}({_mask_key(key)}): " + "; ".join(key_errors)
                all_errors.append(err)
                logger.warning(
                    "multikey key#%s all models failed (%s); %s keys remaining — switch key",
                    key_idx,
                    len(key_errors),
                    n - step - 1,
                )
            raise RuntimeError(
                f"all {len(self.keys)} api keys exhausted: {' || '.join(all_errors)}"
            )

        if budget:
            try:
                return await asyncio.wait_for(_run(), timeout=budget)
            except asyncio.TimeoutError as e:
                logger.warning(
                    "LLM multikey total budget exceeded (%.0fs) models=%s",
                    budget,
                    self.model,
                )
                raise TimeoutError(
                    f"LLM multikey total budget exceeded ({budget:.0f}s) models={self.model}"
                ) from e
        return await _run()


def _mask_key(key: str) -> str:
    if not key:
        return ""
    return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else key[:6] + "..."
