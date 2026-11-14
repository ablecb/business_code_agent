from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config import Settings, get_settings
from app.llm.base import EmbeddingProvider, LLMProvider
from app.llm.circuit_breaker import CircuitBreakerLLMProvider
from app.llm.deepseek import (
    DeepSeekProvider,
    FailoverLLMProvider,
    MultiKeyFailoverLLMProvider,
    OpenAICompatibleProvider,
    SiliconFlowProvider,
)
from app.llm.embedding import (
    HashEmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    SiliconFlowEmbeddingProvider,
)
from app.llm.mock import MockLLMProvider
from app.llm.ollama import OllamaProvider

logger = logging.getLogger("app.llm")

_SILICONFLOW_ALIASES = {"siliconflow", "silicon", "silicon_flow", "硅基流动"}
_OPENAI_COMPAT_ALIASES = {"openai", "openai_compatible", "qwen", "tongyi"}
# Chat-only hosts — no /v1/embeddings.
_NO_EMBEDDING_HOST_MARKERS = ("deepseek.com",)


def build_llm_provider(
    settings: Settings | None = None, *, total_budget: float | None = None
) -> LLMProvider:
    """Build the chat LLM provider (main LLM_* config).

    ``total_budget``: 单次逻辑调用总预算（秒），None = 不限制。
    仅问答入口 get_llm_provider() 显式传入；parse 管线调用本函数时不传，
    保持离线解析不限时（单次解析可能超过几十秒）。
    """
    settings = settings or get_settings()
    provider = (settings.llm_provider or "siliconflow").lower()

    if provider == "mock":
        return MockLLMProvider()
    timeout = float(getattr(settings, "llm_timeout", 180.0) or 180.0)

    def _single(model: str) -> LLMProvider:
        if provider == "ollama":
            return OllamaProvider(
                model=model,
                base_url=settings.llm_base_url,
                api_key=settings.llm_api_key or "ollama",
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=timeout,
            )
        if provider in {"deepseek"}:
            return DeepSeekProvider(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=timeout,
                total_budget=total_budget,
            )
        if provider in _SILICONFLOW_ALIASES:
            return SiliconFlowProvider(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url or "https://api.siliconflow.cn/v1",
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=timeout,
                total_budget=total_budget,
            )
        if provider in _OPENAI_COMPAT_ALIASES:
            return OpenAICompatibleProvider(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=timeout,
                total_budget=total_budget,
            )
        raise ValueError(f"Unsupported LLM_PROVIDER: {provider}")

    primary = _single(settings.llm_model)
    raw_extra = (getattr(settings, "llm_failover_models", "") or "").strip()
    extra_models = [m.strip() for m in raw_extra.split(",") if m.strip()]
    if not extra_models:
        return primary
    chain = [primary]
    seen = {settings.llm_model}
    for m in extra_models:
        if m not in seen:
            seen.add(m)
            chain.append(_single(m))
    logger.info(
        "LLM failover chain: %s (provider=%s base=%s)",
        " -> ".join(p.model for p in chain),
        provider,
        settings.llm_base_url,
    )
    return FailoverLLMProvider(chain, total_budget=total_budget)


def _parse_api_keys(settings: Settings) -> list[str]:
    """Parse-only API keys: PARSE_LLM_API_KEYS (multi, rotation) or single key."""
    raw = (getattr(settings, "parse_llm_api_keys", "") or "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys
    single = (settings.parse_llm_api_key or "").strip()
    return [single] if single else []


def build_parse_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Parse-only LLM (ingest/topo/tagging/flow).

    Uses PARSE_LLM_* when configured; otherwise falls back to the main
    LLM_* provider so production (all-qwen3-32b) needs no extra config.

    Multi-key mode: PARSE_LLM_API_KEYS=key1,key2 rotates keys round-robin;
    inside a key the model chain (model + failover_models) is tried in order,
    and when a key's whole chain fails the next key is tried.
    DeepSeek models are excluded from the chain (PARSE mode wants non-deepseek).
    """
    settings = settings or get_settings()
    has_parse_cfg = any(
        (getattr(settings, f"parse_llm_{k}", "") or "").strip()
        for k in ("provider", "model", "base_url", "api_key", "api_keys")
    )
    if not has_parse_cfg:
        logger.info("parse LLM: no PARSE_LLM_* config → reuse main LLM provider")
        return build_llm_provider(settings)
    if not settings.parse_llm_base_url or not settings.parse_llm_model:
        raise ValueError(
            "parse LLM: PARSE_LLM_BASE_URL + PARSE_LLM_MODEL required when PARSE_LLM_* configured"
        )
    api_keys = _parse_api_keys(settings)
    if not api_keys:
        raise ValueError("parse LLM: PARSE_LLM_API_KEY or PARSE_LLM_API_KEYS required")

    def _chain_for_key(key: str) -> list[LLMProvider]:
        patched = settings.model_copy(
            update={
                "llm_provider": settings.parse_llm_provider or "openai_compatible",
                "llm_model": settings.parse_llm_model,
                "llm_base_url": settings.parse_llm_base_url,
                "llm_api_key": key,
                "llm_failover_models": settings.parse_llm_failover_models,
            }
        )
        provider = build_llm_provider(patched)
        chain = list(getattr(provider, "providers", [provider]))
        # Exclude deepseek models from parse chain.
        chain = [p for p in chain if "deepseek" not in (p.model or "").lower()]
        if not chain:
            raise ValueError("parse LLM chain empty after excluding deepseek models")
        return chain

    if len(api_keys) == 1:
        chain = _chain_for_key(api_keys[0])
        provider = chain[0] if len(chain) == 1 else FailoverLLMProvider(chain)
        logger.info(
            "parse LLM: single key chain=%s",
            " -> ".join(p.model for p in chain),
        )
        return provider

    providers_by_key = [_chain_for_key(k) for k in api_keys]
    logger.info(
        "parse LLM: multi-key rotation keys=%s chains=%s",
        len(api_keys),
        [ " -> ".join(p.model for p in chain) for chain in providers_by_key ],
    )
    return MultiKeyFailoverLLMProvider(api_keys, providers_by_key)


def _is_chat_only_host(base_url: str) -> bool:
    u = (base_url or "").lower()
    return any(m in u for m in _NO_EMBEDDING_HOST_MARKERS)


def _resolve_embedding_endpoint(settings: Settings) -> tuple[str, str, str]:
    """Return (provider, api_key, base_url) for embeddings.

    Independent of chat LLM:
      1) Prefer EMBEDDING_API_KEY / EMBEDDING_BASE_URL
      2) Else reuse LLM_* only when that host supports embeddings (not DeepSeek)
      3) SiliconFlow default base when provider is siliconflow
    """
    provider = (settings.embedding_provider or "siliconflow").lower().strip()
    api_key = (settings.embedding_api_key or "").strip()
    base_url = (settings.embedding_base_url or "").strip()
    llm_base = (settings.llm_base_url or "").strip()
    llm_key = (settings.llm_api_key or "").strip()

    if base_url and _is_chat_only_host(base_url):
        logger.warning(
            "EMBEDDING_BASE_URL=%s has no /embeddings; ignoring.",
            base_url,
        )
        base_url = ""

    if not base_url:
        if llm_base and not _is_chat_only_host(llm_base):
            base_url = llm_base
        elif provider in _SILICONFLOW_ALIASES:
            base_url = "https://api.siliconflow.cn/v1"

    if not api_key and llm_base and not _is_chat_only_host(llm_base):
        api_key = llm_key

    return provider, api_key, base_url


def build_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    provider, api_key, base_url = _resolve_embedding_endpoint(settings)

    if provider in {"hash", "deepseek"}:
        if provider == "deepseek":
            logger.warning(
                "DeepSeek has no /v1/embeddings. Use EMBEDDING_PROVIDER=siliconflow "
                "with EMBEDDING_API_KEY + EMBEDDING_BASE_URL, or hash/local."
            )
        logger.info("Embedding provider=hash dim=%s", settings.embedding_dim)
        return HashEmbeddingProvider(dim=settings.embedding_dim)

    if provider == "local":
        try:
            emb = LocalEmbeddingProvider(
                model_name=settings.embedding_model,
                dim=settings.embedding_dim,
            )
            logger.info("Embedding provider=local model=%s", settings.embedding_model)
            return emb
        except Exception:
            logger.warning("Local embedding unavailable; falling back to hash")
            return HashEmbeddingProvider(dim=settings.embedding_dim)

    if provider in _SILICONFLOW_ALIASES:
        if not api_key:
            logger.warning(
                "Embedding needs EMBEDDING_API_KEY (SiliconFlow). "
                "DeepSeek chat keys cannot call /embeddings. Falling back to hash.\n"
                "  EMBEDDING_PROVIDER=siliconflow\n"
                "  EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1\n"
                "  EMBEDDING_API_KEY=sk-xxx"
            )
            return HashEmbeddingProvider(dim=settings.embedding_dim)
        logger.info(
            "Embedding provider=siliconflow model=%s base_url=%s",
            settings.embedding_model,
            base_url,
        )
        return SiliconFlowEmbeddingProvider(
            api_key=api_key,
            base_url=base_url or "https://api.siliconflow.cn/v1",
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )

    if provider in _OPENAI_COMPAT_ALIASES:
        if not api_key or not base_url or _is_chat_only_host(base_url):
            logger.warning(
                "openai_compatible embedding requires EMBEDDING_API_KEY + "
                "EMBEDDING_BASE_URL (not DeepSeek). Falling back to hash."
            )
            return HashEmbeddingProvider(dim=settings.embedding_dim)
        logger.info(
            "Embedding provider=openai_compatible model=%s base_url=%s",
            settings.embedding_model,
            base_url,
        )
        return OpenAICompatibleEmbeddingProvider(
            api_key=api_key,
            base_url=base_url,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER: {provider}")


@lru_cache
def get_llm_provider() -> LLMProvider:
    # Chat 走主 LLM_* 配置（单模型，默认），并注入单次调用总预算
    # （LLM_TOTAL_TIMEOUT，默认 90s）—— 防止端点卡死时 300s 超时 × 5 重试
    # 把单轮问答拖到几十分钟~几小时（desktop-chat-016f43043466: 13min;
    # chat_3ff56a1cdc8b47298b43b240ade62232: 3.58h）。
    # 需要临时让问答也走 PARSE_LLM_*（sensenova 多 key 轮询）时设
    # CHAT_LLM_USE_PARSE=true（此时仍会带上总预算封顶）。
    #
    # 另外外包一层熔断器（CircuitBreakerLLMProvider）：连续失败超过阈值后
    # chat() 秒级快速失败，让启发式/确定性兜底接管，而不是每步都烧满预算
    # （desktop-chat-5b263372926e: 端点故障把一轮问答拖到 7.3 分钟）。
    settings = get_settings()
    total_budget = float(getattr(settings, "llm_total_timeout", 0.0) or 0.0) or None
    if bool(getattr(settings, "chat_llm_use_parse", False)):
        inner = _parse_llm_with_budget(total_budget)
    else:
        inner = build_llm_provider(settings, total_budget=total_budget)
    breaker = CircuitBreakerLLMProvider(
        inner,
        fail_threshold=getattr(settings, "llm_circuit_fail_threshold", 2),
        open_secs=getattr(settings, "llm_circuit_open_secs", 45.0),
        probe_timeout=getattr(settings, "llm_circuit_probe_timeout", 15.0),
    )
    return breaker


def _parse_llm_with_budget(total_budget: float | None) -> LLMProvider:
    """问答临时走 PARSE_LLM_* 时，把总预算注入多 key 链的最外层。"""
    settings = get_settings()
    has_parse_cfg = any(
        (getattr(settings, f"parse_llm_{k}", "") or "").strip()
        for k in ("provider", "model", "base_url", "api_key", "api_keys")
    )
    if not has_parse_cfg:
        return build_llm_provider(settings, total_budget=total_budget)
    api_keys = _parse_api_keys(settings)
    if not api_keys:
        return build_llm_provider(settings, total_budget=total_budget)

    def _chain_for_key(key: str) -> list[LLMProvider]:
        patched = settings.model_copy(
            update={
                "llm_provider": settings.parse_llm_provider or "openai_compatible",
                "llm_model": settings.parse_llm_model,
                "llm_base_url": settings.parse_llm_base_url,
                "llm_api_key": key,
                "llm_failover_models": settings.parse_llm_failover_models,
            }
        )
        provider = build_llm_provider(patched, total_budget=total_budget)
        chain = list(getattr(provider, "providers", [provider]))
        chain = [p for p in chain if "deepseek" not in (p.model or "").lower()]
        if not chain:
            raise ValueError("parse LLM chain empty after excluding deepseek models")
        return chain

    if len(api_keys) == 1:
        chain = _chain_for_key(api_keys[0])
        if len(chain) == 1:
            return chain[0]
        return FailoverLLMProvider(chain, total_budget=total_budget)
    providers_by_key = [_chain_for_key(k) for k in api_keys]
    return MultiKeyFailoverLLMProvider(
        api_keys, providers_by_key, total_budget=total_budget
    )


@lru_cache
def get_parse_llm_provider() -> LLMProvider:
    return build_parse_llm_provider()


@lru_cache
def get_embedding_provider() -> EmbeddingProvider:
    return build_embedding_provider()


def reset_provider_caches() -> None:
    """Call after .env / settings change (e.g. tests or hot-reload)."""
    get_llm_provider.cache_clear()
    get_parse_llm_provider.cache_clear()
    get_embedding_provider.cache_clear()
    get_settings.cache_clear()
