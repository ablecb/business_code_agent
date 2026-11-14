from __future__ import annotations

from app.core.config import get_settings
from app.llm.factory import build_embedding_provider, build_llm_provider


def test_swap_provider_to_mock(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    get_settings.cache_clear()
    settings = get_settings()
    llm = build_llm_provider(settings)
    assert llm.name == "mock"


def test_siliconflow_llm_provider():
    settings = type(
        "S",
        (),
        {
            "llm_provider": "siliconflow",
            "llm_api_key": "sk-test",
            "llm_base_url": "https://api.siliconflow.cn/v1",
            "llm_model": "Qwen/Qwen3-32b",
            "llm_temperature": 0.2,
            "llm_max_tokens": 4096,
        },
    )()
    llm = build_llm_provider(settings)
    assert llm.name == "siliconflow"
    assert llm._v1_base() == "https://api.siliconflow.cn/v1"


def test_siliconflow_embedding_reuses_llm_key():
    settings = type(
        "S",
        (),
        {
            "embedding_provider": "siliconflow",
            "embedding_model": "BAAI/bge-large-zh-v1.5",
            "embedding_dim": 1024,
            "embedding_api_key": "",
            "embedding_base_url": "",
            "llm_api_key": "sk-shared",
            "llm_base_url": "https://api.siliconflow.cn/v1",
        },
    )()
    emb = build_embedding_provider(settings)
    assert emb.name == "siliconflow"
    assert emb.api_key == "sk-shared"
    assert emb.dim == 1024


def test_hash_embedding():
    emb = build_embedding_provider(
        type(
            "S",
            (),
            {
                "embedding_provider": "hash",
                "embedding_dim": 512,
                "embedding_model": "x",
                "embedding_api_key": "",
                "embedding_base_url": "",
                "llm_api_key": "",
                "llm_base_url": "",
            },
        )()
    )
    vecs = emb.embed(["创建订单接口", "支付回调"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 512


def test_deepseek_llm_does_not_reuse_key_for_embedding():
    """DeepSeek chat credentials must not be used for /embeddings."""
    emb = build_embedding_provider(
        type(
            "S",
            (),
            {
                "embedding_provider": "siliconflow",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dim": 1024,
                "embedding_api_key": "",
                "embedding_base_url": "https://api.siliconflow.cn/v1",
                "llm_api_key": "sk-deepseek",
                "llm_base_url": "https://api.deepseek.com",
            },
        )()
    )
    assert emb.name == "hash"


def test_dedicated_embedding_credentials():
    emb = build_embedding_provider(
        type(
            "S",
            (),
            {
                "embedding_provider": "siliconflow",
                "embedding_model": "BAAI/bge-m3",
                "embedding_dim": 1024,
                "embedding_api_key": "sk-embed",
                "embedding_base_url": "https://api.siliconflow.cn/v1",
                "llm_api_key": "sk-deepseek",
                "llm_base_url": "https://api.deepseek.com",
            },
        )()
    )
    assert emb.name == "siliconflow"
    assert emb.api_key == "sk-embed"
    assert "siliconflow" in emb.base_url


def test_reject_deepseek_embedding_base_url():
    import pytest
    from app.llm.embedding import OpenAICompatibleEmbeddingProvider

    with pytest.raises(ValueError, match="no /embeddings"):
        OpenAICompatibleEmbeddingProvider(
            api_key="sk-x",
            base_url="https://api.deepseek.com/v1",
            model="x",
        )
