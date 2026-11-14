from __future__ import annotations

import hashlib
import logging
import math
import time
from typing import Iterable

import httpx

from app.llm.base import EmbeddingProvider
from app.llm.http_client import make_client

logger = logging.getLogger("app.llm.embedding")

_NO_EMBEDDING_HOST_MARKERS = ("deepseek.com",)


class LocalEmbeddingProvider(EmbeddingProvider):
    """Sentence-transformers local embeddings (default BGE). Lazy-loads model."""

    name = "local"

    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5", dim: int = 512) -> None:
        self.model_name = model_name
        self.dim = dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        model = self._get_model()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return [v.tolist() for v in vectors]


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    """Remote embeddings via OpenAI-compatible /v1/embeddings (SiliconFlow etc.)."""

    name = "openai_compatible"
    # SiliconFlow currently caps array length at 32.
    _max_batch = 32
    # BGE-zh models are typically ~512 tokens; keep chars conservative for CJK.
    _max_chars = 450

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        model: str = "BAAI/bge-large-zh-v1.5",
        dim: int = 1024,
        timeout: float = 60.0,
        dimensions: int | None = None,
        max_chars: int | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self.timeout = timeout
        # Optional; only some models (e.g. Qwen3-Embedding) honor this.
        self.dimensions = dimensions
        self.max_chars = int(max_chars or self._max_chars)
        if any(m in self.base_url.lower() for m in _NO_EMBEDDING_HOST_MARKERS):
            raise ValueError(
                f"Host {self.base_url!r} has no /embeddings API. "
                "Set EMBEDDING_BASE_URL to an embedding-capable endpoint "
                "(e.g. https://api.siliconflow.cn/v1) and EMBEDDING_API_KEY."
            )

    def _v1_base(self) -> str:
        if self.base_url.endswith("/v1"):
            return self.base_url
        return f"{self.base_url}/v1"

    @staticmethod
    def _sanitize(text: str, *, max_chars: int) -> str:
        s = (text or "").replace("\x00", " ").strip() or " "
        if len(s) > max_chars:
            s = s[:max_chars]
        return s

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        items = [self._sanitize(t, max_chars=self.max_chars) for t in texts]
        if not items:
            return []
        out: list[list[float]] = []
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._v1_base()}/embeddings"
        # Fast-fail connect (5s) so a dead host/IP (SYN hang) cannot stall the
        # whole ingest for minutes per batch; read/write keep the full timeout.
        timeout = httpx.Timeout(
            self.timeout,
            connect=5.0,
            read=self.timeout,
            write=self.timeout,
            pool=self.timeout,
        )
        with make_client(timeout=timeout) as client:
            for i in range(0, len(items), self._max_batch):
                batch = items[i : i + self._max_batch]
                vectors = self._embed_batch(client, url, headers, batch)
                out.extend(vectors)
        return out

    def _embed_batch(self, client, url: str, headers: dict, batch: list[str]) -> list[list[float]]:
        # Progressive shrink on 400 (token / payload limits).
        caps = [self.max_chars, 300, 200, 120]
        last_err: Exception | None = None
        for cap in caps:
            trimmed = [self._sanitize(t, max_chars=cap) for t in batch]
            payload: dict = {"model": self.model, "input": trimmed, "encoding_format": "float"}
            if self.dimensions is not None:
                payload["dimensions"] = self.dimensions
            resp = None
            for attempt in range(1, 6):
                resp = client.post(url, headers=headers, json=payload)
                if resp.status_code == 429 and attempt < 5:
                    sleep_s = min(40.0, 1.5 * (2.2**attempt))
                    logger.warning(
                        "embedding HTTP 429 attempt=%s/5 sleep=%.1fs model=%s",
                        attempt,
                        sleep_s,
                        self.model,
                    )
                    time.sleep(sleep_s)
                    continue
                break
            assert resp is not None
            if resp.status_code == 400:
                body = (resp.text or "")[:500]
                logger.warning(
                    "embedding 400 model=%s batch=%s cap=%s body=%s",
                    self.model,
                    len(trimmed),
                    cap,
                    body,
                )
                last_err = httpx.HTTPStatusError(
                    f"400 Bad Request for url '{url}'",
                    request=resp.request,
                    response=resp,
                )
                # Single-item fallback inside batch when still failing at small cap
                if cap <= 200 and len(trimmed) > 1:
                    vectors: list[list[float]] = []
                    for one in trimmed:
                        vectors.extend(self._embed_batch(client, url, headers, [one]))
                    return vectors
                continue
            try:
                resp.raise_for_status()
            except Exception as e:
                last_err = e
                logger.error(
                    "embedding HTTP %s model=%s body=%s",
                    getattr(resp, "status_code", "?"),
                    self.model,
                    (resp.text or "")[:500],
                )
                raise
            data = resp.json()["data"]
            data.sort(key=lambda row: row["index"])
            vectors = [row["embedding"] for row in data]
            if vectors:
                self.dim = len(vectors[0])
            return vectors
        if last_err:
            raise last_err
        return []


class SiliconFlowEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    name = "siliconflow"


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic fallback embedding when local model is unavailable."""

    name = "hash"

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            tokens = text.lower().replace(".", " ").replace("/", " ").split()
            if not tokens:
                tokens = [text or "empty"]
            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                for i in range(0, min(len(digest), 32)):
                    idx = (digest[i] + i * 17) % self.dim
                    vec[idx] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            result.append([x / norm for x in vec])
        return result
