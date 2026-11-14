"""进程内 LRU 缓存：embedding 与 hybrid_search 结果（问答双赢：快 + 口径稳定）。

设计要点：
- hybrid_search 命中缓存 → 跳过 embedding + Milvus + 一串 MySQL LIKE（快）；
  同一 (project_id, question) 返回同一批证据（稳）。
- 缓存的是 SymbolHit 的 model_dump 序列化，取用时重建 SymbolHit，避免调用方
  修改缓存对象造成别名污染。
- ingest 完成后必须调用 clear()（app.services.ingest 已接入），防止旧索引证据复用。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from threading import Lock
from typing import Any

from app.models.schemas import SymbolHit

logger = logging.getLogger("app.search_cache")

_MAX_EMBED = 512
_MAX_SEARCH = 256

_embed_cache: OrderedDict[str, list[float]] = OrderedDict()
_search_cache: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_lock = Lock()


def _get_lru(key: str, cache: OrderedDict) -> Any:
    with _lock:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def _put_lru(key: str, value: Any, cache: OrderedDict, maxsize: int) -> None:
    with _lock:
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > maxsize:
            cache.popitem(last=False)


def cached_embed(
    provider_name: str,
    model_name: str,
    text: str,
    *,
    maxsize: int = _MAX_EMBED,
) -> list[float] | None:
    """按 (provider, model, text) 缓存句向量。返回 None 表示未命中/不可缓存。"""
    if not text:
        return None
    key = f"{provider_name}\u0001{model_name}\u0001{text}"
    return _get_lru(key, _embed_cache)


def put_embed(
    provider_name: str,
    model_name: str,
    text: str,
    vec: list[float],
    *,
    maxsize: int = _MAX_EMBED,
) -> None:
    if not text or not vec:
        return
    key = f"{provider_name}\u0001{model_name}\u0001{text}"
    _put_lru(key, list(vec), _embed_cache, maxsize)


def _search_key(project_id: str, q: str, top_k: int, domain: str | None) -> str:
    return f"{project_id}\u0001{top_k}\u0001{domain or ''}\u0001{q}"


def get_search(
    project_id: str,
    q: str,
    top_k: int,
    domain: str | None = None,
    *,
    maxsize: int = _MAX_SEARCH,
) -> list[SymbolHit] | None:
    """命中返回重建的 SymbolHit 列表；未命中返回 None。"""
    if not (project_id or "").strip() or not (q or "").strip():
        return None
    raw = _get_lru(_search_key(project_id, q, top_k, domain), _search_cache)
    if raw is None:
        return None
    out: list[SymbolHit] = []
    for d in raw:
        try:
            out.append(SymbolHit(**d))
        except Exception:
            continue
    return out


def put_search(
    project_id: str,
    q: str,
    top_k: int,
    domain: str | None,
    hits: list[SymbolHit],
    *,
    maxsize: int = _MAX_SEARCH,
) -> None:
    if not (project_id or "").strip() or not (q or "").strip() or not hits:
        return
    raw = [h.model_dump() if hasattr(h, "model_dump") else dict(h) for h in hits]
    _put_lru(_search_key(project_id, q, top_k, domain), raw, _search_cache, maxsize)


def clear() -> None:
    """清空全部缓存（ingest 完成 / 索引变更后调用）。"""
    with _lock:
        _embed_cache.clear()
        _search_cache.clear()


def stats() -> dict[str, int]:
    with _lock:
        return {
            "embed": len(_embed_cache),
            "search": len(_search_cache),
        }