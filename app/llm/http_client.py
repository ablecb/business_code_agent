"""Shared httpx client options for LLM / embedding HTTP calls."""

from __future__ import annotations

import logging
import warnings

import httpx

logger = logging.getLogger("app.llm.http")

_ssl_off_warned = False


def http_verify_ssl() -> bool:
    try:
        from app.core.config import get_settings

        return bool(getattr(get_settings(), "http_verify_ssl", True))
    except Exception:
        return True


def httpx_client_kwargs(**extra) -> dict:
    """Build kwargs for httpx.Client / AsyncClient (honor HTTP_VERIFY_SSL).

    Default ``trust_env=False`` so Windows/system HTTP(S)_PROXY does not force
    LLM traffic through a broken TLS proxy (empty ConnectError). Callers may
    still pass ``trust_env=True`` / ``proxy=...`` explicitly.
    """
    global _ssl_off_warned
    verify = http_verify_ssl()
    if not verify:
        # Corporate proxy / hostname-mismatch environments
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        if not _ssl_off_warned:
            _ssl_off_warned = True
            logger.warning(
                "HTTP_VERIFY_SSL=false：已关闭证书校验（仅建议在本机/代理证书异常时使用；本进程只提示一次）"
            )
    kwargs: dict = {"verify": verify, "trust_env": False}
    kwargs.update(extra)
    return kwargs


def make_async_client(**extra) -> httpx.AsyncClient:
    return httpx.AsyncClient(**httpx_client_kwargs(**extra))


def make_client(**extra) -> httpx.Client:
    return httpx.Client(**httpx_client_kwargs(**extra))
