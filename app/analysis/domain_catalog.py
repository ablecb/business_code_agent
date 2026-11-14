"""Dynamic business-domain catalog: reuse known labels, else create from code.

Prompts must not hardcode industry nouns. Domains are inferred from source and
accumulated in MySQL ``symbol_desc.business_domain`` (+ in-process cache).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from typing import Iterable

from sqlalchemy import select

from app.db.mysql import get_session
from app.models.orm import SymbolDescORM

logger = logging.getLogger("app.analysis.domain_catalog")

_LOCK = threading.Lock()
# project_id → ordered unique domains seen in this process
_CACHE: dict[str, list[str]] = {}

_GENERIC_FALLBACK = "未分类"
# Structural placeholders — not industry nouns; used only when LLM omitted domain.
_EMPTY_MARKERS = frozenset(
    {
        "",
        "其它",
        "其他",
        "未知",
        "无",
        "null",
        "none",
        "n/a",
        "未分类",
    }
)


def domain_instruction(known: Iterable[str] | None = None) -> str:
    """Prompt fragment: prefer existing domains, otherwise create from code."""
    names = [str(x).strip() for x in (known or []) if str(x).strip()]
    # Drop structural noise from the chooser list
    names = [n for n in names if n not in _EMPTY_MARKERS and n.lower() not in _EMPTY_MARKERS]
    if names:
        catalog = json.dumps(names, ensure_ascii=False)
        return (
            "业务分类规则（必须遵守）：\n"
            f"1) 优先从已有分类中原样选一项：{catalog}，并设 domain_action=\"reuse\"；\n"
            "2) 仅当源码语义与以上任一项都不贴合时，根据当前源码自行归纳一个新的短中文分类名"
            "（2～12字），设 domain_action=\"create\"；\n"
            "3) 禁止使用与当前源码无关的预设行业词；分类名必须能被源码中的类名/字段/注释支撑。"
        )
    return (
        "业务分类规则（必须遵守）：\n"
        "1) 尚无已有分类：请根据当前源码语义自行归纳一个短中文分类名（2～12字），"
        "设 domain_action=\"create\"；\n"
        "2) 禁止使用与当前源码无关的预设行业词；分类名必须能被源码中的类名/字段/注释支撑。"
    )


def list_known_domains(project_id: str, *, refresh: bool = False) -> list[str]:
    """Load distinct business_domain values for a project (DB + process cache)."""
    with _LOCK:
        cached = list(_CACHE.get(project_id) or [])
        if cached and not refresh:
            return list(cached)

    found: list[str] = []
    seen: set[str] = set()
    try:
        with get_session() as session:
            rows = session.scalars(
                select(SymbolDescORM.business_domain).where(
                    SymbolDescORM.project_id == project_id,
                    SymbolDescORM.business_domain.is_not(None),
                )
            ).all()
        for raw in rows:
            name = _normalize_name(str(raw or ""))
            if not name or name in _EMPTY_MARKERS:
                continue
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            found.append(name)
    except Exception:
        logger.debug("list_known_domains failed project=%s", project_id, exc_info=True)

    with _LOCK:
        # Merge DB + any domains created earlier in this process
        merged: list[str] = []
        seen2: set[str] = set()
        for n in list(_CACHE.get(project_id) or []) + found:
            k = n.casefold()
            if k in seen2:
                continue
            seen2.add(k)
            merged.append(n)
        _CACHE[project_id] = merged
        return list(merged)


def remember_domain(project_id: str, domain: str) -> str:
    """Register a domain into the in-process catalog; returns normalized name."""
    name = _normalize_name(domain)
    if not name or name in _EMPTY_MARKERS:
        name = _GENERIC_FALLBACK
    with _LOCK:
        cur = list(_CACHE.get(project_id) or [])
        if not any(x.casefold() == name.casefold() for x in cur):
            cur.append(name)
        _CACHE[project_id] = cur
    return name


def resolve_business_domain(
    project_id: str,
    proposed: str | None,
    *,
    domain_action: str | None = None,
    known: Iterable[str] | None = None,
) -> str:
    """Map LLM output onto catalog: reuse match if possible, else register new."""
    known_list = [str(x).strip() for x in (known or list_known_domains(project_id)) if str(x).strip()]
    name = _normalize_name(str(proposed or ""))
    action = (domain_action or "").strip().lower()

    if name and name not in _EMPTY_MARKERS:
        match = _match_known(name, known_list)
        if match:
            return remember_domain(project_id, match)
        if action == "reuse" and known_list:
            # Model claimed reuse but mismatched — pick closest exact-insensitive, else create
            match = _match_known(name, known_list)
            if match:
                return remember_domain(project_id, match)
        return remember_domain(project_id, name)

    if known_list:
        # No usable proposal: do not invent industry labels; keep unclassified
        return remember_domain(project_id, _GENERIC_FALLBACK)
    return remember_domain(project_id, _GENERIC_FALLBACK)


def _normalize_name(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"\s+", "", s)
    # Strip accidental enum-style pipes from old prompts
    if "|" in s:
        s = s.split("|")[0].strip()
    return s[:64]


def _match_known(name: str, known: list[str]) -> str | None:
    key = name.casefold()
    for k in known:
        if k.casefold() == key:
            return k
    # soft: known contains name or name contains known (short labels only)
    for k in known:
        if len(k) >= 2 and (k in name or name in k):
            return k
    return None
