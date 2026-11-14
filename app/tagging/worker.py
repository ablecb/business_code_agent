from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy import delete, select

from app.core.config import get_settings
from app.core.utils import truncate
from app.db.milvus_client import get_milvus
from app.db.mysql import get_session
from app.llm.factory import get_embedding_provider, get_parse_llm_provider
from app.llm.tracing import traceable
from app.models.orm import SymbolDescORM, SymbolORM, SymbolTagORM, TagDictORM

logger = logging.getLogger("app.tagging")


@dataclass(frozen=True)
class _SymbolSnap:
    """Plain snapshot so tagging can run after the SQLAlchemy Session closes."""

    id: str
    kind: str
    fqcn: str
    simple_name: str
    method_name: str | None
    method_sig: str | None
    file_path: str
    annotations: Any
    raw_code: str | None


TAG_PROMPT = """你是资深 Java/Spring Cloud 代码分析专家。根据给定的类/方法信息，输出严格 JSON（不要 markdown）：
{{"tags":["标签1","标签2"],"summary":"一句话中文描述","business_domain":"短中文分类名","domain_action":"reuse|create"}}
标签要短、可检索，优先技术语义与职责（如 接口入口、参数校验、MQ消费、Feign调用、落库）。
域名与标签均从当前代码内容提炼，禁止使用与源码无关的预设行业词。
{domain_rule}
"""


def _snap(row: SymbolORM) -> _SymbolSnap:
    return _SymbolSnap(
        id=row.id,
        kind=row.kind,
        fqcn=row.fqcn,
        simple_name=row.simple_name,
        method_name=row.method_name,
        method_sig=row.method_sig,
        file_path=row.file_path,
        annotations=row.annotations,
        raw_code=row.raw_code,
    )


def _heuristic_tags(symbol: _SymbolSnap) -> dict:
    text = " ".join(
        filter(
            None,
            [
                symbol.fqcn,
                symbol.simple_name,
                symbol.method_name or "",
                symbol.kind,
                symbol.file_path,
            ],
        )
    ).lower()
    tags: list[str] = [symbol.kind]
    domain = "其他"
    rules = [
        (("controller", "restcontroller"), "接口", ["接口", "入口"]),
        (("service",), "公共", ["服务编排"]),
        (("mapper", "repository", "dao", "entity"), "持久化", ["落库", "持久化"]),
        (("rabbit", "listener", "mq", "kafka"), "异步", ["MQ", "异步"]),
        (("feign", "client"), "远程调用", ["Feign", "远程调用"]),
        (("valid", "check", "assert"), "校验", ["校验"]),
        (("query", "search", "list", "get"), "查询", ["查询"]),
    ]
    for keys, dom, add_tags in rules:
        if any(k in text for k in keys):
            domain = dom
            tags.extend(add_tags)
            break
    if symbol.kind == "Controller":
        tags.append("接口")
        domain = "接口" if domain == "其他" else domain
    if symbol.kind == "Listener":
        tags.append("MQ消费")
        domain = "异步" if domain == "其他" else domain
    seen = set()
    uniq = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    summary = f"{symbol.kind} {symbol.fqcn}" + (f"#{symbol.method_name}" if symbol.method_name else "")
    return {"tags": uniq[:8], "summary": summary, "business_domain": domain}


@traceable(name="tag_symbol_llm", run_type="llm")
async def _llm_tag(project_id: str, symbol: _SymbolSnap) -> dict:
    from app.analysis.domain_catalog import domain_instruction, list_known_domains

    llm = get_parse_llm_provider()
    known = list_known_domains(project_id)
    system = TAG_PROMPT.format(domain_rule=domain_instruction(known))
    code = truncate(symbol.raw_code or "", 2500)
    user = (
        f"kind={symbol.kind}\nfqcn={symbol.fqcn}\nmethod={symbol.method_name}\n"
        f"file={symbol.file_path}\nannotations={symbol.annotations}\ncode:\n{code}"
    )
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
        if not isinstance(data.get("tags"), list):
            raise ValueError("bad tags")
        return data
    except Exception:
        return _heuristic_tags(symbol)


def _ensure_tag(session, project_id: str, tag: str, category: str | None = None) -> int:
    row = session.scalar(
        select(TagDictORM).where(TagDictORM.project_id == project_id, TagDictORM.tag == tag)
    )
    if row:
        return row.id
    row = TagDictORM(project_id=project_id, tag=tag, category=category)
    session.add(row)
    session.flush()
    return row.id


def _persist_symbol_tags(
    project_id: str,
    sym: _SymbolSnap,
    data: dict,
) -> tuple[list[str], str, str, dict]:
    from app.analysis.domain_catalog import resolve_business_domain

    tags = [str(t).strip() for t in data.get("tags", []) if str(t).strip()]
    summary = str(data.get("summary") or "")
    domain = resolve_business_domain(
        project_id,
        str(data.get("business_domain") or ""),
        domain_action=str(data.get("domain_action") or ""),
    )
    with get_session() as session:
        session.merge(
            SymbolDescORM(
                project_id=project_id,
                symbol_id=sym.id,
                summary=summary,
                business_domain=domain,
                extra_json=data,
            )
        )
        session.execute(delete(SymbolTagORM).where(SymbolTagORM.symbol_id == sym.id))
        for tag in tags:
            tid = _ensure_tag(session, project_id, tag, category=domain)
            session.add(SymbolTagORM(project_id=project_id, symbol_id=sym.id, tag_id=tid))
    milvus_row = {
        "id": sym.id,
        "project_id": project_id,
        "symbol_id": sym.id,
        "fqcn": sym.fqcn[:500],
        "tags": ",".join(tags)[:1000],
        "text": truncate(
            f"{domain} {' '.join(tags)} {summary} {sym.fqcn} {sym.method_sig or ''}",
            2000,
        ),
    }
    return tags, summary, domain, milvus_row


@traceable(name="tag_one_symbol", run_type="chain")
async def _tag_one(
    project_id: str,
    sym: _SymbolSnap,
    *,
    use_llm: bool,
    sem: asyncio.Semaphore,
) -> dict:
    async with sem:
        data = await _llm_tag(project_id, sym) if use_llm else _heuristic_tags(sym)
    tags, _summary, _domain, milvus_row = await asyncio.to_thread(
        _persist_symbol_tags, project_id, sym, data
    )
    return {
        "symbol_id": sym.id,
        "fqcn": sym.fqcn,
        "method_name": sym.method_name,
        "tags": tags,
        "milvus_row": milvus_row,
    }


@traceable(name="tag_symbols", run_type="chain")
async def tag_symbols(
    project_id: str,
    *,
    batch_size: int = 20,
    use_llm: bool = True,
    only_kinds: set[str] | None = None,
    progress: Callable[[int, int, str], None] | None = None,
    concurrency: int | None = None,
) -> dict[str, int]:
    settings = get_settings()
    concurrency = max(1, int(concurrency or settings.tagging_concurrency or 1))
    only_kinds = only_kinds or {
        "Controller",
        "Service",
        "Listener",
        "Feign",
        "Job",
        "Mapper",
        "Entity",
    }
    with get_session() as session:
        rows = list(
            session.scalars(
                select(SymbolORM).where(
                    SymbolORM.project_id == project_id,
                    SymbolORM.method_name.is_not(None),
                )
            )
        )
        symbols = [_snap(s) for s in rows if s.kind in only_kinds]
        # Class-level Entity (no method) are handled in enrich_entities; skip here if method-less
        symbols = [s for s in symbols if s.method_name or s.kind != "Entity"]
        priority = {
            "Controller": 0,
            "Listener": 1,
            "Job": 2,
            "Service": 3,
            "Feign": 4,
            "Mapper": 5,
            "Entity": 6,
        }
        symbols.sort(key=lambda s: (priority.get(s.kind, 9), s.fqcn, s.method_name or ""))

    milvus = get_milvus()
    embedder = get_embedding_provider()
    tagged = 0
    total = len(symbols)
    sem = asyncio.Semaphore(concurrency)
    logger.info(
        "tag candidates=%s batch_size=%s concurrency=%s use_llm=%s kinds=%s",
        total,
        batch_size,
        concurrency,
        use_llm,
        sorted(only_kinds),
    )

    for i in range(0, total, batch_size):
        batch = symbols[i : i + batch_size]
        logger.info(
            "tagging batch %s-%s / %s (concurrency=%s) …",
            i + 1,
            min(i + batch_size, total),
            total,
            concurrency,
        )
        results = await asyncio.gather(
            *[_tag_one(project_id, sym, use_llm=use_llm, sem=sem) for sym in batch],
            return_exceptions=True,
        )
        milvus_rows: list[dict] = []
        for sym, result in zip(batch, results):
            if isinstance(result, Exception):
                logger.error(
                    "tag failed %s#%s: %s",
                    sym.fqcn,
                    sym.method_name,
                    result,
                    exc_info=result,
                )
                continue
            tagged += 1
            milvus_rows.append(result["milvus_row"])
            logger.info(
                "tagged mysql ok %s/%s %s#%s tags=%s",
                tagged,
                total,
                result["fqcn"],
                result["method_name"],
                result["tags"],
            )

        if milvus_rows:
            try:
                embeddings = embedder.embed([r["text"] for r in milvus_rows])
                milvus.dim = len(embeddings[0]) if embeddings else milvus.dim
                milvus.upsert(milvus_rows, embeddings)
            except Exception:
                logger.exception(
                    "milvus/embedding upsert failed for batch %s-%s; MySQL tags kept",
                    i + 1,
                    min(i + batch_size, total),
                )

        if progress:
            progress(min(i + batch_size, total), total, f"tagged batch {i // batch_size + 1}")

    return {"tagged": tagged, "total_candidates": total, "concurrency": concurrency}
