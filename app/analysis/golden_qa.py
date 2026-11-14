"""黄金问答库（方案一：把重推理搬到离线）。

背景：内网只有 qwen3-32b 时，让它在线做 DSH 式长链工具推理会大幅掉准确率。
本模块把「咀嚼项目」放到 ingest 期/离线批量任务：用（内网最强的）模型对每个
流程入口/关键接口生成高置信问答对，存 MySQL `golden_qa` 表；运行时对 ask 类
问题先做「语义+关键词」检索，命中高分直接返回存储答案 —— qwen3 只做检索与
模板组装这类轻活，准确率由离线生成质量保证。

离线生成（思考模式 + 自一致性投票）::

    python scripts/generate_golden_qa.py --project demo-loan

运行时接入点: ``app.agent.graph_agent._dispatch_by_intent``（ask 类问题先查本库）。
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.core.utils import truncate
from app.db.mysql import get_session
from app.llm.factory import get_embedding_provider, get_llm_provider
from app.models.orm import GoldenQAORM, SymbolDescORM, SymbolORM

logger = logging.getLogger("app.analysis.golden_qa")

_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_#\.\-]*")

SYSTEM_GENERATE = (
    "你是资深 Java 业务系统架构师。基于提供的项目材料，用中文回答用户问题。"
    "要求：1) 结构清晰，可用要点/编号；2) 只依据材料中的事实，禁止编造表名、"
    "字段、接口地址、类名；3) 材料不足时明确写「材料中未见…」；4) 面向业务同事，"
    "默认不堆砌类名，技术细节按需给出；5) 回答开头直接给结论，再展开依据。"
)

# ---------------------------------------------------------------------------
# 轻量中英分词（检索关键词重叠用；与 explore 的 _question_match_tokens 独立，
# 避免 app.agent -> app.analysis 反向依赖）
# ---------------------------------------------------------------------------


def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for m in _CJK_RE.findall(text or ""):
        if len(m) == 2:
            out.add(m)
        else:
            # 整词 + 双字滑窗（中文检索常用粒度）
            out.add(m)
            for i in range(len(m) - 1):
                out.add(m[i : i + 2])
    for m in _LATIN_RE.findall(text or ""):
        out.add(m.lower())
    return out


def _keyword_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / min(len(a), len(b))


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _embed_one_or_none(embed, text: str) -> list[float] | None:
    try:
        vecs = embed.embed([text])
        return vecs[0] if vecs else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 候选问题推导（从索引里挑「值得预生成答案」的问题）
# ---------------------------------------------------------------------------


def _flow_context(fd: dict[str, Any], ej: dict[str, Any], summary: str, cm: str) -> str:
    parts: list[str] = []
    narrative = str(fd.get("narrative") or "").strip()
    if narrative:
        parts.append(narrative)
    if summary:
        parts.append(f"入口摘要：{summary}")
    stages = fd.get("stages") or []
    if isinstance(stages, list) and stages:
        lines: list[str] = []
        for st in stages[:14]:
            if not isinstance(st, dict):
                continue
            cm2 = str(st.get("class_method") or "").strip()
            title = str(st.get("title") or "").strip()
            desc = str(
                st.get("desc_business") or st.get("desc") or st.get("summary") or ""
            ).strip()
            label = title if title and title != cm2 else cm2
            lines.append(f"- {label}" + (f"：{truncate(desc, 140)}" if desc else ""))
        if lines:
            parts.append("主链步骤：\n" + "\n".join(lines))
    if not parts:
        parts.append(f"入口：{cm}")
    return "\n\n".join(parts)


def candidate_questions_from_index(
    project_id: str, limit: int = 0
) -> list[dict[str, Any]]:
    """从索引推导候选问答。

    优先级：FlowDoc 流程入口（每题 2 问）→ 关键 Controller 方法（每题 1 问）。
    返回 [{question, entry_cm, context, kind}]。
    """
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    # 全部字段在 session 内落成纯 dict（get_session 默认 expire_on_commit=True，
    # session 外访问 ORM 属性会 DetachedInstanceError）
    flow_docs: list[dict[str, Any]] = []
    ctrl_cands: list[dict[str, Any]] = []
    try:
        with get_session() as session:
            rows = session.scalars(
                select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
            ).all()
            symbol_ids: list[str] = []
            for d in rows:
                ej = dict(d.extra_json or {})
                fd = ej.get("flow_doc")
                summary = str(d.summary or "")
                if (
                    isinstance(fd, dict)
                    and isinstance(fd.get("stages"), list)
                    and len(fd["stages"]) >= 3
                ):
                    flow_docs.append(
                        {
                            "cm": str(ej.get("class_method") or "").strip(),
                            "title": str(fd.get("title") or "").strip(),
                            "summary": summary,
                            "fd": fd,
                        }
                    )
                if not isinstance(fd, dict):
                    symbol_ids.append(d.symbol_id)
            if symbol_ids:
                syms = session.scalars(
                    select(SymbolORM).where(
                        SymbolORM.project_id == project_id,
                        SymbolORM.id.in_(symbol_ids),  # type: ignore[attr-defined]
                    )
                ).all()
                descs = {
                    x.symbol_id: x
                    for x in session.scalars(
                        select(SymbolDescORM).where(
                            SymbolDescORM.project_id == project_id,
                            SymbolDescORM.symbol_id.in_(symbol_ids),  # type: ignore[attr-defined]
                        )
                    ).all()
                }
                for s in syms:
                    kind = (s.kind or "").lower()
                    fqcn = s.fqcn or ""
                    if kind not in {"controller", "http入口"} and "controller" not in fqcn.lower():
                        continue
                    d = descs.get(s.id)
                    if d is None or not (d.summary or "").strip():
                        continue
                    ctrl_cands.append(
                        {
                            "simple": s.simple_name or fqcn.split(".")[-1],
                            "method": s.method_name or "",
                            "fqcn": fqcn,
                            "summary": str(d.summary or ""),
                            "domain": str(d.business_domain or "") if d.business_domain else "",
                            "paths": [
                                str(p)
                                for p in (dict(d.extra_json or {}).get("http_paths") or [])
                                if p
                            ],
                        }
                    )
    except Exception:
        logger.warning("golden_qa: read symbol_desc failed (index may be empty)", exc_info=True)
        return []

    # 1) FlowDoc 流程入口
    for fd in flow_docs:
        if limit and len(out) >= limit:
            break
        cm = fd["cm"]
        title = fd["title"]
        simple = cm.split("#", 1)[0].split(".")[-1] if "#" in cm else (cm.split(".")[-1] or "")
        method = cm.split("#", 1)[-1].split("(", 1)[0].strip() if "#" in cm else ""
        label = (title or f"{simple}#{method}" or cm).strip()
        ctx = _flow_context(fd["fd"], {"class_method": cm}, fd["summary"], cm or label)
        key = ("flow", cm or label)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "question": f"{label}的完整流程是什么样的？",
                "entry_cm": cm or None,
                "context": ctx,
                "kind": "flow",
            }
        )
        if limit and len(out) >= limit:
            break
        out.append(
            {
                "question": f"{label}这个接口是做什么的？入口地址和主要业务是什么？",
                "entry_cm": cm or None,
                "context": ctx,
                "kind": "flow",
            }
        )

    # 2) Controller 关键方法（有业务摘要才生成）
    for c in ctrl_cands:
        if limit and len(out) >= limit:
            break
        simple = c["simple"]
        method = c["method"]
        cm = f"{simple}#{method}" if method else simple
        key = ("ctrl", cm)
        if key in seen:
            continue
        seen.add(key)
        ctx_parts = [f"入口摘要：{c['summary']}"]
        if c["domain"]:
            ctx_parts.append(f"业务域：{c['domain']}")
        if c["paths"]:
            ctx_parts.append(f"入口地址：{', '.join(c['paths'])}")
        q = f"{cm} 这个接口是做什么的？" if method else f"{simple} 是做什么的？"
        out.append(
            {
                "question": q,
                "entry_cm": cm,
                "context": "\n".join(ctx_parts),
                "kind": "controller",
            }
        )
    return out


# ---------------------------------------------------------------------------
# 自一致性投票
# ---------------------------------------------------------------------------


def _consensus_pick(
    answers: list[str], embed
) -> tuple[str, float, list[str]]:
    """多份答案取「与其它答案平均相似度最高」的一份（self-consistency）。

    相似度用向量余弦；向量不可用时退化为长度接近中位数（防异常样本）。
    """
    if not answers:
        return "", 0.0, []
    n = len(answers)
    if n == 1:
        return answers[0], 1.0, answers
    try:
        vecs = embed.embed(answers)
        if len(vecs) == n:
            sims: list[list[float]] = [
                [_cosine(vecs[i], vecs[j]) for j in range(n)] for i in range(n)
            ]
            means = [
                sum(sims[i][j] for j in range(n) if j != i) / (n - 1) for i in range(n)
            ]
            best = max(range(n), key=lambda i: means[i])
            return answers[best], float(means[best]), answers
    except Exception:
        logger.debug("golden_qa consensus embed failed; fallback median", exc_info=True)
    lens = [len(a) for a in answers]
    med = sorted(lens)[n // 2]
    best = min(range(n), key=lambda i: abs(lens[i] - med))
    return answers[best], 0.5, answers


# ---------------------------------------------------------------------------
# 离线生成
# ---------------------------------------------------------------------------


def _existing_questions(project_id: str) -> set[str]:
    try:
        with get_session() as session:
            rows = session.scalars(
                select(GoldenQAORM.question).where(GoldenQAORM.project_id == project_id)
            ).all()
        return set(rows)
    except Exception:
        return set()


def _upsert_qa(
    project_id: str,
    question: str,
    answer: str,
    entry_cm: str | None,
    votes: float,
    samples: list[str],
    q_emb: list[float] | None,
    *,
    source: str = "consensus",
) -> bool:
    try:
        with get_session() as session:
            row = session.scalar(
                select(GoldenQAORM).where(
                    GoldenQAORM.project_id == project_id,
                    GoldenQAORM.question == question,
                )
            )
            if row is None:
                row = GoldenQAORM(
                    project_id=project_id,
                    question=question,
                    answer=answer,
                    entry_cm=entry_cm or None,
                    source=source,
                    votes=float(votes or 0.0),
                    samples_json=samples,
                    question_embedding_json=q_emb,
                )
                session.add(row)
            else:
                row.answer = answer
                if entry_cm:
                    row.entry_cm = entry_cm
                row.votes = float(votes or 0.0)
                row.samples_json = samples
                row.question_embedding_json = q_emb
                row.source = source
            session.commit()
        return True
    except Exception:
        logger.exception("golden_qa upsert failed for %s", question[:60])
        return False


def _generation_messages(question: str, context: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_GENERATE},
        {
            "role": "user",
            "content": (
                f"【项目上下文材料】\n{context}\n\n"
                f"【用户问题】\n{question}\n\n请给出准确、有条理的回答。"
            ),
        },
    ]


async def build_golden_qa(
    project_id: str,
    *,
    limit: int = 0,
    n_samples: int | None = None,
    overwrite: bool = False,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """离线生成黄金问答库（思考模式 + 自一致性投票）。

    返回统计：{total, created, skipped, failed, votes_min, votes_avg}。
    """
    settings = get_settings()
    n = int(n_samples or settings.golden_qa_n_samples or 3)
    max_ctx = int(settings.golden_qa_context_max_chars or 2600)
    candidates = candidate_questions_from_index(
        project_id, limit=int(limit or settings.golden_qa_max_candidates or 0)
    )
    existing = set() if overwrite else _existing_questions(project_id)
    llm = get_llm_provider()
    embed = get_embedding_provider()

    stats: dict[str, Any] = {
        "project_id": project_id,
        "total": len(candidates),
        "created": 0,
        "skipped": 0,
        "failed": 0,
        "votes_sum": 0.0,
        "votes_min": 1.0,
    }
    if not candidates:
        logger.warning("golden_qa: no candidates (index empty? run ingest first)")
        return stats

    for i, cand in enumerate(candidates, 1):
        q = str(cand.get("question") or "").strip()
        if not q:
            continue
        if q in existing:
            stats["skipped"] += 1
            continue
        context = truncate(str(cand.get("context") or ""), max_ctx)
        answers: list[str] = []
        for _ in range(n):
            try:
                a = (
                    await llm.chat(
                        _generation_messages(q, context),
                        temperature=0.35,
                        max_tokens=1400,
                    )
                ).strip()
                if a:
                    answers.append(a)
            except Exception:
                logger.warning("golden_qa generate failed for %s", q[:60], exc_info=True)
        if not answers:
            stats["failed"] += 1
            if on_progress:
                on_progress({"index": i, "total": len(candidates), "question": q, "status": "failed"})
            continue
        best, votes, samples = _consensus_pick(answers, embed)
        q_emb = _embed_one_or_none(embed, q)
        if _upsert_qa(
            project_id,
            q,
            best,
            str(cand.get("entry_cm") or "") or None,
            votes,
            samples,
            q_emb,
        ):
            stats["created"] += 1
            stats["votes_sum"] += votes
            stats["votes_min"] = min(stats["votes_min"], votes)
        else:
            stats["failed"] += 1
        if on_progress:
            on_progress(
                {
                    "index": i,
                    "total": len(candidates),
                    "question": q,
                    "status": "ok" if stats.get("created") else "failed",
                    "votes": round(votes, 3),
                }
            )
    created = stats["created"]
    stats["votes_avg"] = round(stats["votes_sum"] / created, 4) if created else 0.0
    return stats


# ---------------------------------------------------------------------------
# 运行时检索
# ---------------------------------------------------------------------------


def _table_name_overlap(q: str, sq: str) -> float:
    """「X表」短语同现比例：问题点名了某张表，存储问题也提到同一张表 → 强信号。

    解决 bge 语义相似度只看整体措辞、导致「放款表和优惠券表」命中到
    「支付信息表和优惠券表」（同构问句）的问题。
    """
    pat = r"[\u4e00-\u9fff]{1,6}表"

    def _tables(text: str) -> set[str]:
        out: set[str] = set()
        for m in re.findall(pat, text or ""):
            # 去掉连接词/定语粘连（「和优惠券表」「的放款表」→「优惠券表」「放款表」）
            t = re.sub(r"^(?:的|和|与|及|跟|同|一个|这张|该|此|其)", "", m)
            if len(t) >= 2:
                out.add(t)
        return out

    q_tables = _tables(q)
    sq_tables = _tables(sq)
    if not q_tables or not sq_tables:
        return 0.0
    inter = q_tables & sq_tables
    return min(1.0, len(inter) / len(q_tables))


# 问句意图 × 黄金答案类型匹配：问「字段/表结构」→ 只让字段黄金加分；
# 问「关联/关系」→ 只让关联黄金加分。防止字段问答与关联问答互相抢答。
_INTENT_SOURCE_BONUS: tuple[tuple[re.Pattern, str, float], ...] = (
    (re.compile(r"字段|表结构|有哪些列|字段列表|有哪些字段|表有哪些"), "table_field", 0.35),
    (re.compile(r"关联|关系|怎么关联|join|related|relations"), "table_relation", 0.35),
    (
        re.compile(
            r"流程|什么样|怎么走|链路|入口|主要做了什么|具体做了什么|做了什么|"
            r"主要做什么|是做什么的|是干啥的"
        ),
        "consensus",
        0.12,
    ),
    (re.compile(r"涉及哪些表|用到哪些表|涉及的表|用到了哪些表|操作了哪些表|都涉及哪些"), "interface_tables", 0.35),
)

# 互斥：问「XX 接口涉及哪些表 / 用到哪些表」时，单表字段/单对关联答案
# 必然答非所问 → 直接判 0 分排除，避免「放款接口都涉及哪些表」命中
# 「放款表有哪些字段」。与 _INTENT_SOURCE_BONUS 的加分互补（加分是偏好，这是硬排除）。
_QUESTION_TABLE_LIST_RE = re.compile(r"涉及哪些表|用到哪些表|用到了哪些表|操作了哪些表|都涉及哪些|哪些表$")
_EXCLUDED_SOURCES_FOR_TABLE_LIST = frozenset({"table_field", "table_relation"})

# 反向互斥：问「XX 是做什么的 / 流程 / 怎么走」等用途描述类问题时，
# interface_tables 的表清单答案答非所问 → 判 0 排除。只在这些问题明确点名
# 「哪些表」时才允许 interface_tables 参与（此时正向 bonus 已加分）。
# 「主要做了什么/做了些什么/具体做了什么」与「主要做什么」同为用途问法 —
# 缺「了」会让表清单答案命中（desktop-chat-264e7dfc904a: 相似度 0.69 命中错库）。
# 用途问法还须排除 table_field（字段清单）与 table_relation（关联关系）：
# 「结清证明申请接口 主要流程是什么」命中「结清证明发送表 表结构/有哪些字段」
# （desktop-chat-fb2229929b4f: 相似度 0.59 命中 27 个字段清单）。
_QUESTION_PURPOSE_RE = re.compile(
    r"是做什么的|做什么的|做了什么|做了些什么|主要做什么|主要做了什么|"
    r"具体做了什么|具体做了啥|是干嘛|干什么的|什么流程|主要流程|流程是什么样的|"
    r"怎么走|怎么运作|如何运作|大致流程|完整流程|讲讲流程|介绍一下|概述|"
    r"主要功能|是干啥的|干了什么|做了啥|在做什么"
)
_EXCLUDED_SOURCES_FOR_PURPOSE = frozenset(
    {"interface_tables", "table_field", "table_relation"}
)


def _intent_source_bonus(question: str, source: str) -> float:
    for pat, src, bonus in _INTENT_SOURCE_BONUS:
        if src == source and pat.search(question or ""):
            return bonus
    return 0.0


# 问「接口/任务」而非「字段/表结构」时，排除字段清单/关联关系/表清单类答案：
# 「结清证明相关的 都有哪几个接口和任务」命中「结清证明发送表 表字段清单」
# （desktop-chat-df4393665934: 相似度 0.59 → 错答 27 个字段）。
# 「结清证明申请接口都有哪些」命中不同业务的「提现申请」表清单
# （desktop-chat-e4472c4b89db: 相似度 0.58 → 错答 6 张提现申请表）。
_QUESTION_API_TASK_RE = re.compile(r"接口|API|任务|job|schedule")
_QUESTION_FIELD_STRUCT_RE = re.compile(r"字段|列|表结构")
_EXCLUDED_SOURCES_FOR_API_TASK = frozenset({"table_field", "table_relation", "interface_tables"})


def _hard_exclude(question: str, source: str) -> bool:
    """双向硬排除：表清单问法排除字段/关联答案；用途/流程问法排除表清单答案；
    接口/任务枚举问法排除字段清单/关联关系/表清单答案。"""
    q = question or ""
    if _QUESTION_TABLE_LIST_RE.search(q):
        return source in _EXCLUDED_SOURCES_FOR_TABLE_LIST
    if _QUESTION_PURPOSE_RE.search(q):
        return source in _EXCLUDED_SOURCES_FOR_PURPOSE
    if _QUESTION_API_TASK_RE.search(q) and not _QUESTION_FIELD_STRUCT_RE.search(q):
        return source in _EXCLUDED_SOURCES_FOR_API_TASK
    return False


def _score_row(
    rec: dict[str, Any],
    q_tokens: set[str],
    q_vec: list[float] | None,
    question: str = "",
) -> float:
    """语义（句向量余弦，0.6）+ 关键词重叠（0.4）+ 表名同现 + 意图×来源匹配。

    双向硬排除：
    - 问「接口涉及哪些表」→ 排除 table_field/table_relation（单表字段/单对关联）
    - 问「接口是做什么的/流程」→ 排除 interface_tables（表清单，答非所问）
    """
    source = str(rec.get("source") or "")
    if _hard_exclude(question, source):
        return 0.0
    sim = 0.0
    emb = rec.get("embedding")
    has_emb = q_vec is not None and isinstance(emb, list) and len(emb) == len(q_vec)
    if has_emb:
        sim = max(0.0, _cosine(q_vec, emb))
    kw = _keyword_overlap(q_tokens, _tokens(rec.get("question") or ""))
    score = 0.6 * sim + 0.4 * kw
    tb = _table_name_overlap(question, rec.get("question") or "")
    if tb > 0:
        score += 0.35 * tb
    score += _intent_source_bonus(question, source)
    return round(score, 4)


def search_golden_qa(
    project_id: str,
    question: str,
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """按语义+关键词相似度检索黄金问答库，按分数降序返回。"""
    settings = get_settings()
    k = int(top_k or settings.golden_qa_top_k or 3)
    records: list[dict[str, Any]] = []
    try:
        with get_session() as session:
            rows = session.scalars(
                select(GoldenQAORM).where(GoldenQAORM.project_id == project_id)
            ).all()
            # 在 session 内落成纯 dict（expire_on_commit 默认开启）
            for r in rows:
                records.append(
                    {
                        "question": r.question or "",
                        "answer": r.answer or "",
                        "entry_cm": r.entry_cm,
                        "source": r.source,
                        "votes": float(r.votes or 0.0),
                        "embedding": r.question_embedding_json,
                    }
                )
    except Exception:
        logger.debug("golden_qa search failed (table/index missing?)", exc_info=True)
        return []
    if not records:
        return []
    q_tokens = _tokens(question)
    q_vec = _embed_one_or_none(get_embedding_provider(), question)
    scored: list[tuple[float, dict[str, Any]]] = []
    for rec in records:
        s = _score_row(rec, q_tokens, q_vec, question=question)
        if s > 0:
            scored.append((s, rec))
    scored.sort(key=lambda x: x[0], reverse=True)
    out: list[dict[str, Any]] = []
    for s, rec in scored[:k]:
        out.append(
            {
                "question": rec["question"],
                "answer": rec["answer"],
                "entry_cm": rec["entry_cm"],
                "source": rec["source"],
                "votes": rec["votes"],
                "score": s,
            }
        )
    return out


def try_golden_answer(
    project_id: str,
    question: str,
    *,
    min_score: float | None = None,
) -> dict[str, Any] | None:
    """命中高分黄金答案则返回，否则 None（fail-open，绝不阻断正常管线）。"""
    settings = get_settings()
    if not settings.golden_qa_enabled:
        return None
    if not (question or "").strip():
        return None
    hits = search_golden_qa(project_id, question, top_k=1)
    if not hits:
        return None
    hit = hits[0]
    threshold = float(min_score if min_score is not None else settings.golden_qa_min_score)
    if float(hit.get("score") or 0.0) < threshold:
        return None
    return hit


# ---------------------------------------------------------------------------
# 人工正向反馈回填（source=manual_feedback）：用户点「有用」的问答对自动入库，
# 命中即免长链推理（快）+ 答案来自真实被接受的回答（准）。护栏见 record_confirmed_qa。
# ---------------------------------------------------------------------------

_FAILURE_MARKERS = (
    "未能完成检索",
    "处理失败",
    "未生成回复",
    "模型服务返回",
    "响应超时",
    "请稍后点重试",
    "好的，正在",
    "待确认改码",
    "已关闭 ask_retrieval",
)


def record_confirmed_qa(
    project_id: str | None,
    question: str,
    answer: str,
    *,
    entry_cm: str | None = None,
) -> dict[str, Any]:
    """用户正向反馈（vote=up）自动回填黄金问答库（source=manual_feedback）。

    best-effort：任何护栏/异常都返回 {ok: False, reason}，绝不抛错。
    重复问题不覆盖离线高置信答案（已有 source!=manual_feedback 且 votes>=0.7 时跳过）。
    """
    try:
        settings = get_settings()
    except Exception:
        return {"ok": False, "reason": "settings_unavailable"}
    if not settings.golden_qa_enabled or not settings.golden_qa_backfill_enabled:
        return {"ok": False, "reason": "disabled"}
    pid = (project_id or "").strip()
    if not pid:
        return {"ok": False, "reason": "no_project"}
    q = (question or "").strip()
    a = (answer or "").strip()
    if len(q) < 4 or len(q) > 240:
        return {"ok": False, "reason": "question_length"}
    if not a or len(a) < 40 or len(a) > 6000:
        return {"ok": False, "reason": "answer_length"}
    low = a.lower()
    for m in _FAILURE_MARKERS:
        if m in low:
            return {"ok": False, "reason": "failure_marker"}
    if "黄金问答库" in a:
        return {"ok": False, "reason": "already_golden"}
    try:
        with get_session() as session:
            row = session.scalar(
                select(GoldenQAORM).where(
                    GoldenQAORM.project_id == pid,
                    GoldenQAORM.question == q,
                )
            )
            if row is not None:
                src = str(row.source or "consensus")
                if src != "manual_feedback" and float(row.votes or 0.0) >= 0.7:
                    return {"ok": False, "reason": "existing_strong"}
    except Exception:
        logger.debug("golden_qa backfill probe failed", exc_info=True)
    q_emb = _embed_one_or_none(get_embedding_provider(), q)
    if _upsert_qa(
        pid,
        q,
        a,
        (entry_cm or "").strip() or None,
        1.0,
        [a],
        q_emb,
        source="manual_feedback",
    ):
        return {"ok": True, "reason": "inserted"}
    return {"ok": False, "reason": "upsert_failed"}


def golden_qa_stats(project_id: str | None = None) -> dict[str, Any]:
    """库存统计（供脚本/诊断）。"""
    try:
        with get_session() as session:
            q = select(GoldenQAORM)
            if project_id:
                q = q.where(GoldenQAORM.project_id == project_id)
            rows = session.scalars(q).all()
            total = len(rows)
            by_source: dict[str, int] = {}
            votes: list[float] = []
            for r in rows:
                src = r.source or "unknown"
                by_source[src] = by_source.get(src, 0) + 1
                if r.votes is not None:
                    votes.append(float(r.votes))
    except Exception:
        return {"ok": False, "error": "golden_qa table unavailable"}
    return {
        "ok": True,
        "total": total,
        "by_source": by_source,
        "votes_avg": round(sum(votes) / len(votes), 4) if votes else 0.0,
        "votes_min": round(min(votes), 4) if votes else 0.0,
    }
