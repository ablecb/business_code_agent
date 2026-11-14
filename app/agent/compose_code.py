"""Evidence-backed code composition (not raw paste).

User asks for a snippet that does X→Y (e.g. query table A then B). We retrieve
related Service/Mapper/Entity snippets, then LLM synthesizes code aligned with
project style. Intent/slots come from Goal LLM — no business-keyword routers.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.core.utils import truncate

logger = logging.getLogger("app.agent.compose_code")

_SYNTH_PROMPT = """你是资深 Java（MyBatis-Plus）助手。根据「用户意图」与「仓库证据」生成一段可放入 Service 的代码片段。
要求：
1. 必须按用户步骤实现（查表顺序、条件键、再查、分组等）；不要改成无关的在途校验/撤销流程。
2. 实体类名、字段 getter、表映射必须采用证据中出现的写法（QueryWrapper / LambdaQueryWrapper / Mapper.select*）。
   证据中标注「实体类定义」的类是仓库真实存在的类，生成代码只能引用这些类或其字段，禁止发明类名/表名。
3. 证据不足时可用合理占位（变量名/TODO），并在文首说明缺什么证据；禁止编造不存在的 Controller 业务方法当「原文」。
4. 输出 Markdown：先用 2～4 行说明「基于哪些证据生成」+ 步骤对齐；再给一个 ```java 代码块。
5. 代码块内只放一段连贯实现（方法体或完整小方法），不要大段复制无关校验。
"""


def _cm_simple(cm: str | None) -> str:
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    return f"{left.split('.')[-1].strip()}#{right.split('(', 1)[0].strip()}"


# 附属表（历史/明细/扩展/子表/计划/关系…）不参与实体兜底——
# 用户说「XX表」通常指主表；把 WithdrawalsApprovalReturnHis 这类
# 历史表喂给合成器只会让 LLM 引用错误实体。
# 注意：不在这里维护中文业务词词典——那是产品名词硬编码，违反
# tests/test_no_product_nouns.py 的架构约束；中文名来自离线 table_dict
# （app/analysis/table_dict.py 在 ingest 期从 @ApiModel/javadoc 提取）。
_SUB_TABLE_RE = re.compile(
    r"(?i)(_his|_return_his|_detail|_ext|_sub|_plan|_relation|_address|"
    r"_cust_info|_device_info|_apply_record|_param_log|_processor_lock)$"
)

# 只对「查表/取数据」形态的 compose ask 做实体补全；纯问关联/流程的
# （「放款表和还款表怎么关联的」）不该把一筐实体塞进证据。
_QUERY_SHAPE_RE = re.compile(
    r"查询|查一下|查\w*表|通过|根据|按|用|取|select|查|读取|获取|list|selectList|selectOne|"
    r"支用|落库|写入|更新|insert|update"
)


def _entity_snippets_from_question(
    project_id: str,
    question: str,
    *,
    max_entities: int = 4,
    max_chars_each: int = 3200,
) -> list[dict[str, Any]]:
    """Read real Entity class definitions referenced by Chinese table names.

    The user names tables in business language（放款表/放款优惠券表/放款审批表）.
    ``search_db_lookup_hits`` drops Entity placeholders (``CouponInfo#``), so the
    synthesis evidence never sees the actual entity fields — the LLM then
    invents placeholder classes (LoanCoupon/LoanApproval) or cites a real
    entity the verifier cannot confirm (CouponInfo flagged "未找到")
    (desktop-chat-ec81c911b4d8 / b58bb8b77eea).
    """
    from app.analysis.table_dict import load_table_dict

    try:
        rows = load_table_dict(project_id)
    except Exception:
        rows = []
    if not rows:
        return []
    from app.agent.source_view import _cjk_bigram_overlap

    q = (question or "").strip()
    segs = [s for s in re.split(r"[\s，,、；;。]", q) if len(s) >= 2]
    # only query/compose-shaped asks get entity augmentation — a pure
    # relationship/flow question should not dump a basket of entities in
    if not _QUERY_SHAPE_RE.search(q):
        return []
    matched: list[str] = []
    # 1) exact phrase containment per question segment first:
    #    「放款表」→ demo_b_withdrawals / Withdrawals, 「优惠券表」→ CouponInfo
    for r in rows:
        ent = str(r.get("entity_class") or "")
        nc = str(r.get("name_cn") or "")
        if not ent or not nc:
            continue
        if any(nc in seg or seg in nc for seg in segs) and ent not in matched:
            matched.append(ent)
    # 2) bigram fallback for tables still unmatched, requiring >=3 shared
    #    bigrams so generic words (放款 in OrgDayQuota/RetryLoan) cannot leak in
    #    (desktop-chat-ec81c911b4d8 evidence hygiene). Runs per-table, NOT only
    #    when nothing matched — 「放款审批表」 (name_cn=放款审批记录表) must still
    #    resolve even when 放款表/优惠券表 already matched.
    for r in rows:
        ent = str(r.get("entity_class") or "")
        nc = str(r.get("name_cn") or "")
        if not ent or not nc or ent in matched:
            continue
        if any(_cjk_bigram_overlap(nc, seg) >= 3 for seg in segs):
            matched.append(ent)
    out: list[dict[str, Any]] = []
    from app.agent.explore_source import _fetch_method_source_body

    for ent in matched[:max_entities]:
        try:
            code, path = _fetch_method_source_body(
                project_id, ent, max_chars=max_chars_each
            )
        except Exception:
            logger.debug("entity source read failed %s", ent, exc_info=True)
            code, path = "", None
        if not code:
            continue
        out.append(
            {
                "class_method": ent,
                "file_path": path,
                "code": code,
                "kind": "entity",
            }
        )
    return out


def gather_compose_evidence(
    project_id: str,
    question: str,
    *,
    lookup_key: str | None = None,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
    max_snippets: int = 6,
    max_chars_each: int = 3500,
) -> dict[str, Any]:
    """Hybrid-search + read method bodies for synthesis context."""
    from app.agent.explore_source import _fetch_method_source_body
    from app.agent.source_view import search_db_lookup_hits

    hits = search_db_lookup_hits(
        project_id,
        question,
        lookup_key=lookup_key,
        lookup_table=lookup_table,
        lookup_steps=lookup_steps,
        top_k=12,
    )
    snippets: list[dict[str, Any]] = _entity_snippets_from_question(
        project_id, question, max_entities=4
    )
    for cm in hits:
        c = _cm_simple(cm)
        if not c or "#" not in c:
            continue
        try:
            code, path = _fetch_method_source_body(
                project_id, c, max_chars=max_chars_each
            )
        except Exception:
            logger.debug("compose evidence read failed %s", c, exc_info=True)
            code, path = "", None
        if not code:
            continue
        snippets.append(
            {
                "class_method": c,
                "file_path": path,
                "code": truncate(code, max_chars_each),
            }
        )
        if len(snippets) >= max_snippets:
            break
    return {
        "hits": hits[:20],
        "snippets": snippets,
        "lookup_key": lookup_key,
        "lookup_table": lookup_table,
        "lookup_steps": list(lookup_steps or []),
    }


def _format_evidence_for_llm(pack: dict[str, Any]) -> str:
    lines: list[str] = []
    key = pack.get("lookup_key")
    table = pack.get("lookup_table")
    if key or table:
        lines.append(f"查询键: {key or '(未给)'}")
        lines.append(f"主表/实体: {table or '(未给)'}")
    steps = pack.get("lookup_steps") or []
    if steps:
        lines.append("步骤计划:")
        for i, step in enumerate(steps[:10], 1):
            if not isinstance(step, dict):
                continue
            bits = [f"op={step.get('op')}"]
            for k in ("table", "key", "key_from"):
                if step.get(k) and str(step.get(k)).lower() != "null":
                    bits.append(f"{k}={step[k]}")
            if step.get("fields"):
                bits.append("fields=" + ",".join(str(x) for x in step["fields"][:6]))
            if step.get("agg"):
                bits.append("agg=" + ",".join(str(x) for x in step["agg"][:6]))
            lines.append(f"  {i}. " + "; ".join(bits))
    snippets = pack.get("snippets") or []
    if not snippets:
        lines.append("仓库证据: (未读到方法体；请仍按步骤生成并标明证据不足)")
    else:
        lines.append("仓库证据片段:")
        for i, sn in enumerate(snippets, 1):
            kind = "（实体类定义）" if sn.get("kind") == "entity" else ""
            lines.append(
                f"--- 证据{i}: `{sn.get('class_method')}` {kind}"
                f"path={sn.get('file_path') or '-'} ---"
            )
            lines.append(str(sn.get("code") or "")[:3500])
    return "\n".join(lines)


async def synthesize_compose_answer(
    question: str,
    pack: dict[str, Any],
    *,
    history_text: str = "",
) -> str:
    from app.llm.factory import get_llm_provider

    evidence_txt = _format_evidence_for_llm(pack)
    user = (
        f"用户问题:\n{question}\n\n"
        f"对话历史(节选):\n{truncate(history_text or '(空)', 800)}\n\n"
        f"{evidence_txt}\n"
    )
    try:
        llm = get_llm_provider()
        content = await llm.chat(
            [
                {"role": "system", "content": _SYNTH_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=2200,
        )
        text = (content or "").strip()
        if text:
            return text
    except Exception:
        logger.debug("synthesize_compose_answer LLM failed", exc_info=True)

    # Deterministic fallback: show evidence + honest gap (no invented logic).
    refs = [
        str(s.get("class_method") or "")
        for s in (pack.get("snippets") or [])
        if s.get("class_method")
    ]
    lines = [
        "未能完成代码生成（模型不可用）。以下为检索到的相关证据，供参考：",
        "",
    ]
    if refs:
        lines.append("参考符号: " + ", ".join(f"`{r}`" for r in refs[:8]))
        lines.append("")
        sn0 = (pack.get("snippets") or [None])[0]
        if sn0 and sn0.get("code"):
            lines.append(f"证据 `{sn0.get('class_method')}`：")
            lines.append("```java")
            lines.append(str(sn0["code"]).rstrip())
            lines.append("```")
            lines.append("")
            lines.append(
                "说明：以上为仓库原文片段，不是按你的多步意图合成的实现；"
                "请稍后重试生成。"
            )
    else:
        lines.append("未检索到可用的 Service/Mapper 证据。")
    return "\n".join(lines)


async def run_compose_code(
    project_id: str,
    question: str,
    *,
    history_text: str = "",
    lookup_key: str | None = None,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
    view_kind: str | None = None,
) -> dict[str, Any]:
    """Gather evidence then synthesize. Returns {ok, answer, evidence, ...}."""
    pack = gather_compose_evidence(
        project_id,
        question,
        lookup_key=lookup_key,
        lookup_table=lookup_table,
        lookup_steps=lookup_steps,
    )
    pack["view_kind"] = view_kind
    answer = await synthesize_compose_answer(
        question, pack, history_text=history_text
    )
    ok = bool(answer and "```" in answer)
    return {
        "ok": ok,
        "answer": answer,
        "evidence": pack,
        "view_kind": view_kind or "compose",
        "refs": [s.get("class_method") for s in pack.get("snippets") or []],
    }
