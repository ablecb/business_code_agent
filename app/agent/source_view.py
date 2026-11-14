"""Generic「用户想看哪一块」→ AI 定位目标 → 读文件/方法返回.

No business-domain step routers. Context (prior answer, symbols, entry)
is passed in; the LLM chooses method / whole file / config; we only fetch.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.utils import truncate

logger = logging.getLogger("app.agent.source_view")

_PLAN_PROMPT = """你是代码定位助手。根据用户当前问题、对话上下文与候选证据，判断用户想看的「那一块」内容。
只输出 JSON（不要 markdown）：
{
  "view_kind": "method|file|config|db_lookup|db_pipeline|unknown",
  "class_method": "SimpleClass#method 或空",
  "fqcn": "可选全限定类名或简单类名",
  "method_name": "可选方法名",
  "file_path": "可选相对工程路径，含 .java/.yml/.yaml/.xml/.properties 等",
  "config_key": "可选配置键/片段提示",
  "reason": "一句中文说明为何是这块"
}

规则：
1. 用户要看源码/代码/怎么写的：优先 method 或 file，禁止编造不存在的类名方法名。
2. class_method / file_path / fqcn 必须来自「候选符号」或「检索命中」列表；没有则 view_kind=unknown。
3. 所选目标必须与用户问题语义一致：问题里的关键名词应对上方法名/摘要；
   不可把问题中的 A 概念答成无关的 B 方法（即使 B 也在候选里）。
4. 若上下文给出「聚焦目标」，优先选它（追问「这个方法/发代码」时不要改选无关检索 top1）。
5. 用户指上一答第 N 步：按该步标题与候选/检索命中对齐到最贴切 Class#method。
6. 整类/配置文件：view_kind=file 或 config。
7. 拼写错误按常见技术词纠正理解（如 controller），不要拒绝。
8. 若补充说明标明 db_lookup / db_pipeline（按字段查表 / 多步查库）：
   - 优先选 ServiceImpl/Mapper 中真正做 select/QueryWrapper/eq/分组的方法，禁止选无关 HTTP 撤销/适配器方法；
   - db_pipeline 优先选「同方法内多段查库或 Mapper XML 联表/分组」的完整方法；
   - 仍必须从候选/检索命中里选，禁止编造。
"""


def _parse_json_obj(content: str) -> dict[str, Any]:
    text = (content or "").strip()
    if not text:
        return {}
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0) if m else text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _simple_cm(cm: str | None) -> str:
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    return f"{left.split('.')[-1].strip()}#{right.split('(', 1)[0].strip()}"


def _search_query_from_question(question: str) -> str:
    """Strip chat glue; keep nouns for hybrid search (no domain lexicon)."""
    text = question or ""
    text = re.sub(
        r"我想看看|我想看|帮我看|看看|看下|代码|怎么写的|怎么写|"
        r"那一块|那块|对应的|流程中|请|一下",
        " ",
        text,
    )
    return re.sub(r"\s+", " ", text).strip() or (question or "").strip()


def search_symbol_hits_for_question(
    project_id: str, question: str, *, top_k: int = 12
) -> list[str]:
    """Hybrid search → Class#method list (embedding/lexical, no business map)."""
    from app.agent.explore_shared import _method_name_from_hit
    from app.agent.explore_tools import search_symbols

    q = _search_query_from_question(question)
    if not q or not project_id:
        return []
    out: list[str] = []
    try:
        sr = search_symbols(project_id, q, top_k=top_k)
        for h in sr.get("hits") or []:
            if not isinstance(h, dict):
                continue
            fqcn = str(h.get("fqcn") or "")
            mname = _method_name_from_hit(h)
            simple = fqcn.split(".")[-1]
            if simple and mname:
                out.append(f"{simple}#{mname}")
    except Exception:
        logger.debug("search_symbol_hits_for_question failed", exc_info=True)
    return list(dict.fromkeys(out))


def _lookup_search_queries(
    *,
    question: str,
    lookup_key: str | None = None,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Compose hybrid-search queries from Goal-LLM slots only (no domain lexicon)."""
    qs: list[str] = []
    key = (lookup_key or "").strip()
    table = (lookup_table or "").strip()
    if table and key:
        qs.append(f"{table} {key}")
    if table:
        qs.append(table)
    if key:
        qs.append(key)
    for step in lookup_steps or []:
        if not isinstance(step, dict):
            continue
        parts = [
            str(step.get(k) or "").strip()
            for k in ("table", "key", "key_from", "op")
            if step.get(k)
        ]
        for fl in list(step.get("fields") or []) + list(step.get("agg") or []):
            if fl:
                parts.append(str(fl).strip())
        blob = " ".join(p for p in parts if p)
        if blob:
            qs.append(blob)
    # Fall back to stripped user question nouns (still not a business word table).
    q0 = _search_query_from_question(question)
    if q0:
        qs.append(q0)
    # Dedupe preserve order
    out: list[str] = []
    seen: set[str] = set()
    for q in qs:
        qn = re.sub(r"\s+", " ", q).strip()
        if qn and qn.lower() not in seen:
            seen.add(qn.lower())
            out.append(qn)
    return out[:8]


def _cjk_bigram_overlap(a: str, b: str) -> int:
    """Count shared CJK bigrams between two Chinese phrases (>=2 → related)."""
    a2 = {a[i : i + 2] for i in range(len(a) - 1) if re.match(r"[\u4e00-\u9fff]", a[i])}
    b2 = {b[i : i + 2] for i in range(len(b) - 1) if re.match(r"[\u4e00-\u9fff]", b[i])}
    return len(a2 & b2)


def _augment_with_table_dict(
    project_id: str,
    queries: list[str],
    *,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
    question: str = "",
) -> list[str]:
    """Map user Chinese table phrases to real entities via the table_dict.

    e.g. a Chinese business-table phrase overlapping a table_dict ``name_cn``
    yields the real entity class / physical table / conventional Service name,
    so compose_code evidence retrieval hits the actual repo classes instead of
    fabricating placeholder names.

    Maps BOTH the primary lookup_table AND every lookup_steps[].table — a
    multi-step query pipeline (查放款表 → 查放款优惠券表 → 查放款审批表) needs
    each step's Chinese table phrase resolved to a real entity, otherwise the
    LLM composes placeholders (LoanCoupon / LoanApproval) for the later steps
    (desktop-chat-b58bb8b77eea).
    """
    if not project_id:
        return queries
    phrase = (lookup_table or "").strip()
    q_extra = (question or "").strip()
    step_tables: list[str] = []
    for st in lookup_steps or []:
        if not isinstance(st, dict):
            continue
        t = str(st.get("table") or "").strip()
        if t:
            step_tables.append(t)
    if not phrase and not q_extra and not step_tables:
        return queries
    try:
        from app.analysis.table_dict import load_table_dict

        rows = load_table_dict(project_id)
    except Exception:
        return queries
    if not rows:
        return queries

    out = list(queries)
    seen = {q.lower() for q in out}
    # Collect matching rows (Chinese table name overlaps user phrase/question/steps,
    # OR Latin entity/table name overlaps Latin tokens in the ask — e.g.
    # 「放款审批表」 has no name_cn but the physical table demo_b_withdrawals_approval
    # and entity WithdrawalsApproval exist, so ApprovalService/WithdrawalsApproval
    # queries must be emitted instead of placeholder names).
    matched_entities: list[str] = []
    latin_ask = [t.lower() for t in _latin_tokens(q_extra)]
    candidates = [p for p in ([phrase] + step_tables) if p] or [q_extra]
    for cand in candidates:
        if not cand:
            continue
        for r in rows:
            nc = str(r.get("name_cn") or "")
            ent = str(r.get("entity_class") or "")
            tbl = str(r.get("table_name") or "")
            if not ent:
                continue
            hit = bool(nc) and (
                nc in cand or cand in nc or _cjk_bigram_overlap(nc, cand) >= 2
            )
            if not hit and q_extra and nc:
                hit = (
                    nc in q_extra
                    or any(
                        _cjk_bigram_overlap(nc, seg) >= 2
                        for seg in re.split(r"[\s，,、]", q_extra)
                        if len(seg) >= 2
                    )
                )
            if not hit and latin_ask:
                # Latin bridge: entity/table token (approval/coupon/loan…) appears
                # in the ask's Latin tokens or vice versa.
                ent_l = ent.lower()
                tbl_l = tbl.lower()
                hit = any(
                    tok and (tok in ent_l or tok in tbl_l) for tok in latin_ask
                )
            if not hit:
                continue
            matched_entities.append(ent)
            for name in (ent, tbl):
                if name and name.lower() not in seen:
                    out.append(name)
                    seen.add(name.lower())
    # Entity → conventional Service name (CouponInfo → CouponInfoService).
    for ent in matched_entities:
        svc = f"{ent}Service"
        if svc.lower() not in seen:
            out.append(svc)
            seen.add(svc.lower())
    return out[:16]


def search_db_lookup_hits(
    project_id: str,
    question: str,
    *,
    lookup_key: str | None = None,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
    top_k: int = 10,
) -> list[str]:
    """Search Service/Mapper(/Entity) using LLM-extracted lookup slots."""
    from app.agent.explore_shared import _method_name_from_hit
    from app.agent.explore_tools import search_symbols

    if not project_id:
        return []
    queries = _lookup_search_queries(
        question=question,
        lookup_key=lookup_key,
        lookup_table=lookup_table,
        lookup_steps=lookup_steps,
    )
    queries = _augment_with_table_dict(
        project_id,
        queries,
        lookup_table=lookup_table,
        lookup_steps=lookup_steps,
        question=question,
    )
    out: list[str] = []
    kinds = ("Service", "Mapper", "Entity", None)
    for q in queries:
        for kind in kinds:
            try:
                kwargs: dict[str, Any] = {"top_k": top_k}
                if kind:
                    kwargs["kind"] = kind
                sr = search_symbols(project_id, q, **kwargs)
            except Exception:
                logger.debug("search_db_lookup_hits failed q=%s", q, exc_info=True)
                continue
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "")
                mname = _method_name_from_hit(h)
                simple = fqcn.split(".")[-1]
                if simple and mname:
                    out.append(f"{simple}#{mname}")
                elif simple and kind == "Entity":
                    # Entity hit without method — still useful as class hint for locator.
                    out.append(f"{simple}#")
    # Drop trailing-empty-method placeholders after collecting real methods
    cleaned: list[str] = []
    seen: set[str] = set()
    for cm in out:
        if cm.endswith("#"):
            continue
        if cm not in seen:
            seen.add(cm)
            cleaned.append(cm)
    # Prefer ServiceImpl/Mapper/Service leaves for code composition — a
    # Controller HTTP adapter (checkApplyReqParam etc.) has no entity/table
    # evidence and makes the LLM fabricate placeholder classes
    # (desktop-chat-b58bb8b77eea: LoanCoupon/LoanApproval). Implementations
    # move to the front; pure-Controller noise sinks to the back.
    impl_key = re.compile(r"(?i)(ServiceImpl|Service|Mapper|Repository|Dao|Client|Listener|Job|Handler|Processor)")
    order = {cm: i for i, cm in enumerate(cleaned)}
    cleaned.sort(key=lambda cm: (0 if impl_key.search(cm.split("#", 1)[0]) else 1, order[cm]))
    return cleaned[:40]


def build_db_lookup_notes(
    *,
    view_kind: str | None,
    lookup_key: str | None = None,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
) -> str:
    """Serialize Goal lookup slots for the locator LLM (no keyword judgment)."""
    vk = (view_kind or "").strip().lower()
    if vk not in {"db_lookup", "db_pipeline"}:
        return ""
    lines = [
        f"查库贴码模式: {vk}",
        "定位目标：实现该查询的 ServiceImpl/Mapper 方法（或含 join/group 的 Mapper XML 对应方法）。",
        "禁止选无关 Controller 撤销/HTTP 适配器方法。",
    ]
    if lookup_key:
        lines.append(f"查询键(用户原话): {lookup_key}")
    if lookup_table:
        lines.append(f"目标表/实体(用户原话): {lookup_table}")
    if lookup_steps:
        for i, step in enumerate(lookup_steps[:8], 1):
            if not isinstance(step, dict):
                continue
            bits = [f"op={step.get('op')}"]
            for k in ("table", "key", "key_from"):
                if step.get(k):
                    bits.append(f"{k}={step[k]}")
            if step.get("fields"):
                bits.append("fields=" + ",".join(str(x) for x in step["fields"][:6]))
            if step.get("agg"):
                bits.append("agg=" + ",".join(str(x) for x in step["agg"][:6]))
            lines.append(f"步骤{i}: " + "; ".join(bits))
    return "\n".join(lines)


def _latin_tokens(question: str) -> list[str]:
    return [
        m.group(0)
        for m in re.finditer(r"[A-Za-z][A-Za-z0-9_]{2,40}", question or "")
    ]


def _cm_matches_latin(cm: str, latin: list[str]) -> bool:
    """True when Class#method overlaps a Latin token from the question (language-agnostic)."""
    text = _simple_cm(cm)
    if not text or "#" not in text or not latin:
        return False
    cls, meth = text.split("#", 1)
    cls_l, meth_l = cls.lower(), meth.lower()
    for t in latin:
        tl = t.lower()
        if tl == meth_l or tl in meth_l or meth_l in tl or tl in cls_l:
            return True
    return False


def _matches_preferred(cm: str, preferred: list[str]) -> bool:
    text = _simple_cm(cm)
    if not text or not preferred:
        return False
    if text in preferred:
        return True
    if "#" not in text:
        return False
    meth = text.split("#", 1)[-1].lower()
    return any(
        p.split("#", 1)[-1].lower() == meth for p in preferred if "#" in p
    )


def _is_strong_source_target(
    cm: str,
    question: str,
    *,
    preferred: list[str] | None = None,
) -> bool:
    """Intent / dig-in focus / Latin method token → do not replace with search top hits."""
    text = _simple_cm(cm)
    if not text or "#" not in text:
        return False
    pref = [_simple_cm(p) for p in (preferred or []) if p and "#" in str(p)]
    if _matches_preferred(text, pref):
        return True
    return _cm_matches_latin(text, _latin_tokens(question))


def prioritize_symbols_for_question(
    question: str,
    symbols: list[str] | None,
    *,
    search_hits: list[str] | None = None,
    preferred_cms: list[str] | None = None,
    limit: int = 40,
) -> list[str]:
    """Prefer dig-in/intent hints, then hybrid-search hits, then remaining symbols."""
    preferred = [
        _simple_cm(s) for s in (preferred_cms or []) if s and "#" in str(s)
    ]
    hits = [_simple_cm(s) for s in (search_hits or []) if s and "#" in str(s)]
    rest = [_simple_cm(str(s)) for s in (symbols or []) if s and "#" in str(s)]
    latin = [t.lower() for t in _latin_tokens(question)]
    if latin:
        rest = sorted(
            rest,
            key=lambda cm: (
                0
                if any(t in cm.lower().split("#", 1)[-1] for t in latin)
                else 1,
                cm,
            ),
        )
    merged: list[str] = []
    seen: set[str] = set()
    for s in preferred + hits + rest:
        if s and s not in seen:
            seen.add(s)
            merged.append(s)
    return merged[:limit]


def refine_source_view_plan(
    question: str,
    plan: dict[str, Any],
    symbols: list[str] | None,
    *,
    search_hits: list[str] | None = None,
    preferred_cms: list[str] | None = None,
) -> dict[str, Any]:
    """Gate search-hit correction: keep strong hints; refine only empty/weak/absent picks.

    Never replace a semantically consistent Class#method (intent / dig-in focus /
    Latin token in question) with an unrelated retrieval top hit.
    """
    hits = [_simple_cm(s) for s in (search_hits or []) if s and "#" in str(s)]
    pool = {_simple_cm(str(s)) for s in (symbols or []) if s and "#" in str(s)}
    preferred = [
        _simple_cm(s) for s in (preferred_cms or []) if s and "#" in str(s)
    ]
    picked = _simple_cm(str(plan.get("class_method") or ""))

    # Prefer / restore dig-in or intent Class#method over unrelated hits.
    if preferred and not _matches_preferred(picked, preferred):
        out = dict(plan)
        out["class_method"] = preferred[0]
        out["view_kind"] = "method"
        out["ok"] = True
        out["reason"] = (
            (out.get("reason") or "")
            + f"（沿用聚焦目标 `{preferred[0]}`）"
        ).strip()
        out["refined"] = "prefer_focus"
        return out

    if picked and _is_strong_source_target(
        picked, question, preferred=preferred
    ):
        return plan

    if not hits:
        return plan
    # Keep LLM pick when it is among top search hits
    if picked and picked in hits[:5]:
        return plan
    # Keep in-evidence pick when no preferred override (already consistent).
    # Still allow correction below when pick is outside the pool.
    weak = (not picked) or ("#" not in picked) or (bool(pool) and picked not in pool)
    if not weak:
        # In pool but not in top hits: only refine when preferred is empty
        # (legacy: LLM drifted to a sibling in-pool method vs search topic).
        if preferred:
            return plan
    # Otherwise take first search hit that is also in the evidence pool (CALLS/session)
    for h in hits:
        if preferred and not _matches_preferred(h, preferred):
            continue
        if h in pool or not pool:
            out = dict(plan)
            out["class_method"] = h
            out["view_kind"] = "method"
            out["ok"] = True
            out["reason"] = (
                (out.get("reason") or "")
                + f"（已按检索命中纠正为 `{h}`）"
            ).strip()
            out["refined"] = True
            return out
    return plan


def build_source_view_context(
    *,
    user_question: str,
    history_text: str = "",
    prior_answer: str = "",
    answer_steps: list[dict[str, Any]] | None = None,
    flow_symbols: list[str] | None = None,
    entry_cm: str | None = None,
    extra_notes: str = "",
    search_hits: list[str] | None = None,
    preferred_cms: list[str] | None = None,
) -> str:
    """Serialize context for the locator LLM."""
    steps_txt = ""
    if answer_steps:
        lines = []
        for s in answer_steps[:20]:
            lines.append(
                f"{s.get('index')}. {s.get('title') or ''}"
                + (f" — {truncate(str(s.get('detail') or ''), 80)}" if s.get("detail") else "")
            )
        steps_txt = "\n".join(lines)
    elif prior_answer:
        steps_txt = truncate(prior_answer, 2000)

    pref = [
        _simple_cm(s) for s in (preferred_cms or []) if s and "#" in str(s)
    ]
    syms = prioritize_symbols_for_question(
        user_question,
        flow_symbols,
        search_hits=search_hits,
        preferred_cms=pref,
        limit=40,
    )
    return (
        f"用户问题:\n{user_question}\n\n"
        f"对话历史(节选):\n{truncate(history_text or '(空)', 2500)}\n\n"
        f"上一答步骤/要点:\n{steps_txt or '(空)'}\n\n"
        f"锁定入口: {_simple_cm(entry_cm) or '(空)'}\n"
        f"聚焦目标(优先): "
        + (", ".join(f"`{s}`" for s in pref[:6]) if pref else "(无)")
        + "\n"
        f"检索命中(参考，勿覆盖聚焦目标):\n"
        + (
            "\n".join(f"- {s}" for s in (search_hits or [])[:12])
            if search_hits
            else "(无)"
        )
        + "\n\n候选符号(Class#method):\n"
        + ("\n".join(f"- {s}" for s in syms) if syms else "(无)")
        + (f"\n\n补充:\n{extra_notes}" if extra_notes else "")
    )


async def plan_source_view_target(
    *,
    user_question: str,
    history_text: str = "",
    prior_answer: str = "",
    answer_steps: list[dict[str, Any]] | None = None,
    flow_symbols: list[str] | None = None,
    entry_cm: str | None = None,
    extra_notes: str = "",
    search_hits: list[str] | None = None,
    preferred_cms: list[str] | None = None,
) -> dict[str, Any]:
    """LLM decides which method/file/config the user wants to see."""
    from app.llm.factory import get_llm_provider

    ctx = build_source_view_context(
        user_question=user_question,
        history_text=history_text,
        prior_answer=prior_answer,
        answer_steps=answer_steps,
        flow_symbols=flow_symbols,
        entry_cm=entry_cm,
        extra_notes=extra_notes,
        search_hits=search_hits,
        preferred_cms=preferred_cms,
    )
    try:
        llm = get_llm_provider()
        content = await llm.chat(
            [
                {"role": "system", "content": _PLAN_PROMPT},
                {"role": "user", "content": ctx},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        data = _parse_json_obj(content)
    except Exception:
        logger.debug("plan_source_view_target LLM failed", exc_info=True)
        data = {}

    kind = str(data.get("view_kind") or "unknown").strip().lower()
    if kind not in {"method", "file", "config", "db_lookup", "db_pipeline", "unknown"}:
        kind = "unknown"
    # Fetch path treats db_* as method paste.
    fetch_kind = kind
    if kind in {"db_lookup", "db_pipeline"}:
        fetch_kind = "method"
    cm = _simple_cm(str(data.get("class_method") or ""))
    if not cm and data.get("fqcn") and data.get("method_name"):
        simple = str(data["fqcn"]).split(".")[-1]
        cm = _simple_cm(f"{simple}#{data['method_name']}")
    pref = [
        _simple_cm(s) for s in (preferred_cms or []) if s and "#" in str(s)
    ]
    if pref and (not cm or not _matches_preferred(cm, pref)):
        if not cm or not _is_strong_source_target(cm, user_question, preferred=pref):
            cm = pref[0]
            if kind == "unknown":
                kind = "method"
                fetch_kind = "method"
    # Prefer candidates from evidence when LLM drifts
    allowed = {_simple_cm(s) for s in (flow_symbols or []) if s and "#" in str(s)}
    allowed.update(pref)
    if cm and allowed and cm not in allowed:
        # keep if class matches any candidate
        cls = cm.split("#", 1)[0]
        same_cls = [s for s in allowed if s.startswith(f"{cls}#")]
        if same_cls and "#" in cm:
            meth = cm.split("#", 1)[1].lower()
            fuzzy = [
                s
                for s in same_cls
                if s.split("#", 1)[1].lower() == meth
                or s.split("#", 1)[1].lower().startswith(meth)
                or meth.startswith(s.split("#", 1)[1].lower())
            ]
            cm = fuzzy[0] if fuzzy else cm
        elif cm not in allowed and kind in {"method", "db_lookup", "db_pipeline"}:
            # drop invented method; leave class-level file possible
            pass

    return {
        "view_kind": fetch_kind if kind in {"db_lookup", "db_pipeline"} else kind,
        "ask_view_kind": kind,
        "class_method": cm,
        "fqcn": str(data.get("fqcn") or "").strip(),
        "method_name": str(data.get("method_name") or "").strip(),
        "file_path": str(data.get("file_path") or "").strip().replace("\\", "/"),
        "config_key": str(data.get("config_key") or "").strip(),
        "reason": str(data.get("reason") or "").strip(),
        "ok": fetch_kind in {"method", "file", "config"}
        and bool(cm or data.get("fqcn") or data.get("file_path")),
    }


def _read_config_file(
    file_path: str, *, config_key: str = "", max_chars: int = 16000
) -> tuple[str, str | None]:
    from pathlib import Path

    from app.core.config import get_settings

    root = get_settings().project_root_path
    path = Path(file_path)
    if not path.is_file():
        path = root / file_path
    if not path.is_file():
        # basename search
        name = Path(file_path).name
        if name:
            for p in root.rglob(name):
                if "target" in p.parts or ".git" in p.parts:
                    continue
                path = p
                break
    if not path.is_file():
        return "", None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "", None
    rel = str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path)
    if config_key and config_key in text:
        # extract a window around the key
        idx = text.find(config_key)
        start = max(0, idx - 200)
        end = min(len(text), idx + 800)
        text = text[start:end]
    return truncate(text, max_chars), rel


def fetch_source_view(
    project_id: str,
    plan: dict[str, Any],
    *,
    max_chars: int = 24000,
) -> dict[str, Any]:
    """Fetch file/method/config bytes for a plan. Never invents source."""
    from app.agent.explore_shared import (
        _locate_java_file_for_class,
        _simple_class_method,
    )
    from app.agent.explore_source import _fetch_method_source_body

    kind = str(plan.get("view_kind") or "unknown")
    cm = _simple_class_method(str(plan.get("class_method") or ""))
    fqcn = str(plan.get("fqcn") or "").strip()
    file_path = str(plan.get("file_path") or "").strip().replace("\\", "/")
    config_key = str(plan.get("config_key") or "").strip()
    reason = str(plan.get("reason") or "").strip()

    if kind == "config" or (
        file_path
        and re.search(r"\.(ya?ml|xml|properties|json|conf)$", file_path, re.I)
    ):
        code, path = _read_config_file(file_path, config_key=config_key, max_chars=max_chars)
        label = path or file_path or "config"
        if code:
            header = [f"`{label}` 配置内容如下（直接读文件，未改写）："]
            if reason:
                header.append(f"定位说明：{reason}")
            if config_key:
                header.append(f"关注键：`{config_key}`")
            body = "\n".join(header) + "\n\n```\n" + code.rstrip() + "\n```"
            return {
                "ok": True,
                "answer": body,
                "target": label,
                "view_kind": "config",
                "file_path": path,
                "reason": reason,
            }
        return {
            "ok": False,
            "answer": f"已定位配置 `{label}`，但未能读取文件；不会编造。",
            "target": label,
            "view_kind": "config",
            "reason": reason,
        }

    if kind == "file" or (fqcn and not plan.get("method_name") and not (cm and "#" in cm)):
        from pathlib import Path

        simple = (fqcn or cm or "").split(".")[-1].split("#")[0]
        if not simple and file_path.endswith(".java"):
            simple = Path(file_path).stem
        if simple:
            path = file_path or _locate_java_file_for_class(project_id, simple)
            code, path2 = _fetch_method_source_body(
                project_id, simple, max_chars=max_chars
            )
            path = path or path2
            if code:
                header = [
                    f"`{simple}` 源码如下（直接读取工程文件，未改写）：",
                ]
                if path:
                    header.append(f"文件：`{path}`")
                if reason:
                    header.append(f"定位说明：{reason}")
                body = "\n".join(header) + "\n\n```java\n" + code.rstrip() + "\n```"
                return {
                    "ok": True,
                    "answer": body,
                    "target": simple,
                    "view_kind": "file",
                    "file_path": path,
                    "reason": reason,
                }

    if cm and "#" in cm:
        code, path = _fetch_method_source_body(project_id, cm, max_chars=max_chars)
        if code:
            header = [
                f"`{cm}` 方法源码如下（直接读取工程文件，未改写）：",
            ]
            if path:
                header.append(f"文件：`{path}`")
            if reason:
                header.append(f"定位说明：{reason}")
            body = "\n".join(header) + "\n\n```java\n" + code.rstrip() + "\n```"
            return {
                "ok": True,
                "answer": body,
                "target": cm,
                "view_kind": "method",
                "file_path": path,
                "reason": reason,
            }
        return {
            "ok": False,
            "answer": f"已定位 `{cm}`，但未能读取方法体；不会编造源码。",
            "target": cm,
            "view_kind": "method",
            "reason": reason,
        }

    return {
        "ok": False,
        "answer": "未能根据上下文定位到要展示的具体方法/文件/配置；请补充类名、方法名或文件路径。",
        "target": None,
        "view_kind": "unknown",
        "reason": reason,
    }


def _expand_symbol_candidates(
    project_id: str,
    flow_symbols: list[str] | None,
    entry_cm: str | None,
) -> list[str]:
    """Union session symbols with CALLS under entry (generic evidence pool)."""
    from app.agent.explore_agent import (
        _collect_entry_calls_closure,
        _simple_class_method,
    )

    out: list[str] = []
    seen: set[str] = set()

    def _push(cm: str | None) -> None:
        c = _simple_class_method(cm)
        if c and "#" in c and c not in seen:
            seen.add(c)
            out.append(c)

    for s in flow_symbols or []:
        _push(str(s))
    entry = _simple_class_method(entry_cm) if entry_cm else ""
    if entry:
        _push(entry)
        try:
            for cm in _collect_entry_calls_closure(
                project_id, entry, depth=4, limit=120
            ):
                _push(cm)
        except Exception:
            logger.debug("expand CALLS for source_view failed", exc_info=True)
    return out


async def run_source_view(
    project_id: str,
    user_question: str,
    *,
    history_text: str = "",
    prior_answer: str = "",
    answer_steps: list[dict[str, Any]] | None = None,
    flow_symbols: list[str] | None = None,
    entry_cm: str | None = None,
    extra_notes: str = "",
    preferred_cms: list[str] | None = None,
    view_kind: str | None = None,
    lookup_key: str | None = None,
    lookup_table: str | None = None,
    lookup_steps: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan with LLM then fetch. Returns {ok, answer, target, plan, ...}."""
    preferred = [
        _simple_cm(s) for s in (preferred_cms or []) if s and "#" in str(s)
    ]
    db_view = str(view_kind or "").strip().lower() in {"db_lookup", "db_pipeline"}
    # Also accept Class#method already present at the front of flow_symbols.
    for s in flow_symbols or []:
        c = _simple_cm(str(s))
        if c and "#" in c and c not in preferred:
            # Only treat leading symbols as soft prefer when explicitly passed via preferred_cms;
            # flow_symbols alone stays an evidence pool.
            break
    if db_view:
        # Free search by Goal slots — do not expand locked Controller CALLS pool.
        symbols: list[str] = []
        search_hits = search_db_lookup_hits(
            project_id,
            user_question,
            lookup_key=lookup_key,
            lookup_table=lookup_table,
            lookup_steps=lookup_steps,
            top_k=10,
        )
    else:
        symbols = _expand_symbol_candidates(project_id, flow_symbols, entry_cm)
        search_hits = search_symbol_hits_for_question(
            project_id, user_question, top_k=12
        )
    for p in preferred:
        if p not in symbols:
            symbols.insert(0, p)
    for h in search_hits:
        symbols.append(h)
    symbols = prioritize_symbols_for_question(
        user_question,
        symbols,
        search_hits=search_hits,
        preferred_cms=preferred,
        limit=60,
    )
    notes = extra_notes or ""
    db_notes = build_db_lookup_notes(
        view_kind=view_kind,
        lookup_key=lookup_key,
        lookup_table=lookup_table,
        lookup_steps=lookup_steps,
    )
    if db_notes:
        notes = f"{notes}\n{db_notes}".strip() if notes else db_notes

    plan = await plan_source_view_target(
        user_question=user_question,
        history_text=history_text,
        prior_answer=prior_answer,
        answer_steps=answer_steps,
        flow_symbols=symbols,
        entry_cm=None if db_view else entry_cm,
        extra_notes=notes,
        search_hits=search_hits,
        preferred_cms=preferred,
    )
    if db_view and plan.get("ask_view_kind") in {None, "", "unknown", "method"}:
        plan["ask_view_kind"] = view_kind
    plan = refine_source_view_plan(
        user_question,
        plan,
        symbols,
        search_hits=search_hits,
        preferred_cms=preferred,
    )
    # db_*: if refine still empty, take top search hit (slot-based, not session pin).
    if db_view and not (
        plan.get("class_method") and "#" in str(plan.get("class_method") or "")
    ):
        if search_hits:
            plan = {
                **plan,
                "view_kind": "method",
                "class_method": search_hits[0],
                "reason": plan.get("reason")
                or f"按查库槽位检索命中 `{search_hits[0]}`",
                "ok": True,
                "fallback": "db_search_top1",
            }
    fetched = fetch_source_view(project_id, plan)
    if fetched.get("ok"):
        out = {**fetched, "plan": plan}
        if db_view:
            out["view_kind"] = str(view_kind or fetched.get("view_kind") or "method")
        return out

    # Deterministic fallbacks when LLM is unsure — still no domain hardcode.
    from app.agent.explore_intent import extract_step_code_index
    from app.agent.explore_source import (
        _assemble_source_code_answer,
        _extract_explicit_class_method,
        resolve_step_code_target,
    )

    if preferred:
        for pref_cm in preferred:
            fb_plan = {
                "view_kind": "method",
                "class_method": pref_cm,
                "reason": f"沿用聚焦目标 `{pref_cm}`",
            }
            fetched_pref = fetch_source_view(project_id, fb_plan)
            if fetched_pref.get("ok"):
                return {
                    **fetched_pref,
                    "plan": {**plan, **fb_plan, "fallback": "prefer_cms"},
                }

    if db_view:
        for hit in search_hits[:5]:
            fb_plan = {
                "view_kind": "method",
                "class_method": hit,
                "reason": f"查库检索候选 `{hit}`",
            }
            fetched_db = fetch_source_view(project_id, fb_plan)
            if fetched_db.get("ok"):
                return {
                    **fetched_db,
                    "view_kind": str(view_kind or "method"),
                    "plan": {**plan, **fb_plan, "fallback": "db_search"},
                }

    explicit = _extract_explicit_class_method(user_question)
    if explicit:
        fb_plan = {
            "view_kind": "method",
            "class_method": explicit,
            "reason": "问题中点名了 Class#method",
        }
        fetched2 = fetch_source_view(project_id, fb_plan)
        if fetched2.get("ok"):
            return {**fetched2, "plan": {**plan, **fb_plan, "fallback": True}}

    if not db_view:
        step_n = extract_step_code_index(user_question)
        if step_n:
            title = ""
            for s in answer_steps or []:
                if int(s.get("index") or 0) == step_n:
                    title = str(s.get("title") or "")
                    break
            target = resolve_step_code_target(
                project_id,
                step_n=step_n,
                answer_steps=answer_steps,
                flow_symbols=symbols,
                entry_cm=entry_cm,
                step_title_hint=title,
            )
            if target:
                fb_plan = {
                    "view_kind": "method",
                    "class_method": target,
                    "reason": f"按上一答第 {step_n} 步与候选符号对齐",
                }
                fetched2 = fetch_source_view(project_id, fb_plan)
                if fetched2.get("ok"):
                    return {**fetched2, "plan": {**plan, **fb_plan, "fallback": True}}

        ans, tgt = _assemble_source_code_answer(project_id, entry_cm, user_question)
        if ans and tgt and "```" in ans:
            return {
                "ok": True,
                "answer": ans,
                "target": tgt,
                "view_kind": "method" if "#" in str(tgt) else "file",
                "plan": {**plan, "fallback": "assemble", "reason": plan.get("reason")},
            }
    return {**fetched, "plan": plan}
