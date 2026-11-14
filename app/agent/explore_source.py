"""Source-location & source-paste helpers extracted from explore_agent.py.

Answers『把源码贴出来 / XxxController 源码 / 第N步代码』by reading real Java
files or MySQL raw_code — never inventing code. Depends only on explore_keys /
explore_intent / explore_shared / question_intent / external modules at import
time; the two functions that also need explore_agent's answer-sanitizer and
gate fallbacks import them lazily inside the function body (no module-level
cycle). explore_agent re-exports every name for import compatibility.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.agent.explore_intent import (
    _api_inventory_query_core,
    _gate_themes_in_question,
    _session_gate_hint_from_question,
    _user_question_core,
    _wants_source_paste,
    extract_step_code_index,
)
from app.agent.explore_keys import (
    _CONTROLLER_HELPER_METHOD_RE,
    _GATE_DETAIL_THEME_KEYS,
    _SOURCE_PASTE_STOP_TOKENS,
)
from app.agent.explore_shared import (
    _collect_entry_calls_closure,
    _locate_java_file_for_class,
    _method_name_from_hit,
    _simple_class_method,
)
from app.agent.question_intent import wants_show_gate_code

logger = logging.getLogger("app.agent.explore_source")


def _normalize_show_code_question(question: str) -> str:
    """Fix common typos so『controoler源码』still routes to Controller file paste."""
    text = (question or "").strip()
    if not text:
        return ""
    # Chinese adjacent to Latin has no \\b — use Latin lookaround instead.
    # controoler / controler / controlller / contoller → controller
    text = re.sub(
        r"(?i)(?<![A-Za-z])cont+r[oa]+l+e*r(?![A-Za-z])",
        "controller",
        text,
    )
    text = re.sub(
        r"(?i)(?<![A-Za-z])contr[a-z]{0,8}ller(?![A-Za-z])",
        "controller",
        text,
    )
    return text


def _wants_controller_source(question: str) -> bool:
    """User asked for Controller (not ServiceImpl) source."""
    text = _normalize_show_code_question(question)
    if re.search(r"(?i)controller|控制器", text):
        return True
    # 『某接口源码』without naming a method → Controller file
    if re.search(r"接口", text) and re.search(r"源码|源代码|代码", text):
        if not _extract_source_paste_method_token(text):
            return True
    return False


def _extract_explicit_class_method(question: str) -> str | None:
    text = _user_question_core(question) or question or ""
    m = re.search(
        r"\b([A-Z][A-Za-z0-9_]{2,80})#([A-Za-z][A-Za-z0-9_]{1,80})\b",
        text,
    )
    if not m:
        return None
    return _simple_class_method(f"{m.group(1)}#{m.group(2)}")


def _extract_java_type_hint(question: str) -> str | None:
    """XxxController / XxxServiceImpl named in the question."""
    text = _user_question_core(question) or question or ""
    m = re.search(
        r"\b([A-Z][A-Za-z0-9_]*(?:Controller|ServiceImpl|Service|Listener|Job))\b",
        text,
    )
    return m.group(1) if m else None


def _extract_source_paste_method_token(question: str) -> str | None:
    """Method name user wants to see — not the word『controller』itself."""
    text = _normalize_show_code_question(_user_question_core(question) or question or "")
    # Class#method
    m = re.search(
        r"\b[A-Z][A-Za-z0-9_]{2,80}#([A-Za-z][A-Za-z0-9_]{1,80})\b",
        text,
    )
    if m:
        return m.group(1)
    # 「withdraw方法」/「withdraw 方法的源码」
    m = re.search(
        r"(?i)\b([A-Za-z][A-Za-z0-9_]{2,80})\s*方法",
        text,
    )
    if m and m.group(1).lower() not in _SOURCE_PASTE_STOP_TOKENS:
        return m.group(1)
    # Remaining camelCase tokens (skip stop words / type suffixes)
    for m in re.finditer(
        r"(?i)\b([A-Za-z][A-Za-z0-9_]{2,80})\b",
        text,
    ):
        tok = m.group(1)
        tl = tok.lower()
        if tl in _SOURCE_PASTE_STOP_TOKENS:
            continue
        if tok[0].isupper() and any(
            tok.endswith(s)
            for s in ("Controller", "Service", "ServiceImpl", "Listener", "Job", "Mapper")
        ):
            continue
        return tok
    return None


def _method_name_fuzzy_score(candidate: str, asked: str) -> int:
    """Score how well candidate method matches user token (withdraw≈withdrawals)."""
    c = (candidate or "").split("(", 1)[0].strip()
    a = (asked or "").split("(", 1)[0].strip()
    if not c or not a:
        return 0
    cl, al = c.lower(), a.lower()
    if cl == al:
        return 100
    if cl.startswith(al) or al.startswith(cl):
        return 80
    if al in cl or cl in al:
        return 50
    return 0


def _pick_best_method_match(
    candidates: list[str], asked_method: str, *, prefer_controller: bool
) -> str | None:
    """Pick Class#method; reject Controller helpers; prefer HTTP facade names."""
    asked = (asked_method or "").strip()
    if not asked:
        return None
    scored: list[tuple[int, str]] = []
    for raw in candidates:
        cm = _simple_class_method(raw)
        if not cm or "#" not in cm:
            continue
        cls, meth = cm.split("#", 1)
        meth = meth.split("(", 1)[0]
        if prefer_controller and "Controller" not in cls and "Resource" not in cls:
            continue
        if _CONTROLLER_HELPER_METHOD_RE.match(meth):
            continue
        sc = _method_name_fuzzy_score(meth, asked)
        if sc <= 0:
            continue
        # Prefer shorter class names that look like facades; boost exact class Controllers
        if "Controller" in cls:
            sc += 5
        scored.append((sc, cm))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(x[1]), x[1]))
    return scored[0][1]


def _expand_controller_methods_mysql(
    project_id: str, class_simple: str
) -> list[str]:
    """List Class#method for a Controller from MySQL symbol rows."""
    simple = (class_simple or "").split(".")[-1].strip()
    if not project_id or not simple:
        return []
    try:
        from sqlalchemy import or_, select

        from app.db.mysql import get_session
        from app.models.orm import SymbolORM

        out: list[str] = []
        with get_session() as session:
            rows = list(
                session.scalars(
                    select(SymbolORM)
                    .where(
                        SymbolORM.project_id == project_id,
                        or_(
                            SymbolORM.simple_name == simple,
                            SymbolORM.fqcn.like(f"%.{simple}"),
                        ),
                        SymbolORM.method_name.is_not(None),
                        SymbolORM.method_name != "",
                    )
                    .limit(80)
                )
            )
        for r in rows:
            meth = str(r.method_name or "").split("(", 1)[0].strip()
            if meth:
                out.append(f"{simple}#{meth}")
        return out
    except Exception:
        logger.debug("expand controller methods mysql failed", exc_info=True)
        return []


def _search_controller_methods_for_paste(
    project_id: str, question: str, *, top_k: int = 24
) -> list[str]:
    """Collect Controller Class#method candidates from entry_route + symbol search."""
    q_core = _api_inventory_query_core(question)
    # Keep method token / type name for search signal
    meth = _extract_source_paste_method_token(question) or ""
    type_hint = _extract_java_type_hint(question) or ""
    q = " ".join(x for x in (q_core, type_hint, meth, "Controller") if x).strip()
    out: list[str] = []
    seen: set[str] = set()
    class_hits: list[str] = []

    def _push(cm: str) -> None:
        c = _simple_class_method(cm)
        if not c or c in seen:
            return
        if "#" not in c:
            # class-only → expand later
            simple = c.split(".")[-1]
            if simple and simple not in class_hits:
                class_hits.append(simple)
            return
        seen.add(c)
        out.append(c)

    try:
        from app.analysis.entry_route import search_entry_routes

        for h in search_entry_routes(q or question, project_id, top_k=top_k) or []:
            _push(str(h.get("class_method") or ""))
    except Exception:
        logger.debug("source-paste entry_route failed", exc_info=True)
    try:
        from app.agent.explore_tools import search_symbols

        for kind in ("Controller", None):
            sr = search_symbols(
                project_id,
                q or meth or question,
                kind=kind,
                top_k=top_k,
            )
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "")
                simple = fqcn.split(".")[-1] if fqcn else ""
                mname = _method_name_from_hit(h)
                if simple and mname:
                    _push(f"{simple}#{mname}")
                elif simple:
                    _push(simple)
            if out or class_hits:
                break
    except Exception:
        logger.debug("source-paste search_symbols failed", exc_info=True)

    if type_hint:
        class_hits.insert(0, type_hint)
    # Dedupe class hits preserving order
    seen_cls: set[str] = set()
    uniq_cls: list[str] = []
    for c in class_hits:
        if c not in seen_cls:
            seen_cls.add(c)
            uniq_cls.append(c)
    for cls in uniq_cls[:6]:
        if "Controller" not in cls and "Resource" not in cls:
            continue
        for cm in _expand_controller_methods_mysql(project_id, cls):
            _push(cm)
    return out


def _resolve_source_paste_target(
    project_id: str,
    entry_cm: str | None,
    question: str,
) -> str | None:
    """Pick Class#method whose body should be pasted for a 源码 ask.

    Order: explicit Class#method → session symbols → Controller/method search
    (fuzzy withdraw→withdrawals) → CALLS under locked entry. Never invent.
    """
    q = _normalize_show_code_question(question)
    entry = _simple_class_method(entry_cm) if entry_cm else ""
    prefer_controller = _wants_controller_source(q)

    explicit = _extract_explicit_class_method(q)
    if explicit:
        return explicit

    # Step-code follow-ups must not paste inj_symbols[0] (usually the entry).
    if extract_step_code_index(q) or extract_step_code_index(question or ""):
        return None

    inj_gate, inj_symbols = _session_gate_hint_from_question(q)
    if inj_symbols:
        return inj_symbols[0]

    meth = _extract_source_paste_method_token(q)
    type_hint = _extract_java_type_hint(q)

    if meth or prefer_controller or type_hint:
        candidates = _search_controller_methods_for_paste(project_id, q)
        if type_hint:
            candidates = [
                c
                for c in candidates
                if c.split("#", 1)[0] == type_hint
                or type_hint.lower() in c.lower()
            ] or candidates
        if meth:
            picked = _pick_best_method_match(
                candidates, meth, prefer_controller=prefer_controller or bool(type_hint)
            )
            if picked:
                return picked
            # Also try methods on locked entry's Controller class
            if entry and "#" in entry:
                cls = entry.split("#", 1)[0]
                same_cls = [c for c in candidates if c.startswith(f"{cls}#")]
                if not same_cls:
                    same_cls = [entry] + candidates
                picked = _pick_best_method_match(
                    same_cls, meth, prefer_controller=False
                )
                if picked:
                    return picked
        elif prefer_controller and candidates:
            # Whole-controller ask without method: prefer HTTP facade (non-helper)
            for c in candidates:
                mname = c.split("#", 1)[-1]
                if not _CONTROLLER_HELPER_METHOD_RE.match(mname):
                    return c
            return candidates[0]

    # Locked entry + method token on CALLS / same class (ServiceImpl dig-in)
    if meth and entry:
        if prefer_controller:
            cls = entry.split("#", 1)[0]
            if "Controller" in cls:
                # Prefer entry itself when fuzzy-matches
                if _method_name_fuzzy_score(entry.split("#", 1)[-1], meth) > 0:
                    return entry
        else:
            impl_hint = _service_impl_hint_from_entry(entry)
            if impl_hint:
                # Only return if we can later fetch code; still prefer search hit
                guessed = f"{impl_hint}#{meth}"
                return guessed
        closure = _collect_entry_calls_closure(
            project_id, entry, depth=4, limit=160
        )
        picked = _pick_best_method_match(
            [entry] + closure, meth, prefer_controller=prefer_controller
        )
        if picked:
            return picked

    themes = _gate_themes_in_question(q)
    if inj_gate and not themes:
        for key in _GATE_DETAIL_THEME_KEYS:
            if key in inj_gate:
                themes = [key]
                break
    if not themes:
        return None
    lock = entry or ""
    if not lock:
        return None
    closure = _collect_entry_calls_closure(project_id, lock, depth=4, limit=160)
    if not closure:
        # gate fallbacks live in explore_agent (gate group) — lazy import to
        # avoid a module-level cycle until that group is extracted too.
        from app.agent.explore_agent import (
            _gate_methods_from_mysql,
            _rank_gate_callees,
        )

        closure = _gate_methods_from_mysql(project_id, themes)
    ranked = _rank_gate_callees(closure, themes, question=q)
    if not ranked:
        return None
    return _simple_class_method(ranked[0])


def _pick_controller_class_for_file_paste(
    project_id: str, question: str
) -> str | None:
    """Pick one Controller simple name for whole-file paste (topic → search)."""
    q = _normalize_show_code_question(question)
    type_hint = _extract_java_type_hint(q)
    if type_hint and "Controller" in type_hint:
        return type_hint

    q_core = _api_inventory_query_core(q) or q
    # Top hybrid hit is the strongest topic signal from the question.
    search_rank: dict[str, int] = {}
    try:
        from app.agent.explore_tools import search_symbols

        sr = search_symbols(project_id, q_core, kind="Controller", top_k=12)
        for i, h in enumerate(sr.get("hits") or []):
            fqcn = str((h or {}).get("fqcn") or "")
            simple = fqcn.split(".")[-1]
            if not simple.endswith("Controller"):
                continue
            # earlier hit → higher boost; keep best (lowest i)
            search_rank[simple] = min(search_rank.get(simple, 99), i)
    except Exception:
        logger.debug("controller file-paste search failed", exc_info=True)

    cms = _search_controller_methods_for_paste(project_id, q)
    if not cms and search_rank:
        cms = [f"{c}#_" for c in search_rank]
    if not cms:
        return None

    by_cls: dict[str, list[str]] = {}
    for cm in cms:
        if "#" not in cm:
            continue
        cls, meth = cm.split("#", 1)
        by_cls.setdefault(cls, []).append(meth.split("(", 1)[0])
    for cls in search_rank:
        by_cls.setdefault(cls, [])

    scored: list[tuple[int, int, str]] = []
    for cls, methods in by_cls.items():
        stem = re.sub(r"Controller$", "", cls, flags=re.I)
        sc = 0
        for m in methods:
            if m in {"_", ""}:
                continue
            ml = m.lower()
            if stem and (ml == stem.lower() or ml.startswith(stem.lower())):
                sc += 50
            if not _CONTROLLER_HELPER_METHOD_RE.match(m):
                sc += 1
        # Hybrid rank: 1st hit +100, 2nd +90, …
        if cls in search_rank:
            sc += max(0, 100 - search_rank[cls] * 10)
        scored.append((sc, search_rank.get(cls, 99), cls))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scored[0][2]


def _fetch_method_source_body(
    project_id: str,
    class_method: str,
    *,
    max_chars: int = 14000,
) -> tuple[str, str | None]:
    """Return (raw_code, file_path_hint). Prefer disk file read, else MySQL raw_code."""
    cm = _simple_class_method(class_method)
    if not cm:
        return "", None
    if "#" in cm:
        simple, meth = cm.split("#", 1)
        simple = simple.split(".")[-1]
        meth = meth.split("(", 1)[0]
    else:
        simple, meth = cm.split(".")[-1], None

    # 1) File-first: locate path → read method or whole class
    path = _locate_java_file_for_class(project_id, simple)
    if path:
        try:
            from app.agent.explore_tools import _read_source_method

            src = _read_source_method(
                file_path=path, method_name=meth, max_chars=max_chars
            )
            if src and src.get("ok"):
                body = str(src.get("raw_code") or "").strip()
                if body and "无 raw_code" not in body:
                    return body, path
        except Exception:
            logger.debug("disk read source failed", exc_info=True)

    # 2) MySQL / get_symbol_code fallback (method only)
    if meth:
        try:
            from app.agent.explore_tools import get_symbol_code

            got = get_symbol_code(
                project_id,
                fqcn=simple,
                method_name=meth,
                max_chars=max_chars,
            )
        except Exception:
            got = {"ok": False}
        code = str((got or {}).get("raw_code") or "").strip()
        sym = (got or {}).get("symbol") if isinstance(got, dict) else None
        if isinstance(sym, dict) and not path:
            path = sym.get("file_path")
        if (
            (got or {}).get("ok")
            and code
            and "无 raw_code" not in code
        ):
            return code, path
    return "", path


def _parse_step_hint_from_question(question: str) -> tuple[int | None, str, list[str]]:
    """Read graph injection （步骤序号：6；步骤标题：…；候选符号：…）."""
    raw = question or ""
    step_n = extract_step_code_index(raw)
    title = ""
    m = re.search(r"步骤标题[：:]\s*([^；;）)]+)", raw)
    if m:
        title = m.group(1).strip()
    symbols: list[str] = []
    m = re.search(r"候选符号[：:]\s*([^）)]+)[）)]", raw)
    if m:
        for part in re.split(r"[,，、\s]+", m.group(1)):
            p = part.strip()
            if p and "#" in p:
                symbols.append(_simple_class_method(p))
    return step_n, title, symbols


def _service_impl_hint_from_entry(entry_cm: str | None) -> str | None:
    """Derive *ServiceImpl simple name from Controller#method entry when possible."""
    cm = _simple_class_method(entry_cm) if entry_cm else ""
    if not cm or "#" not in cm:
        return None
    simple = cm.split("#", 1)[0]
    if simple.endswith("Controller") and len(simple) > len("Controller"):
        return f"{simple[: -len('Controller')]}ServiceImpl"
    if simple.endswith("ServiceImpl"):
        return simple
    if simple.endswith("Service") and not simple.endswith("ServiceImpl"):
        return f"{simple}Impl"
    return None


def _score_symbol_for_step_title(
    cm: str,
    title: str,
    detail: str = "",
    *,
    search_hits: list[str] | None = None,
) -> int:
    """Rank by hybrid-search hit position + Latin token overlap (no domain map)."""
    blob = f"{title} {detail}".strip()
    if not cm or "#" not in cm:
        return 0
    cls, meth = cm.split("#", 1)
    meth = meth.split("(", 1)[0]
    sc = 0
    simple = _simple_class_method(cm)
    for i, h in enumerate(search_hits or []):
        if _simple_class_method(h) == simple:
            sc += max(0, 100 - i * 5)
            break
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", blob):
        tl = tok.lower()
        if tl == meth.lower():
            sc += 40
        elif tl in meth.lower() or meth.lower() in tl:
            sc += 20
        elif tl in cm.lower():
            sc += 8
    if "ServiceImpl" in cls:
        sc += 2
    if _CONTROLLER_HELPER_METHOD_RE.match(meth):
        sc -= 15
    return sc


def resolve_step_code_target(
    project_id: str,
    *,
    step_n: int,
    answer_steps: list[dict[str, Any]] | None,
    flow_symbols: list[str] | None,
    entry_cm: str | None,
    step_title_hint: str = "",
) -> str | None:
    """Map prior-answer step N → Class#method to paste (never invent)."""
    title = (step_title_hint or "").strip()
    detail = ""
    if answer_steps:
        for s in answer_steps:
            if int(s.get("index") or 0) == step_n:
                title = title or str(s.get("title") or "").strip()
                detail = str(s.get("detail") or "").strip()
                if s.get("class_method"):
                    return _simple_class_method(str(s["class_method"]))
                break
        # Fallback: 0-based list order if indexes missing
        if not title and 1 <= step_n <= len(answer_steps):
            s = answer_steps[step_n - 1]
            title = str(s.get("title") or "").strip()
            detail = str(s.get("detail") or "").strip()

    candidates: list[str] = []
    for cm in flow_symbols or []:
        c = _simple_class_method(str(cm))
        if c and "#" in c:
            candidates.append(c)
    entry = _simple_class_method(entry_cm) if entry_cm else ""
    if entry and "#" in entry:
        candidates.append(entry)
        for cm in _collect_entry_calls_closure(
            project_id, entry, depth=4, limit=160
        ):
            candidates.append(_simple_class_method(cm))

    # Dedup preserve order
    seen: set[str] = set()
    uniq: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)

    search_hits: list[str] = []
    q = f"{title} {detail}".strip()
    if q and project_id:
        try:
            from app.agent.explore_tools import search_symbols

            sr = search_symbols(project_id, q, top_k=12)
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "")
                mname = _method_name_from_hit(h)
                simple = fqcn.split(".")[-1]
                if simple and mname:
                    search_hits.append(f"{simple}#{mname}")
            search_hits = list(dict.fromkeys(search_hits))
        except Exception:
            logger.debug("step-code search_symbols failed", exc_info=True)

    pool = set(uniq)
    # Prefer search hits that also appear in CALLS/session evidence
    for h in search_hits:
        hm = _simple_class_method(h)
        if hm in pool:
            return hm
    if search_hits and not pool:
        return _simple_class_method(search_hits[0])

    if title or detail or search_hits:
        scored = [
            (
                _score_symbol_for_step_title(
                    c, title, detail, search_hits=search_hits
                ),
                c,
            )
            for c in uniq
        ]
        scored = [x for x in scored if x[0] > 0]
        if scored:
            scored.sort(
                key=lambda x: (-x[0], 0 if "ServiceImpl#" in x[1] else 1, x[1])
            )
            return scored[0][1]
        # Hit outside pool but title search found something usable
        for h in search_hits:
            return _simple_class_method(h)

    # Weak fallback: N-th flow symbol (skip entry-only if N>1)
    if flow_symbols and 1 <= step_n <= len(flow_symbols):
        return _simple_class_method(str(flow_symbols[step_n - 1]))
    return None


def _assemble_source_code_answer(
    project_id: str,
    entry_cm: str | None,
    question: str,
) -> tuple[str | None, str | None]:
    """Paste real source from disk/MySQL — never invent code.

    - Method ask → Class#method body
    - Controller / 接口源码 ask → whole Java file once class is located
    - 『第N步/N中对应代码』→ map prior step → that method (not entry dump)
    Returns (answer_markdown, target_label).
    """
    q = _normalize_show_code_question(question)
    if not project_id or not (
        _wants_source_paste(q) or wants_show_gate_code(q) or _wants_source_paste(question)
    ):
        return None, None

    step_n, step_title, step_symbols = _parse_step_hint_from_question(question)
    if step_n is None:
        step_n = extract_step_code_index(q)
    if step_n:
        entry = _simple_class_method(entry_cm) if entry_cm else ""
        m_entry = re.search(r"入口[：:]\s*([^；;）)]+)", question or "")
        if m_entry and "#" in m_entry.group(1):
            entry = _simple_class_method(m_entry.group(1).strip()) or entry
        target = resolve_step_code_target(
            project_id,
            step_n=step_n,
            answer_steps=None,
            flow_symbols=step_symbols or None,
            entry_cm=entry,
            step_title_hint=step_title,
        )
        if target and "#" in target:
            code, path = _fetch_method_source_body(project_id, target)
            if code:
                title_bit = f"「{step_title}」" if step_title else ""
                header = [
                    f"上一答第 {step_n} 步{title_bit}对应 `{target}` "
                    f"方法源码如下（直接读文件，未改写）：",
                ]
                if path:
                    header.append(f"文件：`{path}`")
                body = "\n".join(header) + "\n\n```java\n" + code.rstrip() + "\n```"
                # answer sanitizer lives in explore_agent (answers group) — lazy import.
                from app.agent.explore_agent import _sanitize_answer

                return (_sanitize_answer(body, evidence_text=code) or body), target
            return (
                f"已将第 {step_n} 步定位到 `{target}`，但未能读取方法体；不会编造源码。"
            ), target
        title_bit = f"「{step_title}」" if step_title else ""
        return (
            f"未能把上一答第 {step_n} 步{title_bit}对应到具体 Class#method；"
            "请点名方法名或再说一下该步在做什么。"
        ), None

    meth = _extract_source_paste_method_token(q)
    prefer_controller = _wants_controller_source(q)

    # Whole Controller file: locate class → read file (no LLM rewrite).
    if prefer_controller and not meth:
        cls = _pick_controller_class_for_file_paste(project_id, q)
        if not cls and entry_cm and "#" in (entry_cm or ""):
            cls = _simple_class_method(entry_cm).split("#", 1)[0]
        if cls:
            path = _locate_java_file_for_class(project_id, cls)
            code, path2 = _fetch_method_source_body(
                project_id, cls, max_chars=24000
            )
            path = path or path2
            if code:
                header = [
                    f"`{cls}` 源码如下（直接读取工程 Java 文件，未改写）：",
                ]
                if path:
                    header.append(f"文件：`{path}`")
                body = "\n".join(header) + "\n\n```java\n" + code.rstrip() + "\n```"
                from app.agent.explore_agent import _sanitize_answer

                return (_sanitize_answer(body, evidence_text=code) or body), cls
            return (
                f"已定位 Controller `{cls}`，但未能读取对应 Java 文件"
                + (f"（路径提示：{path}）" if path else "")
                + "；不会编造源码。"
            ), cls

    target = _resolve_source_paste_target(project_id, entry_cm, q)
    if not target:
        # Last resort: still try whole controller file for 源码 asks
        if prefer_controller or re.search(r"源码|源代码|代码", q):
            cls = _pick_controller_class_for_file_paste(project_id, q)
            if cls:
                code, path = _fetch_method_source_body(
                    project_id, cls, max_chars=24000
                )
                if code:
                    header = [
                        f"`{cls}` 源码如下（直接读取工程 Java 文件，未改写）：",
                    ]
                    if path:
                        header.append(f"文件：`{path}`")
                    body = (
                        "\n".join(header) + "\n\n```java\n" + code.rstrip() + "\n```"
                    )
                    from app.agent.explore_agent import _sanitize_answer

                    return (_sanitize_answer(body, evidence_text=code) or body), cls
        return (
            "未在已读材料中定位到要展示的具体文件/方法；"
            "请点名 Class#method 或 Controller 类名后再问源码。"
        ), None
    code, path = _fetch_method_source_body(project_id, target)
    label = _simple_class_method(target)
    if not code:
        return (
            f"已定位 `{label}`，但未能读取对应 Java 文件或方法体"
            + (f"（路径提示：{path}）" if path else "")
            + "；不会编造源码。"
        ), label
    kind = "方法" if "#" in label else "类"
    header_bits = [
        f"`{label}` {kind}源码如下（直接读取工程文件或入库 raw_code，未改写）："
    ]
    if path:
        header_bits.append(f"文件：`{path}`")
    body = "\n".join(header_bits) + "\n\n```java\n" + code.rstrip() + "\n```"
    from app.agent.explore_agent import _sanitize_answer

    return (_sanitize_answer(body, evidence_text=code) or body), label
