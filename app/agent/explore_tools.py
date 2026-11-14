"""Read-only tools for layered code exploration (search → read → follow calls)."""
from __future__ import annotations

import re
from typing import Any

from sqlalchemy import or_, select

from app.agent.runtime.registry import EXPLORE_TOOL_SPECS as TOOL_SPECS
from app.core.utils import truncate
from app.db.mysql import get_session
from app.db.neo4j_client import get_neo4j
from app.models.orm import SymbolDescORM, SymbolORM
from app.tagging.search import extract_query_literals, hybrid_search

_JAVA_CALL_SKIP = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "new",
    "catch",
    "throw",
    "super",
    "this",
    "log",
    "logger",
    "equals",
    "hashCode",
    "toString",
    "valueOf",
    "format",
    "println",
    "print",
}
_HELPER_NAME = re.compile(
    r"(?i)^(check|valid|verify|generate|fill|build|parse|assem|convert|prepare|init|bind)",
)


def extract_same_class_calls(raw_code: str, *, current_method: str | None = None) -> list[str]:
    """Heuristic: same-class helper calls often hold param validation (private methods)."""
    if not raw_code:
        return []
    found: list[str] = []
    for m in re.finditer(r"\bthis\.([A-Za-z_]\w*)\s*\(", raw_code):
        name = m.group(1)
        if name != current_method and name not in found:
            found.append(name)
    for m in re.finditer(r"(?<![\w.])([a-z][A-Za-z0-9_]*)\s*\(", raw_code):
        name = m.group(1)
        if name in _JAVA_CALL_SKIP or name == current_method:
            continue
        if _HELPER_NAME.match(name) and name not in found:
            found.append(name)
    return found[:12]


def _explore_source_only() -> bool:
    from app.core.config import get_settings

    return (get_settings().explore_backend or "source").strip().lower() == "source"


def tool_catalog_text() -> str:
    source = _explore_source_only()
    lines = [
        "可用工具（每轮可并行 1～3 个只读工具；finish 单独一轮且只能一个）："
    ]
    for t in TOOL_SPECS:
        lines.append(f"- {t['name']}: {t['desc']} args={t['args']}")
    if source:
        lines.append(
            "当前模式=纯源码探索：search/get_symbol_code/list_* 均直接扫项目 Java 文件，"
            "无离线 FlowDoc/向量库；必须先 search 再读码，禁止编造未读到的类。"
        )
    lines.append(
        "输出格式（严格 JSON，不要夹杂解释）：\n"
        '1) 单工具: {"tool":"search_symbols","args":{"q":"从用户原话提取的实词","kind":"Controller"}}\n'
        '2) 多工具: {"tools":[{"tool":"get_symbol_code","args":{"symbol_id":"..."}},'
        '{"tool":"list_callees","args":{"symbol_id":"..."}}]}\n'
        '3) 结束: {"tool":"finish","args":{"answer":"..."}}'
    )
    return "\n".join(lines)


def _sym_brief(row: SymbolORM, summary: str | None = None) -> dict[str, Any]:
    return {
        "symbol_id": row.id,
        "kind": row.kind,
        "fqcn": row.fqcn,
        "method_name": row.method_name,
        "method_sig": truncate(row.method_sig or "", 180),
        "file_path": row.file_path,
        "summary": truncate(summary or "", 200) if summary else None,
    }


def _search_java_sources(
    project_id: str,
    q: str,
    *,
    literals: list[str] | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Scan project Java files when MySQL/Milvus index lacks the entry (pilot-only ingest)."""
    from pathlib import Path

    from app.core.config import get_settings

    settings = get_settings()
    root = settings.project_root_path
    if not root.is_dir():
        return []
    terms = [t for t in (literals or extract_query_literals(q) or []) if len(t) >= 2]
    path_tokens: list[str] = []
    try:
        from app.tagging.search import extract_api_path_tokens

        path_tokens = extract_api_path_tokens(q)
    except Exception:
        path_tokens = []
    for t in reversed(path_tokens):
        if t not in terms:
            terms.insert(0, t)
    # EN stems from question Latin tokens only — no product CN→EN glossary.
    q_text = q or ""
    # Dedup preserve order
    seen_t: set[str] = set()
    uniq_terms: list[str] = []
    for t in terms:
        if t not in seen_t:
            seen_t.add(t)
            uniq_terms.append(t)
    if not uniq_terms:
        return []

    hits: list[dict[str, Any]] = []
    # Prefer Controllers / files whose name matches.
    candidates: list[Path] = []
    try:
        # Controllers first (name match), then other java name matches.
        for p in root.rglob("*Controller.java"):
            if "target" in p.parts or ".git" in p.parts:
                continue
            name = p.name
            if any(t.lower() in name.lower() for t in uniq_terms):
                candidates.append(p)
        for p in root.rglob("*.java"):
            if "target" in p.parts or ".git" in p.parts or p in candidates:
                continue
            name = p.name
            if "Controller" in name:
                continue  # already considered
            if any(t.lower() in name.lower() for t in uniq_terms):
                candidates.append(p)
        if len([c for c in candidates if "Controller" in c.name]) < 2:
            # Light content scan on Controller-ish files only.
            for p in root.rglob("*Controller.java"):
                if "target" in p.parts or p in candidates:
                    continue
                try:
                    text = p.read_text(encoding="utf-8", errors="ignore")[:12000]
                except Exception:
                    continue
                if any(t in text for t in uniq_terms):
                    candidates.insert(0, p)
                if len(candidates) >= top_k * 4:
                    break
    except Exception:
        return []

    path_l = {t.lower() for t in path_tokens}

    def _cand_rank(p: Path) -> tuple:
        name = p.name.lower()
        controller = 0 if "controller" in name else 1
        strong = 0
        if path_l and any(t in name for t in path_l):
            strong = -3
        return (controller, strong, name)

    candidates = sorted(set(candidates), key=_cand_rank)

    for p in candidates[: top_k * 2]:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        pkg_m = re.search(r"package\s+([\w.]+)\s*;", text)
        cls_m = re.search(r"\b(?:public\s+)?(?:class|interface)\s+(\w+)", text)
        pkg = pkg_m.group(1) if pkg_m else ""
        cls = cls_m.group(1) if cls_m else p.stem
        fqcn = f"{pkg}.{cls}" if pkg else cls
        # Prefer mapping methods that match path tokens / query terms.
        method = ""
        mapping_hit = ""
        best_score = -1
        for m in re.finditer(
            r"@(?:Request|Post|Get|Put|Delete|Patch)Mapping\b([^;]{0,260}?)"
            r"(?:public|protected)\s+[\w.<>,\s\[\]]+\s+(\w+)\s*\(",
            text,
            re.S,
        ):
            ann = m.group(1) or ""
            name = m.group(2)
            window = text[max(0, m.start() - 200) : m.end() + 80]
            path_vals = [
                a or b
                for a, b in re.findall(
                    r'(?:value|path)\s*=\s*"([^"]+)"|@"([^"]+)"',
                    ann,
                )
            ]
            score = 0
            for tok in path_tokens:
                tl = tok.lower()
                weak_tok = tl.isalpha() and tl.islower() and len(tl) <= 6
                base = 35 if weak_tok else 100
                if any(tl == pv.strip("/").split("/")[-1].lower() for pv in path_vals):
                    score = max(score, base)
                elif tl == name.lower():
                    score = max(score, max(base - 5, 80) if not weak_tok else 55)
                elif tl in name.lower() or any(tl in pv.lower() for pv in path_vals):
                    score = max(score, max(base - 30, 40))
            # Prefer the most specific path token (usually the last segment).
            if path_tokens:
                last = path_tokens[-1].lower()
                last_weak = last.isalpha() and last.islower() and len(last) <= 6
                if not last_weak:
                    if any(last == pv.strip("/").split("/")[-1].lower() for pv in path_vals) or last == name.lower():
                        score = max(score, 110)
            if score == 0 and any(
                t in window or t.lower() in name.lower() for t in uniq_terms
            ):
                score = 40
            if score > best_score:
                best_score = score
                method = name
                mapping_hit = path_vals[0] if path_vals else name
        api_m = re.search(r'@ApiOperation\s*\(\s*value\s*=\s*"([^"]+)"', text)
        summary = api_m.group(1) if api_m else f"源码文件命中: {rel}"
        if mapping_hit:
            summary = f"{summary} [{mapping_hit}]"
        hit_score = 3.0 + (
            1.5 if best_score >= 90 else (0.8 if best_score >= 70 else 0.0)
        )
        hits.append(
            {
                "symbol_id": f"source:{rel}#{method}" if method else f"source:{rel}",
                "kind": "Controller" if "Controller" in cls else "Class",
                "fqcn": fqcn,
                "method_name": method or None,
                "method_sig": f"{cls}#{method}" if method else cls,
                "file_path": rel,
                "summary": truncate(summary, 200),
                "tags": ["source_scan"],
                "score": hit_score,
                "path_match_score": best_score,
            }
        )
        if len(hits) >= top_k:
            break
    hits.sort(
        key=lambda h: (
            -float(h.get("path_match_score") or 0),
            -float(h.get("score") or 0),
        )
    )
    return hits[:top_k]


def search_symbols(
    project_id: str,
    q: str,
    *,
    kind: str | None = None,
    top_k: int = 8,
) -> dict[str, Any]:
    literals = extract_query_literals(q)
    if _explore_source_only():
        src_hits = _search_java_sources(project_id, q, literals=literals, top_k=max(top_k, 12))
        if kind:
            kind_l = kind.lower()
            filtered = [
                h
                for h in src_hits
                if (h.get("kind") or "").lower() == kind_l
                or kind_l in (h.get("fqcn") or "").lower()
                or kind_l in (h.get("file_path") or "").lower()
            ]
            src_hits = filtered or src_hits
        return {
            "ok": True,
            "count": len(src_hits),
            "query_literals": literals,
            "backend": "source",
            "note": (
                "源码扫描命中；请用返回的 symbol_id/fqcn+method_name 调用 get_symbol_code 读码。"
            ),
            "hits": src_hits[:top_k],
        }

    hits = hybrid_search(q, project_id, top_k=max(top_k, 12))
    if kind:
        kind_l = kind.lower()
        # FlowDoc / Overview are always useful for process questions — keep them
        filtered = [
            h
            for h in hits
            if (h.kind or "").lower() == kind_l
            or (h.kind or "") in {"FlowDoc", "Overview"}
            or (h.symbol_id or "").startswith(("flow:", "overview:"))
        ]
        # prefer kind match but fall back
        hits = filtered or hits
    # If user didn't ask about 通知/回调, demote Notice/Callback/Notify controllers
    # so "完整流程" hits the apply/submit entry instead of post-hoc notice APIs.
    q_l = (q or "").lower()
    want_notice = any(k in (q or "") for k in ("通知", "回调", "listener", "监听")) or (
        "notice" in q_l or "callback" in q_l or "notify" in q_l
    )
    want_flow = "流程" in (q or "") or any(
        k in (q or "") for k in ("主链", "全链路", "端到端", "从入口")
    )

    def _method_name(h: Any) -> str:
        sig = h.method_sig or ""
        if "#" in sig:
            return sig.split("#", 1)[1].split("(", 1)[0].strip()
        if "(" in sig:
            parts = sig.split("(", 1)[0].strip().split()
            return parts[-1] if parts else ""
        return ""

    def _controller_stem(fqcn: str) -> str:
        simple = (fqcn or "").split(".")[-1]
        return re.sub(
            r"(Controller|RestController|ServiceImpl|Service|Listener|Job)$",
            "",
            simple,
            flags=re.I,
        ).lower()

    def _rank(h: Any) -> tuple:
        blob = f"{h.fqcn or ''} {h.method_sig or ''} {h.file_path or ''}".lower()
        notice_penalty = 0
        if want_notice:
            # Prefer Notice/Notification controllers; demote unrelated FlowDoc.
            if any(
                x in blob
                for x in ("notice", "notify", "notification")
            ):
                notice_penalty = -5
            elif (h.kind or "") in {"FlowDoc", "Overview"} or (h.symbol_id or "").startswith(
                ("flow:", "overview:")
            ):
                notice_penalty = 8
        elif any(
            x in blob for x in ("notice", "callback", "notify", "listenner", "listener")
        ):
            # Do not demote explicit FlowDoc callback sections
            if (h.kind or "") not in {"FlowDoc", "Overview"}:
                notice_penalty = 5
        flow_boost = 0
        if (h.kind or "") == "FlowDoc" or (h.symbol_id or "").startswith("flow:"):
            flow_boost = 6 if want_notice else -3
        elif (h.kind or "") == "Overview" or (h.symbol_id or "").startswith("overview:"):
            flow_boost = 4 if want_notice else -1
        method = _method_name(h)
        queryish_penalty = 0
        stem_boost = 0
        if want_flow and not want_notice:
            if re.search(
                r"(?i)^(query|get|list|find|select|search|count|check|load|fetch|page)",
                method or "",
            ):
                queryish_penalty = 4
            stem = _controller_stem(h.fqcn or "")
            if stem and method and stem in method.lower():
                stem_boost = -2  # lower tuple = better
        return (
            notice_penalty,
            queryish_penalty,
            stem_boost,
            flow_boost,
            -(h.score or 0),
            0 if (h.kind or "") == "Controller" else 1,
            0 if (h.method_sig or "").strip() else 2,
        )

    hits = sorted(hits, key=_rank)
    out = []
    for h in hits[:top_k]:
        out.append(
            {
                "symbol_id": h.symbol_id,
                "kind": h.kind,
                "fqcn": h.fqcn,
                "method_sig": truncate(h.method_sig or "", 180),
                "file_path": h.file_path,
                "summary": truncate(
                    h.summary or "",
                    800 if (h.kind or "") in {"FlowDoc", "Overview"} else 200,
                ),
                "tags": h.tags[:8],
                "score": h.score,
            }
        )
    if not out:
        # SQL fallback
        with get_session() as session:
            like = f"%{q}%"
            stmt = select(SymbolORM).where(
                SymbolORM.project_id == project_id,
                or_(
                    SymbolORM.fqcn.like(like),
                    SymbolORM.simple_name.like(like),
                    SymbolORM.method_name.like(like),
                ),
            )
            if kind:
                stmt = stmt.where(SymbolORM.kind == kind)
            rows = list(session.scalars(stmt.limit(top_k)))
            out = [_sym_brief(r) for r in rows]
    # Index may miss Controllers (pilot-only ingest) — scan Java sources on disk.
    if len(out) < max(2, top_k // 2) or (
        want_notice
        and not any(
            re.search(r"(?i)notice|notify|notification", str(x.get("fqcn") or ""))
            for x in out
        )
    ):
        src_hits = _search_java_sources(project_id, q, literals=literals, top_k=top_k)
        if src_hits:
            seen = {x.get("file_path") or x.get("fqcn") for x in out}
            for sh in src_hits:
                key = sh.get("file_path") or sh.get("fqcn")
                if key in seen:
                    continue
                out.append(sh)
                seen.add(key)
                if len(out) >= top_k:
                    break
    return {
        "ok": True,
        "count": len(out),
        "query_literals": literals,
        "note": (
            "优先选源码/注释/摘要字面命中 query_literals 的入口；"
            "勿用仅语义相近但字面无关的类凑答案。"
            if literals
            else None
        ),
        "hits": out,
    }


def _read_source_method(
    *,
    file_path: str,
    method_name: str | None = None,
    max_chars: int = 12000,
) -> dict[str, Any] | None:
    """Read a Java file (and optional method) from project disk."""
    from pathlib import Path

    from app.core.config import get_settings

    root = get_settings().project_root_path
    path = Path(file_path)
    if not path.is_file():
        path = root / file_path
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    rel = str(path.relative_to(root)).replace("\\", "/") if path.is_relative_to(root) else str(path)
    pkg_m = re.search(r"package\s+([\w.]+)\s*;", text)
    cls_m = re.search(r"\b(?:public\s+)?(?:class|interface)\s+(\w+)", text)
    pkg = pkg_m.group(1) if pkg_m else ""
    cls = cls_m.group(1) if cls_m else path.stem
    fqcn = f"{pkg}.{cls}" if pkg else cls
    code = text
    method = (method_name or "").strip() or None
    if method:
        # Prefer method declarations (not calls): visibility + return type + name(
        sig = re.search(
            rf"(?:public|protected|private)\s+[\w.<>,\s\[\]]+\s+{re.escape(method)}\s*\(",
            text,
        )
        if sig:
            # Walk back only over blank / annotation / javadoc lines.
            lines = text[: sig.start()].splitlines(keepends=True)
            take_from = len(lines)
            i = len(lines) - 1
            while i >= 0:
                raw = lines[i]
                s = raw.strip()
                if not s or s.startswith("@") or s.startswith("*") or s.startswith("/*") or s.endswith("*/"):
                    take_from = i
                    i -= 1
                    continue
                if s.startswith("//"):
                    take_from = i
                    i -= 1
                    continue
                break
            start = sum(len(x) for x in lines[:take_from])
            brace_at = text.find("{", sig.end() - 1)
            if brace_at > 0:
                depth = 0
                end = brace_at
                for i, ch in enumerate(text[brace_at:], brace_at):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                code = text[start:end]
    same_class = extract_same_class_calls(code, current_method=method)
    return {
        "ok": True,
        "symbol": {
            "symbol_id": f"source:{rel}#{method}" if method else f"source:{rel}",
            "kind": "Controller" if "Controller" in cls else "Class",
            "fqcn": fqcn,
            "method_name": method,
            "method_sig": f"{cls}#{method}" if method else cls,
            "file_path": rel,
            "summary": None,
        },
        "start_line": None,
        "end_line": None,
        "raw_code": truncate(code, max_chars),
        "has_full_code": True,
        "same_class_calls": same_class,
        "from_source": True,
    }


def get_symbol_code(
    project_id: str,
    *,
    symbol_id: str | None = None,
    fqcn: str | None = None,
    method_name: str | None = None,
    max_chars: int = 12000,
) -> dict[str, Any]:
    # source:relative/path.java#method — from disk scan when index lacks the symbol.
    if symbol_id and str(symbol_id).startswith("source:"):
        rest = str(symbol_id)[len("source:") :]
        file_path, _, meth = rest.partition("#")
        src = _read_source_method(
            file_path=file_path, method_name=meth or method_name, max_chars=max_chars
        )
        if src:
            return src
    # Pure source mode: never touch MySQL.
    if _explore_source_only():
        simple = (fqcn or "").split(".")[-1] if fqcn else ""
        meth = method_name
        if symbol_id and "#" in str(symbol_id) and not str(symbol_id).startswith("source:"):
            # Allow Class#method style ids from the model.
            left, _, right = str(symbol_id).partition("#")
            simple = simple or left.split(".")[-1]
            meth = meth or right.split("(")[0]
        if simple or meth:
            from app.core.config import get_settings

            root = get_settings().project_root_path
            pattern = f"**/{simple}.java" if simple else "**/*.java"
            for p in root.glob(pattern) if simple else []:
                if "target" in p.parts:
                    continue
                rel = str(p.relative_to(root)).replace("\\", "/")
                src = _read_source_method(
                    file_path=rel, method_name=meth, max_chars=max_chars
                )
                if src:
                    return src
            # method-only: scan Controllers/Services
            if meth and not simple:
                for p in list(root.rglob("*Controller.java")) + list(
                    root.rglob("*ServiceImpl.java")
                ):
                    if "target" in p.parts:
                        continue
                    try:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        continue
                    if re.search(
                        rf"(?:public|protected|private)\s+[\w.<>,\s\[\]]+\s+{re.escape(meth)}\s*\(",
                        txt,
                    ):
                        rel = str(p.relative_to(root)).replace("\\", "/")
                        src = _read_source_method(
                            file_path=rel, method_name=meth, max_chars=max_chars
                        )
                        if src:
                            return src
        return {"ok": False, "error": "symbol not found in source tree", "backend": "source"}
    with get_session() as session:
        row: SymbolORM | None = None
        if symbol_id and not str(symbol_id).startswith("source:"):
            row = session.get(SymbolORM, symbol_id)
            if row and row.project_id != project_id:
                row = None
        if row is None and fqcn:
            stmt = select(SymbolORM).where(
                SymbolORM.project_id == project_id,
                SymbolORM.fqcn == fqcn,
            )
            if method_name:
                stmt = stmt.where(SymbolORM.method_name == method_name)
            else:
                # class-level row first
                stmt = stmt.where(
                    or_(SymbolORM.method_name.is_(None), SymbolORM.method_sig == "")
                )
            row = session.scalars(stmt.limit(1)).first()
        if row is None:
            # Disk fallback by simple class name / CreditController etc.
            simple = (fqcn or "").split(".")[-1] if fqcn else ""
            if simple or method_name:
                from app.core.config import get_settings

                root = get_settings().project_root_path
                pattern = f"**/{simple}.java" if simple else "**/*Controller.java"
                for p in root.glob(pattern) if simple else root.rglob("*Controller.java"):
                    if "target" in p.parts:
                        continue
                    if method_name:
                        try:
                            txt = p.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            continue
                        if method_name not in txt:
                            continue
                    rel = str(p.relative_to(root)).replace("\\", "/")
                    src = _read_source_method(
                        file_path=rel, method_name=method_name, max_chars=max_chars
                    )
                    if src:
                        return src
            return {"ok": False, "error": "symbol not found"}
        desc = session.scalar(select(SymbolDescORM).where(SymbolDescORM.symbol_id == row.id))
        code = row.raw_code or ""
        # DB row without body → still try filesystem (never invent code).
        if not str(code).strip():
            simple = ((row.fqcn or "") or "").split(".")[-1]
            meth = row.method_name or method_name
            if simple or meth:
                from app.core.config import get_settings

                root = get_settings().project_root_path
                pattern = f"**/{simple}.java" if simple else "**/*ServiceImpl.java"
                for p in root.glob(pattern) if simple else root.rglob("*ServiceImpl.java"):
                    if "target" in p.parts:
                        continue
                    if meth:
                        try:
                            txt = p.read_text(encoding="utf-8", errors="ignore")
                        except Exception:
                            continue
                        if meth not in txt:
                            continue
                    rel = str(p.relative_to(root)).replace("\\", "/")
                    src = _read_source_method(
                        file_path=rel, method_name=meth, max_chars=max_chars
                    )
                    if src:
                        # Preserve MySQL symbol metadata when possible.
                        src["symbol"] = {
                            **(src.get("symbol") or {}),
                            **_sym_brief(row, desc.summary if desc else None),
                        }
                        return src
        same_class = extract_same_class_calls(code, current_method=row.method_name)
        out: dict[str, Any] = {
            "ok": True,
            "symbol": _sym_brief(row, desc.summary if desc else None),
            "start_line": row.start_line,
            "end_line": row.end_line,
            "raw_code": truncate(code, max_chars) if code else "(无 raw_code，可结合 list_callees 继续下钻)",
            "has_full_code": bool(code),
            "same_class_calls": same_class,
        }
        if same_class:
            out["hint"] = (
                f"源码调用了本类方法 {same_class}。"
                f"若问参数/字段校验，请立刻 "
                f'get_symbol_code fqcn={row.fqcn} method_name={same_class[0]}；'
                "不要 resolve_impl Controller，也不要跳到无关 Service。"
            )
        return out


def _neo4j_neighbors(
    project_id: str,
    symbol_id: str,
    *,
    direction: str,
    limit: int,
) -> dict[str, Any]:
    try:
        neo = get_neo4j()
        if not neo.verify():
            return {"ok": False, "error": "Neo4j unavailable", "neighbors": []}
    except Exception as e:
        return {"ok": False, "error": str(e), "neighbors": []}

    rels = "CALLS|ASYNC_CALLS|TRIGGERS|FEIGN_CALLS"
    if direction == "out":
        cypher = f"""
        MATCH (a:Method {{id: $id, project_id: $project_id}})-[r:{rels}]->(b)
        RETURN type(r) AS rel, b.id AS id, labels(b) AS labels,
               b.name AS name, b.fqcn AS fqcn, b.signature AS signature, b.kind AS kind
        LIMIT $limit
        """
    else:
        cypher = f"""
        MATCH (b)-[r:{rels}]->(a:Method {{id: $id, project_id: $project_id}})
        RETURN type(r) AS rel, b.id AS id, labels(b) AS labels,
               b.name AS name, b.fqcn AS fqcn, b.signature AS signature, b.kind AS kind
        LIMIT $limit
        """
    rows = neo.run(cypher, {"id": symbol_id, "project_id": project_id, "limit": limit})
    neighbors = []
    for r in rows:
        neighbors.append(
            {
                "rel": r.get("rel"),
                "symbol_id": r.get("id"),
                "labels": r.get("labels") or [],
                "name": r.get("name"),
                "fqcn": r.get("fqcn"),
                "signature": truncate(str(r.get("signature") or ""), 180),
                "kind": r.get("kind"),
            }
        )

    # If Neo4j has no Method node, try resolve via MySQL signature match hint
    if not neighbors:
        with get_session() as session:
            src = session.get(SymbolORM, symbol_id)
            if src and src.method_name:
                # soft hint: same-name methods elsewhere are NOT callees; just report empty
                pass
    return {"ok": True, "symbol_id": symbol_id, "direction": direction, "neighbors": neighbors}


def _list_callees_from_source(
    project_id: str, symbol_id: str, *, limit: int = 15
) -> dict[str, Any]:
    """Parse direct calls from method source (no Neo4j)."""
    code_res = get_symbol_code(project_id, symbol_id=symbol_id)
    if not code_res.get("ok"):
        return {"ok": False, "error": code_res.get("error") or "read failed", "neighbors": []}
    code = code_res.get("raw_code") or ""
    sym = code_res.get("symbol") or {}
    current = sym.get("method_name")
    neighbors: list[dict[str, Any]] = []
    seen: set[str] = set()
    # obj.method( or this.method(
    for m in re.finditer(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*\(", code):
        obj, meth = m.group(1), m.group(2)
        if obj in _JAVA_CALL_SKIP or meth in _JAVA_CALL_SKIP:
            continue
        if meth == current:
            continue
        key = f"{obj}.{meth}"
        if key in seen:
            continue
        seen.add(key)
        neighbors.append(
            {
                "name": meth,
                "fqcn": obj,
                "via": "source_call",
                "symbol_id": None,
                "hint": f"可 search_symbols q={meth} 或 resolve_impl fqcn={obj}Service",
            }
        )
        if len(neighbors) >= limit:
            break
    # same-class helpers
    for name in extract_same_class_calls(code, current_method=current):
        key = f"this.{name}"
        if key in seen:
            continue
        seen.add(key)
        neighbors.append(
            {
                "name": name,
                "fqcn": sym.get("fqcn") or "",
                "via": "same_class",
                "symbol_id": None,
                "hint": f"get_symbol_code fqcn={sym.get('fqcn')} method_name={name}",
            }
        )
        if len(neighbors) >= limit:
            break
    return {
        "ok": True,
        "symbol_id": symbol_id,
        "direction": "out",
        "backend": "source",
        "neighbors": neighbors,
        "hint": "下游来自源码调用解析；对 Service 名 resolve_impl 再 get_symbol_code。",
    }


def _list_callers_from_source(
    project_id: str, symbol_id: str, *, limit: int = 10
) -> dict[str, Any]:
    """Grep Java tree for callers of a method name (heuristic)."""
    from app.core.config import get_settings

    meth = ""
    if "#" in str(symbol_id):
        meth = str(symbol_id).split("#", 1)[-1].split("(")[0]
    if str(symbol_id).startswith("source:") and "#" in str(symbol_id):
        meth = str(symbol_id).split("#", 1)[-1]
    if not meth:
        code_res = get_symbol_code(project_id, symbol_id=symbol_id)
        meth = ((code_res.get("symbol") or {}).get("method_name") or "").strip()
    if not meth:
        return {"ok": False, "error": "cannot infer method name", "neighbors": []}

    root = get_settings().project_root_path
    neighbors: list[dict[str, Any]] = []
    pat = re.compile(rf"\b{re.escape(meth)}\s*\(")
    for p in root.rglob("*.java"):
        if "target" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        # skip the defining method itself roughly
        if re.search(
            rf"(?:public|protected|private)\s+[\w.<>,\s\[\]]+\s+{re.escape(meth)}\s*\(",
            text,
        ) and text.count(meth) <= 2:
            continue
        if not pat.search(text):
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        cls_m = re.search(r"\b(?:class|interface)\s+(\w+)", text)
        cls = cls_m.group(1) if cls_m else p.stem
        neighbors.append(
            {
                "name": cls,
                "fqcn": cls,
                "file_path": rel,
                "via": "source_grep",
                "symbol_id": f"source:{rel}",
            }
        )
        if len(neighbors) >= limit:
            break
    return {
        "ok": True,
        "symbol_id": symbol_id,
        "direction": "in",
        "backend": "source",
        "neighbors": neighbors,
        "method": meth,
    }


def _resolve_impl_from_source(
    project_id: str,
    *,
    fqcn: str | None = None,
    method_name: str | None = None,
    symbol_id: str | None = None,
) -> dict[str, Any]:
    from pathlib import Path

    from app.core.config import get_settings

    simple = ""
    if fqcn:
        simple = fqcn.split(".")[-1]
    elif symbol_id:
        raw = str(symbol_id).split("#")[0].replace("source:", "")
        simple = Path(raw).stem if ("/" in raw or "\\" in raw) else raw.split(".")[-1]
    if not simple:
        return {"ok": False, "error": "fqcn required", "impls": [], "backend": "source"}
    if simple.endswith("Controller"):
        return {
            "ok": False,
            "error": "resolve_impl 不适用于 Controller",
            "impls": [],
            "backend": "source",
        }
    iface = simple.replace("Impl", "")
    impl_name = iface if iface.endswith("Impl") else f"{iface}Impl"
    root = get_settings().project_root_path
    impls: list[dict[str, Any]] = []
    for p in root.rglob(f"{impl_name}.java"):
        if "target" in p.parts:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        pkg_m = re.search(
            r"package\s+([\w.]+)\s*;",
            p.read_text(encoding="utf-8", errors="ignore")[:500],
        )
        pkg = pkg_m.group(1) if pkg_m else ""
        fq = f"{pkg}.{impl_name}" if pkg else impl_name
        sid = f"source:{rel}"
        if method_name:
            sid = f"{sid}#{method_name}"
        impls.append(
            {
                "symbol_id": sid,
                "kind": "Service",
                "fqcn": fq,
                "method_name": method_name,
                "method_sig": f"{impl_name}#{method_name}" if method_name else impl_name,
                "file_path": rel,
                "summary": f"源码实现类 {impl_name}",
            }
        )
        break
    return {
        "ok": bool(impls),
        "impls": impls,
        "backend": "source",
        "hint": (
            "请对 impls 调用 get_symbol_code"
            if impls
            else f"未找到 {impl_name}.java；请 search_symbols q={impl_name}"
        ),
    }


def resolve_impl(
    project_id: str,
    *,
    symbol_id: str | None = None,
    fqcn: str | None = None,
    method_name: str | None = None,
) -> dict[str, Any]:
    """Map XxxService(+method) → XxxServiceImpl(+method)."""
    if _explore_source_only():
        return _resolve_impl_from_source(
            project_id, fqcn=fqcn, method_name=method_name, symbol_id=symbol_id
        )
    with get_session() as session:
        base: SymbolORM | None = None
        if symbol_id:
            base = session.get(SymbolORM, symbol_id)
            if base and base.project_id != project_id:
                base = None
        if base is None and fqcn:
            stmt = select(SymbolORM).where(
                SymbolORM.project_id == project_id,
                or_(SymbolORM.fqcn == fqcn, SymbolORM.simple_name == fqcn),
            )
            if method_name:
                stmt = stmt.where(SymbolORM.method_name == method_name)
            base = session.scalars(stmt.limit(1)).first()
        if base is None:
            return {"ok": False, "error": "interface/symbol not found", "impls": []}

        kind = (base.kind or "").lower()
        simple = base.simple_name or base.fqcn.split(".")[-1]
        if "controller" in kind or simple.endswith("Controller"):
            return {
                "ok": False,
                "error": "resolve_impl 不适用于 Controller",
                "impls": [],
                "hint": (
                    "请对入口方法 raw_code 里的 same_class_calls（如 generateReqParam/checkParam）"
                    "用 get_symbol_code(同 fqcn, method_name=...) 阅读；"
                    "仅当注入的是 XxxService 接口时才 resolve_impl 该 Service。"
                ),
            }

        iface_simple = simple.replace("Impl", "")
        method = method_name or base.method_name
        impl_name = iface_simple if iface_simple.endswith("Impl") else f"{iface_simple}Impl"

        stmt = select(SymbolORM).where(
            SymbolORM.project_id == project_id,
            or_(
                SymbolORM.simple_name == impl_name,
                SymbolORM.fqcn.like(f"%.{impl_name}"),
            ),
        )
        if method:
            stmt = stmt.where(SymbolORM.method_name == method)
        rows = list(session.scalars(stmt.limit(12)))
        if not rows and method:
            # class-level fallback then any methods on impl
            rows = list(
                session.scalars(
                    select(SymbolORM)
                    .where(
                        SymbolORM.project_id == project_id,
                        or_(
                            SymbolORM.simple_name == impl_name,
                            SymbolORM.fqcn.like(f"%.{impl_name}"),
                        ),
                    )
                    .limit(12)
                )
            )
        impls = [_sym_brief(r) for r in rows]
        hint = None
        if not impls:
            hint = f"未找到 {impl_name}；可 search_symbols q={impl_name}"
        elif method and all((r.method_name or "") != method for r in rows):
            hint = f"找到实现类但未精确匹配方法 {method}，请对下列符号 get_symbol_code"
        return {
            "ok": True,
            "from": _sym_brief(base),
            "impls": impls,
            "hint": hint
            or "请对 impls 中的 ServiceImpl 方法 get_symbol_code，继续挖 Feign/远程 Client/落库。",
        }


def list_callees(project_id: str, symbol_id: str, *, limit: int = 15) -> dict[str, Any]:
    if _explore_source_only() or str(symbol_id).startswith("source:"):
        return _list_callees_from_source(project_id, symbol_id, limit=limit)
    out = _neo4j_neighbors(project_id, symbol_id, direction="out", limit=limit)
    neighbors = out.get("neighbors") or []
    # Interface-only edges: suggest resolve_impl
    need_impl = False
    if not neighbors:
        need_impl = True
    else:
        for n in neighbors:
            fq = (n.get("fqcn") or "") + " " + (n.get("name") or "")
            if "Service" in fq and "Impl" not in fq and "Client" not in fq:
                need_impl = True
                break
    if need_impl:
        out["hint"] = (
            "下游可能是接口或图边缺失：请对 Service 接口调用 resolve_impl，"
            "再 get_symbol_code 读 *ServiceImpl（其中常有 Feign/远程 Client 调用）。"
        )
        resolved = resolve_impl(project_id, symbol_id=symbol_id)
        if resolved.get("ok") and resolved.get("impls"):
            out["suggested_impls"] = resolved["impls"][:6]
    return out


def list_callers(project_id: str, symbol_id: str, *, limit: int = 10) -> dict[str, Any]:
    if _explore_source_only() or str(symbol_id).startswith("source:"):
        return _list_callers_from_source(project_id, symbol_id, limit=limit)
    return _neo4j_neighbors(project_id, symbol_id, direction="in", limit=limit)


def _change_impact_tool_result(result: dict[str, Any]) -> dict[str, Any]:
    answer = str(result.get("answer") or "")
    tp = result.get("touchpoints")
    roles: dict[str, Any] = {}
    if tp is not None:
        for role, hit in (getattr(tp, "roles", None) or {}).items():
            roles[role] = {
                "fqcn": getattr(hit, "fqcn", "") or "",
                "method_name": getattr(hit, "method_name", "") or "",
                "file_path": getattr(hit, "file_path", "") or "",
                "class_method": getattr(hit, "class_method", "") or "",
                "why": getattr(hit, "why", "") or "",
                "evidence": getattr(hit, "evidence", "") or "",
                "table_name": getattr(hit, "table_name", None),
                "proposed_const": getattr(hit, "proposed_const", "") or "",
            }
    source = str(result.get("render_source") or "deterministic")
    return {
        "ok": bool(result.get("ok")) or bool(answer),
        "answer": answer,
        "plan_markdown": answer,
        "entry": result.get("entry"),
        "roles": roles,
        "steps": result.get("steps") or [],
        "render_source": source,
        "pass_through": True,
        "note": (
            f"已生成改动方案（{'AI 梳理' if source == 'ai' else '程序草稿'}）；"
            "finish.answer 请原样采用 plan_markdown，勿改写版式、勿增删步骤。"
        ),
    }


def discover_add_field_plan(
    project_id: str,
    question: str,
    *,
    entry_cm: str | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Run change_impact discover+render as a normal explore tool (not a graph short-circuit)."""
    from app.agent.change_impact import run_add_field_plan

    result = run_add_field_plan(
        project_id or "",
        question or "",
        entry_cm=entry_cm,
        on_progress=on_progress,
    )
    return _change_impact_tool_result(result)


def discover_protocol_sync_plan(
    project_id: str,
    question: str,
    *,
    entry_cm: str | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Run protocol-sync field pass-through discover+render as an explore tool."""
    from app.agent.change_impact import run_protocol_sync_plan

    result = run_protocol_sync_plan(
        project_id or "",
        question or "",
        entry_cm=entry_cm,
        on_progress=on_progress,
    )
    return _change_impact_tool_result(result)


def discover_composite_field_plan(
    project_id: str,
    question: str,
    *,
    entry_cm: str | None = None,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Run composite add+persist+gate/external-pass discover+render."""
    from app.agent.change_impact import run_composite_field_plan

    result = run_composite_field_plan(
        project_id or "",
        question or "",
        entry_cm=entry_cm,
        on_progress=on_progress,
    )
    return _change_impact_tool_result(result)


def dispatch_tool(project_id: str, name: str, args: dict[str, Any] | None) -> dict[str, Any]:
    args = args or {}
    if name == "search_symbols":
        return search_symbols(
            project_id,
            str(args.get("q") or ""),
            kind=args.get("kind"),
            top_k=int(args.get("top_k") or 8),
        )
    if name == "get_symbol_code":
        return get_symbol_code(
            project_id,
            symbol_id=args.get("symbol_id"),
            fqcn=args.get("fqcn"),
            method_name=args.get("method_name"),
        )
    if name == "list_callees":
        sid = args.get("symbol_id")
        if not sid:
            return {"ok": False, "error": "symbol_id required"}
        return list_callees(project_id, str(sid), limit=int(args.get("limit") or 15))
    if name == "list_callers":
        sid = args.get("symbol_id")
        if not sid:
            return {"ok": False, "error": "symbol_id required"}
        return list_callers(project_id, str(sid), limit=int(args.get("limit") or 10))
    if name == "resolve_impl":
        return resolve_impl(
            project_id,
            symbol_id=args.get("symbol_id"),
            fqcn=args.get("fqcn"),
            method_name=args.get("method_name"),
        )
    if name == "discover_add_field_plan":
        return discover_add_field_plan(
            project_id,
            str(args.get("question") or args.get("q") or ""),
            entry_cm=args.get("entry_cm") or args.get("entry"),
        )
    if name in {"discover_protocol_sync_plan", "discover_composite_field_plan"}:
        # Domain short-circuit tools removed from the explore surface.
        return {
            "ok": False,
            "error": f"tool removed: {name} (use search/get_symbol_code or discover_add_field_plan)",
        }
    if name == "finish":
        return {"ok": True, "finished": True, "answer": args.get("answer") or ""}
    return {"ok": False, "error": f"unknown tool: {name}"}
