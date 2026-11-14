"""Zero-dependency helper utilities extracted from explore_agent.py.

These functions have NO dependencies on other symbols inside explore_agent,
so they live here and are re-exported by explore_agent.py to keep existing
``from app.agent.explore_agent import _class_method_from_hint`` style imports
working unchanged (graph_agent / pev_agent / ask_retrieval / source_view /
patcher / field_evidence / compose_code / change_impact all rely on them).
"""
from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger("app.agent.explore_shared")


def _hit_text(h: Any) -> str:
    if isinstance(h, dict):
        return f"{h.get('summary') or ''} {h.get('text') or ''} {h.get('fqcn') or ''}"
    return (
        f"{getattr(h, 'summary', '') or ''} {getattr(h, 'text', '') or ''} "
        f"{getattr(h, 'fqcn', '') or ''}"
    )


def _controller_stem(name: str) -> str:
    simple = (name or "").split(".")[-1].split("#")[0]
    return re.sub(
        r"(Controller|RestController|ServiceImpl|Service|Listener|Job)$",
        "",
        simple,
        flags=re.I,
    ).lower()


def _hit_method_name(h: Any) -> str:
    if isinstance(h, dict):
        cm = str(h.get("class_method") or "")
        if "#" in cm:
            return cm.split("#", 1)[1]
        sig = str(h.get("method_sig") or h.get("method_name") or "")
    else:
        cm = ""
        ej = getattr(h, "extra", None)
        sig = str(getattr(h, "method_sig", None) or getattr(h, "method_name", None) or "")
    if "#" in sig:
        # fqcn#method( or Class#method
        mid = sig.split("#", 1)[1]
        return mid.split("(", 1)[0].strip()
    # "public Foo bar(" declaration
    if "(" in sig:
        before = sig.split("(", 1)[0].strip().split()
        return before[-1] if before else ""
    return sig


def _hit_fqcn(h: Any) -> str:
    if isinstance(h, dict):
        return str(h.get("fqcn") or "")
    return str(getattr(h, "fqcn", "") or "")


def _hit_class_method(h: Any) -> str:
    if isinstance(h, dict):
        cm = str(h.get("class_method") or "")
        if cm:
            return cm
        fqcn = _hit_fqcn(h)
        method = _hit_method_name(h)
        simple = fqcn.split(".")[-1] if fqcn else ""
        return f"{simple}#{method}" if simple and method else cm
    fqcn = _hit_fqcn(h)
    method = _hit_method_name(h)
    simple = fqcn.split(".")[-1] if fqcn else ""
    return f"{simple}#{method}" if simple and method else ""


def _method_from_class_method(cm: str) -> str:
    if "#" in (cm or ""):
        return cm.split("#", 1)[1].split("(", 1)[0].strip()
    return ""


def _class_method_from_hint(hint: str | None) -> str | None:
    """Extract Class#method from entry_hint / session last_entry when present."""
    text = (hint or "").strip()
    if not text or "#" not in text:
        return None
    # Prefer first token that looks like Class#method (ignore trailing notes).
    for part in re.split(r"[\s,;；]+", text):
        if "#" not in part:
            continue
        left, right = part.split("#", 1)
        simple = left.split(".")[-1].strip()
        method = right.split("(", 1)[0].strip()
        if simple and method and re.match(r"^[A-Za-z_][\w]*$", simple) and re.match(
            r"^[A-Za-z_][\w]*$", method
        ):
            return f"{simple}#{method}"
    return None


def _simple_class_method(cm: str | None) -> str:
    """Neo4j Method.class_method is SimpleClass#method (not FQCN)."""
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    simple = left.split(".")[-1].strip()
    method = right.split("(", 1)[0].strip()
    if not simple or not method:
        return text
    return f"{simple}#{method}"


def _read_count(steps: list[dict[str, Any]]) -> int:
    return sum(1 for s in steps if s.get("tool") == "get_symbol_code" and s.get("ok"))


def parse_numbered_answer_steps(reply: str) -> list[dict[str, Any]]:
    """Parse『1. **受理请求**：…』style steps from prior assistant reply."""
    text = reply or ""
    out: list[dict[str, Any]] = []
    for m in re.finditer(
        r"(?m)^\s*(\d+)\.\s*\*\*(.+?)\*\*\s*[：:．.]?\s*(.*)$",
        text,
    ):
        out.append(
            {
                "index": int(m.group(1)),
                "title": m.group(2).strip(),
                "detail": (m.group(3) or "").strip(),
            }
        )
    if out:
        return out
    for m in re.finditer(
        r"(?m)^\s*(\d+)\.\s+([^\n：:*]{2,40})[：:．.]?\s*(.*)$",
        text,
    ):
        title = m.group(2).strip().strip("*").strip()
        if not title or title.startswith("依据"):
            continue
        out.append(
            {
                "index": int(m.group(1)),
                "title": title,
                "detail": (m.group(3) or "").strip(),
            }
        )
    return out


def _method_name_from_hit(h: dict[str, Any]) -> str:
    """search_symbols often omits method_name; recover from method_sig."""
    mname = str(h.get("method_name") or "").strip()
    if mname:
        return mname.split("(", 1)[0]
    sig = str(h.get("method_sig") or "").strip()
    if not sig:
        return ""
    if "#" in sig:
        return sig.split("#", 1)[1].split("(", 1)[0].strip()
    if "(" in sig:
        parts = sig.split("(", 1)[0].strip().split()
        return parts[-1] if parts else ""
    return ""


def _locate_java_file_for_class(project_id: str, class_simple: str) -> str | None:
    """Resolve relative Java path for a simple class name (MySQL → disk glob)."""
    simple = (class_simple or "").split(".")[-1].strip()
    if not project_id or not simple:
        return None
    try:
        from sqlalchemy import or_, select

        from app.db.mysql import get_session
        from app.models.orm import SymbolORM

        with get_session() as session:
            row = session.scalars(
                select(SymbolORM)
                .where(
                    SymbolORM.project_id == project_id,
                    or_(
                        SymbolORM.simple_name == simple,
                        SymbolORM.fqcn.like(f"%.{simple}"),
                    ),
                    SymbolORM.file_path.is_not(None),
                    SymbolORM.file_path != "",
                )
                .limit(1)
            ).first()
            if row and row.file_path:
                return str(row.file_path).replace("\\", "/")
    except Exception:
        logger.debug("locate java file mysql failed", exc_info=True)
    try:
        from app.core.config import get_settings

        root = get_settings().project_root_path
        for p in root.glob(f"**/{simple}.java"):
            if "target" in p.parts:
                continue
            return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        pass
    return None


def _collect_entry_calls_closure(
    project_id: str,
    entry_cm: str,
    *,
    depth: int = 4,
    limit: int = 160,
) -> list[str]:
    """BFS-ish CALLS closure under locked entry (Neo4j), entry first."""
    entry = _simple_class_method(entry_cm)
    if not project_id or not entry or "#" not in entry:
        return []
    depth = max(1, min(int(depth), 6))
    limit = max(10, min(int(limit), 300))
    try:
        from app.db.neo4j_client import get_neo4j

        client = get_neo4j()
        rows = client.run(
            f"""
            MATCH (a:Method {{project_id: $pid}})
            WHERE a.class_method = $cm
            MATCH path = (a)-[:CALLS*1..{depth}]->(b:Method)
            WHERE b.project_id = $pid
            RETURN DISTINCT b.class_method AS cm, length(path) AS dist
            ORDER BY dist, b.class_method
            LIMIT $limit
            """,
            {"pid": project_id, "cm": entry, "limit": limit},
        )
    except Exception:
        logger.debug("CALLS closure lookup failed", exc_info=True)
        return []
    out: list[str] = []
    seen = {entry}
    for row in rows or []:
        cm = str((row or {}).get("cm") or "").strip()
        if not cm or "#" not in cm or cm in seen:
            continue
        seen.add(cm)
        out.append(cm)
    return out
