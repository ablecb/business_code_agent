"""Java parse helpers + entrypoint.

Structural parsing is pure AI (see ai_java_parser). This module keeps
annotation/queue/accessor helpers used by graph builders and symbol store.
"""

from __future__ import annotations

import re
from pathlib import Path

from app.core.utils import resolve_spring_placeholder
from app.parser.models import ParsedAnnotation, ParseResult

CALL_RE = re.compile(r"(?:(\w+)\.)?(\w+)\s*\(")


def _decapitalize_bean(name: str) -> str:
    if not name:
        return name
    if len(name) > 1 and name[0].isupper() and name[1].isupper():
        return name
    return name[0].lower() + name[1:]


def bean_property_name(method_name: str) -> str | None:
    """Return property name for getX/setX/isX, else None."""
    if method_name.startswith("get") and len(method_name) > 3 and method_name[3].isupper():
        return _decapitalize_bean(method_name[3:])
    if method_name.startswith("set") and len(method_name) > 3 and method_name[3].isupper():
        return _decapitalize_bean(method_name[3:])
    if method_name.startswith("is") and len(method_name) > 2 and method_name[2].isupper():
        return _decapitalize_bean(method_name[2:])
    return None


def is_accessor_method(method_name: str, param_count: int) -> bool:
    """True for plain JavaBean get/set/is property methods (skip persisting these)."""
    prop = bean_property_name(method_name)
    if not prop:
        return False
    if method_name.startswith("get") or method_name.startswith("is"):
        return param_count == 0
    if method_name.startswith("set"):
        return param_count == 1
    return False


def extract_queue_from_annotation(ann: ParsedAnnotation) -> list[str]:
    if ann.name != "RabbitListener":
        return []
    raw = ann.values.get("queues") or ann.values.get("value") or ""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    return [resolve_spring_placeholder(p) for p in parts]


def extract_publish_queues(method_body: str) -> list[str]:
    queues: list[str] = []
    patterns = [
        r'convertAndSend\s*\(\s*"([^"]+)"',
        r"convertAndSend\s*\(\s*'([^']+)'",
        r'convertAndSend\s*\(\s*([^,\s]+)\s*,',
    ]
    for pat in patterns:
        for m in re.finditer(pat, method_body or ""):
            q = resolve_spring_placeholder(m.group(1))
            if q and not q.endswith("Template") and (
                "." in q
                or q.startswith("q.")
                or q.startswith("loan.")
                or "queue" in q.lower()
                or ".to." in q
            ):
                queues.append(q)
    for m in re.finditer(r"convertAndSend\s*\(\s*([A-Za-z_][\w.]*)\s*,", method_body or ""):
        queues.append(m.group(1))
    seen = set()
    out = []
    for q in queues:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def parse_java_file(file_path: Path, project_root: Path) -> ParseResult:
    """Parse a Java source file with AI only (no javalang)."""
    from app.parser.ai_java_parser import parse_java_file_with_ai

    return parse_java_file_with_ai(file_path, project_root)
