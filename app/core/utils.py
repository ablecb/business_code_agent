from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path


def new_id(prefix: str = "") -> str:
    uid = uuid.uuid4().hex
    return f"{prefix}{uid}" if prefix else uid


def stable_id(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def resolve_spring_placeholder(value: str) -> str:
    """Extract default from ${key:default} or return literal."""
    if not value:
        return value
    value = value.strip().strip('"').strip("'")
    m = re.fullmatch(r"\$\{[^}:]+:([^}]+)\}", value)
    if m:
        return m.group(1).strip()
    m2 = re.fullmatch(r"\$\{([^}]+)\}", value)
    if m2:
        return m2.group(1).strip()
    return value


def module_from_path(project_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.relative_to(project_root)
    except ValueError:
        return "unknown"
    parts = rel.parts
    if "src" in parts:
        idx = parts.index("src")
        if idx > 0:
            return parts[idx - 1]
    return parts[0] if parts else "unknown"


def truncate(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
