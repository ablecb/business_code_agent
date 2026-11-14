"""Project notebook (方案 C): structural memory materialised at ingest.

Deterministic scans only (packages, Controllers, pom/framework signals).
No product-domain nouns and no question-type answer templates.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import ROOT_DIR

logger = logging.getLogger("app.agent.project_notebook")

NOTEBOOK_VERSION = 1


def notebook_dir() -> Path:
    d = ROOT_DIR / "data" / "project_notebooks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def notebook_path(project_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in (project_id or "project"))
    return notebook_dir() / f"{safe}.json"


def build_notebook(project_id: str, root: Path | None = None) -> dict[str, Any]:
    """Scan repo into a durable notebook document."""
    from app.agent.codebase_survey import (
        collect_architecture_evidence,
        resolve_project_root,
    )

    resolved = root or resolve_project_root(project_id)
    if not resolved or not resolved.is_dir():
        raise FileNotFoundError(f"project root not found for {project_id!r}")
    pack = collect_architecture_evidence(resolved)
    doc: dict[str, Any] = {
        "version": NOTEBOOK_VERSION,
        "project_id": project_id,
        "root": str(resolved),
        "built_at": datetime.now(timezone.utc).isoformat(),
        "capability_counts": pack.get("capability_counts") or {},
        "techs": pack.get("techs") or [],
        "source_techs": pack.get("source_techs") or [],
        "layer_techs": pack.get("business_techs") or [],
        "apps": pack.get("apps") or [],
        "aspects": pack.get("aspects") or [],
        "annotations": pack.get("annotations") or [],
        "advices": pack.get("advices") or [],
        "packages": pack.get("packages") or [],
        "modules": list((pack.get("business_modules") or {}).get("modules") or []),
        "controllers": list((pack.get("business_modules") or {}).get("controllers") or []),
        "brief": str(pack.get("brief") or ""),
    }
    return doc


def save_notebook(doc: dict[str, Any]) -> Path:
    pid = str(doc.get("project_id") or "project")
    path = notebook_path(pid)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_notebook(project_id: str) -> dict[str, Any] | None:
    path = notebook_path(project_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.debug("load_notebook failed path=%s", path, exc_info=True)
        return None
    if not isinstance(data, dict):
        return None
    if int(data.get("version") or 0) != NOTEBOOK_VERSION:
        return None
    return data


def build_and_save_notebook(project_id: str, root: Path | None = None) -> dict[str, Any]:
    doc = build_notebook(project_id, root=root)
    path = save_notebook(doc)
    logger.info(
        "project notebook saved project_id=%s path=%s controllers=%s techs=%s",
        project_id,
        path,
        len(doc.get("controllers") or []),
        len(doc.get("techs") or []),
    )
    return doc


def ensure_notebook(
    project_id: str,
    *,
    root: Path | None = None,
    rebuild: bool = False,
) -> dict[str, Any]:
    """Return notebook; build if missing or rebuild=True."""
    if not rebuild:
        existing = load_notebook(project_id)
        if existing and (existing.get("brief") or existing.get("controllers") is not None):
            return existing
    return build_and_save_notebook(project_id, root=root)


def notebook_brief(doc: dict[str, Any] | None) -> str:
    if not doc:
        return ""
    brief = str(doc.get("brief") or "").strip()
    if brief:
        return brief
    # Reconstruct a minimal brief if older/partial docs lack it.
    lines = [
        f"project_id={doc.get('project_id')}",
        f"root={doc.get('root')}",
        f"built_at={doc.get('built_at')}",
        "",
        "## pom 技术线索",
        "、".join(doc.get("techs") or []) or "（无）",
        "",
        "## 源码框架信号",
        "、".join(doc.get("source_techs") or []) or "（无）",
        "",
        "## 分层框架信号",
        "、".join(doc.get("layer_techs") or doc.get("business_techs") or []) or "（无）",
        "",
        "## 能力数量",
    ]
    for k, v in dict(doc.get("capability_counts") or {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## 启动类")
    for a in list(doc.get("apps") or [])[:8]:
        lines.append(f"- {a.get('fqcn')} @ {a.get('path')}")
    lines.append("")
    lines.append("## 分层子模块")
    for m in list(doc.get("modules") or [])[:25]:
        lines.append(f"- {m.get('module')}: {m.get('java_files')}")
    lines.append("")
    lines.append("## Controller 入口")
    for c in list(doc.get("controllers") or [])[:40]:
        extra = f" mappings={c.get('mappings')}" if c.get("mappings") else ""
        lines.append(f"- {c.get('class')} @ {c.get('path')}{extra}")
    return "\n".join(lines)
