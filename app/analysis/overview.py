"""Stage 3: AI project overview from pom/yml/directory → MySQL + Milvus."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from app.core.config import get_settings
from app.core.utils import stable_id, truncate
from app.db.milvus_client import get_milvus
from app.db.mysql import get_session
from app.llm.factory import get_embedding_provider, get_parse_llm_provider
from app.llm.tracing import traceable
from app.models.orm import ProjectMetaORM, ProjectORM
from app.parser.scanner import SKIP_DIR_NAMES, read_pom_modules

logger = logging.getLogger("app.analysis.overview")

OVERVIEW_PROMPT = """你是资深 Java/Spring 架构师。根据给定的项目目录与配置片段，输出严格 JSON（不要 markdown）：
{
  "summary": "一段中文项目总览（2-4句）",
  "tech_stack": ["Spring Boot", "..."],
  "versions": {"spring_boot": "x.y", "java": "17", "其他关键依赖": "版本"},
  "modules": [{"name": "模块名", "role": "一句话职责"}],
  "entry_hints": ["Controller", "RabbitMQ", "XxlJob", "..."],
  "notes": "其他值得注意的点"
}
只根据提供的材料推断，不要编造未出现的依赖版本。
"""


def _read_text(path: Path, limit: int = 8000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except OSError:
        return ""


def _find_files(root: Path, names: tuple[str, ...], *, limit: int = 20) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        if path.name in names or any(path.name.startswith(n.rstrip("*")) for n in names if n.endswith("*")):
            found.append(path)
            if len(found) >= limit:
                break
    # also match application*.yml/yaml/properties
    for pattern in ("application*.yml", "application*.yaml", "application*.properties", "bootstrap*.yml"):
        for path in root.rglob(pattern):
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if path not in found:
                found.append(path)
            if len(found) >= limit:
                return found
    return found


def _dir_tree_summary(root: Path, *, max_depth: int = 3, max_entries: int = 80) -> str:
    lines: list[str] = []
    root = root.resolve()

    def walk(dir_path: Path, depth: int) -> None:
        if len(lines) >= max_entries or depth > max_depth:
            return
        try:
            entries = sorted(dir_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            return
        for entry in entries:
            if entry.name in SKIP_DIR_NAMES or entry.name.startswith("."):
                continue
            rel = entry.relative_to(root).as_posix()
            if entry.is_dir():
                lines.append(f"{'  ' * depth}{rel}/")
                walk(entry, depth + 1)
            elif entry.suffix in {".xml", ".yml", ".yaml", ".gradle", ".kts", ".properties", ".md"}:
                lines.append(f"{'  ' * depth}{rel}")

    walk(root, 0)
    return "\n".join(lines[:max_entries])


def _extract_pom_bits(text: str) -> dict[str, str]:
    bits: dict[str, str] = {}
    for key in ("java.version", "spring-boot.version", "spring-cloud.version"):
        m = re.search(rf"<{re.escape(key)}>\s*([^<]+)\s*</{re.escape(key)}>", text)
        if m:
            bits[key] = m.group(1).strip()
    parent = re.search(
        r"<parent>.*?<artifactId>\s*([^<]+)\s*</artifactId>.*?<version>\s*([^<]+)\s*</version>.*?</parent>",
        text,
        re.DOTALL,
    )
    if parent:
        bits["parent_artifact"] = parent.group(1).strip()
        bits["parent_version"] = parent.group(2).strip()
    deps = re.findall(
        r"<dependency>\s*<groupId>\s*([^<]+)\s*</groupId>\s*<artifactId>\s*([^<]+)\s*</artifactId>"
        r"(?:\s*<version>\s*([^<]+)\s*</version>)?",
        text,
        re.DOTALL,
    )
    interesting = []
    for gid, aid, ver in deps[:40]:
        aid_l = aid.lower()
        if any(
            k in aid_l
            for k in (
                "spring",
                "mybatis",
                "rabbit",
                "kafka",
                "xxl",
                "feign",
                "nacos",
                "redis",
                "mysql",
                "oracle",
                "mongodb",
            )
        ):
            interesting.append(f"{aid}:{ver.strip() if ver else '?'}")
    if interesting:
        bits["key_deps"] = ", ".join(interesting[:25])
    return bits


def collect_project_signals(project_root: Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    modules = read_pom_modules(root)
    pom_files = _find_files(root, ("pom.xml", "build.gradle", "build.gradle.kts"), limit=12)
    yml_files = _find_files(root, ("application.yml", "application.yaml", "bootstrap.yml"), limit=12)

    pom_snippets: list[str] = []
    versions: dict[str, str] = {}
    for pom in pom_files[:8]:
        text = _read_text(pom, 12000)
        bits = _extract_pom_bits(text)
        versions.update(bits)
        rel = pom.name
        try:
            rel = pom.relative_to(root).as_posix()
        except ValueError:
            pass
        pom_snippets.append(f"### {rel}\n{truncate(text, 3500)}")

    yml_snippets: list[str] = []
    for yml in yml_files[:8]:
        text = _read_text(yml, 4000)
        rel = yml.name
        try:
            rel = yml.relative_to(root).as_posix()
        except ValueError:
            pass
        yml_snippets.append(f"### {rel}\n{truncate(text, 2500)}")

    tree = _dir_tree_summary(root)
    return {
        "modules": modules,
        "versions_heuristic": versions,
        "tree": tree,
        "pom_snippets": pom_snippets,
        "yml_snippets": yml_snippets,
        "has_pom": any(p.name == "pom.xml" for p in pom_files),
        "has_gradle": any("gradle" in p.name for p in pom_files),
    }


def _heuristic_overview(signals: dict[str, Any]) -> dict[str, Any]:
    versions = dict(signals.get("versions_heuristic") or {})
    tech: list[str] = []
    if signals.get("has_pom") or signals.get("has_gradle"):
        tech.append("Maven" if signals.get("has_pom") else "Gradle")
    blob = "\n".join(
        [
            *(signals.get("pom_snippets") or []),
            *(signals.get("yml_snippets") or []),
            signals.get("tree") or "",
        ]
    ).lower()
    rules = [
        ("spring-boot", "Spring Boot"),
        ("spring-cloud", "Spring Cloud"),
        ("mybatis", "MyBatis"),
        ("rabbit", "RabbitMQ"),
        ("kafka", "Kafka"),
        ("xxl", "XxlJob"),
        ("feign", "OpenFeign"),
        ("nacos", "Nacos"),
        ("redis", "Redis"),
        ("mysql", "MySQL"),
    ]
    for key, name in rules:
        if key in blob and name not in tech:
            tech.append(name)
    modules = [
        {"name": m, "role": "子模块"}
        for m in (signals.get("modules") or [])[:20]
    ]
    entries = []
    for hint, label in (
        ("controller", "Controller"),
        ("rabbit", "RabbitMQ"),
        ("xxl", "XxlJob"),
        ("kafka", "Kafka"),
    ):
        if hint in blob:
            entries.append(label)
    summary = "Java 多模块项目" if modules else "Java 项目"
    if tech:
        summary += "，技术栈含 " + "、".join(tech[:6])
    if versions.get("parent_version"):
        summary += f"；parent 版本 {versions['parent_version']}"
    return {
        "summary": summary + "。",
        "tech_stack": tech,
        "versions": versions,
        "modules": modules,
        "entry_hints": entries,
        "notes": "",
    }


@traceable(name="overview_llm", run_type="llm")
async def _llm_overview(signals: dict[str, Any]) -> dict[str, Any]:
    llm = get_parse_llm_provider()
    user = (
        f"modules={signals.get('modules')}\n"
        f"heuristic_versions={json.dumps(signals.get('versions_heuristic') or {}, ensure_ascii=False)}\n"
        f"directory_tree:\n{truncate(signals.get('tree') or '', 3000)}\n\n"
        f"poms:\n{truncate(chr(10).join(signals.get('pom_snippets') or []), 8000)}\n\n"
        f"configs:\n{truncate(chr(10).join(signals.get('yml_snippets') or []), 5000)}"
    )
    try:
        content = await llm.chat(
            [
                {"role": "system", "content": OVERVIEW_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
        if not isinstance(data, dict) or not data.get("summary"):
            raise ValueError("bad overview json")
        return data
    except Exception:
        logger.exception("overview LLM failed; using heuristic")
        return _heuristic_overview(signals)


def _persist_overview(project_id: str, data: dict[str, Any]) -> str:
    summary = str(data.get("summary") or "").strip()
    tech = data.get("tech_stack") or []
    if isinstance(tech, list):
        tech_stack = "、".join(str(t) for t in tech if str(t).strip())
    else:
        tech_stack = str(tech)
    with get_session() as session:
        proj = session.get(ProjectORM, project_id)
        if proj:
            proj.description = truncate(summary, 4000)
        session.merge(
            ProjectMetaORM(
                project_id=project_id,
                tech_stack=truncate(tech_stack, 4000) or None,
                overview_json=data,
            )
        )
    return summary


def _milvus_overview_row(project_id: str, data: dict[str, Any], summary: str) -> dict[str, Any]:
    tech = data.get("tech_stack") or []
    tech_s = " ".join(str(t) for t in tech) if isinstance(tech, list) else str(tech)
    versions = data.get("versions") or {}
    text = truncate(
        f"项目总览 {summary} 技术栈 {tech_s} 版本 {json.dumps(versions, ensure_ascii=False)} "
        f"入口线索 {data.get('entry_hints')}",
        2000,
    )
    doc_id = stable_id("overview", project_id)
    return {
        "id": doc_id,
        "project_id": project_id,
        "symbol_id": f"overview:{project_id}"[:64],
        "fqcn": "__project_overview__",
        "tags": "项目总览,tech_stack",
        "text": text,
    }


@traceable(name="build_project_overview", run_type="chain")
async def build_project_overview(
    project_id: str,
    project_root: Path,
    *,
    use_llm: bool = True,
    progress: Callable[[int, int, str], None] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    if progress:
        progress(0, 3, "collecting pom/yml/tree")
    signals = collect_project_signals(Path(project_root))
    if progress:
        progress(1, 3, "analyzing overview with AI")
    data = await _llm_overview(signals) if use_llm else _heuristic_overview(signals)
    # merge heuristic versions if LLM omitted them
    hv = signals.get("versions_heuristic") or {}
    versions = data.get("versions") if isinstance(data.get("versions"), dict) else {}
    for k, v in hv.items():
        versions.setdefault(k, v)
    data["versions"] = versions

    summary = _persist_overview(project_id, data)
    if progress:
        progress(2, 3, "upserting overview to Milvus")

    row = _milvus_overview_row(project_id, data, summary)
    try:
        embedder = get_embedding_provider()
        milvus = get_milvus()
        emb = embedder.embed([row["text"]])
        milvus.dim = len(emb[0]) if emb else settings.embedding_dim
        milvus.upsert([row], emb)
    except Exception:
        logger.exception("overview milvus upsert failed; MySQL kept")

    if progress:
        progress(3, 3, "overview done")
    logger.info(
        "overview done project=%s tech=%s",
        project_id,
        data.get("tech_stack"),
    )
    return {
        "summary_len": len(summary),
        "tech_stack": data.get("tech_stack") or [],
        "modules": len(data.get("modules") or []),
    }
