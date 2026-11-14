from __future__ import annotations

from pathlib import Path

SKIP_DIR_NAMES = {
    "target",
    ".git",
    ".idea",
    "node_modules",
    ".svn",
    "build",
    "out",
    ".agent_workspace",
}


def _load_gitignore_specs(project_root: Path) -> list[tuple[Path, object]]:
    """Load root + nested .gitignore as (base_dir, PathSpec)."""
    try:
        from pathspec import PathSpec
        from pathspec.patterns.gitwildmatch import GitWildMatchPattern
    except ImportError:
        return []

    specs: list[tuple[Path, object]] = []
    root = project_root.resolve()
    for gi in root.rglob(".gitignore"):
        if any(part in SKIP_DIR_NAMES for part in gi.parts):
            continue
        try:
            lines = gi.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        # Drop comments/blank — PathSpec tolerates them, but keep file readable
        spec = PathSpec.from_lines(GitWildMatchPattern, lines)
        specs.append((gi.parent.resolve(), spec))
    # Prefer longer (deeper) bases last so nested rules can be checked;
    # match if ANY ancestor gitignore matches (good enough for exclude use-case).
    specs.sort(key=lambda x: len(x[0].parts))
    return specs


def _is_gitignored(path: Path, specs: list[tuple[Path, object]]) -> bool:
    if not specs:
        return False
    path = path.resolve()
    is_dir = path.is_dir()
    for base, spec in specs:
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            continue
        if not rel or rel == ".":
            continue
        if spec.match_file(rel):
            return True
        if is_dir and spec.match_file(rel.rstrip("/") + "/"):
            return True
    return False


def _looks_like_mapper_xml(path: Path) -> bool:
    name = path.name
    if "Mapper" in name or name.lower().endswith("mapper.xml"):
        return True
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:2048].lower()
    except OSError:
        return False
    return "<mapper" in head and ("mybatis" in head or "namespace=" in head)


def scan_java_files(project_root: Path) -> list[Path]:
    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    gitignore_specs = _load_gitignore_specs(root)
    files: list[Path] = []

    def _walk(dir_path: Path) -> None:
        try:
            entries = list(dir_path.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIR_NAMES:
                    continue
                if _is_gitignored(entry, gitignore_specs):
                    continue
                _walk(entry)
            elif entry.is_file() and entry.suffix == ".java":
                if _is_gitignored(entry, gitignore_specs):
                    continue
                parts = entry.parts
                if "src" in parts and "main" in parts and "java" in parts:
                    files.append(entry)

    _walk(root)
    files.sort()
    return files


def scan_mapper_xml_files(project_root: Path) -> list[Path]:
    """Scan MyBatis Mapper XML under src/main/resources (respects .gitignore)."""
    root = Path(project_root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Project root not found: {root}")

    gitignore_specs = _load_gitignore_specs(root)
    files: list[Path] = []

    def _walk(dir_path: Path) -> None:
        try:
            entries = list(dir_path.iterdir())
        except (PermissionError, OSError):
            return
        for entry in entries:
            if entry.is_dir():
                if entry.name in SKIP_DIR_NAMES:
                    continue
                if _is_gitignored(entry, gitignore_specs):
                    continue
                _walk(entry)
            elif entry.is_file() and entry.suffix.lower() == ".xml":
                if _is_gitignored(entry, gitignore_specs):
                    continue
                parts = entry.parts
                if "src" in parts and "main" in parts and "resources" in parts:
                    if _looks_like_mapper_xml(entry):
                        files.append(entry)

    _walk(root)
    files.sort()
    return files


def scan_source_files(project_root: Path) -> list[Path]:
    """Java sources + MyBatis mapper XMLs."""
    return scan_java_files(project_root) + scan_mapper_xml_files(project_root)


def read_pom_modules(project_root: Path) -> list[str]:
    pom = Path(project_root) / "pom.xml"
    if not pom.exists():
        return []
    text = pom.read_text(encoding="utf-8", errors="ignore")
    import re

    return re.findall(r"<module>\s*([^<]+)\s*</module>", text)
