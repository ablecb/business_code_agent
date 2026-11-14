from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _target_root() -> Path:
    settings = get_settings()
    if settings.agent_write_mode == "inplace":
        return settings.project_root_path
    # workspace: copy-on-write base
    ws = settings.workspace_path
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def ensure_workspace_file(rel_path: str) -> Path:
    settings = get_settings()
    src = settings.project_root_path / rel_path
    if settings.agent_write_mode == "inplace":
        return src
    dst = settings.workspace_path / rel_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() and src.exists():
        shutil.copy2(src, dst)
    return dst


def apply_replacements(rel_path: str, replacements: list[dict[str, str]]) -> dict[str, Any]:
    path = ensure_workspace_file(rel_path)
    if not path.exists():
        return {"file": rel_path, "ok": False, "error": "file not found"}
    text = path.read_text(encoding="utf-8", errors="ignore")
    original = text
    applied = 0
    changes: list[dict[str, Any]] = []
    for rep in replacements:
        old = rep.get("old") or rep.get("old_string") or ""
        new = rep.get("new") or rep.get("new_string") or ""
        if not old:
            continue
        idx = text.find(old)
        if idx < 0:
            continue
        start_line = text.count("\n", 0, idx) + 1
        old_end = start_line + old.count("\n")
        new_end = start_line + new.count("\n")
        text = text[:idx] + new + text[idx + len(old) :]
        applied += 1
        changes.append(
            {
                "start_line": start_line,
                "end_line": max(start_line, new_end),
                "old_start_line": start_line,
                "old_end_line": max(start_line, old_end),
            }
        )
    if applied == 0:
        return {"file": rel_path, "ok": False, "error": "no replacement matched", "applied": 0}
    path.write_text(text, encoding="utf-8")
    return {
        "file": rel_path,
        "ok": True,
        "applied": applied,
        "changed_bytes": abs(len(text) - len(original)),
        "write_path": str(path),
        "changes": changes,
        "line_summary": format_change_line_summary(rel_path, changes=changes),
    }


def write_new_file(rel_path: str, content: str) -> dict[str, Any]:
    path = ensure_workspace_file(rel_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = content or ""
    path.write_text(body, encoding="utf-8")
    if not body:
        n_lines = 0
    elif body.endswith("\n"):
        n_lines = max(1, body.count("\n"))
    else:
        n_lines = body.count("\n") + 1
    changes = [{"start_line": 1, "end_line": max(1, n_lines)}] if n_lines else []
    return {
        "file": rel_path,
        "ok": True,
        "applied": 1,
        "write_path": str(path),
        "new_file": True,
        "changes": changes,
        "line_summary": format_change_line_summary(
            rel_path, new_file=True, end_line=max(1, n_lines) if n_lines else 1
        ),
    }


def format_change_line_summary(
    file_path: str,
    *,
    changes: list[dict[str, Any]] | None = None,
    new_file: bool = False,
    end_line: int | None = None,
) -> str:
    """Human-readable one-liner: path + line range(s)."""
    fp = (file_path or "").replace("\\", "/")
    if new_file:
        el = int(end_line or 1)
        return f"`{fp}` — 新增 第 1–{el} 行"
    spans: list[str] = []
    for c in changes or []:
        s = int(c.get("start_line") or 0)
        e = int(c.get("end_line") or s)
        if s <= 0:
            continue
        spans.append(f"第 {s} 行" if e <= s else f"第 {s}–{e} 行")
    if not spans:
        return f"`{fp}` — 有改动（行号未知）"
    return f"`{fp}` — " + "；".join(spans)


def preview_patch_line_spans(patch: dict[str, Any]) -> list[dict[str, Any]]:
    """Estimate line spans for a patch without writing (for draft replies)."""
    fp = str(patch.get("file") or patch.get("path") or "")
    if patch.get("new_file") or patch.get("content") is not None:
        body = str(patch.get("content") or "")
        n = body.count("\n") + (0 if (not body or body.endswith("\n")) else 1)
        if body.endswith("\n") and body:
            n = max(1, body.count("\n"))
        elif body:
            n = body.count("\n") + 1
        else:
            n = 1
        return [{"start_line": 1, "end_line": n, "new_file": True}]
    try:
        path = ensure_workspace_file(fp) if fp else None
        text = path.read_text(encoding="utf-8", errors="ignore") if path and path.exists() else ""
    except Exception:
        text = ""
    out: list[dict[str, Any]] = []
    working = text
    for rep in patch.get("replacements") or []:
        old = rep.get("old") or rep.get("old_string") or ""
        new = rep.get("new") or rep.get("new_string") or ""
        if not old or not working:
            continue
        idx = working.find(old)
        if idx < 0:
            continue
        start_line = working.count("\n", 0, idx) + 1
        new_end = start_line + new.count("\n")
        out.append(
            {
                "start_line": start_line,
                "end_line": max(start_line, new_end),
                "old_start_line": start_line,
                "old_end_line": start_line + old.count("\n"),
            }
        )
        working = working[:idx] + new + working[idx + len(old) :]
    return out


def summarize_patches_with_lines(patches: list[dict[str, Any]]) -> list[str]:
    """Build bullet lines for UI/reply: file + line ranges."""
    lines: list[str] = []
    for p in patches or []:
        if not isinstance(p, dict):
            continue
        fp = str(p.get("file") or p.get("path") or "").strip()
        if not fp:
            continue
        if p.get("line_summary"):
            lines.append(f"- {p['line_summary']}")
            continue
        changes = p.get("changes")
        if not changes:
            changes = preview_patch_line_spans(p)
        is_new = bool(p.get("new_file") or (p.get("content") is not None and not p.get("replacements")))
        end = None
        if is_new and changes:
            end = int(changes[0].get("end_line") or 1)
        lines.append(
            "- "
            + format_change_line_summary(
                fp, changes=None if is_new else changes, new_file=is_new, end_line=end
            )
        )
    return lines


def summarize_patch_results(results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        fp = str(r.get("file") or r.get("path") or "")
        if r.get("line_summary"):
            mark = "[ok]" if r.get("ok") else "[fail]"
            lines.append(f"- {mark} {r['line_summary']}")
            continue
        if r.get("ok") and (r.get("changes") or r.get("new_file")):
            lines.append(
                "- [ok] "
                + format_change_line_summary(
                    fp,
                    changes=r.get("changes"),
                    new_file=bool(r.get("new_file")),
                    end_line=(r.get("changes") or [{}])[0].get("end_line") if r.get("new_file") else None,
                )
            )
        elif r.get("ok"):
            lines.append(f"- [ok] `{fp}` — 已写入")
        else:
            lines.append(f"- [fail] `{fp}` — {r.get('error') or '失败'}")
    return lines


def parse_patches_from_llm(text: str) -> list[dict[str, Any]]:
    """Extract patch blocks:
    ### FILE: path
    *** OLD
    ...
    *** NEW
    ...
    """
    patches: list[dict[str, Any]] = []
    parts = re.split(r"(?m)^### FILE:\s*", text)
    for part in parts[1:]:
        lines = part.splitlines()
        if not lines:
            continue
        file_path = lines[0].strip()
        body = "\n".join(lines[1:])
        m = re.search(
            r"\*\*\*\s*OLD\s*\n(?P<old>.*?)\n\*\*\*\s*NEW\s*\n(?P<new>.*?)(?:\n###|\Z)",
            body,
            re.DOTALL,
        )
        if not m:
            # new file
            m2 = re.search(r"\*\*\*\s*NEW_FILE\s*\n(?P<new>.*?)(?:\n###|\Z)", body, re.DOTALL)
            if m2:
                patches.append({"file": file_path, "new_file": True, "content": m2.group("new").rstrip() + "\n"})
            continue
        patches.append(
            {
                "file": file_path,
                "replacements": [{"old": m.group("old"), "new": m.group("new")}],
            }
        )
    return patches


def extract_scaffold_file_patches(text: str) -> list[dict[str, Any]]:
    """Parse add_api / impact markdown: `path/Foo.java` + ```java ... ``` into write patches."""
    patches: list[dict[str, Any]] = []
    seen: set[str] = set()
    pattern = re.compile(
        r"`((?:[\w.\-]+/)+[\w.\-]+\.java)`\s*\n+```(?:java)?\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(text or ""):
        path = m.group(1).replace("\\", "/")
        content = (m.group(2) or "").rstrip() + "\n"
        if path in seen:
            continue
        # Require looking like a real Java compilation unit
        if not re.search(r"\b(package|public\s+(class|interface|enum))\b", content):
            continue
        seen.add(path)
        patches.append(
            {
                "file": path,
                "path": path,
                "new_file": True,
                "content": content,
            }
        )
    return patches


def patch_file_key(p: dict[str, Any]) -> str:
    return str(p.get("file") or p.get("path") or "").replace("\\", "/").strip()


def real_scope_files(scope: list[Any] | None) -> list[str]:
    """Repo-relative files only — drop explore fake paths like step0:pev_goal."""
    out: list[str] = []
    seen: set[str] = set()
    for s in scope or []:
        if not isinstance(s, dict):
            continue
        f = str(s.get("file") or "").replace("\\", "/").strip()
        if not f or f.startswith("step"):
            continue
        # reject tool-ish fake ids
        head = f.split("/", 1)[0]
        if ":" in head:
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def filter_patches_to_allowlist(
    patches: list[dict[str, Any]],
    allowlist: list[str] | None,
    *,
    drop_if_empty_allowlist: bool = False,
) -> list[dict[str, Any]]:
    """Keep only patches whose path is in allowlist. Prevents over-scoped edits."""
    if not allowlist:
        return [] if drop_if_empty_allowlist else list(patches or [])
    allowed = {a.replace("\\", "/").strip() for a in allowlist if a}
    if not allowed:
        return [] if drop_if_empty_allowlist else list(patches or [])
    kept: list[dict[str, Any]] = []
    for p in patches or []:
        if not isinstance(p, dict):
            continue
        key = patch_file_key(p)
        if not key:
            continue
        if key in allowed or any(key.endswith("/" + a) or a.endswith("/" + key) for a in allowed):
            kept.append(p)
    return kept


def allowlist_from_patches(patches: list[dict[str, Any]] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for p in patches or []:
        if not isinstance(p, dict):
            continue
        k = patch_file_key(p)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out


def resolve_patch_path(
    raw: str,
    *,
    hints: list[str] | None = None,
    project_root: Path | None = None,
    project_id: str | None = None,
) -> str:
    """Map package-style or basename paths to repo-relative paths via retrieval hints."""
    p = (raw or "").strip().replace("\\", "/")
    if not p:
        return p
    root = project_root or get_settings().project_root_path
    if (root / p).is_file():
        return p
    hints_n = [h.replace("\\", "/").lstrip("./") for h in (hints or []) if h]
    for h in hints_n:
        if h == p or h.endswith("/" + p) or h.endswith(p):
            return h
    if p.endswith(".java"):
        simple = p.rsplit("/", 1)[-1]
        for h in hints_n:
            if h.endswith("/" + simple) or h.endswith(simple):
                return h
        if project_id:
            try:
                from app.agent.explore_shared import _locate_java_file_for_class

                found = _locate_java_file_for_class(project_id, simple[: -len(".java")])
                if found:
                    return str(found).replace("\\", "/")
            except Exception:
                pass
    return p


def normalize_patch_paths(
    patches: list[dict[str, Any]],
    *,
    hints: list[str] | None = None,
    project_id: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in patches or []:
        if not isinstance(item, dict):
            continue
        q = dict(item)
        key = "file" if q.get("file") else ("path" if q.get("path") else None)
        if key:
            q[key] = resolve_patch_path(
                str(q.get(key) or ""),
                hints=hints,
                project_id=project_id,
            )
        out.append(q)
    return out


def apply_patch_list(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for p in patches:
        if p.get("new_file"):
            results.append(write_new_file(p["file"], p.get("content") or ""))
        else:
            results.append(apply_replacements(p["file"], p.get("replacements") or []))
    return results
