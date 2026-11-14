"""Javadoc / line-comment business description extraction.

Pulls the comment block immediately preceding a method declaration from the
tree-sitter CST and turns it into a professional business summary:

  - ``/*** ... */`` Javadoc: body lines + @param / @return annotations
  - ``// ...`` line comments directly above the method
  - cleans leading ``*`` markers, joins multi-line text, strips tags

This gives heuristic summaries a real business sentence instead of the
placeholder "{Class}#{method} 业务节点".
"""
from __future__ import annotations

import re
from typing import Any

_JAVADOC_TAG_RE = re.compile(r"^\s*@(\w+)\s*(.*)$")
_LEADING_STAR_RE = re.compile(r"^\s*\*\s?")
# Structural lines that carry no business meaning for a summary.
_NOISE_TAGS = frozenset({"param", "return", "throws", "author", "date", "since", "version", "deprecated", "see", "link"})


def extract_preceding_comment(source: bytes, method_node: Any, nodes: list[Any] | None = None) -> str:
    """Return the comment text (raw, unprocessed) before a method declaration.

    ``nodes`` may be the pre-computed _walk list to avoid re-walking.
    """
    if method_node is None:
        return ""
    walk = nodes if nodes is not None else _walk_all(source)
    idx = -1
    for i, n in enumerate(walk):
        if n is method_node:
            idx = i
            break
    if idx <= 0:
        return ""
    prev = walk[idx - 1]
    if prev.type in {"block_comment", "line_comment"}:
        # ensure the comment is truly adjacent (only whitespace between)
        if prev.end_point[0] + 1 >= method_node.start_point[0] - 1:
            return _text_of(source, prev)
    return ""


def _walk_all(source: bytes) -> list[Any]:
    from app.analysis.global_ast import _parser, _walk

    return list(_walk(_parser().parse(source).root_node))


def _text_of(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")


def _clean_line(s: str) -> str:
    """Strip one comment line: leading * / delimiters, trailing */."""
    s = (s or "").strip()
    if not s:
        return ""
    # /** or /* or */
    if s.startswith("/**"):
        s = s[3:]
    elif s.startswith("/*"):
        s = s[2:]
    if s.endswith("*/"):
        s = s[:-2]
    # leading star (Javadoc convention: " * text")
    s = re.sub(r"^\s*\*\s?", "", s)
    return s.strip()


def clean_comment(raw: str) -> str:
    """Strip comment delimiters / leading stars → readable text.

    ``@param``-style tag lines are dropped (documentation structure, not prose).
    """
    lines: list[str] = []
    for ln in (raw or "").splitlines():
        s = _clean_line(ln)
        if not s or s.startswith("@"):
            continue
        lines.append(s)
    return " ".join(lines).strip()


def split_javadoc(raw: str) -> tuple[list[str], dict[str, str]]:
    """Split Javadoc into (body_lines, tags: {tag: value})."""
    body: list[str] = []
    tags: dict[str, str] = {}
    for ln in (raw or "").splitlines():
        s = _clean_line(ln)
        if not s:
            continue
        m = _JAVADOC_TAG_RE.match(s)
        if m:
            tags.setdefault(m.group(1), m.group(2).strip())
            continue
        body.append(s)
    return body, tags


def business_summary_from_comment(raw: str, *, fallback: str = "") -> str:
    """Best-effort business sentence from a Javadoc/comment block.

    Prefers non-@param/@return body lines; falls back to the first tag value.
    """
    body, tags = split_javadoc(raw)
    meaningful = [b for b in body if not any(t in b for t in ("@",))]
    if not meaningful:
        for tag, val in tags.items():
            if tag not in _NOISE_TAGS and val:
                return val[:80]
        # fallback: any non-noise tag value
        for tag, val in tags.items():
            if val:
                return val[:80]
    text = " ".join(meaningful).strip()
    return text[:120] or fallback


def param_flow_from_javadoc(raw: str) -> list[dict[str, str]]:
    """Extract @param entries → [{name, desc}] (variable flow documentation)."""
    _body, tags = split_javadoc(raw)
    out: list[dict[str, str]] = []
    for val in tags.get("param", "").splitlines():
        val = val.strip()
        if not val:
            continue
        parts = val.split(None, 1)
        if parts:
            out.append({"name": parts[0], "desc": parts[1] if len(parts) > 1 else ""})
    return out[:12]
