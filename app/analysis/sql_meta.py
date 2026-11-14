"""Deterministic SQL metadata extraction from MyBatis Mapper XML bodies.

Turns raw SQL fragments into structured facts:
  - tables touched (INSERT INTO / UPDATE / DELETE FROM / SELECT FROM / JOIN)
  - columns written (INSERT(cols) / SET col = …)
  - columns read (WHERE / ON / SELECT projection)
  - parameter bindings ( #{…} / ${…} )

This gives change-plan and field-evidence answers a *hard evidence* layer that
does not depend on the LLM deep-analysis agreeing on the same table/column names.
"""
from __future__ import annotations

import re
from typing import Any

# Strip SQL comments (-- and /* */) so column regexes don't match comment text.
_SQL_COMMENT_RE = re.compile(r"/\*.*?\*/|--[^\n]*", re.DOTALL)
# Table references: INSERT INTO t / UPDATE t / DELETE FROM t / FROM t / JOIN t
_TABLE_RE = re.compile(
    r"(?is)\b(?:from|into|update|join)\s+"
    r"(`?[A-Za-z_][\w]*`?|[\u4e00-\u9fff_A-Za-z0-9_.]+)"
)
# INSERT INTO t (a, b, c)
_INSERT_COLS_RE = re.compile(r"(?is)\binsert\s+into\s+\S+\s*\(([^)]*)\)")
# SET col = value (may repeat, comma-separated)
_SET_COL_RE = re.compile(r"(?is)\bset\s+(.*?)(?:\bwhere\b|\breturning\b|;|$)")
_SET_ITEM_RE = re.compile(r"(?is)(`?[A-Za-z_][\w]*`?)\s*=")
# SELECT projection columns: SELECT a, b FROM
_SELECT_COLS_RE = re.compile(
    r"(?is)\bselect\s+(.*?)\bfrom\b", re.DOTALL
)
_PROJECTION_ITEM_RE = re.compile(
    r"(?is)(?:^|,)\s*(?:[A-Za-z_][\w]*\.)?(`?[A-Za-z_][\w]*`?)(?:\s+as\s+\w+)?(?=\s*(?:,|$))"
)
# WHERE col = …
_WHERE_COL_RE = re.compile(r"(?is)\bwhere\b(.*?)(?:\bgroup\s+by\b|\border\s+by\b|\blimit\b|;|$)")
_WHERE_ITEM_RE = re.compile(
    r"(?is)(?:^|\s)(?:[A-Za-z_][\w]*\.)?(`?[A-Za-z_][\w]*`?)\s*(?:=|<>|!=|>|<|>=|<=|like|in|between)"
)
# Parameter bindings #{…} / ${…}
_PARAM_RE = re.compile(r"[#$]\{([^}]+)\}")


def _strip_comments(sql: str) -> str:
    return _SQL_COMMENT_RE.sub(" ", sql or "")


def _bare(name: str) -> str:
    return (name or "").strip().strip("`").strip()


def _clean_param_ref(raw: str) -> str:
    """#{dealRefNo,jdbcType=VARCHAR} → dealRefNo (drop type annotations)."""
    return (raw or "").split(",")[0].split(".")[0].strip()


def _clean_col(raw: str) -> str:
    """Column-like token cleanup: strip jdbcType decorations & quoted keys."""
    t = _bare(raw)
    # SET col = #{x,jdbcType=VARCHAR} → param only, not a column
    if t.lower().startswith("jdbctype") or t.lower() in {"", "values", "value"}:
        return ""
    return t


def _split_top_level(text: str) -> list[str]:
    """Split on commas that are not inside parens (column lists, SET items)."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text or "":
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _clean_projection(text: str) -> list[str]:
    """Extract column names from a SELECT projection, handling count(*)/distinct."""
    cols: list[str] = []
    for item in _split_top_level(text or ""):
        item = item.strip()
        if not item:
            continue
        item = re.sub(r"(?is)^\s*distinct\s+", "", item)
        m = re.search(r"(?is)(`?[A-Za-z_][\w]*`?)\s*$", item)
        if not m:
            continue
        name = _bare(m.group(1))
        if name and name not in {"*", "count"}:
            cols.append(name)
    return cols


def parse_sql_meta(sql: str, *, stmt_tag: str = "select") -> dict[str, Any]:
    """Parse one SQL statement body → structured metadata.

    Returns: {tables, write_cols (by table), read_cols, params, op}
    """
    clean = _strip_comments(sql)
    op = stmt_tag.lower()
    tables: list[str] = []
    seen_tables: set[str] = set()
    for m in _TABLE_RE.finditer(clean):
        t = _bare(m.group(1))
        if t and t not in seen_tables and not t.lower().startswith("select"):
            seen_tables.add(t)
            tables.append(t)

    write_cols: list[str] = []
    if op == "insert":
        m = _INSERT_COLS_RE.search(clean)
        if m:
            write_cols = [_clean_col(x) for x in _split_top_level(m.group(1))]
    elif op == "update":
        sm = _SET_COL_RE.search(clean)
        if sm:
            write_cols = [
                _clean_col(m2.group(1)) for m2 in _SET_ITEM_RE.finditer(sm.group(1))
            ]

    read_cols: list[str] = []
    if op == "select":
        sm = _SELECT_COLS_RE.search(clean)
        if sm:
            read_cols = _clean_projection(sm.group(1))
    where_m = _WHERE_COL_RE.search(clean)
    if where_m:
        read_cols.extend(
            _bare(m2.group(1)) for m2 in _WHERE_ITEM_RE.finditer(where_m.group(1))
        )

    params = sorted(
        {
            _clean_param_ref(m.group(1))
            for m in _PARAM_RE.finditer(clean)
        }
    )

    return {
        "op": op,
        "tables": list(dict.fromkeys(tables)),
        "write_cols": [c for c in dict.fromkeys(write_cols) if c],
        "read_cols": list(dict.fromkeys(read_cols)),
        "params": [p for p in params if p and p not in {"_parameter"}],
    }


def sql_meta_to_table_changes(
    meta: dict[str, Any],
    *,
    class_method: str,
) -> list[dict[str, str]]:
    """Convert SQL metadata to table_field_changes-shaped entries.

    Each write column on each table becomes one change row so field-evidence
    and change-plan rendering can cite deterministic SQL facts.
    """
    out: list[dict[str, str]] = []
    tables = meta.get("tables") or []
    for t in tables:
        for col in meta.get("write_cols") or []:
            out.append(
                {
                    "table_or_entity": t,
                    "field": col,
                    "from_value": "",
                    "to_value": "",
                    "op": str(meta.get("op") or "write"),
                    "when": f"{class_method} Mapper SQL",
                    "source": "mapper_xml",
                }
            )
    return out[:50]
