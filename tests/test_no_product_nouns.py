"""Production agent/analysis code must not hardcode product-domain nouns.

The intent: routing must be LLM + evidence, not product-noun keyword matching.
This scans CODE ONLY — comments, docstrings and prompt strings are documentation /
few-shot examples, not routing logic, so they are skipped. Offline deterministic
golden-QA generators (interface_tables / table_dict) are ingest-time business
label tables and never participate in runtime Q&A routing, so they are exempt.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "app"

# Industry/product nouns for this loan/credit codebase — routing must be LLM + evidence.
_BANNED = (
    "放款",
    "提现",
    "授信",
    "用信",
    "额度",
    "融担",
    "机构限额",
    "门店限额",
    "优惠券",
    "贴息",
    "会员价格",
    "二级门店",
    "风控",
    "担保",
)

# Offline deterministic golden-QA generators: their domain keyword tables are
# ingest-time business labels (generate_golden_qa), never runtime Q&A routing.
_EXEMPT_FILES = frozenset(
    {
        "app/analysis/interface_tables.py",
        "app/analysis/table_dict.py",
    }
)

# Module-level constants whose string values are prompts / few-shot examples
# (e.g. INTENT_PROMPT) — they teach the LLM, they do not route by noun.
_PROMPT_CONST_RE = frozenset(
    {"PROMPT", "INTENT", "SYSTEM", "GOAL"}
)


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that serve as docstrings (module/class/function)."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def _prompt_const_strings(tree: ast.Module) -> list[str]:
    """String values assigned to module-level *PROMPT*/*INTENT*/*SYSTEM* consts."""
    out: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name) and any(
                k in tgt.id.upper() for k in _PROMPT_CONST_RE
            ):
                if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                    out.append(node.value.value)
                elif isinstance(node.value, (ast.Tuple, ast.List)):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            out.append(elt.value)
    return out


def _code_string_values(tree: ast.Module) -> list[str]:
    """String constants inside code (skips docstrings & prompt consts)."""
    doc_ids = _docstring_nodes(tree)
    prompt_vals = set(_prompt_const_strings(tree))
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in doc_ids:
                continue  # docstring body
            if node.value in prompt_vals:
                continue  # prompt / few-shot example text
            out.append(node.value)
    return out


def test_app_python_has_no_product_domain_nouns():
    hits: list[str] = []
    for path in ROOT.rglob("*.py"):
        rel = path.relative_to(ROOT.parent).as_posix()
        if rel in _EXEMPT_FILES:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for value in _code_string_values(tree):
            for noun in _BANNED:
                if noun in value:
                    hits.append(f"{rel}: {noun}")
    assert not hits, "product-domain nouns hardcoded in routing/code strings:\n" + "\n".join(hits)
