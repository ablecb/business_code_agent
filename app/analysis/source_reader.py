"""On-demand Java method body reader (no full-file AI structural parse).

Used by layer-by-layer flow analysis: given a class + method name, pull the
method source via brace matching so recursion can continue into private helpers
and large ServiceImpl files.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from app.analysis.common import find_files_by_simple_name, guess_role_from_name
from app.core.utils import module_from_path
from app.parser.ai_java_parser import _find_matching_brace, _line_of
from app.parser.models import ParsedField, ParsedMethod, ParsedType

logger = logging.getLogger("app.analysis.source_reader")

_CLASS_RE = re.compile(
    r"^\s*(?:public\s+|protected\s+|private\s+)?(?:abstract\s+|final\s+)?"
    r"(?:class|interface|enum)\s+(\w+)",
    re.M,
)
_PKG_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_FIELD_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|volatile|transient|\s)+"
    r"([\w.<>,\[\]]+)\s+(\w+)\s*[;=]",
    re.M,
)


def read_java_text(project_root: Path, rel_or_abs: str | Path) -> tuple[str, Path, str]:
    root = Path(project_root).resolve()
    path = Path(rel_or_abs)
    if not path.is_absolute():
        path = root / path
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        rel = path.relative_to(root).as_posix()
    except ValueError:
        rel = str(path).replace("\\", "/")
    return text, path, rel


def class_shell_from_text(
    text: str,
    *,
    rel_path: str,
    module_name: str = "",
) -> ParsedType:
    pkg = ""
    m = _PKG_RE.search(text)
    if m:
        pkg = m.group(1)
    simple = Path(rel_path).stem
    cm = _CLASS_RE.search(text)
    if cm:
        simple = cm.group(1)
    fqcn = f"{pkg}.{simple}" if pkg else simple
    fields: list[ParsedField] = []
    for fm in _FIELD_RE.finditer(text[: min(len(text), 50000)]):
        type_name, fname = fm.group(1), fm.group(2)
        if fname in {"serialVersionUID"} or type_name in {"return", "new", "package", "import"}:
            continue
        if len(fields) >= 80:
            break
        fields.append(ParsedField(name=fname, type_name=type_name))
    return ParsedType(
        kind="class",
        package=pkg,
        simple_name=simple,
        fqcn=fqcn,
        file_path=rel_path,
        module_name=module_name,
        start_line=1,
        end_line=_line_of(text, len(text) - 1) if text else 1,
        fields=fields,
        symbol_kind=guess_role_from_name(simple, rel_path),
    )


def extract_method_by_name(text: str, method_name: str) -> ParsedMethod | None:
    """Extract one method body by name (works for private helpers, no annotations required)."""
    if not method_name or method_name in {"if", "for", "while", "switch", "catch", "return"}:
        return None
    # Prefer method declarations: modifiers + return type + name(
    pat = re.compile(
        rf"(?m)^[ \t]*(?:(?:public|private|protected|static|final|synchronized|native|default|abstract)\s+)+"
        rf"(?:[\w.<>,\[\]?]+\s+)+{re.escape(method_name)}\s*\("
    )
    matches = list(pat.finditer(text))
    if not matches:
        # fallback: any name(
        pat2 = re.compile(rf"(?m)^[ \t]*[\w.<>,\[\]\s]+?\b{re.escape(method_name)}\s*\(")
        matches = [
            m
            for m in pat2.finditer(text)
            if not text[max(0, m.start() - 40) : m.start()].strip().endswith(".")
        ]
    if not matches:
        return None

    # Prefer the longest body among candidates (implementation over forward decl)
    best: ParsedMethod | None = None
    best_len = -1
    for m in matches[:8]:
        # include annotations / javadoc above
        line_start = text.rfind("\n", 0, m.start()) + 1
        for _ in range(16):
            prev_nl = text.rfind("\n", 0, max(0, line_start - 1))
            if prev_nl < 0:
                break
            prev = text[prev_nl + 1 : line_start].strip()
            if (
                prev.startswith("@")
                or prev.startswith("*")
                or prev.startswith("/**")
                or prev.startswith("//")
                or prev == ""
            ):
                line_start = prev_nl + 1
                continue
            break
        paren = text.find("(", m.start())
        if paren < 0:
            continue
        # skip to opening brace of method (or ; for abstract)
        i = paren
        depth = 0
        brace = -1
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    j = i + 1
                    while j < len(text) and text[j] in " \t\r\n":
                        j += 1
                    # throws ...
                    while j < len(text) and text[j] not in "{;":
                        j += 1
                    if j < len(text) and text[j] == "{":
                        brace = j
                    break
            i += 1
        if brace < 0:
            continue
        end = _find_matching_brace(text, brace)
        raw = text[line_start : end + 1]
        if len(raw) < best_len:
            continue
        start_line = _line_of(text, line_start)
        end_line = _line_of(text, end)
        # Canonical signature aligned with AI parser: fqcn#name(params)
        sig_end = text.find(")", paren)
        param_s = text[paren + 1 : sig_end] if sig_end > paren else ""
        param_s = re.sub(r"\s+", " ", param_s).strip()
        # drop annotations in params for stability
        param_s = re.sub(r"@\w+(?:\([^)]*\))?\s*", "", param_s).strip()
        # keep type tokens roughly (strip param names when possible)
        parts = []
        for p in param_s.split(",") if param_s else []:
            p = p.strip()
            if not p:
                continue
            toks = p.split()
            parts.append(toks[0] if toks else p)
        fqcn = ""  # filled by caller via type; fallback name-only
        # load_method sets type.fqcn — signature finalized after ParsedType known
        sig = f"{method_name}({', '.join(parts)})"
        best = ParsedMethod(
            name=method_name,
            signature=sig[:500],
            start_line=start_line,
            end_line=end_line,
            raw_code=raw[:50000],
            calls=_heuristic_calls(raw),
        )
        best_len = len(raw)
    return best


def _heuristic_calls(raw: str) -> list[str]:
    """Rough call list so recursion can proceed before AI next_calls.

    Prefer business calls: skip log/JSON and bean get/set on VO/DTO so that
    later calls like service.loan are not crowded out of the budget.
    """
    body = raw
    out: list[str] = []
    data_obj = re.compile(
        r"(?i)(Vo|Dto|Req|Resp|Entity|DO|PO|Dmo|Map|List|Set|header|request|response|param|wrapper)$"
    )
    data_names = {
        "reqvo",
        "respvo",
        "dmo",
        "map",
        "header",
        "request",
        "response",
        "body",
        "param",
        "params",
        "log",
    }
    noise_meth = {
        "forEach",
        "stream",
        "equals",
        "hashCode",
        "toString",
        "getClass",
        "info",
        "debug",
        "warn",
        "error",
        "trace",
        "put",
        "format",
        "trim",
        "valueOf",
        "toJSONString",
        "toJsonStr",
        "builder",
        "build",
    }
    for m in re.finditer(r"\b([A-Za-z_][\w]*)\.([A-Za-z_][\w]*)\s*\(", body):
        obj, meth = m.group(1), m.group(2)
        if obj in {"if", "for", "while", "switch", "catch", "return", "new"}:
            continue
        if meth in noise_meth:
            continue
        if re.match(r"^(get|set)[A-Z]", meth):
            if obj.lower() in data_names or data_obj.search(obj):
                continue
        call = f"{obj}.{meth}"
        if call not in out:
            out.append(call)
        if len(out) >= 60:
            break
    return out


def merge_method(t: ParsedType, method: ParsedMethod) -> ParsedMethod:
    for i, existing in enumerate(t.methods):
        if existing.name == method.name and (
            existing.signature == method.signature or not existing.raw_code
        ):
            t.methods[i] = method
            return method
    t.methods.append(method)
    return method


def _resolve_module_name(
    project_root: Path,
    abs_or_rel_path: Path,
    *,
    explicit: str = "",
) -> str:
    """Derive Maven/Gradle module (e.g. loan-service) from path; never leave blank."""
    if (explicit or "").strip():
        return explicit.strip()
    return module_from_path(Path(project_root).resolve(), Path(abs_or_rel_path).resolve())


def load_type_shell(
    project_root: Path,
    rel_path: str,
    *,
    module_name: str = "",
) -> ParsedType:
    text, abs_path, rel = read_java_text(project_root, rel_path)
    mod = _resolve_module_name(project_root, abs_path, explicit=module_name)
    return class_shell_from_text(text, rel_path=rel, module_name=mod)


def load_method(
    project_root: Path,
    rel_path: str,
    method_name: str,
    *,
    type_cache: dict[str, ParsedType] | None = None,
) -> tuple[ParsedType, ParsedMethod] | None:
    """Load class shell + one method body from disk."""
    text, abs_path, rel = read_java_text(project_root, rel_path)
    mod = _resolve_module_name(project_root, abs_path)
    shell = class_shell_from_text(text, rel_path=rel, module_name=mod)
    key = shell.simple_name.lower()
    if type_cache is not None and key in type_cache:
        t = type_cache[key]
        if not (t.module_name or "").strip():
            t.module_name = mod
    else:
        t = shell
        if type_cache is not None:
            type_cache[key] = t
    method = extract_method_by_name(text, method_name)
    if not method:
        logger.info("method not found in source %s#%s", rel, method_name)
        return None
    # Align with AI/discover: fqcn#name(params)
    if "(" in (method.signature or ""):
        params = method.signature[method.signature.find("(") :]
    else:
        params = "()"
    method.signature = f"{t.fqcn}#{method.name}{params}"[:500]
    merge_method(t, method)
    return t, method


def resolve_type_file(
    index: list[dict[str, str]],
    simple: str,
    *,
    prefer_impl: bool = True,
) -> dict[str, str] | None:
    simple = (simple or "").split(".")[-1]
    if not simple:
        return None
    names = [simple]
    if prefer_impl and simple.endswith("Service") and not simple.endswith("Impl"):
        names = [f"{simple}Impl", simple]
    elif prefer_impl and not simple.endswith("Impl"):
        names = [simple, f"{simple}Impl"]
    for name in names:
        hits = find_files_by_simple_name(index, name)
        if hits:
            # prefer exact simple_name
            exact = [h for h in hits if h["simple_name"].lower() == name.lower()]
            return (exact or hits)[0]
    return None

