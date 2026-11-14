"""Parse MyBatis Mapper XML into ParsedType / ParsedMethod (kind=MapperXml)."""
from __future__ import annotations

import re
from pathlib import Path

from app.core.utils import module_from_path
from app.parser.models import ParsedAnnotation, ParsedField, ParsedMethod, ParsedType, ParseResult

NAMESPACE_RE = re.compile(
    r"<mapper\b[^>]*\bnamespace\s*=\s*[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
# select / insert / update / delete / sql fragments
STMT_RE = re.compile(
    r"<(?P<tag>select|insert|update|delete|sql)\b(?P<attrs>[^>]*)>(?P<body>.*?)</(?P=tag)\s*>",
    re.IGNORECASE | re.DOTALL,
)
RESULT_MAP_RE = re.compile(
    r"<resultMap\b(?P<attrs>[^>]*)>",
    re.IGNORECASE,
)
ATTR_RE = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*[\"']([^\"']*)[\"']")


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _attrs(attr_text: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in ATTR_RE.finditer(attr_text or "")}


def _expand_xml_comment_prefix(text: str, start: int) -> int:
    """Include immediately preceding <!-- ... --> comments."""
    pos = start
    while pos > 0:
        # skip whitespace
        left = pos
        while left > 0 and text[left - 1] in " \t\r\n":
            left -= 1
        chunk = text[:left].rstrip()
        end_comment = chunk.rfind("-->")
        if end_comment < 0 or not chunk.endswith("-->"):
            break
        begin = chunk.rfind("<!--", 0, end_comment)
        if begin < 0:
            break
        pos = begin
    # trim leading blank lines
    while pos < start and text[pos] in "\r\n":
        pos += 1
    return pos


def parse_mapper_xml_file(file_path: Path, project_root: Path) -> ParseResult:
    result = ParseResult()
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        result.errors.append(f"{file_path}: {e}")
        return result

    ns_m = NAMESPACE_RE.search(text)
    if not ns_m:
        result.errors.append(f"{file_path}: not a MyBatis mapper (no namespace)")
        return result

    namespace = ns_m.group(1).strip()
    simple_name = namespace.split(".")[-1] if namespace else file_path.stem
    module_name = module_from_path(project_root, file_path)
    try:
        rel_path = str(file_path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        rel_path = str(file_path).replace("\\", "/")

    parsed = ParsedType(
        kind="mapper",
        package=".".join(namespace.split(".")[:-1]) if "." in namespace else "",
        simple_name=simple_name,
        fqcn=namespace,
        file_path=rel_path,
        module_name=module_name,
        start_line=_line_of(text, ns_m.start()),
        end_line=_line_of(text, len(text) - 1) if text else None,
        annotations=[ParsedAnnotation(name="MapperXml", values={"namespace": namespace})],
        symbol_kind="MapperXml",
        extra={"raw_code": text},
    )

    for rm in RESULT_MAP_RE.finditer(text):
        attrs = _attrs(rm.group("attrs"))
        rid = attrs.get("id")
        if rid:
            parsed.fields.append(
                ParsedField(
                    name=rid,
                    type_name=attrs.get("type") or "resultMap",
                    annotations=[ParsedAnnotation(name="resultMap", values=attrs)],
                )
            )

    for sm in STMT_RE.finditer(text):
        tag = sm.group("tag").lower()
        attrs = _attrs(sm.group("attrs"))
        stmt_id = attrs.get("id")
        if not stmt_id:
            continue
        full_start = _expand_xml_comment_prefix(text, sm.start())
        raw = text[full_start : sm.end()].strip("\n")
        sig = f"{namespace}#{stmt_id}"
        ann = [
            ParsedAnnotation(name=tag, values={k: v for k, v in attrs.items() if k != "id"}),
        ]
        if tag == "sql":
            ann.append(ParsedAnnotation(name="sqlFragment", values={}))
        parsed.methods.append(
            ParsedMethod(
                name=stmt_id,
                signature=sig,
                start_line=_line_of(text, full_start),
                end_line=_line_of(text, sm.end() - 1),
                annotations=ann,
                modifiers=[tag],
                raw_code=raw,
            )
        )

    result.types.append(parsed)
    return result
