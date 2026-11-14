"""Global Tree-sitter AST for a Java project — parse each file once.

Built once per parse-project / pipeline job and reused for:
  - class / method / field extraction
  - entity & DB mapping (DO/Entity/Mapper/SQL)
  - directed call-graph edges (incl. ``this.field.method``)

Do **not** re-parse per method during AI analysis.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from app.parser.models import (
    ParsedAnnotation,
    ParsedField,
    ParsedMethod,
    ParsedType,
)
from app.parser.scanner import scan_java_files, scan_mapper_xml_files

logger = logging.getLogger("app.analysis.global_ast")

try:
    from tree_sitter import Language, Parser
    import tree_sitter_java as tsjava
except ModuleNotFoundError as e:  # pragma: no cover
    Language = None  # type: ignore
    Parser = None  # type: ignore
    tsjava = None  # type: ignore
    _TS_IMPORT_ERROR = e
else:
    _TS_IMPORT_ERROR = None

_SKIP_RECV = frozenset(
    {
        "log",
        "logger",
        "System",
        "Objects",
        "Collections",
        "Arrays",
        "Optional",
        "String",
        "Integer",
        "Long",
        "Boolean",
        "BigDecimal",
        "JSON",
        "StringUtil",
        "BeanUtils",
        "ResponseBuilder",
        "FieldNameConstants",
        "ExceptionEnum",
        "map",
        "reqMap",
        "header",
        "request",
        "lock",
        "e",
    }
)
_SKIP_METHOD = frozenset(
    {
        "info",
        "error",
        "debug",
        "warn",
        "toJSONString",
        "getBody",
        "getHeader",
        "put",
        "get",
        "set",
        "equals",
        "hashCode",
        "toString",
        "builder",
        "build",
        "isNull",
        "tryLock",
        "getLock",
    }
)

_ENTITY_SUFFIX = ("Entity", "DO", "PO", "Dmo", "Dao")
_DTO_SUFFIX = (
    "DTO",
    "Dto",
    "VO",
    "Vo",
    "Req",
    "Resp",
    "Request",
    "Response",
    "Body",
    "Param",
    "Model",
    "Bean",
)


def _require_ts() -> None:
    if _TS_IMPORT_ERROR is not None:
        raise RuntimeError(
            f"tree-sitter not installed: {_TS_IMPORT_ERROR}\n"
            "pip install tree-sitter tree-sitter-java"
        )


def _parser() -> Any:
    _require_ts()
    return Parser(Language(tsjava.language()))


def _text(source: bytes, node: Any) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _preceding_comment_text(
    source: bytes, walk: list[Any], method_node: Any, *, max_gap_lines: int = 1
) -> str:
    """Comment block (Javadoc/line comment) immediately above a method.

    ``walk`` is the pre-order node list (e.g. of the enclosing type). Only
    comments within ``max_gap_lines`` lines above the method count — we do not
    want to grab unrelated comments separated by a blank gap + annotation.
    """
    try:
        idx = next(i for i, n in enumerate(walk) if n is method_node)
    except StopIteration:
        return ""
    if idx <= 0:
        return ""
    prev = walk[idx - 1]
    if prev.type not in {"block_comment", "line_comment"}:
        return ""
    gap = method_node.start_point[0] - prev.end_point[0] - 1
    if gap > max_gap_lines:
        return ""
    return _text(source, prev).strip()


def _line(node: Any) -> int:
    return int(node.start_point[0]) + 1


def _end_line(node: Any) -> int:
    return int(node.end_point[0]) + 1


def _walk(node: Any) -> Iterator[Any]:
    """Iterative tree walk (explicit stack) — avoids RecursionError / deep
    recursion stalls on pathological tree-sitter trees (e.g. huge generated
    classes), and is safe for the O(n) scans in this module."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        # Preserve original left-to-right order by reversing children.
        stack.extend(reversed(list(n.children)))


def _annotation_name(source: bytes, ann_node: Any) -> str:
    for c in ann_node.children:
        if c.type in {"identifier", "scoped_type_identifier", "type_identifier"}:
            return _text(source, c).split(".")[-1]
    return _text(source, ann_node).lstrip("@").split("(")[0].strip()


def _annotation_args(source: bytes, ann_node: Any) -> dict[str, str]:
    raw = _text(source, ann_node)
    out: dict[str, str] = {}
    for m in re.finditer(
        r'(?:(\w+)\s*=\s*)?["\']([^"\']+)["\']',
        raw,
    ):
        key = m.group(1) or "value"
        out[key] = m.group(2)
    return out


def _collect_annotations(source: bytes, mods_node: Any | None) -> list[ParsedAnnotation]:
    if mods_node is None:
        return []
    anns: list[ParsedAnnotation] = []
    for m in mods_node.children:
        if "annotation" in m.type:
            anns.append(
                ParsedAnnotation(
                    name=_annotation_name(source, m),
                    values=_annotation_args(source, m),
                )
            )
    return anns


def _method_name(source: bytes, method_node: Any) -> str | None:
    for c in method_node.children:
        if c.type == "identifier":
            return _text(source, c)
    return None


def _invocation_receiver_node(inv_node: Any) -> Any | None:
    arg_idx = next(
        (i for i, c in enumerate(inv_node.children) if c.type == "argument_list"), None
    )
    before = inv_node.children[:arg_idx] if arg_idx is not None else inv_node.children
    objs = [
        c
        for c in before
        if c.type
        in {
            "identifier",
            "field_access",
            "this",
            "method_invocation",
            "object_creation_expression",
            "parenthesized_expression",
        }
    ]
    if len(objs) >= 2:
        return objs[0]
    if len(objs) == 1 and objs[0].type != "identifier":
        return objs[0]
    return None


def _is_fluent_or_builder_call(inv_node: Any) -> bool:
    recv = _invocation_receiver_node(inv_node)
    if recv is None:
        return False
    return recv.type in {"method_invocation", "object_creation_expression"}


def _method_calls_in_body(source: bytes, method_node: Any) -> list[str]:
    body = next((c for c in method_node.children if c.type == "block"), None)
    if body is None:
        return []
    calls: list[str] = []
    seen: set[str] = set()
    for n in _walk(body):
        if n.type != "method_invocation":
            # Method references: OrderItem::getCode → "OrderItem.getCode"
            if n.type == "method_reference":
                ref = _text(source, n)
                key = ref.replace("::", ".").strip()
                if key and key not in seen and not key.startswith("this."):
                    seen.add(key)
                    calls.append(key)
            continue
        if _is_fluent_or_builder_call(n):
            # Fluent chains (items.stream().map(...)) are skipped as call edges —
            # they carry no new local receiver info; the root invocation
            # (items.stream) is still visited as its own node further down.
            continue
        arg_idx = next(
            (i for i, c in enumerate(n.children) if c.type == "argument_list"), None
        )
        before = n.children[:arg_idx] if arg_idx is not None else n.children
        idents = [c for c in before if c.type in {"identifier", "field_access", "this"}]
        if not idents:
            continue
        if len(idents) == 1:
            key = _text(source, idents[0])
            name = key
        else:
            recv = _text(source, idents[0])
            name = _text(source, idents[-1])
            key = f"{recv}.{name}"
        if name in _SKIP_METHOD or key.startswith("log."):
            continue
        if key not in seen:
            seen.add(key)
            calls.append(key)
    return calls


def _class_fields(source: bytes, type_node: Any) -> dict[str, str]:
    fields: dict[str, str] = {}
    for n in _walk(type_node):
        if n.type != "field_declaration":
            continue
        type_name = None
        for c in n.children:
            if c.type in {"type_identifier", "generic_type", "scoped_type_identifier"}:
                type_name = _text(source, c).split("<", 1)[0].split(".")[-1].strip()
            if c.type == "variable_declarator":
                for x in c.children:
                    if x.type == "identifier" and type_name:
                        fields[_text(source, x)] = type_name
    return fields


def _return_type(source: bytes, method_node: Any) -> str:
    for c in method_node.children:
        if c.type in {
            "type_identifier",
            "generic_type",
            "void_type",
            "integral_type",
            "floating_point_type",
            "boolean_type",
            "scoped_type_identifier",
            "array_type",
        }:
            return _text(source, c).split("<", 1)[0].strip()
    return "void"


def _formal_params(source: bytes, method_node: Any) -> list[str]:
    params: list[str] = []
    for c in method_node.children:
        if c.type != "formal_parameters":
            continue
        for p in c.children:
            if p.type != "formal_parameter":
                continue
            tname = ""
            for x in p.children:
                if x.type in {
                    "type_identifier",
                    "generic_type",
                    "scoped_type_identifier",
                    "array_type",
                }:
                    tname = _text(source, x).split("<", 1)[0].strip()
            if tname:
                params.append(tname.split(".")[-1])
    return params


def _guess_symbol_kind(
    simple: str,
    file_path: str,
    annotations: list[ParsedAnnotation],
    *,
    is_interface: bool = False,
) -> str:
    ann = {a.name for a in annotations}
    path_l = file_path.replace("\\", "/").lower()
    if "RestController" in ann or "Controller" in ann or simple.endswith("Controller"):
        return "Controller"
    if "FeignClient" in ann:
        return "Feign"
    if any(x in ann for x in ("RabbitListener", "KafkaListener", "RocketMQMessageListener")):
        return "Listener"
    if simple.endswith("Listener") or "/listener/" in path_l:
        return "Listener"
    if "XxlJob" in ann or simple.endswith("Job") or "/xxljob/" in path_l:
        return "Job"
    if simple.endswith("Mapper") or "/mapper/" in path_l:
        return "Mapper"
    if any(simple.endswith(s) for s in _ENTITY_SUFFIX) or "/entity/" in path_l or "/domain/" in path_l:
        return "Entity"
    if any(simple.endswith(s) for s in _DTO_SUFFIX) or "/dto/" in path_l or "/vo/" in path_l:
        return "DTO"
    if "Service" in ann or simple.endswith("Service") or simple.endswith("ServiceImpl"):
        return "Service"
    if is_interface and simple.endswith("Service"):
        return "Service"
    if "Util" in simple or "Helper" in simple or "Common" in simple or "/util/" in path_l:
        return "Util"
    return "Class"


def _module_from_path(project_root: Path, file_path: Path) -> str:
    try:
        rel = file_path.resolve().relative_to(project_root.resolve())
    except ValueError:
        return ""
    parts = rel.parts
    if parts:
        return parts[0]
    return ""


@dataclass
class AstMethod:
    """One method extracted from the global AST (no re-parse needed)."""

    class_method: str
    fqcn: str
    simple_name: str
    method_name: str
    signature: str
    file_path: str
    module_name: str
    symbol_kind: str
    start_line: int | None
    end_line: int | None
    raw_code: str
    calls: list[str] = field(default_factory=list)
    annotations: list[ParsedAnnotation] = field(default_factory=list)
    parameter_types: list[str] = field(default_factory=list)
    return_type: str = "void"
    has_body: bool = True
    # Resolved local callees: Class#method
    resolved_callees: list[str] = field(default_factory=list)
    # Javadoc / line comment directly above the method (raw, unprocessed).
    doc_comment: str = ""


@dataclass
class AstType:
    fqcn: str
    simple_name: str
    package: str
    kind: str  # class/interface/enum/mapper
    symbol_kind: str
    file_path: str
    module_name: str
    start_line: int | None
    end_line: int | None
    annotations: list[ParsedAnnotation] = field(default_factory=list)
    fields: list[ParsedField] = field(default_factory=list)
    field_map: dict[str, str] = field(default_factory=dict)
    methods: list[AstMethod] = field(default_factory=list)
    # Mapper XML extras
    table_hints: list[str] = field(default_factory=list)
    sql_ids: list[str] = field(default_factory=list)
    # Deterministic SQL facts: {table: [columns written]} from Mapper XML bodies.
    sql_table_fields: dict[str, list[str]] = field(default_factory=dict)
    # Per-method SQL metadata (op / tables / write_cols / read_cols / params).
    method_sql_meta: dict[str, dict[str, Any]] = field(default_factory=dict)
    source_bytes: bytes | None = None


@dataclass
class CallEdge:
    caller: str  # Class#method
    callee: str  # Class#method
    raw_call: str
    status: str = "ok_local"  # ok_local | no_local_source | skipped
    # 1-based appearance order in caller method body (AST extraction order).
    order: int = 0
    # Resolution facts (enriched at AST write time for Neo4j CALLS props).
    receiver: str = ""
    receiver_type: str | None = None
    resolve_how: str = "unknown"  # field_map | local_decl | new | cast | param | this | …


@dataclass
class ScopedName:
    """A name in method/class scope with a resolved (or unknown) type."""

    name: str
    type_name: str | None
    how: str  # field_map | param | local_decl | new | cast | foreach | unknown
    # Generic type arguments for the declared type (e.g. List<OrderItem> → ["OrderItem"]).
    type_args: list[str] = field(default_factory=list)
    # When the variable is reassigned to a more concrete type, latest known type.
    reassigned_to: str | None = None


@dataclass
class GlobalAst:
    """In-memory project AST + call graph for one parse job."""

    project_id: str
    project_root: Path
    types: dict[str, AstType] = field(default_factory=dict)  # fqcn -> type
    by_simple: dict[str, list[str]] = field(default_factory=dict)  # simple -> [fqcn]
    methods: dict[str, AstMethod] = field(default_factory=dict)  # Class#method -> method
    edges: list[CallEdge] = field(default_factory=list)
    # caller -> [callee Class#method]
    callees_of: dict[str, list[str]] = field(default_factory=dict)
    # callee -> [caller Class#method]
    callers_of: dict[str, list[str]] = field(default_factory=dict)
    mapper_xml_count: int = 0
    java_file_count: int = 0
    errors: list[str] = field(default_factory=list)
    package_roots: list[str] = field(default_factory=list)
    # method simple-name → return type simple-name (for var inference).
    method_returns: dict[str, str] = field(default_factory=dict)
    # receiver simple-name → generic type args across the project (lambda typing).
    generic_receivers: dict[str, list[str]] = field(default_factory=dict)
    # type simple-name → field generic args (built once, reused across lookups).
    _field_args_cache: dict[str, list[str]] = field(default_factory=dict)

    def returns_for(self) -> dict[str, str]:
        """Method simple-name → return simple type (project-wide, Impl preferred)."""
        if self.method_returns:
            return self.method_returns
        for m in self.methods.values():
            if not m.method_name or m.return_type in {"void", "None", "null", ""}:
                continue
            cur = self.method_returns.get(m.method_name)
            if cur is None or (m.simple_name.endswith("Impl") and cur is not None):
                self.method_returns[m.method_name] = _simple_type_name(m.return_type)
        return self.method_returns

    def _field_args_for_type(self, t: AstType) -> list[str]:
        """Field generic args of one type — parsed once and cached forever."""
        key = t.simple_name or t.fqcn
        if key not in self._field_args_cache:
            args = _class_field_generic_args(t.source_bytes) if t.source_bytes else []
            self._field_args_cache[key] = list(args)
            if args:
                # Still mirror into generic_receivers for direct name hits.
                self.generic_receivers.setdefault(key, list(args))
        return self._field_args_cache.get(key, [])

    def generic_args_for(self, simple_name: str) -> list[str]:
        """Generic type args of a receiver (e.g. List<OrderItem> → [OrderItem]).

        Resolves by field name first (``items.forEach`` where the class declares
        ``private List<OrderItem> items;``), then by type simple name.

        Results (including "not found") are cached: previously a miss re-scanned
        every class source and re-parsed each one — O(methods × receivers ×
        types × parse), which stalled var-resolve for hours on large projects.
        """
        if simple_name in self.generic_receivers:
            return self.generic_receivers[simple_name]
        for t in self.types.values():
            args = self._field_args_for_type(t)
            if t.simple_name == simple_name:
                result = list(args)
                self.generic_receivers[simple_name] = result
                return result
            if simple_name in t.field_map:
                result = list(args)
                self.generic_receivers[simple_name] = result
                return result
        # Negative cache: "not found" is also memoized so the full scan
        # (which parses every class once) runs at most a single time.
        self.generic_receivers[simple_name] = []
        return []

    def to_parsed_types(self) -> list[ParsedType]:
        out: list[ParsedType] = []
        for t in self.types.values():
            methods = [
                ParsedMethod(
                    name=m.method_name,
                    signature=m.signature,
                    start_line=m.start_line,
                    end_line=m.end_line,
                    annotations=list(m.annotations),
                    calls=list(m.calls),
                    raw_code=m.raw_code,
                    doc_comment=m.doc_comment,
                )
                for m in t.methods
            ]
            out.append(
                ParsedType(
                    kind=t.kind,
                    package=t.package,
                    simple_name=t.simple_name,
                    fqcn=t.fqcn,
                    file_path=t.file_path,
                    module_name=t.module_name,
                    start_line=t.start_line,
                    end_line=t.end_line,
                    annotations=list(t.annotations),
                    fields=list(t.fields),
                    methods=methods,
                    symbol_kind=t.symbol_kind,
                )
            )
        return out

    def entity_fqcns(self) -> list[str]:
        return [
            fqcn
            for fqcn, t in self.types.items()
            if t.symbol_kind in {"Entity", "DTO", "Mapper", "MapperXml"}
            or any(t.simple_name.endswith(s) for s in _ENTITY_SUFFIX)
        ]

    def method_nodes(self) -> list[AstMethod]:
        return list(self.methods.values())

    def stats(self) -> dict[str, int]:
        return {
            "java_files": self.java_file_count,
            "mapper_xml": self.mapper_xml_count,
            "types": len(self.types),
            "methods": len(self.methods),
            "call_edges": len(self.edges),
            "errors": len(self.errors),
        }


def _parse_java_source(
    project_root: Path,
    file_path: Path,
    source: bytes,
    tree_root: Any,
) -> list[AstType]:
    package = ""
    for n in _walk(tree_root):
        if n.type == "package_declaration":
            package = _text(source, n).replace("package", "").replace(";", "").strip()
            break

    try:
        rel = str(file_path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        rel = str(file_path).replace("\\", "/")
    module = _module_from_path(project_root, file_path)
    results: list[AstType] = []

    type_kinds = {
        "class_declaration": "class",
        "interface_declaration": "interface",
        "enum_declaration": "enum",
        "record_declaration": "class",
    }
    for n in _walk(tree_root):
        if n.type not in type_kinds:
            continue
        # Prefer top-level / named types; skip anonymous
        name = None
        mods = None
        for c in n.children:
            if c.type == "modifiers":
                mods = c
            if c.type == "identifier" and name is None:
                name = _text(source, c)
        if not name:
            continue
        anns = _collect_annotations(source, mods)
        kind = type_kinds[n.type]
        symbol_kind = _guess_symbol_kind(
            name, rel, anns, is_interface=(kind == "interface")
        )
        field_map = _class_fields(source, n)
        fields = [
            ParsedField(name=fn, type_name=ft) for fn, ft in field_map.items()
        ]
        # Pre-walked node list for comment adjacency lookups.
        type_walk = list(_walk(n))
        fqcn = f"{package}.{name}" if package else name
        declared: set[str] = set()
        methods: list[AstMethod] = []
        for mn in _walk(n):
            if mn.type != "method_declaration":
                continue
            # Only methods whose nearest type is this node — skip nested types' methods
            # by checking parent chain briefly: accept all under this type for simplicity.
            mname = _method_name(source, mn)
            if not mname:
                continue
            declared.add(mname)
            m_mods = None
            for c in mn.children:
                if c.type == "modifiers":
                    m_mods = c
                    break
            m_anns = _collect_annotations(source, m_mods)
            params = _formal_params(source, mn)
            ret = _return_type(source, mn)
            sig = f"{mname}({', '.join(params)})"
            has_body = any(c.type == "block" for c in mn.children)
            calls = _method_calls_in_body(source, mn) if has_body else []
            raw = _text(source, mn)
            doc_comment = _preceding_comment_text(source, type_walk, mn)
            methods.append(
                AstMethod(
                    class_method=f"{name}#{mname}",
                    fqcn=fqcn,
                    simple_name=name,
                    method_name=mname,
                    signature=sig,
                    file_path=rel,
                    module_name=module,
                    symbol_kind=symbol_kind,
                    start_line=_line(mn),
                    end_line=_end_line(mn),
                    raw_code=raw,
                    calls=calls,
                    annotations=m_anns,
                    parameter_types=params,
                    return_type=ret,
                    has_body=has_body,
                    doc_comment=doc_comment,
                )
            )
        results.append(
            AstType(
                fqcn=fqcn,
                simple_name=name,
                package=package,
                kind=kind,
                symbol_kind=symbol_kind,
                file_path=rel,
                module_name=module,
                start_line=_line(n),
                end_line=_end_line(n),
                annotations=anns,
                fields=fields,
                field_map=field_map,
                methods=methods,
                source_bytes=source,
            )
        )
    return results


def _parse_one_java(project_root: Path, file_path: Path) -> tuple[list[AstType], list[str]]:
    errors: list[str] = []
    try:
        source = file_path.read_bytes()
    except OSError as e:
        return [], [f"{file_path}: {e}"]
    try:
        tree = _parser().parse(source)
        types = _parse_java_source(project_root, file_path, source, tree.root_node)
        if tree.root_node.has_error:
            errors.append(f"{file_path}: tree has error nodes (partial parse kept)")
        return types, errors
    except Exception as e:
        return [], [f"{file_path}: {e}"]


def _parse_one_mapper_xml(project_root: Path, file_path: Path) -> tuple[AstType | None, list[str]]:
    errors: list[str] = []
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        return None, [f"{file_path}: {e}"]
    ns_m = re.search(
        r"<mapper\b[^>]*\bnamespace\s*=\s*[\"']([^\"']+)[\"']",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if not ns_m:
        return None, [f"{file_path}: not a MyBatis mapper"]
    namespace = ns_m.group(1).strip()
    simple = namespace.split(".")[-1]
    package = ".".join(namespace.split(".")[:-1]) if "." in namespace else ""
    try:
        rel = str(file_path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except ValueError:
        rel = str(file_path).replace("\\", "/")
    module = _module_from_path(project_root, file_path)
    sql_ids: list[str] = []
    table_hints: list[str] = []
    sql_table_fields: dict[str, list[str]] = {}
    method_sql_meta: dict[str, dict[str, Any]] = {}
    methods: list[AstMethod] = []
    for m in re.finditer(
        r"<(?P<tag>select|insert|update|delete|sql)\b(?P<attrs>[^>]*)>",
        text,
        re.IGNORECASE,
    ):
        attrs = dict(re.findall(r'\b(\w+)\s*=\s*["\']([^"\']*)["\']', m.group("attrs") or ""))
        sid = attrs.get("id") or ""
        if not sid:
            continue
        sql_ids.append(sid)
        tag = m.group("tag").lower()
        # crude table hint
        body_start = m.end()
        body_end = text.lower().find(f"</{tag}", body_start)
        body = text[body_start : body_end if body_end > 0 else body_start + 400]
        for tm in re.finditer(
            r"\b(?:from|into|update|join)\s+[`\"']?([A-Za-z_][\w]*)",
            body,
            re.IGNORECASE,
        ):
            table_hints.append(tm.group(1))
        # Deterministic SQL metadata (columns / tables / params) — hard evidence
        # for change-plan and field-evidence answers.
        sql_meta: dict[str, Any] = {}
        if tag in {"insert", "update", "select", "delete"}:
            try:
                from app.analysis.sql_meta import parse_sql_meta

                sql_meta = parse_sql_meta(body, stmt_tag=tag)
                for t in sql_meta.get("tables") or []:
                    cols = sql_meta.get("write_cols") or []
                    if cols:
                        sql_table_fields.setdefault(t, [])
                        for c in cols:
                            if c not in sql_table_fields[t]:
                                sql_table_fields[t].append(c)
            except Exception:
                logger.debug("mapper sql_meta parse failed %s#%s", simple, sid, exc_info=True)
        if sql_meta:
            method_sql_meta[f"{simple}#{sid}"] = sql_meta
        methods.append(
            AstMethod(
                class_method=f"{simple}#{sid}",
                fqcn=namespace,
                simple_name=simple,
                method_name=sid,
                signature=f"{sid}()",
                file_path=rel,
                module_name=module,
                symbol_kind="MapperXml",
                start_line=text.count("\n", 0, m.start()) + 1,
                end_line=text.count("\n", 0, body_end if body_end > 0 else m.end()) + 1,
                raw_code=body[:2000],
                calls=[],
                annotations=[ParsedAnnotation(name=tag.upper(), values=attrs)],
                has_body=True,
            )
        )
    # resultMap type hints
    for rm in re.finditer(
        r"<resultMap\b[^>]*\btype\s*=\s*[\"']([^\"']+)[\"']",
        text,
        re.IGNORECASE,
    ):
        table_hints.append(rm.group(1).split(".")[-1])

    return (
        AstType(
            fqcn=namespace,
            simple_name=simple,
            package=package,
            kind="mapper",
            symbol_kind="MapperXml",
            file_path=rel,
            module_name=module,
            start_line=1,
            end_line=text.count("\n") + 1,
            methods=methods,
            table_hints=sorted(set(table_hints)),
            sql_ids=sql_ids,
            sql_table_fields=sql_table_fields,
            method_sql_meta=method_sql_meta,
        ),
        errors,
    )


def _simple_type_name(raw: str) -> str:
    return raw.split("<", 1)[0].split(".")[-1].strip()


def _type_node_simple(source: bytes, node: Any) -> str | None:
    if node is None:
        return None
    if node.type in {
        "type_identifier",
        "generic_type",
        "scoped_type_identifier",
        "array_type",
        "integral_type",
        "floating_point_type",
        "boolean_type",
        "void_type",
    }:
        return _simple_type_name(_text(source, node))
    return None


def _type_node_args(source: bytes, node: Any) -> list[str]:
    """Generic type arguments of a type node (e.g. List<OrderItem> → ["OrderItem"]).

    Returns the raw inner type names (dotted names simplified to simple names).
    """
    if node is None or node.type != "generic_type":
        return []
    for c in node.children:
        if c.type != "type_arguments":
            continue
        out: list[str] = []
        for arg in c.children:
            if arg.type in {
                "type_identifier",
                "generic_type",
                "scoped_type_identifier",
            }:
                name = _simple_type_name(_text(source, arg))
                if name:
                    out.append(name)
        return out
    return []


def _class_field_generic_args(source: bytes | None) -> list[str]:
    """First field's generic args in a type source (e.g. List<OrderItem> → [OrderItem]).

    Used to type lambda params of ``list.forEach(item -> …)``: the receiver's
    class declares ``private List<OrderItem> items;`` → element type OrderItem.
    """
    if not source:
        return []
    try:
        root = _parser().parse(source).root_node
    except Exception:
        return []
    for n in _walk(root):
        if n.type != "field_declaration":
            continue
        for c in n.children:
            args = _type_node_args(source, c)
            if args:
                return args
    return []


def _infer_rhs_type(source: bytes, rhs: Any | None) -> tuple[str | None, str | None]:
    """Infer type from RHS; return (type_simple, how) or (None, None)."""
    if rhs is None:
        return None, None
    if rhs.type == "object_creation_expression":
        for c in rhs.children:
            t = _type_node_simple(source, c)
            if t:
                return t, "new"
        return None, None
    if rhs.type == "cast_expression":
        for c in rhs.children:
            t = _type_node_simple(source, c)
            if t:
                return t, "cast"
        return None, None
    if rhs.type == "parenthesized_expression" and rhs.children:
        return _infer_rhs_type(source, rhs.children[0] if len(rhs.children) == 1 else rhs.children[-1])
    return None, None


def extract_method_locals(
    source: bytes,
    method_node: Any,
    *,
    returns_by_name: dict[str, str] | None = None,
    generic_params: dict[str, list[str]] | None = None,
) -> dict[str, ScopedName]:
    """Collect params + local vars with obvious types (decl / new / cast / foreach).

    Enrichments (optional, via kwargs):
      - ``returns_by_name``: method simple-name → return simple type. Lets
        ``var x = svc.find()`` resolve to the callee's return type.
      - ``generic_params``: receiver simple-name → its generic args. Lets
        ``items.forEach(item -> …)`` type the lambda param as the element type
        when the receiver is a ``List<OrderItem>``-style local.
    First declaration wins; reassignments only *upgrade* an unknown/Object type
    to a more concrete one (recorded in ``reassigned_to``).
    """
    returns_by_name = returns_by_name or {}
    generic_params = generic_params or {}
    out: dict[str, ScopedName] = {}

    def _put(
        name: str,
        type_name: str | None,
        how: str,
        type_args: list[str] | None = None,
    ) -> None:
        if not name:
            return
        if name in out:
            return
        # Preserve semantic how (param/foreach/lambda/new/…) even when the type
        # is unknown — only fall back to "unknown" for genuinely unclassified.
        final_how = how if type_name else ("unknown" if how in {"", None} else how)
        out[name] = ScopedName(
            name=name,
            type_name=type_name,
            how=final_how,
            type_args=list(type_args or []),
        )

    def _upgrade(name: str, type_name: str | None, type_args: list[str] | None = None) -> None:
        """Reassignment: only raise unknown/Object to a more concrete type."""
        cur = out.get(name)
        if cur is None:
            return
        if not type_name:
            return
        if cur.type_name in {None, "Object", "object"} or not cur.type_name:
            cur.type_name = type_name
            if type_args:
                cur.type_args = list(type_args)
            cur.how = "unknown"
        cur.reassigned_to = type_name

    def _infer_call_rhs(rhs: Any) -> tuple[str | None, str | None, list[str]]:
        """RHS method invocation → (return type, how, generic args) via returns map."""
        if rhs is None:
            return None, None, []
        if rhs.type == "method_invocation":
            arg_idx = next(
                (i for i, c in enumerate(rhs.children) if c.type == "argument_list"),
                None,
            )
            before = rhs.children[:arg_idx] if arg_idx is not None else rhs.children
            mname = ""
            for x in before:
                if x.type in {"identifier", "field_access"}:
                    mname = _text(source, x).split(".")[-1]
            if mname:
                rt = returns_by_name.get(mname)
                if rt:
                    return _simple_type_name(rt), "return_type", []
        return None, None, []

    # Formal parameters
    for c in method_node.children:
        if c.type != "formal_parameters":
            continue
        for p in c.children:
            if p.type != "formal_parameter":
                continue
            tname = ""
            targs: list[str] = []
            pname = ""
            for x in p.children:
                tn = _type_node_simple(source, x)
                if tn:
                    tname = tn
                    targs = _type_node_args(source, x)
                if x.type == "identifier":
                    pname = _text(source, x)
            if pname:
                _put(pname, tname or None, "param", targs)

    body = next((c for c in method_node.children if c.type == "block"), None)
    if body is None:
        return out

    for n in _walk(body):
        if n.type == "local_variable_declaration":
            decl_type: str | None = None
            decl_args: list[str] = []
            for c in n.children:
                tn = _type_node_simple(source, c)
                if tn and tn != "var":
                    decl_type = tn
                    decl_args = _type_node_args(source, c)
            for c in n.children:
                if c.type != "variable_declarator":
                    continue
                vname = ""
                rhs = None
                for x in c.children:
                    if x.type == "identifier" and not vname:
                        vname = _text(source, x)
                    elif x.type not in {"=", ","}:
                        rhs = x
                if not vname:
                    continue
                if decl_type:
                    # Prefer more specific how when RHS is new/cast
                    rhs_t, rhs_how = _infer_rhs_type(source, rhs)
                    if rhs_how == "new" and (rhs_t == decl_type or rhs_t is not None):
                        _put(vname, decl_type, "new", decl_args)
                    elif rhs_how == "cast":
                        _put(vname, decl_type, "cast", decl_args)
                    else:
                        _put(vname, decl_type, "local_decl", decl_args)
                else:
                    # var / missing decl type — try RHS (new/cast first, then returns map)
                    rhs_t, rhs_how = _infer_rhs_type(source, rhs)
                    if not rhs_t and rhs is not None and rhs.type == "method_invocation":
                        rt, rhow, rargs = _infer_call_rhs(rhs)
                        if rt:
                            rhs_t, rhs_how, decl_args = rt, rhow, rargs
                    _put(vname, rhs_t, rhs_how or "unknown", decl_args)
        elif n.type == "enhanced_for_statement":
            tname = ""
            targs: list[str] = []
            vname = ""
            for c in n.children:
                tn = _type_node_simple(source, c)
                if tn and tn != "var":
                    tname = tn
                    targs = _type_node_args(source, c)
                if c.type == "identifier" and not vname:
                    # first identifier after type is the loop var
                    if tname:
                        vname = _text(source, c)
            if vname:
                _put(vname, tname or None, "foreach", targs)
        elif n.type == "catch_formal_parameter":
            tname = ""
            vname = ""
            for c in n.children:
                if c.type == "catch_type":
                    # union catch: take first type
                    for x in c.children:
                        tn = _type_node_simple(source, x)
                        if tn:
                            tname = tn
                            break
                elif c.type == "identifier":
                    vname = _text(source, c)
            if vname:
                _put(vname, tname or None, "param")
        elif n.type == "assignment_expression":
            # Reassignment: fill unknowns OR upgrade Object to concrete type.
            if len(n.children) < 3:
                continue
            left, rhs = n.children[0], n.children[-1]
            if left.type != "identifier":
                continue
            vname = _text(source, left)
            rhs_t, rhs_how = _infer_rhs_type(source, rhs)
            if not rhs_t and rhs is not None and rhs.type == "method_invocation":
                rt, rhow, rargs = _infer_call_rhs(rhs)
                if rt:
                    rhs_t, rhs_how = rt, rhow
            if vname not in out:
                # assignment without prior decl (rare) — record what we know
                if rhs_t and rhs_how:
                    _put(vname, rhs_t, rhs_how)
            else:
                _upgrade(vname, rhs_t)

    # Lambda parameters: items.forEach(item -> …) → item typed by receiver generics.
    for n in _walk(body):
        if n.type != "method_invocation":
            continue
        arg_idx = next(
            (i for i, c in enumerate(n.children) if c.type == "argument_list"), None
        )
        if arg_idx is None:
            continue
        recv_node = _invocation_receiver_node(n)
        recv_name = ""
        if recv_node is not None and recv_node.type == "identifier":
            recv_name = _text(source, recv_node)
        elif recv_node is not None and recv_node.type == "method_invocation":
            # fluent chain: items.stream().filter(x -> …) → root receiver "items"
            cur = recv_node
            for _ in range(8):
                inner = _invocation_receiver_node(cur)
                if inner is None:
                    break
                if inner.type == "identifier":
                    recv_name = _text(source, inner)
                    break
                cur = inner
        args = n.children[arg_idx]
        for lam in args.children:
            if lam.type != "lambda_expression":
                continue
            # lambda params: (item) -> …  or  item -> …
            for p in lam.children:
                if p.type != "identifier":
                    continue
                pname = _text(source, p)
                if pname in out:
                    continue
                # element type from receiver generics: List<OrderItem> → OrderItem
                el_type = None
                if recv_name in generic_params and generic_params[recv_name]:
                    el_type = generic_params[recv_name][0]
                elif recv_name in out and out[recv_name].type_args:
                    el_type = out[recv_name].type_args[0]
                _put(pname, el_type, "lambda")

    return out


def find_method_node(
    source: bytes,
    tree_root: Any,
    *,
    method_name: str,
    start_line: int | None = None,
) -> Any | None:
    """Locate a method_declaration node by name (and optional 1-based start line)."""
    matches: list[Any] = []
    for n in _walk(tree_root):
        if n.type != "method_declaration":
            continue
        if _method_name(source, n) != method_name:
            continue
        matches.append(n)
    if not matches:
        return None
    if start_line is not None:
        for n in matches:
            if _line(n) == start_line:
                return n
        # nearest start line
        matches.sort(key=lambda n: abs(_line(n) - start_line))
    return matches[0]


def _resolve_call(
    call: str,
    fields: dict[str, str],
    by_simple: dict[str, list[str]],
    current_class: str,
    declared_methods: set[str],
    locals_map: dict[str, ScopedName] | None = None,
) -> tuple[str, str, str, dict[str, Any]]:
    """Return (type_simple, method, status, detail). Uses field_map + optional locals."""
    detail = resolve_call_detailed(
        call,
        field_map=fields,
        locals_map=locals_map or {},
        by_simple=by_simple,
        current_class=current_class,
        declared_methods=declared_methods,
    )
    status = detail["status"]
    method = detail["method"]
    # Prefer Impl simple name when present (preserves prior call-graph keys).
    resolved = detail.get("resolved") or ""
    if "#" in resolved:
        type_name = resolved.split("#", 1)[0]
    else:
        type_name = detail["type_name"] or ""
    return type_name, method, status, detail


def resolve_call_detailed(
    call: str,
    *,
    field_map: dict[str, str],
    locals_map: dict[str, ScopedName],
    by_simple: dict[str, list[str]],
    current_class: str,
    declared_methods: set[str],
) -> dict[str, Any]:
    """Resolve one call with field_map + local vars.

    Returns keys: raw_call, receiver, recv_lookup, method, type_name, how,
    resolved (Class#method|None), status, skip_reason.
    """
    if "." in call:
        recv, method = call.rsplit(".", 1)
        bare = False
    else:
        recv, method = "this", call
        bare = True

    recv_lookup = recv
    if recv_lookup.startswith("this."):
        recv_lookup = recv_lookup[5:]
    elif recv_lookup.startswith("super."):
        recv_lookup = recv_lookup[6:]
    if "." in recv_lookup:
        recv_lookup = recv_lookup.rsplit(".", 1)[-1]

    base: dict[str, Any] = {
        "raw_call": call,
        "receiver": recv,
        "recv_lookup": recv_lookup,
        "method": method,
        "type_name": None,
        "how": "unknown",
        "resolved": None,
        "status": "skipped",
        "skip_reason": None,
    }

    # Utility / noise skips (still report how if we know the receiver type)
    def _attach_known_recv() -> None:
        if recv_lookup in locals_map and locals_map[recv_lookup].type_name:
            base["type_name"] = locals_map[recv_lookup].type_name
            base["how"] = locals_map[recv_lookup].how
        elif recv_lookup in field_map:
            base["type_name"] = field_map[recv_lookup]
            base["how"] = "field_map"
        elif recv in {"this", "super"}:
            base["type_name"] = current_class
            base["how"] = "this"

    if recv_lookup in _SKIP_RECV or method in _SKIP_METHOD:
        base["skip_reason"] = "skipped_util_or_noise"
        _attach_known_recv()
        return base
    if method.startswith("get") and len(method) > 3 and method[3:4].isupper():
        base["skip_reason"] = "skipped_getter"
        _attach_known_recv()
        return base
    if method.startswith("set") and len(method) > 3 and method[3:4].isupper():
        base["skip_reason"] = "skipped_setter"
        _attach_known_recv()
        return base
    if bare and method not in declared_methods:
        base["skip_reason"] = "bare_not_declared"
        return base

    how = "unknown"
    type_name: str | None = None

    if recv in {"this", "super"}:
        type_name = current_class
        how = "this"
        if method not in declared_methods:
            base["skip_reason"] = "this_method_not_declared"
            base["type_name"] = type_name
            base["how"] = how
            return base
    else:
        if recv_lookup in locals_map and locals_map[recv_lookup].type_name:
            type_name = locals_map[recv_lookup].type_name
            how = locals_map[recv_lookup].how
        elif recv_lookup in field_map:
            type_name = field_map[recv_lookup]
            how = "field_map"
        elif recv in field_map:
            type_name = field_map[recv]
            how = "field_map"
        elif recv_lookup[:1].isupper():
            type_name = recv_lookup
            how = "static_type"
        elif recv[:1].isupper() and "." not in recv:
            type_name = recv
            how = "static_type"
        else:
            base["skip_reason"] = "unknown_receiver"
            return base

    base["type_name"] = type_name
    base["how"] = how

    prefer_impl = bool(by_simple.get(f"{type_name}Impl"))
    resolved_type = f"{type_name}Impl" if prefer_impl else type_name
    candidates = by_simple.get(f"{type_name}Impl") or by_simple.get(type_name) or []
    if not candidates and not by_simple.get(type_name):
        base["status"] = "no_local_source"
        base["resolved"] = f"{resolved_type}#{method}"
        base["skip_reason"] = "no_local_source"
        return base

    base["status"] = "ok_local"
    base["resolved"] = f"{resolved_type}#{method}"
    return base


def _locals_for_method(t: AstType, m: AstMethod) -> dict[str, ScopedName]:
    """Best-effort locals map for call-graph enrichment (empty if no source)."""
    if not t.source_bytes:
        return {}
    try:
        root = _parser().parse(t.source_bytes).root_node
        node = find_method_node(
            t.source_bytes,
            root,
            method_name=m.method_name,
            start_line=m.start_line,
        )
        if node is None:
            return {}
        return extract_method_locals(t.source_bytes, node)
    except Exception:
        return {}


def _build_call_graph(ast: GlobalAst) -> None:
    by_simple = ast.by_simple
    # Index methods by Class#method and also Impl aliases
    method_index: dict[str, AstMethod] = dict(ast.methods)
    for cm, m in list(ast.methods.items()):
        # also index without Impl for interface→impl
        if m.simple_name.endswith("Impl"):
            iface = m.simple_name[: -len("Impl")]
            method_index.setdefault(f"{iface}#{m.method_name}", m)

    edges: list[CallEdge] = []
    callees_of: dict[str, list[str]] = {}
    callers_of: dict[str, list[str]] = {}

    for t in ast.types.values():
        declared = {m.method_name for m in t.methods}
        for m in t.methods:
            locals_map = _locals_for_method(t, m)
            resolved: list[str] = []
            body_order = 0
            for call in m.calls:
                type_name, method, status, detail = _resolve_call(
                    call,
                    t.field_map,
                    by_simple,
                    t.simple_name,
                    declared,
                    locals_map=locals_map,
                )
                if status == "skipped":
                    continue
                callee_cm = f"{type_name}#{method}"
                # Prefer Impl key if present
                if callee_cm not in method_index and by_simple.get(f"{type_name}Impl"):
                    callee_cm = f"{type_name}Impl#{method}"
                recv = str(detail.get("receiver") or "")
                recv_type = detail.get("type_name")
                how = str(detail.get("how") or "unknown")
                if status == "ok_local" and callee_cm in method_index:
                    body_order += 1
                    resolved.append(callee_cm)
                    edges.append(
                        CallEdge(
                            caller=m.class_method,
                            callee=callee_cm,
                            raw_call=call,
                            status="ok_local",
                            order=body_order,
                            receiver=recv,
                            receiver_type=recv_type if isinstance(recv_type, str) else None,
                            resolve_how=how,
                        )
                    )
                    callees_of.setdefault(m.class_method, []).append(callee_cm)
                    callers_of.setdefault(callee_cm, []).append(m.class_method)
                elif status == "no_local_source":
                    body_order += 1
                    edges.append(
                        CallEdge(
                            caller=m.class_method,
                            callee=callee_cm,
                            raw_call=call,
                            status="no_local_source",
                            order=body_order,
                            receiver=recv,
                            receiver_type=recv_type if isinstance(recv_type, str) else None,
                            resolve_how=how,
                        )
                    )
            # dedupe preserve order
            seen: set[str] = set()
            uniq: list[str] = []
            for c in resolved:
                if c not in seen and c != m.class_method:
                    seen.add(c)
                    uniq.append(c)
            m.resolved_callees = uniq

    # Deduplicate edges — keep first appearance (lowest body order).
    seen_e: set[tuple[str, str]] = set()
    deduped: list[CallEdge] = []
    for e in edges:
        key = (e.caller, e.callee)
        if key in seen_e:
            continue
        seen_e.add(key)
        deduped.append(e)
    # Re-number order per caller as dense 1..N after dedupe (stable for Cypher ORDER BY).
    per_caller: dict[str, int] = {}
    for e in deduped:
        per_caller[e.caller] = per_caller.get(e.caller, 0) + 1
        e.order = per_caller[e.caller]
    ast.edges = deduped
    ast.callees_of = {k: list(dict.fromkeys(v)) for k, v in callees_of.items()}
    ast.callers_of = {k: list(dict.fromkeys(v)) for k, v in callers_of.items()}


def build_global_ast(
    project_id: str,
    project_root: Path,
    *,
    package_roots: list[str] | None = None,
    parse_workers: int | None = None,
    progress: Callable[[int, int, str], None] | None = None,
) -> GlobalAst:
    """Parse the whole project once into a reusable GlobalAst.

    package_roots: optional path prefixes under project_root to limit scope
      (e.g. ``["loan-service/src/main/java"]``). Default = entire configured root.
    """
    _require_ts()
    root = Path(project_root).resolve()
    java_files = scan_java_files(root)
    xml_files = scan_mapper_xml_files(root)

    if package_roots:
        prefixes = []
        for pr in package_roots:
            p = Path(pr)
            if not p.is_absolute():
                p = (root / pr).resolve()
            prefixes.append(p)
        def _in_scope(fp: Path) -> bool:
            fp = fp.resolve()
            return any(str(fp).startswith(str(pref)) for pref in prefixes)

        java_files = [f for f in java_files if _in_scope(f)]
        xml_files = [f for f in xml_files if _in_scope(f)]

    workers = max(1, min(int(parse_workers or 8), 32))
    ast = GlobalAst(
        project_id=project_id,
        project_root=root,
        package_roots=list(package_roots or []),
    )
    ast.java_file_count = len(java_files)
    ast.mapper_xml_count = len(xml_files)

    total = len(java_files) + len(xml_files)
    done = 0
    if progress:
        progress(0, max(total, 1), f"global AST: parsing {len(java_files)} java + {len(xml_files)} xml")

    logger.info(
        "global AST parse project=%s java=%s xml=%s workers=%s",
        project_id,
        len(java_files),
        len(xml_files),
        workers,
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_parse_one_java, root, fp): ("java", fp) for fp in java_files}
        for fp in xml_files:
            futs[pool.submit(_parse_one_mapper_xml, root, fp)] = ("xml", fp)

        for fut in as_completed(futs):
            kind, fp = futs[fut]
            done += 1
            try:
                result = fut.result()
            except Exception as e:
                ast.errors.append(f"{fp}: {e}")
                continue
            if kind == "java":
                types, errs = result  # type: ignore[misc]
                ast.errors.extend(errs)
                for t in types:
                    # Prefer first, keep Impl over interface if clash on simple later
                    if t.fqcn in ast.types and ast.types[t.fqcn].symbol_kind == "MapperXml":
                        continue
                    ast.types[t.fqcn] = t
                    ast.by_simple.setdefault(t.simple_name, [])
                    if t.fqcn not in ast.by_simple[t.simple_name]:
                        ast.by_simple[t.simple_name].append(t.fqcn)
                    for m in t.methods:
                        # Prefer Impl methods when duplicate Class#method
                        prev = ast.methods.get(m.class_method)
                        if prev and prev.simple_name.endswith("Impl") and not m.simple_name.endswith("Impl"):
                            continue
                        ast.methods[m.class_method] = m
            else:
                t, errs = result  # type: ignore[misc]
                ast.errors.extend(errs)
                if t is not None:
                    # Mapper XML namespace may collide with Java Mapper interface — keep both
                    # under same fqcn preferring Java for type shell, merge SQL methods.
                    existing = ast.types.get(t.fqcn)
                    if existing and existing.kind != "mapper":
                        # attach sql methods onto interface
                        for m in t.methods:
                            if m.class_method not in {x.class_method for x in existing.methods}:
                                existing.methods.append(m)
                                ast.methods[m.class_method] = m
                        existing.table_hints = list(
                            dict.fromkeys(existing.table_hints + t.table_hints)
                        )
                        existing.sql_ids = list(dict.fromkeys(existing.sql_ids + t.sql_ids))
                        if existing.symbol_kind == "Class":
                            existing.symbol_kind = "Mapper"
                    else:
                        ast.types[t.fqcn] = t
                        ast.by_simple.setdefault(t.simple_name, [])
                        if t.fqcn not in ast.by_simple[t.simple_name]:
                            ast.by_simple[t.simple_name].append(t.fqcn)
                        for m in t.methods:
                            ast.methods[m.class_method] = m

            if progress and (done % 20 == 0 or done == total):
                progress(done, max(total, 1), f"AST parsed {fp.name}")

    _build_call_graph(ast)
    logger.info("global AST done %s", ast.stats())
    if progress:
        progress(total, max(total, 1), f"AST ready {ast.stats()}")
    return ast
