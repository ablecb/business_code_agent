from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ParsedAnnotation:
    name: str
    values: dict[str, str] = field(default_factory=dict)


@dataclass
class ParsedMethod:
    name: str
    signature: str
    start_line: int | None
    end_line: int | None
    annotations: list[ParsedAnnotation] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)  # rough "obj.method" or "method"
    raw_code: str = ""
    http_mappings: list[dict[str, str]] = field(default_factory=list)
    doc_comment: str = ""  # Javadoc / line comment directly above the method


@dataclass
class ParsedField:
    name: str
    type_name: str
    annotations: list[ParsedAnnotation] = field(default_factory=list)


@dataclass
class ParsedType:
    kind: str  # class/interface/enum
    package: str
    simple_name: str
    fqcn: str
    file_path: str
    module_name: str
    start_line: int | None
    end_line: int | None
    annotations: list[ParsedAnnotation] = field(default_factory=list)
    modifiers: list[str] = field(default_factory=list)
    extends: list[str] = field(default_factory=list)
    implements: list[str] = field(default_factory=list)
    fields: list[ParsedField] = field(default_factory=list)
    methods: list[ParsedMethod] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)
    symbol_kind: str = "Class"  # Controller/Service/Listener/Feign/Job/Class
    feign_meta: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    types: list[ParsedType] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
