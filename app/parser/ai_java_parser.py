"""Pure-AI Java structure parser (no javalang / no regex AST).

Outputs the same ParsedType model used by graph / tagging / flow stages.
Source slicing for raw_code uses brace matching only — not structural parsing.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from app.core.config import get_settings
from app.core.utils import module_from_path, truncate
from app.llm.factory import get_parse_llm_provider
from app.parser.models import (
    ParsedAnnotation,
    ParsedField,
    ParsedMethod,
    ParsedType,
    ParseResult,
)

logger = logging.getLogger("app.parser.ai")

PARSE_PROMPT = """你是资深 Java/Spring 源码结构分析器。根据给定 Java 源文件，输出严格 JSON（不要 markdown）：
{
  "package": "包名或空字符串",
  "imports": ["fully.qualified.Name", "..."],
  "types": [
    {
      "kind": "class|interface|enum",
      "simple_name": "类名",
      "modifiers": ["public"],
      "extends": ["父类简名"],
      "implements": ["接口简名"],
      "annotations": [{"name":"RestController","values":{"value":""}}],
      "symbol_kind": "Controller|Service|Listener|Feign|Job|Mapper|Entity|Class",
      "feign_meta": {"name":"","path":"","contextId":""},
      "request_mapping": "/api/xxx",
      "fields": [{"name":"svc","type_name":"FooService","annotations":[{"name":"Autowired","values":{}}]}],
      "methods": [
        {
          "name":"create",
          "params":[{"type":"Req","name":"req"}],
          "return_type":"Result",
          "modifiers":["public"],
          "start_line": 10,
          "end_line": 40,
          "annotations":[{"name":"PostMapping","values":{"value":"/create"}}],
          "http_mappings":[{"http_method":"POST","path":"/create"}],
          "calls":["fooService.save","rabbitTemplate.convertAndSend"],
          "skip_accessor": false
        }
      ]
    }
  ]
}
规则：
1. 只根据源码，不要编造不存在的类/方法。
2. symbol_kind：RestController/Controller→Controller；FeignClient→Feign；RabbitListener 类或方法→Listener；
   @Service→Service；@XxlJob→Job；@Mapper/*Mapper→Mapper；@Entity/*Entity→Entity；否则 Class。
3. 跳过纯 JavaBean get/set/is（skip_accessor=true 或不输出）。
4. calls 列出方法体内有意义的调用（obj.method 或 method），最多 40 个。
5. start_line/end_line 尽量准确（1-based）；不确定时给近似行号。
6. annotations.values 的值一律字符串。
"""

MAPPING_ANN = {
    "GetMapping",
    "PostMapping",
    "PutMapping",
    "DeleteMapping",
    "PatchMapping",
    "RequestMapping",
}

# Method-level entry markers used when condensing oversized source for LLM parse.
_ENTRY_ANN_RE = re.compile(
    r"@(?:Get|Post|Put|Delete|Patch|Request)Mapping\b|@RabbitListener\b|@KafkaListener\b|"
    r"@XxlJob\b|@Scheduled\b|@JobHandler\b",
    re.M,
)


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _find_matching_brace(text: str, open_idx: int) -> int:
    depth = 0
    i = open_idx
    in_str = False
    str_ch = ""
    while i < len(text):
        ch = text[i]
        if in_str:
            if ch == "\\":
                i += 2
                continue
            if ch == str_ch:
                in_str = False
            i += 1
            continue
        if ch in ('"', "'"):
            in_str = True
            str_ch = ch
            i += 1
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return len(text) - 1


def _extract_entry_method_pieces(text: str) -> tuple[str, str, str, list[str]]:
    """Return (package_stmt, imports_block, class_header, entry_method_sources)."""
    pkg_m = re.search(r"^\s*package\s+[\w.]+\s*;", text, re.M)
    pkg = pkg_m.group(0) if pkg_m else ""
    imports = "\n".join(re.findall(r"^\s*import\s+.+?;", text, re.M)[:80])
    class_m = re.search(
        r"((?:^@[^\n]+\n)*)^\s*(?:public\s+|protected\s+|private\s+)?(?:abstract\s+|final\s+)?"
        r"(?:class|interface|enum)\s+\w+[^{]*\{",
        text,
        re.M,
    )
    header = class_m.group(0) if class_m else ""

    pieces: list[str] = []
    seen_spans: set[tuple[int, int]] = set()
    for m in _ENTRY_ANN_RE.finditer(text):
        line_start = text.rfind("\n", 0, m.start()) + 1
        for _ in range(12):
            prev_nl = text.rfind("\n", 0, line_start - 1)
            if prev_nl < 0:
                break
            prev_line = text[prev_nl + 1 : line_start].strip()
            if (
                prev_line.startswith("@")
                or prev_line.startswith("/**")
                or prev_line.startswith("*")
                or prev_line.startswith("public")
                or prev_line.startswith("private")
                or prev_line.startswith("protected")
                or prev_line == ""
            ):
                line_start = prev_nl + 1
                continue
            break
        brace = text.find("{", m.start())
        if brace < 0:
            continue
        end = _find_matching_brace(text, brace)
        span = (line_start, end)
        if span in seen_spans:
            continue
        seen_spans.add(span)
        piece = text[line_start : end + 1]
        # keep method bodies bounded so one endpoint does not blow the batch
        if len(piece) > 8000:
            piece = piece[:7920] + "\n/* truncated method body */\n}"
        pieces.append(piece)
    return pkg, imports, header, pieces


def _assemble_condensed(
    pkg: str,
    imports: str,
    header: str,
    pieces: list[str],
    *,
    max_chars: int,
    original_chars: int,
) -> str:
    body = "\n\n".join(pieces)
    condensed = (
        f"{pkg}\n\n{imports}\n\n"
        f"/* condensed: original_chars={original_chars}; entry_methods={len(pieces)} */\n"
        f"{header}\n\n{body}\n}}\n"
    )
    if len(condensed) > max_chars:
        return condensed[:max_chars]
    return condensed


def _condense_oversized_source(text: str, *, max_chars: int) -> str:
    """For huge Java files, keep package/imports/class header + annotated entry methods."""
    if len(text) <= max_chars:
        return text
    pkg, imports, header, pieces = _extract_entry_method_pieces(text)
    if not pieces:
        return truncate(text, max_chars)
    return _assemble_condensed(
        pkg, imports, header, pieces, max_chars=max_chars, original_chars=len(text)
    )


def _merge_parsed_types(parts: list[ParsedType]) -> list[ParsedType]:
    by_fqcn: dict[str, ParsedType] = {}
    for t in parts:
        cur = by_fqcn.get(t.fqcn)
        if not cur:
            by_fqcn[t.fqcn] = t
            continue
        seen = {m.signature for m in cur.methods}
        for m in t.methods:
            if m.signature not in seen:
                cur.methods.append(m)
                seen.add(m.signature)
        if not cur.fields and t.fields:
            cur.fields = list(t.fields)
        if not cur.annotations and t.annotations:
            cur.annotations = list(t.annotations)
        cur.extra.update(t.extra or {})
    return list(by_fqcn.values())


def _offset_at_line(text: str, line: int) -> int:
    if line <= 1:
        return 0
    n = 0
    for i, ch in enumerate(text):
        if ch == "\n":
            n += 1
            if n >= line - 1:
                return i + 1
    return len(text)


def _slice_method_source(text: str, method_name: str, start_line: int | None) -> tuple[str, str, int | None]:
    """Return (raw_code, body, end_line) using brace match from approx start_line."""
    if not method_name:
        return "", "", start_line
    start_idx = _offset_at_line(text, start_line or 1)
    # search method name near start
    window = text[max(0, start_idx - 200) : start_idx + 8000]
    pat = re.compile(rf"\b{re.escape(method_name)}\s*\(")
    m = pat.search(window)
    if not m:
        m = pat.search(text)
        if not m:
            return "", "", start_line
        name_abs = m.start()
    else:
        name_abs = max(0, start_idx - 200) + m.start()

    # expand back over annotations/modifiers
    full_start = name_abs
    while full_start > 0 and text[full_start - 1] not in "{};":
        full_start -= 1
        if name_abs - full_start > 800:
            break
    # trim to line start of annotation block
    nl = text.rfind("\n", max(0, full_start - 1), name_abs)
    if nl >= 0:
        full_start = nl + 1

    paren = text.find("(", name_abs)
    brace = text.find("{", paren if paren >= 0 else name_abs)
    semi = text.find(";", paren if paren >= 0 else name_abs)
    if brace < 0 or (semi >= 0 and semi < brace):
        # abstract / interface method
        end = (semi + 1) if semi >= 0 else name_abs + len(method_name)
        raw = text[full_start:end].strip("\n")
        return raw, "", _line_of(text, end)

    close = _find_matching_brace(text, brace)
    raw = text[full_start : close + 1].strip("\n")
    body = text[brace + 1 : close]
    return raw, body, _line_of(text, close)


def _anns_from_json(items: Any) -> list[ParsedAnnotation]:
    out: list[ParsedAnnotation] = []
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").split(".")[-1].strip()
        if not name:
            continue
        values_raw = item.get("values") or {}
        values: dict[str, str] = {}
        if isinstance(values_raw, dict):
            for k, v in values_raw.items():
                values[str(k)] = "" if v is None else str(v)
        out.append(ParsedAnnotation(name=name, values=values))
    return out


def _http_mappings(anns: list[ParsedAnnotation], explicit: Any = None) -> list[dict[str, str]]:
    mappings: list[dict[str, str]] = []
    if isinstance(explicit, list):
        for item in explicit:
            if not isinstance(item, dict):
                continue
            mappings.append(
                {
                    "http_method": str(item.get("http_method") or "REQUEST"),
                    "path": str(item.get("path") or item.get("value") or ""),
                }
            )
    if mappings:
        return mappings
    for a in anns:
        if a.name in MAPPING_ANN:
            path = a.values.get("value") or a.values.get("path") or ""
            method = {
                "GetMapping": "GET",
                "PostMapping": "POST",
                "PutMapping": "PUT",
                "DeleteMapping": "DELETE",
                "PatchMapping": "PATCH",
                "RequestMapping": (a.values.get("method") or "REQUEST").replace(
                    "RequestMethod.", ""
                ),
            }.get(a.name, "REQUEST")
            mappings.append({"http_method": method, "path": path})
    return mappings


def _symbol_kind(anns: list[ParsedAnnotation], simple_name: str, hinted: str | None) -> str:
    if hinted in {
        "Controller",
        "Service",
        "Listener",
        "Feign",
        "Job",
        "Mapper",
        "Entity",
        "Class",
    }:
        return hinted
    names = {a.name for a in anns}
    if "FeignClient" in names:
        return "Feign"
    if "RestController" in names or "Controller" in names:
        return "Controller"
    if "RabbitListener" in names or simple_name.endswith("Listener") or simple_name.endswith("Listenner"):
        return "Listener"
    if "Service" in names or simple_name.endswith("Service") or simple_name.endswith("ServiceImpl"):
        return "Service"
    if "XxlJob" in names or simple_name.endswith("Job"):
        return "Job"
    if "Mapper" in names or simple_name.endswith("Mapper"):
        return "Mapper"
    if "Entity" in names or simple_name.endswith("Entity"):
        return "Entity"
    return "Class"


def _llm_chat_sync(messages: list[dict[str, str]], *, max_tokens: int = 4096) -> str:
    import asyncio

    llm = get_parse_llm_provider()

    async def _run() -> str:
        return await llm.chat(
            messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    # Already inside an event loop (rare for parse workers): run in a fresh thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()


def _extract_json(content: str) -> dict[str, Any]:
    content = (content or "").strip()
    if not content:
        raise ValueError("empty LLM response")
    try:
        data = json.loads(content)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in LLM response")
    data = json.loads(m.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON root must be object")
    return data


def _types_from_ai_json(
    data: dict[str, Any],
    *,
    text: str,
    rel_path: str,
    module_name: str,
    fallback_package: str,
) -> list[ParsedType]:
    package = str(data.get("package") or fallback_package or "")
    imports = [str(x) for x in (data.get("imports") or []) if str(x).strip()]
    out: list[ParsedType] = []
    for t in data.get("types") or []:
        if not isinstance(t, dict):
            continue
        simple = str(t.get("simple_name") or "").strip()
        if not simple:
            continue
        kind = str(t.get("kind") or "class")
        fqcn = f"{package}.{simple}" if package else simple
        type_anns = _anns_from_json(t.get("annotations"))
        symbol_kind = _symbol_kind(type_anns, simple, str(t.get("symbol_kind") or "") or None)
        parsed = ParsedType(
            kind=kind,
            package=package,
            simple_name=simple,
            fqcn=fqcn,
            file_path=rel_path,
            module_name=module_name,
            start_line=int(t["start_line"]) if t.get("start_line") else None,
            end_line=int(t["end_line"]) if t.get("end_line") else None,
            annotations=type_anns,
            modifiers=[str(m) for m in (t.get("modifiers") or []) if str(m).strip()],
            extends=[str(x).strip() for x in (t.get("extends") or []) if str(x).strip()],
            implements=[str(x).strip() for x in (t.get("implements") or []) if str(x).strip()],
            imports=imports,
            symbol_kind=symbol_kind,
        )
        feign = t.get("feign_meta") if isinstance(t.get("feign_meta"), dict) else {}
        if feign or any(a.name == "FeignClient" for a in type_anns):
            fc = next((a for a in type_anns if a.name == "FeignClient"), None)
            parsed.feign_meta = {
                "name": str(feign.get("name") or (fc.values.get("name") if fc else "") or (fc.values.get("value") if fc else "") or ""),
                "path": str(feign.get("path") or (fc.values.get("path") if fc else "") or ""),
                "contextId": str(feign.get("contextId") or (fc.values.get("contextId") if fc else "") or ""),
            }
            parsed.symbol_kind = "Feign"
        rm = t.get("request_mapping") or ""
        if not rm:
            for a in type_anns:
                if a.name == "RequestMapping":
                    rm = a.values.get("value") or a.values.get("path") or ""
                    break
        if rm:
            parsed.extra["request_mapping"] = str(rm)

        for f in t.get("fields") or []:
            if not isinstance(f, dict):
                continue
            fname = str(f.get("name") or "").strip()
            if not fname:
                continue
            parsed.fields.append(
                ParsedField(
                    name=fname,
                    type_name=str(f.get("type_name") or f.get("type") or "Object"),
                    annotations=_anns_from_json(f.get("annotations")),
                )
            )

        for m in t.get("methods") or []:
            if not isinstance(m, dict):
                continue
            if m.get("skip_accessor"):
                continue
            mname = str(m.get("name") or "").strip()
            if not mname:
                continue
            params = m.get("params") or []
            if isinstance(params, list):
                param_s = ", ".join(
                    f"{(p.get('type') if isinstance(p, dict) else 'Object')} "
                    f"{(p.get('name') if isinstance(p, dict) else 'arg')}"
                    for p in params
                )
            else:
                param_s = str(params)
            # skip bean accessors if AI forgot the flag
            from app.parser.java_parser import is_accessor_method

            pcount = 0 if not param_s.strip() else param_s.count(",") + 1
            if is_accessor_method(mname, pcount):
                continue

            m_anns = _anns_from_json(m.get("annotations"))
            start_line = int(m["start_line"]) if m.get("start_line") else None
            raw_code, body, end_line = _slice_method_source(text, mname, start_line)
            if m.get("end_line"):
                try:
                    end_line = int(m["end_line"])
                except (TypeError, ValueError):
                    pass
            if not raw_code:
                # fallback: keep a short stub so symbol store still has something
                raw_code = truncate(
                    f"{' '.join('@'+a.name for a in m_anns)} {mname}({param_s}) {{ ... }}",
                    500,
                )

            calls = [str(c) for c in (m.get("calls") or []) if str(c).strip()]
            if any(a.name == "RabbitListener" for a in m_anns) and parsed.symbol_kind == "Class":
                parsed.symbol_kind = "Listener"
            if any(a.name == "XxlJob" for a in m_anns) and parsed.symbol_kind in {
                "Class",
                "Service",
            }:
                parsed.symbol_kind = "Job"

            parsed.methods.append(
                ParsedMethod(
                    name=mname,
                    signature=f"{fqcn}#{mname}({param_s})",
                    start_line=start_line,
                    end_line=end_line,
                    annotations=m_anns,
                    modifiers=[str(x) for x in (m.get("modifiers") or []) if str(x).strip()],
                    calls=calls[:200],
                    raw_code=raw_code,
                    http_mappings=_http_mappings(m_anns, m.get("http_mappings")),
                )
            )
        out.append(parsed)
    return out


def parse_java_file_with_ai(
    file_path: Path,
    project_root: Path,
    *,
    method_piece_filter: Callable[[str], bool] | None = None,
) -> ParseResult:
    result = ParseResult()
    try:
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        result.errors.append(f"{file_path}: {e}")
        return result

    module_name = module_from_path(project_root, file_path)
    try:
        rel_path = str(file_path.relative_to(project_root)).replace("\\", "/")
    except ValueError:
        rel_path = str(file_path).replace("\\", "/")

    pkg_m = re.search(r"^\s*package\s+([\w.]+)\s*;", text, re.MULTILINE)
    fallback_package = pkg_m.group(1) if pkg_m else ""
    max_chars = get_settings().parse_ai_max_chars or 22000

    def _one_shot(source: str) -> list[ParsedType]:
        user = (
            f"file={rel_path}\nmodule={module_name}\n"
            f"hint_package={fallback_package}\n\n"
            f"===== JAVA SOURCE BEGIN =====\n{source}\n===== JAVA SOURCE END ====="
        )
        content = _llm_chat_sync(
            [
                {"role": "system", "content": PARSE_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=4096,
        )
        data = _extract_json(content)
        types = _types_from_ai_json(
            data,
            text=text,
            rel_path=rel_path,
            module_name=module_name,
            fallback_package=fallback_package,
        )
        if not types:
            raise ValueError("AI returned no types")
        return types

    try:
        if len(text) <= max_chars:
            result.types.extend(_one_shot(truncate(text, max_chars)))
            return result

        pkg, imports, header, pieces = _extract_entry_method_pieces(text)
        if method_piece_filter is not None:
            pieces = [p for p in pieces if method_piece_filter(p)]
        if not pieces:
            # fallback: truncated head
            result.types.extend(_one_shot(truncate(text, max_chars)))
            return result

        # Batch entry methods so JSON responses stay within model output limits.
        batch_size = 3
        merged_parts: list[ParsedType] = []
        total_batches = (len(pieces) + batch_size - 1) // batch_size
        logger.info(
            "AI parse oversized %s chars=%s entry_methods=%s batches=%s",
            rel_path,
            len(text),
            len(pieces),
            total_batches,
        )
        errors: list[str] = []
        for bi in range(0, len(pieces), batch_size):
            batch = pieces[bi : bi + batch_size]
            source = _assemble_condensed(
                pkg,
                imports,
                header,
                batch,
                max_chars=max_chars,
                original_chars=len(text),
            )
            try:
                part = _one_shot(source)
                merged_parts.extend(part)
                logger.info(
                    "AI parse batch %s/%s ok methods≈%s",
                    bi // batch_size + 1,
                    total_batches,
                    sum(len(t.methods) for t in part),
                )
            except Exception as e:
                errors.append(f"batch{bi // batch_size + 1}: {e}")
                logger.exception("AI parse batch failed %s batch=%s", rel_path, bi // batch_size + 1)

        merged = _merge_parsed_types(merged_parts)
        if not merged:
            raise ValueError(
                "AI returned no types for oversized file; " + "; ".join(errors[:3])
            )
        result.types.extend(merged)
        if errors:
            result.errors.extend(f"{rel_path}: {e}" for e in errors)
    except Exception as e:
        logger.exception("AI java parse failed for %s", rel_path)
        result.errors.append(f"{rel_path}: AI parse failed: {e}")
    return result
