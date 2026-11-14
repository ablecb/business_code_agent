"""Log/stack diagnose: ingest path → parse frames → map symbols → callers → entry."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger("app.agent.runtime.diagnose")

_STACK_AT_RE = re.compile(
    r"\bat\s+([\w.$]+)\.([\w$]+)\(([\w.$]+\.java):(\d+)\)"
)
_CAUSED_BY_RE = re.compile(
    r"Caused by:\s+([\w.$]+(?:Exception|Error))(?::\s*(.*))?"
)
_EXC_LINE_RE = re.compile(
    r"^([\w.$]*(?:Exception|Error)):\s*(.*)$",
    re.M,
)
_LOGGER_RE = re.compile(
    r"(?i)\b(?:ERROR|WARN|FATAL)\b[^\n]{0,80}?\b([\w.]+(?:Exception|Error))\b"
)
_PATH_LABEL_RE = re.compile(
    r"(?:日志路径|日志文件|logfile|log\s*path|log\s*file)"
    r"[是为:=：\s]+([^\s,，\"']+)",
    re.I,
)
_PATH_BARE_RE = re.compile(
    r"((?:[A-Za-z]:\\|\\\\|/)[^\s\"']+\.(?:log|txt|out))",
    re.I,
)
_CM_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]+#[A-Za-z_][A-Za-z0-9_]*)\b")
_EXC_WORD_RE = re.compile(
    r"(?i)\b((?:[A-Z][A-Za-z0-9_]+)?(?:Exception|Error))\b"
)

_SKIP_FRAMES = {
    "Thread",
    "ThreadPoolExecutor",
    "FutureTask",
    "GeneratedMethodAccessor",
    "DelegatingMethodAccessorImpl",
    "NativeMethodAccessorImpl",
    "Method",
    "AppClassLoader",
}

_LOG_SUFFIXES = {".log", ".txt", ".out"}
_MAX_LOG_BYTES = 200_000
_MAX_FILE_BYTES = 5_000_000


def extract_log_paths(text: str) -> list[str]:
    found: list[str] = []
    for m in _PATH_LABEL_RE.finditer(text or ""):
        found.append(m.group(1).strip().strip("\"'"))
    for m in _PATH_BARE_RE.finditer(text or ""):
        found.append(m.group(1).strip())
    return list(dict.fromkeys(p for p in found if p))


def read_log_file(path: str, *, max_bytes: int = _MAX_LOG_BYTES) -> str:
    """Read a local log file if it exists. Returns empty string on skip/error."""
    raw = (path or "").strip().strip("\"'")
    if not raw:
        return ""
    p = Path(raw).expanduser()
    try:
        if not p.is_file():
            return ""
        size = p.stat().st_size
        if size > _MAX_FILE_BYTES:
            logger.info("diagnose skip oversized log %s size=%s", p, size)
            return ""
        suffix = p.suffix.lower()
        name = p.name.lower()
        if suffix not in _LOG_SUFFIXES and "log" not in name:
            logger.info("diagnose skip non-log path %s", p)
            return ""
        data = p.read_bytes()
        if len(data) > max_bytes:
            data = data[-max_bytes:]
        return data.decode("utf-8", errors="replace")
    except OSError:
        logger.debug("diagnose read_log_file failed", exc_info=True)
        return ""


def _simple_cm(fqcn: str, method: str) -> str:
    cls = (fqcn or "").split(".")[-1].replace("$", ".")
    meth = (method or "").split("$")[0]
    if not cls or not meth or cls in _SKIP_FRAMES:
        return ""
    if meth in {"<init>", "<clinit>", "invoke", "run", "call"}:
        return ""
    return f"{cls}#{meth}"


def parse_stack(text: str) -> dict[str, Any]:
    """Extract exception chain + application frames from a stack / log blob."""
    blob = text or ""
    exceptions: list[dict[str, str]] = []
    for m in _CAUSED_BY_RE.finditer(blob):
        exceptions.append(
            {
                "type": m.group(1),
                "message": (m.group(2) or "").strip()[:300],
                "via": "caused_by",
            }
        )
    if not exceptions:
        for m in _EXC_LINE_RE.finditer(blob):
            exceptions.append(
                {
                    "type": m.group(1),
                    "message": (m.group(2) or "").strip()[:300],
                    "via": "head",
                }
            )
            if len(exceptions) >= 4:
                break
    if not exceptions:
        for m in _LOGGER_RE.finditer(blob):
            exceptions.append({"type": m.group(1), "message": "", "via": "logger"})
            break

    frames: list[dict[str, Any]] = []
    seen: set[str] = set()
    for m in _STACK_AT_RE.finditer(blob):
        fqcn, method, file_name, line_s = m.group(1), m.group(2), m.group(3), m.group(4)
        cm = _simple_cm(fqcn, method)
        if not cm or cm in seen:
            continue
        # Prefer application frames (skip jdk/spring noise).
        low = fqcn.lower()
        if low.startswith(("java.", "javax.", "jdk.", "sun.", "org.springframework.")):
            continue
        seen.add(cm)
        try:
            line_no = int(line_s)
        except ValueError:
            line_no = 0
        frames.append(
            {
                "fqcn": fqcn,
                "method": method,
                "file": file_name,
                "line": line_no,
                "class_method": cm,
            }
        )
        if len(frames) >= 12:
            break

    cms = [str(f["class_method"]) for f in frames if f.get("class_method")]
    for m in _CM_RE.finditer(blob):
        if m.group(1) not in cms:
            cms.append(m.group(1))
    return {
        "exceptions": exceptions[:6],
        "frames": frames,
        "class_methods": cms[:16],
        "has_stack": bool(frames) or bool(exceptions),
    }


def ingest_signals(question: str, evidence: dict[str, Any]) -> None:
    """Fill evidence with log text + parsed stack + lexical signals."""
    q = question or ""
    paths = extract_log_paths(q)
    log_text = ""
    used_path = ""
    for path in paths:
        log_text = read_log_file(path)
        if log_text:
            used_path = path
            break
    blob = q
    if log_text:
        blob = q + "\n" + log_text
    parsed = parse_stack(blob)
    cms = list(parsed.get("class_methods") or [])
    words = _EXC_WORD_RE.findall(q)
    signals = list(dict.fromkeys(cms + [w for w in words if w]))[:24]
    evidence["log_paths"] = paths
    evidence["log_path_used"] = used_path
    evidence["log_text_chars"] = len(log_text)
    evidence["stack"] = parsed
    evidence["signals"] = signals
    if parsed.get("class_methods"):
        evidence["stack_cms"] = list(parsed["class_methods"])[:12]


def _lookup_symbol_id(project_id: str, class_method: str) -> str:
    if "#" not in (class_method or ""):
        return ""
    simple, meth = class_method.split("#", 1)
    simple = simple.split(".")[-1]
    meth = meth.split("(")[0]
    try:
        from app.agent.explore_tools import get_symbol_code

        res = get_symbol_code(project_id, fqcn=simple, method_name=meth)
        if not res.get("ok"):
            return ""
        sym = res.get("symbol") or {}
        return str(sym.get("symbol_id") or "")
    except Exception:
        logger.debug("diagnose symbol lookup failed", exc_info=True)
        return ""


def walk_callers(project_id: str, class_method: str, *, limit: int = 8) -> list[dict[str, Any]]:
    sid = _lookup_symbol_id(project_id, class_method)
    if not sid:
        return []
    try:
        from app.agent.explore_tools import list_callers

        out = list_callers(project_id, sid, limit=limit)
        neighbors = out.get("neighbors") or []
        brief: list[dict[str, Any]] = []
        for n in neighbors[:limit]:
            if not isinstance(n, dict):
                continue
            fqcn = str(n.get("fqcn") or "")
            name = str(n.get("name") or "")
            cm = ""
            if fqcn and name:
                cm = f"{fqcn.split('.')[-1]}#{name}"
            brief.append(
                {
                    "class_method": cm,
                    "fqcn": fqcn,
                    "name": name,
                    "rel": n.get("rel"),
                    "symbol_id": n.get("symbol_id"),
                }
            )
        return brief
    except Exception:
        logger.debug("diagnose walk_callers failed", exc_info=True)
        return []


def correlate_entry(
    project_id: str, question: str, suspects: list[str]
) -> dict[str, Any]:
    entry = ""
    spine: list[str] = []
    try:
        from app.agent.ask_retrieval import (
            _calls_closure_cms,
            resolve_flow_entry_for_question,
        )

        search_q = question
        if suspects:
            search_q = f"{question} {suspects[0]}"
        entry = resolve_flow_entry_for_question(project_id, search_q) or ""
        if entry:
            spine = _calls_closure_cms(project_id, entry)[:40]
    except Exception:
        logger.debug("diagnose correlate_entry failed", exc_info=True)
    on_spine = [s for s in suspects if s in set(spine)]
    return {"entry": entry, "spine": spine[:20], "on_spine": on_spine}


def map_and_read(project_id: str, question: str, evidence: dict[str, Any]) -> None:
    from app.agent.ask_retrieval import _search_hit_cms_for_question
    from app.agent.explore_source import _fetch_method_source_body

    stack_cms = [
        str(x)
        for x in (evidence.get("stack_cms") or [])
        if isinstance(x, str) and "#" in x
    ]
    hits: list[str] = []
    try:
        hits = _search_hit_cms_for_question(project_id, question, top_k=10)
    except Exception:
        logger.debug("diagnose search hits failed", exc_info=True)
    suspects = list(dict.fromkeys(stack_cms + list(evidence.get("signals") or []) + hits))
    suspects = [s for s in suspects if isinstance(s, str) and "#" in s][:8]
    evidence["suspects"] = suspects

    snippets: list[dict[str, Any]] = []
    for cm in suspects[:4]:
        try:
            code, path = _fetch_method_source_body(project_id, cm, max_chars=3500)
        except Exception:
            code, path = "", None
        if code:
            snippets.append({"class_method": cm, "path": path or "", "code": code})
    evidence["suspect_snippets"] = snippets

    callers: dict[str, list[dict[str, Any]]] = {}
    for cm in suspects[:3]:
        walked = walk_callers(project_id, cm)
        if walked:
            callers[cm] = walked
    evidence["callers"] = callers

    corr = correlate_entry(project_id, question, suspects)
    evidence["diagnose_entry"] = corr.get("entry") or ""
    evidence["diagnose_on_spine"] = corr.get("on_spine") or []
    if corr.get("entry") and not evidence.get("entry"):
        evidence["entry"] = corr["entry"]


def render_diagnose_answer(question: str, evidence: dict[str, Any]) -> str:
    snippets = evidence.get("suspect_snippets") or []
    suspects = evidence.get("suspects") or []
    stack = evidence.get("stack") if isinstance(evidence.get("stack"), dict) else {}
    exceptions = list(stack.get("exceptions") or [])
    frames = list(stack.get("frames") or [])
    entry = str(evidence.get("diagnose_entry") or evidence.get("entry") or "")
    on_spine = list(evidence.get("diagnose_on_spine") or [])
    callers = evidence.get("callers") if isinstance(evidence.get("callers"), dict) else {}
    used_path = str(evidence.get("log_path_used") or "")

    if not suspects and not snippets and not exceptions and not frames:
        return (
            "未能从问题/日志信号映射到可核对的代码符号，证据不足，"
            "请补充完整堆栈、异常类名，或给出可读取的日志路径后再问。"
        )

    lines = ["## 诊断（基于日志/堆栈与代码证据）", ""]
    if used_path:
        lines.append(f"已读取日志：`{used_path}`（末尾切片）")
        lines.append("")
    if exceptions:
        lines.append("异常链：")
        for ex in exceptions[:5]:
            msg = ex.get("message") or ""
            extra = f"：{msg}" if msg else ""
            lines.append(f"- `{ex.get('type')}`{extra}")
        lines.append("")
    if frames:
        lines.append("应用栈帧（已过滤 JDK/Spring）：")
        for f in frames[:8]:
            loc = f"{f.get('file')}:{f.get('line')}" if f.get("file") else ""
            lines.append(f"- `{f.get('class_method')}`" + (f"  {loc}" if loc else ""))
        lines.append("")
    if entry:
        lines.append(f"关联业务入口：`{entry}`")
        if on_spine:
            lines.append(
                "主链命中：" + "、".join(f"`{s}`" for s in on_spine[:6])
            )
        else:
            lines.append("未在该入口 CALLS 闭包中直接命中嫌疑方法（可能是工具类或异步线程）。")
        lines.append("")
    if suspects:
        lines.append("嫌疑符号：" + "、".join(f"`{s}`" for s in suspects[:8]))
        lines.append("")
    for cm, neigh in list(callers.items())[:3]:
        names = [n.get("class_method") or n.get("name") for n in neigh[:5] if n]
        names = [x for x in names if x]
        if names:
            lines.append(f"谁调用了 `{cm}`：" + "、".join(f"`{x}`" for x in names))
            lines.append("")
    for s in snippets:
        lines.append(f"### `{s['class_method']}`")
        if s.get("path"):
            lines.append(f"文件：`{s['path']}`")
        lines.append("")
        lines.append("```java")
        lines.append(str(s.get("code") or "").rstrip())
        lines.append("```")
        lines.append("")
    lines.append(
        "说明：以上为与日志/堆栈最相关的已读代码；"
        "若需断言线上根因，请提供完整 caused-by 链与请求上下文。"
    )
    return "\n".join(lines)
