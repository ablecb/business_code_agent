"""Scaffold a new HTTP API from global KB specs, matching project Controller style."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from app.core.config import ROOT_DIR
from app.db.mysql import get_session
from app.knowledge.retrieve import search_knowledge
from app.models.orm import KbChunkORM, KbDocumentORM
from app.services.projects import get_project

logger = logging.getLogger("app.knowledge.add_api")


def wants_add_api(question: str) -> bool:
    """Deprecated product gate — always False.

    Whether the user wants a new API / new Controller method / new VO class
    is decided by the LLM goal contract (``goal=add_api``), not Chinese/English
    keyword matching. Kept as a stub for call-site compatibility.
    """
    del question
    return False


def load_kb_spec_for_question(question: str, *, top_k: int = 6) -> dict[str, Any]:
    """Pull KB hits + full text of the best-matching document."""
    q = (question or "").strip()
    hits = search_knowledge(q, top_k=top_k)
    # Prefer docs whose title/filename appears in the question (e.g. 登录)
    preferred_id = None
    with get_session() as session:
        docs = (
            session.query(KbDocumentORM)
            .filter(KbDocumentORM.status == "ready")
            .order_by(KbDocumentORM.updated_at.desc())
            .limit(50)
            .all()
        )
        for d in docs:
            title = d.title or ""
            fname = d.filename or ""
            stem = Path(fname).stem
            for token in (title, stem):
                if token and len(token) >= 2 and token in q:
                    preferred_id = d.id
                    break
            if preferred_id:
                break
        if not preferred_id and hits:
            preferred_id = str(hits[0].get("document_id") or "") or None
        if not preferred_id and len(docs) == 1:
            preferred_id = docs[0].id

        full_text = ""
        title = ""
        filename = ""
        if preferred_id:
            doc = session.get(KbDocumentORM, preferred_id)
            if doc:
                title = doc.title or ""
                filename = doc.filename or ""
            chunks = (
                session.query(KbChunkORM)
                .filter(KbChunkORM.document_id == preferred_id)
                .order_by(KbChunkORM.chunk_index.asc())
                .all()
            )
            full_text = "\n\n".join(c.text or "" for c in chunks).strip()

    parsed = parse_api_spec(full_text)
    return {
        "document_id": preferred_id or "",
        "title": title,
        "filename": filename,
        "full_text": full_text[:12000],
        "hits": hits,
        "parsed": parsed,
    }


def _split_req_resp_sections(text: str) -> tuple[str, str]:
    """Split API doc into request vs response parts to avoid field bleed."""
    t = text or ""
    m = re.search(
        r"(返回示例|响应示例|返回参数|响应参数|response\s*example|返回结果)",
        t,
        re.I,
    )
    if m:
        return t[: m.start()], t[m.start() :]
    blobs = list(re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", t))
    if len(blobs) >= 2:
        return t[: blobs[1].start()], t[blobs[1].start() :]
    return t, ""


def _parse_field_table(section: str) -> list[dict[str, str]]:
    """Parse markdown pipe tables or tab-separated field tables."""
    fields: list[dict[str, str]] = []
    header_names = {"参数", "参数名", "字段", "字段名", "field"}
    type_tokens = {
        "string",
        "int",
        "integer",
        "long",
        "number",
        "decimal",
        "boolean",
        "bool",
        "list",
        "array",
        "类型",
        "type",
    }
    for line in (section or "").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or set(raw) <= {"-", "|", ":", " "}:
            continue
        if "|" in raw:
            cells = [c.strip() for c in re.split(r"\|", raw) if c.strip()]
        elif "\t" in raw:
            cells = [c.strip() for c in raw.split("\t") if c.strip()]
        else:
            cells = re.split(r"\s{2,}|\s+", raw)
            if len(cells) < 3 or cells[1].lower() not in type_tokens:
                continue
        if len(cells) < 2:
            continue
        name = cells[0]
        if not re.match(r"^[A-Za-z][A-Za-z0-9_]*$", name):
            continue
        # Header row: 参数名 | 类型 | ...
        if name.lower() in header_names:
            continue
        if name.lower() == "name" and cells[1].lower() in {"类型", "type"}:
            continue
        fields.append(
            {
                "name": name,
                "type": cells[1] if len(cells) > 1 else "string",
                "required": cells[2] if len(cells) > 2 else "",
                "desc": cells[3] if len(cells) > 3 else "",
            }
        )
    return fields


def _json_keys_top_level(blob: str) -> list[str]:
    keys: list[str] = []
    for name in re.findall(r'"([A-Za-z][A-Za-z0-9_]{1,40})"\s*:', blob or ""):
        if name in {"code", "msg", "message", "data", "header", "body"}:
            continue
        if name not in keys:
            keys.append(name)
    return keys


def parse_api_spec(text: str) -> dict[str, Any]:
    """Best-effort parse of markdown/API table docs (path/method/fields)."""
    t = text or ""
    path = ""
    method = "POST"
    m = re.search(r"(?:请求地址|路径|path)\s*[:：]\s*`?(/[A-Za-z0-9_/{}\-]+)`?", t, re.I)
    if m:
        path = m.group(1)
    if not path:
        m = re.search(r"`(/[A-Za-z0-9_/{}\-]+)`", t)
        if m:
            path = m.group(1)
    if not path and re.search(r"登录|login", t, re.I):
        path = "/login"
    mm = re.search(r"(?:请求方式|method)\s*[:：]\s*`?(GET|POST|PUT|DELETE|PATCH)`?", t, re.I)
    if mm:
        method = mm.group(1).upper()

    req_sec, resp_sec = _split_req_resp_sections(t)
    # Prefer field table under 请求参数
    req_table_sec = req_sec
    rm = re.search(r"请求参数|入参|Body\s*JSON|请求体", req_sec, re.I)
    if rm:
        req_table_sec = req_sec[rm.start() :]
    fields = _parse_field_table(req_table_sec)
    if not fields:
        # JSON request example keys only (not response)
        ex = re.search(
            r"请求示例[\s\S]{0,40}?(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})",
            req_sec,
            re.I,
        )
        blob = ex.group(1) if ex else ""
        if not blob:
            # first JSON in request section
            jm = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", req_sec)
            blob = jm.group(0) if jm else ""
        for name in _json_keys_top_level(blob):
            fields.append({"name": name, "type": "string", "required": "", "desc": ""})

    resp_fields: list[dict[str, str]] = []
    if resp_sec:
        # data object keys from response example
        dm = re.search(
            r'"data"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',
            resp_sec,
            re.I | re.S,
        )
        data_blob = dm.group(1) if dm else ""
        if not data_blob:
            jm = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", resp_sec)
            data_blob = jm.group(0) if jm else ""
        for name in _json_keys_top_level(data_blob):
            typ = "list" if name in {"records", "list", "items"} else (
                "long" if name == "total" else "string"
            )
            resp_fields.append({"name": name, "type": typ, "required": "", "desc": name})
        # List envelope: keep only data 顶层（total/records…），勿把 records[] 里的字段摊进 VO
        envelope = {"total", "records", "list", "items", "token", "success", "count", "pageNum", "pageSize"}
        names = [f.get("name") for f in resp_fields if f.get("name")]
        if any(n in {"records", "list", "items"} for n in names):
            resp_fields = [f for f in resp_fields if f.get("name") in envelope]

    # login defaults
    if re.search(r"登录|login", t, re.I) and not fields:
        fields = [
            {"name": "username", "type": "string", "required": "是", "desc": "账号"},
            {"name": "password", "type": "string", "required": "是", "desc": "密码"},
        ]
    if re.search(r"登录|login", t, re.I) and not resp_fields:
        resp_fields = [{"name": "token", "type": "string", "desc": "登录令牌"}]

    api_name = "login" if re.search(r"登录|login", t, re.I) else (
        path.strip("/").replace("/", "_") or "newApi"
    )
    summary = "用户登录" if re.search(r"登录|login", t, re.I) else "新接口"
    sm = re.search(r"(?:接口详情|#+)\s*([^\n]{2,40})", t)
    if sm and not re.search(r"登录|login", t, re.I):
        cand = sm.group(1).strip()
        if cand and "API" not in cand.upper():
            summary = cand
    # e.g. line "条件查询商品"
    for line in t.splitlines():
        s = line.strip()
        if re.match(r"^条件查询|^查询|^新增|^创建|^删除", s) and len(s) <= 40:
            summary = s
            break

    return {
        "path": path or "/api/new",
        "method": method,
        "fields": fields[:20],
        "resp_fields": resp_fields[:20],
        "api_name": re.sub(r"[^A-Za-z0-9_]", "", api_name) or "newApi",
        "summary": summary,
    }


def _is_typed_request_style(code: str) -> bool:
    """True when Controller uses Request<SomeDto/Vo> rather than Request<Map…>."""
    if "Request<Map" in (code or ""):
        # Still allow if there is also a typed Request elsewhere in file
        if not re.search(
            r"Request<\s*(?!Map\b)[A-Z][A-Za-z0-9_]*(?:Dto|Vo|Request|InDTO|OutDTO)?\s*>",
            code or "",
        ):
            return False
    return bool(
        re.search(
            r"Request<\s*[A-Z][A-Za-z0-9_]*(?:Dto|Vo|Request|InDTO|InVO)\s*>",
            code or "",
        )
    )


def _score_controller_style(code: str) -> tuple[int, bool]:
    """Rank a Controller as style exemplar. Higher = better project-style match.

    Prefers typed Request<XxxDto/Vo> over Request<Map…>; no hardcoded class names.
    """
    c = code or ""
    typed = _is_typed_request_style(c)
    score = 0
    if "@RestController" in c or "@Controller" in c:
        score += 2
    if "@RequestMapping" in c:
        score += 2
    if "Request<" in c:
        score += 3
    if "Response<" in c:
        score += 2
    if typed:
        score += 10
    elif "Request<Map" in c:
        score += 1
    if "ResponseBuilder" in c:
        score += 3
    if "@RequestBody" in c:
        score += 2
    if "@Resource" in c or "@Autowired" in c:
        score += 1
    if "ExceptionEnum" in c or "BusinessException" in c:
        score += 1
    if "@ApiOperation" in c:
        score += 1
    # Prefer non-test / non-generated noise lightly via size of HTTP method surface
    if re.search(r"RequestMethod\.(GET|POST|PUT|DELETE|PATCH)", c):
        score += 1
    return score, typed


def find_style_exemplar(project_id: str) -> dict[str, Any]:
    """Locate a RestController by scanning the project; prefer typed Request style."""
    root = ""
    try:
        proj = get_project(project_id)
        root = str((proj or {}).get("root_path") or "")
    except Exception:
        logger.debug("get_project failed", exc_info=True)
    if not root:
        cand = ROOT_DIR / "java_project" / project_id
        root = str(cand) if cand.is_dir() else ""
    base = Path(root) if root else None
    if not base or not base.is_dir():
        return {"ok": False, "error": "未找到项目根目录"}

    def _pack(path: Path, code: str, *, typed: bool) -> dict[str, Any]:
        rel = str(path.relative_to(base)).replace("\\", "/")
        notes = _infer_style_notes(code)
        if typed:
            notes = [
                "统一封装：`Request<实体/DTO>` / `Response<实体/DTO>` + `ResponseBuilder`",
                *notes,
            ]
        uniq: list[str] = []
        for n in notes:
            if n not in uniq:
                uniq.append(n)
        return {
            "ok": True,
            "path": rel,
            "class_name": path.stem,
            "typed_body": typed,
            "code_excerpt": _controller_excerpt(code),
            "style_notes": uniq,
        }

    best: tuple[int, Path, str, bool] | None = None
    for path in base.rglob("*Controller.java"):
        # Skip test trees
        parts = {p.lower() for p in path.parts}
        if "test" in parts or "tests" in parts or "target" in parts:
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        if "Request<" not in code and "Response<" not in code and "@RequestMapping" not in code:
            continue
        score, typed = _score_controller_style(code)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, path, code, typed)

    if best is None:
        return {"ok": False, "error": "未找到可用的 Controller 风格样例"}
    _, path, code, typed = best
    return _pack(path, code, typed=typed)


_FEIGN_IMPORT_RE = re.compile(r"import\s+[^;]*FeignClient[^;]*;")
_FEIGN_ANN_RE = re.compile(r"@FeignClient\s*\(([^)]*)\)")
_FEIGN_NAME_RE = re.compile(r'(?:name|value|serviceId)\s*=\s*"([^"]+)"')
_FEIGN_CONTEXT_RE = re.compile(r'contextId\s*=\s*"([^"]+)"')
_FEIGN_MAPPING_RE = re.compile(
    r"@(?:PostMapping|GetMapping|PutMapping|DeleteMapping|RequestMapping)"
    r"(?:\(\s*(?:value\s*=\s*)?\{?\"?([^\"}\s]+)\"?\}?)?[^)]*\)?"
)


def find_remote_call_evidence(
    project_id: str,
    service_hint: str = "",
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Locate Feign clients for a service (e.g. acc-service) + their method signatures.

    Scans project Java sources for ``@FeignClient(name="acc-service")`` interfaces,
    returns the interface FQCN, injected field convention and a few method excerpts
    (``Response<XxxResp> xxx(@RequestBody Request<XxxReq> request)`` style) so the
    scaffold can call the remote service exactly like the project does.
    """
    hint = (service_hint or "").strip().lower()
    root = ""
    try:
        proj = get_project(project_id)
        root = str((proj or {}).get("root_path") or "")
    except Exception:
        logger.debug("get_project failed", exc_info=True)
    if not root:
        cand = ROOT_DIR / "java_project" / project_id
        root = str(cand) if cand.is_dir() else ""
    base = Path(root) if root else None
    if not base or not base.is_dir():
        return {"ok": False, "error": "未找到项目根目录", "clients": []}

    clients: list[dict[str, Any]] = []
    for path in base.rglob("*.java"):
        parts = {p.lower() for p in path.parts}
        if "test" in parts or "tests" in parts or "target" in parts:
            continue
        try:
            code = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        m = _FEIGN_ANN_RE.search(code)
        if not m:
            continue
        ann = m.group(1)
        name_m = _FEIGN_NAME_RE.search(ann)
        ctx_m = _FEIGN_CONTEXT_RE.search(ann)
        svc_name = name_m.group(1) if name_m else (ctx_m.group(1) if ctx_m else "")
        if hint and hint not in svc_name.lower() and hint not in str(path).lower():
            continue
        cls = path.stem
        pkg = ""
        pm = re.search(r"^package\s+([\w.]+);", code, re.M)
        if pm:
            pkg = pm.group(1)
        methods: list[str] = []
        for mm in re.finditer(
            r"@(?:PostMapping|GetMapping|PutMapping|DeleteMapping|RequestMapping)[^\n]*\n"
            r"\s*([A-Za-z0-9_<>\[\],\s.]+)\s+([a-zA-Z][a-zA-Z0-9_]*)"
            r"\s*\(([^)]*)\)",
            code,
        ):
            ret = mm.group(1).strip().replace("\n", " ")
            meth = mm.group(2)
            args = re.sub(r"\s+", " ", mm.group(3).strip())
            methods.append(f"{ret} {meth}({args})")
        rel = str(path.relative_to(base)).replace("\\", "/")
        clients.append(
            {
                "class_name": cls,
                "fqcn": f"{pkg}.{cls}" if pkg else cls,
                "service_name": svc_name,
                "context_id": ctx_m.group(1) if ctx_m else "",
                "path": rel,
                "fallback": bool(re.search(r"fallback\s*=", ann)),
                "methods": methods[:12],
                "inject_field": f"{cls[:1].lower()}{cls[1:]}",
            }
        )
        if len(clients) >= top_k:
            break
    ok = bool(clients)
    return {
        "ok": ok,
        "clients": clients,
        "error": "" if ok else "未找到匹配的 @FeignClient",
    }


def _controller_excerpt(code: str, *, max_lines: int = 55) -> str:
    lines = (code or "").splitlines()
    # Prefer first method body region
    start = 0
    for i, ln in enumerate(lines):
        if "@ApiOperation" in ln or "@RequestMapping" in ln and "class" not in ln:
            start = max(0, i - 8)
            break
    chunk = lines[start : start + max_lines]
    return "\n".join(chunk)


def _infer_style_notes(code: str) -> list[str]:
    notes: list[str] = []
    if _is_typed_request_style(code):
        notes.append("入参：`@RequestBody Request<XxxDto/Vo>`（强类型实体，非 Map）")
    elif "Request<" in code and "Response<" in code:
        notes.append("统一封装：`Request<T>` / `Response<T>` + `ResponseBuilder`")
    if "ResponseBuilder.buildSuccess" in code:
        notes.append("成功：`ResponseBuilder.buildSuccess(...)`")
    if "LoanBusinessException" in code or "ExceptionEnum" in code:
        notes.append("参数校验失败抛 `LoanBusinessException` / `ExceptionEnum`")
    if "@RequestBody" in code and not _is_typed_request_style(code):
        notes.append("入参：`@RequestBody Request<...>`")
    if "@RestController" in code:
        notes.append("类注解：`@RestController` + `@RequestMapping`")
    if "@Resource" in code:
        notes.append("依赖注入：`@Resource` Service")
    if "lombok" in code.lower() or "@Data" in code:
        notes.append("DTO/VO：`lombok @Data`（与 `loan-service-api` 一致）")
    return notes or ["沿用项目现有 Controller 分层与命名"]


def _java_type(doc_type: str, name: str = "") -> str:
    t = (doc_type or "string").strip().lower()
    n = (name or "").lower()
    if t in {"int", "integer"} or n.endswith(("num", "size", "count", "page")):
        return "Integer"
    if t in {"long"}:
        return "Long"
    if t in {"number", "decimal", "double", "float"} or "price" in n or "amount" in n:
        return "BigDecimal"
    if t in {"bool", "boolean"}:
        return "Boolean"
    if t in {"list", "array"} or n in {"records", "list", "items"}:
        return "List<Map<String, Object>>"
    return "String"


def _render_vo_java(
    *,
    class_name: str,
    fields: list[dict[str, str]],
    package: str = "com.demo.loan.dto",
) -> str:
    needs_bd = any(
        _java_type(f.get("type") or "", f.get("name") or "") == "BigDecimal"
        for f in fields
    )
    needs_list = any(
        _java_type(f.get("type") or "", f.get("name") or "").startswith("List")
        for f in fields
    )
    lines = [
        f"package {package};",
        "",
        "import lombok.Data;",
    ]
    if needs_bd:
        lines.append("import java.math.BigDecimal;")
    if needs_list:
        lines.extend(["import java.util.List;", "import java.util.Map;"])
    lines.extend(["", "@Data", f"public class {class_name} {{"])
    for f in fields:
        n = f.get("name") or "field"
        d = f.get("desc") or n
        jt = _java_type(f.get("type") or "", n)
        lines.append(f"    /** {d} */")
        lines.append(f"    private {jt} {n};")
        lines.append("")
    if len(lines) > 0 and lines[-1] == "":
        lines.pop()
    lines.append("}")
    return "\n".join(lines)


# —— 阶段1：ServiceImpl 按用户业务步骤分步生成 ——
# business_steps 由 Goal LLM 从用户原话解析（validate/query/call/log/return），
# 每步要么引用仓库证据（Feign 客户端/实体/Mapper），要么落 TODO 并注明缺什么，
# 禁止编造仓库里不存在的类/服务。
_STEP_OP_RE = re.compile(r"^(validate|query|call|log|return|other)$")


def _find_feign_client_for_service(
    project_id: str, service_hint: str
) -> dict[str, Any] | None:
    """Best-effort: locate a Feign client whose name/context/path matches the hint."""
    if not project_id or not (service_hint or "").strip():
        return None
    ev = find_remote_call_evidence(project_id, service_hint, top_k=3)
    clients = ev.get("clients") or []
    if not clients:
        return None
    hint_l = service_hint.lower()
    # Prefer exact name match, then contextId, then class-name containment.
    exact = [c for c in clients if hint_l == str(c.get("service_name") or "").lower()]
    if exact:
        return exact[0]
    ctx = [c for c in clients if hint_l == str(c.get("context_id") or "").lower()]
    if ctx:
        return ctx[0]
    return clients[0]


def _render_validate_step(step: dict[str, Any], req_vo: str) -> list[str]:
    """校验段：项目惯用 LoanAssertUtil / ExceptionEnum.ERROR_MUST_LACK 风格。

    仅当步骤 target 里有「校验/必填/非空」语义时产出断言；否则给出轻量风格提示。
    """
    target = str(step.get("target") or "")
    lines = [
        "        // 参数校验（对齐项目 LoanAssertUtil + ExceptionEnum 风格）",
        f"        LoanAssertUtil.assertNotNull(req, ExceptionEnum.ERROR_MUST_LACK.getInnerCode(),",
        f'                String.format(ExceptionEnum.ERROR_MUST_LACK.getInnerMsg(), "请求体不能为空"));',
    ]
    if target and re.search(r"校验|必填|非空|notBlank|required", target, re.I):
        lines.append(
            "        // TODO: 按接口文档补充必填字段断言（如 LoanAssertUtil.assertBlank(...)）"
        )
    return lines


def _render_query_step(
    step: dict[str, Any],
    req_vo: str,
    resp_vo: str,
    method_name: str,
) -> list[str]:
    """查询段：查库。若 business_steps 里有 query 步骤，给出 MyBatis-Plus 风格骨架。

    无法确定实体/Mapper 时（知识库文档通常不含仓库实体名），落 TODO + 注明缺什么证据，
    不编造表/实体名。
    """
    target = str(step.get("target") or "")
    lines = [
        "        // 查询数据（MyBatis-Plus 风格，证据不足处标 TODO）",
        "        // TODO: 指定真实实体 + Mapper 后替换：",
    ]
    if target:
        lines.append(f"        //   目标：{target}")
    lines.extend(
        [
            "        LambdaQueryWrapper<?> qw = new LambdaQueryWrapper<>();",
            "        // TODO: qw.eq(XxxEntity::getXxx, req.getXxx()); 按业务键拼装条件",
            "        // TODO: 调用 XxxMapper.selectList/selectPage 读取数据",
            "        Object bizData = null; // TODO: 查询结果",
        ]
    )
    return lines


def _render_call_step(
    step: dict[str, Any],
    project_id: str,
    req_vo: str,
    resp_vo: str,
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    """调用段：调外部服务（如 acc-service 的 FeignClient）。

    优先检索项目里真实 @FeignClient 方法签名，按项目惯用写法生成调用；
    检索不到时落 TODO + 注明服务名，不编造方法。
    返回 (方法体内代码行, client 摘要, 注入字段声明)；注入字段由外层放到类字段区，
    注解 @Qualifier/@Autowired 在字段上才有效（不能放在方法体内）。
    """
    target = str(step.get("target") or "")
    client = _find_feign_client_for_service(project_id, target) if project_id else None
    if not client:
        return (
            [
                "        // 调用外部服务",
                f"        // TODO: 检索到目标远程服务「{target or '(未指定)'}」后，按项目 @FeignClient 惯例注入并调用",
                "        // 例：@Qualifier(\"com.demo.loan.plugin.v2.AccClient\") @Autowired private AccClient accClient;",
                "        //     Request<XxxReqVo> remoteReq = new Request<>(); remoteReq.setBody(...);",
                "        //     Response<XxxRespVo> remoteResp = accClient.xxx(remoteReq);",
            ],
            None,
            None,
        )
    cls = client.get("class_name") or ""
    fqcn = client.get("fqcn") or ""
    field = client.get("inject_field") or (cls[:1].lower() + cls[1:])
    methods = client.get("methods") or []
    meth0 = ""
    # 优先匹配 target 关键词（如「查询放款」→ 方法名含 query/loan/select）；
    # 退而求其次取第一个方法。避免总是调用无关的首方法（如 clanQuery）。
    if methods:
        toks = [
            t.lower()
            for t in re.findall(r"[A-Za-z\u4e00-\u9fff][A-Za-z0-9_\u4e00-\u9fff]*", target)
            if len(t) >= 2 and t.lower() not in {"接口", "服务", "service", "调用", "acc"}
        ]
        scored: list[tuple[int, str]] = []
        for m in methods:
            rm = re.search(r"([A-Za-z][A-Za-z0-9_]*)\s*\(", m)
            if not rm:
                continue
            name = rm.group(1)
            sc = 0
            name_l = name.lower()
            for t in toks:
                if t in name_l:
                    sc += 10
                elif name_l.startswith(t) or t.startswith(name_l):
                    sc += 5
            if re.search(r"(?i)query|select|list|get", name):
                sc += 2
            scored.append((sc, name))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if scored and scored[0][0] > 0:
            meth0 = scored[0][1]
        else:
            rm = re.search(r"([A-Za-z][A-Za-z0-9_]*)\s*\(", methods[0])
            meth0 = rm.group(1) if rm else ""
    inject = {
        "cls": cls,
        "fqcn": fqcn,
        "field": field,
        "qualifier": f'@Qualifier("{fqcn}")',
    }
    lines = ["        // 调用外部服务（对齐项目 @FeignClient 惯例）"]
    if meth0:
        lines.extend(
            [
                f"        Request<{req_vo}> remoteReq = new Request<>();",
                "        remoteReq.setBody(req); // TODO: 按目标接口入参映射",
                f"        Response<?> remoteResp = {field}.{meth0}(remoteReq);",
                "        // TODO: 处理 remoteResp 结果（判空/取 body）",
            ]
        )
    else:
        lines.append(
            f"        // TODO: 调用 {field}.<目标方法>(Request<...> request) —— 见 {fqcn} 方法列表"
        )
    return lines, client, inject


def _render_log_step(step: dict[str, Any], prior_vars: list[str]) -> list[str]:
    """日志段：项目惯用 log.info + JSON.toJSONString 风格。

    prior_vars 为前面步骤产生的局部变量（bizData / remoteResp / result 等），
    优先打印真实变量，避免出现「JSON.toJSONString(结果)」这类伪代码。
    """
    target = str(step.get("target") or "")
    var = ""
    if prior_vars:
        var = prior_vars[-1]
    elif target and re.search(r"[A-Za-z][A-Za-z0-9_]*", target):
        var = re.search(r"[A-Za-z][A-Za-z0-9_]*", target).group(0)
    if var:
        return [
            f'        log.info("业务步骤执行结果 {var}: {{}}", JSON.toJSONString({var}));'
        ]
    return [
        "        // TODO: 打印上一步结果（log.info + JSON.toJSONString）",
    ]


def _render_return_step(step: dict[str, Any], resp_vo: str, prior_vars: list[str]) -> list[str]:
    """返回段：组装 RespVo 返回（项目 Request/Response 统一封装）。"""
    src = prior_vars[-1] if prior_vars else None
    lines = [f"        {resp_vo} result = new {resp_vo}();"]
    if src:
        lines.append(f"        // TODO: 将 {src} 结果装配到 result 字段")
    else:
        lines.append("        // TODO: 将 query/remote 结果装配到 result 字段")
    lines.append("        return result;")
    return lines


def render_service_impl_steps(
    *,
    project_id: str,
    business_steps: list[dict[str, Any]] | None,
    req_vo: str,
    resp_vo: str,
    method_name: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """按业务步骤生成 ServiceImpl 方法体；返回 (代码行, 每步证据/缺口摘要)。"""
    steps = [s for s in (business_steps or []) if isinstance(s, dict) and s.get("op")]
    if not steps:
        return [], []
    lines: list[str] = []
    notes: list[dict[str, Any]] = []
    prior_vars: list[str] = []
    injects: list[dict[str, Any]] = []
    for step in steps:
        op = str(step.get("op") or "").strip().lower()
        if op == "validate":
            seg = _render_validate_step(step, req_vo)
            notes.append({"op": op, "status": "ok", "note": "参数校验（项目断言风格）"})
        elif op == "query":
            seg = _render_query_step(step, req_vo, resp_vo, method_name)
            prior_vars.append("bizData")
            notes.append({"op": op, "status": "todo", "note": "查库（缺实体/Mapper 证据）"})
        elif op == "call":
            seg, client, inject = _render_call_step(step, project_id, req_vo, resp_vo)
            prior_vars.append("remoteResp")
            if inject and inject not in injects:
                injects.append(inject)
            notes.append(
                {
                    "op": op,
                    "status": "ok" if client else "todo",
                    "note": (
                        f"调 {client.get('service_name') or client.get('class_name') or step.get('target')}"
                        if client
                        else f"远程调用未定位（{step.get('target') or '未指定服务'}）"
                    ),
                }
            )
        elif op == "log":
            seg = _render_log_step(step, prior_vars)
            notes.append({"op": op, "status": "ok", "note": "日志（log.info + JSON）"})
        elif op == "return":
            seg = _render_return_step(step, resp_vo, prior_vars)
            notes.append({"op": op, "status": "ok", "note": "返回 RespVo"})
        else:
            seg = [
                f"        // TODO: 业务步骤「{step.get('target') or op}」",
                "        // 证据不足，待补充实现",
            ]
            notes.append({"op": op, "status": "todo", "note": f"{step.get('target') or op}（未归类步骤）"})
        if lines:
            lines.append("")
        lines.extend(seg)
    return lines, notes, injects


def render_new_api_plan(
    question: str,
    *,
    kb: dict[str, Any],
    style: dict[str, Any],
    business_steps: list[dict[str, Any]] | None = None,
    project_id: str = "",
    entity_only: bool = False,
) -> str:
    parsed = kb.get("parsed") or {}
    path = str(parsed.get("path") or "/api/new")
    method = str(parsed.get("method") or "POST")
    api_name = str(parsed.get("api_name") or "newApi")
    summary = str(parsed.get("summary") or "新接口")
    fields = list(parsed.get("fields") or [])
    # Always scaffold typed Request<ReqVo> / Response<RespVo> (project majority).
    # Map Controllers (e.g. CustomerController) are style references only, not body type.
    if not fields:
        fields = [{"name": "payload", "type": "string", "desc": "请求体（文档未解析到字段）", "required": "是"}]
    ctrl = _to_pascal(api_name) + "Controller"
    svc = _to_pascal(api_name) + "Service"
    svc_impl = svc + "Impl"
    req_vo = _to_pascal(api_name) + "ReqVo"
    resp_vo = _to_pascal(api_name) + "RespVo"
    segs = [s for s in path.strip("/").split("/") if s]
    if len(segs) <= 1:
        mapping = segs[0] if segs else api_name
        method_path = ""
    else:
        mapping = segs[0]
        method_path = "/".join(segs[1:])

    required = [
        f.get("name")
        for f in fields
        if f.get("name")
        and re.search(r"^(是|必填|Y|yes|true)$", str(f.get("required") or "").strip(), re.I)
    ]
    # Do not invent required fields when doc marks them optional / unknown.

    style_path = style.get("path") or "(未找到样例)"
    style_notes = style.get("style_notes") or []
    kb_title = kb.get("title") or kb.get("filename") or "知识库文档"
    method_name = _to_camel(api_name)

    # Response VO fields from KB JSON example when present
    resp_fields = list(parsed.get("resp_fields") or [])
    if not resp_fields:
        if re.search(r"total|records", str(kb.get("full_text") or ""), re.I):
            resp_fields = [
                {"name": "total", "type": "long", "desc": "总条数"},
                {"name": "records", "type": "list", "desc": "列表数据"},
            ]
        elif re.search(r"token", str(kb.get("full_text") or ""), re.I):
            resp_fields = [{"name": "token", "type": "string", "desc": "登录令牌"}]
        else:
            resp_fields = [{"name": "success", "type": "boolean", "desc": "是否成功"}]

    # —— 实体类专用模式：用户只要 Entity/VO/DTO，不输出 Controller/Service ——
    # （desktop-chat-d641bff7c836：「根据知识库 商品查询 帮我生成对应的 java 实体类」）
    if entity_only:
        kb_bits = [f"- 文档：`{kb_title}`"]
        if summary:
            kb_bits.append(f"- 说明：{summary}")
        if path and path != "/api/new":
            kb_bits.append(f"- 路径：`{path}` · 方法：`{method}`")
        lines = [
            "## 结论",
            "",
            f"按知识库《{kb_title}》生成对应 Java 实体类（仅实体类，不含 Controller/Service）。",
            "",
            "## 知识库要点",
            "",
            *kb_bits,
            "- 入参字段："
            + (
                "、".join(
                    f"`{f.get('name')}`"
                    + (f"（{f.get('desc')}）" if f.get("desc") else "")
                    for f in fields
                )
                or "（文档未解析到字段）"
            ),
            "",
            "## 实体类",
            "",
            f"### 1. 请求 VO `{req_vo}`",
            "",
            f"`loan-service-api/src/main/java/com/demo/loan/dto/{req_vo}.java`",
            "",
            "```java",
            _render_vo_java(class_name=req_vo, fields=fields),
            "```",
            "",
            f"### 2. 响应 VO `{resp_vo}`",
            "",
            f"`loan-service-api/src/main/java/com/demo/loan/dto/{resp_vo}.java`",
            "",
            "```java",
            _render_vo_java(class_name=resp_vo, fields=resp_fields),
            "```",
            "",
            "> 仅生成实体类；如需 Controller/Service 骨架请说明（例如「按知识库生成完整接口」）。",
        ]
        return "\n".join(lines)

    # —— 阶段1：ServiceImpl 按业务步骤生成（若用户给了编排步骤）——
    # 提前计算：ServiceImpl 代码块与「业务步骤实现」小节都要用到。
    impl_body_lines: list[str] = []
    step_notes: list[dict[str, Any]] = []
    impl_injects: list[dict[str, Any]] = []
    if business_steps:
        impl_body_lines, step_notes, impl_injects = render_service_impl_steps(
            project_id=project_id,
            business_steps=business_steps,
            req_vo=req_vo,
            resp_vo=resp_vo,
            method_name=method_name,
        )

    lines = [
        "## 结论",
        "",
        f"按知识库《{kb_title}》新增 HTTP 接口 `{method} {path}`，"
        f"对齐项目现有风格（样例：`{style_path}`）。",
        "",
        "## 知识库要点",
        "",
        f"- 文档：`{kb_title}`",
        f"- 路径：`{path}` · 方法：`{method}` · 说明：{summary}",
        "- 入参字段："
        + (
            "、".join(
                f"`{f.get('name')}`"
                + (f"（{f.get('desc')}）" if f.get("desc") else "")
                for f in fields
            )
            or "（文档未解析到字段）"
        ),
        "",
        "## 风格依据",
        "",
        "- **不是** LLM 自由仿写：`T2_style` 从仓库选 Controller 样例，"
        "`T3_scaffold` 用规则模板生成（优先 `Request<ReqVo>` / `Response<RespVo>`）。",
    ]
    for n in style_notes:
        lines.append(f"- {n}")
    if style.get("code_excerpt"):
        lines.extend(
            [
                "",
                f"样例摘录 `{style_path}`：",
                "",
                "```java",
                str(style.get("code_excerpt"))[:1800],
                "```",
            ]
        )

    lines.extend(["", "## 新增清单（建议落点）", ""])

    lines.extend(
        [
            f"### 1. 请求 VO `{req_vo}`",
            "",
            f"`loan-service-api/src/main/java/com/demo/loan/dto/{req_vo}.java`",
            "",
            "```java",
            _render_vo_java(class_name=req_vo, fields=fields),
            "```",
            "",
            f"### 2. 响应 VO `{resp_vo}`",
            "",
            f"`loan-service-api/src/main/java/com/demo/loan/dto/{resp_vo}.java`",
            "",
            "```java",
            _render_vo_java(class_name=resp_vo, fields=resp_fields),
            "```",
            "",
            f"### 3. Controller `{ctrl}`",
            "",
            f"`loan-service/src/main/java/com/demo/loan/controller/{ctrl}.java`",
            "",
            "```java",
            "package com.demo.loan.controller;",
            "",
            "import cn.hutool.core.util.ObjectUtil;",
            "import com.alibaba.fastjson.JSON;",
            "import com.demo.common.builder.ResponseBuilder;",
            "import com.demo.common.request.Request;",
            "import com.demo.common.response.Response;",
            "import com.demo.loan.common.enums.ExceptionEnum;",
            "import com.demo.loan.common.exception.LoanBusinessException;",
            f"import com.demo.loan.dto.{req_vo};",
            f"import com.demo.loan.dto.{resp_vo};",
            f"import com.demo.loan.service.{svc};",
            "import io.swagger.annotations.ApiOperation;",
            "import lombok.extern.slf4j.Slf4j;",
            "import org.springframework.web.bind.annotation.*;",
            "",
            "import javax.annotation.Resource;",
            "",
            "@Slf4j",
            "@RestController",
            f'@RequestMapping("{mapping}")',
            f"public class {ctrl} {{",
            "",
            "    @Resource",
            f"    private {svc} {_to_camel(svc)};",
            "",
            f'    @ApiOperation(value = "{summary}")',
            (
                f'    @RequestMapping(value = "{method_path}", method = RequestMethod.{method})'
                if method_path
                else f"    @RequestMapping(method = RequestMethod.{method})"
            ),
            "    @ResponseBody",
            f"    public Response<{resp_vo}> {method_name}("
            f"@RequestBody Request<{req_vo}> request) {{",
            f'        log.info("methodName:{method_name} request:{{}}", JSON.toJSONString(request));',
            "        if (ObjectUtil.isEmpty(request) || ObjectUtil.isEmpty(request.getBody())) {",
            "            throw new LoanBusinessException(ExceptionEnum.ERROR_MUST_LACK.getInnerCode(),",
            '                    String.format(ExceptionEnum.ERROR_MUST_LACK.getInnerMsg(), "请求体不能为空"));',
            "        }",
            f"        {req_vo} body = request.getBody();",
        ]
    )
    for n in required:
        if not n:
            continue
        getter = "get" + n[:1].upper() + n[1:]
        lines.append(f"        if (ObjectUtil.isEmpty(body.{getter}())) {{")
        lines.append(
            "            throw new LoanBusinessException(ExceptionEnum.ERROR_MUST_LACK.getInnerCode(),"
        )
        lines.append(
            f'                    String.format(ExceptionEnum.ERROR_MUST_LACK.getInnerMsg(), "{n}不能为空"));'
        )
        lines.append("        }")
    lines.extend(
        [
            f"        {resp_vo} result = {_to_camel(svc)}.{method_name}(body);",
            f"        Response<{resp_vo}> response = ResponseBuilder.buildSuccess(result);",
            f'        log.info("methodName:{method_name} response:{{}}", JSON.toJSONString(response));',
            "        return response;",
            "    }",
            "}",
            "```",
            "",
            f"### 4. Service 接口 `{svc}`",
            "",
            f"`loan-service/src/main/java/com/demo/loan/service/{svc}.java`",
            "",
            "```java",
            "package com.demo.loan.service;",
            "",
            f"import com.demo.loan.dto.{req_vo};",
            f"import com.demo.loan.dto.{resp_vo};",
            "",
            f"public interface {svc} {{",
            f"    {resp_vo} {method_name}({req_vo} req);",
            "}",
            "```",
            "",
            f"### 5. Service 实现 `{svc_impl}`",
            "",
            f"`loan-service/src/main/java/com/demo/loan/service/impl/{svc_impl}.java`",
            "",
            "```java",
            "package com.demo.loan.service.impl;",
            "",
            f"import com.demo.loan.dto.{req_vo};",
            f"import com.demo.loan.dto.{resp_vo};",
            f"import com.demo.loan.service.{svc};",
            "import org.springframework.stereotype.Service;",
            "",
            "import java.util.ArrayList;",
            "import java.util.List;",
            "import java.util.Map;",
            "",
            "@Service",
            f"public class {svc_impl} implements {svc} {{",
        ]
    )
    if impl_injects:
        # 远程客户端注入字段（@Qualifier/@Autowired 必须放类字段，不能放方法体）
        lines.append("")
        for inj in impl_injects:
            q = str(inj.get("qualifier") or "")
            lines.append(f"    {q}" if q else "    @Qualifier(\"\")")
            lines.append("    @Autowired")
            lines.append(f"    private {inj.get('cls')} {inj.get('field')};")
            lines.append("")
    lines.extend(
        [
            "    @Override",
            f"    public {resp_vo} {method_name}({req_vo} req) {{",
        ]
    )
    if impl_body_lines:
        # 有业务步骤：按步骤生成方法体（校验/查库/调外部/日志/返回）
        lines.extend(impl_body_lines)
        lines.extend(
            [
                "    }",
                "}",
                "```",
            ]
        )
    else:
        # 无业务步骤：保留字段级占位（total/records/token 等）
        lines.append(f"        {resp_vo} resp = new {resp_vo}();")
        lines.append("        // TODO: 按知识库约定查库/拼装；以下为结构占位")
        if any(f.get("name") == "total" for f in resp_fields):
            lines.append("        resp.setTotal(0L);")
        if any(f.get("name") == "records" for f in resp_fields):
            lines.extend(
                [
                    "        List<Map<String, Object>> records = new ArrayList<>();",
                    "        resp.setRecords(records);",
                ]
            )
        if any(f.get("name") == "token" for f in resp_fields):
            lines.append('        resp.setToken("xxxxxxxxxxxx");')
        lines.extend(
            [
                "        return resp;",
                "    }",
                "}",
                "```",
            ]
        )

    http_path = "/" + mapping + (f"/{method_path}" if method_path else "")

    if step_notes:
        lines.extend(
            [
                "",
                "## ServiceImpl 业务步骤实现",
                "",
            ]
        )
        for n in step_notes:
            mark = "✅" if n.get("status") == "ok" else "⚠️ TODO"
            lines.append(f"- {mark} `{n.get('op')}`：{n.get('note')}")
        lines.append("")

    lines.extend(
        [
            "## 联调说明",
            "",
            f"- 调用：`{method} " + "{baseUrl}" + f"{http_path}`",
            "- Body 使用项目统一包装：",
            "```json",
            '{',
            '  "header": {},',
            '  "body": { ...知识库请求字段... }',
            "}",
            "```",
            f"- Java 侧类型：`Request<{req_vo}>` → `Response<{resp_vo}>`"
            + (f"（对齐样例 `{style_path}`）" if style_path else ""),
            "",
            "## 知识库原文摘录",
            "",
            "```markdown",
            (kb.get("full_text") or "")[:2000],
            "```",
        ]
    )
    return "\n".join(lines)


def _to_pascal(name: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", name or "")
    return "".join(p[:1].upper() + p[1:] for p in parts if p) or "New"


def _to_camel(name: str) -> str:
    p = _to_pascal(name)
    return p[:1].lower() + p[1:] if p else "newApi"
