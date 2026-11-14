"""Log-driven code problem diagnosis (日志 → 代码问题定位).

Pipeline: intent signal → anchor extraction → anchor→symbol retrieval →
real-source evidence gathering → structured LLM diagnosis (log_diagnose prompt).

Covers: param-missing (XXL-Job handleFail / Controller @NotBlank), validation
blind spots, exception stack (NPE / NumberFormatException / BadSqlGrammar /
UnknownHostException), RabbitMQ bare-ack / dirty-data silent drop, Feign DNS
failure inside listeners, and branch-routing mistakes (配置不走风控却走了风控 /
isRisk 取值语义错位 / 新老流程配置表名不一致 → 五段证据链).

No fabrication: every class/method/line cited in the reply must come from the
「代码证据」block (index symbol code + real-source grep), never invented.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from app.core.utils import truncate

logger = logging.getLogger("app.agent.log_diagnosis")

# ---------------------------------------------------------------- intent gate

_LOG_DIAG_STRONG = (
    "日志",
    "报错",
    "报异常",
    "抛异常",
    "异常",
    "错误",
    "排查",
    "定位问题",
    "为什么失败",
    "为什么报错",
    "为什么异常",
    "定位一下",
    "查一下这个问题",
    "怎么回事",
    "xxl-job",
    "xxljob",
    "jobhandler",
    "handlecode",
    "handlemsg",
    "exception",
    "npe",
    "nullpointerexception",
    "badsqlgrammar",
    "stacktrace",
    "堆栈",
    "unknownhostexception",
    "unknown column",
    "死信",
    "ack",
    "reject",
    "消费失败",
    "监听失败",
    "脏数据",
    "不走风控",
    "走了风控",
    "过风控",
    "不过风控",
    "风控新引擎",
    "新老流程",
    "决断结果",
    "isrisk",
    "newengineloan",
    "配置了",
    "实际却",
    "实际走了",
    "分支走",
    "分支错",
)

_BRANCH_SIGNALS = (
    "不走风控",
    "走了风控",
    "过风控",
    "不过风控",
    "风控新引擎",
    "新老流程",
    "决断结果",
    "期望新老流程",
    "配置了",
    "实际却",
    "实际走了",
    "分支",
    "isrisk",
    "newengineloan",
    "loanflow",
    "processv2enabled",
    "开关",
    "为什么不生效",
    "配置没生效",
)


def wants_log_diagnose(message: str) -> bool:
    """True when the user pastes a log / reports a failure / asks to locate a bug."""
    text = (message or "").strip().lower()
    if not text:
        return False
    return any(sig in text for sig in _LOG_DIAG_STRONG)


def has_branch_signal(message: str, log_text: str = "") -> bool:
    blob = f"{message or ''} {log_text or ''}".lower()
    return any(sig in blob for sig in _BRANCH_SIGNALS)


# 现象词 → 该工程真实的「配置读取点/路由类」扩展线索（与日志锚点互补）。
# 分支错位日志常常不含类名/配置键名，靠这些显式线索引导检索与源码取证。
_BRANCH_QUERY_MAP: dict[tuple[str, ...], tuple[str, ...]] = {
    ("风控", "不走风控", "走了风控", "过风控", "不过风控", "风控新引擎"): (
        "newEngineLoan", "isRisk", "RiskServiceImpl asynRisk", "loanRiskV2",
        "DealWithAdjustForCreditUsageByAdjustImpl", "riskDetail",
        "WithdrawalsApplyServiceImpl", "newEngineRunLoan",
    ),
    ("新老流程", "决断结果", "期望新老流程", "配了新流程", "走了老流程", "老流程"): (
        "loanDecisionProcessConfig", "demo_p_process_config", "demo_p_loan_apply_config",
        "LoanPProcessConfigServiceImpl", "ProcessRouteConfig", "FlowEnum",
    ),
    ("调额", "调价"): (
        "DealWithAdjustForCreditUsageByAdjustImpl", "checkAdjustRate",
        "riskAdjustForCreditUsage", "isRisk",
    ),
    ("利率", "adjustrate", "numberformatexception", "for input string"): (
        "checkAdjustRate", "QuotaOperationServiceImpl", "adjustRate",
    ),
}


def branch_extra_queries(message: str, log_text: str = "") -> list[str]:
    """Expanded queries/needles for branch-routing & value-semantics scenarios."""
    blob = f"{message or ''} {log_text or ''}".lower()
    out: list[str] = []
    for keys, queries in _BRANCH_QUERY_MAP.items():
        if any(k in blob for k in keys):
            for q in queries:
                if q not in out:
                    out.append(q)
    return out


# ------------------------------------------------------------ anchor extract

_RE_CLASS_TOKEN = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*(?:Job|Listener|ServiceImpl|Service|Controller|"
    r"Util|Processor|Task|Mapper|Aspect|Handler|Client|Agent|Factory|Helper))\b"
)
_RE_FQCN = re.compile(r"\bcom\.demo\.loan(?:\.api)?\.[A-Za-z0-9_.]+\b")
_RE_METHOD_CALL = re.compile(r"#([A-Za-z][A-Za-z0-9_]*)\s*\(")
_RE_STACK_CM = re.compile(
    r"at\s+(com\.[\w.]+)\.([a-z][A-Za-z0-9_]*)\s*\("
)
_RE_EXCEPTION = re.compile(
    r"\b([A-Z][A-Za-z0-9_]*(?:Exception|Error))\b"
)
_RE_CONFIG_KEY = re.compile(
    r"\b(newEngineLoan|isRisk|loanFlow|processV2Enabled|riskAsyncEnabled|"
    r"guaranteeAsyncEnabled|loanPayAsyncEnabled|accountingAsyncEnabled|"
    r"riskLimitOutSumSkip|loanCallbackMqEnabled|sncfcLoanRiskSwitch|"
    r"adjustRate|riskInvokeType|orgInvokeType|pgsInvokeType|accInvokeType|"
    r"riskLimitSumType|loanCallbackType)"
    r"(?:[A-Z][A-Za-z0-9_]*)?\b"
)

_CODE_INTEREST = re.compile(
    r"basicack|basicreject|readvalue|json|objectmapper|handler|handlefail|"
    r"handlesuccess|asserttrue|catch|throw |numberformat|double\.valueof|"
    r"substring|isrisk|newenginel|newris|loanflow|flowenum|getflow|processv2|"
    r"collectionutils|isempty\(|equals\(|默认|default_yes|if \(|ack\(|reject\(|"
    r"activityno|notblank|@valid|signatureid|trylock|redisson|getlock|rlock|"
    r"getlock\(|lock\(|unlock"
)

_FAIL_TEXTS = (
    "执行参数不能为空",
    "配置产品参数不能为空",
    "不能为空",
    "不可为空",
    "脏数据",
    "解析失败",
    "failed to parse",
    "无法解析",
    "unknown column",
    "unknownhostexception",
    "numberformatexception",
    "for input string",
    "获取锁失败",
    "nullpointerexception",
    "空指针",
    "调用风控",
    "风控",
    "校验失败",
    "签名",
    "结清证明",
    "活动编号",
    "activity",
)


def extract_anchors(log_text: str) -> dict[str, list[str]]:
    """Pull the anchors that reliably identify symbols from a log snippet."""
    text = log_text or ""
    classes: list[str] = []
    fqcns: list[str] = []
    methods: list[str] = []
    exceptions: list[str] = []
    config_keys: list[str] = []
    fail_texts: list[str] = []

    def _add(lst: list[str], val: str) -> None:
        v = (val or "").strip()
        if v and v not in lst:
            lst.append(v)

    for m in _RE_FQCN.finditer(text):
        _add(fqcns, m.group(0))
    for m in _RE_CLASS_TOKEN.finditer(text):
        _add(classes, m.group(1))
    for m in _RE_METHOD_CALL.finditer(text):
        _add(methods, m.group(1))
    for m in _RE_STACK_CM.finditer(text):
        _add(fqcns, m.group(1))
        _add(methods, m.group(2))
    for m in _RE_EXCEPTION.finditer(text):
        _add(exceptions, m.group(1))
    for m in _RE_CONFIG_KEY.finditer(text):
        _add(config_keys, m.group(1))
    if "xxl-job" in text.lower() or "jobhandler" in text.lower():
        _add(methods, "handleJob")
        _add(methods, "accErrorProcess")
    for ft in _FAIL_TEXTS:
        if ft in text:
            _add(fail_texts, ft)

    return {
        "classes": classes[:12],
        "fqcns": fqcns[:8],
        "methods": methods[:12],
        "exceptions": exceptions[:6],
        "config_keys": config_keys[:8],
        "fail_texts": fail_texts[:6],
    }


# ------------------------------------------------------------ candidate resolve

def _resolve_candidates(
    project_id: str,
    anchors: dict[str, list[str]],
    extra_queries: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Map anchors to symbols via search_symbols; deterministic rank by anchor hits."""
    from app.agent.explore_tools import search_symbols

    seen_q: set[str] = set()
    queries: list[tuple[str, float]] = []
    for eq in extra_queries or []:
        if eq not in seen_q:
            seen_q.add(eq)
            queries.append((eq, 0.9))
    for cls in anchors.get("classes") or []:
        q = cls
        if q not in seen_q:
            seen_q.add(q)
            queries.append((q, 1.0))
    for fq in anchors.get("fqcns") or []:
        q = fq.split(".")[-1]
        if q not in seen_q:
            seen_q.add(q)
            queries.append((q, 1.0))
    for ft in anchors.get("fail_texts") or []:
        if ft not in seen_q:
            seen_q.add(ft)
            queries.append((ft, 0.8))
    for exc in anchors.get("exceptions") or []:
        if exc not in seen_q:
            seen_q.add(exc)
            queries.append((exc, 0.7))
    for mk in anchors.get("methods") or []:
        if len(mk) >= 4 and mk not in seen_q:
            seen_q.add(mk)
            queries.append((mk, 0.6))

    scored: dict[str, dict[str, Any]] = {}

    def _add(hit: dict[str, Any], q: str, weight: float) -> None:
        sid = str(hit.get("symbol_id") or "")
        fqcn = str(hit.get("fqcn") or "")
        msig = str(hit.get("method_sig") or "")
        if not sid and not fqcn:
            return
        key = sid or fqcn
        prev = scored.get(key)
        base = 0.0
        classes = anchors.get("classes") or []
        methods = anchors.get("methods") or []
        simple = fqcn.split(".")[-1] if fqcn else ""
        if any(c.lower() == simple.lower() for c in classes):
            base += 120.0
        elif any(c.lower() in fqcn.lower() for c in classes):
            base += 60.0
        if any(m.lower() in (msig or "").lower() for m in methods):
            base += 50.0
        if any(ft.lower() in (str(hit.get("summary") or "")).lower() for ft in anchors.get("fail_texts") or []):
            base += 30.0
        total = base + (float(hit.get("score") or 0) * 10.0) + weight * 20.0
        row = {
            "symbol_id": sid,
            "kind": str(hit.get("kind") or ""),
            "fqcn": fqcn,
            "method_sig": msig,
            "file_path": str(hit.get("file_path") or ""),
            "summary": str(hit.get("summary") or ""),
            "score": total,
            "hit_query": q,
        }
        if prev is None or total > prev["score"]:
            scored[key] = row

    for q, weight in queries:
        try:
            sr = search_symbols(project_id, q, top_k=8)
        except Exception:
            logger.debug("search_symbols failed q=%s", q, exc_info=True)
            continue
        for h in sr.get("hits") or []:
            if isinstance(h, dict):
                _add(h, q, weight)
        # Source-mode hits come back the same way; also keep raw snippet hits.
        for h in sr.get("source_hits") or []:
            if isinstance(h, dict):
                _add(h, q, weight * 0.5)

    out = sorted(scored.values(), key=lambda x: x["score"], reverse=True)
    return out[:10]


# ------------------------------------------------------------ source evidence

def _grep_source_evidence(
    project_id: str,
    anchors: dict[str, list[str]],
    extra_needles: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Line-accurate evidence: grep project Java sources for anchor class + config keys.

    More reliable than vector index for branch/config defects; returns the
    matched region with real `file:line`, which the LLM may cite verbatim.
    """
    from pathlib import Path

    from app.core.config import get_settings

    root = get_settings().project_root_path
    if not root or not Path(root).is_dir():
        return []
    classes = anchors.get("classes") or []
    config_keys = anchors.get("config_keys") or []
    fail_texts = anchors.get("fail_texts") or []
    focus = [c for c in classes if len(c) >= 4]
    needles = (
        list(config_keys)
        + [ft for ft in fail_texts if len(ft) >= 4]
        + [x for x in extra_needles or [] if len(x) >= 4]
    )
    if not focus and not needles:
        return []

    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _emmit(rel: str, line_no: int, region: str, needle: str) -> None:
        key = f"{rel}:{line_no}"
        if key in seen or not region:
            return
        seen.add(key)
        evidence.append(
            {
                "file": rel,
                "line": line_no,
                "needle": needle,
                "region": truncate(region, 700),
            }
        )

    try:
        files = [p for p in Path(root).rglob("*.java") if "target" not in p.parts]
    except Exception:
        return []
    scan_needles = [nd for nd in needles if len(nd) >= 4]

    # 扫描顺序 = 相关度优先：文件名命中 any needle 或命中核心类标签（配置/风控/调额/监听/
    # Job/Controller/Mapper/Entity/Adapter）的类先扫，避免无关文件先塞满条数上限。
    _REL_TAG = re.compile(
        r"(?i)processconfig|processroute|riskservice|riskutil|withdrawalsapply|"
        r"dealwithadjust|quotaoperation|listener|job|controller|mapper|entity|adapter|handler"
    )

    def _file_pri(p: Any) -> tuple[int, str]:
        n = p.name
        hit = any(nd.lower() in n.lower() for nd in needles) or bool(_REL_TAG.search(n))
        return (0 if hit else 1, n)

    files.sort(key=_file_pri)
    for p in files:
        try:
            if p.stat().st_size > 512 * 1024:
                continue
        except Exception:
            continue
        rel = str(p.relative_to(root)).replace("\\", "/")
        name = p.name
        name_hit = (not focus) or any(f.lower() in name.lower() for f in focus)
        if not name_hit and not scan_needles:
            continue
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for i, ln in enumerate(lines):
            low = ln.lower()
            hit_needle = None
            for nd in needles:
                if nd.lower() in low:
                    hit_needle = nd
                    break
            if hit_needle is None:
                # 有具体 needle 时只收真实命中的行；无 needle（纯类锚点场景）才允许代码标记兜底。
                if not scan_needles and name_hit and _CODE_INTEREST.search(low):
                    hit_needle = "code-marker"
                else:
                    continue
            start = max(0, i - 4)
            end = min(len(lines), i + 6)
            region = "\n".join(
                f"{j + 1}: {lines[j]}" for j in range(start, end)
            )
            _emmit(rel, i + 1, region, hit_needle or name)
        if len(evidence) >= 40:
            break
    return evidence[:20]


# ---------------------------------------------------------------- entry point

async def run_log_diagnosis(
    message: str,
    *,
    project_id: str,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """Async orchestration: anchors → candidates → evidence → LLM diagnosis."""
    import asyncio

    from app.agent.explore_tools import get_symbol_code
    from app.agent.prompts import load_prompt
    from app.llm.factory import build_llm_provider

    async def _emit(event: dict[str, Any]) -> None:
        if not on_progress:
            return
        try:
            maybe = on_progress(event)
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            logger.debug("progress emit failed", exc_info=True)

    # ---- 1. anchors ----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_anchors",
         "note": "日志锚点提取（类/方法/异常/开关/失败文案）…"}
    )
    anchors = extract_anchors(message)
    await _emit(
        {"type": "step", "status": "done", "step_id": "log_anchors",
         "note": f"锚点: 类{len(anchors['classes'])} 方法{len(anchors['methods'])} "
                 f"异常{len(anchors['exceptions'])} 开关{len(anchors['config_keys'])} "
                 f"失败文案{len(anchors['fail_texts'])}"}
    )

    # ---- 2. candidates ----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_symbol_resolve",
         "note": "锚点→符号检索…"}
    )
    extra_queries = branch_extra_queries(message)
    candidates = _resolve_candidates(project_id, anchors, extra_queries=extra_queries)
    cm_labels = [
        f"{c['fqcn'] or ''}#{(c['method_sig'] or '').replace(c['fqcn'] or '', '')}"
        for c in candidates
    ]

    # ---- 3. evidence ----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_evidence",
         "note": "读取候选符号源码 + 源码精确检索…"}
    )
    branch_flag = has_branch_signal(message)
    sym_parts: list[str] = []
    src_parts: list[str] = []
    evidence_parts: list[str] = []
    _seen_ev: set[str] = set()

    def _add_ev(target: list[str], part: str) -> None:
        key = part.split("\n", 1)[0][:100]
        if key in _seen_ev:
            return
        _seen_ev.add(key)
        target.append(part)

    for c in candidates[:3]:
        sig = str(c.get("method_sig") or "")
        meth = ""
        if "#" in sig:
            meth = sig.split("#", 1)[-1].split("(", 1)[0].strip()
        elif sig.split("(", 1)[0].split():
            meth = sig.split("(", 1)[0].split()[-1]
        code = None
        try:
            code = get_symbol_code(
                project_id,
                symbol_id=str(c.get("symbol_id") or ""),
                fqcn=str(c.get("fqcn") or ""),
                method_name=meth or None,
                max_chars=2000,
            )
        except Exception:
            logger.debug("get_symbol_code failed", exc_info=True)
        if code and code.get("ok"):
            fp = str(code.get("file_path") or "")
            sl = code.get("start_line")
            el = code.get("end_line")
            loc = f"{fp}:{sl}-{el}" if sl else fp
            _add_ev(
                sym_parts,
                f"### {code.get('fqcn') or c.get('fqcn')}#{meth or '?'}（{loc}）\n"
                f"{code.get('raw_code') or '(no raw_code)'}"
            )
    # 源码级定位证据（行号精确，最利于分支/配置类根因），且分支配置键命中排最前。
    src_ev = _grep_source_evidence(project_id, anchors, extra_needles=extra_queries)

    def _ev_rank(ev: dict[str, Any]) -> tuple[int, int]:
        needle = str(ev.get("needle") or "").lower()
        key_hit = (
            needle in {x.lower() for x in extra_queries}
            or any(k in needle for k in ("newenginel", "isrisk", "flow", "processv2", "adjustrate", "loanrisk"))
        )
        # 分支/配置类问题：命中的文件与核心配置/路由实现相关时排最前（如 LoanPProcessConfig*、
        # ProcessRouteConfig、RiskServiceImpl/RiskUtil、WithdrawalsApply、DealWithAdjust、QuotaOperation）。
        fname = str(ev.get("file") or "").lower().replace("\\", "/").rsplit("/", 1)[-1]
        related = any(
            tag in fname
            for tag in (
                "loanpprocessconfig", "processroute", "flowenum",
                "riskserviceimpl", "riskutil", "withdrawalsapply",
                "dealwithadjust", "quotaoperation",
            )
        ) and bool(extra_queries)
        prio = 0 if related else (0 if key_hit else 1)
        return (prio, -int(ev.get("line") or 0))

    for ev in sorted(src_ev, key=_ev_rank)[:12]:
        _add_ev(
            src_parts,
            f"### 源码定位 {ev['file']}:{ev['line']}（命中: {ev['needle']}）\n{ev['region']}"
        )
    # 分支/配置类问题：源码级定位证据优先展示（行号+默认值+分支条件），普通问题符号源码优先。
    parts_raw: list[str] = (src_parts + sym_parts) if branch_flag else (sym_parts + src_parts)
    # 证据块总量限制：过大反而拖慢模型、稀释锚点。
    budget = 10000
    parts: list[str] = []
    used = 0
    for p in parts_raw:
        if used + len(p) > budget:
            break
        parts.append(p)
        used += len(p)
    evidence_block = "\n\n".join(parts) if parts else "（未检索到代码证据）"

    # ---- 4. LLM diagnosis ----
    await _emit(
        {"type": "step", "status": "running", "step_id": "log_diagnose",
         "note": "生成结构化诊断（现象→定位→根因→为什么→修复→证据链）…"}
    )
    log_excerpt = truncate(message, 7000)
    branch_hit = has_branch_signal(message, log_excerpt)
    system = load_prompt("log_diagnose_system")
    if branch_hit:
        system += (
            "\n\n# 补充要求（分支错位场景命中）\n"
            "本问题疑似「分支走错」（配置与实测分支不一致）。必须在「为什么」一节输出五段证据链：\n"
            "1) 配置项（键名/来源） 2) 读取点（类/方法读配置） 3) 兜底/默认值（空值默认哪个方向）\n"
            "4) 分支条件（if 判据原文） 5) 观测日志（哪一行证明实际分支）。\n"
            "重点核查：空值默认 Y / 开关取值语义（1 vs Y/N）/ 注释表名与注解表名不一致 / 查不到配置行静默回退。"
        )
    user_prompt = (
        "【用户描述/日志】\n"
        f"{log_excerpt}\n\n"
        "【已提取日志锚点】\n"
        f"类: {anchors['classes']}  方法: {anchors['methods']}\n"
        f"异常: {anchors['exceptions']}  配置开关: {anchors['config_keys']}  失败文案: {anchors['fail_texts']}\n\n"
        "【代码证据（只允许引用以下真实代码）】\n"
        f"{evidence_block}\n\n"
        "请按系统要求输出结构化诊断（现象/定位/根因/为什么/修复建议/证据链与置信度）。"
    )
    # 诊断走独立 provider（build_llm_provider 未包装熔断、不限总预算）：
    # 慢诊断（几十~几百秒）不应污染聊天主链路熔断器，也不会被熔断级联秒杀。
    llm = build_llm_provider()
    try:
        reply = await asyncio.wait_for(
            llm.chat(
                [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1500,
            ),
            timeout=300,
        )
    except (asyncio.TimeoutError, TimeoutError):
        reply = "**诊断超时**：模型响应超过 300s。可重试，或缩小日志/证据范围后再试。"
    except Exception as e:  # noqa: BLE001
        logger.exception("log diagnosis LLM failed")
        reply = f"**诊断失败**：模型调用异常（{type(e).__name__}）。请稍后重试。"

    steps = [
        {"step_id": "log_anchors", "note": f"锚点: {anchors}", "status": "done"},
        {"step_id": "log_symbol_resolve", "note": f"候选: {cm_labels}", "status": "done"},
        {"step_id": "log_evidence", "note": f"代码证据 {len(parts)} 段", "status": "done"},
        {"step_id": "log_diagnose", "note": "诊断完成" if not reply.startswith("**诊断") else reply,
         "status": "done"},
    ]
    return {
        "reply": reply,
        "anchors": anchors,
        "candidates": candidates,
        "evidence_parts": parts,
        "steps": steps,
        "branch_hit": branch_hit,
    }