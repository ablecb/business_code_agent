"""Structural codebase survey for architecture / tech / symbol-existence asks.

Scans the project filesystem with annotation/shape heuristics only.
No product-domain nouns or business-field special cases.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

logger = logging.getLogger("app.agent.codebase_survey")

# Structural Java type suffixes (language/framework shapes, not business domains).
_TYPE_SUFFIX_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9]*(?:"
    r"Service|Controller|Util|Utils|Helper|Manager|Job|Aspect|"
    r"Handler|Client|Config|Enum|Impl|Processor|Listener|Filter|"
    r"Interceptor|Advice|Factory|Builder|Repository|Mapper|Dao|Constants"
    r"))\b"
)

_ASK_EXISTENCE_RE = re.compile(
    r"(?i)实现了什么|干什么|做了?什么|什么逻辑|是什么类|这个类|"
    r"有没有这个|是否存在|哪里定义|消费什么|处理什么|干什么用|是做什么"
)

_ARCH_ASK_RE = re.compile(
    r"(?i)"
    r"启动类|入口类|Application\b|"
    r"切面|@Aspect|\bAOP\b|"
    r"自定义注解|注解有哪些|有哪些注解|"
    r"包结构|分层|各个包|核心包|"
    r"用到了哪些技术|技术栈|第三方依赖|用了哪些框架|哪些技术|"
    r"技术上用了|用了什么技术|技术有哪些|"
    r"全局异常|ExceptionHandler|ControllerAdvice|RestControllerAdvice|"
    r"配置类|@Configuration|ConfigurationProperties|配置项|"
    r"拦截器|Filter\b|Interceptor\b|"
    r"FeignClient|@FeignClient|Feign\b|"
    r"XXL-?Job|@XxlJob|定时任务|"
    r"Listener\b|监听器|消息监听|"
    r"Redisson|RedisConstants|LOCK_KEY|分布式锁|RLock|"
    r"@Transactional|事务注解|事务传播|"
    # Project / codebase overview (not a single HTTP flow).
    r"描述一下(?:这个)?项目|介绍一下(?:这个)?项目|"
    r"项目(?:概述|概况|简介|整体|全貌)|整体介绍(?:一下)?(?:这个)?项目|"
    r"先描述.*项目|这个项目是什么|项目是做什么|"
    r"\bproject\s+overview\b|\bdescribe\s+(?:the\s+)?project\b|"
    r"仓库概览|代码库概览|codebase\s+overview"
)

_FLOW_SHAPE_RE = re.compile(r"(?i)流程|调用链|主链|怎么走|执行序|FOLLOWS")
_PROJECT_OVERVIEW_RE = re.compile(
    r"(?i)描述一下(?:这个)?项目|介绍一下(?:这个)?项目|"
    r"项目(?:概述|概况|简介|整体|全貌)|整体介绍(?:一下)?(?:这个)?项目|"
    r"先描述.*项目|这个项目是什么|项目是做什么|"
    r"\bproject\s+overview\b|\bdescribe\s+(?:the\s+)?project\b|"
    r"仓库概览|代码库概览|codebase\s+overview"
)

# Free-text topic existence (not typed XxxService; not project-wide overview).
# Structural framing only — no product-domain noun lists.
_TOPIC_EXISTENCE_RE = re.compile(
    r"(?i)"
    r"有没有|有无|是否存在|"
    r"有[\u4e00-\u9fffA-Za-z][\u4e00-\u9fffA-Za-z0-9_]{0,24}"
    r"(?:业务|功能|模块|能力|接口|逻辑|流程)?吗|"
    r"(?:仓库|代码|项目|系统)里?有没有|"
    r"存不存在"
)

# Project-wide capability inventory (notebook), not a single named topic.
_PROJECT_WIDE_CAPABILITY_RE = re.compile(
    r"(?i)"
    r"项目(?:都)?有哪些(?:业务|能力|模块|功能)|"
    r"(?:有哪些|都有哪些)(?:业务|能力|模块)|"
    r"整体(?:有哪些|能力面)|能力面盘点"
)

_SPRING_BOOT_RE = re.compile(r"@SpringBootApplication\b")
_ASPECT_RE = re.compile(r"@Aspect\b")
_ADVICE_RE = re.compile(r"@(?:Rest)?ControllerAdvice\b")
_INTERFACE_ANN_RE = re.compile(r"(?:^|\n)\s*(?:public\s+)?@interface\s+(\w+)")
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;", re.M)
_CLASS_RE = re.compile(
    r"(?:^|\n)\s*public\s+(?:final\s+)?class\s+(\w+)",
)
_POINTCUT_HINT_RE = re.compile(
    r"@(?:Around|Before|After|AfterReturning|AfterThrowing|Pointcut)\s*\(\s*[\"']([^\"']+)[\"']"
)

# Known Maven artifactId → short tech label (framework shapes only).
_ARTIFACT_TECH: dict[str, str] = {
    "spring-boot": "Spring Boot",
    "spring-cloud": "Spring Cloud",
    "spring-web": "Spring Web",
    "spring-aop": "Spring AOP",
    "mybatis": "MyBatis",
    "mybatis-plus": "MyBatis-Plus",
    "pagehelper": "PageHelper",
    "mysql-connector": "MySQL Driver",
    "druid": "Druid",
    "redis": "Redis",
    "redisson": "Redisson",
    "amqp": "RabbitMQ/AMQP",
    "rocketmq": "RocketMQ",
    "kafka": "Kafka",
    "xxl-job": "XXL-Job",
    "openfeign": "OpenFeign",
    "feign": "Feign",
    "nacos": "Nacos",
    "sentinel": "Sentinel",
    "swagger": "Swagger",
    "knife4j": "Knife4j",
    "lombok": "Lombok",
    "hutool": "Hutool",
    "fastjson": "Fastjson",
    "jackson": "Jackson",
    "guava": "Guava",
    "dynamictp": "Dynamic TP",
}


def wants_codebase_architecture(question: str) -> bool:
    """True for structural/framework inventory asks (not answer templates)."""
    q = question or ""
    # Free-text「有没有某话题」→ search path, not notebook.
    if wants_topic_existence_ask(q):
        return False
    if _PROJECT_WIDE_CAPABILITY_RE.search(q) and not _FLOW_SHAPE_RE.search(q):
        return True
    if not _ARCH_ASK_RE.search(q):
        return False
    # Overview wording + flow wording → prefer flow, not architecture dump.
    if _PROJECT_OVERVIEW_RE.search(q) and _FLOW_SHAPE_RE.search(q):
        return False
    return True


def wants_project_overview(question: str) -> bool:
    """True for describe/introduce-the-project asks."""
    q = question or ""
    return bool(_PROJECT_OVERVIEW_RE.search(q)) and not bool(_FLOW_SHAPE_RE.search(q))


def wants_topic_existence_ask(question: str) -> bool:
    """True when asking whether a concrete topic/capability exists in the repo.

    Must use keyword/symbol search — not the architecture notebook alone.
    Excludes project-wide「有哪些业务」overview and typed XxxService probes.
    """
    q = (question or "").strip()
    if not q:
        return False
    if wants_project_overview(q):
        return False
    if _PROJECT_WIDE_CAPABILITY_RE.search(q):
        return False
    # Framework-shape inventory (启动类/切面/技术栈…) still uses notebook.
    if _ARCH_ASK_RE.search(q):
        return False
    if wants_symbol_existence_probe(q):
        return False
    return bool(_TOPIC_EXISTENCE_RE.search(q))


_TOPIC_FRAME_STRIP_RE = re.compile(
    r"(?i)^(?:请问|请|帮我看|看看|看下|仓库|代码|项目|系统)(?:里|中|内)?|"
    r"(?:里|中|内)?(?:有没有|有无|是否存在|存不存在)|"
    r"^有|"
    r"(?:业务|功能|模块|能力|接口|逻辑|流程)?吗?$"
)


def extract_topic_existence_phrase(question: str) -> str:
    """Pull the concrete topic noun from an existence ask (no domain lexicon)."""
    q = (question or "").strip()
    if not q:
        return ""
    m = re.search(
        r"(?i)(?:有没有|有无|是否存在|存不存在)\s*"
        r"(.+?)(?:业务|功能|模块|能力|接口|逻辑|流程)?\s*$",
        q,
    )
    if not m:
        m = re.search(
            r"(?i)有\s*(.+?)(?:业务|功能|模块|能力|接口|逻辑|流程)?\s*吗\s*$",
            q,
        )
    phrase = (m.group(1) if m else q).strip()
    phrase = _TOPIC_FRAME_STRIP_RE.sub("", phrase).strip()
    phrase = re.sub(r"\s+", "", phrase)
    # Drop leftover particles.
    phrase = phrase.strip("的了吗呢啊呀？?。，,")
    if len(phrase) < 2:
        return ""
    return phrase[:40]


_FLOW_ACTION_STRIP_RE = re.compile(
    r"(?i)具体做了什么|做了什么|做了啥|做什么的|做什么|是干啥的|是什么|"
    r"什么样|怎么走|怎么做的|主要做什么|主要流程是什么|主要流程|流程是什么|"
    r"什么流程|的流程|流程说明|流程是|"
    r"请帮我|帮我看|帮我|请帮我|看看|看下|一下|描述下|说明下"
)


def extract_flow_topic_phrase(question: str) -> str:
    """Topic noun from「结清证明具体做了什么」→ 结清证明 (structural strip only)."""
    q = (question or "").strip()
    if not q:
        return ""
    existed = extract_topic_existence_phrase(q)
    # Existence extractor returns whole q when pattern misses — only trust when framed.
    if existed and _TOPIC_EXISTENCE_RE.search(q):
        return existed
    q2 = _FLOW_ACTION_STRIP_RE.sub(" ", q)
    # 删除句尾泛词（「放款接口什么流程」→「放款」），但保留句中「接口」
    # （「结清证明申请接口」中的「接口」是业务名一部分 — 全删会让
    # probe 搜「结清证明申请」而命中 RepayController）。
    q2 = re.sub(r"(?i)(?:接口|入口|业务|功能|模块|能力)\s*$", " ", q2)
    q2 = re.sub(r"\s+", "", q2).strip("的了吗呢啊呀？?。，, ")
    if len(q2) < 2:
        return ""
    return q2[:40]


def needs_topic_entry_grounding(question: str) -> bool:
    """True when ask lacks typed Class#method / XxxService — ground via topic phrase."""
    q = (question or "").strip()
    if not q:
        return False
    if extract_probed_type_names(q):
        return False
    if re.search(r"\b[A-Z][A-Za-z0-9_]*\s*#\s*[A-Za-z_]", q):
        return False
    phrase = extract_flow_topic_phrase(q)
    return len(phrase) >= 2


# Structural role for topic hits (higher = more likely implementation / data contract).
# No product-domain nouns — only type/package shape.
_TOPIC_ROLE_IMPL = 100
_TOPIC_ROLE_DATA = 70
_TOPIC_ROLE_CONFIG = 40
_TOPIC_ROLE_CONST = 10

_IMPL_NAME_RE = re.compile(
    r"(?i)(?:Controller|ServiceImpl|Service|Listener|Job|Processor|Handler|"
    r"Client|Aspect|Manager|Facade|Orchestrat\w*|Worker)$"
)
_DATA_NAME_RE = re.compile(
    r"(?i)(?:Request|Response|VO|Dto|DTO|Entity|Body|Param|Form|Message|Event)$"
)
_CONFIG_NAME_RE = re.compile(
    r"(?i)(?:Config|Configuration|Properties|Property|RouteConfig)$"
)
_CONST_NAME_RE = re.compile(
    r"(?i)(?:Constants?|Constant|Enum|ErrorCode|ErrorMsg)$"
)
_IMPL_PATH_RE = re.compile(
    r"(?i)[/\\](?:controller|service|listener|job|handler|processor|feign|client|adapter)[/\\]"
)
_CONST_PATH_RE = re.compile(
    r"(?i)[/\\](?:constant|constants|enums?)[/\\]|CommonConstants|RedisConstants|DBContants"
)
_TYPE_STEM_STRIP_RE = re.compile(
    r"(?i)(?:Request|Response|VO|Dto|DTO|Body|Param|Form|Constant|Constants|"
    r"Config|Properties|Enum|Impl)+$"
)


def _topic_hit_role(class_name: str, path: str = "") -> tuple[int, str]:
    """Return (score, role_label) from structural type/path shape."""
    name = (class_name or "").strip()
    p = (path or "").replace("\\", "/")
    if _CONST_NAME_RE.search(name) or _CONST_PATH_RE.search(p):
        return _TOPIC_ROLE_CONST, "const"
    if _CONFIG_NAME_RE.search(name) or "/config/" in p.lower():
        return _TOPIC_ROLE_CONFIG, "config"
    if _IMPL_NAME_RE.search(name) or _IMPL_PATH_RE.search(p):
        # Adapter VO path is data; adapter without VO suffix stays impl-ish.
        if _DATA_NAME_RE.search(name):
            return _TOPIC_ROLE_DATA, "data"
        return _TOPIC_ROLE_IMPL, "impl"
    if _DATA_NAME_RE.search(name) or any(
        seg in p.lower() for seg in ("/vo/", "/dto/", "/entity/", "/request/", "/response/")
    ):
        return _TOPIC_ROLE_DATA, "data"
    return _TOPIC_ROLE_CONFIG, "other"


def _type_stem_hints(class_name: str) -> list[str]:
    """UnionSignatureApplyRequestVO → UnionSignatureApply (structural stem)."""
    name = (class_name or "").strip()
    if not name or len(name) < 4:
        return []
    stem = _TYPE_STEM_STRIP_RE.sub("", name)
    if len(stem) < 8 or stem == name:
        return []
    # One stem only — do not over-strip Apply/Query into generic tokens.
    return [stem]


def _stem_matches_class(stem: str, cls: str) -> bool:
    """Prefix / exact-with-suffix match — never loose substring."""
    s = (stem or "").strip()
    c = (cls or "").strip()
    if len(s) < 8 or not c:
        return False
    if c == s or c.startswith(s):
        return True
    for suf in (
        "Service",
        "ServiceImpl",
        "Controller",
        "Listener",
        "Job",
        "Processor",
        "Handler",
        "Client",
        "Impl",
    ):
        if c == f"{s}{suf}":
            return True
    return False


def find_topic_in_sources(
    root: Path,
    phrase: str,
    *,
    limit: int = 120,
) -> list[dict[str, Any]]:
    """Literal phrase hits in Java sources, ranked by structural role."""
    needle = (phrase or "").strip()
    if not needle or not root or not root.is_dir():
        return []
    raw: list[dict[str, Any]] = []
    for path in _iter_java_files(root):
        text = _read_text(path, max_bytes=120_000)
        if needle not in text:
            continue
        snippet = ""
        hit_count = 0
        for i, line in enumerate(text.splitlines(), start=1):
            if needle in line:
                hit_count += 1
                if not snippet:
                    snippet = f"L{i}: {line.strip()[:160]}"
        cls = path.stem
        score, role = _topic_hit_role(cls, str(path))
        # Prefer denser mentions slightly within the same role.
        score += min(hit_count, 5)
        raw.append(
            {
                "path": _rel(root, path),
                "class": cls,
                "snippet": snippet,
                "role": role,
                "score": score,
                "mentions": hit_count,
            }
        )
    raw.sort(key=lambda h: (-int(h.get("score") or 0), str(h.get("class") or "")))
    # Keep both primary and secondary buckets (don't let 100+ Services wipe Constants).
    primary = [h for h in raw if h.get("role") in {"impl", "data"}]
    secondary = [h for h in raw if h.get("role") not in {"impl", "data"}]
    capped = primary[: max(40, limit // 2)] + secondary[:20]
    capped.sort(key=lambda h: (-int(h.get("score") or 0), str(h.get("class") or ""))
    )
    return capped[:limit]


def _expand_impl_hits_from_stems(
    root: Path,
    seed_hits: list[dict[str, Any]],
    *,
    limit: int = 16,
) -> list[dict[str, Any]]:
    """From data-contract class names, hunt related *Service/*Controller by stem."""
    if not root or not root.is_dir() or not seed_hits:
        return []
    stems: list[str] = []
    seen_stem: set[str] = set()
    for h in seed_hits:
        # Only expand from报文/数据契约 — not from Constants.
        if h.get("role") != "data":
            continue
        for s in _type_stem_hints(str(h.get("class") or "")):
            if s not in seen_stem:
                seen_stem.add(s)
                stems.append(s)
        if len(stems) >= 12:
            break
    if not stems:
        return []
    out: list[dict[str, Any]] = []
    seen_path: set[str] = set()
    for path in _iter_java_files(root):
        cls = path.stem
        score, role = _topic_hit_role(cls, str(path))
        if role != "impl":
            continue
        matched = next((s for s in stems if _stem_matches_class(s, cls)), "")
        if not matched:
            continue
        rel = _rel(root, path)
        if rel in seen_path:
            continue
        seen_path.add(rel)
        out.append(
            {
                "path": rel,
                "class": cls,
                "snippet": f"stem={matched} → neighbor of topic data-contract hit",
                "role": role,
                "score": score + 15,
                "mentions": 0,
                "via": "stem",
            }
        )
        if len(out) >= limit:
            break
    out.sort(key=lambda h: (-int(h.get("score") or 0), str(h.get("class") or "")))
    return out


def run_topic_existence_probe(project_id: str, question: str) -> dict[str, Any]:
    """Evidence pack for free-text topic existence (search + source literal)."""
    phrase = extract_topic_existence_phrase(question)
    root = resolve_project_root(project_id)
    source_hits = find_topic_in_sources(root, phrase) if root and phrase else []
    expanded = _expand_impl_hits_from_stems(root, source_hits) if root else []
    by_path: dict[str, dict[str, Any]] = {}
    # Literal hits first; stem neighbors fill gaps without wiping consts.
    for h in list(source_hits) + list(expanded):
        p = str(h.get("path") or "")
        if not p:
            continue
        prev = by_path.get(p)
        if not prev or int(h.get("score") or 0) > int(prev.get("score") or 0):
            by_path[p] = h
    merged = sorted(
        by_path.values(),
        key=lambda h: (-int(h.get("score") or 0), str(h.get("class") or "")),
    )
    # Prefer: literal data/impl, then stem-expanded impl, then config/const.
    literal_primary = [
        h
        for h in merged
        if h.get("role") in {"impl", "data"} and h.get("via") != "stem"
    ]
    stem_primary = [
        h for h in merged if h.get("role") == "impl" and h.get("via") == "stem"
    ]

    def _primary_key(h: dict[str, Any]) -> tuple:
        cls = str(h.get("class") or "")
        bonus = 0
        if cls.endswith("Controller"):
            bonus = 30
        elif "ApplyService" in cls or cls.endswith("ApplyServiceImpl"):
            bonus = 20
        elif h.get("via") == "stem":
            bonus = -5
        return (-(int(h.get("score") or 0) + bonus), cls)

    literal_primary.sort(key=_primary_key)
    stem_primary.sort(key=_primary_key)
    primary = literal_primary + stem_primary
    secondary = [h for h in merged if h.get("role") not in {"impl", "data"}]
    secondary.sort(
        key=lambda h: (-int(h.get("score") or 0), str(h.get("class") or ""))
    )
    index_hits: list[dict[str, Any]] = []
    if phrase:
        try:
            from app.agent.explore_tools import search_symbols

            sr = search_symbols(project_id, phrase, top_k=16)
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "").strip()
                if not fqcn:
                    continue
                cls = fqcn.split(".")[-1]
                score, role = _topic_hit_role(cls, fqcn.replace(".", "/"))
                summary = str(
                    h.get("summary") or h.get("desc") or h.get("name") or ""
                )[:120]
                index_hits.append(
                    {
                        "fqcn": fqcn,
                        "class": cls,
                        "summary": summary,
                        "role": role,
                        "score": score,
                    }
                )
        except Exception:
            logger.debug("topic existence search_symbols failed", exc_info=True)
    index_hits.sort(
        key=lambda h: (-int(h.get("score") or 0), str(h.get("class") or ""))
    )
    seen: set[str] = set()
    uniq_index: list[dict[str, Any]] = []
    for h in index_hits:
        c = h.get("class") or ""
        if c in seen:
            continue
        seen.add(c)
        uniq_index.append(h)
    exists = bool(merged or uniq_index)

    def _fmt_hit(h: dict[str, Any]) -> str:
        via = f" [{h.get('via')}]" if h.get("via") else ""
        return (
            f"- `{h.get('class')}` ({h.get('role')}){via} @ `{h.get('path')}`"
            f" — {h.get('snippet')}"
        )

    lines = [
        f"话题：{phrase or '（未能抽出话题词）'}",
        f"结论：{'存在相关证据' if exists else '未在索引/源码字面量中命中'}",
        "",
        "## 优先：实现/报文类（Controller·Service·Listener·Job·VO/DTO）",
    ]
    if primary:
        for h in primary[:16]:
            lines.append(_fmt_hit(h))
    else:
        lines.append("- （字面量未直接命中实现/报文类；见下方旁证与索引）")
    lines.extend(["", "## 旁证：配置/常量（勿当作主答）"])
    if secondary:
        for h in secondary[:10]:
            lines.append(_fmt_hit(h))
    else:
        lines.append("- （无）")
    lines.extend(["", "## 索引命中（已按结构角色排序）"])
    idx_primary = [h for h in uniq_index if h.get("role") in {"impl", "data"}]
    idx_other = [h for h in uniq_index if h.get("role") not in {"impl", "data"}]
    show_idx = (idx_primary + idx_other)[:12]
    if show_idx:
        for h in show_idx:
            extra = f" — {h.get('summary')}" if h.get("summary") else ""
            lines.append(f"- `{h.get('fqcn')}` ({h.get('role')}){extra}")
    else:
        lines.append("- （无）")
    brief = "\n".join(lines)
    return {
        "ok": True,
        "exists": exists,
        "phrase": phrase,
        "source_hits": primary + secondary,
        "primary_hits": primary,
        "secondary_hits": secondary,
        "index_hits": uniq_index,
        "brief": brief,
        "answer": brief,
    }


def extract_probed_type_names(question: str) -> list[str]:
    """PascalCase framework-shaped type names mentioned in the ask."""
    q = question or ""
    out: list[str] = []
    seen: set[str] = set()
    for m in _TYPE_SUFFIX_RE.finditer(q):
        name = m.group(1)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def wants_symbol_existence_probe(question: str) -> bool:
    """Ask names a Java type and asks what it does / whether it exists."""
    q = question or ""
    names = extract_probed_type_names(q)
    if not names:
        return False
    return bool(_ASK_EXISTENCE_RE.search(q))


def resolve_project_root(project_id: str) -> Path | None:
    from app.core.config import ROOT_DIR, get_settings

    try:
        from app.services.projects import get_project

        data = get_project(project_id) if project_id else None
        if data and data.get("root_path"):
            root = Path(str(data["root_path"]))
            if not root.is_absolute():
                root = (ROOT_DIR / root).resolve()
            if root.is_dir():
                return root
    except Exception:
        logger.debug("resolve_project_root via DB failed", exc_info=True)
    try:
        sample = get_settings().project_root_path
        if sample and Path(sample).is_dir():
            return Path(sample)
    except Exception:
        logger.debug("resolve_project_root via settings failed", exc_info=True)
    return None


def _iter_java_files(root: Path, *, limit: int = 8000) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*.java"):
        parts = {x.lower() for x in p.parts}
        if "target" in parts or "test" in parts or ".git" in parts:
            # Keep src/test only when explicitly needed; default skip tests.
            if "test" in parts and "main" not in parts:
                continue
            if "target" in parts or ".git" in parts:
                continue
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _read_text(path: Path, *, max_bytes: int = 200_000) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="ignore")
    except OSError:
        return ""


def _rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def find_spring_boot_applications(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        if not _SPRING_BOOT_RE.search(text):
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        fqcn = f"{pkg}.{cls}" if pkg else cls
        ann_line = ""
        for line in text.splitlines():
            if "@SpringBootApplication" in line or line.strip().startswith("@Enable"):
                ann_line = (ann_line + " " + line.strip()).strip()
        hits.append(
            {
                "fqcn": fqcn,
                "class": cls,
                "path": _rel(root, path),
                "annotations": ann_line[:400],
            }
        )
    return hits


def find_aspects(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        if not _ASPECT_RE.search(text):
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        tips = [m.group(1) for m in _POINTCUT_HINT_RE.finditer(text)]
        hits.append(
            {
                "fqcn": f"{pkg}.{cls}" if pkg else cls,
                "class": cls,
                "path": _rel(root, path),
                "pointcuts": "; ".join(tips[:6]),
            }
        )
    return hits


def find_exception_advices(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        if not _ADVICE_RE.search(text):
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        handlers = re.findall(r"@ExceptionHandler\s*\(([^)]*)\)", text)
        hits.append(
            {
                "fqcn": f"{pkg}.{cls}" if pkg else cls,
                "class": cls,
                "path": _rel(root, path),
                "exception_handlers": "; ".join(h.strip()[:80] for h in handlers[:12]),
            }
        )
    return hits


def find_custom_annotations(root: Path) -> list[dict[str, str]]:
    """Prefer **/annotation/** folders; also any public @interface in main sources."""
    hits: list[dict[str, str]] = []
    seen: set[str] = set()
    for path in _iter_java_files(root):
        text = _read_text(path)
        for name in _INTERFACE_ANN_RE.findall(text):
            pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
            fqcn = f"{pkg}.{name}" if pkg else name
            if fqcn in seen:
                continue
            # Skip well-known JDK/Spring markers accidentally matched — only @interface.
            seen.add(fqcn)
            in_ann_pkg = "annotation" in {p.lower() for p in path.parts}
            hits.append(
                {
                    "fqcn": fqcn,
                    "name": name,
                    "path": _rel(root, path),
                    "in_annotation_package": "yes" if in_ann_pkg else "no",
                }
            )
    # Prefer annotation-package entries first.
    hits.sort(key=lambda h: (0 if h["in_annotation_package"] == "yes" else 1, h["fqcn"]))
    return hits


def summarize_packages(root: Path, *, depth: int = 4) -> list[dict[str, Any]]:
    """Count .java files under src/main/java package segments."""
    main_roots = list(root.rglob("src/main/java"))
    if not main_roots:
        main_roots = [root]
    counts: dict[str, int] = {}
    for base in main_roots:
        if not base.is_dir():
            continue
        for path in base.rglob("*.java"):
            try:
                rel = path.relative_to(base).as_posix()
            except ValueError:
                continue
            parts = rel.split("/")[:-1]
            if not parts:
                key = "(default)"
            else:
                key = ".".join(parts[:depth])
            counts[key] = counts.get(key, 0) + 1
    rows = [{"package": k, "java_files": v} for k, v in counts.items()]
    rows.sort(key=lambda r: (-int(r["java_files"]), str(r["package"])))
    return rows[:40]


def survey_pom_technologies(root: Path) -> list[str]:
    techs: list[str] = []
    seen: set[str] = set()
    for pom in root.rglob("pom.xml"):
        if "target" in {p.lower() for p in pom.parts}:
            continue
        try:
            tree = ET.parse(pom)
        except Exception:
            continue
        for el in tree.iter():
            tag = el.tag.split("}")[-1] if isinstance(el.tag, str) else ""
            if tag != "artifactId" or el.text is None:
                continue
            aid = el.text.strip().lower()
            for key, label in _ARTIFACT_TECH.items():
                if key in aid and label not in seen:
                    seen.add(label)
                    techs.append(label)
    return techs


_CTRL_CLASS_RE = re.compile(r"(?:^|\n)\s*public\s+class\s+(\w+Controller)\b")
_MAPPING_PATH_RE = re.compile(
    r"""@(?:Request|Get|Post|Put|Delete|Patch)Mapping\s*\(\s*(?:value\s*=\s*|path\s*=\s*)?["']([^"']+)["']"""
)
_INFRA_PKG_SEG = frozenset(
    {
        "impl",
        "util",
        "utils",
        "common",
        "config",
        "base",
        "enums",
        "dto",
        "vo",
        "model",
        "entity",
        "mapper",
        "constants",
        "annotation",
        "aspect",
        "handler",
        "exception",
        "system",
        "cache",
        "adapter",
        "param",
        "feigndecorator",
    }
)
_BIZ_LAYER_SEG = frozenset(
    {"controller", "service", "listener", "xxljob", "plugin", "feign"}
)
_SOURCE_TECH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"org\.redisson|RedissonClient|\bRLock\b|getLock\s*\("), "Redisson/分布式锁"),
    (re.compile(r"RedisTemplate|StringRedisTemplate|opsForValue"), "Redis"),
    (re.compile(r"@RabbitListener|springframework\.amqp"), "RabbitMQ"),
    (re.compile(r"@KafkaListener|springframework\.kafka"), "Kafka"),
    (re.compile(r"@RocketMQMessageListener|rocketmq"), "RocketMQ"),
    (re.compile(r"@FeignClient\b"), "OpenFeign"),
    (re.compile(r"@XxlJob\b"), "XXL-Job"),
    (re.compile(r"@Transactional\b"), "Spring 事务"),
    (re.compile(r"@Mapper\b|org\.apache\.ibatis"), "MyBatis"),
    (re.compile(r"@RestController\b|@Controller\b"), "Spring Web"),
    (re.compile(r"@Scheduled\b"), "Spring Scheduling"),
    (re.compile(r"PageHelper|pagehelper"), "PageHelper"),
]


def find_business_modules(root: Path) -> dict[str, Any]:
    """Infer business-facing modules from package folders + Controller types."""
    subpkg_counts: dict[str, int] = {}
    controllers: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        parts = [p for p in pkg.split(".") if p] if pkg else []
        # com.xxx.yyy.controller.signature → layer=controller, module=signature
        for i, seg in enumerate(parts):
            if seg.lower() not in _BIZ_LAYER_SEG:
                continue
            layer = seg.lower()
            nxt = parts[i + 1] if i + 1 < len(parts) else ""
            if nxt and nxt.lower() not in _INFRA_PKG_SEG and nxt.lower() != "v2":
                key = f"{layer}/{nxt}"
            else:
                key = f"{layer}/(root)"
            subpkg_counts[key] = subpkg_counts.get(key, 0) + 1
            break
        if path.stem.endswith("Controller") or "controller" in {
            p.lower() for p in path.parts
        }:
            m = _CTRL_CLASS_RE.search(text)
            if not m:
                continue
            cls = m.group(1)
            maps = _MAPPING_PATH_RE.findall(text)
            controllers.append(
                {
                    "class": cls,
                    "fqcn": f"{pkg}.{cls}" if pkg else cls,
                    "path": _rel(root, path),
                    "mappings": ", ".join(maps[:6]),
                }
            )
    modules = [
        {"module": k, "java_files": v}
        for k, v in sorted(subpkg_counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    controllers.sort(key=lambda c: c["class"])
    return {"modules": modules[:40], "controllers": controllers[:80]}


def survey_source_tech_signals(
    root: Path, *, business_layers_only: bool = False
) -> list[str]:
    """Framework signals from source annotations/APIs (structural)."""
    found: list[str] = []
    seen: set[str] = set()
    for path in _iter_java_files(root):
        if business_layers_only:
            parts_l = {p.lower() for p in path.parts}
            if not (parts_l & _BIZ_LAYER_SEG):
                continue
            # Skip pure infra packages under those layers when possible.
            if "common" in parts_l and "controller" not in parts_l:
                continue
        text = _read_text(path, max_bytes=80_000)
        for pat, label in _SOURCE_TECH_PATTERNS:
            if label in seen:
                continue
            if pat.search(text):
                seen.add(label)
                found.append(label)
        if len(found) >= len(_SOURCE_TECH_PATTERNS):
            break
    return found


_LOCK_CONST_RE = re.compile(
    r"(?m)^\s*(?:public\s+|private\s+|protected\s+|static\s+|final\s+)+"
    r"String\s+(\w+)\s*=\s*([\"'][^\"']+[\"'])"
)
_LOCK_NAME_OK_RE = re.compile(r"(?i)(?:^|_)LOCK(?:_|$)|LOCK_KEY|LOCK_NAME")
_REDIS_CONST_FILE_RE = re.compile(r"(?i)Redis|Lock")
_GET_LOCK_RE = re.compile(
    r"getLock\s*\(\s*([^)]{0,200})\)"
)
_XXL_JOB_RE = re.compile(
    r"""@XxlJob\s*\(\s*(?:value\s*=\s*)?["']([^"']+)["']"""
)
_FEIGN_RE = re.compile(r"@FeignClient\s*\(([^)]*)\)")
_LISTENER_ANN_RE = re.compile(
    r"@(?:RabbitListener|KafkaListener|JmsListener|RocketMQMessageListener)\b"
)
_TX_RE = re.compile(r"@Transactional\b(?:\s*\(([^)]*)\))?")
_CONF_PROPS_RE = re.compile(
    r"""@ConfigurationProperties\s*\(\s*(?:prefix\s*=\s*)?["']([^"']+)["']"""
)
_VALUE_RE = re.compile(r"""@Value\s*\(\s*["']([^"']+)["']\s*\)""")


def find_lock_constants(root: Path) -> list[dict[str, str]]:
    """String constants whose names look like locks (structural *LOCK*)."""
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        if not _REDIS_CONST_FILE_RE.search(path.name) and "constant" not in path.as_posix().lower():
            # Still scan files that declare LOCK_ fields.
            text_head = _read_text(path, max_bytes=40_000)
            if "LOCK" not in text_head:
                continue
            text = text_head
        else:
            text = _read_text(path)
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        for m in _LOCK_CONST_RE.finditer(text):
            name = m.group(1)
            if not _LOCK_NAME_OK_RE.search(name):
                continue
            hits.append(
                {
                    "class": cls,
                    "fqcn": f"{pkg}.{cls}" if pkg else cls,
                    "name": name,
                    "value": m.group(2).strip("\"'"),
                    "path": _rel(root, path),
                }
            )
    return hits[:80]


def find_redisson_lock_sites(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        if "getLock(" not in text and "RLock" not in text:
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        for m in _GET_LOCK_RE.finditer(text):
            expr = re.sub(r"\s+", " ", m.group(1)).strip()
            hits.append(
                {
                    "class": cls,
                    "fqcn": f"{pkg}.{cls}" if pkg else cls,
                    "expr": expr[:180],
                    "path": _rel(root, path),
                }
            )
            if len(hits) >= 60:
                return hits
    return hits


def find_message_listeners(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        is_listener_name = path.stem.endswith("Listener")
        has_ann = bool(_LISTENER_ANN_RE.search(text))
        if not (is_listener_name or has_ann):
            continue
        if "listener" not in path.as_posix().lower() and not has_ann and not is_listener_name:
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        anns = []
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("@") and any(
                k in s
                for k in (
                    "Listener",
                    "Rabbit",
                    "Kafka",
                    "RocketMQ",
                    "Jms",
                )
            ):
                anns.append(s[:160])
        hits.append(
            {
                "fqcn": f"{pkg}.{cls}" if pkg else cls,
                "class": cls,
                "path": _rel(root, path),
                "annotations": " | ".join(anns[:4]),
            }
        )
    # Prefer package path containing listener
    hits.sort(
        key=lambda h: (0 if "/listener/" in h["path"].replace("\\", "/").lower() else 1, h["fqcn"])
    )
    return hits[:40]


def find_xxl_jobs(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        if "@XxlJob" not in text:
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        for m in _XXL_JOB_RE.finditer(text):
            hits.append(
                {
                    "job": m.group(1),
                    "class": cls,
                    "fqcn": f"{pkg}.{cls}" if pkg else cls,
                    "path": _rel(root, path),
                }
            )
    hits.sort(key=lambda h: h["job"].lower())
    return hits[:120]


def find_feign_clients(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        text = _read_text(path)
        if "@FeignClient" not in text:
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        # interface or class
        m_iface = re.search(
            r"(?:^|\n)\s*public\s+interface\s+(\w+)", text
        ) or _CLASS_RE.search(text)
        cls = m_iface.group(1) if m_iface else path.stem
        attrs = ""
        fm = _FEIGN_RE.search(text)
        if fm:
            attrs = re.sub(r"\s+", " ", fm.group(1)).strip()[:200]
        hits.append(
            {
                "fqcn": f"{pkg}.{cls}" if pkg else cls,
                "class": cls,
                "path": _rel(root, path),
                "attrs": attrs,
            }
        )
    hits.sort(key=lambda h: h["fqcn"])
    return hits[:60]


def find_transactional_methods(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    method_re = re.compile(
        r"(?:public|protected|private)\s+[\w.<>,\s\[\]]+\s+(\w+)\s*\("
    )
    for path in _iter_java_files(root):
        text = _read_text(path)
        if "@Transactional" not in text:
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        # Pair each @Transactional with the next method name (heuristic).
        for m in _TX_RE.finditer(text):
            tail = text[m.end() : m.end() + 400]
            mm = method_re.search(tail)
            method = mm.group(1) if mm else "(class-level-or-unknown)"
            props = (m.group(1) or "").strip()
            hits.append(
                {
                    "fqcn": f"{pkg}.{cls}" if pkg else cls,
                    "method": method,
                    "props": re.sub(r"\s+", " ", props)[:120],
                    "path": _rel(root, path),
                }
            )
            if len(hits) >= 80:
                return hits
    return hits


def find_config_bindings(root: Path, type_names: list[str] | None = None) -> list[dict[str, str]]:
    """@ConfigurationProperties / @Value on Config-shaped types."""
    want = {n for n in (type_names or []) if n}
    hits: list[dict[str, str]] = []
    for path in _iter_java_files(root):
        if want:
            if path.stem not in want:
                continue
        elif "config" not in path.as_posix().lower() and not path.stem.endswith("Config"):
            continue
        text = _read_text(path)
        if "@ConfigurationProperties" not in text and "@Value" not in text:
            continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        cls_m = _CLASS_RE.search(text)
        cls = cls_m.group(1) if cls_m else path.stem
        prefixes = _CONF_PROPS_RE.findall(text)
        values = _VALUE_RE.findall(text)
        hits.append(
            {
                "fqcn": f"{pkg}.{cls}" if pkg else cls,
                "class": cls,
                "path": _rel(root, path),
                "prefixes": ", ".join(prefixes[:6]),
                "values": ", ".join(values[:12]),
            }
        )
    return hits[:40]


def _class_level_annotations(text: str, type_name: str) -> str:
    """Annotations immediately above a type declaration (structural only)."""
    lines = text.splitlines()
    decl = re.compile(
        rf"^\s*public\s+(?:final\s+)?(?:class|interface|enum)\s+{re.escape(type_name)}\b"
    )
    for i, line in enumerate(lines):
        if not decl.search(line):
            continue
        anns: list[str] = []
        j = i - 1
        while j >= 0:
            s = lines[j].strip()
            if not s or s.startswith("//") or s.startswith("/*") or s.startswith("*"):
                j -= 1
                continue
            if s.startswith("@"):
                anns.append(s[:180])
                j -= 1
                continue
            break
        anns.reverse()
        return " | ".join(anns[:10])
    return ""


def probe_type_in_sources(root: Path, type_name: str) -> dict[str, Any]:
    """Return existence evidence for a simple class name under root."""
    name = (type_name or "").strip()
    if not name:
        return {"name": name, "exists": False, "matches": []}
    matches: list[dict[str, str]] = []
    class_decl = re.compile(
        rf"(?:^|\n)\s*public\s+(?:final\s+)?(?:class|interface|enum)\s+{re.escape(name)}\b"
    )
    for path in _iter_java_files(root):
        if path.stem != name and name not in path.name:
            text = _read_text(path)
            if not class_decl.search(text):
                continue
        else:
            text = _read_text(path)
            if not class_decl.search(text) and path.stem != name:
                continue
        pkg = (_PACKAGE_RE.search(text) or [None, ""])[1]
        row: dict[str, str] = {
            "fqcn": f"{pkg}.{name}" if pkg else name,
            "path": _rel(root, path),
        }
        anns = _class_level_annotations(text, name)
        if anns:
            row["annotations"] = anns
        matches.append(row)
        if len(matches) >= 8:
            break
    return {"name": name, "exists": bool(matches), "matches": matches}


def probe_types_in_index(project_id: str, type_names: list[str]) -> list[dict[str, Any]]:
    """Prefer MySQL symbol index; empty list if DB unavailable."""
    out: list[dict[str, Any]] = []
    if not project_id or not type_names:
        return out
    try:
        from app.tagging.search import search_symbols
    except Exception:
        return out
    for name in type_names[:8]:
        try:
            hits = search_symbols(project_id, name, top_k=8) or []
        except Exception:
            hits = []
        matched = []
        for h in hits:
            cm = str(h.get("class_method") or h.get("cm") or "")
            simple = cm.split("#", 1)[0].split(".")[-1]
            if simple == name or name in cm:
                matched.append(
                    {
                        "class_method": cm,
                        "path": str(h.get("file_path") or h.get("path") or ""),
                    }
                )
        out.append({"name": name, "exists": bool(matched), "matches": matched[:8]})
    return out


def render_architecture_answer(
    *,
    question: str,
    root: Path,
    apps: list[dict[str, str]],
    aspects: list[dict[str, str]],
    annotations: list[dict[str, str]],
    advices: list[dict[str, str]],
    packages: list[dict[str, Any]],
    techs: list[str],
    lock_consts: list[dict[str, str]] | None = None,
    lock_sites: list[dict[str, str]] | None = None,
    listeners: list[dict[str, str]] | None = None,
    xxl_jobs: list[dict[str, str]] | None = None,
    feigns: list[dict[str, str]] | None = None,
    tx_methods: list[dict[str, str]] | None = None,
    config_bindings: list[dict[str, str]] | None = None,
    capability_counts: dict[str, int] | None = None,
    business_modules: dict[str, Any] | None = None,
    source_techs: list[str] | None = None,
    business_techs: list[str] | None = None,
) -> str:
    q = question or ""
    lock_consts = lock_consts or []
    lock_sites = lock_sites or []
    listeners = listeners or []
    xxl_jobs = xxl_jobs or []
    feigns = feigns or []
    tx_methods = tx_methods or []
    config_bindings = config_bindings or []
    counts = dict(capability_counts or {})
    biz = business_modules or {}
    source_techs = source_techs or []
    business_techs = business_techs or []
    lines: list[str] = []
    # Note: business/tech/overview lenses are NOT answered here as fixed templates.
    # PEV gathers a neutral evidence brief, then LLM answers the user question.
    overview = wants_project_overview(q)
    want_boot = bool(re.search(r"(?i)启动|入口类|Application", q))
    want_aspect = bool(re.search(r"(?i)切面|AOP|@Aspect", q))
    want_ann = bool(re.search(r"注解", q)) and not bool(
        re.search(r"(?i)@Transactional|@XxlJob|@FeignClient", q)
    )
    want_pkg = bool(re.search(r"包结构|分层|各个包|核心包", q))
    want_tech = bool(re.search(r"技术|框架|依赖", q)) and not bool(
        re.search(r"(?i)Feign|XxlJob|Redisson|Transactional", q)
    )
    want_ex = bool(re.search(r"(?i)异常|Advice|ExceptionHandler", q))
    want_lock = bool(
        re.search(r"(?i)Redisson|RedisConstants|LOCK_KEY|分布式锁|RLock|getLock|锁\s*key", q)
    )
    want_listener = bool(re.search(r"(?i)Listener\b|监听器|消息监听", q))
    want_job = bool(re.search(r"(?i)XXL-?Job|@XxlJob|定时任务", q))
    want_feign = bool(re.search(r"(?i)FeignClient|@FeignClient|\bFeign\b", q))
    want_tx = bool(re.search(r"(?i)@Transactional|事务注解|事务传播", q))
    want_cfg = bool(
        re.search(r"(?i)ConfigurationProperties|配置项|配置绑定|@Value", q)
    ) or (
        bool(extract_probed_type_names(q))
        and bool(re.search(r"绑定|配置", q))
    )
    focused_extra = any(
        (want_lock, want_listener, want_job, want_feign, want_tx, want_cfg)
    )
    focused_core = any(
        (want_boot, want_aspect, want_ann, want_pkg, want_tech, want_ex)
    )
    # Unfocused architecture / project overview → core card only (no lock/job dumps).
    show_core = overview or focused_core or not focused_extra
    # Explicit framework inventories only when asked.
    show_lock = want_lock and not overview
    show_listener = want_listener and not overview
    show_job = want_job and not overview
    show_feign = want_feign and not overview
    show_tx = want_tx and not overview
    show_cfg = want_cfg and not overview

    def _section_intro() -> None:
        if not overview:
            return
        lines.append("## 项目概览（源码结构）")
        lines.append(
            "以下为启动入口、横切能力、包分布与技术依赖的结构扫描结果；"
            "不是业务说明书。若需锁常量 / Listener / Job / Feign 清单，请单独追问。"
        )
        lines.append("")

    def _section_capability_counts() -> None:
        if not overview or not counts:
            return
        lines.append("## 能力形态摘要（数量，非清单）")
        order = (
            ("listeners", "消息 Listener"),
            ("xxl_jobs", "@XxlJob"),
            ("feigns", "@FeignClient"),
            ("lock_consts", "*LOCK* 常量"),
            ("controllers", "Controller 包文件"),
        )
        for key, label in order:
            if key in counts:
                lines.append(f"- {label}：{counts[key]}")
        lines.append("")

    def _section_apps() -> None:
        lines.append("## 启动类（@SpringBootApplication）")
        if not apps:
            lines.append("未在已读源码中找到 `@SpringBootApplication` 启动类。")
            return
        for a in apps[:8]:
            lines.append(f"- `{a['fqcn']}` — `{a['path']}`")
            if a.get("annotations"):
                lines.append(f"  - 注解线索：{a['annotations'][:240]}")

    def _section_aspects() -> None:
        lines.append("## 切面（@Aspect）")
        if not aspects:
            lines.append("未在已读源码中找到 `@Aspect` 类。")
            return
        for a in aspects[:20]:
            extra = f"（切点线索：{a['pointcuts']}）" if a.get("pointcuts") else ""
            lines.append(f"- `{a['fqcn']}` — `{a['path']}`{extra}")

    def _section_ann() -> None:
        lines.append("## 自定义注解（@interface）")
        rows = annotations
        if overview:
            pref = [a for a in annotations if a.get("in_annotation_package") == "yes"]
            rows = pref or annotations[:8]
        if not rows:
            lines.append("未在已读源码中找到自定义 `@interface`。")
            return
        for a in rows[:30]:
            lines.append(f"- `{a['fqcn']}` — `{a['path']}`")

    def _section_advice() -> None:
        lines.append("## 全局异常处理（@ControllerAdvice / @RestControllerAdvice）")
        if not advices:
            lines.append("未在已读源码中找到 ControllerAdvice。")
            return
        for a in advices[:10]:
            lines.append(f"- `{a['fqcn']}` — `{a['path']}`")
            if a.get("exception_handlers"):
                lines.append(f"  - `@ExceptionHandler`：{a['exception_handlers']}")

    def _section_pkg() -> None:
        lines.append("## 包结构（src/main/java 文件数）")
        if not packages:
            lines.append("未能统计包结构。")
            return
        limit = 18 if overview else 25
        for p in packages[:limit]:
            lines.append(f"- `{p['package']}` — {p['java_files']} 个 `.java`")

    def _section_tech() -> None:
        lines.append("## 技术线索（来自 pom.xml artifactId）")
        if not techs:
            lines.append("未能从 pom 解析出已知框架依赖线索。")
            return
        lines.append("、".join(techs))

    def _section_locks() -> None:
        lines.append("## 分布式锁线索（*LOCK* 常量 / getLock）")
        if lock_consts:
            lines.append("### 锁相关字符串常量")
            for c in lock_consts[:40]:
                lines.append(
                    f"- `{c['fqcn']}.{c['name']}` = `{c['value']}` — `{c['path']}`"
                )
        else:
            lines.append("未找到名称含 `LOCK` 的 String 常量。")
        if lock_sites:
            lines.append("### getLock(...) 调用点")
            for s in lock_sites[:30]:
                lines.append(f"- `{s['fqcn']}`：`getLock({s['expr']})` — `{s['path']}`")

    def _section_listeners() -> None:
        lines.append("## Listener（类名 *Listener 或消息监听注解）")
        if not listeners:
            lines.append("未找到 Listener 类或消息监听注解。")
            return
        for h in listeners[:40]:
            extra = f" — {h['annotations']}" if h.get("annotations") else ""
            lines.append(f"- `{h['fqcn']}` — `{h['path']}`{extra}")

    def _section_jobs() -> None:
        lines.append("## @XxlJob 任务")
        if not xxl_jobs:
            lines.append("未找到 `@XxlJob`。")
            return
        for h in xxl_jobs[:80]:
            lines.append(f"- `{h['job']}` — `{h['fqcn']}` — `{h['path']}`")

    def _section_feign() -> None:
        lines.append("## @FeignClient")
        if not feigns:
            lines.append("未找到 `@FeignClient`。")
            return
        for h in feigns[:40]:
            extra = f" ({h['attrs']})" if h.get("attrs") else ""
            lines.append(f"- `{h['fqcn']}`{extra} — `{h['path']}`")

    def _section_tx() -> None:
        lines.append("## @Transactional")
        if not tx_methods:
            lines.append("未找到 `@Transactional`。")
            return
        for h in tx_methods[:50]:
            props = f" ({h['props']})" if h.get("props") else ""
            lines.append(f"- `{h['fqcn']}#{h['method']}`{props} — `{h['path']}`")

    def _section_cfg() -> None:
        lines.append("## 配置绑定（@ConfigurationProperties / @Value）")
        if not config_bindings:
            lines.append("未找到配置绑定线索（或未点名到具体 Config 类型）。")
            return
        for h in config_bindings[:20]:
            lines.append(f"- `{h['fqcn']}` — `{h['path']}`")
            if h.get("prefixes"):
                lines.append(f"  - prefix：`{h['prefixes']}`")
            if h.get("values"):
                lines.append(f"  - @Value：`{h['values']}`")

    _section_intro()
    if show_core:
        if overview or want_boot or not focused_core:
            _section_apps()
            lines.append("")
        if overview or want_aspect or not focused_core:
            _section_aspects()
            lines.append("")
        if overview or want_ann or not focused_core:
            _section_ann()
            lines.append("")
        if overview or want_ex or not focused_core:
            _section_advice()
            lines.append("")
        if overview or want_pkg or not focused_core:
            _section_pkg()
            lines.append("")
        if overview or want_tech or not focused_core:
            _section_tech()
            lines.append("")
        _section_capability_counts()
    else:
        if want_boot:
            _section_apps()
            lines.append("")
        if want_aspect:
            _section_aspects()
            lines.append("")
        if want_ann:
            _section_ann()
            lines.append("")
        if want_ex:
            _section_advice()
            lines.append("")
        if want_pkg:
            _section_pkg()
            lines.append("")
        if want_tech:
            _section_tech()
            lines.append("")
    if show_lock:
        _section_locks()
        lines.append("")
    if show_listener:
        _section_listeners()
        lines.append("")
    if show_job:
        _section_jobs()
        lines.append("")
    if show_feign:
        _section_feign()
        lines.append("")
    if show_tx:
        _section_tx()
        lines.append("")
    if show_cfg:
        _section_cfg()
        lines.append("")
    lines.append(f"依据：扫描根目录=`{root}`")
    return "\n".join(lines).rstrip() + "\n"


def render_symbol_probe_answer(
    *,
    type_names: list[str],
    index_hits: list[dict[str, Any]],
    source_hits: list[dict[str, Any]],
) -> str:
    lines = ["## 类型存在性核查（禁止编造）", ""]
    by_name = {str(h.get("name")): h for h in index_hits}
    by_src = {str(h.get("name")): h for h in source_hits}
    for name in type_names:
        idx = by_name.get(name) or {}
        src = by_src.get(name) or {}
        exists = bool(idx.get("exists") or src.get("exists"))
        if not exists:
            lines.append(f"- `{name}`：**项目中未找到该类/接口/枚举定义**（索引与源码扫描均未命中）。")
            lines.append("  不会编造其实现逻辑。")
            continue
        lines.append(f"- `{name}`：存在。")
        for m in (idx.get("matches") or [])[:5]:
            cm = m.get("class_method") or m.get("fqcn") or ""
            path = m.get("path") or ""
            lines.append(f"  - 索引：`{cm}` {f'— `{path}`' if path else ''}".rstrip())
        for m in (src.get("matches") or [])[:5]:
            ann = m.get("annotations") or ""
            extra = f"；类注解：{ann}" if ann else ""
            lines.append(f"  - 源码：`{m.get('fqcn')}` — `{m.get('path')}`{extra}")
    lines.append("")
    lines.append("依据：符号索引（若可用）+ 源码 `class|interface|enum` 声明扫描。")
    return "\n".join(lines)


def format_architecture_evidence_brief(root: Path, pack: dict[str, Any]) -> str:
    """Neutral evidence dump for an LLM to answer the user question."""
    lines: list[str] = [f"扫描根目录=`{root}`", ""]
    apps = list(pack.get("apps") or [])
    lines.append("## 启动类")
    if not apps:
        lines.append("（无）")
    else:
        for a in apps[:8]:
            lines.append(f"- {a.get('fqcn')} @ {a.get('path')}")
            if a.get("annotations"):
                lines.append(f"  ann: {str(a['annotations'])[:200]}")
    lines.append("")
    lines.append("## 切面")
    for a in list(pack.get("aspects") or [])[:15]:
        lines.append(f"- {a.get('fqcn')} @ {a.get('path')}")
    lines.append("")
    lines.append("## 自定义注解（优先 annotation 包）")
    anns = list(pack.get("annotations") or [])
    pref = [a for a in anns if a.get("in_annotation_package") == "yes"] or anns[:8]
    for a in pref[:12]:
        lines.append(f"- {a.get('fqcn')} @ {a.get('path')}")
    lines.append("")
    lines.append("## 全局异常处理")
    for a in list(pack.get("advices") or [])[:8]:
        lines.append(
            f"- {a.get('fqcn')} @ {a.get('path')} handlers={a.get('exception_handlers')}"
        )
    lines.append("")
    lines.append("## 包文件数（截断）")
    for p in list(pack.get("packages") or [])[:20]:
        lines.append(f"- {p.get('package')}: {p.get('java_files')}")
    lines.append("")
    lines.append("## pom 技术线索")
    lines.append("、".join(pack.get("techs") or []) or "（无）")
    lines.append("")
    lines.append("## 源码框架信号（全局）")
    lines.append("、".join(pack.get("source_techs") or []) or "（无）")
    lines.append("")
    lines.append("## 分层包源码框架信号")
    lines.append("、".join(pack.get("business_techs") or []) or "（无）")
    lines.append("")
    counts = dict(pack.get("capability_counts") or {})
    if counts:
        lines.append("## 能力数量")
        for k, v in counts.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    biz = dict(pack.get("business_modules") or {})
    lines.append("## 分层子模块")
    for m in list(biz.get("modules") or [])[:25]:
        lines.append(f"- {m.get('module')}: {m.get('java_files')}")
    lines.append("")
    lines.append("## Controller 入口")
    for c in list(biz.get("controllers") or [])[:40]:
        extra = f" mappings={c.get('mappings')}" if c.get("mappings") else ""
        lines.append(f"- {c.get('class')} @ {c.get('path')}{extra}")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def collect_architecture_evidence(root: Path) -> dict[str, Any]:
    """Gather a balanced evidence pack — not a question-classified final answer."""
    apps = find_spring_boot_applications(root)
    aspects = find_aspects(root)
    annotations = find_custom_annotations(root)
    advices = find_exception_advices(root)
    packages = summarize_packages(root)
    techs = survey_pom_technologies(root)
    source_techs = survey_source_tech_signals(root, business_layers_only=False)
    business_techs = survey_source_tech_signals(root, business_layers_only=True)
    business_modules = find_business_modules(root)
    listeners = find_message_listeners(root)
    xxl_jobs = find_xxl_jobs(root)
    feigns = find_feign_clients(root)
    lock_consts = find_lock_constants(root)
    ctrl_files = 0
    for p in packages:
        pkg = str(p.get("package") or "")
        if pkg.endswith(".controller") or ".controller." in pkg:
            ctrl_files += int(p.get("java_files") or 0)
    capability_counts = {
        "listeners": len(listeners),
        "xxl_jobs": len(xxl_jobs),
        "feigns": len(feigns),
        "lock_consts": len(lock_consts),
        "controllers": ctrl_files,
    }
    pack: dict[str, Any] = {
        "apps": apps,
        "aspects": aspects,
        "annotations": annotations,
        "advices": advices,
        "packages": packages,
        "techs": techs,
        "source_techs": source_techs,
        "business_techs": business_techs,
        "business_modules": business_modules,
        "capability_counts": capability_counts,
    }
    pack["brief"] = format_architecture_evidence_brief(root, pack)
    return pack


def run_architecture_survey(project_id: str, question: str) -> dict[str, Any]:
    """Load project notebook (方案 C); optional focused deepen for framework inventories.

    Final user-facing prose is produced by LLM in PEV from the brief.
    """
    root = resolve_project_root(project_id)
    if not root:
        return {
            "ok": False,
            "answer": "未能解析项目根目录，无法做结构化源码扫描。",
            "brief": "",
            "evidence": {},
        }
    q = question or ""
    want_lock = bool(
        re.search(r"(?i)Redisson|RedisConstants|LOCK_KEY|分布式锁|RLock|getLock|锁\s*key", q)
    )
    want_listener = bool(re.search(r"(?i)Listener\b|监听器|消息监听", q))
    want_job = bool(re.search(r"(?i)XXL-?Job|@XxlJob|定时任务", q))
    want_feign = bool(re.search(r"(?i)FeignClient|@FeignClient|\bFeign\b", q))
    want_tx = bool(re.search(r"(?i)@Transactional|事务注解|事务传播", q))
    want_cfg = bool(
        re.search(r"(?i)ConfigurationProperties|配置项|配置绑定|@Value", q)
    ) or (
        bool(extract_probed_type_names(q)) and bool(re.search(r"绑定|配置", q))
    )
    # Framework-shape deepen (annotation/API names only — not product domains).
    if want_lock or want_listener or want_job or want_feign or want_tx or want_cfg:
        focused = render_architecture_answer(
            question=question,
            root=root,
            apps=[],
            aspects=[],
            annotations=[],
            advices=find_exception_advices(root)
            if re.search(r"(?i)异常|Advice|ExceptionHandler", q)
            else [],
            packages=[],
            techs=survey_pom_technologies(root),
            lock_consts=find_lock_constants(root) if want_lock else [],
            lock_sites=find_redisson_lock_sites(root) if want_lock else [],
            listeners=find_message_listeners(root) if want_listener else [],
            xxl_jobs=find_xxl_jobs(root) if want_job else [],
            feigns=find_feign_clients(root) if want_feign else [],
            tx_methods=find_transactional_methods(root) if want_tx else [],
            config_bindings=(
                find_config_bindings(
                    root,
                    [n for n in extract_probed_type_names(q) if n.endswith("Config")]
                    or None,
                )
                if want_cfg
                else []
            ),
        )
        return {
            "ok": True,
            "answer": focused,
            "brief": focused,
            "evidence": {"root": str(root), "mode": "focused_scan"},
        }

    try:
        from app.agent.project_notebook import ensure_notebook, notebook_brief

        doc = ensure_notebook(project_id, root=root, rebuild=False)
        brief = notebook_brief(doc)
        return {
            "ok": True,
            "answer": brief,
            "brief": brief,
            "evidence": {
                "root": str(root),
                "mode": "project_notebook",
                "built_at": doc.get("built_at"),
                "apps": len(doc.get("apps") or []),
                "controllers": len(doc.get("controllers") or []),
                "techs": doc.get("techs") or [],
                "capability_counts": doc.get("capability_counts") or {},
            },
        }
    except Exception:
        logger.debug("notebook load/build failed; fallback live scan", exc_info=True)
        pack = collect_architecture_evidence(root)
        brief = str(pack.get("brief") or "")
        return {
            "ok": True,
            "answer": brief,
            "brief": brief,
            "evidence": {
                "root": str(root),
                "mode": "evidence_brief_fallback",
                "apps": len(pack.get("apps") or []),
                "controllers": len(
                    list((pack.get("business_modules") or {}).get("controllers") or [])
                ),
                "techs": pack.get("techs") or [],
            },
        }


def run_symbol_probe(project_id: str, question: str) -> dict[str, Any]:
    names = extract_probed_type_names(question)
    if not names:
        return {
            "ok": False,
            "answer": "问题中未识别到可核查的 Java 类型名（需带 Service/Controller/Util 等结构性后缀）。",
            "evidence": {},
        }
    root = resolve_project_root(project_id)
    index_hits = probe_types_in_index(project_id, names)
    source_hits = [
        probe_type_in_sources(root, n) if root else {"name": n, "exists": False, "matches": []}
        for n in names
    ]
    answer = render_symbol_probe_answer(
        type_names=names, index_hits=index_hits, source_hits=source_hits
    )
    any_exists = any(bool(h.get("exists")) for h in index_hits + source_hits)
    return {
        "ok": True,
        "answer": answer,
        "evidence": {
            "names": names,
            "any_exists": any_exists,
            "all_missing": not any_exists,
        },
    }
