"""接口→表聚合：把散在每个方法上的表信息沿调用链聚合到入口（Item 3 补充）。

背景：``treesitter_deep_flow`` 在每个方法的 ``symbol_desc.extra_json`` 里已经记录了
``business_tables``（触达的实体/表）、``impl_view.db_ops``（select/insert/update/delete +
表名 + 字段来源）、``table_field_changes``（字段级改动）。但它们是"每方法一条"，
运行时「XX 接口都涉及哪些表」需要**按入口聚合**整条调用链的表，这一步之前没人做，
导致该问题命中黄金库的 ``table_field``（"X表有哪些字段"）答非所问。

本模块在解析期（generate_golden_qa 脚本）做两件事：

1. **聚合**：对每个流程入口方法，沿其调用链（同 flow_key 的方法 + flow_doc.stages）
   汇总所有触达表 → 每张表的操作类型（select/insert/update/delete）、关联字段、来源方法。
   结果写回入口方法的 ``symbol_desc.extra_json.interface_tables``（运行时探索也可用）。

2. **确定性黄金问答**：为每个入口生成「XX接口都涉及哪些表 / 用到哪些表」问答对，
   落 ``golden_qa`` 表（source=``interface_tables``），不经过 LLM 重推理。

用法（scripts/generate_golden_qa.py 自动调用）::

    python scripts/generate_golden_qa.py --interface-tables-only
"""
from __future__ import annotations

import logging
import re
from typing import Any

from sqlalchemy import select

from app.analysis.table_dict import load_table_dict
from app.core.config import get_settings
from app.core.utils import truncate
from app.db.mysql import get_session
from app.models.orm import GoldenQAORM, SymbolDescORM

logger = logging.getLogger("app.analysis.interface_tables")

# 问句模板：检索命中靠"业务词 + 涉及哪些表"共同作用，多存变体提高召回。
_QUESTION_TEMPLATES: tuple[str, ...] = (
    "{label}接口都涉及哪些表",
    "{label}接口涉及哪些表",
    "{label}都涉及哪些表",
    "{label}涉及哪些表",
    "{label}用到哪些表",
    "{label}操作了哪些表",
    "{label}用到了哪些表",
)


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------


def _load_symbol_descs(project_id: str) -> list[dict[str, Any]]:
    """全部 symbol_desc → 纯 dict（含 extra_json）。"""
    out: list[dict[str, Any]] = []
    try:
        with get_session() as session:
            rows = session.scalars(
                select(SymbolDescORM).where(SymbolDescORM.project_id == project_id)
            ).all()
            for r in rows:
                extra = dict(r.extra_json) if isinstance(r.extra_json, dict) else {}
                out.append(
                    {
                        "symbol_id": r.symbol_id,
                        "class_method": str(extra.get("class_method") or ""),
                        "extra": extra,
                        "summary_business_clear": str(r.summary_business_clear or ""),
                        "summary_dev_clear": str(r.summary_dev_clear or ""),
                    }
                )
    except Exception:
        logger.warning("interface_tables: read symbol_desc failed (index may be empty)", exc_info=True)
        return []
    return out


def _entity_to_table_map(project_id: str) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """实体/表名 → (物理表名, 表元信息)。用于把 business_tables/db_ops 里的实体名映射到物理表。"""
    alias: dict[str, str] = {}
    meta: dict[str, dict[str, Any]] = {}
    for t in load_table_dict(project_id):
        table_name = str(t.get("table_name") or "")
        entity_class = str(t.get("entity_class") or "")
        if not table_name:
            continue
        meta.setdefault(table_name, t)
        if entity_class:
            alias[entity_class] = table_name
            alias[entity_class.lower()] = table_name
        alias.setdefault(table_name, table_name)
        alias.setdefault(table_name.lower(), table_name)
    return alias, meta


def _normalize_table_name(
    raw: str,
    alias: dict[str, str],
    meta: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any] | None]:
    """实体/表名 → (物理表名, 表元信息)。映射不到就原样返回。"""
    name = str(raw or "").strip().lstrip("`").rstrip("`")
    if not name:
        return "", None
    physical = alias.get(name) or alias.get(name.lower())
    if physical:
        return physical, meta.get(physical)
    # Mapper / Entity 后缀剥掉再试（WithdrawalsMapper → Withdrawals）
    stripped = re.sub(r"(?i)(Mapper|Entity|Ext|DTO|VO|PO|DO)$", "", name)
    if stripped:
        physical = alias.get(stripped) or alias.get(stripped.lower())
        if physical:
            return physical, meta.get(physical)
    return name, None


def _looks_like_physical_table(name: str) -> bool:
    """判断一个名字像不像物理表名（表字典未收录时的兜底）。

    项目表名约定：demo_b_xxx / b_demo_xxx 等。VO/DTO/Request/Response 结尾以及
    内存 map/配置对象（product_datas、special_product_map、xxx_organization 等
    无业务表语义的）一律不算表——宁可少报，不要拿非表对象污染表清单。
    """
    n = str(name or "").strip().lower()
    if not n:
        return False
    if any(n.endswith(s) for s in ("vo", "dto", "request", "response", "req", "resp", "info")):
        return False
    # 配置/内存对象常见尾缀与词根 → 非表
    if any(n.endswith(s) for s in ("_config", "_datas", "_map", "_organization", "_switch")):
        return False
    if any(k in n for k in ("product_datas", "special_product", "switch_organization", "process_config")):
        return False
    if n.startswith(("demo_", "b_demo_", "t_b_")):
        return True
    # 无 demo_ 前缀的下划线小写名（如 loan_assistance、payment_plan）：只有表字典
    # 已收录才接受（在 _touch 里 tmeta 非空即放行），这里不再兜底放行。
    return False


def _normalize_op(op: str | None) -> str:
    """操作词归一化：db_ops 的英文 op 原样；中文操作词映射到标准读写语义。

    未识别的操作词返回空串（不显示），避免「equals/check/获取/排序」这类
    非 CRUD 噪音污染答案。
    """
    if not op:
        return ""
    raw = str(op).strip().lower()
    if raw in {"select", "insert", "update", "delete", "read"}:
        return "select" if raw == "read" else raw
    # 旧引擎 table_field_changes 的中文操作词
    if any(k in raw for k in ("新增", "写入", "插入", "insert", "落库", "创建")):
        return "insert"
    if any(k in raw for k in ("删除", "delete", "移除", "清除")):
        return "delete"
    if any(k in raw for k in (
        "修改", "更新", "update", "改变", "变更", "设置", "赋值", "复制",
        "覆盖", "转换", "格式转换", "条件覆盖", "条件赋值", "equals", "等于",
        "check", "校验", "排序", "获取", "提取", "填充", "构建", "组装",
        "映射", "保存", "save", "写入",
    )):
        return "update"
    if any(k in raw for k in ("查询", "读取", "select", "检索", "筛选", "扫描", "read", "获取值")):
        return "select"
    return ""


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def _flow_chain_cms(project_id: str, entry_cm: str) -> list[str]:
    """从 Neo4j 调用图 + method_var_resolve 调用点拿入口的下游调用链。

    旧解析引擎（topo-priority）没有 flow_keys/flow_doc，调用链有两个来源：
    1. Neo4j CALLS 图（按 class_method 精确匹配入口，沿有向 CALLS 边收集下游）
    2. method_var_resolve.call_sites（status=ok_local 且 resolved 的调用点，
       补充 Neo4j 边不全的情况，例如 Controller→Service）
    拿不到就返回 [entry_cm]（入口自身至少聚合自己的 business_tables）。
    """
    cms: list[str] = []
    seen: set[str] = set()

    def _add(cm: str) -> None:
        cm = str(cm or "").strip()
        if cm and cm not in seen:
            seen.add(cm)
            cms.append(cm)

    # 1) Neo4j 有向 CALLS
    try:
        from app.db.neo4j_client import get_neo4j

        client = get_neo4j()
        if client.verify():
            rows = client.run(
                """
                MATCH (start)
                WHERE start.project_id = $project_id AND start.class_method = $entry
                WITH start LIMIT 3
                CALL {
                  WITH start
                  MATCH path = (start)-[:CALLS|SCHEDULES|CONSUMES*1..4]->(n)
                  RETURN n
                  LIMIT 200
                }
                RETURN collect(DISTINCT n) AS nodes
                """,
                {"project_id": project_id, "entry": entry_cm},
            )
            for n in rows[0].get("nodes") or [] if rows else []:
                props = dict(n) if isinstance(n, dict) else {k: v for k, v in n.items()}
                _add(str(props.get("class_method") or ""))
    except Exception:
        pass

    # 2) method_var_resolve call_sites（ok_local 且 resolved → 真实调用）
    try:
        from sqlalchemy import select

        from app.models.orm import MethodVarResolveORM
        from app.db.mysql import get_session

        with get_session() as session:
            row = session.scalar(
                select(MethodVarResolveORM).where(
                    MethodVarResolveORM.project_id == project_id,
                    MethodVarResolveORM.class_method == entry_cm,
                )
            )
            if row is not None and isinstance(row.payload_json, dict):
                for site in row.payload_json.get("call_sites") or []:
                    if not isinstance(site, dict):
                        continue
                    if str(site.get("status") or "") != "ok_local":
                        continue
                    _add(str(site.get("resolved") or ""))
    except Exception:
        pass

    _add(entry_cm)
    return cms or [entry_cm]


def _aggregate_entry(
    entry: dict[str, Any],
    descs_by_cm: dict[str, dict[str, Any]],
    *,
    alias: dict[str, str],
    meta: dict[str, dict[str, Any]],
    project_id: str,
) -> dict[str, Any]:
    """聚合一个入口方法：沿调用链收集表 + 操作 + 字段。"""
    extra = entry["extra"]
    flow_keys = list(extra.get("flow_keys") or [])
    entry_cm = str(extra.get("class_method") or entry.get("class_method") or "")
    chain_cms: list[str] = []

    # 1) 同 flow_key 的所有方法（新引擎有 flow_keys 时的最稳判定）
    if flow_keys:
        fk_set = set(flow_keys)
        for cm, d in descs_by_cm.items():
            dex = d["extra"]
            if fk_set & set(dex.get("flow_keys") or []):
                chain_cms.append(cm)

    # 2) flow_doc.stages 的 class_method（保留执行序）
    fd = extra.get("flow_doc") if isinstance(extra.get("flow_doc"), dict) else {}
    stages = fd.get("stages") or []
    if isinstance(stages, list) and stages:
        ordered: list[str] = []
        seen: set[str] = set()
        for st in stages:
            cm = str(st.get("class_method") or "") if isinstance(st, dict) else ""
            if cm and cm not in seen:
                seen.add(cm)
                ordered.append(cm)
        # 去重保序：stages 序在前，flow_key 补漏
        tail = [cm for cm in chain_cms if cm not in seen]
        chain_cms = ordered + tail
    elif not chain_cms:
        # 3) 旧引擎：无 flow_keys/flow_doc → Neo4j 调用图 + mvr 调用点
        chain_cms = _flow_chain_cms(project_id, entry_cm)

    if entry_cm and entry_cm not in chain_cms:
        chain_cms.insert(0, entry_cm)

    # 4) 链太短（只有入口自身/极少方法）→ 扩展为同 Controller 类的业务方法集合。
    #    旧引擎调用边不全，「XX接口」业务语义上 ≈ XX Controller 的业务方法群；
    #    business_tables 数据本身可靠，聚合后至少覆盖该接口直接的写表面。
    if len(chain_cms) <= 2 and "Controller" in entry_cm.split("#", 1)[0]:
        simple = entry_cm.split("#", 1)[0]
        for cm, d in descs_by_cm.items():
            if cm.split("#", 1)[0] != simple:
                continue
            if cm == entry_cm or cm in chain_cms:
                continue
            dex = d["extra"]
            if dex.get("business_tables") or dex.get("table_field_changes"):
                chain_cms.append(cm)

    # 聚合每张表
    tables: dict[str, dict[str, Any]] = {}
    via_methods: dict[str, list[str]] = {}

    def _touch(table_key: str, *, op: str | None = None, field: str | None = None, via: str | None = None) -> None:
        if not table_key:
            return
        physical, tmeta = _normalize_table_name(table_key, alias, meta)
        if not physical:
            return
        if tmeta is None and not _looks_like_physical_table(physical):
            # 映射不到表字典且不像物理表名（VO/DTO/Req/Resp 等）→ 不是数据库表，跳过
            return
        rec = tables.setdefault(
            physical,
            {
                "table_name": physical,
                "name_cn": str((tmeta or {}).get("name_cn") or "") if tmeta else "",
                "entity_class": str((tmeta or {}).get("entity_class") or "") if tmeta else "",
                "file_path": str((tmeta or {}).get("file_path") or "") if tmeta else "",
                "ops": set(),
                "fields": [],
                "via": [],
            },
        )
        norm = _normalize_op(op)
        if norm:
            rec["ops"].add(norm)
        # 未归一化的操作词（equals/check/获取/排序等非 CRUD 语义）不显示
        if field:
            if field not in rec["fields"]:
                rec["fields"].append(field)
        if via and via not in rec["via"]:
            rec["via"].append(via)

    for cm in chain_cms:
        d = descs_by_cm.get(cm)
        if d is None:
            continue
        dex = d["extra"]
        # business_tables（实体/表简单类名）
        for tb in dex.get("business_tables") or []:
            _touch(str(tb), via=cm)
        # impl_view.db_ops（含 op + table_or_entity + field_sources）
        iv = dex.get("impl_view") if isinstance(dex.get("impl_view"), dict) else {}
        for op in iv.get("db_ops") or []:
            if not isinstance(op, dict):
                continue
            tbl = str(op.get("table_or_entity") or "")
            fs = op.get("field_sources") if isinstance(op.get("field_sources"), dict) else {}
            fields = list(fs.keys()) if fs else []
            op_name = str(op.get("op") or "")
            via_call = str(op.get("via_call") or "") or cm
            if not tbl and not fields:
                continue
            _touch(tbl, op=op_name, field=(fields[0] if fields else None), via=via_call)
            for f in fields[1:]:
                _touch(tbl, field=f, via=via_call)
        # table_field_changes（字段级）
        for tc in dex.get("table_field_changes") or []:
            if not isinstance(tc, dict):
                continue
            _touch(
                str(tc.get("table_or_entity") or ""),
                op=str(tc.get("op") or ""),
                field=str(tc.get("field") or ""),
                via=cm,
            )

    # 排序：有写操作的表优先，其次中文名，最后表名
    ordered_tables = sorted(
        tables.values(),
        key=lambda r: (
            0 if r["ops"] & {"insert", "update", "delete"} else 1,
            r["name_cn"] or "",
            r["table_name"],
        ),
    )
    return {
        "entry_cm": entry_cm,
        "flow_keys": flow_keys,
        "title": str(fd.get("title") or ""),
        "summary_business_clear": str(entry.get("summary_business_clear") or ""),
        "http_paths": list(extra.get("http_paths") or []),
        "chain_method_count": len(chain_cms),
        "chain_methods": chain_cms[:40],
        "tables": [
            {
                **r,
                "ops": sorted(r["ops"]),
                "via": r["via"][:6],
            }
            for r in ordered_tables
        ],
    }


_BUSINESS_KEYWORDS = (
    # 长词优先（减少「放款」这类短词导致多入口同 label 冲突）
    "放款初始还款计划", "放款资金路由回调", "放款审批结果通知", "放款申请汇总",
    "放款申请", "放款通知", "放款审批", "放款路由", "统一放款", "放款撤销",
    "放款明细", "放款信息", "提现申请", "提现结果", "提现放款", "贷款期次",
    "贷款通知", "贷款重试", "贷款回收", "客户借据", "借据号", "用信申请",
    "用信入账", "用信查询", "还款撤销", "还款请求", "还款列表", "放款", "提现",
    "用信", "还款", "审批", "授信", "额度", "撤销",
)


def _business_label_from_summary(summary: str) -> str:
    """从方法摘要抽业务短标签：优先业务关键词，其次去噪音后的名词片断。

    例：「该方法用于将放款通知数据记录到数据库中…」→「放款通知」；
    「查询放款撤销信息」→「放款撤销」；「请求首先校验借据号」→「借据号校验」。
    失败返回空串。
    """
    text = str(summary or "").strip()
    if not text:
        return ""
    # 1) 优先业务关键词
    for kw in _BUSINESS_KEYWORDS:
        if kw in text:
            return kw
    # 2) 去引导前缀后取第一个名词片断
    text = re.sub(r"[\s，。；、：（）()]+", "", text)
    text = re.sub(
        r"^(?:该方法|此方法|本方法|该接口|接口|用于|主要|用来|负责|根据|通过|当|如果|若|将|从|向|对|请求|首先|先|查询|获取|记录|处理|设置|校验|判断|检查|转换|构建|解析|生成|调用|返回|读取|写入|更新|删除|新增|提交|申请|发放|放款|提现|撤销|确认|审核)+",
        "",
        text,
    )
    m = re.search(r"([\u4e00-\u9fff]{2,10})", text)
    if not m:
        return ""
    frag = m.group(1)
    if len(frag) < 2:
        return ""
    frag = re.split(r"(?:数据库|优先|从|到|并|且|若|则|根据|通过|对|向)", frag, maxsplit=1)[0]
    frag = re.sub(r"(?:到数据库中|到数据库|到缓存|到本地)$", "", frag)
    return frag[:10]


def _entry_label(agg: dict[str, Any]) -> str:
    """入口的友好标签：优先业务标题/摘要，其次类名#方法，最后表名。"""
    title = str(agg.get("title") or "").strip()
    if title and re.search(r"[\u4e00-\u9fff]", title):
        return title
    # 从入口方法摘要抽业务词（旧引擎无 flow_doc.title）
    biz = _business_label_from_summary(str(agg.get("summary_business_clear") or ""))
    if biz:
        return biz
    entry_cm = str(agg.get("entry_cm") or "")
    simple = entry_cm.split("#", 1)[0].split(".")[-1] if entry_cm else ""
    simple = re.sub(r"(?i)Controller$", "", simple)
    if simple:
        return simple
    return entry_cm or "该接口"


def _interface_tables_answer(agg: dict[str, Any]) -> str:
    """确定性答案：表清单 + 每表操作/字段/来源 + 关联提示。"""
    tables = agg.get("tables") or []
    if not tables:
        return "根据已解析的调用链，未发现该接口触达的数据库表（可能无写库/无持久化操作）。"
    label = _entry_label(agg)
    entry_cm = str(agg.get("entry_cm") or "")
    lines = [f"{label}（入口 {entry_cm}）涉及的数据库表（共 {len(tables)} 张）：", ""]
    for i, t in enumerate(tables, 1):
        cn = t.get("name_cn") or t.get("entity_class") or ""
        head = f"{i}. `{t['table_name']}`"
        if cn:
            head += f"（{cn}）"
        lines.append(head)
        ops = t.get("ops") or []
        if ops:
            lines.append(f"   - 操作：{'、'.join(ops)}")
        if t.get("fields"):
            lines.append(f"   - 主要字段：{', '.join(t['fields'][:12])}")
        if t.get("via"):
            lines.append(f"   - 经方法：{', '.join(t['via'][:4])}")
        lines.append("")
    if agg.get("flow_keys"):
        lines.append(f"依据：deep-bfs 调用链聚合（flow: {', '.join(agg['flow_keys'][:3])}），"
                     f"共 {agg.get('chain_method_count', 0)} 个方法。")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# 入口生成 + 落库
# ---------------------------------------------------------------------------


# 纯 CRUD 直通 / 工具方法名 → 不是业务接口，排除（旧引擎无入口标记时的启发式）。
_CRUD_DIRECT_NAMES = frozenset({
    "list", "listPage", "page", "getById", "getBy", "insert", "updateById",
    "update", "deleteById", "deleteByIds", "delete", "save", "saveOrUpdate",
    "removeById", "removeByIds", "count", "exist", "exists",
})
_TOOL_PREFIX_RE = re.compile(
    r"(?i)^(is|check|valid|build|parse|to|convert|format|generate|"
    r"handle|execute|run|make|fill|sync|assemble|compose|extract|"
    r"resolve|load|read|write|send|notify|callback|dispatch|route|"
    r"return|response|request|set|get)[A-Z_0-9]"
)


def _find_entries(descs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """入口候选：新引擎有 entry/flow_doc 标记直接取；旧引擎退化为
    important=True 的 Controller 方法（HTTP 入口最接近「接口」语义）。

    排除明显非业务接口的方法：纯工具/校验（isValidDateTimeFormatOne 等）、
    CRUD 直通（list/listPage/insert/updateById/getById/deleteById 等）——
    它们要么不触表、要么只碰一张表，聚合「接口涉及哪些表」没有价值。
    """
    out: list[dict[str, Any]] = []
    for d in descs:
        extra = d["extra"]
        if not d.get("class_method"):
            continue
        is_entry = bool(extra.get("entry")) or "flow_doc" in extra
        if is_entry:
            out.append(d)
            continue
        cm = d["class_method"]
        simple, _, method = cm.partition("#")
        is_controller = "Controller" in simple
        if not is_controller or not bool(extra.get("important")):
            continue
        method = (method or "").strip()
        if not method:
            continue
        # 纯 CRUD 直通 / 工具方法 → 跳过
        if method in _CRUD_DIRECT_NAMES or _TOOL_PREFIX_RE.match(method):
            continue
        out.append(d)
    return out


def build_interface_tables(
    project_id: str,
    *,
    max_entries: int = 0,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """聚合入口→表并落库。返回统计。

    1. 对每个入口方法，聚合调用链上的表/操作/字段，写回 ``extra_json.interface_tables``。
    2. 生成「XX接口涉及哪些表」确定性问答对，落 golden_qa（source=interface_tables）。
    """
    from app.analysis.golden_qa import _embed_one_or_none, _upsert_qa
    from app.llm.factory import get_embedding_provider

    descs = _load_symbol_descs(project_id)
    if not descs:
        return {"ok": False, "entries": 0, "created": 0, "skipped": 0, "error": "empty index"}
    alias, meta = _entity_to_table_map(project_id)
    descs_by_cm = {d["class_method"]: d for d in descs if d["class_method"]}
    entries = _find_entries(descs)
    if max_entries > 0:
        entries = entries[:max_entries]

    embed = get_embedding_provider()
    created = 0
    skipped = 0
    with_tables = 0

    # 先聚合全部入口，再按 label 去重：同一业务 label 只保留覆盖表最多的入口，
    # 避免「放款申请」等泛 label 被 2 张表的小入口抢占，检索时命中最全的答案。
    aggs: list[dict[str, Any]] = []
    for i, entry in enumerate(entries, 1):
        try:
            agg = _aggregate_entry(
                entry, descs_by_cm, alias=alias, meta=meta, project_id=project_id
            )
        except Exception:
            logger.exception("aggregate entry failed: %s", entry.get("class_method"))
            skipped += 1
            continue

        # 写回 symbol_desc.extra_json.interface_tables（所有入口都写，供运行时探索）
        try:
            _write_back_interface_tables(project_id, entry["symbol_id"], agg)
        except Exception:
            logger.debug("interface_tables write-back failed for %s", entry.get("class_method"), exc_info=True)

        tables = agg.get("tables") or []
        if tables:
            with_tables += 1
        agg["_entry"] = entry.get("class_method")
        agg["_tables_count"] = len(tables)
        aggs.append(agg)

    # 按 label 分组 → 每组取表最多的入口
    by_label: dict[str, dict[str, Any]] = {}
    for agg in aggs:
        if agg["_tables_count"] == 0:
            continue
        label = _entry_label(agg)
        prev = by_label.get(label)
        if prev is None or agg["_tables_count"] > prev["_tables_count"]:
            by_label[label] = agg

    for i, (label, agg) in enumerate(by_label.items(), 1):
        tables = agg.get("tables") or []
        # 黄金问答对（多问法变体，语义检索按需命中）
        answer = _interface_tables_answer(agg)
        qs: list[str] = []
        for tpl in _QUESTION_TEMPLATES:
            q = tpl.format(label=label).strip()
            if q and q not in qs:
                qs.append(q)
        # 类名#方法变体（含英文入口名，提高技术问法召回）
        entry_cm = str(agg.get("entry_cm") or "")
        for suffix in ("涉及哪些表", "都涉及哪些表", "用到哪些表"):
            q = f"{entry_cm} {suffix}"
            if q and q not in qs:
                qs.append(q)

        try:
            vecs = embed.embed(qs)
        except Exception:
            logger.warning("embed interface-table questions failed", exc_info=True)
            vecs = []
        for idx, q in enumerate(qs):
            vec = vecs[idx] if idx < len(vecs) else None
            if _upsert_qa(project_id, q, answer, agg.get("entry_cm"), 1.0, [], vec, source="interface_tables"):
                created += 1
            else:
                skipped += 1
        if on_progress:
            on_progress(
                {
                    "index": i,
                    "total": len(by_label),
                    "entry": agg.get("_entry"),
                    "tables": len(tables),
                    "questions": len(qs),
                    "status": "ok" if tables else "no_tables",
                }
            )

    # 3) 业务聚合问答对：对核心业务词，union 所有含该词的入口的表。
    #    解决「放款接口都涉及哪些表」这类泛问命中 4 张表小入口的问题——
    #    业务聚合答案覆盖最全，检索时更可能命中它。
    created += _build_business_union_qa(project_id, aggs, embed)
    return {
        "ok": True,
        "entries": len(entries),
        "with_tables": with_tables,
        "labels": len(by_label),
        "created": created,
        "skipped": skipped,
    }


def _build_business_union_qa(
    project_id: str,
    aggs: list[dict[str, Any]],
    embed,
) -> int:
    """按核心业务词 union 入口的表，生成「XX接口涉及哪些表」综合问答。

    用短业务词（放款/提现/用信…）聚合：一个入口可归入多个词（如「放款申请」
    同时归入「放款」「申请」→ 只用核心词），「放款」聚合覆盖所有放款相关入口，
    泛问「放款接口都涉及哪些表」能命中覆盖最全的综合答案。
    只对覆盖 >= 3 个入口且表数 >= 5 的业务词生成，避免噪音聚合。
    """
    from app.analysis.golden_qa import _upsert_qa

    # 短业务词：只留能泛化覆盖多入口的核心词
    _BUSINESS_WORDS = ("放款", "提现", "用信", "还款", "审批", "授信", "撤销", "额度", "借据")

    # 业务词 → [agg...]（一个入口可归入多个词）
    by_word: dict[str, list[dict[str, Any]]] = {}
    for agg in aggs:
        if agg["_tables_count"] == 0:
            continue
        label = _entry_label(agg)
        for word in _BUSINESS_WORDS:
            if word in label:
                by_word.setdefault(word, []).append(agg)

    created = 0
    for word, group in by_word.items():
        if len(group) < 3:
            continue
        # union 所有入口的表（按表名去重，合并 ops/fields/via）
        merged: dict[str, dict[str, Any]] = {}
        for agg in group:
            for t in agg.get("tables") or []:
                key = t["table_name"]
                rec = merged.setdefault(
                    key,
                    {
                        "table_name": key,
                        "name_cn": t.get("name_cn") or "",
                        "entity_class": t.get("entity_class") or "",
                        "ops": set(),
                        "fields": [],
                        "via": [],
                    },
                )
                rec["ops"] |= set(t.get("ops") or [])
                for f in t.get("fields") or []:
                    if f and f not in rec["fields"]:
                        rec["fields"].append(f)
                for v in t.get("via") or []:
                    if v and v not in rec["via"]:
                        rec["via"].append(v)
        if len(merged) < 5:
            continue
        # 有写操作的表优先
        ordered = sorted(
            merged.values(),
            key=lambda r: (
                0 if r["ops"] & {"insert", "update", "delete"} else 1,
                r["name_cn"] or "",
                r["table_name"],
            ),
        )
        lines = [
            f"{word}相关接口涉及的数据库表（共 {len(ordered)} 张，union {len(group)} 个入口）：",
            "",
        ]
        for i, t in enumerate(ordered, 1):
            cn = t.get("name_cn") or t.get("entity_class") or ""
            head = f"{i}. `{t['table_name']}`"
            if cn:
                head += f"（{cn}）"
            lines.append(head)
            ops = sorted(t["ops"])
            if ops:
                lines.append(f"   - 操作：{'、'.join(ops)}")
            if t["fields"]:
                lines.append(f"   - 主要字段：{', '.join(t['fields'][:12])}")
            if t["via"]:
                lines.append(f"   - 经方法：{', '.join(t['via'][:4])}")
            lines.append("")
        lines.append(f"依据：{word}相关 {len(group)} 个接口（如 {', '.join(str(a.get('_entry') or '') for a in group[:4])}…）的调用链聚合。")
        answer = "\n".join(lines).strip()

        qs: list[str] = []
        for tpl in (
            "{word}接口都涉及哪些表",
            "{word}接口涉及哪些表",
            "{word}相关接口都涉及哪些表",
            "{word}相关接口涉及哪些表",
            "{word}都涉及哪些表",
            "{word}涉及哪些表",
            "{word}用到哪些表",
        ):
            q = tpl.format(word=word).strip()
            if q and q not in qs:
                qs.append(q)
        try:
            vecs = embed.embed(qs)
        except Exception:
            logger.warning("embed business-union questions failed", exc_info=True)
            vecs = []
        for idx, q in enumerate(qs):
            vec = vecs[idx] if idx < len(vecs) else None
            if _upsert_qa(project_id, q, answer, None, 1.0, [], vec, source="interface_tables"):
                created += 1
    return created


def _write_back_interface_tables(project_id: str, symbol_id: str, agg: dict[str, Any]) -> None:
    """把聚合结果写回 symbol_desc.extra_json.interface_tables（供运行时探索使用）。"""
    with get_session() as session:
        row = session.scalar(
            select(SymbolDescORM).where(
                SymbolDescORM.project_id == project_id,
                SymbolDescORM.symbol_id == symbol_id,
            )
        )
        if row is None:
            return
        extra = dict(row.extra_json) if isinstance(row.extra_json, dict) else {}
        extra["interface_tables"] = {
            "entry_cm": agg.get("entry_cm"),
            "flow_keys": agg.get("flow_keys"),
            "tables": [
                {
                    "table_name": t.get("table_name"),
                    "name_cn": t.get("name_cn"),
                    "ops": t.get("ops"),
                    "fields": (t.get("fields") or [])[:12],
                    "via": (t.get("via") or [])[:4],
                }
                for t in (agg.get("tables") or [])
            ],
        }
        row.extra_json = extra
        session.commit()


def clear_stale_interface_tables(project_id: str) -> dict[str, Any]:
    """删除旧的 interface_tables 黄金问答（重新生成前调用）。"""
    try:
        with get_session() as session:
            rows = session.scalars(
                select(GoldenQAORM).where(
                    GoldenQAORM.project_id == project_id,
                    GoldenQAORM.source == "interface_tables",
                )
            ).all()
            n = len(rows)
            for r in rows:
                session.delete(r)
            session.commit()
        return {"ok": True, "cleared": n}
    except Exception:
        logger.exception("clear stale interface_tables goldens failed")
        return {"ok": False, "cleared": 0}
