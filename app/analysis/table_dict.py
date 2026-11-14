"""表结构字典 + 表关联推导（Item 2）。

离线从项目 Entity 源码解析：
  * @TableName → 物理表名
  * 字段声明 + 前置注释（javadoc / @ApiModelProperty / // 注释）→ 字段字典
  * 中文表名（@ApiModel / 类 javadoc / Controller @Api tags 推导，尽力而为）

然后自动推导「表关联」关系（同名字段 + 命名语义），生成黄金问答对
（确定性答案，不经过 LLM），落 golden_qa 表 —— 运行时「A表和B表怎么关联的」
命中即答，不再依赖 qwen3 在线推理。

用法（scripts/generate_golden_qa.py 会自动调用）::

    python scripts/generate_golden_qa.py --build-table-dict
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.core.config import get_settings
from app.db.mysql import get_session
from app.models.orm import GoldenQAORM, TableDictORM

logger = logging.getLogger("app.analysis.table_dict")

# 字段命名中像"关联键"的尾缀（排除 id/主键/时间戳这类噪音）
_JOIN_NAME_RE = re.compile(r"(?i)(?:^|_)(order|loan|coupon|apply|deal|serial|credit|contract|limit|protocol|outer|out|trade|pay|agr|trans|flow|ref|biz|relate|merch|product|inst|cust|store|bank|union|quota|batch|channel|case|record|parent|child)?(?:id|no|code|serial|ref|seq|number|seqno|flowno)$")
_NOISE_COMMENT_RE = re.compile(r"(?:主键|创建时间|更新时间|createTime|updateTime|主键id|乐观锁|逻辑删除|version|remark|备注)")
_GENERIC_ID_FIELDS = {"id"}

_TABLENAME_RE = re.compile(r'@TableName\s*\(\s*(?:value\s*=\s*)?"([A-Za-z0-9_]+)"')
_PACKAGE_RE = re.compile(r"^\s*package\s+([\w.]+)\s*;")
_CLASS_RE = re.compile(r"\bclass\s+(\w+)")
_APIMODEL_RE = re.compile(r'@ApiModel\s*\(\s*"([^"]*)"')
_APIMODEL_DESC_RE = re.compile(r'@ApiModel\s*\(\s*(?:value\s*=\s*"[^"]*"\s*,\s*)?description\s*=\s*"([^"]*)"')
_APIPROPERTY_RE = re.compile(r'@ApiModelProperty\s*\(\s*(?:value\s*=\s*)?"([^"]*)"')
_APITAG_RE = re.compile(r'@Api\s*\(\s*tags\s*=\s*"([^"]*)"')
_FIELD_RE = re.compile(
    r"^\s*(?:private|protected|public)\s+([\w<>?\[\],\s.]+?)\s+(\w+)\s*;"
)
_JAVADOC_RE = re.compile(r"/\*\*(.*?)\*/", re.DOTALL)
_METHOD_OPEN_RE = re.compile(r"[\w<>\[\],\s]+\s+\w+\s*\([^)]*\)\s*(?:throws\s+[\w,.\s]+)?\{")


def _clean_comment(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^/\*\*|\*/$", "", s).strip()
    s = re.sub(r"^\s*\*\s?", "", s, flags=re.MULTILINE)
    s = re.sub(r"\s+", " ", s)
    return s.strip(" 　·-—")


def _sanitize_cn(cn: str) -> str:
    """清洗中文表名：去括号内容/数字/空白/英文字母，只留干净的业务名。

    「还款信息记录表(按借据)360」→「还款信息记录表」；
    「DEMO_放款审批记录表」→「放款审批记录表」；清洗后不合格返回空串。
    """
    s = (cn or "").strip()
    s = re.sub(r"[（(].*?[)）]", "", s)
    s = re.sub(r"[\d\s_\-—·]+", "", s)
    # 去掉英文/混合前缀（DEMO_、B_、demo_b_ 等），只保留连续中文
    m = re.search(r"[\u4e00-\u9fff]+", s)
    s = m.group(0) if m else ""
    if not s:
        return ""
    if not re.fullmatch(r"[\u4e00-\u9fff]{2,12}", s):
        return ""
    if re.search(r"(?:实体|类|API|Controller|实体类)$", s):
        return ""
    return s


def _extract_name_cn(entity_class: str, table_name: str, api_model: str, javadoc: str) -> str:
    """中文表名：@ApiModel(value|description) > 类 javadoc > Controller @Api tags > 空。"""
    cand = (api_model or "").strip().strip('"')
    if cand:
        # 「放款表实体类」→「放款表」；「优惠券表实体类」→「优惠券表」（保留"表"）
        cand = re.sub(r"实体(?:类)?$", "", cand).strip()
        clean = _sanitize_cn(cand)
        if clean:
            return clean
    jd = _clean_comment(javadoc)
    # 中文表名后允许跟空白/@(后续 javadoc 内容)而不仅是括号/行尾——
    # _clean_comment 会把多行 javadoc 合成单行（「风控异步调用记录表 @TableName …」），
    # 旧正则要求 (|（|$ 导致 35/85 表的中文名丢失（LoanBRiskAsync 等）。
    m = re.search(
        r"([\u4e00-\u9fff]{2,12}?(?:表|信息|记录|流水|单据|明细))"
        r"\s*(?:\(|（|$|[\s@。，,、；;]|实体)",
        jd,
    )
    if m:
        cn = _sanitize_cn(m.group(1))
        if cn and cn != table_name:
            return cn
    return ""


def _parse_entity_file(fp: Path, root: Path) -> dict[str, Any] | None:
    try:
        text = fp.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    pm = _PACKAGE_RE.search(text)
    pkg = pm.group(1) if pm else ""
    tm = _TABLENAME_RE.search(text)
    if not tm:
        return None
    table_name = tm.group(1)

    # 类名 + 附近 @ApiModel + javadoc（行级向上扫描，容忍注解夹在中间）
    entity_class = fp.stem
    api_model = ""
    javadoc = ""
    class_idx = -1
    for i, line in enumerate(text.splitlines()):
        m = re.search(r"\b(?:public\s+)?class\s+(\w+)", line)
        if m:
            entity_class = m.group(1)
            class_idx = i
            break
    if class_idx >= 0:
        jd_parts: list[str] = []
        for j in range(class_idx - 1, max(-1, class_idx - 14), -1):
            prev = text.splitlines()[j].strip()
            if not prev:
                continue
            am = _APIMODEL_RE.search(prev)
            if am:
                api_model = am.group(1)
            amd = _APIMODEL_DESC_RE.search(prev)
            if amd and not api_model:
                api_model = amd.group(1)
            if prev.startswith("*") or prev.startswith("/**") or prev.startswith("/*"):
                jd_parts.append(prev)
                continue
            if prev.startswith("@") or prev.startswith("public") or prev.startswith("import"):
                continue
            break
        if jd_parts:
            jd_parts.reverse()
            if jd_parts[0].startswith("/**"):
                javadoc = " ".join(jd_parts)
    fqcn = f"{pkg}.{entity_class}" if pkg else entity_class

    # 字段解析：行级扫描（实体是简单 POJO：@Data + 字段 + 注释）
    fields: list[dict[str, Any]] = []
    in_method = 0
    lines = text.splitlines()
    for i, line in enumerate(lines):
        line_s = line.strip()
        if not line_s:
            continue
        # 方法体跳过（实体一般没有方法，防御性处理）
        if _METHOD_OPEN_RE.search(line_s):
            in_method += 1
            continue
        if in_method:
            in_method += line_s.count("{") - line_s.count("}")
            if in_method <= 0:
                in_method = 0
            continue
        fm = _FIELD_RE.match(line_s)
        if fm:
            fname = fm.group(2)
            ftype = re.sub(r"\s+", " ", fm.group(1)).strip()
            # 向后回看最近注释：本行前的 javadoc 块 / @ApiModelProperty / // 行
            comment = ""
            for j in range(i - 1, max(-1, i - 16), -1):
                prev = lines[j].strip()
                if not prev:
                    continue
                # 先匹配 @ApiModelProperty（注意 @ApiModel 是它的前缀，不能先跳过）
                m = _APIPROPERTY_RE.search(prev)
                if m:
                    comment = _clean_comment(m.group(1))
                    break
                if prev.startswith(("@Accessors", "@Data", "@NoArgsConstructor", "import", "package", "@TableField", "@TableName", "@ApiModel(", "@SuppressWarnings")):
                    continue
                if prev.startswith("//"):
                    comment = _clean_comment(prev.lstrip("/"))
                    break
                if prev.startswith(("*", "/**", "/*")):
                    parts: list[str] = []
                    k = j
                    while k >= 0:
                        pk = lines[k].strip()
                        if pk.startswith("/**"):
                            parts.append(pk)
                            break
                        if pk.startswith("*") or pk.startswith("/*"):
                            parts.append(pk)
                            k -= 1
                            continue
                        break
                    if parts:
                        comment = _clean_comment("\n".join(reversed(parts)))
                    break
                if not comment:
                    continue
            fields.append({"name": fname, "type": ftype, "comment": comment[:200]})
    return {
        "table_name": table_name,
        "entity_class": entity_class,
        "entity_fqcn": fqcn,
        "name_cn": _extract_name_cn(entity_class, table_name, api_model, javadoc),
        "file_path": fp.relative_to(root).as_posix() if root else str(fp),
        "fields": fields,
    }


def _controller_tag_name_cn(root: Path, entity_class: str, table_name: str) -> str:
    """兜底中文名：找引用该实体的 XxxController，取其 @Api(tags=...) 去掉 API 尾缀。"""
    try:
        for fp in root.rglob("*.java"):
            if not fp.name.lower().endswith("controller.java"):
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
            if entity_class not in text:
                continue
            m = _APITAG_RE.search(text)
            if m:
                tag = m.group(1).strip()
                tag = re.sub(r"(?:API|Controller|接口|管理)$", "", tag).strip()
                clean = _sanitize_cn(tag)
                if clean:
                    return clean
    except Exception:
        pass
    return ""


def scan_entity_files(root: Path) -> list[dict[str, Any]]:
    """扫描项目根下所有带 @TableName 的 Entity 文件。"""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for fp in root.rglob("*.java"):
        if any(part in {"node_modules", ".git", "target"} for part in fp.parts):
            continue
        try:
            ent = _parse_entity_file(fp, root)
        except Exception:
            continue
        if not ent:
            continue
        if ent["table_name"] in seen:
            continue
        seen.add(ent["table_name"])
        if not ent["name_cn"]:
            ent["name_cn"] = _controller_tag_name_cn(root, ent["entity_class"], ent["table_name"])
        out.append(ent)
    return out


def build_table_dict(project_id: str, root: Path | str | None = None) -> dict[str, Any]:
    """全量解析 Entity → upsert table_dict。返回统计。"""
    settings = get_settings()
    root = Path(root or settings.project_root_path)
    entities = scan_entity_files(root)
    created = updated = 0
    try:
        with get_session() as session:
            existing = {
                r.table_name: r
                for r in session.scalars(
                    select(TableDictORM).where(TableDictORM.project_id == project_id)
                ).all()
            }
            for ent in entities:
                row = existing.get(ent["table_name"])
                if row is None:
                    session.add(
                        TableDictORM(
                            project_id=project_id,
                            table_name=ent["table_name"],
                            entity_fqcn=ent["entity_fqcn"],
                            entity_class=ent["entity_class"],
                            name_cn=ent["name_cn"] or None,
                            file_path=ent["file_path"],
                            fields_json=ent["fields"],
                        )
                    )
                    created += 1
                else:
                    row.entity_fqcn = ent["entity_fqcn"]
                    row.entity_class = ent["entity_class"]
                    row.name_cn = ent["name_cn"] or row.name_cn
                    row.file_path = ent["file_path"]
                    row.fields_json = ent["fields"]
                    updated += 1
            session.commit()
    except Exception:
        logger.exception("build_table_dict failed")
        return {"ok": False, "tables": 0, "created": 0, "updated": 0, "error": "db"}
    return {"ok": True, "tables": len(entities), "created": created, "updated": updated}


def load_table_dict(project_id: str) -> list[dict[str, Any]]:
    try:
        with get_session() as session:
            rows = session.scalars(
                select(TableDictORM).where(TableDictORM.project_id == project_id)
            ).all()
            out = []
            for r in rows:
                out.append(
                    {
                        "table_name": r.table_name,
                        "entity_class": r.entity_class,
                        "entity_fqcn": r.entity_fqcn,
                        "name_cn": r.name_cn,
                        "file_path": r.file_path,
                        "fields": r.fields_json or [],
                    }
                )
            return out
    except Exception:
        return []


def _key_priority(field_name: str, comment: str) -> int:
    """关联键质量：业务主键(0) > 注释含编号/流水号(1) > 其他(2)。"""
    n = (field_name or "").lower()
    if n in {
        "orderid", "couponid", "loanid", "applyid", "applycode", "dealrefno",
        "serialnumber", "creditserialno", "outloanid", "contractid",
        "protocolcode", "limitno", "tradeid", "outtradeno", "outorderid",
        "loanapplycode", "refundno", "cancellationno",
    }:
        return 0
    c = (comment or "") or ""
    if re.search(r"(编号|流水号|单号|号|id)", c):
        return 1
    return 2


def derive_table_relations(project_id: str) -> list[dict[str, Any]]:
    """同名字段 + 关联键命名 → 表关联候选（每对表最多保留 2 个最优键）。"""
    tables = load_table_dict(project_id)
    idx: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for t in tables:
        for f in t["fields"]:
            n = f["name"].lower()
            if n in _GENERIC_ID_FIELDS:
                continue
            if _NOISE_COMMENT_RE.search(f["comment"] or ""):
                continue
            idx.setdefault(n, []).append((t, f))
    by_pair: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen_rel: set[tuple[str, str, str]] = set()
    for t in tables:
        for f in t["fields"]:
            n = f["name"].lower()
            if n in _GENERIC_ID_FIELDS or not _JOIN_NAME_RE.search(n):
                continue
            if _NOISE_COMMENT_RE.search(f["comment"] or ""):
                continue
            peers = idx.get(n) or []
            for t2, f2 in peers:
                if t2["table_name"] == t["table_name"]:
                    continue
                pair = tuple(sorted([t["table_name"], t2["table_name"]]))
                rel_key = (pair[0], pair[1], n)
                if rel_key in seen_rel:
                    continue  # 同一 (表对, 键) 只保留一次，避免交换顺序的重复
                seen_rel.add(rel_key)
                by_pair.setdefault(pair, []).append(
                    {
                        "table1": t["table_name"],
                        "cn1": t["name_cn"] or t["entity_class"] or t["table_name"],
                        "entity1": t["entity_class"] or t["table_name"],
                        "file1": t["file_path"] or "",
                        "key": f["name"],
                        "priority": _key_priority(f["name"], f["comment"]),
                        "comment1": f["comment"] or "",
                        "table2": t2["table_name"],
                        "cn2": t2["name_cn"] or t2["entity_class"] or t2["table_name"],
                        "entity2": t2["entity_class"] or t2["table_name"],
                        "file2": t2["file_path"] or "",
                        "comment2": f2["comment"] or "",
                    }
                )
    relations: list[dict[str, Any]] = []
    for pair, cands in by_pair.items():
        cands.sort(key=lambda r: (r["priority"], len(r["key"])))
        relations.extend(cands[:2])
    # 稳定性排序：优先"两边中文名都干净"的（业务意义最强）
    def _cn_rank(r: dict[str, Any]) -> int:
        c1 = re.fullmatch(r"[\u4e00-\u9fff]{2,10}", r["cn1"]) is not None
        c2 = re.fullmatch(r"[\u4e00-\u9fff]{2,10}", r["cn2"]) is not None
        return 0 if (c1 and c2) else (1 if (c1 or c2) else 2)

    relations.sort(
        key=lambda r: (
            _cn_rank(r),
            r["table1"],
            r["table2"],
            r["priority"],
            r["key"],
        )
    )
    return relations


def _relation_questions(r: dict[str, Any]) -> list[str]:
    qs: list[str] = []
    cn1, cn2 = r["cn1"], r["cn2"]
    if re.search(r"[\u4e00-\u9fff]", cn1) and re.search(r"[\u4e00-\u9fff]", cn2):
        for pat in (
            f"{cn1}和{cn2}怎么关联的",
            f"{cn1}和{cn2}通过什么字段关联",
            f"{cn1}和{cn2}是什么关系",
            f"{cn1}和{cn2}怎么关联",
        ):
            qs.append(pat)
        if not cn1.endswith("表") or not cn2.endswith("表"):
            qs.append(f"{cn1}表和{cn2}表怎么关联")
    e1, e2 = r["entity1"], r["entity2"]
    qs.append(f"{e1}表和{e2}表怎么关联的")
    qs.append(f"{r['table1']}和{r['table2']}怎么关联")
    # 去重保序
    seen: set[str] = set()
    out: list[str] = []
    for q in qs:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def _relation_answer(r: dict[str, Any]) -> str:
    c1 = f"（{r['comment1']}）" if r["comment1"] else ""
    c2 = f"（{r['comment2']}）" if r["comment2"] else ""
    lines = [
        f"`{r['table1']}`（{r['cn1']}）与 `{r['table2']}`（{r['cn2']}）通过 **{r['key']}** 字段关联：",
        "",
        f"- {r['table1']}.{r['key']} = {r['table2']}.{r['key']}",
        f"- {r['cn1']}.{r['key']} {c1}",
        f"- {r['cn2']}.{r['key']} {c2}",
        "",
        "依据：",
        f"- `{r['entity1']}` 实体（{r['file1']}）",
        f"- `{r['entity2']}` 实体（{r['file2']}）",
    ]
    return "\n".join(lines)


def _field_list_answer(t: dict[str, Any]) -> str:
    fields = t["fields"] or []
    cn = t["name_cn"] or t["entity_class"] or t["table_name"]
    lines = [f"{t['table_name']}（{cn}）字段（共 {len(fields)} 个）：", ""]
    for f in fields[:60]:
        comment = f"  // {f['comment']}" if f.get("comment") else ""
        lines.append(f"- {f['name']} ({f['type']}){comment}")
    lines.append("")
    lines.append(f"依据：`{t['entity_class']}` 实体（{t['file_path'] or '?'}）")
    return "\n".join(lines)


def build_table_field_goldens(
    project_id: str,
    *,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """为表字典里的每张表生成「X表有哪些字段」黄金问答（确定性，不耗 LLM）。

    解决「放款表有哪些字段」这类问题被表关联黄金误命中、答非所问的问题。
    """
    from app.analysis.golden_qa import _embed_one_or_none, _upsert_qa
    from app.llm.factory import get_embedding_provider

    tables = load_table_dict(project_id)
    embed = get_embedding_provider()
    created = 0
    skipped = 0
    seen_q: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for t in tables:
        if not t["fields"]:
            continue
        answer = _field_list_answer(t)
        cn = t["name_cn"] or t["entity_class"] or t["table_name"]
        qs = []
        if re.search(r"[\u4e00-\u9fff]", cn):
            qs.append(f"{cn}有哪些字段")
            qs.append(f"{cn}的表结构是什么")
            if not cn.endswith("表"):
                qs.append(f"{cn}表有哪些字段")
        qs.append(f"{t['table_name']}有哪些字段")
        qs.append(f"{t['table_name']}的表结构")
        for q in qs:
            if q in seen_q:
                skipped += 1
                continue
            seen_q.add(q)
            pairs.append((q, answer))
        if on_progress:
            on_progress({"table": t["table_name"], "cn": cn, "fields": len(t["fields"] or []), "status": "ok"})
    if pairs:
        try:
            vecs = embed.embed([q for q, _ in pairs])
        except Exception:
            logger.warning("batch embed field questions failed", exc_info=True)
            vecs = []
        for idx, (q, answer) in enumerate(pairs):
            vec = vecs[idx] if idx < len(vecs) else None
            if _upsert_qa(project_id, q, answer, None, 1.0, [], vec, source="table_field"):
                created += 1
            else:
                skipped += 1
    return {"tables": len(tables), "questions": len(pairs), "created": created, "skipped": skipped}


def build_table_relation_goldens(
    project_id: str,
    *,
    max_relations: int = 500,
    on_progress: Any | None = None,
) -> dict[str, Any]:
    """从表字典推导关联 → 生成确定性黄金问答对落 golden_qa。

    重建语义：先清掉该项目旧的 source=table_relation 行，再按当前字典重插，
    避免「同问句被弱关联键覆盖」的历史脏数据残留。
    """
    from app.analysis.golden_qa import _embed_one_or_none, _upsert_qa
    from app.llm.factory import get_embedding_provider

    # 清旧
    try:
        with get_session() as session:
            stale = session.scalars(
                select(GoldenQAORM).where(
                    GoldenQAORM.project_id == project_id,
                    GoldenQAORM.source == "table_relation",
                )
            ).all()
            for row in stale:
                session.delete(row)
            session.commit()
    except Exception:
        logger.warning("clear stale table_relation goldens failed", exc_info=True)

    relations = derive_table_relations(project_id)[: max(1, max_relations)]
    embed = get_embedding_provider()
    created = 0
    skipped = 0
    # 先收集 (question, answer)，再批量 embedding（provider 内部按 32 分批），
    # 避免上千条逐条 HTTP 调用超时
    pairs: list[tuple[str, str]] = []
    seen_q: set[str] = set()
    for i, r in enumerate(relations, 1):
        answer = _relation_answer(r)
        for q in _relation_questions(r):
            if q in seen_q:
                skipped += 1
                continue  # 同问句先到先得，防止弱关联键覆盖强关联键
            seen_q.add(q)
            pairs.append((q, answer))
        if on_progress:
            on_progress(
                {
                    "index": i,
                    "total": min(len(relations), max(1, max_relations)),
                    "relation": f"{r['cn1']} ~ {r['cn2']} via {r['key']}",
                    "status": "ok",
                }
            )
    if pairs:
        try:
            vecs = embed.embed([q for q, _ in pairs])
        except Exception:
            logger.warning("batch embed relation questions failed", exc_info=True)
            vecs = []
        for idx, (q, answer) in enumerate(pairs):
            vec = vecs[idx] if idx < len(vecs) else None
            if _upsert_qa(
                project_id,
                q,
                answer,
                None,
                1.0,
                [],
                vec,
                source="table_relation",
            ):
                created += 1
            else:
                skipped += 1
    return {
        "relations": min(len(relations), max(1, max_relations)),
        "questions": len(pairs),
        "created": created,
        "skipped": skipped,
    }


def table_dict_stats(project_id: str) -> dict[str, Any]:
    try:
        with get_session() as session:
            rows = session.scalars(
                select(TableDictORM).where(TableDictORM.project_id == project_id)
            ).all()
            total = len(rows)
            with_cn = sum(1 for r in rows if r.name_cn)
            field_count = sum(len(r.fields_json or []) for r in rows)
            relations = len(derive_table_relations(project_id))
    except Exception:
        return {"ok": False, "error": "table_dict unavailable"}
    return {
        "ok": True,
        "tables": total,
        "with_name_cn": with_cn,
        "fields": field_count,
        "relations": relations,
    }
