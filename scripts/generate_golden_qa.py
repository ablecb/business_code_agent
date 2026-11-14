#!/usr/bin/env python
"""离线生成黄金问答库（方案一：把重推理搬到离线）。

用当前配置的 LLM（内网 qwen3-32b 等）对索引里的流程入口/关键接口生成高置信
问答对，自一致性投票后落 MySQL golden_qa 表。运行时 chat 对 ask 类问题先查
本库，命中高分直接返回存储答案 —— qwen3 在线只做检索/组装，不做长链推理。

用法::

    python scripts/generate_golden_qa.py                          # 全量（按配置上限）
    python scripts/generate_golden_qa.py --project demo-loan --limit 100
    python scripts/generate_golden_qa.py --samples 3 --overwrite
    python scripts/generate_golden_qa.py --dry-run                # 只列候选，不调模型
    python scripts/generate_golden_qa.py --stats                  # 只看库存统计

要点:
    * 脚本强制开启思考模式（LLM_ENABLE_THINKING=true，qwen3 生成质量提升明显），
      只影响本脚本进程，不改 .env 的运行时默认值。
    * 候选来自 FlowDoc 流程入口（每题 2 问）+ Controller 方法（每题 1 问）。
    * 先确保已 ingest：python scripts/run_ingest.py --full-refresh
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 强制思考模式 —— 必须在首次 import app.*（首次 get_settings）之前设置
os.environ.setdefault("LLM_ENABLE_THINKING", "true")


def _print_progress(event: dict) -> None:
    i = event.get("index", 0)
    total = event.get("total", 0)
    q = str(event.get("question") or "")[:50]
    status = event.get("status")
    votes = event.get("votes")
    print(f"[{i}/{total}] {status} votes={votes}  {q}", flush=True)


def _print_interface_progress(event: dict) -> None:
    i = event.get("index", 0)
    total = event.get("total", 0)
    entry = str(event.get("entry") or "")[:50]
    tables = event.get("tables", 0)
    status = event.get("status")
    print(f"[{i}/{total}] {status} tables={tables}  {entry}", flush=True)


async def _run(args: argparse.Namespace) -> int:
    from app.analysis.golden_qa import (
        build_golden_qa,
        candidate_questions_from_index,
        golden_qa_stats,
    )
    from app.analysis.table_dict import (
        build_table_dict,
        build_table_field_goldens,
        build_table_relation_goldens,
        table_dict_stats,
    )
    from app.core.config import get_settings

    settings = get_settings()
    project_id = args.project or settings.project_id

    if args.interface_tables_only:
        from app.analysis.interface_tables import (
            build_interface_tables,
            clear_stale_interface_tables,
        )

        print("== 接口→表聚合（确定性，不耗 LLM）==")
        cleared = clear_stale_interface_tables(project_id)
        print(f"清理旧 interface_tables 黄金: {cleared}")
        st = build_interface_tables(project_id, max_entries=args.limit or 0, on_progress=_print_interface_progress)
        print(f"接口表聚合: {st}")
        return 0

    if args.table_dict_only:
        st = build_table_dict(project_id, root=None)
        print(f"表结构字典构建: {st}")
        rel = build_table_relation_goldens(project_id, max_relations=args.max_relations)
        print(f"表关联黄金问答: {rel}")
        fld = build_table_field_goldens(project_id)
        print(f"表字段黄金问答: {fld}")
        print(f"字典统计: {table_dict_stats(project_id)}")
        return 0

    if args.stats:
        print("golden_qa:", golden_qa_stats(project_id or None))
        print("table_dict:", table_dict_stats(project_id or ""))
        return 0

    if args.dry_run:
        cands = candidate_questions_from_index(
            project_id, limit=int(args.limit or settings.golden_qa_max_candidates or 0)
        )
        print(f"候选 {len(cands)} 条（project={project_id}）:")
        for c in cands[: max(args.preview, 1)]:
            print(
                f"  [{c.get('kind')}] {c.get('question')}  entry={c.get('entry_cm')}"
            )
        print("\n表关联（table_dict 推导）:")
        from app.analysis.table_dict import derive_table_relations

        for r in derive_table_relations(project_id)[: max(args.preview, 1)]:
            print(
                f"  [{r['cn1']} ~ {r['cn2']} via {r['key']}] "
                f"{r['table1']} <=> {r['table2']}"
            )
        return 0

    # 1) 表结构字典 + 表关联黄金问答（确定性，不耗模型）
    print("== 步骤 0/3：表结构字典 + 表关联黄金问答 ==")
    st = build_table_dict(project_id, root=None)
    print("字典:", st)
    rel = build_table_relation_goldens(project_id, max_relations=args.max_relations)
    print("表关联黄金问答:", rel)

    # 1.5) 接口→表聚合黄金问答（确定性，不耗模型）
    from app.analysis.interface_tables import (
        build_interface_tables,
        clear_stale_interface_tables,
    )

    print("== 步骤 0.5/3：接口→表聚合黄金问答 ==")
    cleared = clear_stale_interface_tables(project_id)
    print(f"清理旧 interface_tables 黄金: {cleared}")
    ist = build_interface_tables(project_id, max_entries=args.limit or 0, on_progress=_print_interface_progress)
    print("接口表聚合:", ist)

    # 2) 常规候选（FlowDoc/Controller）LLM 生成
    print("== 步骤 1/3：常规候选问答生成（思考模式 + 自一致性投票）==")
    print(
        f"project={project_id} "
        f"limit={args.limit or settings.golden_qa_max_candidates} "
        f"samples={args.samples or settings.golden_qa_n_samples} "
        f"overwrite={args.overwrite}"
    )
    print("LLM_ENABLE_THINKING=true（脚本强制，思考模式提升生成质量）")
    stats = await build_golden_qa(
        project_id,
        limit=args.limit or 0,
        n_samples=args.samples or None,
        overwrite=args.overwrite,
        on_progress=_print_progress,
    )
    print(f"\n完成: {stats}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="离线生成黄金问答库")
    parser.add_argument("--project", default="", help="project_id（默认取 .env PROJECT_ID）")
    parser.add_argument("--limit", type=int, default=0, help="候选上限（0=配置默认）")
    parser.add_argument("--samples", type=int, default=0, help="每题采样数（0=配置默认3）")
    parser.add_argument("--overwrite", action="store_true", help="已存在的问题也重新生成")
    parser.add_argument("--dry-run", action="store_true", help="只列出候选问题，不调用模型")
    parser.add_argument("--stats", action="store_true", help="只看库存统计")
    parser.add_argument("--preview", type=int, default=20, help="--dry-run 预览条数")
    parser.add_argument("--table-dict-only", action="store_true", help="只构建表结构字典 + 表关联黄金问答（不调 LLM）")
    parser.add_argument("--interface-tables-only", action="store_true", help="只构建接口→表聚合黄金问答（不调 LLM）")
    parser.add_argument("--max-relations", type=int, default=500, help="表关联生成上限")
    args = parser.parse_args()
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
