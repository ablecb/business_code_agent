"""日志查看与日志驱动问题定位 API.

- GET  /api/v1/logs/meta     浏览可用的日志文件（loan-service.log / err / xxl-job 目录树）
- GET  /api/v1/logs/tail     读取日志尾部 N 行，支持正则筛选（排查用）
- POST /api/v1/logs/diagnose 给定日志文本（或日志文件+筛选），返回结构化代码问题诊断

诊断链路复用 app/agent/log_diagnosis（锚点提取 → 锚点检索 → 源码证据 → LLM 诊断），
与 /api/v1/chat 的 log_diagnose 意图共享同一实现。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.config import ROOT_DIR

router = APIRouter(tags=["logs"])

_MAX_TAIL_LINES = 5000


def _logs_base() -> Path:
    base = ROOT_DIR / "java_project"
    return base if base.is_dir() else ROOT_DIR


def _is_inside(root: Path, p: Path) -> bool:
    try:
        p.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_log_path(rel_path: str) -> Path:
    root = _logs_base()
    target = (root / rel_path).resolve()
    if not _is_inside(root, target):
        raise HTTPException(status_code=400, detail=f"非法路径: {rel_path}")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"日志文件不存在: {rel_path}")
    return target


@router.get("/logs/meta")
def logs_meta() -> dict[str, Any]:
    """列出 java_project/*/logs 下的日志文件（含 xxl-job 定时任务日志目录树）。"""
    root = _logs_base()
    entries: list[dict[str, Any]] = []
    for proj_dir in sorted(root.iterdir()) if root.is_dir() else []:
        if not proj_dir.is_dir():
            continue
        logs_dir = proj_dir / "logs"
        if not logs_dir.is_dir():
            continue
        project_name = proj_dir.name
        for p in sorted(logs_dir.rglob("*")):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            try:
                from datetime import datetime

                mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
            except OSError:
                mtime = ""
            entries.append(
                {
                    "project": project_name,
                    "path": str(p.relative_to(root)).replace("\\", "/"),
                    "name": p.name,
                    "size": size,
                    "mtime": mtime,
                }
            )
    return {"projects_root": str(root), "count": len(entries), "files": entries[:500]}


@router.get("/logs/tail")
def logs_tail(
    path: str,
    lines: int = Query(default=300, ge=1, le=_MAX_TAIL_LINES),
    pattern: str | None = None,
) -> dict[str, Any]:
    """读取日志尾部 N 行；pattern 非空时只返回命中行（上下各带 2 行上下文）。"""
    target = _safe_log_path(path)
    try:
        raw = target.read_text(encoding="utf-8", errors="ignore")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"读取失败: {e}") from e
    rows = raw.splitlines()
    rex = None
    if pattern:
        try:
            rex = re.compile(pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"正则非法: {e}") from e
    out: list[dict[str, Any]] = []
    start = max(0, len(rows) - lines)
    ctx = 2
    for i in range(start, len(rows)):
        if rex is None:
            out.append({"line": i + 1, "text": rows[i]})
            continue
        if rex.search(rows[i]):
            for j in range(max(start, i - ctx), min(len(rows), i + ctx + 1)):
                out.append({"line": j + 1, "text": rows[j]})
    return {"path": path, "count": len(out), "lines": out}


class DiagnoseRequest(BaseModel):
    project_id: str = Field(default="", description="目标工程，缺省取默认工程")
    log_text: str = Field(default="", description="日志文本（与 file 二选一）")
    file: str = Field(default="", description="日志文件相对路径（与 log_text 二选一）")
    lines: int = Field(default=400, ge=1, le=_MAX_TAIL_LINES, description="读取尾部行数")
    pattern: str | None = Field(default=None, description="正则筛选（可选）")
    question: str | None = Field(default=None, description="补充描述（可选）")


@router.post("/logs/diagnose")
async def logs_diagnose(req: DiagnoseRequest) -> dict[str, Any]:
    """结构化代码问题诊断：输入日志文本/文件 → 输出现象/定位/根因/为什么/修复/证据链。"""
    from app.agent.log_diagnosis import run_log_diagnosis
    from app.core.config import get_settings

    if req.log_text.strip():
        log_text = req.log_text
    elif req.file.strip():
        target = _safe_log_path(req.file)
        try:
            raw = target.read_text(encoding="utf-8", errors="ignore")
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"读取失败: {e}") from e
        rows = raw.splitlines()
        rex = re.compile(req.pattern) if req.pattern else None
        picked = [rows[i] for i in range(max(0, len(rows) - req.lines), len(rows))]
        if rex:
            picked = [r for r in picked if rex.search(r)]
        log_text = "\n".join(picked[:2000])
        if not log_text:
            raise HTTPException(status_code=400, detail="按 pattern 未命中任何行")
    else:
        raise HTTPException(status_code=400, detail="log_text 与 file 至少提供一个")

    project_id = req.project_id or get_settings().project_id
    if req.question and req.question.strip():
        log_text = f"{req.question.strip()}\n\n【生产日志】\n{log_text}"

    result = await run_log_diagnosis(log_text, project_id=project_id, on_progress=None)
    return {
        "intent": "log_diagnose",
        "reply": result["reply"],
        "anchors": result["anchors"],
        "steps": result["steps"],
        "branch_hit": result.get("branch_hit", False),
        "project_id": project_id,
    }