"""Compile / test / snapshot / rollback helpers for implement→verify→repair."""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from app.core.config import get_settings
from app.core.utils import truncate

logger = logging.getLogger("app.agent.verify_loop")

DEFAULT_MAX_REPAIRS = 3


def snapshot_dir() -> Path:
    settings = get_settings()
    root = settings.workspace_path.parent / ".agent_snapshots"
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_workspace_snapshot(*, label: str = "") -> dict[str, Any]:
    """Copy current agent workspace (or project if inplace) to a snapshot id."""
    settings = get_settings()
    src = (
        settings.project_root_path
        if settings.agent_write_mode == "inplace"
        else settings.workspace_path
    )
    sid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    dst = snapshot_dir() / sid
    if src.exists():
        shutil.copytree(
            src,
            dst,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(
                ".git",
                "target",
                "node_modules",
                ".idea",
                "*.class",
            ),
        )
    else:
        dst.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": sid,
        "label": label or "",
        "src": str(src),
        "path": str(dst),
        "created_at": time.time(),
        "write_mode": settings.agent_write_mode,
    }
    (dst / "_snapshot_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta


def rollback_workspace(snapshot_id: str | None) -> dict[str, Any]:
    """Restore workspace/project from snapshot. Returns status dict."""
    if not snapshot_id:
        return {"ok": False, "error": "missing snapshot_id"}
    settings = get_settings()
    src = snapshot_dir() / snapshot_id
    if not src.exists():
        return {"ok": False, "error": f"snapshot not found: {snapshot_id}"}
    dst = (
        settings.project_root_path
        if settings.agent_write_mode == "inplace"
        else settings.workspace_path
    )
    if dst.exists():
        # Clear destination contents carefully (keep root)
        for child in dst.iterdir():
            if child.name in {".git"}:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except OSError:
                    pass
    else:
        dst.mkdir(parents=True, exist_ok=True)
    for child in src.iterdir():
        if child.name == "_snapshot_meta.json":
            continue
        target = dst / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
    return {"ok": True, "snapshot_id": snapshot_id, "restored_to": str(dst)}


def detect_build_root(project_root: Path | None = None) -> tuple[Path, str]:
    """Return (build_root, tool) where tool is mvn|gradle|none."""
    settings = get_settings()
    root = project_root or settings.project_root_path
    # Prefer workspace overlay root when patches were applied there and it has a build file
    candidates = []
    if settings.agent_write_mode != "inplace":
        candidates.append(settings.workspace_path)
    candidates.append(root)
    for base in candidates:
        if (base / "pom.xml").exists():
            return base, "mvn"
        if (base / "build.gradle").exists() or (base / "build.gradle.kts").exists():
            return base, "gradle"
        # multi-module: search one level
        try:
            for child in base.iterdir():
                if not child.is_dir():
                    continue
                if (child / "pom.xml").exists():
                    return child, "mvn"
                if (child / "build.gradle").exists() or (
                    child / "build.gradle.kts"
                ).exists():
                    return child, "gradle"
        except OSError:
            pass
    return root, "none"


def _run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    timeout: float = 600.0,
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
            env={**os.environ, "MAVEN_OPTS": os.environ.get("MAVEN_OPTS", "-Xmx1g")},
        )
        out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
        return {
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "cmd": cmd,
            "cwd": str(cwd),
            "log": truncate(out.strip(), 12000),
        }
    except FileNotFoundError as e:
        return {
            "ok": False,
            "returncode": 127,
            "cmd": cmd,
            "cwd": str(cwd),
            "log": f"command not found: {e}",
            "error": str(e),
        }
    except subprocess.TimeoutExpired as e:
        return {
            "ok": False,
            "returncode": 124,
            "cmd": cmd,
            "cwd": str(cwd),
            "log": f"timeout after {timeout}s: {e}",
            "error": "timeout",
        }


def run_compile(project_root: Path | None = None) -> dict[str, Any]:
    build_root, tool = detect_build_root(project_root)
    if tool == "mvn":
        # Windows mvn.cmd often needed; try mvn then mvn.cmd
        for bin_name in ("mvn", "mvn.cmd"):
            result = _run_cmd(
                [bin_name, "-q", "-DskipTests", "compile"],
                cwd=build_root,
                timeout=900.0,
            )
            if result.get("returncode") != 127:
                result["tool"] = "mvn"
                result["phase"] = "compile"
                return result
        return {
            "ok": False,
            "tool": "mvn",
            "phase": "compile",
            "log": "mvn not found on PATH",
            "returncode": 127,
        }
    if tool == "gradle":
        gradlew = build_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        cmd = (
            [str(gradlew), "compileJava", "-q"]
            if gradlew.exists()
            else ["gradle", "compileJava", "-q"]
        )
        result = _run_cmd(cmd, cwd=build_root, timeout=900.0)
        result["tool"] = "gradle"
        result["phase"] = "compile"
        return result
    return {
        "ok": False,
        "tool": "none",
        "phase": "compile",
        "log": f"no pom.xml/build.gradle under {build_root}",
        "returncode": 2,
    }


def run_tests(
    project_root: Path | None = None,
    *,
    changed_files: list[str] | None = None,
) -> dict[str, Any]:
    """Run tests; if none discoverable, treat as skipped (ok=True, skipped=True)."""
    del changed_files  # reserved for future targeted test selection
    build_root, tool = detect_build_root(project_root)
    # Prefer a cheap compile-bound test if surefire exists; otherwise skip.
    test_dirs = list(build_root.rglob("src/test/java"))
    if not test_dirs:
        return {
            "ok": True,
            "skipped": True,
            "tool": tool,
            "phase": "test",
            "log": "no src/test/java found; skipped tests",
            "returncode": 0,
        }
    if tool == "mvn":
        for bin_name in ("mvn", "mvn.cmd"):
            result = _run_cmd(
                [bin_name, "-q", "test"],
                cwd=build_root,
                timeout=1200.0,
            )
            if result.get("returncode") != 127:
                result["tool"] = "mvn"
                result["phase"] = "test"
                result["skipped"] = False
                return result
        return {
            "ok": False,
            "skipped": False,
            "tool": "mvn",
            "phase": "test",
            "log": "mvn not found on PATH",
            "returncode": 127,
        }
    if tool == "gradle":
        gradlew = build_root / ("gradlew.bat" if os.name == "nt" else "gradlew")
        cmd = [str(gradlew), "test", "-q"] if gradlew.exists() else ["gradle", "test", "-q"]
        result = _run_cmd(cmd, cwd=build_root, timeout=1200.0)
        result["tool"] = "gradle"
        result["phase"] = "test"
        result["skipped"] = False
        return result
    return {
        "ok": True,
        "skipped": True,
        "tool": "none",
        "phase": "test",
        "log": "no build tool; skipped tests",
        "returncode": 0,
    }


async def emit_progress(on_progress: Any, event: dict[str, Any]) -> None:
    if not on_progress:
        return
    try:
        maybe = on_progress(event)
        if hasattr(maybe, "__await__"):
            await maybe  # type: ignore[misc]
    except Exception:
        logger.debug("verify progress emit failed", exc_info=True)


async def run_verify_once(
    *,
    changed_files: list[str] | None = None,
    on_progress: Any = None,
    attempt: int = 1,
) -> dict[str, Any]:
    """One compile (+ optional test) pass."""
    await emit_progress(
        on_progress,
        {"type": "verify", "status": "running", "note": f"编译中（第 {attempt} 次）"},
    )
    compile_res = run_compile()
    await emit_progress(
        on_progress,
        {
            "type": "verify",
            "status": "done" if compile_res.get("ok") else "failed",
            "note": "编译通过" if compile_res.get("ok") else "编译失败",
            "compile": {
                "ok": compile_res.get("ok"),
                "log": truncate(str(compile_res.get("log") or ""), 2000),
            },
        },
    )
    test_res: dict[str, Any] = {
        "ok": True,
        "skipped": True,
        "log": "compile failed; tests not run",
        "returncode": 0,
    }
    if compile_res.get("ok"):
        await emit_progress(
            on_progress,
            {"type": "verify", "status": "running", "note": f"测试中（第 {attempt} 次）"},
        )
        test_res = run_tests(changed_files=changed_files)
        await emit_progress(
            on_progress,
            {
                "type": "verify",
                "status": "done" if test_res.get("ok") else "failed",
                "note": (
                    "测试跳过"
                    if test_res.get("skipped")
                    else ("测试通过" if test_res.get("ok") else "测试失败")
                ),
                "test": {
                    "ok": test_res.get("ok"),
                    "skipped": test_res.get("skipped"),
                    "log": truncate(str(test_res.get("log") or ""), 2000),
                },
            },
        )
    ok = bool(compile_res.get("ok")) and bool(test_res.get("ok"))
    report = {
        "ok": ok,
        "compile_ok": bool(compile_res.get("ok")),
        "test_ok": bool(test_res.get("ok")),
        "test_skipped": bool(test_res.get("skipped")),
        "attempts": attempt,
        "logs": truncate(
            f"[compile]\n{compile_res.get('log') or ''}\n\n[test]\n{test_res.get('log') or ''}",
            16000,
        ),
        "compile": compile_res,
        "test": test_res,
    }
    return report


def build_repair_prompt(
    *,
    requirement: str,
    verify_report: dict[str, Any],
    patches: list[dict[str, Any]] | None,
    scope: list[dict[str, Any]] | None,
) -> str:
    return (
        "上一轮补丁在编译/测试时失败，请根据错误日志修复并重新输出补丁。\n"
        "补丁格式：\n### FILE: relative/path.java\n*** OLD\n旧代码\n*** NEW\n新代码\n"
        "新增文件：### FILE: path\n*** NEW_FILE\n完整内容\n\n"
        f"原始需求: {requirement}\n"
        f"修改范围: {json.dumps(scope or [], ensure_ascii=False)}\n"
        f"上一轮补丁: {json.dumps(patches or [], ensure_ascii=False)[:6000]}\n"
        f"验证报告: {truncate(str((verify_report or {}).get('logs') or ''), 8000)}\n"
    )


def render_verify_reply(report: dict[str, Any], *, repaired: bool = False) -> str:
    lines = ["## 验证结果", ""]
    if repaired:
        lines.append(f"（含自动修复，共尝试 {report.get('attempts') or 1} 次）")
        lines.append("")
    lines.append(f"- 编译: {'通过' if report.get('compile_ok') else '失败'}")
    if report.get("test_skipped"):
        lines.append("- 测试: 跳过（未发现测试目录或未配置）")
    else:
        lines.append(f"- 测试: {'通过' if report.get('test_ok') else '失败'}")
    lines.append("")
    if report.get("ok"):
        lines.append("改动已通过验证。")
    else:
        lines.append("验证未通过。关键日志：")
        lines.append("```")
        lines.append(truncate(str(report.get("logs") or ""), 4000))
        lines.append("```")
    return "\n".join(lines)
