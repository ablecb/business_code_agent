"""Policy for client actions: confirm gates and risk tiers."""
from __future__ import annotations

import re
from typing import Any

# Always require explicit user confirm in the desktop UI before execute.
CONFIRM_ALWAYS: frozenset[str] = frozenset(
    {
        "run_shell",
        "run_powershell",
        "git_push",
        "git_merge",
        "git_commit",
        "sys_cleanup",
        "milvus_mutate",
    }
)

# Soft confirm (recommended) — still confirmed by default for non-technical users.
CONFIRM_SOFT: frozenset[str] = frozenset(
    {
        "start_app",
        "stop_app",
        "maven_install",
        "docker_compose_up",
        "git_checkout",
        "git_pull",
        "open_browser",
    }
)

# Never auto-confirm these even if server sets needs_confirm=false.
# Note: run_shell / run_powershell are NOT here — read-only probes may skip confirm.
FORCE_CONFIRM: frozenset[str] = frozenset(
    {
        "git_push",
        "sys_cleanup",
        "milvus_mutate",
    }
)

# Structured actions that only inspect state (never mutate).
READ_ONLY_ACTIONS: frozenset[str] = frozenset(
    {
        "list_dir",
        "read_file",
        "open_file",
        "git_status",
        "git_branch",
        "sys_disk_overview",
        "sys_scan_heavy",
        "http_smoke",
        "milvus_search",
    }
)

# If command matches any of these, it is NOT a safe probe.
_SHELL_MUTATING_RE = re.compile(
    r"(?is)(?:"
    r"\b(?:rm|rmdir|del|erase|move|mv|copy|cp|ren|rename)\b|"
    r"\b(?:Remove-Item|Move-Item|Copy-Item|Rename-Item|New-Item|Set-Content|"
    r"Add-Content|Out-File|Clear-Content|Set-Item|New-ItemProperty|Set-ItemProperty|"
    r"Stop-Process|Kill|Stop-Service|Restart-Service|Remove-Service|"
    r"Clear-RecycleBin|Format-Volume|diskpart)\b|"
    r"(?:^|[;&|])\s*(?:git\s+(?:push|commit|merge|rebase|reset|clean|checkout|add|rm|mv)\b)|"
    r"(?:^|[;&|])\s*(?:npm\s+(?:install|i|uninstall|publish)\b|pip\s+install\b|mvn\s+install\b)|"
    # File redirects; allow 2>&1 / 1>&2 style stream redirects used by version probes.
    r"(?<!\d)>{1,2}(?!&\d)"
    r")"
)

# Entire command must be one or more read-only segments (`;` / `&` separated).
_SHELL_READONLY_SEG_RE = re.compile(
    r"(?is)^\s*(?:"
    r"where(?:\.exe)?\b|"
    r"which\b|"
    r"Get-Command\b|"
    r"Get-ChildItem\b|\bdir\b|\bls\b|"
    r"Get-Content\b|\btype\b|\bcat\b|"
    r"Get-Item\b|Test-Path\b|Resolve-Path\b|"
    r"Write-Host\b|Write-Output\b|\becho\b|"
    r"Get-PSDrive\b|Get-Location\b|pwd\b|"
    r"hostname\b|whoami\b|"
    r"printenv\b|"
    r"Get-ChildItem\s+Env:|"
    r"\$env:[A-Za-z_][\w]*|"
    r"echo\s+%[A-Za-z_][\w]*%|"
    r"cmd(?:\.exe)?\s+/c\s+|"
    r"git\s+(?:--version|-v|version|status|branch|remote|log|show|diff|"
    r"rev-parse|config\s+--get|config\s+--list)\b|"
    r"(?:java|javac|mvn|maven|node|npm|npx|python|python3|pip|pip3|docker|kubectl|"
    r"dotnet|go|rustc|cargo)\s+(?:--version|-v|version|-version)\b|"
    r"[A-Za-z0-9_.\\/-]+\s+(?:--version|-v|-version)\b"
    r").*$"
)

_SQL_READONLY_HEAD_RE = re.compile(
    r"(?is)^\s*(?:/\*.*?\*/\s*)*(?:--[^\n]*\n\s*)*"
    r"(?:WITH\b|SELECT\b|SHOW\b|DESCRIBE\b|DESC\b|EXPLAIN\b)"
)

_SQL_WRITE_RE = re.compile(
    r"(?is)\b(?:"
    r"INSERT|UPDATE|DELETE|REPLACE|UPSERT|MERGE|TRUNCATE|DROP|ALTER|CREATE|"
    r"GRANT|REVOKE|CALL|LOAD\s+DATA|LOCK\s+TABLES|UNLOCK\s+TABLES|"
    r"RENAME|FLUSH|OPTIMIZE|ANALYZE\s+TABLE|HANDLER|DO\b"
    r")\b"
)

_CYPHER_WRITE_RE = re.compile(
    r"(?is)\b(?:"
    r"CREATE|MERGE|DELETE|SET|DROP|REMOVE|DETACH|FOREACH|"
    r"LOAD\s+CSV|CALL\s*\{"
    r")\b"
)


def is_readonly_shell_command(command: str | None) -> bool:
    """True when the shell/PowerShell command only inspects (no mutate)."""
    raw = (command or "").strip()
    if not raw:
        return False
    if _SHELL_MUTATING_RE.search(raw):
        return False
    # Split on ; / && only — do NOT split on & inside 2>&1 redirects.
    parts = re.split(r"\s*(?:&&|;)\s*", raw)
    parts = [p.strip() for p in parts if p and p.strip()]
    if not parts:
        return False
    return all(_SHELL_READONLY_SEG_RE.match(p) is not None for p in parts)


def is_readonly_sql(sql: str | None) -> bool:
    """True when SQL only reads (SELECT/SHOW/DESC/EXPLAIN/WITH…SELECT)."""
    raw = (sql or "").strip()
    if not raw:
        return False
    # Strip trailing semicolons; reject multi-statement mixes.
    parts = [p.strip() for p in raw.split(";") if p.strip()]
    if not parts:
        return False
    for part in parts:
        if not _SQL_READONLY_HEAD_RE.match(part):
            return False
        # WITH … INSERT etc.
        if _SQL_WRITE_RE.search(part) and not re.match(
            r"(?is)^\s*(?:/\*.*?\*/\s*)*(?:--[^\n]*\n\s*)*(?:SELECT|SHOW|DESCRIBE|DESC|EXPLAIN)\b",
            part,
        ):
            # Allow SELECT that mentions write keywords only as identifiers is rare;
            # if head is WITH/SELECT and body has INSERT/UPDATE — treat as write.
            if re.search(
                r"(?is)\b(?:INSERT|UPDATE|DELETE|REPLACE|TRUNCATE|DROP|ALTER|CREATE|GRANT|REVOKE)\b",
                part,
            ):
                # SELECT … FROM — UPDATE as column name unlikely; be conservative for DML verbs
                # after WITH only when followed by non-SELECT.
                if re.search(
                    r"(?is)\b(?:INSERT|UPDATE|DELETE|REPLACE|TRUNCATE|DROP|ALTER|CREATE)\b",
                    part,
                ) and not re.match(
                    r"(?is)^\s*(?:/\*.*?\*/\s*)*(?:--[^\n]*\n\s*)*SELECT\b",
                    part,
                ):
                    # WITH x AS (...) INSERT — write
                    if re.search(
                        r"(?is)\)\s*(?:INSERT|UPDATE|DELETE|REPLACE|CREATE)\b",
                        part,
                    ):
                        return False
    return True


def is_readonly_cypher(query: str | None) -> bool:
    """True when Cypher has no write keywords (MATCH/RETURN/WITH/UNWIND only)."""
    raw = (query or "").strip()
    if not raw:
        return False
    if _CYPHER_WRITE_RE.search(raw):
        return False
    return bool(
        re.search(r"(?is)\b(?:MATCH|RETURN|WITH|UNWIND|OPTIONAL|CALL|SHOW|EXPLAIN|PROFILE)\b", raw)
    )


def action_tier(atype: str) -> str:
    t = atype or ""
    if t in {"run_shell", "run_powershell"}:
        return "shell"
    if t.startswith("git_"):
        return "git"
    if t.startswith("sys_"):
        return "sys"
    if t in {"start_app", "stop_app", "docker_compose_up"}:
        return "process"
    if t in {"write_file", "apply_patch", "maven_compile", "maven_install"}:
        return "code"
    if t in {"run_sql", "run_cypher", "milvus_search", "milvus_mutate"}:
        return "db"
    if t == "open_browser":
        return "browser"
    return "read"


def needs_confirm(atype: str, *, explicit: bool | None = None) -> bool:
    if atype in READ_ONLY_ACTIONS:
        return False
    if atype in FORCE_CONFIRM:
        return True
    if explicit is True:
        return True
    if explicit is False and atype not in FORCE_CONFIRM:
        return False
    return atype in CONFIRM_ALWAYS or atype in CONFIRM_SOFT


def needs_confirm_action(action: dict[str, Any]) -> bool:
    """Confirm policy for a full action payload (command-aware for shell)."""
    atype = str(action.get("type") or "")
    if atype in READ_ONLY_ACTIONS:
        return False
    if atype in {"run_shell", "run_powershell"}:
        cmd = str(action.get("command") or "")
        argv = action.get("argv") or []
        if not cmd and isinstance(argv, list):
            cmd = " ".join(str(x) for x in argv)
        if is_readonly_shell_command(cmd):
            return False
        return True
    if atype == "run_sql":
        if is_readonly_sql(str(action.get("sql") or "")):
            return False
        return True
    if atype == "run_cypher":
        q = str(action.get("query") or action.get("cypher") or "")
        if is_readonly_cypher(q):
            return False
        return True
    if atype == "milvus_mutate":
        return True
    explicit = action.get("needs_confirm")
    if isinstance(explicit, bool):
        return needs_confirm(atype, explicit=explicit)
    return needs_confirm(atype)


def annotate_action_dict(action: dict[str, Any]) -> dict[str, Any]:
    """Attach policy fields for the client UI (mutates a copy)."""
    out = dict(action or {})
    atype = str(out.get("type") or "")
    cmd = str(out.get("command") or "")
    if not cmd and isinstance(out.get("argv"), list):
        cmd = " ".join(str(x) for x in out.get("argv") or [])
    readonly = atype in READ_ONLY_ACTIONS or (
        atype in {"run_shell", "run_powershell"} and is_readonly_shell_command(cmd)
    )
    if atype == "run_sql" and is_readonly_sql(str(out.get("sql") or "")):
        readonly = True
    if atype == "run_cypher" and is_readonly_cypher(
        str(out.get("query") or out.get("cypher") or "")
    ):
        readonly = True
    out["read_only"] = bool(readonly)
    out["needs_confirm"] = needs_confirm_action(out)
    out["policy_tier"] = "read" if readonly else action_tier(atype)
    if not out.get("ui_title"):
        out["ui_title"] = _default_title(atype, out)
    if atype == "git_checkout" and (
        out.get("create") or out.get("new_branch") or out.get("create_branch")
    ):
        out["ui_title"] = "新建并切换分支"
    if readonly and atype in {"run_shell", "run_powershell"}:
        out["ui_title"] = "查看本机信息"
    if readonly and atype == "run_sql":
        out["ui_title"] = "查询 MySQL"
    if readonly and atype == "run_cypher":
        out["ui_title"] = "查询 Neo4j"
    # Always refresh preview so path/target is visible for non-technical users.
    # Password is masked only in ui_preview (_mask_mysql_preview), not in mysql payload
    # (desktop must forward real credentials to /api/connectors/mysql).
    out["ui_preview"] = _default_preview(atype, out)
    return out


def _default_title(atype: str, a: dict[str, Any]) -> str:
    mapping = {
        "run_shell": "执行命令",
        "run_powershell": "执行 PowerShell",
        "git_status": "查看 Git 状态",
        "git_branch": "查看分支",
        "git_checkout": "切换/新建分支",
        "git_pull": "拉取远端",
        "git_merge": "合并分支",
        "git_commit": "提交变更",
        "git_push": "推送到远端",
        "maven_compile": "Maven 编译",
        "maven_install": "Maven 安装依赖",
        "start_app": "启动应用",
        "stop_app": "停止应用",
        "http_smoke": "HTTP 冒烟自测",
        "apply_patch": "应用补丁",
        "write_file": "写入文件",
        "read_file": "读取文件",
        "open_file": "打开文件",
        "list_dir": "列出目录",
        "sys_cleanup": "清理磁盘项",
        "sys_disk_overview": "磁盘概况",
        "sys_scan_heavy": "扫描占用",
        "open_browser": "打开浏览器",
        "run_sql": "执行 SQL",
        "run_cypher": "执行 Cypher",
        "milvus_search": "查询 Milvus",
        "milvus_mutate": "修改 Milvus",
    }
    return mapping.get(atype, atype or "本机动作")


def _display_path(path: Any) -> str:
    p = str(path or "").strip()
    if not p:
        return ""
    if p.upper() in {"RECYCLE_BIN", "RECYCLEBIN"}:
        return "系统回收站"
    return p


def _mask_mysql_preview(mysql: Any) -> str:
    if not isinstance(mysql, dict) or not mysql:
        return "Settings MySQL"
    host = mysql.get("host") or "?"
    port = mysql.get("port") or 3306
    user = mysql.get("user") or "?"
    db = mysql.get("database") or "?"
    return f"{user}@{host}:{port}/{db}"


def _cleanup_target_lines(a: dict[str, Any]) -> list[str]:
    """Build human-readable cleanup targets with absolute paths."""
    import json

    ids = [str(x) for x in (a.get("ids") or []) if x]
    catalog: list[dict[str, Any]] = []
    raw = a.get("catalog_json")
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                catalog = [c for c in parsed if isinstance(c, dict)]
        except json.JSONDecodeError:
            catalog = []
    elif isinstance(a.get("catalog"), list):
        catalog = [c for c in a["catalog"] if isinstance(c, dict)]

    by_id = {
        str(c.get("id")): c
        for c in catalog
        if c.get("id") is not None
    }
    lines: list[str] = []
    for i in ids or list(by_id.keys()):
        c = by_id.get(str(i)) or {}
        label = str(c.get("label") or c.get("title") or i)
        path = _display_path(c.get("path"))
        size = str(c.get("size_human") or "").strip()
        bit = label
        if path:
            bit += f"\n  路径: {path}"
        if size:
            bit += f"\n  大小: {size}"
        lines.append(bit)
    if not lines and ids:
        lines.append("将清理: " + ", ".join(ids))
    return lines


def _default_preview(atype: str, a: dict[str, Any]) -> str:
    """Always prefer concrete target (path/command/url) so users can judge risk."""
    lines: list[str] = []

    if atype in {"run_shell", "run_powershell"}:
        cmd = a.get("command") or ""
        argv = a.get("argv") or []
        if cmd:
            lines.append(f"命令: {cmd}")
        elif isinstance(argv, list) and argv:
            lines.append("命令: " + " ".join(str(x) for x in argv))
        if a.get("cwd"):
            lines.append(f"工作目录: {a.get('cwd')}")
        if a.get("read_only"):
            lines.append("（只读查看，无需确认）")
        return "\n".join(lines)[:800] or atype

    if atype == "open_browser":
        url = str(a.get("url") or "").strip()
        if url:
            lines.append(f"地址: {url}")
        browser = str(a.get("browser") or "default").strip() or "default"
        lines.append(f"浏览器: {browser}")
        return "\n".join(lines)[:800] or atype

    if atype == "run_sql":
        lines.append(f"连接: {_mask_mysql_preview(a.get('mysql'))}")
        sql = str(a.get("sql") or "").strip()
        if sql:
            lines.append(f"SQL: {sql[:500]}")
        if a.get("read_only"):
            lines.append("（只读查询，无需确认）")
        return "\n".join(lines)[:800] or atype

    if atype == "run_cypher":
        q = str(a.get("query") or a.get("cypher") or "").strip()
        lines.append("连接: Settings Neo4j")
        if q:
            lines.append(f"Cypher: {q[:500]}")
        if a.get("read_only"):
            lines.append("（只读查询，无需确认）")
        return "\n".join(lines)[:800] or atype

    if atype in {"milvus_search", "milvus_mutate"}:
        lines.append("连接: Settings Milvus")
        if a.get("collection"):
            lines.append(f"集合: {a.get('collection')}")
        if a.get("op"):
            lines.append(f"操作: {a.get('op')}")
        filt = a.get("filter") or a.get("expr")
        if filt:
            lines.append(f"过滤: {str(filt)[:300]}")
        if a.get("limit") is not None:
            lines.append(f"limit: {a.get('limit')}")
        return "\n".join(lines)[:800] or atype

    if atype == "http_smoke":
        urls = []
        if a.get("url"):
            urls.append(str(a.get("url")))
        if isinstance(a.get("urls"), list):
            urls.extend(str(u) for u in a["urls"] if u)
        if urls:
            lines.append("探测地址:")
            lines.extend(f"  - {u}" for u in urls[:6])
        if a.get("expect_status") is not None:
            lines.append(f"期望状态码: {a.get('expect_status')}")
        return "\n".join(lines)[:800] or atype

    if atype == "sys_cleanup":
        targets = _cleanup_target_lines(a)
        if targets:
            lines.append("将清理以下目标（请核对路径）:")
            lines.extend(targets)
        return "\n".join(lines)[:1200] or "清理磁盘项（未提供路径明细）"

    if atype in {"sys_disk_overview", "sys_scan_heavy"}:
        drive = (a.get("drive") or a.get("path") or "C")
        drive_s = str(drive).strip()
        if len(drive_s) == 1:
            drive_s = f"{drive_s.upper()}:\\"
        lines.append(f"目标磁盘: {drive_s}")
        return "\n".join(lines)

    if atype.startswith("git_"):
        bits = [f"操作: {atype}"]
        if atype == "git_checkout" and (
            a.get("create") or a.get("new_branch") or a.get("create_branch")
        ):
            bits[0] = "操作: 新建并切换分支 (checkout -b)"
        for k, label in (
            ("branch", "分支"),
            ("ref", "引用"),
            ("remote", "远端"),
            ("message", "说明"),
        ):
            if a.get(k):
                bits.append(f"{label}: {a.get(k)}")
        if a.get("cwd") or a.get("path"):
            bits.append(f"仓库路径: {a.get('cwd') or a.get('path')}")
        return "\n".join(bits)[:500]

    if atype in {"write_file", "apply_patch", "read_file", "open_file", "list_dir"}:
        p = _display_path(a.get("path"))
        if p:
            lines.append(f"路径: {p}")
        if atype == "apply_patch":
            reps = a.get("replacements") or []
            lines.append(f"替换段数: {len(reps) if isinstance(reps, list) else 0}")
        if a.get("note"):
            lines.append(str(a.get("note"))[:200])
        return "\n".join(lines)[:800] or atype

    if atype in {"start_app", "stop_app", "maven_compile", "maven_install", "docker_compose_up"}:
        if a.get("path") or a.get("cwd"):
            lines.append(f"项目路径: {a.get('path') or a.get('cwd')}")
        if a.get("note"):
            lines.append(str(a.get("note"))[:200])
        if atype == "start_app" and a.get("wait_ms"):
            lines.append(f"等待: {a.get('wait_ms')} ms")
        return "\n".join(lines)[:500] or _default_title(atype, a)

    # Generic fallback — surface any concrete target fields.
    for key, label in (
        ("path", "路径"),
        ("cwd", "工作目录"),
        ("drive", "磁盘"),
        ("command", "命令"),
        ("url", "地址"),
        ("branch", "分支"),
        ("note", "说明"),
    ):
        if a.get(key):
            val = _display_path(a.get(key)) if key in {"path", "cwd"} else a.get(key)
            lines.append(f"{label}: {val}")
    if isinstance(a.get("urls"), list) and a["urls"]:
        lines.append("地址: " + ", ".join(str(u) for u in a["urls"][:4]))
    return "\n".join(lines)[:800] or atype
