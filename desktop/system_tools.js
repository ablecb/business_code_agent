/**
 * Whitelisted Windows local system tools (disk scan + safe cleanup).
 * No arbitrary shell from model strings.
 */
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawn } = require("child_process");

function formatBytes(n) {
  const v = Number(n) || 0;
  if (v < 1024) return `${v} B`;
  const u = ["KB", "MB", "GB", "TB"];
  let x = v;
  let i = -1;
  do {
    x /= 1024;
    i += 1;
  } while (x >= 1024 && i < u.length - 1);
  return `${x.toFixed(i >= 2 ? 1 : 0)} ${u[i]}`;
}

function runPs(script, timeoutMs = 120000) {
  return new Promise((resolve) => {
    const child = spawn(
      "powershell.exe",
      ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script],
      { windowsHide: true }
    );
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
      resolve({ ok: false, stdout, stderr, error: "timeout" });
    }, timeoutMs);
    child.stdout.on("data", (b) => {
      stdout += b.toString("utf8");
    });
    child.stderr.on("data", (b) => {
      stderr += b.toString("utf8");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      resolve({ ok: false, stdout, stderr, error: err.message });
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      resolve({ ok: code === 0, exit_code: code, stdout, stderr });
    });
  });
}

function dirSizeBytes(root, maxDepth = 2, maxEntries = 80) {
  let total = 0;
  let walked = 0;
  function walk(dir, depth) {
    if (walked > 5000) return;
    let entries;
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true });
    } catch {
      return;
    }
    for (const ent of entries) {
      walked += 1;
      if (walked > 5000) return;
      const p = path.join(dir, ent.name);
      try {
        if (ent.isFile()) {
          total += fs.statSync(p).size;
        } else if (ent.isDirectory() && depth < maxDepth) {
          walk(p, depth + 1);
        } else if (ent.isDirectory()) {
          // shallow: count immediate files only
          try {
            for (const f of fs.readdirSync(p, { withFileTypes: true }).slice(0, maxEntries)) {
              if (f.isFile()) {
                try {
                  total += fs.statSync(path.join(p, f.name)).size;
                } catch {
                  /* ignore */
                }
              }
            }
          } catch {
            /* ignore */
          }
        }
      } catch {
        /* ignore */
      }
    }
  }
  walk(root, 0);
  return total;
}

function safeCleanupProfiles() {
  const home = os.homedir();
  const local = process.env.LOCALAPPDATA || path.join(home, "AppData", "Local");
  const roaming = process.env.APPDATA || path.join(home, "AppData", "Roaming");
  const winTemp = process.env.TEMP || path.join(home, "AppData", "Local", "Temp");
  return [
    {
      id: "user_temp",
      label: "用户临时目录 (Temp)",
      path: winTemp,
      safe: true,
      kind: "clear_dir_contents",
    },
    {
      id: "win_temp",
      label: "系统临时目录 Windows\\Temp",
      path: "C:\\Windows\\Temp",
      safe: true,
      kind: "clear_dir_contents",
    },
    {
      id: "recycle_bin",
      label: "回收站",
      path: "RECYCLE_BIN",
      safe: true,
      kind: "recycle_bin",
    },
    {
      id: "edge_cache",
      label: "Edge 缓存",
      path: path.join(local, "Microsoft", "Edge", "User Data", "Default", "Cache"),
      safe: true,
      kind: "clear_dir_contents",
    },
    {
      id: "chrome_cache",
      label: "Chrome 缓存",
      path: path.join(local, "Google", "Chrome", "User Data", "Default", "Cache"),
      safe: true,
      kind: "clear_dir_contents",
    },
    {
      id: "npm_cache",
      label: "npm 缓存",
      path: path.join(roaming, "npm-cache"),
      safe: true,
      kind: "clear_dir_contents",
    },
    {
      id: "pip_cache",
      label: "pip 缓存",
      path: path.join(local, "pip", "Cache"),
      safe: true,
      kind: "clear_dir_contents",
    },
  ];
}

async function diskOverview(drive = "C") {
  const letter = String(drive || "C").replace(/[^A-Za-z]/g, "").slice(0, 1).toUpperCase() || "C";
  const script = `
$d = Get-PSDrive -Name '${letter}' -ErrorAction SilentlyContinue
if (-not $d) { Write-Output '{"ok":false,"error":"drive not found"}'; exit 0 }
$used = [int64]$d.Used
$free = [int64]$d.Free
$total = $used + $free
$obj = [ordered]@{
  ok = $true
  drive = '${letter}:'
  used_bytes = $used
  free_bytes = $free
  total_bytes = $total
  used_human = ('{0:N1} GB' -f ($used/1GB))
  free_human = ('{0:N1} GB' -f ($free/1GB))
  total_human = ('{0:N1} GB' -f ($total/1GB))
  pct_used = [math]::Round(100.0 * $used / [math]::Max($total,1), 1)
}
$obj | ConvertTo-Json -Compress
`;
  const r = await runPs(script, 30000);
  try {
    const data = JSON.parse((r.stdout || "").trim() || "{}");
    return {
      ok: !!data.ok,
      stdout: JSON.stringify(data, null, 2),
      content: JSON.stringify(data),
      meta: data,
      error: data.error || r.error || null,
    };
  } catch {
    return { ok: false, error: r.error || "parse failed", stdout: r.stdout, stderr: r.stderr };
  }
}

async function scanHeavy(drive = "C") {
  const letter = String(drive || "C").replace(/[^A-Za-z]/g, "").slice(0, 1).toUpperCase() || "C";
  const profiles = safeCleanupProfiles();
  const candidates = [];

  for (const p of profiles) {
    if (p.kind === "recycle_bin") {
      candidates.push({
        ...p,
        size_bytes: null,
        size_human: "未知（清空回收站）",
        exists: true,
      });
      continue;
    }
    const exists = fs.existsSync(p.path);
    let size = 0;
    if (exists) {
      try {
        size = dirSizeBytes(p.path, 2);
      } catch {
        size = 0;
      }
    }
    candidates.push({
      ...p,
      exists,
      size_bytes: size,
      size_human: exists ? formatBytes(size) : "不存在",
    });
  }

  // Top-level folders under drive root (names + rough size, limited)
  const root = `${letter}:\\`;
  const top = [];
  try {
    const entries = fs.readdirSync(root, { withFileTypes: true }).filter((e) => e.isDirectory());
    const interesting = entries.filter((e) =>
      /^(Users|Program Files|Program Files \(x86\)|ProgramData|Windows)$/i.test(e.name)
    );
    for (const ent of interesting.slice(0, 8)) {
      const p = path.join(root, ent.name);
      let size = 0;
      try {
        size = dirSizeBytes(p, 1);
      } catch {
        size = 0;
      }
      top.push({
        id: `top_${ent.name.replace(/\W+/g, "_")}`,
        label: `${letter}:\\${ent.name}`,
        path: p,
        safe: false,
        kind: "report_only",
        exists: true,
        size_bytes: size,
        size_human: formatBytes(size),
        note: "仅分析，不自动删除系统目录",
      });
    }
  } catch (err) {
    /* ignore */
  }

  // Large dirs under user profile (Downloads, AppData\Local)
  const home = os.homedir();
  for (const rel of ["Downloads", path.join("AppData", "Local")]) {
    const p = path.join(home, rel);
    if (!fs.existsSync(p)) continue;
    let size = 0;
    try {
      size = dirSizeBytes(p, 2);
    } catch {
      size = 0;
    }
    const id = `user_${rel.replace(/\W+/g, "_")}`;
    candidates.push({
      id,
      label: `用户目录 ${rel}`,
      path: p,
      safe: /Downloads/i.test(rel), // Downloads: report; cleanup only clear empty? mark safe=false for delete whole
      kind: "report_only",
      exists: true,
      size_bytes: size,
      size_human: formatBytes(size),
      note: /Downloads/i.test(rel)
        ? "下载目录体积（默认不自动删文件，仅提示）"
        : "AppData\\Local 体积（默认不自动整目录删除）",
    });
  }

  const payload = {
    ok: true,
    drive: `${letter}:`,
    cleanup_candidates: candidates.filter((c) => c.safe && c.kind !== "report_only"),
    report_only: [...top, ...candidates.filter((c) => !c.safe || c.kind === "report_only")],
    all: [...candidates, ...top],
  };
  return {
    ok: true,
    stdout: JSON.stringify(payload, null, 2),
    content: JSON.stringify(payload),
    meta: payload,
  };
}

function clearDirContents(dirPath) {
  if (!dirPath || dirPath === "RECYCLE_BIN") {
    return { ok: false, error: "invalid path" };
  }
  const resolved = path.resolve(dirPath);
  // Refuse wiping drive roots / Windows tree wholesale
  const denied = [
    "C:\\",
    "C:\\Windows",
    "C:\\Windows\\System32",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
  ].map((p) => path.resolve(p).toLowerCase());
  if (denied.includes(resolved.toLowerCase())) {
    return { ok: false, error: "拒绝清理受保护路径" };
  }
  if (!fs.existsSync(resolved)) {
    return { ok: true, stdout: "路径不存在，跳过", meta: { skipped: true } };
  }
  let removed = 0;
  let failed = 0;
  for (const name of fs.readdirSync(resolved)) {
    const p = path.join(resolved, name);
    try {
      fs.rmSync(p, { recursive: true, force: true, maxRetries: 1 });
      removed += 1;
    } catch {
      failed += 1;
    }
  }
  return {
    ok: true,
    stdout: `已清理 ${resolved}：删除 ${removed} 项，失败 ${failed} 项`,
    meta: { removed, failed, path: resolved },
  };
}

async function emptyRecycleBin() {
  const script = `Clear-RecycleBin -Force -ErrorAction SilentlyContinue; Write-Output '{"ok":true}'`;
  const r = await runPs(script, 60000);
  return {
    ok: r.ok || /ok/i.test(r.stdout || ""),
    stdout: r.stdout || "已尝试清空回收站",
    stderr: r.stderr,
    error: r.ok ? null : r.error,
  };
}

async function cleanupSelected(ids, catalog) {
  const byId = new Map();
  for (const c of catalog || []) byId.set(c.id, c);
  const profiles = safeCleanupProfiles();
  for (const p of profiles) byId.set(p.id, p);

  const results = [];
  for (const id of ids || []) {
    const item = byId.get(id);
    if (!item) {
      results.push({ id, ok: false, error: "未知清理项" });
      continue;
    }
    if (!item.safe || item.kind === "report_only") {
      results.push({ id, ok: false, error: "该项仅可分析，不能自动清理" });
      continue;
    }
    if (item.kind === "recycle_bin") {
      const r = await emptyRecycleBin();
      results.push({ id, label: item.label, ...r });
      continue;
    }
    if (item.kind === "clear_dir_contents") {
      const r = clearDirContents(item.path);
      results.push({
        id,
        label: item.label,
        path: item.path,
        ...r,
        stdout: r.ok
          ? `${r.stdout || "ok"} · 路径 ${item.path}`
          : r.stdout,
      });
      continue;
    }
    results.push({ id, ok: false, error: `不支持的 kind: ${item.kind}` });
  }
  const ok = results.every((r) => r.ok);
  return {
    ok,
    stdout: results.map((r) => `${r.label || r.id}: ${r.ok ? r.stdout || "ok" : r.error}`).join("\n"),
    content: JSON.stringify({ results }, null, 2),
    meta: { results },
  };
}

async function executeSystemAction(action) {
  const type = action?.type;
  const drive = action?.drive || action?.path || "C";
  if (type === "sys_disk_overview") return diskOverview(drive);
  if (type === "sys_scan_heavy") return scanHeavy(drive);
  if (type === "sys_cleanup") {
    const ids = action?.ids || action?.option_ids || [];
    let catalog = [];
    try {
      catalog = JSON.parse(action?.catalog_json || "[]");
    } catch {
      catalog = [];
    }
    return cleanupSelected(ids, catalog);
  }
  throw new Error(`不支持的系统动作: ${type}`);
}

module.exports = {
  executeSystemAction,
  safeCleanupProfiles,
  formatBytes,
};
