/**
 * General command runner with deny-list + timeout.
 * Dangerous actions must be confirmed in the renderer before calling here.
 */
const { spawn } = require("child_process");
const path = require("path");

const MAX_STDOUT = 24000;

const DENY_PATTERNS = [
  /format\s+[a-z]:/i,
  /rm\s+-rf\s+[\/\\]/i,
  /del\s+\/f\s+\/s\s+\/q\s+[a-z]:\\/i,
  /Remove-Item\s+-Recurse\s+-Force\s+[A-Z]:\\/i,
  /shutdown\b/i,
  /Stop-Computer/i,
  /rd\s+\/s\s+\/q\s+[a-z]:\\/i,
  /mkfs\b/i,
  /dd\s+if=/i,
  /:\(\)\s*\{\s*:\|:&\s*\}\s*;/i,
  /curl\s+.*\|\s*(iex|Invoke-Expression)/i,
  /wget\s+.*\|\s*(iex|Invoke-Expression)/i,
  /Invoke-Expression\s*\(\s*DownloadString/i,
  /reg\s+delete\s+HKLM/i,
  /net\s+user\s+\S+\s+\/add/i,
];

function truncate(s, n) {
  const t = String(s ?? "");
  if (t.length <= n) return t;
  return t.slice(0, Math.floor(n / 2)) + "\n…[truncated]…\n" + t.slice(-Math.floor(n / 2));
}

function assertAllowed(commandText) {
  const text = String(commandText || "");
  for (const re of DENY_PATTERNS) {
    if (re.test(text)) {
      throw new Error(`命令被安全策略拒绝（命中危险模式）: ${re}`);
    }
  }
}

function resolveCwd(root, cwd) {
  if (!root) throw new Error("未打开项目文件夹");
  const rel = String(cwd || ".").replace(/\\/g, "/");
  if (rel.includes("\0") || rel.startsWith("/") || /^[A-Za-z]:/.test(rel)) {
    // Allow absolute only if under root
    const absTry = path.resolve(rel);
    const rootWithSep = root.endsWith(path.sep) ? root : root + path.sep;
    if (absTry === root || absTry.startsWith(rootWithSep)) return absTry;
    throw new Error("cwd 越界：必须在项目根目录内");
  }
  const abs = path.resolve(root, rel);
  const rootWithSep = root.endsWith(path.sep) ? root : root + path.sep;
  if (abs !== root && !abs.startsWith(rootWithSep)) {
    throw new Error("cwd 越界：必须在项目根目录内");
  }
  return abs;
}

/**
 * @param {object} opts
 * @param {string} opts.root
 * @param {string} [opts.command]
 * @param {string[]} [opts.argv]
 * @param {string} [opts.cwd]
 * @param {number} [opts.timeoutMs]
 * @param {"powershell"|"cmd"|"none"} [opts.shellKind]
 * @param {(chunk:string, stream:string)=>void} [opts.onLog]
 */
function runCommand({
  root,
  command,
  argv,
  cwd,
  timeoutMs = 600000,
  shellKind = "powershell",
  onLog,
}) {
  const workdir = resolveCwd(root, cwd);
  let spawnCmd;
  let spawnArgs;
  let preview;

  if (Array.isArray(argv) && argv.length) {
    preview = argv.map(String).join(" ");
    assertAllowed(preview);
    spawnCmd = String(argv[0]);
    spawnArgs = argv.slice(1).map(String);
    // On Windows, bare commands often need shell
    return _spawn(spawnCmd, spawnArgs, {
      cwd: workdir,
      shell: process.platform === "win32",
      timeoutMs,
      onLog,
      preview,
    });
  }

  preview = String(command || "").trim();
  if (!preview) {
    return Promise.resolve({
      ok: false,
      error: "command/argv 为空",
      exit_code: 1,
    });
  }
  // Prefer cmd for chained -version probes so a broken PATH stub cannot
  // crash the whole PowerShell pipeline before useful where/version text.
  preview = normalizeProbeCommand(preview);
  assertAllowed(preview);

  if (shellKind === "cmd" || (shellKind === "none" && process.platform === "win32")) {
    spawnCmd = process.env.ComSpec || "cmd.exe";
    spawnArgs = ["/d", "/s", "/c", preview];
  } else if (shellKind === "none") {
    const parts = preview.split(/\s+/);
    spawnCmd = parts[0];
    spawnArgs = parts.slice(1);
  } else {
    // powershell default on Windows desktop
    spawnCmd = "powershell.exe";
    spawnArgs = [
      "-NoProfile",
      "-NonInteractive",
      "-ExecutionPolicy",
      "Bypass",
      "-Command",
      preview,
    ];
  }

  return _spawn(spawnCmd, spawnArgs, {
    cwd: workdir,
    shell: false,
    timeoutMs,
    onLog,
    preview,
  });
}

/**
 * Normalize version probes so a broken PATH stub (Oracle javapath) cannot
 * crash the whole pipeline before useful where/version text appears.
 * - `java -version` → `where.exe java; cmd /c "java -version 2>&1"`
 * - `where.exe X; X -version` → `where.exe X; cmd /c "X -version 2>&1"`
 */
function normalizeProbeCommand(command) {
  let c = String(command || "").trim();
  if (
    !/\bwhere(?:\.exe)?\b/i.test(c) &&
    /^[A-Za-z][A-Za-z0-9_.-]*\s+(-version|--version|-v)\b/i.test(c)
  ) {
    const m = c.match(
      /^([A-Za-z][A-Za-z0-9_.-]*)\s+(-version|--version|-v)\b(.*)$/i
    );
    if (m) {
      c = `where.exe ${m[1]}; cmd /c "${m[1]} ${m[2]}${m[3]} 2>&1"`;
    }
  }
  c = c.replace(
    /;\s*([A-Za-z][A-Za-z0-9_.-]*)\s+(-version|--version|-v)\b([^;&|]*)/gi,
    '; cmd /c "$1 $2$3 2>&1"'
  );
  return c;
}

function _spawn(command, args, { cwd, shell, timeoutMs, onLog, preview }) {
  return new Promise((resolve) => {
    let stdout = "";
    let stderr = "";
    let settled = false;
    let child;
    try {
      child = spawn(command, args, {
        cwd,
        env: process.env,
        shell: !!shell,
        windowsHide: true,
      });
    } catch (err) {
      resolve({ ok: false, error: err.message, exit_code: 1, meta: { preview } });
      return;
    }
    const timer = setTimeout(() => {
      try {
        if (process.platform === "win32" && child.pid) {
          spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
            windowsHide: true,
            shell: true,
          });
        } else {
          child.kill();
        }
      } catch {
        /* ignore */
      }
      if (!settled) {
        settled = true;
        resolve({
          ok: false,
          exit_code: 124,
          stdout: truncate(stdout, MAX_STDOUT),
          stderr: truncate(stderr, MAX_STDOUT),
          error: `timeout after ${timeoutMs}ms`,
          meta: { preview },
        });
      }
    }, timeoutMs);

    child.stdout?.on("data", (buf) => {
      const chunk = buf.toString("utf8");
      stdout += chunk;
      if (typeof onLog === "function") onLog(chunk, "stdout");
    });
    child.stderr?.on("data", (buf) => {
      const chunk = buf.toString("utf8");
      stderr += chunk;
      if (typeof onLog === "function") onLog(chunk, "stderr");
    });
    child.on("error", (err) => {
      clearTimeout(timer);
      if (settled) return;
      settled = true;
      resolve({
        ok: false,
        exit_code: 1,
        stdout: truncate(stdout, MAX_STDOUT),
        stderr: truncate(stderr + "\n" + err.message, MAX_STDOUT),
        error: err.message,
        meta: { preview },
      });
    });
    child.on("exit", (code) => {
      clearTimeout(timer);
      if (settled) return;
      settled = true;
      const out = truncate(stdout, MAX_STDOUT);
      const err = truncate(stderr, MAX_STDOUT);
      const soft = probeLooksSuccessful(preview, code, out, err);
      resolve({
        ok: soft,
        exit_code: code,
        stdout: out,
        stderr: err,
        error: soft ? null : `exit ${code}`,
        meta: {
          preview,
          soft_ok: soft && code !== 0,
        },
      });
    });
  });
}

/**
 * where / --version style probes often write to stderr (java -version)
 * or hit a broken stub on PATH (Oracle javapath crash) while still
 * producing useful path/version text. Treat those as success for UX.
 */
function probeLooksSuccessful(command, code, stdout, stderr) {
  if (code === 0) return true;
  const cmd = String(command || "");
  const blob = `${stdout || ""}\n${stderr || ""}`;
  const isProbe =
    /\bwhere(?:\.exe)?\b/i.test(cmd) ||
    /\bGet-Command\b/i.test(cmd) ||
    /\b(?:--version|-version|version)\b/i.test(cmd);
  if (!isProbe) return false;
  if (
    /(?:^|\n)\s*[A-Za-z]:\\[^\r\n]*\.(?:exe|bat|cmd)\s*(?:\r?\n|$)/i.test(blob) ||
    /(?:git version|java version|openjdk|node v|Python \d|Apache Maven|npm\s)/i.test(
      blob
    )
  ) {
    return true;
  }
  // Access violation / crash codes but prior where lines already printed paths.
  if (
    (code === -1073741819 || code === 3221225477) &&
    /[A-Za-z]:\\/i.test(blob)
  ) {
    return true;
  }
  return false;
}

module.exports = {
  runCommand,
  assertAllowed,
  DENY_PATTERNS,
  probeLooksSuccessful,
  normalizeProbeCommand,
};
