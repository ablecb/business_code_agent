/**
 * Local whitelist executor for client patch-verify / start / OS assist.
 * No arbitrary shell — only named actions.
 */
const fs = require("fs");
const path = require("path");
const http = require("http");
const https = require("https");
const { spawn } = require("child_process");
const { URL } = require("url");
const { shell } = require("electron");
const { executeSystemAction } = require("./system_tools");
const { runCommand } = require("./shell_gate");
const { executeGit } = require("./git_ops");
const { runHttpSmoke } = require("./smoke_http");

const MAX_STDOUT = 24000;
const DEFAULT_API_BASE = "http://127.0.0.1:8000";

const CHROME_CANDIDATES = [
  process.env.CHROME_PATH,
  "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
  "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
].filter(Boolean);

function truncate(s, n) {
  const t = String(s ?? "");
  if (t.length <= n) return t;
  return t.slice(0, Math.floor(n / 2)) + "\n…[truncated]…\n" + t.slice(-Math.floor(n / 2));
}

class LocalExecutor {
  constructor() {
    this.root = null;
    this.apiBase = DEFAULT_API_BASE;
    /** @type {import('child_process').ChildProcess|null} */
    this.appChild = null;
    this.appLog = "";
  }

  setApiBase(base) {
    const t = String(base || "").trim().replace(/\/+$/, "");
    this.apiBase = t || DEFAULT_API_BASE;
    return this.apiBase;
  }

  setRoot(rootPath) {
    if (!rootPath) {
      this.root = null;
      return { ok: false, error: "empty root" };
    }
    const resolved = path.resolve(rootPath);
    if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) {
      return { ok: false, error: `not a directory: ${resolved}` };
    }
    this.root = resolved;
    return { ok: true, root: this.root };
  }

  getRoot() {
    return this.root;
  }

  resolveSafe(rel) {
    if (!this.root) throw new Error("未打开项目文件夹（请到「项目」点「打开文件夹」）");
    const relNorm = String(rel || ".").replace(/\\/g, "/");
    if (relNorm.includes("\0")) throw new Error("非法路径");
    const abs = path.resolve(this.root, relNorm);
    const rootWithSep = this.root.endsWith(path.sep) ? this.root : this.root + path.sep;
    if (abs !== this.root && !abs.startsWith(rootWithSep)) {
      throw new Error(`路径越界: ${rel}`);
    }
    return abs;
  }

  async execute(action, { onLog } = {}) {
    const started = Date.now();
    const type = action?.type;
    const base = {
      action_id: action?.id,
      type,
      ok: false,
      exit_code: null,
      stdout: "",
      stderr: "",
      content: null,
      path: action?.path || null,
      elapsed_ms: 0,
      error: null,
      meta: {},
    };
    try {
      let result;
      switch (type) {
        case "write_file":
          result = await this.writeFile(action.path, action.content ?? "");
          break;
        case "apply_patch":
          result = await this.applyPatch(action.path, action.replacements || []);
          break;
        case "maven_compile":
          result = await this.runMaven(["-DskipTests", "compile"], { onLog });
          break;
        case "maven_install":
          result = await this.runMaven(["-DskipTests", "install"], { onLog });
          break;
        case "read_file":
          result = await this.readFile(action.path);
          break;
        case "sys_disk_overview":
        case "sys_scan_heavy":
        case "sys_cleanup":
          result = await executeSystemAction(action);
          break;
        case "list_dir":
          result = await this.listDir(action.path || ".");
          break;
        case "open_file":
          result = await this.openFile(action.path);
          break;
        case "start_app":
          result = await this.startApp({
            waitMs: Number(action.wait_ms) || 45000,
            onLog,
          });
          break;
        case "stop_app":
          result = await this.stopApp();
          break;
        case "http_smoke":
          result = await runHttpSmoke(action);
          break;
        case "open_browser":
          result = await this.openBrowser(action);
          break;
        case "run_sql":
          result = await this.connectorPost("/api/v1/connectors/mysql", {
            sql: action.sql || "",
            mysql: action.mysql || null,
          });
          break;
        case "run_cypher":
          result = await this.connectorPost("/api/v1/connectors/neo4j", {
            query: action.query || action.cypher || "",
            params: action.params || null,
          });
          break;
        case "milvus_search":
          result = await this.connectorPost("/api/v1/connectors/milvus", {
            op: "search",
            collection: action.collection || null,
            filter: action.filter || null,
            expr: action.expr || null,
            limit: action.limit != null ? Number(action.limit) : null,
          });
          break;
        case "milvus_mutate":
          result = await this.connectorPost("/api/v1/connectors/milvus", {
            op: action.op || "upsert",
            collection: action.collection || null,
            filter: action.filter || null,
            expr: action.expr || null,
            rows: action.rows || [],
            embeddings: action.embeddings || [],
            ids: action.ids || [],
          });
          break;
        case "run_shell":
        case "run_powershell":
          if (!this.root) throw new Error("未打开项目文件夹（请到「项目」点「打开文件夹」）");
          result = await runCommand({
            root: this.root,
            command: action.command,
            argv: action.argv,
            cwd: action.cwd || ".",
            timeoutMs: Number(action.timeout_ms) || 600000,
            shellKind: type === "run_powershell" ? "powershell" : action.shell || "powershell",
            onLog,
          });
          break;
        case "git_status":
        case "git_branch":
        case "git_checkout":
        case "git_pull":
        case "git_merge":
        case "git_commit":
        case "git_push":
          if (!this.root) throw new Error("未打开项目文件夹（请到「项目」点「打开文件夹」）");
          result = await executeGit(action, this.root, onLog);
          break;
        case "ask_user":
        case "done":
        case "fail":
          result = { ok: true, stdout: action.note || action.reply || type };
          break;
        default:
          throw new Error(`不支持的 action: ${type}`);
      }
      Object.assign(base, result, {
        ok: result.ok !== false && !result.error,
        type,
        action_id: action.id,
      });
      if (result.error) base.ok = false;
    } catch (err) {
      base.error = err.message || String(err);
      base.ok = false;
      base.stderr = base.error;
    }
    base.elapsed_ms = Date.now() - started;
    base.stdout = truncate(base.stdout, MAX_STDOUT);
    base.stderr = truncate(base.stderr, MAX_STDOUT);
    return base;
  }

  async readFile(rel) {
    const abs = this.resolveSafe(rel);
    if (!fs.existsSync(abs)) return { ok: false, error: "文件不存在", path: rel };
    const content = fs.readFileSync(abs, "utf8");
    return { ok: true, path: rel, content, stdout: `read ${rel}` };
  }

  async listDir(rel) {
    const abs = this.resolveSafe(rel || ".");
    if (!fs.existsSync(abs) || !fs.statSync(abs).isDirectory()) {
      return { ok: false, error: "目录不存在", path: rel };
    }
    const names = fs.readdirSync(abs).slice(0, 200);
    const lines = names.map((name) => {
      try {
        const st = fs.statSync(path.join(abs, name));
        return `${st.isDirectory() ? "d" : "f"}\t${name}`;
      } catch {
        return `?\t${name}`;
      }
    });
    return {
      ok: true,
      path: rel || ".",
      content: lines.join("\n"),
      stdout: `listed ${names.length} entries`,
    };
  }

  async writeFile(rel, content) {
    const abs = this.resolveSafe(rel);
    fs.mkdirSync(path.dirname(abs), { recursive: true });
    const body = String(content ?? "");
    fs.writeFileSync(abs, body, "utf8");
    let nLines = 0;
    if (body) {
      nLines = body.endsWith("\n") ? Math.max(1, (body.match(/\n/g) || []).length) : (body.match(/\n/g) || []).length + 1;
    }
    const changes = nLines ? [{ start_line: 1, end_line: nLines }] : [];
    const lineSummary = nLines
      ? `\`${rel}\` — 新增 第 1–${nLines} 行`
      : `\`${rel}\` — 新增空文件`;
    return {
      ok: true,
      path: rel,
      stdout: `wrote ${rel} lines=1-${nLines || 0}`,
      meta: { applied: 1, new_file: true, changes, line_summary: lineSummary },
    };
  }

  async applyPatch(rel, replacements) {
    const abs = this.resolveSafe(rel);
    if (!fs.existsSync(abs)) return { ok: false, error: "文件不存在", path: rel };
    let text = fs.readFileSync(abs, "utf8");
    let applied = 0;
    const changes = [];
    for (const rep of replacements || []) {
      const old = rep.old || rep.old_string || "";
      const neu = rep.new || rep.new_string || "";
      if (!old) continue;
      const idx = text.indexOf(old);
      if (idx < 0) continue;
      const startLine = text.slice(0, idx).split("\n").length;
      const oldEnd = startLine + (old.match(/\n/g) || []).length;
      const newEnd = startLine + (neu.match(/\n/g) || []).length;
      text = text.slice(0, idx) + neu + text.slice(idx + old.length);
      applied += 1;
      changes.push({
        start_line: startLine,
        end_line: Math.max(startLine, newEnd),
        old_start_line: startLine,
        old_end_line: Math.max(startLine, oldEnd),
      });
    }
    if (!applied) return { ok: false, error: "no replacement matched", path: rel };
    fs.writeFileSync(abs, text, "utf8");
    const spans = changes.map((c) =>
      c.end_line <= c.start_line ? `第 ${c.start_line} 行` : `第 ${c.start_line}–${c.end_line} 行`
    );
    const lineSummary = `\`${rel}\` — ${spans.join("；")}`;
    return {
      ok: true,
      path: rel,
      stdout: `patched ${rel} applied=${applied} ${lineSummary}`,
      meta: { applied, changes, line_summary: lineSummary },
    };
  }

  async openFile(rel) {
    const abs = this.resolveSafe(rel || ".");
    if (!fs.existsSync(abs)) return { ok: false, error: "路径不存在", path: rel };
    const err = await shell.openPath(abs);
    if (err) return { ok: false, error: err, path: rel };
    return { ok: true, path: rel, stdout: `opened ${rel}` };
  }

  async stopApp() {
    const child = this.appChild;
    if (!child || child.killed) {
      this.appChild = null;
      return { ok: true, stdout: "no running app", meta: { alive: false } };
    }
    try {
      if (process.platform === "win32" && child.pid) {
        spawn("taskkill", ["/pid", String(child.pid), "/t", "/f"], {
          windowsHide: true,
          shell: true,
        });
      } else {
        child.kill("SIGTERM");
      }
    } catch (err) {
      return { ok: false, error: err.message || String(err), meta: { alive: false } };
    }
    this.appChild = null;
    return {
      ok: true,
      stdout: truncate(this.appLog, MAX_STDOUT),
      meta: { alive: false },
    };
  }

  async startApp({ waitMs = 45000, onLog } = {}) {
    if (!this.root) throw new Error("未打开项目文件夹（请到「项目」点「打开文件夹」）");
    if (this.appChild && !this.appChild.killed) {
      await this.stopApp();
    }
    this.appLog = "";
    const command = this.mavenCmd();
    const args = ["-DskipTests", "spring-boot:run"];
    let child;
    try {
      child = spawn(command, args, {
        cwd: this.root,
        env: process.env,
        shell: process.platform === "win32",
        windowsHide: true,
      });
    } catch (err) {
      return { ok: false, error: err.message, exit_code: 1, meta: { alive: false } };
    }
    this.appChild = child;
    const append = (chunk, stream) => {
      this.appLog += chunk;
      if (this.appLog.length > MAX_STDOUT * 2) {
        this.appLog = this.appLog.slice(-MAX_STDOUT);
      }
      if (typeof onLog === "function") onLog(chunk, stream);
    };
    child.stdout?.on("data", (buf) => append(buf.toString("utf8"), "stdout"));
    child.stderr?.on("data", (buf) => append(buf.toString("utf8"), "stderr"));
    let exited = null;
    child.on("exit", (code) => {
      exited = code;
      if (this.appChild === child) this.appChild = null;
    });
    child.on("error", (err) => {
      append(`\n${err.message}\n`, "stderr");
      exited = 1;
      if (this.appChild === child) this.appChild = null;
    });

    const deadline = Date.now() + Math.max(5000, waitMs);
    const successRe = /Started \S+Application/;
    while (Date.now() < deadline) {
      if (successRe.test(this.appLog)) {
        return {
          ok: true,
          exit_code: null,
          stdout: truncate(this.appLog, MAX_STDOUT),
          stderr: "",
          meta: { alive: !child.killed && exited === null },
        };
      }
      if (exited !== null) {
        return {
          ok: false,
          exit_code: exited,
          stdout: truncate(this.appLog, MAX_STDOUT),
          stderr: "",
          error: `process exited ${exited}`,
          meta: { alive: false },
        };
      }
      await new Promise((r) => setTimeout(r, 400));
    }
    const alive = !child.killed && exited === null;
    return {
      ok: alive,
      exit_code: exited,
      stdout: truncate(this.appLog, MAX_STDOUT),
      stderr: alive ? "" : "start timeout without success pattern",
      error: alive ? null : "start timeout without success pattern",
      meta: { alive },
    };
  }

  mavenCmd() {
    const mvnw = process.platform === "win32" ? "mvnw.cmd" : "mvnw";
    const local = path.join(this.root, mvnw);
    if (fs.existsSync(local)) return local;
    return process.platform === "win32" ? "mvn.cmd" : "mvn";
  }

  runMaven(args, { onLog, timeoutMs = 900000 } = {}) {
    const command = this.mavenCmd();
    return new Promise((resolve) => {
      let stdout = "";
      let stderr = "";
      let settled = false;
      let child;
      try {
        child = spawn(command, args, {
          cwd: this.root,
          env: process.env,
          shell: process.platform === "win32",
          windowsHide: true,
        });
      } catch (err) {
        resolve({ ok: false, error: err.message, exit_code: 1 });
        return;
      }
      const timer = setTimeout(() => {
        try {
          child.kill();
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
        });
      });
      child.on("exit", (code) => {
        clearTimeout(timer);
        if (settled) return;
        settled = true;
        resolve({
          ok: code === 0,
          exit_code: code,
          stdout: truncate(stdout, MAX_STDOUT),
          stderr: truncate(stderr, MAX_STDOUT),
          error: code === 0 ? null : `exit ${code}`,
        });
      });
    });
  }

  /**
   * @param {object} action
   */
  async openBrowser(action) {
    const raw = String(action?.url || "").trim();
    if (!raw) return { ok: false, error: "empty url" };
    let u;
    try {
      u = new URL(raw);
    } catch {
      return { ok: false, error: `非法 URL: ${raw}` };
    }
    if (!["http:", "https:", "file:"].includes(u.protocol)) {
      return { ok: false, error: `仅允许 http(s)/file，拒绝: ${u.protocol}` };
    }
    const preferChrome = String(action?.browser || "default").toLowerCase() === "chrome";
    if (preferChrome) {
      for (const exe of CHROME_CANDIDATES) {
        try {
          if (!fs.existsSync(exe)) continue;
          const child = spawn(exe, [raw], { detached: true, stdio: "ignore" });
          child.unref();
          return {
            ok: true,
            stdout: `opened chrome: ${raw}`,
            meta: { browser: "chrome", path: exe },
          };
        } catch {
          /* try next */
        }
      }
    }
    try {
      await shell.openExternal(raw);
      return { ok: true, stdout: `opened default browser: ${raw}`, meta: { browser: "default" } };
    } catch (err) {
      return { ok: false, error: err.message || String(err) };
    }
  }

  /**
   * POST JSON to local API connectors.
   * @param {string} apiPath
   * @param {object} body
   */
  async connectorPost(apiPath, body) {
    const base = String(this.apiBase || DEFAULT_API_BASE).replace(/\/+$/, "");
    const urlStr = `${base}${apiPath.startsWith("/") ? apiPath : `/${apiPath}`}`;
    let u;
    try {
      u = new URL(urlStr);
    } catch {
      return { ok: false, error: `非法 API base: ${base}` };
    }
    const lib = u.protocol === "https:" ? https : http;
    const payload = JSON.stringify(body || {});
    const opts = {
      protocol: u.protocol,
      hostname: u.hostname,
      port: u.port || (u.protocol === "https:" ? 443 : 80),
      path: `${u.pathname}${u.search || ""}`,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(payload),
      },
      timeout: 60000,
    };
    return new Promise((resolve) => {
      const req = lib.request(opts, (res) => {
        const chunks = [];
        res.on("data", (c) => chunks.push(c));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let data = null;
          try {
            data = JSON.parse(text);
          } catch {
            data = null;
          }
          if (!data || typeof data !== "object") {
            resolve({
              ok: false,
              error: `connectors 非 JSON 响应 HTTP ${res.statusCode}`,
              stderr: truncate(text, MAX_STDOUT),
              stdout: "",
            });
            return;
          }
          const ok = Boolean(data.ok) && (res.statusCode || 0) < 400;
          resolve({
            ok,
            stdout: truncate(String(data.stdout || ""), MAX_STDOUT),
            stderr: truncate(String(data.stderr || data.error || ""), MAX_STDOUT),
            error: ok ? null : data.error || `HTTP ${res.statusCode}`,
            content: data.rows != null ? JSON.stringify(data.rows) : null,
            meta: {
              row_count: data.row_count,
              ...(data.meta || {}),
            },
          });
        });
      });
      req.on("timeout", () => {
        req.destroy();
        resolve({ ok: false, error: "connectors timeout 60s" });
      });
      req.on("error", (err) => {
        resolve({ ok: false, error: err.message || String(err) });
      });
      req.write(payload);
      req.end();
    });
  }
}

module.exports = { LocalExecutor };
