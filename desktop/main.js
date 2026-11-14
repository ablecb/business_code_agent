const { app, BrowserWindow, ipcMain, dialog, shell, Menu } = require("electron");
const path = require("path");
const fs = require("fs");
const { LocalExecutor } = require("./executor");

// 2026-08-16: 默认 userData（%APPDATA%/business-code-agent-desktop）在本机
// 写入被拒（platform_channel/network_sandbox 0x5），electron 子进程崩溃并拖垮
// 主进程。把 userData 落到项目本地目录，绕开该权限问题。必须在首次读取
// userData（下方 CONFIG_PATH）之前设置。
try {
  const localUserData = path.join(__dirname, "..", "data", "electron_userdata");
  fs.mkdirSync(localUserData, { recursive: true });
  app.setPath("userData", localUserData);
} catch {
  // 设置失败则退回默认 userData（保持原有行为）
}

const CONFIG_PATH = path.join(app.getPath("userData"), "config.json");
const DEFAULT_API_BASE = "http://127.0.0.1:8000";

const executor = new LocalExecutor();
/** @type {BrowserWindow | null} */
let mainWindow = null;

function normalizeTheme(theme) {
  if (theme === "light" || theme === "white") return "light";
  return "dark";
}

function themeBackground(theme) {
  return normalizeTheme(theme) === "light" ? "#f3f3f3" : "#1e1e1e";
}

function readConfig() {
  try {
    const raw = fs.readFileSync(CONFIG_PATH, "utf8");
    const cfg = {
      apiBase: DEFAULT_API_BASE,
      localProjectRoot: "",
      projectLocalRoots: {},
      theme: "dark",
      ...JSON.parse(raw),
    };
    cfg.theme = normalizeTheme(cfg.theme);
    if (!cfg.projectLocalRoots || typeof cfg.projectLocalRoots !== "object") {
      cfg.projectLocalRoots = {};
    }
    return cfg;
  } catch {
    return {
      apiBase: DEFAULT_API_BASE,
      localProjectRoot: "",
      projectLocalRoots: {},
      theme: "dark",
    };
  }
}

function writeConfig(partial) {
  const cur = readConfig();
  const next = { ...cur, ...partial };
  if (partial && Object.prototype.hasOwnProperty.call(partial, "projectLocalRoots")) {
    next.projectLocalRoots = { ...(partial.projectLocalRoots || {}) };
  }
  fs.mkdirSync(path.dirname(CONFIG_PATH), { recursive: true });
  fs.writeFileSync(CONFIG_PATH, JSON.stringify(next, null, 2), "utf8");
  return next;
}

function getProjectLocalRoot(projectId) {
  if (!projectId) return "";
  const roots = readConfig().projectLocalRoots || {};
  const p = roots[projectId];
  if (p && fs.existsSync(p) && fs.statSync(p).isDirectory()) return path.resolve(p);
  return "";
}

function setProjectLocalRoot(projectId, rootPath) {
  if (!projectId) return { ok: false, error: "empty project id" };
  const roots = { ...(readConfig().projectLocalRoots || {}) };
  if (!rootPath) {
    delete roots[projectId];
    writeConfig({ projectLocalRoots: roots });
    return { ok: true, path: "" };
  }
  const resolved = path.resolve(rootPath);
  if (!fs.existsSync(resolved) || !fs.statSync(resolved).isDirectory()) {
    return { ok: false, error: `not a directory: ${resolved}` };
  }
  roots[projectId] = resolved;
  writeConfig({ projectLocalRoots: roots });
  return { ok: true, path: resolved };
}

function installAppMenu() {
  // Standard Edit roles so Ctrl/Cmd+C/V/X/A work in the composer & selection.
  const template = [
    ...(process.platform === "darwin"
      ? [
          {
            label: app.name,
            submenu: [
              { role: "about" },
              { type: "separator" },
              { role: "services" },
              { type: "separator" },
              { role: "hide" },
              { role: "hideOthers" },
              { role: "unhide" },
              { type: "separator" },
              { role: "quit" },
            ],
          },
        ]
      : []),
    {
      label: "编辑",
      submenu: [
        { role: "undo", label: "撤销" },
        { role: "redo", label: "重做" },
        { type: "separator" },
        { role: "cut", label: "剪切" },
        { role: "copy", label: "复制" },
        { role: "paste", label: "粘贴" },
        { role: "delete", label: "删除" },
        { type: "separator" },
        { role: "selectAll", label: "全选" },
      ],
    },
    {
      label: "查看",
      submenu: [
        { role: "reload", label: "重新加载" },
        { role: "toggleDevTools", label: "开发者工具" },
        { type: "separator" },
        { role: "resetZoom", label: "实际大小" },
        { role: "zoomIn", label: "放大" },
        { role: "zoomOut", label: "缩小" },
        { type: "separator" },
        { role: "togglefullscreen", label: "全屏" },
      ],
    },
    {
      label: "窗口",
      submenu: [
        { role: "minimize", label: "最小化" },
        { role: "close", label: "关闭" },
      ],
    },
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}

function attachContextMenu(win) {
  win.webContents.on("context-menu", (_event, params) => {
    const items = [];
    if (params.isEditable) {
      items.push(
        { role: "undo", label: "撤销", enabled: params.editFlags.canUndo },
        { role: "redo", label: "重做", enabled: params.editFlags.canRedo },
        { type: "separator" },
        { role: "cut", label: "剪切", enabled: params.editFlags.canCut },
        { role: "copy", label: "复制", enabled: params.editFlags.canCopy },
        { role: "paste", label: "粘贴", enabled: params.editFlags.canPaste },
        { role: "delete", label: "删除", enabled: params.editFlags.canDelete },
        { type: "separator" },
        {
          role: "selectAll",
          label: "全选",
          enabled: params.editFlags.canSelectAll,
        }
      );
    } else if (params.selectionText && params.selectionText.trim()) {
      items.push({ role: "copy", label: "复制" });
      if (params.selectionText.trim().length > 0) {
        items.push({
          role: "selectAll",
          label: "全选",
          enabled: params.editFlags.canSelectAll,
        });
      }
    }
    if (!items.length) return;
    Menu.buildFromTemplate(items).popup({ window: win });
  });
}

function createWindow() {
  const cfg = readConfig();
  mainWindow = new BrowserWindow({
    width: 1380,
    height: 880,
    minWidth: 960,
    minHeight: 640,
    backgroundColor: themeBackground(cfg.theme),
    title: "Code Agent",
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      // 2026-08-16: renderer sandbox 在本机触发 platform_channel 拒绝访问(0x5)
      // 导致窗口崩溃退出；关闭 sandbox（contextIsolation+nodeIntegration=false
      // 仍保留 preload 隔离边界）。若环境恢复可改回 true。
      sandbox: false,
      spellcheck: false,
    },
  });

  try {
    mainWindow.webContents.session.setSpellCheckerEnabled(false);
  } catch {
    /* older Electron */
  }
  attachContextMenu(mainWindow);
  mainWindow.once("ready-to-show", () => {
    mainWindow.show();
    mainWindow.focus();
  });
  // Do not force webContents.focus() on every window focus — that steals
  // caret from the composer textarea when the user clicks to type.
  mainWindow.loadFile(path.join(__dirname, "renderer", "index.html"));
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  installAppMenu();
  createWindow();
  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("config:get", () => {
  const cfg = readConfig();
  return {
    ...cfg,
    projectLocalRoots: { ...(cfg.projectLocalRoots || {}) },
    localProjectRoot: executor.getRoot() || "",
  };
});

ipcMain.handle("config:set", (_e, partial) => {
  const patch = { ...(partial || {}) };
  if (patch.theme != null) patch.theme = normalizeTheme(patch.theme);
  // localProjectRoot no longer a global setting; ignore if sent
  delete patch.localProjectRoot;
  const next = writeConfig(patch);
  if (next.apiBase) executor.setApiBase(next.apiBase);
  if (patch.theme && mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.setBackgroundColor(themeBackground(patch.theme));
  }
  return {
    ...next,
    projectLocalRoots: { ...(next.projectLocalRoots || {}) },
    localProjectRoot: executor.getRoot() || "",
  };
});

ipcMain.handle("dialog:openProjectDir", async () => {
  const result = await dialog.showOpenDialog({
    title: "打开项目文件夹",
    properties: ["openDirectory"],
  });
  if (result.canceled || !result.filePaths?.length) return null;
  const dir = path.resolve(result.filePaths[0]);
  if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) {
    return { ok: false, error: `不是有效目录: ${dir}` };
  }
  return {
    ok: true,
    path: dir,
    name: path.basename(dir),
  };
});

ipcMain.handle("projectRoots:get", () => ({ ...(readConfig().projectLocalRoots || {}) }));
ipcMain.handle("projectRoots:getOne", (_e, projectId) => getProjectLocalRoot(projectId));
ipcMain.handle("projectRoots:set", (_e, projectId, rootPath) =>
  setProjectLocalRoot(projectId, rootPath)
);

ipcMain.handle("executor:getRoot", () => executor.getRoot());
ipcMain.handle("executor:setRoot", (_e, rootPath) => executor.setRoot(rootPath || ""));
ipcMain.handle("executor:useProject", (_e, projectId) => {
  const root = getProjectLocalRoot(projectId);
  if (!root) {
    return { ok: false, error: "该项目未关联本机目录，请重新「打开文件夹」" };
  }
  return executor.setRoot(root);
});

ipcMain.handle("executor:run", async (_e, action) => {
  executor.setApiBase(readConfig().apiBase || DEFAULT_API_BASE);
  return executor.execute(action || {}, {
    onLog: (chunk, stream) => {
      if (mainWindow && !mainWindow.isDestroyed()) {
        mainWindow.webContents.send("executor:log", {
          chunk: String(chunk || ""),
          stream: stream || "stdout",
          action_id: action?.id || null,
        });
      }
    },
  });
});

/** Reveal a server-side chat log in Explorer (same machine as API). */
ipcMain.handle("shell:showItemInFolder", (_e, targetPath) => {
  const p = String(targetPath || "").trim();
  if (!p) return { ok: false, error: "empty path" };
  try {
    if (fs.existsSync(p)) {
      shell.showItemInFolder(path.resolve(p));
      return { ok: true };
    }
    const dir = path.dirname(path.resolve(p));
    if (fs.existsSync(dir)) {
      shell.openPath(dir);
      return { ok: true, missing: true };
    }
    return { ok: false, error: `not found: ${p}` };
  } catch (err) {
    return { ok: false, error: String(err?.message || err) };
  }
});
