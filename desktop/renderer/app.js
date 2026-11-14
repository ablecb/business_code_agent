(() => {
  const STORAGE_KEY = "bca_desktop_sessions_v1";
  const ACTIVE_KEY = "bca_desktop_active_session_v1";
  const ACTIVE_PROJECT_KEY = "bca_desktop_active_project_v1";
  const ACTIVE_PROJECT_IDS_KEY = "bca_desktop_active_project_ids_v1";
  const THEME_KEY = "bca_desktop_theme_v1";

  const statusText = {
    ready: "待解析",
    parsing: "解析中",
    parsed: "已解析",
    failed: "失败",
    empty: "无 Java 源码",
  };
  const stageText = {
    queued: "排队中",
    prepare: "准备存储",
    overview: "项目总览",
    entries: "数据源头",
    flow: "递归梳流程",
    entity: "业务表(流程中)",
    status: "实体状态流转",
    parse: "解析代码",
    graph: "构建调用图",
    tagging: "打标向量化",
    global_ast: "Tree-sitter 全局 AST",
    persist: "写入符号结构",
    topo_ai: "拓扑 LLM 分析",
    milvus: "写入 Milvus",
    neo4j_calls: "写入 Neo4j CALLS",
    entry_route: "入口路由",
    done: "完成",
    cancelled: "已取消",
  };

  /** Recent parse progress lines per project (for sidebar log). */
  const parseLogs = Object.create(null);

  const els = {
    sidebar: document.getElementById("sidebar"),
    sessionList: document.getElementById("sessionList"),
    projectList: document.getElementById("projectList"),
    activeProjectChip: document.getElementById("activeProjectChip"),
    transcript: document.getElementById("transcript"),
    emptyState: document.getElementById("emptyState"),
    form: document.getElementById("chatForm"),
    input: document.getElementById("messageInput"),
    send: document.getElementById("btnSend"),
    chatTitle: document.getElementById("chatTitle"),
    healthStatus: document.getElementById("healthStatus"),
    healthDot: document.getElementById("healthDot"),
    btnNewChat: document.getElementById("btnNewChat"),
    btnRefreshProjects: document.getElementById("btnRefreshProjects"),
    uploadName: document.getElementById("uploadName"),
    uploadId: document.getElementById("uploadId"),
    uploadMsg: document.getElementById("uploadMsg"),
    settingsModal: document.getElementById("settingsModal"),
    settingsBackdrop: document.getElementById("settingsBackdrop"),
    settingsDialog: document.querySelector(".settings-dialog"),
    settingsForm: document.getElementById("settingsForm"),
    settingsApiBase: document.getElementById("settingsApiBase"),
    settingsMsg: document.getElementById("settingsMsg"),
    btnSettings: document.getElementById("btnSettings"),
    btnCloseSettings: document.getElementById("btnCloseSettings"),
    btnCancelSettings: document.getElementById("btnCancelSettings"),
    btnSaveSettings: document.getElementById("btnSaveSettings"),
    btnToggleTheme: document.getElementById("btnToggleTheme"),
    themePickDark: document.getElementById("themePickDark"),
    themePickLight: document.getElementById("themePickLight"),
    workbench: document.querySelector(".workbench"),
    previewDock: document.getElementById("previewDock"),
    previewPane: document.getElementById("previewPane"),
    previewResizer: document.getElementById("previewResizer"),
    previewFilebar: document.getElementById("previewFilebar"),
    previewSplits: document.getElementById("previewSplits"),
    previewTitle: document.getElementById("previewTitle"),
    btnClosePreview: document.getElementById("btnClosePreview"),
    btnSplitPreview: document.getElementById("btnSplitPreview"),
    btnPreviewUnified: document.getElementById("btnPreviewUnified"),
    btnPreviewCompare: document.getElementById("btnPreviewCompare"),
    patchStatus: document.getElementById("patchStatus"),
    btnStopPatch: document.getElementById("btnStopPatch"),
    btnPickDir: document.getElementById("btnPickDir"),
    kbList: document.getElementById("kbList"),
    kbMsg: document.getElementById("kbMsg"),
    btnUploadKb: document.getElementById("btnUploadKb"),
    btnRefreshKb: document.getElementById("btnRefreshKb"),
    kbFileInput: document.getElementById("kbFileInput"),
  };

  /** @type {Array<{id:string,title:string,filename:string,status:string,chunk_count?:number,error?:string}>} */
  let kbDocs = [];

  const PREVIEW_W_KEY = "bca_desktop_preview_w_v1";
  const PREVIEW_VIEW_KEY = "bca_desktop_preview_view_v1";
  /** @type {{path:string, body:string, kind?:string, diffHtml?:string, compareHtml?:string}[]} */
  let previewFiles = [];
  /** @type {{id:string, fileIndex:number}[]} */
  let previewWindows = [];
  let previewWinSeq = 0;
  let previewLayout = "col"; // col | row
  let previewViewMode =
    localStorage.getItem(PREVIEW_VIEW_KEY) === "compare" ? "compare" : "unified";

  /** @type {{id:string|null,title:string,updatedAt:number,messages:Array}[]} */
  let sessions = loadSessions();
  let activeId = localStorage.getItem(ACTIVE_KEY) || null;
  let projects = [];
  /** @type {Record<string, string>} */
  let projectLocalRoots = {};
  let sending = false;
  let pollTimer = null;
  let thinkingTimer = null;
  let currentView = "chat";
  let settingsOpen = false;

  function escapeHtml(str) {
    return String(str ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function loadSessions() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      return (Array.isArray(raw) ? raw : []).map((s) => ({
        ...s,
        messages: (s.messages || []).filter((m) => !m.pending),
      }));
    } catch {
      return [];
    }
  }

  function saveSessions() {
    const snapshot = sessions.map((s) => ({
      ...s,
      messages: (s.messages || [])
        .filter((m) => !m.pending)
        .map((m) => {
          const { thinkingStartedAt, pending, ...rest } = m;
          return rest;
        }),
    }));
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot));
  }

  function activeProjectIds() {
    try {
      const raw = JSON.parse(localStorage.getItem(ACTIVE_PROJECT_IDS_KEY) || "[]");
      if (Array.isArray(raw) && raw.length) return raw.map(String);
    } catch {
      /* ignore */
    }
    const one = localStorage.getItem(ACTIVE_PROJECT_KEY);
    return one ? [one] : [];
  }

  function activeProjectId() {
    const ids = activeProjectIds();
    return ids[0] || null;
  }

  function setActiveProjects(ids, nameById) {
    const list = [...new Set((ids || []).map(String).filter(Boolean))];
    localStorage.setItem(ACTIVE_PROJECT_IDS_KEY, JSON.stringify(list));
    if (list[0]) localStorage.setItem(ACTIVE_PROJECT_KEY, list[0]);
    else localStorage.removeItem(ACTIVE_PROJECT_KEY);
    let label = "未选择项目";
    if (list.length === 1) {
      label = (nameById && nameById[list[0]]) || list[0];
    } else if (list.length > 1) {
      label = `${list.length} 个项目`;
    }
    els.activeProjectChip.textContent = label;
    els.activeProjectChip.classList.toggle("has-project", list.length > 0);
    els.activeProjectChip.title = list.join(", ");
  }

  function setActiveProject(id, name) {
    if (!id) {
      setActiveProjects([]);
      return;
    }
    const ids = activeProjectIds();
    if (!ids.includes(id)) ids.push(id);
    const map = {};
    if (name) map[id] = name;
    for (const p of projects) map[p.id] = p.name;
    setActiveProjects(ids, map);
  }

  function toggleProjectSelected(id, name, on) {
    let ids = activeProjectIds();
    if (on) {
      if (!ids.includes(id)) ids.push(id);
    } else {
      ids = ids.filter((x) => x !== id);
    }
    const map = { [id]: name };
    for (const p of projects) map[p.id] = p.name;
    setActiveProjects(ids, map);
  }

  function getActive() {
    return sessions.find((s) => s.id === activeId) || null;
  }

  /** Same shape as LocalChatTester session_id: desktop-chat-<hex>. */
  function newDesktopSessionId() {
    const hex =
      (typeof crypto !== "undefined" && crypto.randomUUID
        ? crypto.randomUUID().replace(/-/g, "")
        : `${Date.now().toString(16)}${Math.random().toString(16).slice(2)}`
      ).slice(0, 12);
    return `desktop-chat-${hex}`;
  }

  function ensureActive() {
    let s = getActive();
    if (s) {
      // Bind a stable id before first ask so server log file can append by session.
      if (!s.id) {
        s.id = newDesktopSessionId();
        activeId = s.id;
        localStorage.setItem(ACTIVE_KEY, activeId);
        saveSessions();
      }
      return s;
    }
    s = {
      id: newDesktopSessionId(),
      title: "新对话",
      updatedAt: Date.now(),
      messages: [],
    };
    sessions.unshift(s);
    activeId = s.id;
    localStorage.setItem(ACTIVE_KEY, activeId);
    saveSessions();
    return s;
  }

  function formatElapsed(ms) {
    const s = Math.max(0, ms / 1000);
    return s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;
  }

  const THINK_STEP_LABELS = {
    local_start: "已收到请求",
    intent: "识别意图",
    session_gate: "判断是否延续上一问",
    goal: "分析问题目标",
    plan: "规划检索步骤",
    exec: "检索代码证据",
    verify: "核对回答",
    locate: "定位要看的代码",
    fetch: "读取文件/方法",
  };

  function isNoiseThinkItem(item) {
    if (!item) return true;
    const sid = String(item.step_id || "");
    const note = String(item.note || item.title || "");
    if (item.type === "log" || sid === "local_start") return true;
    if (/对话日志|local_chat_interactive|heartbeat|正在处理…/.test(note)) return true;
    if (/本机路由/.test(note)) return true;
    if (!note.trim() && !THINK_STEP_LABELS[sid]) return true;
    return false;
  }

  function humanizeThinkNote(item) {
    const sid = String(item?.step_id || "");
    let note = String(item?.note || item?.title || "").trim();
    note = note.replace(/\[PEV\](\[[^\]]+\])*/g, "").trim();
    if (/^会话关联/.test(note)) {
      return /use_lock=True|dig_in=True|related=True/.test(note)
        ? "延续上一问，深入当前入口"
        : "作为新问题处理";
    }
    if (/^意图识别/.test(note)) {
      const m = note.match(/意图识别\s*→\s*(\w+)/);
      return m ? `意图：${m[1]}` : "已识别意图";
    }
    if (THINK_STEP_LABELS[sid] && (!note || note === sid || note.length < 3)) {
      return THINK_STEP_LABELS[sid] + (item?.status === "running" ? "…" : "");
    }
    return note || THINK_STEP_LABELS[sid] || sid || "…";
  }

  function usefulThinkTrace(trace) {
    return (Array.isArray(trace) ? trace : []).filter((x) => !isNoiseThinkItem(x));
  }

  /* ── Thinking Pulse Animation ── */

  function createPulseDots() {
    const wrap = document.createElement("span");
    wrap.className = "thinking-pulse";
    wrap.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span>`;
    return wrap;
  }

  function stopThinkingTimer() {
    if (thinkingTimer != null) {
      clearInterval(thinkingTimer);
      thinkingTimer = null;
    }
  }

  function startThinkingTimer(startedAt) {
    stopThinkingTimer();
    const tick = () => {
      const nodes = els.transcript.querySelectorAll(".msg.assistant .timing.thinking");
      if (!nodes.length) return;
      const elapsed = performance.now() - startedAt;
      const label = `思考中 · ${formatElapsed(elapsed)}`;
      nodes.forEach((el) => {
        // Keep pulse dots if they exist, update text
        const pulse = el.querySelector(".thinking-pulse");
        if (pulse) {
          el.textContent = "";
          el.appendChild(pulse);
          const textSpan = document.createElement("span");
          textSpan.style.marginLeft = "6px";
          textSpan.textContent = label;
          el.appendChild(textSpan);
        } else {
          el.textContent = label;
        }
      });
    };
    tick();
    thinkingTimer = setInterval(tick, 200);
  }

  function paintSettingsMsg(text, kind) {
    if (!els.settingsMsg) return;
    if (!text) {
      els.settingsMsg.hidden = true;
      els.settingsMsg.textContent = "";
      els.settingsMsg.classList.remove("is-error");
      return;
    }
    els.settingsMsg.hidden = false;
    els.settingsMsg.textContent = text;
    els.settingsMsg.classList.toggle("is-error", kind === "error");
  }

  async function fillSettingsForm() {
    paintSettingsMsg("");
    paintThemeButtons(currentTheme());
    let apiBase = window.BcaApi?.base || "http://127.0.0.1:8000";
    try {
      if (window.desktop?.getConfig) {
        const cfg = await window.desktop.getConfig();
        if (cfg?.apiBase) apiBase = cfg.apiBase;
        if (cfg?.theme) paintThemeButtons(cfg.theme);
      } else {
        apiBase = localStorage.getItem("bca_desktop_api_base") || apiBase;
      }
    } catch {
      /* keep current */
    }
    if (els.settingsApiBase) els.settingsApiBase.value = apiBase;
  }

  async function openSettings() {
    settingsOpen = true;
    await fillSettingsForm();
    if (els.settingsModal) els.settingsModal.hidden = false;
    if (els.btnSettings) els.btnSettings.setAttribute("aria-expanded", "true");
    document.body.classList.add("settings-open");
    queueMicrotask(() => els.settingsApiBase?.focus());
  }

  function closeSettings() {
    settingsOpen = false;
    if (els.settingsModal) els.settingsModal.hidden = true;
    if (els.btnSettings) els.btnSettings.setAttribute("aria-expanded", "false");
    document.body.classList.remove("settings-open");
    paintSettingsMsg("");
    if (currentView === "settings") currentView = "chat";
  }

  function toggleSettings() {
    if (settingsOpen) closeSettings();
    else openSettings();
  }

  async function saveSettings(e) {
    e?.preventDefault?.();
    const raw = String(els.settingsApiBase?.value || "").trim().replace(/\/+$/, "");
    if (!raw) {
      paintSettingsMsg("请填写后端服务地址", "error");
      return;
    }
    let url;
    try {
      url = new URL(raw);
    } catch {
      paintSettingsMsg("地址格式无效，请使用 http(s)://…", "error");
      return;
    }
    if (url.protocol !== "http:" && url.protocol !== "https:") {
      paintSettingsMsg("仅支持 http 或 https", "error");
      return;
    }
    const apiBase = url.origin + (url.pathname === "/" ? "" : url.pathname.replace(/\/+$/, ""));
    try {
      window.BcaApi.setBase(apiBase);
      localStorage.setItem("bca_desktop_api_base", apiBase);
      if (window.desktop?.setConfig) {
        await window.desktop.setConfig({ apiBase, theme: currentTheme() });
      }
      paintSettingsMsg("已保存");
      checkHealth();
      setTimeout(() => {
        if (settingsOpen) closeSettings();
      }, 400);
    } catch (err) {
      paintSettingsMsg(err?.message || "保存失败", "error");
    }
  }

  function setView(view) {
    currentView = view;
    if (view === "settings") {
      openSettings();
      return;
    }
    if (view === "projects") {
      closeSettings();
      loadProjects();
      els.sidebar?.scrollTo?.({ top: 0, behavior: "smooth" });
      return;
    }
    closeSettings();
    els.chatTitle.textContent = getActive()?.title || "新对话";
  }

  function formatAgo(iso) {
    if (!iso) return "";
    const t = Date.parse(iso);
    if (!Number.isFinite(t)) return "";
    const mins = Math.max(0, Math.round((Date.now() - t) / 60000));
    if (mins < 1) return "刚刚";
    if (mins < 60) return `${mins}m`;
    const hrs = Math.round(mins / 60);
    if (hrs < 48) return `${hrs}h`;
    return `${Math.round(hrs / 24)}d`;
  }

  /** Format job wall-clock seconds as 12分3秒 / 45秒 / 1小时2分. */
  function formatDurationZh(sec) {
    if (sec == null || !Number.isFinite(Number(sec)) || Number(sec) < 0) return "";
    const s = Math.max(0, Math.round(Number(sec)));
    if (s < 60) return `${s}秒`;
    const hours = Math.floor(s / 3600);
    const minutes = Math.floor((s % 3600) / 60);
    const secs = s % 60;
    if (hours) {
      let out = `${hours}小时`;
      if (minutes) out += `${minutes}分`;
      if (secs && minutes) out += `${secs}秒`;
      else if (secs && !minutes) out += `${secs}秒`;
      return out;
    }
    return secs ? `${minutes}分${secs}秒` : `${minutes}分`;
  }

  async function removeSession(target) {
    if (!target) return;
    const idx = sessions.findIndex((s) => s === target || (target.id && s.id === target.id));
    if (idx < 0) return;
    const wasActive =
      sessions[idx].id === activeId || (!sessions[idx].id && !activeId && sessions[idx] === getActive());
    const serverId = sessions[idx].id;
    sessions.splice(idx, 1);
    if (wasActive) {
      activeId = sessions[0]?.id ?? null;
      if (activeId) localStorage.setItem(ACTIVE_KEY, activeId);
      else localStorage.removeItem(ACTIVE_KEY);
    }
    saveSessions();
    renderAll();
    if (serverId) {
      try {
        await window.BcaApi.deleteSession(serverId);
      } catch {
        /* local list already cleared */
      }
    }
  }

  function renderSessions() {
    if (!sessions.length) {
      els.sessionList.innerHTML = `<div class="empty-list">暂无会话</div>`;
      return;
    }
    const sorted = [...sessions].sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    els.sessionList.replaceChildren();
    for (const s of sorted) {
      const active =
        (s.id && s.id === activeId) || (!s.id && !activeId && s === getActive());
      const meta = new Date(s.updatedAt || Date.now()).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      });
      const row = document.createElement("div");
      row.className = "side-item" + (active ? " active" : "");

      const main = document.createElement("button");
      main.type = "button";
      main.className = "side-item-main";
      main.innerHTML = `<span class="t">${escapeHtml(s.title || "新对话")}</span><span class="m">${escapeHtml(
        meta
      )}</span>`;
      main.addEventListener("click", () => {
        activeId = s.id || null;
        if (s.id) localStorage.setItem(ACTIVE_KEY, s.id);
        else localStorage.removeItem(ACTIVE_KEY);
        if (!s.id) {
          sessions = [s, ...sessions.filter((x) => x !== s)];
        }
        setView("chat");
        renderAll();
        els.input.focus();
      });

      const del = document.createElement("button");
      del.type = "button";
      del.className = "side-item-del";
      del.title = "删除";
      del.setAttribute("aria-label", "删除会话");
      del.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
        '<path d="M9 4h6m-8 4h10m-1.5 0-.7 11.2a1.5 1.5 0 0 1-1.5 1.4H9.7a1.5 1.5 0 0 1-1.5-1.4L7.5 8M10 11v6m4-6v6" ' +
        'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        const label = (s.title || "新对话").slice(0, 24);
        if (!confirm(`删除会话「${label}」？`)) return;
        void removeSession(s);
      });

      row.appendChild(main);
      row.appendChild(del);
      els.sessionList.appendChild(row);
    }
  }

  function messageVote(m) {
    if (m?.vote === "up" || m?.vote === "down") return m.vote;
    const v = m?.meta?.vote;
    return v === "up" || v === "down" ? v : null;
  }

  function thumbSvg(kind) {
    const path =
      '<path d="M7 11v9a1 1 0 0 1-1 1H4.5A1.5 1.5 0 0 1 3 19.5v-7A1.5 1.5 0 0 1 4.5 11H7zm0 0' +
      "c0-.8.2-1.6.6-2.3L11.2 3a1.8 1.8 0 0 1 3.3 1.1V8h4.2a2 2 0 0 1 2 2.3l-1.2 7" +
      'A2.5 2.5 0 0 1 17 19.5h-6.2A2.8 2.8 0 0 1 8 17.2V11z" ' +
      'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>';
    const inner = kind === "down" ? `<g transform="rotate(180 12 12)">${path}</g>` : path;
    return `<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">${inner}</svg>`;
  }

  function paintFeedbackBar(barEl, vote, busy) {
    if (!barEl) return;
    barEl.querySelectorAll(".feedback-btn").forEach((btn) => {
      const kind = btn.classList.contains("is-down") ? "down" : "up";
      const on = vote === kind;
      btn.classList.toggle("is-selected", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      btn.disabled = !!busy;
    });
  }

  async function submitFeedback(m, vote, barEl) {
    if (!m || m._feedbackBusy) return;
    const prev = messageVote(m);
    if (prev === vote) return;
    const session = getActive();
    const sessionId = session?.id;
    const messageId = m.id ?? m.assistant_message_id ?? m.meta?.assistant_message_id;
    if (!sessionId || messageId == null) {
      m.vote = vote;
      m.meta = { ...(m.meta || {}), vote };
      saveSessions();
      paintFeedbackBar(barEl, vote);
      return;
    }
    m._feedbackBusy = true;
    paintFeedbackBar(barEl, vote, true);
    try {
      await window.BcaApi.feedback({
        session_id: sessionId,
        assistant_message_id: Number(messageId),
        vote,
        project_id: activeProjectIds()[0] || undefined,
      });
      m.vote = vote;
      m.meta = { ...(m.meta || {}), vote };
      saveSessions();
      paintFeedbackBar(barEl, vote);
    } catch {
      paintFeedbackBar(barEl, prev);
    } finally {
      m._feedbackBusy = false;
    }
  }

  function messageCopyText(m) {
    const parts = [String(m?.content || m?.error || "")];
    if (m?.meta?.run_reply) parts.push(String(m.meta.run_reply));
    if (usefulList(m?.meta?.tool_timeline)) {
      for (const t of m.meta.tool_timeline) {
        const title =
          t.action?.ui_title || t.action?.type || t.content || t.tool_event || "";
        const body = t.preview || t.detail || t.logs || "";
        parts.push(`[${title}]\n${body}`.trim());
      }
    }
    return parts
      .join("\n\n")
      .replace(/\r\n/g, "\n")
      .trim();
  }

  async function copyTextToClipboard(text) {
    const value = String(text || "");
    if (!value) return false;
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
        return true;
      }
    } catch {
      /* fall through */
    }
    try {
      const ta = document.createElement("textarea");
      ta.value = value;
      ta.setAttribute("readonly", "");
      ta.style.position = "fixed";
      ta.style.left = "-9999px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      ta.remove();
      return ok;
    } catch {
      return false;
    }
  }

  function renderCopyMessageBtn(m) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "btn-mini msg-copy-btn";
    btn.textContent = "复制";
    btn.title = "复制消息内容";
    btn.setAttribute("aria-label", "复制消息内容");
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const text = messageCopyText(m);
      if (!text) return;
      const ok = await copyTextToClipboard(text);
      const prev = btn.textContent;
      btn.textContent = ok ? "已复制" : "复制失败";
      setTimeout(() => {
        btn.textContent = prev;
      }, 1200);
    });
    return btn;
  }

  function renderMsgActions(m) {
    const bar = document.createElement("div");
    bar.className = "msg-actions";
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", "消息操作");
    bar.appendChild(renderCopyMessageBtn(m));
    return bar;
  }

  function renderFeedbackBar(m) {
    const bar = document.createElement("div");
    bar.className = "feedback-bar";
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", "评价回复");
    bar.appendChild(renderCopyMessageBtn(m));
    const current = messageVote(m);
    const mkBtn = (vote, label) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className =
        "feedback-btn" +
        (vote === "down" ? " is-down" : "") +
        (current === vote ? " is-selected" : "");
      btn.title = label;
      btn.setAttribute("aria-label", label);
      btn.setAttribute("aria-pressed", current === vote ? "true" : "false");
      btn.innerHTML = thumbSvg(vote);
      btn.addEventListener("click", () => void submitFeedback(m, vote, bar));
      return btn;
    };
    bar.appendChild(mkBtn("up", "有用"));
    bar.appendChild(mkBtn("down", "不佳"));
    return bar;
  }

  function progressPct(job) {
    if (!job || !job.total_files) return 0;
    return Math.min(100, Math.round((job.processed / job.total_files) * 100));
  }

  function stageLabel(stage) {
    const key = String(stage || "").trim();
    if (!key) return "";
    return stageText[key] || key;
  }

  function pushParseLog(projectId, job) {
    if (!projectId || !job) return;
    const stage = stageLabel(job.stage);
    const msg = String(job.message || "").trim();
    const counts =
      job.total_files > 0
        ? `${job.processed || 0}/${job.total_files}`
        : "";
    const parts = [stage, counts, msg].filter(Boolean);
    if (!parts.length) return;
    const line = parts.join(" · ");
    const arr = parseLogs[projectId] || (parseLogs[projectId] = []);
    if (arr[arr.length - 1] === line) return;
    arr.push(line);
    while (arr.length > 14) arr.shift();
  }

  function clearParseLog(projectId) {
    if (projectId) delete parseLogs[projectId];
  }

  function renderParseDetail(p, job, busy) {
    if (!busy || !job) return "";
    pushParseLog(p.id, job);
    const stage = stageLabel(job.stage);
    const msg = String(job.message || "").trim();
    const counts =
      job.total_files > 0
        ? `${job.processed || 0} / ${job.total_files}`
        : "";
    const head = [stage, counts].filter(Boolean).join(" · ");
    const log = (parseLogs[p.id] || [])
      .slice(-8)
      .map((line) => `<div class="parse-log-line">${escapeHtml(line)}</div>`)
      .join("");
    return `<div class="parse-detail">
      ${head ? `<div class="parse-stage">${escapeHtml(head)}</div>` : ""}
      ${msg ? `<div class="msg parse-msg" title="${escapeHtml(msg)}">${escapeHtml(msg)}</div>` : ""}
      ${log ? `<div class="parse-log" aria-label="解析进度日志">${log}</div>` : ""}
    </div>`;
  }

  async function refreshProjectLocalRoots() {
    if (!window.desktop?.getProjectLocalRoots) {
      projectLocalRoots = {};
      return;
    }
    try {
      projectLocalRoots = (await window.desktop.getProjectLocalRoots()) || {};
    } catch {
      projectLocalRoots = {};
    }
  }

  function renderProjects() {
    if (!projects.length) {
      els.projectList.innerHTML = `<div class="empty-list">暂无项目，请打开文件夹</div>`;
      return;
    }
    const selected = new Set(activeProjectIds());
    els.projectList.innerHTML = projects
      .map((p) => {
        const st = p.status || "ready";
        const job = p.latest_job;
        const pct = progressPct(job);
        const busy = st === "parsing";
        const on = selected.has(p.id);
        const ago = formatAgo(p.updated_at) || statusText[st] || st;
        const elapsedLabel =
          !busy &&
          (st === "parsed" || st === "failed") &&
          job?.elapsed_sec != null
            ? `耗时 ${formatDurationZh(job.elapsed_sec)}`
            : "";
        const err = job?.error
          ? `<div class="err">${escapeHtml(String(job.error).slice(0, 160))}</div>`
          : "";
        return `<article class="proj-item${on ? " active" : ""}" data-id="${escapeHtml(p.id)}">
          <button type="button" class="proj-row${on ? " active" : ""}" data-pick="${escapeHtml(p.id)}" data-name="${escapeHtml(p.name)}">
            <span class="folder-ico" aria-hidden="true">
              <svg viewBox="0 0 24 24" width="16" height="16"><path fill="currentColor" d="M3 6a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V6z"/></svg>
            </span>
            <span class="name" title="${escapeHtml(p.root_path || p.name)}">${escapeHtml(p.name)}</span>
            <span class="ago">${escapeHtml(busy ? `${pct}%` : ago)}</span>
          </button>
          <label class="proj-check">
            <input type="checkbox" class="proj-toggle" data-id="${escapeHtml(
              p.id
            )}" data-name="${escapeHtml(p.name)}" ${on ? "checked" : ""} />
            <span>纳入对话</span>
            <span class="status-chip status-${escapeHtml(st)}">${statusText[st] || st}</span>
          </label>
          ${
            busy
              ? `<div class="progress"><i style="width:${pct}%"></i></div>${renderParseDetail(
                  p,
                  job,
                  busy
                )}`
              : elapsedLabel
                ? `<div class="msg parse-msg parse-elapsed">${escapeHtml(elapsedLabel)}</div>`
                : st === "failed" && job?.message
                  ? `<div class="msg parse-msg">${escapeHtml(
                      String(job.message).slice(0, 240)
                    )}</div>`
                  : ""
          }
          ${err}
          <div class="proj-actions">
            <button type="button" class="btn-mini btn-parse" data-id="${escapeHtml(
              p.id
            )}" data-force="${busy ? "1" : "0"}">${
          busy ? "强制重解析" : st === "parsed" ? "重新解析" : "解析"
        }</button>
          </div>
        </article>`;
      })
      .join("");

    // Drop stale logs for projects that finished parsing.
    for (const id of Object.keys(parseLogs)) {
      const row = projects.find((x) => x.id === id);
      if (!row || row.status !== "parsing") clearParseLog(id);
    }

    els.projectList.querySelectorAll(".proj-toggle").forEach((box) => {
      box.addEventListener("change", () => {
        toggleProjectSelected(
          box.getAttribute("data-id"),
          box.getAttribute("data-name"),
          box.checked
        );
        renderProjects();
      });
    });
    els.projectList.querySelectorAll("[data-pick]").forEach((btn) => {
      btn.addEventListener("click", () => {
        setActiveProject(btn.getAttribute("data-pick"), btn.getAttribute("data-name"));
        renderProjects();
        closeSettings();
        els.input.focus();
      });
    });
    els.projectList.querySelectorAll(".btn-parse").forEach((btn) => {
      btn.addEventListener("click", () =>
        parseProject(btn.getAttribute("data-id"), btn.getAttribute("data-force") === "1")
      );
    });

    // Keep parse log scrolled to latest line.
    els.projectList.querySelectorAll(".parse-log").forEach((el) => {
      el.scrollTop = el.scrollHeight;
    });

    const anyParsing = projects.some((p) => p.status === "parsing");
    if (anyParsing) startPoll();
    else stopPoll();
  }

  function looksLikeLocalAbsPath(p) {
    return /^[A-Za-z]:[\\/]/.test(String(p || "")) || String(p || "").startsWith("\\\\");
  }

  async function loadProjects() {
    try {
      projects = await window.BcaApi.listProjects();
      await refreshProjectLocalRoots();
      // Same-machine open-folder projects: sync server root_path → local binding.
      for (const p of projects) {
        if (projectLocalRoots[p.id] || !looksLikeLocalAbsPath(p.root_path)) continue;
        projectLocalRoots[p.id] = p.root_path;
        if (window.desktop?.setProjectLocalRoot) {
          try {
            await window.desktop.setProjectLocalRoot(p.id, p.root_path);
          } catch {
            /* ignore */
          }
        }
      }
      const map = {};
      for (const p of projects) map[p.id] = p.name;
      setActiveProjects(activeProjectIds(), map);
      renderProjects();
    } catch (err) {
      els.projectList.innerHTML = `<div class="empty-list">加载失败：${escapeHtml(
        err.message || err
      )}</div>`;
    }
  }

  async function resolveImportLocalRoot(pids) {
    const ids = pids && pids.length ? pids : activeProjectIds();
    for (const id of ids) {
      const root =
        projectLocalRoots[id] ||
        (window.desktop?.getProjectLocalRoot
          ? await window.desktop.getProjectLocalRoot(id)
          : "");
      if (root) return { projectId: id, root };
    }
    return null;
  }

  function startPoll() {
    if (pollTimer) return;
    // Faster while parsing so stage/message log stays fresh.
    pollTimer = setInterval(loadProjects, 1200);
  }

  function stopPoll() {
    if (!pollTimer) return;
    clearInterval(pollTimer);
    pollTimer = null;
  }

  async function parseProject(projectId, force = false) {
    try {
      if (force && !confirm("将取消当前卡住的任务并重新解析，是否继续？")) return;
      clearParseLog(projectId);
      await window.BcaApi.parseProject(projectId, { force });
      const p = projects.find((x) => x.id === projectId);
      setActiveProject(projectId, p?.name);
      await loadProjects();
      startPoll();
    } catch (err) {
      alert(`解析启动失败：${err.message || err}`);
    }
  }

  function fillBubbleContent(bubble, content, asAssistant) {
    const raw = String(content || "").replace(/\r\n/g, "\n");
    if (asAssistant && /^##\s+\S/m.test(raw)) {
      bubble.classList.add("bubble-md");
      bubble.innerHTML = raw
        .split("\n")
        .map((line) => {
          const h = line.match(/^##\s+(.+)$/);
          if (h) return `<div class="ans-h">${escapeHtml(h[1])}</div>`;
          if (line.trim() === "") return `<div class="ans-gap" aria-hidden="true"></div>`;
          return `<div class="ans-line">${escapeHtml(line)}</div>`;
        })
        .join("");
      return;
    }
    bubble.textContent = raw;
  }

  function expectedDesktopLogPath(sessionId) {
    const sid = String(sessionId || "").trim();
    if (!sid) return "";
    // Server writes under repo data/desktop_chat/<session_id>.log (absolute path arrives via SSE).
    return `data/desktop_chat/${sid}.log`;
  }

  function renderLogPathRow(logPath, { sessionId = "", pending = false } = {}) {
    const row = document.createElement("div");
    row.className = "chat-log-row";
    const pathEl = document.createElement("div");
    pathEl.className = "chat-log-path";
    const label = pending ? "对话日志（追加）" : "对话日志";
    const pathText = logPath || expectedDesktopLogPath(sessionId);
    pathEl.textContent = pathText
      ? `${label}：${pathText}`
      : `${label}：发送后由服务端写入 data/desktop_chat/<session_id>.log`;
    if (sessionId) {
      const sid = document.createElement("div");
      sid.className = "chat-log-session";
      sid.textContent = `session_id=${sessionId}`;
      row.appendChild(sid);
    }
    row.appendChild(pathEl);
    const actions = document.createElement("div");
    actions.className = "chat-log-actions";
    const copyBtn = document.createElement("button");
    copyBtn.type = "button";
    copyBtn.className = "btn-mini";
    copyBtn.textContent = "复制路径";
    copyBtn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const text = logPath || expectedDesktopLogPath(sessionId);
      if (!text) return;
      try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = "已复制";
        setTimeout(() => {
          copyBtn.textContent = "复制路径";
        }, 1200);
      } catch {
        /* ignore */
      }
    });
    actions.appendChild(copyBtn);
    if (logPath && window.desktop?.showItemInFolder) {
      const openBtn = document.createElement("button");
      openBtn.type = "button";
      openBtn.className = "btn-mini";
      openBtn.textContent = "打开所在文件夹";
      openBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const res = await window.desktop.showItemInFolder(logPath);
        if (res && res.ok === false) {
          alert(res.error || "无法打开日志路径（请确认 API 与桌面在同一台机器）");
        } else if (res && res.missing) {
          alert("日志文件尚未生成，已打开目录。请确认已重启带桌面日志的 API。");
        }
      });
      actions.appendChild(openBtn);
    }
    row.appendChild(actions);
    return row;
  }

  /**
   * Apple-style thinking bubble:
   * - Header with PEV badge (when applicable) and current step
   * - Entry meta strip
   * - Expandable step-by-step thinking trace (click to show detail)
   * - Plan steps summary if present
   */
  function renderThinkingBubble(m) {
    const wrap = document.createElement("div");
    wrap.className = "thinking-panel";
    const trace = usefulThinkTrace(m.trace);

    /* —— Header —— */
    const head = document.createElement("div");
    head.className = "thinking-header";
    head.textContent = "分析中…";
    wrap.appendChild(head);

    /* —— Current step —— */
    if (m.currentNote && !isNoiseThinkItem({ note: m.currentNote, step_id: "" })) {
      const cur = document.createElement("div");
      cur.className = "thinking-current-step";
      cur.textContent = humanizeThinkNote({ note: m.currentNote, status: "running" });
      wrap.appendChild(cur);
    }

    /* —— Log path (debug) —— */
    wrap.appendChild(
      renderLogPathRow(m.log_file || "", {
        sessionId: m.session_id || getActive()?.id || "",
        pending: true,
      })
    );

    /* —— Step-by-step trace —— */
    if (trace.length) {
      const sec = document.createElement("div");
      sec.className = "thinking-trace";
      const title = document.createElement("div");
      title.className = "thinking-trace-title";
      title.textContent = "思考过程";
      sec.appendChild(title);
      const list = document.createElement("ol");
      list.className = "thinking-trace-list";
      for (const item of trace) {
        const li = document.createElement("li");
        const st = item.status || "done";
        li.className = `thinking-trace-item is-${st === "error" ? "error" : st}`;

        const label = document.createElement("div");
        label.className = "thinking-trace-label";
        label.tabIndex = 0;
        label.setAttribute("role", "button");
        label.setAttribute("aria-expanded", "false");

        const icon = document.createElement("span");
        icon.style.flexShrink = "0";
        icon.style.width = "16px";
        icon.style.height = "16px";
        icon.style.display = "inline-flex";
        icon.style.alignItems = "center";
        icon.style.justifyContent = "center";
        if (st === "running") {
          icon.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14"><circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-dasharray="28" stroke-dashoffset="10"><animateTransform attributeName="transform" type="rotate" from="0 8 8" to="360 8 8" dur="0.8s" repeatCount="indefinite"/></circle></svg>`;
          icon.style.color = "var(--accent)";
        } else if (st === "error") {
          icon.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14"><path fill="var(--bad)" d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm1 10.5H7V7h2v4.5zm0-6H7V3h2v2.5z"/></svg>`;
        } else {
          icon.innerHTML = `<svg viewBox="0 0 16 16" width="14" height="14"><path fill="var(--ok)" d="M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-1 10.2L4 8.5l1.2-1.2L7 9.8l3.8-3.7L12 7.4 7 11.2z"/></svg>`;
        }
        label.appendChild(icon);

        const text = document.createElement("span");
        text.style.flex = "1";
        text.textContent = humanizeThinkNote(item);
        label.appendChild(text);

        if (st === "running") {
          const timeEl = document.createElement("span");
          timeEl.style.flexShrink = "0";
          timeEl.style.fontSize = "10px";
          timeEl.style.color = "var(--accent)";
          timeEl.style.fontFamily = "var(--mono)";
          timeEl.style.marginLeft = "auto";
          timeEl.textContent = "进行中";
          label.appendChild(timeEl);
        }

        li.appendChild(label);

        const detailText = item.detail || item.result || item.message || "";
        const shown = humanizeThinkNote(item);
        if (detailText && detailText !== shown && !/\[PEV\]/.test(detailText)) {
          const detail = document.createElement("div");
          detail.className = "thinking-trace-detail";
          detail.textContent = detailText;
          label.addEventListener("click", () => {
            const expanded = label.getAttribute("aria-expanded") === "true";
            label.setAttribute("aria-expanded", expanded ? "false" : "true");
            li.classList.toggle("is-expanded", !expanded);
          });
          label.addEventListener("keydown", (e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              label.click();
            }
          });
          li.appendChild(detail);
        }

        list.appendChild(li);
      }
      sec.appendChild(list);
      wrap.appendChild(sec);
    } else {
      const wait = document.createElement("div");
      wait.className = "thinking-waiting";
      wait.textContent = "等待服务端进度 ";
      wait.appendChild(createPulseDots());
      wrap.appendChild(wait);
    }

    return wrap;
  }

  function renderCollapsedThinking(trace) {
    const items = usefulThinkTrace(trace);
    if (!items.length) return document.createTextNode("");
    const d = document.createElement("details");
    d.className = "thinking-done";
    const sum = document.createElement("summary");
    sum.textContent = `思考过程（${items.length} 步）`;
    d.appendChild(sum);
    const list = document.createElement("ol");
    list.className = "thinking-trace-list";
    for (const item of items) {
      const li = document.createElement("li");
      li.className = "thinking-trace-item is-done";
      li.textContent = humanizeThinkNote(item);
      list.appendChild(li);
    }
    d.appendChild(list);
    return d;
  }

  function appendThinkingTrace(pending, ev) {
    if (!pending || !ev) return;
    if (!Array.isArray(pending.trace)) pending.trace = [];
    if (ev.type === "log" || ev.type === "heartbeat") return;
    const note =
      String(ev.note || ev.title || ev.message || "").trim() ||
      (ev.intent ? `意图识别 → ${ev.intent}` : "") ||
      String(ev.step_id || ev.type || "").trim();
    if (!note) return;
    if (isNoiseThinkItem({ type: ev.type, step_id: ev.step_id, note })) return;
    const last = pending.trace[pending.trace.length - 1];
    const sid = ev.step_id || "";
    // Update in place when same step is still running / completes.
    if (
      last &&
      sid &&
      last.step_id === sid &&
      (last.status === "running" || ev.status === "running")
    ) {
      last.status = ev.status || last.status;
      last.note = note;
      if (ev.detail) last.detail = String(ev.detail).slice(0, 400);
      if (ev.intent) last.intent = ev.intent;
    } else if (!last || last.note !== note || last.step_id !== sid) {
      pending.trace.push({
        at: Date.now(),
        type: ev.type || "step",
        status: ev.status || "done",
        step_id: sid,
        note,
        detail: ev.detail ? String(ev.detail).slice(0, 400) : "",
        intent: ev.intent || "",
      });
    }
    if (pending.trace.length > 50) pending.trace = pending.trace.slice(-50);
    pending.currentNote = note;
  }

  function usefulList(v) {
    return Array.isArray(v) && v.length > 0;
  }

  /** Build a git-style unified diff string from a patch object. */
  function buildUnifiedDiff(p) {
    if (!p || typeof p !== "object") return String(p || "");
    if (p.diff) return String(p.diff);
    const path = String(p.path || p.file || p.target || "file").replace(/\\/g, "/");
    const reps = Array.isArray(p.replacements) ? p.replacements : [];
    const isNew =
      p.new_file === true ||
      (p.content != null && !reps.length && !p.old && !p.old_string);

    if (isNew) {
      const body = String(p.content || "");
      const lines = body.length ? body.replace(/\r\n/g, "\n").split("\n") : [""];
      if (lines.length && lines[lines.length - 1] === "") lines.pop();
      const out = [
        `diff --git a/${path} b/${path}`,
        "new file mode 100644",
        "--- /dev/null",
        `+++ b/${path}`,
        `@@ -0,0 +1,${Math.max(1, lines.length)} @@`,
      ];
      for (const ln of lines) out.push("+" + ln);
      return out.join("\n");
    }

    if (reps.length) {
      const out = [
        `diff --git a/${path} b/${path}`,
        `--- a/${path}`,
        `+++ b/${path}`,
      ];
      for (const r of reps) {
        const oldLines = String(r.old || r.old_string || "")
          .replace(/\r\n/g, "\n")
          .split("\n");
        const newLines = String(r.new || r.new_string || "")
          .replace(/\r\n/g, "\n")
          .split("\n");
        if (oldLines.length && oldLines[oldLines.length - 1] === "") oldLines.pop();
        if (newLines.length && newLines[newLines.length - 1] === "") newLines.pop();
        const oStart = Number(r.old_start_line || r.start_line || 1) || 1;
        const nStart = Number(r.start_line || r.new_start_line || oStart) || 1;
        out.push(
          `@@ -${oStart},${Math.max(1, oldLines.length)} +${nStart},${Math.max(
            1,
            newLines.length
          )} @@`
        );
        for (const ln of oldLines) out.push("-" + ln);
        for (const ln of newLines) out.push("+" + ln);
      }
      return out.join("\n");
    }

    if (p.content != null) return buildUnifiedDiff({ ...p, new_file: true });
    return String(p.patch || p.code || JSON.stringify(p, null, 2));
  }

  function renderGitDiffHtml(diffText) {
    const lines = String(diffText || "").replace(/\r\n/g, "\n").split("\n");
    return lines
      .map((line) => {
        let cls = "diff-line";
        if (line.startsWith("+++") || line.startsWith("---")) cls += " diff-file";
        else if (line.startsWith("diff --git") || line.startsWith("index ") || line.startsWith("new file") || line.startsWith("deleted file"))
          cls += " diff-meta";
        else if (line.startsWith("@@")) cls += " diff-hunk";
        else if (line.startsWith("+")) cls += " diff-add";
        else if (line.startsWith("-")) cls += " diff-del";
        else cls += " diff-ctx";
        return `<div class="${cls}">${escapeHtml(line) || " "}</div>`;
      })
      .join("");
  }

  /** Align unified-diff lines into left(before) / right(after) rows. */
  function buildSideBySideRows(diffText) {
    const lines = String(diffText || "").replace(/\r\n/g, "\n").split("\n");
    const rows = [];
    /** @type {string[]} */
    let pendingDel = [];
    const flushDels = () => {
      while (pendingDel.length) {
        rows.push({ left: pendingDel.shift(), right: "", kind: "del" });
      }
    };
    for (const raw of lines) {
      const line = raw ?? "";
      if (
        line.startsWith("diff --git") ||
        line.startsWith("index ") ||
        line.startsWith("new file") ||
        line.startsWith("deleted file") ||
        line.startsWith("---") ||
        line.startsWith("+++")
      ) {
        continue;
      }
      if (line.startsWith("@@")) {
        flushDels();
        rows.push({ left: line, right: line, kind: "hunk" });
        continue;
      }
      if (line.startsWith("-")) {
        pendingDel.push(line.slice(1));
        continue;
      }
      if (line.startsWith("+")) {
        const add = line.slice(1);
        if (pendingDel.length) {
          rows.push({ left: pendingDel.shift(), right: add, kind: "change" });
        } else {
          rows.push({ left: "", right: add, kind: "add" });
        }
        continue;
      }
      flushDels();
      const ctx = line.startsWith(" ") ? line.slice(1) : line;
      rows.push({ left: ctx, right: ctx, kind: "ctx" });
    }
    flushDels();
    return rows;
  }

  function renderSideBySideHtml(diffText, kind) {
    const rows = buildSideBySideRows(diffText);
    if (!rows.length) {
      const emptyHint =
        kind === "added"
          ? "新建文件：左侧无旧内容"
          : "暂无可对照内容";
      return `<div class="diff-compare-empty">${escapeHtml(emptyHint)}</div>`;
    }
    const body = rows
      .map((r) => {
        const leftCls =
          r.kind === "del" || r.kind === "change"
            ? "diff-cell del"
            : r.kind === "hunk"
              ? "diff-cell hunk"
              : r.kind === "add"
                ? "diff-cell empty"
                : "diff-cell ctx";
        const rightCls =
          r.kind === "add" || r.kind === "change"
            ? "diff-cell add"
            : r.kind === "hunk"
              ? "diff-cell hunk"
              : r.kind === "del"
                ? "diff-cell empty"
                : "diff-cell ctx";
        const leftText = escapeHtml(r.left);
        const rightText = escapeHtml(r.right);
        return `<div class="diff-compare-row kind-${escapeHtml(r.kind)}">
          <div class="${leftCls}">${leftText || "&nbsp;"}</div>
          <div class="${rightCls}">${rightText || "&nbsp;"}</div>
        </div>`;
      })
      .join("");
    return `<div class="diff-compare">
      <div class="diff-compare-labels">
        <span>修改前</span>
        <span>修改后</span>
      </div>
      <div class="diff-compare-body">${body}</div>
    </div>`;
  }

  function syncPreviewViewButtons() {
    const unified = previewViewMode !== "compare";
    els.btnPreviewUnified?.classList.toggle("is-active", unified);
    els.btnPreviewCompare?.classList.toggle("is-active", !unified);
  }

  function setPreviewViewMode(mode) {
    previewViewMode = mode === "compare" ? "compare" : "unified";
    localStorage.setItem(PREVIEW_VIEW_KEY, previewViewMode);
    syncPreviewViewButtons();
    renderPreviewWindows();
  }

  function patchToFile(p, i) {
    const path = String(p.path || p.file || p.target || `patch-${i + 1}`);
    const diff = buildUnifiedDiff(p);
    const kind =
      p.new_file || (p.content != null && !(p.replacements || []).length)
        ? "added"
        : (p.replacements || []).length
          ? "modified"
          : "changed";
    return {
      path,
      body: diff,
      kind,
      diffHtml: renderGitDiffHtml(diff),
      compareHtml: renderSideBySideHtml(diff, kind),
    };
  }

  function applyPreviewWidth(px) {
    const w = Math.max(260, Math.min(Math.floor(window.innerWidth * 0.7), Number(px) || 380));
    document.documentElement.style.setProperty("--preview-w", `${w}px`);
    if (els.previewDock) {
      els.previewDock.style.flexBasis = `${w}px`;
      els.previewDock.style.width = `${w}px`;
    }
    return w;
  }

  function restorePreviewWidth() {
    const raw = localStorage.getItem(PREVIEW_W_KEY);
    applyPreviewWidth(raw ? Number(raw) : 380);
  }

  function renderPreviewFilebar() {
    if (!els.previewFilebar) return;
    els.previewFilebar.innerHTML = previewFiles
      .map((f, i) => {
        const name = f.path.split(/[\\/]/).pop() || f.path;
        const badge =
          f.kind === "added" ? "A" : f.kind === "modified" ? "M" : "C";
        return `<button type="button" class="preview-file-chip kind-${escapeHtml(
          f.kind || "changed"
        )}" draggable="true" data-fi="${i}" title="${escapeHtml(
          f.path
        )}"><span class="diff-status">${badge}</span>${escapeHtml(name)}</button>`;
      })
      .join("");
    els.previewFilebar.querySelectorAll(".preview-file-chip").forEach((chip) => {
      chip.addEventListener("dragstart", (e) => {
        e.dataTransfer.setData("text/preview-file", chip.getAttribute("data-fi") || "0");
        e.dataTransfer.effectAllowed = "copy";
      });
      chip.addEventListener("click", () => {
        const fi = Number(chip.getAttribute("data-fi") || 0);
        if (!previewWindows.length) addPreviewWindow(fi);
        else {
          previewWindows[previewWindows.length - 1].fileIndex = fi;
          renderPreviewWindows();
        }
      });
    });
  }

  function fileOptionsHtml(selectedIndex) {
    return previewFiles
      .map(
        (f, i) =>
          `<option value="${i}"${i === selectedIndex ? " selected" : ""}>${escapeHtml(
            f.path
          )}</option>`
      )
      .join("");
  }

  function renderPreviewWindows() {
    if (!els.previewSplits) return;
    els.previewSplits.classList.toggle("split-row", previewLayout === "row");
    els.previewSplits.classList.toggle("split-col", previewLayout === "col");
    if (!previewWindows.length) {
      els.previewSplits.innerHTML = `<div class="preview-empty">从上方文件列表点选或拖入此处</div>`;
      return;
    }
    els.previewSplits.innerHTML = previewWindows
      .map((win) => {
        const file = previewFiles[win.fileIndex] || previewFiles[0];
        const useCompare = previewViewMode === "compare";
        const bodyHtml = file
          ? useCompare
            ? file.compareHtml || ""
            : file.diffHtml || `<pre>${escapeHtml(file.body || "")}</pre>`
          : "";
        const badge =
          file?.kind === "added" ? "新增" : file?.kind === "modified" ? "修改" : "改动";
        const bodyClass = useCompare ? "preview-window-body is-compare" : "preview-window-body";
        const inner = useCompare
          ? bodyHtml
          : `<div class="git-diff">${bodyHtml}</div>`;
        return `<section class="preview-window" data-wid="${escapeHtml(win.id)}">
          <header class="preview-window-head">
            <select data-wid="${escapeHtml(win.id)}" aria-label="选择文件">
              ${fileOptionsHtml(win.fileIndex)}
            </select>
            <span class="diff-kind-tag">${escapeHtml(badge)}</span>
            <button type="button" class="win-close" data-close="${escapeHtml(
              win.id
            )}" title="关闭此窗">×</button>
          </header>
          <div class="${bodyClass}">${inner}</div>
        </section>`;
      })
      .join("");

    els.previewSplits.querySelectorAll("select[data-wid]").forEach((sel) => {
      sel.addEventListener("change", () => {
        const wid = sel.getAttribute("data-wid");
        const win = previewWindows.find((w) => w.id === wid);
        if (!win) return;
        win.fileIndex = Number(sel.value) || 0;
        renderPreviewWindows();
      });
    });
    els.previewSplits.querySelectorAll("[data-close]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const wid = btn.getAttribute("data-close");
        previewWindows = previewWindows.filter((w) => w.id !== wid);
        if (!previewWindows.length && previewFiles.length) {
          addPreviewWindow(0);
          return;
        }
        renderPreviewWindows();
      });
    });
    els.previewSplits.querySelectorAll(".preview-window").forEach((pane) => {
      pane.addEventListener("dragover", (e) => {
        e.preventDefault();
        pane.classList.add("drag-over");
      });
      pane.addEventListener("dragleave", () => pane.classList.remove("drag-over"));
      pane.addEventListener("drop", (e) => {
        e.preventDefault();
        pane.classList.remove("drag-over");
        const fi = Number(e.dataTransfer.getData("text/preview-file"));
        const wid = pane.getAttribute("data-wid");
        const win = previewWindows.find((w) => w.id === wid);
        if (!win || !Number.isFinite(fi)) return;
        win.fileIndex = fi;
        renderPreviewWindows();
      });
    });
    if (els.previewTitle) {
      const added = previewFiles.filter((f) => f.kind === "added").length;
      const modified = previewFiles.filter((f) => f.kind === "modified").length;
      const bits = [`${previewFiles.length} 文件`];
      if (added) bits.push(`${added} 新增`);
      if (modified) bits.push(`${modified} 修改`);
      bits.push(`${previewWindows.length} 窗`);
      els.previewTitle.textContent = `改动预览 · ${bits.join(" · ")}`;
    }
  }

  function addPreviewWindow(fileIndex = 0) {
    if (!previewFiles.length) return;
    previewWinSeq += 1;
    previewWindows.push({
      id: `pw_${previewWinSeq}`,
      fileIndex: Math.max(0, Math.min(previewFiles.length - 1, fileIndex)),
    });
    if (previewWindows.length >= 2 && previewLayout === "col") {
      // keep current layout; user can toggle via split button cycling
    }
    renderPreviewWindows();
  }

  function showPreview(patches) {
    if (!usefulList(patches)) {
      closePreview();
      return;
    }
    previewFiles = patches.map(patchToFile);
    previewWindows = [];
    previewWinSeq = 0;
    restorePreviewWidth();
    if (els.previewDock) els.previewDock.hidden = false;
    els.workbench.classList.add("has-preview");
    // Default: one window per file up to 2, rest selectable
    const initial = Math.min(2, previewFiles.length);
    for (let i = 0; i < initial; i += 1) addPreviewWindow(i);
    if (previewWindows.length > 1) previewLayout = "col";
    syncPreviewViewButtons();
    renderPreviewFilebar();
    renderPreviewWindows();
  }

  function closePreview() {
    if (els.previewDock) els.previewDock.hidden = true;
    els.workbench.classList.remove("has-preview");
    previewFiles = [];
    previewWindows = [];
    if (els.previewFilebar) els.previewFilebar.innerHTML = "";
    if (els.previewSplits) els.previewSplits.innerHTML = "";
    if (els.previewTitle) els.previewTitle.textContent = "改动预览";
  }

  function bindPreviewResizer() {
    const handle = els.previewResizer;
    const dock = els.previewDock;
    if (!handle || !dock) return;
    let dragging = false;
    handle.addEventListener("mousedown", (e) => {
      if (dock.hidden) return;
      dragging = true;
      handle.classList.add("active");
      document.body.classList.add("resizing-preview");
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (!dragging) return;
      const main = els.workbench?.getBoundingClientRect();
      if (!main) return;
      const width = main.right - e.clientX;
      const w = applyPreviewWidth(width);
      localStorage.setItem(PREVIEW_W_KEY, String(w));
    });
    window.addEventListener("mouseup", () => {
      if (!dragging) return;
      dragging = false;
      handle.classList.remove("active");
      document.body.classList.remove("resizing-preview");
    });
  }

  /** Modal confirm for dangerous / shell / git actions (Promise<boolean>). */
  window.BcaConfirmAction = function BcaConfirmAction(action) {
    return new Promise((resolve) => {
      const overlay = document.createElement("div");
      overlay.className = "confirm-overlay";
      const card = document.createElement("div");
      card.className = "confirm-card";
      const title = action?.ui_title || action?.type || "本机动作";
      const preview =
        action?.ui_preview ||
        action?.command ||
        (Array.isArray(action?.argv) ? action.argv.join(" ") : "") ||
        action?.path ||
        action?.note ||
        "";
      const tier = action?.policy_tier || "";
      card.innerHTML = `
        <h3>允许执行？</h3>
        <p class="confirm-title">${escapeHtml(String(title))}</p>
        <p class="confirm-tier">风险档位：${escapeHtml(String(tier || "unknown"))}</p>
        <pre class="confirm-preview">${escapeHtml(String(preview).slice(0, 2000))}</pre>
        <div class="confirm-actions">
          <button type="button" class="btn-mini" data-act="deny">拒绝</button>
          <button type="button" class="btn-mini primary" data-act="allow">允许本次</button>
        </div>`;
      overlay.appendChild(card);
      document.body.appendChild(overlay);
      const done = (ok) => {
        overlay.remove();
        resolve(!!ok);
      };
      card.querySelector('[data-act="deny"]')?.addEventListener("click", () => done(false));
      card.querySelector('[data-act="allow"]')?.addEventListener("click", () => done(true));
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) done(false);
      });
    });
  };

  /** Attach client-run tools under the owning assistant turn (not separate rows). */
  function findToolHostMessage(session, runId) {
    const msgs = session.messages || [];
    let fallback = null;
    for (let i = msgs.length - 1; i >= 0; i--) {
      const m = msgs[i];
      if (m.role !== "assistant" || m.pending || m.error) continue;
      if (runId && m.meta?.client_run_id === runId) {
        if (!m.meta) m.meta = {};
        if (!Array.isArray(m.meta.tool_timeline)) m.meta.tool_timeline = [];
        return m;
      }
      if (!fallback) fallback = m;
    }
    if (fallback) {
      if (!fallback.meta) fallback.meta = {};
      if (!Array.isArray(fallback.meta.tool_timeline)) fallback.meta.tool_timeline = [];
      if (runId && !fallback.meta.client_run_id) fallback.meta.client_run_id = runId;
      return fallback;
    }
    const host = {
      role: "assistant",
      content: "正在本机执行…",
      meta: { tool_timeline: [], client_run_id: runId || null },
    };
    msgs.push(host);
    return host;
  }

  function findTimelineEntry(timeline, actionId) {
    if (!Array.isArray(timeline) || !actionId) return -1;
    for (let i = timeline.length - 1; i >= 0; i--) {
      if (timeline[i]?.action?.id === actionId) return i;
    }
    return -1;
  }

  function appendToolTimeline(session, ev) {
    if (!session || !ev) return;
    const action = ev.action || {};
    const result = ev.result || {};
    const host = findToolHostMessage(session, ev.run_id || "");
    const timeline = host.meta.tool_timeline;

    if (ev.event === "task_plan") {
      const steps = Array.isArray(ev.plan_steps) ? ev.plan_steps : [];
      const lines = steps.map((s, i) => `${i + 1}. ${s}`);
      if (ev.success_criteria) {
        lines.push(`成功标准：${ev.success_criteria}`);
      }
      timeline.push({
        tool_event: "task_plan",
        content: ev.message || ev.plan_text || "任务计划",
        plan_steps: steps,
        success_criteria: ev.success_criteria || "",
        detail: lines.join("\n"),
        client_run_id: ev.run_id || "",
      });
    } else if (ev.event === "run_status") {
      const msg = String(ev.message || "").trim();
      if (!msg) return;
      const content = msg.includes("换方法") ? msg : `${msg}（换方法重试）`;
      const last = timeline[timeline.length - 1];
      // Collapse consecutive identical repairing notes.
      if (
        last &&
        last.tool_event === "run_progress" &&
        last.content === content
      ) {
        last.repair_count = ev.repair_count;
        last.max_repairs = ev.max_repairs;
        session.updatedAt = Date.now();
        saveSessions();
        renderAll();
        return;
      }
      timeline.push({
        tool_event: "run_progress",
        content,
        status: ev.status || "repairing",
        repair_count: ev.repair_count,
        max_repairs: ev.max_repairs,
        client_run_id: ev.run_id || "",
      });
    } else if (ev.event === "tool_call") {
      const idx = findTimelineEntry(timeline, action.id);
      if (idx >= 0 && timeline[idx].tool_event === "tool_call") {
        return;
      }
      timeline.push({
        tool_event: "tool_call",
        content: action.ui_title || action.type || "本机动作",
        action,
        client_run_id: ev.run_id || "",
      });
    } else if (ev.event === "tool_result") {
      const ok = result.ok !== false && !result.error;
      const preview = (
        result.stdout ||
        result.stderr ||
        result.error ||
        result.content ||
        ""
      ).slice(0, 4000);
      const entry = {
        tool_event: "tool_result",
        content: ok
          ? "执行成功"
          : `执行失败：${result.error || result.stderr || "error"}`,
        action,
        result,
        ok,
        preview,
        client_run_id: ev.run_id || "",
      };
      const idx = findTimelineEntry(timeline, action.id);
      if (idx >= 0) {
        entry.logs = timeline[idx].logs;
        timeline[idx] = entry;
      } else {
        timeline.push(entry);
      }
    } else if (ev.event === "tool_log") {
      const aid = ev.action_id;
      const idx = findTimelineEntry(timeline, aid);
      if (idx >= 0) {
        const item = timeline[idx];
        item.logs = (item.logs || "") + (ev.chunk || "");
        if (item.logs.length > 12000) item.logs = item.logs.slice(-12000);
      } else {
        // Legacy sessions / race: fall back to standalone tool messages
        for (let i = session.messages.length - 1; i >= 0; i--) {
          const m = session.messages[i];
          if (
            m.role === "tool" &&
            m.meta?.tool_event === "tool_call" &&
            (!aid || m.meta?.action?.id === aid)
          ) {
            m.meta.logs = (m.meta.logs || "") + (ev.chunk || "");
            if (m.meta.logs.length > 12000) {
              m.meta.logs = m.meta.logs.slice(-12000);
            }
            break;
          }
        }
      }
      refreshToolLogInDom(aid, ev.chunk);
      return;
    } else {
      return;
    }
    session.updatedAt = Date.now();
    saveSessions();
    renderAll();
  }

  function refreshToolLogInDom(actionId, chunk) {
    if (!chunk) return;
    const nodes = els.transcript.querySelectorAll(".tool-card[data-action-id]");
    for (const node of nodes) {
      if (actionId && node.getAttribute("data-action-id") !== actionId) continue;
      const pre = node.querySelector(".tool-log");
      if (!pre) continue;
      pre.textContent = (pre.textContent || "") + chunk;
      if (pre.textContent.length > 12000) {
        pre.textContent = pre.textContent.slice(-12000);
      }
      els.transcript.scrollTop = els.transcript.scrollHeight;
      break;
    }
  }

  function renderToolCardFromMeta(meta, fallbackContent) {
    const action = meta.action || {};
    const wrap = document.createElement("div");
    wrap.className =
      "tool-card " +
      (meta.tool_event === "task_plan"
        ? "tool-plan"
        : meta.tool_event === "run_progress"
          ? "tool-plan"
          : meta.tool_event === "tool_result"
            ? meta.ok
              ? "tool-ok"
              : "tool-fail"
            : "tool-running");
    if (action.id) wrap.setAttribute("data-action-id", action.id);
    const head = document.createElement("div");
    head.className = "tool-card-head";
    const badge = document.createElement("span");
    badge.className = "tool-badge";
    badge.textContent =
      meta.tool_event === "task_plan"
        ? "计划"
        : meta.tool_event === "run_progress"
          ? "换方法"
          : meta.tool_event === "tool_result"
            ? meta.ok
              ? "完成"
              : "失败"
            : "执行";
    const title = document.createElement("strong");
    title.textContent =
      meta.tool_event === "task_plan"
        ? "任务分解"
        : meta.tool_event === "run_progress"
          ? "根据控制台失败换方法重试"
          : action.ui_title || action.type || fallbackContent || "本机动作";
    head.appendChild(badge);
    head.appendChild(title);
    wrap.appendChild(head);
    const preview =
      meta.tool_event === "task_plan"
        ? meta.detail || fallbackContent || ""
        : meta.tool_event === "run_progress"
          ? meta.content || fallbackContent || ""
          : action.ui_preview || action.command || action.path || "";
    if (preview) {
      const pre = document.createElement("pre");
      pre.className = "tool-preview";
      pre.textContent = String(preview).slice(0, 800);
      wrap.appendChild(pre);
    }
    if (meta.preview) {
      const details = document.createElement("details");
      details.open = !meta.ok;
      details.innerHTML = `<summary>输出</summary><pre class="tool-log">${escapeHtml(
        String(meta.preview).slice(0, 4000)
      )}</pre>`;
      wrap.appendChild(details);
    } else if (meta.logs) {
      const details = document.createElement("details");
      details.open = true;
      const sum = document.createElement("summary");
      sum.textContent = "实时日志";
      const pre = document.createElement("pre");
      pre.className = "tool-log";
      pre.textContent = meta.logs;
      details.appendChild(sum);
      details.appendChild(pre);
      wrap.appendChild(details);
    }
    return wrap;
  }

  function renderToolCard(m) {
    return renderToolCardFromMeta(m.meta || {}, m.content);
  }

  function renderAssistantToolTimeline(timeline) {
    if (!usefulList(timeline)) return null;
    const wrap = document.createElement("div");
    wrap.className = "assistant-tools";
    for (const entry of timeline) {
      wrap.appendChild(
        renderToolCardFromMeta(entry, entry.content || "")
      );
    }
    return wrap;
  }

  function startClientRun(runId, session) {
    if (!runId || !window.BcaClientRuns) return;
    if (!window.desktop?.runAction) {
      paintPatchStatus("本机任务需要桌面客户端", "bad");
      return;
    }
    // 记录 run 发起时刻，onFinished 用「发起→结束」更新消息的用时，
    // 避免 elapsed_ms 停在发起时的 2s 固定不变。
    const runStartedAt = performance.now();
    window.BcaClientRuns.onStatus = paintPatchStatus;
    window.BcaClientRuns.onToolEvent = (ev) => {
      const session2 = getActive() || session;
      if (!session2) return;
      appendToolTimeline(session2, ev);
    };
    window.BcaClientRuns.onFinished = (msg) => {
      const session2 = getActive() || session;
      if (!session2) return;
      const reply = msg.reply || msg.message || "";
      const hasOpts = usefulList(msg.options);
      if (!reply && !hasOpts) return;
      const st = msg.status || "finished";
      const host = findToolHostMessage(session2, runId);
      if (!host.meta) host.meta = {};
      // 本机 run 真正结束时更新用时（发起 → 结束）
      host.elapsed_ms = Math.round(performance.now() - runStartedAt);
      const finishedText =
        reply || (st === "succeeded" ? "本机任务完成" : `本机任务结束：${st}`);
      // Keep reply + tools in one AI turn instead of a second avatar row.
      host.meta.run_reply = finishedText;
      host.meta.intent =
        hasOpts || String(msg.phase || "").includes("awaiting")
          ? "client_agent"
          : msg.phase
            ? "client_agent"
            : host.meta.intent || "verify";
      host.meta.phase =
        msg.phase ||
        (hasOpts ? "awaiting_cleanup" : st === "succeeded" ? "done" : "failed");
      if (hasOpts) host.meta.options = msg.options || [];
      host.meta.cleanup_catalog = msg.catalog || msg.options || host.meta.cleanup_catalog;
      session2.updatedAt = Date.now();
      saveSessions();
      renderAll();
    };
    window.BcaClientRuns.start(runId);
  }

  function renderOptionsCards(options, meta) {
    if (!usefulList(options)) return null;
    const displayTargetPath = (p) => {
      const s = String(p || "").trim();
      if (!s) return "";
      if (/^RECYCLE_BIN$/i.test(s)) return "系统回收站";
      return s;
    };
    const wrap = document.createElement("div");
    wrap.className = "option-cards";
    const phase = String(meta?.phase || "");
    const intent = String(meta?.intent || "");
    // 仅当后端明确进入「等待清理」阶段（awaiting_cleanup）才用勾选+清理 UI；
    // 普通 client_agent 的选项（继续/停止/方案等）一律用「选择方案」卡片，
    // 避免把普通选项误渲染成「清理所选」勾选框。
    const cleanupMode = phase === "awaiting_cleanup";

    if (cleanupMode) {
      const picks = new Set();
      for (const opt of options) {
        if (!opt || !opt.id) continue;
        const card = document.createElement("label");
        card.className = "option-card option-card-check";
        const row = document.createElement("div");
        row.className = "option-check-row";
        const cb = document.createElement("input");
        cb.type = "checkbox";
        cb.value = String(opt.id);
        cb.addEventListener("change", () => {
          if (cb.checked) picks.add(cb.value);
          else picks.delete(cb.value);
        });
        const title = document.createElement("div");
        title.className = "option-card-title";
        title.textContent = opt.title || opt.id;
        row.appendChild(cb);
        row.appendChild(title);
        card.appendChild(row);
        if (opt.summary) {
          const sum = document.createElement("div");
          sum.className = "option-card-summary";
          sum.textContent = String(opt.summary);
          card.appendChild(sum);
        }
        const pathText = displayTargetPath(opt.path);
        if (pathText) {
          const pathEl = document.createElement("div");
          pathEl.className = "option-card-path";
          pathEl.textContent = `路径：${pathText}`;
          card.appendChild(pathEl);
        }
        wrap.appendChild(card);
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-mini primary option-pick-btn";
      btn.textContent = "清理所选";
      btn.disabled = !!sending;
      btn.addEventListener("click", () => {
        const ids = [...picks];
        if (!ids.length) {
          alert("请先勾选要清理的项");
          return;
        }
        const names = options
          .filter((o) => ids.includes(String(o.id)))
          .map((o) => {
            const title = o.title || o.id;
            const p = displayTargetPath(o.path);
            return p ? `${title}（${p}）` : title;
          })
          .join("、");
        sendDesktopAction(`请帮我清理：${names}`, {
          action: "cleanup_selected",
          option_ids: ids,
          cleanup_catalog: meta?.cleanup_catalog || options,
        });
      });
      wrap.appendChild(btn);
      return wrap;
    }

    for (const opt of options) {
      if (!opt || !opt.id) continue;
      const card = document.createElement("div");
      card.className = "option-card";
      const title = document.createElement("div");
      title.className = "option-card-title";
      title.textContent =
        String(opt.id) === "implement_now"
          ? opt.title || "直接改代码并本机编译验证"
          : `方案 ${opt.id}：${opt.title || ""}`;
      card.appendChild(title);
      if (opt.summary) {
        const sum = document.createElement("div");
        sum.className = "option-card-summary";
        sum.textContent = String(opt.summary);
        card.appendChild(sum);
      }
      const pathText = displayTargetPath(opt.path);
      if (pathText) {
        const pathEl = document.createElement("div");
        pathEl.className = "option-card-path";
        pathEl.textContent = `路径：${pathText}`;
        card.appendChild(pathEl);
      }
      if (opt.tradeoffs) {
        const tr = document.createElement("div");
        tr.className = "option-card-tradeoffs";
        tr.textContent = `利弊：${opt.tradeoffs}`;
        card.appendChild(tr);
      }
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-mini primary option-pick-btn";
      const picked = meta?.selected_option_id === opt.id;
      const canPick =
        !phase ||
        phase === "awaiting_option" ||
        String(opt.id) === "implement_now";
      btn.textContent =
        String(opt.id) === "implement_now"
          ? picked
            ? "已开始落地"
            : "开始改代码"
          : picked
            ? `已选 ${opt.id}`
            : `选择方案 ${opt.id}`;
      btn.disabled = !!sending || picked || !canPick;
      btn.addEventListener("click", () => {
        if (sending || picked) return;
        // Instant visual feedback before network / project-root bind.
        btn.disabled = true;
        btn.textContent =
          String(opt.id) === "implement_now" ? "正在改代码…" : `正在选择 ${opt.id}…`;
        sendDesktopAction(
          String(opt.id) === "implement_now"
            ? "请按上述改动直接改代码，先不要编译自测"
            : `选方案${opt.id}`,
          {
            action: "select_option",
            option_id: String(opt.id),
          }
        );
      });
      card.appendChild(btn);
      wrap.appendChild(card);
    }
    return wrap;
  }

  function renderMessageMeta(meta) {
    if (!meta) return null;
    const scopeOk = usefulList(meta.scope);
    const patchesOk = usefulList(meta.patches);
    const optionsOk = usefulList(meta.options);
    const hasLog = !!(meta.log_file || meta.session_id);
    if (!scopeOk && !patchesOk && !optionsOk && !hasLog && !meta.verify_report) {
      return null;
    }
    const box = document.createElement("div");
    box.className = "details";
    if (hasLog) {
      box.appendChild(
        renderLogPathRow(meta.log_file || "", {
          sessionId: meta.session_id || getActive()?.id || "",
        })
      );
    }
    if (optionsOk) {
      const cards = renderOptionsCards(meta.options, meta);
      if (cards) box.appendChild(cards);
    }
    if (scopeOk) {
      const d = document.createElement("details");
      d.innerHTML = `<summary>修改范围</summary><pre>${escapeHtml(
        JSON.stringify(meta.scope, null, 2)
      )}</pre>`;
      box.appendChild(d);
    }
    if (patchesOk) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "btn-mini primary";
      btn.style.marginTop = "6px";
      btn.textContent = "查看 git diff 预览";
      btn.addEventListener("click", () => showPreview(meta.patches));
      box.appendChild(btn);
      const diffWrap = document.createElement("details");
      diffWrap.className = "git-diff-details";
      const sum = document.createElement("summary");
      sum.textContent = `文件改动（${meta.patches.length}，类 git）`;
      diffWrap.appendChild(sum);
      for (const p of meta.patches.slice(0, 12)) {
        const file = patchToFile(p, 0);
        const block = document.createElement("div");
        block.className = "git-diff-inline";
        const head = document.createElement("div");
        head.className = "git-diff-inline-path";
        head.textContent = `${file.kind === "added" ? "A" : "M"}  ${file.path}`;
        block.appendChild(head);
        const pre = document.createElement("div");
        pre.className = "git-diff";
        pre.innerHTML = file.diffHtml;
        block.appendChild(pre);
        diffWrap.appendChild(block);
      }
      box.appendChild(diffWrap);
    }
    if (meta.verify_report) {
      const d = document.createElement("details");
      const vr = meta.verify_report;
      d.innerHTML = `<summary>验证报告</summary><pre>${escapeHtml(
        `compile=${vr.compile_ok ? "ok" : "fail"} test=${
          vr.test_skipped ? "skipped" : vr.test_ok ? "ok" : "fail"
        }\n${vr.logs || ""}`.slice(0, 4000)
      )}</pre>`;
      box.appendChild(d);
    }
    return box;
  }

  function renderTranscript() {
    const session = getActive();
    els.chatTitle.textContent = session?.title || "新对话";
    const msgs = session?.messages || [];
    if (!msgs.length) {
      els.transcript.replaceChildren(els.emptyState);
      els.emptyState.hidden = false;
      return;
    }
    els.emptyState.hidden = true;
    const frag = document.createDocumentFragment();
    for (const m of msgs) {
      const row = document.createElement("div");
      row.className = `msg ${
        m.role === "user" ? "user" : m.role === "tool" ? "tool" : "assistant"
      }`;
      const av = document.createElement("div");
      av.className = "avatar";
      av.textContent = m.role === "user" ? "你" : m.role === "tool" ? "执" : "AI";
      const col = document.createElement("div");
      col.className = "bubble-col";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      if (m.role === "tool") {
        bubble.classList.add("bubble-tool");
        bubble.appendChild(renderToolCard(m));
      } else if (m.pending) {
        bubble.classList.add("bubble-thinking");
        const hasThink = usefulThinkTrace(m.trace).length > 0 || m.plan;
        if (m._usePulse && !hasThink) {
          const pulseWrap = document.createElement("div");
          pulseWrap.className = "thinking-panel";
          pulseWrap.appendChild(createPulseDots());
          bubble.appendChild(pulseWrap);
        } else {
          bubble.appendChild(renderThinkingBubble(m));
        }
      } else if (m.error) {
        bubble.classList.add("bubble-error");
        bubble.textContent = m.content || m.error;
      } else if (m._streaming) {
        // Leave empty for streaming — the streaming timer fills it live
        bubble.classList.add("bubble-streaming");
        bubble.textContent = "";
      } else {
        fillBubbleContent(bubble, m.content, m.role === "assistant");
      }
      col.appendChild(bubble);
      if (m.role === "assistant" && !m.pending && usefulThinkTrace(m.thinkingTrace).length) {
        col.insertBefore(renderCollapsedThinking(m.thinkingTrace), bubble);
      }
      // Nest client-run tools under the same AI turn as thinking/reply.
      if (m.role === "assistant" && !m.pending) {
        const toolsEl = renderAssistantToolTimeline(m.meta?.tool_timeline);
        if (toolsEl) col.appendChild(toolsEl);
        if (m.meta?.run_reply) {
          const runBubble = document.createElement("div");
          runBubble.className = "bubble bubble-run-reply";
          fillBubbleContent(runBubble, m.meta.run_reply, true);
          col.appendChild(runBubble);
        }
      }
      if (m.pending) {
        const t = document.createElement("div");
        t.className = "timing thinking";
        const pulse = createPulseDots();
        const textSpan = document.createElement("span");
        textSpan.style.marginLeft = "6px";
        textSpan.textContent = "思考中…";
        t.appendChild(pulse);
        t.appendChild(textSpan);
        col.appendChild(t);
      } else if (
        m.meta?.client_run_id &&
        !m.meta?.run_reply &&
        !m.meta?.phase?.includes("done")
      ) {
        // 本机 run 还在执行中（WebSocket 继续跑）：不显示固定用时
        const t = document.createElement("div");
        t.className = "timing thinking";
        t.textContent = "本机执行中…";
        col.appendChild(t);
      } else if (typeof m.elapsed_ms === "number") {
        const t = document.createElement("div");
        t.className = "timing";
        t.textContent = `用时 ${formatElapsed(m.elapsed_ms)}`;
        col.appendChild(t);
      }
      const metaEl = renderMessageMeta(m.meta);
      if (metaEl) col.appendChild(metaEl);
      if (
        m.role === "assistant" &&
        !m.pending &&
        !m.error &&
        !String(m.content || "").startsWith("请求失败")
      ) {
        col.appendChild(renderFeedbackBar(m));
      } else if (!m.pending && messageCopyText(m)) {
        // User / tool / error bubbles: copy after the dialog.
        col.appendChild(renderMsgActions(m));
      }
      row.appendChild(av);
      row.appendChild(col);
      frag.appendChild(row);
    }
    els.transcript.replaceChildren(frag);
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function renderAll() {
    renderSessions();
    renderTranscript();
  }

  function refreshPendingBubble(session) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    const thinking = els.transcript.querySelector(".msg.assistant .bubble.bubble-thinking");
    if (thinking) {
      const hasThink = usefulThinkTrace(pending.trace).length > 0 || pending.plan;
      if (pending._usePulse && !hasThink) {
        const pulseWrap = document.createElement("div");
        pulseWrap.className = "thinking-panel";
        pulseWrap.appendChild(createPulseDots());
        thinking.replaceChildren(pulseWrap);
      } else {
        thinking.replaceChildren(renderThinkingBubble(pending));
        pending._usePulse = false;
      }
      els.transcript.scrollTop = els.transcript.scrollHeight;
      return;
    }
    renderAll();
  }

  function patchPendingPlan(session, plan, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending || !plan) return;
    pending._usePulse = false;
    pending.plan = plan;
    appendThinkingTrace(pending, {
      type: "plan",
      status: ev?.status || "running",
      step_id: ev?.step_id || "goal",
      note: ev?.note || "正在分析问题目标…",
    });
    refreshPendingBubble(session);
  }

  function patchPendingStep(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    pending._usePulse = false; // Switch to detailed view
    if (ev.plan) {
      pending.plan = ev.plan;
    } else {
      if (!pending.plan) {
        pending.plan = {
          kind: "flow",
          title: "分析中",
          steps: [],
        };
      }
      const steps = pending.plan.steps || (pending.plan.steps = []);
      const sid = ev.step_id || "step";
      let step = steps.find((s) => s.id === sid);
      if (!step) {
        step = {
          id: sid,
          title: ev.note || sid,
          status: "pending",
        };
        steps.push(step);
      }
      if (ev.status) step.status = ev.status === "error" ? "error" : ev.status;
      if (ev.note) {
        step.note = ev.note;
        if (!step.title || step.title === sid) step.title = ev.note;
      }
    }
    appendThinkingTrace(pending, ev);
    refreshPendingBubble(session);
  }

  function patchPendingIntent(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    pending._usePulse = false; // Switch to detailed view
    const intent = ev.intent || "";
    const note =
      ev.note ||
      (intent
        ? `意图识别 → ${intent}${ev.want_apply ? "（将动手改代码）" : ""}`
        : "意图识别完成");
    appendThinkingTrace(pending, {
      type: "intent",
      status: "done",
      step_id: "intent",
      note,
      detail: ev.detail || "",
      intent,
    });
    // Mirror into plan steps for visibility
    if (!pending.plan) {
      pending.plan = { kind: "flow", title: "执行进度", steps: [] };
    }
    const steps = pending.plan.steps || (pending.plan.steps = []);
    let step = steps.find((s) => s.id === "intent");
    if (!step) {
      step = { id: "intent", title: "意图识别", status: "done" };
      steps.push(step);
    }
    step.status = "done";
    step.note = note;
    refreshPendingBubble(session);
  }

  function patchPendingLog(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    if (ev.log_file) pending.log_file = ev.log_file;
    refreshPendingBubble(session);
  }

  function patchPendingReview(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    const status = ev.status || "running";
    if (status === "running") pending.reviewNote = ev.note || "核对回答细节…";
    else if (status === "rejected") {
      pending.reviewNote = "补充更多细节后继续完善…";
    } else {
      pending.reviewNote = "细节已较完整：" + (ev.note || "可以作答");
    }
    appendThinkingTrace(pending, {
      type: "review",
      status: status === "rejected" ? "error" : status,
      step_id: "review",
      note: pending.reviewNote,
    });
    refreshPendingBubble(session);
  }

  function patchPendingHeartbeat(session) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    if (usefulThinkTrace(pending.trace).length || pending.plan) return;
    if (!pending.currentNote) {
      pending.currentNote = "正在处理…";
      pending._usePulse = true;
      refreshPendingBubble(session);
    }
  }

  function autoResize() {
    if (!els.input) return;
    els.input.style.height = "auto";
    const next = Math.min(Math.max(els.input.scrollHeight || 0, 42), 200);
    els.input.style.height = `${next}px`;
  }

  function paintPatchStatus(text, kind) {
    if (!els.patchStatus) return;
    if (!text) {
      els.patchStatus.hidden = true;
      els.patchStatus.textContent = "";
      if (els.btnStopPatch) els.btnStopPatch.hidden = true;
      return;
    }
    els.patchStatus.hidden = false;
    els.patchStatus.textContent = text;
    els.patchStatus.dataset.kind = kind || "";
    if (els.btnStopPatch) els.btnStopPatch.hidden = kind !== "busy";
  }

  async function postChat(body, session) {
    let data = null;
    try {
      let streamError = null;
      // /chat/stream ≡ LocalChatTester on_progress + final result
      await window.BcaApi.chatStream(body, (ev) => {
        if (!ev || !ev.type) return;
        if (ev.type === "log") {
          patchPendingLog(session, ev);
        } else if (ev.type === "plan") {
          if (ev.plan) patchPendingPlan(session, ev.plan, ev);
          else patchPendingStep(session, ev);
        } else if (ev.type === "step") {
          patchPendingStep(session, ev);
        } else if (ev.type === "intent") {
          patchPendingIntent(session, ev);
        } else if (ev.type === "review") {
          patchPendingReview(session, ev);
        } else if (ev.type === "heartbeat") {
          patchPendingHeartbeat(session);
        } else if (ev.type === "final") {
          data = ev;
        } else if (ev.type === "error") {
          streamError = ev.message || "流式处理失败";
        } else if (ev.note || ev.step_id) {
          // Unknown progress types that still carry a note
          patchPendingStep(session, { ...ev, type: ev.type || "step" });
        }
      });
      if (streamError) throw new Error(streamError);
      if (!data) throw new Error("流式响应未返回最终结果");
    } catch {
      data = await window.BcaApi.chat(body);
    }
    return data;
  }

  /* ── Streaming Text Effect ── */
  let streamTimer = null;

  function streamTextIntoElement(el, text, speed = 18, callback) {
    if (streamTimer) clearInterval(streamTimer);
    el.textContent = "";
    // Add a cursor span for the typing effect
    const cursor = document.createElement("span");
    cursor.className = "stream-cursor";
    cursor.textContent = "|";
    cursor.style.animation = "cursorBlink 0.8s step-end infinite";
    cursor.style.color = "var(--accent)";
    cursor.style.fontWeight = "300";
    el.appendChild(cursor);
    let i = 0;
    const chars = [...text];
    const step = () => {
      if (i >= chars.length) {
        cursor.remove();
        clearInterval(streamTimer);
        streamTimer = null;
        if (callback) callback();
        return;
      }
      // Add multiple characters per tick for natural feel
      const batchSize = Math.max(1, Math.floor(chars.length / 70) || 1);
      for (let j = 0; j < batchSize && i < chars.length; j++, i++) {
        el.insertBefore(document.createTextNode(chars[i]), cursor);
      }
      // Auto-scroll transcript as text appears
      const transcript = els.transcript;
      if (transcript) transcript.scrollTop = transcript.scrollHeight;
    };
    step(); // immediate first batch
    streamTimer = setInterval(step, speed);
  }

  function applyChatResult(session, data, t0, content) {
    const clientMs = Math.round(performance.now() - t0);
    const elapsedMs =
      typeof data.elapsed_ms === "number" && data.elapsed_ms >= 0
        ? data.elapsed_ms
        : clientMs;
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    const thinkingTrace = usefulThinkTrace(pending?.trace);
    session.messages = session.messages.filter((m) => !m.pending);
    if (data.session_id) {
      session.id = data.session_id;
      activeId = data.session_id;
      localStorage.setItem(ACTIVE_KEY, activeId);
    }
    const assistantMessageId =
      typeof data.assistant_message_id === "number" ? data.assistant_message_id : null;
    const plan = data.plan || null;
    // Attach spine/entry from PEV evidence footer if present on plan.
    if (plan && data.explore_steps) {
      for (const s of data.explore_steps) {
        if (s && s.entry && !plan.entry) plan.entry = s.entry;
        if (s && s.spine_source && !plan.spine_source) plan.spine_source = s.spine_source;
      }
    }
    session.messages.push({
      role: "assistant",
      id: assistantMessageId,
      content: data.reply || "(空回复)",
      elapsed_ms: elapsedMs,
      thinkingTrace,
      meta: {
        intent: data.intent,
        scope: data.scope,
        patches: data.patches,
        plan,
        review: data.review || null,
        phase: data.phase || null,
        options: data.options || [],
        selected_option_id: data.selected_option_id || null,
        verify_report: data.verify_report || null,
        client_run_id: data.client_run_id || null,
        log_file: data.log_file || null,
        session_id: data.session_id || session.id || null,
        explore_steps: data.explore_steps || [],
      },
      _streaming: true, // mark for streaming render
    });
    session.updatedAt = Date.now();
    saveSessions();
    stopThinkingTimer();
    renderAll();
    if (usefulList(data.patches)) showPreview(data.patches);
    if (data.client_run_id) {
      startClientRun(data.client_run_id, session);
    }
    // Start streaming the final answer text (plain text only; markdown renders immediately)
    const lastMsg = session.messages[session.messages.length - 1];
    if (lastMsg && lastMsg._streaming && lastMsg.content && !/^##\s+\S/m.test(lastMsg.content)) {
      const bubble = els.transcript.querySelector(".msg:last-child .bubble:not(.bubble-thinking):not(.bubble-error)");
      if (bubble) {
        const replyText = lastMsg.content;
        // Clear the bubble (which was pre-filled with plain text) and stream
        bubble.textContent = "";
        streamTextIntoElement(bubble, replyText, 14, () => {
          lastMsg._streaming = false;
          // Re-render with markdown formatting once streaming completes
          fillBubbleContent(bubble, replyText, true);
          saveSessions();
        });
      }
    }
  }

  function applyChatError(session, err, content) {
    session.messages = session.messages.filter((m) => !m.pending);
    session.messages.push({
      role: "assistant",
      error: true,
      content: `请求失败：${err.message || err}`,
      retryText: content,
    });
    session.updatedAt = Date.now();
    saveSessions();
    stopThinkingTimer();
    renderAll();
  }

  async function beginTurn(content, { clearComposer = true } = {}) {
    sending = true;
    els.send.disabled = true;
    const session = ensureActive();
    if (session.title === "新对话") session.title = content.slice(0, 28);
    const t0 = performance.now();
    session.messages.push({ role: "user", content });
    session.messages.push({
      role: "assistant",
      content: "",
      pending: true,
      thinkingStartedAt: t0,
      session_id: session.id || null,
      log_file: null,
      currentNote: "连接服务端…",
      _usePulse: true, // use animated pulse dots
      trace: [
        {
          at: t0,
          type: "local",
          status: "running",
          step_id: "local_start",
          note: "已收到请求，正在连接服务端…",
        },
      ],
    });
    session.updatedAt = Date.now();
    saveSessions();
    renderAll();
    startThinkingTimer(t0);
    if (clearComposer) {
      els.input.value = "";
      autoResize();
    }
    return { session, t0 };
  }

  function endTurn() {
    sending = false;
    els.send.disabled = false;
    els.input.focus();
  }

  /** True in Electron desktop — enables local whitelist executor (code + OS assist). */
  function isDesktopClient() {
    return Boolean(window.desktop?.runAction);
  }

  /**
   * Desktop composer: enable client_agent so OS assist / patch+compile / start_app
   * go through remote_loop instead of falling into code-only PEV diagnose.
   * CLI LocalChatTester stays unchanged (no window.desktop).
   */
  async function sendMessage(text) {
    const content = (text || "").trim();
    if (!content || sending) return;
    const pids = activeProjectIds();
    const desktop = isDesktopClient();
    if (!pids.length && !desktop) {
      alert("请先在「项目」中勾选项目后再提问（与 local_chat_interactive 一致）。");
      setView("projects");
      return;
    }
    if (!pids.length && desktop) {
      // 未选择项目：普通问答（带上下文记忆）；OS/本机问题仍可走本机助手；
      // 若问题需要基于项目代码，服务端会提示先勾选项目。
    }

    const projectId = pids[0] || undefined;
    if (projectId && window.desktop?.useProjectRoot) {
      const bound = await resolveImportLocalRoot([projectId]);
      if (bound) {
        await window.desktop.useProjectRoot(bound.projectId);
      }
    }

    const { session, t0 } = await beginTurn(content);
    try {
      const body = {
        message: content,
        session_id: session.id,
        project_id: projectId,
        // 显式携带勾选状态：[] = 未选择项目（服务端走普通问答），非空 = 项目问答。
        project_ids: pids,
      };
      if (desktop) {
        body.client_agent = true;
        body.client_apply = true;
      }
      const data = await postChat(body, session);
      applyChatResult(session, data, t0, content);
    } catch (err) {
      applyChatError(session, err, content);
    } finally {
      endTurn();
    }
  }

  /**
   * Desktop-only structured actions (cleanup / select_option / …).
   */
  async function sendDesktopAction(text, extras) {
    const content = (text || "").trim();
    if (!content || sending || !extras) return;
    const pids = activeProjectIds();
    const desktop = isDesktopClient();
    const wantCleanup = extras.action === "cleanup_selected";
    if (!pids.length && !wantCleanup && !desktop) {
      alert("请先在「项目」中勾选项目。");
      setView("projects");
      return;
    }

    // Show thinking bubble immediately — don't wait for project-root IPC.
    const { session, t0 } = await beginTurn(content, { clearComposer: false });
    try {
      if (pids.length && window.desktop?.useProjectRoot) {
        try {
          const bound = await resolveImportLocalRoot(pids);
          if (bound) {
            await window.desktop.useProjectRoot(bound.projectId);
          }
        } catch (bindErr) {
          console.warn("useProjectRoot failed", bindErr);
        }
      }
      const body = {
        message: content,
        session_id: session.id,
        project_id: pids[0] || undefined,
        // 显式携带勾选状态：[] = 未选择项目（服务端走普通问答），非空 = 项目问答。
        project_ids: pids,
      };
      if (desktop || wantCleanup) {
        body.client_agent = true;
        body.client_apply = true;
      }
      if (extras.action) body.action = extras.action;
      if (extras.option_id) body.option_id = extras.option_id;
      if (extras.option_ids) body.option_ids = extras.option_ids;
      if (extras.cleanup_catalog) body.cleanup_catalog = extras.cleanup_catalog;

      const data = await postChat(body, session);
      applyChatResult(session, data, t0, content);
    } catch (err) {
      applyChatError(session, err, content);
    } finally {
      endTurn();
    }
  }

  async function checkHealth() {
    try {
      const data = await window.BcaApi.health();
      const ok = data.status === "ok" || data.status === "degraded";
      els.healthStatus.textContent = ok ? data.llm_provider || "已连接" : "异常";
      els.healthStatus.className = "pill " + (ok ? "ok" : "bad");
      els.healthDot.className = "health " + (ok ? "ok" : "bad");
      els.healthDot.title = ok ? `后端正常 · ${window.BcaApi.base}` : "后端异常";
    } catch {
      els.healthStatus.textContent = "离线";
      els.healthStatus.className = "pill bad";
      els.healthDot.className = "health bad";
      els.healthDot.title = `无法连接 ${window.BcaApi.base}`;
    }
  }

  async function openProjectFolder() {
    els.uploadMsg.textContent = "";
    if (!window.desktop?.openProjectDirDialog) {
      els.uploadMsg.textContent = "需桌面客户端打开文件夹";
      return;
    }
    try {
      els.btnPickDir.disabled = true;
      const picked = await window.desktop.openProjectDirDialog();
      if (!picked) return;
      if (picked.ok === false) {
        els.uploadMsg.textContent = picked.error || "打开失败";
        return;
      }
      const folderName = picked.name || "project";
      const name = els.uploadName.value.trim() || folderName;
      const projectId = els.uploadId.value.trim() || undefined;
      els.uploadMsg.textContent = "正在注册项目…";
      const data = await window.BcaApi.createProject({
        name,
        projectId,
        path: picked.path,
        description: `本机文件夹 ${picked.path}`,
      });
      if (window.desktop?.setProjectLocalRoot) {
        await window.desktop.setProjectLocalRoot(data.id, picked.path);
        projectLocalRoots[data.id] = picked.path;
      }
      await window.desktop.useProjectRoot?.(data.id);
      els.uploadMsg.textContent = `已打开：${data.name}\n${picked.path}`;
      els.uploadName.value = "";
      els.uploadId.value = "";
      setActiveProject(data.id, data.name);
      await loadProjects();
      // Auto-parse like opening a workspace
      els.uploadMsg.textContent = `已打开：${data.name}，开始解析…`;
      try {
        await parseProject(data.id, false);
        els.uploadMsg.textContent = `已打开并开始解析：${data.name}`;
      } catch (err) {
        els.uploadMsg.textContent = `已打开：${data.name}（解析启动失败：${err.message || err}）`;
      }
    } catch (err) {
      els.uploadMsg.textContent = `打开失败：${err.message || err}`;
    } finally {
      els.btnPickDir.disabled = false;
    }
  }

  function normalizeTheme(theme) {
    if (theme === "light" || theme === "white") return "light";
    return "dark";
  }

  function currentTheme() {
    return normalizeTheme(
      document.documentElement.getAttribute("data-theme") ||
        localStorage.getItem(THEME_KEY) ||
        "dark"
    );
  }

  function paintThemeButtons(theme) {
    const t = normalizeTheme(theme);
    els.themePickDark?.classList.toggle("active", t === "dark");
    els.themePickLight?.classList.toggle("active", t === "light");
    if (els.btnToggleTheme) {
      els.btnToggleTheme.textContent = t === "light" ? "白色" : "黑色";
      els.btnToggleTheme.title =
        t === "light" ? "当前白色主题，点击切换为黑色" : "当前黑色主题，点击切换为白色";
    }
  }

  async function applyTheme(theme, { persist = true } = {}) {
    const t = normalizeTheme(theme);
    document.documentElement.setAttribute("data-theme", t);
    localStorage.setItem(THEME_KEY, t);
    paintThemeButtons(t);
    if (persist && window.desktop?.setConfig) {
      try {
        await window.desktop.setConfig({ theme: t });
      } catch {
        /* ignore */
      }
    }
  }

  async function initConfig() {
    let apiBase = "http://127.0.0.1:8000";
    let theme = localStorage.getItem(THEME_KEY) || "dark";
    if (window.desktop?.getConfig) {
      const cfg = await window.desktop.getConfig();
      apiBase = cfg.apiBase || apiBase;
      if (cfg.theme) theme = cfg.theme;
      if (cfg.projectLocalRoots) projectLocalRoots = cfg.projectLocalRoots;
    } else {
      apiBase = localStorage.getItem("bca_desktop_api_base") || apiBase;
    }
    window.BcaApi.setBase(apiBase);
    await applyTheme(theme, { persist: false });
    await refreshProjectLocalRoots();
  }

  // —— Events ——
  els.btnSettings?.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleSettings();
  });
  els.btnCloseSettings?.addEventListener("click", () => closeSettings());
  els.btnCancelSettings?.addEventListener("click", () => closeSettings());
  els.settingsBackdrop?.addEventListener("click", () => closeSettings());
  els.settingsForm?.addEventListener("submit", (e) => {
    void saveSettings(e);
  });
  els.settingsDialog?.addEventListener("click", (e) => e.stopPropagation());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && settingsOpen) closeSettings();
  });

  els.btnNewChat.addEventListener("click", () => {
    const id = newDesktopSessionId();
    activeId = id;
    localStorage.setItem(ACTIVE_KEY, activeId);
    sessions.unshift({
      id,
      title: "新对话",
      updatedAt: Date.now(),
      messages: [],
    });
    // keep list bounded
    sessions = sessions.slice(0, 40);
    saveSessions();
    setView("chat");
    renderAll();
    els.input.focus();
  });

  function paintKbMsg(text, kind) {
    if (!els.kbMsg) return;
    els.kbMsg.textContent = text || "";
    els.kbMsg.dataset.kind = kind || "";
  }

  function renderKbList() {
    if (!els.kbList) return;
    if (!kbDocs.length) {
      els.kbList.innerHTML = `<div class="empty-list">暂无文档，可上传接口说明</div>`;
      return;
    }
    els.kbList.replaceChildren();
    for (const d of kbDocs) {
      const row = document.createElement("div");
      row.className = "side-item kb-item";
      const st = d.status || "";
      const stLabel =
        st === "ready"
          ? `就绪 · ${d.chunk_count || 0} 块`
          : st === "indexing" || st === "queued"
            ? "索引中…"
            : st === "failed"
              ? "失败"
              : st;
      const main = document.createElement("div");
      main.className = "side-item-main kb-item-main";
      main.innerHTML =
        `<span class="t" title="${escapeHtml(d.filename || "")}">${escapeHtml(
          d.title || d.filename || d.id
        )}</span>` +
        `<span class="m ${st === "failed" ? "is-bad" : st === "ready" ? "is-ok" : ""}">${escapeHtml(
          stLabel
        )}</span>`;
      if (d.error) main.title = d.error;
      const del = document.createElement("button");
      del.type = "button";
      del.className = "side-item-del";
      del.title = "删除";
      del.setAttribute("aria-label", "删除文档");
      del.textContent = "×";
      del.addEventListener("click", async () => {
        if (!confirm(`删除知识库文档「${d.title || d.filename}」？`)) return;
        try {
          await window.BcaApi.deleteKnowledge(d.id);
          await loadKnowledge();
          paintKbMsg("已删除", "ok");
        } catch (err) {
          paintKbMsg(err.message || String(err), "bad");
        }
      });
      row.appendChild(main);
      row.appendChild(del);
      els.kbList.appendChild(row);
    }
  }

  async function loadKnowledge() {
    if (!els.kbList) return;
    try {
      const data = await window.BcaApi.listKnowledge();
      kbDocs = data.items || data || [];
      if (!Array.isArray(kbDocs)) kbDocs = [];
      renderKbList();
    } catch (err) {
      kbDocs = [];
      renderKbList();
      paintKbMsg(err.message || "知识库加载失败", "bad");
    }
  }

  async function uploadKbFiles(fileList) {
    const files = [...(fileList || [])];
    if (!files.length) return;
    paintKbMsg(`上传中（${files.length}）…`, "");
    let ok = 0;
    for (const f of files) {
      try {
        await window.BcaApi.uploadKnowledge(f);
        ok += 1;
      } catch (err) {
        paintKbMsg(`${f.name}: ${err.message || err}`, "bad");
      }
    }
    await loadKnowledge();
    if (ok) paintKbMsg(`已入库 ${ok} 个文档`, "ok");
  }

  els.btnRefreshProjects.addEventListener("click", loadProjects);
  els.btnRefreshKb?.addEventListener("click", () => loadKnowledge());
  els.btnUploadKb?.addEventListener("click", () => els.kbFileInput?.click());
  els.kbFileInput?.addEventListener("change", () => {
    const files = els.kbFileInput.files;
    void uploadKbFiles(files);
    els.kbFileInput.value = "";
  });
  els.btnPickDir?.addEventListener("click", openProjectFolder);
  els.btnClosePreview?.addEventListener("click", closePreview);
  els.btnPreviewUnified?.addEventListener("click", () => setPreviewViewMode("unified"));
  els.btnPreviewCompare?.addEventListener("click", () => setPreviewViewMode("compare"));
  els.btnSplitPreview?.addEventListener("click", () => {
    if (!previewFiles.length) return;
    if (previewWindows.length >= 4) {
      previewLayout = previewLayout === "col" ? "row" : "col";
      renderPreviewWindows();
      return;
    }
    const used = new Set(previewWindows.map((w) => w.fileIndex));
    let next = previewFiles.findIndex((_, i) => !used.has(i));
    if (next < 0) next = 0;
    addPreviewWindow(next);
    if (previewWindows.length >= 2 && previewWindows.length % 2 === 0) {
      previewLayout = previewWindows.length > 2 ? "row" : "col";
      renderPreviewWindows();
    }
  });
  bindPreviewResizer();
  restorePreviewWidth();
  syncPreviewViewButtons();

  els.btnStopPatch?.addEventListener("click", () => {
    window.BcaClientRuns?.stop();
  });

  if (window.BcaClientRuns) {
    window.BcaClientRuns.onStatus = paintPatchStatus;
  }

  els.themePickDark?.addEventListener("click", () => applyTheme("dark"));
  els.themePickLight?.addEventListener("click", () => applyTheme("light"));
  els.btnToggleTheme?.addEventListener("click", () => {
    applyTheme(currentTheme() === "light" ? "dark" : "light");
  });

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(els.input.value);
  });

  const focusComposer = () => {
    const input = document.getElementById("messageInput") || els.input;
    if (!input || input.disabled || input.readOnly) return;
    els.input = input;
    try {
      input.focus({ preventScroll: true });
    } catch {
      input.focus();
    }
  };
  // Click padding of the composer box → focus textarea.
  // Do NOT re-focus during pointerdown on the textarea itself (steals caret).
  const onComposerActivate = (e) => {
    if (!e.target || !e.target.closest) return;
    if (e.target.closest("button, a, .send")) return;
    if (e.target === els.input || e.target.id === "messageInput") return;
    e.preventDefault();
    focusComposer();
  };
  els.form.addEventListener("pointerdown", onComposerActivate);
  els.form.addEventListener("click", onComposerActivate);
  // Keep body.resizing-preview from sticking if mouseup was lost.
  window.addEventListener("mouseup", () => {
    document.body.classList.remove("resizing-preview");
  });
  window.addEventListener("blur", () => {
    document.body.classList.remove("resizing-preview");
  });
  window.addEventListener("focus", () => {
    if (document.activeElement === document.body) {
      queueMicrotask(focusComposer);
    }
  });

  els.input.addEventListener("keydown", (e) => {
    if (e.isComposing || e.keyCode === 229) return;
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(els.input.value);
    }
  });
  // Ctrl/Cmd+V paste fallback when OS menu roles are unavailable.
  els.input.addEventListener("paste", () => {
    queueMicrotask(autoResize);
  });
  els.input.addEventListener("input", autoResize);
  autoResize();
  // Initial focus so the first click is not fighting a blank activeElement.
  queueMicrotask(focusComposer);

  document.querySelectorAll(".chips [data-prompt]").forEach((btn) => {
    btn.addEventListener("click", () => sendMessage(btn.getAttribute("data-prompt")));
  });

  // —— Boot ——
  (async () => {
    await initConfig();
    setActiveProjects(activeProjectIds());
    setView("chat");
    renderAll();
    await checkHealth();
    setInterval(checkHealth, 15000);
    // Warm project list + knowledge base in background
    loadProjects().catch(() => {});
    loadKnowledge().catch(() => {});
  })();
})();
