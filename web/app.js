(() => {
  const API_BASE = "";
  const STORAGE_KEY = "bca_chat_sessions_v1";
  const ACTIVE_KEY = "bca_active_session_v1";
  const ACTIVE_PROJECT_KEY = "bca_active_project_v1";

  const urlProject = new URLSearchParams(location.search).get("project_id");
  if (urlProject) {
    localStorage.setItem(ACTIVE_PROJECT_KEY, urlProject);
  }
  const activeProjectId = () => localStorage.getItem(ACTIVE_PROJECT_KEY) || null;

  const els = {
    transcript: document.getElementById("transcript"),
    emptyState: document.getElementById("emptyState"),
    form: document.getElementById("chatForm"),
    input: document.getElementById("messageInput"),
    send: document.getElementById("btnSend"),
    sessionList: document.getElementById("sessionList"),
    chatTitle: document.getElementById("chatTitle"),
    sessionMeta: document.getElementById("sessionMeta"),
    healthStatus: document.getElementById("healthStatus"),
    btnNewChat: document.getElementById("btnNewChat"),
    btnClearLocal: document.getElementById("btnClearLocal"),
    btnSessions: document.getElementById("btnSessions"),
    btnCloseDrawer: document.getElementById("btnCloseDrawer"),
    drawer: document.getElementById("sessionDrawer"),
  };

  /** @type {{id:string|null,title:string,updatedAt:number,messages:Array}[]} */
  let sessions = loadSessions();
  let activeId = localStorage.getItem(ACTIVE_KEY) || null;
  let sending = false;
  /** @type {ReturnType<typeof setInterval>|null} */
  let thinkingTimer = null;

  function loadSessions() {
    try {
      const raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
      // Drop interrupted in-flight bubbles from a prior page load.
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

  function stopThinkingTimer() {
    if (thinkingTimer != null) {
      clearInterval(thinkingTimer);
      thinkingTimer = null;
    }
  }

  function startThinkingTimer(startedAt) {
    stopThinkingTimer();
    const tick = () => {
      const nodes = els.transcript.querySelectorAll(
        ".msg.assistant .timing.thinking"
      );
      if (!nodes.length) return;
      const label = `思考中 · ${formatElapsed(performance.now() - startedAt)}`;
      nodes.forEach((el) => {
        el.textContent = label;
      });
    };
    tick();
    thinkingTimer = setInterval(tick, 200);
  }

  function nowLabel(ts) {
    return new Date(ts).toLocaleString("zh-CN", {
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  }

  function getActive() {
    return sessions.find((s) => s.id === activeId) || null;
  }

  function ensureActive() {
    let s = getActive();
    if (!s) {
      s = {
        id: null,
        title: "新对话",
        updatedAt: Date.now(),
        messages: [],
      };
      sessions.unshift(s);
      activeId = null;
      localStorage.removeItem(ACTIVE_KEY);
      saveSessions();
    }
    return s;
  }

  function setActive(id) {
    activeId = id;
    if (id) localStorage.setItem(ACTIVE_KEY, id);
    else localStorage.removeItem(ACTIVE_KEY);
    closeDrawer();
    renderAll();
  }

  function upsertSession(sessionId, title, messages) {
    let s = sessions.find((x) => x.id === sessionId);
    if (!s) {
      const draft = sessions.find((x) => x.id === null);
      if (draft) {
        s = draft;
        s.id = sessionId;
      } else {
        s = { id: sessionId, title: title || "对话", updatedAt: Date.now(), messages: [] };
        sessions.unshift(s);
      }
    }
    if (title) s.title = title;
    if (messages) s.messages = messages;
    s.updatedAt = Date.now();
    sessions = [s, ...sessions.filter((x) => x !== s)];
    activeId = sessionId;
    localStorage.setItem(ACTIVE_KEY, sessionId);
    saveSessions();
  }

  function autoResize() {
    const el = els.input;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 140) + "px";
  }

  function openDrawer() {
    els.drawer.hidden = false;
  }

  function closeDrawer() {
    els.drawer.hidden = true;
  }

  async function removeSession(target) {
    const key = target?.id ?? null;
    const idx = sessions.findIndex((s) => s === target || (key != null && s.id === key));
    if (idx < 0) return;
    const wasActive =
      sessions[idx].id === activeId || (!sessions[idx].id && !activeId);
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
        await fetch(`${API_BASE}/chat/${encodeURIComponent(serverId)}`, {
          method: "DELETE",
        });
      } catch {
        /* local list already cleared */
      }
    }
  }

  function renderSessions() {
    els.sessionList.innerHTML = "";
    const sorted = [...sessions].sort((a, b) => b.updatedAt - a.updatedAt);
    if (!sorted.length) {
      els.sessionList.innerHTML = `<div class="session-empty">暂无会话</div>`;
      return;
    }
    for (const s of sorted) {
      const row = document.createElement("div");
      row.className =
        "session-item" + (s.id === activeId || (!s.id && !activeId) ? " active" : "");

      const main = document.createElement("button");
      main.type = "button";
      main.className = "session-main";
      main.innerHTML = `
        <div class="title">${escapeHtml(s.title || "新对话")}</div>
        <div class="time">${nowLabel(s.updatedAt)}</div>
      `;
      main.addEventListener("click", () => setActive(s.id));

      const del = document.createElement("button");
      del.type = "button";
      del.className = "session-del";
      del.setAttribute("aria-label", "删除会话");
      del.title = "删除";
      del.innerHTML =
        '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
        '<path d="M9 4h6m-8 4h10m-1.5 0-.7 11.2a1.5 1.5 0 0 1-1.5 1.4H9.7a1.5 1.5 0 0 1-1.5-1.4L7.5 8M10 11v6m4-6v6" ' +
        'stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>';
      del.addEventListener("click", (e) => {
        e.stopPropagation();
        const label = (s.title || "新对话").slice(0, 24);
        if (!confirm(`删除会话「${label}」？`)) return;
        removeSession(s);
      });

      row.appendChild(main);
      row.appendChild(del);
      els.sessionList.appendChild(row);
    }
  }

  function renderTranscript() {
    const s = getActive();
    const messages = s?.messages || [];
    els.chatTitle.textContent = s?.title || "新对话";
    els.sessionMeta.textContent = s?.id
      ? `连续对话 · ${s.id.slice(0, 10)}…`
      : "连续多轮，上下文自动保持";

    const empty = els.emptyState;
    els.transcript.replaceChildren();
    if (!messages.length) {
      empty.hidden = false;
      els.transcript.appendChild(empty);
      return;
    }
    empty.hidden = true;
    for (const m of messages) {
      els.transcript.appendChild(renderMessage(m));
    }
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function formatElapsed(ms) {
    if (ms == null || Number.isNaN(Number(ms))) return "";
    const n = Math.max(0, Number(ms));
    if (n < 1000) return `${Math.round(n)} ms`;
    if (n < 60_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)} s`;
    const m = Math.floor(n / 60_000);
    const s = Math.round((n % 60_000) / 1000);
    return `${m} 分 ${s} s`;
  }

  function messageElapsedMs(m) {
    if (m == null) return null;
    if (typeof m.elapsed_ms === "number" && m.elapsed_ms >= 0) return m.elapsed_ms;
    const metaMs = m.meta?.elapsed_ms;
    if (typeof metaMs === "number" && metaMs >= 0) return metaMs;
    return null;
  }

  function renderMessage(m) {
    const wrap = document.createElement("article");
    wrap.className = `msg ${m.role}`;
    const role = document.createElement("div");
    role.className = "role";
    const roleLabel = document.createElement("span");
    roleLabel.textContent = m.role === "user" ? "你" : "Agent";
    role.appendChild(roleLabel);
    if (m.role === "assistant" && m.pending) {
      const timing = document.createElement("span");
      timing.className = "timing thinking";
      timing.title = "正在生成回复";
      const started = typeof m.thinkingStartedAt === "number"
        ? m.thinkingStartedAt
        : performance.now();
      timing.textContent = `思考中 · ${formatElapsed(performance.now() - started)}`;
      role.appendChild(timing);
    } else if (m.role === "assistant") {
      const elapsed = messageElapsedMs(m);
      if (elapsed != null) {
        const timing = document.createElement("span");
        timing.className = "timing";
        timing.title = "从接到请求到生成回复的耗时";
        timing.textContent = `耗时 ${formatElapsed(elapsed)}`;
        role.appendChild(timing);
      }
    }
    const bubble = document.createElement("div");
    bubble.className = "bubble" + (m.error ? " bubble-error" : "");
    if (m.pending) {
      bubble.classList.add("bubble-thinking");
      bubble.appendChild(renderThinkingBubble(m));
    } else if (m.error || String(m.content || "").startsWith("请求失败")) {
      if (!m.error) {
        m.error = true;
        // Best-effort: previous user turn is the retry payload for old local messages.
        if (!m.retryText) {
          const s = getActive();
          const idx = s?.messages?.indexOf(m) ?? -1;
          for (let i = idx - 1; i >= 0; i--) {
            if (s.messages[i].role === "user") {
              m.retryText = s.messages[i].content;
              break;
            }
          }
        }
      }
      bubble.classList.add("bubble-error");
      const row = document.createElement("div");
      row.className = "error-row";
      const text = document.createElement("span");
      text.className = "error-text";
      text.textContent = m.content || "请求失败";
      const retry = document.createElement("button");
      retry.type = "button";
      retry.className = "retry-btn";
      retry.title = "重试";
      retry.setAttribute("aria-label", "重试");
      retry.disabled = !m.retryText;
      retry.innerHTML =
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" aria-hidden="true">' +
        '<path d="M20 12a8 8 0 1 1-2.3-5.5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>' +
        '<path d="M20 5v5h-5" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>' +
        "</svg>";
      retry.addEventListener("click", () => retryFailed(m));
      row.appendChild(text);
      row.appendChild(retry);
      bubble.appendChild(row);
    } else {
      fillBubbleContent(bubble, m.content || "", m.role === "assistant");
    }
    wrap.appendChild(role);
    wrap.appendChild(bubble);

    if (
      m.role === "assistant" &&
      !m.pending &&
      !m.error &&
      !String(m.content || "").startsWith("请求失败")
    ) {
      wrap.appendChild(renderFeedbackBar(m));
    }

    const metaEl = renderMessageMeta(m.meta);
    if (metaEl) wrap.appendChild(metaEl);
    return wrap;
  }

  function usefulList(v) {
    return Array.isArray(v) && v.length > 0;
  }

  function renderThinkingBubble(m) {
    const root = document.createElement("div");
    root.className = "thinking-panel";

    const row = document.createElement("div");
    row.className = "thinking-row";
    row.innerHTML =
      `<span class="typing" aria-hidden="true"><i></i><i></i><i></i></span>` +
      `<span class="thinking-label">思考中…</span>`;
    root.appendChild(row);

    const plan = m.plan;
    if (plan && Array.isArray(plan.steps) && plan.steps.length) {
      const head = document.createElement("div");
      head.className = "plan-head";
      head.textContent = plan.title || "研究计划";
      root.appendChild(head);

      const list = document.createElement("ol");
      list.className = "plan-steps";
      for (const step of plan.steps) {
        const li = document.createElement("li");
        const status = String(step.status || "pending");
        li.className = `plan-step is-${status}`;
        const mark = document.createElement("span");
        mark.className = "plan-mark";
        mark.setAttribute("aria-hidden", "true");
        mark.textContent =
          status === "done"
            ? "✓"
            : status === "running"
              ? "…"
              : status === "error"
                ? "!"
                : "○";
        const body = document.createElement("div");
        body.className = "plan-step-body";
        const title = document.createElement("div");
        title.className = "plan-step-title";
        title.textContent = step.title || step.id || "步骤";
        body.appendChild(title);
        const note = step.note || (status === "running" ? step.detail : "");
        if (note) {
          const noteEl = document.createElement("div");
          noteEl.className = "plan-step-note";
          noteEl.textContent = note;
          body.appendChild(noteEl);
        }
        li.appendChild(mark);
        li.appendChild(body);
        list.appendChild(li);
      }
      root.appendChild(list);
    }
    if (m.reviewNote) {
      const rev = document.createElement("div");
      // Treat refine/reject as quiet thinking, not a loud failure banner.
      rev.className =
        "review-note" +
        (m.reviewStatus === "rejected" ? " is-refining" : "");
      rev.textContent = m.reviewNote;
      root.appendChild(rev);
    }
    return root;
  }

  function patchPendingPlan(session, plan) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending || !plan) return;
    pending.plan = plan;
    // Prefer in-place DOM update to avoid full transcript flicker.
    const thinking = els.transcript.querySelector(".msg.assistant .bubble.bubble-thinking");
    if (thinking) {
      thinking.replaceChildren(renderThinkingBubble(pending));
      els.transcript.scrollTop = els.transcript.scrollHeight;
      return;
    }
    renderAll();
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  function patchPendingReview(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    const status = ev.status || "running";
    pending.reviewStatus = status;
    if (status === "running") {
      pending.reviewNote = ev.note || "核对回答细节…";
    } else if (status === "rejected") {
      const gaps = (ev.review && ev.review.gaps) || [];
      const gapHint = gaps.length
        ? "（补全" + gaps.slice(0, 2).join("；") + "）"
        : "";
      pending.reviewNote =
        "补充更多细节：" +
        (ev.note || "再查一层实现") +
        gapHint +
        "，继续完善…";
    } else {
      pending.reviewNote = "细节已较完整：" + (ev.note || "可以作答");
    }
    const thinking = els.transcript.querySelector(".msg.assistant .bubble.bubble-thinking");
    if (thinking) {
      thinking.replaceChildren(renderThinkingBubble(pending));
      els.transcript.scrollTop = els.transcript.scrollHeight;
    }
  }

  function addTraceEntry(pending, ev) {
    if (!pending) return;
    if (!Array.isArray(pending.trace)) pending.trace = [];
    const note =
      String(ev.note || ev.title || ev.message || "").trim() ||
      (ev.intent ? `意图识别 → ${ev.intent}` : "") ||
      String(ev.type || ev.step_id || "").trim();
    if (!note) return;
    const last = pending.trace[pending.trace.length - 1];
    const sid = ev.step_id || "";
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
  }

  function patchPendingStep(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    if (ev.plan) {
      pending.plan = ev.plan;
    } else {
      if (!pending.plan) {
        pending.plan = {
          kind: "pev",
          title: "[PEV] 规划器–执行器–校验器",
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
    addTraceEntry(pending, ev);
    refreshPendingBubble(session);
  }

  function patchPendingIntent(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    const intent = ev.intent || "";
    const note =
      ev.note ||
      (intent
        ? `意图识别 → ${intent}${ev.want_apply ? "（将要动手改代码）" : ""}`
        : "意图识别完成");
    addTraceEntry(pending, {
      type: "intent",
      status: "done",
      step_id: "intent",
      note,
      detail: ev.detail || "",
      intent,
    });
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
    if (ev.note) addTraceEntry(pending, { type: "log", status: "done", note: ev.note });
    refreshPendingBubble(session);
  }

  function patchPendingReview(session, ev) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    const status = ev.status || "running";
    if (status === "running") {
      pending.reviewNote = ev.note || "核对回答细节…";
    } else if (status === "rejected") {
      pending.reviewNote = "补充更多细节后继续完善…";
    } else {
      pending.reviewNote = "细节已比较完整：" + (ev.note || "可以作答");
    }
    addTraceEntry(pending, {
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
    if (!pending.currentNote) {
      pending.currentNote = "仍在处理中，请稍候…";
      refreshPendingBubble(session);
    }
  }

  function refreshPendingBubble(session) {
    const pending = [...(session.messages || [])].reverse().find((m) => m.pending);
    if (!pending) return;
    const thinking = els.transcript.querySelector(".msg.assistant .bubble.bubble-thinking");
    if (thinking) {
      thinking.replaceChildren(renderThinkingBubble(pending));
      els.transcript.scrollTop = els.transcript.scrollHeight;
      return;
    }
    renderAll();
    els.transcript.scrollTop = els.transcript.scrollHeight;
  }

  async function consumeChatStream(res, onEvent) {
    if (!res.body || !res.body.getReader) {
      throw new Error("浏览器不支持流式响应");
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let sep;
      while ((sep = buf.indexOf("\n\n")) >= 0) {
        const chunk = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const raw = line.slice(5).trim();
          if (!raw) continue;
          try {
            onEvent(JSON.parse(raw));
          } catch {
            /* ignore malformed */
          }
        }
      }
    }
  }

  function isFlowLikeIntent(intent) {
    const i = String(intent || "").toLowerCase();
    // ask / plan (+ legacy impact/options) read-only Q&A
    return (
      !i ||
      i === "ask" ||
      i === "plan" ||
      i === "impact" ||
      i === "options" ||
      i === "meta" ||
      i === "ask_flow" ||
      i === "general"
    );
  }

  function isModifyLikeIntent(intent) {
    const i = String(intent || "").toLowerCase();
    return (
      i === "modify" ||
      i === "execute" ||
      i === "implement" ||
      i === "verify" ||
      i === "repair" ||
      i === "modify_field" ||
      i === "new_api"
    );
  }

  function renderOptionsCards(options, meta) {
    if (!usefulList(options)) return null;
    const wrap = document.createElement("div");
    wrap.className = "option-cards";
    const phase = String(meta?.phase || "");
    for (const opt of options) {
      if (!opt || !opt.id) continue;
      const card = document.createElement("div");
      card.className = "option-card";
      const title = document.createElement("div");
      title.className = "option-card-title";
      title.textContent = `方案 ${opt.id}：${opt.title || ""}`;
      card.appendChild(title);
      if (opt.summary) {
        const sum = document.createElement("div");
        sum.className = "option-card-summary";
        sum.textContent = String(opt.summary);
        card.appendChild(sum);
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
      btn.textContent =
        meta?.selected_option_id === opt.id ? `已选 ${opt.id}` : `选择方案 ${opt.id}`;
      btn.disabled =
        !!sending ||
        meta?.selected_option_id === opt.id ||
        (phase && phase !== "awaiting_option");
      btn.addEventListener("click", () => {
        sendMessage(`选方案${opt.id}`, {
          action: "select_option",
          option_id: String(opt.id),
        });
      });
      card.appendChild(btn);
      wrap.appendChild(card);
    }
    return wrap;
  }

  /** Footer meta under agent reply. Flow answers: optional collapsed plan. Modify: disclosures. */
  function renderMessageMeta(meta) {
    if (!meta) return null;

    const plan = meta.plan || meta.research_plan;
    const showPlan =
      plan && Array.isArray(plan.steps) && plan.steps.length > 0 && isFlowLikeIntent(meta.intent);
    const optionsOk = usefulList(meta.options);
    const scopeOk = usefulList(meta.scope);
    const patchesOk = usefulList(meta.patches);
    const verifyOk = meta.verify_report && typeof meta.verify_report === "object";

    if (isFlowLikeIntent(meta.intent) && !showPlan && !optionsOk && !verifyOk) return null;
    if (!showPlan && !scopeOk && !patchesOk && !optionsOk && !verifyOk && !isModifyLikeIntent(meta.intent)) {
      return null;
    }

    const box = document.createElement("div");
    box.className = "details";
    if (showPlan) {
      const det = document.createElement("details");
      det.className = "plan-details";
      const sum = document.createElement("summary");
      sum.textContent = "研究计划";
      det.appendChild(sum);
      const ol = document.createElement("ol");
      ol.className = "plan-steps plan-steps-static";
      for (const step of plan.steps) {
        const li = document.createElement("li");
        li.className = `plan-step is-${step.status || "done"}`;
        li.textContent = `${step.title || step.id}${step.note ? " — " + step.note : ""}`;
        ol.appendChild(li);
      }
      det.appendChild(ol);
      box.appendChild(det);
    }
    if (optionsOk) {
      const cards = renderOptionsCards(meta.options, meta);
      if (cards) box.appendChild(cards);
    }
    if ((isModifyLikeIntent(meta.intent) || meta.phase) && meta.intent) {
      const bits = [`意图：${meta.intent}`];
      if (meta.phase) bits.push(`阶段：${meta.phase}`);
      if (meta.selected_option_id) bits.push(`方案：${meta.selected_option_id}`);
      box.appendChild(detailBlock("状态", bits.join(" · ")));
    }
    if (scopeOk) {
      box.appendChild(detailBlock("修改范围", JSON.stringify(meta.scope, null, 2)));
    }
    if (patchesOk) {
      box.appendChild(detailBlock("补丁", JSON.stringify(meta.patches, null, 2)));
    }
    if (verifyOk) {
      const vr = meta.verify_report;
      box.appendChild(
        detailBlock(
          "验证",
          `compile=${vr.compile_ok ? "ok" : "fail"} test=${
            vr.test_skipped ? "skipped" : vr.test_ok ? "ok" : "fail"
          }\n${vr.logs || ""}`.slice(0, 4000)
        )
      );
    }
    return box;
  }

  function messageVote(m) {
    if (m?.vote === "up" || m?.vote === "down") return m.vote;
    const v = m?.meta?.vote;
    return v === "up" || v === "down" ? v : null;
  }

  function thumbSvg(kind) {
    // SF-style outline thumbs; down rotates the path inside a <g> (not the SVG root).
    const path =
      '<path d="M7 11v9a1 1 0 0 1-1 1H4.5A1.5 1.5 0 0 1 3 19.5v-7A1.5 1.5 0 0 1 4.5 11H7zm0 0' +
      "c0-.8.2-1.6.6-2.3L11.2 3a1.8 1.8 0 0 1 3.3 1.1V8h4.2a2 2 0 0 1 2 2.3l-1.2 7" +
      'A2.5 2.5 0 0 1 17 19.5h-6.2A2.8 2.8 0 0 1 8 17.2V11z" ' +
      'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>';
    const inner =
      kind === "down" ? `<g transform="rotate(180 12 12)">${path}</g>` : path;
    return (
      `<svg width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">` +
      inner +
      "</svg>"
    );
  }

  function renderFeedbackBar(m) {
    const bar = document.createElement("div");
    bar.className = "feedback-bar";
    bar.setAttribute("role", "group");
    bar.setAttribute("aria-label", "评价回复");

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
      btn.addEventListener("click", () => submitFeedback(m, vote, bar));
      return btn;
    };

    bar.appendChild(mkBtn("up", "有用"));
    bar.appendChild(mkBtn("down", "不佳"));
    return bar;
  }

  async function submitFeedback(m, vote, barEl) {
    if (!m || m._feedbackBusy) return;
    const prev = messageVote(m);
    if (prev === vote) return; // ignore same-vote spam

    const session = getActive();
    const sessionId = session?.id;
    const messageId = m.id ?? m.assistant_message_id ?? m.meta?.assistant_message_id;
    if (!sessionId || messageId == null) {
      // Optimistic local-only if server id missing (should be rare after chat response).
      m.vote = vote;
      m.meta = { ...(m.meta || {}), vote };
      saveSessions();
      paintFeedbackBar(barEl, vote);
      return;
    }

    m._feedbackBusy = true;
    paintFeedbackBar(barEl, vote, true);
    try {
      const res = await fetch(`${API_BASE}/api/v1/chat/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          assistant_message_id: Number(messageId),
          vote,
          project_id: activeProjectId(),
        }),
      });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }
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

  function paintFeedbackBar(barEl, vote, busy) {
    if (!barEl) return;
    const buttons = barEl.querySelectorAll(".feedback-btn");
    buttons.forEach((btn) => {
      const isUp = !btn.classList.contains("is-down");
      const kind = isUp ? "up" : "down";
      const on = vote === kind;
      btn.classList.toggle("is-selected", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      btn.disabled = !!busy;
    });
  }

  function retryFailed(m) {
    const session = getActive();
    if (!session || sending) return;
    const retryText = (m.retryText || "").trim();
    if (!retryText) return;
    // Drop this error bubble (and matching trailing user turn) before resend.
    if (session.messages[session.messages.length - 1] === m) {
      session.messages.pop();
    } else {
      const i = session.messages.lastIndexOf(m);
      if (i >= 0) session.messages.splice(i, 1);
    }
    const last = session.messages[session.messages.length - 1];
    if (last?.role === "user" && last.content === retryText) {
      session.messages.pop();
    }
    saveSessions();
    sendMessage(retryText);
  }

  function detailBlock(title, body) {
    const d = document.createElement("details");
    const s = document.createElement("summary");
    s.textContent = title;
    const pre = document.createElement("pre");
    pre.textContent = body;
    d.appendChild(s);
    d.appendChild(pre);
    return d;
  }

  function renderAll() {
    renderSessions();
    renderTranscript();
  }

  function escapeHtml(str) {
    return String(str)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  /** Preserve newlines; give ## sections visual spacing so change-plan answers scan cleanly. */
  function fillBubbleContent(bubble, content, asAssistant) {
    const raw = String(content || "").replace(/\r\n/g, "\n");
    if (
      asAssistant &&
      /^##\s+\S/m.test(raw)
    ) {
      bubble.classList.add("bubble-md");
      bubble.innerHTML = raw
        .split("\n")
        .map((line) => {
          const h = line.match(/^##\s+(.+)$/);
          if (h) {
            return `<div class="ans-h">${escapeHtml(h[1])}</div>`;
          }
          if (line.trim() === "") {
            return `<div class="ans-gap" aria-hidden="true"></div>`;
          }
          return `<div class="ans-line">${escapeHtml(line)}</div>`;
        })
        .join("");
      return;
    }
    bubble.textContent = raw;
  }

  async function checkHealth() {
    try {
      const res = await fetch(`${API_BASE}/api/v1/health`);
      const data = await res.json();
      const ok = data.status === "ok" || data.status === "degraded";
      els.healthStatus.textContent = ok ? data.llm_provider : "异常";
      els.healthStatus.className = "pill " + (ok ? "ok" : "bad");
    } catch {
      els.healthStatus.textContent = "离线";
      els.healthStatus.className = "pill bad";
    }
  }

  async function sendMessage(text, extras) {
    const content = (text || "").trim();
    if (!content || sending) return;
    sending = true;
    els.send.disabled = true;

    const session = ensureActive();
    const title = session.title === "新对话" ? content.slice(0, 28) : session.title;
    session.title = title;
    const t0 = performance.now();
    session.messages.push({ role: "user", content });
    session.messages.push({
      role: "assistant",
      content: "",
      pending: true,
      thinkingStartedAt: t0,
    });
    session.updatedAt = Date.now();
    saveSessions();
    renderAll();
    startThinkingTimer(t0);
    els.input.value = "";
    autoResize();

    try {
      const body = {
        message: content,
        session_id: session.id,
        project_id: activeProjectId(),
      };
      if (extras && extras.action) body.action = extras.action;
      if (extras && extras.option_id) body.option_id = extras.option_id;
      let data = null;

      async function postChatBlocking() {
        const res2 = await fetch(`${API_BASE}/api/v1/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        if (!res2.ok) {
          const errText = await res2.text();
          throw new Error(errText || `HTTP ${res2.status}`);
        }
        return res2.json();
      }

      // Prefer SSE for live plan/progress; fall back to blocking /chat on any stream failure.
      try {
        const res = await fetch(`${API_BASE}/api/v1/chat/stream`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Accept: "text/event-stream",
          },
          body: JSON.stringify(body),
        });
        const ct = res.headers.get("content-type") || "";
        if (res.ok && ct.includes("text/event-stream")) {
          let streamError = null;
          await consumeChatStream(res, (ev) => {
            if (!ev || !ev.type) return;
            if (ev.type === "plan" && ev.plan) {
              patchPendingPlan(session, ev.plan);
            } else if (ev.type === "step") {
              if (ev.plan) patchPendingPlan(session, ev.plan);
              patchPendingStep(session, ev);
            } else if (ev.type === "intent") {
              patchPendingIntent(session, ev);
            } else if (ev.type === "log") {
              patchPendingLog(session, ev);
            } else if (ev.type === "review") {
              patchPendingReview(session, ev);
            } else if (ev.type === "heartbeat") {
              patchPendingHeartbeat(session);
            } else if (ev.type === "final") {
              data = ev;
            } else if (ev.type === "error") {
              streamError = ev.message || "流式处理失败";
            }
          });
          if (streamError) throw new Error(streamError);
          if (!data) throw new Error("流式响应未返回最终结果");
        } else if (res.status === 404) {
          data = await postChatBlocking();
        } else {
          const errText = await res.text();
          throw new Error(errText || `HTTP ${res.status}`);
        }
      } catch (streamErr) {
        // Network / embedded-browser SSE issues → blocking chat still answers.
        if (!data) {
          data = await postChatBlocking();
        }
      }
      const clientMs = Math.round(performance.now() - t0);
      const elapsedMs =
        typeof data.elapsed_ms === "number" && data.elapsed_ms >= 0
          ? data.elapsed_ms
          : clientMs;
      const messages = session.messages.filter((m) => !m.pending);
      const assistantMessageId =
        typeof data.assistant_message_id === "number"
          ? data.assistant_message_id
          : null;
      messages.push({
        role: "assistant",
        id: assistantMessageId,
        content: data.reply || "(空回复)",
        elapsed_ms: elapsedMs,
        meta: {
          intent: data.intent,
          scope: data.scope,
          patches: data.patches,
          plan: data.plan || null,
          review: data.review || null,
          elapsed_ms: elapsedMs,
          assistant_message_id: assistantMessageId,
          phase: data.phase || null,
          options: data.options || [],
          selected_option_id: data.selected_option_id || null,
          verify_report: data.verify_report || null,
        },
      });
      upsertSession(data.session_id, title, messages);
      renderAll();
    } catch (err) {
      const clientMs = Math.round(performance.now() - t0);
      session.messages = session.messages.filter((m) => !m.pending);
      let msg = err.message || String(err);
      // Keep bubble short for HTML error pages
      if (msg.includes("Internal Server Error")) msg = "Internal Server Error";
      else if (/failed to fetch|networkerror|load failed/i.test(msg)) {
        msg =
          "无法连接后端（Failed to fetch）。请确认已用 http://127.0.0.1:8000 打开页面，且 uvicorn 正在运行；然后点重试。";
      } else if (msg.length > 180) msg = msg.slice(0, 180) + "…";
      session.messages.push({
        role: "assistant",
        content: `请求失败：${msg}`,
        error: true,
        retryText: content,
        elapsed_ms: clientMs,
      });
      saveSessions();
      renderAll();
    } finally {
      stopThinkingTimer();
      sending = false;
      els.send.disabled = false;
      els.input.focus();
    }
  }

  els.form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(els.input.value);
  });

  els.input.addEventListener("input", autoResize);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(els.input.value);
    }
  });

  els.btnNewChat.addEventListener("click", () => {
    sessions.unshift({
      id: null,
      title: "新对话",
      updatedAt: Date.now(),
      messages: [],
    });
    const drafts = sessions.filter((s) => s.id === null);
    if (drafts.length > 1) {
      sessions = [drafts[0], ...sessions.filter((s) => s.id !== null)];
    }
    activeId = null;
    localStorage.removeItem(ACTIVE_KEY);
    saveSessions();
    closeDrawer();
    renderAll();
    els.input.focus();
  });

  els.btnSessions.addEventListener("click", openDrawer);
  els.btnCloseDrawer.addEventListener("click", closeDrawer);
  els.drawer.addEventListener("click", (e) => {
    if (e.target === els.drawer) closeDrawer();
  });

  els.btnClearLocal.addEventListener("click", () => {
    if (!confirm("清除本浏览器中的会话列表？")) return;
    sessions = [];
    activeId = null;
    localStorage.removeItem(STORAGE_KEY);
    localStorage.removeItem(ACTIVE_KEY);
    renderAll();
  });

  els.emptyState.addEventListener("click", (e) => {
    const btn = e.target.closest("[data-prompt]");
    if (!btn) return;
    sendMessage(btn.getAttribute("data-prompt"));
  });

  if (activeId && !sessions.some((s) => s.id === activeId)) {
    activeId = sessions[0]?.id || null;
  }

  renderAll();
  checkHealth();
  autoResize();
  els.input.focus();
})();
