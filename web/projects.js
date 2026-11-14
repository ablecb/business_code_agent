(() => {
  const ACTIVE_PROJECT_KEY = "bca_active_project_v1";
  const els = {
    list: document.getElementById("projectList"),
    form: document.getElementById("uploadForm"),
    zip: document.getElementById("zipFile"),
    fileLabel: document.getElementById("fileLabel"),
    name: document.getElementById("projectName"),
    id: document.getElementById("projectId"),
    uploadMsg: document.getElementById("uploadMsg"),
    btnUpload: document.getElementById("btnUpload"),
    btnRefresh: document.getElementById("btnRefresh"),
    health: document.getElementById("healthStatus"),
  };

  let pollTimer = null;
  let loadSeq = 0;
  let loadCtrl = null;

  const statusText = {
    ready: "待解析",
    parsing: "解析中",
    parsed: "已解析",
    failed: "失败",
    empty: "无 Java 源码",
  };

  function escapeHtml(str) {
    return String(str ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/v1/health");
      const data = await res.json();
      const ok = data.status === "ok" || data.status === "degraded";
      els.health.textContent = ok ? data.llm_provider : "异常";
      els.health.className = "pill " + (ok ? "ok" : "bad");
    } catch {
      els.health.textContent = "离线";
      els.health.className = "pill bad";
    }
  }

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
    done: "完成",
    cancelled: "已取消",
  };

  function progressPct(job) {
    if (!job || !job.total_files) return 0;
    return Math.min(100, Math.round((job.processed / job.total_files) * 100));
  }

  function stageLabel(stage) {
    if (!stage) return "";
    return stageText[stage] || stage;
  }

  function renderProjects(items) {
    if (!items.length) {
      els.list.innerHTML = `<div class="empty-list">暂无项目，请先上传 zip。</div>`;
      return;
    }

    els.list.innerHTML = items
      .map((p) => {
        const st = p.status || "ready";
        const job = p.latest_job;
        const pct = progressPct(job);
        const busy = st === "parsing";
        const stage = job?.stage ? ` · ${stageLabel(job.stage)}` : "";
        const msg = job?.message || p.description || p.root_path;
        const err = job?.error
          ? `<div class="proj-error">${escapeHtml(String(job.error).slice(0, 240))}</div>`
          : "";
        return `
          <article class="proj-card" data-id="${escapeHtml(p.id)}">
            <div class="proj-main">
              <div class="proj-title-row">
                <h3>${escapeHtml(p.name)}</h3>
                <span class="status-chip status-${escapeHtml(st)}">${statusText[st] || st}${stage}</span>
              </div>
              <p class="proj-id">${escapeHtml(p.id)}</p>
              <p class="proj-meta">
                Java 文件 ${p.java_files ?? 0} · 符号 ${p.symbol_count ?? 0}
              </p>
              <p class="proj-msg">${escapeHtml(msg || "")}</p>
              ${
                busy
                  ? `<div class="progress"><i style="width:${pct}%"></i></div>
                     <p class="proj-meta">${pct}% · ${job?.processed || 0}/${job?.total_files || 0}</p>`
                  : ""
              }
              ${err}
            </div>
            <div class="proj-actions">
              <button type="button" class="btn-blue btn-parse" data-id="${escapeHtml(p.id)}" data-force="${
                busy ? "1" : "0"
              }">
                ${busy ? "强制重新解析" : st === "parsed" ? "重新解析" : "解析"}
              </button>
              <a class="btn-ghost-solid" href="/?project_id=${encodeURIComponent(p.id)}">去对话</a>
            </div>
          </article>
        `;
      })
      .join("");

    els.list.querySelectorAll(".btn-parse").forEach((btn) => {
      btn.addEventListener("click", () =>
        parseProject(btn.getAttribute("data-id"), btn.getAttribute("data-force") === "1")
      );
    });
    els.list.querySelectorAll(".proj-card").forEach((card) => {
      card.addEventListener("click", (e) => {
        if (e.target.closest("button,a")) return;
        localStorage.setItem(ACTIVE_PROJECT_KEY, card.getAttribute("data-id"));
      });
    });

    const anyParsing = items.some((p) => p.status === "parsing");
    if (anyParsing) startPoll();
    else stopPoll();
  }

  async function loadProjects() {
    const seq = ++loadSeq;
    if (loadCtrl) loadCtrl.abort();
    loadCtrl = new AbortController();
    const slowTimer = setTimeout(() => {
      if (seq !== loadSeq) return;
      if (!els.list.querySelector(".proj-card")) {
        els.list.innerHTML = `<div class="empty-list">列表加载较慢（解析占用数据库时常见），仍在等待…</div>`;
      }
    }, 3000);

    try {
      const res = await fetch("/api/v1/projects", {
        signal: loadCtrl.signal,
        cache: "no-store",
      });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      if (seq !== loadSeq) return;
      renderProjects(data);
    } catch (err) {
      if (err?.name === "AbortError") return;
      if (els.list.querySelector(".proj-card")) {
        console.warn("refresh projects failed", err);
        return;
      }
      els.list.innerHTML = `<div class="empty-list">加载失败：${escapeHtml(err.message || err)}</div>`;
    } finally {
      clearTimeout(slowTimer);
    }
  }

  async function parseProject(projectId, force = false) {
    try {
      if (force && !confirm("将取消当前卡住的任务并重新解析，是否继续？")) {
        return;
      }
      const res = await fetch(`/api/v1/projects/${encodeURIComponent(projectId)}/parse`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ full_refresh: true, tagging: true, force }),
      });
      if (!res.ok) throw new Error(await res.text());
      localStorage.setItem(ACTIVE_PROJECT_KEY, projectId);
      await loadProjects();
      startPoll();
    } catch (err) {
      alert(`解析启动失败：${err.message || err}`);
    }
  }

  function startPoll() {
    if (pollTimer) return;
    pollTimer = setInterval(loadProjects, 2500);
  }

  function stopPoll() {
    if (!pollTimer) return;
    clearInterval(pollTimer);
    pollTimer = null;
  }

  els.zip.addEventListener("change", () => {
    const f = els.zip.files?.[0];
    els.fileLabel.textContent = f ? f.name : "选择 .zip 文件";
    if (f && !els.name.value) {
      els.name.value = f.name.replace(/\.zip$/i, "");
    }
  });

  els.form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = els.zip.files?.[0];
    if (!file) {
      els.uploadMsg.textContent = "请先选择 zip 文件";
      return;
    }
    els.btnUpload.disabled = true;
    els.uploadMsg.textContent = "上传中…";
    try {
      const fd = new FormData();
      fd.append("file", file);
      if (els.name.value.trim()) fd.append("name", els.name.value.trim());
      if (els.id.value.trim()) fd.append("project_id", els.id.value.trim());
      const res = await fetch("/api/v1/projects/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      els.uploadMsg.textContent = `已上传：${data.name}`;
      els.form.reset();
      els.fileLabel.textContent = "选择 .zip 文件";
      localStorage.setItem(ACTIVE_PROJECT_KEY, data.id);
      await loadProjects();
    } catch (err) {
      els.uploadMsg.textContent = `上传失败：${err.message || err}`;
    } finally {
      els.btnUpload.disabled = false;
    }
  });

  els.btnRefresh.addEventListener("click", loadProjects);

  checkHealth();
  loadProjects();
})();
