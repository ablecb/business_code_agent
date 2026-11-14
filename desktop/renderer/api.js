(() => {
  /** Thin API client; base URL comes from desktop config / settings. */
  const Api = {
    base: "http://127.0.0.1:8000",

    setBase(url) {
      const trimmed = String(url || "").trim().replace(/\/+$/, "");
      this.base = trimmed || "http://127.0.0.1:8000";
    },

    url(path) {
      return `${this.base}${path.startsWith("/") ? path : `/${path}`}`;
    },

    async health() {
      const res = await fetch(this.url("/api/v1/health"), { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return res.json();
    },

    async listProjects() {
      const res = await fetch(this.url("/api/v1/projects"), { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async createProject({ name, projectId, path, description }) {
      const res = await fetch(this.url("/api/v1/projects"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          project_id: projectId || undefined,
          path,
          description: description || undefined,
        }),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async parseProject(projectId, { force = false } = {}) {
      const res = await fetch(
        this.url(`/api/v1/projects/${encodeURIComponent(projectId)}/parse`),
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ full_refresh: true, tagging: true, force }),
        }
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async chat(body) {
      const res = await fetch(this.url("/api/v1/chat"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async chatStream(body, onEvent) {
      const res = await fetch(this.url("/api/v1/chat/stream"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
        },
        body: JSON.stringify(body),
      });
      const ct = res.headers.get("content-type") || "";
      if (!res.ok || !ct.includes("text/event-stream")) {
        const errText = await res.text();
        const err = new Error(errText || `HTTP ${res.status}`);
        err.status = res.status;
        throw err;
      }
      if (!res.body?.getReader) throw new Error("不支持流式响应");

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
          for (const line of chunk.split("\n")) {
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
    },

    async feedback(payload) {
      const res = await fetch(this.url("/api/v1/chat/feedback"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async deleteSession(sessionId) {
      const res = await fetch(
        this.url(`/api/v1/chat/${encodeURIComponent(sessionId)}`),
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async listKnowledge() {
      const res = await fetch(this.url("/api/v1/knowledge"), { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async uploadKnowledge(file, title) {
      const fd = new FormData();
      fd.append("file", file, file.name);
      if (title) fd.append("title", title);
      const res = await fetch(this.url("/api/v1/knowledge/upload"), {
        method: "POST",
        body: fd,
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async deleteKnowledge(docId) {
      const res = await fetch(
        this.url(`/api/v1/knowledge/${encodeURIComponent(docId)}`),
        { method: "DELETE" }
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async stopPatchRun(runId) {
      const res = await fetch(
        this.url(`/api/v1/agent/patch-runs/${encodeURIComponent(runId)}/stop`),
        { method: "POST" }
      );
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async createLocalAnalyze(body) {
      const res = await fetch(this.url("/api/v1/agent/local-runs"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    async createLocalCleanup(body) {
      const res = await fetch(this.url("/api/v1/agent/local-cleanup-runs"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}),
      });
      if (!res.ok) throw new Error(await res.text());
      return res.json();
    },

    agentWsUrl(runId) {
      const u = new URL(this.url(`/api/v1/agent/ws`));
      u.protocol = u.protocol === "https:" ? "wss:" : "ws:";
      u.searchParams.set("run_id", runId);
      return u.toString();
    },
  };

  window.BcaApi = Api;
})();

