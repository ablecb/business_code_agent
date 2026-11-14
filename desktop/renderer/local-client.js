(() => {
  /**
   * Client run runner: whitelist + shell/git with UI confirm.
   * Emits tool timeline events into the chat session via callbacks.
   */
  const ClientRuns = {
    ws: null,
    runId: null,
    onStatus: null,
    onFinished: null,
    /** (ev) => void — { event, action, result, chunk, stream, run_id } */
    onToolEvent: null,
    unsubLog: null,

    setStatus(text, kind) {
      if (typeof this.onStatus === "function") this.onStatus(text, kind);
    },

    emitTool(ev) {
      if (typeof this.onToolEvent === "function") {
        this.onToolEvent({ ...ev, run_id: this.runId });
      }
    },

    async start(runId) {
      if (!runId) return;
      if (!window.desktop?.runAction) {
        this.setStatus("需桌面客户端执行本机动作", "bad");
        return;
      }
      this.stopLocal();
      this.runId = runId;
      this.setStatus(`本机任务进行中 · ${runId}`, "busy");

      if (typeof window.desktop.onExecutorLog === "function") {
        this.unsubLog = window.desktop.onExecutorLog((payload) => {
          if (!payload) return;
          this.emitTool({
            event: "tool_log",
            action_id: payload.action_id,
            chunk: payload.chunk,
            stream: payload.stream || "stdout",
          });
          // also forward log chunks to server if WS open
          if (this.ws && this.ws.readyState === WebSocket.OPEN && payload.chunk) {
            try {
              this.ws.send(
                JSON.stringify({
                  type: "log_chunk",
                  run_id: this.runId,
                  action_id: payload.action_id,
                  chunk: payload.chunk,
                  stream: payload.stream || "stdout",
                })
              );
            } catch {
              /* ignore */
            }
          }
        });
      }

      const url = window.BcaApi.agentWsUrl(runId);
      const ws = new WebSocket(url);
      this.ws = ws;
      ws.onopen = () => ws.send(JSON.stringify({ type: "client_ready", run_id: runId }));
      ws.onmessage = (ev) => {
        let msg;
        try {
          msg = JSON.parse(ev.data);
        } catch {
          return;
        }
        this.onMessage(msg);
      };
      ws.onerror = () => this.setStatus("本机任务连接异常", "bad");
    },

    async onMessage(msg) {
      if (!msg || !msg.type) return;
      if (msg.type === "run_status" && msg.message) {
        this.setStatus(msg.message, "busy");
        const st = msg.status || (msg.run && msg.run.status) || "";
        const repairCount =
          (msg.run && typeof msg.run.repair_count === "number"
            ? msg.run.repair_count
            : null) ?? null;
        const maxRepairs =
          (msg.run && typeof msg.run.max_repairs === "number"
            ? msg.run.max_repairs
            : null) ?? null;
        // Surface repairing / 换方法 into the chat thinking timeline.
        if (
          st === "repairing" ||
          /进入修复|换方法|修复/.test(String(msg.message))
        ) {
          this.emitTool({
            event: "run_status",
            status: st || "repairing",
            message: msg.message,
            repair_count: repairCount,
            max_repairs: maxRepairs,
          });
        }
      }
      if (msg.type === "tool_event" && msg.event === "tool_call" && msg.action) {
        this.emitTool({ event: "tool_call", action: msg.action });
      }
      if (msg.type === "tool_event" && msg.event === "task_plan") {
        this.emitTool({
          event: "task_plan",
          run_id: msg.run_id,
          message: msg.message,
          plan_steps: msg.plan_steps || [],
          success_criteria: msg.success_criteria || "",
          plan_text: msg.plan_text || "",
        });
      }
      if (msg.type === "action" && msg.action) {
        await this.handleAction(msg.action);
      }
      if (msg.type === "run_finished") {
        const st = msg.status || "finished";
        const kind = st === "succeeded" ? "ok" : st === "stopped" ? "dim" : "bad";
        this.setStatus(msg.message || st, kind);
        if (typeof this.onFinished === "function") {
          this.onFinished(msg);
        }
        this.stopLocal();
      }
      if (msg.type === "error") {
        this.setStatus(msg.message || "服务端错误", "bad");
      }
    },

    needsConfirm(action) {
      if (!action) return false;
      // Server policy wins: read-only probes set needs_confirm=false.
      if (action.needs_confirm === false) return false;
      if (action.read_only === true) return false;
      if (action.needs_confirm === true) return true;
      const t = String(action.type || "");
      return [
        "run_shell",
        "run_powershell",
        "git_push",
        "git_merge",
        "git_commit",
        "sys_cleanup",
        "start_app",
        "stop_app",
        "maven_install",
        "docker_compose_up",
        "git_checkout",
        "git_pull",
        "open_browser",
        "milvus_mutate",
        "run_sql",
        "run_cypher",
      ].includes(t);
    },

    async handleAction(action) {
      this.setStatus(`执行 ${action.type}`, "busy");
      this.emitTool({ event: "tool_call", action });

      if (this.needsConfirm(action)) {
        this.setStatus(`等待确认：${action.ui_title || action.type}`, "busy");
        let approved = false;
        try {
          if (typeof window.BcaConfirmAction === "function") {
            approved = await window.BcaConfirmAction(action);
          } else {
            const preview =
              action.ui_preview ||
              action.command ||
              action.path ||
              action.note ||
              action.type;
            approved = window.confirm(
              `允许本机执行？\n\n${action.ui_title || action.type}\n${preview}`
            );
          }
        } catch {
          approved = false;
        }
        if (!approved) {
          const denied = {
            action_id: action.id,
            type: action.type,
            ok: false,
            error: "user_denied",
            stderr: "用户拒绝执行",
            meta: { denied: true },
          };
          this.emitTool({ event: "tool_result", action, result: denied });
          this.reply(denied);
          return;
        }
      }

      try {
        const result = await window.desktop.runAction(action);
        this.emitTool({ event: "tool_result", action, result });
        this.reply(result);
      } catch (err) {
        const fail = {
          action_id: action.id,
          type: action.type,
          ok: false,
          error: err.message || String(err),
          stderr: err.message || String(err),
        };
        this.emitTool({ event: "tool_result", action, result: fail });
        this.reply(fail);
      }
    },

    reply(result) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
      this.ws.send(
        JSON.stringify({
          type: "action_result",
          run_id: this.runId,
          result,
        })
      );
    },

    async stop() {
      try {
        if (this.runId) await window.BcaApi.stopPatchRun(this.runId);
      } catch {
        /* ignore */
      }
      try {
        this.ws?.send(JSON.stringify({ type: "stop", run_id: this.runId }));
      } catch {
        /* ignore */
      }
      this.stopLocal();
      this.setStatus("已停止本机任务", "dim");
    },

    stopLocal() {
      try {
        if (typeof this.unsubLog === "function") this.unsubLog();
      } catch {
        /* ignore */
      }
      this.unsubLog = null;
      try {
        this.ws?.close();
      } catch {
        /* ignore */
      }
      this.ws = null;
      this.runId = null;
    },
  };

  window.BcaPatchClient = ClientRuns;
  window.BcaClientRuns = ClientRuns;
})();
