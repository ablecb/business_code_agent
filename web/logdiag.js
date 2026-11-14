/* 日志查看与问题定位页逻辑 */
"use strict";

const $ = (id) => document.getElementById(id);

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let text = "";
    try { text = (await res.json()).detail || res.statusText; } catch (_) { text = res.statusText; }
    throw new Error(`${res.status} ${text}`);
  }
  return res.json();
}

function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function fmtSize(n) {
  if (n >= 1048576) return (n / 1048576).toFixed(1) + " MB";
  if (n >= 1024) return (n / 1024).toFixed(1) + " KB";
  return n + " B";
}

/* —— 日志文件浏览 —— */
let allFiles = [];
let curFile = null;

async function loadFiles() {
  try {
    const data = await api("/api/v1/logs/meta");
    allFiles = data.files || [];
    renderFiles();
  } catch (e) {
    $("files").innerHTML = `<li class="muted">加载失败: ${esc(e.message)}</li>`;
  }
}

function renderFiles() {
  const q = ($("filterFiles").value || "").toLowerCase();
  const list = allFiles.filter((f) => !q || f.path.toLowerCase().includes(q));
  $("files").innerHTML = list
    .map((f) => `<li data-path="${esc(f.path)}"><span class="badge">${esc(f.project)}</span> ${esc(f.name)} <span class="sz">${fmtSize(f.size)}</span></li>`)
    .join("");
}

$("files").addEventListener("click", (e) => {
  const li = e.target.closest("li[data-path]");
  if (!li) return;
  curFile = li.dataset.path;
  $("curFile").value = curFile;
});

$("filterFiles").addEventListener("input", renderFiles);

/* —— 尾部读取 / 筛选 —— */
$("btnTail").addEventListener("click", async () => {
  if (!curFile) { alert("请先在左侧选择日志文件"); return; }
  $("btnTail").disabled = true;
  try {
    const p = ($("tailPattern").value || "").trim();
    const data = await api(`/api/v1/logs/tail?path=${encodeURIComponent(curFile)}&lines=${$("tailLines").value || 400}` +
      (p ? `&pattern=${encodeURIComponent(p)}` : ""));
    const text = (data.lines || []).map((r) => r.text).join("\n");
    $("logText").value = text;
    $("diag").innerHTML = `<span class="muted">已载入 ${data.count} 行（${esc(curFile)}）。可再编辑后点击「开始定位问题」。</span>`;
  } catch (e) {
    alert("读取失败: " + e.message);
  } finally {
    $("btnTail").disabled = false;
  }
});

/* —— 诊断 —— */
function renderDiag(md) {
  const body = String(md || "（无输出）");
  const html = esc(body)
    .split(/\n{1,}/)
    .map((block) => {
      const m = block.match(/^##\s*(.*)$/);
      if (m) return `<h2>${esc(m[1])}</h2>`;
      return `<div>${block}</div>`;
    })
    .join("");
  $("diag").innerHTML = html;
}

$("btnDiag").addEventListener("click", async () => {
  const logText = ($("logText").value || "").trim();
  if (!logText) { alert("请粘贴日志或先读取日志文件"); return; }
  $("btnDiag").disabled = true;
  $("diagState").textContent = "诊断中（锚点提取 → 检索 → 读码 → 生成诊断）…";
  $("diag").innerHTML = `<span class="muted">处理中…（约 20~120 秒，请勿重复点击）</span>`;
  try {
    const body = { log_text: logText, question: ($("question").value || "").trim() || null };
    const data = await api("/api/v1/logs/diagnose", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    renderDiag(data.reply);
    $("diagState").textContent =
      `完成 · 锚点: ${Object.values(data.anchors || {}).flat().length} · ` +
      `分支走错信号: ${data.branch_hit ? "是（强制五段证据链）" : "否"}`;
  } catch (e) {
    $("diagState").textContent = "";
    $("diag").innerHTML = `<span style="color:#ff8f8f">诊断失败: ${esc(e.message)}</span>`;
  } finally {
    $("btnDiag").disabled = false;
  }
});

loadFiles();