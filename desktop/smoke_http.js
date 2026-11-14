/**
 * Localhost-only HTTP smoke checks after start_app.
 */
const http = require("http");
const https = require("https");
const { URL } = require("url");

function assertLocalUrl(raw) {
  const u = new URL(String(raw || ""));
  if (u.protocol !== "http:" && u.protocol !== "https:") {
    throw new Error("仅允许 http/https");
  }
  const host = (u.hostname || "").toLowerCase();
  if (!["127.0.0.1", "localhost", "::1"].includes(host)) {
    throw new Error(`冒烟仅允许本机地址，拒绝: ${host}`);
  }
  return u;
}

function requestOnce(urlStr, { method = "GET", timeoutMs = 10000, headers = {}, body } = {}) {
  const u = assertLocalUrl(urlStr);
  const lib = u.protocol === "https:" ? https : http;
  const payload = body != null ? String(body) : null;
  const opts = {
    protocol: u.protocol,
    hostname: u.hostname,
    port: u.port || (u.protocol === "https:" ? 443 : 80),
    path: `${u.pathname}${u.search || ""}`,
    method: String(method || "GET").toUpperCase(),
    headers: { ...(headers || {}) },
    timeout: timeoutMs,
  };
  if (payload && !opts.headers["Content-Length"]) {
    opts.headers["Content-Length"] = Buffer.byteLength(payload);
  }
  return new Promise((resolve) => {
    const req = lib.request(opts, (res) => {
      const chunks = [];
      res.on("data", (c) => chunks.push(c));
      res.on("end", () => {
        const text = Buffer.concat(chunks).toString("utf8").slice(0, 8000);
        resolve({
          ok: true,
          status: res.statusCode || 0,
          headers: res.headers,
          body: text,
        });
      });
    });
    req.on("timeout", () => {
      req.destroy();
      resolve({ ok: false, error: `timeout ${timeoutMs}ms`, status: 0, body: "" });
    });
    req.on("error", (err) => {
      resolve({ ok: false, error: err.message || String(err), status: 0, body: "" });
    });
    if (payload) req.write(payload);
    req.end();
  });
}

/**
 * @param {object} action
 * @returns {Promise<object>}
 */
async function runHttpSmoke(action) {
  const timeoutMs = Number(action.timeout_ms) || 12000;
  const method = action.method || "GET";
  const expectStatus = Number(action.expect_status != null ? action.expect_status : 200);
  const expectRe = action.expect_body_regex
    ? new RegExp(String(action.expect_body_regex))
    : null;
  const urls = [];
  if (action.url) urls.push(String(action.url));
  if (Array.isArray(action.urls)) {
    for (const u of action.urls) if (u) urls.push(String(u));
  }
  if (!urls.length) {
    urls.push(
      "http://127.0.0.1:8080/actuator/health",
      "http://127.0.0.1:8080/health",
      "http://127.0.0.1:8080/"
    );
  }

  const attempts = [];
  for (const url of urls.slice(0, 8)) {
    let res;
    try {
      res = await requestOnce(url, {
        method,
        timeoutMs,
        headers: action.headers || {},
        body: action.body,
      });
    } catch (err) {
      attempts.push({ url, ok: false, error: err.message || String(err) });
      continue;
    }
    const statusOk = res.status === expectStatus || (expectStatus === 200 && res.status >= 200 && res.status < 400);
    const bodyOk = expectRe ? expectRe.test(res.body || "") : true;
    const pass = !!res.ok && !res.error && statusOk && bodyOk;
    attempts.push({
      url,
      ok: pass,
      status: res.status,
      error: res.error || (!statusOk ? `status ${res.status} != ${expectStatus}` : null),
      body_preview: (res.body || "").slice(0, 500),
    });
    if (pass) {
      return {
        ok: true,
        stdout: `smoke ok ${method} ${url} → ${res.status}\n${(res.body || "").slice(0, 800)}`,
        content: JSON.stringify({ attempts }, null, 2),
        meta: { attempts, passed_url: url, status: res.status },
      };
    }
  }

  return {
    ok: false,
    error: "所有冒烟探测均失败",
    stdout: attempts
      .map((a) => `${a.ok ? "OK" : "FAIL"} ${a.url} ${a.status || ""} ${a.error || ""}`)
      .join("\n"),
    stderr: "smoke failed",
    content: JSON.stringify({ attempts }, null, 2),
    meta: { attempts },
  };
}

module.exports = { runHttpSmoke, assertLocalUrl };
