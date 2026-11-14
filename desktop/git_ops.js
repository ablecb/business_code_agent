/**
 * Structured git actions — no free-form git argv from the model.
 */
const { runCommand, assertAllowed } = require("./shell_gate");

function safeName(v, label) {
  const s = String(v || "").trim();
  if (!s) throw new Error(`${label} 不能为空`);
  if (!/^[A-Za-z0-9._\-/@]+$/.test(s)) {
    throw new Error(`${label} 含非法字符`);
  }
  return s;
}

function safeMessage(v) {
  const s = String(v || "").trim();
  if (!s) throw new Error("commit message 不能为空");
  if (s.length > 500) throw new Error("commit message 过长");
  if (/[`$]|\\n|\\r/.test(s)) throw new Error("commit message 含非法字符");
  return s;
}

/**
 * @param {object} action
 * @param {string} root
 * @param {(c:string,s:string)=>void} [onLog]
 */
async function executeGit(action, root, onLog) {
  const type = action?.type;
  let argv;
  switch (type) {
    case "git_status":
      argv = ["git", "status", "-sb"];
      break;
    case "git_branch":
      argv = ["git", "branch", "-vv"];
      break;
    case "git_checkout": {
      const branch = safeName(action.branch || action.ref, "branch");
      const create =
        action.create === true ||
        action.new_branch === true ||
        action.create_branch === true;
      if (create) {
        const createArgv = ["git", "checkout", "-b", branch];
        assertAllowed(createArgv.join(" "));
        const created = await runCommand({
          root,
          argv: createArgv,
          timeoutMs: Number(action.timeout_ms) || 120000,
          shellKind: "none",
          onLog,
        });
        const blob = `${created.stderr || ""}\n${created.stdout || ""}\n${created.error || ""}`;
        if (created.ok) {
          return created;
        }
        if (/already exists|已存在/i.test(blob)) {
          argv = ["git", "checkout", branch];
          break;
        }
        return created;
      }
      argv = ["git", "checkout", branch];
      break;
    }
    case "git_pull": {
      const remote = safeName(action.remote || "origin", "remote");
      const ref = action.ref || action.branch;
      argv = ref
        ? ["git", "pull", remote, safeName(ref, "ref")]
        : ["git", "pull", remote];
      break;
    }
    case "git_merge": {
      const ref = safeName(action.ref || action.branch, "merge ref");
      argv = ["git", "merge", "--no-ff", ref];
      break;
    }
    case "git_commit": {
      const msg = safeMessage(action.message);
      // stage all tracked/untracked in project (user already confirmed)
      await runCommand({
        root,
        argv: ["git", "add", "-A"],
        timeoutMs: 120000,
        shellKind: "none",
        onLog,
      });
      argv = ["git", "commit", "-m", msg];
      break;
    }
    case "git_push": {
      const remote = safeName(action.remote || "origin", "remote");
      const ref = action.ref || action.branch;
      if (action.force) {
        throw new Error("禁止 force push（安全策略）");
      }
      argv = ref
        ? ["git", "push", "-u", remote, safeName(ref, "ref")]
        : ["git", "push", remote];
      break;
    }
    default:
      throw new Error(`未知 git action: ${type}`);
  }
  assertAllowed(argv.join(" "));
  return runCommand({
    root,
    argv,
    timeoutMs: Number(action.timeout_ms) || 300000,
    shellKind: "none",
    onLog,
  });
}

module.exports = { executeGit };
