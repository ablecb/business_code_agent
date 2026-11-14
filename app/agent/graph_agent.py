from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, TypedDict

import httpx

from app.agent import session_store
from app.agent.explore_agent import (
    _class_method_from_hint,
    _lexical_main_entry,
    _wants_api_inventory,
    _wants_endpoint_lookup,
    _wants_flow_downstream_apis,
    _wants_full_flow,
    _wants_gate_detail,
    _wants_source_paste,
    _wants_sync_entry_first,
    run_explore_agent,
)
from app.agent.general_chat import general_chat_reply, is_general_chat_question
from app.agent.intent_taxonomy import (
    INTENT_PROMPT,
    get_option_by_id,
    heuristic_intent as _heuristic_intent,
    is_explore_intent,
    normalize_intent,
    parse_option_selection,
    wants_meta as _taxonomy_wants_meta,
)
from app.agent.options_planner import build_change_options, render_options_reply
from app.agent.patcher import (
    allowlist_from_patches,
    apply_patch_list,
    extract_scaffold_file_patches,
    filter_patches_to_allowlist,
    normalize_patch_paths,
    parse_patches_from_llm,
    real_scope_files,
    summarize_patch_results,
    summarize_patches_with_lines,
)
from app.agent.verify_loop import (
    DEFAULT_MAX_REPAIRS,
    build_repair_prompt,
    emit_progress as _emit_verify_progress,
    render_verify_reply,
    rollback_workspace,
    run_verify_once,
)
from app.agent.runtime.profile import get_runtime_profile
from app.agent.trajectory import AppendTrajectory, wrap_progress
from app.core.config import get_settings
from app.core.utils import truncate
from app.graph.builder import query_flow
from app.llm.factory import get_llm_provider
from app.llm.tracing import traceable
from app.models.schemas import ChatResponse, SymbolHit
from app.tagging.search import hybrid_search, hybrid_search_multi
from app.agent.prompts import load_prompt

logger = logging.getLogger("app.agent")


class AgentState(TypedDict, total=False):
    message: str
    project_id: str
    project_ids: list[str]
    session_id: str
    client_apply: bool
    client_run_id: str
    action: str
    option_id: str
    intent: str
    keywords: list[str]
    entry_hint: str
    detail: str
    need_options: bool
    want_apply: bool
    want_verify: bool
    retrieved: list[dict[str, Any]]
    flow: dict[str, Any]
    scope: list[dict[str, Any]]
    patches: list[dict[str, Any]]
    patch_results: list[dict[str, Any]]
    explore_steps: list[dict[str, Any]]
    research_plan: dict[str, Any]
    answer_review: dict[str, Any]
    reply: str
    on_progress: Any
    phase: str
    options: list[dict[str, Any]]
    selected_option_id: str
    verify_report: dict[str, Any]
    workspace_snapshot_id: str


async def _emit_progress(state: AgentState, event: dict[str, Any]) -> None:
    on_progress = state.get("on_progress")
    if not on_progress:
        return
    try:
        maybe = on_progress(event)
        if hasattr(maybe, "__await__"):
            await maybe  # type: ignore[misc]
    except Exception:
        logger.debug("progress emit failed", exc_info=True)


async def _emit_intent_progress(state: AgentState, *, source: str) -> None:
    """Surface classified intent early for UI / local chat logs."""
    intent = normalize_intent(state.get("intent"), state.get("message") or "")
    await _emit_progress(
        state,
        {
            "type": "intent",
            "status": "done",
            "note": f"意图识别 → {intent}（{source}）",
            "intent": intent,
            "keywords": list(state.get("keywords") or []),
            "entry_hint": state.get("entry_hint") or "",
            "detail": truncate(str(state.get("detail") or ""), 400),
            "need_options": bool(state.get("need_options")),
            "want_apply": bool(state.get("want_apply")),
            "want_verify": bool(state.get("want_verify")),
            "source": source,
        },
    )


async def _understand(state: AgentState) -> AgentState:
    msg = state["message"]
    history = session_store.list_messages(state["session_id"])
    ctx = session_store.get_context(state["session_id"])
    # Always ask the intent LLM (heuristic only on failure).
    llm = get_llm_provider()
    hist_txt = "\n".join(
        f"{m['role']}: {truncate(m['content'], 300)}" for m in history[-8:]
    )
    prior_topic = str(ctx.get("last_explore_question") or "").strip()
    locked = _locked_flow_entry(ctx) or ""
    phase = str(ctx.get("phase") or state.get("phase") or "idle")
    pending = state.get("options") or ctx.get("pending_options") or []
    pending_ids = [
        str(o.get("id"))
        for o in pending
        if isinstance(o, dict) and o.get("id")
    ]
    # Session signal only (not user-utterance keywords): pending implement CTA.
    awaiting_apply = phase == "awaiting_option" and "implement_now" in pending_ids
    source = "llm"
    await _emit_progress(
        state,
        {
            "type": "step",
            "status": "running",
            "step_id": "intent",
            "note": "正在识别意图与是否动手改代码（调用模型）…",
        },
    )
    try:
        import asyncio

        content = await asyncio.wait_for(
            llm.chat(
                [
                    {"role": "system", "content": INTENT_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"历史:\n{hist_txt or '(无)'}\n\n"
                            f"上一问主题: {prior_topic or '(无)'}\n"
                            f"锁定入口: {locked or '(无)'}\n"
                            f"会话阶段: {phase}\n"
                            f"待确认改码: {'yes' if awaiting_apply else 'no'}\n"
                            f"待选方案id: {pending_ids or '(无)'}\n\n"
                            f"当前用户:{state['message']}\n"
                            "若当前是承接上一问的短追问（如「那这一步是怎么校验的」），"
                            "intent 必须为 ask（理解流程/关卡），不要判成 meta。\n"
                            "若「待确认改码=yes」：根据用户本轮是否同意动手改代码设置 "
                            "want_apply（同意→true+modify；拒绝→false+meta；"
                            "继续问清单细节→want_apply=false+ask/plan）。"
                            "由语义判断，不要假设固定口令。"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=400,
            ),
            timeout=12.0,
        )
        m = re.search(r"\{.*\}", content, re.DOTALL)
        data = json.loads(m.group(0) if m else content)
    except Exception as e:
        logger.warning("intent LLM failed (%s); using heuristic intent", e)
        data = _heuristic_intent(msg)
        source = "llm_fallback_heuristic"
    if not isinstance(data, dict):
        data = _heuristic_intent(msg)
        source = "llm_fallback_heuristic"
    state["intent"] = normalize_intent(data.get("intent"), msg)
    state["capability"] = str(data.get("capability") or "").strip()
    state["keywords"] = data.get("keywords") or []
    state["entry_hint"] = data.get("entry_hint") or ""
    state["detail"] = data.get("detail") or state["message"]
    state["need_options"] = bool(data.get("need_options"))
    state["want_apply"] = bool(data.get("want_apply"))
    state["want_verify"] = bool(data.get("want_verify"))
    await _emit_intent_progress(state, source=source)
    return state


async def _retrieve(state: AgentState) -> AgentState:
    settings = get_settings()
    q_parts = [state["message"]] + (state.get("keywords") or [])
    if state.get("entry_hint"):
        q_parts.append(state["entry_hint"])
    q = " ".join(q_parts)
    pids = [p for p in (state.get("project_ids") or []) if p]
    if not pids:
        pids = [state["project_id"]] if state.get("project_id") else []
    top_k = 12
    hits = (
        hybrid_search_multi(q, pids, top_k=top_k)
        if len(pids) > 1
        else hybrid_search(q, pids[0] if pids else state["project_id"], top_k=top_k)
    )
    state["retrieved"] = [h.model_dump() for h in hits]
    entry = state.get("entry_hint") or (hits[0].fqcn if hits else "")
    # prefer controller-like hit → Class#method when available
    entry_cm = ""
    for h in hits:
        if h.kind == "Controller" or "Controller" in (h.fqcn or ""):
            entry = h.fqcn.split(".")[-1]
            # Prefer method from tags / summary Class#method
            blob = f"{' '.join(h.tags or [])} {h.summary or ''}"
            m = re.search(r"([A-Za-z][\w]*Controller)#[A-Za-z_][\w]*", blob)
            if m:
                entry_cm = m.group(0)
            break
    intent = normalize_intent(state.get("intent"), state.get("message") or "")
    # ask/plan business order: Milvus entry → Neo4j FOLLOWS (not CALLS neighborhood).
    if intent in {"ask", "plan"}:
        from app.agent.explore_agent import (
            _class_method_from_hint,
            _lexical_main_entry,
            _overlay_follows_on_flow,
        )

        cm = (
            entry_cm
            or _class_method_from_hint(state.get("entry_hint") or "")
            or _lexical_main_entry(state.get("message") or "")
            or ""
        )
        if not cm:
            # Fall back to first Controller#method mentioned in top hits.
            for h in hits:
                blob = f"{' '.join(h.tags or [])} {h.summary or ''} {h.fqcn or ''}"
                m = re.search(r"([A-Za-z][\w]*Controller)#[A-Za-z_][\w]*", blob)
                if m:
                    cm = m.group(0)
                    break
        flow_struct = _overlay_follows_on_flow(
            state["project_id"], None, entry_cm=cm or None
        )
        if flow_struct and (flow_struct.get("flow_doc") or {}).get("stages"):
            stages = (flow_struct.get("flow_doc") or {}).get("stages") or []
            nodes = []
            edges = []
            prev_id = None
            for st in stages:
                mid = str(st.get("class_method") or "")
                if not mid:
                    continue
                nodes.append(
                    {
                        "id": mid,
                        "labels": ["Method"],
                        "properties": {
                            "class_method": mid,
                            "title": st.get("title"),
                            "desc": st.get("desc_business") or st.get("desc"),
                            "flow_key": flow_struct.get("flow_key"),
                        },
                    }
                )
                if prev_id:
                    edges.append(
                        {
                            "start": prev_id,
                            "end": mid,
                            "type": "FOLLOWS",
                            "properties": {"flow_key": flow_struct.get("flow_key")},
                        }
                    )
                prev_id = mid
            state["flow"] = {
                "entry": flow_struct.get("class_method") or cm or entry,
                "nodes": nodes,
                "edges": edges,
                "follows_source": flow_struct.get("follows_source"),
                "structured": flow_struct,
            }
            return state
        state["flow"] = {
            "entry": cm or entry,
            "nodes": [],
            "edges": [],
            "follows_source": "",
        }
        return state

    if entry:
        try:
            # Non-flow intents may still use CALLS neighborhood for scope planning.
            flow = query_flow(entry, state["project_id"], depth=settings.graph_max_depth)
        except Exception:
            flow = {"entry": entry, "nodes": [], "edges": []}
    else:
        flow = {"entry": "", "nodes": [], "edges": []}
    state["flow"] = flow
    return state


async def _scope_plan(state: AgentState) -> AgentState:
    llm = get_llm_provider()
    retrieved = state.get("retrieved") or []
    flow = state.get("flow") or {}
    files = sorted({r.get("file_path") for r in retrieved if r.get("file_path")})
    node_files = []
    for n in flow.get("nodes") or []:
        props = n.get("properties") or {}
        if props.get("file_path"):
            node_files.append(props["file_path"])
    files = sorted(set(files) | set(node_files))

    prompt = f"""你是 Java 业务改造规划助手。仅根据用户真实需求与检索/调用图列出修改范围。
不要默认写成「兼容 Spring Cloud / Feign / Stream」——除非用户明确要求微服务改造。
用户需求: {state.get('detail') or state['message']}
意图: {state.get('intent')}
相关符号: {json.dumps(retrieved[:10], ensure_ascii=False)}
相关文件: {files[:30]}
调用图边数量: {len(flow.get('edges') or [])}

输出严格 JSON 数组，每项:
{{"file":"相对路径","reason":"为什么改（贴近业务需求）","action":"add_field|persist|downstream|new_endpoint|validate|other","priority":1}}
file 必须从上方「相关文件」列表原样复制（含模块前缀如 loan-service/src/main/java/...）。
禁止只写包路径（如 com/demo/.../Foo.java）而不含模块目录。
优先覆盖: Controller/DTO/VO/Entity/Mapper/Processor/Service 等与需求相关的层。
新增 HTTP 接口时优先 Controller / API / DTO / Service；除非用户点名，不要把无关业务类塞进 scope。
"""
    content = await llm.chat(
        [
            {"role": "system", "content": "只输出 JSON 数组"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1500,
    )
    try:
        m = re.search(r"\[.*\]", content, re.DOTALL)
        scope = json.loads(m.group(0) if m else content)
        if not isinstance(scope, list):
            raise ValueError("not list")
        scope = [s for s in scope if isinstance(s, dict) and s.get("file")]
        if not scope:
            raise ValueError("empty scope")
    except Exception:
        scope = [
            {"file": f, "reason": "与检索命中相关", "action": "other", "priority": i + 1}
            for i, f in enumerate(files[:8])
        ]
        if not scope:
            scope = []
    state["scope"] = scope
    session_store.update_context(
        state["session_id"],
        {"scope": scope, "intent": state.get("intent"), "entry": flow.get("entry")},
    )
    return state


async def _patch_fixed(state: AgentState) -> AgentState:
    intent = normalize_intent(state.get("intent"), state.get("message") or "")
    state["patches"] = state.get("patches") or []
    state["patch_results"] = state.get("patch_results") or []
    if intent != "modify" and not state.get("selected_option_id"):
        return state

    llm = get_llm_provider()
    scope = state.get("scope") or []
    settings = get_settings()
    snippets = []
    for item in scope[:5]:
        fp = settings.project_root_path / (item.get("file") or "")
        if fp.exists():
            snippets.append(
                f"FILE {item.get('file')}:\n{truncate(fp.read_text(encoding='utf-8', errors='ignore'), 2500)}"
            )
    selected = state.get("selected_option_id")
    opt = get_option_by_id(state.get("options") or [], selected)
    opt_txt = ""
    if opt:
        opt_txt = (
            f"\n已选方案 {opt.get('id')}: {opt.get('title')}\n"
            f"{opt.get('summary')}\n"
            f"scope_preview: {json.dumps(opt.get('scope_preview') or [], ensure_ascii=False)}\n"
        )
        if not scope and isinstance(opt.get("scope_preview"), list):
            scope = [s for s in opt["scope_preview"] if isinstance(s, dict)]
            state["scope"] = scope
    file_hints = sorted(
        {
            str(r.get("file_path") or "").replace("\\", "/")
            for r in (state.get("retrieved") or [])
            if r.get("file_path")
        }
        | {
            str(s.get("file") or "").replace("\\", "/")
            for s in scope
            if isinstance(s, dict) and s.get("file")
        }
    )
    await _emit_progress(
        state,
        {
            "type": "step",
            "status": "running",
            "step_id": "patch_llm",
            "note": "正在生成代码补丁（调用模型，可能需要几十秒）…",
        },
    )
    content = await llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是资深 Java 工程师。按原项目代码风格给出改造方案与补丁。\n"
                    "补丁格式：\n### FILE: relative/path.java\n*** OLD\n旧代码\n*** NEW\n新代码\n"
                    "新增文件：### FILE: path\n*** NEW_FILE\n完整内容\n"
                    "### FILE 路径必须从「可选文件」列表原样复制；禁止裸包路径。\n"
                    "绝对禁止修改「可选文件」列表之外的任何文件；不要顺带改工具类/无关 Service。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"需求: {state.get('detail')}\n{opt_txt}"
                    f"范围: {json.dumps(scope, ensure_ascii=False)}\n"
                    f"可选文件（仅可改这些）: {json.dumps(file_hints[:40], ensure_ascii=False)}\n"
                    f"代码片段:\n{chr(10).join(snippets)}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=3500,
    )
    patches = normalize_patch_paths(
        parse_patches_from_llm(content),
        hints=file_hints,
        project_id=state.get("project_id"),
    )
    # Hard filter: never keep over-scoped files.
    allow = real_scope_files(scope) or list(file_hints)
    ctx = session_store.get_context(state["session_id"])
    locked_allow = ctx.get("locked_file_allowlist")
    if isinstance(locked_allow, list) and locked_allow:
        allow = [str(x) for x in locked_allow if x]
    if allow:
        patches = filter_patches_to_allowlist(patches, allow, drop_if_empty_allowlist=True)
    state["patches"] = patches
    state["_draft"] = content
    return state


async def _verify_patch_writes(state: AgentState) -> AgentState:
    """Static check: patch files written successfully (not compile/test)."""
    results = state.get("patch_results") or []
    if not results:
        return state
    bad = [r for r in results if not r.get("ok")]
    state["verify_ok"] = len(bad) == 0
    state["verify_errors"] = bad
    return state


async def _verify(state: AgentState) -> AgentState:
    """Backward-compatible name used by LangGraph node — patch write check only."""
    return await _verify_patch_writes(state)


def _history_before_current(state: AgentState) -> list[dict[str, Any]]:
    """Session messages excluding the just-appended current user turn."""
    msgs = session_store.list_messages(state["session_id"])
    cur = state.get("message") or ""
    if msgs and msgs[-1].get("role") == "user" and (msgs[-1].get("content") or "") == cur:
        return msgs[:-1]
    return msgs


def _wants_meta(intent: str, message: str) -> bool:
    """Only true for explicit conversation-recall asks; ignore bare intent=meta."""
    del intent
    return _taxonomy_wants_meta(message)


def _is_correction(message: str) -> bool:
    """Structural-only: system may inject a correction marker; no CN keyword lists."""
    text = (message or "").strip()
    if not text:
        return False
    # System / protocol markers only (not natural-language vocabulary gates).
    return text.startswith("[CORRECTION]") or text.startswith("<<CORRECTION>>")


# 「现在再回答一下 / 重新回答 / 回答一下」——纯重问句（重跑上一问，无自身代码内容）。
# 保守匹配：仅在消息不含代码锚点 && 命中重问动词短语时解析回上一问，
# 避免劫持真实新问题（如「回答一下放款流程」——含「流程」锚点，不走重问）。
_REANSWER_RE = re.compile(
    r"^(?:现在|那|请|麻烦|帮我|帮我再|麻烦请|请再|你|我|你能|能不能|可以|好吗|好)?\s*"
    r"(?:再|重新|又|继续|接着)?\s*"
    r"(?:回答|答|说一下|说说|重复|答复|回复|说)"
    r"(?:一下|一遍|一次|一遍这个问题|一下这个问题|这个问题|这个|一遍吧|一下吧|一遍好吗)?\s*"
    r"[。！!？?…]*$"
)

# English re-answer forms (language-agnostic guard): "answer again",
# "answer it", "re-answer", "please answer", "say it again", "repeat"...
_REANSWER_RE_EN = re.compile(
    r"^(?:(?:please|now|hey|ok)\s*)?"
    r"(?:re-?answer|answer\s+(?:again|it|this|that|me)|answer|repeat|"
    r"say\s+(?:it|that)\s+again|give\s+me\s+the\s+answer|"
    r"answer\s+it\s+again|answer\s+this\s+again|answer\s+that\s+again)"
    r"\s*[.!?…]*$",
    re.IGNORECASE,
)


def _is_reanswer_request(message: str) -> bool:
    """True when message is a pure re-answer request with NO new code content.

    Conservative gate: the message must (a) contain no code anchors
    (is_general_chat_question -> True) and (b) be entirely consumed by the
    re-ask phrasing. Real new questions like 「回答一下放款流程」 contain
    「流程」anchor -> False (kept as a fresh ask). Pure recall phrases like
    「我刚才问你什么了 / 再说一遍」 stay meta (taxonomy gate first).
    """
    text = (message or "").strip()
    if not text or _is_correction(text):
        return False
    if _taxonomy_wants_meta(text):
        return False
    if text in _REANSWER_BARE:
        return False  # 裸动词（答/回答）可能是输入残片，不当作重问
    if (
        text not in _REANSWER_RE_WHOLE
        and not _REANSWER_RE.fullmatch(text)
        and not _REANSWER_RE_EN.fullmatch(text)
    ):
        return False
    if not is_general_chat_question(text):
        return False
    return True


# 裸动词不算重问（可能是输入残片或新问题的开头）。
_REANSWER_BARE = frozenset(
    {
        "回答",
        "答",
        "答复",
        "回复",
        "说一下",
        "说说",
        "重复",
        "说",
        # English bare verbs are NOT re-asks (may be input fragments / new asks),
        # except unambiguous re-answer commands (re-answer / reanswer) that the
        # English regex owns.
        "answer",
        "repeat",
        "say",
        "again",
        "once more",
    }
)

# Shortcut set for re-answers whose regex shape is obvious but the verb
# alone wouldn't match the full regex (e.g.「再回答一下这句」).
_REANSWER_RE_WHOLE = frozenset(
    {
        "再回答一下",
        "再回答一遍",
        "现在再回答一下",
        "现在再回答",
        "现在重新回答",
        "重新回答一下",
        "重新回答",
        "重新答",
        "再回答",
        "回答一下这个问题",
        "回答这个问题",
        "回答一下",
        "再答一下",
        "再答一遍",
        "再答",
        "答一下",
        "重新答一下",
        "重答一下",
        "重新说一遍",
        "重复一下回答",
        "重新回答一遍",
        "回答这个",
        "回答一下这个",
        "请再回答一下",
        "麻烦再回答一下",
    }
)


def _resolve_reanswer(session_id: str, message: str) -> str | None:
    """Resolve a bare re-answer request to the previous non-meta, non-reask user question."""
    if not _is_reanswer_request(message):
        return None
    msgs = session_store.list_messages(session_id)
    if msgs and msgs[-1].get("role") == "user" and (msgs[-1].get("content") or "") == message:
        msgs = msgs[:-1]
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if not content or _is_correction(content):
            continue
        if _wants_meta("general", content):
            continue
        if _is_reanswer_request(content):
            continue  # 跳过更早的纯重问胶水，避免递归回自身
        return content
    return None


def _is_followup(message: str) -> bool:
    """Deprecated keyword follow-up detector — always False; LLM session gate decides."""
    del message
    return False


def _locked_flow_entry(ctx: dict[str, Any] | None) -> str | None:
    """Prior-turn Class#method lock."""
    if not ctx:
        return None
    for key in ("last_class_method", "last_entry"):
        cm = _class_method_from_hint(str(ctx.get(key) or ""))
        if cm and "#" in cm:
            return cm
    return None


def _prior_spine_method_names(ctx: dict[str, Any] | None) -> set[str]:
    """Method-name tokens from prior flow symbols / answer steps (lowercase)."""
    if not ctx:
        return set()
    names: set[str] = set()
    for raw in list(ctx.get("last_flow_symbols") or []) + list(
        ctx.get("last_symbols") or []
    ):
        text = str(raw or "")
        if "#" in text:
            meth = text.split("#", 1)[1].split("(", 1)[0].strip()
            if meth:
                names.add(meth.lower())
        elif text:
            names.add(text.strip().lower())
    for step in ctx.get("last_answer_steps") or []:
        if not isinstance(step, dict):
            continue
        for key in ("class_method", "cm", "method"):
            raw = str(step.get(key) or "")
            if "#" in raw:
                meth = raw.split("#", 1)[1].split("(", 1)[0].strip()
                if meth:
                    names.add(meth.lower())
    return names


def _prior_answer_step_blobs(ctx: dict[str, Any] | None) -> list[str]:
    """Compact prior answer step texts for LLM context (not sole gates)."""
    if not ctx:
        return []
    blobs: list[str] = []
    for step in ctx.get("last_answer_steps") or []:
        if not isinstance(step, dict):
            continue
        parts = [
            str(step.get("title") or "").strip(),
            str(step.get("detail") or "").strip(),
            str(step.get("class_method") or step.get("cm") or "").strip(),
        ]
        blob = re.sub(r"\s+", " ", " ".join(p for p in parts if p)).strip()
        if blob:
            blobs.append(blob)
    return blobs


def _overlap_prior_answer_score(
    message: str, ctx: dict[str, Any] | None
) -> int:
    """Weak overlap feature for LLM context only — never a sole lock gate."""
    text = (message or "").strip()
    if not text or not ctx:
        return 0
    try:
        from app.agent.pev_agent import _cjk_overlap_score, _token_overlap_score
    except Exception:
        return 0
    best = 0
    for blob in _prior_answer_step_blobs(ctx):
        sc = _cjk_overlap_score(blob, text) + _token_overlap_score(blob, text)
        if sc > best:
            best = sc
    locked = _locked_flow_entry(ctx) or ""
    if locked:
        sc = _token_overlap_score(locked, text) + _cjk_overlap_score(locked, text)
        best = max(best, sc)
    prior_q = str(ctx.get("last_explore_question") or "").strip()
    if prior_q:
        sc = _cjk_overlap_score(prior_q, text) + _token_overlap_score(prior_q, text)
        best = max(best, sc // 2)
    return best


def _high_overlap_prior_answer(
    message: str, ctx: dict[str, Any] | None, *, min_score: int = 6
) -> bool:
    """Weak feature helper; do not use as sole session lock gate."""
    return _overlap_prior_answer_score(message, ctx) >= min_score


def _mentions_prior_spine_latin(message: str, ctx: dict[str, Any] | None) -> bool:
    """Structural: ask literally names a prior spine Latin method id."""
    text = message or ""
    names = _prior_spine_method_names(ctx)
    if not names:
        return False
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]{2,80})\b", text):
        tok = m.group(1).lower()
        if tok in names:
            return True
    for name in names:
        if len(name) >= 3 and re.search(
            rf"(?i)(^|[^A-Za-z0-9_]){re.escape(name)}([^A-Za-z0-9_]|$)",
            text,
        ):
            return True
    return False


def _mentions_prior_spine_method(message: str, ctx: dict[str, Any] | None) -> bool:
    """Structural Latin-only spine mention (CJK overlap is not a dig-in gate)."""
    return _mentions_prior_spine_latin(message, ctx)


def _explicit_cm_in_question(message: str) -> str | None:
    cm = _class_method_from_hint(message)
    if cm:
        return cm
    m = re.search(
        r"\b([A-Z][A-Za-z0-9_]{2,80}(?:Controller|ServiceImpl|Listener|Job))"
        r"#([A-Za-z_][A-Za-z0-9_]{1,80})\b",
        message or "",
    )
    if m:
        return f"{m.group(1)}#{m.group(2)}"
    return None


def is_new_topic_ask(
    message: str,
    locked_cm: str | None,
    ctx: dict[str, Any] | None = None,
) -> bool:
    """Structural-only new-topic probe (not authoritative).

    Authoritative gate is ``resolve_session_ctx_gate`` / LLM JSON.
    True only when question names a different Class#method than the lock.
    """
    text = (message or "").strip()
    locked = _class_method_from_hint(locked_cm) or (locked_cm or "").strip()
    if not text or not locked or "#" not in locked:
        return False
    explicit = _explicit_cm_in_question(text)
    if explicit and explicit.lower() != locked.lower():
        return True
    if _mentions_prior_spine_latin(text, ctx):
        return False
    return False


def is_dig_in_under_lock(
    message: str,
    locked_cm: str | None,
    ctx: dict[str, Any] | None = None,
) -> bool:
    """Structural-only dig-in probe (not authoritative).

    True only when the ask literally names a prior spine Latin method id
    or the same locked Class#method. LLM decides all other dig-in cases.
    """
    text = (message or "").strip()
    locked = _class_method_from_hint(locked_cm) or (locked_cm or "").strip()
    if not text or not locked or "#" not in locked:
        return False
    if is_new_topic_ask(text, locked, ctx):
        return False
    explicit = _explicit_cm_in_question(text)
    if explicit and explicit.lower() == locked.lower():
        return True
    return _mentions_prior_spine_latin(text, ctx)


_CONTEXT_LINKAGE_PROMPT = load_prompt("context_linkage_prompt")
def _parse_bool_flag(data: dict[str, Any], key: str) -> bool:
    flag = data.get(key)
    return flag is True or str(flag).strip().lower() in {"1", "true", "yes"}


async def decide_continue_locked_entry(
    question: str,
    locked_cm: str,
    *,
    prior_context: str = "",
    session_ctx: dict[str, Any] | None = None,
    entry_level_ask: bool | None = None,
) -> dict[str, Any]:
    """LLM JSON: context relatedness / continue lock / dig-in (sole authority).

    No keyword or CJK-overlap short-circuit. On timeout/error/uncertain → unlock
    (fail-open new topic) unless a structural Latin dig-in already applies upstream.
    ``entry_level_ask`` is ignored (kept for call-site compatibility).
    """
    import asyncio

    del entry_level_ask  # LLM decides ask_kind; no heuristic gate.
    q = (question or "").strip()
    locked = _class_method_from_hint(locked_cm) or (locked_cm or "").strip()
    overlap = _overlap_prior_answer_score(q, session_ctx)
    default_off: dict[str, Any] = {
        "continue": False,
        "related": False,
        "dig_in": False,
        "ask_kind": "new_topic",
        "reason": "fallback_unlock",
        "source": "default",
        "overlap": overlap,
    }
    if not q or not locked or "#" not in locked:
        return default_off

    try:
        llm = get_llm_provider()

        async def _call() -> str:
            return await llm.chat(
                [
                    {"role": "system", "content": _CONTEXT_LINKAGE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Locked entry: {locked}\n"
                            f"Weak overlap score (feature only, not a decision): {overlap}\n"
                            f"Prior context:\n{truncate(prior_context or '(empty)', 900)}\n\n"
                            f"Question:\n{q}"
                        ),
                    },
                ],
                temperature=0.0,
                max_tokens=160,
            )

        raw = await asyncio.wait_for(_call(), timeout=10.0)
        data: dict[str, Any] = {}
        try:
            m = re.search(r"\{.*\}", (raw or "").strip(), re.DOTALL)
            parsed = json.loads(m.group(0) if m else (raw or ""))
            data = parsed if isinstance(parsed, dict) else {}
        except Exception:
            data = {}
        if not data:
            return {
                **default_off,
                "reason": "empty_llm_unlock",
                "source": "default_unlock",
            }
        related = _parse_bool_flag(data, "related")
        cont = _parse_bool_flag(data, "continue")
        dig = _parse_bool_flag(data, "dig_in")
        keep = related or cont
        ask_kind = str(data.get("ask_kind") or "").strip().lower() or (
            "dig_in" if dig and keep else ("full_flow" if keep else "new_topic")
        )
        if not keep:
            dig = False
            ask_kind = "new_topic"
        return {
            "continue": keep,
            "related": keep,
            "dig_in": bool(dig and keep),
            "ask_kind": ask_kind[:40],
            "reason": str(data.get("reason") or "").strip()[:200],
            "source": "llm",
            "overlap": overlap,
        }
    except asyncio.TimeoutError:
        logger.warning("decide_continue_locked_entry timed out")
        # Conservative keep on timeout when the ask still overlaps the prior
        # answer — fail-open unlock was drifting follow-ups like「放款落库写哪」
        # onto unrelated entries (LoanApprovalJob). Zero/low overlap still unlocks.
        if overlap >= 3 or (_structural_continuation(q) and overlap >= 1):
            return {
                "continue": True,
                "related": True,
                "dig_in": False,
                "ask_kind": "followup",
                "reason": "timeout_conservative_keep",
                "source": "timeout_conservative",
                "overlap": overlap,
            }
        return {
            **default_off,
            "reason": "timeout_unlock",
            "source": "timeout_fail_open_unlock",
        }
    except Exception:
        logger.debug("decide_continue_locked_entry failed", exc_info=True)
        if overlap >= 3 or (_structural_continuation(q) and overlap >= 1):
            return {
                "continue": True,
                "related": True,
                "dig_in": False,
                "ask_kind": "followup",
                "reason": "error_conservative_keep",
                "source": "error_conservative",
                "overlap": overlap,
            }
        return {
            **default_off,
            "reason": "error_unlock",
            "source": "error_fail_open_unlock",
        }


def _structural_continuation(question: str) -> bool:
    """True when the ask opens with a follow-up connector (那/然后/再/接着…).

    Such turns almost always continue the previous topic; used only as a
    conservative keep signal when the relatedness LLM times out.
    """
    t = (question or "").strip()
    return bool(
        re.match(
            r"^(那|那么|然后|再|接着|还有|接下来|那再|继续|上面|刚才|之前那个)\s*",
            t,
        )
    )


def _prior_context_bits_for_lock(
    *,
    session_id: str,
    message: str,
    ctx: dict[str, Any] | None,
) -> str:
    """Compact prior context string for LLM context linkage."""
    prior_bits: list[str] = []
    prior_q_lock = _last_user_question(session_id, message) if session_id else None
    if prior_q_lock:
        prior_bits.append(f"prior_question: {truncate(prior_q_lock, 200)}")
    elif ctx and ctx.get("last_explore_question"):
        prior_bits.append(
            f"prior_question: {truncate(str(ctx.get('last_explore_question')), 200)}"
        )
    flow_lock = [
        str(x) for x in ((ctx or {}).get("last_flow_symbols") or [])[:8] if x
    ]
    if flow_lock:
        prior_bits.append("spine: " + ", ".join(flow_lock))
    step_blobs = _prior_answer_step_blobs(ctx)[:6]
    if step_blobs:
        prior_bits.append(
            "answer_steps: " + " | ".join(truncate(b, 80) for b in step_blobs)
        )
    return "\n".join(prior_bits)


async def resolve_session_ctx_gate(
    message: str,
    ctx: dict[str, Any] | None,
    *,
    session_id: str = "",
    entry_hint: str = "",
    detail: str = "",
    gate_followup: bool | None = None,
    protocol_sync_followup: bool = False,
    composite_followup: bool = False,
) -> dict[str, Any]:
    """Authoritative session lock / dig-in gate (LLM primary).

    Structural-only fast paths (no natural-language keyword lists):
    - no lock → unlocked
    - question names a different Class#method → unlock
    - question names same Class#method or a prior spine Latin method → keep + dig-in
    - system correction marker in detail → keep

    Everything else → LLM ``decide_continue_locked_entry``. Fail-open unlock.
    """
    del gate_followup  # Chinese gate-detail heuristics removed; LLM decides.
    msg = (message or "").strip()
    locked_cm = _locked_flow_entry(ctx)

    if not locked_cm:
        return {
            "use_session_ctx": False,
            "dig_in": False,
            "new_topic": False,
            "continue_decision": None,
            "locked_cm": None,
            "entry_level_ask": False,
        }

    locked_n = _class_method_from_hint(locked_cm) or locked_cm
    explicit = _explicit_cm_in_question(msg)
    latin_dig = _mentions_prior_spine_latin(msg, ctx)
    system_keep = "用户反馈上一答不正确" in (detail or "") or _is_correction(msg)

    if explicit and explicit.lower() != locked_n.lower():
        return {
            "use_session_ctx": False,
            "dig_in": False,
            "new_topic": True,
            "continue_decision": {
                "continue": False,
                "related": False,
                "dig_in": False,
                "ask_kind": "new_topic",
                "reason": "explicit_foreign_class_method",
                "source": "structural",
            },
            "locked_cm": locked_cm,
            "entry_level_ask": False,
        }

    if (explicit and explicit.lower() == locked_n.lower()) or latin_dig:
        return {
            "use_session_ctx": True,
            "dig_in": True,
            "new_topic": False,
            "continue_decision": {
                "continue": True,
                "related": True,
                "dig_in": True,
                "ask_kind": "dig_in",
                "reason": "structural_latin_or_same_cm",
                "source": "structural",
            },
            "locked_cm": locked_cm,
            "entry_level_ask": False,
        }

    if system_keep or protocol_sync_followup or composite_followup:
        return {
            "use_session_ctx": True,
            "dig_in": False,
            "new_topic": False,
            "continue_decision": {
                "continue": True,
                "related": True,
                "dig_in": False,
                "ask_kind": "followup",
                "reason": "caller_structural_keep",
                "source": "structural",
            },
            "locked_cm": locked_cm,
            "entry_level_ask": False,
        }

    # Do NOT treat intent LLM entry_hint == locked_cm as proof of follow-up.
    # Intent often copies the prior entry into entry_hint on a NEW topic
    # (e.g. 用信申请 after 放款), which used to skip the relatedness LLM.
    del entry_hint  # informational only; relatedness is LLM-owned below.

    prior = _prior_context_bits_for_lock(
        session_id=session_id, message=msg, ctx=ctx
    )
    cont_decision = await decide_continue_locked_entry(
        msg,
        locked_n,
        prior_context=prior,
        session_ctx=ctx,
    )
    keep = bool(cont_decision.get("continue") or cont_decision.get("related"))
    # Structural continuation (那/然后/再/接着…) is a keep signal when the ask
    # still overlaps the locked topic — but a drifted dig-in leaf (risk
    # pendingOperaiton) must not swallow an unrelated follow-up (那 MIY 呢 is a
    # withdrawals-chain ask, not a risk one). Zero overlap → unlock & re-resolve.
    if not keep and _structural_continuation(msg):
        if _overlap_prior_answer_score(msg, ctx) >= 1:
            keep = True
            cont_decision = {
                **cont_decision,
                "continue": True,
                "related": True,
                "dig_in": False,
                "ask_kind": "followup",
                "reason": "structural_continuation_keep",
            }
    dig_in = bool(keep and cont_decision.get("dig_in"))
    ask_kind = str(cont_decision.get("ask_kind") or "")
    entry_level = (not keep) or (ask_kind == "full_flow")
    return {
        "use_session_ctx": keep,
        "dig_in": dig_in,
        "new_topic": not keep,
        "continue_decision": cont_decision,
        "locked_cm": locked_cm,
        "entry_level_ask": bool(entry_level),
    }


def _is_gate_detail_followup(message: str, ctx: dict[str, Any] | None) -> bool:
    """Structural-only: show-code / Latin Class#method under lock. No CN keyword lists."""
    text = (message or "").strip()
    locked = _locked_flow_entry(ctx)
    if not text or not locked:
        return False
    from app.agent.question_intent import wants_show_gate_code

    if wants_show_gate_code(text):
        return True
    if _mentions_prior_spine_latin(text, ctx):
        return True
    explicit = _explicit_cm_in_question(text)
    if explicit and explicit.lower() == locked.lower():
        return True
    return False


def _question_has_content_noun(message: str) -> bool:
    """Structural: any Latin identifier or CJK run remains after stripping punctuation.

    No stopword / business vocabulary lists — used only as a weak feature, not a
    session lock gate (LLM decides relatedness).
    """
    text = re.sub(r"[？?！!。．，,、：:；;…\s]+", " ", message or "").strip()
    return bool(
        re.search(r"[\u4e00-\u9fff]{2,}", text) or re.search(r"[A-Za-z]{3,}", text)
    )


async def _meta_reply(state: AgentState) -> AgentState:
    """Answer conversational / history questions without code-scope planning."""
    llm = get_llm_provider()
    prior = _history_before_current(state)
    hist_txt = "\n".join(
        f"{m.get('role')}: {truncate(m.get('content') or '', 800)}" for m in prior[-12:]
    )
    last_user = next(
        (m.get("content") for m in reversed(prior) if m.get("role") == "user"),
        None,
    )
    content = await llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是会话助手。根据对话历史回答用户关于「刚才问了什么 / 再说一遍」的问题。"
                    "直接复述上一轮用户问题；若用户要总结，概括上一轮问答即可。"
                    "禁止输出「修改范围」、禁止规划 Spring Cloud/Feign 改造、不要查代码。"
                    "历史为空时如实说明没有上一轮问题。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"对话历史:\n{hist_txt or '(空)'}\n\n"
                    f"可直接引用的上一轮用户原话: {last_user or '(无)'}\n\n"
                    f"当前用户: {state['message']}"
                ),
            },
        ],
        temperature=0.1,
        max_tokens=800,
    )
    answer = (content or "").strip()
    if not answer and last_user:
        answer = f"你上一问是：{last_user}"
    elif not answer:
        answer = "当前会话里还没有上一轮问题。"
    # Prefer deterministic recall when user clearly asks what they asked
    msg = state.get("message") or ""
    if last_user and any(k in msg for k in ("刚才问", "问了什么", "问你什么", "上一问", "上一个问题")):
        answer = f"你上一问是：{last_user}"
    state["reply"] = answer
    state["intent"] = "meta"
    state["scope"] = []
    state["retrieved"] = []
    return state


async def _general_reply(state: AgentState) -> AgentState:
    """Business/code Q&A without forcing a modification-scope dump."""
    llm = get_llm_provider()
    prior = _history_before_current(state)
    hist_txt = "\n".join(
        f"{m.get('role')}: {truncate(m.get('content') or '', 400)}" for m in prior[-8:]
    )
    retrieved = state.get("retrieved") or []
    hits = []
    for r in retrieved[:10]:
        hits.append(
            {
                "kind": r.get("kind"),
                "fqcn": r.get("fqcn"),
                "method_sig": r.get("method_sig"),
                "file_path": r.get("file_path"),
                "summary": r.get("summary"),
                "tags": r.get("tags"),
            }
        )
    # 通用兜底：检索无命中（非代码问题 / 索引未覆盖）→ 直接 LLM 回答，
    # 保证任何问题都有回复（弱模型只做一次性回答，不做长链推理）。
    if not hits:
        try:
            settings = get_settings()
            fallback_on = bool(settings.general_fallback_enabled)
        except Exception:
            fallback_on = True
        if fallback_on:
            content = await general_chat_reply(
                state.get("detail") or state["message"],
                hist_txt,
                llm,
            )
            state["reply"] = (
                content or "暂时没有检索到相关资料，请换个问法或先完成解析。"
            )
            state["scope"] = state.get("scope") or []
            return state
    content = await llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "你是业务+代码问答助手。用偏业务的中文回答。准确优先，禁止误导。"
                    "默认不要主动给出类名/方法名；仅当用户明确问技术细节时再引用类#方法。"
                    "禁止输出 symbol_id。"
                    "不要默认输出「修改范围」或 Spring Cloud 改造清单，除非用户明确要改代码。"
                    "只依据检索命中中的事实；证据不足写「未在已读材料中看到…」，"
                    "禁止编造表名/字段/接口/外部 URL（如臆造 t_user_quota）。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"历史:\n{hist_txt or '(空)'}\n\n"
                    f"问题: {state.get('detail') or state['message']}\n"
                    f"检索命中: {json.dumps(hits, ensure_ascii=False)}"
                ),
            },
        ],
        temperature=0.2,
        max_tokens=2000,
    )
    state["reply"] = (content or "").strip() or "暂时无法根据现有索引回答，请换个问法或先完成解析。"
    state["scope"] = state.get("scope") or []
    return state


async def _reply(state: AgentState) -> AgentState:
    intent = normalize_intent(state.get("intent"), state.get("message") or "")
    flow = state.get("flow") or {}
    scope = state.get("scope") or []
    retrieved = state.get("retrieved") or []

    if intent in {"ask", "plan"}:
        # Prefer FOLLOWS-structured reply when available (set by _retrieve).
        structured = flow.get("structured")
        if isinstance(structured, dict) and (structured.get("flow_doc") or {}).get("stages"):
            from app.agent.explore_agent import (
                _append_tech_anchor,
                _render_ordered_main_chain,
                _wants_tech_detail,
            )

            tech = _wants_tech_detail(state.get("message") or "")
            main = _render_ordered_main_chain(structured, tech=tech, deep=False)
            state["reply"] = _append_tech_anchor(main, structured) if main else main
            if state["reply"]:
                return state
        nodes = flow.get("nodes") or []
        edges = flow.get("edges") or []
        # If edges are FOLLOWS, present as execution spine (not CALLS dump).
        follows_edges = [e for e in edges if (e.get("type") or "") == "FOLLOWS"]
        if follows_edges and nodes:
            lines = [
                f"## 业务主链（FOLLOWS 执行序；入口: {flow.get('entry')}）",
                f"- 步骤数: {len(nodes)}，FOLLOWS 边: {len(follows_edges)}",
                "",
                "### 主链步骤",
            ]
            for i, n in enumerate(nodes, 1):
                props = n.get("properties") or {}
                cm = props.get("class_method") or n.get("id")
                title = props.get("title") or ""
                desc = truncate(str(props.get("desc") or ""), 160)
                if title and title != cm:
                    lines.append(f"{i}. {cm} — {title}" + (f"：{desc}" if desc else ""))
                else:
                    lines.append(f"{i}. {cm}" + (f"：{desc}" if desc else ""))
            state["reply"] = "\n".join(lines)
            return state
        lines = [
            f"## 流程概览（入口: {flow.get('entry')}）",
            f"- 节点数: {len(nodes)}，边数: {len(edges)}",
            "",
            "### 关键节点",
        ]
        for n in nodes[:20]:
            props = n.get("properties") or {}
            label = ",".join(n.get("labels") or [])
            name = props.get("fqcn") or props.get("name") or props.get("path") or n.get("id")
            lines.append(f"- [{label}] {name}")
        lines.append("")
        lines.append("### 关键边（含异步 TRIGGERS）")
        for e in edges[:30]:
            lines.append(f"- {e.get('start')} -[{e.get('type')}]-> {e.get('end')} via={((e.get('properties') or {}).get('via'))}")
        if retrieved:
            lines.append("")
            lines.append("### 语义检索命中")
            for r in retrieved[:8]:
                lines.append(
                    f"- {r.get('kind')} {r.get('fqcn')}#{(r.get('method_sig') or '').split('#')[-1]} "
                    f"tags={r.get('tags')}"
                )
        state["reply"] = "\n".join(lines)
        return state

    if intent != "modify":
        return await _general_reply(state)

    draft = state.get("_draft") or ""
    lines = [
        f"## 意图: {intent}",
        "",
        "### 修改范围",
    ]
    if state.get("selected_option_id"):
        lines.append(f"- 已选方案: `{state.get('selected_option_id')}`")
    for s in scope[:15]:
        lines.append(
            f"- ({s.get('priority')}) `{s.get('file')}` — {s.get('action')}: {s.get('reason')}"
        )
    if state.get("patches"):
        lines.append("")
        lines.append(f"### 修改文件与行号（{len(state['patches'])} 个文件）")
        lines.extend(summarize_patches_with_lines(list(state["patches"])))
    if state.get("patch_results"):
        lines.append("")
        lines.append("### 应用结果")
        lines.extend(summarize_patch_results(list(state["patch_results"])))
    if draft and not state.get("patches"):
        lines.append("")
        lines.append("### 方案说明")
        lines.append(draft[:4000])
    elif draft:
        lines.append("")
        lines.append("### 详细说明")
        lines.append(truncate(draft, 2500))
    if not scope and draft and not state.get("patches"):
        state["reply"] = draft
    else:
        state["reply"] = "\n".join(lines)
    return state


def _wants_knowledge_base(message: str) -> bool:
    """Only pull global KB when the user explicitly asks for it."""
    text = (message or "").strip()
    if not text:
        return False
    if "知识库" in text:
        return True
    return bool(
        re.search(
            r"(?:按|根据|参考|依据|查阅|查一下|用一下).{0,12}(?:接口文档|上传的文档|文档说明)",
            text,
        )
    )


def _wants_explore(intent: str, message: str) -> bool:
    text = message or ""
    if _wants_meta(intent, text):
        return False
    intent_n = normalize_intent(intent, text)
    if is_explore_intent(intent_n):
        return True
    keys = (
        "流程",
        "链路",
        "校验",
        "字段",
        "参数",
        "入参",
        "必填",
        "怎么走",
        "调用链",
        "哪些检查",
        "哪些字段",
        "改哪些",
        "修改哪些",
        "哪些类",
        "走什么",
        "地址",
        "服务地址",
        "哪个服务",
        "服务名",
        "url",
        "URL",
        "Feign",
        "发下",
        "状态",
        "怎么更新",
        "如何更新",
        "状态更新",
        "状态变更",
        "哪张表",
        "哪个表",
        "实体",
        "落库",
        "谁更新",
        "赋值",
        "DDL",
        "修改方案",
        "几种方案",
        "多种方案",
        "对比方案",
    )
    return any(k in text for k in keys)


def _last_user_question(session_id: str, current: str) -> str | None:
    msgs = session_store.list_messages(session_id)
    if msgs and msgs[-1].get("role") == "user" and (msgs[-1].get("content") or "") == current:
        msgs = msgs[:-1]
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        content = (m.get("content") or "").strip()
        if not content or _is_correction(content):
            continue
        if _wants_meta("general", content):
            continue
        return content
    return None


async def _explore(state: AgentState) -> AgentState:
    ctx = session_store.get_context(state["session_id"])
    msg = (state.get("message") or "").strip()
    detail = state.get("detail") or ""
    # Only bind session last_entry for true follow-ups / corrections / dig-ins.
    # New questions must not reuse prior Controller from an unrelated topic.
    from app.agent.question_intent import (
        assumed_new_field_name,
        gate_themes_in_question,
        wants_add_field_change_plan,
        wants_composite_change,
        wants_extend_field_to_protocol_sync,
        wants_show_gate_code,
    )

    protocol_sync_ask = wants_extend_field_to_protocol_sync(msg)
    composite_ask = wants_composite_change(msg)
    show_code_ask = wants_show_gate_code(msg) or _wants_source_paste(msg)
    # Protocol-sync under an existing lock: keep lock structurally (LLM still
    # decides unrelated new topics when protocol_sync_ask is false).
    protocol_sync_followup = bool(
        protocol_sync_ask
        and (ctx.get("last_class_method") or ctx.get("last_entry"))
    )
    # Composite change under prior entry lock: keep when session has a lock.
    composite_followup = bool(
        composite_ask
        and (ctx.get("last_class_method") or ctx.get("last_entry"))
    )
    # LLM-primary context gate (no CN keyword dig-in / new-topic lists).
    lock_gate = await resolve_session_ctx_gate(
        msg,
        ctx,
        session_id=state["session_id"],
        entry_hint=str(state.get("entry_hint") or ""),
        detail=detail,
        gate_followup=False,
        protocol_sync_followup=protocol_sync_followup,
        composite_followup=composite_followup,
    )
    use_session_ctx = bool(lock_gate.get("use_session_ctx"))
    dig_in = bool(lock_gate.get("dig_in"))
    new_topic = bool(lock_gate.get("new_topic"))
    locked_cm = lock_gate.get("locked_cm")
    # Structural override: show-code / source-view requests always retain
    # session context when a locked entry exists.  The LLM-based gate often
    # misclassifies bare Chinese "看看源码" as a new topic, which clears
    # entry_hint and loses flow_symbols / prefer_cms that source_view needs
    # to locate the correct method from the prior turn.
    if not use_session_ctx and locked_cm and _is_gate_detail_followup(msg, ctx):
        use_session_ctx = True
        dig_in = True
        new_topic = False
        lock_gate = {
            **lock_gate,
            "use_session_ctx": True,
            "dig_in": True,
            "new_topic": False,
            "continue_decision": {
                **lock_gate.get("continue_decision", {}),
                "continue": True,
                "related": True,
                "dig_in": True,
                "ask_kind": "dig_in",
                "reason": "structural_show_code_keep",
                "source": "structural",
            },
        }
    if state.get("turn_stage_isolated"):
        use_session_ctx = False
        new_topic = True
        dig_in = False
    # Surface relatedness gate early (desktop chat log / SSE).
    cont = lock_gate.get("continue_decision") or {}
    await _emit_progress(
        state,
        {
            "type": "step",
            "status": "done",
            "step_id": "session_gate",
            "note": (
                f"会话关联 → related={bool(cont.get('related') or use_session_ctx)} "
                f"use_lock={use_session_ctx} dig_in={dig_in} "
                f"ask_kind={cont.get('ask_kind') or ('dig_in' if dig_in else ('followup' if use_session_ctx else 'new_topic'))} "
                f"src={cont.get('source') or 'n/a'} "
                f"locked={locked_cm or '(none)'}"
            ),
            "related": bool(cont.get("related") or use_session_ctx),
            "use_session_ctx": use_session_ctx,
            "dig_in": dig_in,
            "new_topic": new_topic,
            "locked_cm": locked_cm,
            "continue_decision": cont,
        },
    )
    # New topic: drop intent-copied Class#method so PEV cannot soft-pin last entry.
    if not use_session_ctx:
        intent_cm0 = _class_method_from_hint(str(state.get("entry_hint") or ""))
        explicit0 = _explicit_cm_in_question(msg)
        if intent_cm0 and (
            not explicit0 or explicit0.lower() != intent_cm0.lower()
        ):
            state["entry_hint"] = ""
    # From lock_gate / structural show-code·Latin probes — not CN keyword lists.
    gate_followup = bool(dig_in) or _is_gate_detail_followup(msg, ctx)
    question = msg
    # Prefer user message for routing; detail is only a soft paraphrase and must
    # not replace HTTP path / method tokens (LLM detail often drifts to 通知).
    if detail and detail.strip() and detail.strip() != msg:
        question = f"{msg}\n（需求复述：{truncate(detail, 240)}）"
    # Lexical lock from the *user message* first.
    lexical = _lexical_main_entry(msg) or _lexical_main_entry(question)
    entry_hint: str | None = None
    if lexical and _wants_flow_downstream_apis(msg or detail):
        entry_hint = lexical
        use_session_ctx = False
    elif use_session_ctx:
        # Prefer Class#method lock from prior turn over bare Controller FQCN.
        entry_hint = (
            ctx.get("last_class_method")
            or ctx.get("last_entry")
            or None
        )
    if use_session_ctx:
        # Gate dig-ins / follow-ups keep the lock even when lexical is set —
        # do NOT append「新问题：不要沿用…」.
        prior_q = _last_user_question(state["session_id"], state["message"])
        if prior_q and prior_q != msg:
            if _wants_flow_downstream_apis(msg):
                question = (
                    f"{question}\n"
                    f"（追问：承接上一问「{truncate(prior_q, 120)}」的锁定入口 "
                    f"{entry_hint or '上一问已定位的入口'}；"
                    f"只列该主链下游调用的 Service/Feign/Client/MQ，"
                    f"禁止罗列同 Controller 兄弟 HTTP 接口。）"
                )
            elif gate_followup:
                question = (
                    f"{question}\n"
                    f"（追问：承接锁定入口 {entry_hint or '上一问入口'} 主链上的业务关卡；"
                    f"只围绕该入口 CALLS 闭包内与用户本题相关的方法作答，"
                    f"禁止改答无关的其他业务入口。）"
                )
            else:
                question = (
                    f"{question}\n"
                    f"（追问：请承接上一问主题「{truncate(prior_q, 120)}」；"
                    f"入口优先 {entry_hint or '上一问已定位的 Controller'}，"
                    f"参数校验请读入口方法 same_class_calls。）"
                )
    else:
        # Explicit: ignore prior topic; search by current question literals only.
        question = (
            f"{question}\n"
            "（新问题：不要沿用会话里上一业务入口；按本题实词重新 search_symbols。）"
        )
        if lexical and _wants_flow_downstream_apis(msg or detail):
            question = (
                f"{question}\n"
                f"（已锁定主链入口 {lexical}；只列该 FlowDoc 主链下游 Service/Feign/MQ，"
                f"禁止罗列同模块其他 Controller。）"
            )
    # —— 会话证据继承（双赢）：追问/dig_in 轮注入上一轮已读证据，禁止重复读码 ——
    evidence_block = ""
    if (use_session_ctx or dig_in) and not new_topic:
        try:
            from app.core.config import get_settings as _gs2

            if _gs2().agent_evidence_reuse:
                evidence_block = session_store.get_evidence_context(
                    state["session_id"],
                    max_chars=int(_gs2().evidence_reuse_max_chars or 2400),
                )
        except Exception:
            logger.debug("evidence context load failed", exc_info=True)
    if evidence_block:
        question = f"{question}\n\n{evidence_block}"
        state["evidence_block"] = evidence_block
    # Source-only: drop stale session Controllers, but KEEP lexical API-path locks.
    # Index backend: also pass lexical lock for sync/full-flow Qs so explore
    # does not promote mid-chain Milvus hits (e.g. applyResultDealWith) as entry.
    from app.core.config import get_settings as _gs

    backend = (_gs().explore_backend or "source").strip().lower()
    if backend == "source" and not use_session_ctx:
        entry_hint = lexical
    elif lexical and (
        _wants_sync_entry_first(msg)
        or _wants_full_flow(msg)
        or _wants_flow_downstream_apis(msg or detail)
        or _wants_gate_detail(msg)
        or _wants_source_paste(msg)
    ):
        # First-turn gate dig-in under lexical entry lock (no session lock yet).
        entry_hint = lexical
    # Add-field / protocol-sync / composite: keep lexical entry even on first turn.
    # Change-impact runs as a normal explore tool, not an early short-circuit here.
    if (
        wants_add_field_change_plan(msg)
        or protocol_sync_ask
        or composite_ask
    ) and lexical:
        entry_hint = entry_hint or lexical

    # Inherit field name for deictic「这个字段」follow-ups.
    if (
        protocol_sync_ask or composite_ask
    ) and assumed_new_field_name(msg) == "newField":
        prior_q = _last_user_question(state["session_id"], state["message"])
        if prior_q:
            fname = assumed_new_field_name(prior_q)
            if fname and fname != "newField":
                question = f"{question}\n（字段名：{fname}）"

    # Show-code / 看配置：通用 AI 定位「想看哪一块」→ 读文件返回（不走业务步骤硬编码）。
    result: dict[str, Any] | None = None

    # PEV main loop (test phase): Goal → Plan → Execute → Verify (+ [PEV][TEST] tags).
    try:
        from app.core.config import get_settings as _get_settings

        _pev_on = bool(_get_settings().agent_use_pev)
        _pev_fallback = bool(_get_settings().agent_pev_fallback)
    except Exception:
        _pev_on = False
        _pev_fallback = True
    if _pev_on and result is None:
        intent_pev = normalize_intent(state.get("intent"), msg)
        if intent_pev in {"ask", "plan"} or _wants_explore(
            intent_pev, msg
        ):
            try:
                from app.agent.explore_agent import parse_numbered_answer_steps
                from app.agent.pev_agent import run_pev_pipeline

                hist_msgs = session_store.list_messages(state["session_id"])
                hist_txt = "\n".join(
                    f"{m['role']}: {truncate(m['content'], 300)}"
                    for m in hist_msgs[-8:]
                )
                prior_q = (
                    str(ctx.get("last_explore_question") or "").strip()
                    or _last_user_question(state["session_id"], state["message"])
                )
                if prior_q == msg:
                    prior_q = ""
                answer_steps = list(ctx.get("last_answer_steps") or [])
                if not answer_steps:
                    for m in reversed(hist_msgs):
                        if m.get("role") == "assistant" and (
                            m.get("content") or ""
                        ).strip():
                            answer_steps = parse_numbered_answer_steps(
                                str(m["content"])
                            )
                            break
                flow_syms = list(
                    ctx.get("last_flow_symbols") or ctx.get("last_symbols") or []
                )
                # Dig-in / show_code: prefer intent Class#method + last dig-in focus
                # over session Controller lock (lock stays as entry_hint).
                # New-topic / unlocked turns must NOT reuse prior dig-in leaves
                # (desktop-chat-d667b9c1da0d: wrong prior dig-in leaf as view_hint).
                prefer_cms: list[str] = []
                intent_cm = _class_method_from_hint(
                    str(state.get("entry_hint") or "")
                )
                explicit_cm = _explicit_cm_in_question(msg)
                # Only pin intent Class#method when user named it, or session gate kept lock.
                if intent_cm and "#" in intent_cm and (
                    use_session_ctx
                    or dig_in
                    or (
                        explicit_cm
                        and explicit_cm.lower() == intent_cm.lower()
                    )
                ):
                    prefer_cms.append(intent_cm)
                carry_prior_dig_in = bool(use_session_ctx or dig_in)
                try:
                    if not get_runtime_profile().carry_dig_in_pins:
                        carry_prior_dig_in = bool(dig_in)
                except Exception:
                    pass
                if carry_prior_dig_in:
                    for raw in list(ctx.get("last_dig_in_focus_cms") or []) + list(
                        ctx.get("last_dig_in_related") or []
                    ):
                        cm = _class_method_from_hint(str(raw)) or (
                            str(raw).strip() if "#" in str(raw) else ""
                        )
                        if cm and "#" in cm and cm not in prefer_cms:
                            prefer_cms.append(cm)
                # show_code under dig-in: front-load focus into flow symbols
                if prefer_cms and (
                    show_code_ask
                    or _wants_source_paste(msg)
                    or dig_in
                    or use_session_ctx
                ):
                    flow_syms = list(
                        dict.fromkeys([*prefer_cms, *flow_syms])
                    )[:40]
                kb_block = ""
                kb_hits: list[dict[str, Any]] = []
                if _wants_knowledge_base(msg):
                    try:
                        from app.knowledge.retrieve import (
                            format_kb_context,
                            search_knowledge,
                        )

                        kb_hits = search_knowledge(msg, top_k=4)
                        kb_block = format_kb_context(kb_hits)
                    except Exception:
                        logger.debug("knowledge retrieve skipped", exc_info=True)
                        kb_hits = []
                pev_hist = hist_txt or ""
                if kb_block:
                    pev_hist = (pev_hist + "\n\n" + kb_block).strip()
                ev_block = str(state.get("evidence_block") or "")
                if ev_block:
                    pev_hist = ((pev_hist + "\n\n") if pev_hist else "") + ev_block
                prior_block = str(state.get("turn_prior_block") or "").strip()
                if prior_block:
                    pev_hist = (prior_block + "\n\n" + pev_hist).strip()
                # show_code follow-up: do not let session Controller overwrite a
                # strong intent / dig-in Class#method view target in prefer_cms.
                pev = await run_pev_pipeline(
                    msg,
                    project_id=state["project_id"],
                    history_text=pev_hist or None,
                    prior_question=prior_q or None,
                    entry_hint=entry_hint,
                    lock_entry=bool(
                        use_session_ctx
                        and _class_method_from_hint(str(entry_hint or ""))
                    ),
                    dig_in_likely=bool(dig_in),
                    answer_steps=answer_steps,
                    flow_symbols=flow_syms,
                    prefer_cms=prefer_cms or None,
                    on_progress=state.get("on_progress"),
                )
                if (pev.get("answer") or "").strip():
                    if kb_hits:
                        titles = sorted(
                            {
                                str(h.get("title") or h.get("filename") or "")
                                for h in kb_hits
                                if h.get("title") or h.get("filename")
                            }
                        )
                        if titles:
                            ans = str(pev.get("answer") or "")
                            note = (
                                "\n\n> 知识库参考："
                                + "、".join(f"《{t}》" for t in titles[:5])
                            )
                            if "知识库参考" not in ans:
                                pev["answer"] = ans.rstrip() + note
                    # Persist spine Class#method list for dig-in method matching.
                    pev_ev = pev.get("evidence") or {}
                    pev_calls = [
                        str(x)
                        for x in (pev_ev.get("calls") or [])
                        if x and "#" in str(x)
                    ]
                    if pev_calls:
                        for s in pev.get("steps") or []:
                            if isinstance(s, dict) and s.get("tool") in {
                                "pev_expand_calls",
                                "pev_mysql_enrich",
                                "pev_summarize_flow",
                            }:
                                s["symbols"] = pev_calls[:20]
                                break
                        else:
                            (pev.get("steps") or []).append(
                                {
                                    "step": 50,
                                    "tool": "pev_expand_calls",
                                    "ok": True,
                                    "symbols": pev_calls[:20],
                                    "entry": pev_ev.get("entry"),
                                }
                            )
                    result = pev
            except Exception:
                logger.exception("PEV pipeline failed; falling back")
                result = None

    _allow_fallback = (not _pev_on) or _pev_fallback
    if result is None and _allow_fallback and (
        show_code_ask or _wants_source_paste(msg)
    ):
        try:
            from app.agent.explore_agent import parse_numbered_answer_steps
            from app.agent.source_view import run_source_view

            answer_steps = list(ctx.get("last_answer_steps") or [])
            prior_answer = ""
            hist_msgs = session_store.list_messages(state["session_id"])
            for m in reversed(hist_msgs):
                if m.get("role") == "assistant" and (m.get("content") or "").strip():
                    prior_answer = str(m["content"])
                    if not answer_steps:
                        answer_steps = parse_numbered_answer_steps(prior_answer)
                    break
            hist_txt = "\n".join(
                f"{m['role']}: {truncate(m['content'], 300)}" for m in hist_msgs[-8:]
            )
            flow_syms = list(
                ctx.get("last_flow_symbols") or ctx.get("last_symbols") or []
            )
            prefer_cms: list[str] = []
            intent_cm = _class_method_from_hint(str(state.get("entry_hint") or ""))
            if intent_cm and "#" in intent_cm:
                prefer_cms.append(intent_cm)
            # Source-view under unlocked/new-topic must not inherit prior dig-in.
            if use_session_ctx or dig_in:
                for raw in list(ctx.get("last_dig_in_focus_cms") or []) + list(
                    ctx.get("last_dig_in_related") or []
                ):
                    cm = _class_method_from_hint(str(raw)) or (
                        str(raw).strip() if "#" in str(raw) else ""
                    )
                    if cm and "#" in cm and cm not in prefer_cms:
                        prefer_cms.append(cm)
            if prefer_cms and (use_session_ctx or dig_in or show_code_ask):
                flow_syms = list(dict.fromkeys([*prefer_cms, *flow_syms]))[:40]
            locked = (
                _class_method_from_hint(str(ctx.get("last_class_method") or ""))
                or _class_method_from_hint(str(ctx.get("last_entry") or ""))
                or _class_method_from_hint(entry_hint)
            )
            viewed = await run_source_view(
                state["project_id"],
                msg,
                history_text=hist_txt,
                prior_answer=prior_answer,
                answer_steps=answer_steps,
                flow_symbols=flow_syms,
                entry_cm=locked,
                preferred_cms=prefer_cms or None,
            )
            if viewed.get("ok") and (viewed.get("answer") or "").strip():
                on_progress = state.get("on_progress")
                if on_progress:
                    try:
                        plan_ev = {
                            "type": "plan",
                            "plan": {
                                "kind": "source_view",
                                "title": f"定位并贴出 {viewed.get('target') or ''}",
                                "steps": [
                                    {
                                        "id": "locate",
                                        "title": "AI 分析要看的内容",
                                        "status": "done",
                                        "note": (viewed.get("plan") or {}).get(
                                            "reason"
                                        )
                                        or "",
                                    },
                                    {
                                        "id": "fetch",
                                        "title": "读取文件/方法",
                                        "status": "done",
                                    },
                                ],
                            },
                        }
                        maybe = on_progress(plan_ev)
                        if hasattr(maybe, "__await__"):
                            await maybe  # type: ignore[misc]
                    except Exception:
                        pass
                result = {
                    "answer": viewed["answer"],
                    "steps": [
                        {
                            "step": 0,
                            "tool": "source_view",
                            "ok": True,
                            "target": viewed.get("target"),
                            "view_kind": viewed.get("view_kind"),
                            "plan": viewed.get("plan"),
                        }
                    ],
                    "retrieved": [],
                    "plan": {
                        "kind": "source_view",
                        "title": f"贴出 {viewed.get('target') or ''}",
                        "source_view": viewed.get("plan"),
                    },
                    "review": {"ok": True, "source": "source_view"},
                }
        except Exception:
            logger.exception("source_view failed; falling back")
            result = None

    # Ask-only secondary path (PEV+skills is primary when agent_use_pev).
    # Fail soft → ask_retrieval, then explore_agent as last resort (KIND path deprecated).
    # Skip ask_retrieval for list/paste modes — that pipeline narrates one flow
    # and strips Class#method, which is wrong for「接口有哪些」/「看代码」.
    intent_now = normalize_intent(state.get("intent"), msg)
    skip_ask_retrieval = bool(
        _wants_api_inventory(msg)
        or _wants_source_paste(msg)
        or show_code_ask
        or wants_add_field_change_plan(msg)
        or result is not None
    )
    if (
        result is None
        and _allow_fallback
        and intent_now == "ask"
        and not skip_ask_retrieval
    ):
        try:
            from app.agent.ask_retrieval import run_ask_retrieval_pipeline

            # Generic ask path: always pass history; no business lock / keyword routers.
            prior_q = (
                str(ctx.get("last_explore_question") or "").strip()
                or _last_user_question(state["session_id"], state["message"])
            )
            if prior_q == msg:
                prior_q = ""
            hist_msgs = session_store.list_messages(state["session_id"])
            hist_txt = "\n".join(
                f"{m['role']}: {truncate(m['content'], 300)}"
                for m in hist_msgs[-8:]
            )
            if prior_q and prior_q not in (hist_txt or ""):
                hist_txt = f"user: {truncate(prior_q, 300)}\n{hist_txt or ''}".strip()
            ev_block = str(state.get("evidence_block") or "")
            if ev_block:
                hist_txt = f"{hist_txt or ''}\n\n{ev_block}".strip()
            result = await run_ask_retrieval_pipeline(
                msg,
                project_id=state["project_id"],
                entry_hint=entry_hint,
                prior_question=prior_q or None,
                history_text=hist_txt or None,
                on_progress=state.get("on_progress"),
            )
            if not (result.get("answer") or "").strip():
                result = None
        except Exception:
            logger.exception("ask_retrieval pipeline failed; falling back to explore_agent")
            result = None

    # Last-resort explore_agent (task KIND planner deprecated; prefer PEV skills).
    # Step budget is adaptive inside explore_agent (≈8–14); do not inflate here.
    if result is None and _allow_fallback:
        result = await run_explore_agent(
            question,
            project_id=state["project_id"],
            entry_hint=entry_hint,
            on_progress=state.get("on_progress"),
        )
    if result is None:
        result = {
            "answer": (
                "未能完成检索：主路径未产出答案"
                + ("，且已关闭 ask_retrieval/explore 回退。" if not _allow_fallback else "。")
            ),
            "steps": [],
            "plan": {"kind": "pev", "title": "PEV"},
            "review": {"ok": False, "source": "pev"},
        }
    reply = result.get("answer") or "未生成回复"
    from app.agent.explore_answers import strip_answer_chrome
    from app.agent.explore_agent import (
        _sanitize_add_field_change_plan,
        _sanitize_protocol_sync_plan,
    )

    reply = strip_answer_chrome(reply)

    if wants_extend_field_to_protocol_sync(question) or protocol_sync_ask:
        reply = _sanitize_protocol_sync_plan(reply, question=question) or reply
    elif wants_composite_change(question) or wants_add_field_change_plan(question):
        reply = _sanitize_add_field_change_plan(reply, question=question) or reply
    state["reply"] = reply
    state["explore_steps"] = result.get("steps") or []
    if result.get("plan"):
        state["research_plan"] = result["plan"]
    if result.get("review"):
        state["answer_review"] = result["review"]
    intent_now = normalize_intent(state.get("intent"), state.get("message") or "")
    if intent_now == "plan":
        state["phase"] = "analyzing" if state.get("need_options") else "idle"
    else:
        state["phase"] = "idle"
    # Surface ask-retrieval neo4j_query on research_plan for debug/UI.
    if result.get("neo4j_query") and isinstance(state.get("research_plan"), dict):
        state["research_plan"] = {
            **state["research_plan"],
            "neo4j_query": result.get("neo4j_query"),
        }
    elif result.get("neo4j_query"):
        state["research_plan"] = {
            **(result.get("plan") or {}),
            "neo4j_query": result.get("neo4j_query"),
        }
    # normalize retrieved for ChatResponse
    retrieved = []
    last_entry = entry_hint
    last_class_method = _class_method_from_hint(entry_hint) or ctx.get("last_class_method")
    last_gate = ctx.get("last_gate")
    last_themes: list[str] = list(ctx.get("last_themes") or [])
    last_symbols: list[str] = list(ctx.get("last_symbols") or [])
    entry_locked = False
    gate_locked = False
    for s in result.get("steps") or []:
        if not entry_locked:
            anchor = s.get("entry_anchor") or s.get("entry")
            cm = _class_method_from_hint(str(anchor) if anchor else None)
            if cm:
                last_class_method = cm
                last_entry = cm
                entry_locked = True
        # Harvest symbols from any explore step (no domain tool names required).
        syms = s.get("symbols") or []
        if s.get("target") and not syms:
            syms = [s.get("target")]
        if isinstance(syms, list) and syms:
            last_symbols = [str(x) for x in syms if x and "#" in str(x)][:8]
            gate_locked = True
        if entry_locked and gate_locked:
            break
    # E: do not overwrite session lock with a drifted/failed resolve vs view_hint.
    pev_ev_early = (
        result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    )
    review_ok = bool((result.get("review") or {}).get("ok"))
    hint_keep = _class_method_from_hint(
        str(pev_ev_early.get("view_hint") or "")
    ) or _class_method_from_hint(str(entry_hint or ""))
    resolved_cm = _class_method_from_hint(str(pev_ev_early.get("entry") or ""))
    prior_cm = _class_method_from_hint(str(ctx.get("last_class_method") or ""))
    if pev_ev_early.get("lock_entry") and (
        hint_keep or prior_cm or _class_method_from_hint(str(entry_hint or ""))
    ):
        keep = (
            _class_method_from_hint(str(entry_hint or ""))
            or hint_keep
            or prior_cm
        )
        if keep:
            last_class_method = keep
            last_entry = keep
            entry_locked = True
    elif (
        hint_keep
        and resolved_cm
        and hint_keep.lower() != resolved_cm.lower()
        and not review_ok
    ):
        last_class_method = prior_cm or hint_keep
        last_entry = last_class_method
        entry_locked = True
    # General: persist Class#method citations from the answer into session symbols.
    if reply and not last_symbols:
        harvested = re.findall(
            r"[A-Za-z][A-Za-z0-9_]{2,80}#[A-Za-z][A-Za-z0-9_]{1,80}",
            reply,
        )
        if harvested:
            # Prefer ServiceImpl#… over Controller when both appear.
            ranked = sorted(
                dict.fromkeys(harvested),
                key=lambda cm: (
                    0 if "ServiceImpl#" in cm else 1,
                    len(cm),
                ),
            )
            last_symbols = ranked[:8]
    for h in result.get("retrieved") or []:
        if not h or not h.get("symbol_id"):
            continue
        retrieved.append(
            {
                "symbol_id": h.get("symbol_id"),
                "project_id": state["project_id"],
                "fqcn": h.get("fqcn") or "",
                "method_sig": h.get("method_sig") or h.get("signature"),
                "kind": h.get("kind") or "Class",
                "file_path": h.get("file_path") or "",
                "summary": h.get("summary"),
                "tags": h.get("tags") or [],
                "score": h.get("score"),
            }
        )
        kind = (h.get("kind") or "")
        fqcn = h.get("fqcn") or ""
        if not last_class_method and (
            kind == "Controller" or fqcn.endswith("Controller") or "Controller" in fqcn
        ):
            last_entry = fqcn
    state["retrieved"] = retrieved
    # 记录本轮证据（evidence_seen）供下一轮追问复用（双赢：省重复读码 + 口径不漂移）。
    try:
        ev_entries: list[dict[str, Any]] = []
        for r in retrieved:
            cm = str(r.get("method_sig") or "").strip()
            if not cm or "#" not in cm:
                fq = str(r.get("fqcn") or "").strip()
                if "Controller" in fq or "Service" in fq:
                    cm = fq
            if not cm:
                continue
            ev_entries.append(
                {
                    "cm": cm,
                    "kind": str(r.get("kind") or ""),
                    "summary": str(r.get("summary") or ""),
                    "file_path": str(r.get("file_path") or ""),
                }
            )
        if ev_entries:
            session_store.record_evidence(state["session_id"], ev_entries)
    except Exception:
        logger.debug("record evidence failed", exc_info=True)
    ctx_update: dict[str, Any] = {"last_explore_question": state.get("message")}
    if last_entry:
        ctx_update["last_entry"] = last_entry
    if last_class_method:
        ctx_update["last_class_method"] = last_class_method
        ctx_update["last_entry"] = last_class_method
    if last_gate:
        ctx_update["last_gate"] = last_gate
    if last_themes:
        ctx_update["last_themes"] = last_themes
    if last_symbols:
        ctx_update["last_symbols"] = last_symbols
    # Persist numbered answer steps + ordered flow symbols for「第N步代码」follow-ups.
    from app.agent.explore_agent import parse_numbered_answer_steps

    answer_steps = parse_numbered_answer_steps(reply or "")
    if answer_steps:
        ctx_update["last_answer_steps"] = answer_steps
    flow_syms: list[str] = []
    pev_ev = result.get("evidence") if isinstance(result.get("evidence"), dict) else {}
    if pev_ev.get("calls"):
        flow_syms = [
            str(x) for x in (pev_ev.get("calls") or []) if x and "#" in str(x)
        ][:20]
    if not flow_syms:
        for s in result.get("steps") or []:
            syms = s.get("symbols") or []
            if isinstance(syms, list) and syms and (
                s.get("tool")
                in {
                    "mysql_enrich",
                    "neo4j_query",
                    "ask_mysql",
                    "pev_expand_calls",
                    "pev_mysql_enrich",
                }
                or len(syms) >= 3
            ):
                flow_syms = [str(x) for x in syms if x and "#" in str(x)]
                break
    if not flow_syms and last_symbols:
        flow_syms = list(last_symbols)
    if flow_syms:
        ctx_update["last_flow_symbols"] = flow_syms[:20]
        if not last_symbols:
            ctx_update["last_symbols"] = flow_syms[:8]
    # Persist dig-in focus so show_code follow-ups keep the same Class#method.
    if pev_ev.get("dig_in_focus_cms"):
        dig_cms = [
            str(x)
            for x in (pev_ev.get("dig_in_focus_cms") or [])
            if x and "#" in str(x)
        ][:8]
        if dig_cms:
            ctx_update["last_dig_in_focus_cms"] = dig_cms
            related = [
                str(x)
                for x in (pev_ev.get("dig_in_related") or [])
                if x and "#" in str(x)
            ][:12]
            if related:
                ctx_update["last_dig_in_related"] = related
            # Front-load focus + related into flow symbols for next turn.
            merged = list(
                dict.fromkeys(
                    [*dig_cms, *related, *(ctx_update.get("last_flow_symbols") or flow_syms)]
                )
            )[:20]
            ctx_update["last_flow_symbols"] = merged
            if not last_symbols:
                ctx_update["last_symbols"] = merged[:8]
    elif pev_ev.get("prefer_cms") and (
        show_code_ask or _wants_source_paste(state.get("message") or "")
    ):
        # show_code may set prefer without dig_in_focus flag
        pref = [
            str(x)
            for x in (pev_ev.get("prefer_cms") or [])
            if x and "#" in str(x)
        ][:8]
        if pref:
            ctx_update["last_dig_in_focus_cms"] = pref
    elif new_topic or not use_session_ctx:
        # Switched entry / new topic: drop stale dig-in so the next turn cannot
        # re-inject foreign leaves (e.g. stale dig-in Class#method).
        ctx_update["last_dig_in_focus_cms"] = []
        ctx_update["last_dig_in_related"] = []
    if (
        last_entry
        or last_class_method
        or last_gate
        or last_symbols
        or answer_steps
        or flow_syms
        or ctx_update.get("last_dig_in_focus_cms")
    ):
        session_store.update_context(state["session_id"], ctx_update)
    # expose explore trail in scope for UI/debug
    state["scope"] = [
        {
            "file": f"step{s.get('step')}:{s.get('tool')}",
            "reason": json.dumps(s.get("args") or {}, ensure_ascii=False)[:200],
            "action": "explore",
            "priority": s.get("step") or 0,
        }
        for s in state["explore_steps"]
    ]
    return state


def _load_locked_scaffold_patches(state: AgentState) -> list[dict[str, Any]]:
    """Strict file list from CTA / prior scaffold reply — never invent extra paths."""
    sid = state.get("session_id") or ""
    ctx = session_store.get_context(sid) if sid else {}
    for key in ("pending_patches", "locked_patches", "last_scaffold_patches"):
        raw = ctx.get(key)
        if isinstance(raw, list) and raw:
            out = [p for p in raw if isinstance(p, dict) and (p.get("file") or p.get("path"))]
            if out:
                return out
    # Scan recent assistant messages (not only the last short “已写入” reply).
    for m in reversed(session_store.list_messages(sid) or []):
        if m.get("role") != "assistant":
            continue
        scaffolds = extract_scaffold_file_patches(str(m.get("content") or ""))
        if scaffolds:
            return scaffolds
    return []


def _persist_locked_patches(state: AgentState, patches: list[dict[str, Any]]) -> None:
    if not patches:
        return
    allow = allowlist_from_patches(patches)
    session_store.update_context(
        state["session_id"],
        {
            "pending_patches": patches,
            "locked_patches": patches,
            "last_scaffold_patches": patches,
            "locked_file_allowlist": allow,
            "pending_options": state.get("options") or [],
            "phase": state.get("phase") or "awaiting_option",
        },
    )


async def _run_implement_pipeline(state: AgentState) -> AgentState:
    # List-only / no-write: LLM said want_apply=false → plan path, not patches.
    if not state.get("want_apply"):
        state["intent"] = "plan"
        state = await _explore(state)
        if not (state.get("reply") or "").strip():
            state = await _retrieve(state)
            state = await _scope_plan(state)
            state = await _general_reply(state)
        return _attach_implement_cta(state)

    state["intent"] = "modify"
    state["phase"] = "implementing"
    user_msg = str(state.get("message") or "")
    from app.agent.remote_loop import _goal_needs_followup

    with_follow = "，并按完整目标继续后续步骤" if _goal_needs_followup(user_msg) else ""
    await _emit_progress(
        state,
        {
            "type": "step",
            "status": "running",
            "step_id": "implement",
            "note": f"开始落地改代码（仅清单内文件{with_follow}）…",
        },
    )

    # Hard rule: prefer locked scaffold list; never open-ended LLM expand when list exists.
    locked = list(state.get("patches") or []) if state.get("patches") else _load_locked_scaffold_patches(state)
    if locked and not state.get("_force_regen_patches"):
        state["patches"] = locked
        allow = allowlist_from_patches(locked)
        state["scope"] = [
            {"file": f, "reason": "上一轮确认清单（禁止扩改）", "action": "write", "priority": i + 1}
            for i, f in enumerate(allow)
        ]
        await _emit_progress(
            state,
            {
                "type": "step",
                "status": "done",
                "step_id": "reuse_scaffold",
                "note": f"严格按清单写入 {len(allow)} 个文件（不会改清单外文件）",
            },
        )
    elif not state.get("_force_regen_patches") and state.get("client_apply"):
        # No prior list, but user asked to modify now (e.g. 按接口文档直接改):
        # first generate scaffold/list from KB, then apply ONLY those files — never open retrieve→乱改.
        await _emit_progress(
            state,
            {
                "type": "step",
                "status": "running",
                "step_id": "build_list",
                "note": "尚未有确认清单：先按接口文档/知识库生成修改清单…",
            },
        )
        state["intent"] = "plan"
        # Temporarily clear want_apply so explore stays list-only.
        state["want_apply"] = False
        state = await _explore(state)
        if not (state.get("reply") or "").strip():
            state = await _retrieve(state)
            state = await _scope_plan(state)
            state = await _general_reply(state)
        state = _attach_implement_cta(state)
        scaffolds = extract_scaffold_file_patches(state.get("reply") or "") or _load_locked_scaffold_patches(
            state
        )
        if not scaffolds:
            await _emit_progress(
                state,
                {
                    "type": "step",
                    "status": "error",
                    "step_id": "build_list",
                    "note": "未能解析出明确文件清单，已拒绝改代码",
                },
            )
            state["patches"] = []
            state["phase"] = "awaiting_option"
            state["want_apply"] = False
            reply = (state.get("reply") or "").rstrip()
            state["reply"] = (
                (reply + "\n\n" if reply else "")
                + "未能从接口文档解析出**明确要改的文件清单**，为避免改多/改错，本轮未修改任何代码。\n"
                "请确认知识库文档已就绪，或先只要清单，确认后再改。"
            )
            return state
        _persist_locked_patches(state, scaffolds)
        state["patches"] = scaffolds
        state["want_apply"] = True
        allow = allowlist_from_patches(scaffolds)
        state["scope"] = [
            {"file": f, "reason": "本轮按文档生成的清单（禁止扩改）", "action": "write", "priority": i + 1}
            for i, f in enumerate(allow)
        ]
        await _emit_progress(
            state,
            {
                "type": "step",
                "status": "done",
                "step_id": "build_list",
                "note": f"已生成清单 {len(allow)} 个文件，仅写入这些文件",
            },
        )
        # Keep the generated list visible in the reply before apply note.
        list_reply = (state.get("reply") or "").rstrip()
        if list_reply and "按确认清单写入" not in list_reply:
            state["_list_reply_prefix"] = list_reply
    else:
        # Server-side / explicit regen: still constrain to real scope files when present.
        try:
            from langgraph.graph import END, StateGraph

            g = StateGraph(AgentState)
            g.add_node("retrieve", _retrieve)
            g.add_node("scope_plan", _scope_plan)
            g.add_node("patch", _patch_fixed)
            g.add_node("verify", _verify)
            g.add_node("reply", _reply)
            g.set_entry_point("retrieve")
            g.add_edge("retrieve", "scope_plan")
            g.add_edge("scope_plan", "patch")
            g.add_edge("patch", "verify")
            g.add_edge("verify", "reply")
            g.add_edge("reply", END)
            app = g.compile()
            state = await app.ainvoke(state)
        except Exception:
            state = await _retrieve(state)
            state = await _scope_plan(state)
            state = await _patch_fixed(state)
            state = await _verify_patch_writes(state)
            state = await _reply(state)
        # Drop over-scoped patches not in real scope allowlist.
        allow = real_scope_files(state.get("scope"))
        ctx_allow = session_store.get_context(state["session_id"]).get("locked_file_allowlist")
        if isinstance(ctx_allow, list) and ctx_allow:
            allow = [str(x) for x in ctx_allow if x]
        if allow:
            before = len(state.get("patches") or [])
            state["patches"] = filter_patches_to_allowlist(
                list(state.get("patches") or []), allow, drop_if_empty_allowlist=True
            )
            dropped = before - len(state.get("patches") or [])
            if dropped > 0:
                await _emit_progress(
                    state,
                    {
                        "type": "step",
                        "status": "done",
                        "step_id": "scope_guard",
                        "note": f"已丢弃清单外补丁 {dropped} 个，仅保留清单文件",
                    },
                )

    hints = [
        str(r.get("file_path") or "")
        for r in (state.get("retrieved") or [])
        if r.get("file_path")
    ] + [
        str(s.get("file") or "")
        for s in (state.get("scope") or [])
        if isinstance(s, dict) and s.get("file")
    ]
    # Prefer allowlist as path hints so normalize cannot wander.
    allow_hints = allowlist_from_patches(state.get("patches")) or real_scope_files(state.get("scope"))
    state["patches"] = normalize_patch_paths(
        list(state.get("patches") or []),
        hints=allow_hints or hints,
        project_id=state.get("project_id"),
    )
    if allow_hints:
        state["patches"] = filter_patches_to_allowlist(
            list(state.get("patches") or []), allow_hints, drop_if_empty_allowlist=True
        )

    # Client applies patches when want_apply; compile only when want_verify.
    # Remaining goal parts (branch/commit/push/…) are decided by remote_loop LLM.
    if state.get("client_apply") and state.get("patches") and state.get("want_apply"):
        from app.agent.remote_loop import _goal_needs_followup, create_patch_agent_run

        with_compile = bool(state.get("want_verify"))
        user_msg = str(state.get("message") or "").strip()
        detail = str(state.get("detail") or "").strip()
        files = allowlist_from_patches(state.get("patches"))
        follow = _goal_needs_followup(user_msg) or _goal_needs_followup(detail)
        allow_note = (
            f"硬约束：仅允许写入这 {len(files)} 个文件"
            + (f"：{', '.join(files[:20])}" if files else "")
            + "；禁止修改清单外任何文件。"
        )
        goal_core = user_msg or detail or "按确认清单写入本机"
        goal = f"{goal_core}\n\n{allow_note}"
        if len(goal) > 900:
            goal = goal[:900]
        run = create_patch_agent_run(
            patches=list(state.get("patches") or []),
            session_id=state.get("session_id"),
            project_id=state.get("project_id"),
            goal=goal,
            max_repairs=5 if follow else 3,
            path_hints=files,
            with_compile=with_compile,
            continue_after_write=follow if follow else None,
        )
        state["client_run_id"] = run.id
        state["phase"] = "client_verifying" if with_compile else "client_applying"
        if follow and with_compile:
            extra = (
                f"\n\n已创建本机任务 `{run.id}`："
                "先写入清单内文件并编译，再按你的完整目标继续编排"
                "（如新建分支/提交/推送等；需确认的步骤会弹窗）。"
            )
        elif follow:
            extra = (
                f"\n\n已创建本机任务 `{run.id}`："
                "先写入清单内文件，再按你的完整目标继续编排"
                "（如新建分支/提交/推送等；需确认的步骤会弹窗）。"
            )
        elif with_compile:
            extra = (
                f"\n\n已创建本机改码验证任务 `{run.id}`："
                "客户端将**仅**应用清单内补丁 → 编译验证（不扩改其它文件）。"
            )
        else:
            extra = (
                f"\n\n已创建本机改码任务 `{run.id}`："
                "客户端将**仅**写入清单内文件，不编译、不自测、不改清单外文件。"
            )
        if not (state.get("reply") or "").strip():
            state["reply"] = (
                "按确认清单写入本机（仅以下文件）："
                + ("、".join(f"`{f}`" for f in files[:12]) if files else "（空清单）")
                + ("，并继续完成你提到的后续步骤。" if follow else "。")
            )
        prefix = state.pop("_list_reply_prefix", None)
        if prefix and prefix not in (state.get("reply") or ""):
            state["reply"] = prefix.rstrip() + "\n\n---\n\n" + (state.get("reply") or "")
        state["reply"] = (state.get("reply") or "") + extra
        # Keep locked_patches for re-apply; clear only pending queue pointer.
        session_store.update_context(
            state["session_id"],
            {
                "pending_patches": list(state.get("patches") or []),
                "locked_patches": list(state.get("patches") or []),
                "last_scaffold_patches": list(state.get("patches") or []),
                "locked_file_allowlist": files,
                "phase": state["phase"],
            },
        )
        return state

    # Server workspace is never written anymore (apply_patch removed); client apply
    # path returns above, so this is a list/plan turn.
    state["phase"] = "done" if state.get("patches") else "idle"
    return state


async def _run_options_pipeline(state: AgentState) -> AgentState:
    state["intent"] = "plan"
    state["need_options"] = True
    state["phase"] = "analyzing"
    state = await _explore(state)
    options = await build_change_options(
        question=state.get("detail") or state.get("message") or "",
        explore_answer=state.get("reply") or "",
        scope=state.get("scope") or [],
        explore_steps=state.get("explore_steps") or [],
        count=3,
    )
    state["options"] = options
    state["phase"] = "awaiting_option"
    state["reply"] = render_options_reply(
        options, question=state.get("message") or ""
    )
    on_progress = state.get("on_progress")
    if on_progress:
        try:
            maybe = on_progress({"type": "options", "options": options, "phase": "awaiting_option"})
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            pass
    return state


async def _run_verify_repair_loop(
    state: AgentState, *, auto: bool = False
) -> AgentState:
    """Compile + test; on failure repair with LLM patches up to N times."""
    on_progress = state.get("on_progress")
    try:
        max_repairs = int(get_runtime_profile().max_repairs)
    except Exception:
        max_repairs = DEFAULT_MAX_REPAIRS
    ctx = session_store.get_context(state["session_id"])
    repair_count = int(ctx.get("repair_count") or 0)
    if not auto:
        repair_count = 0

    changed = [
        str(p.get("file"))
        for p in (state.get("patches") or [])
        if isinstance(p, dict) and p.get("file")
    ]
    attempt = 1
    state["phase"] = "verifying"
    report = await run_verify_once(
        changed_files=changed, on_progress=on_progress, attempt=attempt
    )
    state["verify_report"] = report

    while (not report.get("ok")) and repair_count < max_repairs:
        repair_count += 1
        state["phase"] = "repairing"
        state["intent"] = "execute"
        await _emit_verify_progress(
            on_progress,
            {
                "type": "repair",
                "status": "running",
                "note": f"根据报错自动修复（第 {repair_count}/{max_repairs} 次）",
                "attempt": repair_count,
            },
        )
        llm = get_llm_provider()
        prompt = build_repair_prompt(
            requirement=state.get("detail") or state.get("message") or "",
            verify_report=report,
            patches=state.get("patches") or [],
            scope=state.get("scope") or [],
        )
        content = await llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "你是资深 Java 工程师。只输出补丁块修复编译/测试错误。\n"
                        "补丁格式：\n### FILE: relative/path.java\n*** OLD\n旧代码\n*** NEW\n新代码\n"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=3500,
        )
        patches = normalize_patch_paths(
            parse_patches_from_llm(content or ""),
            hints=[
                str(r.get("file_path") or "")
                for r in (state.get("retrieved") or [])
                if r.get("file_path")
            ]
            + [
                str(s.get("file") or "")
                for s in (state.get("scope") or [])
                if isinstance(s, dict) and s.get("file")
            ],
            project_id=state.get("project_id"),
        )
        if not patches:
            await _emit_verify_progress(
                on_progress,
                {
                    "type": "repair",
                    "status": "failed",
                    "note": "未能生成修复补丁",
                    "attempt": repair_count,
                },
            )
            break
        state["patches"] = patches
        state["patch_results"] = apply_patch_list(patches)
        await _emit_verify_progress(
            on_progress,
            {
                "type": "repair",
                "status": "done",
                "note": f"已应用修复补丁 {len(patches)} 个文件",
                "attempt": repair_count,
            },
        )
        attempt += 1
        state["phase"] = "verifying"
        report = await run_verify_once(
            changed_files=[
                str(p.get("file"))
                for p in patches
                if isinstance(p, dict) and p.get("file")
            ],
            on_progress=on_progress,
            attempt=attempt,
        )
        state["verify_report"] = report

    state["verify_report"] = {**(state.get("verify_report") or {}), "repair_count": repair_count}
    if report.get("ok"):
        state["phase"] = "done"
        base = state.get("reply") or ""
        vr = render_verify_reply(report, repaired=repair_count > 0)
        state["reply"] = (base + "\n\n" + vr).strip() if base else vr
    else:
        pending = state.get("options") or ctx.get("pending_options") or []
        if pending:
            state["options"] = list(pending)
            state["phase"] = "awaiting_option"
            state["reply"] = (
                (state.get("reply") or "")
                + "\n\n"
                + render_verify_reply(report, repaired=repair_count > 0)
                + "\n\n验证仍失败。可换一个方案重试，或发送「回滚」撤销本次改动。"
            ).strip()
        else:
            state["phase"] = "failed"
            state["reply"] = (
                (state.get("reply") or "")
                + "\n\n"
                + render_verify_reply(report, repaired=repair_count > 0)
                + "\n\n已达自动修复上限。可发送「回滚」撤销，或补充说明后重试。"
            ).strip()
    session_store.update_context(
        state["session_id"],
        {
            "phase": state.get("phase"),
            "repair_count": repair_count,
            "verify_report": state.get("verify_report"),
            "workspace_snapshot_id": state.get("workspace_snapshot_id")
            or ctx.get("workspace_snapshot_id"),
            "pending_options": state.get("options") or ctx.get("pending_options") or [],
        },
    )
    return state


async def _handle_select_option(state: AgentState, option_id: str) -> AgentState:
    ctx = session_store.get_context(state["session_id"])
    pending = state.get("options") or ctx.get("pending_options") or []
    opt = get_option_by_id(pending, option_id)
    if not opt:
        state["intent"] = "plan"
        state["phase"] = "awaiting_option"
        state["options"] = list(pending)
        state["reply"] = (
            f"未找到方案 `{option_id}`。请回复「选方案A」或点击卡片重新选择。"
        )
        return state
    state["selected_option_id"] = str(opt["id"])
    state["options"] = list(pending)
    state["intent"] = "modify"
    state["want_apply"] = True
    # Desktop / implement CTA: apply patches; compile only if want_verify already true.
    if state.get("client_apply") or str(opt.get("id")) == "implement_now":
        state["want_apply"] = True
        # Default CTA = apply only; user/LLM can set want_verify for compile.
        if "want_verify" not in state:
            state["want_verify"] = False
    # Prefer option scope as starting point (real files only)
    preview = opt.get("scope_preview")
    if isinstance(preview, list) and preview:
        real = real_scope_files(preview)
        if real:
            state["scope"] = [
                {"file": f, "reason": "方案范围", "action": "edit", "priority": i + 1}
                for i, f in enumerate(real)
            ]
    locked = _load_locked_scaffold_patches(state)
    if locked:
        state["patches"] = locked
    detail = state.get("detail") or state.get("message") or ""
    prior = ""
    for m in reversed(session_store.list_messages(state["session_id"]) or []):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            prior = truncate(str(m.get("content") or ""), 2500)
            break
    state["detail"] = (
        f"{detail}\n（用户已选方案 {opt.get('id')}: {opt.get('title')}；"
        f"{opt.get('summary') or ''}；禁止改清单外文件）\n"
        f"上一轮分析摘要：\n{prior or '(无)'}"
    )
    return await _run_implement_pipeline(state)


async def _handle_rollback(state: AgentState) -> AgentState:
    ctx = session_store.get_context(state["session_id"])
    snap_id = state.get("workspace_snapshot_id") or ctx.get("workspace_snapshot_id")
    result = rollback_workspace(snap_id)
    state["intent"] = "execute"
    if result.get("ok"):
        state["phase"] = "idle"
        state["reply"] = f"已回滚到快照 `{snap_id}`。"
    else:
        state["phase"] = "failed"
        state["reply"] = f"回滚失败：{result.get('error') or 'unknown'}"
    state["verify_report"] = {"rollback": result}
    return state


def _promote_client_implement(state: AgentState) -> bool:
    """Desktop: only promote when intent LLM already set want_apply."""
    if not state.get("client_apply"):
        return False
    if not state.get("want_apply"):
        return False
    # Do not force want_verify — LLM decides compile/self-test this turn.
    state["intent"] = "modify"
    return True


def _attach_implement_cta(state: AgentState) -> AgentState:
    """After a change list, ask whether to apply; consent is judged by intent LLM next turn."""
    if not state.get("client_apply"):
        return state
    if state.get("intent") != "plan":
        return state
    cta = {
        "id": "implement_now",
        "title": "帮我修改代码",
        "summary": "按上述清单在本机应用补丁；若你同时说提交/推送，会在写入后继续 git。",
        "tradeoffs": "会修改本机已绑定项目目录中的源码",
    }
    existing = [o for o in (state.get("options") or []) if isinstance(o, dict)]
    if not any(str(o.get("id")) == "implement_now" for o in existing):
        state["options"] = existing + [cta]
    state["phase"] = "awaiting_option"
    reply = (state.get("reply") or "").rstrip()
    if "需要我帮您修改吗" not in reply and "需要我帮你修改吗" not in reply:
        state["reply"] = reply + "\n\n需要我帮您修改吗？"
    # Persist scaffold files so "开始改代码" can write without regenerating / expanding.
    scaffolds = extract_scaffold_file_patches(state.get("reply") or "")
    if scaffolds:
        state["scope"] = [
            {
                "file": str(p.get("file") or p.get("path")),
                "reason": "新增接口清单（仅这些文件）",
                "action": "write",
                "priority": i + 1,
            }
            for i, p in enumerate(scaffolds)
            if p.get("file") or p.get("path")
        ]
        _persist_locked_patches(state, scaffolds)
    return state


def _prepare_confirmed_implement(state: AgentState) -> AgentState:
    """Carry prior change-list analysis into implement after user consents."""
    ctx = session_store.get_context(state["session_id"])
    pending = state.get("options") or ctx.get("pending_options") or []
    if pending and not state.get("selected_option_id"):
        state["selected_option_id"] = "implement_now"
        state["options"] = list(pending)
    locked = _load_locked_scaffold_patches(state)
    if locked:
        state["patches"] = locked
        state["scope"] = [
            {
                "file": str(p.get("file") or p.get("path")),
                "reason": "上一轮确认清单",
                "action": "write",
                "priority": i + 1,
            }
            for i, p in enumerate(locked)
            if p.get("file") or p.get("path")
        ]
    elif isinstance(ctx.get("scope"), list) and ctx.get("scope") and not state.get("scope"):
        # Only keep real files from prior scope — never explore step fakes.
        real = real_scope_files(ctx["scope"])
        if real:
            state["scope"] = [
                {"file": f, "reason": "上一轮范围", "action": "edit", "priority": i + 1}
                for i, f in enumerate(real)
            ]
    prior = ""
    for m in reversed(session_store.list_messages(state["session_id"]) or []):
        if m.get("role") == "assistant" and (m.get("content") or "").strip():
            prior = truncate(str(m.get("content") or ""), 2500)
            break
    detail = state.get("detail") or state.get("message") or ""
    if prior and "上一轮分析摘要" not in detail:
        state["detail"] = (
            f"{detail}\n（用户确认：请按上一轮修改清单动手改代码；禁止改清单外文件）\n"
            f"上一轮分析摘要：\n{prior}"
        )
    state["intent"] = "modify"
    state["want_apply"] = True
    # Keep LLM want_verify (只改不测 → false；改完并自测 → true).
    return state


async def _decline_apply_reply(state: AgentState) -> AgentState:
    """Short ack when user declines modify-after-list; wording from LLM."""
    llm = get_llm_provider()
    try:
        text = await llm.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "用户刚看过代码修改清单，并表示暂不需要你改代码。"
                        "用一两句自然确认即可；不要再列清单，不要开始改代码。"
                    ),
                },
                {
                    "role": "user",
                    "content": f"用户说：{state.get('message') or ''}",
                },
            ],
            temperature=0.2,
            max_tokens=120,
        )
        state["reply"] = (text or "").strip() or "好的，先不改代码。需要时再说。"
    except Exception:
        state["reply"] = "好的，先不改代码。需要时再说。"
    state["phase"] = "idle"
    state["options"] = []
    return state


async def _try_golden_qa_reply(state: AgentState) -> AgentState | None:
    """黄金问答库快路径：命中高分直接返回离线存储答案，不消耗模型长链。

    fail-open：任何异常（无表/无库/检索失败）都返回 None，绝不阻断正常管线。
    """
    try:
        from app.analysis.golden_qa import try_golden_answer

        settings = get_settings()
        if not settings.golden_qa_enabled:
            return None
        hit = try_golden_answer(
            state["project_id"],
            state.get("message") or "",
            min_score=settings.golden_qa_min_score,
        )
    except Exception:
        logger.debug("golden_qa fast path failed (fail-open)", exc_info=True)
        return None
    if not hit or not str(hit.get("answer") or "").strip():
        return None
    reply = str(hit["answer"]).strip()
    score = float(hit.get("score") or 0.0)
    note = f"命中黄金问答库（相似度 {score:.2f}）"
    await _emit_progress(
        state,
        {"type": "step", "status": "done", "step_id": "golden_qa", "note": note},
    )
    state["intent"] = "ask"
    state["phase"] = "done"
    state["reply"] = f"{reply}\n\n---\n（{note}，来源：离线黄金问答库）"
    state["retrieved"] = []
    state["golden_hit"] = {
        "score": score,
        "source": hit.get("source"),
        "votes": hit.get("votes"),
        "question": hit.get("question"),
    }
    return state


def _heuristic_capability(message: str) -> str:
    """LLM 意图失败时的正则兜底：把散落的分类正则合并为一次 capability 判定。"""
    from app.agent.general_chat import is_general_chat_question, wants_verify_action

    if wants_verify_action(message):
        return "verify"
    if is_general_chat_question(message):
        return "general"
    return "code_qa"


async def _dispatch_by_intent(state: AgentState) -> AgentState:
    msg = state.get("message") or ""
    intent = normalize_intent(state.get("intent"), msg)
    state["intent"] = intent

    # —— capability 路由（LLM 判定为主，正则仅兜底 LLM 失败）——
    capability = str(state.get("capability") or "").strip()
    if capability not in {"general", "code_qa", "modify", "verify", "meta"}:
        capability = _heuristic_capability(msg)
    # add-api scaffold signal must never route to modify (new-API asks are code
    # generation, not edits to existing code) — desktop-chat-fa14438a45aa.
    if capability == "modify":
        try:
            from app.agent.pev_plan import wants_add_api_scaffold

            if wants_add_api_scaffold(msg):
                capability = "code_qa"
                intent = "ask"
                state["intent"] = "ask"
                state.pop("want_apply", None)
                state.pop("want_verify", None)
        except Exception:
            logger.debug("add_api capability guard failed", exc_info=True)
    state["capability"] = capability
    qa_guard = (
        not state.get("need_options")
        and not state.get("turn_stage_isolated")
        and not _is_correction(msg)
    )

    if _wants_meta(intent, msg):
        state["intent"] = "meta"
        state["phase"] = "idle"
        return await _meta_reply(state)

    # 黄金问答库快路径：仅代码域纯问答（code_qa）才命中；
    # modify/general 等不进黄金（避免「写个方法」被字段黄金抢答）；
    # add-api scaffold asks must not hit golden either (they are code
    # generation, not Q&A — desktop-chat-fa14438a45aa).
    # compose_code stories (multi-step code generation) also skip golden
    # — they ask for composed code, not a field/flow answer
    # (desktop-chat-95272f967734: golden hit with 0.63 gave table relationship
    # instead of the requested code).
    try:
        from app.agent.pev_plan import wants_add_api_scaffold

        _add_api_ask = bool(wants_add_api_scaffold(msg))
    except Exception:
        _add_api_ask = False
    try:
        from app.agent.runtime.turn_plan import _is_compose_code_story

        _compose_code_ask = _is_compose_code_story(msg)
    except Exception:
        _compose_code_ask = False
    if (
        capability == "code_qa"
        and intent == "ask"
        and qa_guard
        and not _add_api_ask
        and not _compose_code_ask
        # Endpoint/URL asks must not hit golden (Feign/service address answers
        # come from explore source_view, not from a golden table-dump essay).
        and not _wants_endpoint_lookup(msg)
        and not _wants_api_inventory(msg)
    ):
        golden = await _try_golden_qa_reply(state)
        if golden is not None:
            return golden

    # 通用问答：非代码域 → 直接 LLM 回答（不进工具循环）
    if capability == "general" and qa_guard and intent in {"ask", "plan", "meta"}:
        # Guard: LLM may misjudge a code-domain ask as general (e.g. 「XX服务的
        # 地址是什么」). Deterministic anchor re-check routes it back to code_qa.
        from app.agent.general_chat import is_general_chat_question

        if not is_general_chat_question(msg):
            capability = "code_qa"
        else:
            state["phase"] = "idle"
            state["reply"] = await general_chat_reply(
                msg,
                _history_before_current(state),
            )
            return state

    if _promote_client_implement(state):
        return await _run_implement_pipeline(state)

    # Land patches only when LLM set want_apply (or user confirmed via CTA).
    if state.get("want_apply"):
        state["intent"] = "modify"
        return await _run_implement_pipeline(state)

    # modify without want_apply = list/analysis this turn
    if intent == "modify":
        intent = "plan"
        state["intent"] = "plan"

    if intent == "execute":
        # 防误路由：「写个快速排序」等被意图模型判成 execute 时，若用户没有明确
        # 要求编译/验证/运行，降级为 ask（通用代码/知识问题直接直答，不进验证循环）。
        # capability==verify 只作提示，真正触发仍需 want_verify 或明确验证词。
        from app.agent.general_chat import is_general_chat_question, wants_verify_action

        if state.get("want_verify") or wants_verify_action(msg):
            return await _run_verify_repair_loop(state, auto=False)
        state["intent"] = "ask"
        intent = "ask"
        # —— capability 兜底复查（对齐上方 capability==general 分支的锚点复查）——
        # LLM 常把「按知识库文档生成实体类 / 写请求响应 VO」这类代码生成问法误判成
        # capability=general（desktop-chat-997ffe53135d：判成 execute+general 后在此
        # 短路进闲聊兜底，7.5 分钟只回「换个更具体的问法」，实体类管线根本没跑）。
        # 只要消息含项目/代码域锚点（is_general_chat_question=False）就纠正为 code_qa，
        # 交回代码问答/PEV 管线；仅当消息本身确属通用问题才走 general_chat_reply。
        if capability == "general" and not is_general_chat_question(msg):
            capability = "code_qa"
            state["capability"] = "code_qa"
        if capability == "general" or is_general_chat_question(msg):
            state["phase"] = "idle"
            state["reply"] = await general_chat_reply(
                msg,
                _history_before_current(state),
            )
            return state

    if state.get("need_options"):
        return await _run_options_pipeline(state)

    if intent in {"ask", "plan"} or _wants_explore(intent, msg) or _wants_explore(
        intent, state.get("detail") or ""
    ):
        if intent not in {"ask", "plan"}:
            state["intent"] = "ask"
        else:
            state["intent"] = intent

        state = await _explore(state)
        if state.get("intent") == "plan" and not (state.get("reply") or "").strip():
            state = await _retrieve(state)
            state = await _general_reply(state)
        if state.get("intent") == "plan":
            state = _attach_implement_cta(state)
        return state

    # Fallback: treat as project ask via retrieve + reply
    state["intent"] = "ask"
    state = await _retrieve(state)
    state = await _general_reply(state)
    state["phase"] = "idle"
    return state


async def run_chat_pipeline(state: AgentState) -> AgentState:
    state = await _understand(state)
    msg = state["message"]
    if _is_correction(msg):
        prev = _last_user_question(state["session_id"], msg)
        if prev:
            state["message"] = prev
            state["detail"] = (
                f"{prev}\n（用户反馈上一答不正确，请重新查代码作答；"
                "若问参数/字段校验，必须读入口方法的 same_class_calls。）"
            )
            state["intent"] = "ask"
            msg = prev
    elif _is_reanswer_request(msg):
        prev = _resolve_reanswer(state["session_id"], msg)
        if prev:
            state["message"] = prev
            state["detail"] = (
                f"{prev}\n（用户要求重新回答上一问；已选项目，请重新查代码作答；"
                "若问参数/字段校验，必须读入口方法的 same_class_calls。）"
            )
            state["intent"] = "ask"
            state["phase"] = "idle"
            state.pop("entry_hint", None)
            state.pop("keywords", None)
            state.pop("need_options", None)
            state["capability"] = (
                "general" if is_general_chat_question(prev) else "code_qa"
            )
            msg = prev
    return await _dispatch_by_intent(state)


async def _run_sequential_qa(state: AgentState, turn: Any) -> AgentState:
    """Run ask/plan stages in the user's semantic order; later stages see prior answers."""
    from app.agent.runtime.turn_plan import compose_ordered_qa_reply
    from app.core.utils import truncate

    on_progress = state.get("on_progress")
    orig_msg = state.get("message") or ""
    orig_detail = state.get("detail") or ""
    collected: list[tuple[Any, str]] = []
    all_steps: list[Any] = []
    nodes = list(getattr(turn, "qa_nodes", None) or [])
    n = len(nodes)
    prior_block = ""
    prev_goal = ""
    try:
        from app.core.config import get_settings as _gs

        pev_on = bool(_gs().agent_use_pev)
        pev_fallback = bool(_gs().agent_pev_fallback)
    except Exception:
        pev_on = False
        pev_fallback = True

    hist_msgs = session_store.list_messages(state["session_id"])
    base_hist = "\n".join(
        f"{m['role']}: {truncate(m['content'], 300)}" for m in hist_msgs[-8:]
    )

    for i, node in enumerate(nodes):
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "running",
                        "step_id": f"turn_{node.id}",
                        "note": (
                            f"按语义顺序 {i + 1}/{n} → {node.title}："
                            f"{(node.question or '')[:60]}"
                        ),
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("sequential qa progress failed", exc_info=True)
        ans = ""
        hist = base_hist
        if prior_block:
            hist = (
                "本回合必须按用户原话的先后顺序作答；"
                "后一问里的「这个/该/上述」指前序结论，禁止跳过前序意图。\n"
                + prior_block
                + (("\n\n" + hist) if hist else "")
            )
        if pev_on:
            try:
                from app.agent.pev_agent import run_pev_pipeline

                pev = await run_pev_pipeline(
                    node.question,
                    project_id=state["project_id"],
                    history_text=hist or None,
                    prior_question=(nodes[i - 1].question if i else None),
                    prior_goal=prev_goal,
                    lock_entry=False,
                    on_progress=on_progress,
                )
                ans = str(pev.get("answer") or "").strip()
                all_steps.extend(list(pev.get("steps") or []))
                if pev.get("plan"):
                    state["research_plan"] = pev.get("plan")
                    pev_goal = (pev.get("plan") or {}).get("goal") or {}
                    prev_goal = str(pev_goal.get("goal") or "") if pev_goal else ""
            except Exception:
                logger.exception("sequential PEV failed at %s", getattr(node, "id", ""))
                ans = ""
        if not ans and pev_fallback:
            state["message"] = node.question
            state["intent"] = "plan" if node.layer == "plan" else "ask"
            state["turn_prior_block"] = prior_block
            state["turn_stage_isolated"] = True
            state = await _explore(state)
            ans = str(state.get("reply") or "").strip()
            all_steps.extend(list(state.get("explore_steps") or []))
        collected.append((node, ans))
        if ans:
            prior_block = (
                f"{prior_block}\n\n【前序意图 {i + 1}. {node.title}】\n"
                f"问：{node.question}\n答：{ans[:1500]}"
            ).strip()
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "done",
                        "step_id": f"turn_{node.id}",
                        "note": f"完成 {i + 1}/{n} {node.title}",
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("sequential qa progress done failed", exc_info=True)

    state["message"] = orig_msg
    state["detail"] = orig_detail
    state["turn_prior_block"] = ""
    state["turn_stage_isolated"] = False
    state["reply"] = compose_ordered_qa_reply(collected)
    state["explore_steps"] = all_steps
    state["intent"] = (
        "plan" if any(n.layer == "plan" for n, _ in collected) else "ask"
    )
    if state.get("intent") == "plan":
        state = _attach_implement_cta(state)
    state["phase"] = state.get("phase") or "idle"
    return state


_PROJECT_TABLE_COUNT_RE = re.compile(
    r"(?is)(?:项目中|项目里|系统里|一共有|共有|总共有|有多少|几个|多少)?"
    r"(?:数据库)?(?:业务)?表(?:的)?(?:数量|个数|几张|多少张|一共|总计|清单|有哪些|有哪几张|列表)"
)
# 「项目里有哪些表/多少张表」是解析期已固化的表结构知识，不该路由成"本机连库查询"。
# 覆盖三种问法：①带项目/系统前缀；②表+数量/清单/多少张；③承接追问（具体都是哪些表 / 都是哪些表）。
_PROJECT_TABLE_INVENTORY_RE = re.compile(
    r"(?is)(?:项目|系统|数据库).{0,6}(?:表|表结构|表的清单)|"
    r"(?:表|表结构|业务表).{0,6}(?:数量|清单|列表|共有|一共|几张|多少张)|"
    r"(?:业务)?表(?:的)?(?:数量|清单|列表|共有|一共|几张|多少张)|"
    r"一共有多少张表|共有多少张表|有多少张表|多少张表|表数量|表的数量|"
    r"一共有多少张(?:业务|数据库)?表|共(?:有|计)(?:多少|几)张(?:业务)?表|"
    r"(?:具体|详细)?(?:都是|有哪些|有哪几张|哪些|都包含|包含).{0,2}(?:哪些)?表|"
    r"(?:具体|详细)(?:都是|有哪些)?哪些表|都是哪些表|具体都是哪些表|具体有哪些表"
)


# 承接追问：「85张都列出来 / 全部列出来 / 都列出来」—— 指上一轮的表清单。
_PROJECT_TABLE_FOLLOWUP_RE = re.compile(
    r"(?is)(?:\d+张?|全部|所有|完整|全).{0,6}(?:都|全|一起)?(?:列出来|列出|列一下|给出来|展示)|"
    r"(?:都|全部|所有).{0,4}列出来|列出来(?:给我)?看|列出全部"
)


def _try_project_table_answer(
    message: str,
    project_id: str | None,
    session_id: str | None = None,
) -> str | None:
    """表清单/表数量类问题的确定性快答：查 table_dict 直接返回，不调 LLM。

    覆盖：
    - 独立问法（项目里有多少张表 / 有哪些表 / 表清单）
    - 承接追问（85张都列出来 / 全部列出来）—— 结合会话上一轮是否表清单
    返回答案字符串（命中）；不匹配返回 None 走正常管线。
    """
    direct_hit = bool(_PROJECT_TABLE_INVENTORY_RE.search(message or ""))
    followup_hit = bool(_PROJECT_TABLE_FOLLOWUP_RE.search(message or ""))
    # 承接追问：优先看会话上一轮是否表语境；无历史时，若问句含「N张都列出来」
    # 且 N 与 table_dict 总数匹配（"张"是表专属量词，85 张 ≈ 85 张表），
    # 也按表处理，避免被路由 LLM 误判成本机文件操作。
    if followup_hit and not direct_hit:
        if session_id is not None and _session_was_table_context(session_id):
            pass  # 有表语境 → 命中
        else:
            m = re.search(r"(?is)(\d+)\s*张", message or "")
            if m is None:
                return None
            try:
                from app.analysis.table_dict import load_table_dict

                pids = [project_id] if project_id else []
                total = len(load_table_dict(pids[0])) if pids else 0
            except Exception:
                total = 0
            if int(m.group(1)) != total:
                return None
    if not (direct_hit or followup_hit):
        return None
    try:
        from app.analysis.table_dict import load_table_dict

        pids = [project_id] if project_id else []
        tables: list[dict] = []
        if pids:
            try:
                tables = load_table_dict(pids[0])
            except Exception:
                tables = []
        if not tables:
            return None
        total = len(tables)
        with_name = [t for t in tables if t.get("name_cn")]
        # 问数量（且不是"全部列出来"）→ 只答数字
        want_count = re.search(r"(?is)(数量|几张|多少张|几个|一共有|共有|总共有|一共|总计)", message or "")
        if want_count and not followup_hit:
            lines = [
                f"当前项目（{pids[0]}）的 `table_dict` 表结构字典共收录 **{total} 张表**"
                + (f"（{len(with_name)} 张有中文名）" if with_name else "") + "。",
                "",
                "（数据来源：解析期从 Entity 的 @TableName 推导，落 MySQL `table_dict` 表。）",
            ]
            return "\n".join(lines)
        # 清单 → 列表；承接"都列出来"时列全部，否则前 40
        lines = [f"当前项目（{pids[0]}）的表结构字典共 **{total} 张表**：", ""]
        limit = total if followup_hit else 40
        for i, t in enumerate(tables[:limit], 1):
            cn = t.get("name_cn") or ""
            label = f"{t.get('table_name')}" + (f"（{cn}）" if cn else "")
            lines.append(f"{i}. `{label}`")
        if total > limit:
            lines.append(f"\n…共 {total} 张，仅列前 {limit} 张（可问「XX 表的字段」查看具体表）。")
        lines.append("")
        lines.append("（数据来源：解析期从 Entity 的 @TableName 推导，落 MySQL `table_dict` 表。）")
        return "\n".join(lines)
    except Exception:
        logger.debug("project table answer failed (fallback to normal pipeline)", exc_info=True)
        return None


def _session_was_table_context(session_id: str) -> bool:
    """判断会话最近一轮是否表清单/表数量语境（供承接追问判定）。"""
    try:
        ctx = session_store.get_context(session_id)
        last_intent = str(ctx.get("last_intent") or "")
        # 最近的 assistant 回复含"表结构字典共"/"张表" → 表语境
        msgs = session_store.list_messages(session_id)
        for m in reversed(msgs[-6:]):
            content = str(m.get("content") or "")
            if "表结构字典共" in content or "共收录" in content and "张表" in content:
                return True
            if content.startswith("当前项目") and "张表" in content:
                return True
        return last_intent in {"ask"} and any(
            "张表" in str(m.get("content") or "") for m in msgs[-6:]
        )
    except Exception:
        return False


async def _run_no_project_chat(
    *,
    message: str,
    sid: str,
    settings: Any,
    want_client: bool,
    act: str,
    on_progress: Any | None,
    traj: AppendTrajectory | None,
) -> ChatResponse:
    """未选择项目时的回合行为：普通 AI 问答（带会话上下文记忆）。

    - 桌面端开启 client_agent 时：先走本机路由（OS/运维类需求 → 本机执行闭环，
      与之前「不选项目直接清理」的行为一致）；非本机需求 → 直接普通问答。
    - 本轮无论是否曾选中项目，都会带上同一会话的历史（可混合：勾选项目问代码、
      取消勾选问普通问题，上下文不断）。
    """
    from app.agent.general_chat import format_history_text, general_chat_reply
    from app.agent.remote_loop import create_agent_run, route_client_need

    route_mode = "none"
    route = None
    if want_client and act not in {"select_option", "verify", "rollback"}:
        try:
            route = await route_client_need(message)
            route_mode = str(route.get("mode") or "none")
        except Exception:
            logger.debug("no-project client route failed; plain answer", exc_info=True)
            route_mode = "none"
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "done",
                        "step_id": "client_route",
                        "note": (
                            f"本机路由 → {route_mode}"
                            + (
                                f" goal={str(route.get('goal') or '')[:80]}"
                                if route_mode != "none" and route is not None
                                else ""
                            )
                        ),
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("no-project client route progress failed", exc_info=True)
        if route_mode == "direct":
            run = create_agent_run(
                goal=str(route.get("goal") or message),
                project_id=settings.project_id or "local",
                session_id=sid,
            )
            reply = str(
                route.get("reply_preview")
                or "好的，我在本机执行相关操作，完成后告诉你结果。"
            )
            mid = session_store.append_message(
                sid,
                "assistant",
                reply,
                meta={
                    "client_run_id": run.id,
                    "phase": "client_running",
                    "intent": "client_agent",
                },
            )
            session_store.update_context(
                sid,
                {"phase": "client_running", "client_run_id": run.id, "pending_options": []},
            )
            try:
                if traj is not None:
                    traj.emit(
                        "final",
                        type="final",
                        intent="client_agent",
                        phase="client_running",
                        assistant_message_id=mid,
                        ok=True,
                    )
            except Exception:
                logger.debug(
                    "trajectory final emit failed (no-project client run)", exc_info=True
                )
            return ChatResponse(
                session_id=sid,
                reply=reply,
                intent="client_agent",
                phase="client_running",
                assistant_message_id=mid,
                client_run_id=run.id,
                project_ids=[],
                trajectory_file=str(traj.path) if traj is not None else None,
                runtime_mode=get_runtime_profile().mode,
            )

    if on_progress:
        try:
            maybe = on_progress(
                {
                    "type": "step",
                    "status": "running",
                    "step_id": "general_chat",
                    "note": "未选择项目 → 普通问答（带上下文记忆）",
                }
            )
            if hasattr(maybe, "__await__"):
                await maybe  # type: ignore[misc]
        except Exception:
            logger.debug("no-project progress emit failed", exc_info=True)

    msgs = session_store.list_messages(sid)
    if msgs and msgs[-1].get("role") == "user" and (msgs[-1].get("content") or "") == message:
        history = msgs[:-1]  # 排除本轮刚追加的用户消息，避免自答
    else:
        history = msgs
    reply = await general_chat_reply(message, format_history_text(history))
    if route_mode == "code" or not is_general_chat_question(message):
        reply = (
            reply.rstrip()
            + "\n\n---\n\n"
            "（当前未选择项目。若希望基于项目代码回答或改代码，请先在左侧勾选项目。）"
        )
    mid = session_store.append_message(
        sid,
        "assistant",
        reply,
        meta={"intent": "ask", "phase": "done", "no_project": True},
    )
    session_store.update_context(
        sid, {"last_intent": "ask", "phase": "done", "no_project": True}
    )
    try:
        if traj is not None:
            traj.emit(
                "final",
                type="final",
                intent="ask",
                phase="done",
                assistant_message_id=mid,
                ok=True,
            )
    except Exception:
        logger.debug("trajectory final emit failed (no-project chat)", exc_info=True)
    return ChatResponse(
        session_id=sid,
        reply=reply,
        intent="ask",
        phase="done",
        assistant_message_id=mid,
        project_ids=[],
        trajectory_file=str(traj.path) if traj is not None else None,
        runtime_mode=get_runtime_profile().mode,
    )


@traceable(name="chat", run_type="chain")
async def run_chat(
    message: str,
    *,
    project_id: str | None = None,
    project_ids: list[str] | None = None,
    session_id: str | None = None,
    client_apply: bool = False,
    client_agent: bool = False,
    local_assist: bool = False,  # deprecated alias → client_agent
    drive: str | None = None,
    on_progress: Any | None = None,
    action: str | None = None,
    option_id: str | None = None,
    option_ids: list[str] | None = None,
    cleanup_catalog: list[dict[str, Any]] | None = None,
) -> ChatResponse:
    # —— AgentScope 迁移层开关：AGENT_RUNTIME_MODE=agentscope 时走 agentscope_bridge ——
    # 学习样板（对照 ACP 课程）：通用问答走 GeneralAgent、代码问答走 ReActAgent 工具循环；
    # 原编排完整保留，改回 standard/minimal 即可回退。
    if str(get_settings().agent_runtime_mode).strip().lower() == "agentscope":
        from app.agent.agentscope_bridge.adapter import run_chat_scope

        return await run_chat_scope(
            message,
            project_id=project_id,
            project_ids=project_ids,
            session_id=session_id,
            client_apply=client_apply,
            client_agent=client_agent,
            local_assist=local_assist,
            drive=drive,
            on_progress=on_progress,
            action=action,
            option_id=option_id,
            option_ids=option_ids,
            cleanup_catalog=cleanup_catalog,
        )

    from app.agent.remote_loop import (
        create_agent_run,
        create_cleanup_agent_run,
        route_client_need,
    )
    from app.agent.runtime.turn_plan import heuristic_turn_plan, llm_turn_plan

    t0 = time.perf_counter()
    settings = get_settings()
    act = (action or "").strip().lower()
    # Desktop: client_agent (or legacy local_assist / client_apply) enables remote command loop.
    want_client = bool(client_agent or local_assist or client_apply)
    # LLM 主判回合意图拆分：启发式拆不出(单段)且句子明显复合(多意图无连接词)时，
    # 让 LLM 按语义顺序拆分 ask/plan/modify/execute/client；任何失败回退启发式。
    turn = await llm_turn_plan(message)
    mixed_ask_ops = bool(turn.has_qa and turn.has_post_qa)
    ordered_qa = bool(turn.needs_sequential_qa)
    pending_client_goal = turn.follow_on_goal if mixed_ask_ops else ""
    pending_client_mode = turn.follow_on_client_mode if mixed_ask_ops else ""

    # —— User confirmed cleanup options → client sys_cleanup run ——
    if act == "cleanup_selected":
        ids_sel = [str(x) for x in (option_ids or []) if x]
        if not ids_sel:
            sid = session_store.get_or_create_session(
                project_id or settings.project_id or "local", session_id, title="本机清理"
            )
            session_store.append_message(sid, "user", message)
            reply = "请至少勾选一项要清理的内容。"
            mid = session_store.append_message(sid, "assistant", reply)
            return ChatResponse(
                session_id=sid,
                reply=reply,
                intent="client_agent",
                phase="awaiting_cleanup",
                assistant_message_id=mid,
                project_ids=[project_id or settings.project_id or "local"],
            )
        run = create_cleanup_agent_run(
            goal=message or "按选择清理",
            option_ids=ids_sel,
            catalog=cleanup_catalog or [],
            session_id=session_id,
        )
        sid = session_store.get_or_create_session(
            project_id or settings.project_id or "local", session_id, title="本机清理"
        )
        session_store.append_message(sid, "user", message)
        reply = "好的，正在按你的选择清理本机…完成后会告诉你结果。"
        mid = session_store.append_message(
            sid, "assistant", reply, meta={"client_run_id": run.id, "phase": "client_running"}
        )
        session_store.update_context(sid, {"phase": "client_running", "client_run_id": run.id})
        return ChatResponse(
            session_id=sid,
            reply=reply,
            intent="client_agent",
            phase="client_running",
            assistant_message_id=mid,
            client_run_id=run.id,
            project_ids=[project_id or settings.project_id or "local"],
        )

    ids = [p for p in (project_ids or []) if p]
    if project_id and project_id not in ids:
        ids.insert(0, project_id)
    # 显式传了 project_ids（即使为空数组）＝用户明确的选择（勾选与否）；
    # 未传 project_ids 且无 project_id 时仍回退默认项目（兼容 API / 旧客户端）。
    explicit_selection = project_ids is not None
    if not ids and not explicit_selection:
        ids = [settings.project_id or "local"]
    deduped: list[str] = []
    seen: set[str] = set()
    for p in ids:
        if p in seen:
            continue
        seen.add(p)
        deduped.append(p)
    ids = deduped
    project_id = ids[0] if ids else None
    sid = session_store.get_or_create_session(project_id, session_id, title=message[:40])
    session_store.append_message(sid, "user", message)

    traj: AppendTrajectory | None = None
    try:
        from app.agent.trajectory import trajectory_enabled

        if trajectory_enabled():
            prof = get_runtime_profile()
            traj = AppendTrajectory.for_session(
                sid, compact=prof.compact_trajectory
            )
            on_progress = wrap_progress(on_progress, traj)
            traj.emit(
                "user_turn",
                type="user_turn",
                mode=prof.mode,
                message=truncate(message, 500),
                project_id=project_id,
                action=act or None,
            )
    except Exception:
        logger.debug("trajectory bootstrap failed", exc_info=True)
        traj = None

    # —— 未选择项目：普通问答（带会话上下文记忆），本机助手（OS 类需求）仍可用 ——
    if not ids:
        return await _run_no_project_chat(
            message=message,
            sid=sid,
            settings=settings,
            want_client=want_client,
            act=act,
            on_progress=on_progress,
            traj=traj,
        )

    # —— 日志驱动问题定位（log_diagnose 意图）：先于默认流程管线，不走放款主链兜底 ——
    try:
        _ld_ctx = session_store.get_context(sid)
        _ld_phase = str(_ld_ctx.get("phase") or "idle")
        _ld_pending = bool(_ld_ctx.get("pending_options"))
    except Exception:
        _ld_phase, _ld_pending = "idle", False
    if (
        (not act or act in {"", "diagnose", "log_diagnose"})
        and not _ld_pending
        and _ld_phase != "awaiting_option"
    ):
        from app.agent.log_diagnosis import run_log_diagnosis, wants_log_diagnose

        if wants_log_diagnose(message):
            t_diag = time.perf_counter()
            try:
                _ld = await run_log_diagnosis(
                    message, project_id=project_id, on_progress=on_progress
                )
                _reply = str(_ld.get("reply") or "（诊断未生成）")
                _steps = list(_ld.get("steps") or [])
                _ok = not str(_reply).startswith("**诊断")
            except Exception as e:  # noqa: BLE001
                logger.exception("log diagnosis pipeline failed")
                _reply = f"**诊断失败**：{type(e).__name__}。请稍后重试。"
                _steps = [
                    {"step_id": "log_diagnose", "status": "failed", "note": str(e)}
                ]
                _ok = False
            _elapsed = max(0, int((time.perf_counter() - t_diag) * 1000))
            mid = session_store.append_message(
                sid,
                "assistant",
                _reply,
                meta={
                    "intent": "log_diagnose",
                    "phase": "done",
                    "elapsed_ms": _elapsed,
                    "explore_steps": _steps,
                    "research_plan": {"kind": "log_diagnosis"},
                },
            )
            session_store.update_context(
                sid,
                {
                    "last_intent": "log_diagnose",
                    "phase": "done",
                    "explore_steps": _steps,
                    "elapsed_ms": _elapsed,
                    "pending_options": [],
                    "last_explore_question": message,
                },
            )
            try:
                if traj is not None:
                    traj.emit(
                        "final",
                        type="final",
                        intent="log_diagnose",
                        phase="done",
                        elapsed_ms=_elapsed,
                        assistant_message_id=mid,
                        ok=_ok,
                        verify_ok=None,
                    )
            except Exception:
                logger.debug("trajectory final emit failed (log diagnosis)", exc_info=True)
            _runtime_mode = None
            try:
                _runtime_mode = get_runtime_profile().mode
            except Exception:
                pass
            return ChatResponse(
                session_id=sid,
                reply=_reply,
                intent="log_diagnose",
                phase="done",
                assistant_message_id=mid,
                project_ids=ids,
                elapsed_ms=_elapsed,
                explore_steps=_steps,
                plan={"kind": "log_diagnosis"},
                review={"ok": _ok, "source": "log_diagnosis"},
                runtime_mode=_runtime_mode,
                trajectory_file=str(traj.path) if traj is not None else None,
            )

    # —— LLM route: direct client run (OS/ops) without keyword matching ——
    # Mixed ask+ops (先看代码再 git)：先问答，本机动作挂在答完之后，禁止短路掉 PEV。
    route_mode: str | None = None
    # 表清单/表数量短路：解析期已固化的表结构知识，直接答，不路由成本机连库操作。
    if not (mixed_ask_ops or ordered_qa):
        table_reply = _try_project_table_answer(message, project_id, session_id=sid)
        if table_reply is not None:
            mid = session_store.append_message(
                sid, "assistant", table_reply,
                meta={"intent": "ask", "phase": "done", "elapsed_ms": 0},
            )
            try:
                if traj is not None:
                    traj.emit("final", type="final", intent="ask", phase="done",
                              assistant_message_id=mid, ok=True)
            except Exception:
                logger.debug("trajectory final emit failed (table answer)", exc_info=True)
            return ChatResponse(
                session_id=sid,
                reply=table_reply,
                intent="ask",
                phase="done",
                assistant_message_id=mid,
                project_ids=ids,
                trajectory_file=str(traj.path) if traj is not None else None,
                runtime_mode=get_runtime_profile().mode,
            )
    if mixed_ask_ops or ordered_qa:
        if mixed_ask_ops:
            route_mode = "none"
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "done",
                        "step_id": "turn_plan",
                        "note": (
                            "回合编排 → "
                            + " → ".join(
                                {"ask": "查代码", "plan": "改动清单", "modify": "改码", "execute": "编译", "client": "本机git"}.get(
                                    n.layer, n.layer
                                )
                                for n in turn.nodes
                            )
                            + "（写文件/push 需确认）"
                        ),
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("turn_plan progress emit failed", exc_info=True)
    elif want_client and act not in {"select_option", "verify", "rollback"}:
        # 路由判定前先发 running 事件，UI 立即有反馈；调用本身在
        # route_client_need 内有超时上限（LLM_ROUTE_TIMEOUT，默认 30s），
        # 超时 fail-closed none —— 不再出现整 240s 无任何进度的白等
        # （desktop-chat-997ffe53135d：路由调用拖满总预算后只看到 done 事件）。
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "running",
                        "step_id": "client_route",
                        "note": "本机路由判定中（调用模型）…",
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("client route progress emit failed", exc_info=True)
        route = await route_client_need(message)
        route_mode = str(route.get("mode") or "none")
        if on_progress:
            try:
                maybe = on_progress(
                    {
                        "type": "step",
                        "status": "done",
                        "step_id": "client_route",
                        "note": (
                            f"本机路由 → {route_mode}"
                            + (
                                f" goal={str(route.get('goal') or '')[:80]}"
                                if route_mode != "none"
                                else ""
                            )
                        ),
                    }
                )
                if hasattr(maybe, "__await__"):
                    await maybe  # type: ignore[misc]
            except Exception:
                logger.debug("client route progress emit failed", exc_info=True)
        if route_mode == "direct":
            run = create_agent_run(
                goal=str(route.get("goal") or message),
                project_id=project_id,
                session_id=sid,
            )
            reply = str(route.get("reply_preview") or "好的，我在本机执行相关操作，完成后告诉你结果。")
            mid = session_store.append_message(
                sid,
                "assistant",
                reply,
                meta={"client_run_id": run.id, "phase": "client_running", "intent": "client_agent"},
            )
            session_store.update_context(
                sid, {"phase": "client_running", "client_run_id": run.id, "pending_options": []}
            )
            try:
                if traj is not None:
                    traj.emit(
                        "final",
                        type="final",
                        intent="client_agent",
                        phase="client_running",
                        assistant_message_id=mid,
                        ok=True,
                    )
            except Exception:
                logger.debug("trajectory final emit failed", exc_info=True)
            return ChatResponse(
                session_id=sid,
                reply=reply,
                intent="client_agent",
                phase="client_running",
                assistant_message_id=mid,
                client_run_id=run.id,
                project_ids=ids,
                trajectory_file=str(traj.path) if traj is not None else None,
                runtime_mode=get_runtime_profile().mode,
            )
        # mode=code → desktop can apply; whether to write this turn is intent LLM's want_apply
        if route_mode == "code":
            client_apply = True

    state: AgentState = {
        "message": (
            message
            if ordered_qa
            else (turn.qa_text if mixed_ask_ops and turn.qa_text else message)
        ),
        "project_id": project_id,
        "project_ids": ids,
        "session_id": sid,
        "client_apply": bool(client_apply),
        "on_progress": on_progress,
        "action": action or "",
        "option_id": option_id or "",
        "phase": "idle",
        "options": [],
    }

    try:
        ctx = session_store.get_context(sid)
        phase = str(ctx.get("phase") or "idle")
        pending_options = list(ctx.get("pending_options") or [])
        if pending_options:
            state["options"] = pending_options
        if ctx.get("workspace_snapshot_id"):
            state["workspace_snapshot_id"] = str(ctx["workspace_snapshot_id"])

        # Structured / natural-language option pick while awaiting_option
        if act == "rollback" or message.strip() in {"回滚", "撤销改动", "rollback"}:
            state = await _handle_rollback(state)
        elif act == "select_option" or (
            phase == "awaiting_option"
            and (option_id or parse_option_selection(message, pending_options))
        ):
            oid = option_id or parse_option_selection(message, pending_options)
            if oid:
                state = await _handle_select_option(state, oid)
            else:
                state["intent"] = "plan"
                state["phase"] = "awaiting_option"
                state["reply"] = "请明确选择方案（例如「选方案A」）或点击方案卡片。"
        elif act == "verify":
            state["intent"] = "execute"
            state = await _run_verify_repair_loop(state, auto=False)
        elif ordered_qa and not _is_correction(message):
            state["intent"] = "ask"
            state = await _run_sequential_qa(state, turn)
        else:
            state = await _understand(state)
            # 「不对」→ 用上一问重新探索
            if _is_correction(message):
                prev = _last_user_question(sid, message)
                if prev:
                    state["message"] = prev
                    state["detail"] = (
                        f"{prev}\n（用户反馈上一答不正确，请重新查代码作答；"
                        "若问参数/字段校验，必须读入口方法的 same_class_calls。）"
                    )
                    state["intent"] = "ask"
            # 「现在再回答一下」等纯重问 → 解析回上一问重新作答（选中项目后重跑代码管线）
            else:
                prev = _resolve_reanswer(sid, message)
                if prev:
                    state["message"] = prev
                    state["detail"] = (
                        f"{prev}\n（用户要求重新回答上一问；已选项目，请重新查代码作答；"
                        "若问参数/字段校验，必须读入口方法的 same_class_calls。）"
                    )
                    state["intent"] = "ask"
                    state["phase"] = "idle"
                    # LLM 对本轮重问短句的解析（entry_hint=「再回答一下」等）全部作废，
                    # 上一问的检索信息由 dispatch/explore 重新按 prev 生成。
                    state.pop("entry_hint", None)
                    state.pop("keywords", None)
                    state.pop("need_options", None)
                    state["capability"] = (
                        "general" if is_general_chat_question(prev) else "code_qa"
                    )
            # After change-list CTA: LLM consent → implement with prior analysis.
            skip_dispatch = False
            if phase == "awaiting_option" and state.get("want_apply"):
                state = _prepare_confirmed_implement(state)
            elif (
                phase == "awaiting_option"
                and not state.get("want_apply")
                and state.get("intent") == "meta"
            ):
                state = await _decline_apply_reply(state)
                skip_dispatch = True
            elif route_mode == "code":
                # route=code must NOT override intent LLM want_apply / intent.
                state["client_apply"] = True
                if state.get("want_apply"):
                    state["want_verify"] = True
                    state["intent"] = "modify"
            if not skip_dispatch:
                state = await _dispatch_by_intent(state)
    except httpx.HTTPStatusError as e:
        logger.exception("chat LLM HTTP error: %s", e)
        code = e.response.status_code if e.response is not None else "?"
        body = ""
        try:
            body = (e.response.text or "")[:200] if e.response is not None else ""
        except Exception:
            pass
        state["reply"] = (
            f"模型服务返回 HTTP {code}（{body or e.request.url}）。"
            "若为 405，多为代理/证书中间层拦截 POST，可检查网络代理，"
            "确认 HTTP_VERIFY_SSL 与 LLM_BASE_URL=https://api.siliconflow.cn/v1，然后重试。"
        )
        state["intent"] = state.get("intent") or "ask"
        state["scope"] = []
        state["patches"] = []
        state["retrieved"] = []
        state["phase"] = "failed"
    except (httpx.TimeoutException, httpx.TransportError) as e:
        logger.exception("chat LLM/network failure: %s", e)
        state["reply"] = (
            "模型服务响应超时或网络异常，请稍后点重试。"
            "也可在 .env 调大 LLM_TIMEOUT（当前默认 180 秒）。"
        )
        state["intent"] = state.get("intent") or "ask"
        state["scope"] = []
        state["patches"] = []
        state["retrieved"] = []
        state["phase"] = "failed"
    except Exception as e:
        logger.exception("chat pipeline failure: %s", e)
        state["reply"] = f"处理失败：{type(e).__name__}。请稍后重试。"
        state["intent"] = state.get("intent") or "ask"
        state["scope"] = []
        state["patches"] = []
        state["retrieved"] = []
        state["phase"] = "failed"

    reply = state.get("reply") or "未生成回复"
    if mixed_ask_ops:
        post_labels = " → ".join(
            {"modify": "改代码", "execute": "编译/自测", "client": "本机提交/推送"}.get(
                n.layer, n.layer
            )
            for n in turn.post_qa_nodes
        )
        # implement/verify in this turn: never silent-write or silent-push.
        if pending_client_mode == "code":
            reply = (
                reply.rstrip()
                + "\n\n---\n\n问答已完成。后续意图："
                + (post_labels or "改码/本机操作")
                + "。若要继续，请明确说「按这个改并提交」；本轮先不写文件、不推送。"
            )
        elif want_client and pending_client_mode == "direct" and pending_client_goal:
            run = create_agent_run(
                goal=pending_client_goal,
                project_id=project_id,
                session_id=sid,
                allow_write=False,
            )
            state["client_run_id"] = run.id
            state["phase"] = "client_running"
            reply = (
                reply.rstrip()
                + "\n\n---\n\n问答已完成。接下来按你的要求在本机提交并推送"
                "（commit / push 会再请你确认）。"
            )
            session_store.update_context(
                sid,
                {
                    "phase": "client_running",
                    "client_run_id": run.id,
                    "turn_follow_on": pending_client_goal,
                },
            )
        elif pending_client_goal:
            reply = (
                reply.rstrip()
                + "\n\n---\n\n问答已完成。提交并推送需要打开桌面客户端并开启本机循环，"
                "且 commit/push 仍会请你确认。"
            )
        state["reply"] = reply
    elapsed_ms = max(0, int((time.perf_counter() - t0) * 1000))
    research_plan = state.get("research_plan")
    answer_review = state.get("answer_review")
    phase_out = state.get("phase") or "idle"
    options_out = list(state.get("options") or [])
    assistant_message_id = session_store.append_message(
        sid,
        "assistant",
        reply,
        meta={
            "intent": state.get("intent"),
            "scope": state.get("scope"),
            "patches": state.get("patches"),
            "patch_results": state.get("patch_results"),
            "explore_steps": state.get("explore_steps"),
            "research_plan": research_plan,
            "review": answer_review,
            "elapsed_ms": elapsed_ms,
            "phase": phase_out,
            "options": options_out,
            "selected_option_id": state.get("selected_option_id"),
            "verify_report": state.get("verify_report"),
        },
    )
    ctx_update: dict[str, Any] = {
        "last_intent": state.get("intent"),
        "scope": state.get("scope"),
        "patches": state.get("patches"),
        "explore_steps": state.get("explore_steps"),
        "research_plan": research_plan,
        "review": answer_review,
        "elapsed_ms": elapsed_ms,
        "phase": phase_out,
        "pending_options": options_out if phase_out == "awaiting_option" else [],
        "selected_option_id": state.get("selected_option_id"),
        "verify_report": state.get("verify_report"),
    }
    if state.get("workspace_snapshot_id"):
        ctx_update["workspace_snapshot_id"] = state.get("workspace_snapshot_id")
    if state.get("client_run_id"):
        ctx_update["client_run_id"] = state.get("client_run_id")
    if phase_out == "done":
        ctx_update["repair_count"] = 0
    session_store.update_context(sid, ctx_update)

    retrieved: list[SymbolHit] = []
    for r in state.get("retrieved") or []:
        try:
            retrieved.append(SymbolHit(**r))
        except Exception:
            continue
    try:
        if traj is not None:
            traj.emit(
                "final",
                type="final",
                intent=state.get("intent"),
                phase=phase_out,
                elapsed_ms=elapsed_ms,
                assistant_message_id=assistant_message_id,
                entry=(
                    (state.get("explore_steps") or [{}])[-1].get("entry")
                    if isinstance(state.get("explore_steps"), list)
                    and state.get("explore_steps")
                    else None
                ),
                ok=phase_out not in {"failed"},
                verify_ok=(state.get("verify_report") or {}).get("ok")
                if isinstance(state.get("verify_report"), dict)
                else None,
            )
    except Exception:
        logger.debug("trajectory final emit failed", exc_info=True)
    runtime_mode = None
    try:
        runtime_mode = get_runtime_profile().mode
    except Exception:
        pass
    return ChatResponse(
        session_id=sid,
        reply=reply,
        intent=state.get("intent"),
        scope=state.get("scope") or [],
        patches=state.get("patches") or [],
        retrieved=retrieved,
        elapsed_ms=elapsed_ms,
        assistant_message_id=assistant_message_id,
        plan=research_plan,
        review=answer_review,
        explore_steps=state.get("explore_steps") or [],
        phase=phase_out,
        options=options_out,
        selected_option_id=state.get("selected_option_id"),
        verify_report=state.get("verify_report"),
        project_ids=list(state.get("project_ids") or [project_id]),
        client_run_id=state.get("client_run_id"),
        trajectory_file=str(traj.path) if traj is not None else None,
        runtime_mode=runtime_mode,
    )
