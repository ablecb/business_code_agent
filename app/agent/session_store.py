from __future__ import annotations

from typing import Any, Literal

from app.core.utils import new_id

_MEMORY_SESSIONS: dict[str, dict[str, Any]] = {}
_MEMORY_MESSAGES: dict[str, list[dict[str, Any]]] = {}
_MEMORY_FEEDBACK: dict[tuple[str, int], dict[str, Any]] = {}
_MEMORY_FEEDBACK_SEQ = 0
_USE_MEMORY = False

# 会话未绑定任何项目时的占位 project_id（chat_session.project_id 列非空）。
# 仅为存储占位：回合级路由仍以"未选择项目"处理（普通问答/本机助手）。
NO_PROJECT_MARKER = "_no_project"


def _norm_project_id(project_id: str | None) -> str:
    return (project_id or "").strip() or NO_PROJECT_MARKER


def _mysql_available() -> bool:
    global _USE_MEMORY
    if _USE_MEMORY:
        return False
    try:
        from sqlalchemy import text

        from app.db.mysql import get_engine

        eng = get_engine()
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        _USE_MEMORY = True
        return False


def get_or_create_session(
    project_id: str | None,
    session_id: str | None = None,
    title: str | None = None,
) -> str:
    # chat_session.project_id 列非空：未选择项目时写入占位标记，
    # 不影响回合级路由（路由看的是本次请求是否显式携带项目）。
    pid = _norm_project_id(project_id)
    if not _mysql_available():
        sid = session_id or new_id("chat_")
        _MEMORY_SESSIONS.setdefault(
            sid,
            {"project_id": pid, "title": title or "new chat", "context_json": {}},
        )
        _MEMORY_MESSAGES.setdefault(sid, [])
        return sid

    from sqlalchemy import select

    from app.db.mysql import get_session
    from app.models.orm import ChatSessionORM

    with get_session() as session:
        if session_id:
            row = session.get(ChatSessionORM, session_id)
            if row:
                return row.id
        sid = session_id or new_id("chat_")
        session.add(
            ChatSessionORM(
                id=sid,
                project_id=pid,
                title=title or "new chat",
                context_json={},
            )
        )
        return sid


def append_message(
    session_id: str,
    role: str,
    content: str,
    meta: dict[str, Any] | None = None,
) -> int | None:
    """Append a chat message; return its id when available."""
    if not _mysql_available():
        mid = len(_MEMORY_MESSAGES.get(session_id, [])) + 1
        _MEMORY_MESSAGES.setdefault(session_id, []).append(
            {"id": mid, "role": role, "content": content, "meta": meta}
        )
        return mid

    from app.db.mysql import get_session
    from app.models.orm import ChatMessageORM

    with get_session() as session:
        row = ChatMessageORM(
            session_id=session_id,
            role=role,
            content=content,
            meta_json=meta,
        )
        session.add(row)
        session.flush()
        return int(row.id)


def update_context(session_id: str, context: dict[str, Any]) -> None:
    if not _mysql_available():
        sess = _MEMORY_SESSIONS.setdefault(session_id, {"context_json": {}})
        merged = dict(sess.get("context_json") or {})
        merged.update(context)
        sess["context_json"] = merged
        return

    from app.db.mysql import get_session
    from app.models.orm import ChatSessionORM

    with get_session() as session:
        row = session.get(ChatSessionORM, session_id)
        if row:
            merged = dict(row.context_json or {})
            merged.update(context)
            row.context_json = merged


def get_context(session_id: str) -> dict[str, Any]:
    if not _mysql_available():
        return dict((_MEMORY_SESSIONS.get(session_id) or {}).get("context_json") or {})

    from app.db.mysql import get_session
    from app.models.orm import ChatSessionORM

    with get_session() as session:
        row = session.get(ChatSessionORM, session_id)
        return dict(row.context_json or {}) if row else {}


# ---------------------------------------------------------------------------
# 会话证据继承：追问轮复用上一轮已读证据（双赢：省重复读码 + 口径不漂移）
# ---------------------------------------------------------------------------

_EVIDENCE_MAX = 20


def _evidence_cm(entry: dict[str, Any]) -> str:
    cm = str(entry.get("cm") or "").strip()
    if cm:
        return cm
    fq = str(entry.get("fqcn") or "").strip()
    if fq and ("Controller" in fq or "Service" in fq):
        return fq
    return ""


def record_evidence(session_id: str, entries: list[dict[str, Any]]) -> None:
    """把本轮已读证据（cm + kind + summary + file_path）追加进会话上下文。

    按 Class#method 去重、上限 20 条；只存轻量摘要，下一轮追问直接引用，
    避免重新 search→get_symbol_code 定位同一批符号。
    """
    if not session_id or not entries:
        return
    ctx = get_context(session_id)
    seen: list[dict[str, Any]] = list(ctx.get("evidence_seen") or [])
    seen_by_cm = {
        _evidence_cm(e).lower(): e for e in seen if _evidence_cm(e)
    }
    for e in entries:
        cm = _evidence_cm(e)
        if not cm:
            continue
        seen_by_cm[cm.lower()] = {
            "cm": cm,
            "kind": str(e.get("kind") or ""),
            "summary": str(e.get("summary") or "")[:140],
            "file_path": str(e.get("file_path") or ""),
        }
    merged = list(seen_by_cm.values())[-_EVIDENCE_MAX:]
    if merged != seen:
        update_context(session_id, {"evidence_seen": merged})


def get_evidence_context(session_id: str, *, max_chars: int = 2400) -> str:
    """渲染上一轮已读证据块（追问轮注入探索提示，禁止重复 get_symbol_code）。"""
    if not session_id:
        return ""
    ctx = get_context(session_id)
    seen = list(ctx.get("evidence_seen") or [])
    if not seen:
        return ""
    lines = [
        "【上一轮已读证据】（承接追问时直接引用；"
        "禁止重复 get_symbol_code 定位同一批符号，只在需要新事实时继续下钻）"
    ]
    used = 0
    for e in seen:
        cm = _evidence_cm(e)
        if not cm:
            continue
        s = str(e.get("summary") or "").strip()
        line = f"- {cm}" + (f"：{s[:120]}" if s else "")
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def list_messages(session_id: str) -> list[dict[str, Any]]:
    if not _mysql_available():
        return list(_MEMORY_MESSAGES.get(session_id, []))

    from sqlalchemy import select

    from app.db.mysql import get_session
    from app.models.orm import ChatMessageORM

    with get_session() as session:
        rows = session.scalars(
            select(ChatMessageORM)
            .where(ChatMessageORM.session_id == session_id)
            .order_by(ChatMessageORM.id.asc())
        )
        return [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "meta": r.meta_json,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def delete_session(session_id: str) -> bool:
    """Delete a chat session and its messages. Returns True if something was removed."""
    if not session_id:
        return False
    if not _mysql_available():
        existed = session_id in _MEMORY_SESSIONS or session_id in _MEMORY_MESSAGES
        _MEMORY_SESSIONS.pop(session_id, None)
        _MEMORY_MESSAGES.pop(session_id, None)
        for key in [k for k in _MEMORY_FEEDBACK if k[0] == session_id]:
            _MEMORY_FEEDBACK.pop(key, None)
        return existed

    from sqlalchemy import delete

    from app.db.mysql import get_session
    from app.models.orm import ChatFeedbackORM, ChatMessageORM, ChatSessionORM

    with get_session() as session:
        row = session.get(ChatSessionORM, session_id)
        session.execute(delete(ChatMessageORM).where(ChatMessageORM.session_id == session_id))
        session.execute(delete(ChatFeedbackORM).where(ChatFeedbackORM.session_id == session_id))
        if row:
            session.delete(row)
            return True
        return False


def get_session_project_id(session_id: str) -> str | None:
    if not session_id:
        return None
    if not _mysql_available():
        sess = _MEMORY_SESSIONS.get(session_id) or {}
        return sess.get("project_id")

    from app.db.mysql import get_session
    from app.models.orm import ChatSessionORM

    with get_session() as session:
        row = session.get(ChatSessionORM, session_id)
        return row.project_id if row else None


def update_message_meta(
    session_id: str,
    message_id: int,
    patch: dict[str, Any],
) -> bool:
    """Merge keys into an existing message's meta. Returns False if not found."""
    if not session_id or message_id is None:
        return False
    if not _mysql_available():
        for m in _MEMORY_MESSAGES.get(session_id, []):
            if m.get("id") == message_id:
                meta = dict(m.get("meta") or {})
                meta.update(patch)
                m["meta"] = meta
                return True
        return False

    from app.db.mysql import get_session
    from app.models.orm import ChatMessageORM

    with get_session() as session:
        row = session.get(ChatMessageORM, message_id)
        if not row or row.session_id != session_id:
            return False
        meta = dict(row.meta_json or {})
        meta.update(patch)
        row.meta_json = meta
        return True


def _excerpt(text: str | None, limit: int = 280) -> str | None:
    if not text:
        return None
    s = str(text).strip()
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def upsert_feedback(
    *,
    session_id: str,
    assistant_message_id: int,
    vote: Literal["up", "down"],
    project_id: str | None = None,
    comment: str | None = None,
) -> dict[str, Any]:
    """
    Persist like/dislike with a full conversation snapshot at click time.
    One row per (session_id, assistant_message_id); re-vote updates.
    """
    if vote not in {"up", "down"}:
        raise ValueError("vote must be 'up' or 'down'")

    messages = list_messages(session_id)
    if not messages:
        raise LookupError("session not found or empty")

    target = next(
        (m for m in messages if m.get("id") == assistant_message_id and m.get("role") == "assistant"),
        None,
    )
    if not target:
        raise LookupError("assistant message not found in session")

    # Full conversation at click time (ops review).
    snapshot: list[dict[str, Any]] = [
        {
            "id": m.get("id"),
            "role": m.get("role"),
            "content": m.get("content"),
            "meta": m.get("meta"),
            "created_at": m.get("created_at"),
        }
        for m in messages
    ]

    # Prefer the user turn immediately before the voted assistant reply.
    question = None
    for i, m in enumerate(messages):
        if m.get("id") == assistant_message_id:
            for j in range(i - 1, -1, -1):
                if messages[j].get("role") == "user":
                    question = messages[j].get("content")
                    break
            break
    if question is None:
        for m in reversed(messages):
            if m.get("role") == "user":
                question = m.get("content")
                break

    pid = project_id or get_session_project_id(session_id) or "unknown"
    q_ex = _excerpt(question)
    r_ex = _excerpt(target.get("content"))
    comment_clean = (comment or "").strip() or None

    update_message_meta(session_id, assistant_message_id, {"vote": vote})

    if not _mysql_available():
        global _MEMORY_FEEDBACK_SEQ
        key = (session_id, assistant_message_id)
        existing = _MEMORY_FEEDBACK.get(key)
        updated = existing is not None
        if existing:
            existing.update(
                {
                    "vote": vote,
                    "comment": comment_clean,
                    "question_excerpt": q_ex,
                    "reply_excerpt": r_ex,
                    "conversation_json": snapshot,
                    "project_id": pid,
                }
            )
            row = existing
        else:
            _MEMORY_FEEDBACK_SEQ += 1
            row = {
                "id": _MEMORY_FEEDBACK_SEQ,
                "project_id": pid,
                "session_id": session_id,
                "assistant_message_id": assistant_message_id,
                "vote": vote,
                "comment": comment_clean,
                "question_excerpt": q_ex,
                "reply_excerpt": r_ex,
                "conversation_json": snapshot,
                "created_at": None,
            }
            _MEMORY_FEEDBACK[key] = row
        return {
            "id": row["id"],
            "session_id": session_id,
            "assistant_message_id": assistant_message_id,
            "vote": vote,
            "created_at": row.get("created_at"),
            "updated": updated,
        }

    from sqlalchemy import select

    from app.db.mysql import get_session
    from app.models.orm import ChatFeedbackORM

    with get_session() as session:
        row = session.scalars(
            select(ChatFeedbackORM).where(
                ChatFeedbackORM.session_id == session_id,
                ChatFeedbackORM.assistant_message_id == assistant_message_id,
            )
        ).first()
        updated = row is not None
        if row:
            row.vote = vote
            row.comment = comment_clean
            row.question_excerpt = q_ex
            row.reply_excerpt = r_ex
            row.conversation_json = snapshot
            row.project_id = pid
        else:
            row = ChatFeedbackORM(
                project_id=pid,
                session_id=session_id,
                assistant_message_id=assistant_message_id,
                vote=vote,
                comment=comment_clean,
                question_excerpt=q_ex,
                reply_excerpt=r_ex,
                conversation_json=snapshot,
            )
            session.add(row)
            session.flush()
        return {
            "id": int(row.id),
            "session_id": session_id,
            "assistant_message_id": assistant_message_id,
            "vote": vote,
            "created_at": row.created_at.isoformat() if row.created_at else None,
            "updated": updated,
        }


def list_feedback(
    *,
    session_id: str | None = None,
    project_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[dict[str, Any]], int]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))

    if not _mysql_available():
        rows = list(_MEMORY_FEEDBACK.values())
        if session_id:
            rows = [r for r in rows if r.get("session_id") == session_id]
        if project_id:
            rows = [r for r in rows if r.get("project_id") == project_id]
        rows.sort(key=lambda r: r.get("id") or 0, reverse=True)
        total = len(rows)
        page = rows[offset : offset + limit]
        return (
            [
                {
                    "id": r["id"],
                    "project_id": r.get("project_id") or "",
                    "session_id": r["session_id"],
                    "assistant_message_id": r["assistant_message_id"],
                    "vote": r["vote"],
                    "comment": r.get("comment"),
                    "question_excerpt": r.get("question_excerpt"),
                    "reply_excerpt": r.get("reply_excerpt"),
                    "created_at": r.get("created_at"),
                }
                for r in page
            ],
            total,
        )

    from sqlalchemy import func, select

    from app.db.mysql import get_session
    from app.models.orm import ChatFeedbackORM

    with get_session() as session:
        filters = []
        if session_id:
            filters.append(ChatFeedbackORM.session_id == session_id)
        if project_id:
            filters.append(ChatFeedbackORM.project_id == project_id)
        count_q = select(func.count()).select_from(ChatFeedbackORM)
        list_q = select(ChatFeedbackORM).order_by(ChatFeedbackORM.id.desc())
        if filters:
            count_q = count_q.where(*filters)
            list_q = list_q.where(*filters)
        total = int(session.scalar(count_q) or 0)
        rows = session.scalars(list_q.offset(offset).limit(limit)).all()
        return (
            [
                {
                    "id": int(r.id),
                    "project_id": r.project_id,
                    "session_id": r.session_id,
                    "assistant_message_id": int(r.assistant_message_id),
                    "vote": r.vote,
                    "comment": r.comment,
                    "question_excerpt": r.question_excerpt,
                    "reply_excerpt": r.reply_excerpt,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ],
            total,
        )
