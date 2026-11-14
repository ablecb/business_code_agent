"""want_apply from intent LLM must not be overridden by route=code / client_apply."""
from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

os.environ["LLM_PROVIDER"] = "mock"

from app.agent.graph_agent import _dispatch_by_intent, _run_implement_pipeline, run_chat
from app.llm.factory import get_llm_provider


def _use_mock():
    os.environ["LLM_PROVIDER"] = "mock"
    get_llm_provider.cache_clear()


def _sid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def test_modify_without_want_apply_becomes_plan_no_client_run():
    async def _run():
        state: dict[str, Any] = {
            "message": "先只给清单不要改代码",
            "project_id": "demo",
            "project_ids": ["demo"],
            "session_id": "sess_test_want_apply",
            "intent": "modify",
            "want_apply": False,
            "want_verify": False,
            "client_apply": True,
            "detail": "先只给清单不要改代码",
            "keywords": [],
            "options": [],
        }
        with (
            patch(
                "app.agent.graph_agent._explore",
                new=AsyncMock(
                    return_value={
                        **state,
                        "intent": "plan",
                        "reply": "## 修改范围\n- Foo.java",
                    }
                ),
            ),
            patch("app.agent.remote_loop.create_patch_agent_run") as create_run,
        ):
            out = await _dispatch_by_intent(state)
            create_run.assert_not_called()
        assert out.get("intent") == "plan"
        assert not out.get("want_apply")
        assert not out.get("client_run_id")
        assert out.get("phase") == "awaiting_option"

    asyncio.run(_run())


def test_run_implement_skips_patch_run_when_want_apply_false():
    async def _run():
        state: dict[str, Any] = {
            "message": "只要清单",
            "project_id": "demo",
            "session_id": "sess_test_impl_gate",
            "intent": "modify",
            "want_apply": False,
            "client_apply": True,
            "detail": "只要清单",
            "options": [],
        }
        with (
            patch(
                "app.agent.graph_agent._explore",
                new=AsyncMock(
                    return_value={
                        **state,
                        "intent": "plan",
                        "reply": "清单内容",
                    }
                ),
            ),
            patch("app.agent.remote_loop.create_patch_agent_run") as create_run,
        ):
            out = await _run_implement_pipeline(state)
            create_run.assert_not_called()
        assert out.get("intent") == "plan"
        assert not out.get("client_run_id")

    asyncio.run(_run())


def test_route_code_does_not_force_want_apply():
    """Even with client_agent + route=code, list-only mock intent keeps no client_run_id."""
    _use_mock()
    with patch(
        "app.agent.graph_agent._explore",
        new=AsyncMock(
            side_effect=lambda state: {
                **state,
                "intent": "plan",
                "reply": "## 修改范围\n- Demo.java",
                "explore_steps": [],
            }
        ),
    ):
        resp = asyncio.run(
            run_chat(
                "按接口文档新增商品查询接口，先只给我要改的类/方法清单，不要直接改代码",
                project_id="demo",
                session_id=_sid("sess_list_only"),
                client_apply=True,
                client_agent=True,
            )
        )
    assert resp.client_run_id in (None, "")
    assert resp.intent == "plan"
    assert "需要我帮您修改吗" in (resp.reply or "")


def test_confirm_yes_after_list_sets_want_apply():
    """After awaiting_option CTA, LLM consent → modify path prepares want_apply."""
    from app.agent import session_store
    from app.agent.graph_agent import _understand, _prepare_confirmed_implement

    _use_mock()
    sid = session_store.get_or_create_session("demo", _sid("sess_confirm_yes"), title="t")
    session_store.update_context(
        sid,
        {
            "phase": "awaiting_option",
            "pending_options": [
                {
                    "id": "implement_now",
                    "title": "帮我修改代码",
                    "summary": "按清单改",
                }
            ],
            "scope": [
                {
                    "file": "src/main/java/Demo.java",
                    "reason": "入口",
                    "action": "other",
                    "priority": 1,
                }
            ],
        },
    )
    session_store.append_message(
        sid, "assistant", "## 修改范围\n- Demo.java\n\n需要我帮您修改吗？"
    )

    async def _run():
        state = {
            "message": "可以",
            "project_id": "demo",
            "session_id": sid,
            "client_apply": True,
            "options": [
                {
                    "id": "implement_now",
                    "title": "帮我修改代码",
                    "summary": "按清单改",
                }
            ],
        }
        state = await _understand(state)
        assert state.get("want_apply") is True
        state = _prepare_confirmed_implement(state)
        assert state.get("intent") == "modify"
        assert state.get("want_apply") is True
        assert state.get("scope")

    asyncio.run(_run())


def test_attach_cta_asks_modify():
    from app.agent.graph_agent import _attach_implement_cta

    state = {
        "intent": "plan",
        "client_apply": True,
        "reply": "## 修改范围\n- A.java",
        "options": [],
    }
    out = _attach_implement_cta(state)
    assert "需要我帮您修改吗" in out["reply"]
    assert out["phase"] == "awaiting_option"
    assert any(o.get("id") == "implement_now" for o in out["options"])
