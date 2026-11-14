"""Change-impact discovery: entry → CALLS/code roles → evidence bag → plan.



Capability-oriented (add_field, protocol_sync, composite_field; AI polish).

No hardcoded Controller/method names.

"""

from __future__ import annotations



from app.agent.change_impact.discover import (

    Touchpoints,

    discover_add_field_touchpoints,

    discover_composite_field_touchpoints,

    discover_protocol_sync_touchpoints,

)

from app.agent.change_impact.render import (

    render_add_field_plan,

    render_composite_field_plan,

    render_protocol_sync_plan,

)





def _progress_safe(on_progress: object | None, ev: dict) -> None:

    if not on_progress:

        return

    try:

        maybe = on_progress(ev)  # type: ignore[operator]

        if hasattr(maybe, "__await__"):

            return

    except Exception:

        pass





def run_add_field_plan(

    project_id: str,

    question: str,

    *,

    entry_cm: str | None = None,

    on_progress: object | None = None,

) -> dict:

    """Discover add-field touchpoints; AI-polish the plan (deterministic fallback)."""

    tp = discover_add_field_touchpoints(

        project_id,

        question,

        entry_cm=entry_cm,

        on_progress=on_progress,

    )

    _progress_safe(

        on_progress,

        {

            "type": "step",

            "step_id": "render",

            "status": "running",

            "note": "AI 梳理改动方案",

        },

    )

    from app.agent.change_impact.ai_render import render_change_plan_with_ai



    answer, source = render_change_plan_with_ai(

        question, tp, capability="add_field"

    )

    _progress_safe(

        on_progress,

        {

            "type": "step",

            "step_id": "render",

            "status": "done",

            "note": (

                "AI 梳理完成"

                if source == "ai"

                else "已用程序草稿（AI 不可用或未通过校验）"

            ),

        },

    )

    return {

        "answer": answer,

        "touchpoints": tp,

        "steps": tp.steps,

        "entry": tp.entry,

        "ok": tp.ok,

        "render_source": source,

    }





def run_protocol_sync_plan(

    project_id: str,

    question: str,

    *,

    entry_cm: str | None = None,

    on_progress: object | None = None,

) -> dict:

    """Discover protocol-sync field pass-through touchpoints; AI-polish plan."""

    tp = discover_protocol_sync_touchpoints(

        project_id,

        question,

        entry_cm=entry_cm,

        on_progress=on_progress,

    )

    _progress_safe(

        on_progress,

        {

            "type": "step",

            "step_id": "render",

            "status": "running",

            "note": "AI 梳理协议同步透传方案",

        },

    )

    from app.agent.change_impact.ai_render import render_change_plan_with_ai



    answer, source = render_change_plan_with_ai(

        question, tp, capability="protocol_sync"

    )

    _progress_safe(

        on_progress,

        {

            "type": "step",

            "step_id": "render",

            "status": "done",

            "note": (

                "AI 梳理完成"

                if source == "ai"

                else "已用程序草稿（AI 不可用或未通过校验）"

            ),

        },

    )

    return {

        "answer": answer,

        "touchpoints": tp,

        "steps": tp.steps,

        "entry": tp.entry,

        "ok": tp.ok,

        "render_source": source,

    }





def run_composite_field_plan(

    project_id: str,

    question: str,

    *,

    entry_cm: str | None = None,

    on_progress: object | None = None,

) -> dict:

    """Discover add+persist+gate/external-pass touchpoints; AI-polish."""

    tp = discover_composite_field_touchpoints(

        project_id,

        question,

        entry_cm=entry_cm,

        on_progress=on_progress,

    )

    _progress_safe(

        on_progress,

        {

            "type": "step",

            "step_id": "render",

            "status": "running",

            "note": "AI 梳理复合改动方案",

        },

    )

    from app.agent.change_impact.ai_render import render_change_plan_with_ai



    answer, source = render_change_plan_with_ai(

        question, tp, capability="composite_field"

    )

    _progress_safe(

        on_progress,

        {

            "type": "step",

            "step_id": "render",

            "status": "done",

            "note": (

                "AI 梳理完成"

                if source == "ai"

                else "已用程序草稿（AI 不可用或未通过校验）"

            ),

        },

    )

    return {

        "answer": answer,

        "touchpoints": tp,

        "steps": tp.steps,

        "entry": tp.entry,

        "ok": tp.ok,

        "render_source": source,

        "roles": {k: v.class_method for k, v in (tp.roles or {}).items()},

    }





__all__ = [

    "Touchpoints",

    "discover_add_field_touchpoints",

    "discover_composite_field_touchpoints",

    "discover_protocol_sync_touchpoints",

    "render_add_field_plan",

    "render_composite_field_plan",

    "render_protocol_sync_plan",

    "run_add_field_plan",

    "run_composite_field_plan",

    "run_protocol_sync_plan",

]


