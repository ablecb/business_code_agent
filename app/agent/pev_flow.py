"""PEV flow layer extracted from pev_agent.py.

Entry resolution / spine / dig-in focus / flow rendering: verify_contract,
_exec_resolve_entry, _resolve_flow_spine, decide_dig_in_focus,
_exec_summarize_flow and the topic-focus machinery. pev_agent re-exports every
name for import compatibility.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from app.core.utils import truncate

from app.agent.prompts import load_prompt

from app.agent.pev_plan import (
    _SPINE_NOISE_RE,
    _parse_json_obj,
    heuristic_goal,
    wants_field_value_ask_safe,
)

logger = logging.getLogger("app.agent.pev_flow")

ProgressCb = Callable[[dict[str, Any]], Any] | None

_DIG_IN_FOCUS_PROMPT = load_prompt("dig_in_focus_prompt")
_LOAN_LEXICON_PATH = Path(__file__).resolve().parents[2] / "data" / "domain_lexicon.json"
_LOAN_ACTION_RE: re.Pattern | None = None


def _get_loan_action_re() -> re.Pattern:
    """Lazy-compiled regex matching loan-related action words (data-driven).

    Loads ``data/domain_lexicon.json``'s ``loan_action_words`` + ``loan_action_latin``
    at runtime — keeps product-domain nouns out of Python source code strings
    (``test_no_product_nouns`` compliance).  Falls back to safe Latin-only pattern
    when the file is absent.
    """
    global _LOAN_ACTION_RE
    if _LOAN_ACTION_RE is not None:
        return _LOAN_ACTION_RE
    try:
        data = json.loads(_LOAN_LEXICON_PATH.read_text(encoding="utf-8"))
        words: list[str] = list(data.get("loan_action_words") or [])
        latin: list[str] = list(data.get("loan_action_latin") or [])
    except Exception:
        logger.warning("domain_lexicon.json unreadable; fallback to Latin-only loan match")
        words, latin = [], ["withdrawal", "loan", "disburs"]
    latin = [w for w in latin if re.match(r"^[A-Za-z][a-zA-Z]*$", w)]
    all_parts = words + latin
    if not all_parts:
        all_parts = ["withdrawal", "loan"]
    pat = "(?i)" + "|".join(re.escape(p) for p in all_parts)
    _LOAN_ACTION_RE = re.compile(pat)
    return _LOAN_ACTION_RE


def verify_contract(
    goal: dict[str, Any], evidence: dict[str, Any], answer: str
) -> dict[str, Any]:
    """Check must-clauses against evidence (deterministic, test-friendly)."""
    gaps: list[dict[str, Any]] = []
    g = str(goal.get("goal") or "")
    also = {str(x) for x in (goal.get("also") or []) if x}
    ans = answer or ""
    need_add = g == "add_api" or (g == "compound" and "add_api" in also) or g == "compound"
    need_change = (
        g == "change_plan"
        or (g == "compound" and "change_plan" in also)
        or g == "compound"
    )

    if g in {"explain_flow", "change_plan", "general"} or need_change:
        if not evidence.get("entry") and g != "add_api":
            gaps.append({"id": "entry", "desc": "未锁定系统入口"})
    if need_add:
        if not evidence.get("kb_ok"):
            gaps.append({"id": "kb_spec", "desc": "未读到知识库接口文档"})
        if not evidence.get("api_style_ok"):
            gaps.append({"id": "style", "desc": "未找到项目 Controller 风格样例"})
        if len(ans) < 80 or "Controller" not in ans:
            gaps.append({"id": "scaffold", "desc": "未产出新增接口清单/代码骨架"})
        if "Request<" not in ans and "ResponseBuilder" not in ans:
            gaps.append(
                {"id": "style_match", "desc": "建议代码未体现项目 Request/Response 风格"}
            )
        # 业务步骤（validate/query/call/log/return）必须逐段有交代：
        # call 步骤要求要么命中真实 Feign 客户端、要么显式 TODO 标注证据缺口，
        # 禁止整段静默缺失（否则用户拿到空的 service 实现）。
        bsteps = list(evidence.get("business_steps") or [])
        if bsteps:
            ops = [str(s.get("op") or "") for s in bsteps if isinstance(s, dict)]
            if "call" in ops and not re.search(r"TODO|FeignClient|@Autowired", ans):
                gaps.append(
                    {
                        "id": "step_call",
                        "desc": "业务步骤含远程调用(call)但输出未体现客户端注入或证据缺口说明",
                    }
                )
            if "validate" in ops and not re.search(r"校验|assert|LoanAssert|ExceptionEnum", ans):
                gaps.append(
                    {
                        "id": "step_validate",
                        "desc": "业务步骤含参数校验(validate)但输出未给出校验断言/风格",
                    }
                )
            if "query" in ops and not re.search(r"查询|Mapper|LambdaQueryWrapper|TODO", ans):
                gaps.append(
                    {
                        "id": "step_query",
                        "desc": "业务步骤含查库(query)但输出未给出查询骨架/证据说明",
                    }
                )
            if "log" in ops and not re.search(r"log\.info|打印|日志", ans):
                gaps.append(
                    {
                        "id": "step_log",
                        "desc": "业务步骤含日志(log)但输出未给出打印语句",
                    }
                )
            if "return" in ops and "return" not in ans and "返回" not in ans:
                gaps.append(
                    {
                        "id": "step_return",
                        "desc": "业务步骤含返回(return)但输出未体现结果组装返回",
                    }
                )
    if g == "inventory" and goal.get("inventory_kind") == "repo_files":
        if evidence.get("repo_file_count") is None and "共" not in ans:
            gaps.append({"id": "scan_files", "desc": "未完成仓库文件扫描计数"})
        elif evidence.get("repo_file_count") == 0 and len(ans) < 20:
            gaps.append({"id": "count_list", "desc": "扫描结果为空且回答过短"})
    if g == "inventory" and goal.get("inventory_kind") == "architecture":
        if not evidence.get("architecture_ok") and len(ans) < 40:
            gaps.append({"id": "scan_structure", "desc": "未完成结构化源码扫描"})
    if g == "inventory" and goal.get("inventory_kind") == "symbol_probe":
        if not evidence.get("symbol_probe_ok") and "未找到" not in ans:
            gaps.append({"id": "probe_type", "desc": "未完成类型存在性核查"})
    if g == "inventory" and goal.get("inventory_kind") == "topic_probe":
        if not evidence.get("topic_probe_ok"):
            gaps.append({"id": "search", "desc": "未完成话题存在性检索"})
        elif evidence.get("topic_probe_exists") is False and "未" not in ans[:80]:
            gaps.append({"id": "answer", "desc": "未命中时应明确说明未检索到"})
    if g == "explain_flow":
        rows = evidence.get("mysql_rows") or []
        dig_in = bool(evidence.get("dig_in_focus"))
        if not evidence.get("calls_count") and not rows and not dig_in:
            gaps.append({"id": "spine", "desc": "主链/摘要证据不足"})
        if len(ans) < 40:
            gaps.append({"id": "summarize", "desc": "回答过短"})
        # Dig-in focus answers are method-scoped, not full numbered spines.
        if dig_in:
            focus_cms = [
                _simple_cm(str(c))
                for c in (evidence.get("dig_in_focus_cms") or [])
                if c and "#" in str(c)
            ]
            if not focus_cms:
                hint = str(
                    (evidence.get("dig_in_decision") or {}).get("focus_method") or ""
                ).strip()
                if hint and "#" in hint:
                    focus_cms = [_simple_cm(hint)]
            covered = False
            for cm in focus_cms:
                meth = cm.split("#", 1)[-1].split("(", 1)[0]
                if cm and (cm in ans or (meth and meth in ans)):
                    covered = True
                    break
            if focus_cms and not covered:
                gaps.append(
                    {
                        "id": "dig_in_focus",
                        "desc": "追问聚焦回答未覆盖 focus 方法 Class#method",
                    }
                )
            elif "`" not in ans or "#" not in ans:
                gaps.append(
                    {
                        "id": "method_anchors",
                        "desc": "追问聚焦回答未给出 Class#method 锚点",
                    }
                )
            # Dig-in answers must not rehash a numbered full spine.
            if re.search(r"(?m)^\s*1\.\s+`", ans) and re.search(
                r"(?m)^\s*3\.\s+`", ans
            ):
                gaps.append(
                    {
                        "id": "dig_in_full_chain",
                        "desc": "追问聚焦不应输出与全链相同的编号主链",
                    }
                )
        elif (
            evidence.get("dig_in_likely")
            and _allows_entry_scoped_dig_in(evidence)
            and str((evidence.get("dig_in_decision") or {}).get("source") or "")
            in {"timeout", "error", "default"}
            and re.search(r"(?m)^\s*1\.\s+`", ans)
            and re.search(r"(?m)^\s*3\.\s+`", ans)
        ):
            gaps.append(
                {
                    "id": "dig_in_expected",
                    "desc": "入口下钻期望聚焦方法，但回答仍是编号全链",
                }
            )
        else:
            # Require Class#method anchors from the ordered spine (anti-paraphrase).
            spine_cms = [
                str(r.get("class_method") or "").strip()
                for r in rows
                if str(r.get("class_method") or "").strip()
                and "#" in str(r.get("class_method"))
            ]
            hit = 0
            if len(spine_cms) >= 2:
                hit = sum(
                    1
                    for cm in spine_cms[:10]
                    if cm in ans or cm.split("#", 1)[-1] in ans
                )
                if hit < min(2, len(spine_cms)):
                    gaps.append(
                        {
                            "id": "method_anchors",
                            "desc": "流程回答未锚定主链 Class#method（疑似乱序业务套话）",
                        }
                    )
            steps_n = len(re.findall(r"(?m)^\s*\d+\.\s+", ans))
            if len(spine_cms) >= 4 and steps_n < 3:
                gaps.append(
                    {
                        "id": "step_coverage",
                        "desc": (
                            f"主链证据 {len(spine_cms)} 步但回答编号步骤过少 ({steps_n})"
                        ),
                    }
                )
            src = str(evidence.get("spine_source") or "")
            if (
                src in {"calls_closure", "calls_closure_filtered"}
                and len(spine_cms) >= 3
                and hit < 3
                and "执行序" not in ans
            ):
                gaps.append(
                    {
                        "id": "spine_order",
                        "desc": "主链来自无序 CALLS 闭包且回答未体现执行序锚点",
                    }
                )
    if need_change:
        if not evidence.get("change_ok") and "改动" not in ans and "DDL" not in ans:
            gaps.append({"id": "deliverable", "desc": "未产出改动清单"})
        tables = [
            str(t).lower()
            for t in (goal.get("tables") or [])
            if t and re.match(r"(?i)^demo_[a-z0-9_]+$", str(t).strip())
        ]
        if tables and not any(t in ans.lower() for t in tables):
            gaps.append(
                {
                    "id": "persist_tables",
                    "desc": f"回答未覆盖用户点名表: {', '.join(tables)}",
                }
            )
        # Entity binding must match named tables — Ext sibling ≠ user table.
        bound = [str(t).lower() for t in (evidence.get("persist_tables") or []) if t]
        if tables and bound:
            missing_bind = [t for t in tables if t not in bound]
            if missing_bind:
                gaps.append(
                    {
                        "id": "entity_table_mismatch",
                        "desc": (
                            "实体未绑定用户点名表: "
                            + ", ".join(missing_bind)
                            + f"（当前实体表: {', '.join(bound)}）"
                        ),
                    }
                )
        elif tables and not bound:
            # roles without table_name still a gap when user named tables
            roles = evidence.get("change_roles") or []
            if roles and "persist_entity" in roles:
                gaps.append(
                    {
                        "id": "entity_table_mismatch",
                        "desc": "已有实体角色但未读到与用户点名表一致的 @TableName",
                    }
                )
        if goal.get("field_label_zh") and str(goal["field_label_zh"]) not in ans:
            if "COMMENT" in ans or "/**" in ans:
                gaps.append(
                    {
                        "id": "zh_comment",
                        "desc": "注释/COMMENT 未使用问题中的中文含义",
                    }
                )
        qblob = evidence.get("question") or ""
        wants_out = bool(re.search(r"传递|外送|作为入参|调用.{0,12}校验", qblob))
        inter = evidence.get("intersect_cms") or []
        if wants_out and not inter:
            gaps.append(
                {"id": "outbound", "desc": "未找到外送/关卡相关触点代码"}
            )
        elif wants_out and inter:
            # Persist/update helpers are not gate/outbound pass-through.
            top = str(inter[0])
            if re.search(r"(?i)#(update|insert|save)", top) and re.search(
                r"校验", qblob
            ):
                gaps.append(
                    {
                        "id": "outbound_mismatch",
                        "desc": f"外送命中疑似落库更新 `{top}`，不像校验/外呼关卡",
                    }
                )
            if wants_out and not evidence.get("outbound_suggested"):
                # Soft gap: found gate but no pasteable put/set suggestion
                if not re.search(
                    r"(?i)\.put\s*\(|set[A-Z]\w*\s*\(\s*\w+\.get",
                    ans,
                ):
                    gaps.append(
                        {
                            "id": "outbound_snippet",
                            "desc": "未给出外送新增入参的建议代码",
                        }
                    )
    if g == "show_code":
        if "```" not in ans:
            gaps.append({"id": "paste", "desc": "未贴出源码块"})
        # db_lookup / db_pipeline on legacy show_code: skip prefer pin check.
        vk = str(
            goal.get("view_kind")
            or (evidence.get("source_view") or {}).get("view_kind")
            or evidence.get("view_kind")
            or ""
        ).strip().lower()
        if vk in {"db_lookup", "db_pipeline"}:
            pass
        else:
            # If a Class#method target was chosen (intent / dig-in focus), require it
            # in the answer — not just any code fence (anti QueueHelper#submit drift).
            expected: list[str] = []
            for raw in list(evidence.get("prefer_cms") or []) + list(
                evidence.get("dig_in_focus_cms") or []
            ):
                cm = _simple_cm(str(raw))
                if cm and "#" in cm and cm not in expected:
                    expected.append(cm)
            view_hint = _simple_cm(
                str(
                    evidence.get("view_hint")
                    or goal.get("view_hint")
                    or ""
                )
            )
            if view_hint and "#" in view_hint and view_hint not in expected:
                expected.insert(0, view_hint)
            sv = evidence.get("source_view") or {}
            plan_cm = _simple_cm(
                str(
                    (sv.get("plan") or {}).get("class_method")
                    or sv.get("target")
                    or ""
                )
            )
            # Prefer focus/intent expected; fall back to chosen plan target.
            check = expected[:1] or (
                [plan_cm] if plan_cm and "#" in plan_cm else []
            )
            if check:
                cm = check[0]
                meth = cm.split("#", 1)[-1].split("(", 1)[0]
                if cm not in ans and (not meth or meth not in ans):
                    gaps.append(
                        {
                            "id": "target_method",
                            "desc": f"贴码未覆盖目标方法 `{cm}`",
                        }
                    )
    if g == "compose_code":
        if "```" not in ans:
            gaps.append({"id": "compose_paste", "desc": "未生成代码块"})
        pack = evidence.get("compose_code") or {}
        # Soft: prefer answers that cite evidence refs when we had snippets.
        refs = [
            str(x)
            for x in (pack.get("refs") or [])
            if x
        ]
        if refs and not any(r.split("#")[0] in ans for r in refs[:4]):
            # Not a hard fail — synthesis may still be valid without class names.
            pass
        steps = list(goal.get("lookup_steps") or evidence.get("lookup_steps") or [])
        if len(steps) >= 2 and len(ans) < 80:
            gaps.append(
                {
                    "id": "compose_thin",
                    "desc": "多步生成回答过短，可能未覆盖查询编排",
                }
            )
        # 生成代码引用了自定义类（实体/DTO/VO/Mapper）但证据里没有对应定义 →
        # 用户拿到的代码无法编译，典型「没有对应的实体类」。软缺口但需提示。
        snip_cms: set[str] = set()
        for s in (pack.get("snippets") or []):
            cm = str(s.get("class_method") or "").split("#")[0].split(".")[-1]
            if cm:
                snip_cms.add(cm)
            # 实体类名常以字段类型/泛型出现在证据代码里（ServiceImpl 用对应 Entity）。
            code = str(s.get("code") or "")
            for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]{2,60})\b", code):
                name = m.group(1)
                if not _KNOWN_CODE_CLASS_RE.match(name):
                    snip_cms.add(name)
        # refs（Class#method）里的类名 + ServiceImpl/Mapper → 实体名的惯例推导，
        # 避免把「WithdrawalsServiceImpl 引用 Withdrawals 实体」误判为缺失定义。
        for ref in refs:
            cls = str(ref).split("#", 1)[0].split(".")[-1].strip()
            if not cls:
                continue
            snip_cms.add(cls)
            for suf in ("ServiceImpl", "Service", "Mapper", "Repository", "Dao", "Controller"):
                if cls.endswith(suf) and len(cls) > len(suf):
                    snip_cms.add(cls[: -len(suf)])
        kb_text = str(
            (evidence.get("kb_spec") or {}).get("full_text")
            or (evidence.get("kb_block") or "")
            or ""
        )
        custom = _undefined_custom_classes_in_code(ans, known=snip_cms)
        # A class the repo actually defines (e.g. FieldNameConstants, common
        # constants) is NOT an undefined class — fuzzy symbol search misses it
        # but its .java file exists. Drop it from the gap list so compose_code
        # does not refuse valid code over a real class (desktop-chat-ec81c911b4d8).
        if custom:
            try:
                from app.agent.explore_shared import _locate_java_file_for_class

                custom = [
                    c
                    for c in custom
                    if not _locate_java_file_for_class(project_id, c)
                ]
            except Exception:
                logger.debug("compose undef class-file check failed", exc_info=True)
        if custom:
            gaps.append(
                {
                    "id": "compose_undef_class",
                    "desc": (
                        "生成代码引用了未定义的类（实体/DTO/Mapper 缺失）: "
                        + ", ".join(sorted(custom)[:6])
                        + "。若为知识库接口，应走 add_api 脚手架生成完整 ReqVo/RespVo/Controller/Service；"
                        + "若为已有实体，请补充对应类定义或基于仓库证据重写。"
                    ),
                }
            )
        elif kb_text and re.search(r"接口|API", kb_text, re.I) and "Request<" not in ans:
            gaps.append(
                {
                    "id": "compose_kb_api_style",
                    "desc": (
                        "基于知识库接口文档生成代码但未体现项目 Request/Response 统一封装"
                        "（如 Request<ReqVo> / ResponseBuilder）。建议核对项目 Controller 风格。"
                    ),
                }
            )
    if g == "field_value":
        if not evidence.get("entry"):
            gaps.append({"id": "entry", "desc": "未锁定系统入口"})
        pack = evidence.get("field_pack") or {}
        if not pack.get("set_hits") and not pack.get("tfc_hits"):
            gaps.append(
                {
                    "id": "field_ev",
                    "desc": "主链上未命中 method_var_resolve/table_field_changes 字段证据",
                }
            )
        if len(ans) < 40 or "字段证据" not in ans:
            gaps.append({"id": "answer", "desc": "字段回答过短或未按证据模板输出"})
    if g == "api_io":
        if not evidence.get("entry"):
            gaps.append({"id": "entry", "desc": "未锁定系统入口"})
        pack = evidence.get("api_io_pack") or {}
        req_n = len(pack.get("request_params") or [])
        resp_n = len(pack.get("response_params") or [])
        if req_n + resp_n < 1:
            gaps.append(
                {
                    "id": "io_ev",
                    "desc": "未从入口/辅助方法抽到 reqMap.get / response put 等参数证据",
                }
            )
        if len(ans) < 40:
            gaps.append({"id": "answer", "desc": "接口入出参回答过短"})
    if g == "diagnose":
        if not evidence.get("suspects"):
            gaps.append({"id": "map_code", "desc": "未映射到嫌疑符号"})

    return {
        "ok": len(gaps) == 0,
        "gaps": gaps,
        "checked": [m.get("id") for m in (goal.get("must") or [])],
    }


def _pinnable_controller_entry(raw: str | None) -> str:
    """Return Class#method when raw looks like a non-helper Controller facade."""
    from app.agent.ask_retrieval import is_controller_helper_method
    from app.agent.explore_agent import _class_method_from_hint

    cm = _class_method_from_hint(str(raw or "")) or _simple_cm(str(raw or ""))
    if not cm or "#" not in cm:
        return ""
    cls = cm.split("#", 1)[0].split(".")[-1]
    if not cls.endswith("Controller"):
        return ""
    if is_controller_helper_method(cm):
        return ""
    return cm


def _explicit_class_method_from_question(question: str) -> str:
    """Explicit Class#method anchor named in the question (WithdrawalsServiceImpl#loan)."""
    try:
        from app.agent.explore_source import _extract_explicit_class_method

        return str(_extract_explicit_class_method(question or "") or "").strip()
    except Exception:
        logger.debug("explicit class#method extraction failed", exc_info=True)
        return ""


async def _exec_resolve_entry(
    project_id: str, question: str, evidence: dict[str, Any]
) -> None:
    from app.agent.ask_retrieval import resolve_flow_entries_with_meta
    from app.agent.explore_agent import _class_method_from_hint

    hint_raw = str(evidence.get("entry_hint") or "").strip() or None
    hint = _class_method_from_hint(hint_raw) or ""
    # Pin order: lock_entry hint → view_hint → prefer_cms[0].
    # Do NOT soft-pin Class#method entry_hint without lock — session last_entry
    # pollution (wrong sibling Controller) would skip free search forever.
    pin_candidates: list[str | None] = []
    if evidence.get("lock_entry"):
        pin_candidates.append(hint_raw)
    pin_candidates.append(str(evidence.get("view_hint") or "") or None)
    prefer = list(evidence.get("prefer_cms") or [])
    if prefer:
        pin_candidates.append(str(prefer[0]) or None)
    # show_code / dig-in view target: the view_hint IS the display target even
    # when it is a ServiceImpl method (not a *Controller facade). Pinning it
    # prevents a free-search drift to an unrelated entry (e.g. Product feign
    # decorator) when the user asks to see a specific method's code.
    # view_hint is only ever set under lock / dig-in / show_code (see
    # pev_agent.extract_goal_contract), so its presence is the gate.
    view_hint_raw = str(evidence.get("view_hint") or "").strip()
    view_hint_cm = (
        _class_method_from_hint(view_hint_raw) or _simple_cm(view_hint_raw)
    )
    is_view_target = bool(
        view_hint_cm
        and "#" in view_hint_cm
        and evidence.get("view_hint")
    )
    for raw in pin_candidates:
        pinned = _pinnable_controller_entry(raw)
        if pinned:
            evidence["entry"] = pinned
            evidence["entry_matches"] = [pinned]
            evidence["entry_match_meta"] = []
            # Locked interface/flow ask: still attach same-method siblings (multi-reply).
            if evidence.get("lock_entry") and re.search(
                r"接口|流程|入口", question or ""
            ):
                try:
                    from app.agent.ask_retrieval import (
                        _method_name_of_cm,
                        _order_entries_primary_first,
                    )

                    matches, meta = resolve_flow_entries_with_meta(
                        project_id,
                        question,
                        entry_hint=None,
                        prefer_hint=False,
                    )
                    method = _method_name_of_cm(pinned)
                    sibs = [
                        m for m in matches if _method_name_of_cm(m) == method
                    ]
                    if len(sibs) >= 2:
                        evidence["entry_matches"] = _order_entries_primary_first(
                            sibs, pinned
                        )
                        want = set(evidence["entry_matches"])
                        by_cm = {
                            str(h.get("cm") or ""): h for h in (meta or [])
                        }
                        evidence["entry_match_meta"] = [
                            by_cm[cm] for cm in evidence["entry_matches"] if cm in by_cm
                        ]
                except Exception:
                    logger.debug("sibling expand after lock pin failed", exc_info=True)
            return
        # Non-Controller view target under show_code / dig-in: pin it directly.
        if is_view_target and pinned == "":
            vh = _class_method_from_hint(str(raw or "")) or _simple_cm(str(raw or ""))
            if vh and "#" in vh and vh == view_hint_cm:
                evidence["entry"] = vh
                evidence["entry_matches"] = [vh]
                evidence["entry_match_meta"] = []
                return
    # Dig-in under session lock with non-facade hint: still keep exact hint.
    if evidence.get("lock_entry") and hint and "#" in hint:
        evidence["entry"] = hint
        evidence["entry_matches"] = [hint]
        evidence["entry_match_meta"] = []
        return

    # Explicit Class#method anchor in the question itself
    # (WithdrawalsServiceImpl#loan 这个方法主要做了什么): method-level asks must
    # answer THAT method. The flow-entries resolver returns only
    # Controller/Job/Listener facades (never Service helpers), and the fuzzy
    # symbol ranking can drop the exact method entirely, so a Controller sibling
    # wins by default and the answer drifts to the whole business chain
    # (desktop-chat-0f3cb1d6b410: answered the withdrawal chain instead of #loan).
    # Pin the named symbol directly when it exists in the index.
    if not evidence.get("entry"):
        explicit = _explicit_class_method_from_question(question)
        if explicit and "#" in explicit and _entry_is_real(project_id, explicit):
            cls = explicit.split("#", 1)[0].split(".")[-1]
            is_controller = cls.endswith("Controller")
            method_level = bool(
                re.search(
                    r"这个方法|方法[做是干]|做了什么|干嘛|干什么|什么逻辑|逻辑是|功能是|"
                    r"是干什么|主要做|负责.*什么|作用是什么|在做啥|干了些什么",
                    question or "",
                )
            )
            flow_level = bool(
                re.search(
                    r"流程|主链|链路|全链路|串一下|调用链|接口|入口|位置|在哪",
                    question or "",
                )
            )
            if (method_level or not is_controller) and not flow_level:
                evidence["entry"] = explicit
                evidence["entry_matches"] = [explicit]
                evidence["entry_match_meta"] = []
                return

    # Topic grounding: bare Chinese topic without typed Class#method.
    grounded = _topic_ground_entry_candidates(project_id, question)
    if grounded.get("phrase"):
        evidence["topic_ground_phrase"] = grounded["phrase"]
    if grounded.get("controllers"):
        evidence["topic_ground_controllers"] = list(grounded["controllers"])

    matches, meta = resolve_flow_entries_with_meta(
        project_id,
        question,
        entry_hint=hint_raw,
        prefer_hint=bool(evidence.get("lock_entry")),
    )
    matches, meta = _apply_topic_ground_to_entries(
        project_id,
        question,
        matches=list(matches or []),
        meta=list(meta or []),
        grounded=grounded,
    )
    evidence["entry_matches"] = list(matches or [])
    evidence["entry_match_meta"] = list(meta or [])
    evidence["entry"] = matches[0] if matches else ""
    if grounded.get("controllers") and matches:
        evidence["entry_source"] = "topic_ground"
    elif grounded.get("controllers") and not matches:
        evidence["entry_source"] = "topic_ground_miss"
    # Bare-topic / method-name asks resolve via symbol search. Two cases:
    #   (a) no HTTP entry at all (风控/loanSwitch) → fallback;
    #   (b) weak HTTP entry with zero topic overlap while symbol search is a
    #       much better fit (loanSwitch→RouteController#route is wrong).
    sym_best = _resolve_entry_via_symbol_search(project_id, question)
    sym, sym_score = (sym_best or ("", 0.0))
    # Topic grounding is authoritative (it already mapped 电子签章→SignatureController):
    # never let symbol fallback/override replace a grounded candidate.
    # EXCEPTION: when the grounded/current entry is a known "wrong-for-放款" Controller
    # (AmountController tagged「渠道日放款额」, not the real 放款 flow), keep the symbol
    # search result so the is_wrong_controller cross-Controller override below can
    # correct the entry instead of answering a quota-adjust sub-routine.
    if grounded.get("controllers"):
        cur_cls = (
            str(evidence.get("entry") or "").split("#", 1)[0]
            if evidence.get("entry")
            else ""
        )
        wrong_grounded = bool(
            cur_cls in ("AmountController", "QuotaController", "LimitController")
            and _get_loan_action_re().search(question or "")
        )
        if not wrong_grounded:
            sym, sym_score = "", 0.0
    cur_entry = str(evidence.get("entry") or "")
    # Synthesized-facade guard: QuotaController#quota / RiskController#risk have
    # no real method in the index — clear the entry so symbol fallback picks a
    # real one instead of failing with「未能定位到系统入口」.
    if cur_entry and not _entry_is_real(project_id, cur_entry):
        evidence["entry"] = ""
        evidence["entry_matches"] = []
        cur_entry = ""
        evidence["entry_source"] = "facade_cleared"
    cur_score = _entry_topic_score(cur_entry, question) if cur_entry else 0.0
    # Full-flow asks keep the HTTP entry (放款流程 → WithdrawalsController#withdrawals).
    # Method-level / calc topics (loanSwitch/MIY/提现试算) may override a weak entry —
    # but a "calc" ask may only override onto a real *Calc* method (queryWithdrawals
    # Result is a query, not 试算; keep the main entry instead).
    calc_word = bool(re.search(r"试算|计算|预估|测算", question or ""))
    sym_is_calc = bool(sym and re.search(r"(?i)calc", str(sym).split("#", 1)[-1]))
    # Even full-flow asks (放款接口什么流程) allow symbol-search override when the
    # topic-grounded entry has near-zero topic overlap (cur_score < 0.5) — meaning
    # the topic grounding likely mapped the Chinese phrase to the wrong Controller
    # (AmountController for 放款).  The explicit_anchor guard (below) still blocks
    # override at the elif level when the user named a specific class by hand.
    #
    # Additionally, when the grounded entry is a known "wrong-for-放款" Controller
    # such as AmountController (tagged「渠道日放款额」, not the real 放款 flow),
    # cross-Controller override is permitted unconditionally so the answer
    # describes the real loan-dispersal flow instead of a quota-adjust sub-routine.
    topic_already_matched = cur_score >= 0.5 if cur_entry else False
    cur_simple_class = (cur_entry or "").split("#", 1)[0] if cur_entry else ""
    sym_simple_class = (sym or "").split("#", 1)[0] if sym else ""
    # Change-plan / add-field questions must never override a resolved main entry:
    # the anchor is the interface the user named (e.g. 放款接口), and symbol search
    # may return an unrelated method from another class entirely.
    from app.agent.question_intent import (
        wants_add_field_change_plan,
        wants_composite_change,
        wants_extend_field_to_protocol_sync,
    )

    _is_change_plan = bool(
        wants_add_field_change_plan(question)
        or wants_composite_change(question)
        or wants_extend_field_to_protocol_sync(question)
    )
    is_wrong_controller = bool(
        cur_simple_class
        and sym_simple_class
        and cur_simple_class != sym_simple_class
        and cur_simple_class in ("AmountController", "QuotaController", "LimitController")
        and _get_loan_action_re().search(question or "")
    )
    allow_override = (
        (calc_word and sym_is_calc)
        or is_wrong_controller
        or not re.search(r"流程|主链|链路|全链路|串一下", question or "")
        or not topic_already_matched
    ) and not _is_change_plan

    # An explicit anchor (Class#method / Controller/Service/Mapper name) must be
    # respected — never override a resolve result the user pointed at by name.
    # Bare「接口」or「入口」preceded by CJK (放款接口) is a business-topic phrase,
    # not an explicit code anchor — only treat as anchor when preceded by a Latin
    # identifier like AmountController接口 or at start-of-string (接口 path).
    explicit_anchor = bool(
        re.search(
            r"#"
            r"|[A-Za-z][A-Za-z0-9_]*(?:Controller|Service|Mapper|Client|Listener)\b"
            r"|(?:(?<=^)|(?<![\u4e00-\u9fff]))(?:接口|入口)(?![\u4e00-\u9fff])",
            question or "",
        )
    )
    if not cur_entry and sym:
        evidence["entry"] = sym
        evidence["entry_matches"] = [sym]
        evidence["entry_source"] = "symbol_search_fallback"
    elif (
        is_wrong_controller
        and not _is_change_plan
        and sym
        and not explicit_anchor
        and sym != cur_entry
        and str(sym).split("#", 1)[0].endswith("Controller")
    ):
        # 已知错误 Controller（AmountController 等，标注「渠道日放款额」）：
        # 只要符号搜索给出其它 Controller 的真实入口，就直接跨类覆盖，
        # 不再要求 sym_score>=2.5（弱匹配也应让位于真实放款流程）。
        evidence["entry"] = sym
        evidence["entry_matches"] = [sym]
        evidence["entry_source"] = "symbol_search_wrong_controller_fix"
    elif (
        allow_override
        and not explicit_anchor
        and cur_score < 2.0
        and sym_score >= 2.5
        and sym != cur_entry
    ):
        evidence["entry"] = sym
        evidence["entry_matches"] = [sym]
        evidence["entry_source"] = "symbol_search_override"
    elif not cur_entry and grounded.get("controllers") and question.strip():
        # Facade cleared but topic grounding named a class — pick the real method
        # of that class that best matches the ask (QuotaController#quota → the
        # real quotaAdjust / adjustResult rather than a synthesized facade).
        real = _pick_real_class_method(
            project_id, str(grounded["controllers"][0]), question
        )
        if real:
            evidence["entry"] = real
            evidence["entry_matches"] = [real]
            evidence["entry_source"] = "facade_class_method"


def _pick_real_class_method(project_id: str, controller_fqcn: str, question: str) -> str:
    """Best real method of a Controller class for the ask (by summary overlap)."""
    from app.agent.explore_tools import search_symbols

    cls = str(controller_fqcn).split(".")[-1].strip()
    if not cls:
        return ""
    try:
        sr = search_symbols(project_id, cls, kind="Controller", top_k=20)
    except Exception:
        return ""
    best_cm, best_score = "", 0.0
    for h in sr.get("hits") or []:
        fqcn = str(
            (h.get("fqcn") if isinstance(h, dict) else getattr(h, "fqcn", "")) or ""
        )
        if fqcn.split(".")[-1] != cls:
            continue
        sig = str(
            (h.get("method_sig") if isinstance(h, dict) else getattr(h, "method_sig", ""))
            or ""
        )
        meth = sig.split("(", 1)[0].strip() if sig else ""
        summary = str(
            (h.get("summary") if isinstance(h, dict) else getattr(h, "summary", "")) or ""
        )
        if not meth:
            continue
        cm = f"{cls}#{meth}"
        if not _entry_is_real(project_id, cm):
            continue
        sc = _cjk_overlap_score(summary, question) + _token_overlap_score(
            f"{cm} {summary}", question
        )
        if sc > best_score:
            best_cm, best_score = cm, sc
    return best_cm if best_score > 0 else ""


def _entry_is_real(project_id: str, cm: str) -> bool:
    """True when the Class#method actually exists in the symbol index.

    Guards synthesized facades (FooController#foo with no real foo method) that
    entry_route / topic grounding can produce for bare-topic asks.
    """
    if not project_id or "#" not in str(cm):
        return False
    try:
        from app.agent.ask_retrieval import _resolve_symbol_id

        return bool(_resolve_symbol_id(project_id, str(cm)))
    except Exception:
        return True  # fail-open when the index lookup itself fails


def _entry_topic_score(cm: str, question: str) -> float:
    """Topic overlap between a Class#method and the ask (no summary available yet)."""
    if not cm or not question:
        return 0.0
    return float(
        _token_overlap_score(cm, question) + _cjk_overlap_score(cm, question)
    )


def _resolve_entry_via_symbol_search(
    project_id: str, question: str, min_score: float = 1.2
) -> tuple[str, float] | None:
    """Fallback entry: symbol search for bare-topic / method-name asks.

    Prefers ServiceImpl/Service/Listener/Client methods (CALLS-expandable) whose
    name or class matches question literals (loanSwitch → …#loanSwitch), then
    topic overlap on method summaries. Returns (Class#method, score) or None.
    """
    from app.agent.explore_tools import search_symbols

    q = (question or "").strip()
    if not q:
        return None
    literals = list(
        dict.fromkeys(
            re.findall(r"[A-Za-z][A-Za-z0-9_]{2,40}", q)
        )
    )[:4]
    # Persist bridge: 落库/入库/写入/写哪 ↔ insert*/update*/save* methods
    # (generic EN↔CN correspondence, cross-domain safe — 还款落库 → RepayServiceImpl#insert…).
    persist_hint = bool(
        re.search(r"落库|入库|写入|持久化|写哪|写库|写表|落到|落表|存到|保存到|新增列", q)
    )
    # CJK topic variants: 「风控是怎么做的」 → 风控 / 风控是怎么做的, plus a
    # 2-gram window so mid-sentence business words (放款/接口/字段/落库) become
    # queries — run[:2] alone only took the sentence prefix (我想) and missed
    # the actual topic, drifting change-plan asks to unrelated entries.
    cjk_extra: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]{2,16}", q)[:3]:
        if len(run) >= 4:
            cjk_extra.append(run[:2])
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            if gram not in cjk_extra:
                cjk_extra.append(gram)
        cjk_extra.append(run)
    queries = [*literals, *cjk_extra[:8], q][:14]
    best_cm, best_score = "", 0.0
    for qq in queries:
        try:
            sr = search_symbols(project_id, qq, top_k=10)
        except Exception:
            continue
        for h in sr.get("hits") or []:
            fqcn = str(
                (h.get("fqcn") if isinstance(h, dict) else getattr(h, "fqcn", ""))
                or ""
            )
            simple = fqcn.split(".")[-1]
            meth = str(
                (h.get("method_name") if isinstance(h, dict) else getattr(h, "method_name", ""))
                or ""
            )
            sig = str(
                (h.get("method_sig") if isinstance(h, dict) else getattr(h, "method_sig", ""))
                or ""
            )
            # Search hits carry the method name inside method_sig (e.g.
            # quotaCheckMIY(WithdrawalsReqVo, ...)) — method_name is often None.
            if "#" in sig:
                meth = sig.split("#", 1)[1].split("(", 1)[0].strip() or meth
            elif not meth and sig:
                meth = sig.split("(", 1)[0].strip()
            summary = str(
                (h.get("summary") if isinstance(h, dict) else getattr(h, "summary", ""))
                or ""
            )
            if not meth:
                continue
            cm = f"{simple}#{meth}"
            # Impl / service / listener / client leaves get a small preference;
            # Controller entries are still valid answers (风控回调入口 etc).
            impl_leaf = bool(re.search(
                r"(?i)(ServiceImpl|Service|Listener|Client|Job|Processor|Handler)$",
                simple,
            ))
            score = 0.0
            # Exact method-name literal match dominates (MIY→quotaCheckMIY beats
            # the generic checkUseCreditAmount that only shares 额度/校验 2-grams).
            # Substring containment must score strictly lower than an exact method
            # name — 'loan' was tied 8.0 with both WithdrawalsServiceImpl#loan and
            # #acmCompareLoan, letting a same-class sibling win (desktop-chat-0f3cb1d6b410).
            for lit in literals:
                ml = (meth or "").lower()
                ll = (lit or "").lower()
                if ml == ll:
                    score += 16.0
                elif len(ll) >= 3 and (ll in ml or ml in ll):
                    score += 8.0
                elif ll in simple.lower():
                    score += 2.0
            # CJK topic overlap on summary (full weight — the search already
            # picked topic-relevant rows; diluting by 0.5 starved 风控→callBack).
            if summary:
                score += min(_cjk_overlap_score(summary, q), 8.0) + (
                    _token_overlap_score(f"{cm} {summary}", q) * 0.3
                )
                # Topic-density tie-break: a summary that *is about* the topic
                # beats one that merely mentions it (quotaAdjust mentions 风控
                # once; RiskController#callBack is the actual risk callback).
                score += min(_topic_summary_specificity(summary, q), 4.0) * 0.2
            if impl_leaf:
                score += 0.5
            # Structural domain hint: class/method names carrying "risk" win
            # over incidental topic mentions in other domains.
            if re.search(r"(?i)risk", f"{simple} {meth}"):
                score += 0.5
            # Persist-ask bridge: prefer insert/update/save methods for 落库 asks.
            if persist_hint and re.search(r"(?i)(insert|update|save|persist|write)", meth):
                score += 2.0
            # Reverse-operation methods (退款/取消/冲正/删除) contradict an
            # add-field/persist ask — demote them (LoanRefundServiceImpl drifted
            # in on shared 放款 2-grams).
            if persist_hint and re.search(
                r"(?i)(refund|cancel|reverse|rollback|revoke|delete|stop)",
                f"{simple} {meth}",
            ):
                score -= 3.0
            if score > best_score:
                best_cm, best_score = cm, score
    if not best_cm or best_score < min_score:
        return None
    return best_cm, best_score


def _cm_simple_class(cm: str) -> str:
    text = (cm or "").strip()
    if not text:
        return ""
    left = text.split("#", 1)[0]
    return left.split(".")[-1].strip()


def _topic_ground_entry_candidates(
    project_id: str, question: str
) -> dict[str, Any]:
    """Structural topic → Controller names (no product lexicon)."""
    out: dict[str, Any] = {"phrase": "", "controllers": [], "primary": []}
    try:
        from app.agent.codebase_survey import (
            needs_topic_entry_grounding,
            extract_flow_topic_phrase,
            run_topic_existence_probe,
        )
    except Exception:
        return out
    if not needs_topic_entry_grounding(question):
        return out
    phrase = extract_flow_topic_phrase(question)
    if not phrase:
        return out
    out["phrase"] = phrase
    try:
        pack = run_topic_existence_probe(project_id, f"有{phrase}业务吗")
    except Exception:
        logger.debug("topic ground probe failed", exc_info=True)
        return out
    primary = list(pack.get("primary_hits") or [])
    out["primary"] = primary
    ctrls: list[str] = []
    seen: set[str] = set()
    for h in primary:
        cls = str(h.get("class") or "")
        if not cls.endswith("Controller") or cls in seen:
            continue
        seen.add(cls)
        ctrls.append(cls)
    # Fallback: any impl class ending with Controller from source_hits.
    if not ctrls:
        for h in list(pack.get("source_hits") or []):
            cls = str(h.get("class") or "")
            if cls.endswith("Controller") and cls not in seen:
                seen.add(cls)
                ctrls.append(cls)
    # Fallback 2: index_hits (symbol index search with summaries / descriptions).
    # The source-text literal search may miss Controllers whose methods use
    # English names (retryLoan / disbursement etc.) while the symbol index
    # has richer Chinese summaries that match the topic phrase.
    if not ctrls:
        for h in list(pack.get("index_hits") or []):
            cls = str(h.get("class") or "")
            if cls.endswith("Controller") and cls not in seen:
                seen.add(cls)
                ctrls.append(cls)
    out["controllers"] = ctrls[:8]
    return out


def _has_main_entry(matches: list[str], meta: list[dict]) -> bool:
    """Any match is a real main HTTP entry (has entry_route evidence)."""
    from app.agent.ask_retrieval import is_controller_http_facade

    cm_set = set(matches)
    for h in meta:
        cm = str(h.get("cm") or "")
        if cm in cm_set:
            if h.get("is_http_facade") or h.get("flow_role") == "main":
                return True
            if is_controller_http_facade(cm):
                return True
    return False


def _apply_topic_ground_to_entries(
    project_id: str,
    question: str,
    *,
    matches: list[str],
    meta: list[dict[str, Any]],
    grounded: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Prefer topic Controllers; never keep unrelated weak locks when topic hit."""
    ctrls = [str(c) for c in (grounded.get("controllers") or []) if c]
    if not ctrls:
        return matches, meta
    ctrl_set = set(ctrls)
    phrase = str(grounded.get("phrase") or "").strip()

    def _in_topic(cm: str) -> bool:
        return _cm_simple_class(cm) in ctrl_set

    filtered = [m for m in matches if _in_topic(m)]
    if filtered:
        want = set(filtered)
        meta_f = [h for h in meta if str(h.get("cm") or "") in want]
        return filtered, meta_f or meta

    # Original resolve matches exist but none are "in topic" (the topic Controllers
    # list is structural — may miss English-named Controllers like WithdrawalsController
    # for 放款 questions).  Keep them if they are real main HTTP entries — never replace
    # with arbitrary helper methods from unrelated topic Controllers.
    if matches and _has_main_entry(matches, meta):
        return matches, meta

    # Free resolve missed topic Controllers — resolve with phrase + Controller hint.
    from app.agent.ask_retrieval import resolve_flow_entries_with_meta

    for ctrl in ctrls[:4]:
        try:
            m2, meta2 = resolve_flow_entries_with_meta(
                project_id,
                phrase or question,
                entry_hint=ctrl,
                prefer_hint=True,
            )
        except Exception:
            logger.debug("topic ground re-resolve failed ctrl=%s", ctrl, exc_info=True)
            continue
        hit = [m for m in (m2 or []) if _cm_simple_class(m) == ctrl]
        if hit:
            want = set(hit)
            meta_h = [h for h in (meta2 or []) if str(h.get("cm") or "") in want]
            return hit, meta_h or list(meta2 or [])

    # Still no facade: search_symbols for Controller methods.
    try:
        from app.agent.explore_shared import _method_name_from_hit
        from app.agent.explore_tools import search_symbols

        for ctrl in ctrls[:3]:
            sr = search_symbols(project_id, ctrl, top_k=12)
            for h in sr.get("hits") or []:
                if not isinstance(h, dict):
                    continue
                fqcn = str(h.get("fqcn") or "")
                simple = fqcn.split(".")[-1]
                if simple != ctrl:
                    continue
                mname = _method_name_from_hit(h)
                if not mname:
                    continue
                cm = f"{simple}#{mname}"
                # Return the real method as found — never convert to a structural
                # facade name (XxxController#xxx) that may not exist as a real
                # method (e.g. RepayController#repay). The caller's entry_is_real
                # guard will confirm; downstream expand_calls needs a real symbol.
                return [cm], [{"cm": cm, "source": "topic_ground_search"}]
    except Exception:
        logger.debug("topic ground symbol search failed", exc_info=True)

    # Prefer empty over an unrelated Controller (e.g. repay vs signature topic).
    if matches and not any(_in_topic(m) for m in matches):
        return [], []
    return matches, meta


def _simple_cm(cm: str | None) -> str:
    text = (cm or "").strip()
    if not text or "#" not in text:
        return text
    left, right = text.split("#", 1)
    return f"{left.split('.')[-1].strip()}#{right.split('(', 1)[0].strip()}"


# JDK / Spring / MyBatis-Plus / fastjson / lombok / 项目公共类——生成代码引用它们不算缺失定义。
_KNOWN_CODE_CLASS_RE = re.compile(
    r"(?i)^(String|Integer|Long|Double|Float|BigDecimal|Boolean|Byte|Short|Character|"
    r"Object|Number|Date|LocalDate|LocalTime|LocalDateTime|Instant|List|ArrayList|"
    r"LinkedList|Map|HashMap|LinkedHashMap|TreeMap|Set|HashSet|TreeSet|Collection|"
    r"Collections|Arrays|Optional|Stream|Collectors|Objects|Math|Random|UUID|TimeUnit|"
    r"Pattern|Matcher|StringBuilder|StringBuffer|JSON|JSONObject|JSONArray|Page|Rows|"
    r"QueryWrapper|LambdaQueryWrapper|UpdateWrapper|LambdaUpdateWrapper|Wrapper|"
    r"IService|ServiceImpl|BaseMapper|Page|IPage|Response|Request|ResponseBuilder|"
    r"Result|ResultCode|ExceptionEnum|LoanBusinessException|BusinessException|RuntimeException|"
    r"Exception|Throwable|Error|Controller|RestController|Service|Repository|Component|"
    r"Configuration|Bean|Autowired|Resource|Value|Qualifier|Transactional|Slf4j|Data|"
    r"Getter|Setter|Builder|AllArgsConstructor|NoArgsConstructor|ApiOperation|Api|"
    r"RequestBody|RequestParam|PathVariable|ResponseBody|GetMapping|PostMapping|"
    r"PutMapping|DeleteMapping|RequestMapping|Valid|Validated|NotBlank|NotNull|"
    r"NotEmpty|Size|Min|Max|Pattern|DateTimeFormat|JsonFormat|ConstraintViolation|"
    r"BindingResult|HttpServletRequest|HttpServletResponse|ServletContext|MultipartFile|"
    r"ModelAndView|RedirectView|UriComponentsBuilder|Locale|TimeZone|Charset|StandardCharsets|"
    r"ObjectUtil|StringUtils|CollectionUtils|BeanUtils|BeanUtil|DateUtil|JSONUtil|"
    r"TypeUtils|ParseConfig|SerializeConfig|Feature|SerializerFeature|Field|Method|"
    r"Class|Thread|Runnable|Callable|Future|CompletableFuture|Executor|ExecutorService|"
    r"Executors|ScheduledExecutorService|AtomicInteger|AtomicLong|AtomicBoolean|"
    r"BigInteger|Comparable|Comparator|Iterable|Iterator|Enumeration|Hashtable|"
    r"Stack|Vector|Deque|ArrayDeque|PriorityQueue|Queue|BlockingQueue|LinkedBlockingQueue|"
    r"ConcurrentHashMap|CopyOnWriteArrayList|CopyOnWriteArraySet|Timer|TimerTask|"
    r"Properties|ResourceBundle|Scanner|Console|System|Runtime|Process|ProcessBuilder|"
    r"File|Path|Paths|Files|InputStream|OutputStream|Reader|Writer|BufferedReader|"
    r"BufferedWriter|FileReader|FileWriter|ByteArrayInputStream|ByteArrayOutputStream|"
    r"SqlSessionFactory|SqlSession|MappedStatement|RowBounds|ResultHandler|StatementHandler|"
    r"ParameterHandler|TypeHandler|Connection|PreparedStatement|ResultSet|DriverManager|"
    r"Order|Sort|SortBy|Specification|Entity|Id|TableName|TableId|TableField|EnumValue|"
    r"JsonIgnore|JsonInclude|JsonProperty|JsonAlias|JsonFormat|JacksonAnnotation|"
    r"Value|Alias|Since|Until|Deprecated|SuppressWarnings|Override|FunctionalInterface|"
    r"SafeVarargs|Native|Strictfp|Transient|Volatile|Synchronized|Public|Private|Protected|"
    r"FieldNameConstants|Wrappers|ObjectUtils|Mapper|Dao|Repository|"
    r"LambdaQuery|Query|Update|LambdaUpdate|ChainQuery|ChainUpdate|ServiceImpl|IService|"
    r"TableInfo|TableFieldInfo|SqlHelper|Db|DbUtil|SqlRunner)"
    r"$"
)


def _undefined_custom_classes_in_code(answer: str, *, known: set[str]) -> list[str]:
    """PascalCase class references in java code fences not covered by evidence/JDK.

    Picks generic custom-looking identifiers (e.g. ``Goods``, ``GoodsQueryRequest``,
    ``GoodsMapper``) that the synthesized code uses but no evidence snippet defines.
    Returns sorted unique class names; empty when none found.
    """
    known = {str(k).strip() for k in (known or set()) if str(k).strip()}
    found: set[str] = set()
    for fence in re.findall(r"```(?:java)?\s*(.*?)```", answer or "", re.S):
        # References: type in generics/declarations, `new Xxx(` or `Xxx::` or `Xxx.` etc.
        for m in re.finditer(r"\b([A-Z][A-Za-z0-9_]{2,60})\b", fence):
            name = m.group(1)
            # All-caps tokens are constants (SUCCESS / FAIL / VALID), not classes.
            if re.fullmatch(r"[A-Z][A-Z0-9_]{2,60}", name):
                continue
            if _KNOWN_CODE_CLASS_RE.match(name):
                continue
            if name in known:
                continue
            found.add(name)
    return sorted(found)


def _is_spine_noise(cm: str) -> bool:
    return bool(_SPINE_NOISE_RE.search(cm or ""))


def _filter_spine_cms(cms: list[str], *, limit: int = 14) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in cms:
        cm = _simple_cm(raw)
        if not cm or "#" not in cm or cm in seen or _is_spine_noise(cm):
            continue
        seen.add(cm)
        out.append(cm)
        if len(out) >= limit:
            break
    return out


def _resolve_flow_spine(
    project_id: str, entry_cm: str
) -> tuple[list[str], str]:
    """Return (ordered callees after entry, source tag). Prefer FOLLOWS / L1 CALLS order."""
    entry = _simple_cm(entry_cm)
    if not project_id or not entry:
        return [], ""

    # 1) FOLLOWS / mysql spine / ordered L1 via get_follows_path
    try:
        from app.analysis.flow_neo4j_sync import get_follows_path

        path = get_follows_path(project_id, entry)
        ordered = [
            _simple_cm(x)
            for x in (path.get("ordered_path") or [])
            if x and "#" in str(x)
        ]
        if len(ordered) >= 2:
            rest = _filter_spine_cms(
                [c for c in ordered if c != entry], limit=14
            )
            if rest:
                src = str(path.get("source") or "follows")
                return rest, src
    except Exception:
        logger.debug("PEV get_follows_path failed", exc_info=True)

    # 2) Direct L1 CALLS with r.order (AST topo)
    try:
        from app.analysis.flow_neo4j_sync import _ordered_calls_l1_spine

        l1 = [_simple_cm(x) for x in _ordered_calls_l1_spine(project_id, entry) if x]
        if len(l1) >= 2:
            rest = _filter_spine_cms([c for c in l1 if c != entry], limit=14)
            if rest:
                return rest, "neo4j_calls_l1"
    except Exception:
        logger.debug("PEV ordered L1 CALLS failed", exc_info=True)

    # 3) Last resort: CALLS closure, drop noise (order still imperfect)
    try:
        from app.agent.ask_retrieval import _calls_closure_cms

        raw = _calls_closure_cms(project_id, entry) if entry else []
        rest = _filter_spine_cms(raw, limit=14)
        if rest:
            return rest, "calls_closure_filtered"
        return [], "empty"
    except Exception:
        logger.debug("PEV CALLS closure failed", exc_info=True)
        return [], "empty"


async def _exec_expand_calls(
    project_id: str, evidence: dict[str, Any], question: str = ""
) -> None:
    entry = evidence.get("entry") or ""
    matches = [
        _simple_cm(str(x))
        for x in (evidence.get("entry_matches") or [])
        if x and "#" in str(x)
    ]
    if entry and _simple_cm(entry) not in matches:
        matches = [_simple_cm(entry), *matches]
    if not matches and entry:
        matches = [_simple_cm(entry)]

    spines: dict[str, dict[str, Any]] = {}
    for cm in matches[:4]:
        calls, source = _resolve_flow_spine(project_id, cm) if cm else ([], "")
        spines[cm] = {"calls": calls, "spine_source": source, "calls_count": len(calls)}

    evidence["entry_spines"] = spines
    primary = _simple_cm(entry) or (matches[0] if matches else "")
    primary_spine = spines.get(primary) or {"calls": [], "spine_source": "", "calls_count": 0}
    evidence["calls"] = list(primary_spine.get("calls") or [])
    evidence["calls_count"] = int(primary_spine.get("calls_count") or 0)
    evidence["spine_source"] = str(primary_spine.get("spine_source") or "")

    # A synthesized facade entry (RiskController#risk) has no real CALLS spine,
    # and a weak 1-node spine (QuotaController#adjustResult) is often a leaf
    # while the main entry (quotaAdjust) has a rich chain. For bare-topic asks,
    # recover via symbol search and expand the real implementation — only when
    # it yields a richer spine.
    #
    # IMPORTANT: guard against cross-Controller overwrite.  If the symbol search
    # returns a method from a *different* Controller class, the original entry
    # (which topic-grounding / entry resolution already matched to the question)
    # must NOT be silently replaced — that would make the answer describe a
    # completely different business flow (desktop-chat-8f2f3f715f23:
    # AmountController#addDayQuota → WithdrawalsController#retryLoan).
    if int(evidence.get("calls_count") or 0) < 2 and question.strip():
        sym = _resolve_entry_via_symbol_search(project_id, question, min_score=0.8)
        if sym and sym[0] and _simple_cm(sym[0]) != _simple_cm(entry):
            # Extract the simple class name (last segment before #) from both entries.
            _orig_cls = _simple_cm(entry).split("#", 1)[0].split(".")[-1] if "#" in _simple_cm(entry) else ""
            _new_cls = _simple_cm(sym[0]).split("#", 1)[0].split(".")[-1] if "#" in _simple_cm(sym[0]) else ""
            # Only allow spine recovery when both entries belong to the same
            # Controller class — a different class means a different business
            # entry, and overwriting would answer the wrong flow.
            if _orig_cls and _new_cls and _orig_cls == _new_cls:
                calls, source = _resolve_flow_spine(project_id, sym[0])
                if calls and len(calls) > int(evidence.get("calls_count") or 0):
                    evidence["entry"] = sym[0]
                    evidence["entry_matches"] = [sym[0]]
                    evidence["entry_source"] = "symbol_search_spine_recovery"
                    evidence["calls"] = list(calls)
                    evidence["calls_count"] = len(calls)
                    evidence["spine_source"] = source or ""


def _wants_dig_in_change(question: str) -> bool:
    """Short dig-in asks about what to change under a locked entry."""
    return bool(
        re.search(
            r"要改什么|改哪些|怎么改|修改什么|改动点|影响哪些|改哪里",
            question or "",
        )
    )


def _is_method_dig_in_ask(question: str) -> bool:
    """Optional fast-path hint: change dig-in or Latin identifier present.

    Dig-in under ``lock_entry`` prefers description match; LLM is fallback only.
    Latin method tokens are language-agnostic and remain a cheap signal only.
    """
    q = question or ""
    return bool(
        _wants_dig_in_change(q)
        or re.search(r"(?i)\b[A-Za-z][A-Za-z0-9_]{2,40}\b", q)
    )


def _allows_entry_scoped_dig_in(evidence: dict[str, Any]) -> bool:
    """True when dig-in focus helpers may run (session lock / dig-in flags).

    Dig-in is gated by lock_entry, dig_in_likely, dig_in_focus, or ask_scope
    already set from session / lock compress — not by Goal LLM classification.
    """
    if evidence.get("lock_entry"):
        return True
    if evidence.get("dig_in_likely") or evidence.get("dig_in_focus"):
        return True
    scope = str(evidence.get("ask_scope") or "").strip().lower()
    return scope in {"entry_scoped_dig_in", "dig_in"}


def _resolve_method_topic(question: str, evidence: dict[str, Any]) -> str:
    """Subtopic probe for method-first dig-in (LLM fields preferred; no product nouns)."""
    for key in ("method_topic", "focus_topic"):
        t = str(evidence.get(key) or "").strip()
        if t:
            return t[:120]
    q = (question or "").strip()
    if not q:
        return ""
    if not (
        evidence.get("dig_in_likely")
        or evidence.get("dig_in_focus")
        or _allows_entry_scoped_dig_in(evidence)
    ):
        return ""
    try:
        from app.agent.ask_retrieval import _dig_in_search_query

        return (_dig_in_search_query(q) or q)[:120]
    except Exception:
        return q[:120]


def _cached_calls_closure_cms(
    project_id: str,
    root: str,
    evidence: dict[str, Any] | None = None,
) -> list[str]:
    """Neo4j CALLS closure with per-evidence memo (one query per root)."""
    pid = (project_id or "").strip()
    cm = _simple_cm(root)
    if not pid or not cm or "#" not in cm:
        return []
    cache: dict[str, list[str]] | None = None
    if isinstance(evidence, dict):
        raw_cache = evidence.setdefault("_calls_closure_cache", {})
        if isinstance(raw_cache, dict):
            cache = raw_cache
            hit = cache.get(cm)
            if isinstance(hit, list):
                return list(hit)
    try:
        from app.agent.ask_retrieval import _calls_closure_cms

        rows = [
            _simple_cm(c)
            for c in _calls_closure_cms(pid, cm)[:120]
            if c and "#" in str(c)
        ]
    except Exception:
        logger.debug("cached CALLS closure failed", exc_info=True)
        rows = []
    if cache is not None:
        cache[cm] = list(rows)
    return list(rows)


def _entry_chain_cms_set(
    evidence: dict[str, Any],
    *,
    project_id: str = "",
) -> set[str]:
    """Spine ∪ entry CALLS closure — membership set for method-first dig-in.

    One Neo4j traversal from the HTTP entry (depth 4 already covers typical
    Controller → ServiceImpl → leaf). Extra per-spine-root closures were the
    dominant T4 cost on first-turn dig-in.
    """
    out: set[str] = set(_spine_cms_for_focus(evidence))
    entry = _simple_cm(str(evidence.get("entry") or ""))
    if entry:
        out.add(entry)
    pid = project_id or str(evidence.get("project_id") or "")
    if not pid or not entry:
        return out
    for c in _cached_calls_closure_cms(pid, entry, evidence):
        if c and "#" in c:
            out.add(c)
    return out


def _try_method_first_on_chain_focus(
    question: str,
    evidence: dict[str, Any],
    *,
    project_id: str = "",
) -> tuple[list[str], dict[str, Any]] | None:
    """Invert dig-in: lock method by topic, then require entry-chain membership.

    1) Milvus vector similarity for ``method_topic`` (hybrid only if vector empty)
    2) Keep only hits on entry spine ∪ CALLS closure
    3) If empty → None (no false lock; caller falls back to entry→spine→leaf)
    """
    q = (question or "").strip()
    if not q or not _allows_entry_scoped_dig_in(evidence):
        return None
    entry = _simple_cm(str(evidence.get("entry") or ""))
    topic = _resolve_method_topic(q, evidence)
    if not entry or not topic:
        return None
    # Require dig-in intent / scoped ask / explicit topic — not bare lock_entry
    # (full-flow re-ask under lock must not false-lock via incidental desc overlap).
    if not (
        evidence.get("dig_in_likely")
        or evidence.get("dig_in_focus")
        or evidence.get("method_topic")
        or evidence.get("focus_topic")
        or str(evidence.get("ask_scope") or "").strip().lower()
        in {"entry_scoped_dig_in", "dig_in"}
        or _wants_dig_in_change(q)
    ):
        return None

    pid = project_id or str(evidence.get("project_id") or "")
    chain = _entry_chain_cms_set(evidence, project_id=pid)
    if len(chain) <= 1 and entry in chain:
        # No spine/closure yet — cannot verify membership; defer.
        return None

    candidates: list[str] = []
    seen: set[str] = set()
    search_rank: dict[str, int] = {}

    def _add(raw: str | None, *, rank: int | None = None) -> None:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or cm in seen:
            return
        if entry and cm == entry:
            return
        seen.add(cm)
        candidates.append(cm)
        if rank is not None and cm not in search_rank:
            search_rank[cm] = int(rank)

    used_vector = False
    # 1a) Vector similarity by method topic (not full entry phrase / hybrid LIKE).
    if pid:
        try:
            from app.agent.ask_retrieval import (
                _search_hit_cms_for_question,
                _vector_search_hit_cms_for_question,
            )

            vector_hits = _vector_search_hit_cms_for_question(pid, topic, top_k=16)
            vector_on_chain = [h for h in vector_hits if h in chain]
            if vector_on_chain:
                used_vector = True
                for i, h in enumerate(vector_on_chain):
                    _add(h, rank=i)
            else:
                # Vector empty or all off-chain → hybrid; do not mix LIKE into vector hits.
                for i, h in enumerate(
                    _search_hit_cms_for_question(pid, topic, top_k=16)
                ):
                    _add(h, rank=i)
        except Exception:
            logger.debug("method-first search failed", exc_info=True)

    # 1b) MySQL description overlap only when vector/hybrid found nothing.
    desc_hits: list[tuple[int, str]] = []
    if not candidates:
        for r in evidence.get("mysql_rows") or []:
            if not isinstance(r, dict):
                continue
            cm = _simple_cm(str(r.get("class_method") or ""))
            if not cm or "#" not in cm or (entry and cm == entry):
                continue
            summary = str(
                r.get("summary") or r.get("neo4j_desc") or r.get("title") or ""
            )
            if not summary.strip():
                continue
            sc = _token_overlap_score(f"{cm} {summary}", topic) + _cjk_overlap_score(
                summary, topic
            )
            sc += _topic_summary_specificity(summary, topic)
            sc += _method_name_topic_bonus(cm, topic)
            if sc >= 8:
                desc_hits.append((sc, cm))
                _add(cm)
        desc_hits.sort(key=lambda x: (-x[0], x[1]))

    if not candidates:
        return None

    # 2) Membership: method hit ∩ entry chain. Off-chain hits must not lock.
    on_chain = [cm for cm in candidates if cm in chain]
    evidence["method_first_candidates"] = list(candidates)[:12]
    evidence["method_first_on_chain"] = list(on_chain)[:8]
    if not on_chain:
        evidence["method_first_miss"] = True
        return None

    # Pull sibling summaries before grouping: spine enrich often only has the
    # umbrella parent row; topic leaves on CALLS need MySQL desc to multi-lock.
    if pid:
        peer_cms = _topic_peer_cms_for_enrich(
            chain, list(on_chain), limit=32
        )
        try:
            _enrich_focus_rows(pid, evidence, peer_cms)
        except Exception:
            logger.debug("method-first peer enrich failed", exc_info=True)

    collected = _collect_topic_cms_on_chain(
        topic,
        chain,
        evidence,
        entry=entry,
        search_rank=search_rank,
        question=q,
    )
    if collected:
        evidence["method_first_on_chain"] = list(collected)[:8]
        decision: dict[str, Any] = {
            "dig_in": True,
            "focus_method": collected[0],
            "topic": topic,
            "reason": (
                "method_first_on_chain; chain_topic"
                if len(collected) > 1
                else "method_first_on_chain"
            ),
            "source": "method_first_on_chain",
        }
        if used_vector:
            decision["search"] = "vector"
            evidence["method_first_backend"] = "vector"
        return collected, decision

    spine_set = set(_spine_cms_for_focus(evidence))
    # Prefer top search∩chain leaf over spine umbrellas whose summary only
    # incidentally mentions the topic (production dig-in leaf often lacks mysql row).
    focus_cm = ""
    if search_rank:
        off_spine_leaves = [
            cm
            for cm in on_chain
            if cm in search_rank
            and cm not in spine_set
            and _is_service_or_client_leaf(cm, entry=entry)
        ]
        if off_spine_leaves:
            off_spine_leaves.sort(key=lambda c: (search_rank.get(c, 99), c))
            focus_cm = off_spine_leaves[0]
        if not focus_cm:
            promoted = _best_search_closure_topic_leaf(
                root=entry,
                closure_set=chain,
                search_rank=search_rank,
                entry=entry,
                spine_set=spine_set,
            )
            if promoted and promoted in on_chain:
                focus_cm = promoted
    if not focus_cm:
        pick = _pick_best_topic_leaf(
            on_chain,
            evidence,
            topic,
            entry=entry,
            spine_set=spine_set,
            closure_set=chain,
            search_rank=search_rank or None,
        )
        if not pick:
            leafish = [
                cm
                for cm in on_chain
                if _is_service_or_client_leaf(cm, entry=entry)
            ]
            if search_rank:
                leafish.sort(key=lambda c: search_rank.get(c, 99))
            pick = leafish[:1] if leafish else on_chain[:1]
        focus_cm = pick[0] if pick else ""
    if not focus_cm:
        return None
    decision: dict[str, Any] = {
        "dig_in": True,
        "focus_method": focus_cm,
        "topic": topic,
        "reason": "method_first_on_chain",
        "source": "method_first_on_chain",
    }
    if used_vector:
        decision["search"] = "vector"
        evidence["method_first_backend"] = "vector"
    return [focus_cm], decision


def _should_compress_lock_dig_in_goal(
    question: str, *, lock_entry: bool, dig_in_likely: bool
) -> bool:
    """Skip heavy goal LLM when locked dig-in can force explain_flow."""
    if not lock_entry:
        return False
    if not (dig_in_likely or _wants_dig_in_change(question)):
        return False
    if wants_field_value_ask_safe(question):
        return False
    hg = heuristic_goal(question)
    if str(hg.get("goal") or "") in {
        "show_code",
        "compose_code",
        "api_io",
        "diagnose",
        "change_plan",
        "add_api",
        "compound",
        "inventory",
        "field_value",
    }:
        return False
    return True


def _lock_dig_in_goal(*, entry_hint: str | None = None) -> dict[str, Any]:
    """Compressed explain_flow contract for lock_entry dig-in (no goal LLM)."""
    return {
        "goal": "explain_flow",
        "also": [],
        "must": [
            {"id": "entry", "desc": "沿用锁定主入口"},
            {"id": "spine", "desc": "复用会话主链符号"},
            {"id": "summarize", "desc": "描述命中后聚焦叶子方法作答"},
        ],
        "field_name": None,
        "field_label_zh": None,
        "tables": [],
        "inventory_kind": None,
        "view_kind": None,
        "lookup_key": None,
        "lookup_table": None,
        "lookup_steps": [],
        "file_globs": [],
        "deliverable": "accurate_evidence_backed_answer",
        "source": "heuristic+dig_in_lock",
        "dig_in_focus": True,
        "ask_scope": "entry_scoped_dig_in",
        "entry_hint": (entry_hint or "").strip() or None,
    }


def _seed_reuse_locked_spine(evidence: dict[str, Any]) -> bool:
    """Reuse session entry / flow_symbols / answer_steps; skip T1–T3 when possible.

    Sets ``reuse_shared_spine`` when entry + (calls or answer_steps) are seeded.
    Session-drift guard: when the locked entry is an unrelated method but the
    current dig-in names a view_hint, the view_hint is the real display target
    (desktop-chat-e47d3b34dba4: Product feign decorator was wrongly reused as
    the locked entry for a synchroMemberPrice dig-in).
    """
    if not evidence.get("lock_entry"):
        return False
    from app.agent.explore_agent import _class_method_from_hint

    hint = _class_method_from_hint(
        str(evidence.get("entry_hint") or "")
    ) or _simple_cm(str(evidence.get("entry") or ""))
    if not hint or "#" not in hint:
        return False

    flow = [
        _simple_cm(str(x))
        for x in (evidence.get("flow_symbols") or [])
        if x and "#" in str(x)
    ]
    steps = [s for s in (evidence.get("answer_steps") or []) if isinstance(s, dict)]
    if not flow and not steps and not (evidence.get("calls") or []):
        return False

    # Session-drift guard: prefer view_hint as the display target when present
    # and different from the stale locked entry.
    view_hint_cm = _class_method_from_hint(
        str(evidence.get("view_hint") or "")
    ) or _simple_cm(str(evidence.get("view_hint") or ""))
    if view_hint_cm and "#" in view_hint_cm and view_hint_cm != _simple_cm(hint):
        hint = view_hint_cm
        evidence["entry"] = hint

    evidence["entry"] = hint
    entry_s = _simple_cm(hint)
    if flow and not (evidence.get("calls") or []):
        # Ensure the view target itself is on the spine candidates (it may be
        # the entry now, but flow_symbols order should still surface it).
        calls = [c for c in flow if c and c != entry_s]
        if view_hint_cm and view_hint_cm not in calls:
            calls.insert(0, view_hint_cm)
        evidence["calls"] = calls
        evidence["calls_count"] = len(calls)
        if not evidence.get("spine_source"):
            evidence["spine_source"] = "session_reuse"
    elif evidence.get("calls") and not evidence.get("calls_count"):
        evidence["calls_count"] = len(evidence.get("calls") or [])

    # Seed lightweight mysql_rows from prior answer steps when empty.
    if not (evidence.get("mysql_rows") or []) and steps:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()
        for i, step in enumerate(steps, start=1):
            cm = ""
            for key in ("class_method", "cm", "method"):
                raw = str(step.get(key) or "").strip()
                if raw and "#" in raw:
                    cm = _simple_cm(raw)
                    break
            if not cm or cm in seen:
                continue
            seen.add(cm)
            title = str(step.get("title") or "").strip()
            detail = str(step.get("detail") or "").strip()
            summary = re.sub(r"\s+", " ", f"{title} {detail}").strip()
            rows.append(
                {
                    "order": i,
                    "class_method": cm,
                    "summary": summary[:400],
                    "title": title[:120],
                    "rel": "SESSION",
                }
            )
        if rows:
            evidence["mysql_rows"] = rows

    evidence["reuse_shared_spine"] = True
    return True


def _prior_answer_snippet(history_text: str, *, limit: int = 900) -> str:
    """Best-effort prior assistant answer from PEV history_text."""
    text = (history_text or "").strip()
    if not text:
        return ""
    last = ""
    for line in text.splitlines():
        low = line.lower()
        if low.startswith("assistant:") or line.startswith("助手:"):
            last = line.split(":", 1)[-1].strip()
    if last:
        return truncate(last, limit)
    return truncate(text, limit)


def _spine_candidates_for_llm(evidence: dict[str, Any]) -> list[dict[str, str]]:
    """Spine + limited one-hop callees for dig-in LLM (leaf methods visible)."""
    out: list[dict[str, str]] = []
    seen: set[str] = set()

    def _add(cm: str) -> None:
        cm_s = _simple_cm(cm)
        if not cm_s or "#" not in cm_s or cm_s in seen:
            return
        seen.add(cm_s)
        row = _row_for_cm(evidence, cm_s) or {}
        summary = str(
            row.get("summary") or row.get("neo4j_desc") or row.get("title") or ""
        ).strip()
        summary = re.sub(r"\s+", " ", summary)[:220]
        out.append({"class_method": cm_s, "summary": summary})

    spine = _spine_cms_for_focus(evidence)
    for cm in spine:
        _add(cm)
    # One-hop leaves under locked entry (cached entry closure only).
    pid = str(evidence.get("project_id") or "").strip()
    if pid and len(out) < 24:
        try:
            for c in _entry_chain_cms_set(evidence, project_id=pid):
                _add(c)
                if len(out) >= 24:
                    break
        except Exception:
            logger.debug("dig-in one-hop candidates failed", exc_info=True)
    return out[:24]


def _fallback_dig_in_focus_cms(
    question: str,
    evidence: dict[str, Any],
    *,
    project_id: str = "",
) -> list[str]:
    """Pick dig-in focus via closure + search when dig-in LLM times out/fails.

    Language-agnostic: token overlap on Class#method + summaries; prefer
    ServiceImpl/Client leaves inside the locked entry's call closure.
    Search-rank boost covers CJK questions that do not lexically match Latin names.
    """
    q = (question or "").strip()
    if not q:
        return []
    pid = project_id or str(evidence.get("project_id") or "")
    entry = _simple_cm(str(evidence.get("entry") or ""))
    spine = _spine_cms_for_focus(evidence)
    spine_set = set(spine)

    candidates: list[str] = []
    seen: set[str] = set()
    closure_set: set[str] = set()
    chain_set: set[str] = set()
    search_rank: dict[str, int] = {}

    def _add(raw: str | None) -> None:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or cm in seen:
            return
        seen.add(cm)
        candidates.append(cm)

    for cm in spine:
        _add(cm)
    for r in evidence.get("mysql_rows") or []:
        if isinstance(r, dict):
            _add(str(r.get("class_method") or ""))

    if pid:
        try:
            from app.agent.ask_retrieval import _search_hit_cms_for_question

            chain = _entry_chain_cms_set(evidence, project_id=pid)
            chain_set.update(chain)
            # Scoring bonuses use Neo4j callees only — spine parents are on-chain
            # members but must not get the search∩closure jackpot.
            if entry:
                closure_set.update(_cached_calls_closure_cms(pid, entry, evidence))
            for cm in chain:
                _add(cm)
            for i, h in enumerate(_search_hit_cms_for_question(pid, q, top_k=16)):
                cm = _simple_cm(h)
                if not cm:
                    continue
                if chain and cm not in chain:
                    continue
                _add(cm)
                if cm not in search_rank:
                    search_rank[cm] = i
        except Exception:
            logger.debug("dig-in retrieval fallback pool failed", exc_info=True)

    scored: list[tuple[int, str]] = []
    for cm in candidates:
        if entry and cm == entry:
            continue
        row = _row_for_cm(evidence, cm) or {}
        summary = str(row.get("summary") or row.get("neo4j_desc") or "")
        sc = _token_overlap_score(f"{cm} {summary}", q) + _cjk_overlap_score(
            summary, q
        )
        spec = _topic_summary_specificity(summary, q)
        sc += spec
        sc += _method_name_topic_bonus(cm, q)
        if cm in closure_set:
            sc += 5
        if cm in search_rank:
            # Top hybrid-search hit wins even when question tokens ≠ method Latin id.
            sc += max(6, 22 - int(search_rank[cm]))
            if cm in closure_set:
                sc += 10
                # Retrieval leaf may lack mysql summary enrichment yet — still prefer
                # search∩closure over spine parents that only incidentally mention topic.
                if not summary.strip():
                    sc += 50
                sc += max(0, 30 - int(search_rank[cm]) * 3)
        sc += _leaf_focus_bonus(
            cm,
            entry=entry,
            spine_set=spine_set,
            closure_set=closure_set,
            summary_spec=spec,
        )
        if cm not in spine_set and sc > 0:
            sc += 2
        if sc > 0:
            scored.append((sc, cm))
    scored.sort(key=lambda x: (-x[0], x[1]))
    if not scored:
        return []
    # Search hits off the entry chain must not lock a random Client leaf.
    if search_rank and chain_set:
        on_chain_search = [
            cm for _sc, cm in scored if cm in search_rank and cm in chain_set
        ]
        if on_chain_search:
            return on_chain_search[:1]
        return []
    # Single most specific leaf (search∩closure preferred via score).
    strong = [cm for sc, cm in scored if sc >= 6]
    if strong:
        return strong[:1]
    return [scored[0][1]]


def _focus_zero_overlap(cm: str, probe: str, evidence: dict[str, Any]) -> bool:
    """True when a fallback-picked focus method is *provably* unrelated to the ask.

    Guards retrieval-fallback focus (desktop-chat-1b5c244b4a5e: a timeout picked
    ApiOrgServiceClient#queryProductInfos for a coupon-usage ask). Such a method
    must not be presented as a confident answer — degrade to an honest note.

    Zero lexical overlap is only damning when we had a Chinese summary that still
    does not match (we can see what the method does), or when the focus is an
    external *Client facade (cross-domain drift). A summary-less ServiceImpl leaf
    with zero lexical overlap is NOT demoted — its body is read before rendering.
    """
    row = _row_for_cm(evidence, cm) or {}
    summary = str(row.get("summary") or row.get("neo4j_desc") or "")
    cls = str(cm).split("#", 1)[0].split(".")[-1] if "#" in str(cm) else str(cm)
    is_client = bool(re.search(r"(?i)(Client|Facade|Api|Feign|Proxy)$", cls))
    blob = f"{cm} {summary}"
    overlap = _token_overlap_score(blob, probe) + _cjk_overlap_score(summary, probe)
    if overlap > 0:
        return False
    if summary.strip():
        return True
    if is_client:
        return True
    return False


def _token_overlap_score(blob: str, needle: str) -> int:
    """Language-agnostic overlap: Class#method / identifiers / contiguous runs."""
    b = (blob or "").lower()
    n = (needle or "").strip()
    if not b or not n:
        return 0
    nl = n.lower()
    score = 0
    if "#" in n:
        cm = _simple_cm(n)
        if cm and cm.lower() in b:
            return 100
        meth = cm.split("#", 1)[-1].lower() if cm and "#" in cm else ""
        if meth and meth in b:
            score += 40
    if nl in b:
        score += max(8, min(30, len(n)))
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,40}|[\u4e00-\u9fff]{2,12}", n):
        tl = tok.lower()
        if tl in b:
            score += max(2, len(tok) // 2)
    return score


def _cjk_runs(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]{2,16}", text or "")


def _cjk_ngrams(text: str, *, n: int = 2) -> set[str]:
    """Character n-grams from CJK runs (generic; no keyword wordlists)."""
    out: set[str] = set()
    for run in _cjk_runs(text):
        if len(run) < n:
            out.add(run)
            continue
        for i in range(len(run) - n + 1):
            out.add(run[i : i + n])
        if len(run) >= 3:
            for i in range(len(run) - 2):
                out.add(run[i : i + 3])
    return out


def _cjk_overlap_score(blob: str, needle: str) -> int:
    """Generic CJK run / n-gram overlap between needle and blob text."""
    n = (needle or "").strip()
    b = blob or ""
    if not n or not b:
        return 0
    score = 0
    for run in _cjk_runs(n):
        if run in b:
            score += max(6, min(24, len(run) * 2))
    ngrams = _cjk_ngrams(n)
    if not ngrams:
        return score
    b_grams = _cjk_ngrams(b)
    hit = len(ngrams & b_grams)
    if hit:
        score += min(20, hit * 2)
    return score


def _topic_phrase_in_text(text: str, topic: str, *, min_run: int = 4) -> bool:
    """True when a substantial CJK/Latin phrase from topic appears in text.

    2-char n-grams (接口) are too weak to lock a leaf; require a 4+ char
    run or an explicit Latin identifier from the topic.
    """
    blob = text or ""
    t = (topic or "").strip()
    if not blob or not t:
        return False
    if t in blob:
        return True
    for run in _cjk_runs(t):
        if len(run) >= min_run and run in blob:
            return True
        if len(run) >= min_run:
            for i in range(len(run) - min_run + 1):
                if run[i : i + min_run] in blob:
                    return True
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{3,40}", t):
        if tok.lower() in blob.lower():
            return True
    return False


def _topic_summary_specificity(summary: str, topic: str) -> int:
    """Higher when summary is tightly about topic (not an incidental mention)."""
    s = (summary or "").strip()
    t = (topic or "").strip()
    if not s or not t:
        return 0
    overlap = _token_overlap_score(s, t) + _cjk_overlap_score(s, t)
    if overlap <= 0:
        return 0
    # Dominance: topic run length vs summary length (umbrella parents score low).
    dom = 0
    for run in _cjk_runs(t):
        if run in s:
            dom = max(dom, (len(run) * 100) // max(len(s), 1))
    if not dom:
        dom = min(100, (overlap * 10) // max(len(s), 1))
    # Cap absolute overlap so long parents cannot swamp tighter leaves.
    return min(20, overlap) + dom


def _method_name_topic_bonus(cm: str, topic: str) -> int:
    """Bonus when Latin tokens from topic appear in the method name."""
    if not cm or "#" not in cm or not topic:
        return 0
    meth = cm.split("#", 1)[-1].split("(", 1)[0].lower()
    score = 0
    for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,40}", topic):
        tl = tok.lower()
        if tl == meth:
            score += 50
        elif tl in meth or meth in tl:
            score += 20
    return score


def _leaf_focus_bonus(
    cm: str,
    *,
    entry: str,
    spine_set: set[str],
    closure_set: set[str] | None = None,
    summary_spec: int = 0,
) -> int:
    """Prefer deeper ServiceImpl/Client leaves over umbrella spine parents."""
    score = 0
    if "Controller#" in cm:
        score -= 12
    if "ServiceImpl#" in cm or re.search(r"(?i)Client#", cm):
        score += 6
    if entry and cm == entry:
        score -= 10
    if closure_set is not None and cm in closure_set:
        score += 4
    # Spine parent with weak/incidental topic hit → penalize.
    # Dominance scale: tight leaf ~100+, incidental umbrella often <70.
    if cm in spine_set and summary_spec > 0 and summary_spec < 70:
        score -= 24
    elif cm not in spine_set and summary_spec >= 70:
        score += 8
    return score


def _pick_best_topic_leaf(
    candidates: list[str],
    evidence: dict[str, Any],
    query: str,
    *,
    entry: str = "",
    spine_set: set[str] | None = None,
    closure_set: set[str] | None = None,
    search_rank: dict[str, int] | None = None,
    hint: str = "",
) -> list[str]:
    """Score candidates; return the single most specific leaf (or empty)."""
    spine_set = spine_set or set()
    scored: list[tuple[int, str]] = []
    for cm in candidates:
        if not cm or "#" not in cm:
            continue
        if entry and cm == entry:
            continue
        row = _row_for_cm(evidence, cm) or {}
        summary = str(row.get("summary") or row.get("neo4j_desc") or "")
        blob = f"{cm} {summary}"
        sc = _token_overlap_score(blob, query) + _cjk_overlap_score(summary, query)
        sc += _method_name_topic_bonus(cm, query)
        spec = _topic_summary_specificity(summary, query)
        sc += spec
        if hint and "#" not in hint:
            meth = cm.split("#", 1)[-1].split("(", 1)[0]
            if meth.lower() == hint.lower():
                sc += 50
            elif hint.lower() in meth.lower() or meth.lower() in hint.lower():
                sc += 20
        # Hybrid search∩closure: leaves without mysql summary can still outrank
        # umbrella parents whose summary only incidentally mentions the topic.
        if search_rank is not None and cm in search_rank:
            sc += max(6, 22 - int(search_rank[cm]))
            if closure_set is not None and cm in closure_set:
                sc += 10
                if not summary.strip():
                    sc += 50
                sc += max(0, 30 - int(search_rank[cm]) * 3)
        sc += _leaf_focus_bonus(
            cm,
            entry=entry,
            spine_set=spine_set,
            closure_set=closure_set,
            summary_spec=spec,
        )
        if sc > 0:
            scored.append((sc, cm))
    if not scored:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [scored[0][1]]


def _is_service_or_client_leaf(cm: str, *, entry: str = "") -> bool:
    """True for ServiceImpl/Client methods that are not the locked Controller entry."""
    c = _simple_cm(cm)
    if not c or "#" not in c:
        return False
    if entry and c == entry:
        return False
    if "Controller#" in c:
        return False
    return bool("ServiceImpl#" in c or re.search(r"(?i)Client#", c))


def _topic_peer_cms_for_enrich(
    chain: set[str] | list[str],
    search_cms: list[str],
    *,
    limit: int = 32,
) -> list[str]:
    """Order same-class CALLS peers for MySQL enrich before topic grouping.

    Prefer vector/hybrid hits, then same-class methods that share a long Latin
    method-name prefix with a hit (generic identifier family, not a product map).
    """
    seeds: list[str] = []
    seen: set[str] = set()
    for raw in search_cms:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or cm in seen:
            continue
        seen.add(cm)
        seeds.append(cm)
    seed_parts = [(c.split("#", 1)[0], c.split("#", 1)[1].lower()) for c in seeds]
    classes = {cls for cls, _ in seed_parts}
    prefix_peers: list[str] = []
    other_peers: list[str] = []
    for raw in chain:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or cm in seen:
            continue
        cls, meth = cm.split("#", 1)
        if cls not in classes:
            continue
        ml = meth.lower()
        share = False
        for _scls, sm in seed_parts:
            if cls != _scls:
                continue
            n = 0
            for a, b in zip(ml, sm):
                if a != b:
                    break
                n += 1
            if n >= 8:
                share = True
                break
        seen.add(cm)
        if share:
            prefix_peers.append(cm)
        else:
            other_peers.append(cm)
    return list(dict.fromkeys([*seeds, *prefix_peers, *other_peers]))[: max(8, limit)]


def _incidental_topic_umbrella(summary: str, topic: str) -> bool:
    """True when summary only casually mentions topic inside a parent flow.

    Structural parent markers only (no product nouns). Real topic leaves that
    themselves describe multi-step checks must not match.
    """
    s = (summary or "").strip()
    t = (topic or "").strip()
    if not s or not t or not _topic_phrase_in_text(s, t):
        return False
    return bool(re.search(r"主流程|主处理|编排|核心流程", s))


def _collect_topic_cms_on_chain(
    topic: str,
    chain: set[str] | list[str],
    evidence: dict[str, Any],
    *,
    entry: str = "",
    search_rank: dict[str, int] | None = None,
    question: str = "",
) -> list[str]:
    """On-chain methods whose summaries/search rank match the dig-in topic.

    Groups peer leaves on the CALLS closure. Does not group by same file or
    method-name prefix, and does not expand a CJK-only leaf to umbrellas or
    outbound Clients. Returns empty when fewer than two hits so the caller
    keeps a single locked leaf.
    """
    topic_s = (topic or "").strip()
    if not topic_s or not chain:
        return []
    entry_cm = _simple_cm(entry)
    # FOLLOWS/CALLS spine only — do not treat dig-in mysql enrich rows as spine
    # umbrellas (that would drop the very leaves we just loaded).
    spine_set = {
        _simple_cm(str(c))
        for c in (evidence.get("calls") or [])
        if c and "#" in str(c)
    }
    if entry_cm:
        spine_set.add(entry_cm)
    needles = _topic_needles_from_question(question or topic_s, [], topic=topic_s)
    search_rank = search_rank or {}
    scored: list[tuple[int, str]] = []
    for raw in chain:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or (entry_cm and cm == entry_cm):
            continue
        row = _row_for_cm(evidence, cm) or {}
        summary = str(
            row.get("summary") or row.get("neo4j_desc") or row.get("title") or ""
        )
        name_sc = _score_topic_hit(cm, "", needles)
        spec = _topic_summary_specificity(summary, topic_s)
        phrase = _topic_phrase_in_text(summary, topic_s)
        ranked = cm in search_rank
        rank = int(search_rank.get(cm, 99)) if ranked else 99
        # Parent flow that only lists the topic must never lock (even as rank-0
        # seed from a later post-pass that mistakenly ranked the umbrella first).
        if _incidental_topic_umbrella(summary, topic_s):
            continue
        if cm in spine_set and phrase and spec < 50 and not ranked:
            continue
        # Any search∩chain hit may seed inclusion when leaf MySQL row is still
        # missing; umbrella parents are already excluded above.
        seed = ranked
        if not (phrase or name_sc >= 6 or seed):
            continue
        sc = (
            name_sc
            + _score_topic_hit(cm, summary, needles)
            + spec
            + _method_name_topic_bonus(cm, topic_s)
        )
        if cm in search_rank:
            sc += max(10, 28 - int(search_rank[cm]))
        if cm in spine_set and spec < 50:
            sc -= 30
        scored.append((sc, cm))
    if len(scored) < 2:
        return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [cm for _, cm in scored[:8]]


def _best_search_closure_topic_leaf(
    *,
    root: str,
    closure_set: set[str],
    search_rank: dict[str, int],
    entry: str = "",
    spine_set: set[str] | None = None,
) -> str:
    """Promote top search∩closure ServiceImpl/Client leaf over an umbrella root."""
    spine_set = spine_set or set()
    scored: list[tuple[int, int, str]] = []
    for cm, rank in search_rank.items():
        if cm not in closure_set:
            continue
        if cm == root or not _is_service_or_client_leaf(cm, entry=entry):
            continue
        # Prefer non-spine / deeper leaves, then better search rank.
        scored.append((0 if cm not in spine_set else 1, int(rank), cm))
    if not scored:
        return ""
    scored.sort()
    return scored[0][2]


def _anchor_leaf_from_answer_steps(
    question: str,
    evidence: dict[str, Any],
    *,
    topic: str = "",
    project_id: str = "",
) -> list[str]:
    """CJK-overlap prior answer step → Class#method → best topic leaf in closure.

    Uses generic character/n-gram overlap only (no Chinese keyword gates).
    Enrichs closure via search∩closure (+ optional MySQL summaries) before scoring
    so leaves without a pre-seeded summary can still beat umbrella parents.
    """
    steps = list(evidence.get("answer_steps") or [])
    q = (question or "").strip()
    topic_s = (topic or "").strip()
    query = f"{topic_s} {q}".strip() or q
    if not steps or not query:
        return []
    if not (_cjk_runs(q) or _cjk_runs(topic_s)):
        return []

    best_step: dict[str, Any] | None = None
    best_sc = 0
    for step in steps:
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or "")
        detail = str(step.get("detail") or "")
        blob = f"{title} {detail}".strip()
        if not blob:
            continue
        sc = _cjk_overlap_score(blob, query) + _token_overlap_score(blob, query)
        if sc > best_sc:
            best_sc = sc
            best_step = step
    if not best_step or best_sc < 6:
        return []

    title = str(best_step.get("title") or "")
    detail = str(best_step.get("detail") or "")
    step_blob = f"{title} {detail}".strip()
    entry = _simple_cm(str(evidence.get("entry") or ""))
    spine = _spine_cms_for_focus(evidence)
    spine_set = set(spine)

    # Resolve step → Class#method: explicit field, then summary/title match.
    root = ""
    for key in ("class_method", "cm", "method"):
        raw = str(best_step.get(key) or "").strip()
        if raw and "#" in raw:
            root = _simple_cm(raw)
            break
    if not root:
        # Prefer method whose summary/name aligns with the step text.
        step_hits = _pick_best_topic_leaf(
            spine
            + [
                _simple_cm(str(r.get("class_method") or ""))
                for r in (evidence.get("mysql_rows") or [])
                if isinstance(r, dict)
            ],
            evidence,
            step_blob,
            entry=entry,
            spine_set=spine_set,
        )
        root = step_hits[0] if step_hits else ""
    if not root:
        return []

    pid = project_id or str(evidence.get("project_id") or "")
    closure: list[str] = [root]
    closure_set: set[str] = {root}
    root_cls = root.split("#", 1)[0].split(".")[-1].lower() if "#" in root else ""
    search_rank: dict[str, int] = {}

    # Same-class spine / mysql siblings are candidate leaves even without graph.
    for cm in spine + [
        _simple_cm(str(r.get("class_method") or ""))
        for r in (evidence.get("mysql_rows") or [])
        if isinstance(r, dict)
    ]:
        if not cm or cm in closure_set or "#" not in cm:
            continue
        cls = cm.split("#", 1)[0].split(".")[-1].lower()
        if root_cls and cls == root_cls:
            closure_set.add(cm)
            closure.append(cm)

    if pid:
        try:
            from app.agent.ask_retrieval import _calls_closure_cms

            for c in _calls_closure_cms(pid, root)[:80]:
                cm = _simple_cm(c)
                if cm and "#" in cm and cm not in closure_set:
                    closure_set.add(cm)
                    closure.append(cm)
        except Exception:
            logger.debug("answer-step anchor closure failed", exc_info=True)
        try:
            from app.agent.ask_retrieval import _search_hit_cms_for_question

            for i, h in enumerate(_search_hit_cms_for_question(pid, query, top_k=16)):
                cm = _simple_cm(h)
                if not cm or "#" not in cm:
                    continue
                if cm not in search_rank:
                    search_rank[cm] = i
                # search∩(same-class or already-closed) expands scoring pool.
                if cm in closure_set:
                    continue
                cls = cm.split("#", 1)[0].split(".")[-1].lower()
                if root_cls and cls == root_cls:
                    closure_set.add(cm)
                    closure.append(cm)
                elif cm in spine_set:
                    closure_set.add(cm)
                    closure.append(cm)
        except Exception:
            logger.debug("answer-step anchor search failed", exc_info=True)
    # Same-class private callees from root source (when available).
    try:
        from app.agent.explore_source import _fetch_method_source_body
        from app.agent.explore_tools import extract_same_class_calls

        if pid:
            code, _path = _fetch_method_source_body(pid, root, max_chars=8000)
            if code:
                cls = root.split("#", 1)[0].split(".")[-1]
                meth = root.split("#", 1)[1]
                for name in extract_same_class_calls(code, current_method=meth)[:30]:
                    cm = _simple_cm(f"{cls}#{name}")
                    if cm not in closure_set:
                        closure_set.add(cm)
                        closure.append(cm)
    except Exception:
        logger.debug("answer-step same_class_calls failed", exc_info=True)
    # Pull leaf summaries into evidence before scoring (production spine often
    # only has the umbrella parent row; leaf summary arrives too late otherwise).
    if pid:
        try:
            _enrich_focus_rows(pid, evidence, closure)
        except Exception:
            logger.debug("answer-step anchor enrich failed", exc_info=True)

    leaf = _pick_best_topic_leaf(
        closure,
        evidence,
        query,
        entry=entry,
        spine_set=spine_set,
        closure_set=closure_set,
        search_rank=search_rank or None,
    )
    pick = leaf[0] if leaf else root
    # Hard rule: step-root / spine umbrella must not win when search∩closure has
    # a more topic-specific ServiceImpl/Client leaf.
    if pick == root:
        promoted = _best_search_closure_topic_leaf(
            root=root,
            closure_set=closure_set,
            search_rank=search_rank,
            entry=entry,
            spine_set=spine_set,
        )
        if promoted:
            return [promoted]
    if leaf and leaf[0] != root:
        return leaf
    if leaf:
        # Leaf scored but equals root — try siblings with search boost.
        sibling_leaf = _pick_best_topic_leaf(
            list(closure_set | spine_set),
            evidence,
            query,
            entry=entry,
            spine_set=spine_set,
            closure_set=closure_set,
            search_rank=search_rank or None,
        )
        if sibling_leaf and sibling_leaf[0] != root:
            if sibling_leaf[0] == pick or _is_service_or_client_leaf(
                sibling_leaf[0], entry=entry
            ):
                return sibling_leaf
        return leaf
    # Topic matches a sibling leaf better than the step root → prefer leaf.
    sibling_leaf = _pick_best_topic_leaf(
        list(closure_set | spine_set),
        evidence,
        query,
        entry=entry,
        spine_set=spine_set,
        closure_set=closure_set,
        search_rank=search_rank or None,
    )
    if sibling_leaf and sibling_leaf[0] != root:
        return sibling_leaf
    return [root]


def _topic_needles_from_question(
    question: str,
    focus_cms: list[str],
    *,
    topic: str = "",
) -> list[str]:
    """Needles for related-callee scoring. Latin + LLM topic first; CJK n-grams weak."""
    q = question or ""
    skip_meth = {
        cm.split("#", 1)[1].split("(", 1)[0].lower()
        for cm in focus_cms
        if "#" in cm
    }
    primary: list[str] = []
    weak: list[str] = []
    if topic:
        for m in re.finditer(
            r"[A-Za-z][A-Za-z0-9_]{2,40}|[\u4e00-\u9fff]{2,12}", topic
        ):
            tok = m.group(0)
            if tok.lower() not in skip_meth:
                primary.append(tok)
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9_]{2,40})\b", q):
        tok = m.group(1)
        if tok.lower() in skip_meth:
            continue
        primary.append(tok)
    stripped = re.sub(
        r"方法中|方法里|是什么|具体做了什么|做了什么|做什么的|做什么|什么样|"
        r"怎么走|主要|具体|这一块|这块|接口中|流程中",
        " ",
        q,
    )
    for m in re.finditer(r"[\u4e00-\u9fff]{2,12}", stripped):
        tok = m.group(0)
        if tok.lower() in skip_meth:
            continue
        primary.append(tok)
        for nlen in (2, 3, 4):
            if len(tok) < nlen:
                continue
            for i in range(len(tok) - nlen + 1):
                primary.append(tok[i : i + nlen])
    for m in re.finditer(r"[\u4e00-\u9fff]{2,8}", q):
        tok = m.group(0)
        weak.append(tok)
    merged = list(dict.fromkeys(primary + weak))
    return sorted(merged, key=len, reverse=True)[:16]


def _score_topic_hit(cm: str, summary: str, needles: list[str]) -> int:
    blob = f"{cm} {summary}".lower()
    score = 0
    for n in needles:
        nl = n.lower()
        if nl and nl in blob:
            score += max(2, len(n) // 2)
    return score


def _topic_leaf_score(
    cm: str,
    evidence: dict[str, Any],
    query: str,
    *,
    entry: str = "",
    spine_set: set[str] | None = None,
    related_set: set[str] | None = None,
) -> int:
    """Score one cms for dig-in topic leaf quality (summary/overlap/depth)."""
    c = _simple_cm(cm)
    if not c or "#" not in c:
        return -10_000
    if entry and c == entry:
        return -5_000
    spine_set = spine_set or set()
    related_set = related_set or set()
    row = _row_for_cm(evidence, c) or {}
    summary = str(row.get("summary") or row.get("neo4j_desc") or "")
    blob = f"{c} {summary}"
    sc = _token_overlap_score(blob, query) + _cjk_overlap_score(summary, query)
    sc += _method_name_topic_bonus(c, query)
    spec = _topic_summary_specificity(summary, query)
    sc += spec
    sc += _leaf_focus_bonus(
        c,
        entry=entry,
        spine_set=spine_set,
        closure_set=related_set or None,
        summary_spec=spec,
    )
    # Related callees are deeper than the current focus umbrella.
    if c in related_set:
        sc += 8
        if _is_service_or_client_leaf(c, entry=entry):
            sc += 6
    return sc


def _promote_related_leaf_to_focus(
    focus_cms: list[str],
    related: list[str],
    evidence: dict[str, Any],
    *,
    question: str = "",
    topic: str = "",
) -> tuple[list[str], list[str], str]:
    """Post-pass: if related already has a stronger topic leaf, promote it to focus.

    Production safety net (desktop-chat-022e6fc524a1): prefer/anchor may leave an
    umbrella parent as dig-in focus while ``_related_cms_under_focus`` already
    found the true leaf (e.g. synchroMemberPrice). LLM dig_in true/false is OK,
    but must not keep the umbrella when related has a better leaf.

    Returns ``(focus, related, demoted_parent)`` — demoted_parent is empty when
    no promotion occurs.
    """
    focus = [_simple_cm(c) for c in (focus_cms or []) if c and "#" in _simple_cm(c)]
    rel = [_simple_cm(c) for c in (related or []) if c and "#" in _simple_cm(c)]
    if not focus or not rel:
        return focus[:1] if focus else [], rel, ""

    cur = focus[0]
    # Explicit Latin method name in the question that equals current focus:
    # describing that method IS the leaf dig-in — do not demote it unless a
    # related leaf has a strictly stronger method-name match to the topic.
    q = (question or "").strip()
    topic_s = (topic or "").strip()
    query = f"{topic_s} {q}".strip() or q
    if not query:
        return [cur], rel, ""

    cur_meth = cur.split("#", 1)[-1].split("(", 1)[0]
    named_focus = bool(
        cur_meth
        and len(cur_meth) >= 3
        and re.search(
            rf"(?i)(^|[^A-Za-z0-9_]){re.escape(cur_meth)}([^A-Za-z0-9_]|$|方法)",
            q,
        )
    )

    entry = _simple_cm(str(evidence.get("entry") or ""))
    spine_set = set(_spine_cms_for_focus(evidence))
    related_set = set(rel)
    cur_on_spine = cur in spine_set

    def _pick_best_related() -> str:
        best_cm = ""
        best_sc = -10_000
        for cm in rel:
            if not _is_service_or_client_leaf(cm, entry=entry):
                continue
            sc = _topic_leaf_score(
                cm,
                evidence,
                query,
                entry=entry,
                spine_set=spine_set,
                related_set=related_set,
            )
            if sc > best_sc:
                best_sc = sc
                best_cm = cm
        return best_cm or (rel[0] if rel else "")

    # Already a non-spine leaf focus: only promote when related has a clearly
    # stronger method-name match (do not steal focus to a deeper client callee).
    if not cur_on_spine:
        leaf = _pick_best_related()
        if (
            leaf
            and leaf != cur
            and _method_name_topic_bonus(leaf, query)
            > _method_name_topic_bonus(cur, query)
        ):
            if named_focus and _method_name_topic_bonus(leaf, query) <= (
                _method_name_topic_bonus(cur, query)
            ):
                return [cur], rel, ""
            return [leaf], [c for c in rel if c != leaf], cur
        return [cur], rel, ""

    # Spine umbrella focus: related is already topic-ranked under this parent.
    # Never keep the umbrella as sole dig-in focus when a ServiceImpl/Client leaf
    # sits in related and the focus method-name does not uniquely own the topic
    # (CJK-only asks included — parent summary incidental mention must not win).
    leaf = _pick_best_related()
    if not leaf or leaf == cur:
        return [cur], rel, ""

    cur_name = _method_name_topic_bonus(cur, query)
    leaf_name = _method_name_topic_bonus(leaf, query)
    cur_sc = _topic_leaf_score(
        cur, evidence, query, entry=entry, spine_set=spine_set, related_set=related_set
    )
    leaf_sc = _topic_leaf_score(
        leaf, evidence, query, entry=entry, spine_set=spine_set, related_set=related_set
    )

    should = False
    if leaf_name > cur_name:
        should = True
    elif cur_name == 0:
        # Chinese / non-Latin topic: related already matched needles → promote.
        should = True
    elif leaf_sc > cur_sc:
        should = True

    if not should:
        return [cur], rel, ""

    if named_focus and leaf_name <= cur_name:
        return [cur], rel, ""

    return [leaf], [c for c in rel if c != leaf], cur


def _spine_cms_for_focus(evidence: dict[str, Any]) -> list[str]:
    """Ordered Class#method candidates from current spine / prior session symbols."""
    out: list[str] = []
    seen: set[str] = set()

    def _add(raw: str | None) -> None:
        cm = _simple_cm(str(raw or ""))
        if not cm or "#" not in cm or cm in seen:
            return
        seen.add(cm)
        out.append(cm)

    for raw in evidence.get("calls") or []:
        _add(str(raw))
    for r in evidence.get("mysql_rows") or []:
        if isinstance(r, dict):
            _add(str(r.get("class_method") or ""))
    for raw in evidence.get("flow_symbols") or []:
        _add(str(raw))
    entry = _simple_cm(str(evidence.get("entry") or ""))
    if entry:
        _add(entry)
    return out


def _focus_cms_from_question(
    question: str, evidence: dict[str, Any]
) -> list[str]:
    """Match method tokens in the question to spine Class#method symbols."""
    q = question or ""
    if not q.strip():
        return []
    hits: list[str] = []
    for cm in _spine_cms_for_focus(evidence):
        meth = cm.split("#", 1)[1].split("(", 1)[0].strip()
        if len(meth) < 3:
            continue
        if re.search(
            rf"(?i)(^|[^A-Za-z0-9_]){re.escape(meth)}([^A-Za-z0-9_]|$|方法)",
            q,
        ):
            hits.append(cm)
    return hits


def _row_for_cm(evidence: dict[str, Any], cm: str) -> dict[str, Any] | None:
    target = _simple_cm(cm)
    for r in evidence.get("mysql_rows") or []:
        if not isinstance(r, dict):
            continue
        if _simple_cm(str(r.get("class_method") or "")) == target:
            return r
    return None


def _order_focus_shallow_to_deep(focus_cms: list[str]) -> list[str]:
    """Order dig-in focus shallow → deep without product glossaries.

    Same-class methods where one Latin name is a proper prefix of another are
    treated as router (shallow) then specialization (deep). Otherwise keep
    input order (already topic-scored).
    """
    items = [_simple_cm(c) for c in (focus_cms or []) if c and "#" in str(c)]
    if len(items) <= 1:
        return items
    meth = {c: c.split("#", 1)[-1].lower() for c in items}

    def _key(c: str) -> tuple[int, int, int, int, str]:
        m = meth[c]
        children = sum(
            1
            for o, om in meth.items()
            if o != c and om.startswith(m) and len(om) > len(m)
        )
        parents = sum(
            1
            for o, om in meth.items()
            if o != c and m.startswith(om) and len(m) > len(om)
        )
        # Shallow first: has children / is not an extension / shorter name.
        return (0 if children else 1, parents, -children, len(m), c)

    return sorted(items, key=_key)


def _focus_depth_label(cm: str, focus_cms: list[str]) -> str:
    """Label shallow router vs deeper specialization among focus peers."""
    cms = [_simple_cm(c) for c in focus_cms if c and "#" in str(c)]
    if len(cms) <= 1:
        return "语义命中"
    cur = _simple_cm(cm)
    if not cur or "#" not in cur:
        return "语义命中"
    m = cur.split("#", 1)[-1].lower()
    peers = {c: c.split("#", 1)[-1].lower() for c in cms}
    has_child = any(
        o != cur and om.startswith(m) and len(om) > len(m) for o, om in peers.items()
    )
    has_parent = any(
        o != cur and m.startswith(om) and len(m) > len(om) for o, om in peers.items()
    )
    if has_child and not has_parent:
        return "浅·路由总览"
    if has_parent:
        return "深·细分实现"
    # Multi-hit peers without prefix relation: first = 浅, rest = 深 by order.
    if cms and cur == cms[0]:
        return "浅·语义命中"
    return "深·语义命中"


def _focus_was_in_prior_answer(
    focus_cm: str, evidence: dict[str, Any]
) -> bool:
    """Check if *focus_cm* was already listed in prior turn's *answer_steps*.

    When the user drills into a sub-step that was described inside a parent
    method, the focus often resolves back to the same parent (because its
    summary mentions the sub-step).  We should then read the parent's source
    code to extract the *specific* sub-section rather than repeating the
    same high-level summary.
    """
    cm = _simple_cm(focus_cm)
    if not cm or "#" not in cm:
        return False
    for step in (evidence.get("answer_steps") or []):
        if not isinstance(step, dict):
            continue
        for key in ("class_method", "cm", "method"):
            raw = str(step.get(key) or "").strip()
            if raw and "#" in raw and _simple_cm(raw) == cm:
                return True
    return False


async def _deepen_dig_in_with_source_code(
    project_id: str,
    focus_cm: str,
    question: str,
    *,
    topic: str = "",
) -> str:
    """Read the focus method's source code and identify the section matching
    the user's sub-topic, then return a detailed explanation of that section.

    This is called when the focus method was already shown in a prior turn
    (the user is drilling into a sub-step of an already-described method).
    """
    import asyncio

    cm = _simple_cm(focus_cm)
    if not cm or "#" not in cm or not project_id:
        return ""

    try:
        from app.agent.explore_source import _fetch_method_source_body

        code, path = _fetch_method_source_body(project_id, cm, max_chars=12000)
    except Exception:
        logger.debug("deepen dig_in: source read failed", exc_info=True)
        return ""

    if not (code or "").strip():
        return ""

    # Use LLM to find the specific code section matching the sub-topic
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        q = question or ""
        tpc = topic or ""

        prompt = (
            f"下面是一个 Java 方法的完整源码。用户问的是该方法中「{q}」"
            f"{f'（子话题：{tpc}）' if tpc and tpc != q else ''}的具体实现。\n\n"
            f"请执行以下操作：\n"
            f"1. 在源码中找到与「{q}」最相关的代码段落（可能是一个 if 块、"
            f"一段调用链、或一个完整的子步骤）；\n"
            f"2. 用中文解释这段代码具体做了什么（参数、校验、调用、返回等）；\n"
            f"3. 如果涉及远程调用或数据库操作，说明调用了什么、数据怎么流转；\n"
            f"4. 把对应的源码片段（关键 10～20 行）放在中文解释后面，"
            f"用 ```java 代码块包裹。\n\n"
            f"源码路径：{path or 'unknown'}\n\n"
            f"{'=' * 60}\n"
            f"{code}\n"
            f"{'=' * 60}\n\n"
            f"请用中文回答，格式：\n"
            f"【XX 具体实现】\n"
            f"（中文解释）\n"
            f"```java\n"
            f"（关键代码片段）\n"
            f"```"
        )

        raw = await asyncio.wait_for(
            llm.chat(
                [
                    {"role": "system", "content": "你是一名资深 Java 业务代码分析师。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=1800,
            ),
            timeout=20.0,
        )
        text = (raw or "").strip()
        if text:
            return f"\n\n【深入分析】\n{text}"
    except asyncio.TimeoutError:
        logger.warning("deepen dig_in LLM timed out")
    except Exception:
        logger.debug("deepen dig_in LLM failed", exc_info=True)

    return ""


def _render_focus_method_answer(
    evidence: dict[str, Any],
    focus_cms: list[str],
    *,
    question: str = "",
    related_cms: list[str] | None = None,
    project_id: str = "",
    caller_cms: list[str] | None = None,
) -> str:
    """Dig-in answer: entry → chain topic hits, shallow → deep."""
    entry = _simple_cm(str(evidence.get("entry") or ""))
    src = evidence.get("spine_source") or "unknown"
    related = [_simple_cm(c) for c in (related_cms or []) if c]
    callers = [_simple_cm(c) for c in (caller_cms or []) if c]
    pid = project_id or str(evidence.get("project_id") or "")
    topic = str(
        (evidence.get("dig_in_decision") or {}).get("topic")
        or evidence.get("method_topic")
        or evidence.get("focus_topic")
        or ""
    ).strip()
    focus = _order_focus_shallow_to_deep(
        [_simple_cm(c) for c in (focus_cms or []) if c]
    )
    # Drop focus / callers from related so leaf is never demoted to「相关调用」.
    skip = set(focus) | set(callers)
    related = [c for c in related if c not in skip]
    topic_bit = f"「{topic}」" if topic else "子话题"
    lines = [
        f"业务入口：`{entry}`" if entry else "业务入口：",
        (
            f"先锁定入口主链，再在 CALLS 闭包上按{topic_bit}语义相似度筛选"
            "（由浅入深，未切换入口）："
        ),
        "",
    ]

    def _emit_focus(cm: str) -> None:
        cm_s = _simple_cm(cm)
        if not cm_s:
            return
        row = _row_for_cm(evidence, cm_s) or {}
        summary = str(
            row.get("summary")
            or row.get("neo4j_desc")
            or row.get("title")
            or ""
        ).strip()
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > 800:
            summary = summary[:800] + "…"
        label = _focus_depth_label(cm_s, focus)
        lines.append(f"- 主链叶子 `{cm_s}`（{label}）")
        if summary:
            lines.append(f"  摘要：{summary}")
        else:
            lines.append("  摘要：（库中暂无方法摘要；可继续追问源码）")
        extra = str(row.get("summary_dev_clear") or row.get("neo4j_desc") or "").strip()
        extra = re.sub(r"\s+", " ", extra)
        if extra and extra != summary:
            if len(extra) > 400:
                extra = extra[:400] + "…"
            lines.append(f"  补充：{extra}")
        for note in _render_focus_source_notes(pid, cm_s, max_lines=16):
            lines.append(note)
        lines.append("")

    for cm in focus:
        _emit_focus(cm)
    # Demoted umbrella / caller: one-line context only.
    for cm in callers:
        if cm in focus:
            continue
        lines.append(f"- 调用方（主链上层）：`{cm}`")
        lines.append("")
    if _wants_dig_in_change(question):
        lines.append(
            "改动提示：以上方法落在当前锁定入口的主链内；相关改动应落在这些方法及其"
            "直接调用上，不要切换到无关 Controller 入口。"
        )
        lines.append("")
    lines.append(
        f"依据：聚焦方法数={len(focus)}；相关={len(related)}；主链来源={src}；顺序=由浅入深"
    )
    return "\n".join(lines)


def _render_flow_answer_deterministic(evidence: dict[str, Any]) -> str:
    """Grounded main-chain text: order + Class#method + summary (no LLM reorder)."""
    entry = _simple_cm(str(evidence.get("entry") or ""))
    rows = evidence.get("mysql_rows") or []
    src = evidence.get("spine_source") or "unknown"
    lines = [
        f"业务入口：`{entry}`" if entry else "业务入口：",
        "主链：",
        "",
    ]
    step_i = 0
    for r in rows:
        cm = _simple_cm(str(r.get("class_method") or ""))
        if not cm:
            continue
        # Never repeat the entry method in the numbered spine.
        if entry and cm == entry:
            continue
        if str(r.get("rel") or "").upper() == "ENTRY":
            continue
        summary = str(
            r.get("summary") or r.get("neo4j_desc") or r.get("title") or ""
        ).strip()
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > 220:
            summary = summary[:220] + "…"
        step_i += 1
        if summary:
            lines.append(f"{step_i}. `{cm}` — {summary}")
        else:
            lines.append(f"{step_i}. `{cm}`")
    lines.append("")
    lines.append(f"依据：节点数={step_i}；主链来源={src}")
    return "\n".join(lines)


def _entry_match_meta(evidence: dict[str, Any], cm: str) -> dict[str, str]:
    """Pull api_doc / path / routing blurb for a matched entry if present."""
    cm_s = _simple_cm(cm)
    for key in ("entry_route_hits", "entry_match_meta", "resolve_routes"):
        blob = evidence.get(key)
        if not isinstance(blob, list):
            continue
        for h in blob:
            if not isinstance(h, dict):
                continue
            hit = _simple_cm(str(h.get("cm") or h.get("class_method") or ""))
            if hit != cm_s:
                continue
            return {
                "api_doc": str(h.get("api_doc") or "").strip(),
                "http_paths": str(h.get("http_paths") or "").strip(),
                "routing_text": str(h.get("routing_text") or "").strip(),
            }
    return {"api_doc": "", "http_paths": "", "routing_text": ""}


def _render_one_entry_flow_block(
    evidence: dict[str, Any],
    entry_cm: str,
    *,
    index: int,
) -> str:
    cm = _simple_cm(entry_cm)
    meta = _entry_match_meta(evidence, cm)
    spines = evidence.get("entry_spines") if isinstance(evidence.get("entry_spines"), dict) else {}
    spine = spines.get(cm) if isinstance(spines, dict) else None
    rows_map = (
        evidence.get("entry_mysql_rows")
        if isinstance(evidence.get("entry_mysql_rows"), dict)
        else {}
    )
    rows = list((rows_map or {}).get(cm) or [])
    src = ""
    if isinstance(spine, dict):
        src = str(spine.get("spine_source") or "")
        if not rows:
            # Fall back to synthesizing thin rows from calls list.
            for i, call in enumerate(list(spine.get("calls") or [])[:10], start=1):
                rows.append({"order": i, "class_method": call, "summary": ""})
    lines = [f"### {index}. `{cm}`"]
    if meta.get("api_doc"):
        lines.append(f"- 接口说明：{meta['api_doc']}")
    if meta.get("http_paths"):
        lines.append(f"- 路径：{meta['http_paths']}")
    if meta.get("routing_text") and meta.get("routing_text") != meta.get("api_doc"):
        rt = re.sub(r"\s+", " ", meta["routing_text"]).strip()
        if len(rt) > 180:
            rt = rt[:180] + "…"
        lines.append(f"- 摘要：{rt}")
    lines.append("主链：")
    step_i = 0
    for r in rows:
        step_cm = _simple_cm(str(r.get("class_method") or ""))
        if not step_cm or step_cm == cm:
            continue
        summary = str(
            r.get("summary") or r.get("neo4j_desc") or r.get("title") or ""
        ).strip()
        summary = re.sub(r"\s+", " ", summary)
        if len(summary) > 180:
            summary = summary[:180] + "…"
        step_i += 1
        if summary:
            lines.append(f"{step_i}. `{step_cm}` — {summary}")
        else:
            lines.append(f"{step_i}. `{step_cm}`")
    if step_i == 0:
        lines.append("（暂未展开到可核对的下游步骤）")
    lines.append(f"依据：节点数={step_i}" + (f"；主链来源={src}" if src else ""))
    return "\n".join(lines)


def _render_multi_entry_flow_answer(evidence: dict[str, Any]) -> str:
    """When several sibling Controllers match, reply with all of them."""
    matches = [
        _simple_cm(str(x))
        for x in (evidence.get("entry_matches") or [])
        if x and "#" in str(x)
    ]
    if len(matches) < 2:
        return _render_flow_answer_deterministic(evidence)
    lines = [
        f"检索到 {len(matches)} 个相关系统入口（同名方法/兄弟 Controller），如下一并说明：",
        "",
    ]
    for i, cm in enumerate(matches[:4], start=1):
        lines.append(_render_one_entry_flow_block(evidence, cm, index=i))
        lines.append("")
    lines.append("可点名其中一个 `Class#method` 继续追问细节。")
    return "\n".join(lines).rstrip() + "\n"


_PEV_FLOW_SUMMARIZE_PROMPT = load_prompt("pev_flow_summarize_prompt")
async def _exec_summarize_flow(
    question: str,
    evidence: dict[str, Any],
    *,
    project_id: str = "",
    history_text: str = "",
) -> str:
    import asyncio

    from app.agent.ask_retrieval import (
        _REFUSE_NO_ENTRY,
        _REFUSE_THIN_EVIDENCE,
        _mysql_evidence_row_count,
    )

    entry = evidence.get("entry") or ""
    rows = evidence.get("mysql_rows") or []
    if not entry:
        return _REFUSE_NO_ENTRY

    matches = [
        _simple_cm(str(x))
        for x in (evidence.get("entry_matches") or [])
        if x and "#" in str(x)
    ]
    # Sibling Controllers (e.g. two #creditNotification): answer all, skip single-entry dig-in.
    if len(matches) >= 2 and not evidence.get("lock_entry"):
        multi = _render_multi_entry_flow_answer(evidence)
        # Thin primary spine alone should not refuse when siblings were listed.
        if "相关系统入口" in multi:
            return multi

    # Scheme 3: entry-scoped dig-in → focused method answer (forbid full-chain rehash).
    # Works under session lock_entry OR first-turn ask_scope=entry_scoped_dig_in.
    # Description / answer-step match FIRST — never wait on dig_in LLM for the cheap path.
    dig_decision: dict[str, Any] = {}
    focus_cms: list[str] = []
    dig_in_focus = False
    pid = project_id or str(evidence.get("project_id") or "")
    if _allows_entry_scoped_dig_in(evidence):
        fast = _try_lock_dig_in_description_focus(
            question, evidence, project_id=pid
        )
        if fast:
            focus_cms, dig_decision = fast
            dig_in_focus = bool(focus_cms)
        else:
            # Ambiguity / no description hit → short dig_in LLM, then retrieval.
            dig_timeout = 3.0 if evidence.get("dig_in_likely") else 12.0
            dig_decision = await decide_dig_in_focus(
                question,
                evidence,
                prior_answer=_prior_answer_snippet(history_text),
                timeout=dig_timeout,
            )
            unresolved = False
            if dig_decision.get("dig_in"):
                focus_cms = _resolve_focus_method_symbol(
                    str(dig_decision.get("focus_method") or ""),
                    evidence,
                    project_id=pid,
                    topic=str(
                        dig_decision.get("topic")
                        or evidence.get("focus_topic")
                        or ""
                    ),
                )
                focus_cms = _prefer_topic_focus_cms(
                    focus_cms,
                    dig_decision,
                    evidence,
                    project_id=pid,
                    question=question,
                )[:1]
                dig_in_focus = bool(focus_cms)
                if not focus_cms:
                    unresolved = True
                    dig_decision = {
                        **dig_decision,
                        "dig_in": False,
                        "reason": (
                            str(dig_decision.get("reason") or "")
                            + "; unresolved_focus"
                        ).strip("; "),
                    }
            # Timeout / error / unresolved: retrieval focus (do not silent full-chain).
            # Respect an explicit LLM dig_in=false (source=llm) — no override.
            src = str(dig_decision.get("source") or "")
            need_retrieval_fb = (not dig_in_focus) and (
                src in {"timeout", "error"}
                or unresolved
                or (
                    bool(evidence.get("dig_in_likely"))
                    and src in {"timeout", "error", "default"}
                )
            )
            if need_retrieval_fb:
                probe = (
                    str(
                        evidence.get("method_topic")
                        or evidence.get("focus_topic")
                        or ""
                    ).strip()
                    or question
                )
                focus_cms = _fallback_dig_in_focus_cms(
                    probe, evidence, project_id=pid
                )[:1]
                if focus_cms:
                    # Answer-step CJK anchor can still promote a tighter leaf.
                    dig_decision = {
                        "dig_in": True,
                        "focus_method": focus_cms[0],
                        "topic": str(
                            dig_decision.get("topic")
                            or evidence.get("focus_topic")
                            or ""
                        ),
                        "reason": f"retrieval_fallback_after_{src or 'miss'}",
                        "source": "retrieval_fallback",
                    }
                    focus_cms = _prefer_topic_focus_cms(
                        focus_cms,
                        dig_decision,
                        evidence,
                        project_id=pid,
                        question=question,
                    )[:1]
                    dig_in_focus = True
                    dig_decision["focus_method"] = focus_cms[0]
                    # E: zero-overlap guard — a fallback focus with no topic
                    # overlap must not be answered confidently (it is a timeout
                    # artifact, not a match). Degrade to an honest note instead.
                    if _focus_zero_overlap(focus_cms[0], probe, evidence):
                        evidence["dig_in_demoted_zero_overlap"] = True
                        evidence["dig_in_demoted_topic"] = str(probe)
                        evidence["dig_in_demoted_focus"] = focus_cms[0]
                        focus_cms = []
                        dig_in_focus = False
                        dig_decision = {
                            **dig_decision,
                            "dig_in": False,
                            "demoted_focus": True,
                            "reason": (
                                f"retrieval_fallback_after_{src or 'miss'}; "
                                "zero_overlap_focus_demoted"
                            ),
                        }

    if dig_in_focus and focus_cms:
        topic = str(
            dig_decision.get("topic") or evidence.get("focus_topic") or ""
        )
        src_lock = str(dig_decision.get("source") or "")
        related: list[str] = []
        demoted_parent = ""
        # method-first already locked topic method(s) — skip extra Neo4j/source.
        if not src_lock.startswith("method_first"):
            focus_cms = list(focus_cms)[:1]
            related = _related_cms_under_focus(
                pid, focus_cms, question, evidence, topic=topic
            )
            # Post-pass safety net: related leaf beats umbrella parent focus.
            # (desktop-chat-022e6fc524a1: synchroMemberPrice already in related.)
            focus_cms, related, demoted_parent = _promote_related_leaf_to_focus(
                focus_cms,
                related,
                evidence,
                question=question,
                topic=topic,
            )
            if demoted_parent:
                dig_decision["reason"] = (
                    str(dig_decision.get("reason") or "") + "; related_leaf_promote"
                ).strip("; ")
                src0 = str(dig_decision.get("source") or "")
                if src0 and "related_leaf" not in src0:
                    dig_decision["source"] = f"{src0}+related_leaf_promote"
                elif not src0:
                    dig_decision["source"] = "related_leaf_promote"
                related = _related_cms_under_focus(
                    pid, focus_cms, question, evidence, topic=topic
                )
        # Expand to every topic-related method on the Neo4j entry CALLS chain.
        chain = _entry_chain_cms_set(evidence, project_id=pid)
        seed_rank = {
            _simple_cm(c): i
            for i, c in enumerate(focus_cms)
            if c and "#" in str(c)
        }
        # Prefer earlier method-first search ranks over a mistaken umbrella seed.
        for i, raw in enumerate(evidence.get("method_first_on_chain") or []):
            cm = _simple_cm(str(raw or ""))
            if cm and "#" in cm and cm not in seed_rank:
                seed_rank[cm] = i
        if pid:
            peer_cms = _topic_peer_cms_for_enrich(
                chain,
                list(focus_cms)
                + list(evidence.get("method_first_candidates") or [])
                + list(evidence.get("method_first_on_chain") or []),
                limit=32,
            )
            try:
                _enrich_focus_rows(pid, evidence, peer_cms)
            except Exception:
                logger.debug("chain-topic peer enrich failed", exc_info=True)
        topic_cms = _collect_topic_cms_on_chain(
            topic,
            chain,
            evidence,
            entry=_simple_cm(str(evidence.get("entry") or "")),
            search_rank=seed_rank,
            question=question,
        )
        if topic_cms:
            fam_set = set(topic_cms)
            related = [c for c in related if _simple_cm(c) not in fam_set]
            if [_simple_cm(c) for c in topic_cms] != [
                _simple_cm(c) for c in focus_cms if c
            ]:
                dig_decision["reason"] = (
                    str(dig_decision.get("reason") or "") + "; chain_topic"
                ).strip("; ")
            focus_cms = topic_cms
        else:
            # Drop incidental umbrella parents when a tighter seed leaf exists.
            kept = [
                c
                for c in focus_cms
                if c
                and not _incidental_topic_umbrella(
                    str(
                        (_row_for_cm(evidence, _simple_cm(c)) or {}).get("summary")
                        or (_row_for_cm(evidence, _simple_cm(c)) or {}).get("neo4j_desc")
                        or ""
                    ),
                    topic,
                )
            ]
            focus_cms = list(kept or focus_cms)[:1]
        caller_cms = [demoted_parent] if demoted_parent else []
        _enrich_focus_rows(
            pid, evidence, list(focus_cms) + list(related) + list(caller_cms)
        )
        focus_cms = _order_focus_shallow_to_deep(list(focus_cms))
        dig_decision = {
            **dig_decision,
            "dig_in": True,
            "focus_method": focus_cms[0] if focus_cms else "",
            "focus_cms": list(focus_cms),
            "dig_in_likely": bool(evidence.get("dig_in_likely")),
            "demoted_parent": demoted_parent or "",
            "ask_scope": str(evidence.get("ask_scope") or ""),
            "order": "shallow_to_deep",
        }
        evidence["dig_in_focus"] = True
        evidence["dig_in_focus_cms"] = list(focus_cms)
        evidence["dig_in_related"] = related
        evidence["dig_in_caller"] = list(caller_cms)
        evidence["dig_in_decision"] = dig_decision
        # Dig-in: lead with leaf body; parent only as caller context.
        answer = _render_focus_method_answer(
            evidence,
            focus_cms,
            question=question,
            related_cms=related,
            project_id=pid,
            caller_cms=caller_cms,
        )
        # Deepen: if focus method was already in prior answer (user is drilling
        # into a sub-step), read source code and add focused analysis.
        if (
            pid
            and len(focus_cms) == 1
            and _focus_was_in_prior_answer(focus_cms[0], evidence)
        ):
            deeper = await _deepen_dig_in_with_source_code(
                pid,
                focus_cms[0],
                question,
                topic=str(
                    dig_decision.get("topic")
                    or evidence.get("focus_topic")
                    or evidence.get("method_topic")
                    or ""
                ),
            )
            if deeper:
                answer = f"{answer}\n{deeper}"
        return answer

    if dig_decision:
        dig_decision = {
            **dig_decision,
            "focus_cms": list(focus_cms)[:1] if focus_cms else [],
            "dig_in_likely": bool(evidence.get("dig_in_likely")),
        }
        evidence["dig_in_decision"] = dig_decision

    # Reused spine skipped full mysql_enrich — fill rows before full-chain render.
    if (
        not dig_in_focus
        and evidence.get("reuse_shared_spine")
        and not (evidence.get("mysql_rows") or [])
        and pid
        and evidence.get("calls")
    ):
        try:
            await _exec_mysql_enrich(pid, evidence)
        except Exception:
            logger.debug("reuse full-chain mysql enrich failed", exc_info=True)
    rows = evidence.get("mysql_rows") or []

    # E: retrieval-fallback focus was demoted (zero topic overlap) — never
    # answer with an unrelated spine; emit an honest "not located" note.
    if not dig_in_focus and evidence.get("dig_in_demoted_zero_overlap"):
        topic = str(evidence.get("dig_in_demoted_topic") or question or "")
        focus = str(evidence.get("dig_in_demoted_focus") or "")
        entry = str(evidence.get("entry") or "")
        # If the entry method itself matches the topic, answer it directly —
        # demoting a bad *focus leaf* must not swallow a good *entry* answer
        # (loanSwitch: entry=…#creditUsageApplyLoanSwitch matches "loanSwitch").
        if entry and _focus_zero_overlap(entry, topic, evidence) is False:
            _enrich_focus_rows(pid, evidence, [entry])
            evidence["dig_in_focus"] = True
            evidence["dig_in_focus_cms"] = [entry]
            evidence["dig_in_decision"] = {
                **dict(evidence.get("dig_in_decision") or {}),
                "dig_in": True,
                "focus_method": entry,
                "reason": "entry_matches_topic_after_demote",
            }
            return _render_focus_method_answer(
                evidence,
                [entry],
                question=question,
                related_cms=[],
                project_id=pid,
                caller_cms=[],
            )
        lines = [
            f"未能定位到与「{topic}」直接相关的代码。",
        ]
        if entry:
            lines.append(f"已检索入口 `{entry}` 的调用链及其中各方法摘要，均与本题主题不匹配。")
        if focus:
            lines.append(f"检索命中的最近方法 `{focus}` 与本题无关，未作为答案给出。")
        lines.append(
            "如目标代码确实存在，请补充具体表名/接口名/方法名；"
            "或改为该入口的完整流程类问法，我可以给出整条链路。"
        )
        return "\n".join(lines)

    if _mysql_evidence_row_count(rows) < 2 and (evidence.get("calls_count") or 0) < 2:
        return _REFUSE_THIN_EVIDENCE

    grounded = _render_flow_answer_deterministic(evidence)
    src = str(evidence.get("spine_source") or "")

    # Business-narrative polish: even ordered/CALLS spines get an LLM pass so the
    # answer reads as a business story, not a mechanical "该方法用于…" list
    # (desktop-chat-75fe902deb0b). The deterministic grounded text is kept as the
    # verified fallback if the LLM drifts (drops anchors / reorders / times out).
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        steps_payload = [
            {
                "order": r.get("order"),
                "class_method": _simple_cm(str(r.get("class_method") or "")),
                "summary": str(r.get("summary") or r.get("neo4j_desc") or "")[:300],
            }
            for r in rows
        ]

        async def _polish() -> str:
            raw = await llm.chat(
                [
                    {"role": "system", "content": _PEV_FLOW_SUMMARIZE_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"问题: {question}\n"
                            f"SOURCE: {evidence.get('spine_source')}\n"
                            f"已定稿主链（勿打乱）:\n{grounded}\n\n"
                            f"证据 steps JSON:\n"
                            f"{json.dumps(steps_payload, ensure_ascii=False)}"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=2200,
            )
            return (raw or "").strip()

        text = await asyncio.wait_for(_polish(), timeout=40.0)
        if text:
            cms = [
                _simple_cm(str(r.get("class_method") or ""))
                for r in rows
                if r.get("class_method")
            ]
            keep = _simple_cm(entry) in text or (
                cms and cms[0].split("#")[-1] in text
            )
            hits = sum(
                1
                for cm in cms[:8]
                if cm and (cm in text or cm.split("#")[-1] in text)
            )
            if keep and hits >= min(2, len(cms)):
                return text
    except asyncio.TimeoutError:
        logger.warning("PEV flow polish timed out after 25s; use deterministic")
    except Exception:
        logger.debug("PEV flow polish LLM failed; use deterministic", exc_info=True)

    return grounded





# --- lazy re-exports (monkeypatch-compatible) ---

async def decide_dig_in_focus(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import decide_dig_in_focus as _impl
    return await _impl(*args, **kwargs)



def _related_cms_under_focus(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _related_cms_under_focus as _impl
    return _impl(*args, **kwargs)



def _enrich_focus_rows(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _enrich_focus_rows as _impl
    return _impl(*args, **kwargs)



def _render_focus_source_notes(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _render_focus_source_notes as _impl
    return _impl(*args, **kwargs)




# --- lazy re-exports (monkeypatch-compatible) ---

def _prefer_topic_focus_cms(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _prefer_topic_focus_cms as _impl
    return _impl(*args, **kwargs)



def _resolve_focus_method_symbol(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _resolve_focus_method_symbol as _impl
    return _impl(*args, **kwargs)




# --- lazy re-exports (monkeypatch-compatible) ---

def _try_lock_dig_in_description_focus(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _try_lock_dig_in_description_focus as _impl
    return _impl(*args, **kwargs)



async def _exec_mysql_enrich(*args: Any, **kwargs: Any) -> Any:
    """Lazy re-export from pev_agent so monkeypatch on pev_agent works."""
    from app.agent.pev_agent import _exec_mysql_enrich as _impl
    return await _impl(*args, **kwargs)

