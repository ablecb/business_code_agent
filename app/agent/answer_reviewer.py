"""Answer completeness reviewer — second AI pass before showing the user reply."""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from app.core.utils import truncate

logger = logging.getLogger("app.agent.reviewer")

_REVIEW_PROMPT = """你是代码流程解读的「评审员」。根据用户问题与助手回答，判断回答是否足够回答该问题。
准确优先：回答不得误导；编造未在证据中出现的表名/字段/接口 → 必须判不够。
只输出严格 JSON（不要 markdown）：
{
  "ok": true/false,
  "score": 0到100整数,
  "summary": "一句话评审结论",
  "gaps": ["缺失点1", "缺失点2"],
  "followups": ["若不够，下一步应查什么（可执行的 search/读码指令）"]
}

评审标准（按问题类型取相关项；没问到的不苛求）：
1. 入口是否说清（接口/方法/请求从哪来）
2. 主链步骤是否按真实调用顺序，且每步有可核对事实（不能只写「继续处理」）
3. 校验：校验了哪些字段、失败怎么返回
4. 落库：何时写库、写哪张表/实体、成功/失败标准、关键字段如何变
5. 外部协同：调用了哪个外部系统/Client/MQ，怎么调（同步/异步、关键入参或接口名），结果如何影响本流程
6. 状态流转：关键业务状态如何变化
7. 禁止空话：若出现「构造对象并继续处理」「进行后续业务」而无具体下游，必须判不够
8. 表名/字段/接口：必须能在「已读代码摘要」中核对；摘要未出现则不得写臆测名；
   若回答编造了摘要中没有的 t_* 表 → ok=false，gaps 写「编造未在证据中出现的表名」
9. 关卡追问有 symbol_desc/业务摘要覆盖时：要求规则接地气，不要求粘贴完整源码

判定：
- 流程/链路类问题：缺落库或外部调用细节之一，且正文没有明确「代码未见…」说明 → ok=false
- 仅问地址/清单/校验字段 → 按该项标准即可
- 追问细节类：若已点名具体方法并写清规则/外部调用，即使未贴完整源码、也未写落库表名 → 可 ok=true
- 回答出现证据摘要中没有的表名 → 一律 ok=false
- gaps/followups 要具体，便于助手继续 search/get_symbol_code；勿要求编造表名
"""

_TABLE_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])(t_[a-z][a-z0-9_]{2,40})(?![A-Za-z0-9_])", re.I)

# No product-specific fake-table denylist in agent core.
_KNOWN_FAKE_TABLES: frozenset[str] = frozenset()


def _is_gate_detail_ask(question: str) -> bool:
    """True for a short dig-in into a check/gate, not a full-flow ask."""
    from app.agent.question_intent import (
        wants_add_field_change_plan,
        wants_show_gate_code,
    )

    q = question or ""
    if wants_add_field_change_plan(q) or wants_show_gate_code(q):
        return False
    if re.search(r"流程|链路|主链|完整|串一下|有哪些接口", q) and not re.search(
        r"校验|关卡|这一块|这块|干嘛", q
    ):
        return False
    return bool(
        re.search(r"校验|关卡|这一块|这块|干嘛的|具体逻辑", q)
        and len(q.strip()) <= 48
    )


def invented_tables_not_in_evidence(answer: str, evidence: str = "") -> list[str]:
    """Return t_* names cited in answer but absent from evidence (or known fakes)."""
    a = answer or ""
    ev = (evidence or "").lower()
    found: list[str] = []
    for m in _TABLE_NAME_RE.finditer(a):
        name = m.group(1)
        low = name.lower()
        if low in ev:
            continue
        found.append(name)
    # Dedup preserve order
    return list(dict.fromkeys(found))


async def review_answer(
    question: str,
    answer: str,
    *,
    evidence: str = "",
    plan_title: str = "",
) -> dict[str, Any]:
    """LLM review of explore answer. Returns ok/score/gaps/followups."""
    heuristic = _heuristic_review(question, answer, evidence=evidence)
    from app.agent.question_intent import (
        wants_add_field_change_plan,
        wants_composite_change,
        wants_extend_field_to_protocol_sync,
        wants_show_gate_code,
    )

    # Change-impact plans must stay scannable — skip generic flow LLM review.
    if wants_extend_field_to_protocol_sync(question):
        return {**heuristic, "source": "heuristic_protocol_sync"}
    if wants_composite_change(question):
        return {**heuristic, "source": "heuristic_composite_field"}
    if wants_add_field_change_plan(question):
        return {**heuristic, "source": "heuristic_add_field"}
    if wants_show_gate_code(question):
        return {**heuristic, "source": "heuristic_show_gate_code"}
    try:
        from app.llm.factory import get_llm_provider

        llm = get_llm_provider()
        content = await llm.chat(
            [
                {"role": "system", "content": _REVIEW_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"用户问题：{question}\n"
                        f"研究目标：{plan_title or '（无）'}\n"
                        f"助手回答：\n{truncate(answer, 3500)}\n"
                        f"已读代码摘要（可选）：\n{truncate(evidence, 1800)}"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=700,
        )
        parsed = _parse_review(content)
        if parsed:
            # Merge heuristic gaps if LLM missed obvious ones.
            gaps = list(parsed.get("gaps") or [])
            for g in heuristic.get("gaps") or []:
                if g not in gaps:
                    gaps.append(g)
            followups = list(parsed.get("followups") or [])
            for f in heuristic.get("followups") or []:
                if f not in followups:
                    followups.append(f)
            gaps = _soften_source_code_gaps(gaps, question, answer, evidence)
            followups = _soften_source_code_gaps(followups, question, answer, evidence)
            ok = bool(parsed.get("ok"))
            if gaps and heuristic.get("force_fail"):
                ok = False
            # Invented entities always fail grounding (sanitize should have stripped;
            # if still present, reject).
            if heuristic.get("force_fail") and any(
                "编造" in str(g) or "未在证据" in str(g) for g in gaps
            ):
                ok = False
            # Gate dig-in with method name + no invented entities → soften only
            # persist/source gaps (do not auto-pass solely on method-name mention).
            if (
                _is_gate_detail_ask(question)
                and _answer_mentions_gate_method(answer)
                and not heuristic.get("force_fail")
                and not gaps
            ):
                ok = True
            score = int(parsed.get("score") or 0)
            if ok and score < 70:
                score = max(score, 75)
            if not ok and score >= 75:
                score = min(score, 60)
            return {
                "ok": ok,
                "score": max(0, min(100, score)),
                "summary": str(parsed.get("summary") or heuristic.get("summary") or ""),
                "gaps": gaps[:8],
                "followups": followups[:6],
                "source": "llm",
            }
    except Exception as e:
        logger.warning("answer review LLM failed (%s); heuristic only", e)

    return {**heuristic, "source": "heuristic"}


def review_reject_message(review: dict[str, Any], *, question: str = "") -> str:
    """Turn review result into a finish-rejection nudge for the explore loop.

    Agent-facing only (not end-user UI). Keep directive enough to drive dig-deeper.
    """
    from app.agent.question_intent import (
        wants_add_field_change_plan,
        wants_extend_field_to_protocol_sync,
    )

    gaps = review.get("gaps") or []
    followups = review.get("followups") or []
    if wants_extend_field_to_protocol_sync(question):
        lines = [
            "本题是「字段在协议同步时作为入参透传」改动方案，不是校验清单。",
            f"缺口提示：{review.get('summary') or '缺协议同步组装触点'}",
        ]
        if gaps:
            lines.append("缺失：" + "；".join(str(g) for g in gaps[:5]))
        if followups:
            lines.append("请继续：" + "；".join(str(f) for f in followups[:4]))
        lines.append(
            "优先 finish 原样采用 discover_protocol_sync_plan 的 plan_markdown；"
            "版式：## 结论 → ## 改动清单（可粘贴 java）；不要硬写 DDL；"
            "禁止校验字段罗列/关卡清单/状态流转。"
        )
        return "\n".join(lines)
    if wants_add_field_change_plan(question):
        lines = [
            "本题是「新增字段要改哪些」改动方案，不是流程解说。",
            f"缺口提示：{review.get('summary') or '缺改动顺序/DDL'}",
        ]
        if gaps:
            lines.append("缺失：" + "；".join(str(g) for g in gaps[:5]))
        if followups:
            lines.append("请继续：" + "；".join(str(f) for f in followups[:4]))
        lines.append(
            "优先 finish 原样采用 discover_add_field_plan 的 plan_markdown；"
            "否则按已读证据重写：## 结论 → ## 改动顺序 → ## 从 → 到 → ## DDL → ## 说明"
            "（节与节空一行）。禁止旧标题「改动清单/建议改动」；"
            "只列已读到的角色；禁止套固定方法清单；"
            "MP insert 一般不必改 Mapper.xml；"
            "禁止业务状态流转/外部系统调用/评审提示；禁止只罗列 insert* 关卡。"
        )
        return "\n".join(lines)
    lines = [
        "回答还不够具体，请补充更多实现细节后再 finish。",
        f"缺口提示：{review.get('summary') or '缺少关键业务细节'}",
    ]
    if gaps:
        lines.append("缺失：" + "；".join(str(g) for g in gaps[:5]))
    if followups:
        lines.append("请继续：" + "；".join(str(f) for f in followups[:4]))
    else:
        lines.append(
            "请 get_symbol_code 下钻 ServiceImpl：写清何时落库、写哪张表/字段、"
            "外部系统如何调用（Feign Client 方法名或 MQ topic），不要再用「继续处理」收尾。"
        )
    lines.append(
        "禁止只写 Redisson/锁当外部协同；必须读到 *Client/Feign/MQ 与 Mapper insert/update/save。"
        "禁止编造未在源码/摘要中出现的表名（如 t_user_quota）。"
        "证据够后再 finish，把上述事实写进主链步骤；未见则写「未在已读材料中看到…」。"
    )
    return "\n".join(lines)


def _parse_review(content: str) -> dict[str, Any] | None:
    if not content:
        return None
    m = re.search(r"\{.*\}", content, re.DOTALL)
    raw = m.group(0) if m else content
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    gaps = data.get("gaps") if isinstance(data.get("gaps"), list) else []
    followups = data.get("followups") if isinstance(data.get("followups"), list) else []
    try:
        score = int(data.get("score") or 0)
    except Exception:
        score = 0
    return {
        "ok": bool(data.get("ok")),
        "score": score,
        "summary": str(data.get("summary") or ""),
        "gaps": [str(g).strip() for g in gaps if str(g).strip()],
        "followups": [str(f).strip() for f in followups if str(f).strip()],
    }


def _answer_mentions_gate_method(answer: str) -> bool:
    """True when answer cites any Class#method (generic, not product method list)."""
    a = answer or ""
    return bool(re.search(r"[A-Za-z][A-Za-z0-9_]{2,80}#[A-Za-z][A-Za-z0-9_]{1,80}", a))


def _soften_source_code_gaps(
    items: list[str],
    question: str,
    answer: str,
    evidence: str,
) -> list[str]:
    """Drop gaps that demand pasted source when narrative/evidence already covers it."""
    out: list[str] = []
    has_method = _answer_mentions_gate_method(answer) or bool(
        re.search(r"[A-Za-z][A-Za-z0-9_]+#[A-Za-z]", evidence or "")
    )
    has_desc = bool(
        re.search(
            r"(?i)Client\.|Service\.|调用|校验|比较|判断|返回",
            (answer or "") + "\n" + (evidence or ""),
        )
    )
    gate = _is_gate_detail_ask(question)
    for raw in items:
        s = str(raw).strip()
        if not s:
            continue
        # Gate dig-in with method and/or desc facts → never demand pasted source.
        if (has_method or has_desc) and re.search(
            r"具体实现代码|未提供.*源码|未读到.*实现|未查看.*实现|贴出.*代码",
            s,
        ):
            continue
        if gate and re.search(r"落库|写哪张表|实体表结构", s):
            continue
        if re.search(r"t_user_quota|t_product_config|臆测表|编造表", s):
            # Keep grounding gaps that flag invention; drop soft "should invent table" noise.
            if re.search(r"编造|未在证据|臆测", s):
                out.append(s)
            continue
        out.append(s)
    return out


def _heuristic_review(
    question: str,
    answer: str,
    *,
    evidence: str = "",
) -> dict[str, Any]:
    """Deterministic shallow checks for flow-like answers."""
    q = question or ""
    a = answer or ""
    gaps: list[str] = []
    followups: list[str] = []
    gateish = _is_gate_detail_ask(q)
    flowish = any(k in q for k in ("流程", "链路", "怎么走", "什么样", "通知", "回调"))
    # Gate dig-ins are not full-flow persist reviews.
    if gateish:
        flowish = False
    from app.agent.question_intent import (
        wants_add_field_change_plan,
        wants_extend_field_to_protocol_sync,
    )

    add_field = wants_add_field_change_plan(q)
    protocol_sync = wants_extend_field_to_protocol_sync(q)
    if add_field or protocol_sync:
        flowish = False
    force_fail = False
    if not a.strip():
        return {
            "ok": False,
            "score": 0,
            "summary": "空回答",
            "gaps": ["无有效回答"],
            "followups": ["重新 search 入口并读码"],
            "force_fail": True,
        }

    # Grounding: inventing t_* not in evidence bag is always a hard fail.
    invented = invented_tables_not_in_evidence(a, evidence)
    known_hits = [t for t in invented if t.lower() in _KNOWN_FAKE_TABLES]
    if invented:
        label = "、".join(invented[:4])
        gaps.append(f"编造未在证据中出现的表名（{label}）")
        followups.append(
            "删除臆测表名；仅引用 raw_code/symbol_desc/SQL 中出现的表，"
            "否则写「未在已读材料中看到具体表名」"
        )
        force_fail = True
    elif any(t in a.lower() for t in _KNOWN_FAKE_TABLES):
        # Evidence somehow contains the fake name — still soft-flag known hallucinations
        # when they appear as the only table claim without Client/SQL grounding.
        if not re.search(r"(?i)aqcClient|Mapper\.|FROM\s+t_|INTO\s+t_", a + evidence):
            gaps.append("疑似沿用已知臆测表名，请对照源码核对")
            force_fail = True

    if add_field:
        has_vo = bool(re.search(r"(?i)ReqVo|Request\b|Dto\b|入参\s*VO|private\s+String", a))
        has_bind = bool(
            re.search(
                r"(?i)reqMap\.get|RequestBody|绑定|check\w*Param|set[A-Z]\w*\s*\(",
                a,
            )
        )
        has_ext = bool(
            re.search(
                r"(?i)@TableName|Ext\b|Dmo\b|实体|Mapper\.insert|BaseMapper|落库组装",
                a,
            )
        )
        has_ddl = bool(re.search(r"(?i)ALTER\s+TABLE|##\s*DDL|不能确定表名", a))
        honest_gap = bool(
            re.search(r"未定位到|未读全|尚未读到|证据不足|缺口|仅含已发现", a)
        )
        # Full chain OR honest partial (found roles + gap note) — never require
        # inventing the missing bind/assemble steps to pass review.
        has_plan = bool(re.search(r"改动清单|改动顺序", a)) and has_ddl and (
            (has_vo and has_bind and has_ext)
            or (honest_gap and (has_vo or has_bind or has_ext))
        )
        insert_only = bool(
            re.search(r"(?i)insertLoanData|insertOrUpdateLoanInfo|写入主表|回写主表", a)
        ) and not has_plan
        if insert_only or not has_plan:
            gaps.append("未给出新增字段改动清单（已读角色 + DDL/缺口说明）")
            followups.append(
                "按目标入口已读链路列改动；缺角色时标明缺口，禁止套固定方法清单"
            )
            force_fail = True
        if not has_bind and not honest_gap:
            gaps.append("缺入参绑定/校验改动说明")
            followups.append("补读该入口的绑定/校验方法（reqMap.get 或等价）")
            force_fail = True
        if re.search(
            r"(?i)(?:复制到|赋值到)\s*`?WithdrawalsApply`?(?!ReqVo|RespVo|Service)\s*实体"
            r"|(?:WithdrawalsApply(?!ReqVo|RespVo|Service)|demo_b_withdrawals_apply)"
            r".{0,60}(?:ADD\s+COLUMN|加列)",
            a,
        ):
            gaps.append("误把字段落到未证实的主表/Apply 实体；应对齐已读持久化实体")
            followups.append("按已读组装方法写入对应 Ext/实体；DDL 只改证据中的表")
            force_fail = True
        # Synth leftovers from flow-review — must not ship in add-field answers.
        if re.search(
            r"业务状态流转|外部系统调用[：:].{0,40}未在已读|"
            r"[（(]评审提示|可追问「|##\s*技术锚点",
            a,
        ):
            gaps.append("含业务状态流转/外部调用/评审/技术锚点等残留，应删除")
            followups.append(
                "只保留 ## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL / ## 说明，"
                "删掉状态与外部调用段落"
            )
            force_fail = True
        if not re.search(r"改动清单|改动顺序", a):
            gaps.append("版式缺「改动顺序/改动清单」标题，不便扫读")
            followups.append(
                "用 ## 结论 / ## 改动顺序 / ## 从 → 到 / ## DDL 重写，节与节空一行"
            )
            force_fail = True
        return {
            "ok": not force_fail and has_plan,
            "score": 85 if has_plan and not force_fail else 40,
            "summary": "新增字段改动方案可用" if has_plan and not force_fail else "缺改动清单",
            "gaps": gaps,
            "followups": followups,
            "force_fail": force_fail,
        }

    if protocol_sync:
        has_plan = bool(re.search(r"改动清单|##\s*结论", a))
        has_sync = bool(
            re.search(
                r"(?i)协议同步|syncSign|\.put\s*\(|同步入参|透传",
                a,
            )
        )
        honest_gap = bool(
            re.search(r"未定位到|未读全|尚未读到|证据不足|缺口|仅含已发现", a)
        )
        if not has_plan or (not has_sync and not honest_gap):
            gaps.append("未给出协议同步透传改动清单（同步 VO / *Map.put）")
            followups.append("调用 discover_protocol_sync_plan 或读 syncSign* 方法后重写")
            force_fail = True
        if re.search(
            r"queryProductGuarateeName|业务状态流转|##\s*技术锚点|评审提示",
            a,
        ):
            gaps.append("含校验关卡/状态流转等离题残留，应删除")
            followups.append("只保留 ## 结论 / ## 改动清单")
            force_fail = True
        ok = bool(has_plan and (has_sync or honest_gap) and not force_fail)
        return {
            "ok": ok,
            "score": 85 if ok else 40,
            "summary": "协议同步透传方案可用" if ok else "缺协议同步透传清单",
            "gaps": gaps,
            "followups": followups,
            "force_fail": force_fail,
        }

    if gateish:
        if force_fail:
            return {
                "ok": False,
                "score": 40,
                "summary": "关卡回答含未接地气的表名/实体",
                "gaps": gaps,
                "followups": followups,
                "force_fail": True,
            }
        # Require method anchor + at least one evidence-ish cue (not name alone).
        has_evidence_cue = bool(
            re.search(
                r"(?i)调用|校验|比较|判断|返回|拒绝|失败|可用|冻结|开关|"
                r"Client\.|Service\.|CALLS|主链|闭包",
                a,
            )
        )
        if _answer_mentions_gate_method(a) and has_evidence_cue and len(a.strip()) >= 40:
            return {
                "ok": True,
                "score": 82,
                "summary": "关卡细节已覆盖",
                "gaps": gaps[:4],
                "followups": followups[:4],
                "force_fail": False,
            }
        if len(a.strip()) < 40:
            gaps.append("关卡追问回答过短，未写清校验规则")
            followups.append("按锁定入口 CALLS 闭包读主题匹配方法并引用证据")
            force_fail = True
        return {
            "ok": not force_fail,
            "score": 80 if not force_fail else 45,
            "summary": "关卡细节不足" if force_fail else "关卡回答可用",
            "gaps": gaps,
            "followups": followups,
            "force_fail": force_fail,
        }
    if flowish:
        vague_tail = bool(
            re.search(
                r"(构造.{0,12}对象.{0,12}继续|进行后续|继续处理|后续业务处理|并继续处理)",
                a,
            )
        )
        has_persist = bool(
            re.search(
                r"落库|写入|insert|update|save|Mapper\.|持久化|保存到|保存|写(入)?表|实体|"
                r"代码中未见.{0,8}(落库|写库|持久化)|未在已读材料中看到.{0,12}(表|落库)",
                a,
                re.I,
            )
        )
        has_external = bool(
            re.search(
                r"Feign|Client\.|MQ|Rabbit|Kafka|HTTP|远程|外部系统|"
                r"调用.{0,16}(服务|系统|接口)|同步通知|异步|"
                r"代码中未见.{0,12}(外部|远程|Feign|Client|MQ)|"
                r"未在已读材料中看到.{0,12}(外部|Feign|Client|MQ)",
                a,
                re.I,
            )
        )
        has_how_external = bool(
            re.search(
                r"(Feign|Client|#\w+).{0,40}(调用|请求)|"
                r"(通过|经).{0,20}(Feign|Client|MQ|队列|HTTP|接口)|"
                r"(同步|异步).{0,20}(调用|通知|发送)|"
                r"(发送|publish|send).{0,24}(MQ|队列|消息)",
                a,
                re.I,
            )
        ) or ("MQ" in a and re.search(r"队列|发送|send|publish", a, re.I))
        if vague_tail:
            gaps.append("结尾停留在「构造对象/继续处理」，未展开真实下游")
            followups.append(
                "get_symbol_code 读锁定入口对应 ServiceImpl 的实现方法"
            )
            force_fail = True
        if not has_persist:
            gaps.append("未说明何时落库、写哪张表/实体、成功失败标准")
            followups.append("在 ServiceImpl 中查找 Mapper/insert/update/save 并归纳落库条件")
            force_fail = True
        if not has_external:
            gaps.append("未说明外部系统如何协同（Client/Feign/MQ 等）")
            followups.append("搜索并阅读 Feign Client / MQ 发送处，写清调用方式")
            force_fail = True
        elif not has_how_external:
            gaps.append("提到外部协同但未说明具体怎么调用")
            followups.append("读 Client 方法或 MQ 发送代码，补调用方式与关键参数")
            force_fail = True
    ok = not force_fail
    score = 80 if ok else max(25, 70 - 15 * len(gaps))
    summary = "基本完整" if ok else "关键链路细节不足（落库/外部调用/下游展开）"
    if known_hits or invented:
        summary = "回答含未在证据中出现的表名，存在误导风险"
    return {
        "ok": ok,
        "score": score,
        "summary": summary,
        "gaps": gaps,
        "followups": followups,
        "force_fail": force_fail,
    }


def evidence_from_retrieved(retrieved: list[dict[str, Any]], *, limit: int = 6) -> str:
    parts: list[str] = []
    for h in (retrieved or [])[:limit]:
        if not h:
            continue
        fq = h.get("fqcn") or ""
        meth = h.get("method_name") or ""
        code = truncate(h.get("raw_code") or "", 400)
        if code:
            parts.append(f"- {fq}#{meth}\n{code}")
        elif h.get("summary"):
            parts.append(f"- {fq}#{meth}: {truncate(str(h.get('summary')), 200)}")
    return "\n".join(parts)
