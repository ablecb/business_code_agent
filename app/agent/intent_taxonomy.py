"""Top-level chat intents + session phase helpers.

Intent = capability boundary (ask / plan / modify / execute).
Flags want_apply / want_verify refine modify+execute; they are not intents.
Legacy labels normalize for backward compatibility.
"""
from __future__ import annotations

import re
from typing import Any, Literal

Intent = Literal[
    "meta",
    "ask",
    "plan",
    "modify",
    "execute",
]

Phase = Literal[
    "idle",
    "analyzing",
    "awaiting_option",
    "implementing",
    "verifying",
    "repairing",
    "done",
    "failed",
]

VALID_INTENTS: frozenset[str] = frozenset(
    {"meta", "ask", "plan", "modify", "execute"}
)
VALID_PHASES: frozenset[str] = frozenset(
    {
        "idle",
        "analyzing",
        "awaiting_option",
        "implementing",
        "verifying",
        "repairing",
        "done",
        "failed",
    }
)
VALID_ACTIONS: frozenset[str] = frozenset(
    {"select_option", "verify", "rollback"}
)

# Legacy → converged taxonomy
_LEGACY_MAP: dict[str, str] = {
    "ask_flow": "ask",
    "general": "ask",
    "impact": "plan",
    "options": "plan",
    "implement": "modify",
    "modify_field": "modify",
    "new_api": "modify",
    "verify": "execute",
    "repair": "execute",
}

INTENT_PROMPT = """你是代码助手的意图识别器。根据用户输入与历史输出严格 JSON：
{"intent":"meta|ask|plan|modify|execute","capability":"general|code_qa|modify|verify|meta","keywords":["..."],"entry_hint":"类名或关键词或空","detail":"简要复述需求","need_options":false,"want_apply":false,"want_verify":false}

capability（走哪条处理管线，按语义判断，不按关键词表）：
- general: 非代码域问答（通用知识/闲聊/语言无关概念）→ 直接回答，不查项目代码
- code_qa: 问项目代码（流程/字段/入口/调用链/表关联/报错/入口在哪）→ 查代码后回答
- modify: 要改现有代码（改/加字段/优化/重构/落库/替换）→ 修改流程
- verify: 编译/跑测/修编译错误 → 验证流程
- meta: 只谈对话本身
示例：放款流程什么样→code_qa；放款表和优惠券表怎么关联→code_qa；
    把CouponInfo加remark字段→modify；
    什么是数据库事务→general；用Java写个快速排序→general（非本项目代码）；
    编译一下→verify；我刚才问你什么了→meta。

intent 含义（按能力边界，不要按业务域名分类）：
- meta: 仅当用户明确问「我刚才问你什么 / 再说一遍 / 你听到了什么」——只谈对话本身
- ask: 理解项目/代码（流程、校验、字段含义、调用链、入口、日志反查、清单盘点等）——只读问答
- plan: 改动点/影响面/改动清单/多方案对比/要什么脚本——只读分析，不写盘
- modify: 明确要求直接改代码/落地补丁/按某方案实现（含新增接口）
- execute: 编译、跑测、调试当前改动、按报错修编译（跑命令侧）

信号字段（不是 intent）：
- need_options: 用户要多方案对比时 true（intent 仍为 plan）
- want_apply: 用户要求立刻写文件/落地时 true（通常配 modify）
- want_verify: 用户要求改完还要编译测试/自测时 true（modify 之后走 execute）

keywords：原样保留用户提到的领域/对象词；不要臆造。
entry_hint：仅填用户本轮明确点到的类#方法、Controller 名或接口短语；换话题/换接口时必须留空或只填本题对象。
禁止把「锁定入口 / 上一轮 Class#method」原样抄进 entry_hint（会话是否追问由服务端另判）。
注意：「那这里面…校验了哪些字段」是 ask，不是 meta。
「现在再回答一下 / 再回答一下 / 重新回答一下（承接上一问，无新内容）」→ ask，
capability 跟随上一问（代码域=code_qa，通用=general）。这类纯重问由服务端解析回
上一问重新作答，不是 meta（不要只复述问题、不要拒绝「需要查询项目代码」）。
English equivalents (answer again / re-answer / answer it / please answer /
answer this / now answer / give me the answer — without new code content) → ask
too; capability follows the previous question. English 再说一遍/重复一下
(say it again / repeat it / repeat that) are still meta (conversation recall only).
「我刚才问你什么 / 再说一遍 / 你听到了什么」才是 meta（只谈对话本身）。
「我想某接口增加一个字段并落库，都要修改哪些」是 plan（先分析改动点）。
「某接口加字段 labelBusiness，保存到业务表，再传给外部系统」是 plan
（陈述改造需求=要改动点，即使没写「改哪些」）。
「给我三种实现方案」是 plan，need_options=true。
「直接改代码 / apply patch / 按方案B实现 / 帮我改并自测」是 modify
（want_apply=true；若还要自测则 want_verify=true）。
「先只给我要改的类/方法清单，不要直接改代码」→ plan，want_apply=false，want_verify=false。
「按方案落地并本机编译」→ modify，want_apply=true，want_verify=true。
「按照你给的代码帮我改下 / 只改代码不用测 / 先不要编译」→ modify，want_apply=true，want_verify=false。
「帮我修改代码并提交/推送」「改完新建分支再 commit/push」→ modify，want_apply=true
（先按清单写文件，本机循环再按完整目标动态编排 git；不要收成只写文件或只做 git 的 meta）。
「帮我提交/commit/推送/push 代码」（仅 git、不再改业务代码）→ meta（或 ask），want_apply=false，want_verify=false
（纯 git 发布由桌面 direct 路由处理）。
「编译一下 / 跑测 / 修编译错误」→ execute。
want_apply 表示本轮是否写文件/落地补丁；只要分析或清单时必须为 false。
want_verify 表示落地后是否编译/自测；用户只要改代码、明确不要测时必须为 false。
仅说「某字段怎么落库/流程什么样」且无「增加/新增字段」改造意图 → ask。

确认改码（必须结合历史，勿靠固定词表）：
- 上一轮助手已给出修改清单，并询问是否要帮忙改代码时：
  · 用户表示同意/可以开始改 → intent=modify，want_apply=true；
    若同时要求编译/自测则 want_verify=true，否则 want_verify=false；
    若同时要求新建分支/提交/推送远程 → 仍 want_apply=true（本机写入后继续编排 git）。
  · 用户表示暂不改/先不用 → intent=meta，want_apply=false（只结束确认，不要再整篇重做影响分析）。
  · 用户继续追问清单细节 → intent=ask 或 plan，want_apply=false。
"""

_IMPACT_KEYS = (
    "改哪些",
    "修改哪些",
    "都要修改",
    "要改什么",
    "哪些类",
    "哪些文件",
    "修改方案",
    "改动点",
    "改动清单",
    "改动哪些",
    "影响面",
    "影响范围",
    "DDL",
    "脚本",
)
_OPTIONS_KEYS = (
    "几种方案",
    "多种方案",
    "多方案",
    "几个方案",
    "对比方案",
    "方案对比",
    "可选方案",
    "给我方案",
    "三种做法",
    "两种做法",
    "几种做法",
    "对比利弊",
    "哪种更好",
)
_IMPLEMENT_KEYS = (
    "直接改代码",
    "落地补丁",
    "apply patch",
    "apply_patch",
    "按方案实现",
    "按这个方案改",
    "开始改代码",
    "帮我改代码",
    "帮我实现",
    "帮我改并",
    "写入代码",
    "应用补丁",
    "落地并验证",
    "改完自测",
    "改完编译",
    "改完验证",
    "并自测",
    "直接实现",
)
_VERIFY_KEYS = (
    "编译一下",
    "帮我编译",
    "跑一下测试",
    "跑测",
    "单元测试",
    "调试一下",
    "验证一下改动",
    "mvn test",
    "maven编译",
    "自测",
    "本机编译",
    "编译验证",
)
_ANALYSIS_ONLY_KEYS = (
    "改哪些",
    "修改哪些",
    "都要修改",
    "要改什么",
    "哪些类",
    "哪些文件",
    "改动点",
    "改动清单",
    "影响面",
    "影响范围",
)
_REPAIR_KEYS = (
    "修一下报错",
    "修编译错误",
    "根据报错修",
    "修复错误",
    "编译失败修",
    "测试失败修",
)
_META_KEYS = (
    "刚才问",
    "我问你什么",
    "问了什么",
    "上一问",
    "上一个问题",
    "再说一遍",
    "重复一下",
    "你听到了什么",
    "总结一下刚才",
    "刚刚说",
    # English conversation-recall phrases (same semantics as the CN ones above:
    # 再说一遍/重复一下 are meta — they talk about the conversation itself, not
    # new code content). "repeat it/react that/say it again" == 重复一下/再说一遍.
    "what did i ask",
    "what did i say",
    "what did you hear",
    "repeat that",
    "repeat it",
    "say it again",
)
_CONTENT_OVERRIDE_KEYS = (
    "流程",
    "校验",
    "字段",
    "参数",
    "入参",
    "必填",
    "链路",
    "怎么走",
    "调用链",
    "哪些检查",
    "走什么",
    "地址",
    "服务",
    "url",
    "URL",
    "Feign",
    "改哪些",
    "落库",
)


def _wants_impact_change_plan(message: str) -> bool:
    """True for developer change asks: add field + persist / outbound / 改哪些."""
    from app.agent.question_intent import wants_add_field_change_plan

    text = message or ""
    if any(k in text for k in _IMPACT_KEYS):
        return True
    return wants_add_field_change_plan(text)


def normalize_intent(raw: str | None, message: str = "") -> str:
    """Map legacy / unknown labels onto ask|plan|modify|execute(+meta).

    Prefer the model's intent when it is already a valid label; do not
    override ask→plan with keyword heuristics.
    """
    intent = (raw or "").strip().lower()
    if intent in _LEGACY_MAP:
        return _LEGACY_MAP[intent]
    if intent in VALID_INTENTS:
        return intent
    # Unknown label only — structural fallback (not keyword→Goal routing).
    text = message or ""
    if any(k in text for k in _REPAIR_KEYS) or any(k in text for k in _VERIFY_KEYS):
        if not _wants_impact_change_plan(text) and not _positive_implement_key(text):
            return "execute"
    if _wants_impact_change_plan(text):
        return "plan"
    if any(k in text for k in ("方案对比", "几种做法", "对比利弊", "三种实现", "多方案")):
        return "plan"
    if _positive_implement_key(text) and not is_analysis_only_ask(text):
        return "modify"
    return "ask"


def wants_meta(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    # Case-insensitive match so English recall phrases ("What did I ask",
    # "Repeat that") work alongside the CN keys (Chinese has no case).
    lower = text.lower()
    if any(k.lower() in lower for k in _CONTENT_OVERRIDE_KEYS):
        return False
    return any(k.lower() in lower for k in _META_KEYS)


def is_analysis_only_ask(message: str) -> bool:
    """True when user only wants a change list / impact, not code writes."""
    return any(k in (message or "") for k in _ANALYSIS_ONLY_KEYS)


def _positive_implement_key(text: str) -> bool:
    """True when an implement key appears without nearby negation (LLM-fallback only)."""
    for key in _IMPLEMENT_KEYS:
        start = 0
        while True:
            idx = text.find(key, start)
            if idx < 0:
                break
            window = text[max(0, idx - 4) : idx]
            if not any(n in window for n in ("不要", "别", "先不", "先别")):
                return True
            start = idx + len(key)
    return False


def wants_implement_and_verify(message: str) -> bool:
    """Explicit 'change code then self-test' asks (heuristic fallback only)."""
    text = message or ""
    if is_analysis_only_ask(text):
        return False
    if _positive_implement_key(text):
        return True
    return any(
        k in text
        for k in ("并验证", "并测试", "并编译", "再编译", "跑通", "自测", "本机验证")
    ) and any(
        k in text for k in ("改", "实现", "落地", "补丁", "加上", "增加", "新增")
    )


def heuristic_intent(message: str) -> dict[str, Any]:
    """Minimal fallback when Intent LLM fails — no keyword→intent cascade."""
    text = message or ""
    keywords = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_]{1,}", text)[:8]
    return {
        "intent": "ask",
        "keywords": keywords,
        "entry_hint": "",
        "detail": text,
        "need_options": False,
        "want_apply": False,
        "want_verify": False,
    }


def parse_option_selection(
    message: str,
    pending_options: list[dict[str, Any]] | None,
) -> str | None:
    """Resolve natural-language option pick → option id, or None."""
    opts = [o for o in (pending_options or []) if isinstance(o, dict) and o.get("id")]
    if not opts:
        return None
    text = (message or "").strip()
    if not text:
        return None

    # Exact / contained id
    for o in opts:
        oid = str(o.get("id") or "")
        if oid and (text == oid or oid in text):
            return oid

    # 方案A / 方案 B / 选项1
    m = re.search(r"(?:方案|选项|option)\s*([A-Za-z]|[0-9一二三四五六七八九十]+)", text, re.I)
    if m:
        token = m.group(1).strip()
        return _token_to_option_id(token, opts)

    # 选2 / 用第一个 / 第2个
    m2 = re.search(
        r"(?:选|用|选择|采用)\s*(?:第)?\s*([A-Za-z]|[0-9一二三四五六七八九十]+)\s*(?:个|种|套)?",
        text,
    )
    if m2:
        return _token_to_option_id(m2.group(1).strip(), opts)

    m3 = re.search(r"第\s*([0-9一二三四五六七八九十]+)\s*个", text)
    if m3:
        return _token_to_option_id(m3.group(1).strip(), opts)

    # Bare letter/number when short
    if re.fullmatch(r"[A-Za-z]", text):
        return _token_to_option_id(text, opts)
    if re.fullmatch(r"[0-9]+", text):
        return _token_to_option_id(text, opts)

    # Title substring match
    for o in opts:
        title = str(o.get("title") or "").strip()
        if title and title in text:
            return str(o["id"])
    return None


_CN_NUM = {
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def _token_to_option_id(token: str, opts: list[dict[str, Any]]) -> str | None:
    if not token or not opts:
        return None
    # Letter index A/B/C
    if re.fullmatch(r"[A-Za-z]", token):
        idx = ord(token.upper()) - ord("A")
        if 0 <= idx < len(opts):
            return str(opts[idx]["id"])
        # match id like "A"
        for o in opts:
            if str(o.get("id") or "").upper() == token.upper():
                return str(o["id"])
    # Digit
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(opts):
            return str(opts[idx]["id"])
    # Chinese numeral
    if token in _CN_NUM:
        idx = _CN_NUM[token] - 1
        if 0 <= idx < len(opts):
            return str(opts[idx]["id"])
    return None


def get_option_by_id(
    pending_options: list[dict[str, Any]] | None, option_id: str | None
) -> dict[str, Any] | None:
    if not option_id:
        return None
    for o in pending_options or []:
        if isinstance(o, dict) and str(o.get("id") or "") == str(option_id):
            return o
    return None


def is_explore_intent(intent: str) -> bool:
    """Read-only paths that use PEV / skills (ask + plan)."""
    return normalize_intent(intent) in {"ask", "plan"}


def is_qa_intent(intent: str) -> bool:
    return normalize_intent(intent) in {"ask", "plan", "meta"}


def is_implement_intent(intent: str) -> bool:
    """True for modify (legacy name kept for callers)."""
    return normalize_intent(intent) == "modify"


def is_execute_intent(intent: str) -> bool:
    return normalize_intent(intent) == "execute"
