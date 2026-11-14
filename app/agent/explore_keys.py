"""Question-intent keyword / pattern constants extracted from explore_agent.py.

Pure data — no functions, no imports besides ``re``. Kept in one module so the
intent classifiers (explore_intent) and answer assemblers can share the same
vocabulary without coupling to the agent loop.
"""
from __future__ import annotations

import re

# 「技术细节 / 类名方法名」类问题 —— 回答才带 Class#method 与可核对事实。
_TECH_DETAIL_KEYS = (
    "代码",
    "类名",
    "方法名",
    "源码",
    "错误码",
    "异常码",
    "异常信息",
    "接口路径",
    "http路径",
    "HTTP路径",
    "实现细节",
    "调用链",
    "fqcn",
    "哪个类",
    "哪个方法",
    "技术细节",
)
_ENDPOINT_KEYS = (
    "地址",
    "服务地址",
    "哪个服务",
    "什么服务",
    "服务名",
    "url",
    "URL",
    "Uri",
    "uri",
    "接口地址",
    "请求地址",
    "调用地址",
    "Feign",
    "feign",
    "path",
    "Path",
    "发下",
    "服务 id",
    "serviceId",
)

# 「xx相关的接口有几个 / 有哪些接口」— list Controller endpoints, not a flow essay.
_API_INVENTORY_KEYS = (
    "相关的接口",
    "相关接口",
    "接口有几个",
    "有几个接口",
    "有哪些接口",
    "哪些接口",
    "几个接口",
    "接口列表",
    "接口有哪些",
    "多少个接口",
    "接口都有哪些",
    "都有哪些接口",
    # English inventory asks (case-insensitive key list).
    "which apis",
    "what apis",
    "api list",
    "list of apis",
    "apis are there",
    "available apis",
    "endpoint list",
    "list of endpoints",
    "endpoints are there",
    "which interfaces",
    "interfaces are there",
)

# 「主链/过程中调用了哪些接口」— downstream callees of a locked entry, NOT sibling Controllers.
_FLOW_DOWNSTREAM_API_KEYS = (
    "调用了哪些",
    "调了哪些",
    "掉了哪些",  # common typo for 调了
    "都调用了",
    "过程中调",
    "过程中调用",
    "回答过程中",
    "下游接口",
    "下游服务",
    "下游调用",
    "主链调用",
    "流程中调用",
    "流程里调用",
    "链路上调用",
    "调用了哪些接口",
    "调用了哪些服务",
    "调了哪些接口",
    "调了哪些服务",
)

_API_INVENTORY_STOP = (
    "帮我看看",
    "帮我看下",
    "帮我看",
    "我想看看",
    "我想看",
    "请看一下",
    "看一下",
    "看看",
    "相关的",
    "相关",
    # 「都有哪些」必须先于「有哪些」被整体剥离，否则「有」被先吃掉只剩
    # 一个孤立「都」（desktop-chat-02ce27303f59: 「结清证明申请 都」）。
    "都有哪些",
    "有哪些",
    "有几个",
    "哪些",
    "几个",
    "多少个",
    "列表",
    "接口",
    "一下",
    "都有",
    "源码",
    "源代码",
    "代码",
    "controller",
    "Controller",
    "controoler",
)

# English inventory glue, stripped **whole-word only** (word-boundary, regex).
# Never substring: "in" must not damage "include", "the" must not damage
# "other", "is" must not damage "this". Each entry is a standalone token.
_API_INVENTORY_STOP_EN = frozenset(
    {
        "which",
        "what",
        "list",
        "apis",
        "api",
        "endpoint",
        "endpoints",
        "interface",
        "interfaces",
        "available",
        "exist",
        "exists",
        "are",
        "is",
        "there",
        "the",
        "of",
        "for",
        "to",
        "me",
        "please",
        "any",
        "all",
        "with",
        "does",
        "do",
        "this",
        "that",
    }
)

_CONTROLLER_HELPER_METHOD_RE = re.compile(
    r"(?i)^(check\w*|validate\w*|generate\w*|insert\w*|update\w*|applyresult\w*|"
    r".*dealwith$|after\w*|before\w*|build\w*|convert\w*|to\w*vo$|from\w*)"
)

_VALIDATION_KEYS = ("校验", "字段", "参数", "入参", "必填", "哪些字段", "校验了")

# Business-gate theme/method tables removed — dig-in uses session + CALLS evidence.
_GATE_DETAIL_THEME_KEYS: tuple[str, ...] = ()
_GATE_DETAIL_DIG_IN: tuple[str, ...] = ()
_GATE_DETAIL_METHOD_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])[A-Za-z][A-Za-z0-9_]{2,80}(?![A-Za-z0-9_])"
)
# Structural CN→EN method stems removed — dig-in / ranking uses evidence + LLM.
_GATE_THEME_METHOD_STEMS: dict[str, tuple[str, ...]] = {}
_GATE_PRIMARY_METHODS: tuple[str, ...] = ()
# Anti-hallucination: only flag tables absent from evidence (no product denylist).
_KNOWN_FAKE_TABLES: tuple[str, ...] = ()
_HOLLOW_TABLE_PHRASE_RE = re.compile(
    r"查询[^，。\n]{0,24}表|读取[^，。\n]{0,24}表|写入[^，。\n]{0,16}表"
)
_FAKE_IMPL_DISCLAIMER_RE = re.compile(
    r"(?m)^[ \t]*[-*•]?\s*"
    r"(?:当前)?(?:未获取到|未读到|未能获取|没有获取到|未查看)"
    r".{0,40}(?:具体)?(?:实现)?(?:代码|源码).{0,80}$"
)
_REVIEW_FOOTER_RE = re.compile(
    r"\n*[（(](?:补充说明|评审提示)[：:].*?[）)]\s*$",
    re.S,
)
_TABLE_NAME_RE = re.compile(r"(?<![A-Za-z0-9_])(t_[a-z][a-z0-9_]{2,40})(?![A-Za-z0-9_])")
# Explicit full-chain asks — includes colloquial「从入口到…串一下」, not bare「xx流程」.
_FULL_FLOW_KEYS = (
    "完整",
    "全链路",
    "端到端",
    "全流程",
    "端到端流程",
    "完整流程",
    "从入口到",
    "串一下",
    "串起来",
    "串一串",
    "走一遍",
    "走一趟",
    "整条链",
    "整条流程",
    "下游回调",
)

_CONCRETE_FLOW_MARKERS = re.compile(
    # Framework / structure facts only — no industry domain nouns.
    r"校验|状态|开关|落库|回调|通知|MQ|Listener|通过|失败|必填|枚举|异步|"
    r"Controller|ServiceImpl|update|insert|Mapper",
)

_QUERY_STOP_FOR_FLOW = {
    "流程",
    "什么样",
    "怎么走",
    "怎样",
    "怎么",
    "如何",
    "介绍",
    "一下",
    "什么",
    "哪些",
    "有哪些",
    "调用链",
    "链路",
    "串一下",
    "完整",
    "全链路",
    "端到端",
    "从入口到",
    "下游",
    "过程中",
    "过程",
    "调用",
    "接口",
    "服务",
    "客户端",
    "主链",
    "都",
    "的",
    "是",
    "了",
    "吗",
    "呢",
    "把",
    "和",
    "与",
    "及",
}

# Business noun → Class#method lexicon removed from product code.
# Entry resolution must use session lock, search_symbols, or project config — never
# hardcoded product-noun maps in the agent.
_LEXICAL_MAIN_ENTRY_RULES: tuple[tuple[tuple[str, ...], str], ...] = ()

_RESULT_ENTRY_MARKERS = re.compile(
    r"(?i)callback|callBack|notice|notify|listener|listenner|consumer|回调|通知"
)

_QUERYISH_METHOD = re.compile(
    r"(?i)^(query|get|list|find|select|search|count|sync|check|load|fetch|page)"
)

_CONTROLLER_CM_RE = re.compile(
    r"\b([A-Z][\w.]*?Controller)#([A-Za-z_][\w]*)\b"
)

# Internal correction asides baked into FlowDoc descs — hide in business presentation.
_PEDANTIC_NOTE_RE = re.compile(
    r"(?:^|[。；;])\s*注意[:：].*$|"
    r"(?:^|[。；;])\s*纠正[:：].*$|"
    r"不是\s*updateLoanInfo[^。；;]*[。；;]?|"
    r"也不写主表[。；;]?|"
    r"此步写主表，不是申请表/扩展表[。；;]?|"
    r"源码注释[:：][^。；;]*[。；;]?",
    re.M,
)

# Explicit「把源码贴出来 / …源码是什么」— answer with real method body, not prose.
_SOURCE_PASTE_PHRASES = (
    "源码是什么",
    "源代码是什么",
    "实现代码是什么",
    "把代码贴出来",
    "把源码贴出来",
    "贴出代码",
    "贴出源码",
    "贴一下代码",
    "贴一下源码",
    "代码贴出来",
    "源码贴出来",
    "完整源码",
    "方法实现代码",
    "看下源码",
    "看看源码",
    "看下代码",
    "看看代码",
    "对应的代码",
    "对应代码",
    "发下源码",
    "发一下源码",
    "部分代码",
    "代码什么样",
    "代码是什么",
)
_SOURCE_PASTE_RE = re.compile(
    r"(?i)(?:源码|源代码|实现代码|部分代码|代码)\s*(?:是什么|什么样|怎么样|如何|在哪|给我)|"
    r"(?:把|将)?(?:代码|源码).{0,8}(?:贴|发|给|看)|"
    r"(?:贴|发|看).{0,12}(?:代码|源码)|"
    r"我想看.{0,8}(?:代码|源码)"
)

_SOURCE_PASTE_STOP_TOKENS = frozenset(
    {
        "controller",
        "service",
        "serviceimpl",
        "impl",
        "class",
        "method",
        "java",
        "code",
        "source",
        "api",
        "http",
        "rest",
        "mapper",
        "client",
        "feign",
        "listener",
        "job",
        "controoler",
        "controler",
        "contoller",
    }
)
