"""通用问答兜底：让系统能回答「所有问题」，而不只是代码域问题。

内网 qwen3-32b 在线长链工具推理会掉准确率；对非代码域问题（知识问答、闲聊、
通用技术概念等）根本不该进 explore 工具循环 —— 直接给 LLM 一次性回答即可。

判定：`is_general_chat_question` 用「代码/项目域锚点」负向过滤 —— 只要问题里
出现项目域关键词或代码标识符就按代码问题处理（走现有检索/explore 管线），
否则按通用问题直答。判定保守：宁可多走一次现有管线，也不把代码问题误判成闲聊。
"""

from __future__ import annotations

import re
from typing import Any

from app.core.utils import truncate
from app.llm.base import LLMProvider

# 项目/代码域锚点（出现任一即视为代码域问题，不直答）
# 注意：只放"本项目代码"相关的词；通用计算机概念（事务/锁/队列/缓存/线程/排序等）
# 不得放进来，否则「什么是数据库事务」这类问题会被错误路由进 explore/PEV。
_CODE_ANCHOR_RE = re.compile(
    r"(?:"
    r"流程|接口|入口|方法|类|代码|报错|异常|日志|字段|落库|入库|持久化|"
    r"表|查询|调用|链路|校验|参数|入参|出参|返回值|改造|修改|改码|方案|"
    r"加字段|新增|地址|监听|回调|异步|同步|网关|依赖|编译|构建|部署|"
    r"单测|提交|推送|分支|回滚|"
    r"Controller|Service|Mapper|Feign|Client|Entity|DTO|VO|Util|Job|Listener|"
    r"Handler|Processor|Config|启动|注册|配置|环境|版本|"
    r"java|spring|mybatis|sql|redis|mq|nacos|xxl|docker|maven|git"
    r")"
)

# 通用直答的强信号：明确要求"用某语言写一段代码/算法/脚本"（非本项目接口）。
# 例如「用 Java 写一个快速排序」「写一段 Python 爬虫」→ 直接回答，不进 PEV。
_GENERAL_WRITE_RE = re.compile(
    r"(?:"
    r"用\s*(?:Java|Python|Go|Golang|SQL|C\+\+|C#|JavaScript|TypeScript|PHP|Rust|Shell|Bash|Kotlin|Scala)\s*写"
    r"|"
    r"写(?:一个|一段|个|份)?\s*.{0,10}?(?:程序|函数|方法|算法|排序|脚本|爬虫|正则|查询语句|SQL语句|类|工具)"
    r")"
)

# 触发 execute/verify 的强信号（否则 intent=execute 会被降级为 ask，防止
# 「写个快速排序」这类问题误入编译验证循环）
_VERIFY_ACTION_RE = re.compile(
    r"(?:编译|验证|自测|跑一|运行|执行|测试|检查代码|单测|构建|启动服务|"
    r"verify|compile|test|build|run|mvn|gradle)"
)

# 代码标识符形态：XxxController / XxxService#method / com.xxx.xxx / Foo.java / #method
_CODE_IDENTIFIER_RE = re.compile(
    r"(?:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:Controller|Service|Mapper|Client|Util|DTO|VO|Entity|"
    r"Config|Job|Listener|Handler|Processor|Factory|Repository|Feign)\b"
    r"|"
    r"[a-z]+\.[a-z]+\.[a-zA-Z0-9_.]+"  # fqcn 形态
    r"|"
    r"\b\w+\.java\b"
    r"|"
    r"#\w+\([^)]*\)"  # Class#method( 形态
    r")"
)

GENERAL_SYSTEM = (
    "你是部署在内网的企业智能助手。请用中文清晰、有条理地回答用户的问题。"
    "回答要求：1) 先给结论再展开；2) 不确定的内容明确说明，禁止编造事实；"
    "3) 若用户问题涉及本公司的 Java 业务系统而你不知道具体实现，"
    "请说明「这需要查询项目代码，可以换个更具体的问法」，不要臆造类名、表名、"
    "接口；4) 编程问题给出简洁、可运行的示例；5) 普通知识问答/闲聊正常回答。"
)


def is_general_chat_question(message: str) -> bool:
    """True = 看起来不是代码/项目域问题，可走通用直答（不触发 explore 循环）。"""
    msg = (message or "").strip()
    if not msg:
        return True
    # 强通用信号：用某语言写代码/算法/脚本（非本项目接口）
    if _GENERAL_WRITE_RE.search(msg):
        return True
    if _CODE_IDENTIFIER_RE.search(msg):
        return False
    if _CODE_ANCHOR_RE.search(msg):
        return False
    return True


def wants_verify_action(message: str) -> bool:
    """True = 用户明确要求编译/验证/运行（intent=execute 仅在此为真时保留）。"""
    return bool(_VERIFY_ACTION_RE.search(message or ""))


def format_history_text(
    history: Any,
    *,
    max_messages: int = 8,
    per_line: int = 300,
) -> str:
    """会话记忆 → 'role: content' 文本，供通用问答/普通问答注入上下文。

    - 传入 str：原样返回（兼容既有调用方）。
    - 传入消息列表（session_store.list_messages 结构）：取最近 max_messages 条，
      每条保留前 per_line 字符，保证多轮上下文在 token 预算内。
    """
    if isinstance(history, str):
        return history or "(空)"
    if not history:
        return "(空)"
    lines: list[str] = []
    for m in list(history)[-max_messages:]:
        role = str(m.get("role") or "user")
        content = str(m.get("content") or "")
        lines.append(f"{role}: {truncate(content, per_line)}")
    return "\n".join(lines)


async def general_chat_reply(
    message: str,
    history_text: str | list[dict[str, Any]] = "",
    llm: LLMProvider | None = None,
    *,
    max_tokens: int = 2000,
) -> str:
    """通用兜底 / 未选择项目时的普通问答：不依赖检索，直接把问题交给 LLM 回答。

    带会话上下文记忆：history_text 为 str 或消息列表（自动格式化为
    'role: content' 行，取最近若干条）。
    """
    from app.llm.factory import get_llm_provider

    provider = llm or get_llm_provider()
    history_text = format_history_text(history_text)
    content = await provider.chat(
        [
            {"role": "system", "content": GENERAL_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"历史对话:\n{history_text or '(空)'}\n\n用户问题:\n{message}"
                ),
            },
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return (content or "").strip()
