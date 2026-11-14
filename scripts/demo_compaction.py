"""Demo: compaction vs hard-truncation on a long conversation (real LLM).

Shows what semantic compaction preserves vs. what _trim_messages drops.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.agent.compaction import compact_messages, should_compact
from app.agent.explore_agent import _trim_messages
from app.llm.factory import get_llm_provider

# A realistic 16-message conversation about the loan flow (8 turns).
MSGS = [
    {"role": "system", "content": "你是面向业务同学的代码流程解读助手（底层是 Java/Spring）。"},
    {"role": "user", "content": "放款流程是什么样"},
    {"role": "assistant", "content": "入口 WithdrawalsController#withdrawals，主链：校验订单 → 组装落库 → 回调。"},
    {"role": "user", "content": "校验时都校验哪些字段"},
    {"role": "assistant", "content": "checkApplyReqParam 校验：订单号、金额、客户号、渠道号、产品号非空。"},
    {"role": "user", "content": "落库写哪张表"},
    {"role": "assistant", "content": "写入 WithdrawalsExt 扩展表，@TableName 对应 t_withdrawals_ext。"},
    {"role": "user", "content": "回调是同步还是异步"},
    {"role": "assistant", "content": "异步，RabbitMQ 发送放款结果通知，Listener 消费后更新状态。"},
    {"role": "user", "content": "放款失败会怎样"},
    {"role": "assistant", "content": "失败时记录失败原因，不落主表，返回错误码 4001。"},
    {"role": "user", "content": "提现和放款是同一个入口吗"},
    {"role": "assistant", "content": "不同：放款入口是 WithdrawalsController#withdrawals，提现走单独流程。"},
    {"role": "user", "content": "放款结果怎么通知客户"},
    {"role": "assistant", "content": "通过短信网关发通知，模板在 SMSConstants，含成功/失败两种模板。"},
]


def est_tokens(msgs: list[dict]) -> int:
    return sum(len(str(m.get("content") or "")) // 2 for m in msgs)


async def main() -> None:
    print(f"messages={len(MSGS)} should_compact={should_compact(MSGS)}")
    print(f"est chars->tokens(before) ≈ {est_tokens(MSGS)}")
    print("\n--- hard truncation (_trim_messages) keeps ---")
    truncated = [dict(m) for m in MSGS]
    _trim_messages(truncated)
    print(f"  messages: {len(MSGS)} -> {len(truncated)}")
    print(f"  dropped middle turns contain facts like: 落库表名 t_withdrawals_ext、错误码 4001、短信模板")
    print(f"  kept head: {truncated[1]['content'][:24]}…")
    print(f"  kept tail: {truncated[-1]['content'][:24]}…")
    print(f"  est tokens(after) ≈ {est_tokens(truncated)}")
    print(f"  → 中间轮的关键事实(表名/错误码/模板)全部丢失")

    llm = get_llm_provider()
    print("\n--- semantic compaction (LLM summary) ---")
    compacted = await compact_messages(llm, MSGS, keep_tail=6)
    if not compacted:
        print("  compaction returned None (LLM failure?)")
        return
    print(f"  messages: {len(MSGS)} -> {len(compacted)}")
    print(f"  est tokens(after) ≈ {est_tokens(compacted)}")
    note = next(m for m in compacted if m.get("role") == "system" and "摘要" in str(m.get("content") or ""))
    print(f"  summary note ({len(note['content'])} chars):")
    print(f"    {note['content']}")
    print(f"  tail verbatim: {compacted[-1]['content'][:30]}… / {compacted[-2]['content'][:30]}…")


if __name__ == "__main__":
    asyncio.run(main())
