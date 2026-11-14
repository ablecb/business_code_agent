"""Log/stack diagnose helpers (no DB required)."""
from pathlib import Path

from app.agent.runtime.diagnose import (
    extract_log_paths,
    ingest_signals,
    parse_stack,
    read_log_file,
    render_diagnose_answer,
)


SAMPLE_STACK = """
2026-08-13 09:01:02 ERROR c.loan.service.WithdrawalsServiceImpl - quota failed
java.lang.IllegalStateException: org quota exceeded
	at com.demo.loan.service.impl.OrgQuotaChecker.check(OrgQuotaChecker.java:88)
	at com.demo.loan.service.impl.WithdrawalsServiceImpl.quotaCheck(WithdrawalsServiceImpl.java:412)
	at com.demo.loan.controller.WithdrawalsController.withdrawals(WithdrawalsController.java:90)
	at java.lang.Thread.run(Thread.java:840)
Caused by: com.demo.loan.exception.BizException: QUOTA_LIMIT
	at com.demo.loan.service.impl.OrgQuotaChecker.fail(OrgQuotaChecker.java:101)
	... 12 more
"""


def test_parse_stack_extracts_app_frames_and_caused_by():
    parsed = parse_stack(SAMPLE_STACK)
    cms = parsed["class_methods"]
    assert "OrgQuotaChecker#check" in cms
    assert "WithdrawalsServiceImpl#quotaCheck" in cms
    assert "WithdrawalsController#withdrawals" in cms
    assert not any(x.startswith("Thread#") for x in cms)
    types = [e["type"] for e in parsed["exceptions"]]
    assert any("BizException" in t for t in types)
    frames = parsed["frames"]
    assert any(f.get("line") == 88 for f in frames)


def test_extract_log_paths_windows_and_label():
    q = r"帮我看日志路径 D:\logs\app.log 里哪块有问题"
    paths = extract_log_paths(q)
    assert any(p.endswith("app.log") for p in paths)
    q2 = r"日志路径是 C:\temp\error.txt"
    assert extract_log_paths(q2)[0].endswith("error.txt")


def test_read_log_file_tail(tmp_path: Path):
    p = tmp_path / "biz.log"
    p.write_text("hello\n" + SAMPLE_STACK, encoding="utf-8")
    text = read_log_file(str(p))
    assert "OrgQuotaChecker" in text
    missing = read_log_file(str(tmp_path / "nope.log"))
    assert missing == ""


def test_ingest_and_render_without_code():
    evidence: dict = {}
    ingest_signals("看这段堆栈哪块有问题\n" + SAMPLE_STACK, evidence)
    assert "OrgQuotaChecker#check" in (evidence.get("signals") or [])
    evidence["suspects"] = evidence.get("stack_cms") or evidence.get("signals")
    evidence["suspect_snippets"] = []
    ans = render_diagnose_answer("哪块有问题", evidence)
    assert "OrgQuotaChecker#check" in ans
    assert "异常链" in ans
    assert "BizException" in ans


def test_render_refuses_when_empty():
    ans = render_diagnose_answer("随便问问", {})
    assert "证据不足" in ans


def test_pev_parse_signal_uses_runtime():
    import asyncio

    from app.agent.pev_agent import _exec_parse_signal

    ev: dict = {}
    asyncio.run(_exec_parse_signal("看这段堆栈哪块有问题\n" + SAMPLE_STACK, ev))
    assert "OrgQuotaChecker#check" in (ev.get("signals") or [])
    assert (ev.get("stack") or {}).get("has_stack")
