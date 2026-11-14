"""Structural codebase survey — no product-domain nouns."""
from __future__ import annotations

from pathlib import Path

from app.agent.codebase_survey import (
    extract_probed_type_names,
    find_aspects,
    find_custom_annotations,
    find_exception_advices,
    find_feign_clients,
    find_lock_constants,
    find_message_listeners,
    find_spring_boot_applications,
    find_xxl_jobs,
    probe_type_in_sources,
    run_architecture_survey,
    run_symbol_probe,
    wants_codebase_architecture,
    wants_symbol_existence_probe,
    wants_topic_existence_ask,
)
from app.agent.pev_agent import heuristic_goal, wants_repo_file_inventory
from app.agent.project_notebook import (
    build_and_save_notebook,
    ensure_notebook,
    load_notebook,
    notebook_brief,
    notebook_path,
)


def test_wants_architecture_not_repo_files():
    q = "项目中用到了哪些技术"
    assert wants_codebase_architecture(q)
    assert not wants_repo_file_inventory(q)
    assert wants_codebase_architecture(q)
    g = heuristic_goal(q)
    assert g["goal"] == "general"  # LLM owns inventory routing


def test_topic_existence_not_architecture_notebook():
    """Concrete「有没有某话题」→ topic_probe; project-wide overview stays notebook."""
    q = "有结清证明业务吗"
    assert wants_topic_existence_ask(q)
    assert not wants_codebase_architecture(q)
    g = heuristic_goal(q)
    assert g["goal"] == "general"
    assert wants_topic_existence_ask(q)
    # Project-wide capability inventory still uses notebook.
    wide = "项目都有哪些业务"
    assert not wants_topic_existence_ask(wide)
    assert wants_codebase_architecture(wide)
    # Framework-shape with 有没有 still architecture.
    assert wants_codebase_architecture("有没有启动类")
    assert not wants_topic_existence_ask("有没有启动类")


def test_topic_existence_phrase_and_probe(monkeypatch):
    from app.agent.codebase_survey import (
        extract_topic_existence_phrase,
        run_topic_existence_probe,
    )

    assert extract_topic_existence_phrase("有结清证明业务吗") == "结清证明"
    assert extract_topic_existence_phrase("有没有发票开具功能") == "发票开具"
    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.is_dir():
        return
    monkeypatch.setattr(
        "app.agent.codebase_survey.resolve_project_root",
        lambda _pid: root,
    )
    out = run_topic_existence_probe("demo-loan", "有结清证明业务吗")
    assert out["ok"]
    assert out["phrase"] == "结清证明"
    assert out["exists"] is True
    assert out["source_hits"] or out["index_hits"]
    primary = out.get("primary_hits") or []
    secondary = out.get("secondary_hits") or []
    # Prefer data/impl before const when both exist.
    if primary and secondary:
        assert all(h.get("role") in {"impl", "data"} for h in primary)
        assert all(h.get("role") not in {"impl", "data"} for h in secondary)
    brief = out.get("brief") or ""
    assert "优先：实现/报文类" in brief
    assert "旁证：配置/常量" in brief
    # First listed primary hit should not be a *Constant class when data hits exist.
    if primary:
        assert not str(primary[0].get("class") or "").endswith(
            ("Constant", "Constants")
        ), primary[0]
    # Controller / ApplyService should rank above generic const when present.
    top_names = [str(h.get("class") or "") for h in primary[:5]]
    assert any(
        n.endswith("Controller") or "ApplyService" in n or n.endswith("VO")
        for n in top_names
    ), top_names
    if secondary:
        assert any(
            str(h.get("class") or "").endswith(("Constant", "Constants", "Config"))
            or h.get("role") in {"const", "config"}
            for h in secondary
        )


def test_project_overview_not_explain_flow():
    q = "先描述一下项目demo-loan"
    assert wants_codebase_architecture(q)
    assert wants_codebase_architecture(q)
    g = heuristic_goal(q)
    assert g["goal"] == "general"  # LLM owns inventory routing
    # Overview + flow wording → do not force architecture.
    assert not wants_codebase_architecture("描述一下项目里某接口流程")


def test_project_notebook_roundtrip(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.is_dir():
        return
    monkeypatch.setattr(
        "app.agent.project_notebook.notebook_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.resolve_project_root",
        lambda _pid: root,
    )
    doc = build_and_save_notebook("demo-loan", root=root)
    assert doc.get("controllers")
    assert doc.get("techs") is not None
    assert notebook_path("demo-loan").is_file()
    loaded = load_notebook("demo-loan")
    assert loaded and loaded.get("built_at")
    brief = notebook_brief(loaded)
    assert "Controller" in brief or "启动类" in brief
    # Q&A path reads notebook (not question-classified templates).
    out = run_architecture_survey("demo-loan", "先描述一下项目")
    assert out["ok"]
    assert out["evidence"].get("mode") == "project_notebook"
    again = ensure_notebook("demo-loan", root=root, rebuild=False)
    assert again.get("built_at") == loaded.get("built_at")


def test_open_ended_asks_share_notebook_evidence(monkeypatch, tmp_path):
    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.is_dir():
        return
    monkeypatch.setattr(
        "app.agent.project_notebook.notebook_dir",
        lambda: tmp_path,
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.resolve_project_root",
        lambda _pid: root,
    )
    build_and_save_notebook("demo-loan", root=root)
    briefs = []
    for q in (
        "先描述一下项目",
        "项目中用到了哪些技术",
        "启动类是哪个",
    ):
        out = run_architecture_survey("demo-loan", q)
        assert out["ok"], q
        briefs.append(out.get("brief") or "")
        assert out["evidence"].get("mode") == "project_notebook", q
    # Same notebook brief backbone (flexible LLM answer happens in PEV).
    assert all("启动类" in b or "Controller" in b for b in briefs)


def test_wants_startup_and_aspects():
    assert wants_codebase_architecture("整个项目入口启动类是哪个？")
    assert wants_codebase_architecture("项目中用到了哪些切面？")
    assert wants_codebase_architecture("项目中所有自定义注解有哪些？")
    assert wants_codebase_architecture("全局异常处理器在哪个文件")
    g = heuristic_goal("全局异常处理器在哪个文件")
    assert g["goal"] == "general"


def test_wants_framework_shape_inventory():
    for q in (
        "RedisConstants 里所有 LOCK_KEY 常量有哪些？",
        "WithdrawalsController 里 Redisson 锁 key 常量名是什么？",
        "listener 包里有哪些 Listener？",
        "项目中所有 @XxlJob 任务名有哪些？",
        "FeignClient 有哪些？",
        "@Transactional 用在哪些 Service 方法？",
        "IdempotentConfig 绑定了哪些配置项？",
    ):
        assert wants_codebase_architecture(q), q
        g = heuristic_goal(q)
        assert g["goal"] == "general", q


def test_symbol_probe_trap_question():
    q = "LoanIllegalQuotaService 这个类实现了什么逻辑？"
    assert wants_symbol_existence_probe(q)
    assert "LoanIllegalQuotaService" in extract_probed_type_names(q)
    g = heuristic_goal(q)
    assert g["goal"] == "general"


def test_symbol_probe_listener_consume():
    q = "PhantomLoanListener 消费什么？"
    assert wants_symbol_existence_probe(q)
    g = heuristic_goal(q)
    assert g["goal"] == "general"


def test_scan_real_project_root():
    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.is_dir():
        return
    apps = find_spring_boot_applications(root)
    assert any(a["class"] == "Application" for a in apps)
    aspects = find_aspects(root)
    assert len(aspects) >= 1
    anns = find_custom_annotations(root)
    names = {a["name"] for a in anns}
    assert "LogAuto" in names or "LogAutoController" in names
    advices = find_exception_advices(root)
    assert any("ExceptionHandler" in a["class"] or "Handler" in a["class"] for a in advices)
    locks = find_lock_constants(root)
    assert any("LOCK" in c["name"].upper() for c in locks)
    assert find_message_listeners(root)
    assert find_xxl_jobs(root)
    assert find_feign_clients(root)


def test_run_architecture_survey_smoke(monkeypatch):
    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.is_dir():
        return
    monkeypatch.setattr(
        "app.agent.codebase_survey.resolve_project_root",
        lambda _pid: root,
    )
    out = run_architecture_survey("demo-loan", "启动类是哪个？用了哪些切面？")
    assert out["ok"]
    assert "Application" in out["answer"]
    assert "@Aspect" in out["answer"] or "切面" in out["answer"]
    lock_out = run_architecture_survey("demo-loan", "RedisConstants 里所有 LOCK_KEY 常量有哪些？")
    assert lock_out["ok"]
    assert "LOCK" in lock_out["answer"].upper()
    job_out = run_architecture_survey("demo-loan", "项目中所有 @XxlJob 任务名有哪些？")
    assert job_out["ok"]
    assert "@XxlJob" in job_out["answer"] or "XxlJob" in job_out["answer"]


def test_run_symbol_probe_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.agent.codebase_survey.resolve_project_root",
        lambda _pid: tmp_path,
    )
    monkeypatch.setattr(
        "app.agent.codebase_survey.probe_types_in_index",
        lambda *_a, **_k: [],
    )
    out = run_symbol_probe("demo", "PhantomWidgetService 实现了什么逻辑？")
    assert out["ok"]
    assert "未找到" in out["answer"]
    assert out["evidence"].get("all_missing") is True


def test_probe_type_exists(tmp_path):
    java = tmp_path / "src" / "main" / "java" / "demo"
    java.mkdir(parents=True)
    (java / "DemoUtil.java").write_text(
        "package demo;\npublic class DemoUtil {\n}\n",
        encoding="utf-8",
    )
    hit = probe_type_in_sources(tmp_path, "DemoUtil")
    assert hit["exists"] is True


def test_extract_flow_topic_phrase_keeps_mid_sentence_interface():
    """句中「接口」是业务名一部分，只有句尾泛词「接口」才删除。

    回归 desktop-chat-fb2229929b4f：「结清证明申请接口 主要流程是什么」
    若全局删「接口」→「结清证明申请」，probe 命中 RepayController；
    保留句中「接口」→「结清证明申请接口」，probe 命中 SignatureController。
    """
    from app.agent.codebase_survey import extract_flow_topic_phrase

    # 核心修复：句尾接口+主要流程 → 干净 topic，保留句中「申请」
    phrase = extract_flow_topic_phrase("结清证明申请接口 主要流程是什么")
    assert phrase == "结清证明申请", phrase
    assert "接口" not in phrase and "流程" not in phrase and "主要" not in phrase
