from __future__ import annotations

from pathlib import Path

from app.parser.java_parser import (
    extract_publish_queues,
    extract_queue_from_annotation,
    is_accessor_method,
    parse_java_file,
)
from app.parser.models import ParsedAnnotation
from app.parser.scanner import scan_java_files


def test_scan_and_parse_sample(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from app.llm.factory import get_llm_provider
    get_llm_provider.cache_clear()
    from app.core.config import get_settings
    get_settings.cache_clear()

    root = Path(__file__).resolve().parents[1] / "java_project" / "demo-loan"
    if not root.exists():
        return
    files = scan_java_files(root)
    assert len(files) > 0
    # prefer known listener
    target = next((f for f in files if "RiskControlRabbitMQListener" in f.name), files[0])
    result = parse_java_file(target, root)
    assert result.types
    t = result.types[0]
    assert t.fqcn
    assert t.methods


def test_scan_respects_gitignore(tmp_path: Path):
    root = tmp_path
    (root / ".gitignore").write_text("**/ignored/**\nSkipMe.java\n", encoding="utf-8")

    keep = root / "mod" / "src" / "main" / "java" / "a"
    keep.mkdir(parents=True)
    (keep / "Keep.java").write_text("class Keep {}", encoding="utf-8")
    (keep / "SkipMe.java").write_text("class SkipMe {}", encoding="utf-8")

    ignored = root / "mod" / "src" / "main" / "java" / "ignored"
    ignored.mkdir(parents=True)
    (ignored / "Secret.java").write_text("class Secret {}", encoding="utf-8")

    files = scan_java_files(root)
    names = {f.name for f in files}
    assert "Keep.java" in names
    assert "SkipMe.java" not in names
    assert "Secret.java" not in names


def test_scan_unanchored_dir_rule_keeps_deep_business_packages(tmp_path: Path):
    """未锚定目录规则（signature/）不得排除深层同名业务源码包。

    回归 desktop-chat-fb2229929b4f：.gitignore 里 `signature/` 本意排除顶层
    签名文档目录，但 git wildmatch 语义匹配任意层级 →
    src/main/java/.../controller/signature/ 等结清证明源码全部未入库，
    导致「结清证明申请接口」入口解析失败（命中 RepayController/空）。
    锚定为 /signature/ 后深层包恢复收录。
    """
    root = tmp_path
    (root / ".gitignore").write_text("/signature/\n", encoding="utf-8")

    # 顶层签名文档目录（应排除）
    (root / "signature" / "doc").mkdir(parents=True)
    (root / "signature" / "doc" / "gen.txt").write_text("x", encoding="utf-8")

    # 深层业务源码包（必须保留）
    biz = root / "mod" / "src" / "main" / "java" / "com" / "demo" / "controller" / "signature"
    biz.mkdir(parents=True)
    (biz / "SignatureController.java").write_text(
        "package com.demo.controller.signature;\n"
        "public class SignatureController { public void signatureApply() {} }\n",
        encoding="utf-8",
    )

    files = scan_java_files(root)
    names = {f.name for f in files}
    assert "SignatureController.java" in names, names
    assert not any("gen.txt" in str(f) for f in files)


def test_accessor_detect():
    assert is_accessor_method("getName", 0)
    assert is_accessor_method("setName", 1)
    assert is_accessor_method("isActive", 0)
    assert not is_accessor_method("getName", 1)
    assert not is_accessor_method("execute", 0)
    assert not is_accessor_method("get", 0)


def test_parse_mapper_xml(tmp_path: Path):
    from app.parser.mapper_xml_parser import parse_mapper_xml_file

    xml = tmp_path / "src" / "main" / "resources" / "FooMapper.xml"
    xml.parent.mkdir(parents=True)
    xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN"
  "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.demo.FooMapper">
  <!-- find by id -->
  <select id="findById" resultType="com.demo.Foo">
    SELECT * FROM foo WHERE id = #{id}
  </select>
  <insert id="insertFoo">
    INSERT INTO foo(name) VALUES(#{name})
  </insert>
</mapper>
""",
        encoding="utf-8",
    )
    result = parse_mapper_xml_file(xml, tmp_path)
    assert result.types
    t = result.types[0]
    assert t.symbol_kind == "MapperXml"
    assert t.fqcn == "com.demo.FooMapper"
    names = {m.name for m in t.methods}
    assert "findById" in names
    assert "insertFoo" in names
    find = next(m for m in t.methods if m.name == "findById")
    assert "<select" in find.raw_code
    assert "find by id" in find.raw_code


def test_queue_placeholder():
    ann = ParsedAnnotation(
        name="RabbitListener",
        values={"queues": "${loan.listener.queue.riskResult:rce.to.loan.risk.result}"},
    )
    qs = extract_queue_from_annotation(ann)
    assert qs == ["rce.to.loan.risk.result"]


def test_publish_extract():
    body = 'rabbitTemplate.convertAndSend("q.demo.loan.withdrawals.create_order", msg);'
    qs = extract_publish_queues(body)
    assert "q.demo.loan.withdrawals.create_order" in qs
