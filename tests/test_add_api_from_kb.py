"""PEV add_api — new HTTP API from knowledge-base specs."""
from __future__ import annotations

import asyncio


def test_heuristic_does_not_keyword_gate_add_api():
    """New API / new method / new class is LLM-decided — no Chinese keyword gate."""
    from app.agent.pev_agent import build_task_plan, heuristic_goal
    from app.knowledge.add_api import wants_add_api

    q = "帮我新增一个接口 这个接口文档是知识库的登录接口文档"
    assert wants_add_api(q) is False
    g = heuristic_goal(q)
    assert g["goal"] != "add_api"
    # Execution path still works when LLM (or caller) sets goal=add_api.
    tools = [t["tool"] for t in build_task_plan({"goal": "add_api"})]
    assert tools == ["load_kb_spec", "find_api_style", "render_new_api"]


def test_extract_goal_llm_chooses_add_api(monkeypatch):
    from app.agent import pev_agent as pev
    from app.llm.base import LLMProvider

    q = "帮我新增一个接口 这个接口文档是知识库的登录接口文档"

    class _FakeLLM(LLMProvider):
        async def chat(self, messages, **kwargs):  # noqa: ARG002
            return (
                '{"goal":"add_api","must":['
                '{"id":"kb_spec","desc":"读知识库"},'
                '{"id":"style","desc":"对齐 Controller"},'
                '{"id":"scaffold","desc":"输出清单与代码"}],'
                '"field_name":null,"field_label_zh":null,'
                '"tables":[],"inventory_kind":null,'
                '"deliverable":"新增接口清单与建议代码"}'
            )

        async def embed(self, texts, **kwargs):  # noqa: ARG002
            return [[0.0] * 8 for _ in texts]

    monkeypatch.setattr("app.llm.factory.get_llm_provider", lambda: _FakeLLM())
    out = asyncio.run(pev.extract_goal_contract(q))
    assert out["goal"] == "add_api"
    assert out.get("source") == "llm"
    tools = [t["tool"] for t in pev.build_task_plan(out)]
    assert tools == ["load_kb_spec", "find_api_style", "render_new_api"]


def test_parse_login_spec_and_render():
    from app.knowledge.add_api import parse_api_spec, render_new_api_plan

    text = """
# API接口文档
请求地址：/login
请求方式：POST
| 参数名 | 类型 | 必填 | 说明 |
| username | string | 是 | 账号 |
| password | string | 是 | 密码 |
{"token":"xxxxxxxxxxxx"}
"""
    parsed = parse_api_spec(text)
    assert parsed["path"] == "/login"
    assert parsed["method"] == "POST"
    assert any(f["name"] == "username" for f in parsed["fields"])

    ans = render_new_api_plan(
        "按知识库登录文档新增接口",
        kb={
            "title": "登录",
            "filename": "登录.md",
            "full_text": text,
            "parsed": parsed,
        },
        style={
            "ok": True,
            "path": "loan-service/.../QueryController.java",
            "typed_body": True,
            "style_notes": ["入参：`@RequestBody Request<XxxDto/Vo>`（强类型实体，非 Map）"],
            "code_excerpt": "public Response<ContractParamResponseDto> query(...)",
        },
    )
    assert "LoginController" in ans or "login" in ans.lower()
    assert "Request<LoginReqVo>" in ans
    assert "Response<LoginRespVo>" in ans
    assert "ResponseBuilder" in ans
    assert "username" in ans
    assert "token" in ans
    assert "Request<Map" not in ans


def test_find_style_exemplar_demo():
    import pytest
    from pathlib import Path

    from app.knowledge.add_api import find_style_exemplar

    if not (Path(__file__).resolve().parents[1] / "java_project" / "demo-loan").is_dir():
        pytest.skip("demo-loan 工程目录不存在（本地未放置被索引仓库）")

    style = find_style_exemplar("demo-loan")
    assert style.get("ok") is True
    assert "Controller" in (style.get("class_name") or "")
    # Generic scan: prefer typed Request<XxxDto/Vo> over Map-style when present
    assert style.get("typed_body") is True
    assert str(style.get("path") or "").endswith("Controller.java")


def test_render_goods_query_typed():
    from app.knowledge.add_api import parse_api_spec, render_new_api_plan

    # Mirrors desktop KB doc: tab table + request/response JSON examples
    text = """
条件查询商品
请求地址：/goods/query
请求方式：POST
请求参数（Body JSON）
参数名	类型	必填	说明
name	string	否	商品名称模糊搜索
minPrice	number	否	最低价格
maxPrice	number	否	最高价格
pageNum	int	是	当前页码
pageSize	int	是	每页数量
请求示例
{
    "name": "耳机",
    "minPrice": 50,
    "pageNum": 1,
    "pageSize": 10
}
返回示例
{
    "code": 200,
    "msg": "查询成功",
    "data": {
        "total": 16,
        "records": [{"id": 1001, "goodsName": "蓝牙耳机"}]
    }
}
"""
    parsed = parse_api_spec(text)
    assert [f["name"] for f in parsed["fields"]] == [
        "name",
        "minPrice",
        "maxPrice",
        "pageNum",
        "pageSize",
    ]
    assert "total" not in {f["name"] for f in parsed["fields"]}
    assert any(f["name"] == "total" for f in parsed["resp_fields"])

    ans = render_new_api_plan(
        "按商品查询文档新增接口",
        kb={"title": "商品查询", "full_text": text, "parsed": parsed},
        style={"ok": True, "path": "QueryController.java", "typed_body": True, "style_notes": []},
    )
    assert "Request<GoodsQueryReqVo>" in ans
    assert "private BigDecimal minPrice" in ans
    assert "pageNum不能为空" in ans
    assert "name不能为空" not in ans
    assert "Request<Map" not in ans
    # response fields must not bleed into ReqVo block
    req_block = ans.split("### 2. 响应 VO")[0]
    assert "goodsName" not in req_block
    assert "private Long total" not in req_block


def test_wants_entity_only_detection():
    """「只要实体类」与「完整接口」的意图区分（desktop-chat-d641bff7c836）。"""
    from app.agent.pev_plan import _wants_entity_only

    assert _wants_entity_only("根据知识库 商品查询 帮我生成对应的 java 实体类")
    assert _wants_entity_only("根据知识库生成实体类")
    assert _wants_entity_only("按文档帮我生成 GoodsQueryReqVo 和 RespVo")
    # 完整接口意图不算 entity_only
    assert not _wants_entity_only("按知识库新增一个登录接口")
    assert not _wants_entity_only("按知识库完整生成商品查询接口")
    assert not _wants_entity_only("按知识库生成商品查询的 Controller")


def test_render_entity_only_omits_controller_and_service():
    """entity_only 模式只输出请求/响应 VO，不输出 Controller/Service/ServiceImpl。"""
    from app.knowledge.add_api import parse_api_spec, render_new_api_plan

    text = """
# 商品查询
请求地址：/goods/query
请求方式：POST
| 参数名 | 类型 | 必填 | 说明 |
| goodsName | string | 是 | 商品名 |
| pageNum | int | 否 | 页码 |
"""
    parsed = parse_api_spec(text)
    ans = render_new_api_plan(
        "根据知识库 商品查询 帮我生成对应的 java 实体类",
        kb={"title": "商品查询", "full_text": text, "parsed": parsed},
        style={"ok": True, "path": "QueryController.java", "style_notes": []},
        entity_only=True,
    )
    assert "仅实体类" in ans
    assert "GoodsQueryReqVo" in ans
    assert "GoodsQueryRespVo" in ans
    # 不得出现完整接口的 Controller / Service / ServiceImpl 骨架
    assert "public class GoodsQueryController" not in ans
    assert "public interface GoodsQueryService" not in ans
    assert "ServiceImpl" not in ans


def test_wants_add_api_scaffold_entity_only_pattern():
    """wants_add_api_scaffold 必须覆盖「按知识库文档生成实体类」场景。

    原始正则要求句尾有 新增/实现/生成/写/落地 等动作动词，但用户常用
    「给我对应的java 实体类代码」这类措辞，缺少动作动词，导致路由降级
    到 explain_flow → empty spine → 拒绝改代码（desktop-chat-75fe902deb0b）。
    """
    from app.agent.pev_plan import wants_add_api_scaffold

    # 必须匹配的实体类生成场景
    assert wants_add_api_scaffold("按照知识库商品查询文档 给我对应的java 实体类代码")
    assert wants_add_api_scaffold("根据知识库商品查询文档生成实体类")
    assert wants_add_api_scaffold("按知识库文档生成 VO")
    assert wants_add_api_scaffold("根据知识库 商品查询 帮我生成对应的 java 实体类")
    # 完整接口场景仍匹配
    assert wants_add_api_scaffold("按知识库商品查询文档新增一个登录接口")
    assert wants_add_api_scaffold("根据知识库文档实现商品查询接口")
    # 不应误伤的流程问法（无实体类/VO 关键词）
    assert not wants_add_api_scaffold("根据知识库商品查询文档 商品查询接口主要流程是什么")
    assert not wants_add_api_scaffold("按知识库文档看看商品查询接口的代码")
    assert not wants_add_api_scaffold("根据知识库文档列出所有接口")
    assert not wants_add_api_scaffold("知识库商品查询文档的接口有哪些字段")
