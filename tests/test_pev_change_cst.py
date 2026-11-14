"""Complex code-gen / business QA: api_io param flow + change plan rendering.

Level-5 tests: request/response parameter flow (how request vars become
response vars), api_io helper discovery, and deterministic change-plan
render helpers — all with mocked source/LLM, no middleware needed.
"""
from __future__ import annotations

import asyncio

import pytest

from app.agent.pev_change import (
    _exec_collect_api_io,
    _exec_render_api_io,
    _extract_request_param_keys,
    _extract_response_param_keys,
    _outbound_focus_queries,
    _outbound_hit_score,
    _outbound_peer_excerpt,
    _rewrite_placeholder_field,
    _simple_cm,
    suggest_outbound_param_code,
)

_ENTRY_SRC = """package com.demo;

@RestController
public class ApplyController {
    @PostMapping("/apply")
    public Result apply(@RequestBody Map<String, Object> reqMap) {
        String memberId = (String) reqMap.get("memberId");
        String amount = (String) reqMap.get("amount");
        reqMap.put("loanStatus", "S");
        Result result = new Result();
        result.setResponseCode("0000");
        result.setResponseMsg("成功");
        return result;
    }

    private String generateReqParam(Map<String, Object> reqMap) {
        return (String) reqMap.get("sceneCode");
    }
}
"""

_HELPER_SRC = """package com.demo;

public class ApplyServiceImpl {
    public void apply(Map<String, Object> reqMap) {
        String scene = (String) reqMap.get("sceneCode");
        resultMap.put("scene", scene);
        resultMap.put("applyNo", "A" + scene);
    }
}
"""


@pytest.fixture
def _patch_io_sources(monkeypatch):
    def _fake_fetch(project_id: str, cm: str, **kwargs):
        del project_id, kwargs
        if cm.endswith("#apply") and "Controller" in cm:
            return _ENTRY_SRC, "ApplyController.java"
        if cm.endswith("#generateReqParam"):
            return _ENTRY_SRC, "ApplyController.java"
        if cm.endswith("#apply"):
            return _HELPER_SRC, "ApplyServiceImpl.java"
        return "", None

    monkeypatch.setattr(
        "app.agent.explore_source._fetch_method_source_body", _fake_fetch
    )


# ---------- param extraction (request → response flow) ----------

def test_extract_request_param_keys():
    code = 'String m = (String) reqMap.get("memberId");\nString a = body.get("amount");'
    keys = _extract_request_param_keys(code)
    assert "memberId" in keys
    assert "amount" in keys


def test_extract_response_param_keys_put_and_set():
    code = (
        'resp.put("loanStatus", "S");\n'
        "result.setResponseCode(\"0000\");\n"
        "result.setResponseMsg(\"成功\");"
    )
    keys = _extract_response_param_keys(code)
    assert "loanStatus" in keys
    assert "responseCode" in keys
    assert "responseMsg" in keys


def test_simple_cm_normalizes():
    assert _simple_cm("com.demo.ApplyController#apply") == "ApplyController#apply"
    assert _simple_cm("ApplyController#apply(x)") == "ApplyController#apply"
    assert _simple_cm("no-hash") == "no-hash"


# ---------- api_io helper discovery + collection ----------

def test_api_io_helper_cms_picks_req_builders(_patch_io_sources):
    from app.agent.pev_change import _api_io_helper_cms

    cms = _api_io_helper_cms(
        "demo",
        "ApplyController#apply",
        {"calls": ["ApplyServiceImpl#apply", "Repo#query"]},
    )
    assert "ApplyController#generateReqParam" in cms


def test_collect_api_io_gathers_params(_patch_io_sources):
    evidence = {"entry": "ApplyController#apply", "calls": []}
    asyncio.run(_exec_collect_api_io("demo", evidence))
    pack = evidence.get("api_io_pack")
    assert pack is not None
    assert "memberId" in pack["request_params"]
    assert "amount" in pack["request_params"]
    assert "responseCode" in pack["response_params"]
    assert "loanStatus" in pack["response_params"]
    assert pack["sources"], pack


def test_render_api_io_answer(_patch_io_sources):
    evidence = {"entry": "ApplyController#apply", "calls": []}
    asyncio.run(_exec_collect_api_io("demo", evidence))
    ans = _exec_render_api_io("该接口请求参数有哪些 响应参数有哪些", evidence)
    assert "ApplyController#apply" in ans
    assert "memberId" in ans
    assert "responseCode" in ans


# ---------- outbound param suggestion (variable flow into external calls) ----------

def test_suggest_outbound_param_code_map_put_peer():
    peer = (
        'reqMap.put(ApplyConstants.AMOUNT, reqVo.getAmount());\n'
        'reqMap.put(ApplyConstants.SCENE, reqVo.getScene());'
    )
    code = suggest_outbound_param_code(peer, "businessLabel", label_zh="业务标签")
    assert "setBusinessLabel" not in code  # map.put clone path uses getter
    assert "getBusinessLabel" in code
    assert "BUSINESS_LABEL" in code


def test_suggest_outbound_param_code_dto_set_peer():
    peer = "dstVo.setAmount(srcVo.getAmount());"
    code = suggest_outbound_param_code(peer, "businessLabel", label_zh="业务标签")
    assert "setBusinessLabel" in code
    assert "getBusinessLabel" in code


def test_suggest_outbound_empty_without_field():
    assert suggest_outbound_param_code("reqMap.put(A.B, x.getA());", None) == ""


def test_outbound_peer_excerpt_returns_head_without_peer():
    code = "a();\nb();\nc();"
    excerpt = _outbound_peer_excerpt(code, limit=2)
    assert excerpt.count("\n") + 1 <= 2


# ---------- outbound focus + scoring ----------

def test_outbound_focus_queries_picks_check_phrase():
    qs = _outbound_focus_queries("先参数校验，然后调用额度校验时将这个字段作为入参传递")
    assert any("校验" in s for s in qs)


def test_outbound_hit_score_prefers_check():
    assert _outbound_hit_score("Svc#quotaCheck", "额度校验") >= _outbound_hit_score(
        "Svc#updateLoan", "额度校验"
    )
    # demote persist methods
    assert _outbound_hit_score("Svc#updateLoan", "x") < 0


# ---------- change-plan render (code generation path) ----------

def test_rewrite_placeholder_field_all_forms():
    raw = (
        "private String newField;\n"
        "public static final String NEW_FIELD = \"newField\";\n"
        "ADD COLUMN `new_field` ... COMMENT 'newField';\n"
        "reqVo.setNewField(newField);"
    )
    out = _rewrite_placeholder_field(raw, "businessLabel")
    assert "businessLabel" in out
    assert "BUSINESS_LABEL" in out
    assert "business_label" in out
    assert "setBusinessLabel" in out
