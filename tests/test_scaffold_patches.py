from app.agent.patcher import extract_scaffold_file_patches, filter_patches_to_allowlist


def test_extract_scaffold_file_patches_from_add_api_markdown():
    text = """
## 新增清单

### 1. 请求 VO `GoodsQueryReqVo`

`loan-service-api/src/main/java/com/demo/loan/dto/GoodsQueryReqVo.java`

```java
package com.demo.loan.dto;

import lombok.Data;

@Data
public class GoodsQueryReqVo {
    private String name;
}
```

### 2. Controller

`loan-service/src/main/java/com/demo/loan/controller/GoodsQueryController.java`

```java
package com.demo.loan.controller;

public class GoodsQueryController {
}
```
"""
    patches = extract_scaffold_file_patches(text)
    assert len(patches) == 2
    assert patches[0]["file"].endswith("GoodsQueryReqVo.java")
    assert patches[0]["new_file"] is True
    assert "class GoodsQueryReqVo" in patches[0]["content"]
    assert patches[1]["file"].endswith("GoodsQueryController.java")


def test_filter_patches_drops_over_scoped_files():
    allow = [
        "loan-service/src/main/java/com/demo/loan/controller/GoodsQueryController.java",
    ]
    patches = [
        {"file": allow[0], "content": "ok"},
        {"file": "loan-service/src/main/java/com/demo/loan/utils/signature/MapUtils.java", "content": "bad"},
        {
            "file": "loan-service/src/main/java/com/demo/loan/service/impl/WithdrawalsServiceImpl.java",
            "replacements": [{"old": "a", "new": "b"}],
        },
    ]
    kept = filter_patches_to_allowlist(patches, allow, drop_if_empty_allowlist=True)
    assert len(kept) == 1
    assert kept[0]["file"] == allow[0]


def test_parse_api_spec_resp_fields_no_record_item_bleed():
    from app.knowledge.add_api import parse_api_spec

    text = """
条件查询商品
请求地址：/goods/query
请求方式：POST
返回示例
```json
{
  "code": 200,
  "data": {
    "total": 16,
    "records": [{"id": 1001, "goodsName": "耳机", "price": 159, "stock": 236}]
  }
}
```
"""
    parsed = parse_api_spec(text)
    names = {f["name"] for f in parsed["resp_fields"]}
    assert "total" in names
    assert "records" in names
    assert "id" not in names
    assert "goodsName" not in names
    assert "price" not in names
