#!/usr/bin/env python
"""Non-destructive probe: POST /api/v1/chat with a branch log, check the reply
carries the mandatory 「嫌疑代码块与不确定性」 section."""
import json
import sys
import urllib.request

HOST = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"

log = (
    "2026-08-17 19:12:05.160 INFO 22222 (...) [http-nio-18085-exec-2] ?:? -> 期望新老流程:{},false\n"
    "2026-08-17 19:12:05.161 INFO 22222 (...) [http-nio-18085-exec-2] ?:? -> "
    "新老流程决断结果:{\"processV2EnabledFinal\":false,\"riskAsyncEnabledFinal\":false,"
    "\"guaranteeAsyncEnabledFinal\":false,\"loanPayAsyncEnabledFinal\":false,"
    "\"accountingAsyncEnabledFinal\":false,\"riskLimitOutSumSkipFinal\":false,"
    "\"loanCallbackMqEnabledFinal\":false}\n"
    "2026-08-17 19:12:05.162 INFO 22222 (...) [http-nio-18085-exec-2] ?:? -> 按老流程执行\n"
)
msg = "配置了放款走新流程，但决断结果全 false 走了老流程，为什么？【生产日志】\n" + log

payload = json.dumps(
    {
        "message": msg,
        "session_id": "probe-35884",
        "project_id": "demo-loan",
        "project_ids": ["demo-loan"],
        "client_apply": False,
    }
).encode("utf-8")

req = urllib.request.Request(
    HOST + "/api/v1/chat",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=300) as resp:
    data = json.loads(resp.read().decode("utf-8"))

reply = data.get("reply", "")
print("intent:", data.get("intent"))
print("has_suspect_section:", "嫌疑代码块" in reply)
print("has_five_chain:", "五段证据链" in reply or "配置项" in reply)
print("---reply head---")
print(reply[:600])