#!/usr/bin/env python
"""Quick smoke test after restart."""
import json
import sys
import time
import urllib.request

API = "http://127.0.0.1:8000"

def ask(q: str, timeout: int = 180) -> dict:
    body = json.dumps({"message": q, "project_id": "demo-loan"}).encode("utf-8")
    req = urllib.request.Request(
        f"{API}/api/v1/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        elapsed = round(time.time() - t0, 1)
        print(f"[{elapsed}s] {q[:30]}... -> intent={data.get('intent')}")
        reply = str(data.get("reply") or "")
        if "WithdrawalsController" in reply:
            print(f"  ✓ WithdrawalsController mentioned")
        if "RepayController" in reply:
            print(f"  ✓ RepayController mentioned")
        if reply:
            print(f"  Head: {reply[:120]}")
        else:
            print(f"  ⚠ Empty reply")
        return data
    except Exception as e:
        elapsed = round(time.time() - t0, 1)
        print(f"[{elapsed}s] FAIL: {e}")
        return {"reply": ""}


if __name__ == "__main__":
    # Smoke 1: 放款加字段
    ask("我想放款接口增加一个字段 batchNo 并落库，都要修改哪些")
    # Smoke 2: 还款流程
    ask("还款流程是什么样的")
    # Smoke 3: 授信通知
    ask("授信通知是怎么实现的")
    print("\nDone.")