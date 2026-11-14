from __future__ import annotations

import json
import re
from typing import Any, AsyncIterator

from app.llm.base import LLMProvider


def _keywords_from_user(user: str) -> list[str]:
    toks = re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}", user or "")
    # drop common intent glue words
    stop = {
        "流程", "怎么", "怎么走", "什么样", "增加", "新增", "加个", "接口", "字段",
        "需要", "请", "按", "现有", "代码", "风格", "实现", "落库", "透传", "下游",
    }
    out: list[str] = []
    for t in toks:
        if t in stop:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= 4:
            break
    return out or ["接口"]


def _mock_goal_contract(user: str) -> dict[str, Any]:
    """Simulate Goal LLM: split entry vs subtopic without product-noun tables."""
    q = user or ""
    m = re.search(r"问题:\s*\n(.+)", q, re.S)
    if m:
        q = m.group(1).strip().split("\n", 1)[0].strip()
    q = re.sub(r"\s+", " ", q).strip()
    speech = re.compile(
        r"具体做了什么|具体做了啥|是做什么|主要是做|主要做了什么|主要做了啥|"
        r"主要做什么|做什么的|做了什么|做了啥|做什么|怎么实现|如何实现|"
        r"怎么做的|是干啥的|干了什么|干了啥|这一步|这块|什么样|怎么走|"
        r"是什么|什么流程|流程"
    )
    anchor = ""
    topic = ""
    am = re.search(r"([\u4e00-\u9fffA-Za-z0-9_]{2,16}接口)", q)
    if am:
        anchor = am.group(1)
        rest = (q[: am.start()] + " " + q[am.end() :]).strip()
        rest = speech.sub(" ", rest)
        rest = re.sub(r"^[中里内的]\s*", "", rest)
        rest = re.sub(r"\s+", " ", rest).strip(" ，,。.;；的中里内")
        if len(re.sub(r"\s+", "", rest)) >= 2:
            topic = rest[:120]
    ask_scope = "entry_scoped_dig_in" if (anchor and topic) else "full_flow"
    must = [
        {"id": "entry", "desc": "定位主入口"},
        {"id": "spine", "desc": "展开主链/调用证据"},
        {"id": "summarize", "desc": "按证据回答流程"},
    ]
    if ask_scope == "entry_scoped_dig_in":
        must[-1] = {
            "id": "summarize",
            "desc": "按子话题先锁方法再校验入口链并只答该方法描述",
        }
    out: dict[str, Any] = {
        "goal": "explain_flow",
        "also": [],
        "must": must,
        "ask_scope": ask_scope,
        "entry_anchor": anchor or None,
        "method_topic": topic or None,
        "focus_topic": topic or None,
        "field_name": None,
        "field_label_zh": None,
        "tables": [],
        "inventory_kind": None,
        "deliverable": "accurate_evidence_backed_answer",
    }
    if re.search(r"要改什么|改哪些|怎么改|落库|加字段|新增字段", q):
        out["goal"] = "change_plan"
        out["ask_scope"] = None
    elif re.search(r"日志|报错|异常|堆栈|error|exception", q, re.I):
        out["goal"] = "diagnose"
        out["ask_scope"] = None
    elif re.search(r"源码|看代码|贴代码|代码是什么|java代码|Java代码|给我一段代码|生成一段", q):
        out["ask_scope"] = None
        # Mock Goal: structural multi-step vs single lookup (tests/offline only).
        if re.search(r"先.+再|分组|group\s*by|关联查询|如果查到|再根据", q, re.I):
            out["goal"] = "compose_code"
            out["view_kind"] = "db_pipeline"
            out["lookup_steps"] = [
                {"op": "select", "table": "表A"},
                {"op": "select", "table": "表B"},
            ]
        elif re.search(r"通过|按|根据|.+查.+表|查询.+表|给我一段", q):
            out["goal"] = "compose_code"
            out["view_kind"] = "db_lookup"
            out["lookup_key"] = "查询键"
            out["lookup_table"] = "目标表"
        else:
            out["goal"] = "show_code"
    return out


class MockLLMProvider(LLMProvider):
    """Offline provider for tests / provider-swap smoke checks."""

    name = "mock"

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        system = messages[0].get("content", "") if messages else ""
        user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        blob = (system + "\n" + user).lower()
        # Turn-splitter (回合意图拆分): mirror the heuristic splitter so mock
        # tests stay deterministic — real LLM does the smarter semantic split.
        if "回合意图拆分" in system:
            from app.agent.runtime.turn_plan import (
                _MODIFY_RE,
                _PLAN_RE,
                _EXECUTE_RE,
                _GIT_RE,
                _NUM_SPLIT_RE,
                _ORDER_SPLIT_RE,
            )

            # Heuristic misses long compound sentences that use 顺便/还有/同时
            # or a junction word like 再看看/顺便看 (no standard 先/然后/再
            # connectives). The real LLM splits those semantically; mock mirrors
            # the connector set so tests exercise the multi-intent path
            # deterministically. Verb-junction compact forms (「帮我改下代码」)
            # have no clean separator — real LLM splits them; mock keeps the
            # utterance whole (documented mock limitation, not a product gap).
            _COMPOUND_SPLIT_RE = re.compile(
                r"顺便|同时|还有|另外|以及|顺带|并且|还要|接着再|然后再|"
                r"再看看|顺便看|然后顺带|接着帮我|再帮我"
            )

            q = (user or "").strip()
            stages: list[str] = []
            numbered = [p.strip() for p in _NUM_SPLIT_RE.split(q) if p and p.strip()]
            if len(numbered) >= 2:
                stages = numbered[:8]
            else:
                parts = [p.strip(" ，,;；") for p in _ORDER_SPLIT_RE.split(q) if p and p.strip()]
                if len(parts) < 2:
                    parts = [p.strip(" ，,;；") for p in _COMPOUND_SPLIT_RE.split(q) if p and p.strip()]
                stages = parts[:8] if len(parts) >= 2 else [q]
            out_stages = []
            for st in stages:
                if _GIT_RE.search(st):
                    layer = "client"
                elif _EXECUTE_RE.search(st):
                    layer = "execute"
                elif _MODIFY_RE.search(st):
                    layer = "modify"
                elif _PLAN_RE.search(st):
                    layer = "plan"
                else:
                    layer = "ask"
                out_stages.append({"layer": layer, "text": st})
            return json.dumps({"stages": out_stages}, ensure_ascii=False)
        # Unified client-agent router (no keyword product gate — mock still uses intent cues).
        if (
            "判断用户这句话如何处理" in system
            or "You route the user's message" in system
            or ("mode" in system and "direct" in system and "reply_preview" in system)
        ):
            u = user or ""
            # Router user payload is "User message:\n..." — strip wrapper for cues/goal.
            if re.match(r"(?is)^\s*user\s*message\s*:", u):
                u = re.sub(r"(?is)^\s*user\s*message\s*:\s*", "", u).strip()
            if any(k in u for k in ("C盘", "c盘", "磁盘", "垃圾", "清理", "回收站", "缓存")):
                mode = "direct"
                goal = "分析磁盘占用并给出可清理项"
                preview = "好的，我先在本机扫一下磁盘占用。"
            elif any(
                k in u
                for k in (
                    "清单",
                    "不要改",
                    "不要直接改",
                    "改哪些",
                    "哪些类",
                    "改动点",
                    "影响面",
                )
            ):
                mode = "none"
                goal = ""
                preview = ""
            elif any(k in u for k in ("加字段", "补丁", "自测", "编译", "改造", "落库", "修改接口")):
                mode = "code"
                goal = u[:120]
                preview = "好的，我会改代码并在本机编译自测。"
            elif any(
                k in u
                for k in (
                    "装依赖",
                    "安装依赖",
                    "启动",
                    "部署",
                    "compose",
                    "maven install",
                )
            ) or (
                # Mock-only heuristics (product code uses real LLM intent).
                any(k in u.lower() for k in ("git", "java", "maven", "node", "docker"))
                and any(
                    k in u.lower()
                    for k in (
                        "安装",
                        "按照",
                        "装了",
                        "是否",
                        "有没有",
                        "路径",
                        "版本",
                        "本地",
                        "本机",
                        "install",
                        "installed",
                        "path",
                        "version",
                        "machine",
                        "local",
                        "where",
                        "环境变量",
                        "env",
                    )
                )
            ):
                mode = "direct"
                goal = u[:120] or "检查本机环境"
                preview = "好的，我在本机检查一下。"
            else:
                mode = "none"
                goal = ""
                preview = ""
            return json.dumps(
                {"mode": mode, "goal": goal, "reply_preview": preview},
                ensure_ascii=False,
            )

        if (
            "结果解说员" in system
            or "You explain local-command results" in system
            or "最终中文结论" in user
            or "Write the final answer in the user's language" in system
        ):
            path_m = re.search(
                r"([A-Za-z]:\\[^\r\n\"']+?(?:git|java|mvn|node|python|docker)\.exe)",
                user,
                re.I,
            )
            path = path_m.group(1) if path_m else ""
            ver_m = re.search(
                r"((?:git version|java version|openjdk|node v|Python |Apache Maven)[^\r\n]+)",
                user,
                re.I,
            )
            ver = ver_m.group(1) if ver_m else ""
            tool = "工具"
            for name in ("git", "java", "maven", "node", "python", "docker"):
                if re.search(rf"\b{name}\b", user, re.I) or name in (path or "").lower():
                    tool = name.upper() if name == "git" else name
                    if name == "git":
                        tool = "Git"
                    elif name == "java":
                        tool = "Java"
                    break
            if any(k in user.lower() for k in ("git", "java", "maven", "node", "python", "docker")):
                if path or ver or "ok=True" in user or "ok=true" in user or "soft_ok" in user:
                    bits = [f"本地已经安装 {tool}。"]
                    # Prefer listing all where.exe paths when present
                    paths = re.findall(
                        r"(?im)^([A-Za-z]:\\[^\r\n]+?\.(?:exe|bat|cmd))\s*$",
                        user,
                    )
                    if paths:
                        bits.append("可执行文件路径：")
                        bits.extend(f"- {p}" for p in paths[:6])
                    elif path:
                        bits.append(f"可执行文件路径：{path}")
                    if ver:
                        bits.append(f"版本：{ver}")
                    if not path and not ver and not paths:
                        bits.append("命令已成功执行，请结合输出确认具体路径与版本。")
                    return "\n".join(bits)
                return f"本机未检测到可用的 {tool}（where/--version 无有效输出），可能未安装或未加入 PATH。"
            return "本机检查已完成。根据命令输出，关键步骤已执行完毕。"

        # Post-compile judge (client_patch_loop._llm_after_compile): decide
        # success/failure from the raw mvn compile log. Mock answers by looking
        # for BUILD SUCCESS / failure markers instead of deferring to a real LLM.
        if "编译结果分析助手" in system or (
            "mvn compile" in blob and "编译日志" in user and "actions" in blob
        ):
            log = (user or "").lower()
            if any(k in log for k in ("build success", "编译成功", "compile succeeded", "0 errors")):
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "done",
                                "note": "编译通过",
                                "reply": "根据编译日志，本机 `mvn compile` 已通过。",
                            }
                        ],
                        "reasoning": "mock post-compile judge: BUILD SUCCESS",
                    },
                    ensure_ascii=False,
                )
            if any(k in log for k in ("build failure", "error:", "failed", "compile error", "cannot find symbol")):
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "fail",
                                "note": "编译失败（mock 判定）",
                                "reply": "编译日志显示失败，请查看日志后继续修复。",
                            }
                        ],
                        "reasoning": "mock post-compile judge: failure markers",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "actions": [
                        {
                            "type": "done",
                            "note": "编译通过",
                            "reply": "根据编译日志，本机 `mvn compile` 已通过。",
                        }
                    ],
                    "reasoning": "mock post-compile judge: no failure markers",
                },
                ensure_ascii=False,
            )

        if (
            "本机任务的规划器" in system
            or "You plan local PC tasks" in system
            or ("任务分解" in system and "plan_steps" in system)
            or ("plan_steps" in system and "success_criteria" in system and "plan only" in system.lower())
        ):
            m_goal = re.search(r"(?m)^用户目标:\s*(.+)$", user)
            goal = (m_goal.group(1).strip() if m_goal else user)[:200]
            if re.match(r"(?is)^\s*user\s*message\s*:", goal):
                goal = re.sub(r"(?is)^\s*user\s*message\s*:\s*", "", goal).strip() or goal
            if any(k in goal for k in ("磁盘", "清理", "垃圾", "C盘", "c盘", "占用")):
                steps = [
                    "查看磁盘概况",
                    "扫描可安全清理项",
                    "汇总报告并请用户确认",
                    "仅清理用户勾选项",
                ]
                criteria = "用户确认后完成勾选项清理，或用户选择不清理"
            elif any(k in goal for k in ("启动", "start", "部署", "联调")):
                steps = [
                    "确认项目类型与启动方式",
                    "安装依赖（如需要）",
                    "启动应用",
                    "根据日志或健康检查确认启动成功",
                    "失败则读错修复后重试",
                ]
                criteria = "启动日志显示应用已就绪，或健康检查通过"
            elif (
                any(k in goal.lower() for k in ("pull", "拉取", "最新代码", "latest"))
                and any(k in goal.lower() for k in ("git", "安装", "install", "有没有", "是否"))
            ):
                # Mock compound plan — product path uses real LLM planning.
                steps = [
                    "确认版本控制工具是否可用",
                    "若可用则拉取远端最新代码",
                    "汇总路径/版本与拉取结果",
                ]
                criteria = "已确认工具可用并完成拉取（或明确说明不可用）"
            elif any(k in goal.lower() for k in ("git", "java", "maven", "node", "docker", "环境变量", "java_home", "path")) and any(
                k in goal for k in ("安装", "按照", "路径", "版本", "是否", "有没有", "本地", "本机", "配置", "install", "path", "配")
            ):
                steps = [
                    "用 where/Get-Command 或 --version 确认工具是否安装",
                    "输出可执行文件完整路径与版本",
                    "汇总给用户（含路径）",
                ]
                criteria = "已确认是否安装，并给出可执行路径或明确未安装"
            else:
                steps = [
                    "了解现状",
                    f"执行：{goal}",
                    "验证结果；失败则修复重试",
                ]
                criteria = "用户目标完成且关键步骤成功"
            return json.dumps(
                {
                    "plan_steps": steps,
                    "success_criteria": criteria,
                    "reply": "已完成任务分解：" + " → ".join(steps[:4]),
                },
                ensure_ascii=False,
            )

        if (
            "本机助手的编排大脑" in system
            or "远程运维大脑" in system
            or ("remote agent" in blob and "maven_install" in blob)
            or ("sys_disk_overview" in system and "ask_user" in system)
        ):
            # Client local execution loop — emit a safe plan from goal / observations.
            # Keyword routing must use the goal line only — decide prompts mention 编译/失败 etc.
            m_goal = re.search(r"(?m)^目标:\s*(.+)$", user)
            goal_line = (m_goal.group(1).strip() if m_goal else "") or user.split("\n", 1)[0]
            if re.match(r"(?is)^\s*user\s*message\s*:", goal_line):
                m2 = re.search(r"(?m)^目标:\s*User message:\s*\n(.+)$", user) or re.search(
                    r"(?is)user\s*message:\s*(.+)", user
                )
                if m2:
                    goal_line = m2.group(1).strip().split("\n", 1)[0][:200]
            goal_blob = goal_line
            has_obs = "### ACTION" in user
            last_failed = "ok=False" in user or "ok=false" in user
            user_denied = "user_denied" in user or "用户拒绝" in user
            if user_denied and has_obs:
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "ask_user",
                                "reply": "你拒绝了上一步本机操作。要换一种检查方式继续，还是先停在这里？",
                                "options": [
                                    {"id": "continue_repair", "title": "换命令继续"},
                                    {"id": "abandon", "title": "先停止"},
                                ],
                            }
                        ],
                        "reasoning": "mock stop after user_denied",
                    },
                    ensure_ascii=False,
                )
            if has_obs and "### ACTION http_smoke" in user:
                if not last_failed and ("smoke ok" in user.lower() or "ok=True" in user or "ok=true" in user):
                    return json.dumps(
                        {
                            "actions": [
                                {
                                    "type": "done",
                                    "note": "冒烟通过",
                                    "reply": "根据冒烟结果，服务已联通，任务完成。",
                                }
                            ],
                            "reasoning": "mock analyze smoke ok",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "fail",
                                "note": "冒烟未通过",
                                "reply": "根据冒烟输出，服务未联通，请检查端口与路径。",
                            }
                        ],
                        "reasoning": "mock analyze smoke fail",
                    },
                    ensure_ascii=False,
                )
            if has_obs and "### ACTION sys_cleanup" in user:
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "done",
                                "note": "清理完成",
                                "reply": user.split("stdout:\n")[-1][:500] if "stdout:" in user else "清理完成",
                            }
                        ],
                        "reasoning": "mock analyze cleanup",
                    },
                    ensure_ascii=False,
                )
            if has_obs and "### ACTION start_app" in user:
                # Only the latest start_app section counts (older failures may remain in history).
                last_start = user.split("### ACTION start_app")[-1].split("### ACTION")[0]
                start_failed = "ok=False" in last_start or "ok=false" in last_start
                if start_failed:
                    return json.dumps(
                        {
                            "actions": [
                                {"type": "list_dir", "path": "."},
                                {"type": "read_file", "path": "pom.xml"},
                                {"type": "start_app", "wait_ms": 30000},
                            ],
                            "reasoning": "mock repair after start fail",
                        },
                        ensure_ascii=False,
                    )
                if "自测" in goal_blob or "联调" in goal_blob or "smoke" in goal_blob.lower():
                    return json.dumps(
                        {
                            "actions": [
                                {
                                    "type": "http_smoke",
                                    "urls": [
                                        "http://127.0.0.1:8080/actuator/health",
                                        "http://127.0.0.1:8080/health",
                                    ],
                                    "expect_status": 200,
                                }
                            ],
                            "reasoning": "mock decide smoke after start",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "done",
                                "note": "日志显示应用已启动",
                                "reply": "根据启动日志判断，应用已启动。",
                            }
                        ],
                        "reasoning": "mock analyze start ok",
                    },
                    ensure_ascii=False,
                )
            if has_obs and "### ACTION maven_compile" in user and not last_failed:
                if any(k in goal_blob for k in ("启动", "start", "联调", "部署")):
                    return json.dumps(
                        {
                            "actions": [{"type": "start_app", "wait_ms": 30000}],
                            "reasoning": "mock compile ok → start",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "done",
                                "note": "编译通过",
                                "reply": "根据编译日志，本机 `mvn compile` 已通过。",
                            }
                        ],
                        "reasoning": "mock analyze compile ok",
                    },
                    ensure_ascii=False,
                )
            if has_obs and "### ACTION sys_scan_heavy" in user:
                opts = []
                try:
                    # Prefer options hinted in observation meta JSON
                    m = re.search(
                        r'"cleanup_options_hint":\s*(\[[^\]]*\])',
                        user,
                    )
                    if m:
                        opts = json.loads(m.group(1))
                except Exception:
                    opts = []
                if not opts:
                    opts = [
                        {
                            "id": "user_temp",
                            "title": "用户临时文件",
                            "summary": "可安全清理",
                        }
                    ]
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "ask_user",
                                "reply": "扫描完成，请勾选要清理的项。",
                                "options": opts,
                                "note": "mock ask cleanup",
                            }
                        ],
                        "reasoning": "mock ask after scan",
                    },
                    ensure_ascii=False,
                )
            if any(k in goal_blob for k in ("磁盘", "清理", "垃圾", "C盘", "c盘", "占用")) and "编译" not in goal_blob:
                return json.dumps(
                    {
                        "actions": [
                            {"type": "sys_disk_overview", "drive": "C"},
                            {"type": "sys_scan_heavy", "drive": "C"},
                        ],
                        "reasoning": "mock disk scan",
                    },
                    ensure_ascii=False,
                )
            if (
                any(k in goal_blob.lower() for k in ("git", "java", "maven", "node", "python", "docker"))
                and any(
                    k in goal_blob.lower()
                    for k in (
                        "安装",
                        "按照",
                        "路径",
                        "版本",
                        "是否",
                        "本地",
                        "本机",
                        "有没有",
                        "install",
                        "path",
                        "version",
                    )
                )
                and "编译" not in goal_blob
                and "启动" not in goal_blob
            ):
                tool = "git"
                for name in ("java", "maven", "mvn", "node", "python", "docker", "git"):
                    if name in goal_blob.lower():
                        tool = "maven" if name == "mvn" else name
                        break
                ver_cmd = {
                    "git": "git --version",
                    "java": 'cmd /c "java -version 2>&1"',
                    "maven": "mvn -version",
                    "node": "node -v",
                    "python": "python --version",
                    "docker": "docker --version",
                }.get(tool, f"{tool} --version")
                where_cmd = f"where.exe {tool}"
                if tool == "maven":
                    where_cmd = "where.exe mvn"
                wants_pull = any(
                    k in goal_blob.lower() for k in ("pull", "拉取", "最新代码", "latest")
                )
                if has_obs and "### ACTION git_pull" in user:
                    return json.dumps(
                        {
                            "actions": [
                                {
                                    "type": "done",
                                    "note": "检查并拉取完成",
                                    "reply": "工具可用，已尝试拉取最新代码；详见执行输出。",
                                }
                            ],
                            "reasoning": "mock done after pull",
                        },
                        ensure_ascii=False,
                    )
                if has_obs and (
                    "### ACTION run_powershell" in user or "### ACTION run_shell" in user
                ):
                    if wants_pull and tool == "git" and not last_failed:
                        return json.dumps(
                            {
                                "actions": [
                                    {
                                        "type": "git_pull",
                                        "remote": "origin",
                                        "ref": "main",
                                    }
                                ],
                                "reasoning": "mock continue plan: pull after check",
                            },
                            ensure_ascii=False,
                        )
                    return json.dumps(
                        {
                            "actions": [
                                {
                                    "type": "done",
                                    "note": "已完成本机环境检查",
                                    "reply": "已根据本机命令输出汇总：工具是否安装及路径见上方执行结果。",
                                }
                            ],
                            "reasoning": f"mock env check done ({tool})",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "actions": [
                            {
                                "type": "run_powershell",
                                "command": f"{where_cmd}; {ver_cmd}",
                                "cwd": ".",
                            }
                        ],
                        "reasoning": f"mock check {tool} then continue plan",
                    },
                    ensure_ascii=False,
                )
            if any(k in goal_blob for k in ("补丁", "编译", "apply_patch")) and "启动" not in goal_blob:
                if last_failed:
                    return json.dumps(
                        {
                            "actions": [
                                {"type": "read_file", "path": "pom.xml"},
                                {"type": "maven_compile"},
                            ],
                            "reasoning": "mock repair compile",
                        },
                        ensure_ascii=False,
                    )
                return json.dumps(
                    {
                        "actions": [{"type": "maven_compile"}],
                        "reasoning": "mock compile verify",
                    },
                    ensure_ascii=False,
                )
            if last_failed and has_obs:
                return json.dumps(
                    {
                        "actions": [
                            {"type": "read_file", "path": "pom.xml"},
                            {"type": "list_dir", "path": "."},
                            {"type": "maven_install"},
                            {"type": "start_app", "wait_ms": 30000},
                        ],
                        "reasoning": "mock repair after failure",
                    },
                    ensure_ascii=False,
                )
            if any(k in goal_blob for k in ("依赖", "启动", "部署", "compose", "install")):
                return json.dumps(
                    {
                        "actions": [
                            {"type": "list_dir", "path": "."},
                            {"type": "maven_install"},
                            {"type": "start_app", "wait_ms": 30000},
                        ],
                        "reasoning": "mock install and start",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "actions": [
                        {"type": "list_dir", "path": "."},
                        {"type": "docker_compose_up"},
                        {"type": "maven_install"},
                        {"type": "start_app", "wait_ms": 30000},
                    ],
                    "reasoning": "mock bootstrap",
                },
                ensure_ascii=False,
            )
        if "目标契约抽取器" in system:
            return json.dumps(_mock_goal_contract(user), ensure_ascii=False)
        if "意图识别" in system or "intent" in blob and "keywords" in blob:
            intent = "ask"
            want_apply = False
            want_verify = False
            need_options = False
            awaiting = "待确认改码: yes" in user or "待确认改码:yes" in user
            # Current utterance after "当前用户:"
            cur_m = re.search(r"当前用户\s*[:：]\s*(.+)$", user, re.S)
            cur = (cur_m.group(1) if cur_m else user).strip().splitlines()[0].strip()
            # Only explicit recall phrases — not content questions that merely contain「问」
            if awaiting and cur:
                # Mock-only: simulate LLM consent judgment for tests.
                if any(
                    k in cur
                    for k in ("不用", "先不", "暂时不", "别改", "先看看")
                ):
                    intent = "meta"
                    want_apply = False
                else:
                    intent = "modify"
                    want_apply = True
                    want_verify = True
            # Match cues on the current utterance only — the full `user` blob
            # embeds classifier instructions (e.g. 「继续问清单细节」) that would
            # otherwise always trip plan/meta heuristics.
            elif any(
                k in cur
                for k in (
                    "刚才问",
                    "我问你什么",
                    "问了什么",
                    "问你什么",
                    "再说一遍",
                    "你听到了什么",
                    "上一问",
                    # English conversation-recall cues (mirror taxonomy
                    # _META_KEYS; matched case-insensitively — "What did I ask").
                    "what did i ask",
                    "what did i say",
                    "what did you hear",
                    "repeat that",
                    "repeat it",
                    "say it again",
                )
            ) and not any(k in cur.lower() for k in ("校验", "字段", "参数", "流程", "链路")):
                intent = "meta"
            elif any(
                k in cur
                for k in (
                    "几种方案",
                    "多种方案",
                    "三种实现",
                    "两种做法",
                    "对比利弊",
                    "方案对比",
                    "给我方案",
                    "可选方案",
                )
            ):
                intent = "plan"
                need_options = True
            elif any(
                k in cur
                for k in (
                    "清单",
                    "不要改",
                    "不要直接改",
                    "改哪些",
                    "哪些类",
                    "改动点",
                    "影响面",
                    "都要修改",
                )
            ):
                intent = "plan"
                want_apply = False
            elif any(
                k in cur
                for k in (
                    "编译一下",
                    "帮我编译",
                    "跑一下测试",
                    "修编译错误",
                    "修一下报错",
                )
            ):
                intent = "execute"
            elif any(
                k in cur
                for k in (
                    "直接改代码",
                    "落地补丁",
                    "按方案实现",
                    "帮我改并",
                    "并自测",
                    "改完自测",
                )
            ):
                intent = "modify"
                want_apply = True
                want_verify = True
            elif any(k in cur for k in ["增加", "字段", "落库", "改造", "修改"]):
                intent = "modify"
                want_apply = True
                want_verify = any(k in cur for k in ("自测", "编译", "验证"))
            elif any(k in cur for k in ["新接口", "新增接口"]):
                intent = "modify"
                want_apply = True
            keywords = _keywords_from_user(cur or user)
            hint = next((k for k in keywords if re.search(r"[A-Za-z]", k)), keywords[0]) if keywords else ""
            return json.dumps(
                {
                    "intent": intent,
                    "capability": {
                        "ask": "code_qa",
                        "plan": "code_qa",
                        "modify": "modify",
                        "execute": "verify",
                        "meta": "meta",
                    }.get(intent, "code_qa"),
                    "keywords": keywords,
                    "entry_hint": hint,
                    "detail": (cur or user)[:200],
                    "need_options": need_options,
                    "want_apply": want_apply,
                    "want_verify": want_verify,
                },
                ensure_ascii=False,
            )
        if "暂不需要你改代码" in system or "表示暂不需要" in system:
            return "好的，先不改代码。需要时随时叫我。"
        if "修改范围" in system or "只输出 json 数组" in blob or "priority" in blob:
            return json.dumps(
                [
                    {
                        "file": "src/main/java/com/example/controller/DemoController.java",
                        "reason": "接口入口",
                        "action": "add_field",
                        "priority": 1,
                    },
                    {
                        "file": "src/main/java/com/example/dto/DemoRequest.java",
                        "reason": "请求字段扩展",
                        "action": "add_field",
                        "priority": 2,
                    },
                    {
                        "file": "src/main/java/com/example/entity/DemoEntity.java",
                        "reason": "落库实体",
                        "action": "persist",
                        "priority": 3,
                    },
                    {
                        "file": "src/main/java/com/example/service/DemoService.java",
                        "reason": "下游参数组装/透传",
                        "action": "downstream",
                        "priority": 4,
                    },
                ],
                ensure_ascii=False,
            )
        if "挑选数据库实体" in system or "controllers" in blob and "listeners" in blob and "entities" in blob:
            # Mock file classification from listed paths
            entities, controllers, listeners, jobs = [], [], [], []
            for line in user.splitlines():
                if "\t" not in line and "/" not in line:
                    continue
                path = line.split("\t")[0].strip()
                if not path or path.startswith("文件"):
                    continue
                low = path.lower()
                item = {"path": path, "reason": "mock"}
                if "entity" in low or low.endswith("do.java") or low.endswith("po.java"):
                    entities.append(item)
                elif "controller" in low:
                    controllers.append(item)
                elif "listener" in low:
                    listeners.append(item)
                elif "job" in low:
                    jobs.append(item)
            return json.dumps(
                {
                    "entities": entities[:20],
                    "controllers": controllers[:20],
                    "listeners": listeners[:20],
                    "jobs": jobs[:20],
                    "other_entries": [],
                },
                ensure_ascii=False,
            )
        if "业务流程中的作用" in system or "next_calls" in blob and "important_types" in blob:
            return json.dumps(
                {
                    "important": True,
                    "important_types": ["other_important"],
                    "summary": "MockClass#mockMethod：mock 重要节点",
                    "status_transitions": [],
                    "external_calls": [],
                    "validation_points": [],
                    "next_calls": [],
                },
                ensure_ascii=False,
            )
        if "有序中文阶段" in system or ("narrative" in blob and "stages" in blob and "流程" in system):
            return json.dumps(
                {
                    "title": "Mock流程",
                    "narrative": "1. 入口受理\n2. 业务处理\n3. 结果回写",
                    "stages": [
                        {"step": 1, "title": "入口", "class_method": "A#a", "desc": "入口受理"},
                        {"step": 2, "title": "处理", "class_method": "B#b", "desc": "业务处理"},
                    ],
                },
                ensure_ascii=False,
            )
        if "Java结构解析" in system or "java 源文件" in blob or "symbol_kind" in blob and "feign_meta" in blob:
            # Offline mock: emit a minimal but valid structure from the source snippet.
            src = user
            pkg_m = re.search(r"package\s+([\w.]+)\s*;", src)
            package = pkg_m.group(1) if pkg_m else ""
            class_m = re.search(
                r"(class|interface|enum)\s+([A-Za-z_]\w*)",
                src,
            )
            simple = class_m.group(2) if class_m else "Demo"
            kind = class_m.group(1) if class_m else "class"
            anns = []
            for am in re.finditer(r"@([A-Za-z_]\w+)", src):
                anns.append({"name": am.group(1), "values": {}})
            symbol_kind = "Class"
            ann_names = {a["name"] for a in anns}
            if "RestController" in ann_names or "Controller" in ann_names:
                symbol_kind = "Controller"
            elif "FeignClient" in ann_names:
                symbol_kind = "Feign"
            elif "RabbitListener" in ann_names or simple.endswith("Listener"):
                symbol_kind = "Listener"
            elif "Service" in ann_names or simple.endswith("Service"):
                symbol_kind = "Service"
            elif "Entity" in ann_names or simple.endswith("Entity"):
                symbol_kind = "Entity"
            elif "Mapper" in ann_names or simple.endswith("Mapper"):
                symbol_kind = "Mapper"
            methods = []
            for mm in re.finditer(
                r"(?:public|protected|private)\s+[\w.<>,\[\]]+\s+([A-Za-z_]\w*)\s*\(([^)]*)\)\s*\{",
                src,
            ):
                name = mm.group(1)
                if name in {"if", "for", "while", "switch", "catch"} or name == simple:
                    continue
                if name.startswith("get") or name.startswith("set") or name.startswith("is"):
                    continue
                params_raw = (mm.group(2) or "").strip()
                params = []
                if params_raw:
                    for p in params_raw.split(","):
                        parts = p.strip().split()
                        if len(parts) >= 2:
                            params.append({"type": parts[-2], "name": parts[-1]})
                line = src[: mm.start()].count("\n") + 1
                methods.append(
                    {
                        "name": name,
                        "params": params,
                        "return_type": "Object",
                        "modifiers": ["public"],
                        "start_line": line,
                        "end_line": line + 20,
                        "annotations": [],
                        "http_mappings": [],
                        "calls": [],
                        "skip_accessor": False,
                    }
                )
            if not methods:
                methods.append(
                    {
                        "name": "execute",
                        "params": [],
                        "return_type": "void",
                        "modifiers": ["public"],
                        "start_line": 1,
                        "end_line": 5,
                        "annotations": [],
                        "http_mappings": [],
                        "calls": [],
                        "skip_accessor": False,
                    }
                )
            return json.dumps(
                {
                    "package": package,
                    "imports": [],
                    "types": [
                        {
                            "kind": kind,
                            "simple_name": simple,
                            "modifiers": ["public"],
                            "extends": [],
                            "implements": [],
                            "annotations": anns[:12],
                            "symbol_kind": symbol_kind,
                            "feign_meta": {},
                            "request_mapping": "",
                            "fields": [],
                            "methods": methods[:30],
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if "实体状态字段如何变化" in system or "touches_entity_status" in blob:
            return json.dumps(
                {
                    "touches_entity_status": True,
                    "summary": "MockService#update：实体 status 变更",
                    "status_transitions": [
                        {
                            "entity": "OrderEntity",
                            "field": "status",
                            "from": "",
                            "to": "SUCCESS",
                            "evidence": "setStatus(SUCCESS)",
                        }
                    ],
                    "next_calls": [],
                },
                ensure_ascii=False,
            )
        if "状态流转说明" in system or ("status_fields" in blob and "transitions" in blob and "entity" in blob):
            return json.dumps(
                {
                    "entity": "OrderEntity",
                    "status_fields": ["status"],
                    "title": "OrderEntity状态流转",
                    "narrative": "1. MockService#update：status → SUCCESS",
                    "transitions": [
                        {
                            "from": "",
                            "to": "SUCCESS",
                            "class_method": "MockService#update",
                            "desc": "更新成功",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        if "持久层专家" in system or ("table_name" in blob and "status_fields" in blob):
            return json.dumps(
                {
                    "tags": ["实体", "落库", "Entity"],
                    "summary": "实体类（mock）含 status 字段",
                    "business_domain": "持久化",
                    "table_name": "",
                    "status_fields": ["status"],
                    "status_values": {"status": ["INIT", "SUCCESS"]},
                    "key_fields": ["id"],
                },
                ensure_ascii=False,
            )
        if "入口定位器" in system or ("选出最可能的业务入口" in system) or (
            "选出最可能的 HTTP" in system
        ):
            # Prefer suggested main entry / Controller#stem over mid-chain helpers
            m_sug = re.search(r"建议主入口:\s*([A-Za-z_][\w]*#[A-Za-z_][\w]*)", user)
            if m_sug:
                entry = m_sug.group(1)
            else:
                cms = re.findall(
                    r"([A-Za-z][\w]*Controller)#[A-Za-z_][\w]*",
                    user,
                )
                entry = "DemoController#demo"
                for cm in cms:
                    meth = cm.split("#")[-1]
                    cls = cm.split("#")[0]
                    stem = re.sub(r"Controller$", "", cls, flags=re.I).lower()
                    if meth.lower() == stem:
                        entry = cm
                        break
                else:
                    if cms:
                        entry = next(
                            (
                                c
                                for c in cms
                                if not re.search(
                                    r"(?i)dealwith|check\w*param|generate",
                                    c.split("#")[-1],
                                )
                            ),
                            cms[0],
                        )
            return json.dumps(
                {"entry": entry, "reason": "mock entry pick"},
                ensure_ascii=False,
            )
        if "关卡方法定位器" in system or ("选出最匹配该关卡" in system):
            focus = ""
            fm = re.search(r"关注点 focus:\s*(.+)", user)
            if fm:
                focus = fm.group(1).strip()
            # Prefer ServiceImpl + check/validate from candidate list (no product map).
            cms = re.findall(r"([A-Za-z][\w]*#[A-Za-z_][\w]*)", user)
            target = ""
            for cm in cms:
                meth = cm.split("#")[-1].lower()
                cls = cm.split("#")[0].lower()
                if "controller" in cls and "deal" in meth:
                    continue
                if re.search(r"校验|check|validate|verify|switch", focus, re.I):
                    if re.search(r"(?i)check|validate|verify|switch", meth):
                        target = cm
                        if "serviceimpl" in cls:
                            break
            if not target:
                for cm in cms:
                    if "serviceimpl#" in cm.lower() and "check" in cm.lower():
                        target = cm
                        break
            if not target and cms:
                target = cms[0]
            return json.dumps(
                {"target": target, "reason": "mock gate target"},
                ensure_ascii=False,
            )
        if "Ask 检索规划器" in system or "ask_kind" in system:
            prior_m = re.search(r"上一问主题:\s*(.+)", user)
            cur_m = re.search(r"本轮问题:\s*(.+)", user)
            prior = (prior_m.group(1).strip() if prior_m else "").replace("(无)", "")
            cur = (cur_m.group(1).strip() if cur_m else user[:80]).strip()
            is_fu = bool(prior) and prior != "(空)"
            ask_kind = "general"
            if any(k in cur for k in ("调用方", "谁调用", "上游")):
                ask_kind = "caller"
            elif any(k in cur for k in ("下游", "调用了哪些", "调用链")):
                ask_kind = "downstream"
            elif any(k in cur for k in ("字段", "参数", "入参")):
                ask_kind = "field"
            elif any(
                k in cur for k in ("校验", "判断", "开关")
            ) and (is_fu or "部分" in cur):
                ask_kind = "gate"
            elif any(k in f"{prior}{cur}" for k in ("流程", "怎么走", "步骤", "链路")):
                ask_kind = "flow"
            tid = {
                "caller": "calls_callers",
                "downstream": "calls_callees",
                "flow": "follows_spine",
                "gate": "calls_callees",
                "field": "follows_spine",
                "general": "follows_spine",
            }[ask_kind]
            focus = ""
            fm = re.search(r"([\u4e00-\u9fff]{2,12}(?:校验|判断|检查|开关))", cur)
            if fm:
                focus = fm.group(1)
            search_q = cur
            if is_fu:
                search_q = (
                    f"在上一问「{prior[:80]}」这一业务下，继续追问：{cur}"
                )
            return json.dumps(
                {
                    "intent": "ask",
                    "ask_kind": ask_kind,
                    "is_followup": is_fu or ask_kind == "gate",
                    "keep_entry_lock": is_fu or ask_kind == "gate",
                    "search_question": search_q,
                    "template_hint": tid,
                    "focus": focus or cur[:40],
                    "reason": "mock ask plan",
                },
                ensure_ascii=False,
            )
        if "检索规划器" in system or ("follows_spine" in system and "template_id" in system):
            tid = "follows_spine"
            if any(k in user for k in ("调用方", "谁调用", "上游")):
                tid = "calls_callers"
            elif any(k in user for k in ("Feign", "feign", "外部")):
                tid = "feign_neighbors"
            elif any(k in user for k in ("下游", "调用了哪些", "调用链")):
                tid = "calls_callees"
            m = re.search(r"([A-Za-z][\w]*Controller)#[A-Za-z_][\w]*", user)
            entry = m.group(0) if m else "DemoController#demo"
            m_entry = re.search(r"已定位入口:\s*([A-Za-z_][\w]*#[A-Za-z_][\w]*)", user)
            if m_entry:
                entry = m_entry.group(1)
            return json.dumps(
                {
                    "template_id": tid,
                    "params": {"entry": entry, "depth": 4, "limit": 20},
                    "reason": "mock neo4j template",
                },
                ensure_ascii=False,
            )
        if "代码库结构分析助手" in system or "项目笔记本" in system:
            qm = re.search(
                r"用户问题：\s*\n(.+?)\n\n(?:项目笔记本|结构扫描)证据",
                user,
                re.S,
            )
            q = (qm.group(1).strip() if qm else user[:80]).splitlines()[0][:120]
            return (
                f"（mock）已根据项目笔记本证据回答：{q}\n"
                "请以笔记本中的启动类、包、Controller、pom/框架信号为准，证据不足处不编造。"
            )
        if "话题存在性" in system or "代码库检索助手" in system:
            qm = re.search(
                r"用户问题：\s*\n(.+?)\n\n话题存在性证据",
                user,
                re.S,
            )
            q = (qm.group(1).strip() if qm else user[:80]).splitlines()[0][:120]
            return (
                f"（mock）已根据话题检索证据回答：{q}\n"
                "有源码/索引命中则确认存在并点出位置；无命中则说明未检索到。"
            )
        if "严格依据下方证据回答" in system or "依据：节点数=" in system:
            return (
                "## 流程说明（Mock Ask 检索）\n"
                "1. 入口受理请求\n"
                "2. 业务校验与编排\n"
                "3. 下游处理与结果回写\n\n"
                "依据：节点数=3"
            )
        if "tags" in blob or "business_domain" in blob or response_format:
            return json.dumps(
                {
                    "tags": ["接口", "入口"],
                    "summary": "相关方法（mock 描述）",
                    "business_domain": "接口",
                },
                ensure_ascii=False,
            )
        # explore-agent tool loop（q 跟随用户问题里的关键词）
        if "可用工具" in system or ("search_symbols" in system and "finish" in system):
            q_word = _keywords_from_user(user)[0]
            if any(m.get("role") == "user" and "工具" in (m.get("content") or "") for m in messages):
                return json.dumps(
                    {
                        "tool": "finish",
                        "args": {
                            "answer": (
                                f"## {q_word}流程（Mock 探索）\n"
                                "1. 入口 Controller 接收请求并做参数校验\n"
                                "2. Service 编排逻辑与下游校验\n"
                                "3. 沿调用图下钻可继续看校验点\n\n"
                                "（MockLLM 演示；真实模型会按问题检索对应代码）"
                            )
                        },
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {"tool": "search_symbols", "args": {"q": q_word, "kind": "Controller"}},
                ensure_ascii=False,
            )
        if "探索步骤" in system or "根据已探索步骤" in system:
            return (
                "## 流程概览（Mock）\n"
                "- 入口校验 → Service → 下游校验/落库\n"
                "- 详细步骤需真实 LLM + 已解析代码"
            )
        return (
            "## 分析结果（Mock）\n"
            f"已理解需求：{user[:300]}\n\n"
            "建议修改范围将基于检索与调用图生成。\n"
            "切换 LLM_PROVIDER=siliconflow 并配置 LLM_API_KEY 后可获得真实模型输出。"
        )

    async def chat_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        text = await self.chat(messages, temperature=temperature, max_tokens=max_tokens)
        yield text
