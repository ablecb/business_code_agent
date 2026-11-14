# Architecture alignment — zero business-domain routers

## Hard rule

Product code must not contain business-domain special cases
(放款/提现/额度校验/协议同步/单笔单批/Withdrawals*/quotaCheck keyword branches,
loan-only fast paths, hardcoded checklist method names).

## Removed / neutralized (this pass)

| Area | Action |
|------|--------|
| `question_intent.py` | Deleted domain detectors (`protocol_sync`, `composite` 机构/单笔单批, `drill_gate` theme enums, theme→method maps). Kept generic `wants_add_field_change_plan` + `wants_show_gate_code`. Compatibility stubs return False/`[]`. |
| `explore_agent.py` | Emptied `_LEXICAL_MAIN_ENTRY_RULES` (always None). Emptied spine/nested-gate/theme/primary-method maps. Deleted composite/protocol/gate_detail/source_paste **early-return short-circuits**. `_assemble_gate_detail_from_calls` no-op. Removed `loan_disburse` default, `com.demo.loan.*` package invent, product SYSTEM prompt lines. |
| `graph_agent.py` | Removed 放款里/那额度… follow-up keys. Dig-in follow-up = session lock + show-code / short dig-in words only. |
| `task_planner.py` | Removed `KIND_PROTOCOL_SYNC` / composite / gate_detail from valid kinds & heuristic templates. Planner prompt no longer lists `discover_protocol_sync_plan`. LLM plans scrubbed of 协议同步 steps. |
| `explore_tools.py` | Removed `discover_protocol_sync_plan` / `discover_composite_field_plan` from tool catalog; dispatch returns tool-removed. |
| `answer_reviewer.py` | Removed quota theme detector + fake-table denylist + product method mention list. |
| `change_impact/discover.py` | Disabled orgQuota/单笔单批 seed regexes; external-pass scoring uses Client/Feign/Map.put shapes + question tokens. |

## Kept (general)

1. Session `last_class_method` / `last_symbols` / opaque topics
2. Generic add-field change_impact by VO/Mapper/@TableName roles
3. Explore tool loop: `search_symbols` → `get_symbol_code` → `list_callees` / `resolve_impl` → `finish`
4. FOLLOWS/FlowDoc rendering from graph titles (no hardcoded Chinese label map)
5. Generic show-code phrase detection (源码/对应的代码) without gate themes

## Residual (honest)

- Some sanitizer regexes may still mention historical entity strings (e.g. DemoWithdrawals) as **anti-hallucination filters**, not routers — further deletion is safe if they still fire on domain asks.
- `discover_protocol_sync_plan` / `discover_composite_field_plan` **tools** may still exist for LLM invocation; they are no longer keyword-triggered short-circuits. Prefer shape-based discovery only.
- Project lexicon (noun→entry) belongs in config/DB, not agent code (not re-added here).

## Request path

```
User → graph_agent (session lock / search, no business enum)
     → explore_agent tool loop (+ optional generic add_field discover)
     → evidence-grounded answer
     → persist last_entry / last_symbols
```
