# DeerFlow 2.0 → Hallways/Boardroom Integration Analysis

**Date:** 2026-05-18  
**Analyst:** Hermie (Visionary Layer)  
**Source:** bytedance/deer-flow v2.0 (67.9k ⭐)  
**Fork:** github.com/IAMGODIAM/deer-flow

---

## Executive Summary

DeerFlow 2.0 is a well-engineered super agent harness built on LangGraph. We don't need the framework itself — we run our own stack (Hallways mesh + Boardroom + Hermes agents). But **5 patterns** are worth extracting and adapting:

| # | Pattern | Value | Effort |
|---|---------|-------|--------|
| 1 | Sub-agent executor with thread pools | **HIGH** — our mesh delegation lacks structured concurrency | Medium |
| 2 | Memory middleware with debounced updates | **HIGH** — our fact_store has no automatic conversation-driven updates | Medium |
| 3 | Skills tool policy (allowed-tools gating) | **MEDIUM** — our skills load but don't gate tool access | Low |
| 4 | Middleware chain architecture | **MEDIUM** — patterns we can adapt for Boardroom agent middleware | Low |
| 5 | Summarization with filesystem offload | **MEDIUM** — context management for long-running mesh tasks | Medium |

---

## 1. Sub-Agent Executor Pattern

### What DeerFlow Does

```
Lead Agent
  └── task() tool → SubagentExecutor
        ├── _scheduler_pool (3 workers) — orchestration
        ├── _execution_pool (3 workers) — actual execution  
        └── _isolated_loop_pool (3 workers) — sync calls from running loops
```

Key design decisions:
- **3 isolated thread pools** prevent scheduling from blocking execution
- **SubagentLimitMiddleware** enforces max 4 concurrent `task()` calls per response (hard truncate)
- Subagents get `subagent_enabled=False` — no recursive nesting
- **15-minute timeout** per subagent with polling-based result collection
- Token usage attributed back to parent via `_subagent_usage_cache`

### What We Should Adapt

Our Hallways mesh already has multi-node delegation (Hermie → Sue/Scout/Forge/Scribe/Draco). But we lack:

1. **Structured concurrency control** — no limit on parallel sub-agent spawns
2. **Thread pool isolation** — our `delegate_task` is synchronous, blocks the caller
3. **Timeout enforcement** — no per-subagent timeout with graceful degradation
4. **Token attribution** — sub-agent costs aren't tracked back to the delegating task

### Recommendation

Create a `SubagentExecutor` class for the Hallways Boardroom that:
- Wraps `delegate_task` calls in a thread pool (max 3 concurrent)
- Enforces configurable timeout per sub-agent type
- Tracks token/cost attribution back to parent task
- Prevents recursive delegation (leaf agents can't spawn more agents)

**File reference:** `backend/packages/harness/deerflow/subagents/executor.py`

---

## 2. Memory Middleware with Debounced Updates

### What DeerFlow Does

```
MemoryMiddleware (after_agent hook)
  └── Filters messages (user + assistant only, no tool calls)
      └── Detects corrections vs reinforcements
          └── MemoryUpdateQueue.add() (debounced)
              └── Timer-based batching
                  └── LLM summarization → JSON memory file
```

Memory structure:
```json
{
  "version": "1.0",
  "user": {
    "workContext": {"summary": "", "updatedAt": ""},
    "personalContext": {"summary": "", "updatedAt": ""},
    "topOfMind": {"summary": "", "updatedAt": ""}
  },
  "history": {
    "recentMonths": {"summary": "", "updatedAt": ""},
    "earlierContext": {"summary": "", "updatedAt": ""},
    "longTermBackground": {"summary": "", "updatedAt": ""}
  },
  "facts": []
}
```

Key design decisions:
- **Debounced queue** — batches multiple conversation turns before updating
- **Correction detection** — user corrections get high-confidence (0.9+) fact priority
- **Reinforcement detection** — positive signals strengthen existing facts
- **File-based storage** with per-user/per-agent isolation
- **Thread-safe caching** with mtime-based invalidation

### What We Should Adapt

Our `fact_store` has entity resolution and trust scoring (superior to DeerFlow's flat facts array). But we lack:

1. **Automatic conversation-driven updates** — facts are only added manually or via explicit tool calls
2. **Debounced batching** — every conversation turn doesn't need a memory update
3. **Correction detection** — when Israel corrects us, that should auto-boost fact confidence
4. **Temporal context layers** — our facts don't have "recent vs long-term" stratification

### Recommendation

Add a `MemoryMiddleware` to the Boardroom agent runtime that:
- Hooks into `after_agent` (post-execution)
- Filters to user + assistant messages only
- Detects corrections (user said "no, that's wrong" → high-priority fact update)
- Debounces updates (30-second window)
- Writes to our existing `fact_store` (not a parallel system)
- Uses temporal stratification: `topOfMind` → active session, `recentMonths` → rolling window, `longTermBackground` → stable facts

**File references:**
- `backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py`
- `backend/packages/harness/deerflow/agents/memory/queue.py`
- `backend/packages/harness/deerflow/agents/memory/prompt.py`

---

## 3. Skills Tool Policy (allowed-tools gating)

### What DeerFlow Does

Each skill has optional `allowed-tools` in frontmatter:
```yaml
---
name: research
description: Deep research and analysis
allowed-tools: [web_search, web_fetch, read_file, write_file]
---
```

At runtime:
```python
filter_tools_by_skill_allowed_tools(tools, active_skills)
# Returns only tools in the union of all active skill allowed-tools
```

Key design decisions:
- **None = allow-all** (backward compatible)
- **Empty list = no tools** (explicit denial)
- **Union of all active skills** (additive, not restrictive)
- Applied at agent creation time, not per-call

### What We Should Adapt

Our Hermes skills system loads SKILL.md files progressively but has **no tool gating**. Any skill can invoke any tool. This is a security and focus issue.

### Recommendation

Add `allowed-tools` frontmatter parsing to our skill loader:
- Parse `allowed-tools` from SKILL.md frontmatter
- When a skill is active, filter available tools to the union
- Skills without `allowed-tools` declaration → all tools (backward compatible)
- Empty `allowed-tools: []` → skill provides context only, no tool access

**File reference:** `backend/packages/harness/deerflow/skills/tool_policy.py`

---

## 4. Middleware Chain Architecture

### What DeerFlow Does

```
Request → Nginx → Gateway → Middleware Chain → Agent Core
                                    │
                                    ├── 1. ThreadDataMiddleware (paths)
                                    ├── 2. UploadsMiddleware (files)
                                    ├── 3. SandboxMiddleware (env)
                                    ├── 4. SummarizationMiddleware (context)
                                    ├── 5. TitleMiddleware (auto-title)
                                    ├── 6. TodoListMiddleware (plan mode)
                                    ├── 7. ViewImageMiddleware (vision)
                                    └── 8. ClarificationMiddleware (clarify)
```

Each middleware implements:
- `before_agent(state, runtime) → state modifications`
- `after_agent(state, runtime) → side effects`

### What We Should Adapt

Boardroom's agent runtime is simpler. We should add:

1. **ThreadDataMiddleware** — initialize workspace paths per thread
2. **SummarizationMiddleware** — context compression for long conversations
3. **MemoryMiddleware** — conversation-driven memory updates (see #2)
4. **SubagentLimitMiddleware** — concurrency control (see #1)

We don't need sandbox middleware (we use real execution environments) or clarification middleware (Hermie handles that via `clarify` tool).

---

## 5. Summarization with Filesystem Offload

### What DeerFlow Does

When context approaches token limit:
1. `SummarizationMiddleware` fires `BeforeSummarizationHook`
2. Messages to be removed are summarized via LLM
3. Summary is written to filesystem
4. Original messages replaced with `RemoveMessage` + summary pointer
5. Agent continues with compressed context

### What We Should Adapt

For long-running mesh tasks (research, code reviews, multi-step operations):
- Summarize completed sub-agent results to filesystem
- Keep only summaries in context window
- Full results available on disk for drill-down

---

## What NOT to Integrate

| Component | Reason |
|-----------|--------|
| LangGraph runtime | We use Hermes/Hallways, not LangGraph |
| Next.js frontend | We use Telegram + Boardroom console |
| Docker/K8s sandbox | We execute on real machines (MC, Mac, Azure) |
| DeerFlow config.yaml | We use Hermes config + Boardroom config |
| DeerFlow memory JSON format | Our fact_store is superior (entity resolution, trust scores) |
| ACP agent tool | We use delegate_task + Hallways API |

---

## Integration Priority

### Phase 1 (This Week)
1. **Skills tool policy** — add `allowed-tools` parsing to skill loader (low effort, high security value)
2. **SubagentLimitMiddleware** — add concurrency control to Boardroom task delegation

### Phase 2 (Next 2 Weeks)
3. **Memory middleware** — conversation-driven fact_store updates with debouncing
4. **SubagentExecutor** — structured thread pool delegation with timeouts

### Phase 3 (Month 2)
5. **Summarization offload** — context compression for long-running mesh tasks
6. **Middleware chain** — formalize Boardroom agent middleware pipeline

---

## File Mapping

| DeerFlow Source | Our Target |
|----------------|------------|
| `subagents/executor.py` | `boardroom/agents/subagent_executor.py` |
| `tools/builtins/task_tool.py` | `boardroom/tools/delegate_tool.py` |
| `agents/middlewares/memory_middleware.py` | `boardroom/middlewares/memory_middleware.py` |
| `agents/memory/queue.py` | `boardroom/memory/update_queue.py` |
| `agents/memory/prompt.py` | `boardroom/memory/update_prompts.py` |
| `skills/tool_policy.py` | `hermes/skills/tool_policy.py` |
| `agents/middlewares/summarization_middleware.py` | `boardroom/middlewares/summarization.py` |
