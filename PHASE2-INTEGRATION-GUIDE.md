# Phase 2 Integration Guide: Applying to Boardroom

**Date:** 2026-05-18  
**Status:** Code complete, tested, pushed to IAMGODIAM/deer-flow

---

## What Was Built

### 1. Memory Middleware (`MemoryUpdateQueue` + helpers)

**File:** `phase2_integration.py`

**Integration point:** Boardroom agent runtime — hook into `after_agent` callback.

```python
# In Boardroom agent setup:
from phase2_integration import (
    MemoryUpdateQueue, filter_messages_for_memory,
    detect_correction, detect_reinforcement,
)

memory_queue = MemoryUpdateQueue(debounce_seconds=30)

def after_agent_hook(state, runtime):
    """Called after each agent execution."""
    messages = state.get("messages", [])
    filtered = filter_messages_for_memory(messages)
    
    if not filtered:
        return
    
    correction = detect_correction(filtered)
    reinforcement = detect_reinforcement(filtered) if not correction else False
    
    thread_id = runtime.context.get("thread_id", "default")
    memory_queue.add(
        thread_id=thread_id,
        messages=filtered,
        agent_name=runtime.context.get("agent_name"),
        user_id=runtime.context.get("user_id"),
        correction_detected=correction,
        reinforcement_detected=reinforcement,
    )
```

**Key behaviors:**
- Debounces updates (30s window) to avoid flooding fact_store
- Corrections get +0.2 confidence boost (high priority)
- Reinforcements get +0.1 trust delta (strengthen existing)
- Chinese + English pattern matching for both signals
- Writes to existing `fact_store` (not a parallel system)

### 2. SubagentExecutor

**File:** `phase2_integration.py`

**Integration point:** Replace direct `delegate_task` calls with structured execution.

```python
# In Boardroom task delegation:
from phase2_integration import SubagentExecutor, SubagentTask

executor = SubagentExecutor(max_concurrent=3, default_timeout=900)

# Instead of direct delegate_task:
task = SubagentTask(
    task_id=f"research-{uuid.uuid4().hex[:8]}",
    description="Research deer-flow architecture",
    prompt="Analyze the deer-flow GitHub repo...",
    subagent_type="general-purpose",
    timeout_seconds=900,
)

task_id = executor.submit(task)

# Non-blocking result check:
result = executor.get_result(task_id)  # None if still running
result = executor.get_result(task_id, timeout=60)  # Wait up to 60s

# Status monitoring:
status = executor.get_status()
# {"max_concurrent": 3, "active_count": 1, "available_slots": 2, ...}
```

**Key behaviors:**
- Dual thread pools (scheduler + execution) prevent scheduling from blocking
- Max 4 concurrent subagents (configurable, clamped)
- 15-minute default timeout per task
- Cancellation support
- Status reporting for monitoring dashboard

---

## Applying to Boardroom

### Step 1: Add Memory Middleware Hook

In Boardroom's agent runtime (`boardroom/agents/`), add the `after_agent` hook:

1. Import `MemoryUpdateQueue` and helpers
2. Create a single queue instance (shared across agents)
3. Register the hook in agent middleware chain
4. The queue auto-processes on a timer — no manual triggering needed

### Step 2: Replace Direct Delegation with SubagentExecutor

In Boardroom's task delegation logic:

1. Create a `SubagentExecutor` instance at module level
2. Replace `delegate_task()` calls with `executor.submit(SubagentTask(...))`
3. Use `executor.get_result()` for synchronous waits
4. Add `executor.get_status()` to the monitoring endpoint

### Step 3: Wire Skills Tool Policy (Phase 1)

In Boardroom's skill loader:

1. Parse `allowed-tools` from SKILL.md frontmatter
2. Create `SkillsToolPolicy` from active skills
3. Filter agent tools at creation time: `policy.filter_tools(tools)`

---

## Testing

All tests pass:
- Phase 1: 24 tests (SkillsToolPolicy, ConcurrencyController, ToolCallTruncator)
- Phase 2: 30 tests (MemoryMiddleware, SubagentExecutor)

```bash
python -m pytest test_phase1_integration.py test_phase2_integration.py -v
# 54 passed
```

---

## Next: Phase 3

Phase 3 (Month 2):
- Summarization offload for long-running mesh tasks
- Formalize Boardroom agent middleware pipeline
- Integration testing with live Hallways mesh
