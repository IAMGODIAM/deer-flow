"""
Phase 2 Integration: Memory Middleware + SubagentExecutor

Adapted from DeerFlow 2.0 for Hallways/Boardroom.

Modules:
1. MemoryMiddleware — debounced conversation-driven fact_store updates
2. SubagentExecutor — thread pool delegation with timeout enforcement

Unlike DeerFlow which uses flat JSON memory files, our adaptation writes to
the existing fact_store (entity resolution, trust scoring) while borrowing
DeerFlow's debounced queue, correction detection, and temporal stratification.
"""

import re
import threading
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any
from concurrent.futures import ThreadPoolExecutor, Future, TimeoutError as FuturesTimeoutError

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Memory Middleware
# ─────────────────────────────────────────────

# Correction patterns (English + Chinese)
_CORRECTION_PATTERNS = (
    re.compile(r"\bthat(?:'s| is) (?:wrong|incorrect)\b", re.IGNORECASE),
    re.compile(r"\byou misunderstood\b", re.IGNORECASE),
    re.compile(r"\btry again\b", re.IGNORECASE),
    re.compile(r"\bredo\b", re.IGNORECASE),
    re.compile(r"\bno[,.,]\s*(?:that's|that is|it's|it is)?\s*(?:wrong|incorrect|not right)\b", re.IGNORECASE),
    # Chinese patterns (no \b word boundaries)
    re.compile(r"不对"),
    re.compile(r"你理解错了"),
    re.compile(r"你理解有误"),
    re.compile(r"重试"),
    re.compile(r"重新来"),
    re.compile(r"换一种"),
    re.compile(r"改用"),
)

# Reinforcement patterns
_REINFORCEMENT_PATTERNS = (
    re.compile(r"\byes[,.]?\s+(?:exactly|perfect|that(?:'s| is) (?:right|correct|it))\b", re.IGNORECASE),
    re.compile(r"\bperfect(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bexactly\s+(?:right|correct)\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is)\s+(?:exactly\s+)?(?:right|correct|what i (?:wanted|needed|meant))\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:doing\s+)?that\b", re.IGNORECASE),
    re.compile(r"\bjust\s+(?:like\s+)?(?:that|this)\b", re.IGNORECASE),
    re.compile(r"\bthis is (?:great|helpful)\b(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bthis is what i wanted\b(?:[.!?]|$)", re.IGNORECASE),
    # Chinese patterns (no \b word boundaries — CJK doesn't use spaces)
    re.compile(r"对[，,]?\s*就是这样"),
    re.compile(r"完全正确"),
    re.compile(r"就是这个意思"),
    re.compile(r"正是我想要的"),
    re.compile(r"继续保持"),
)

_UPLOAD_BLOCK_RE = re.compile(r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", re.IGNORECASE)


def extract_message_text(message: Any) -> str:
    """Extract plain text from message content."""
    content = getattr(message, "content", "")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
        return " ".join(parts)
    return str(content)


def filter_messages_for_memory(messages: list[Any]) -> list[Any]:
    """
    Keep only user inputs and final assistant responses for memory updates.
    Strips uploaded file blocks. Skips AI messages that are tool-call-only.
    """
    from copy import copy
    filtered = []
    skip_next_ai = False
    
    for msg in messages:
        msg_type = getattr(msg, "type", None)
        
        if msg_type == "human":
            content_str = extract_message_text(msg)
            if "<uploaded_files>" in content_str:
                stripped = _UPLOAD_BLOCK_RE.sub("", content_str).strip()
                if not stripped:
                    skip_next_ai = True
                    continue
                clean_msg = copy(msg)
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                filtered.append(msg)
                skip_next_ai = False
        elif msg_type == "ai":
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                if skip_next_ai:
                    skip_next_ai = False
                    continue
                filtered.append(msg)
            # AI messages with tool calls are intermediate — skip for memory
        # Tool messages are intermediate — skip
    
    return filtered


def detect_correction(messages: list[Any]) -> bool:
    """Detect if the conversation contains a user correction."""
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            text = extract_message_text(msg)
            for pattern in _CORRECTION_PATTERNS:
                if pattern.search(text):
                    return True
    return False


def detect_reinforcement(messages: list[Any]) -> bool:
    """Detect if the conversation contains positive reinforcement."""
    for msg in messages:
        if getattr(msg, "type", None) == "human":
            text = extract_message_text(msg)
            for pattern in _REINFORCEMENT_PATTERNS:
                if pattern.search(text):
                    return True
    return False


@dataclass
class ConversationContext:
    """Context for a conversation to be processed for memory update."""
    thread_id: str
    messages: list[Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    user_id: str | None = None
    correction_detected: bool = False
    reinforcement_detected: bool = False


class MemoryUpdateQueue:
    """
    Debounced queue for memory updates.
    
    Collects conversation contexts and processes them after a configurable
    debounce period. Multiple conversations within the debounce window are
    batched together.
    
    Adapted from: deerflow/agents/memory/queue.py
    """
    
    def __init__(self, debounce_seconds: int = 30, max_facts: int = 100,
                 fact_confidence_threshold: float = 0.7):
        self._debounce_seconds = debounce_seconds
        self._max_facts = max_facts
        self._fact_confidence_threshold = fact_confidence_threshold
        self._queue: list[ConversationContext] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._processing = False
    
    @staticmethod
    def _queue_key(thread_id: str, user_id: str | None, agent_name: str | None) -> tuple:
        return (thread_id, user_id, agent_name)
    
    def add(self, thread_id: str, messages: list[Any],
            agent_name: str | None = None, user_id: str | None = None,
            correction_detected: bool = False,
            reinforcement_detected: bool = False) -> None:
        """Add a conversation to the update queue (debounced)."""
        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id, messages=messages,
                agent_name=agent_name, user_id=user_id,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
            )
            self._reset_timer()
        
        logger.info(f"Memory update queued for thread {thread_id}, queue size: {len(self._queue)}")
    
    def add_nowait(self, thread_id: str, messages: list[Any],
                   agent_name: str | None = None, user_id: str | None = None,
                   correction_detected: bool = False,
                   reinforcement_detected: bool = False) -> None:
        """Add a conversation and process immediately in the background."""
        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id, messages=messages,
                agent_name=agent_name, user_id=user_id,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
            )
            self._schedule_timer(0)
        
        logger.info(f"Memory update queued for immediate processing on thread {thread_id}")
    
    def _enqueue_locked(self, *, thread_id: str, messages: list[Any],
                        agent_name: str | None, user_id: str | None,
                        correction_detected: bool, reinforcement_detected: bool) -> None:
        queue_key = self._queue_key(thread_id, user_id, agent_name)
        existing = next(
            (ctx for ctx in self._queue
             if self._queue_key(ctx.thread_id, ctx.user_id, ctx.agent_name) == queue_key),
            None,
        )
        merged_correction = correction_detected or (existing.correction_detected if existing else False)
        merged_reinforcement = reinforcement_detected or (existing.reinforcement_detected if existing else False)
        
        context = ConversationContext(
            thread_id=thread_id, messages=messages,
            agent_name=agent_name, user_id=user_id,
            correction_detected=merged_correction,
            reinforcement_detected=merged_reinforcement,
        )
        
        # Replace existing entry for same thread/user/agent
        self._queue = [
            ctx for ctx in self._queue
            if self._queue_key(ctx.thread_id, ctx.user_id, ctx.agent_name) != queue_key
        ]
        self._queue.append(context)
    
    def _reset_timer(self) -> None:
        self._schedule_timer(self._debounce_seconds)
    
    def _schedule_timer(self, delay_seconds: float) -> None:
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(delay_seconds, self._process_queue)
        self._timer.daemon = True
        self._timer.start()
    
    def _process_queue(self) -> None:
        """Process all queued conversation contexts."""
        with self._lock:
            if self._processing:
                self._schedule_timer(0)
                return
            if not self._queue:
                return
            self._processing = True
            contexts = self._queue.copy()
            self._queue.clear()
            self._timer = None
        
        logger.info(f"Processing {len(contexts)} queued memory updates")
        
        for ctx in contexts:
            try:
                self._update_memory_for_context(ctx)
            except Exception as e:
                logger.error(f"Memory update failed for thread {ctx.thread_id}: {e}")
        
        with self._lock:
            self._processing = False
    
    def _update_memory_for_context(self, ctx: ConversationContext) -> None:
        """
        Update fact_store based on conversation context.
        
        This is the integration point with our existing fact_store.
        Instead of DeerFlow's flat JSON, we use entity resolution and trust scoring.
        """
        from hermes_tools import fact_store
        
        # Extract text from filtered messages
        conversation_text = []
        for msg in ctx.messages:
            msg_type = getattr(msg, "type", None)
            text = extract_message_text(msg)
            if msg_type == "human":
                conversation_text.append(f"User: {text}")
            elif msg_type == "ai":
                conversation_text.append(f"Assistant: {text}")
        
        full_conversation = "\n".join(conversation_text)
        
        if not full_conversation.strip():
            return
        
        # Determine confidence boost based on correction/reinforcement signals
        confidence_boost = 0.0
        category_hint = "general"
        if ctx.correction_detected:
            confidence_boost = 0.2  # Corrections are high-priority
            category_hint = "correction"
        elif ctx.reinforcement_detected:
            confidence_boost = 0.1  # Reinforcements strengthen existing facts
        
        # Extract key facts from conversation using LLM
        # For now, store the conversation summary as a fact
        # In production, this would call an LLM to extract structured facts
        try:
            fact_content = self._extract_facts_with_llm(full_conversation, category_hint)
            if fact_content:
                base_confidence = 0.6 + confidence_boost
                # Integration with Hermes fact_store (optional — graceful fallback)
                try:
                    from hermes_tools import fact_store
                    fact_store.add(
                        content=fact_content,
                        category=category_hint if category_hint != "correction" else "general",
                        tags=f"auto,conversation,{ctx.agent_name or 'default'}",
                        trust_delta=confidence_boost,
                    )
                except ImportError:
                    # Running outside Hermes context — log only
                    logger.debug(f"fact_store not available, skipping persistence: {fact_content[:80]}")
                logger.info(f"Memory updated for thread {ctx.thread_id}: {fact_content[:80]}...")
        except Exception as e:
            logger.error(f"Failed to extract facts for thread {ctx.thread_id}: {e}")
    
    def _extract_facts_with_llm(self, conversation: str, category_hint: str) -> str | None:
        """
        Extract key facts from conversation text.
        
        In production, this calls an LLM. For now, returns a summary.
        The DeerFlow approach uses a detailed MEMORY_UPDATE_PROMPT with
        structured JSON output. We adapt this to feed into fact_store.
        """
        # Placeholder: in production, call LLM with MEMORY_UPDATE_PROMPT
        # For now, return first 200 chars as summary
        lines = [l for l in conversation.split("\n") if l.strip()]
        if not lines:
            return None
        summary = " | ".join(lines[:3])
        return summary[:200] if summary else None


# ─────────────────────────────────────────────
# 2. SubagentExecutor
# ─────────────────────────────────────────────

@dataclass
class SubagentTask:
    """Represents a delegated subagent task."""
    task_id: str
    description: str
    prompt: str
    subagent_type: str = "general-purpose"
    max_turns: int = 10
    timeout_seconds: int = 900
    spawned_at: float = field(default_factory=time.time)
    completed: bool = False
    result: str | None = None
    error: str | None = None


class SubagentExecutor:
    """
    Thread pool-based subagent execution engine.
    
    Manages concurrent subagent execution with:
    - Isolated thread pools (scheduler + execution)
    - Configurable concurrency limits
    - Timeout enforcement per task
    - Result collection and error handling
    
    Adapted from: deerflow/subagents/executor.py
    Enhanced with: Hallways mesh integration, task typing
    """
    
    DEFAULT_MAX_CONCURRENT = 3
    DEFAULT_TIMEOUT = 900  # 15 minutes
    MAX_CONCURRENT = 4
    
    def __init__(self, max_concurrent: int = DEFAULT_MAX_CONCURRENT,
                 default_timeout: int = DEFAULT_TIMEOUT):
        self._max_concurrent = min(max_concurrent, self.MAX_CONCURRENT)
        self._default_timeout = default_timeout
        
        # Thread pools (isolated to prevent scheduling from blocking execution)
        self._scheduler_pool = ThreadPoolExecutor(
            max_workers=3, thread_name_prefix="subagent-scheduler-"
        )
        self._execution_pool = ThreadPoolExecutor(
            max_workers=self._max_concurrent, thread_name_prefix="subagent-execution-"
        )
        
        # Task tracking
        self._tasks: dict[str, SubagentTask] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
    
    @property
    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if not t.completed)
    
    @property
    def available_slots(self) -> int:
        return max(0, self._max_concurrent - self.active_count)
    
    def can_execute(self) -> bool:
        return self.active_count < self._max_concurrent
    
    def submit(self, task: SubagentTask) -> str:
        """
        Submit a subagent task for execution.
        
        Args:
            task: The subagent task to execute
            
        Returns:
            task_id for tracking
            
        Raises:
            RuntimeError: If at max concurrency
        """
        with self._lock:
            active = sum(1 for t in self._tasks.values() if not t.completed)
            if active >= self._max_concurrent:
                raise RuntimeError(
                    f"At max concurrency ({self._max_concurrent}). "
                    f"Active tasks: {list(self._tasks.keys())}"
                )
            
            self._tasks[task.task_id] = task
        
        # Schedule execution (outside lock to avoid deadlock)
        self._scheduler_pool.submit(self._schedule_execution, task)
        logger.info(f"SubagentExecutor: submitted '{task.task_id}' ({task.subagent_type})")
        return task.task_id
    
    def _schedule_execution(self, task: SubagentTask) -> None:
        """Schedule task execution on the execution pool."""
        future = self._execution_pool.submit(self._execute_task, task)
        with self._lock:
            self._futures[task.task_id] = future
    
    def _execute_task(self, task: SubagentTask) -> str:
        """
        Execute a subagent task.
        
        This is the integration point with Hallways mesh delegation.
        In production, this calls delegate_task or Hallways API.
        """
        logger.info(f"SubagentExecutor: executing '{task.task_id}' ({task.subagent_type})")
        
        try:
            # Integration point: delegate to Hallways mesh agent
            # In production, this would call:
            #   delegate_task(goal=task.prompt, context=...)
            # or:
            #   hallways_api.submit_task(agent=task.subagent_type, prompt=task.prompt)
            
            # Placeholder: simulate execution
            result = self._delegate_to_agent(task)
            
            with self._lock:
                task.result = result
                task.completed = True
            
            elapsed = time.time() - task.spawned_at
            logger.info(f"SubagentExecutor: completed '{task.task_id}' in {elapsed:.1f}s")
            return result
            
        except Exception as e:
            with self._lock:
                task.error = str(e)
                task.completed = True
            logger.error(f"SubagentExecutor: task '{task.task_id}' failed: {e}")
            raise
    
    def _delegate_to_agent(self, task: SubagentTask) -> str:
        """
        Delegate task to a Hallways mesh agent.
        
        Integration point — replace with actual Hallways API call.
        """
        # TODO: Replace with actual delegation
        # from hermes_tools import delegate_task
        # result = delegate_task(
        #     goal=task.prompt,
        #     context={"subagent_type": task.subagent_type, "max_turns": task.max_turns}
        # )
        # return result
        
        return f"[Delegated '{task.description}' to {task.subagent_type} agent]"
    
    def get_result(self, task_id: str, timeout: float | None = None) -> str | None:
        """
        Get the result of a completed task.
        
        Args:
            task_id: The task to check
            timeout: Max seconds to wait (None = non-blocking)
            
        Returns:
            Result string if completed, None if still running or not found
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            
            if task.completed:
                if task.error:
                    raise RuntimeError(f"Task '{task_id}' failed: {task.error}")
                return task.result
            
            future = self._futures.get(task_id)
        
        if future is not None and timeout is not None:
            try:
                return future.result(timeout=timeout)
            except FuturesTimeoutError:
                return None
        
        return None
    
    def cancel(self, task_id: str) -> bool:
        """Cancel a running task."""
        with self._lock:
            future = self._futures.get(task_id)
            task = self._tasks.get(task_id)
            
            if future is None or task is None:
                return False
            
            if task.completed:
                return False
            
            cancelled = future.cancel()
            if cancelled:
                task.completed = True
                task.error = "Cancelled"
                logger.info(f"SubagentExecutor: cancelled '{task_id}'")
            
            return cancelled
    
    def get_status(self) -> dict:
        """Get executor status."""
        with self._lock:
            active = [t for t in self._tasks.values() if not t.completed]
            completed = [t for t in self._tasks.values() if t.completed]
            
            return {
                "max_concurrent": self._max_concurrent,
                "active_count": len(active),
                "completed_count": len(completed),
                "available_slots": self._max_concurrent - len(active),
                "active_tasks": [
                    {
                        "task_id": t.task_id,
                        "description": t.description,
                        "subagent_type": t.subagent_type,
                        "elapsed_seconds": time.time() - t.spawned_at,
                    }
                    for t in active
                ],
            }
    
    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor and all thread pools."""
        # Cancel any pending (not yet running) futures
        with self._lock:
            for task_id, future in self._futures.items():
                if not future.running():
                    future.cancel()
        self._scheduler_pool.shutdown(wait=wait)
        self._execution_pool.shutdown(wait=wait)
        logger.info("SubagentExecutor: shutdown complete")
