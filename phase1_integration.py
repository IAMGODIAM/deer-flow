"""
Phase 1 Integration: Skills Tool Policy + Subagent Limit Middleware

Adapted from DeerFlow 2.0 (bytedance/deer-flow) for Hallways/Boardroom.

This module provides:
1. SkillsToolPolicy — allowed-tools gating per skill at agent creation time
2. ConcurrencyController — subagent spawn limiting with configurable thresholds

Usage:
    from phase1_integration import SkillsToolPolicy, ConcurrencyController
    
    # Skills tool policy
    policy = SkillsToolPolicy(skills)
    allowed_tools = policy.filter_tools(available_tools)
    
    # Concurrency control
    controller = ConcurrencyController(max_concurrent=3, timeout_seconds=900)
    controller.can_spawn()  # True/False
    controller.spawn(task_id)  # Register a spawn
    controller.complete(task_id)  # Mark complete
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# 1. Skills Tool Policy
# ─────────────────────────────────────────────

class NamedTool(Protocol):
    """Protocol for any tool with a name attribute."""
    name: str


@dataclass
class SkillPolicy:
    """Represents a skill's tool access policy."""
    name: str
    allowed_tools: list[str] | None  # None = allow all (legacy), [] = deny all
    
    @property
    def has_explicit_declaration(self) -> bool:
        return self.allowed_tools is not None


class SkillsToolPolicy:
    """
    Filters available tools based on active skills' allowed-tools declarations.
    
    Behavior:
    - If NO skill declares allowed-tools → all tools allowed (backward compatible)
    - If ANY skill declares allowed-tools → only tools in the union are allowed
    - Skills without the declaration contribute NO tools (not all tools)
    - Empty allowed-tools list = skill provides context only, no tool access
    
    Adapted from: deerflow/skills/tool_policy.py
    """
    
    def __init__(self, skills: list[SkillPolicy]):
        self._skills = skills
        self._allowed: set[str] | None = self._compute_allowed()
    
    def _compute_allowed(self) -> set[str] | None:
        """Compute the union of all skill allowed-tools declarations."""
        if not self._skills:
            return None
        
        allowed: set[str] = set()
        has_explicit = False
        
        for skill in self._skills:
            if skill.allowed_tools is None:
                continue
            has_explicit = True
            if not skill.allowed_tools:
                logger.info(f"Skill '{skill.name}' declared empty allowed-tools (context-only)")
            allowed.update(skill.allowed_tools)
        
        if not has_explicit:
            return None  # Legacy: no restrictions
        return allowed
    
    @property
    def is_restricted(self) -> bool:
        """Whether tool filtering is active."""
        return self._allowed is not None
    
    @property
    def allowed_tool_names(self) -> set[str] | None:
        """The set of allowed tool names, or None if unrestricted."""
        return self._allowed
    
    def filter_tools(self, tools: list[Any]) -> list[Any]:
        """
        Filter tools to only those allowed by active skill policies.
        
        Args:
            tools: List of tool objects with a 'name' attribute
            
        Returns:
            Filtered list of tools
        """
        if self._allowed is None:
            return tools
        
        filtered = [t for t in tools if getattr(t, 'name', None) in self._allowed]
        dropped = len(tools) - len(filtered)
        if dropped > 0:
            logger.info(f"SkillsToolPolicy: filtered out {dropped} tools ({len(filtered)} remain)")
        return filtered
    
    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a specific tool is allowed."""
        if self._allowed is None:
            return True
        return tool_name in self._allowed


# ─────────────────────────────────────────────
# 2. Concurrency Controller (Subagent Limit)
# ─────────────────────────────────────────────

@dataclass
class SubagentSlot:
    """Tracks a single subagent execution slot."""
    task_id: str
    spawned_at: float = field(default_factory=time.time)
    completed: bool = False
    completed_at: float | None = None


class ConcurrencyController:
    """
    Controls concurrent subagent spawning with timeout enforcement.
    
    Adapted from: deerflow/agents/middlewares/subagent_limit_middleware.py
    Enhanced with: timeout tracking, slot management, graceful degradation
    
    Usage:
        controller = ConcurrencyController(max_concurrent=3, timeout_seconds=900)
        
        if controller.can_spawn():
            slot = controller.spawn(task_id="research-1")
            # ... execute subagent ...
            controller.complete(task_id="research-1")
    """
    
    MIN_CONCURRENT = 1
    MAX_CONCURRENT = 4
    DEFAULT_TIMEOUT = 900  # 15 minutes
    
    def __init__(self, max_concurrent: int = 3, timeout_seconds: int = DEFAULT_TIMEOUT):
        self._max = self._clamp(max_concurrent)
        self._timeout = timeout_seconds
        self._slots: dict[str, SubagentSlot] = {}
        self._lock = threading.Lock()
    
    @staticmethod
    def _clamp(value: int) -> int:
        return max(ConcurrencyController.MIN_CONCURRENT, 
                   min(ConcurrencyController.MAX_CONCURRENT, value))
    
    @property
    def max_concurrent(self) -> int:
        return self._max
    
    @property
    def active_count(self) -> int:
        with self._lock:
            self._cleanup_expired()
            return sum(1 for s in self._slots.values() if not s.completed)
    
    @property
    def available_slots(self) -> int:
        return max(0, self._max - self.active_count)
    
    def can_spawn(self) -> bool:
        """Check if a new subagent can be spawned."""
        return self.active_count < self._max
    
    def spawn(self, task_id: str) -> SubagentSlot | None:
        """
        Register a new subagent spawn.
        
        Args:
            task_id: Unique identifier for this subagent task
            
        Returns:
            SubagentSlot if spawned successfully, None if at capacity
        """
        with self._lock:
            self._cleanup_expired()
            
            if len([s for s in self._slots.values() if not s.completed]) >= self._max:
                logger.warning(f"ConcurrencyController: at capacity ({self._max}), cannot spawn '{task_id}'")
                return None
            
            if task_id in self._slots and not self._slots[task_id].completed:
                logger.warning(f"ConcurrencyController: task '{task_id}' already running")
                return self._slots[task_id]
            
            slot = SubagentSlot(task_id=task_id)
            self._slots[task_id] = slot
            active = sum(1 for s in self._slots.values() if not s.completed)
            logger.info(f"ConcurrencyController: spawned '{task_id}' ({active}/{self._max} active)")
            return slot
    
    def complete(self, task_id: str) -> bool:
        """Mark a subagent as completed."""
        with self._lock:
            if task_id not in self._slots:
                logger.warning(f"ConcurrencyController: unknown task '{task_id}'")
                return False
            
            slot = self._slots[task_id]
            slot.completed = True
            slot.completed_at = time.time()
            duration = slot.completed_at - slot.spawned_at
            logger.info(f"ConcurrencyController: completed '{task_id}' in {duration:.1f}s")
            return True
    
    def _cleanup_expired(self):
        """Remove expired (timed out) slots. Must be called with _lock held."""
        now = time.time()
        expired = [
            tid for tid, slot in self._slots.items()
            if not slot.completed and (now - slot.spawned_at) > self._timeout
        ]
        for tid in expired:
            logger.warning(f"ConcurrencyController: task '{tid}' timed out after {self._timeout}s")
            self._slots[tid].completed = True
            self._slots[tid].completed_at = now
    
    def get_status(self) -> dict:
        """Get current controller status."""
        with self._lock:
            self._cleanup_expired()
            active = [s for s in self._slots.values() if not s.completed]
            return {
                "max_concurrent": self._max,
                "active_count": len(active),
                "available_slots": self._max - len(active),
                "timeout_seconds": self._timeout,
                "active_tasks": [
                    {
                        "task_id": s.task_id,
                        "elapsed_seconds": time.time() - s.spawned_at
                    }
                    for s in active
                ]
            }


# ─────────────────────────────────────────────
# 3. Tool Call Truncator (Middleware Pattern)
# ─────────────────────────────────────────────

class ToolCallTruncator:
    """
    Truncates excess 'task' tool calls from a single model response.
    
    When an LLM generates more than max_concurrent parallel task tool calls
    in one response, this keeps only the first max_concurrent and drops the rest.
    More reliable than prompt-based limits.
    
    Adapted from: deerflow/agents/middlewares/subagent_limit_middleware.py
    """
    
    def __init__(self, max_concurrent: int = 3, task_tool_name: str = "task"):
        self._max = ConcurrencyController._clamp(max_concurrent)
        self._task_tool_name = task_tool_name
    
    def truncate(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Truncate excess task tool calls beyond the concurrency limit.
        
        Args:
            tool_calls: List of tool call dicts from model response
            
        Returns:
            Truncated list of tool calls
        """
        task_indices = [
            i for i, tc in enumerate(tool_calls)
            if tc.get("name") == self._task_tool_name
        ]
        
        if len(task_indices) <= self._max:
            return tool_calls
        
        indices_to_drop = set(task_indices[self._max:])
        truncated = [tc for i, tc in enumerate(tool_calls) if i not in indices_to_drop]
        
        logger.info(
            f"ToolCallTruncator: dropped {len(indices_to_drop)} excess '{self._task_tool_name}' calls "
            f"(kept {self._max}/{len(task_indices)})"
        )
        return truncated
