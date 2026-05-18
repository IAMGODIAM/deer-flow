"""
Tests for Phase 1 Integration: Skills Tool Policy + Concurrency Controller

Run: python -m pytest test_phase1_integration.py -v
"""

import time
import threading
import pytest
from unittest.mock import MagicMock

from phase1_integration import (
    SkillsToolPolicy,
    SkillPolicy,
    ConcurrencyController,
    ToolCallTruncator,
)


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────

def make_tool(name: str) -> MagicMock:
    """Create a mock tool with a name attribute."""
    tool = MagicMock()
    tool.name = name
    return tool


SAMPLE_TOOLS = [
    make_tool("web_search"),
    make_tool("web_fetch"),
    make_tool("read_file"),
    make_tool("write_file"),
    make_tool("bash"),
    make_tool("task"),
    make_tool("memory_search"),
]


# ─────────────────────────────────────────────
# Skills Tool Policy Tests
# ─────────────────────────────────────────────

class TestSkillsToolPolicy:
    
    def test_no_skills_allows_all(self):
        """No skills = all tools allowed (backward compatible)."""
        policy = SkillsToolPolicy([])
        assert not policy.is_restricted
        assert policy.filter_tools(SAMPLE_TOOLS) == SAMPLE_TOOLS
    
    def test_skills_without_declaration_allows_all(self):
        """Skills without allowed-tools = all tools allowed (legacy)."""
        skills = [
            SkillPolicy(name="research", allowed_tools=None),
            SkillPolicy(name="coding", allowed_tools=None),
        ]
        policy = SkillsToolPolicy(skills)
        assert not policy.is_restricted
        assert policy.filter_tools(SAMPLE_TOOLS) == SAMPLE_TOOLS
    
    def test_single_skill_restricts(self):
        """Single skill with allowed-tools restricts to that set."""
        skills = [
            SkillPolicy(name="research", allowed_tools=["web_search", "web_fetch", "read_file"]),
        ]
        policy = SkillsToolPolicy(skills)
        assert policy.is_restricted
        filtered = policy.filter_tools(SAMPLE_TOOLS)
        assert len(filtered) == 3
        assert {t.name for t in filtered} == {"web_search", "web_fetch", "read_file"}
    
    def test_multiple_skills_union(self):
        """Multiple skills = union of allowed tools."""
        skills = [
            SkillPolicy(name="research", allowed_tools=["web_search", "web_fetch"]),
            SkillPolicy(name="coding", allowed_tools=["read_file", "write_file", "bash"]),
        ]
        policy = SkillsToolPolicy(skills)
        filtered = policy.filter_tools(SAMPLE_TOOLS)
        assert {t.name for t in filtered} == {"web_search", "web_fetch", "read_file", "write_file", "bash"}
    
    def test_empty_allowed_tools_denies_all(self):
        """Empty allowed-tools list = no tool access (context-only skill)."""
        skills = [
            SkillPolicy(name="style-guide", allowed_tools=[]),
        ]
        policy = SkillsToolPolicy(skills)
        assert policy.is_restricted
        assert policy.filter_tools(SAMPLE_TOOLS) == []
    
    def test_mixed_explicit_and_legacy_skills(self):
        """When any skill declares allowed-tools, legacy skills contribute nothing."""
        skills = [
            SkillPolicy(name="research", allowed_tools=["web_search"]),
            SkillPolicy(name="legacy-skill", allowed_tools=None),  # No declaration
        ]
        policy = SkillsToolPolicy(skills)
        assert policy.is_restricted
        filtered = policy.filter_tools(SAMPLE_TOOLS)
        assert len(filtered) == 1
        assert filtered[0].name == "web_search"
    
    def test_is_tool_allowed(self):
        """Check individual tool permission."""
        skills = [
            SkillPolicy(name="research", allowed_tools=["web_search", "web_fetch"]),
        ]
        policy = SkillsToolPolicy(skills)
        assert policy.is_tool_allowed("web_search") is True
        assert policy.is_tool_allowed("bash") is False
    
    def test_is_tool_allowed_unrestricted(self):
        """Unrestricted policy allows all tools."""
        policy = SkillsToolPolicy([])
        assert policy.is_tool_allowed("anything") is True


# ─────────────────────────────────────────────
# Concurrency Controller Tests
# ─────────────────────────────────────────────

class TestConcurrencyController:
    
    def test_default_settings(self):
        controller = ConcurrencyController()
        assert controller.max_concurrent == 3
        assert controller.active_count == 0
        assert controller.can_spawn() is True
    
    def test_clamp_min(self):
        controller = ConcurrencyController(max_concurrent=0)
        assert controller.max_concurrent == 1
    
    def test_clamp_max(self):
        controller = ConcurrencyController(max_concurrent=10)
        assert controller.max_concurrent == 4
    
    def test_spawn_and_complete(self):
        controller = ConcurrencyController(max_concurrent=2)
        
        slot1 = controller.spawn("task-1")
        assert slot1 is not None
        assert controller.active_count == 1
        
        slot2 = controller.spawn("task-2")
        assert slot2 is not None
        assert controller.active_count == 2
        
        # At capacity
        slot3 = controller.spawn("task-3")
        assert slot3 is None
        
        # Complete one
        controller.complete("task-1")
        assert controller.active_count == 1
        assert controller.can_spawn() is True
    
    def test_available_slots(self):
        controller = ConcurrencyController(max_concurrent=3)
        assert controller.available_slots == 3
        
        controller.spawn("task-1")
        assert controller.available_slots == 2
        
        controller.spawn("task-2")
        controller.spawn("task-3")
        assert controller.available_slots == 0
    
    def test_get_status(self):
        controller = ConcurrencyController(max_concurrent=3, timeout_seconds=600)
        controller.spawn("task-1")
        
        status = controller.get_status()
        assert status["max_concurrent"] == 3
        assert status["active_count"] == 1
        assert status["available_slots"] == 2
        assert status["timeout_seconds"] == 600
        assert len(status["active_tasks"]) == 1
        assert status["active_tasks"][0]["task_id"] == "task-1"
    
    def test_duplicate_spawn_returns_existing(self):
        controller = ConcurrencyController(max_concurrent=3)
        slot1 = controller.spawn("task-1")
        slot2 = controller.spawn("task-1")
        assert slot1 is slot2
        assert controller.active_count == 1
    
    def test_complete_unknown_task(self):
        controller = ConcurrencyController()
        assert controller.complete("nonexistent") is False
    
    def test_timeout_cleanup(self):
        controller = ConcurrencyController(max_concurrent=3, timeout_seconds=1)
        controller.spawn("task-1")
        assert controller.active_count == 1
        
        time.sleep(1.5)
        # Next call should clean up expired
        assert controller.active_count == 0
    
    def test_concurrent_spawn_thread_safety(self):
        controller = ConcurrencyController(max_concurrent=2)
        results = []
        errors = []
        
        def spawn_task(task_id):
            try:
                slot = controller.spawn(task_id)
                results.append((task_id, slot is not None))
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=spawn_task, args=(f"task-{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        # At most 2 should have succeeded
        successful = sum(1 for _, success in results if success)
        assert successful <= 2


# ─────────────────────────────────────────────
# Tool Call Truncator Tests
# ─────────────────────────────────────────────

class TestToolCallTruncator:
    
    def test_no_truncation_needed(self):
        truncator = ToolCallTruncator(max_concurrent=3)
        calls = [
            {"name": "task", "args": {"prompt": "research A"}},
            {"name": "task", "args": {"prompt": "research B"}},
        ]
        result = truncator.truncate(calls)
        assert len(result) == 2
    
    def test_truncate_excess(self):
        truncator = ToolCallTruncator(max_concurrent=2)
        calls = [
            {"name": "task", "args": {"prompt": "research A"}},
            {"name": "task", "args": {"prompt": "research B"}},
            {"name": "task", "args": {"prompt": "research C"}},
            {"name": "task", "args": {"prompt": "research D"}},
        ]
        result = truncator.truncate(calls)
        assert len(result) == 2
        assert result[0]["args"]["prompt"] == "research A"
        assert result[1]["args"]["prompt"] == "research B"
    
    def test_preserves_non_task_calls(self):
        truncator = ToolCallTruncator(max_concurrent=1)
        calls = [
            {"name": "web_search", "args": {"query": "test"}},
            {"name": "task", "args": {"prompt": "research A"}},
            {"name": "task", "args": {"prompt": "research B"}},
            {"name": "read_file", "args": {"path": "/tmp/file"}},
        ]
        result = truncator.truncate(calls)
        assert len(result) == 3  # web_search + 1 task + read_file
        names = [c["name"] for c in result]
        assert "web_search" in names
        assert "read_file" in names
        assert names.count("task") == 1
    
    def test_no_task_calls(self):
        truncator = ToolCallTruncator(max_concurrent=2)
        calls = [
            {"name": "web_search", "args": {"query": "test"}},
            {"name": "read_file", "args": {"path": "/tmp/file"}},
        ]
        result = truncator.truncate(calls)
        assert len(result) == 2
    
    def test_empty_calls(self):
        truncator = ToolCallTruncator(max_concurrent=3)
        assert truncator.truncate([]) == []
    
    def test_custom_task_tool_name(self):
        truncator = ToolCallTruncator(max_concurrent=1, task_tool_name="delegate")
        calls = [
            {"name": "delegate", "args": {"prompt": "A"}},
            {"name": "delegate", "args": {"prompt": "B"}},
            {"name": "task", "args": {"prompt": "C"}},
        ]
        result = truncator.truncate(calls)
        assert len(result) == 2  # 1 delegate + 1 task (different name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
