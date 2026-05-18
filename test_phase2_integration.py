"""
Tests for Phase 2 Integration: Memory Middleware + SubagentExecutor

Run: python -m pytest test_phase2_integration.py -v
"""

import time
import threading
import pytest
from unittest.mock import MagicMock, patch

from phase2_integration import (
    extract_message_text,
    filter_messages_for_memory,
    detect_correction,
    detect_reinforcement,
    MemoryUpdateQueue,
    ConversationContext,
    SubagentExecutor,
    SubagentTask,
)


# ─────────────────────────────────────────────
# Test Fixtures
# ─────────────────────────────────────────────

def make_message(msg_type: str, content: str, tool_calls=None) -> MagicMock:
    """Create a mock message."""
    msg = MagicMock()
    msg.type = msg_type
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def make_ai_with_tool_calls(calls: list[dict]) -> MagicMock:
    """Create a mock AI message with tool calls."""
    msg = MagicMock()
    msg.type = "ai"
    msg.content = ""
    msg.tool_calls = calls
    return msg


SAMPLE_MESSAGES = [
    make_message("human", "Research deer-flow architecture"),
    make_ai_with_tool_calls([{"name": "web_search", "args": {"query": "deer-flow"}}]),
    make_message("tool", '{"results": "deer-flow is an agent framework"}'),
    make_message("ai", "DeerFlow is a super agent harness by Bytedance."),
    make_message("human", "That's wrong, it's actually a research framework"),
    make_message("ai", "You're right, I apologize for the error."),
]


# ─────────────────────────────────────────────
# Message Processing Tests
# ─────────────────────────────────────────────

class TestMessageProcessing:
    
    def test_extract_text_string(self):
        msg = make_message("human", "Hello world")
        assert extract_message_text(msg) == "Hello world"
    
    def test_extract_text_list(self):
        msg = MagicMock()
        msg.content = [{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]
        assert extract_message_text(msg) == "Hello world"
    
    def test_extract_text_mixed_list(self):
        msg = MagicMock()
        msg.content = ["prefix", {"type": "text", "text": "suffix"}]
        assert extract_message_text(msg) == "prefix suffix"
    
    def test_filter_keeps_user_and_final_ai(self):
        messages = [
            make_message("human", "Hello"),
            make_message("ai", "Hi there"),
        ]
        filtered = filter_messages_for_memory(messages)
        assert len(filtered) == 2
    
    def test_filter_removes_tool_calls(self):
        messages = [
            make_message("human", "Hello"),
            make_ai_with_tool_calls([{"name": "search"}]),
            make_message("tool", "result"),
            make_message("ai", "Here's the answer"),
        ]
        filtered = filter_messages_for_memory(messages)
        assert len(filtered) == 2  # human + final ai
    
    def test_filter_strips_upload_blocks(self):
        messages = [
            make_message("human", "<uploaded_files>/tmp/file.txt</uploaded_files>"),
        ]
        filtered = filter_messages_for_memory(messages)
        assert len(filtered) == 0  # Only upload block, stripped away
    
    def test_filter_preserves_text_with_upload(self):
        messages = [
            make_message("human", "Please review <uploaded_files>/tmp/file.txt</uploaded_files>"),
        ]
        filtered = filter_messages_for_memory(messages)
        assert len(filtered) == 1
        assert "review" in filtered[0].content


# ─────────────────────────────────────────────
# Correction Detection Tests
# ─────────────────────────────────────────────

class TestCorrectionDetection:
    
    def test_detects_thats_wrong(self):
        messages = [make_message("human", "That's wrong, try again")]
        assert detect_correction(messages) is True
    
    def test_detects_you_misunderstood(self):
        messages = [make_message("human", "You misunderstood my request")]
        assert detect_correction(messages) is True
    
    def test_detects_try_again(self):
        messages = [make_message("human", "No, try again")]
        assert detect_correction(messages) is True
    
    def test_detects_chinese_correction(self):
        messages = [make_message("human", "不对，你理解错了")]
        assert detect_correction(messages) is True
    
    def test_no_correction(self):
        messages = [make_message("human", "Thanks, that's helpful")]
        assert detect_correction(messages) is False
    
    def test_ignores_ai_messages(self):
        messages = [make_message("ai", "That's wrong, let me fix it")]
        assert detect_correction(messages) is False


# ─────────────────────────────────────────────
# Reinforcement Detection Tests
# ─────────────────────────────────────────────

class TestReinforcementDetection:
    
    def test_detects_exactly_right(self):
        messages = [make_message("human", "Yes, exactly right!")]
        assert detect_reinforcement(messages) is True
    
    def test_detects_perfect(self):
        messages = [make_message("human", "Perfect.")]
        assert detect_reinforcement(messages) is True
    
    def test_detects_keep_doing_that(self):
        messages = [make_message("human", "Keep doing that")]
        assert detect_reinforcement(messages) is True
    
    def test_detects_chinese_reinforcement(self):
        messages = [make_message("human", "对，就是这样")]
        assert detect_reinforcement(messages) is True
    
    def test_no_reinforcement(self):
        messages = [make_message("human", "What's the weather?")]
        assert detect_reinforcement(messages) is False


# ─────────────────────────────────────────────
# Memory Update Queue Tests
# ─────────────────────────────────────────────

class TestMemoryUpdateQueue:
    
    def test_add_and_process(self):
        queue = MemoryUpdateQueue(debounce_seconds=0)
        messages = [make_message("human", "Hello"), make_message("ai", "Hi")]
        
        with patch.object(queue, '_update_memory_for_context') as mock_update:
            queue.add("thread-1", messages)
            time.sleep(0.5)  # Wait for timer to fire
            mock_update.assert_called_once()
    
    def test_debounce_merges(self):
        queue = MemoryUpdateQueue(debounce_seconds=60)  # Long debounce
        messages1 = [make_message("human", "Hello")]
        messages2 = [make_message("human", "World")]
        
        queue.add("thread-1", messages1)
        queue.add("thread-1", messages2)
        
        # Should only have one entry (merged)
        assert len(queue._queue) == 1
    
    def test_different_threads_separate(self):
        queue = MemoryUpdateQueue(debounce_seconds=60)
        
        queue.add("thread-1", [make_message("human", "A")])
        queue.add("thread-2", [make_message("human", "B")])
        
        assert len(queue._queue) == 2
    
    def test_correction_flag_preserved(self):
        queue = MemoryUpdateQueue(debounce_seconds=60)
        messages = [make_message("human", "That's wrong")]
        
        queue.add("thread-1", messages, correction_detected=True)
        
        assert queue._queue[0].correction_detected is True
    
    def test_reinforcement_flag_preserved(self):
        queue = MemoryUpdateQueue(debounce_seconds=60)
        messages = [make_message("human", "Perfect!")]
        
        queue.add("thread-1", messages, reinforcement_detected=True)
        
        assert queue._queue[0].reinforcement_detected is True
    
    def test_add_nowait_processes_immediately(self):
        queue = MemoryUpdateQueue(debounce_seconds=60)
        messages = [make_message("human", "Hello")]
        
        with patch.object(queue, '_update_memory_for_context') as mock_update:
            queue.add_nowait("thread-1", messages)
            time.sleep(0.5)
            mock_update.assert_called_once()


# ─────────────────────────────────────────────
# SubagentExecutor Tests
# ─────────────────────────────────────────────

class TestSubagentExecutor:
    
    def test_submit_and_complete(self):
        executor = SubagentExecutor(max_concurrent=2)
        task = SubagentTask(
            task_id="test-1",
            description="Test task",
            prompt="Do something",
        )
        
        task_id = executor.submit(task)
        assert task_id == "test-1"
        assert executor.active_count <= 2
        
        # Wait for completion
        time.sleep(1)
        result = executor.get_result("test-1", timeout=5)
        assert result is not None
        executor.shutdown()
    
    def test_concurrency_limit(self):
        executor = SubagentExecutor(max_concurrent=1)
        
        task1 = SubagentTask(task_id="t1", description="T1", prompt="P1")
        task2 = SubagentTask(task_id="t2", description="T2", prompt="P2")
        
        executor.submit(task1)
        
        with pytest.raises(RuntimeError, match="At max concurrency"):
            executor.submit(task2)
        
        executor.shutdown()
    
    def test_get_status(self):
        executor = SubagentExecutor(max_concurrent=3)
        task = SubagentTask(task_id="s1", description="Status test", prompt="test")
        
        executor.submit(task)
        status = executor.get_status()
        
        assert status["max_concurrent"] == 3
        assert status["active_count"] <= 3
        assert "active_tasks" in status
        
        executor.shutdown()
    
    def test_cancel_task(self):
        executor = SubagentExecutor(max_concurrent=2)
        task = SubagentTask(
            task_id="cancel-1",
            description="Long task",
            prompt="Sleep for 100 seconds",
            timeout_seconds=100,
        )
        
        executor.submit(task)
        # Task should be running (or completed quickly with placeholder)
        # Just verify cancel doesn't crash
        executor.cancel("cancel-1")
        executor.shutdown()
    
    def test_get_result_nonexistent(self):
        executor = SubagentExecutor()
        assert executor.get_result("nonexistent") is None
        executor.shutdown()
    
    def test_available_slots(self):
        executor = SubagentExecutor(max_concurrent=2)
        assert executor.available_slots == 2
        
        task = SubagentTask(task_id="slot-test", description="Test", prompt="test")
        executor.submit(task)
        
        # Should have at most 1 slot used (task may complete quickly)
        assert executor.available_slots >= 1
        executor.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
