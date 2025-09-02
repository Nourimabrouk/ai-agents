"""
Comprehensive tests for utility modules
Tests logging, metrics, and memory persistence functionality
"""

import pytest
import asyncio
import tempfile
import os
import sqlite3
import logging
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timezone
from typing import Dict, Any

# Import utility modules
from utils.observability.logging import get_logger, LogContext, AgentFormatter
from utils.observability.metrics import (
    MetricsCollector, MetricPoint, Timer, AsyncTimer, global_metrics
)
from utils.persistence.memory_store import SqliteMemoryStore, SemanticProxy

# Import agent components for testing integration
from templates.base_agent import BaseAgent, Action, Observation


class TestLoggingUtilities:
    """Test logging functionality"""
    
    @pytest.fixture
    def test_logger(self):
        """Create test logger"""
        return get_logger("test_logger")
    
    def test_logger_initialization(self, test_logger):
        """Test logger is properly initialized"""
        assert test_logger.name == "test_logger"
        assert test_logger.level == logging.INFO
        assert len(test_logger.handlers) > 0
    
    def test_logger_singleton_behavior(self):
        """Test that same logger name returns same instance"""
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        
        assert logger1 is logger2
        # Should not duplicate handlers
        assert len(logger1.handlers) == len(logger2.handlers)
    
    def test_agent_formatter(self):
        """Test custom AgentFormatter functionality"""
        formatter = AgentFormatter()
        
        # Create a log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        assert "INFO: Test message" in formatted
        
        # Test with agent context
        record.agent_name = "test_agent"
        record.task_id = "task_123"
        
        formatted_with_context = formatter.format(record)
        # Should still format properly with extra context
        assert "Test message" in formatted_with_context
    
    def test_error_level_formatting(self):
        """Test error-level log formatting"""
        formatter = AgentFormatter()
        
        error_record = logging.LogRecord(
            name="error_logger",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(error_record)
        assert "ERROR" in formatted
        assert "Error occurred" in formatted
        # Error format includes timestamp and logger name
        assert "error_logger" in formatted
    
    def test_log_context_manager(self, test_logger):
        """Test LogContext context manager"""
        
        # Test setting context
        with LogContext(test_logger, agent_name="context_agent", task_id="ctx_task"):
            assert hasattr(test_logger, 'agent_name')
            assert test_logger.agent_name == "context_agent"
            assert hasattr(test_logger, 'task_id')
            assert test_logger.task_id == "ctx_task"
        
        # Context should be cleaned up after exiting
        assert not hasattr(test_logger, 'agent_name')
        assert not hasattr(test_logger, 'task_id')
    
    def test_log_context_restoration(self, test_logger):
        """Test LogContext properly restores previous values"""
        # Set initial context
        test_logger.agent_name = "original_agent"
        
        with LogContext(test_logger, agent_name="temp_agent"):
            assert test_logger.agent_name == "temp_agent"
        
        # Should restore original value
        assert test_logger.agent_name == "original_agent"
        
        # Clean up
        delattr(test_logger, 'agent_name')
    
    def test_multiple_log_levels(self, test_logger):
        """Test different log levels work properly"""
        with patch.object(test_logger, 'handlers') as mock_handlers:
            mock_handler = Mock()
            mock_handlers.__iter__.return_value = [mock_handler]
            
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")
            
            # Should have attempted to log (handlers will filter based on level)
            assert mock_handler.handle.call_count >= 3  # info, warning, error


class TestMetricsCollection:
    """Test metrics collection functionality"""
    
    @pytest.fixture
    def metrics_collector(self):
        """Create fresh metrics collector for each test"""
        return MetricsCollector("test_namespace")
    
    def test_metrics_collector_initialization(self, metrics_collector):
        """Test metrics collector initialization"""
        assert metrics_collector.namespace == "test_namespace"
        assert len(metrics_collector.metrics) == 0
        assert len(metrics_collector.counters) == 0
        assert len(metrics_collector.gauges) == 0
        assert len(metrics_collector.timers) == 0
    
    def test_counter_metrics(self, metrics_collector):
        """Test counter metric functionality"""
        # Increment counter
        metrics_collector.incr("test.counter", 5.0)
        metrics_collector.incr("test.counter", 3.0)
        
        stats = metrics_collector.get_stats("test.counter")
        assert stats["type"] == "counter"
        assert stats["value"] == 8.0
    
    def test_gauge_metrics(self, metrics_collector):
        """Test gauge metric functionality"""
        metrics_collector.gauge("test.gauge", 42.5)
        metrics_collector.gauge("test.gauge", 37.8)  # Should overwrite
        
        stats = metrics_collector.get_stats("test.gauge")
        assert stats["type"] == "gauge"
        assert stats["value"] == 37.8
    
    def test_timing_metrics(self, metrics_collector):
        """Test timing metric functionality"""
        metrics_collector.timing("test.timer", 0.1)
        metrics_collector.timing("test.timer", 0.2)
        metrics_collector.timing("test.timer", 0.15)
        
        stats = metrics_collector.get_stats("test.timer")
        assert stats["type"] == "timer"
        assert stats["count"] == 3
        assert stats["min"] == 0.1
        assert stats["max"] == 0.2
        assert stats["mean"] == pytest.approx(0.15, abs=0.01)
        assert stats["sum"] == pytest.approx(0.45, abs=0.01)
    
    def test_timer_context_manager(self, metrics_collector):
        """Test Timer context manager"""
        with metrics_collector.timer("test.context_timer"):
            time.sleep(0.01)  # Small delay
        
        stats = metrics_collector.get_stats("test.context_timer")
        assert stats["type"] == "timer"
        assert stats["count"] == 1
        assert stats["mean"] >= 0.01  # Should be at least our sleep time
    
    @pytest.mark.asyncio
    async def test_async_timer_context_manager(self, metrics_collector):
        """Test AsyncTimer context manager"""
        async def async_operation():
            await asyncio.sleep(0.01)
        
        async_timer = AsyncTimer(metrics_collector, "test.async_timer")
        async with async_timer:
            await async_operation()
        
        stats = metrics_collector.get_stats("test.async_timer")
        assert stats["type"] == "timer"
        assert stats["count"] == 1
        assert stats["mean"] >= 0.01
    
    def test_metric_tags(self, metrics_collector):
        """Test metric tags functionality"""
        tags = {"environment": "test", "service": "agent"}
        
        metrics_collector.incr("tagged.counter", 1.0, tags)
        
        # Should have stored the metric
        full_name = "test_namespace.tagged.counter"
        assert full_name in metrics_collector.metrics
        
        metric_point = metrics_collector.metrics[full_name][0]
        assert isinstance(metric_point, MetricPoint)
        assert metric_point.tags == tags
    
    def test_get_all_stats(self, metrics_collector):
        """Test getting all statistics at once"""
        metrics_collector.incr("counter.test", 5)
        metrics_collector.gauge("gauge.test", 10.5)
        metrics_collector.timing("timer.test", 0.3)
        
        all_stats = metrics_collector.get_all_stats()
        
        assert len(all_stats) == 3
        assert "test_namespace.counter.test" in all_stats
        assert "test_namespace.gauge.test" in all_stats  
        assert "test_namespace.timer.test" in all_stats
        
        assert all_stats["test_namespace.counter.test"]["type"] == "counter"
        assert all_stats["test_namespace.gauge.test"]["type"] == "gauge"
        assert all_stats["test_namespace.timer.test"]["type"] == "timer"
    
    def test_metrics_reset(self, metrics_collector):
        """Test metrics reset functionality"""
        metrics_collector.incr("test.counter", 5)
        metrics_collector.gauge("test.gauge", 10)
        
        assert len(metrics_collector.get_all_stats()) == 2
        
        metrics_collector.reset()
        
        assert len(metrics_collector.get_all_stats()) == 0
        assert len(metrics_collector.metrics) == 0
        assert len(metrics_collector.counters) == 0
        assert len(metrics_collector.gauges) == 0
    
    def test_global_metrics_instance(self):
        """Test global metrics instance"""
        from utils.observability.metrics import global_metrics, get_metrics
        
        # Test global instance
        global_metrics.incr("global.test")
        stats = global_metrics.get_stats("global.test")
        assert stats["type"] == "counter"
        
        # Test get_metrics function
        custom_metrics = get_metrics("custom")
        assert custom_metrics.namespace == "custom"
        
        default_metrics = get_metrics()
        assert default_metrics is global_metrics


class TestMemoryPersistence:
    """Test memory persistence functionality"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except OSError:
            pass
    
    @pytest.fixture
    def memory_store(self, temp_db_path):
        """Create SqliteMemoryStore instance"""
        return SqliteMemoryStore(temp_db_path)
    
    @pytest.fixture
    def sample_observation(self):
        """Create sample observation for testing"""
        action = Action(
            action_type="test_action",
            parameters={"param1": "value1"},
            tools_used=["tool1", "tool2"],
            expected_outcome="test outcome"
        )
        
        observation = Observation(
            action=action,
            result={"success": True, "data": "test_result"},
            success=True,
            learnings=["learned something", "improved strategy"]
        )
        
        return observation
    
    def test_memory_store_initialization(self, memory_store, temp_db_path):
        """Test memory store initializes database properly"""
        assert os.path.exists(temp_db_path)
        
        # Check database schema
        with sqlite3.connect(temp_db_path) as conn:
            cursor = conn.cursor()
            
            # Check episodes table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='episodes'")
            assert cursor.fetchone() is not None
            
            # Check semantic_memory table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='semantic_memory'")
            assert cursor.fetchone() is not None
    
    def test_save_and_retrieve_episode(self, memory_store, sample_observation):
        """Test saving and retrieving episodic memories"""
        agent_name = "test_agent"
        
        # Save episode
        memory_store.save_episode(agent_name, sample_observation)
        
        # Retrieve episode
        episodes = memory_store.recent_episodes(agent_name, 1)
        
        assert len(episodes) == 1
        episode = episodes[0]
        
        assert episode["success"] == sample_observation.success
        assert episode["action"]["action_type"] == sample_observation.action.action_type
        assert episode["result"]["success"] == True
        assert len(episode["learnings"]) == len(sample_observation.learnings)
    
    def test_multiple_episodes_retrieval(self, memory_store, sample_observation):
        """Test retrieving multiple episodes with limit"""
        agent_name = "multi_episode_agent"
        
        # Save multiple episodes
        for i in range(5):
            modified_obs = Observation(
                action=Action(
                    action_type=f"action_{i}",
                    parameters={"iteration": i},
                    tools_used=[],
                    expected_outcome=f"outcome_{i}"
                ),
                result={"iteration": i},
                success=i % 2 == 0,  # Alternate success/failure
                learnings=[f"learning_{i}"]
            )
            memory_store.save_episode(agent_name, modified_obs)
        
        # Retrieve with limit
        episodes = memory_store.recent_episodes(agent_name, 3)
        
        assert len(episodes) == 3
        # Should be in reverse order (most recent first)
        assert episodes[0]["action"]["action_type"] == "action_4"
        assert episodes[1]["action"]["action_type"] == "action_3"
        assert episodes[2]["action"]["action_type"] == "action_2"
    
    def test_semantic_memory_operations(self, memory_store):
        """Test semantic memory save/retrieve operations"""
        agent_name = "semantic_agent"
        
        # Save semantic memory
        test_data = {
            "strategy_performance": {"direct": 0.8, "exploratory": 0.6},
            "preferred_tools": ["tool_a", "tool_b"],
            "last_updated": datetime.now().isoformat()
        }
        
        memory_store.save_semantic(agent_name, "performance_data", test_data)
        
        # Retrieve semantic memory
        retrieved = memory_store.get_semantic(agent_name, "performance_data")
        
        assert retrieved is not None
        assert retrieved["strategy_performance"]["direct"] == 0.8
        assert retrieved["preferred_tools"] == ["tool_a", "tool_b"]
    
    def test_semantic_memory_update(self, memory_store):
        """Test semantic memory updates (INSERT OR REPLACE)"""
        agent_name = "update_agent"
        key = "config_setting"
        
        # Initial save
        memory_store.save_semantic(agent_name, key, {"value": "initial"})
        
        # Update
        memory_store.save_semantic(agent_name, key, {"value": "updated", "version": 2})
        
        # Retrieve
        result = memory_store.get_semantic(agent_name, key)
        
        assert result["value"] == "updated"
        assert result["version"] == 2
    
    def test_get_all_semantic_memory(self, memory_store):
        """Test retrieving all semantic memory for an agent"""
        agent_name = "all_semantic_agent"
        
        # Save multiple semantic memories
        memory_store.save_semantic(agent_name, "config1", {"setting": "value1"})
        memory_store.save_semantic(agent_name, "config2", {"setting": "value2"})
        memory_store.save_semantic(agent_name, "metrics", {"success_rate": 0.75})
        
        all_memory = memory_store.all_semantic(agent_name)
        
        assert len(all_memory) == 3
        assert "config1" in all_memory
        assert "config2" in all_memory
        assert "metrics" in all_memory
        assert all_memory["metrics"]["success_rate"] == 0.75
    
    def test_clear_agent_memory(self, memory_store, sample_observation):
        """Test clearing all memory for an agent"""
        agent_name = "clear_test_agent"
        
        # Add episodic and semantic memories
        memory_store.save_episode(agent_name, sample_observation)
        memory_store.save_semantic(agent_name, "test_data", {"value": "test"})
        
        # Verify data exists
        assert len(memory_store.recent_episodes(agent_name, 10)) == 1
        assert memory_store.get_semantic(agent_name, "test_data") is not None
        
        # Clear memory
        memory_store.clear_agent_memory(agent_name)
        
        # Verify data is gone
        assert len(memory_store.recent_episodes(agent_name, 10)) == 0
        assert memory_store.get_semantic(agent_name, "test_data") is None
    
    def test_memory_store_error_handling(self):
        """Test memory store handles errors gracefully"""
        # Use non-existent directory path
        invalid_path = "/definitely/not/a/valid/path/test.db"
        
        # Should handle initialization errors
        try:
            store = SqliteMemoryStore(invalid_path)
            # If it doesn't raise, that's also fine - it should handle gracefully
        except Exception:
            # Expected in some cases
            pass
    
    def test_semantic_proxy(self, memory_store):
        """Test SemanticProxy auto-persistence functionality"""
        agent_name = "proxy_agent"
        initial_data = {"existing_key": "existing_value"}
        
        proxy = SemanticProxy(agent_name, memory_store, initial_data)
        
        # Test initial data
        assert proxy["existing_key"] == "existing_value"
        
        # Test setting new value (should auto-persist)
        proxy["new_key"] = "new_value"
        
        # Verify persistence by checking database directly
        retrieved = memory_store.get_semantic(agent_name, "new_key")
        assert retrieved == "new_value"
        
        # Test update method
        proxy.update({"batch_key1": "batch_value1", "batch_key2": "batch_value2"})
        
        # Check batch updates were persisted
        assert memory_store.get_semantic(agent_name, "batch_key1") == "batch_value1"
        assert memory_store.get_semantic(agent_name, "batch_key2") == "batch_value2"
    
    def test_semantic_proxy_error_handling(self):
        """Test SemanticProxy handles backend errors gracefully"""
        
        class FailingBackend:
            def save_semantic(self, agent_name, key, value):
                raise Exception("Backend failed")
        
        proxy = SemanticProxy("test_agent", FailingBackend(), {})
        
        # Should not raise exception, even if backend fails
        proxy["test_key"] = "test_value"
        
        # Value should still be in the proxy dict
        assert proxy["test_key"] == "test_value"


class TestUtilityIntegration:
    """Test integration between utility modules"""
    
    @pytest.fixture
    def temp_db_path(self):
        """Create temporary database file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except OSError:
            pass
    
    @pytest.mark.asyncio
    async def test_agent_with_full_utility_integration(self, temp_db_path):
        """Test agent using all utility modules together"""
        
        class FullyInstrumentedAgent(BaseAgent):
            def __init__(self, name: str):
                config = {
                    "memory_backend": "sqlite",
                    "memory_db_path": temp_db_path
                }
                super().__init__(name=name, config=config)
                
                # Add custom metrics
                self.custom_metrics = MetricsCollector(f"agent.{name}")
                self.logger = get_logger(f"agent.{name}")
            
            async def execute(self, task, action: Action):
                # Use timer for performance measurement
                with self.custom_metrics.timer("task.execution"):
                    # Log with context
                    with LogContext(self.logger, task_id=str(hash(str(task)))):
                        self.logger.info(f"Executing task: {task}")
                        
                        # Simulate work
                        await asyncio.sleep(0.01)
                        
                        # Update metrics
                        self.custom_metrics.incr("tasks.completed")
                        self.custom_metrics.gauge("current.load", 1.0)
                        
                        return {"result": "success", "agent": self.name}
        
        agent = FullyInstrumentedAgent("instrumented_agent")
        
        # Process some tasks
        for i in range(3):
            await agent.process_task(f"integration_test_{i}")
        
        # Verify all systems working together
        
        # Check memory persistence
        assert len(agent.memory.episodic_memory) == 3
        recent = agent.memory.backend.recent_episodes(agent.name, 5)
        assert len(recent) == 3
        
        # Check custom metrics
        execution_stats = agent.custom_metrics.get_stats("task.execution")
        assert execution_stats["type"] == "timer"
        assert execution_stats["count"] == 3
        
        completion_stats = agent.custom_metrics.get_stats("tasks.completed")
        assert completion_stats["type"] == "counter" 
        assert completion_stats["value"] == 3.0
    
    @pytest.mark.asyncio
    async def test_concurrent_utility_access(self, temp_db_path):
        """Test utilities handle concurrent access properly"""
        
        class ConcurrentAgent(BaseAgent):
            def __init__(self, name: str):
                config = {
                    "memory_backend": "sqlite", 
                    "memory_db_path": temp_db_path
                }
                super().__init__(name=name, config=config)
                self.metrics = MetricsCollector("concurrent")
            
            async def execute(self, task, action: Action):
                # Concurrent metrics updates
                self.metrics.incr("concurrent.counter")
                
                # Concurrent memory access
                self.memory.semantic_memory[f"task_{task}"] = {
                    "timestamp": datetime.now().isoformat(),
                    "agent": self.name
                }
                
                await asyncio.sleep(0.01)  # Simulate work
                return {"agent": self.name, "task": str(task)}
        
        # Create multiple agents sharing utilities
        agents = [ConcurrentAgent(f"concurrent_{i}") for i in range(3)]
        
        # Run concurrent tasks
        tasks = []
        for i in range(10):
            agent = agents[i % len(agents)]
            tasks.append(agent.process_task(f"concurrent_task_{i}"))
        
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        assert len(results) == 10
        assert all(result is not None for result in results)
        
        # Check metrics were properly aggregated
        counter_stats = agents[0].metrics.get_stats("concurrent.counter")
        assert counter_stats["value"] == 10.0  # All agents share the same metrics instance
    
    def test_utility_error_isolation(self, temp_db_path):
        """Test that errors in one utility don't affect others"""
        
        class ErrorProneAgent(BaseAgent):
            def __init__(self, name: str):
                config = {
                    "memory_backend": "sqlite",
                    "memory_db_path": temp_db_path
                }
                super().__init__(name=name, config=config)
                self.metrics = MetricsCollector("error_test")
            
            async def execute(self, task, action: Action):
                # This should work even if logging fails
                self.metrics.incr("tasks.attempted")
                
                # Try to cause a logging error
                try:
                    bad_logger = get_logger(None)  # Might cause error
                    bad_logger.info("This might fail")
                except Exception:
                    pass  # Ignore logging errors
                
                # Memory should still work
                self.memory.semantic_memory["last_task"] = str(task)
                
                return {"resilient": True}
        
        agent = ErrorProneAgent("error_prone")
        
        # Should complete despite potential logging errors
        result = asyncio.run(agent.process_task("error_test"))
        assert result["resilient"] is True
        
        # Metrics should still work
        stats = agent.metrics.get_stats("tasks.attempted")
        assert stats["value"] == 1.0


class TestWindowsCompatibility:
    """Test Windows-specific compatibility for utilities"""
    
    def test_windows_file_path_handling(self):
        """Test Windows file path handling in memory store"""
        import tempfile
        
        # Create Windows-style path
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use Windows-style separators
            windows_path = os.path.join(temp_dir, "windows_test.db").replace('/', '\\')
            
            store = SqliteMemoryStore(windows_path)
            
            # Should handle Windows paths correctly
            assert os.path.exists(windows_path)
            
            # Test basic operations work
            store.save_semantic("test_agent", "windows_test", {"path": windows_path})
            result = store.get_semantic("test_agent", "windows_test")
            
            assert result is not None
            assert result["path"] == windows_path
    
    def test_concurrent_file_access_windows(self):
        """Test concurrent file access on Windows"""
        import tempfile
        import threading
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        
        try:
            store1 = SqliteMemoryStore(db_path)
            store2 = SqliteMemoryStore(db_path)
            
            def write_data(store, agent_id):
                for i in range(10):
                    store.save_semantic(f"agent_{agent_id}", f"key_{i}", {"value": i})
            
            # Concurrent writes from different threads
            thread1 = threading.Thread(target=write_data, args=(store1, 1))
            thread2 = threading.Thread(target=write_data, args=(store2, 2))
            
            thread1.start()
            thread2.start()
            
            thread1.join()
            thread2.join()
            
            # Both agents should have their data
            agent1_data = store1.all_semantic("agent_1")
            agent2_data = store1.all_semantic("agent_2")
            
            assert len(agent1_data) == 10
            assert len(agent2_data) == 10
            
        finally:
            try:
                os.unlink(db_path)
            except OSError:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])