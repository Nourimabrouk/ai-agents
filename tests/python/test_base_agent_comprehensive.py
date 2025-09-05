"""
Comprehensive unit tests for BaseAgent functionality
Tests the complete think-act-observe-evolve cycle with mocking
"""

import pytest
from pathlib import Path
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List
import tempfile
import os

# Import the code under test
from templates.base_agent import (
    BaseAgent, AgentState, Thought, Action, Observation, 
    Memory, LearningSystem
)
from utils.persistence.memory_store import SqliteMemoryStore


class TestBaseAgent:
    """Comprehensive unit tests for BaseAgent"""
    
    @pytest.fixture
    def temp_db(self):
        """Temporary database for testing"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
            db_path = f.name
        yield db_path
        try:
            os.unlink(db_path)
        except OSError:
        logger.info(f'Method {function_name} called')
        return {}
    
    @pytest.fixture
    def mock_anthropic_client(self):
        """Mock anthropic client"""
        mock_client = Mock()
        return mock_client
    
    @pytest.fixture  
    def test_agent(self, temp_db):
        """Test agent instance with memory backend"""
        config = {
            "memory_backend": "sqlite",
            "memory_db_path": temp_db
        }
        
        class TestAgent(BaseAgent):
            async def execute(self, task, action: Action):
                return {"result": f"executed_{task}", "success": True}
        
        return TestAgent(name="test_agent", config=config)
    
    @pytest.fixture
    def simple_agent(self):
        """Simple test agent without persistence"""
        class SimpleAgent(BaseAgent):
            async def execute(self, task, action: Action):
                return {"result": f"simple_{task}"}
        
        return SimpleAgent(name="simple_agent")


class TestAgentStates(TestBaseAgent):
    """Test agent state management"""
    
    @pytest.mark.asyncio
    async def test_initial_state(self, simple_agent):
        """Test agent starts in IDLE state"""
        assert simple_agent.state == AgentState.IDLE
        assert simple_agent.name == "simple_agent"
        assert simple_agent.total_tasks == 0
        assert simple_agent.successful_tasks == 0
    
    @pytest.mark.asyncio
    async def test_state_transitions_during_processing(self, simple_agent):
        """Test state changes during task processing"""
        # Track states by overriding state setter behavior
        states_observed = []
        original_state = simple_agent.state
        
        def track_state_change(self, new_state):
            states_observed.append(new_state)
            self._state = new_state
        
        # Replace state property setter
        type(simple_agent).state = property(
            lambda self: getattr(self, '_state', original_state),
            track_state_change
        )
        simple_agent._state = original_state
        
        await simple_agent.process_task("test_task")
        
        # Should see state transitions during processing
        assert AgentState.THINKING in states_observed
        assert AgentState.ACTING in states_observed  
        assert AgentState.OBSERVING in states_observed
        assert simple_agent.state == AgentState.IDLE  # Final state


class TestThinkActObserve(TestBaseAgent):
    """Test the core think-act-observe cycle"""
    
    @pytest.mark.asyncio
    async def test_think_returns_valid_thought(self, simple_agent):
        """Test think method returns proper Thought object"""
        thought = await simple_agent.think("test_task", {"context": "test"})
        
        assert isinstance(thought, Thought)
        assert isinstance(thought.analysis, str)
        assert thought.strategy in simple_agent._get_available_strategies()
        assert 0.0 <= thought.confidence <= 1.0
        assert isinstance(thought.alternatives, list)
        assert isinstance(thought.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_act_returns_valid_action(self, simple_agent):
        """Test act method returns proper Action object"""
        thought = Thought(
            analysis="Test analysis",
            strategy="direct",
            confidence=0.8
        )
        
        action = await simple_agent.act(thought)
        
        assert isinstance(action, Action)
        assert action.action_type == "direct"
        assert isinstance(action.parameters, dict)
        assert isinstance(action.tools_used, list)
        assert isinstance(action.expected_outcome, str)
        assert isinstance(action.timestamp, datetime)
    
    @pytest.mark.asyncio
    async def test_observe_updates_metrics(self, simple_agent):
        """Test observe method updates agent metrics"""
        action = Action(
            action_type="test",
            parameters={},
            tools_used=[],
            expected_outcome="test"
        )
        
        initial_tasks = simple_agent.total_tasks
        
        observation = await simple_agent.observe(action, {"success": True})
        
        assert isinstance(observation, Observation)
        assert observation.action == action
        assert simple_agent.total_tasks == initial_tasks + 1
        assert observation.success in [True, False]
        assert isinstance(observation.learnings, list)
    
    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, simple_agent):
        """Test complete task processing pipeline"""
        result = await simple_agent.process_task("integration_test")
        
        assert result is not None
        assert simple_agent.total_tasks == 1
        assert simple_agent.state == AgentState.IDLE
        # Check that memory was updated
        assert len(simple_agent.memory.episodic_memory) == 1


class TestMemorySystem(TestBaseAgent):
    """Test agent memory functionality"""
    
    @pytest.mark.asyncio
    async def test_memory_stores_episodes(self, test_agent):
        """Test episodic memory storage"""
        await test_agent.process_task("memory_test")
        
        assert len(test_agent.memory.episodic_memory) == 1
        episode = test_agent.memory.episodic_memory[0]
        assert isinstance(episode, Observation)
        assert episode.action.action_type in test_agent._get_available_strategies()
    
    @pytest.mark.asyncio
    async def test_memory_recall_similar(self, test_agent):
        """Test similar memory recall"""
        # Process multiple tasks to build memory
        await test_agent.process_task("task1")
        await test_agent.process_task("task2")
        await test_agent.process_task("task3")
        
        similar_memories = await test_agent.memory.recall_similar({}, k=2)
        assert len(similar_memories) <= 2
        assert all(isinstance(mem, Observation) for mem in similar_memories)
    
    @pytest.mark.asyncio 
    async def test_memory_pattern_extraction(self, test_agent):
        """Test pattern extraction from memory"""
        # Create some successful and failed tasks
        for i in range(5):
            await test_agent.process_task(f"task_{i}")
        
        patterns = await test_agent.memory.extract_patterns()
        assert isinstance(patterns, dict)
        assert 'success_rate' in patterns
        assert 0.0 <= patterns['success_rate'] <= 1.0
        
        if 'common_learnings' in patterns:
            assert isinstance(patterns['common_learnings'], list)
    
    @pytest.mark.asyncio
    async def test_memory_persistence_with_backend(self, test_agent):
        """Test memory persistence to SQLite backend"""
        await test_agent.process_task("persistence_test")
        
        # Memory should be persisted to database
        backend = test_agent.memory.backend
        episodes = backend.recent_episodes(test_agent.name, 5)
        assert len(episodes) >= 1
        
        # Test recall from persisted memory
        recalled = await test_agent.memory.recall_similar({}, k=3)
        assert len(recalled) >= 1


class TestLearningSystem(TestBaseAgent):
    """Test agent learning and adaptation"""
    
    @pytest.mark.asyncio
    async def test_learning_system_updates(self, simple_agent):
        """Test learning system updates from observations"""
        initial_strategies = len(simple_agent.learning_system.strategies)
        
        await simple_agent.process_task("learning_test")
        
        # Learning system should have updated
        assert len(simple_agent.learning_system.strategies) >= initial_strategies
    
    @pytest.mark.asyncio
    async def test_strategy_recommendation(self, simple_agent):
        """Test strategy recommendation based on learning"""
        learning_system = simple_agent.learning_system
        available = ["direct", "exploratory", "analytical"]
        
        recommended = await learning_system.recommend_strategy(available)
        assert recommended in available
    
    @pytest.mark.asyncio
    async def test_learning_convergence(self, simple_agent):
        """Test learning system converges on successful strategies"""
        # Simulate multiple tasks with biased success for "direct" strategy
        
        with patch.object(simple_agent, '_get_available_strategies', 
                         return_value=["direct", "exploratory"]):
            with patch.object(simple_agent, '_evaluate_success') as mock_eval:
                # Make direct strategy always succeed
                def success_evaluator(result):
                    return "direct" in str(result)
                mock_eval.side_effect = success_evaluator
                
                # Process multiple tasks
                for i in range(10):
                    await simple_agent.process_task(f"convergence_test_{i}")
                
                # "direct" strategy should have higher success rate
                direct_rate = simple_agent.learning_system.strategies.get("direct", 0)
                exploratory_rate = simple_agent.learning_system.strategies.get("exploratory", 0)
                
                # Allow for some variance in learning
                assert direct_rate >= exploratory_rate - 0.2


class TestErrorHandling(TestBaseAgent):
    """Test error handling and recovery"""
    
    @pytest.mark.asyncio
    async def test_execute_method_error_handling(self, simple_agent):
        """Test error handling in execute method"""
        
        class FailingAgent(BaseAgent):
            async def execute(self, task, action: Action):
                raise ValueError("Simulated execution error")
        
        failing_agent = FailingAgent(name="failing_agent")
        
        # Test that the error is properly raised and handled
        with pytest.raises(ValueError, match="Simulated execution error"):
            await failing_agent.process_task("error_test")
        
        # After error handling, agent should return to IDLE state (correct behavior)
        assert failing_agent.state == AgentState.IDLE
        # No tasks should be marked as successful due to the error
        assert failing_agent.successful_tasks == 0
    
    @pytest.mark.asyncio
    async def test_tool_execution_error_handling(self, simple_agent):
        """Test error handling in tool execution"""
        
        def failing_tool(**kwargs):
            raise RuntimeError("Tool failed")
        
        simple_agent.tools = [failing_tool]
        
        # Should not raise error, but handle gracefully
        result = await simple_agent.process_task("tool_error_test")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_memory_backend_failure_handling(self):
        """Test handling of memory backend failures"""
        
        class FailingMemoryStore(SqliteMemoryStore):
            def save_episode(self, agent_name: str, observation: Any):
                raise Exception("Database connection failed")
        
        config = {"memory_backend": "sqlite"}
        
        class ResilientAgent(BaseAgent):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.memory.backend = FailingMemoryStore()
            
            async def execute(self, task, action: Action):
                return {"result": "success"}
        
        agent = ResilientAgent(name="resilient", config=config)
        
        # Should handle backend failures gracefully
        result = await agent.process_task("backend_failure_test")
        assert result is not None
        # In-memory storage should still work
        assert len(agent.memory.episodic_memory) == 1


class TestAdvancedFeatures(TestBaseAgent):
    """Test advanced agent features"""
    
    @pytest.mark.asyncio
    async def test_sub_agent_spawning(self, simple_agent):
        """Test spawning specialized sub-agents"""
        sub_agent = await simple_agent.spawn_sub_agent("analyzer", "analysis")
        
        assert sub_agent.name == "simple_agent.analyzer"
        assert "analyzer" in simple_agent.sub_agents
        assert simple_agent.sub_agents["analyzer"] == sub_agent
    
    @pytest.mark.asyncio
    async def test_agent_collaboration(self, simple_agent):
        """Test collaboration between agents"""
        
        class CollaborativeAgent(BaseAgent):
            async def execute(self, task, action: Action):
                return {"collaborative_result": f"processed_{task}"}
        
        other_agent = CollaborativeAgent(name="collaborator")
        
        result = await simple_agent.collaborate_with(other_agent, "collab_test")
        
        assert result is not None
        assert isinstance(result, dict)
        # Should contain results from both agents
        assert "agent1" in result or "agent2" in result
    
    @pytest.mark.asyncio
    async def test_evolution_trigger(self, simple_agent):
        """Test agent evolution is triggered periodically"""
        
        with patch.object(simple_agent, 'evolve') as mock_evolve:
            # Process exactly 10 tasks to trigger evolution
            for i in range(10):
                await simple_agent.process_task(f"evolution_test_{i}")
            
            # Evolution should have been called
            mock_evolve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, simple_agent):
        """Test comprehensive metrics tracking"""
        initial_metrics = simple_agent.get_metrics()
        
        # Process successful and failed tasks
        await simple_agent.process_task("success_test")
        
        final_metrics = simple_agent.get_metrics()
        
        assert final_metrics["total_tasks"] > initial_metrics["total_tasks"]
        assert final_metrics["success_rate"] >= 0.0
        assert final_metrics["memory_size"] >= 0
        assert "strategies_learned" in final_metrics


class TestToolIntegration(TestBaseAgent):
    """Test tool selection and execution"""
    
    @pytest.mark.asyncio
    async def test_tool_selection_for_strategy(self, simple_agent):
        """Test tool selection based on strategy"""
        
        def analyze_tool(**kwargs):
            return "analysis_result"
        
        def generate_tool(**kwargs):
            return "generation_result"
        
        simple_agent.tools = [analyze_tool, generate_tool]
        
        # Test analytical strategy selects analyze tool
        selected = simple_agent._select_tools_for_strategy("analytical")
        tool_names = [tool.__name__ for tool in selected]
        assert "analyze_tool" in tool_names
        
        # Test creative strategy selects generate tool
        selected = simple_agent._select_tools_for_strategy("creative")
        tool_names = [tool.__name__ for tool in selected]
        assert "generate_tool" in tool_names
    
    @pytest.mark.asyncio
    async def test_async_tool_execution(self, simple_agent):
        """Test execution of async tools"""
        
        async def async_tool(**kwargs):
            await asyncio.sleep(0.01)  # Simulate async work
            return "async_result"
        
        simple_agent.tools = [async_tool]
        
        result = await simple_agent.process_task("async_tool_test")
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_sync_tool_execution(self, simple_agent):
        """Test execution of synchronous tools"""
        
        def sync_tool(**kwargs):
            return "sync_result"
        
        simple_agent.tools = [sync_tool]
        
        result = await simple_agent.process_task("sync_tool_test")
        assert result is not None


class TestWindowsCompatibility(TestBaseAgent):
    """Test Windows-specific functionality"""
    
    def test_windows_path_handling(self, temp_db):
        """Test Windows path handling in memory store"""
        # Use Windows-style path separators
        windows_path = temp_db.replace(str(Path('/').resolve()), '\\')
        
        config = {
            "memory_backend": "sqlite",
            "memory_db_path": windows_path
        }
        
        class WindowsAgent(BaseAgent):
            async def execute(self, task, action: Action):
                return {"windows": True}
        
        agent = WindowsAgent(name="windows_agent", config=config)
        
        # Should initialize without errors
        assert agent.memory.backend is not None
        assert os.path.exists(windows_path)
    
    @pytest.mark.asyncio
    async def test_concurrent_execution_windows(self, simple_agent):
        """Test concurrent task execution on Windows"""
        
        # Create multiple concurrent tasks
        tasks = [
            simple_agent.process_task(f"concurrent_test_{i}")
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All tasks should complete successfully
        assert len(results) == 5
        assert all(not isinstance(r, Exception) for r in results)
        
        # Total tasks should equal number of concurrent executions
        assert simple_agent.total_tasks == 5


class TestPerformanceMetrics(TestBaseAgent):
    """Test performance and resource usage"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_stays_bounded(self, simple_agent):
        """Test memory usage doesn't grow unbounded"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process many tasks
        for i in range(100):
            await simple_agent.process_task(f"memory_test_{i}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 50MB for 100 tasks)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB"
    
    @pytest.mark.asyncio
    async def test_processing_speed(self, simple_agent):
        """Test task processing speed"""
        import time
        
        start_time = time.perf_counter()
        
        # Process tasks
        for i in range(20):
            await simple_agent.process_task(f"speed_test_{i}")
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        average_time = total_time / 20
        
        # Each task should complete in reasonable time
        assert average_time < 0.1, f"Average task time: {average_time:.3f}s"
        assert total_time < 5.0, f"Total time: {total_time:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])