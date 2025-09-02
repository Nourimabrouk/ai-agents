"""
Comprehensive integration tests for AgentOrchestrator
Tests multi-agent coordination patterns and communication
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List
import time

# Import the code under test
from orchestrator import (
    AgentOrchestrator, Task, Message, Blackboard,
    CommunicationProtocol, CustomerSupportAgent, DataAnalystAgent
)
from templates.base_agent import BaseAgent, Action, AgentState


class TestAgentOrchestrator:
    """Integration tests for AgentOrchestrator"""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance"""
        return AgentOrchestrator(name="test_orchestrator")
    
    @pytest.fixture
    def mock_agents(self):
        """Create mock agents for testing"""
        
        class MockAgent(BaseAgent):
            def __init__(self, name: str, success_rate: float = 0.8):
                super().__init__(name=name)
                self._success_rate = success_rate
                self.processing_times = []
            
            async def execute(self, task, action: Action):
                # Simulate processing time
                processing_time = 0.01  # 10ms
                await asyncio.sleep(processing_time)
                self.processing_times.append(processing_time)
                
                return {
                    "agent": self.name,
                    "task": str(task),
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
            
            def get_success_rate(self):
                return self._success_rate
        
        return [
            MockAgent("agent_1", 0.9),
            MockAgent("agent_2", 0.8),
            MockAgent("agent_3", 0.7),
            MockAgent("agent_4", 0.6)
        ]
    
    @pytest.fixture
    def sample_task(self):
        """Create sample task"""
        return Task(
            id="test_task_001",
            description="Process user request for data analysis",
            requirements={
                "data_source": "users_db",
                "analysis_type": "trend_analysis",
                "output_format": "json"
            },
            deadline=datetime.now() + timedelta(hours=1)
        )


class TestBasicOrchestration(TestAgentOrchestrator):
    """Test basic orchestration functionality"""
    
    @pytest.mark.asyncio
    async def test_agent_registration(self, orchestrator, mock_agents):
        """Test agent registration and unregistration"""
        agent = mock_agents[0]
        
        # Register agent
        orchestrator.register_agent(agent)
        assert agent.name in orchestrator.agents
        assert orchestrator.agents[agent.name] == agent
        
        # Unregister agent
        orchestrator.unregister_agent(agent.name)
        assert agent.name not in orchestrator.agents
    
    @pytest.mark.asyncio
    async def test_single_agent_delegation(self, orchestrator, mock_agents, sample_task):
        """Test task delegation to single agent"""
        agent = mock_agents[0]
        orchestrator.register_agent(agent)
        
        result = await orchestrator.delegate_task(sample_task)
        
        assert result is not None
        assert result["agent"] == agent.name
        assert sample_task.status == "completed"
        assert orchestrator.total_tasks_completed == 1
        assert len(sample_task.assigned_agents) == 1
    
    @pytest.mark.asyncio
    async def test_multi_agent_delegation(self, orchestrator, mock_agents, sample_task):
        """Test task delegation to multiple agents"""
        for agent in mock_agents[:3]:
            orchestrator.register_agent(agent)
        
        # Modify task to require multiple agents
        sample_task.description = "Complex task requiring collaboration"
        
        result = await orchestrator.delegate_task(sample_task)
        
        assert result is not None
        assert len(sample_task.assigned_agents) >= 1
        assert sample_task.status == "completed"
    
    @pytest.mark.asyncio
    async def test_no_suitable_agents(self, orchestrator, sample_task):
        """Test handling when no suitable agents are available"""
        # No agents registered
        result = await orchestrator.delegate_task(sample_task)
        
        assert result is None
        assert sample_task.status == "pending"  # Status unchanged


class TestExecutionPatterns(TestAgentOrchestrator):
    """Test different execution patterns"""
    
    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, mock_agents):
        """Test parallel execution pattern"""
        for agent in mock_agents[:3]:
            orchestrator.register_agent(agent)
        
        task = Task(
            id="parallel_task",
            description="Parallel processing task",
            requirements={"pattern": "parallel"}
        )
        
        start_time = time.perf_counter()
        results = await orchestrator.parallel_execution(mock_agents[:3], task)
        end_time = time.perf_counter()
        
        # Should complete faster than sequential
        assert len(results) == 3
        assert all(result is not None for result in results)
        
        # Parallel execution should be roughly as fast as single agent
        processing_time = end_time - start_time
        assert processing_time < 0.1  # Should complete quickly in parallel
    
    @pytest.mark.asyncio
    async def test_sequential_execution(self, orchestrator, mock_agents):
        """Test sequential execution pattern"""
        for agent in mock_agents[:3]:
            orchestrator.register_agent(agent)
        
        task = Task(
            id="sequential_task",
            description="Sequential processing task",
            requirements={"pattern": "sequential"}
        )
        
        result = await orchestrator.sequential_execution(mock_agents[:3], task)
        
        assert result is not None
        # Result should come from the last agent in sequence
        assert result["agent"] == mock_agents[2].name
    
    @pytest.mark.asyncio
    async def test_collaborative_execution(self, orchestrator, mock_agents):
        """Test collaborative execution with conversation"""
        for agent in mock_agents[:2]:
            orchestrator.register_agent(agent)
        
        task = Task(
            id="collaborative_task",
            description="Collaborative problem solving",
            requirements={"pattern": "collaborative"}
        )
        
        result = await orchestrator.collaborative_execution(mock_agents[:2], task)
        
        assert result is not None
        assert "conversation_rounds" in result
        assert "participating_agents" in result
        assert result["conversation_rounds"] > 0
        assert len(result["participating_agents"]) == 2
    
    @pytest.mark.asyncio
    async def test_consensus_execution(self, orchestrator, mock_agents):
        """Test consensus-based execution with voting"""
        for agent in mock_agents[:3]:
            orchestrator.register_agent(agent)
        
        task = Task(
            id="consensus_task",
            description="Consensus-based decision making",
            requirements={"pattern": "consensus"}
        )
        
        result = await orchestrator.consensus_execution(mock_agents[:3], task)
        
        assert result is not None
        # Should be one of the agent results
        assert "agent" in result


class TestHierarchicalDelegation(TestAgentOrchestrator):
    """Test hierarchical task decomposition"""
    
    @pytest.mark.asyncio
    async def test_task_decomposition(self, orchestrator, mock_agents):
        """Test automatic task decomposition"""
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        complex_task = Task(
            id="complex_hierarchical_task",
            description="Complex multi-stage data processing pipeline",
            requirements={
                "stages": ["analysis", "implementation", "validation"],
                "complexity": "high"
            }
        )
        
        subtasks = await orchestrator._decompose_task(complex_task)
        
        assert len(subtasks) >= 3  # Should create multiple subtasks
        assert all(isinstance(st, Task) for st in subtasks)
        
        # Check dependency structure
        analysis_task = next((st for st in subtasks if "analysis" in st.id), None)
        impl_task = next((st for st in subtasks if "implementation" in st.id), None)
        validation_task = next((st for st in subtasks if "validation" in st.id), None)
        
        assert analysis_task is not None
        assert impl_task is not None
        assert validation_task is not None
        
        # Validation should depend on implementation
        assert analysis_task.id in impl_task.dependencies
        assert impl_task.id in validation_task.dependencies
    
    @pytest.mark.asyncio
    async def test_hierarchical_delegation_execution(self, orchestrator, mock_agents):
        """Test full hierarchical delegation execution"""
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        complex_task = Task(
            id="hierarchical_test",
            description="Multi-stage hierarchical processing",
            requirements={"complexity": "high"}
        )
        
        result = await orchestrator.hierarchical_delegation(complex_task)
        
        assert result is not None
        assert "subtask_results" in result
        assert "synthesis" in result
        assert isinstance(result["subtask_results"], dict)
        assert len(result["subtask_results"]) >= 3


class TestSwarmIntelligence(TestAgentOrchestrator):
    """Test swarm intelligence algorithms"""
    
    @pytest.mark.asyncio
    async def test_swarm_creation_and_optimization(self, orchestrator):
        """Test swarm agent creation and optimization"""
        objective = "Optimize resource allocation for maximum efficiency"
        swarm_size = 5
        
        result = await orchestrator.swarm_intelligence(objective, swarm_size)
        
        assert result is not None
        # Should have created swarm agents
        swarm_agent_names = [f"swarm_{i}" for i in range(swarm_size)]
        for agent_name in swarm_agent_names:
            assert agent_name in orchestrator.agents
        
        # Agents should have developed local bests
        swarm_agents = [orchestrator.agents[name] for name in swarm_agent_names]
        local_bests = [
            agent.memory.semantic_memory.get("local_best")
            for agent in swarm_agents
        ]
        
        # At least some agents should have found solutions
        assert any(best is not None for best in local_bests)
    
    @pytest.mark.asyncio
    async def test_swarm_convergence(self, orchestrator):
        """Test swarm convergence detection"""
        # Create mock swarm agents with similar solutions
        class MockSwarmAgent(BaseAgent):
            def __init__(self, name: str):
                super().__init__(name=name)
                self.memory.semantic_memory["local_best"] = "similar_solution"
            
            async def execute(self, task, action: Action):
                return "swarm_result"
        
        swarm = [MockSwarmAgent(f"swarm_{i}") for i in range(3)]
        
        converged = await orchestrator._check_swarm_convergence(swarm)
        assert converged is True  # Similar solutions should indicate convergence


class TestBlackboardCommunication(TestAgentOrchestrator):
    """Test blackboard-based communication"""
    
    @pytest.mark.asyncio
    async def test_blackboard_read_write(self, orchestrator):
        """Test basic blackboard operations"""
        blackboard = orchestrator.blackboard
        
        # Write knowledge
        await blackboard.write("agent_1", "task_status", {"completed": True, "result": "success"})
        
        # Read knowledge
        result = await blackboard.read("task_status")
        assert result is not None
        assert result["completed"] is True
        assert result["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_blackboard_subscriptions(self, orchestrator):
        """Test blackboard subscription system"""
        blackboard = orchestrator.blackboard
        
        # Subscribe to updates
        await blackboard.subscribe("agent_2", "shared_data")
        
        # Verify subscription
        assert "agent_2" in blackboard.subscriptions["shared_data"]
        
        # Write data (should trigger notification)
        await blackboard.write("agent_1", "shared_data", {"update": "new_info"})
        
        # Check that history tracks the write
        assert len(blackboard.history) > 0
        latest_entry = blackboard.history[-1]
        assert latest_entry[1] == "agent_1"  # Agent ID
        assert latest_entry[2] == "shared_data"  # Key
    
    @pytest.mark.asyncio
    async def test_blackboard_pattern_querying(self, orchestrator):
        """Test pattern-based blackboard queries"""
        blackboard = orchestrator.blackboard
        
        # Add various data
        await blackboard.write("agent_1", "task_001_status", {"status": "completed"})
        await blackboard.write("agent_2", "task_002_status", {"status": "pending"})
        await blackboard.write("agent_3", "config_setting", {"value": "enabled"})
        
        # Query for task-related data
        results = await blackboard.query({"task": True})
        
        # Should find task-related entries
        assert len(results) >= 2
        assert any("task_001" in key for key in results.keys())
        assert any("task_002" in key for key in results.keys())


class TestEmergentBehaviorDetection(TestAgentOrchestrator):
    """Test emergent behavior detection"""
    
    @pytest.mark.asyncio
    async def test_message_pattern_analysis(self, orchestrator, mock_agents):
        """Test analysis of communication patterns"""
        for agent in mock_agents[:3]:
            orchestrator.register_agent(agent)
        
        patterns = await orchestrator._analyze_message_patterns()
        
        assert isinstance(patterns, list)
        if patterns:
            pattern = patterns[0]
            assert "type" in pattern
            assert "description" in pattern
            assert pattern["type"] == "communication_cluster"
    
    @pytest.mark.asyncio
    async def test_specialization_detection(self, orchestrator, mock_agents):
        """Test detection of agent specialization"""
        # Set up agents with different specializations
        for agent in mock_agents[:2]:
            orchestrator.register_agent(agent)
            # Simulate task history with specialization
            for i in range(10):
                await agent.process_task(f"specialized_task_{agent.name}_{i}")
        
        patterns = await orchestrator._analyze_specialization_patterns()
        
        assert isinstance(patterns, list)
        # Should detect some specialization patterns
        if patterns:
            pattern = patterns[0]
            assert pattern["type"] == "agent_specialization"
            assert "specializations" in pattern
    
    @pytest.mark.asyncio
    async def test_full_emergent_behavior_detection(self, orchestrator, mock_agents):
        """Test comprehensive emergent behavior detection"""
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        # Simulate some agent activity
        for agent in mock_agents[:2]:
            await agent.process_task("behavior_test")
        
        emergent_patterns = await orchestrator.emergent_behavior_detection()
        
        assert isinstance(emergent_patterns, list)
        # Should detect various types of patterns
        pattern_types = {p["type"] for p in emergent_patterns}
        expected_types = {
            "communication_cluster", 
            "task_specialization",
            "knowledge_convergence"
        }
        
        # Should find at least some expected patterns
        assert len(pattern_types & expected_types) > 0


class TestPerformanceAndScalability(TestAgentOrchestrator):
    """Test performance and scalability"""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_processing(self, orchestrator, mock_agents):
        """Test handling multiple concurrent tasks"""
        for agent in mock_agents:
            orchestrator.register_agent(agent)
        
        # Create multiple tasks
        tasks = [
            Task(
                id=f"concurrent_task_{i}",
                description=f"Concurrent processing test {i}",
                requirements={"test_id": i}
            )
            for i in range(10)
        ]
        
        start_time = time.perf_counter()
        
        # Process tasks concurrently
        results = await asyncio.gather(*[
            orchestrator.delegate_task(task) for task in tasks
        ])
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        # All tasks should complete
        assert len(results) == 10
        assert all(result is not None for result in results)
        assert orchestrator.total_tasks_completed == 10
        
        # Should complete in reasonable time
        assert processing_time < 5.0  # Less than 5 seconds for 10 tasks
    
    @pytest.mark.asyncio
    async def test_large_swarm_performance(self, orchestrator):
        """Test performance with large swarm"""
        large_swarm_size = 20
        objective = "Large scale optimization test"
        
        start_time = time.perf_counter()
        
        result = await orchestrator.swarm_intelligence(objective, large_swarm_size)
        
        end_time = time.perf_counter()
        processing_time = end_time - start_time
        
        assert result is not None
        assert len(orchestrator.agents) >= large_swarm_size
        
        # Should complete within reasonable time even for large swarm
        assert processing_time < 30.0  # Less than 30 seconds
    
    @pytest.mark.asyncio
    async def test_memory_usage_with_many_agents(self, orchestrator, mock_agents):
        """Test memory usage with many registered agents"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Register many agents
        extended_agents = mock_agents * 10  # 40 agents total
        for agent in extended_agents:
            orchestrator.register_agent(agent)
        
        # Process tasks with all agents
        for i in range(20):
            task = Task(
                id=f"memory_test_{i}",
                description="Memory usage test",
                requirements={"iteration": i}
            )
            await orchestrator.delegate_task(task)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 100  # Less than 100MB increase


class TestErrorHandlingAndRecovery(TestAgentOrchestrator):
    """Test error handling and recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_agent_failure_handling(self, orchestrator):
        """Test handling of individual agent failures"""
        
        class FailingAgent(BaseAgent):
            async def execute(self, task, action: Action):
                raise RuntimeError("Agent processing failed")
        
        class WorkingAgent(BaseAgent):
            async def execute(self, task, action: Action):
                return {"status": "success", "agent": self.name}
        
        failing_agent = FailingAgent("failing_agent")
        working_agent = WorkingAgent("working_agent")
        
        orchestrator.register_agent(failing_agent)
        orchestrator.register_agent(working_agent)
        
        task = Task(
            id="error_recovery_test",
            description="Test error recovery",
            requirements={}
        )
        
        # Should still be able to delegate to working agent
        result = await orchestrator.delegate_task(task)
        
        # Should get result from working agent (orchestrator should handle failures)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_partial_parallel_execution_failure(self, orchestrator):
        """Test handling of partial failures in parallel execution"""
        
        class MixedReliabilityAgent(BaseAgent):
            def __init__(self, name: str, should_fail: bool = False):
                super().__init__(name=name)
                self.should_fail = should_fail
            
            async def execute(self, task, action: Action):
                if self.should_fail:
                    raise Exception("Simulated failure")
                return {"agent": self.name, "success": True}
        
        agents = [
            MixedReliabilityAgent("reliable_1", False),
            MixedReliabilityAgent("failing", True),
            MixedReliabilityAgent("reliable_2", False)
        ]
        
        task = Task(id="partial_failure_test", description="Test partial failures", requirements={})
        
        results = await orchestrator.parallel_execution(agents, task)
        
        # Should get results from non-failing agents
        assert len(results) == 2  # Two successful agents
        assert all("success" in result and result["success"] for result in results)
    
    @pytest.mark.asyncio
    async def test_blackboard_resilience(self, orchestrator):
        """Test blackboard resilience to errors"""
        blackboard = orchestrator.blackboard
        
        # Test with invalid data
        await blackboard.write("agent_1", "test_key", {"valid": True})
        
        # Should handle read of non-existent key gracefully
        result = await blackboard.read("non_existent_key")
        assert result is None
        
        # Should handle query with empty pattern
        results = await blackboard.query({})
        assert isinstance(results, dict)


class TestSpecializedAgents(TestAgentOrchestrator):
    """Test specialized agent implementations"""
    
    @pytest.mark.asyncio
    async def test_customer_support_agent(self, orchestrator):
        """Test CustomerSupportAgent functionality"""
        support_agent = CustomerSupportAgent("support_agent")
        orchestrator.register_agent(support_agent)
        
        task = Task(
            id="support_task",
            description="Handle customer inquiry about billing",
            requirements={
                "customer_id": "12345",
                "inquiry_type": "billing",
                "priority": "high"
            }
        )
        
        result = await orchestrator.delegate_task(task)
        
        assert result is not None
        assert "response" in result
        assert "satisfaction_score" in result
        assert isinstance(result["satisfaction_score"], (int, float))
    
    @pytest.mark.asyncio
    async def test_data_analyst_agent(self, orchestrator):
        """Test DataAnalystAgent functionality"""
        analyst_agent = DataAnalystAgent("analyst_agent")
        orchestrator.register_agent(analyst_agent)
        
        task = Task(
            id="analysis_task",
            description="Analyze sales data for Q4 trends",
            requirements={
                "data_source": "sales_db",
                "time_period": "Q4_2024",
                "metrics": ["revenue", "conversion_rate"]
            }
        )
        
        result = await orchestrator.delegate_task(task)
        
        assert result is not None
        assert "analysis" in result
        assert "visualizations" in result
        assert isinstance(result["visualizations"], list)


class TestMetricsAndObservability(TestAgentOrchestrator):
    """Test metrics collection and observability"""
    
    @pytest.mark.asyncio
    async def test_orchestrator_metrics_collection(self, orchestrator, mock_agents):
        """Test metrics collection during orchestration"""
        for agent in mock_agents[:2]:
            orchestrator.register_agent(agent)
        
        # Process some tasks
        for i in range(5):
            task = Task(
                id=f"metrics_test_{i}",
                description="Metrics collection test",
                requirements={"test_number": i}
            )
            await orchestrator.delegate_task(task)
        
        metrics = orchestrator.get_metrics()
        
        assert metrics["name"] == "test_orchestrator"
        assert metrics["registered_agents"] == 2
        assert metrics["completed_tasks"] == 5
        assert "agent_metrics" in metrics
        
        # Check individual agent metrics
        agent_metrics = metrics["agent_metrics"]
        for agent_name, agent_metric in agent_metrics.items():
            assert "total_tasks" in agent_metric
            assert "success_rate" in agent_metric
            assert agent_metric["total_tasks"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])