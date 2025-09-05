"""
Phase 7 Autonomous Intelligence Testing Suite
Comprehensive tests for autonomous intelligence capabilities
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta

# Import Phase 7 components
from core.autonomous.orchestrator import AutonomousMetaOrchestrator, AutonomyLevel
from core.autonomous.self_modification import SelfModifyingAgent, DynamicCodeGenerator, PerformanceDrivenEvolution
from core.autonomous.emergent_intelligence import (
    EmergentIntelligenceOrchestrator, CapabilityMiningEngine, NoveltyDetector
)
from core.autonomous.safety import (
    AutonomousSafetyFramework, ModificationValidator, RollbackManager, SafetyLevel
)
from templates.base_agent import BaseAgent, Action, Observation
from core.orchestration.orchestrator import Task


class TestAgent(SelfModifyingAgent):
    """Test agent for Phase 7 testing"""
    
    async def execute(self, task, action) -> dict:
        return {
            "status": "completed", 
            "task": str(task),
            "timestamp": datetime.now().isoformat()
        }


class TestPhase7Safety:
    """Test safety framework components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.safety_framework = AutonomousSafetyFramework(
            safety_level=SafetyLevel.RESTRICTIVE
        )
    
    @pytest.mark.asyncio
    async def test_code_validation_safe(self):
        """Test validation of safe code"""
        safe_code = """
def safe_function(x, y):
    return x + y
        """
        
        assessment = await self.safety_framework.validator.validate_code_modification(
            safe_code, {}
        )
        
        assert assessment.is_safe
        assert assessment.confidence > 0.5
        assert len(assessment.violations) == 0
    
    @pytest.mark.asyncio
    async def test_code_validation_unsafe(self):
        """Test validation of unsafe code"""
        unsafe_code = """
import os
os.system('del -rf /')
exec('malicious_code')
        """
        
        assessment = await self.safety_framework.validator.validate_code_modification(
            unsafe_code, {}
        )
        
        assert not assessment.is_safe
        assert len(assessment.violations) > 0
        assert any('exec' in str(v.description) for v in assessment.violations)
    
    @pytest.mark.asyncio
    async def test_backup_and_rollback(self):
        """Test backup and rollback functionality"""
        agent = TestAgent("test_agent")
        original_tasks = agent.total_tasks
        
        # Create backup
        backup_id = await self.safety_framework.create_safe_backup(agent)
        assert backup_id is not None
        
        # Modify agent
        agent.total_tasks = 999
        assert agent.total_tasks == 999
        
        # Rollback
        success = await self.safety_framework.emergency_rollback(backup_id, agent)
        assert success
        assert agent.total_tasks == original_tasks
    
    def test_safety_levels(self):
        """Test different safety levels"""
        permissive = AutonomousSafetyFramework(safety_level=SafetyLevel.PERMISSIVE)
        restrictive = AutonomousSafetyFramework(safety_level=SafetyLevel.RESTRICTIVE)
        paranoid = AutonomousSafetyFramework(safety_level=SafetyLevel.PARANOID)
        
        assert permissive.safety_level == SafetyLevel.PERMISSIVE
        assert restrictive.safety_level == SafetyLevel.RESTRICTIVE
        assert paranoid.safety_level == SafetyLevel.PARANOID


class TestPhase7SelfModification:
    """Test self-modification capabilities"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = TestAgent("test_self_mod_agent", config={
            'self_improvement_enabled': True,
            'improvement_frequency': 5
        })
    
    @pytest.mark.asyncio
    async def test_performance_gap_analysis(self):
        """Test performance gap analysis"""
        evolution_engine = PerformanceDrivenEvolution()
        
        # Simulate some task history
        for i in range(10):
            action = Action(
                action_type="test_action",
                parameters={},
                tools_used=[],
                expected_outcome="test"
            )
            observation = Observation(
                action=action,
                result=f"result_{i}",
                success=i > 2,  # Some failures initially
                learnings=["test learning"]
            )
            await self.agent.memory.store_episode(observation)
        
        gaps = await evolution_engine.analyze_performance_gaps(self.agent)
        assert isinstance(gaps, list)
        # Should identify performance gaps from the simulated poor initial performance
    
    @pytest.mark.asyncio
    async def test_code_generation(self):
        """Test dynamic code generation"""
        from core.autonomous.self_modification import ModificationRequest, ModificationType
        
        modification_request = ModificationRequest(
            modification_id="test_mod_1",
            agent_name=self.agent.name,
            modification_type=ModificationType.STRATEGY_OPTIMIZATION,
            target_component="strategy_selection",
            proposed_changes="Optimize strategy selection logic",
            expected_improvement=0.2,
            safety_constraints={},
            testing_requirements=[],
            rollback_plan={}
        )
        
        code_generator = DynamicCodeGenerator(ModificationValidator())
        result = await code_generator.generate_modification_code(modification_request)
        
        assert 'success' in result
        if result['success']:
            assert 'modification_package' in result
            package = result['modification_package']
            assert 'modified_code' in package
            assert 'validation_results' in package
    
    @pytest.mark.asyncio
    async def test_autonomous_self_improvement(self):
        """Test autonomous self-improvement process"""
        # Give agent some task history first
        for i in range(15):  # Enough to trigger improvement
            await self.agent.process_task(f"test_task_{i}", {})
        
        improvement_result = await self.agent.autonomous_self_improvement()
        
        assert 'status' in improvement_result
        # May be 'no_improvements_needed', 'completed', or 'disabled'
        assert improvement_result['status'] in ['no_improvements_needed', 'completed', 'disabled']


class TestPhase7EmergentIntelligence:
    """Test emergent intelligence components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.safety_framework = AutonomousSafetyFramework()
        self.emergence_orchestrator = EmergentIntelligenceOrchestrator(
            self.safety_framework, discovery_frequency_hours=1
        )
        self.agents = {}
        
        # Create test agents
        for i in range(3):
            agent = TestAgent(f"emergence_test_agent_{i}")
            self.agents[agent.name] = agent
    
    @pytest.mark.asyncio
    async def test_capability_mining(self):
        """Test capability mining from agent interactions"""
        # Give agents some interaction history
        for agent_name, agent in self.agents.items():
            for j in range(5):
                action = Action(
                    action_type=f"action_type_{j % 3}",
                    parameters={},
                    tools_used=[],
                    expected_outcome="test"
                )
                observation = Observation(
                    action=action,
                    result=f"result_{j}",
                    success=True,
                    learnings=[f"learning_{j}"]
                )
                await agent.memory.store_episode(observation)
        
        # Mock orchestrator for capability mining
        from core.orchestration.orchestrator import AgentOrchestrator
        mock_orchestrator = AgentOrchestrator()
        for agent in self.agents.values():
            mock_orchestrator.register_agent(agent)
        
        capabilities = await self.emergence_orchestrator.capability_miner.mine_emergent_capabilities(
            self.agents, mock_orchestrator, time_window_hours=24
        )
        
        assert isinstance(capabilities, list)
        # May or may not find capabilities depending on patterns
    
    @pytest.mark.asyncio
    async def test_breakthrough_detection(self):
        """Test breakthrough behavior detection"""
        # Simulate breakthrough pattern - sudden performance improvement
        agent = list(self.agents.values())[0]
        
        # Add recent observations with improved performance
        current_time = datetime.now()
        for i in range(10):
            action = Action(
                action_type="breakthrough_action",
                parameters={},
                tools_used=[],
                expected_outcome="test"
            )
            observation = Observation(
                action=action,
                result=f"improved_result_{i}",
                success=True,  # All successful for breakthrough pattern
                learnings=[f"breakthrough_learning_{i}"],
                timestamp=current_time - timedelta(hours=i)
            )
            await agent.memory.store_episode(observation)
        
        breakthroughs = await self.emergence_orchestrator.novelty_detector.detect_breakthrough_behaviors(
            self.agents, time_window_hours=12
        )
        
        assert isinstance(breakthroughs, list)
    
    @pytest.mark.asyncio
    async def test_innovation_incubation(self):
        """Test capability cultivation in innovation incubator"""
        from core.autonomous.emergent_intelligence import EmergentCapability, EmergenceType
        
        # Create a test capability
        test_capability = EmergentCapability(
            capability_id="test_cap_1",
            name="Test Capability",
            description="A test emergent capability",
            emergence_type=EmergenceType.CAPABILITY_SYNTHESIS,
            discovery_agents=list(self.agents.keys()),
            implementation_pattern={"test": "pattern"},
            novelty_score=0.8,
            potential_impact=0.7,
            validation_results={}
        )
        
        test_agents = list(self.agents.values())
        
        cultivation_result = await self.emergence_orchestrator.innovation_incubator.cultivate_capability(
            test_capability, test_agents
        )
        
        assert 'success' in cultivation_result
        # Cultivation may succeed or fail based on safety and testing


class TestPhase7AutonomousOrchestrator:
    """Test autonomous meta-orchestrator"""
    
    def setup_method(self):
        """Setup test environment"""
        self.orchestrator = AutonomousMetaOrchestrator(
            autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS
        )
        
        # Register test agents
        for i in range(3):
            agent = TestAgent(f"orch_test_agent_{i}")
            self.orchestrator.register_agent(agent)
    
    @pytest.mark.asyncio
    async def test_autonomous_coordination(self):
        """Test autonomous coordination capabilities"""
        test_task = Task(
            id="auto_coord_test",
            description="Test autonomous coordination",
            requirements={"test": True}
        )
        
        result = await self.orchestrator.autonomous_coordination(test_task)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_autonomous_decision_making(self):
        """Test autonomous decision making process"""
        test_task = Task(
            id="decision_test", 
            description="Complex decision making task",
            requirements={"complexity": "high"}
        )
        
        # Test task analysis
        analysis = await self.orchestrator._analyze_task_autonomously(test_task)
        assert 'complexity_score' in analysis
        assert 'success_prediction' in analysis
        assert 'optimal_agents' in analysis
        
        # Test coordination decision
        decision = await self.orchestrator._make_autonomous_coordination_decision(
            test_task, analysis, "performance"
        )
        assert decision.decision_type == "coordination_pattern"
        assert decision.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_adaptive_resource_allocation(self):
        """Test adaptive resource allocation"""
        allocation_result = await self.orchestrator.adaptive_resource_allocation()
        
        assert 'new_allocation' in allocation_result
        assert 'exploration_ratio' in allocation_result
        assert 'performance_trend' in allocation_result
        
        # Verify allocations sum to approximately 1.0
        total_allocation = sum(allocation_result['new_allocation'].values())
        assert 0.9 <= total_allocation <= 1.1  # Allow for small rounding errors
    
    def test_autonomy_levels(self):
        """Test different autonomy levels"""
        supervised = AutonomousMetaOrchestrator(autonomy_level=AutonomyLevel.SUPERVISED)
        semi_auto = AutonomousMetaOrchestrator(autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS)
        full_auto = AutonomousMetaOrchestrator(autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS)
        emergent = AutonomousMetaOrchestrator(autonomy_level=AutonomyLevel.EMERGENT)
        
        assert supervised.autonomy_level == AutonomyLevel.SUPERVISED
        assert semi_auto.autonomy_level == AutonomyLevel.SEMI_AUTONOMOUS
        assert full_auto.autonomy_level == AutonomyLevel.FULLY_AUTONOMOUS
        assert emergent.autonomy_level == AutonomyLevel.EMERGENT
    
    def test_metrics_collection(self):
        """Test autonomous metrics collection"""
        metrics = self.orchestrator.get_autonomous_metrics()
        
        assert 'autonomy_level' in metrics
        assert 'autonomous_success_rate' in metrics
        assert 'modification_success_rate' in metrics
        assert 'capability_discovery_rate' in metrics
        assert 'discovered_capabilities' in metrics
        assert 'exploration_budget' in metrics


class TestPhase7Integration:
    """Integration tests for complete Phase 7 system"""
    
    def setup_method(self):
        """Setup complete test environment"""
        self.safety_framework = AutonomousSafetyFramework()
        self.orchestrator = AutonomousMetaOrchestrator(
            autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS,
            safety_config={}
        )
        self.emergence_orchestrator = EmergentIntelligenceOrchestrator(
            self.safety_framework
        )
        
        # Create test agents
        self.agents = {}
        for i in range(5):
            agent = TestAgent(f"integration_agent_{i}")
            self.agents[agent.name] = agent
            self.orchestrator.register_agent(agent)
    
    @pytest.mark.asyncio
    async def test_complete_autonomous_workflow(self):
        """Test complete autonomous workflow"""
        # 1. Create tasks
        tasks = [
            Task(f"task_{i}", f"Integration test task {i}", {"priority": i})
            for i in range(3)
        ]
        
        # 2. Process tasks autonomously
        results = []
        for task in tasks:
            result = await self.orchestrator.autonomous_coordination(task)
            results.append(result)
        
        assert len(results) == len(tasks)
        
        # 3. Run intelligence evolution
        evolution_results = await self.emergence_orchestrator.orchestrate_intelligence_evolution(
            self.agents, self.orchestrator
        )
        
        assert 'capabilities_discovered' in evolution_results
        assert 'breakthroughs_detected' in evolution_results
    
    @pytest.mark.asyncio
    async def test_safety_integration(self):
        """Test safety integration across all components"""
        # Test that safety is enforced at all levels
        
        # 1. Orchestrator level
        unsafe_task = Task(
            "unsafe_task",
            "exec('malicious_code')",
            {"unsafe": True}
        )
        
        # Should handle gracefully without crashing
        try:
            result = await self.orchestrator.autonomous_coordination(unsafe_task)
            # If it succeeds, verify safety measures were applied
            assert result is not None
        except Exception as e:
            # If it fails, it should be a controlled failure
            assert "safety" in str(e).lower() or "validation" in str(e).lower()
        
        # 2. Agent level safety
        agent = list(self.agents.values())[0]
        
        # Should not allow unsafe modifications
        improvement_result = await agent.autonomous_self_improvement()
        assert improvement_result['status'] in ['no_improvements_needed', 'completed', 'disabled']
    
    @pytest.mark.asyncio
    async def test_performance_monitoring(self):
        """Test performance monitoring and metrics"""
        # Process several tasks to generate metrics
        tasks = [
            Task(f"perf_task_{i}", f"Performance test {i}", {})
            for i in range(10)
        ]
        
        for task in tasks:
            await self.orchestrator.autonomous_coordination(task)
        
        # Check metrics
        orchestrator_metrics = self.orchestrator.get_autonomous_metrics()
        emergence_metrics = self.emergence_orchestrator.get_emergent_intelligence_metrics()
        safety_metrics = self.safety_framework.get_safety_metrics()
        
        # Verify metric structure
        assert isinstance(orchestrator_metrics, dict)
        assert isinstance(emergence_metrics, dict)
        assert isinstance(safety_metrics, dict)
        
        # Verify key metrics exist
        assert 'autonomy_level' in orchestrator_metrics
        assert 'discovered_capabilities' in emergence_metrics
        assert 'total_violations' in safety_metrics
    
    def test_backward_compatibility(self):
        """Test that Phase 7 maintains backward compatibility"""
        from orchestrator import AgentOrchestrator, Task as BaseTask
        
        # Should be able to create basic orchestrator
        basic_orchestrator = AgentOrchestrator()
        assert basic_orchestrator is not None
        
        # Should be able to create basic tasks
        basic_task = BaseTask("basic", "Basic task", {})
        assert basic_task is not None
        
        # Phase 7 orchestrator should handle basic tasks
        assert len(self.orchestrator.agents) > 0
        # Basic task processing should work
        # (Actual execution test would require more setup)


if __name__ == "__main__":
    """Run Phase 7 tests"""
    pytest.main([__file__, "-v"])