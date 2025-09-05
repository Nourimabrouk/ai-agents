"""
Phase 7 Self-Modification Testing Suite
Tests autonomous improvement capabilities with 15% performance target
Validates self-modifying agents, dynamic code generation, and performance-driven evolution
"""

import asyncio
import pytest
import tempfile
import shutil
import json
import time
import statistics
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch
import numpy as np

# Import Phase 7 self-modification components
from core.autonomous.self_modification import (
    SelfModifyingAgent, DynamicCodeGenerator, PerformanceDrivenEvolution,
    ModificationType, ModificationRequest, EvolutionStrategy
)
from core.autonomous.safety import AutonomousSafetyFramework, SafetyLevel
from core.security.autonomous_security import AutonomousSecurityFramework, SecurityLevel
from templates.base_agent import BaseAgent, AgentState
from . import PHASE7_TEST_CONFIG


class PerformanceTestAgent(SelfModifyingAgent):
    """Test agent that can demonstrate autonomous improvement"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.performance_history = []
        self.modification_history = []
        self.baseline_performance = {
            'response_time': 1.0,
            'accuracy': 0.80,
            'throughput': 10.0,
            'error_rate': 0.05
        }
        self.current_performance = self.baseline_performance.copy()
        
    async def execute_benchmark_task(self, complexity: float = 0.5) -> Dict[str, Any]:
        """Execute a benchmark task to measure performance"""
        start_time = time.perf_counter()
        
        # Simulate task processing based on current performance
        processing_time = self.current_performance['response_time'] * complexity
        await asyncio.sleep(processing_time * 0.1)  # Scaled for testing
        
        # Simulate accuracy and errors
        success = np.random.random() > self.current_performance['error_rate']
        accuracy = self.current_performance['accuracy'] if success else 0.0
        
        execution_time = time.perf_counter() - start_time
        
        result = {
            'success': success,
            'accuracy': accuracy,
            'execution_time': execution_time,
            'complexity': complexity,
            'timestamp': datetime.now()
        }
        
        self.performance_history.append(result)
        return result
        
    async def apply_performance_modification(self, modification: Dict[str, Any]) -> bool:
        """Apply a performance improvement modification"""
        mod_type = modification.get('type')
        improvement_factor = modification.get('improvement_factor', 1.05)
        
        old_performance = self.current_performance.copy()
        
        if mod_type == 'response_time_optimization':
            self.current_performance['response_time'] *= (2.0 - improvement_factor)  # Inverse for time
        elif mod_type == 'accuracy_enhancement':
            self.current_performance['accuracy'] = min(1.0, 
                self.current_performance['accuracy'] * improvement_factor)
        elif mod_type == 'throughput_boost':
            self.current_performance['throughput'] *= improvement_factor
        elif mod_type == 'error_reduction':
            self.current_performance['error_rate'] *= (2.0 - improvement_factor)  # Inverse for errors
            
        self.modification_history.append({
            'type': mod_type,
            'improvement_factor': improvement_factor,
            'old_performance': old_performance,
            'new_performance': self.current_performance.copy(),
            'timestamp': datetime.now()
        })
        
        return True
        
    def calculate_performance_improvement(self) -> float:
        """Calculate overall performance improvement from baseline"""
        if not self.performance_history:
            return 0.0
            
        # Composite performance score
        def performance_score(perf):
            return (
                (1.0 / perf['response_time']) * 0.3 +  # Lower is better
                perf['accuracy'] * 0.3 +
                perf['throughput'] * 0.002 +  # Scale down
                (1.0 - perf['error_rate']) * 0.4  # Lower error rate is better
            )
            
        baseline_score = performance_score(self.baseline_performance)
        current_score = performance_score(self.current_performance)
        
        improvement = (current_score - baseline_score) / baseline_score
        return improvement
        
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        if not self.performance_history:
            return {}
            
        recent_results = self.performance_history[-10:]  # Last 10 results
        
        return {
            'avg_response_time': statistics.mean(r['execution_time'] for r in recent_results),
            'success_rate': statistics.mean(r['success'] for r in recent_results),
            'avg_accuracy': statistics.mean(r['accuracy'] for r in recent_results if r['success']),
            'total_executions': len(self.performance_history),
            'performance_improvement': self.calculate_performance_improvement()
        }


class TestSelfModifyingAgentCapabilities:
    """Test basic self-modification capabilities"""
    
    @pytest.fixture
    async def self_modifying_agent(self):
        """Create self-modifying agent for testing"""
        safety_framework = AutonomousSafetyFramework(safety_level=SafetyLevel.MODERATE)
        security_framework = AutonomousSecurityFramework(security_level=SecurityLevel.MEDIUM)
        
        agent = PerformanceTestAgent("self_mod_test_001")
        await agent.initialize_self_modification(
            safety_framework=safety_framework,
            security_framework=security_framework
        )
        return agent
        
    @pytest.mark.asyncio
    async def test_code_generation_capabilities(self, self_modifying_agent):
        """Test dynamic code generation for improvements"""
        agent = self_modifying_agent
        
        # Request code generation for performance improvement
        improvement_request = ModificationRequest(
            modification_type=ModificationType.PERFORMANCE_TUNING,
            target_metric='response_time',
            improvement_target=0.20,  # 20% improvement
            description="Optimize response time through caching"
        )
        
        generated_code = await agent.generate_improvement_code(improvement_request)
        
        assert generated_code is not None
        assert generated_code.modification_type == ModificationType.PERFORMANCE_TUNING
        assert 'cache' in generated_code.code_content.lower()
        assert generated_code.safety_validated == True
        assert generated_code.estimated_improvement > 0.15
        
        # Test code application
        application_result = await agent.apply_generated_code(generated_code)
        
        assert application_result['success'] == True
        assert application_result['performance_impact'] > 0.10
        assert 'rollback_plan' in application_result
        
        print(f"Generated code improvement: {application_result['performance_impact']:.1%}")
        
    @pytest.mark.asyncio
    async def test_strategy_modification(self, self_modifying_agent):
        """Test modification of agent strategies"""
        agent = self_modifying_agent
        
        # Baseline strategy performance
        baseline_tasks = []
        for _ in range(10):
            result = await agent.execute_benchmark_task(0.5)
            baseline_tasks.append(result)
            
        baseline_avg_time = statistics.mean(r['execution_time'] for r in baseline_tasks)
        
        # Apply strategy modification
        strategy_mod = {
            'type': 'response_time_optimization',
            'improvement_factor': 1.25,  # 25% improvement
            'strategy_changes': ['implement_caching', 'parallel_processing']
        }
        
        await agent.apply_performance_modification(strategy_mod)
        
        # Test performance after modification
        modified_tasks = []
        for _ in range(10):
            result = await agent.execute_benchmark_task(0.5)
            modified_tasks.append(result)
            
        modified_avg_time = statistics.mean(r['execution_time'] for r in modified_tasks)
        
        # Should see performance improvement
        time_improvement = (baseline_avg_time - modified_avg_time) / baseline_avg_time
        assert time_improvement > 0.15, f"Time improvement {time_improvement:.1%} below 15%"
        
        print(f"Strategy modification improvement: {time_improvement:.1%}")
        
    @pytest.mark.asyncio
    async def test_capability_extension(self, self_modifying_agent):
        """Test addition of new capabilities"""
        agent = self_modifying_agent
        
        # Check initial capabilities
        initial_capabilities = await agent.get_capabilities()
        initial_count = len(initial_capabilities)
        
        # Request capability extension
        extension_request = ModificationRequest(
            modification_type=ModificationType.CAPABILITY_EXTENSION,
            target_capability='advanced_analytics',
            description="Add statistical analysis capabilities"
        )
        
        extension_result = await agent.extend_capabilities(extension_request)
        
        assert extension_result['success'] == True
        assert 'advanced_analytics' in extension_result['new_capabilities']
        
        # Verify capabilities were added
        updated_capabilities = await agent.get_capabilities()
        assert len(updated_capabilities) > initial_count
        assert 'advanced_analytics' in updated_capabilities
        
        # Test new capability
        analytics_result = await agent.execute_capability('advanced_analytics', 
                                                         {'data': [1, 2, 3, 4, 5]})
        assert analytics_result['success'] == True
        assert 'statistics' in analytics_result['output']
        
        print(f"Added {len(updated_capabilities) - initial_count} new capabilities")
        
    @pytest.mark.asyncio
    async def test_learning_enhancement(self, self_modifying_agent):
        """Test enhancement of learning algorithms"""
        agent = self_modifying_agent
        
        # Baseline learning performance
        baseline_learning_rate = await agent.measure_learning_rate()
        
        # Apply learning enhancement
        learning_mod = ModificationRequest(
            modification_type=ModificationType.LEARNING_ENHANCEMENT,
            target_metric='learning_rate',
            improvement_target=0.30,  # 30% faster learning
            description="Enhance learning algorithm efficiency"
        )
        
        enhancement_result = await agent.enhance_learning_system(learning_mod)
        
        assert enhancement_result['success'] == True
        assert enhancement_result['improvement_factor'] > 1.2
        
        # Test enhanced learning
        enhanced_learning_rate = await agent.measure_learning_rate()
        
        learning_improvement = (enhanced_learning_rate - baseline_learning_rate) / baseline_learning_rate
        assert learning_improvement > 0.25, f"Learning improvement {learning_improvement:.1%} below 25%"
        
        print(f"Learning enhancement improvement: {learning_improvement:.1%}")


class TestPerformanceDrivenEvolution:
    """Test autonomous performance evolution achieving 15% improvement target"""
    
    @pytest.fixture
    async def evolution_system(self):
        """Create performance-driven evolution system"""
        target_improvement = PHASE7_TEST_CONFIG["performance_targets"]["autonomous_improvement"]
        
        evolution_system = PerformanceDrivenEvolution(
            improvement_target=target_improvement,
            evolution_strategies=['genetic_programming', 'gradient_based', 'random_search'],
            measurement_window=timedelta(minutes=1),  # Short for testing
            min_samples=10
        )
        await evolution_system.initialize()
        return evolution_system
        
    @pytest.fixture
    async def evolving_agents(self, evolution_system):
        """Create population of evolving agents"""
        agents = []
        for i in range(5):
            agent = PerformanceTestAgent(f"evolving_agent_{i}")
            await evolution_system.register_agent(agent)
            agents.append(agent)
        return agents
        
    @pytest.mark.asyncio
    async def test_15_percent_improvement_target(self, evolution_system, evolving_agents):
        """Test system achieves 15% autonomous improvement target"""
        target_improvement = PHASE7_TEST_CONFIG["performance_targets"]["autonomous_improvement"]
        
        # Baseline performance measurement
        baseline_metrics = {}
        for agent in evolving_agents:
            # Execute baseline tasks
            for _ in range(10):
                await agent.execute_benchmark_task(0.5)
            baseline_metrics[agent.agent_id] = agent.get_performance_metrics()
        
        print("Baseline performance:")
        for agent_id, metrics in baseline_metrics.items():
            print(f"  {agent_id}: improvement={metrics.get('performance_improvement', 0):.1%}")
        
        # Run evolution cycles
        evolution_cycles = 8
        for cycle in range(evolution_cycles):
            print(f"\nEvolution cycle {cycle + 1}/{evolution_cycles}")
            
            # Performance analysis and improvement generation
            improvements = await evolution_system.analyze_and_generate_improvements(evolving_agents)
            
            print(f"Generated {len(improvements)} improvements")
            
            # Apply improvements to agents
            for agent_id, improvement in improvements.items():
                agent = next(a for a in evolving_agents if a.agent_id == agent_id)
                await agent.apply_performance_modification(improvement)
            
            # Execute tasks to measure new performance
            for agent in evolving_agents:
                for _ in range(5):
                    await agent.execute_benchmark_task(np.random.uniform(0.3, 0.8))
                    
            # Brief pause between cycles
            await asyncio.sleep(0.1)
        
        # Final performance measurement
        final_metrics = {}
        for agent in evolving_agents:
            final_metrics[agent.agent_id] = agent.get_performance_metrics()
        
        print("\nFinal performance:")
        improvements_achieved = []
        for agent_id, metrics in final_metrics.items():
            improvement = metrics.get('performance_improvement', 0)
            improvements_achieved.append(improvement)
            print(f"  {agent_id}: improvement={improvement:.1%}")
        
        # Verify improvement target met
        avg_improvement = statistics.mean(improvements_achieved)
        max_improvement = max(improvements_achieved)
        agents_meeting_target = sum(1 for imp in improvements_achieved if imp >= target_improvement)
        
        print(f"\nImprovement summary:")
        print(f"  Average improvement: {avg_improvement:.1%}")
        print(f"  Maximum improvement: {max_improvement:.1%}")
        print(f"  Agents meeting target: {agents_meeting_target}/{len(evolving_agents)}")
        
        # Assertions for 15% improvement target
        assert avg_improvement >= target_improvement, f"Average improvement {avg_improvement:.1%} below target {target_improvement:.1%}"
        assert agents_meeting_target >= 3, f"Only {agents_meeting_target} agents met target, need at least 3"
        
    @pytest.mark.asyncio
    async def test_evolution_strategy_effectiveness(self, evolution_system, evolving_agents):
        """Test effectiveness of different evolution strategies"""
        # Test different evolution strategies
        strategies = ['genetic_programming', 'gradient_based', 'random_search']
        strategy_results = {}
        
        for strategy in strategies:
            print(f"\nTesting evolution strategy: {strategy}")
            
            # Reset agents to baseline
            for agent in evolving_agents:
                agent.current_performance = agent.baseline_performance.copy()
                agent.performance_history = []
                agent.modification_history = []
            
            # Configure evolution system for this strategy
            await evolution_system.set_evolution_strategy(strategy)
            
            # Run evolution with this strategy
            improvements = []
            for cycle in range(5):
                cycle_improvements = await evolution_system.analyze_and_generate_improvements(
                    evolving_agents, strategy=strategy
                )
                
                # Apply and measure
                for agent_id, improvement in cycle_improvements.items():
                    agent = next(a for a in evolving_agents if a.agent_id == agent_id)
                    await agent.apply_performance_modification(improvement)
                    
                for agent in evolving_agents:
                    for _ in range(3):
                        await agent.execute_benchmark_task(0.5)
                    
            # Measure final improvement for this strategy
            strategy_improvements = []
            for agent in evolving_agents:
                metrics = agent.get_performance_metrics()
                improvement = metrics.get('performance_improvement', 0)
                strategy_improvements.append(improvement)
                
            avg_strategy_improvement = statistics.mean(strategy_improvements)
            strategy_results[strategy] = avg_strategy_improvement
            
            print(f"  Average improvement with {strategy}: {avg_strategy_improvement:.1%}")
        
        # Verify strategies show different effectiveness
        best_strategy = max(strategy_results, key=strategy_results.get)
        best_improvement = strategy_results[best_strategy]
        
        assert best_improvement > 0.10, f"Best strategy only achieved {best_improvement:.1%}"
        
        # At least one strategy should significantly outperform random
        random_improvement = strategy_results.get('random_search', 0)
        better_strategies = [imp for strat, imp in strategy_results.items() 
                           if strat != 'random_search' and imp > random_improvement + 0.05]
        
        assert len(better_strategies) > 0, "No strategy significantly outperformed random search"
        
        print(f"Best strategy: {best_strategy} ({best_improvement:.1%} improvement)")
        
    @pytest.mark.asyncio
    async def test_adaptive_evolution_parameters(self, evolution_system, evolving_agents):
        """Test adaptive adjustment of evolution parameters"""
        # Initial evolution parameters
        initial_params = await evolution_system.get_evolution_parameters()
        
        # Run evolution with parameter adaptation enabled
        await evolution_system.enable_parameter_adaptation()
        
        parameter_history = []
        for cycle in range(6):
            # Record current parameters
            current_params = await evolution_system.get_evolution_parameters()
            parameter_history.append(current_params.copy())
            
            # Run evolution cycle
            improvements = await evolution_system.analyze_and_generate_improvements(evolving_agents)
            
            for agent_id, improvement in improvements.items():
                agent = next(a for a in evolving_agents if a.agent_id == agent_id)
                await agent.apply_performance_modification(improvement)
            
            # Execute tasks
            for agent in evolving_agents:
                for _ in range(3):
                    await agent.execute_benchmark_task(0.5)
            
            # Evolution system should adapt parameters based on results
            await evolution_system.adapt_parameters_based_on_performance()
        
        # Verify parameters adapted over time
        final_params = await evolution_system.get_evolution_parameters()
        
        # At least some parameters should have changed
        param_changes = 0
        for key in initial_params:
            if abs(final_params[key] - initial_params[key]) > 0.1:
                param_changes += 1
                
        assert param_changes >= 2, f"Only {param_changes} parameters adapted"
        
        # Performance should improve with adaptation
        final_performance = statistics.mean(
            agent.calculate_performance_improvement() for agent in evolving_agents
        )
        
        assert final_performance > 0.08, f"Final performance {final_performance:.1%} too low"
        
        print(f"Parameter adaptation resulted in {final_performance:.1%} improvement")
        
    @pytest.mark.asyncio
    async def test_multi_objective_optimization(self, evolution_system, evolving_agents):
        """Test optimization across multiple performance objectives"""
        # Define multiple objectives
        objectives = ['response_time', 'accuracy', 'throughput', 'error_rate']
        
        # Configure multi-objective optimization
        await evolution_system.configure_multi_objective_optimization(
            objectives=objectives,
            weights={'response_time': 0.3, 'accuracy': 0.3, 'throughput': 0.2, 'error_rate': 0.2}
        )
        
        # Baseline measurements across all objectives
        baseline_scores = {}
        for agent in evolving_agents:
            for _ in range(10):
                await agent.execute_benchmark_task(0.5)
            baseline_scores[agent.agent_id] = await evolution_system.calculate_multi_objective_score(agent)
        
        # Run multi-objective evolution
        for cycle in range(6):
            improvements = await evolution_system.generate_multi_objective_improvements(evolving_agents)
            
            for agent_id, improvement in improvements.items():
                agent = next(a for a in evolving_agents if a.agent_id == agent_id)
                await agent.apply_performance_modification(improvement)
            
            for agent in evolving_agents:
                for _ in range(5):
                    await agent.execute_benchmark_task(0.5)
        
        # Final multi-objective scores
        final_scores = {}
        for agent in evolving_agents:
            final_scores[agent.agent_id] = await evolution_system.calculate_multi_objective_score(agent)
        
        # Verify improvement across objectives
        improvements = []
        for agent_id in baseline_scores:
            baseline = baseline_scores[agent_id]
            final = final_scores[agent_id]
            improvement = (final - baseline) / baseline if baseline > 0 else 0
            improvements.append(improvement)
        
        avg_multi_objective_improvement = statistics.mean(improvements)
        
        assert avg_multi_objective_improvement > 0.12, f"Multi-objective improvement {avg_multi_objective_improvement:.1%} below 12%"
        
        # Verify no single objective degraded significantly
        objective_degradations = await evolution_system.analyze_objective_trade_offs(
            evolving_agents, baseline_scores, final_scores
        )
        
        significant_degradations = [deg for deg in objective_degradations if deg < -0.1]  # More than 10% worse
        assert len(significant_degradations) == 0, f"Significant degradation in {len(significant_degradations)} objectives"
        
        print(f"Multi-objective improvement: {avg_multi_objective_improvement:.1%}")


class TestEvolutionStabilityAndSafety:
    """Test evolution stability and safety mechanisms"""
    
    @pytest.fixture
    async def safe_evolution_system(self):
        """Create evolution system with enhanced safety"""
        evolution_system = PerformanceDrivenEvolution(
            improvement_target=0.15,
            safety_constraints_enabled=True,
            rollback_on_degradation=True,
            max_modification_risk=0.3
        )
        await evolution_system.initialize()
        return evolution_system
        
    @pytest.mark.asyncio
    async def test_evolution_stability(self, safe_evolution_system):
        """Test that evolution remains stable over extended periods"""
        agents = [PerformanceTestAgent(f"stable_agent_{i}") for i in range(3)]
        
        for agent in agents:
            await safe_evolution_system.register_agent(agent)
        
        # Run extended evolution
        performance_over_time = []
        
        for cycle in range(15):  # Extended evolution
            # Measure current performance
            cycle_performance = []
            for agent in agents:
                for _ in range(5):
                    await agent.execute_benchmark_task(0.5)
                improvement = agent.calculate_performance_improvement()
                cycle_performance.append(improvement)
            
            avg_cycle_performance = statistics.mean(cycle_performance)
            performance_over_time.append(avg_cycle_performance)
            
            # Apply evolution
            improvements = await safe_evolution_system.analyze_and_generate_improvements(agents)
            
            for agent_id, improvement in improvements.items():
                agent = next(a for a in agents if a.agent_id == agent_id)
                await agent.apply_performance_modification(improvement)
        
        # Analyze stability
        performance_std = statistics.stdev(performance_over_time[-5:])  # Last 5 cycles
        performance_trend = np.polyfit(range(len(performance_over_time)), performance_over_time, 1)[0]
        
        # Should be stable (low standard deviation) and non-negative trend
        assert performance_std < 0.05, f"Performance instability: std={performance_std:.3f}"
        assert performance_trend > -0.01, f"Performance degrading: trend={performance_trend:.3f}"
        
        final_performance = statistics.mean(performance_over_time[-3:])
        assert final_performance > 0.10, f"Final performance {final_performance:.1%} too low"
        
        print(f"Evolution stability: std={performance_std:.3f}, trend={performance_trend:.3f}")
        print(f"Final performance: {final_performance:.1%}")
        
    @pytest.mark.asyncio
    async def test_rollback_on_performance_degradation(self, safe_evolution_system):
        """Test automatic rollback when modifications cause degradation"""
        agent = PerformanceTestAgent("rollback_test_agent")
        await safe_evolution_system.register_agent(agent)
        
        # Establish good baseline
        for _ in range(10):
            await agent.execute_benchmark_task(0.5)
        baseline_performance = agent.calculate_performance_improvement()
        
        # Apply beneficial modification
        good_modification = {
            'type': 'response_time_optimization',
            'improvement_factor': 1.15
        }
        await agent.apply_performance_modification(good_modification)
        
        # Measure improved performance
        for _ in range(5):
            await agent.execute_benchmark_task(0.5)
        improved_performance = agent.calculate_performance_improvement()
        
        assert improved_performance > baseline_performance
        
        # Simulate harmful modification (intentionally bad)
        harmful_modification = {
            'type': 'response_time_optimization', 
            'improvement_factor': 0.7  # Makes things worse
        }
        
        # System should detect and rollback
        rollback_result = await safe_evolution_system.apply_modification_with_monitoring(
            agent, harmful_modification
        )
        
        assert rollback_result['rollback_performed'] == True
        assert rollback_result['reason'] == 'performance_degradation'
        
        # Performance should be restored
        for _ in range(5):
            await agent.execute_benchmark_task(0.5)
        restored_performance = agent.calculate_performance_improvement()
        
        # Should be back to improved level, not degraded level
        assert abs(restored_performance - improved_performance) < 0.05
        
        print(f"Rollback test: baseline={baseline_performance:.1%}, improved={improved_performance:.1%}, restored={restored_performance:.1%}")


if __name__ == "__main__":
    # Run self-modification tests
    pytest.main([__file__, "-v", "--tb=short"])