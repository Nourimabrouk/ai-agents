"""
Phase 7 Performance Testing Suite
Validates performance targets: 1000+ concurrent agents, sub-second response times,
10,000+ token working memory, and autonomous improvement metrics
"""

import asyncio
import pytest
import time
import statistics
import psutil
import gc
import resource
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass
import json

# Import Phase 7 components
from core.autonomous.orchestrator import AutonomousMetaOrchestrator, AutonomyLevel
from core.autonomous.self_modification import SelfModifyingAgent, PerformanceDrivenEvolution
from core.reasoning.working_memory import DistributedWorkingMemory, WorkingMemoryManager
from core.reasoning.performance_optimizer import AdaptivePerformanceOptimizer
from templates.base_agent import BaseAgent
from . import PHASE7_TEST_CONFIG


@dataclass
class PerformanceMetrics:
    """Performance measurement data"""
    response_time: float
    throughput: float
    memory_usage: float
    cpu_usage: float
    error_rate: float
    concurrent_operations: int
    timestamp: datetime


class PerformanceTestAgent(BaseAgent):
    """High-performance test agent for load testing"""
    
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        self.execution_count = 0
        self.total_execution_time = 0.0
        self.error_count = 0
        
    async def execute_simple_task(self) -> Dict[str, Any]:
        """Execute simple task for response time testing"""
        start_time = time.perf_counter()
        
        # Simulate simple processing
        await asyncio.sleep(0.001)  # 1ms processing time
        result = {"status": "completed", "agent_id": self.agent_id}
        
        execution_time = time.perf_counter() - start_time
        self.execution_count += 1
        self.total_execution_time += execution_time
        
        return result
        
    async def execute_complex_task(self, complexity: float = 0.5) -> Dict[str, Any]:
        """Execute complex task for throughput testing"""
        start_time = time.perf_counter()
        
        # Simulate complex processing based on complexity
        processing_time = complexity * 0.1  # Scale processing time
        await asyncio.sleep(processing_time)
        
        # Simulate some CPU work
        data = [i * i for i in range(int(complexity * 1000))]
        result_sum = sum(data)
        
        execution_time = time.perf_counter() - start_time
        self.execution_count += 1
        self.total_execution_time += execution_time
        
        return {
            "status": "completed",
            "agent_id": self.agent_id,
            "complexity": complexity,
            "processing_time": execution_time,
            "result": result_sum
        }
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get agent performance statistics"""
        if self.execution_count == 0:
            return {"avg_response_time": 0.0, "total_executions": 0}
            
        return {
            "avg_response_time": self.total_execution_time / self.execution_count,
            "total_executions": self.execution_count,
            "total_time": self.total_execution_time,
            "error_rate": self.error_count / self.execution_count
        }


class TestConcurrentAgentPerformance:
    """Test performance with 1000+ concurrent agents"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create high-performance orchestrator"""
        orchestrator = AutonomousMetaOrchestrator(
            autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS,
            max_concurrent_agents=1500,
            performance_mode=True
        )
        await orchestrator.initialize()
        return orchestrator
        
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_1000_concurrent_agents(self, orchestrator):
        """Test system handles 1000+ concurrent agents"""
        target_agents = PHASE7_TEST_CONFIG["performance_targets"]["concurrent_agents"]
        
        # Create large number of agents
        agents = [
            PerformanceTestAgent(f"perf_agent_{i}")
            for i in range(target_agents)
        ]
        
        # Register all agents
        start_time = time.perf_counter()
        registration_tasks = [orchestrator.register_agent(agent) for agent in agents]
        await asyncio.gather(*registration_tasks)
        registration_time = time.perf_counter() - start_time
        
        # Verify all agents registered
        registered_agents = await orchestrator.get_registered_agents()
        assert len(registered_agents) >= target_agents
        
        print(f"Registered {len(registered_agents)} agents in {registration_time:.2f}s")
        
        # Test concurrent simple task execution
        start_time = time.perf_counter()
        execution_tasks = [agent.execute_simple_task() for agent in agents]
        results = await asyncio.gather(*execution_tasks, return_exceptions=True)
        execution_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]
        
        success_rate = len(successful_results) / len(results)
        throughput = len(successful_results) / execution_time
        
        # Performance assertions
        assert success_rate >= 0.95, f"Success rate {success_rate} below 95%"
        assert throughput >= 500, f"Throughput {throughput} ops/sec below 500"
        
        print(f"Concurrent execution: {len(successful_results)} tasks in {execution_time:.2f}s")
        print(f"Throughput: {throughput:.1f} ops/sec")
        print(f"Success rate: {success_rate:.1%}")
        
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_sustained_load_performance(self, orchestrator):
        """Test sustained performance under continuous load"""
        num_agents = 100
        duration_seconds = 60
        
        # Create agents
        agents = [PerformanceTestAgent(f"sustained_{i}") for i in range(num_agents)]
        for agent in agents:
            await orchestrator.register_agent(agent)
        
        # Sustained load test
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        completed_tasks = 0
        performance_samples = []
        
        while time.perf_counter() < end_time:
            sample_start = time.perf_counter()
            
            # Execute batch of tasks
            batch_tasks = [
                agents[i % len(agents)].execute_simple_task() 
                for i in range(50)
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            successful_batch = [r for r in batch_results if not isinstance(r, Exception)]
            
            sample_time = time.perf_counter() - sample_start
            batch_throughput = len(successful_batch) / sample_time
            
            performance_samples.append({
                "throughput": batch_throughput,
                "timestamp": time.perf_counter() - start_time,
                "memory_mb": psutil.Process().memory_info().rss / 1024 / 1024
            })
            
            completed_tasks += len(successful_batch)
            
            # Brief pause to prevent overwhelming
            await asyncio.sleep(0.1)
        
        total_time = time.perf_counter() - start_time
        avg_throughput = completed_tasks / total_time
        
        # Analyze performance stability
        throughput_values = [sample["throughput"] for sample in performance_samples]
        throughput_std = statistics.stdev(throughput_values)
        throughput_cv = throughput_std / statistics.mean(throughput_values)
        
        # Performance assertions
        assert avg_throughput >= 100, f"Average throughput {avg_throughput} below 100 ops/sec"
        assert throughput_cv < 0.3, f"Throughput variability {throughput_cv} too high"
        
        print(f"Sustained load test: {completed_tasks} tasks in {total_time:.1f}s")
        print(f"Average throughput: {avg_throughput:.1f} ops/sec")
        print(f"Throughput stability (CV): {throughput_cv:.3f}")
        
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_scalability(self, orchestrator):
        """Test memory usage scales reasonably with agent count"""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Test different agent counts
        agent_counts = [100, 500, 1000]
        memory_measurements = []
        
        for agent_count in agent_counts:
            # Clean up previous agents
            gc.collect()
            
            # Create new agents
            agents = [
                PerformanceTestAgent(f"memory_test_{i}") 
                for i in range(agent_count)
            ]
            
            for agent in agents:
                await orchestrator.register_agent(agent)
            
            # Execute some tasks to stabilize memory usage
            tasks = [agent.execute_simple_task() for agent in agents[:50]]
            await asyncio.gather(*tasks)
            
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_per_agent = (current_memory - initial_memory) / agent_count
            
            memory_measurements.append({
                "agent_count": agent_count,
                "total_memory_mb": current_memory,
                "memory_per_agent_kb": memory_per_agent * 1024
            })
            
            print(f"Agents: {agent_count}, Memory: {current_memory:.1f}MB, Per agent: {memory_per_agent*1024:.1f}KB")
        
        # Memory should scale sub-linearly (efficiency gains)
        memory_efficiency = (
            memory_measurements[-1]["memory_per_agent_kb"] / 
            memory_measurements[0]["memory_per_agent_kb"]
        )
        
        assert memory_efficiency < 1.5, f"Memory efficiency degraded too much: {memory_efficiency}"
        
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_response_time_distribution(self, orchestrator):
        """Test response time distribution meets targets"""
        target_response_time = PHASE7_TEST_CONFIG["performance_targets"]["response_time_simple"]
        
        # Create test agents
        agents = [PerformanceTestAgent(f"response_{i}") for i in range(100)]
        for agent in agents:
            await orchestrator.register_agent(agent)
        
        # Collect response time samples
        response_times = []
        sample_count = 1000
        
        for _ in range(sample_count):
            agent = agents[_ % len(agents)]
            
            start_time = time.perf_counter()
            result = await agent.execute_simple_task()
            response_time = time.perf_counter() - start_time
            
            response_times.append(response_time)
        
        # Analyze response time distribution
        avg_response_time = statistics.mean(response_times)
        p50_response_time = statistics.median(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]
        p99_response_time = sorted(response_times)[int(0.99 * len(response_times))]
        
        # Performance assertions
        assert avg_response_time < target_response_time, f"Average response time {avg_response_time:.3f}s exceeds target {target_response_time}s"
        assert p95_response_time < target_response_time * 2, f"95th percentile {p95_response_time:.3f}s too high"
        assert p99_response_time < target_response_time * 5, f"99th percentile {p99_response_time:.3f}s too high"
        
        print(f"Response time analysis (n={sample_count}):")
        print(f"  Average: {avg_response_time*1000:.1f}ms")
        print(f"  Median: {p50_response_time*1000:.1f}ms")
        print(f"  95th percentile: {p95_response_time*1000:.1f}ms")
        print(f"  99th percentile: {p99_response_time*1000:.1f}ms")


class TestWorkingMemoryPerformance:
    """Test working memory performance with 10,000+ tokens"""
    
    @pytest.fixture
    async def working_memory(self):
        """Create high-capacity working memory system"""
        memory_manager = WorkingMemoryManager(
            max_tokens=15000,  # Above target
            compression_enabled=True,
            distributed_mode=True
        )
        await memory_manager.initialize()
        return memory_manager
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_10k_token_capacity(self, working_memory):
        """Test working memory handles 10,000+ tokens efficiently"""
        target_tokens = PHASE7_TEST_CONFIG["performance_targets"]["working_memory_tokens"]
        
        # Generate large memory content
        large_content = self._generate_memory_content(target_tokens)
        
        # Test memory storage
        start_time = time.perf_counter()
        memory_id = await working_memory.store_content(large_content)
        storage_time = time.perf_counter() - start_time
        
        # Verify token count
        stored_tokens = await working_memory.get_token_count(memory_id)
        assert stored_tokens >= target_tokens, f"Stored only {stored_tokens} tokens, target {target_tokens}"
        
        # Test memory retrieval
        start_time = time.perf_counter()
        retrieved_content = await working_memory.retrieve_content(memory_id)
        retrieval_time = time.perf_counter() - start_time
        
        # Test memory search
        start_time = time.perf_counter()
        search_results = await working_memory.search_content("important concept")
        search_time = time.perf_counter() - start_time
        
        # Performance assertions
        assert storage_time < 2.0, f"Storage time {storage_time:.2f}s too slow"
        assert retrieval_time < 1.0, f"Retrieval time {retrieval_time:.2f}s too slow"
        assert search_time < 0.5, f"Search time {search_time:.2f}s too slow"
        
        print(f"Working memory performance (tokens: {stored_tokens}):")
        print(f"  Storage time: {storage_time*1000:.1f}ms")
        print(f"  Retrieval time: {retrieval_time*1000:.1f}ms")
        print(f"  Search time: {search_time*1000:.1f}ms")
        
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_coherence_under_load(self, working_memory):
        """Test memory coherence during high-frequency operations"""
        # Create multiple memory contexts
        contexts = []
        for i in range(10):
            content = self._generate_memory_content(1000)  # 1K tokens each
            context_id = await working_memory.store_content(content)
            contexts.append(context_id)
        
        # Concurrent operations test
        operations = []
        
        async def read_operation(context_id):
            return await working_memory.retrieve_content(context_id)
            
        async def write_operation(context_id, new_content):
            return await working_memory.update_content(context_id, new_content)
            
        async def search_operation(query):
            return await working_memory.search_content(query)
        
        # Generate concurrent operations
        for _ in range(100):
            context_id = contexts[_ % len(contexts)]
            
            if _ % 3 == 0:
                operations.append(read_operation(context_id))
            elif _ % 3 == 1:
                new_content = f"Updated content {_}"
                operations.append(write_operation(context_id, new_content))
            else:
                operations.append(search_operation(f"query_{_}"))
        
        # Execute all operations concurrently
        start_time = time.perf_counter()
        results = await asyncio.gather(*operations, return_exceptions=True)
        execution_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_ops = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_ops) / len(results)
        throughput = len(successful_ops) / execution_time
        
        # Coherence assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.1%} below 95%"
        assert throughput >= 50, f"Throughput {throughput:.1f} ops/sec below 50"
        
        print(f"Memory coherence test:")
        print(f"  Operations: {len(results)}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Throughput: {throughput:.1f} ops/sec")
        
    def _generate_memory_content(self, target_tokens: int) -> str:
        """Generate memory content with approximately target token count"""
        # Estimate ~4 characters per token
        target_chars = target_tokens * 4
        
        content_parts = []
        concepts = [
            "artificial intelligence", "machine learning", "autonomous systems",
            "cognitive architecture", "reasoning engine", "knowledge graph",
            "decision making", "optimization", "pattern recognition",
            "neural networks", "natural language processing", "computer vision"
        ]
        
        current_chars = 0
        while current_chars < target_chars:
            concept = concepts[current_chars % len(concepts)]
            paragraph = f"This is important content about {concept}. " * 10
            content_parts.append(paragraph)
            current_chars += len(paragraph)
        
        return " ".join(content_parts)


class TestAutonomousImprovementPerformance:
    """Test autonomous improvement capabilities (15% target)"""
    
    @pytest.fixture
    async def improvement_system(self):
        """Create autonomous improvement system"""
        orchestrator = AutonomousMetaOrchestrator(
            autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS
        )
        
        improvement_engine = PerformanceDrivenEvolution(
            target_improvement=0.15,  # 15% target
            measurement_window=timedelta(minutes=5)
        )
        
        await orchestrator.initialize()
        await improvement_engine.initialize()
        
        return orchestrator, improvement_engine
        
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_15_percent_improvement_target(self, improvement_system):
        """Test system achieves 15% autonomous performance improvement"""
        orchestrator, improvement_engine = improvement_system
        target_improvement = PHASE7_TEST_CONFIG["performance_targets"]["autonomous_improvement"]
        
        # Create baseline agents
        baseline_agents = [
            PerformanceTestAgent(f"baseline_{i}")
            for i in range(20)
        ]
        
        for agent in baseline_agents:
            await orchestrator.register_agent(agent)
        
        # Measure baseline performance
        baseline_metrics = await self._measure_system_performance(baseline_agents)
        
        print(f"Baseline performance:")
        print(f"  Avg response time: {baseline_metrics['avg_response_time']*1000:.1f}ms")
        print(f"  Throughput: {baseline_metrics['throughput']:.1f} ops/sec")
        print(f"  Success rate: {baseline_metrics['success_rate']:.1%}")
        
        # Enable autonomous improvement
        await improvement_engine.enable_autonomous_improvement()
        
        # Run improvement cycles
        improvement_cycles = 5
        for cycle in range(improvement_cycles):
            print(f"Running improvement cycle {cycle + 1}/{improvement_cycles}")
            
            # Execute tasks to generate performance data
            tasks = [agent.execute_complex_task(0.5) for agent in baseline_agents]
            await asyncio.gather(*tasks)
            
            # Trigger improvement analysis
            improvements = await improvement_engine.analyze_and_improve()
            
            if improvements:
                print(f"Applied {len(improvements)} improvements in cycle {cycle + 1}")
                
                # Apply improvements to agents
                for improvement in improvements:
                    await self._apply_improvement_to_agents(baseline_agents, improvement)
            
            # Brief pause between cycles
            await asyncio.sleep(2)
        
        # Measure final performance
        final_metrics = await self._measure_system_performance(baseline_agents)
        
        # Calculate improvement
        response_time_improvement = (
            baseline_metrics['avg_response_time'] - final_metrics['avg_response_time']
        ) / baseline_metrics['avg_response_time']
        
        throughput_improvement = (
            final_metrics['throughput'] - baseline_metrics['throughput']
        ) / baseline_metrics['throughput']
        
        overall_improvement = (response_time_improvement + throughput_improvement) / 2
        
        print(f"Final performance:")
        print(f"  Avg response time: {final_metrics['avg_response_time']*1000:.1f}ms")
        print(f"  Throughput: {final_metrics['throughput']:.1f} ops/sec")
        print(f"  Success rate: {final_metrics['success_rate']:.1%}")
        print(f"Overall improvement: {overall_improvement:.1%}")
        
        # Improvement assertion
        assert overall_improvement >= target_improvement, f"Improvement {overall_improvement:.1%} below target {target_improvement:.1%}"
        
    async def _measure_system_performance(self, agents: List[PerformanceTestAgent]) -> Dict[str, float]:
        """Measure current system performance metrics"""
        # Execute performance measurement tasks
        start_time = time.perf_counter()
        tasks = [agent.execute_complex_task(0.3) for agent in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not isinstance(r, Exception)]
        
        response_times = [r['processing_time'] for r in successful_results]
        avg_response_time = statistics.mean(response_times)
        throughput = len(successful_results) / total_time
        success_rate = len(successful_results) / len(results)
        
        return {
            'avg_response_time': avg_response_time,
            'throughput': throughput,
            'success_rate': success_rate,
            'total_executions': len(results)
        }
        
    async def _apply_improvement_to_agents(self, agents: List[PerformanceTestAgent], improvement: Dict[str, Any]):
        """Apply improvement to agents (mock implementation)"""
        # Mock improvement application
        improvement_factor = improvement.get('improvement_factor', 1.05)
        
        for agent in agents:
            # Simulate performance improvement
            if hasattr(agent, 'performance_multiplier'):
                agent.performance_multiplier *= improvement_factor
            else:
                agent.performance_multiplier = improvement_factor


class TestSystemResourceUtilization:
    """Test system resource utilization efficiency"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cpu_utilization_efficiency(self):
        """Test CPU utilization remains efficient under load"""
        orchestrator = AutonomousMetaOrchestrator()
        await orchestrator.initialize()
        
        # Create CPU-intensive agents
        agents = [PerformanceTestAgent(f"cpu_{i}") for i in range(50)]
        for agent in agents:
            await orchestrator.register_agent(agent)
        
        # Monitor CPU usage during load test
        cpu_samples = []
        
        async def cpu_monitor():
            while True:
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_samples.append(cpu_percent)
                await asyncio.sleep(1)
        
        # Start CPU monitoring
        monitor_task = asyncio.create_task(cpu_monitor())
        
        # Execute CPU-intensive workload
        workload_tasks = []
        for _ in range(5):  # 5 iterations of heavy workload
            tasks = [agent.execute_complex_task(0.8) for agent in agents]
            workload_tasks.extend(tasks)
        
        # Execute workload
        await asyncio.gather(*workload_tasks)
        
        # Stop monitoring
        monitor_task.cancel()
        
        # Analyze CPU utilization
        avg_cpu = statistics.mean(cpu_samples)
        max_cpu = max(cpu_samples)
        
        # CPU efficiency assertions
        assert avg_cpu < 80, f"Average CPU {avg_cpu:.1f}% too high"
        assert max_cpu < 95, f"Peak CPU {max_cpu:.1f}% too high"
        
        print(f"CPU utilization: avg {avg_cpu:.1f}%, peak {max_cpu:.1f}%")
        
    @pytest.mark.asyncio
    @pytest.mark.performance 
    async def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation"""
        orchestrator = AutonomousMetaOrchestrator()
        await orchestrator.initialize()
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_samples = [initial_memory]
        
        # Run extended operations
        agents = [PerformanceTestAgent(f"leak_{i}") for i in range(100)]
        
        for iteration in range(20):
            # Create and destroy agents to test cleanup
            for agent in agents:
                await orchestrator.register_agent(agent)
            
            # Execute tasks
            tasks = [agent.execute_simple_task() for agent in agents]
            await asyncio.gather(*tasks)
            
            # Cleanup agents
            for agent in agents:
                await orchestrator.unregister_agent(agent.agent_id)
            
            # Force garbage collection
            gc.collect()
            
            # Sample memory
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
            
            await asyncio.sleep(0.1)
        
        # Analyze memory trend
        final_memory = memory_samples[-1]
        memory_growth = final_memory - initial_memory
        
        # Linear regression to detect trend
        x = list(range(len(memory_samples)))
        y = memory_samples
        n = len(x)
        
        slope = (n * sum(x_i * y_i for x_i, y_i in zip(x, y)) - sum(x) * sum(y)) / (n * sum(x_i * x_i for x_i in x) - sum(x) ** 2)
        
        # Memory leak assertions
        assert memory_growth < 50, f"Memory growth {memory_growth:.1f}MB too high"
        assert slope < 1.0, f"Memory growth rate {slope:.2f}MB/iteration suggests leak"
        
        print(f"Memory analysis: growth {memory_growth:.1f}MB, rate {slope:.2f}MB/iteration")


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])