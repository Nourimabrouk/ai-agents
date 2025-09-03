"""
Performance Benchmarking Suite
Comprehensive performance testing and optimization across all systems
"""

import asyncio
import time
import statistics
import psutil
import gc
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import numpy as np

from templates.base_agent import BaseAgent
from core.orchestration.orchestrator import AgentOrchestrator, Task
from agents.temporal.temporal_agent import TemporalAgent
from agents.learning.meta_learning_agent import MetaLearningAgent
from utils.memory.vector_memory import VectorMemoryStore
from utils.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a performance benchmark"""
    test_name: str
    component: str
    duration: float
    throughput: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    success_rate: float
    error_count: int
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark tests"""
    test_name: str
    iterations: int = 100
    concurrency_level: int = 1
    timeout_seconds: int = 30
    warm_up_iterations: int = 10
    measure_memory: bool = True
    measure_cpu: bool = True
    target_throughput: Optional[float] = None
    target_memory_mb: Optional[float] = None


class PerformanceBenchmarks:
    """
    Comprehensive performance benchmarking suite
    Tests all major components for performance, scalability, and efficiency
    """
    
    def __init__(self, results_file: str = "benchmark_results.json"):
        self.results_file = results_file
        self.benchmark_results: List[BenchmarkResult] = []
        self.system_baseline: Dict[str, float] = {}
        
        # Component instances for testing
        self.orchestrator: Optional[AgentOrchestrator] = None
        self.temporal_agent: Optional[TemporalAgent] = None
        self.learning_agent: Optional[MetaLearningAgent] = None
        self.memory_store: Optional[VectorMemoryStore] = None
        
        logger.info("Initialized performance benchmarking suite")
    
    async def setup_benchmark_environment(self):
        """Set up the benchmark environment"""
        logger.info("Setting up benchmark environment...")
        
        # Initialize components
        self.orchestrator = AgentOrchestrator("benchmark_orchestrator")
        self.temporal_agent = TemporalAgent("benchmark_temporal")
        self.learning_agent = MetaLearningAgent("benchmark_learning")
        self.memory_store = VectorMemoryStore("benchmark_memory", max_memories=10000)
        
        # Register agents with orchestrator
        self.orchestrator.register_agent(self.temporal_agent)
        self.orchestrator.register_agent(self.learning_agent)
        
        # Create test agents
        for i in range(5):
            test_agent = self._create_test_agent(f"test_agent_{i}")
            self.orchestrator.register_agent(test_agent)
        
        # Record system baseline
        await self._record_system_baseline()
        
        logger.info("Benchmark environment setup complete")
    
    def _create_test_agent(self, name: str) -> BaseAgent:
        """Create a test agent for benchmarking"""
        
        class BenchmarkTestAgent(BaseAgent):
            async def execute(self, task: Any, action) -> Any:
                # Simulate varying amounts of work
                work_intensity = hash(str(task)) % 10 / 10.0
                await asyncio.sleep(work_intensity * 0.1)  # 0-100ms work
                
                return {
                    "result": f"Processed by {self.name}",
                    "work_intensity": work_intensity,
                    "task_hash": hash(str(task)),
                    "timestamp": datetime.now().isoformat()
                }
        
        return BenchmarkTestAgent(name)
    
    async def _record_system_baseline(self):
        """Record system baseline metrics"""
        process = psutil.Process()
        
        self.system_baseline = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "open_files": len(process.open_files()),
            "threads": process.num_threads()
        }
        
        logger.info(f"System baseline recorded: {self.system_baseline}")
    
    async def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """Run all performance benchmarks"""
        logger.info("Starting comprehensive performance benchmarks...")
        
        benchmark_configs = [
            # Core orchestration benchmarks
            BenchmarkConfig("orchestrator_basic_delegation", iterations=100, concurrency_level=1),
            BenchmarkConfig("orchestrator_parallel_execution", iterations=50, concurrency_level=5),
            BenchmarkConfig("orchestrator_hierarchical_delegation", iterations=30, concurrency_level=1),
            BenchmarkConfig("orchestrator_swarm_intelligence", iterations=20, concurrency_level=1),
            
            # Temporal reasoning benchmarks
            BenchmarkConfig("temporal_event_processing", iterations=200, concurrency_level=1),
            BenchmarkConfig("temporal_prediction_generation", iterations=100, concurrency_level=3),
            BenchmarkConfig("temporal_multi_horizon_optimization", iterations=50, concurrency_level=2),
            BenchmarkConfig("temporal_pattern_analysis", iterations=80, concurrency_level=1),
            
            # Learning system benchmarks
            BenchmarkConfig("learning_strategy_selection", iterations=150, concurrency_level=1),
            BenchmarkConfig("learning_experience_recording", iterations=300, concurrency_level=2),
            BenchmarkConfig("learning_pattern_extraction", iterations=100, concurrency_level=1),
            BenchmarkConfig("learning_knowledge_transfer", iterations=60, concurrency_level=1),
            
            # Memory system benchmarks
            BenchmarkConfig("memory_storage_operations", iterations=500, concurrency_level=5),
            BenchmarkConfig("memory_similarity_search", iterations=200, concurrency_level=10),
            BenchmarkConfig("memory_bulk_operations", iterations=50, concurrency_level=3),
            BenchmarkConfig("memory_concurrent_access", iterations=100, concurrency_level=20),
            
            # Integration benchmarks
            BenchmarkConfig("end_to_end_task_processing", iterations=100, concurrency_level=3),
            BenchmarkConfig("multi_component_workflow", iterations=50, concurrency_level=2),
            BenchmarkConfig("system_stress_test", iterations=200, concurrency_level=10)
        ]
        
        # Run each benchmark
        for config in benchmark_configs:
            try:
                logger.info(f"Running benchmark: {config.test_name}")
                result = await self._run_benchmark(config)
                self.benchmark_results.append(result)
                
                # Brief pause between benchmarks
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Benchmark {config.test_name} failed: {e}")
                
                # Record failure result
                failure_result = BenchmarkResult(
                    test_name=config.test_name,
                    component="system",
                    duration=float('inf'),
                    throughput=0.0,
                    memory_usage={},
                    cpu_usage=0.0,
                    success_rate=0.0,
                    error_count=1,
                    additional_metrics={"error": str(e)}
                )
                self.benchmark_results.append(failure_result)
        
        # Save results
        await self._save_results()
        
        logger.info(f"Completed {len(benchmark_configs)} benchmarks")
        return self.benchmark_results
    
    async def _run_benchmark(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run a specific benchmark"""
        
        # Warm-up phase
        if config.warm_up_iterations > 0:
            await self._run_warm_up(config)
        
        # Clear any previous state
        gc.collect()
        
        # Record initial system state
        initial_memory = self._get_memory_usage()
        initial_cpu = psutil.cpu_percent()
        
        # Run benchmark
        start_time = time.time()
        errors = 0
        successful_operations = 0
        
        if config.concurrency_level == 1:
            # Sequential execution
            for i in range(config.iterations):
                try:
                    await asyncio.wait_for(
                        self._execute_benchmark_operation(config, i),
                        timeout=config.timeout_seconds
                    )
                    successful_operations += 1
                except Exception as e:
                    errors += 1
                    logger.debug(f"Benchmark operation failed: {e}")
        
        else:
            # Concurrent execution
            semaphore = asyncio.Semaphore(config.concurrency_level)
            
            async def limited_operation(op_id):
                async with semaphore:
                    try:
                        await asyncio.wait_for(
                            self._execute_benchmark_operation(config, op_id),
                            timeout=config.timeout_seconds
                        )
                        return True
                    except Exception as e:
                        logger.debug(f"Concurrent operation {op_id} failed: {e}")
                        return False
            
            # Execute all operations concurrently
            tasks = [limited_operation(i) for i in range(config.iterations)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful_operations = sum(1 for r in results if r is True)
            errors = config.iterations - successful_operations
        
        # Calculate metrics
        end_time = time.time()
        duration = end_time - start_time
        throughput = successful_operations / duration if duration > 0 else 0.0
        success_rate = successful_operations / config.iterations if config.iterations > 0 else 0.0
        
        # Resource usage
        final_memory = self._get_memory_usage()
        final_cpu = psutil.cpu_percent()
        
        memory_usage = {
            "initial_mb": initial_memory,
            "final_mb": final_memory,
            "delta_mb": final_memory - initial_memory
        }
        
        # Additional metrics
        additional_metrics = {
            "iterations": config.iterations,
            "concurrency_level": config.concurrency_level,
            "successful_operations": successful_operations,
            "avg_operation_time": duration / config.iterations if config.iterations > 0 else 0
        }
        
        # Component-specific metrics
        if hasattr(self, f"_get_{config.test_name.split('_')[0]}_metrics"):
            component_metrics_func = getattr(self, f"_get_{config.test_name.split('_')[0]}_metrics")
            additional_metrics.update(await component_metrics_func())
        
        return BenchmarkResult(
            test_name=config.test_name,
            component=config.test_name.split('_')[0],
            duration=duration,
            throughput=throughput,
            memory_usage=memory_usage,
            cpu_usage=final_cpu,
            success_rate=success_rate,
            error_count=errors,
            additional_metrics=additional_metrics
        )
    
    async def _run_warm_up(self, config: BenchmarkConfig):
        """Run warm-up operations"""
        for i in range(config.warm_up_iterations):
            try:
                await self._execute_benchmark_operation(config, f"warmup_{i}")
            except Exception:
                pass  # Ignore warm-up failures
    
    async def _execute_benchmark_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Execute a single benchmark operation"""
        
        if config.test_name.startswith("orchestrator_"):
            return await self._benchmark_orchestrator_operation(config, operation_id)
        
        elif config.test_name.startswith("temporal_"):
            return await self._benchmark_temporal_operation(config, operation_id)
        
        elif config.test_name.startswith("learning_"):
            return await self._benchmark_learning_operation(config, operation_id)
        
        elif config.test_name.startswith("memory_"):
            return await self._benchmark_memory_operation(config, operation_id)
        
        elif config.test_name.startswith("end_to_end"):
            return await self._benchmark_end_to_end_operation(config, operation_id)
        
        elif config.test_name.startswith("multi_component"):
            return await self._benchmark_multi_component_operation(config, operation_id)
        
        elif config.test_name.startswith("system_stress"):
            return await self._benchmark_stress_operation(config, operation_id)
        
        else:
            raise ValueError(f"Unknown benchmark type: {config.test_name}")
    
    # Orchestrator benchmarks
    async def _benchmark_orchestrator_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark orchestrator operations"""
        
        if "basic_delegation" in config.test_name:
            task = Task(
                id=f"bench_{operation_id}",
                description=f"Basic benchmark task {operation_id}",
                requirements={"benchmark": True}
            )
            return await self.orchestrator.delegate_task(task)
        
        elif "parallel_execution" in config.test_name:
            agents = list(self.orchestrator.agents.values())[:3]
            task = Task(
                id=f"parallel_{operation_id}",
                description="Parallel benchmark task",
                requirements={"parallel": True}
            )
            return await self.orchestrator.parallel_execution(agents, task)
        
        elif "hierarchical_delegation" in config.test_name:
            task = Task(
                id=f"hierarchical_{operation_id}",
                description="Complex hierarchical task requiring breakdown",
                requirements={"complexity": "high", "hierarchical": True}
            )
            return await self.orchestrator.hierarchical_delegation(task)
        
        elif "swarm_intelligence" in config.test_name:
            return await self.orchestrator.swarm_intelligence(
                f"Optimize benchmark objective {operation_id}",
                swarm_size=5
            )
        
        else:
            raise ValueError(f"Unknown orchestrator benchmark: {config.test_name}")
    
    # Temporal benchmarks
    async def _benchmark_temporal_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark temporal reasoning operations"""
        
        if "event_processing" in config.test_name:
            return await self.temporal_agent.process_task(
                f"add_event event_type=benchmark_{operation_id} horizon=second"
            )
        
        elif "prediction_generation" in config.test_name:
            return await self.temporal_agent.process_task(
                "predict future events for horizon=minute confidence=0.5"
            )
        
        elif "multi_horizon_optimization" in config.test_name:
            return await self.temporal_agent.process_task(
                f"optimize objective=benchmark_{operation_id} across all horizons"
            )
        
        elif "pattern_analysis" in config.test_name:
            return await self.temporal_agent.process_task("analyze_patterns in temporal data")
        
        else:
            raise ValueError(f"Unknown temporal benchmark: {config.test_name}")
    
    # Learning benchmarks
    async def _benchmark_learning_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark learning system operations"""
        
        if "strategy_selection" in config.test_name:
            return await self.learning_agent.process_task(
                f"learn from benchmark task {operation_id}"
            )
        
        elif "experience_recording" in config.test_name:
            return await self.learning_agent.process_task(
                f"record experience from operation {operation_id}"
            )
        
        elif "pattern_extraction" in config.test_name:
            return await self.learning_agent.process_task("extract patterns from learning history")
        
        elif "knowledge_transfer" in config.test_name:
            return await self.learning_agent.process_task(
                f"transfer knowledge from domain=benchmark target=general"
            )
        
        else:
            raise ValueError(f"Unknown learning benchmark: {config.test_name}")
    
    # Memory benchmarks
    async def _benchmark_memory_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark memory system operations"""
        
        if "storage_operations" in config.test_name:
            return await self.memory_store.store_memory(
                f"Benchmark memory content {operation_id}",
                metadata={"benchmark": True, "operation_id": operation_id},
                tags=["benchmark", f"op_{operation_id}"]
            )
        
        elif "similarity_search" in config.test_name:
            results = await self.memory_store.search_similar(
                f"benchmark search query {operation_id}",
                limit=5
            )
            return len(results)
        
        elif "bulk_operations" in config.test_name:
            # Store multiple memories in bulk
            tasks = []
            for i in range(10):
                tasks.append(self.memory_store.store_memory(
                    f"Bulk memory {operation_id}_{i}",
                    metadata={"bulk": True, "batch": operation_id}
                ))
            return await asyncio.gather(*tasks)
        
        elif "concurrent_access" in config.test_name:
            # Concurrent read/write operations
            search_task = self.memory_store.search_similar("concurrent test", limit=3)
            store_task = self.memory_store.store_memory(
                f"Concurrent memory {operation_id}",
                metadata={"concurrent": True}
            )
            return await asyncio.gather(search_task, store_task)
        
        else:
            raise ValueError(f"Unknown memory benchmark: {config.test_name}")
    
    # Integration benchmarks
    async def _benchmark_end_to_end_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark end-to-end task processing"""
        
        # Create a task that exercises multiple components
        task = Task(
            id=f"e2e_{operation_id}",
            description=f"End-to-end benchmark task {operation_id} requiring temporal analysis and learning",
            requirements={
                "temporal_analysis": True,
                "learning_required": True,
                "memory_storage": True
            }
        )
        
        # Process through orchestrator
        orchestrator_result = await self.orchestrator.delegate_task(task)
        
        # Store result in memory
        memory_id = await self.memory_store.store_memory(
            f"E2E result: {str(orchestrator_result)[:100]}",
            metadata={"e2e_benchmark": True, "operation_id": operation_id}
        )
        
        # Learn from the experience
        learning_result = await self.learning_agent.process_task(
            f"learn from end-to-end task {operation_id}"
        )
        
        return {
            "orchestrator_result": orchestrator_result,
            "memory_id": memory_id,
            "learning_result": learning_result
        }
    
    async def _benchmark_multi_component_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark multi-component workflows"""
        
        # Step 1: Temporal analysis
        temporal_result = await self.temporal_agent.process_task(
            f"analyze temporal patterns for workflow {operation_id}"
        )
        
        # Step 2: Memory search for similar workflows
        similar_workflows = await self.memory_store.search_similar(
            "workflow execution patterns",
            limit=3
        )
        
        # Step 3: Learning-based optimization
        learning_optimization = await self.learning_agent.process_task(
            "optimize workflow based on historical patterns"
        )
        
        # Step 4: Orchestrated execution
        workflow_task = Task(
            id=f"workflow_{operation_id}",
            description="Multi-component workflow task",
            requirements={
                "temporal_insights": temporal_result,
                "similar_patterns": len(similar_workflows),
                "optimization": learning_optimization
            }
        )
        
        final_result = await self.orchestrator.delegate_task(workflow_task)
        
        return {
            "temporal_analysis": temporal_result,
            "similar_workflows": len(similar_workflows),
            "learning_optimization": learning_optimization,
            "final_execution": final_result
        }
    
    async def _benchmark_stress_operation(self, config: BenchmarkConfig, operation_id: Any) -> Any:
        """Benchmark system under stress"""
        
        # Create multiple concurrent operations
        stress_tasks = []
        
        # Orchestrator stress
        for i in range(3):
            task = Task(
                id=f"stress_{operation_id}_{i}",
                description=f"Stress test task {i}",
                requirements={"stress_test": True}
            )
            stress_tasks.append(self.orchestrator.delegate_task(task))
        
        # Memory stress
        for i in range(5):
            stress_tasks.append(self.memory_store.store_memory(
                f"Stress memory {operation_id}_{i}",
                metadata={"stress_test": True}
            ))
        
        # Temporal stress
        stress_tasks.append(self.temporal_agent.process_task(
            f"handle stress event {operation_id}"
        ))
        
        # Learning stress
        stress_tasks.append(self.learning_agent.process_task(
            f"learn under stress {operation_id}"
        ))
        
        # Execute all stress tasks concurrently
        return await asyncio.gather(*stress_tasks, return_exceptions=True)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    async def _get_orchestrator_metrics(self) -> Dict[str, Any]:
        """Get orchestrator-specific metrics"""
        if not self.orchestrator:
            return {}
        
        return {
            "total_agents": len(self.orchestrator.agents),
            "completed_tasks": self.orchestrator.total_tasks_completed,
            "messages_sent": self.orchestrator.total_messages_sent
        }
    
    async def _get_temporal_metrics(self) -> Dict[str, Any]:
        """Get temporal system metrics"""
        if not self.temporal_agent:
            return {}
        
        state = self.temporal_agent.temporal_engine.get_temporal_state()
        return {
            "total_events": state.get("total_events", 0),
            "active_patterns": state.get("active_patterns", 0),
            "cache_size": state.get("cache_size", 0)
        }
    
    async def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning system metrics"""
        if not self.learning_agent:
            return {}
        
        stats = self.learning_agent.get_learning_statistics()
        return {
            "total_strategies": stats.get("total_strategies", 0),
            "total_experiences": stats.get("total_experiences", 0),
            "recent_success_rate": stats.get("recent_success_rate", 0)
        }
    
    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory system metrics"""
        if not self.memory_store:
            return {}
        
        stats = self.memory_store.get_statistics()
        return {
            "total_memories": stats.get("total_memories", 0),
            "total_searches": stats.get("total_searches", 0),
            "cache_hit_rate": stats.get("cache_hit_rate", 0)
        }
    
    async def _save_results(self):
        """Save benchmark results to file"""
        try:
            results_data = {
                "benchmark_timestamp": datetime.now().isoformat(),
                "system_baseline": self.system_baseline,
                "results": [
                    {
                        "test_name": r.test_name,
                        "component": r.component,
                        "duration": r.duration,
                        "throughput": r.throughput,
                        "memory_usage": r.memory_usage,
                        "cpu_usage": r.cpu_usage,
                        "success_rate": r.success_rate,
                        "error_count": r.error_count,
                        "additional_metrics": r.additional_metrics,
                        "timestamp": r.timestamp.isoformat()
                    }
                    for r in self.benchmark_results
                ]
            }
            
            with open(self.results_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            logger.info(f"Benchmark results saved to {self.results_file}")
            
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}
        
        # Aggregate results by component
        by_component = {}
        for result in self.benchmark_results:
            component = result.component
            if component not in by_component:
                by_component[component] = []
            by_component[component].append(result)
        
        # Generate statistics for each component
        component_stats = {}
        for component, results in by_component.items():
            throughputs = [r.throughput for r in results if r.throughput > 0]
            success_rates = [r.success_rate for r in results]
            durations = [r.duration for r in results if r.duration < float('inf')]
            
            component_stats[component] = {
                "total_tests": len(results),
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
                "avg_duration": statistics.mean(durations) if durations else 0,
                "total_errors": sum(r.error_count for r in results)
            }
        
        # Overall system performance
        all_throughputs = [r.throughput for r in self.benchmark_results if r.throughput > 0]
        all_success_rates = [r.success_rate for r in self.benchmark_results]
        
        overall_stats = {
            "total_benchmarks": len(self.benchmark_results),
            "overall_avg_throughput": statistics.mean(all_throughputs) if all_throughputs else 0,
            "overall_success_rate": statistics.mean(all_success_rates) if all_success_rates else 0,
            "total_errors": sum(r.error_count for r in self.benchmark_results),
            "benchmark_duration": max(r.timestamp for r in self.benchmark_results) - min(r.timestamp for r in self.benchmark_results) if self.benchmark_results else timedelta(0)
        }
        
        # Performance bottlenecks
        bottlenecks = []
        for result in self.benchmark_results:
            if result.success_rate < 0.9:
                bottlenecks.append({
                    "test": result.test_name,
                    "issue": "low_success_rate",
                    "value": result.success_rate
                })
            
            if result.throughput < 10:  # Arbitrary threshold
                bottlenecks.append({
                    "test": result.test_name,
                    "issue": "low_throughput",
                    "value": result.throughput
                })
        
        # Recommendations
        recommendations = self._generate_performance_recommendations(component_stats, bottlenecks)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_baseline": self.system_baseline,
            "overall_performance": overall_stats,
            "component_performance": component_stats,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations
        }
    
    def _generate_performance_recommendations(self, 
                                           component_stats: Dict[str, Any],
                                           bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        # Component-specific recommendations
        for component, stats in component_stats.items():
            if stats["avg_success_rate"] < 0.9:
                recommendations.append(f"Improve reliability in {component} component (success rate: {stats['avg_success_rate']:.2%})")
            
            if stats["total_errors"] > 10:
                recommendations.append(f"Investigate error patterns in {component} component ({stats['total_errors']} errors)")
            
            if stats["avg_throughput"] < 50:  # Arbitrary threshold
                recommendations.append(f"Optimize {component} throughput (current: {stats['avg_throughput']:.1f} ops/sec)")
        
        # Bottleneck-specific recommendations
        low_throughput_tests = [b for b in bottlenecks if b["issue"] == "low_throughput"]
        if len(low_throughput_tests) > 3:
            recommendations.append("Multiple throughput bottlenecks detected - consider system-wide optimization")
        
        low_success_tests = [b for b in bottlenecks if b["issue"] == "low_success_rate"]
        if len(low_success_tests) > 2:
            recommendations.append("Multiple reliability issues detected - review error handling strategies")
        
        # General recommendations
        if not recommendations:
            recommendations.append("System performing well - consider stress testing with higher loads")
        
        return recommendations