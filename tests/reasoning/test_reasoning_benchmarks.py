"""
Performance benchmarking suite for reasoning systems
Tests performance under various load conditions and optimization scenarios
"""

import pytest
import asyncio
import time
import statistics
import psutil
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import json
from pathlib import Path
import logging

# Import reasoning systems
try:
    from core.reasoning.causal_inference import CausalReasoningEngine
    from core.reasoning.working_memory import WorkingMemorySystem
    from core.reasoning.tree_of_thoughts import EnhancedTreeOfThoughts
    from core.reasoning.temporal_reasoning import TemporalReasoningEngine
    from core.reasoning.integrated_reasoning_controller import IntegratedReasoningController
    from core.reasoning.performance_optimizer import PerformanceOptimizer
    REASONING_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Reasoning system imports not available: {e}")
    REASONING_IMPORTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REASONING_IMPORTS_AVAILABLE,
    reason="Reasoning systems not available"
)

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Performance metrics for benchmarking"""
    test_name: str
    system_name: str
    throughput: float  # operations per second
    latency_mean: float  # average response time
    latency_p50: float  # median response time
    latency_p95: float  # 95th percentile response time
    latency_p99: float  # 99th percentile response time
    memory_peak_mb: float  # peak memory usage
    memory_average_mb: float  # average memory usage
    cpu_usage_avg: float  # average CPU usage
    success_rate: float  # percentage of successful operations
    error_count: int
    concurrent_operations: int
    total_operations: int
    total_duration: float
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


class LoadGenerator:
    """Generate different types of load patterns for benchmarking"""
    
    @staticmethod
    def constant_load(operations_per_second: int, duration_seconds: int) -> List[float]:
        """Generate constant load pattern"""
        interval = 1.0 / operations_per_second
        timestamps = []
        current_time = 0.0
        
        while current_time < duration_seconds:
            timestamps.append(current_time)
            current_time += interval
        
        return timestamps
    
    @staticmethod 
    def burst_load(burst_size: int, burst_interval: float, num_bursts: int) -> List[float]:
        """Generate bursty load pattern"""
        timestamps = []
        
        for burst in range(num_bursts):
            burst_start = burst * burst_interval
            
            # Generate operations within burst (compressed time)
            for op in range(burst_size):
                timestamp = burst_start + (op * 0.01)  # 10ms apart within burst
                timestamps.append(timestamp)
        
        return sorted(timestamps)
    
    @staticmethod
    def ramp_load(start_ops: int, end_ops: int, duration_seconds: int) -> List[float]:
        """Generate ramping load pattern"""
        timestamps = []
        current_time = 0.0
        
        while current_time < duration_seconds:
            # Calculate current load based on ramp
            progress = current_time / duration_seconds
            current_ops = start_ops + (end_ops - start_ops) * progress
            interval = 1.0 / max(current_ops, 0.1)
            
            timestamps.append(current_time)
            current_time += interval
        
        return timestamps
    
    @staticmethod
    def random_load(avg_ops_per_second: int, variance: float, duration_seconds: int) -> List[float]:
        """Generate random load pattern with specified average and variance"""
        np.random.seed(42)
        timestamps = []
        current_time = 0.0
        
        while current_time < duration_seconds:
            # Random interval based on exponential distribution
            rate = max(0.1, np.random.normal(avg_ops_per_second, avg_ops_per_second * variance))
            interval = np.random.exponential(1.0 / rate)
            
            current_time += interval
            if current_time < duration_seconds:
                timestamps.append(current_time)
        
        return sorted(timestamps)


class ResourceMonitor:
    """Monitor system resources during benchmarking"""
    
    def __init__(self):
        self.monitoring = False
        self.metrics = {
            "cpu_usage": [],
            "memory_usage_mb": [],
            "memory_percent": [],
            "timestamps": []
        }
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start resource monitoring"""
        self.monitoring = True
        self.metrics = {key: [] for key in self.metrics.keys()}
        
        def monitor():
            process = psutil.Process()
            
            while self.monitoring:
                try:
                    cpu = process.cpu_percent()
                    memory_info = process.memory_info()
                    memory_mb = memory_info.rss / 1024 / 1024
                    memory_percent = process.memory_percent()
                    
                    self.metrics["cpu_usage"].append(cpu)
                    self.metrics["memory_usage_mb"].append(memory_mb)
                    self.metrics["memory_percent"].append(memory_percent)
                    self.metrics["timestamps"].append(time.time())
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logger.warning(f"Resource monitoring error: {e}")
                    break
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return collected metrics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        if not self.metrics["memory_usage_mb"]:
            return {
                "peak_memory_mb": 0,
                "avg_memory_mb": 0,
                "avg_cpu_usage": 0
            }
        
        return {
            "peak_memory_mb": max(self.metrics["memory_usage_mb"]),
            "avg_memory_mb": statistics.mean(self.metrics["memory_usage_mb"]),
            "avg_cpu_usage": statistics.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "memory_timeline": self.metrics["memory_usage_mb"],
            "cpu_timeline": self.metrics["cpu_usage"]
        }


class ReasoningBenchmarkSuite:
    """Comprehensive benchmark suite for reasoning systems"""
    
    def __init__(self):
        self.results: List[BenchmarkMetrics] = []
        self.resource_monitor = ResourceMonitor()
    
    async def benchmark_causal_inference_performance(self, causal_engine: CausalReasoningEngine) -> List[BenchmarkMetrics]:
        """Benchmark causal inference engine performance"""
        benchmarks = []
        
        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000]
        
        for size in data_sizes:
            # Generate test data
            np.random.seed(42)
            n_features = min(10, max(5, size // 100))
            
            # Create causal data with known structure
            data = self._generate_causal_benchmark_data(size, n_features)
            
            # Benchmark different discovery methods
            methods = ["pc", "ges", "lingam"]
            
            for method in methods:
                logger.info(f"Benchmarking causal discovery: {method} on {size} samples")
                
                # Warm-up run
                await causal_engine.discover_causal_relationships(data.iloc[:50], discovery_method=method)
                gc.collect()
                
                # Start monitoring
                self.resource_monitor.start_monitoring()
                
                # Benchmark run
                latencies = []
                errors = 0
                start_time = time.time()
                
                # Run multiple iterations for statistical significance
                iterations = max(1, 50 // (size // 100))  # Fewer iterations for larger datasets
                
                for i in range(iterations):
                    try:
                        iter_start = time.time()
                        
                        result = await causal_engine.discover_causal_relationships(
                            data,
                            discovery_method=method
                        )
                        
                        iter_time = time.time() - iter_start
                        latencies.append(iter_time)
                        
                        # Validate result
                        if not result or len(result.get_edge_list()) == 0:
                            errors += 1
                            
                    except Exception as e:
                        logger.warning(f"Causal discovery iteration failed: {e}")
                        errors += 1
                
                total_time = time.time() - start_time
                resource_metrics = self.resource_monitor.stop_monitoring()
                
                # Calculate metrics
                if latencies:
                    metrics = BenchmarkMetrics(
                        test_name=f"causal_{method}_{size}_samples",
                        system_name="causal_inference",
                        throughput=len(latencies) / total_time,
                        latency_mean=statistics.mean(latencies),
                        latency_p50=statistics.median(latencies),
                        latency_p95=np.percentile(latencies, 95),
                        latency_p99=np.percentile(latencies, 99),
                        memory_peak_mb=resource_metrics["peak_memory_mb"],
                        memory_average_mb=resource_metrics["avg_memory_mb"],
                        cpu_usage_avg=resource_metrics["avg_cpu_usage"],
                        success_rate=(iterations - errors) / iterations,
                        error_count=errors,
                        concurrent_operations=1,
                        total_operations=iterations,
                        total_duration=total_time,
                        additional_metrics={
                            "data_size": size,
                            "n_features": n_features,
                            "discovery_method": method,
                            "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0
                        }
                    )
                    
                    benchmarks.append(metrics)
                    self.results.append(metrics)
        
        return benchmarks
    
    async def benchmark_working_memory_performance(self, working_memory: WorkingMemorySystem) -> List[BenchmarkMetrics]:
        """Benchmark working memory system performance"""
        benchmarks = []
        
        # Test memory operations at different scales
        memory_sizes = [100, 500, 1000, 5000]
        
        for size in memory_sizes:
            logger.info(f"Benchmarking working memory with {size} items")
            
            # Generate test memories
            test_memories = self._generate_memory_benchmark_data(size)
            
            # Clear previous state
            await working_memory.clear_all_memories()
            gc.collect()
            
            # Benchmark storage operations
            self.resource_monitor.start_monitoring()
            storage_latencies = []
            errors = 0
            
            start_time = time.time()
            
            for memory in test_memories:
                try:
                    iter_start = time.time()
                    
                    await working_memory.store_memory(
                        memory["content"],
                        memory_type=memory["type"],
                        importance=memory["importance"]
                    )
                    
                    storage_latencies.append(time.time() - iter_start)
                    
                except Exception as e:
                    logger.warning(f"Memory storage failed: {e}")
                    errors += 1
            
            storage_time = time.time() - start_time
            storage_resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Benchmark retrieval operations
            self.resource_monitor.start_monitoring()
            retrieval_latencies = []
            retrieval_errors = 0
            
            # Test various retrieval queries
            queries = [
                "test memory content",
                "factual information",
                "procedural knowledge",
                "episodic memory",
                "important data"
            ]
            
            retrieval_start = time.time()
            
            for query in queries * (size // 100 + 1):  # Scale queries with data size
                try:
                    iter_start = time.time()
                    
                    results = await working_memory.recall_memories(
                        query=query,
                        limit=min(20, size // 10)
                    )
                    
                    retrieval_latencies.append(time.time() - iter_start)
                    
                    if not results:
                        retrieval_errors += 1
                        
                except Exception as e:
                    logger.warning(f"Memory retrieval failed: {e}")
                    retrieval_errors += 1
            
            retrieval_time = time.time() - retrieval_start
            retrieval_resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Storage benchmark metrics
            storage_metrics = BenchmarkMetrics(
                test_name=f"memory_storage_{size}_items",
                system_name="working_memory",
                throughput=len(storage_latencies) / storage_time,
                latency_mean=statistics.mean(storage_latencies),
                latency_p50=statistics.median(storage_latencies),
                latency_p95=np.percentile(storage_latencies, 95),
                latency_p99=np.percentile(storage_latencies, 99),
                memory_peak_mb=storage_resource_metrics["peak_memory_mb"],
                memory_average_mb=storage_resource_metrics["avg_memory_mb"],
                cpu_usage_avg=storage_resource_metrics["avg_cpu_usage"],
                success_rate=(size - errors) / size,
                error_count=errors,
                concurrent_operations=1,
                total_operations=size,
                total_duration=storage_time,
                additional_metrics={
                    "operation_type": "storage",
                    "memory_items": size,
                    "latency_std": statistics.stdev(storage_latencies) if len(storage_latencies) > 1 else 0
                }
            )
            
            # Retrieval benchmark metrics
            retrieval_metrics = BenchmarkMetrics(
                test_name=f"memory_retrieval_{size}_items",
                system_name="working_memory",
                throughput=len(retrieval_latencies) / retrieval_time,
                latency_mean=statistics.mean(retrieval_latencies),
                latency_p50=statistics.median(retrieval_latencies),
                latency_p95=np.percentile(retrieval_latencies, 95),
                latency_p99=np.percentile(retrieval_latencies, 99),
                memory_peak_mb=retrieval_resource_metrics["peak_memory_mb"],
                memory_average_mb=retrieval_resource_metrics["avg_memory_mb"],
                cpu_usage_avg=retrieval_resource_metrics["avg_cpu_usage"],
                success_rate=(len(retrieval_latencies) - retrieval_errors) / len(retrieval_latencies),
                error_count=retrieval_errors,
                concurrent_operations=1,
                total_operations=len(retrieval_latencies),
                total_duration=retrieval_time,
                additional_metrics={
                    "operation_type": "retrieval",
                    "memory_items": size,
                    "queries_tested": len(queries),
                    "latency_std": statistics.stdev(retrieval_latencies) if len(retrieval_latencies) > 1 else 0
                }
            )
            
            benchmarks.extend([storage_metrics, retrieval_metrics])
            self.results.extend([storage_metrics, retrieval_metrics])
        
        return benchmarks
    
    async def benchmark_temporal_reasoning_performance(self, temporal_engine: TemporalReasoningEngine) -> List[BenchmarkMetrics]:
        """Benchmark temporal reasoning engine performance"""
        benchmarks = []
        
        # Test different temporal data sizes
        data_sizes = [1000, 5000, 10000, 20000]
        
        for size in data_sizes:
            logger.info(f"Benchmarking temporal reasoning with {size} data points")
            
            # Generate temporal test data
            temporal_data = self._generate_temporal_benchmark_data(size)
            
            # Clear previous state
            await temporal_engine.clear_temporal_state()
            gc.collect()
            
            # Benchmark data ingestion
            self.resource_monitor.start_monitoring()
            ingestion_start = time.time()
            
            try:
                await temporal_engine.add_temporal_data(temporal_data)
                ingestion_time = time.time() - ingestion_start
                ingestion_success = True
            except Exception as e:
                logger.warning(f"Temporal data ingestion failed: {e}")
                ingestion_time = time.time() - ingestion_start
                ingestion_success = False
            
            ingestion_resource_metrics = self.resource_monitor.stop_monitoring()
            
            if ingestion_success:
                # Benchmark pattern detection
                self.resource_monitor.start_monitoring()
                pattern_latencies = []
                pattern_errors = 0
                
                pattern_start = time.time()
                
                # Multiple pattern detection runs
                for i in range(5):  # Multiple runs for consistency
                    try:
                        iter_start = time.time()
                        patterns = await temporal_engine.detect_patterns()
                        pattern_latencies.append(time.time() - iter_start)
                        
                        if not patterns:
                            pattern_errors += 1
                            
                    except Exception as e:
                        logger.warning(f"Pattern detection failed: {e}")
                        pattern_errors += 1
                
                pattern_time = time.time() - pattern_start
                pattern_resource_metrics = self.resource_monitor.stop_monitoring()
                
                # Benchmark prediction generation
                self.resource_monitor.start_monitoring()
                prediction_latencies = []
                prediction_errors = 0
                
                horizons = ["minute", "hour", "day"]
                prediction_start = time.time()
                
                for horizon in horizons:
                    for i in range(3):  # Multiple predictions per horizon
                        try:
                            iter_start = time.time()
                            prediction = await temporal_engine.predict(
                                horizon=horizon,
                                confidence_level=0.95
                            )
                            prediction_latencies.append(time.time() - iter_start)
                            
                            if not prediction:
                                prediction_errors += 1
                                
                        except Exception as e:
                            logger.warning(f"Prediction failed: {e}")
                            prediction_errors += 1
                
                prediction_time = time.time() - prediction_start
                prediction_resource_metrics = self.resource_monitor.stop_monitoring()
                
                # Create benchmark metrics
                ingestion_metrics = BenchmarkMetrics(
                    test_name=f"temporal_ingestion_{size}_points",
                    system_name="temporal_reasoning",
                    throughput=size / ingestion_time if ingestion_success else 0,
                    latency_mean=ingestion_time,
                    latency_p50=ingestion_time,
                    latency_p95=ingestion_time,
                    latency_p99=ingestion_time,
                    memory_peak_mb=ingestion_resource_metrics["peak_memory_mb"],
                    memory_average_mb=ingestion_resource_metrics["avg_memory_mb"],
                    cpu_usage_avg=ingestion_resource_metrics["avg_cpu_usage"],
                    success_rate=1.0 if ingestion_success else 0.0,
                    error_count=0 if ingestion_success else 1,
                    concurrent_operations=1,
                    total_operations=1,
                    total_duration=ingestion_time,
                    additional_metrics={
                        "operation_type": "data_ingestion",
                        "data_points": size
                    }
                )
                
                if pattern_latencies:
                    pattern_metrics = BenchmarkMetrics(
                        test_name=f"temporal_pattern_detection_{size}_points",
                        system_name="temporal_reasoning",
                        throughput=len(pattern_latencies) / pattern_time,
                        latency_mean=statistics.mean(pattern_latencies),
                        latency_p50=statistics.median(pattern_latencies),
                        latency_p95=np.percentile(pattern_latencies, 95),
                        latency_p99=np.percentile(pattern_latencies, 99),
                        memory_peak_mb=pattern_resource_metrics["peak_memory_mb"],
                        memory_average_mb=pattern_resource_metrics["avg_memory_mb"],
                        cpu_usage_avg=pattern_resource_metrics["avg_cpu_usage"],
                        success_rate=(len(pattern_latencies) - pattern_errors) / len(pattern_latencies),
                        error_count=pattern_errors,
                        concurrent_operations=1,
                        total_operations=len(pattern_latencies),
                        total_duration=pattern_time,
                        additional_metrics={
                            "operation_type": "pattern_detection",
                            "data_points": size,
                            "latency_std": statistics.stdev(pattern_latencies) if len(pattern_latencies) > 1 else 0
                        }
                    )
                    benchmarks.append(pattern_metrics)
                    self.results.append(pattern_metrics)
                
                if prediction_latencies:
                    prediction_metrics = BenchmarkMetrics(
                        test_name=f"temporal_prediction_{size}_points",
                        system_name="temporal_reasoning",
                        throughput=len(prediction_latencies) / prediction_time,
                        latency_mean=statistics.mean(prediction_latencies),
                        latency_p50=statistics.median(prediction_latencies),
                        latency_p95=np.percentile(prediction_latencies, 95),
                        latency_p99=np.percentile(prediction_latencies, 99),
                        memory_peak_mb=prediction_resource_metrics["peak_memory_mb"],
                        memory_average_mb=prediction_resource_metrics["avg_memory_mb"],
                        cpu_usage_avg=prediction_resource_metrics["avg_cpu_usage"],
                        success_rate=(len(prediction_latencies) - prediction_errors) / len(prediction_latencies),
                        error_count=prediction_errors,
                        concurrent_operations=1,
                        total_operations=len(prediction_latencies),
                        total_duration=prediction_time,
                        additional_metrics={
                            "operation_type": "prediction",
                            "data_points": size,
                            "horizons_tested": len(horizons),
                            "latency_std": statistics.stdev(prediction_latencies) if len(prediction_latencies) > 1 else 0
                        }
                    )
                    benchmarks.append(prediction_metrics)
                    self.results.append(prediction_metrics)
                
                benchmarks.append(ingestion_metrics)
                self.results.append(ingestion_metrics)
        
        return benchmarks
    
    async def benchmark_integrated_reasoning_concurrency(self, integrated_controller: IntegratedReasoningController) -> List[BenchmarkMetrics]:
        """Benchmark integrated reasoning under concurrent load"""
        benchmarks = []
        
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20]
        
        for concurrency in concurrency_levels:
            logger.info(f"Benchmarking integrated reasoning with {concurrency} concurrent sessions")
            
            # Generate test queries
            queries = [
                f"Analyze business scenario {i} for optimization opportunities"
                for i in range(concurrency * 3)  # 3 queries per concurrent session
            ]
            
            self.resource_monitor.start_monitoring()
            
            # Execute concurrent reasoning sessions
            start_time = time.time()
            latencies = []
            errors = 0
            
            # Create semaphore to control concurrency
            semaphore = asyncio.Semaphore(concurrency)
            
            async def execute_query(query_id: int, query: str):
                async with semaphore:
                    try:
                        iter_start = time.time()
                        
                        result = await integrated_controller.reason(
                            query=query,
                            session_id=f"benchmark_session_{query_id}"
                        )
                        
                        iter_time = time.time() - iter_start
                        
                        if result and result.final_answer:
                            return iter_time
                        else:
                            return None  # Failed operation
                            
                    except Exception as e:
                        logger.warning(f"Integrated reasoning query failed: {e}")
                        return {}
            
            # Execute all queries concurrently
            tasks = [execute_query(i, query) for i, query in enumerate(queries)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                elif result is not None:
                    latencies.append(result)
                else:
                    errors += 1
            
            total_time = time.time() - start_time
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Create benchmark metrics
            if latencies:
                metrics = BenchmarkMetrics(
                    test_name=f"integrated_reasoning_concurrent_{concurrency}",
                    system_name="integrated_controller",
                    throughput=len(latencies) / total_time,
                    latency_mean=statistics.mean(latencies),
                    latency_p50=statistics.median(latencies),
                    latency_p95=np.percentile(latencies, 95),
                    latency_p99=np.percentile(latencies, 99),
                    memory_peak_mb=resource_metrics["peak_memory_mb"],
                    memory_average_mb=resource_metrics["avg_memory_mb"],
                    cpu_usage_avg=resource_metrics["avg_cpu_usage"],
                    success_rate=len(latencies) / len(queries),
                    error_count=errors,
                    concurrent_operations=concurrency,
                    total_operations=len(queries),
                    total_duration=total_time,
                    additional_metrics={
                        "concurrency_level": concurrency,
                        "queries_per_session": 3,
                        "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0
                    }
                )
                
                benchmarks.append(metrics)
                self.results.append(metrics)
        
        return benchmarks
    
    def _generate_causal_benchmark_data(self, n_samples: int, n_features: int) -> pd.DataFrame:
        """Generate benchmark data for causal inference testing"""
        np.random.seed(42)
        
        # Generate features with realistic causal relationships
        data = {}
        
        # Base features
        for i in range(min(2, n_features)):
            data[f'x{i}'] = np.random.normal(0, 1, n_samples)
        
        # Causally dependent features
        for i in range(2, n_features):
            # Each feature depends on 1-2 previous features
            dependencies = np.random.choice(list(data.keys()), size=min(2, len(data)), replace=False)
            
            dependent_values = np.zeros(n_samples)
            for dep in dependencies:
                weight = np.random.uniform(0.3, 0.8)
                dependent_values += weight * data[dep]
            
            # Add noise
            dependent_values += np.random.normal(0, 0.3, n_samples)
            data[f'x{i}'] = dependent_values
        
        return pd.DataFrame(data)
    
    def _generate_memory_benchmark_data(self, n_items: int) -> List[Dict[str, Any]]:
        """Generate benchmark data for memory testing"""
        memory_types = ["factual", "procedural", "episodic", "semantic"]
        
        memories = []
        for i in range(n_items):
            memory_type = memory_types[i % len(memory_types)]
            
            memories.append({
                "content": f"Benchmark memory content {i} of type {memory_type}. " +
                          f"This is detailed information about concept {i % 100} " +
                          f"with relevance score {i % 10} and complexity level {i % 5}.",
                "type": memory_type,
                "importance": np.random.uniform(0.1, 1.0),
                "metadata": {
                    "concept_id": i % 100,
                    "relevance": i % 10,
                    "complexity": i % 5,
                    "source": f"benchmark_source_{i % 10}"
                }
            })
        
        return memories
    
    def _generate_temporal_benchmark_data(self, n_points: int) -> List[Dict[str, Any]]:
        """Generate benchmark temporal data"""
        np.random.seed(42)
        
        start_time = datetime.now() - timedelta(hours=n_points)
        temporal_data = []
        
        for i in range(n_points):
            timestamp = start_time + timedelta(hours=i)
            
            # Generate value with trend, seasonality, and noise
            trend = i * 0.01
            seasonal = 5 * np.sin(2 * np.pi * i / 24)  # Daily pattern
            noise = np.random.normal(0, 1)
            
            value = 50 + trend + seasonal + noise
            
            temporal_data.append({
                "timestamp": timestamp,
                "value": value,
                "metadata": {"point_id": i}
            })
        
        return temporal_data
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.results:
            return {"error": "No benchmark results available"}
        
        # Group results by system
        by_system = {}
        for result in self.results:
            system = result.system_name
            if system not in by_system:
                by_system[system] = []
            by_system[system].append(result)
        
        # Calculate system performance summaries
        system_summaries = {}
        for system, results in by_system.items():
            throughputs = [r.throughput for r in results if r.throughput > 0]
            latencies = [r.latency_mean for r in results]
            memory_peaks = [r.memory_peak_mb for r in results]
            success_rates = [r.success_rate for r in results]
            
            system_summaries[system] = {
                "total_benchmarks": len(results),
                "avg_throughput": statistics.mean(throughputs) if throughputs else 0,
                "max_throughput": max(throughputs) if throughputs else 0,
                "min_latency": min(latencies) if latencies else 0,
                "avg_latency": statistics.mean(latencies) if latencies else 0,
                "max_latency": max(latencies) if latencies else 0,
                "peak_memory_mb": max(memory_peaks) if memory_peaks else 0,
                "avg_success_rate": statistics.mean(success_rates) if success_rates else 0,
                "total_errors": sum(r.error_count for r in results)
            }
        
        # Overall performance metrics
        all_throughputs = [r.throughput for r in self.results if r.throughput > 0]
        all_latencies = [r.latency_mean for r in self.results]
        all_success_rates = [r.success_rate for r in self.results]
        
        overall_summary = {
            "total_benchmarks": len(self.results),
            "systems_tested": len(by_system),
            "overall_avg_throughput": statistics.mean(all_throughputs) if all_throughputs else 0,
            "overall_avg_latency": statistics.mean(all_latencies) if all_latencies else 0,
            "overall_success_rate": statistics.mean(all_success_rates) if all_success_rates else 0,
            "benchmark_timestamp": datetime.now().isoformat()
        }
        
        # Performance targets and assessment
        target_assessment = {
            "sub_second_response": {
                "target": 1.0,
                "achieved": sum(1 for r in self.results if r.latency_mean < 1.0),
                "total_tests": len(self.results),
                "pass_rate": sum(1 for r in self.results if r.latency_mean < 1.0) / len(self.results)
            },
            "high_throughput": {
                "target": 10.0,
                "achieved": sum(1 for r in self.results if r.throughput >= 10.0),
                "total_tests": len([r for r in self.results if r.throughput > 0]),
                "pass_rate": sum(1 for r in self.results if r.throughput >= 10.0) / len([r for r in self.results if r.throughput > 0]) if len([r for r in self.results if r.throughput > 0]) > 0 else 0
            },
            "high_reliability": {
                "target": 0.95,
                "achieved": sum(1 for r in self.results if r.success_rate >= 0.95),
                "total_tests": len(self.results),
                "pass_rate": sum(1 for r in self.results if r.success_rate >= 0.95) / len(self.results)
            }
        }
        
        # Detailed results
        detailed_results = [
            {
                "test_name": r.test_name,
                "system": r.system_name,
                "throughput": r.throughput,
                "latency_mean": r.latency_mean,
                "latency_p95": r.latency_p95,
                "memory_peak_mb": r.memory_peak_mb,
                "cpu_usage_avg": r.cpu_usage_avg,
                "success_rate": r.success_rate,
                "concurrent_operations": r.concurrent_operations,
                "total_operations": r.total_operations,
                "additional_metrics": r.additional_metrics
            }
            for r in self.results
        ]
        
        return {
            "benchmark_summary": overall_summary,
            "system_performance": system_summaries,
            "target_assessment": target_assessment,
            "detailed_results": detailed_results,
            "recommendations": self._generate_performance_recommendations(system_summaries, target_assessment)
        }
    
    def _generate_performance_recommendations(self, system_summaries: Dict, target_assessment: Dict) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # System-specific recommendations
        for system, summary in system_summaries.items():
            if summary["avg_success_rate"] < 0.95:
                recommendations.append(
                    f"Improve {system} reliability (current: {summary['avg_success_rate']:.1%})"
                )
            
            if summary["avg_latency"] > 2.0:
                recommendations.append(
                    f"Optimize {system} response time (current: {summary['avg_latency']:.2f}s)"
                )
            
            if summary["avg_throughput"] < 10.0:
                recommendations.append(
                    f"Increase {system} throughput (current: {summary['avg_throughput']:.1f} ops/sec)"
                )
            
            if summary["peak_memory_mb"] > 1000:
                recommendations.append(
                    f"Optimize {system} memory usage (peak: {summary['peak_memory_mb']:.1f} MB)"
                )
        
        # Target-based recommendations
        for target_name, assessment in target_assessment.items():
            if assessment["pass_rate"] < 0.8:
                recommendations.append(
                    f"Improve {target_name} performance (pass rate: {assessment['pass_rate']:.1%})"
                )
        
        # General recommendations
        if not recommendations:
            recommendations.append("Performance targets met - consider stress testing at higher loads")
        
        return recommendations


@pytest.mark.performance
class TestReasoningBenchmarks:
    """Performance benchmark tests"""
    
    @pytest.mark.asyncio
    async def test_full_benchmark_suite(self):
        """Run complete benchmark suite"""
        try:
            # Initialize systems
            causal_engine = CausalReasoningEngine()
            await causal_engine.initialize()
            
            working_memory = WorkingMemorySystem()
            await working_memory.initialize()
            
            temporal_engine = TemporalReasoningEngine()
            await temporal_engine.initialize()
            
            integrated_controller = IntegratedReasoningController(
                causal_engine=causal_engine,
                working_memory=working_memory,
                temporal_engine=temporal_engine
            )
            await integrated_controller.initialize()
            
            # Run benchmark suite
            benchmark_suite = ReasoningBenchmarkSuite()
            
            # Run individual system benchmarks
            causal_benchmarks = await benchmark_suite.benchmark_causal_inference_performance(causal_engine)
            memory_benchmarks = await benchmark_suite.benchmark_working_memory_performance(working_memory)
            temporal_benchmarks = await benchmark_suite.benchmark_temporal_reasoning_performance(temporal_engine)
            
            # Run integrated system benchmarks
            integration_benchmarks = await benchmark_suite.benchmark_integrated_reasoning_concurrency(integrated_controller)
            
            # Generate report
            report = benchmark_suite.generate_benchmark_report()
            
            # Assertions
            assert len(causal_benchmarks) >= 12  # Different methods x data sizes
            assert len(memory_benchmarks) >= 8   # Storage/retrieval x data sizes  
            assert len(temporal_benchmarks) >= 9  # Different operations x data sizes
            assert len(integration_benchmarks) >= 4  # Different concurrency levels
            
            # Performance assertions
            assert report["benchmark_summary"]["overall_success_rate"] >= 0.8
            assert report["target_assessment"]["high_reliability"]["pass_rate"] >= 0.7
            
            # Log results
            logger.info(f"Benchmark completed: {report['benchmark_summary']}")
            
            # Save detailed report
            report_path = Path("test_reports/benchmark_report.json")
            report_path.parent.mkdir(exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            return report
            
        except Exception as e:
            pytest.fail(f"Benchmark suite failed: {e}")
    
    @pytest.mark.asyncio
    async def test_performance_targets(self):
        """Test specific performance targets"""
        causal_engine = CausalReasoningEngine()
        await causal_engine.initialize()
        
        # Test sub-second response for simple causal discovery
        simple_data = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100)
        })
        
        start_time = time.time()
        result = await causal_engine.discover_causal_relationships(simple_data, discovery_method="pc")
        response_time = time.time() - start_time
        
        # Assert sub-second response
        assert response_time < 1.0, f"Response time {response_time:.2f}s exceeds 1.0s target"
        assert result is not None


if __name__ == "__main__":
    # Run benchmark tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "performance"])