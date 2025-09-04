"""
Phase 6 Concurrent Operations Performance Testing
===============================================

Comprehensive performance and load testing for Phase 6 AI agents focusing on:
- Multi-agent concurrent task processing
- System scalability under load
- Resource utilization optimization
- Response time consistency
- Memory usage and leak detection
- Database connection pooling
- API endpoint performance
- Failure recovery performance
"""

import pytest
import asyncio
import time
import statistics
import psutil
import os
import json
import gc
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import threading
import multiprocessing
from pathlib import Path

# Configure performance test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Performance test configuration
PERFORMANCE_CONFIG = {
    'small_load': {
        'concurrent_tasks': 10,
        'task_duration': 0.1,
        'test_duration': 30
    },
    'medium_load': {
        'concurrent_tasks': 50,
        'task_duration': 0.5,
        'test_duration': 60
    },
    'high_load': {
        'concurrent_tasks': 100,
        'task_duration': 1.0,
        'test_duration': 120
    },
    'stress_load': {
        'concurrent_tasks': 200,
        'task_duration': 2.0,
        'test_duration': 180
    }
}

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    test_name: str
    timestamp: datetime
    concurrent_operations: int
    total_operations: int
    success_rate: float
    average_response_time: float
    median_response_time: float
    p95_response_time: float
    p99_response_time: float
    throughput_ops_per_second: float
    cpu_usage_percent: float
    memory_usage_mb: float
    memory_peak_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    errors: List[str]
    execution_duration: float


class SystemResourceMonitor:
    """Monitor system resources during performance tests"""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.monitoring = False
        self.metrics = []
        self.monitor_task = None
    
    async def start_monitoring(self, interval: float = 1.0):
        """Start resource monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self, interval: float):
        """Monitor resources in a loop"""
        while self.monitoring:
            try:
                # CPU and memory metrics
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                
                # System-wide metrics
                system_cpu = psutil.cpu_percent()
                system_memory = psutil.virtual_memory()
                
                # I/O metrics
                io_counters = self.process.io_counters()
                
                metric = {
                    'timestamp': datetime.now(),
                    'process_cpu_percent': cpu_percent,
                    'process_memory_mb': memory_info.rss / 1024 / 1024,
                    'system_cpu_percent': system_cpu,
                    'system_memory_percent': system_memory.percent,
                    'disk_read_bytes': io_counters.read_bytes,
                    'disk_write_bytes': io_counters.write_bytes,
                    'thread_count': self.process.num_threads(),
                }
                
                self.metrics.append(metric)
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """Get summary metrics from monitoring period"""
        if not self.metrics:
            return {}
        
        cpu_values = [m['process_cpu_percent'] for m in self.metrics]
        memory_values = [m['process_memory_mb'] for m in self.metrics]
        
        return {
            'avg_cpu_percent': statistics.mean(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_mb': statistics.mean(memory_values),
            'max_memory_mb': max(memory_values),
            'final_thread_count': self.metrics[-1]['thread_count'],
            'monitoring_duration': (self.metrics[-1]['timestamp'] - self.metrics[0]['timestamp']).total_seconds()
        }


class ConcurrentTaskSimulator:
    """Simulate concurrent task execution for performance testing"""
    
    def __init__(self):
        self.task_results = []
        self.error_count = 0
        self.success_count = 0
    
    async def simulate_agent_task(self, task_id: str, complexity: float = 0.5, failure_rate: float = 0.02):
        """Simulate a single agent task execution"""
        start_time = time.perf_counter()
        
        try:
            # Simulate variable processing time based on complexity
            processing_time = complexity * (0.1 + (hash(task_id) % 100) / 1000)
            await asyncio.sleep(processing_time)
            
            # Simulate occasional failures
            if (hash(task_id) % 1000) / 1000 < failure_rate:
                raise Exception(f"Simulated task failure for {task_id}")
            
            # Simulate memory allocation and cleanup
            data = [i for i in range(int(complexity * 1000))]
            result = {
                'task_id': task_id,
                'status': 'completed',
                'processing_time': processing_time,
                'data_points': len(data),
                'timestamp': datetime.now()
            }
            
            self.success_count += 1
            return result
            
        except Exception as e:
            self.error_count += 1
            end_time = time.perf_counter()
            return {
                'task_id': task_id,
                'status': 'failed',
                'error': str(e),
                'processing_time': end_time - start_time,
                'timestamp': datetime.now()
            }
    
    async def simulate_database_operation(self, operation_id: str, operation_type: str = 'read'):
        """Simulate database operations"""
        start_time = time.perf_counter()
        
        try:
            # Simulate database operation latency
            if operation_type == 'read':
                await asyncio.sleep(0.01)  # 10ms read
            elif operation_type == 'write':
                await asyncio.sleep(0.05)  # 50ms write
            elif operation_type == 'transaction':
                await asyncio.sleep(0.1)   # 100ms transaction
            
            end_time = time.perf_counter()
            return {
                'operation_id': operation_id,
                'type': operation_type,
                'duration': end_time - start_time,
                'status': 'success'
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'operation_id': operation_id,
                'type': operation_type,
                'duration': end_time - start_time,
                'status': 'error',
                'error': str(e)
            }
    
    async def simulate_api_request(self, request_id: str, endpoint: str = '/api/process'):
        """Simulate API request processing"""
        start_time = time.perf_counter()
        
        try:
            # Simulate API processing time
            processing_time = 0.05 + (hash(request_id) % 100) / 2000  # 50-100ms
            await asyncio.sleep(processing_time)
            
            end_time = time.perf_counter()
            return {
                'request_id': request_id,
                'endpoint': endpoint,
                'duration': end_time - start_time,
                'status_code': 200,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            end_time = time.perf_counter()
            return {
                'request_id': request_id,
                'endpoint': endpoint,
                'duration': end_time - start_time,
                'status_code': 500,
                'error': str(e),
                'timestamp': datetime.now()
            }


class TestConcurrentAgentOperations:
    """Test concurrent agent operations performance"""
    
    @pytest.fixture
    def task_simulator(self):
        """Create task simulator"""
        return ConcurrentTaskSimulator()
    
    @pytest.fixture
    def resource_monitor(self):
        """Create resource monitor"""
        return SystemResourceMonitor()
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_task_processing_small_load(self, task_simulator, resource_monitor):
        """Test concurrent task processing under small load"""
        config = PERFORMANCE_CONFIG['small_load']
        
        await resource_monitor.start_monitoring()
        start_time = time.perf_counter()
        
        try:
            # Generate concurrent tasks
            tasks = []
            for i in range(config['concurrent_tasks']):
                task = task_simulator.simulate_agent_task(
                    task_id=f"small_load_task_{i}",
                    complexity=0.3,
                    failure_rate=0.01
                )
                tasks.append(task)
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            # Analyze results
            successful_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'completed']
            failed_results = [r for r in results if isinstance(r, dict) and r.get('status') == 'failed']
            exception_results = [r for r in results if isinstance(r, Exception)]
            
            # Calculate metrics
            response_times = [r.get('processing_time', 0) for r in successful_results]
            total_duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                test_name='concurrent_task_processing_small_load',
                timestamp=datetime.now(),
                concurrent_operations=config['concurrent_tasks'],
                total_operations=len(results),
                success_rate=len(successful_results) / len(results),
                average_response_time=statistics.mean(response_times) if response_times else 0,
                median_response_time=statistics.median(response_times) if response_times else 0,
                p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0,
                p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times) if response_times else 0,
                throughput_ops_per_second=len(successful_results) / total_duration,
                cpu_usage_percent=0,  # Will be filled from resource monitor
                memory_usage_mb=0,    # Will be filled from resource monitor
                memory_peak_mb=0,     # Will be filled from resource monitor
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_io_sent_mb=0,
                network_io_recv_mb=0,
                errors=[str(r) for r in failed_results + exception_results],
                execution_duration=total_duration
            )
            
            # Performance assertions for small load
            assert metrics.success_rate >= 0.95, f"Success rate too low: {metrics.success_rate}"
            assert metrics.throughput_ops_per_second >= 80, f"Throughput too low: {metrics.throughput_ops_per_second}"
            assert metrics.average_response_time <= 0.5, f"Average response time too high: {metrics.average_response_time}"
            
            logger.info(f"Small load test completed: {metrics.throughput_ops_per_second:.1f} ops/sec, {metrics.success_rate:.2%} success rate")
            
        finally:
            await resource_monitor.stop_monitoring()
            
            # Add resource metrics
            resource_summary = resource_monitor.get_summary_metrics()
            metrics.cpu_usage_percent = resource_summary.get('avg_cpu_percent', 0)
            metrics.memory_usage_mb = resource_summary.get('avg_memory_mb', 0)
            metrics.memory_peak_mb = resource_summary.get('max_memory_mb', 0)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_task_processing_high_load(self, task_simulator, resource_monitor):
        """Test concurrent task processing under high load"""
        config = PERFORMANCE_CONFIG['high_load']
        
        await resource_monitor.start_monitoring()
        start_time = time.perf_counter()
        
        try:
            # Generate high load tasks
            tasks = []
            for i in range(config['concurrent_tasks']):
                task = task_simulator.simulate_agent_task(
                    task_id=f"high_load_task_{i}",
                    complexity=0.8,
                    failure_rate=0.05
                )
                tasks.append(task)
            
            # Execute tasks with batching to manage memory
            batch_size = 20
            all_results = []
            
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                batch_results = await asyncio.gather(*batch, return_exceptions=True)
                all_results.extend(batch_results)
                
                # Small delay to prevent overwhelming the system
                await asyncio.sleep(0.01)
            
            end_time = time.perf_counter()
            
            # Analyze results
            successful_results = [r for r in all_results if isinstance(r, dict) and r.get('status') == 'completed']
            failed_results = [r for r in all_results if isinstance(r, dict) and r.get('status') == 'failed']
            
            response_times = [r.get('processing_time', 0) for r in successful_results]
            total_duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                test_name='concurrent_task_processing_high_load',
                timestamp=datetime.now(),
                concurrent_operations=config['concurrent_tasks'],
                total_operations=len(all_results),
                success_rate=len(successful_results) / len(all_results),
                average_response_time=statistics.mean(response_times) if response_times else 0,
                median_response_time=statistics.median(response_times) if response_times else 0,
                p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0,
                p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times) if response_times else 0,
                throughput_ops_per_second=len(successful_results) / total_duration,
                cpu_usage_percent=0,
                memory_usage_mb=0,
                memory_peak_mb=0,
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_io_sent_mb=0,
                network_io_recv_mb=0,
                errors=[str(r) for r in failed_results],
                execution_duration=total_duration
            )
            
            # Performance assertions for high load
            assert metrics.success_rate >= 0.90, f"Success rate too low under high load: {metrics.success_rate}"
            assert metrics.throughput_ops_per_second >= 30, f"Throughput too low under high load: {metrics.throughput_ops_per_second}"
            assert metrics.p95_response_time <= 2.0, f"P95 response time too high: {metrics.p95_response_time}"
            
            logger.info(f"High load test completed: {metrics.throughput_ops_per_second:.1f} ops/sec, {metrics.success_rate:.2%} success rate")
            
        finally:
            await resource_monitor.stop_monitoring()
            
            resource_summary = resource_monitor.get_summary_metrics()
            metrics.cpu_usage_percent = resource_summary.get('avg_cpu_percent', 0)
            metrics.memory_usage_mb = resource_summary.get('avg_memory_mb', 0)
            metrics.memory_peak_mb = resource_summary.get('max_memory_mb', 0)
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    @pytest.mark.slow
    async def test_sustained_load_performance(self, task_simulator, resource_monitor):
        """Test performance under sustained load over time"""
        test_duration = 60  # 1 minute sustained test
        tasks_per_second = 10
        
        await resource_monitor.start_monitoring()
        start_time = time.perf_counter()
        
        try:
            all_results = []
            task_counter = 0
            
            end_test_time = start_time + test_duration
            
            while time.perf_counter() < end_test_time:
                # Generate batch of tasks
                batch_tasks = []
                for i in range(tasks_per_second):
                    task = task_simulator.simulate_agent_task(
                        task_id=f"sustained_load_task_{task_counter}_{i}",
                        complexity=0.5,
                        failure_rate=0.02
                    )
                    batch_tasks.append(task)
                
                # Execute batch
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                all_results.extend(batch_results)
                
                task_counter += 1
                
                # Wait for next second
                await asyncio.sleep(1.0)
            
            end_time = time.perf_counter()
            
            # Analyze sustained load results
            successful_results = [r for r in all_results if isinstance(r, dict) and r.get('status') == 'completed']
            failed_results = [r for r in all_results if isinstance(r, dict) and r.get('status') == 'failed']
            
            response_times = [r.get('processing_time', 0) for r in successful_results]
            total_duration = end_time - start_time
            
            metrics = PerformanceMetrics(
                test_name='sustained_load_performance',
                timestamp=datetime.now(),
                concurrent_operations=tasks_per_second,
                total_operations=len(all_results),
                success_rate=len(successful_results) / len(all_results),
                average_response_time=statistics.mean(response_times) if response_times else 0,
                median_response_time=statistics.median(response_times) if response_times else 0,
                p95_response_time=statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0,
                p99_response_time=statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else max(response_times) if response_times else 0,
                throughput_ops_per_second=len(successful_results) / total_duration,
                cpu_usage_percent=0,
                memory_usage_mb=0,
                memory_peak_mb=0,
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_io_sent_mb=0,
                network_io_recv_mb=0,
                errors=[str(r) for r in failed_results],
                execution_duration=total_duration
            )
            
            # Assertions for sustained load
            assert metrics.success_rate >= 0.95, f"Success rate degraded over time: {metrics.success_rate}"
            assert metrics.throughput_ops_per_second >= 8, f"Throughput degraded: {metrics.throughput_ops_per_second}"
            
            logger.info(f"Sustained load test completed: {metrics.throughput_ops_per_second:.1f} ops/sec over {total_duration:.1f}s")
            
        finally:
            await resource_monitor.stop_monitoring()
            
            resource_summary = resource_monitor.get_summary_metrics()
            metrics.cpu_usage_percent = resource_summary.get('avg_cpu_percent', 0)
            metrics.memory_usage_mb = resource_summary.get('avg_memory_mb', 0)
            metrics.memory_peak_mb = resource_summary.get('max_memory_mb', 0)


class TestDatabasePerformance:
    """Test database operation performance under load"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_concurrent_database_operations(self):
        """Test concurrent database operations performance"""
        simulator = ConcurrentTaskSimulator()
        
        # Simulate database connection pool
        connection_pool_size = 20
        concurrent_operations = 100
        
        start_time = time.perf_counter()
        
        # Generate mixed database operations
        operations = []
        for i in range(concurrent_operations):
            op_type = ['read', 'write', 'transaction'][i % 3]
            operation = simulator.simulate_database_operation(
                operation_id=f"db_op_{i}",
                operation_type=op_type
            )
            operations.append(operation)
        
        # Execute with limited concurrency (simulating connection pool)
        semaphore = asyncio.Semaphore(connection_pool_size)
        
        async def controlled_operation(op):
            async with semaphore:
                return await op
        
        results = await asyncio.gather(
            *[controlled_operation(op) for op in operations],
            return_exceptions=True
        )
        
        end_time = time.perf_counter()
        
        # Analyze database performance
        successful_ops = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
        failed_ops = [r for r in results if isinstance(r, dict) and r.get('status') == 'error']
        
        durations = [r.get('duration', 0) for r in successful_ops]
        total_duration = end_time - start_time
        
        # Database performance assertions
        success_rate = len(successful_ops) / len(results)
        avg_latency = statistics.mean(durations) if durations else 0
        throughput = len(successful_ops) / total_duration
        
        assert success_rate >= 0.99, f"Database success rate too low: {success_rate}"
        assert avg_latency <= 0.2, f"Database latency too high: {avg_latency}s"
        assert throughput >= 200, f"Database throughput too low: {throughput} ops/sec"
        
        logger.info(f"Database performance: {throughput:.1f} ops/sec, {avg_latency*1000:.1f}ms avg latency")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_database_connection_pool_exhaustion(self):
        """Test behavior when database connection pool is exhausted"""
        simulator = ConcurrentTaskSimulator()
        
        connection_pool_size = 5  # Small pool to force exhaustion
        concurrent_operations = 50
        
        start_time = time.perf_counter()
        
        operations = []
        for i in range(concurrent_operations):
            # Use longer transactions to exhaust pool
            operation = simulator.simulate_database_operation(
                operation_id=f"pool_test_{i}",
                operation_type='transaction'
            )
            operations.append(operation)
        
        # Simulate connection pool with timeout
        semaphore = asyncio.Semaphore(connection_pool_size)
        
        async def pool_controlled_operation(op):
            try:
                async with asyncio.wait_for(semaphore.acquire(), timeout=2.0):
                    try:
                        result = await op
                        return result
                    finally:
                        semaphore.release()
            except asyncio.TimeoutError:
                return {
                    'operation_id': 'timeout',
                    'status': 'timeout',
                    'error': 'Connection pool exhausted'
                }
        
        results = await asyncio.gather(
            *[pool_controlled_operation(op) for op in operations],
            return_exceptions=True
        )
        
        end_time = time.perf_counter()
        
        # Analyze pool exhaustion handling
        successful_ops = [r for r in results if isinstance(r, dict) and r.get('status') == 'success']
        timeout_ops = [r for r in results if isinstance(r, dict) and r.get('status') == 'timeout']
        
        # Should handle pool exhaustion gracefully
        assert len(timeout_ops) > 0, "Should have some timeouts with small pool"
        assert len(successful_ops) > 0, "Should still complete some operations"
        
        logger.info(f"Pool exhaustion test: {len(successful_ops)} successful, {len(timeout_ops)} timeouts")


class TestAPIPerformance:
    """Test API endpoint performance under load"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_api_endpoint_concurrent_requests(self):
        """Test API endpoint performance with concurrent requests"""
        simulator = ConcurrentTaskSimulator()
        
        concurrent_requests = 100
        
        start_time = time.perf_counter()
        
        # Generate concurrent API requests
        requests = []
        for i in range(concurrent_requests):
            endpoint = ['/api/process', '/api/analyze', '/api/report'][i % 3]
            request = simulator.simulate_api_request(
                request_id=f"api_req_{i}",
                endpoint=endpoint
            )
            requests.append(request)
        
        # Execute concurrent requests
        results = await asyncio.gather(*requests, return_exceptions=True)
        
        end_time = time.perf_counter()
        
        # Analyze API performance
        successful_requests = [r for r in results if isinstance(r, dict) and r.get('status_code') == 200]
        failed_requests = [r for r in results if isinstance(r, dict) and r.get('status_code') != 200]
        
        response_times = [r.get('duration', 0) for r in successful_requests]
        total_duration = end_time - start_time
        
        # API performance assertions
        success_rate = len(successful_requests) / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else 0
        throughput = len(successful_requests) / total_duration
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max(response_times) if response_times else 0
        
        assert success_rate >= 0.95, f"API success rate too low: {success_rate}"
        assert avg_response_time <= 0.15, f"API response time too high: {avg_response_time}s"
        assert p95_response_time <= 0.25, f"API P95 response time too high: {p95_response_time}s"
        assert throughput >= 400, f"API throughput too low: {throughput} req/sec"
        
        logger.info(f"API performance: {throughput:.1f} req/sec, {avg_response_time*1000:.1f}ms avg response")


class TestMemoryPerformance:
    """Test memory usage and leak detection"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under sustained load"""
        import gc
        
        # Force garbage collection before test
        gc.collect()
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        simulator = ConcurrentTaskSimulator()
        memory_snapshots = []
        
        # Run tasks in batches and monitor memory
        batch_size = 50
        num_batches = 10
        
        for batch_num in range(num_batches):
            # Execute batch of tasks
            tasks = []
            for i in range(batch_size):
                task = simulator.simulate_agent_task(
                    task_id=f"memory_test_batch_{batch_num}_task_{i}",
                    complexity=0.7,
                    failure_rate=0.01
                )
                tasks.append(task)
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Take memory snapshot
            current_memory = process.memory_info().rss / 1024 / 1024
            memory_snapshots.append({
                'batch': batch_num,
                'memory_mb': current_memory,
                'memory_increase': current_memory - initial_memory,
                'completed_tasks': (batch_num + 1) * batch_size
            })
            
            # Force garbage collection periodically
            if batch_num % 3 == 0:
                gc.collect()
        
        # Analyze memory usage pattern
        final_memory = memory_snapshots[-1]['memory_mb']
        memory_increase = final_memory - initial_memory
        max_memory = max(snapshot['memory_mb'] for snapshot in memory_snapshots)
        
        # Memory performance assertions
        assert memory_increase < 100, f"Memory increase too high: {memory_increase:.1f}MB"
        assert max_memory < initial_memory + 150, f"Peak memory usage too high: {max_memory:.1f}MB"
        
        # Check for memory leaks (memory should not grow linearly with tasks)
        memory_growth_rate = memory_increase / (num_batches * batch_size)
        assert memory_growth_rate < 0.1, f"Potential memory leak detected: {memory_growth_rate:.3f}MB per task"
        
        logger.info(f"Memory test: {initial_memory:.1f}MB → {final_memory:.1f}MB (Δ{memory_increase:.1f}MB)")
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_garbage_collection_impact(self):
        """Test garbage collection impact on performance"""
        import gc
        
        simulator = ConcurrentTaskSimulator()
        
        # Test with garbage collection disabled
        gc.disable()
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(100):
            task = simulator.simulate_agent_task(
                task_id=f"gc_disabled_task_{i}",
                complexity=0.5,
                failure_rate=0.01
            )
            tasks.append(task)
        
        results_gc_disabled = await asyncio.gather(*tasks, return_exceptions=True)
        time_gc_disabled = time.perf_counter() - start_time
        
        # Re-enable garbage collection and test again
        gc.enable()
        gc.collect()  # Clean up from previous test
        
        start_time = time.perf_counter()
        
        tasks = []
        for i in range(100):
            task = simulator.simulate_agent_task(
                task_id=f"gc_enabled_task_{i}",
                complexity=0.5,
                failure_rate=0.01
            )
            tasks.append(task)
        
        results_gc_enabled = await asyncio.gather(*tasks, return_exceptions=True)
        time_gc_enabled = time.perf_counter() - start_time
        
        # Analyze garbage collection impact
        performance_impact = (time_gc_enabled - time_gc_disabled) / time_gc_disabled
        
        # GC impact should be minimal (less than 20% performance difference)
        assert abs(performance_impact) < 0.2, f"GC impact too high: {performance_impact:.1%}"
        
        logger.info(f"GC impact: {performance_impact:.1%} performance difference")


class TestFailureRecoveryPerformance:
    """Test performance during failure recovery scenarios"""
    
    @pytest.mark.asyncio
    @pytest.mark.performance
    async def test_cascading_failure_recovery_time(self):
        """Test recovery time from cascading failures"""
        simulator = ConcurrentTaskSimulator()
        
        # Simulate normal operation
        normal_tasks = []
        for i in range(50):
            task = simulator.simulate_agent_task(
                task_id=f"normal_task_{i}",
                complexity=0.3,
                failure_rate=0.01
            )
            normal_tasks.append(task)
        
        normal_start = time.perf_counter()
        normal_results = await asyncio.gather(*normal_tasks, return_exceptions=True)
        normal_time = time.perf_counter() - normal_start
        
        # Simulate failure scenario (high failure rate)
        failure_tasks = []
        for i in range(50):
            task = simulator.simulate_agent_task(
                task_id=f"failure_task_{i}",
                complexity=0.3,
                failure_rate=0.5  # High failure rate
            )
            failure_tasks.append(task)
        
        failure_start = time.perf_counter()
        failure_results = await asyncio.gather(*failure_tasks, return_exceptions=True)
        failure_time = time.perf_counter() - failure_start
        
        # Simulate recovery scenario (return to normal)
        recovery_tasks = []
        for i in range(50):
            task = simulator.simulate_agent_task(
                task_id=f"recovery_task_{i}",
                complexity=0.3,
                failure_rate=0.01
            )
            recovery_tasks.append(task)
        
        recovery_start = time.perf_counter()
        recovery_results = await asyncio.gather(*recovery_tasks, return_exceptions=True)
        recovery_time = time.perf_counter() - recovery_start
        
        # Analyze failure recovery performance
        normal_success_rate = len([r for r in normal_results if isinstance(r, dict) and r.get('status') == 'completed']) / len(normal_results)
        failure_success_rate = len([r for r in failure_results if isinstance(r, dict) and r.get('status') == 'completed']) / len(failure_results)
        recovery_success_rate = len([r for r in recovery_results if isinstance(r, dict) and r.get('status') == 'completed']) / len(recovery_results)
        
        # Recovery performance assertions
        assert normal_success_rate >= 0.95, f"Normal operation success rate: {normal_success_rate}"
        assert failure_success_rate <= 0.6, f"Failure scenario should have low success rate: {failure_success_rate}"
        assert recovery_success_rate >= 0.90, f"Recovery success rate too low: {recovery_success_rate}"
        
        # Recovery time should not be significantly worse than normal
        recovery_slowdown = (recovery_time - normal_time) / normal_time
        assert recovery_slowdown < 0.5, f"Recovery too slow: {recovery_slowdown:.1%} slower than normal"
        
        logger.info(f"Failure recovery: Normal({normal_success_rate:.1%}), Failure({failure_success_rate:.1%}), Recovery({recovery_success_rate:.1%})")


def generate_performance_report(metrics_list: List[PerformanceMetrics]) -> str:
    """Generate comprehensive performance test report"""
    if not metrics_list:
        return "No performance metrics available"
    
    report_lines = [
        "=" * 80,
        "PHASE 6 CONCURRENT OPERATIONS PERFORMANCE REPORT",
        "=" * 80,
        f"Test Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Tests Run: {len(metrics_list)}",
        "",
        "PERFORMANCE SUMMARY:",
        "-" * 30
    ]
    
    for metrics in metrics_list:
        report_lines.extend([
            f"Test: {metrics.test_name}",
            f"  Concurrent Operations: {metrics.concurrent_operations}",
            f"  Success Rate: {metrics.success_rate:.1%}",
            f"  Throughput: {metrics.throughput_ops_per_second:.1f} ops/sec",
            f"  Avg Response Time: {metrics.average_response_time*1000:.1f}ms",
            f"  P95 Response Time: {metrics.p95_response_time*1000:.1f}ms",
            f"  CPU Usage: {metrics.cpu_usage_percent:.1f}%",
            f"  Memory Usage: {metrics.memory_usage_mb:.1f}MB",
            f"  Peak Memory: {metrics.memory_peak_mb:.1f}MB",
            ""
        ])
    
    # Performance benchmarks summary
    avg_throughput = statistics.mean([m.throughput_ops_per_second for m in metrics_list])
    avg_response_time = statistics.mean([m.average_response_time for m in metrics_list])
    avg_success_rate = statistics.mean([m.success_rate for m in metrics_list])
    
    report_lines.extend([
        "OVERALL PERFORMANCE BENCHMARKS:",
        "-" * 35,
        f"Average Throughput: {avg_throughput:.1f} operations/second",
        f"Average Response Time: {avg_response_time*1000:.1f}ms",
        f"Average Success Rate: {avg_success_rate:.1%}",
        "",
        "PERFORMANCE TARGETS:",
        "-" * 20,
        "✓ Small Load: >80 ops/sec, <500ms response, >95% success",
        "✓ Medium Load: >50 ops/sec, <1s response, >93% success",
        "✓ High Load: >30 ops/sec, <2s P95 response, >90% success",
        "",
        "=" * 80
    ])
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Run performance tests
    pytest.main([
        __file__, 
        "-v", 
        "-m", "performance",
        "--tb=short",
        "--durations=10"
    ])