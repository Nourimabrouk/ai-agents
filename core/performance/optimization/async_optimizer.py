"""
Async Performance Optimizer for Phase 7 - High-Performance Async Operations
Optimizes async/await patterns, connection pooling, and concurrent processing
"""

import asyncio
import aiohttp
import asyncpg
import time
from typing import Dict, List, Any, Optional, Callable, Tuple, Union, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing
import logging
from contextlib import asynccontextmanager
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)


@dataclass
class AsyncMetrics:
    """Async performance metrics tracking"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_execution_time: float = 0.0
    max_concurrent_tasks: int = 0
    current_concurrent_tasks: int = 0
    connection_pool_usage: Dict[str, int] = field(default_factory=dict)
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0


@dataclass
class ConnectionPoolConfig:
    """Connection pool configuration"""
    min_size: int = 10
    max_size: int = 50
    command_timeout: int = 60
    server_settings: Dict[str, str] = field(default_factory=dict)
    retry_attempts: int = 3
    retry_delay: float = 1.0


class AsyncOptimizer:
    """
    Advanced async performance optimizer with connection pooling,
    task batching, and concurrency management
    """
    
    def __init__(self, 
                 max_concurrent_tasks: int = 1000,
                 max_workers: int = None,
                 enable_profiling: bool = True):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.max_workers = max_workers or multiprocessing.cpu_count() * 2
        self.enable_profiling = enable_profiling
        
        # Task management
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_queue = asyncio.Queue(maxsize=max_concurrent_tasks * 2)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_groups: Dict[str, List[asyncio.Task]] = defaultdict(list)
        
        # Connection pools
        self.connection_pools: Dict[str, Any] = {}
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        # Thread and process pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers or multiprocessing.cpu_count())
        
        # Rate limiting
        self.rate_limiters: Dict[str, deque] = defaultdict(deque)
        
        # Performance tracking
        self.metrics = AsyncMetrics()
        self.execution_times: deque = deque(maxlen=1000)  # Keep last 1000 execution times
        
        logger.info(f"AsyncOptimizer initialized with {max_concurrent_tasks} max concurrent tasks")
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self._initialize_pools()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()
    
    async def _initialize_pools(self):
        """Initialize connection pools and HTTP session"""
        # Initialize HTTP session with optimized settings
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent_tasks,
            limit_per_host=50,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        self.http_session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'Phase7-AsyncOptimizer/1.0'}
        )
        
        logger.info("Initialized HTTP session with optimized connector")
    
    async def create_database_pool(self, 
                                 database_url: str, 
                                 pool_name: str = "default",
                                 config: Optional[ConnectionPoolConfig] = None) -> asyncpg.Pool:
        """Create optimized database connection pool"""
        
        if config is None:
            config = ConnectionPoolConfig()
        
        try:
            pool = await asyncpg.create_pool(
                database_url,
                min_size=config.min_size,
                max_size=config.max_size,
                command_timeout=config.command_timeout,
                server_settings=config.server_settings or {
                    'jit': 'off',  # Disable JIT for predictable performance
                    'application_name': f'phase7_async_pool_{pool_name}'
                }
            )
            
            self.connection_pools[pool_name] = pool
            self.metrics.connection_pool_usage[pool_name] = 0
            
            logger.info(f"Created database pool '{pool_name}' with {config.min_size}-{config.max_size} connections")
            return pool
            
        except Exception as e:
            logger.error(f"Failed to create database pool '{pool_name}': {e}")
            raise
    
    def get_database_pool(self, pool_name: str = "default") -> Optional[asyncpg.Pool]:
        """Get database connection pool by name"""
        return self.connection_pools.get(pool_name)
    
    @asynccontextmanager
    async def database_transaction(self, pool_name: str = "default"):
        """Optimized database transaction context manager"""
        pool = self.get_database_pool(pool_name)
        if not pool:
            raise ValueError(f"Database pool '{pool_name}' not found")
        
        async with pool.acquire() as connection:
            self.metrics.connection_pool_usage[pool_name] += 1
            try:
                async with connection.transaction():
                    yield connection
            finally:
                self.metrics.connection_pool_usage[pool_name] -= 1
    
    async def execute_with_concurrency_limit(self, 
                                           coro: Coroutine, 
                                           task_id: Optional[str] = None) -> Any:
        """Execute coroutine with concurrency limiting"""
        
        async with self.semaphore:
            if task_id:
                self.active_tasks[task_id] = asyncio.current_task()
            
            self.metrics.current_concurrent_tasks += 1
            self.metrics.max_concurrent_tasks = max(
                self.metrics.max_concurrent_tasks,
                self.metrics.current_concurrent_tasks
            )
            
            start_time = time.perf_counter()
            
            try:
                result = await coro
                self.metrics.completed_tasks += 1
                return result
                
            except Exception as e:
                self.metrics.failed_tasks += 1
                logger.error(f"Task {task_id} failed: {e}")
                raise
                
            finally:
                execution_time = time.perf_counter() - start_time
                self.execution_times.append(execution_time)
                
                # Update average execution time
                if self.execution_times:
                    self.metrics.avg_execution_time = sum(self.execution_times) / len(self.execution_times)
                
                self.metrics.current_concurrent_tasks -= 1
                self.metrics.total_tasks += 1
                
                if task_id and task_id in self.active_tasks:
                    del self.active_tasks[task_id]
    
    async def batch_execute(self, 
                          coroutines: List[Coroutine], 
                          batch_size: int = 50,
                          delay_between_batches: float = 0.1,
                          group_id: Optional[str] = None) -> List[Any]:
        """Execute coroutines in batches to prevent overwhelming the system"""
        
        results = []
        total_batches = (len(coroutines) + batch_size - 1) // batch_size
        
        logger.info(f"Executing {len(coroutines)} coroutines in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(coroutines), batch_size):
            batch = coroutines[i:i + batch_size]
            batch_id = f"batch_{i//batch_size + 1}"
            
            # Create tasks with concurrency limiting
            tasks = [
                self.execute_with_concurrency_limit(coro, f"{batch_id}_task_{j}")
                for j, coro in enumerate(batch)
            ]
            
            if group_id:
                self.task_groups[group_id].extend(tasks)
            
            # Execute batch
            try:
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend(batch_results)
                
                logger.debug(f"Completed batch {i//batch_size + 1}/{total_batches}")
                
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                # Add None for failed batch items
                results.extend([None] * len(batch))
            
            # Delay between batches to prevent overload
            if i + batch_size < len(coroutines) and delay_between_batches > 0:
                await asyncio.sleep(delay_between_batches)
        
        return results
    
    async def parallel_map(self, 
                         func: Callable, 
                         items: List[Any],
                         max_concurrent: int = 100,
                         use_process_pool: bool = False) -> List[Any]:
        """Parallel map with concurrency control"""
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_func(item):
            async with semaphore:
                if asyncio.iscoroutinefunction(func):
                    return await func(item)
                elif use_process_pool:
                    # Use process pool for CPU-intensive tasks
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.process_pool, func, item)
                else:
                    # Use thread pool for I/O-bound tasks
                    loop = asyncio.get_event_loop()
                    return await loop.run_in_executor(self.thread_pool, func, item)
        
        tasks = [limited_func(item) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    async def rate_limited_execute(self, 
                                 coro: Coroutine, 
                                 operation_id: str,
                                 max_per_second: float = 10.0) -> Any:
        """Execute coroutine with rate limiting"""
        
        now = time.time()
        window_start = now - 1.0  # 1-second window
        
        # Clean old entries
        rate_limiter = self.rate_limiters[operation_id]
        while rate_limiter and rate_limiter[0] < window_start:
            rate_limiter.popleft()
        
        # Check rate limit
        if len(rate_limiter) >= max_per_second:
            # Calculate wait time
            oldest_request = rate_limiter[0]
            wait_time = oldest_request + 1.0 - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        # Record request
        rate_limiter.append(now)
        
        # Execute coroutine
        return await coro
    
    async def timeout_wrapper(self, 
                            coro: Coroutine, 
                            timeout: float,
                            fallback_value: Any = None) -> Any:
        """Execute coroutine with timeout and fallback"""
        try:
            return await asyncio.wait_for(coro, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Operation timed out after {timeout}s, returning fallback value")
            return fallback_value
    
    async def retry_with_backoff(self,
                               coro_func: Callable,
                               max_retries: int = 3,
                               base_delay: float = 1.0,
                               max_delay: float = 60.0,
                               backoff_multiplier: float = 2.0,
                               exceptions: Tuple = (Exception,)) -> Any:
        """Execute coroutine with exponential backoff retry"""
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(coro_func):
                    return await coro_func()
                else:
                    return await coro_func
                    
            except exceptions as e:
                if attempt == max_retries - 1:
                    logger.error(f"Max retries ({max_retries}) exceeded for operation")
                    raise
                
                delay = min(base_delay * (backoff_multiplier ** attempt), max_delay)
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s")
                await asyncio.sleep(delay)
    
    async def circuit_breaker(self,
                            coro_func: Callable,
                            failure_threshold: int = 5,
                            recovery_timeout: float = 60.0,
                            circuit_id: str = "default") -> Any:
        """Circuit breaker pattern for resilient async operations"""
        
        # Initialize circuit state if not exists
        if not hasattr(self, '_circuit_states'):
            self._circuit_states = {}
        
        if circuit_id not in self._circuit_states:
            self._circuit_states[circuit_id] = {
                'failures': 0,
                'last_failure_time': 0,
                'state': 'closed'  # closed, open, half_open
            }
        
        circuit = self._circuit_states[circuit_id]
        now = time.time()
        
        # Check if circuit should be half-open
        if (circuit['state'] == 'open' and 
            now - circuit['last_failure_time'] > recovery_timeout):
            circuit['state'] = 'half_open'
        
        # Reject if circuit is open
        if circuit['state'] == 'open':
            raise Exception(f"Circuit breaker '{circuit_id}' is open")
        
        try:
            result = await coro_func()
            
            # Reset on success
            if circuit['state'] == 'half_open':
                circuit['state'] = 'closed'
                circuit['failures'] = 0
            
            return result
            
        except Exception as e:
            circuit['failures'] += 1
            circuit['last_failure_time'] = now
            
            if circuit['failures'] >= failure_threshold:
                circuit['state'] = 'open'
                logger.warning(f"Circuit breaker '{circuit_id}' opened after {failure_threshold} failures")
            
            raise
    
    async def optimized_http_request(self, 
                                   method: str, 
                                   url: str,
                                   **kwargs) -> aiohttp.ClientResponse:
        """Optimized HTTP request with connection reuse"""
        if not self.http_session:
            await self._initialize_pools()
        
        return await self.http_session.request(method, url, **kwargs)
    
    async def bulk_http_requests(self, 
                               requests: List[Dict[str, Any]],
                               max_concurrent: int = 50) -> List[Any]:
        """Execute multiple HTTP requests concurrently"""
        
        async def make_request(request_config):
            method = request_config.get('method', 'GET')
            url = request_config['url']
            kwargs = {k: v for k, v in request_config.items() if k not in ['method', 'url']}
            
            try:
                async with await self.optimized_http_request(method, url, **kwargs) as response:
                    return {
                        'url': url,
                        'status': response.status,
                        'data': await response.text() if response.status == 200 else None,
                        'headers': dict(response.headers)
                    }
            except Exception as e:
                return {
                    'url': url,
                    'error': str(e),
                    'status': None
                }
        
        # Execute with concurrency control
        coroutines = [make_request(req) for req in requests]
        return await self.batch_execute(coroutines, batch_size=max_concurrent)
    
    def monitor_task_group(self, group_id: str) -> Dict[str, Any]:
        """Monitor the status of a task group"""
        tasks = self.task_groups.get(group_id, [])
        
        completed = sum(1 for task in tasks if task.done())
        failed = sum(1 for task in tasks if task.done() and task.exception())
        running = sum(1 for task in tasks if not task.done())
        
        return {
            'group_id': group_id,
            'total_tasks': len(tasks),
            'completed': completed,
            'failed': failed,
            'running': running,
            'success_rate': (completed - failed) / max(1, completed) * 100 if completed > 0 else 0
        }
    
    async def cancel_task_group(self, group_id: str) -> int:
        """Cancel all tasks in a group"""
        tasks = self.task_groups.get(group_id, [])
        cancelled = 0
        
        for task in tasks:
            if not task.done():
                task.cancel()
                cancelled += 1
        
        # Wait for cancellation
        if cancelled > 0:
            await asyncio.sleep(0.1)
        
        logger.info(f"Cancelled {cancelled} tasks in group '{group_id}'")
        return cancelled
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive async performance metrics"""
        return {
            'total_tasks': self.metrics.total_tasks,
            'completed_tasks': self.metrics.completed_tasks,
            'failed_tasks': self.metrics.failed_tasks,
            'success_rate': f"{self.metrics.success_rate:.2f}%",
            'avg_execution_time': f"{self.metrics.avg_execution_time:.4f}s",
            'max_concurrent_tasks': self.metrics.max_concurrent_tasks,
            'current_concurrent_tasks': self.metrics.current_concurrent_tasks,
            'active_tasks_count': len(self.active_tasks),
            'task_groups_count': len(self.task_groups),
            'connection_pools': list(self.connection_pools.keys()),
            'connection_pool_usage': dict(self.metrics.connection_pool_usage),
            'rate_limiters_active': len(self.rate_limiters),
            'thread_pool_active': self.thread_pool._threads,
            'process_pool_active': len(self.process_pool._processes) if hasattr(self.process_pool, '_processes') else 0
        }
    
    async def optimize_existing_coroutine(self, coro: Coroutine) -> Any:
        """Apply multiple optimizations to an existing coroutine"""
        # Wrap with timeout
        timeout_coro = self.timeout_wrapper(coro, timeout=30.0)
        
        # Apply concurrency limiting
        limited_coro = self.execute_with_concurrency_limit(timeout_coro)
        
        # Apply retry logic
        return await self.retry_with_backoff(lambda: limited_coro)
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Starting AsyncOptimizer cleanup...")
        
        # Cancel all active tasks
        for task_id, task in self.active_tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Cancelled active task: {task_id}")
        
        # Cancel task groups
        for group_id in list(self.task_groups.keys()):
            await self.cancel_task_group(group_id)
        
        # Close HTTP session
        if self.http_session:
            await self.http_session.close()
            logger.debug("Closed HTTP session")
        
        # Close database pools
        for pool_name, pool in self.connection_pools.items():
            await pool.close()
            logger.debug(f"Closed database pool: {pool_name}")
        
        # Shutdown thread and process pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        logger.info("AsyncOptimizer cleanup completed")


# Global async optimizer instance
optimizer = None

async def get_optimizer(max_concurrent_tasks: int = 1000) -> AsyncOptimizer:
    """Get global async optimizer instance"""
    global optimizer
    if optimizer is None:
        optimizer = AsyncOptimizer(max_concurrent_tasks=max_concurrent_tasks)
        await optimizer._initialize_pools()
    return optimizer


# Convenience decorators
def async_optimized(max_concurrent: int = 100, timeout: float = 30.0):
    """Decorator to optimize async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            opt = await get_optimizer()
            coro = func(*args, **kwargs)
            
            # Apply optimizations
            timeout_coro = opt.timeout_wrapper(coro, timeout)
            return await opt.execute_with_concurrency_limit(timeout_coro)
            
        return wrapper
    return decorator


def rate_limited(max_per_second: float = 10.0, operation_id: str = None):
    """Decorator for rate-limited async functions"""
    def decorator(func):
        nonlocal operation_id
        if operation_id is None:
            operation_id = func.__name__
            
        @wraps(func)
        async def wrapper(*args, **kwargs):
            opt = await get_optimizer()
            coro = func(*args, **kwargs)
            return await opt.rate_limited_execute(coro, operation_id, max_per_second)
            
        return wrapper
    return decorator
