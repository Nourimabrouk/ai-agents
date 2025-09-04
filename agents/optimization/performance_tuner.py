"""
Performance Optimizer - Cost-Effective Operations Manager
=========================================================

Advanced performance optimization system for AI agent operations with focus on:
- Token usage minimization
- API cost management
- Caching strategies
- Batch processing optimization
- Parallel execution management
- Resource allocation optimization

Author: META-ORCHESTRATOR Agent
Phase: 6 - Self-Improving Agent Ecosystem
"""

import asyncio
import time
import hashlib
import json
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from functools import wraps
import statistics
import pickle
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizationType(Enum):
    """Types of optimization strategies"""
    TOKEN_REDUCTION = "token_reduction"
    CACHING = "caching"
    BATCHING = "batching"
    PARALLEL = "parallel"
    COST_REDUCTION = "cost_reduction"
    LATENCY = "latency"
    THROUGHPUT = "throughput"

class CacheStrategy(Enum):
    """Cache eviction strategies"""
    LRU = "least_recently_used"
    LFU = "least_frequently_used"
    TTL = "time_to_live"
    ADAPTIVE = "adaptive"

@dataclass
class PerformanceMetric:
    """Performance measurement data"""
    operation_id: str
    start_time: float
    end_time: float
    duration: float
    token_usage: int
    api_cost: float
    success: bool
    error_message: Optional[str] = None
    cache_hit: bool = False
    optimization_type: Optional[OptimizationType] = None
    
    @property
    def throughput(self) -> float:
        """Operations per second"""
        return 1.0 / self.duration if self.duration > 0 else 0.0

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    cost_saved: float = 0.0
    size_bytes: Optional[int] = None
    ttl: Optional[timedelta] = None
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired"""
        if not self.ttl:
            return False
        return datetime.now() - self.created_at > self.ttl

@dataclass
class BatchOperation:
    """Batch operation container"""
    operation_type: str
    items: List[Any]
    batch_size: int
    priority: int = 1
    timeout: Optional[float] = None
    callback: Optional[Callable] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class ResourceUsage:
    """Resource usage tracking"""
    cpu_percent: float
    memory_mb: float
    active_connections: int
    pending_operations: int
    cache_size_mb: float
    token_rate_per_minute: float
    cost_per_hour: float
    timestamp: datetime = field(default_factory=datetime.now)

class TokenOptimizer:
    """Token usage optimization strategies"""
    
    def __init__(self):
        self.compression_patterns = {
            'redundant_whitespace': r'\s+',
            'verbose_phrases': {
                'in order to': 'to',
                'for the purpose of': 'to',
                'due to the fact that': 'because',
                'in the event that': 'if',
                'at this point in time': 'now'
            },
            'technical_abbreviations': {
                'application programming interface': 'API',
                'artificial intelligence': 'AI',
                'machine learning': 'ML',
                'natural language processing': 'NLP'
            }
        }
        
    def optimize_prompt(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        """Optimize prompt for token efficiency"""
        original_length = len(prompt)
        optimized = prompt
        
        # Remove redundant whitespace
        import re
        optimized = re.sub(r'\s+', ' ', optimized.strip())
        
        # Replace verbose phrases
        for verbose, concise in self.compression_patterns['verbose_phrases'].items():
            optimized = optimized.replace(verbose, concise)
            
        # Use technical abbreviations
        for full_term, abbrev in self.compression_patterns['technical_abbreviations'].items():
            optimized = optimized.replace(full_term, abbrev)
        
        # Estimate token savings (rough approximation)
        estimated_tokens_saved = (original_length - len(optimized)) // 4
        
        optimization_info = {
            'original_length': original_length,
            'optimized_length': len(optimized),
            'estimated_tokens_saved': estimated_tokens_saved,
            'compression_ratio': len(optimized) / original_length if original_length > 0 else 1.0
        }
        
        return optimized, optimization_info
    
    def create_context_window_manager(self, max_tokens: int = 4000):
        """Create context window manager for token budget"""
        return ContextWindowManager(max_tokens)

class ContextWindowManager:
    """Manage context window and token budget"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.current_tokens = 0
        self.context_segments = deque()
        self.priority_weights = {
            'system': 3.0,
            'user': 2.0,
            'assistant': 1.5,
            'context': 1.0
        }
    
    def add_context(self, content: str, segment_type: str = 'context', priority: int = 1):
        """Add context with priority-based management"""
        estimated_tokens = len(content) // 4  # Rough estimation
        
        segment = {
            'content': content,
            'type': segment_type,
            'priority': priority,
            'tokens': estimated_tokens,
            'timestamp': datetime.now()
        }
        
        self.context_segments.append(segment)
        self.current_tokens += estimated_tokens
        
        # Trim context if needed
        self._trim_context_if_needed()
    
    def _trim_context_if_needed(self):
        """Remove low-priority context to stay within token budget"""
        while self.current_tokens > self.max_tokens * 0.8 and self.context_segments:
            # Find lowest priority segment
            min_priority = min(seg['priority'] for seg in self.context_segments)
            
            # Remove oldest segment with minimum priority
            for i, segment in enumerate(self.context_segments):
                if segment['priority'] == min_priority:
                    removed = self.context_segments.popleft()
                    self.current_tokens -= removed['tokens']
                    logger.info(f"Trimmed context: {removed['type']}, saved {removed['tokens']} tokens")
                    break
    
    def get_optimized_context(self) -> str:
        """Get optimized context within token budget"""
        return '\n'.join(seg['content'] for seg in self.context_segments)

class IntelligentCache:
    """Advanced caching system with multiple strategies"""
    
    def __init__(self, max_size_mb: float = 100.0, strategy: CacheStrategy = CacheStrategy.ADAPTIVE):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.strategy = strategy
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order = deque()  # For LRU
        self.access_frequency = defaultdict(int)  # For LFU
        self.current_size_bytes = 0
        self.hit_count = 0
        self.miss_count = 0
        self.cost_savings = 0.0
        
    def _generate_key(self, data: Any) -> str:
        """Generate cache key from data"""
        if isinstance(data, (str, int, float, bool)):
            return hashlib.md5(str(data).encode()).hexdigest()
        elif isinstance(data, dict):
            return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()
        else:
            return hashlib.md5(pickle.dumps(data)).hexdigest()
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _evict_entries(self, required_space: int):
        """Evict entries based on strategy"""
        if self.strategy == CacheStrategy.LRU:
            self._evict_lru(required_space)
        elif self.strategy == CacheStrategy.LFU:
            self._evict_lfu(required_space)
        elif self.strategy == CacheStrategy.TTL:
            self._evict_expired()
        elif self.strategy == CacheStrategy.ADAPTIVE:
            self._evict_adaptive(required_space)
    
    def _evict_lru(self, required_space: int):
        """Evict least recently used entries"""
        space_freed = 0
        while space_freed < required_space and self.access_order:
            oldest_key = self.access_order.popleft()
            if oldest_key in self.cache:
                entry = self.cache.pop(oldest_key)
                space_freed += entry.size_bytes or 0
                self.current_size_bytes -= entry.size_bytes or 0
                logger.debug(f"Evicted LRU entry: {oldest_key}")
    
    def _evict_lfu(self, required_space: int):
        """Evict least frequently used entries"""
        space_freed = 0
        sorted_entries = sorted(self.cache.items(), key=lambda x: x[1].access_count)
        
        for key, entry in sorted_entries:
            if space_freed >= required_space:
                break
            self.cache.pop(key)
            space_freed += entry.size_bytes or 0
            self.current_size_bytes -= entry.size_bytes or 0
            logger.debug(f"Evicted LFU entry: {key}")
    
    def _evict_expired(self):
        """Remove expired entries"""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired]
        for key in expired_keys:
            entry = self.cache.pop(key)
            self.current_size_bytes -= entry.size_bytes or 0
            logger.debug(f"Evicted expired entry: {key}")
    
    def _evict_adaptive(self, required_space: int):
        """Adaptive eviction considering cost and frequency"""
        # First remove expired entries
        self._evict_expired()
        
        if self.current_size_bytes + required_space <= self.max_size_bytes:
            return
        
        # Calculate eviction scores (lower = evict first)
        scored_entries = []
        for key, entry in self.cache.items():
            age_factor = (datetime.now() - entry.last_accessed).total_seconds()
            frequency_factor = 1.0 / (entry.access_count + 1)
            cost_factor = 1.0 / (entry.cost_saved + 1)
            
            score = age_factor * frequency_factor * cost_factor
            scored_entries.append((score, key, entry))
        
        # Sort by score and evict lowest scoring entries
        scored_entries.sort(key=lambda x: x[0])
        space_freed = 0
        
        for score, key, entry in scored_entries:
            if space_freed >= required_space:
                break
            self.cache.pop(key)
            space_freed += entry.size_bytes or 0
            self.current_size_bytes -= entry.size_bytes or 0
            logger.debug(f"Evicted adaptive entry: {key} (score: {score:.3f})")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cached value"""
        if key in self.cache:
            entry = self.cache[key]
            if entry.is_expired:
                self.cache.pop(key)
                self.current_size_bytes -= entry.size_bytes or 0
                self.miss_count += 1
                return default
            
            # Update access metadata
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            self.access_frequency[key] += 1
            
            # Update LRU order
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            
            self.hit_count += 1
            return entry.value
        
        self.miss_count += 1
        return default
    
    def put(self, key: str, value: Any, ttl: Optional[timedelta] = None, cost_saved: float = 0.0):
        """Store value in cache"""
        size = self._calculate_size(value)
        
        # Check if we need to evict
        if self.current_size_bytes + size > self.max_size_bytes:
            self._evict_entries(size)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            size_bytes=size,
            ttl=ttl,
            cost_saved=cost_saved
        )
        
        self.cache[key] = entry
        self.current_size_bytes += size
        self.cost_savings += cost_saved
        
        # Update access tracking
        self.access_order.append(key)
        logger.debug(f"Cached entry: {key} ({size} bytes)")
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_order.clear()
        self.access_frequency.clear()
        self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'cost_savings': self.cost_savings,
            'strategy': self.strategy.value
        }

class BatchProcessor:
    """Intelligent batch processing system"""
    
    def __init__(self, default_batch_size: int = 10, max_wait_time: float = 5.0):
        self.default_batch_size = default_batch_size
        self.max_wait_time = max_wait_time
        self.batch_queues: Dict[str, List[BatchOperation]] = defaultdict(list)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.batch_stats = defaultdict(list)
        
    async def add_to_batch(self, operation_type: str, item: Any, 
                          batch_size: Optional[int] = None, 
                          priority: int = 1,
                          callback: Optional[Callable] = None) -> Any:
        """Add item to batch queue"""
        batch_size = batch_size or self.default_batch_size
        
        batch_op = BatchOperation(
            operation_type=operation_type,
            items=[item],
            batch_size=batch_size,
            priority=priority,
            callback=callback
        )
        
        self.batch_queues[operation_type].append(batch_op)
        
        # Start processing if not already running
        if operation_type not in self.processing_tasks:
            self.processing_tasks[operation_type] = asyncio.create_task(
                self._process_batch_queue(operation_type)
            )
        
        # Return future result
        future = asyncio.Future()
        batch_op.future = future
        return await future
    
    async def _process_batch_queue(self, operation_type: str):
        """Process batch queue for specific operation type"""
        while True:
            queue = self.batch_queues[operation_type]
            if not queue:
                await asyncio.sleep(0.1)
                continue
            
            # Collect items for batch
            batch_items = []
            callbacks = []
            futures = []
            
            # Wait for batch to fill or timeout
            start_time = time.time()
            target_size = queue[0].batch_size if queue else self.default_batch_size
            
            while (len(batch_items) < target_size and 
                   time.time() - start_time < self.max_wait_time and
                   queue):
                
                batch_op = queue.pop(0)
                batch_items.extend(batch_op.items)
                if batch_op.callback:
                    callbacks.append(batch_op.callback)
                if hasattr(batch_op, 'future'):
                    futures.append(batch_op.future)
            
            if batch_items:
                try:
                    # Process batch
                    start_process_time = time.time()
                    results = await self._process_batch(operation_type, batch_items)
                    process_time = time.time() - start_process_time
                    
                    # Record stats
                    self.batch_stats[operation_type].append({
                        'batch_size': len(batch_items),
                        'process_time': process_time,
                        'throughput': len(batch_items) / process_time,
                        'timestamp': datetime.now()
                    })
                    
                    # Execute callbacks and resolve futures
                    for callback in callbacks:
                        if callback:
                            await callback(results)
                    
                    for future in futures:
                        if not future.done():
                            future.set_result(results)
                            
                except Exception as e:
                    logger.error(f"Batch processing error for {operation_type}: {e}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
            
            # Small delay to prevent busy waiting
            await asyncio.sleep(0.01)
    
    async def _process_batch(self, operation_type: str, items: List[Any]) -> List[Any]:
        """Override this method for specific batch processing logic"""
        logger.info(f"Processing batch of {len(items)} items for {operation_type}")
        # Simulate processing
        await asyncio.sleep(0.1)
        return [f"processed_{item}" for item in items]
    
    def get_batch_stats(self, operation_type: Optional[str] = None) -> Dict[str, Any]:
        """Get batch processing statistics"""
        if operation_type:
            stats = self.batch_stats.get(operation_type, [])
        else:
            all_stats = []
            for op_stats in self.batch_stats.values():
                all_stats.extend(op_stats)
            stats = all_stats
        
        if not stats:
            return {}
        
        batch_sizes = [s['batch_size'] for s in stats]
        process_times = [s['process_time'] for s in stats]
        throughputs = [s['throughput'] for s in stats]
        
        return {
            'total_batches': len(stats),
            'avg_batch_size': statistics.mean(batch_sizes),
            'avg_process_time': statistics.mean(process_times),
            'avg_throughput': statistics.mean(throughputs),
            'max_throughput': max(throughputs),
            'total_items_processed': sum(batch_sizes)
        }

class ParallelExecutionManager:
    """Manage parallel execution with resource limits"""
    
    def __init__(self, max_concurrent: int = 10, resource_monitor: bool = True):
        self.max_concurrent = max_concurrent
        self.resource_monitor = resource_monitor
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.resource_usage = deque(maxlen=100)  # Keep last 100 measurements
        self.performance_metrics: List[PerformanceMetric] = []
        
    async def execute_parallel(self, 
                             tasks: List[Callable], 
                             task_names: Optional[List[str]] = None,
                             priority_weights: Optional[List[float]] = None) -> List[Any]:
        """Execute tasks in parallel with resource management"""
        if task_names is None:
            task_names = [f"task_{i}" for i in range(len(tasks))]
        
        if priority_weights is None:
            priority_weights = [1.0] * len(tasks)
        
        # Sort by priority (higher weight = higher priority)
        task_data = list(zip(tasks, task_names, priority_weights))
        task_data.sort(key=lambda x: x[2], reverse=True)
        
        # Create coroutines with semaphore
        coroutines = []
        for task, name, weight in task_data:
            coroutine = self._execute_with_monitoring(task, name, weight)
            coroutines.append(coroutine)
        
        # Execute all tasks
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Task {task_names[i]} failed: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_monitoring(self, task: Callable, name: str, priority: float) -> Any:
        """Execute task with performance monitoring"""
        async with self.semaphore:
            start_time = time.time()
            operation_id = f"{name}_{start_time}"
            
            try:
                # Monitor resource usage if enabled
                if self.resource_monitor:
                    resource_snapshot = self._capture_resource_usage()
                    self.resource_usage.append(resource_snapshot)
                
                # Execute the task
                if asyncio.iscoroutinefunction(task):
                    result = await task()
                else:
                    result = task()
                
                end_time = time.time()
                duration = end_time - start_time
                
                # Create performance metric
                metric = PerformanceMetric(
                    operation_id=operation_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    token_usage=0,  # Would be updated by actual API calls
                    api_cost=0.0,   # Would be updated by actual API calls
                    success=True
                )
                
                self.performance_metrics.append(metric)
                logger.debug(f"Task {name} completed in {duration:.3f}s")
                
                return result
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                # Record failed metric
                metric = PerformanceMetric(
                    operation_id=operation_id,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    token_usage=0,
                    api_cost=0.0,
                    success=False,
                    error_message=str(e)
                )
                
                self.performance_metrics.append(metric)
                logger.error(f"Task {name} failed after {duration:.3f}s: {e}")
                raise
    
    def _capture_resource_usage(self) -> ResourceUsage:
        """Capture current resource usage snapshot"""
        # This would integrate with actual system monitoring
        # For now, return mock data
        return ResourceUsage(
            cpu_percent=50.0,
            memory_mb=512.0,
            active_connections=len(self.active_tasks),
            pending_operations=self.semaphore._waiters.__len__() if hasattr(self.semaphore, '_waiters') else 0,
            cache_size_mb=10.0,
            token_rate_per_minute=100.0,
            cost_per_hour=5.0
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.performance_metrics:
            return {}
        
        successful_metrics = [m for m in self.performance_metrics if m.success]
        failed_metrics = [m for m in self.performance_metrics if not m.success]
        
        durations = [m.duration for m in successful_metrics]
        throughputs = [m.throughput for m in successful_metrics]
        
        return {
            'total_operations': len(self.performance_metrics),
            'successful_operations': len(successful_metrics),
            'failed_operations': len(failed_metrics),
            'success_rate': len(successful_metrics) / len(self.performance_metrics),
            'avg_duration': statistics.mean(durations) if durations else 0,
            'avg_throughput': statistics.mean(throughputs) if throughputs else 0,
            'max_throughput': max(throughputs) if throughputs else 0,
            'total_cost': sum(m.api_cost for m in self.performance_metrics),
            'total_tokens': sum(m.token_usage for m in self.performance_metrics)
        }

class CostTracker:
    """Track and optimize API costs"""
    
    def __init__(self):
        self.cost_history: List[Dict[str, Any]] = []
        self.model_costs = {
            'claude-3-sonnet': {'input': 0.003, 'output': 0.015},  # per 1K tokens
            'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            'gpt-4': {'input': 0.03, 'output': 0.06},
            'gpt-3.5-turbo': {'input': 0.0015, 'output': 0.002}
        }
        self.daily_budgets = {}
        self.monthly_budgets = {}
        
    def track_api_call(self, model: str, input_tokens: int, output_tokens: int, 
                      operation_type: str = 'general') -> float:
        """Track API call and calculate cost"""
        model_pricing = self.model_costs.get(model, {'input': 0.001, 'output': 0.001})
        
        input_cost = (input_tokens / 1000) * model_pricing['input']
        output_cost = (output_tokens / 1000) * model_pricing['output']
        total_cost = input_cost + output_cost
        
        cost_record = {
            'timestamp': datetime.now(),
            'model': model,
            'operation_type': operation_type,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'input_cost': input_cost,
            'output_cost': output_cost,
            'total_cost': total_cost
        }
        
        self.cost_history.append(cost_record)
        logger.debug(f"API call cost: ${total_cost:.4f} ({model})")
        
        return total_cost
    
    def check_budget_limits(self) -> Dict[str, Any]:
        """Check if approaching budget limits"""
        now = datetime.now()
        today = now.date()
        month_start = today.replace(day=1)
        
        # Calculate daily spending
        daily_spending = sum(
            record['total_cost'] for record in self.cost_history
            if record['timestamp'].date() == today
        )
        
        # Calculate monthly spending
        monthly_spending = sum(
            record['total_cost'] for record in self.cost_history
            if record['timestamp'].date() >= month_start
        )
        
        warnings = []
        if hasattr(self, 'daily_budget') and daily_spending > self.daily_budget * 0.8:
            warnings.append(f"Daily spending: ${daily_spending:.2f} (80% of ${self.daily_budget} budget)")
        
        if hasattr(self, 'monthly_budget') and monthly_spending > self.monthly_budget * 0.8:
            warnings.append(f"Monthly spending: ${monthly_spending:.2f} (80% of ${self.monthly_budget} budget)")
        
        return {
            'daily_spending': daily_spending,
            'monthly_spending': monthly_spending,
            'warnings': warnings,
            'within_budget': len(warnings) == 0
        }
    
    def get_cost_optimization_suggestions(self) -> List[str]:
        """Generate cost optimization suggestions"""
        suggestions = []
        
        # Analyze model usage patterns
        model_usage = defaultdict(lambda: {'calls': 0, 'cost': 0.0})
        for record in self.cost_history:
            model = record['model']
            model_usage[model]['calls'] += 1
            model_usage[model]['cost'] += record['total_cost']
        
        # Suggest cheaper models
        if 'gpt-4' in model_usage and model_usage['gpt-4']['calls'] > 10:
            suggestions.append("Consider using GPT-3.5-turbo for simpler tasks to reduce costs by ~95%")
        
        if 'claude-3-sonnet' in model_usage and model_usage['claude-3-sonnet']['calls'] > 10:
            suggestions.append("Consider using Claude-3-haiku for basic tasks to reduce costs by ~90%")
        
        # Analyze token usage patterns
        recent_records = self.cost_history[-100:]  # Last 100 calls
        if recent_records:
            avg_input_tokens = statistics.mean(r['input_tokens'] for r in recent_records)
            avg_output_tokens = statistics.mean(r['output_tokens'] for r in recent_records)
            
            if avg_input_tokens > 2000:
                suggestions.append("High input token usage detected. Consider prompt optimization and context trimming")
            
            if avg_output_tokens > 1000:
                suggestions.append("High output token usage detected. Consider more focused prompts")
        
        # Caching opportunities
        operation_types = defaultdict(int)
        for record in recent_records:
            operation_types[record['operation_type']] += 1
        
        for op_type, count in operation_types.items():
            if count > 5:
                suggestions.append(f"High frequency operation '{op_type}' detected. Implement caching")
        
        return suggestions

class PerformanceTuner:
    """Main performance optimization coordinator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.token_optimizer = TokenOptimizer()
        self.cache = IntelligentCache(
            max_size_mb=self.config.get('cache_size_mb', 100.0),
            strategy=CacheStrategy(self.config.get('cache_strategy', 'adaptive'))
        )
        self.batch_processor = BatchProcessor(
            default_batch_size=self.config.get('batch_size', 10),
            max_wait_time=self.config.get('batch_wait_time', 5.0)
        )
        self.parallel_manager = ParallelExecutionManager(
            max_concurrent=self.config.get('max_concurrent', 10),
            resource_monitor=self.config.get('resource_monitor', True)
        )
        self.cost_tracker = CostTracker()
        
        # Performance history
        self.optimization_history: List[Dict[str, Any]] = []
        self.active_optimizations: Dict[str, Any] = {}
        
        logger.info("Performance Tuner initialized with advanced optimization capabilities")
    
    async def optimize_operation(self, 
                               operation_func: Callable, 
                               operation_type: str,
                               context: Optional[Dict[str, Any]] = None) -> Any:
        """Apply comprehensive optimization to an operation"""
        start_time = time.time()
        context = context or {}
        
        # Check cache first
        cache_key = self._generate_operation_key(operation_func, context)
        cached_result = self.cache.get(cache_key)
        
        if cached_result is not None:
            logger.info(f"Cache hit for operation: {operation_type}")
            return cached_result
        
        # Apply optimizations based on operation type
        optimizations_applied = []
        
        try:
            # Token optimization for text-based operations
            if 'prompt' in context or 'text' in context:
                optimized_context = await self._apply_token_optimization(context)
                context.update(optimized_context)
                optimizations_applied.append('token_optimization')
            
            # Batch processing if applicable
            if context.get('batchable', False):
                result = await self._apply_batch_optimization(operation_func, operation_type, context)
                optimizations_applied.append('batch_processing')
            else:
                # Parallel execution management
                result = await self._apply_parallel_optimization(operation_func, operation_type, context)
                optimizations_applied.append('parallel_execution')
            
            # Cache the result
            cache_ttl = timedelta(hours=self.config.get('cache_ttl_hours', 24))
            estimated_cost_saved = self._estimate_operation_cost(operation_type)
            
            self.cache.put(
                key=cache_key,
                value=result,
                ttl=cache_ttl,
                cost_saved=estimated_cost_saved
            )
            optimizations_applied.append('caching')
            
            # Record optimization metrics
            end_time = time.time()
            optimization_record = {
                'timestamp': datetime.now(),
                'operation_type': operation_type,
                'optimizations_applied': optimizations_applied,
                'execution_time': end_time - start_time,
                'cache_hit': False,
                'estimated_cost_saved': estimated_cost_saved
            }
            
            self.optimization_history.append(optimization_record)
            logger.info(f"Operation optimized: {operation_type} (applied: {optimizations_applied})")
            
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {operation_type}: {e}")
            # Fallback to unoptimized execution
            if asyncio.iscoroutinefunction(operation_func):
                return await operation_func(**context)
            else:
                return operation_func(**context)
    
    async def _apply_token_optimization(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply token optimization to context"""
        optimized_context = context.copy()
        
        if 'prompt' in context:
            optimized_prompt, optimization_info = self.token_optimizer.optimize_prompt(context['prompt'])
            optimized_context['prompt'] = optimized_prompt
            optimized_context['token_optimization_info'] = optimization_info
        
        if 'text' in context:
            optimized_text, optimization_info = self.token_optimizer.optimize_prompt(context['text'])
            optimized_context['text'] = optimized_text
            optimized_context['text_optimization_info'] = optimization_info
        
        return optimized_context
    
    async def _apply_batch_optimization(self, 
                                      operation_func: Callable, 
                                      operation_type: str, 
                                      context: Dict[str, Any]) -> Any:
        """Apply batch processing optimization"""
        batch_size = context.get('batch_size', self.batch_processor.default_batch_size)
        
        # Add to batch queue
        result = await self.batch_processor.add_to_batch(
            operation_type=operation_type,
            item=context,
            batch_size=batch_size
        )
        
        return result
    
    async def _apply_parallel_optimization(self, 
                                         operation_func: Callable, 
                                         operation_type: str, 
                                         context: Dict[str, Any]) -> Any:
        """Apply parallel execution optimization"""
        # Wrap operation in parallel manager
        tasks = [lambda: operation_func(**context)]
        task_names = [operation_type]
        
        results = await self.parallel_manager.execute_parallel(
            tasks=tasks,
            task_names=task_names
        )
        
        return results[0] if results else None
    
    def _generate_operation_key(self, operation_func: Callable, context: Dict[str, Any]) -> str:
        """Generate cache key for operation"""
        key_data = {
            'function_name': operation_func.__name__,
            'context': context
        }
        return self.cache._generate_key(key_data)
    
    def _estimate_operation_cost(self, operation_type: str) -> float:
        """Estimate cost savings from caching this operation"""
        # Default cost estimates based on operation type
        cost_estimates = {
            'llm_inference': 0.01,
            'document_processing': 0.005,
            'api_call': 0.001,
            'data_analysis': 0.002,
            'default': 0.001
        }
        
        return cost_estimates.get(operation_type, cost_estimates['default'])
    
    async def analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends and suggest improvements"""
        if not self.optimization_history:
            return {'status': 'no_data', 'message': 'No optimization data available'}
        
        recent_records = self.optimization_history[-100:]  # Last 100 operations
        
        # Analyze optimization effectiveness
        optimization_effectiveness = defaultdict(list)
        for record in recent_records:
            for opt in record['optimizations_applied']:
                optimization_effectiveness[opt].append(record['execution_time'])
        
        # Calculate performance metrics
        performance_analysis = {}
        for opt_type, times in optimization_effectiveness.items():
            if times:
                performance_analysis[opt_type] = {
                    'avg_execution_time': statistics.mean(times),
                    'min_execution_time': min(times),
                    'max_execution_time': max(times),
                    'usage_count': len(times)
                }
        
        # Cache performance
        cache_stats = self.cache.get_stats()
        
        # Batch performance
        batch_stats = self.batch_processor.get_batch_stats()
        
        # Parallel execution performance
        parallel_stats = self.parallel_manager.get_performance_summary()
        
        # Cost analysis
        cost_analysis = self.cost_tracker.check_budget_limits()
        cost_suggestions = self.cost_tracker.get_cost_optimization_suggestions()
        
        return {
            'timestamp': datetime.now().isoformat(),
            'optimization_effectiveness': performance_analysis,
            'cache_performance': cache_stats,
            'batch_performance': batch_stats,
            'parallel_performance': parallel_stats,
            'cost_analysis': cost_analysis,
            'optimization_suggestions': cost_suggestions,
            'total_operations_optimized': len(self.optimization_history)
        }
    
    async def auto_tune_parameters(self) -> Dict[str, Any]:
        """Automatically tune optimization parameters based on performance"""
        analysis = await self.analyze_performance_trends()
        
        tuning_actions = []
        
        # Auto-tune cache strategy
        cache_hit_rate = analysis.get('cache_performance', {}).get('hit_rate', 0.0)
        if cache_hit_rate < 0.3:  # Low hit rate
            if self.cache.strategy != CacheStrategy.ADAPTIVE:
                self.cache.strategy = CacheStrategy.ADAPTIVE
                tuning_actions.append('Switched to adaptive cache strategy')
        
        # Auto-tune batch sizes
        batch_perf = analysis.get('batch_performance', {})
        if batch_perf.get('avg_throughput', 0) < 10:  # Low throughput
            self.batch_processor.default_batch_size = min(self.batch_processor.default_batch_size + 5, 50)
            tuning_actions.append(f'Increased batch size to {self.batch_processor.default_batch_size}')
        
        # Auto-tune parallel execution
        parallel_perf = analysis.get('parallel_performance', {})
        success_rate = parallel_perf.get('success_rate', 1.0)
        if success_rate < 0.9:  # Low success rate
            self.parallel_manager.max_concurrent = max(self.parallel_manager.max_concurrent - 2, 2)
            tuning_actions.append(f'Reduced max concurrent to {self.parallel_manager.max_concurrent}')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'tuning_actions': tuning_actions,
            'current_parameters': {
                'cache_strategy': self.cache.strategy.value,
                'batch_size': self.batch_processor.default_batch_size,
                'max_concurrent': self.parallel_manager.max_concurrent
            }
        }
    
    async def run_performance_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive performance benchmark"""
        logger.info("Starting performance benchmark suite...")
        
        benchmark_results = {}
        
        # Cache benchmark
        logger.info("Running cache benchmark...")
        cache_start = time.time()
        for i in range(100):
            key = f"benchmark_key_{i}"
            value = f"benchmark_value_{i}" * 100  # Larger values
            self.cache.put(key, value)
            retrieved = self.cache.get(key)
            assert retrieved == value
        cache_time = time.time() - cache_start
        
        benchmark_results['cache'] = {
            'operations': 200,  # 100 puts + 100 gets
            'total_time': cache_time,
            'ops_per_second': 200 / cache_time,
            'cache_stats': self.cache.get_stats()
        }
        
        # Parallel execution benchmark
        logger.info("Running parallel execution benchmark...")
        
        async def dummy_task():
            await asyncio.sleep(0.01)
            return "completed"
        
        parallel_start = time.time()
        tasks = [dummy_task for _ in range(20)]
        results = await self.parallel_manager.execute_parallel(tasks)
        parallel_time = time.time() - parallel_start
        
        benchmark_results['parallel_execution'] = {
            'tasks': 20,
            'total_time': parallel_time,
            'tasks_per_second': 20 / parallel_time,
            'success_count': len([r for r in results if r == "completed"]),
            'performance_summary': self.parallel_manager.get_performance_summary()
        }
        
        # Token optimization benchmark
        logger.info("Running token optimization benchmark...")
        
        test_prompts = [
            "In order to complete this task, I need to process the data due to the fact that it contains important information.",
            "For the purpose of optimization, we should utilize artificial intelligence and machine learning techniques.",
            "At this point in time, the application programming interface needs natural language processing capabilities."
        ]
        
        token_optimization_results = []
        token_start = time.time()
        
        for prompt in test_prompts:
            optimized, info = self.token_optimizer.optimize_prompt(prompt)
            token_optimization_results.append(info)
        
        token_time = time.time() - token_start
        
        avg_compression = statistics.mean(r['compression_ratio'] for r in token_optimization_results)
        total_tokens_saved = sum(r['estimated_tokens_saved'] for r in token_optimization_results)
        
        benchmark_results['token_optimization'] = {
            'prompts_processed': len(test_prompts),
            'total_time': token_time,
            'avg_compression_ratio': avg_compression,
            'total_tokens_saved': total_tokens_saved,
            'optimization_rate': len(test_prompts) / token_time
        }
        
        # Overall system performance
        overall_start_time = min(
            cache_start,
            parallel_start,
            token_start
        )
        overall_time = time.time() - overall_start_time
        
        benchmark_results['overall'] = {
            'total_benchmark_time': overall_time,
            'components_tested': 3,
            'timestamp': datetime.now().isoformat(),
            'system_healthy': all(
                result.get('ops_per_second', 0) > 10 or 
                result.get('tasks_per_second', 0) > 1
                for result in benchmark_results.values()
                if isinstance(result, dict)
            )
        }
        
        logger.info("Performance benchmark completed successfully")
        return benchmark_results
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'cache_stats': self.cache.get_stats(),
            'batch_stats': self.batch_processor.get_batch_stats(),
            'parallel_stats': self.parallel_manager.get_performance_summary(),
            'cost_summary': {
                'total_operations': len(self.optimization_history),
                'total_cost_saved': sum(r.get('estimated_cost_saved', 0) for r in self.optimization_history),
                'optimization_types_used': list(set(
                    opt for record in self.optimization_history
                    for opt in record.get('optimizations_applied', [])
                ))
            },
            'performance_trends': {
                'avg_execution_time': statistics.mean([
                    r['execution_time'] for r in self.optimization_history
                ]) if self.optimization_history else 0,
                'cache_hit_rate_trend': self.cache.get_stats().get('hit_rate', 0),
                'optimization_adoption_rate': len(self.optimization_history) / max(len(self.optimization_history), 1)
            }
        }

async def demo_performance_optimization():
    """Demonstration of performance optimization capabilities"""
    logger.info("=== Performance Optimization Demo ===")
    
    # Initialize performance tuner
    config = {
        'cache_size_mb': 50.0,
        'cache_strategy': 'adaptive',
        'batch_size': 5,
        'batch_wait_time': 2.0,
        'max_concurrent': 5,
        'resource_monitor': True
    }
    
    tuner = PerformanceTuner(config)
    
    # Demo 1: Token Optimization
    logger.info("\n--- Demo 1: Token Optimization ---")
    test_prompt = """
    In order to complete this complex task, I need to process the data due to the fact that 
    it contains important information about artificial intelligence and machine learning 
    techniques that will help with natural language processing applications.
    """
    
    optimized_prompt, optimization_info = tuner.token_optimizer.optimize_prompt(test_prompt)
    logger.info(f"Original length: {optimization_info['original_length']}")
    logger.info(f"Optimized length: {optimization_info['optimized_length']}")
    logger.info(f"Tokens saved: {optimization_info['estimated_tokens_saved']}")
    logger.info(f"Compression ratio: {optimization_info['compression_ratio']:.2f}")
    
    # Demo 2: Intelligent Caching
    logger.info("\n--- Demo 2: Intelligent Caching ---")
    
    async def expensive_operation(data: str) -> str:
        await asyncio.sleep(0.5)  # Simulate expensive operation
        return f"processed_{data}"
    
    # First call (cache miss)
    start = time.time()
    result1 = await tuner.optimize_operation(
        operation_func=expensive_operation,
        operation_type='llm_inference',
        context={'data': 'test_data_1'}
    )
    first_call_time = time.time() - start
    
    # Second call (cache hit)
    start = time.time()
    result2 = await tuner.optimize_operation(
        operation_func=expensive_operation,
        operation_type='llm_inference',
        context={'data': 'test_data_1'}
    )
    second_call_time = time.time() - start
    
    logger.info(f"First call time: {first_call_time:.3f}s")
    logger.info(f"Second call time: {second_call_time:.3f}s")
    logger.info(f"Speedup: {first_call_time / second_call_time:.1f}x")
    
    cache_stats = tuner.cache.get_stats()
    logger.info(f"Cache hit rate: {cache_stats['hit_rate']:.2f}")
    logger.info(f"Cost savings: ${cache_stats['cost_savings']:.4f}")
    
    # Demo 3: Batch Processing
    logger.info("\n--- Demo 3: Batch Processing ---")
    
    async def batch_operation(items: List[str]) -> List[str]:
        await asyncio.sleep(0.1)  # Simulate API call
        return [f"batch_processed_{item}" for item in items]
    
    # Override batch processor method
    tuner.batch_processor._process_batch = lambda op_type, items: batch_operation(items)
    
    # Process multiple items
    batch_tasks = []
    for i in range(10):
        task = tuner.batch_processor.add_to_batch('batch_demo', f'item_{i}', batch_size=3)
        batch_tasks.append(task)
    
    batch_start = time.time()
    batch_results = await asyncio.gather(*batch_tasks)
    batch_time = time.time() - batch_start
    
    logger.info(f"Processed {len(batch_tasks)} items in {batch_time:.3f}s")
    batch_stats = tuner.batch_processor.get_batch_stats('batch_demo')
    if batch_stats:
        logger.info(f"Average batch size: {batch_stats['avg_batch_size']:.1f}")
        logger.info(f"Average throughput: {batch_stats['avg_throughput']:.1f} items/sec")
    
    # Demo 4: Parallel Execution
    logger.info("\n--- Demo 4: Parallel Execution ---")
    
    async def parallel_task(task_id: int) -> str:
        await asyncio.sleep(0.2)  # Simulate work
        return f"task_{task_id}_completed"
    
    parallel_tasks = [lambda i=i: parallel_task(i) for i in range(8)]
    task_names = [f"parallel_task_{i}" for i in range(8)]
    
    parallel_start = time.time()
    parallel_results = await tuner.parallel_manager.execute_parallel(
        tasks=parallel_tasks,
        task_names=task_names
    )
    parallel_time = time.time() - parallel_start
    
    logger.info(f"Executed {len(parallel_tasks)} tasks in {parallel_time:.3f}s")
    parallel_stats = tuner.parallel_manager.get_performance_summary()
    logger.info(f"Success rate: {parallel_stats['success_rate']:.2f}")
    logger.info(f"Average throughput: {parallel_stats['avg_throughput']:.1f} ops/sec")
    
    # Demo 5: Performance Analysis
    logger.info("\n--- Demo 5: Performance Analysis ---")
    
    analysis = await tuner.analyze_performance_trends()
    logger.info("Performance Analysis:")
    logger.info(f"Total operations optimized: {analysis['total_operations_optimized']}")
    logger.info(f"Cache performance: {analysis['cache_performance']}")
    
    if analysis['optimization_suggestions']:
        logger.info("Optimization suggestions:")
        for suggestion in analysis['optimization_suggestions']:
            logger.info(f"  - {suggestion}")
    
    # Demo 6: Auto-tuning
    logger.info("\n--- Demo 6: Auto-tuning ---")
    
    tuning_results = await tuner.auto_tune_parameters()
    logger.info("Auto-tuning results:")
    if tuning_results['tuning_actions']:
        for action in tuning_results['tuning_actions']:
            logger.info(f"  - {action}")
    else:
        logger.info("  - No tuning actions needed, system already optimized")
    
    logger.info(f"Current parameters: {tuning_results['current_parameters']}")
    
    # Demo 7: Benchmark
    logger.info("\n--- Demo 7: Performance Benchmark ---")
    
    benchmark_results = await tuner.run_performance_benchmark()
    logger.info("Benchmark Results:")
    logger.info(f"Cache: {benchmark_results['cache']['ops_per_second']:.1f} ops/sec")
    logger.info(f"Parallel: {benchmark_results['parallel_execution']['tasks_per_second']:.1f} tasks/sec")
    logger.info(f"Token optimization: {benchmark_results['token_optimization']['optimization_rate']:.1f} prompts/sec")
    logger.info(f"System healthy: {benchmark_results['overall']['system_healthy']}")
    
    # Final Report
    logger.info("\n--- Final Optimization Report ---")
    final_report = tuner.get_optimization_report()
    logger.info(f"Total cost saved: ${final_report['cost_summary']['total_cost_saved']:.4f}")
    logger.info(f"Cache hit rate: {final_report['performance_trends']['cache_hit_rate_trend']:.2f}")
    logger.info(f"Average execution time: {final_report['performance_trends']['avg_execution_time']:.3f}s")
    
    logger.info("=== Performance Optimization Demo Complete ===")
    return tuner

if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demo_performance_optimization())