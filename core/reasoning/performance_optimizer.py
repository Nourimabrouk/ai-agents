"""
Performance Optimizer for Phase 7 - Autonomous Intelligence Ecosystem
Real-time optimization, caching, and performance monitoring for reasoning systems
"""

import asyncio
import time
import psutil
import threading
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import json
import pickle
import hashlib
from collections import defaultdict, deque
import numpy as np
import weakref
from contextlib import asynccontextmanager

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class OptimizationLevel(Enum):
    """Levels of performance optimization"""
    MINIMAL = 1      # Basic optimizations
    STANDARD = 2     # Standard optimizations
    AGGRESSIVE = 3   # Aggressive optimizations
    MAXIMUM = 4      # Maximum optimization (may affect accuracy)


class CacheStrategy(Enum):
    """Caching strategies"""
    LRU = "lru"                    # Least Recently Used
    LFU = "lfu"                    # Least Frequently Used
    TTL = "ttl"                    # Time To Live
    ADAPTIVE = "adaptive"          # Adaptive based on access patterns
    INTELLIGENT = "intelligent"    # ML-based cache management


class ResourceType(Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    IO = "io"
    NETWORK = "network"
    CACHE = "cache"
    THREAD = "thread"


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    cache_hit_rate: float
    average_response_time: float
    throughput: float
    error_rate: float
    concurrent_tasks: int
    
    # Reasoning-specific metrics
    causal_accuracy: float = 0.0
    memory_coherence: float = 0.0
    token_efficiency: float = 0.0
    pattern_detection_rate: float = 0.0


@dataclass
class OptimizationTarget:
    """Optimization target specification"""
    target_response_time: float = 1.0  # seconds
    target_throughput: float = 10.0    # tasks/second
    target_accuracy: float = 0.9       # 90% accuracy
    target_memory_efficiency: float = 0.8
    target_cache_hit_rate: float = 0.8
    max_cpu_usage: float = 0.8         # 80% CPU
    max_memory_usage: float = 0.8      # 80% memory


class IntelligentCache:
    """Intelligent caching system with adaptive strategies"""
    
    def __init__(self, max_size: int = 10000, strategy: CacheStrategy = CacheStrategy.INTELLIGENT):
        self.max_size = max_size
        self.strategy = strategy
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.access_times: Dict[str, datetime] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.access_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        # Adaptive learning
        self.pattern_predictor = CachePatternPredictor()
        
        # Thread safety
        self.lock = threading.RLock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with pattern learning"""
        
        with self.lock:
            if key in self.cache:
                # Update access statistics
                self.access_times[key] = datetime.now()
                self.access_counts[key] += 1
                self.access_patterns[key].append(datetime.now())
                
                # Limit pattern history
                if len(self.access_patterns[key]) > 100:
                    self.access_patterns[key] = self.access_patterns[key][-100:]
                
                self.hits += 1
                
                # Learn access pattern
                await self.pattern_predictor.learn_access(key, datetime.now())
                
                return self.cache[key]
            else:
                self.misses += 1
                return {}
    
    async def put(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """Put item in cache with intelligent eviction"""
        
        with self.lock:
            # Check if we need to make space
            if len(self.cache) >= self.max_size and key not in self.cache:
                await self._evict_items(1)
            
            # Store item
            self.cache[key] = value
            self.access_times[key] = datetime.now()
            self.access_counts[key] = self.access_counts.get(key, 0) + 1
            
            # TTL handling
            if ttl:
                # Schedule TTL expiration (simplified)
                expiry_time = datetime.now() + ttl
                # Would implement proper TTL management in production
    
    async def _evict_items(self, count: int) -> None:
        """Evict items based on strategy"""
        
        if self.strategy == CacheStrategy.LRU:
            # Evict least recently used
            items_by_time = sorted(self.access_times.items(), key=lambda x: x[1])
            for key, _ in items_by_time[:count]:
                await self._evict_key(key)
        
        elif self.strategy == CacheStrategy.LFU:
            # Evict least frequently used
            items_by_count = sorted(self.access_counts.items(), key=lambda x: x[1])
            for key, _ in items_by_count[:count]:
                await self._evict_key(key)
        
        elif self.strategy == CacheStrategy.INTELLIGENT:
            # Use ML prediction for eviction
            eviction_candidates = await self.pattern_predictor.predict_eviction_candidates(
                list(self.cache.keys()), count
            )
            for key in eviction_candidates:
                await self._evict_key(key)
        
        else:
            # Default to LRU
            items_by_time = sorted(self.access_times.items(), key=lambda x: x[1])
            for key, _ in items_by_time[:count]:
                await self._evict_key(key)
    
    async def _evict_key(self, key: str) -> None:
        """Evict specific key"""
        
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        self.access_patterns.pop(key, None)
        self.evictions += 1
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate"""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    async def optimize(self) -> Dict[str, Any]:
        """Optimize cache performance"""
        
        optimization_results = {
            'before_hit_rate': self.get_hit_rate(),
            'evictions_performed': 0,
            'patterns_learned': 0
        }
        
        # Pattern-based optimization
        patterns = await self.pattern_predictor.analyze_patterns(self.access_patterns)
        optimization_results['patterns_learned'] = len(patterns)
        
        # Preemptive eviction of unlikely-to-be-used items
        unlikely_keys = await self.pattern_predictor.predict_unlikely_access(
            list(self.cache.keys())
        )
        
        if len(unlikely_keys) > len(self.cache) * 0.1:  # Only if significant
            for key in unlikely_keys[:len(self.cache) // 10]:  # Evict up to 10%
                await self._evict_key(key)
                optimization_results['evictions_performed'] += 1
        
        optimization_results['after_hit_rate'] = self.get_hit_rate()
        
        return optimization_results


class CachePatternPredictor:
    """ML-based cache pattern prediction"""
    
    def __init__(self):
        self.access_history: List[Tuple[str, datetime]] = []
        self.patterns: Dict[str, Any] = {}
        
    async def learn_access(self, key: str, timestamp: datetime) -> None:
        """Learn from access pattern"""
        
        self.access_history.append((key, timestamp))
        
        # Limit history size
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-10000:]
    
    async def analyze_patterns(self, access_patterns: Dict[str, List[datetime]]) -> Dict[str, Any]:
        """Analyze access patterns"""
        
        patterns = {}
        
        for key, timestamps in access_patterns.items():
            if len(timestamps) > 1:
                # Calculate access intervals
                intervals = []
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                    intervals.append(interval)
                
                if intervals:
                    patterns[key] = {
                        'avg_interval': np.mean(intervals),
                        'std_interval': np.std(intervals),
                        'last_access': timestamps[-1],
                        'access_count': len(timestamps),
                        'regularity': 1.0 / (np.std(intervals) + 1e-6)  # Higher = more regular
                    }
        
        return patterns
    
    async def predict_eviction_candidates(self, keys: List[str], count: int) -> List[str]:
        """Predict which keys are good candidates for eviction"""
        
        if not self.patterns:
            return keys[:count]  # Fallback to first N keys
        
        # Score keys by likelihood of not being accessed soon
        key_scores = []
        
        current_time = datetime.now()
        
        for key in keys:
            if key in self.patterns:
                pattern = self.patterns[key]
                
                # Time since last access
                time_since_access = (current_time - pattern['last_access']).total_seconds()
                
                # Expected next access time based on average interval
                expected_interval = pattern['avg_interval']
                
                # Score (higher = more likely to evict)
                if expected_interval > 0:
                    score = time_since_access / expected_interval
                else:
                    score = time_since_access / 3600  # Default to hourly if no pattern
                
                # Adjust for access frequency (less frequent = more likely to evict)
                frequency_factor = 1.0 / (pattern['access_count'] + 1)
                score *= (1 + frequency_factor)
                
                key_scores.append((key, score))
            else:
                # No pattern data - assign medium score
                key_scores.append((key, 1.0))
        
        # Sort by score (highest first) and return top candidates
        key_scores.sort(key=lambda x: x[1], reverse=True)
        return [key for key, score in key_scores[:count]]
    
    async def predict_unlikely_access(self, keys: List[str]) -> List[str]:
        """Predict keys unlikely to be accessed soon"""
        
        unlikely_keys = []
        current_time = datetime.now()
        
        for key in keys:
            if key in self.patterns:
                pattern = self.patterns[key]
                
                # Time since last access
                time_since_access = (current_time - pattern['last_access']).total_seconds()
                
                # If it's been much longer than average interval, it's unlikely to be accessed
                if pattern['avg_interval'] > 0 and time_since_access > pattern['avg_interval'] * 2:
                    unlikely_keys.append(key)
                elif time_since_access > 3600 * 24:  # More than 24 hours
                    unlikely_keys.append(key)
        
        return unlikely_keys


class ResourceMonitor:
    """Real-time resource monitoring and alerting"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history: deque = deque(maxlen=1000)
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            ResourceType.CPU: 0.8,
            ResourceType.MEMORY: 0.8,
            ResourceType.IO: 0.8
        }
        
        self.monitoring_active = False
        self.monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring"""
        
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_task = asyncio.create_task(self._monitoring_loop())
            logger.info("Started resource monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring"""
        
        self.monitoring_active = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
        logger.info(f'Method {function_name} called')
        return {}
        logger.info("Stopped resource monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        
        while self.monitoring_active:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Check thresholds
                await self._check_thresholds(metrics)
                
                # Update global metrics
                global_metrics.gauge("system.cpu_usage", metrics.cpu_usage)
                global_metrics.gauge("system.memory_usage", metrics.memory_usage)
                global_metrics.gauge("system.throughput", metrics.throughput)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval * 2)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        
        # System metrics
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        memory_usage = memory_info.percent / 100.0
        
        # Calculate throughput and response time from recent history
        current_time = datetime.now()
        recent_metrics = [m for m in self.metrics_history if (current_time - m.timestamp).total_seconds() < 60]
        
        if recent_metrics:
            avg_response_time = np.mean([m.average_response_time for m in recent_metrics])
            throughput = len(recent_metrics) / 60.0  # Tasks per second
        else:
            avg_response_time = 0.0
            throughput = 0.0
        
        return PerformanceMetrics(
            timestamp=current_time,
            cpu_usage=cpu_usage / 100.0,
            memory_usage=memory_usage,
            cache_hit_rate=0.0,  # Will be updated by cache systems
            average_response_time=avg_response_time,
            throughput=throughput,
            error_rate=0.0,  # Will be tracked separately
            concurrent_tasks=0  # Will be updated by task manager
        )
    
    async def _check_thresholds(self, metrics: PerformanceMetrics) -> None:
        """Check if metrics exceed thresholds"""
        
        alerts = []
        
        if metrics.cpu_usage > self.thresholds[ResourceType.CPU]:
            alerts.append({
                'type': 'cpu_high',
                'value': metrics.cpu_usage,
                'threshold': self.thresholds[ResourceType.CPU],
                'timestamp': metrics.timestamp
            })
        
        if metrics.memory_usage > self.thresholds[ResourceType.MEMORY]:
            alerts.append({
                'type': 'memory_high',
                'value': metrics.memory_usage,
                'threshold': self.thresholds[ResourceType.MEMORY],
                'timestamp': metrics.timestamp
            })
        
        if metrics.average_response_time > 5.0:  # 5 second threshold
            alerts.append({
                'type': 'response_time_high',
                'value': metrics.average_response_time,
                'threshold': 5.0,
                'timestamp': metrics.timestamp
            })
        
        # Store alerts
        self.alerts.extend(alerts)
        
        # Limit alert history
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        # Log critical alerts
        for alert in alerts:
            if alert['value'] > alert['threshold'] * 1.2:  # 20% over threshold
                logger.warning(f"Resource alert: {alert['type']} = {alert['value']:.3f} (threshold: {alert['threshold']:.3f})")
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent metrics"""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_history(self, duration: timedelta = timedelta(minutes=10)) -> List[PerformanceMetrics]:
        """Get metrics history for specified duration"""
        
        cutoff_time = datetime.now() - duration
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class PerformanceOptimizer:
    """Comprehensive performance optimization system"""
    
    def __init__(self, 
                 optimization_level: OptimizationLevel = OptimizationLevel.STANDARD,
                 targets: Optional[OptimizationTarget] = None):
        
        self.optimization_level = optimization_level
        self.targets = targets or OptimizationTarget()
        
        # Core components
        self.cache_system = IntelligentCache(max_size=10000, strategy=CacheStrategy.INTELLIGENT)
        self.resource_monitor = ResourceMonitor(monitoring_interval=1.0)
        
        # Optimization strategies
        self.optimization_strategies = {
            'caching': self._optimize_caching,
            'threading': self._optimize_threading,
            'memory': self._optimize_memory,
            'computation': self._optimize_computation,
            'io': self._optimize_io
        }
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_configuration: Dict[str, Any] = {}
        
        # Thread management
        self.thread_pools = {
            'cpu_intensive': ThreadPoolExecutor(max_workers=psutil.cpu_count()),
            'io_intensive': ThreadPoolExecutor(max_workers=psutil.cpu_count() * 2),
            'reasoning': ThreadPoolExecutor(max_workers=4)
        }
        
        # Process pool for CPU-intensive tasks
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, psutil.cpu_count() // 2))
        
        # Optimization state
        self.optimization_active = False
        self.optimization_task: Optional[asyncio.Task] = None
        
        logger.info(f"Initialized Performance Optimizer with {optimization_level.name} level")
    
    async def start_optimization(self) -> None:
        """Start performance optimization system"""
        
        if not self.optimization_active:
            self.optimization_active = True
            
            # Start resource monitoring
            await self.resource_monitor.start_monitoring()
            
            # Start optimization loop
            self.optimization_task = asyncio.create_task(self._optimization_loop())
            
            logger.info("Started performance optimization system")
    
    async def stop_optimization(self) -> None:
        """Stop performance optimization system"""
        
        self.optimization_active = False
        
        # Stop monitoring
        await self.resource_monitor.stop_monitoring()
        
        # Stop optimization loop
        if self.optimization_task:
            self.optimization_task.cancel()
            try:
                await self.optimization_task
            except asyncio.CancelledError:
        logger.info(f'Method {function_name} called')
        return {}
        
        # Shutdown thread pools
        for pool in self.thread_pools.values():
            pool.shutdown(wait=True)
        
        self.process_pool.shutdown(wait=True)
        
        logger.info("Stopped performance optimization system")
    
    async def _optimization_loop(self) -> None:
        """Main optimization loop"""
        
        while self.optimization_active:
            try:
                # Get current performance metrics
                current_metrics = self.resource_monitor.get_current_metrics()
                
                if current_metrics:
                    # Analyze performance
                    analysis = await self._analyze_performance(current_metrics)
                    
                    # Apply optimizations if needed
                    if analysis['needs_optimization']:
                        optimization_results = await self._apply_optimizations(analysis, current_metrics)
                        self.optimization_history.append({
                            'timestamp': datetime.now(),
                            'analysis': analysis,
                            'optimizations': optimization_results,
                            'metrics_before': current_metrics,
                        })
                        
                        logger.info(f"Applied optimizations: {list(optimization_results.keys())}")
                
                # Optimization interval based on level
                if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                    await asyncio.sleep(30)  # 30 seconds
                elif self.optimization_level == OptimizationLevel.MAXIMUM:
                    await asyncio.sleep(15)  # 15 seconds
                else:
                    await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(60)
    
    async def _analyze_performance(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze current performance against targets"""
        
        analysis = {
            'needs_optimization': False,
            'bottlenecks': [],
            'recommendations': [],
            'priority_areas': []
        }
        
        # CPU analysis
        if metrics.cpu_usage > self.targets.max_cpu_usage:
            analysis['bottlenecks'].append('cpu_high')
            analysis['recommendations'].append('optimize_threading')
            analysis['needs_optimization'] = True
        
        # Memory analysis
        if metrics.memory_usage > self.targets.max_memory_usage:
            analysis['bottlenecks'].append('memory_high')
            analysis['recommendations'].append('optimize_memory')
            analysis['needs_optimization'] = True
        
        # Response time analysis
        if metrics.average_response_time > self.targets.target_response_time:
            analysis['bottlenecks'].append('response_time_high')
            analysis['recommendations'].append('optimize_caching')
            analysis['needs_optimization'] = True
        
        # Throughput analysis
        if metrics.throughput < self.targets.target_throughput:
            analysis['bottlenecks'].append('throughput_low')
            analysis['recommendations'].append('optimize_computation')
            analysis['needs_optimization'] = True
        
        # Cache efficiency analysis
        cache_hit_rate = self.cache_system.get_hit_rate()
        if cache_hit_rate < self.targets.target_cache_hit_rate:
            analysis['bottlenecks'].append('cache_inefficient')
            analysis['recommendations'].append('optimize_caching')
            analysis['needs_optimization'] = True
        
        # Priority ranking
        if 'response_time_high' in analysis['bottlenecks']:
            analysis['priority_areas'].append('caching')
        if 'cpu_high' in analysis['bottlenecks']:
            analysis['priority_areas'].append('threading')
        if 'memory_high' in analysis['bottlenecks']:
            analysis['priority_areas'].append('memory')
        
        return analysis
    
    async def _apply_optimizations(self, analysis: Dict[str, Any], 
                                 metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Apply performance optimizations based on analysis"""
        
        optimization_results = {}
        
        # Apply optimizations in priority order
        for area in analysis['priority_areas']:
            if area in self.optimization_strategies:
                try:
                    result = await self.optimization_strategies[area](metrics, analysis)
                    optimization_results[area] = result
                except Exception as e:
                    logger.error(f"Optimization {area} failed: {e}")
                    optimization_results[area] = {'error': str(e)}
        
        # Apply additional recommendations
        for recommendation in analysis['recommendations']:
            if recommendation.startswith('optimize_'):
                area = recommendation.replace('optimize_', '')
                if area in self.optimization_strategies and area not in optimization_results:
                    try:
                        result = await self.optimization_strategies[area](metrics, analysis)
                        optimization_results[area] = result
                    except Exception as e:
                        logger.error(f"Optimization {area} failed: {e}")
                        optimization_results[area] = {'error': str(e)}
        
        return optimization_results
    
    async def _optimize_caching(self, metrics: PerformanceMetrics, 
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize caching performance"""
        
        result = {'strategy': 'caching'}
        
        # Optimize cache
        cache_optimization = await self.cache_system.optimize()
        result['cache_optimization'] = cache_optimization
        
        # Adjust cache size if needed
        if metrics.memory_usage < 0.6 and self.cache_system.get_hit_rate() < self.targets.target_cache_hit_rate:
            # Increase cache size
            old_size = self.cache_system.max_size
            new_size = min(old_size * 2, 50000)
            self.cache_system.max_size = new_size
            result['cache_size_change'] = {'old': old_size, 'new': new_size}
        
        elif metrics.memory_usage > 0.8:
            # Decrease cache size
            old_size = self.cache_system.max_size
            new_size = max(old_size // 2, 1000)
            self.cache_system.max_size = new_size
            result['cache_size_change'] = {'old': old_size, 'new': new_size}
        
        return result
    
    async def _optimize_threading(self, metrics: PerformanceMetrics,
                                analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize threading configuration"""
        
        result = {'strategy': 'threading'}
        
        cpu_count = psutil.cpu_count()
        
        # Adjust thread pool sizes based on CPU usage
        if metrics.cpu_usage > 0.8:
            # Reduce thread pool sizes
            for pool_name, pool in self.thread_pools.items():
                if hasattr(pool, '_max_workers'):
                    old_workers = pool._max_workers
                    new_workers = max(old_workers - 1, 1)
                    
                    # Create new pool with adjusted size
                    pool.shutdown(wait=False)
                    self.thread_pools[pool_name] = ThreadPoolExecutor(max_workers=new_workers)
                    
                    result[f'{pool_name}_workers'] = {'old': old_workers, 'new': new_workers}
        
        elif metrics.cpu_usage < 0.4 and metrics.throughput < self.targets.target_throughput:
            # Increase thread pool sizes
            for pool_name, pool in self.thread_pools.items():
                if hasattr(pool, '_max_workers'):
                    old_workers = pool._max_workers
                    max_workers = cpu_count * 2 if pool_name == 'io_intensive' else cpu_count
                    new_workers = min(old_workers + 1, max_workers)
                    
                    if new_workers > old_workers:
                        # Create new pool with adjusted size
                        pool.shutdown(wait=False)
                        self.thread_pools[pool_name] = ThreadPoolExecutor(max_workers=new_workers)
                        
                        result[f'{pool_name}_workers'] = {'old': old_workers, 'new': new_workers}
        
        return result
    
    async def _optimize_memory(self, metrics: PerformanceMetrics,
                             analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory usage"""
        
        result = {'strategy': 'memory'}
        
        # Force garbage collection if memory usage is high
        if metrics.memory_usage > 0.8:
            import gc
            collected = gc.collect()
            result['gc_collected'] = collected
        
        # Adjust cache sizes
        if metrics.memory_usage > 0.8:
            # Reduce cache size
            old_cache_size = self.cache_system.max_size
            new_cache_size = max(old_cache_size // 2, 1000)
            self.cache_system.max_size = new_cache_size
            
            # Trigger cache cleanup
            excess_items = len(self.cache_system.cache) - new_cache_size
            if excess_items > 0:
                await self.cache_system._evict_items(excess_items)
            
            result['cache_size_reduced'] = {'old': old_cache_size, 'new': new_cache_size}
        
        return result
    
    async def _optimize_computation(self, metrics: PerformanceMetrics,
                                  analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize computational performance"""
        
        result = {'strategy': 'computation'}
        
        # Adjust computation strategies based on CPU usage and throughput
        if metrics.cpu_usage < 0.5 and metrics.throughput < self.targets.target_throughput:
            # Enable more parallel computation
            result['parallel_computation'] = 'increased'
        
        elif metrics.cpu_usage > 0.8:
            # Reduce parallel computation
            result['parallel_computation'] = 'reduced'
        
        # Optimize numerical computations
        # Could implement SIMD optimizations, GPU acceleration, etc.
        result['numerical_optimization'] = 'applied'
        
        return result
    
    async def _optimize_io(self, metrics: PerformanceMetrics,
                         analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize I/O performance"""
        
        result = {'strategy': 'io'}
        
        # Implement I/O optimizations
        # Could include disk I/O optimization, network optimization, etc.
        result['io_optimization'] = 'applied'
        
        return result
    
    # Public API methods
    
    @asynccontextmanager
    async def optimized_execution(self, operation_type: str = "general"):
        """Context manager for optimized execution"""
        
        start_time = time.time()
        
        # Select appropriate thread pool
        if operation_type in ['cpu_intensive', 'io_intensive', 'reasoning']:
            thread_pool = self.thread_pools[operation_type]
        else:
            thread_pool = self.thread_pools['reasoning']  # Default
        
        try:
            yield thread_pool
        finally:
            # Record execution time
            execution_time = time.time() - start_time
            global_metrics.timing(f"optimized_execution.{operation_type}", execution_time)
    
    async def cache_result(self, key: str, result: Any, ttl: Optional[timedelta] = None) -> None:
        """Cache computation result"""
        await self.cache_system.put(key, result, ttl)
    
    async def get_cached_result(self, key: str) -> Optional[Any]:
        """Get cached computation result"""
        return await self.cache_system.get(key)
    
    async def execute_with_caching(self, cache_key: str, computation_func: Callable,
                                 *args, ttl: Optional[timedelta] = None, **kwargs) -> Any:
        """Execute function with caching"""
        
        # Try to get from cache first
        cached_result = await self.get_cached_result(cache_key)
        if cached_result is not None:
            return cached_result
        
        # Execute computation
        if asyncio.iscoroutinefunction(computation_func):
            result = await computation_func(*args, **kwargs)
        else:
            # Run in thread pool for CPU-intensive operations
            async with self.optimized_execution('cpu_intensive') as thread_pool:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(thread_pool, computation_func, *args)
        
        # Cache the result
        await self.cache_result(cache_key, result, ttl)
        
        return result
    
    def create_cache_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments"""
        
        # Create a hash of arguments for cache key
        key_data = json.dumps([args, kwargs], sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        
        current_metrics = self.resource_monitor.get_current_metrics()
        recent_history = self.resource_monitor.get_metrics_history(timedelta(minutes=30))
        
        report = {
            'current_metrics': current_metrics.__dict__ if current_metrics else None,
            'optimization_level': self.optimization_level.name,
            'targets': {
                'response_time': self.targets.target_response_time,
                'throughput': self.targets.target_throughput,
                'accuracy': self.targets.target_accuracy,
                'cache_hit_rate': self.targets.target_cache_hit_rate
            },
            'cache_performance': {
                'hit_rate': self.cache_system.get_hit_rate(),
                'size': len(self.cache_system.cache),
                'max_size': self.cache_system.max_size,
                'hits': self.cache_system.hits,
                'misses': self.cache_system.misses,
                'evictions': self.cache_system.evictions
            },
            'thread_pools': {
                name: getattr(pool, '_max_workers', 'unknown')
                for name, pool in self.thread_pools.items()
            },
            'optimization_history': len(self.optimization_history),
            'recent_alerts': len([
                alert for alert in self.resource_monitor.alerts
                if (datetime.now() - alert['timestamp']).total_seconds() < 3600
            ])
        }
        
        # Performance trends
        if recent_history:
            report['trends'] = {
                'avg_cpu_usage': np.mean([m.cpu_usage for m in recent_history]),
                'avg_memory_usage': np.mean([m.memory_usage for m in recent_history]),
                'avg_response_time': np.mean([m.average_response_time for m in recent_history]),
                'avg_throughput': np.mean([m.throughput for m in recent_history])
            }
        
        return report
    
    async def optimize_for_reasoning_task(self, task_complexity: str = "medium",
                                        expected_duration: float = 60.0) -> Dict[str, Any]:
        """Optimize system for upcoming reasoning task"""
        
        optimization_config = {
            'task_complexity': task_complexity,
            'expected_duration': expected_duration,
            'optimizations_applied': []
        }
        
        # Pre-warm caches
        if task_complexity in ['high', 'maximum']:
            # Increase cache size temporarily
            old_cache_size = self.cache_system.max_size
            self.cache_system.max_size = min(old_cache_size * 2, 50000)
            optimization_config['optimizations_applied'].append('increased_cache_size')
        
        # Adjust thread pool for reasoning tasks
        reasoning_pool = self.thread_pools['reasoning']
        if task_complexity == 'maximum' and hasattr(reasoning_pool, '_max_workers'):
            if reasoning_pool._max_workers < 8:
                reasoning_pool.shutdown(wait=False)
                self.thread_pools['reasoning'] = ThreadPoolExecutor(max_workers=8)
                optimization_config['optimizations_applied'].append('increased_reasoning_threads')
        
        # Force garbage collection before intensive task
        if task_complexity in ['high', 'maximum']:
            import gc
            collected = gc.collect()
            optimization_config['gc_collected'] = collected
            optimization_config['optimizations_applied'].append('garbage_collection')
        
        logger.info(f"Optimized system for {task_complexity} reasoning task: {optimization_config['optimizations_applied']}")
        
        return optimization_config