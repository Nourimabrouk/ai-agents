"""
Memory Performance Profiler for Phase 7 - Advanced Memory Analysis
Detects memory leaks, tracks object allocation, and optimizes memory usage
"""

import gc
import sys
import psutil
import tracemalloc
import weakref
import threading
import time
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from functools import wraps
import logging
from contextlib import contextmanager
import objgraph

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Memory usage snapshot with detailed metrics"""
    timestamp: datetime
    total_memory: int  # bytes
    available_memory: int
    memory_percent: float
    process_memory: int  # Process RSS memory
    tracemalloc_current: int
    tracemalloc_peak: int
    object_counts: Dict[str, int] = field(default_factory=dict)
    largest_objects: List[Dict[str, Any]] = field(default_factory=list)
    gc_stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_memory_mb': self.total_memory // (1024 * 1024),
            'available_memory_mb': self.available_memory // (1024 * 1024),
            'memory_percent': self.memory_percent,
            'process_memory_mb': self.process_memory // (1024 * 1024),
            'tracemalloc_current_mb': self.tracemalloc_current // (1024 * 1024),
            'tracemalloc_peak_mb': self.tracemalloc_peak // (1024 * 1024),
            'object_counts': self.object_counts,
            'largest_objects': self.largest_objects,
            'gc_stats': self.gc_stats
        }


@dataclass
class MemoryLeak:
    """Detected memory leak information"""
    object_type: str
    growth_rate: float  # objects per second
    total_objects: int
    memory_usage_mb: float
    first_detected: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    stack_trace: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'object_type': self.object_type,
            'growth_rate': f"{self.growth_rate:.2f} objects/sec",
            'total_objects': self.total_objects,
            'memory_usage_mb': f"{self.memory_usage_mb:.2f} MB",
            'first_detected': self.first_detected.isoformat(),
            'severity': self.severity,
            'stack_trace': self.stack_trace[:10]  # Limit to top 10 frames
        }


class MemoryProfiler:
    """
    Advanced memory profiler for detecting leaks and optimizing usage
    Designed for Phase 7 high-performance requirements
    """
    
    def __init__(self, 
                 tracking_enabled: bool = True,
                 snapshot_interval: float = 60.0,  # seconds
                 leak_detection_threshold: float = 100.0,  # objects per minute
                 max_snapshots: int = 1000):
        
        self.tracking_enabled = tracking_enabled
        self.snapshot_interval = snapshot_interval
        self.leak_detection_threshold = leak_detection_threshold / 60.0  # Convert to per second
        self.max_snapshots = max_snapshots
        
        # Memory tracking data
        self.snapshots: deque = deque(maxlen=max_snapshots)
        self.baseline_snapshot: Optional[MemorySnapshot] = None
        self.detected_leaks: List[MemoryLeak] = []
        
        # Object tracking
        self.object_trackers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.allocation_trackers: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        
        # Monitoring thread
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Tracemalloc initialization
        if tracking_enabled:
            if not tracemalloc.is_tracing():
                tracemalloc.start(25)  # Keep 25 frames for detailed traces
                logger.info("Started tracemalloc with 25 frame depth")
        
        logger.info(f"MemoryProfiler initialized with {snapshot_interval}s intervals")
    
    def start_monitoring(self):
        """Start continuous memory monitoring"""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return {}
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self.monitor_thread.start()
        
        # Take baseline snapshot
        self.baseline_snapshot = self.take_snapshot()
        
        logger.info("Started continuous memory monitoring")
    
    def stop_monitoring(self):
        """Stop continuous memory monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped memory monitoring")
    
    def _monitor_memory(self):
        """Background memory monitoring loop"""
        while self.monitoring_active:
            try:
                snapshot = self.take_snapshot()
                self.snapshots.append(snapshot)
                
                # Detect memory leaks
                self._detect_memory_leaks()
                
                # Trigger GC if memory usage is high
                if snapshot.memory_percent > 85.0:
                    logger.warning(f"High memory usage: {snapshot.memory_percent:.1f}%")
                    self._trigger_gc()
                
                time.sleep(self.snapshot_interval)
                
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.snapshot_interval)
    
    def take_snapshot(self) -> MemorySnapshot:
        """Take a comprehensive memory snapshot"""
        # System memory info
        memory_info = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info().rss
        
        # Tracemalloc info
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
        else:
            current = peak = 0
        
        # Object counts
        object_counts = self._get_object_counts()
        
        # Largest objects
        largest_objects = self._get_largest_objects()
        
        # Garbage collection stats
        gc_stats = self._get_gc_stats()
        
        snapshot = MemorySnapshot(
            timestamp=datetime.now(),
            total_memory=memory_info.total,
            available_memory=memory_info.available,
            memory_percent=memory_info.percent,
            process_memory=process_memory,
            tracemalloc_current=current,
            tracemalloc_peak=peak,
            object_counts=object_counts,
            largest_objects=largest_objects,
            gc_stats=gc_stats
        )
        
        return snapshot
    
    def _get_object_counts(self) -> Dict[str, int]:
        """Get counts of different object types"""
        try:
            # Use objgraph for detailed object tracking
            most_common = objgraph.most_common_types(limit=20)
            return dict(most_common)
        except Exception as e:
            logger.debug(f"Could not get object counts with objgraph: {e}")
            
            # Fallback to basic gc tracking
            gc.collect()  # Ensure accurate counts
            objects = gc.get_objects()
            
            counts = defaultdict(int)
            for obj in objects:
                obj_type = type(obj).__name__
                counts[obj_type] += 1
            
            # Return top 20 most common types
            return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    def _get_largest_objects(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Identify largest objects in memory"""
        if not tracemalloc.is_tracing():
            return []
        
        try:
            # Get top memory allocations
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:limit]
            
            largest = []
            for stat in top_stats:
                largest.append({
                    'filename': stat.traceback.format()[-1] if stat.traceback else 'unknown',
                    'size_mb': stat.size / (1024 * 1024),
                    'count': stat.count
                })
            
            return largest
            
        except Exception as e:
            logger.debug(f"Could not get largest objects: {e}")
            return []
    
    def _get_gc_stats(self) -> Dict[str, int]:
        """Get garbage collection statistics"""
        try:
            stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in stats)
            total_collected = sum(stat['collected'] for stat in stats)
            total_uncollectable = sum(stat['uncollectable'] for stat in stats)
            
            return {
                'total_collections': total_collections,
                'total_collected': total_collected,
                'total_uncollectable': total_uncollectable,
                'current_objects': len(gc.get_objects())
            }
        except Exception:
            return {}
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks by analyzing growth patterns"""
        if len(self.snapshots) < 3:
            return  # Need at least 3 snapshots for trend analysis
        
        recent_snapshots = list(self.snapshots)[-10:]  # Analyze last 10 snapshots
        
        # Track object count changes
        object_growth = defaultdict(list)
        
        for i in range(1, len(recent_snapshots)):
            prev_snapshot = recent_snapshots[i-1]
            current_snapshot = recent_snapshots[i]
            
            time_diff = (current_snapshot.timestamp - prev_snapshot.timestamp).total_seconds()
            
            for obj_type, current_count in current_snapshot.object_counts.items():
                prev_count = prev_snapshot.object_counts.get(obj_type, 0)
                growth = (current_count - prev_count) / max(1, time_diff)  # Growth per second
                object_growth[obj_type].append(growth)
        
        # Detect consistent growth patterns
        current_time = datetime.now()
        
        for obj_type, growth_rates in object_growth.items():
            if len(growth_rates) < 3:
                continue
            
            avg_growth = sum(growth_rates) / len(growth_rates)
            
            # Check if growth exceeds threshold and is consistent
            if avg_growth > self.leak_detection_threshold:
                # Check if this leak is already detected
                existing_leak = next(
                    (leak for leak in self.detected_leaks if leak.object_type == obj_type),
                    None
                )
                
                if not existing_leak:
                    # Determine severity
                    if avg_growth > self.leak_detection_threshold * 10:
                        severity = "critical"
                    elif avg_growth > self.leak_detection_threshold * 5:
                        severity = "high"
                    elif avg_growth > self.leak_detection_threshold * 2:
                        severity = "medium"
                    else:
                        severity = "low"
                    
                    # Get current object count and estimated memory usage
                    latest_snapshot = recent_snapshots[-1]
                    current_count = latest_snapshot.object_counts.get(obj_type, 0)
                    estimated_memory_mb = self._estimate_object_memory_usage(obj_type, current_count)
                    
                    leak = MemoryLeak(
                        object_type=obj_type,
                        growth_rate=avg_growth,
                        total_objects=current_count,
                        memory_usage_mb=estimated_memory_mb,
                        first_detected=current_time,
                        severity=severity,
                        stack_trace=self._get_allocation_stack_trace(obj_type)
                    )
                    
                    self.detected_leaks.append(leak)
                    logger.warning(f"Memory leak detected: {obj_type} growing at {avg_growth:.2f} objects/sec")
    
    def _estimate_object_memory_usage(self, obj_type: str, count: int) -> float:
        """Estimate memory usage for object type"""
        # Rough estimates for common Python object types (in bytes)
        size_estimates = {
            'dict': 240,
            'list': 64,
            'tuple': 48,
            'str': 50,
            'int': 28,
            'float': 24,
            'set': 224,
            'function': 80,
            'type': 400,
            'module': 400,
            'frame': 400
        }
        
        estimated_size = size_estimates.get(obj_type, 100)  # Default 100 bytes
        total_bytes = estimated_size * count
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def _get_allocation_stack_trace(self, obj_type: str) -> List[str]:
        """Get stack trace for object allocations"""
        if not tracemalloc.is_tracing():
            return []
        
        try:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('traceback')
            
            # Find allocations that might be related to this object type
            for stat in top_stats[:50]:
                if stat.traceback:
                    frames = stat.traceback.format()
                    # Simple heuristic to match object type
                    for frame in frames:
                        if obj_type.lower() in frame.lower():
                            return frames
            
            return []
            
        except Exception as e:
            logger.debug(f"Could not get stack trace: {e}")
            return []
    
    def _trigger_gc(self):
        """Trigger garbage collection and log results"""
        before_objects = len(gc.get_objects())
        before_memory = psutil.Process().memory_info().rss
        
        collected = gc.collect()
        
        after_objects = len(gc.get_objects())
        after_memory = psutil.Process().memory_info().rss
        
        objects_freed = before_objects - after_objects
        memory_freed = before_memory - after_memory
        
        logger.info(f"GC collected {collected} objects, freed {objects_freed} objects, "  
                   f"reclaimed {memory_freed / (1024*1024):.2f} MB")
    
    @contextmanager
    def profile_memory(self, operation_name: str):
        """Context manager for profiling memory usage of operations"""
        if not self.tracking_enabled:
            yield
            return {}
        
        # Take before snapshot
        before_snapshot = self.take_snapshot()
        
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            # Take after snapshot
            end_time = time.perf_counter()
            after_snapshot = self.take_snapshot()
            
            # Calculate differences
            memory_diff = after_snapshot.process_memory - before_snapshot.process_memory
            tracemalloc_diff = after_snapshot.tracemalloc_current - before_snapshot.tracemalloc_current
            execution_time = end_time - start_time
            
            # Log results
            logger.info(f"Memory profile for '{operation_name}': "
                       f"Process memory: {memory_diff / (1024*1024):+.2f} MB, "
                       f"Tracemalloc: {tracemalloc_diff / (1024*1024):+.2f} MB, "
                       f"Time: {execution_time:.4f}s")
    
    def profile_function(self, func: Callable, profile_name: Optional[str] = None) -> Callable:
        """Decorator for profiling function memory usage"""
        if profile_name is None:
            profile_name = func.__name__
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.profile_memory(profile_name):
                return func(*args, **kwargs)
        
        return wrapper
    
    def analyze_memory_growth(self, time_window: timedelta = timedelta(minutes=30)) -> Dict[str, Any]:
        """Analyze memory growth over specified time window"""
        if len(self.snapshots) < 2:
            return {'error': 'Not enough snapshots for analysis'}
        
        current_time = datetime.now()
        cutoff_time = current_time - time_window
        
        # Get snapshots within time window
        relevant_snapshots = [
            snapshot for snapshot in self.snapshots
            if snapshot.timestamp >= cutoff_time
        ]
        
        if len(relevant_snapshots) < 2:
            return {'error': 'Not enough recent snapshots for analysis'}
        
        first_snapshot = relevant_snapshots[0]
        last_snapshot = relevant_snapshots[-1]
        
        time_diff = (last_snapshot.timestamp - first_snapshot.timestamp).total_seconds()
        
        # Calculate growth rates
        memory_growth = last_snapshot.process_memory - first_snapshot.process_memory
        memory_growth_rate = memory_growth / time_diff  # bytes per second
        
        tracemalloc_growth = last_snapshot.tracemalloc_current - first_snapshot.tracemalloc_current
        tracemalloc_growth_rate = tracemalloc_growth / time_diff
        
        # Object count changes
        object_changes = {}
        for obj_type, last_count in last_snapshot.object_counts.items():
            first_count = first_snapshot.object_counts.get(obj_type, 0)
            change = last_count - first_count
            change_rate = change / time_diff
            
            if abs(change) > 10:  # Only report significant changes
                object_changes[obj_type] = {
                    'total_change': change,
                    'change_rate': f"{change_rate:.2f} objects/sec"
                }
        
        return {
            'time_window_minutes': time_window.total_seconds() / 60,
            'snapshots_analyzed': len(relevant_snapshots),
            'memory_growth_mb': memory_growth / (1024 * 1024),
            'memory_growth_rate_mb_per_sec': memory_growth_rate / (1024 * 1024),
            'tracemalloc_growth_mb': tracemalloc_growth / (1024 * 1024),
            'tracemalloc_growth_rate_mb_per_sec': tracemalloc_growth_rate / (1024 * 1024),
            'object_count_changes': object_changes,
            'detected_leaks_count': len(self.detected_leaks),
            'current_memory_usage_mb': last_snapshot.process_memory / (1024 * 1024),
            'memory_usage_percent': last_snapshot.memory_percent
        }
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Generate comprehensive memory usage report"""
        if not self.snapshots:
            return {'error': 'No memory snapshots available'}
        
        latest_snapshot = self.snapshots[-1]
        
        # Calculate trends if we have baseline
        trends = {}
        if self.baseline_snapshot:
            time_diff = (latest_snapshot.timestamp - self.baseline_snapshot.timestamp).total_seconds()
            memory_change = latest_snapshot.process_memory - self.baseline_snapshot.process_memory
            memory_trend = memory_change / max(1, time_diff)  # bytes per second
            
            trends = {
                'memory_change_mb': memory_change / (1024 * 1024),
                'memory_trend_mb_per_sec': memory_trend / (1024 * 1024),
                'monitoring_duration_minutes': time_diff / 60
            }
        
        # Memory leak summary
        leak_summary = {
            'total_leaks': len(self.detected_leaks),
            'critical_leaks': len([leak for leak in self.detected_leaks if leak.severity == 'critical']),
            'high_severity_leaks': len([leak for leak in self.detected_leaks if leak.severity == 'high']),
            'leaks_by_type': [leak.to_dict() for leak in self.detected_leaks[-10:]]  # Last 10 leaks
        }
        
        return {
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'current_memory_usage': latest_snapshot.to_dict(),
            'trends': trends,
            'memory_leaks': leak_summary,
            'snapshots_collected': len(self.snapshots),
            'monitoring_active': self.monitoring_active,
            'recommendations': self._get_memory_recommendations(latest_snapshot)
        }
    
    def _get_memory_recommendations(self, snapshot: MemorySnapshot) -> List[str]:
        """Generate memory optimization recommendations"""
        recommendations = []
        
        if snapshot.memory_percent > 90:
            recommendations.append("Critical: System memory usage above 90%. Consider freeing memory or scaling resources.")
        elif snapshot.memory_percent > 75:
            recommendations.append("Warning: High memory usage detected. Monitor for potential leaks.")
        
        if len(self.detected_leaks) > 0:
            critical_leaks = [leak for leak in self.detected_leaks if leak.severity in ['critical', 'high']]
            if critical_leaks:
                recommendations.append(f"Critical: {len(critical_leaks)} high-severity memory leaks detected. Investigate immediately.")
        
        # Check for object count anomalies
        for obj_type, count in snapshot.object_counts.items():
            if count > 100000:  # Arbitrary threshold
                recommendations.append(f"High object count: {count:,} {obj_type} objects in memory. Consider optimization.")
        
        if snapshot.gc_stats.get('total_uncollectable', 0) > 1000:
            recommendations.append("Many uncollectable objects detected. Check for circular references.")
        
        if not recommendations:
            recommendations.append("Memory usage appears healthy. Continue monitoring.")
        
        return recommendations
    
    def cleanup(self):
        """Clean up profiler resources"""
        self.stop_monitoring()
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        self.snapshots.clear()
        self.detected_leaks.clear()
        self.object_trackers.clear()
        self.allocation_trackers.clear()
        
        logger.info("Memory profiler cleanup completed")


# Global memory profiler instance
profiler = MemoryProfiler()


# Convenience decorators
def profile_memory(operation_name: str = None):
    """Decorator for profiling function memory usage"""
    def decorator(func):
        name = operation_name or func.__name__
        return profiler.profile_function(func, name)
    return decorator


@contextmanager
def memory_context(operation_name: str):
    """Convenience context manager for memory profiling"""
    with profiler.profile_memory(operation_name):
        yield
