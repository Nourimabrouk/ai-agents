"""
Metrics collection and reporting for AI agents
Provides performance tracking and monitoring capabilities
"""

from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from collections import defaultdict
import time
import asyncio
from dataclasses import dataclass, field


@dataclass
class MetricPoint:
    """Single metric data point"""
    name: str
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """Collects and aggregates metrics"""
    
    def __init__(self, namespace: str = "ai_agents"):
        self.namespace = namespace
        self.metrics: Dict[str, List[MetricPoint]] = defaultdict(list)
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.timers: Dict[str, List[float]] = defaultdict(list)
    
    def incr(self, name: str, value: float = 1.0, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric"""
        full_name = f"{self.namespace}.{name}"
        self.counters[full_name] += value
        self.metrics[full_name].append(
            MetricPoint(full_name, self.counters[full_name], tags or {})
        )
    
    def gauge(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric"""
        full_name = f"{self.namespace}.{name}"
        self.gauges[full_name] = value
        self.metrics[full_name].append(
            MetricPoint(full_name, value, tags or {})
        )
    
    def timing(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None):
        """Record a timing metric"""
        full_name = f"{self.namespace}.{name}"
        self.timers[full_name].append(duration)
        self.metrics[full_name].append(
            MetricPoint(full_name, duration, tags or {})
        )
    
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations"""
        return Timer(self, name, tags)
    
    def get_stats(self, name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        full_name = f"{self.namespace}.{name}"
        
        if full_name in self.counters:
            return {"type": "counter", "value": self.counters[full_name]}
        elif full_name in self.gauges:
            return {"type": "gauge", "value": self.gauges[full_name]}
        elif full_name in self.timers:
            timings = self.timers[full_name]
            if timings:
                return {
                    "type": "timer",
                    "count": len(timings),
                    "mean": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "sum": sum(timings)
                }
        
        return {}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get all collected statistics"""
        stats = {}
        
        # Add counters
        for name, value in self.counters.items():
            stats[name] = {"type": "counter", "value": value}
        
        # Add gauges
        for name, value in self.gauges.items():
            stats[name] = {"type": "gauge", "value": value}
        
        # Add timers
        for name, timings in self.timers.items():
            if timings:
                stats[name] = {
                    "type": "timer",
                    "count": len(timings),
                    "mean": sum(timings) / len(timings),
                    "min": min(timings),
                    "max": max(timings),
                    "sum": sum(timings)
                }
        
        return stats
    
    def reset(self):
        """Reset all metrics"""
        self.metrics.clear()
        self.counters.clear()
        self.gauges.clear()
        self.timers.clear()


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.collector.timing(self.name, duration, self.tags)


class AsyncTimer:
    """Async context manager for timing async operations"""
    
    def __init__(self, collector: MetricsCollector, name: str, tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.name = name
        self.tags = tags
        self.start_time = None
    
    async def __aenter__(self):
        self.start_time = time.perf_counter()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            self.collector.timing(self.name, duration, self.tags)


# Global metrics instance
global_metrics = MetricsCollector()


def get_metrics(namespace: Optional[str] = None) -> MetricsCollector:
    """Get metrics collector instance"""
    if namespace:
        return MetricsCollector(namespace)
    return global_metrics