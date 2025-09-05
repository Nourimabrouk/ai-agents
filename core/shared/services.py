"""
Base Service Implementations
Provides default implementations of core services
"""

import json
import logging
import time
import asyncio
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict
from datetime import datetime, timedelta

from .interfaces import IMemoryStore, IMetricsCollector, IConfigurationProvider

logger = logging.getLogger(__name__)


class BaseMemoryStore(IMemoryStore):
    """
    In-memory implementation of memory store
    Suitable for development and testing
    """
    
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._ttl: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value with optional TTL in seconds"""
        async with self._lock:
            self._data[key] = value
            
            if ttl:
                self._ttl[key] = datetime.utcnow() + timedelta(seconds=ttl)
            elif key in self._ttl:
                del self._ttl[key]
        
        logger.debug(f"Stored key {key} with TTL {ttl}")
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value by key, checking TTL"""
        async with self._lock:
            # Check TTL first
            if key in self._ttl:
                if datetime.utcnow() > self._ttl[key]:
                    # Expired, remove
                    self._data.pop(key, None)
                    del self._ttl[key]
                    return {}
            
            return self._data.get(key)
    
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        async with self._lock:
            existed = key in self._data
            self._data.pop(key, None)
            self._ttl.pop(key, None)
            return existed
    
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern (basic glob support)"""
        async with self._lock:
            # Clean up expired keys first
            expired_keys = []
            for key, expiry in self._ttl.items():
                if datetime.utcnow() > expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._data.pop(key, None)
                del self._ttl[key]
            
            # Simple pattern matching
            if pattern == "*":
                return list(self._data.keys())
            else:
                # Basic prefix/suffix matching
                if pattern.endswith("*"):
                    prefix = pattern[:-1]
                    return [k for k in self._data.keys() if k.startswith(prefix)]
                elif pattern.startswith("*"):
                    suffix = pattern[1:]
                    return [k for k in self._data.keys() if k.endswith(suffix)]
                else:
                    return [k for k in self._data.keys() if k == pattern]
    
    async def cleanup_expired(self) -> int:
        """Cleanup expired keys, return count removed"""
        async with self._lock:
            expired_keys = []
            for key, expiry in self._ttl.items():
                if datetime.utcnow() > expiry:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._data.pop(key, None)
                del self._ttl[key]
            
            logger.debug(f"Cleaned up {len(expired_keys)} expired keys")
            return len(expired_keys)


class BaseMetricsCollector(IMetricsCollector):
    """
    In-memory metrics collector
    Suitable for development and basic monitoring
    """
    
    def __init__(self):
        self._counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._gauges: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._histograms: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        self._lock = asyncio.Lock()
        self._start_time = time.time()
    
    def _tags_key(self, tags: Dict[str, str] = None) -> str:
        """Create key from tags"""
        if not tags:
            return "default"
        return ",".join(f"{k}={v}" for k, v in sorted(tags.items()))
    
    async def increment_counter(self, name: str, tags: Dict[str, str] = None) -> None:
        """Increment counter metric"""
        tags_key = self._tags_key(tags)
        async with self._lock:
            self._counters[name][tags_key] += 1
        
        logger.debug(f"Incremented counter {name} with tags {tags}")
    
    async def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record gauge metric"""
        tags_key = self._tags_key(tags)
        async with self._lock:
            self._gauges[name][tags_key] = value
        
        logger.debug(f"Recorded gauge {name} = {value} with tags {tags}")
    
    async def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record histogram metric"""
        tags_key = self._tags_key(tags)
        async with self._lock:
            self._histograms[name][tags_key].append(value)
            # Keep only last 1000 values to prevent memory issues
            if len(self._histograms[name][tags_key]) > 1000:
                self._histograms[name][tags_key] = self._histograms[name][tags_key][-1000:]
        
        logger.debug(f"Recorded histogram {name} = {value} with tags {tags}")
    
    async def get_metrics(self, name_pattern: str = "*") -> Dict[str, Any]:
        """Get metrics matching pattern"""
        async with self._lock:
            result = {
                "system": {
                    "uptime": time.time() - self._start_time,
                    "timestamp": datetime.utcnow().isoformat()
                },
                "counters": {},
                "gauges": {},
                "histograms": {}
            }
            
            # Filter metrics by pattern
            for name, values in self._counters.items():
                if name_pattern == "*" or name_pattern in name:
                    result["counters"][name] = dict(values)
            
            for name, values in self._gauges.items():
                if name_pattern == "*" or name_pattern in name:
                    result["gauges"][name] = dict(values)
            
            for name, values in self._histograms.items():
                if name_pattern == "*" or name_pattern in name:
                    # Calculate statistics for histograms
                    result["histograms"][name] = {}
                    for tags_key, hist_values in values.items():
                        if hist_values:
                            result["histograms"][name][tags_key] = {
                                "count": len(hist_values),
                                "min": min(hist_values),
                                "max": max(hist_values),
                                "avg": sum(hist_values) / len(hist_values),
                                "p95": self._percentile(hist_values, 95),
                                "p99": self._percentile(hist_values, 99)
                            }
            
            return result
    
    def _percentile(self, values: List[float], p: int) -> float:
        """Calculate percentile"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * p / 100
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c


class BaseConfigurationProvider(IConfigurationProvider):
    """
    File-based configuration provider
    Supports JSON and environment variable override
    """
    
    def __init__(self, config_file: str = None):
        self._config: Dict[str, Any] = {}
        self._config_file = config_file
        self._lock = asyncio.Lock()
        
        if config_file:
            self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file"""
        try:
            with open(self._config_file, 'r') as f:
                self._config = json.load(f)
            logger.info(f"Loaded configuration from {self._config_file}")
        except FileNotFoundError:
            logger.warning(f"Configuration file {self._config_file} not found, using defaults")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing configuration file: {e}")
            raise
    
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value with dot notation support"""
        async with self._lock:
            # Support dot notation (e.g., "database.host")
            keys = key.split(".")
            value = self._config
            
            try:
                for k in keys:
                    value = value[k]
                return value
            except (KeyError, TypeError):
                return default
    
    async def set_config(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support"""
        async with self._lock:
            keys = key.split(".")
            config_ref = self._config
            
            # Navigate to parent
            for k in keys[:-1]:
                if k not in config_ref:
                    config_ref[k] = {}
                config_ref = config_ref[k]
            
            # Set final value
            config_ref[keys[-1]] = value
            
            # Save to file if configured
            if self._config_file:
                try:
                    with open(self._config_file, 'w') as f:
                        json.dump(self._config, f, indent=2)
                except Exception as e:
                    logger.error(f"Error saving configuration: {e}")
    
    async def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        async with self._lock:
            return self._config.copy()
    
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary"""
        self._config.update(config_dict)