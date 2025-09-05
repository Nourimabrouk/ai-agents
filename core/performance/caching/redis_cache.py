"""
Redis-based distributed caching for high-performance operations
Implements intelligent caching with compression, TTL, and cache warming
"""

import redis
import json
import pickle
import zlib
import hashlib
import asyncio
from typing import Any, Optional, Union, Dict, List, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Redis cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    total_requests: int = 0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests * 100.0


class RedisCache:
    """
    High-performance Redis cache with compression and intelligent invalidation
    Designed for Phase 7 system optimization
    """
    
    def __init__(self,
                 redis_url: str = "redis://localhost:6379",
                 max_memory: str = "1gb",
                 compression_threshold: int = 1024,
                 default_ttl: int = 3600,
                 key_prefix: str = "phase7:"):
        
        self.redis_url = redis_url
        self.compression_threshold = compression_threshold
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        
        # Initialize Redis connection with optimized settings
        self.client = redis.from_url(
            redis_url,
            decode_responses=False,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )
        
        # Configure Redis for performance
        try:
            self.client.config_set('maxmemory', max_memory)
            self.client.config_set('maxmemory-policy', 'allkeys-lru')
            self.client.config_set('save', '')  # Disable persistence for performance
        except Exception as e:
            logger.warning(f"Could not configure Redis settings: {e}")
        
        # Performance statistics
        self.stats = CacheStats()
        self.start_time = datetime.now()
        
        logger.info(f"Initialized RedisCache with compression threshold: {compression_threshold} bytes")
    
    def _generate_key(self, key: str) -> str:
        """Generate prefixed cache key"""
        return f"{self.key_prefix}{key}"
    
    def _serialize_and_compress(self, value: Any) -> bytes:
        """Serialize and optionally compress data"""
        # Serialize using pickle for Python objects
        serialized = pickle.dumps(value)
        
        # Compress if data is large enough
        if len(serialized) > self.compression_threshold:
            compressed = zlib.compress(serialized, level=6)  # Balanced compression
            return b'compressed:' + compressed
        else:
            return b'raw:' + serialized
    
    def _decompress_and_deserialize(self, data: bytes) -> Any:
        """Decompress and deserialize cached data"""
        if data.startswith(b'compressed:'):
            # Decompress first
            compressed_data = data[11:]  # Remove 'compressed:' prefix
            serialized = zlib.decompress(compressed_data)
        elif data.startswith(b'raw:'):
            serialized = data[4:]  # Remove 'raw:' prefix
        else:
            # Legacy format - assume raw
            serialized = data
        
        return pickle.loads(serialized)
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache with performance tracking"""
        start_time = datetime.now()
        cache_key = self._generate_key(key)
        
        try:
            data = self.client.get(cache_key)
            
            if data is not None:
                self.stats.hits += 1
                value = self._decompress_and_deserialize(data)
                
                # Track response time
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                self._update_avg_response_time(response_time)
                
                logger.debug(f"Cache HIT for key: {key} (took {response_time:.2f}ms)")
                return value
            else:
                self.stats.misses += 1
                logger.debug(f"Cache MISS for key: {key}")
                return {}
                
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            self.stats.misses += 1
            return {}
        finally:
            self.stats.total_requests += 1
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL"""
        start_time = datetime.now()
        cache_key = self._generate_key(key)
        ttl = ttl or self.default_ttl
        
        try:
            # Serialize and compress
            data = self._serialize_and_compress(value)
            
            # Set with TTL
            result = self.client.setex(cache_key, ttl, data)
            
            if result:
                self.stats.sets += 1
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                logger.debug(f"Cache SET for key: {key} (took {response_time:.2f}ms, size: {len(data)} bytes)")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        cache_key = self._generate_key(key)
        
        try:
            result = self.client.delete(cache_key)
            if result > 0:
                self.stats.deletes += 1
                logger.debug(f"Cache DELETE for key: {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        cache_pattern = self._generate_key(pattern)
        
        try:
            keys = self.client.keys(cache_pattern)
            if keys:
                deleted = self.client.delete(*keys)
                self.stats.deletes += deleted
                logger.info(f"Invalidated {deleted} keys matching pattern: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Redis pattern invalidation error for {pattern}: {e}")
            return 0
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        cache_key = self._generate_key(key)
        try:
            return bool(self.client.exists(cache_key))
        except Exception as e:
            logger.error(f"Redis EXISTS error for key {key}: {e}")
            return False
    
    async def get_ttl(self, key: str) -> int:
        """Get TTL for key (-1 if no TTL, -2 if key doesn't exist)"""
        cache_key = self._generate_key(key)
        try:
            return self.client.ttl(cache_key)
        except Exception as e:
            logger.error(f"Redis TTL error for key {key}: {e}")
            return -2
    
    def cache_decorator(self, ttl: int = None, key_prefix: str = ""):
        """Decorator for automatic function result caching"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key from function name and arguments
                key_parts = [key_prefix, func.__name__, str(args), str(sorted(kwargs.items()))]
                cache_key = hashlib.md5(':'.join(key_parts).encode()).hexdigest()
                
                # Try to get from cache
                cached_result = await self.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl)
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # For sync functions, create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
            # Return appropriate wrapper based on function type
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
                
        return decorator
    
    async def warm_cache(self, warm_data: Dict[str, Any]) -> int:
        """Pre-populate cache with frequently accessed data"""
        warmed = 0
        
        for key, value in warm_data.items():
            if await self.set(key, value):
                warmed += 1
        
        logger.info(f"Warmed cache with {warmed} entries")
        return warmed
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get Redis memory usage statistics"""
        try:
            info = self.client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'used_memory_peak': info.get('used_memory_peak', 0),
                'used_memory_peak_human': info.get('used_memory_peak_human', '0B'),
                'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0.0)
            }
        except Exception as e:
            logger.error(f"Error getting Redis memory info: {e}")
            return {}
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        if self.stats.avg_response_time == 0:
            self.stats.avg_response_time = response_time
        else:
            self.stats.avg_response_time = (
                alpha * response_time + (1 - alpha) * self.stats.avg_response_time
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'hit_rate': f"{self.stats.hit_rate:.2f}%",
            'total_requests': self.stats.total_requests,
            'hits': self.stats.hits,
            'misses': self.stats.misses,
            'sets': self.stats.sets,
            'deletes': self.stats.deletes,
            'avg_response_time_ms': f"{self.stats.avg_response_time:.2f}",
            'requests_per_second': f"{self.stats.total_requests / max(1, uptime):.2f}",
            'uptime_seconds': uptime
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform Redis health check"""
        try:
            start_time = datetime.now()
            
            # Test basic operations
            test_key = "health_check"
            test_value = {"timestamp": start_time.isoformat()}
            
            # Test SET
            await self.set(test_key, test_value, ttl=60)
            
            # Test GET
            result = await self.get(test_key)
            
            # Test DELETE
            await self.delete(test_key)
            
            response_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Get memory stats
            memory_stats = await self.get_memory_usage()
            
            return {
                'status': 'healthy',
                'response_time_ms': response_time,
                'memory_stats': memory_stats,
                'performance_stats': self.get_performance_stats(),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    async def close(self):
        """Close Redis connection"""
        try:
            self.client.close()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {e}")


# Global cache instance for easy import
cache = None

def get_cache(redis_url: str = "redis://localhost:6379") -> RedisCache:
    """Get global cache instance (singleton pattern)"""
    global cache
    if cache is None:
        cache = RedisCache(redis_url=redis_url)
    return cache


# Convenience decorators
def cached(ttl: int = 3600, key_prefix: str = ""):
    """Convenience decorator using global cache instance"""
    return get_cache().cache_decorator(ttl=ttl, key_prefix=key_prefix)
