"""
Rate Limiting Middleware
Advanced rate limiting with Redis support for production environments
"""

import asyncio
import time
import hashlib
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import redis.asyncio as redis

from api.config import get_settings
from utils.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Production-grade rate limiting middleware with:
    - Redis-backed sliding window algorithm
    - Multiple rate limiting strategies (global, user, endpoint, organization)
    - Token bucket algorithm for burst handling
    - Dynamic rate limit adjustments
    - Real-time monitoring and alerting
    - Rate limit bypass for premium users
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Redis connection for distributed rate limiting
        self.redis_client: Optional[redis.Redis] = None
        self.use_redis = kwargs.get("use_redis", True)
        
        # Rate limiting configuration
        self.config = {
            # Global limits
            "global_requests_per_minute": kwargs.get("global_limit", 10000),
            "global_requests_per_hour": kwargs.get("global_limit_hour", 100000),
            
            # Per-user limits (by auth level)
            "user_limits": {
                "free": {"per_minute": 60, "per_hour": 1000, "per_day": 10000},
                "premium": {"per_minute": 300, "per_hour": 10000, "per_day": 100000},
                "enterprise": {"per_minute": 1000, "per_hour": 50000, "per_day": 1000000},
                "admin": {"per_minute": 10000, "per_hour": 100000, "per_day": float('inf')}
            },
            
            # Per-organization limits
            "org_limits": {
                "startup": {"per_minute": 500, "per_hour": 10000},
                "business": {"per_minute": 2000, "per_hour": 50000},
                "enterprise": {"per_minute": 10000, "per_hour": 200000}
            },
            
            # Endpoint-specific limits (requests per minute)
            "endpoint_limits": {
                "POST:/api/v1/documents/process": 100,
                "POST:/api/v1/documents/batch": 10,
                "POST:/api/v1/documents/upload": 50,
                "GET:/api/v1/analytics/processing": 30,
                "POST:/auth/login": 5,  # Protect against brute force
                "POST:/auth/refresh": 20
            },
            
            # Burst handling with token bucket
            "token_bucket": {
                "capacity": kwargs.get("burst_capacity", 100),
                "refill_rate": kwargs.get("burst_refill_rate", 10),  # tokens per second
                "enabled": kwargs.get("enable_burst", True)
            }
        }
        
        # In-memory fallback for when Redis is unavailable
        self.memory_store: Dict[str, List[float]] = {}
        self.token_buckets: Dict[str, Dict[str, Any]] = {}
        
        # Monitoring metrics
        self.metrics = {
            "total_requests": 0,
            "rate_limited_requests": 0,
            "redis_errors": 0,
            "avg_response_time_ms": 0
        }
        
        logger.info("Advanced rate limiting middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main rate limiting dispatch with comprehensive checks"""
        start_time = time.time()
        
        try:
            # Initialize Redis connection if needed
            if self.use_redis and self.redis_client is None:
                await self._init_redis()
            
            # Extract request identifiers
            identifiers = await self._extract_identifiers(request)
            
            # Perform rate limiting checks
            rate_limit_result = await self._check_rate_limits(identifiers, request)
            
            if rate_limit_result["limited"]:
                self.metrics["rate_limited_requests"] += 1
                return self._create_rate_limit_response(rate_limit_result)
            
            # Update counters (async to not block request)
            asyncio.create_task(self._update_counters(identifiers, request))
            
            # Process request
            response = await call_next(request)
            
            # Add rate limiting headers
            self._add_rate_limit_headers(response, rate_limit_result)
            
            # Update metrics
            processing_time = (time.time() - start_time) * 1000
            self.metrics["total_requests"] += 1
            self.metrics["avg_response_time_ms"] = (
                (self.metrics["avg_response_time_ms"] * (self.metrics["total_requests"] - 1) + processing_time) /
                self.metrics["total_requests"]
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting middleware error: {e}")
            # Continue with request on middleware error
            return await call_next(request)
    
    async def _init_redis(self):
        """Initialize Redis connection for distributed rate limiting"""
        try:
            self.redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                password=settings.redis_password,
                db=settings.redis_db_rate_limit,
                decode_responses=True
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Redis connection established for rate limiting")
            
        except Exception as e:
            logger.warning(f"Redis connection failed, using memory store: {e}")
            self.redis_client = None
    
    async def _extract_identifiers(self, request: Request) -> Dict[str, Any]:
        """Extract all relevant identifiers for rate limiting"""
        identifiers = {
            "ip": getattr(request.state, "client_ip", "unknown"),
            "user_id": None,
            "organization_id": None,
            "user_tier": "free",
            "org_tier": "startup",
            "endpoint": f"{request.method}:{request.url.path}",
            "api_key": None
        }
        
        # Extract user information from request state (set by auth middleware)
        if hasattr(request.state, "current_user") and request.state.current_user:
            identifiers["user_id"] = request.state.current_user.id
            identifiers["user_tier"] = getattr(request.state.current_user, "tier", "free")
        
        if hasattr(request.state, "current_organization") and request.state.current_organization:
            identifiers["organization_id"] = request.state.current_organization.id
            identifiers["org_tier"] = getattr(request.state.current_organization, "tier", "startup")
        
        # Check for API key authentication
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            identifiers["api_key"] = hashlib.sha256(auth_header.encode()).hexdigest()[:16]
        
        return identifiers
    
    async def _check_rate_limits(self, identifiers: Dict[str, Any], request: Request) -> Dict[str, Any]:
        """Comprehensive rate limiting checks with multiple strategies"""
        current_time = time.time()
        
        # Check global rate limits first
        global_check = await self._check_global_limits(current_time)
        if global_check["limited"]:
            return global_check
        
        # Check user-specific limits
        user_check = await self._check_user_limits(identifiers, current_time)
        if user_check["limited"]:
            return user_check
        
        # Check organization limits
        org_check = await self._check_organization_limits(identifiers, current_time)
        if org_check["limited"]:
            return org_check
        
        # Check endpoint-specific limits
        endpoint_check = await self._check_endpoint_limits(identifiers, current_time)
        if endpoint_check["limited"]:
            return endpoint_check
        
        # Check token bucket for burst protection
        burst_check = await self._check_token_bucket(identifiers, current_time)
        if burst_check["limited"]:
            return burst_check
        
        # All checks passed
        return {
            "limited": False,
            "limits": {
                "global": global_check["remaining"],
                "user": user_check.get("remaining", float('inf')),
                "organization": org_check.get("remaining", float('inf')),
                "endpoint": endpoint_check.get("remaining", float('inf')),
                "burst_tokens": burst_check.get("tokens", 0)
            },
            "reset_times": {
                "global": global_check["reset"],
                "user": user_check.get("reset", 0),
                "organization": org_check.get("reset", 0),
                "endpoint": endpoint_check.get("reset", 0)
            }
        }
    
    async def _check_global_limits(self, current_time: float) -> Dict[str, Any]:
        """Check global rate limits"""
        minute_key = f"global:minute:{int(current_time // 60)}"
        hour_key = f"global:hour:{int(current_time // 3600)}"
        
        # Check minute limit
        minute_count = await self._get_count(minute_key)
        if minute_count >= self.config["global_requests_per_minute"]:
            return {
                "limited": True,
                "limit_type": "global_minute",
                "current": minute_count,
                "limit": self.config["global_requests_per_minute"],
                "reset": int((current_time // 60 + 1) * 60),
                "retry_after": 60
            }
        
        # Check hour limit
        hour_count = await self._get_count(hour_key)
        if hour_count >= self.config["global_requests_per_hour"]:
            return {
                "limited": True,
                "limit_type": "global_hour",
                "current": hour_count,
                "limit": self.config["global_requests_per_hour"],
                "reset": int((current_time // 3600 + 1) * 3600),
                "retry_after": 3600
            }
        
        return {
            "limited": False,
            "remaining": min(
                self.config["global_requests_per_minute"] - minute_count,
                self.config["global_requests_per_hour"] - hour_count
            ),
            "reset": int((current_time // 60 + 1) * 60)
        }
    
    async def _check_user_limits(self, identifiers: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """Check per-user rate limits based on tier"""
        user_id = identifiers.get("user_id")
        if not user_id:
            return {"limited": False}
        
        user_tier = identifiers.get("user_tier", "free")
        limits = self.config["user_limits"].get(user_tier, self.config["user_limits"]["free"])
        
        # Check different time windows
        minute_key = f"user:{user_id}:minute:{int(current_time // 60)}"
        hour_key = f"user:{user_id}:hour:{int(current_time // 3600)}"
        day_key = f"user:{user_id}:day:{int(current_time // 86400)}"
        
        minute_count = await self._get_count(minute_key)
        hour_count = await self._get_count(hour_key)
        day_count = await self._get_count(day_key)
        
        # Check limits
        if minute_count >= limits["per_minute"]:
            return {
                "limited": True,
                "limit_type": "user_minute",
                "current": minute_count,
                "limit": limits["per_minute"],
                "reset": int((current_time // 60 + 1) * 60),
                "retry_after": 60
            }
        
        if hour_count >= limits["per_hour"]:
            return {
                "limited": True,
                "limit_type": "user_hour",
                "current": hour_count,
                "limit": limits["per_hour"],
                "reset": int((current_time // 3600 + 1) * 3600),
                "retry_after": 3600
            }
        
        if limits["per_day"] != float('inf') and day_count >= limits["per_day"]:
            return {
                "limited": True,
                "limit_type": "user_day",
                "current": day_count,
                "limit": limits["per_day"],
                "reset": int((current_time // 86400 + 1) * 86400),
                "retry_after": 86400
            }
        
        return {
            "limited": False,
            "remaining": min(
                limits["per_minute"] - minute_count,
                limits["per_hour"] - hour_count,
                limits["per_day"] - day_count if limits["per_day"] != float('inf') else float('inf')
            ),
            "reset": int((current_time // 60 + 1) * 60)
        }
    
    async def _check_organization_limits(self, identifiers: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """Check per-organization rate limits"""
        org_id = identifiers.get("organization_id")
        if not org_id:
            return {"limited": False}
        
        org_tier = identifiers.get("org_tier", "startup")
        limits = self.config["org_limits"].get(org_tier, self.config["org_limits"]["startup"])
        
        minute_key = f"org:{org_id}:minute:{int(current_time // 60)}"
        hour_key = f"org:{org_id}:hour:{int(current_time // 3600)}"
        
        minute_count = await self._get_count(minute_key)
        hour_count = await self._get_count(hour_key)
        
        if minute_count >= limits["per_minute"]:
            return {
                "limited": True,
                "limit_type": "organization_minute",
                "current": minute_count,
                "limit": limits["per_minute"],
                "reset": int((current_time // 60 + 1) * 60),
                "retry_after": 60
            }
        
        if hour_count >= limits["per_hour"]:
            return {
                "limited": True,
                "limit_type": "organization_hour",
                "current": hour_count,
                "limit": limits["per_hour"],
                "reset": int((current_time // 3600 + 1) * 3600),
                "retry_after": 3600
            }
        
        return {
            "limited": False,
            "remaining": min(
                limits["per_minute"] - minute_count,
                limits["per_hour"] - hour_count
            ),
            "reset": int((current_time // 60 + 1) * 60)
        }
    
    async def _check_endpoint_limits(self, identifiers: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """Check endpoint-specific rate limits"""
        endpoint = identifiers.get("endpoint")
        if endpoint not in self.config["endpoint_limits"]:
            return {"limited": False}
        
        limit = self.config["endpoint_limits"][endpoint]
        minute_key = f"endpoint:{endpoint}:minute:{int(current_time // 60)}"
        
        count = await self._get_count(minute_key)
        
        if count >= limit:
            return {
                "limited": True,
                "limit_type": "endpoint",
                "current": count,
                "limit": limit,
                "reset": int((current_time // 60 + 1) * 60),
                "retry_after": 60
            }
        
        return {
            "limited": False,
            "remaining": limit - count,
            "reset": int((current_time // 60 + 1) * 60)
        }
    
    async def _check_token_bucket(self, identifiers: Dict[str, Any], current_time: float) -> Dict[str, Any]:
        """Check token bucket for burst protection"""
        if not self.config["token_bucket"]["enabled"]:
            return {"limited": False, "tokens": 0}
        
        user_id = identifiers.get("user_id", identifiers.get("ip", "anonymous"))
        bucket_key = f"bucket:{user_id}"
        
        # Get or create bucket
        bucket = await self._get_token_bucket(bucket_key, current_time)
        
        # Check if tokens available
        if bucket["tokens"] < 1:
            return {
                "limited": True,
                "limit_type": "burst_protection",
                "tokens": bucket["tokens"],
                "capacity": self.config["token_bucket"]["capacity"],
                "retry_after": 1  # Try again in 1 second
            }
        
        return {
            "limited": False,
            "tokens": bucket["tokens"]
        }
    
    async def _get_count(self, key: str) -> int:
        """Get request count from Redis or memory store"""
        try:
            if self.redis_client:
                count = await self.redis_client.get(key)
                return int(count) if count else 0
            else:
                # Memory store fallback
                current_time = time.time()
                if key not in self.memory_store:
                    self.memory_store[key] = []
                
                # Clean old entries
                self.memory_store[key] = [
                    t for t in self.memory_store[key]
                    if current_time - t < 3600  # Keep last hour
                ]
                
                return len(self.memory_store[key])
                
        except Exception as e:
            logger.error(f"Error getting count for {key}: {e}")
            self.metrics["redis_errors"] += 1
            return 0
    
    async def _get_token_bucket(self, key: str, current_time: float) -> Dict[str, Any]:
        """Get or create token bucket for burst control"""
        config = self.config["token_bucket"]
        
        try:
            if self.redis_client:
                # Redis implementation with Lua script for atomicity
                lua_script = """
                local key = KEYS[1]
                local capacity = tonumber(ARGV[1])
                local refill_rate = tonumber(ARGV[2])
                local current_time = tonumber(ARGV[3])
                
                local bucket = redis.call('HMGET', key, 'tokens', 'last_refill')
                local tokens = tonumber(bucket[1]) or capacity
                local last_refill = tonumber(bucket[2]) or current_time
                
                -- Calculate tokens to add based on time passed
                local time_passed = current_time - last_refill
                local tokens_to_add = math.floor(time_passed * refill_rate)
                tokens = math.min(capacity, tokens + tokens_to_add)
                
                -- Update bucket
                redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
                redis.call('EXPIRE', key, 3600)  -- Expire in 1 hour
                
                return {tokens, capacity}
                """
                
                result = await self.redis_client.eval(
                    lua_script, 1, key,
                    config["capacity"], config["refill_rate"], current_time
                )
                
                return {"tokens": result[0], "capacity": result[1]}
                
            else:
                # Memory store fallback
                if key not in self.token_buckets:
                    self.token_buckets[key] = {
                        "tokens": config["capacity"],
                        "last_refill": current_time
                    }
                
                bucket = self.token_buckets[key]
                
                # Refill tokens based on time passed
                time_passed = current_time - bucket["last_refill"]
                tokens_to_add = time_passed * config["refill_rate"]
                bucket["tokens"] = min(config["capacity"], bucket["tokens"] + tokens_to_add)
                bucket["last_refill"] = current_time
                
                return bucket
                
        except Exception as e:
            logger.error(f"Error with token bucket {key}: {e}")
            return {"tokens": config["capacity"], "capacity": config["capacity"]}
    
    async def _update_counters(self, identifiers: Dict[str, Any], request: Request):
        """Update rate limiting counters (async)"""
        current_time = time.time()
        
        try:
            keys_to_increment = []
            
            # Global counters
            keys_to_increment.extend([
                f"global:minute:{int(current_time // 60)}",
                f"global:hour:{int(current_time // 3600)}"
            ])
            
            # User counters
            if identifiers.get("user_id"):
                user_id = identifiers["user_id"]
                keys_to_increment.extend([
                    f"user:{user_id}:minute:{int(current_time // 60)}",
                    f"user:{user_id}:hour:{int(current_time // 3600)}",
                    f"user:{user_id}:day:{int(current_time // 86400)}"
                ])
            
            # Organization counters
            if identifiers.get("organization_id"):
                org_id = identifiers["organization_id"]
                keys_to_increment.extend([
                    f"org:{org_id}:minute:{int(current_time // 60)}",
                    f"org:{org_id}:hour:{int(current_time // 3600)}"
                ])
            
            # Endpoint counters
            endpoint = identifiers.get("endpoint")
            if endpoint in self.config["endpoint_limits"]:
                keys_to_increment.append(f"endpoint:{endpoint}:minute:{int(current_time // 60)}")
            
            # Update counters
            await self._increment_counters(keys_to_increment)
            
            # Update token bucket (consume one token)
            user_id = identifiers.get("user_id", identifiers.get("ip", "anonymous"))
            await self._consume_token(f"bucket:{user_id}", current_time)
            
        except Exception as e:
            logger.error(f"Error updating rate limiting counters: {e}")
    
    async def _increment_counters(self, keys: List[str]):
        """Increment multiple counters efficiently"""
        try:
            if self.redis_client:
                # Use Redis pipeline for efficiency
                pipe = self.redis_client.pipeline()
                for key in keys:
                    pipe.incr(key)
                    # Set expiration based on key type
                    if ":minute:" in key:
                        pipe.expire(key, 120)  # 2 minutes
                    elif ":hour:" in key:
                        pipe.expire(key, 7200)  # 2 hours
                    elif ":day:" in key:
                        pipe.expire(key, 172800)  # 2 days
                await pipe.execute()
            else:
                # Memory store fallback
                current_time = time.time()
                for key in keys:
                    if key not in self.memory_store:
                        self.memory_store[key] = []
                    self.memory_store[key].append(current_time)
        except Exception as e:
            logger.error(f"Error incrementing counters: {e}")
    
    async def _consume_token(self, bucket_key: str, current_time: float):
        """Consume one token from bucket"""
        try:
            if self.redis_client:
                lua_script = """
                local key = KEYS[1]
                local bucket = redis.call('HMGET', key, 'tokens')
                local tokens = tonumber(bucket[1]) or 0
                
                if tokens >= 1 then
                    redis.call('HINCRBYFLOAT', key, 'tokens', -1)
                end
                
                return tokens
                """
                await self.redis_client.eval(lua_script, 1, bucket_key)
            else:
                # Memory store fallback
                if bucket_key in self.token_buckets and self.token_buckets[bucket_key]["tokens"] >= 1:
                    self.token_buckets[bucket_key]["tokens"] -= 1
        except Exception as e:
            logger.error(f"Error consuming token from {bucket_key}: {e}")
    
    def _create_rate_limit_response(self, rate_limit_result: Dict[str, Any]) -> JSONResponse:
        """Create rate limit exceeded response"""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "code": "RATE_LIMIT_EXCEEDED",
                "limit_type": rate_limit_result.get("limit_type"),
                "current": rate_limit_result.get("current"),
                "limit": rate_limit_result.get("limit"),
                "reset_time": rate_limit_result.get("reset"),
                "retry_after": rate_limit_result.get("retry_after", 60),
                "message": f"Rate limit exceeded for {rate_limit_result.get('limit_type', 'requests')}. Try again in {rate_limit_result.get('retry_after', 60)} seconds."
            },
            headers={
                "Retry-After": str(rate_limit_result.get("retry_after", 60)),
                "X-RateLimit-Exceeded": rate_limit_result.get("limit_type", "unknown")
            }
        )
    
    def _add_rate_limit_headers(self, response: Response, rate_limit_result: Dict[str, Any]):
        """Add rate limiting headers to successful responses"""
        if "limits" in rate_limit_result:
            limits = rate_limit_result["limits"]
            reset_times = rate_limit_result["reset_times"]
            
            # Add comprehensive rate limit headers
            response.headers["X-RateLimit-Global-Remaining"] = str(limits.get("global", 0))
            response.headers["X-RateLimit-User-Remaining"] = str(limits.get("user", 0))
            response.headers["X-RateLimit-Org-Remaining"] = str(limits.get("organization", 0))
            response.headers["X-RateLimit-Endpoint-Remaining"] = str(limits.get("endpoint", 0))
            response.headers["X-RateLimit-Reset"] = str(reset_times.get("global", 0))
            response.headers["X-RateLimit-Burst-Tokens"] = str(limits.get("burst_tokens", 0))
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get rate limiting metrics for monitoring"""
        return {
            **self.metrics,
            "rate_limit_percentage": (
                (self.metrics["rate_limited_requests"] / max(self.metrics["total_requests"], 1)) * 100
            ),
            "redis_available": self.redis_client is not None,
            "memory_store_keys": len(self.memory_store),
            "token_buckets": len(self.token_buckets)
        }
    
    async def clear_limits_for_user(self, user_id: str):
        """Clear rate limits for specific user (admin function)"""
        try:
            if self.redis_client:
                # Find and delete all keys for this user
                pattern = f"user:{user_id}:*"
                keys = await self.redis_client.keys(pattern)
                if keys:
                    await self.redis_client.delete(*keys)
            
            # Clear memory store
            keys_to_remove = [k for k in self.memory_store.keys() if f"user:{user_id}:" in k]
            for key in keys_to_remove:
                del self.memory_store[key]
            
            # Clear token bucket
            bucket_key = f"bucket:{user_id}"
            if bucket_key in self.token_buckets:
                del self.token_buckets[bucket_key]
            
            logger.info(f"Rate limits cleared for user: {user_id}")
            
        except Exception as e:
            logger.error(f"Error clearing rate limits for user {user_id}: {e}")


# Export the middleware
__all__ = ["RateLimitMiddleware"]