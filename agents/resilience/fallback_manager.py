"""
Resilience Framework: Phase 6 - Comprehensive Fallback and Recovery System
Features:
- Circuit breakers for API failures
- Retry strategies with exponential backoff
- Graceful degradation patterns
- Local model fallbacks
"""

import asyncio
import time
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict, deque
import functools
from abc import ABC, abstractmethod

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class FailureType(Enum):
    """Types of failures the system can handle"""
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    API_ERROR = "api_error"
    NETWORK_ERROR = "network_error"
    AUTHENTICATION_ERROR = "auth_error"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    PROCESSING_ERROR = "processing_error"
    VALIDATION_ERROR = "validation_error"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class RetryConfig:
    """Configuration for retry strategies"""
    max_attempts: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers"""
    failure_threshold: int = 5  # failures before opening
    recovery_timeout: int = 60  # seconds to wait before half-open
    success_threshold: int = 3   # successes needed to close from half-open
    monitoring_window: int = 300  # seconds to track failures


@dataclass
class FailureRecord:
    """Record of a system failure"""
    failure_id: str
    failure_type: FailureType
    component: str
    error_message: str
    timestamp: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovered_at: Optional[datetime] = None


class CircuitBreaker:
    """Circuit breaker implementation for resilient service calls"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.failure_window = deque(maxlen=100)  # Recent failures for analysis
        
    def _record_success(self):
        """Record a successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = max(0, self.failure_count - 1)  # Gradual recovery
    
    def _record_failure(self, failure_type: FailureType = None):
        """Record a failed operation"""
        current_time = datetime.now()
        self.failure_count += 1
        self.last_failure_time = current_time
        self.failure_window.append((current_time, failure_type))
        
        if self.state == CircuitState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()
        elif self.state == CircuitState.HALF_OPEN:
            self._open_circuit()
    
    def _open_circuit(self):
        """Open the circuit breaker"""
        self.state = CircuitState.OPEN
        self.success_count = 0
        logger.warning(f"Circuit breaker {self.name} OPENED after {self.failure_count} failures")
        global_metrics.incr(f"circuit_breaker.{self.name}.opened")
    
    def _close_circuit(self):
        """Close the circuit breaker"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        logger.info(f"Circuit breaker {self.name} CLOSED - service recovered")
        global_metrics.incr(f"circuit_breaker.{self.name}.closed")
    
    def _attempt_reset(self):
        """Attempt to reset circuit breaker to half-open"""
        if (self.state == CircuitState.OPEN and 
            self.last_failure_time and
            datetime.now() - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout)):
            
            self.state = CircuitState.HALF_OPEN
            self.success_count = 0
            logger.info(f"Circuit breaker {self.name} moved to HALF-OPEN for testing")
    
    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            self._attempt_reset()
            return self.state != CircuitState.OPEN
        else:  # HALF_OPEN
            return True
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if not self.can_execute():
            raise Exception(f"Circuit breaker {self.name} is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            failure_type = self._classify_error(e)
            self._record_failure(failure_type)
            raise
    
    def _classify_error(self, error: Exception) -> FailureType:
        """Classify error type for circuit breaker logic"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureType.API_TIMEOUT
        elif "rate limit" in error_str or "429" in error_str:
            return FailureType.API_RATE_LIMIT
        elif "connection" in error_str or "network" in error_str:
            return FailureType.NETWORK_ERROR
        elif "auth" in error_str or "401" in error_str or "403" in error_str:
            return FailureType.AUTHENTICATION_ERROR
        else:
            return FailureType.API_ERROR
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        recent_failures = [
            f for f in self.failure_window 
            if f[0] > datetime.now() - timedelta(seconds=self.config.monitoring_window)
        ]
        
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "recent_failures": len(recent_failures),
            "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
        }


class RetryManager:
    """Manages retry strategies with various backoff algorithms"""
    
    def __init__(self):
        self.retry_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """Calculate delay for retry attempt"""
        if config.backoff_strategy == "exponential":
            delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        elif config.backoff_strategy == "linear":
            delay = config.base_delay * attempt
        else:  # fixed
            delay = config.base_delay
        
        # Cap at max delay
        delay = min(delay, config.max_delay)
        
        # Add jitter to prevent thundering herd
        if config.jitter:
            import random
            jitter_factor = random.uniform(0.5, 1.5)
            delay *= jitter_factor
        
        return delay
    
    def _should_retry(self, error: Exception, attempt: int, config: RetryConfig) -> bool:
        """Determine if operation should be retried"""
        if attempt >= config.max_attempts:
            return False
        
        # Don't retry certain types of errors
        error_str = str(error).lower()
        non_retryable_errors = ["authentication", "401", "403", "validation"]
        
        if any(err in error_str for err in non_retryable_errors):
            return False
        
        return True
    
    async def retry_with_backoff(self, func: Callable, operation_id: str, 
                               config: RetryConfig = None, *args, **kwargs):
        """Execute function with retry and backoff logic"""
        if config is None:
            config = RetryConfig()
        
        attempt = 1
        start_time = time.time()
        
        while attempt <= config.max_attempts:
            try:
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Record successful retry
                if attempt > 1:
                    self.retry_history[operation_id].append({
                        "timestamp": datetime.now(),
                        "attempts": attempt,
                        "success": True,
                        "total_time": time.time() - start_time
                    })
                    logger.info(f"Operation {operation_id} succeeded after {attempt} attempts")
                
                return result
                
            except Exception as e:
                if not self._should_retry(e, attempt, config):
                    # Record failed retry
                    self.retry_history[operation_id].append({
                        "timestamp": datetime.now(),
                        "attempts": attempt,
                        "success": False,
                        "error": str(e),
                        "total_time": time.time() - start_time
                    })
                    logger.error(f"Operation {operation_id} failed after {attempt} attempts: {e}")
                    raise
                
                if attempt < config.max_attempts:
                    delay = self._calculate_delay(attempt, config)
                    logger.warning(f"Attempt {attempt} failed for {operation_id}, retrying in {delay:.2f}s: {e}")
                    await asyncio.sleep(delay)
                
                attempt += 1
        
        # This shouldn't be reached, but just in case
        raise Exception(f"Operation {operation_id} exhausted all retry attempts")
    
    def get_retry_stats(self, operation_id: str = None) -> Dict[str, Any]:
        """Get retry statistics"""
        if operation_id:
            history = self.retry_history.get(operation_id, [])
            if not history:
                return {"operation_id": operation_id, "attempts": 0}
            
            successful_retries = [h for h in history if h["success"]]
            failed_retries = [h for h in history if not h["success"]]
            
            return {
                "operation_id": operation_id,
                "total_operations": len(history),
                "successful_retries": len(successful_retries),
                "failed_operations": len(failed_retries),
                "average_attempts": sum(h["attempts"] for h in history) / len(history),
                "average_success_time": sum(h["total_time"] for h in successful_retries) / len(successful_retries) if successful_retries else 0
            }
        else:
            # Global stats
            all_operations = sum(len(history) for history in self.retry_history.values())
            if all_operations == 0:
                return {"total_operations": 0}
            
            total_attempts = sum(
                sum(h["attempts"] for h in history) 
                for history in self.retry_history.values()
            )
            
            successful_operations = sum(
                len([h for h in history if h["success"]]) 
                for history in self.retry_history.values()
            )
            
            return {
                "total_operations": all_operations,
                "successful_operations": successful_operations,
                "success_rate": successful_operations / all_operations,
                "average_attempts_per_operation": total_attempts / all_operations,
                "operation_count": len(self.retry_history)
            }


class FallbackChain:
    """Manages fallback strategies when primary services fail"""
    
    def __init__(self):
        self.fallback_chains: Dict[str, List[Callable]] = {}
        self.fallback_history: List[Dict[str, Any]] = []
    
    def register_fallback_chain(self, service_name: str, primary_func: Callable, 
                               fallback_funcs: List[Callable]):
        """Register a fallback chain for a service"""
        self.fallback_chains[service_name] = [primary_func] + fallback_funcs
        logger.info(f"Registered fallback chain for {service_name} with {len(fallback_funcs)} fallbacks")
    
    async def execute_with_fallback(self, service_name: str, *args, **kwargs):
        """Execute service with fallback chain"""
        if service_name not in self.fallback_chains:
            raise ValueError(f"No fallback chain registered for service: {service_name}")
        
        chain = self.fallback_chains[service_name]
        last_error = None
        
        for i, func in enumerate(chain):
            try:
                is_fallback = i > 0
                logger.debug(f"Attempting {service_name} with {'fallback' if is_fallback else 'primary'} method {i}")
                
                result = await func(*args, **kwargs) if asyncio.iscoroutinefunction(func) else func(*args, **kwargs)
                
                # Record successful execution
                self.fallback_history.append({
                    "timestamp": datetime.now(),
                    "service": service_name,
                    "method_used": i,
                    "is_fallback": is_fallback,
                    "success": True
                })
                
                if is_fallback:
                    logger.info(f"Service {service_name} succeeded using fallback method {i}")
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Method {i} failed for {service_name}: {e}")
                continue
        
        # All methods failed
        self.fallback_history.append({
            "timestamp": datetime.now(),
            "service": service_name,
            "method_used": -1,
            "is_fallback": True,
            "success": False,
            "error": str(last_error)
        })
        
        logger.error(f"All fallback methods failed for {service_name}")
        raise Exception(f"Service {service_name} completely failed: {last_error}")
    
    def get_fallback_stats(self, service_name: str = None) -> Dict[str, Any]:
        """Get fallback usage statistics"""
        if service_name:
            service_history = [h for h in self.fallback_history if h["service"] == service_name]
        else:
            service_history = self.fallback_history
        
        if not service_history:
            return {"total_calls": 0}
        
        successful_calls = [h for h in service_history if h["success"]]
        fallback_calls = [h for h in service_history if h["is_fallback"] and h["success"]]
        
        fallback_usage_by_method = defaultdict(int)
        for call in successful_calls:
            fallback_usage_by_method[call["method_used"]] += 1
        
        return {
            "total_calls": len(service_history),
            "successful_calls": len(successful_calls),
            "success_rate": len(successful_calls) / len(service_history),
            "fallback_usage_rate": len(fallback_calls) / len(service_history),
            "fallback_usage_by_method": dict(fallback_usage_by_method),
            "service_name": service_name
        }


class GracefulDegradation:
    """Manages graceful degradation of service quality under stress"""
    
    def __init__(self):
        self.degradation_levels = {
            "normal": 1.0,
            "reduced": 0.7,
            "minimal": 0.4,
            "emergency": 0.2
        }
        self.current_level = "normal"
        self.degradation_history: List[Dict[str, Any]] = []
        self.feature_flags: Dict[str, bool] = {}
    
    def assess_system_health(self, metrics: Dict[str, float]) -> str:
        """Assess system health and determine degradation level"""
        health_score = 1.0
        
        # Check various health indicators
        if "cpu_usage" in metrics:
            if metrics["cpu_usage"] > 0.9:
                health_score *= 0.6
            elif metrics["cpu_usage"] > 0.8:
                health_score *= 0.8
        
        if "memory_usage" in metrics:
            if metrics["memory_usage"] > 0.9:
                health_score *= 0.6
            elif metrics["memory_usage"] > 0.8:
                health_score *= 0.8
        
        if "error_rate" in metrics:
            if metrics["error_rate"] > 0.1:  # 10% error rate
                health_score *= 0.5
            elif metrics["error_rate"] > 0.05:  # 5% error rate
                health_score *= 0.7
        
        if "response_time" in metrics:
            if metrics["response_time"] > 10.0:  # 10 second response time
                health_score *= 0.6
            elif metrics["response_time"] > 5.0:  # 5 second response time
                health_score *= 0.8
        
        # Determine degradation level
        if health_score >= 0.8:
            new_level = "normal"
        elif health_score >= 0.6:
            new_level = "reduced"
        elif health_score >= 0.3:
            new_level = "minimal"
        else:
            new_level = "emergency"
        
        if new_level != self.current_level:
            self._change_degradation_level(new_level, health_score)
        
        return new_level
    
    def _change_degradation_level(self, new_level: str, health_score: float):
        """Change system degradation level"""
        old_level = self.current_level
        self.current_level = new_level
        
        # Update feature flags based on degradation level
        self._update_feature_flags(new_level)
        
        # Record degradation event
        self.degradation_history.append({
            "timestamp": datetime.now(),
            "old_level": old_level,
            "new_level": new_level,
            "health_score": health_score,
            "feature_flags": self.feature_flags.copy()
        })
        
        logger.warning(f"System degradation level changed from {old_level} to {new_level} "
                      f"(health score: {health_score:.3f})")
    
    def _update_feature_flags(self, degradation_level: str):
        """Update feature flags based on degradation level"""
        if degradation_level == "normal":
            self.feature_flags = {
                "detailed_logging": True,
                "caching": True,
                "analytics": True,
                "background_tasks": True,
                "full_features": True
            }
        elif degradation_level == "reduced":
            self.feature_flags = {
                "detailed_logging": True,
                "caching": True,
                "analytics": True,
                "background_tasks": False,
                "full_features": True
            }
        elif degradation_level == "minimal":
            self.feature_flags = {
                "detailed_logging": False,
                "caching": True,
                "analytics": False,
                "background_tasks": False,
                "full_features": False
            }
        else:  # emergency
            self.feature_flags = {
                "detailed_logging": False,
                "caching": False,
                "analytics": False,
                "background_tasks": False,
                "full_features": False
            }
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled given current degradation level"""
        return self.feature_flags.get(feature_name, True)
    
    def get_quality_multiplier(self) -> float:
        """Get quality multiplier for current degradation level"""
        return self.degradation_levels.get(self.current_level, 1.0)
    
    def get_degradation_stats(self) -> Dict[str, Any]:
        """Get degradation statistics"""
        level_counts = defaultdict(int)
        for event in self.degradation_history:
            level_counts[event["new_level"]] += 1
        
        recent_events = [
            e for e in self.degradation_history 
            if e["timestamp"] > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "current_level": self.current_level,
            "quality_multiplier": self.get_quality_multiplier(),
            "feature_flags": self.feature_flags.copy(),
            "degradation_events": len(self.degradation_history),
            "recent_events": len(recent_events),
            "level_history": dict(level_counts)
        }


class ResilienceFramework:
    """Main resilience framework coordinating all resilience components"""
    
    def __init__(self):
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.retry_manager = RetryManager()
        self.fallback_chain = FallbackChain()
        self.graceful_degradation = GracefulDegradation()
        self.failure_records: List[FailureRecord] = []
        
    def create_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a circuit breaker"""
        if config is None:
            config = CircuitBreakerConfig()
        
        circuit_breaker = CircuitBreaker(name, config)
        self.circuit_breakers[name] = circuit_breaker
        logger.info(f"Created circuit breaker: {name}")
        return circuit_breaker
    
    def resilient_call(self, circuit_breaker_name: str, retry_config: RetryConfig = None):
        """Decorator for resilient function calls"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Get or create circuit breaker
                if circuit_breaker_name not in self.circuit_breakers:
                    self.create_circuit_breaker(circuit_breaker_name)
                
                circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                
                # Execute with circuit breaker and retry
                async def protected_call():
                    return await circuit_breaker.call(func, *args, **kwargs)
                
                if retry_config:
                    return await self.retry_manager.retry_with_backoff(
                        protected_call, f"{circuit_breaker_name}_{func.__name__}", retry_config
                    )
                else:
                    return await protected_call()
            
            return wrapper
        return decorator
    
    async def record_failure(self, failure_type: FailureType, component: str, 
                           error_message: str, context: Dict[str, Any] = None):
        """Record a system failure for analysis"""
        failure_record = FailureRecord(
            failure_id=f"fail_{len(self.failure_records):06d}",
            failure_type=failure_type,
            component=component,
            error_message=error_message,
            context=context or {}
        )
        
        self.failure_records.append(failure_record)
        
        # Keep only recent failure records
        if len(self.failure_records) > 1000:
            self.failure_records = self.failure_records[-1000:]
        
        logger.error(f"Recorded failure in {component}: {error_message}")
        global_metrics.incr(f"failures.{component}.{failure_type.value}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive system health check"""
        health_status = {
            "overall_status": "healthy",
            "timestamp": datetime.now(),
            "components": {}
        }
        
        # Check circuit breakers
        circuit_breaker_issues = 0
        for name, cb in self.circuit_breakers.items():
            cb_stats = cb.get_stats()
            health_status["components"][f"circuit_breaker_{name}"] = {
                "status": "unhealthy" if cb_stats["state"] == "open" else "healthy",
                "details": cb_stats
            }
            if cb_stats["state"] == "open":
                circuit_breaker_issues += 1
        
        # Check recent failures
        recent_failures = [
            f for f in self.failure_records
            if f.timestamp > datetime.now() - timedelta(minutes=10)
        ]
        
        failure_rate = len(recent_failures) / 10.0  # failures per minute
        
        # Check degradation level
        degradation_status = self.graceful_degradation.get_degradation_stats()
        
        health_status["components"]["failure_rate"] = {
            "status": "unhealthy" if failure_rate > 1.0 else "healthy",
            "failures_per_minute": failure_rate,
            "recent_failures": len(recent_failures)
        }
        
        health_status["components"]["degradation"] = {
            "status": "healthy" if degradation_status["current_level"] == "normal" else "degraded",
            "level": degradation_status["current_level"],
            "quality_multiplier": degradation_status["quality_multiplier"]
        }
        
        # Overall health assessment
        if circuit_breaker_issues > 0 or failure_rate > 2.0 or degradation_status["current_level"] in ["minimal", "emergency"]:
            health_status["overall_status"] = "unhealthy"
        elif circuit_breaker_issues > 0 or failure_rate > 0.5 or degradation_status["current_level"] == "reduced":
            health_status["overall_status"] = "degraded"
        
        return health_status
    
    async def get_resilience_report(self) -> Dict[str, Any]:
        """Generate comprehensive resilience report"""
        
        # Circuit breaker stats
        circuit_breaker_stats = {
            name: cb.get_stats() 
            for name, cb in self.circuit_breakers.items()
        }
        
        # Retry stats
        retry_stats = self.retry_manager.get_retry_stats()
        
        # Fallback stats
        fallback_stats = self.fallback_chain.get_fallback_stats()
        
        # Degradation stats
        degradation_stats = self.graceful_degradation.get_degradation_stats()
        
        # Failure analysis
        if self.failure_records:
            failure_types = defaultdict(int)
            failure_components = defaultdict(int)
            
            recent_failures = [
                f for f in self.failure_records
                if f.timestamp > datetime.now() - timedelta(hours=24)
            ]
            
            for failure in recent_failures:
                failure_types[failure.failure_type.value] += 1
                failure_components[failure.component] += 1
            
            failure_analysis = {
                "total_failures_24h": len(recent_failures),
                "failure_types": dict(failure_types),
                "failure_components": dict(failure_components),
                "mttr": self._calculate_mttr(),
                "mtbf": self._calculate_mtbf()
            }
        else:
            failure_analysis = {"total_failures_24h": 0}
        
        return {
            "circuit_breakers": circuit_breaker_stats,
            "retry_manager": retry_stats,
            "fallback_chains": fallback_stats,
            "graceful_degradation": degradation_stats,
            "failure_analysis": failure_analysis,
            "health_check": await self.health_check()
        }
    
    def _calculate_mttr(self) -> float:
        """Calculate Mean Time To Recovery"""
        recovery_times = []
        for failure in self.failure_records:
            if failure.recovered_at:
                recovery_time = (failure.recovered_at - failure.timestamp).total_seconds()
                recovery_times.append(recovery_time)
        
        return sum(recovery_times) / len(recovery_times) if recovery_times else 0.0
    
    def _calculate_mtbf(self) -> float:
        """Calculate Mean Time Between Failures"""
        if len(self.failure_records) < 2:
            return float('inf')
        
        time_between_failures = []
        for i in range(1, len(self.failure_records)):
            time_diff = (self.failure_records[i].timestamp - self.failure_records[i-1].timestamp).total_seconds()
            time_between_failures.append(time_diff)
        
        return sum(time_between_failures) / len(time_between_failures)


# Example usage decorators and utilities
def resilient_api_call(circuit_breaker_name: str, max_retries: int = 3):
    """Convenience decorator for resilient API calls"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            framework = ResilienceFramework()  # In practice, would be injected
            
            @framework.resilient_call(
                circuit_breaker_name,
                RetryConfig(max_attempts=max_retries, base_delay=1.0, exponential_base=2.0)
            )
            async def protected_func():
                return await func(*args, **kwargs)
            
            return await protected_func()
        
        return wrapper
    return decorator


if __name__ == "__main__":
    async def demo_resilience_framework():
        """Demonstrate resilience framework capabilities"""
        framework = ResilienceFramework()
        
        print("=" * 80)
        print("RESILIENCE FRAMEWORK DEMONSTRATION")
        print("=" * 80)
        
        # Create circuit breakers
        api_cb = framework.create_circuit_breaker("api_service", CircuitBreakerConfig(failure_threshold=3))
        db_cb = framework.create_circuit_breaker("database", CircuitBreakerConfig(failure_threshold=2))
        
        # Simulate service calls with failures
        async def flaky_api_service():
            import random
            if random.random() < 0.7:  # 70% failure rate for demo
                raise Exception("API service temporarily unavailable")
            return {"status": "success", "data": "API response"}
        
        async def reliable_fallback():
            return {"status": "fallback", "data": "Cached response"}
        
        # Register fallback chain
        framework.fallback_chain.register_fallback_chain(
            "api_service", 
            flaky_api_service, 
            [reliable_fallback]
        )
        
        print("TESTING CIRCUIT BREAKER AND FALLBACKS:")
        print("-" * 50)
        
        success_count = 0
        for i in range(10):
            try:
                # Try with circuit breaker protection
                result = await api_cb.call(flaky_api_service)
                success_count += 1
                print(f"Call {i+1}: SUCCESS - {result['status']}")
            except Exception as e:
                print(f"Call {i+1}: FAILED - {e}")
                
                # Record failure
                await framework.record_failure(
                    FailureType.API_ERROR, "api_service", str(e)
                )
                
                # Try fallback
                try:
                    fallback_result = await framework.fallback_chain.execute_with_fallback("api_service")
                    print(f"Call {i+1}: FALLBACK SUCCESS - {fallback_result['status']}")
                except Exception as fe:
                    print(f"Call {i+1}: FALLBACK FAILED - {fe}")
        
        # Demonstrate retry mechanism
        print("\\nTESTING RETRY MECHANISM:")
        print("-" * 30)
        
        retry_count = 0
        async def intermittent_service():
            nonlocal retry_count
            retry_count += 1
            if retry_count <= 2:
                raise Exception("Service temporarily down")
            return {"message": f"Success after {retry_count} attempts"}
        
        try:
            result = await framework.retry_manager.retry_with_backoff(
                intermittent_service, 
                "test_service",
                RetryConfig(max_attempts=5, base_delay=0.5)
            )
            print(f"Retry result: {result}")
        except Exception as e:
            print(f"Retry failed: {e}")
        
        # Test graceful degradation
        print("\\nTESTING GRACEFUL DEGRADATION:")
        print("-" * 35)
        
        # Simulate various system health conditions
        health_scenarios = [
            {"cpu_usage": 0.5, "memory_usage": 0.6, "error_rate": 0.01, "response_time": 2.0},
            {"cpu_usage": 0.85, "memory_usage": 0.9, "error_rate": 0.05, "response_time": 7.0},
            {"cpu_usage": 0.95, "memory_usage": 0.95, "error_rate": 0.15, "response_time": 15.0}
        ]
        
        for i, metrics in enumerate(health_scenarios, 1):
            level = framework.graceful_degradation.assess_system_health(metrics)
            quality = framework.graceful_degradation.get_quality_multiplier()
            print(f"Scenario {i}: {level} (quality: {quality:.2f})")
            print(f"  Metrics: CPU {metrics['cpu_usage']:.1%}, Memory {metrics['memory_usage']:.1%}")
            print(f"  Error rate: {metrics['error_rate']:.1%}, Response time: {metrics['response_time']:.1f}s")
        
        # Generate comprehensive report
        print("\\nRESILIENCE REPORT:")
        print("-" * 20)
        
        report = await framework.get_resilience_report()
        
        print(f"Circuit Breakers: {len(report['circuit_breakers'])} configured")
        for name, stats in report['circuit_breakers'].items():
            print(f"  {name}: {stats['state']} (failures: {stats['failure_count']})")
        
        print(f"\\nRetry Manager:")
        retry_stats = report['retry_manager']
        if retry_stats.get('total_operations', 0) > 0:
            print(f"  Total operations: {retry_stats['total_operations']}")
            print(f"  Success rate: {retry_stats['success_rate']:.1%}")
            print(f"  Average attempts: {retry_stats['average_attempts_per_operation']:.1f}")
        else:
            print("  No retry operations recorded")
        
        print(f"\\nFallback Chains:")
        fallback_stats = report['fallback_chains']
        if fallback_stats.get('total_calls', 0) > 0:
            print(f"  Total calls: {fallback_stats['total_calls']}")
            print(f"  Success rate: {fallback_stats['success_rate']:.1%}")
            print(f"  Fallback usage: {fallback_stats['fallback_usage_rate']:.1%}")
        else:
            print("  No fallback calls recorded")
        
        degradation = report['graceful_degradation']
        print(f"\\nSystem Degradation:")
        print(f"  Current level: {degradation['current_level']}")
        print(f"  Quality multiplier: {degradation['quality_multiplier']:.2f}")
        
        failure_analysis = report['failure_analysis']
        print(f"\\nFailure Analysis (24h):")
        print(f"  Total failures: {failure_analysis['total_failures_24h']}")
        if failure_analysis.get('failure_types'):
            print(f"  Failure types: {failure_analysis['failure_types']}")
        
        health_check = report['health_check']
        print(f"\\nOverall Health: {health_check['overall_status'].upper()}")
    
    # Run demonstration
    asyncio.run(demo_resilience_framework())