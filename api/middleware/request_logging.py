"""
Request Logging Middleware
Comprehensive request/response logging with performance metrics
"""

import asyncio
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Set
from datetime import datetime
from pathlib import Path

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import get_settings
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)
settings = get_settings()


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Advanced request logging middleware providing:
    - Structured request/response logging
    - Performance metrics tracking
    - Request correlation tracking
    - Error context capture
    - Sensitive data filtering
    - Audit trail generation
    - Performance alerting
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Configuration
        self.log_requests = kwargs.get("log_requests", True)
        self.log_responses = kwargs.get("log_responses", True)
        self.log_request_body = kwargs.get("log_request_body", False)
        self.log_response_body = kwargs.get("log_response_body", False)
        self.max_body_size = kwargs.get("max_body_size", 1000)  # Max chars to log
        self.exclude_paths = set(kwargs.get("exclude_paths", [str(Path("/health").resolve()), str(Path("/metrics").resolve())]))
        self.exclude_headers = set(kwargs.get("exclude_headers", [
            "authorization", "cookie", "x-api-key", "x-csrf-token"
        ]))
        
        # Performance tracking
        self.request_times: List[float] = []
        self.slow_request_threshold = kwargs.get("slow_request_threshold", 2.0)  # seconds
        self.error_count = 0
        self.total_requests = 0
        
        # Sensitive data patterns (regex patterns to filter out)
        self.sensitive_patterns = [
            r'"password":\s*"[^"]*"',
            r'"api_key":\s*"[^"]*"',
            r'"token":\s*"[^"]*"',
            r'"secret":\s*"[^"]*"',
            r'"credit_card":\s*"[^"]*"',
            r'"ssn":\s*"[^"]*"'
        ]
        
        logger.info("Request logging middleware initialized")
    
    async def dispatch(self, request: Request, call_next) -> Response:
        """Main middleware dispatch with comprehensive logging"""
        
        # Skip excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        start_time = time.time()
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        
        # Prepare request context
        request_context = await self._prepare_request_context(request, request_id, start_time)
        
        # Log incoming request
        if self.log_requests:
            self._log_request(request_context)
        
        # Process request and capture response
        response = None
        error = None
        
        try:
            response = await call_next(request)
            
        except Exception as e:
            error = e
            self.error_count += 1
            global_metrics.incr("requests.errors.total")
            
            # Create error response for logging
            response = Response(
                content=json.dumps({"error": "Internal server error"}),
                status_code=500,
                media_type="application/json"
            )
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        self.total_requests += 1
        
        # Track performance metrics
        self._track_performance_metrics(processing_time, response.status_code)
        
        # Prepare response context
        response_context = await self._prepare_response_context(
            response, request_context, processing_time, error
        )
        
        # Log response
        if self.log_responses:
            self._log_response(response_context)
        
        # Log audit trail for important operations
        await self._log_audit_trail(request, response, request_context, response_context)
        
        # Alert on slow requests
        if processing_time > self.slow_request_threshold:
            self._alert_slow_request(request_context, response_context, processing_time)
        
        # Re-raise exception if there was one
        if error:
            raise error
        
        return response
    
    async def _prepare_request_context(self, request: Request, request_id: str, start_time: float) -> Dict[str, Any]:
        """Prepare comprehensive request context for logging"""
        
        # Basic request information
        context = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "url": str(request.url),
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_ip": getattr(request.state, "client_ip", "unknown"),
            "user_agent": request.headers.get("user-agent", ""),
            "content_type": request.headers.get("content-type", ""),
            "content_length": request.headers.get("content-length", 0),
            "headers": self._filter_headers(dict(request.headers)),
            "start_time": start_time
        }
        
        # Add user context if available
        user = getattr(request.state, "current_user", None)
        if user:
            context["user"] = {
                "user_id": getattr(user, "id", "unknown"),
                "username": getattr(user, "username", "unknown"),
                "organization_id": getattr(user, "organization_id", "unknown")
            }
        
        # Add request body if enabled and reasonable size
        if (self.log_request_body and 
            request.method in ["POST", "PUT", "PATCH"] and
            context["content_length"] and 
            int(context["content_length"]) < 10000):
            
            try:
                body = await request.body()
                if body:
                    body_str = body.decode('utf-8', errors='ignore')
                    
                    # Filter sensitive data
                    filtered_body = self._filter_sensitive_data(body_str)
                    
                    # Truncate if too long
                    if len(filtered_body) > self.max_body_size:
                        filtered_body = filtered_body[:self.max_body_size] + "...[truncated]"
                    
                    context["request_body"] = filtered_body
                    
            except Exception as e:
                context["request_body_error"] = f"Failed to read body: {e}"
        
        return context
    
    async def _prepare_response_context(
        self, 
        response: Response, 
        request_context: Dict[str, Any], 
        processing_time: float,
        error: Optional[Exception]
    ) -> Dict[str, Any]:
        """Prepare response context for logging"""
        
        context = {
            "request_id": request_context["request_id"],
            "timestamp": datetime.utcnow().isoformat(),
            "status_code": response.status_code,
            "processing_time_ms": round(processing_time * 1000, 2),
            "response_headers": self._filter_headers(dict(response.headers)),
            "content_length": response.headers.get("content-length", 0)
        }
        
        # Add error information if present
        if error:
            context["error"] = {
                "type": error.__class__.__name__,
                "message": str(error),
                "traceback": self._get_traceback_summary(error)
            }
        
        # Add response body if enabled and not too large
        if (self.log_response_body and 
            hasattr(response, "body") and 
            response.headers.get("content-type", "").startswith("application/json")):
            
            try:
                if hasattr(response, "body"):
                    body_content = response.body
                    if isinstance(body_content, bytes):
                        body_str = body_content.decode('utf-8', errors='ignore')
                    else:
                        body_str = str(body_content)
                    
                    # Filter sensitive data
                    filtered_body = self._filter_sensitive_data(body_str)
                    
                    # Truncate if too long
                    if len(filtered_body) > self.max_body_size:
                        filtered_body = filtered_body[:self.max_body_size] + "...[truncated]"
                    
                    context["response_body"] = filtered_body
                    
            except Exception as e:
                context["response_body_error"] = f"Failed to read response body: {e}"
        
        return context
    
    def _log_request(self, context: Dict[str, Any]):
        """Log incoming request with structured format"""
        
        log_level = "INFO"
        message = f"Incoming request: {context['method']} {context['path']}"
        
        # Use different log levels based on method
        if context["method"] in ["DELETE"]:
            log_level = "WARNING"
        elif context["path"].startswith(str(Path("/admin/").resolve())):
            log_level = "WARNING"
        
        logger.log(
            getattr(logger, log_level.lower())._log_level,
            message,
            extra={
                "event_type": "http_request",
                "request_data": context,
                "component": "request_middleware"
            }
        )
        
        # Update metrics
        global_metrics.incr(f"requests.method.{context['method'].lower()}")
        global_metrics.incr("requests.total")
    
    def _log_response(self, context: Dict[str, Any]):
        """Log response with structured format"""
        
        # Determine log level based on status code
        log_level = "INFO"
        if context["status_code"] >= 400:
            log_level = "ERROR" if context["status_code"] >= 500 else "WARNING"
        
        message = f"Response: {context['status_code']} ({context['processing_time_ms']}ms)"
        
        logger.log(
            getattr(logger, log_level.lower())._log_level,
            message,
            extra={
                "event_type": "http_response",
                "response_data": context,
                "component": "request_middleware"
            }
        )
        
        # Update metrics
        status_class = f"{context['status_code'] // 100}xx"
        global_metrics.incr(f"requests.status.{status_class}")
        
        if context["status_code"] >= 400:
            global_metrics.incr("requests.errors.total")
    
    async def _log_audit_trail(
        self, 
        request: Request, 
        response: Response,
        request_context: Dict[str, Any], 
        response_context: Dict[str, Any]
    ):
        """Log audit trail for important operations"""
        
        # Define operations that require audit logging
        audit_paths = [
            str(Path("/api/v1/auth/login").resolve()),
            str(Path("/api/v1/auth/logout").resolve()), 
            str(Path("/api/v1/documents/process").resolve()),
            str(Path("/api/v1/admin/").resolve())
        ]
        
        # Check if this request needs audit logging
        should_audit = any(
            request.url.path.startswith(path) for path in audit_paths
        )
        
        if not should_audit:
            return {}
        
        # Create audit log entry
        audit_entry = {
            "event_type": "api_audit",
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": request_context["request_id"],
            "user_id": request_context.get("user", {}).get("user_id"),
            "organization_id": request_context.get("user", {}).get("organization_id"),
            "action": f"{request.method} {request.url.path}",
            "client_ip": request_context["client_ip"],
            "user_agent": request_context["user_agent"],
            "status_code": response.status_code,
            "processing_time_ms": response_context["processing_time_ms"],
            "success": response.status_code < 400
        }
        
        # Add specific details based on endpoint
        if request.url.path.startswith(str(Path("/api/v1/documents/process").resolve())):
            audit_entry["document_processing"] = True
            audit_entry["content_type"] = request_context.get("content_type")
            audit_entry["content_length"] = request_context.get("content_length")
        
        logger.info(
            f"Audit: {audit_entry['action']} by user {audit_entry.get('user_id', 'anonymous')}",
            extra={
                "event_type": "audit_log",
                "audit_data": audit_entry,
                "component": "audit_middleware"
            }
        )
    
    def _track_performance_metrics(self, processing_time: float, status_code: int):
        """Track performance metrics"""
        
        # Add to request times list (keep last 1000 requests)
        self.request_times.append(processing_time)
        if len(self.request_times) > 1000:
            self.request_times.pop(0)
        
        # Update global metrics
        global_metrics.histogram("request_duration_seconds", processing_time)
        global_metrics.incr("requests.total")
        
        if status_code >= 400:
            global_metrics.incr("requests.errors.total")
        
        if processing_time > self.slow_request_threshold:
            global_metrics.incr("requests.slow.total")
    
    def _alert_slow_request(
        self, 
        request_context: Dict[str, Any], 
        response_context: Dict[str, Any],
        processing_time: float
    ):
        """Alert on slow requests"""
        
        alert_data = {
            "alert_type": "slow_request",
            "request_id": request_context["request_id"],
            "method": request_context["method"],
            "path": request_context["path"],
            "processing_time_ms": response_context["processing_time_ms"],
            "threshold_ms": self.slow_request_threshold * 1000,
            "status_code": response_context["status_code"],
            "client_ip": request_context["client_ip"],
            "user_id": request_context.get("user", {}).get("user_id")
        }
        
        logger.warning(
            f"Slow request detected: {processing_time:.2f}s for {request_context['method']} {request_context['path']}",
            extra={
                "event_type": "performance_alert",
                "alert_data": alert_data,
                "component": "performance_monitor"
            }
        )
    
    def _filter_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Filter sensitive headers"""
        filtered = {}
        
        for key, value in headers.items():
            if key.lower() in self.exclude_headers:
                filtered[key] = "[FILTERED]"
            else:
                filtered[key] = value
        
        return filtered
    
    def _filter_sensitive_data(self, data: str) -> str:
        """Filter sensitive data from request/response bodies"""
        import re
        
        filtered_data = data
        
        for pattern in self.sensitive_patterns:
            filtered_data = re.sub(pattern, '"[FILTERED]"', filtered_data, flags=re.IGNORECASE)
        
        return filtered_data
    
    def _get_traceback_summary(self, error: Exception) -> str:
        """Get a summary of the error traceback"""
        import traceback
        
        # Get last few lines of traceback
        tb_lines = traceback.format_exception(type(error), error, error.__traceback__)
        
        # Return last 3 lines (usually most relevant)
        return "".join(tb_lines[-3:]).strip()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics summary"""
        
        if not self.request_times:
            return {
                "total_requests": self.total_requests,
                "error_count": self.error_count,
                "avg_response_time_ms": 0,
                "p95_response_time_ms": 0,
                "p99_response_time_ms": 0,
                "error_rate_percent": 0
            }
        
        # Calculate percentiles
        sorted_times = sorted(self.request_times)
        count = len(sorted_times)
        
        avg_time = sum(sorted_times) / count
        p95_time = sorted_times[int(count * 0.95)] if count > 0 else 0
        p99_time = sorted_times[int(count * 0.99)] if count > 0 else 0
        
        error_rate = (self.error_count / max(self.total_requests, 1)) * 100
        
        return {
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "avg_response_time_ms": round(avg_time * 1000, 2),
            "p95_response_time_ms": round(p95_time * 1000, 2),
            "p99_response_time_ms": round(p99_time * 1000, 2),
            "error_rate_percent": round(error_rate, 2),
            "slow_requests": len([t for t in self.request_times if t > self.slow_request_threshold])
        }
    
    def reset_metrics(self):
        """Reset performance metrics (useful for testing)"""
        self.request_times.clear()
        self.error_count = 0
        self.total_requests = 0


# Export middleware class
__all__ = ["RequestLoggingMiddleware"]