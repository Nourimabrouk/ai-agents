"""
Security Middleware
Comprehensive security features for enterprise API
"""

import hashlib
from pathlib import Path
import hmac
import time
import uuid
from typing import Callable, Dict, List, Optional, Set
from datetime import datetime, timedelta

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from api.config import get_settings
from utils.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware providing:
    - Request ID generation and tracking
    - Security headers
    - IP filtering and rate limiting by IP
    - Request validation and sanitization
    - CSRF protection
    - Request size limits
    - Suspicious activity detection
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Security configuration
        self.max_request_size = kwargs.get("max_request_size", 50 * 1024 * 1024)  # 50MB
        self.enable_csrf_protection = kwargs.get("enable_csrf_protection", True)
        self.blocked_ips: Set[str] = set(kwargs.get("blocked_ips", []))
        self.allowed_ips: Optional[Set[str]] = set(kwargs.get("allowed_ips", [])) if kwargs.get("allowed_ips") else None
        
        # Rate limiting by IP (simple in-memory implementation)
        self.ip_requests: Dict[str, List[float]] = {}
        self.rate_limit_window = 60  # 1 minute
        self.rate_limit_requests = 1000  # requests per minute per IP
        
        # Suspicious activity tracking
        self.suspicious_ips: Dict[str, Dict] = {}
        self.suspicious_threshold = 100  # requests per minute to be considered suspicious
        
        # Security headers
        self.security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
            "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
        }
        
        logger.info("Security middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Main middleware dispatch method"""
        start_time = time.time()
        
        try:
            # Generate request ID
            request_id = str(uuid.uuid4())
            request.state.request_id = request_id
            request.state.start_time = start_time
            
            # Get client IP
            client_ip = self._get_client_ip(request)
            request.state.client_ip = client_ip
            
            # Security checks
            security_check = await self._perform_security_checks(request, client_ip)
            if security_check:
                return security_check
            
            # Process request
            response = await call_next(request)
            
            # Add security headers
            self._add_security_headers(response)
            
            # Add request ID to response
            response.headers["X-Request-ID"] = request_id
            
            # Update IP tracking
            self._update_ip_tracking(client_ip)
            
            # Log successful request
            processing_time = (time.time() - start_time) * 1000
            logger.info(
                f"Request completed: {request.method} {request.url.path}",
                extra={
                    "request_id": request_id,
                    "client_ip": client_ip,
                    "processing_time_ms": round(processing_time, 2),
                    "status_code": response.status_code,
                    "user_agent": request.headers.get("user-agent", ""),
                    "content_length": response.headers.get("content-length", 0)
                }
            )
            
            return response
            
        except Exception as e:
            # Log security error
            logger.error(
                f"Security middleware error: {e}",
                extra={
                    "request_id": getattr(request.state, "request_id", "unknown"),
                    "client_ip": getattr(request.state, "client_ip", "unknown"),
                    "path": request.url.path,
                    "method": request.method
                }
            )
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal security error",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )
    
    async def _perform_security_checks(self, request: Request, client_ip: str) -> Optional[JSONResponse]:
        """Perform all security checks and return error response if needed"""
        
        # 1. IP filtering
        if self.blocked_ips and client_ip in self.blocked_ips:
            logger.warning(f"Blocked IP attempted access: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "code": "IP_BLOCKED"}
            )
        
        if self.allowed_ips and client_ip not in self.allowed_ips:
            logger.warning(f"Non-allowed IP attempted access: {client_ip}")
            return JSONResponse(
                status_code=403,
                content={"error": "Access denied", "code": "IP_NOT_ALLOWED"}
            )
        
        # 2. Rate limiting by IP
        if self._is_rate_limited(client_ip):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "code": "RATE_LIMIT_EXCEEDED",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )
        
        # 3. Request size validation
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_request_size:
            logger.warning(f"Request size too large: {content_length} bytes from {client_ip}")
            return JSONResponse(
                status_code=413,
                content={
                    "error": "Request entity too large",
                    "code": "REQUEST_TOO_LARGE",
                    "max_size": self.max_request_size
                }
            )
        
        # 4. Suspicious activity detection
        if self._is_suspicious_activity(client_ip):
            logger.warning(f"Suspicious activity detected from IP: {client_ip}")
            # Don't block immediately, just log and monitor
            await self._record_suspicious_activity(client_ip, request)
        
        # 5. CSRF protection for state-changing methods
        if (self.enable_csrf_protection and 
            request.method in ["POST", "PUT", "DELETE", "PATCH"] and
            not self._validate_csrf_token(request)):
            
            # Skip CSRF for API key authentication
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer ") and "api_key" not in str(request.url):
                logger.warning(f"CSRF token validation failed for {client_ip}")
                return JSONResponse(
                    status_code=403,
                    content={"error": "CSRF token required", "code": "CSRF_REQUIRED"}
                )
        
        # 6. Path traversal protection
        if self._has_path_traversal(request.url.path):
            logger.warning(f"Path traversal attempt from {client_ip}: {request.url.path}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request path", "code": "PATH_TRAVERSAL"}
            )
        
        # 7. SQL injection detection (basic)
        if await self._has_sql_injection_attempt(request):
            logger.warning(f"Potential SQL injection from {client_ip}")
            return JSONResponse(
                status_code=400,
                content={"error": "Invalid request parameters", "code": "INVALID_PARAMETERS"}
            )
        
        # All checks passed
        return {}
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request"""
        # Check various headers for real IP (in order of preference)
        ip_headers = [
            "X-Forwarded-For",
            "X-Real-IP", 
            "X-Client-IP",
            "CF-Connecting-IP",  # Cloudflare
            "True-Client-IP"
        ]
        
        for header in ip_headers:
            ip = request.headers.get(header)
            if ip:
                # Take first IP if comma-separated list
                return ip.split(",")[0].strip()
        
        # Fallback to connection IP
        if hasattr(request, "client") and request.client:
            return request.client.host
        
        return "unknown"
    
    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if IP is rate limited"""
        current_time = time.time()
        
        # Clean old entries
        if client_ip in self.ip_requests:
            self.ip_requests[client_ip] = [
                req_time for req_time in self.ip_requests[client_ip]
                if current_time - req_time < self.rate_limit_window
            ]
        else:
            self.ip_requests[client_ip] = []
        
        # Check rate limit
        if len(self.ip_requests[client_ip]) >= self.rate_limit_requests:
            return True
        
        # Add current request
        self.ip_requests[client_ip].append(current_time)
        return False
    
    def _is_suspicious_activity(self, client_ip: str) -> bool:
        """Detect suspicious activity patterns"""
        current_time = time.time()
        
        # Check request frequency
        if client_ip in self.ip_requests:
            recent_requests = [
                req_time for req_time in self.ip_requests[client_ip]
                if current_time - req_time < 60  # Last minute
            ]
            
            if len(recent_requests) > self.suspicious_threshold:
                return True
        
        return False
    
    async def _record_suspicious_activity(self, client_ip: str, request: Request):
        """Record suspicious activity for analysis"""
        current_time = time.time()
        
        if client_ip not in self.suspicious_ips:
            self.suspicious_ips[client_ip] = {
                "first_seen": current_time,
                "last_seen": current_time,
                "request_count": 0,
                "suspicious_patterns": []
            }
        
        activity = self.suspicious_ips[client_ip]
        activity["last_seen"] = current_time
        activity["request_count"] += 1
        
        # Record pattern
        pattern = {
            "timestamp": current_time,
            "method": request.method,
            "path": request.url.path,
            "user_agent": request.headers.get("user-agent", "")
        }
        
        activity["suspicious_patterns"].append(pattern)
        
        # Keep only recent patterns (last hour)
        activity["suspicious_patterns"] = [
            p for p in activity["suspicious_patterns"]
            if current_time - p["timestamp"] < 3600
        ]
    
    def _validate_csrf_token(self, request: Request) -> bool:
        """Validate CSRF token (basic implementation)"""
        # Skip CSRF for API endpoints (they use other authentication)
        if request.url.path.startswith(str(Path("/api/").resolve())):
            return True
        
        # For web forms, check CSRF token
        csrf_token = request.headers.get("X-CSRF-Token")
        if not csrf_token:
            return False
        
        # Basic CSRF validation (should be more sophisticated in production)
        return len(csrf_token) > 10  # Placeholder validation
    
    def _has_path_traversal(self, path: str) -> bool:
        """Check for path traversal attempts"""
        dangerous_patterns = [
            "../", "..\\",
            "%2e%2e%2f", "%2e%2e%5c",
            "..%2f", "..%5c",
            "%252e%252e%252f", "%252e%252e%255c"
        ]
        
        path_lower = path.lower()
        return any(pattern in path_lower for pattern in dangerous_patterns)
    
    async def _has_sql_injection_attempt(self, request: Request) -> bool:
        """Basic SQL injection detection"""
        sql_patterns = [
            "' or '1'='1", "' or 1=1", "' or true",
            "union select", "drop table", "delete from",
            "insert into", "update set", "exec(",
            "script>", "<script", "javascript:",
            "'; --", "'; #", "str(Path('/*").resolve())
        ]
        
        # Check query parameters
        query_string = str(request.url.query).lower()
        if any(pattern in query_string for pattern in sql_patterns):
            return True
        
        # Check request body (if available and reasonable size)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Only check small bodies to avoid performance impact
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) < 10000:
                    body = await request.body()
                    body_str = body.decode('utf-8', errors='ignore').lower()
                    if any(pattern in body_str for pattern in sql_patterns):
                        return True
            except Exception:
                # If we can't read body, skip this check
        return True
        
        return False
    
    def _add_security_headers(self, response: Response):
        """Add security headers to response"""
        for header, value in self.security_headers.items():
            response.headers[header] = value
    
    def _update_ip_tracking(self, client_ip: str):
        """Update IP tracking for analytics"""
        # This could update database or external monitoring system
        logger.info(f'Method {function_name} called')
        return {}
    
    def get_security_metrics(self) -> Dict:
        """Get security metrics for monitoring"""
        current_time = time.time()
        
        # Count active IPs in last hour
        active_ips = set()
        for ip, requests in self.ip_requests.items():
            recent_requests = [
                req_time for req_time in requests
                if current_time - req_time < 3600
            ]
            if recent_requests:
                active_ips.add(ip)
        
        # Count suspicious IPs
        suspicious_count = len([
            ip for ip, data in self.suspicious_ips.items()
            if current_time - data["last_seen"] < 3600
        ])
        
        return {
            "active_ips_last_hour": len(active_ips),
            "blocked_ips": len(self.blocked_ips),
            "suspicious_ips_last_hour": suspicious_count,
            "total_tracked_ips": len(self.ip_requests)
        }
    
    def block_ip(self, ip: str, reason: str = "Manual block"):
        """Manually block an IP address"""
        self.blocked_ips.add(ip)
        logger.warning(f"IP blocked: {ip} - Reason: {reason}")
    
    def unblock_ip(self, ip: str):
        """Unblock an IP address"""
        self.blocked_ips.discard(ip)
        logger.info(f"IP unblocked: {ip}")


# Rate limiting middleware
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Advanced rate limiting middleware with multiple strategies:
    - Global rate limiting
    - Per-user rate limiting  
    - Per-endpoint rate limiting
    - Sliding window algorithm
    - Token bucket algorithm
    """
    
    def __init__(self, app, **kwargs):
        super().__init__(app)
        
        # Rate limiting configuration
        self.global_rate_limit = kwargs.get("global_rate_limit", 10000)  # requests per minute
        self.user_rate_limit = kwargs.get("user_rate_limit", 1000)      # per user per minute
        self.endpoint_rate_limits = kwargs.get("endpoint_rate_limits", {})  # per endpoint
        
        # Storage for rate limiting data
        self.global_requests: List[float] = []
        self.user_requests: Dict[str, List[float]] = {}
        self.endpoint_requests: Dict[str, List[float]] = {}
        
        # Window configuration
        self.window_size = 60  # 1 minute
        
        logger.info("Rate limiting middleware initialized")
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Rate limiting dispatch"""
        current_time = time.time()
        client_ip = getattr(request.state, "client_ip", "unknown")
        
        try:
            # Check global rate limit
            if self._is_globally_rate_limited(current_time):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Global rate limit exceeded",
                        "code": "GLOBAL_RATE_LIMIT",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Check user rate limit (by IP for now)
            if self._is_user_rate_limited(client_ip, current_time):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "User rate limit exceeded", 
                        "code": "USER_RATE_LIMIT",
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Check endpoint-specific rate limit
            endpoint = f"{request.method}:{request.url.path}"
            if self._is_endpoint_rate_limited(endpoint, current_time):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "Endpoint rate limit exceeded",
                        "code": "ENDPOINT_RATE_LIMIT", 
                        "retry_after": 60
                    },
                    headers={"Retry-After": "60"}
                )
            
            # Update counters
            self._record_request(current_time, client_ip, endpoint)
            
            # Process request
            response = await call_next(request)
            
            # Add rate limit headers
            self._add_rate_limit_headers(response, client_ip, endpoint, current_time)
            
            return response
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            return await call_next(request)  # Continue on rate limiting errors
    
    def _is_globally_rate_limited(self, current_time: float) -> bool:
        """Check global rate limit"""
        self._cleanup_old_requests(self.global_requests, current_time)
        
        if len(self.global_requests) >= self.global_rate_limit:
            return True
        
        self.global_requests.append(current_time)
        return False
    
    def _is_user_rate_limited(self, user_id: str, current_time: float) -> bool:
        """Check per-user rate limit"""
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        
        self._cleanup_old_requests(self.user_requests[user_id], current_time)
        
        if len(self.user_requests[user_id]) >= self.user_rate_limit:
            return True
        
        return False
    
    def _is_endpoint_rate_limited(self, endpoint: str, current_time: float) -> bool:
        """Check endpoint-specific rate limit"""
        if endpoint not in self.endpoint_rate_limits:
            return False
        
        limit = self.endpoint_rate_limits[endpoint]
        
        if endpoint not in self.endpoint_requests:
            self.endpoint_requests[endpoint] = []
        
        self._cleanup_old_requests(self.endpoint_requests[endpoint], current_time)
        
        if len(self.endpoint_requests[endpoint]) >= limit:
            return True
        
        return False
    
    def _record_request(self, current_time: float, user_id: str, endpoint: str):
        """Record request for rate limiting"""
        # User requests
        if user_id not in self.user_requests:
            self.user_requests[user_id] = []
        self.user_requests[user_id].append(current_time)
        
        # Endpoint requests
        if endpoint in self.endpoint_rate_limits:
            if endpoint not in self.endpoint_requests:
                self.endpoint_requests[endpoint] = []
            self.endpoint_requests[endpoint].append(current_time)
    
    def _cleanup_old_requests(self, requests: List[float], current_time: float):
        """Remove requests outside the time window"""
        cutoff_time = current_time - self.window_size
        # Remove old requests in-place
        while requests and requests[0] < cutoff_time:
            requests.pop(0)
    
    def _add_rate_limit_headers(self, response: Response, user_id: str, endpoint: str, current_time: float):
        """Add rate limiting headers to response"""
        # Global rate limit headers
        global_remaining = max(0, self.global_rate_limit - len(self.global_requests))
        response.headers["X-RateLimit-Global-Limit"] = str(self.global_rate_limit)
        response.headers["X-RateLimit-Global-Remaining"] = str(global_remaining)
        
        # User rate limit headers
        if user_id in self.user_requests:
            user_remaining = max(0, self.user_rate_limit - len(self.user_requests[user_id]))
            response.headers["X-RateLimit-User-Limit"] = str(self.user_rate_limit)
            response.headers["X-RateLimit-User-Remaining"] = str(user_remaining)
        
        # Reset time
        reset_time = int(current_time + self.window_size)
        response.headers["X-RateLimit-Reset"] = str(reset_time)


# Export middleware classes
__all__ = ["SecurityMiddleware", "RateLimitMiddleware"]