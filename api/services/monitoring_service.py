"""
Monitoring and Analytics Service
Comprehensive system monitoring, metrics collection, and analytics
"""

import asyncio
import json
import psutil
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from sqlalchemy import select, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from api.database.session import get_database_session, get_database_health
from api.models.database_models import (
    Document, ProcessingLog, SystemMetrics, AuditLog, 
    BatchProcessing, Organization, User
)
from api.config import get_settings
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)
settings = get_settings()


class HealthStatus(str, Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertLevel(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str]
    unit: str = ""


@dataclass
class SystemAlert:
    """System alert information"""
    alert_id: str
    level: AlertLevel
    title: str
    description: str
    timestamp: datetime
    component: str
    metadata: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MonitoringService:
    """
    Comprehensive monitoring and analytics service
    
    Features:
    - Real-time system health monitoring
    - Performance metrics collection and analysis
    - Usage analytics and reporting
    - Cost tracking and optimization insights
    - Alert generation and management
    - Business intelligence dashboards
    - Predictive analytics
    """
    
    def __init__(self):
        self.settings = settings
        
        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.start_time = datetime.utcnow()
        
        # Metrics storage (in-memory for real-time, database for persistence)
        self.metrics_buffer: deque = deque(maxlen=1000)
        self.performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.alert_history: List[SystemAlert] = []
        
        # Health monitoring
        self.health_checks: Dict[str, callable] = {}
        self.last_health_check = datetime.utcnow()
        
        # Performance tracking
        self.response_times: deque = deque(maxlen=1000)
        self.error_counts: Dict[str, int] = defaultdict(int)
        self.processing_stats: Dict[str, Any] = {
            "total_documents": 0,
            "successful_documents": 0,
            "failed_documents": 0,
            "total_cost": Decimal("0.00"),
            "avg_confidence": 0.0,
            "avg_processing_time": 0.0
        }
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("MonitoringService initialized")
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.is_monitoring:
            logger.warning("Monitoring already started")
            return
        
        self.is_monitoring = True
        
        # Start monitoring task
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Monitoring service started")
    
    async def stop_monitoring(self):
        """Stop monitoring tasks"""
        self.is_monitoring = False
        
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring service stopped")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        try:
            health_results = {}
            overall_healthy = True
            
            # Run all health checks
            for check_name, check_func in self.health_checks.items():
                try:
                    result = await check_func()
                    health_results[check_name] = result
                    
                    if not result.get("healthy", False):
                        overall_healthy = False
                        
                except Exception as e:
                    logger.error(f"Health check {check_name} failed: {e}")
                    health_results[check_name] = {
                        "healthy": False,
                        "error": str(e)
                    }
                    overall_healthy = False
            
            # System uptime
            uptime_seconds = (datetime.utcnow() - self.start_time).total_seconds()
            
            # Performance metrics
            avg_response_time = 0.0
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
            
            return {
                "overall_healthy": overall_healthy,
                "health_checks": health_results,
                "uptime_seconds": int(uptime_seconds),
                "database_connected": health_results.get("database", {}).get("healthy", False),
                "processing_service_active": health_results.get("processing", {}).get("healthy", True),
                "external_services": {
                    name: result.get("healthy", False) 
                    for name, result in health_results.items() 
                    if name.startswith("external_")
                },
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "active_connections": self._get_active_connections(),
                "memory_usage_mb": self._get_memory_usage(),
                "cpu_usage_percent": self._get_cpu_usage()
            }
            
        except Exception as e:
            logger.error(f"Health status check failed: {e}")
            return {
                "overall_healthy": False,
                "error": str(e),
                "uptime_seconds": int((datetime.utcnow() - self.start_time).total_seconds()),
                "database_connected": False,
                "processing_service_active": False,
                "external_services": {},
                "avg_response_time_ms": 0,
                "active_connections": 0,
                "memory_usage_mb": 0,
                "cpu_usage_percent": 0
            }
    
    async def get_comprehensive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        try:
            # Processing metrics
            processing_metrics = await self._get_processing_metrics()
            
            # Performance metrics
            performance_metrics = await self._get_performance_metrics()
            
            # Usage metrics
            usage_metrics = await self._get_usage_metrics()
            
            # Cost metrics
            cost_metrics = await self._get_cost_metrics()
            
            # Error metrics
            error_metrics = await self._get_error_metrics()
            
            # Integration metrics
            integration_metrics = await self._get_integration_metrics()
            
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "processing": processing_metrics,
                "performance": performance_metrics,
                "usage": usage_metrics,
                "cost": cost_metrics,
                "errors": error_metrics,
                "integrations": integration_metrics
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive metrics: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
                "processing": {},
                "performance": {},
                "usage": {},
                "cost": {},
                "errors": {},
                "integrations": {}
            }
    
    async def get_processing_analytics(
        self,
        organization_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        document_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed processing analytics"""
        try:
            # Date range
            if start_date:
                start_dt = datetime.fromisoformat(start_date)
            else:
                start_dt = datetime.utcnow() - timedelta(days=30)
            
            if end_date:
                end_dt = datetime.fromisoformat(end_date)
            else:
                end_dt = datetime.utcnow()
            
            async with get_database_session() as db:
                # Base query
                base_query = select(Document).where(
                    and_(
                        Document.organization_id == organization_id,
                        Document.created_at >= start_dt,
                        Document.created_at <= end_dt
                    )
                )
                
                # Add document type filter if specified
                if document_type:
                    base_query = base_query.where(Document.document_type == document_type)
                
                # Get documents
                result = await db.execute(base_query)
                documents = result.scalars().all()
                
                # Calculate analytics
                analytics = await self._calculate_processing_analytics(documents, start_dt, end_dt)
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get processing analytics: {e}")
            return {"error": str(e)}
    
    async def record_performance_metric(self, metric: PerformanceMetric):
        """Record a performance metric"""
        try:
            # Add to buffer
            self.metrics_buffer.append({
                "name": metric.name,
                "value": metric.value,
                "timestamp": metric.timestamp.isoformat(),
                "tags": metric.tags,
                "unit": metric.unit
            })
            
            # Add to performance history
            self.performance_history[metric.name].append(metric.value)
            
            # Update global metrics
            global_metrics.histogram(metric.name, metric.value)
            
        except Exception as e:
            logger.error(f"Failed to record performance metric: {e}")
    
    async def create_alert(
        self,
        level: AlertLevel,
        title: str,
        description: str,
        component: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SystemAlert:
        """Create and store system alert"""
        try:
            alert = SystemAlert(
                alert_id=f"alert_{int(time.time())}_{len(self.alert_history)}",
                level=level,
                title=title,
                description=description,
                timestamp=datetime.utcnow(),
                component=component,
                metadata=metadata or {}
            )
            
            # Store alert
            self.alert_history.append(alert)
            
            # Log alert
            log_level = "info"
            if level == AlertLevel.WARNING:
                log_level = "warning"
            elif level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                log_level = "error"
            
            getattr(logger, log_level)(
                f"System alert: {title}",
                extra={
                    "alert_id": alert.alert_id,
                    "alert_level": level.value,
                    "component": component,
                    "metadata": metadata
                }
            )
            
            # Send to external alerting system if configured
            await self._send_external_alert(alert)
            
            return alert
            
        except Exception as e:
            logger.error(f"Failed to create alert: {e}")
            raise
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve system alert"""
        try:
            for alert in self.alert_history:
                if alert.alert_id == alert_id and not alert.resolved:
                    alert.resolved = True
                    alert.resolved_at = datetime.utcnow()
                    
                    logger.info(f"Alert resolved: {alert.title}", extra={"alert_id": alert_id})
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to resolve alert: {e}")
            return False
    
    async def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active (unresolved) alerts"""
        return [alert for alert in self.alert_history if not alert.resolved]
    
    # Private methods
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        try:
            while self.is_monitoring:
                # Health checks
                await self._perform_health_checks()
                
                # Collect metrics
                await self._collect_system_metrics()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                # Persist metrics to database
                await self._persist_metrics()
                
                # Clean up old data
                await self._cleanup_old_data()
                
                # Sleep for next cycle
                await asyncio.sleep(settings.monitoring.health_check_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
            # Continue monitoring even if there's an error
            await asyncio.sleep(60)  # Wait before retrying
    
    def _register_default_health_checks(self):
        """Register default health check functions"""
        self.health_checks["database"] = self._check_database_health
        self.health_checks["processing"] = self._check_processing_health
        self.health_checks["memory"] = self._check_memory_health
        self.health_checks["disk"] = self._check_disk_health
        self.health_checks["external_services"] = self._check_external_services
    
    async def _check_database_health(self) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        try:
            db_health = await get_database_health()
            
            # Check connection pool utilization
            utilization = db_health.get("pool_utilization_percent", 0)
            healthy = db_health.get("connection_test", False) and utilization < 80
            
            return {
                "healthy": healthy,
                "connection_test": db_health.get("connection_test", False),
                "pool_utilization": utilization,
                "pool_status": db_health.get("pool_status", {})
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_processing_health(self) -> Dict[str, Any]:
        """Check document processing system health"""
        try:
            # Simple check - just return healthy for now
            # In production, this would check processing queues, worker status, etc.
            return {
                "healthy": True,
                "active_processes": 0,  # Would check actual processing tasks
                "queue_depth": 0       # Would check processing queue
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_memory_health(self) -> Dict[str, Any]:
        """Check system memory usage"""
        try:
            memory = psutil.virtual_memory()
            healthy = memory.percent < 85  # Alert if memory usage > 85%
            
            return {
                "healthy": healthy,
                "usage_percent": memory.percent,
                "available_mb": memory.available // (1024 * 1024),
                "total_mb": memory.total // (1024 * 1024)
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_disk_health(self) -> Dict[str, Any]:
        """Check disk space usage"""
        try:
            disk = psutil.disk_usage('/')
            healthy = (disk.free / disk.total) > 0.1  # Alert if less than 10% free
            
            return {
                "healthy": healthy,
                "usage_percent": (disk.used / disk.total) * 100,
                "free_gb": disk.free // (1024 * 1024 * 1024),
                "total_gb": disk.total // (1024 * 1024 * 1024)
            }
            
        except Exception as e:
            return {"healthy": False, "error": str(e)}
    
    async def _check_external_services(self) -> Dict[str, Any]:
        """Check external service connectivity"""
        # This would check integrations, external APIs, etc.
        return {"healthy": True, "services_checked": 0}
    
    async def _perform_health_checks(self):
        """Perform all health checks and generate alerts if needed"""
        try:
            health_status = await self.get_health_status()
            
            # Check for unhealthy components
            for component, result in health_status.get("health_checks", {}).items():
                if not result.get("healthy", True):
                    # Create alert for unhealthy component
                    await self.create_alert(
                        AlertLevel.ERROR,
                        f"{component.title()} Health Check Failed",
                        f"Health check for {component} failed: {result.get('error', 'Unknown error')}",
                        component,
                        {"health_check_result": result}
                    )
            
            self.last_health_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Health checks failed: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            timestamp = datetime.utcnow()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self.record_performance_metric(PerformanceMetric(
                name="system.cpu.usage_percent",
                value=cpu_percent,
                timestamp=timestamp,
                tags={"component": "system"},
                unit="percent"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            await self.record_performance_metric(PerformanceMetric(
                name="system.memory.usage_percent",
                value=memory.percent,
                timestamp=timestamp,
                tags={"component": "system"},
                unit="percent"
            ))
            
            # Process metrics
            process = psutil.Process()
            await self.record_performance_metric(PerformanceMetric(
                name="process.memory.usage_mb",
                value=process.memory_info().rss / (1024 * 1024),
                timestamp=timestamp,
                tags={"component": "process"},
                unit="mb"
            ))
            
        except Exception as e:
            logger.error(f"System metrics collection failed: {e}")
    
    async def _check_alert_conditions(self):
        """Check conditions that should trigger alerts"""
        try:
            # Check error rate
            total_requests = sum(1 for _ in self.response_times)
            error_count = sum(self.error_counts.values())
            
            if total_requests > 100:  # Only check if we have enough data
                error_rate = (error_count / total_requests) * 100
                
                if error_rate > 10:  # Alert if error rate > 10%
                    await self.create_alert(
                        AlertLevel.WARNING,
                        "High Error Rate Detected",
                        f"Error rate is {error_rate:.1f}% ({error_count}/{total_requests})",
                        "performance",
                        {"error_rate": error_rate, "total_requests": total_requests}
                    )
            
            # Check response time
            if self.response_times:
                avg_response_time = sum(self.response_times) / len(self.response_times)
                
                if avg_response_time > 5.0:  # Alert if avg response time > 5s
                    await self.create_alert(
                        AlertLevel.WARNING,
                        "Slow Response Times",
                        f"Average response time is {avg_response_time:.2f}s",
                        "performance",
                        {"avg_response_time": avg_response_time}
                    )
            
        except Exception as e:
            logger.error(f"Alert condition checking failed: {e}")
    
    async def _persist_metrics(self):
        """Persist metrics to database"""
        try:
            if not self.metrics_buffer:
                return
            
            # Create SystemMetrics records for persistence
            metrics_to_save = list(self.metrics_buffer)
            self.metrics_buffer.clear()
            
            async with get_database_session() as db:
                for metric_data in metrics_to_save:
                    metric = SystemMetrics(
                        timestamp=datetime.fromisoformat(metric_data["timestamp"]),
                        metric_date=datetime.fromisoformat(metric_data["timestamp"]).date(),
                        metric_name=metric_data["name"],
                        metric_type="gauge",  # Default type
                        value=metric_data["value"],
                        tags=metric_data["tags"]
                    )
                    db.add(metric)
                
                await db.commit()
                
        except Exception as e:
            logger.error(f"Metrics persistence failed: {e}")
    
    async def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Keep only recent alerts (last 30 days)
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            self.alert_history = [
                alert for alert in self.alert_history
                if alert.timestamp > cutoff_date
            ]
            
        except Exception as e:
            logger.error(f"Data cleanup failed: {e}")
    
    def _get_active_connections(self) -> int:
        """Get number of active connections"""
        try:
            # This would check database connection pool, HTTP connections, etc.
            return 0  # Placeholder
        except Exception:
            return 0
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        try:
            return psutil.cpu_percent()
        except Exception:
            return 0.0
    
    async def _send_external_alert(self, alert: SystemAlert):
        """Send alert to external alerting system"""
        # This would integrate with services like PagerDuty, Slack, etc.
        pass
    
    # Metric calculation methods
    
    async def _get_processing_metrics(self) -> Dict[str, Any]:
        """Get document processing metrics"""
        try:
            async with get_database_session() as db:
                # Get processing statistics for last 24 hours
                since = datetime.utcnow() - timedelta(hours=24)
                
                result = await db.execute(
                    select(func.count(Document.id)).where(Document.created_at >= since)
                )
                total_docs = result.scalar() or 0
                
                result = await db.execute(
                    select(func.count(Document.id)).where(
                        and_(Document.created_at >= since, Document.processing_status == "completed")
                    )
                )
                completed_docs = result.scalar() or 0
                
                result = await db.execute(
                    select(func.count(Document.id)).where(
                        and_(Document.created_at >= since, Document.processing_status == "failed")
                    )
                )
                failed_docs = result.scalar() or 0
                
                success_rate = (completed_docs / max(total_docs, 1)) * 100
                
                return {
                    "total_documents_24h": total_docs,
                    "completed_documents_24h": completed_docs,
                    "failed_documents_24h": failed_docs,
                    "success_rate_percent": round(success_rate, 2),
                    "documents_per_hour": round(total_docs / 24, 1)
                }
                
        except Exception as e:
            logger.error(f"Processing metrics calculation failed: {e}")
            return {}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            # Response time statistics
            response_times = list(self.response_times)
            if response_times:
                avg_response_time = sum(response_times) / len(response_times)
                sorted_times = sorted(response_times)
                p95_time = sorted_times[int(len(sorted_times) * 0.95)]
                p99_time = sorted_times[int(len(sorted_times) * 0.99)]
            else:
                avg_response_time = p95_time = p99_time = 0
            
            return {
                "avg_response_time_ms": round(avg_response_time * 1000, 2),
                "p95_response_time_ms": round(p95_time * 1000, 2),
                "p99_response_time_ms": round(p99_time * 1000, 2),
                "total_requests": len(response_times),
                "error_count": sum(self.error_counts.values()),
                "memory_usage_mb": self._get_memory_usage(),
                "cpu_usage_percent": self._get_cpu_usage()
            }
            
        except Exception as e:
            logger.error(f"Performance metrics calculation failed: {e}")
            return {}
    
    async def _get_usage_metrics(self) -> Dict[str, Any]:
        """Get usage metrics"""
        # Implementation would get user activity, API calls, etc.
        return {
            "active_users_24h": 0,
            "api_calls_24h": 0,
            "peak_concurrent_users": 0
        }
    
    async def _get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost analysis metrics"""
        # Implementation would calculate costs from processing data
        return {
            "total_cost_24h": 0.0,
            "cost_per_document": 0.0,
            "estimated_monthly_cost": 0.0
        }
    
    async def _get_error_metrics(self) -> Dict[str, Any]:
        """Get error analysis metrics"""
        return {
            "error_rate_percent": 0.0,
            "top_errors": [],
            "error_trend": []
        }
    
    async def _get_integration_metrics(self) -> Dict[str, Any]:
        """Get integration performance metrics"""
        return {
            "active_integrations": 0,
            "successful_posts": 0,
            "failed_posts": 0
        }
    
    async def _calculate_processing_analytics(
        self, 
        documents: List[Document], 
        start_date: datetime, 
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate comprehensive processing analytics"""
        try:
            if not documents:
                return {"message": "No documents found for the specified period"}
            
            total_docs = len(documents)
            successful_docs = len([d for d in documents if d.processing_status == "completed"])
            failed_docs = len([d for d in documents if d.processing_status == "failed"])
            
            # Calculate averages
            confidence_scores = [d.confidence_score for d in documents if d.confidence_score is not None]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
            
            processing_times = [d.processing_duration_ms for d in documents if d.processing_duration_ms is not None]
            avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
            
            # Cost analysis
            costs = [d.processing_cost for d in documents if d.processing_cost is not None]
            total_cost = sum(costs) if costs else Decimal("0.00")
            
            # Document type breakdown
            type_breakdown = {}
            for doc in documents:
                doc_type = doc.document_type or "unknown"
                if doc_type not in type_breakdown:
                    type_breakdown[doc_type] = {"count": 0, "successful": 0, "avg_confidence": 0}
                
                type_breakdown[doc_type]["count"] += 1
                if doc.processing_status == "completed":
                    type_breakdown[doc_type]["successful"] += 1
            
            return {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "days": (end_date - start_date).days
                },
                "summary": {
                    "total_documents": total_docs,
                    "successful_documents": successful_docs,
                    "failed_documents": failed_docs,
                    "success_rate_percent": round((successful_docs / total_docs) * 100, 2),
                    "average_confidence_score": round(avg_confidence, 3),
                    "average_processing_time_ms": round(avg_processing_time, 2),
                    "total_cost": float(total_cost),
                    "cost_per_document": float(total_cost / total_docs) if total_docs > 0 else 0
                },
                "document_type_breakdown": type_breakdown,
                "daily_volume": await self._calculate_daily_volume(documents),
                "performance_trends": await self._calculate_performance_trends(documents)
            }
            
        except Exception as e:
            logger.error(f"Processing analytics calculation failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_daily_volume(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Calculate daily document processing volume"""
        daily_counts = defaultdict(int)
        
        for doc in documents:
            date_key = doc.created_at.date().isoformat()
            daily_counts[date_key] += 1
        
        return [
            {"date": date, "count": count}
            for date, count in sorted(daily_counts.items())
        ]
    
    async def _calculate_performance_trends(self, documents: List[Document]) -> Dict[str, List[float]]:
        """Calculate performance trends over time"""
        # Group by day and calculate averages
        daily_metrics = defaultdict(lambda: {"confidence": [], "processing_time": []})
        
        for doc in documents:
            date_key = doc.created_at.date().isoformat()
            
            if doc.confidence_score is not None:
                daily_metrics[date_key]["confidence"].append(doc.confidence_score)
            
            if doc.processing_duration_ms is not None:
                daily_metrics[date_key]["processing_time"].append(doc.processing_duration_ms)
        
        # Calculate daily averages
        confidence_trend = []
        speed_trend = []
        
        for date in sorted(daily_metrics.keys()):
            metrics = daily_metrics[date]
            
            if metrics["confidence"]:
                avg_confidence = sum(metrics["confidence"]) / len(metrics["confidence"])
                confidence_trend.append(avg_confidence)
            
            if metrics["processing_time"]:
                avg_time = sum(metrics["processing_time"]) / len(metrics["processing_time"])
                speed_trend.append(avg_time)
        
        return {
            "accuracy_trend": confidence_trend,
            "speed_trend": speed_trend
        }


# Export service class
__all__ = ["MonitoringService", "PerformanceMetric", "SystemAlert", "AlertLevel", "HealthStatus"]