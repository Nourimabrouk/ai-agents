"""
Deployment Service
Manages system deployment, monitoring, and lifecycle management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ...shared import (
    SystemState, AgentId, DomainEvent, IEventBus, get_service,
    IMetricsCollector, IConfigurationProvider
)

logger = logging.getLogger(__name__)


class DeploymentStatus(Enum):
    """Deployment status values"""
    PENDING = "pending"
    DEPLOYING = "deploying"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class HealthStatus(Enum):
    """Health check status values"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class DeploymentConfiguration:
    """Configuration for system deployment"""
    deployment_id: str
    version: str
    environment: str
    services: Dict[str, Any]
    resources: Dict[str, Any]
    scaling: Dict[str, Any]
    monitoring: Dict[str, Any]
    rollback_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServiceHealth:
    """Health status of a service"""
    service_name: str
    status: HealthStatus
    last_check: datetime
    response_time: float
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    active_agents: int
    active_tasks: int
    error_rate: float
    throughput: float


class DeploymentManager:
    """
    Service for managing system deployment and monitoring
    """
    
    def __init__(self):
        self._deployment_config: Optional[DeploymentConfiguration] = None
        self._current_status: DeploymentStatus = DeploymentStatus.STOPPED
        self._system_state: SystemState = SystemState.SHUTDOWN
        self._service_health: Dict[str, ServiceHealth] = {}
        self._metrics_history: List[SystemMetrics] = []
        self._event_bus: Optional[IEventBus] = None
        self._metrics_collector: Optional[IMetricsCollector] = None
        self._config_provider: Optional[IConfigurationProvider] = None
        self._monitoring_active = False
        self._lock = asyncio.Lock()
        
        # Health check configuration
        self._health_check_interval = 30  # seconds
        self._health_check_timeout = 5   # seconds
        
    async def initialize(self) -> None:
        """Initialize deployment manager"""
        try:
            self._event_bus = get_service(IEventBus)
            self._metrics_collector = get_service(IMetricsCollector)
            self._config_provider = get_service(IConfigurationProvider)
        except ValueError:
            logger.warning("Some services not available")
        
        # Load configuration
        await self._load_deployment_config()
        
        logger.info("Deployment manager initialized")
    
    async def deploy_system(self, config: DeploymentConfiguration) -> bool:
        """Deploy system with given configuration"""
        try:
            self._deployment_config = config
            await self._update_status(DeploymentStatus.DEPLOYING)
            
            logger.info(f"Starting deployment {config.deployment_id} version {config.version}")
            
            # Deploy services
            success = await self._deploy_services(config.services)
            
            if success:
                await self._update_status(DeploymentStatus.RUNNING)
                await self._update_system_state(SystemState.RUNNING)
                
                # Start monitoring
                await self._start_monitoring()
                
                logger.info(f"Deployment {config.deployment_id} completed successfully")
                
                if self._event_bus:
                    await self._event_bus.publish(DomainEvent(
                        event_id=f"deployment_success_{config.deployment_id}",
                        event_type="deployment.success",
                        source=AgentId("system", "deployment_manager"),
                        timestamp=datetime.utcnow(),
                        data={
                            "deployment_id": config.deployment_id,
                            "version": config.version,
                            "environment": config.environment
                        }
                    ))
                
                return True
            else:
                await self._update_status(DeploymentStatus.FAILED)
                logger.error(f"Deployment {config.deployment_id} failed")
                
                if self._event_bus:
                    await self._event_bus.publish(DomainEvent(
                        event_id=f"deployment_failed_{config.deployment_id}",
                        event_type="deployment.failed",
                        source=AgentId("system", "deployment_manager"),
                        timestamp=datetime.utcnow(),
                        data={"deployment_id": config.deployment_id}
                    ))
                
                return False
                
        except Exception as e:
            logger.error(f"Error during deployment: {e}")
            await self._update_status(DeploymentStatus.FAILED)
            return False
    
    async def stop_system(self, graceful: bool = True) -> bool:
        """Stop the system"""
        try:
            await self._update_status(DeploymentStatus.STOPPING)
            await self._update_system_state(SystemState.SHUTDOWN)
            
            logger.info(f"Stopping system ({'graceful' if graceful else 'immediate'})")
            
            # Stop monitoring
            await self._stop_monitoring()
            
            # Stop services
            success = await self._stop_services(graceful)
            
            if success:
                await self._update_status(DeploymentStatus.STOPPED)
                logger.info("System stopped successfully")
                
                if self._event_bus:
                    await self._event_bus.publish(DomainEvent(
                        event_id="system_stopped",
                        event_type="deployment.stopped",
                        source=AgentId("system", "deployment_manager"),
                        timestamp=datetime.utcnow(),
                        data={"graceful": graceful}
                    ))
                
                return True
            else:
                logger.error("Error stopping system")
                return False
                
        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")
            return False
    
    async def restart_system(self) -> bool:
        """Restart the system"""
        logger.info("Restarting system")
        
        # Stop system
        stop_success = await self.stop_system(graceful=True)
        if not stop_success:
            logger.error("Failed to stop system for restart")
            return False
        
        # Wait a moment
        await asyncio.sleep(2)
        
        # Start system with existing config
        if self._deployment_config:
            return await self.deploy_system(self._deployment_config)
        else:
            logger.error("No deployment configuration available for restart")
            return False
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        async with self._lock:
            return {
                "deployment_status": self._current_status.value,
                "system_state": self._system_state.value,
                "deployment_config": {
                    "deployment_id": self._deployment_config.deployment_id if self._deployment_config else None,
                    "version": self._deployment_config.version if self._deployment_config else None,
                    "environment": self._deployment_config.environment if self._deployment_config else None
                } if self._deployment_config else None,
                "service_health": {
                    name: {
                        "status": health.status.value,
                        "last_check": health.last_check.isoformat(),
                        "response_time": health.response_time,
                        "issues": health.issues
                    }
                    for name, health in self._service_health.items()
                },
                "monitoring_active": self._monitoring_active,
                "metrics_available": len(self._metrics_history),
                "overall_health": await self._calculate_overall_health()
            }
    
    async def get_system_metrics(self, limit: int = 100) -> List[SystemMetrics]:
        """Get recent system metrics"""
        async with self._lock:
            return self._metrics_history[-limit:] if limit > 0 else self._metrics_history
    
    async def get_service_health(self, service_name: str = None) -> Dict[str, ServiceHealth]:
        """Get health status of services"""
        async with self._lock:
            if service_name:
                return {service_name: self._service_health.get(service_name)}
            else:
                return self._service_health.copy()
    
    async def perform_health_check(self, service_name: str = None) -> Dict[str, ServiceHealth]:
        """Perform health check on services"""
        services_to_check = [service_name] if service_name else list(self._service_health.keys())
        
        for service in services_to_check:
            try:
                start_time = datetime.utcnow()
                
                # Simulate health check (in real implementation, would check actual service)
                await asyncio.sleep(0.1)  # Simulate network call
                is_healthy = True  # Simulate health check result
                
                end_time = datetime.utcnow()
                response_time = (end_time - start_time).total_seconds()
                
                status = HealthStatus.HEALTHY if is_healthy else HealthStatus.UNHEALTHY
                
                health = ServiceHealth(
                    service_name=service,
                    status=status,
                    last_check=end_time,
                    response_time=response_time,
                    details={"check_type": "automated"},
                    issues=[] if is_healthy else ["Service not responding"]
                )
                
                async with self._lock:
                    self._service_health[service] = health
                
            except Exception as e:
                logger.error(f"Error checking health of service {service}: {e}")
                
                health = ServiceHealth(
                    service_name=service,
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.utcnow(),
                    response_time=0.0,
                    issues=[f"Health check error: {str(e)}"]
                )
                
                async with self._lock:
                    self._service_health[service] = health
        
        return await self.get_service_health(service_name)
    
    async def scale_service(self, service_name: str, instances: int) -> bool:
        """Scale service to specified number of instances"""
        logger.info(f"Scaling {service_name} to {instances} instances")
        
        try:
            # Simulate scaling operation
            await asyncio.sleep(1)
            
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id=f"service_scaled_{service_name}",
                    event_type="deployment.service_scaled",
                    source=AgentId("system", "deployment_manager"),
                    timestamp=datetime.utcnow(),
                    data={
                        "service_name": service_name,
                        "instances": instances
                    }
                ))
            
            return True
            
        except Exception as e:
            logger.error(f"Error scaling service {service_name}: {e}")
            return False
    
    async def _load_deployment_config(self) -> None:
        """Load deployment configuration"""
        if self._config_provider:
            try:
                config_data = await self._config_provider.get_config("deployment", {})
                if config_data:
                    self._deployment_config = DeploymentConfiguration(**config_data)
            except Exception as e:
                logger.warning(f"Could not load deployment config: {e}")
    
    async def _deploy_services(self, services: Dict[str, Any]) -> bool:
        """Deploy all services"""
        logger.info(f"Deploying {len(services)} services")
        
        for service_name, service_config in services.items():
            try:
                # Simulate service deployment
                logger.info(f"Deploying service: {service_name}")
                await asyncio.sleep(0.5)  # Simulate deployment time
                
                # Initialize health status
                health = ServiceHealth(
                    service_name=service_name,
                    status=HealthStatus.HEALTHY,
                    last_check=datetime.utcnow(),
                    response_time=0.0
                )
                
                async with self._lock:
                    self._service_health[service_name] = health
                
                logger.info(f"Service {service_name} deployed successfully")
                
            except Exception as e:
                logger.error(f"Failed to deploy service {service_name}: {e}")
                return False
        
        return True
    
    async def _stop_services(self, graceful: bool) -> bool:
        """Stop all services"""
        logger.info(f"Stopping services ({'graceful' if graceful else 'immediate'})")
        
        service_names = list(self._service_health.keys())
        
        for service_name in service_names:
            try:
                logger.info(f"Stopping service: {service_name}")
                
                if graceful:
                    await asyncio.sleep(0.5)  # Simulate graceful shutdown
                
                async with self._lock:
                    if service_name in self._service_health:
                        del self._service_health[service_name]
                
                logger.info(f"Service {service_name} stopped")
                
            except Exception as e:
                logger.error(f"Error stopping service {service_name}: {e}")
                return False
        
        return True
    
    async def _start_monitoring(self) -> None:
        """Start system monitoring"""
        self._monitoring_active = True
        asyncio.create_task(self._monitoring_loop())
        logger.info("System monitoring started")
    
    async def _stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self._monitoring_active = False
        logger.info("System monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                # Collect system metrics
                metrics = await self._collect_system_metrics()
                
                async with self._lock:
                    self._metrics_history.append(metrics)
                    # Keep only last 1000 metrics
                    if len(self._metrics_history) > 1000:
                        self._metrics_history.pop(0)
                
                # Perform health checks
                await self.perform_health_check()
                
                # Check for issues
                await self._check_system_health()
                
                await asyncio.sleep(self._health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self._health_check_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        # Simulate metric collection (in real implementation, would collect actual metrics)
        return SystemMetrics(
            timestamp=datetime.utcnow(),
            cpu_usage=50.0,  # Simulated values
            memory_usage=60.0,
            disk_usage=30.0,
            network_io={"bytes_in": 1000.0, "bytes_out": 800.0},
            active_agents=len(self._service_health),
            active_tasks=10,
            error_rate=0.01,
            throughput=100.0
        )
    
    async def _check_system_health(self) -> None:
        """Check overall system health and trigger alerts if needed"""
        unhealthy_services = [
            name for name, health in self._service_health.items()
            if health.status in [HealthStatus.UNHEALTHY, HealthStatus.DEGRADED]
        ]
        
        if unhealthy_services:
            logger.warning(f"Unhealthy services detected: {unhealthy_services}")
            
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id="health_alert",
                    event_type="deployment.health_alert",
                    source=AgentId("system", "deployment_manager"),
                    timestamp=datetime.utcnow(),
                    data={
                        "unhealthy_services": unhealthy_services,
                        "total_services": len(self._service_health)
                    }
                ))
    
    async def _calculate_overall_health(self) -> str:
        """Calculate overall system health"""
        if not self._service_health:
            return "UNKNOWN"
        
        healthy_count = sum(
            1 for health in self._service_health.values()
            if health.status == HealthStatus.HEALTHY
        )
        
        total_services = len(self._service_health)
        health_ratio = healthy_count / total_services
        
        if health_ratio >= 0.9:
            return "HEALTHY"
        elif health_ratio >= 0.7:
            return "DEGRADED"
        else:
            return "UNHEALTHY"
    
    async def _update_status(self, status: DeploymentStatus) -> None:
        """Update deployment status"""
        old_status = self._current_status
        self._current_status = status
        
        logger.info(f"Deployment status changed: {old_status.value} -> {status.value}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"status_changed_{status.value}",
                event_type="deployment.status_changed",
                source=AgentId("system", "deployment_manager"),
                timestamp=datetime.utcnow(),
                data={
                    "old_status": old_status.value,
                    "new_status": status.value
                }
            ))
    
    async def _update_system_state(self, state: SystemState) -> None:
        """Update system state"""
        old_state = self._system_state
        self._system_state = state
        
        logger.info(f"System state changed: {old_state.value} -> {state.value}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"state_changed_{state.value}",
                event_type="deployment.state_changed",
                source=AgentId("system", "deployment_manager"),
                timestamp=datetime.utcnow(),
                data={
                    "old_state": old_state.value,
                    "new_state": state.value
                }
            ))