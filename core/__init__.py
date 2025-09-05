"""
Core Module - AI Agents Architecture
Production-grade microservice architecture with clean separation of concerns
"""

from typing import Any, Dict
from datetime import datetime

# Shared kernel - Foundation for all domains
from .shared import (
    # Value Objects
    SystemState,
    Priority,
    AgentId,
    TaskId,
    ExecutionContext,
    ExecutionResult,
    DomainEvent,
    
    # Core Interfaces
    IAgent,
    IAgentRepository,
    IOrchestrator,
    IOrchestrationStrategy,
    IEventBus,
    
    # Reasoning
    ReasoningMode,
    ReasoningRequest,
    ReasoningResult,
    IReasoningEngine,
    
    # Security
    SecurityLevel,
    SecurityContext,
    ISecurityMonitor,
    
    # Infrastructure
    IMemoryStore,
    IMetricsCollector,
    IConfigurationProvider,
    
    # Service Registry
    ServiceRegistry,
    get_service,
    register_service,
    register_factory,
    
    # Base Implementations
    EventBus,
    BaseMemoryStore,
    BaseMetricsCollector,
    BaseConfigurationProvider,
)

# Domain Services
from .autonomous import (
    AutonomousIntelligenceService,
    AutonomyLevel,
    AutonomousCapability,
    AutonomousDecision,
    InMemoryAgentRepository
)

from .reasoning import (
    ReasoningOrchestrationService,
    ReasoningStrategy
)

from .security import (
    SecurityMonitoringService,
    ThreatLevel,
    SecurityEventType,
    SecurityEvent,
    BehaviorPattern,
    ThreatRule
)

from .integration import (
    OrchestrationService,
    OrchestrationPattern,
    OrchestrationPlan,
    OrchestrationResult,
    DeploymentManager,
    DeploymentStatus,
    HealthStatus,
    DeploymentConfiguration,
    ServiceHealth,
    SystemMetrics
)

__all__ = [
    # Shared Kernel
    "SystemState",
    "Priority",
    "AgentId",
    "TaskId",
    "ExecutionContext",
    "ExecutionResult",
    "DomainEvent",
    "IAgent",
    "IAgentRepository",
    "IOrchestrator",
    "IOrchestrationStrategy",
    "IEventBus",
    "ReasoningMode",
    "ReasoningRequest",
    "ReasoningResult",
    "IReasoningEngine",
    "SecurityLevel",
    "SecurityContext",
    "ISecurityMonitor",
    "IMemoryStore",
    "IMetricsCollector",
    "IConfigurationProvider",
    "ServiceRegistry",
    "get_service",
    "register_service",
    "register_factory",
    "EventBus",
    "BaseMemoryStore",
    "BaseMetricsCollector",
    "BaseConfigurationProvider",
    
    # Autonomous Intelligence Domain
    "AutonomousIntelligenceService",
    "AutonomyLevel",
    "AutonomousCapability",
    "AutonomousDecision",
    "InMemoryAgentRepository",
    
    # Reasoning Domain
    "ReasoningOrchestrationService",
    "ReasoningStrategy",
    
    # Security Domain
    "SecurityMonitoringService",
    "ThreatLevel",
    "SecurityEventType",
    "SecurityEvent",
    "BehaviorPattern",
    "ThreatRule",
    
    # Integration Domain
    "OrchestrationService",
    "OrchestrationPattern",
    "OrchestrationPlan",
    "OrchestrationResult",
    "DeploymentManager",
    "DeploymentStatus",
    "HealthStatus",
    "DeploymentConfiguration",
    "ServiceHealth",
    "SystemMetrics"
]


# Architecture Initialization
async def initialize_system() -> None:
    """
    Initialize the entire AI Agents system with proper dependency injection
    """
    import logging
    logger = logging.getLogger(__name__)
    
    # Initialize service registry
    logger.info("Initializing AI Agents System...")
    
    # Register infrastructure services
    register_service(IEventBus, EventBus())
    register_service(IMemoryStore, BaseMemoryStore())
    register_service(IMetricsCollector, BaseMetricsCollector())
    register_service(IConfigurationProvider, BaseConfigurationProvider())
    
    # Register domain services
    register_service(IAgentRepository, InMemoryAgentRepository())
    register_service(ISecurityMonitor, SecurityMonitoringService())
    register_service(IOrchestrator, OrchestrationService())
    
    # Register specialized services
    autonomous_service = AutonomousIntelligenceService()
    reasoning_service = ReasoningOrchestrationService()
    deployment_manager = DeploymentManager()
    
    register_service(AutonomousIntelligenceService, autonomous_service)
    register_service(ReasoningOrchestrationService, reasoning_service)
    register_service(DeploymentManager, deployment_manager)
    
    # Initialize all services
    await get_service(ISecurityMonitor).initialize()
    await autonomous_service.initialize()
    await reasoning_service.initialize()
    await get_service(IOrchestrator).initialize()
    await deployment_manager.initialize()
    
    logger.info("AI Agents System initialized successfully")


async def shutdown_system() -> None:
    """
    Gracefully shutdown the entire system
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("Shutting down AI Agents System...")
    
    try:
        # Shutdown deployment manager first
        deployment_manager = get_service(DeploymentManager)
        await deployment_manager.stop_system(graceful=True)
        
        # Shutdown security monitoring
        security_monitor = get_service(ISecurityMonitor)
        if hasattr(security_monitor, 'shutdown'):
            await security_monitor.shutdown()
        
        logger.info("AI Agents System shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during system shutdown: {e}")


# System Status and Health Checks
async def get_system_health() -> Dict[str, Any]:
    """
    Get comprehensive system health status
    """
    try:
        orchestrator = get_service(IOrchestrator)
        security_monitor = get_service(ISecurityMonitor)
        deployment_manager = get_service(DeploymentManager)
        
        return {
            "overall_status": "healthy",
            "orchestrator": await orchestrator.get_system_status(),
            "security": await security_monitor.get_security_status(),
            "deployment": await deployment_manager.get_system_status(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


# Version information
__version__ = "1.0.0"
__architecture_version__ = "production-microservices-v1"
__description__ = "Production-grade AI Agents system with microservice architecture"