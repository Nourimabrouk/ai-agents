"""
Shared Kernel Module
Common abstractions and contracts for the AI Agents system
"""

from .interfaces import (
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
)
from .events import EventBus
from .services import (
    BaseMemoryStore,
    BaseMetricsCollector,
    BaseConfigurationProvider,
)

__all__ = [
    # Value Objects
    "SystemState",
    "Priority", 
    "AgentId",
    "TaskId",
    "ExecutionContext",
    "ExecutionResult",
    "DomainEvent",
    
    # Core Interfaces
    "IAgent",
    "IAgentRepository",
    "IOrchestrator",
    "IOrchestrationStrategy",
    "IEventBus",
    
    # Reasoning
    "ReasoningMode",
    "ReasoningRequest", 
    "ReasoningResult",
    "IReasoningEngine",
    
    # Security
    "SecurityLevel",
    "SecurityContext",
    "ISecurityMonitor",
    
    # Infrastructure
    "IMemoryStore",
    "IMetricsCollector", 
    "IConfigurationProvider",
    
    # Service Registry
    "ServiceRegistry",
    "get_service",
    "register_service",
    "register_factory",
    
    # Implementations
    "EventBus",
    "BaseMemoryStore",
    "BaseMetricsCollector",
    "BaseConfigurationProvider",
]