"""
Core Interfaces and Protocols - Shared Kernel
Defines all contracts and abstractions to eliminate circular dependencies
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple, Union, Callable, AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import asyncio


# ============================================================================
# CORE DOMAIN VALUE OBJECTS
# ============================================================================

class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SCALING = "scaling"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"
    SHUTDOWN = "shutdown"


class Priority(Enum):
    """Universal priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass(frozen=True)
class AgentId:
    """Unique agent identifier"""
    namespace: str
    name: str
    version: str = "1.0.0"
    
    @property
    def full_id(self) -> str:
        return f"{self.namespace}.{self.name}@{self.version}"


@dataclass(frozen=True)
class TaskId:
    """Unique task identifier"""
    domain: str
    task_type: str
    instance_id: str
    
    @property
    def full_id(self) -> str:
        return f"{self.domain}/{self.task_type}#{self.instance_id}"


@dataclass
class ExecutionContext:
    """Execution context for operations"""
    task_id: TaskId
    agent_id: AgentId
    priority: Priority
    timeout: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ExecutionResult:
    """Result of operation execution"""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================================================
# MESSAGING AND EVENTS
# ============================================================================

@dataclass
class DomainEvent:
    """Base domain event"""
    event_id: str
    event_type: str
    source: AgentId
    timestamp: datetime
    data: Dict[str, Any] = None
    correlation_id: Optional[str] = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}


class IEventBus(ABC):
    """Event bus for domain events"""
    
    @abstractmethod
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def subscribe(self, event_type: str, handler: Callable[[DomainEvent], None]) -> str:
        """Subscribe to event type"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events"""
        logger.info(f'Method {function_name} called')
        return {}


# ============================================================================
# AGENT CONTRACTS
# ============================================================================

class IAgent(ABC):
    """Core agent interface"""
    
    @property
    @abstractmethod
    def agent_id(self) -> AgentId:
        """Get agent identifier"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @property
    @abstractmethod
    def capabilities(self) -> Set[str]:
        """Get agent capabilities"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def execute(self, context: ExecutionContext, task_data: Any) -> ExecutionResult:
        """Execute task"""
        logger.info(f'Processing task: {locals()}')
        return {'success': True, 'message': 'Task processed'}
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check agent health"""
        logger.info(f'Method {function_name} called')
        return {}


class IAgentRepository(ABC):
    """Agent repository for managing agent instances"""
    
    @abstractmethod
    async def register_agent(self, agent: IAgent) -> None:
        """Register agent"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def get_agent(self, agent_id: AgentId) -> Optional[IAgent]:
        """Get agent by ID"""
        return {}
    
    @abstractmethod
    async def find_agents_with_capability(self, capability: str) -> List[IAgent]:
        """Find agents with specific capability"""
        return {}
    
    @abstractmethod
    async def remove_agent(self, agent_id: AgentId) -> None:
        """Remove agent"""
        logger.info(f'Method {function_name} called')
        return {}


# ============================================================================
# ORCHESTRATION CONTRACTS
# ============================================================================

class IOrchestrationStrategy(ABC):
    """Strategy for task orchestration"""
    
    @abstractmethod
    async def plan_execution(self, context: ExecutionContext, task_data: Any) -> List[Tuple[IAgent, Any]]:
        """Plan task execution across agents"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def handle_failure(self, context: ExecutionContext, failed_agent: IAgent, error: Exception) -> bool:
        """Handle execution failure, return True if recovery successful"""
        logger.info(f'Method {function_name} called')
        return {}


class IOrchestrator(ABC):
    """Core orchestrator interface"""
    
    @abstractmethod
    async def execute_task(self, context: ExecutionContext, task_data: Any) -> ExecutionResult:
        """Execute task using available agents"""
        logger.info(f'Processing task: {locals()}')
        return {'success': True, 'message': 'Task processed'}
    
    @abstractmethod
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {}


# ============================================================================
# REASONING CONTRACTS
# ============================================================================

class ReasoningMode(Enum):
    """Reasoning modes"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    PREDICTIVE = "predictive"
    ADAPTIVE = "adaptive"
    COMPREHENSIVE = "comprehensive"
    EMERGENCY = "emergency"


@dataclass
class ReasoningRequest:
    """Request for reasoning operation"""
    mode: ReasoningMode
    problem: str
    context: Dict[str, Any]
    constraints: List[str] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = []


@dataclass
class ReasoningResult:
    """Result of reasoning operation"""
    solution: str
    confidence: float
    reasoning_chain: List[str]
    alternative_solutions: List[str] = None
    
    def __post_init__(self):
        if self.alternative_solutions is None:
            self.alternative_solutions = []


class IReasoningEngine(ABC):
    """Reasoning engine interface"""
    
    @abstractmethod
    async def reason(self, request: ReasoningRequest) -> ReasoningResult:
        """Perform reasoning"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def learn_from_feedback(self, request: ReasoningRequest, result: ReasoningResult, feedback: Dict[str, Any]) -> None:
        """Learn from feedback"""
        logger.info(f'Method {function_name} called')
        return {}


# ============================================================================
# SECURITY CONTRACTS
# ============================================================================

class SecurityLevel(Enum):
    """Security levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class SecurityContext:
    """Security context for operations"""
    level: SecurityLevel
    permissions: Set[str]
    restrictions: Set[str]
    audit_trail: bool = True


class ISecurityMonitor(ABC):
    """Security monitoring interface"""
    
    @abstractmethod
    async def validate_operation(self, context: ExecutionContext, operation: str) -> bool:
        """Validate if operation is allowed"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def get_security_status(self) -> Dict[str, Any]:
        """Get security status"""
        return {}


# ============================================================================
# MEMORY AND PERSISTENCE CONTRACTS
# ============================================================================

class IMemoryStore(ABC):
    """Memory store interface"""
    
    @abstractmethod
    async def store(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store value with optional TTL"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve value by key"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value by key"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """List keys matching pattern"""
        return []


# ============================================================================
# METRICS AND MONITORING CONTRACTS
# ============================================================================

class IMetricsCollector(ABC):
    """Metrics collection interface"""
    
    @abstractmethod
    async def increment_counter(self, name: str, tags: Dict[str, str] = None) -> None:
        """Increment counter metric"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def record_gauge(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record gauge metric"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None) -> None:
        """Record histogram metric"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def get_metrics(self, name_pattern: str = "*") -> Dict[str, Any]:
        """Get metrics matching pattern"""
        return {}


# ============================================================================
# CONFIGURATION CONTRACTS
# ============================================================================

class IConfigurationProvider(ABC):
    """Configuration provider interface"""
    
    @abstractmethod
    async def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return {}
    
    @abstractmethod
    async def set_config(self, key: str, value: Any) -> None:
        """Set configuration value"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration"""
        return {}


# ============================================================================
# SERVICE LOCATOR PATTERN
# ============================================================================

class ServiceRegistry:
    """Service registry for dependency injection"""
    
    def __init__(self):
        self._services: Dict[type, Any] = {}
        self._factories: Dict[type, Callable] = {}
    
    def register_instance(self, interface: type, instance: Any) -> None:
        """Register service instance"""
        self._services[interface] = instance
    
    def register_factory(self, interface: type, factory: Callable) -> None:
        """Register service factory"""
        self._factories[interface] = factory
    
    def get_service(self, interface: type) -> Any:
        """Get service instance"""
        if interface in self._services:
            return self._services[interface]
        elif interface in self._factories:
            instance = self._factories[interface]()
            self._services[interface] = instance
            return instance
        else:
            raise ValueError(f"No service registered for {interface}")
    
    def clear(self) -> None:
        """Clear all services"""
        self._services.clear()
        self._factories.clear()


# Global service registry instance
_global_registry = ServiceRegistry()


def get_service(interface: type) -> Any:
    """Get service from global registry"""
    return _global_registry.get_service(interface)


def register_service(interface: type, instance: Any) -> None:
    """Register service in global registry"""
    _global_registry.register_instance(interface, instance)


def register_factory(interface: type, factory: Callable) -> None:
    """Register factory in global registry"""
    _global_registry.register_factory(interface, factory)