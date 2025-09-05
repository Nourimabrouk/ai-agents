"""
Import bridge for orchestrator components - Phase 7 Enhanced
Provides access to the sophisticated orchestrator implementation with autonomous intelligence
"""

# Import from the actual implementation
from core.orchestration.orchestrator import (
    AgentOrchestrator,
    Task,
    CommunicationProtocol,
    Message,
    Blackboard,
    CustomerSupportAgent,
    DataAnalystAgent,
    ClaudeCodeAgent,
    CodeReviewAgent
)

# Import advanced coordination
from core.coordination.advanced_orchestrator import AdvancedOrchestrator

# Import Phase 7 Autonomous Intelligence components
from core.autonomous.orchestrator import AutonomousMetaOrchestrator, AutonomyLevel
from core.autonomous.self_modification import SelfModifyingAgent, DynamicCodeGenerator, PerformanceDrivenEvolution
from core.autonomous.emergent_intelligence import (
    EmergentIntelligenceOrchestrator, 
    CapabilityMiningEngine, 
    NoveltyDetector,
    InnovationIncubator
)
from core.autonomous.safety import (
    AutonomousSafetyFramework, 
    ModificationValidator, 
    RollbackManager,
    SafetyLevel
)

# Re-export for backward compatibility and new Phase 7 features
__all__ = [
    # Original components
    'AgentOrchestrator',
    'Task',
    'CommunicationProtocol', 
    'Message',
    'Blackboard',
    'CustomerSupportAgent',
    'DataAnalystAgent',
    'ClaudeCodeAgent',
    'CodeReviewAgent',
    # Advanced coordination
    'AdvancedOrchestrator',
    # Phase 7 Autonomous Intelligence
    'AutonomousMetaOrchestrator',
    'AutonomyLevel',
    'SelfModifyingAgent',
    'DynamicCodeGenerator',
    'PerformanceDrivenEvolution',
    'EmergentIntelligenceOrchestrator',
    'CapabilityMiningEngine',
    'NoveltyDetector',
    'InnovationIncubator',
    'AutonomousSafetyFramework',
    'ModificationValidator',
    'RollbackManager',
    'SafetyLevel'
]