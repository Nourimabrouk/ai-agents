"""
Autonomous Intelligence Domain
Provides autonomous decision-making and self-modification capabilities
"""

from .services import (
    AutonomousIntelligenceService,
    AutonomyLevel,
    AutonomousCapability,
    AutonomousDecision
)
from .repositories import InMemoryAgentRepository
from .events import (
    AutonomousCapabilityRegistered,
    AutonomyLevelChanged,
    AutonomousDecisionRequested,
    AutonomousDecisionApproved,
    AutonomousDecisionExecuted,
    EmergentBehaviorDetected,
    SelfModificationAttempted
)

__all__ = [
    # Services
    "AutonomousIntelligenceService",
    "AutonomyLevel",
    "AutonomousCapability",
    "AutonomousDecision",
    
    # Repositories
    "InMemoryAgentRepository",
    
    # Events
    "AutonomousCapabilityRegistered",
    "AutonomyLevelChanged",
    "AutonomousDecisionRequested", 
    "AutonomousDecisionApproved",
    "AutonomousDecisionExecuted",
    "EmergentBehaviorDetected",
    "SelfModificationAttempted"
]