"""
Autonomous Intelligence Domain Events
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from ...shared import DomainEvent, AgentId


@dataclass
class AutonomousCapabilityRegistered(DomainEvent):
    """Event when new autonomous capability is registered"""
    capability_name: str = ""
    autonomy_level: str = ""
    risk_level: int = 0
    
    def __post_init__(self):
        self.event_type = "autonomous.capability_registered"
        if not hasattr(self, 'event_id'):
            self.event_id = f"capability_registered_{self.capability_name}_{self.timestamp.timestamp()}"


@dataclass
class AutonomyLevelChanged(DomainEvent):
    """Event when agent autonomy level changes"""
    old_level: str = ""
    new_level: str = ""
    
    def __post_init__(self):
        self.event_type = "autonomous.level_changed"
        if not hasattr(self, 'event_id'):
            self.event_id = f"level_changed_{self.source.full_id}_{self.timestamp.timestamp()}"


@dataclass
class AutonomousDecisionRequested(DomainEvent):
    """Event when autonomous decision is requested"""
    capability: str = ""
    confidence: float = 0.0
    risk_level: int = 0
    approval_required: bool = False
    
    def __post_init__(self):
        self.event_type = "autonomous.decision_requested" 
        if not hasattr(self, 'event_id'):
            self.event_id = f"decision_requested_{self.capability}_{self.timestamp.timestamp()}"


@dataclass
class AutonomousDecisionApproved(DomainEvent):
    """Event when autonomous decision is approved"""
    decision_id: str = ""
    approved: bool = False
    
    def __post_init__(self):
        self.event_type = "autonomous.decision_approved" if self.approved else "autonomous.decision_rejected"
        if not hasattr(self, 'event_id'):
            self.event_id = f"decision_approved_{self.decision_id}_{self.timestamp.timestamp()}"


@dataclass
class AutonomousDecisionExecuted(DomainEvent):
    """Event when autonomous decision is executed"""
    decision_id: str = ""
    capability: str = ""
    success: bool = False
    execution_time: float = 0.0
    
    def __post_init__(self):
        self.event_type = "autonomous.decision_executed"
        if not hasattr(self, 'event_id'):
            self.event_id = f"decision_executed_{self.decision_id}_{self.timestamp.timestamp()}"


@dataclass
class EmergentBehaviorDetected(DomainEvent):
    """Event when emergent behavior is detected"""
    behavior_pattern: str = ""
    novelty_score: float = 0.0
    agents_involved: list = None
    
    def __post_init__(self):
        if self.agents_involved is None:
            self.agents_involved = []
        self.event_type = "autonomous.emergent_behavior"
        if not hasattr(self, 'event_id'):
            self.event_id = f"emergent_behavior_{self.behavior_pattern}_{self.timestamp.timestamp()}"


@dataclass
class SelfModificationAttempted(DomainEvent):
    """Event when agent attempts self-modification"""
    modification_type: str = ""
    description: str = ""
    safety_validated: bool = False
    
    def __post_init__(self):
        self.event_type = "autonomous.self_modification_attempted"
        if not hasattr(self, 'event_id'):
            self.event_id = f"self_modification_{self.modification_type}_{self.timestamp.timestamp()}"


# Factory functions for creating autonomous events
def create_capability_registered_event(source: AgentId, capability_name: str, 
                                     autonomy_level: str, risk_level: int) -> AutonomousCapabilityRegistered:
    """Create capability registered event"""
    return AutonomousCapabilityRegistered(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        capability_name=capability_name,
        autonomy_level=autonomy_level,
        risk_level=risk_level
    )


def create_autonomy_level_changed_event(source: AgentId, old_level: str, 
                                      new_level: str) -> AutonomyLevelChanged:
    """Create autonomy level changed event"""
    return AutonomyLevelChanged(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        old_level=old_level,
        new_level=new_level
    )


def create_decision_requested_event(source: AgentId, capability: str, confidence: float,
                                  risk_level: int, approval_required: bool) -> AutonomousDecisionRequested:
    """Create decision requested event"""
    return AutonomousDecisionRequested(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        capability=capability,
        confidence=confidence,
        risk_level=risk_level,
        approval_required=approval_required
    )


def create_emergent_behavior_event(source: AgentId, behavior_pattern: str, 
                                 novelty_score: float, agents_involved: list) -> EmergentBehaviorDetected:
    """Create emergent behavior detected event"""
    return EmergentBehaviorDetected(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        behavior_pattern=behavior_pattern,
        novelty_score=novelty_score,
        agents_involved=agents_involved
    )