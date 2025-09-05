"""
Security Domain Events
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Set

from ...shared import DomainEvent, AgentId


@dataclass 
class SecurityThreatDetected(DomainEvent):
    """Event when security threat is detected"""
    threat_type: str = ""
    threat_level: str = ""
    rule_triggered: str = ""
    blocked_operation: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "security.threat_detected"
        if not hasattr(self, 'event_id'):
            self.event_id = f"threat_detected_{self.threat_type}_{self.timestamp.timestamp()}"


@dataclass
class PermissionsDenied(DomainEvent):
    """Event when permissions are denied"""
    requested_operation: str = ""
    required_permission: str = ""
    current_permissions: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.current_permissions is None:
            self.current_permissions = []
        self.event_type = "security.permissions_denied"
        if not hasattr(self, 'event_id'):
            self.event_id = f"permission_denied_{self.requested_operation}_{self.timestamp.timestamp()}"


@dataclass
class AnomalousBehaviorDetected(DomainEvent):
    """Event when anomalous behavior is detected"""
    behavior_type: str = ""
    anomaly_score: float = 0.0
    typical_pattern: Dict[str, Any] = None
    observed_pattern: Dict[str, Any] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.typical_pattern is None:
            self.typical_pattern = {}
        if self.observed_pattern is None:
            self.observed_pattern = {}
        self.event_type = "security.anomalous_behavior"
        if not hasattr(self, 'event_id'):
            self.event_id = f"anomaly_{self.behavior_type}_{self.timestamp.timestamp()}"


@dataclass
class EmergencyProtocolActivated(DomainEvent):
    """Event when emergency protocol is activated"""
    protocol_type: str = ""
    trigger_reason: str = ""
    threat_level: str = ""
    actions_taken: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.actions_taken is None:
            self.actions_taken = []
        self.event_type = "security.emergency_protocol"
        if not hasattr(self, 'event_id'):
            self.event_id = f"emergency_{self.protocol_type}_{self.timestamp.timestamp()}"


@dataclass
class SecurityPolicyUpdated(DomainEvent):
    """Event when security policy is updated"""
    policy_type: str = ""
    changes: Dict[str, Any] = None
    affected_agents: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.changes is None:
            self.changes = {}
        if self.affected_agents is None:
            self.affected_agents = []
        self.event_type = "security.policy_updated"
        if not hasattr(self, 'event_id'):
            self.event_id = f"policy_updated_{self.policy_type}_{self.timestamp.timestamp()}"


@dataclass
class SecurityAuditCompleted(DomainEvent):
    """Event when security audit is completed"""
    audit_type: str = ""
    findings: List[Dict[str, Any]] = None
    security_score: float = 0.0
    recommendations: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.findings is None:
            self.findings = []
        if self.recommendations is None:
            self.recommendations = []
        self.event_type = "security.audit_completed"
        if not hasattr(self, 'event_id'):
            self.event_id = f"audit_completed_{self.audit_type}_{self.timestamp.timestamp()}"


@dataclass
class BehaviorPatternLearned(DomainEvent):
    """Event when new behavior pattern is learned"""
    pattern_type: str = ""
    confidence: float = 0.0
    operations_learned: Set[str] = None
    time_patterns: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.operations_learned is None:
            self.operations_learned = set()
        if self.time_patterns is None:
            self.time_patterns = []
        self.event_type = "security.pattern_learned"
        if not hasattr(self, 'event_id'):
            self.event_id = f"pattern_learned_{self.pattern_type}_{self.timestamp.timestamp()}"


@dataclass
class ThreatRuleTriggered(DomainEvent):
    """Event when threat detection rule is triggered"""
    rule_name: str = ""
    rule_id: str = ""
    conditions_met: List[str] = None
    actions_executed: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.conditions_met is None:
            self.conditions_met = []
        if self.actions_executed is None:
            self.actions_executed = []
        self.event_type = "security.rule_triggered"
        if not hasattr(self, 'event_id'):
            self.event_id = f"rule_triggered_{self.rule_id}_{self.timestamp.timestamp()}"


# Factory functions for creating security events
def create_threat_detected_event(source: AgentId, threat_type: str, threat_level: str,
                               rule_triggered: str, blocked_operation: str) -> SecurityThreatDetected:
    """Create threat detected event"""
    return SecurityThreatDetected(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        threat_type=threat_type,
        threat_level=threat_level,
        rule_triggered=rule_triggered,
        blocked_operation=blocked_operation
    )


def create_permissions_denied_event(source: AgentId, requested_operation: str,
                                  required_permission: str, 
                                  current_permissions: List[str]) -> PermissionsDenied:
    """Create permissions denied event"""
    return PermissionsDenied(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        requested_operation=requested_operation,
        required_permission=required_permission,
        current_permissions=current_permissions
    )


def create_anomaly_detected_event(source: AgentId, behavior_type: str, anomaly_score: float,
                                typical_pattern: Dict[str, Any],
                                observed_pattern: Dict[str, Any]) -> AnomalousBehaviorDetected:
    """Create anomalous behavior detected event"""
    return AnomalousBehaviorDetected(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        behavior_type=behavior_type,
        anomaly_score=anomaly_score,
        typical_pattern=typical_pattern,
        observed_pattern=observed_pattern
    )


def create_emergency_protocol_event(source: AgentId, protocol_type: str, trigger_reason: str,
                                  threat_level: str, actions_taken: List[str]) -> EmergencyProtocolActivated:
    """Create emergency protocol activated event"""
    return EmergencyProtocolActivated(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        protocol_type=protocol_type,
        trigger_reason=trigger_reason,
        threat_level=threat_level,
        actions_taken=actions_taken
    )


def create_audit_completed_event(source: AgentId, audit_type: str, findings: List[Dict[str, Any]],
                               security_score: float, recommendations: List[str]) -> SecurityAuditCompleted:
    """Create security audit completed event"""
    return SecurityAuditCompleted(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        audit_type=audit_type,
        findings=findings,
        security_score=security_score,
        recommendations=recommendations
    )