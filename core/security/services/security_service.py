"""
Security Service
Provides comprehensive security monitoring, threat detection, and behavioral analysis
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib

from ...shared import (
    ISecurityMonitor, SecurityLevel, SecurityContext, AgentId, 
    ExecutionContext, DomainEvent, IEventBus, get_service
)

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels"""
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class SecurityEventType(Enum):
    """Types of security events"""
    ACCESS_ATTEMPT = "access_attempt"
    PERMISSION_DENIED = "permission_denied"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    THREAT_DETECTED = "threat_detected"
    SECURITY_VIOLATION = "security_violation"
    EMERGENCY_RESPONSE = "emergency_response"


@dataclass
class SecurityEvent:
    """Security event record"""
    event_id: str
    event_type: SecurityEventType
    agent_id: AgentId
    timestamp: datetime
    threat_level: ThreatLevel
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


@dataclass
class BehaviorPattern:
    """Agent behavior pattern"""
    agent_id: AgentId
    pattern_type: str
    frequency: float
    typical_times: List[str]
    typical_operations: Set[str]
    confidence: float
    
    
@dataclass
class ThreatRule:
    """Security threat detection rule"""
    rule_id: str
    name: str
    description: str
    threat_level: ThreatLevel
    conditions: Dict[str, Any]
    actions: List[str]
    enabled: bool = True


class SecurityMonitoringService(ISecurityMonitor):
    """
    Comprehensive security monitoring service
    Provides threat detection, behavioral analysis, and emergency response
    """
    
    def __init__(self):
        self._security_events: List[SecurityEvent] = []
        self._behavior_patterns: Dict[AgentId, BehaviorPattern] = {}
        self._threat_rules: Dict[str, ThreatRule] = {}
        self._permissions: Dict[AgentId, Set[str]] = {}
        self._restrictions: Dict[AgentId, Set[str]] = {}
        self._emergency_protocols: Dict[ThreatLevel, List[Callable]] = {}
        self._event_bus: Optional[IEventBus] = None
        self._lock = asyncio.Lock()
        self._monitoring_active = False
        
        # Performance tracking
        self._operations_count = 0
        self._blocked_operations = 0
        self._threats_detected = 0
    
    async def initialize(self) -> None:
        """Initialize the security service"""
        try:
            self._event_bus = get_service(IEventBus)
        except ValueError:
            logger.warning("EventBus not available, running without events")
        
        # Register built-in threat rules
        await self._register_builtin_threat_rules()
        
        # Start monitoring
        self._monitoring_active = True
        asyncio.create_task(self._background_monitoring())
        
        logger.info("Security monitoring service initialized")
    
    async def validate_operation(self, context: ExecutionContext, operation: str) -> bool:
        """Validate if operation is allowed for agent"""
        self._operations_count += 1
        
        async with self._lock:
            agent_id = context.agent_id
            
            # Check basic permissions
            if agent_id in self._permissions:
                if operation not in self._permissions[agent_id]:
                    await self._log_security_event(
                        SecurityEventType.PERMISSION_DENIED,
                        agent_id,
                        ThreatLevel.MEDIUM,
                        f"Operation {operation} not permitted for agent {agent_id.full_id}"
                    )
                    self._blocked_operations += 1
                    return False
            
            # Check restrictions
            if agent_id in self._restrictions:
                if operation in self._restrictions[agent_id]:
                    await self._log_security_event(
                        SecurityEventType.PERMISSION_DENIED,
                        agent_id,
                        ThreatLevel.HIGH,
                        f"Operation {operation} explicitly restricted for agent {agent_id.full_id}"
                    )
                    self._blocked_operations += 1
                    return False
            
            # Check against threat detection rules
            threat_detected = await self._check_threat_rules(context, operation)
            if threat_detected:
                self._blocked_operations += 1
                return False
            
            # Log successful access
            await self._log_security_event(
                SecurityEventType.ACCESS_ATTEMPT,
                agent_id,
                ThreatLevel.NONE,
                f"Operation {operation} approved for agent {agent_id.full_id}"
            )
            
            # Update behavior patterns
            await self._update_behavior_pattern(agent_id, operation)
            
            return True
    
    async def log_security_event(self, event_type: str, details: Dict[str, Any]) -> None:
        """Log security event"""
        agent_id = AgentId(
            details.get("namespace", "unknown"),
            details.get("agent_name", "unknown")
        )
        
        await self._log_security_event(
            SecurityEventType(event_type),
            agent_id,
            ThreatLevel(details.get("threat_level", ThreatLevel.LOW.value)),
            details.get("description", "Security event"),
            details.get("metadata", {})
        )
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        async with self._lock:
            return {
                "monitoring_active": self._monitoring_active,
                "total_events": len(self._security_events),
                "recent_events": len([e for e in self._security_events 
                                    if e.timestamp > datetime.utcnow() - timedelta(hours=24)]),
                "operations_count": self._operations_count,
                "blocked_operations": self._blocked_operations,
                "threats_detected": self._threats_detected,
                "active_threat_rules": len([r for r in self._threat_rules.values() if r.enabled]),
                "monitored_agents": len(self._behavior_patterns),
                "threat_level_distribution": self._get_threat_level_distribution(),
                "security_score": self._calculate_security_score()
            }
    
    async def set_agent_permissions(self, agent_id: AgentId, permissions: Set[str]) -> None:
        """Set permissions for an agent"""
        async with self._lock:
            self._permissions[agent_id] = permissions.copy()
        
        logger.info(f"Set permissions for {agent_id.full_id}: {permissions}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"permissions_set_{agent_id.full_id}",
                event_type="security.permissions_set",
                source=agent_id,
                timestamp=datetime.utcnow(),
                data={"permissions": list(permissions)}
            ))
    
    async def add_agent_restriction(self, agent_id: AgentId, restriction: str) -> None:
        """Add restriction for an agent"""
        async with self._lock:
            if agent_id not in self._restrictions:
                self._restrictions[agent_id] = set()
            self._restrictions[agent_id].add(restriction)
        
        logger.warning(f"Added restriction for {agent_id.full_id}: {restriction}")
        
        await self._log_security_event(
            SecurityEventType.SECURITY_VIOLATION,
            agent_id,
            ThreatLevel.MEDIUM,
            f"Restriction added: {restriction}"
        )
    
    async def register_threat_rule(self, rule: ThreatRule) -> None:
        """Register new threat detection rule"""
        async with self._lock:
            self._threat_rules[rule.rule_id] = rule
        
        logger.info(f"Registered threat rule: {rule.name}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"threat_rule_registered_{rule.rule_id}",
                event_type="security.threat_rule_registered",
                source=AgentId("system", "security_service"),
                timestamp=datetime.utcnow(),
                data={"rule_name": rule.name, "threat_level": rule.threat_level.value}
            ))
    
    async def register_emergency_protocol(self, threat_level: ThreatLevel, 
                                        protocol: Callable) -> None:
        """Register emergency response protocol"""
        if threat_level not in self._emergency_protocols:
            self._emergency_protocols[threat_level] = []
        
        self._emergency_protocols[threat_level].append(protocol)
        logger.info(f"Registered emergency protocol for threat level: {threat_level.value}")
    
    async def get_security_events(self, agent_id: Optional[AgentId] = None,
                                threat_level: Optional[ThreatLevel] = None,
                                since: Optional[datetime] = None,
                                limit: int = 100) -> List[SecurityEvent]:
        """Get security events with filtering"""
        async with self._lock:
            events = self._security_events.copy()
            
            # Apply filters
            if agent_id:
                events = [e for e in events if e.agent_id == agent_id]
            
            if threat_level:
                events = [e for e in events if e.threat_level == threat_level]
            
            if since:
                events = [e for e in events if e.timestamp >= since]
            
            # Sort by timestamp (newest first) and limit
            events.sort(key=lambda x: x.timestamp, reverse=True)
            return events[:limit]
    
    async def get_behavior_analysis(self, agent_id: AgentId) -> Optional[Dict[str, Any]]:
        """Get behavioral analysis for an agent"""
        async with self._lock:
            pattern = self._behavior_patterns.get(agent_id)
            
            if not pattern:
                return {}
            
            # Calculate anomaly score
            recent_events = [e for e in self._security_events 
                           if e.agent_id == agent_id and 
                           e.timestamp > datetime.utcnow() - timedelta(hours=24)]
            
            anomaly_score = self._calculate_anomaly_score(agent_id, recent_events)
            
            return {
                "pattern_type": pattern.pattern_type,
                "frequency": pattern.frequency,
                "typical_times": pattern.typical_times,
                "typical_operations": list(pattern.typical_operations),
                "confidence": pattern.confidence,
                "anomaly_score": anomaly_score,
                "recent_events_count": len(recent_events),
                "risk_level": self._assess_risk_level(anomaly_score)
            }
    
    async def _register_builtin_threat_rules(self) -> None:
        """Register built-in threat detection rules"""
        builtin_rules = [
            ThreatRule(
                rule_id="rapid_operations",
                name="Rapid Operations Detection",
                description="Detect unusually rapid operation sequences",
                threat_level=ThreatLevel.MEDIUM,
                conditions={"operations_per_minute": 100},
                actions=["alert", "throttle"]
            ),
            ThreatRule(
                rule_id="unauthorized_access",
                name="Unauthorized Access Attempt",
                description="Detect attempts to access restricted resources",
                threat_level=ThreatLevel.HIGH,
                conditions={"operation_type": "restricted"},
                actions=["block", "alert", "log"]
            ),
            ThreatRule(
                rule_id="anomalous_behavior",
                name="Anomalous Behavior Pattern",
                description="Detect behavior that deviates significantly from normal patterns",
                threat_level=ThreatLevel.MEDIUM,
                conditions={"anomaly_score": 0.8},
                actions=["alert", "analyze"]
            ),
            ThreatRule(
                rule_id="self_modification_attempt",
                name="Self-Modification Attempt",
                description="Detect attempts at unauthorized self-modification",
                threat_level=ThreatLevel.CRITICAL,
                conditions={"operation_type": "self_modify"},
                actions=["block", "emergency_stop", "alert"]
            )
        ]
        
        for rule in builtin_rules:
            await self.register_threat_rule(rule)
    
    async def _check_threat_rules(self, context: ExecutionContext, operation: str) -> bool:
        """Check if operation triggers any threat detection rules"""
        for rule in self._threat_rules.values():
            if not rule.enabled:
                continue
            
            if await self._evaluate_threat_rule(rule, context, operation):
                await self._handle_threat_detection(rule, context, operation)
                return True
        
        return False
    
    async def _evaluate_threat_rule(self, rule: ThreatRule, 
                                  context: ExecutionContext, operation: str) -> bool:
        """Evaluate if a threat rule is triggered"""
        conditions = rule.conditions
        
        # Check operation type conditions
        if "operation_type" in conditions:
            if operation == conditions["operation_type"]:
                return True
        
        # Check rapid operations
        if "operations_per_minute" in conditions:
            agent_id = context.agent_id
            recent_events = [e for e in self._security_events 
                           if e.agent_id == agent_id and 
                           e.timestamp > datetime.utcnow() - timedelta(minutes=1)]
            
            if len(recent_events) > conditions["operations_per_minute"]:
                return True
        
        # Check anomaly score
        if "anomaly_score" in conditions:
            anomaly_score = self._calculate_anomaly_score(
                context.agent_id, 
                [e for e in self._security_events 
                 if e.agent_id == context.agent_id and 
                 e.timestamp > datetime.utcnow() - timedelta(hours=1)]
            )
            
            if anomaly_score > conditions["anomaly_score"]:
                return True
        
        return False
    
    async def _handle_threat_detection(self, rule: ThreatRule, 
                                     context: ExecutionContext, operation: str) -> None:
        """Handle detected threat"""
        self._threats_detected += 1
        
        # Log threat detection
        await self._log_security_event(
            SecurityEventType.THREAT_DETECTED,
            context.agent_id,
            rule.threat_level,
            f"Threat rule '{rule.name}' triggered for operation '{operation}'"
        )
        
        # Execute threat response actions
        for action in rule.actions:
            await self._execute_threat_action(action, rule, context, operation)
        
        # Trigger emergency protocols if needed
        if rule.threat_level in self._emergency_protocols:
            for protocol in self._emergency_protocols[rule.threat_level]:
                try:
                    await protocol(rule, context, operation)
                except Exception as e:
                    logger.error(f"Error executing emergency protocol: {e}")
    
    async def _execute_threat_action(self, action: str, rule: ThreatRule,
                                   context: ExecutionContext, operation: str) -> None:
        """Execute threat response action"""
        if action == "block":
            # Already blocked by returning False
            logger.info(f'Processing task: {locals()}')
            return {'success': True, 'message': 'Task processed'}
        elif action == "alert":
            logger.warning(f"SECURITY ALERT: {rule.name} - Agent: {context.agent_id.full_id}")
        elif action == "log":
            logger.info(f"Security log: {rule.name} - Operation: {operation}")
        elif action == "throttle":
            # Add restriction to throttle future operations
            await self.add_agent_restriction(context.agent_id, f"throttle_{operation}")
        elif action == "emergency_stop":
            await self._log_security_event(
                SecurityEventType.EMERGENCY_RESPONSE,
                context.agent_id,
                ThreatLevel.CRITICAL,
                f"Emergency stop triggered by {rule.name}"
            )
    
    async def _log_security_event(self, event_type: SecurityEventType, 
                                agent_id: AgentId, threat_level: ThreatLevel,
                                description: str, metadata: Dict[str, Any] = None) -> None:
        """Log security event"""
        event = SecurityEvent(
            event_id=f"{event_type.value}_{agent_id.full_id}_{datetime.utcnow().timestamp()}",
            event_type=event_type,
            agent_id=agent_id,
            timestamp=datetime.utcnow(),
            threat_level=threat_level,
            description=description,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self._security_events.append(event)
            # Keep only last 10000 events
            if len(self._security_events) > 10000:
                self._security_events = self._security_events[-10000:]
        
        # Publish event
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=event.event_id,
                event_type=f"security.{event_type.value}",
                source=agent_id,
                timestamp=event.timestamp,
                data={
                    "threat_level": threat_level.value,
                    "description": description,
                    "metadata": metadata or {}
                }
            ))
    
    async def _update_behavior_pattern(self, agent_id: AgentId, operation: str) -> None:
        """Update behavior pattern for an agent"""
        now = datetime.utcnow()
        time_str = now.strftime("%H:%M")
        
        async with self._lock:
            if agent_id not in self._behavior_patterns:
                self._behavior_patterns[agent_id] = BehaviorPattern(
                    agent_id=agent_id,
                    pattern_type="normal",
                    frequency=1.0,
                    typical_times=[time_str],
                    typical_operations={operation},
                    confidence=0.5
                )
            else:
                pattern = self._behavior_patterns[agent_id]
                pattern.typical_operations.add(operation)
                if time_str not in pattern.typical_times:
                    pattern.typical_times.append(time_str)
                pattern.frequency = min(10.0, pattern.frequency + 0.1)
                pattern.confidence = min(1.0, pattern.confidence + 0.01)
    
    def _calculate_anomaly_score(self, agent_id: AgentId, 
                               recent_events: List[SecurityEvent]) -> float:
        """Calculate anomaly score for an agent based on recent events"""
        if agent_id not in self._behavior_patterns:
            return 0.5  # Neutral score for unknown agents
        
        pattern = self._behavior_patterns[agent_id]
        
        # Check frequency anomaly
        current_frequency = len(recent_events)
        frequency_anomaly = abs(current_frequency - pattern.frequency) / max(pattern.frequency, 1)
        
        # Check time anomaly
        current_time = datetime.utcnow().strftime("%H:%M")
        time_anomaly = 0.0 if current_time in pattern.typical_times else 0.3
        
        # Check operation anomaly
        recent_operations = {e.description for e in recent_events}
        operation_anomaly = len(recent_operations - pattern.typical_operations) / max(len(recent_operations), 1)
        
        # Weighted average
        anomaly_score = (frequency_anomaly * 0.4 + time_anomaly * 0.3 + operation_anomaly * 0.3)
        return min(1.0, anomaly_score)
    
    def _assess_risk_level(self, anomaly_score: float) -> str:
        """Assess risk level based on anomaly score"""
        if anomaly_score >= 0.8:
            return "HIGH"
        elif anomaly_score >= 0.6:
            return "MEDIUM"
        elif anomaly_score >= 0.4:
            return "LOW"
        else:
            return "NORMAL"
    
    def _get_threat_level_distribution(self) -> Dict[str, int]:
        """Get distribution of threat levels in recent events"""
        recent_events = [e for e in self._security_events 
                        if e.timestamp > datetime.utcnow() - timedelta(hours=24)]
        
        distribution = {level.name: 0 for level in ThreatLevel}
        for event in recent_events:
            distribution[event.threat_level.name] += 1
        
        return distribution
    
    def _calculate_security_score(self) -> float:
        """Calculate overall security score (0-100)"""
        if self._operations_count == 0:
            return 100.0
        
        # Base score
        blocked_ratio = self._blocked_operations / self._operations_count
        base_score = max(0, 100 - (blocked_ratio * 50))
        
        # Adjust for recent threats
        recent_threats = len([e for e in self._security_events 
                            if e.threat_level.value >= ThreatLevel.MEDIUM.value and
                            e.timestamp > datetime.utcnow() - timedelta(hours=24)])
        
        threat_penalty = min(30, recent_threats * 5)
        
        return max(0, base_score - threat_penalty)
    
    async def _background_monitoring(self) -> None:
        """Background monitoring task"""
        while self._monitoring_active:
            try:
                await asyncio.sleep(60)  # Monitor every minute
                
                # Clean old events
                cutoff = datetime.utcnow() - timedelta(days=7)
                async with self._lock:
                    self._security_events = [e for e in self._security_events if e.timestamp > cutoff]
                
                # Log monitoring status
                if len(self._security_events) > 0:
                    logger.debug(f"Security monitoring: {len(self._security_events)} events in memory")
                
            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
    
    async def shutdown(self) -> None:
        """Shutdown security monitoring"""
        self._monitoring_active = False
        logger.info("Security monitoring service shut down")