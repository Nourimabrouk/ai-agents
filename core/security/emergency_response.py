"""
Emergency Response System - Phase 7
Automated incident response and security containment for autonomous systems
Handles security incidents with rapid response and automated mitigation
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .autonomous_security import SecurityThreat, SecurityThreatLevel
from .behavioral_monitoring import BehavioralAnomaly
from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ResponseAction(Enum):
    """Types of emergency response actions"""
    QUARANTINE_AGENT = "quarantine_agent"
    TERMINATE_PROCESS = "terminate_process"
    RESET_AGENT_STATE = "reset_agent_state"
    BLOCK_NETWORK_ACCESS = "block_network_access"
    DISABLE_MODIFICATIONS = "disable_modifications"
    ACTIVATE_MONITORING = "activate_monitoring"
    NOTIFY_ADMINISTRATORS = "notify_administrators"
    SYSTEM_LOCKDOWN = "system_lockdown"


class ResponsePriority(Enum):
    """Priority levels for responses"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


@dataclass
class SecurityIncident:
    """Represents a security incident"""
    incident_id: str
    title: str
    description: str
    severity: SecurityThreatLevel
    affected_agents: List[str]
    threats: List[SecurityThreat]
    anomalies: List[BehavioralAnomaly]
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "open"  # open, investigating, contained, resolved
    assigned_responder: Optional[str] = None
    response_actions: List[Dict[str, Any]] = field(default_factory=list)
    containment_applied: bool = False
    resolution_time_seconds: Optional[float] = None


@dataclass
class ResponsePlan:
    """Automated response plan for security incidents"""
    plan_id: str
    name: str
    trigger_conditions: Dict[str, Any]
    response_actions: List[ResponseAction]
    priority: ResponsePriority
    automation_level: str  # "manual", "semi-automatic", "automatic"
    estimated_execution_time: int
    rollback_possible: bool


class SecurityIncidentHandler:
    """
    Handles security incidents with automated and manual responses
    Provides rapid containment and mitigation
    """
    
    def __init__(self):
        self.active_incidents: Dict[str, SecurityIncident] = {}
        self.incident_history: List[SecurityIncident] = []
        self.response_plans = self._load_response_plans()
        self.automated_responses_enabled = True
        self.quarantined_agents: Set[str] = set()
        
        # Response metrics
        self.response_metrics = {
            'incidents_handled': 0,
            'automated_responses': 0,
            'manual_responses': 0,
            'avg_response_time': 0.0,
            'containment_success_rate': 0.0
        }
        
        logger.info("Security Incident Handler initialized")
    
    async def handle_security_incident(self,
                                     threats: List[SecurityThreat],
                                     anomalies: List[BehavioralAnomaly],
                                     severity: SecurityThreatLevel) -> SecurityIncident:
        """Handle a security incident with appropriate response"""
        
        # Create incident
        incident_id = f"incident_{int(datetime.now().timestamp())}"
        incident = SecurityIncident(
            incident_id=incident_id,
            title=f"Security Incident - {severity.value.upper()}",
            description=self._generate_incident_description(threats, anomalies),
            severity=severity,
            affected_agents=self._extract_affected_agents(threats, anomalies),
            threats=threats,
            anomalies=anomalies
        )
        
        # Store incident
        self.active_incidents[incident_id] = incident
        self.incident_history.append(incident)
        self.response_metrics['incidents_handled'] += 1
        
        logger.critical(f"Security incident created: {incident_id} - {severity.value}")
        global_metrics.incr(f"security.incident.{severity.value}")
        
        # Determine and execute response
        await self._execute_incident_response(incident)
        
        return incident
    
    async def _execute_incident_response(self, incident: SecurityIncident):
        """Execute appropriate response for incident"""
        start_time = datetime.now()
        
        try:
            # Find matching response plan
            response_plan = self._match_response_plan(incident)
            
            if response_plan and self.automated_responses_enabled:
                # Execute automated response
                await self._execute_automated_response(incident, response_plan)
                self.response_metrics['automated_responses'] += 1
            else:
                # Execute manual response
                await self._execute_manual_response(incident)
                self.response_metrics['manual_responses'] += 1
            
            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()
            
            # Update metrics
            current_avg = self.response_metrics['avg_response_time']
            total_responses = self.response_metrics['automated_responses'] + self.response_metrics['manual_responses']
            self.response_metrics['avg_response_time'] = (current_avg * (total_responses - 1) + response_time) / total_responses
            
        except Exception as e:
            logger.error(f"Incident response failed for {incident.incident_id}: {e}")
            # Emergency fallback
            await self._emergency_fallback_response(incident)
    
    async def _execute_automated_response(self, incident: SecurityIncident, plan: ResponsePlan):
        """Execute automated response plan"""
        logger.info(f"Executing automated response plan: {plan.name} for incident {incident.incident_id}")
        
        for action in plan.response_actions:
            try:
                result = await self._execute_response_action(action, incident)
                
                incident.response_actions.append({
                    'action': action.value,
                    'timestamp': datetime.now().isoformat(),
                    'result': result,
                    'automated': True
                })
                
                logger.info(f"Response action completed: {action.value}")
                
            except Exception as e:
                logger.error(f"Response action failed: {action.value} - {e}")
                incident.response_actions.append({
                    'action': action.value,
                    'timestamp': datetime.now().isoformat(),
                    'result': {'success': False, 'error': str(e)},
                    'automated': True
                })
        
        # Mark as contained if all actions succeeded
        successful_actions = sum(1 for a in incident.response_actions if a.get('result', {}).get('success', False))
        if successful_actions >= len(plan.response_actions) * 0.8:  # 80% success rate
            incident.containment_applied = True
            incident.status = "contained"
            logger.info(f"Incident {incident.incident_id} successfully contained")
        
    async def _execute_manual_response(self, incident: SecurityIncident):
        """Execute manual response for high-severity incidents"""
        logger.warning(f"Manual response required for incident: {incident.incident_id}")
        
        # Apply basic containment measures
        basic_actions = [
            ResponseAction.QUARANTINE_AGENT,
            ResponseAction.ACTIVATE_MONITORING,
            ResponseAction.NOTIFY_ADMINISTRATORS
        ]
        
        for action in basic_actions:
            if incident.severity in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL, SecurityThreatLevel.EMERGENCY]:
                try:
                    result = await self._execute_response_action(action, incident)
                    incident.response_actions.append({
                        'action': action.value,
                        'timestamp': datetime.now().isoformat(),
                        'result': result,
                        'automated': False
                    })
                except Exception as e:
                    logger.error(f"Manual response action failed: {action.value} - {e}")
        
        # Mark for manual review
        incident.status = "investigating"
        incident.assigned_responder = "security_team"
    
    async def _execute_response_action(self, action: ResponseAction, incident: SecurityIncident) -> Dict[str, Any]:
        """Execute a specific response action"""
        
        if action == ResponseAction.QUARANTINE_AGENT:
            return await self._quarantine_agents(incident.affected_agents)
        
        elif action == ResponseAction.RESET_AGENT_STATE:
            return await self._reset_agent_states(incident.affected_agents)
        
        elif action == ResponseAction.DISABLE_MODIFICATIONS:
            return await self._disable_agent_modifications(incident.affected_agents)
        
        elif action == ResponseAction.ACTIVATE_MONITORING:
            return await self._activate_enhanced_monitoring(incident.affected_agents)
        
        elif action == ResponseAction.NOTIFY_ADMINISTRATORS:
            return await self._notify_administrators(incident)
        
        elif action == ResponseAction.SYSTEM_LOCKDOWN:
            return await self._initiate_system_lockdown(incident)
        
        else:
            return {'success': False, 'error': f'Unknown action: {action.value}'}
    
    async def _quarantine_agents(self, agent_names: List[str]) -> Dict[str, Any]:
        """Quarantine specified agents"""
        quarantined_count = 0
        errors = []
        
        for agent_name in agent_names:
            try:
                self.quarantined_agents.add(agent_name)
                quarantined_count += 1
                logger.warning(f"Agent {agent_name} quarantined")
                global_metrics.incr("security.agent_quarantined")
                
            except Exception as e:
                errors.append(f"Failed to quarantine {agent_name}: {e}")
        
        return {
            'success': quarantined_count > 0,
            'quarantined_count': quarantined_count,
            'errors': errors,
            'total_quarantined': len(self.quarantined_agents)
        }
    
    async def _reset_agent_states(self, agent_names: List[str]) -> Dict[str, Any]:
        """Reset agent states to clean configuration"""
        reset_count = 0
        errors = []
        
        for agent_name in agent_names:
            try:
                # In a real implementation, this would restore agent to known-good state
                logger.info(f"Agent {agent_name} state reset initiated")
                reset_count += 1
                
            except Exception as e:
                errors.append(f"Failed to reset {agent_name}: {e}")
        
        return {
            'success': reset_count > 0,
            'reset_count': reset_count,
            'errors': errors
        }
    
    async def _disable_agent_modifications(self, agent_names: List[str]) -> Dict[str, Any]:
        """Disable self-modification capabilities for agents"""
        disabled_count = 0
        
        for agent_name in agent_names:
            # In a real implementation, this would disable modification capabilities
            logger.info(f"Modifications disabled for agent {agent_name}")
            disabled_count += 1
        
        return {
            'success': True,
            'disabled_count': disabled_count,
            'message': f'Modifications disabled for {disabled_count} agents'
        }
    
    async def _activate_enhanced_monitoring(self, agent_names: List[str]) -> Dict[str, Any]:
        """Activate enhanced monitoring for specified agents"""
        monitored_count = len(agent_names)
        
        logger.info(f"Enhanced monitoring activated for {monitored_count} agents")
        
        return {
            'success': True,
            'monitored_count': monitored_count,
            'monitoring_level': 'enhanced'
        }
    
    async def _notify_administrators(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Notify system administrators about the incident"""
        
        notification = {
            'incident_id': incident.incident_id,
            'severity': incident.severity.value,
            'affected_agents': incident.affected_agents,
            'threat_count': len(incident.threats),
            'timestamp': incident.created_at.isoformat()
        }
        
        # In a real implementation, this would send notifications via email, Slack, etc.
        logger.critical(f"SECURITY ALERT: {json.dumps(notification, indent=2)}")
        
        return {
            'success': True,
            'notification_sent': True,
            'recipients': ['security_team', 'system_administrators']
        }
    
    async def _initiate_system_lockdown(self, incident: SecurityIncident) -> Dict[str, Any]:
        """Initiate emergency system lockdown"""
        
        logger.critical(f"EMERGENCY SYSTEM LOCKDOWN INITIATED: {incident.incident_id}")
        
        # In a real implementation, this would:
        # - Stop all non-essential processes
        # - Block network connections
        # - Disable user access
        # - Preserve forensic evidence
        
        return {
            'success': True,
            'lockdown_initiated': True,
            'timestamp': datetime.now().isoformat(),
            'reason': incident.description
        }
    
    async def _emergency_fallback_response(self, incident: SecurityIncident):
        """Emergency fallback when normal response fails"""
        logger.critical(f"EMERGENCY FALLBACK: Normal response failed for {incident.incident_id}")
        
        # Quarantine all affected agents as emergency measure
        await self._quarantine_agents(incident.affected_agents)
        
        # Notify administrators
        await self._notify_administrators(incident)
        
        # Mark incident as requiring urgent attention
        incident.status = "emergency_containment"
        incident.containment_applied = True
    
    def _match_response_plan(self, incident: SecurityIncident) -> Optional[ResponsePlan]:
        """Find matching response plan for incident"""
        
        for plan in self.response_plans:
            if self._incident_matches_plan(incident, plan):
                return plan
        
        return {}
    
    def _incident_matches_plan(self, incident: SecurityIncident, plan: ResponsePlan) -> bool:
        """Check if incident matches response plan conditions"""
        conditions = plan.trigger_conditions
        
        # Check severity
        if 'min_severity' in conditions:
            severity_levels = {
                SecurityThreatLevel.LOW: 1,
                SecurityThreatLevel.MEDIUM: 2,
                SecurityThreatLevel.HIGH: 3,
                SecurityThreatLevel.CRITICAL: 4,
                SecurityThreatLevel.EMERGENCY: 5
            }
            
            if severity_levels.get(incident.severity, 0) < conditions['min_severity']:
                return False
        
        # Check threat types
        if 'threat_types' in conditions:
            incident_threat_types = set(threat.threat_type for threat in incident.threats)
            required_types = set(conditions['threat_types'])
            
            if not incident_threat_types.intersection(required_types):
                return False
        
        # Check agent count
        if 'max_affected_agents' in conditions:
            if len(incident.affected_agents) > conditions['max_affected_agents']:
                return False
        
        return True
    
    def _generate_incident_description(self, threats: List[SecurityThreat], anomalies: List[BehavioralAnomaly]) -> str:
        """Generate description for security incident"""
        description_parts = []
        
        if threats:
            threat_types = set(threat.threat_type for threat in threats)
            description_parts.append(f"Threats detected: {', '.join(threat_types)}")
        
        if anomalies:
            anomaly_types = set(anomaly.anomaly_type.value for anomaly in anomalies)
            description_parts.append(f"Behavioral anomalies: {', '.join(anomaly_types)}")
        
        return "; ".join(description_parts) if description_parts else "Security incident detected"
    
    def _extract_affected_agents(self, threats: List[SecurityThreat], anomalies: List[BehavioralAnomaly]) -> List[str]:
        """Extract list of affected agents"""
        affected_agents = set()
        
        for threat in threats:
            affected_agents.update(threat.affected_agents)
        
        for anomaly in anomalies:
            affected_agents.add(anomaly.agent_name)
        
        return list(affected_agents)
    
    def _load_response_plans(self) -> List[ResponsePlan]:
        """Load predefined response plans"""
        return [
            ResponsePlan(
                plan_id="critical_threat_response",
                name="Critical Threat Response",
                trigger_conditions={
                    'min_severity': 4,  # Critical or higher
                    'threat_types': ['code_injection', 'system_compromise']
                },
                response_actions=[
                    ResponseAction.QUARANTINE_AGENT,
                    ResponseAction.DISABLE_MODIFICATIONS,
                    ResponseAction.NOTIFY_ADMINISTRATORS,
                    ResponseAction.RESET_AGENT_STATE
                ],
                priority=ResponsePriority.CRITICAL,
                automation_level="automatic",
                estimated_execution_time=30,
                rollback_possible=True
            ),
            ResponsePlan(
                plan_id="behavioral_anomaly_response",
                name="Behavioral Anomaly Response",
                trigger_conditions={
                    'min_severity': 2,  # Medium or higher
                    'max_affected_agents': 5
                },
                response_actions=[
                    ResponseAction.ACTIVATE_MONITORING,
                    ResponseAction.QUARANTINE_AGENT
                ],
                priority=ResponsePriority.HIGH,
                automation_level="semi-automatic",
                estimated_execution_time=15,
                rollback_possible=True
            ),
            ResponsePlan(
                plan_id="emergency_lockdown",
                name="Emergency System Lockdown",
                trigger_conditions={
                    'min_severity': 5,  # Emergency
                    'threat_types': ['system_compromise', 'malware']
                },
                response_actions=[
                    ResponseAction.SYSTEM_LOCKDOWN,
                    ResponseAction.QUARANTINE_AGENT,
                    ResponseAction.NOTIFY_ADMINISTRATORS
                ],
                priority=ResponsePriority.EMERGENCY,
                automation_level="automatic",
                estimated_execution_time=60,
                rollback_possible=False
            )
        ]
    
    def get_incident_status(self, incident_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a security incident"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            return {
                'incident_id': incident.incident_id,
                'status': incident.status,
                'severity': incident.severity.value,
                'affected_agents': incident.affected_agents,
                'containment_applied': incident.containment_applied,
                'response_actions': len(incident.response_actions),
                'created_at': incident.created_at.isoformat()
            }
        return {}
    
    def resolve_incident(self, incident_id: str, resolution_notes: str = ""):
        """Mark incident as resolved"""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.status = "resolved"
            incident.resolution_time_seconds = (datetime.now() - incident.created_at).total_seconds()
            
            # Remove from active incidents
            del self.active_incidents[incident_id]
            
            logger.info(f"Incident resolved: {incident_id} - {resolution_notes}")
            global_metrics.incr("security.incident_resolved")
    
    def get_response_metrics(self) -> Dict[str, Any]:
        """Get emergency response metrics"""
        total_incidents = len(self.incident_history)
        contained_incidents = sum(1 for inc in self.incident_history if inc.containment_applied)
        
        containment_success_rate = contained_incidents / max(total_incidents, 1)
        self.response_metrics['containment_success_rate'] = containment_success_rate
        
        return {
            'active_incidents': len(self.active_incidents),
            'total_incidents': total_incidents,
            'quarantined_agents': len(self.quarantined_agents),
            'automated_responses_enabled': self.automated_responses_enabled,
            **self.response_metrics
        }


class EmergencyResponseSystem:
    """
    Main emergency response system coordinating incident handling
    """
    
    def __init__(self):
        self.incident_handler = SecurityIncidentHandler()
        self.response_queue: List[SecurityIncident] = []
        self.system_lockdown_active = False
        
        logger.info("Emergency Response System initialized")
    
    async def respond_to_security_event(self,
                                      threats: List[SecurityThreat],
                                      anomalies: List[BehavioralAnomaly]) -> SecurityIncident:
        """Main entry point for emergency response"""
        
        # Determine severity
        max_severity = SecurityThreatLevel.LOW
        for threat in threats:
            if threat.severity.value in ['critical', 'emergency']:
                max_severity = SecurityThreatLevel.CRITICAL
                break
            elif threat.severity.value == 'high' and max_severity != SecurityThreatLevel.CRITICAL:
                max_severity = SecurityThreatLevel.HIGH
        
        for anomaly in anomalies:
            if anomaly.severity.value in ['critical', 'high']:
                max_severity = max(max_severity, SecurityThreatLevel.HIGH)
        
        # Handle incident
        incident = await self.incident_handler.handle_security_incident(
            threats, anomalies, max_severity
        )
        
        # Add to response queue for tracking
        self.response_queue.append(incident)
        
        return incident
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall emergency response system status"""
        return {
            'system_lockdown_active': self.system_lockdown_active,
            'pending_responses': len(self.response_queue),
            'quarantined_agents': len(self.incident_handler.quarantined_agents),
            'active_incidents': len(self.incident_handler.active_incidents),
            'response_metrics': self.incident_handler.get_response_metrics()
        }