"""
Threat Detection System - Phase 7
Advanced threat detection and security monitoring for autonomous systems
Identifies and analyzes security threats in real-time
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import hashlib
from collections import defaultdict

from .autonomous_security import SecurityThreat, SecurityThreatLevel
from .behavioral_monitoring import BehavioralAnomaly, AnomalyType
from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ThreatCategory(Enum):
    """Categories of security threats"""
    MALWARE = "malware"
    DATA_BREACH = "data_breach"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    CODE_INJECTION = "code_injection"
    DENIAL_OF_SERVICE = "denial_of_service"
    SOCIAL_ENGINEERING = "social_engineering"
    INSIDER_THREAT = "insider_threat"
    SYSTEM_COMPROMISE = "system_compromise"


@dataclass
class ThreatPattern:
    """Pattern used for threat detection"""
    pattern_id: str
    name: str
    category: ThreatCategory
    severity: SecurityThreatLevel
    indicators: List[str]
    description: str
    false_positive_rate: float
    detection_logic: str


@dataclass
class ThreatIntelligence:
    """Threat intelligence information"""
    threat_id: str
    threat_name: str
    category: ThreatCategory
    severity: SecurityThreatLevel
    description: str
    indicators_of_compromise: List[str]
    mitigation_strategies: List[str]
    last_updated: datetime
    source: str


class SecurityThreatAnalyzer:
    """
    Analyzes security data to identify potential threats
    Uses pattern matching and behavioral analysis
    """
    
    def __init__(self):
        self.threat_patterns = self._load_threat_patterns()
        self.threat_intelligence = self._load_threat_intelligence()
        self.analysis_history: List[Dict[str, Any]] = []
        
    async def analyze_security_data(self, 
                                  security_events: List[Dict[str, Any]],
                                  behavioral_anomalies: List[BehavioralAnomaly],
                                  system_logs: Optional[List[str]] = None) -> List[SecurityThreat]:
        """Analyze security data to identify threats"""
        start_time = datetime.now()
        detected_threats = []
        
        try:
            # Pattern-based threat detection
            pattern_threats = await self._detect_pattern_threats(security_events)
            detected_threats.extend(pattern_threats)
            
            # Behavioral anomaly analysis
            anomaly_threats = await self._analyze_behavioral_threats(behavioral_anomalies)
            detected_threats.extend(anomaly_threats)
            
            # System log analysis
            if system_logs:
                log_threats = await self._analyze_system_logs(system_logs)
                detected_threats.extend(log_threats)
            
            # Correlation analysis
            correlated_threats = await self._correlate_threats(detected_threats)
            
            # Record analysis
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'events_analyzed': len(security_events),
                'anomalies_analyzed': len(behavioral_anomalies),
                'threats_detected': len(correlated_threats),
                'analysis_duration_ms': (datetime.now() - start_time).total_seconds() * 1000
            })
            
            logger.info(f"Security analysis complete: {len(correlated_threats)} threats detected")
            return correlated_threats
            
        except Exception as e:
            logger.error(f"Security threat analysis failed: {e}")
            return detected_threats
    
    async def _detect_pattern_threats(self, security_events: List[Dict[str, Any]]) -> List[SecurityThreat]:
        """Detect threats using predefined patterns"""
        threats = []
        
        for event in security_events:
            event_data = json.dumps(event, default=str).lower()
            
            for pattern in self.threat_patterns:
                matches = []
                for indicator in pattern.indicators:
                    if indicator.lower() in event_data:
                        matches.append(indicator)
                
                # If enough indicators match, create threat
                if len(matches) >= len(pattern.indicators) * 0.6:  # 60% threshold
                    threat = SecurityThreat(
                        threat_id=f"pattern_{pattern.pattern_id}_{int(datetime.now().timestamp())}",
                        threat_type=pattern.category.value,
                        severity=pattern.severity,
                        description=f"Pattern-based threat detected: {pattern.name}",
                        evidence={
                            'pattern_id': pattern.pattern_id,
                            'matched_indicators': matches,
                            'event_data': event
                        },
                        affected_agents=[event.get('agent_name', 'unknown')],
                        detection_method="pattern_matching",
                        confidence_score=min(1.0, len(matches) / len(pattern.indicators))
                    )
                    threats.append(threat)
        
        return threats
    
    async def _analyze_behavioral_threats(self, behavioral_anomalies: List[BehavioralAnomaly]) -> List[SecurityThreat]:
        """Analyze behavioral anomalies for security threats"""
        threats = []
        
        # Map anomaly types to threat types
        anomaly_to_threat_mapping = {
            AnomalyType.RAPID_MODIFICATION_ATTEMPTS: (ThreatCategory.SYSTEM_COMPROMISE, SecurityThreatLevel.HIGH),
            AnomalyType.UNAUTHORIZED_ACCESS_ATTEMPTS: (ThreatCategory.PRIVILEGE_ESCALATION, SecurityThreatLevel.CRITICAL),
            AnomalyType.EXCESSIVE_RESOURCE_USAGE: (ThreatCategory.DENIAL_OF_SERVICE, SecurityThreatLevel.MEDIUM),
            AnomalyType.SUSPICIOUS_COORDINATION_BEHAVIOR: (ThreatCategory.INSIDER_THREAT, SecurityThreatLevel.HIGH)
        }
        
        for anomaly in behavioral_anomalies:
            if anomaly.anomaly_type in anomaly_to_threat_mapping:
                threat_category, threat_severity = anomaly_to_threat_mapping[anomaly.anomaly_type]
                
                threat = SecurityThreat(
                    threat_id=f"behavioral_{anomaly.anomaly_id}",
                    threat_type=threat_category.value,
                    severity=threat_severity,
                    description=f"Behavioral threat: {anomaly.description}",
                    evidence={
                        'anomaly_id': anomaly.anomaly_id,
                        'anomaly_evidence': anomaly.evidence,
                        'confidence_score': anomaly.confidence_score
                    },
                    affected_agents=[anomaly.agent_name],
                    detection_method="behavioral_analysis",
                    confidence_score=anomaly.confidence_score
                )
                threats.append(threat)
        
        return threats
    
    async def _analyze_system_logs(self, system_logs: List[str]) -> List[SecurityThreat]:
        """Analyze system logs for security threats"""
        threats = []
        
        # Define suspicious log patterns
        suspicious_patterns = [
            ('failed login', ThreatCategory.PRIVILEGE_ESCALATION, SecurityThreatLevel.MEDIUM),
            ('permission denied', ThreatCategory.PRIVILEGE_ESCALATION, SecurityThreatLevel.LOW),
            ('segmentation fault', ThreatCategory.SYSTEM_COMPROMISE, SecurityThreatLevel.HIGH),
            ('access violation', ThreatCategory.SYSTEM_COMPROMISE, SecurityThreatLevel.HIGH),
            ('buffer overflow', ThreatCategory.CODE_INJECTION, SecurityThreatLevel.CRITICAL),
            ('sql injection', ThreatCategory.CODE_INJECTION, SecurityThreatLevel.CRITICAL)
        ]
        
        for log_entry in system_logs:
            log_lower = log_entry.lower()
            
            for pattern, category, severity in suspicious_patterns:
                if pattern in log_lower:
                    threat = SecurityThreat(
                        threat_id=f"log_{hashlib.md5(log_entry.encode()).hexdigest()[:8]}",
                        threat_type=category.value,
                        severity=severity,
                        description=f"Suspicious log entry detected: {pattern}",
                        evidence={'log_entry': log_entry, 'pattern': pattern},
                        affected_agents=['system'],
                        detection_method="log_analysis",
                        confidence_score=0.7
                    )
                    threats.append(threat)
        
        return threats
    
    async def _correlate_threats(self, threats: List[SecurityThreat]) -> List[SecurityThreat]:
        """Correlate related threats and enhance severity"""
        if len(threats) <= 1:
            return threats
        
        # Group threats by agent
        threats_by_agent = defaultdict(list)
        for threat in threats:
            for agent in threat.affected_agents:
                threats_by_agent[agent].append(threat)
        
        enhanced_threats = []
        
        for agent_name, agent_threats in threats_by_agent.items():
            if len(agent_threats) > 1:
                # Multiple threats for same agent - increase severity
                for threat in agent_threats:
                    # Create enhanced threat
                    enhanced_threat = SecurityThreat(
                        threat_id=f"corr_{threat.threat_id}",
                        threat_type=threat.threat_type,
                        severity=self._escalate_severity(threat.severity),
                        description=f"Correlated threat: {threat.description} (part of {len(agent_threats)} threats)",
                        evidence={**threat.evidence, 'correlation_count': len(agent_threats)},
                        affected_agents=threat.affected_agents,
                        detection_method=f"correlated_{threat.detection_method}",
                        confidence_score=min(1.0, threat.confidence_score * 1.2)
                    )
                    enhanced_threats.append(enhanced_threat)
            else:
                enhanced_threats.extend(agent_threats)
        
        return enhanced_threats
    
    def _escalate_severity(self, current_severity: SecurityThreatLevel) -> SecurityThreatLevel:
        """Escalate threat severity due to correlation"""
        escalation_map = {
            SecurityThreatLevel.LOW: SecurityThreatLevel.MEDIUM,
            SecurityThreatLevel.MEDIUM: SecurityThreatLevel.HIGH,
            SecurityThreatLevel.HIGH: SecurityThreatLevel.CRITICAL,
            SecurityThreatLevel.CRITICAL: SecurityThreatLevel.EMERGENCY
        }
        return escalation_map.get(current_severity, current_severity)
    
    def _load_threat_patterns(self) -> List[ThreatPattern]:
        """Load predefined threat detection patterns"""
        return [
            ThreatPattern(
                pattern_id="code_injection_1",
                name="Code Injection Attack",
                category=ThreatCategory.CODE_INJECTION,
                severity=SecurityThreatLevel.CRITICAL,
                indicators=["eval(", "exec(", "__import__", "subprocess"],
                description="Potential code injection attack detected",
                false_positive_rate=0.1,
                detection_logic="multiple_dangerous_functions"
            ),
            ThreatPattern(
                pattern_id="privilege_escalation_1",
                name="Privilege Escalation",
                category=ThreatCategory.PRIVILEGE_ESCALATION,
                severity=SecurityThreatLevel.HIGH,
                indicators=["sudo", "admin", "root", "privilege"],
                description="Potential privilege escalation attempt",
                false_positive_rate=0.2,
                detection_logic="privilege_keywords"
            ),
            ThreatPattern(
                pattern_id="data_exfiltration_1",
                name="Data Exfiltration",
                category=ThreatCategory.DATA_BREACH,
                severity=SecurityThreatLevel.HIGH,
                indicators=["download", "export", "copy", "sensitive"],
                description="Potential data exfiltration detected",
                false_positive_rate=0.3,
                detection_logic="data_access_keywords"
            )
        ]
    
    def _load_threat_intelligence(self) -> List[ThreatIntelligence]:
        """Load threat intelligence data"""
        return [
            ThreatIntelligence(
                threat_id="malware_001",
                threat_name="Agent Code Virus",
                category=ThreatCategory.MALWARE,
                severity=SecurityThreatLevel.CRITICAL,
                description="Malicious code that spreads between agents",
                indicators_of_compromise=[
                    "self-replicating code patterns",
                    "unauthorized agent modifications",
                    "rapid capability spread"
                ],
                mitigation_strategies=[
                    "Quarantine affected agents",
                    "Validate all code modifications",
                    "Reset to clean state"
                ],
                last_updated=datetime.now(),
                source="internal_analysis"
            )
        ]


class ThreatDetectionSystem:
    """
    Main threat detection system coordinating all detection components
    """
    
    def __init__(self):
        self.threat_analyzer = SecurityThreatAnalyzer()
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.threat_history: List[SecurityThreat] = []
        self.detection_metrics = {
            'total_threats': 0,
            'false_positives': 0,
            'threats_mitigated': 0,
            'detection_accuracy': 0.0
        }
        
        logger.info("Threat Detection System initialized")
    
    async def detect_threats(self,
                           security_events: List[Dict[str, Any]],
                           behavioral_anomalies: List[BehavioralAnomaly],
                           system_logs: Optional[List[str]] = None) -> List[SecurityThreat]:
        """Main threat detection method"""
        
        logger.info(f"Starting threat detection: {len(security_events)} events, {len(behavioral_anomalies)} anomalies")
        
        try:
            # Run threat analysis
            detected_threats = await self.threat_analyzer.analyze_security_data(
                security_events, behavioral_anomalies, system_logs
            )
            
            # Process detected threats
            new_threats = []
            for threat in detected_threats:
                if threat.threat_id not in self.active_threats:
                    # New threat
                    self.active_threats[threat.threat_id] = threat
                    self.threat_history.append(threat)
                    new_threats.append(threat)
                    
                    # Update metrics
                    self.detection_metrics['total_threats'] += 1
                    
                    # Log threat
                    logger.warning(f"New threat detected: {threat.threat_type} - {threat.description}")
                    global_metrics.incr(f"security.threat_detected.{threat.threat_type}")
            
            logger.info(f"Threat detection complete: {len(new_threats)} new threats")
            return new_threats
            
        except Exception as e:
            logger.error(f"Threat detection failed: {e}")
            return []
    
    def get_active_threats(self, severity_filter: Optional[SecurityThreatLevel] = None) -> List[SecurityThreat]:
        """Get currently active threats"""
        threats = list(self.active_threats.values())
        
        if severity_filter:
            threats = [t for t in threats if t.severity == severity_filter]
        
        return threats
    
    def resolve_threat(self, threat_id: str, resolution_notes: str = ""):
        """Mark a threat as resolved"""
        if threat_id in self.active_threats:
            threat = self.active_threats[threat_id]
            threat.mitigated = True
            threat.response_actions.append(f"Resolved: {resolution_notes}")
            
            del self.active_threats[threat_id]
            self.detection_metrics['threats_mitigated'] += 1
            
            logger.info(f"Threat resolved: {threat_id}")
            global_metrics.incr("security.threat_resolved")
    
    def mark_false_positive(self, threat_id: str):
        """Mark a threat as false positive"""
        if threat_id in self.active_threats:
            threat = self.active_threats[threat_id]
            del self.active_threats[threat_id]
            
            self.detection_metrics['false_positives'] += 1
            logger.info(f"Threat marked as false positive: {threat_id}")
    
    def get_detection_metrics(self) -> Dict[str, Any]:
        """Get threat detection metrics"""
        total = self.detection_metrics['total_threats']
        false_pos = self.detection_metrics['false_positives']
        
        accuracy = 1.0 - (false_pos / max(total, 1))
        self.detection_metrics['detection_accuracy'] = accuracy
        
        return {
            'active_threats': len(self.active_threats),
            'total_threats_detected': total,
            'false_positives': false_pos,
            'threats_mitigated': self.detection_metrics['threats_mitigated'],
            'detection_accuracy': accuracy,
            'threat_categories': self._get_threat_distribution(),
            'severity_distribution': self._get_severity_distribution()
        }
    
    def _get_threat_distribution(self) -> Dict[str, int]:
        """Get distribution of threat types"""
        distribution = defaultdict(int)
        for threat in self.threat_history:
            distribution[threat.threat_type] += 1
        return dict(distribution)
    
    def _get_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of threat severities"""
        distribution = defaultdict(int)
        for threat in self.threat_history:
            distribution[threat.severity.value] += 1
        return dict(distribution)