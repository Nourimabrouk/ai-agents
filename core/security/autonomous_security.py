"""
Autonomous Security Framework - Phase 7 Security Implementation
Multi-layer security validation, monitoring, and response system
Protects against autonomous system security threats
"""

import asyncio
import logging
import json
import hashlib
import hmac
import secrets
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import inspect
import ast
import sys
from contextlib import contextmanager
try:
    import resource  # Unix/Linux only
except ImportError:
    resource = None  # Windows compatibility
import threading
import multiprocessing

from templates.base_agent import BaseAgent
from ..autonomous.safety import SafetyViolation, ViolationType
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class SecurityLevel(Enum):
    """Security enforcement levels for autonomous systems"""
    DEVELOPMENT = "development"    # Relaxed for testing
    STAGING = "staging"           # Standard security
    PRODUCTION = "production"     # Maximum security
    CRITICAL = "critical"        # Emergency lockdown


class SecurityThreatLevel(Enum):
    """Security threat severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SecurityThreat:
    """Represents a detected security threat"""
    threat_id: str
    threat_type: str
    severity: SecurityThreatLevel
    description: str
    evidence: Dict[str, Any]
    affected_agents: List[str]
    detection_method: str
    confidence_score: float
    mitigation_required: bool = True
    mitigated: bool = False
    detected_at: datetime = field(default_factory=datetime.now)
    response_actions: List[str] = field(default_factory=list)


@dataclass
class SecurityContext:
    """Security context for operations"""
    operation_id: str
    agent_name: str
    operation_type: str
    security_level: SecurityLevel
    allowed_resources: Set[str]
    time_limit_seconds: int
    memory_limit_mb: int
    network_access: bool = False
    file_system_access: bool = False
    dangerous_operations: bool = False


@dataclass
class SecurityAuditResult:
    """Result of security audit"""
    is_secure: bool
    confidence: float
    threats_detected: List[SecurityThreat]
    violations: List[SafetyViolation]
    recommendations: List[str]
    audit_duration_ms: float
    security_score: float  # 0-100 scale


class SecureMemoryManager:
    """Secure memory management without pickle"""
    
    def __init__(self):
        self.master_key = self._generate_master_key()
        self.encrypted_memories: Dict[str, str] = {}
        self.access_log: List[Dict[str, Any]] = []
    
    def _generate_master_key(self) -> bytes:
        """Generate cryptographically secure master key"""
        return secrets.token_bytes(32)  # 256-bit key
    
    def _derive_key(self, agent_id: str, salt: bytes) -> bytes:
        """Derive encryption key for specific agent"""
        return hashlib.pbkdf2_hmac('sha256', 
                                  self.master_key + agent_id.encode(),
                                  salt, 
                                  100000)  # 100k iterations
    
    async def secure_store_memory(self, agent_id: str, memory_data: Dict[str, Any]) -> str:
        """Securely store agent memory with encryption (no pickle)"""
        try:
            # Generate unique salt for this storage
            salt = secrets.token_bytes(16)
            key = self._derive_key(agent_id, salt)
            
            # Serialize memory data safely (JSON only - NO PICKLE)
            serialized_data = json.dumps(memory_data, default=str)
            
            # Simple XOR encryption (for demo - use proper crypto in production)
            encrypted_data = self._xor_encrypt(serialized_data.encode(), key)
            
            # Create storage package
            storage_package = {
                'salt': salt.hex(),
                'encrypted_data': encrypted_data.hex(),
                'timestamp': datetime.now().isoformat(),
                'agent_id': agent_id
            }
            
            # Create storage ID
            storage_id = hashlib.sha256(f"{agent_id}_{datetime.now().isoformat()}".encode()).hexdigest()
            
            # Store encrypted package as JSON (secure)
            self.encrypted_memories[storage_id] = json.dumps(storage_package)
            
            # Log access
            self.access_log.append({
                'action': 'store',
                'agent_id': agent_id,
                'storage_id': storage_id,
                'timestamp': datetime.now().isoformat(),
                'data_size': len(serialized_data)
            })
            
            logger.info(f"Securely stored memory for agent {agent_id}: {storage_id}")
            return storage_id
            
        except Exception as e:
            logger.error(f"Secure memory storage failed: {e}")
            raise
    
    def _xor_encrypt(self, data: bytes, key: bytes) -> bytes:
        """Simple XOR encryption"""
        return bytes(a ^ b for a, b in zip(data, (key * (len(data) // len(key) + 1))[:len(data)]))


class AutonomousSecurityFramework:
    """
    Main autonomous security framework
    Comprehensive protection for autonomous systems
    """
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.PRODUCTION,
                 config: Optional[Dict[str, Any]] = None):
        self.security_level = security_level
        self.config = config or {}
        
        # Initialize security components
        self.memory_manager = SecureMemoryManager()
        
        # Security state
        self.active_threats: Dict[str, SecurityThreat] = {}
        self.security_incidents: List[Dict[str, Any]] = []
        self.quarantined_agents: Set[str] = set()
        
        # Security metrics
        self.threats_detected_count = 0
        self.violations_prevented_count = 0
        
        logger.info(f"Autonomous Security Framework initialized (Level: {security_level.value})")
    
    async def validate_autonomous_operation(self,
                                          operation_type: str,
                                          operation_data: Dict[str, Any],
                                          agent_context: Dict[str, Any]) -> SecurityAuditResult:
        """Validate autonomous operation for security threats"""
        start_time = datetime.now()
        
        threats = []
        violations = []
        recommendations = []
        
        try:
            # 1. Check for dangerous operations
            if operation_type in ['code_execution', 'system_modification', 'file_access']:
                threats.append(SecurityThreat(
                    threat_id=f"dangerous_op_{int(datetime.now().timestamp())}",
                    threat_type="dangerous_operation",
                    severity=SecurityThreatLevel.HIGH,
                    description=f"Dangerous operation type: {operation_type}",
                    evidence={"operation_type": operation_type},
                    affected_agents=[agent_context.get('agent_name', 'unknown')],
                    detection_method="operation_validation",
                    confidence_score=0.9
                ))
            
            # 2. Check for code injection patterns
            data_str = json.dumps(operation_data, default=str).lower()
            dangerous_patterns = ['eval(', 'exec(', '__import__', 'subprocess', 'pickle']
            
            for pattern in dangerous_patterns:
                if pattern in data_str:
                    threats.append(SecurityThreat(
                        threat_id=f"injection_{int(datetime.now().timestamp())}",
                        threat_type="code_injection",
                        severity=SecurityThreatLevel.CRITICAL,
                        description=f"Code injection pattern detected: {pattern}",
                        evidence={"pattern": pattern},
                        affected_agents=[agent_context.get('agent_name', 'unknown')],
                        detection_method="pattern_detection",
                        confidence_score=0.95
                    ))
            
            # 3. Check modification limits
            mod_count = agent_context.get('modification_count', 0)
            if mod_count > 5:
                threats.append(SecurityThreat(
                    threat_id=f"excessive_mod_{int(datetime.now().timestamp())}",
                    threat_type="excessive_modifications",
                    severity=SecurityThreatLevel.MEDIUM,
                    description=f"Excessive modifications: {mod_count}",
                    evidence={"modification_count": mod_count},
                    affected_agents=[agent_context.get('agent_name', 'unknown')],
                    detection_method="behavioral_analysis",
                    confidence_score=0.8
                ))
            
            # Calculate security metrics
            critical_threats = [t for t in threats if t.severity == SecurityThreatLevel.CRITICAL]
            is_secure = len(critical_threats) == 0
            confidence = 0.9 - (len(threats) * 0.1)
            security_score = 100.0 - (len(threats) * 20)
            
            # Generate recommendations
            if critical_threats:
                recommendations.append("IMMEDIATE ACTION: Critical security threats detected")
                recommendations.append("Quarantine affected agents immediately")
            
            if threats:
                recommendations.append("Enhanced monitoring required")
                recommendations.append("Review security policies")
            
        except Exception as e:
            logger.error(f"Security validation failed: {e}")
            is_secure = False
            confidence = 0.0
            security_score = 0.0
            threats.append(SecurityThreat(
                threat_id=f"validation_error_{int(datetime.now().timestamp())}",
                threat_type="validation_failure",
                severity=SecurityThreatLevel.HIGH,
                description=f"Validation failed: {e}",
                evidence={"error": str(e)},
                affected_agents=[agent_context.get('agent_name', 'unknown')],
                detection_method="exception_handler",
                confidence_score=1.0
            ))
        
        audit_duration = (datetime.now() - start_time).total_seconds() * 1000
        
        return SecurityAuditResult(
            is_secure=is_secure,
            confidence=max(0.1, min(1.0, confidence)),
            threats_detected=threats,
            violations=violations,
            recommendations=recommendations,
            audit_duration_ms=audit_duration,
            security_score=max(0.0, min(100.0, security_score))
        )
    
    async def quarantine_agent(self, agent_name: str, reason: str):
        """Quarantine an agent for security reasons"""
        self.quarantined_agents.add(agent_name)
        
        incident = {
            'incident_id': f"quarantine_{int(datetime.now().timestamp())}",
            'type': 'agent_quarantine',
            'agent_name': agent_name,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        self.security_incidents.append(incident)
        logger.critical(f"Agent {agent_name} quarantined: {reason}")
        global_metrics.incr("security.agent_quarantined")
    
    def is_agent_quarantined(self, agent_name: str) -> bool:
        """Check if agent is quarantined"""
        return agent_name in self.quarantined_agents
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics"""
        return {
            'security_level': self.security_level.value,
            'active_threats': len(self.active_threats),
            'security_incidents': len(self.security_incidents),
            'quarantined_agents': len(self.quarantined_agents),
            'threats_detected': self.threats_detected_count,
            'violations_prevented': self.violations_prevented_count,
            'memory_storages': len(self.memory_manager.encrypted_memories),
            'quarantined_agent_list': list(self.quarantined_agents)
        }


class SecurityError(Exception):
    """Custom exception for security-related errors"""
    pass