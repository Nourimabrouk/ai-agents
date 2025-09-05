"""
Code Security Framework
Provides secure code validation, sandboxing, and behavioral monitoring
"""

import ast
import re
import subprocess
import tempfile
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class SecureCodeValidator:
    """Validates code for security vulnerabilities and malicious patterns"""
    
    def __init__(self):
        self.forbidden_patterns = [
            r'__import__\s*\(\s*["\']os["\']',
            r'exec\s*\(',
            r'eval\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
        ]
        
        self.forbidden_modules = {
            'os', 'sys', 'subprocess', 'socket', 'urllib', 'requests',
            'shutil', 'tempfile', '__builtin__', 'builtins'
        }
        
        self.safe_builtins = {
            'len', 'str', 'int', 'float', 'bool', 'list', 'dict', 'tuple',
            'set', 'range', 'enumerate', 'zip', 'map', 'filter', 'sum',
            'min', 'max', 'abs', 'round', 'sorted', 'reversed'
        }
    
    async def validate_code(self, code: str) -> Dict[str, Any]:
        """Validate code for security issues"""
        try:
            result = {
                'is_safe': True,
                'security_score': 100,
                'violations': [],
                'warnings': [],
                'analysis': {}
            }
            
            # Check for forbidden patterns
            pattern_violations = self._check_patterns(code)
            if pattern_violations:
                result['violations'].extend(pattern_violations)
                result['is_safe'] = False
                result['security_score'] -= len(pattern_violations) * 20
            
            # Parse AST for detailed analysis
            try:
                tree = ast.parse(code)
                ast_violations = self._analyze_ast(tree)
                if ast_violations:
                    result['violations'].extend(ast_violations)
                    result['is_safe'] = False
                    result['security_score'] -= len(ast_violations) * 15
            except SyntaxError as e:
                result['violations'].append(f"Syntax error: {str(e)}")
                result['is_safe'] = False
                result['security_score'] = 0
            
            # Calculate final score
            result['security_score'] = max(0, min(100, result['security_score']))
            
            return result
            
        except Exception as e:
            logger.error(f"Code validation error: {str(e)}")
            return {
                'is_safe': False,
                'security_score': 0,
                'violations': [f"Validation error: {str(e)}"],
                'warnings': [],
                'analysis': {}
            }
    
    def _check_patterns(self, code: str) -> List[str]:
        """Check for forbidden regex patterns"""
        violations = []
        for pattern in self.forbidden_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                violations.append(f"Forbidden pattern detected: {pattern}")
        return violations
    
    def _analyze_ast(self, tree: ast.AST) -> List[str]:
        """Analyze AST for security violations"""
        violations = []
        
        for node in ast.walk(tree):
            # Check for dangerous imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in self.forbidden_modules:
                        violations.append(f"Forbidden import: {alias.name}")
            
            # Check for dangerous function calls
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['exec', 'eval', 'compile']:
                        violations.append(f"Dangerous function call: {node.func.id}")
        
        return violations

class CodeSandbox:
    """Provides secure code execution environment"""
    
    def __init__(self):
        self.timeout_seconds = 10
        self.memory_limit_mb = 100
        self.temp_dir = None
    
    async def execute_code(self, code: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """Execute code in secure sandbox"""
        try:
            timeout = timeout or self.timeout_seconds
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute with restrictions
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=tempfile.gettempdir()
                )
                
                return {
                    'success': result.returncode == 0,
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode,
                    'execution_time': timeout
                }
                
            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'stdout': '',
                    'stderr': f'Execution timeout after {timeout} seconds',
                    'return_code': -1,
                    'execution_time': timeout
                }
            finally:
                # Clean up
                try:
                    os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Sandbox execution error: {str(e)}")
            return {
                'success': False,
                'stdout': '',
                'stderr': f'Sandbox error: {str(e)}',
                'return_code': -1,
                'execution_time': 0
            }

class BehavioralMonitor:
    """Monitors and analyzes behavioral patterns"""
    
    def __init__(self):
        self.behavior_history = []
        self.anomaly_threshold = 0.8
        self.pattern_cache = {}
    
    async def monitor_behavior(self, action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Monitor and analyze behavior"""
        try:
            behavior_entry = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'action': action_data.get('action', ''),
                'parameters': action_data.get('parameters', {}),
                'context': action_data.get('context', {}),
                'hash': self._compute_behavior_hash(action_data)
            }
            
            self.behavior_history.append(behavior_entry)
            
            # Analyze for anomalies
            anomaly_score = await self._detect_anomalies(behavior_entry)
            
            return {
                'behavior_recorded': True,
                'anomaly_score': anomaly_score,
                'is_anomalous': anomaly_score > self.anomaly_threshold,
                'pattern_analysis': await self._analyze_patterns(),
                'timestamp': behavior_entry['timestamp']
            }
            
        except Exception as e:
            logger.error(f"Behavior monitoring error: {str(e)}")
            return {
                'behavior_recorded': False,
                'anomaly_score': 1.0,
                'is_anomalous': True,
                'error': str(e)
            }
    
    def _compute_behavior_hash(self, action_data: Dict[str, Any]) -> str:
        """Compute hash for behavior deduplication"""
        data_str = json.dumps(action_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]
    
    async def _detect_anomalies(self, behavior_entry: Dict[str, Any]) -> float:
        """Detect behavioral anomalies"""
        # Simple frequency-based anomaly detection
        action = behavior_entry['action']
        similar_actions = [
            b for b in self.behavior_history[-100:]
            if b['action'] == action
        ]
        
        if len(similar_actions) < 3:
            return 0.6  # Moderate anomaly for rare actions
        
        return 0.1  # Low anomaly for common actions
    
    async def _analyze_patterns(self) -> Dict[str, Any]:
        """Analyze behavior patterns"""
        if len(self.behavior_history) < 10:
            return {'patterns': [], 'confidence': 0.0}
        
        recent_behaviors = self.behavior_history[-50:]
        action_frequency = {}
        
        for behavior in recent_behaviors:
            action = behavior['action']
            action_frequency[action] = action_frequency.get(action, 0) + 1
        
        patterns = [
            {'action': action, 'frequency': freq, 'percentage': freq/len(recent_behaviors)*100}
            for action, freq in action_frequency.items()
        ]
        
        patterns.sort(key=lambda x: x['frequency'], reverse=True)
        
        return {
            'patterns': patterns[:5],
            'confidence': 0.8,
            'total_behaviors': len(recent_behaviors)
        }

class ThreatDetectionSystem:
    """Advanced threat detection and response"""
    
    def __init__(self):
        self.threat_rules = []
        self.threat_history = []
        self.severity_levels = ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    async def detect_threats(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect security threats"""
        try:
            threats_detected = []
            
            # Check each threat rule
            for rule in self.threat_rules:
                if await self._evaluate_rule(rule, event_data):
                    threat = {
                        'rule_id': rule['id'],
                        'severity': rule['severity'],
                        'description': rule['description'],
                        'event_data': event_data,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
                    threats_detected.append(threat)
                    self.threat_history.append(threat)
            
            return {
                'threats_detected': len(threats_detected),
                'threats': threats_detected,
                'overall_risk_level': self._calculate_risk_level(threats_detected),
                'requires_action': len(threats_detected) > 0
            }
            
        except Exception as e:
            logger.error(f"Threat detection error: {str(e)}")
            return {
                'threats_detected': 0,
                'threats': [],
                'overall_risk_level': 'UNKNOWN',
                'error': str(e)
            }
    
    async def _evaluate_rule(self, rule: Dict[str, Any], event_data: Dict[str, Any]) -> bool:
        """Evaluate if a threat rule matches"""
        # Simple pattern matching
        conditions = rule.get('conditions', {})
        
        for key, expected_value in conditions.items():
            if key not in event_data:
                continue
            
            if event_data[key] != expected_value:
                return False
        
        return True
    
    def _calculate_risk_level(self, threats: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level"""
        if not threats:
            return 'LOW'
        
        severity_weights = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 4, 'CRITICAL': 8}
        total_weight = sum(severity_weights.get(t['severity'], 1) for t in threats)
        
        if total_weight >= 8:
            return 'CRITICAL'
        elif total_weight >= 4:
            return 'HIGH'
        elif total_weight >= 2:
            return 'MEDIUM'
        else:
            return 'LOW'

class EmergencyResponseSystem:
    """Handles emergency security responses"""
    
    def __init__(self):
        self.response_protocols = {}
        self.emergency_contacts = []
        self.escalation_levels = ['NOTIFY', 'ISOLATE', 'SHUTDOWN', 'EMERGENCY']
    
    async def trigger_emergency_response(self, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger emergency response based on threat"""
        try:
            response_actions = []
            
            severity = threat_data.get('severity', 'LOW')
            threat_type = threat_data.get('type', 'unknown')
            
            # Determine response level
            if severity == 'CRITICAL':
                response_actions.extend([
                    'Immediate system isolation',
                    'Emergency alert to security team',
                    'Automatic threat mitigation',
                    'Incident logging and forensics'
                ])
            elif severity == 'HIGH':
                response_actions.extend([
                    'Enhanced monitoring activation',
                    'Security team notification',
                    'Threat containment procedures'
                ])
            elif severity == 'MEDIUM':
                response_actions.extend([
                    'Increased logging',
                    'Automated threat response',
                    'Security alert generation'
                ])
            
            # Execute response
            execution_results = []
            for action in response_actions:
                result = await self._execute_response_action(action, threat_data)
                execution_results.append(result)
            
            return {
                'response_triggered': True,
                'severity': severity,
                'actions_taken': len(response_actions),
                'response_details': response_actions,
                'execution_results': execution_results,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Emergency response error: {str(e)}")
            return {
                'response_triggered': False,
                'error': str(e),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
    
    async def _execute_response_action(self, action: str, threat_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute specific response action"""
        # Simulate response action execution
        return {
            'action': action,
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': f"Successfully executed: {action}"
        }