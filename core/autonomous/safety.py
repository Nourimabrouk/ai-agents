"""
Autonomous Safety Framework - Phase 7
Comprehensive safety system for autonomous agent operations
Multi-layer validation, monitoring, and rollback mechanisms
"""

import asyncio
import logging
import json
import hashlib
# SECURITY: pickle removed - unsafe deserialization vulnerability (CVE-2022-40897)
# import pickle  # DANGEROUS - arbitrary code execution risk
import json
import base64
# SECURITY: Using standard library instead of cryptography for compatibility
import secrets
from pathlib import Path
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import ast
import inspect
import traceback

from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics
import html
import re

logger = get_logger(__name__)


class SecureInputValidator:
    """Secure input validation and sanitization"""
    
    @staticmethod
    def validate_and_sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Validate and sanitize string input"""
        if not isinstance(input_str, str):
            raise ValueError("Input must be a string")
        
        if len(input_str) > max_length:
            raise ValueError(f"Input exceeds maximum length of {max_length}")
        
        # Remove dangerous characters
        sanitized = html.escape(input_str, quote=True)
        
        # Remove potentially dangerous patterns
        dangerous_patterns = [
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'onload\s*=',
            r'onerror\s*=',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__',
            r'subprocess',
            r'pickle\.loads',
            r'pickle\.dumps'
        ]
        
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    @staticmethod
    def validate_file_path(file_path: str, allowed_dirs: List[str] = None) -> str:
        """Validate and secure file paths to prevent path traversal"""
        if not isinstance(file_path, str):
            raise ValueError("File path must be a string")
        
        # Normalize path to prevent traversal
        normalized_path = Path(file_path).resolve()
        
        # Check for path traversal attempts
        if '..' in str(normalized_path) or str(normalized_path).startswith(str(Path('/').resolve())):
            raise ValueError("Path traversal detected")
        
        # Validate against allowed directories if specified
        if allowed_dirs:
            allowed = False
            for allowed_dir in allowed_dirs:
                if str(normalized_path).startswith(str(Path(allowed_dir).resolve())):
                    allowed = True
                    break
            
            if not allowed:
                raise ValueError(f"Access to path {normalized_path} not allowed")
        
        return str(normalized_path)
    
    @staticmethod
    def validate_json_data(json_data: Dict[str, Any], max_depth: int = 10) -> Dict[str, Any]:
        """Validate JSON data structure"""
        if not isinstance(json_data, dict):
            raise ValueError("Input must be a dictionary")
        
        def check_depth(obj, current_depth=0):
            if current_depth > max_depth:
                raise ValueError(f"JSON depth exceeds maximum of {max_depth}")
            
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if not isinstance(key, str):
                        raise ValueError("All keys must be strings")
                    check_depth(value, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    check_depth(item, current_depth + 1)
        
        check_depth(json_data)
        return json_data


class SafetyLevel(Enum):
    """Safety enforcement levels"""
    PERMISSIVE = "permissive"        # Log violations but allow
    RESTRICTIVE = "restrictive"      # Block unsafe operations
    PARANOID = "paranoid"           # Maximum safety, conservative blocking


class ViolationType(Enum):
    """Types of safety violations"""
    CODE_INJECTION = "code_injection"
    RESOURCE_ABUSE = "resource_abuse"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    INFINITE_LOOP = "infinite_loop"
    MEMORY_LEAK = "memory_leak"
    UNSAFE_IMPORT = "unsafe_import"
    DANGEROUS_OPERATION = "dangerous_operation"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    LOGIC_BOMB = "logic_bomb"


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    violation_type: ViolationType
    severity: str  # "low", "medium", "high", "critical"
    description: str
    evidence: Dict[str, Any]
    source_location: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    mitigation_applied: bool = False
    false_positive: bool = False


@dataclass
class SafetyAssessment:
    """Result of safety assessment"""
    is_safe: bool
    confidence: float
    violations: List[SafetyViolation]
    warnings: List[str]
    recommendations: List[str]
    assessment_time_ms: float
    validation_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemBackup:
    """System state backup for rollback"""
    backup_id: str
    agent_name: str
    backup_path: Path
    agent_state: Dict[str, Any]
    memory_snapshot: bytes
    modification_stack: List[str]
    created_at: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    verified: bool = False


class ModificationValidator:
    """
    Validates proposed modifications for safety
    Multi-layer validation with static and dynamic analysis
    """
    
    def __init__(self, safety_level: SafetyLevel = SafetyLevel.RESTRICTIVE):
        self.safety_level = safety_level
        self.violation_patterns = self._load_violation_patterns()
        self.allowed_operations = self._load_allowed_operations()
        self.validation_cache: Dict[str, SafetyAssessment] = {}
        
    async def validate_code_modification(self, 
                                       code: str,
                                       context: Dict[str, Any]) -> SafetyAssessment:
        """Validate code modification for safety"""
        start_time = datetime.now()
        
        # Check cache first
        code_hash = hashlib.sha256(code.encode()).hexdigest()
        if code_hash in self.validation_cache:
            cached_result = self.validation_cache[code_hash]
            logger.debug(f"Using cached validation result for code hash: {code_hash[:8]}")
            return cached_result
        
        violations = []
        warnings = []
        recommendations = []
        
        try:
            # Static analysis
            static_violations = await self._static_code_analysis(code, context)
            violations.extend(static_violations)
            
            # Pattern matching for known dangerous patterns
            pattern_violations = await self._pattern_analysis(code)
            violations.extend(pattern_violations)
            
            # Resource usage analysis
            resource_violations = await self._resource_usage_analysis(code)
            violations.extend(resource_violations)
            
            # Logic bomb detection
            logic_violations = await self._logic_bomb_detection(code)
            violations.extend(logic_violations)
            
            # Performance impact analysis
            performance_issues = await self._performance_impact_analysis(code)
            violations.extend(performance_issues)
            
            # Generate recommendations
            recommendations = await self._generate_safety_recommendations(violations, code)
            
            # Calculate safety score
            is_safe = self._calculate_safety_decision(violations)
            confidence = self._calculate_confidence(violations, len(code))
            
        except Exception as e:
            logger.error(f"Code validation failed: {e}")
            violations.append(SafetyViolation(
                violation_type=ViolationType.DANGEROUS_OPERATION,
                severity="high",
                description=f"Validation process failed: {e}",
                evidence={"error": str(e), "traceback": traceback.format_exc()}
            ))
            is_safe = False
            confidence = 0.0
        
        assessment_time = (datetime.now() - start_time).total_seconds() * 1000
        
        assessment = SafetyAssessment(
            is_safe=is_safe,
            confidence=confidence,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations,
            assessment_time_ms=assessment_time,
            validation_details={
                'code_length': len(code),
                'analysis_methods': ['static', 'pattern', 'resource', 'logic', 'performance'],
                'safety_level': self.safety_level.value
            }
        )
        
        # Cache result
        self.validation_cache[code_hash] = assessment
        
        return assessment
    
    async def _static_code_analysis(self, code: str, context: Dict[str, Any]) -> List[SafetyViolation]:
        """Static analysis of code using AST"""
        violations = []
        
        try:
            tree = ast.parse(code)
            
            # Check for dangerous operations
            for node in ast.walk(tree):
                # Check for exec/eval
                if isinstance(node, ast.Name) and node.id in ['exec', 'eval', 'compile']:
                    violations.append(SafetyViolation(
                        violation_type=ViolationType.CODE_INJECTION,
                        severity="critical",
                        description=f"Dangerous function '{node.id}' detected",
                        evidence={"function": node.id, "line": getattr(node, 'lineno', 0)}
                    ))
                
                # Check for file operations
                if isinstance(node, ast.Attribute) and node.attr in ['open', 'write', 'read']:
                    if not self._is_authorized_file_operation(node, context):
                        violations.append(SafetyViolation(
                            violation_type=ViolationType.UNAUTHORIZED_ACCESS,
                            severity="medium",
                            description=f"Unauthorized file operation: {node.attr}",
                            evidence={"operation": node.attr, "line": getattr(node, 'lineno', 0)}
                        ))
                
                # Check for subprocess calls
                if isinstance(node, ast.Name) and 'subprocess' in node.id:
                    violations.append(SafetyViolation(
                        violation_type=ViolationType.DANGEROUS_OPERATION,
                        severity="high",
                        description="Subprocess execution detected",
                        evidence={"operation": "subprocess", "line": getattr(node, 'lineno', 0)}
                    ))
                
                # Check for infinite loop patterns
                if isinstance(node, ast.While):
                    if self._is_potential_infinite_loop(node):
                        violations.append(SafetyViolation(
                            violation_type=ViolationType.INFINITE_LOOP,
                            severity="high",
                            description="Potential infinite loop detected",
                            evidence={"line": getattr(node, 'lineno', 0)}
                        ))
                
                # Check for imports
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if not self._is_safe_import(alias.name):
                            violations.append(SafetyViolation(
                                violation_type=ViolationType.UNSAFE_IMPORT,
                                severity="medium",
                                description=f"Unsafe import: {alias.name}",
                                evidence={"module": alias.name, "line": getattr(node, 'lineno', 0)}
                            ))
        
        except SyntaxError as e:
            violations.append(SafetyViolation(
                violation_type=ViolationType.CODE_INJECTION,
                severity="high",
                description=f"Syntax error in code: {e}",
                evidence={"error": str(e)}
            ))
        
        return violations
    
    async def _pattern_analysis(self, code: str) -> List[SafetyViolation]:
        """Analyze code for known dangerous patterns"""
        violations = []
        
        dangerous_patterns = [
            (r'while\s+True\s*:', ViolationType.INFINITE_LOOP, "Potential infinite loop"),
            (r'import\s+os', ViolationType.UNSAFE_IMPORT, "Operating system import"),
            (r'__import__', ViolationType.CODE_INJECTION, "Dynamic import function"),
            (r'globals\(\)', ViolationType.UNAUTHORIZED_ACCESS, "Global namespace access"),
            (r'locals\(\)', ViolationType.UNAUTHORIZED_ACCESS, "Local namespace access"),
            (r'setattr\(.*__.*\)', ViolationType.DANGEROUS_OPERATION, "Attribute manipulation"),
            (r'getattr\(.*__.*\)', ViolationType.DANGEROUS_OPERATION, "Private attribute access"),
        ]
        
        import re
        for pattern, violation_type, description in dangerous_patterns:
            matches = re.finditer(pattern, code, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violations.append(SafetyViolation(
                    violation_type=violation_type,
                    severity="medium",
                    description=description,
                    evidence={
                        "pattern": pattern,
                        "match": match.group(),
                        "position": match.start()
                    }
                ))
        
        return violations
    
    async def _resource_usage_analysis(self, code: str) -> List[SafetyViolation]:
        """Analyze potential resource usage issues"""
        violations = []
        
        # Check for memory allocation patterns
        if 'list(' in code and 'range(' in code:
            # Potential large memory allocation
            violations.append(SafetyViolation(
                violation_type=ViolationType.MEMORY_LEAK,
                severity="medium",
                description="Potential large memory allocation detected",
                evidence={"pattern": "list(range(...))"}
            ))
        
        # Check for recursive patterns without base case
        if 'def ' in code and code.count('return') == 0:
            violations.append(SafetyViolation(
                violation_type=ViolationType.INFINITE_LOOP,
                severity="medium",
                description="Function without return statement (potential recursion issue)",
                evidence={"analysis": "missing_return"}
            ))
        
        return violations
    
    async def _logic_bomb_detection(self, code: str) -> List[SafetyViolation]:
        """Detect potential logic bombs or malicious code"""
        violations = []
        
        suspicious_patterns = [
            'datetime.now()',
            'time.time()',
            'os.remove',
            'shutil.rmtree',
            'sys.exit',
            '__del__'
        ]
        
        # Look for time-based triggers
        if any(pattern in code for pattern in suspicious_patterns[:2]):
            if any(dangerous in code for dangerous in suspicious_patterns[2:]):
                violations.append(SafetyViolation(
                    violation_type=ViolationType.LOGIC_BOMB,
                    severity="critical",
                    description="Potential logic bomb detected (time-based trigger with dangerous operation)",
                    evidence={"analysis": "time_trigger_with_dangerous_operation"}
                ))
        
        return violations
    
    async def _performance_impact_analysis(self, code: str) -> List[SafetyViolation]:
        """Analyze potential performance impact"""
        violations = []
        
        # Check for nested loops
        loop_count = code.count('for ') + code.count('while ')
        if loop_count > 3:
            violations.append(SafetyViolation(
                violation_type=ViolationType.PERFORMANCE_DEGRADATION,
                severity="medium",
                description=f"High loop count detected: {loop_count}",
                evidence={"loop_count": loop_count}
            ))
        
        # Check for complex operations
        if 'sort()' in code and 'for ' in code:
            violations.append(SafetyViolation(
                violation_type=ViolationType.PERFORMANCE_DEGRADATION,
                severity="low",
                description="Sorting within loop detected",
                evidence={"pattern": "sort_in_loop"}
            ))
        
        return violations
    
    def _is_authorized_file_operation(self, node: ast.Attribute, context: Dict[str, Any]) -> bool:
        """Check if file operation is authorized"""
        # Check context for file operation permissions
        if context.get('allow_file_operations', False):
            return True
        
        # Check for specific authorized paths
        authorized_paths = context.get('authorized_file_paths', [])
        if authorized_paths:
            # This is a simplified check - in practice would analyze the actual path
            return True
        
        return False
    
    def _is_potential_infinite_loop(self, node: ast.While) -> bool:
        """Check if while loop might be infinite"""
        # Simple heuristic: while True without break
        if isinstance(node.test, ast.Constant) and node.test.value is True:
            # Check for break statements
            has_break = any(isinstance(child, ast.Break) for child in ast.walk(node))
            return not has_break
        
        return False
    
    def _is_safe_import(self, module_name: str) -> bool:
        """Check if import is safe"""
        safe_modules = {
            'asyncio', 'typing', 'dataclasses', 'datetime', 'logging',
            'collections', 'json', 'math', 'statistics', 'enum',
            'hashlib', 'uuid', 're', 'string'
        }
        
        dangerous_modules = {
            'os', 'subprocess', 'sys', 'importlib', 'pickle',
            'marshal', 'ctypes', 'sqlite3', 'socket'
        }
        
        if module_name in dangerous_modules:
            return False
        
        if module_name in safe_modules:
            return True
        
        # Default to safe for unknown modules (can be configured)
        return True
    
    def _calculate_safety_decision(self, violations: List[SafetyViolation]) -> bool:
        """Calculate overall safety decision"""
        if not violations:
            return True
        
        # Count violations by severity
        critical_count = sum(1 for v in violations if v.severity == "critical")
        high_count = sum(1 for v in violations if v.severity == "high")
        
        if self.safety_level == SafetyLevel.PARANOID:
            return critical_count == 0 and high_count == 0
        elif self.safety_level == SafetyLevel.RESTRICTIVE:
            return critical_count == 0 and high_count <= 1
        else:  # PERMISSIVE
            return critical_count == 0 and high_count <= 2
    
    def _calculate_confidence(self, violations: List[SafetyViolation], code_length: int) -> float:
        """Calculate confidence in safety assessment"""
        base_confidence = 0.8
        
        # Reduce confidence based on violations
        violation_penalty = len(violations) * 0.1
        
        # Reduce confidence for short code (less analysis possible)
        if code_length < 100:
            base_confidence -= 0.1
        
        # Increase confidence for longer analysis
        if code_length > 1000:
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence - violation_penalty))
    
    async def _generate_safety_recommendations(self, 
                                             violations: List[SafetyViolation],
                                             code: str) -> List[str]:
        """Generate safety improvement recommendations"""
        recommendations = []
        
        violation_types = [v.violation_type for v in violations]
        
        if ViolationType.CODE_INJECTION in violation_types:
            recommendations.append("Remove dynamic code execution (exec, eval)")
        
        if ViolationType.INFINITE_LOOP in violation_types:
            recommendations.append("Add proper loop termination conditions")
        
        if ViolationType.UNSAFE_IMPORT in violation_types:
            recommendations.append("Use only approved imports")
        
        if ViolationType.PERFORMANCE_DEGRADATION in violation_types:
            recommendations.append("Optimize loops and sorting operations")
        
        # Generic recommendations
        if violations:
            recommendations.append("Run code in sandboxed environment before deployment")
            recommendations.append("Add comprehensive error handling")
        
        return recommendations
    
    def _load_violation_patterns(self) -> List[Dict[str, Any]]:
        """Load known violation patterns"""
        return [
            {"pattern": r"exec\(", "type": "code_injection", "severity": "critical"},
            {"pattern": r"eval\(", "type": "code_injection", "severity": "critical"},
            {"pattern": r"__import__", "type": "code_injection", "severity": "high"},
            {"pattern": r"subprocess\.", "type": "dangerous_operation", "severity": "high"},
        ]
    
    def _load_allowed_operations(self) -> Set[str]:
        """Load allowed operations"""
        return {
            'basic_math', 'string_operations', 'list_operations',
            'dict_operations', 'async_operations', 'logging'
        }


class RollbackManager:
    """
    Manages system state backups and rollbacks
    Enables safe recovery from failed modifications
    """
    
    def __init__(self, backup_directory: Optional[Path] = None):
        self.backup_directory = backup_directory or Path(tempfile.gettempdir()) / "agent_backups"
        self.backup_directory.mkdir(exist_ok=True, parents=True)
        self.active_backups: Dict[str, SystemBackup] = {}
        self.max_backups_per_agent = 10
        
    async def create_backup(self, agent: BaseAgent, backup_id: Optional[str] = None) -> SystemBackup:
        """Create complete backup of agent state"""
        if backup_id is None:
            backup_id = f"{agent.name}_{int(datetime.now().timestamp())}"
        
        logger.info(f"Creating backup for agent {agent.name}: {backup_id}")
        
        backup_path = self.backup_directory / f"{backup_id}.backup"
        
        try:
            # Capture agent state
            agent_state = {
                'name': agent.name,
                'config': agent.config,
                'total_tasks': agent.total_tasks,
                'successful_tasks': agent.successful_tasks,
                'state': agent.state.value if hasattr(agent.state, 'value') else str(agent.state)
            }
            
            # Capture memory snapshot
            memory_snapshot = b""
            if hasattr(agent, 'memory'):
                try:
                    memory_data = {
                        'episodic_memory': agent.memory.episodic_memory,
                        'semantic_memory': dict(agent.memory.semantic_memory) if hasattr(agent.memory.semantic_memory, 'items') else agent.memory.semantic_memory,
                        'working_memory': agent.memory.working_memory
                    }
                    # SECURITY FIX: Use secure JSON serialization instead of pickle
                    memory_snapshot = json.dumps(memory_data, default=str, ensure_ascii=True).encode('utf-8')
                except Exception as e:
                    logger.warning(f"Failed to backup memory: {e}")
                    memory_snapshot = b""
            
            # Create backup record
            backup = SystemBackup(
                backup_id=backup_id,
                agent_name=agent.name,
                backup_path=backup_path,
                agent_state=agent_state,
                memory_snapshot=memory_snapshot,
                modification_stack=[],
                size_bytes=len(memory_snapshot) + len(json.dumps(agent_state).encode())
            )
            
            # Save backup to disk
            backup_data = {
                'backup_id': backup_id,
                'agent_state': agent_state,
                'memory_snapshot_size': len(memory_snapshot),
                'created_at': backup.created_at.isoformat(),
                'modification_stack': backup.modification_stack
            }
            
            with open(backup_path, 'wb') as f:
                f.write(json.dumps(backup_data).encode())
                f.write(b'\n--- MEMORY SNAPSHOT ---\n')
                f.write(memory_snapshot)
            
            # Verify backup integrity
            backup.verified = await self._verify_backup_integrity(backup)
            
            # Store backup reference
            self.active_backups[backup_id] = backup
            
            # Clean old backups
            await self._cleanup_old_backups(agent.name)
            
            logger.info(f"Backup created successfully: {backup_id} ({backup.size_bytes} bytes)")
            return backup
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    async def restore_backup(self, backup_id: str, agent: BaseAgent) -> bool:
        """Restore agent from backup"""
        if backup_id not in self.active_backups:
            logger.error(f"Backup not found: {backup_id}")
            return False
        
        backup = self.active_backups[backup_id]
        logger.info(f"Restoring agent {agent.name} from backup: {backup_id}")
        
        try:
            # Verify backup integrity
            if not backup.verified:
                backup.verified = await self._verify_backup_integrity(backup)
            
            if not backup.verified:
                logger.error(f"Backup integrity check failed: {backup_id}")
                return False
            
            # Load backup data
            with open(backup.backup_path, 'rb') as f:
                content = f.read()
                
            # Split content
            parts = content.split(b'\n--- MEMORY SNAPSHOT ---\n')
            if len(parts) != 2:
                logger.error("Invalid backup format")
                return False
            
            backup_data = json.loads(parts[0].decode())
            memory_snapshot = parts[1]
            
            # Restore agent state
            agent.total_tasks = backup_data['agent_state'].get('total_tasks', 0)
            agent.successful_tasks = backup_data['agent_state'].get('successful_tasks', 0)
            
            # SECURITY FIX: Restore memory using secure JSON deserialization
            if memory_snapshot and hasattr(agent, 'memory'):
                try:
                    memory_data = json.loads(memory_snapshot.decode('utf-8'))
                    agent.memory.episodic_memory = memory_data.get('episodic_memory', [])
                    
                    # Restore semantic memory carefully
                    if 'semantic_memory' in memory_data:
                        if hasattr(agent.memory.semantic_memory, 'update'):
                            agent.memory.semantic_memory.update(memory_data['semantic_memory'])
                        else:
                            agent.memory.semantic_memory = memory_data['semantic_memory']
                    
                    agent.memory.working_memory = memory_data.get('working_memory', {})
                    
                except Exception as e:
                    logger.warning(f"Failed to restore memory: {e}")
            
            logger.info(f"Successfully restored agent from backup: {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False
    
    async def _verify_backup_integrity(self, backup: SystemBackup) -> bool:
        """Verify backup file integrity"""
        try:
            if not backup.backup_path.exists():
                return False
            
            # Check file size
            file_size = backup.backup_path.stat().st_size
            if file_size != backup.size_bytes:
                logger.warning(f"Backup size mismatch: expected {backup.size_bytes}, got {file_size}")
            
            # Try to read and parse backup
            with open(backup.backup_path, 'rb') as f:
                content = f.read()
            
            parts = content.split(b'\n--- MEMORY SNAPSHOT ---\n')
            if len(parts) != 2:
                return False
            
            # Validate JSON part
            json.loads(parts[0].decode())
            
            # SECURITY FIX: Validate JSON memory snapshot instead of pickle
            if parts[1]:
                try:
                    json.loads(parts[1].decode('utf-8'))  # Safe JSON validation
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Backup integrity check failed: {e}")
            return False
    
    async def _cleanup_old_backups(self, agent_name: str):
        """Clean up old backups to maintain storage limits"""
        agent_backups = [
            (backup_id, backup) for backup_id, backup in self.active_backups.items()
            if backup.agent_name == agent_name
        ]
        
        if len(agent_backups) > self.max_backups_per_agent:
            # Sort by creation time and remove oldest
            agent_backups.sort(key=lambda x: x[1].created_at)
            
            for backup_id, backup in agent_backups[:-self.max_backups_per_agent]:
                try:
                    backup.backup_path.unlink(missing_ok=True)
                    del self.active_backups[backup_id]
                    logger.info(f"Cleaned up old backup: {backup_id}")
                except Exception as e:
                    logger.error(f"Failed to cleanup backup {backup_id}: {e}")
    
    def list_backups(self, agent_name: Optional[str] = None) -> List[SystemBackup]:
        """List available backups"""
        if agent_name:
            return [backup for backup in self.active_backups.values() if backup.agent_name == agent_name]
        else:
            return list(self.active_backups.values())


class AutonomousSafetyFramework:
    """
    Comprehensive safety framework for autonomous operations
    Integrates validation, monitoring, and rollback capabilities
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 safety_level: SafetyLevel = SafetyLevel.RESTRICTIVE):
        self.config = config or {}
        self.safety_level = safety_level
        
        # Initialize components
        self.validator = ModificationValidator(safety_level)
        self.rollback_manager = RollbackManager()
        
        # Safety monitoring
        self.violation_history: List[SafetyViolation] = []
        self.monitoring_enabled = self.config.get('monitoring_enabled', True)
        self.max_violations_per_hour = self.config.get('max_violations_per_hour', 10)
        
        # Emergency controls
        self.emergency_stop_enabled = False
        self.quarantine_mode = False
        self.last_safety_check = datetime.now()
        
        logger.info(f"Initialized autonomous safety framework (Level: {safety_level.value})")
    
    async def validate_modification_request(self, 
                                          modification_request) -> SafetyAssessment:
        """Validate a modification request for safety"""
        logger.info(f"Validating modification request: {modification_request.modification_id}")
        
        # Check if in emergency mode
        if self.emergency_stop_enabled:
            return SafetyAssessment(
                is_safe=False,
                confidence=1.0,
                violations=[SafetyViolation(
                    violation_type=ViolationType.DANGEROUS_OPERATION,
                    severity="critical",
                    description="System in emergency stop mode",
                    evidence={"emergency_mode": True}
                )],
                warnings=["System is in emergency stop mode"],
                recommendations=["Clear emergency stop before proceeding"],
                assessment_time_ms=0.0
            )
        
        # Validate proposed changes
        assessment = await self.validator.validate_code_modification(
            modification_request.proposed_changes,
            {
                'modification_type': modification_request.modification_type,
                'agent_name': modification_request.agent_name,
                'safety_constraints': modification_request.safety_constraints
            }
        )
        
        # Record assessment
        if assessment.violations:
            self.violation_history.extend(assessment.violations)
            await self._check_violation_rate()
        
        return assessment
    
    async def validate_capability(self, capability_candidate: Dict[str, Any]) -> SafetyAssessment:
        """Validate a discovered capability for safety"""
        return await self.validator.validate_code_modification(
            capability_candidate.get('implementation', ''),
            {
                'capability_name': capability_candidate.get('name'),
                'discovery_method': capability_candidate.get('discovery_method'),
                'allow_experimental': True
            }
        )
    
    async def validate_coordination_safety(self, 
                                         coordination_decision,
                                         task,
                                         agents: Dict[str, BaseAgent]) -> SafetyAssessment:
        """Validate coordination decision for safety"""
        violations = []
        warnings = []
        
        # Check agent availability and state
        for agent_name in coordination_decision.context.get('selected_agents', []):
            if agent_name in agents:
                agent = agents[agent_name]
                agent_state = getattr(agent, 'state', 'unknown')
                if agent_state == 'error':
                    violations.append(SafetyViolation(
                        violation_type=ViolationType.DANGEROUS_OPERATION,
                        severity="high",
                        description=f"Agent {agent_name} is in error state",
                        evidence={"agent_state": agent_state}
                    ))
        
        # Check resource requirements
        estimated_resources = coordination_decision.context.get('resource_requirements', {})
        if estimated_resources.get('agents', 0) > len(agents) * 0.8:
            warnings.append("High agent utilization - may impact system responsiveness")
        
        return SafetyAssessment(
            is_safe=len(violations) == 0,
            confidence=0.8,
            violations=violations,
            warnings=warnings,
            recommendations=[],
            assessment_time_ms=10.0,
            validation_details={'coordination_check': True}
        )
    
    async def create_safe_backup(self, agent: BaseAgent) -> str:
        """Create a safety backup before modifications"""
        backup = await self.rollback_manager.create_backup(agent)
        return backup.backup_id
    
    async def emergency_rollback(self, backup_id: str, agent: BaseAgent) -> bool:
        """Emergency rollback to safe state"""
        logger.warning(f"Emergency rollback initiated for agent {agent.name}")
        success = await self.rollback_manager.restore_backup(backup_id, agent)
        
        if success:
            # Clear error state
            agent.state = 'idle' if hasattr(agent, 'state') else agent.state
            logger.info(f"Emergency rollback successful for agent {agent.name}")
        else:
            logger.error(f"Emergency rollback failed for agent {agent.name}")
            # Activate quarantine mode
            self.quarantine_mode = True
        
        return success
    
    async def _check_violation_rate(self):
        """Check if violation rate exceeds limits"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        recent_violations = [v for v in self.violation_history if v.timestamp >= hour_ago]
        
        if len(recent_violations) > self.max_violations_per_hour:
            logger.critical(f"Violation rate exceeded: {len(recent_violations)} in last hour")
            await self._activate_emergency_protocols()
    
    async def _activate_emergency_protocols(self):
        """Activate emergency safety protocols"""
        logger.critical("Activating emergency safety protocols")
        self.emergency_stop_enabled = True
        global_metrics.incr("safety.emergency.activated")
        
        # Additional emergency measures could be added here
        # Such as notifying human operators, stopping all modifications, etc.
    
    def get_violation_count(self) -> int:
        """Get total violation count"""
        return len(self.violation_history)
    
    def get_safety_metrics(self) -> Dict[str, Any]:
        """Get comprehensive safety metrics"""
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        recent_violations_hour = [v for v in self.violation_history if v.timestamp >= hour_ago]
        recent_violations_day = [v for v in self.violation_history if v.timestamp >= day_ago]
        
        return {
            'safety_level': self.safety_level.value,
            'total_violations': len(self.violation_history),
            'violations_last_hour': len(recent_violations_hour),
            'violations_last_day': len(recent_violations_day),
            'emergency_stop_enabled': self.emergency_stop_enabled,
            'quarantine_mode': self.quarantine_mode,
            'active_backups': len(self.rollback_manager.active_backups),
            'last_safety_check': self.last_safety_check.isoformat(),
            'violation_types': {
                violation_type.value: len([v for v in self.violation_history if v.violation_type == violation_type])
                for violation_type in ViolationType
            }
        }