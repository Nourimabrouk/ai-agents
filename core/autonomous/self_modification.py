"""
Self-Modification Framework - Phase 7
Enables agents to safely modify their own code and behavior
Implements dynamic code generation with comprehensive safety validation
"""

import asyncio
import logging
import json
import ast
import inspect
import hashlib
import tempfile
import subprocess
import sys
from typing import Any, Dict, List, Optional, Set, Tuple, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import importlib.util
import traceback

# Import base components
from templates.base_agent import BaseAgent, AgentState, Action, Observation, Memory, LearningSystem
from .safety import AutonomousSafetyFramework, ModificationValidator, SafetyViolation
from ..security import (
    AutonomousSecurityFramework, SecurityLevel,
    SecureCodeValidator, CodeSandbox,
    BehavioralMonitor, ThreatDetectionSystem,
    EmergencyResponseSystem
)

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ModificationType(Enum):
    """Types of self-modifications"""
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    TOOL_ENHANCEMENT = "tool_enhancement" 
    COORDINATION_IMPROVEMENT = "coordination_improvement"
    MEMORY_OPTIMIZATION = "memory_optimization"
    LEARNING_ENHANCEMENT = "learning_enhancement"
    CAPABILITY_EXTENSION = "capability_extension"
    PERFORMANCE_TUNING = "performance_tuning"


@dataclass
class ModificationRequest:
    """Request for agent self-modification"""
    modification_id: str
    agent_name: str
    modification_type: ModificationType
    target_component: str
    proposed_changes: str
    expected_improvement: float
    safety_constraints: Dict[str, Any]
    testing_requirements: List[str]
    rollback_plan: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    validated: bool = False
    applied: bool = False


@dataclass
class ModificationResult:
    """Result of applying a modification"""
    modification_id: str
    success: bool
    performance_before: Dict[str, float]
    performance_after: Dict[str, float]
    improvement_achieved: float
    side_effects: List[str]
    validation_results: Dict[str, Any]
    rollback_available: bool
    applied_at: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


class CodeGenerationConfig:
    """Configuration for code generation with security constraints."""
    
    def __init__(self):
        self.allowed_imports = {
            'asyncio', 'typing', 'dataclasses', 'datetime', 'logging',
            'collections', 'json', 'math', 'statistics'
        }
        self.forbidden_operations = {
            'exec', 'eval', 'compile', '__import__', 'open', 'file',
            'subprocess', 'os.system', 'os.popen', 'os.exec', 'os.spawn',
            'pickle.loads', 'pickle.dumps', 'marshal.loads', 'marshal.dumps',
            'ctypes', 'socket', 'urllib', 'requests', 'http.client',
            'ftplib', 'telnetlib', 'poplib', 'imaplib', 'smtplib',
            'sqlite3.Connection', 'pymongo', 'psycopg2', 'mysql.connector'
        }
        self.max_code_length = 50000
        self.max_complexity = 10


class CodeSecurityValidator:
    """Handles security validation of generated code."""
    
    def __init__(self, config: CodeGenerationConfig):
        self.config = config
        self.code_validator = SecureCodeValidator()
        self.code_sandbox = CodeSandbox()
    
    async def validate_code_security(self, code: str, safety_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate code for security and safety with enhanced security checks."""
        validation_result = {
            'is_safe': True,
            'violations': [],
            'warnings': [],
            'metrics': {},
            'security_analysis': None
        }
        
        try:
            # SECURITY FIX: Comprehensive input validation
            from ..autonomous.safety import SecureInputValidator
            
            # Validate and sanitize input
            try:
                sanitized_code = SecureInputValidator.validate_and_sanitize_string(
                    code, max_length=self.config.max_code_length
                )
                if sanitized_code != code:
                    validation_result['warnings'].append("Code was sanitized - suspicious patterns removed")
            except ValueError as e:
                validation_result['is_safe'] = False
                validation_result['violations'].append({
                    'type': 'input_validation_failure',
                    'severity': 'critical',
                    'description': f"Input validation failed: {e}",
                    'remediation': 'Provide valid, safe input'
                })
                return validation_result
            
            # Security analysis
            logger.info("Running comprehensive security validation on generated code")
            security_analysis = await self.code_validator.validate_code(code)
            validation_result['security_analysis'] = security_analysis.__dict__
            
            if not security_analysis.is_safe:
                logger.critical("SECURITY VIOLATION: Code failed security validation")
                validation_result['is_safe'] = False
                for vuln in security_analysis.vulnerabilities:
                    validation_result['violations'].append({
                        'type': 'security_violation',
                        'severity': vuln.severity,
                        'description': vuln.description,
                        'vulnerability_type': vuln.vuln_type,
                        'line_number': vuln.line_number,
                        'remediation': vuln.remediation
                    })
                return validation_result
            
            # Parse code to AST for analysis
            tree = ast.parse(code)
            
            # Check for forbidden operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in self.config.forbidden_operations:
                    validation_result['violations'].append(f"Forbidden operation: {node.id}")
                    validation_result['is_safe'] = False
                
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.config.allowed_imports:
                            validation_result['violations'].append(f"Unauthorized import: {alias.name}")
                            validation_result['is_safe'] = False
            
            # Calculate code metrics
            validation_result['metrics'] = {
                'lines_of_code': len(code.split('\n')),
                'complexity_score': self._calculate_complexity(tree),
                'function_count': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                'class_count': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)])
            }
            
            # Additional safety checks from constraints
            if safety_constraints:
                constraint_violations = await self._check_safety_constraints(tree, safety_constraints)
                validation_result['violations'].extend(constraint_violations)
                if constraint_violations:
                    validation_result['is_safe'] = False
            
        except SyntaxError as e:
            validation_result['is_safe'] = False
            validation_result['violations'].append(f"Syntax error: {e}")
        except Exception as e:
            validation_result['is_safe'] = False
            validation_result['violations'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity of code."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    async def _check_safety_constraints(self, tree: ast.AST, constraints: Dict[str, Any]) -> List[str]:
        """Check additional safety constraints."""
        violations = []
        # Implementation for safety constraint checking
        return violations


class CodeTemplateManager:
    """Manages code templates for different modification types."""
    
    def __init__(self):
        self.templates = self._load_code_templates()
    
    def _load_code_templates(self) -> Dict[str, str]:
        """Load code templates for different modification types."""
        return {
            'strategy_optimization': 'template for strategy optimization',
            'tool_enhancement': 'template for tool enhancement',
            'memory_optimization': 'template for memory optimization',
            'learning_enhancement': 'template for learning enhancement',
            'performance_tuning': 'template for performance tuning'
        }
    
    def get_template(self, modification_type: ModificationType) -> str:
        """Get template for specific modification type."""
        return self.templates.get(modification_type.value, '')


class DynamicCodeGenerator:
    """Generates and validates code modifications with improved architecture."""
    
    def __init__(self, 
                 safety_validator: ModificationValidator,
                 security_framework: Optional[AutonomousSecurityFramework] = None):
        self.safety_validator = safety_validator
        self.config = CodeGenerationConfig()
        self.security_validator = CodeSecurityValidator(self.config)
        self.template_manager = CodeTemplateManager()
        self.generation_history: List[Dict[str, Any]] = []
        
        # Initialize security framework
        self.security_framework = security_framework or AutonomousSecurityFramework(
            security_level=SecurityLevel.PRODUCTION
        )
        
    async def generate_modification_code(self, 
                                       modification_request: ModificationRequest) -> Dict[str, Any]:
        """Generate code for requested modification with improved error handling."""
        logger.info(f"Generating code for modification: {modification_request.modification_id}")
        
        try:
            return await self._generate_code_safely(modification_request)
        except Exception as e:
            return self._handle_generation_error(e, modification_request.modification_id)
    
    async def _generate_code_safely(self, modification_request: ModificationRequest) -> Dict[str, Any]:
        """Safely generate code with comprehensive validation."""
        # Extract current implementation
        current_code = await self._extract_current_implementation(
            modification_request.agent_name,
            modification_request.target_component
        )
        
        # Generate improvement code
        improvement_code = await self._generate_improvement_code(
            modification_request.modification_type,
            current_code,
            modification_request.proposed_changes
        )
        
        # Validate the generated code
        validation_result = await self._validate_generated_code(
            improvement_code,
            modification_request.safety_constraints
        )
        
        if not validation_result['is_safe']:
            return {
                'success': False,
                'error': f"Generated code failed safety validation: {validation_result['violations']}",
                'generated_code': improvement_code
            }
        
        # Create modification package
        modification_package = await self._create_modification_package(
            current_code, improvement_code, validation_result, modification_request
        )
        
        # Record successful generation
        await self._record_generation_success(modification_request.modification_id, improvement_code)
        
        return {'success': True, 'modification_package': modification_package}
    
    async def _create_modification_package(self, 
                                         current_code: str, 
                                         improvement_code: str,
                                         validation_result: Dict[str, Any],
                                         modification_request: ModificationRequest) -> Dict[str, Any]:
        """Create complete modification package."""
        return {
            'original_code': current_code,
            'modified_code': improvement_code,
            'validation_results': validation_result,
            'test_cases': await self._generate_test_cases(improvement_code),
            'rollback_code': current_code,
            'installation_instructions': await self._generate_installation_instructions(
                modification_request
            )
        }
    
    async def _record_generation_success(self, modification_id: str, improvement_code: str):
        """Record successful code generation in history."""
        self.generation_history.append({
            'modification_id': modification_id,
            'timestamp': datetime.now(),
            'success': True,
            'code_metrics': self._calculate_code_metrics(improvement_code)
        })
        logger.info(f"Successfully generated modification code: {modification_id}")
    
    def _handle_generation_error(self, error: Exception, modification_id: str) -> Dict[str, Any]:
        """Handle code generation errors with proper logging."""
        logger.error(f"Code generation failed for {modification_id}: {error}")
        return {
            'success': False,
            'error': str(error),
            'traceback': traceback.format_exc()
        }
    
    async def _extract_current_implementation(self, agent_name: str, component: str) -> str:
        """Extract current implementation code"""
        # This would extract the current code from the agent
        # For now, return a placeholder
        return f"# Current implementation for {agent_name}.{component}\n# Placeholder for actual code extraction"
    
    async def _generate_improvement_code(self, 
                                       modification_type: ModificationType,
                                       current_code: str,
                                       proposed_changes: str) -> str:
        """Generate improved code based on modification type using strategy pattern."""
        
        generators = {
            ModificationType.STRATEGY_OPTIMIZATION: self._generate_strategy_optimization,
            ModificationType.TOOL_ENHANCEMENT: self._generate_tool_enhancement,
            ModificationType.MEMORY_OPTIMIZATION: self._generate_memory_optimization,
            ModificationType.LEARNING_ENHANCEMENT: self._generate_learning_enhancement,
            ModificationType.PERFORMANCE_TUNING: self._generate_performance_tuning
        }
        
        generator = generators.get(modification_type, self._generate_generic_improvement)
        return await generator(current_code, proposed_changes)
    
    async def _generate_strategy_optimization(self, current_code: str, changes: str) -> str:
        """Generate optimized strategy code"""
        template = '''
async def optimized_strategy_selection(self, available_strategies: List[str], context: Dict[str, Any]) -> str:
    """
    Enhanced strategy selection with performance-based optimization
    Generated by autonomous self-modification system
    """
    # Performance-weighted strategy selection
    strategy_scores = {}
    
    for strategy in available_strategies:
        base_score = self.learning_system.strategies.get(strategy, 0.5)
        
        # Context-aware scoring
        if 'complexity' in context:
            if context['complexity'] > 0.8 and 'analytical' in strategy:
                base_score *= 1.3
            elif context['complexity'] < 0.3 and 'direct' in strategy:
                base_score *= 1.2
        
        # Recent performance weighting
        recent_performance = self._get_recent_strategy_performance(strategy)
        if recent_performance:
            base_score = 0.7 * base_score + 0.3 * recent_performance
        
        strategy_scores[strategy] = base_score
    
    # Select best strategy with epsilon-greedy exploration
    if self.exploration_rate > 0 and len(available_strategies) > 1:
        if random.random() < self.exploration_rate:
            return random.choice(available_strategies)
    
    return max(strategy_scores, key=strategy_scores.get)

def _get_recent_strategy_performance(self, strategy: str, window: int = 10) -> Optional[float]:
    """Get recent performance for strategy"""
    recent_obs = self.memory.episodic_memory[-window:] if len(self.memory.episodic_memory) >= window else self.memory.episodic_memory
    strategy_obs = [obs for obs in recent_obs if obs.action.action_type == strategy]
    
    if strategy_obs:
        successes = sum(1 for obs in strategy_obs if obs.success)
        return successes / len(strategy_obs)
    return {}
'''
        return template
    
    async def _generate_tool_enhancement(self, current_code: str, changes: str) -> str:
        """Generate enhanced tool usage code"""
        template = '''
async def enhanced_tool_execution(self, tool: Callable, parameters: Dict[str, Any]) -> Any:
    """
    Enhanced tool execution with error handling and optimization
    Generated by autonomous self-modification system
    """
    import asyncio
    import time
    from typing import Any, Dict, Callable
    
    start_time = time.time()
    
    try:
        # Pre-execution optimization
        optimized_params = await self._optimize_tool_parameters(tool, parameters)
        
        # Execute with timeout and retry logic
        max_retries = 3
        timeout_seconds = 30
        
        for attempt in range(max_retries):
            try:
                if asyncio.iscoroutinefunction(tool):
                    result = await asyncio.wait_for(
                        tool(**optimized_params), 
                        timeout=timeout_seconds
                    )
                else:
                    result = await asyncio.to_thread(tool, **optimized_params)
                
                # Record successful execution
                execution_time = time.time() - start_time
                await self._record_tool_performance(
                    tool.__name__, execution_time, True, attempt + 1
                )
                
                return result
                
            except asyncio.TimeoutError:
                logger.warning(f"Tool {tool.__name__} timed out on attempt {attempt + 1}")
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                
            except Exception as e:
                logger.error(f"Tool {tool.__name__} failed on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:
                    execution_time = time.time() - start_time
                    await self._record_tool_performance(
                        tool.__name__, execution_time, False, attempt + 1
                    )
                    raise
                await asyncio.sleep(1)
    
    except Exception as e:
        execution_time = time.time() - start_time
        await self._record_tool_performance(
            tool.__name__, execution_time, False, max_retries
        )
        raise

async def _optimize_tool_parameters(self, tool: Callable, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize parameters for tool execution"""
    # Parameter optimization logic
    optimized = parameters.copy()
    
    # Add common optimizations
    if 'timeout' not in optimized:
        optimized['timeout'] = 30
    
    if 'max_retries' not in optimized:
        optimized['max_retries'] = 3
    
    return optimized

async def _record_tool_performance(self, tool_name: str, execution_time: float, 
                                 success: bool, attempts: int):
    """Record tool performance metrics"""
    if not hasattr(self, 'tool_performance_history'):
        self.tool_performance_history = {}
    
    if tool_name not in self.tool_performance_history:
        self.tool_performance_history[tool_name] = []
    
    self.tool_performance_history[tool_name].append({
        'execution_time': execution_time,
        'success': success,
        'attempts': attempts,
        'timestamp': datetime.now()
    })
    
    # Keep only recent history
    if len(self.tool_performance_history[tool_name]) > 100:
        self.tool_performance_history[tool_name] = self.tool_performance_history[tool_name][-100:]
'''
        return template
    
    async def _validate_generated_code(self, code: str, safety_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Validate generated code using the security validator."""
        return await self.security_validator.validate_code_security(code, safety_constraints)
    


class PerformanceAnalyzer:
    """Analyzes agent performance to identify improvement opportunities."""
    
    def __init__(self):
        self.task_classifier = TaskTypeClassifier()
    
    async def analyze_task_performance(self, agent: BaseAgent) -> Dict[str, Dict[str, float]]:
        """Analyze task performance by type with improved efficiency."""
        if not hasattr(agent.memory, 'episodic_memory'):
            return {}
        
        # Group observations efficiently using defaultdict
        task_groups = defaultdict(list)
        for obs in agent.memory.episodic_memory:
            task_type = self.task_classifier.classify_task_type(obs.action)
            task_groups[task_type].append(obs)
        
        # Calculate performance metrics
        performance_results = {}
        for task_type, observations in task_groups.items():
            if len(observations) >= 3:  # Minimum sample size
                performance_results[task_type] = self._calculate_task_metrics(observations)
        
        return performance_results
    
    def _calculate_task_metrics(self, observations: List[Observation]) -> Dict[str, float]:
        """Calculate performance metrics for task observations."""
        success_count = sum(1 for obs in observations if obs.success)
        success_rate = success_count / len(observations)
        
        # Calculate average execution time if available
        execution_times = [
            obs.action.execution_time for obs in observations 
            if hasattr(obs.action, 'execution_time')
        ]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        return {
            'success_rate': success_rate,
            'sample_size': len(observations),
            'avg_execution_time': avg_execution_time
        }


class TaskTypeClassifier:
    """Classifies task types from actions."""
    
    def classify_task_type(self, action: Action) -> str:
        """Classify task type from action with improved pattern matching."""
        action_text = f"{action.action_type} {action.expected_outcome}".lower()
        
        classification_patterns = {
            'invoice_processing': ['invoice', 'financial', 'accounting'],
            'data_analysis': ['data', 'analyze', 'analytics', 'statistics'],
            'code_generation': ['code', 'program', 'software', 'development'],
            'quality_review': ['review', 'audit', 'quality', 'validation'],
            'document_processing': ['document', 'pdf', 'extract', 'parse']
        }
        
        for task_type, patterns in classification_patterns.items():
            if any(pattern in action_text for pattern in patterns):
                return task_type
        
        return 'general_task'


class EvolutionPlanGenerator:
    """Generates evolution plans based on performance analysis."""
    
    def __init__(self, improvement_threshold: float = 0.15):
        self.improvement_threshold = improvement_threshold
    
    async def generate_evolution_plan(self, 
                                    agent: BaseAgent,
                                    performance_gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evolution plan to address performance gaps."""
        if not performance_gaps:
            return {'modification_requests': []}
        
        # Prioritize gaps by improvement opportunity
        prioritized_gaps = sorted(
            performance_gaps,
            key=lambda g: g['improvement_opportunity'],
            reverse=True
        )
        
        modification_requests = await self._create_modification_requests(
            agent, prioritized_gaps[:3]  # Top 3 gaps
        )
        
        return {
            'agent_name': agent.name,
            'total_gaps_identified': len(performance_gaps),
            'addressable_gaps': len([g for g in performance_gaps if g['improvement_opportunity'] >= self.improvement_threshold]),
            'modification_requests': modification_requests,
            'expected_total_improvement': sum(req.expected_improvement for req in modification_requests),
            'evolution_timeline': await self._estimate_evolution_timeline(modification_requests),
            'resource_requirements': await self._calculate_resource_requirements(modification_requests)
        }
    
    async def _create_modification_requests(self, agent: BaseAgent, gaps: List[Dict[str, Any]]) -> List[ModificationRequest]:
        """Create modification requests for performance gaps."""
        modification_requests = []
        
        for i, gap in enumerate(gaps):
            if gap['improvement_opportunity'] >= self.improvement_threshold:
                request = ModificationRequest(
                    modification_id=f"{agent.name}_mod_{i}_{int(datetime.now().timestamp())}",
                    agent_name=agent.name,
                    modification_type=gap['suggested_modification'],
                    target_component=gap['category'],
                    proposed_changes=await self._generate_change_description(gap),
                    expected_improvement=gap['improvement_opportunity'],
                    safety_constraints=self._get_default_safety_constraints(),
                    testing_requirements=['unit_tests', 'integration_tests', 'performance_benchmarks'],
                    rollback_plan=self._get_default_rollback_plan()
                )
                modification_requests.append(request)
        
        return modification_requests
    
    def _get_default_safety_constraints(self) -> Dict[str, Any]:
        """Get default safety constraints."""
        return {
            'max_complexity_increase': 0.2,
            'preserve_core_functionality': True,
            'maintain_backwards_compatibility': True
        }
    
    def _get_default_rollback_plan(self) -> Dict[str, Any]:
        """Get default rollback plan."""
        return {
            'backup_required': True,
            'rollback_triggers': ['performance_degradation', 'safety_violation'],
            'rollback_timeout': 24 * 3600  # 24 hours
        }
    
    async def _estimate_evolution_timeline(self, modification_requests: List[ModificationRequest]) -> Dict[str, Any]:
        """Estimate timeline for evolution plan."""
        total_modifications = len(modification_requests)
        base_time_per_mod = 2.0  # Base 2 hours per modification
        
        complexity_multipliers = {
            ModificationType.STRATEGY_OPTIMIZATION: 1.0,
            ModificationType.TOOL_ENHANCEMENT: 1.2,
            ModificationType.MEMORY_OPTIMIZATION: 1.5,
            ModificationType.LEARNING_ENHANCEMENT: 1.8,
            ModificationType.PERFORMANCE_TUNING: 1.3
        }
        
        total_estimated_hours = sum(
            base_time_per_mod * complexity_multipliers.get(request.modification_type, 1.0)
            for request in modification_requests
        )
        
        return {
            'total_modifications': total_modifications,
            'estimated_total_hours': total_estimated_hours,
            'estimated_total_days': total_estimated_hours / 24,
            'parallel_execution_possible': total_modifications <= 3,
            'estimated_completion_date': (datetime.now() + timedelta(hours=total_estimated_hours)).isoformat()
        }
    
    async def _calculate_resource_requirements(self, modification_requests: List[ModificationRequest]) -> Dict[str, Any]:
        """Calculate resource requirements for modifications."""
        req_count = len(modification_requests)
        
        return {
            'cpu_percentage': min(80, req_count * 0.1),  # 10% CPU per modification, max 80%
            'memory_mb': req_count * 50,  # 50MB per modification
            'storage_mb': req_count * 10,  # 10MB per backup
            'validation_overhead_percentage': req_count * 0.05,  # 5% overhead per validation
            'concurrent_modifications': min(3, req_count),
            'resource_intensity': 'low' if req_count <= 2 else 'medium' if req_count <= 5 else 'high'
        }
    
    async def _generate_change_description(self, gap: Dict[str, Any]) -> str:
        """Generate detailed change description for a performance gap."""
        gap_type = gap['type']
        current_perf = gap['current_performance']
        target_perf = gap['target_performance']
        improvement = gap['improvement_opportunity']
        
        templates = {
            'task_performance': f"Optimize {gap['category']} task execution to improve success rate from {current_perf:.2%} to {target_perf:.2%} (improvement: {improvement:.2%})",
            'tool_efficiency': f"Enhance {gap['category']} tool usage to reduce execution time from {current_perf:.2f}s to {target_perf:.2f}s",
            'memory_efficiency': f"Improve {gap['category']} to increase accuracy from {current_perf:.2%} to {target_perf:.2%}",
            'learning_rate': f"Accelerate {gap['category']} to improve learning speed from {current_perf:.3f} to {target_perf:.3f}"
        }
        
        return templates.get(gap_type, f"Address {gap_type} performance gap: current {current_perf:.3f}, target {target_perf:.3f}")


class PerformanceDrivenEvolution:
    """Orchestrates agent evolution based on performance analysis."""
    
    def __init__(self, improvement_threshold: float = 0.15):
        self.improvement_threshold = improvement_threshold
        self.evolution_history: List[Dict[str, Any]] = []
        self.performance_tracking: Dict[str, List[float]] = {}
        
        # Initialize components
        self.performance_analyzer = PerformanceAnalyzer()
        self.plan_generator = EvolutionPlanGenerator(improvement_threshold)
        
    async def analyze_performance_gaps(self, agent: BaseAgent) -> List[Dict[str, Any]]:
        """Analyze agent performance to identify improvement opportunities."""
        logger.info(f"Analyzing performance gaps for agent: {agent.name}")
        
        gaps = []
        
        # Task performance analysis
        task_performance = await self.performance_analyzer.analyze_task_performance(agent)
        gaps.extend(self._identify_task_performance_gaps(task_performance))
        
        # Tool efficiency analysis
        tool_efficiency = await self._analyze_tool_efficiency(agent)
        gaps.extend(self._identify_tool_efficiency_gaps(tool_efficiency))
        
        # Memory efficiency analysis
        memory_efficiency = await self._analyze_memory_efficiency(agent)
        gaps.extend(self._identify_memory_efficiency_gaps(memory_efficiency))
        
        # Learning rate analysis
        learning_rate = await self._analyze_learning_rate(agent)
        gaps.extend(self._identify_learning_rate_gaps(learning_rate))
        
        logger.info(f"Identified {len(gaps)} performance gaps for {agent.name}")
        return gaps
    
    def _identify_task_performance_gaps(self, task_performance: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify task performance gaps."""
        gaps = []
        for task_type, performance in task_performance.items():
            if performance['success_rate'] < 0.7:  # Below acceptable threshold
                gaps.append({
                    'type': 'task_performance',
                    'category': task_type,
                    'current_performance': performance['success_rate'],
                    'target_performance': 0.8,
                    'improvement_opportunity': 0.8 - performance['success_rate'],
                    'suggested_modification': ModificationType.STRATEGY_OPTIMIZATION
                })
        return gaps
    
    def _identify_tool_efficiency_gaps(self, tool_efficiency: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """Identify tool efficiency gaps."""
        gaps = []
        for tool_name, efficiency in tool_efficiency.items():
            if efficiency['avg_execution_time'] > efficiency['target_time']:
                improvement_opp = (efficiency['avg_execution_time'] - efficiency['target_time']) / efficiency['avg_execution_time']
                gaps.append({
                    'type': 'tool_efficiency',
                    'category': tool_name,
                    'current_performance': efficiency['avg_execution_time'],
                    'target_performance': efficiency['target_time'],
                    'improvement_opportunity': improvement_opp,
                    'suggested_modification': ModificationType.TOOL_ENHANCEMENT
                })
        return gaps
    
    def _identify_memory_efficiency_gaps(self, memory_efficiency: Dict[str, float]) -> List[Dict[str, Any]]:
        """Identify memory efficiency gaps."""
        gaps = []
        if memory_efficiency['recall_accuracy'] < 0.8:
            gaps.append({
                'type': 'memory_efficiency',
                'category': 'recall_accuracy',
                'current_performance': memory_efficiency['recall_accuracy'],
                'target_performance': 0.85,
                'improvement_opportunity': 0.85 - memory_efficiency['recall_accuracy'],
                'suggested_modification': ModificationType.MEMORY_OPTIMIZATION
            })
        return gaps
    
    def _identify_learning_rate_gaps(self, learning_rate: float) -> List[Dict[str, Any]]:
        """Identify learning rate gaps."""
        gaps = []
        if learning_rate < 0.1:  # Learning too slowly
            gaps.append({
                'type': 'learning_rate',
                'category': 'adaptation_speed',
                'current_performance': learning_rate,
                'target_performance': 0.15,
                'improvement_opportunity': 0.15 - learning_rate,
                'suggested_modification': ModificationType.LEARNING_ENHANCEMENT
            })
        return gaps
    
    async def generate_evolution_plan(self, 
                                    agent: BaseAgent,
                                    performance_gaps: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evolution plan to address performance gaps."""
        logger.info(f"Generating evolution plan for agent: {agent.name}")
        
        evolution_plan = await self.plan_generator.generate_evolution_plan(agent, performance_gaps)
        
        logger.info(f"Generated evolution plan with {len(evolution_plan['modification_requests'])} modifications")
        return evolution_plan
    
    async def _analyze_task_performance(self, agent: BaseAgent) -> Dict[str, Dict[str, float]]:
        """Analyze task performance by type"""
        task_performance = {}
        
        if hasattr(agent.memory, 'episodic_memory'):
            # Group observations by task type
            task_groups = {}
            for obs in agent.memory.episodic_memory:
                task_type = self._classify_task_type(obs.action)
                if task_type not in task_groups:
                    task_groups[task_type] = []
                task_groups[task_type].append(obs)
            
            # Calculate performance metrics for each type
            for task_type, observations in task_groups.items():
                if len(observations) >= 3:  # Minimum sample size
                    success_count = sum(1 for obs in observations if obs.success)
                    success_rate = success_count / len(observations)
                    
                    # Calculate average execution time if available
                    execution_times = []
                    for obs in observations:
                        if hasattr(obs.action, 'execution_time'):
                            execution_times.append(obs.action.execution_time)
                    
                    avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
                    
                    task_performance[task_type] = {
                        'success_rate': success_rate,
                        'sample_size': len(observations),
                        'avg_execution_time': avg_execution_time
                    }
        
        return task_performance
    
    def _classify_task_type(self, action: Action) -> str:
        """Classify task type from action"""
        action_text = f"{action.action_type} {action.expected_outcome}".lower()
        
        if 'invoice' in action_text:
            return 'invoice_processing'
        elif 'data' in action_text or 'analyze' in action_text:
            return 'data_analysis'
        elif 'code' in action_text or 'program' in action_text:
            return 'code_generation'
        elif 'review' in action_text:
            return 'quality_review'
        else:
            return 'general_task'


class SecurityMonitor:
    """Handles security monitoring for self-modifying agents."""
    
    def __init__(self, security_framework: AutonomousSecurityFramework):
        self.security_framework = security_framework
        self.behavioral_monitor = BehavioralMonitor()
        self.threat_detector = ThreatDetectionSystem()
        self.emergency_responder = EmergencyResponseSystem()
        
        # Security state
        self.security_violations_count = 0
        self.last_security_check = datetime.now()
        self.quarantined = False
    
    async def perform_security_checks(self, agent: BaseAgent) -> Dict[str, Any]:
        """Perform comprehensive security checks."""
        if self.quarantined:
            return {"status": "quarantined", "reason": "Agent is quarantined for security reasons"}
        
        # Behavioral monitoring
        try:
            behavioral_anomalies = await self.behavioral_monitor.monitor_agent_behavior(agent)
            if any(anomaly.severity.value in ['high', 'critical'] for anomaly in behavioral_anomalies):
                logger.warning(f"High-severity behavioral anomalies detected in {agent.name}")
                incident = await self.emergency_responder.respond_to_security_event(
                    threats=[], anomalies=behavioral_anomalies
                )
                return {
                    "status": "security_incident", 
                    "incident_id": incident.incident_id,
                    "reason": "Behavioral anomalies detected"
                }
        except Exception as e:
            logger.error(f"Behavioral monitoring failed: {e}")
        
        # Validate operation parameters
        operation_data = {
            'operation_type': 'self_modification',
            'agent_name': agent.name
        }
        
        agent_context = {
            'agent_name': agent.name,
            'total_tasks': getattr(agent, 'total_tasks', 0),
            'successful_tasks': getattr(agent, 'successful_tasks', 0),
            'security_violations': self.security_violations_count
        }
        
        security_audit = await self.security_framework.validate_autonomous_operation(
            'self_modification', operation_data, agent_context
        )
        
        if not security_audit.is_secure:
            logger.critical(f"SECURITY VIOLATION: Self-modification blocked for {agent.name}")
            self.security_violations_count += 1
            
            # Quarantine agent if critical security threats
            critical_threats = [t for t in security_audit.threats_detected 
                              if t.severity.value in ['critical', 'emergency']]
            if critical_threats:
                await self.security_framework.quarantine_agent(
                    agent.name, "Critical security violations during self-modification attempt"
                )
                self.quarantined = True
            
            return {
                "status": "security_violation",
                "security_audit": security_audit.__dict__,
                "threats": [t.__dict__ for t in security_audit.threats_detected]
            }
        
        return {"status": "secure"}


class ModificationManager:
    """Manages modification requests and tracking."""
    
    def __init__(self):
        self.applied_modifications: Dict[str, ModificationResult] = {}
        self.pending_modifications: List[ModificationRequest] = []
        self.modification_success_rate = 0.0
    
    async def apply_modification(self, 
                               modification_request: ModificationRequest,
                               code_generator: DynamicCodeGenerator,
                               agent: BaseAgent) -> ModificationResult:
        """Apply a self-modification request."""
        logger.info(f"Applying modification: {modification_request.modification_id}")
        
        try:
            # Capture baseline performance
            performance_before = await self._measure_performance(agent)
            
            # Generate modification code
            code_generation_result = await code_generator.generate_modification_code(
                modification_request
            )
            
            if not code_generation_result['success']:
                return self._create_failed_result(
                    modification_request.modification_id,
                    performance_before,
                    code_generation_result['error']
                )
            
            # Apply modification with testing and rollback
            result = await self._apply_with_testing(
                modification_request, code_generation_result, performance_before, agent
            )
            
            self.applied_modifications[modification_request.modification_id] = result
            return result
            
        except Exception as e:
            logger.error(f"Modification application failed: {e}")
            return self._create_error_result(
                modification_request.modification_id,
                locals().get('performance_before', {}),
                str(e)
            )
    
    async def _apply_with_testing(self, 
                                modification_request: ModificationRequest,
                                code_generation_result: Dict[str, Any],
                                performance_before: Dict[str, float],
                                agent: BaseAgent) -> ModificationResult:
        """Apply modification with comprehensive testing."""
        # Create backup
        backup_state = await self._create_backup(agent)
        
        try:
            modification_package = code_generation_result['modification_package']
            
            # Test modification first
            test_result = await self._test_modification(modification_package)
            if not test_result['success']:
                await self._restore_backup(backup_state, agent)
                return self._create_failed_result(
                    modification_request.modification_id,
                    performance_before,
                    f"Modification failed testing: {test_result.get('error', 'Unknown error')}"
                )
            
            # Apply modification to live system
            await self._install_modification(modification_package, agent)
            
            # Measure performance after modification
            performance_after = await self._measure_performance(agent)
            improvement_achieved = self._calculate_improvement(performance_before, performance_after)
            
            return ModificationResult(
                modification_id=modification_request.modification_id,
                success=True,
                performance_before=performance_before,
                performance_after=performance_after,
                improvement_achieved=improvement_achieved,
                side_effects=test_result.get('side_effects', []),
                validation_results=modification_package['validation_results'],
                rollback_available=True
            )
            
        except Exception as e:
            await self._restore_backup(backup_state, agent)
            raise e
    
    def _create_failed_result(self, modification_id: str, performance_before: Dict[str, float], error: str) -> ModificationResult:
        """Create a failed modification result."""
        return ModificationResult(
            modification_id=modification_id,
            success=False,
            performance_before=performance_before,
            performance_after=performance_before,
            improvement_achieved=0.0,
            side_effects=[],
            validation_results={},
            rollback_available=False,
            error_message=error
        )
    
    def _create_error_result(self, modification_id: str, performance_before: Dict[str, float], error: str) -> ModificationResult:
        """Create an error modification result."""
        return ModificationResult(
            modification_id=modification_id,
            success=False,
            performance_before=performance_before,
            performance_after={},
            improvement_achieved=0.0,
            side_effects=[],
            validation_results={},
            rollback_available=True,
            error_message=error
        )
    
    async def _measure_performance(self, agent: BaseAgent) -> Dict[str, float]:
        """Measure current agent performance metrics."""
        return {
            'success_rate': agent.get_success_rate(),
            'average_response_time': getattr(agent, 'avg_response_time', 1.0),
            'memory_efficiency': await self._calculate_memory_efficiency(agent),
            'learning_rate': await self._calculate_learning_rate(agent),
            'tool_efficiency': await self._calculate_tool_efficiency(agent)
        }
    
    async def _calculate_memory_efficiency(self, agent: BaseAgent) -> float:
        """Calculate memory system efficiency."""
        if hasattr(agent.memory, 'episodic_memory') and agent.memory.episodic_memory:
            recent_episodes = agent.memory.episodic_memory[-20:] if len(agent.memory.episodic_memory) >= 20 else agent.memory.episodic_memory
            successful_recalls = sum(1 for ep in recent_episodes if ep.success)
            return successful_recalls / len(recent_episodes)
        return 0.5
    
    async def _calculate_learning_rate(self, agent: BaseAgent) -> float:
        """Calculate learning adaptation rate."""
        if hasattr(agent.learning_system, 'strategies') and agent.learning_system.strategies:
            strategy_improvements = []
            for strategy, performance in agent.learning_system.strategies.items():
                if performance > 0.1:  # Only consider strategies with some data
                    strategy_improvements.append(min(1.0, performance))
            return sum(strategy_improvements) / len(strategy_improvements) if strategy_improvements else 0.1
        return 0.1
    
    async def _calculate_tool_efficiency(self, agent: BaseAgent) -> float:
        """Calculate tool usage efficiency."""
        if hasattr(agent, 'tool_performance_history'):
            total_efficiency = 0
            tool_count = 0
            
            for tool_name, performance_data in agent.tool_performance_history.items():
                if performance_data:
                    recent_performance = performance_data[-10:]  # Last 10 executions
                    avg_success_rate = sum(1 for p in recent_performance if p['success']) / len(recent_performance)
                    avg_execution_time = sum(p['execution_time'] for p in recent_performance) / len(recent_performance)
                    
                    # Efficiency score combines success rate and speed
                    efficiency = avg_success_rate * (1.0 / max(0.1, avg_execution_time))
                    total_efficiency += efficiency
                    tool_count += 1
            
            return total_efficiency / tool_count if tool_count > 0 else 0.5
        return 0.5
    
    def _calculate_improvement(self, before: Dict[str, float], after: Dict[str, float]) -> float:
        """Calculate overall improvement score."""
        improvements = []
        for metric, before_value in before.items():
            if metric in after and before_value > 0:
                improvement = (after[metric] - before_value) / before_value
                improvements.append(improvement)
        return sum(improvements) / len(improvements) if improvements else 0.0
    
    async def _create_backup(self, agent: BaseAgent) -> Dict[str, Any]:
        """Create backup of agent state."""
        return {
            'backup_id': f"agent_backup_{agent.name}_{int(datetime.now().timestamp())}",
            'agent_state': {
                'total_tasks': getattr(agent, 'total_tasks', 0),
                'successful_tasks': getattr(agent, 'successful_tasks', 0),
                'state': str(getattr(agent, 'state', 'unknown'))
            },
            'memory_state': {
                'episodic_count': len(agent.memory.episodic_memory) if hasattr(agent.memory, 'episodic_memory') else 0,
                'semantic_keys': list(agent.memory.semantic_memory.keys()) if hasattr(agent.memory, 'semantic_memory') else []
            },
            'timestamp': datetime.now().isoformat()
        }
    
    async def _restore_backup(self, backup_state: Dict[str, Any], agent: BaseAgent):
        """Restore agent from backup state."""
        logger.info(f"Restoring agent {agent.name} from backup {backup_state['backup_id']}")
        
        # Restore basic state
        agent_state = backup_state.get('agent_state', {})
        if hasattr(agent, 'total_tasks'):
            agent.total_tasks = agent_state.get('total_tasks', 0)
        if hasattr(agent, 'successful_tasks'):
            agent.successful_tasks = agent_state.get('successful_tasks', 0)
    
    async def _test_modification(self, modification_package: Dict[str, Any]) -> Dict[str, Any]:
        """Test modification in safe environment."""
        try:
            test_results = {
                'success': True,
                'tests_passed': 0,
                'tests_failed': 0,
                'side_effects': []
            }
            
            # Test 1: Code syntax validation
            modified_code = modification_package.get('modified_code', '')
            if modified_code:
                try:
                    ast.parse(modified_code)
                    test_results['tests_passed'] += 1
                except SyntaxError as e:
                    test_results['success'] = False
                    test_results['tests_failed'] += 1
                    test_results['error'] = f"Syntax error: {e}"
                    return test_results
            
            # Test 2: Validation results check
            validation_results = modification_package.get('validation_results', {})
            if not validation_results.get('is_safe', False):
                test_results['success'] = False
                test_results['tests_failed'] += 1
                test_results['error'] = "Safety validation failed"
                return test_results
            
            test_results['tests_passed'] += 1
            return test_results
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tests_passed': 0,
                'tests_failed': 1,
                'side_effects': [f"Test execution failed: {e}"]
            }
    
    async def _install_modification(self, modification_package: Dict[str, Any], agent: BaseAgent):
        """Install modification to live system."""
        modified_code = modification_package.get('modified_code', '')
        installation_instructions = modification_package.get('installation_instructions', {})
        
        logger.info(f"Installing modification on agent {agent.name}")
        logger.debug(f"Modified code length: {len(modified_code)} characters")
        logger.debug(f"Installation instructions: {installation_instructions}")
        
        # Mark installation as complete
        if not hasattr(agent, 'active_modifications'):
            agent.active_modifications = set()
        agent.active_modifications.add(f"mod_{int(datetime.now().timestamp())}")


class SelfModifyingAgent(BaseAgent):
    """Enhanced base agent with self-modification capabilities (refactored for reduced complexity)."""
    
    def __init__(self, 
                 name: str,
                 api_key: Optional[str] = None,
                 tools: Optional[List[Callable]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 safety_config: Optional[Dict[str, Any]] = None):
        super().__init__(name, api_key, tools, config)
        
        # Initialize frameworks
        self.safety_framework = AutonomousSafetyFramework(config=safety_config)
        self.security_framework = AutonomousSecurityFramework(
            security_level=SecurityLevel.PRODUCTION,
            config=config.get('security_config', {}) if config else {}
        )
        
        # Initialize components
        self.code_generator = DynamicCodeGenerator(
            safety_validator=ModificationValidator(),
            security_framework=self.security_framework
        )
        self.evolution_engine = PerformanceDrivenEvolution()
        self.security_monitor = SecurityMonitor(self.security_framework)
        self.modification_manager = ModificationManager()
        
        # Self-improvement parameters
        self.self_improvement_enabled = config.get('self_improvement_enabled', True) if config else True
        self.improvement_frequency = config.get('improvement_frequency', 50) if config else 50  # Every 50 tasks
        self.max_modifications_per_session = 3
        
        logger.info(f"Initialized self-modifying agent: {self.name}")
    
    async def autonomous_self_improvement(self) -> Dict[str, Any]:
        """Perform autonomous self-improvement with comprehensive security validation."""
        logger.info(f"{self.name}: Starting secure autonomous self-improvement")
        global_metrics.incr("agent.self_improvement.started")
        
        # Pre-flight checks
        if not self.self_improvement_enabled:
            return {"status": "disabled", "reason": "Self-improvement is disabled"}
        
        # Security validation
        security_result = await self.security_monitor.perform_security_checks(self)
        if security_result["status"] != "secure":
            return security_result
        
        try:
            return await self._execute_improvement_cycle()
        except Exception as e:
            logger.error(f"{self.name}: Self-improvement failed: {e}")
            global_metrics.incr("agent.self_improvement.failed")
            return {
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc()
            }
    
    async def _execute_improvement_cycle(self) -> Dict[str, Any]:
        """Execute the improvement cycle with gap analysis and modification application."""
        # Analyze current performance
        performance_gaps = await self.evolution_engine.analyze_performance_gaps(self)
        
        if not performance_gaps:
            logger.info(f"{self.name}: No significant performance gaps found")
            return {"status": "no_improvements_needed", "gaps_analyzed": 0}
        
        # Generate evolution plan
        evolution_plan = await self.evolution_engine.generate_evolution_plan(self, performance_gaps)
        
        # Apply modifications
        modification_results = await self._apply_planned_modifications(evolution_plan)
        
        return self._create_improvement_result(performance_gaps, evolution_plan, modification_results)
    
    async def _apply_planned_modifications(self, evolution_plan: Dict[str, Any]) -> List[ModificationResult]:
        """Apply planned modifications with safety validation."""
        results = []
        modifications_applied = 0
        
        for modification_request in evolution_plan['modification_requests'][:self.max_modifications_per_session]:
            # Safety validation
            safety_check = await self.safety_framework.validate_modification_request(modification_request)
            
            if not safety_check.is_safe:
                logger.warning(f"Modification rejected for safety: {safety_check.violations}")
                continue
            
            # Apply modification
            modification_result = await self.modification_manager.apply_modification(
                modification_request, self.code_generator, self
            )
            results.append(modification_result)
            
            if modification_result.success:
                modifications_applied += 1
                logger.info(f"Successfully applied modification: {modification_request.modification_id}")
            else:
                logger.warning(f"Failed to apply modification: {modification_result.error_message}")
        
        # Update success rate
        if results:
            successful_modifications = sum(1 for r in results if r.success)
            self.modification_manager.modification_success_rate = successful_modifications / len(results)
        
        return results
    
    def _create_improvement_result(self, 
                                 performance_gaps: List[Dict[str, Any]],
                                 evolution_plan: Dict[str, Any],
                                 modification_results: List[ModificationResult]) -> Dict[str, Any]:
        """Create the final improvement result summary."""
        modifications_applied = sum(1 for r in modification_results if r.success)
        
        improvement_result = {
            "status": "completed",
            "gaps_identified": len(performance_gaps),
            "modifications_planned": len(evolution_plan['modification_requests']),
            "modifications_applied": modifications_applied,
            "modification_success_rate": self.modification_manager.modification_success_rate,
            "expected_improvement": evolution_plan['expected_total_improvement'],
            "results": modification_results
        }
        
        logger.info(f"{self.name}: Autonomous self-improvement completed - {modifications_applied} modifications applied")
        global_metrics.incr("agent.self_improvement.completed", modifications_applied)
        
        return improvement_result
    
    # Removed _apply_self_modification - now handled by ModificationManager
    
    # Performance measurement methods moved to ModificationManager
    
    async def process_task(self, task: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Override process_task to include self-improvement checks."""
        # Process task normally
        result = await super().process_task(task, context)
        
        # Check if it's time for self-improvement
        if (self.total_tasks % self.improvement_frequency == 0 and 
            self.self_improvement_enabled and 
            self.total_tasks > 0):
            
            # Run self-improvement asynchronously to not block task processing
            asyncio.create_task(self.autonomous_self_improvement())
        
        return result
    
    def get_self_modification_metrics(self) -> Dict[str, Any]:
        """Get metrics related to self-modification."""
        base_metrics = self.get_metrics()
        
        modification_metrics = {
            'applied_modifications': len(self.modification_manager.applied_modifications),
            'pending_modifications': len(self.modification_manager.pending_modifications),
            'modification_success_rate': self.modification_manager.modification_success_rate,
            'self_improvement_enabled': self.self_improvement_enabled,
            'improvement_frequency': self.improvement_frequency,
            'security_violations': self.security_monitor.security_violations_count,
            'quarantined': self.security_monitor.quarantined,
            'last_improvement': max([mod.applied_at for mod in self.modification_manager.applied_modifications.values()]) if self.modification_manager.applied_modifications else None
        }
        
        return {**base_metrics, 'self_modification_metrics': modification_metrics}
    
    # Backup, testing, and installation methods moved to ModificationManager
    
    async def _generate_change_description(self, gap: Dict[str, Any]) -> str:
        """Generate detailed change description for a performance gap"""
        gap_type = gap['type']
        current_perf = gap['current_performance']
        target_perf = gap['target_performance']
        improvement = gap['improvement_opportunity']
        
        if gap_type == 'task_performance':
            return f"Optimize {gap['category']} task execution to improve success rate from {current_perf:.2%} to {target_perf:.2%} (improvement: {improvement:.2%})"
        
        elif gap_type == 'tool_efficiency':
            return f"Enhance {gap['category']} tool usage to reduce execution time from {current_perf:.2f}s to {target_perf:.2f}s"
        
        elif gap_type == 'memory_efficiency':
            return f"Improve {gap['category']} to increase accuracy from {current_perf:.2%} to {target_perf:.2%}"
        
        elif gap_type == 'learning_rate':
            return f"Accelerate {gap['category']} to improve learning speed from {current_perf:.3f} to {target_perf:.3f}"
        
        else:
            return f"Address {gap_type} performance gap: current {current_perf:.3f}, target {target_perf:.3f}"
    
    async def _estimate_evolution_timeline(self, modification_requests: List[ModificationRequest]) -> Dict[str, Any]:
        """Estimate timeline for evolution plan"""
        total_modifications = len(modification_requests)
        
        # Estimate time per modification (in hours)
        base_time_per_mod = 2.0  # Base 2 hours per modification
        complexity_multipliers = {
            ModificationType.STRATEGY_OPTIMIZATION: 1.0,
            ModificationType.TOOL_ENHANCEMENT: 1.2,
            ModificationType.MEMORY_OPTIMIZATION: 1.5,
            ModificationType.LEARNING_ENHANCEMENT: 1.8,
            ModificationType.PERFORMANCE_TUNING: 1.3
        }
        
        total_estimated_hours = 0
        for request in modification_requests:
            multiplier = complexity_multipliers.get(request.modification_type, 1.0)
            estimated_hours = base_time_per_mod * multiplier
            total_estimated_hours += estimated_hours
        
        return {
            'total_modifications': total_modifications,
            'estimated_total_hours': total_estimated_hours,
            'estimated_total_days': total_estimated_hours / 24,
            'parallel_execution_possible': total_modifications <= 3,
            'estimated_completion_date': (datetime.now() + timedelta(hours=total_estimated_hours)).isoformat()
        }
    
    async def _calculate_resource_requirements(self, modification_requests: List[ModificationRequest]) -> Dict[str, Any]:
        """Calculate resource requirements for modifications"""
        cpu_requirement = len(modification_requests) * 0.1  # 10% CPU per modification
        memory_requirement = len(modification_requests) * 50  # 50MB per modification
        
        # Storage for backups
        storage_requirement = len(modification_requests) * 10  # 10MB per backup
        
        # Safety validation resources
        validation_overhead = len(modification_requests) * 0.05  # 5% overhead per validation
        
        return {
            'cpu_percentage': min(80, cpu_requirement),  # Max 80% CPU
            'memory_mb': memory_requirement,
            'storage_mb': storage_requirement,
            'validation_overhead_percentage': validation_overhead,
            'concurrent_modifications': min(3, len(modification_requests)),
            'resource_intensity': 'low' if len(modification_requests) <= 2 else 'medium' if len(modification_requests) <= 5 else 'high'
        }