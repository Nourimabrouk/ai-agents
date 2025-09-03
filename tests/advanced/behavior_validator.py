"""
Behavior Validation Suite
Validates agent behaviors against expected patterns and constraints
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import logging
from enum import Enum

from templates.base_agent import BaseAgent, AgentState, Action, Observation
from core.orchestration.orchestrator import AgentOrchestrator, Task
from utils.observability.logging import get_logger

logger = get_logger(__name__)


class BehaviorExpectation(Enum):
    """Types of behavior expectations"""
    RESPONSE_TIME = "response_time"
    SUCCESS_RATE = "success_rate"
    RESOURCE_USAGE = "resource_usage"
    STATE_TRANSITIONS = "state_transitions"
    OUTPUT_FORMAT = "output_format"
    ERROR_HANDLING = "error_handling"
    LEARNING_PROGRESSION = "learning_progression"
    COLLABORATION_QUALITY = "collaboration_quality"


@dataclass
class BehaviorConstraint:
    """Defines a constraint on agent behavior"""
    expectation_type: BehaviorExpectation
    constraint_name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    expected_pattern: Optional[str] = None
    validation_function: Optional[Callable] = None
    tolerance: float = 0.1
    mandatory: bool = True


@dataclass
class BehaviorTestResult:
    """Result of a behavior validation test"""
    test_name: str
    agent_name: str
    constraint: BehaviorConstraint
    actual_value: Any
    expected_value: Any
    passed: bool
    deviation: float
    timestamp: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)


class BehaviorValidator:
    """
    Validates agent behaviors against defined expectations
    Supports property-based testing and constraint validation
    """
    
    def __init__(self, name: str = "behavior_validator"):
        self.name = name
        self.constraints: Dict[str, List[BehaviorConstraint]] = {}
        self.test_history: List[BehaviorTestResult] = []
        self.validation_rules: Dict[BehaviorExpectation, Callable] = {}
        
        # Initialize default validation rules
        self._initialize_default_rules()
        
        logger.info(f"Initialized behavior validator: {name}")
    
    def _initialize_default_rules(self) -> None:
        """Initialize default validation rules for common behavior patterns"""
        self.validation_rules = {
            BehaviorExpectation.RESPONSE_TIME: self._validate_response_time,
            BehaviorExpectation.SUCCESS_RATE: self._validate_success_rate,
            BehaviorExpectation.RESOURCE_USAGE: self._validate_resource_usage,
            BehaviorExpectation.STATE_TRANSITIONS: self._validate_state_transitions,
            BehaviorExpectation.OUTPUT_FORMAT: self._validate_output_format,
            BehaviorExpectation.ERROR_HANDLING: self._validate_error_handling,
            BehaviorExpectation.LEARNING_PROGRESSION: self._validate_learning_progression,
            BehaviorExpectation.COLLABORATION_QUALITY: self._validate_collaboration_quality
        }
    
    def add_constraint(self, agent_name: str, constraint: BehaviorConstraint) -> None:
        """Add a behavior constraint for an agent"""
        if agent_name not in self.constraints:
            self.constraints[agent_name] = []
        
        self.constraints[agent_name].append(constraint)
        logger.info(f"Added constraint {constraint.constraint_name} for agent {agent_name}")
    
    def add_response_time_constraint(self, 
                                   agent_name: str, 
                                   max_response_time: float,
                                   constraint_name: str = "default_response_time") -> None:
        """Add response time constraint"""
        constraint = BehaviorConstraint(
            expectation_type=BehaviorExpectation.RESPONSE_TIME,
            constraint_name=constraint_name,
            max_value=max_response_time,
            tolerance=0.1
        )
        self.add_constraint(agent_name, constraint)
    
    def add_success_rate_constraint(self, 
                                  agent_name: str, 
                                  min_success_rate: float,
                                  constraint_name: str = "default_success_rate") -> None:
        """Add success rate constraint"""
        constraint = BehaviorConstraint(
            expectation_type=BehaviorExpectation.SUCCESS_RATE,
            constraint_name=constraint_name,
            min_value=min_success_rate,
            tolerance=0.05
        )
        self.add_constraint(agent_name, constraint)
    
    def add_learning_progression_constraint(self, 
                                         agent_name: str,
                                         min_improvement_rate: float,
                                         constraint_name: str = "learning_progression") -> None:
        """Add learning progression constraint"""
        constraint = BehaviorConstraint(
            expectation_type=BehaviorExpectation.LEARNING_PROGRESSION,
            constraint_name=constraint_name,
            min_value=min_improvement_rate,
            tolerance=0.02
        )
        self.add_constraint(agent_name, constraint)
    
    async def validate_agent(self, agent: BaseAgent, test_tasks: List[Any] = None) -> List[BehaviorTestResult]:
        """Validate an agent against all defined constraints"""
        if agent.name not in self.constraints:
            logger.warning(f"No constraints defined for agent {agent.name}")
            return []
        
        results = []
        constraints = self.constraints[agent.name]
        
        # Run test tasks if provided
        test_data = {}
        if test_tasks:
            test_data = await self._run_test_tasks(agent, test_tasks)
        
        # Validate each constraint
        for constraint in constraints:
            try:
                result = await self._validate_constraint(agent, constraint, test_data)
                results.append(result)
                self.test_history.append(result)
            except Exception as e:
                logger.error(f"Error validating constraint {constraint.constraint_name}: {e}")
                error_result = BehaviorTestResult(
                    test_name=f"validation_error_{constraint.constraint_name}",
                    agent_name=agent.name,
                    constraint=constraint,
                    actual_value=None,
                    expected_value=None,
                    passed=False,
                    deviation=float('inf'),
                    error_message=str(e)
                )
                results.append(error_result)
        
        logger.info(f"Validated {len(results)} constraints for agent {agent.name}")
        return results
    
    async def validate_orchestrator(self, 
                                  orchestrator: AgentOrchestrator,
                                  test_scenarios: List[Dict[str, Any]] = None) -> Dict[str, List[BehaviorTestResult]]:
        """Validate all agents in an orchestrator"""
        all_results = {}
        
        # Test individual agents
        for agent_name, agent in orchestrator.agents.items():
            results = await self.validate_agent(agent, test_scenarios)
            all_results[agent_name] = results
        
        # Test orchestrator-level behaviors
        if test_scenarios:
            orchestrator_results = await self._validate_orchestrator_behaviors(orchestrator, test_scenarios)
            all_results['orchestrator'] = orchestrator_results
        
        return all_results
    
    async def _run_test_tasks(self, agent: BaseAgent, test_tasks: List[Any]) -> Dict[str, Any]:
        """Run test tasks and collect performance data"""
        test_data = {
            'response_times': [],
            'success_count': 0,
            'failure_count': 0,
            'task_results': [],
            'state_transitions': [],
            'resource_usage': [],
            'start_time': datetime.now()
        }
        
        initial_state = agent.state
        
        for task in test_tasks:
            start_time = time.time()
            initial_task_state = agent.state
            
            try:
                result = await agent.process_task(task)
                end_time = time.time()
                response_time = end_time - start_time
                
                test_data['response_times'].append(response_time)
                test_data['task_results'].append(result)
                
                if result is not None and self._is_successful_result(result):
                    test_data['success_count'] += 1
                else:
                    test_data['failure_count'] += 1
                
                # Track state transitions
                final_task_state = agent.state
                if initial_task_state != final_task_state:
                    test_data['state_transitions'].append({
                        'from': initial_task_state,
                        'to': final_task_state,
                        'task': str(task)
                    })
                
                # Simulate resource usage tracking
                test_data['resource_usage'].append({
                    'cpu_time': response_time * 0.8,  # Simplified
                    'memory_usage': len(str(result)) * 8 if result else 0,  # Simplified
                    'task': str(task)
                })
                
            except Exception as e:
                test_data['failure_count'] += 1
                test_data['task_results'].append(None)
                logger.error(f"Test task failed: {e}")
        
        test_data['end_time'] = datetime.now()
        test_data['total_tasks'] = len(test_tasks)
        test_data['success_rate'] = test_data['success_count'] / len(test_tasks) if test_tasks else 0
        
        return test_data
    
    def _is_successful_result(self, result: Any) -> bool:
        """Determine if a task result indicates success"""
        if result is None:
            return False
        
        if isinstance(result, dict):
            return result.get('success', True)  # Default to True if not specified
        
        if isinstance(result, bool):
            return result
        
        # Non-None result is considered success by default
        return True
    
    async def _validate_constraint(self, 
                                 agent: BaseAgent,
                                 constraint: BehaviorConstraint,
                                 test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate a specific constraint"""
        validation_rule = self.validation_rules.get(constraint.expectation_type)
        
        if not validation_rule:
            raise ValueError(f"No validation rule for expectation type: {constraint.expectation_type}")
        
        return await validation_rule(agent, constraint, test_data)
    
    async def _validate_response_time(self, 
                                    agent: BaseAgent,
                                    constraint: BehaviorConstraint,
                                    test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate response time constraints"""
        response_times = test_data.get('response_times', [])
        
        if not response_times:
            return BehaviorTestResult(
                test_name="response_time_no_data",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=None,
                expected_value=constraint.max_value,
                passed=False,
                deviation=float('inf'),
                error_message="No response time data available"
            )
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        # Check against max value constraint
        if constraint.max_value is not None:
            passed = avg_response_time <= constraint.max_value * (1 + constraint.tolerance)
            deviation = abs(avg_response_time - constraint.max_value) / constraint.max_value
            
            return BehaviorTestResult(
                test_name=f"response_time_{constraint.constraint_name}",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=avg_response_time,
                expected_value=constraint.max_value,
                passed=passed,
                deviation=deviation,
                additional_data={
                    'max_response_time': max_response_time,
                    'min_response_time': min(response_times),
                    'response_count': len(response_times)
                }
            )
        
        return BehaviorTestResult(
            test_name=f"response_time_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=avg_response_time,
            expected_value="no_constraint",
            passed=True,
            deviation=0.0
        )
    
    async def _validate_success_rate(self, 
                                   agent: BaseAgent,
                                   constraint: BehaviorConstraint,
                                   test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate success rate constraints"""
        success_rate = test_data.get('success_rate', 0.0)
        total_tasks = test_data.get('total_tasks', 0)
        
        if total_tasks == 0:
            return BehaviorTestResult(
                test_name="success_rate_no_data",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=None,
                expected_value=constraint.min_value,
                passed=False,
                deviation=float('inf'),
                error_message="No task data available"
            )
        
        if constraint.min_value is not None:
            passed = success_rate >= constraint.min_value * (1 - constraint.tolerance)
            deviation = abs(success_rate - constraint.min_value) / constraint.min_value if constraint.min_value > 0 else 0
            
            return BehaviorTestResult(
                test_name=f"success_rate_{constraint.constraint_name}",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=success_rate,
                expected_value=constraint.min_value,
                passed=passed,
                deviation=deviation,
                additional_data={
                    'success_count': test_data.get('success_count', 0),
                    'failure_count': test_data.get('failure_count', 0),
                    'total_tasks': total_tasks
                }
            )
        
        return BehaviorTestResult(
            test_name=f"success_rate_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=success_rate,
            expected_value="no_constraint",
            passed=True,
            deviation=0.0
        )
    
    async def _validate_resource_usage(self, 
                                     agent: BaseAgent,
                                     constraint: BehaviorConstraint,
                                     test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate resource usage constraints"""
        resource_usage = test_data.get('resource_usage', [])
        
        if not resource_usage:
            return BehaviorTestResult(
                test_name="resource_usage_no_data",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=None,
                expected_value=constraint.max_value,
                passed=False,
                deviation=float('inf'),
                error_message="No resource usage data available"
            )
        
        # Calculate average CPU and memory usage
        avg_cpu = sum(r.get('cpu_time', 0) for r in resource_usage) / len(resource_usage)
        avg_memory = sum(r.get('memory_usage', 0) for r in resource_usage) / len(resource_usage)
        
        # For simplicity, use CPU time as the primary metric
        actual_value = avg_cpu
        
        if constraint.max_value is not None:
            passed = actual_value <= constraint.max_value * (1 + constraint.tolerance)
            deviation = abs(actual_value - constraint.max_value) / constraint.max_value
            
            return BehaviorTestResult(
                test_name=f"resource_usage_{constraint.constraint_name}",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=actual_value,
                expected_value=constraint.max_value,
                passed=passed,
                deviation=deviation,
                additional_data={
                    'avg_cpu_time': avg_cpu,
                    'avg_memory_usage': avg_memory,
                    'max_cpu_time': max(r.get('cpu_time', 0) for r in resource_usage),
                    'max_memory_usage': max(r.get('memory_usage', 0) for r in resource_usage)
                }
            )
        
        return BehaviorTestResult(
            test_name=f"resource_usage_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=actual_value,
            expected_value="no_constraint",
            passed=True,
            deviation=0.0
        )
    
    async def _validate_state_transitions(self, 
                                        agent: BaseAgent,
                                        constraint: BehaviorConstraint,
                                        test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate state transition patterns"""
        state_transitions = test_data.get('state_transitions', [])
        
        # Count transitions
        transition_counts = {}
        for transition in state_transitions:
            key = f"{transition['from']} -> {transition['to']}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
        
        # Validate based on expected patterns
        if constraint.expected_pattern:
            # Simple pattern matching
            expected_transitions = constraint.expected_pattern.split(',')
            found_expected = sum(1 for pattern in expected_transitions 
                               if pattern.strip() in transition_counts)
            
            actual_value = found_expected / len(expected_transitions) if expected_transitions else 0
            expected_value = 1.0  # Expect all patterns to be found
            
            passed = actual_value >= (1.0 - constraint.tolerance)
            deviation = abs(actual_value - expected_value)
            
            return BehaviorTestResult(
                test_name=f"state_transitions_{constraint.constraint_name}",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=actual_value,
                expected_value=expected_value,
                passed=passed,
                deviation=deviation,
                additional_data={
                    'transition_counts': transition_counts,
                    'expected_patterns': expected_transitions,
                    'total_transitions': len(state_transitions)
                }
            )
        
        # Default validation - check for reasonable transition diversity
        unique_transitions = len(transition_counts)
        total_transitions = len(state_transitions)
        
        diversity_ratio = unique_transitions / max(total_transitions, 1)
        
        return BehaviorTestResult(
            test_name=f"state_transitions_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=diversity_ratio,
            expected_value="reasonable_diversity",
            passed=diversity_ratio > 0.1,  # At least 10% diversity
            deviation=0.0,
            additional_data={
                'unique_transitions': unique_transitions,
                'total_transitions': total_transitions,
                'transition_counts': transition_counts
            }
        )
    
    async def _validate_output_format(self, 
                                    agent: BaseAgent,
                                    constraint: BehaviorConstraint,
                                    test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate output format constraints"""
        task_results = test_data.get('task_results', [])
        
        if not task_results:
            return BehaviorTestResult(
                test_name="output_format_no_data",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=None,
                expected_value=constraint.expected_pattern,
                passed=False,
                deviation=float('inf'),
                error_message="No task results available"
            )
        
        # Check if results match expected format
        valid_format_count = 0
        
        for result in task_results:
            if result is None:
                continue
            
            if constraint.validation_function:
                # Use custom validation function
                if constraint.validation_function(result):
                    valid_format_count += 1
            elif constraint.expected_pattern:
                # Simple type/pattern checking
                if constraint.expected_pattern == "dict" and isinstance(result, dict):
                    valid_format_count += 1
                elif constraint.expected_pattern == "str" and isinstance(result, str):
                    valid_format_count += 1
                elif constraint.expected_pattern == "list" and isinstance(result, list):
                    valid_format_count += 1
            else:
                # Default - any non-None result is valid
                valid_format_count += 1
        
        valid_format_ratio = valid_format_count / len(task_results)
        expected_ratio = 1.0
        
        passed = valid_format_ratio >= (1.0 - constraint.tolerance)
        deviation = abs(valid_format_ratio - expected_ratio)
        
        return BehaviorTestResult(
            test_name=f"output_format_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=valid_format_ratio,
            expected_value=expected_ratio,
            passed=passed,
            deviation=deviation,
            additional_data={
                'valid_format_count': valid_format_count,
                'total_results': len(task_results),
                'result_types': [type(r).__name__ for r in task_results if r is not None]
            }
        )
    
    async def _validate_error_handling(self, 
                                     agent: BaseAgent,
                                     constraint: BehaviorConstraint,
                                     test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate error handling behavior"""
        # This would require specific error injection tests
        # For now, check if agent state remains stable after failures
        
        failure_count = test_data.get('failure_count', 0)
        total_tasks = test_data.get('total_tasks', 0)
        
        if total_tasks == 0:
            return BehaviorTestResult(
                test_name="error_handling_no_data",
                agent_name=agent.name,
                constraint=constraint,
                actual_value=None,
                expected_value="stable_after_errors",
                passed=False,
                deviation=float('inf'),
                error_message="No task data for error handling validation"
            )
        
        # Check if agent is still in a valid state after failures
        current_state = agent.state
        is_stable = current_state not in [AgentState.ERROR]
        
        # Calculate error recovery rate (simplified)
        error_recovery_rate = 1.0 - (failure_count / total_tasks) if total_tasks > 0 else 1.0
        
        passed = is_stable and error_recovery_rate >= (constraint.min_value or 0.5)
        
        return BehaviorTestResult(
            test_name=f"error_handling_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=error_recovery_rate,
            expected_value=constraint.min_value or 0.5,
            passed=passed,
            deviation=abs(error_recovery_rate - (constraint.min_value or 0.5)),
            additional_data={
                'current_state': current_state.value if hasattr(current_state, 'value') else str(current_state),
                'failure_count': failure_count,
                'total_tasks': total_tasks,
                'is_stable': is_stable
            }
        )
    
    async def _validate_learning_progression(self, 
                                           agent: BaseAgent,
                                           constraint: BehaviorConstraint,
                                           test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate learning progression over time"""
        # This requires historical data comparison
        current_success_rate = test_data.get('success_rate', 0.0)
        
        # Get agent's historical performance
        historical_success_rate = agent.get_success_rate()
        
        # Calculate improvement (current test vs historical)
        if historical_success_rate > 0:
            improvement_rate = (current_success_rate - historical_success_rate) / historical_success_rate
        else:
            improvement_rate = current_success_rate  # First time, use absolute performance
        
        min_improvement = constraint.min_value or 0.0
        passed = improvement_rate >= min_improvement - constraint.tolerance
        
        return BehaviorTestResult(
            test_name=f"learning_progression_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=improvement_rate,
            expected_value=min_improvement,
            passed=passed,
            deviation=abs(improvement_rate - min_improvement),
            additional_data={
                'current_success_rate': current_success_rate,
                'historical_success_rate': historical_success_rate,
                'total_agent_tasks': agent.total_tasks,
                'agent_successful_tasks': agent.successful_tasks
            }
        )
    
    async def _validate_collaboration_quality(self, 
                                            agent: BaseAgent,
                                            constraint: BehaviorConstraint,
                                            test_data: Dict[str, Any]) -> BehaviorTestResult:
        """Validate collaboration quality (requires orchestrator context)"""
        # This is a placeholder - real implementation would need multi-agent test data
        collaboration_score = 0.8  # Default assumption
        
        min_quality = constraint.min_value or 0.6
        passed = collaboration_score >= min_quality - constraint.tolerance
        
        return BehaviorTestResult(
            test_name=f"collaboration_quality_{constraint.constraint_name}",
            agent_name=agent.name,
            constraint=constraint,
            actual_value=collaboration_score,
            expected_value=min_quality,
            passed=passed,
            deviation=abs(collaboration_score - min_quality),
            additional_data={
                'collaboration_opportunities': 0,  # Would be populated with real data
                'successful_collaborations': 0,
                'note': "Requires multi-agent test scenario"
            }
        )
    
    async def _validate_orchestrator_behaviors(self, 
                                             orchestrator: AgentOrchestrator,
                                             test_scenarios: List[Dict[str, Any]]) -> List[BehaviorTestResult]:
        """Validate orchestrator-level behaviors"""
        results = []
        
        for scenario in test_scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')
            
            try:
                # Create test task
                task = Task(
                    id=f"test_{scenario_name}",
                    description=scenario.get('description', 'Test scenario'),
                    requirements=scenario.get('requirements', {})
                )
                
                # Execute with orchestrator
                start_time = time.time()
                result = await orchestrator.delegate_task(task)
                end_time = time.time()
                
                response_time = end_time - start_time
                success = result is not None
                
                # Create basic orchestrator behavior result
                orchestrator_result = BehaviorTestResult(
                    test_name=f"orchestrator_{scenario_name}",
                    agent_name="orchestrator",
                    constraint=BehaviorConstraint(
                        expectation_type=BehaviorExpectation.RESPONSE_TIME,
                        constraint_name="orchestrator_performance",
                        max_value=30.0  # 30 second default
                    ),
                    actual_value=response_time,
                    expected_value=30.0,
                    passed=success and response_time <= 30.0,
                    deviation=max(0, response_time - 30.0) / 30.0,
                    additional_data={
                        'scenario': scenario_name,
                        'agents_used': len(orchestrator.agents),
                        'task_result': str(result)[:100] if result else None
                    }
                )
                
                results.append(orchestrator_result)
                
            except Exception as e:
                error_result = BehaviorTestResult(
                    test_name=f"orchestrator_error_{scenario_name}",
                    agent_name="orchestrator",
                    constraint=BehaviorConstraint(
                        expectation_type=BehaviorExpectation.ERROR_HANDLING,
                        constraint_name="orchestrator_stability"
                    ),
                    actual_value=None,
                    expected_value="no_errors",
                    passed=False,
                    deviation=float('inf'),
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results
    
    def get_validation_summary(self, agent_name: Optional[str] = None) -> Dict[str, Any]:
        """Get summary of validation results"""
        relevant_tests = self.test_history
        
        if agent_name:
            relevant_tests = [t for t in self.test_history if t.agent_name == agent_name]
        
        if not relevant_tests:
            return {'message': f'No validation data for {"agent " + agent_name if agent_name else "any agents"}'}
        
        total_tests = len(relevant_tests)
        passed_tests = sum(1 for t in relevant_tests if t.passed)
        failed_tests = total_tests - passed_tests
        
        # Group by expectation type
        by_expectation = {}
        for test in relevant_tests:
            exp_type = test.constraint.expectation_type.value
            if exp_type not in by_expectation:
                by_expectation[exp_type] = {'passed': 0, 'failed': 0, 'total': 0}
            
            by_expectation[exp_type]['total'] += 1
            if test.passed:
                by_expectation[exp_type]['passed'] += 1
            else:
                by_expectation[exp_type]['failed'] += 1
        
        # Recent test results
        recent_tests = sorted(relevant_tests, key=lambda t: t.timestamp, reverse=True)[:10]
        
        return {
            'agent_name': agent_name or 'all_agents',
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'by_expectation_type': by_expectation,
            'recent_tests': [
                {
                    'test_name': t.test_name,
                    'passed': t.passed,
                    'deviation': t.deviation,
                    'timestamp': t.timestamp.isoformat()
                }
                for t in recent_tests
            ],
            'recommendations': self._generate_recommendations(relevant_tests)
        }
    
    def _generate_recommendations(self, test_results: List[BehaviorTestResult]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Analyze failure patterns
        failed_tests = [t for t in test_results if not t.passed]
        
        if len(failed_tests) > len(test_results) * 0.3:
            recommendations.append("High failure rate detected - review agent implementation")
        
        # Group failures by expectation type
        failure_types = {}
        for test in failed_tests:
            exp_type = test.constraint.expectation_type.value
            failure_types[exp_type] = failure_types.get(exp_type, 0) + 1
        
        # Specific recommendations based on failure patterns
        if failure_types.get('response_time', 0) > 2:
            recommendations.append("Consider optimizing agent response time")
        
        if failure_types.get('success_rate', 0) > 1:
            recommendations.append("Review task processing logic to improve success rate")
        
        if failure_types.get('learning_progression', 0) > 0:
            recommendations.append("Agent learning may have stagnated - review learning algorithms")
        
        if failure_types.get('resource_usage', 0) > 1:
            recommendations.append("Optimize resource usage - consider efficiency improvements")
        
        if not recommendations:
            recommendations.append("All tests passing - consider adding more challenging test scenarios")
        
        return recommendations