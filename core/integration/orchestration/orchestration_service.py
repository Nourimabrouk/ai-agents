"""
Orchestration Service
Coordinates agent execution using different orchestration strategies
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field

from ...shared import (
    IOrchestrator, IOrchestrationStrategy, IAgent, IAgentRepository,
    AgentId, TaskId, ExecutionContext, ExecutionResult, Priority,
    DomainEvent, IEventBus, get_service
)

logger = logging.getLogger(__name__)


class OrchestrationPattern(Enum):
    """Orchestration patterns for agent coordination"""
    SEQUENTIAL = "sequential"  # Execute agents one after another
    PARALLEL = "parallel"  # Execute agents in parallel
    HIERARCHICAL = "hierarchical"  # Tree-like delegation
    PIPELINE = "pipeline"  # Data flows through agents
    COMPETITIVE = "competitive"  # Multiple agents compete
    COLLABORATIVE = "collaborative"  # Agents work together
    ADAPTIVE = "adaptive"  # Pattern adapts based on context


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating task execution"""
    task_id: TaskId
    pattern: OrchestrationPattern
    agent_assignments: List[Tuple[IAgent, Any]]  # (agent, task_data)
    dependencies: Dict[int, List[int]]  # step -> [dependent_steps]
    timeout: Optional[float] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    
    
@dataclass
class OrchestrationResult:
    """Result of orchestration execution"""
    plan: OrchestrationPlan
    execution_results: List[ExecutionResult]
    overall_success: bool
    execution_time: float
    errors: List[Exception] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SequentialOrchestrationStrategy(IOrchestrationStrategy):
    """Execute agents sequentially"""
    
    async def plan_execution(self, context: ExecutionContext, task_data: Any) -> List[Tuple[IAgent, Any]]:
        """Plan sequential execution"""
        agent_repo = get_service(IAgentRepository)
        
        # Find agents capable of handling the task
        required_capability = self._extract_required_capability(task_data)
        agents = await agent_repo.find_agents_with_capability(required_capability)
        
        if not agents:
            raise ValueError(f"No agents found with capability: {required_capability}")
        
        # For sequential, use best agent
        best_agent = await self._select_best_agent(agents, context, task_data)
        return [(best_agent, task_data)]
    
    async def handle_failure(self, context: ExecutionContext, failed_agent: IAgent, 
                           error: Exception) -> bool:
        """Handle execution failure"""
        logger.warning(f"Agent {failed_agent.agent_id.full_id} failed: {error}")
        
        # Try to find alternative agent
        agent_repo = get_service(IAgentRepository)
        all_agents = await agent_repo.get_all_agents()
        
        # Find agents with same capabilities, excluding the failed one
        alternative_agents = [
            agent for agent in all_agents
            if (agent != failed_agent and 
                any(cap in failed_agent.capabilities for cap in agent.capabilities))
        ]
        
        if alternative_agents:
            logger.info(f"Found {len(alternative_agents)} alternative agents")
            return True
        
        return False
    
    def _extract_required_capability(self, task_data: Any) -> str:
        """Extract required capability from task data"""
        if isinstance(task_data, dict):
            return task_data.get("capability", "general")
        return "general"
    
    async def _select_best_agent(self, agents: List[IAgent], 
                               context: ExecutionContext, task_data: Any) -> IAgent:
        """Select best agent for the task"""
        # Simple selection: return first agent
        # In real implementation, this would use more sophisticated selection criteria
        return agents[0]


class ParallelOrchestrationStrategy(IOrchestrationStrategy):
    """Execute agents in parallel"""
    
    async def plan_execution(self, context: ExecutionContext, task_data: Any) -> List[Tuple[IAgent, Any]]:
        """Plan parallel execution"""
        agent_repo = get_service(IAgentRepository)
        
        # For parallel execution, distribute work across multiple agents
        if isinstance(task_data, dict) and "subtasks" in task_data:
            # Split task into subtasks
            subtasks = task_data["subtasks"]
            assignments = []
            
            for subtask in subtasks:
                required_capability = subtask.get("capability", "general")
                agents = await agent_repo.find_agents_with_capability(required_capability)
                
                if agents:
                    best_agent = agents[0]  # Simple selection
                    assignments.append((best_agent, subtask))
            
            return assignments
        else:
            # Find all capable agents and distribute work
            required_capability = self._extract_required_capability(task_data)
            agents = await agent_repo.find_agents_with_capability(required_capability)
            
            return [(agent, task_data) for agent in agents[:3]]  # Limit to 3 agents
    
    async def handle_failure(self, context: ExecutionContext, failed_agent: IAgent, 
                           error: Exception) -> bool:
        """Handle execution failure in parallel context"""
        # In parallel execution, other agents might still succeed
        logger.warning(f"Parallel agent {failed_agent.agent_id.full_id} failed: {error}")
        return True  # Continue with other agents
    
    def _extract_required_capability(self, task_data: Any) -> str:
        """Extract required capability from task data"""
        if isinstance(task_data, dict):
            return task_data.get("capability", "general")
        return "general"


class HierarchicalOrchestrationStrategy(IOrchestrationStrategy):
    """Execute agents in hierarchical pattern"""
    
    async def plan_execution(self, context: ExecutionContext, task_data: Any) -> List[Tuple[IAgent, Any]]:
        """Plan hierarchical execution"""
        agent_repo = get_service(IAgentRepository)
        
        # Create hierarchy: coordinator -> specialists
        coordinator_agents = await agent_repo.find_agents_with_capability("coordination")
        specialist_agents = await agent_repo.find_agents_with_capability("specialist")
        
        assignments = []
        
        # Add coordinator if available
        if coordinator_agents:
            coordinator_task = {
                "type": "coordination",
                "original_task": task_data,
                "specialists_available": len(specialist_agents)
            }
            assignments.append((coordinator_agents[0], coordinator_task))
        
        # Add specialists
        for i, agent in enumerate(specialist_agents[:3]):  # Limit specialists
            specialist_task = {
                "type": "execution",
                "task_part": i,
                "original_task": task_data
            }
            assignments.append((agent, specialist_task))
        
        return assignments
    
    async def handle_failure(self, context: ExecutionContext, failed_agent: IAgent, 
                           error: Exception) -> bool:
        """Handle failure in hierarchical context"""
        # If coordinator fails, try to promote a specialist
        # If specialist fails, continue with others
        return True


class OrchestrationService(IOrchestrator):
    """
    Service for orchestrating agent execution using various patterns
    """
    
    def __init__(self):
        self._strategies: Dict[OrchestrationPattern, IOrchestrationStrategy] = {}
        self._execution_history: List[OrchestrationResult] = []
        self._active_executions: Dict[TaskId, OrchestrationPlan] = {}
        self._performance_metrics: Dict[OrchestrationPattern, Dict[str, float]] = {}
        self._event_bus: Optional[IEventBus] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize orchestration service"""
        try:
            self._event_bus = get_service(IEventBus)
        except ValueError:
            logger.warning("EventBus not available, running without events")
        
        # Register built-in strategies
        await self._register_builtin_strategies()
        
        # Initialize performance metrics
        for pattern in OrchestrationPattern:
            self._performance_metrics[pattern] = {
                "total_executions": 0,
                "successful_executions": 0,
                "average_execution_time": 0.0,
                "success_rate": 0.0
            }
        
        logger.info("Orchestration service initialized")
    
    async def _register_builtin_strategies(self) -> None:
        """Register built-in orchestration strategies"""
        self._strategies[OrchestrationPattern.SEQUENTIAL] = SequentialOrchestrationStrategy()
        self._strategies[OrchestrationPattern.PARALLEL] = ParallelOrchestrationStrategy()
        self._strategies[OrchestrationPattern.HIERARCHICAL] = HierarchicalOrchestrationStrategy()
    
    async def execute_task(self, context: ExecutionContext, task_data: Any) -> ExecutionResult:
        """Execute task using appropriate orchestration pattern"""
        start_time = datetime.utcnow()
        
        try:
            # Select orchestration pattern
            pattern = await self._select_orchestration_pattern(context, task_data)
            
            # Create execution plan
            plan = await self._create_execution_plan(context, task_data, pattern)
            
            # Execute plan
            result = await self._execute_plan(plan)
            
            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_performance_metrics(pattern, result.overall_success, execution_time)
            
            # Store in history
            async with self._lock:
                self._execution_history.append(result)
                if len(self._execution_history) > 1000:  # Keep last 1000 executions
                    self._execution_history.pop(0)
            
            # Publish event
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id=f"orchestration_completed_{context.task_id.full_id}",
                    event_type="orchestration.completed",
                    source=AgentId("system", "orchestration_service"),
                    timestamp=datetime.utcnow(),
                    data={
                        "pattern": pattern.value,
                        "success": result.overall_success,
                        "execution_time": execution_time,
                        "agents_used": len(result.execution_results)
                    }
                ))
            
            # Convert to ExecutionResult
            return ExecutionResult(
                success=result.overall_success,
                result=result,
                execution_time=execution_time,
                metadata=result.metadata
            )
            
        except Exception as e:
            logger.error(f"Error in orchestration: {e}")
            
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id=f"orchestration_failed_{context.task_id.full_id}",
                    event_type="orchestration.failed",
                    source=AgentId("system", "orchestration_service"),
                    timestamp=datetime.utcnow(),
                    data={"error": str(e)}
                ))
            
            return ExecutionResult(success=False, error=e)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get orchestration system status"""
        async with self._lock:
            return {
                "active_executions": len(self._active_executions),
                "total_executions": len(self._execution_history),
                "available_patterns": [pattern.value for pattern in self._strategies.keys()],
                "performance_metrics": self._performance_metrics.copy(),
                "recent_success_rate": await self._calculate_recent_success_rate()
            }
    
    async def register_strategy(self, pattern: OrchestrationPattern, 
                              strategy: IOrchestrationStrategy) -> None:
        """Register new orchestration strategy"""
        self._strategies[pattern] = strategy
        logger.info(f"Registered orchestration strategy: {pattern.value}")
    
    async def _select_orchestration_pattern(self, context: ExecutionContext, 
                                          task_data: Any) -> OrchestrationPattern:
        """Select appropriate orchestration pattern for the task"""
        # Simple selection logic - in real implementation, this would be more sophisticated
        if isinstance(task_data, dict):
            if "subtasks" in task_data and len(task_data["subtasks"]) > 1:
                return OrchestrationPattern.PARALLEL
            elif task_data.get("complexity", "simple") == "complex":
                return OrchestrationPattern.HIERARCHICAL
            elif context.priority == Priority.CRITICAL:
                return OrchestrationPattern.SEQUENTIAL  # For reliability
        
        # Default to sequential
        return OrchestrationPattern.SEQUENTIAL
    
    async def _create_execution_plan(self, context: ExecutionContext, task_data: Any,
                                   pattern: OrchestrationPattern) -> OrchestrationPlan:
        """Create execution plan for the task"""
        if pattern not in self._strategies:
            raise ValueError(f"No strategy registered for pattern: {pattern}")
        
        strategy = self._strategies[pattern]
        agent_assignments = await strategy.plan_execution(context, task_data)
        
        plan = OrchestrationPlan(
            task_id=context.task_id,
            pattern=pattern,
            agent_assignments=agent_assignments,
            dependencies={},  # Simple implementation
            timeout=context.timeout,
            retry_policy={"max_retries": 3, "backoff": 1.0}
        )
        
        async with self._lock:
            self._active_executions[context.task_id] = plan
        
        return plan
    
    async def _execute_plan(self, plan: OrchestrationPlan) -> OrchestrationResult:
        """Execute orchestration plan"""
        start_time = datetime.utcnow()
        results = []
        errors = []
        
        try:
            if plan.pattern == OrchestrationPattern.SEQUENTIAL:
                results = await self._execute_sequential(plan)
            elif plan.pattern == OrchestrationPattern.PARALLEL:
                results = await self._execute_parallel(plan)
            elif plan.pattern == OrchestrationPattern.HIERARCHICAL:
                results = await self._execute_hierarchical(plan)
            else:
                raise ValueError(f"Execution not implemented for pattern: {plan.pattern}")
            
            # Determine overall success
            overall_success = all(result.success for result in results)
            
        except Exception as e:
            errors.append(e)
            overall_success = False
        
        finally:
            # Clean up active execution
            async with self._lock:
                self._active_executions.pop(plan.task_id, None)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return OrchestrationResult(
            plan=plan,
            execution_results=results,
            overall_success=overall_success,
            execution_time=execution_time,
            errors=errors,
            metadata={"pattern": plan.pattern.value}
        )
    
    async def _execute_sequential(self, plan: OrchestrationPlan) -> List[ExecutionResult]:
        """Execute agents sequentially"""
        results = []
        
        for agent, task_data in plan.agent_assignments:
            try:
                context = ExecutionContext(
                    task_id=plan.task_id,
                    agent_id=agent.agent_id,
                    priority=Priority.NORMAL,
                    timeout=plan.timeout
                )
                
                result = await agent.execute(context, task_data)
                results.append(result)
                
                # Stop on first failure in sequential execution
                if not result.success:
                    break
                    
            except Exception as e:
                results.append(ExecutionResult(success=False, error=e))
                break
        
        return results
    
    async def _execute_parallel(self, plan: OrchestrationPlan) -> List[ExecutionResult]:
        """Execute agents in parallel"""
        async def execute_agent(agent: IAgent, task_data: Any) -> ExecutionResult:
            try:
                context = ExecutionContext(
                    task_id=plan.task_id,
                    agent_id=agent.agent_id,
                    priority=Priority.NORMAL,
                    timeout=plan.timeout
                )
                
                return await agent.execute(context, task_data)
                
            except Exception as e:
                return ExecutionResult(success=False, error=e)
        
        # Execute all agents in parallel
        tasks = [execute_agent(agent, task_data) for agent, task_data in plan.agent_assignments]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to ExecutionResults
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(ExecutionResult(success=False, error=result))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_hierarchical(self, plan: OrchestrationPlan) -> List[ExecutionResult]:
        """Execute agents in hierarchical pattern"""
        results = []
        
        # Execute coordinator first if present
        coordinator_result = None
        specialist_assignments = []
        
        for agent, task_data in plan.agent_assignments:
            if isinstance(task_data, dict) and task_data.get("type") == "coordination":
                context = ExecutionContext(
                    task_id=plan.task_id,
                    agent_id=agent.agent_id,
                    priority=Priority.HIGH
                )
                coordinator_result = await agent.execute(context, task_data)
                results.append(coordinator_result)
            else:
                specialist_assignments.append((agent, task_data))
        
        # Execute specialists in parallel if coordinator succeeded
        if not coordinator_result or coordinator_result.success:
            specialist_results = []
            tasks = []
            
            for agent, task_data in specialist_assignments:
                async def execute_specialist(a: IAgent, td: Any) -> ExecutionResult:
                    context = ExecutionContext(
                        task_id=plan.task_id,
                        agent_id=a.agent_id,
                        priority=Priority.NORMAL
                    )
                    return await a.execute(context, td)
                
                tasks.append(execute_specialist(agent, task_data))
            
            if tasks:
                specialist_results = await asyncio.gather(*tasks, return_exceptions=True)
                for result in specialist_results:
                    if isinstance(result, Exception):
                        results.append(ExecutionResult(success=False, error=result))
                    else:
                        results.append(result)
        
        return results
    
    async def _update_performance_metrics(self, pattern: OrchestrationPattern, 
                                        success: bool, execution_time: float) -> None:
        """Update performance metrics for orchestration pattern"""
        metrics = self._performance_metrics[pattern]
        
        metrics["total_executions"] += 1
        if success:
            metrics["successful_executions"] += 1
        
        # Update running average of execution time
        n = metrics["total_executions"]
        metrics["average_execution_time"] = (
            (metrics["average_execution_time"] * (n - 1) + execution_time) / n
        )
        
        # Update success rate
        metrics["success_rate"] = metrics["successful_executions"] / metrics["total_executions"]
    
    async def _calculate_recent_success_rate(self) -> float:
        """Calculate success rate for recent executions"""
        recent_executions = [
            result for result in self._execution_history
            if (datetime.utcnow() - datetime.fromisoformat(str(result.metadata.get("timestamp", "1970-01-01")))) < timedelta(hours=1)
        ]
        
        if not recent_executions:
            return 0.0
        
        successful = sum(1 for result in recent_executions if result.overall_success)
        return successful / len(recent_executions)
    
    async def get_execution_history(self, limit: int = 100) -> List[OrchestrationResult]:
        """Get recent execution history"""
        async with self._lock:
            return self._execution_history[-limit:] if limit > 0 else self._execution_history