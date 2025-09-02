"""
Enhanced Meta-Optimal Orchestrator
==================================
Advanced orchestration system that meta-optimally utilizes all agents through:
- Intelligent parallel/sequential execution planning
- Dynamic sub-agent spawning and coordination
- Task dependency graph resolution with NetworkX
- One-shot difficult task handling
- Performance-based agent selection
- Real-time adaptation and optimization
- Tournament, ensemble, and scatter-gather execution patterns

This extends the base meta_orchestrator with advanced meta-optimization capabilities.
"""

import asyncio
import json
import networkx as nx
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
import sys
import os

# Import the base meta orchestrator
from .meta_orchestrator import MetaOrchestrator as BaseMetaOrchestrator, DevelopmentTask, AgentRole, TaskPriority

logger = logging.getLogger(__name__)


class ExecutionStrategy(Enum):
    """Advanced execution strategies for task processing"""
    PARALLEL = "parallel"  # Independent tasks run simultaneously
    SEQUENTIAL = "sequential"  # Tasks run in order with data flow
    HYBRID = "hybrid"  # Mix of parallel and sequential
    PIPELINE = "pipeline"  # Streaming pipeline execution
    MAP_REDUCE = "map_reduce"  # Map-reduce pattern
    SCATTER_GATHER = "scatter_gather"  # Scatter work, gather results
    TOURNAMENT = "tournament"  # Competitive agent selection
    ENSEMBLE = "ensemble"  # Multiple agents vote on solution
    ONE_SHOT = "one_shot"  # Single agent handles entire task
    ADAPTIVE = "adaptive"  # Dynamically adapt strategy based on performance


class TaskComplexity(Enum):
    """Task complexity levels for intelligent decomposition"""
    TRIVIAL = 1  # Single agent, simple execution
    SIMPLE = 2  # Single agent, moderate complexity
    MODERATE = 3  # Multiple agents, clear decomposition
    COMPLEX = 4  # Multiple agents, interdependencies
    EXTREME = 5  # Many agents, complex coordination


@dataclass
class TaskNode:
    """Enhanced task node for dependency graph analysis"""
    task: DevelopmentTask
    complexity: TaskComplexity
    estimated_duration: timedelta
    required_capabilities: Set[str]
    can_parallelize: bool = True
    priority: int = 0
    retry_count: int = 0
    max_retries: int = 3
    children: List['TaskNode'] = field(default_factory=list)
    parents: List['TaskNode'] = field(default_factory=list)
    execution_result: Optional[Any] = None
    execution_strategy: Optional[ExecutionStrategy] = None
    assigned_agents: List[str] = field(default_factory=list)
    performance_score: float = 0.0
    
    def __hash__(self):
        return hash(self.task.id)


@dataclass
class ExecutionPlan:
    """Comprehensive execution plan with optimization metrics"""
    root_task: TaskNode
    dependency_graph: nx.DiGraph
    execution_stages: List[List[TaskNode]]  # Tasks grouped by parallel execution stages
    strategy: ExecutionStrategy
    estimated_total_time: timedelta
    resource_requirements: Dict[str, Any]
    optimization_score: float  # How optimal this plan is (0-1)
    parallel_efficiency: float = 0.0
    resource_utilization: float = 0.0
    risk_assessment: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentPerformanceProfile:
    """Detailed performance profile for intelligent agent selection"""
    agent_id: str
    capabilities: Set[str]
    success_rate: float
    average_execution_time: timedelta
    task_type_performance: Dict[str, float]  # Task type -> success rate
    concurrent_capacity: int
    reliability_score: float
    specializations: List[str]
    failure_patterns: List[str]
    learning_rate: float = 0.1
    adaptation_score: float = 0.5
    last_updated: datetime = field(default_factory=datetime.now)


class EnhancedMetaOrchestrator(BaseMetaOrchestrator):
    """
    Enhanced Meta-Optimal Orchestrator with advanced coordination capabilities.
    
    Features:
    - Meta-optimal execution planning with multiple strategies
    - Dynamic agent performance tracking and selection
    - Intelligent task decomposition with complexity analysis
    - One-shot execution for simple tasks
    - Parallel/sequential hybrid execution
    - Real-time adaptation and learning
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__(config_path)
        
        # Enhanced orchestration components
        self.active_executions: Dict[str, ExecutionPlan] = {}
        self.agent_performance: Dict[str, AgentPerformanceProfile] = {}
        self.execution_strategies: Dict[str, ExecutionStrategy] = {}
        self.performance_history: List[Dict[str, Any]] = []
        
        # Resource management
        self.resource_pool = ResourcePool()
        self.execution_semaphore = asyncio.Semaphore(self.config.get("max_parallel_agents", 10))
        
        # Meta-optimization metrics
        self.optimization_metrics = {
            "parallelization_ratio": 0.0,
            "average_completion_time": timedelta(),
            "resource_utilization": 0.0,
            "adaptation_frequency": 0,
            "one_shot_success_rate": 0.0,
            "strategy_success_rates": defaultdict(float),
            "agent_utilization": defaultdict(float)
        }
        
        # Initialize agent performance profiles
        self._initialize_agent_performance()
        
        logger.info(f"Enhanced MetaOrchestrator initialized with {len(self.agents)} agents")
    
    def _initialize_agent_performance(self):
        """Initialize performance profiles for all agents"""
        for role, agent in self.agents.items():
            capabilities = set()
            
            # Map agent roles to capabilities
            role_capability_map = {
                AgentRole.ARCHITECT: {"design", "architecture", "planning", "system_design"},
                AgentRole.DEVELOPER: {"coding", "implementation", "programming", "development"},
                AgentRole.TESTER: {"testing", "validation", "quality_assurance", "test_automation"},
                AgentRole.REVIEWER: {"code_review", "quality_control", "security_analysis", "best_practices"},
                AgentRole.DOCUMENTER: {"documentation", "technical_writing", "user_guides", "api_docs"},
                AgentRole.INTEGRATOR: {"integration", "deployment", "system_integration", "devops"},
                AgentRole.REFACTORER: {"refactoring", "code_cleanup", "optimization", "maintenance"},
                AgentRole.DEBUGGER: {"debugging", "troubleshooting", "problem_solving", "error_analysis"}
            }
            
            capabilities = role_capability_map.get(role, set())
            
            self.agent_performance[role.value] = AgentPerformanceProfile(
                agent_id=role.value,
                capabilities=capabilities,
                success_rate=0.8,  # Start with optimistic assumption
                average_execution_time=timedelta(minutes=10),
                task_type_performance={},
                concurrent_capacity=2,
                reliability_score=0.8,
                specializations=[role.value],
                failure_patterns=[]
            )
    
    async def process_request(self, request: str, context: Dict[str, Any] = None) -> Any:
        """
        Main entry point for meta-optimal request processing.
        
        This method:
        1. Analyzes request complexity and requirements
        2. Determines if one-shot execution is possible
        3. Creates optimal execution plan if needed
        4. Executes with chosen strategy
        5. Learns from execution for future optimization
        """
        logger.info(f"Processing request with meta-optimization: {request[:100]}...")
        start_time = datetime.now()
        
        try:
            # Analyze request complexity and requirements
            analysis = await self._analyze_request_complexity(request, context)
            
            # Create root task
            root_task = await self._create_root_task(request, context, analysis)
            
            # Determine if one-shot execution is possible
            if await self._can_execute_one_shot(analysis):
                logger.info("Attempting one-shot execution")
                result = await self._execute_one_shot(root_task, analysis)
                if result is not None:
                    self._record_execution(root_task, result, datetime.now() - start_time, ExecutionStrategy.ONE_SHOT)
                    return result
                logger.info("One-shot failed, falling back to orchestrated execution")
            
            # Create optimal execution plan
            execution_plan = await self._create_optimal_execution_plan(root_task, analysis)
            
            # Execute plan with chosen strategy
            result = await self._execute_plan_optimally(execution_plan)
            
            # Record execution and learn
            execution_time = datetime.now() - start_time
            self._record_execution(root_task, result, execution_time, execution_plan.strategy)
            await self._learn_from_execution(execution_plan, result, execution_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in meta-optimal processing: {e}")
            # Fallback to base orchestrator
            return await super().plan_development(request, context)
    
    async def _analyze_request_complexity(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Advanced complexity analysis using multiple heuristics.
        """
        analysis = {
            "complexity": TaskComplexity.MODERATE,
            "estimated_duration": timedelta(minutes=15),
            "required_capabilities": set(),
            "can_parallelize": True,
            "task_type": "development",
            "priority": 0,
            "risk_factors": [],
            "optimization_potential": 0.5
        }
        
        request_lower = request.lower()
        
        # Complexity detection with multiple indicators
        complexity_indicators = {
            TaskComplexity.TRIVIAL: ["simple", "basic", "quick", "easy", "straightforward"],
            TaskComplexity.SIMPLE: ["implement", "create", "add", "update", "modify"],
            TaskComplexity.MODERATE: ["develop", "build", "design", "analyze", "integrate"],
            TaskComplexity.COMPLEX: ["advanced", "complex", "sophisticated", "comprehensive", "multi-step"],
            TaskComplexity.EXTREME: ["extreme", "very complex", "highly sophisticated", "enterprise-grade", "full-scale"]
        }
        
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                analysis["complexity"] = complexity
                break
        
        # Duration estimation based on complexity and context
        duration_map = {
            TaskComplexity.TRIVIAL: timedelta(minutes=2),
            TaskComplexity.SIMPLE: timedelta(minutes=8),
            TaskComplexity.MODERATE: timedelta(minutes=20),
            TaskComplexity.COMPLEX: timedelta(minutes=45),
            TaskComplexity.EXTREME: timedelta(hours=2)
        }
        
        base_duration = duration_map[analysis["complexity"]]
        
        # Adjust based on context
        if context:
            if context.get("priority") == "high":
                base_duration *= 0.8  # Faster execution under pressure
            if context.get("quality") == "high":
                base_duration *= 1.5  # More time for quality
            if context.get("frameworks"):
                base_duration *= (1 + len(context["frameworks"]) * 0.1)  # More frameworks = more time
        
        analysis["estimated_duration"] = base_duration
        
        # Capability detection
        capability_keywords = {
            "coding": ["code", "implement", "program", "develop", "script"],
            "design": ["design", "architect", "structure", "model", "blueprint"],
            "testing": ["test", "validate", "verify", "check", "quality"],
            "documentation": ["document", "explain", "describe", "guide", "manual"],
            "integration": ["integrate", "connect", "api", "system", "deploy"],
            "analysis": ["analyze", "evaluate", "assess", "review", "audit"],
            "data_processing": ["data", "process", "extract", "transform", "load"],
            "machine_learning": ["ml", "ai", "model", "training", "prediction"]
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in request_lower for keyword in keywords):
                analysis["required_capabilities"].add(capability)
        
        # Parallelization detection
        sequential_indicators = ["step by step", "sequential", "one by one", "in order", "dependencies"]
        if any(indicator in request_lower for indicator in sequential_indicators):
            analysis["can_parallelize"] = False
        
        # Risk assessment
        risk_keywords = ["critical", "security", "production", "live", "financial"]
        analysis["risk_factors"] = [keyword for keyword in risk_keywords if keyword in request_lower]
        
        # Optimization potential
        if analysis["complexity"] in [TaskComplexity.COMPLEX, TaskComplexity.EXTREME]:
            analysis["optimization_potential"] = 0.8
        elif analysis["can_parallelize"]:
            analysis["optimization_potential"] = 0.7
        
        return analysis
    
    async def _create_root_task(self, request: str, context: Dict[str, Any], analysis: Dict[str, Any]) -> DevelopmentTask:
        """Create enhanced root task with analysis results"""
        return DevelopmentTask(
            id=self._generate_task_id(request),
            description=request,
            priority=TaskPriority.HIGH if context and context.get("priority") == "high" else TaskPriority.MEDIUM,
            context=context or {},
            complexity=analysis["complexity"],
            estimated_duration=analysis["estimated_duration"],
            required_capabilities=analysis["required_capabilities"],
            can_parallelize=analysis["can_parallelize"]
        )
    
    async def _can_execute_one_shot(self, analysis: Dict[str, Any]) -> bool:
        """
        Determine if task can be completed in one shot.
        Uses agent performance history and task characteristics.
        """
        complexity = analysis["complexity"]
        
        # Simple tasks are good candidates for one-shot
        if complexity in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
            return True
        
        # Check if we have a high-performing specialist
        required_capabilities = analysis["required_capabilities"]
        if not required_capabilities:
            return False
        
        for profile in self.agent_performance.values():
            capability_match = len(required_capabilities.intersection(profile.capabilities)) / len(required_capabilities)
            if capability_match > 0.8 and profile.reliability_score > 0.85:
                return True
        
        # Consider optimization potential
        if analysis["optimization_potential"] < 0.3:
            return True
        
        return False
    
    async def _execute_one_shot(self, task: DevelopmentTask, analysis: Dict[str, Any]) -> Optional[Any]:
        """
        Execute task with single best-matched agent.
        """
        try:
            # Select best agent
            best_agent_role = await self._select_optimal_agent(
                analysis["required_capabilities"],
                analysis.get("task_type", "development")
            )
            
            if not best_agent_role or best_agent_role not in self.agents:
                return None
            
            best_agent = self.agents[best_agent_role]
            
            # Execute with timeout and monitoring
            timeout_seconds = analysis["estimated_duration"].total_seconds() * 1.5
            
            logger.info(f"Executing one-shot with {best_agent_role.value}")
            
            result = await asyncio.wait_for(
                self._execute_single_task_enhanced(task),
                timeout=timeout_seconds
            )
            
            # Update success metrics
            self.optimization_metrics["one_shot_success_rate"] = (
                self.optimization_metrics["one_shot_success_rate"] * 0.9 + 0.1
            )
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"One-shot execution timed out for task {task.id}")
        except Exception as e:
            logger.error(f"One-shot execution failed: {e}")
        
        return None
    
    async def _create_optimal_execution_plan(self, root_task: DevelopmentTask, analysis: Dict[str, Any]) -> ExecutionPlan:
        """
        Create execution plan with advanced optimization.
        """
        logger.info(f"Creating optimal execution plan for task {root_task.id}")
        
        # Decompose into subtasks
        subtasks = await self._intelligent_task_decomposition(root_task, analysis)
        
        # Create task nodes
        task_nodes = [TaskNode(
            task=task,
            complexity=TaskComplexity.SIMPLE if analysis["complexity"] <= TaskComplexity.MODERATE else TaskComplexity.MODERATE,
            estimated_duration=analysis["estimated_duration"] / len(subtasks) if subtasks else analysis["estimated_duration"],
            required_capabilities=analysis["required_capabilities"]
        ) for task in subtasks]
        
        # Build dependency graph
        dependency_graph = self._build_advanced_dependency_graph(task_nodes)
        
        # Identify execution stages
        execution_stages = self._identify_optimal_execution_stages(dependency_graph)
        
        # Choose execution strategy
        strategy = await self._select_optimal_strategy(execution_stages, analysis)
        
        # Estimate resources
        resource_requirements = self._estimate_resource_requirements(task_nodes, strategy)
        
        # Calculate optimization scores
        optimization_score = self._calculate_comprehensive_optimization_score(
            execution_stages, strategy, resource_requirements, analysis
        )
        
        return ExecutionPlan(
            root_task=TaskNode(
                task=root_task,
                complexity=analysis["complexity"],
                estimated_duration=analysis["estimated_duration"],
                required_capabilities=analysis["required_capabilities"],
                children=task_nodes
            ),
            dependency_graph=dependency_graph,
            execution_stages=execution_stages,
            strategy=strategy,
            estimated_total_time=analysis["estimated_duration"],
            resource_requirements=resource_requirements,
            optimization_score=optimization_score
        )
    
    async def _intelligent_task_decomposition(self, task: DevelopmentTask, analysis: Dict[str, Any]) -> List[DevelopmentTask]:
        """
        Advanced task decomposition using pattern recognition.
        """
        if analysis["complexity"] in [TaskComplexity.TRIVIAL, TaskComplexity.SIMPLE]:
            return [task]
        
        # Advanced decomposition patterns
        decomposition_patterns = {
            "development": {
                "high_level": ["research", "design", "implement", "test", "deploy", "monitor"],
                "implementation": ["setup", "core_logic", "integration", "testing", "optimization"],
                "analysis": ["data_collection", "preprocessing", "analysis", "visualization", "reporting"]
            },
            "research": {
                "investigation": ["problem_definition", "literature_review", "data_gathering", "analysis", "synthesis"],
                "evaluation": ["criteria_definition", "option_identification", "comparison", "recommendation"]
            }
        }
        
        task_type = analysis.get("task_type", "development")
        pattern_type = "high_level"  # Could be made more sophisticated
        
        if task_type in decomposition_patterns:
            pattern = decomposition_patterns[task_type].get(pattern_type, ["analyze", "plan", "execute", "validate"])
        else:
            pattern = ["analyze", "plan", "execute", "validate"]
        
        subtasks = []
        for i, step in enumerate(pattern):
            subtask = DevelopmentTask(
                id=f"{task.id}_{step}",
                description=f"{step} for {task.description}",
                priority=task.priority,
                context=task.context,
                dependencies=[f"{task.id}_{pattern[i-1]}"] if i > 0 else [],
                estimated_duration=analysis["estimated_duration"] / len(pattern)
            )
            subtasks.append(subtask)
        
        return subtasks
    
    def _build_advanced_dependency_graph(self, task_nodes: List[TaskNode]) -> nx.DiGraph:
        """
        Build sophisticated dependency graph with NetworkX.
        """
        graph = nx.DiGraph()
        
        # Add nodes with attributes
        for node in task_nodes:
            graph.add_node(node.task.id, 
                          task_node=node,
                          complexity=node.complexity.value,
                          duration=node.estimated_duration.total_seconds(),
                          capabilities=list(node.required_capabilities))
        
        # Add dependency edges
        for node in task_nodes:
            for dep_id in node.task.dependencies:
                if graph.has_node(dep_id):
                    graph.add_edge(dep_id, node.task.id, dependency_type="sequential")
        
        # Add inferred dependencies based on capability overlap
        for node1 in task_nodes:
            for node2 in task_nodes:
                if (node1 != node2 and 
                    node1.required_capabilities.intersection(node2.required_capabilities) and
                    not graph.has_edge(node1.task.id, node2.task.id) and
                    not graph.has_edge(node2.task.id, node1.task.id)):
                    
                    # Add weak dependency for resource sharing
                    graph.add_edge(node1.task.id, node2.task.id, dependency_type="resource")
        
        return graph
    
    def _identify_optimal_execution_stages(self, graph: nx.DiGraph) -> List[List[TaskNode]]:
        """
        Advanced stage identification with optimization.
        """
        if not graph.nodes():
            return []
        
        # Use topological sort with level detection
        try:
            stages = []
            remaining_nodes = set(graph.nodes())
            
            while remaining_nodes:
                # Find nodes with no dependencies in remaining set
                current_stage = []
                for node_id in remaining_nodes.copy():
                    predecessors = set(graph.predecessors(node_id))
                    # Only consider sequential dependencies for staging
                    sequential_preds = {
                        pred for pred in predecessors 
                        if graph[pred][node_id].get("dependency_type") == "sequential"
                    }
                    
                    if not sequential_preds.intersection(remaining_nodes):
                        current_stage.append(graph.nodes[node_id]["task_node"])
                        remaining_nodes.remove(node_id)
                
                if not current_stage:
                    # Handle cycles by taking the node with minimum incoming edges
                    min_node = min(remaining_nodes, 
                                  key=lambda n: len(list(graph.predecessors(n))))
                    current_stage.append(graph.nodes[min_node]["task_node"])
                    remaining_nodes.remove(min_node)
                
                stages.append(current_stage)
            
            return stages
            
        except Exception as e:
            logger.error(f"Error in stage identification: {e}")
            # Fallback: return all tasks in single stage
            return [[graph.nodes[node_id]["task_node"] for node_id in graph.nodes()]]
    
    async def _select_optimal_strategy(self, stages: List[List[TaskNode]], analysis: Dict[str, Any]) -> ExecutionStrategy:
        """
        Select optimal execution strategy based on multiple factors.
        """
        if not stages:
            return ExecutionStrategy.SEQUENTIAL
        
        # Calculate metrics
        total_tasks = sum(len(stage) for stage in stages)
        max_parallel = max(len(stage) for stage in stages) if stages else 1
        parallelization_ratio = max_parallel / total_tasks if total_tasks > 0 else 0
        
        complexity = analysis["complexity"]
        risk_factors = analysis.get("risk_factors", [])
        
        # Strategy selection with multiple criteria
        if len(risk_factors) > 2:
            # High risk = more conservative approach
            return ExecutionStrategy.SEQUENTIAL
        elif complexity == TaskComplexity.EXTREME and parallelization_ratio > 0.6:
            return ExecutionStrategy.MAP_REDUCE
        elif parallelization_ratio > 0.8:
            return ExecutionStrategy.PARALLEL
        elif parallelization_ratio > 0.5 and len(stages) > 3:
            return ExecutionStrategy.HYBRID
        elif any("data" in cap for cap in analysis["required_capabilities"]):
            return ExecutionStrategy.PIPELINE
        elif complexity == TaskComplexity.COMPLEX:
            return ExecutionStrategy.TOURNAMENT  # Let agents compete for complex tasks
        else:
            return ExecutionStrategy.SEQUENTIAL
    
    def _estimate_resource_requirements(self, task_nodes: List[TaskNode], strategy: ExecutionStrategy) -> Dict[str, Any]:
        """
        Advanced resource estimation.
        """
        total_complexity = sum(node.complexity.value for node in task_nodes)
        total_duration = sum(node.estimated_duration.total_seconds() for node in task_nodes)
        unique_capabilities = set()
        for node in task_nodes:
            unique_capabilities.update(node.required_capabilities)
        
        # Base resource calculation
        base_agents = len(task_nodes)
        base_memory = total_complexity * 150  # MB per complexity unit
        base_cpu = total_complexity * 15  # CPU units per complexity unit
        
        # Strategy-specific adjustments
        strategy_multipliers = {
            ExecutionStrategy.PARALLEL: {"agents": 1.0, "memory": 1.2, "cpu": 1.1},
            ExecutionStrategy.SEQUENTIAL: {"agents": 0.3, "memory": 0.8, "cpu": 0.9},
            ExecutionStrategy.HYBRID: {"agents": 0.7, "memory": 1.0, "cpu": 1.0},
            ExecutionStrategy.TOURNAMENT: {"agents": 1.5, "memory": 1.3, "cpu": 1.2},
            ExecutionStrategy.ENSEMBLE: {"agents": 2.0, "memory": 1.5, "cpu": 1.3}
        }
        
        multipliers = strategy_multipliers.get(strategy, {"agents": 1.0, "memory": 1.0, "cpu": 1.0})
        
        return {
            "agents_required": min(int(base_agents * multipliers["agents"]), self.config.get("max_parallel_agents", 10)),
            "memory_mb": int(base_memory * multipliers["memory"]),
            "cpu_units": int(base_cpu * multipliers["cpu"]),
            "estimated_duration_seconds": total_duration,
            "unique_capabilities": len(unique_capabilities),
            "strategy": strategy.value
        }
    
    def _calculate_comprehensive_optimization_score(self, stages: List[List[TaskNode]], 
                                                  strategy: ExecutionStrategy,
                                                  resources: Dict[str, Any],
                                                  analysis: Dict[str, Any]) -> float:
        """
        Advanced optimization score calculation.
        """
        scores = []
        
        # Parallelization efficiency (0-1)
        if stages:
            total_tasks = sum(len(stage) for stage in stages)
            parallel_tasks = sum(max(0, len(stage) - 1) for stage in stages)
            if total_tasks > 0:
                parallelization_score = parallel_tasks / total_tasks
                scores.append(parallelization_score * 0.3)
        
        # Resource efficiency (0-1)
        max_agents = self.config.get("max_parallel_agents", 10)
        resource_efficiency = 1.0 - (resources.get("agents_required", 1) / max_agents)
        scores.append(resource_efficiency * 0.25)
        
        # Strategy appropriateness (0-1)
        strategy_scores = {
            ExecutionStrategy.PARALLEL: 0.9,
            ExecutionStrategy.HYBRID: 0.85,
            ExecutionStrategy.PIPELINE: 0.8,
            ExecutionStrategy.MAP_REDUCE: 0.85,
            ExecutionStrategy.TOURNAMENT: 0.75,
            ExecutionStrategy.ENSEMBLE: 0.8,
            ExecutionStrategy.ONE_SHOT: 0.95,
            ExecutionStrategy.SEQUENTIAL: 0.6
        }
        scores.append(strategy_scores.get(strategy, 0.5) * 0.2)
        
        # Complexity handling (0-1)
        complexity = analysis["complexity"]
        complexity_scores = {
            TaskComplexity.TRIVIAL: 0.95,
            TaskComplexity.SIMPLE: 0.9,
            TaskComplexity.MODERATE: 0.8,
            TaskComplexity.COMPLEX: 0.7,
            TaskComplexity.EXTREME: 0.6
        }
        scores.append(complexity_scores.get(complexity, 0.5) * 0.15)
        
        # Risk mitigation (0-1)
        risk_factors = len(analysis.get("risk_factors", []))
        risk_score = max(0, 1.0 - risk_factors * 0.2)
        scores.append(risk_score * 0.1)
        
        return sum(scores)
    
    async def _execute_plan_optimally(self, plan: ExecutionPlan) -> Any:
        """
        Execute plan with chosen strategy and real-time monitoring.
        """
        logger.info(f"Executing plan with strategy: {plan.strategy.value}")
        self.active_executions[plan.root_task.task.id] = plan
        
        try:
            # Update optimization metrics
            self.optimization_metrics["parallelization_ratio"] = (
                self.optimization_metrics["parallelization_ratio"] * 0.9 + 
                plan.optimization_score * 0.1
            )
            
            # Execute based on strategy
            if plan.strategy == ExecutionStrategy.PARALLEL:
                result = await self._execute_parallel_optimized(plan)
            elif plan.strategy == ExecutionStrategy.SEQUENTIAL:
                result = await self._execute_sequential_optimized(plan)
            elif plan.strategy == ExecutionStrategy.HYBRID:
                result = await self._execute_hybrid_optimized(plan)
            elif plan.strategy == ExecutionStrategy.TOURNAMENT:
                result = await self._execute_tournament_optimized(plan)
            elif plan.strategy == ExecutionStrategy.ENSEMBLE:
                result = await self._execute_ensemble_optimized(plan)
            elif plan.strategy == ExecutionStrategy.SCATTER_GATHER:
                result = await self._execute_scatter_gather_optimized(plan)
            else:
                result = await self._execute_sequential_optimized(plan)
            
            return result
            
        finally:
            if plan.root_task.task.id in self.active_executions:
                del self.active_executions[plan.root_task.task.id]
    
    async def _execute_parallel_optimized(self, plan: ExecutionPlan) -> Any:
        """Optimized parallel execution with load balancing."""
        results = {}
        
        for stage_idx, stage in enumerate(plan.execution_stages):
            logger.info(f"Executing stage {stage_idx + 1}/{len(plan.execution_stages)} with {len(stage)} tasks")
            
            # Execute all tasks in stage concurrently
            stage_tasks = []
            for node in stage:
                stage_tasks.append(self._execute_task_node_optimized(node))
            
            if stage_tasks:
                stage_results = await asyncio.gather(*stage_tasks, return_exceptions=True)
                
                for i, node in enumerate(stage):
                    if i < len(stage_results):
                        result = stage_results[i]
                        if not isinstance(result, Exception):
                            results[node.task.id] = result
                            node.execution_result = result
                            node.performance_score = 1.0
                        else:
                            logger.error(f"Task {node.task.id} failed: {result}")
                            node.performance_score = 0.0
        
        return self._synthesize_results_optimized(results, plan)
    
    async def _execute_sequential_optimized(self, plan: ExecutionPlan) -> Any:
        """Optimized sequential execution with context passing."""
        result = None
        context = {}
        
        for stage_idx, stage in enumerate(plan.execution_stages):
            for node in stage:
                # Pass accumulated context
                node.task.context.update(context)
                if result is not None:
                    node.task.context["previous_result"] = result
                
                result = await self._execute_task_node_optimized(node)
                
                if result and isinstance(result, dict):
                    context.update(result.get("context", {}))
                
                node.execution_result = result
        
        return result
    
    async def _execute_hybrid_optimized(self, plan: ExecutionPlan) -> Any:
        """Advanced hybrid execution with dynamic adaptation."""
        results = {}
        stage_context = None
        
        for stage_idx, stage in enumerate(plan.execution_stages):
            if len(stage) > 1:
                # Parallel execution for multiple tasks
                logger.info(f"Parallel execution for stage {stage_idx + 1} with {len(stage)} tasks")
                
                stage_tasks = []
                for node in stage:
                    if stage_context:
                        node.task.context["stage_context"] = stage_context
                    stage_tasks.append(self._execute_task_node_optimized(node))
                
                stage_results = await asyncio.gather(*stage_tasks, return_exceptions=True)
                
                for i, node in enumerate(stage):
                    if i < len(stage_results) and not isinstance(stage_results[i], Exception):
                        results[node.task.id] = stage_results[i]
                        node.execution_result = stage_results[i]
                
                stage_context = {"parallel_results": stage_results}
                
            else:
                # Sequential execution for single task
                node = stage[0]
                if stage_context:
                    node.task.context["stage_context"] = stage_context
                
                result = await self._execute_task_node_optimized(node)
                results[node.task.id] = result
                node.execution_result = result
                stage_context = result
        
        return self._synthesize_results_optimized(results, plan)
    
    async def _execute_tournament_optimized(self, plan: ExecutionPlan) -> Any:
        """Tournament execution where agents compete for best results."""
        if not plan.execution_stages:
            return None
        
        logger.info("Executing tournament strategy")
        
        # Get all available agents for competition
        competing_agents = list(self.agents.keys())
        results = []
        
        # Execute same task with different agents
        for stage in plan.execution_stages[:1]:  # Use first stage for tournament
            for node in stage:
                agent_results = []
                
                # Have multiple agents attempt the same task
                for agent_role in competing_agents[:3]:  # Top 3 agents compete
                    try:
                        node_copy = TaskNode(
                            task=DevelopmentTask(
                                id=f"{node.task.id}_{agent_role.value}",
                                description=node.task.description,
                                priority=node.task.priority,
                                context=node.task.context.copy()
                            ),
                            complexity=node.complexity,
                            estimated_duration=node.estimated_duration,
                            required_capabilities=node.required_capabilities
                        )
                        
                        result = await self._execute_task_with_specific_agent(node_copy, agent_role)
                        if result:
                            agent_results.append((agent_role, result, await self._evaluate_result_quality(result)))
                    
                    except Exception as e:
                        logger.warning(f"Agent {agent_role.value} failed in tournament: {e}")
                
                # Select best result
                if agent_results:
                    best_agent, best_result, best_score = max(agent_results, key=lambda x: x[2])
                    logger.info(f"Tournament winner: {best_agent.value} with score {best_score}")
                    results.append(best_result)
                    
                    # Update agent performance
                    await self._update_agent_performance(best_agent.value, True, node)
        
        return results[0] if results else None
    
    async def _execute_ensemble_optimized(self, plan: ExecutionPlan) -> Any:
        """Ensemble execution where multiple agents vote on solution."""
        logger.info("Executing ensemble strategy")
        
        ensemble_results = []
        
        for stage in plan.execution_stages:
            for node in stage:
                # Get multiple agent perspectives
                agent_results = []
                available_agents = list(self.agents.keys())
                
                for agent_role in available_agents[:min(5, len(available_agents))]:  # Use up to 5 agents
                    try:
                        node_copy = TaskNode(
                            task=DevelopmentTask(
                                id=f"{node.task.id}_{agent_role.value}",
                                description=node.task.description,
                                priority=node.task.priority,
                                context=node.task.context.copy()
                            ),
                            complexity=node.complexity,
                            estimated_duration=node.estimated_duration,
                            required_capabilities=node.required_capabilities
                        )
                        
                        result = await self._execute_task_with_specific_agent(node_copy, agent_role)
                        if result:
                            agent_results.append(result)
                    
                    except Exception as e:
                        logger.warning(f"Agent {agent_role.value} failed in ensemble: {e}")
                
                # Aggregate results through voting/averaging
                if agent_results:
                    ensemble_result = await self._aggregate_ensemble_results(agent_results)
                    ensemble_results.append(ensemble_result)
        
        return self._synthesize_results_optimized({"ensemble": ensemble_results}, plan)
    
    async def _execute_scatter_gather_optimized(self, plan: ExecutionPlan) -> Any:
        """Scatter-gather execution for exploring multiple solution approaches."""
        logger.info("Executing scatter-gather strategy")
        
        scattered_results = []
        
        for stage in plan.execution_stages:
            for node in stage:
                # Create multiple approaches to the same problem
                approaches = []
                
                for i in range(3):  # Try 3 different approaches
                    approach_context = node.task.context.copy()
                    approach_context["approach"] = f"approach_{i + 1}"
                    approach_context["strategy"] = ["conservative", "aggressive", "balanced"][i]
                    
                    approach_node = TaskNode(
                        task=DevelopmentTask(
                            id=f"{node.task.id}_approach_{i + 1}",
                            description=f"{node.task.description} (using {approach_context['strategy']} approach)",
                            priority=node.task.priority,
                            context=approach_context
                        ),
                        complexity=node.complexity,
                        estimated_duration=node.estimated_duration,
                        required_capabilities=node.required_capabilities
                    )
                    
                    approaches.append(self._execute_task_node_optimized(approach_node))
                
                # Execute all approaches in parallel
                approach_results = await asyncio.gather(*approaches, return_exceptions=True)
                
                # Filter successful results
                valid_results = [r for r in approach_results if not isinstance(r, Exception) and r is not None]
                
                if valid_results:
                    # Select best approach or combine results
                    best_result = await self._select_best_approach(valid_results)
                    scattered_results.append(best_result)
        
        return scattered_results[0] if scattered_results else None
    
    async def _execute_task_node_optimized(self, node: TaskNode) -> Any:
        """Enhanced task execution with performance monitoring."""
        start_time = datetime.now()
        attempt = 0
        last_error = None
        
        while attempt <= node.max_retries:
            try:
                # Select optimal agent for this task
                optimal_agent_role = await self._select_optimal_agent(
                    node.required_capabilities,
                    node.task.context.get("task_type", "development")
                )
                
                if not optimal_agent_role:
                    raise ValueError("No suitable agent found")
                
                # Execute with monitoring
                timeout_seconds = node.estimated_duration.total_seconds() * (attempt + 1.5)
                
                result = await asyncio.wait_for(
                    self._execute_task_with_specific_agent(node, optimal_agent_role),
                    timeout=timeout_seconds
                )
                
                # Update performance metrics
                execution_time = datetime.now() - start_time
                await self._update_agent_performance(optimal_agent_role.value, True, node)
                node.performance_score = 1.0 - (execution_time.total_seconds() / timeout_seconds)
                
                return result
                
            except asyncio.TimeoutError:
                last_error = f"Timeout after {timeout_seconds} seconds"
                logger.warning(f"Task {node.task.id} timed out on attempt {attempt + 1}")
            except Exception as e:
                last_error = str(e)
                logger.error(f"Task {node.task.id} failed on attempt {attempt + 1}: {e}")
            
            attempt += 1
            node.retry_count = attempt
            
            # Exponential backoff with jitter
            if attempt <= node.max_retries:
                backoff_time = (2 ** attempt) + (hash(node.task.id) % 3)
                await asyncio.sleep(backoff_time)
        
        # Final failure
        await self._update_agent_performance(optimal_agent_role.value, False, node)
        node.performance_score = 0.0
        
        logger.error(f"Task {node.task.id} failed after {attempt} attempts: {last_error}")
        return {"success": False, "error": last_error, "attempts": attempt}
    
    async def _execute_task_with_specific_agent(self, node: TaskNode, agent_role: AgentRole) -> Any:
        """Execute task with specific agent."""
        if agent_role not in self.agents:
            raise ValueError(f"Agent {agent_role.value} not available")
        
        agent = self.agents[agent_role]
        
        # Use the base orchestrator's task execution
        return await self._execute_single_task(node.task)
    
    async def _select_optimal_agent(self, required_capabilities: Set[str], task_type: str) -> Optional[AgentRole]:
        """Advanced agent selection based on performance and capabilities."""
        if not required_capabilities:
            # Default to developer for general tasks
            return AgentRole.DEVELOPER
        
        best_agent = None
        best_score = 0.0
        
        for role, profile in self.agent_performance.items():
            # Calculate capability match score
            capability_overlap = len(required_capabilities.intersection(profile.capabilities))
            capability_score = capability_overlap / len(required_capabilities) if required_capabilities else 0
            
            # Calculate performance score
            performance_score = (profile.success_rate * 0.4 + 
                               profile.reliability_score * 0.4 +
                               profile.adaptation_score * 0.2)
            
            # Task type specific performance
            task_type_score = profile.task_type_performance.get(task_type, 0.5)
            
            # Combined score
            total_score = (capability_score * 0.5 + 
                          performance_score * 0.3 + 
                          task_type_score * 0.2)
            
            if total_score > best_score:
                best_score = total_score
                best_agent = AgentRole(role) if isinstance(role, str) else role
        
        return best_agent
    
    async def _evaluate_result_quality(self, result: Any) -> float:
        """Evaluate quality of execution result."""
        if result is None:
            return 0.0
        
        score = 0.5  # Base score
        
        if isinstance(result, dict):
            if result.get("success"):
                score += 0.3
            if "error" in result:
                score -= 0.2
            if "results" in result or "output" in result:
                score += 0.2
            if result.get("quality_metrics"):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _aggregate_ensemble_results(self, results: List[Any]) -> Any:
        """Intelligent aggregation of ensemble results."""
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Analyze result types
        dict_results = [r for r in results if isinstance(r, dict)]
        string_results = [r for r in results if isinstance(r, str)]
        
        if dict_results:
            # Aggregate dictionary results
            aggregated = {}
            
            # Merge common keys
            all_keys = set()
            for result in dict_results:
                all_keys.update(result.keys())
            
            for key in all_keys:
                values = [r.get(key) for r in dict_results if key in r]
                if values:
                    # Use majority vote for boolean/string values
                    if all(isinstance(v, bool) for v in values):
                        aggregated[key] = sum(values) > len(values) / 2
                    elif all(isinstance(v, (int, float)) for v in values):
                        aggregated[key] = sum(values) / len(values)  # Average
                    else:
                        # Use most common value
                        from collections import Counter
                        aggregated[key] = Counter(values).most_common(1)[0][0]
            
            aggregated["ensemble_confidence"] = len(dict_results) / len(results)
            return aggregated
        
        elif string_results:
            # For string results, return the longest one (assuming more detail is better)
            return max(string_results, key=len)
        
        # Fallback: return first result
        return results[0]
    
    async def _select_best_approach(self, results: List[Any]) -> Any:
        """Select best approach from scatter-gather results."""
        if not results:
            return None
        
        if len(results) == 1:
            return results[0]
        
        # Score each result
        scored_results = []
        for result in results:
            quality_score = await self._evaluate_result_quality(result)
            scored_results.append((result, quality_score))
        
        # Return highest scoring result
        return max(scored_results, key=lambda x: x[1])[0]
    
    def _synthesize_results_optimized(self, results: Dict[str, Any], plan: ExecutionPlan) -> Any:
        """Advanced result synthesis with metadata."""
        if not results:
            return {"success": False, "error": "No results generated", "plan": plan.strategy.value}
        
        if len(results) == 1:
            single_result = list(results.values())[0]
            if isinstance(single_result, dict):
                single_result.update({
                    "execution_strategy": plan.strategy.value,
                    "optimization_score": plan.optimization_score,
                    "execution_stages": len(plan.execution_stages)
                })
            return single_result
        
        # Multi-result synthesis
        synthesized = {
            "execution_strategy": plan.strategy.value,
            "optimization_score": plan.optimization_score,
            "execution_stages": len(plan.execution_stages),
            "total_tasks": len(results),
            "results": results,
            "success": True,
            "summary": f"Successfully executed {len(results)} tasks using {plan.strategy.value} strategy"
        }
        
        # Calculate aggregate metrics
        if all(isinstance(r, dict) and "success" in r for r in results.values()):
            success_count = sum(1 for r in results.values() if r.get("success"))
            synthesized["success_rate"] = success_count / len(results)
        
        return synthesized
    
    async def _update_agent_performance(self, agent_id: str, success: bool, node: TaskNode):
        """Update agent performance with learning."""
        if agent_id not in self.agent_performance:
            return
        
        profile = self.agent_performance[agent_id]
        
        # Update success rate with exponential moving average
        learning_rate = profile.learning_rate
        profile.success_rate = profile.success_rate * (1 - learning_rate) + (1.0 if success else 0.0) * learning_rate
        
        # Update task type performance
        task_type = node.task.context.get("task_type", "general")
        if task_type not in profile.task_type_performance:
            profile.task_type_performance[task_type] = 0.5
        
        profile.task_type_performance[task_type] = (
            profile.task_type_performance[task_type] * (1 - learning_rate) + 
            (1.0 if success else 0.0) * learning_rate
        )
        
        # Update reliability score
        profile.reliability_score = (
            profile.success_rate * 0.6 + 
            sum(profile.task_type_performance.values()) / max(len(profile.task_type_performance), 1) * 0.4
        )
        
        # Update adaptation score based on recent performance
        profile.adaptation_score = min(1.0, profile.adaptation_score + (0.1 if success else -0.1))
        profile.last_updated = datetime.now()
        
        # Update global metrics
        self.optimization_metrics["agent_utilization"][agent_id] = (
            self.optimization_metrics["agent_utilization"][agent_id] * 0.9 + 
            (1.0 if success else 0.5) * 0.1
        )
    
    def _record_execution(self, task: DevelopmentTask, result: Any, duration: timedelta, strategy: ExecutionStrategy):
        """Enhanced execution recording with strategy tracking."""
        execution_record = {
            "task_id": task.id,
            "description": task.description,
            "strategy": strategy.value,
            "result": result,
            "duration": duration.total_seconds(),
            "timestamp": datetime.now(),
            "success": result is not None and (not isinstance(result, dict) or result.get("success", True)),
            "complexity": getattr(task, "complexity", TaskComplexity.MODERATE).value
        }
        
        self.performance_history.append(execution_record)
        
        # Update strategy success rates
        success = execution_record["success"]
        current_rate = self.optimization_metrics["strategy_success_rates"][strategy.value]
        self.optimization_metrics["strategy_success_rates"][strategy.value] = (
            current_rate * 0.9 + (1.0 if success else 0.0) * 0.1
        )
        
        # Update average completion time
        if self.performance_history:
            recent_durations = [e["duration"] for e in self.performance_history[-10:]]
            avg_duration = sum(recent_durations) / len(recent_durations)
            self.optimization_metrics["average_completion_time"] = timedelta(seconds=avg_duration)
    
    async def _learn_from_execution(self, plan: ExecutionPlan, result: Any, duration: timedelta):
        """Advanced learning from execution patterns."""
        success = result is not None and (not isinstance(result, dict) or result.get("success", True))
        
        if success:
            logger.info(f"Successful execution with {plan.strategy.value} strategy in {duration.total_seconds():.2f}s")
            
            # Reinforce successful patterns
            if plan.optimization_score > 0.8:
                # This was a good plan, remember the patterns
                self._reinforce_successful_patterns(plan)
        else:
            logger.warning(f"Failed execution with {plan.strategy.value} strategy after {duration.total_seconds():.2f}s")
            self.optimization_metrics["adaptation_frequency"] += 1
            
            # Learn from failures
            await self._adapt_from_failure(plan, result)
        
        # Update resource utilization metrics
        required_agents = plan.resource_requirements.get("agents_required", 1)
        max_agents = self.config.get("max_parallel_agents", 10)
        utilization = required_agents / max_agents
        
        self.optimization_metrics["resource_utilization"] = (
            self.optimization_metrics["resource_utilization"] * 0.9 + utilization * 0.1
        )
    
    def _reinforce_successful_patterns(self, plan: ExecutionPlan):
        """Reinforce patterns from successful executions."""
        # Store successful execution patterns for future use
        pattern_key = f"{plan.strategy.value}_{plan.root_task.complexity.value}_{len(plan.execution_stages)}"
        
        if not hasattr(self, "success_patterns"):
            self.success_patterns = {}
        
        if pattern_key not in self.success_patterns:
            self.success_patterns[pattern_key] = {"count": 0, "score": 0.0}
        
        self.success_patterns[pattern_key]["count"] += 1
        self.success_patterns[pattern_key]["score"] = (
            self.success_patterns[pattern_key]["score"] * 0.8 + plan.optimization_score * 0.2
        )
    
    async def _adapt_from_failure(self, plan: ExecutionPlan, result: Any):
        """Adapt strategies based on failure analysis."""
        # Analyze failure patterns and adjust future strategy selection
        failure_key = f"{plan.strategy.value}_{plan.root_task.complexity.value}"
        
        if not hasattr(self, "failure_patterns"):
            self.failure_patterns = defaultdict(int)
        
        self.failure_patterns[failure_key] += 1
        
        # If a strategy fails frequently, reduce its selection probability
        if self.failure_patterns[failure_key] > 3:
            logger.info(f"Strategy {plan.strategy.value} showing frequent failures, adjusting selection logic")
            # This could trigger strategy rebalancing in future executions
    
    def _generate_task_id(self, description: str) -> str:
        """Generate unique task ID with enhanced entropy."""
        timestamp = datetime.now().isoformat()
        hash_input = f"{description}_{timestamp}_{hash(self)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    async def get_optimization_report(self) -> str:
        """Generate comprehensive optimization report."""
        report = []
        report.append("=" * 70)
        report.append("ENHANCED META-ORCHESTRATOR OPTIMIZATION REPORT")
        report.append("=" * 70)
        report.append("")
        
        # Execution statistics
        total_executions = len(self.performance_history)
        if total_executions > 0:
            successful_executions = sum(1 for e in self.performance_history if e["success"])
            success_rate = successful_executions / total_executions * 100
            
            report.append(f"Execution Statistics:")
            report.append(f"  Total Executions: {total_executions}")
            report.append(f"  Success Rate: {success_rate:.1f}%")
            report.append(f"  Average Duration: {self.optimization_metrics['average_completion_time']}")
            report.append("")
        
        # Strategy performance
        report.append("Strategy Performance:")
        for strategy, success_rate in self.optimization_metrics["strategy_success_rates"].items():
            report.append(f"  {strategy}: {success_rate:.1%}")
        report.append("")
        
        # Agent performance
        report.append("Agent Performance:")
        for agent_id, profile in self.agent_performance.items():
            report.append(f"  {agent_id}:")
            report.append(f"    Success Rate: {profile.success_rate:.1%}")
            report.append(f"    Reliability: {profile.reliability_score:.1%}")
            report.append(f"    Capabilities: {len(profile.capabilities)}")
        report.append("")
        
        # Optimization metrics
        report.append("Optimization Metrics:")
        report.append(f"  Parallelization Ratio: {self.optimization_metrics['parallelization_ratio']:.1%}")
        report.append(f"  Resource Utilization: {self.optimization_metrics['resource_utilization']:.1%}")
        report.append(f"  One-Shot Success Rate: {self.optimization_metrics['one_shot_success_rate']:.1%}")
        report.append(f"  Adaptation Frequency: {self.optimization_metrics['adaptation_frequency']}")
        report.append("")
        
        # Active executions
        report.append(f"Active Executions: {len(self.active_executions)}")
        for task_id, plan in self.active_executions.items():
            report.append(f"  {task_id}: {plan.strategy.value} (score: {plan.optimization_score:.2f})")
        
        report.append("=" * 70)
        
        return "\n".join(report)


class ResourcePool:
    """Enhanced resource management for optimal agent coordination."""
    
    def __init__(self, max_agents: int = 10, max_memory_mb: int = 2000, max_cpu_units: int = 200):
        self.max_agents = max_agents
        self.max_memory_mb = max_memory_mb
        self.max_cpu_units = max_cpu_units
        
        self.available_agents = max_agents
        self.available_memory_mb = max_memory_mb
        self.available_cpu_units = max_cpu_units
        
        self.allocations: Dict[str, Dict[str, Any]] = {}
        self.allocation_history: List[Dict[str, Any]] = []
    
    async def allocate(self, task_id: str, requirements: Dict[str, Any]) -> bool:
        """Allocate resources with tracking."""
        required_agents = requirements.get("agents_required", 1)
        required_memory = requirements.get("memory_mb", 100)
        required_cpu = requirements.get("cpu_units", 10)
        
        # Check availability
        if (self.available_agents >= required_agents and 
            self.available_memory_mb >= required_memory and 
            self.available_cpu_units >= required_cpu):
            
            # Allocate resources
            self.available_agents -= required_agents
            self.available_memory_mb -= required_memory
            self.available_cpu_units -= required_cpu
            
            # Track allocation
            allocation = {
                "agents": required_agents,
                "memory_mb": required_memory,
                "cpu_units": required_cpu,
                "timestamp": datetime.now()
            }
            self.allocations[task_id] = allocation
            self.allocation_history.append({
                "task_id": task_id,
                "action": "allocate",
                **allocation
            })
            
            return True
        
        return False
    
    def release(self, task_id: str):
        """Release allocated resources."""
        if task_id in self.allocations:
            allocation = self.allocations[task_id]
            
            # Release resources
            self.available_agents += allocation["agents"]
            self.available_memory_mb += allocation["memory_mb"]
            self.available_cpu_units += allocation["cpu_units"]
            
            # Track release
            self.allocation_history.append({
                "task_id": task_id,
                "action": "release",
                **allocation,
                "release_timestamp": datetime.now()
            })
            
            del self.allocations[task_id]
    
    def get_utilization(self) -> Dict[str, float]:
        """Get current resource utilization percentages."""
        return {
            "agents": (self.max_agents - self.available_agents) / self.max_agents,
            "memory": (self.max_memory_mb - self.available_memory_mb) / self.max_memory_mb,
            "cpu": (self.max_cpu_units - self.available_cpu_units) / self.max_cpu_units
        }


# Usage example and testing
async def test_enhanced_meta_orchestrator():
    """Test the enhanced meta-orchestrator with complex scenarios."""
    orchestrator = EnhancedMetaOrchestrator()
    
    # Complex multi-domain request
    complex_request = """
    Build a comprehensive AI-powered financial analysis platform that includes:
    
    1. Multi-format document processing (PDF, CSV, JSON, Excel)
    2. Advanced ML models for fraud detection and risk assessment
    3. Real-time data streaming from multiple financial APIs
    4. Interactive dashboards with predictive analytics
    5. Automated report generation and compliance checking
    6. Integration with major accounting systems (QuickBooks, Xero, Sage)
    7. Multi-tenant architecture with role-based access control
    8. Comprehensive audit logging and data lineage tracking
    9. Advanced search and filtering capabilities
    10. Mobile-responsive design with offline capabilities
    """
    
    context = {
        "priority": "high",
        "complexity": "extreme",
        "deadline": datetime.now() + timedelta(days=7),
        "frameworks": ["fastapi", "react", "tensorflow", "apache-kafka", "postgresql"],
        "integrations": ["quickbooks", "xero", "sage", "stripe", "plaid"],
        "compliance": ["sox", "gdpr", "pci-dss"],
        "performance_requirements": {
            "max_response_time": "200ms",
            "concurrent_users": 10000,
            "uptime": "99.9%"
        }
    }
    
    print("🚀 Testing Enhanced Meta-Orchestrator")
    print("=" * 60)
    print(f"Request: {complex_request[:100]}...")
    print(f"Context: {len(context)} parameters")
    print("")
    
    start_time = datetime.now()
    
    try:
        # Process with meta-optimal orchestration
        result = await orchestrator.process_request(complex_request, context)
        
        execution_time = datetime.now() - start_time
        
        print("✅ Execution completed successfully!")
        print(f"⏱️  Total time: {execution_time.total_seconds():.2f} seconds")
        print("")
        
        if isinstance(result, dict):
            print("📊 Results:")
            print(f"  Strategy: {result.get('execution_strategy', 'Unknown')}")
            print(f"  Optimization Score: {result.get('optimization_score', 0):.2f}")
            print(f"  Success Rate: {result.get('success_rate', 'N/A')}")
            print(f"  Total Tasks: {result.get('total_tasks', 0)}")
        
        print("")
        print("📈 Optimization Metrics:")
        for metric, value in orchestrator.optimization_metrics.items():
            if isinstance(value, timedelta):
                print(f"  {metric}: {value.total_seconds():.2f}s")
            elif isinstance(value, (int, float)):
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: {value}")
        
        # Generate detailed report
        print("\n" + "=" * 60)
        optimization_report = await orchestrator.get_optimization_report()
        print(optimization_report)
        
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the comprehensive test
    asyncio.run(test_enhanced_meta_orchestrator())