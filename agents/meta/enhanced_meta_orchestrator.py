"""
Enhanced Meta-Orchestrator: Phase 6 - Self-Improving Agent Ecosystem
Builds on existing orchestrator with advanced meta-capabilities:
- Self-improvement through strategy learning
- Dynamic agent spawning based on task analysis
- Performance tracking and optimization
- Multi-model fallback chains (Claude → GPT → Local)
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict, deque
import hashlib
from abc import ABC, abstractmethod

# Import existing orchestrator components
from .meta_orchestrator import MetaOrchestrator, DevelopmentTask, TaskPriority, AgentRole
from core.orchestration.orchestrator import AgentOrchestrator, Task
from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ModelProvider(Enum):
    """Available AI model providers for fallback chain"""
    CLAUDE = "claude"
    GPT = "gpt"
    LOCAL = "local"
    GEMINI = "gemini"


class StrategyType(Enum):
    """Types of execution strategies"""
    HIERARCHICAL = "hierarchical"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    COLLABORATIVE = "collaborative"
    CONSENSUS = "consensus"
    SWARM = "swarm"
    HYBRID = "hybrid"


@dataclass
class PerformanceMetrics:
    """Performance tracking for strategies and agents"""
    success_rate: float = 0.0
    average_time: float = 0.0
    cost_efficiency: float = 0.0
    quality_score: float = 0.0
    total_executions: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, success: bool, time_taken: float, cost: float, quality: float):
        """Update metrics with new execution data"""
        self.total_executions += 1
        
        # Calculate running averages
        alpha = 0.1  # Learning rate
        self.success_rate = (1 - alpha) * self.success_rate + alpha * (1.0 if success else 0.0)
        self.average_time = (1 - alpha) * self.average_time + alpha * time_taken
        self.cost_efficiency = (1 - alpha) * self.cost_efficiency + alpha * (1.0 / cost if cost > 0 else 1.0)
        self.quality_score = (1 - alpha) * self.quality_score + alpha * quality
        self.last_updated = datetime.now()


@dataclass
class Strategy:
    """Execution strategy with learned parameters"""
    name: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def matches_task(self, task_analysis: Dict[str, Any]) -> float:
        """Calculate match score for this strategy given task analysis"""
        score = 0.0
        
        # Check complexity match
        if "complexity" in self.conditions and "complexity" in task_analysis:
            complexity_diff = abs(self.conditions["complexity"] - task_analysis["complexity"])
            score += max(0, 1.0 - complexity_diff / 5.0) * 0.3
        
        # Check domain match
        if "domain" in self.conditions and "domain" in task_analysis:
            if self.conditions["domain"] == task_analysis["domain"]:
                score += 0.3
        
        # Check parallelizability match
        if "parallel" in self.conditions and "parallel" in task_analysis:
            if self.conditions["parallel"] == task_analysis["parallel"]:
                score += 0.2
        
        # Performance bonus
        score += self.metrics.success_rate * 0.2
        
        return min(score, 1.0)


class TaskAnalyzer:
    """Analyzes tasks to determine optimal execution strategies"""
    
    def __init__(self):
        self.analysis_history: List[Tuple[str, Dict[str, Any]]] = []
        self.patterns: Dict[str, Any] = {}
    
    async def analyze_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze task characteristics for strategy selection"""
        analysis = {
            "task_id": hashlib.md5(task_description.encode()).hexdigest()[:8],
            "timestamp": datetime.now(),
            "complexity": await self._assess_complexity(task_description),
            "domain": await self._identify_domain(task_description),
            "parallel": await self._assess_parallelizability(task_description),
            "dependencies": await self._identify_dependencies(task_description, context),
            "required_skills": await self._identify_required_skills(task_description),
            "estimated_time": await self._estimate_time(task_description),
            "priority": await self._assess_priority(task_description, context)
        }
        
        self.analysis_history.append((task_description, analysis))
        await self._update_patterns()
        
        return analysis
    
    async def _assess_complexity(self, task_description: str) -> int:
        """Assess task complexity on scale 1-10"""
        complexity_indicators = [
            ("multiple agents", 3),
            ("integration", 2),
            ("machine learning", 3),
            ("database", 2),
            ("api", 2),
            ("testing", 1),
            ("optimization", 3),
            ("security", 2),
            ("deploy", 2),
            ("scale", 3)
        ]
        
        complexity = 1  # Base complexity
        task_lower = task_description.lower()
        
        for indicator, weight in complexity_indicators:
            if indicator in task_lower:
                complexity += weight
        
        return min(complexity, 10)
    
    async def _identify_domain(self, task_description: str) -> str:
        """Identify the primary domain of the task"""
        domain_keywords = {
            "financial": ["invoice", "accounting", "finance", "payment", "budget"],
            "data_analysis": ["data", "analysis", "visualization", "report", "metric"],
            "development": ["code", "implement", "program", "develop", "build"],
            "testing": ["test", "verify", "validate", "quality", "bug"],
            "integration": ["integrate", "connect", "api", "system", "service"],
            "optimization": ["optimize", "improve", "performance", "efficiency"],
            "security": ["security", "auth", "encrypt", "protect", "secure"]
        }
        
        task_lower = task_description.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in task_lower)
            if score > 0:
                domain_scores[domain] = score
        
        return max(domain_scores, key=domain_scores.get) if domain_scores else "general"
    
    async def _assess_parallelizability(self, task_description: str) -> bool:
        """Assess if task can be parallelized"""
        parallel_indicators = ["parallel", "concurrent", "batch", "multiple", "independent"]
        sequential_indicators = ["sequential", "order", "depend", "chain", "pipeline"]
        
        task_lower = task_description.lower()
        parallel_score = sum(1 for indicator in parallel_indicators if indicator in task_lower)
        sequential_score = sum(1 for indicator in sequential_indicators if indicator in task_lower)
        
        return parallel_score > sequential_score
    
    async def _identify_dependencies(self, task_description: str, context: Dict[str, Any]) -> List[str]:
        """Identify task dependencies"""
        dependencies = []
        
        if context and "dependencies" in context:
            dependencies.extend(context["dependencies"])
        
        # Analyze description for implicit dependencies
        if "after" in task_description.lower():
            dependencies.append("sequential_dependency")
        if "requires" in task_description.lower():
            dependencies.append("prerequisite_dependency")
        
        return dependencies
    
    async def _identify_required_skills(self, task_description: str) -> List[str]:
        """Identify skills required for the task"""
        skill_keywords = {
            "programming": ["code", "implement", "develop", "program"],
            "analysis": ["analyze", "investigate", "research", "study"],
            "design": ["design", "architecture", "structure", "model"],
            "testing": ["test", "verify", "validate", "check"],
            "integration": ["integrate", "connect", "combine", "merge"],
            "optimization": ["optimize", "improve", "enhance", "tune"],
            "documentation": ["document", "write", "explain", "describe"]
        }
        
        task_lower = task_description.lower()
        required_skills = []
        
        for skill, keywords in skill_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                required_skills.append(skill)
        
        return required_skills or ["general"]
    
    async def _estimate_time(self, task_description: str) -> float:
        """Estimate task execution time in minutes"""
        base_time = 10  # Base 10 minutes
        
        # Time multipliers based on task characteristics
        time_multipliers = {
            "simple": 0.5,
            "complex": 2.0,
            "integrate": 1.5,
            "test": 1.2,
            "deploy": 1.8,
            "optimize": 2.5,
            "debug": 1.5
        }
        
        task_lower = task_description.lower()
        multiplier = 1.0
        
        for keyword, mult in time_multipliers.items():
            if keyword in task_lower:
                multiplier = max(multiplier, mult)
        
        return base_time * multiplier
    
    async def _assess_priority(self, task_description: str, context: Dict[str, Any]) -> TaskPriority:
        """Assess task priority"""
        if context and "priority" in context:
            return TaskPriority[context["priority"].upper()]
        
        # Assess based on keywords
        priority_keywords = {
            TaskPriority.CRITICAL: ["critical", "urgent", "emergency", "blocking"],
            TaskPriority.HIGH: ["important", "high", "priority", "asap"],
            TaskPriority.MEDIUM: ["normal", "standard", "regular"],
            TaskPriority.LOW: ["low", "minor", "nice-to-have"],
            TaskPriority.BACKGROUND: ["background", "cleanup", "maintenance"]
        }
        
        task_lower = task_description.lower()
        
        for priority, keywords in priority_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                return priority
        
        return TaskPriority.MEDIUM
    
    async def _update_patterns(self):
        """Update learned patterns from analysis history"""
        if len(self.analysis_history) < 10:  # Need minimum data
            return
        
        # Analyze patterns in the last 100 analyses
        recent_analyses = self.analysis_history[-100:]
        
        # Domain frequency patterns
        domain_counts = defaultdict(int)
        complexity_by_domain = defaultdict(list)
        
        for _, analysis in recent_analyses:
            domain = analysis["domain"]
            complexity = analysis["complexity"]
            
            domain_counts[domain] += 1
            complexity_by_domain[domain].append(complexity)
        
        self.patterns = {
            "domain_frequency": dict(domain_counts),
            "average_complexity_by_domain": {
                domain: np.mean(complexities)
                for domain, complexities in complexity_by_domain.items()
            },
            "last_updated": datetime.now()
        }


class StrategyLearner:
    """Learns and optimizes execution strategies based on performance"""
    
    def __init__(self):
        self.strategies: Dict[str, Strategy] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.learning_rate = 0.1
        self._initialize_base_strategies()
    
    def _initialize_base_strategies(self):
        """Initialize base strategies with default parameters"""
        base_strategies = [
            Strategy(
                name="hierarchical_decomposition",
                strategy_type=StrategyType.HIERARCHICAL,
                parameters={"max_depth": 3, "min_subtask_size": 1},
                conditions={"complexity": 7, "parallel": False}
            ),
            Strategy(
                name="parallel_execution",
                strategy_type=StrategyType.PARALLEL,
                parameters={"max_agents": 5, "timeout": 300},
                conditions={"parallel": True, "complexity": 5}
            ),
            Strategy(
                name="sequential_pipeline",
                strategy_type=StrategyType.SEQUENTIAL,
                parameters={"max_stages": 5, "handoff_validation": True},
                conditions={"parallel": False, "complexity": 4}
            ),
            Strategy(
                name="collaborative_discussion",
                strategy_type=StrategyType.COLLABORATIVE,
                parameters={"max_rounds": 5, "consensus_threshold": 0.8},
                conditions={"complexity": 6, "domain": "general"}
            ),
            Strategy(
                name="consensus_voting",
                strategy_type=StrategyType.CONSENSUS,
                parameters={"min_agreements": 3, "voting_rounds": 3},
                conditions={"complexity": 5, "parallel": True}
            ),
            Strategy(
                name="swarm_optimization",
                strategy_type=StrategyType.SWARM,
                parameters={"swarm_size": 10, "max_iterations": 20},
                conditions={"complexity": 8, "domain": "optimization"}
            )
        ]
        
        for strategy in base_strategies:
            self.strategies[strategy.name] = strategy
    
    async def select_strategy(self, task_analysis: Dict[str, Any]) -> Strategy:
        """Select optimal strategy based on task analysis and learned performance"""
        strategy_scores = {}
        
        for name, strategy in self.strategies.items():
            # Calculate match score
            match_score = strategy.matches_task(task_analysis)
            
            # Weight by performance metrics
            performance_weight = (
                strategy.metrics.success_rate * 0.4 +
                strategy.metrics.quality_score * 0.3 +
                strategy.metrics.cost_efficiency * 0.3
            )
            
            # Apply exploration bonus for less-tried strategies
            exploration_bonus = 0.1 if strategy.metrics.total_executions < 5 else 0.0
            
            strategy_scores[name] = match_score * performance_weight + exploration_bonus
        
        # Select best strategy
        best_strategy_name = max(strategy_scores, key=strategy_scores.get)
        selected_strategy = self.strategies[best_strategy_name]
        
        logger.info(f"Selected strategy: {best_strategy_name} (score: {strategy_scores[best_strategy_name]:.3f})")
        return selected_strategy
    
    async def update_strategy_performance(self, strategy_name: str, 
                                        success: bool, time_taken: float, 
                                        cost: float, quality: float):
        """Update strategy performance metrics"""
        if strategy_name in self.strategies:
            self.strategies[strategy_name].metrics.update(success, time_taken, cost, quality)
            
            # Record execution for learning
            execution_record = {
                "strategy": strategy_name,
                "timestamp": datetime.now(),
                "success": success,
                "time_taken": time_taken,
                "cost": cost,
                "quality": quality
            }
            
            self.execution_history.append(execution_record)
            
            # Learn new strategies if needed
            await self._learn_new_strategies()
    
    async def _learn_new_strategies(self):
        """Learn new strategies by combining successful ones"""
        if len(self.execution_history) < 50:  # Need minimum data
            return
        
        # Analyze recent successful executions
        recent_successes = [
            record for record in self.execution_history[-50:]
            if record["success"] and record["quality"] > 0.7
        ]
        
        if len(recent_successes) < 10:
            return
        
        # Find patterns in successful strategies
        successful_strategies = defaultdict(list)
        for record in recent_successes:
            successful_strategies[record["strategy"]].append(record)
        
        # Create hybrid strategies from successful ones
        for strategy_name, records in successful_strategies.items():
            if len(records) >= 5 and strategy_name in self.strategies:
                await self._create_hybrid_strategy(strategy_name, records)
    
    async def _create_hybrid_strategy(self, base_strategy_name: str, successful_records: List[Dict]):
        """Create hybrid strategy based on successful patterns"""
        base_strategy = self.strategies[base_strategy_name]
        
        # Calculate optimal parameters from successful executions
        avg_time = np.mean([r["time_taken"] for r in successful_records])
        avg_quality = np.mean([r["quality"] for r in successful_records])
        
        # Create hybrid strategy name
        hybrid_name = f"hybrid_{base_strategy_name}_{len(self.strategies)}"
        
        if hybrid_name not in self.strategies:
            hybrid_strategy = Strategy(
                name=hybrid_name,
                strategy_type=StrategyType.HYBRID,
                parameters=base_strategy.parameters.copy(),
                conditions=base_strategy.conditions.copy()
            )
            
            # Initialize with learned performance
            hybrid_strategy.metrics.success_rate = 0.8  # Start optimistic
            hybrid_strategy.metrics.average_time = avg_time
            hybrid_strategy.metrics.quality_score = avg_quality
            
            self.strategies[hybrid_name] = hybrid_strategy
            logger.info(f"Created hybrid strategy: {hybrid_name}")


class ModelFallbackChain:
    """Manages fallback chain across different AI models"""
    
    def __init__(self):
        self.providers = {
            ModelProvider.CLAUDE: {
                "available": True,
                "cost_per_token": 0.001,
                "quality_score": 0.95,
                "rate_limit": 1000,  # requests per hour
                "current_usage": 0
            },
            ModelProvider.GPT: {
                "available": True,
                "cost_per_token": 0.002,
                "quality_score": 0.90,
                "rate_limit": 500,
                "current_usage": 0
            },
            ModelProvider.LOCAL: {
                "available": True,
                "cost_per_token": 0.0,
                "quality_score": 0.70,
                "rate_limit": float('inf'),
                "current_usage": 0
            }
        }
        self.circuit_breakers = {}
        self.last_reset = datetime.now()
    
    async def execute_with_fallback(self, task: str, context: Dict[str, Any]) -> Tuple[Any, ModelProvider, float]:
        """Execute task with fallback chain"""
        providers = self._get_available_providers()
        
        for provider in providers:
            try:
                if await self._check_circuit_breaker(provider):
                    result, cost = await self._execute_with_provider(provider, task, context)
                    self._update_usage(provider)
                    return result, provider, cost
            except Exception as e:
                logger.warning(f"Provider {provider} failed: {e}")
                await self._trip_circuit_breaker(provider)
                continue
        
        raise Exception("All providers failed")
    
    def _get_available_providers(self) -> List[ModelProvider]:
        """Get providers sorted by preference (cost/quality balance)"""
        available = [
            provider for provider, config in self.providers.items()
            if config["available"] and config["current_usage"] < config["rate_limit"]
        ]
        
        # Sort by quality/cost ratio
        return sorted(available, key=lambda p: (
            self.providers[p]["quality_score"] / (self.providers[p]["cost_per_token"] + 0.001)
        ), reverse=True)
    
    async def _check_circuit_breaker(self, provider: ModelProvider) -> bool:
        """Check if circuit breaker allows provider usage"""
        if provider not in self.circuit_breakers:
            return True
        
        breaker = self.circuit_breakers[provider]
        
        # Reset circuit breaker after timeout
        if datetime.now() - breaker["last_failure"] > timedelta(minutes=5):
            del self.circuit_breakers[provider]
            return True
        
        return breaker["failure_count"] < 3
    
    async def _trip_circuit_breaker(self, provider: ModelProvider):
        """Trip circuit breaker for provider"""
        if provider not in self.circuit_breakers:
            self.circuit_breakers[provider] = {
                "failure_count": 0,
                "last_failure": datetime.now()
            }
        
        self.circuit_breakers[provider]["failure_count"] += 1
        self.circuit_breakers[provider]["last_failure"] = datetime.now()
    
    async def _execute_with_provider(self, provider: ModelProvider, task: str, context: Dict[str, Any]) -> Tuple[Any, float]:
        """Execute task with specific provider"""
        # Simulate model execution - replace with actual implementations
        if provider == ModelProvider.CLAUDE:
            result = await self._execute_claude(task, context)
        elif provider == ModelProvider.GPT:
            result = await self._execute_gpt(task, context)
        elif provider == ModelProvider.LOCAL:
            result = await self._execute_local(task, context)
        else:
            raise Exception(f"Unknown provider: {provider}")
        
        # Estimate cost (simplified)
        estimated_tokens = len(task.split()) + len(str(context).split())
        cost = estimated_tokens * self.providers[provider]["cost_per_token"]
        
        return result, cost
    
    async def _execute_claude(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute with Claude API"""
        # Placeholder - implement actual Claude API call
        await asyncio.sleep(0.1)  # Simulate API call
        return {
            "result": f"Claude result for: {task[:50]}...",
            "provider": "claude",
            "quality": 0.95
        }
    
    async def _execute_gpt(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute with GPT API"""
        # Placeholder - implement actual GPT API call
        await asyncio.sleep(0.15)  # Simulate API call
        return {
            "result": f"GPT result for: {task[:50]}...",
            "provider": "gpt",
            "quality": 0.90
        }
    
    async def _execute_local(self, task: str, context: Dict[str, Any]) -> Any:
        """Execute with local model"""
        # Placeholder - implement local model execution
        await asyncio.sleep(0.5)  # Simulate local processing
        return {
            "result": f"Local result for: {task[:50]}...",
            "provider": "local",
            "quality": 0.70
        }
    
    def _update_usage(self, provider: ModelProvider):
        """Update provider usage statistics"""
        self.providers[provider]["current_usage"] += 1
        
        # Reset usage counters hourly
        if datetime.now() - self.last_reset > timedelta(hours=1):
            for p in self.providers:
                self.providers[p]["current_usage"] = 0
            self.last_reset = datetime.now()


class EnhancedMetaOrchestrator(MetaOrchestrator):
    """Enhanced meta-orchestrator with self-improvement capabilities"""
    
    def __init__(self, config_path: Optional[Path] = None):
        super().__init__(config_path)
        
        # Enhanced components
        self.task_analyzer = TaskAnalyzer()
        self.strategy_learner = StrategyLearner()
        self.model_fallback = ModelFallbackChain()
        self.agent_orchestrator = AgentOrchestrator("enhanced_orchestrator")
        
        # Performance tracking
        self.execution_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        self.learning_insights: List[Dict[str, Any]] = []
        
        # Dynamic agent spawning
        self.spawned_agents: Dict[str, BaseAgent] = {}
        self.agent_creation_history: List[Dict[str, Any]] = []
        
        logger.info("Enhanced Meta-Orchestrator initialized with self-improvement capabilities")
    
    async def process_request(self, request: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process request with enhanced capabilities"""
        start_time = datetime.now()
        context = context or {}
        
        try:
            # Analyze task
            task_analysis = await self.task_analyzer.analyze_task(request, context)
            logger.info(f"Task analysis completed: complexity={task_analysis['complexity']}, domain={task_analysis['domain']}")
            
            # Select optimal strategy
            strategy = await self.strategy_learner.select_strategy(task_analysis)
            
            # Spawn specialized agents if needed
            specialized_agents = await self._spawn_specialized_agents(task_analysis)
            
            # Execute with fallback chain
            result, provider, cost = await self.model_fallback.execute_with_fallback(
                request, {**context, "strategy": strategy.name, "analysis": task_analysis}
            )
            
            # Execute with selected strategy
            execution_result = await self._execute_with_strategy(strategy, request, task_analysis, specialized_agents)
            
            # Calculate performance metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            quality_score = await self._assess_quality(execution_result, task_analysis)
            
            # Update strategy performance
            await self.strategy_learner.update_strategy_performance(
                strategy.name, 
                success=True,
                time_taken=execution_time,
                cost=cost,
                quality=quality_score
            )
            
            # Learn from execution
            await self._learn_from_execution(request, task_analysis, strategy, execution_result)
            
            return {
                "success": True,
                "result": execution_result,
                "strategy_used": strategy.name,
                "provider_used": provider.value,
                "execution_time": execution_time,
                "cost": cost,
                "quality_score": quality_score,
                "spawned_agents": len(specialized_agents),
                "insights": self.learning_insights[-5:]  # Latest insights
            }
            
        except Exception as e:
            # Handle failure and learn from it
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if 'strategy' in locals():
                await self.strategy_learner.update_strategy_performance(
                    strategy.name,
                    success=False,
                    time_taken=execution_time,
                    cost=0,
                    quality=0
                )
            
            logger.error(f"Enhanced orchestrator execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "fallback_result": await self._fallback_simple_execution(request)
            }
    
    async def _spawn_specialized_agents(self, task_analysis: Dict[str, Any]) -> List[BaseAgent]:
        """Dynamically spawn specialized agents based on task analysis"""
        required_skills = task_analysis.get("required_skills", [])
        domain = task_analysis.get("domain", "general")
        complexity = task_analysis.get("complexity", 5)
        
        spawned_agents = []
        
        # Determine what agents to spawn
        agents_to_spawn = []
        
        if "programming" in required_skills:
            agents_to_spawn.append(("code_specialist", "CodeSpecialistAgent"))
        
        if "analysis" in required_skills or domain == "data_analysis":
            agents_to_spawn.append(("data_analyst", "DataAnalystAgent"))
        
        if "testing" in required_skills:
            agents_to_spawn.append(("test_specialist", "TestSpecialistAgent"))
        
        if domain == "financial":
            agents_to_spawn.append(("financial_specialist", "FinancialSpecialistAgent"))
        
        if complexity >= 8:
            agents_to_spawn.append(("complexity_manager", "ComplexityManagerAgent"))
        
        # Spawn agents
        for agent_id, agent_type in agents_to_spawn:
            if agent_id not in self.spawned_agents:
                agent = await self._create_specialized_agent(agent_id, agent_type, task_analysis)
                self.spawned_agents[agent_id] = agent
                self.agent_orchestrator.register_agent(agent)
                spawned_agents.append(agent)
                
                # Record creation
                self.agent_creation_history.append({
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "created_for_task": task_analysis["task_id"],
                    "timestamp": datetime.now(),
                    "task_domain": domain,
                    "task_complexity": complexity
                })
        
        return spawned_agents
    
    async def _create_specialized_agent(self, agent_id: str, agent_type: str, task_analysis: Dict[str, Any]) -> BaseAgent:
        """Create a specialized agent for specific tasks"""
        # Create agent with specialized capabilities
        agent = BaseAgent(agent_id)
        
        # Configure agent based on type
        if agent_type == "CodeSpecialistAgent":
            agent.capabilities = ["programming", "code_review", "debugging", "optimization"]
        elif agent_type == "DataAnalystAgent":
            agent.capabilities = ["data_analysis", "visualization", "statistics", "reporting"]
        elif agent_type == "TestSpecialistAgent":
            agent.capabilities = ["testing", "quality_assurance", "validation", "verification"]
        elif agent_type == "FinancialSpecialistAgent":
            agent.capabilities = ["financial_analysis", "accounting", "invoice_processing", "compliance"]
        elif agent_type == "ComplexityManagerAgent":
            agent.capabilities = ["project_management", "task_decomposition", "coordination", "optimization"]
        else:
            agent.capabilities = ["general"]
        
        logger.info(f"Created specialized agent: {agent_id} ({agent_type})")
        return agent
    
    async def _execute_with_strategy(self, strategy: Strategy, request: str, 
                                   task_analysis: Dict[str, Any], 
                                   specialized_agents: List[BaseAgent]) -> Any:
        """Execute request using selected strategy"""
        if strategy.strategy_type == StrategyType.HIERARCHICAL:
            return await self._execute_hierarchical(request, task_analysis, specialized_agents)
        elif strategy.strategy_type == StrategyType.PARALLEL:
            return await self._execute_parallel(request, task_analysis, specialized_agents)
        elif strategy.strategy_type == StrategyType.SEQUENTIAL:
            return await self._execute_sequential(request, task_analysis, specialized_agents)
        elif strategy.strategy_type == StrategyType.COLLABORATIVE:
            return await self._execute_collaborative(request, task_analysis, specialized_agents)
        elif strategy.strategy_type == StrategyType.CONSENSUS:
            return await self._execute_consensus(request, task_analysis, specialized_agents)
        elif strategy.strategy_type == StrategyType.SWARM:
            return await self._execute_swarm(request, task_analysis, specialized_agents)
        elif strategy.strategy_type == StrategyType.HYBRID:
            return await self._execute_hybrid(request, task_analysis, specialized_agents)
        else:
            # Default to collaborative execution
            return await self._execute_collaborative(request, task_analysis, specialized_agents)
    
    async def _execute_hierarchical(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using hierarchical strategy"""
        # Create task for orchestrator
        task = Task(
            id=task_analysis["task_id"],
            description=request,
            requirements=task_analysis
        )
        
        return await self.agent_orchestrator.hierarchical_delegation(task)
    
    async def _execute_parallel(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using parallel strategy"""
        if not agents:
            agents = list(self.agent_orchestrator.agents.values())[:3]  # Use available agents
        
        task = Task(
            id=task_analysis["task_id"],
            description=request,
            requirements=task_analysis
        )
        
        return await self.agent_orchestrator.parallel_execution(agents, task)
    
    async def _execute_sequential(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using sequential strategy"""
        if not agents:
            agents = list(self.agent_orchestrator.agents.values())[:3]
        
        task = Task(
            id=task_analysis["task_id"],
            description=request,
            requirements=task_analysis
        )
        
        return await self.agent_orchestrator.sequential_execution(agents, task)
    
    async def _execute_collaborative(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using collaborative strategy"""
        if not agents:
            agents = list(self.agent_orchestrator.agents.values())[:3]
        
        task = Task(
            id=task_analysis["task_id"],
            description=request,
            requirements=task_analysis
        )
        
        return await self.agent_orchestrator.collaborative_execution(agents, task)
    
    async def _execute_consensus(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using consensus strategy"""
        if not agents:
            agents = list(self.agent_orchestrator.agents.values())[:5]
        
        task = Task(
            id=task_analysis["task_id"],
            description=request,
            requirements=task_analysis
        )
        
        return await self.agent_orchestrator.consensus_execution(agents, task)
    
    async def _execute_swarm(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using swarm intelligence strategy"""
        return await self.agent_orchestrator.swarm_intelligence(request, swarm_size=10)
    
    async def _execute_hybrid(self, request: str, task_analysis: Dict[str, Any], agents: List[BaseAgent]) -> Any:
        """Execute using hybrid strategy"""
        # Combine multiple strategies based on task characteristics
        complexity = task_analysis.get("complexity", 5)
        
        if complexity >= 8:
            # Use hierarchical + parallel for complex tasks
            hierarchical_result = await self._execute_hierarchical(request, task_analysis, agents)
            parallel_result = await self._execute_parallel(request, task_analysis, agents)
            
            return {
                "hybrid_approach": "hierarchical_parallel",
                "hierarchical_result": hierarchical_result,
                "parallel_result": parallel_result,
                "synthesis": "Combined hierarchical and parallel approaches"
            }
        else:
            # Use collaborative + consensus for simpler tasks
            collaborative_result = await self._execute_collaborative(request, task_analysis, agents)
            return {
                "hybrid_approach": "collaborative_enhanced",
                "result": collaborative_result
            }
    
    async def _assess_quality(self, result: Any, task_analysis: Dict[str, Any]) -> float:
        """Assess quality of execution result"""
        quality_factors = {
            "completeness": 0.3,
            "accuracy": 0.3,
            "efficiency": 0.2,
            "innovation": 0.2
        }
        
        quality_score = 0.0
        
        # Assess completeness
        if result and isinstance(result, dict):
            if "result" in result or "synthesis" in result:
                quality_score += quality_factors["completeness"]
        
        # Assess accuracy (simplified - would need domain-specific evaluation)
        if result and not isinstance(result, Exception):
            quality_score += quality_factors["accuracy"]
        
        # Efficiency bonus for fast execution
        estimated_time = task_analysis.get("estimated_time", 10)
        if estimated_time > 0:
            efficiency_bonus = min(estimated_time / 20.0, 1.0) * quality_factors["efficiency"]
            quality_score += efficiency_bonus
        
        # Innovation bonus for using specialized agents
        if isinstance(result, dict) and "spawned_agents" in str(result):
            quality_score += quality_factors["innovation"]
        
        return min(quality_score, 1.0)
    
    async def _learn_from_execution(self, request: str, task_analysis: Dict[str, Any], 
                                  strategy: Strategy, result: Any):
        """Learn insights from execution for future improvements"""
        insight = {
            "timestamp": datetime.now(),
            "task_domain": task_analysis.get("domain"),
            "task_complexity": task_analysis.get("complexity"),
            "strategy_used": strategy.name,
            "success": result is not None and not isinstance(result, Exception),
            "patterns": []
        }
        
        # Extract patterns
        if task_analysis.get("complexity", 0) > 7 and strategy.strategy_type == StrategyType.HIERARCHICAL:
            insight["patterns"].append("complex_tasks_benefit_from_hierarchical_approach")
        
        if task_analysis.get("parallel") and strategy.strategy_type == StrategyType.PARALLEL:
            insight["patterns"].append("parallel_tasks_match_parallel_strategy")
        
        # Store insight
        self.learning_insights.append(insight)
        
        # Keep only recent insights (last 100)
        if len(self.learning_insights) > 100:
            self.learning_insights = self.learning_insights[-100:]
    
    async def _fallback_simple_execution(self, request: str) -> Any:
        """Simple fallback execution when main processing fails"""
        try:
            # Use basic agent processing
            if self.agent_orchestrator.agents:
                agent = list(self.agent_orchestrator.agents.values())[0]
                return await agent.process_task(request, {})
            else:
                return {"fallback_result": f"Simple processing of: {request[:100]}..."}
        except Exception as e:
            return {"error": f"Fallback also failed: {e}"}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "orchestrator_status": await self.generate_report(),
            "strategy_performance": {
                name: {
                    "success_rate": strategy.metrics.success_rate,
                    "avg_time": strategy.metrics.average_time,
                    "quality_score": strategy.metrics.quality_score,
                    "executions": strategy.metrics.total_executions
                }
                for name, strategy in self.strategy_learner.strategies.items()
            },
            "spawned_agents": {
                agent_id: {
                    "capabilities": getattr(agent, 'capabilities', []),
                    "total_tasks": getattr(agent, 'total_tasks', 0),
                    "success_rate": getattr(agent, 'get_success_rate', lambda: 0.0)()
                }
                for agent_id, agent in self.spawned_agents.items()
            },
            "model_providers": {
                provider.value: config
                for provider, config in self.model_fallback.providers.items()
            },
            "learning_insights_count": len(self.learning_insights),
            "task_analysis_patterns": self.task_analyzer.patterns
        }
    
    async def optimize_performance(self) -> Dict[str, Any]:
        """Perform self-optimization of the system"""
        optimizations = []
        
        # Optimize strategy selection
        strategy_optimization = await self._optimize_strategies()
        optimizations.append(strategy_optimization)
        
        # Optimize agent spawning
        agent_optimization = await self._optimize_agent_spawning()
        optimizations.append(agent_optimization)
        
        # Optimize model provider usage
        provider_optimization = await self._optimize_provider_usage()
        optimizations.append(provider_optimization)
        
        return {
            "optimizations_applied": optimizations,
            "timestamp": datetime.now(),
            "next_optimization_due": datetime.now() + timedelta(hours=24)
        }
    
    async def _optimize_strategies(self) -> Dict[str, Any]:
        """Optimize strategy selection based on performance data"""
        low_performing_strategies = [
            name for name, strategy in self.strategy_learner.strategies.items()
            if strategy.metrics.success_rate < 0.6 and strategy.metrics.total_executions > 10
        ]
        
        high_performing_strategies = [
            name for name, strategy in self.strategy_learner.strategies.items()
            if strategy.metrics.success_rate > 0.8 and strategy.metrics.total_executions > 5
        ]
        
        # Adjust strategy parameters for low performers
        adjustments = []
        for strategy_name in low_performing_strategies:
            strategy = self.strategy_learner.strategies[strategy_name]
            # Increase timeout parameters
            if "timeout" in strategy.parameters:
                strategy.parameters["timeout"] = int(strategy.parameters["timeout"] * 1.2)
                adjustments.append(f"Increased timeout for {strategy_name}")
        
        return {
            "optimization_type": "strategy_parameters",
            "low_performers": low_performing_strategies,
            "high_performers": high_performing_strategies,
            "adjustments_made": adjustments
        }
    
    async def _optimize_agent_spawning(self) -> Dict[str, Any]:
        """Optimize agent spawning patterns"""
        # Analyze agent creation patterns
        domain_agent_success = defaultdict(list)
        
        for record in self.agent_creation_history:
            agent_id = record["agent_id"]
            if agent_id in self.spawned_agents:
                agent = self.spawned_agents[agent_id]
                success_rate = getattr(agent, 'get_success_rate', lambda: 0.0)()
                domain_agent_success[record["task_domain"]].append(success_rate)
        
        # Find optimal agent types per domain
        domain_recommendations = {}
        for domain, success_rates in domain_agent_success.items():
            if success_rates:
                avg_success = np.mean(success_rates)
                domain_recommendations[domain] = {
                    "avg_success_rate": avg_success,
                    "recommendation": "continue" if avg_success > 0.7 else "review"
                }
        
        return {
            "optimization_type": "agent_spawning",
            "domain_recommendations": domain_recommendations,
            "total_agents_spawned": len(self.spawned_agents)
        }
    
    async def _optimize_provider_usage(self) -> Dict[str, Any]:
        """Optimize AI model provider usage for cost efficiency"""
        provider_adjustments = []
        
        # Check rate limits and adjust preferences
        for provider, config in self.model_fallback.providers.items():
            usage_ratio = config["current_usage"] / config["rate_limit"]
            
            if usage_ratio > 0.8:  # Near rate limit
                # Reduce preference for this provider
                config["quality_score"] *= 0.9
                provider_adjustments.append(f"Reduced preference for {provider.value} due to high usage")
            elif usage_ratio < 0.2:  # Underutilized
                # Increase preference
                config["quality_score"] *= 1.1
                config["quality_score"] = min(config["quality_score"], 1.0)
                provider_adjustments.append(f"Increased preference for {provider.value} due to low usage")
        
        return {
            "optimization_type": "provider_usage",
            "adjustments_made": provider_adjustments,
            "current_usage_ratios": {
                provider.value: config["current_usage"] / config["rate_limit"]
                for provider, config in self.model_fallback.providers.items()
            }
        }


if __name__ == "__main__":
    async def demo_enhanced_orchestrator():
        """Demonstrate enhanced meta-orchestrator capabilities"""
        orchestrator = EnhancedMetaOrchestrator()
        
        # Test various scenarios
        test_cases = [
            {
                "request": "Create a financial invoice processing system with ML-based categorization",
                "context": {"priority": "high", "domain": "financial", "complexity": 8}
            },
            {
                "request": "Implement parallel data analysis for customer satisfaction metrics",
                "context": {"priority": "medium", "domain": "data_analysis", "parallel": True}
            },
            {
                "request": "Debug and optimize the existing agent coordination system",
                "context": {"priority": "high", "complexity": 6}
            }
        ]
        
        print("=" * 80)
        print("ENHANCED META-ORCHESTRATOR DEMONSTRATION")
        print("=" * 80)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['request'][:60]}...")
            print("-" * 60)
            
            result = await orchestrator.process_request(
                test_case["request"],
                test_case["context"]
            )
            
            print(f"Success: {result['success']}")
            print(f"Strategy Used: {result.get('strategy_used', 'N/A')}")
            print(f"Provider Used: {result.get('provider_used', 'N/A')}")
            print(f"Execution Time: {result.get('execution_time', 0):.2f}s")
            print(f"Quality Score: {result.get('quality_score', 0):.2f}")
            print(f"Spawned Agents: {result.get('spawned_agents', 0)}")
            
            if result.get('insights'):
                print(f"Recent Insights: {len(result['insights'])} patterns identified")
        
        # Show system status
        print("\n" + "=" * 80)
        print("SYSTEM STATUS AFTER DEMONSTRATIONS")
        print("=" * 80)
        
        status = await orchestrator.get_system_status()
        print(f"Total Strategies: {len(status['strategy_performance'])}")
        print(f"Spawned Agents: {len(status['spawned_agents'])}")
        print(f"Learning Insights: {status['learning_insights_count']}")
        
        # Perform self-optimization
        print("\nPerforming self-optimization...")
        optimization_result = await orchestrator.optimize_performance()
        print(f"Optimizations applied: {len(optimization_result['optimizations_applied'])}")
        
        for opt in optimization_result['optimizations_applied']:
            print(f"  - {opt['optimization_type']}: {len(opt.get('adjustments_made', []))} adjustments")
    
    # Run demonstration
    asyncio.run(demo_enhanced_orchestrator())