"""
Advanced Multi-Agent Coordination System
Implements swarm intelligence, competitive selection, and meta-learning patterns
Built on existing orchestrator foundation with Windows-first architecture
"""

import asyncio
import logging
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod

# Import base orchestrator
from core.orchestration.orchestrator import (
    AgentOrchestrator, Task, Message, CommunicationProtocol,
    Blackboard, AgentState
)
from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class CoordinationPattern(Enum):
    """Advanced coordination patterns"""
    COMPETITIVE_SELECTION = "competitive_selection"
    SWARM_INTELLIGENCE = "swarm_intelligence" 
    META_LEARNING = "meta_learning"
    EMERGENT_SPECIALIZATION = "emergent_specialization"
    ADAPTIVE_HIERARCHY = "adaptive_hierarchy"
    CONSENSUS_VOTING = "consensus_voting"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class CompetitiveResult:
    """Result from competitive agent processing"""
    agent_name: str
    result: Any
    confidence_score: float
    processing_time_ms: int
    cost_estimate: float
    method_used: str
    validation_score: float = 0.0
    

@dataclass
class SwarmParticle:
    """Represents an agent in swarm intelligence optimization"""
    agent: BaseAgent
    position: Dict[str, float]
    velocity: Dict[str, float] 
    personal_best: Optional[Dict[str, Any]] = None
    personal_best_fitness: float = float('-inf')
    

@dataclass
class MetaLearningPattern:
    """Pattern discovered through meta-learning"""
    pattern_type: str
    conditions: Dict[str, Any]
    recommended_strategy: str
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedOrchestrator(AgentOrchestrator):
    """
    Enhanced orchestrator with advanced coordination patterns
    Builds on existing orchestrator with new capabilities
    """
    
    def __init__(self, name: str = "advanced_orchestrator"):
        super().__init__(name)
        
        # Advanced coordination state
        self.competitive_history: List[List[CompetitiveResult]] = []
        self.swarm_particles: Dict[str, SwarmParticle] = {}
        self.meta_patterns: List[MetaLearningPattern] = []
        self.specialization_matrix: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Learning parameters
        self.exploration_rate = 0.1
        self.exploitation_threshold = 0.8
        self.meta_learning_window = 100  # Number of tasks to analyze for patterns
        
        # Performance tracking
        self.pattern_usage_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.emergent_behaviors: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized advanced orchestrator: {self.name}")
    
    async def competitive_agent_selection(
        self, 
        task: Task, 
        candidate_agents: Optional[List[BaseAgent]] = None,
        selection_criteria: str = "highest_confidence"
    ) -> CompetitiveResult:
        """
        Run multiple agents competitively and select best result
        Agents compete in parallel and results are evaluated
        """
        logger.info(f"Starting competitive selection for task: {task.id}")
        global_metrics.incr("orchestrator.competitive.started")
        
        # Select candidate agents if not provided
        if candidate_agents is None:
            candidate_agents = await self._select_competitive_candidates(task)
        
        if len(candidate_agents) < 2:
            logger.warning("Not enough agents for competition, falling back to single agent")
            if candidate_agents:
                result = await candidate_agents[0].process_task(task.description, task.requirements)
                return CompetitiveResult(
                    agent_name=candidate_agents[0].name,
                    result=result,
                    confidence_score=0.8,
                    processing_time_ms=1000,
                    cost_estimate=0.0,
                    method_used="single_agent_fallback"
                )
        
        # Run competitive processing
        start_time = datetime.now()
        competitive_tasks = []
        
        for agent in candidate_agents:
            competitive_tasks.append(self._run_competitive_agent(agent, task))
        
        # Execute all agents in parallel
        results = await asyncio.gather(*competitive_tasks, return_exceptions=True)
        
        # Filter and process results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Agent {candidate_agents[i].name} failed: {result}")
                continue
            valid_results.append(result)
        
        if not valid_results:
            raise RuntimeError("All competitive agents failed")
        
        # Select winner based on criteria
        winner = await self._select_competitive_winner(valid_results, selection_criteria)
        
        # Store competitive history for learning
        self.competitive_history.append(valid_results)
        if len(self.competitive_history) > 50:  # Keep last 50 competitions
            self.competitive_history.pop(0)
        
        total_time = (datetime.now() - start_time).total_seconds() * 1000
        logger.info(f"Competitive selection completed in {total_time:.0f}ms, winner: {winner.agent_name}")
        
        global_metrics.incr("orchestrator.competitive.completed")
        return winner
    
    async def _select_competitive_candidates(self, task: Task) -> List[BaseAgent]:
        """Select agents suitable for competitive processing"""
        # Get agents with relevant specializations
        suitable_agents = []
        
        for agent in self.agents.values():
            # Check specialization scores
            agent_specializations = self.specialization_matrix.get(agent.name, {})
            task_keywords = task.description.lower().split()
            
            relevance_score = 0.0
            for keyword in task_keywords:
                relevance_score += agent_specializations.get(keyword, 0.0)
            
            if relevance_score > 0.3 or len(agent_specializations) == 0:  # Include new agents
                suitable_agents.append((agent, relevance_score))
        
        # Sort by relevance and select top candidates
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 3-5 agents for competition
        max_candidates = min(5, len(suitable_agents))
        return [agent for agent, _ in suitable_agents[:max_candidates]]
    
    async def _run_competitive_agent(self, agent: BaseAgent, task: Task) -> CompetitiveResult:
        """Run single agent in competitive context"""
        start_time = datetime.now()
        
        try:
            # Add competitive context
            competitive_context = task.requirements.copy()
            competitive_context['competitive_mode'] = True
            competitive_context['urgency'] = 'high'
            
            result = await agent.process_task(task.description, competitive_context)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Estimate confidence (agent-specific or heuristic)
            confidence = getattr(result, 'confidence_score', None) or self._estimate_confidence(result)
            
            return CompetitiveResult(
                agent_name=agent.name,
                result=result,
                confidence_score=confidence,
                processing_time_ms=int(processing_time),
                cost_estimate=self._estimate_cost(agent, result),
                method_used=getattr(result, 'method_used', 'unknown'),
                validation_score=await self._validate_result(result, task)
            )
            
        except Exception as e:
            logger.error(f"Competitive agent {agent.name} failed: {e}")
            raise
    
    async def _select_competitive_winner(
        self, 
        results: List[CompetitiveResult], 
        criteria: str
    ) -> CompetitiveResult:
        """Select winning result based on criteria"""
        
        if criteria == "highest_confidence":
            return max(results, key=lambda r: r.confidence_score)
        
        elif criteria == "fastest_processing":
            return min(results, key=lambda r: r.processing_time_ms)
        
        elif criteria == "best_value":
            # Balance confidence and cost
            def value_score(r):
                return r.confidence_score / max(0.01, r.cost_estimate)
            return max(results, key=value_score)
        
        elif criteria == "consensus_voting":
            return await self._consensus_winner(results)
        
        elif criteria == "validation_score":
            return max(results, key=lambda r: r.validation_score)
        
        else:
            # Default to highest confidence
            return max(results, key=lambda r: r.confidence_score)
    
    async def _consensus_winner(self, results: List[CompetitiveResult]) -> CompetitiveResult:
        """Select winner through cross-validation consensus"""
        scores = defaultdict(float)
        
        # Each result "votes" on others based on similarity
        for i, result_a in enumerate(results):
            for j, result_b in enumerate(results):
                if i != j:
                    similarity = await self._calculate_result_similarity(
                        result_a.result, result_b.result
                    )
                    scores[j] += similarity * result_a.confidence_score
        
        # Winner is result with highest consensus score
        winner_idx = max(scores.keys(), key=lambda k: scores[k])
        return results[winner_idx]
    
    async def advanced_swarm_optimization(
        self, 
        objective: str,
        swarm_size: int = 8,
        max_iterations: int = 30,
        target_fitness: float = 0.95
    ) -> Dict[str, Any]:
        """
        Enhanced swarm intelligence with adaptive parameters
        """
        logger.info(f"Starting swarm optimization: {objective}")
        global_metrics.incr("orchestrator.swarm.started")
        
        # Initialize swarm if needed
        if len(self.swarm_particles) < swarm_size:
            await self._initialize_swarm(objective, swarm_size)
        
        # Swarm optimization parameters (adaptive)
        w = 0.7  # Inertia weight
        c1 = 1.5  # Personal best acceleration
        c2 = 1.5  # Global best acceleration
        
        global_best = None
        global_best_fitness = float('-inf')
        stagnation_count = 0
        
        # Optimization loop
        for iteration in range(max_iterations):
            logger.info(f"Swarm iteration {iteration + 1}/{max_iterations}")
            
            # Evaluate all particles
            fitness_improvements = 0
            iteration_results = []
            
            for particle_id, particle in self.swarm_particles.items():
                # Generate solution based on particle position
                solution = await self._particle_explore(particle, objective, iteration)
                fitness = await self._evaluate_swarm_fitness(solution, objective)
                
                iteration_results.append({
                    'particle_id': particle_id,
                    'solution': solution,
                    'fitness': fitness,
                    'position': particle.position.copy()
                })
                
                # Update personal best
                if fitness > particle.personal_best_fitness:
                    particle.personal_best = solution
                    particle.personal_best_fitness = fitness
                    fitness_improvements += 1
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best = solution
                    global_best_fitness = fitness
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            
            # Update particle velocities and positions
            await self._update_swarm_particles(global_best, w, c1, c2)
            
            # Adaptive parameters
            if fitness_improvements == 0:
                # Increase exploration if no improvements
                w = min(0.9, w + 0.05)
                c1 = min(2.0, c1 + 0.1)
            else:
                # Increase exploitation if improving
                w = max(0.4, w - 0.02)
                c2 = min(2.0, c2 + 0.1)
            
            # Early stopping if target reached
            if global_best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at iteration {iteration + 1}")
                break
            
            # Diversity maintenance
            if stagnation_count > 10:
                await self._inject_swarm_diversity()
                stagnation_count = 0
        
        result = {
            'best_solution': global_best,
            'best_fitness': global_best_fitness,
            'iterations_completed': iteration + 1,
            'convergence_achieved': global_best_fitness >= target_fitness,
            'final_diversity': await self._calculate_swarm_diversity(),
            'emergent_patterns': await self._detect_swarm_patterns()
        }
        
        logger.info(f"Swarm optimization completed: fitness={global_best_fitness:.3f}")
        global_metrics.incr("orchestrator.swarm.completed")
        
        return result
    
    async def meta_learning_coordinator(self, task: Task) -> Any:
        """
        Apply meta-learning to select optimal coordination pattern
        """
        logger.info(f"Applying meta-learning for task: {task.id}")
        
        # Analyze task characteristics
        task_features = await self._extract_task_features(task)
        
        # Find matching patterns from history
        matching_patterns = await self._find_matching_patterns(task_features)
        
        if matching_patterns:
            # Use most successful pattern
            best_pattern = max(matching_patterns, key=lambda p: p.success_rate)
            logger.info(f"Applying learned pattern: {best_pattern.pattern_type}")
            
            return await self._apply_meta_pattern(task, best_pattern)
        
        else:
            # No matching pattern, use exploration
            logger.info("No matching pattern found, using exploration strategy")
            return await self._explore_new_pattern(task)
    
    async def detect_emergent_specialization(self) -> Dict[str, Any]:
        """
        Detect and analyze emergent agent specializations
        """
        logger.info("Analyzing emergent specializations")
        
        specializations = {}
        
        # Analyze agent performance across different task types
        for agent_name, agent in self.agents.items():
            agent_specialization = {}
            
            # Analyze episodic memory for specialization patterns
            if hasattr(agent.memory, 'episodic_memory'):
                task_performance = defaultdict(list)
                
                for observation in agent.memory.episodic_memory:
                    task_type = self._classify_task_type(observation.action)
                    task_performance[task_type].append(observation.success)
                
                # Calculate specialization scores
                for task_type, successes in task_performance.items():
                    if len(successes) >= 3:  # Minimum sample size
                        success_rate = sum(successes) / len(successes)
                        sample_size = len(successes)
                        
                        # Weight by sample size and success rate
                        specialization_score = success_rate * min(1.0, sample_size / 10.0)
                        agent_specialization[task_type] = specialization_score
            
            # Update specialization matrix
            self.specialization_matrix[agent_name] = agent_specialization
            specializations[agent_name] = agent_specialization
        
        # Detect emergent roles
        emergent_roles = await self._identify_emergent_roles(specializations)
        
        # Detect collaboration patterns
        collaboration_patterns = await self._analyze_collaboration_patterns()
        
        return {
            'agent_specializations': specializations,
            'emergent_roles': emergent_roles,
            'collaboration_patterns': collaboration_patterns,
            'specialization_strength': self._calculate_specialization_strength(specializations),
            'recommendations': await self._generate_specialization_recommendations(specializations)
        }
    
    async def chain_of_thought_coordination(self, task: Task, max_agents: int = 5) -> Any:
        """
        Implement chain-of-thought reasoning across multiple agents
        """
        logger.info(f"Starting chain-of-thought coordination for: {task.id}")
        
        # Select agents for the chain
        chain_agents = await self._select_chain_agents(task, max_agents)
        
        # Initialize reasoning chain
        reasoning_chain = []
        current_context = task.requirements.copy()
        
        # Execute reasoning chain
        for i, agent in enumerate(chain_agents):
            logger.info(f"Chain step {i + 1}: Agent {agent.name}")
            
            # Add chain context
            chain_context = current_context.copy()
            chain_context['chain_position'] = i
            chain_context['previous_reasoning'] = reasoning_chain[-3:] if reasoning_chain else []
            chain_context['chain_agents'] = [a.name for a in chain_agents]
            
            # Get agent's reasoning step
            reasoning_step = await agent.process_task(
                f"Reasoning step {i + 1}: {task.description}",
                chain_context
            )
            
            # Add to chain
            reasoning_chain.append({
                'step': i + 1,
                'agent': agent.name,
                'reasoning': reasoning_step,
                'timestamp': datetime.now()
            })
            
            # Update context for next agent
            current_context['previous_step'] = reasoning_step
            
            # Check for early completion
            if await self._is_reasoning_complete(reasoning_step, task):
                logger.info(f"Reasoning chain completed early at step {i + 1}")
                break
        
        # Synthesize final result
        final_result = await self._synthesize_chain_reasoning(reasoning_chain, task)
        
        return {
            'final_result': final_result,
            'reasoning_chain': reasoning_chain,
            'chain_length': len(reasoning_chain),
            'participating_agents': [step['agent'] for step in reasoning_chain]
        }
    
    # Helper methods for advanced coordination
    
    async def _initialize_swarm(self, objective: str, swarm_size: int):
        """Initialize swarm particles for optimization"""
        # Clear existing particles
        self.swarm_particles.clear()
        
        # Create new particles
        available_agents = list(self.agents.values())
        
        for i in range(swarm_size):
            agent = available_agents[i % len(available_agents)]
            
            # Initialize random position and velocity
            position = {
                'exploration': np.random.uniform(0.0, 1.0),
                'exploitation': np.random.uniform(0.0, 1.0),
                'confidence_threshold': np.random.uniform(0.5, 1.0),
                'processing_speed': np.random.uniform(0.0, 1.0)
            }
            
            velocity = {k: np.random.uniform(-0.1, 0.1) for k in position.keys()}
            
            particle = SwarmParticle(
                agent=agent,
                position=position,
                velocity=velocity
            )
            
            self.swarm_particles[f"particle_{i}"] = particle
    
    def _estimate_confidence(self, result: Any) -> float:
        """Estimate confidence score for result"""
        if isinstance(result, dict) and 'confidence' in result:
            return result['confidence']
        elif hasattr(result, 'confidence_score'):
            return result.confidence_score
        else:
            # Heuristic based on result content
            if result is None:
                return 0.0
            elif isinstance(result, str) and len(result) < 10:
                return 0.3
            else:
                return 0.7
    
    def _estimate_cost(self, agent: BaseAgent, result: Any) -> float:
        """Estimate processing cost"""
        # Base cost estimation (can be enhanced with actual tracking)
        base_cost = 0.01
        
        if hasattr(agent, 'total_tokens_used'):
            token_cost = agent.total_tokens_used * 0.000002  # Rough token cost
            return base_cost + token_cost
        
        return base_cost
    
    async def _validate_result(self, result: Any, task: Task) -> float:
        """Validate result quality"""
        # Basic validation heuristics
        score = 0.5
        
        if result is not None:
            score += 0.2
            
        if isinstance(result, dict) and len(result) > 0:
            score += 0.2
            
        # Task-specific validation
        if 'invoice' in task.description.lower():
            if isinstance(result, dict):
                required_fields = ['invoice_number', 'total_amount', 'vendor_name']
                found_fields = sum(1 for field in required_fields if field in result)
                score += (found_fields / len(required_fields)) * 0.3
        
        return min(1.0, score)
    
    async def _calculate_result_similarity(self, result1: Any, result2: Any) -> float:
        """Calculate similarity between two results"""
        if type(result1) != type(result2):
            return 0.0
        
        if isinstance(result1, dict) and isinstance(result2, dict):
            common_keys = set(result1.keys()) & set(result2.keys())
            if not common_keys:
                return 0.0
            
            similarity_sum = 0.0
            for key in common_keys:
                if result1[key] == result2[key]:
                    similarity_sum += 1.0
                elif isinstance(result1[key], str) and isinstance(result2[key], str):
                    # Simple string similarity
                    if result1[key].lower() in result2[key].lower() or result2[key].lower() in result1[key].lower():
                        similarity_sum += 0.5
            
            return similarity_sum / len(common_keys)
        
        elif str(result1) == str(result2):
            return 1.0
        
        return 0.0
    
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
    
    async def _particle_explore(self, particle: SwarmParticle, objective: str, iteration: int) -> Dict[str, Any]:
        """Particle exploration based on position and velocity"""
        # Implement particle exploration logic
        exploration_params = {
            'exploration_rate': particle.position.get('exploration', 0.5),
            'confidence_threshold': particle.position.get('confidence_threshold', 0.7),
            'processing_speed': particle.position.get('processing_speed', 0.5)
        }
        
        # Execute exploration task with particle's agent
        task_description = f"Explore solution space for: {objective} (iteration {iteration})"
        solution = await particle.agent.process_task(task_description, exploration_params)
        
        return {
            'solution': solution,
            'parameters': exploration_params,
            'particle_id': id(particle),
            'agent': particle.agent.name
        }
    
    def get_advanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics including advanced coordination patterns"""
        base_metrics = self.get_metrics()
        
        advanced_metrics = {
            'competitive_competitions': len(self.competitive_history),
            'swarm_particles': len(self.swarm_particles),
            'meta_patterns_learned': len(self.meta_patterns),
            'emergent_behaviors_detected': len(self.emergent_behaviors),
            'specialization_agents': len(self.specialization_matrix),
            'pattern_usage_stats': dict(self.pattern_usage_stats),
            'coordination_effectiveness': self._calculate_coordination_effectiveness()
        }
        
        return {**base_metrics, 'advanced_metrics': advanced_metrics}
    
    def _calculate_coordination_effectiveness(self) -> float:
        """Calculate overall coordination effectiveness score"""
        if not self.competitive_history:
            return 0.5
        
        # Calculate average competitive improvement
        improvements = []
        for competition in self.competitive_history[-10:]:  # Last 10 competitions
            if len(competition) > 1:
                best_score = max(r.confidence_score for r in competition)
                avg_score = sum(r.confidence_score for r in competition) / len(competition)
                improvement = (best_score - avg_score) / max(avg_score, 0.1)
                improvements.append(improvement)
        
        return statistics.mean(improvements) if improvements else 0.5