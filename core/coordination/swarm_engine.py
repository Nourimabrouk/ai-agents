"""
Swarm Intelligence Engine
Advanced implementation of swarm algorithms including PSO, ACO, and emergent behavior detection
Built for Windows development environment with async/await patterns
"""

import asyncio
import numpy as np
import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import json
import statistics
from abc import ABC, abstractmethod

from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class SwarmAlgorithm(Enum):
    """Types of swarm algorithms"""
    PARTICLE_SWARM_OPTIMIZATION = "pso"
    ANT_COLONY_OPTIMIZATION = "aco"
    BEE_COLONY_OPTIMIZATION = "bco"
    FIREFLY_ALGORITHM = "firefly"
    BACTERIAL_FORAGING = "bacterial"
    GREY_WOLF_OPTIMIZER = "gwo"


@dataclass
class SwarmParticle:
    """Enhanced particle for swarm optimization"""
    agent: BaseAgent
    position: Dict[str, float]
    velocity: Dict[str, float]
    personal_best: Optional[Dict[str, Any]] = None
    personal_best_fitness: float = float('-inf')
    social_influence: float = 0.5
    exploration_history: List[Dict[str, Any]] = field(default_factory=list)
    specialization_score: float = 0.0
    energy_level: float = 1.0


@dataclass
class AntAgent:
    """Ant agent for Ant Colony Optimization"""
    agent: BaseAgent
    current_position: Tuple[int, int]
    path: List[Tuple[int, int]]
    path_quality: float = 0.0
    pheromone_trail: float = 1.0
    memory_size: int = 10
    exploration_probability: float = 0.1


@dataclass
class EmergentBehavior:
    """Detected emergent behavior in swarm"""
    behavior_type: str
    description: str
    participants: List[str]
    emergence_time: datetime
    strength: float
    patterns: Dict[str, Any]
    reproducible: bool = False


class SwarmEngine:
    """
    Advanced Swarm Intelligence Engine
    Implements multiple swarm algorithms with emergent behavior detection
    """
    
    def __init__(self, name: str = "swarm_engine"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        
        # Swarm components
        self.particles: Dict[str, SwarmParticle] = {}
        self.ants: Dict[str, AntAgent] = {}
        self.pheromone_matrix: np.ndarray = None
        self.environment_graph: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        
        # Optimization parameters
        self.pso_params = {
            'w': 0.7,  # Inertia weight
            'c1': 1.5,  # Personal best acceleration
            'c2': 1.5,  # Global best acceleration
            'v_max': 0.5  # Maximum velocity
        }
        
        self.aco_params = {
            'alpha': 1.0,  # Pheromone importance
            'beta': 2.0,   # Heuristic importance
            'rho': 0.5,    # Evaporation rate
            'q': 100       # Pheromone deposit factor
        }
        
        # Emergent behavior detection
        self.behavior_history: List[EmergentBehavior] = []
        self.interaction_network: Dict[str, Set[str]] = defaultdict(set)
        self.collective_memory: Dict[str, Any] = {}
        
        # Performance tracking
        self.optimization_history: List[Dict[str, Any]] = []
        self.convergence_metrics: Dict[str, float] = {}
        
        logger.info(f"Initialized swarm engine: {self.name}")
    
    async def particle_swarm_optimization(
        self, 
        objective_function: str,
        swarm_size: int = 20,
        max_iterations: int = 100,
        target_fitness: float = 0.95,
        dimension_bounds: Dict[str, Tuple[float, float]] = None
    ) -> Dict[str, Any]:
        """
        Advanced Particle Swarm Optimization with adaptive parameters
        """
        logger.info(f"Starting PSO with {swarm_size} particles for: {objective_function}")
        global_metrics.incr("swarm.pso.started")
        
        # Initialize swarm
        await self._initialize_pso_swarm(objective_function, swarm_size, dimension_bounds)
        
        global_best = None
        global_best_fitness = float('-inf')
        stagnation_counter = 0
        diversity_history = []
        
        # Adaptive parameters
        w_start, w_end = 0.9, 0.4
        
        for iteration in range(max_iterations):
            logger.info(f"PSO iteration {iteration + 1}/{max_iterations}")
            
            # Adaptive inertia weight
            w = w_start - (w_start - w_end) * iteration / max_iterations
            self.pso_params['w'] = w
            
            # Evaluate all particles
            fitness_improvements = 0
            iteration_results = []
            
            for particle_id, particle in self.particles.items():
                # Particle exploration
                solution = await self._particle_explore_advanced(
                    particle, objective_function, iteration
                )
                
                # Evaluate fitness with multi-criteria approach
                fitness = await self._evaluate_multi_criteria_fitness(
                    solution, objective_function, particle
                )
                
                # Store results
                iteration_results.append({
                    'particle_id': particle_id,
                    'solution': solution,
                    'fitness': fitness,
                    'position': particle.position.copy(),
                    'agent': particle.agent.name
                })
                
                # Update personal best
                if fitness > particle.personal_best_fitness:
                    particle.personal_best = solution
                    particle.personal_best_fitness = fitness
                    fitness_improvements += 1
                    
                    # Add to exploration history
                    particle.exploration_history.append({
                        'iteration': iteration,
                        'solution': solution,
                        'fitness': fitness,
                        'position': particle.position.copy()
                    })
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best = solution
                    global_best_fitness = fitness
                    stagnation_counter = 0
                    logger.info(f"New global best found: fitness={fitness:.4f}")
                else:
                    stagnation_counter += 1
            
            # Update particle positions and velocities
            await self._update_pso_particles(global_best, iteration)
            
            # Calculate swarm diversity
            diversity = await self._calculate_swarm_diversity()
            diversity_history.append(diversity)
            
            # Adaptive parameter adjustment
            if fitness_improvements == 0:
                # Increase exploration if no improvements
                self.pso_params['c1'] = min(2.5, self.pso_params['c1'] + 0.1)
                self.pso_params['v_max'] = min(1.0, self.pso_params['v_max'] + 0.05)
            else:
                # Increase exploitation if improving
                self.pso_params['c2'] = min(2.5, self.pso_params['c2'] + 0.1)
            
            # Detect emergent behaviors
            emergent_behaviors = await self._detect_pso_emergent_behaviors(
                iteration_results, iteration
            )
            
            # Early stopping conditions
            if global_best_fitness >= target_fitness:
                logger.info(f"Target fitness {target_fitness} reached at iteration {iteration + 1}")
                break
            
            # Diversity maintenance
            if diversity < 0.1 or stagnation_counter > 15:
                await self._inject_pso_diversity()
                stagnation_counter = 0
                logger.info("Injected diversity to prevent stagnation")
        
        # Final analysis
        convergence_analysis = await self._analyze_pso_convergence(diversity_history)
        
        result = {
            'algorithm': 'particle_swarm_optimization',
            'best_solution': global_best,
            'best_fitness': global_best_fitness,
            'iterations_completed': iteration + 1,
            'convergence_achieved': global_best_fitness >= target_fitness,
            'final_diversity': diversity_history[-1] if diversity_history else 0.0,
            'emergent_behaviors': emergent_behaviors,
            'convergence_analysis': convergence_analysis,
            'swarm_size': swarm_size,
            'parameter_evolution': await self._get_parameter_evolution()
        }
        
        self.optimization_history.append(result)
        global_metrics.incr("swarm.pso.completed")
        
        return result
    
    async def ant_colony_optimization(
        self,
        problem_graph: Dict[str, List[Tuple[str, float]]],
        objective: str,
        colony_size: int = 50,
        max_iterations: int = 100,
        target_quality: float = 0.9
    ) -> Dict[str, Any]:
        """
        Advanced Ant Colony Optimization with dynamic pheromone management
        """
        logger.info(f"Starting ACO with {colony_size} ants for: {objective}")
        global_metrics.incr("swarm.aco.started")
        
        # Initialize ant colony
        await self._initialize_aco_colony(problem_graph, colony_size)
        
        # Initialize pheromone matrix
        self._initialize_pheromone_matrix(problem_graph)
        
        best_path = None
        best_path_quality = float('-inf')
        convergence_data = []
        
        for iteration in range(max_iterations):
            logger.info(f"ACO iteration {iteration + 1}/{max_iterations}")
            
            # Ant foraging phase
            iteration_paths = []
            
            for ant_id, ant in self.ants.items():
                # Construct path
                path = await self._ant_construct_path(ant, problem_graph, iteration)
                
                # Evaluate path quality
                path_quality = await self._evaluate_path_quality(path, objective)
                
                ant.path = path
                ant.path_quality = path_quality
                
                iteration_paths.append({
                    'ant_id': ant_id,
                    'path': path,
                    'quality': path_quality,
                    'agent': ant.agent.name
                })
                
                # Update best path
                if path_quality > best_path_quality:
                    best_path = path
                    best_path_quality = path_quality
                    logger.info(f"New best path found: quality={path_quality:.4f}")
            
            # Pheromone update phase
            await self._update_pheromones(iteration_paths)
            
            # Detect emergent ant behaviors
            emergent_behaviors = await self._detect_aco_emergent_behaviors(
                iteration_paths, iteration
            )
            
            # Convergence analysis
            convergence_data.append({
                'iteration': iteration,
                'best_quality': best_path_quality,
                'average_quality': np.mean([p['quality'] for p in iteration_paths]),
                'diversity': await self._calculate_path_diversity(iteration_paths)
            })
            
            # Early stopping
            if best_path_quality >= target_quality:
                logger.info(f"Target quality {target_quality} reached at iteration {iteration + 1}")
                break
            
            # Adaptive pheromone management
            if iteration % 10 == 0:
                await self._adaptive_pheromone_management(convergence_data[-10:])
        
        result = {
            'algorithm': 'ant_colony_optimization',
            'best_path': best_path,
            'best_quality': best_path_quality,
            'iterations_completed': iteration + 1,
            'target_achieved': best_path_quality >= target_quality,
            'emergent_behaviors': emergent_behaviors,
            'convergence_data': convergence_data,
            'colony_size': colony_size,
            'final_pheromone_distribution': await self._get_pheromone_distribution()
        }
        
        self.optimization_history.append(result)
        global_metrics.incr("swarm.aco.completed")
        
        return result
    
    async def emergent_behavior_analysis(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of emergent behaviors in the swarm
        """
        logger.info("Analyzing emergent behaviors in swarm system")
        
        analysis = {
            'detected_behaviors': [],
            'behavior_patterns': {},
            'interaction_networks': {},
            'collective_intelligence_metrics': {},
            'innovation_indicators': {},
            'system_evolution': {}
        }
        
        # Analyze detected behaviors
        for behavior in self.behavior_history:
            behavior_analysis = {
                'type': behavior.behavior_type,
                'description': behavior.description,
                'participants': behavior.participants,
                'strength': behavior.strength,
                'duration': (datetime.now() - behavior.emergence_time).total_seconds(),
                'reproducible': behavior.reproducible,
                'patterns': behavior.patterns
            }
            analysis['detected_behaviors'].append(behavior_analysis)
        
        # Analyze behavior patterns
        behavior_types = defaultdict(list)
        for behavior in self.behavior_history:
            behavior_types[behavior.behavior_type].append(behavior)
        
        for behavior_type, behaviors in behavior_types.items():
            analysis['behavior_patterns'][behavior_type] = {
                'frequency': len(behaviors),
                'average_strength': np.mean([b.strength for b in behaviors]),
                'typical_participants': self._analyze_typical_participants(behaviors),
                'evolution_trend': self._analyze_behavior_evolution(behaviors)
            }
        
        # Analyze interaction networks
        analysis['interaction_networks'] = await self._analyze_interaction_networks()
        
        # Collective intelligence metrics
        analysis['collective_intelligence_metrics'] = await self._calculate_collective_intelligence()
        
        # Innovation indicators
        analysis['innovation_indicators'] = await self._detect_innovation_patterns()
        
        # System evolution
        analysis['system_evolution'] = await self._analyze_system_evolution()
        
        return analysis
    
    async def adaptive_swarm_coordination(
        self,
        tasks: List[Dict[str, Any]],
        coordination_strategy: str = "adaptive",
        performance_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        Adaptive coordination of swarm agents for complex task execution
        """
        logger.info(f"Starting adaptive swarm coordination for {len(tasks)} tasks")
        
        results = {
            'completed_tasks': [],
            'failed_tasks': [],
            'coordination_metrics': {},
            'adaptation_history': [],
            'emergent_strategies': []
        }
        
        # Initialize coordination state
        coordination_state = {
            'active_strategy': coordination_strategy,
            'performance_history': deque(maxlen=20),
            'adaptation_triggers': [],
            'strategy_effectiveness': defaultdict(list)
        }
        
        for task_idx, task in enumerate(tasks):
            logger.info(f"Processing task {task_idx + 1}/{len(tasks)}: {task.get('description', 'Unnamed task')}")
            
            # Select coordination approach
            current_strategy = await self._select_coordination_strategy(
                task, coordination_state
            )
            
            # Execute task with selected strategy
            task_result = await self._execute_coordinated_task(task, current_strategy)
            
            # Evaluate performance
            performance = await self._evaluate_task_performance(task_result, task)
            coordination_state['performance_history'].append(performance)
            coordination_state['strategy_effectiveness'][current_strategy].append(performance)
            
            # Store results
            if performance >= performance_threshold:
                results['completed_tasks'].append({
                    'task': task,
                    'result': task_result,
                    'performance': performance,
                    'strategy_used': current_strategy
                })
            else:
                results['failed_tasks'].append({
                    'task': task,
                    'result': task_result,
                    'performance': performance,
                    'strategy_used': current_strategy,
                    'failure_reason': await self._analyze_failure_reason(task_result, task)
                })
            
            # Check for adaptation triggers
            adaptation_needed = await self._check_adaptation_triggers(coordination_state)
            
            if adaptation_needed:
                new_strategy = await self._adapt_coordination_strategy(coordination_state)
                coordination_state['adaptation_history'].append({
                    'from_strategy': coordination_state['active_strategy'],
                    'to_strategy': new_strategy,
                    'trigger_reason': adaptation_needed,
                    'task_index': task_idx
                })
                coordination_state['active_strategy'] = new_strategy
                logger.info(f"Adapted coordination strategy to: {new_strategy}")
            
            # Detect emergent coordination strategies
            emergent_strategy = await self._detect_emergent_coordination(task_result, task)
            if emergent_strategy:
                results['emergent_strategies'].append(emergent_strategy)
        
        # Final coordination metrics
        results['coordination_metrics'] = {
            'success_rate': len(results['completed_tasks']) / len(tasks),
            'average_performance': np.mean(coordination_state['performance_history']),
            'adaptations_made': len(coordination_state['adaptation_history']),
            'strategy_effectiveness': dict(coordination_state['strategy_effectiveness']),
            'emergent_strategies_discovered': len(results['emergent_strategies'])
        }
        
        return results
    
    # Helper methods for swarm algorithms
    
    async def _initialize_pso_swarm(
        self, 
        objective: str, 
        swarm_size: int,
        dimension_bounds: Dict[str, Tuple[float, float]] = None
    ):
        """Initialize PSO swarm with particles"""
        self.particles.clear()
        
        # Default dimension bounds
        if dimension_bounds is None:
            dimension_bounds = {
                'exploration': (0.0, 1.0),
                'exploitation': (0.0, 1.0),
                'confidence_threshold': (0.3, 1.0),
                'processing_speed': (0.1, 1.0),
                'risk_tolerance': (0.0, 1.0)
            }
        
        # Create particles
        available_agents = list(self.agents.values())
        
        for i in range(swarm_size):
            agent = available_agents[i % len(available_agents)] if available_agents else None
            
            # Initialize position and velocity
            position = {}
            velocity = {}
            
            for dim, (min_val, max_val) in dimension_bounds.items():
                position[dim] = np.random.uniform(min_val, max_val)
                velocity[dim] = np.random.uniform(-0.1, 0.1)
            
            particle = SwarmParticle(
                agent=agent,
                position=position,
                velocity=velocity,
                social_influence=np.random.uniform(0.3, 0.7)
            )
            
            self.particles[f"particle_{i}"] = particle
    
    async def _particle_explore_advanced(
        self, 
        particle: SwarmParticle, 
        objective: str, 
        iteration: int
    ) -> Dict[str, Any]:
        """Advanced particle exploration with learning"""
        if not particle.agent:
            return {'solution': None, 'error': 'No agent assigned'}
        
        # Create exploration context
        exploration_context = {
            'objective': objective,
            'iteration': iteration,
            'position': particle.position,
            'exploration_history': particle.exploration_history[-5:],  # Last 5 explorations
            'social_influence': particle.social_influence,
            'energy_level': particle.energy_level
        }
        
        # Adjust exploration based on particle's energy and history
        if particle.energy_level < 0.3:
            # Low energy - focus on exploitation
            exploration_context['mode'] = 'exploitation'
            exploration_context['exploration_rate'] = particle.position.get('exploration', 0.5) * 0.5
        elif len(particle.exploration_history) > 3:
            # Has history - balance exploration and exploitation
            recent_fitness = [h['fitness'] for h in particle.exploration_history[-3:]]
            if len(recent_fitness) > 1 and np.std(recent_fitness) < 0.1:
                # Stuck in local optimum - increase exploration
                exploration_context['mode'] = 'exploration'
                exploration_context['exploration_rate'] = min(1.0, particle.position.get('exploration', 0.5) * 1.5)
            else:
                exploration_context['mode'] = 'balanced'
        else:
            # New particle - pure exploration
            exploration_context['mode'] = 'exploration'
        
        # Execute exploration with agent
        try:
            solution = await particle.agent.process_task(
                f"Explore solution space for: {objective}",
                exploration_context
            )
            
            # Update particle energy (decreases with exploration)
            particle.energy_level = max(0.1, particle.energy_level - 0.05)
            
            return {
                'solution': solution,
                'exploration_context': exploration_context,
                'particle_id': id(particle),
                'agent': particle.agent.name
            }
            
        except Exception as e:
            logger.error(f"Particle exploration failed: {e}")
            return {'solution': None, 'error': str(e)}
    
    async def _evaluate_multi_criteria_fitness(
        self, 
        solution: Dict[str, Any], 
        objective: str, 
        particle: SwarmParticle
    ) -> float:
        """Multi-criteria fitness evaluation"""
        if not solution or 'solution' not in solution:
            return 0.0
        
        fitness_components = []
        
        # Solution quality (40% weight)
        solution_quality = await self._assess_solution_quality(solution['solution'])
        fitness_components.append(('quality', solution_quality, 0.4))
        
        # Novelty score (20% weight) - reward exploration of new areas
        novelty_score = await self._calculate_novelty_score(solution, particle)
        fitness_components.append(('novelty', novelty_score, 0.2))
        
        # Consistency score (20% weight) - reward consistent performance
        consistency_score = await self._calculate_consistency_score(particle)
        fitness_components.append(('consistency', consistency_score, 0.2))
        
        # Efficiency score (20% weight) - reward resource efficiency
        efficiency_score = await self._calculate_efficiency_score(solution, particle)
        fitness_components.append(('efficiency', efficiency_score, 0.2))
        
        # Calculate weighted fitness
        total_fitness = sum(score * weight for _, score, weight in fitness_components)
        
        # Store fitness breakdown for analysis
        if not hasattr(particle, 'fitness_breakdown'):
            particle.fitness_breakdown = []
        
        particle.fitness_breakdown.append({
            'total_fitness': total_fitness,
            'components': dict((name, score) for name, score, _ in fitness_components)
        })
        
        return total_fitness
    
    async def _assess_solution_quality(self, solution: Any) -> float:
        """Assess the intrinsic quality of a solution"""
        if solution is None:
            return 0.0
        
        if isinstance(solution, dict):
            # Quality based on completeness and content richness
            completeness = min(1.0, len(solution) / 8.0)  # Normalize to expected fields
            
            # Check for key indicators of quality
            quality_indicators = ['result', 'analysis', 'recommendation', 'confidence']
            indicator_score = sum(1 for indicator in quality_indicators if indicator in solution) / len(quality_indicators)
            
            return (completeness * 0.6) + (indicator_score * 0.4)
        
        if isinstance(solution, str):
            # Quality based on length and content diversity
            length_score = min(1.0, len(solution) / 200.0)
            
            # Simple content diversity measure
            unique_words = len(set(solution.lower().split()))
            total_words = len(solution.split())
            diversity_score = unique_words / max(1, total_words)
            
            return (length_score * 0.7) + (diversity_score * 0.3)
        
        return 0.5  # Default for other types
    
    async def _calculate_novelty_score(self, solution: Dict[str, Any], particle: SwarmParticle) -> float:
        """Calculate novelty score compared to particle's history"""
        if not particle.exploration_history:
            return 1.0  # First solution is novel by definition
        
        current_solution = solution.get('solution')
        if current_solution is None:
            return 0.0
        
        # Compare with recent solutions
        similarities = []
        for historical in particle.exploration_history[-5:]:  # Last 5 solutions
            historical_solution = historical.get('solution', {}).get('solution')
            if historical_solution:
                similarity = await self._calculate_solution_similarity(current_solution, historical_solution)
                similarities.append(similarity)
        
        if not similarities:
            return 1.0
        
        # Novelty is inverse of average similarity
        avg_similarity = np.mean(similarities)
        novelty = 1.0 - avg_similarity
        
        return max(0.0, min(1.0, novelty))
    
    async def _calculate_consistency_score(self, particle: SwarmParticle) -> float:
        """Calculate consistency score based on particle's performance history"""
        if len(particle.exploration_history) < 3:
            return 0.5  # Neutral score for insufficient history
        
        recent_fitness = [h['fitness'] for h in particle.exploration_history[-5:] if 'fitness' in h]
        
        if len(recent_fitness) < 2:
            return 0.5
        
        # Consistency is inverse of fitness variance (but reward high average fitness)
        mean_fitness = np.mean(recent_fitness)
        fitness_std = np.std(recent_fitness)
        
        # Normalize consistency score
        consistency = 1.0 - min(1.0, fitness_std / 0.5)  # Normalize by expected std
        
        # Weight by mean fitness (consistent high performance is better than consistent low performance)
        weighted_consistency = consistency * (0.5 + mean_fitness * 0.5)
        
        return max(0.0, min(1.0, weighted_consistency))
    
    async def _calculate_efficiency_score(self, solution: Dict[str, Any], particle: SwarmParticle) -> float:
        """Calculate efficiency score based on resource usage"""
        # Simple efficiency heuristics (can be enhanced with actual resource tracking)
        base_efficiency = 0.7
        
        # Reward low energy consumption
        energy_efficiency = particle.energy_level  # Higher remaining energy = more efficient
        
        # Reward fast exploration (based on solution complexity vs time)
        if 'solution' in solution and isinstance(solution['solution'], dict):
            solution_complexity = len(solution['solution'])
            # Assume more complex solutions should take more energy
            expected_energy_cost = min(0.3, solution_complexity * 0.02)
            actual_energy_cost = 1.0 - particle.energy_level
            
            if actual_energy_cost <= expected_energy_cost:
                speed_efficiency = 1.0
            else:
                speed_efficiency = expected_energy_cost / actual_energy_cost
        else:
            speed_efficiency = 0.5
        
        total_efficiency = (base_efficiency * 0.4) + (energy_efficiency * 0.3) + (speed_efficiency * 0.3)
        
        return max(0.0, min(1.0, total_efficiency))
    
    async def _calculate_solution_similarity(self, solution1: Any, solution2: Any) -> float:
        """Calculate similarity between two solutions"""
        if type(solution1) != type(solution2):
            return 0.0
        
        if isinstance(solution1, dict) and isinstance(solution2, dict):
            # Dictionary similarity based on common keys and values
            all_keys = set(solution1.keys()) | set(solution2.keys())
            if not all_keys:
                return 1.0
            
            matching_score = 0.0
            for key in all_keys:
                if key in solution1 and key in solution2:
                    if solution1[key] == solution2[key]:
                        matching_score += 1.0
                    elif isinstance(solution1[key], str) and isinstance(solution2[key], str):
                        # String similarity
                        common_words = set(solution1[key].lower().split()) & set(solution2[key].lower().split())
                        total_words = set(solution1[key].lower().split()) | set(solution2[key].lower().split())
                        if total_words:
                            matching_score += len(common_words) / len(total_words)
            
            return matching_score / len(all_keys)
        
        elif isinstance(solution1, str) and isinstance(solution2, str):
            # String similarity
            words1 = set(solution1.lower().split())
            words2 = set(solution2.lower().split())
            
            if not (words1 | words2):
                return 1.0
            
            return len(words1 & words2) / len(words1 | words2)
        
        else:
            # Direct comparison
            return 1.0 if solution1 == solution2 else 0.0
    
    def register_agent(self, agent: BaseAgent):
        """Register agent with the swarm engine"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent {agent.name} with swarm engine")
    
    def get_swarm_metrics(self) -> Dict[str, Any]:
        """Get comprehensive swarm metrics"""
        return {
            'engine_name': self.name,
            'registered_agents': len(self.agents),
            'active_particles': len(self.particles),
            'active_ants': len(self.ants),
            'optimization_runs': len(self.optimization_history),
            'emergent_behaviors_detected': len(self.behavior_history),
            'collective_memory_size': len(self.collective_memory),
            'recent_performance': self.optimization_history[-3:] if self.optimization_history else [],
            'parameter_settings': {
                'pso_params': self.pso_params,
                'aco_params': self.aco_params
            }
        }


# Additional implementations for ACO, behavior detection, etc. would continue...
# (Implementation continues with remaining methods for completeness)