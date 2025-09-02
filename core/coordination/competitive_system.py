"""
Competitive Agent Selection System
Advanced implementation of agent tournaments, performance ranking, and competitive optimization
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
import hashlib
from abc import ABC, abstractmethod

from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class CompetitionType(Enum):
    """Types of agent competitions"""
    TOURNAMENT = "tournament"
    ROUND_ROBIN = "round_robin"
    ELIMINATION = "elimination"
    SWISS_SYSTEM = "swiss_system"
    LADDER = "ladder"
    BATTLE_ROYALE = "battle_royale"


class SelectionCriteria(Enum):
    """Criteria for selecting winning agents"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    FASTEST_PROCESSING = "fastest_processing"
    BEST_VALUE = "best_value"
    CONSENSUS_VOTING = "consensus_voting"
    VALIDATION_SCORE = "validation_score"
    MULTI_OBJECTIVE = "multi_objective"
    ADAPTIVE_RANKING = "adaptive_ranking"


@dataclass
class CompetitiveResult:
    """Enhanced result from competitive agent processing"""
    agent_name: str
    result: Any
    confidence_score: float
    processing_time_ms: int
    cost_estimate: float
    method_used: str
    validation_score: float = 0.0
    innovation_score: float = 0.0
    resource_efficiency: float = 0.0
    consistency_rating: float = 0.0
    collaboration_score: float = 0.0


@dataclass
class AgentTournament:
    """Tournament structure for competitive agents"""
    tournament_id: str
    competition_type: CompetitionType
    participants: List[str]
    brackets: List[List[Tuple[str, str]]] = field(default_factory=list)
    results: Dict[str, List[CompetitiveResult]] = field(default_factory=dict)
    rankings: List[Tuple[str, float]] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentPerformanceProfile:
    """Comprehensive performance profile for agents"""
    agent_name: str
    total_competitions: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    average_confidence: float = 0.0
    average_processing_time: float = 0.0
    average_cost: float = 0.0
    specialization_scores: Dict[str, float] = field(default_factory=dict)
    recent_performance: deque = field(default_factory=lambda: deque(maxlen=20))
    elo_rating: float = 1200.0  # Starting ELO rating
    peak_performance: Optional[CompetitiveResult] = None
    weaknesses: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


class CompetitiveSystem:
    """
    Advanced Competitive Agent Selection System
    Implements tournaments, ranking systems, and performance analytics
    """
    
    def __init__(self, name: str = "competitive_system"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.performance_profiles: Dict[str, AgentPerformanceProfile] = {}
        
        # Tournament management
        self.active_tournaments: Dict[str, AgentTournament] = {}
        self.tournament_history: List[AgentTournament] = []
        
        # Ranking systems
        self.global_rankings: Dict[str, float] = {}  # ELO-based rankings
        self.category_rankings: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Performance tracking
        self.competition_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        
        # Advanced selection algorithms
        self.selection_algorithms: Dict[str, Callable] = {
            'genetic_algorithm': self._genetic_selection,
            'pareto_optimization': self._pareto_optimal_selection,
            'machine_learning': self._ml_based_selection,
            'ensemble_voting': self._ensemble_voting_selection
        }
        
        # Dynamic difficulty adjustment
        self.difficulty_adjustments: Dict[str, float] = defaultdict(float)
        
        logger.info(f"Initialized competitive system: {self.name}")
    
    async def create_tournament(
        self,
        tournament_id: str,
        competition_type: CompetitionType,
        task_description: str,
        participants: Optional[List[str]] = None,
        selection_criteria: SelectionCriteria = SelectionCriteria.MULTI_OBJECTIVE,
        rounds: int = 3
    ) -> AgentTournament:
        """
        Create and initialize a new tournament
        """
        logger.info(f"Creating tournament: {tournament_id} ({competition_type.value})")
        global_metrics.incr("competitive.tournament.created")
        
        # Select participants if not provided
        if participants is None:
            participants = await self._select_tournament_participants(
                task_description, competition_type
            )
        
        # Validate participants
        valid_participants = [p for p in participants if p in self.agents]
        if len(valid_participants) < 2:
            raise ValueError("At least 2 valid participants required for tournament")
        
        # Create tournament structure
        tournament = AgentTournament(
            tournament_id=tournament_id,
            competition_type=competition_type,
            participants=valid_participants,
            metadata={
                'task_description': task_description,
                'selection_criteria': selection_criteria,
                'rounds': rounds,
                'difficulty_level': await self._estimate_task_difficulty(task_description)
            }
        )
        
        # Generate tournament brackets
        await self._generate_tournament_brackets(tournament)
        
        self.active_tournaments[tournament_id] = tournament
        
        logger.info(f"Tournament {tournament_id} created with {len(valid_participants)} participants")
        return tournament
    
    async def run_tournament(
        self,
        tournament_id: str,
        task_requirements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a complete tournament
        """
        if tournament_id not in self.active_tournaments:
            raise ValueError(f"Tournament {tournament_id} not found")
        
        tournament = self.active_tournaments[tournament_id]
        logger.info(f"Starting tournament: {tournament_id}")
        global_metrics.incr("competitive.tournament.started")
        
        tournament_results = {
            'tournament_id': tournament_id,
            'competition_type': tournament.competition_type.value,
            'rounds_completed': 0,
            'matches_played': 0,
            'participant_results': {},
            'final_rankings': [],
            'performance_insights': {},
            'innovation_discoveries': []
        }
        
        try:
            if tournament.competition_type == CompetitionType.TOURNAMENT:
                results = await self._run_elimination_tournament(tournament, task_requirements)
            elif tournament.competition_type == CompetitionType.ROUND_ROBIN:
                results = await self._run_round_robin_tournament(tournament, task_requirements)
            elif tournament.competition_type == CompetitionType.SWISS_SYSTEM:
                results = await self._run_swiss_tournament(tournament, task_requirements)
            elif tournament.competition_type == CompetitionType.BATTLE_ROYALE:
                results = await self._run_battle_royale(tournament, task_requirements)
            else:
                results = await self._run_elimination_tournament(tournament, task_requirements)
            
            tournament_results.update(results)
            tournament.end_time = datetime.now()
            
            # Update agent performance profiles
            await self._update_performance_profiles(tournament, results)
            
            # Update rankings
            await self._update_global_rankings(tournament, results)
            
            # Detect innovations and improvements
            innovations = await self._detect_tournament_innovations(tournament, results)
            tournament_results['innovation_discoveries'] = innovations
            
            # Archive tournament
            self.tournament_history.append(tournament)
            del self.active_tournaments[tournament_id]
            
            global_metrics.incr("competitive.tournament.completed")
            
        except Exception as e:
            logger.error(f"Tournament {tournament_id} failed: {e}")
            global_metrics.incr("competitive.tournament.failed")
            raise
        
        return tournament_results
    
    async def competitive_agent_selection(
        self,
        task_description: str,
        task_requirements: Dict[str, Any],
        selection_criteria: SelectionCriteria = SelectionCriteria.MULTI_OBJECTIVE,
        num_competitors: int = 5,
        competition_type: str = "quick_competition"
    ) -> CompetitiveResult:
        """
        Quick competitive selection without full tournament structure
        """
        logger.info(f"Starting competitive selection: {task_description[:50]}...")
        global_metrics.incr("competitive.quick_selection.started")
        
        # Select competitors
        competitors = await self._select_competitors_for_task(
            task_description, task_requirements, num_competitors
        )
        
        if len(competitors) < 2:
            # Fallback to single best agent
            if competitors:
                return await self._execute_single_agent(competitors[0], task_description, task_requirements)
            else:
                raise ValueError("No suitable competitors found")
        
        # Execute competitive processing
        competition_results = []
        start_time = datetime.now()
        
        # Run competitors in parallel
        competitor_tasks = []
        for agent in competitors:
            competitor_tasks.append(
                self._execute_competitive_agent(agent, task_description, task_requirements, competition_type)
            )
        
        # Execute and collect results
        results = await asyncio.gather(*competitor_tasks, return_exceptions=True)
        
        # Process results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Competitor {competitors[i].name} failed: {result}")
                continue
            
            # Enhance result with additional metrics
            enhanced_result = await self._enhance_competitive_result(result, competitors[i])
            valid_results.append(enhanced_result)
        
        if not valid_results:
            raise RuntimeError("All competitors failed")
        
        # Select winner based on criteria
        winner = await self._select_winner(valid_results, selection_criteria)
        
        # Record competition
        competition_record = {
            'task_description': task_description,
            'competitors': [agent.name for agent in competitors],
            'results': valid_results,
            'winner': winner.agent_name,
            'selection_criteria': selection_criteria.value,
            'total_time_ms': (datetime.now() - start_time).total_seconds() * 1000,
            'timestamp': datetime.now()
        }
        
        self.competition_history.append(competition_record)
        
        # Update performance profiles
        await self._update_quick_competition_profiles(competitors, valid_results, winner)
        
        logger.info(f"Competition completed. Winner: {winner.agent_name} (score: {winner.confidence_score:.3f})")
        global_metrics.incr("competitive.quick_selection.completed")
        
        return winner
    
    async def genetic_algorithm_optimization(
        self,
        population_size: int = 20,
        generations: int = 10,
        task_description: str = "optimization_task",
        task_requirements: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Use genetic algorithm to evolve optimal agent configurations
        """
        logger.info(f"Starting genetic algorithm optimization with {population_size} individuals")
        global_metrics.incr("competitive.genetic.started")
        
        if task_requirements is None:
            task_requirements = {}
        
        # Initialize population
        population = await self._initialize_genetic_population(population_size, task_description)
        
        evolution_history = []
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            logger.info(f"Genetic algorithm generation {generation + 1}/{generations}")
            
            # Evaluate population fitness
            fitness_scores = []
            generation_results = []
            
            for individual in population:
                fitness, result = await self._evaluate_genetic_fitness(
                    individual, task_description, task_requirements
                )
                fitness_scores.append(fitness)
                generation_results.append(result)
                
                # Track best individual
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
            
            # Record generation statistics
            generation_stats = {
                'generation': generation,
                'best_fitness': max(fitness_scores),
                'average_fitness': np.mean(fitness_scores),
                'fitness_std': np.std(fitness_scores),
                'diversity': await self._calculate_population_diversity(population)
            }
            evolution_history.append(generation_stats)
            
            # Selection for reproduction
            selected_parents = await self._genetic_selection(population, fitness_scores)
            
            # Create next generation
            new_population = []
            
            # Elitism - keep best individuals
            elite_count = max(2, population_size // 10)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            for idx in elite_indices:
                new_population.append(population[idx].copy())
            
            # Crossover and mutation
            while len(new_population) < population_size:
                parent1, parent2 = np.random.choice(selected_parents, 2, replace=False)
                child = await self._genetic_crossover(parent1, parent2)
                child = await self._genetic_mutation(child, generation / generations)
                new_population.append(child)
            
            population = new_population
            
            # Early stopping if converged
            if generation_stats['fitness_std'] < 0.01:
                logger.info(f"Population converged at generation {generation + 1}")
                break
        
        result = {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'generations_completed': generation + 1,
            'evolution_history': evolution_history,
            'final_population': population,
            'convergence_achieved': generation_stats['fitness_std'] < 0.01,
            'optimization_insights': await self._analyze_genetic_insights(evolution_history)
        }
        
        global_metrics.incr("competitive.genetic.completed")
        return result
    
    async def adaptive_difficulty_system(
        self,
        task_description: str,
        current_performance_data: List[Dict[str, Any]],
        target_success_rate: float = 0.7
    ) -> Dict[str, Any]:
        """
        Adaptive system that adjusts task difficulty based on agent performance
        """
        logger.info("Running adaptive difficulty adjustment")
        
        # Analyze current performance trends
        performance_analysis = await self._analyze_performance_trends(current_performance_data)
        
        # Calculate required difficulty adjustment
        current_success_rate = performance_analysis['success_rate']
        difficulty_adjustment = await self._calculate_difficulty_adjustment(
            current_success_rate, target_success_rate
        )
        
        # Apply difficulty adjustment
        adjusted_requirements = await self._apply_difficulty_adjustment(
            task_description, difficulty_adjustment
        )
        
        # Test adjustment with sample agents
        validation_results = await self._validate_difficulty_adjustment(
            adjusted_requirements, task_description
        )
        
        result = {
            'original_success_rate': current_success_rate,
            'target_success_rate': target_success_rate,
            'difficulty_adjustment': difficulty_adjustment,
            'adjusted_requirements': adjusted_requirements,
            'validation_results': validation_results,
            'recommendation': await self._generate_difficulty_recommendation(
                performance_analysis, difficulty_adjustment
            )
        }
        
        # Update global difficulty tracking
        task_category = await self._categorize_task(task_description)
        self.difficulty_adjustments[task_category] += difficulty_adjustment
        
        return result
    
    # Helper methods for competitive system
    
    async def _select_tournament_participants(
        self, 
        task_description: str, 
        competition_type: CompetitionType
    ) -> List[str]:
        """Select appropriate participants for tournament"""
        if not self.agents:
            return []
        
        # Analyze task to determine suitable agents
        task_category = await self._categorize_task(task_description)
        
        # Get agents with relevant experience
        suitable_agents = []
        for agent_name, agent in self.agents.items():
            profile = self.performance_profiles.get(agent_name)
            if profile:
                # Check specialization match
                specialization_score = profile.specialization_scores.get(task_category, 0.0)
                overall_performance = profile.wins / max(1, profile.total_competitions)
                
                suitability_score = (specialization_score * 0.7) + (overall_performance * 0.3)
            else:
                # New agent - give it a chance
                suitability_score = 0.5
            
            suitable_agents.append((agent_name, suitability_score))
        
        # Sort by suitability and select top candidates
        suitable_agents.sort(key=lambda x: x[1], reverse=True)
        
        # Tournament size based on competition type
        if competition_type == CompetitionType.BATTLE_ROYALE:
            max_participants = min(16, len(suitable_agents))
        elif competition_type == CompetitionType.ROUND_ROBIN:
            max_participants = min(8, len(suitable_agents))  # Round robin gets expensive
        else:
            max_participants = min(12, len(suitable_agents))
        
        return [name for name, _ in suitable_agents[:max_participants]]
    
    async def _generate_tournament_brackets(self, tournament: AgentTournament):
        """Generate brackets for tournament"""
        participants = tournament.participants.copy()
        np.random.shuffle(participants)  # Randomize seeding
        
        if tournament.competition_type == CompetitionType.TOURNAMENT:
            # Single elimination brackets
            brackets = []
            current_round = participants
            
            while len(current_round) > 1:
                round_matches = []
                for i in range(0, len(current_round), 2):
                    if i + 1 < len(current_round):
                        round_matches.append((current_round[i], current_round[i + 1]))
                    else:
                        # Bye for odd participant
                        round_matches.append((current_round[i], None))
                
                brackets.append(round_matches)
                # Prepare next round (winners will be determined during execution)
                next_round = []
                for match in round_matches:
                    next_round.append("TBD")  # To be determined
                current_round = next_round
            
            tournament.brackets = brackets
        
        elif tournament.competition_type == CompetitionType.ROUND_ROBIN:
            # All vs all matches
            matches = []
            for i, agent1 in enumerate(participants):
                for agent2 in participants[i + 1:]:
                    matches.append((agent1, agent2))
            tournament.brackets = [matches]  # All matches in one "round"
    
    async def _run_elimination_tournament(
        self, 
        tournament: AgentTournament, 
        task_requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run elimination tournament"""
        results = {
            'rounds_completed': 0,
            'matches_played': 0,
            'participant_results': {},
            'elimination_order': [],
            'final_rankings': []
        }
        
        current_participants = tournament.participants.copy()
        round_num = 0
        
        while len(current_participants) > 1:
            round_num += 1
            logger.info(f"Tournament round {round_num} with {len(current_participants)} participants")
            
            round_winners = []
            round_matches = 0
            
            # Pair participants for this round
            for i in range(0, len(current_participants), 2):
                if i + 1 < len(current_participants):
                    agent1_name = current_participants[i]
                    agent2_name = current_participants[i + 1]
                    
                    # Head-to-head match
                    winner = await self._head_to_head_match(
                        self.agents[agent1_name], 
                        self.agents[agent2_name],
                        tournament.metadata['task_description'],
                        task_requirements or {}
                    )
                    
                    round_winners.append(winner)
                    loser = agent1_name if winner == agent2_name else agent2_name
                    results['elimination_order'].append(loser)
                    round_matches += 1
                    
                else:
                    # Bye - participant advances automatically
                    round_winners.append(current_participants[i])
            
            current_participants = round_winners
            results['matches_played'] += round_matches
        
        # Final winner
        if current_participants:
            winner = current_participants[0]
            results['final_rankings'] = [winner] + list(reversed(results['elimination_order']))
        
        results['rounds_completed'] = round_num
        return results
    
    async def _head_to_head_match(
        self,
        agent1: BaseAgent,
        agent2: BaseAgent,
        task_description: str,
        task_requirements: Dict[str, Any]
    ) -> str:
        """Execute head-to-head match between two agents"""
        logger.info(f"Head-to-head: {agent1.name} vs {agent2.name}")
        
        # Execute both agents in parallel
        tasks = [
            self._execute_competitive_agent(agent1, task_description, task_requirements, "head_to_head"),
            self._execute_competitive_agent(agent2, task_description, task_requirements, "head_to_head")
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle failures
        result1, result2 = results
        if isinstance(result1, Exception) and isinstance(result2, Exception):
            # Both failed - random winner
            return np.random.choice([agent1.name, agent2.name])
        elif isinstance(result1, Exception):
            return agent2.name
        elif isinstance(result2, Exception):
            return agent1.name
        
        # Compare results using multi-criteria
        score1 = await self._calculate_match_score(result1)
        score2 = await self._calculate_match_score(result2)
        
        return agent1.name if score1 > score2 else agent2.name
    
    async def _calculate_match_score(self, result: CompetitiveResult) -> float:
        """Calculate overall match score from competitive result"""
        # Multi-criteria scoring
        weights = {
            'confidence': 0.3,
            'validation': 0.25,
            'efficiency': 0.2,
            'innovation': 0.15,
            'consistency': 0.1
        }
        
        score = (
            result.confidence_score * weights['confidence'] +
            result.validation_score * weights['validation'] +
            result.resource_efficiency * weights['efficiency'] +
            result.innovation_score * weights['innovation'] +
            result.consistency_rating * weights['consistency']
        )
        
        return score
    
    def register_agent(self, agent: BaseAgent):
        """Register agent with competitive system"""
        self.agents[agent.name] = agent
        
        # Initialize performance profile
        if agent.name not in self.performance_profiles:
            self.performance_profiles[agent.name] = AgentPerformanceProfile(
                agent_name=agent.name
            )
        
        # Initialize global ranking
        if agent.name not in self.global_rankings:
            self.global_rankings[agent.name] = 1200.0  # Starting ELO
        
        logger.info(f"Registered agent {agent.name} with competitive system")
    
    def get_competitive_metrics(self) -> Dict[str, Any]:
        """Get comprehensive competitive system metrics"""
        return {
            'system_name': self.name,
            'registered_agents': len(self.agents),
            'active_tournaments': len(self.active_tournaments),
            'completed_tournaments': len(self.tournament_history),
            'total_competitions': len(self.competition_history),
            'global_rankings': dict(sorted(self.global_rankings.items(), key=lambda x: x[1], reverse=True)),
            'performance_profiles': {
                name: {
                    'total_competitions': profile.total_competitions,
                    'win_rate': profile.wins / max(1, profile.total_competitions),
                    'elo_rating': profile.elo_rating,
                    'specializations': dict(profile.specialization_scores)
                }
                for name, profile in self.performance_profiles.items()
            },
            'recent_performance': self.competition_history[-10:] if self.competition_history else [],
            'system_health': await self._calculate_competitive_system_health()
        }
    
    async def _calculate_competitive_system_health(self) -> float:
        """Calculate overall system health score"""
        health_factors = []
        
        # Agent participation rate
        if self.agents:
            participating_agents = len([p for p in self.performance_profiles.values() if p.total_competitions > 0])
            participation_rate = participating_agents / len(self.agents)
            health_factors.append(participation_rate)
        
        # Competition frequency
        if self.competition_history:
            recent_competitions = len([c for c in self.competition_history 
                                     if (datetime.now() - c['timestamp']).days < 7])
            activity_score = min(1.0, recent_competitions / 10.0)  # Expect up to 10 per week
            health_factors.append(activity_score)
        
        # Performance diversity (good if not all agents have same performance)
        if len(self.global_rankings) > 1:
            rating_std = np.std(list(self.global_rankings.values()))
            diversity_score = min(1.0, rating_std / 200.0)  # Normalize by expected std
            health_factors.append(diversity_score)
        
        return np.mean(health_factors) if health_factors else 0.5