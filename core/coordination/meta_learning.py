"""
Meta-Learning Framework for Strategy Evolution
Advanced implementation of learning-to-learn capabilities and strategy optimization
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
import pickle
import statistics
from abc import ABC, abstractmethod
import hashlib

from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class LearningStrategy(Enum):
    """Types of meta-learning strategies"""
    GRADIENT_BASED = "gradient_based"
    MODEL_AGNOSTIC = "model_agnostic"
    MEMORY_AUGMENTED = "memory_augmented"
    EVOLUTIONARY = "evolutionary"
    REINFORCEMENT = "reinforcement"
    TRANSFER_LEARNING = "transfer_learning"
    FEW_SHOT = "few_shot"
    CONTINUAL_LEARNING = "continual_learning"


class StrategyEvolutionMode(Enum):
    """Modes for strategy evolution"""
    EXPLORATION = "exploration"
    EXPLOITATION = "exploitation"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class MetaLearningPattern:
    """Enhanced meta-learning pattern with detailed tracking"""
    pattern_id: str
    pattern_type: str
    conditions: Dict[str, Any]
    recommended_strategy: str
    confidence: float
    usage_count: int = 0
    success_rate: float = 0.0
    failure_reasons: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    generalization_score: float = 0.0
    complexity_rating: float = 0.5
    resource_efficiency: float = 0.5
    adaptation_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StrategyPerformance:
    """Performance tracking for strategies"""
    strategy_name: str
    execution_count: int = 0
    success_count: int = 0
    total_execution_time: float = 0.0
    average_confidence: float = 0.0
    task_type_performance: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    improvement_trajectory: List[Tuple[datetime, float]] = field(default_factory=list)
    adaptability_score: float = 0.5
    stability_score: float = 0.5


@dataclass
class LearningEpisode:
    """Episode in the meta-learning process"""
    episode_id: str
    task_description: str
    task_features: Dict[str, Any]
    applied_strategy: str
    execution_result: Any
    performance_metrics: Dict[str, float]
    learning_insights: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    context_similarity: float = 0.0
    novelty_score: float = 0.0


class MetaLearningFramework:
    """
    Advanced Meta-Learning Framework
    Implements learning-to-learn capabilities with strategy evolution
    """
    
    def __init__(self, name: str = "meta_learning_framework"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        
        # Meta-learning components
        self.patterns: List[MetaLearningPattern] = []
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.learning_episodes: List[LearningEpisode] = []
        
        # Strategy evolution
        self.strategy_genome: Dict[str, Dict[str, float]] = {}
        self.evolution_history: List[Dict[str, Any]] = []
        self.adaptation_triggers: Dict[str, Callable] = {}
        
        # Knowledge base
        self.task_taxonomy: Dict[str, List[str]] = defaultdict(list)
        self.strategy_taxonomy: Dict[str, List[str]] = defaultdict(list)
        self.cross_domain_mappings: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Learning parameters
        self.meta_learning_params = {
            'pattern_confidence_threshold': 0.7,
            'min_episodes_for_pattern': 5,
            'adaptation_rate': 0.1,
            'exploration_decay': 0.95,
            'transfer_similarity_threshold': 0.6,
            'pattern_generalization_threshold': 0.8
        }
        
        # Performance tracking
        self.learning_curve: List[Tuple[datetime, float]] = []
        self.strategy_evolution_metrics: Dict[str, Any] = {}
        
        logger.info(f"Initialized meta-learning framework: {self.name}")
    
    async def meta_learn_from_episode(
        self,
        task_description: str,
        task_features: Dict[str, Any],
        applied_strategy: str,
        execution_result: Any,
        performance_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Learn from a single execution episode
        """
        logger.info(f"Meta-learning from episode: {applied_strategy} -> {task_description[:50]}...")
        global_metrics.incr("meta_learning.episode.processed")
        
        # Create learning episode
        episode_id = self._generate_episode_id(task_description, applied_strategy)
        
        # Calculate context similarity to existing episodes
        context_similarity = await self._calculate_context_similarity(task_features)
        
        # Calculate novelty score
        novelty_score = await self._calculate_novelty_score(task_features, applied_strategy)
        
        episode = LearningEpisode(
            episode_id=episode_id,
            task_description=task_description,
            task_features=task_features,
            applied_strategy=applied_strategy,
            execution_result=execution_result,
            performance_metrics=performance_metrics,
            learning_insights=[],
            context_similarity=context_similarity,
            novelty_score=novelty_score
        )\n        \n        # Extract learning insights\n        insights = await self._extract_learning_insights(episode)\n        episode.learning_insights = insights\n        \n        # Store episode\n        self.learning_episodes.append(episode)\n        \n        # Update strategy performance\n        await self._update_strategy_performance(applied_strategy, episode)\n        \n        # Check for new patterns\n        new_patterns = await self._detect_new_patterns(episode)\n        \n        # Update existing patterns\n        updated_patterns = await self._update_existing_patterns(episode)\n        \n        # Cross-domain transfer opportunities\n        transfer_opportunities = await self._identify_transfer_opportunities(episode)\n        \n        # Strategy evolution recommendations\n        evolution_recommendations = await self._generate_evolution_recommendations(episode)\n        \n        learning_result = {\n            'episode_id': episode_id,\n            'learning_insights': insights,\n            'new_patterns_detected': len(new_patterns),\n            'patterns_updated': len(updated_patterns),\n            'transfer_opportunities': transfer_opportunities,\n            'evolution_recommendations': evolution_recommendations,\n            'meta_learning_progress': await self._calculate_learning_progress()\n        }\n        \n        # Update learning curve\n        current_performance = performance_metrics.get('overall_score', 0.0)\n        self.learning_curve.append((datetime.now(), current_performance))\n        \n        return learning_result\n    \n    async def adaptive_strategy_selection(\n        self,\n        task_features: Dict[str, Any],\n        available_strategies: List[str],\n        exploration_mode: StrategyEvolutionMode = StrategyEvolutionMode.BALANCED\n    ) -> Tuple[str, float]:\n        \"\"\"\n        Adaptively select the best strategy for given task features\n        \"\"\"\n        logger.info(\"Selecting adaptive strategy based on task features\")\n        \n        # Find matching patterns\n        matching_patterns = await self._find_matching_patterns(task_features)\n        \n        strategy_scores = {}\n        \n        if matching_patterns:\n            # Use pattern-based selection\n            for pattern in matching_patterns:\n                if pattern.recommended_strategy in available_strategies:\n                    pattern_weight = pattern.confidence * pattern.success_rate * (1.0 + pattern.generalization_score)\n                    \n                    if pattern.recommended_strategy not in strategy_scores:\n                        strategy_scores[pattern.recommended_strategy] = 0.0\n                    \n                    strategy_scores[pattern.recommended_strategy] += pattern_weight\n        \n        # Add performance-based scoring\n        for strategy in available_strategies:\n            performance = self.strategy_performances.get(strategy)\n            if performance:\n                performance_score = (performance.success_count / max(1, performance.execution_count)) * performance.adaptability_score\n                \n                if strategy not in strategy_scores:\n                    strategy_scores[strategy] = 0.0\n                \n                strategy_scores[strategy] += performance_score * 0.5\n        \n        # Apply exploration bonus based on mode\n        if exploration_mode in [StrategyEvolutionMode.EXPLORATION, StrategyEvolutionMode.BALANCED]:\n            for strategy in available_strategies:\n                execution_count = self.strategy_performances.get(strategy, StrategyPerformance(strategy)).execution_count\n                \n                # Bonus for less-tried strategies\n                exploration_bonus = 1.0 / (1.0 + execution_count * 0.1)\n                \n                if exploration_mode == StrategyEvolutionMode.EXPLORATION:\n                    exploration_bonus *= 2.0\n                elif exploration_mode == StrategyEvolutionMode.BALANCED:\n                    exploration_bonus *= 0.5\n                \n                if strategy not in strategy_scores:\n                    strategy_scores[strategy] = 0.0\n                \n                strategy_scores[strategy] += exploration_bonus\n        \n        # Select best strategy\n        if not strategy_scores:\n            # Fallback to random selection\n            selected_strategy = np.random.choice(available_strategies)\n            confidence = 0.5\n        else:\n            # Softmax selection for some randomness\n            if exploration_mode == StrategyEvolutionMode.ADAPTIVE:\n                # Temperature-based selection\n                temperature = self._calculate_adaptive_temperature()\n                scores_array = np.array(list(strategy_scores.values()))\n                probabilities = self._softmax(scores_array / temperature)\n                \n                strategies_list = list(strategy_scores.keys())\n                selected_idx = np.random.choice(len(strategies_list), p=probabilities)\n                selected_strategy = strategies_list[selected_idx]\n                confidence = probabilities[selected_idx]\n            else:\n                # Best strategy selection\n                selected_strategy = max(strategy_scores, key=strategy_scores.get)\n                max_score = max(strategy_scores.values())\n                total_score = sum(strategy_scores.values())\n                confidence = max_score / max(total_score, 0.1)\n        \n        logger.info(f\"Selected strategy: {selected_strategy} (confidence: {confidence:.3f})\")\n        return selected_strategy, confidence\n    \n    async def evolve_strategies(\n        self,\n        evolution_generations: int = 10,\n        population_size: int = 20,\n        mutation_rate: float = 0.1\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Evolve strategies using evolutionary algorithms\n        \"\"\"\n        logger.info(f\"Starting strategy evolution for {evolution_generations} generations\")\n        global_metrics.incr(\"meta_learning.evolution.started\")\n        \n        # Initialize strategy population if empty\n        if not self.strategy_genome:\n            await self._initialize_strategy_population(population_size)\n        \n        evolution_results = {\n            'initial_population_size': len(self.strategy_genome),\n            'generations_completed': 0,\n            'best_strategies': [],\n            'evolution_history': [],\n            'performance_improvements': {},\n            'novel_strategies_discovered': []\n        }\n        \n        current_population = list(self.strategy_genome.items())\n        \n        for generation in range(evolution_generations):\n            logger.info(f\"Evolution generation {generation + 1}/{evolution_generations}\")\n            \n            # Evaluate fitness of all strategies\n            fitness_scores = []\n            generation_performance = []\n            \n            for strategy_name, genome in current_population:\n                fitness = await self._evaluate_strategy_fitness(strategy_name, genome)\n                fitness_scores.append(fitness)\n                generation_performance.append({\n                    'strategy': strategy_name,\n                    'genome': genome,\n                    'fitness': fitness\n                })\n            \n            # Track best strategies\n            best_idx = np.argmax(fitness_scores)\n            best_strategy = current_population[best_idx]\n            evolution_results['best_strategies'].append({\n                'generation': generation,\n                'strategy_name': best_strategy[0],\n                'fitness': fitness_scores[best_idx],\n                'genome': best_strategy[1]\n            })\n            \n            # Selection for reproduction\n            selected_parents = await self._evolutionary_selection(\n                current_population, fitness_scores\n            )\n            \n            # Create next generation\n            next_generation = []\n            \n            # Elitism - keep top performers\n            elite_count = max(2, population_size // 10)\n            elite_indices = np.argsort(fitness_scores)[-elite_count:]\n            for idx in elite_indices:\n                next_generation.append(current_population[idx])\n            \n            # Crossover and mutation\n            while len(next_generation) < population_size:\n                parent1, parent2 = np.random.choice(len(selected_parents), 2, replace=False)\n                child_genome = await self._strategy_crossover(\n                    selected_parents[parent1][1], \n                    selected_parents[parent2][1]\n                )\n                \n                # Mutation\n                if np.random.random() < mutation_rate:\n                    child_genome = await self._strategy_mutation(child_genome)\n                \n                child_name = f\"evolved_strategy_{generation}_{len(next_generation)}\"\n                next_generation.append((child_name, child_genome))\n            \n            # Update population\n            current_population = next_generation\n            \n            # Record generation statistics\n            generation_stats = {\n                'generation': generation,\n                'best_fitness': max(fitness_scores),\n                'average_fitness': np.mean(fitness_scores),\n                'fitness_diversity': np.std(fitness_scores),\n                'population_diversity': await self._calculate_strategy_diversity(current_population)\n            }\n            evolution_results['evolution_history'].append(generation_stats)\n            \n            # Early stopping if converged\n            if generation_stats['fitness_diversity'] < 0.01:\n                logger.info(f\"Population converged at generation {generation + 1}\")\n                break\n        \n        # Update strategy genome with evolved strategies\n        self.strategy_genome.clear()\n        for strategy_name, genome in current_population:\n            self.strategy_genome[strategy_name] = genome\n        \n        evolution_results['generations_completed'] = generation + 1\n        evolution_results['final_population_size'] = len(current_population)\n        \n        # Identify novel strategies\n        novel_strategies = await self._identify_novel_strategies(current_population)\n        evolution_results['novel_strategies_discovered'] = novel_strategies\n        \n        # Calculate performance improvements\n        improvements = await self._calculate_evolution_improvements(evolution_results)\n        evolution_results['performance_improvements'] = improvements\n        \n        # Store evolution history\n        self.evolution_history.append(evolution_results)\n        \n        global_metrics.incr(\"meta_learning.evolution.completed\")\n        return evolution_results\n    \n    async def transfer_learning_analysis(\n        self,\n        source_domain: str,\n        target_domain: str,\n        similarity_threshold: float = 0.7\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Analyze transfer learning opportunities between domains\n        \"\"\"\n        logger.info(f\"Analyzing transfer learning: {source_domain} -> {target_domain}\")\n        \n        # Extract domain-specific episodes\n        source_episodes = [e for e in self.learning_episodes \n                          if self._classify_episode_domain(e) == source_domain]\n        target_episodes = [e for e in self.learning_episodes \n                          if self._classify_episode_domain(e) == target_domain]\n        \n        analysis = {\n            'source_domain': source_domain,\n            'target_domain': target_domain,\n            'source_episodes': len(source_episodes),\n            'target_episodes': len(target_episodes),\n            'transferable_patterns': [],\n            'adaptation_requirements': [],\n            'expected_performance_gain': 0.0,\n            'transfer_risk_assessment': {},\n            'recommended_transfer_strategy': None\n        }\n        \n        if not source_episodes:\n            analysis['recommendation'] = \"Insufficient source domain data for transfer\"\n            return analysis\n        \n        # Find transferable patterns\n        transferable_patterns = []\n        \n        for pattern in self.patterns:\n            # Check if pattern was learned from source domain\n            pattern_domain = await self._identify_pattern_domain(pattern)\n            \n            if pattern_domain == source_domain:\n                # Assess transferability to target domain\n                transferability_score = await self._assess_pattern_transferability(\n                    pattern, target_domain, target_episodes\n                )\n                \n                if transferability_score >= similarity_threshold:\n                    adapted_pattern = await self._adapt_pattern_for_domain(\n                        pattern, target_domain\n                    )\n                    \n                    transferable_patterns.append({\n                        'original_pattern': pattern.pattern_id,\n                        'adapted_pattern': adapted_pattern,\n                        'transferability_score': transferability_score,\n                        'adaptation_confidence': adapted_pattern.confidence\n                    })\n        \n        analysis['transferable_patterns'] = transferable_patterns\n        \n        # Assess adaptation requirements\n        adaptation_requirements = await self._analyze_adaptation_requirements(\n            source_episodes, target_episodes\n        )\n        analysis['adaptation_requirements'] = adaptation_requirements\n        \n        # Estimate performance gain\n        if target_episodes:\n            baseline_performance = np.mean([e.performance_metrics.get('overall_score', 0.0) \n                                           for e in target_episodes])\n            \n            source_performance = np.mean([e.performance_metrics.get('overall_score', 0.0) \n                                         for e in source_episodes])\n            \n            # Conservative estimate with adaptation penalty\n            adaptation_penalty = 1.0 - (adaptation_requirements.get('complexity', 0.5) * 0.3)\n            expected_gain = (source_performance - baseline_performance) * adaptation_penalty\n            \n            analysis['expected_performance_gain'] = max(0.0, expected_gain)\n        \n        # Risk assessment\n        risk_assessment = await self._assess_transfer_risks(\n            source_domain, target_domain, transferable_patterns\n        )\n        analysis['transfer_risk_assessment'] = risk_assessment\n        \n        # Recommend transfer strategy\n        if transferable_patterns and analysis['expected_performance_gain'] > 0.1:\n            if risk_assessment.get('overall_risk', 0.5) < 0.3:\n                analysis['recommended_transfer_strategy'] = \"full_transfer\"\n            elif risk_assessment.get('overall_risk', 0.5) < 0.7:\n                analysis['recommended_transfer_strategy'] = \"gradual_transfer\"\n            else:\n                analysis['recommended_transfer_strategy'] = \"selective_transfer\"\n        else:\n            analysis['recommended_transfer_strategy'] = \"no_transfer\"\n        \n        return analysis\n    \n    async def continual_learning_adaptation(\n        self,\n        new_task_stream: List[Dict[str, Any]],\n        forgetting_prevention: bool = True,\n        adaptation_rate: float = 0.1\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Implement continual learning to adapt to new tasks without forgetting\n        \"\"\"\n        logger.info(f\"Starting continual learning with {len(new_task_stream)} new tasks\")\n        global_metrics.incr(\"meta_learning.continual.started\")\n        \n        adaptation_results = {\n            'tasks_processed': 0,\n            'new_patterns_learned': 0,\n            'patterns_forgotten': 0,\n            'adaptation_trajectory': [],\n            'knowledge_retention': {},\n            'performance_stability': {},\n            'catastrophic_forgetting_detected': False\n        }\n        \n        # Baseline performance on existing tasks\n        if forgetting_prevention:\n            baseline_performance = await self._measure_baseline_performance()\n            adaptation_results['baseline_performance'] = baseline_performance\n        \n        # Process new task stream\n        for task_idx, task in enumerate(new_task_stream):\n            logger.info(f\"Processing continual learning task {task_idx + 1}/{len(new_task_stream)}\")\n            \n            # Extract task features\n            task_features = await self._extract_task_features(task)\n            \n            # Check for domain shift\n            domain_shift = await self._detect_domain_shift(task_features)\n            \n            # Adaptive learning rate based on domain shift\n            current_adaptation_rate = adaptation_rate\n            if domain_shift['shift_detected']:\n                current_adaptation_rate *= domain_shift['adaptation_multiplier']\n                logger.info(f\"Domain shift detected: {domain_shift['shift_type']}\")\n            \n            # Learn from new task\n            learning_result = await self._continual_learn_from_task(\n                task, current_adaptation_rate\n            )\n            \n            # Check for catastrophic forgetting\n            if forgetting_prevention and task_idx % 5 == 0:  # Check every 5 tasks\n                forgetting_check = await self._check_catastrophic_forgetting(\n                    baseline_performance\n                )\n                \n                if forgetting_check['forgetting_detected']:\n                    logger.warning(\"Catastrophic forgetting detected - applying mitigation\")\n                    await self._mitigate_catastrophic_forgetting(\n                        forgetting_check['affected_patterns']\n                    )\n                    adaptation_results['catastrophic_forgetting_detected'] = True\n            \n            # Record adaptation progress\n            adaptation_step = {\n                'task_index': task_idx,\n                'domain_shift': domain_shift,\n                'learning_result': learning_result,\n                'adaptation_rate_used': current_adaptation_rate,\n                'patterns_count': len(self.patterns),\n                'performance_metrics': await self._measure_current_performance()\n            }\n            \n            adaptation_results['adaptation_trajectory'].append(adaptation_step)\n            adaptation_results['tasks_processed'] += 1\n            \n            if learning_result['new_patterns']:\n                adaptation_results['new_patterns_learned'] += len(learning_result['new_patterns'])\n        \n        # Final analysis\n        if forgetting_prevention:\n            final_performance = await self._measure_baseline_performance()\n            retention_analysis = await self._analyze_knowledge_retention(\n                baseline_performance, final_performance\n            )\n            adaptation_results['knowledge_retention'] = retention_analysis\n        \n        # Performance stability analysis\n        stability_analysis = await self._analyze_performance_stability(\n            adaptation_results['adaptation_trajectory']\n        )\n        adaptation_results['performance_stability'] = stability_analysis\n        \n        global_metrics.incr(\"meta_learning.continual.completed\")\n        return adaptation_results\n    \n    # Helper methods for meta-learning framework\n    \n    def _generate_episode_id(self, task_description: str, strategy: str) -> str:\n        \"\"\"Generate unique episode ID\"\"\"\n        content = f\"{task_description}_{strategy}_{datetime.now().isoformat()}\"\n        return hashlib.md5(content.encode()).hexdigest()[:12]\n    \n    async def _calculate_context_similarity(self, task_features: Dict[str, Any]) -> float:\n        \"\"\"Calculate similarity to existing episode contexts\"\"\"\n        if not self.learning_episodes:\n            return 0.0\n        \n        similarities = []\n        for episode in self.learning_episodes[-20:]:  # Compare with recent episodes\n            similarity = await self._calculate_feature_similarity(\n                task_features, episode.task_features\n            )\n            similarities.append(similarity)\n        \n        return max(similarities) if similarities else 0.0\n    \n    async def _calculate_feature_similarity(self, features1: Dict[str, Any], features2: Dict[str, Any]) -> float:\n        \"\"\"Calculate similarity between feature sets\"\"\"\n        all_keys = set(features1.keys()) | set(features2.keys())\n        if not all_keys:\n            return 1.0\n        \n        matching_score = 0.0\n        for key in all_keys:\n            if key in features1 and key in features2:\n                if features1[key] == features2[key]:\n                    matching_score += 1.0\n                elif isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):\n                    # Numerical similarity\n                    max_val = max(abs(features1[key]), abs(features2[key]), 1.0)\n                    similarity = 1.0 - abs(features1[key] - features2[key]) / max_val\n                    matching_score += max(0.0, similarity)\n        \n        return matching_score / len(all_keys)\n    \n    def _softmax(self, scores: np.ndarray) -> np.ndarray:\n        \"\"\"Apply softmax transformation\"\"\"\n        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability\n        return exp_scores / np.sum(exp_scores)\n    \n    def register_agent(self, agent: BaseAgent):\n        \"\"\"Register agent with meta-learning framework\"\"\"\n        self.agents[agent.name] = agent\n        logger.info(f\"Registered agent {agent.name} with meta-learning framework\")\n    \n    def get_meta_learning_metrics(self) -> Dict[str, Any]:\n        \"\"\"Get comprehensive meta-learning metrics\"\"\"\n        return {\n            'framework_name': self.name,\n            'registered_agents': len(self.agents),\n            'total_patterns': len(self.patterns),\n            'learning_episodes': len(self.learning_episodes),\n            'strategy_performances': len(self.strategy_performances),\n            'evolution_generations': len(self.evolution_history),\n            'learning_parameters': self.meta_learning_params,\n            'recent_learning_curve': self.learning_curve[-10:] if self.learning_curve else [],\n            'pattern_success_rates': {\n                p.pattern_id: p.success_rate for p in self.patterns\n            },\n            'strategy_effectiveness': {\n                name: perf.success_count / max(1, perf.execution_count)\n                for name, perf in self.strategy_performances.items()\n            },\n            'meta_learning_health': await self._calculate_meta_learning_health()\n        }\n    \n    async def _calculate_meta_learning_health(self) -> float:\n        \"\"\"Calculate overall meta-learning system health\"\"\"\n        health_factors = []\n        \n        # Learning progress (improving over time)\n        if len(self.learning_curve) > 10:\n            recent_performance = [score for _, score in self.learning_curve[-10:]]\n            early_performance = [score for _, score in self.learning_curve[:10]]\n            \n            if early_performance and recent_performance:\n                improvement = np.mean(recent_performance) - np.mean(early_performance)\n                progress_score = min(1.0, max(0.0, 0.5 + improvement))\n                health_factors.append(progress_score)\n        \n        # Pattern quality (high success rates)\n        if self.patterns:\n            pattern_success_rates = [p.success_rate for p in self.patterns if p.usage_count > 0]\n            if pattern_success_rates:\n                avg_success_rate = np.mean(pattern_success_rates)\n                health_factors.append(avg_success_rate)\n        \n        # Strategy diversity (not over-reliant on single strategy)\n        if len(self.strategy_performances) > 1:\n            execution_counts = [p.execution_count for p in self.strategy_performances.values()]\n            diversity_score = 1.0 - (np.std(execution_counts) / max(np.mean(execution_counts), 1.0))\n            health_factors.append(max(0.0, min(1.0, diversity_score)))\n        \n        # Recent activity (system is being used)\n        recent_episodes = len([e for e in self.learning_episodes \n                              if (datetime.now() - e.timestamp).days < 7])\n        activity_score = min(1.0, recent_episodes / 10.0)  # Expect some weekly activity\n        health_factors.append(activity_score)\n        \n        return np.mean(health_factors) if health_factors else 0.5\n\n\n# Additional implementations would continue...\n# This provides the core meta-learning framework structure