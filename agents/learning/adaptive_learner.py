"""
Adaptive Learning System: Phase 6 - Pattern Extraction and Strategy Optimization
Features:
- Pattern extraction from agent experiences
- Strategy optimization based on success rates  
- Transfer learning between domains
- Meta-learning for improved learning strategies
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict, deque
from abc import ABC, abstractmethod
import pickle
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class LearningType(Enum):
    """Types of learning approaches"""
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    REINFORCEMENT = "reinforcement"
    TRANSFER = "transfer"
    META = "meta"


class PatternType(Enum):
    """Types of patterns that can be extracted"""
    TASK_SUCCESS = "task_success"
    AGENT_PERFORMANCE = "agent_performance"
    EXECUTION_STRATEGY = "execution_strategy"
    ERROR_PATTERN = "error_pattern"
    RESOURCE_USAGE = "resource_usage"
    TEMPORAL = "temporal"


@dataclass
class Experience:
    """Represents a learning experience"""
    experience_id: str
    agent_id: str
    task_description: str
    context: Dict[str, Any]
    actions_taken: List[str]
    outcome: Dict[str, Any]
    success: bool
    quality_score: float
    execution_time: float
    resources_used: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    domain: str = "general"
    
    @property
    def feature_vector(self) -> Dict[str, Any]:
        """Extract features for ML algorithms"""
        return {
            'task_complexity': len(self.task_description.split()),
            'context_size': len(self.context),
            'actions_count': len(self.actions_taken),
            'success': int(self.success),
            'quality_score': self.quality_score,
            'execution_time': self.execution_time,
            'domain': self.domain,
            'hour_of_day': self.timestamp.hour,
            'day_of_week': self.timestamp.weekday()
        }


@dataclass
class Pattern:
    """Represents a discovered pattern"""
    pattern_id: str
    pattern_type: PatternType
    description: str
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    confidence: float
    support: int  # Number of experiences supporting this pattern
    discovered_at: datetime = field(default_factory=datetime.now)
    domains: Set[str] = field(default_factory=set)
    
    def matches(self, experience: Experience) -> bool:
        """Check if experience matches this pattern"""
        for key, expected_value in self.conditions.items():
            if key in experience.feature_vector:
                actual_value = experience.feature_vector[key]
                if isinstance(expected_value, dict) and 'min' in expected_value:
                    if actual_value < expected_value['min'] or actual_value > expected_value.get('max', float('inf')):
                        return False
                elif actual_value != expected_value:
                    return False
        return True


@dataclass
class LearningStrategy:
    """Represents a learning strategy with its performance"""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    performance_history: List[float] = field(default_factory=list)
    success_rate: float = 0.0
    domains_applied: Set[str] = field(default_factory=set)
    created_at: datetime = field(default_factory=datetime.now)
    last_used: Optional[datetime] = None
    
    @property
    def average_performance(self) -> float:
        """Calculate average performance score"""
        return np.mean(self.performance_history) if self.performance_history else 0.0
    
    def update_performance(self, score: float, domain: str = "general"):
        """Update performance metrics"""
        self.performance_history.append(score)
        self.domains_applied.add(domain)
        self.success_rate = self.average_performance
        self.last_used = datetime.now()
        
        # Keep only recent performance data
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]


class PatternMiner:
    """Extracts patterns from experiences using various ML techniques"""
    
    def __init__(self):
        self.min_support = 3  # Minimum experiences to form a pattern
        self.min_confidence = 0.6  # Minimum confidence for pattern validity
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        
    async def extract_patterns(self, experiences: List[Experience], 
                              pattern_types: List[PatternType] = None) -> List[Pattern]:
        """Extract patterns from experiences"""
        if not experiences:
            return []
        
        if pattern_types is None:
            pattern_types = list(PatternType)
        
        patterns = []
        
        for pattern_type in pattern_types:
            if pattern_type == PatternType.TASK_SUCCESS:
                patterns.extend(await self._extract_success_patterns(experiences))
            elif pattern_type == PatternType.AGENT_PERFORMANCE:
                patterns.extend(await self._extract_performance_patterns(experiences))
            elif pattern_type == PatternType.EXECUTION_STRATEGY:
                patterns.extend(await self._extract_strategy_patterns(experiences))
            elif pattern_type == PatternType.ERROR_PATTERN:
                patterns.extend(await self._extract_error_patterns(experiences))
            elif pattern_type == PatternType.RESOURCE_USAGE:
                patterns.extend(await self._extract_resource_patterns(experiences))
            elif pattern_type == PatternType.TEMPORAL:
                patterns.extend(await self._extract_temporal_patterns(experiences))
        
        logger.info(f"Extracted {len(patterns)} patterns from {len(experiences)} experiences")
        return patterns
    
    async def _extract_success_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns related to task success"""
        patterns = []
        
        # Group by domain and success
        domain_groups = defaultdict(lambda: {'success': [], 'failure': []})
        for exp in experiences:
            if exp.success:
                domain_groups[exp.domain]['success'].append(exp)
            else:
                domain_groups[exp.domain]['failure'].append(exp)
        
        for domain, groups in domain_groups.items():
            if len(groups['success']) >= self.min_support:
                # Analyze successful experiences
                avg_quality = np.mean([exp.quality_score for exp in groups['success']])
                avg_time = np.mean([exp.execution_time for exp in groups['success']])
                
                pattern = Pattern(
                    pattern_id=f"success_{domain}_{len(patterns)}",
                    pattern_type=PatternType.TASK_SUCCESS,
                    description=f"Successful task execution in {domain} domain",
                    conditions={
                        'domain': domain,
                        'quality_score': {'min': avg_quality * 0.8},
                        'execution_time': {'min': 0, 'max': avg_time * 1.2}
                    },
                    outcomes={'success_probability': 0.8},
                    confidence=len(groups['success']) / (len(groups['success']) + len(groups['failure'])),
                    support=len(groups['success']),
                    domains={domain}
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_performance_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns related to agent performance"""
        patterns = []
        
        # Group by agent
        agent_groups = defaultdict(list)
        for exp in experiences:
            agent_groups[exp.agent_id].append(exp)
        
        for agent_id, agent_exps in agent_groups.items():
            if len(agent_exps) >= self.min_support:
                high_performers = [exp for exp in agent_exps if exp.quality_score > 0.8]
                
                if len(high_performers) >= self.min_support:
                    # Find common characteristics of high performance
                    avg_context_size = np.mean([len(exp.context) for exp in high_performers])
                    common_domains = set.intersection(*[{exp.domain} for exp in high_performers])
                    
                    pattern = Pattern(
                        pattern_id=f"performance_{agent_id}_{len(patterns)}",
                        pattern_type=PatternType.AGENT_PERFORMANCE,
                        description=f"High performance pattern for agent {agent_id}",
                        conditions={
                            'agent_id': agent_id,
                            'context_size': {'min': avg_context_size * 0.8, 'max': avg_context_size * 1.2}
                        },
                        outcomes={'expected_quality': 0.85},
                        confidence=len(high_performers) / len(agent_exps),
                        support=len(high_performers),
                        domains=common_domains
                    )
                    patterns.append(pattern)
        
        return patterns
    
    async def _extract_strategy_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns related to execution strategies"""
        patterns = []
        
        # Cluster experiences by actions taken
        if len(experiences) < 10:
            return patterns
        
        try:
            # Vectorize action sequences
            action_texts = [' '.join(exp.actions_taken) for exp in experiences]
            if not action_texts:
                return patterns
            
            action_vectors = self.vectorizer.fit_transform(action_texts)
            
            # Cluster similar action patterns
            n_clusters = min(5, len(experiences) // 3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(action_vectors.toarray())
            
            # Analyze each cluster
            for cluster_id in range(n_clusters):
                cluster_experiences = [exp for i, exp in enumerate(experiences) if clusters[i] == cluster_id]
                
                if len(cluster_experiences) >= self.min_support:
                    avg_success_rate = np.mean([exp.success for exp in cluster_experiences])
                    avg_quality = np.mean([exp.quality_score for exp in cluster_experiences])
                    
                    if avg_success_rate > 0.6:  # Only consider successful patterns
                        pattern = Pattern(
                            pattern_id=f"strategy_cluster_{cluster_id}_{len(patterns)}",
                            pattern_type=PatternType.EXECUTION_STRATEGY,
                            description=f"Successful execution strategy cluster {cluster_id}",
                            conditions={'strategy_cluster': cluster_id},
                            outcomes={
                                'success_rate': avg_success_rate,
                                'expected_quality': avg_quality
                            },
                            confidence=avg_success_rate,
                            support=len(cluster_experiences),
                            domains=set(exp.domain for exp in cluster_experiences)
                        )
                        patterns.append(pattern)
        
        except Exception as e:
            logger.error(f"Error in strategy pattern extraction: {e}")
        
        return patterns
    
    async def _extract_error_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns related to errors and failures"""
        patterns = []
        
        failed_experiences = [exp for exp in experiences if not exp.success]
        
        if len(failed_experiences) >= self.min_support:
            # Analyze common failure characteristics
            failure_times = [exp.timestamp.hour for exp in failed_experiences]
            common_failure_hours = []
            
            for hour in range(24):
                hour_failures = sum(1 for h in failure_times if h == hour)
                if hour_failures >= self.min_support:
                    common_failure_hours.append(hour)
            
            if common_failure_hours:
                pattern = Pattern(
                    pattern_id=f"error_temporal_{len(patterns)}",
                    pattern_type=PatternType.ERROR_PATTERN,
                    description=f"Higher failure rate during hours {common_failure_hours}",
                    conditions={'hour_of_day': common_failure_hours},
                    outcomes={'failure_probability': 0.7},
                    confidence=0.7,
                    support=len(failed_experiences),
                    domains=set(exp.domain for exp in failed_experiences)
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_resource_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract patterns related to resource usage"""
        patterns = []
        
        # Analyze resource efficiency
        efficient_experiences = []
        for exp in experiences:
            if exp.success and exp.execution_time > 0:
                efficiency = exp.quality_score / exp.execution_time
                if efficiency > 0.1:  # Threshold for efficiency
                    efficient_experiences.append(exp)
        
        if len(efficient_experiences) >= self.min_support:
            avg_resources = defaultdict(list)
            for exp in efficient_experiences:
                for resource, usage in exp.resources_used.items():
                    avg_resources[resource].append(usage)
            
            resource_conditions = {}
            for resource, usages in avg_resources.items():
                if len(usages) >= self.min_support:
                    avg_usage = np.mean(usages)
                    resource_conditions[f"{resource}_usage"] = {'max': avg_usage * 1.1}
            
            if resource_conditions:
                pattern = Pattern(
                    pattern_id=f"resource_efficient_{len(patterns)}",
                    pattern_type=PatternType.RESOURCE_USAGE,
                    description="Efficient resource usage pattern",
                    conditions=resource_conditions,
                    outcomes={'efficiency': 'high'},
                    confidence=0.75,
                    support=len(efficient_experiences),
                    domains=set(exp.domain for exp in efficient_experiences)
                )
                patterns.append(pattern)
        
        return patterns
    
    async def _extract_temporal_patterns(self, experiences: List[Experience]) -> List[Pattern]:
        """Extract temporal patterns"""
        patterns = []
        
        # Analyze performance by time of day
        hourly_performance = defaultdict(list)
        for exp in experiences:
            hourly_performance[exp.timestamp.hour].append(exp.quality_score)
        
        peak_hours = []
        for hour, scores in hourly_performance.items():
            if len(scores) >= self.min_support and np.mean(scores) > 0.8:
                peak_hours.append(hour)
        
        if peak_hours:
            pattern = Pattern(
                pattern_id=f"temporal_peak_{len(patterns)}",
                pattern_type=PatternType.TEMPORAL,
                description=f"Peak performance hours: {peak_hours}",
                conditions={'hour_of_day': peak_hours},
                outcomes={'expected_quality': 0.85},
                confidence=0.8,
                support=sum(len(hourly_performance[h]) for h in peak_hours),
                domains=set(exp.domain for exp in experiences)
            )
            patterns.append(pattern)
        
        return patterns


class TransferLearner:
    """Handles transfer learning between domains"""
    
    def __init__(self):
        self.domain_similarities: Dict[Tuple[str, str], float] = {}
        self.transferable_patterns: Dict[str, List[Pattern]] = defaultdict(list)
        
    async def calculate_domain_similarity(self, source_domain: str, target_domain: str, 
                                        experiences: List[Experience]) -> float:
        """Calculate similarity between domains based on experiences"""
        
        source_experiences = [exp for exp in experiences if exp.domain == source_domain]
        target_experiences = [exp for exp in experiences if exp.domain == target_domain]
        
        if not source_experiences or not target_experiences:
            return 0.0
        
        # Feature-based similarity
        source_features = np.array([list(exp.feature_vector.values())[:-3] for exp in source_experiences])  # Exclude domain-specific features
        target_features = np.array([list(exp.feature_vector.values())[:-3] for exp in target_experiences])
        
        if source_features.shape[1] == 0 or target_features.shape[1] == 0:
            return 0.0
        
        # Calculate mean feature vectors
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        
        # Cosine similarity
        similarity = cosine_similarity([source_mean], [target_mean])[0][0]
        
        self.domain_similarities[(source_domain, target_domain)] = similarity
        return similarity
    
    async def transfer_patterns(self, source_domain: str, target_domain: str, 
                              patterns: List[Pattern], similarity_threshold: float = 0.7) -> List[Pattern]:
        """Transfer patterns from source to target domain"""
        
        # Check domain similarity
        similarity = self.domain_similarities.get((source_domain, target_domain))
        if similarity is None:
            return []
        
        if similarity < similarity_threshold:
            logger.info(f"Domain similarity ({similarity:.3f}) below threshold for transfer")
            return []
        
        transferred_patterns = []
        
        for pattern in patterns:
            if source_domain in pattern.domains:
                # Create transferred pattern
                transferred_pattern = Pattern(
                    pattern_id=f"transfer_{pattern.pattern_id}_{target_domain}",
                    pattern_type=pattern.pattern_type,
                    description=f"Transferred: {pattern.description}",
                    conditions=pattern.conditions.copy(),
                    outcomes=pattern.outcomes.copy(),
                    confidence=pattern.confidence * similarity,  # Reduce confidence based on similarity
                    support=pattern.support,
                    domains={target_domain}
                )
                
                # Adjust conditions for target domain
                if 'domain' in transferred_pattern.conditions:
                    transferred_pattern.conditions['domain'] = target_domain
                
                transferred_patterns.append(transferred_pattern)
                self.transferable_patterns[target_domain].append(transferred_pattern)
        
        logger.info(f"Transferred {len(transferred_patterns)} patterns from {source_domain} to {target_domain}")
        return transferred_patterns
    
    async def get_transfer_candidates(self, target_domain: str, 
                                    all_patterns: List[Pattern]) -> List[Tuple[Pattern, str, float]]:
        """Get patterns that could be transferred to target domain"""
        candidates = []
        
        for pattern in all_patterns:
            for source_domain in pattern.domains:
                if source_domain != target_domain:
                    similarity = self.domain_similarities.get((source_domain, target_domain), 0.0)
                    if similarity > 0.5:  # Minimum similarity for consideration
                        candidates.append((pattern, source_domain, similarity))
        
        # Sort by similarity
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates


class MetaLearner:
    """Learns how to learn - optimizes learning strategies"""
    
    def __init__(self):
        self.learning_strategies: Dict[str, LearningStrategy] = {}
        self.strategy_performance_history: List[Dict[str, Any]] = []
        self._initialize_base_strategies()
    
    def _initialize_base_strategies(self):
        """Initialize base learning strategies"""
        strategies = [
            LearningStrategy(
                strategy_id="pattern_frequency",
                name="Pattern Frequency Learning",
                description="Learn based on frequency of successful patterns",
                parameters={"min_frequency": 3, "confidence_threshold": 0.6}
            ),
            LearningStrategy(
                strategy_id="quality_weighted",
                name="Quality Weighted Learning",
                description="Weight patterns by quality scores",
                parameters={"quality_weight": 0.7, "success_weight": 0.3}
            ),
            LearningStrategy(
                strategy_id="temporal_decay",
                name="Temporal Decay Learning",
                description="Give more weight to recent experiences",
                parameters={"decay_rate": 0.95, "window_size": 50}
            ),
            LearningStrategy(
                strategy_id="domain_transfer",
                name="Domain Transfer Learning",
                description="Transfer knowledge between similar domains",
                parameters={"similarity_threshold": 0.7, "transfer_weight": 0.5}
            )
        ]
        
        for strategy in strategies:
            self.learning_strategies[strategy.strategy_id] = strategy
    
    async def select_optimal_strategy(self, domain: str, task_type: str, 
                                    available_data: int) -> LearningStrategy:
        """Select the optimal learning strategy for given conditions"""
        
        # Score each strategy
        strategy_scores = {}
        
        for strategy_id, strategy in self.learning_strategies.items():
            score = strategy.average_performance
            
            # Adjust score based on domain experience
            if domain in strategy.domains_applied:
                score *= 1.2
            
            # Adjust score based on data availability
            if strategy_id == "pattern_frequency" and available_data < 10:
                score *= 0.5
            elif strategy_id == "temporal_decay" and available_data > 100:
                score *= 1.3
            
            # Recency bonus
            if strategy.last_used:
                days_since_used = (datetime.now() - strategy.last_used).days
                if days_since_used < 7:
                    score *= 1.1
                elif days_since_used > 30:
                    score *= 0.9
            
            strategy_scores[strategy_id] = score
        
        # Select best strategy
        best_strategy_id = max(strategy_scores, key=strategy_scores.get)
        selected_strategy = self.learning_strategies[best_strategy_id]
        
        logger.info(f"Selected learning strategy: {selected_strategy.name} (score: {strategy_scores[best_strategy_id]:.3f})")
        return selected_strategy
    
    async def evaluate_strategy_performance(self, strategy_id: str, patterns_found: List[Pattern], 
                                          validation_experiences: List[Experience]) -> float:
        """Evaluate how well a learning strategy performed"""
        if not patterns_found or not validation_experiences:
            return 0.0
        
        # Test patterns against validation data
        correct_predictions = 0
        total_predictions = 0
        
        for exp in validation_experiences:
            for pattern in patterns_found:
                if pattern.matches(exp):
                    total_predictions += 1
                    # Check if pattern prediction was correct
                    if pattern.pattern_type == PatternType.TASK_SUCCESS:
                        if exp.success and pattern.outcomes.get('success_probability', 0) > 0.7:
                            correct_predictions += 1
                    elif pattern.pattern_type == PatternType.AGENT_PERFORMANCE:
                        expected_quality = pattern.outcomes.get('expected_quality', 0.5)
                        if abs(exp.quality_score - expected_quality) < 0.2:
                            correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        # Update strategy performance
        if strategy_id in self.learning_strategies:
            self.learning_strategies[strategy_id].update_performance(accuracy)
        
        return accuracy
    
    async def optimize_strategies(self) -> Dict[str, Any]:
        """Optimize learning strategies based on historical performance"""
        optimizations = []
        
        for strategy_id, strategy in self.learning_strategies.items():
            if len(strategy.performance_history) >= 5:
                recent_performance = np.mean(strategy.performance_history[-5:])
                overall_performance = strategy.average_performance
                
                # If recent performance is declining, adjust parameters
                if recent_performance < overall_performance * 0.8:
                    if "min_frequency" in strategy.parameters:
                        strategy.parameters["min_frequency"] = max(1, strategy.parameters["min_frequency"] - 1)
                        optimizations.append(f"Reduced min_frequency for {strategy.name}")
                    
                    if "confidence_threshold" in strategy.parameters:
                        strategy.parameters["confidence_threshold"] = max(0.3, strategy.parameters["confidence_threshold"] - 0.1)
                        optimizations.append(f"Reduced confidence_threshold for {strategy.name}")
                
                # If performing well, be more selective
                elif recent_performance > overall_performance * 1.2:
                    if "confidence_threshold" in strategy.parameters:
                        strategy.parameters["confidence_threshold"] = min(0.9, strategy.parameters["confidence_threshold"] + 0.05)
                        optimizations.append(f"Increased confidence_threshold for {strategy.name}")
        
        return {"optimizations_applied": optimizations}


class AdaptiveLearningSystem:
    """Main adaptive learning system that coordinates all learning components"""
    
    def __init__(self):
        self.pattern_miner = PatternMiner()
        self.transfer_learner = TransferLearner()
        self.meta_learner = MetaLearner()
        
        # Storage
        self.experiences: List[Experience] = []
        self.discovered_patterns: Dict[str, Pattern] = {}
        self.learning_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.max_experiences = 1000  # Keep only recent experiences
        self.pattern_update_interval = 50  # Update patterns every N experiences
        self.meta_learning_interval = 100  # Meta-learn every N experiences
        
    async def record_experience(self, experience: Experience):
        """Record a new learning experience"""
        self.experiences.append(experience)
        
        # Maintain maximum experiences
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences:]
        
        # Trigger pattern updates if needed
        if len(self.experiences) % self.pattern_update_interval == 0:
            await self.update_patterns()
        
        # Trigger meta-learning if needed
        if len(self.experiences) % self.meta_learning_interval == 0:
            await self.perform_meta_learning()
        
        logger.debug(f"Recorded experience: {experience.experience_id}")
    
    async def update_patterns(self):
        """Update discovered patterns based on recent experiences"""
        if len(self.experiences) < 5:
            return {}
        
        # Select optimal learning strategy
        domain_counts = defaultdict(int)
        for exp in self.experiences[-50:]:  # Recent experiences
            domain_counts[exp.domain] += 1
        
        primary_domain = max(domain_counts, key=domain_counts.get) if domain_counts else "general"
        
        strategy = await self.meta_learner.select_optimal_strategy(
            domain=primary_domain,
            task_type="pattern_discovery",
            available_data=len(self.experiences)
        )
        
        # Extract patterns using selected strategy
        new_patterns = await self.pattern_miner.extract_patterns(self.experiences[-100:])  # Recent patterns
        
        # Update pattern database
        for pattern in new_patterns:
            if pattern.confidence >= strategy.parameters.get("confidence_threshold", 0.6):
                self.discovered_patterns[pattern.pattern_id] = pattern
        
        # Evaluate strategy performance
        if len(self.experiences) > 50:
            validation_experiences = self.experiences[-20:]  # Use recent as validation
            performance = await self.meta_learner.evaluate_strategy_performance(
                strategy.strategy_id, new_patterns, validation_experiences
            )
            
            logger.info(f"Pattern update completed: {len(new_patterns)} patterns found, "
                       f"strategy performance: {performance:.3f}")
        
        # Record learning event
        self.learning_history.append({
            "timestamp": datetime.now(),
            "strategy_used": strategy.strategy_id,
            "patterns_found": len(new_patterns),
            "experiences_analyzed": len(self.experiences)
        })
    
    async def perform_meta_learning(self):
        """Perform meta-learning to optimize learning strategies"""
        if len(self.learning_history) < 5:
            return {}
        
        # Analyze learning performance over time
        recent_performance = []
        for event in self.learning_history[-10:]:
            patterns_per_experience = event["patterns_found"] / max(1, event["experiences_analyzed"])
            recent_performance.append(patterns_per_experience)
        
        # Optimize strategies
        optimization_result = await self.meta_learner.optimize_strategies()
        
        logger.info(f"Meta-learning completed: {len(optimization_result['optimizations_applied'])} optimizations applied")
    
    async def transfer_knowledge(self, source_domain: str, target_domain: str) -> int:
        """Transfer knowledge between domains"""
        
        # Calculate domain similarity
        similarity = await self.transfer_learner.calculate_domain_similarity(
            source_domain, target_domain, self.experiences
        )
        
        if similarity < 0.5:
            logger.info(f"Domains too dissimilar for transfer: {similarity:.3f}")
            return 0
        
        # Find transferable patterns
        source_patterns = [p for p in self.discovered_patterns.values() if source_domain in p.domains]
        
        # Transfer patterns
        transferred_patterns = await self.transfer_learner.transfer_patterns(
            source_domain, target_domain, source_patterns
        )
        
        # Add transferred patterns to discovered patterns
        for pattern in transferred_patterns:
            self.discovered_patterns[pattern.pattern_id] = pattern
        
        logger.info(f"Transferred {len(transferred_patterns)} patterns from {source_domain} to {target_domain}")
        return len(transferred_patterns)
    
    async def get_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get recommendations based on learned patterns"""
        recommendations = []
        
        # Create a mock experience for pattern matching
        mock_experience = Experience(
            experience_id="mock",
            agent_id=context.get("agent_id", "unknown"),
            task_description=context.get("task_description", ""),
            context=context,
            actions_taken=[],
            outcome={},
            success=True,
            quality_score=0.0,
            execution_time=0.0,
            resources_used={},
            domain=context.get("domain", "general")
        )
        
        # Find matching patterns
        matching_patterns = []
        for pattern in self.discovered_patterns.values():
            if pattern.matches(mock_experience):
                matching_patterns.append(pattern)
        
        # Sort by confidence
        matching_patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        # Generate recommendations
        for pattern in matching_patterns[:5]:  # Top 5 recommendations
            recommendation = {
                "pattern_id": pattern.pattern_id,
                "type": pattern.pattern_type.value,
                "description": pattern.description,
                "confidence": pattern.confidence,
                "expected_outcomes": pattern.outcomes,
                "recommendation": self._generate_recommendation_text(pattern)
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def _generate_recommendation_text(self, pattern: Pattern) -> str:
        """Generate human-readable recommendation text"""
        if pattern.pattern_type == PatternType.TASK_SUCCESS:
            return f"Based on similar tasks, expect {pattern.outcomes.get('success_probability', 0)*100:.0f}% success rate"
        elif pattern.pattern_type == PatternType.AGENT_PERFORMANCE:
            expected_quality = pattern.outcomes.get('expected_quality', 0.5)
            return f"Agent typically achieves {expected_quality:.2f} quality score in this context"
        elif pattern.pattern_type == PatternType.EXECUTION_STRATEGY:
            success_rate = pattern.outcomes.get('success_rate', 0.5)
            return f"This execution strategy has {success_rate:.1%} success rate"
        elif pattern.pattern_type == PatternType.RESOURCE_USAGE:
            return "This configuration typically uses resources efficiently"
        elif pattern.pattern_type == PatternType.TEMPORAL:
            return f"Performance is typically higher during these conditions"
        else:
            return "Pattern-based recommendation available"
    
    async def get_learning_analytics(self) -> Dict[str, Any]:
        """Get learning system analytics"""
        
        # Experience statistics
        domain_distribution = defaultdict(int)
        success_rates_by_domain = defaultdict(list)
        
        for exp in self.experiences:
            domain_distribution[exp.domain] += 1
            success_rates_by_domain[exp.domain].append(int(exp.success))
        
        # Pattern statistics
        pattern_types = defaultdict(int)
        pattern_confidence_scores = []
        
        for pattern in self.discovered_patterns.values():
            pattern_types[pattern.pattern_type.value] += 1
            pattern_confidence_scores.append(pattern.confidence)
        
        # Learning strategy performance
        strategy_performance = {
            strategy_id: {
                "average_performance": strategy.average_performance,
                "total_uses": len(strategy.performance_history),
                "domains_applied": len(strategy.domains_applied)
            }
            for strategy_id, strategy in self.meta_learner.learning_strategies.items()
        }
        
        return {
            "experience_statistics": {
                "total_experiences": len(self.experiences),
                "domain_distribution": dict(domain_distribution),
                "success_rates_by_domain": {
                    domain: np.mean(rates) for domain, rates in success_rates_by_domain.items()
                }
            },
            "pattern_statistics": {
                "total_patterns": len(self.discovered_patterns),
                "pattern_types": dict(pattern_types),
                "average_confidence": np.mean(pattern_confidence_scores) if pattern_confidence_scores else 0,
                "confidence_distribution": {
                    "high (>0.8)": sum(1 for c in pattern_confidence_scores if c > 0.8),
                    "medium (0.5-0.8)": sum(1 for c in pattern_confidence_scores if 0.5 <= c <= 0.8),
                    "low (<0.5)": sum(1 for c in pattern_confidence_scores if c < 0.5)
                }
            },
            "learning_strategy_performance": strategy_performance,
            "transfer_learning": {
                "domain_similarities": dict(self.transfer_learner.domain_similarities),
                "transferred_patterns_count": sum(
                    len(patterns) for patterns in self.transfer_learner.transferable_patterns.values()
                )
            },
            "system_metrics": {
                "learning_events": len(self.learning_history),
                "last_pattern_update": max(
                    (event["timestamp"] for event in self.learning_history), 
                    default=None
                )
            }
        }


if __name__ == "__main__":
    async def demo_adaptive_learning():
        """Demonstrate adaptive learning system capabilities"""
        learning_system = AdaptiveLearningSystem()
        
        print("=" * 80)
        print("ADAPTIVE LEARNING SYSTEM DEMONSTRATION")
        print("=" * 80)
        
        # Simulate learning experiences
        domains = ["financial", "data_analysis", "development", "testing"]
        agents = ["agent_001", "agent_002", "agent_003"]
        
        print("RECORDING LEARNING EXPERIENCES:")
        print("-" * 40)
        
        # Generate sample experiences
        for i in range(100):
            domain = np.random.choice(domains)
            agent = np.random.choice(agents)
            
            # Simulate realistic experience patterns
            if domain == "financial":
                success_prob = 0.8
                quality_base = 0.7
            elif domain == "development":
                success_prob = 0.7
                quality_base = 0.75
            else:
                success_prob = 0.75
                quality_base = 0.6
            
            success = np.random.random() < success_prob
            quality = max(0, min(1, np.random.normal(quality_base, 0.15)))
            
            experience = Experience(
                experience_id=f"exp_{i:03d}",
                agent_id=agent,
                task_description=f"Task in {domain} domain with complexity level {np.random.randint(1, 6)}",
                context={"complexity": np.random.randint(1, 6), "urgent": np.random.random() > 0.7},
                actions_taken=[f"action_{j}" for j in range(np.random.randint(1, 5))],
                outcome={"result": "success" if success else "failure"},
                success=success,
                quality_score=quality,
                execution_time=np.random.uniform(5, 120),
                resources_used={"cpu": np.random.uniform(0.1, 1.0), "memory": np.random.uniform(0.2, 0.8)},
                domain=domain,
                timestamp=datetime.now() - timedelta(days=np.random.randint(0, 30))
            )
            
            await learning_system.record_experience(experience)
        
        print(f"Recorded {len(learning_system.experiences)} experiences")
        print(f"Discovered {len(learning_system.discovered_patterns)} patterns")
        
        # Demonstrate knowledge transfer
        print("\nKNOWLEDGE TRANSFER:")
        print("-" * 40)
        
        transferred_count = await learning_system.transfer_knowledge("financial", "data_analysis")
        print(f"Transferred {transferred_count} patterns from financial to data_analysis")
        
        # Get recommendations
        print("\nRECOMMENDATIONS:")
        print("-" * 40)
        
        context = {
            "agent_id": "agent_001",
            "task_description": "Process financial report with high complexity",
            "domain": "financial",
            "complexity": 5,
            "urgent": True
        }
        
        recommendations = await learning_system.get_recommendations(context)
        
        for i, rec in enumerate(recommendations[:3], 1):
            print(f"Recommendation {i}:")
            print(f"  Type: {rec['type']}")
            print(f"  Confidence: {rec['confidence']:.3f}")
            print(f"  Advice: {rec['recommendation']}")
            print()
        
        # Show analytics
        print("LEARNING ANALYTICS:")
        print("-" * 40)
        
        analytics = await learning_system.get_learning_analytics()
        
        exp_stats = analytics["experience_statistics"]
        print(f"Total experiences: {exp_stats['total_experiences']}")
        print(f"Domains: {list(exp_stats['domain_distribution'].keys())}")
        print(f"Success rates by domain:")
        for domain, rate in exp_stats['success_rates_by_domain'].items():
            print(f"  {domain}: {rate:.2%}")
        
        pattern_stats = analytics["pattern_statistics"]
        print(f"\\nTotal patterns: {pattern_stats['total_patterns']}")
        print(f"Average confidence: {pattern_stats['average_confidence']:.3f}")
        print(f"Pattern types: {pattern_stats['pattern_types']}")
        
        print(f"\\nLearning events: {analytics['system_metrics']['learning_events']}")
        print(f"Transfer learning active: {len(analytics['transfer_learning']['domain_similarities'])} domain pairs")
        
        # Demonstrate pattern matching
        print("\\nPATTERN MATCHING EXAMPLE:")
        print("-" * 40)
        
        if learning_system.discovered_patterns:
            sample_pattern = list(learning_system.discovered_patterns.values())[0]
            print(f"Pattern: {sample_pattern.description}")
            print(f"Type: {sample_pattern.pattern_type.value}")
            print(f"Confidence: {sample_pattern.confidence:.3f}")
            print(f"Support: {sample_pattern.support} experiences")
            print(f"Domains: {sample_pattern.domains}")
    
    # Run demonstration
    asyncio.run(demo_adaptive_learning())