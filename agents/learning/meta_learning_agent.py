"""
Meta-Learning Agent
Learns how to learn and adapts its own learning strategies
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from templates.base_agent import BaseAgent, Action, Observation, Thought
from .strategy_optimizer import StrategyOptimizer
from .pattern_recognizer import PatternRecognizer
from .knowledge_transfer import KnowledgeTransfer
from utils.observability.logging import get_logger

logger = get_logger(__name__)


class LearningMode(Enum):
    """Different modes of learning"""
    EXPLORATION = "exploration"      # Trying new strategies
    EXPLOITATION = "exploitation"    # Using best known strategies
    ADAPTATION = "adaptation"        # Adapting to new environments
    TRANSFER = "transfer"           # Transferring knowledge across domains
    REFLECTION = "reflection"       # Analyzing past performance


@dataclass
class LearningStrategy:
    """Represents a learning strategy"""
    strategy_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    success_rate: float = 0.0
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    domains_effective: List[str] = field(default_factory=list)
    complexity_level: float = 0.5  # 0 = simple, 1 = complex
    adaptation_rate: float = 0.1   # How fast it adapts


@dataclass
class LearningExperience:
    """Represents a learning experience"""
    experience_id: str
    task_context: Dict[str, Any]
    strategy_used: str
    outcome: Any
    success: bool
    performance_metrics: Dict[str, float]
    lessons_learned: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class MetaLearningAgent(BaseAgent):
    """
    Agent that learns how to learn
    Implements meta-learning algorithms and strategy optimization
    """
    
    def __init__(self, name: str = "meta_learning_agent", **kwargs):
        super().__init__(name, **kwargs)
        
        # Meta-learning components
        self.strategy_optimizer = StrategyOptimizer()
        self.pattern_recognizer = PatternRecognizer()
        self.knowledge_transfer = KnowledgeTransfer()
        
        # Learning state
        self.current_learning_mode = LearningMode.EXPLORATION
        self.learning_strategies: Dict[str, LearningStrategy] = {}
        self.learning_experiences: List[LearningExperience] = []
        
        # Meta-parameters
        self.exploration_rate = 0.3
        self.adaptation_threshold = 0.7
        self.strategy_retirement_threshold = 0.2
        self.max_experiences = 1000
        
        # Performance tracking
        self.performance_history: List[Dict[str, float]] = []
        self.learning_curve_data: List[float] = []
        
        # Initialize default strategies
        self._initialize_default_strategies()
        
        logger.info(f"Initialized meta-learning agent: {name}")
    
    def _initialize_default_strategies(self) -> None:
        """Initialize default learning strategies"""
        default_strategies = [
            LearningStrategy(
                strategy_id="reinforcement_learning",
                name="Reinforcement Learning",
                description="Learn through reward/penalty feedback",
                parameters={"learning_rate": 0.1, "discount_factor": 0.9},
                complexity_level=0.6
            ),
            LearningStrategy(
                strategy_id="imitation_learning", 
                name="Imitation Learning",
                description="Learn by copying successful behaviors",
                parameters={"similarity_threshold": 0.8},
                complexity_level=0.4
            ),
            LearningStrategy(
                strategy_id="curiosity_driven",
                name="Curiosity-Driven Learning",
                description="Learn by exploring novel situations",
                parameters={"novelty_threshold": 0.5, "exploration_bonus": 0.1},
                complexity_level=0.8
            ),
            LearningStrategy(
                strategy_id="meta_gradient",
                name="Meta-Gradient Learning",
                description="Learn how to update learning rules",
                parameters={"meta_lr": 0.01, "inner_lr": 0.1},
                complexity_level=0.9
            ),
            LearningStrategy(
                strategy_id="few_shot_learning",
                name="Few-Shot Learning",
                description="Learn from very few examples",
                parameters={"support_set_size": 5, "similarity_metric": "cosine"},
                complexity_level=0.7
            )
        ]
        
        for strategy in default_strategies:
            self.learning_strategies[strategy.strategy_id] = strategy
        
        logger.info(f"Initialized {len(default_strategies)} default learning strategies")
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute meta-learning tasks"""
        task_str = str(task).lower()
        
        if "learn" in task_str or "adapt" in task_str:
            return await self._handle_learning_task(task, action)
        elif "optimize" in task_str:
            return await self._handle_optimization_task(task, action)
        elif "transfer" in task_str:
            return await self._handle_transfer_task(task, action)
        elif "reflect" in task_str:
            return await self._handle_reflection_task(task, action)
        else:
            return await self._handle_general_learning_task(task, action)
    
    async def _handle_learning_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle learning and adaptation tasks"""
        
        # Extract task context
        task_context = self._extract_task_context(task)
        
        # Select learning strategy
        selected_strategy = await self._select_learning_strategy(task_context)
        
        # Apply learning strategy
        learning_result = await self._apply_learning_strategy(
            selected_strategy, task, task_context
        )
        
        # Record experience
        experience = LearningExperience(
            experience_id=f"exp_{len(self.learning_experiences)}_{datetime.now().isoformat()}",
            task_context=task_context,
            strategy_used=selected_strategy.strategy_id,
            outcome=learning_result,
            success=learning_result.get("success", False),
            performance_metrics=learning_result.get("metrics", {}),
            lessons_learned=learning_result.get("lessons", [])
        )
        
        await self._record_experience(experience)
        
        return {
            "learning_result": learning_result,
            "strategy_used": selected_strategy.name,
            "experience_id": experience.experience_id,
            "lessons_learned": experience.lessons_learned,
            "performance_improvement": await self._calculate_performance_improvement()
        }
    
    async def _handle_optimization_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle strategy optimization tasks"""
        
        # Run strategy optimization
        optimization_context = {
            "strategies": self.learning_strategies,
            "experiences": self.learning_experiences
        }
        optimization_results = await self.strategy_optimizer.optimize_strategy(
            optimization_context
        )
        
        # Update strategies based on optimization
        await self._update_strategies(optimization_results)
        
        return {
            "optimized_strategies": len(optimization_results),
            "performance_improvements": optimization_results,
            "best_strategies": await self._get_best_strategies(),
            "retired_strategies": await self._retire_poor_strategies()
        }
    
    async def _handle_transfer_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle knowledge transfer tasks"""
        
        task_params = self._extract_task_parameters(task)
        source_domain = task_params.get("source_domain", "general")
        target_domain = task_params.get("target_domain", "specific")
        
        # Extract transferable knowledge
        transferable_knowledge = await self.knowledge_transfer.extract_transferable_knowledge(
            self.learning_experiences,
            source_domain
        )
        
        # Apply transfer learning
        transfer_result = await self.knowledge_transfer.apply_transfer_learning(
            transferable_knowledge,
            target_domain,
            self.learning_strategies
        )
        
        return {
            "transfer_successful": transfer_result.get("success", False),
            "knowledge_transferred": len(transferable_knowledge),
            "new_strategies_created": transfer_result.get("new_strategies", 0),
            "performance_boost": transfer_result.get("performance_improvement", 0.0),
            "transfer_efficiency": transfer_result.get("efficiency", 0.0)
        }
    
    async def _handle_reflection_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle reflection and meta-analysis tasks"""
        
        # Analyze learning patterns
        learning_patterns = await self.pattern_extractor.extract_learning_patterns(
            self.learning_experiences
        )
        
        # Analyze strategy effectiveness
        strategy_analysis = await self._analyze_strategy_effectiveness()
        
        # Generate insights
        insights = await self._generate_learning_insights()
        
        # Update learning mode if needed
        new_mode = await self._determine_optimal_learning_mode()
        if new_mode != self.current_learning_mode:
            logger.info(f"Switching learning mode from {self.current_learning_mode.value} to {new_mode.value}")
            self.current_learning_mode = new_mode
        
        return {
            "learning_patterns": learning_patterns,
            "strategy_analysis": strategy_analysis,
            "insights": insights,
            "learning_mode": self.current_learning_mode.value,
            "recommendations": await self._generate_recommendations()
        }
    
    async def _handle_general_learning_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle general learning tasks"""
        
        # Automatic learning based on current mode
        if self.current_learning_mode == LearningMode.EXPLORATION:
            return await self._explore_new_strategies(task)
        elif self.current_learning_mode == LearningMode.EXPLOITATION:
            return await self._exploit_best_strategies(task)
        elif self.current_learning_mode == LearningMode.ADAPTATION:
            return await self._adapt_to_environment(task)
        elif self.current_learning_mode == LearningMode.TRANSFER:
            return await self._transfer_knowledge(task)
        else:  # REFLECTION
            return await self._reflect_on_performance(task)
    
    def _extract_task_context(self, task: Any) -> Dict[str, Any]:
        """Extract context information from task"""
        context = {
            "task_type": type(task).__name__,
            "task_complexity": self._estimate_task_complexity(task),
            "domain": self._identify_task_domain(task),
            "timestamp": datetime.now().isoformat(),
            "requires_creativity": "creative" in str(task).lower(),
            "requires_analysis": "analyze" in str(task).lower(),
            "requires_planning": "plan" in str(task).lower()
        }
        
        if isinstance(task, dict):
            context.update(task.get("context", {}))
        
        return context
    
    def _estimate_task_complexity(self, task: Any) -> float:
        """Estimate complexity of a task (0-1 scale)"""
        # Simple heuristic based on task description length and keywords
        task_str = str(task).lower()
        
        complexity = 0.5  # Base complexity
        
        # Adjust based on keywords
        complex_keywords = ["optimize", "multi", "complex", "advanced", "sophisticated"]
        simple_keywords = ["basic", "simple", "easy", "straightforward"]
        
        complexity += 0.1 * sum(1 for kw in complex_keywords if kw in task_str)
        complexity -= 0.1 * sum(1 for kw in simple_keywords if kw in task_str)
        
        # Adjust based on length
        if len(task_str) > 200:
            complexity += 0.2
        elif len(task_str) < 50:
            complexity -= 0.1
        
        return np.clip(complexity, 0.0, 1.0)
    
    def _identify_task_domain(self, task: Any) -> str:
        """Identify the domain of a task"""
        task_str = str(task).lower()
        
        domain_keywords = {
            "nlp": ["text", "language", "parse", "sentiment", "translation"],
            "vision": ["image", "visual", "picture", "recognize", "detect"],
            "analysis": ["analyze", "statistics", "data", "metrics", "patterns"],
            "planning": ["plan", "schedule", "optimize", "strategy", "goal"],
            "reasoning": ["logic", "reasoning", "infer", "deduce", "conclude"],
            "creative": ["generate", "create", "design", "creative", "innovative"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in task_str for kw in keywords):
                return domain
        
        return "general"
    
    async def _select_learning_strategy(self, task_context: Dict[str, Any]) -> LearningStrategy:
        """Select the most appropriate learning strategy"""
        
        # Filter strategies based on task context
        suitable_strategies = []
        
        for strategy in self.learning_strategies.values():
            # Check if strategy is effective in this domain
            task_domain = task_context.get("domain", "general")
            if (not strategy.domains_effective or 
                task_domain in strategy.domains_effective or
                "general" in strategy.domains_effective):
                
                # Check complexity match
                task_complexity = task_context.get("task_complexity", 0.5)
                complexity_diff = abs(strategy.complexity_level - task_complexity)
                
                if complexity_diff <= 0.3:  # Strategy complexity should be reasonably close
                    suitable_strategies.append(strategy)
        
        if not suitable_strategies:
            suitable_strategies = list(self.learning_strategies.values())
        
        # Select based on current learning mode
        if self.current_learning_mode == LearningMode.EXPLORATION:
            # Prefer strategies with lower usage or higher novelty
            selected = min(suitable_strategies, 
                          key=lambda s: (s.usage_count, -s.complexity_level))
        
        elif self.current_learning_mode == LearningMode.EXPLOITATION:
            # Prefer strategies with highest success rate
            selected = max(suitable_strategies, key=lambda s: s.success_rate)
        
        else:  # Other modes
            # Use epsilon-greedy selection
            if np.random.random() < self.exploration_rate:
                selected = np.random.choice(suitable_strategies)
            else:
                selected = max(suitable_strategies, key=lambda s: s.success_rate)
        
        # Update usage
        selected.usage_count += 1
        selected.last_used = datetime.now()
        
        return selected
    
    async def _apply_learning_strategy(self, 
                                     strategy: LearningStrategy, 
                                     task: Any, 
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a specific learning strategy"""
        
        if strategy.strategy_id == "reinforcement_learning":
            return await self._apply_reinforcement_learning(strategy, task, context)
        
        elif strategy.strategy_id == "imitation_learning":
            return await self._apply_imitation_learning(strategy, task, context)
        
        elif strategy.strategy_id == "curiosity_driven":
            return await self._apply_curiosity_driven_learning(strategy, task, context)
        
        elif strategy.strategy_id == "meta_gradient":
            return await self._apply_meta_gradient_learning(strategy, task, context)
        
        elif strategy.strategy_id == "few_shot_learning":
            return await self._apply_few_shot_learning(strategy, task, context)
        
        else:
            # Default strategy application
            return await self._apply_default_strategy(strategy, task, context)
    
    async def _apply_reinforcement_learning(self, 
                                          strategy: LearningStrategy,
                                          task: Any,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply reinforcement learning strategy"""
        learning_rate = strategy.parameters.get("learning_rate", 0.1)
        discount_factor = strategy.parameters.get("discount_factor", 0.9)
        
        # Simulate RL learning process
        initial_performance = np.random.uniform(0.3, 0.7)
        
        # Learning iterations
        performance_trajectory = [initial_performance]
        for i in range(10):
            reward = np.random.uniform(-0.1, 0.2)
            current_perf = performance_trajectory[-1]
            new_perf = current_perf + learning_rate * (reward + discount_factor * current_perf - current_perf)
            new_perf = np.clip(new_perf, 0.0, 1.0)
            performance_trajectory.append(new_perf)
        
        final_performance = performance_trajectory[-1]
        improvement = final_performance - initial_performance
        
        return {
            "success": improvement > 0,
            "metrics": {
                "initial_performance": initial_performance,
                "final_performance": final_performance,
                "improvement": improvement,
                "learning_rate": learning_rate
            },
            "lessons": [
                f"Reinforcement learning achieved {improvement:.3f} improvement",
                f"Learning trajectory showed {'consistent improvement' if improvement > 0.1 else 'gradual progress'}"
            ]
        }
    
    async def _apply_imitation_learning(self, 
                                      strategy: LearningStrategy,
                                      task: Any,
                                      context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply imitation learning strategy"""
        similarity_threshold = strategy.parameters.get("similarity_threshold", 0.8)
        
        # Find similar past experiences
        similar_experiences = await self._find_similar_experiences(context, similarity_threshold)
        
        if similar_experiences:
            # Imitate successful behaviors
            successful_experiences = [exp for exp in similar_experiences if exp.success]
            
            if successful_experiences:
                # Average performance from successful imitations
                avg_performance = np.mean([
                    exp.performance_metrics.get("success_rate", 0.5) 
                    for exp in successful_experiences
                ])
                
                return {
                    "success": True,
                    "metrics": {
                        "imitated_experiences": len(successful_experiences),
                        "average_performance": avg_performance,
                        "similarity_threshold": similarity_threshold
                    },
                    "lessons": [
                        f"Successfully imitated {len(successful_experiences)} similar experiences",
                        f"Achieved {avg_performance:.3f} average performance through imitation"
                    ]
                }
        
        # No good examples to imitate
        return {
            "success": False,
            "metrics": {"similar_experiences": len(similar_experiences)},
            "lessons": ["Insufficient similar experiences for effective imitation learning"]
        }
    
    async def _apply_curiosity_driven_learning(self, 
                                             strategy: LearningStrategy,
                                             task: Any,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply curiosity-driven learning strategy"""
        novelty_threshold = strategy.parameters.get("novelty_threshold", 0.5)
        exploration_bonus = strategy.parameters.get("exploration_bonus", 0.1)
        
        # Calculate novelty of current task
        novelty_score = await self._calculate_novelty(context)
        
        # Higher novelty gives exploration bonus
        base_performance = np.random.uniform(0.4, 0.8)
        
        if novelty_score > novelty_threshold:
            performance = base_performance + exploration_bonus
            curiosity_reward = exploration_bonus
        else:
            performance = base_performance
            curiosity_reward = 0
        
        performance = np.clip(performance, 0.0, 1.0)
        
        return {
            "success": performance > 0.6,
            "metrics": {
                "novelty_score": novelty_score,
                "curiosity_reward": curiosity_reward,
                "final_performance": performance,
                "exploration_bonus_applied": novelty_score > novelty_threshold
            },
            "lessons": [
                f"Task novelty: {novelty_score:.3f}",
                f"{'Applied curiosity bonus' if novelty_score > novelty_threshold else 'Task too familiar for curiosity bonus'}"
            ]
        }
    
    async def _apply_meta_gradient_learning(self, 
                                          strategy: LearningStrategy,
                                          task: Any,
                                          context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply meta-gradient learning strategy"""
        meta_lr = strategy.parameters.get("meta_lr", 0.01)
        inner_lr = strategy.parameters.get("inner_lr", 0.1)
        
        # Simulate meta-learning process
        # Learn how to learn by updating learning rates
        
        performance_improvements = []
        current_inner_lr = inner_lr
        
        # Meta-learning iterations
        for meta_step in range(5):
            # Inner loop: regular learning
            inner_performance = []
            for inner_step in range(3):
                perf = np.random.uniform(0.3, 0.9) * (1 + current_inner_lr)
                inner_performance.append(np.clip(perf, 0.0, 1.0))
            
            # Meta update: improve learning rate based on inner performance
            meta_gradient = np.mean(inner_performance) - 0.5  # Simplified gradient
            current_inner_lr += meta_lr * meta_gradient
            current_inner_lr = np.clip(current_inner_lr, 0.01, 0.5)
            
            performance_improvements.append(np.mean(inner_performance))
        
        final_performance = performance_improvements[-1]
        meta_improvement = final_performance - performance_improvements[0]
        
        return {
            "success": meta_improvement > 0,
            "metrics": {
                "initial_performance": performance_improvements[0],
                "final_performance": final_performance,
                "meta_improvement": meta_improvement,
                "final_learning_rate": current_inner_lr,
                "meta_steps": len(performance_improvements)
            },
            "lessons": [
                f"Meta-learning improved performance by {meta_improvement:.3f}",
                f"Optimized learning rate to {current_inner_lr:.3f}"
            ]
        }
    
    async def _apply_few_shot_learning(self, 
                                     strategy: LearningStrategy,
                                     task: Any,
                                     context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply few-shot learning strategy"""
        support_set_size = strategy.parameters.get("support_set_size", 5)
        similarity_metric = strategy.parameters.get("similarity_metric", "cosine")
        
        # Get few examples from experience
        support_examples = await self._get_support_examples(context, support_set_size)
        
        if len(support_examples) >= 2:  # Need minimum examples
            # Simulate few-shot learning
            example_quality = np.mean([
                exp.performance_metrics.get("success_rate", 0.5) 
                for exp in support_examples
            ])
            
            # Few-shot performance depends on example quality
            few_shot_performance = 0.6 + 0.3 * example_quality
            few_shot_performance = np.clip(few_shot_performance, 0.0, 1.0)
            
            return {
                "success": few_shot_performance > 0.7,
                "metrics": {
                    "support_examples": len(support_examples),
                    "example_quality": example_quality,
                    "few_shot_performance": few_shot_performance,
                    "similarity_metric": similarity_metric
                },
                "lessons": [
                    f"Learned from {len(support_examples)} examples",
                    f"Achieved {few_shot_performance:.3f} performance with few-shot learning"
                ]
            }
        else:
            return {
                "success": False,
                "metrics": {"support_examples": len(support_examples)},
                "lessons": ["Insufficient examples for few-shot learning"]
            }
    
    async def _apply_default_strategy(self, 
                                    strategy: LearningStrategy,
                                    task: Any,
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply default/unknown strategy"""
        # Generic learning approach
        performance = np.random.uniform(0.4, 0.8)
        
        return {
            "success": performance > 0.6,
            "metrics": {"performance": performance},
            "lessons": [f"Applied default strategy: {strategy.name}"]
        }
    
    async def add_experience(self, experience: Dict[str, Any]) -> None:
        """Add a learning experience"""
        learning_exp = LearningExperience(
            experience_id=f"exp_{len(self.learning_experiences):06d}",
            task_context=experience.get("context", {}),
            strategy_used=experience.get("strategy", "unknown"),
            outcome=experience.get("performance", 0.5),
            success=experience.get("performance", 0.5) > 0.5,
            performance_metrics={"performance": experience.get("performance", 0.5)},
            lessons_learned=[],
            timestamp=experience.get("timestamp", datetime.now())
        )
        await self._record_experience(learning_exp)
    
    async def extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from experiences"""
        return await self.pattern_recognizer.extract_patterns([
            {
                'strategy': exp.strategy_used,
                'performance': exp.performance_metrics.get('performance', 0.5),
                'context': exp.task_context,
                'timestamp': exp.timestamp
            }
            for exp in self.learning_experiences
        ])
    
    async def recommend_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend best strategy for context"""
        return await self.strategy_optimizer.optimize_strategy(context)
    
    async def transfer_knowledge(self, source_context: Dict[str, Any], target_context: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge between contexts"""
        # Get current strategies
        strategies = {}
        for exp in self.learning_experiences[-20:]:  # Recent experiences
            strategy = exp.strategy_used
            performance = exp.performance_metrics.get('performance', 0.5)
            if strategy not in strategies:
                strategies[strategy] = {"expected_performance": performance}
            else:
                # Average performance
                current_perf = strategies[strategy]["expected_performance"]
                strategies[strategy]["expected_performance"] = (current_perf + performance) / 2
        
        return await self.knowledge_transfer.transfer_knowledge(source_context, target_context, strategies)
    
    async def get_meta_insights(self) -> Dict[str, Any]:
        """Get meta-learning insights"""
        return {
            "learning_progress": {"summary": f"Processed {len(self.learning_experiences)} experiences"},
            "strategy_effectiveness": {"summary": f"Using {len(set(exp.strategy_used for exp in self.learning_experiences))} different strategies"},
            "performance_trends": {"summary": "Performance tracking active"}
        }
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.learning_experiences:
            return {
                "total_experiences": 0,
                "average_performance": 0.5,
                "learning_rate": 0.0,
                "strategy_diversity": 0.0
            }
        
        performances = [exp.performance_metrics.get('performance', 0.5) for exp in self.learning_experiences]
        strategies = set(exp.strategy_used for exp in self.learning_experiences)
        
        return {
            "total_experiences": len(self.learning_experiences),
            "average_performance": np.mean(performances),
            "learning_rate": 0.1,  # Placeholder
            "strategy_diversity": len(strategies) / max(len(self.learning_experiences), 1)
        }
    
    async def _record_experience(self, experience: LearningExperience) -> None:
        """Record a learning experience"""
        self.learning_experiences.append(experience)
        
        # Maintain experience limit
        if len(self.learning_experiences) > self.max_experiences:
            self.learning_experiences.pop(0)
        
        # Update strategy success rate
        strategy = self.learning_strategies.get(experience.strategy_used)
        if strategy:
            # Update success rate with exponential moving average
            alpha = strategy.adaptation_rate
            new_success = 1.0 if experience.success else 0.0
            strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * new_success
            
            # Update domain effectiveness
            domain = experience.task_context.get("domain", "general")
            if experience.success and domain not in strategy.domains_effective:
                strategy.domains_effective.append(domain)
        
        # Update performance history
        metrics = experience.performance_metrics
        self.performance_history.append({
            "timestamp": experience.timestamp.isoformat(),
            "performance": metrics.get("final_performance", metrics.get("performance", 0.5)),
            "strategy": experience.strategy_used,
            "domain": experience.task_context.get("domain", "general")
        })
        
        logger.debug(f"Recorded learning experience: {experience.experience_id}")
    
    async def _find_similar_experiences(self, 
                                      context: Dict[str, Any],
                                      similarity_threshold: float) -> List[LearningExperience]:
        """Find similar past experiences"""
        similar = []
        
        current_domain = context.get("domain", "general")
        current_complexity = context.get("task_complexity", 0.5)
        
        for experience in self.learning_experiences:
            exp_domain = experience.task_context.get("domain", "general")
            exp_complexity = experience.task_context.get("task_complexity", 0.5)
            
            # Simple similarity calculation
            domain_match = 1.0 if exp_domain == current_domain else 0.5
            complexity_match = 1.0 - abs(exp_complexity - current_complexity)
            
            similarity = (domain_match + complexity_match) / 2
            
            if similarity >= similarity_threshold:
                similar.append(experience)
        
        return similar
    
    async def _calculate_novelty(self, context: Dict[str, Any]) -> float:
        """Calculate novelty score for a task context"""
        if not self.learning_experiences:
            return 1.0  # Everything is novel if no experience
        
        # Count similar experiences
        similar_count = len(await self._find_similar_experiences(context, 0.7))
        
        # Novelty inversely related to similarity count
        novelty = 1.0 / (1.0 + similar_count * 0.1)
        
        return novelty
    
    async def _get_support_examples(self, 
                                  context: Dict[str, Any],
                                  max_examples: int) -> List[LearningExperience]:
        """Get support examples for few-shot learning"""
        similar_experiences = await self._find_similar_experiences(context, 0.6)
        
        # Sort by success and recency
        similar_experiences.sort(
            key=lambda x: (x.success, x.timestamp), 
            reverse=True
        )
        
        return similar_experiences[:max_examples]
    
    async def _calculate_performance_improvement(self) -> float:
        """Calculate recent performance improvement"""
        if len(self.performance_history) < 10:
            return 0.0
        
        # Compare recent performance to earlier performance
        recent_perf = np.mean([
            p["performance"] for p in self.performance_history[-5:]
        ])
        
        earlier_perf = np.mean([
            p["performance"] for p in self.performance_history[-15:-10]
        ])
        
        return recent_perf - earlier_perf
    
    async def _update_strategies(self, optimization_results: Dict[str, Any]) -> None:
        """Update strategies based on optimization results"""
        for strategy_id, improvements in optimization_results.items():
            if strategy_id in self.learning_strategies:
                strategy = self.learning_strategies[strategy_id]
                
                # Update parameters
                if "parameters" in improvements:
                    strategy.parameters.update(improvements["parameters"])
                
                # Update success rate if provided
                if "success_rate" in improvements:
                    strategy.success_rate = improvements["success_rate"]
                
                logger.debug(f"Updated strategy {strategy_id} with optimization results")
    
    async def _get_best_strategies(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get best performing strategies"""
        strategies = list(self.learning_strategies.values())
        strategies.sort(key=lambda s: s.success_rate, reverse=True)
        
        return [
            {
                "strategy_id": s.strategy_id,
                "name": s.name,
                "success_rate": s.success_rate,
                "usage_count": s.usage_count,
                "domains_effective": s.domains_effective
            }
            for s in strategies[:limit]
        ]
    
    async def _retire_poor_strategies(self) -> List[str]:
        """Retire poorly performing strategies"""
        retired = []
        
        for strategy_id, strategy in list(self.learning_strategies.items()):
            if (strategy.success_rate < self.strategy_retirement_threshold and 
                strategy.usage_count > 10):  # Only retire if used enough times
                
                retired.append(strategy_id)
                del self.learning_strategies[strategy_id]
                logger.info(f"Retired poorly performing strategy: {strategy.name}")
        
        return retired
    
    async def _analyze_strategy_effectiveness(self) -> Dict[str, Any]:
        """Analyze effectiveness of different strategies"""
        analysis = {}
        
        for strategy_id, strategy in self.learning_strategies.items():
            # Get experiences using this strategy
            strategy_experiences = [
                exp for exp in self.learning_experiences 
                if exp.strategy_used == strategy_id
            ]
            
            if strategy_experiences:
                success_count = sum(1 for exp in strategy_experiences if exp.success)
                avg_performance = np.mean([
                    exp.performance_metrics.get("final_performance", 
                                              exp.performance_metrics.get("performance", 0.5))
                    for exp in strategy_experiences
                ])
                
                analysis[strategy_id] = {
                    "name": strategy.name,
                    "usage_count": len(strategy_experiences),
                    "success_rate": success_count / len(strategy_experiences),
                    "average_performance": avg_performance,
                    "domains_used": list(set(
                        exp.task_context.get("domain", "general")
                        for exp in strategy_experiences
                    ))
                }
        
        return analysis
    
    async def _generate_learning_insights(self) -> List[str]:
        """Generate insights from learning experiences"""
        insights = []
        
        if len(self.learning_experiences) < 5:
            insights.append("Need more experience to generate meaningful insights")
            return insights
        
        # Analyze success patterns
        recent_experiences = self.learning_experiences[-20:]
        success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
        
        if success_rate > 0.8:
            insights.append("High recent success rate indicates effective learning")
        elif success_rate < 0.3:
            insights.append("Low success rate suggests need for strategy adjustment")
        
        # Analyze domain distribution
        domains = [exp.task_context.get("domain", "general") for exp in recent_experiences]
        domain_counts = {domain: domains.count(domain) for domain in set(domains)}
        
        if len(domain_counts) > 3:
            insights.append("Learning across multiple domains - good generalization")
        
        # Analyze learning curve
        if len(self.performance_history) > 10:
            recent_trend = np.mean([p["performance"] for p in self.performance_history[-5:]])
            earlier_trend = np.mean([p["performance"] for p in self.performance_history[-15:-10]])
            
            if recent_trend > earlier_trend + 0.1:
                insights.append("Performance showing strong upward trend")
            elif recent_trend < earlier_trend - 0.1:
                insights.append("Performance declining - may need intervention")
        
        return insights
    
    async def _determine_optimal_learning_mode(self) -> LearningMode:
        """Determine optimal learning mode based on current state"""
        
        if len(self.learning_experiences) < 10:
            return LearningMode.EXPLORATION
        
        recent_success_rate = sum(
            1 for exp in self.learning_experiences[-10:] if exp.success
        ) / 10
        
        # Check performance trend
        improvement = await self._calculate_performance_improvement()
        
        if recent_success_rate < 0.4:
            return LearningMode.ADAPTATION  # Need to adapt
        elif improvement > 0.1:
            return LearningMode.EXPLOITATION  # Riding a good streak
        elif improvement < -0.1:
            return LearningMode.EXPLORATION  # Try something new
        elif len(set(exp.task_context.get("domain") for exp in self.learning_experiences[-5:])) > 2:
            return LearningMode.TRANSFER  # Multiple domains, try transfer
        else:
            return LearningMode.REFLECTION  # Stable performance, time to reflect
    
    async def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for learning improvement"""
        recommendations = []
        
        # Analyze current state
        if not self.learning_strategies:
            recommendations.append("Initialize learning strategies")
            return recommendations
        
        # Check strategy diversity
        active_strategies = [s for s in self.learning_strategies.values() if s.usage_count > 0]
        
        if len(active_strategies) < 3:
            recommendations.append("Explore more diverse learning strategies")
        
        # Check domain coverage
        domains_covered = set()
        for exp in self.learning_experiences[-20:]:
            domains_covered.add(exp.task_context.get("domain", "general"))
        
        if len(domains_covered) < 3:
            recommendations.append("Practice in more diverse domains")
        
        # Check learning progression
        improvement = await self._calculate_performance_improvement()
        
        if improvement < 0:
            recommendations.append("Consider strategy optimization or mode change")
        elif improvement > 0.2:
            recommendations.append("Excellent progress - consider knowledge transfer")
        
        if not recommendations:
            recommendations.append("Learning system performing well - continue current approach")
        
        return recommendations
    
    # Exploration, exploitation, adaptation methods
    async def _explore_new_strategies(self, task: Any) -> Dict[str, Any]:
        """Explore new learning strategies"""
        # Try least used strategies or create new ones
        sorted_strategies = sorted(
            self.learning_strategies.values(), 
            key=lambda s: s.usage_count
        )
        
        underused_strategy = sorted_strategies[0] if sorted_strategies else None
        
        if underused_strategy:
            context = self._extract_task_context(task)
            result = await self._apply_learning_strategy(underused_strategy, task, context)
            
            return {
                "exploration_result": result,
                "strategy_explored": underused_strategy.name,
                "mode": "exploration"
            }
        
        return {"message": "No strategies to explore"}
    
    async def _exploit_best_strategies(self, task: Any) -> Dict[str, Any]:
        """Exploit best performing strategies"""
        best_strategy = max(
            self.learning_strategies.values(),
            key=lambda s: s.success_rate
        )
        
        context = self._extract_task_context(task)
        result = await self._apply_learning_strategy(best_strategy, task, context)
        
        return {
            "exploitation_result": result,
            "strategy_used": best_strategy.name,
            "mode": "exploitation"
        }
    
    async def _adapt_to_environment(self, task: Any) -> Dict[str, Any]:
        """Adapt strategies to current environment"""
        context = self._extract_task_context(task)
        
        # Analyze recent failures to adapt
        recent_failures = [
            exp for exp in self.learning_experiences[-10:]
            if not exp.success
        ]
        
        if recent_failures:
            # Try to adapt based on failure patterns
            failure_strategies = set(exp.strategy_used for exp in recent_failures)
            
            # Use strategies not in failure set
            adaptive_strategies = [
                s for s in self.learning_strategies.values()
                if s.strategy_id not in failure_strategies
            ]
            
            if adaptive_strategies:
                strategy = max(adaptive_strategies, key=lambda s: s.success_rate)
                result = await self._apply_learning_strategy(strategy, task, context)
                
                return {
                    "adaptation_result": result,
                    "adapted_strategy": strategy.name,
                    "avoided_failures": len(failure_strategies),
                    "mode": "adaptation"
                }
        
        # Default adaptation
        return {"message": "No adaptation needed"}
    
    async def _transfer_knowledge(self, task: Any) -> Dict[str, Any]:
        """Transfer knowledge across domains"""
        context = self._extract_task_context(task)
        current_domain = context.get("domain", "general")
        
        # Find knowledge from other domains
        other_domain_experiences = [
            exp for exp in self.learning_experiences
            if (exp.task_context.get("domain", "general") != current_domain and 
                exp.success)
        ]
        
        if other_domain_experiences:
            # Apply successful strategies from other domains
            transferred_strategies = set(exp.strategy_used for exp in other_domain_experiences)
            
            if transferred_strategies:
                # Use most successful transferred strategy
                strategy_success_rates = {}
                for strategy_id in transferred_strategies:
                    if strategy_id in self.learning_strategies:
                        strategy_success_rates[strategy_id] = self.learning_strategies[strategy_id].success_rate
                
                best_transferred = max(strategy_success_rates, key=strategy_success_rates.get)
                strategy = self.learning_strategies[best_transferred]
                
                result = await self._apply_learning_strategy(strategy, task, context)
                
                return {
                    "transfer_result": result,
                    "transferred_strategy": strategy.name,
                    "source_domains": list(set(
                        exp.task_context.get("domain", "general") 
                        for exp in other_domain_experiences
                    )),
                    "mode": "transfer"
                }
        
        return {"message": "No knowledge to transfer"}
    
    async def _reflect_on_performance(self, task: Any) -> Dict[str, Any]:
        """Reflect on overall performance and learning"""
        
        # Comprehensive performance analysis
        analysis = {
            "total_experiences": len(self.learning_experiences),
            "total_strategies": len(self.learning_strategies),
            "current_mode": self.current_learning_mode.value,
            "performance_trend": await self._calculate_performance_improvement(),
            "strategy_analysis": await self._analyze_strategy_effectiveness(),
            "insights": await self._generate_learning_insights(),
            "recommendations": await self._generate_recommendations()
        }
        
        return {
            "reflection_result": analysis,
            "mode": "reflection"
        }
    
    def _extract_task_parameters(self, task: Any) -> Dict[str, Any]:
        """Extract parameters from task description"""
        params = {}
        
        if isinstance(task, dict):
            return task
        
        task_str = str(task).lower()
        
        # Extract domain if specified
        if "domain=" in task_str:
            domain_part = task_str.split("domain=")[1].split()[0]
            params["source_domain"] = domain_part.strip(",")
        
        if "target=" in task_str:
            target_part = task_str.split("target=")[1].split()[0]
            params["target_domain"] = target_part.strip(",")
        
        return params
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        return {
            "agent_name": self.name,
            "learning_mode": self.current_learning_mode.value,
            "total_strategies": len(self.learning_strategies),
            "total_experiences": len(self.learning_experiences),
            "performance_history_length": len(self.performance_history),
            "exploration_rate": self.exploration_rate,
            "best_strategies": asyncio.run(self._get_best_strategies(3)),
            "recent_success_rate": (
                sum(1 for exp in self.learning_experiences[-10:] if exp.success) / 
                min(10, len(self.learning_experiences))
            ) if self.learning_experiences else 0,
            "domains_experienced": list(set(
                exp.task_context.get("domain", "general")
                for exp in self.learning_experiences
            )),
            "learning_insights": asyncio.run(self._generate_learning_insights())[:3]
        }