"""
Enhanced Tree of Thoughts for Phase 7 - Autonomous Intelligence Ecosystem
Advanced reasoning with causal integration, adaptive pruning, and working memory coherence
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path
import heapq
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import networkx as nx

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ThoughtQuality(Enum):
    """Quality levels for thoughts"""
    POOR = 1
    FAIR = 2
    GOOD = 3
    EXCELLENT = 4
    BREAKTHROUGH = 5


class ReasoningStrategy(Enum):
    """Types of reasoning strategies"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    SYSTEMATIC = "systematic"
    INTUITIVE = "intuitive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    COUNTERFACTUAL = "counterfactual"
    ABDUCTIVE = "abductive"


class SearchStrategy(Enum):
    """Tree search strategies"""
    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BEST_FIRST = "best_first"
    MONTE_CARLO = "monte_carlo"
    BEAM_SEARCH = "beam_search"
    ADAPTIVE = "adaptive"


@dataclass
class Thought:
    """Enhanced thought node with causal integration and working memory coherence"""
    id: str
    content: str
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    depth: int = 0
    
    # Quality and scoring
    quality_score: float = 0.0
    confidence: float = 0.0
    novelty_score: float = 0.0
    feasibility_score: float = 0.0
    completeness_score: float = 0.0
    coherence_score: float = 0.0
    
    # Reasoning metadata
    reasoning_strategy: ReasoningStrategy = ReasoningStrategy.ANALYTICAL
    is_solution_candidate: bool = False
    is_breakthrough: bool = False
    
    # Causal reasoning integration
    causal_assumptions: List[str] = field(default_factory=list)
    causal_predictions: Dict[str, float] = field(default_factory=dict)
    causal_evidence: List[str] = field(default_factory=list)
    
    # Working memory integration
    memory_references: List[str] = field(default_factory=list)
    context_coherence: float = 0.0
    token_count: int = 0
    
    # Performance metrics
    evaluation_metrics: Dict[str, float] = field(default_factory=dict)
    generation_time: float = 0.0
    evaluation_time: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    reasoning_session: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        """Initialize computed fields"""
        if self.token_count == 0:
            self.token_count = max(1, len(self.content) // 4)  # Rough token estimate
    
    def calculate_composite_score(self) -> float:
        """Calculate composite score from all quality metrics"""
        weights = {
            'quality_score': 0.25,
            'confidence': 0.2,
            'novelty_score': 0.15,
            'feasibility_score': 0.15,
            'completeness_score': 0.15,
            'coherence_score': 0.1
        }
        
        composite = sum(
            getattr(self, metric) * weight 
            for metric, weight in weights.items()
        )
        
        return min(1.0, composite)
    
    def is_high_quality(self, threshold: float = 0.7) -> bool:
        """Check if thought meets high quality threshold"""
        return self.calculate_composite_score() >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'content': self.content,
            'parent_id': self.parent_id,
            'depth': self.depth,
            'quality_score': self.quality_score,
            'confidence': self.confidence,
            'reasoning_strategy': self.reasoning_strategy.value,
            'is_solution_candidate': self.is_solution_candidate,
            'causal_assumptions': self.causal_assumptions,
            'causal_predictions': self.causal_predictions,
            'memory_references': self.memory_references,
            'token_count': self.token_count,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ReasoningPath:
    """Complete reasoning path from root to solution"""
    thoughts: List[Thought]
    total_quality: float
    confidence: float
    reasoning_trace: str
    solution_quality: float
    causal_chain: List[Dict[str, Any]] = field(default_factory=list)
    memory_coherence: float = 0.0
    token_efficiency: float = 0.0
    
    def get_path_summary(self) -> Dict[str, Any]:
        """Get summary of reasoning path"""
        return {
            'thought_count': len(self.thoughts),
            'total_quality': self.total_quality,
            'confidence': self.confidence,
            'solution_quality': self.solution_quality,
            'memory_coherence': self.memory_coherence,
            'token_efficiency': self.token_efficiency,
            'reasoning_depth': max(t.depth for t in self.thoughts) if self.thoughts else 0,
            'breakthrough_thoughts': sum(1 for t in self.thoughts if t.is_breakthrough),
            'causal_chain_length': len(self.causal_chain)
        }


class CausalThoughtEvaluator:
    """Evaluates thoughts with causal reasoning integration"""
    
    def __init__(self, causal_engine=None):
        self.causal_engine = causal_engine
        self.evaluation_history = []
        
    async def evaluate_thought_with_causality(self, thought: Thought, 
                                            problem_context: Dict[str, Any],
                                            parent_thought: Optional[Thought] = None) -> float:
        """Evaluate thought quality with causal reasoning integration"""
        
        evaluation_start = datetime.now()
        
        # Base quality evaluation
        base_quality = await self._evaluate_base_quality(thought, problem_context)
        
        # Causal coherence evaluation
        causal_coherence = await self._evaluate_causal_coherence(thought, parent_thought)
        
        # Causal predictive power
        predictive_power = await self._evaluate_predictive_power(thought, problem_context)
        
        # Evidence consistency
        evidence_consistency = await self._evaluate_evidence_consistency(thought)
        
        # Combined causal-aware quality score
        causal_quality = (
            base_quality * 0.4 +
            causal_coherence * 0.25 +
            predictive_power * 0.2 +
            evidence_consistency * 0.15
        )
        
        # Update thought metrics
        thought.evaluation_metrics.update({
            'base_quality': base_quality,
            'causal_coherence': causal_coherence,
            'predictive_power': predictive_power,
            'evidence_consistency': evidence_consistency
        })
        
        evaluation_time = (datetime.now() - evaluation_start).total_seconds()
        thought.evaluation_time = evaluation_time
        
        return causal_quality
    
    async def _evaluate_base_quality(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Evaluate base thought quality"""
        
        # Relevance to problem
        relevance = await self._assess_relevance(thought.content, context.get('problem', ''))
        
        # Logical consistency
        logical_consistency = await self._assess_logical_consistency(thought.content)
        
        # Specificity and actionability
        specificity = await self._assess_specificity(thought.content)
        
        # Combine base metrics
        base_quality = (relevance * 0.4 + logical_consistency * 0.4 + specificity * 0.2)
        
        return base_quality
    
    async def _assess_relevance(self, content: str, problem: str) -> float:
        """Assess relevance to problem"""
        if not problem:
            return 0.5
        
        # Simple keyword overlap (would use embeddings in practice)
        problem_words = set(problem.lower().split())
        content_words = set(content.lower().split())
        
        if not problem_words:
            return 0.5
        
        overlap = len(problem_words & content_words)
        relevance = min(1.0, overlap / len(problem_words) * 2)  # Scale appropriately
        
        return relevance
    
    async def _assess_logical_consistency(self, content: str) -> float:
        """Assess logical consistency of thought"""
        
        # Check for logical indicators
        positive_indicators = ['therefore', 'thus', 'because', 'since', 'leads to', 'results in']
        negative_indicators = ['contradiction', 'impossible', 'cannot both', 'mutually exclusive']
        
        content_lower = content.lower()
        
        positive_count = sum(1 for indicator in positive_indicators if indicator in content_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in content_lower)
        
        # Base consistency score
        consistency = 0.7  # Default neutral score
        
        # Boost for logical connectors
        if positive_count > 0:
            consistency += min(0.2, positive_count * 0.1)
        
        # Penalty for contradictions
        if negative_count > 0:
            consistency -= min(0.4, negative_count * 0.2)
        
        return max(0.1, min(1.0, consistency))
    
    async def _assess_specificity(self, content: str) -> float:
        """Assess specificity and actionability"""
        
        # Look for specific indicators
        specific_indicators = ['step', 'method', 'approach', 'technique', 'algorithm', 'process']
        vague_indicators = ['maybe', 'perhaps', 'might', 'could', 'possibly']
        
        content_lower = content.lower()
        
        specific_count = sum(1 for indicator in specific_indicators if indicator in content_lower)
        vague_count = sum(1 for indicator in vague_indicators if indicator in content_lower)
        
        # Length as proxy for detail
        length_factor = min(1.0, len(content) / 200.0)
        
        specificity = (specific_count * 0.2 + length_factor * 0.5 - vague_count * 0.1)
        
        return max(0.1, min(1.0, specificity))
    
    async def _evaluate_causal_coherence(self, thought: Thought, parent_thought: Optional[Thought]) -> float:
        """Evaluate causal coherence with parent thought"""
        
        if not parent_thought:
            return 0.7  # Neutral score for root thoughts
        
        # Check for causal flow
        causal_flow = await self._assess_causal_flow(parent_thought.content, thought.content)
        
        # Check causal assumptions consistency
        assumption_consistency = await self._assess_assumption_consistency(
            parent_thought.causal_assumptions, thought.causal_assumptions
        )
        
        # Check prediction alignment
        prediction_alignment = await self._assess_prediction_alignment(
            parent_thought.causal_predictions, thought.causal_predictions
        )
        
        coherence = (causal_flow * 0.5 + assumption_consistency * 0.3 + prediction_alignment * 0.2)
        
        return coherence
    
    async def _assess_causal_flow(self, parent_content: str, child_content: str) -> float:
        """Assess causal flow between parent and child thoughts"""
        
        causal_connectors = [
            'therefore', 'thus', 'consequently', 'as a result', 'leading to',
            'because of this', 'this means', 'it follows that'
        ]
        
        child_lower = child_content.lower()
        
        # Check for explicit causal connectors
        connector_count = sum(1 for connector in causal_connectors if connector in child_lower)
        
        if connector_count > 0:
            return min(1.0, connector_count * 0.5 + 0.3)
        
        # Check for implicit causal flow (content building on previous)
        parent_words = set(parent_content.lower().split())
        child_words = set(child_content.lower().split())
        
        overlap = len(parent_words & child_words)
        if overlap > 0:
            return min(0.8, overlap / len(parent_words) + 0.3)
        
        return 0.3  # Minimal flow
    
    async def _assess_assumption_consistency(self, parent_assumptions: List[str], 
                                          child_assumptions: List[str]) -> float:
        """Assess consistency of causal assumptions"""
        
        if not parent_assumptions and not child_assumptions:
            return 1.0
        
        if not parent_assumptions or not child_assumptions:
            return 0.7  # Neutral when one is empty
        
        # Check for consistency (no contradictions)
        parent_set = set(assumption.lower() for assumption in parent_assumptions)
        child_set = set(assumption.lower() for assumption in child_assumptions)
        
        # Look for contradictions
        contradiction_pairs = [
            ('not', ''), ('cannot', 'can'), ('never', 'always'),
            ('impossible', 'possible'), ('false', 'true')
        ]
        
        contradictions = 0
        for parent_assumption in parent_set:
            for child_assumption in child_set:
                for neg, pos in contradiction_pairs:
                    if (neg in parent_assumption and pos in child_assumption) or \
                       (pos in parent_assumption and neg in child_assumption):
                        contradictions += 1
        
        # Calculate consistency
        total_comparisons = len(parent_set) * len(child_set)
        if total_comparisons == 0:
            return 1.0
        
        consistency = 1.0 - (contradictions / total_comparisons)
        return max(0.0, consistency)
    
    async def _assess_prediction_alignment(self, parent_predictions: Dict[str, float],
                                         child_predictions: Dict[str, float]) -> float:
        """Assess alignment of causal predictions"""
        
        if not parent_predictions or not child_predictions:
            return 0.7  # Neutral when predictions missing
        
        # Find common prediction variables
        common_vars = set(parent_predictions.keys()) & set(child_predictions.keys())
        
        if not common_vars:
            return 0.5  # No overlap
        
        # Calculate alignment for common variables
        alignment_scores = []
        
        for var in common_vars:
            parent_pred = parent_predictions[var]
            child_pred = child_predictions[var]
            
            # Calculate similarity (1 - normalized difference)
            max_val = max(abs(parent_pred), abs(child_pred), 1.0)
            difference = abs(parent_pred - child_pred) / max_val
            alignment = 1.0 - difference
            
            alignment_scores.append(alignment)
        
        return np.mean(alignment_scores) if alignment_scores else 0.5
    
    async def _evaluate_predictive_power(self, thought: Thought, context: Dict[str, Any]) -> float:
        """Evaluate predictive power of thought"""
        
        # Check for predictive statements
        predictive_indicators = [
            'will', 'should', 'expect', 'predict', 'forecast', 'anticipate',
            'likely', 'probable', 'estimate', 'project'
        ]
        
        content_lower = thought.content.lower()
        predictive_count = sum(1 for indicator in predictive_indicators if indicator in content_lower)
        
        # Base predictive power from indicators
        base_power = min(0.8, predictive_count * 0.2)
        
        # Boost if thought includes quantitative predictions
        quantitative_boost = 0.0
        if thought.causal_predictions:
            quantitative_boost = min(0.3, len(thought.causal_predictions) * 0.1)
        
        # Check if predictions are grounded in evidence
        evidence_grounding = 0.0
        if thought.causal_evidence:
            evidence_grounding = min(0.2, len(thought.causal_evidence) * 0.1)
        
        predictive_power = base_power + quantitative_boost + evidence_grounding
        
        return min(1.0, predictive_power)
    
    async def _evaluate_evidence_consistency(self, thought: Thought) -> float:
        """Evaluate consistency of evidence with thought"""
        
        if not thought.causal_evidence:
            return 0.6  # Neutral score when no evidence
        
        # Check for evidence quality indicators
        strong_evidence_indicators = [
            'data shows', 'research indicates', 'studies demonstrate',
            'empirical evidence', 'statistical analysis', 'proven'
        ]
        
        weak_evidence_indicators = [
            'anecdotal', 'unverified', 'rumor', 'speculation',
            'without evidence', 'unsubstantiated'
        ]
        
        content_lower = thought.content.lower()
        
        strong_count = sum(1 for indicator in strong_evidence_indicators if indicator in content_lower)
        weak_count = sum(1 for indicator in weak_evidence_indicators if indicator in content_lower)
        
        # Base consistency from evidence count
        evidence_count = len(thought.causal_evidence)
        base_consistency = min(0.8, evidence_count * 0.1 + 0.3)
        
        # Adjust based on evidence quality indicators
        quality_adjustment = (strong_count * 0.1) - (weak_count * 0.15)
        
        consistency = base_consistency + quality_adjustment
        
        return max(0.1, min(1.0, consistency))


class WorkingMemoryIntegrator:
    """Integrates Tree of Thoughts with Working Memory System"""
    
    def __init__(self, working_memory_system=None):
        self.working_memory_system = working_memory_system
        self.thought_memory_mapping = {}  # thought_id -> memory_id
        
    async def integrate_thought_with_memory(self, thought: Thought, 
                                          session_id: Optional[str] = None) -> List[str]:
        """Integrate thought with working memory system"""
        
        if not self.working_memory_system:
            return []
        
        memory_ids = []
        
        # Store thought as working memory
        try:
            from .working_memory import MemoryType, MemoryImportance
            
            # Determine importance based on thought quality
            if thought.quality_score >= 0.8:
                importance = MemoryImportance.CRITICAL
            elif thought.quality_score >= 0.6:
                importance = MemoryImportance.HIGH
            else:
                importance = MemoryImportance.MEDIUM
            
            # Create memory context
            memory_context = {
                'thought_id': thought.id,
                'reasoning_strategy': thought.reasoning_strategy.value,
                'depth': thought.depth,
                'is_solution_candidate': thought.is_solution_candidate,
                'causal_assumptions': thought.causal_assumptions,
                'causal_predictions': thought.causal_predictions
            }
            
            # Store in working memory
            memory_id = await self.working_memory_system.store_memory(
                content=thought.content,
                memory_type=MemoryType.WORKING,
                importance=importance,
                tags=thought.tags.union({'tree_of_thoughts', f'depth_{thought.depth}'}),
                context=memory_context,
                session_id=session_id
            )
            
            memory_ids.append(memory_id)
            self.thought_memory_mapping[thought.id] = memory_id
            thought.memory_references.append(memory_id)
            
        except Exception as e:
            logger.error(f"Failed to integrate thought with memory: {e}")
        
        return memory_ids
    
    async def retrieve_relevant_memories(self, thought: Thought, 
                                       session_id: Optional[str] = None,
                                       max_memories: int = 5) -> List[Any]:
        """Retrieve memories relevant to current thought"""
        
        if not self.working_memory_system:
            return []
        
        try:
            from .working_memory import MemoryQuery, MemoryType
            
            # Create query based on thought content and context
            query = MemoryQuery(
                content=thought.content,
                context={'reasoning_strategy': thought.reasoning_strategy.value},
                memory_types=[MemoryType.WORKING, MemoryType.SEMANTIC, MemoryType.EPISODIC],
                max_results=max_memories,
                max_tokens=1000,  # Limit token budget
                min_relevance=0.3,
                require_session_continuity=True
            )
            
            # Retrieve relevant memories
            memories = await self.working_memory_system.retrieve_memories(query, session_id)
            
            # Update thought with memory references
            for memory in memories:
                if memory.id not in thought.memory_references:
                    thought.memory_references.append(memory.id)
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant memories: {e}")
            return []
    
    async def calculate_memory_coherence(self, thoughts: List[Thought], 
                                       session_id: Optional[str] = None) -> float:
        """Calculate coherence between thoughts and memory system"""
        
        if not self.working_memory_system or not thoughts:
            return 0.5
        
        try:
            # Get all memory references from thoughts
            all_memory_refs = set()
            for thought in thoughts:
                all_memory_refs.update(thought.memory_references)
            
            if not all_memory_refs:
                return 0.5
            
            # Get memories and calculate coherence
            memories = []
            for memory_id in all_memory_refs:
                memory = self.working_memory_system._get_memory_by_id(memory_id)
                if memory:
                    memories.append(memory)
            
            if len(memories) < 2:
                return 1.0
            
            # Calculate coherence using working memory system
            coherence = await self.working_memory_system.coherence_tracker.calculate_coherence(memories)
            
            return coherence
            
        except Exception as e:
            logger.error(f"Failed to calculate memory coherence: {e}")
            return 0.5


class EnhancedTreeOfThoughts:
    """
    Enhanced Tree of Thoughts with causal reasoning and working memory integration
    Optimized for complex problem-solving with adaptive search strategies
    """
    
    def __init__(self, 
                 branching_factor: int = 3,
                 max_depth: int = 6,
                 pruning_threshold: float = 0.3,
                 token_budget: int = 5000,
                 causal_engine=None,
                 working_memory_system=None):
        
        self.branching_factor = branching_factor
        self.max_depth = max_depth
        self.base_pruning_threshold = pruning_threshold
        self.token_budget = token_budget
        
        # Integration components
        self.causal_engine = causal_engine
        self.working_memory_system = working_memory_system
        self.memory_integrator = WorkingMemoryIntegrator(working_memory_system)
        self.causal_evaluator = CausalThoughtEvaluator(causal_engine)
        
        # Tree structure
        self.thought_tree: Dict[str, Thought] = {}
        self.root_thoughts: List[Thought] = []
        self.solution_candidates: List[Thought] = []
        
        # Search configuration
        self.search_strategy = SearchStrategy.ADAPTIVE
        self.current_session_id: Optional[str] = None
        
        # Performance tracking
        self.performance_metrics = {
            'thoughts_generated': 0,
            'thoughts_evaluated': 0,
            'thoughts_pruned': 0,
            'solutions_found': 0,
            'average_depth': 0.0,
            'total_tokens_used': 0,
            'search_efficiency': 0.0
        }
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Initialized Enhanced Tree of Thoughts with causal and memory integration")
    
    async def solve_with_integration(self, 
                                   problem: str,
                                   context: Optional[Dict[str, Any]] = None,
                                   session_id: Optional[str] = None,
                                   target_accuracy: float = 0.8) -> Optional[ReasoningPath]:
        """Solve problem with full causal and memory integration"""
        
        logger.info(f"Starting integrated reasoning for: {problem[:100]}...")
        start_time = datetime.now()
        
        if context is None:
            context = {}
        
        self.current_session_id = session_id
        context['problem'] = problem
        context['target_accuracy'] = target_accuracy
        
        try:
            # Initialize working memory session if needed
            if session_id and self.working_memory_system:
                await self.working_memory_system.start_reasoning_session(session_id)
            
            # Generate diverse initial thoughts with causal awareness
            root_thoughts = await self._generate_causal_aware_roots(problem, context)
            self.root_thoughts = root_thoughts
            
            # Add thoughts to tree and integrate with memory
            for thought in root_thoughts:
                self.thought_tree[thought.id] = thought
                if self.memory_integrator:
                    await self.memory_integrator.integrate_thought_with_memory(thought, session_id)
            
            # Adaptive search with multiple strategies
            best_solution = await self._adaptive_search_with_integration(context, target_accuracy)
            
            if best_solution:
                # Construct comprehensive reasoning path
                reasoning_path = await self._construct_integrated_path(best_solution, context)
                
                # Update performance metrics
                await self._update_performance_metrics(reasoning_path)
                
                # Maintain memory coherence
                if self.working_memory_system:
                    await self.working_memory_system.maintain_coherence(session_id)
                
                search_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Integrated reasoning completed in {search_time:.2f}s with quality {reasoning_path.solution_quality:.3f}")
                
                global_metrics.timing("enhanced_tot.solve_time", search_time)
                global_metrics.gauge("enhanced_tot.solution_quality", reasoning_path.solution_quality)
                
                return reasoning_path
            
            else:
                logger.warning("No solution found within search limits")
                return {}
        
        except Exception as e:
            logger.error(f"Integrated reasoning failed: {e}")
            global_metrics.incr("enhanced_tot.solve_errors")
            raise
    
    async def _generate_causal_aware_roots(self, problem: str, context: Dict[str, Any]) -> List[Thought]:
        """Generate root thoughts with causal awareness"""
        
        root_thoughts = []
        
        # Different reasoning strategies for diversity
        strategies = [
            ReasoningStrategy.ANALYTICAL,
            ReasoningStrategy.CAUSAL,
            ReasoningStrategy.CREATIVE,
            ReasoningStrategy.SYSTEMATIC,
            ReasoningStrategy.COUNTERFACTUAL
        ]
        
        generation_tasks = []
        for i, strategy in enumerate(strategies[:self.branching_factor]):
            task = self._generate_strategic_thought(problem, context, strategy, i)
            generation_tasks.append(task)
        
        # Generate thoughts in parallel
        thoughts = await asyncio.gather(*generation_tasks)
        
        # Evaluate and enhance with causal information
        for thought in thoughts:
            if thought:
                # Evaluate with causal integration
                thought.quality_score = await self.causal_evaluator.evaluate_thought_with_causality(
                    thought, context
                )
                
                # Add causal enhancements
                await self._enhance_with_causal_reasoning(thought, problem, context)
                
                # Retrieve relevant memories
                if self.memory_integrator:
                    memories = await self.memory_integrator.retrieve_relevant_memories(
                        thought, self.current_session_id
                    )
                    thought.context_coherence = await self._calculate_context_coherence(thought, memories)
                
                root_thoughts.append(thought)
        
        # Sort by quality and return best thoughts
        root_thoughts.sort(key=lambda t: t.quality_score, reverse=True)
        return root_thoughts[:self.branching_factor]
    
    async def _generate_strategic_thought(self, problem: str, context: Dict[str, Any], 
                                        strategy: ReasoningStrategy, index: int) -> Optional[Thought]:
        """Generate thought using specific reasoning strategy"""
        
        generation_start = datetime.now()
        
        try:
            # Strategy-specific content generation
            content = await self._generate_content_for_strategy(problem, context, strategy)
            
            if not content:
                return {}
            
            # Create thought
            thought = Thought(
                id=f"root_{strategy.value}_{index:02d}",
                content=content,
                depth=0,
                reasoning_strategy=strategy,
                reasoning_session=self.current_session_id,
                tags={strategy.value, 'root_thought'}
            )
            
            # Add strategy-specific metadata
            if strategy == ReasoningStrategy.CAUSAL:
                thought.tags.add('causal_reasoning')
                thought.causal_assumptions = await self._extract_causal_assumptions(content)
            elif strategy == ReasoningStrategy.COUNTERFACTUAL:
                thought.tags.add('counterfactual')
                thought.causal_predictions = await self._generate_counterfactual_predictions(content, context)
            
            thought.generation_time = (datetime.now() - generation_start).total_seconds()
            
            self.performance_metrics['thoughts_generated'] += 1
            global_metrics.incr("enhanced_tot.thoughts_generated")
            
            return thought
            
        except Exception as e:
            logger.error(f"Failed to generate {strategy.value} thought: {e}")
            return {}
    
    async def _generate_content_for_strategy(self, problem: str, context: Dict[str, Any], 
                                           strategy: ReasoningStrategy) -> str:
        """Generate content based on reasoning strategy"""
        
        strategy_prompts = {
            ReasoningStrategy.ANALYTICAL: f"Let me analyze this problem systematically by breaking down the key components and their relationships: {problem}",
            
            ReasoningStrategy.CAUSAL: f"To understand this problem, I need to identify the causal relationships and mechanisms at work: {problem}",
            
            ReasoningStrategy.CREATIVE: f"Looking at this problem from a creative perspective, what novel approaches could work: {problem}",
            
            ReasoningStrategy.SYSTEMATIC: f"I'll approach this problem using a structured, step-by-step methodology: {problem}",
            
            ReasoningStrategy.COUNTERFACTUAL: f"What would happen if we changed key assumptions or conditions in this problem: {problem}",
            
            ReasoningStrategy.ABDUCTIVE: f"Given the observations, what's the most likely explanation for this problem: {problem}",
            
            ReasoningStrategy.ANALOGICAL: f"This problem reminds me of similar situations. By analogy: {problem}",
            
            ReasoningStrategy.INTUITIVE: f"My intuitive sense of this problem suggests: {problem}"
        }
        
        base_prompt = strategy_prompts.get(strategy, f"Thinking about: {problem}")
        
        # Strategy-specific content generation
        if strategy == ReasoningStrategy.ANALYTICAL:
            content = f"{base_prompt} First, I need to identify the main variables: problem scope, constraints, available resources, and success criteria. The key relationships appear to involve..."
            
        elif strategy == ReasoningStrategy.CAUSAL:
            content = f"{base_prompt} The causal structure suggests that X influences Y through mechanism Z. If we intervene on A, we can expect changes in B because..."
            
        elif strategy == ReasoningStrategy.CREATIVE:
            content = f"{base_prompt} What if we approached this completely differently? Instead of traditional methods, we could combine concepts from different domains..."
            
        elif strategy == ReasoningStrategy.SYSTEMATIC:
            content = f"{base_prompt} Step 1: Define the problem clearly. Step 2: Gather relevant information. Step 3: Generate alternatives. Step 4: Evaluate options..."
            
        elif strategy == ReasoningStrategy.COUNTERFACTUAL:
            content = f"{base_prompt} If the initial conditions were different, or if we removed certain constraints, the problem would transform into..."
            
        else:
            content = f"{base_prompt} This requires careful consideration of multiple factors and their interactions..."
        
        return content
    
    async def _enhance_with_causal_reasoning(self, thought: Thought, problem: str, context: Dict[str, Any]):
        """Enhance thought with causal reasoning capabilities"""
        
        # Extract causal assumptions if not already present
        if not thought.causal_assumptions:
            thought.causal_assumptions = await self._extract_causal_assumptions(thought.content)
        
        # Generate causal predictions if applicable
        if thought.reasoning_strategy in [ReasoningStrategy.CAUSAL, ReasoningStrategy.COUNTERFACTUAL]:
            thought.causal_predictions = await self._generate_causal_predictions(thought, context)
        
        # Collect causal evidence
        thought.causal_evidence = await self._collect_causal_evidence(thought, problem)
        
        # Integrate with causal engine if available
        if self.causal_engine:
            try:
                # Add domain knowledge or constraints from causal engine
                causal_context = await self._get_causal_context(thought, problem)
                thought.context.update(causal_context)
            except Exception as e:
                logger.debug(f"Could not integrate with causal engine: {e}")
    
    async def _extract_causal_assumptions(self, content: str) -> List[str]:
        """Extract causal assumptions from thought content"""
        
        assumptions = []
        
        # Look for assumption indicators
        assumption_indicators = [
            'assuming', 'given that', 'if we assume', 'supposing that',
            'based on the assumption', 'under the condition that'
        ]
        
        content_lower = content.lower()
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if any(indicator in sentence.lower() for indicator in assumption_indicators):
                assumptions.append(sentence)
        
        # Default assumptions for causal reasoning
        if not assumptions:
            assumptions = [
                "The observed relationships reflect causal mechanisms",
                "The system operates under stable conditions",
                "Key variables are observable and measurable"
            ]
        
        return assumptions[:5]  # Limit to top 5 assumptions
    
    async def _generate_causal_predictions(self, thought: Thought, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate causal predictions from thought"""
        
        predictions = {}
        
        # Extract predictive statements
        predictive_phrases = [
            'will result in', 'should lead to', 'is likely to cause',
            'we can expect', 'this will increase', 'this will decrease'
        ]
        
        content_lower = thought.content.lower()
        
        # Simple prediction extraction
        if 'increase' in content_lower:
            predictions['positive_outcome'] = 0.7
        if 'decrease' in content_lower:
            predictions['negative_outcome'] = 0.6
        if 'improve' in content_lower:
            predictions['improvement'] = 0.8
        if 'problem' in content_lower and 'solve' in content_lower:
            predictions['solution_effectiveness'] = 0.75
        
        return predictions
    
    async def _generate_counterfactual_predictions(self, content: str, context: Dict[str, Any]) -> Dict[str, float]:
        """Generate counterfactual predictions"""
        
        predictions = {}
        
        # Counterfactual indicators
        if 'if not' in content.lower():
            predictions['alternative_outcome'] = 0.6
        if 'without' in content.lower():
            predictions['absence_effect'] = 0.5
        if 'instead' in content.lower():
            predictions['substitution_effect'] = 0.7
        
        return predictions
    
    async def _collect_causal_evidence(self, thought: Thought, problem: str) -> List[str]:
        """Collect evidence supporting causal claims"""
        
        evidence = []
        
        # Look for evidence indicators
        evidence_indicators = [
            'research shows', 'data indicates', 'studies demonstrate',
            'evidence suggests', 'empirical findings', 'statistical analysis'
        ]
        
        content_lower = thought.content.lower()
        
        for indicator in evidence_indicators:
            if indicator in content_lower:
                evidence.append(f"Thought references {indicator}")
        
        # Add domain-specific evidence
        if 'financial' in problem.lower():
            evidence.append("Financial domain expertise")
        if 'technical' in problem.lower() or 'system' in problem.lower():
            evidence.append("Technical domain knowledge")
        
        return evidence
    
    async def _get_causal_context(self, thought: Thought, problem: str) -> Dict[str, Any]:
        """Get causal context from causal engine"""
        
        causal_context = {}
        
        try:
            if self.causal_engine and hasattr(self.causal_engine, 'get_causal_summary'):
                summary = await self.causal_engine.get_causal_summary()
                causal_context['causal_relationships'] = summary.get('total_relationships', 0)
                causal_context['causal_confidence'] = summary.get('strong_relationships', 0)
        except Exception as e:
            logger.debug(f"Could not get causal context: {e}")
        
        return causal_context
    
    async def _calculate_context_coherence(self, thought: Thought, memories: List[Any]) -> float:
        """Calculate context coherence with retrieved memories"""
        
        if not memories:
            return 0.5
        
        try:
            # Simple coherence based on content overlap
            thought_words = set(thought.content.lower().split())
            
            coherence_scores = []
            for memory in memories:
                memory_words = set(memory.content.lower().split())
                if thought_words and memory_words:
                    overlap = len(thought_words & memory_words)
                    union = len(thought_words | memory_words)
                    coherence = overlap / union if union > 0 else 0.0
                    coherence_scores.append(coherence)
            
            return np.mean(coherence_scores) if coherence_scores else 0.5
            
        except Exception as e:
            logger.debug(f"Could not calculate context coherence: {e}")
            return 0.5
    
    async def _adaptive_search_with_integration(self, context: Dict[str, Any], 
                                              target_accuracy: float) -> Optional[Thought]:
        """Adaptive search with causal and memory integration"""
        
        # Try different search strategies
        strategies = [SearchStrategy.BEST_FIRST, SearchStrategy.BEAM_SEARCH, SearchStrategy.MONTE_CARLO]
        
        best_solution = None
        best_quality = 0.0
        
        for strategy in strategies:
            logger.info(f"Trying search strategy: {strategy.value}")
            
            # Reset for new strategy
            self.solution_candidates.clear()
            
            try:
                solution = await self._execute_search_strategy(strategy, context, target_accuracy)
                
                if solution and solution.quality_score > best_quality:
                    best_solution = solution
                    best_quality = solution.quality_score
                
                # Early termination if target accuracy reached
                if best_quality >= target_accuracy:
                    logger.info(f"Target accuracy {target_accuracy:.3f} reached with {strategy.value}")
                    break
                    
            except Exception as e:
                logger.warning(f"Search strategy {strategy.value} failed: {e}")
                continue
        
        return best_solution
    
    async def _execute_search_strategy(self, strategy: SearchStrategy, 
                                     context: Dict[str, Any], 
                                     target_accuracy: float) -> Optional[Thought]:
        """Execute specific search strategy"""
        
        if strategy == SearchStrategy.BEST_FIRST:
            return await self._best_first_search(context, target_accuracy)
        elif strategy == SearchStrategy.BEAM_SEARCH:
            return await self._beam_search(context, target_accuracy)
        elif strategy == SearchStrategy.MONTE_CARLO:
            return await self._monte_carlo_search(context, target_accuracy)
        else:
            return await self._depth_first_search(context, target_accuracy)
    
    async def _best_first_search(self, context: Dict[str, Any], target_accuracy: float) -> Optional[Thought]:
        """Best-first search with priority queue"""
        
        # Priority queue: (negative_quality_score, thought)
        priority_queue = []
        
        # Initialize with root thoughts
        for thought in self.root_thoughts:
            heapq.heappush(priority_queue, (-thought.quality_score, thought))
        
        explored = set()
        
        while priority_queue and len(explored) < 1000:  # Limit exploration
            neg_quality, current_thought = heapq.heappop(priority_queue)
            
            if current_thought.id in explored:
                continue
            
            explored.add(current_thought.id)
            
            # Check if solution candidate
            if (current_thought.quality_score >= target_accuracy or 
                current_thought.completeness_score >= 0.8):
                current_thought.is_solution_candidate = True
                self.solution_candidates.append(current_thought)
                return current_thought
            
            # Generate children if not at max depth
            if current_thought.depth < self.max_depth:
                children = await self._generate_child_thoughts_integrated(current_thought, context)
                
                for child in children:
                    if child.id not in explored:
                        heapq.heappush(priority_queue, (-child.quality_score, child))
        
        # Return best solution candidate if any
        if self.solution_candidates:
            return max(self.solution_candidates, key=lambda t: t.quality_score)
        
        return {}
    
    async def _beam_search(self, context: Dict[str, Any], target_accuracy: float, 
                          beam_width: int = 3) -> Optional[Thought]:
        """Beam search with limited branching"""
        
        current_beam = list(self.root_thoughts)
        
        for depth in range(1, self.max_depth + 1):
            next_beam = []
            
            # Generate children for each thought in current beam
            for thought in current_beam:
                children = await self._generate_child_thoughts_integrated(thought, context)
                next_beam.extend(children)
            
            if not next_beam:
                break
            
            # Keep only top beam_width thoughts
            next_beam.sort(key=lambda t: t.quality_score, reverse=True)
            current_beam = next_beam[:beam_width]
            
            # Check for solutions
            for thought in current_beam:
                if thought.quality_score >= target_accuracy:
                    thought.is_solution_candidate = True
                    self.solution_candidates.append(thought)
                    return thought
        
        # Return best from final beam
        if current_beam:
            return max(current_beam, key=lambda t: t.quality_score)
        
        return {}
    
    async def _monte_carlo_search(self, context: Dict[str, Any], target_accuracy: float,
                                simulations: int = 50) -> Optional[Thought]:
        """Monte Carlo Tree Search"""
        
        best_thought = None
        best_quality = 0.0
        
        for _ in range(simulations):
            # Random path exploration
            current_thought = np.random.choice(self.root_thoughts)
            path = [current_thought]
            
            # Simulate random path
            for depth in range(self.max_depth):
                if np.random.random() < 0.7:  # 70% chance to continue
                    children = await self._generate_child_thoughts_integrated(current_thought, context)
                    if children:
                        current_thought = np.random.choice(children)
                        path.append(current_thought)
                else:
                    break
            
            # Evaluate final thought
            final_thought = path[-1]
            if final_thought.quality_score > best_quality:
                best_thought = final_thought
                best_quality = final_thought.quality_score
                
                if best_quality >= target_accuracy:
                    break
        
        if best_thought:
            best_thought.is_solution_candidate = True
            self.solution_candidates.append(best_thought)
        
        return best_thought
    
    async def _depth_first_search(self, context: Dict[str, Any], target_accuracy: float) -> Optional[Thought]:
        """Depth-first search with early termination"""
        
        for root_thought in self.root_thoughts:
            result = await self._dfs_recursive(root_thought, context, target_accuracy, 0)
            if result:
                return result
        
        return {}
    
    async def _dfs_recursive(self, thought: Thought, context: Dict[str, Any], 
                           target_accuracy: float, depth: int) -> Optional[Thought]:
        """Recursive DFS helper"""
        
        if thought.quality_score >= target_accuracy:
            thought.is_solution_candidate = True
            self.solution_candidates.append(thought)
            return thought
        
        if depth >= self.max_depth:
            return {}
        
        # Generate and explore children
        children = await self._generate_child_thoughts_integrated(thought, context)
        
        for child in children:
            result = await self._dfs_recursive(child, context, target_accuracy, depth + 1)
            if result:
                return result
        
        return {}
    
    async def _generate_child_thoughts_integrated(self, parent_thought: Thought, 
                                                context: Dict[str, Any]) -> List[Thought]:
        """Generate child thoughts with full integration"""
        
        child_thoughts = []
        
        # Generate children based on parent's reasoning strategy
        generation_tasks = []
        for i in range(self.branching_factor):
            task = self._generate_single_child(parent_thought, context, i)
            generation_tasks.append(task)
        
        # Generate in parallel
        children = await asyncio.gather(*generation_tasks)
        
        # Process and integrate each child
        for child in children:
            if child:
                # Evaluate with causal integration
                child.quality_score = await self.causal_evaluator.evaluate_thought_with_causality(
                    child, context, parent_thought
                )
                
                # Calculate coherence scores
                child.coherence_score = await self._calculate_thought_coherence(child, parent_thought)
                
                # Integrate with memory
                if self.memory_integrator:
                    await self.memory_integrator.integrate_thought_with_memory(
                        child, self.current_session_id
                    )
                    
                    memories = await self.memory_integrator.retrieve_relevant_memories(
                        child, self.current_session_id
                    )
                    child.context_coherence = await self._calculate_context_coherence(child, memories)
                
                # Add to tree
                self.thought_tree[child.id] = child
                parent_thought.children_ids.append(child.id)
                
                # Check if should be pruned
                if child.quality_score >= self.base_pruning_threshold:
                    child_thoughts.append(child)
                else:
                    self.performance_metrics['thoughts_pruned'] += 1
        
        self.performance_metrics['thoughts_evaluated'] += len(children)
        
        return child_thoughts
    
    async def _generate_single_child(self, parent_thought: Thought, context: Dict[str, Any], 
                                   child_index: int) -> Optional[Thought]:
        """Generate single child thought"""
        
        generation_start = datetime.now()
        
        try:
            # Determine child reasoning strategy
            child_strategy = await self._select_child_strategy(parent_thought, child_index)
            
            # Generate content based on parent and strategy
            child_content = await self._generate_child_content(parent_thought, context, child_strategy)
            
            if not child_content:
                return {}
            
            # Create child thought
            child_thought = Thought(
                id=f"{parent_thought.id}_child_{child_index:02d}",
                content=child_content,
                parent_id=parent_thought.id,
                depth=parent_thought.depth + 1,
                reasoning_strategy=child_strategy,
                reasoning_session=self.current_session_id,
                tags=parent_thought.tags.union({child_strategy.value, f'depth_{parent_thought.depth + 1}'})
            )
            
            # Inherit and extend causal information
            child_thought.causal_assumptions = parent_thought.causal_assumptions.copy()
            child_thought.causal_predictions.update(parent_thought.causal_predictions)
            
            # Add new causal information
            if child_strategy == ReasoningStrategy.CAUSAL:
                new_assumptions = await self._extract_causal_assumptions(child_content)
                child_thought.causal_assumptions.extend(new_assumptions)
            
            child_thought.generation_time = (datetime.now() - generation_start).total_seconds()
            self.performance_metrics['thoughts_generated'] += 1
            
            return child_thought
            
        except Exception as e:
            logger.error(f"Failed to generate child thought: {e}")
            return {}
    
    async def _select_child_strategy(self, parent_thought: Thought, child_index: int) -> ReasoningStrategy:
        """Select reasoning strategy for child thought"""
        
        # Strategy progression patterns
        strategy_progressions = {
            ReasoningStrategy.ANALYTICAL: [ReasoningStrategy.SYSTEMATIC, ReasoningStrategy.CAUSAL, ReasoningStrategy.CREATIVE],
            ReasoningStrategy.CAUSAL: [ReasoningStrategy.COUNTERFACTUAL, ReasoningStrategy.ABDUCTIVE, ReasoningStrategy.SYSTEMATIC],
            ReasoningStrategy.CREATIVE: [ReasoningStrategy.ANALOGICAL, ReasoningStrategy.INTUITIVE, ReasoningStrategy.ANALYTICAL],
            ReasoningStrategy.SYSTEMATIC: [ReasoningStrategy.ANALYTICAL, ReasoningStrategy.CAUSAL, ReasoningStrategy.CREATIVE],
            ReasoningStrategy.COUNTERFACTUAL: [ReasoningStrategy.CAUSAL, ReasoningStrategy.ABDUCTIVE, ReasoningStrategy.ANALYTICAL]
        }
        
        possible_strategies = strategy_progressions.get(
            parent_thought.reasoning_strategy,
            [ReasoningStrategy.ANALYTICAL, ReasoningStrategy.CAUSAL, ReasoningStrategy.CREATIVE]
        )
        
        return possible_strategies[child_index % len(possible_strategies)]
    
    async def _generate_child_content(self, parent_thought: Thought, context: Dict[str, Any], 
                                    child_strategy: ReasoningStrategy) -> str:
        """Generate content for child thought"""
        
        # Build on parent content
        parent_content = parent_thought.content
        
        if child_strategy == ReasoningStrategy.CAUSAL:
            child_content = f"Building on the idea that '{parent_content[:50]}...', the causal mechanism suggests that if we implement this approach, we can expect specific outcomes because..."
            
        elif child_strategy == ReasoningStrategy.COUNTERFACTUAL:
            child_content = f"Considering '{parent_content[:50]}...', what would happen if we changed key assumptions? Without these constraints, the solution would involve..."
            
        elif child_strategy == ReasoningStrategy.SYSTEMATIC:
            child_content = f"To systematically implement '{parent_content[:50]}...', we need to break this down into specific steps: First, establish the framework. Second, identify resources. Third, execute the plan..."
            
        elif child_strategy == ReasoningStrategy.CREATIVE:
            child_content = f"Taking a creative approach to '{parent_content[:50]}...', we could combine this with unconventional methods from other domains..."
            
        elif child_strategy == ReasoningStrategy.ABDUCTIVE:
            child_content = f"Given the premise in '{parent_content[:50]}...', the most likely explanation that accounts for all observations is..."
            
        else:
            child_content = f"Expanding on '{parent_content[:50]}...', this leads us to consider additional factors and implications..."
        
        return child_content
    
    async def _calculate_thought_coherence(self, child_thought: Thought, parent_thought: Thought) -> float:
        """Calculate coherence between child and parent thoughts"""
        
        # Content coherence
        content_coherence = await self._calculate_content_coherence(child_thought.content, parent_thought.content)
        
        # Strategy coherence
        strategy_coherence = await self._calculate_strategy_coherence(child_thought.reasoning_strategy, parent_thought.reasoning_strategy)
        
        # Causal coherence
        causal_coherence = await self._calculate_causal_coherence(child_thought, parent_thought)
        
        # Combined coherence
        total_coherence = (content_coherence * 0.4 + strategy_coherence * 0.3 + causal_coherence * 0.3)
        
        return total_coherence
    
    async def _calculate_content_coherence(self, child_content: str, parent_content: str) -> float:
        """Calculate content coherence between thoughts"""
        
        # Word overlap
        child_words = set(child_content.lower().split())
        parent_words = set(parent_content.lower().split())
        
        if not parent_words:
            return 0.5
        
        overlap = len(child_words & parent_words)
        coherence = min(1.0, overlap / len(parent_words) * 2)
        
        # Boost for logical connectors
        logical_connectors = ['therefore', 'thus', 'because', 'since', 'building on', 'expanding']
        connector_boost = sum(0.1 for connector in logical_connectors if connector in child_content.lower())
        
        return min(1.0, coherence + connector_boost)
    
    async def _calculate_strategy_coherence(self, child_strategy: ReasoningStrategy, parent_strategy: ReasoningStrategy) -> float:
        """Calculate strategy coherence"""
        
        # Compatible strategy pairs
        compatible_pairs = {
            (ReasoningStrategy.ANALYTICAL, ReasoningStrategy.SYSTEMATIC): 0.9,
            (ReasoningStrategy.CAUSAL, ReasoningStrategy.COUNTERFACTUAL): 0.9,
            (ReasoningStrategy.CREATIVE, ReasoningStrategy.ANALOGICAL): 0.8,
            (ReasoningStrategy.SYSTEMATIC, ReasoningStrategy.ANALYTICAL): 0.9,
            (ReasoningStrategy.ABDUCTIVE, ReasoningStrategy.CAUSAL): 0.8
        }
        
        pair = (parent_strategy, child_strategy)
        return compatible_pairs.get(pair, 0.6)  # Default moderate coherence
    
    async def _calculate_causal_coherence(self, child_thought: Thought, parent_thought: Thought) -> float:
        """Calculate causal coherence between thoughts"""
        
        # Check assumption consistency
        assumption_consistency = 1.0
        if parent_thought.causal_assumptions and child_thought.causal_assumptions:
            parent_set = set(parent_thought.causal_assumptions)
            child_set = set(child_thought.causal_assumptions)
            
            # Look for contradictions
            contradictions = 0
            for p_assumption in parent_set:
                for c_assumption in child_set:
                    if ('not' in p_assumption and 'not' not in c_assumption) or \
                       ('not' not in p_assumption and 'not' in c_assumption):
                        if any(word in p_assumption for word in c_assumption.split()):
                            contradictions += 1
            
            if contradictions > 0:
                assumption_consistency = max(0.3, 1.0 - contradictions * 0.3)
        
        # Check prediction consistency
        prediction_consistency = 1.0
        if parent_thought.causal_predictions and child_thought.causal_predictions:
            common_vars = set(parent_thought.causal_predictions.keys()) & set(child_thought.causal_predictions.keys())
            
            if common_vars:
                consistency_scores = []
                for var in common_vars:
                    p_pred = parent_thought.causal_predictions[var]
                    c_pred = child_thought.causal_predictions[var]
                    
                    # Calculate consistency (1 - normalized difference)
                    max_val = max(abs(p_pred), abs(c_pred), 1.0)
                    difference = abs(p_pred - c_pred) / max_val
                    consistency = 1.0 - difference
                    consistency_scores.append(consistency)
                
                prediction_consistency = np.mean(consistency_scores)
        
        return (assumption_consistency + prediction_consistency) / 2.0
    
    async def _construct_integrated_path(self, solution_thought: Thought, 
                                       context: Dict[str, Any]) -> ReasoningPath:
        """Construct comprehensive reasoning path with all integrations"""
        
        # Trace path from solution to root
        path_thoughts = []
        current_thought = solution_thought
        
        while current_thought:
            path_thoughts.append(current_thought)
            parent_id = current_thought.parent_id
            current_thought = self.thought_tree.get(parent_id) if parent_id else None
        
        path_thoughts.reverse()  # Root to solution order
        
        # Calculate path metrics
        total_quality = sum(t.quality_score for t in path_thoughts) / len(path_thoughts)
        confidence = solution_thought.confidence if solution_thought.confidence > 0 else solution_thought.quality_score
        solution_quality = solution_thought.quality_score
        
        # Memory coherence
        memory_coherence = 0.0
        if self.memory_integrator:
            memory_coherence = await self.memory_integrator.calculate_memory_coherence(
                path_thoughts, self.current_session_id
            )
        
        # Token efficiency
        total_tokens = sum(t.token_count for t in path_thoughts)
        token_efficiency = min(1.0, self.token_budget / max(total_tokens, 1))
        
        # Causal chain construction
        causal_chain = await self._construct_causal_chain(path_thoughts)
        
        # Generate reasoning trace
        reasoning_trace = await self._generate_comprehensive_trace(path_thoughts)
        
        reasoning_path = ReasoningPath(
            thoughts=path_thoughts,
            total_quality=total_quality,
            confidence=confidence,
            reasoning_trace=reasoning_trace,
            solution_quality=solution_quality,
            causal_chain=causal_chain,
            memory_coherence=memory_coherence,
            token_efficiency=token_efficiency
        )
        
        return reasoning_path
    
    async def _construct_causal_chain(self, thoughts: List[Thought]) -> List[Dict[str, Any]]:
        """Construct causal reasoning chain from thoughts"""
        
        causal_chain = []
        
        for i, thought in enumerate(thoughts):
            chain_link = {
                'step': i + 1,
                'thought_id': thought.id,
                'reasoning_strategy': thought.reasoning_strategy.value,
                'causal_assumptions': thought.causal_assumptions,
                'causal_predictions': thought.causal_predictions,
                'causal_evidence': thought.causal_evidence,
                'quality_score': thought.quality_score,
                'coherence_score': thought.coherence_score
            }
            
            # Add causal relationship to previous step
            if i > 0:
                previous_thought = thoughts[i - 1]
                chain_link['causal_relationship'] = await self._identify_causal_relationship(
                    previous_thought, thought
                )
            
            causal_chain.append(chain_link)
        
        return causal_chain
    
    async def _identify_causal_relationship(self, thought1: Thought, thought2: Thought) -> Dict[str, Any]:
        """Identify causal relationship between consecutive thoughts"""
        
        relationship = {
            'type': 'logical_progression',
            'strength': await self._calculate_causal_coherence(thought2, thought1),
            'mechanism': 'reasoning_development'
        }
        
        # Identify specific relationship type
        if thought2.reasoning_strategy == ReasoningStrategy.CAUSAL:
            relationship['type'] = 'causal_inference'
            relationship['mechanism'] = 'cause_effect_reasoning'
        elif thought2.reasoning_strategy == ReasoningStrategy.COUNTERFACTUAL:
            relationship['type'] = 'counterfactual_analysis'
            relationship['mechanism'] = 'alternative_scenario_evaluation'
        elif 'therefore' in thought2.content.lower():
            relationship['type'] = 'logical_conclusion'
            relationship['mechanism'] = 'deductive_reasoning'
        
        return relationship
    
    async def _generate_comprehensive_trace(self, thoughts: List[Thought]) -> str:
        """Generate comprehensive reasoning trace"""
        
        trace_parts = []
        
        for i, thought in enumerate(thoughts):
            depth_indent = "  " * thought.depth
            
            # Add thought with metadata
            trace_parts.append(f"{depth_indent}Step {i+1} ({thought.reasoning_strategy.value}):")
            trace_parts.append(f"{depth_indent}  {thought.content}")
            trace_parts.append(f"{depth_indent}  Quality: {thought.quality_score:.3f}, Coherence: {thought.coherence_score:.3f}")
            
            if thought.causal_assumptions:
                trace_parts.append(f"{depth_indent}  Assumptions: {', '.join(thought.causal_assumptions[:2])}")
            
            if thought.causal_predictions:
                trace_parts.append(f"{depth_indent}  Predictions: {list(thought.causal_predictions.keys())[:2]}")
            
            trace_parts.append("")  # Empty line for readability
        
        return "\n".join(trace_parts)
    
    async def _update_performance_metrics(self, reasoning_path: ReasoningPath) -> None:
        """Update system performance metrics"""
        
        self.performance_metrics.update({
            'solutions_found': len(self.solution_candidates),
            'average_depth': np.mean([t.depth for t in reasoning_path.thoughts]),
            'total_tokens_used': sum(t.token_count for t in self.thought_tree.values()),
            'search_efficiency': reasoning_path.solution_quality / max(1, len(self.thought_tree))
        })
        
        # Update global metrics
        global_metrics.gauge("enhanced_tot.solution_quality", reasoning_path.solution_quality)
        global_metrics.gauge("enhanced_tot.memory_coherence", reasoning_path.memory_coherence)
        global_metrics.gauge("enhanced_tot.token_efficiency", reasoning_path.token_efficiency)
        global_metrics.gauge("enhanced_tot.search_efficiency", self.performance_metrics['search_efficiency'])
    
    async def generate_solution_report(self, reasoning_path: ReasoningPath) -> str:
        """Generate comprehensive solution report with all integrations"""
        
        if not reasoning_path:
            return "No solution found."
        
        path_summary = reasoning_path.get_path_summary()
        
        report_parts = [
            "=" * 80,
            "ENHANCED TREE OF THOUGHTS - INTEGRATED SOLUTION REPORT",
            "=" * 80,
            "",
            "SOLUTION QUALITY METRICS:",
            "-" * 40,
            f"Overall Solution Quality: {reasoning_path.solution_quality:.3f}",
            f"Path Confidence: {reasoning_path.confidence:.3f}",
            f"Memory Coherence: {reasoning_path.memory_coherence:.3f}",
            f"Token Efficiency: {reasoning_path.token_efficiency:.3f}",
            f"Reasoning Depth: {path_summary['reasoning_depth']}",
            f"Breakthrough Thoughts: {path_summary['breakthrough_thoughts']}",
            "",
            "CAUSAL REASONING ANALYSIS:",
            "-" * 40,
            f"Causal Chain Length: {len(reasoning_path.causal_chain)}",
            f"Causal Coherence: {np.mean([link['coherence_score'] for link in reasoning_path.causal_chain]) if reasoning_path.causal_chain else 0.0:.3f}",
            "",
            "REASONING TRACE:",
            "-" * 40,
            reasoning_path.reasoning_trace,
            "",
            "CAUSAL CHAIN SUMMARY:",
            "-" * 40
        ]
        
        # Add causal chain details
        for i, link in enumerate(reasoning_path.causal_chain[:5]):  # Show first 5 steps
            report_parts.append(f"Step {i+1}: {link['reasoning_strategy']} (Quality: {link['quality_score']:.3f})")
            if link.get('causal_assumptions'):
                report_parts.append(f"  Assumptions: {', '.join(link['causal_assumptions'][:2])}")
        
        if len(reasoning_path.causal_chain) > 5:
            report_parts.append(f"... and {len(reasoning_path.causal_chain) - 5} more steps")
        
        report_parts.extend([
            "",
            "PERFORMANCE METRICS:",
            "-" * 40,
            f"Thoughts Generated: {self.performance_metrics['thoughts_generated']}",
            f"Thoughts Evaluated: {self.performance_metrics['thoughts_evaluated']}",
            f"Thoughts Pruned: {self.performance_metrics['thoughts_pruned']}",
            f"Search Efficiency: {self.performance_metrics['search_efficiency']:.3f}",
            "",
            "=" * 80
        ])
        
        return "\n".join(report_parts)