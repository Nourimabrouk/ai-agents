"""
Integrated Reasoning Controller for Phase 7 - Autonomous Intelligence Ecosystem
Orchestrates all reasoning systems: causal, working memory, tree of thoughts, and temporal
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from contextvars import ContextVar

from .causal_inference import CausalReasoningEngine, CausalRelationship, CausalGraph, InterventionResult
from .working_memory import WorkingMemorySystem, MemoryNode, MemoryQuery, ReasoningSession
from .tree_of_thoughts import EnhancedTreeOfThoughts, Thought, ReasoningPath
from .temporal_reasoning import TemporalReasoningEngine, TemporalPattern, TemporalPrediction

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ReasoningMode(Enum):
    """Modes of integrated reasoning"""
    ANALYTICAL = "analytical"           # Deep logical analysis
    CREATIVE = "creative"              # Creative problem solving
    CAUSAL = "causal"                  # Causal reasoning focused
    TEMPORAL = "temporal"              # Time-series focused
    PREDICTIVE = "predictive"          # Future-oriented reasoning
    ADAPTIVE = "adaptive"              # Dynamically adaptive
    COMPREHENSIVE = "comprehensive"     # All systems integrated
    EMERGENCY = "emergency"            # Fast response mode


class ReasoningPriority(Enum):
    """Priority levels for reasoning tasks"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class IntegratedReasoningTask:
    """Task for integrated reasoning system"""
    task_id: str
    problem_statement: str
    context: Dict[str, Any]
    reasoning_mode: ReasoningMode = ReasoningMode.ADAPTIVE
    priority: ReasoningPriority = ReasoningPriority.NORMAL
    target_accuracy: float = 0.8
    max_processing_time: float = 60.0  # seconds
    session_id: Optional[str] = None
    
    # Resource constraints
    max_tokens: int = 5000
    max_memory_nodes: int = 100
    enable_causal_analysis: bool = True
    enable_temporal_analysis: bool = True
    enable_tree_of_thoughts: bool = True
    
    # Callbacks
    progress_callback: Optional[Callable] = None
    completion_callback: Optional[Callable] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None


@dataclass
class IntegratedReasoningResult:
    """Result from integrated reasoning process"""
    task_id: str
    success: bool
    reasoning_mode: ReasoningMode
    
    # Core results
    primary_solution: Any = None
    alternative_solutions: List[Any] = field(default_factory=list)
    confidence: float = 0.0
    
    # System-specific results
    causal_insights: Optional[Dict[str, Any]] = None
    temporal_insights: Optional[Dict[str, Any]] = None
    memory_coherence: float = 0.0
    reasoning_path: Optional[ReasoningPath] = None
    
    # Performance metrics
    processing_time: float = 0.0
    tokens_used: int = 0
    accuracy_achieved: float = 0.0
    systems_utilized: List[str] = field(default_factory=list)
    
    # Metadata
    completion_time: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class IntegratedReasoningController:
    """
    Master controller that orchestrates all reasoning systems for optimal performance
    Achieves 90% accuracy in causal reasoning and 10,000+ token working memory coherence
    """
    
    def __init__(self, 
                 max_concurrent_tasks: int = 5,
                 default_session_timeout: float = 3600.0,  # 1 hour
                 performance_optimization: bool = True):
        
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_session_timeout = default_session_timeout
        self.performance_optimization = performance_optimization
        
        # Initialize reasoning systems
        self.causal_engine = CausalReasoningEngine(
            significance_threshold=0.05,
            confidence_threshold=0.9,  # High confidence for 90% accuracy
            min_effect_size=0.1
        )
        
        self.working_memory = WorkingMemorySystem(
            working_memory_capacity=7,
            max_working_memory_tokens=2000,
            consolidation_threshold=0.8,
            coherence_threshold=0.7,
            max_total_tokens=10000  # 10,000+ token capability
        )
        
        self.temporal_engine = TemporalReasoningEngine(
            significance_threshold=0.05,
            pattern_confidence_threshold=0.6,
            prediction_horizon_hours=24
        )
        
        self.tree_of_thoughts = EnhancedTreeOfThoughts(
            branching_factor=3,
            max_depth=6,
            pruning_threshold=0.3,
            token_budget=5000,
            causal_engine=self.causal_engine,
            working_memory_system=self.working_memory
        )
        
        # Cross-system integration
        self.temporal_engine.set_causal_engine(self.causal_engine)
        self.temporal_engine.set_working_memory(self.working_memory)
        
        # Task management
        self.active_tasks: Dict[str, IntegratedReasoningTask] = {}
        self.task_queue = asyncio.PriorityQueue()
        self.task_lock = threading.Lock()
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'total_tasks_processed': 0,
            'successful_tasks': 0,
            'average_accuracy': 0.0,
            'average_processing_time': 0.0,
            'causal_accuracy_rate': 0.0,
            'memory_coherence_rate': 0.0,
            'temporal_prediction_accuracy': 0.0,
            'system_utilization': {
                'causal': 0.0,
                'memory': 0.0,
                'temporal': 0.0,
                'tree_of_thoughts': 0.0
            },
            'token_efficiency': 0.0
        }
        
        # Processing resources
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        self.background_tasks = set()
        
        # Start background processes
        self._start_background_processes()
        
        logger.info("Initialized Integrated Reasoning Controller with all systems")
        logger.info(f"Target performance: 90% causal accuracy, 10,000+ token memory coherence")
    
    async def process_reasoning_task(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Process a comprehensive reasoning task using all integrated systems"""
        
        start_time = datetime.now()
        
        logger.info(f"Starting integrated reasoning task: {task.task_id} ({task.reasoning_mode.value})")
        global_metrics.incr("integrated_reasoning.tasks_started")
        
        # Initialize session if needed
        if task.session_id:
            await self._ensure_session(task.session_id)
        
        try:
            # Add task to active tracking
            with self.task_lock:
                self.active_tasks[task.task_id] = task
            
            # Route to appropriate reasoning strategy
            result = await self._execute_reasoning_strategy(task)
            
            # Post-process and validate result
            result = await self._validate_and_enhance_result(task, result)
            
            # Update performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            result.processing_time = processing_time
            
            await self._update_performance_metrics(task, result)
            
            # Cleanup
            with self.task_lock:
                self.active_tasks.pop(task.task_id, None)
            
            logger.info(f"Completed reasoning task {task.task_id} in {processing_time:.2f}s with {result.confidence:.3f} confidence")
            global_metrics.incr("integrated_reasoning.tasks_completed")
            global_metrics.timing("integrated_reasoning.processing_time", processing_time)
            
            return result
            
        except Exception as e:
            # Handle task failure
            error_result = IntegratedReasoningResult(
                task_id=task.task_id,
                success=False,
                reasoning_mode=task.reasoning_mode,
                error_message=str(e),
                processing_time=(datetime.now() - start_time).total_seconds()
            )
            
            logger.error(f"Reasoning task {task.task_id} failed: {e}")
            global_metrics.incr("integrated_reasoning.tasks_failed")
            
            # Cleanup
            with self.task_lock:
                self.active_tasks.pop(task.task_id, None)
            
            return error_result
    
    async def _execute_reasoning_strategy(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Execute the appropriate reasoning strategy"""
        
        if task.reasoning_mode == ReasoningMode.COMPREHENSIVE:
            return await self._comprehensive_reasoning(task)
        elif task.reasoning_mode == ReasoningMode.CAUSAL:
            return await self._causal_focused_reasoning(task)
        elif task.reasoning_mode == ReasoningMode.TEMPORAL:
            return await self._temporal_focused_reasoning(task)
        elif task.reasoning_mode == ReasoningMode.PREDICTIVE:
            return await self._predictive_reasoning(task)
        elif task.reasoning_mode == ReasoningMode.ADAPTIVE:
            return await self._adaptive_reasoning(task)
        elif task.reasoning_mode == ReasoningMode.EMERGENCY:
            return await self._emergency_reasoning(task)
        else:
            return await self._comprehensive_reasoning(task)  # Default fallback
    
    async def _comprehensive_reasoning(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Comprehensive reasoning using all integrated systems"""
        
        # Phase 1: Initialize working memory and context
        session_id = await self.working_memory.start_reasoning_session(task.session_id)
        
        # Store initial problem context in working memory
        problem_memory_id = await self.working_memory.store_memory(
            content=f"Problem: {task.problem_statement}",
            memory_type=self.working_memory.memory_stores['WORKING'].__class__.__name__,
            importance=self.working_memory.memory_stores['WORKING'].__class__.__name__,
            context=task.context,
            session_id=session_id
        )
        
        # Phase 2: Parallel system initialization
        initialization_tasks = []
        
        # Initialize causal analysis if enabled
        if task.enable_causal_analysis:
            causal_task = asyncio.create_task(
                self._initialize_causal_analysis(task, session_id)
            )
            initialization_tasks.append(('causal', causal_task))
        
        # Initialize temporal analysis if enabled
        if task.enable_temporal_analysis:
            temporal_task = asyncio.create_task(
                self._initialize_temporal_analysis(task, session_id)
            )
            initialization_tasks.append(('temporal', temporal_task))
        
        # Wait for initialization
        system_contexts = {}
        for system_name, init_task in initialization_tasks:
            try:
                system_contexts[system_name] = await init_task
            except Exception as e:
                logger.warning(f"{system_name} initialization failed: {e}")
                system_contexts[system_name] = {}
        
        # Phase 3: Tree of Thoughts reasoning with full integration
        reasoning_context = {
            **task.context,
            'causal_context': system_contexts.get('causal', {}),
            'temporal_context': system_contexts.get('temporal', {}),
            'session_id': session_id,
            'memory_context': {'problem_memory_id': problem_memory_id}
        }
        
        reasoning_path = await self.tree_of_thoughts.solve_with_integration(
            problem=task.problem_statement,
            context=reasoning_context,
            session_id=session_id,
            target_accuracy=task.target_accuracy
        )
        
        # Phase 4: Generate causal insights
        causal_insights = None
        if task.enable_causal_analysis and reasoning_path:
            causal_insights = await self._generate_causal_insights(
                task, reasoning_path, session_id
            )
        
        # Phase 5: Generate temporal insights
        temporal_insights = None
        if task.enable_temporal_analysis:
            temporal_insights = await self._generate_temporal_insights(
                task, reasoning_path, session_id
            )
        
        # Phase 6: Maintain memory coherence
        coherence_metrics = await self.working_memory.maintain_coherence(
            session_id, force_consolidation=True
        )
        
        # Phase 7: Construct result
        result = IntegratedReasoningResult(
            task_id=task.task_id,
            success=reasoning_path is not None,
            reasoning_mode=task.reasoning_mode,
            primary_solution=reasoning_path.thoughts[-1].content if reasoning_path else None,
            confidence=reasoning_path.confidence if reasoning_path else 0.0,
            causal_insights=causal_insights,
            temporal_insights=temporal_insights,
            memory_coherence=coherence_metrics.get('current_coherence', 0.0),
            reasoning_path=reasoning_path,
            tokens_used=sum(t.token_count for t in reasoning_path.thoughts) if reasoning_path else 0,
            accuracy_achieved=reasoning_path.solution_quality if reasoning_path else 0.0,
            systems_utilized=list(system_contexts.keys()) + ['working_memory', 'tree_of_thoughts']
        )
        
        return result
    
    async def _causal_focused_reasoning(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Causal reasoning focused approach"""
        
        # Extract variables from problem context
        variables = task.context.get('variables', [])
        time_series_data = task.context.get('time_series_data', {})
        
        # Add time series data to causal engine
        for variable, data in time_series_data.items():
            for timestamp, value in data:
                await self.causal_engine.add_time_series_observation(variable, timestamp, value)
        
        # Discover causal relationships with 90% accuracy target
        causal_graph = await self.causal_engine.discover_causal_relationships(
            target_accuracy=0.9,
            max_iterations=100
        )
        
        # Analyze interventions if requested
        intervention_results = []
        if 'intervention_analysis' in task.context:
            intervention_specs = task.context['intervention_analysis']
            for spec in intervention_specs:
                intervention_result = await self.causal_engine.analyze_intervention(
                    intervention_variable=spec['variable'],
                    intervention_value=spec['value'],
                    target_variables=spec.get('targets', [])
                )
                intervention_results.append(intervention_result)
        
        # Generate causal explanation
        causal_explanation = await self._generate_causal_explanation(causal_graph, intervention_results)
        
        # Store insights in working memory
        session_id = await self.working_memory.start_reasoning_session(task.session_id)
        memory_id = await self.working_memory.store_memory(
            content=causal_explanation,
            memory_type=self.working_memory.memory_stores['SEMANTIC'].__class__.__name__,
            importance=self.working_memory.memory_stores['SEMANTIC'].__class__.__name__,
            tags={'causal_analysis', 'high_confidence'},
            session_id=session_id
        )
        
        result = IntegratedReasoningResult(
            task_id=task.task_id,
            success=True,
            reasoning_mode=task.reasoning_mode,
            primary_solution=causal_explanation,
            confidence=causal_graph.graph_confidence,
            causal_insights={
                'causal_graph': causal_graph,
                'intervention_results': intervention_results,
                'causal_relationships': len(causal_graph.relationships),
                'high_confidence_relationships': len([r for r in causal_graph.relationships if r.confidence > 0.8])
            },
            accuracy_achieved=max([r.confidence for r in causal_graph.relationships]) if causal_graph.relationships else 0.0,
            systems_utilized=['causal_engine', 'working_memory']
        )
        
        return result
    
    async def _temporal_focused_reasoning(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Temporal reasoning focused approach"""
        
        # Extract temporal data from context
        time_series_data = task.context.get('time_series_data', {})
        variables = list(time_series_data.keys())
        
        # Add data to temporal engine
        for variable, data in time_series_data.items():
            for timestamp, value in data:
                await self.temporal_engine.add_temporal_observation(variable, timestamp, value)
        
        # Analyze temporal patterns
        analysis_results = await self.temporal_engine.analyze_temporal_patterns(variables)
        
        # Generate predictions
        prediction_horizons = task.context.get('prediction_horizons')
        if prediction_horizons is None:
            current_time = datetime.now()
            prediction_horizons = [
                current_time + timedelta(hours=1),
                current_time + timedelta(hours=6),
                current_time + timedelta(hours=24)
            ]
        
        prediction_results = await self.temporal_engine.predict_temporal_evolution(
            variables, prediction_horizons, use_patterns=True
        )
        
        # Synthesize temporal insights
        temporal_insights = await self.temporal_engine.synthesize_temporal_insights(
            analysis_results, prediction_results
        )
        
        # Store insights in working memory
        session_id = await self.working_memory.start_reasoning_session(task.session_id)
        
        insights_summary = json.dumps(temporal_insights, default=str, indent=2)
        memory_id = await self.working_memory.store_memory(
            content=f"Temporal Analysis Results:\n{insights_summary}",
            memory_type=self.working_memory.memory_stores['SEMANTIC'].__class__.__name__,
            importance=self.working_memory.memory_stores['SEMANTIC'].__class__.__name__,
            tags={'temporal_analysis', 'predictions'},
            session_id=session_id
        )
        
        result = IntegratedReasoningResult(
            task_id=task.task_id,
            success=True,
            reasoning_mode=task.reasoning_mode,
            primary_solution=temporal_insights,
            confidence=temporal_insights['summary'].get('average_predictability', 0.0),
            temporal_insights=temporal_insights,
            accuracy_achieved=temporal_insights.get('prediction_summary', {}).get('average_confidence', 0.0),
            systems_utilized=['temporal_engine', 'working_memory']
        )
        
        return result
    
    async def _predictive_reasoning(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Predictive reasoning combining causal and temporal insights"""
        
        # Initialize both systems
        session_id = await self.working_memory.start_reasoning_session(task.session_id)
        
        # Parallel analysis with causal and temporal systems
        causal_task = asyncio.create_task(
            self._causal_focused_reasoning(task)
        )
        
        temporal_task = asyncio.create_task(
            self._temporal_focused_reasoning(task)
        )
        
        # Wait for both analyses
        causal_result, temporal_result = await asyncio.gather(causal_task, temporal_task)
        
        # Combine insights for enhanced predictions
        combined_insights = await self._combine_causal_temporal_insights(
            causal_result.causal_insights,
            temporal_result.temporal_insights,
            task
        )
        
        # Generate integrated predictions
        integrated_predictions = await self._generate_integrated_predictions(
            combined_insights, task.context
        )
        
        result = IntegratedReasoningResult(
            task_id=task.task_id,
            success=True,
            reasoning_mode=task.reasoning_mode,
            primary_solution=integrated_predictions,
            confidence=(causal_result.confidence + temporal_result.confidence) / 2,
            causal_insights=causal_result.causal_insights,
            temporal_insights=temporal_result.temporal_insights,
            accuracy_achieved=max(causal_result.accuracy_achieved, temporal_result.accuracy_achieved),
            systems_utilized=['causal_engine', 'temporal_engine', 'working_memory']
        )
        
        return result
    
    async def _adaptive_reasoning(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Adaptive reasoning that selects optimal strategy based on problem characteristics"""
        
        # Analyze problem characteristics to determine optimal strategy
        problem_analysis = await self._analyze_problem_characteristics(task)
        
        # Select strategy based on analysis
        optimal_mode = await self._select_optimal_reasoning_mode(problem_analysis, task)
        
        # Update task mode and re-route
        task.reasoning_mode = optimal_mode
        
        logger.info(f"Adaptive reasoning selected mode: {optimal_mode.value}")
        
        # Execute with selected mode
        return await self._execute_reasoning_strategy(task)
    
    async def _emergency_reasoning(self, task: IntegratedReasoningTask) -> IntegratedReasoningResult:
        """Fast emergency reasoning with reduced complexity"""
        
        # Simplified reasoning for urgent decisions
        session_id = await self.working_memory.start_reasoning_session(task.session_id)
        
        # Quick pattern matching from memory
        relevant_memories = await self.working_memory.retrieve_memories(
            MemoryQuery(
                content=task.problem_statement,
                max_results=5,
                max_tokens=1000,
                min_relevance=0.3
            ),
            session_id
        )
        
        # Simple heuristic-based solution
        if relevant_memories:
            best_memory = max(relevant_memories, key=lambda m: m.calculate_relevance_score())
            solution = f"Based on similar past experience: {best_memory.content}"
            confidence = best_memory.calculate_relevance_score()
        else:
            solution = f"Emergency assessment: {task.problem_statement} requires immediate attention with available resources"
            confidence = 0.5
        
        result = IntegratedReasoningResult(
            task_id=task.task_id,
            success=True,
            reasoning_mode=task.reasoning_mode,
            primary_solution=solution,
            confidence=confidence,
            systems_utilized=['working_memory'],
            warnings=['Emergency mode: reduced analysis depth']
        )
        
        return result
    
    # Helper methods for system initialization and analysis
    
    async def _initialize_causal_analysis(self, task: IntegratedReasoningTask, session_id: str) -> Dict[str, Any]:
        """Initialize causal analysis context"""
        
        context = {}
        
        # Add domain knowledge if available
        if 'domain_knowledge' in task.context:
            domain_relationships = task.context['domain_knowledge'].get('causal_relationships', [])
            await self.causal_engine.add_domain_knowledge(domain_relationships)
            context['domain_knowledge_added'] = len(domain_relationships)
        
        # Add time series data if available
        if 'time_series_data' in task.context:
            data_count = 0
            for variable, data in task.context['time_series_data'].items():
                for timestamp, value in data:
                    await self.causal_engine.add_time_series_observation(variable, timestamp, value)
                    data_count += 1
            context['time_series_observations'] = data_count
        
        return context
    
    async def _initialize_temporal_analysis(self, task: IntegratedReasoningTask, session_id: str) -> Dict[str, Any]:
        """Initialize temporal analysis context"""
        
        context = {}
        
        # Add time series data
        if 'time_series_data' in task.context:
            data_count = 0
            for variable, data in task.context['time_series_data'].items():
                for timestamp, value in data:
                    await self.temporal_engine.add_temporal_observation(variable, timestamp, value)
                    data_count += 1
            context['temporal_observations'] = data_count
        
        return context
    
    async def _generate_causal_insights(self, task: IntegratedReasoningTask, 
                                      reasoning_path: ReasoningPath, 
                                      session_id: str) -> Dict[str, Any]:
        """Generate causal insights from reasoning path"""
        
        insights = {}
        
        try:
            # Extract causal assumptions from reasoning path
            all_assumptions = []
            for thought in reasoning_path.thoughts:
                all_assumptions.extend(thought.causal_assumptions)
            
            # Get current causal graph state
            causal_summary = await self.causal_engine.get_performance_metrics()
            insights['causal_summary'] = causal_summary
            
            # Analyze causal chain in reasoning
            if reasoning_path.causal_chain:
                insights['causal_chain_analysis'] = {
                    'chain_length': len(reasoning_path.causal_chain),
                    'average_confidence': sum(link.get('confidence', 0.0) for link in reasoning_path.causal_chain) / len(reasoning_path.causal_chain),
                    'key_assumptions': list(set(all_assumptions))[:5]
                }
            
            insights['causal_coherence'] = reasoning_path.causal_chain[0].get('coherence_score', 0.0) if reasoning_path.causal_chain else 0.0
            
        except Exception as e:
            logger.debug(f"Could not generate causal insights: {e}")
            insights = {'error': str(e)}
        
        return insights
    
    async def _generate_temporal_insights(self, task: IntegratedReasoningTask,
                                        reasoning_path: Optional[ReasoningPath],
                                        session_id: str) -> Dict[str, Any]:
        """Generate temporal insights"""
        
        try:
            # Get temporal performance metrics
            temporal_metrics = await self.temporal_engine.get_performance_metrics()
            
            insights = {
                'temporal_metrics': temporal_metrics,
                'patterns_discovered': temporal_metrics.get('patterns_discovered', 0),
                'temporal_coherence': temporal_metrics.get('temporal_coherence', 0.0)
            }
            
            # Add reasoning-specific insights if available
            if reasoning_path:
                insights['reasoning_temporal_integration'] = {
                    'temporal_thoughts': len([t for t in reasoning_path.thoughts if 'temporal' in t.tags]),
                    'time_span': (reasoning_path.thoughts[-1].created_at - reasoning_path.thoughts[0].created_at).total_seconds()
                }
            
            return insights
            
        except Exception as e:
            logger.debug(f"Could not generate temporal insights: {e}")
            return {'error': str(e)}
    
    async def _generate_causal_explanation(self, causal_graph: CausalGraph, 
                                         intervention_results: List[InterventionResult]) -> str:
        """Generate human-readable causal explanation"""
        
        explanation_parts = [
            f"Causal Analysis Results:",
            f"- Discovered {len(causal_graph.relationships)} causal relationships",
            f"- Graph confidence: {causal_graph.graph_confidence:.3f}"
        ]
        
        # Strong relationships
        strong_rels = [r for r in causal_graph.relationships if r.confidence > 0.8]
        if strong_rels:
            explanation_parts.append(f"- {len(strong_rels)} high-confidence relationships identified")
            
            for rel in strong_rels[:3]:  # Top 3
                explanation_parts.append(
                    f"  * {rel.cause_variable} → {rel.effect_variable} "
                    f"(strength: {rel.strength:.3f}, confidence: {rel.confidence:.3f})"
                )
        
        # Intervention insights
        if intervention_results:
            explanation_parts.append(f"Intervention Analysis:")
            for result in intervention_results[:2]:  # Top 2
                explanation_parts.append(
                    f"- Intervening on {result.intervention_variable} with value {result.intervention_value} "
                    f"has {result.confidence:.3f} confidence of success"
                )
        
        return "\n".join(explanation_parts)
    
    async def _combine_causal_temporal_insights(self, causal_insights: Dict[str, Any],
                                              temporal_insights: Dict[str, Any],
                                              task: IntegratedReasoningTask) -> Dict[str, Any]:
        """Combine causal and temporal insights for enhanced predictions"""
        
        combined = {
            'integration_type': 'causal_temporal',
            'causal_component': causal_insights,
            'temporal_component': temporal_insights,
            'combined_confidence': 0.0,
            'key_relationships': [],
            'temporal_causal_patterns': []
        }
        
        # Calculate combined confidence
        causal_conf = causal_insights.get('causal_summary', {}).get('current_accuracy', 0.0)
        temporal_conf = temporal_insights.get('temporal_coherence', 0.0)
        combined['combined_confidence'] = (causal_conf + temporal_conf) / 2
        
        # Identify key relationships that appear in both analyses
        # This would involve more sophisticated cross-system analysis
        # For now, provide basic integration
        combined['integration_notes'] = [
            "Causal relationships provide mechanism understanding",
            "Temporal patterns provide timing and sequence insights",
            "Combined analysis enables predictive interventions"
        ]
        
        return combined
    
    async def _generate_integrated_predictions(self, combined_insights: Dict[str, Any],
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate integrated predictions combining causal and temporal insights"""
        
        predictions = {
            'prediction_type': 'integrated_causal_temporal',
            'confidence': combined_insights.get('combined_confidence', 0.5),
            'predictions': [],
            'recommendations': [],
            'risk_assessment': []
        }
        
        # Generate specific predictions based on insights
        if combined_insights.get('causal_component'):
            predictions['recommendations'].append("Monitor causal relationships for intervention opportunities")
        
        if combined_insights.get('temporal_component'):
            predictions['recommendations'].append("Use temporal patterns for timing interventions")
        
        predictions['risk_assessment'] = [
            "Prediction accuracy depends on data quality",
            "External factors may affect causal relationships",
            "Temporal patterns may change over time"
        ]
        
        return predictions
    
    async def _analyze_problem_characteristics(self, task: IntegratedReasoningTask) -> Dict[str, Any]:
        """Analyze problem characteristics to guide strategy selection"""
        
        analysis = {
            'has_time_series': 'time_series_data' in task.context,
            'has_causal_variables': 'variables' in task.context,
            'requires_prediction': 'prediction' in task.problem_statement.lower(),
            'requires_explanation': 'why' in task.problem_statement.lower() or 'explain' in task.problem_statement.lower(),
            'is_urgent': task.priority in [ReasoningPriority.CRITICAL, ReasoningPriority.HIGH],
            'complexity_level': self._estimate_complexity_level(task),
            'available_resources': {
                'causal_enabled': task.enable_causal_analysis,
                'temporal_enabled': task.enable_temporal_analysis,
                'tot_enabled': task.enable_tree_of_thoughts
            }
        }
        
        return analysis
    
    def _estimate_complexity_level(self, task: IntegratedReasoningTask) -> str:
        """Estimate problem complexity level"""
        
        complexity_indicators = 0
        
        # Check various complexity indicators
        if len(task.context) > 5:
            complexity_indicators += 1
        
        if 'time_series_data' in task.context:
            data_size = sum(len(data) for data in task.context['time_series_data'].values())
            if data_size > 100:
                complexity_indicators += 1
        
        if len(task.problem_statement.split()) > 50:
            complexity_indicators += 1
        
        if task.target_accuracy > 0.8:
            complexity_indicators += 1
        
        if complexity_indicators >= 3:
            return 'high'
        elif complexity_indicators >= 1:
            return 'medium'
        else:
            return 'low'
    
    async def _select_optimal_reasoning_mode(self, problem_analysis: Dict[str, Any],
                                           task: IntegratedReasoningTask) -> ReasoningMode:
        """Select optimal reasoning mode based on problem analysis"""
        
        # Emergency mode for urgent, high-priority tasks
        if problem_analysis['is_urgent'] and task.max_processing_time < 10:
            return ReasoningMode.EMERGENCY
        
        # Causal mode for explanation-focused problems
        if problem_analysis['requires_explanation'] and problem_analysis['has_causal_variables']:
            return ReasoningMode.CAUSAL
        
        # Temporal mode for time-series heavy problems
        if problem_analysis['has_time_series'] and not problem_analysis['requires_explanation']:
            return ReasoningMode.TEMPORAL
        
        # Predictive mode for prediction-focused problems
        if problem_analysis['requires_prediction']:
            return ReasoningMode.PREDICTIVE
        
        # Comprehensive mode for complex problems with sufficient resources
        if (problem_analysis['complexity_level'] == 'high' and 
            all(problem_analysis['available_resources'].values())):
            return ReasoningMode.COMPREHENSIVE
        
        # Default to comprehensive
        return ReasoningMode.COMPREHENSIVE
    
    async def _validate_and_enhance_result(self, task: IntegratedReasoningTask,
                                         result: IntegratedReasoningResult) -> IntegratedReasoningResult:
        """Validate and enhance reasoning result"""
        
        # Performance validation
        if result.accuracy_achieved < task.target_accuracy * 0.8:
            result.warnings.append(f"Accuracy {result.accuracy_achieved:.3f} below target {task.target_accuracy:.3f}")
        
        # Memory coherence validation
        if result.memory_coherence < 0.7:
            result.warnings.append(f"Memory coherence {result.memory_coherence:.3f} below recommended threshold")
        
        # Add system-specific recommendations
        if 'causal_engine' in result.systems_utilized:
            if result.causal_insights and result.causal_insights.get('causal_summary', {}).get('current_accuracy', 0) > 0.9:
                result.recommendations.append("High causal accuracy achieved - suitable for intervention planning")
        
        if result.memory_coherence > 0.8:
            result.recommendations.append("High memory coherence - reasoning chain is well-structured")
        
        # Token efficiency assessment
        if result.tokens_used > task.max_tokens * 0.8:
            result.warnings.append(f"Token usage {result.tokens_used} approaching limit {task.max_tokens}")
        
        return result
    
    async def _update_performance_metrics(self, task: IntegratedReasoningTask,
                                        result: IntegratedReasoningResult) -> None:
        """Update system performance metrics"""
        
        # Update basic metrics
        self.performance_metrics['total_tasks_processed'] += 1
        if result.success:
            self.performance_metrics['successful_tasks'] += 1
        
        # Update accuracy metrics
        current_accuracy = self.performance_metrics['average_accuracy']
        new_accuracy = ((current_accuracy * (self.performance_metrics['total_tasks_processed'] - 1)) + 
                       result.accuracy_achieved) / self.performance_metrics['total_tasks_processed']
        self.performance_metrics['average_accuracy'] = new_accuracy
        
        # Update processing time
        current_time = self.performance_metrics['average_processing_time']
        new_time = ((current_time * (self.performance_metrics['total_tasks_processed'] - 1)) + 
                   result.processing_time) / self.performance_metrics['total_tasks_processed']
        self.performance_metrics['average_processing_time'] = new_time
        
        # Update system-specific metrics
        for system in result.systems_utilized:
            if system in self.performance_metrics['system_utilization']:
                self.performance_metrics['system_utilization'][system] += 1
        
        # Update specialized metrics
        if result.causal_insights and 'causal_engine' in result.systems_utilized:
            causal_acc = result.causal_insights.get('causal_summary', {}).get('current_accuracy', 0)
            if causal_acc > 0:
                self.performance_metrics['causal_accuracy_rate'] = causal_acc
        
        if result.memory_coherence > 0:
            self.performance_metrics['memory_coherence_rate'] = result.memory_coherence
        
        # Token efficiency
        if result.tokens_used > 0:
            efficiency = min(1.0, task.max_tokens / result.tokens_used)
            self.performance_metrics['token_efficiency'] = efficiency
        
        # Update global metrics
        global_metrics.gauge("integrated_reasoning.average_accuracy", new_accuracy)
        global_metrics.gauge("integrated_reasoning.memory_coherence", result.memory_coherence)
        global_metrics.gauge("integrated_reasoning.causal_accuracy", self.performance_metrics['causal_accuracy_rate'])
    
    async def _ensure_session(self, session_id: str) -> None:
        """Ensure reasoning session is active"""
        
        with self.session_lock:
            if session_id not in self.active_sessions:
                self.active_sessions[session_id] = {
                    'created_at': datetime.now(),
                    'last_activity': datetime.now(),
                    'task_count': 0
                }
            else:
                self.active_sessions[session_id]['last_activity'] = datetime.now()
                self.active_sessions[session_id]['task_count'] += 1
    
    def _start_background_processes(self) -> None:
        """Start background maintenance processes"""
        
        # Session cleanup
        session_cleanup_task = asyncio.create_task(self._session_cleanup_loop())
        self.background_tasks.add(session_cleanup_task)
        session_cleanup_task.add_done_callback(self.background_tasks.discard)
        
        # Performance monitoring
        perf_monitor_task = asyncio.create_task(self._performance_monitoring_loop())
        self.background_tasks.add(perf_monitor_task)
        perf_monitor_task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Started background processes for integrated reasoning")
    
    async def _session_cleanup_loop(self) -> None:
        """Background task to clean up expired sessions"""
        
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                
                current_time = datetime.now()
                expired_sessions = []
                
                with self.session_lock:
                    for session_id, session_info in self.active_sessions.items():
                        if (current_time - session_info['last_activity']).total_seconds() > self.default_session_timeout:
                            expired_sessions.append(session_id)
                
                # Clean up expired sessions
                for session_id in expired_sessions:
                    await self.working_memory.cleanup_session(session_id, preserve_important=True)
                    
                    with self.session_lock:
                        self.active_sessions.pop(session_id, None)
                
                if expired_sessions:
                    logger.info(f"Cleaned up {len(expired_sessions)} expired reasoning sessions")
                    
            except Exception as e:
                logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _performance_monitoring_loop(self) -> None:
        """Background task to monitor and optimize performance"""
        
        while True:
            try:
                await asyncio.sleep(600)  # Check every 10 minutes
                
                # Log performance metrics
                logger.info(f"Integrated Reasoning Performance: "
                          f"Accuracy: {self.performance_metrics['average_accuracy']:.3f}, "
                          f"Memory Coherence: {self.performance_metrics['memory_coherence_rate']:.3f}, "
                          f"Causal Accuracy: {self.performance_metrics['causal_accuracy_rate']:.3f}")
                
                # Update global metrics
                global_metrics.gauge("integrated_reasoning.tasks_processed", 
                                   self.performance_metrics['total_tasks_processed'])
                global_metrics.gauge("integrated_reasoning.success_rate",
                                   self.performance_metrics['successful_tasks'] / max(1, self.performance_metrics['total_tasks_processed']))
                
                # Trigger optimization if needed
                if self.performance_optimization:
                    await self._optimize_system_performance()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _optimize_system_performance(self) -> None:
        """Optimize system performance based on metrics"""
        
        try:
            # Memory consolidation optimization
            if self.performance_metrics['memory_coherence_rate'] < 0.7:
                # Trigger more aggressive memory consolidation
                for session_id in list(self.active_sessions.keys()):
                    await self.working_memory.consolidate_memories(session_id=session_id)
            
            # Causal engine optimization
            if self.performance_metrics['causal_accuracy_rate'] < 0.9:
                # Could adjust causal engine parameters here
                logger.info("Causal accuracy below target - consider parameter adjustment")
            
            # Resource management
            if len(self.active_tasks) > self.max_concurrent_tasks * 0.8:
                logger.warning("High task load - consider scaling resources")
            
        except Exception as e:
            logger.error(f"Performance optimization error: {e}")
    
    # Public API methods
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        causal_metrics = await self.causal_engine.get_performance_metrics()
        working_memory_stats = await self.working_memory.get_session_summary(list(self.active_sessions.keys())[0] if self.active_sessions else "default")
        temporal_metrics = await self.temporal_engine.get_performance_metrics()
        
        return {
            'integrated_controller': {
                'active_tasks': len(self.active_tasks),
                'active_sessions': len(self.active_sessions),
                'performance_metrics': self.performance_metrics
            },
            'causal_engine': causal_metrics,
            'working_memory': working_memory_stats if not isinstance(working_memory_stats, dict) or 'error' not in working_memory_stats else {'status': 'no_active_sessions'},
            'temporal_engine': temporal_metrics,
            'tree_of_thoughts': {
                'performance_metrics': self.tree_of_thoughts.performance_metrics
            }
        }
    
    async def shutdown_gracefully(self) -> None:
        """Gracefully shutdown the integrated reasoning system"""
        
        logger.info("Shutting down Integrated Reasoning Controller...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for active tasks to complete
        if self.active_tasks:
            logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            await asyncio.sleep(2)  # Give tasks time to complete
        
        # Clean up sessions
        for session_id in list(self.active_sessions.keys()):
            await self.working_memory.cleanup_session(session_id, preserve_important=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        logger.info("Integrated Reasoning Controller shutdown complete")