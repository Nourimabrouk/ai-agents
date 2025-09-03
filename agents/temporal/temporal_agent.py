"""
Temporal Intelligence Agent
Uses multi-horizon temporal reasoning for optimization
"""

import asyncio
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import logging

from templates.base_agent import BaseAgent, Action, Observation
from .temporal_engine import TemporalReasoningEngine, TemporalEvent, TimeHorizon
from utils.observability.logging import get_logger

logger = get_logger(__name__)


class TemporalAgent(BaseAgent):
    """
    Agent specialized in temporal reasoning and multi-horizon optimization
    Optimizes decisions across microsecond to month-scale simultaneously
    """
    
    def __init__(self, name: str = "temporal_agent", **kwargs):
        super().__init__(name, **kwargs)
        
        # Initialize temporal reasoning engine
        self.temporal_engine = TemporalReasoningEngine()
        
        # Specialized temporal capabilities
        self.active_optimizations: Dict[str, Any] = {}
        self.temporal_objectives: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized temporal agent: {name}")
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute temporal reasoning tasks"""
        task_str = str(task).lower()
        
        if "predict" in task_str:
            return await self._handle_prediction_task(task, action)
        elif "optimize" in task_str:
            return await self._handle_optimization_task(task, action)
        elif "analyze_patterns" in task_str:
            return await self._handle_pattern_analysis(task, action)
        elif "add_event" in task_str:
            return await self._handle_event_addition(task, action)
        else:
            return await self._handle_general_temporal_task(task, action)
    
    async def _handle_prediction_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle prediction requests"""
        # Extract parameters from task
        task_params = self._extract_task_parameters(task)
        
        horizon = TimeHorizon(task_params.get('horizon', 'hour'))
        prediction_window = timedelta(
            seconds=task_params.get('window_seconds', 3600)
        )
        confidence_threshold = task_params.get('confidence_threshold', 0.7)
        
        # Generate predictions
        predictions = await self.temporal_engine.predict_events(
            horizon=horizon,
            prediction_window=prediction_window,
            confidence_threshold=confidence_threshold
        )
        
        return {
            'predictions': [
                {
                    'timestamp': p.timestamp.isoformat(),
                    'event_type': p.event_type,
                    'confidence': p.confidence,
                    'horizon': p.horizon.value,
                    'data': p.data
                }
                for p in predictions
            ],
            'prediction_count': len(predictions),
            'horizon': horizon.value,
            'window_seconds': prediction_window.total_seconds()
        }
    
    async def _handle_optimization_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle multi-horizon optimization requests"""
        task_params = self._extract_task_parameters(task)
        
        objective = task_params.get('objective', 'maximize_efficiency')
        constraints = task_params.get('constraints', {})
        
        # Run multi-horizon optimization
        optimization_results = await self.temporal_engine.optimize_across_horizons(
            objective=objective,
            constraints=constraints
        )
        
        # Store active optimization
        optimization_id = f"opt_{datetime.now().isoformat()}"
        self.active_optimizations[optimization_id] = {
            'objective': objective,
            'results': optimization_results,
            'started_at': datetime.now()
        }
        
        return {
            'optimization_id': optimization_id,
            'objective': objective,
            'horizons_optimized': list(optimization_results.keys()),
            'results_by_horizon': {
                horizon.value: result
                for horizon, result in optimization_results.items()
            },
            'coordination_status': 'completed'
        }
    
    async def _handle_pattern_analysis(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle temporal pattern analysis requests"""
        # Get current patterns from temporal engine
        patterns = self.temporal_engine.patterns
        
        # Analyze pattern characteristics
        pattern_analysis = {
            'total_patterns': len(patterns),
            'pattern_types': {},
            'most_active_patterns': [],
            'pattern_confidence_stats': {}
        }
        
        # Group patterns by type
        for pattern in patterns.values():
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_analysis['pattern_types']:
                pattern_analysis['pattern_types'][pattern_type] = 0
            pattern_analysis['pattern_types'][pattern_type] += 1
        
        # Find most recently active patterns
        recent_patterns = sorted(
            patterns.values(),
            key=lambda p: p.last_observed,
            reverse=True
        )[:5]
        
        pattern_analysis['most_active_patterns'] = [
            {
                'pattern_id': p.pattern_id,
                'pattern_type': p.pattern_type,
                'confidence': p.confidence,
                'last_observed': p.last_observed.isoformat(),
                'time_window_seconds': p.time_window.total_seconds()
            }
            for p in recent_patterns
        ]
        
        # Calculate confidence statistics
        confidences = [p.confidence for p in patterns.values()]
        if confidences:
            pattern_analysis['pattern_confidence_stats'] = {
                'average': sum(confidences) / len(confidences),
                'min': min(confidences),
                'max': max(confidences),
                'count': len(confidences)
            }
        
        return pattern_analysis
    
    async def _handle_event_addition(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle adding new temporal events"""
        task_params = self._extract_task_parameters(task)
        
        # Create temporal event from parameters
        event = TemporalEvent(
            timestamp=datetime.now(),
            event_type=task_params.get('event_type', 'generic_event'),
            data=task_params.get('event_data', {}),
            confidence=task_params.get('confidence', 0.8),
            horizon=TimeHorizon(task_params.get('horizon', 'minute'))
        )
        
        # Add to temporal engine
        await self.temporal_engine.add_event(event)
        
        return {
            'event_added': True,
            'event_type': event.event_type,
            'horizon': event.horizon.value,
            'timestamp': event.timestamp.isoformat(),
            'confidence': event.confidence
        }
    
    async def _handle_general_temporal_task(self, task: Any, action: Action) -> Dict[str, Any]:
        """Handle general temporal reasoning tasks"""
        # Get current temporal state
        temporal_state = self.temporal_engine.get_temporal_state()
        
        # Provide comprehensive temporal analysis
        analysis = {
            'temporal_state': temporal_state,
            'active_optimizations': len(self.active_optimizations),
            'temporal_objectives': len(self.temporal_objectives),
            'capabilities': [
                'multi_horizon_prediction',
                'cross_horizon_optimization',
                'pattern_detection',
                'temporal_coordination',
                'real_time_processing',
                'strategic_planning'
            ],
            'recommendation': self._generate_temporal_recommendation(temporal_state)
        }
        
        return analysis
    
    def _extract_task_parameters(self, task: Any) -> Dict[str, Any]:
        """Extract parameters from task description"""
        params = {}
        
        if isinstance(task, dict):
            params = task
        elif isinstance(task, str):
            # Simple parameter extraction from string
            if 'horizon=' in task:
                horizon_part = task.split('horizon=')[1].split()[0]
                params['horizon'] = horizon_part.strip(',')
            
            if 'objective=' in task:
                obj_part = task.split('objective=')[1].split()[0]
                params['objective'] = obj_part.strip(',')
            
            if 'confidence=' in task:
                conf_part = task.split('confidence=')[1].split()[0]
                try:
                    params['confidence_threshold'] = float(conf_part.strip(','))
                except ValueError:
                    params['confidence_threshold'] = 0.7
        
        return params
    
    def _generate_temporal_recommendation(self, temporal_state: Dict[str, Any]) -> str:
        """Generate recommendation based on temporal state"""
        total_events = temporal_state.get('total_events', 0)
        active_patterns = temporal_state.get('active_patterns', 0)
        
        if total_events < 10:
            return "Collect more temporal data to improve prediction accuracy"
        elif active_patterns == 0:
            return "No patterns detected yet - consider running pattern analysis"
        elif active_patterns > 20:
            return "High pattern activity detected - consider pattern consolidation"
        else:
            return "Temporal system operating normally - ready for optimization tasks"
    
    async def add_temporal_objective(self, 
                                   objective: str, 
                                   horizon: TimeHorizon,
                                   priority: int = 1,
                                   constraints: Dict[str, Any] = None) -> str:
        """Add a temporal objective for ongoing optimization"""
        objective_id = f"obj_{len(self.temporal_objectives)}_{datetime.now().isoformat()}"
        
        temporal_objective = {
            'id': objective_id,
            'objective': objective,
            'horizon': horizon,
            'priority': priority,
            'constraints': constraints or {},
            'created_at': datetime.now(),
            'status': 'active'
        }
        
        self.temporal_objectives.append(temporal_objective)
        
        logger.info(f"Added temporal objective: {objective} for horizon {horizon.value}")
        
        return objective_id
    
    async def process_temporal_batch(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of temporal events efficiently"""
        results = []
        
        # Process events in parallel
        tasks = []
        for event_data in events:
            event = TemporalEvent(
                timestamp=datetime.fromisoformat(event_data.get('timestamp', datetime.now().isoformat())),
                event_type=event_data.get('event_type', 'batch_event'),
                data=event_data.get('data', {}),
                confidence=event_data.get('confidence', 0.8),
                horizon=TimeHorizon(event_data.get('horizon', 'minute'))
            )
            tasks.append(self.temporal_engine.add_event(event))
        
        # Execute all additions in parallel
        await asyncio.gather(*tasks)
        
        # Generate batch analysis
        return {
            'events_processed': len(events),
            'processing_time': datetime.now().isoformat(),
            'batch_analysis': await self._analyze_batch_patterns(events)
        }
    
    async def _analyze_batch_patterns(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in a batch of events"""
        event_types = [e.get('event_type', 'unknown') for e in events]
        horizons = [e.get('horizon', 'minute') for e in events]
        
        return {
            'unique_event_types': len(set(event_types)),
            'horizons_represented': list(set(horizons)),
            'most_common_event_type': max(set(event_types), key=event_types.count) if event_types else None,
            'most_common_horizon': max(set(horizons), key=horizons.count) if horizons else None
        }
    
    async def get_temporal_insights(self) -> Dict[str, Any]:
        """Get comprehensive temporal insights"""
        temporal_state = self.temporal_engine.get_temporal_state()
        
        # Generate insights across all time horizons
        insights = {
            'system_health': self._assess_temporal_health(temporal_state),
            'prediction_readiness': self._assess_prediction_readiness(temporal_state),
            'optimization_opportunities': await self._identify_optimization_opportunities(),
            'pattern_insights': self._analyze_pattern_insights(),
            'resource_efficiency': self._calculate_resource_efficiency()
        }
        
        return insights
    
    def _assess_temporal_health(self, temporal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall temporal system health"""
        total_events = temporal_state.get('total_events', 0)
        active_patterns = temporal_state.get('active_patterns', 0)
        cache_size = temporal_state.get('cache_size', 0)
        
        health_score = min(1.0, (total_events / 1000) * 0.5 + (active_patterns / 10) * 0.3 + (cache_size / 100) * 0.2)
        
        return {
            'health_score': health_score,
            'status': 'healthy' if health_score > 0.7 else 'needs_attention' if health_score > 0.3 else 'critical',
            'total_events': total_events,
            'active_patterns': active_patterns,
            'cache_efficiency': cache_size
        }
    
    def _assess_prediction_readiness(self, temporal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Assess readiness for making predictions"""
        events_by_horizon = temporal_state.get('events_by_horizon', {})
        
        readiness = {}
        for horizon, event_count in events_by_horizon.items():
            if event_count >= 10:
                readiness[horizon] = 'ready'
            elif event_count >= 3:
                readiness[horizon] = 'limited'
            else:
                readiness[horizon] = 'insufficient_data'
        
        return {
            'overall_readiness': 'ready' if all(r == 'ready' for r in readiness.values()) else 'partial',
            'horizon_readiness': readiness,
            'recommendation': 'Collect more data for better predictions' if 'insufficient_data' in readiness.values() else 'System ready for predictions'
        }
    
    async def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify opportunities for optimization"""
        opportunities = []
        
        # Check for underutilized time horizons
        temporal_state = self.temporal_engine.get_temporal_state()
        events_by_horizon = temporal_state.get('events_by_horizon', {})
        
        for horizon, count in events_by_horizon.items():
            if count > 100:  # High activity horizon
                opportunities.append({
                    'type': 'high_activity_optimization',
                    'horizon': horizon,
                    'description': f'High activity in {horizon} horizon - optimize for throughput',
                    'priority': 'high'
                })
            elif count < 5:  # Low activity horizon
                opportunities.append({
                    'type': 'low_activity_investigation',
                    'horizon': horizon,
                    'description': f'Low activity in {horizon} horizon - investigate coverage',
                    'priority': 'medium'
                })
        
        return opportunities
    
    def _analyze_pattern_insights(self) -> Dict[str, Any]:
        """Analyze insights from detected patterns"""
        patterns = self.temporal_engine.patterns
        
        if not patterns:
            return {'insight': 'No patterns detected yet'}
        
        pattern_types = {}
        for pattern in patterns.values():
            pattern_type = pattern.pattern_type
            if pattern_type not in pattern_types:
                pattern_types[pattern_type] = []
            pattern_types[pattern_type].append(pattern.confidence)
        
        insights = {}
        for pattern_type, confidences in pattern_types.items():
            insights[pattern_type] = {
                'count': len(confidences),
                'avg_confidence': sum(confidences) / len(confidences),
                'max_confidence': max(confidences)
            }
        
        return insights
    
    def _calculate_resource_efficiency(self) -> Dict[str, Any]:
        """Calculate resource efficiency across time horizons"""
        # Simplified resource efficiency calculation
        active_opts = len(self.active_optimizations)
        temporal_objs = len(self.temporal_objectives)
        
        # Calculate efficiency based on active optimizations vs objectives
        efficiency = (active_opts / max(temporal_objs, 1)) if temporal_objs > 0 else 1.0
        
        return {
            'efficiency_score': min(1.0, efficiency),
            'active_optimizations': active_opts,
            'temporal_objectives': temporal_objs,
            'recommendation': 'Increase optimization coverage' if efficiency < 0.5 else 'Good optimization coverage'
        }