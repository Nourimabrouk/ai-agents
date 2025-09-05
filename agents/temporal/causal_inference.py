"""
Causal Inference Engine for Temporal Reasoning
Identifies causal relationships in temporal data
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from utils.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CausalRelationship:
    """Represents a causal relationship between variables"""
    cause_variable: str
    effect_variable: str
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    time_delay: timedelta
    relationship_type: str  # "linear", "non_linear", "threshold", etc.
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class CausalGraph:
    """Directed acyclic graph of causal relationships"""
    nodes: Set[str] = field(default_factory=set)
    relationships: List[CausalRelationship] = field(default_factory=list)
    
    def add_node(self, node: str) -> None:
        """Add a node to the graph"""
        self.nodes.add(node)
    
    def add_relationship(self, relationship: CausalRelationship) -> None:
        """Add a causal relationship"""
        self.nodes.add(relationship.cause_variable)
        self.nodes.add(relationship.effect_variable)
        self.relationships.append(relationship)
    
    def get_causes(self, effect: str) -> List[CausalRelationship]:
        """Get all causes for a given effect"""
        return [rel for rel in self.relationships if rel.effect_variable == effect]
    
    def get_effects(self, cause: str) -> List[CausalRelationship]:
        """Get all effects for a given cause"""
        return [rel for rel in self.relationships if rel.cause_variable == cause]


class CausalInferenceEngine:
    """
    Identifies causal relationships in temporal data
    Uses correlation and time-lag analysis for causal discovery
    """
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
        self.data_store: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.causal_graph = CausalGraph()
        self.analysis_cache: Dict[str, Any] = {}
        
        logger.info("Initialized causal inference engine")
    
    async def add_observation(self, variable: str, timestamp: datetime, value: float) -> None:
        """Add an observation for causal analysis"""
        self.data_store[variable].append((timestamp, value))
        
        # Keep only recent data (last 1000 points per variable)
        if len(self.data_store[variable]) > 1000:
            self.data_store[variable] = self.data_store[variable][-1000:]
    
    async def discover_causal_relationships(self, max_time_delay: timedelta = timedelta(hours=1)) -> CausalGraph:
        """Discover causal relationships between variables"""
        try:
            variables = list(self.data_store.keys())
            
            if len(variables) < 2:
                return self.causal_graph
            
            # Test all pairs of variables for causal relationships
            for cause_var in variables:
                for effect_var in variables:
                    if cause_var == effect_var:
                        continue
                    
                    relationship = await self._test_causal_relationship(
                        cause_var, effect_var, max_time_delay
                    )
                    
                    if relationship:
                        self.causal_graph.add_relationship(relationship)
            
            logger.info(f"Discovered {len(self.causal_graph.relationships)} causal relationships")
            return self.causal_graph
            
        except Exception as e:
            logger.error(f"Error discovering causal relationships: {e}")
            return self.causal_graph
    
    async def _test_causal_relationship(self, cause_var: str, effect_var: str, 
                                       max_delay: timedelta) -> Optional[CausalRelationship]:
        """Test for causal relationship between two variables"""
        try:
            cause_data = self.data_store[cause_var]
            effect_data = self.data_store[effect_var]
            
            if len(cause_data) < 10 or len(effect_data) < 10:
                return {}
            
            # Test different time delays
            best_correlation = 0.0
            best_delay = timedelta(0)
            best_strength = 0.0
            
            # Test delays from 0 to max_delay in 5-minute increments
            delay_increment = timedelta(minutes=5)
            current_delay = timedelta(0)
            
            while current_delay <= max_delay:
                correlation, strength = await self._calculate_delayed_correlation(
                    cause_data, effect_data, current_delay
                )
                
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_delay = current_delay
                    best_strength = abs(correlation)
                
                current_delay += delay_increment
            
            # Check if relationship is significant
            if best_strength > 0.3:  # Minimum correlation threshold
                confidence = min(1.0, best_strength * 2.0)  # Simple confidence mapping
                
                relationship_type = "linear"
                if best_correlation < 0:
                    relationship_type = "negative_linear"
                
                return CausalRelationship(
                    cause_variable=cause_var,
                    effect_variable=effect_var,
                    strength=best_strength,
                    confidence=confidence,
                    time_delay=best_delay,
                    relationship_type=relationship_type
                )
            
            return {}
            
        except Exception as e:
            logger.error(f"Error testing causal relationship: {e}")
            return {}
    
    async def _calculate_delayed_correlation(self, cause_data: List[Tuple[datetime, float]], 
                                           effect_data: List[Tuple[datetime, float]], 
                                           delay: timedelta) -> Tuple[float, float]:
        """Calculate correlation with time delay"""
        try:
            # Align data with delay
            cause_values = []
            effect_values = []
            
            for cause_time, cause_val in cause_data:
                # Find corresponding effect value at cause_time + delay
                target_time = cause_time + delay
                
                # Find closest effect measurement
                closest_effect = None
                min_time_diff = timedelta.max
                
                for effect_time, effect_val in effect_data:
                    time_diff = abs(effect_time - target_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_effect = effect_val
                
                # Only include if we found a reasonably close match (within 10 minutes)
                if min_time_diff < timedelta(minutes=10) and closest_effect is not None:
                    cause_values.append(cause_val)
                    effect_values.append(closest_effect)
            
            if len(cause_values) < 3:
                return 0.0, 0.0
            
            # Calculate Pearson correlation
            cause_array = np.array(cause_values)
            effect_array = np.array(effect_values)
            
            correlation = np.corrcoef(cause_array, effect_array)[0, 1]
            
            if np.isnan(correlation):
                return 0.0, 0.0
            
            strength = abs(correlation)
            
            return correlation, strength
            
        except Exception as e:
            logger.error(f"Error calculating delayed correlation: {e}")
            return 0.0, 0.0
    
    async def predict_effect(self, cause_var: str, cause_value: float, 
                            effect_var: str) -> Optional[Tuple[float, float]]:
        """Predict effect value given cause value"""
        try:
            # Find causal relationship
            relationships = [
                rel for rel in self.causal_graph.relationships 
                if rel.cause_variable == cause_var and rel.effect_variable == effect_var
            ]
            
            if not relationships:
                return {}
            
            # Use the strongest relationship
            relationship = max(relationships, key=lambda r: r.strength)
            
            # Get recent data for baseline prediction
            cause_data = self.data_store[cause_var]
            effect_data = self.data_store[effect_var]
            
            if not cause_data or not effect_data:
                return {}
            
            # Simple linear prediction based on recent data
            recent_cause = [val for _, val in cause_data[-20:]]
            recent_effect = [val for _, val in effect_data[-20:]]
            
            if len(recent_cause) < 2 or len(recent_effect) < 2:
                return {}
            
            cause_mean = np.mean(recent_cause)
            effect_mean = np.mean(recent_effect)
            
            # Simple linear scaling
            cause_deviation = cause_value - cause_mean
            predicted_effect = effect_mean + (cause_deviation * relationship.strength)
            
            # Estimate confidence based on relationship strength
            confidence = relationship.confidence
            
            return predicted_effect, confidence
            
        except Exception as e:
            logger.error(f"Error predicting effect: {e}")
            return {}
    
    async def get_intervention_recommendations(self, target_var: str, 
                                             target_value: float) -> List[Dict[str, Any]]:
        """Get recommendations for interventions to achieve target value"""
        try:
            recommendations = []
            
            # Find causes of the target variable
            causes = self.causal_graph.get_causes(target_var)
            
            for cause_rel in causes:
                cause_var = cause_rel.cause_variable
                
                # Get current average value of cause
                if cause_var not in self.data_store:
                    continue
                
                recent_cause_values = [val for _, val in self.data_store[cause_var][-10:]]
                if not recent_cause_values:
                    continue
                
                current_cause_avg = np.mean(recent_cause_values)
                
                # Calculate required change in cause to achieve target
                # This is a simplified linear model
                recent_effect_values = [val for _, val in self.data_store[target_var][-10:]]
                if not recent_effect_values:
                    continue
                
                current_effect_avg = np.mean(recent_effect_values)
                effect_change_needed = target_value - current_effect_avg
                
                # Scale by relationship strength
                cause_change_needed = effect_change_needed / max(cause_rel.strength, 0.1)
                recommended_cause_value = current_cause_avg + cause_change_needed
                
                recommendations.append({
                    'intervention_variable': cause_var,
                    'current_value': current_cause_avg,
                    'recommended_value': recommended_cause_value,
                    'change_magnitude': abs(cause_change_needed),
                    'confidence': cause_rel.confidence,
                    'time_delay': cause_rel.time_delay.total_seconds(),
                    'relationship_strength': cause_rel.strength
                })
            
            # Sort by confidence and feasibility
            recommendations.sort(key=lambda x: x['confidence'], reverse=True)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating intervention recommendations: {e}")
            return []
    
    async def get_causal_summary(self) -> Dict[str, Any]:
        """Get summary of discovered causal relationships"""
        try:
            return {
                'total_variables': len(self.causal_graph.nodes),
                'total_relationships': len(self.causal_graph.relationships),
                'strong_relationships': len([r for r in self.causal_graph.relationships if r.strength > 0.7]),
                'moderate_relationships': len([r for r in self.causal_graph.relationships if 0.4 <= r.strength <= 0.7]),
                'weak_relationships': len([r for r in self.causal_graph.relationships if r.strength < 0.4]),
                'variables': list(self.causal_graph.nodes),
                'strongest_relationship': max(
                    self.causal_graph.relationships, 
                    key=lambda r: r.strength, 
                    default=None
                )
            }
        except Exception as e:
            logger.error(f"Error generating causal summary: {e}")
            return {}