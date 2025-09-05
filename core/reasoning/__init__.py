"""
Reasoning Domain
Provides intelligent reasoning capabilities including causal inference, 
working memory, tree of thoughts, and temporal reasoning
"""

from .services import ReasoningOrchestrationService, ReasoningStrategy
from .events import (
    ReasoningEngineRegistered,
    ReasoningCompleted,
    ReasoningFailed,
    CausalRelationshipDiscovered,
    PatternRecognized,
    ReasoningChainCompleted,
    PredictionMade
)

__all__ = [
    # Services
    "ReasoningOrchestrationService",
    "ReasoningStrategy",
    
    # Events
    "ReasoningEngineRegistered",
    "ReasoningCompleted",
    "ReasoningFailed",
    "CausalRelationshipDiscovered",
    "PatternRecognized",
    "ReasoningChainCompleted",
    "PredictionMade"
]