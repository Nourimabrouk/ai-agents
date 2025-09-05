"""
Reasoning Domain Events
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ...shared import DomainEvent, AgentId


@dataclass
class ReasoningEngineRegistered(DomainEvent):
    """Event when new reasoning engine is registered"""
    reasoning_mode: str = ""
    engine_type: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "reasoning.engine_registered"
        if not hasattr(self, 'event_id'):
            self.event_id = f"engine_registered_{self.reasoning_mode}_{self.timestamp.timestamp()}"


@dataclass
class ReasoningCompleted(DomainEvent):
    """Event when reasoning task is completed"""
    reasoning_mode: str = ""
    strategy: str = ""
    confidence: float = 0.0
    execution_time: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "reasoning.completed"
        if not hasattr(self, 'event_id'):
            self.event_id = f"reasoning_completed_{self.reasoning_mode}_{self.timestamp.timestamp()}"


@dataclass
class ReasoningFailed(DomainEvent):
    """Event when reasoning task fails"""
    reasoning_mode: str = ""
    strategy: str = ""
    error_message: str = ""
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "reasoning.failed"
        if not hasattr(self, 'event_id'):
            self.event_id = f"reasoning_failed_{self.reasoning_mode}_{self.timestamp.timestamp()}"


@dataclass
class CausalRelationshipDiscovered(DomainEvent):
    """Event when new causal relationship is discovered"""
    cause: str = ""
    effect: str = ""
    strength: float = 0.0
    confidence: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        self.event_type = "reasoning.causal_discovered"
        if not hasattr(self, 'event_id'):
            self.event_id = f"causal_discovered_{hash(self.cause + self.effect)}_{self.timestamp.timestamp()}"


@dataclass
class PatternRecognized(DomainEvent):
    """Event when reasoning pattern is recognized"""
    pattern_type: str = ""
    pattern_description: str = ""
    confidence: float = 0.0
    instances: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.instances is None:
            self.instances = []
        self.event_type = "reasoning.pattern_recognized"
        if not hasattr(self, 'event_id'):
            self.event_id = f"pattern_recognized_{self.pattern_type}_{self.timestamp.timestamp()}"


@dataclass
class ReasoningChainCompleted(DomainEvent):
    """Event when reasoning chain is completed"""
    chain_id: str = ""
    steps_count: int = 0
    final_confidence: float = 0.0
    reasoning_modes_used: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.reasoning_modes_used is None:
            self.reasoning_modes_used = []
        self.event_type = "reasoning.chain_completed"
        if not hasattr(self, 'event_id'):
            self.event_id = f"chain_completed_{self.chain_id}_{self.timestamp.timestamp()}"


@dataclass
class PredictionMade(DomainEvent):
    """Event when prediction is made"""
    prediction_type: str = ""
    prediction: str = ""
    confidence: float = 0.0
    time_horizon: str = ""
    factors_considered: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.factors_considered is None:
            self.factors_considered = []
        self.event_type = "reasoning.prediction_made"
        if not hasattr(self, 'event_id'):
            self.event_id = f"prediction_made_{self.prediction_type}_{self.timestamp.timestamp()}"


# Factory functions for creating reasoning events
def create_engine_registered_event(source: AgentId, reasoning_mode: str, 
                                 engine_type: str) -> ReasoningEngineRegistered:
    """Create reasoning engine registered event"""
    return ReasoningEngineRegistered(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        reasoning_mode=reasoning_mode,
        engine_type=engine_type
    )


def create_reasoning_completed_event(source: AgentId, reasoning_mode: str, strategy: str,
                                   confidence: float, execution_time: float) -> ReasoningCompleted:
    """Create reasoning completed event"""
    return ReasoningCompleted(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        reasoning_mode=reasoning_mode,
        strategy=strategy,
        confidence=confidence,
        execution_time=execution_time
    )


def create_causal_discovered_event(source: AgentId, cause: str, effect: str,
                                 strength: float, confidence: float) -> CausalRelationshipDiscovered:
    """Create causal relationship discovered event"""
    return CausalRelationshipDiscovered(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        cause=cause,
        effect=effect,
        strength=strength,
        confidence=confidence
    )


def create_pattern_recognized_event(source: AgentId, pattern_type: str, 
                                  pattern_description: str, confidence: float,
                                  instances: List[Dict[str, Any]]) -> PatternRecognized:
    """Create pattern recognized event"""
    return PatternRecognized(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        pattern_type=pattern_type,
        pattern_description=pattern_description,
        confidence=confidence,
        instances=instances
    )


def create_prediction_made_event(source: AgentId, prediction_type: str, prediction: str,
                               confidence: float, time_horizon: str, 
                               factors_considered: List[str]) -> PredictionMade:
    """Create prediction made event"""
    return PredictionMade(
        event_id="",  # Will be set in __post_init__
        event_type="",  # Will be set in __post_init__
        source=source,
        timestamp=datetime.utcnow(),
        data={},
        prediction_type=prediction_type,
        prediction=prediction,
        confidence=confidence,
        time_horizon=time_horizon,
        factors_considered=factors_considered
    )