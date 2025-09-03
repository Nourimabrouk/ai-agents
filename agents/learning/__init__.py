"""
Meta-Learning Pipeline for Self-Improving AI Agents
Develops strategies that improve through experience and cross-domain knowledge transfer
"""

from .meta_learning_agent import MetaLearningAgent
from .strategy_optimizer import StrategyOptimizer
from .pattern_recognizer import PatternRecognizer
from .knowledge_transfer import KnowledgeTransfer

__all__ = [
    'MetaLearningAgent',
    'StrategyOptimizer',
    'PatternRecognizer',
    'KnowledgeTransfer'
]