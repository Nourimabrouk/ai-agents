"""
Comprehensive Reinforcement Learning Framework for AI Agents
Integrates advanced RL algorithms with existing agent architecture
"""

from .environments import TradingEnvironment, SupplyChainEnvironment, ResourceAllocationEnvironment
from .algorithms import PPOAgent, SACAgent, MADDPGAgent, QMIXAgent
from .curriculum import CurriculumManager, DifficultyLevel
from .training import RLTrainingPipeline, ExperimentManager
from .evaluation import RLEvaluationSuite, BenchmarkRunner
from .deployment import RLModelServer, PolicyABTesting
from .monitoring import RLMonitoringDashboard, PerformanceTracker

__version__ = "1.0.0"
__all__ = [
    # Environments
    "TradingEnvironment",
    "SupplyChainEnvironment", 
    "ResourceAllocationEnvironment",
    
    # Algorithms
    "PPOAgent",
    "SACAgent",
    "MADDPGAgent",
    "QMIXAgent",
    
    # Training Framework
    "CurriculumManager",
    "DifficultyLevel",
    "RLTrainingPipeline",
    "ExperimentManager",
    
    # Evaluation
    "RLEvaluationSuite",
    "BenchmarkRunner",
    
    # Production
    "RLModelServer",
    "PolicyABTesting",
    "RLMonitoringDashboard",
    "PerformanceTracker"
]