"""
Curriculum Learning System for RL Training
Progressive difficulty adjustment based on agent performance
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from collections import deque
import json
import os

from utils.observability.logging import get_logger

logger = get_logger(__name__)

class DifficultyLevel(Enum):
    """Curriculum difficulty levels"""
    BEGINNER = "beginner"
    EASY = "easy" 
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"
    MASTER = "master"

@dataclass
class CurriculumStage:
    """Single stage in curriculum"""
    name: str
    difficulty: DifficultyLevel
    parameters: Dict[str, Any]
    success_criteria: Dict[str, float]
    min_episodes: int = 100
    max_episodes: int = 1000
    advancement_threshold: float = 0.8
    regression_threshold: float = 0.4
    stability_window: int = 50
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'difficulty': self.difficulty.value,
            'parameters': self.parameters,
            'success_criteria': self.success_criteria,
            'min_episodes': self.min_episodes,
            'max_episodes': self.max_episodes,
            'advancement_threshold': self.advancement_threshold,
            'regression_threshold': self.regression_threshold,
            'stability_window': self.stability_window
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CurriculumStage':
        """Create from dictionary"""
        data['difficulty'] = DifficultyLevel(data['difficulty'])
        return cls(**data)

class CurriculumScheduler:
    """Base curriculum scheduler"""
    
    def should_advance(self, performance_history: List[float], current_stage: CurriculumStage) -> bool:
        """Determine if should advance to next stage"""
        if len(performance_history) < current_stage.min_episodes:
            return False
        
        # Check stability window
        recent_performance = performance_history[-current_stage.stability_window:]
        if len(recent_performance) < current_stage.stability_window:
            recent_performance = performance_history
        
        avg_performance = np.mean(recent_performance)
        return avg_performance >= current_stage.advancement_threshold
    
    def should_regress(self, performance_history: List[float], current_stage: CurriculumStage) -> bool:
        """Determine if should regress to previous stage"""
        if len(performance_history) < current_stage.stability_window:
            return False
        
        recent_performance = performance_history[-current_stage.stability_window:]
        avg_performance = np.mean(recent_performance)
        return avg_performance < current_stage.regression_threshold

class AdaptiveCurriculumScheduler(CurriculumScheduler):
    """Adaptive curriculum scheduler that adjusts thresholds"""
    
    def __init__(self, adaptation_rate: float = 0.1):
        self.adaptation_rate = adaptation_rate
        self.stage_attempt_counts = {}
        self.stage_success_rates = {}
    
    def should_advance(self, performance_history: List[float], current_stage: CurriculumStage) -> bool:
        """Adaptive advancement decision"""
        stage_name = current_stage.name
        
        # Track attempts
        if stage_name not in self.stage_attempt_counts:
            self.stage_attempt_counts[stage_name] = 0
            self.stage_success_rates[stage_name] = []
        
        self.stage_attempt_counts[stage_name] += 1
        
        # Basic advancement check
        should_advance = super().should_advance(performance_history, current_stage)
        
        if should_advance:
            self.stage_success_rates[stage_name].append(1.0)
        else:
            self.stage_success_rates[stage_name].append(0.0)
        
        # Adapt threshold if stage is too difficult
        if self.stage_attempt_counts[stage_name] > current_stage.max_episodes:
            success_rate = np.mean(self.stage_success_rates[stage_name][-50:])
            if success_rate < 0.1:  # Very low success rate
                # Lower the threshold
                current_stage.advancement_threshold = max(0.5, 
                    current_stage.advancement_threshold - self.adaptation_rate)
                logger.info(f"Adapted {stage_name} threshold to {current_stage.advancement_threshold:.2f}")
        
        return should_advance

class CurriculumManager:
    """
    Manages curriculum learning for RL training
    Automatically adjusts environment difficulty based on agent performance
    """
    
    def __init__(
        self,
        stages: List[CurriculumStage],
        scheduler: Optional[CurriculumScheduler] = None,
        save_path: Optional[str] = None
    ):
        self.stages = stages
        self.scheduler = scheduler or CurriculumScheduler()
        self.save_path = save_path
        
        # Current state
        self.current_stage_index = 0
        self.episodes_in_stage = 0
        self.performance_history = deque(maxlen=1000)
        self.stage_history = []
        
        # Statistics
        self.total_episodes = 0
        self.total_advancements = 0
        self.total_regressions = 0
        self.stage_completion_times = {}
        
        logger.info(f"Initialized curriculum with {len(stages)} stages")
        self._log_current_stage()
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage"""
        return self.stages[self.current_stage_index]
    
    @property
    def is_final_stage(self) -> bool:
        """Check if on final stage"""
        return self.current_stage_index == len(self.stages) - 1
    
    def update(self, performance_score: float, episode_info: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update curriculum based on performance
        
        Args:
            performance_score: Performance metric (higher is better)
            episode_info: Additional episode information
            
        Returns:
            True if stage changed, False otherwise
        """
        self.performance_history.append(performance_score)
        self.episodes_in_stage += 1
        self.total_episodes += 1
        
        stage_changed = False
        current_stage = self.current_stage
        
        # Check advancement
        if not self.is_final_stage:
            if (self.episodes_in_stage >= current_stage.min_episodes and 
                self.scheduler.should_advance(list(self.performance_history), current_stage)):
                
                self._advance_stage()
                stage_changed = True
                
        # Check regression (not from first stage)
        elif (self.current_stage_index > 0 and 
              self.episodes_in_stage >= current_stage.stability_window and
              self.scheduler.should_regress(list(self.performance_history), current_stage)):
            
            self._regress_stage()
            stage_changed = True
        
        # Check if stuck in stage too long
        if self.episodes_in_stage >= current_stage.max_episodes and not self.is_final_stage:
            logger.warning(f"Max episodes reached in stage {current_stage.name}, forcing advancement")
            self._advance_stage()
            stage_changed = True
        
        # Log stage change
        if stage_changed:
            self._log_current_stage()
            if self.save_path:
                self.save_state()
        
        return stage_changed
    
    def _advance_stage(self):
        """Advance to next curriculum stage"""
        old_stage = self.current_stage.name
        
        # Record completion time
        self.stage_completion_times[old_stage] = self.episodes_in_stage
        
        # Record stage transition
        self.stage_history.append({
            'from_stage': self.current_stage_index,
            'to_stage': self.current_stage_index + 1,
            'type': 'advancement',
            'episodes_in_stage': self.episodes_in_stage,
            'performance': np.mean(list(self.performance_history)[-self.current_stage.stability_window:]),
            'timestamp': datetime.now().isoformat()
        })
        
        self.current_stage_index += 1
        self.episodes_in_stage = 0
        self.total_advancements += 1
        
        logger.info(f"Advanced from {old_stage} to {self.current_stage.name}")
        logger.info(f"Completed {old_stage} in {self.stage_completion_times[old_stage]} episodes")
    
    def _regress_stage(self):
        """Regress to previous curriculum stage"""
        old_stage = self.current_stage.name
        
        # Record stage transition
        self.stage_history.append({
            'from_stage': self.current_stage_index,
            'to_stage': self.current_stage_index - 1,
            'type': 'regression',
            'episodes_in_stage': self.episodes_in_stage,
            'performance': np.mean(list(self.performance_history)[-self.current_stage.stability_window:]),
            'timestamp': datetime.now().isoformat()
        })
        
        self.current_stage_index -= 1
        self.episodes_in_stage = 0
        self.total_regressions += 1
        
        logger.info(f"Regressed from {old_stage} to {self.current_stage.name}")
    
    def _log_current_stage(self):
        """Log current stage information"""
        stage = self.current_stage
        logger.info(f"Current curriculum stage: {stage.name} ({stage.difficulty.value})")
        logger.info(f"Stage parameters: {stage.parameters}")
        logger.info(f"Episodes in stage: {self.episodes_in_stage}")
        
        if self.performance_history:
            recent_perf = list(self.performance_history)[-10:]
            logger.info(f"Recent performance: {np.mean(recent_perf):.3f} (±{np.std(recent_perf):.3f})")
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """Get current stage parameters for environment configuration"""
        return self.current_stage.parameters.copy()
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get detailed progress information"""
        return {
            'current_stage': {
                'index': self.current_stage_index,
                'name': self.current_stage.name,
                'difficulty': self.current_stage.difficulty.value,
                'parameters': self.current_stage.parameters,
                'episodes_in_stage': self.episodes_in_stage,
                'progress': min(1.0, self.episodes_in_stage / self.current_stage.min_episodes)
            },
            'overall_progress': {
                'total_episodes': self.total_episodes,
                'total_advancements': self.total_advancements,
                'total_regressions': self.total_regressions,
                'overall_completion': (self.current_stage_index + 1) / len(self.stages)
            },
            'performance': {
                'recent_performance': np.mean(list(self.performance_history)[-50:]) if self.performance_history else 0.0,
                'performance_trend': self._calculate_performance_trend(),
                'stage_performance': np.mean(list(self.performance_history)[-self.episodes_in_stage:]) if self.episodes_in_stage > 0 else 0.0
            },
            'next_milestone': self._get_next_milestone()
        }
    
    def _calculate_performance_trend(self) -> float:
        """Calculate performance trend (slope of recent performance)"""
        if len(self.performance_history) < 20:
            return 0.0
        
        recent_performance = list(self.performance_history)[-50:]
        x = np.arange(len(recent_performance))
        
        try:
            slope, _ = np.polyfit(x, recent_performance, 1)
            return float(slope)
        except:
            return 0.0
    
    def _get_next_milestone(self) -> Dict[str, Any]:
        """Get next milestone information"""
        stage = self.current_stage
        
        if self.is_final_stage:
            return {
                'type': 'mastery',
                'description': 'Continue mastering final stage',
                'episodes_remaining': max(0, stage.min_episodes - self.episodes_in_stage)
            }
        
        advancement_possible = self.episodes_in_stage >= stage.min_episodes
        if advancement_possible:
            recent_perf = np.mean(list(self.performance_history)[-stage.stability_window:])
            performance_gap = stage.advancement_threshold - recent_perf
            
            return {
                'type': 'advancement',
                'description': f'Advance to {self.stages[self.current_stage_index + 1].name}',
                'performance_needed': max(0.0, performance_gap),
                'performance_gap': performance_gap
            }
        else:
            return {
                'type': 'stabilization',
                'description': 'Complete minimum episodes in current stage',
                'episodes_remaining': stage.min_episodes - self.episodes_in_stage
            }
    
    def save_state(self, filepath: Optional[str] = None):
        """Save curriculum state to file"""
        filepath = filepath or self.save_path or "curriculum_state.json"
        
        state = {
            'current_stage_index': self.current_stage_index,
            'episodes_in_stage': self.episodes_in_stage,
            'performance_history': list(self.performance_history),
            'stage_history': self.stage_history,
            'total_episodes': self.total_episodes,
            'total_advancements': self.total_advancements,
            'total_regressions': self.total_regressions,
            'stage_completion_times': self.stage_completion_times,
            'stages': [stage.to_dict() for stage in self.stages],
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved curriculum state to {filepath}")
    
    def load_state(self, filepath: str):
        """Load curriculum state from file"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        self.current_stage_index = state['current_stage_index']
        self.episodes_in_stage = state['episodes_in_stage']
        self.performance_history = deque(state['performance_history'], maxlen=1000)
        self.stage_history = state['stage_history']
        self.total_episodes = state['total_episodes']
        self.total_advancements = state['total_advancements']
        self.total_regressions = state['total_regressions']
        self.stage_completion_times = state['stage_completion_times']
        
        # Recreate stages if provided
        if 'stages' in state:
            self.stages = [CurriculumStage.from_dict(stage_data) for stage_data in state['stages']]
        
        logger.info(f"Loaded curriculum state from {filepath}")
        logger.info(f"Resumed at stage {self.current_stage.name} (episode {self.episodes_in_stage})")
    
    def reset(self):
        """Reset curriculum to beginning"""
        logger.info("Resetting curriculum to beginning")
        
        self.current_stage_index = 0
        self.episodes_in_stage = 0
        self.performance_history.clear()
        self.stage_history.clear()
        
        self.total_episodes = 0
        self.total_advancements = 0
        self.total_regressions = 0
        self.stage_completion_times.clear()
        
        self._log_current_stage()
    
    def set_stage(self, stage_index: int):
        """Manually set curriculum stage"""
        if 0 <= stage_index < len(self.stages):
            old_index = self.current_stage_index
            self.current_stage_index = stage_index
            self.episodes_in_stage = 0
            
            logger.info(f"Manually set stage from {old_index} to {stage_index}")
            self._log_current_stage()
        else:
            raise ValueError(f"Invalid stage index: {stage_index}")
    
    def get_completion_report(self) -> Dict[str, Any]:
        """Generate curriculum completion report"""
        return {
            'curriculum_summary': {
                'total_stages': len(self.stages),
                'completed_stages': self.current_stage_index,
                'completion_percentage': (self.current_stage_index / len(self.stages)) * 100,
                'total_episodes': self.total_episodes,
                'total_advancements': self.total_advancements,
                'total_regressions': self.total_regressions
            },
            'stage_performance': {
                stage.name: {
                    'episodes_to_complete': self.stage_completion_times.get(stage.name, None),
                    'difficulty': stage.difficulty.value,
                    'parameters': stage.parameters
                }
                for stage in self.stages[:self.current_stage_index + 1]
            },
            'learning_trajectory': [
                {
                    'stage_name': self.stages[transition['to_stage']].name,
                    'transition_type': transition['type'],
                    'episodes': transition['episodes_in_stage'],
                    'performance': transition['performance'],
                    'timestamp': transition['timestamp']
                }
                for transition in self.stage_history
            ],
            'final_performance': {
                'current_stage_performance': np.mean(list(self.performance_history)[-self.episodes_in_stage:]) if self.episodes_in_stage > 0 else 0.0,
                'overall_performance_trend': self._calculate_performance_trend(),
                'peak_performance': max(self.performance_history) if self.performance_history else 0.0
            }
        }

def create_trading_curriculum() -> List[CurriculumStage]:
    """Create curriculum for trading environment"""
    return [
        CurriculumStage(
            name="basic_trading",
            difficulty=DifficultyLevel.BEGINNER,
            parameters={
                'num_symbols': 2,
                'volatility': 0.1,
                'market_regime': 'sideways',
                'transaction_cost': 0.0001,
                'max_position_size': 0.1
            },
            success_criteria={'profit': 0.05, 'sharpe_ratio': 0.5},
            min_episodes=100,
            advancement_threshold=0.7
        ),
        CurriculumStage(
            name="multi_asset_trading",
            difficulty=DifficultyLevel.EASY,
            parameters={
                'num_symbols': 5,
                'volatility': 0.15,
                'market_regime': 'mixed',
                'transaction_cost': 0.001,
                'max_position_size': 0.3
            },
            success_criteria={'profit': 0.08, 'sharpe_ratio': 0.8},
            min_episodes=150,
            advancement_threshold=0.75
        ),
        CurriculumStage(
            name="volatile_markets",
            difficulty=DifficultyLevel.MEDIUM,
            parameters={
                'num_symbols': 8,
                'volatility': 0.25,
                'market_regime': 'volatile',
                'transaction_cost': 0.001,
                'max_position_size': 0.5
            },
            success_criteria={'profit': 0.10, 'sharpe_ratio': 1.0},
            min_episodes=200,
            advancement_threshold=0.8
        ),
        CurriculumStage(
            name="crisis_trading",
            difficulty=DifficultyLevel.HARD,
            parameters={
                'num_symbols': 10,
                'volatility': 0.4,
                'market_regime': 'crisis',
                'transaction_cost': 0.002,
                'max_position_size': 1.0
            },
            success_criteria={'profit': 0.12, 'sharpe_ratio': 1.2},
            min_episodes=300,
            advancement_threshold=0.8
        ),
        CurriculumStage(
            name="master_trader",
            difficulty=DifficultyLevel.MASTER,
            parameters={
                'num_symbols': 15,
                'volatility': 0.3,
                'market_regime': 'mixed',
                'transaction_cost': 0.002,
                'max_position_size': 1.0,
                'regime_switching': True
            },
            success_criteria={'profit': 0.15, 'sharpe_ratio': 1.5},
            min_episodes=500,
            advancement_threshold=0.85
        )
    ]

def create_supply_chain_curriculum() -> List[CurriculumStage]:
    """Create curriculum for supply chain environment"""
    return [
        CurriculumStage(
            name="simple_chain",
            difficulty=DifficultyLevel.BEGINNER,
            parameters={
                'num_suppliers': 2,
                'num_products': 3,
                'demand_variability': 0.1,
                'disruption_probability': 0.0,
                'lead_time_variability': 0.1
            },
            success_criteria={'service_level': 0.95, 'cost_efficiency': 0.8},
            min_episodes=100,
            advancement_threshold=0.75
        ),
        CurriculumStage(
            name="medium_complexity",
            difficulty=DifficultyLevel.MEDIUM,
            parameters={
                'num_suppliers': 4,
                'num_products': 6,
                'demand_variability': 0.2,
                'disruption_probability': 0.1,
                'lead_time_variability': 0.2
            },
            success_criteria={'service_level': 0.90, 'cost_efficiency': 0.75},
            min_episodes=200,
            advancement_threshold=0.8
        ),
        CurriculumStage(
            name="complex_network",
            difficulty=DifficultyLevel.HARD,
            parameters={
                'num_suppliers': 8,
                'num_products': 10,
                'demand_variability': 0.3,
                'disruption_probability': 0.2,
                'lead_time_variability': 0.3
            },
            success_criteria={'service_level': 0.85, 'cost_efficiency': 0.7},
            min_episodes=300,
            advancement_threshold=0.8
        )
    ]

def create_resource_allocation_curriculum() -> List[CurriculentStage]:
    """Create curriculum for resource allocation environment"""
    return [
        CurriculumStage(
            name="basic_allocation",
            difficulty=DifficultyLevel.BEGINNER,
            parameters={
                'num_resources': 3,
                'num_agents': 4,
                'resource_capacity': 1000,
                'demand_variability': 0.1,
                'fairness_weight': 0.5
            },
            success_criteria={'fairness_index': 0.8, 'efficiency': 0.85},
            min_episodes=100,
            advancement_threshold=0.75
        ),
        CurriculumStage(
            name="competitive_allocation",
            difficulty=DifficultyLevel.MEDIUM,
            parameters={
                'num_resources': 6,
                'num_agents': 8,
                'resource_capacity': 800,
                'demand_variability': 0.25,
                'fairness_weight': 0.3
            },
            success_criteria={'fairness_index': 0.7, 'efficiency': 0.8},
            min_episodes=200,
            advancement_threshold=0.8
        ),
        CurriculumStage(
            name="dynamic_allocation",
            difficulty=DifficultyLevel.HARD,
            parameters={
                'num_resources': 8,
                'num_agents': 12,
                'resource_capacity': 600,
                'demand_variability': 0.4,
                'fairness_weight': 0.2,
                'dynamic_pricing': True
            },
            success_criteria={'fairness_index': 0.6, 'efficiency': 0.75},
            min_episodes=300,
            advancement_threshold=0.8
        )
    ]