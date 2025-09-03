"""
Base Multi-Agent RL Environment
Foundation for all MARL environments in the system
"""

import asyncio
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
import logging

from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger

logger = get_logger(__name__)

class AgentRole(Enum):
    """Roles agents can take in environments"""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    MONITOR = "monitor"

@dataclass
class AgentConfig:
    """Configuration for agents in RL environment"""
    agent_id: str
    role: AgentRole
    initial_balance: float = 100000.0
    risk_tolerance: float = 0.5
    learning_rate: float = 0.001
    exploration_rate: float = 0.1
    capabilities: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class EnvironmentState:
    """Complete environment state"""
    global_state: Dict[str, Any]
    agent_states: Dict[str, Dict[str, Any]]
    market_conditions: Dict[str, Any]
    timestamp: datetime
    step_count: int
    
    def to_observation_dict(self) -> Dict[str, np.ndarray]:
        """Convert state to observation format for RL algorithms"""
        observations = {}
        
        # Global features
        global_features = self._extract_global_features()
        
        # Agent-specific observations
        for agent_id, agent_state in self.agent_states.items():
            agent_features = self._extract_agent_features(agent_id, agent_state)
            
            # Combine global and agent features
            obs_vector = np.concatenate([global_features, agent_features])
            observations[agent_id] = obs_vector
            
        return observations
    
    def _extract_global_features(self) -> np.ndarray:
        """Extract global environment features"""
        features = []
        
        # Market conditions
        if 'volatility' in self.market_conditions:
            features.append(self.market_conditions['volatility'])
        else:
            features.append(0.1)  # Default volatility
            
        if 'trend' in self.market_conditions:
            features.append(self.market_conditions['trend'])
        else:
            features.append(0.0)  # Neutral trend
            
        # Time features
        features.append(self.step_count / 1000.0)  # Normalized step count
        
        # Number of active agents
        features.append(len(self.agent_states) / 10.0)  # Normalized agent count
        
        return np.array(features, dtype=np.float32)
    
    def _extract_agent_features(self, agent_id: str, agent_state: Dict[str, Any]) -> np.ndarray:
        """Extract agent-specific features"""
        features = []
        
        # Agent performance metrics
        features.append(agent_state.get('balance', 100000.0) / 100000.0)  # Normalized balance
        features.append(agent_state.get('success_rate', 0.5))
        features.append(agent_state.get('risk_level', 0.5))
        features.append(agent_state.get('utilization', 0.5))
        
        # Agent relationships (simplified)
        features.append(agent_state.get('collaboration_score', 0.5))
        features.append(len(agent_state.get('connections', [])) / 10.0)  # Normalized connections
        
        return np.array(features, dtype=np.float32)

class BaseMultiAgentEnvironment(gym.Env, ABC):
    """
    Base class for multi-agent RL environments
    Provides common functionality for MARL systems
    """
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        max_steps: int = 1000,
        reward_shaping: bool = True,
        curriculum_enabled: bool = False,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.agent_configs = {config.agent_id: config for config in agent_configs}
        self.num_agents = len(agent_configs)
        self.max_steps = max_steps
        self.reward_shaping = reward_shaping
        self.curriculum_enabled = curriculum_enabled
        
        # Environment state
        self.current_state: Optional[EnvironmentState] = None
        self.step_count = 0
        self.episode_count = 0
        self.done = False
        
        # Metrics tracking
        self.episode_rewards = {agent_id: 0.0 for agent_id in self.agent_configs.keys()}
        self.episode_actions = {agent_id: [] for agent_id in self.agent_configs.keys()}
        self.performance_history = []
        
        # Curriculum learning
        self.difficulty_level = 0.1 if curriculum_enabled else 1.0
        
        # Set seed
        if seed is not None:
            self.seed(seed)
        
        # Define action and observation spaces
        self._setup_spaces()
        
        logger.info(f"Initialized environment with {self.num_agents} agents")
    
    def _setup_spaces(self):
        """Setup action and observation spaces for all agents"""
        # Base observation space (to be extended by subclasses)
        obs_dim = self._get_observation_dimension()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Base action space (to be extended by subclasses)
        action_dim = self._get_action_dimension()
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        
        # Multi-agent spaces
        self.observation_spaces = {
            agent_id: self.observation_space
            for agent_id in self.agent_configs.keys()
        }
        
        self.action_spaces = {
            agent_id: self.action_space
            for agent_id in self.agent_configs.keys()
        }
    
    @abstractmethod
    def _get_observation_dimension(self) -> int:
        """Get the dimension of observation vector"""
        pass
    
    @abstractmethod
    def _get_action_dimension(self) -> int:
        """Get the dimension of action vector"""
        pass
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment for a new episode"""
        super().reset(seed=seed)
        
        self.step_count = 0
        self.episode_count += 1
        self.done = False
        
        # Reset episode tracking
        self.episode_rewards = {agent_id: 0.0 for agent_id in self.agent_configs.keys()}
        self.episode_actions = {agent_id: [] for agent_id in self.agent_configs.keys()}
        
        # Initialize environment state
        self.current_state = self._initialize_environment_state()
        
        # Get initial observations
        observations = self.current_state.to_observation_dict()
        
        # Info dictionary
        info = {
            "episode": self.episode_count,
            "difficulty": self.difficulty_level,
            "agents": list(self.agent_configs.keys())
        }
        
        logger.info(f"Reset environment for episode {self.episode_count}")
        
        return observations, info
    
    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[
        Dict[str, np.ndarray],  # observations
        Dict[str, float],       # rewards
        Dict[str, bool],        # terminated
        Dict[str, bool],        # truncated  
        Dict[str, Any]          # info
    ]:
        """Execute one step in the environment"""
        if self.done:
            raise RuntimeError("Environment is done. Call reset() to start new episode.")
        
        self.step_count += 1
        
        # Validate actions
        validated_actions = self._validate_actions(actions)
        
        # Store actions for analysis
        for agent_id, action in validated_actions.items():
            self.episode_actions[agent_id].append(action.copy())
        
        # Execute actions and update state
        self.current_state = self._execute_actions(validated_actions, self.current_state)
        
        # Calculate rewards
        rewards = self._calculate_rewards(validated_actions, self.current_state)
        
        # Track episode rewards
        for agent_id, reward in rewards.items():
            self.episode_rewards[agent_id] += reward
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self._check_truncation()
        
        # Get new observations
        observations = self.current_state.to_observation_dict()
        
        # Create info dictionary
        info = self._create_info_dict(validated_actions, rewards)
        
        # Check if episode is done
        if any(terminated.values()) or any(truncated.values()):
            self.done = True
            self._handle_episode_end()
        
        return observations, rewards, terminated, truncated, info
    
    @abstractmethod
    def _initialize_environment_state(self) -> EnvironmentState:
        """Initialize the environment state"""
        pass
    
    @abstractmethod
    def _execute_actions(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> EnvironmentState:
        """Execute agent actions and return new state"""
        pass
    
    @abstractmethod
    def _calculate_rewards(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> Dict[str, float]:
        """Calculate rewards for each agent"""
        pass
    
    def _validate_actions(self, actions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Validate and clip actions to valid ranges"""
        validated = {}
        
        for agent_id, action in actions.items():
            if agent_id not in self.agent_configs:
                logger.warning(f"Unknown agent {agent_id} provided action")
                continue
            
            # Clip action to valid range
            clipped_action = np.clip(action, -1.0, 1.0)
            validated[agent_id] = clipped_action
        
        # Add default actions for missing agents
        for agent_id in self.agent_configs.keys():
            if agent_id not in validated:
                validated[agent_id] = np.zeros(self._get_action_dimension(), dtype=np.float32)
        
        return validated
    
    def _check_termination(self) -> Dict[str, bool]:
        """Check termination conditions for each agent"""
        terminated = {}
        
        for agent_id in self.agent_configs.keys():
            # Default termination conditions
            agent_state = self.current_state.agent_states.get(agent_id, {})
            
            # Terminate if agent runs out of resources
            balance = agent_state.get('balance', 100000.0)
            if balance <= 0:
                terminated[agent_id] = True
            else:
                terminated[agent_id] = False
        
        return terminated
    
    def _check_truncation(self) -> Dict[str, bool]:
        """Check truncation conditions (time limits, etc.)"""
        truncated = {agent_id: self.step_count >= self.max_steps 
                    for agent_id in self.agent_configs.keys()}
        return truncated
    
    def _create_info_dict(self, actions: Dict[str, np.ndarray], rewards: Dict[str, float]) -> Dict[str, Any]:
        """Create info dictionary with debugging information"""
        return {
            "step": self.step_count,
            "episode": self.episode_count,
            "actions_taken": len(actions),
            "total_reward": sum(rewards.values()),
            "average_reward": np.mean(list(rewards.values())) if rewards else 0.0,
            "market_conditions": self.current_state.market_conditions,
            "difficulty_level": self.difficulty_level
        }
    
    def _handle_episode_end(self):
        """Handle end of episode - logging, metrics, curriculum updates"""
        # Log episode results
        total_episode_reward = sum(self.episode_rewards.values())
        avg_episode_reward = total_episode_reward / len(self.episode_rewards)
        
        logger.info(f"Episode {self.episode_count} ended:")
        logger.info(f"  Steps: {self.step_count}")
        logger.info(f"  Total reward: {total_episode_reward:.2f}")
        logger.info(f"  Average reward: {avg_episode_reward:.2f}")
        
        # Store performance history
        episode_performance = {
            'episode': self.episode_count,
            'steps': self.step_count,
            'total_reward': total_episode_reward,
            'average_reward': avg_episode_reward,
            'agent_rewards': self.episode_rewards.copy(),
            'difficulty': self.difficulty_level,
            'timestamp': datetime.now().isoformat()
        }
        self.performance_history.append(episode_performance)
        
        # Update curriculum difficulty if enabled
        if self.curriculum_enabled:
            self._update_curriculum_difficulty(avg_episode_reward)
        
        # Clean up episode data
        self._cleanup_episode_data()
    
    def _update_curriculum_difficulty(self, avg_reward: float):
        """Update curriculum difficulty based on performance"""
        # Simple curriculum logic - increase difficulty if performing well
        if len(self.performance_history) >= 10:
            recent_performance = [ep['average_reward'] for ep in self.performance_history[-10:]]
            avg_recent_performance = np.mean(recent_performance)
            
            # Increase difficulty if consistently good performance
            if avg_recent_performance > 0.8 and self.difficulty_level < 1.0:
                self.difficulty_level = min(1.0, self.difficulty_level + 0.1)
                logger.info(f"Increased difficulty to {self.difficulty_level:.2f}")
            
            # Decrease difficulty if consistently poor performance
            elif avg_recent_performance < 0.3 and self.difficulty_level > 0.1:
                self.difficulty_level = max(0.1, self.difficulty_level - 0.05)
                logger.info(f"Decreased difficulty to {self.difficulty_level:.2f}")
    
    def _cleanup_episode_data(self):
        """Clean up episode-specific data"""
        # Keep only last 100 episodes in performance history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.performance_history:
            return {}
        
        recent_episodes = self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        
        return {
            'total_episodes': len(self.performance_history),
            'avg_episode_length': np.mean([ep['steps'] for ep in self.performance_history]),
            'avg_total_reward': np.mean([ep['total_reward'] for ep in self.performance_history]),
            'recent_avg_reward': np.mean([ep['average_reward'] for ep in recent_episodes]),
            'current_difficulty': self.difficulty_level,
            'reward_trend': np.polyfit(range(len(self.performance_history)), 
                                     [ep['average_reward'] for ep in self.performance_history], 1)[0] if len(self.performance_history) > 1 else 0,
            'agent_performance': self._calculate_agent_performance_metrics()
        }
    
    def _calculate_agent_performance_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics for each agent"""
        agent_metrics = {}
        
        for agent_id in self.agent_configs.keys():
            if self.performance_history:
                agent_rewards = [ep['agent_rewards'].get(agent_id, 0.0) 
                               for ep in self.performance_history]
                
                agent_metrics[agent_id] = {
                    'avg_reward': np.mean(agent_rewards),
                    'std_reward': np.std(agent_rewards),
                    'best_reward': np.max(agent_rewards) if agent_rewards else 0.0,
                    'worst_reward': np.min(agent_rewards) if agent_rewards else 0.0,
                    'consistency': 1.0 - (np.std(agent_rewards) / (np.mean(agent_rewards) + 1e-8))
                }
        
        return agent_metrics
    
    def render(self, mode: str = "human"):
        """Render the environment (for debugging/visualization)"""
        if mode == "human":
            self._render_text()
        elif mode == "rgb_array":
            return self._render_rgb_array()
    
    def _render_text(self):
        """Simple text rendering of environment state"""
        print(f"\n--- Environment Step {self.step_count} ---")
        print(f"Episode: {self.episode_count}")
        print(f"Difficulty: {self.difficulty_level:.2f}")
        
        if self.current_state:
            print(f"Market Conditions: {self.current_state.market_conditions}")
            print("Agent States:")
            for agent_id, state in self.current_state.agent_states.items():
                print(f"  {agent_id}: Balance={state.get('balance', 0):.2f}, "
                      f"Success Rate={state.get('success_rate', 0):.2f}")
    
    def _render_rgb_array(self) -> np.ndarray:
        """Create RGB array representation (placeholder)"""
        # Return simple placeholder image
        return np.zeros((64, 64, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up environment resources"""
        self.performance_history.clear()
        logger.info("Environment closed")

class RewardShaper:
    """Helper class for reward shaping in MARL environments"""
    
    @staticmethod
    def difference_reward(agent_id: str, global_reward: float, 
                         counterfactual_reward: float) -> float:
        """Calculate difference reward for individual agent contribution"""
        return global_reward - counterfactual_reward
    
    @staticmethod
    def shaped_reward(base_reward: float, potential_function: callable,
                     state: Any, next_state: Any, gamma: float = 0.99) -> float:
        """Apply potential-based reward shaping"""
        return base_reward + gamma * potential_function(next_state) - potential_function(state)
    
    @staticmethod
    def coordination_bonus(agent_actions: Dict[str, np.ndarray],
                          coordination_target: Optional[np.ndarray] = None,
                          bonus_weight: float = 0.1) -> Dict[str, float]:
        """Calculate coordination bonus for aligned actions"""
        if coordination_target is None or len(agent_actions) < 2:
            return {agent_id: 0.0 for agent_id in agent_actions.keys()}
        
        bonuses = {}
        for agent_id, action in agent_actions.items():
            # Simple cosine similarity bonus
            similarity = np.dot(action.flatten(), coordination_target.flatten())
            similarity /= (np.linalg.norm(action.flatten()) * np.linalg.norm(coordination_target.flatten()) + 1e-8)
            bonuses[agent_id] = similarity * bonus_weight
        
        return bonuses
    
    @staticmethod
    def fairness_reward(resource_allocation: Dict[str, float],
                       agent_needs: Dict[str, float]) -> Dict[str, float]:
        """Calculate fairness-based rewards for resource allocation"""
        allocations = np.array(list(resource_allocation.values()))
        needs = np.array(list(agent_needs.values()))
        
        # Gini coefficient for fairness
        gini = RewardShaper._calculate_gini_coefficient(allocations)
        fairness_bonus = (1.0 - gini) * 0.1
        
        # Efficiency bonus
        satisfaction_ratios = np.minimum(allocations, needs) / (needs + 1e-8)
        efficiency_bonus = np.mean(satisfaction_ratios) * 0.1
        
        total_bonus = fairness_bonus + efficiency_bonus
        
        return {agent_id: total_bonus for agent_id in resource_allocation.keys()}
    
    @staticmethod
    def _calculate_gini_coefficient(values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement"""
        if len(values) == 0:
            return 0.0
        
        sorted_values = np.sort(values)
        n = len(values)
        cumulative = np.cumsum(sorted_values)
        
        return (2.0 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumulative[-1]) - (n + 1) / n