"""
Base RL Agent and Configuration
Foundation for all RL algorithms in the system
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from datetime import datetime
import logging
import random
import os
import json

from utils.observability.logging import get_logger

logger = get_logger(__name__)

@dataclass
class RLConfig:
    """Base configuration for RL agents"""
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    
    # Learning parameters
    learning_rate: float = 3e-4
    discount_factor: float = 0.99
    batch_size: int = 256
    
    # Exploration
    exploration_noise: float = 0.1
    exploration_decay: float = 0.995
    min_exploration: float = 0.01
    
    # Training
    update_frequency: int = 1
    target_update_frequency: int = 100
    max_gradient_norm: float = 10.0
    
    # Experience replay
    buffer_size: int = 1000000
    warmup_steps: int = 1000
    
    # Device and optimization
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer: str = "adam"
    weight_decay: float = 0.0
    
    # Checkpointing
    save_frequency: int = 10000
    checkpoint_dir: str = "rl_checkpoints"
    
    # Evaluation
    eval_frequency: int = 5000
    eval_episodes: int = 10
    
    # Logging
    log_frequency: int = 1000
    tensorboard_log: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'batch_size': self.batch_size,
            'exploration_noise': self.exploration_noise,
            'exploration_decay': self.exploration_decay,
            'min_exploration': self.min_exploration,
            'update_frequency': self.update_frequency,
            'target_update_frequency': self.target_update_frequency,
            'max_gradient_norm': self.max_gradient_norm,
            'buffer_size': self.buffer_size,
            'warmup_steps': self.warmup_steps,
            'device': self.device,
            'optimizer': self.optimizer,
            'weight_decay': self.weight_decay
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RLConfig':
        """Create config from dictionary"""
        return cls(**{k: v for k, v in config_dict.items() if hasattr(cls, k)})

class BaseRLAgent(ABC):
    """
    Base class for all RL agents in the system
    Provides common functionality for training, evaluation, and checkpointing
    """
    
    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        config: RLConfig,
        action_space_type: str = "continuous"  # "continuous" or "discrete"
    ):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.action_space_type = action_space_type
        
        # Set device
        self.device = torch.device(config.device)
        
        # Training state
        self.total_steps = 0
        self.total_episodes = 0
        self.current_exploration = config.exploration_noise
        
        # Performance tracking
        self.episode_rewards = []
        self.training_losses = []
        self.evaluation_scores = []
        
        # Networks (to be initialized by subclasses)
        self.networks = {}
        self.optimizers = {}
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__} agent {agent_id}")
        logger.info(f"State dim: {state_dim}, Action dim: {action_dim}")
        logger.info(f"Device: {self.device}")
    
    @abstractmethod
    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        """Select action given state"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    def update(self, experiences: Dict[str, Any]) -> Dict[str, float]:
        """Update agent parameters from experiences"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint"""
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint"""
        logger.info(f'Method {function_name} called')
        return {}
    
    def preprocess_state(self, state: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Preprocess state for network input"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        
        return state.to(self.device)
    
    def postprocess_action(self, action: torch.Tensor) -> np.ndarray:
        """Postprocess action for environment"""
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        
        if len(action.shape) == 2 and action.shape[0] == 1:
            action = action[0]
        
        return action
    
    def add_exploration_noise(self, action: np.ndarray) -> np.ndarray:
        """Add exploration noise to action"""
        if self.action_space_type == "continuous":
            # Gaussian noise for continuous actions
            noise = np.random.normal(0, self.current_exploration, action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        else:
            # Epsilon-greedy for discrete actions
            if np.random.random() < self.current_exploration:
                action = np.random.randint(0, self.action_dim)
        
        return action
    
    def update_exploration(self):
        """Update exploration rate"""
        self.current_exploration = max(
            self.config.min_exploration,
            self.current_exploration * self.config.exploration_decay
        )
    
    def soft_update_target_networks(self, tau: float = 0.005):
        """Soft update target networks"""
        for name, network in self.networks.items():
            if "target" in name:
                main_name = name.replace("_target", "")
                if main_name in self.networks:
                    main_network = self.networks[main_name]
                    target_network = network
                    
                    for target_param, main_param in zip(
                        target_network.parameters(), main_network.parameters()
                    ):
                        target_param.data.copy_(
                            tau * main_param.data + (1.0 - tau) * target_param.data
                        )
    
    def hard_update_target_networks(self):
        """Hard update target networks"""
        for name, network in self.networks.items():
            if "target" in name:
                main_name = name.replace("_target", "")
                if main_name in self.networks:
                    network.load_state_dict(self.networks[main_name].state_dict())
    
    def clip_gradients(self, networks: List[nn.Module]):
        """Clip gradients to prevent exploding gradients"""
        for network in networks:
            torch.nn.utils.clip_grad_norm_(
                network.parameters(), self.config.max_gradient_norm
            )
    
    def get_optimizer(self, parameters, lr: Optional[float] = None) -> torch.optim.Optimizer:
        """Create optimizer for network parameters"""
        lr = lr or self.config.learning_rate
        
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(parameters, lr=lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "rmsprop":
            return optim.RMSprop(parameters, lr=lr, weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(parameters, lr=lr, weight_decay=self.config.weight_decay, 
                           momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def log_training_metrics(self, step: int, metrics: Dict[str, float]):
        """Log training metrics"""
        if step % self.config.log_frequency == 0:
            metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            logger.info(f"Agent {self.agent_id} Step {step}: {metrics_str}")
            
            # Store training losses
            self.training_losses.append({
                'step': step,
                'metrics': metrics.copy(),
                'timestamp': datetime.now().isoformat()
            })
    
    def evaluate_agent(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate agent performance"""
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Convert single agent state to dict format if needed
                if isinstance(state, np.ndarray):
                    action = self.select_action(state, evaluation=True)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    episode_reward += reward
                else:
                    # Multi-agent environment
                    if self.agent_id in state:
                        action = self.select_action(state[self.agent_id], evaluation=True)
                        actions = {self.agent_id: action}
                        next_state, rewards, terminated, truncated, _ = env.step(actions)
                        done = terminated.get(self.agent_id, False) or truncated.get(self.agent_id, False)
                        episode_reward += rewards.get(self.agent_id, 0.0)
                    else:
                        break
                
                state = next_state
                episode_length += 1
                
                if episode_length > 1000:  # Prevent infinite episodes
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        eval_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'max_reward': np.max(episode_rewards),
            'min_reward': np.min(episode_rewards)
        }
        
        self.evaluation_scores.append({
            'step': self.total_steps,
            'metrics': eval_metrics,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Agent {self.agent_id} Evaluation: "
                   f"Mean Reward: {eval_metrics['mean_reward']:.2f} "
                   f"(±{eval_metrics['std_reward']:.2f})")
        
        return eval_metrics
    
    def should_update(self) -> bool:
        """Check if agent should update"""
        return (self.total_steps >= self.config.warmup_steps and 
                self.total_steps % self.config.update_frequency == 0)
    
    def should_update_target(self) -> bool:
        """Check if target networks should be updated"""
        return self.total_steps % self.config.target_update_frequency == 0
    
    def should_evaluate(self) -> bool:
        """Check if agent should be evaluated"""
        return self.total_steps % self.config.eval_frequency == 0
    
    def should_save_checkpoint(self) -> bool:
        """Check if checkpoint should be saved"""
        return self.total_steps % self.config.save_frequency == 0
    
    def get_checkpoint_path(self, step: Optional[int] = None) -> str:
        """Get checkpoint file path"""
        step = step or self.total_steps
        filename = f"{self.agent_id}_{self.__class__.__name__}_step_{step}.pt"
        return os.path.join(self.config.checkpoint_dir, filename)
    
    def save_training_metrics(self, filepath: Optional[str] = None):
        """Save training metrics to file"""
        if filepath is None:
            filepath = os.path.join(
                self.config.checkpoint_dir,
                f"{self.agent_id}_metrics_{self.total_steps}.json"
            )
        
        metrics_data = {
            'agent_id': self.agent_id,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'config': self.config.to_dict(),
            'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
            'training_losses': self.training_losses[-100:],   # Last 100 updates
            'evaluation_scores': self.evaluation_scores,
            'current_exploration': self.current_exploration,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        logger.info(f"Saved training metrics to {filepath}")
    
    def load_training_metrics(self, filepath: str):
        """Load training metrics from file"""
        with open(filepath, 'r') as f:
            metrics_data = json.load(f)
        
        self.total_steps = metrics_data.get('total_steps', 0)
        self.total_episodes = metrics_data.get('total_episodes', 0)
        self.episode_rewards = metrics_data.get('episode_rewards', [])
        self.training_losses = metrics_data.get('training_losses', [])
        self.evaluation_scores = metrics_data.get('evaluation_scores', [])
        self.current_exploration = metrics_data.get('current_exploration', self.config.exploration_noise)
        
        logger.info(f"Loaded training metrics from {filepath}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'agent_info': {
                'agent_id': self.agent_id,
                'agent_type': self.__class__.__name__,
                'total_steps': self.total_steps,
                'total_episodes': self.total_episodes,
                'current_exploration': self.current_exploration
            },
            'training_performance': {},
            'evaluation_performance': {},
            'recent_performance': {}
        }
        
        # Training performance
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-50:]  # Last 50 episodes
            summary['training_performance'] = {
                'total_episodes': len(self.episode_rewards),
                'mean_reward': np.mean(self.episode_rewards),
                'recent_mean_reward': np.mean(recent_rewards),
                'best_reward': np.max(self.episode_rewards),
                'reward_trend': np.polyfit(range(len(recent_rewards)), recent_rewards, 1)[0] if len(recent_rewards) > 1 else 0.0
            }
        
        # Evaluation performance
        if self.evaluation_scores:
            latest_eval = self.evaluation_scores[-1]['metrics']
            summary['evaluation_performance'] = latest_eval
        
        # Recent training losses
        if self.training_losses:
            recent_losses = self.training_losses[-10:]  # Last 10 updates
            summary['recent_performance'] = {
                'recent_updates': len(recent_losses),
                'average_losses': {
                    key: np.mean([loss['metrics'].get(key, 0) for loss in recent_losses])
                    for key in recent_losses[0]['metrics'].keys() if recent_losses
                }
            }
        
        return summary
    
    def reset_episode(self):
        """Reset for new episode"""
        self.total_episodes += 1
    
    def step(self):
        """Increment step counter"""
        self.total_steps += 1
    
    def set_training_mode(self, training: bool = True):
        """Set networks to training or evaluation mode"""
        for network in self.networks.values():
            if hasattr(network, 'train'):
                network.train(training)
    
    def to_device(self, data: Union[torch.Tensor, Dict, List, Tuple]):
        """Move data to device recursively"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(item) for item in data)
        else:
            return data
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics"""
        if torch.cuda.is_available() and self.device.type == 'cuda':
            return {
                'gpu_memory_allocated': torch.cuda.memory_allocated(self.device) / 1024**3,  # GB
                'gpu_memory_cached': torch.cuda.memory_reserved(self.device) / 1024**3,      # GB
                'gpu_memory_max': torch.cuda.max_memory_allocated(self.device) / 1024**3     # GB
            }
        else:
            return {'cpu_memory': 0.0}
    
    def cleanup(self):
        """Clean up resources"""
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear large data structures
        self.episode_rewards = self.episode_rewards[-100:]  # Keep last 100
        self.training_losses = self.training_losses[-100:]  # Keep last 100
        
        logger.info(f"Cleaned up resources for agent {self.agent_id}")

class MultiAgentRLMixin:
    """Mixin class for multi-agent specific functionality"""
    
    def __init__(self, num_agents: int, agent_ids: List[str]):
        self.num_agents = num_agents
        self.agent_ids = agent_ids
        self.agent_id_to_index = {agent_id: i for i, agent_id in enumerate(agent_ids)}
    
    def get_agent_index(self, agent_id: str) -> int:
        """Get agent index from agent ID"""
        return self.agent_id_to_index.get(agent_id, 0)
    
    def process_multi_agent_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Process batch for multi-agent learning"""
        processed_batch = {}
        
        # Ensure all tensors have agent dimension
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if len(value.shape) == 2:  # [batch_size, feature_dim]
                    # Reshape to [batch_size, num_agents, feature_dim]
                    batch_size, feature_dim = value.shape
                    if feature_dim % self.num_agents == 0:
                        agent_feature_dim = feature_dim // self.num_agents
                        processed_batch[key] = value.view(batch_size, self.num_agents, agent_feature_dim)
                    else:
                        processed_batch[key] = value
                else:
                    processed_batch[key] = value
            else:
                processed_batch[key] = value
        
        return processed_batch
    
    def compute_centralized_critic_input(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Compute input for centralized critic"""
        batch_size = states.shape[0]
        
        # Flatten states and actions across agents
        states_flat = states.view(batch_size, -1)  # [batch_size, num_agents * state_dim]
        actions_flat = actions.view(batch_size, -1)  # [batch_size, num_agents * action_dim]
        
        # Concatenate states and actions
        critic_input = torch.cat([states_flat, actions_flat], dim=1)
        
        return critic_input