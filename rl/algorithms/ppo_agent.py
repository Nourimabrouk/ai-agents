"""
Proximal Policy Optimization (PPO) Agent
State-of-the-art on-policy RL algorithm optimized for business environments
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import collections
from torch.utils.data import DataLoader, TensorDataset

from .base_algorithm import BaseRLAgent, RLConfig
from .networks import PolicyNetwork, ValueNetwork, MLP
from utils.observability.logging import get_logger

logger = get_logger(__name__)

@dataclass
class PPOConfig(RLConfig):
    """PPO-specific configuration"""
    # PPO hyperparameters
    epsilon_clip: float = 0.2  # Clipping parameter
    value_loss_coeff: float = 0.5  # Value function loss coefficient
    entropy_coeff: float = 0.01  # Entropy bonus coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training parameters
    ppo_epochs: int = 4  # Number of PPO epochs per update
    num_minibatches: int = 32  # Number of minibatches per epoch
    gae_lambda: float = 0.95  # GAE lambda parameter
    normalize_advantages: bool = True  # Normalize advantages
    
    # Value function parameters
    use_clipped_value_loss: bool = True  # Use clipped value loss
    
    # Adaptive parameters
    adaptive_kl_penalty: bool = False  # Use adaptive KL penalty
    target_kl: float = 0.01  # Target KL divergence
    
    # Curriculum learning
    curriculum_learning: bool = False
    success_rate_threshold: float = 0.8

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, max_size: int, state_dim: int, action_dim: int, device: str = "cpu"):
        self.max_size = max_size
        self.device = device
        
        # Initialize buffers
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(max_size, dtype=torch.bool, device=device)
        
        # GAE computation
        self.advantages = torch.zeros(max_size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(max_size, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.size = 0
    
    def store(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        done: bool
    ):
        """Store experience"""
        if self.ptr >= self.max_size:
            logger.warning("PPO buffer overflow - consider increasing buffer size")
            return
        
        self.states[self.ptr] = torch.FloatTensor(state).to(self.device)
        self.actions[self.ptr] = torch.FloatTensor(action).to(self.device)
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr += 1
        self.size = min(self.size + 1, self.max_size)
    
    def compute_gae(self, last_value: float = 0.0, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation"""
        gae = 0
        
        for step in reversed(range(self.ptr)):
            if step == self.ptr - 1:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = last_value
            else:
                next_non_terminal = 1.0 - float(self.dones[step])
                next_value = self.values[step + 1]
            
            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            self.advantages[step] = gae
            self.returns[step] = gae + self.values[step]
    
    def get_data(self) -> Dict[str, torch.Tensor]:
        """Get all stored data"""
        if self.ptr == 0:
            return {}
        
        return {
            'states': self.states[:self.ptr],
            'actions': self.actions[:self.ptr],
            'rewards': self.rewards[:self.ptr],
            'values': self.values[:self.ptr],
            'log_probs': self.log_probs[:self.ptr],
            'dones': self.dones[:self.ptr],
            'advantages': self.advantages[:self.ptr],
            'returns': self.returns[:self.ptr]
        }
    
    def clear(self):
        """Clear buffer"""
        self.ptr = 0
        self.size = 0

class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization Agent
    Implements PPO algorithm with GAE and clipping
    """
    
    def __init__(
        self,
        agent_id: str,
        state_dim: int,
        action_dim: int,
        config: PPOConfig,
        max_action: float = 1.0
    ):
        super().__init__(agent_id, state_dim, action_dim, config, "continuous")
        
        self.config: PPOConfig = config
        self.max_action = max_action
        
        # Create networks
        self.policy_network = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            max_action=max_action
        ).to(self.device)
        
        self.value_network = ValueNetwork(
            state_dim=state_dim,
            hidden_dims=config.hidden_dims,
            activation=config.activation
        ).to(self.device)
        
        # Store networks
        self.networks = {
            'policy': self.policy_network,
            'value': self.value_network
        }
        
        # Create optimizers
        self.policy_optimizer = self.get_optimizer(self.policy_network.parameters())
        self.value_optimizer = self.get_optimizer(self.value_network.parameters())
        
        self.optimizers = {
            'policy': self.policy_optimizer,
            'value': self.value_optimizer
        }
        
        # Experience buffer
        self.buffer = PPOBuffer(
            max_size=config.buffer_size,
            state_dim=state_dim,
            action_dim=action_dim,
            device=self.device
        )
        
        # Adaptive KL penalty
        self.kl_penalty = 0.0
        self.kl_penalty_lr = 0.1
        
        logger.info(f"Initialized PPO agent {agent_id}")
        logger.info(f"Policy parameters: {sum(p.numel() for p in self.policy_network.parameters())}")
        logger.info(f"Value parameters: {sum(p.numel() for p in self.value_network.parameters())}")
    
    def select_action(self, state: np.ndarray, evaluation: bool = False) -> np.ndarray:
        """Select action using current policy"""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            if evaluation:
                # Deterministic action for evaluation
                action = self.policy_network.deterministic_action(state_tensor)
            else:
                # Stochastic action for training
                action, _ = self.policy_network.sample(state_tensor, reparameterize=False)
            
            action = self.postprocess_action(action)
        
        # Add exploration noise if not evaluating
        if not evaluation:
            action = self.add_exploration_noise(action)
        
        return action
    
    def select_action_with_info(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Select action and return additional info (value, log_prob)"""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            # Sample action
            action, log_prob = self.policy_network.sample(state_tensor, reparameterize=False)
            
            # Get value estimate
            value = self.value_network(state_tensor)
            
            action_np = self.postprocess_action(action)
            log_prob_np = log_prob.cpu().numpy().flatten()[0]
            value_np = value.cpu().numpy().flatten()[0]
        
        info = {
            'value': value_np,
            'log_prob': log_prob_np
        }
        
        return action_np, info
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        info: Dict[str, Any]
    ):
        """Store experience in buffer"""
        self.buffer.store(
            state=state,
            action=action,
            reward=reward,
            value=info.get('value', 0.0),
            log_prob=info.get('log_prob', 0.0),
            done=done
        )
    
    def update(self, experiences: Dict[str, Any] = None) -> Dict[str, float]:
        """Update PPO agent"""
        if self.buffer.ptr < self.config.batch_size:
            return {}
        
        # Compute advantages using GAE
        last_value = 0.0
        if experiences and 'last_state' in experiences:
            with torch.no_grad():
                last_state = self.preprocess_state(experiences['last_state'])
                last_value = self.value_network(last_state).cpu().numpy().flatten()[0]
        
        self.buffer.compute_gae(
            last_value=last_value,
            gamma=self.config.discount_factor,
            gae_lambda=self.config.gae_lambda
        )
        
        # Get buffer data
        data = self.buffer.get_data()
        
        # Normalize advantages
        if self.config.normalize_advantages and len(data['advantages']) > 1:
            advantages = data['advantages']
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            data['advantages'] = advantages
        
        # PPO update
        update_info = self._ppo_update(data)
        
        # Clear buffer
        self.buffer.clear()
        
        # Update exploration
        self.update_exploration()
        
        # Log metrics
        self.log_training_metrics(self.total_steps, update_info)
        
        return update_info
    
    def _ppo_update(self, data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform PPO update"""
        states = data['states']
        actions = data['actions']
        old_log_probs = data['log_probs']
        advantages = data['advantages']
        returns = data['returns']
        old_values = data['values']
        
        batch_size = states.shape[0]
        
        # Training statistics
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []
        clipfracs = []
        
        # Create dataset and dataloader
        dataset = TensorDataset(states, actions, old_log_probs, advantages, returns, old_values)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size // self.config.num_minibatches,
            shuffle=True
        )
        
        for epoch in range(self.config.ppo_epochs):
            for batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_old_values in dataloader:
                
                # Current policy evaluation
                mean, log_std = self.policy_network(batch_states)
                current_values = self.value_network(batch_states).squeeze(-1)
                
                # Compute current log probabilities
                current_log_probs = self._compute_log_prob(mean, log_std, batch_actions)
                current_log_probs = current_log_probs.squeeze(-1)
                
                # Compute entropy
                entropy = self._compute_entropy(log_std)
                
                # Policy loss
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                
                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.epsilon_clip, 1.0 + self.config.epsilon_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                if self.config.use_clipped_value_loss:
                    # Clipped value loss
                    value_pred_clipped = batch_old_values + torch.clamp(
                        current_values - batch_old_values,
                        -self.config.epsilon_clip,
                        self.config.epsilon_clip
                    )
                    value_losses_unclipped = F.mse_loss(current_values, batch_returns, reduction='none')
                    value_losses_clipped = F.mse_loss(value_pred_clipped, batch_returns, reduction='none')
                    value_loss = torch.max(value_losses_unclipped, value_losses_clipped).mean()
                else:
                    value_loss = F.mse_loss(current_values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # KL divergence (for adaptive penalty)
                kl_div = (batch_old_log_probs - current_log_probs).mean()
                
                # Total policy loss
                total_policy_loss = policy_loss + self.config.entropy_coeff * entropy_loss
                
                # Adaptive KL penalty
                if self.config.adaptive_kl_penalty:
                    total_policy_loss += self.kl_penalty * kl_div
                
                # Update policy
                self.policy_optimizer.zero_grad()
                total_policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                
                # Update value function
                self.value_optimizer.zero_grad()
                (self.config.value_loss_coeff * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.config.max_grad_norm)
                self.value_optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                kl_divergences.append(kl_div.item())
                
                # Compute clipfrac (fraction of ratios clipped)
                clipfrac = ((ratio - 1.0).abs() > self.config.epsilon_clip).float().mean()
                clipfracs.append(clipfrac.item())
        
        # Update adaptive KL penalty
        if self.config.adaptive_kl_penalty:
            mean_kl = np.mean(kl_divergences)
            if mean_kl > 2 * self.config.target_kl:
                self.kl_penalty *= 1.5
            elif mean_kl < 0.5 * self.config.target_kl:
                self.kl_penalty *= 0.8
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences),
            'clipfrac': np.mean(clipfracs),
            'kl_penalty': self.kl_penalty,
            'exploration_rate': self.current_exploration
        }
    
    def _compute_log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action under current policy"""
        return self.policy_network._compute_log_prob(mean, log_std, action)
    
    def _compute_entropy(self, log_std: torch.Tensor) -> torch.Tensor:
        """Compute entropy of current policy"""
        # Entropy of multivariate Gaussian
        return (log_std + 0.5 * np.log(2.0 * np.pi * np.e)).sum(dim=-1)
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save agent checkpoint"""
        checkpoint = {
            'agent_id': self.agent_id,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'config': self.config.to_dict(),
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'value_optimizer': self.value_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'current_exploration': self.current_exploration,
            'kl_penalty': self.kl_penalty,
            'episode_rewards': self.episode_rewards[-100:],  # Last 100 episodes
            'training_losses': self.training_losses[-100:]   # Last 100 updates
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"Saved PPO checkpoint to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Load network states
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        
        # Load optimizer states
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer'])
        
        # Load training state
        self.total_steps = checkpoint.get('total_steps', 0)
        self.total_episodes = checkpoint.get('total_episodes', 0)
        self.current_exploration = checkpoint.get('current_exploration', self.config.exploration_noise)
        self.kl_penalty = checkpoint.get('kl_penalty', 0.0)
        
        # Load performance history
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        
        logger.info(f"Loaded PPO checkpoint from {filepath}")
        logger.info(f"Resumed at step {self.total_steps}, episode {self.total_episodes}")
    
    def get_action_distribution_info(self, state: np.ndarray) -> Dict[str, Any]:
        """Get information about action distribution"""
        state_tensor = self.preprocess_state(state)
        
        with torch.no_grad():
            mean, log_std = self.policy_network(state_tensor)
            std = log_std.exp()
            
            return {
                'mean': mean.cpu().numpy().flatten(),
                'std': std.cpu().numpy().flatten(),
                'log_std': log_std.cpu().numpy().flatten(),
                'entropy': self._compute_entropy(log_std).cpu().numpy().flatten()[0]
            }
    
    def set_exploration_schedule(self, schedule_fn):
        """Set custom exploration schedule"""
        self.exploration_schedule = schedule_fn
    
    def update_exploration(self):
        """Update exploration rate (can be customized)"""
        if hasattr(self, 'exploration_schedule'):
            self.current_exploration = self.exploration_schedule(self.total_steps)
        else:
            # Default exponential decay
            super().update_exploration()

def create_ppo_agent(
    agent_id: str,
    env,
    config: Optional[PPOConfig] = None,
    **kwargs
) -> PPOAgent:
    """Factory function to create PPO agent"""
    
    if config is None:
        config = PPOConfig(**kwargs)
    
    # Get environment dimensions
    if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        max_action = float(env.action_space.high[0])
    else:
        # Multi-agent environment
        state_dim = kwargs.get('state_dim', 64)
        action_dim = kwargs.get('action_dim', 4)
        max_action = kwargs.get('max_action', 1.0)
    
    agent = PPOAgent(
        agent_id=agent_id,
        state_dim=state_dim,
        action_dim=action_dim,
        config=config,
        max_action=max_action
    )
    
    return agent