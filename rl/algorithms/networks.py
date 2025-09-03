"""
Neural Network Architectures for RL Algorithms
Optimized networks for different RL algorithm requirements
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Union
import math

class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
        batch_norm: bool = False,
        layer_norm: bool = False
    ):
        super().__init__()
        
        self.activation = self._get_activation(activation)
        self.output_activation = self._get_activation(output_activation) if output_activation else None
        
        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add normalization (except for last layer)
            if i < len(dims) - 2:
                if batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                elif layer_norm:
                    layers.append(nn.LayerNorm(dims[i + 1]))
                
                # Add activation (except for last layer)
                layers.append(self.activation)
                
                # Add dropout
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function"""
        if activation.lower() == "relu":
            return nn.ReLU()
        elif activation.lower() == "tanh":
            return nn.Tanh()
        elif activation.lower() == "sigmoid":
            return nn.Sigmoid()
        elif activation.lower() == "leaky_relu":
            return nn.LeakyReLU(0.2)
        elif activation.lower() == "elu":
            return nn.ELU()
        elif activation.lower() == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier/Glorot initialization
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        if self.output_activation:
            x = self.output_activation(x)
        return x

class PolicyNetwork(nn.Module):
    """Policy network for continuous control"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        max_action: float = 1.0,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared trunk
        self.trunk = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=hidden_dims[-1],
            activation=activation
        )
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize final layers with smaller weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning mean and log_std"""
        features = self.trunk(state)
        
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        
        # Clip log_std to prevent numerical issues
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, reparameterize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        if reparameterize:
            # Reparameterization trick
            normal = torch.randn_like(mean)
            action = mean + std * normal
        else:
            # Direct sampling
            normal = torch.distributions.Normal(mean, std)
            action = normal.sample()
        
        # Apply tanh squashing and scale
        action = torch.tanh(action) * self.max_action
        
        # Compute log probability
        log_prob = self._compute_log_prob(mean, log_std, action)
        
        return action, log_prob
    
    def _compute_log_prob(self, mean: torch.Tensor, log_std: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action under current policy"""
        # Inverse tanh to get pre-squashed action
        action_unsquashed = torch.atanh(torch.clamp(action / self.max_action, -0.999, 0.999))
        
        # Log probability of Gaussian
        std = log_std.exp()
        log_prob = -0.5 * (((action_unsquashed - mean) / std) ** 2 + 2 * log_std + math.log(2 * math.pi))
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        # Correct for tanh squashing
        log_prob = log_prob - torch.log(1 - (action / self.max_action) ** 2 + 1e-6).sum(dim=-1, keepdim=True)
        
        return log_prob
    
    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        """Get deterministic action (mean of policy)"""
        mean, _ = self.forward(state)
        return torch.tanh(mean) * self.max_action

class ValueNetwork(nn.Module):
    """Value network for estimating state values"""
    
    def __init__(
        self,
        state_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: Optional[str] = None
    ):
        super().__init__()
        
        self.network = MLP(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation,
            output_activation=output_activation
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class CriticNetwork(nn.Module):
    """Critic network for state-action value estimation"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        concat_layer: int = 1  # Layer at which to concatenate actions
    ):
        super().__init__()
        
        self.concat_layer = concat_layer
        
        if concat_layer == 0:
            # Concatenate at input
            self.network = MLP(
                input_dim=state_dim + action_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                activation=activation
            )
        else:
            # Separate state and action processing
            self.state_encoder = MLP(
                input_dim=state_dim,
                hidden_dims=hidden_dims[:concat_layer],
                output_dim=hidden_dims[concat_layer - 1],
                activation=activation
            )
            
            # Combined processing after concatenation
            self.combined_network = MLP(
                input_dim=hidden_dims[concat_layer - 1] + action_dim,
                hidden_dims=hidden_dims[concat_layer:],
                output_dim=1,
                activation=activation
            )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.concat_layer == 0:
            # Concatenate at input
            x = torch.cat([state, action], dim=-1)
            return self.network(x)
        else:
            # Process state first, then concatenate with action
            state_features = self.state_encoder(state)
            x = torch.cat([state_features, action], dim=-1)
            return self.combined_network(x)

class QNetwork(nn.Module):
    """Q-network for discrete action spaces"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        dueling: bool = False
    ):
        super().__init__()
        
        self.action_dim = action_dim
        self.dueling = dueling
        
        if dueling:
            # Dueling DQN architecture
            self.feature_network = MLP(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=hidden_dims[-1],
                activation=activation
            )
            
            # Value stream
            self.value_stream = nn.Linear(hidden_dims[-1], 1)
            
            # Advantage stream
            self.advantage_stream = nn.Linear(hidden_dims[-1], action_dim)
        else:
            # Standard DQN architecture
            self.network = MLP(
                input_dim=state_dim,
                hidden_dims=hidden_dims,
                output_dim=action_dim,
                activation=activation
            )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if self.dueling:
            features = self.feature_network(state)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Combine value and advantage
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            return q_values
        else:
            return self.network(state)

class AttentionNetwork(nn.Module):
    """Multi-head attention network for processing sequential or multi-agent data"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Attention mask [batch_size, seq_len] (True for valid positions)
            return_attention: Whether to return attention weights
        """
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Create attention mask for transformer (inverted: True for positions to mask)
        if mask is not None:
            # Transformer expects True for positions to mask
            attention_mask = ~mask
        else:
            attention_mask = None
        
        # Apply transformer
        if return_attention:
            # Manual attention computation for returning weights
            output = self.transformer(x, src_key_padding_mask=attention_mask)
            attention_weights = None  # Simplified - would need custom transformer layer
            return self.layer_norm(output), attention_weights
        else:
            output = self.transformer(x, src_key_padding_mask=attention_mask)
            return self.layer_norm(output)

class MixingNetwork(nn.Module):
    """Mixing network for QMIX algorithm"""
    
    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        hidden_dim: int = 32,
        hypernet_hidden_dim: int = 64
    ):
        super().__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # Hypernetwork for generating mixing network weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, num_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, hidden_dim)
        )
        
        # Hypernetwork for biases
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_hidden_dim),
            nn.ReLU(),
            nn.Linear(hypernet_hidden_dim, 1)
        )
    
    def forward(self, agent_q_values: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of mixing network
        
        Args:
            agent_q_values: Individual agent Q-values [batch_size, num_agents]
            state: Global state [batch_size, state_dim]
        
        Returns:
            Mixed Q-value [batch_size, 1]
        """
        batch_size = agent_q_values.shape[0]
        
        # Generate weights and biases from state
        w1 = torch.abs(self.hyper_w1(state))  # Ensure non-negative weights
        b1 = self.hyper_b1(state)
        w2 = torch.abs(self.hyper_w2(state))
        b2 = self.hyper_b2(state)
        
        # Reshape weights
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        w2 = w2.view(batch_size, self.hidden_dim, 1)
        b1 = b1.view(batch_size, 1, self.hidden_dim)
        
        # First layer: agent_q_values -> hidden
        agent_q_values = agent_q_values.view(batch_size, 1, self.num_agents)
        hidden = F.elu(torch.bmm(agent_q_values, w1) + b1)
        
        # Second layer: hidden -> output
        output = torch.bmm(hidden, w2) + b2.view(batch_size, 1, 1)
        
        return output.view(batch_size, 1)

class NoiseNetwork(nn.Module):
    """Noisy network for exploration (NoisyNet)"""
    
    def __init__(self, input_dim: int, output_dim: int, std_init: float = 0.5):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        
        # Noise buffers (not parameters)
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Reset network parameters"""
        mu_range = 1 / math.sqrt(self.input_dim)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.input_dim))
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.input_dim))
    
    def reset_noise(self):
        """Reset noise for both weights and biases"""
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)
        
        # Outer product for weight noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise"""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with noisy linear transformation"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)

def create_network(
    network_type: str,
    input_dim: int,
    output_dim: int,
    hidden_dims: List[int] = [256, 256],
    **kwargs
) -> nn.Module:
    """Factory function to create networks"""
    
    if network_type.lower() == "mlp":
        return MLP(input_dim, hidden_dims, output_dim, **kwargs)
    
    elif network_type.lower() == "policy":
        return PolicyNetwork(input_dim, output_dim, hidden_dims, **kwargs)
    
    elif network_type.lower() == "value":
        return ValueNetwork(input_dim, hidden_dims, **kwargs)
    
    elif network_type.lower() == "critic":
        # Assume action_dim is in kwargs
        action_dim = kwargs.pop('action_dim', output_dim)
        return CriticNetwork(input_dim, action_dim, hidden_dims, **kwargs)
    
    elif network_type.lower() == "q_network":
        return QNetwork(input_dim, output_dim, hidden_dims, **kwargs)
    
    elif network_type.lower() == "attention":
        return AttentionNetwork(input_dim, output_dim, **kwargs)
    
    elif network_type.lower() == "mixing":
        num_agents = kwargs.pop('num_agents')
        return MixingNetwork(num_agents, input_dim, output_dim, **kwargs)
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")

# Network initialization utilities
def init_weights(module: nn.Module, init_type: str = "xavier"):
    """Initialize network weights"""
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        if init_type == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight)
        else:
            nn.init.normal_(module.weight, 0.0, 0.02)
        
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)

def count_parameters(network: nn.Module) -> int:
    """Count total number of trainable parameters"""
    return sum(p.numel() for p in network.parameters() if p.requires_grad)