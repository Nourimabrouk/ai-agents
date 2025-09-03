"""
RL Training Pipeline
Comprehensive training system for multi-agent RL with curriculum learning
"""

import asyncio
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import json
import os
from pathlib import Path
import time
from collections import defaultdict, deque
import traceback

from ..algorithms.base_algorithm import BaseRLAgent
from ..algorithms.ppo_agent import PPOAgent, PPOConfig
from ..environments.base_environment import BaseMultiAgentEnvironment
from ..environments.trading_environment import TradingEnvironment
from ..environments.supply_chain_environment import SupplyChainEnvironment
from ..environments.resource_allocation_environment import ResourceAllocationEnvironment
from .curriculum import CurriculumManager, create_trading_curriculum
from data.synthetic_data_generator import SyntheticDataGenerator
from utils.observability.logging import get_logger

logger = get_logger(__name__)

@dataclass
class TrainingConfig:
    """Configuration for RL training pipeline"""
    # Training parameters
    total_episodes: int = 10000
    max_steps_per_episode: int = 1000
    eval_frequency: int = 500
    eval_episodes: int = 20
    
    # Environment settings
    environment_type: str = "trading"  # trading, supply_chain, resource_allocation
    environment_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    # Agent settings
    agent_type: str = "ppo"
    agent_configs: Dict[str, Any] = field(default_factory=dict)
    num_agents: int = 4
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: Optional[List] = None
    
    # Synthetic data integration
    use_synthetic_data: bool = True
    synthetic_data_refresh_frequency: int = 1000
    
    # Logging and checkpointing
    log_frequency: int = 100
    checkpoint_frequency: int = 2000
    save_best_only: bool = True
    
    # Performance monitoring
    performance_metrics: List[str] = field(default_factory=lambda: ["reward", "success_rate", "episode_length"])
    early_stopping_patience: int = 2000
    early_stopping_threshold: float = -np.inf
    
    # Distributed training
    use_parallel_envs: bool = False
    num_parallel_envs: int = 4
    
    # Experiment tracking
    experiment_name: str = "rl_training"
    output_dir: str = "rl_experiments"
    tags: List[str] = field(default_factory=list)

class TrainingLogger:
    """Training logger for RL experiments"""
    
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_file = self.log_dir / f"{experiment_name}.log"
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics storage
        self.metrics_history = defaultdict(list)
        self.episode_data = []
        
        # Performance tracking
        self.best_performance = -np.inf
        self.no_improvement_count = 0
        
        logger.info(f"Training logger initialized: {self.log_file}")
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log episode metrics"""
        timestamp = datetime.now().isoformat()
        
        # Store metrics
        for key, value in metrics.items():
            self.metrics_history[key].append(value)
        
        # Store episode data
        episode_record = {
            'episode': episode,
            'timestamp': timestamp,
            'metrics': metrics
        }
        self.episode_data.append(episode_record)
        
        # Update performance tracking
        if 'total_reward' in metrics:
            if metrics['total_reward'] > self.best_performance:
                self.best_performance = metrics['total_reward']
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
        
        # Write to log file periodically
        if episode % 100 == 0:
            self._write_to_file()
    
    def log_evaluation(self, episode: int, eval_metrics: Dict[str, Any]):
        """Log evaluation results"""
        logger.info(f"Evaluation at episode {episode}: {eval_metrics}")
        
        eval_record = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'type': 'evaluation',
            'metrics': eval_metrics
        }
        self.episode_data.append(eval_record)
    
    def log_curriculum_change(self, episode: int, stage_info: Dict[str, Any]):
        """Log curriculum stage changes"""
        logger.info(f"Curriculum change at episode {episode}: {stage_info}")
        
        curriculum_record = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'type': 'curriculum_change',
            'stage_info': stage_info
        }
        self.episode_data.append(curriculum_record)
    
    def _write_to_file(self):
        """Write accumulated data to file"""
        with open(self.log_file, 'a') as f:
            for record in self.episode_data[-100:]:  # Write last 100 records
                f.write(json.dumps(record) + '\n')
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
        
        summary = {
            'total_episodes': len(self.episode_data),
            'best_performance': self.best_performance,
            'no_improvement_count': self.no_improvement_count,
            'final_metrics': {}
        }
        
        # Calculate final metrics
        for key, values in self.metrics_history.items():
            if values:
                summary['final_metrics'][key] = {
                    'mean': np.mean(values[-100:]) if len(values) >= 100 else np.mean(values),
                    'std': np.std(values[-100:]) if len(values) >= 100 else np.std(values),
                    'trend': np.polyfit(range(len(values[-100:])), values[-100:], 1)[0] if len(values) >= 10 else 0.0
                }
        
        return summary
    
    def save_final_report(self):
        """Save final training report"""
        report_path = self.log_dir / f"{self.experiment_name}_report.json"
        
        report = {
            'experiment_name': self.experiment_name,
            'total_episodes': len(self.episode_data),
            'training_duration': self._calculate_training_duration(),
            'performance_summary': self.get_performance_summary(),
            'metrics_history': {k: v[-1000:] for k, v in self.metrics_history.items()},  # Last 1000 points
            'final_timestamp': datetime.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved final training report to {report_path}")
    
    def _calculate_training_duration(self) -> str:
        """Calculate total training duration"""
        if len(self.episode_data) < 2:
            return "0:00:00"
        
        start_time = datetime.fromisoformat(self.episode_data[0]['timestamp'])
        end_time = datetime.fromisoformat(self.episode_data[-1]['timestamp'])
        duration = end_time - start_time
        
        return str(duration).split('.')[0]  # Remove microseconds

class RLTrainingPipeline:
    """
    Comprehensive RL training pipeline with curriculum learning and synthetic data integration
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        environment: Optional[BaseMultiAgentEnvironment] = None,
        agents: Optional[Dict[str, BaseRLAgent]] = None
    ):
        self.config = config
        self.start_time = time.time()
        
        # Initialize synthetic data generator
        if config.use_synthetic_data:
            self.synthetic_generator = SyntheticDataGenerator(seed=42)
            self._generate_initial_synthetic_data()
        
        # Initialize environment
        self.environment = environment or self._create_environment()
        
        # Initialize agents
        self.agents = agents or self._create_agents()
        
        # Initialize curriculum
        self.curriculum_manager = None
        if config.use_curriculum:
            self.curriculum_manager = self._create_curriculum_manager()
        
        # Initialize logger
        self.logger = TrainingLogger(config.output_dir, config.experiment_name)
        
        # Training state
        self.current_episode = 0
        self.total_steps = 0
        self.best_eval_score = -np.inf
        self.training_active = False
        
        # Performance tracking
        self.episode_rewards = {agent_id: deque(maxlen=1000) for agent_id in self.agents.keys()}
        self.episode_lengths = deque(maxlen=1000)
        self.eval_scores = []
        
        logger.info(f"Initialized training pipeline: {config.experiment_name}")
        logger.info(f"Environment: {type(self.environment).__name__}")
        logger.info(f"Agents: {list(self.agents.keys())}")
        logger.info(f"Total parameters: {self._count_total_parameters()}")
    
    def _generate_initial_synthetic_data(self):
        """Generate initial synthetic data for training"""
        logger.info("Generating initial synthetic data...")
        
        # Generate comprehensive dataset
        self.synthetic_data = self.synthetic_generator.generate_complete_dataset(save_to_file=False)
        
        # Extract components for environment initialization
        self.agent_network_data = self.synthetic_data['agent_network']
        self.market_data = self.synthetic_data['market_data']
        self.trading_events = self.synthetic_data['trading_events']
        
        logger.info(f"Generated synthetic data with {len(self.agent_network_data['agents'])} agents and {len(self.market_data)} market data points")
    
    def _create_environment(self) -> BaseMultiAgentEnvironment:
        """Create training environment"""
        if self.config.environment_type == "trading":
            # Use synthetic market data if available
            if hasattr(self, 'market_data') and self.market_data:
                symbols = list(set([data['symbol'] for data in self.market_data]))
                initial_prices = {symbol: data['price'] for data in self.market_data[-len(symbols):] for symbol in symbols if data['symbol'] == symbol}
            else:
                symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT']
                initial_prices = {symbol: np.random.uniform(100, 500) for symbol in symbols}
            
            from ..environments.base_environment import AgentConfig
            agent_configs = [
                AgentConfig(agent_id=f"agent_{i}", role="executor", initial_balance=100000)
                for i in range(self.config.num_agents)
            ]
            
            return TradingEnvironment(
                agent_configs=agent_configs,
                symbols=symbols,
                initial_prices=initial_prices,
                max_steps=self.config.max_steps_per_episode,
                use_synthetic_data=True,
                **self.config.environment_kwargs
            )
            
        elif self.config.environment_type == "supply_chain":
            from ..environments.base_environment import AgentConfig
            agent_configs = [
                AgentConfig(agent_id=f"agent_{i}", role="executor", initial_balance=100000)
                for i in range(self.config.num_agents)
            ]
            
            return SupplyChainEnvironment(
                agent_configs=agent_configs,
                max_steps=self.config.max_steps_per_episode,
                **self.config.environment_kwargs
            )
            
        elif self.config.environment_type == "resource_allocation":
            from ..environments.base_environment import AgentConfig
            agent_configs = [
                AgentConfig(agent_id=f"agent_{i}", role="executor", initial_balance=100000)
                for i in range(self.config.num_agents)
            ]
            
            return ResourceAllocationEnvironment(
                agent_configs=agent_configs,
                max_steps=self.config.max_steps_per_episode,
                **self.config.environment_kwargs
            )
        else:
            raise ValueError(f"Unknown environment type: {self.config.environment_type}")
    
    def _create_agents(self) -> Dict[str, BaseRLAgent]:
        """Create RL agents"""
        agents = {}
        
        # Get observation and action dimensions from environment
        obs_dim = self.environment._get_observation_dimension()
        action_dim = self.environment._get_action_dimension()
        
        for i in range(self.config.num_agents):
            agent_id = f"agent_{i}"
            
            if self.config.agent_type == "ppo":
                agent_config = PPOConfig(**self.config.agent_configs)
                agent = PPOAgent(
                    agent_id=agent_id,
                    state_dim=obs_dim,
                    action_dim=action_dim,
                    config=agent_config
                )
            else:
                raise ValueError(f"Unknown agent type: {self.config.agent_type}")
            
            agents[agent_id] = agent
        
        return agents
    
    def _create_curriculum_manager(self) -> CurriculumManager:
        """Create curriculum manager"""
        if self.config.curriculum_stages:
            stages = self.config.curriculum_stages
        elif self.config.environment_type == "trading":
            stages = create_trading_curriculum()
        else:
            # Create basic curriculum
            from .curriculum import CurriculumStage, DifficultyLevel
            stages = [
                CurriculumStage(
                    name="basic",
                    difficulty=DifficultyLevel.EASY,
                    parameters={},
                    success_criteria={'reward': 0.5}
                )
            ]
        
        return CurriculumManager(
            stages=stages,
            save_path=os.path.join(self.config.output_dir, f"{self.config.experiment_name}_curriculum.json")
        )
    
    def _count_total_parameters(self) -> int:
        """Count total trainable parameters across all agents"""
        total_params = 0
        for agent in self.agents.values():
            for network in agent.networks.values():
                total_params += sum(p.numel() for p in network.parameters() if p.requires_grad)
        return total_params
    
    async def train(self) -> Dict[str, Any]:
        """Main training loop"""
        logger.info(f"Starting training for {self.config.total_episodes} episodes")
        self.training_active = True
        
        try:
            for episode in range(self.config.total_episodes):
                self.current_episode = episode
                
                # Run episode
                episode_results = await self._run_episode()
                
                # Log episode results
                self._log_episode_results(episode, episode_results)
                
                # Update agents
                await self._update_agents(episode_results)
                
                # Update curriculum
                if self.curriculum_manager:
                    await self._update_curriculum(episode_results)
                
                # Refresh synthetic data periodically
                if (self.config.use_synthetic_data and 
                    episode % self.config.synthetic_data_refresh_frequency == 0 and 
                    episode > 0):
                    await self._refresh_synthetic_data()
                
                # Evaluation
                if episode % self.config.eval_frequency == 0 and episode > 0:
                    eval_results = await self._evaluate_agents()
                    self._process_evaluation_results(episode, eval_results)
                
                # Checkpointing
                if episode % self.config.checkpoint_frequency == 0 and episode > 0:
                    await self._save_checkpoint(episode)
                
                # Early stopping check
                if self._should_stop_early():
                    logger.info(f"Early stopping triggered at episode {episode}")
                    break
                
                # Progress logging
                if episode % self.config.log_frequency == 0:
                    self._log_training_progress(episode)
        
        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            logger.error(traceback.format_exc())
        finally:
            self.training_active = False
            await self._finalize_training()
        
        return self._get_training_summary()
    
    async def _run_episode(self) -> Dict[str, Any]:
        """Run single training episode"""
        # Reset environment (with curriculum parameters if available)
        if self.curriculum_manager:
            curriculum_params = self.curriculum_manager.get_current_parameters()
            # Apply curriculum parameters to environment
            self._apply_curriculum_parameters(curriculum_params)
        
        observations, info = self.environment.reset()
        
        episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
        episode_length = 0
        episode_info = defaultdict(list)
        
        # Store experiences for PPO
        experiences = {agent_id: [] for agent_id in self.agents.keys()}
        
        done = {agent_id: False for agent_id in self.agents.keys()}
        
        while not all(done.values()) and episode_length < self.config.max_steps_per_episode:
            actions = {}
            action_infos = {}
            
            # Select actions for all agents
            for agent_id, agent in self.agents.items():
                if not done[agent_id] and agent_id in observations:
                    if hasattr(agent, 'select_action_with_info'):
                        # PPO agents return additional info
                        action, action_info = agent.select_action_with_info(observations[agent_id])
                        action_infos[agent_id] = action_info
                    else:
                        action = agent.select_action(observations[agent_id])
                        action_infos[agent_id] = {}
                    
                    actions[agent_id] = action
            
            # Execute actions in environment
            next_observations, rewards, terminated, truncated, env_info = self.environment.step(actions)
            
            # Store experiences
            for agent_id in self.agents.keys():
                if agent_id in observations and agent_id in actions:
                    experience = {
                        'state': observations[agent_id],
                        'action': actions[agent_id],
                        'reward': rewards.get(agent_id, 0.0),
                        'next_state': next_observations.get(agent_id, observations[agent_id]),
                        'done': terminated.get(agent_id, False) or truncated.get(agent_id, False),
                        'info': action_infos.get(agent_id, {})
                    }
                    experiences[agent_id].append(experience)
                    
                    # Update episode rewards
                    episode_rewards[agent_id] += rewards.get(agent_id, 0.0)
            
            # Update done status
            for agent_id in self.agents.keys():
                if agent_id in terminated:
                    done[agent_id] = terminated[agent_id] or truncated.get(agent_id, False)
            
            observations = next_observations
            episode_length += 1
            self.total_steps += 1
            
            # Store episode info
            for key, value in env_info.items():
                episode_info[key].append(value)
        
        # Store experiences in agent buffers (for PPO)
        for agent_id, agent_experiences in experiences.items():
            if hasattr(self.agents[agent_id], 'store_experience'):
                for exp in agent_experiences:
                    self.agents[agent_id].store_experience(
                        state=exp['state'],
                        action=exp['action'],
                        reward=exp['reward'],
                        done=exp['done'],
                        info=exp['info']
                    )
        
        return {
            'episode_rewards': episode_rewards,
            'episode_length': episode_length,
            'total_reward': sum(episode_rewards.values()),
            'experiences': experiences,
            'episode_info': dict(episode_info),
            'final_observations': observations
        }
    
    def _apply_curriculum_parameters(self, parameters: Dict[str, Any]):
        """Apply curriculum parameters to environment"""
        for key, value in parameters.items():
            if hasattr(self.environment, key):
                setattr(self.environment, key, value)
    
    async def _update_agents(self, episode_results: Dict[str, Any]):
        """Update all agents based on episode results"""
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'should_update') and agent.should_update():
                # Prepare update data
                update_data = {
                    'last_state': episode_results['final_observations'].get(agent_id)
                }
                
                # Update agent
                update_metrics = agent.update(update_data)
                
                # Log update metrics
                if update_metrics:
                    agent.step()  # Increment step counter
    
    async def _update_curriculum(self, episode_results: Dict[str, Any]):
        """Update curriculum based on episode performance"""
        # Calculate performance score (can be customized)
        total_reward = episode_results['total_reward']
        episode_length = episode_results['episode_length']
        
        # Normalize performance score
        performance_score = total_reward / max(1, episode_length) + 0.5  # Add bias to keep positive
        
        # Update curriculum
        stage_changed = self.curriculum_manager.update(performance_score)
        
        if stage_changed:
            stage_info = self.curriculum_manager.get_progress_info()
            self.logger.log_curriculum_change(self.current_episode, stage_info)
    
    async def _refresh_synthetic_data(self):
        """Refresh synthetic data for continued training"""
        logger.info("Refreshing synthetic data...")
        
        # Generate new synthetic data
        new_data = self.synthetic_generator.generate_complete_dataset(save_to_file=False)
        
        # Update stored data
        self.agent_network_data = new_data['agent_network']
        self.market_data = new_data['market_data']
        self.trading_events = new_data['trading_events']
        
        # Update environment with new data if applicable
        if hasattr(self.environment, 'update_market_data'):
            self.environment.update_market_data(self.market_data)
    
    async def _evaluate_agents(self) -> Dict[str, Any]:
        """Evaluate agent performance"""
        logger.info(f"Evaluating agents at episode {self.current_episode}")
        
        eval_rewards = {agent_id: [] for agent_id in self.agents.keys()}
        eval_lengths = []
        
        for eval_episode in range(self.config.eval_episodes):
            observations, _ = self.environment.reset()
            
            episode_rewards = {agent_id: 0.0 for agent_id in self.agents.keys()}
            episode_length = 0
            done = {agent_id: False for agent_id in self.agents.keys()}
            
            while not all(done.values()) and episode_length < self.config.max_steps_per_episode:
                actions = {}
                
                # Select actions (evaluation mode)
                for agent_id, agent in self.agents.items():
                    if not done[agent_id] and agent_id in observations:
                        action = agent.select_action(observations[agent_id], evaluation=True)
                        actions[agent_id] = action
                
                # Execute actions
                next_observations, rewards, terminated, truncated, _ = self.environment.step(actions)
                
                # Update episode rewards
                for agent_id in self.agents.keys():
                    if agent_id in rewards:
                        episode_rewards[agent_id] += rewards[agent_id]
                
                # Update done status
                for agent_id in self.agents.keys():
                    if agent_id in terminated:
                        done[agent_id] = terminated[agent_id] or truncated.get(agent_id, False)
                
                observations = next_observations
                episode_length += 1
            
            # Store evaluation results
            for agent_id in self.agents.keys():
                eval_rewards[agent_id].append(episode_rewards[agent_id])
            eval_lengths.append(episode_length)
        
        # Calculate evaluation metrics
        eval_results = {
            'mean_rewards': {agent_id: np.mean(rewards) for agent_id, rewards in eval_rewards.items()},
            'std_rewards': {agent_id: np.std(rewards) for agent_id, rewards in eval_rewards.items()},
            'mean_length': np.mean(eval_lengths),
            'total_mean_reward': np.mean([np.mean(rewards) for rewards in eval_rewards.values()])
        }
        
        return eval_results
    
    def _process_evaluation_results(self, episode: int, eval_results: Dict[str, Any]):
        """Process evaluation results"""
        self.logger.log_evaluation(episode, eval_results)
        
        # Update best score
        current_score = eval_results['total_mean_reward']
        if current_score > self.best_eval_score:
            self.best_eval_score = current_score
            if self.config.save_best_only:
                asyncio.create_task(self._save_best_checkpoint(episode))
        
        self.eval_scores.append(current_score)
    
    async def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            checkpoint_path = checkpoint_dir / f"{agent_id}_episode_{episode}.pt"
            agent.save_checkpoint(str(checkpoint_path))
        
        # Save curriculum state if available
        if self.curriculum_manager:
            curriculum_path = checkpoint_dir / f"curriculum_episode_{episode}.json"
            self.curriculum_manager.save_state(str(curriculum_path))
        
        logger.info(f"Saved checkpoint at episode {episode}")
    
    async def _save_best_checkpoint(self, episode: int):
        """Save best performing checkpoint"""
        checkpoint_dir = Path(self.config.output_dir) / "best_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            checkpoint_path = checkpoint_dir / f"{agent_id}_best.pt"
            agent.save_checkpoint(str(checkpoint_path))
        
        logger.info(f"Saved best checkpoint at episode {episode} (score: {self.best_eval_score:.3f})")
    
    def _should_stop_early(self) -> bool:
        """Check early stopping criteria"""
        if len(self.eval_scores) < 3:
            return False
        
        # Check if no improvement for patience episodes
        if self.logger.no_improvement_count >= self.config.early_stopping_patience:
            return True
        
        # Check if performance threshold reached
        if self.best_eval_score < self.config.early_stopping_threshold:
            return False
        
        return False
    
    def _log_episode_results(self, episode: int, results: Dict[str, Any]):
        """Log episode results"""
        metrics = {
            'total_reward': results['total_reward'],
            'episode_length': results['episode_length'],
            'average_agent_reward': results['total_reward'] / len(self.agents)
        }
        
        # Add per-agent metrics
        for agent_id, reward in results['episode_rewards'].items():
            metrics[f'{agent_id}_reward'] = reward
        
        self.logger.log_episode(episode, metrics)
        
        # Update tracking
        for agent_id, reward in results['episode_rewards'].items():
            self.episode_rewards[agent_id].append(reward)
        self.episode_lengths.append(results['episode_length'])
    
    def _log_training_progress(self, episode: int):
        """Log training progress"""
        elapsed_time = time.time() - self.start_time
        episodes_per_second = episode / elapsed_time if elapsed_time > 0 else 0
        
        # Calculate recent performance
        recent_rewards = []
        for agent_rewards in self.episode_rewards.values():
            if len(agent_rewards) > 0:
                recent_rewards.extend(list(agent_rewards)[-100:])  # Last 100 episodes
        
        recent_mean = np.mean(recent_rewards) if recent_rewards else 0.0
        recent_std = np.std(recent_rewards) if recent_rewards else 0.0
        
        logger.info(f"Episode {episode}/{self.config.total_episodes}")
        logger.info(f"  Recent performance: {recent_mean:.3f} ± {recent_std:.3f}")
        logger.info(f"  Episodes/sec: {episodes_per_second:.2f}")
        logger.info(f"  Total steps: {self.total_steps}")
        logger.info(f"  Best eval score: {self.best_eval_score:.3f}")
        
        if self.curriculum_manager:
            progress = self.curriculum_manager.get_progress_info()
            logger.info(f"  Curriculum: {progress['current_stage']['name']} "
                       f"({progress['current_stage']['episodes_in_stage']} episodes)")
    
    async def _finalize_training(self):
        """Finalize training and save results"""
        logger.info("Finalizing training...")
        
        # Save final checkpoint
        await self._save_checkpoint(self.current_episode)
        
        # Save training logs
        self.logger.save_final_report()
        
        # Save curriculum completion report if available
        if self.curriculum_manager:
            curriculum_report = self.curriculum_manager.get_completion_report()
            report_path = Path(self.config.output_dir) / f"{self.config.experiment_name}_curriculum_report.json"
            with open(report_path, 'w') as f:
                json.dump(curriculum_report, f, indent=2, default=str)
        
        # Clean up resources
        for agent in self.agents.values():
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
        
        total_time = time.time() - self.start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        return {
            'config': self.config,
            'total_episodes': self.current_episode,
            'total_steps': self.total_steps,
            'best_eval_score': self.best_eval_score,
            'training_time_hours': (time.time() - self.start_time) / 3600,
            'final_performance': self.logger.get_performance_summary(),
            'curriculum_progress': (
                self.curriculum_manager.get_completion_report() 
                if self.curriculum_manager else None
            )
        }

# Factory functions
def create_trading_pipeline(
    num_agents: int = 4,
    total_episodes: int = 10000,
    use_curriculum: bool = True,
    **kwargs
) -> RLTrainingPipeline:
    """Create trading environment training pipeline"""
    config = TrainingConfig(
        environment_type="trading",
        num_agents=num_agents,
        total_episodes=total_episodes,
        use_curriculum=use_curriculum,
        **kwargs
    )
    return RLTrainingPipeline(config)

def create_supply_chain_pipeline(
    num_agents: int = 6,
    total_episodes: int = 15000,
    use_curriculum: bool = True,
    **kwargs
) -> RLTrainingPipeline:
    """Create supply chain environment training pipeline"""
    config = TrainingConfig(
        environment_type="supply_chain",
        num_agents=num_agents,
        total_episodes=total_episodes,
        use_curriculum=use_curriculum,
        **kwargs
    )
    return RLTrainingPipeline(config)

def create_resource_allocation_pipeline(
    num_agents: int = 8,
    total_episodes: int = 12000,
    use_curriculum: bool = True,
    **kwargs
) -> RLTrainingPipeline:
    """Create resource allocation environment training pipeline"""
    config = TrainingConfig(
        environment_type="resource_allocation",
        num_agents=num_agents,
        total_episodes=total_episodes,
        use_curriculum=use_curriculum,
        **kwargs
    )
    return RLTrainingPipeline(config)