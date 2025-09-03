"""
Comprehensive RL Training Demo
Demonstrates the complete reinforcement learning framework with multiple environments
"""

import asyncio
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from pathlib import Path
import argparse
import json
import sys
import os

# Add rl module to path
sys.path.append(str(Path(__file__).parent))

from rl.environments.trading_environment import TradingEnvironment, TradingAgent, MarketRegime
from rl.environments.supply_chain_environment import SupplyChainEnvironment
from rl.environments.resource_allocation_environment import ResourceAllocationEnvironment
from rl.environments.base_environment import AgentConfig, AgentRole
from rl.algorithms.ppo_agent import PPOAgent, PPOConfig
from rl.training.training_pipeline import RLTrainingPipeline, TrainingConfig
from rl.training.curriculum import (
    CurriculumManager, create_trading_curriculum, 
    create_supply_chain_curriculum, AdaptiveCurriculumScheduler
)
from data.synthetic_data_generator import SyntheticDataGenerator
from utils.observability.logging import get_logger

logger = get_logger(__name__)

class RLDemoRunner:
    """Comprehensive RL demo runner"""
    
    def __init__(self, demo_type: str = "trading", num_agents: int = 4, episodes: int = 2000):
        self.demo_type = demo_type
        self.num_agents = num_agents
        self.episodes = episodes
        self.output_dir = Path("rl_demo_results") / f"{demo_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing RL demo: {demo_type} with {num_agents} agents for {episodes} episodes")
        logger.info(f"Output directory: {self.output_dir}")
    
    async def run_demo(self) -> dict:
        """Run complete RL training demo"""
        logger.info("Starting comprehensive RL training demo...")
        
        # Step 1: Generate synthetic data
        synthetic_data = await self._generate_synthetic_data()
        
        # Step 2: Create environment
        environment = await self._create_environment(synthetic_data)
        
        # Step 3: Create and configure agents
        agents = await self._create_agents(environment)
        
        # Step 4: Set up curriculum learning
        curriculum = await self._setup_curriculum()
        
        # Step 5: Run training
        training_results = await self._run_training(environment, agents, curriculum)
        
        # Step 6: Evaluate trained agents
        evaluation_results = await self._evaluate_trained_agents(environment, agents)
        
        # Step 7: Generate visualizations and reports
        await self._generate_reports(training_results, evaluation_results, synthetic_data)
        
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'output_dir': str(self.output_dir)
        }
    
    async def _generate_synthetic_data(self) -> dict:
        """Generate synthetic data for training"""
        logger.info("Generating synthetic data...")
        
        generator = SyntheticDataGenerator(seed=42)
        
        if self.demo_type == "trading":
            # Generate comprehensive trading data
            synthetic_data = generator.generate_complete_dataset(save_to_file=False)
            
            # Add specialized trading scenarios
            high_volatility_data = self._generate_high_volatility_scenario(generator)
            crisis_data = self._generate_crisis_scenario(generator)
            
            synthetic_data['specialized_scenarios'] = {
                'high_volatility': high_volatility_data,
                'crisis': crisis_data
            }
            
        else:
            # Generate basic synthetic data for other environments
            synthetic_data = generator.generate_complete_dataset(save_to_file=False)
        
        # Save synthetic data
        data_file = self.output_dir / "synthetic_data.json"
        with open(data_file, 'w') as f:
            json.dump(synthetic_data, f, indent=2, default=str)
        
        logger.info(f"Generated and saved synthetic data to {data_file}")
        return synthetic_data
    
    def _generate_high_volatility_scenario(self, generator: SyntheticDataGenerator) -> dict:
        """Generate high volatility market scenario"""
        # Temporarily increase volatility
        original_market_conditions = generator.market_symbols.copy()
        
        # Generate high volatility trading events
        agents = generator.generate_agent_network(num_agents=self.num_agents)['agents']
        high_vol_events = generator.generate_trading_scenarios(agents, num_events=100)
        
        # Modify events to simulate high volatility
        for event in high_vol_events:
            if 'volatility' in event:
                event['volatility'] *= 2.0  # Double volatility
            event['market_condition'] = 'high_volatility'
        
        return {
            'events': high_vol_events,
            'description': 'High volatility market scenario for stress testing'
        }
    
    def _generate_crisis_scenario(self, generator: SyntheticDataGenerator) -> dict:
        """Generate market crisis scenario"""
        agents = generator.generate_agent_network(num_agents=self.num_agents)['agents']
        
        # Generate crisis events with high correlation and large price moves
        crisis_events = []
        for i in range(50):
            # Generate correlated negative events
            event = {
                'id': f'crisis_event_{i}',
                'type': 'sell',
                'timestamp': datetime.now().isoformat(),
                'market_condition': 'crisis',
                'correlation_factor': 0.9,  # High correlation between assets
                'volatility': 3.0,  # Very high volatility
                'liquidity_factor': 0.3  # Low liquidity
            }
            crisis_events.append(event)
        
        return {
            'events': crisis_events,
            'description': 'Market crisis scenario with high correlation and low liquidity'
        }
    
    async def _create_environment(self, synthetic_data: dict):
        """Create training environment"""
        logger.info(f"Creating {self.demo_type} environment...")
        
        # Create agent configurations
        agent_configs = [
            AgentConfig(
                agent_id=f"agent_{i}",
                role=AgentRole.EXECUTOR,
                initial_balance=100000.0,
                risk_tolerance=np.random.uniform(0.3, 0.8),
                learning_rate=0.001,
                capabilities=['trading', 'analysis', 'coordination']
            )
            for i in range(self.num_agents)
        ]
        
        if self.demo_type == "trading":
            # Extract market symbols from synthetic data
            market_data = synthetic_data.get('market_data', [])
            symbols = list(set([data['symbol'] for data in market_data[:20]]))  # First 20 symbols
            
            # Get initial prices
            initial_prices = {}
            for symbol in symbols:
                symbol_data = [d for d in market_data if d['symbol'] == symbol]
                if symbol_data:
                    initial_prices[symbol] = symbol_data[0]['price']
                else:
                    initial_prices[symbol] = np.random.uniform(100, 500)
            
            environment = TradingEnvironment(
                agent_configs=agent_configs,
                symbols=symbols,
                initial_prices=initial_prices,
                max_steps=1000,
                market_regime=MarketRegime.SIDEWAYS,
                transaction_cost=0.001,
                use_synthetic_data=True
            )
            
        elif self.demo_type == "supply_chain":
            environment = SupplyChainEnvironment(
                agent_configs=agent_configs,
                max_steps=1000,
                disruption_probability=0.1
            )
            
        elif self.demo_type == "resource_allocation":
            environment = ResourceAllocationEnvironment(
                agent_configs=agent_configs,
                max_steps=1000,
                allocation_mechanism="proportional",
                fairness_weight=0.3
            )
            
        else:
            raise ValueError(f"Unknown demo type: {self.demo_type}")
        
        logger.info(f"Created {type(environment).__name__} with {len(agent_configs)} agents")
        return environment
    
    async def _create_agents(self, environment) -> dict:
        """Create and configure RL agents"""
        logger.info("Creating RL agents...")
        
        # Get environment dimensions
        obs_dim = environment._get_observation_dimension()
        action_dim = environment._get_action_dimension()
        
        logger.info(f"Environment dimensions - Obs: {obs_dim}, Action: {action_dim}")
        
        # Create PPO configuration
        ppo_config = PPOConfig(
            hidden_dims=[256, 256, 128],
            learning_rate=3e-4,
            batch_size=256,
            ppo_epochs=4,
            epsilon_clip=0.2,
            entropy_coeff=0.01,
            value_loss_coeff=0.5,
            discount_factor=0.99,
            gae_lambda=0.95,
            buffer_size=10000,
            eval_frequency=500,
            checkpoint_dir=str(self.output_dir / "checkpoints")
        )
        
        # Create agents
        agents = {}
        for i in range(self.num_agents):
            agent_id = f"agent_{i}"
            
            agent = PPOAgent(
                agent_id=agent_id,
                state_dim=obs_dim,
                action_dim=action_dim,
                config=ppo_config,
                max_action=1.0
            )
            
            agents[agent_id] = agent
        
        logger.info(f"Created {len(agents)} PPO agents")
        
        # Log agent architecture details
        sample_agent = list(agents.values())[0]
        total_params = sum(p.numel() for network in sample_agent.networks.values() 
                          for p in network.parameters() if p.requires_grad)
        logger.info(f"Each agent has {total_params:,} trainable parameters")
        
        return agents
    
    async def _setup_curriculum(self):
        """Set up curriculum learning"""
        logger.info("Setting up curriculum learning...")
        
        if self.demo_type == "trading":
            curriculum_stages = create_trading_curriculum()
        elif self.demo_type == "supply_chain":
            curriculum_stages = create_supply_chain_curriculum()
        else:
            # Create basic curriculum for resource allocation
            from rl.training.curriculum import CurriculumStage, DifficultyLevel
            curriculum_stages = [
                CurriculumStage(
                    name="basic",
                    difficulty=DifficultyLevel.EASY,
                    parameters={'difficulty': 0.5},
                    success_criteria={'reward': 0.5},
                    min_episodes=200,
                    advancement_threshold=0.75
                ),
                CurriculumStage(
                    name="advanced",
                    difficulty=DifficultyLevel.HARD,
                    parameters={'difficulty': 1.0},
                    success_criteria={'reward': 0.8},
                    min_episodes=500,
                    advancement_threshold=0.8
                )
            ]
        
        # Use adaptive scheduler
        scheduler = AdaptiveCurriculumScheduler(adaptation_rate=0.1)
        
        curriculum = CurriculumManager(
            stages=curriculum_stages,
            scheduler=scheduler,
            save_path=str(self.output_dir / "curriculum_state.json")
        )
        
        logger.info(f"Created curriculum with {len(curriculum_stages)} stages")
        return curriculum
    
    async def _run_training(self, environment, agents: dict, curriculum) -> dict:
        """Run RL training"""
        logger.info("Starting RL training...")
        
        # Create training configuration
        training_config = TrainingConfig(
            total_episodes=self.episodes,
            max_steps_per_episode=1000,
            eval_frequency=200,
            eval_episodes=20,
            environment_type=self.demo_type,
            agent_type="ppo",
            num_agents=self.num_agents,
            use_curriculum=True,
            use_synthetic_data=True,
            synthetic_data_refresh_frequency=500,
            log_frequency=50,
            checkpoint_frequency=500,
            experiment_name=f"{self.demo_type}_demo",
            output_dir=str(self.output_dir),
            tags=[self.demo_type, "demo", "curriculum_learning"]
        )
        
        # Create training pipeline
        pipeline = RLTrainingPipeline(
            config=training_config,
            environment=environment,
            agents=agents
        )
        
        # Inject curriculum manager
        pipeline.curriculum_manager = curriculum
        
        # Run training
        results = await pipeline.train()
        
        logger.info(f"Training completed after {results['total_episodes']} episodes")
        logger.info(f"Best evaluation score: {results['best_eval_score']:.3f}")
        
        return results
    
    async def _evaluate_trained_agents(self, environment, agents: dict) -> dict:
        """Evaluate trained agents on various scenarios"""
        logger.info("Evaluating trained agents...")
        
        evaluation_results = {}
        
        # Standard evaluation
        logger.info("Running standard evaluation...")
        standard_results = await self._run_evaluation_episodes(
            environment, agents, num_episodes=50, scenario="standard"
        )
        evaluation_results['standard'] = standard_results
        
        # Stress test evaluation (if trading environment)
        if self.demo_type == "trading" and hasattr(environment, 'market_regime'):
            logger.info("Running stress test evaluation...")
            
            # High volatility test
            original_regime = environment.market_regime
            environment.market_regime = MarketRegime.VOLATILE
            stress_results = await self._run_evaluation_episodes(
                environment, agents, num_episodes=30, scenario="high_volatility"
            )
            evaluation_results['stress_test'] = stress_results
            
            # Crisis test
            environment.market_regime = MarketRegime.CRISIS
            crisis_results = await self._run_evaluation_episodes(
                environment, agents, num_episodes=20, scenario="crisis"
            )
            evaluation_results['crisis_test'] = crisis_results
            
            # Restore original regime
            environment.market_regime = original_regime
        
        # Generalization test (different environment settings)
        logger.info("Running generalization test...")
        generalization_results = await self._test_generalization(environment, agents)
        evaluation_results['generalization'] = generalization_results
        
        return evaluation_results
    
    async def _run_evaluation_episodes(self, environment, agents: dict, num_episodes: int, scenario: str) -> dict:
        """Run evaluation episodes"""
        episode_rewards = {agent_id: [] for agent_id in agents.keys()}
        episode_lengths = []
        episode_details = []
        
        for episode in range(num_episodes):
            observations, _ = environment.reset()
            
            episode_reward_sum = {agent_id: 0.0 for agent_id in agents.keys()}
            episode_length = 0
            done = {agent_id: False for agent_id in agents.keys()}
            
            while not all(done.values()) and episode_length < 1000:
                actions = {}
                
                # Select actions (evaluation mode)
                for agent_id, agent in agents.items():
                    if not done[agent_id] and agent_id in observations:
                        action = agent.select_action(observations[agent_id], evaluation=True)
                        actions[agent_id] = action
                
                # Execute actions
                next_observations, rewards, terminated, truncated, info = environment.step(actions)
                
                # Update rewards and done status
                for agent_id in agents.keys():
                    if agent_id in rewards:
                        episode_reward_sum[agent_id] += rewards[agent_id]
                    if agent_id in terminated:
                        done[agent_id] = terminated[agent_id] or truncated.get(agent_id, False)
                
                observations = next_observations
                episode_length += 1
            
            # Store results
            for agent_id, reward in episode_reward_sum.items():
                episode_rewards[agent_id].append(reward)
            episode_lengths.append(episode_length)
            
            # Store episode details
            episode_details.append({
                'episode': episode,
                'rewards': episode_reward_sum,
                'length': episode_length,
                'total_reward': sum(episode_reward_sum.values())
            })
        
        # Calculate statistics
        results = {
            'scenario': scenario,
            'num_episodes': num_episodes,
            'mean_rewards': {agent_id: np.mean(rewards) for agent_id, rewards in episode_rewards.items()},
            'std_rewards': {agent_id: np.std(rewards) for agent_id, rewards in episode_rewards.items()},
            'mean_episode_length': np.mean(episode_lengths),
            'total_mean_reward': np.mean([np.mean(rewards) for rewards in episode_rewards.values()]),
            'success_rate': len([r for r in episode_lengths if r < 1000]) / num_episodes,  # Episodes that didn't timeout
            'episode_details': episode_details
        }
        
        return results
    
    async def _test_generalization(self, environment, agents: dict) -> dict:
        """Test agent generalization to different environment settings"""
        generalization_results = {}
        
        if self.demo_type == "trading":
            # Test different market conditions
            test_conditions = [
                ('low_volatility', {'volatility': 0.1}),
                ('medium_volatility', {'volatility': 0.2}),
                ('high_volatility', {'volatility': 0.4})
            ]
            
            for condition_name, params in test_conditions:
                # Modify environment parameters
                original_volatility = getattr(environment, 'volatility', 0.2)
                environment.volatility = params['volatility']
                
                # Run evaluation
                results = await self._run_evaluation_episodes(
                    environment, agents, num_episodes=20, scenario=condition_name
                )
                generalization_results[condition_name] = results
                
                # Restore original settings
                environment.volatility = original_volatility
        
        return generalization_results
    
    async def _generate_reports(self, training_results: dict, evaluation_results: dict, synthetic_data: dict):
        """Generate comprehensive reports and visualizations"""
        logger.info("Generating reports and visualizations...")
        
        # Create reports directory
        reports_dir = self.output_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Generate training report
        await self._generate_training_report(training_results, reports_dir)
        
        # Generate evaluation report
        await self._generate_evaluation_report(evaluation_results, reports_dir)
        
        # Generate performance visualizations
        await self._generate_visualizations(training_results, evaluation_results, reports_dir)
        
        # Generate synthetic data analysis
        await self._generate_synthetic_data_analysis(synthetic_data, reports_dir)
        
        # Generate summary report
        await self._generate_summary_report(training_results, evaluation_results, reports_dir)
        
        logger.info(f"Reports generated in {reports_dir}")
    
    async def _generate_training_report(self, results: dict, output_dir: Path):
        """Generate detailed training report"""
        report = {
            'experiment_info': {
                'demo_type': self.demo_type,
                'num_agents': self.num_agents,
                'total_episodes': self.episodes,
                'timestamp': datetime.now().isoformat()
            },
            'training_summary': {
                'total_episodes_completed': results.get('total_episodes', 0),
                'total_steps': results.get('total_steps', 0),
                'best_evaluation_score': results.get('best_eval_score', 0),
                'training_time_hours': results.get('training_time_hours', 0)
            },
            'performance_analysis': results.get('final_performance', {}),
            'curriculum_progress': results.get('curriculum_progress', {})
        }
        
        report_file = output_dir / "training_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    async def _generate_evaluation_report(self, results: dict, output_dir: Path):
        """Generate evaluation report"""
        report = {
            'evaluation_summary': {
                'scenarios_tested': list(results.keys()),
                'total_evaluation_episodes': sum(
                    r.get('num_episodes', 0) for r in results.values()
                )
            },
            'scenario_results': results
        }
        
        report_file = output_dir / "evaluation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    async def _generate_visualizations(self, training_results: dict, evaluation_results: dict, output_dir: Path):
        """Generate performance visualizations"""
        try:
            import matplotlib.pyplot as plt
            
            # Training performance plot
            if 'final_performance' in training_results:
                perf = training_results['final_performance']
                if 'episode_rewards' in perf:
                    plt.figure(figsize=(12, 8))
                    
                    rewards = perf['episode_rewards']
                    if rewards:
                        plt.subplot(2, 2, 1)
                        plt.plot(rewards)
                        plt.title('Training Rewards Over Time')
                        plt.xlabel('Episode')
                        plt.ylabel('Reward')
                        
                        plt.subplot(2, 2, 2)
                        # Moving average
                        window = min(100, len(rewards) // 10)
                        if window > 1:
                            moving_avg = pd.Series(rewards).rolling(window=window).mean()
                            plt.plot(moving_avg)
                            plt.title(f'Training Rewards (Moving Average, window={window})')
                            plt.xlabel('Episode')
                            plt.ylabel('Average Reward')
                    
                    # Evaluation performance comparison
                    if evaluation_results:
                        plt.subplot(2, 2, 3)
                        scenarios = []
                        mean_rewards = []
                        
                        for scenario, results in evaluation_results.items():
                            if 'total_mean_reward' in results:
                                scenarios.append(scenario)
                                mean_rewards.append(results['total_mean_reward'])
                        
                        if scenarios and mean_rewards:
                            plt.bar(scenarios, mean_rewards)
                            plt.title('Evaluation Performance by Scenario')
                            plt.ylabel('Mean Reward')
                            plt.xticks(rotation=45)
                    
                    plt.tight_layout()
                    plt.savefig(output_dir / "performance_plots.png", dpi=300, bbox_inches='tight')
                    plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available for visualization generation")
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
    
    async def _generate_synthetic_data_analysis(self, synthetic_data: dict, output_dir: Path):
        """Generate synthetic data analysis report"""
        analysis = {
            'data_overview': {
                'agents_count': len(synthetic_data.get('agent_network', {}).get('agents', [])),
                'connections_count': len(synthetic_data.get('agent_network', {}).get('connections', [])),
                'market_data_points': len(synthetic_data.get('market_data', [])),
                'trading_events': len(synthetic_data.get('trading_events', []))
            }
        }
        
        # Agent network analysis
        if 'agent_network' in synthetic_data:
            agents = synthetic_data['agent_network'].get('agents', [])
            if agents:
                agent_types = [agent.get('type', 'unknown') for agent in agents]
                type_counts = {t: agent_types.count(t) for t in set(agent_types)}
                analysis['agent_distribution'] = type_counts
                
                # Performance distribution
                throughputs = [agent.get('performance', {}).get('throughput', 0) for agent in agents]
                accuracies = [agent.get('performance', {}).get('accuracy', 0) for agent in agents]
                
                analysis['performance_stats'] = {
                    'throughput': {
                        'mean': np.mean(throughputs),
                        'std': np.std(throughputs),
                        'min': np.min(throughputs),
                        'max': np.max(throughputs)
                    },
                    'accuracy': {
                        'mean': np.mean(accuracies),
                        'std': np.std(accuracies),
                        'min': np.min(accuracies),
                        'max': np.max(accuracies)
                    }
                }
        
        report_file = output_dir / "synthetic_data_analysis.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
    
    async def _generate_summary_report(self, training_results: dict, evaluation_results: dict, output_dir: Path):
        """Generate executive summary report"""
        summary = {
            'experiment_overview': {
                'demo_type': self.demo_type,
                'num_agents': self.num_agents,
                'episodes_planned': self.episodes,
                'episodes_completed': training_results.get('total_episodes', 0),
                'success': training_results.get('total_episodes', 0) >= self.episodes * 0.8  # 80% completion threshold
            },
            'key_findings': {
                'best_performance': training_results.get('best_eval_score', 0),
                'training_stability': self._assess_training_stability(training_results),
                'generalization_ability': self._assess_generalization(evaluation_results),
                'curriculum_effectiveness': self._assess_curriculum_effectiveness(training_results)
            },
            'recommendations': self._generate_recommendations(training_results, evaluation_results),
            'technical_details': {
                'total_parameters': self._estimate_total_parameters(),
                'training_time_hours': training_results.get('training_time_hours', 0),
                'computational_efficiency': self._calculate_computational_efficiency(training_results)
            }
        }
        
        report_file = output_dir / "executive_summary.json"
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Also create a markdown version
        await self._create_markdown_summary(summary, output_dir)
    
    def _assess_training_stability(self, training_results: dict) -> str:
        """Assess training stability"""
        if 'final_performance' not in training_results:
            return "unknown"
        
        perf = training_results['final_performance']
        if 'training_performance' in perf:
            trend = perf['training_performance'].get('reward_trend', 0)
            if trend > 0.01:
                return "improving"
            elif trend < -0.01:
                return "declining"
            else:
                return "stable"
        
        return "unknown"
    
    def _assess_generalization(self, evaluation_results: dict) -> str:
        """Assess generalization ability"""
        if not evaluation_results:
            return "unknown"
        
        if 'generalization' in evaluation_results:
            gen_results = evaluation_results['generalization']
            if gen_results:
                # Compare performance across different conditions
                performances = [r.get('total_mean_reward', 0) for r in gen_results.values()]
                if performances:
                    std_dev = np.std(performances)
                    if std_dev < 0.1:
                        return "excellent"
                    elif std_dev < 0.3:
                        return "good"
                    else:
                        return "limited"
        
        return "unknown"
    
    def _assess_curriculum_effectiveness(self, training_results: dict) -> str:
        """Assess curriculum learning effectiveness"""
        if 'curriculum_progress' not in training_results:
            return "not_used"
        
        progress = training_results['curriculum_progress']
        if progress:
            completion = progress.get('curriculum_summary', {}).get('completion_percentage', 0)
            if completion > 80:
                return "highly_effective"
            elif completion > 50:
                return "moderately_effective"
            else:
                return "limited_effectiveness"
        
        return "unknown"
    
    def _generate_recommendations(self, training_results: dict, evaluation_results: dict) -> list:
        """Generate recommendations based on results"""
        recommendations = []
        
        # Training performance recommendations
        if training_results.get('best_eval_score', 0) < 0.5:
            recommendations.append({
                'category': 'performance',
                'recommendation': 'Consider adjusting hyperparameters or increasing training duration',
                'priority': 'high'
            })
        
        # Curriculum recommendations
        if 'curriculum_progress' in training_results:
            progress = training_results['curriculum_progress']
            if progress.get('curriculum_summary', {}).get('completion_percentage', 0) < 50:
                recommendations.append({
                    'category': 'curriculum',
                    'recommendation': 'Adjust curriculum difficulty progression for better learning',
                    'priority': 'medium'
                })
        
        # Generalization recommendations
        if self._assess_generalization(evaluation_results) == "limited":
            recommendations.append({
                'category': 'generalization',
                'recommendation': 'Increase environment diversity and regularization techniques',
                'priority': 'medium'
            })
        
        return recommendations
    
    def _estimate_total_parameters(self) -> int:
        """Estimate total parameters in the system"""
        # Rough estimation based on typical PPO architecture
        return self.num_agents * (256 * 256 + 256 * 128 + 128 * 4)  # Simplified calculation
    
    def _calculate_computational_efficiency(self, training_results: dict) -> dict:
        """Calculate computational efficiency metrics"""
        training_time = training_results.get('training_time_hours', 1)
        episodes_completed = training_results.get('total_episodes', 1)
        
        return {
            'episodes_per_hour': episodes_completed / training_time,
            'hours_per_1k_episodes': (training_time * 1000) / episodes_completed,
            'estimated_parameters': self._estimate_total_parameters()
        }
    
    async def _create_markdown_summary(self, summary: dict, output_dir: Path):
        """Create markdown version of summary"""
        md_content = f"""# RL Training Demo Summary

## Experiment Overview
- **Demo Type**: {summary['experiment_overview']['demo_type']}
- **Number of Agents**: {summary['experiment_overview']['num_agents']}
- **Episodes Completed**: {summary['experiment_overview']['episodes_completed']}/{summary['experiment_overview']['episodes_planned']}
- **Success**: {'✅' if summary['experiment_overview']['success'] else '❌'}

## Key Findings
- **Best Performance**: {summary['key_findings']['best_performance']:.3f}
- **Training Stability**: {summary['key_findings']['training_stability']}
- **Generalization**: {summary['key_findings']['generalization_ability']}
- **Curriculum Effectiveness**: {summary['key_findings']['curriculum_effectiveness']}

## Technical Details
- **Total Parameters**: {summary['technical_details']['total_parameters']:,}
- **Training Time**: {summary['technical_details']['training_time_hours']:.2f} hours
- **Efficiency**: {summary['technical_details']['computational_efficiency']['episodes_per_hour']:.1f} episodes/hour

## Recommendations
"""
        
        for i, rec in enumerate(summary['recommendations'], 1):
            md_content += f"{i}. **{rec['category'].title()}** ({rec['priority']} priority): {rec['recommendation']}\n"
        
        md_file = output_dir / "summary.md"
        with open(md_file, 'w') as f:
            f.write(md_content)

async def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description="RL Training Demo")
    parser.add_argument("--demo_type", choices=["trading", "supply_chain", "resource_allocation"], 
                       default="trading", help="Type of demo to run")
    parser.add_argument("--num_agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--episodes", type=int, default=2000, help="Number of training episodes")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = RLDemoRunner(
        demo_type=args.demo_type,
        num_agents=args.num_agents,
        episodes=args.episodes
    )
    
    try:
        results = await demo.run_demo()
        
        print(f"\n{'='*60}")
        print("RL TRAINING DEMO COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"Demo Type: {args.demo_type}")
        print(f"Number of Agents: {args.num_agents}")
        print(f"Episodes: {args.episodes}")
        print(f"Results saved to: {results['output_dir']}")
        
        if 'training_results' in results:
            tr = results['training_results']
            print(f"Best Performance: {tr.get('best_eval_score', 0):.3f}")
            print(f"Training Time: {tr.get('training_time_hours', 0):.2f} hours")
        
        print(f"\nCheck the results directory for detailed reports and visualizations:")
        print(f"{results['output_dir']}")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(main())