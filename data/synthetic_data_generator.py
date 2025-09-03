"""
Synthetic Data Generator for AI Agents Visualization
Generates realistic agent behavior, performance metrics, and trading scenarios
"""

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import math
import os
from pathlib import Path

@dataclass
class AgentBehaviorProfile:
    """Defines behavioral characteristics for agent types"""
    agent_type: str
    base_throughput: Tuple[float, float]  # (min, max)
    base_accuracy: Tuple[float, float]
    response_time_range: Tuple[float, float]
    utilization_patterns: List[float]  # Hourly utilization pattern (24 values)
    error_probability: float
    learning_rate: float
    collaboration_preference: float
    specialization_domains: List[str]

class SyntheticDataGenerator:
    """Generate comprehensive synthetic data for agent visualization"""
    
    def __init__(self, seed: int = 42):
        self.random = random.Random(seed)
        self.np_random = np.random.RandomState(seed)
        self.agent_profiles = self._create_agent_profiles()
        self.market_symbols = [
            'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'BRK.A',
            'JPM', 'JNJ', 'WMT', 'PG', 'UNH', 'V', 'MA', 'HD', 'DIS', 'PYPL'
        ]
        
    def _create_agent_profiles(self) -> Dict[str, AgentBehaviorProfile]:
        """Create realistic behavioral profiles for different agent types"""
        return {
            'coordinating': AgentBehaviorProfile(
                agent_type='coordinating',
                base_throughput=(80, 150),
                base_accuracy=(92, 98),
                response_time_range=(150, 350),
                utilization_patterns=[
                    30, 25, 20, 18, 20, 35, 55, 75, 85, 90, 88, 85,
                    82, 85, 88, 90, 92, 90, 85, 75, 65, 50, 40, 35
                ],
                error_probability=0.02,
                learning_rate=0.05,
                collaboration_preference=0.9,
                specialization_domains=['coordination', 'task_delegation', 'resource_management']
            ),
            'processing': AgentBehaviorProfile(
                agent_type='processing',
                base_throughput=(100, 200),
                base_accuracy=(88, 96),
                response_time_range=(50, 200),
                utilization_patterns=[
                    40, 35, 30, 25, 30, 45, 70, 85, 95, 98, 95, 90,
                    88, 90, 95, 98, 95, 90, 85, 70, 60, 50, 45, 42
                ],
                error_probability=0.05,
                learning_rate=0.08,
                collaboration_preference=0.6,
                specialization_domains=['data_processing', 'computation', 'transformation']
            ),
            'analysis': AgentBehaviorProfile(
                agent_type='analysis',
                base_throughput=(60, 120),
                base_accuracy=(90, 99),
                response_time_range=(200, 800),
                utilization_patterns=[
                    20, 15, 12, 10, 15, 25, 40, 60, 75, 85, 90, 92,
                    90, 88, 85, 80, 78, 75, 70, 60, 45, 35, 28, 22
                ],
                error_probability=0.03,
                learning_rate=0.03,
                collaboration_preference=0.4,
                specialization_domains=['pattern_recognition', 'prediction', 'anomaly_detection']
            ),
            'integration': AgentBehaviorProfile(
                agent_type='integration',
                base_throughput=(70, 130),
                base_accuracy=(85, 94),
                response_time_range=(100, 400),
                utilization_patterns=[
                    35, 30, 25, 22, 25, 40, 60, 78, 88, 92, 90, 85,
                    80, 82, 85, 88, 90, 85, 80, 70, 60, 50, 42, 38
                ],
                error_probability=0.07,
                learning_rate=0.06,
                collaboration_preference=0.8,
                specialization_domains=['system_integration', 'api_management', 'data_flow']
            )
        }
    
    def generate_agent_network(self, num_agents: int = 12) -> Dict[str, Any]:
        """Generate a realistic agent network with connections"""
        agents = []
        agent_types = list(self.agent_profiles.keys())
        
        # Create agents with realistic distributions
        type_distribution = [3, 4, 3, 2]  # coordinating, processing, analysis, integration
        
        agent_id = 0
        for type_idx, agent_type in enumerate(agent_types):
            count = type_distribution[type_idx] if type_idx < len(type_distribution) else 1
            
            for i in range(count):
                agent_id += 1
                profile = self.agent_profiles[agent_type]
                
                # Position in 3D space with some clustering by type
                cluster_center = {
                    'coordinating': (0, 2, 0),
                    'processing': (-5, 1, -5),
                    'analysis': (5, 1, 5),
                    'integration': (0, 1, -8)
                }[agent_type]
                
                position = {
                    'x': cluster_center[0] + self.np_random.normal(0, 3),
                    'y': cluster_center[1] + self.np_random.uniform(0, 2),
                    'z': cluster_center[2] + self.np_random.normal(0, 3)
                }
                
                # Generate performance based on profile
                current_hour = datetime.now().hour
                base_utilization = profile.utilization_patterns[current_hour]
                
                agent = {
                    'id': f'agent-{agent_id:02d}',
                    'name': f'{agent_type.title()}-Agent-{i+1:02d}',
                    'type': agent_type,
                    'status': self.random.choices(
                        ['active', 'idle', 'error', 'offline'],
                        weights=[75, 20, 4, 1]
                    )[0],
                    'position': position,
                    'performance': {
                        'throughput': self.np_random.uniform(*profile.base_throughput),
                        'accuracy': self.np_random.uniform(*profile.base_accuracy),
                        'responseTime': self.np_random.uniform(*profile.response_time_range),
                        'utilization': max(10, min(100, base_utilization + self.np_random.normal(0, 10)))
                    },
                    'connections': [],
                    'lastActivity': (datetime.now() - timedelta(seconds=self.random.randint(0, 300))).isoformat(),
                    'metadata': {
                        'specialization': self.random.choice(profile.specialization_domains),
                        'version': f'1.{self.random.randint(0, 9)}.{self.random.randint(0, 9)}',
                        'capabilities': self.random.sample(
                            profile.specialization_domains + ['monitoring', 'optimization', 'learning'],
                            k=self.random.randint(3, 6)
                        ),
                        'learning_rate': profile.learning_rate,
                        'collaboration_score': profile.collaboration_preference
                    }
                }
                agents.append(agent)
        
        # Generate realistic connections
        connections = self._generate_connections(agents)
        
        # Calculate network metrics
        active_agents = [a for a in agents if a['status'] == 'active']
        metrics = {
            'totalAgents': len(agents),
            'activeAgents': len(active_agents),
            'totalConnections': len(connections),
            'averageLatency': np.mean([c['latency'] for c in connections]) if connections else 0,
            'networkThroughput': sum(a['performance']['throughput'] for a in active_agents),
            'overallAccuracy': np.mean([a['performance']['accuracy'] for a in agents]),
            'systemHealth': len(active_agents) / len(agents) if agents else 0,
            'loadDistribution': [a['performance']['utilization'] for a in agents]
        }
        
        return {
            'agents': agents,
            'connections': connections,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_connections(self, agents: List[Dict]) -> List[Dict]:
        """Generate realistic connections between agents"""
        connections = []
        connection_types = ['data_flow', 'coordination', 'feedback', 'error']
        
        # Create a connection matrix based on agent types and collaboration preferences
        for i, source_agent in enumerate(agents):
            source_profile = self.agent_profiles[source_agent['type']]
            
            # Number of connections based on collaboration preference
            max_connections = max(1, int(source_profile.collaboration_preference * len(agents) * 0.4))
            num_connections = self.random.randint(1, max_connections)
            
            # Select target agents based on compatibility
            potential_targets = []
            for j, target_agent in enumerate(agents):
                if i != j:
                    # Calculate compatibility score
                    compatibility = self._calculate_agent_compatibility(source_agent, target_agent)
                    potential_targets.append((j, target_agent, compatibility))
            
            # Sort by compatibility and select top candidates
            potential_targets.sort(key=lambda x: x[2], reverse=True)
            selected_targets = potential_targets[:num_connections]
            
            for _, target_agent, compatibility in selected_targets:
                connection_id = f"{source_agent['id']}-{target_agent['id']}"
                
                # Avoid duplicate connections
                if not any(c['id'] == connection_id for c in connections):
                    connection_type = self._select_connection_type(source_agent, target_agent)
                    
                    connection = {
                        'id': connection_id,
                        'sourceId': source_agent['id'],
                        'targetId': target_agent['id'],
                        'type': connection_type,
                        'strength': compatibility,
                        'latency': self.np_random.uniform(10, 200),
                        'bandwidth': self.np_random.uniform(100, 2000),
                        'messages': self.random.randint(10, 1000),
                        'direction': self.random.choice(['bidirectional', 'source_to_target', 'target_to_source'])
                    }
                    connections.append(connection)
                    
                    # Update agent connections
                    source_agent['connections'].append(target_agent['id'])
        
        return connections
    
    def _calculate_agent_compatibility(self, agent1: Dict, agent2: Dict) -> float:
        """Calculate compatibility score between two agents"""
        type_compatibility = {
            ('coordinating', 'processing'): 0.9,
            ('coordinating', 'analysis'): 0.8,
            ('coordinating', 'integration'): 0.85,
            ('processing', 'analysis'): 0.7,
            ('processing', 'integration'): 0.8,
            ('analysis', 'integration'): 0.6
        }
        
        # Base compatibility from types
        type_pair = tuple(sorted([agent1['type'], agent2['type']]))
        base_compatibility = type_compatibility.get(type_pair, 0.5)
        
        # Distance factor (closer agents are more likely to connect)
        pos1, pos2 = agent1['position'], agent2['position']
        distance = math.sqrt(
            (pos1['x'] - pos2['x'])**2 + 
            (pos1['y'] - pos2['y'])**2 + 
            (pos1['z'] - pos2['z'])**2
        )
        distance_factor = max(0.1, 1.0 - (distance / 20.0))
        
        # Performance compatibility
        perf1, perf2 = agent1['performance'], agent2['performance']
        perf_diff = abs(perf1['accuracy'] - perf2['accuracy']) / 100.0
        perf_factor = max(0.3, 1.0 - perf_diff)
        
        # Collaboration preference
        collab1 = agent1['metadata'].get('collaboration_score', 0.5)
        collab2 = agent2['metadata'].get('collaboration_score', 0.5)
        collab_factor = (collab1 + collab2) / 2.0
        
        # Combine factors
        final_score = (base_compatibility * 0.4 + 
                      distance_factor * 0.2 + 
                      perf_factor * 0.2 + 
                      collab_factor * 0.2)
        
        return min(1.0, max(0.1, final_score + self.np_random.normal(0, 0.1)))
    
    def _select_connection_type(self, source_agent: Dict, target_agent: Dict) -> str:
        """Select appropriate connection type based on agent types"""
        type_connections = {
            ('coordinating', 'processing'): ['coordination', 'data_flow'],
            ('coordinating', 'analysis'): ['coordination', 'feedback'],
            ('coordinating', 'integration'): ['coordination', 'data_flow'],
            ('processing', 'analysis'): ['data_flow', 'feedback'],
            ('processing', 'integration'): ['data_flow'],
            ('analysis', 'integration'): ['feedback', 'data_flow']
        }
        
        type_pair = tuple(sorted([source_agent['type'], target_agent['type']]))
        possible_types = type_connections.get(type_pair, ['data_flow'])
        
        return self.random.choice(possible_types)
    
    def generate_performance_timeline(self, agents: List[Dict], duration_hours: int = 24) -> List[Dict]:
        """Generate realistic performance metrics over time"""
        metrics = []
        start_time = datetime.now() - timedelta(hours=duration_hours)
        
        for hour in range(duration_hours * 60):  # Generate per minute
            current_time = start_time + timedelta(minutes=hour)
            current_hour = current_time.hour
            
            for agent in agents:
                profile = self.agent_profiles[agent['type']]
                base_utilization = profile.utilization_patterns[current_hour]
                
                # Simulate various metrics with realistic correlations
                utilization = max(5, min(100, base_utilization + self.np_random.normal(0, 8)))
                
                # Throughput correlates with utilization but has diminishing returns
                utilization_factor = min(1.0, utilization / 80.0)
                throughput = (profile.base_throughput[0] + 
                             (profile.base_throughput[1] - profile.base_throughput[0]) * utilization_factor +
                             self.np_random.normal(0, 5))
                
                # Accuracy slightly decreases under very high utilization
                accuracy_penalty = max(0, (utilization - 85) * 0.1) if utilization > 85 else 0
                accuracy = (profile.base_accuracy[0] + 
                           (profile.base_accuracy[1] - profile.base_accuracy[0]) * self.np_random.uniform(0.8, 1.0) -
                           accuracy_penalty + self.np_random.normal(0, 0.5))
                
                # Response time increases with utilization
                response_multiplier = 1.0 + (utilization / 100.0) * 0.5
                response_time = (profile.response_time_range[0] * response_multiplier + 
                               self.np_random.exponential(profile.response_time_range[1] * 0.3))
                
                # Generate metrics for each measurement type
                metric_types = [
                    ('throughput', throughput, 'ops/sec', 'performance'),
                    ('accuracy', accuracy, 'percentage', 'accuracy'),
                    ('response_time', response_time, 'milliseconds', 'performance'),
                    ('utilization', utilization, 'percentage', 'efficiency')
                ]
                
                for metric_name, value, unit, category in metric_types:
                    if self.random.random() < 0.8:  # 80% chance of metric generation
                        metrics.append({
                            'timestamp': current_time.isoformat(),
                            'agentId': agent['id'],
                            'metric': metric_name,
                            'value': max(0, value),
                            'unit': unit,
                            'category': category
                        })
        
        return metrics
    
    def generate_trading_scenarios(self, agents: List[Dict], num_events: int = 200) -> List[Dict]:
        """Generate realistic trading events and scenarios"""
        events = []
        start_time = datetime.now() - timedelta(hours=8)  # 8 hours of trading history
        
        # Market conditions that affect trading patterns
        market_conditions = [
            {'condition': 'bullish', 'probability': 0.3, 'volatility': 0.8},
            {'condition': 'bearish', 'probability': 0.25, 'volatility': 1.2},
            {'condition': 'sideways', 'probability': 0.35, 'volatility': 0.6},
            {'condition': 'high_volatility', 'probability': 0.1, 'volatility': 2.0}
        ]
        
        active_agents = [a for a in agents if a['status'] == 'active']
        
        for i in range(num_events):
            # Select random time within trading period
            event_time = start_time + timedelta(
                seconds=self.random.randint(0, int(8 * 3600))
            )
            
            # Select agent (analysis agents more likely to generate events)
            agent_weights = []
            for agent in active_agents:
                weight = 3 if agent['type'] == 'analysis' else 2 if agent['type'] == 'coordinating' else 1
                agent_weights.append(weight)
            
            agent = self.random.choices(active_agents, weights=agent_weights)[0]
            
            # Select market condition
            market_condition = self.random.choices(
                market_conditions,
                weights=[c['probability'] for c in market_conditions]
            )[0]
            
            # Event type probabilities based on agent type and market conditions
            event_type_probs = {
                'analysis': {'analysis': 0.4, 'decision': 0.3, 'buy': 0.15, 'sell': 0.15},
                'coordinating': {'decision': 0.4, 'analysis': 0.2, 'buy': 0.2, 'sell': 0.2},
                'processing': {'execution': 0.4, 'buy': 0.3, 'sell': 0.3},
                'integration': {'analysis': 0.5, 'decision': 0.3, 'execution': 0.2}
            }
            
            type_probs = event_type_probs.get(agent['type'], event_type_probs['analysis'])
            event_type = self.random.choices(
                list(type_probs.keys()),
                weights=list(type_probs.values())
            )[0]
            
            # Generate event details
            symbol = self.random.choice(self.market_symbols)
            base_price = self.random.uniform(50, 500)
            
            # Confidence based on agent accuracy and market volatility
            base_confidence = agent['performance']['accuracy'] / 100.0
            volatility_factor = 1.0 / market_condition['volatility']
            confidence = min(0.95, max(0.3, base_confidence * volatility_factor + self.np_random.normal(0, 0.1)))
            
            # Generate profit/loss for buy/sell events
            profit = None
            if event_type in ['buy', 'sell']:
                # Profit influenced by confidence and market conditions
                expected_return = (confidence - 0.5) * 0.2  # -10% to +10% expected
                volatility_adjustment = market_condition['volatility'] * self.np_random.normal(0, 0.1)
                return_rate = expected_return + volatility_adjustment
                
                quantity = self.random.randint(10, 1000)
                profit = quantity * base_price * return_rate
            
            # Outcome probabilities based on confidence
            outcome_probs = [
                max(0.1, confidence - 0.1),  # success
                min(0.3, 0.4 - confidence),  # failure
                0.3  # pending
            ]
            outcome_probs = [p / sum(outcome_probs) for p in outcome_probs]  # normalize
            
            outcome = self.random.choices(['success', 'failure', 'pending'], weights=outcome_probs)[0]
            
            # Generate reasoning based on market conditions and agent type
            reasoning_templates = {
                'bullish': [
                    "Strong upward momentum detected in {symbol}",
                    "Positive market sentiment analysis for {symbol}",
                    "Technical indicators suggest bullish trend for {symbol}"
                ],
                'bearish': [
                    "Downward pressure identified in {symbol}",
                    "Negative sentiment analysis indicates decline for {symbol}",
                    "Technical analysis suggests bearish pattern for {symbol}"
                ],
                'sideways': [
                    "Range-bound trading pattern detected for {symbol}",
                    "Consolidation phase identified in {symbol}",
                    "Neutral market conditions for {symbol}"
                ],
                'high_volatility': [
                    "High volatility opportunity in {symbol}",
                    "Significant price movement expected for {symbol}",
                    "Market uncertainty creates opportunity in {symbol}"
                ]
            }
            
            reasoning = self.random.choice(
                reasoning_templates[market_condition['condition']]
            ).format(symbol=symbol)
            
            event = {
                'id': f'event-{int(event_time.timestamp() * 1000)}-{self.random.randint(1000, 9999)}',
                'timestamp': event_time.isoformat(),
                'type': event_type,
                'agentId': agent['id'],
                'symbol': symbol if event_type in ['buy', 'sell', 'analysis'] else None,
                'quantity': self.random.randint(10, 1000) if event_type in ['buy', 'sell'] else None,
                'price': base_price if event_type in ['buy', 'sell'] else None,
                'confidence': confidence,
                'reasoning': reasoning,
                'outcome': outcome,
                'profit': profit,
                'marketCondition': market_condition['condition'],
                'volatility': market_condition['volatility']
            }
            
            events.append(event)
        
        # Sort by timestamp
        events.sort(key=lambda x: x['timestamp'])
        return events
    
    def generate_market_data_timeline(self, duration_hours: int = 8) -> List[Dict]:
        """Generate realistic market data over time"""
        market_data = []
        start_time = datetime.now() - timedelta(hours=duration_hours)
        
        # Initialize base prices for each symbol
        base_prices = {symbol: self.random.uniform(100, 500) for symbol in self.market_symbols}
        
        # Generate data points every 5 minutes
        for minute in range(0, duration_hours * 60, 5):
            timestamp = start_time + timedelta(minutes=minute)
            
            for symbol in self.market_symbols:
                # Simulate price movement using random walk with trend
                current_price = base_prices[symbol]
                
                # Add some market-wide correlation
                market_sentiment = self.np_random.normal(0, 0.005)  # Overall market trend
                individual_movement = self.np_random.normal(0, 0.02)  # Individual stock movement
                
                # Price change
                price_change = market_sentiment + individual_movement
                new_price = current_price * (1 + price_change)
                base_prices[symbol] = max(10, new_price)  # Prevent negative prices
                
                # Calculate percentage change
                change_percent = (new_price - current_price) / current_price * 100
                
                # Generate volume (higher during market hours)
                hour = timestamp.hour
                if 9 <= hour <= 16:  # Market hours
                    base_volume = self.random.randint(500000, 5000000)
                else:
                    base_volume = self.random.randint(50000, 500000)
                
                # Volume spikes with large price movements
                volume_multiplier = 1 + abs(change_percent) * 0.5
                volume = int(base_volume * volume_multiplier)
                
                # Technical indicators
                rsi = max(0, min(100, 50 + self.np_random.normal(0, 20)))
                macd = self.np_random.normal(0, 2)
                
                bollinger_middle = new_price
                bollinger_std = new_price * 0.02
                bollinger_upper = bollinger_middle + (2 * bollinger_std)
                bollinger_lower = bollinger_middle - (2 * bollinger_std)
                
                # Sentiment score (-1 to 1)
                sentiment = np.tanh(change_percent * 0.1 + self.np_random.normal(0, 0.3))
                
                market_point = {
                    'symbol': symbol,
                    'price': round(new_price, 2),
                    'change': round(change_percent, 2),
                    'volume': volume,
                    'timestamp': timestamp.isoformat(),
                    'indicators': {
                        'rsi': round(rsi, 1),
                        'macd': round(macd, 3),
                        'bollinger': {
                            'upper': round(bollinger_upper, 2),
                            'lower': round(bollinger_lower, 2),
                            'middle': round(bollinger_middle, 2)
                        },
                        'sentiment': round(sentiment, 3)
                    }
                }
                
                market_data.append(market_point)
        
        return market_data
    
    def save_dataset(self, dataset: Dict[str, Any], filename: str):
        """Save generated dataset to file"""
        output_dir = Path(__file__).parent / 'generated'
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / f'{filename}.json'
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        
        print(f"Dataset saved to: {filepath}")
    
    def generate_complete_dataset(self, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate a complete synthetic dataset for visualization"""
        print("Generating synthetic agent network...")
        network = self.generate_agent_network(num_agents=12)
        
        print("Generating performance timeline...")
        performance_metrics = self.generate_performance_timeline(network['agents'], duration_hours=24)
        
        print("Generating trading scenarios...")
        trading_events = self.generate_trading_scenarios(network['agents'], num_events=300)
        
        print("Generating market data...")
        market_data = self.generate_market_data_timeline(duration_hours=8)
        
        dataset = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'description': 'Synthetic dataset for AI agents visualization system',
                'agents_count': len(network['agents']),
                'metrics_count': len(performance_metrics),
                'events_count': len(trading_events),
                'market_data_points': len(market_data)
            },
            'agent_network': network,
            'performance_metrics': performance_metrics,
            'trading_events': trading_events,
            'market_data': market_data
        }
        
        if save_to_file:
            self.save_dataset(dataset, f'ai_agents_dataset_{int(datetime.now().timestamp())}')
        
        print(f"Generated complete dataset with:")
        print(f"  - {len(network['agents'])} agents")
        print(f"  - {len(network['connections'])} connections")
        print(f"  - {len(performance_metrics)} performance metrics")
        print(f"  - {len(trading_events)} trading events")
        print(f"  - {len(market_data)} market data points")
        
        return dataset

def main():
    """Generate and save synthetic dataset"""
    generator = SyntheticDataGenerator(seed=42)
    dataset = generator.generate_complete_dataset(save_to_file=True)
    
    # Generate additional specialized datasets
    print("\nGenerating specialized datasets...")
    
    # High-performance scenario
    generator_hp = SyntheticDataGenerator(seed=123)
    agents = generator_hp.generate_agent_network(num_agents=16)['agents']
    # Boost all agent performance
    for agent in agents:
        agent['performance']['accuracy'] = min(99.9, agent['performance']['accuracy'] + 5)
        agent['performance']['throughput'] *= 1.5
        agent['performance']['responseTime'] *= 0.7
    
    hp_dataset = {
        'metadata': {
            'scenario': 'high_performance',
            'description': 'Optimized high-performance agent scenario'
        },
        'agent_network': {
            'agents': agents,
            'connections': generator_hp._generate_connections(agents),
            'timestamp': datetime.now().isoformat()
        }
    }
    generator_hp.save_dataset(hp_dataset, 'high_performance_scenario')
    
    # Crisis scenario
    generator_crisis = SyntheticDataGenerator(seed=456)
    crisis_agents = generator_crisis.generate_agent_network(num_agents=10)['agents']
    # Simulate system under stress
    for agent in crisis_agents:
        if generator_crisis.random.random() < 0.3:  # 30% of agents affected
            agent['status'] = 'error'
            agent['performance']['accuracy'] *= 0.7
            agent['performance']['responseTime'] *= 2.0
    
    crisis_dataset = {
        'metadata': {
            'scenario': 'crisis_response',
            'description': 'System under stress with multiple agent failures'
        },
        'agent_network': {
            'agents': crisis_agents,
            'connections': generator_crisis._generate_connections(crisis_agents),
            'timestamp': datetime.now().isoformat()
        }
    }
    generator_crisis.save_dataset(crisis_dataset, 'crisis_scenario')
    
    print("Synthetic data generation completed!")

if __name__ == "__main__":
    main()