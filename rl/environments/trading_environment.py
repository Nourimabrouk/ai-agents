"""
Multi-Agent Trading Environment
Advanced trading simulation with multiple AI agents, realistic market dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import random
import math
from enum import Enum

from .base_environment import (
    BaseMultiAgentEnvironment, 
    AgentConfig, 
    EnvironmentState,
    RewardShaper
)
from utils.observability.logging import get_logger

logger = get_logger(__name__)

class MarketRegime(Enum):
    """Different market conditions"""
    BULL = "bull"
    BEAR = "bear" 
    SIDEWAYS = "sideways"
    VOLATILE = "volatile"
    CRISIS = "crisis"

@dataclass
class MarketData:
    """Market data point"""
    symbol: str
    price: float
    volume: int
    bid: float
    ask: float
    timestamp: datetime
    indicators: Dict[str, float] = field(default_factory=dict)

@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: int  # Positive for long, negative for short
    entry_price: float
    entry_time: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Order:
    """Trading order"""
    agent_id: str
    symbol: str
    quantity: int
    order_type: str  # "market", "limit", "stop"
    price: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)

class TradingAgent:
    """Individual trading agent within the environment"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.balance = config.initial_balance
        self.positions: Dict[str, Position] = {}
        self.portfolio_value = config.initial_balance
        self.pnl = 0.0
        self.trade_history = []
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'volatility': 0.0
        }
    
    def calculate_portfolio_value(self, market_prices: Dict[str, float]) -> float:
        """Calculate current portfolio value"""
        total_value = self.balance
        
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position_value = position.quantity * market_prices[symbol]
                total_value += position_value
        
        return total_value
    
    def calculate_pnl(self, market_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L"""
        unrealized_pnl = 0.0
        
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                current_price = market_prices[symbol]
                position_pnl = position.quantity * (current_price - position.entry_price)
                unrealized_pnl += position_pnl
        
        return unrealized_pnl
    
    def update_metrics(self, market_prices: Dict[str, float]):
        """Update agent performance metrics"""
        current_portfolio_value = self.calculate_portfolio_value(market_prices)
        self.portfolio_value = current_portfolio_value
        
        # Calculate returns for metrics
        if self.trade_history:
            returns = [trade['pnl'] / self.config.initial_balance for trade in self.trade_history]
            
            if returns:
                self.metrics['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
                if self.metrics['volatility'] > 0:
                    self.metrics['sharpe_ratio'] = np.mean(returns) / self.metrics['volatility'] * np.sqrt(252)
        
        # Update drawdown
        peak_value = max([trade.get('portfolio_value', self.config.initial_balance) 
                         for trade in self.trade_history] + [current_portfolio_value])
        drawdown = (peak_value - current_portfolio_value) / peak_value
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)

class TradingEnvironment(BaseMultiAgentEnvironment):
    """
    Multi-agent trading environment with realistic market dynamics
    """
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        symbols: List[str] = None,
        max_steps: int = 1000,
        initial_prices: Optional[Dict[str, float]] = None,
        market_regime: MarketRegime = MarketRegime.SIDEWAYS,
        transaction_cost: float = 0.001,
        market_impact_factor: float = 0.0001,
        use_synthetic_data: bool = True,
        seed: Optional[int] = None
    ):
        self.symbols = symbols or ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN']
        self.market_regime = market_regime
        self.transaction_cost = transaction_cost
        self.market_impact_factor = market_impact_factor
        self.use_synthetic_data = use_synthetic_data
        
        # Initialize market data
        self.current_prices = initial_prices or {symbol: random.uniform(100, 500) for symbol in self.symbols}
        self.price_history = {symbol: [self.current_prices[symbol]] for symbol in self.symbols}
        self.market_data_history = []
        
        # Trading agents
        self.trading_agents = {config.agent_id: TradingAgent(config.agent_id, config) 
                             for config in agent_configs}
        
        # Order book simulation
        self.pending_orders: List[Order] = []
        self.executed_trades = []
        
        # Market conditions
        self.volatility = self._get_regime_volatility()
        self.trend = self._get_regime_trend()
        
        super().__init__(agent_configs, max_steps, reward_shaping=True, seed=seed)
        logger.info(f"Initialized trading environment with {len(self.symbols)} symbols")
    
    def _get_observation_dimension(self) -> int:
        """Calculate observation dimension"""
        # Global features (4) + symbol features (6 per symbol) + agent features (8) + other agents (4 per agent)
        global_features = 4  # volatility, trend, step_count, num_agents
        symbol_features = len(self.symbols) * 6  # price, change, volume, rsi, macd, bb_position
        agent_features = 8  # balance, portfolio_value, pnl, num_positions, risk_level, success_rate, etc.
        other_agents_features = (self.num_agents - 1) * 4  # other agents' performance
        
        return global_features + symbol_features + agent_features + other_agents_features
    
    def _get_action_dimension(self) -> int:
        """Calculate action dimension"""
        # Actions per symbol: [position_size, stop_loss, take_profit] = 3 actions per symbol
        return len(self.symbols) * 3
    
    def _get_regime_volatility(self) -> float:
        """Get volatility based on market regime"""
        regime_volatilities = {
            MarketRegime.BULL: 0.15,
            MarketRegime.BEAR: 0.25,
            MarketRegime.SIDEWAYS: 0.10,
            MarketRegime.VOLATILE: 0.35,
            MarketRegime.CRISIS: 0.50
        }
        return regime_volatilities.get(self.market_regime, 0.20)
    
    def _get_regime_trend(self) -> float:
        """Get trend based on market regime"""
        regime_trends = {
            MarketRegime.BULL: 0.15,
            MarketRegime.BEAR: -0.15,
            MarketRegime.SIDEWAYS: 0.0,
            MarketRegime.VOLATILE: 0.0,
            MarketRegime.CRISIS: -0.25
        }
        return regime_trends.get(self.market_regime, 0.0)
    
    def _initialize_environment_state(self) -> EnvironmentState:
        """Initialize trading environment state"""
        # Update market data
        self._update_market_data()
        
        # Initialize agent states
        agent_states = {}
        for agent_id, trading_agent in self.trading_agents.items():
            trading_agent.update_metrics(self.current_prices)
            
            agent_states[agent_id] = {
                'balance': trading_agent.balance,
                'portfolio_value': trading_agent.portfolio_value,
                'pnl': trading_agent.pnl,
                'positions': len(trading_agent.positions),
                'success_rate': 0.5,  # Initial success rate
                'risk_level': self.agent_configs[agent_id].risk_tolerance,
                'utilization': len(trading_agent.positions) / len(self.symbols),  # Position utilization
                'collaboration_score': 0.5,
                'connections': []
            }
        
        # Global state
        global_state = {
            'market_regime': self.market_regime.value,
            'volatility': self.volatility,
            'trend': self.trend,
            'total_volume': sum(random.randint(100000, 1000000) for _ in self.symbols),
            'market_cap': sum(self.current_prices.values()) * 1000000
        }
        
        # Market conditions
        market_conditions = {
            'volatility': self.volatility,
            'trend': self.trend,
            'regime': self.market_regime.value,
            'liquidity': 1.0,  # High liquidity initially
            'sentiment': 0.0   # Neutral sentiment
        }
        
        return EnvironmentState(
            global_state=global_state,
            agent_states=agent_states,
            market_conditions=market_conditions,
            timestamp=datetime.now(),
            step_count=0
        )
    
    def _execute_actions(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> EnvironmentState:
        """Execute trading actions and update environment state"""
        # Update market data first
        self._update_market_data()
        
        # Process each agent's actions
        for agent_id, action_vector in actions.items():
            if agent_id in self.trading_agents:
                self._process_agent_actions(agent_id, action_vector)
        
        # Execute pending orders
        self._execute_pending_orders()
        
        # Update agent metrics and states
        updated_agent_states = {}
        for agent_id, trading_agent in self.trading_agents.items():
            trading_agent.update_metrics(self.current_prices)
            
            # Calculate success rate based on recent trades
            recent_trades = trading_agent.trade_history[-10:] if len(trading_agent.trade_history) >= 10 else trading_agent.trade_history
            success_rate = sum(1 for trade in recent_trades if trade['pnl'] > 0) / len(recent_trades) if recent_trades else 0.5
            
            updated_agent_states[agent_id] = {
                'balance': trading_agent.balance,
                'portfolio_value': trading_agent.portfolio_value,
                'pnl': trading_agent.calculate_pnl(self.current_prices),
                'positions': len(trading_agent.positions),
                'success_rate': success_rate,
                'risk_level': self.agent_configs[agent_id].risk_tolerance,
                'utilization': len(trading_agent.positions) / len(self.symbols),
                'collaboration_score': self._calculate_collaboration_score(agent_id),
                'connections': self._get_agent_connections(agent_id)
            }
        
        # Update global state
        total_portfolio_value = sum(agent.portfolio_value for agent in self.trading_agents.values())
        market_conditions = {
            'volatility': self.volatility,
            'trend': self.trend,
            'regime': self.market_regime.value,
            'liquidity': self._calculate_market_liquidity(),
            'sentiment': self._calculate_market_sentiment()
        }
        
        return EnvironmentState(
            global_state={
                'market_regime': self.market_regime.value,
                'volatility': self.volatility,
                'trend': self.trend,
                'total_volume': sum(random.randint(100000, 1000000) for _ in self.symbols),
                'total_portfolio_value': total_portfolio_value
            },
            agent_states=updated_agent_states,
            market_conditions=market_conditions,
            timestamp=datetime.now(),
            step_count=state.step_count + 1
        )
    
    def _process_agent_actions(self, agent_id: str, action_vector: np.ndarray):
        """Process individual agent's actions"""
        trading_agent = self.trading_agents[agent_id]
        
        # Decode action vector: [position_size, stop_loss, take_profit] for each symbol
        for i, symbol in enumerate(self.symbols):
            action_start = i * 3
            position_size_action = action_vector[action_start]
            stop_loss_action = action_vector[action_start + 1] 
            take_profit_action = action_vector[action_start + 2]
            
            # Convert actions to trading decisions
            position_size = int(position_size_action * 100)  # Scale to reasonable position size
            
            if abs(position_size) > 5:  # Minimum threshold for trading
                current_price = self.current_prices[symbol]
                
                # Calculate stop loss and take profit prices
                stop_loss_pct = (stop_loss_action + 1) * 0.05  # 0 to 10% stop loss
                take_profit_pct = (take_profit_action + 1) * 0.1  # 0 to 20% take profit
                
                stop_loss = current_price * (1 - stop_loss_pct) if position_size > 0 else current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct) if position_size > 0 else current_price * (1 - take_profit_pct)
                
                # Create order
                order = Order(
                    agent_id=agent_id,
                    symbol=symbol,
                    quantity=position_size,
                    order_type="market",
                    price=current_price
                )
                
                self.pending_orders.append(order)
    
    def _execute_pending_orders(self):
        """Execute all pending orders"""
        executed_orders = []
        
        for order in self.pending_orders:
            if self._can_execute_order(order):
                self._execute_order(order)
                executed_orders.append(order)
        
        # Remove executed orders
        self.pending_orders = [order for order in self.pending_orders if order not in executed_orders]
    
    def _can_execute_order(self, order: Order) -> bool:
        """Check if order can be executed"""
        trading_agent = self.trading_agents[order.agent_id]
        
        # Check if agent has sufficient balance
        required_capital = abs(order.quantity) * self.current_prices[order.symbol]
        transaction_cost = required_capital * self.transaction_cost
        
        return trading_agent.balance >= (required_capital + transaction_cost)
    
    def _execute_order(self, order: Order):
        """Execute a trading order"""
        trading_agent = self.trading_agents[order.agent_id]
        symbol = order.symbol
        current_price = self.current_prices[symbol]
        
        # Calculate costs
        trade_value = abs(order.quantity) * current_price
        transaction_cost = trade_value * self.transaction_cost
        
        # Execute trade
        if symbol in trading_agent.positions:
            # Modify existing position
            existing_position = trading_agent.positions[symbol]
            new_quantity = existing_position.quantity + order.quantity
            
            if new_quantity == 0:
                # Close position
                pnl = existing_position.quantity * (current_price - existing_position.entry_price)
                trading_agent.balance += pnl - transaction_cost
                del trading_agent.positions[symbol]
                
                # Record trade
                trading_agent.trade_history.append({
                    'symbol': symbol,
                    'quantity': existing_position.quantity,
                    'entry_price': existing_position.entry_price,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'timestamp': datetime.now(),
                    'portfolio_value': trading_agent.calculate_portfolio_value(self.current_prices)
                })
                
                trading_agent.metrics['total_trades'] += 1
                if pnl > 0:
                    trading_agent.metrics['winning_trades'] += 1
                else:
                    trading_agent.metrics['losing_trades'] += 1
            else:
                # Update position
                existing_position.quantity = new_quantity
                # Weighted average entry price
                total_value = (existing_position.quantity * existing_position.entry_price + 
                             order.quantity * current_price)
                existing_position.entry_price = total_value / new_quantity
        else:
            # Create new position
            if order.quantity != 0:
                trading_agent.positions[symbol] = Position(
                    symbol=symbol,
                    quantity=order.quantity,
                    entry_price=current_price,
                    entry_time=datetime.now()
                )
        
        # Update agent balance
        trading_agent.balance -= transaction_cost
        
        # Record executed trade for market impact
        self.executed_trades.append({
            'agent_id': order.agent_id,
            'symbol': symbol,
            'quantity': order.quantity,
            'price': current_price,
            'timestamp': datetime.now()
        })
    
    def _update_market_data(self):
        """Update market prices and data"""
        # Market microstructure effects
        market_impact = self._calculate_market_impact()
        
        for symbol in self.symbols:
            current_price = self.current_prices[symbol]
            
            # Base price movement with regime characteristics
            random_shock = np.random.normal(0, self.volatility / np.sqrt(252))  # Daily volatility
            trend_component = self.trend / 252  # Daily trend
            
            # Apply market impact
            impact = market_impact.get(symbol, 0.0)
            
            # Calculate new price
            price_change = trend_component + random_shock + impact
            new_price = current_price * (1 + price_change)
            
            # Prevent negative prices
            new_price = max(new_price, current_price * 0.1)
            
            self.current_prices[symbol] = new_price
            self.price_history[symbol].append(new_price)
            
            # Keep limited history
            if len(self.price_history[symbol]) > 100:
                self.price_history[symbol] = self.price_history[symbol][-100:]
        
        # Update market regime occasionally
        if random.random() < 0.01:  # 1% chance per step
            self._potentially_change_regime()
    
    def _calculate_market_impact(self) -> Dict[str, float]:
        """Calculate market impact from recent trades"""
        impact = {symbol: 0.0 for symbol in self.symbols}
        
        # Consider recent trades (last few steps)
        recent_trades = [trade for trade in self.executed_trades 
                        if (datetime.now() - trade['timestamp']).total_seconds() < 60]
        
        for trade in recent_trades:
            symbol = trade['symbol']
            quantity = trade['quantity']
            
            # Simple market impact model
            relative_size = abs(quantity) / 1000.0  # Normalize by typical trade size
            impact_magnitude = relative_size * self.market_impact_factor
            
            # Positive quantity (buy) increases price, negative (sell) decreases price
            impact[symbol] += np.sign(quantity) * impact_magnitude
        
        return impact
    
    def _potentially_change_regime(self):
        """Occasionally change market regime"""
        regimes = list(MarketRegime)
        if random.random() < 0.3:  # 30% chance of regime change
            new_regime = random.choice(regimes)
            if new_regime != self.market_regime:
                logger.info(f"Market regime changed from {self.market_regime.value} to {new_regime.value}")
                self.market_regime = new_regime
                self.volatility = self._get_regime_volatility()
                self.trend = self._get_regime_trend()
    
    def _calculate_market_liquidity(self) -> float:
        """Calculate current market liquidity"""
        # Simple liquidity model based on recent trading activity
        recent_volume = len([trade for trade in self.executed_trades 
                           if (datetime.now() - trade['timestamp']).total_seconds() < 300])
        return min(1.0, recent_volume / 50.0 + 0.1)  # Base liquidity of 10%
    
    def _calculate_market_sentiment(self) -> float:
        """Calculate market sentiment based on recent price movements"""
        sentiment = 0.0
        for symbol in self.symbols:
            if len(self.price_history[symbol]) >= 2:
                price_change = (self.price_history[symbol][-1] - self.price_history[symbol][-2]) / self.price_history[symbol][-2]
                sentiment += price_change
        
        return np.tanh(sentiment / len(self.symbols))  # Normalize to [-1, 1]
    
    def _calculate_collaboration_score(self, agent_id: str) -> float:
        """Calculate how well agent collaborates with others"""
        # Simple collaboration score based on market impact consideration
        trading_agent = self.trading_agents[agent_id]
        recent_trades = [trade for trade in self.executed_trades 
                        if trade['agent_id'] == agent_id 
                        and (datetime.now() - trade['timestamp']).total_seconds() < 300]
        
        if not recent_trades:
            return 0.5
        
        # Lower score for larger market impact trades
        avg_trade_size = np.mean([abs(trade['quantity']) for trade in recent_trades])
        collaboration_score = max(0.0, 1.0 - (avg_trade_size / 1000.0))
        return collaboration_score
    
    def _get_agent_connections(self, agent_id: str) -> List[str]:
        """Get list of agents this agent is connected to (trading similar instruments)"""
        agent = self.trading_agents[agent_id]
        agent_symbols = set(agent.positions.keys())
        
        connections = []
        for other_agent_id, other_agent in self.trading_agents.items():
            if other_agent_id != agent_id:
                other_symbols = set(other_agent.positions.keys())
                if agent_symbols.intersection(other_symbols):
                    connections.append(other_agent_id)
        
        return connections
    
    def _calculate_rewards(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> Dict[str, float]:
        """Calculate rewards for each trading agent"""
        rewards = {}
        
        for agent_id, trading_agent in self.trading_agents.items():
            # Base reward: Portfolio return
            portfolio_value = trading_agent.calculate_portfolio_value(self.current_prices)
            portfolio_return = (portfolio_value - self.agent_configs[agent_id].initial_balance) / self.agent_configs[agent_id].initial_balance
            base_reward = portfolio_return * 10.0  # Scale reward
            
            # Risk-adjusted reward (Sharpe ratio component)
            risk_adjustment = 0.0
            if len(trading_agent.trade_history) > 5:
                returns = [trade['pnl'] / self.agent_configs[agent_id].initial_balance 
                          for trade in trading_agent.trade_history[-10:]]
                if returns and np.std(returns) > 0:
                    sharpe_like = np.mean(returns) / np.std(returns)
                    risk_adjustment = sharpe_like * 0.5
            
            # Collaboration bonus
            collaboration_bonus = 0.0
            if self.reward_shaping:
                collaboration_score = state.agent_states[agent_id]['collaboration_score']
                collaboration_bonus = (collaboration_score - 0.5) * 0.1
            
            # Diversity bonus (trading different instruments)
            diversity_bonus = 0.0
            if len(trading_agent.positions) > 1:
                diversity_bonus = min(0.1, len(trading_agent.positions) / len(self.symbols) * 0.2)
            
            # Transaction cost penalty
            recent_trades_count = len([trade for trade in self.executed_trades 
                                     if trade['agent_id'] == agent_id 
                                     and (datetime.now() - trade['timestamp']).total_seconds() < 60])
            transaction_penalty = recent_trades_count * self.transaction_cost * 0.1
            
            # Market impact penalty
            market_impact_penalty = 0.0
            if self.reward_shaping:
                agent_recent_trades = [trade for trade in self.executed_trades 
                                     if trade['agent_id'] == agent_id 
                                     and (datetime.now() - trade['timestamp']).total_seconds() < 60]
                if agent_recent_trades:
                    avg_trade_impact = np.mean([abs(trade['quantity']) for trade in agent_recent_trades])
                    market_impact_penalty = (avg_trade_impact / 1000.0) * 0.05
            
            # Combine all reward components
            total_reward = (base_reward + 
                          risk_adjustment + 
                          collaboration_bonus + 
                          diversity_bonus - 
                          transaction_penalty - 
                          market_impact_penalty)
            
            rewards[agent_id] = total_reward
        
        return rewards
    
    def get_market_data(self) -> Dict[str, Any]:
        """Get current market data for analysis"""
        market_data = {}
        
        for symbol in self.symbols:
            price_series = np.array(self.price_history[symbol])
            
            # Technical indicators
            rsi = self._calculate_rsi(price_series) if len(price_series) >= 14 else 50.0
            macd, signal = self._calculate_macd(price_series) if len(price_series) >= 26 else (0.0, 0.0)
            bb_upper, bb_lower = self._calculate_bollinger_bands(price_series) if len(price_series) >= 20 else (price_series[-1] * 1.02, price_series[-1] * 0.98)
            
            market_data[symbol] = {
                'price': self.current_prices[symbol],
                'change': ((price_series[-1] - price_series[-2]) / price_series[-2]) * 100 if len(price_series) >= 2 else 0.0,
                'volume': random.randint(100000, 1000000),  # Simulated volume
                'rsi': rsi,
                'macd': macd,
                'macd_signal': signal,
                'bollinger_upper': bb_upper,
                'bollinger_lower': bb_lower,
                'price_history': price_series[-50:].tolist() if len(price_series) >= 50 else price_series.tolist()
            }
        
        return market_data
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[float, float]:
        """Calculate MACD indicator"""
        if len(prices) < slow:
            return 0.0, 0.0
        
        ema_fast = self._calculate_ema(prices, fast)
        ema_slow = self._calculate_ema(prices, slow)
        
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line (EMA of MACD)
        if len(prices) < slow + signal_period:
            return macd_line, 0.0
        
        # Simplified signal calculation
        macd_history = [self._calculate_ema(prices[:i+1], fast) - self._calculate_ema(prices[:i+1], slow) 
                       for i in range(slow-1, len(prices))]
        
        if len(macd_history) >= signal_period:
            signal_line = self._calculate_ema(np.array(macd_history), signal_period)
        else:
            signal_line = macd_line
        
        return macd_line, signal_line
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average"""
        if len(prices) == 0:
            return 0.0
        
        if len(prices) < period:
            return np.mean(prices)
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            sma = np.mean(prices)
            std = np.std(prices) if len(prices) > 1 else prices[-1] * 0.02
        else:
            recent_prices = prices[-period:]
            sma = np.mean(recent_prices)
            std = np.std(recent_prices)
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, lower_band
    
    def get_trading_summary(self) -> Dict[str, Any]:
        """Get comprehensive trading summary"""
        summary = {
            'environment': {
                'step': self.step_count,
                'episode': self.episode_count,
                'market_regime': self.market_regime.value,
                'volatility': self.volatility,
                'trend': self.trend
            },
            'market_data': self.get_market_data(),
            'agents': {}
        }
        
        for agent_id, trading_agent in self.trading_agents.items():
            portfolio_value = trading_agent.calculate_portfolio_value(self.current_prices)
            pnl = trading_agent.calculate_pnl(self.current_prices)
            
            summary['agents'][agent_id] = {
                'balance': trading_agent.balance,
                'portfolio_value': portfolio_value,
                'total_pnl': pnl,
                'positions': {symbol: pos.quantity for symbol, pos in trading_agent.positions.items()},
                'metrics': trading_agent.metrics,
                'recent_trades': trading_agent.trade_history[-5:] if trading_agent.trade_history else []
            }
        
        return summary
    
    def render(self, mode: str = "human"):
        """Render the trading environment"""
        if mode == "human":
            self._render_trading_text()
        elif mode == "rgb_array":
            return self._render_trading_chart()
    
    def _render_trading_text(self):
        """Text rendering of trading environment"""
        print(f"\n--- Trading Environment Step {self.step_count} ---")
        print(f"Market Regime: {self.market_regime.value.title()}")
        print(f"Volatility: {self.volatility:.2%}")
        print(f"Trend: {self.trend:.2%}")
        
        print("\nMarket Prices:")
        for symbol in self.symbols:
            change = 0.0
            if len(self.price_history[symbol]) >= 2:
                change = ((self.price_history[symbol][-1] - self.price_history[symbol][-2]) 
                         / self.price_history[symbol][-2]) * 100
            
            print(f"  {symbol}: ${self.current_prices[symbol]:.2f} ({change:+.2f}%)")
        
        print("\nAgent Performance:")
        for agent_id, trading_agent in self.trading_agents.items():
            portfolio_value = trading_agent.calculate_portfolio_value(self.current_prices)
            pnl = trading_agent.calculate_pnl(self.current_prices)
            
            print(f"  {agent_id}:")
            print(f"    Portfolio: ${portfolio_value:,.2f} (P&L: ${pnl:+.2f})")
            print(f"    Positions: {len(trading_agent.positions)}")
            print(f"    Trades: {trading_agent.metrics['total_trades']} "
                  f"(Win Rate: {trading_agent.metrics['winning_trades']}/{trading_agent.metrics['total_trades']})")
    
    def _render_trading_chart(self) -> np.ndarray:
        """Create a simple chart representation"""
        # Create a simple visualization of price movements
        chart = np.zeros((200, 300, 3), dtype=np.uint8)
        
        # Draw price lines for each symbol
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        for i, symbol in enumerate(self.symbols[:len(colors)]):
            if len(self.price_history[symbol]) > 1:
                prices = np.array(self.price_history[symbol][-50:])  # Last 50 prices
                normalized_prices = ((prices - np.min(prices)) / (np.max(prices) - np.min(prices) + 1e-8) * 150 + 25).astype(int)
                
                for j in range(len(normalized_prices) - 1):
                    x1 = int(j * (300 / len(normalized_prices)))
                    y1 = 200 - normalized_prices[j]
                    x2 = int((j + 1) * (300 / len(normalized_prices)))
                    y2 = 200 - normalized_prices[j + 1]
                    
                    # Simple line drawing
                    if 0 <= x1 < 300 and 0 <= y1 < 200 and 0 <= x2 < 300 and 0 <= y2 < 200:
                        chart[y1, x1] = colors[i]
                        chart[y2, x2] = colors[i]
        
        return chart