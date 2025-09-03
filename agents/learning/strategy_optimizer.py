"""
Strategy Optimizer for Meta-Learning
Optimizes agent strategies based on performance feedback
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from collections import defaultdict

from utils.observability.logging import get_logger

logger = get_logger(__name__)


class StrategyOptimizer:
    """
    Optimizes strategies based on performance feedback
    """
    
    def __init__(self):
        self.strategy_performance = defaultdict(list)
        self.strategy_contexts = defaultdict(list)
        self.optimization_history = []
        
        logger.info("Initialized strategy optimizer")
    
    async def add_performance_data(self, strategy: str, performance: float, context: Dict[str, Any]) -> None:
        """Add performance data for a strategy"""
        self.strategy_performance[strategy].append(performance)
        self.strategy_contexts[strategy].append(context)
    
    async def optimize_strategy(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Find optimal strategy for given context"""
        if not self.strategy_performance:
            return {"strategy": "default", "expected_performance": 0.5, "confidence": 0.1}
        
        best_strategy = None
        best_score = -1.0
        
        for strategy, performances in self.strategy_performance.items():
            if performances:
                avg_performance = np.mean(performances)
                if avg_performance > best_score:
                    best_score = avg_performance
                    best_strategy = strategy
        
        return {
            "strategy": best_strategy or "default",
            "expected_performance": best_score if best_score > 0 else 0.5,
            "confidence": min(len(self.strategy_performance[best_strategy]) / 10.0, 1.0) if best_strategy else 0.1
        }
    
    async def get_strategy_rankings(self) -> List[Tuple[str, float]]:
        """Get strategies ranked by performance"""
        rankings = []
        
        for strategy, performances in self.strategy_performance.items():
            if performances:
                avg_perf = np.mean(performances)
                rankings.append((strategy, avg_perf))
        
        return sorted(rankings, key=lambda x: x[1], reverse=True)