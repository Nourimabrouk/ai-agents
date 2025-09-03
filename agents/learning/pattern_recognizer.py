"""
Pattern Recognizer for Learning Pipeline
Identifies patterns in agent behavior and performance
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from collections import Counter, defaultdict

from utils.observability.logging import get_logger

logger = get_logger(__name__)


class PatternRecognizer:
    """
    Recognizes patterns in agent learning and performance data
    """
    
    def __init__(self):
        self.patterns = {}
        logger.info("Initialized pattern recognizer")
    
    async def extract_patterns(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract patterns from experiences"""
        if not experiences:
            return {}
        
        patterns = {}
        
        # Strategy performance patterns
        strategy_performance = defaultdict(list)
        for exp in experiences:
            strategy = exp.get('strategy', 'unknown')
            performance = exp.get('performance', 0.5)
            strategy_performance[strategy].append(performance)
        
        patterns['strategy_performance'] = {}
        for strategy, perfs in strategy_performance.items():
            patterns['strategy_performance'][strategy] = {
                'average': np.mean(perfs),
                'variance': np.var(perfs),
                'confidence': min(len(perfs) / 10.0, 1.0)
            }
        
        # Context patterns
        if experiences and 'context' in experiences[0]:
            context_keys = set()
            for exp in experiences:
                if 'context' in exp:
                    context_keys.update(exp['context'].keys())
            
            patterns['context_patterns'] = {}
            for key in context_keys:
                values = []
                for exp in experiences:
                    if 'context' in exp and key in exp['context']:
                        values.append(exp['context'][key])
                
                if values:
                    patterns['context_patterns'][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'range': [min(values), max(values)]
                    }
        
        return patterns
    
    async def identify_trends(self, experiences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify trends in the data"""
        if len(experiences) < 3:
            return {}
        
        # Sort by timestamp if available
        if experiences and 'timestamp' in experiences[0]:
            experiences = sorted(experiences, key=lambda x: x.get('timestamp', datetime.now()))
        
        trends = {}
        
        # Performance trend
        performances = [exp.get('performance', 0.5) for exp in experiences]
        if len(performances) > 1:
            # Simple linear trend
            x = np.arange(len(performances))
            coeffs = np.polyfit(x, performances, 1)
            trend_slope = coeffs[0]
            
            trends['performance_trend'] = {
                'direction': 'improving' if trend_slope > 0 else 'declining' if trend_slope < 0 else 'stable',
                'slope': trend_slope,
                'magnitude': abs(trend_slope)
            }
        
        return trends