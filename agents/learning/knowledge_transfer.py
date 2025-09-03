"""
Knowledge Transfer System
Transfers learning between different contexts and domains
"""

import asyncio
import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from utils.observability.logging import get_logger

logger = get_logger(__name__)


class KnowledgeTransfer:
    """
    Handles knowledge transfer between different learning contexts
    """
    
    def __init__(self):
        self.transfer_mappings = {}
        self.transfer_history = []
        logger.info("Initialized knowledge transfer system")
    
    async def compute_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Compute similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        # Find common keys
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            val1 = context1[key]
            val2 = context2[key]
            
            # Simple numeric similarity
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Normalize difference to [0, 1] similarity
                max_diff = max(abs(val1), abs(val2), 1.0)
                similarity = 1.0 - abs(val1 - val2) / max_diff
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    async def transfer_knowledge(self, 
                                source_context: Dict[str, Any],
                                target_context: Dict[str, Any],
                                source_strategies: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer knowledge from source to target context"""
        
        similarity = await self.compute_context_similarity(source_context, target_context)
        
        if similarity > 0.3:  # Sufficient similarity for transfer
            # Find best performing strategy from source
            best_strategy = None
            best_performance = -1.0
            
            for strategy, info in source_strategies.items():
                performance = info.get('expected_performance', 0.5)
                if performance > best_performance:
                    best_performance = performance
                    best_strategy = strategy
            
            transfer_confidence = similarity * 0.8  # Discount for uncertainty
            
            result = {
                'strategy': best_strategy or 'default',
                'expected_performance': best_performance * similarity,
                'transfer_confidence': transfer_confidence,
                'source_similarity': similarity
            }
        else:
            # Low similarity - use conservative default
            result = {
                'strategy': 'default',
                'expected_performance': 0.5,
                'transfer_confidence': 0.1,
                'source_similarity': similarity
            }
        
        # Record transfer attempt
        self.transfer_history.append({
            'source_context': source_context,
            'target_context': target_context,
            'result': result,
            'timestamp': datetime.now()
        })
        
        return result
    
    async def get_transfer_insights(self) -> Dict[str, Any]:
        """Get insights about knowledge transfer performance"""
        if not self.transfer_history:
            return {}
        
        confidences = [t['result']['transfer_confidence'] for t in self.transfer_history]
        similarities = [t['result']['source_similarity'] for t in self.transfer_history]
        
        return {
            'total_transfers': len(self.transfer_history),
            'average_confidence': np.mean(confidences),
            'average_similarity': np.mean(similarities),
            'successful_transfers': len([t for t in self.transfer_history if t['result']['transfer_confidence'] > 0.5])
        }
    
    async def extract_transferable_knowledge(self, experiences: List[Dict[str, Any]], source_domain: str = None) -> Dict[str, Any]:
        """Extract transferable knowledge patterns from experiences"""
        if not experiences:
            return {'patterns': [], 'insights': 'No experiences to analyze'}
        
        # Simple pattern extraction based on successful experiences
        successful_patterns = []
        for exp in experiences:
            # Handle both dict and LearningExperience object types
            if hasattr(exp, 'success'):
                success = getattr(exp, 'success', False)
                strategy = getattr(exp, 'strategy_used', 'unknown')
                context = getattr(exp, 'context', {})
            else:
                success = exp.get('success', False)
                strategy = exp.get('strategy', 'unknown')
                context = exp.get('context', {})
            
            if success:
                pattern = {
                    'strategy': strategy,
                    'context': context,
                    'outcome': 'success'
                }
                successful_patterns.append(pattern)
        
        return {
            'patterns': successful_patterns,
            'insights': f'Extracted {len(successful_patterns)} successful patterns',
            'transferability_score': len(successful_patterns) / len(experiences) if experiences else 0.0
        }
    
    async def apply_transfer_learning(self, 
                                    source_knowledge: Dict[str, Any],
                                    target_context: str,
                                    strategies: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply transfer learning from source knowledge to target context"""
        patterns = source_knowledge.get('patterns', [])
        transferability_score = source_knowledge.get('transferability_score', 0.0)
        
        if not patterns or transferability_score < 0.3:
            return {
                'success': False,
                'reason': 'Insufficient transferable knowledge',
                'recommendations': []
            }
        
        # Apply patterns to target context
        applicable_patterns = []
        target_ctx = {'domain': target_context} if isinstance(target_context, str) else target_context
        for pattern in patterns:
            # Simple context matching (can be enhanced)
            if self._pattern_matches_context(pattern, target_ctx):
                applicable_patterns.append(pattern)
        
        return {
            'success': len(applicable_patterns) > 0,
            'applicable_patterns': applicable_patterns,
            'transfer_confidence': len(applicable_patterns) / len(patterns) if patterns else 0.0,
            'recommendations': [p.get('strategy', 'unknown') for p in applicable_patterns]
        }
    
    def _pattern_matches_context(self, pattern: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Simple pattern-context matching logic"""
        pattern_context = pattern.get('context', {})
        
        # Simple matching based on common keys
        common_keys = set(pattern_context.keys()) & set(context.keys())
        if not common_keys:
            return False
        
        # Check if at least 30% of context matches
        matches = 0
        for key in common_keys:
            if pattern_context[key] == context.get(key):
                matches += 1
        
        return matches / len(common_keys) >= 0.3 if common_keys else False