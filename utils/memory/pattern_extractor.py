"""
Pattern Extractor for Memory System
Extracts patterns and insights from stored memories
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging
from collections import Counter

from utils.observability.logging import get_logger
from .vector_memory import MemoryEntry

logger = get_logger(__name__)


class PatternExtractor:
    """
    Extracts patterns from memory entries
    """
    
    def __init__(self):
        logger.info("Initialized pattern extractor")
    
    async def extract_patterns(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Extract patterns from a set of memories"""
        if not memories:
            return {}
        
        # Extract tag patterns
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags)
        
        tag_frequency = Counter(all_tags)
        
        # Extract metadata patterns
        metadata_patterns = {}
        for memory in memories:
            for key, value in memory.metadata.items():
                if key not in metadata_patterns:
                    metadata_patterns[key] = []
                metadata_patterns[key].append(value)
        
        return {
            "tag_patterns": dict(tag_frequency.most_common(10)),
            "metadata_patterns": {k: Counter(v).most_common(5) for k, v in metadata_patterns.items()},
            "memory_count": len(memories),
            "avg_importance": sum(m.relevance_score for m in memories) / len(memories)
        }
    
    async def find_trending_patterns(self, memories: List[MemoryEntry], time_window_hours: int = 24) -> Dict[str, Any]:
        """Find trending patterns in recent memories"""
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        recent_memories = [m for m in memories if m.created_at > cutoff_time]
        
        return await self.extract_patterns(recent_memories)