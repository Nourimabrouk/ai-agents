"""
Memory Consolidator
Consolidates and summarizes related memories
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging

from utils.observability.logging import get_logger
from .vector_memory import MemoryEntry

logger = get_logger(__name__)


class MemoryConsolidator:
    """
    Consolidates similar memories into summaries
    """
    
    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold
        logger.info("Initialized memory consolidator")
    
    async def consolidate_similar_memories(self, memories: List[MemoryEntry]) -> Dict[str, Any]:
        """Consolidate similar memories into summaries"""
        if len(memories) < 2:
            return {"consolidated_count": 0, "summary_count": 0}
        
        # Simple consolidation based on shared tags
        tag_groups = {}
        for memory in memories:
            for tag in memory.tags:
                if tag not in tag_groups:
                    tag_groups[tag] = []
                tag_groups[tag].append(memory)
        
        consolidated_count = 0
        summary_count = 0
        
        # Consolidate groups with multiple memories
        for tag, group_memories in tag_groups.items():
            if len(group_memories) > 2:  # Worth consolidating
                consolidated_count += len(group_memories) - 1  # All but one
                summary_count += 1
        
        return {
            "consolidated_count": consolidated_count,
            "summary_count": summary_count,
            "tag_groups": len(tag_groups)
        }
    
    async def create_summary(self, memories: List[MemoryEntry], summary_type: str = "auto") -> MemoryEntry:
        """Create a summary memory from multiple memories"""
        if not memories:
            raise ValueError("Cannot create summary from empty memories")
        
        # Combine content
        combined_content = f"Summary of {len(memories)} related memories: "
        combined_content += "; ".join([m.content[:50] + "..." for m in memories[:3]])
        
        # Combine metadata
        combined_metadata = {
            "type": "summary",
            "source_count": len(memories),
            "avg_importance": sum(m.relevance_score for m in memories) / len(memories)
        }
        
        # Combine tags
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.tags)
        unique_tags = list(set(all_tags))
        
        return MemoryEntry(
            id=f"summary_{hash(combined_content) % 100000:05d}",
            content=combined_content,
            metadata=combined_metadata,
            tags=unique_tags[:5],  # Limit to 5 most relevant tags
            relevance_score=min(1.0, sum(m.relevance_score for m in memories) / len(memories) + 0.1)
        )