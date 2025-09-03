"""
Distributed Memory Manager
Manages memory across multiple nodes/processes
"""

import asyncio
from typing import List, Dict, Any, Optional
import logging

from utils.observability.logging import get_logger
from .vector_memory import MemoryEntry, VectorMemoryStore

logger = get_logger(__name__)


class DistributedMemoryManager:
    """
    Manages distributed memory across multiple storage nodes
    Simple implementation for single-node operation
    """
    
    def __init__(self, nodes: Optional[List[str]] = None):
        self.nodes = nodes or ["local"]
        self.local_store = None
        logger.info(f"Initialized distributed memory manager with {len(self.nodes)} nodes")
    
    async def initialize(self) -> None:
        """Initialize the distributed memory system"""
        self.local_store = VectorMemoryStore()
    
    async def store_distributed(self, memory: MemoryEntry) -> bool:
        """Store memory across distributed nodes"""
        if self.local_store:
            await self.local_store.store_memory(memory)
            return True
        return False
    
    async def search_distributed(self, query: str, k: int = 5) -> List[tuple]:
        """Search across all distributed nodes"""
        if self.local_store:
            return await self.local_store.similarity_search(query, k)
        return []
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of distributed memory system"""
        return {
            "nodes": self.nodes,
            "status": "operational",
            "local_memories": await self.local_store.get_statistics() if self.local_store else {}
        }