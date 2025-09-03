"""
Advanced Memory System for AI Agents
Distributed memory with vector embeddings and semantic search
"""

from .vector_memory import VectorMemoryStore, MemoryEntry
from .semantic_search import SemanticSearchEngine
from .distributed_memory import DistributedMemoryManager
from .pattern_extractor import PatternExtractor
from .memory_consolidator import MemoryConsolidator

__all__ = [
    'VectorMemoryStore',
    'MemoryEntry',
    'SemanticSearchEngine',
    'DistributedMemoryManager',
    'PatternExtractor', 
    'MemoryConsolidator'
]