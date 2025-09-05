"""
Phase 7 Working Memory System: Dynamic context management and memory consolidation
Implements hierarchical memory architecture inspired by cognitive psychology and Zettelkasten method
"""
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryImportance(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryNode:
    """Individual memory node with rich metadata"""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    tags: Set[str] = field(default_factory=set)
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    consolidation_score: float = 0.0
    decay_rate: float = 0.1
    context: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_relevance_score(self, current_time: Optional[datetime] = None) -> float:
        """Calculate relevance score based on recency, importance, and access patterns"""
        if current_time is None:
            current_time = datetime.now()
        
        # Recency factor (exponential decay)
        time_diff = (current_time - self.last_accessed).total_seconds() / 3600  # hours
        recency_score = np.exp(-self.decay_rate * time_diff)
        
        # Importance factor
        importance_score = self.importance.value / 4.0
        
        # Access frequency factor
        frequency_score = min(1.0, self.access_count / 10.0)
        
        # Consolidation factor
        consolidation_factor = min(1.0, self.consolidation_score)
        
        relevance = (
            recency_score * 0.3 +
            importance_score * 0.3 +
            frequency_score * 0.2 +
            consolidation_factor * 0.2
        )
        
        return relevance


@dataclass
class MemoryQuery:
    """Query for memory retrieval"""
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    memory_types: List[MemoryType] = field(default_factory=lambda: list(MemoryType))
    max_results: int = 10
    min_relevance: float = 0.3
    include_context: bool = True
    temporal_range: Optional[Tuple[datetime, datetime]] = None


class WorkingMemorySystem:
    """Advanced working memory system with hierarchical organization and consolidation"""
    
    def __init__(self, working_memory_capacity: int = 7, consolidation_threshold: float = 0.8):
        self.working_memory_capacity = working_memory_capacity
        self.consolidation_threshold = consolidation_threshold
        
        # Memory stores by type
        self.memory_stores = {
            MemoryType.WORKING: {},
            MemoryType.SHORT_TERM: {},
            MemoryType.LONG_TERM: {},
            MemoryType.EPISODIC: {},
            MemoryType.SEMANTIC: {},
            MemoryType.PROCEDURAL: {}
        }
        
        # Connection network (adjacency structure)
        self.connection_network = {}
        
        # Consolidation system
        self.consolidation_engine = MemoryConsolidationEngine()
        
        # Context tracking
        self.current_context = {}
        self.context_history = []
        
        # Performance metrics
        self.access_statistics = {
            'total_accesses': 0,
            'cache_hits': 0,
            'consolidations': 0,
            'memory_efficiency': 0.0
        }
    
    async def store_memory(self, content: str, memory_type: MemoryType, 
                          importance: MemoryImportance = MemoryImportance.MEDIUM,
                          tags: Set[str] = None, context: Dict[str, Any] = None) -> str:
        """Store new memory with automatic consolidation management"""
        
        if tags is None:
            tags = set()
        if context is None:
            context = {}
        
        # Generate unique memory ID
        memory_id = str(uuid.uuid4())
        
        # Create memory node
        memory_node = MemoryNode(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            context=context,
            embedding=await self._generate_embedding(content)
        )
        
        # Store in appropriate memory store
        self.memory_stores[memory_type][memory_id] = memory_node
        
        # Add to connection network
        self.connection_network[memory_id] = {}
        
        # Find and create connections to related memories
        await self._create_memory_connections(memory_node)
        
        # Working memory management
        if memory_type == MemoryType.WORKING:
            await self._manage_working_memory_capacity()
        
        # Trigger consolidation if needed
        if memory_node.importance.value >= 3:  # HIGH or CRITICAL
            await self._trigger_consolidation_check()
        
        logger.debug(f"Stored memory {memory_id} in {memory_type.value} store")
        return memory_id
    
    async def retrieve_memories(self, query: MemoryQuery) -> List[MemoryNode]:
        """Retrieve memories based on query with relevance ranking"""
        
        self.access_statistics['total_accesses'] += 1
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query.content)
        
        candidate_memories = []
        
        # Search relevant memory stores
        for memory_type in query.memory_types:
            store = self.memory_stores.get(memory_type, {})
            
            for memory_node in store.values():
                # Apply filters
                if query.temporal_range:
                    start_time, end_time = query.temporal_range
                    if not (start_time <= memory_node.created_at <= end_time):
                        continue
                
                # Calculate relevance
                semantic_similarity = await self._calculate_semantic_similarity(
                    query_embedding, memory_node.embedding
                )
                context_similarity = self._calculate_context_similarity(
                    query.context, memory_node.context
                ) if query.include_context else 0.0
                
                temporal_relevance = memory_node.calculate_relevance_score()
                
                # Combined relevance score
                relevance_score = (
                    semantic_similarity * 0.5 +
                    context_similarity * 0.3 +
                    temporal_relevance * 0.2
                )
                
                if relevance_score >= query.min_relevance:
                    candidate_memories.append((memory_node, relevance_score))
        
        # Sort by relevance and return top results
        candidate_memories.sort(key=lambda x: x[1], reverse=True)
        
        retrieved_memories = []
        for memory_node, relevance_score in candidate_memories[:query.max_results]:
            # Update access statistics
            memory_node.access_count += 1
            memory_node.last_accessed = datetime.now()
            retrieved_memories.append(memory_node)
        
        logger.debug(f"Retrieved {len(retrieved_memories)} memories for query")
        return retrieved_memories
    
    async def consolidate_memory(self, memory_id: str) -> bool:
        """Consolidate specific memory from short-term to long-term storage"""
        
        # Find memory in short-term store
        short_term_store = self.memory_stores[MemoryType.SHORT_TERM]
        if memory_id not in short_term_store:
            logger.warning(f"Memory {memory_id} not found in short-term store")
            return False
        
        memory_node = short_term_store[memory_id]
        
        # Check if memory meets consolidation criteria
        consolidation_score = await self._calculate_consolidation_score(memory_node)
        
        if consolidation_score >= self.consolidation_threshold:
            # Move to long-term storage
            del short_term_store[memory_id]
            memory_node.memory_type = MemoryType.LONG_TERM
            memory_node.consolidation_score = consolidation_score
            self.memory_stores[MemoryType.LONG_TERM][memory_id] = memory_node
            
            # Strengthen connections
            await self._strengthen_memory_connections(memory_node)
            
            self.access_statistics['consolidations'] += 1
            logger.info(f"Consolidated memory {memory_id} to long-term storage")
            return True
        
        logger.debug(f"Memory {memory_id} did not meet consolidation criteria")
        return False
    
    async def update_context(self, new_context: Dict[str, Any]) -> None:
        """Update current context and manage context history"""
        
        # Store previous context
        if self.current_context:
            self.context_history.append({
                'context': self.current_context.copy(),
                'timestamp': datetime.now()
            })
        
        # Update current context
        self.current_context = new_context.copy()
        
        # Limit context history size
        max_history = 100
        if len(self.context_history) > max_history:
            self.context_history = self.context_history[-max_history:]
        
        logger.debug("Updated working memory context")
    
    async def create_episodic_memory(self, experience: Dict[str, Any]) -> str:
        """Create episodic memory from experience"""
        
        # Structure episodic memory
        episodic_content = {
            'experience': experience,
            'context': self.current_context.copy(),
            'timestamp': datetime.now(),
            'emotional_valence': experience.get('emotional_valence', 0.0),
            'significance': experience.get('significance', 0.5),
            'participants': experience.get('participants', []),
            'outcomes': experience.get('outcomes', [])
        }
        
        # Determine importance based on experience characteristics
        significance = experience.get('significance', 0.5)
        if significance > 0.8:
            importance = MemoryImportance.CRITICAL
        elif significance > 0.6:
            importance = MemoryImportance.HIGH
        else:
            importance = MemoryImportance.MEDIUM
        
        # Generate tags from experience
        tags = set()
        if 'category' in experience:
            tags.add(experience['category'])
        if 'participants' in experience:
            tags.update(experience['participants'])
        
        memory_id = await self.store_memory(
            content=json.dumps(episodic_content, default=str),
            memory_type=MemoryType.EPISODIC,
            importance=importance,
            tags=tags,
            context=episodic_content['context']
        )
        
        logger.info(f"Created episodic memory {memory_id}")
        return memory_id
    
    async def extract_semantic_knowledge(self, memories: List[MemoryNode]) -> List[str]:
        """Extract semantic knowledge patterns from memories"""
        
        if not memories:
            return []
        
        # Analyze memory patterns for semantic extraction
        content_patterns = []
        tag_patterns = {}
        context_patterns = {}
        
        for memory in memories:
            # Content analysis
            content_keywords = await self._extract_keywords(memory.content)
            content_patterns.extend(content_keywords)
            
            # Tag pattern analysis
            for tag in memory.tags:
                tag_patterns[tag] = tag_patterns.get(tag, 0) + 1
            
            # Context pattern analysis
            for key, value in memory.context.items():
                if key not in context_patterns:
                    context_patterns[key] = {}
                context_patterns[key][str(value)] = context_patterns[key].get(str(value), 0) + 1
        
        # Generate semantic knowledge
        semantic_knowledge = []
        
        # Frequent patterns become semantic knowledge
        keyword_counts = {}
        for keyword in content_patterns:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        frequent_keywords = [k for k, v in keyword_counts.items() if v >= 3]
        if frequent_keywords:
            semantic_knowledge.append(f"Frequent concepts: {', '.join(frequent_keywords[:10])}")
        
        # Common tag patterns
        common_tags = [tag for tag, count in tag_patterns.items() if count >= 2]
        if common_tags:
            semantic_knowledge.append(f"Common categories: {', '.join(common_tags[:5])}")
        
        # Context patterns
        for context_key, value_counts in context_patterns.items():
            most_common_value = max(value_counts, key=value_counts.get)
            if value_counts[most_common_value] >= 2:
                semantic_knowledge.append(f"Common {context_key}: {most_common_value}")
        
        return semantic_knowledge
    
    async def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics"""
        
        stats = {
            'memory_stores': {},
            'total_memories': 0,
            'connection_density': 0.0,
            'consolidation_rate': 0.0,
            'access_statistics': self.access_statistics.copy(),
            'working_memory_utilization': 0.0
        }
        
        # Memory store statistics
        total_memories = 0
        for memory_type, store in self.memory_stores.items():
            count = len(store)
            stats['memory_stores'][memory_type.value] = count
            total_memories += count
        
        stats['total_memories'] = total_memories
        
        # Connection density
        if total_memories > 0:
            total_connections = sum(len(connections) for connections in self.connection_network.values())
            max_connections = total_memories * (total_memories - 1) / 2
            stats['connection_density'] = total_connections / max_connections if max_connections > 0 else 0.0
        
        # Working memory utilization
        working_count = len(self.memory_stores[MemoryType.WORKING])
        stats['working_memory_utilization'] = working_count / self.working_memory_capacity
        
        # Consolidation rate
        if self.access_statistics['total_accesses'] > 0:
            stats['consolidation_rate'] = self.access_statistics['consolidations'] / self.access_statistics['total_accesses']
        
        return stats
    
    # Private helper methods
    async def _generate_embedding(self, content: str) -> List[float]:
        """Generate embedding for content (mock implementation)"""
        # Mock embedding - would use actual embedding model
        hash_value = int(hashlib.md5(content.encode()).hexdigest(), 16)
        np.random.seed(hash_value % (2**32))
        return np.random.normal(0, 1, 384).tolist()  # 384-dim embedding
    
    async def _calculate_semantic_similarity(self, embedding1: List[float], 
                                           embedding2: Optional[List[float]]) -> float:
        """Calculate semantic similarity between embeddings"""
        if embedding2 is None:
            return 0.0
        
        # Cosine similarity
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norms = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norms == 0:
            return 0.0
        
        similarity = dot_product / norms
        return max(0.0, similarity)  # Ensure non-negative
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], 
                                    context2: Dict[str, Any]) -> float:
        """Calculate context similarity between two contexts"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if str(context1[key]) == str(context2[key]):
                matches += 1
        
        similarity = matches / len(common_keys)
        return similarity
    
    async def _create_memory_connections(self, new_memory: MemoryNode) -> None:
        """Create connections between new memory and existing related memories"""
        
        # Find related memories across all stores
        related_memories = []
        
        for memory_type, store in self.memory_stores.items():
            for existing_memory in store.values():
                if existing_memory.id == new_memory.id:
                    continue
                
                # Calculate relatedness
                semantic_similarity = await self._calculate_semantic_similarity(
                    new_memory.embedding, existing_memory.embedding
                )
                
                context_similarity = self._calculate_context_similarity(
                    new_memory.context, existing_memory.context
                )
                
                tag_overlap = len(new_memory.tags & existing_memory.tags) / max(1, len(new_memory.tags | existing_memory.tags))
                
                # Combined relatedness score
                relatedness = (semantic_similarity * 0.5 + context_similarity * 0.3 + tag_overlap * 0.2)
                
                if relatedness > 0.3:  # Threshold for creating connection
                    related_memories.append((existing_memory.id, relatedness))
        
        # Create bidirectional connections
        for related_id, strength in related_memories[:10]:  # Limit connections
            self.connection_network[new_memory.id][related_id] = strength
            if related_id in self.connection_network:
                self.connection_network[related_id][new_memory.id] = strength
    
    async def _manage_working_memory_capacity(self) -> None:
        """Manage working memory capacity by moving old memories to short-term storage"""
        
        working_store = self.memory_stores[MemoryType.WORKING]
        
        if len(working_store) <= self.working_memory_capacity:
            return {}
        
        # Sort by relevance and move least relevant to short-term storage
        memories_by_relevance = [
            (memory_id, memory.calculate_relevance_score())
            for memory_id, memory in working_store.items()
        ]
        memories_by_relevance.sort(key=lambda x: x[1])
        
        # Move excess memories to short-term storage
        excess_count = len(working_store) - self.working_memory_capacity
        for i in range(excess_count):
            memory_id, _ = memories_by_relevance[i]
            memory = working_store.pop(memory_id)
            memory.memory_type = MemoryType.SHORT_TERM
            self.memory_stores[MemoryType.SHORT_TERM][memory_id] = memory
            
            logger.debug(f"Moved memory {memory_id} from working to short-term storage")
    
    async def _calculate_consolidation_score(self, memory_node: MemoryNode) -> float:
        """Calculate consolidation score for memory"""
        
        # Factors influencing consolidation
        access_factor = min(1.0, memory_node.access_count / 5.0)  # Frequently accessed
        importance_factor = memory_node.importance.value / 4.0
        connection_strength = len(self.connection_network.get(memory_node.id, {})) / 10.0
        age_factor = min(1.0, (datetime.now() - memory_node.created_at).total_seconds() / 86400)  # Age in days
        
        consolidation_score = (
            access_factor * 0.3 +
            importance_factor * 0.4 +
            connection_strength * 0.2 +
            age_factor * 0.1
        )
        
        return consolidation_score
    
    async def _strengthen_memory_connections(self, memory_node: MemoryNode) -> None:
        """Strengthen connections for consolidated memory"""
        
        memory_id = memory_node.id
        if memory_id not in self.connection_network:
            return {}
        
        # Strengthen existing connections
        for connected_id, strength in self.connection_network[memory_id].items():
            new_strength = min(1.0, strength * 1.2)  # Increase by 20%
            self.connection_network[memory_id][connected_id] = new_strength
            
            # Update bidirectional connection
            if connected_id in self.connection_network:
                self.connection_network[connected_id][memory_id] = new_strength
    
    async def _trigger_consolidation_check(self) -> None:
        """Check and trigger consolidation for eligible short-term memories"""
        
        short_term_store = self.memory_stores[MemoryType.SHORT_TERM]
        
        for memory_id, memory in list(short_term_store.items()):
            consolidation_score = await self._calculate_consolidation_score(memory)
            
            if consolidation_score >= self.consolidation_threshold:
                await self.consolidate_memory(memory_id)
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content (simple implementation)"""
        
        # Simple keyword extraction - would use NLP in practice
        words = content.lower().split()
        
        # Filter stop words (simplified)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:10]  # Top 10 keywords


class MemoryConsolidationEngine:
    """Engine for managing memory consolidation processes"""
    
    def __init__(self):
        self.consolidation_rules = []
        self.pattern_detector = MemoryPatternDetector()
    
    async def run_consolidation_cycle(self, memory_system: WorkingMemorySystem) -> Dict[str, Any]:
        """Run complete consolidation cycle"""
        
        consolidation_results = {
            'memories_consolidated': 0,
            'patterns_discovered': [],
            'semantic_knowledge_extracted': [],
            'connection_strengthening': 0
        }
        
        # Consolidate eligible short-term memories
        short_term_store = memory_system.memory_stores[MemoryType.SHORT_TERM]
        for memory_id in list(short_term_store.keys()):
            if await memory_system.consolidate_memory(memory_id):
                consolidation_results['memories_consolidated'] += 1
        
        # Pattern detection
        all_memories = []
        for store in memory_system.memory_stores.values():
            all_memories.extend(store.values())
        
        patterns = await self.pattern_detector.detect_patterns(all_memories)
        consolidation_results['patterns_discovered'] = patterns
        
        # Semantic knowledge extraction
        if all_memories:
            semantic_knowledge = await memory_system.extract_semantic_knowledge(all_memories)
            consolidation_results['semantic_knowledge_extracted'] = semantic_knowledge
        
        return consolidation_results


class MemoryPatternDetector:
    """Detects patterns in memory structures"""
    
    async def detect_patterns(self, memories: List[MemoryNode]) -> List[str]:
        """Detect patterns across memories"""
        
        patterns = []
        
        if len(memories) < 3:
            return patterns
        
        # Temporal patterns
        temporal_pattern = await self._detect_temporal_patterns(memories)
        if temporal_pattern:
            patterns.append(temporal_pattern)
        
        # Content patterns
        content_patterns = await self._detect_content_patterns(memories)
        patterns.extend(content_patterns)
        
        # Access patterns
        access_pattern = await self._detect_access_patterns(memories)
        if access_pattern:
            patterns.append(access_pattern)
        
        return patterns
    
    async def _detect_temporal_patterns(self, memories: List[MemoryNode]) -> Optional[str]:
        """Detect temporal patterns in memory creation"""
        
        # Group memories by hour of day
        hour_distribution = {}
        for memory in memories:
            hour = memory.created_at.hour
            hour_distribution[hour] = hour_distribution.get(hour, 0) + 1
        
        if hour_distribution:
            peak_hour = max(hour_distribution, key=hour_distribution.get)
            if hour_distribution[peak_hour] > len(memories) * 0.3:
                return f"Peak memory formation at hour {peak_hour}"
        
        return {}
    
    async def _detect_content_patterns(self, memories: List[MemoryNode]) -> List[str]:
        """Detect content patterns across memories"""
        
        patterns = []
        
        # Tag co-occurrence patterns
        tag_pairs = {}
        for memory in memories:
            tags_list = list(memory.tags)
            for i, tag1 in enumerate(tags_list):
                for tag2 in tags_list[i+1:]:
                    pair = tuple(sorted([tag1, tag2]))
                    tag_pairs[pair] = tag_pairs.get(pair, 0) + 1
        
        frequent_pairs = [pair for pair, count in tag_pairs.items() if count >= 3]
        if frequent_pairs:
            patterns.append(f"Frequent tag combinations: {frequent_pairs[:3]}")
        
        return patterns
    
    async def _detect_access_patterns(self, memories: List[MemoryNode]) -> Optional[str]:
        """Detect access patterns in memories"""
        
        high_access_memories = [m for m in memories if m.access_count > 5]
        if len(high_access_memories) > len(memories) * 0.1:
            avg_importance = np.mean([m.importance.value for m in high_access_memories])
            return f"High-access memories average importance: {avg_importance:.2f}"
        
        return {}


if __name__ == "__main__":
    async def demo_working_memory_system():
        """Demonstrate working memory system capabilities"""
        
        memory_system = WorkingMemorySystem(
            working_memory_capacity=5,
            consolidation_threshold=0.6
        )
        
        print("=== Working Memory System Demo ===\n")
        
        # Store various types of memories
        print("1. Storing memories...")
        
        # Working memory
        working_memories = [
            ("Current task: Analyze financial data", MemoryType.WORKING, MemoryImportance.HIGH, {"task_type", "analysis"}),
            ("User requested quarterly report", MemoryType.WORKING, MemoryImportance.MEDIUM, {"user_request", "report"}),
            ("Database connection established", MemoryType.WORKING, MemoryImportance.LOW, {"system", "database"}),
            ("Processing invoice data", MemoryType.WORKING, MemoryImportance.MEDIUM, {"processing", "invoice"}),
            ("Error in data validation", MemoryType.WORKING, MemoryImportance.HIGH, {"error", "validation"}),
            ("Backup completed successfully", MemoryType.WORKING, MemoryImportance.LOW, {"backup", "success"}),
        ]
        
        for content, mem_type, importance, tags in working_memories:
            memory_id = await memory_system.store_memory(
                content, mem_type, importance, set(tags)
            )
            print(f"  Stored: {content[:30]}... (ID: {memory_id[:8]})")
        
        # Create episodic memory
        experience = {
            'category': 'problem_solving',
            'description': 'Successfully resolved data validation error',
            'participants': ['user', 'system'],
            'outcomes': ['error_fixed', 'process_improved'],
            'significance': 0.8,
            'emotional_valence': 0.6
        }
        
        episodic_id = await memory_system.create_episodic_memory(experience)
        print(f"  Created episodic memory: {episodic_id[:8]}")
        print()
        
        # Query memories
        print("2. Querying memories...")
        
        query = MemoryQuery(
            content="data analysis error",
            memory_types=[MemoryType.WORKING, MemoryType.EPISODIC],
            max_results=5,
            min_relevance=0.2
        )
        
        retrieved_memories = await memory_system.retrieve_memories(query)
        print(f"  Retrieved {len(retrieved_memories)} memories:")
        for memory in retrieved_memories:
            print(f"    - {memory.content[:50]}... (Relevance: {memory.calculate_relevance_score():.3f})")
        print()
        
        # Memory consolidation
        print("3. Memory consolidation...")
        consolidation_engine = MemoryConsolidationEngine()
        consolidation_results = await consolidation_engine.run_consolidation_cycle(memory_system)
        
        print(f"  Consolidated {consolidation_results['memories_consolidated']} memories")
        print(f"  Discovered {len(consolidation_results['patterns_discovered'])} patterns")
        if consolidation_results['patterns_discovered']:
            for pattern in consolidation_results['patterns_discovered']:
                print(f"    Pattern: {pattern}")
        print()
        
        # Extract semantic knowledge
        print("4. Semantic knowledge extraction...")
        all_memories = []
        for store in memory_system.memory_stores.values():
            all_memories.extend(store.values())
        
        semantic_knowledge = await memory_system.extract_semantic_knowledge(all_memories)
        print("  Extracted semantic knowledge:")
        for knowledge in semantic_knowledge:
            print(f"    - {knowledge}")
        print()
        
        # System statistics
        print("5. Memory system statistics...")
        stats = await memory_system.get_memory_statistics()
        
        print(f"  Total memories: {stats['total_memories']}")
        print(f"  Working memory utilization: {stats['working_memory_utilization']:.1%}")
        print(f"  Connection density: {stats['connection_density']:.3f}")
        print(f"  Consolidation rate: {stats['consolidation_rate']:.3f}")
        
        print("  Memory stores:")
        for memory_type, count in stats['memory_stores'].items():
            print(f"    {memory_type}: {count} memories")
        
        print("\n=== Working Memory System Demo Complete ===")
    
    # Run demonstration
    asyncio.run(demo_working_memory_system())