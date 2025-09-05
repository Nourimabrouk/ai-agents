"""
Advanced Working Memory System for Phase 7 - Autonomous Intelligence Ecosystem
Maintains coherent reasoning across 10,000+ tokens with hierarchical memory consolidation
"""

import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import numpy as np
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import heapq
import threading
from contextvars import ContextVar

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)

# Context variable for current reasoning session
current_session: ContextVar[Optional[str]] = ContextVar('current_session', default=None)


class MemoryType(Enum):
    """Types of memory storage"""
    WORKING = "working"          # Active reasoning context (limited capacity)
    SHORT_TERM = "short_term"    # Recent memories (minutes to hours)
    LONG_TERM = "long_term"      # Consolidated memories (persistent)
    EPISODIC = "episodic"        # Experience-based memories
    SEMANTIC = "semantic"        # Knowledge-based memories
    PROCEDURAL = "procedural"    # Skill-based memories
    META_COGNITIVE = "meta_cognitive"  # Self-awareness memories


class MemoryImportance(Enum):
    """Importance levels for memory prioritization"""
    CRITICAL = 5     # Core reasoning elements
    HIGH = 4         # Important context
    MEDIUM = 3       # Standard information
    LOW = 2          # Background context
    MINIMAL = 1      # Supplementary details


class MemoryAccessPattern(Enum):
    """Memory access patterns for optimization"""
    SEQUENTIAL = "sequential"
    RANDOM = "random"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"
    CAUSAL = "causal"


class CoherenceLevel(Enum):
    """Levels of memory coherence"""
    INCOHERENT = 1    # Contradictory or unrelated
    FRAGMENTED = 2    # Partially related but gaps
    CONSISTENT = 3    # Logically consistent
    INTEGRATED = 4    # Well-connected and integrated
    COHERENT = 5      # Fully coherent and unified


@dataclass
class MemoryNode:
    """Enhanced memory node with comprehensive metadata and token tracking"""
    id: str
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    token_count: int = 0
    tags: Set[str] = field(default_factory=set)
    connections: Dict[str, float] = field(default_factory=dict)  # node_id -> strength
    embedding: Optional[List[float]] = None
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    
    # Coherence tracking
    coherence_score: float = 0.0
    coherence_level: CoherenceLevel = CoherenceLevel.FRAGMENTED
    coherence_contributors: List[str] = field(default_factory=list)
    
    # Context tracking
    context: Dict[str, Any] = field(default_factory=dict)
    reasoning_session: Optional[str] = None
    parent_memories: Set[str] = field(default_factory=set)
    child_memories: Set[str] = field(default_factory=set)
    
    # Performance metrics
    consolidation_score: float = 0.0
    decay_rate: float = 0.1
    access_pattern: MemoryAccessPattern = MemoryAccessPattern.SEQUENTIAL
    retrieval_latency: float = 0.0
    
    def __post_init__(self):
        """Initialize computed fields after creation"""
        if self.token_count == 0:
            self.token_count = self._estimate_token_count()
        if not self.reasoning_session:
            self.reasoning_session = current_session.get()
    
    def _estimate_token_count(self) -> int:
        """Estimate token count for content"""
        # Rough approximation: 4 characters per token on average
        return max(1, len(self.content) // 4)
    
    def calculate_relevance_score(self, query_context: Dict[str, Any] = None, 
                                current_time: Optional[datetime] = None) -> float:
        """Calculate comprehensive relevance score"""
        if current_time is None:
            current_time = datetime.now()
        
        # Time-based decay
        time_diff = (current_time - self.last_accessed).total_seconds() / 3600  # hours
        recency_score = np.exp(-self.decay_rate * time_diff)
        
        # Importance weighting
        importance_score = self.importance.value / 5.0
        
        # Access frequency (with diminishing returns)
        frequency_score = min(1.0, np.log1p(self.access_count) / 5.0)
        
        # Coherence contribution
        coherence_score = self.coherence_score
        
        # Context relevance
        context_score = 0.5
        if query_context and self.context:
            context_score = self._calculate_context_similarity(query_context, self.context)
        
        # Connection strength (how well connected to other memories)
        connection_strength = min(1.0, len(self.connections) / 10.0)
        
        # Combined relevance with adaptive weighting
        relevance = (
            recency_score * 0.15 +
            importance_score * 0.25 +
            frequency_score * 0.15 +
            coherence_score * 0.2 +
            context_score * 0.15 +
            connection_strength * 0.1
        )
        
        return min(1.0, relevance)
    
    def _calculate_context_similarity(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        """Calculate similarity between contexts"""
        if not context1 or not context2:
            return 0.0
        
        common_keys = set(context1.keys()) & set(context2.keys())
        if not common_keys:
            return 0.0
        
        matches = 0
        for key in common_keys:
            if str(context1[key]).lower() == str(context2[key]).lower():
                matches += 1
            elif isinstance(context1[key], (int, float)) and isinstance(context2[key], (int, float)):
                # Numerical similarity
                if abs(context1[key] - context2[key]) / max(abs(context1[key]), abs(context2[key]), 1) < 0.1:
                    matches += 0.8
        
        return matches / len(common_keys)
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()
        
        # Update access pattern detection
        if self.access_count > 1:
            time_since_last = (self.last_accessed - self.created_at).total_seconds()
            if time_since_last < 60:  # Within a minute
                self.access_pattern = MemoryAccessPattern.SEQUENTIAL
            else:
                self.access_pattern = MemoryAccessPattern.RANDOM
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'id': self.id,
            'content': self.content,
            'memory_type': self.memory_type.value,
            'importance': self.importance.value,
            'token_count': self.token_count,
            'tags': list(self.tags),
            'access_count': self.access_count,
            'coherence_score': self.coherence_score,
            'coherence_level': self.coherence_level.value,
            'context': self.context,
            'reasoning_session': self.reasoning_session,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat()
        }


@dataclass 
class MemoryQuery:
    """Enhanced query for memory retrieval with token budget management"""
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    memory_types: List[MemoryType] = field(default_factory=lambda: list(MemoryType))
    max_results: int = 10
    max_tokens: int = 2000  # Token budget for retrieved memories
    min_relevance: float = 0.3
    min_coherence: float = 0.3
    include_context: bool = True
    temporal_range: Optional[Tuple[datetime, datetime]] = None
    require_session_continuity: bool = True
    prioritize_recent: bool = True
    coherence_optimization: bool = True


@dataclass
class ReasoningSession:
    """Tracks coherent reasoning across multiple interactions"""
    session_id: str
    started_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    memory_nodes: Set[str] = field(default_factory=set)
    total_tokens: int = 0
    coherence_score: float = 0.0
    reasoning_chain: List[str] = field(default_factory=list)
    context_evolution: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)


class MemoryCoherenceTracker:
    """Tracks and maintains coherence across memory network"""
    
    def __init__(self):
        self.coherence_graph = {}  # memory_id -> coherence connections
        self.global_coherence_score = 0.0
        self.coherence_history = []
        self.inconsistency_detector = InconsistencyDetector()
        
    async def calculate_coherence(self, memories: List[MemoryNode], 
                                context: Dict[str, Any] = None) -> float:
        """Calculate overall coherence score for memory set"""
        if len(memories) < 2:
            return 1.0  # Single memory is coherent by definition
        
        # Pairwise coherence analysis
        pairwise_scores = []
        
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i+1:]:
                coherence = await self._calculate_pairwise_coherence(memory1, memory2, context)
                pairwise_scores.append(coherence)
        
        # Global coherence metrics
        global_coherence = np.mean(pairwise_scores) if pairwise_scores else 0.0
        
        # Temporal coherence (memories should form logical sequence)
        temporal_coherence = await self._calculate_temporal_coherence(memories)
        
        # Semantic coherence (content should be logically consistent) 
        semantic_coherence = await self._calculate_semantic_coherence(memories)
        
        # Causal coherence (cause-effect relationships should be consistent)
        causal_coherence = await self._calculate_causal_coherence(memories)
        
        # Combined coherence score
        combined_coherence = (
            global_coherence * 0.3 +
            temporal_coherence * 0.25 +
            semantic_coherence * 0.25 +
            causal_coherence * 0.2
        )
        
        self.global_coherence_score = combined_coherence
        self.coherence_history.append({
            'timestamp': datetime.now(),
            'score': combined_coherence,
            'memory_count': len(memories)
        })
        
        return combined_coherence
    
    async def _calculate_pairwise_coherence(self, memory1: MemoryNode, 
                                          memory2: MemoryNode,
                                          context: Dict[str, Any] = None) -> float:
        """Calculate coherence between two memories"""
        coherence_factors = []
        
        # Content similarity
        content_similarity = await self._content_similarity(memory1.content, memory2.content)
        coherence_factors.append(content_similarity)
        
        # Tag overlap
        if memory1.tags and memory2.tags:
            tag_overlap = len(memory1.tags & memory2.tags) / len(memory1.tags | memory2.tags)
            coherence_factors.append(tag_overlap)
        
        # Context consistency
        context_consistency = memory1._calculate_context_similarity(memory1.context, memory2.context)
        coherence_factors.append(context_consistency)
        
        # Temporal consistency (memories close in time should be more coherent)
        time_diff = abs((memory1.created_at - memory2.created_at).total_seconds())
        temporal_factor = np.exp(-time_diff / 3600.0)  # Exponential decay over hours
        coherence_factors.append(temporal_factor)
        
        # Connection strength
        connection_strength = memory1.connections.get(memory2.id, 0.0)
        coherence_factors.append(connection_strength)
        
        return np.mean(coherence_factors)
    
    async def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity between two memory contents"""
        # Simple word overlap similarity (would use embeddings in practice)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _calculate_temporal_coherence(self, memories: List[MemoryNode]) -> float:
        """Calculate temporal coherence of memory sequence"""
        if len(memories) < 2:
            return 1.0
        
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        temporal_coherence_scores = []
        
        for i in range(len(sorted_memories) - 1):
            curr_memory = sorted_memories[i]
            next_memory = sorted_memories[i + 1]
            
            # Check for logical progression
            time_gap = (next_memory.created_at - curr_memory.created_at).total_seconds()
            
            # Prefer moderate time gaps (not too fast, not too slow)
            optimal_gap = 60.0  # 1 minute
            gap_score = np.exp(-abs(time_gap - optimal_gap) / optimal_gap)
            
            # Check content progression
            content_progression = await self._assess_content_progression(curr_memory, next_memory)
            
            temporal_coherence_scores.append((gap_score + content_progression) / 2)
        
        return np.mean(temporal_coherence_scores) if temporal_coherence_scores else 1.0
    
    async def _assess_content_progression(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Assess if content shows logical progression"""
        # Look for progression indicators
        progression_indicators = [
            'therefore', 'thus', 'consequently', 'as a result', 'leading to',
            'next', 'then', 'following', 'building on', 'expanding on'
        ]
        
        content2_lower = memory2.content.lower()
        progression_count = sum(1 for indicator in progression_indicators 
                              if indicator in content2_lower)
        
        # Also check for conceptual building (shared concepts)
        content_similarity = await self._content_similarity(memory1.content, memory2.content)
        
        return min(1.0, progression_count * 0.3 + content_similarity * 0.7)
    
    async def _calculate_semantic_coherence(self, memories: List[MemoryNode]) -> float:
        """Calculate semantic coherence across memories"""
        if len(memories) < 2:
            return 1.0
        
        # Check for semantic consistency
        semantic_scores = []
        
        # Topic consistency
        all_tags = set()
        for memory in memories:
            all_tags.update(memory.tags)
        
        topic_scores = []
        for memory in memories:
            if memory.tags and all_tags:
                topic_overlap = len(memory.tags & all_tags) / len(all_tags)
                topic_scores.append(topic_overlap)
        
        if topic_scores:
            semantic_scores.append(np.mean(topic_scores))
        
        # Contradiction detection
        contradiction_score = await self._detect_contradictions(memories)
        semantic_scores.append(1.0 - contradiction_score)  # Invert contradiction score
        
        return np.mean(semantic_scores) if semantic_scores else 0.5
    
    async def _detect_contradictions(self, memories: List[MemoryNode]) -> float:
        """Detect contradictions in memory content"""
        contradiction_indicators = [
            ('not', 'is'), ('cannot', 'can'), ('never', 'always'),
            ('impossible', 'possible'), ('false', 'true'), ('no', 'yes')
        ]
        
        contradiction_count = 0
        total_pairs = 0
        
        for i, memory1 in enumerate(memories):
            for memory2 in memories[i+1:]:
                total_pairs += 1
                content1_lower = memory1.content.lower()
                content2_lower = memory2.content.lower()
                
                for neg_word, pos_word in contradiction_indicators:
                    if (neg_word in content1_lower and pos_word in content2_lower) or \
                       (pos_word in content1_lower and neg_word in content2_lower):
                        contradiction_count += 1
                        break
        
        return contradiction_count / max(1, total_pairs)
    
    async def _calculate_causal_coherence(self, memories: List[MemoryNode]) -> float:
        """Calculate causal coherence in memory chain"""
        if len(memories) < 2:
            return 1.0
        
        # Look for causal relationships and check consistency
        causal_indicators = [
            'because', 'since', 'due to', 'caused by', 'results from',
            'leads to', 'causes', 'results in', 'therefore', 'thus'
        ]
        
        causal_connections = []
        
        for i, memory in enumerate(memories):
            content_lower = memory.content.lower()
            causal_count = sum(1 for indicator in causal_indicators if indicator in content_lower)
            
            if causal_count > 0:
                # Check if causal relationship is consistent with temporal order
                if i > 0:  # Has predecessor
                    causal_connections.append(1.0)
                else:
                    causal_connections.append(0.5)  # Partial score for initial cause
        
        return np.mean(causal_connections) if causal_connections else 0.5


class InconsistencyDetector:
    """Detects and resolves inconsistencies in memory"""
    
    def __init__(self):
        self.contradiction_patterns = [
            (r'\b(not|never|cannot|impossible)\b', r'\b(is|always|can|possible)\b'),
            (r'\b(false|incorrect|wrong)\b', r'\b(true|correct|right)\b'),
            (r'\b(increase|rise|up)\b', r'\b(decrease|fall|down)\b')
        ]
    
    async def detect_inconsistencies(self, memories: List[MemoryNode]) -> List[Dict[str, Any]]:
        """Detect inconsistencies in memory set"""
        inconsistencies = []
        
        # Check pairwise contradictions
        for i, memory1 in enumerate(memories):
            for j, memory2 in enumerate(memories[i+1:], i+1):
                contradiction = await self._check_contradiction(memory1, memory2)
                if contradiction:
                    inconsistencies.append({
                        'type': 'contradiction',
                        'memory1_id': memory1.id,
                        'memory2_id': memory2.id,
                        'description': contradiction,
                        'severity': 'high'
                    })
        
        # Check temporal inconsistencies
        temporal_issues = await self._check_temporal_consistency(memories)
        inconsistencies.extend(temporal_issues)
        
        return inconsistencies
    
    async def _check_contradiction(self, memory1: MemoryNode, memory2: MemoryNode) -> Optional[str]:
        """Check for contradictions between two memories"""
        import re
        
        content1 = memory1.content.lower()
        content2 = memory2.content.lower()
        
        for neg_pattern, pos_pattern in self.contradiction_patterns:
            if re.search(neg_pattern, content1) and re.search(pos_pattern, content2):
                return f"Contradiction detected: {memory1.content[:50]}... vs {memory2.content[:50]}..."
            if re.search(pos_pattern, content1) and re.search(neg_pattern, content2):
                return f"Contradiction detected: {memory1.content[:50]}... vs {memory2.content[:50]}..."
        
        return {}
    
    async def _check_temporal_consistency(self, memories: List[MemoryNode]) -> List[Dict[str, Any]]:
        """Check for temporal inconsistencies"""
        inconsistencies = []
        
        # Sort by creation time
        sorted_memories = sorted(memories, key=lambda m: m.created_at)
        
        for i in range(len(sorted_memories) - 1):
            curr = sorted_memories[i]
            next_mem = sorted_memories[i + 1]
            
            # Check for temporal contradictions (effects before causes)
            if 'result' in next_mem.content.lower() and 'because' in curr.content.lower():
                # This is expected
                continue
            elif 'because' in next_mem.content.lower() and 'result' in curr.content.lower():
                inconsistencies.append({
                    'type': 'temporal_inconsistency',
                    'memory1_id': curr.id,
                    'memory2_id': next_mem.id,
                    'description': 'Effect appears before cause',
                    'severity': 'medium'
                })
        
        return inconsistencies


class WorkingMemorySystem:
    """
    Advanced Working Memory System maintaining coherence across 10,000+ tokens
    Implements hierarchical memory organization with real-time coherence tracking
    """
    
    def __init__(self, 
                 working_memory_capacity: int = 7,
                 max_working_memory_tokens: int = 2000,
                 consolidation_threshold: float = 0.8,
                 coherence_threshold: float = 0.7,
                 max_total_tokens: int = 10000):
        
        self.working_memory_capacity = working_memory_capacity
        self.max_working_memory_tokens = max_working_memory_tokens
        self.consolidation_threshold = consolidation_threshold
        self.coherence_threshold = coherence_threshold
        self.max_total_tokens = max_total_tokens
        
        # Memory stores organized by type
        self.memory_stores = {
            memory_type: {} for memory_type in MemoryType
        }
        
        # Connection network for associative retrieval
        self.connection_network = {}  # memory_id -> {connected_id: strength}
        
        # Session management
        self.active_sessions: Dict[str, ReasoningSession] = {}
        self.session_lock = threading.Lock()
        
        # Coherence and consolidation systems
        self.coherence_tracker = MemoryCoherenceTracker()
        self.consolidation_engine = MemoryConsolidationEngine()
        
        # Performance tracking
        self.performance_metrics = {
            'total_memories': 0,
            'total_tokens': 0,
            'coherence_score': 0.0,
            'retrieval_latency': 0.0,
            'consolidation_rate': 0.0,
            'session_count': 0,
            'memory_efficiency': 0.0
        }
        
        # Real-time processing
        self.processing_queue = asyncio.Queue()
        self.background_tasks = set()
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized Working Memory System with {max_total_tokens} token capacity")
    
    async def start_reasoning_session(self, session_id: Optional[str] = None) -> str:
        """Start a new coherent reasoning session"""
        if session_id is None:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        with self.session_lock:
            session = ReasoningSession(session_id=session_id)
            self.active_sessions[session_id] = session
            self.performance_metrics['session_count'] = len(self.active_sessions)
        
        # Set context variable for this session
        current_session.set(session_id)
        
        logger.info(f"Started reasoning session: {session_id}")
        global_metrics.incr("working_memory.sessions_started")
        
        return session_id
    
    async def store_memory(self, 
                          content: str,
                          memory_type: MemoryType = MemoryType.WORKING,
                          importance: MemoryImportance = MemoryImportance.MEDIUM,
                          tags: Set[str] = None,
                          context: Dict[str, Any] = None,
                          session_id: Optional[str] = None) -> str:
        """Store memory with automatic coherence management and token tracking"""
        
        if tags is None:
            tags = set()
        if context is None:
            context = {}
        
        # Use current session if not specified
        if session_id is None:
            session_id = current_session.get()
        
        # Create memory node
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        memory_node = MemoryNode(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            context=context,
            reasoning_session=session_id,
            embedding=await self._generate_embedding(content)
        )
        
        # Token budget management
        if not await self._check_token_budget(memory_node):
            await self._make_space_for_memory(memory_node)
        
        # Store memory
        self.memory_stores[memory_type][memory_id] = memory_node
        self.connection_network[memory_id] = {}
        
        # Update session
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.memory_nodes.add(memory_id)
            session.total_tokens += memory_node.token_count
            session.last_active = datetime.now()
            session.reasoning_chain.append(memory_id)
        
        # Create connections to related memories
        await self._create_memory_connections(memory_node)
        
        # Working memory management
        if memory_type == MemoryType.WORKING:
            await self._manage_working_memory_capacity()
        
        # Update coherence
        await self._update_coherence_tracking(memory_node)
        
        # Trigger consolidation if needed
        if memory_node.importance.value >= 4:  # HIGH or CRITICAL
            await self._schedule_consolidation(memory_id)
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.debug(f"Stored memory {memory_id} ({memory_node.token_count} tokens) in {memory_type.value}")
        global_metrics.incr("working_memory.memories_stored")
        global_metrics.incr(f"working_memory.{memory_type.value}_memories")
        
        return memory_id
    
    async def retrieve_memories(self, 
                              query: MemoryQuery,
                              session_id: Optional[str] = None) -> List[MemoryNode]:
        """Retrieve memories with coherence optimization and token budget management"""
        
        start_time = datetime.now()
        
        if session_id is None:
            session_id = current_session.get()
        
        # Generate query embedding
        query_embedding = await self._generate_embedding(query.content)
        
        # Multi-stage retrieval process
        candidate_memories = await self._candidate_selection(query, session_id)
        
        # Relevance scoring with coherence optimization
        scored_memories = []
        for memory in candidate_memories:
            relevance_score = memory.calculate_relevance_score(query.context)
            
            # Boost score for session continuity
            if query.require_session_continuity and memory.reasoning_session == session_id:
                relevance_score *= 1.2
            
            # Boost score for coherence
            if query.coherence_optimization:
                coherence_boost = memory.coherence_score * 0.3
                relevance_score += coherence_boost
            
            if relevance_score >= query.min_relevance:
                scored_memories.append((memory, relevance_score))
        
        # Sort by relevance
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        # Token budget optimization
        selected_memories = await self._optimize_token_selection(
            scored_memories, query.max_tokens, query.max_results
        )
        
        # Update access patterns
        for memory in selected_memories:
            memory.update_access()
        
        # Calculate retrieval latency
        retrieval_time = (datetime.now() - start_time).total_seconds()
        self.performance_metrics['retrieval_latency'] = retrieval_time
        
        # Update session context
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.last_active = datetime.now()
            retrieved_ids = [mem.id for mem in selected_memories]
            session.context_evolution.append({
                'query': query.content,
                'retrieved_memories': retrieved_ids,
                'timestamp': datetime.now()
            })
        
        logger.debug(f"Retrieved {len(selected_memories)} memories ({sum(m.token_count for m in selected_memories)} tokens) in {retrieval_time:.3f}s")
        global_metrics.incr("working_memory.retrievals")
        global_metrics.timing("working_memory.retrieval_latency", retrieval_time)
        
        return selected_memories
    
    async def maintain_coherence(self, 
                               session_id: Optional[str] = None,
                               force_consolidation: bool = False) -> Dict[str, float]:
        """Actively maintain coherence across memory network"""
        
        if session_id is None:
            session_id = current_session.get()
        
        coherence_metrics = {}
        
        # Get session memories
        session_memories = []
        if session_id and session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session_memories = [
                self._get_memory_by_id(mem_id) for mem_id in session.memory_nodes
            ]
            session_memories = [mem for mem in session_memories if mem is not None]
        
        if not session_memories:
            return {'coherence_score': 1.0}
        
        # Calculate current coherence
        current_coherence = await self.coherence_tracker.calculate_coherence(
            session_memories, context={'session_id': session_id}
        )
        
        coherence_metrics['current_coherence'] = current_coherence
        
        # Detect inconsistencies
        inconsistencies = await self.coherence_tracker.inconsistency_detector.detect_inconsistencies(
            session_memories
        )
        coherence_metrics['inconsistency_count'] = len(inconsistencies)
        
        # Resolve inconsistencies if found
        if inconsistencies:
            resolved_count = await self._resolve_inconsistencies(inconsistencies, session_id)
            coherence_metrics['resolved_inconsistencies'] = resolved_count
        
        # Consolidate if coherence is low or force requested
        if current_coherence < self.coherence_threshold or force_consolidation:
            consolidation_result = await self.consolidation_engine.consolidate_session_memories(
                session_memories, session_id
            )
            coherence_metrics.update(consolidation_result)
        
        # Update session coherence
        if session_id in self.active_sessions:
            self.active_sessions[session_id].coherence_score = current_coherence
        
        # Update global coherence
        self.performance_metrics['coherence_score'] = current_coherence
        
        logger.info(f"Coherence maintenance completed for session {session_id}: {current_coherence:.3f}")
        global_metrics.gauge("working_memory.coherence_score", current_coherence)
        
        return coherence_metrics
    
    async def consolidate_memories(self, 
                                 memory_ids: List[str] = None,
                                 session_id: Optional[str] = None) -> Dict[str, Any]:
        """Consolidate memories from short-term to long-term storage"""
        
        if memory_ids is None:
            # Auto-select memories for consolidation
            memory_ids = await self._select_consolidation_candidates(session_id)
        
        consolidation_results = {
            'consolidated_count': 0,
            'failed_count': 0,
            'token_reduction': 0,
            'coherence_improvement': 0.0
        }
        
        initial_coherence = self.performance_metrics.get('coherence_score', 0.0)
        
        for memory_id in memory_ids:
            success = await self._consolidate_single_memory(memory_id)
            if success:
                consolidation_results['consolidated_count'] += 1
            else:
                consolidation_results['failed_count'] += 1
        
        # Update coherence after consolidation
        if session_id:
            coherence_metrics = await self.maintain_coherence(session_id)
            final_coherence = coherence_metrics.get('current_coherence', initial_coherence)
            consolidation_results['coherence_improvement'] = final_coherence - initial_coherence
        
        # Update performance metrics
        self._update_performance_metrics()
        
        logger.info(f"Consolidated {consolidation_results['consolidated_count']} memories")
        global_metrics.incr("working_memory.consolidations", consolidation_results['consolidated_count'])
        
        return consolidation_results
    
    async def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive summary of reasoning session"""
        
        if session_id not in self.active_sessions:
            return {'error': f'Session {session_id} not found'}
        
        session = self.active_sessions[session_id]
        
        # Get session memories
        session_memories = [
            self._get_memory_by_id(mem_id) for mem_id in session.memory_nodes
        ]
        session_memories = [mem for mem in session_memories if mem is not None]
        
        # Calculate session metrics
        coherence_score = await self.coherence_tracker.calculate_coherence(session_memories)
        
        # Memory type distribution
        memory_type_dist = {}
        for memory in session_memories:
            mem_type = memory.memory_type.value
            memory_type_dist[mem_type] = memory_type_dist.get(mem_type, 0) + 1
        
        # Reasoning chain analysis
        chain_length = len(session.reasoning_chain)
        chain_coherence = coherence_score
        
        # Token utilization
        total_tokens = sum(mem.token_count for mem in session_memories)
        token_utilization = total_tokens / self.max_total_tokens
        
        summary = {
            'session_id': session_id,
            'duration_minutes': (datetime.now() - session.started_at).total_seconds() / 60,
            'memory_count': len(session_memories),
            'total_tokens': total_tokens,
            'token_utilization': token_utilization,
            'coherence_score': coherence_score,
            'reasoning_chain_length': chain_length,
            'memory_type_distribution': memory_type_dist,
            'context_evolution_steps': len(session.context_evolution),
            'performance_metrics': session.performance_metrics
        }
        
        return summary
    
    async def cleanup_session(self, session_id: str, preserve_important: bool = True) -> Dict[str, int]:
        """Clean up session memories with optional preservation of important memories"""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session = self.active_sessions[session_id]
        cleanup_stats = {
            'removed_memories': 0,
            'preserved_memories': 0,
            'tokens_freed': 0
        }
        
        for memory_id in list(session.memory_nodes):
            memory = self._get_memory_by_id(memory_id)
            if memory:
                # Preserve important memories by promoting them
                if preserve_important and memory.importance.value >= 4:
                    if memory.memory_type == MemoryType.WORKING:
                        await self._promote_memory(memory, MemoryType.LONG_TERM)
                        cleanup_stats['preserved_memories'] += 1
                else:
                    # Remove memory
                    await self._remove_memory(memory_id)
                    cleanup_stats['removed_memories'] += 1
                    cleanup_stats['tokens_freed'] += memory.token_count
        
        # Remove session
        del self.active_sessions[session_id]
        self.performance_metrics['session_count'] = len(self.active_sessions)
        
        logger.info(f"Cleaned up session {session_id}: removed {cleanup_stats['removed_memories']}, preserved {cleanup_stats['preserved_memories']}")
        
        return cleanup_stats
    
    # Private helper methods
    
    async def _check_token_budget(self, new_memory: MemoryNode) -> bool:
        """Check if adding new memory exceeds token budget"""
        current_tokens = sum(
            sum(mem.token_count for mem in store.values())
            for store in self.memory_stores.values()
        )
        
        return current_tokens + new_memory.token_count <= self.max_total_tokens
    
    async def _make_space_for_memory(self, new_memory: MemoryNode) -> None:
        """Make space for new memory by removing less important memories"""
        
        tokens_needed = new_memory.token_count
        current_tokens = sum(
            sum(mem.token_count for mem in store.values())
            for store in self.memory_stores.values()
        )
        
        if current_tokens + tokens_needed <= self.max_total_tokens:
            return {}
        
        # Find candidates for removal (low importance, old, rarely accessed)
        removal_candidates = []
        
        for memory_type in [MemoryType.WORKING, MemoryType.SHORT_TERM]:
            for memory in self.memory_stores[memory_type].values():
                score = self._calculate_removal_score(memory)
                removal_candidates.append((memory, score))
        
        # Sort by removal score (higher score = better candidate for removal)
        removal_candidates.sort(key=lambda x: x[1], reverse=True)
        
        tokens_freed = 0
        for memory, score in removal_candidates:
            if tokens_freed >= tokens_needed:
                break
            
            await self._remove_memory(memory.id)
            tokens_freed += memory.token_count
        
        logger.info(f"Freed {tokens_freed} tokens by removing {len(removal_candidates)} memories")
    
    def _calculate_removal_score(self, memory: MemoryNode) -> float:
        """Calculate score for memory removal (higher = more likely to remove)"""
        
        # Age factor (older memories more likely to remove)
        age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
        age_score = min(1.0, age_hours / 24)  # Normalize to 24 hours
        
        # Importance factor (less important more likely to remove)
        importance_score = 1.0 - (memory.importance.value / 5.0)
        
        # Access frequency factor (less accessed more likely to remove)
        access_score = 1.0 - min(1.0, memory.access_count / 10.0)
        
        # Coherence factor (less coherent more likely to remove)
        coherence_score = 1.0 - memory.coherence_score
        
        # Combined removal score
        removal_score = (
            age_score * 0.3 +
            importance_score * 0.3 +
            access_score * 0.2 +
            coherence_score * 0.2
        )
        
        return removal_score
    
    async def _generate_embedding(self, content: str) -> List[float]:
        """Generate embedding for memory content"""
        # Mock implementation - would use actual embedding model
        hash_value = int(hashlib.md5(content.encode()).hexdigest(), 16)
        np.random.seed(hash_value % (2**32))
        return np.random.normal(0, 1, 768).tolist()  # 768-dim embedding
    
    async def _create_memory_connections(self, new_memory: MemoryNode) -> None:
        """Create connections between new memory and existing related memories"""
        
        max_connections = 10  # Limit connections per memory
        connections_created = 0
        
        # Search for related memories across all stores
        for memory_type, store in self.memory_stores.items():
            if connections_created >= max_connections:
                break
                
            for existing_memory in store.values():
                if existing_memory.id == new_memory.id:
                    continue
                
                # Calculate relatedness
                relatedness = await self._calculate_memory_relatedness(new_memory, existing_memory)
                
                if relatedness > 0.3:  # Threshold for creating connection
                    # Create bidirectional connection
                    self.connection_network[new_memory.id][existing_memory.id] = relatedness
                    if existing_memory.id in self.connection_network:
                        self.connection_network[existing_memory.id][new_memory.id] = relatedness
                    
                    # Update memory objects
                    new_memory.connections[existing_memory.id] = relatedness
                    existing_memory.connections[new_memory.id] = relatedness
                    
                    connections_created += 1
    
    async def _calculate_memory_relatedness(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Calculate relatedness between two memories"""
        
        # Semantic similarity (would use embeddings in practice)
        semantic_similarity = await self._semantic_similarity(memory1.content, memory2.content)
        
        # Context similarity
        context_similarity = memory1._calculate_context_similarity(memory1.context, memory2.context)
        
        # Tag overlap
        tag_overlap = 0.0
        if memory1.tags and memory2.tags:
            tag_overlap = len(memory1.tags & memory2.tags) / len(memory1.tags | memory2.tags)
        
        # Session continuity
        session_bonus = 0.2 if memory1.reasoning_session == memory2.reasoning_session else 0.0
        
        # Temporal proximity
        time_diff = abs((memory1.created_at - memory2.created_at).total_seconds())
        temporal_factor = np.exp(-time_diff / 3600.0)  # Decay over hours
        
        # Combined relatedness
        relatedness = (
            semantic_similarity * 0.4 +
            context_similarity * 0.2 +
            tag_overlap * 0.2 +
            temporal_factor * 0.1 +
            session_bonus
        )
        
        return min(1.0, relatedness)
    
    async def _semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between content"""
        # Simple word overlap (would use embeddings in practice)
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    async def _manage_working_memory_capacity(self) -> None:
        """Manage working memory capacity by promoting or demoting memories"""
        
        working_store = self.memory_stores[MemoryType.WORKING]
        
        if len(working_store) <= self.working_memory_capacity:
            return {}
        
        # Calculate token count in working memory
        working_tokens = sum(mem.token_count for mem in working_store.values())
        
        if working_tokens <= self.max_working_memory_tokens:
            return {}
        
        # Sort memories by relevance/importance
        memories_by_priority = list(working_store.values())
        memories_by_priority.sort(
            key=lambda m: (m.importance.value, m.coherence_score, m.calculate_relevance_score()),
            reverse=True
        )
        
        # Keep top memories, promote others to short-term
        keep_count = 0
        keep_tokens = 0
        
        for memory in memories_by_priority:
            if (keep_count < self.working_memory_capacity and 
                keep_tokens + memory.token_count <= self.max_working_memory_tokens):
                keep_count += 1
                keep_tokens += memory.token_count
            else:
                # Promote to short-term storage
                await self._promote_memory(memory, MemoryType.SHORT_TERM)
                logger.debug(f"Promoted memory {memory.id} to short-term storage")
    
    async def _promote_memory(self, memory: MemoryNode, target_type: MemoryType) -> None:
        """Promote memory from current type to target type"""
        
        # Remove from current store
        current_store = self.memory_stores[memory.memory_type]
        if memory.id in current_store:
            del current_store[memory.id]
        
        # Update memory type
        memory.memory_type = target_type
        memory.modified_at = datetime.now()
        
        # Add to target store
        self.memory_stores[target_type][memory.id] = memory
        
        global_metrics.incr(f"working_memory.promoted_to_{target_type.value}")
    
    async def _update_coherence_tracking(self, new_memory: MemoryNode) -> None:
        """Update coherence tracking with new memory"""
        
        # Get related memories from the same session
        session_id = new_memory.reasoning_session
        if not session_id or session_id not in self.active_sessions:
            return {}
        
        session = self.active_sessions[session_id]
        session_memories = [
            self._get_memory_by_id(mem_id) for mem_id in session.memory_nodes
        ]
        session_memories = [mem for mem in session_memories if mem is not None]
        
        if len(session_memories) > 1:
            # Update coherence scores
            coherence_score = await self.coherence_tracker.calculate_coherence(session_memories)
            
            # Update individual memory coherence scores
            for memory in session_memories:
                memory.coherence_score = coherence_score
                
                # Determine coherence level
                if coherence_score >= 0.9:
                    memory.coherence_level = CoherenceLevel.COHERENT
                elif coherence_score >= 0.7:
                    memory.coherence_level = CoherenceLevel.INTEGRATED
                elif coherence_score >= 0.5:
                    memory.coherence_level = CoherenceLevel.CONSISTENT
                elif coherence_score >= 0.3:
                    memory.coherence_level = CoherenceLevel.FRAGMENTED
                else:
                    memory.coherence_level = CoherenceLevel.INCOHERENT
    
    async def _schedule_consolidation(self, memory_id: str) -> None:
        """Schedule memory for consolidation"""
        
        # Add to processing queue for background consolidation
        await self.processing_queue.put({
            'type': 'consolidation',
            'memory_id': memory_id,
            'timestamp': datetime.now()
        })
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryNode]:
        """Get memory by ID from any store"""
        for store in self.memory_stores.values():
            if memory_id in store:
                return store[memory_id]
        return {}
    
    async def _remove_memory(self, memory_id: str) -> bool:
        """Remove memory from all stores and connections"""
        
        memory = self._get_memory_by_id(memory_id)
        if not memory:
            return False
        
        # Remove from store
        store = self.memory_stores[memory.memory_type]
        if memory_id in store:
            del store[memory_id]
        
        # Remove from connections
        if memory_id in self.connection_network:
            # Remove bidirectional connections
            for connected_id in self.connection_network[memory_id]:
                if connected_id in self.connection_network:
                    self.connection_network[connected_id].pop(memory_id, None)
            
            del self.connection_network[memory_id]
        
        # Remove from sessions
        for session in self.active_sessions.values():
            session.memory_nodes.discard(memory_id)
            if memory_id in session.reasoning_chain:
                session.reasoning_chain.remove(memory_id)
        
        global_metrics.incr("working_memory.memories_removed")
        return True
    
    def _update_performance_metrics(self) -> None:
        """Update system performance metrics"""
        
        # Count memories and tokens
        total_memories = sum(len(store) for store in self.memory_stores.values())
        total_tokens = sum(
            sum(mem.token_count for mem in store.values())
            for store in self.memory_stores.values()
        )
        
        # Memory efficiency (useful memories / total memories)
        high_importance_count = sum(
            sum(1 for mem in store.values() if mem.importance.value >= 4)
            for store in self.memory_stores.values()
        )
        memory_efficiency = high_importance_count / max(1, total_memories)
        
        # Update metrics
        self.performance_metrics.update({
            'total_memories': total_memories,
            'total_tokens': total_tokens,
            'token_utilization': total_tokens / self.max_total_tokens,
            'memory_efficiency': memory_efficiency,
            'consolidation_rate': len(self.memory_stores[MemoryType.LONG_TERM]) / max(1, total_memories)
        })
        
        # Update global metrics
        global_metrics.gauge("working_memory.total_memories", total_memories)
        global_metrics.gauge("working_memory.total_tokens", total_tokens)
        global_metrics.gauge("working_memory.memory_efficiency", memory_efficiency)
    
    # Additional methods for retrieval optimization
    
    async def _candidate_selection(self, query: MemoryQuery, session_id: Optional[str]) -> List[MemoryNode]:
        """Select candidate memories for retrieval"""
        candidates = []
        
        # Search relevant memory stores
        for memory_type in query.memory_types:
            store = self.memory_stores.get(memory_type, {})
            
            for memory in store.values():
                # Apply basic filters
                if query.temporal_range:
                    start_time, end_time = query.temporal_range
                    if not (start_time <= memory.created_at <= end_time):
                        continue
                
                # Session continuity filter
                if query.require_session_continuity and session_id:
                    if memory.reasoning_session != session_id:
                        continue
                
                # Minimum coherence filter
                if memory.coherence_score < query.min_coherence:
                    continue
                
                candidates.append(memory)
        
        return candidates
    
    async def _optimize_token_selection(self, 
                                      scored_memories: List[Tuple[MemoryNode, float]], 
                                      max_tokens: int, 
                                      max_results: int) -> List[MemoryNode]:
        """Optimize memory selection based on token budget and relevance"""
        
        if not scored_memories:
            return []
        
        # Sort by relevance score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        total_tokens = 0
        
        for memory, score in scored_memories:
            if len(selected) >= max_results:
                break
            
            if total_tokens + memory.token_count <= max_tokens:
                selected.append(memory)
                total_tokens += memory.token_count
            else:
                # Check if we can fit smaller memories
                remaining_budget = max_tokens - total_tokens
                for other_memory, other_score in scored_memories:
                    if (other_memory not in selected and 
                        other_memory.token_count <= remaining_budget and
                        len(selected) < max_results):
                        selected.append(other_memory)
                        total_tokens += other_memory.token_count
                        break
        
        return selected
    
    async def _select_consolidation_candidates(self, session_id: Optional[str] = None) -> List[str]:
        """Select memories for consolidation"""
        candidates = []
        
        # Select from short-term memory
        short_term_store = self.memory_stores[MemoryType.SHORT_TERM]
        
        for memory in short_term_store.values():
            # Calculate consolidation score
            consolidation_score = await self._calculate_consolidation_score(memory)
            
            if consolidation_score >= self.consolidation_threshold:
                candidates.append(memory.id)
        
        return candidates
    
    async def _calculate_consolidation_score(self, memory: MemoryNode) -> float:
        """Calculate consolidation score for memory"""
        
        # Access frequency
        access_factor = min(1.0, memory.access_count / 5.0)
        
        # Importance
        importance_factor = memory.importance.value / 5.0
        
        # Connection strength
        connection_strength = len(memory.connections) / 10.0
        
        # Age (older memories more likely to consolidate)
        age_hours = (datetime.now() - memory.created_at).total_seconds() / 3600
        age_factor = min(1.0, age_hours / 24.0)
        
        # Coherence
        coherence_factor = memory.coherence_score
        
        consolidation_score = (
            access_factor * 0.25 +
            importance_factor * 0.3 +
            connection_strength * 0.15 +
            age_factor * 0.1 +
            coherence_factor * 0.2
        )
        
        return consolidation_score
    
    async def _consolidate_single_memory(self, memory_id: str) -> bool:
        """Consolidate single memory from short-term to long-term storage"""
        
        memory = self._get_memory_by_id(memory_id)
        if not memory or memory.memory_type != MemoryType.SHORT_TERM:
            return False
        
        # Check consolidation criteria
        consolidation_score = await self._calculate_consolidation_score(memory)
        
        if consolidation_score >= self.consolidation_threshold:
            # Promote to long-term storage
            await self._promote_memory(memory, MemoryType.LONG_TERM)
            
            # Strengthen connections
            await self._strengthen_memory_connections(memory)
            
            memory.consolidation_score = consolidation_score
            logger.debug(f"Consolidated memory {memory_id} to long-term storage")
            return True
        
        return False
    
    async def _strengthen_memory_connections(self, memory: MemoryNode) -> None:
        """Strengthen connections for consolidated memory"""
        
        # Increase connection strengths
        for connected_id, strength in memory.connections.items():
            new_strength = min(1.0, strength * 1.1)  # 10% increase
            memory.connections[connected_id] = new_strength
            
            # Update bidirectional connection
            if connected_id in self.connection_network:
                self.connection_network[connected_id][memory.id] = new_strength
    
    async def _resolve_inconsistencies(self, 
                                     inconsistencies: List[Dict[str, Any]], 
                                     session_id: str) -> int:
        """Resolve detected inconsistencies in memory"""
        
        resolved_count = 0
        
        for inconsistency in inconsistencies:
            if inconsistency['type'] == 'contradiction':
                # Handle contradictions by marking lower confidence memory
                memory1_id = inconsistency['memory1_id']
                memory2_id = inconsistency['memory2_id']
                
                memory1 = self._get_memory_by_id(memory1_id)
                memory2 = self._get_memory_by_id(memory2_id)
                
                if memory1 and memory2:
                    # Keep memory with higher importance/coherence
                    if memory1.importance.value < memory2.importance.value:
                        memory1.tags.add('potential_contradiction')
                        memory1.importance = MemoryImportance.LOW
                    else:
                        memory2.tags.add('potential_contradiction')
                        memory2.importance = MemoryImportance.LOW
                    
                    resolved_count += 1
            
            elif inconsistency['type'] == 'temporal_inconsistency':
                # Handle temporal issues by reordering or marking
                memory1_id = inconsistency['memory1_id']
                memory2_id = inconsistency['memory2_id']
                
                memory1 = self._get_memory_by_id(memory1_id)
                memory2 = self._get_memory_by_id(memory2_id)
                
                if memory1 and memory2:
                    # Add temporal context to clarify ordering
                    memory1.tags.add('temporal_sequence')
                    memory2.tags.add('temporal_sequence')
                    resolved_count += 1
        
        logger.info(f"Resolved {resolved_count} inconsistencies in session {session_id}")
        return resolved_count


class MemoryConsolidationEngine:
    """Engine for advanced memory consolidation with pattern recognition"""
    
    def __init__(self):
        self.consolidation_patterns = []
        self.semantic_clusterer = SemanticClusterer()
        
    async def consolidate_session_memories(self, 
                                         memories: List[MemoryNode], 
                                         session_id: str) -> Dict[str, Any]:
        """Consolidate memories from a reasoning session"""
        
        if not memories:
            return {'consolidated_memories': 0}
        
        consolidation_results = {
            'initial_memory_count': len(memories),
            'consolidated_memories': 0,
            'created_semantic_memories': 0,
            'identified_patterns': 0,
            'token_efficiency_gain': 0.0
        }
        
        initial_tokens = sum(mem.token_count for mem in memories)
        
        # Cluster related memories
        memory_clusters = await self.semantic_clusterer.cluster_memories(memories)
        consolidation_results['identified_patterns'] = len(memory_clusters)
        
        # Consolidate each cluster
        for cluster in memory_clusters:
            if len(cluster) >= 2:  # Only consolidate clusters with multiple memories
                consolidated_memory = await self._consolidate_cluster(cluster, session_id)
                if consolidated_memory:
                    consolidation_results['consolidated_memories'] += 1
        
        # Create semantic abstractions
        semantic_memories = await self._extract_semantic_patterns(memories, session_id)
        consolidation_results['created_semantic_memories'] = len(semantic_memories)
        
        # Calculate efficiency gain
        final_tokens = sum(mem.token_count for mem in memories)  # Updated token counts
        if initial_tokens > 0:
            consolidation_results['token_efficiency_gain'] = (initial_tokens - final_tokens) / initial_tokens
        
        return consolidation_results
    
    async def _consolidate_cluster(self, cluster: List[MemoryNode], session_id: str) -> Optional[MemoryNode]:
        """Consolidate a cluster of related memories into a single memory"""
        
        if len(cluster) < 2:
            return {}
        
        # Combine content intelligently
        combined_content = await self._intelligently_combine_content([mem.content for mem in cluster])
        
        # Merge metadata
        combined_tags = set()
        max_importance = MemoryImportance.MINIMAL
        combined_context = {}
        
        for memory in cluster:
            combined_tags.update(memory.tags)
            if memory.importance.value > max_importance.value:
                max_importance = memory.importance
            combined_context.update(memory.context)
        
        # Create consolidated memory
        consolidated_memory = MemoryNode(
            id=f"consolidated_{uuid.uuid4().hex[:8]}",
            content=combined_content,
            memory_type=MemoryType.SEMANTIC,  # Consolidated memories become semantic
            importance=max_importance,
            tags=combined_tags,
            context=combined_context,
            reasoning_session=session_id
        )
        
        consolidated_memory.tags.add('consolidated')
        consolidated_memory.consolidation_score = 1.0
        
        return consolidated_memory
    
    async def _intelligently_combine_content(self, contents: List[str]) -> str:
        """Intelligently combine multiple content strings"""
        
        # Simple combination - would use more sophisticated NLP in practice
        if len(contents) == 1:
            return contents[0]
        
        # Extract key points from each content
        key_points = []
        for content in contents:
            # Simple extraction of sentences with key indicators
            sentences = content.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and any(keyword in sentence.lower() for keyword in 
                                  ['important', 'key', 'result', 'conclusion', 'therefore']):
                    key_points.append(sentence)
        
        if key_points:
            combined = '. '.join(key_points)
            return f"Consolidated insight: {combined}"
        else:
            # Fallback: summarize all content
            return f"Combined from {len(contents)} related memories: " + " | ".join(contents[:3])
    
    async def _extract_semantic_patterns(self, memories: List[MemoryNode], session_id: str) -> List[MemoryNode]:
        """Extract high-level semantic patterns from memories"""
        
        semantic_memories = []
        
        # Topic extraction
        topic_clusters = await self._extract_topics(memories)
        
        for topic, topic_memories in topic_clusters.items():
            if len(topic_memories) >= 3:  # Minimum memories for semantic pattern
                semantic_content = f"Semantic pattern identified: {topic}. " + \
                                 f"Observed across {len(topic_memories)} memories with consistent themes."
                
                semantic_memory = MemoryNode(
                    id=f"semantic_{uuid.uuid4().hex[:8]}",
                    content=semantic_content,
                    memory_type=MemoryType.SEMANTIC,
                    importance=MemoryImportance.HIGH,
                    tags={topic, 'semantic_pattern'},
                    reasoning_session=session_id
                )
                
                semantic_memories.append(semantic_memory)
        
        return semantic_memories
    
    async def _extract_topics(self, memories: List[MemoryNode]) -> Dict[str, List[MemoryNode]]:
        """Extract topics from memories"""
        
        topic_clusters = defaultdict(list)
        
        # Simple topic extraction based on common tags and keywords
        for memory in memories:
            memory_topics = set()
            
            # Use tags as topics
            memory_topics.update(memory.tags)
            
            # Extract keywords as potential topics
            keywords = await self._extract_keywords(memory.content)
            memory_topics.update(keywords[:3])  # Top 3 keywords
            
            # Assign memory to topic clusters
            for topic in memory_topics:
                topic_clusters[topic].append(memory)
        
        return dict(topic_clusters)
    
    async def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        
        # Simple keyword extraction
        words = content.lower().split()
        
        # Filter stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
                     'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 
                     'been', 'have', 'has', 'had', 'do', 'does', 'did'}
        
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        
        # Count frequency
        word_counts = defaultdict(int)
        for word in keywords:
            word_counts[word] += 1
        
        # Return top keywords
        sorted_keywords = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_keywords[:10]]


class SemanticClusterer:
    """Clusters memories by semantic similarity"""
    
    async def cluster_memories(self, memories: List[MemoryNode], 
                             similarity_threshold: float = 0.7) -> List[List[MemoryNode]]:
        """Cluster memories by semantic similarity"""
        
        if len(memories) < 2:
            return [[mem] for mem in memories]
        
        # Calculate pairwise similarities
        similarity_matrix = await self._calculate_similarity_matrix(memories)
        
        # Simple clustering based on similarity threshold
        clusters = []
        assigned = set()
        
        for i, memory in enumerate(memories):
            if i in assigned:
                continue
            
            cluster = [memory]
            assigned.add(i)
            
            # Find similar memories
            for j, other_memory in enumerate(memories):
                if j != i and j not in assigned and similarity_matrix[i][j] >= similarity_threshold:
                    cluster.append(other_memory)
                    assigned.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    async def _calculate_similarity_matrix(self, memories: List[MemoryNode]) -> np.ndarray:
        """Calculate similarity matrix between memories"""
        
        n = len(memories)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                similarity = await self._calculate_semantic_similarity(memories[i], memories[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity  # Symmetric matrix
        
        return similarity_matrix
    
    async def _calculate_semantic_similarity(self, memory1: MemoryNode, memory2: MemoryNode) -> float:
        """Calculate semantic similarity between two memories"""
        
        # Content similarity
        content_sim = await self._content_similarity(memory1.content, memory2.content)
        
        # Tag overlap
        tag_sim = 0.0
        if memory1.tags and memory2.tags:
            tag_sim = len(memory1.tags & memory2.tags) / len(memory1.tags | memory2.tags)
        
        # Context similarity
        context_sim = memory1._calculate_context_similarity(memory1.context, memory2.context)
        
        # Combined similarity
        combined_sim = (content_sim * 0.6 + tag_sim * 0.3 + context_sim * 0.1)
        
        return combined_sim
    
    async def _content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0