"""
Agent Collaboration Protocols: Phase 6 - Advanced Multi-Agent Communication
Features:
- Blackboard pattern for shared knowledge
- Event-driven agent communication
- Consensus mechanisms for decisions
- Conflict resolution strategies
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict, deque
import uuid
from abc import ABC, abstractmethod
import threading

from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class MessageType(Enum):
    """Types of inter-agent messages"""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    PROPOSAL = "proposal"
    VOTE = "vote"
    CONSENSUS = "consensus"
    CONFLICT = "conflict"
    HEARTBEAT = "heartbeat"


class EventType(Enum):
    """Types of collaboration events"""
    AGENT_JOINED = "agent_joined"
    AGENT_LEFT = "agent_left"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    CONSENSUS_REACHED = "consensus_reached"
    CONFLICT_DETECTED = "conflict_detected"
    TIMEOUT = "timeout"


class MessagePriority(Enum):
    """Message priority levels"""
    URGENT = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class CollaborationMessage:
    """Message structure for agent collaboration"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    message_type: MessageType
    priority: MessagePriority
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    response_timeout: Optional[float] = None
    correlation_id: Optional[str] = None  # For request-response correlation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())


@dataclass
class CollaborationEvent:
    """Event for agent collaboration system"""
    event_id: str
    event_type: EventType
    source_agent: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    handled_by: Set[str] = field(default_factory=set)


@dataclass
class KnowledgeItem:
    """Item in shared knowledge base"""
    key: str
    value: Any
    contributor: str
    timestamp: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    version: int = 1
    tags: Set[str] = field(default_factory=set)
    access_count: int = 0
    
    def update_value(self, new_value: Any, contributor: str, confidence: float = 1.0):
        """Update knowledge item value"""
        self.value = new_value
        self.contributor = contributor
        self.confidence = confidence
        self.version += 1
        self.timestamp = datetime.now()


class MessageRouter:
    """Routes messages between agents efficiently"""
    
    def __init__(self):
        self.agents: Dict[str, 'CollaborativeAgent'] = {}
        self.message_queue = asyncio.Queue()
        self.routing_table: Dict[str, Set[str]] = defaultdict(set)  # topic -> subscribers
        self.message_history: deque = deque(maxlen=1000)
        self.routing_stats = defaultdict(int)
        self.is_running = False
        
    async def start(self):
        """Start the message router"""
        self.is_running = True
        asyncio.create_task(self._process_messages())
        logger.info("Message router started")
    
    async def stop(self):
        """Stop the message router"""
        self.is_running = False
        logger.info("Message router stopped")
    
    def register_agent(self, agent: 'CollaborativeAgent'):
        """Register an agent with the router"""
        self.agents[agent.agent_id] = agent
        logger.debug(f"Registered agent {agent.agent_id} with message router")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            # Remove from all subscriptions
            for topic, subscribers in self.routing_table.items():
                subscribers.discard(agent_id)
        logger.debug(f"Unregistered agent {agent_id} from message router")
    
    def subscribe(self, agent_id: str, topic: str):
        """Subscribe agent to a topic"""
        self.routing_table[topic].add(agent_id)
        logger.debug(f"Agent {agent_id} subscribed to topic {topic}")
    
    def unsubscribe(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic"""
        self.routing_table[topic].discard(agent_id)
        logger.debug(f"Agent {agent_id} unsubscribed from topic {topic}")
    
    async def send_message(self, message: CollaborationMessage):
        """Send message through the router"""
        await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self._route_message(message)
                self.routing_stats['messages_processed'] += 1
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
                self.routing_stats['routing_errors'] += 1
    
    async def _route_message(self, message: CollaborationMessage):
        """Route message to appropriate recipients"""
        recipients = []
        
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.agents:
                recipients = [message.recipient_id]
        else:
            # Broadcast or topic-based message
            topic = message.metadata.get('topic')
            if topic and topic in self.routing_table:
                recipients = list(self.routing_table[topic])
            else:
                # Broadcast to all agents
                recipients = list(self.agents.keys())
        
        # Deliver messages
        delivery_tasks = []
        for recipient_id in recipients:
            if recipient_id in self.agents and recipient_id != message.sender_id:
                agent = self.agents[recipient_id]
                delivery_tasks.append(agent._receive_message(message))
        
        if delivery_tasks:
            await asyncio.gather(*delivery_tasks, return_exceptions=True)
        
        # Store in history
        self.message_history.append({
            'message_id': message.message_id,
            'sender': message.sender_id,
            'recipients': recipients,
            'type': message.message_type.value,
            'timestamp': message.timestamp,
            'delivered': len(delivery_tasks)
        })
    
    def get_routing_stats(self) -> Dict[str, Any]:
        """Get routing statistics"""
        return {
            'registered_agents': len(self.agents),
            'active_topics': len(self.routing_table),
            'total_subscriptions': sum(len(subs) for subs in self.routing_table.values()),
            'messages_in_queue': self.message_queue.qsize(),
            'processing_stats': dict(self.routing_stats),
            'message_history_size': len(self.message_history)
        }


class SharedBlackboard:
    """Shared knowledge space using blackboard pattern"""
    
    def __init__(self):
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.knowledge_lock = threading.RLock()
        self.subscribers: Dict[str, Set[str]] = defaultdict(set)  # key -> agent_ids
        self.access_log: deque = deque(maxlen=1000)
        
    async def write(self, key: str, value: Any, contributor: str, 
                   confidence: float = 1.0, tags: Set[str] = None):
        """Write knowledge to blackboard"""
        with self.knowledge_lock:
            if key in self.knowledge_base:
                # Update existing knowledge
                self.knowledge_base[key].update_value(value, contributor, confidence)
            else:
                # Create new knowledge item
                self.knowledge_base[key] = KnowledgeItem(
                    key=key,
                    value=value,
                    contributor=contributor,
                    confidence=confidence,
                    tags=tags or set()
                )
            
            knowledge_item = self.knowledge_base[key]
            
            # Log access
            self.access_log.append({
                'action': 'write',
                'key': key,
                'contributor': contributor,
                'timestamp': datetime.now(),
                'version': knowledge_item.version
            })
            
            # Notify subscribers
            await self._notify_subscribers(key, 'updated', knowledge_item)
            
            logger.debug(f"Knowledge written: {key} by {contributor} (v{knowledge_item.version})")
    
    async def read(self, key: str, reader: str) -> Optional[Any]:
        """Read knowledge from blackboard"""
        with self.knowledge_lock:
            if key in self.knowledge_base:
                knowledge_item = self.knowledge_base[key]
                knowledge_item.access_count += 1
                
                # Log access
                self.access_log.append({
                    'action': 'read',
                    'key': key,
                    'reader': reader,
                    'timestamp': datetime.now(),
                    'version': knowledge_item.version
                })
                
                logger.debug(f"Knowledge read: {key} by {reader}")
                return knowledge_item.value
            
            return {}
    
    async def query(self, predicate: Callable[[KnowledgeItem], bool], 
                   requester: str) -> Dict[str, Any]:
        """Query blackboard with custom predicate"""
        results = {}
        
        with self.knowledge_lock:
            for key, item in self.knowledge_base.items():
                if predicate(item):
                    results[key] = item.value
                    item.access_count += 1
        
        # Log query
        self.access_log.append({
            'action': 'query',
            'requester': requester,
            'results_count': len(results),
            'timestamp': datetime.now()
        })
        
        logger.debug(f"Knowledge query by {requester}: {len(results)} results")
        return results
    
    async def subscribe(self, key_pattern: str, agent_id: str):
        """Subscribe to knowledge updates"""
        self.subscribers[key_pattern].add(agent_id)
        logger.debug(f"Agent {agent_id} subscribed to knowledge pattern {key_pattern}")
    
    async def unsubscribe(self, key_pattern: str, agent_id: str):
        """Unsubscribe from knowledge updates"""
        self.subscribers[key_pattern].discard(agent_id)
        logger.debug(f"Agent {agent_id} unsubscribed from knowledge pattern {key_pattern}")
    
    async def _notify_subscribers(self, key: str, action: str, item: KnowledgeItem):
        """Notify subscribers of knowledge changes"""
        # For simplicity, using exact key matching
        # In practice, would support pattern matching
        if key in self.subscribers:
            # Would send notifications to subscribers here
            logger.debug(f"Notifying {len(self.subscribers[key])} subscribers about {key} {action}")
    
    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get blackboard statistics"""
        with self.knowledge_lock:
            total_items = len(self.knowledge_base)
            total_accesses = sum(item.access_count for item in self.knowledge_base.values())
            
            # Analyze by contributor
            contributor_stats = defaultdict(int)
            confidence_stats = []
            
            for item in self.knowledge_base.values():
                contributor_stats[item.contributor] += 1
                confidence_stats.append(item.confidence)
            
            return {
                'total_knowledge_items': total_items,
                'total_accesses': total_accesses,
                'average_confidence': sum(confidence_stats) / len(confidence_stats) if confidence_stats else 0,
                'contributors': dict(contributor_stats),
                'total_subscribers': sum(len(subs) for subs in self.subscribers.values()),
                'access_log_size': len(self.access_log)
            }


class ConsensusManager:
    """Manages consensus protocols for multi-agent decisions"""
    
    def __init__(self, consensus_threshold: float = 0.6):
        self.consensus_threshold = consensus_threshold
        self.active_consensuses: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []
        
    async def initiate_consensus(self, consensus_id: str, proposal: Any, 
                               participants: List[str], timeout: float = 60.0) -> str:
        """Initiate a consensus process"""
        
        self.active_consensuses[consensus_id] = {
            'proposal': proposal,
            'participants': set(participants),
            'votes': {},
            'start_time': datetime.now(),
            'timeout': timeout,
            'status': 'active'
        }
        
        logger.info(f"Consensus initiated: {consensus_id} with {len(participants)} participants")
        return consensus_id
    
    async def cast_vote(self, consensus_id: str, agent_id: str, 
                       vote: bool, reasoning: str = "") -> bool:
        """Cast a vote in consensus process"""
        
        if consensus_id not in self.active_consensuses:
            logger.warning(f"Consensus {consensus_id} not found")
            return False
        
        consensus = self.active_consensuses[consensus_id]
        
        if consensus['status'] != 'active':
            logger.warning(f"Consensus {consensus_id} is not active")
            return False
        
        if agent_id not in consensus['participants']:
            logger.warning(f"Agent {agent_id} not a participant in consensus {consensus_id}")
            return False
        
        # Record vote
        consensus['votes'][agent_id] = {
            'vote': vote,
            'reasoning': reasoning,
            'timestamp': datetime.now()
        }
        
        logger.debug(f"Vote cast by {agent_id} in consensus {consensus_id}: {vote}")
        
        # Check if consensus reached
        await self._check_consensus_completion(consensus_id)
        
        return True
    
    async def _check_consensus_completion(self, consensus_id: str):
        """Check if consensus has been reached or timed out"""
        
        consensus = self.active_consensuses[consensus_id]
        
        # Check timeout
        elapsed_time = (datetime.now() - consensus['start_time']).total_seconds()
        if elapsed_time > consensus['timeout']:
            await self._complete_consensus(consensus_id, 'timeout')
            return {}
        
        # Check if all participants voted
        total_participants = len(consensus['participants'])
        total_votes = len(consensus['votes'])
        
        if total_votes == total_participants:
            # Calculate consensus
            positive_votes = sum(1 for vote_data in consensus['votes'].values() if vote_data['vote'])
            consensus_ratio = positive_votes / total_participants
            
            if consensus_ratio >= self.consensus_threshold:
                await self._complete_consensus(consensus_id, 'consensus_reached')
            else:
                await self._complete_consensus(consensus_id, 'consensus_failed')
    
    async def _complete_consensus(self, consensus_id: str, outcome: str):
        """Complete consensus process"""
        
        consensus = self.active_consensuses[consensus_id]
        consensus['status'] = 'completed'
        consensus['outcome'] = outcome
        consensus['completion_time'] = datetime.now()
        
        # Calculate final statistics
        positive_votes = sum(1 for vote_data in consensus['votes'].values() if vote_data['vote'])
        total_votes = len(consensus['votes'])
        final_ratio = positive_votes / total_votes if total_votes > 0 else 0
        
        consensus['final_vote_ratio'] = final_ratio
        consensus['positive_votes'] = positive_votes
        consensus['total_votes'] = total_votes
        
        # Move to history
        self.consensus_history.append(consensus.copy())
        del self.active_consensuses[consensus_id]
        
        logger.info(f"Consensus {consensus_id} completed: {outcome} "
                   f"({positive_votes}/{total_votes} = {final_ratio:.2%})")
        
        global_metrics.incr(f"consensus.{outcome}")
    
    def get_consensus_status(self, consensus_id: str) -> Optional[Dict[str, Any]]:
        """Get status of consensus process"""
        
        if consensus_id in self.active_consensuses:
            consensus = self.active_consensuses[consensus_id]
            return {
                'consensus_id': consensus_id,
                'status': consensus['status'],
                'participants': len(consensus['participants']),
                'votes_received': len(consensus['votes']),
                'time_remaining': max(0, consensus['timeout'] - 
                                    (datetime.now() - consensus['start_time']).total_seconds())
            }
        
        # Check history
        for historical_consensus in reversed(self.consensus_history):
            if historical_consensus.get('consensus_id') == consensus_id:
                return {
                    'consensus_id': consensus_id,
                    'status': historical_consensus['status'],
                    'outcome': historical_consensus['outcome'],
                    'final_vote_ratio': historical_consensus['final_vote_ratio']
                }
        
        return {}
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus manager statistics"""
        
        # Analyze historical consensuses
        if not self.consensus_history:
            return {
                'active_consensuses': len(self.active_consensuses),
                'total_completed': 0
            }
        
        outcomes = defaultdict(int)
        completion_times = []
        participation_rates = []
        
        for consensus in self.consensus_history:
            outcomes[consensus['outcome']] += 1
            
            if 'completion_time' in consensus and 'start_time' in consensus:
                duration = (consensus['completion_time'] - consensus['start_time']).total_seconds()
                completion_times.append(duration)
            
            if 'total_votes' in consensus and 'participants' in consensus:
                participation_rate = consensus['total_votes'] / len(consensus['participants'])
                participation_rates.append(participation_rate)
        
        return {
            'active_consensuses': len(self.active_consensuses),
            'total_completed': len(self.consensus_history),
            'outcomes': dict(outcomes),
            'average_completion_time': sum(completion_times) / len(completion_times) if completion_times else 0,
            'average_participation_rate': sum(participation_rates) / len(participation_rates) if participation_rates else 0,
            'consensus_threshold': self.consensus_threshold
        }


class ConflictResolver:
    """Resolves conflicts between agents"""
    
    def __init__(self):
        self.resolution_strategies = {
            'priority_based': self._resolve_by_priority,
            'reputation_based': self._resolve_by_reputation,
            'voting': self._resolve_by_voting,
            'compromise': self._resolve_by_compromise,
            'random': self._resolve_randomly
        }
        self.conflict_history: List[Dict[str, Any]] = []
        
    async def detect_conflict(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect potential conflicts in agent interactions"""
        
        conflicts = []
        
        # Resource conflicts
        if 'resource_requests' in context:
            resource_conflicts = self._detect_resource_conflicts(context['resource_requests'])
            conflicts.extend(resource_conflicts)
        
        # Goal conflicts  
        if 'agent_goals' in context:
            goal_conflicts = self._detect_goal_conflicts(context['agent_goals'])
            conflicts.extend(goal_conflicts)
        
        # Priority conflicts
        if 'task_priorities' in context:
            priority_conflicts = self._detect_priority_conflicts(context['task_priorities'])
            conflicts.extend(priority_conflicts)
        
        if conflicts:
            conflict_id = str(uuid.uuid4())
            conflict_summary = {
                'conflict_id': conflict_id,
                'conflicts': conflicts,
                'timestamp': datetime.now(),
                'context': context
            }
            
            logger.warning(f"Conflict detected: {conflict_id} with {len(conflicts)} issues")
            return conflict_summary
        
        return {}
    
    def _detect_resource_conflicts(self, resource_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts in resource allocation"""
        conflicts = []
        resource_usage = defaultdict(list)
        
        # Group requests by resource
        for request in resource_requests:
            resource_id = request.get('resource_id')
            if resource_id:
                resource_usage[resource_id].append(request)
        
        # Check for over-allocation
        for resource_id, requests in resource_usage.items():
            if len(requests) > 1:
                total_requested = sum(req.get('amount', 1) for req in requests)
                available = requests[0].get('available', 1)  # Assume same for all requests
                
                if total_requested > available:
                    conflicts.append({
                        'type': 'resource_conflict',
                        'resource_id': resource_id,
                        'total_requested': total_requested,
                        'available': available,
                        'conflicting_agents': [req.get('agent_id') for req in requests]
                    })
        
        return conflicts
    
    def _detect_goal_conflicts(self, agent_goals: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect conflicts between agent goals"""
        conflicts = []
        
        # Simple goal conflict detection based on keywords
        goal_keywords = {}
        for agent_id, goal_info in agent_goals.items():
            goal_text = goal_info.get('description', '').lower()
            keywords = set(goal_text.split())
            goal_keywords[agent_id] = keywords
        
        # Check for conflicting keywords
        conflict_pairs = [
            ('increase', 'decrease'),
            ('maximize', 'minimize'),
            ('add', 'remove'),
            ('start', 'stop')
        ]
        
        agents = list(agent_goals.keys())
        for i, agent1 in enumerate(agents):
            for agent2 in agents[i+1:]:
                keywords1 = goal_keywords.get(agent1, set())
                keywords2 = goal_keywords.get(agent2, set())
                
                for word1, word2 in conflict_pairs:
                    if word1 in keywords1 and word2 in keywords2:
                        conflicts.append({
                            'type': 'goal_conflict',
                            'agent1': agent1,
                            'agent2': agent2,
                            'conflict_keywords': (word1, word2)
                        })
        
        return conflicts
    
    def _detect_priority_conflicts(self, task_priorities: Dict[str, int]) -> List[Dict[str, Any]]:
        """Detect priority conflicts"""
        conflicts = []
        
        # Check for tasks with same high priority
        high_priority_tasks = [
            task_id for task_id, priority in task_priorities.items() 
            if priority <= 2  # High priority (1-2)
        ]
        
        if len(high_priority_tasks) > 3:  # Too many high priority tasks
            conflicts.append({
                'type': 'priority_conflict',
                'description': 'Too many high priority tasks',
                'high_priority_count': len(high_priority_tasks),
                'tasks': high_priority_tasks
            })
        
        return conflicts
    
    async def resolve_conflict(self, conflict: Dict[str, Any], 
                              strategy: str = 'priority_based') -> Dict[str, Any]:
        """Resolve conflict using specified strategy"""
        
        if strategy not in self.resolution_strategies:
            logger.error(f"Unknown resolution strategy: {strategy}")
            strategy = 'random'  # Fallback
        
        resolver = self.resolution_strategies[strategy]
        resolution = await resolver(conflict)
        
        # Record resolution
        resolution_record = {
            'conflict_id': conflict['conflict_id'],
            'strategy_used': strategy,
            'resolution': resolution,
            'timestamp': datetime.now(),
            'original_conflict': conflict
        }
        
        self.conflict_history.append(resolution_record)
        
        logger.info(f"Conflict {conflict['conflict_id']} resolved using {strategy}")
        return resolution
    
    async def _resolve_by_priority(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict based on agent/task priorities"""
        
        resolutions = []
        
        for conflict_item in conflict['conflicts']:
            if conflict_item['type'] == 'resource_conflict':
                # Assign resource to highest priority agent
                conflicting_agents = conflict_item['conflicting_agents']
                # In practice, would look up actual agent priorities
                chosen_agent = conflicting_agents[0]  # Simplified
                
                resolutions.append({
                    'conflict_type': 'resource_conflict',
                    'resource_id': conflict_item['resource_id'],
                    'assigned_to': chosen_agent,
                    'denied_agents': conflicting_agents[1:]
                })
        
        return {'resolutions': resolutions}
    
    async def _resolve_by_reputation(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict based on agent reputation"""
        # Placeholder implementation
        return await self._resolve_by_priority(conflict)
    
    async def _resolve_by_voting(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through voting mechanism"""
        # Would integrate with ConsensusManager
        return {'resolution_method': 'voting', 'status': 'initiated'}
    
    async def _resolve_by_compromise(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict through compromise"""
        
        resolutions = []
        
        for conflict_item in conflict['conflicts']:
            if conflict_item['type'] == 'resource_conflict':
                # Split resources proportionally
                conflicting_agents = conflict_item['conflicting_agents']
                available = conflict_item['available']
                allocation_per_agent = available / len(conflicting_agents)
                
                resolutions.append({
                    'conflict_type': 'resource_conflict',
                    'resource_id': conflict_item['resource_id'],
                    'allocation': {agent: allocation_per_agent for agent in conflicting_agents}
                })
        
        return {'resolutions': resolutions}
    
    async def _resolve_randomly(self, conflict: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve conflict randomly"""
        import random
        
        resolutions = []
        
        for conflict_item in conflict['conflicts']:
            if conflict_item['type'] == 'resource_conflict':
                conflicting_agents = conflict_item['conflicting_agents']
                chosen_agent = random.choice(conflicting_agents)
                
                resolutions.append({
                    'conflict_type': 'resource_conflict',
                    'resource_id': conflict_item['resource_id'],
                    'assigned_to': chosen_agent,
                    'method': 'random_selection'
                })
        
        return {'resolutions': resolutions}
    
    def get_conflict_stats(self) -> Dict[str, Any]:
        """Get conflict resolution statistics"""
        
        if not self.conflict_history:
            return {'total_conflicts': 0}
        
        strategies_used = defaultdict(int)
        resolution_times = []
        
        for record in self.conflict_history:
            strategies_used[record['strategy_used']] += 1
            
            # Calculate resolution time if available
            conflict_time = record['original_conflict']['timestamp']
            resolution_time = record['timestamp']
            duration = (resolution_time - conflict_time).total_seconds()
            resolution_times.append(duration)
        
        return {
            'total_conflicts': len(self.conflict_history),
            'strategies_used': dict(strategies_used),
            'average_resolution_time': sum(resolution_times) / len(resolution_times) if resolution_times else 0,
            'available_strategies': list(self.resolution_strategies.keys())
        }


class CollaborativeAgent(BaseAgent):
    """Enhanced agent with collaboration capabilities"""
    
    def __init__(self, agent_id: str, router: MessageRouter = None):
        super().__init__(agent_id)
        self.agent_id = agent_id
        self.router = router
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.event_handlers: Dict[EventType, Callable] = {}
        self.subscriptions: Set[str] = set()
        self.collaboration_stats = defaultdict(int)
        self.pending_responses: Dict[str, asyncio.Future] = {}
        
        # Register with router if provided
        if self.router:
            self.router.register_agent(self)
        
        # Set up default handlers
        self._setup_default_handlers()
    
    def _setup_default_handlers(self):
        """Set up default message handlers"""
        self.message_handlers[MessageType.HEARTBEAT] = self._handle_heartbeat
        self.message_handlers[MessageType.REQUEST] = self._handle_request
        self.message_handlers[MessageType.RESPONSE] = self._handle_response
    
    async def send_message(self, recipient_id: Optional[str], message_type: MessageType,
                          content: Any, priority: MessagePriority = MessagePriority.NORMAL,
                          requires_response: bool = False, timeout: Optional[float] = None) -> Optional[Any]:
        """Send message to another agent"""
        
        if not self.router:
            logger.error("No router available for sending messages")
            return {}
        
        message = CollaborationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=recipient_id,
            message_type=message_type,
            priority=priority,
            content=content,
            requires_response=requires_response,
            response_timeout=timeout
        )
        
        if requires_response:
            # Create future for response
            response_future = asyncio.Future()
            self.pending_responses[message.message_id] = response_future
            
            # Send message
            await self.router.send_message(message)
            
            try:
                # Wait for response
                response = await asyncio.wait_for(response_future, timeout=timeout or 30.0)
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Message {message.message_id} timed out waiting for response")
                del self.pending_responses[message.message_id]
                return {}
        else:
            # Fire and forget
            await self.router.send_message(message)
            return {}
    
    async def broadcast_message(self, message_type: MessageType, content: Any,
                               topic: str = None, priority: MessagePriority = MessagePriority.NORMAL):
        """Broadcast message to all agents or topic subscribers"""
        
        message = CollaborationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            message_type=message_type,
            priority=priority,
            content=content,
            metadata={'topic': topic} if topic else {}
        )
        
        if self.router:
            await self.router.send_message(message)
        
        self.collaboration_stats['messages_sent'] += 1
    
    async def _receive_message(self, message: CollaborationMessage):
        """Receive and process message"""
        
        self.collaboration_stats['messages_received'] += 1
        
        # Handle response messages
        if (message.message_type == MessageType.RESPONSE and 
            message.correlation_id in self.pending_responses):
            
            future = self.pending_responses.pop(message.correlation_id)
            if not future.done():
                future.set_result(message.content)
            return {}
        
        # Handle other message types
        if message.message_type in self.message_handlers:
            try:
                handler = self.message_handlers[message.message_type]
                await handler(message)
            except Exception as e:
                logger.error(f"Error handling message {message.message_id}: {e}")
                self.collaboration_stats['message_handling_errors'] += 1
        else:
            logger.debug(f"No handler for message type {message.message_type}")
    
    async def _handle_heartbeat(self, message: CollaborationMessage):
        """Handle heartbeat message"""
        # Send heartbeat response
        response = CollaborationMessage(
            message_id=str(uuid.uuid4()),
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type=MessageType.RESPONSE,
            priority=MessagePriority.HIGH,
            content={'status': 'alive', 'timestamp': datetime.now()},
            correlation_id=message.message_id
        )
        
        if self.router:
            await self.router.send_message(response)
    
    async def _handle_request(self, message: CollaborationMessage):
        """Handle request message"""
        # Default request handler - should be overridden by subclasses
        response_content = {'status': 'received', 'message': 'Request processed'}
        
        if message.requires_response:
            response = CollaborationMessage(
                message_id=str(uuid.uuid4()),
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type=MessageType.RESPONSE,
                priority=message.priority,
                content=response_content,
                correlation_id=message.message_id
            )
            
            if self.router:
                await self.router.send_message(response)
    
    async def _handle_response(self, message: CollaborationMessage):
        """Handle response message"""
        logger.debug(f"Received response from {message.sender_id}")
    
    def subscribe_to_topic(self, topic: str):
        """Subscribe to topic for receiving broadcasts"""
        if self.router:
            self.router.subscribe(self.agent_id, topic)
            self.subscriptions.add(topic)
    
    def unsubscribe_from_topic(self, topic: str):
        """Unsubscribe from topic"""
        if self.router:
            self.router.unsubscribe(self.agent_id, topic)
            self.subscriptions.discard(topic)
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics for this agent"""
        return {
            'agent_id': self.agent_id,
            'subscriptions': list(self.subscriptions),
            'pending_responses': len(self.pending_responses),
            'stats': dict(self.collaboration_stats)
        }


class CollaborationOrchestrator:
    """Main orchestrator for agent collaboration protocols"""
    
    def __init__(self):
        self.message_router = MessageRouter()
        self.blackboard = SharedBlackboard()
        self.consensus_manager = ConsensusManager()
        self.conflict_resolver = ConflictResolver()
        self.agents: Dict[str, CollaborativeAgent] = {}
        self.event_log: deque = deque(maxlen=1000)
        
    async def start(self):
        """Start collaboration orchestrator"""
        await self.message_router.start()
        logger.info("Collaboration orchestrator started")
    
    async def stop(self):
        """Stop collaboration orchestrator"""
        await self.message_router.stop()
        logger.info("Collaboration orchestrator stopped")
    
    def register_agent(self, agent: CollaborativeAgent):
        """Register collaborative agent"""
        self.agents[agent.agent_id] = agent
        agent.router = self.message_router
        self.message_router.register_agent(agent)
        
        # Log event
        event = CollaborationEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.AGENT_JOINED,
            source_agent=agent.agent_id,
            data={'timestamp': datetime.now()}
        )
        self.event_log.append(event)
        
        logger.info(f"Registered collaborative agent: {agent.agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            self.message_router.unregister_agent(agent_id)
            
            # Log event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.AGENT_LEFT,
                source_agent=agent_id,
                data={'timestamp': datetime.now()}
            )
            self.event_log.append(event)
            
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def initiate_group_consensus(self, proposal: Any, participants: List[str],
                                     timeout: float = 120.0) -> str:
        """Initiate group consensus process"""
        consensus_id = str(uuid.uuid4())
        
        await self.consensus_manager.initiate_consensus(
            consensus_id, proposal, participants, timeout
        )
        
        # Notify participants
        for agent_id in participants:
            if agent_id in self.agents:
                agent = self.agents[agent_id]
                await agent.send_message(
                    None, MessageType.PROPOSAL,
                    {
                        'consensus_id': consensus_id,
                        'proposal': proposal,
                        'timeout': timeout
                    },
                    MessagePriority.HIGH
                )
        
        return consensus_id
    
    async def detect_and_resolve_conflicts(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect and resolve conflicts in agent collaboration"""
        
        conflict = await self.conflict_resolver.detect_conflict(context)
        
        if conflict:
            resolution = await self.conflict_resolver.resolve_conflict(conflict)
            
            # Log event
            event = CollaborationEvent(
                event_id=str(uuid.uuid4()),
                event_type=EventType.CONFLICT_DETECTED,
                source_agent="system",
                data={
                    'conflict_id': conflict['conflict_id'],
                    'resolution': resolution
                }
            )
            self.event_log.append(event)
            
            return {
                'conflict': conflict,
                'resolution': resolution
            }
        
        return {}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive collaboration system status"""
        
        return {
            'agents': {
                'total_agents': len(self.agents),
                'agent_stats': {
                    agent_id: agent.get_collaboration_stats()
                    for agent_id, agent in self.agents.items()
                }
            },
            'message_router': self.message_router.get_routing_stats(),
            'blackboard': self.blackboard.get_knowledge_stats(),
            'consensus_manager': self.consensus_manager.get_consensus_stats(),
            'conflict_resolver': self.conflict_resolver.get_conflict_stats(),
            'event_log_size': len(self.event_log),
            'system_uptime': datetime.now()  # Would track actual uptime
        }


if __name__ == "__main__":
    async def demo_collaboration_protocols():
        """Demonstrate collaboration protocols"""
        
        orchestrator = CollaborationOrchestrator()
        await orchestrator.start()
        
        print("=" * 80)
        print("AGENT COLLABORATION PROTOCOLS DEMONSTRATION")
        print("=" * 80)
        
        # Create collaborative agents
        agents = []
        for i in range(4):
            agent = CollaborativeAgent(f"agent_{i:02d}")
            agents.append(agent)
            orchestrator.register_agent(agent)
        
        print(f"Created {len(agents)} collaborative agents")
        
        # Demonstrate messaging
        print("\\nTESTING AGENT MESSAGING:")
        print("-" * 30)
        
        # Direct message
        response = await agents[0].send_message(
            agents[1].agent_id,
            MessageType.REQUEST,
            {"task": "process_data", "data": [1, 2, 3]},
            requires_response=True,
            timeout=5.0
        )
        print(f"Direct message response: {response}")
        
        # Broadcast message
        await agents[0].broadcast_message(
            MessageType.NOTIFICATION,
            {"announcement": "System maintenance in 1 hour"},
            topic="system_alerts"
        )
        print("Broadcast message sent")
        
        # Demonstrate blackboard knowledge sharing
        print("\\nTESTING BLACKBOARD KNOWLEDGE SHARING:")
        print("-" * 42)
        
        # Write knowledge
        await orchestrator.blackboard.write(
            "task_progress", 
            {"completed": 5, "total": 10},
            agents[0].agent_id
        )
        
        await orchestrator.blackboard.write(
            "system_load",
            {"cpu": 0.75, "memory": 0.60},
            agents[1].agent_id
        )
        
        # Read knowledge
        progress = await orchestrator.blackboard.read("task_progress", agents[2].agent_id)
        load = await orchestrator.blackboard.read("system_load", agents[2].agent_id)
        
        print(f"Task progress: {progress}")
        print(f"System load: {load}")
        
        # Demonstrate consensus
        print("\\nTESTING CONSENSUS MECHANISM:")
        print("-" * 32)
        
        # Initiate consensus
        consensus_id = await orchestrator.initiate_group_consensus(
            proposal={"action": "scale_up", "instances": 3},
            participants=[agent.agent_id for agent in agents],
            timeout=10.0
        )
        
        print(f"Consensus initiated: {consensus_id}")
        
        # Simulate voting
        import random
        for i, agent in enumerate(agents):
            vote = random.choice([True, True, True, False])  # 75% yes
            reasoning = f"Agent {i} analysis indicates {'approval' if vote else 'rejection'}"
            
            await orchestrator.consensus_manager.cast_vote(
                consensus_id, agent.agent_id, vote, reasoning
            )
            print(f"Agent {agent.agent_id} voted: {vote}")
        
        # Check consensus result
        await asyncio.sleep(1)  # Allow processing time
        consensus_status = orchestrator.consensus_manager.get_consensus_status(consensus_id)
        if consensus_status:
            print(f"Consensus result: {consensus_status}")
        
        # Demonstrate conflict detection and resolution
        print("\\nTESTING CONFLICT RESOLUTION:")
        print("-" * 33)
        
        conflict_context = {
            'resource_requests': [
                {'agent_id': 'agent_00', 'resource_id': 'cpu', 'amount': 8, 'available': 10},
                {'agent_id': 'agent_01', 'resource_id': 'cpu', 'amount': 6, 'available': 10},
                {'agent_id': 'agent_02', 'resource_id': 'cpu', 'amount': 4, 'available': 10}
            ],
            'agent_goals': {
                'agent_00': {'description': 'maximize throughput'},
                'agent_01': {'description': 'minimize latency'}
            }
        }
        
        conflict_resolution = await orchestrator.detect_and_resolve_conflicts(conflict_context)
        
        if conflict_resolution:
            print("Conflict detected and resolved:")
            print(f"  Conflict ID: {conflict_resolution['conflict']['conflict_id']}")
            print(f"  Resolutions: {len(conflict_resolution['resolution']['resolutions'])}")
        else:
            print("No conflicts detected")
        
        # Show system status
        print("\\nSYSTEM STATUS:")
        print("-" * 15)
        
        status = await orchestrator.get_system_status()
        
        print(f"Total agents: {status['agents']['total_agents']}")
        print(f"Messages processed: {status['message_router']['processing_stats'].get('messages_processed', 0)}")
        print(f"Knowledge items: {status['blackboard']['total_knowledge_items']}")
        print(f"Consensus processes: {status['consensus_manager']['total_completed']}")
        print(f"Conflicts resolved: {status['conflict_resolver']['total_conflicts']}")
        print(f"System events: {status['event_log_size']}")
        
        # Show individual agent statistics
        print("\\nAGENT STATISTICS:")
        print("-" * 18)
        
        for agent_id, agent_stats in status['agents']['agent_stats'].items():
            stats = agent_stats['stats']
            print(f"{agent_id}: {stats.get('messages_sent', 0)} sent, "
                  f"{stats.get('messages_received', 0)} received")
        
        await orchestrator.stop()
    
    # Run demonstration
    asyncio.run(demo_collaboration_protocols())