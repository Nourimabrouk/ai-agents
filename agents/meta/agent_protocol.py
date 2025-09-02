"""
Agent Communication Protocol: Enables efficient inter-agent communication
Supports both synchronous and asynchronous message passing with parallel execution
"""
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages agents can exchange"""
    REQUEST = "request"
    RESPONSE = "response"
    BROADCAST = "broadcast"
    NOTIFICATION = "notification"
    QUERY = "query"
    COMMAND = "command"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


@dataclass
class Message:
    """Universal message format for agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: MessageType = MessageType.REQUEST
    sender: str = ""
    recipient: str = ""
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: Optional[int] = None  # Time to live in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        data = asdict(self)
        data['type'] = self.type.value
        data['priority'] = self.priority.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        data['type'] = MessageType(data['type'])
        data['priority'] = MessagePriority(data['priority'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class MessageBus:
    """
    Central message bus for agent communication
    Supports pub/sub, request/response, and broadcast patterns
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.agents: Dict[str, 'AgentInterface'] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_history: List[Message] = []
        self.routing_table: Dict[str, str] = {}
        self._running = False
        self._processor_task = None
    
    async def start(self):
        """Start the message bus"""
        if not self._running:
            self._running = True
            self._processor_task = asyncio.create_task(self._process_messages())
            logger.info("Message bus started")
    
    async def stop(self):
        """Stop the message bus"""
        self._running = False
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
        logger.info("Message bus stopped")
    
    def register_agent(self, agent_id: str, agent: 'AgentInterface'):
        """Register an agent with the message bus"""
        self.agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    def subscribe(self, topic: str, handler: Callable):
        """Subscribe to a topic"""
        self.subscribers[topic].append(handler)
        logger.debug(f"Subscribed to topic: {topic}")
    
    def unsubscribe(self, topic: str, handler: Callable):
        """Unsubscribe from a topic"""
        if handler in self.subscribers[topic]:
            self.subscribers[topic].remove(handler)
            logger.debug(f"Unsubscribed from topic: {topic}")
    
    async def send(self, message: Message) -> Optional[Any]:
        """Send a message"""
        # Add to history
        self.message_history.append(message)
        
        # Check TTL
        if message.ttl and message.ttl > 0:
            asyncio.create_task(self._expire_message(message, message.ttl))
        
        # Route based on message type
        if message.type == MessageType.REQUEST:
            return await self._handle_request(message)
        elif message.type == MessageType.BROADCAST:
            await self._handle_broadcast(message)
        elif message.type == MessageType.NOTIFICATION:
            await self._handle_notification(message)
        elif message.type == MessageType.QUERY:
            return await self._handle_query(message)
        elif message.type == MessageType.COMMAND:
            await self._handle_command(message)
        elif message.type == MessageType.EVENT:
            await self._handle_event(message)
        else:
            await self.message_queue.put(message)
    
    async def _process_messages(self):
        """Process messages from the queue"""
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                
                # Process based on priority
                if message.priority == MessagePriority.CRITICAL:
                    await self._process_critical_message(message)
                else:
                    await self._process_normal_message(message)
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_request(self, message: Message) -> Any:
        """Handle request/response pattern"""
        if message.recipient in self.agents:
            agent = self.agents[message.recipient]
            
            # Create response future
            response_future = asyncio.Future()
            self.pending_responses[message.id] = response_future
            
            # Send to agent
            try:
                response = await agent.handle_message(message)
                response_future.set_result(response)
                return response
            except Exception as e:
                response_future.set_exception(e)
                raise
            finally:
                del self.pending_responses[message.id]
        else:
            raise ValueError(f"Agent {message.recipient} not found")
    
    async def _handle_broadcast(self, message: Message):
        """Handle broadcast messages"""
        tasks = []
        for agent_id, agent in self.agents.items():
            if agent_id != message.sender:
                tasks.append(agent.handle_message(message))
        
        # Execute all broadcasts in parallel
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _handle_notification(self, message: Message):
        """Handle notification messages"""
        topic = message.metadata.get("topic", "general")
        handlers = self.subscribers.get(topic, [])
        
        tasks = []
        for handler in handlers:
            tasks.append(handler(message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _handle_query(self, message: Message) -> Any:
        """Handle query messages"""
        query_type = message.metadata.get("query_type", "general")
        
        if query_type == "agent_status":
            return await self._query_agent_status(message)
        elif query_type == "message_history":
            return await self._query_message_history(message)
        elif query_type == "routing_table":
            return self.routing_table
        else:
            # Forward to specific agent
            return await self._handle_request(message)
    
    async def _handle_command(self, message: Message):
        """Handle command messages"""
        command = message.content.get("command")
        args = message.content.get("args", {})
        
        if command == "pause":
            await self._pause_agent(message.recipient)
        elif command == "resume":
            await self._resume_agent(message.recipient)
        elif command == "restart":
            await self._restart_agent(message.recipient)
        else:
            # Forward to agent
            if message.recipient in self.agents:
                await self.agents[message.recipient].handle_message(message)
    
    async def _handle_event(self, message: Message):
        """Handle event messages"""
        event_type = message.metadata.get("event_type", "general")
        
        # Notify all subscribers
        await self._handle_notification(message)
        
        # Log important events
        if event_type in ["error", "warning", "critical"]:
            logger.warning(f"Event {event_type}: {message.content}")
    
    async def _process_critical_message(self, message: Message):
        """Process critical priority messages immediately"""
        logger.info(f"Processing critical message: {message.id}")
        
        # Skip queue and process immediately
        if message.recipient in self.agents:
            await self.agents[message.recipient].handle_message(message)
    
    async def _process_normal_message(self, message: Message):
        """Process normal priority messages"""
        if message.recipient in self.agents:
            await self.agents[message.recipient].handle_message(message)
        elif message.recipient == "*":
            # Broadcast to all
            await self._handle_broadcast(message)
    
    async def _expire_message(self, message: Message, ttl: int):
        """Expire a message after TTL"""
        await asyncio.sleep(ttl)
        
        # Check if message is still pending
        if message.id in self.pending_responses:
            future = self.pending_responses[message.id]
            future.set_exception(TimeoutError(f"Message {message.id} expired"))
            del self.pending_responses[message.id]
    
    async def _query_agent_status(self, message: Message) -> Dict[str, Any]:
        """Query status of all agents"""
        status = {}
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'get_status'):
                status[agent_id] = await agent.get_status()
            else:
                status[agent_id] = {"status": "active"}
        return status
    
    async def _query_message_history(self, message: Message) -> List[Dict[str, Any]]:
        """Query message history"""
        filters = message.content or {}
        history = []
        
        for msg in self.message_history:
            # Apply filters
            if filters.get("sender") and msg.sender != filters["sender"]:
                continue
            if filters.get("recipient") and msg.recipient != filters["recipient"]:
                continue
            if filters.get("type") and msg.type.value != filters["type"]:
                continue
            
            history.append(msg.to_dict())
        
        # Limit results
        limit = filters.get("limit", 100)
        return history[-limit:]
    
    async def _pause_agent(self, agent_id: str):
        """Pause an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if hasattr(agent, 'pause'):
                await agent.pause()
    
    async def _resume_agent(self, agent_id: str):
        """Resume a paused agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if hasattr(agent, 'resume'):
                await agent.resume()
    
    async def _restart_agent(self, agent_id: str):
        """Restart an agent"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            if hasattr(agent, 'restart'):
                await agent.restart()


class AgentInterface:
    """
    Base interface for agents to communicate via the message bus
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.paused = False
        self.message_handlers: Dict[MessageType, Callable] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup message handlers"""
        self.message_handlers = {
            MessageType.REQUEST: self._handle_request,
            MessageType.QUERY: self._handle_query,
            MessageType.COMMAND: self._handle_command,
            MessageType.NOTIFICATION: self._handle_notification,
            MessageType.EVENT: self._handle_event
        }
    
    async def handle_message(self, message: Message) -> Any:
        """Handle incoming message"""
        if self.paused:
            logger.info(f"Agent {self.agent_id} is paused, queuing message")
            await self.inbox.put(message)
            return None
        
        handler = self.message_handlers.get(message.type)
        if handler:
            return await handler(message)
        else:
            logger.warning(f"No handler for message type: {message.type}")
            return None
    
    async def send_message(self, 
                          recipient: str,
                          content: Any,
                          message_type: MessageType = MessageType.REQUEST,
                          priority: MessagePriority = MessagePriority.NORMAL,
                          **kwargs) -> Any:
        """Send a message to another agent"""
        message = Message(
            type=message_type,
            sender=self.agent_id,
            recipient=recipient,
            content=content,
            priority=priority,
            **kwargs
        )
        
        return await self.message_bus.send(message)
    
    async def broadcast(self, content: Any, **kwargs):
        """Broadcast a message to all agents"""
        await self.send_message(
            recipient="*",
            content=content,
            message_type=MessageType.BROADCAST,
            **kwargs
        )
    
    async def publish_event(self, event_type: str, data: Any):
        """Publish an event"""
        await self.send_message(
            recipient="*",
            content=data,
            message_type=MessageType.EVENT,
            metadata={"event_type": event_type}
        )
    
    async def query(self, target: str, query_type: str, params: Dict[str, Any] = None) -> Any:
        """Query another agent"""
        return await self.send_message(
            recipient=target,
            content=params,
            message_type=MessageType.QUERY,
            metadata={"query_type": query_type}
        )
    
    async def _handle_request(self, message: Message) -> Any:
        """Handle request message - override in subclass"""
        return {"status": "received", "agent": self.agent_id}
    
    async def _handle_query(self, message: Message) -> Any:
        """Handle query message - override in subclass"""
        return {"status": "active", "agent": self.agent_id}
    
    async def _handle_command(self, message: Message):
        """Handle command message - override in subclass"""
        command = message.content.get("command")
        if command == "pause":
            await self.pause()
        elif command == "resume":
            await self.resume()
        elif command == "restart":
            await self.restart()
    
    async def _handle_notification(self, message: Message):
        """Handle notification message - override in subclass"""
        logger.info(f"Agent {self.agent_id} received notification: {message.content}")
    
    async def _handle_event(self, message: Message):
        """Handle event message - override in subclass"""
        event_type = message.metadata.get("event_type")
        logger.info(f"Agent {self.agent_id} received event {event_type}: {message.content}")
    
    async def pause(self):
        """Pause the agent"""
        self.paused = True
        logger.info(f"Agent {self.agent_id} paused")
    
    async def resume(self):
        """Resume the agent"""
        self.paused = False
        logger.info(f"Agent {self.agent_id} resumed")
        
        # Process queued messages
        while not self.inbox.empty():
            message = await self.inbox.get()
            await self.handle_message(message)
    
    async def restart(self):
        """Restart the agent"""
        logger.info(f"Agent {self.agent_id} restarting")
        await self.pause()
        # Clear queues
        self.inbox = asyncio.Queue()
        self.outbox = asyncio.Queue()
        await self.resume()
    
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "agent_id": self.agent_id,
            "paused": self.paused,
            "inbox_size": self.inbox.qsize(),
            "outbox_size": self.outbox.qsize()
        }


class ParallelCoordinator:
    """
    Coordinates parallel execution of agent tasks
    """
    
    def __init__(self, message_bus: MessageBus):
        self.message_bus = message_bus
        self.execution_pools: Dict[str, asyncio.Semaphore] = {}
        self.task_tracker: Dict[str, List[str]] = defaultdict(list)
    
    def create_execution_pool(self, pool_name: str, max_concurrent: int = 5):
        """Create an execution pool with concurrency limit"""
        self.execution_pools[pool_name] = asyncio.Semaphore(max_concurrent)
        logger.info(f"Created execution pool '{pool_name}' with max {max_concurrent} concurrent tasks")
    
    async def execute_parallel(self, 
                              tasks: List[Dict[str, Any]],
                              pool_name: str = "default") -> List[Any]:
        """Execute tasks in parallel with pool constraints"""
        if pool_name not in self.execution_pools:
            self.create_execution_pool(pool_name)
        
        semaphore = self.execution_pools[pool_name]
        
        async def execute_with_limit(task: Dict[str, Any]) -> Any:
            async with semaphore:
                return await self._execute_single_task(task)
        
        # Execute all tasks in parallel with concurrency limit
        results = await asyncio.gather(
            *[execute_with_limit(task) for task in tasks],
            return_exceptions=True
        )
        
        return results
    
    async def _execute_single_task(self, task: Dict[str, Any]) -> Any:
        """Execute a single task"""
        agent_id = task.get("agent")
        message_content = task.get("content")
        message_type = MessageType(task.get("type", "request"))
        
        message = Message(
            type=message_type,
            sender="coordinator",
            recipient=agent_id,
            content=message_content
        )
        
        return await self.message_bus.send(message)
    
    async def map_reduce(self,
                        map_tasks: List[Dict[str, Any]],
                        reduce_func: Callable,
                        pool_name: str = "mapreduce") -> Any:
        """Execute map-reduce pattern"""
        # Map phase - parallel execution
        map_results = await self.execute_parallel(map_tasks, pool_name)
        
        # Filter out exceptions
        valid_results = [r for r in map_results if not isinstance(r, Exception)]
        
        # Reduce phase
        return await reduce_func(valid_results)
    
    async def pipeline(self,
                      stages: List[List[Dict[str, Any]]],
                      pool_name: str = "pipeline") -> List[Any]:
        """Execute pipeline pattern - stages in sequence, tasks in parallel within each stage"""
        results = []
        previous_results = None
        
        for stage_idx, stage_tasks in enumerate(stages):
            logger.info(f"Executing pipeline stage {stage_idx + 1}/{len(stages)}")
            
            # Inject previous results if available
            if previous_results:
                for task in stage_tasks:
                    task["previous_results"] = previous_results
            
            # Execute stage in parallel
            stage_results = await self.execute_parallel(stage_tasks, pool_name)
            results.append(stage_results)
            previous_results = stage_results
        
        return results
    
    async def scatter_gather(self,
                           scatter_task: Dict[str, Any],
                           gather_agents: List[str],
                           aggregation_func: Callable) -> Any:
        """Execute scatter-gather pattern"""
        # Scatter phase - send same task to multiple agents
        scatter_tasks = []
        for agent_id in gather_agents:
            task = scatter_task.copy()
            task["agent"] = agent_id
            scatter_tasks.append(task)
        
        # Gather phase - collect results
        results = await self.execute_parallel(scatter_tasks)
        
        # Aggregation
        return await aggregation_func(results)


# Example usage and test
if __name__ == "__main__":
    async def demo():
        """Demonstration of agent communication protocol"""
        
        # Create message bus
        bus = MessageBus()
        await bus.start()
        
        # Create sample agents
        class SampleAgent(AgentInterface):
            async def _handle_request(self, message: Message) -> Any:
                print(f"{self.agent_id} handling request: {message.content}")
                return {"response": f"Processed by {self.agent_id}"}
        
        # Register agents
        agent1 = SampleAgent("agent1", bus)
        agent2 = SampleAgent("agent2", bus)
        agent3 = SampleAgent("agent3", bus)
        
        bus.register_agent("agent1", agent1)
        bus.register_agent("agent2", agent2)
        bus.register_agent("agent3", agent3)
        
        # Test direct messaging
        response = await agent1.send_message("agent2", {"data": "test"})
        print(f"Direct message response: {response}")
        
        # Test broadcast
        await agent1.broadcast({"announcement": "Hello all agents!"})
        
        # Test parallel coordination
        coordinator = ParallelCoordinator(bus)
        
        # Parallel execution
        tasks = [
            {"agent": "agent1", "content": {"task": 1}},
            {"agent": "agent2", "content": {"task": 2}},
            {"agent": "agent3", "content": {"task": 3}}
        ]
        
        results = await coordinator.execute_parallel(tasks)
        print(f"Parallel execution results: {results}")
        
        # Pipeline execution
        pipeline_stages = [
            [{"agent": "agent1", "content": {"stage": 1}}],
            [{"agent": "agent2", "content": {"stage": 2}}],
            [{"agent": "agent3", "content": {"stage": 3}}]
        ]
        
        pipeline_results = await coordinator.pipeline(pipeline_stages)
        print(f"Pipeline results: {pipeline_results}")
        
        # Scatter-gather
        async def avg_aggregator(results):
            return {"average": len(results)}
        
        scatter_result = await coordinator.scatter_gather(
            {"content": {"compute": "value"}},
            ["agent1", "agent2", "agent3"],
            avg_aggregator
        )
        print(f"Scatter-gather result: {scatter_result}")
        
        # Stop message bus
        await bus.stop()
    
    # Run demo
    asyncio.run(demo())