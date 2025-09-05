"""
Event Bus Implementation
Provides event-driven communication between domains
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Callable, Set
from collections import defaultdict
from datetime import datetime

from .interfaces import IEventBus, DomainEvent

logger = logging.getLogger(__name__)


class EventBus(IEventBus):
    """
    In-memory event bus implementation
    Provides asynchronous event publishing and subscription
    """
    
    def __init__(self):
        self._subscribers: Dict[str, Dict[str, Callable[[DomainEvent], None]]] = defaultdict(dict)
        self._event_history: List[DomainEvent] = []
        self._max_history = 1000
        self._lock = asyncio.Lock()
    
    async def publish(self, event: DomainEvent) -> None:
        """Publish domain event to all subscribers"""
        try:
            async with self._lock:
                # Store in history
                self._event_history.append(event)
                if len(self._event_history) > self._max_history:
                    self._event_history.pop(0)
            
            # Notify subscribers
            if event.event_type in self._subscribers:
                for subscription_id, handler in self._subscribers[event.event_type].items():
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event)
                        else:
                            handler(event)
                    except Exception as e:
                        logger.error(f"Error in event handler {subscription_id}: {e}")
            
            logger.debug(f"Published event {event.event_type} from {event.source.full_id}")
            
        except Exception as e:
            logger.error(f"Error publishing event {event.event_type}: {e}")
            raise
    
    async def subscribe(self, event_type: str, handler: Callable[[DomainEvent], None]) -> str:
        """Subscribe to event type, returns subscription ID"""
        subscription_id = str(uuid.uuid4())
        
        async with self._lock:
            self._subscribers[event_type][subscription_id] = handler
        
        logger.debug(f"Subscribed to {event_type} with ID {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe from events using subscription ID"""
        async with self._lock:
            for event_type in self._subscribers:
                if subscription_id in self._subscribers[event_type]:
                    del self._subscribers[event_type][subscription_id]
                    logger.debug(f"Unsubscribed {subscription_id} from {event_type}")
                    break
    
    async def get_event_history(self, event_type: str = None, limit: int = 100) -> List[DomainEvent]:
        """Get event history, optionally filtered by type"""
        async with self._lock:
            events = self._event_history[-limit:] if limit > 0 else self._event_history
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            return events
    
    async def get_subscribers_count(self, event_type: str = None) -> Dict[str, int]:
        """Get subscriber counts by event type"""
        async with self._lock:
            if event_type:
                return {event_type: len(self._subscribers.get(event_type, {}))}
            else:
                return {et: len(subs) for et, subs in self._subscribers.items()}
    
    async def clear_history(self) -> None:
        """Clear event history"""
        async with self._lock:
            self._event_history.clear()
        logger.info("Event history cleared")


class EventHandler:
    """Decorator for event handlers"""
    
    def __init__(self, event_type: str, event_bus: EventBus = None):
        self.event_type = event_type
        self.event_bus = event_bus
        self.subscription_id = None
    
    def __call__(self, func: Callable[[DomainEvent], None]):
        """Decorator implementation"""
        if self.event_bus:
            # Auto-subscribe if event bus provided
            async def auto_subscribe():
                self.subscription_id = await self.event_bus.subscribe(self.event_type, func)
            
            # Schedule subscription
            asyncio.create_task(auto_subscribe())
        
        func._event_type = self.event_type
        func._subscription_id = self.subscription_id
        return func


# Factory functions for common events
def create_agent_event(event_type: str, agent_id: "AgentId", data: Dict) -> DomainEvent:
    """Create agent-related event"""
    return DomainEvent(
        event_id=str(uuid.uuid4()),
        event_type=f"agent.{event_type}",
        source=agent_id,
        timestamp=datetime.utcnow(),
        data=data
    )


def create_task_event(event_type: str, agent_id: "AgentId", task_id: str, data: Dict) -> DomainEvent:
    """Create task-related event"""
    return DomainEvent(
        event_id=str(uuid.uuid4()),
        event_type=f"task.{event_type}",
        source=agent_id,
        timestamp=datetime.utcnow(),
        data={**data, "task_id": task_id}
    )


def create_system_event(event_type: str, source_agent: "AgentId", data: Dict) -> DomainEvent:
    """Create system-related event"""
    return DomainEvent(
        event_id=str(uuid.uuid4()),
        event_type=f"system.{event_type}",
        source=source_agent,
        timestamp=datetime.utcnow(),
        data=data
    )