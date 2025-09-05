"""
Agent Repository Implementation
Manages agent registration and discovery
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

from ...shared import IAgent, IAgentRepository, AgentId

logger = logging.getLogger(__name__)


class InMemoryAgentRepository(IAgentRepository):
    """
    In-memory implementation of agent repository
    Suitable for single-node deployments and testing
    """
    
    def __init__(self):
        self._agents: Dict[AgentId, IAgent] = {}
        self._capabilities_index: Dict[str, Set[AgentId]] = defaultdict(set)
        self._lock = asyncio.Lock()
    
    async def register_agent(self, agent: IAgent) -> None:
        """Register agent and index its capabilities"""
        agent_id = agent.agent_id
        
        async with self._lock:
            self._agents[agent_id] = agent
            
            # Index capabilities
            for capability in agent.capabilities:
                self._capabilities_index[capability].add(agent_id)
        
        logger.info(f"Registered agent {agent_id.full_id} with capabilities: {agent.capabilities}")
    
    async def get_agent(self, agent_id: AgentId) -> Optional[IAgent]:
        """Get agent by ID"""
        async with self._lock:
            return self._agents.get(agent_id)
    
    async def find_agents_with_capability(self, capability: str) -> List[IAgent]:
        """Find all agents with specific capability"""
        async with self._lock:
            agent_ids = self._capabilities_index.get(capability, set())
            return [self._agents[agent_id] for agent_id in agent_ids if agent_id in self._agents]
    
    async def remove_agent(self, agent_id: AgentId) -> None:
        """Remove agent and clean up capability index"""
        async with self._lock:
            agent = self._agents.get(agent_id)
            if agent:
                # Clean up capability index
                for capability in agent.capabilities:
                    self._capabilities_index[capability].discard(agent_id)
                    if not self._capabilities_index[capability]:
                        del self._capabilities_index[capability]
                
                # Remove agent
                del self._agents[agent_id]
                logger.info(f"Removed agent {agent_id.full_id}")
    
    async def get_all_agents(self) -> List[IAgent]:
        """Get all registered agents"""
        async with self._lock:
            return list(self._agents.values())
    
    async def get_agents_by_namespace(self, namespace: str) -> List[IAgent]:
        """Get all agents in a namespace"""
        async with self._lock:
            return [agent for agent in self._agents.values() 
                   if agent.agent_id.namespace == namespace]
    
    async def get_capability_counts(self) -> Dict[str, int]:
        """Get count of agents per capability"""
        async with self._lock:
            return {capability: len(agent_ids) 
                   for capability, agent_ids in self._capabilities_index.items()}
    
    async def health_check_all(self) -> Dict[AgentId, Dict[str, any]]:
        """Perform health check on all agents"""
        results = {}
        
        async with self._lock:
            agents = list(self._agents.values())
        
        # Perform health checks in parallel
        async def check_agent(agent: IAgent) -> Tuple[AgentId, Dict[str, Any]]:
            try:
                health = await agent.health_check()
                return agent.agent_id, health
            except Exception as e:
                return agent.agent_id, {"status": "error", "error": str(e)}
        
        tasks = [check_agent(agent) for agent in agents]
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in health_results:
            if isinstance(result, Exception):
                logger.error(f"Error during health check: {result}")
            else:
                agent_id, health = result
                results[agent_id] = health
        
        return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get repository statistics"""
        async with self._lock:
            namespace_counts = defaultdict(int)
            total_capabilities = set()
            
            for agent in self._agents.values():
                namespace_counts[agent.agent_id.namespace] += 1
                total_capabilities.update(agent.capabilities)
            
            return {
                "total_agents": len(self._agents),
                "namespaces": dict(namespace_counts),
                "unique_capabilities": len(total_capabilities),
                "capability_distribution": {
                    capability: len(agent_ids)
                    for capability, agent_ids in self._capabilities_index.items()
                }
            }