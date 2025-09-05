"""
Autonomous Intelligence Service
Core service for autonomous behavior management
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ...shared import (
    IAgent, AgentId, TaskId, ExecutionContext, ExecutionResult, 
    Priority, DomainEvent, IEventBus, get_service
)

logger = logging.getLogger(__name__)


class AutonomyLevel(Enum):
    """Levels of agent autonomy"""
    SUPERVISED = "supervised"
    SEMI_AUTONOMOUS = "semi_autonomous"
    FULLY_AUTONOMOUS = "fully_autonomous"
    EMERGENT = "emergent"


@dataclass
class AutonomousCapability:
    """Represents an autonomous capability"""
    name: str
    description: str
    autonomy_level: AutonomyLevel
    risk_level: int  # 1-10 scale
    approval_required: bool = False
    constraints: Set[str] = field(default_factory=set)
    
    def can_execute_at_level(self, current_level: AutonomyLevel) -> bool:
        """Check if capability can execute at given autonomy level"""
        level_hierarchy = {
            AutonomyLevel.SUPERVISED: 1,
            AutonomyLevel.SEMI_AUTONOMOUS: 2,
            AutonomyLevel.FULLY_AUTONOMOUS: 3,
            AutonomyLevel.EMERGENT: 4
        }
        
        return level_hierarchy[current_level] >= level_hierarchy[self.autonomy_level]


@dataclass
class AutonomousDecision:
    """Represents an autonomous decision"""
    decision_id: str
    agent_id: AgentId
    capability: str
    reasoning: str
    confidence: float
    risk_assessment: Dict[str, Any]
    approved: bool = False
    executed: bool = False
    timestamp: datetime = field(default_factory=datetime.utcnow)


class AutonomousIntelligenceService:
    """
    Service for managing autonomous intelligence capabilities
    Handles decision-making, capability discovery, and autonomous execution
    """
    
    def __init__(self):
        self._capabilities: Dict[str, AutonomousCapability] = {}
        self._pending_decisions: Dict[str, AutonomousDecision] = {}
        self._decision_history: List[AutonomousDecision] = []
        self._autonomy_levels: Dict[AgentId, AutonomyLevel] = {}
        self._event_bus: Optional[IEventBus] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the service"""
        try:
            self._event_bus = get_service(IEventBus)
        except ValueError:
            logger.warning("EventBus not available, running without events")
        
        # Register built-in capabilities
        await self._register_builtin_capabilities()
    
    async def _register_builtin_capabilities(self) -> None:
        """Register built-in autonomous capabilities"""
        builtin_capabilities = [
            AutonomousCapability(
                name="task_optimization",
                description="Optimize task execution strategies",
                autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS,
                risk_level=3
            ),
            AutonomousCapability(
                name="resource_allocation", 
                description="Allocate computational resources",
                autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS,
                risk_level=5
            ),
            AutonomousCapability(
                name="strategy_adaptation",
                description="Adapt strategies based on performance",
                autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS,
                risk_level=4
            ),
            AutonomousCapability(
                name="capability_discovery",
                description="Discover new capabilities through experimentation",
                autonomy_level=AutonomyLevel.EMERGENT,
                risk_level=7,
                approval_required=True
            ),
            AutonomousCapability(
                name="self_modification",
                description="Modify own behavior and strategies",
                autonomy_level=AutonomyLevel.EMERGENT,
                risk_level=9,
                approval_required=True,
                constraints={"safety_validated", "human_oversight"}
            )
        ]
        
        for capability in builtin_capabilities:
            await self.register_capability(capability)
    
    async def register_capability(self, capability: AutonomousCapability) -> None:
        """Register an autonomous capability"""
        async with self._lock:
            self._capabilities[capability.name] = capability
        
        logger.info(f"Registered autonomous capability: {capability.name}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"capability_registered_{capability.name}",
                event_type="autonomous.capability_registered",
                source=AgentId("system", "autonomous_service"),
                timestamp=datetime.utcnow(),
                data={"capability_name": capability.name, "autonomy_level": capability.autonomy_level.value}
            ))
    
    async def set_agent_autonomy_level(self, agent_id: AgentId, level: AutonomyLevel) -> None:
        """Set autonomy level for an agent"""
        async with self._lock:
            self._autonomy_levels[agent_id] = level
        
        logger.info(f"Set autonomy level for {agent_id.full_id}: {level.value}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"autonomy_level_set_{agent_id.full_id}",
                event_type="autonomous.level_changed",
                source=agent_id,
                timestamp=datetime.utcnow(),
                data={"autonomy_level": level.value}
            ))
    
    async def get_agent_autonomy_level(self, agent_id: AgentId) -> AutonomyLevel:
        """Get autonomy level for an agent"""
        async with self._lock:
            return self._autonomy_levels.get(agent_id, AutonomyLevel.SUPERVISED)
    
    async def request_autonomous_decision(self, 
                                        agent_id: AgentId,
                                        capability: str,
                                        context: Dict[str, Any],
                                        reasoning: str,
                                        confidence: float) -> str:
        """Request permission for autonomous decision"""
        
        if capability not in self._capabilities:
            raise ValueError(f"Unknown capability: {capability}")
        
        cap_obj = self._capabilities[capability]
        agent_level = await self.get_agent_autonomy_level(agent_id)
        
        # Create decision record
        decision = AutonomousDecision(
            decision_id=f"{agent_id.full_id}_{capability}_{datetime.utcnow().timestamp()}",
            agent_id=agent_id,
            capability=capability,
            reasoning=reasoning,
            confidence=confidence,
            risk_assessment=await self._assess_risk(cap_obj, context, confidence)
        )
        
        # Determine if approval needed
        if cap_obj.can_execute_at_level(agent_level) and not cap_obj.approval_required:
            decision.approved = True
            logger.info(f"Auto-approved autonomous decision: {decision.decision_id}")
        else:
            logger.info(f"Autonomous decision requires approval: {decision.decision_id}")
        
        async with self._lock:
            self._pending_decisions[decision.decision_id] = decision
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=decision.decision_id,
                event_type="autonomous.decision_requested",
                source=agent_id,
                timestamp=datetime.utcnow(),
                data={
                    "capability": capability,
                    "approved": decision.approved,
                    "requires_approval": not decision.approved,
                    "risk_level": cap_obj.risk_level,
                    "confidence": confidence
                }
            ))
        
        return decision.decision_id
    
    async def approve_decision(self, decision_id: str, approved: bool = True) -> None:
        """Approve or reject an autonomous decision"""
        async with self._lock:
            if decision_id not in self._pending_decisions:
                raise ValueError(f"Unknown decision ID: {decision_id}")
            
            decision = self._pending_decisions[decision_id]
            decision.approved = approved
        
        logger.info(f"Decision {decision_id} {'approved' if approved else 'rejected'}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"decision_approved_{decision_id}",
                event_type="autonomous.decision_approved" if approved else "autonomous.decision_rejected",
                source=decision.agent_id,
                timestamp=datetime.utcnow(),
                data={"decision_id": decision_id, "approved": approved}
            ))
    
    async def execute_decision(self, decision_id: str) -> ExecutionResult:
        """Execute an approved autonomous decision"""
        async with self._lock:
            if decision_id not in self._pending_decisions:
                raise ValueError(f"Unknown decision ID: {decision_id}")
            
            decision = self._pending_decisions[decision_id]
            
            if not decision.approved:
                return ExecutionResult(
                    success=False,
                    error=Exception("Decision not approved for execution")
                )
            
            if decision.executed:
                return ExecutionResult(
                    success=False,
                    error=Exception("Decision already executed")
                )
        
        # Execute the capability
        try:
            result = await self._execute_capability(decision)
            decision.executed = True
            
            # Move to history
            async with self._lock:
                self._decision_history.append(decision)
                del self._pending_decisions[decision_id]
            
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id=f"decision_executed_{decision_id}",
                    event_type="autonomous.decision_executed",
                    source=decision.agent_id,
                    timestamp=datetime.utcnow(),
                    data={"decision_id": decision_id, "success": result.success}
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing decision {decision_id}: {e}")
            return ExecutionResult(success=False, error=e)
    
    async def _assess_risk(self, capability: AutonomousCapability, 
                          context: Dict[str, Any], confidence: float) -> Dict[str, Any]:
        """Assess risk of executing capability"""
        base_risk = capability.risk_level
        
        # Adjust risk based on confidence
        confidence_factor = 1.0 - confidence  # Lower confidence = higher risk
        adjusted_risk = min(10, base_risk + (confidence_factor * 3))
        
        return {
            "base_risk": base_risk,
            "confidence_factor": confidence_factor,
            "adjusted_risk": adjusted_risk,
            "constraints": list(capability.constraints),
            "approval_required": capability.approval_required
        }
    
    async def _execute_capability(self, decision: AutonomousDecision) -> ExecutionResult:
        """Execute the autonomous capability"""
        capability = self._capabilities[decision.capability]
        
        logger.info(f"Executing autonomous capability: {capability.name}")
        
        # For now, simulate execution
        # In real implementation, this would delegate to specific capability handlers
        await asyncio.sleep(0.1)  # Simulate work
        
        return ExecutionResult(
            success=True,
            result=f"Executed {capability.name} successfully",
            execution_time=0.1,
            metadata={
                "capability": capability.name,
                "agent_id": decision.agent_id.full_id,
                "confidence": decision.confidence
            }
        )
    
    async def get_pending_decisions(self, agent_id: Optional[AgentId] = None) -> List[AutonomousDecision]:
        """Get pending decisions, optionally filtered by agent"""
        async with self._lock:
            decisions = list(self._pending_decisions.values())
            
            if agent_id:
                decisions = [d for d in decisions if d.agent_id == agent_id]
            
            return decisions
    
    async def get_decision_history(self, agent_id: Optional[AgentId] = None, 
                                 limit: int = 100) -> List[AutonomousDecision]:
        """Get decision history, optionally filtered by agent"""
        async with self._lock:
            history = self._decision_history[-limit:] if limit > 0 else self._decision_history
            
            if agent_id:
                history = [d for d in history if d.agent_id == agent_id]
            
            return history
    
    async def get_capabilities(self) -> Dict[str, AutonomousCapability]:
        """Get all registered capabilities"""
        async with self._lock:
            return self._capabilities.copy()
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get autonomous intelligence system status"""
        async with self._lock:
            return {
                "total_capabilities": len(self._capabilities),
                "pending_decisions": len(self._pending_decisions),
                "decision_history_count": len(self._decision_history),
                "active_agents": len(self._autonomy_levels),
                "autonomy_distribution": {
                    level.value: sum(1 for l in self._autonomy_levels.values() if l == level)
                    for level in AutonomyLevel
                }
            }