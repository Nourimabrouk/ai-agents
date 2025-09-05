"""
Intelligent Task Allocator: Phase 6 - Market-Based Task Allocation
Features:
- Market-based task allocation (agents bid on tasks)
- Reputation system for agent reliability
- Load balancing across agents
- Cost optimization for API usage
"""

import asyncio
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict, deque
import heapq
from abc import ABC, abstractmethod

from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class BidType(Enum):
    """Types of bidding strategies"""
    COST_OPTIMIZED = "cost_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    TIME_OPTIMIZED = "time_optimized"
    BALANCED = "balanced"


class TaskStatus(Enum):
    """Task status in allocation system"""
    PENDING = "pending"
    BIDDING = "bidding"
    ALLOCATED = "allocated"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskBid:
    """Represents a bid from an agent for a task"""
    agent_id: str
    task_id: str
    bid_amount: float  # Cost to complete task
    estimated_time: float  # Time in minutes
    confidence: float  # 0-1 confidence in completion
    quality_promise: float  # 0-1 expected quality
    bid_type: BidType
    timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def value_score(self) -> float:
        """Calculate overall value score for this bid"""
        # Higher quality and confidence, lower cost and time = better score
        time_score = max(0, 1.0 - (self.estimated_time / 60.0))  # Normalize to hours
        cost_score = max(0, 1.0 - (self.bid_amount / 100.0))  # Normalize to $100
        
        return (
            self.quality_promise * 0.3 +
            self.confidence * 0.3 +
            time_score * 0.2 +
            cost_score * 0.2
        )


@dataclass
class AgentReputation:
    """Tracks agent reputation and performance metrics"""
    agent_id: str
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    average_quality: float = 0.0
    average_time_accuracy: float = 0.0  # How accurate time estimates are
    average_cost_accuracy: float = 0.0  # How accurate cost estimates are
    reliability_score: float = 1.0  # Overall reliability (0-1)
    specializations: Dict[str, float] = field(default_factory=dict)  # Domain -> skill level
    last_updated: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_tasks / self.total_tasks
    
    def update_performance(self, task_completed: bool, estimated_time: float, 
                          actual_time: float, estimated_cost: float, 
                          actual_cost: float, quality: float, domain: str):
        """Update reputation based on task performance"""
        self.total_tasks += 1
        
        if task_completed:
            self.completed_tasks += 1
            
            # Update quality average
            alpha = 0.1  # Learning rate
            self.average_quality = (1 - alpha) * self.average_quality + alpha * quality
            
            # Update time accuracy
            if estimated_time > 0:
                time_accuracy = 1.0 - abs(actual_time - estimated_time) / estimated_time
                time_accuracy = max(0, min(1, time_accuracy))
                self.average_time_accuracy = (1 - alpha) * self.average_time_accuracy + alpha * time_accuracy
            
            # Update cost accuracy
            if estimated_cost > 0:
                cost_accuracy = 1.0 - abs(actual_cost - estimated_cost) / estimated_cost
                cost_accuracy = max(0, min(1, cost_accuracy))
                self.average_cost_accuracy = (1 - alpha) * self.average_cost_accuracy + alpha * cost_accuracy
            
            # Update specialization for domain
            if domain in self.specializations:
                self.specializations[domain] = (1 - alpha) * self.specializations[domain] + alpha * quality
            else:
                self.specializations[domain] = quality
        else:
            self.failed_tasks += 1
        
        # Update overall reliability score
        self.reliability_score = (
            self.success_rate * 0.4 +
            self.average_quality * 0.3 +
            self.average_time_accuracy * 0.15 +
            self.average_cost_accuracy * 0.15
        )
        
        self.last_updated = datetime.now()


@dataclass
class AllocationTask:
    """Task in the allocation system"""
    task_id: str
    description: str
    requirements: Dict[str, Any]
    priority: float  # 0-1 priority level
    max_budget: float
    deadline: Optional[datetime] = None
    domain: str = "general"
    required_skills: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    bids: List[TaskBid] = field(default_factory=list)
    allocated_agent: Optional[str] = None
    allocation_time: Optional[datetime] = None
    completion_time: Optional[datetime] = None
    actual_cost: float = 0.0
    actual_quality: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)


class MarketMaker:
    """Manages the bidding process and market dynamics"""
    
    def __init__(self):
        self.current_market_conditions = {
            "demand_pressure": 1.0,  # Multiplier for bid amounts
            "quality_premium": 1.0,   # Premium for high-quality agents
            "urgency_multiplier": 1.0, # Multiplier for urgent tasks
            "load_balancing_factor": 0.1  # How much to consider load balancing
        }
        self.market_history: List[Dict[str, Any]] = []
    
    async def calculate_task_value(self, task: AllocationTask) -> float:
        """Calculate the market value of a task"""
        base_value = task.priority * 100  # Base value in currency units
        
        # Adjust for urgency
        if task.deadline:
            time_left = (task.deadline - datetime.now()).total_seconds() / 3600  # hours
            urgency_multiplier = max(0.5, min(2.0, 24 / max(1, time_left)))
            base_value *= urgency_multiplier
        
        # Adjust for complexity (based on requirements)
        complexity_multiplier = 1.0 + len(task.required_skills) * 0.2
        base_value *= complexity_multiplier
        
        # Market conditions
        base_value *= self.current_market_conditions["demand_pressure"]
        
        return base_value
    
    async def evaluate_bid_competitiveness(self, bid: TaskBid, task: AllocationTask, 
                                         all_bids: List[TaskBid]) -> float:
        """Evaluate how competitive a bid is"""
        if not all_bids:
            return 1.0
        
        # Compare against other bids
        other_bids = [b for b in all_bids if b.agent_id != bid.agent_id]
        if not other_bids:
            return 1.0
        
        # Calculate percentile rankings
        costs = [b.bid_amount for b in other_bids]
        times = [b.estimated_time for b in other_bids]
        qualities = [b.quality_promise for b in other_bids]
        
        cost_percentile = self._calculate_percentile(bid.bid_amount, costs, lower_better=True)
        time_percentile = self._calculate_percentile(bid.estimated_time, times, lower_better=True)
        quality_percentile = self._calculate_percentile(bid.quality_promise, qualities, lower_better=False)
        
        # Weight the competitiveness score
        competitiveness = (
            cost_percentile * 0.3 +
            time_percentile * 0.3 +
            quality_percentile * 0.4
        )
        
        return competitiveness
    
    def _calculate_percentile(self, value: float, others: List[float], lower_better: bool) -> float:
        """Calculate percentile rank of value in others list"""
        if not others:
            return 0.5
        
        if lower_better:
            better_count = sum(1 for x in others if x > value)
        else:
            better_count = sum(1 for x in others if x < value)
        
        return better_count / len(others)
    
    async def update_market_conditions(self, completed_tasks: List[AllocationTask]):
        """Update market conditions based on recent task completions"""
        if not completed_tasks:
            return {}
        
        # Calculate demand pressure
        recent_tasks = [t for t in completed_tasks if t.completion_time and 
                       t.completion_time > datetime.now() - timedelta(hours=24)]
        
        if recent_tasks:
            avg_time_to_completion = np.mean([
                (t.completion_time - t.created_at).total_seconds() / 3600
                for t in recent_tasks
            ])
            
            # High demand if tasks are completing quickly (agents are eager)
            if avg_time_to_completion < 2:  # Less than 2 hours
                self.current_market_conditions["demand_pressure"] *= 1.1
            elif avg_time_to_completion > 8:  # More than 8 hours
                self.current_market_conditions["demand_pressure"] *= 0.95
            
            # Quality premium based on recent quality scores
            avg_quality = np.mean([t.actual_quality for t in recent_tasks])
            self.current_market_conditions["quality_premium"] = 0.8 + (avg_quality * 0.4)
        
        # Keep values in reasonable ranges
        self.current_market_conditions["demand_pressure"] = max(0.5, min(2.0, 
                                                               self.current_market_conditions["demand_pressure"]))


class LoadBalancer:
    """Manages load balancing across agents"""
    
    def __init__(self):
        self.agent_loads: Dict[str, int] = defaultdict(int)  # Current task count per agent
        self.agent_capacities: Dict[str, int] = defaultdict(lambda: 3)  # Max concurrent tasks
        self.load_history: List[Dict[str, Any]] = []
    
    def get_agent_load_factor(self, agent_id: str) -> float:
        """Get load factor for agent (0 = no load, 1 = at capacity)"""
        current_load = self.agent_loads.get(agent_id, 0)
        capacity = self.agent_capacities.get(agent_id, 3)
        return min(1.0, current_load / capacity)
    
    def get_load_balancing_bonus(self, agent_id: str) -> float:
        """Calculate bonus for load balancing (higher for less loaded agents)"""
        load_factor = self.get_agent_load_factor(agent_id)
        return (1.0 - load_factor) * 0.2  # Up to 20% bonus for unloaded agents
    
    async def allocate_task(self, agent_id: str, task_id: str):
        """Record task allocation to agent"""
        self.agent_loads[agent_id] += 1
        
        # Record for analytics
        self.load_history.append({
            "timestamp": datetime.now(),
            "agent_id": agent_id,
            "task_id": task_id,
            "action": "allocate",
            "load_after": self.agent_loads[agent_id]
        })
    
    async def complete_task(self, agent_id: str, task_id: str):
        """Record task completion"""
        if self.agent_loads[agent_id] > 0:
            self.agent_loads[agent_id] -= 1
        
        # Record for analytics
        self.load_history.append({
            "timestamp": datetime.now(),
            "agent_id": agent_id,
            "task_id": task_id,
            "action": "complete",
            "load_after": self.agent_loads[agent_id]
        })
    
    async def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution statistics"""
        if not self.agent_loads:
            return {"total_agents": 0, "total_load": 0, "average_load": 0, "load_variance": 0}
        
        loads = list(self.agent_loads.values())
        return {
            "total_agents": len(loads),
            "total_load": sum(loads),
            "average_load": np.mean(loads),
            "load_variance": np.var(loads),
            "max_load": max(loads),
            "min_load": min(loads)
        }


class IntelligentTaskAllocator:
    """Main task allocator with market-based allocation"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_reputations: Dict[str, AgentReputation] = {}
        self.tasks: Dict[str, AllocationTask] = {}
        self.market_maker = MarketMaker()
        self.load_balancer = LoadBalancer()
        
        # Configuration
        self.bidding_timeout = 30.0  # seconds
        self.min_bids_required = 2
        self.reputation_weight = 0.3
        self.cost_weight = 0.25
        self.quality_weight = 0.25
        self.time_weight = 0.2
        
        # Analytics
        self.allocation_history: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = defaultdict(list)
    
    def register_agent(self, agent: BaseAgent, capacity: int = 3):
        """Register an agent with the allocator"""
        self.agents[agent.name] = agent
        if agent.name not in self.agent_reputations:
            self.agent_reputations[agent.name] = AgentReputation(agent_id=agent.name)
        self.load_balancer.agent_capacities[agent.name] = capacity
        logger.info(f"Registered agent {agent.name} with capacity {capacity}")
    
    async def submit_task(self, task: AllocationTask) -> str:
        """Submit a task for allocation"""
        self.tasks[task.task_id] = task
        logger.info(f"Task {task.task_id} submitted: {task.description[:50]}...")
        
        # Start allocation process
        asyncio.create_task(self._allocate_task(task.task_id))
        return task.task_id
    
    async def _allocate_task(self, task_id: str):
        """Main allocation process using market-based bidding"""
        task = self.tasks[task_id]
        task.status = TaskStatus.BIDDING
        
        try:
            # Phase 1: Collect bids
            bids = await self._collect_bids(task)
            
            if not bids:
                logger.warning(f"No bids received for task {task_id}")
                task.status = TaskStatus.FAILED
                return {}
            
            task.bids = bids
            
            # Phase 2: Evaluate and select winner
            winning_bid = await self._select_winning_bid(task, bids)
            
            if not winning_bid:
                logger.warning(f"No suitable bid found for task {task_id}")
                task.status = TaskStatus.FAILED
                return {}
            
            # Phase 3: Allocate task
            await self._allocate_to_agent(task, winning_bid)
            
        except Exception as e:
            logger.error(f"Task allocation failed for {task_id}: {e}")
            task.status = TaskStatus.FAILED
    
    async def _collect_bids(self, task: AllocationTask) -> List[TaskBid]:
        """Collect bids from eligible agents"""
        eligible_agents = await self._find_eligible_agents(task)
        
        if not eligible_agents:
            return []
        
        # Request bids from eligible agents
        bid_tasks = []
        for agent_id in eligible_agents:
            bid_tasks.append(self._request_bid(agent_id, task))
        
        # Wait for bids with timeout
        try:
            bid_results = await asyncio.wait_for(
                asyncio.gather(*bid_tasks, return_exceptions=True),
                timeout=self.bidding_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Bidding timeout for task {task.task_id}")
            bid_results = []
        
        # Collect successful bids
        bids = []
        for result in bid_results:
            if isinstance(result, TaskBid):
                bids.append(result)
            elif isinstance(result, Exception):
                logger.debug(f"Bid request failed: {result}")
        
        logger.info(f"Collected {len(bids)} bids for task {task.task_id}")
        return bids
    
    async def _find_eligible_agents(self, task: AllocationTask) -> List[str]:
        """Find agents eligible to bid on task"""
        eligible = []
        
        for agent_id, agent in self.agents.items():
            # Check capacity
            if self.load_balancer.get_agent_load_factor(agent_id) >= 1.0:
                continue  # Agent at capacity
            
            # Check reputation threshold
            reputation = self.agent_reputations.get(agent_id)
            if reputation and reputation.reliability_score < 0.3:
                continue  # Too unreliable
            
            # Check skill matching
            agent_skills = getattr(agent, 'capabilities', [])
            if task.required_skills:
                skill_match = any(skill in agent_skills for skill in task.required_skills)
                if not skill_match and 'general' not in agent_skills:
                    continue
            
            eligible.append(agent_id)
        
        return eligible
    
    async def _request_bid(self, agent_id: str, task: AllocationTask) -> Optional[TaskBid]:
        """Request a bid from specific agent"""
        try:
            agent = self.agents[agent_id]
            reputation = self.agent_reputations[agent_id]
            
            # Generate bid based on agent characteristics and reputation
            bid = await self._generate_agent_bid(agent, task, reputation)
            
            return bid
            
        except Exception as e:
            logger.error(f"Failed to get bid from agent {agent_id}: {e}")
            return {}
    
    async def _generate_agent_bid(self, agent: BaseAgent, task: AllocationTask, 
                                 reputation: AgentReputation) -> TaskBid:
        """Generate a bid for an agent based on their characteristics"""
        
        # Base estimates
        base_time = 10.0  # Base 10 minutes
        base_cost = 5.0   # Base $5
        
        # Adjust based on task complexity
        complexity_factor = 1.0 + len(task.required_skills) * 0.3
        estimated_time = base_time * complexity_factor
        estimated_cost = base_cost * complexity_factor
        
        # Adjust based on agent specialization
        domain_skill = reputation.specializations.get(task.domain, 0.5)
        skill_factor = 2.0 - domain_skill  # Better skills = lower cost/time
        estimated_time *= skill_factor
        estimated_cost *= skill_factor
        
        # Adjust based on current load
        load_factor = self.load_balancer.get_agent_load_factor(agent.name)
        load_multiplier = 1.0 + (load_factor * 0.5)  # Higher load = higher cost
        estimated_cost *= load_multiplier
        
        # Add some randomness for market dynamics
        time_variance = np.random.uniform(0.8, 1.2)
        cost_variance = np.random.uniform(0.9, 1.1)
        
        estimated_time *= time_variance
        estimated_cost *= cost_variance
        
        # Calculate confidence and quality promise
        confidence = min(1.0, reputation.reliability_score + domain_skill * 0.3)
        quality_promise = min(1.0, domain_skill + reputation.average_quality * 0.5)
        
        # Determine bid type based on agent characteristics
        if reputation.average_cost_accuracy > 0.8:
            bid_type = BidType.COST_OPTIMIZED
        elif reputation.average_quality > 0.8:
            bid_type = BidType.QUALITY_OPTIMIZED
        elif reputation.average_time_accuracy > 0.8:
            bid_type = BidType.TIME_OPTIMIZED
        else:
            bid_type = BidType.BALANCED
        
        return TaskBid(
            agent_id=agent.name,
            task_id=task.task_id,
            bid_amount=estimated_cost,
            estimated_time=estimated_time,
            confidence=confidence,
            quality_promise=quality_promise,
            bid_type=bid_type
        )
    
    async def _select_winning_bid(self, task: AllocationTask, bids: List[TaskBid]) -> Optional[TaskBid]:
        """Select winning bid using multi-criteria decision making"""
        if not bids:
            return {}
        
        # Calculate scores for each bid
        bid_scores = []
        
        for bid in bids:
            reputation = self.agent_reputations[bid.agent_id]
            
            # Reputation score
            reputation_score = reputation.reliability_score
            
            # Cost score (lower cost = higher score)
            max_cost = max(b.bid_amount for b in bids)
            cost_score = 1.0 - (bid.bid_amount / max_cost) if max_cost > 0 else 1.0
            
            # Time score (lower time = higher score)
            max_time = max(b.estimated_time for b in bids)
            time_score = 1.0 - (bid.estimated_time / max_time) if max_time > 0 else 1.0
            
            # Quality score
            quality_score = bid.quality_promise
            
            # Load balancing bonus
            load_bonus = self.load_balancer.get_load_balancing_bonus(bid.agent_id)
            
            # Market competitiveness
            competitiveness = await self.market_maker.evaluate_bid_competitiveness(bid, task, bids)
            
            # Overall score
            overall_score = (
                reputation_score * self.reputation_weight +
                cost_score * self.cost_weight +
                time_score * self.time_weight +
                quality_score * self.quality_weight +
                load_bonus +
                competitiveness * 0.1
            )
            
            bid_scores.append((bid, overall_score))
        
        # Sort by score and return winner
        bid_scores.sort(key=lambda x: x[1], reverse=True)
        winning_bid = bid_scores[0][0]
        
        logger.info(f"Selected winning bid from {winning_bid.agent_id} for task {task.task_id} "
                   f"(score: {bid_scores[0][1]:.3f})")
        
        return winning_bid
    
    async def _allocate_to_agent(self, task: AllocationTask, winning_bid: TaskBid):
        """Allocate task to winning agent"""
        task.status = TaskStatus.ALLOCATED
        task.allocated_agent = winning_bid.agent_id
        task.allocation_time = datetime.now()
        
        # Update load balancer
        await self.load_balancer.allocate_task(winning_bid.agent_id, task.task_id)
        
        # Record allocation
        self.allocation_history.append({
            "task_id": task.task_id,
            "agent_id": winning_bid.agent_id,
            "bid_amount": winning_bid.bid_amount,
            "estimated_time": winning_bid.estimated_time,
            "quality_promise": winning_bid.quality_promise,
            "allocation_time": task.allocation_time,
            "total_bids": len(task.bids)
        })
        
        # Start task execution
        asyncio.create_task(self._execute_task(task, winning_bid))
        
        logger.info(f"Task {task.task_id} allocated to agent {winning_bid.agent_id}")
    
    async def _execute_task(self, task: AllocationTask, bid: TaskBid):
        """Execute task with allocated agent"""
        task.status = TaskStatus.IN_PROGRESS
        start_time = datetime.now()
        
        try:
            agent = self.agents[bid.agent_id]
            
            # Execute task
            result = await agent.process_task(task.description, task.requirements)
            
            # Task completed successfully
            end_time = datetime.now()
            actual_time = (end_time - start_time).total_seconds() / 60.0  # minutes
            
            # Assess quality (simplified - in practice would need domain-specific evaluation)
            quality = await self._assess_task_quality(result, task)
            
            task.status = TaskStatus.COMPLETED
            task.completion_time = end_time
            task.actual_cost = bid.bid_amount  # Use bid amount as actual cost
            task.actual_quality = quality
            
            # Update reputation
            reputation = self.agent_reputations[bid.agent_id]
            reputation.update_performance(
                task_completed=True,
                estimated_time=bid.estimated_time,
                actual_time=actual_time,
                estimated_cost=bid.bid_amount,
                actual_cost=bid.bid_amount,
                quality=quality,
                domain=task.domain
            )
            
            # Update load balancer
            await self.load_balancer.complete_task(bid.agent_id, task.task_id)
            
            logger.info(f"Task {task.task_id} completed by {bid.agent_id} "
                       f"(time: {actual_time:.1f}min, quality: {quality:.2f})")
            
        except Exception as e:
            # Task failed
            task.status = TaskStatus.FAILED
            task.completion_time = datetime.now()
            
            # Update reputation with failure
            reputation = self.agent_reputations[bid.agent_id]
            reputation.update_performance(
                task_completed=False,
                estimated_time=bid.estimated_time,
                actual_time=0,
                estimated_cost=bid.bid_amount,
                actual_cost=0,
                quality=0,
                domain=task.domain
            )
            
            # Update load balancer
            await self.load_balancer.complete_task(bid.agent_id, task.task_id)
            
            logger.error(f"Task {task.task_id} failed: {e}")
    
    async def _assess_task_quality(self, result: Any, task: AllocationTask) -> float:
        """Assess quality of task completion"""
        # Simplified quality assessment
        if result is None:
            return 0.0
        
        if isinstance(result, Exception):
            return 0.0
        
        # Basic quality factors
        quality = 0.5  # Base quality
        
        # Check if result has expected structure
        if isinstance(result, dict):
            quality += 0.2
            if "result" in result or "success" in result:
                quality += 0.2
        
        # Add randomness for simulation
        quality += np.random.uniform(-0.1, 0.3)
        
        return max(0.0, min(1.0, quality))
    
    async def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get allocation statistics and performance metrics"""
        # Task statistics
        total_tasks = len(self.tasks)
        completed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
        failed_tasks = sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED)
        
        # Agent statistics
        agent_stats = {}
        for agent_id, reputation in self.agent_reputations.items():
            agent_stats[agent_id] = {
                "total_tasks": reputation.total_tasks,
                "success_rate": reputation.success_rate,
                "reliability_score": reputation.reliability_score,
                "average_quality": reputation.average_quality,
                "specializations": dict(reputation.specializations),
                "current_load": self.load_balancer.agent_loads.get(agent_id, 0)
            }
        
        # Market statistics
        if self.allocation_history:
            recent_allocations = [a for a in self.allocation_history 
                                if a["allocation_time"] > datetime.now() - timedelta(hours=24)]
            
            avg_bids_per_task = np.mean([a["total_bids"] for a in recent_allocations]) if recent_allocations else 0
            avg_cost = np.mean([a["bid_amount"] for a in recent_allocations]) if recent_allocations else 0
            avg_time = np.mean([a["estimated_time"] for a in recent_allocations]) if recent_allocations else 0
        else:
            avg_bids_per_task = avg_cost = avg_time = 0
        
        # Load distribution
        load_stats = await self.load_balancer.get_load_distribution()
        
        return {
            "task_statistics": {
                "total_tasks": total_tasks,
                "completed_tasks": completed_tasks,
                "failed_tasks": failed_tasks,
                "success_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
            },
            "agent_statistics": agent_stats,
            "market_statistics": {
                "average_bids_per_task": avg_bids_per_task,
                "average_cost": avg_cost,
                "average_estimated_time": avg_time,
                "market_conditions": self.market_maker.current_market_conditions
            },
            "load_statistics": load_stats
        }
    
    async def optimize_allocations(self) -> Dict[str, Any]:
        """Optimize allocation parameters based on performance"""
        optimizations = []
        
        # Analyze recent performance
        completed_tasks = [t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]
        
        if len(completed_tasks) >= 10:
            # Analyze cost prediction accuracy
            cost_errors = []
            time_errors = []
            
            for task in completed_tasks[-20:]:  # Last 20 tasks
                if task.bids:
                    winning_bid = next((b for b in task.bids if b.agent_id == task.allocated_agent), None)
                    if winning_bid:
                        # Cost accuracy (using bid amount vs actual cost)
                        cost_error = abs(winning_bid.bid_amount - task.actual_cost) / winning_bid.bid_amount
                        cost_errors.append(cost_error)
                        
                        # Time accuracy
                        if task.completion_time and task.allocation_time:
                            actual_time = (task.completion_time - task.allocation_time).total_seconds() / 60
                            time_error = abs(winning_bid.estimated_time - actual_time) / winning_bid.estimated_time
                            time_errors.append(time_error)
            
            # Adjust weights based on prediction accuracy
            if cost_errors:
                avg_cost_error = np.mean(cost_errors)
                if avg_cost_error > 0.3:  # High cost prediction error
                    self.cost_weight = max(0.1, self.cost_weight - 0.05)
                    self.quality_weight = min(0.4, self.quality_weight + 0.03)
                    optimizations.append("Reduced cost weight due to poor prediction accuracy")
            
            if time_errors:
                avg_time_error = np.mean(time_errors)
                if avg_time_error > 0.3:  # High time prediction error
                    self.time_weight = max(0.1, self.time_weight - 0.05)
                    self.reputation_weight = min(0.4, self.reputation_weight + 0.03)
                    optimizations.append("Reduced time weight due to poor prediction accuracy")
            
            # Update market conditions
            await self.market_maker.update_market_conditions(completed_tasks)
            optimizations.append("Updated market conditions based on recent completions")
        
        return {
            "optimizations_applied": optimizations,
            "current_weights": {
                "reputation_weight": self.reputation_weight,
                "cost_weight": self.cost_weight,
                "quality_weight": self.quality_weight,
                "time_weight": self.time_weight
            },
            "market_conditions": self.market_maker.current_market_conditions
        }


if __name__ == "__main__":
    async def demo_task_allocator():
        """Demonstrate task allocator capabilities"""
        allocator = IntelligentTaskAllocator()
        
        # Create mock agents
        agents = []
        for i in range(5):
            agent = BaseAgent(f"agent_{i}")
            agent.capabilities = ["general", "data_analysis"] if i % 2 == 0 else ["programming", "general"]
            agents.append(agent)
            allocator.register_agent(agent, capacity=3)
        
        print("=" * 80)
        print("INTELLIGENT TASK ALLOCATOR DEMONSTRATION")
        print("=" * 80)
        
        # Submit various tasks
        tasks = [
            AllocationTask(
                task_id="task_001",
                description="Analyze customer satisfaction data and generate report",
                requirements={"data_size": "large", "urgency": "medium"},
                priority=0.8,
                max_budget=50.0,
                domain="data_analysis",
                required_skills=["data_analysis"],
                deadline=datetime.now() + timedelta(hours=4)
            ),
            AllocationTask(
                task_id="task_002", 
                description="Implement user authentication system",
                requirements={"complexity": "medium", "security": "high"},
                priority=0.9,
                max_budget=100.0,
                domain="development",
                required_skills=["programming"],
                deadline=datetime.now() + timedelta(hours=8)
            ),
            AllocationTask(
                task_id="task_003",
                description="Optimize database queries for better performance",
                requirements={"database": "postgresql", "performance": "critical"},
                priority=0.7,
                max_budget=75.0,
                domain="optimization",
                required_skills=["programming", "optimization"]
            )
        ]
        
        # Submit tasks
        for task in tasks:
            task_id = await allocator.submit_task(task)
            print(f"Submitted task: {task_id}")
        
        # Wait for allocations to complete
        await asyncio.sleep(5)
        
        # Show results
        print("\nALLOCATION RESULTS:")
        print("-" * 40)
        
        for task_id, task in allocator.tasks.items():
            print(f"Task {task_id}:")
            print(f"  Status: {task.status.value}")
            print(f"  Allocated to: {task.allocated_agent}")
            print(f"  Bids received: {len(task.bids)}")
            if task.bids:
                winning_bid = next((b for b in task.bids if b.agent_id == task.allocated_agent), None)
                if winning_bid:
                    print(f"  Winning bid: ${winning_bid.bid_amount:.2f}, {winning_bid.estimated_time:.1f} min")
            print()
        
        # Show statistics
        stats = await allocator.get_allocation_statistics()
        print("SYSTEM STATISTICS:")
        print("-" * 40)
        print(f"Total tasks: {stats['task_statistics']['total_tasks']}")
        print(f"Success rate: {stats['task_statistics']['success_rate']:.2%}")
        print(f"Average bids per task: {stats['market_statistics']['average_bids_per_task']:.1f}")
        print(f"Average cost: ${stats['market_statistics']['average_cost']:.2f}")
        print()
        
        print("AGENT PERFORMANCE:")
        print("-" * 40)
        for agent_id, agent_stats in stats['agent_statistics'].items():
            print(f"{agent_id}:")
            print(f"  Tasks completed: {agent_stats['total_tasks']}")
            print(f"  Success rate: {agent_stats['success_rate']:.2%}")
            print(f"  Reliability: {agent_stats['reliability_score']:.3f}")
            print(f"  Current load: {agent_stats['current_load']}")
    
    # Run demonstration
    asyncio.run(demo_task_allocator())