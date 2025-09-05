"""
Resource Allocation Multi-Agent RL Environment
Dynamic resource allocation with fairness, efficiency, and adaptive demand
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import heapq
from collections import defaultdict

from .base_environment import (
    BaseMultiAgentEnvironment, 
    AgentConfig, 
    EnvironmentState,
    RewardShaper
)
from utils.observability.logging import get_logger

logger = get_logger(__name__)

class ResourceType(Enum):
    """Types of resources in the system"""
    COMPUTE = "compute"
    MEMORY = "memory"
    STORAGE = "storage"
    BANDWIDTH = "bandwidth"
    ENERGY = "energy"
    HUMAN_RESOURCES = "human_resources"

class PriorityLevel(Enum):
    """Priority levels for resource requests"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5

@dataclass
class Resource:
    """Resource definition"""
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    unit_cost: float
    efficiency_factor: float = 1.0  # Resource efficiency multiplier
    maintenance_cost: float = 0.0
    depreciation_rate: float = 0.001  # Per step depreciation
    
    def allocate(self, amount: float) -> bool:
        """Allocate resource if available"""
        if self.available_capacity >= amount:
            self.available_capacity -= amount
            return True
        return False
    
    def deallocate(self, amount: float):
        """Return resource to available pool"""
        self.available_capacity = min(self.total_capacity, 
                                    self.available_capacity + amount)
    
    def get_utilization(self) -> float:
        """Get current utilization rate"""
        if self.total_capacity == 0:
            return 0.0
        return 1.0 - (self.available_capacity / self.total_capacity)

@dataclass
class ResourceRequest:
    """Resource allocation request"""
    id: str
    requester_id: str
    resource_type: ResourceType
    amount: float
    duration: int  # Duration in steps
    priority: PriorityLevel
    deadline: Optional[int] = None  # Step by which resource is needed
    utility_function: Optional[callable] = None  # Custom utility calculation
    created_at: int = 0
    allocated_amount: float = 0.0
    status: str = "pending"  # pending, allocated, fulfilled, expired, denied
    
    def calculate_utility(self, allocated_amount: float) -> float:
        """Calculate utility based on allocated amount"""
        if self.utility_function:
            return self.utility_function(allocated_amount, self.amount)
        
        # Default utility: diminishing returns
        if self.amount == 0:
            return 0.0
        
        ratio = allocated_amount / self.amount
        return ratio * (2 - ratio)  # Concave utility function

@dataclass 
class Allocation:
    """Resource allocation record"""
    request_id: str
    agent_id: str
    resource_type: ResourceType
    amount: float
    start_step: int
    end_step: int
    cost: float
    efficiency: float

class ResourceAgent:
    """Individual agent in resource allocation system"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        
        # Resource budget and usage
        self.budget = config.initial_balance
        self.spent = 0.0
        self.resource_credits = 1000.0  # Alternative currency
        
        # Active requests and allocations
        self.active_requests: List[ResourceRequest] = []
        self.active_allocations: List[Allocation] = []
        self.allocation_history: List[Allocation] = []
        
        # Performance metrics
        self.metrics = {
            'requests_fulfilled': 0,
            'requests_denied': 0,
            'average_utility': 0.0,
            'resource_efficiency': 1.0,
            'fairness_score': 0.5,
            'collaboration_score': 0.5,
            'budget_utilization': 0.0
        }
        
        # Learning and adaptation
        self.demand_pattern = {}  # Historical demand by resource type
        self.success_rates = {}   # Success rates by request characteristics
        
        # Preferences and constraints
        self.resource_preferences = {rt: random.uniform(0.5, 1.5) 
                                   for rt in ResourceType}
        self.max_concurrent_requests = 5
        self.risk_tolerance = config.risk_tolerance
    
    def create_request(self, resource_type: ResourceType, amount: float, 
                      duration: int, priority: PriorityLevel, 
                      deadline: Optional[int] = None) -> ResourceRequest:
        """Create a new resource request"""
        request_id = f"{self.agent_id}_req_{len(self.active_requests) + len(self.allocation_history) + 1}"
        
        request = ResourceRequest(
            id=request_id,
            requester_id=self.agent_id,
            resource_type=resource_type,
            amount=amount,
            duration=duration,
            priority=priority,
            deadline=deadline,
            created_at=0  # Will be set by environment
        )
        
        self.active_requests.append(request)
        return request
    
    def update_metrics(self, current_step: int):
        """Update agent performance metrics"""
        # Calculate fulfillment rates
        total_requests = self.metrics['requests_fulfilled'] + self.metrics['requests_denied']
        if total_requests > 0:
            fulfillment_rate = self.metrics['requests_fulfilled'] / total_requests
        else:
            fulfillment_rate = 0.5
        
        # Calculate average utility from recent allocations
        recent_allocations = [a for a in self.allocation_history 
                            if current_step - a.end_step <= 50]  # Last 50 steps
        if recent_allocations:
            avg_efficiency = np.mean([a.efficiency for a in recent_allocations])
            self.metrics['resource_efficiency'] = avg_efficiency
        
        # Budget utilization
        self.metrics['budget_utilization'] = self.spent / self.budget if self.budget > 0 else 0.0
        
        # Update demand patterns
        for resource_type in ResourceType:
            recent_requests = [req for req in self.allocation_history 
                             if req.resource_type == resource_type 
                             and current_step - req.end_step <= 100]
            if recent_requests:
                avg_demand = np.mean([req.amount for req in recent_requests])
                self.demand_pattern[resource_type] = avg_demand

class ResourceAllocationEnvironment(BaseMultiAgentEnvironment):
    """
    Multi-agent resource allocation environment with dynamic demand and fairness considerations
    """
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        resource_types: Optional[List[ResourceType]] = None,
        total_resources: Optional[Dict[ResourceType, float]] = None,
        allocation_mechanism: str = "proportional",  # proportional, auction, priority, fair_share
        fairness_weight: float = 0.3,
        efficiency_weight: float = 0.7,
        max_steps: int = 1000,
        seed: Optional[int] = None
    ):
        # Initialize resource types and capacities
        self.resource_types = resource_types or list(ResourceType)
        self.total_resources = total_resources or self._create_default_resources()
        
        # Create resource pool
        self.resources = {
            rt: Resource(
                resource_type=rt,
                total_capacity=self.total_resources[rt],
                available_capacity=self.total_resources[rt],
                unit_cost=self._get_default_unit_cost(rt),
                efficiency_factor=random.uniform(0.8, 1.2)
            )
            for rt in self.resource_types
        }
        
        # Resource agents
        self.resource_agents = {config.agent_id: ResourceAgent(config.agent_id, config) 
                              for config in agent_configs}
        
        # Allocation mechanism
        self.allocation_mechanism = allocation_mechanism
        self.fairness_weight = fairness_weight
        self.efficiency_weight = efficiency_weight
        
        # Request management
        self.pending_requests: List[ResourceRequest] = []
        self.active_allocations: List[Allocation] = []
        self.allocation_history: List[Allocation] = []
        self.request_counter = 0
        
        # Market dynamics
        self.demand_multipliers = {rt: 1.0 for rt in self.resource_types}
        self.price_multipliers = {rt: 1.0 for rt in self.resource_types}
        self.congestion_factors = {rt: 0.0 for rt in self.resource_types}
        
        super().__init__(agent_configs, max_steps, reward_shaping=True, seed=seed)
        logger.info(f"Initialized resource allocation environment with {len(self.resource_types)} resource types")
    
    def _create_default_resources(self) -> Dict[ResourceType, float]:
        """Create default resource capacities"""
        return {
            ResourceType.COMPUTE: 1000.0,
            ResourceType.MEMORY: 2000.0,
            ResourceType.STORAGE: 5000.0,
            ResourceType.BANDWIDTH: 1500.0,
            ResourceType.ENERGY: 800.0,
            ResourceType.HUMAN_RESOURCES: 50.0
        }
    
    def _get_default_unit_cost(self, resource_type: ResourceType) -> float:
        """Get default unit cost for resource type"""
        costs = {
            ResourceType.COMPUTE: 0.1,
            ResourceType.MEMORY: 0.05,
            ResourceType.STORAGE: 0.01,
            ResourceType.BANDWIDTH: 0.08,
            ResourceType.ENERGY: 0.15,
            ResourceType.HUMAN_RESOURCES: 1.0
        }
        return costs.get(resource_type, 0.1)
    
    def _get_observation_dimension(self) -> int:
        """Calculate observation dimension for resource allocation"""
        # Global features + resource features + agent features + market features
        global_features = 6  # step, total_agents, fairness_index, efficiency_index, etc.
        resource_features = len(self.resource_types) * 5  # capacity, utilization, price, demand, congestion
        agent_features = 10  # budget, active_requests, allocations, metrics, etc.
        market_features = len(self.resource_types) * 2  # demand_multiplier, price_multiplier
        
        return global_features + resource_features + agent_features + market_features
    
    def _get_action_dimension(self) -> int:
        """Calculate action dimension for resource allocation"""
        # Actions: [request_amounts, priorities, bid_prices, duration_preferences]
        num_resources = len(self.resource_types)
        
        request_amounts = num_resources  # Amount to request for each resource type
        priority_weights = num_resources  # Priority weighting for each resource type
        bid_prices = num_resources  # Bid prices for auction mechanism
        duration_preferences = num_resources  # Preferred duration for each resource type
        
        return request_amounts + priority_weights + bid_prices + duration_preferences
    
    def _initialize_environment_state(self) -> EnvironmentState:
        """Initialize resource allocation environment state"""
        # Initialize agent states
        agent_states = {}
        for agent_id, res_agent in self.resource_agents.items():
            agent_states[agent_id] = {
                'balance': res_agent.budget,
                'spent': res_agent.spent,
                'resource_credits': res_agent.resource_credits,
                'active_requests': len(res_agent.active_requests),
                'active_allocations': len(res_agent.active_allocations),
                'fulfillment_rate': 0.5,  # Initial fulfillment rate
                'resource_efficiency': 1.0,
                'fairness_score': 0.5,
                'utilization': len(res_agent.active_allocations) / self.max_concurrent_requests,
                'success_rate': 0.5,
                'risk_level': res_agent.risk_tolerance,
                'collaboration_score': 0.5,
                'connections': []  # Will be populated based on resource sharing
            }
        
        # Global state
        global_state = {
            'total_resources': sum(r.total_capacity for r in self.resources.values()),
            'available_resources': sum(r.available_capacity for r in self.resources.values()),
            'resource_utilization': 1.0 - sum(r.available_capacity for r in self.resources.values()) / 
                                  sum(r.total_capacity for r in self.resources.values()),
            'pending_requests': len(self.pending_requests),
            'active_allocations': len(self.active_allocations),
            'fairness_index': self._calculate_fairness_index(),
            'efficiency_index': self._calculate_efficiency_index()
        }
        
        # Market conditions
        market_conditions = {
            'demand_pressure': np.mean(list(self.demand_multipliers.values())),
            'price_volatility': np.std(list(self.price_multipliers.values())),
            'resource_scarcity': global_state['resource_utilization'],
            'competition_level': len(self.pending_requests) / len(self.resource_agents)
        }
        
        return EnvironmentState(
            global_state=global_state,
            agent_states=agent_states,
            market_conditions=market_conditions,
            timestamp=datetime.now(),
            step_count=0
        )
    
    def _execute_actions(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> EnvironmentState:
        """Execute resource allocation actions"""
        # Process agent actions (create requests)
        new_requests = []
        for agent_id, action_vector in actions.items():
            if agent_id in self.resource_agents:
                agent_requests = self._process_agent_actions(agent_id, action_vector)
                new_requests.extend(agent_requests)
        
        # Add new requests to pending queue
        self.pending_requests.extend(new_requests)
        
        # Process resource allocation
        self._allocate_resources()
        
        # Update resource states (depreciation, maintenance)
        self._update_resource_states()
        
        # Update market conditions
        self._update_market_conditions()
        
        # Clean up expired allocations
        self._cleanup_expired_allocations()
        
        # Update agent metrics
        for agent_id, res_agent in self.resource_agents.items():
            res_agent.update_metrics(self.step_count + 1)
        
        # Calculate updated agent states
        updated_agent_states = {}
        for agent_id, res_agent in self.resource_agents.items():
            # Calculate connections based on resource sharing
            connections = self._get_agent_connections(agent_id)
            
            updated_agent_states[agent_id] = {
                'balance': res_agent.budget,
                'spent': res_agent.spent,
                'resource_credits': res_agent.resource_credits,
                'active_requests': len(res_agent.active_requests),
                'active_allocations': len(res_agent.active_allocations),
                'fulfillment_rate': res_agent.metrics.get('requests_fulfilled', 0) / 
                                  max(1, res_agent.metrics.get('requests_fulfilled', 0) + 
                                      res_agent.metrics.get('requests_denied', 0)),
                'resource_efficiency': res_agent.metrics['resource_efficiency'],
                'fairness_score': res_agent.metrics['fairness_score'],
                'utilization': len(res_agent.active_allocations) / 10.0,  # Normalized
                'success_rate': res_agent.metrics.get('requests_fulfilled', 0) / 
                              max(1, res_agent.metrics.get('requests_fulfilled', 0) + 
                                  res_agent.metrics.get('requests_denied', 0)),
                'risk_level': res_agent.risk_tolerance,
                'collaboration_score': self._calculate_collaboration_score(agent_id),
                'connections': connections
            }
        
        # Update global state
        updated_global_state = {
            'total_resources': sum(r.total_capacity for r in self.resources.values()),
            'available_resources': sum(r.available_capacity for r in self.resources.values()),
            'resource_utilization': 1.0 - sum(r.available_capacity for r in self.resources.values()) / 
                                  sum(r.total_capacity for r in self.resources.values()),
            'pending_requests': len(self.pending_requests),
            'active_allocations': len(self.active_allocations),
            'fairness_index': self._calculate_fairness_index(),
            'efficiency_index': self._calculate_efficiency_index()
        }
        
        # Update market conditions
        updated_market_conditions = {
            'demand_pressure': np.mean(list(self.demand_multipliers.values())),
            'price_volatility': np.std(list(self.price_multipliers.values())),
            'resource_scarcity': updated_global_state['resource_utilization'],
            'competition_level': len(self.pending_requests) / max(1, len(self.resource_agents))
        }
        
        return EnvironmentState(
            global_state=updated_global_state,
            agent_states=updated_agent_states,
            market_conditions=updated_market_conditions,
            timestamp=datetime.now(),
            step_count=state.step_count + 1
        )
    
    def _process_agent_actions(self, agent_id: str, action_vector: np.ndarray) -> List[ResourceRequest]:
        """Process individual agent's resource requests"""
        res_agent = self.resource_agents[agent_id]
        new_requests = []
        
        num_resources = len(self.resource_types)
        
        # Decode action vector
        request_amounts = action_vector[:num_resources]
        priority_weights = action_vector[num_resources:num_resources*2]
        bid_prices = action_vector[num_resources*2:num_resources*3]
        duration_preferences = action_vector[num_resources*3:num_resources*4]
        
        # Create requests for resources with significant demand
        for i, resource_type in enumerate(self.resource_types):
            if i < len(request_amounts):
                # Scale and threshold request amounts
                request_amount = max(0, request_amounts[i] * 100)
                
                if request_amount > 5.0:  # Minimum request threshold
                    # Determine priority based on priority weights
                    priority_weight = priority_weights[i] if i < len(priority_weights) else 0.0
                    priority_level = self._convert_to_priority_level(priority_weight)
                    
                    # Duration (1 to 20 steps)
                    duration = max(1, int(abs(duration_preferences[i] if i < len(duration_preferences) else 0.0) * 20))
                    
                    # Check if agent can afford the request
                    estimated_cost = request_amount * self.resources[resource_type].unit_cost * duration
                    if res_agent.budget - res_agent.spent >= estimated_cost:
                        
                        # Create request
                        request = res_agent.create_request(
                            resource_type=resource_type,
                            amount=request_amount,
                            duration=duration,
                            priority=priority_level,
                            deadline=self.step_count + duration + random.randint(5, 15)
                        )
                        
                        request.created_at = self.step_count + 1
                        new_requests.append(request)
                        self.request_counter += 1
        
        return new_requests
    
    def _convert_to_priority_level(self, priority_weight: float) -> PriorityLevel:
        """Convert priority weight to priority level"""
        # Map [-1, 1] to priority levels
        normalized = (priority_weight + 1) / 2  # Convert to [0, 1]
        
        if normalized < 0.2:
            return PriorityLevel.LOW
        elif normalized < 0.4:
            return PriorityLevel.NORMAL
        elif normalized < 0.6:
            return PriorityLevel.HIGH
        elif normalized < 0.8:
            return PriorityLevel.CRITICAL
        else:
            return PriorityLevel.EMERGENCY
    
    def _allocate_resources(self):
        """Allocate resources to pending requests based on allocation mechanism"""
        if not self.pending_requests:
            return {}
        
        if self.allocation_mechanism == "proportional":
            self._allocate_proportional()
        elif self.allocation_mechanism == "priority":
            self._allocate_priority_based()
        elif self.allocation_mechanism == "auction":
            self._allocate_auction_based()
        elif self.allocation_mechanism == "fair_share":
            self._allocate_fair_share()
        else:
            self._allocate_proportional()  # Default
    
    def _allocate_proportional(self):
        """Proportional allocation based on request amounts and priorities"""
        allocated_requests = []
        
        # Group requests by resource type
        requests_by_type = defaultdict(list)
        for req in self.pending_requests:
            requests_by_type[req.resource_type].append(req)
        
        # Allocate each resource type separately
        for resource_type, requests in requests_by_type.items():
            available = self.resources[resource_type].available_capacity
            
            if available <= 0 or not requests:
                continue
            
            # Calculate weighted demand
            total_weighted_demand = sum(
                req.amount * req.priority.value for req in requests
            )
            
            if total_weighted_demand <= available:
                # Enough resources for everyone
                for req in requests:
                    self._create_allocation(req, req.amount)
                    allocated_requests.append(req)
            else:
                # Proportional allocation
                for req in requests:
                    weighted_demand = req.amount * req.priority.value
                    allocation_ratio = weighted_demand / total_weighted_demand
                    allocated_amount = min(req.amount, available * allocation_ratio)
                    
                    if allocated_amount > 0.1:  # Minimum allocation threshold
                        self._create_allocation(req, allocated_amount)
                        allocated_requests.append(req)
        
        # Remove allocated requests from pending
        self.pending_requests = [req for req in self.pending_requests 
                               if req not in allocated_requests]
    
    def _allocate_priority_based(self):
        """Priority-based allocation (highest priority first)"""
        allocated_requests = []
        
        # Sort by priority and creation time
        sorted_requests = sorted(self.pending_requests, 
                               key=lambda x: (-x.priority.value, x.created_at))
        
        for req in sorted_requests:
            resource = self.resources[req.resource_type]
            
            if resource.available_capacity >= req.amount:
                # Full allocation
                self._create_allocation(req, req.amount)
                allocated_requests.append(req)
            elif resource.available_capacity > 0.1:  # Partial allocation possible
                # Partial allocation for high priority requests
                if req.priority.value >= PriorityLevel.HIGH.value:
                    allocated_amount = min(req.amount, resource.available_capacity)
                    self._create_allocation(req, allocated_amount)
                    allocated_requests.append(req)
        
        # Remove allocated requests
        self.pending_requests = [req for req in self.pending_requests 
                               if req not in allocated_requests]
    
    def _allocate_fair_share(self):
        """Fair share allocation ensuring equal access"""
        allocated_requests = []
        
        # Group requests by resource type
        requests_by_type = defaultdict(list)
        for req in self.pending_requests:
            requests_by_type[req.resource_type].append(req)
        
        for resource_type, requests in requests_by_type.items():
            available = self.resources[resource_type].available_capacity
            
            if available <= 0 or not requests:
                continue
            
            # Calculate fair share
            num_agents = len(set(req.requester_id for req in requests))
            fair_share_per_agent = available / num_agents
            
            # Group by agent
            requests_by_agent = defaultdict(list)
            for req in requests:
                requests_by_agent[req.requester_id].append(req)
            
            # Allocate fair share to each agent
            for agent_id, agent_requests in requests_by_agent.items():
                agent_share = fair_share_per_agent
                
                # Distribute agent's share among their requests
                total_agent_demand = sum(req.amount for req in agent_requests)
                
                for req in agent_requests:
                    if total_agent_demand > 0:
                        request_ratio = req.amount / total_agent_demand
                        allocated_amount = min(req.amount, agent_share * request_ratio)
                        
                        if allocated_amount > 0.1:
                            self._create_allocation(req, allocated_amount)
                            allocated_requests.append(req)
        
        # Remove allocated requests
        self.pending_requests = [req for req in self.pending_requests 
                               if req not in allocated_requests]
    
    def _allocate_auction_based(self):
        """Auction-based allocation (simplified second-price auction)"""
        allocated_requests = []
        
        # Group requests by resource type  
        requests_by_type = defaultdict(list)
        for req in self.pending_requests:
            # Use priority as bid price for simplicity
            requests_by_type[req.resource_type].append((req, req.priority.value))
        
        for resource_type, request_bid_pairs in requests_by_type.items():
            available = self.resources[resource_type].available_capacity
            
            if available <= 0 or not request_bid_pairs:
                continue
            
            # Sort by bid (priority) in descending order
            sorted_pairs = sorted(request_bid_pairs, key=lambda x: -x[1])
            
            for req, bid in sorted_pairs:
                if available >= req.amount:
                    self._create_allocation(req, req.amount)
                    allocated_requests.append(req)
                    available -= req.amount
                elif available > 0.1:
                    # Partial allocation for remaining resources
                    allocated_amount = min(req.amount, available)
                    self._create_allocation(req, allocated_amount)
                    allocated_requests.append(req)
                    break
        
        # Remove allocated requests
        self.pending_requests = [req for req in self.pending_requests 
                               if req not in allocated_requests]
    
    def _create_allocation(self, request: ResourceRequest, allocated_amount: float):
        """Create a resource allocation"""
        resource = self.resources[request.resource_type]
        
        if resource.allocate(allocated_amount):
            # Calculate cost
            cost = allocated_amount * resource.unit_cost * request.duration
            cost *= self.price_multipliers[request.resource_type]  # Apply price multiplier
            
            # Create allocation record
            allocation = Allocation(
                request_id=request.id,
                agent_id=request.requester_id,
                resource_type=request.resource_type,
                amount=allocated_amount,
                start_step=self.step_count + 1,
                end_step=self.step_count + 1 + request.duration,
                cost=cost,
                efficiency=resource.efficiency_factor
            )
            
            self.active_allocations.append(allocation)
            
            # Update agent
            res_agent = self.resource_agents[request.requester_id]
            res_agent.active_allocations.append(allocation)
            res_agent.spent += cost
            res_agent.budget -= cost
            
            # Update request status
            request.allocated_amount = allocated_amount
            request.status = "allocated"
            
            # Update agent metrics
            if allocated_amount >= request.amount * 0.8:  # 80% fulfillment threshold
                res_agent.metrics['requests_fulfilled'] += 1
            else:
                res_agent.metrics['requests_denied'] += 1
                
            # Remove request from agent's active requests
            if request in res_agent.active_requests:
                res_agent.active_requests.remove(request)
    
    def _cleanup_expired_allocations(self):
        """Clean up expired resource allocations"""
        current_step = self.step_count + 1
        expired_allocations = []
        
        for allocation in self.active_allocations:
            if current_step >= allocation.end_step:
                # Return resources
                resource = self.resources[allocation.resource_type]
                resource.deallocate(allocation.amount)
                
                # Move to history
                expired_allocations.append(allocation)
                self.allocation_history.append(allocation)
                
                # Update agent
                res_agent = self.resource_agents[allocation.agent_id]
                if allocation in res_agent.active_allocations:
                    res_agent.active_allocations.remove(allocation)
                res_agent.allocation_history.append(allocation)
        
        # Remove expired allocations
        self.active_allocations = [a for a in self.active_allocations 
                                 if a not in expired_allocations]
        
        # Clean up expired requests
        expired_requests = []
        for req in self.pending_requests:
            if (req.deadline and current_step > req.deadline) or \
               (current_step - req.created_at > 50):  # Max wait time
                expired_requests.append(req)
                
                # Update agent metrics
                res_agent = self.resource_agents[req.requester_id]
                res_agent.metrics['requests_denied'] += 1
                
                if req in res_agent.active_requests:
                    res_agent.active_requests.remove(req)
        
        self.pending_requests = [req for req in self.pending_requests 
                               if req not in expired_requests]
    
    def _update_resource_states(self):
        """Update resource states (depreciation, maintenance)"""
        for resource in self.resources.values():
            # Apply depreciation
            capacity_loss = resource.total_capacity * resource.depreciation_rate
            resource.total_capacity = max(0, resource.total_capacity - capacity_loss)
            resource.available_capacity = min(resource.available_capacity, 
                                            resource.total_capacity)
            
            # Random efficiency changes
            if random.random() < 0.01:  # 1% chance per step
                efficiency_change = random.uniform(-0.05, 0.05)
                resource.efficiency_factor = max(0.5, min(1.5, 
                    resource.efficiency_factor + efficiency_change))
    
    def _update_market_conditions(self):
        """Update market demand and pricing"""
        for resource_type in self.resource_types:
            # Update demand multiplier based on recent requests
            recent_requests = [req for req in self.pending_requests + 
                             [a.request_id for a in self.active_allocations[-10:]]
                             if hasattr(req, 'resource_type') and req.resource_type == resource_type]
            
            if len(recent_requests) > 5:
                self.demand_multipliers[resource_type] = min(2.0, 
                    self.demand_multipliers[resource_type] * 1.05)
            elif len(recent_requests) < 2:
                self.demand_multipliers[resource_type] = max(0.5,
                    self.demand_multipliers[resource_type] * 0.98)
            
            # Update price multiplier based on utilization
            utilization = self.resources[resource_type].get_utilization()
            if utilization > 0.8:
                self.price_multipliers[resource_type] = min(2.0,
                    self.price_multipliers[resource_type] * 1.02)
            elif utilization < 0.4:
                self.price_multipliers[resource_type] = max(0.7,
                    self.price_multipliers[resource_type] * 0.99)
            
            # Update congestion factor
            self.congestion_factors[resource_type] = utilization
    
    def _calculate_fairness_index(self) -> float:
        """Calculate Jain's fairness index for resource allocation"""
        if not self.resource_agents:
            return 1.0
        
        # Calculate total resource value allocated to each agent
        agent_allocations = defaultdict(float)
        for allocation in self.active_allocations + self.allocation_history[-50:]:
            value = allocation.amount * self.resources[allocation.resource_type].unit_cost
            agent_allocations[allocation.agent_id] += value
        
        # Ensure all agents are represented
        for agent_id in self.resource_agents.keys():
            if agent_id not in agent_allocations:
                agent_allocations[agent_id] = 0.0
        
        allocations = list(agent_allocations.values())
        n = len(allocations)
        
        if n == 0:
            return 1.0
        
        sum_x = sum(allocations)
        sum_x_squared = sum(x**2 for x in allocations)
        
        if sum_x_squared == 0:
            return 1.0
        
        fairness_index = (sum_x**2) / (n * sum_x_squared)
        return fairness_index
    
    def _calculate_efficiency_index(self) -> float:
        """Calculate resource utilization efficiency"""
        if not self.resources:
            return 0.0
        
        total_capacity = sum(r.total_capacity for r in self.resources.values())
        total_used = sum(r.total_capacity - r.available_capacity 
                        for r in self.resources.values())
        
        if total_capacity == 0:
            return 0.0
        
        return total_used / total_capacity
    
    def _get_agent_connections(self, agent_id: str) -> List[str]:
        """Get agents that share resources with this agent"""
        connections = set()
        res_agent = self.resource_agents[agent_id]
        
        # Find agents with overlapping resource usage
        agent_resources = set()
        for allocation in res_agent.active_allocations:
            agent_resources.add(allocation.resource_type)
        
        for other_id, other_agent in self.resource_agents.items():
            if other_id != agent_id:
                other_resources = set()
                for allocation in other_agent.active_allocations:
                    other_resources.add(allocation.resource_type)
                
                if agent_resources.intersection(other_resources):
                    connections.add(other_id)
        
        return list(connections)
    
    def _calculate_collaboration_score(self, agent_id: str) -> float:
        """Calculate collaboration score based on resource sharing patterns"""
        res_agent = self.resource_agents[agent_id]
        
        if not res_agent.allocation_history:
            return 0.5
        
        # Analyze recent allocation patterns
        recent_allocations = res_agent.allocation_history[-20:]  # Last 20 allocations
        
        # Score based on resource diversity (sharing different types)
        resource_types_used = set(alloc.resource_type for alloc in recent_allocations)
        diversity_score = len(resource_types_used) / len(self.resource_types)
        
        # Score based on allocation efficiency (using resources effectively)
        if recent_allocations:
            avg_efficiency = np.mean([alloc.efficiency for alloc in recent_allocations])
            efficiency_score = avg_efficiency
        else:
            efficiency_score = 0.5
        
        # Score based on fairness (not hogging resources)
        total_allocated = sum(alloc.amount for alloc in recent_allocations)
        avg_agent_allocation = np.mean([
            sum(alloc.amount for alloc in agent.allocation_history[-20:])
            for agent in self.resource_agents.values()
        ])
        
        if avg_agent_allocation > 0:
            fairness_ratio = min(2.0, total_allocated / avg_agent_allocation)
            fairness_score = 1.0 / fairness_ratio if fairness_ratio > 1.0 else 1.0
        else:
            fairness_score = 0.5
        
        # Combine scores
        collaboration_score = (diversity_score * 0.3 + 
                             efficiency_score * 0.4 + 
                             fairness_score * 0.3)
        
        return collaboration_score
    
    def _calculate_rewards(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> Dict[str, float]:
        """Calculate rewards for resource allocation agents"""
        rewards = {}
        
        for agent_id, res_agent in self.resource_agents.items():
            # Utility-based reward from allocations
            utility_reward = 0.0
            for allocation in res_agent.active_allocations:
                # Find the original request
                utility_value = allocation.amount * allocation.efficiency
                utility_reward += utility_value / 100.0  # Scale reward
            
            # Fairness reward
            fairness_reward = 0.0
            if self.reward_shaping:
                fairness_index = self._calculate_fairness_index()
                if fairness_index > 0.8:  # High fairness
                    fairness_reward = (fairness_index - 0.8) * 0.5
            
            # Efficiency reward  
            efficiency_reward = 0.0
            if res_agent.allocation_history:
                recent_allocations = res_agent.allocation_history[-10:]
                avg_efficiency = np.mean([alloc.efficiency for alloc in recent_allocations])
                efficiency_reward = (avg_efficiency - 1.0) * 0.2  # Bonus for above-average efficiency
            
            # Budget utilization penalty/reward
            budget_reward = 0.0
            utilization = res_agent.spent / res_agent.budget if res_agent.budget > 0 else 0
            if 0.7 <= utilization <= 0.9:  # Optimal budget usage
                budget_reward = 0.1
            elif utilization > 0.95:  # Over-spending penalty
                budget_reward = -0.2
            
            # Collaboration bonus
            collaboration_bonus = 0.0
            if self.reward_shaping:
                collab_score = state.agent_states[agent_id]['collaboration_score']
                collaboration_bonus = (collab_score - 0.5) * 0.1
            
            # Request fulfillment reward
            fulfillment_reward = 0.0
            fulfillment_rate = state.agent_states[agent_id]['fulfillment_rate']
            fulfillment_reward = (fulfillment_rate - 0.5) * 0.3
            
            # Penalty for resource waste
            waste_penalty = 0.0
            for allocation in res_agent.active_allocations:
                if allocation.efficiency < 0.8:  # Low efficiency usage
                    waste_penalty += 0.05
            
            # Combine all reward components
            total_reward = (utility_reward + 
                          fairness_reward + 
                          efficiency_reward + 
                          budget_reward + 
                          collaboration_bonus + 
                          fulfillment_reward - 
                          waste_penalty)
            
            rewards[agent_id] = total_reward
        
        return rewards
    
    def get_allocation_metrics(self) -> Dict[str, Any]:
        """Get comprehensive allocation metrics"""
        return {
            'environment': {
                'step': self.step_count,
                'episode': self.episode_count,
                'fairness_index': self._calculate_fairness_index(),
                'efficiency_index': self._calculate_efficiency_index(),
                'allocation_mechanism': self.allocation_mechanism
            },
            'resources': {
                rt.value: {
                    'total_capacity': resource.total_capacity,
                    'available_capacity': resource.available_capacity,
                    'utilization': resource.get_utilization(),
                    'unit_cost': resource.unit_cost,
                    'efficiency_factor': resource.efficiency_factor,
                    'demand_multiplier': self.demand_multipliers[rt],
                    'price_multiplier': self.price_multipliers[rt]
                }
                for rt, resource in self.resources.items()
            },
            'requests_and_allocations': {
                'pending_requests': len(self.pending_requests),
                'active_allocations': len(self.active_allocations),
                'completed_allocations': len(self.allocation_history),
                'total_requests': self.request_counter
            },
            'agents': {
                agent_id: {
                    'budget_remaining': agent.budget,
                    'total_spent': agent.spent,
                    'active_requests': len(agent.active_requests),
                    'active_allocations': len(agent.active_allocations),
                    'fulfillment_rate': agent.metrics.get('requests_fulfilled', 0) / 
                                       max(1, agent.metrics.get('requests_fulfilled', 0) + 
                                           agent.metrics.get('requests_denied', 0)),
                    'resource_efficiency': agent.metrics['resource_efficiency'],
                    'fairness_score': agent.metrics['fairness_score']
                }
                for agent_id, agent in self.resource_agents.items()
            }
        }
    
    def render(self, mode: str = "human"):
        """Render the resource allocation environment"""
        if mode == "human":
            self._render_allocation_text()
    
    def _render_allocation_text(self):
        """Text rendering of resource allocation state"""
        print(f"\n--- Resource Allocation Environment Step {self.step_count} ---")
        print(f"Allocation Mechanism: {self.allocation_mechanism.title()}")
        print(f"Fairness Index: {self._calculate_fairness_index():.3f}")
        print(f"Efficiency Index: {self._calculate_efficiency_index():.3f}")
        print(f"Pending Requests: {len(self.pending_requests)}")
        print(f"Active Allocations: {len(self.active_allocations)}")
        
        print("\nResource Status:")
        for resource_type, resource in self.resources.items():
            utilization = resource.get_utilization()
            print(f"  {resource_type.value.title()}:")
            print(f"    Capacity: {resource.available_capacity:.1f}/{resource.total_capacity:.1f}")
            print(f"    Utilization: {utilization:.1%}")
            print(f"    Price Multiplier: {self.price_multipliers[resource_type]:.2f}")
        
        print("\nAgent Status:")
        for agent_id, res_agent in self.resource_agents.items():
            fulfillment_rate = res_agent.metrics.get('requests_fulfilled', 0) / max(1, 
                res_agent.metrics.get('requests_fulfilled', 0) + res_agent.metrics.get('requests_denied', 0))
            
            print(f"  {agent_id}:")
            print(f"    Budget: ${res_agent.budget:,.2f} (Spent: ${res_agent.spent:,.2f})")
            print(f"    Active Requests: {len(res_agent.active_requests)}")
            print(f"    Active Allocations: {len(res_agent.active_allocations)}")
            print(f"    Fulfillment Rate: {fulfillment_rate:.1%}")
            print(f"    Resource Efficiency: {res_agent.metrics['resource_efficiency']:.2f}")