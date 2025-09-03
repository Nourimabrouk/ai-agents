"""
Supply Chain Multi-Agent RL Environment
Complex supply chain simulation with demand forecasting, inventory management, and logistics
"""

import numpy as np
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from .base_environment import (
    BaseMultiAgentEnvironment, 
    AgentConfig, 
    EnvironmentState,
    RewardShaper
)
from utils.observability.logging import get_logger

logger = get_logger(__name__)

class NodeType(Enum):
    """Types of supply chain nodes"""
    SUPPLIER = "supplier"
    MANUFACTURER = "manufacturer" 
    DISTRIBUTOR = "distributor"
    RETAILER = "retailer"
    CUSTOMER = "customer"

class EventType(Enum):
    """Supply chain disruption events"""
    DEMAND_SPIKE = "demand_spike"
    SUPPLIER_SHORTAGE = "supplier_shortage"
    LOGISTICS_DELAY = "logistics_delay"
    QUALITY_ISSUE = "quality_issue"
    REGULATORY_CHANGE = "regulatory_change"

@dataclass
class Product:
    """Product in the supply chain"""
    id: str
    name: str
    base_demand: float
    seasonality_factor: float = 1.0
    perishability: float = 0.0  # 0 = non-perishable, 1 = highly perishable
    unit_cost: float = 10.0
    storage_cost_per_unit: float = 0.1
    
@dataclass
class InventoryLevel:
    """Inventory tracking"""
    product_id: str
    quantity: float
    reserved_quantity: float = 0.0
    incoming_quantity: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class Order:
    """Order in the supply chain"""
    id: str
    from_node: str
    to_node: str
    product_id: str
    quantity: float
    unit_price: float
    order_date: datetime
    expected_delivery: datetime
    status: str = "pending"  # pending, confirmed, shipped, delivered, cancelled
    priority: int = 1  # 1 = low, 5 = high

@dataclass
class SupplyChainEvent:
    """Disruption event"""
    event_type: EventType
    affected_nodes: List[str]
    affected_products: List[str]
    impact_magnitude: float
    duration_steps: int
    start_step: int

class SupplyChainAgent:
    """Individual agent in supply chain network"""
    
    def __init__(self, agent_id: str, config: AgentConfig, node_type: NodeType):
        self.agent_id = agent_id
        self.config = config
        self.node_type = node_type
        
        # Inventory management
        self.inventory: Dict[str, InventoryLevel] = {}
        self.capacity: Dict[str, float] = {}  # Storage capacity per product
        
        # Financial tracking
        self.cash_balance = config.initial_balance
        self.revenue = 0.0
        self.costs = 0.0
        
        # Orders
        self.pending_orders: List[Order] = []
        self.order_history: List[Order] = []
        
        # Performance metrics
        self.metrics = {
            'fill_rate': 1.0,  # Orders fulfilled / total orders
            'inventory_turnover': 0.0,
            'service_level': 1.0,
            'cost_efficiency': 1.0,
            'sustainability_score': 0.5
        }
        
        # Relationships
        self.suppliers: List[str] = []
        self.customers: List[str] = []
        
        # AI capabilities
        self.demand_forecasting_accuracy = 0.7
        self.optimization_capability = 0.6
    
    def add_inventory(self, product_id: str, quantity: float, capacity: float = 1000.0):
        """Add inventory tracking for a product"""
        if product_id not in self.inventory:
            self.inventory[product_id] = InventoryLevel(product_id, quantity)
            self.capacity[product_id] = capacity
    
    def get_available_inventory(self, product_id: str) -> float:
        """Get available inventory (not reserved)"""
        if product_id not in self.inventory:
            return 0.0
        inv = self.inventory[product_id]
        return max(0.0, inv.quantity - inv.reserved_quantity)
    
    def reserve_inventory(self, product_id: str, quantity: float) -> bool:
        """Reserve inventory for an order"""
        available = self.get_available_inventory(product_id)
        if available >= quantity:
            self.inventory[product_id].reserved_quantity += quantity
            return True
        return False
    
    def fulfill_order(self, product_id: str, quantity: float) -> bool:
        """Fulfill an order by reducing inventory"""
        if product_id in self.inventory:
            inv = self.inventory[product_id]
            if inv.reserved_quantity >= quantity and inv.quantity >= quantity:
                inv.quantity -= quantity
                inv.reserved_quantity -= quantity
                return True
        return False
    
    def update_metrics(self, orders_received: int, orders_fulfilled: int, 
                      inventory_values: Dict[str, float], sales: float):
        """Update performance metrics"""
        # Fill rate
        if orders_received > 0:
            self.metrics['fill_rate'] = orders_fulfilled / orders_received
        
        # Inventory turnover
        if inventory_values:
            avg_inventory_value = sum(inventory_values.values()) / len(inventory_values)
            if avg_inventory_value > 0:
                self.metrics['inventory_turnover'] = sales / avg_inventory_value
        
        # Cost efficiency
        if self.revenue > 0:
            self.metrics['cost_efficiency'] = (self.revenue - self.costs) / self.revenue
        
        # Service level (simplified - could be more complex)
        self.metrics['service_level'] = self.metrics['fill_rate'] * 0.8 + self.metrics['inventory_turnover'] * 0.2

class SupplyChainEnvironment(BaseMultiAgentEnvironment):
    """
    Multi-agent supply chain optimization environment
    """
    
    def __init__(
        self,
        agent_configs: List[AgentConfig],
        products: List[Product] = None,
        network_topology: Optional[Dict[str, List[str]]] = None,
        max_steps: int = 1000,
        disruption_probability: float = 0.1,
        seasonality_enabled: bool = True,
        seed: Optional[int] = None
    ):
        # Initialize products
        self.products = products or self._create_default_products()
        self.product_dict = {p.id: p for p in self.products}
        
        # Network topology (who can trade with whom)
        self.network_topology = network_topology or self._create_default_topology(agent_configs)
        
        # Supply chain agents
        self.supply_chain_agents = self._create_supply_chain_agents(agent_configs)
        
        # Environment state
        self.disruption_probability = disruption_probability
        self.seasonality_enabled = seasonality_enabled
        self.active_events: List[SupplyChainEvent] = []
        
        # Market conditions
        self.base_demand_multiplier = 1.0
        self.global_cost_inflation = 1.0
        self.logistics_efficiency = 1.0
        
        # Orders and transactions
        self.active_orders: List[Order] = []
        self.completed_orders: List[Order] = []
        self.order_counter = 0
        
        super().__init__(agent_configs, max_steps, reward_shaping=True, seed=seed)
        logger.info(f"Initialized supply chain with {len(self.products)} products and {len(agent_configs)} agents")
    
    def _create_default_products(self) -> List[Product]:
        """Create default products for the supply chain"""
        return [
            Product("raw_material_1", "Steel", 100.0, 1.0, 0.0, 5.0, 0.05),
            Product("raw_material_2", "Plastic", 80.0, 1.1, 0.1, 3.0, 0.03),
            Product("component_1", "Engine", 20.0, 0.9, 0.0, 150.0, 1.0),
            Product("component_2", "Electronics", 30.0, 1.2, 0.05, 100.0, 0.8),
            Product("finished_product", "Vehicle", 10.0, 1.0, 0.0, 2000.0, 15.0)
        ]
    
    def _create_default_topology(self, agent_configs: List[AgentConfig]) -> Dict[str, List[str]]:
        """Create default network topology"""
        topology = {}
        agent_ids = [config.agent_id for config in agent_configs]
        
        # Simple linear supply chain: supplier -> manufacturer -> distributor -> retailer
        for i, agent_id in enumerate(agent_ids):
            if i == 0:
                topology[agent_id] = agent_ids[1:2]  # Supplier connects to manufacturer
            elif i == len(agent_ids) - 1:
                topology[agent_id] = []  # Retailer connects to no one downstream
            else:
                topology[agent_id] = agent_ids[i+1:i+2]  # Connect to next in chain
        
        return topology
    
    def _create_supply_chain_agents(self, agent_configs: List[AgentConfig]) -> Dict[str, SupplyChainAgent]:
        """Create supply chain agents with appropriate node types"""
        agents = {}
        node_types = [NodeType.SUPPLIER, NodeType.MANUFACTURER, 
                     NodeType.DISTRIBUTOR, NodeType.RETAILER]
        
        for i, config in enumerate(agent_configs):
            node_type = node_types[i % len(node_types)]
            agent = SupplyChainAgent(config.agent_id, config, node_type)
            
            # Initialize inventory based on node type
            self._initialize_agent_inventory(agent)
            
            # Set up relationships
            agent.customers = self.network_topology.get(config.agent_id, [])
            for other_id, customers in self.network_topology.items():
                if config.agent_id in customers:
                    agent.suppliers.append(other_id)
            
            agents[config.agent_id] = agent
        
        return agents
    
    def _initialize_agent_inventory(self, agent: SupplyChainAgent):
        """Initialize inventory based on agent's role in supply chain"""
        if agent.node_type == NodeType.SUPPLIER:
            # Suppliers have raw materials
            agent.add_inventory("raw_material_1", 1000.0, 5000.0)
            agent.add_inventory("raw_material_2", 800.0, 4000.0)
        elif agent.node_type == NodeType.MANUFACTURER:
            # Manufacturers have components and some raw materials
            agent.add_inventory("raw_material_1", 200.0, 1000.0)
            agent.add_inventory("raw_material_2", 150.0, 800.0)
            agent.add_inventory("component_1", 50.0, 200.0)
            agent.add_inventory("component_2", 75.0, 300.0)
        elif agent.node_type == NodeType.DISTRIBUTOR:
            # Distributors have finished products
            agent.add_inventory("finished_product", 100.0, 500.0)
        elif agent.node_type == NodeType.RETAILER:
            # Retailers have small amounts of finished products
            agent.add_inventory("finished_product", 20.0, 100.0)
    
    def _get_observation_dimension(self) -> int:
        """Calculate observation dimension for supply chain"""
        # Global features + product features + agent features + network features
        global_features = 8  # demand_multiplier, cost_inflation, logistics_efficiency, etc.
        product_features = len(self.products) * 4  # demand, price, availability, trend
        agent_features = 12  # inventory levels, cash, metrics, etc.
        network_features = len(self.agent_configs) * 3  # relationship strengths
        
        return global_features + product_features + agent_features + network_features
    
    def _get_action_dimension(self) -> int:
        """Calculate action dimension for supply chain"""
        # Actions: [order_quantities (per product per supplier), pricing_decisions, inventory_targets]
        num_products = len(self.products)
        max_suppliers = 3  # Assume max 3 suppliers per agent
        
        order_actions = num_products * max_suppliers  # Order quantities
        pricing_actions = num_products  # Pricing decisions
        inventory_actions = num_products  # Target inventory levels
        
        return order_actions + pricing_actions + inventory_actions
    
    def _initialize_environment_state(self) -> EnvironmentState:
        """Initialize supply chain environment state"""
        # Generate initial demand
        self._update_market_demand()
        
        # Initialize agent states
        agent_states = {}
        for agent_id, sc_agent in self.supply_chain_agents.items():
            # Calculate total inventory value
            inventory_value = sum(
                inv.quantity * self.product_dict[product_id].unit_cost
                for product_id, inv in sc_agent.inventory.items()
                if product_id in self.product_dict
            )
            
            agent_states[agent_id] = {
                'balance': sc_agent.cash_balance,
                'inventory_value': inventory_value,
                'revenue': sc_agent.revenue,
                'costs': sc_agent.costs,
                'fill_rate': sc_agent.metrics['fill_rate'],
                'service_level': sc_agent.metrics['service_level'],
                'inventory_levels': {pid: inv.quantity for pid, inv in sc_agent.inventory.items()},
                'pending_orders': len(sc_agent.pending_orders),
                'utilization': min(1.0, inventory_value / (sum(sc_agent.capacity.values()) * 10.0)),
                'success_rate': sc_agent.metrics['fill_rate'],
                'risk_level': 1.0 - sc_agent.metrics['service_level'],
                'collaboration_score': len(sc_agent.suppliers + sc_agent.customers) / len(self.agent_configs),
                'connections': sc_agent.suppliers + sc_agent.customers
            }
        
        # Global state
        global_state = {
            'demand_multiplier': self.base_demand_multiplier,
            'cost_inflation': self.global_cost_inflation,
            'logistics_efficiency': self.logistics_efficiency,
            'active_disruptions': len(self.active_events),
            'total_orders': len(self.active_orders),
            'network_health': self._calculate_network_health()
        }
        
        # Market conditions
        market_conditions = {
            'demand_volatility': 0.2,
            'supply_stability': 0.8,
            'logistics_reliability': self.logistics_efficiency,
            'cost_pressure': self.global_cost_inflation - 1.0
        }
        
        return EnvironmentState(
            global_state=global_state,
            agent_states=agent_states,
            market_conditions=market_conditions,
            timestamp=datetime.now(),
            step_count=0
        )
    
    def _execute_actions(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> EnvironmentState:
        """Execute supply chain actions"""
        # Process agent actions
        for agent_id, action_vector in actions.items():
            if agent_id in self.supply_chain_agents:
                self._process_agent_actions(agent_id, action_vector)
        
        # Update market conditions
        self._update_market_demand()
        self._process_disruptions()
        
        # Process orders
        self._process_orders()
        
        # Update agent metrics
        for agent_id, sc_agent in self.supply_chain_agents.items():
            orders_received = len([o for o in self.active_orders if o.to_node == agent_id])
            orders_fulfilled = len([o for o in self.completed_orders 
                                  if o.to_node == agent_id and o.status == "delivered"])
            
            inventory_values = {pid: inv.quantity * self.product_dict[pid].unit_cost 
                              for pid, inv in sc_agent.inventory.items() 
                              if pid in self.product_dict}
            
            sc_agent.update_metrics(orders_received, orders_fulfilled, inventory_values, sc_agent.revenue)
        
        # Update agent states
        updated_agent_states = {}
        for agent_id, sc_agent in self.supply_chain_agents.items():
            inventory_value = sum(
                inv.quantity * self.product_dict[product_id].unit_cost
                for product_id, inv in sc_agent.inventory.items()
                if product_id in self.product_dict
            )
            
            updated_agent_states[agent_id] = {
                'balance': sc_agent.cash_balance,
                'inventory_value': inventory_value,
                'revenue': sc_agent.revenue,
                'costs': sc_agent.costs,
                'fill_rate': sc_agent.metrics['fill_rate'],
                'service_level': sc_agent.metrics['service_level'],
                'inventory_levels': {pid: inv.quantity for pid, inv in sc_agent.inventory.items()},
                'pending_orders': len(sc_agent.pending_orders),
                'utilization': min(1.0, inventory_value / (sum(sc_agent.capacity.values()) * 10.0)),
                'success_rate': sc_agent.metrics['fill_rate'],
                'risk_level': 1.0 - sc_agent.metrics['service_level'],
                'collaboration_score': self._calculate_collaboration_score(agent_id),
                'connections': sc_agent.suppliers + sc_agent.customers
            }
        
        # Update global state
        updated_global_state = {
            'demand_multiplier': self.base_demand_multiplier,
            'cost_inflation': self.global_cost_inflation,
            'logistics_efficiency': self.logistics_efficiency,
            'active_disruptions': len(self.active_events),
            'total_orders': len(self.active_orders),
            'network_health': self._calculate_network_health()
        }
        
        return EnvironmentState(
            global_state=updated_global_state,
            agent_states=updated_agent_states,
            market_conditions=state.market_conditions,
            timestamp=datetime.now(),
            step_count=state.step_count + 1
        )
    
    def _process_agent_actions(self, agent_id: str, action_vector: np.ndarray):
        """Process individual agent's supply chain actions"""
        sc_agent = self.supply_chain_agents[agent_id]
        
        num_products = len(self.products)
        max_suppliers = 3
        
        # Decode action vector
        order_actions = action_vector[:num_products * max_suppliers]
        pricing_actions = action_vector[num_products * max_suppliers:num_products * max_suppliers + num_products]
        inventory_actions = action_vector[num_products * max_suppliers + num_products:]
        
        # Process ordering actions
        for i, product in enumerate(self.products):
            for j, supplier_id in enumerate(sc_agent.suppliers[:max_suppliers]):
                action_idx = i * max_suppliers + j
                if action_idx < len(order_actions):
                    order_quantity = max(0, order_actions[action_idx] * 100)  # Scale to reasonable quantity
                    
                    if order_quantity > 10:  # Minimum order threshold
                        self._create_order(supplier_id, agent_id, product.id, order_quantity)
        
        # Process inventory management (simplified)
        for i, product in enumerate(self.products):
            if i < len(inventory_actions) and product.id in sc_agent.inventory:
                target_level = max(0, inventory_actions[i] * sc_agent.capacity.get(product.id, 100))
                current_level = sc_agent.inventory[product.id].quantity
                
                # Adjust production/ordering based on target (simplified logic)
                if target_level > current_level * 1.2:
                    # Need more inventory - increase ordering in next cycle
                    pass
    
    def _create_order(self, from_node: str, to_node: str, product_id: str, quantity: float):
        """Create a new order"""
        if from_node in self.supply_chain_agents and to_node in self.supply_chain_agents:
            self.order_counter += 1
            
            # Calculate price (simplified)
            base_cost = self.product_dict[product_id].unit_cost
            unit_price = base_cost * (1 + random.uniform(0.1, 0.3))  # Add margin
            
            order = Order(
                id=f"order_{self.order_counter:06d}",
                from_node=from_node,
                to_node=to_node,
                product_id=product_id,
                quantity=quantity,
                unit_price=unit_price,
                order_date=datetime.now(),
                expected_delivery=datetime.now() + timedelta(days=random.randint(1, 7))
            )
            
            self.active_orders.append(order)
            self.supply_chain_agents[to_node].pending_orders.append(order)
    
    def _process_orders(self):
        """Process active orders"""
        completed_orders = []
        
        for order in self.active_orders:
            supplier = self.supply_chain_agents[order.from_node]
            customer = self.supply_chain_agents[order.to_node]
            
            if order.status == "pending":
                # Check if supplier can fulfill
                available = supplier.get_available_inventory(order.product_id)
                if available >= order.quantity:
                    # Reserve inventory
                    if supplier.reserve_inventory(order.product_id, order.quantity):
                        order.status = "confirmed"
            
            elif order.status == "confirmed":
                # Simulate shipping time
                if datetime.now() >= order.expected_delivery:
                    # Fulfill order
                    if supplier.fulfill_order(order.product_id, order.quantity):
                        # Add inventory to customer
                        if order.product_id not in customer.inventory:
                            customer.add_inventory(order.product_id, 0.0)
                        
                        customer.inventory[order.product_id].quantity += order.quantity
                        
                        # Financial transaction
                        total_cost = order.quantity * order.unit_price
                        supplier.cash_balance += total_cost
                        supplier.revenue += total_cost
                        customer.cash_balance -= total_cost
                        customer.costs += total_cost
                        
                        order.status = "delivered"
                        completed_orders.append(order)
                        
                        # Remove from pending orders
                        if order in customer.pending_orders:
                            customer.pending_orders.remove(order)
        
        # Move completed orders
        for order in completed_orders:
            if order in self.active_orders:
                self.active_orders.remove(order)
                self.completed_orders.append(order)
    
    def _update_market_demand(self):
        """Update market demand based on seasonality and events"""
        # Seasonal effects
        if self.seasonality_enabled:
            day_of_year = datetime.now().timetuple().tm_yday
            seasonal_factor = 1.0 + 0.2 * np.sin(2 * np.pi * day_of_year / 365.0)
            self.base_demand_multiplier = seasonal_factor
        
        # Random demand fluctuations
        demand_shock = random.uniform(0.9, 1.1)
        self.base_demand_multiplier *= demand_shock
        
        # Event-based demand changes
        for event in self.active_events:
            if event.event_type == EventType.DEMAND_SPIKE:
                self.base_demand_multiplier *= (1.0 + event.impact_magnitude)
    
    def _process_disruptions(self):
        """Process supply chain disruptions"""
        # Remove expired events
        self.active_events = [event for event in self.active_events 
                             if self.step_count < event.start_step + event.duration_steps]
        
        # Generate new disruptions
        if random.random() < self.disruption_probability:
            event = self._generate_disruption_event()
            self.active_events.append(event)
            logger.info(f"New disruption: {event.event_type.value} affecting {len(event.affected_nodes)} nodes")
        
        # Apply disruption effects
        for event in self.active_events:
            self._apply_disruption_effects(event)
    
    def _generate_disruption_event(self) -> SupplyChainEvent:
        """Generate a random disruption event"""
        event_types = list(EventType)
        event_type = random.choice(event_types)
        
        # Select affected nodes and products
        affected_nodes = random.sample(list(self.supply_chain_agents.keys()), 
                                     random.randint(1, max(1, len(self.supply_chain_agents) // 2)))
        affected_products = random.sample([p.id for p in self.products], 
                                        random.randint(1, len(self.products)))
        
        return SupplyChainEvent(
            event_type=event_type,
            affected_nodes=affected_nodes,
            affected_products=affected_products,
            impact_magnitude=random.uniform(0.1, 0.5),
            duration_steps=random.randint(5, 20),
            start_step=self.step_count
        )
    
    def _apply_disruption_effects(self, event: SupplyChainEvent):
        """Apply effects of disruption event"""
        if event.event_type == EventType.SUPPLIER_SHORTAGE:
            # Reduce available inventory
            for node_id in event.affected_nodes:
                if node_id in self.supply_chain_agents:
                    agent = self.supply_chain_agents[node_id]
                    for product_id in event.affected_products:
                        if product_id in agent.inventory:
                            reduction = agent.inventory[product_id].quantity * event.impact_magnitude
                            agent.inventory[product_id].quantity = max(0, 
                                agent.inventory[product_id].quantity - reduction)
        
        elif event.event_type == EventType.LOGISTICS_DELAY:
            # Extend delivery times
            for order in self.active_orders:
                if (order.from_node in event.affected_nodes or 
                    order.to_node in event.affected_nodes):
                    delay_days = int(event.impact_magnitude * 10)
                    order.expected_delivery += timedelta(days=delay_days)
        
        elif event.event_type == EventType.QUALITY_ISSUE:
            # Reduce usable inventory
            for node_id in event.affected_nodes:
                if node_id in self.supply_chain_agents:
                    agent = self.supply_chain_agents[node_id]
                    for product_id in event.affected_products:
                        if product_id in agent.inventory:
                            loss = agent.inventory[product_id].quantity * event.impact_magnitude * 0.5
                            agent.inventory[product_id].quantity = max(0, 
                                agent.inventory[product_id].quantity - loss)
                            agent.costs += loss * self.product_dict[product_id].unit_cost
    
    def _calculate_network_health(self) -> float:
        """Calculate overall supply chain network health"""
        if not self.supply_chain_agents:
            return 0.0
        
        # Average service levels
        avg_service_level = np.mean([agent.metrics['service_level'] 
                                   for agent in self.supply_chain_agents.values()])
        
        # Order fulfillment rate
        total_orders = len(self.completed_orders) + len(self.active_orders)
        fulfilled_orders = len([o for o in self.completed_orders if o.status == "delivered"])
        fulfillment_rate = fulfilled_orders / total_orders if total_orders > 0 else 1.0
        
        # Disruption impact
        disruption_impact = max(0, 1.0 - len(self.active_events) * 0.1)
        
        return (avg_service_level * 0.4 + fulfillment_rate * 0.4 + disruption_impact * 0.2)
    
    def _calculate_collaboration_score(self, agent_id: str) -> float:
        """Calculate collaboration score for an agent"""
        sc_agent = self.supply_chain_agents[agent_id]
        
        # Base score from network connections
        max_connections = len(self.agent_configs) - 1
        connection_score = len(sc_agent.suppliers + sc_agent.customers) / max_connections if max_connections > 0 else 0
        
        # Collaboration quality (simplified)
        recent_orders = [o for o in self.completed_orders[-20:] 
                        if o.from_node == agent_id or o.to_node == agent_id]
        
        if recent_orders:
            on_time_deliveries = len([o for o in recent_orders 
                                    if o.status == "delivered" and o.from_node == agent_id])
            collaboration_quality = on_time_deliveries / len([o for o in recent_orders if o.from_node == agent_id])
        else:
            collaboration_quality = 0.5
        
        return (connection_score * 0.3 + collaboration_quality * 0.7)
    
    def _calculate_rewards(self, actions: Dict[str, np.ndarray], state: EnvironmentState) -> Dict[str, float]:
        """Calculate rewards for supply chain agents"""
        rewards = {}
        
        for agent_id, sc_agent in self.supply_chain_agents.items():
            # Profit-based reward
            profit = sc_agent.revenue - sc_agent.costs
            profit_reward = profit / 1000.0  # Scale reward
            
            # Service level reward
            service_reward = (sc_agent.metrics['service_level'] - 0.5) * 2.0
            
            # Inventory efficiency reward
            total_capacity = sum(sc_agent.capacity.values())
            total_inventory = sum(inv.quantity for inv in sc_agent.inventory.values())
            utilization = total_inventory / total_capacity if total_capacity > 0 else 0
            
            # Optimal utilization around 70-80%
            if 0.7 <= utilization <= 0.8:
                efficiency_reward = 1.0
            elif utilization < 0.7:
                efficiency_reward = utilization / 0.7
            else:
                efficiency_reward = max(0, 1.0 - (utilization - 0.8) * 2)
            
            # Collaboration reward
            collaboration_reward = 0.0
            if self.reward_shaping:
                collab_score = state.agent_states[agent_id]['collaboration_score']
                collaboration_reward = (collab_score - 0.5) * 0.5
            
            # Sustainability bonus (simplified)
            sustainability_bonus = 0.0
            if len(self.active_events) == 0:  # No disruptions
                sustainability_bonus = 0.1
            
            # Penalty for stockouts
            stockout_penalty = 0.0
            for product_id, inv in sc_agent.inventory.items():
                if inv.quantity == 0 and product_id in self.product_dict:
                    stockout_penalty += 0.1
            
            # Combine rewards
            total_reward = (profit_reward * 0.4 + 
                          service_reward * 0.3 + 
                          efficiency_reward * 0.2 + 
                          collaboration_reward * 0.1 + 
                          sustainability_bonus - 
                          stockout_penalty)
            
            rewards[agent_id] = total_reward
        
        return rewards
    
    def get_supply_chain_metrics(self) -> Dict[str, Any]:
        """Get comprehensive supply chain metrics"""
        return {
            'network_health': self._calculate_network_health(),
            'total_orders': len(self.active_orders) + len(self.completed_orders),
            'active_orders': len(self.active_orders),
            'completed_orders': len(self.completed_orders),
            'fulfillment_rate': len([o for o in self.completed_orders if o.status == "delivered"]) / 
                               len(self.completed_orders) if self.completed_orders else 0,
            'active_disruptions': len(self.active_events),
            'demand_multiplier': self.base_demand_multiplier,
            'cost_inflation': self.global_cost_inflation,
            'logistics_efficiency': self.logistics_efficiency,
            'agent_metrics': {
                agent_id: {
                    'balance': agent.cash_balance,
                    'revenue': agent.revenue,
                    'costs': agent.costs,
                    'inventory_value': sum(inv.quantity * self.product_dict.get(pid, Product("", "", 0)).unit_cost 
                                         for pid, inv in agent.inventory.items()),
                    'fill_rate': agent.metrics['fill_rate'],
                    'service_level': agent.metrics['service_level'],
                    'pending_orders': len(agent.pending_orders)
                }
                for agent_id, agent in self.supply_chain_agents.items()
            }
        }
    
    def render(self, mode: str = "human"):
        """Render the supply chain environment"""
        if mode == "human":
            self._render_supply_chain_text()
    
    def _render_supply_chain_text(self):
        """Text rendering of supply chain state"""
        print(f"\n--- Supply Chain Environment Step {self.step_count} ---")
        print(f"Network Health: {self._calculate_network_health():.2%}")
        print(f"Active Orders: {len(self.active_orders)}")
        print(f"Active Disruptions: {len(self.active_events)}")
        print(f"Demand Multiplier: {self.base_demand_multiplier:.2f}")
        
        print("\nAgent Status:")
        for agent_id, sc_agent in self.supply_chain_agents.items():
            inventory_value = sum(inv.quantity * self.product_dict.get(pid, Product("", "", 0)).unit_cost 
                                for pid, inv in sc_agent.inventory.items())
            
            print(f"  {agent_id} ({sc_agent.node_type.value}):")
            print(f"    Cash: ${sc_agent.cash_balance:,.2f}")
            print(f"    Inventory Value: ${inventory_value:,.2f}")
            print(f"    Service Level: {sc_agent.metrics['service_level']:.2%}")
            print(f"    Pending Orders: {len(sc_agent.pending_orders)}")
        
        if self.active_events:
            print("\nActive Disruptions:")
            for event in self.active_events:
                remaining_steps = event.start_step + event.duration_steps - self.step_count
                print(f"  {event.event_type.value}: {remaining_steps} steps remaining")
                print(f"    Affected: {', '.join(event.affected_nodes[:3])}")
                print(f"    Impact: {event.impact_magnitude:.1%}")