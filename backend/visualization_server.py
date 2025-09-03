"""
Advanced Visualization Backend Server
Real-time WebSocket streaming for AI agent network visualization
"""

import asyncio
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import numpy as np
from pydantic import BaseModel
import math
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class AgentPerformance(BaseModel):
    throughput: float
    accuracy: float
    responseTime: float
    utilization: float

class AgentPosition(BaseModel):
    x: float
    y: float
    z: float

class Agent(BaseModel):
    id: str
    name: str
    type: str  # 'coordinating', 'processing', 'analysis', 'integration'
    status: str  # 'active', 'idle', 'error', 'offline'
    position: AgentPosition
    performance: AgentPerformance
    connections: List[str]
    lastActivity: datetime
    metadata: Dict[str, Any]

class Connection(BaseModel):
    id: str
    sourceId: str
    targetId: str
    type: str  # 'data_flow', 'coordination', 'feedback', 'error'
    strength: float
    latency: float
    bandwidth: float
    messages: int
    direction: str  # 'bidirectional', 'source_to_target', 'target_to_source'

class NetworkMetrics(BaseModel):
    totalAgents: int
    activeAgents: int
    totalConnections: int
    averageLatency: float
    networkThroughput: float
    overallAccuracy: float
    systemHealth: float
    loadDistribution: List[float]

class AgentNetwork(BaseModel):
    agents: List[Agent]
    connections: List[Connection]
    metrics: NetworkMetrics
    timestamp: datetime

class PerformanceMetric(BaseModel):
    timestamp: datetime
    agentId: str
    metric: str
    value: float
    unit: str
    category: str  # 'performance', 'accuracy', 'efficiency', 'health'

class TradingEvent(BaseModel):
    id: str
    timestamp: datetime
    type: str  # 'buy', 'sell', 'analysis', 'decision', 'execution'
    agentId: str
    symbol: Optional[str] = None
    quantity: Optional[int] = None
    price: Optional[float] = None
    confidence: float
    reasoning: str
    outcome: Optional[str] = None  # 'success', 'failure', 'pending'
    profit: Optional[float] = None

class MarketData(BaseModel):
    symbol: str
    price: float
    change: float
    volume: int
    timestamp: datetime
    indicators: Dict[str, Any]

# FastAPI app
app = FastAPI(
    title="AI Agents Visualization Server",
    description="Real-time WebSocket streaming for advanced agent visualization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_count = 0

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_count += 1
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: str, websocket: WebSocket):
        try:
            await websocket.send_text(message)
        except Exception as e:
            logger.error(f"Error sending message to client: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            self.disconnect(conn)

manager = ConnectionManager()

# Data simulation classes
class AgentSimulator:
    def __init__(self):
        self.agents = self._create_initial_agents()
        self.connections = self._create_initial_connections()
        self.metrics_history: List[PerformanceMetric] = []
        self.trading_events: List[TradingEvent] = []
        self.market_symbols = ['AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'BRK.A']
        self.start_time = datetime.now()

    def _create_initial_agents(self) -> List[Agent]:
        """Create initial set of AI agents with realistic configurations"""
        agent_types = ['coordinating', 'processing', 'analysis', 'integration']
        agents = []
        
        for i in range(12):  # Create 12 agents for a good visualization
            agent_type = agent_types[i % len(agent_types)]
            
            # Position agents in a 3D grid pattern
            grid_size = 3
            x = (i % grid_size - 1) * 8
            y = random.uniform(0, 5)
            z = (i // grid_size - 1) * 8
            
            agent = Agent(
                id=f"agent-{i+1:02d}",
                name=f"Agent-{agent_type.title()}-{i+1:02d}",
                type=agent_type,
                status=random.choices(['active', 'idle', 'error'], weights=[70, 25, 5])[0],
                position=AgentPosition(x=x, y=y, z=z),
                performance=AgentPerformance(
                    throughput=random.uniform(50, 150),
                    accuracy=random.uniform(85, 99),
                    responseTime=random.uniform(100, 500),
                    utilization=random.uniform(30, 95)
                ),
                connections=[],
                lastActivity=datetime.now(),
                metadata={
                    'specialization': f'{agent_type}_specialist',
                    'version': f'1.{random.randint(0, 9)}.{random.randint(0, 9)}',
                    'capabilities': [
                        'real_time_processing',
                        'pattern_recognition',
                        'decision_making',
                        'coordination'
                    ][:random.randint(2, 4)]
                }
            )
            agents.append(agent)
        
        return agents

    def _create_initial_connections(self) -> List[Connection]:
        """Create realistic connection patterns between agents"""
        connections = []
        connection_types = ['data_flow', 'coordination', 'feedback', 'error']
        
        for i, source_agent in enumerate(self.agents):
            # Each agent connects to 2-4 other agents
            num_connections = random.randint(2, 4)
            potential_targets = [a for a in self.agents if a.id != source_agent.id]
            
            targets = random.sample(potential_targets, min(num_connections, len(potential_targets)))
            
            for target_agent in targets:
                connection_id = f"{source_agent.id}-{target_agent.id}"
                
                # Avoid duplicate connections
                if not any(c.id == connection_id for c in connections):
                    connection = Connection(
                        id=connection_id,
                        sourceId=source_agent.id,
                        targetId=target_agent.id,
                        type=random.choice(connection_types),
                        strength=random.uniform(0.3, 1.0),
                        latency=random.uniform(10, 100),
                        bandwidth=random.uniform(100, 1000),
                        messages=random.randint(10, 1000),
                        direction=random.choice(['bidirectional', 'source_to_target'])
                    )
                    connections.append(connection)
                    
                    # Update agent connections list
                    source_agent.connections.append(target_agent.id)
        
        return connections

    def update_agents_realtime(self):
        """Simulate real-time agent updates"""
        current_time = datetime.now()
        
        for agent in self.agents:
            # Simulate performance fluctuations
            if random.random() < 0.3:  # 30% chance of performance update
                # Throughput varies based on utilization
                utilization_factor = agent.performance.utilization / 100
                agent.performance.throughput += random.uniform(-10, 10) * utilization_factor
                agent.performance.throughput = max(10, min(200, agent.performance.throughput))
                
                # Accuracy slowly degrades under high utilization
                if agent.performance.utilization > 80:
                    agent.performance.accuracy += random.uniform(-0.5, 0.1)
                else:
                    agent.performance.accuracy += random.uniform(-0.1, 0.3)
                agent.performance.accuracy = max(70, min(99.9, agent.performance.accuracy))
                
                # Response time varies inversely with utilization
                agent.performance.responseTime += random.uniform(-20, 20)
                agent.performance.responseTime = max(50, min(1000, agent.performance.responseTime))
                
                # Utilization changes gradually
                agent.performance.utilization += random.uniform(-5, 5)
                agent.performance.utilization = max(10, min(100, agent.performance.utilization))
            
            # Status changes
            if random.random() < 0.05:  # 5% chance of status change
                if agent.status == 'active' and agent.performance.utilization < 20:
                    agent.status = 'idle'
                elif agent.status == 'idle' and random.random() < 0.7:
                    agent.status = 'active'
                elif agent.performance.accuracy < 80:
                    agent.status = 'error'
                elif agent.status == 'error' and random.random() < 0.3:
                    agent.status = 'active'
                    # Recovery boost
                    agent.performance.accuracy = random.uniform(85, 95)
            
            agent.lastActivity = current_time
            
            # Generate performance metrics
            if random.random() < 0.4:  # 40% chance of metric generation
                metric_types = ['throughput', 'accuracy', 'latency', 'utilization']
                metric_type = random.choice(metric_types)
                
                value = getattr(agent.performance, metric_type if metric_type != 'latency' else 'responseTime')
                
                metric = PerformanceMetric(
                    timestamp=current_time,
                    agentId=agent.id,
                    metric=metric_type,
                    value=value,
                    unit='percentage' if metric_type in ['accuracy', 'utilization'] else 'numeric',
                    category='performance'
                )
                
                self.metrics_history.append(metric)
                
                # Keep only recent metrics (last 500)
                if len(self.metrics_history) > 500:
                    self.metrics_history = self.metrics_history[-500:]

    def update_connections_realtime(self):
        """Simulate real-time connection updates"""
        for connection in self.connections:
            # Update connection metrics
            if random.random() < 0.2:  # 20% chance of connection update
                connection.strength += random.uniform(-0.1, 0.1)
                connection.strength = max(0.1, min(1.0, connection.strength))
                
                connection.latency += random.uniform(-5, 5)
                connection.latency = max(5, min(200, connection.latency))
                
                connection.bandwidth += random.uniform(-50, 50)
                connection.bandwidth = max(50, min(2000, connection.bandwidth))
                
                connection.messages += random.randint(-10, 50)
                connection.messages = max(0, connection.messages)

    def generate_trading_event(self) -> Optional[TradingEvent]:
        """Generate realistic trading events"""
        if random.random() < 0.15:  # 15% chance of trading event per update
            active_agents = [a for a in self.agents if a.status == 'active']
            if not active_agents:
                return None
                
            agent = random.choice(active_agents)
            event_types = ['buy', 'sell', 'analysis', 'decision', 'execution']
            event_type = random.choice(event_types)
            
            event = TradingEvent(
                id=f"event-{int(time.time() * 1000)}-{random.randint(1000, 9999)}",
                timestamp=datetime.now(),
                type=event_type,
                agentId=agent.id,
                symbol=random.choice(self.market_symbols),
                quantity=random.randint(10, 1000) if event_type in ['buy', 'sell'] else None,
                price=random.uniform(50, 500) if event_type in ['buy', 'sell'] else None,
                confidence=random.uniform(0.6, 0.95),
                reasoning=f"AI analysis indicates {random.choice(['bullish', 'bearish', 'neutral'])} sentiment",
                outcome=random.choices(['success', 'failure', 'pending'], weights=[60, 10, 30])[0],
                profit=random.uniform(-5000, 15000) if event_type in ['buy', 'sell'] else None
            )
            
            self.trading_events.append(event)
            
            # Keep only recent events (last 100)
            if len(self.trading_events) > 100:
                self.trading_events = self.trading_events[-100:]
            
            return event
        
        return None

    def generate_market_data(self) -> List[MarketData]:
        """Generate realistic market data"""
        market_data = []
        
        for symbol in self.market_symbols:
            # Simulate price movement
            base_price = random.uniform(100, 500)
            change_percent = random.uniform(-5, 5)
            
            market_data.append(MarketData(
                symbol=symbol,
                price=base_price,
                change=change_percent,
                volume=random.randint(100000, 10000000),
                timestamp=datetime.now(),
                indicators={
                    'rsi': random.uniform(20, 80),
                    'macd': random.uniform(-2, 2),
                    'bollinger': {
                        'upper': base_price * 1.05,
                        'lower': base_price * 0.95,
                        'middle': base_price
                    },
                    'sentiment': random.uniform(-1, 1)
                }
            ))
        
        return market_data

    def get_network_state(self) -> AgentNetwork:
        """Get current network state"""
        active_agents = [a for a in self.agents if a.status == 'active']
        
        metrics = NetworkMetrics(
            totalAgents=len(self.agents),
            activeAgents=len(active_agents),
            totalConnections=len(self.connections),
            averageLatency=sum(c.latency for c in self.connections) / len(self.connections) if self.connections else 0,
            networkThroughput=sum(a.performance.throughput for a in active_agents),
            overallAccuracy=sum(a.performance.accuracy for a in self.agents) / len(self.agents) if self.agents else 0,
            systemHealth=len(active_agents) / len(self.agents) if self.agents else 0,
            loadDistribution=[a.performance.utilization for a in self.agents]
        )
        
        return AgentNetwork(
            agents=self.agents,
            connections=self.connections,
            metrics=metrics,
            timestamp=datetime.now()
        )

# Global simulator instance
simulator = AgentSimulator()

# API Routes
@app.get("/")
async def root():
    return {"message": "AI Agents Visualization Server", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "connections": len(manager.active_connections),
        "agents": len(simulator.agents)
    }

@app.get("/api/network", response_model=AgentNetwork)
async def get_network():
    """Get current agent network state"""
    return simulator.get_network_state()

@app.get("/api/metrics")
async def get_metrics():
    """Get recent performance metrics"""
    return simulator.metrics_history[-100:]  # Return last 100 metrics

@app.get("/api/events")
async def get_trading_events():
    """Get recent trading events"""
    return simulator.trading_events

@app.get("/api/market")
async def get_market_data():
    """Get current market data"""
    return simulator.generate_market_data()

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    try:
        # Send initial data
        initial_data = {
            "type": "initial_state",
            "network": simulator.get_network_state().dict(),
            "metrics": [m.dict() for m in simulator.metrics_history[-50:]],
            "events": [e.dict() for e in simulator.trading_events[-20:]],
            "market": [m.dict() for m in simulator.generate_market_data()]
        }
        await websocket.send_text(json.dumps(initial_data, default=str))
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for client messages (heartbeat, commands, etc.)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                message = json.loads(data)
                
                # Handle client commands
                if message.get("type") == "heartbeat":
                    await websocket.send_text(json.dumps({
                        "type": "heartbeat_response",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message.get("type") == "request_update":
                    # Send current state
                    update_data = {
                        "type": "network_update",
                        "network": simulator.get_network_state().dict()
                    }
                    await websocket.send_text(json.dumps(update_data, default=str))
                    
            except asyncio.TimeoutError:
                # No message received, continue the loop
                continue
            except json.JSONDecodeError:
                logger.warning("Received invalid JSON from client")
                continue
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# Background task for real-time updates
async def broadcast_updates():
    """Background task to broadcast real-time updates to all connected clients"""
    while True:
        try:
            if manager.active_connections:
                # Update simulation state
                simulator.update_agents_realtime()
                simulator.update_connections_realtime()
                
                # Generate trading event
                trading_event = simulator.generate_trading_event()
                
                # Broadcast network updates
                network_update = {
                    "type": "agent_network_update",
                    "data": simulator.get_network_state().dict()
                }
                await manager.broadcast(json.dumps(network_update, default=str))
                
                # Broadcast performance metrics
                recent_metrics = simulator.metrics_history[-10:] if len(simulator.metrics_history) >= 10 else simulator.metrics_history
                if recent_metrics:
                    metrics_update = {
                        "type": "performance_metrics",
                        "data": [m.dict() for m in recent_metrics]
                    }
                    await manager.broadcast(json.dumps(metrics_update, default=str))
                
                # Broadcast trading event
                if trading_event:
                    event_update = {
                        "type": "trading_event",
                        "data": trading_event.dict()
                    }
                    await manager.broadcast(json.dumps(event_update, default=str))
                
                # Broadcast market data every 5 seconds
                if int(time.time()) % 5 == 0:
                    market_update = {
                        "type": "market_data",
                        "data": [m.dict() for m in simulator.generate_market_data()]
                    }
                    await manager.broadcast(json.dumps(market_update, default=str))
            
            await asyncio.sleep(0.5)  # Update every 500ms
            
        except Exception as e:
            logger.error(f"Error in broadcast loop: {e}")
            await asyncio.sleep(1)  # Wait longer on error

@app.on_event("startup")
async def startup_event():
    """Start background tasks on server startup"""
    logger.info("Starting AI Agents Visualization Server...")
    asyncio.create_task(broadcast_updates())

if __name__ == "__main__":
    uvicorn.run(
        "visualization_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )