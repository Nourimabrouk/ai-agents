"""
Real-Time Coordination Dashboard Server
FastAPI + WebSocket server for agent visualization
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timedelta
import logging
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    WebSocket = None  # Fallback for type annotations
    logging.warning("FastAPI not available - dashboard will use fallback mode")

from core.orchestration.orchestrator import AgentOrchestrator
from templates.base_agent import BaseAgent
# from .websocket_handler import WebSocketHandler  # Not implemented yet
# from .metrics_collector import MetricsCollector  # Not implemented yet
# from .visualization_engine import VisualizationEngine  # Not implemented yet
# from .interaction_tracker import InteractionTracker  # Not implemented yet
from utils.observability.logging import get_logger

logger = get_logger(__name__)


class CoordinationDashboard:
    """
    Real-time dashboard for agent coordination visualization
    Provides WebSocket-based updates and interactive monitoring
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 8080,
                 dashboard_title: str = "AI Agent Coordination Dashboard"):
        
        self.host = host
        self.port = port
        self.dashboard_title = dashboard_title
        
        # Core components
        self.orchestrator: Optional[AgentOrchestrator] = None
        # self.websocket_handler = WebSocketHandler()  # Not implemented yet
        # self.metrics_collector = MetricsCollector()  # Not implemented yet
        # self.visualization_engine = VisualizationEngine()  # Not implemented yet
        # self.interaction_tracker = InteractionTracker()  # Not implemented yet
        
        # Dashboard state
        self.active_connections: Set[WebSocket] = set()
        self.update_interval = 1.0  # seconds
        self.is_running = False
        
        # Create FastAPI app if available
        if FASTAPI_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            self.app = None
            logger.warning("FastAPI not available - dashboard will run in limited mode")
        
        logger.info(f"Initialized coordination dashboard on {host}:{port}")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application with routes"""
        app = FastAPI(
            title=self.dashboard_title,
            description="Real-time AI Agent Coordination Dashboard",
            version="1.0.0"
        )
        
        # Static files
        dashboard_path = Path(__file__).parent
        static_path = dashboard_path / "static"
        if static_path.exists():
            app.mount(str(Path("/static").resolve()), StaticFiles(directory=str(static_path)), name="static")
        
        # WebSocket endpoint
        @app.websocket(str(Path("/ws").resolve()))
        async def websocket_endpoint(websocket: WebSocket):
            await self.websocket_handler.connect(websocket)
            self.active_connections.add(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle client messages
                    await self._handle_client_message(websocket, message)
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
                await self.websocket_handler.disconnect(websocket)
        
        # Main dashboard page
        @app.get(str(Path("/").resolve()), response_class=HTMLResponse)
        async def dashboard_home():
            return self._generate_dashboard_html()
        
        # API endpoints
        @app.get(str(Path("/api/agents").resolve()))
        async def get_agents():
            """Get current agents status"""
            if not self.orchestrator:
                return {"agents": {}}
            
            agents_data = {}
            for name, agent in self.orchestrator.agents.items():
                agents_data[name] = {
                    "name": name,
                    "state": agent.state.value if hasattr(agent.state, 'value') else str(agent.state),
                    "metrics": agent.get_metrics(),
                    "last_update": datetime.now().isoformat()
                }
            
            return {"agents": agents_data}
        
        @app.get(str(Path("/api/metrics").resolve()))
        async def get_metrics():
            """Get system metrics"""
            return await self.metrics_collector.get_current_metrics()
        
        @app.get(str(Path("/api/interactions").resolve()))
        async def get_interactions():
            """Get recent agent interactions"""
            return {
                "interactions": await self.interaction_tracker.get_recent_interactions(),
                "network_graph": await self.visualization_engine.generate_network_graph()
            }
        
        @app.post(str(Path("/api/agents/{agent_name}/task").resolve()))
        async def assign_task(agent_name: str, task_data: Dict[str, Any]):
            """Assign a task to a specific agent"""
            if not self.orchestrator or agent_name not in self.orchestrator.agents:
                return {"error": "Agent not found"}
            
            try:
                agent = self.orchestrator.agents[agent_name]
                result = await agent.process_task(
                    task_data.get("description", "Manual task"),
                    task_data.get("context", {})
                )
                
                return {
                    "success": True,
                    "result": str(result)[:500],  # Limit result size
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                return {"error": str(e)}
        
        return app
    
    def attach_orchestrator(self, orchestrator: AgentOrchestrator) -> None:
        """Attach an orchestrator for monitoring"""
        self.orchestrator = orchestrator
        
        # Start collecting metrics if available
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            self.metrics_collector.start_monitoring(orchestrator)
        if hasattr(self, 'interaction_tracker') and self.interaction_tracker:
            self.interaction_tracker.start_tracking(orchestrator)
        
        logger.info(f"Attached orchestrator with {len(orchestrator.agents)} agents")
    
    async def start_server(self) -> None:
        """Start the dashboard server"""
        if not FASTAPI_AVAILABLE:
            logger.error("Cannot start dashboard server - FastAPI not available")
            return {}
        
        self.is_running = True
        
        # Start background tasks
        asyncio.create_task(self._update_clients_loop())
        asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Dashboard server starting on {self.host}:{self.port}")
        
        # Import uvicorn here to avoid dependency issues
        try:
            import uvicorn
            
            config = uvicorn.Config(
                self.app,
                host=self.host,
                port=self.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except ImportError:
            logger.error("uvicorn not available - cannot start server")
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
        
        self.is_running = False
    
    async def _update_clients_loop(self) -> None:
        """Background loop to update connected clients"""
        while self.is_running:
            if self.active_connections and self.orchestrator:
                try:
                    # Collect current data
                    dashboard_data = await self._collect_dashboard_data()
                    
                    # Send to all connected clients
                    await self.websocket_handler.broadcast(dashboard_data)
                    
                except Exception as e:
                    logger.error(f"Error updating clients: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup tasks"""
        while self.is_running:
            try:
                # Clean up old metrics and interactions
                await self.metrics_collector.cleanup_old_data()
                await self.interaction_tracker.cleanup_old_data()
                
                # Clean up disconnected WebSocket connections
                disconnected = []
                for connection in self.active_connections:
                    try:
                        await connection.ping()
                    except Exception:
                        disconnected.append(connection)
                
                for conn in disconnected:
                    self.active_connections.discard(conn)
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
            
            await asyncio.sleep(60)  # Run cleanup every minute
    
    async def _collect_dashboard_data(self) -> Dict[str, Any]:
        """Collect all data needed for dashboard update"""
        data = {
            "timestamp": datetime.now().isoformat(),
            "agents": {},
            "metrics": {},
            "interactions": [],
            "network_graph": {},
            "system_status": "active"
        }
        
        if not self.orchestrator:
            data["system_status"] = "no_orchestrator"
            return data
        
        try:
            # Agent data
            for name, agent in self.orchestrator.agents.items():
                data["agents"][name] = {
                    "name": name,
                    "state": agent.state.value if hasattr(agent.state, 'value') else str(agent.state),
                    "metrics": agent.get_metrics(),
                    "position": await self._calculate_agent_position(name),
                    "connections": await self._get_agent_connections(name)
                }
            
            # System metrics
            data["metrics"] = await self.metrics_collector.get_current_metrics()
            
            # Recent interactions
            data["interactions"] = await self.interaction_tracker.get_recent_interactions(limit=50)
            
            # Network visualization
            data["network_graph"] = await self.visualization_engine.generate_network_graph()
            
            # Orchestrator metrics
            data["orchestrator_metrics"] = self.orchestrator.get_metrics()
            
        except Exception as e:
            logger.error(f"Error collecting dashboard data: {e}")
            data["system_status"] = f"error: {str(e)}"
        
        return data
    
    async def _calculate_agent_position(self, agent_name: str) -> Dict[str, float]:
        """Calculate position for agent in network visualization"""
        # Simple positioning algorithm - could be enhanced with force-directed layout
        agents_list = list(self.orchestrator.agents.keys())
        
        if agent_name not in agents_list:
            return {"x": 0.5, "y": 0.5}
        
        index = agents_list.index(agent_name)
        total_agents = len(agents_list)
        
        # Arrange in circle
        import math
        angle = (2 * math.pi * index) / total_agents
        radius = 0.3
        
        x = 0.5 + radius * math.cos(angle)
        y = 0.5 + radius * math.sin(angle)
        
        return {"x": x, "y": y}
    
    async def _get_agent_connections(self, agent_name: str) -> List[str]:
        """Get list of agents this agent has interacted with recently"""
        interactions = await self.interaction_tracker.get_recent_interactions(limit=100)
        
        connections = set()
        for interaction in interactions:
            if interaction.get("from_agent") == agent_name:
                connections.add(interaction.get("to_agent"))
            elif interaction.get("to_agent") == agent_name:
                connections.add(interaction.get("from_agent"))
        
        return list(connections)
    
    async def _handle_client_message(self, websocket: WebSocket, message: Dict[str, Any]) -> None:
        """Handle messages from dashboard clients"""
        message_type = message.get("type")
        
        if message_type == "ping":
            await websocket.send_json({"type": "pong", "timestamp": datetime.now().isoformat()})
        
        elif message_type == "get_agent_details":
            agent_name = message.get("agent_name")
            if self.orchestrator and agent_name in self.orchestrator.agents:
                agent = self.orchestrator.agents[agent_name]
                details = {
                    "type": "agent_details",
                    "agent_name": agent_name,
                    "details": {
                        "metrics": agent.get_metrics(),
                        "memory_size": len(agent.memory.episodic_memory),
                        "learning_strategies": len(agent.learning_system.strategies),
                        "sub_agents": list(agent.sub_agents.keys()),
                        "tools_available": len(agent.tools)
                    }
                }
                await websocket.send_json(details)
        
        elif message_type == "pause_updates":
            # Client wants to pause updates temporarily
            # This could be implemented with per-client settings
        logger.info(f'Method {function_name} called')
        return {}
        
        elif message_type == "resume_updates":
            # Client wants to resume updates
            pass
    
    def _generate_dashboard_html(self) -> str:
        """Generate the main dashboard HTML"""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{self.dashboard_title}</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 0;
                    background: #1a1a1a;
                    color: #ffffff;
                }}
                
                .header {{
                    background: #2d3748;
                    padding: 1rem;
                    border-bottom: 1px solid #4a5568;
                }}
                
                .header h1 {{
                    margin: 0;
                    color: #63b3ed;
                }}
                
                .dashboard-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    grid-template-rows: 300px 300px 1fr;
                    gap: 1rem;
                    padding: 1rem;
                    height: calc(100vh - 80px);
                }}
                
                .panel {{
                    background: #2d3748;
                    border: 1px solid #4a5568;
                    border-radius: 8px;
                    padding: 1rem;
                    overflow: auto;
                }}
                
                .network-visualization {{
                    grid-column: 1 / 3;
                    position: relative;
                }}
                
                .metrics-panel {{
                    grid-row: 2;
                }}
                
                .interactions-panel {{
                    grid-row: 2;
                }}
                
                .agents-list {{
                    grid-column: 1 / 3;
                }}
                
                .agent-node {{
                    position: absolute;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: #63b3ed;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: bold;
                    font-size: 12px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }}
                
                .agent-node:hover {{
                    background: #90cdf4;
                    transform: scale(1.1);
                }}
                
                .agent-node.active {{
                    background: #68d391;
                }}
                
                .agent-node.error {{
                    background: #fc8181;
                }}
                
                .connection-line {{
                    position: absolute;
                    height: 2px;
                    background: #4a5568;
                    transform-origin: left center;
                    pointer-events: none;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 0.5rem;
                }}
                
                .metric-card {{
                    background: #1a202c;
                    padding: 0.75rem;
                    border-radius: 4px;
                    text-align: center;
                }}
                
                .metric-value {{
                    font-size: 1.5rem;
                    font-weight: bold;
                    color: #63b3ed;
                }}
                
                .metric-label {{
                    font-size: 0.875rem;
                    color: #a0aec0;
                }}
                
                .interaction-item {{
                    background: #1a202c;
                    margin: 0.5rem 0;
                    padding: 0.75rem;
                    border-radius: 4px;
                    font-size: 0.875rem;
                }}
                
                .status-indicator {{
                    display: inline-block;
                    width: 8px;
                    height: 8px;
                    border-radius: 50%;
                    margin-right: 0.5rem;
                }}
                
                .status-active {{ background: #68d391; }}
                .status-idle {{ background: #a0aec0; }}
                .status-error {{ background: #fc8181; }}
                
                .connection-status {{
                    position: fixed;
                    top: 1rem;
                    right: 1rem;
                    padding: 0.5rem 1rem;
                    border-radius: 4px;
                    font-size: 0.875rem;
                }}
                
                .connection-status.connected {{
                    background: #38a169;
                    color: white;
                }}
                
                .connection-status.disconnected {{
                    background: #e53e3e;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{self.dashboard_title}</h1>
            </div>
            
            <div class="connection-status" id="connectionStatus">Connecting...</div>
            
            <div class="dashboard-grid">
                <div class="panel network-visualization">
                    <h3>Agent Network</h3>
                    <div id="networkViz" style="position: relative; height: calc(100% - 2rem);"></div>
                </div>
                
                <div class="panel metrics-panel">
                    <h3>System Metrics</h3>
                    <div id="metricsContainer" class="metrics-grid"></div>
                </div>
                
                <div class="panel interactions-panel">
                    <h3>Recent Interactions</h3>
                    <div id="interactionsContainer"></div>
                </div>
                
                <div class="panel agents-list">
                    <h3>Active Agents</h3>
                    <div id="agentsContainer"></div>
                </div>
            </div>
            
            <script>
                class DashboardClient {{
                    constructor() {{
                        this.ws = null;
                        this.reconnectAttempts = 0;
                        this.maxReconnectAttempts = 5;
                        this.data = {{}};
                        this.connect();
                    }}
                    
                    connect() {{
                        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                        const wsUrl = `${{protocol}}//${{window.location.host}}/ws`;
                        
                        this.ws = new WebSocket(wsUrl);
                        
                        this.ws.onopen = () => {{
                            console.log('Connected to dashboard');
                            this.reconnectAttempts = 0;
                            this.updateConnectionStatus(true);
                        }};
                        
                        this.ws.onmessage = (event) => {{
                            const data = JSON.parse(event.data);
                            this.handleMessage(data);
                        }};
                        
                        this.ws.onclose = () => {{
                            console.log('Disconnected from dashboard');
                            this.updateConnectionStatus(false);
                            this.attemptReconnect();
                        }};
                        
                        this.ws.onerror = (error) => {{
                            console.error('WebSocket error:', error);
                        }};
                    }}
                    
                    handleMessage(data) {{
                        this.data = data;
                        this.updateDashboard();
                    }}
                    
                    updateConnectionStatus(connected) {{
                        const statusEl = document.getElementById('connectionStatus');
                        if (connected) {{
                            statusEl.textContent = 'Connected';
                            statusEl.className = 'connection-status connected';
                        }} else {{
                            statusEl.textContent = 'Disconnected';
                            statusEl.className = 'connection-status disconnected';
                        }}
                    }}
                    
                    attemptReconnect() {{
                        if (this.reconnectAttempts < this.maxReconnectAttempts) {{
                            this.reconnectAttempts++;
                            setTimeout(() => this.connect(), 2000 * this.reconnectAttempts);
                        }}
                    }}
                    
                    updateDashboard() {{
                        this.updateNetworkVisualization();
                        this.updateMetrics();
                        this.updateInteractions();
                        this.updateAgentsList();
                    }}
                    
                    updateNetworkVisualization() {{
                        const container = document.getElementById('networkViz');
                        container.innerHTML = '';
                        
                        if (!this.data.agents) return;
                        
                        const containerRect = container.getBoundingClientRect();
                        const width = containerRect.width || 400;
                        const height = containerRect.height || 200;
                        
                        // Draw agents
                        Object.entries(this.data.agents).forEach(([name, agent]) => {{
                            const node = document.createElement('div');
                            node.className = `agent-node ${{agent.state.toLowerCase()}}`;
                            node.textContent = name.substring(0, 4);
                            node.title = `${{name}} - ${{agent.state}}`;
                            
                            const x = (agent.position?.x || 0.5) * (width - 60);
                            const y = (agent.position?.y || 0.5) * (height - 60);
                            
                            node.style.left = `${{x}}px`;
                            node.style.top = `${{y}}px`;
                            
                            container.appendChild(node);
                        }});
                    }}
                    
                    updateMetrics() {{
                        const container = document.getElementById('metricsContainer');
                        container.innerHTML = '';
                        
                        if (!this.data.metrics) return;
                        
                        const metrics = [
                            {{ label: 'Total Agents', value: Object.keys(this.data.agents || {{}}).length }},
                            {{ label: 'Active Tasks', value: this.data.orchestrator_metrics?.active_tasks || 0 }},
                            {{ label: 'Completed Tasks', value: this.data.orchestrator_metrics?.completed_tasks || 0 }},
                            {{ label: 'Success Rate', value: `${{(this.data.metrics.success_rate * 100).toFixed(1)}}%` || 'N/A' }}
                        ];
                        
                        metrics.forEach(metric => {{
                            const card = document.createElement('div');
                            card.className = 'metric-card';
                            card.innerHTML = `
                                <div class="metric-value">${{metric.value}}</div>
                                <div class="metric-label">${{metric.label}}</div>
                            `;
                            container.appendChild(card);
                        }});
                    }}
                    
                    updateInteractions() {{
                        const container = document.getElementById('interactionsContainer');
                        container.innerHTML = '';
                        
                        if (!this.data.interactions) return;
                        
                        this.data.interactions.slice(0, 10).forEach(interaction => {{
                            const item = document.createElement('div');
                            item.className = 'interaction-item';
                            item.innerHTML = `
                                <div>
                                    <span class="status-indicator status-active"></span>
                                    ${{interaction.from_agent || 'System'}} → ${{interaction.to_agent || 'System'}}
                                </div>
                                <div style="font-size: 0.75rem; color: #a0aec0; margin-top: 0.25rem;">
                                    ${{new Date(interaction.timestamp).toLocaleTimeString()}}
                                </div>
                            `;
                            container.appendChild(item);
                        }});
                    }}
                    
                    updateAgentsList() {{
                        const container = document.getElementById('agentsContainer');
                        container.innerHTML = '';
                        
                        if (!this.data.agents) return;
                        
                        Object.entries(this.data.agents).forEach(([name, agent]) => {{
                            const agentEl = document.createElement('div');
                            agentEl.style.cssText = `
                                background: #1a202c;
                                margin: 0.5rem 0;
                                padding: 1rem;
                                border-radius: 4px;
                                cursor: pointer;
                            `;
                            
                            agentEl.innerHTML = `
                                <div style="display: flex; justify-content: space-between; align-items: center;">
                                    <div>
                                        <strong>${{name}}</strong>
                                        <div style="font-size: 0.875rem; color: #a0aec0;">
                                            State: ${{agent.state}} | Tasks: ${{agent.metrics?.total_tasks || 0}}
                                        </div>
                                    </div>
                                    <span class="status-indicator status-${{agent.state.toLowerCase()}}"></span>
                                </div>
                            `;
                            
                            agentEl.onclick = () => this.showAgentDetails(name);
                            container.appendChild(agentEl);
                        }});
                    }}
                    
                    showAgentDetails(agentName) {{
                        this.ws.send(JSON.stringify({{
                            type: 'get_agent_details',
                            agent_name: agentName
                        }}));
                    }}
                }}
                
                // Initialize dashboard
                const dashboard = new DashboardClient();
            </script>
        </body>
        </html>
        """
    
    def get_dashboard_statistics(self) -> Dict[str, Any]:
        """Get dashboard usage statistics"""
        return {
            "active_connections": len(self.active_connections),
            "is_running": self.is_running,
            "update_interval": self.update_interval,
            "orchestrator_attached": self.orchestrator is not None,
            "metrics_collected": self.metrics_collector.get_collection_count(),
            "interactions_tracked": self.interaction_tracker.get_interaction_count()
        }