"""
Real-time Monitoring Dashboard - Phase 7 Autonomous Intelligence Ecosystem
Comprehensive monitoring and observability for autonomous systems
"""

import asyncio
from pathlib import Path
import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import statistics
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Web framework for dashboard
try:
    from fastapi import FastAPI, WebSocket, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    WEB_FRAMEWORK_AVAILABLE = True
except ImportError:
    WEB_FRAMEWORK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("FastAPI not available - dashboard will run in console mode only")

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


@dataclass
class SystemMetrics:
    """Real-time system metrics"""
    timestamp: datetime
    
    # System health
    overall_health: str
    component_health: Dict[str, str]
    
    # Performance metrics
    success_rate: float
    response_time_ms: float
    throughput_rps: float
    error_rate: float
    
    # Resource utilization
    cpu_utilization: float
    memory_utilization: float
    disk_utilization: float
    network_utilization: float
    
    # Autonomous intelligence metrics
    autonomous_success_rate: float
    reasoning_accuracy: float
    memory_coherence: float
    causal_accuracy: float
    
    # Business metrics
    roi_percentage: float
    business_value_generated: float
    cost_reduction_achieved: float
    processes_automated: int
    
    # Evolution metrics
    improvement_rate: float
    capabilities_discovered: int
    breakthrough_behaviors: int
    evolution_active: bool


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str
    name: str
    metric_path: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: str    # 'info', 'warning', 'critical'
    enabled: bool = True
    consecutive_violations: int = 1
    current_violations: int = 0


@dataclass
class Alert:
    """System alert"""
    alert_id: str
    rule_id: str
    severity: str
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime
    acknowledged: bool = False
    resolved: bool = False


class RealTimeMonitoringDashboard:
    """
    Real-time monitoring dashboard for autonomous intelligence ecosystem
    
    Features:
    - Real-time system health monitoring
    - Performance metrics visualization
    - Business intelligence tracking
    - Evolution progress monitoring
    - Alert management
    - Interactive web interface
    """
    
    def __init__(self, 
                 update_interval_seconds: int = 10,
                 metrics_retention_hours: int = 24,
                 enable_web_interface: bool = True):
        
        self.update_interval = update_interval_seconds
        self.metrics_retention_hours = metrics_retention_hours
        self.enable_web_interface = enable_web_interface and WEB_FRAMEWORK_AVAILABLE
        
        # Connected systems for monitoring
        self.monitored_systems: Dict[str, Any] = {}
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=int(metrics_retention_hours * 3600 / update_interval_seconds))
        self.current_metrics: Optional[SystemMetrics] = None
        
        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # Web interface
        self.app: Optional[FastAPI] = None
        self.websocket_connections: List[WebSocket] = []
        
        # Background tasks
        self.background_tasks: set = set()
        self.shutdown_event = asyncio.Event()
        
        # Dashboard state
        self.dashboard_start_time = datetime.now()
        self.total_metrics_collected = 0
        self.total_alerts_generated = 0
        
        # Initialize default alert rules
        self._initialize_default_alert_rules()
        
        logger.info(f"Real-time Monitoring Dashboard initialized")
        logger.info(f"Update interval: {update_interval_seconds} seconds")
        logger.info(f"Metrics retention: {metrics_retention_hours} hours")
        logger.info(f"Web interface: {'enabled' if self.enable_web_interface else 'disabled'}")
    
    async def connect_systems(self, **systems):
        """Connect systems to monitor"""
        
        self.monitored_systems.update(systems)
        
        logger.info(f"Connected {len(systems)} systems for monitoring:")
        for name in systems.keys():
            logger.info(f"  - {name}")
    
    async def start_monitoring(self) -> Dict[str, Any]:
        """Start real-time monitoring"""
        
        logger.info("🔄 Starting real-time monitoring")
        
        # Start metrics collection
        metrics_task = asyncio.create_task(self._metrics_collection_loop())
        self.background_tasks.add(metrics_task)
        
        # Start alert processing
        alerts_task = asyncio.create_task(self._alert_processing_loop())
        self.background_tasks.add(alerts_task)
        
        # Start web interface if enabled
        if self.enable_web_interface:
            web_task = asyncio.create_task(self._start_web_interface())
            self.background_tasks.add(web_task)
        
        # Start console monitoring
        console_task = asyncio.create_task(self._console_monitoring_loop())
        self.background_tasks.add(console_task)
        
        logger.info("✅ Real-time monitoring started")
        
        return {
            "monitoring_active": True,
            "connected_systems": len(self.monitored_systems),
            "update_interval": self.update_interval,
            "web_interface": self.enable_web_interface,
            "alert_rules": len(self.alert_rules)
        }
    
    async def _metrics_collection_loop(self):
        """Background metrics collection loop"""
        
        logger.info("📊 Starting metrics collection loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Collect metrics from all connected systems
                metrics = await self._collect_system_metrics()
                
                if metrics:
                    self.current_metrics = metrics
                    self.metrics_history.append(metrics)
                    self.total_metrics_collected += 1
                    
                    # Broadcast to websockets if web interface is active
                    if self.enable_web_interface and self.websocket_connections:
                        await self._broadcast_metrics_update(metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(min(60, self.update_interval * 2))
    
    async def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect comprehensive system metrics"""
        
        try:
            # Initialize metrics with defaults
            metrics_data = {
                'timestamp': datetime.now(),
                'overall_health': 'unknown',
                'component_health': {},
                'success_rate': 0.0,
                'response_time_ms': 0.0,
                'throughput_rps': 0.0,
                'error_rate': 0.0,
                'cpu_utilization': 0.0,
                'memory_utilization': 0.0,
                'disk_utilization': 0.0,
                'network_utilization': 0.0,
                'autonomous_success_rate': 0.0,
                'reasoning_accuracy': 0.0,
                'memory_coherence': 0.0,
                'causal_accuracy': 0.0,
                'roi_percentage': 0.0,
                'business_value_generated': 0.0,
                'cost_reduction_achieved': 0.0,
                'processes_automated': 0,
                'improvement_rate': 0.0,
                'capabilities_discovered': 0,
                'breakthrough_behaviors': 0,
                'evolution_active': False
            }
            
            # Collect from master controller
            master_controller = self.monitored_systems.get('master_controller')
            if master_controller:
                try:
                    system_status = await master_controller.get_comprehensive_system_status()
                    
                    # System health
                    system_overview = system_status.get('system_overview', {})
                    metrics_data['overall_health'] = system_overview.get('state', 'unknown')
                    metrics_data['component_health'] = system_overview.get('components_operational', {})
                    
                    # Performance metrics
                    perf_metrics = system_status.get('performance_metrics', {})
                    metrics_data['success_rate'] = perf_metrics.get('overall_success_rate', 0.0)
                    metrics_data['business_value_generated'] = perf_metrics.get('business_value', 0.0)
                    
                    # Autonomous intelligence metrics
                    autonomous_metrics = system_status.get('autonomous_intelligence', {}).get('autonomous_metrics', {})
                    metrics_data['autonomous_success_rate'] = autonomous_metrics.get('autonomous_success_rate', 0.0)
                    
                    # Reasoning metrics
                    reasoning_status = system_status.get('autonomous_intelligence', {}).get('reasoning_status', {})
                    if 'integrated_controller' in reasoning_status:
                        reasoning_perf = reasoning_status['integrated_controller']['performance_metrics']
                        metrics_data['reasoning_accuracy'] = reasoning_perf.get('average_accuracy', 0.0)
                        metrics_data['memory_coherence'] = reasoning_perf.get('memory_coherence_rate', 0.0)
                        metrics_data['causal_accuracy'] = reasoning_perf.get('causal_accuracy_rate', 0.0)
                    
                    # Business metrics
                    business_integration = system_status.get('business_integration', {})
                    if business_integration:
                        metrics_data['roi_percentage'] = business_integration.get('current_roi', 0.0)
                        metrics_data['processes_automated'] = business_integration.get('processes_automated', 0)
                        metrics_data['cost_reduction_achieved'] = business_integration.get('cost_reduction_achieved', 0.0)
                    
                    # Evolution metrics
                    evolution_system = system_status.get('evolution_system', {})
                    if evolution_system:
                        metrics_data['improvement_rate'] = evolution_system.get('quarterly_improvement', 0.0)
                        metrics_data['capabilities_discovered'] = evolution_system.get('capabilities_discovered', 0)
                        metrics_data['breakthrough_behaviors'] = evolution_system.get('breakthrough_behaviors', 0)
                        metrics_data['evolution_active'] = evolution_system.get('evolution_active', False)
                
                except Exception as e:
                    logger.debug(f"Error collecting master controller metrics: {e}")
            
            # Collect from deployment manager
            deployment_manager = self.monitored_systems.get('deployment_manager')
            if deployment_manager:
                try:
                    deployment_status = await deployment_manager.get_deployment_status()
                    
                    # Resource utilization
                    resource_util = deployment_status.get('resource_utilization', {})
                    metrics_data['cpu_utilization'] = resource_util.get('cpu_utilization', 0.0)
                    metrics_data['memory_utilization'] = resource_util.get('memory_utilization', 0.0)
                    metrics_data['disk_utilization'] = resource_util.get('disk_utilization', 0.0)
                    metrics_data['network_utilization'] = resource_util.get('network_utilization', 0.0)
                    
                    # Performance metrics
                    perf_metrics = deployment_status.get('performance_metrics', {})
                    metrics_data['response_time_ms'] = perf_metrics.get('average_response_time_ms', 0.0)
                    metrics_data['throughput_rps'] = perf_metrics.get('requests_per_second', 0.0)
                    metrics_data['error_rate'] = perf_metrics.get('error_rate', 0.0)
                
                except Exception as e:
                    logger.debug(f"Error collecting deployment manager metrics: {e}")
            
            # Collect from business orchestrator
            business_orchestrator = self.monitored_systems.get('business_orchestrator')
            if business_orchestrator:
                try:
                    business_metrics = await business_orchestrator.get_business_metrics()
                    
                    roi_metrics = business_metrics.get('roi_metrics', {})
                    metrics_data['roi_percentage'] = roi_metrics.get('current_roi', 0.0)
                    
                    business_value = business_metrics.get('business_value', {})
                    metrics_data['business_value_generated'] = business_value.get('total_business_value', 0.0)
                    metrics_data['cost_reduction_achieved'] = business_value.get('cost_reduction_achieved', 0.0)
                    
                    automation_metrics = business_metrics.get('automation_metrics', {})
                    metrics_data['processes_automated'] = automation_metrics.get('processes_automated', 0)
                
                except Exception as e:
                    logger.debug(f"Error collecting business orchestrator metrics: {e}")
            
            # Collect from evolution engine
            evolution_engine = self.monitored_systems.get('evolution_engine')
            if evolution_engine:
                try:
                    evolution_metrics = await evolution_engine.get_evolution_metrics()
                    
                    perf_metrics = evolution_metrics.get('performance_metrics', {})
                    metrics_data['improvement_rate'] = perf_metrics.get('quarterly_improvement', 0.0)
                    
                    evolution_stats = evolution_metrics.get('evolution_statistics', {})
                    metrics_data['capabilities_discovered'] = evolution_stats.get('breakthrough_discoveries', 0)
                    
                    evolution_status = evolution_metrics.get('evolution_status', {})
                    metrics_data['evolution_active'] = evolution_status.get('active', False)
                
                except Exception as e:
                    logger.debug(f"Error collecting evolution engine metrics: {e}")
            
            # Create SystemMetrics object
            return SystemMetrics(**metrics_data)
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    async def _alert_processing_loop(self):
        """Background alert processing loop"""
        
        logger.info("🚨 Starting alert processing loop")
        
        while not self.shutdown_event.is_set():
            try:
                if self.current_metrics:
                    await self._process_alert_rules(self.current_metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Alert processing error: {e}")
                await asyncio.sleep(60)
    
    async def _process_alert_rules(self, metrics: SystemMetrics):
        """Process alert rules against current metrics"""
        
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue
            
            try:
                # Get metric value
                metric_value = self._get_metric_value(metrics, rule.metric_path)
                
                if metric_value is None:
                    continue
                
                # Check threshold
                violation = False
                if rule.comparison == 'gt' and metric_value > rule.threshold:
                    violation = True
                elif rule.comparison == 'lt' and metric_value < rule.threshold:
                    violation = True
                elif rule.comparison == 'eq' and abs(metric_value - rule.threshold) < 0.001:
                    violation = True
                
                if violation:
                    rule.current_violations += 1
                    
                    if rule.current_violations >= rule.consecutive_violations:
                        # Generate alert
                        await self._generate_alert(rule, metric_value)
                        rule.current_violations = 0  # Reset after alerting
                else:
                    rule.current_violations = 0  # Reset counter
                
            except Exception as e:
                logger.debug(f"Error processing alert rule {rule.name}: {e}")
    
    def _get_metric_value(self, metrics: SystemMetrics, metric_path: str) -> Optional[float]:
        """Get metric value by path"""
        
        try:
            # Simple path resolution
            if hasattr(metrics, metric_path):
                return getattr(metrics, metric_path)
            
            # Handle nested paths (e.g., 'component_health.orchestrator')
            if '.' in metric_path:
                parts = metric_path.split('.')
                value = metrics
                for part in parts:
                    if hasattr(value, part):
                        value = getattr(value, part)
                    elif isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return {}
                
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Convert health status to numeric
                    if value in ['healthy', 'operational']:
                        return 1.0
                    elif value in ['unhealthy', 'failed']:
                        return 0.0
                    elif value == 'degraded':
                        return 0.5
            
            return {}
            
        except Exception:
            return {}
    
    async def _generate_alert(self, rule: AlertRule, metric_value: float):
        """Generate an alert"""
        
        alert_id = f"alert_{rule.rule_id}_{int(time.time())}"
        
        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            message=f"{rule.name}: {metric_value:.3f} {rule.comparison} {rule.threshold}",
            metric_value=metric_value,
            threshold=rule.threshold,
            timestamp=datetime.now()
        )
        
        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        self.total_alerts_generated += 1
        
        # Log alert
        log_level = logging.CRITICAL if rule.severity == 'critical' else logging.WARNING
        logger.log(log_level, f"ALERT: {alert.message}")
        
        # Broadcast to websockets
        if self.enable_web_interface and self.websocket_connections:
            await self._broadcast_alert(alert)
    
    async def _console_monitoring_loop(self):
        """Console-based monitoring display"""
        
        logger.info("💻 Starting console monitoring display")
        
        while not self.shutdown_event.is_set():
            try:
                if self.current_metrics:
                    await self._display_console_dashboard()
                
                await asyncio.sleep(30)  # Update console every 30 seconds
                
            except Exception as e:
                logger.error(f"Console monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _display_console_dashboard(self):
        """Display dashboard in console"""
        
        if not self.current_metrics:
            return {}
        
        metrics = self.current_metrics
        
        # Clear screen (simple version)
        print("\n" * 2)
        print("="*80)
        print("🤖 AUTONOMOUS INTELLIGENCE ECOSYSTEM - REAL-TIME MONITORING")
        print("="*80)
        
        # System Health
        print(f"🔍 SYSTEM HEALTH: {metrics.overall_health.upper()}")
        print(f"⏰ Last Update: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Performance Metrics
        print("📊 PERFORMANCE METRICS")
        print(f"  Success Rate: {metrics.success_rate:.1%}")
        print(f"  Response Time: {metrics.response_time_ms:.1f} ms")
        print(f"  Throughput: {metrics.throughput_rps:.1f} RPS")
        print(f"  Error Rate: {metrics.error_rate:.1%}")
        print()
        
        # Resource Utilization
        print("💻 RESOURCE UTILIZATION")
        print(f"  CPU: {metrics.cpu_utilization:.1f}%")
        print(f"  Memory: {metrics.memory_utilization:.1f}%")
        print(f"  Disk: {metrics.disk_utilization:.1f}%")
        print(f"  Network: {metrics.network_utilization:.1f}%")
        print()
        
        # Autonomous Intelligence
        print("🤖 AUTONOMOUS INTELLIGENCE")
        print(f"  Autonomous Success: {metrics.autonomous_success_rate:.1%}")
        print(f"  Reasoning Accuracy: {metrics.reasoning_accuracy:.1%}")
        print(f"  Memory Coherence: {metrics.memory_coherence:.1%}")
        print(f"  Causal Accuracy: {metrics.causal_accuracy:.1%}")
        print()
        
        # Business Metrics
        print("💼 BUSINESS INTELLIGENCE")
        print(f"  ROI: {metrics.roi_percentage:.0f}%")
        print(f"  Business Value: ${metrics.business_value_generated:,.2f}")
        print(f"  Cost Reduction: {metrics.cost_reduction_achieved:.1%}")
        print(f"  Processes Automated: {metrics.processes_automated}")
        print()
        
        # Evolution Metrics
        print("🧬 CONTINUOUS EVOLUTION")
        print(f"  Evolution Active: {'Yes' if metrics.evolution_active else 'No'}")
        print(f"  Improvement Rate: {metrics.improvement_rate:.1%}")
        print(f"  Capabilities Discovered: {metrics.capabilities_discovered}")
        print(f"  Breakthrough Behaviors: {metrics.breakthrough_behaviors}")
        print()
        
        # Active Alerts
        if self.active_alerts:
            print("🚨 ACTIVE ALERTS")
            for alert in list(self.active_alerts.values())[-5:]:  # Last 5 alerts
                severity_icon = "🔴" if alert.severity == 'critical' else "🟡" if alert.severity == 'warning' else "🔵"
                print(f"  {severity_icon} {alert.message}")
            print()
        
        # Statistics
        uptime = datetime.now() - self.dashboard_start_time
        print("📈 MONITORING STATISTICS")
        print(f"  Uptime: {self._format_timedelta(uptime)}")
        print(f"  Metrics Collected: {self.total_metrics_collected:,}")
        print(f"  Alerts Generated: {self.total_alerts_generated}")
        print(f"  Connected Systems: {len(self.monitored_systems)}")
        
        print("="*80)
    
    def _format_timedelta(self, td: timedelta) -> str:
        """Format timedelta for display"""
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    
    def _initialize_default_alert_rules(self):
        """Initialize default alert rules"""
        
        default_rules = [
            AlertRule("system_health", "System Health Critical", "overall_health", 0.5, "lt", "critical"),
            AlertRule("success_rate_low", "Success Rate Below 80%", "success_rate", 0.8, "lt", "warning"),
            AlertRule("response_time_high", "Response Time Above 1s", "response_time_ms", 1000.0, "gt", "warning"),
            AlertRule("error_rate_high", "Error Rate Above 5%", "error_rate", 0.05, "gt", "warning"),
            AlertRule("cpu_high", "CPU Utilization Above 90%", "cpu_utilization", 90.0, "gt", "warning"),
            AlertRule("memory_high", "Memory Utilization Above 90%", "memory_utilization", 90.0, "gt", "warning"),
            AlertRule("autonomous_success_low", "Autonomous Success Below 70%", "autonomous_success_rate", 0.7, "lt", "warning"),
            AlertRule("reasoning_accuracy_low", "Reasoning Accuracy Below 80%", "reasoning_accuracy", 0.8, "lt", "warning"),
            AlertRule("roi_target_missed", "ROI Below 1000%", "roi_percentage", 1000.0, "lt", "info"),
            AlertRule("evolution_stopped", "Evolution Engine Stopped", "evolution_active", 0.5, "lt", "warning")
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule
        
        logger.info(f"Initialized {len(default_rules)} default alert rules")
    
    # Web interface methods (if FastAPI is available)
    
    async def _start_web_interface(self):
        """Start web interface"""
        
        if not WEB_FRAMEWORK_AVAILABLE:
            logger.warning("FastAPI not available - web interface disabled")
            return {}
        
        try:
            # Initialize FastAPI app
            self.app = FastAPI(title="Autonomous Intelligence Monitoring Dashboard")
            
            # Setup routes
            self._setup_web_routes()
            
            # Start server
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=8080,
                log_level="info"
            )
            server = uvicorn.Server(config)
            
            logger.info("🌐 Starting web interface on http://localhost:8080")
            await server.serve()
            
        except Exception as e:
            logger.error(f"Web interface failed to start: {e}")
    
    def _setup_web_routes(self):
        """Setup web routes"""
        
        if not self.app:
            return {}
        
        @self.app.get(str(Path("/").resolve()))
        async def dashboard_home():
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get(str(Path("/api/metrics").resolve()))
        async def get_current_metrics():
            if self.current_metrics:
                return self._serialize_metrics(self.current_metrics)
            return {}
        
        @self.app.get(str(Path("/api/alerts").resolve()))
        async def get_active_alerts():
            return [self._serialize_alert(alert) for alert in self.active_alerts.values()]
        
        @self.app.get(str(Path("/api/status").resolve()))
        async def get_system_status():
            return {
                "monitoring_active": not self.shutdown_event.is_set(),
                "connected_systems": len(self.monitored_systems),
                "metrics_collected": self.total_metrics_collected,
                "alerts_generated": self.total_alerts_generated,
                "uptime_seconds": (datetime.now() - self.dashboard_start_time).total_seconds()
            }
        
        @self.app.websocket(str(Path("/ws").resolve()))
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    await websocket.receive_text()
            except:
            logger.info(f'Method {function_name} called')
            return {}
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Autonomous Intelligence Monitoring</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { text-align: center; background: #2c3e50; color: white; padding: 20px; border-radius: 8px; }
                .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }
                .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .metric-title { font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #2c3e50; }
                .metric-value { font-size: 24px; font-weight: bold; color: #3498db; }
                .alert { background: #e74c3c; color: white; padding: 10px; margin: 5px 0; border-radius: 4px; }
                .status-healthy { color: #27ae60; }
                .status-warning { color: #f39c12; }
                .status-critical { color: #e74c3c; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 Autonomous Intelligence Ecosystem</h1>
                    <h2>Real-Time Monitoring Dashboard</h2>
                </div>
                
                <div id="dashboard-content">
                    <p>Loading dashboard...</p>
                </div>
            </div>
            
            <script>
                // WebSocket connection for real-time updates
                const ws = new WebSocket('ws://localhost:8080/ws');
                
                ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    updateDashboard(data);
                };
                
                // Initial load
                fetch(str(Path('/api/metrics').resolve()))
                    .then(response => response.json())
                    .then(data => updateDashboard(data));
                
                function updateDashboard(metrics) {
                    const content = document.getElementById('dashboard-content');
                    
                    if (metrics.timestamp) {
                        content.innerHTML = `
                            <div class="metrics-grid">
                                <div class="metric-card">
                                    <div class="metric-title">System Health</div>
                                    <div class="metric-value status-${getHealthClass(metrics.overall_health)}">
                                        ${metrics.overall_health.toUpperCase()}
                                    </div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">Success Rate</div>
                                    <div class="metric-value">${(metrics.success_rate * 100).toFixed(1)}%</div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">Response Time</div>
                                    <div class="metric-value">${metrics.response_time_ms.toFixed(1)} ms</div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">ROI Achievement</div>
                                    <div class="metric-value">${metrics.roi_percentage.toFixed(0)}%</div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">Autonomous Success</div>
                                    <div class="metric-value">${(metrics.autonomous_success_rate * 100).toFixed(1)}%</div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">Reasoning Accuracy</div>
                                    <div class="metric-value">${(metrics.reasoning_accuracy * 100).toFixed(1)}%</div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">Evolution Active</div>
                                    <div class="metric-value">${metrics.evolution_active ? 'Yes' : 'No'}</div>
                                </div>
                                
                                <div class="metric-card">
                                    <div class="metric-title">Processes Automated</div>
                                    <div class="metric-value">${metrics.processes_automated}</div>
                                </div>
                            </div>
                            
                            <p><strong>Last Updated:</strong> ${new Date(metrics.timestamp).toLocaleString()}</p>
                        `;
                    }
                }
                
                function getHealthClass(health) {
                    if (health === 'operational' || health === 'healthy') return 'healthy';
                    if (health === 'degraded') return 'warning';
                    return 'critical';
                }
                
                // Refresh every 10 seconds
                setInterval(() => {
                    fetch(str(Path('/api/metrics').resolve()))
                        .then(response => response.json())
                        .then(data => updateDashboard(data));
                }, 10000);
            </script>
        </body>
        </html>
        """
    
    async def _broadcast_metrics_update(self, metrics: SystemMetrics):
        """Broadcast metrics update to websockets"""
        
        if not self.websocket_connections:
            return {}
        
        metrics_data = self._serialize_metrics(metrics)
        message = json.dumps({"type": "metrics_update", "data": metrics_data})
        
        # Send to all connected websockets
        for websocket in self.websocket_connections.copy():
            try:
                await websocket.send_text(message)
            except:
                self.websocket_connections.remove(websocket)
    
    async def _broadcast_alert(self, alert: Alert):
        """Broadcast alert to websockets"""
        
        if not self.websocket_connections:
            return {}
        
        alert_data = self._serialize_alert(alert)
        message = json.dumps({"type": "alert", "data": alert_data})
        
        for websocket in self.websocket_connections.copy():
            try:
                await websocket.send_text(message)
            except:
                self.websocket_connections.remove(websocket)
    
    def _serialize_metrics(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Serialize metrics for JSON transmission"""
        
        return {
            "timestamp": metrics.timestamp.isoformat(),
            "overall_health": metrics.overall_health,
            "component_health": metrics.component_health,
            "success_rate": metrics.success_rate,
            "response_time_ms": metrics.response_time_ms,
            "throughput_rps": metrics.throughput_rps,
            "error_rate": metrics.error_rate,
            "cpu_utilization": metrics.cpu_utilization,
            "memory_utilization": metrics.memory_utilization,
            "disk_utilization": metrics.disk_utilization,
            "network_utilization": metrics.network_utilization,
            "autonomous_success_rate": metrics.autonomous_success_rate,
            "reasoning_accuracy": metrics.reasoning_accuracy,
            "memory_coherence": metrics.memory_coherence,
            "causal_accuracy": metrics.causal_accuracy,
            "roi_percentage": metrics.roi_percentage,
            "business_value_generated": metrics.business_value_generated,
            "cost_reduction_achieved": metrics.cost_reduction_achieved,
            "processes_automated": metrics.processes_automated,
            "improvement_rate": metrics.improvement_rate,
            "capabilities_discovered": metrics.capabilities_discovered,
            "breakthrough_behaviors": metrics.breakthrough_behaviors,
            "evolution_active": metrics.evolution_active
        }
    
    def _serialize_alert(self, alert: Alert) -> Dict[str, Any]:
        """Serialize alert for JSON transmission"""
        
        return {
            "alert_id": alert.alert_id,
            "rule_id": alert.rule_id,
            "severity": alert.severity,
            "message": alert.message,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "timestamp": alert.timestamp.isoformat(),
            "acknowledged": alert.acknowledged,
            "resolved": alert.resolved
        }
    
    # Public API methods
    
    async def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status and statistics"""
        
        return {
            "dashboard_active": not self.shutdown_event.is_set(),
            "uptime_seconds": (datetime.now() - self.dashboard_start_time).total_seconds(),
            "connected_systems": len(self.monitored_systems),
            "metrics_collected": self.total_metrics_collected,
            "alerts_generated": self.total_alerts_generated,
            "active_alerts": len(self.active_alerts),
            "update_interval": self.update_interval,
            "web_interface_enabled": self.enable_web_interface,
            "websocket_connections": len(self.websocket_connections) if self.enable_web_interface else 0,
            "current_metrics": self._serialize_metrics(self.current_metrics) if self.current_metrics else None
        }
    
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert acknowledged: {alert_id}")
            return True
        
        return False
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.acknowledged = True
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            logger.info(f"Alert resolved: {alert_id}")
            return True
        
        return False
    
    async def add_alert_rule(self, rule: AlertRule) -> bool:
        """Add a new alert rule"""
        
        self.alert_rules[rule.rule_id] = rule
        logger.info(f"Added alert rule: {rule.name}")
        return True
    
    async def remove_alert_rule(self, rule_id: str) -> bool:
        """Remove an alert rule"""
        
        if rule_id in self.alert_rules:
            del self.alert_rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
            return True
        
        return False
    
    async def shutdown(self):
        """Gracefully shutdown monitoring dashboard"""
        
        logger.info("🔄 Shutting down monitoring dashboard...")
        
        # Stop monitoring
        self.shutdown_event.set()
        
        # Close websocket connections
        for websocket in self.websocket_connections:
            try:
                await websocket.close()
            except:
        logger.info(f'Method {function_name} called')
        return {}
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Final statistics
        uptime = datetime.now() - self.dashboard_start_time
        
        logger.info("📊 Final Monitoring Statistics:")
        logger.info(f"Total uptime: {self._format_timedelta(uptime)}")
        logger.info(f"Metrics collected: {self.total_metrics_collected:,}")
        logger.info(f"Alerts generated: {self.total_alerts_generated}")
        logger.info(f"Systems monitored: {len(self.monitored_systems)}")
        
        logger.info("✅ Monitoring dashboard shutdown complete")


# Standalone dashboard runner
async def run_standalone_dashboard():
    """Run dashboard as standalone application"""
    
    logger.info("🚀 Starting standalone monitoring dashboard")
    
    dashboard = RealTimeMonitoringDashboard(
        update_interval_seconds=5,
        enable_web_interface=True
    )
    
    # Start monitoring (will run in demo mode without connected systems)
    await dashboard.start_monitoring()
    
    try:
        # Keep running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Dashboard interrupted by user")
    finally:
        await dashboard.shutdown()


if __name__ == "__main__":
    # Run standalone dashboard for demonstration
    asyncio.run(run_standalone_dashboard())