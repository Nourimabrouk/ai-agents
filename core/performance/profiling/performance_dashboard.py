"""
Performance Dashboard for Phase 7 - Real-time Performance Monitoring
Centralized dashboard for CPU, memory, async operations, and optimization metrics
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: datetime
    cpu_metrics: Dict[str, Any]
    memory_metrics: Dict[str, Any]
    async_metrics: Dict[str, Any]
    algorithm_metrics: Dict[str, Any]
    cache_metrics: Dict[str, Any]
    system_metrics: Dict[str, Any]
    performance_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_metrics': self.cpu_metrics,
            'memory_metrics': self.memory_metrics,
            'async_metrics': self.async_metrics,
            'algorithm_metrics': self.algorithm_metrics,
            'cache_metrics': self.cache_metrics,
            'system_metrics': self.system_metrics,
            'performance_score': self.performance_score
        }


class PerformanceDashboard:
    """
    Centralized performance monitoring dashboard
    Aggregates metrics from all optimization components
    """
    
    def __init__(self, 
                 update_interval: float = 5.0,
                 max_metrics_history: int = 1000,
                 dashboard_port: int = 8080):
        
        self.update_interval = update_interval
        self.max_metrics_history = max_metrics_history
        self.dashboard_port = dashboard_port
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_metrics_history)
        self.current_metrics: Optional[PerformanceMetrics] = None
        
        # Component references (initialized later)
        self.cpu_profiler = None
        self.memory_profiler = None
        self.async_optimizer = None
        self.algorithm_optimizer = None
        self.cache_system = None
        
        # Dashboard state
        self.monitoring_active = False
        self.dashboard_task: Optional[asyncio.Task] = None
        
        # Performance thresholds
        self.thresholds = {
            'cpu_usage_warning': 70.0,
            'cpu_usage_critical': 90.0,
            'memory_usage_warning': 75.0,
            'memory_usage_critical': 90.0,
            'response_time_warning': 1.0,  # seconds
            'response_time_critical': 5.0,
            'cache_hit_rate_warning': 70.0,  # percent
            'cache_hit_rate_critical': 50.0
        }
        
        # Alert tracking
        self.active_alerts: List[Dict[str, Any]] = []
        self.alert_history: deque = deque(maxlen=100)
        
        logger.info(f"PerformanceDashboard initialized with {update_interval}s update interval")
    
    def register_components(self, 
                          cpu_profiler=None, 
                          memory_profiler=None,
                          async_optimizer=None, 
                          algorithm_optimizer=None,
                          cache_system=None):
        """Register performance monitoring components"""
        self.cpu_profiler = cpu_profiler
        self.memory_profiler = memory_profiler
        self.async_optimizer = async_optimizer
        self.algorithm_optimizer = algorithm_optimizer
        self.cache_system = cache_system
        
        logger.info("Performance monitoring components registered")
    
    async def start_monitoring(self):
        """Start the performance monitoring dashboard"""
        if self.monitoring_active:
            logger.warning("Performance monitoring already active")
            return {}
        
        self.monitoring_active = True
        self.dashboard_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Started performance monitoring dashboard")
    
    async def stop_monitoring(self):
        """Stop the performance monitoring dashboard"""
        self.monitoring_active = False
        
        if self.dashboard_task:
            self.dashboard_task.cancel()
            try:
                await self.dashboard_task
            except asyncio.CancelledError:
        logger.info(f'Method {function_name} called')
        return {}
        
        logger.info("Stopped performance monitoring dashboard")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Collect metrics from all components
                metrics = await self._collect_metrics()
                
                # Calculate performance score
                metrics.performance_score = self._calculate_performance_score(metrics)
                
                # Store metrics
                self.current_metrics = metrics
                self.metrics_history.append(metrics)
                
                # Check for alerts
                await self._check_alerts(metrics)
                
                # Log performance summary
                self._log_performance_summary(metrics)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.update_interval)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect metrics from all registered components"""
        
        # CPU metrics
        cpu_metrics = {}
        if self.cpu_profiler:
            try:
                cpu_metrics = self.cpu_profiler.get_performance_summary()
            except Exception as e:
                logger.debug(f"Error collecting CPU metrics: {e}")
                cpu_metrics = {'error': str(e)}
        
        # Memory metrics
        memory_metrics = {}
        if self.memory_profiler:
            try:
                memory_report = self.memory_profiler.get_memory_report()
                memory_metrics = memory_report.get('current_memory_usage', {})
            except Exception as e:
                logger.debug(f"Error collecting memory metrics: {e}")
                memory_metrics = {'error': str(e)}
        
        # Async optimizer metrics
        async_metrics = {}
        if self.async_optimizer:
            try:
                async_metrics = self.async_optimizer.get_performance_metrics()
            except Exception as e:
                logger.debug(f"Error collecting async metrics: {e}")
                async_metrics = {'error': str(e)}
        
        # Algorithm optimizer metrics
        algorithm_metrics = {}
        if self.algorithm_optimizer:
            try:
                algorithm_metrics = self.algorithm_optimizer.get_optimization_summary()
            except Exception as e:
                logger.debug(f"Error collecting algorithm metrics: {e}")
                algorithm_metrics = {'error': str(e)}
        
        # Cache metrics
        cache_metrics = {}
        if self.cache_system:
            try:
                cache_metrics = self.cache_system.get_performance_stats()
            except Exception as e:
                logger.debug(f"Error collecting cache metrics: {e}")
                cache_metrics = {'error': str(e)}
        
        # System metrics
        system_metrics = await self._collect_system_metrics()
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_metrics=cpu_metrics,
            memory_metrics=memory_metrics,
            async_metrics=async_metrics,
            algorithm_metrics=algorithm_metrics,
            cache_metrics=cache_metrics,
            system_metrics=system_metrics
        )
    
    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level performance metrics"""
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            
            # Memory usage
            memory = psutil.virtual_memory()
            
            # Disk usage
            disk = psutil.disk_usage(str(Path('/').resolve()))
            
            # Network I/O
            network = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'memory_total_gb': memory.total / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'memory_percent': memory.percent,
                'disk_total_gb': disk.total / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'disk_percent': (disk.used / disk.total) * 100,
                'network_bytes_sent': network.bytes_sent,
                'network_bytes_recv': network.bytes_recv,
                'process_memory_mb': process_memory.rss / (1024**2),
                'process_cpu_percent': process.cpu_percent()
            }
            
        except Exception as e:
            logger.debug(f"Error collecting system metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)"""
        score_components = []
        
        # CPU score (lower usage = higher score)
        cpu_usage = metrics.system_metrics.get('cpu_percent', 50)
        cpu_score = max(0, 100 - cpu_usage)
        score_components.append(('cpu', cpu_score, 0.25))
        
        # Memory score (lower usage = higher score)
        memory_usage = metrics.system_metrics.get('memory_percent', 50)
        memory_score = max(0, 100 - memory_usage)
        score_components.append(('memory', memory_score, 0.25))
        
        # Async performance score
        async_success_rate = 100.0  # Default
        if 'success_rate' in metrics.async_metrics:
            try:
                async_success_rate = float(metrics.async_metrics['success_rate'].rstrip('%'))
            except (ValueError, AttributeError):
        logger.info(f'Method {function_name} called')
        return {}
        score_components.append(('async', async_success_rate, 0.2))
        
        # Cache performance score
        cache_hit_rate = 80.0  # Default
        if 'hit_rate' in metrics.cache_metrics:
            try:
                cache_hit_rate = float(metrics.cache_metrics['hit_rate'].rstrip('%'))
            except (ValueError, AttributeError):
                pass
        score_components.append(('cache', cache_hit_rate, 0.15))
        
        # Algorithm optimization score
        algorithm_score = 85.0  # Default good score
        if 'total_optimizations' in metrics.algorithm_metrics:
            opt_count = metrics.algorithm_metrics['total_optimizations']
            # More optimizations = higher score (with diminishing returns)
            algorithm_score = min(100, 70 + (opt_count * 2))
        score_components.append(('algorithm', algorithm_score, 0.15))
        
        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in score_components)
        
        # Apply penalties for critical issues
        penalties = 0
        
        # CPU penalty
        if cpu_usage > self.thresholds['cpu_usage_critical']:
            penalties += 20
        elif cpu_usage > self.thresholds['cpu_usage_warning']:
            penalties += 10
        
        # Memory penalty
        if memory_usage > self.thresholds['memory_usage_critical']:
            penalties += 20
        elif memory_usage > self.thresholds['memory_usage_warning']:
            penalties += 10
        
        # Cache penalty
        if cache_hit_rate < self.thresholds['cache_hit_rate_critical']:
            penalties += 15
        elif cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
            penalties += 5
        
        final_score = max(0, total_score - penalties)
        return round(final_score, 1)
    
    async def _check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts"""
        current_alerts = []
        
        # CPU alerts
        cpu_usage = metrics.system_metrics.get('cpu_percent', 0)
        if cpu_usage > self.thresholds['cpu_usage_critical']:
            current_alerts.append({
                'type': 'cpu',
                'severity': 'critical',
                'message': f'Critical CPU usage: {cpu_usage:.1f}%',
                'value': cpu_usage,
                'threshold': self.thresholds['cpu_usage_critical']
            })
        elif cpu_usage > self.thresholds['cpu_usage_warning']:
            current_alerts.append({
                'type': 'cpu',
                'severity': 'warning',
                'message': f'High CPU usage: {cpu_usage:.1f}%',
                'value': cpu_usage,
                'threshold': self.thresholds['cpu_usage_warning']
            })
        
        # Memory alerts
        memory_usage = metrics.system_metrics.get('memory_percent', 0)
        if memory_usage > self.thresholds['memory_usage_critical']:
            current_alerts.append({
                'type': 'memory',
                'severity': 'critical',
                'message': f'Critical memory usage: {memory_usage:.1f}%',
                'value': memory_usage,
                'threshold': self.thresholds['memory_usage_critical']
            })
        elif memory_usage > self.thresholds['memory_usage_warning']:
            current_alerts.append({
                'type': 'memory',
                'severity': 'warning',
                'message': f'High memory usage: {memory_usage:.1f}%',
                'value': memory_usage,
                'threshold': self.thresholds['memory_usage_warning']
            })
        
        # Cache alerts
        if 'hit_rate' in metrics.cache_metrics:
            try:
                cache_hit_rate = float(metrics.cache_metrics['hit_rate'].rstrip('%'))
                if cache_hit_rate < self.thresholds['cache_hit_rate_critical']:
                    current_alerts.append({
                        'type': 'cache',
                        'severity': 'critical',
                        'message': f'Poor cache hit rate: {cache_hit_rate:.1f}%',
                        'value': cache_hit_rate,
                        'threshold': self.thresholds['cache_hit_rate_critical']
                    })
                elif cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
                    current_alerts.append({
                        'type': 'cache',
                        'severity': 'warning',
                        'message': f'Low cache hit rate: {cache_hit_rate:.1f}%',
                        'value': cache_hit_rate,
                        'threshold': self.thresholds['cache_hit_rate_warning']
                    })
            except (ValueError, AttributeError):
        logger.info(f'Method {function_name} called')
        return {}
        
        # Performance score alert
        if metrics.performance_score < 60:
            current_alerts.append({
                'type': 'performance',
                'severity': 'critical' if metrics.performance_score < 40 else 'warning',
                'message': f'Low performance score: {metrics.performance_score:.1f}/100',
                'value': metrics.performance_score,
                'threshold': 60
            })
        
        # Update alerts
        new_alerts = []
        for alert in current_alerts:
            alert['timestamp'] = datetime.now()
            # Check if this is a new alert
            existing_alert = next(
                (a for a in self.active_alerts 
                 if a['type'] == alert['type'] and a['severity'] == alert['severity']),
                None
            )
            
            if not existing_alert:
                new_alerts.append(alert)
                logger.warning(f"Performance alert: {alert['message']}")
        
        # Store new alerts
        for alert in new_alerts:
            self.alert_history.append(alert)
        
        self.active_alerts = current_alerts
    
    def _log_performance_summary(self, metrics: PerformanceMetrics):
        """Log performance summary"""
        cpu_usage = metrics.system_metrics.get('cpu_percent', 0)
        memory_usage = metrics.system_metrics.get('memory_percent', 0)
        
        logger.info(f"Performance Score: {metrics.performance_score:.1f}/100 | "
                   f"CPU: {cpu_usage:.1f}% | Memory: {memory_usage:.1f}% | "
                   f"Alerts: {len(self.active_alerts)}")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for web interface"""
        if not self.current_metrics:
            return {'error': 'No metrics available'}
        
        # Get recent metrics for trends
        recent_metrics = list(self.metrics_history)[-50:]  # Last 50 data points
        
        # Calculate trends
        trends = self._calculate_trends(recent_metrics)
        
        return {
            'current_metrics': self.current_metrics.to_dict(),
            'performance_score': self.current_metrics.performance_score,
            'active_alerts': self.active_alerts,
            'recent_alerts': list(self.alert_history)[-10:],
            'trends': trends,
            'monitoring_status': {
                'active': self.monitoring_active,
                'update_interval': self.update_interval,
                'metrics_history_count': len(self.metrics_history),
                'uptime_minutes': self._get_uptime_minutes()
            },
            'thresholds': self.thresholds
        }
    
    def _calculate_trends(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Calculate performance trends"""
        if len(metrics_list) < 2:
            return {}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_list]
        scores = [m.performance_score for m in metrics_list]
        cpu_usage = [m.system_metrics.get('cpu_percent', 0) for m in metrics_list]
        memory_usage = [m.system_metrics.get('memory_percent', 0) for m in metrics_list]
        
        def calculate_trend(values):
            if len(values) < 2:
                return 0
            return (values[-1] - values[0]) / len(values)
        
        return {
            'performance_score_trend': calculate_trend(scores),
            'cpu_usage_trend': calculate_trend(cpu_usage),
            'memory_usage_trend': calculate_trend(memory_usage),
            'time_window_minutes': (timestamps[-1] - timestamps[0]).total_seconds() / 60
        }
    
    def _get_uptime_minutes(self) -> float:
        """Get monitoring uptime in minutes"""
        if not self.metrics_history:
            return 0
        
        first_metric = self.metrics_history[0]
        current_time = datetime.now()
        uptime = (current_time - first_metric.timestamp).total_seconds() / 60
        return round(uptime, 1)
    
    def export_metrics(self, filepath: str, format: str = 'json'):
        """Export metrics to file"""
        if not self.metrics_history:
            logger.warning("No metrics to export")
            return {}
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'metrics_count': len(self.metrics_history),
            'time_range': {
                'start': self.metrics_history[0].timestamp.isoformat(),
                'end': self.metrics_history[-1].timestamp.isoformat()
            },
            'metrics': [m.to_dict() for m in self.metrics_history]
        }
        
        filepath = Path(filepath)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported {len(self.metrics_history)} metrics to {filepath}")
    
    def get_performance_report(self, time_window: timedelta = timedelta(hours=1)) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return {'error': 'No metrics available for report'}
        
        # Filter metrics by time window
        cutoff_time = datetime.now() - time_window
        recent_metrics = [
            m for m in self.metrics_history 
            if m.timestamp >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'error': 'No metrics in specified time window'}
        
        # Calculate statistics
        scores = [m.performance_score for m in recent_metrics]
        cpu_values = [m.system_metrics.get('cpu_percent', 0) for m in recent_metrics]
        memory_values = [m.system_metrics.get('memory_percent', 0) for m in recent_metrics]
        
        def calc_stats(values):
            if not values:
                return {'min': 0, 'max': 0, 'avg': 0}
            return {
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values)
            }
        
        # Count alerts by severity
        alert_counts = defaultdict(int)
        for alert in self.alert_history:
            if alert['timestamp'] >= cutoff_time:
                alert_counts[alert['severity']] += 1
        
        return {
            'report_period': {
                'start': recent_metrics[0].timestamp.isoformat(),
                'end': recent_metrics[-1].timestamp.isoformat(),
                'duration_minutes': time_window.total_seconds() / 60,
                'data_points': len(recent_metrics)
            },
            'performance_summary': {
                'current_score': self.current_metrics.performance_score if self.current_metrics else 0,
                'score_stats': calc_stats(scores),
                'cpu_stats': calc_stats(cpu_values),
                'memory_stats': calc_stats(memory_values)
            },
            'alert_summary': {
                'total_alerts': sum(alert_counts.values()),
                'by_severity': dict(alert_counts),
                'active_alerts': len(self.active_alerts)
            },
            'optimization_summary': self._get_optimization_summary(recent_metrics)
        }
    
    def _get_optimization_summary(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Summarize optimizations applied during the period"""
        if not metrics_list:
            return {}
        
        # Extract optimization data from the latest metrics
        latest = metrics_list[-1]
        
        summary = {
            'algorithm_optimizations': latest.algorithm_metrics.get('total_optimizations', 0),
            'cache_performance': latest.cache_metrics.get('hit_rate', 'N/A'),
            'async_task_success_rate': latest.async_metrics.get('success_rate', 'N/A')
        }
        
        return summary


# Global dashboard instance
dashboard = PerformanceDashboard()


# Convenience functions
async def start_performance_monitoring():
    """Start global performance monitoring"""
    await dashboard.start_monitoring()

async def stop_performance_monitoring():
    """Stop global performance monitoring"""
    await dashboard.stop_monitoring()

def get_current_performance_score() -> float:
    """Get current performance score"""
    if dashboard.current_metrics:
        return dashboard.current_metrics.performance_score
    return 0.0
