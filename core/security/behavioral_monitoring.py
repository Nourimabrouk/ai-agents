"""
Behavioral Monitoring System - Phase 7
Real-time behavioral analysis and anomaly detection for autonomous agents
Detects abnormal agent behaviors that may indicate security breaches
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import statistics
import json
from collections import defaultdict, deque

from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class AnomalyType(Enum):
    """Types of behavioral anomalies"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    EXCESSIVE_RESOURCE_USAGE = "excessive_resource_usage"
    UNUSUAL_COMMUNICATION_PATTERN = "unusual_communication_pattern"
    RAPID_MODIFICATION_ATTEMPTS = "rapid_modification_attempts"
    UNAUTHORIZED_ACCESS_ATTEMPTS = "unauthorized_access_attempts"
    ABNORMAL_TASK_EXECUTION = "abnormal_task_execution"
    SUSPICIOUS_COORDINATION_BEHAVIOR = "suspicious_coordination_behavior"


class AnomalySeverity(Enum):
    """Severity levels for detected anomalies"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class BehavioralAnomaly:
    """Represents a detected behavioral anomaly"""
    anomaly_id: str
    agent_name: str
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    evidence: Dict[str, Any]
    confidence_score: float
    baseline_data: Dict[str, Any]
    anomalous_data: Dict[str, Any]
    detected_at: datetime = field(default_factory=datetime.now)
    investigation_status: str = "pending"
    false_positive: bool = False


@dataclass
class AgentBehaviorProfile:
    """Behavioral profile for an agent"""
    agent_name: str
    creation_date: datetime
    total_observations: int
    avg_task_duration: float
    success_rate: float
    typical_actions: Dict[str, int]
    communication_frequency: float
    modification_frequency: float
    resource_usage_patterns: Dict[str, List[float]]
    last_updated: datetime = field(default_factory=datetime.now)


class BehavioralMonitor:
    """
    Monitors agent behavior in real-time
    Builds behavioral profiles and detects anomalies
    """
    
    def __init__(self, 
                 observation_window_minutes: int = 60,
                 anomaly_threshold: float = 2.0):
        self.observation_window = timedelta(minutes=observation_window_minutes)
        self.anomaly_threshold = anomaly_threshold  # Standard deviations from normal
        
        # Behavioral profiles
        self.agent_profiles: Dict[str, AgentBehaviorProfile] = {}
        
        # Recent observations for analysis
        self.recent_observations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Detected anomalies
        self.detected_anomalies: List[BehavioralAnomaly] = []
        
        # Monitoring state
        self.monitoring_active = True
        self.last_profile_update: Dict[str, datetime] = {}
        
        logger.info("Behavioral monitoring system initialized")
    
    async def monitor_agent_behavior(self, agent: BaseAgent) -> List[BehavioralAnomaly]:
        """Monitor and analyze agent behavior for anomalies"""
        if not self.monitoring_active:
            return []
        
        agent_name = agent.name
        current_anomalies = []
        
        try:
            # Update agent profile
            await self._update_agent_profile(agent)
            
            # Collect recent behavior data
            behavior_data = await self._collect_behavior_data(agent)
            
            # Store observation
            self.recent_observations[agent_name].append({
                'timestamp': datetime.now(),
                'behavior_data': behavior_data,
                'agent_state': getattr(agent, 'state', 'unknown')
            })
            
            # Analyze for anomalies
            anomalies = await self._detect_behavioral_anomalies(agent, behavior_data)
            current_anomalies.extend(anomalies)
            
            # Store detected anomalies
            for anomaly in anomalies:
                self.detected_anomalies.append(anomaly)
                logger.warning(f"Behavioral anomaly detected: {anomaly.anomaly_type.value} in agent {agent_name}")
                global_metrics.incr("security.behavioral_anomaly_detected")
            
        except Exception as e:
            logger.error(f"Behavioral monitoring failed for agent {agent_name}: {e}")
        
        return current_anomalies
    
    async def _update_agent_profile(self, agent: BaseAgent):
        """Update behavioral profile for an agent"""
        agent_name = agent.name
        
        # Check if it's time to update profile
        if agent_name in self.last_profile_update:
            if datetime.now() - self.last_profile_update[agent_name] < timedelta(minutes=5):
                return {}
        
        try:
            # Extract current metrics
            total_tasks = getattr(agent, 'total_tasks', 0)
            successful_tasks = getattr(agent, 'successful_tasks', 0)
            success_rate = successful_tasks / max(total_tasks, 1)
            
            # Analyze recent task patterns
            typical_actions = {}
            avg_duration = 0.0
            
            if hasattr(agent.memory, 'episodic_memory') and agent.memory.episodic_memory:
                recent_episodes = agent.memory.episodic_memory[-50:]  # Last 50 episodes
                
                # Analyze action types
                action_counts = defaultdict(int)
                durations = []
                
                for episode in recent_episodes:
                    action_type = episode.action.action_type
                    action_counts[action_type] += 1
                    
                    # Extract duration if available
                    if hasattr(episode.action, 'execution_time'):
                        durations.append(episode.action.execution_time)
                
                typical_actions = dict(action_counts)
                avg_duration = statistics.mean(durations) if durations else 0.0
            
            # Calculate modification frequency
            modification_count = len(getattr(agent, 'applied_modifications', []))
            modification_frequency = modification_count / max(total_tasks, 1)
            
            # Create/update profile
            profile = AgentBehaviorProfile(
                agent_name=agent_name,
                creation_date=self.agent_profiles.get(agent_name, AgentBehaviorProfile(
                    agent_name, datetime.now(), 0, 0, 0, {}, 0, 0, {}
                )).creation_date,
                total_observations=total_tasks,
                avg_task_duration=avg_duration,
                success_rate=success_rate,
                typical_actions=typical_actions,
                communication_frequency=0.0,  # Would track inter-agent communication
                modification_frequency=modification_frequency,
                resource_usage_patterns={}  # Would track CPU/memory usage patterns
            )
            
            self.agent_profiles[agent_name] = profile
            self.last_profile_update[agent_name] = datetime.now()
            
        except Exception as e:
            logger.error(f"Failed to update profile for agent {agent_name}: {e}")
    
    async def _collect_behavior_data(self, agent: BaseAgent) -> Dict[str, Any]:
        """Collect current behavioral data from agent"""
        behavior_data = {
            'timestamp': datetime.now().isoformat(),
            'agent_state': str(getattr(agent, 'state', 'unknown')),
            'total_tasks': getattr(agent, 'total_tasks', 0),
            'successful_tasks': getattr(agent, 'successful_tasks', 0),
            'modification_count': len(getattr(agent, 'applied_modifications', [])),
            'memory_size': 0,
            'recent_performance': []
        }
        
        # Memory analysis
        if hasattr(agent.memory, 'episodic_memory'):
            behavior_data['memory_size'] = len(agent.memory.episodic_memory)
            
            # Recent performance analysis
            recent_episodes = agent.memory.episodic_memory[-10:]
            behavior_data['recent_performance'] = [
                {
                    'success': ep.success,
                    'action_type': ep.action.action_type,
                    'timestamp': ep.timestamp.isoformat() if hasattr(ep, 'timestamp') else 'unknown'
                }
                for ep in recent_episodes
            ]
        
        # Performance metrics
        behavior_data['current_success_rate'] = agent.get_success_rate() if hasattr(agent, 'get_success_rate') else 0.0
        
        return behavior_data
    
    async def _detect_behavioral_anomalies(self, 
                                         agent: BaseAgent, 
                                         current_behavior: Dict[str, Any]) -> List[BehavioralAnomaly]:
        """Detect behavioral anomalies by comparing with historical profile"""
        anomalies = []
        agent_name = agent.name
        
        if agent_name not in self.agent_profiles:
            return anomalies  # No baseline yet
        
        profile = self.agent_profiles[agent_name]
        
        # 1. Performance degradation detection
        current_success_rate = current_behavior.get('current_success_rate', 0.0)
        if profile.success_rate > 0.5 and current_success_rate < profile.success_rate - 0.3:
            anomalies.append(BehavioralAnomaly(
                anomaly_id=f"perf_deg_{agent_name}_{int(datetime.now().timestamp())}",
                agent_name=agent_name,
                anomaly_type=AnomalyType.PERFORMANCE_DEGRADATION,
                severity=AnomalySeverity.HIGH,
                description=f"Significant performance degradation: {current_success_rate:.2%} vs {profile.success_rate:.2%}",
                evidence={
                    'current_success_rate': current_success_rate,
                    'baseline_success_rate': profile.success_rate,
                    'degradation': profile.success_rate - current_success_rate
                },
                confidence_score=0.8,
                baseline_data={'success_rate': profile.success_rate},
                anomalous_data={'success_rate': current_success_rate}
            ))
        
        # 2. Excessive modification attempts
        current_mod_count = current_behavior.get('modification_count', 0)
        if current_mod_count > 10 and current_mod_count > profile.total_observations * 0.2:
            anomalies.append(BehavioralAnomaly(
                anomaly_id=f"excess_mod_{agent_name}_{int(datetime.now().timestamp())}",
                agent_name=agent_name,
                anomaly_type=AnomalyType.RAPID_MODIFICATION_ATTEMPTS,
                severity=AnomalySeverity.MEDIUM,
                description=f"Excessive modification attempts: {current_mod_count}",
                evidence={
                    'modification_count': current_mod_count,
                    'total_tasks': profile.total_observations
                },
                confidence_score=0.7,
                baseline_data={'modification_frequency': profile.modification_frequency},
                anomalous_data={'current_modifications': current_mod_count}
            ))
        
        # 3. Abnormal task execution patterns
        recent_performance = current_behavior.get('recent_performance', [])
        if len(recent_performance) >= 5:
            recent_failures = sum(1 for p in recent_performance if not p.get('success', True))
            failure_rate = recent_failures / len(recent_performance)
            
            if failure_rate > 0.8 and profile.success_rate > 0.5:  # Sudden high failure rate
                anomalies.append(BehavioralAnomaly(
                    anomaly_id=f"task_fail_{agent_name}_{int(datetime.now().timestamp())}",
                    agent_name=agent_name,
                    anomaly_type=AnomalyType.ABNORMAL_TASK_EXECUTION,
                    severity=AnomalySeverity.HIGH,
                    description=f"Abnormal task failure pattern: {failure_rate:.2%} recent failures",
                    evidence={
                        'recent_failure_rate': failure_rate,
                        'recent_performance': recent_performance
                    },
                    confidence_score=0.9,
                    baseline_data={'baseline_success_rate': profile.success_rate},
                    anomalous_data={'recent_failure_rate': failure_rate}
                ))
        
        # 4. Memory growth anomaly
        current_memory_size = current_behavior.get('memory_size', 0)
        if current_memory_size > 1000:  # Threshold for excessive memory growth
            anomalies.append(BehavioralAnomaly(
                anomaly_id=f"memory_growth_{agent_name}_{int(datetime.now().timestamp())}",
                agent_name=agent_name,
                anomaly_type=AnomalyType.EXCESSIVE_RESOURCE_USAGE,
                severity=AnomalySeverity.MEDIUM,
                description=f"Excessive memory usage: {current_memory_size} episodes",
                evidence={'memory_size': current_memory_size},
                confidence_score=0.6,
                baseline_data={'typical_memory_size': 'unknown'},
                anomalous_data={'current_memory_size': current_memory_size}
            ))
        
        return anomalies
    
    def get_agent_profile(self, agent_name: str) -> Optional[AgentBehaviorProfile]:
        """Get behavioral profile for an agent"""
        return self.agent_profiles.get(agent_name)
    
    def get_recent_anomalies(self, 
                           agent_name: Optional[str] = None,
                           time_window_hours: int = 24) -> List[BehavioralAnomaly]:
        """Get recent anomalies for an agent or all agents"""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        filtered_anomalies = [
            anomaly for anomaly in self.detected_anomalies
            if anomaly.detected_at >= cutoff_time
        ]
        
        if agent_name:
            filtered_anomalies = [
                anomaly for anomaly in filtered_anomalies
                if anomaly.agent_name == agent_name
            ]
        
        return filtered_anomalies
    
    def mark_false_positive(self, anomaly_id: str):
        """Mark an anomaly as false positive"""
        for anomaly in self.detected_anomalies:
            if anomaly.anomaly_id == anomaly_id:
                anomaly.false_positive = True
                logger.info(f"Marked anomaly {anomaly_id} as false positive")
                break
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get behavioral monitoring metrics"""
        total_anomalies = len(self.detected_anomalies)
        false_positives = sum(1 for a in self.detected_anomalies if a.false_positive)
        
        # Anomaly distribution by type
        anomaly_types = defaultdict(int)
        for anomaly in self.detected_anomalies:
            anomaly_types[anomaly.anomaly_type.value] += 1
        
        return {
            'monitoring_active': self.monitoring_active,
            'tracked_agents': len(self.agent_profiles),
            'total_anomalies': total_anomalies,
            'false_positives': false_positives,
            'false_positive_rate': false_positives / max(total_anomalies, 1),
            'anomaly_distribution': dict(anomaly_types),
            'observation_window_minutes': self.observation_window.total_seconds() / 60,
            'anomaly_threshold': self.anomaly_threshold
        }


class AnomalyDetector:
    """
    Advanced anomaly detection using statistical methods
    Detects deviations from normal behavioral patterns
    """
    
    def __init__(self):
        self.detection_algorithms = {
            'statistical_threshold': self._statistical_threshold_detection,
            'isolation_forest': self._isolation_forest_detection,
            'behavioral_clustering': self._behavioral_clustering_detection
        }
        
        self.detection_history: List[Dict[str, Any]] = []
    
    async def detect_anomalies(self, 
                             behavioral_data: List[Dict[str, Any]],
                             detection_method: str = 'statistical_threshold') -> List[Dict[str, Any]]:
        """Detect anomalies in behavioral data"""
        
        if detection_method not in self.detection_algorithms:
            logger.error(f"Unknown detection method: {detection_method}")
            return []
        
        try:
            detection_func = self.detection_algorithms[detection_method]
            anomalies = await detection_func(behavioral_data)
            
            # Record detection run
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'method': detection_method,
                'data_points': len(behavioral_data),
                'anomalies_detected': len(anomalies)
            })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return []
    
    async def _statistical_threshold_detection(self, behavioral_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Statistical threshold-based anomaly detection"""
        anomalies = []
        
        if len(behavioral_data) < 10:
            return anomalies  # Need minimum data for statistical analysis
        
        # Extract numeric features for analysis
        features = ['total_tasks', 'successful_tasks', 'modification_count', 'memory_size']
        
        for feature in features:
            values = []
            for data in behavioral_data:
                if feature in data and isinstance(data[feature], (int, float)):
                    values.append(data[feature])
            
            if len(values) < 5:
                continue
            
            # Calculate statistics
            mean_val = statistics.mean(values)
            stdev_val = statistics.stdev(values) if len(values) > 1 else 0
            
            if stdev_val == 0:
                continue
            
            # Detect outliers (values > 2 standard deviations from mean)
            threshold = 2.0
            
            for i, data in enumerate(behavioral_data):
                if feature in data:
                    value = data[feature]
                    z_score = abs(value - mean_val) / stdev_val
                    
                    if z_score > threshold:
                        anomalies.append({
                            'data_index': i,
                            'feature': feature,
                            'value': value,
                            'z_score': z_score,
                            'mean': mean_val,
                            'stdev': stdev_val,
                            'anomaly_type': 'statistical_outlier'
                        })
        
        return anomalies
    
    async def _isolation_forest_detection(self, behavioral_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Isolation forest-based anomaly detection (simplified)"""
        # This would use scikit-learn's IsolationForest in a real implementation
        # For now, return empty list as placeholder
        return []
    
    async def _behavioral_clustering_detection(self, behavioral_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clustering-based anomaly detection (simplified)"""
        # This would use clustering algorithms to identify outlier behaviors
        # For now, return empty list as placeholder
        return []