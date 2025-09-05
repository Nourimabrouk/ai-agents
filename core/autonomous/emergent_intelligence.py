"""
Emergent Intelligence Orchestrator - Phase 7
Discovers and cultivates breakthrough capabilities through network analysis
Enables agents to discover new capabilities beyond their original programming
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, Counter
import statistics
import hashlib
import random

# Optional networkx import
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

# Import base components
from templates.base_agent import BaseAgent, Action, Observation
from core.orchestration.orchestrator import AgentOrchestrator, Task
from .safety import AutonomousSafetyFramework, SafetyAssessment

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class EmergenceType(Enum):
    """Types of emergent intelligence phenomena"""
    CAPABILITY_SYNTHESIS = "capability_synthesis"      # Combining existing capabilities
    NOVEL_STRATEGY = "novel_strategy"                  # New problem-solving strategies
    COLLABORATIVE_PATTERN = "collaborative_pattern"    # New collaboration methods
    OPTIMIZATION_BREAKTHROUGH = "optimization_breakthrough"  # Performance breakthroughs
    CROSS_DOMAIN_TRANSFER = "cross_domain_transfer"    # Knowledge transfer between domains
    EMERGENT_SPECIALIZATION = "emergent_specialization"  # Spontaneous role specialization
    COLLECTIVE_INTELLIGENCE = "collective_intelligence"  # Group intelligence phenomena


@dataclass
class EmergentCapability:
    """Represents a discovered emergent capability"""
    capability_id: str
    name: str
    description: str
    emergence_type: EmergenceType
    discovery_agents: List[str]
    implementation_pattern: Dict[str, Any]
    novelty_score: float
    potential_impact: float
    validation_results: Dict[str, Any]
    reproducibility_score: float = 0.0
    cultivation_status: str = "discovered"  # discovered, tested, validated, deployed
    discovery_timestamp: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0


@dataclass
class BreakthroughBehavior:
    """Represents a breakthrough behavior pattern"""
    behavior_id: str
    pattern_description: str
    triggering_conditions: Dict[str, Any]
    participating_agents: List[str]
    performance_improvement: float
    confidence_score: float
    reproducibility_evidence: List[Dict[str, Any]]
    potential_for_generalization: float
    discovered_at: datetime = field(default_factory=datetime.now)


@dataclass
class InnovationExperiment:
    """Represents an experimental innovation attempt"""
    experiment_id: str
    hypothesis: str
    experimental_setup: Dict[str, Any]
    target_agents: List[str]
    expected_outcomes: List[str]
    actual_outcomes: List[Dict[str, Any]]
    success_metrics: Dict[str, float]
    safety_assessment: SafetyAssessment
    lessons_learned: List[str]
    status: str = "planned"  # planned, running, completed, failed
    created_at: datetime = field(default_factory=datetime.now)


class NetworkAnalyzer:
    """Handles network analysis for capability discovery."""
    
    def __init__(self):
        if HAS_NETWORKX:
            self.interaction_network = nx.DiGraph()
        else:
            self.interaction_network = {'nodes': set(), 'edges': {}}
    
    async def update_interaction_network(self, 
                                       agents: Dict[str, BaseAgent],
                                       orchestrator: AgentOrchestrator,
                                       time_window_hours: int) -> None:
        """Update network with recent agent interactions."""
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Add agents as nodes
        for agent_name, agent in agents.items():
            if not self.interaction_network.has_node(agent_name):
                self.interaction_network.add_node(agent_name, 
                                                 agent_type=type(agent).__name__,
                                                 capabilities=getattr(agent, 'capabilities', []))
        
        # Add interactions from orchestrator history
        await self._add_orchestrator_interactions(orchestrator, cutoff_time)
        
        # Add collaboration edges from agent memories
        await self._add_collaboration_edges(agents)
    
    async def analyze_network_patterns(self) -> Dict[str, Any]:
        """Analyze network structure for emergent patterns."""
        if len(self.interaction_network.nodes) < 2:
            return {}
        
        patterns = {}
        
        # Community detection
        try:
            if HAS_NETWORKX:
                communities = list(nx.community.greedy_modularity_communities(
                    self.interaction_network.to_undirected()
                ))
                patterns['communities'] = [list(community) for community in communities]
        except:
            patterns['communities'] = []
        
        if HAS_NETWORKX:
            # Centrality measures
            patterns['betweenness_centrality'] = nx.betweenness_centrality(self.interaction_network)
            patterns['closeness_centrality'] = nx.closeness_centrality(self.interaction_network)
            patterns['pagerank'] = nx.pagerank(self.interaction_network)
            
            # Structural patterns
            patterns['clustering_coefficient'] = nx.average_clustering(
                self.interaction_network.to_undirected()
            )
            patterns['density'] = nx.density(self.interaction_network)
            patterns['strong_components'] = list(nx.strongly_connected_components(self.interaction_network))
        
        return patterns
    
    async def _add_orchestrator_interactions(self, orchestrator: AgentOrchestrator, cutoff_time: datetime):
        """Add interactions from orchestrator history."""
        if hasattr(orchestrator, 'competitive_history'):
            for competition in orchestrator.competitive_history:
                competition_time = datetime.now()  # Simplified - would track actual time
                if competition_time >= cutoff_time:
                    agent_names = [result.agent_name for result in competition]
                    for i, agent_a in enumerate(agent_names):
                        for j, agent_b in enumerate(agent_names):
                            if i != j:
                                self._add_interaction_edge(agent_a, agent_b, 'competition')
    
    async def _add_collaboration_edges(self, agents: Dict[str, BaseAgent]):
        """Add collaboration edges from agent memories."""
        for agent_name, agent in agents.items():
            if hasattr(agent.memory, 'episodic_memory'):
                for observation in agent.memory.episodic_memory:
                    if 'collaboration' in str(observation.action.action_type):
                        partners = self._extract_collaboration_partners(observation)
                        for partner in partners:
                            if partner in agents:
                                self._add_interaction_edge(agent_name, partner, 'collaboration')
    
    def _add_interaction_edge(self, agent_a: str, agent_b: str, interaction_type: str):
        """Add interaction edge between agents."""
        if HAS_NETWORKX:
            if self.interaction_network.has_edge(agent_a, agent_b):
                edge_data = self.interaction_network[agent_a][agent_b]
                edge_data['weight'] = edge_data.get('weight', 0) + 1
                edge_data[f'{interaction_type}_count'] = edge_data.get(f'{interaction_type}_count', 0) + 1
            else:
                self.interaction_network.add_edge(agent_a, agent_b, 
                                                 weight=1, 
                                                 interaction_type=interaction_type)
    
    def _extract_collaboration_partners(self, observation: Observation) -> List[str]:
        """Extract collaboration partners from observation."""
        partners = []
        
        # Look for partner information in action parameters
        if hasattr(observation.action, 'parameters'):
            partners_data = observation.action.parameters.get('collaboration_partners', [])
            if isinstance(partners_data, list):
                partners.extend(partners_data)
            elif isinstance(partners_data, str):
                partners.append(partners_data)
        
        # Look for collaboration info in result
        if isinstance(observation.result, dict):
            collaboration_info = observation.result.get('collaborated_with', [])
            if isinstance(collaboration_info, list):
                partners.extend(collaboration_info)
        
        return partners


class CapabilityDetector:
    """Detects different types of emergent capabilities."""
    
    def __init__(self, novelty_threshold: float = 0.7, minimum_observations: int = 5):
        self.novelty_threshold = novelty_threshold
        self.minimum_observations = minimum_observations
    
    async def detect_capability_synthesis(self, 
                                        agents: Dict[str, BaseAgent],
                                        network_patterns: Dict[str, Any]) -> List[EmergentCapability]:
        """Detect synthesis of existing capabilities into new ones."""
        capabilities = []
        
        # Look for agents that frequently interact and have complementary capabilities
        for community in network_patterns.get('communities', []):
            if len(community) >= 2:
                community_agents = [agents[name] for name in community if name in agents]
                
                # Extract capabilities from each agent
                agent_capabilities = {}
                for agent in community_agents:
                    agent_capabilities[agent.name] = await self._extract_agent_capabilities(agent)
                
                # Find novel capability combinations
                combinations = await self._find_capability_combinations(agent_capabilities)
                
                for combination in combinations:
                    if combination['novelty_score'] >= self.novelty_threshold:
                        capability = EmergentCapability(
                            capability_id=self._generate_capability_id(combination),
                            name=combination['name'],
                            description=combination['description'],
                            emergence_type=EmergenceType.CAPABILITY_SYNTHESIS,
                            discovery_agents=community,
                            implementation_pattern=combination['pattern'],
                            novelty_score=combination['novelty_score'],
                            potential_impact=combination.get('impact_score', 0.5),
                            validation_results={}
                        )
                        capabilities.append(capability)
        
        return capabilities
    
    async def detect_novel_strategies(self, agents: Dict[str, BaseAgent]) -> List[EmergentCapability]:
        """Detect novel problem-solving strategies."""
        capabilities = []
        
        for agent_name, agent in agents.items():
            if hasattr(agent.memory, 'episodic_memory'):
                # Analyze successful strategies that differ from known patterns
                successful_strategies = [
                    obs for obs in agent.memory.episodic_memory 
                    if obs.success and obs.action.action_type not in ['direct', 'analytical', 'collaborative']
                ]
                
                if len(successful_strategies) >= self.minimum_observations:
                    strategy_pattern = await self._analyze_strategy_pattern(successful_strategies)
                    
                    if strategy_pattern['novelty_score'] >= self.novelty_threshold:
                        capability = EmergentCapability(
                            capability_id=self._generate_capability_id(strategy_pattern),
                            name=f"Novel Strategy: {strategy_pattern['name']}",
                            description=strategy_pattern['description'],
                            emergence_type=EmergenceType.NOVEL_STRATEGY,
                            discovery_agents=[agent_name],
                            implementation_pattern=strategy_pattern,
                            novelty_score=strategy_pattern['novelty_score'],
                            potential_impact=strategy_pattern.get('impact_score', 0.5),
                            validation_results={}
                        )
                        capabilities.append(capability)
        
        return capabilities
    
    async def _extract_agent_capabilities(self, agent: BaseAgent) -> List[str]:
        """Extract capabilities from agent."""
        capabilities = []
        
        # Extract from tools
        if hasattr(agent, 'tools'):
            capabilities.extend([tool.__name__ for tool in agent.tools])
        
        # Extract from successful task patterns
        if hasattr(agent.memory, 'episodic_memory'):
            successful_tasks = [obs for obs in agent.memory.episodic_memory if obs.success]
            task_types = [obs.action.action_type for obs in successful_tasks]
            capabilities.extend(list(set(task_types)))
        
        return capabilities
    
    async def _find_capability_combinations(self, agent_capabilities: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Find novel capability combinations."""
        combinations = []
        
        # Simple combination analysis
        all_capabilities = set()
        for caps in agent_capabilities.values():
            all_capabilities.update(caps)
        
        capability_list = list(all_capabilities)
        
        # Generate combinations
        for i in range(len(capability_list)):
            for j in range(i + 1, len(capability_list)):
                cap_a, cap_b = capability_list[i], capability_list[j]
                
                # Check if this combination appears frequently
                combination_count = sum(
                    1 for agent_name, caps in agent_capabilities.items()
                    if cap_a in caps and cap_b in caps
                )
                
                if combination_count >= 2:  # Appears in at least 2 agents
                    combinations.append({
                        'name': f"Combined {cap_a} and {cap_b}",
                        'description': f"Synthesis of {cap_a} and {cap_b} capabilities",
                        'pattern': {
                            'capabilities': [cap_a, cap_b],
                            'combination_strength': combination_count / len(agent_capabilities)
                        },
                        'novelty_score': min(1.0, combination_count / len(agent_capabilities) + 0.3),
                        'impact_score': 0.6
                    })
        
        return combinations
    
    async def _analyze_strategy_pattern(self, successful_strategies: List[Observation]) -> Dict[str, Any]:
        """Analyze strategy patterns for novelty."""
        if not successful_strategies:
            return {'novelty_score': 0.0}
        
        # Analyze action types and parameters
        action_types = [obs.action.action_type for obs in successful_strategies]
        unique_types = set(action_types)
        
        # Calculate novelty based on uniqueness and success pattern
        novelty_score = 0.5  # Base score
        
        # Bonus for unique action types
        if len(unique_types) > 2:
            novelty_score += 0.2
        
        # Bonus for consistent success
        success_rate = sum(1 for obs in successful_strategies if obs.success) / len(successful_strategies)
        novelty_score += success_rate * 0.3
        
        return {
            'name': f"Strategy Pattern ({', '.join(list(unique_types)[:2])})",
            'description': f"Novel strategy using {len(unique_types)} different approaches",
            'novelty_score': min(1.0, novelty_score),
            'impact_score': success_rate,
            'pattern_details': {
                'action_types': list(unique_types),
                'success_rate': success_rate,
                'sample_size': len(successful_strategies)
            }
        }
    
    def _generate_capability_id(self, pattern: Dict[str, Any]) -> str:
        """Generate unique capability ID."""
        content = json.dumps(pattern, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]


class CapabilityMiningEngine:
    """Orchestrates capability mining using network analysis and capability detection."""
    
    def __init__(self):
        self.network_analyzer = NetworkAnalyzer()
        self.capability_detector = CapabilityDetector()
        self.capability_cache: Dict[str, EmergentCapability] = {}
        self.mining_history: List[Dict[str, Any]] = []
        
        # Mining parameters
        self.significance_threshold = 0.6
        
    async def mine_emergent_capabilities(self, 
                                       agents: Dict[str, BaseAgent],
                                       orchestrator: AgentOrchestrator,
                                       time_window_hours: int = 24) -> List[EmergentCapability]:
        """Mine emergent capabilities from agent network."""
        logger.info(f"Mining emergent capabilities across {len(agents)} agents")
        global_metrics.incr("emergent.mining.started")
        
        # Update interaction network
        await self.network_analyzer.update_interaction_network(agents, orchestrator, time_window_hours)
        
        # Analyze network patterns
        network_patterns = await self.network_analyzer.analyze_network_patterns()
        
        # Detect various types of capabilities
        capabilities = await self._detect_all_capability_types(agents, network_patterns)
        
        # Filter for novelty and significance
        filtered_capabilities = await self._filter_capabilities(capabilities)
        
        # Cache discovered capabilities
        for capability in filtered_capabilities:
            self.capability_cache[capability.capability_id] = capability
        
        logger.info(f"Discovered {len(filtered_capabilities)} emergent capabilities")
        global_metrics.incr("emergent.mining.completed", len(filtered_capabilities))
        
        return filtered_capabilities
    
    async def _detect_all_capability_types(self, 
                                          agents: Dict[str, BaseAgent], 
                                          network_patterns: Dict[str, Any]) -> List[EmergentCapability]:
        """Detect all types of emergent capabilities."""
        capabilities = []
        
        # Detect capability synthesis
        synthesis_capabilities = await self.capability_detector.detect_capability_synthesis(agents, network_patterns)
        capabilities.extend(synthesis_capabilities)
        
        # Detect novel strategies
        strategy_capabilities = await self.capability_detector.detect_novel_strategies(agents)
        capabilities.extend(strategy_capabilities)
        
        # Detect collaborative patterns
        collaboration_capabilities = await self._detect_collaborative_patterns(network_patterns)
        capabilities.extend(collaboration_capabilities)
        
        # Detect optimization breakthroughs
        optimization_capabilities = await self._detect_optimization_breakthroughs(agents)
        capabilities.extend(optimization_capabilities)
        
        # Detect cross-domain transfers
        transfer_capabilities = await self._detect_cross_domain_transfers(agents)
        capabilities.extend(transfer_capabilities)
        
        return capabilities
    
    # Network interaction update moved to NetworkAnalyzer
    
    # Network pattern analysis moved to NetworkAnalyzer
    
    # Capability synthesis detection moved to CapabilityDetector
    
    # Novel strategy detection moved to CapabilityDetector
    
    async def _detect_collaborative_patterns(self, network_patterns: Dict[str, Any]) -> List[EmergentCapability]:
        """Detect novel collaboration patterns"""
        capabilities = []
        
        # Analyze collaboration networks for emergent patterns
        strong_components = network_patterns.get('strong_components', [])
        
        for component in strong_components:
            if len(component) >= 3:  # Minimum for interesting collaboration
                # Analyze collaboration efficiency and novel patterns
                collaboration_analysis = await self._analyze_collaboration_pattern(list(component))
                
                if collaboration_analysis['novelty_score'] >= self.novelty_threshold:
                    capability = EmergentCapability(
                        capability_id=self._generate_capability_id(collaboration_analysis),
                        name=f"Collaborative Pattern: {collaboration_analysis['name']}",
                        description=collaboration_analysis['description'],
                        emergence_type=EmergenceType.COLLABORATIVE_PATTERN,
                        discovery_agents=list(component),
                        implementation_pattern=collaboration_analysis,
                        novelty_score=collaboration_analysis['novelty_score'],
                        potential_impact=collaboration_analysis.get('impact_score', 0.5),
                        validation_results={}
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    async def _detect_optimization_breakthroughs(self, agents: Dict[str, BaseAgent]) -> List[EmergentCapability]:
        """Detect performance optimization breakthroughs"""
        capabilities = []
        
        for agent_name, agent in agents.items():
            # Analyze performance trends for breakthroughs
            if hasattr(agent, 'performance_history') and len(agent.performance_history) >= 10:
                breakthrough = await self._analyze_performance_breakthrough(agent.performance_history)
                
                if breakthrough and breakthrough['significance'] >= self.significance_threshold:
                    capability = EmergentCapability(
                        capability_id=self._generate_capability_id(breakthrough),
                        name=f"Performance Breakthrough: {breakthrough['name']}",
                        description=breakthrough['description'],
                        emergence_type=EmergenceType.OPTIMIZATION_BREAKTHROUGH,
                        discovery_agents=[agent_name],
                        implementation_pattern=breakthrough,
                        novelty_score=breakthrough.get('novelty_score', 0.8),
                        potential_impact=breakthrough['significance'],
                        validation_results={}
                    )
                    capabilities.append(capability)
        
        return capabilities
    
    async def _detect_cross_domain_transfers(self, agents: Dict[str, BaseAgent]) -> List[EmergentCapability]:
        """Detect knowledge transfer between domains"""
        capabilities = []
        
        # Group agents by domain/specialization
        domain_agents = defaultdict(list)
        for agent_name, agent in agents.items():
            # Extract domain from agent's task history
            domain = await self._identify_agent_domain(agent)
            domain_agents[domain].append((agent_name, agent))
        
        # Look for cross-domain knowledge transfer
        for domain_a, agents_a in domain_agents.items():
            for domain_b, agents_b in domain_agents.items():
                if domain_a != domain_b:
                    transfer_patterns = await self._detect_knowledge_transfer(
                        agents_a, agents_b, domain_a, domain_b
                    )
                    
                    for pattern in transfer_patterns:
                        if pattern['novelty_score'] >= self.novelty_threshold:
                            capability = EmergentCapability(
                                capability_id=self._generate_capability_id(pattern),
                                name=f"Cross-Domain Transfer: {domain_a} → {domain_b}",
                                description=pattern['description'],
                                emergence_type=EmergenceType.CROSS_DOMAIN_TRANSFER,
                                discovery_agents=pattern['agents'],
                                implementation_pattern=pattern,
                                novelty_score=pattern['novelty_score'],
                                potential_impact=pattern.get('impact_score', 0.5),
                                validation_results={}
                            )
                            capabilities.append(capability)
        
        return capabilities
    
    def _generate_capability_id(self, pattern: Dict[str, Any]) -> str:
        """Generate unique capability ID"""
        content = json.dumps(pattern, sort_keys=True, default=str)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    # Helper methods moved to CapabilityDetector and NetworkAnalyzer
    
    async def _analyze_collaboration_pattern(self, component_agents: List[str]) -> Dict[str, Any]:
        """Analyze collaboration patterns for novelty"""
        # Simple collaboration analysis
        return {
            'name': f"Multi-Agent Collaboration ({len(component_agents)} agents)",
            'description': f"Novel collaboration pattern involving {len(component_agents)} agents",
            'novelty_score': min(1.0, 0.5 + len(component_agents) * 0.1),
            'impact_score': 0.6,
            'collaboration_details': {
                'participant_count': len(component_agents),
                'participants': component_agents
            }
        }
    
    async def _analyze_performance_breakthrough(self, performance_history: List[float]) -> Optional[Dict[str, Any]]:
        """Analyze performance data for breakthroughs"""
        if len(performance_history) < 10:
            return {}
        
        # Look for sudden performance jumps
        recent_performance = performance_history[-5:]
        older_performance = performance_history[-15:-5]
        
        recent_avg = sum(recent_performance) / len(recent_performance)
        older_avg = sum(older_performance) / len(older_performance)
        
        improvement = recent_avg - older_avg
        
        if improvement >= 0.2:  # 20% improvement threshold
            return {
                'name': 'Performance Breakthrough',
                'description': f'Sudden performance improvement of {improvement:.1%}',
                'significance': improvement,
                'novelty_score': min(1.0, improvement * 2),
                'breakthrough_details': {
                    'improvement': improvement,
                    'recent_avg': recent_avg,
                    'older_avg': older_avg
                }
            }
        
        return {}
    
    async def _identify_agent_domain(self, agent: BaseAgent) -> str:
        """Identify agent's primary domain from task history"""
        if not hasattr(agent.memory, 'episodic_memory'):
            return 'general'
        
        # Analyze task types to determine domain
        task_types = []
        for obs in agent.memory.episodic_memory:
            action_text = f"{obs.action.action_type} {obs.action.expected_outcome}".lower()
            
            if 'invoice' in action_text or 'financial' in action_text:
                task_types.append('finance')
            elif 'data' in action_text or 'analyze' in action_text:
                task_types.append('analytics')
            elif 'code' in action_text or 'program' in action_text:
                task_types.append('development')
            else:
                task_types.append('general')
        
        if not task_types:
            return 'general'
        
        # Return most common domain
        domain_counts = Counter(task_types)
        return domain_counts.most_common(1)[0][0]
    
    async def _detect_knowledge_transfer(self, agents_a: List[Tuple[str, BaseAgent]], 
                                       agents_b: List[Tuple[str, BaseAgent]], 
                                       domain_a: str, domain_b: str) -> List[Dict[str, Any]]:
        """Detect knowledge transfer between domains"""
        transfer_patterns = []
        
        # Simple transfer detection - look for similar successful patterns across domains
        patterns_a = {}
        patterns_b = {}
        
        # Extract patterns from domain A
        for agent_name, agent in agents_a:
            if hasattr(agent.memory, 'episodic_memory'):
                successful_actions = [obs.action.action_type for obs in agent.memory.episodic_memory if obs.success]
                patterns_a[agent_name] = Counter(successful_actions)
        
        # Extract patterns from domain B  
        for agent_name, agent in agents_b:
            if hasattr(agent.memory, 'episodic_memory'):
                successful_actions = [obs.action.action_type for obs in agent.memory.episodic_memory if obs.success]
                patterns_b[agent_name] = Counter(successful_actions)
        
        # Look for common patterns
        for agent_a_name, patterns_a_dict in patterns_a.items():
            for agent_b_name, patterns_b_dict in patterns_b.items():
                common_patterns = set(patterns_a_dict.keys()) & set(patterns_b_dict.keys())
                
                if common_patterns and len(common_patterns) >= 2:
                    transfer_patterns.append({
                        'description': f"Knowledge transfer from {domain_a} to {domain_b}",
                        'novelty_score': min(1.0, 0.6 + len(common_patterns) * 0.1),
                        'impact_score': 0.5,
                        'agents': [agent_a_name, agent_b_name],
                        'transferred_patterns': list(common_patterns),
                        'transfer_strength': len(common_patterns) / max(len(patterns_a_dict), len(patterns_b_dict))
                    })
        
        return transfer_patterns
    
    async def _filter_capabilities(self, capabilities: List[EmergentCapability]) -> List[EmergentCapability]:
        """Filter capabilities for novelty and significance"""
        filtered = []
        
        for capability in capabilities:
            # Filter by novelty threshold
            if capability.novelty_score >= self.novelty_threshold:
                # Filter by significance threshold  
                if capability.potential_impact >= self.significance_threshold:
                    filtered.append(capability)
        
        # Sort by combined score
        filtered.sort(key=lambda c: c.novelty_score * c.potential_impact, reverse=True)
        
        return filtered


class BreakthroughAnalyzer:
    """Analyzes individual agents for breakthrough patterns."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity
    
    async def analyze_agent_breakthroughs(self, 
                                        agent: BaseAgent,
                                        cutoff_time: datetime) -> List[BreakthroughBehavior]:
        """Analyze individual agent for breakthrough patterns."""
        breakthroughs = []
        
        if not hasattr(agent.memory, 'episodic_memory'):
            return breakthroughs
        
        # Get recent observations
        recent_obs = [
            obs for obs in agent.memory.episodic_memory 
            if obs.timestamp >= cutoff_time
        ]
        
        if len(recent_obs) < 5:
            return breakthroughs
        
        # Analyze performance anomalies
        performance_breakthrough = await self._detect_performance_anomaly(agent, recent_obs)
        if performance_breakthrough:
            breakthroughs.append(performance_breakthrough)
        
        # Analyze strategy innovation
        strategy_breakthrough = await self._detect_strategy_innovation(agent, recent_obs)
        if strategy_breakthrough:
            breakthroughs.append(strategy_breakthrough)
        
        # Analyze learning acceleration
        learning_breakthrough = await self._detect_learning_acceleration(agent, recent_obs)
        if learning_breakthrough:
            breakthroughs.append(learning_breakthrough)
        
        return breakthroughs
    
    async def _detect_performance_anomaly(self, 
                                        agent: BaseAgent,
                                        recent_observations: List[Observation]) -> Optional[BreakthroughBehavior]:
        """Detect anomalous performance improvements."""
        # Calculate recent success rate
        recent_successes = [obs for obs in recent_observations if obs.success]
        recent_success_rate = len(recent_successes) / len(recent_observations)
        
        # Compare with historical baseline
        baseline_success_rate = agent.get_success_rate()
        improvement = recent_success_rate - baseline_success_rate
        
        # Significant improvement threshold
        if improvement >= 0.3:  # 30% improvement
            return BreakthroughBehavior(
                behavior_id=f"perf_{agent.name}_{int(datetime.now().timestamp())}",
                pattern_description=f"Significant performance improvement: {improvement:.2%}",
                triggering_conditions={
                    'baseline_success_rate': baseline_success_rate,
                    'recent_success_rate': recent_success_rate,
                    'observation_count': len(recent_observations)
                },
                participating_agents=[agent.name],
                performance_improvement=improvement,
                confidence_score=min(1.0, improvement * 2),
                reproducibility_evidence=[{
                    'timestamp': obs.timestamp.isoformat(),
                    'success': obs.success,
                    'action_type': obs.action.action_type
                } for obs in recent_observations],
                potential_for_generalization=0.7
            )
        
        return {}
    
    async def _detect_strategy_innovation(self, agent: BaseAgent, recent_obs: List[Observation]) -> Optional[BreakthroughBehavior]:
        """Detect innovative strategy usage."""
        # Placeholder for strategy innovation detection
        return {}
    
    async def _detect_learning_acceleration(self, agent: BaseAgent, recent_obs: List[Observation]) -> Optional[BreakthroughBehavior]:
        """Detect learning acceleration patterns."""
        # Placeholder for learning acceleration detection
        return {}


class CollectiveAnalyzer:
    """Analyzes collective breakthrough patterns across multiple agents."""
    
    async def analyze_collective_breakthroughs(self,
                                             agents: Dict[str, BaseAgent],
                                             cutoff_time: datetime) -> List[BreakthroughBehavior]:
        """Analyze collective breakthrough patterns."""
        breakthroughs = []
        
        # Analyze synchronized improvements
        sync_breakthrough = await self._detect_synchronized_improvement(agents, cutoff_time)
        if sync_breakthrough:
            breakthroughs.append(sync_breakthrough)
        
        # Analyze emergent specialization
        specialization_breakthrough = await self._detect_emergent_specialization(agents, cutoff_time)
        if specialization_breakthrough:
            breakthroughs.append(specialization_breakthrough)
        
        return breakthroughs
    
    async def _detect_synchronized_improvement(self,
                                             agents: Dict[str, BaseAgent],
                                             cutoff_time: datetime) -> Optional[BreakthroughBehavior]:
        """Detect synchronized improvement across multiple agents."""
        improvements = {}
        for agent_name, agent in agents.items():
            if hasattr(agent.memory, 'episodic_memory'):
                recent_obs = [obs for obs in agent.memory.episodic_memory if obs.timestamp >= cutoff_time]
                if len(recent_obs) >= 3:
                    recent_success_rate = sum(1 for obs in recent_obs if obs.success) / len(recent_obs)
                    baseline_success_rate = agent.get_success_rate()
                    improvement = recent_success_rate - baseline_success_rate
                    if improvement > 0.1:  # 10% improvement threshold
                        improvements[agent_name] = improvement
        
        # Check for synchronized improvement
        if len(improvements) >= 3:  # At least 3 agents improved simultaneously
            avg_improvement = sum(improvements.values()) / len(improvements)
            
            if avg_improvement >= 0.2:  # 20% average improvement
                return BreakthroughBehavior(
                    behavior_id=f"sync_improve_{int(datetime.now().timestamp())}",
                    pattern_description=f"Synchronized improvement across {len(improvements)} agents",
                    triggering_conditions={
                        'improved_agents': list(improvements.keys()),
                        'average_improvement': avg_improvement,
                        'time_window': cutoff_time.isoformat()
                    },
                    participating_agents=list(improvements.keys()),
                    performance_improvement=avg_improvement,
                    confidence_score=min(1.0, len(improvements) / len(agents)),
                    reproducibility_evidence=[{
                        'agent': agent_name,
                        'improvement': improvement
                    } for agent_name, improvement in improvements.items()],
                    potential_for_generalization=0.8
                )
        
        return {}
    
    async def _detect_emergent_specialization(self, agents: Dict[str, BaseAgent], cutoff_time: datetime) -> Optional[BreakthroughBehavior]:
        """Detect emergent specialization patterns."""
        # Placeholder for specialization detection
        return {}


class NoveltyDetector:
    """Detects novel patterns and breakthrough behaviors in agent operations."""
    
    def __init__(self, sensitivity: float = 0.8):
        self.sensitivity = sensitivity  # Higher = more sensitive to novelty
        self.baseline_patterns: Dict[str, Any] = {}
        self.novelty_history: List[Dict[str, Any]] = []
        
        # Initialize analyzers
        self.breakthrough_analyzer = BreakthroughAnalyzer(sensitivity)
        self.collective_analyzer = CollectiveAnalyzer()
        
    async def detect_breakthrough_behaviors(self, 
                                          agents: Dict[str, BaseAgent],
                                          time_window_hours: int = 12) -> List[BreakthroughBehavior]:
        """Detect breakthrough behaviors in agent operations."""
        logger.info(f"Detecting breakthrough behaviors in {len(agents)} agents")
        
        breakthroughs = []
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        # Analyze individual agent breakthroughs
        for agent_name, agent in agents.items():
            agent_breakthroughs = await self.breakthrough_analyzer.analyze_agent_breakthroughs(
                agent, cutoff_time
            )
            breakthroughs.extend(agent_breakthroughs)
        
        # Analyze collective breakthroughs
        collective_breakthroughs = await self.collective_analyzer.analyze_collective_breakthroughs(
            agents, cutoff_time
        )
        breakthroughs.extend(collective_breakthroughs)
        
        # Filter and rank breakthroughs
        significant_breakthroughs = await self._filter_breakthroughs(breakthroughs)
        
        logger.info(f"Detected {len(significant_breakthroughs)} breakthrough behaviors")
        return significant_breakthroughs
    
    # Individual agent analysis moved to BreakthroughAnalyzer
    
    async def _detect_performance_anomaly(self, 
                                        agent: BaseAgent,
                                        recent_observations: List[Observation]) -> Optional[BreakthroughBehavior]:
        """Detect anomalous performance improvements"""
        
        # Calculate recent success rate
        recent_successes = [obs for obs in recent_observations if obs.success]
        recent_success_rate = len(recent_successes) / len(recent_observations)
        
        # Compare with historical baseline
        baseline_success_rate = agent.get_success_rate()
        
        improvement = recent_success_rate - baseline_success_rate
        
        # Significant improvement threshold
        if improvement >= 0.3:  # 30% improvement
            return BreakthroughBehavior(
                behavior_id=f"perf_{agent.name}_{int(datetime.now().timestamp())}",
                pattern_description=f"Significant performance improvement: {improvement:.2%}",
                triggering_conditions={
                    'baseline_success_rate': baseline_success_rate,
                    'recent_success_rate': recent_success_rate,
                    'observation_count': len(recent_observations)
                },
                participating_agents=[agent.name],
                performance_improvement=improvement,
                confidence_score=min(1.0, improvement * 2),
                reproducibility_evidence=[{
                    'timestamp': obs.timestamp.isoformat(),
                    'success': obs.success,
                    'action_type': obs.action.action_type
                } for obs in recent_observations],
                potential_for_generalization=0.7
            )
        
        return {}
    
    # Collective analysis moved to CollectiveAnalyzer
    
    async def _detect_synchronized_improvement(self,
                                             agents: Dict[str, BaseAgent],
                                             cutoff_time: datetime) -> Optional[BreakthroughBehavior]:
        """Detect synchronized improvement across multiple agents"""
        
        improvements = {}
        for agent_name, agent in agents.items():
            if hasattr(agent.memory, 'episodic_memory'):
                recent_obs = [obs for obs in agent.memory.episodic_memory if obs.timestamp >= cutoff_time]
                if len(recent_obs) >= 3:
                    recent_success_rate = sum(1 for obs in recent_obs if obs.success) / len(recent_obs)
                    baseline_success_rate = agent.get_success_rate()
                    improvement = recent_success_rate - baseline_success_rate
                    if improvement > 0.1:  # 10% improvement threshold
                        improvements[agent_name] = improvement
        
        # Check for synchronized improvement
        if len(improvements) >= 3:  # At least 3 agents improved simultaneously
            avg_improvement = sum(improvements.values()) / len(improvements)
            
            if avg_improvement >= 0.2:  # 20% average improvement
                return BreakthroughBehavior(
                    behavior_id=f"sync_improve_{int(datetime.now().timestamp())}",
                    pattern_description=f"Synchronized improvement across {len(improvements)} agents",
                    triggering_conditions={
                        'improved_agents': list(improvements.keys()),
                        'average_improvement': avg_improvement,
                        'time_window': cutoff_time.isoformat()
                    },
                    participating_agents=list(improvements.keys()),
                    performance_improvement=avg_improvement,
                    confidence_score=min(1.0, len(improvements) / len(agents)),
                    reproducibility_evidence=[{
                        'agent': agent_name,
                        'improvement': improvement
                    } for agent_name, improvement in improvements.items()],
                    potential_for_generalization=0.8
                )
        
        return {}


class InnovationIncubator:
    """
    Safe environment for testing and cultivating emergent capabilities
    Provides controlled experimentation with rollback capabilities
    """
    
    def __init__(self, safety_framework: AutonomousSafetyFramework):
        self.safety_framework = safety_framework
        self.active_experiments: Dict[str, InnovationExperiment] = {}
        self.experiment_history: List[InnovationExperiment] = []
        self.success_rate = 0.0
        
    async def cultivate_capability(self, 
                                 capability: EmergentCapability,
                                 test_agents: List[BaseAgent]) -> Dict[str, Any]:
        """Cultivate an emergent capability through safe experimentation"""
        logger.info(f"Cultivating capability: {capability.name}")
        
        # Design experiment
        experiment = await self._design_cultivation_experiment(capability, test_agents)
        
        # Safety validation
        safety_check = await self.safety_framework.validate_capability({
            'name': capability.name,
            'implementation': str(capability.implementation_pattern),
            'discovery_method': capability.emergence_type.value
        })
        
        if not safety_check.is_safe:
            return {
                'success': False,
                'reason': 'safety_violation',
                'violations': [v.description for v in safety_check.violations]
            }
        
        experiment.safety_assessment = safety_check
        
        # Create backups
        backup_ids = {}
        for agent in test_agents:
            backup_id = await self.safety_framework.create_safe_backup(agent)
            backup_ids[agent.name] = backup_id
        
        try:
            # Run experiment
            experiment.status = "running"
            self.active_experiments[experiment.experiment_id] = experiment
            
            results = await self._execute_cultivation_experiment(experiment, test_agents)
            
            experiment.actual_outcomes = results['outcomes']
            experiment.success_metrics = results['metrics']
            experiment.status = "completed"
            
            # Evaluate success
            cultivation_success = await self._evaluate_cultivation_success(experiment)
            
            if cultivation_success['success']:
                capability.cultivation_status = "validated"
                capability.reproducibility_score = cultivation_success['reproducibility']
                
                return {
                    'success': True,
                    'capability_validated': True,
                    'reproducibility_score': cultivation_success['reproducibility'],
                    'performance_improvement': cultivation_success['improvement'],
                    'experiment_results': results
                }
            else:
                # Rollback changes
                for agent in test_agents:
                    await self.safety_framework.emergency_rollback(backup_ids[agent.name], agent)
                
                return {
                    'success': False,
                    'reason': 'cultivation_failed',
                    'issues': cultivation_success['issues']
                }
        
        except Exception as e:
            logger.error(f"Cultivation experiment failed: {e}")
            experiment.status = "failed"
            
            # Emergency rollback
            for agent in test_agents:
                await self.safety_framework.emergency_rollback(backup_ids[agent.name], agent)
            
            return {
                'success': False,
                'reason': 'experiment_error',
                'error': str(e)
            }
        
        finally:
            # Clean up
            if experiment.experiment_id in self.active_experiments:
                del self.active_experiments[experiment.experiment_id]
            self.experiment_history.append(experiment)
            
            # Update success rate
            successful_experiments = sum(1 for exp in self.experiment_history if exp.status == "completed")
            self.success_rate = successful_experiments / len(self.experiment_history)
    
    async def _design_cultivation_experiment(self,
                                           capability: EmergentCapability,
                                           test_agents: List[BaseAgent]) -> InnovationExperiment:
        """Design experiment to test and cultivate capability"""
        
        experiment_id = f"exp_{capability.capability_id}_{int(datetime.now().timestamp())}"
        
        # Generate hypothesis
        hypothesis = f"The emergent capability '{capability.name}' can be reproduced and generalized across agents"
        
        # Design experimental setup
        experimental_setup = {
            'capability_to_test': capability.capability_id,
            'test_duration_minutes': 30,
            'test_tasks': await self._generate_test_tasks(capability),
            'success_criteria': {
                'min_reproducibility': 0.7,
                'min_performance_improvement': 0.15,
                'max_safety_violations': 0
            },
            'control_group_size': len(test_agents) // 2,
            'treatment_group_size': len(test_agents) - (len(test_agents) // 2)
        }
        
        # Expected outcomes
        expected_outcomes = [
            f"Agents can reproduce the {capability.emergence_type.value} pattern",
            f"Performance improvement of at least 15%",
            "No safety violations during testing",
            f"Capability generalizes to new tasks"
        ]
        
        return InnovationExperiment(
            experiment_id=experiment_id,
            hypothesis=hypothesis,
            experimental_setup=experimental_setup,
            target_agents=[agent.name for agent in test_agents],
            expected_outcomes=expected_outcomes,
            actual_outcomes=[],
            success_metrics={},
            safety_assessment=SafetyAssessment(
                is_safe=True, confidence=0.0, violations=[], warnings=[], 
                recommendations=[], assessment_time_ms=0.0
            ),
            lessons_learned=[]
        )
    
    async def _execute_cultivation_experiment(self,
                                            experiment: InnovationExperiment,
                                            test_agents: List[BaseAgent]) -> Dict[str, Any]:
        """Execute the cultivation experiment"""
        
        outcomes = []
        metrics = {}
        
        # Split agents into control and treatment groups
        control_group = test_agents[:experiment.experimental_setup['control_group_size']]
        treatment_group = test_agents[experiment.experimental_setup['control_group_size']:]
        
        # Run control group (without capability)
        control_results = await self._run_agent_group_test(
            control_group, 
            experiment.experimental_setup['test_tasks'],
            apply_capability=False
        )
        
        # Run treatment group (with capability)
        treatment_results = await self._run_agent_group_test(
            treatment_group,
            experiment.experimental_setup['test_tasks'],
            apply_capability=True,
            capability_pattern=experiment.experimental_setup.get('capability_to_test')
        )
        
        # Calculate metrics
        metrics['control_success_rate'] = control_results['success_rate']
        metrics['treatment_success_rate'] = treatment_results['success_rate']
        metrics['improvement_ratio'] = treatment_results['success_rate'] / max(0.01, control_results['success_rate'])
        metrics['absolute_improvement'] = treatment_results['success_rate'] - control_results['success_rate']
        
        outcomes.append({
            'type': 'performance_comparison',
            'control_performance': control_results,
            'treatment_performance': treatment_results,
            'improvement': metrics['absolute_improvement']
        })
        
        return {
            'outcomes': outcomes,
            'metrics': metrics
        }
    
    async def _run_agent_group_test(self,
                                  agents: List[BaseAgent],
                                  test_tasks: List[Dict[str, Any]],
                                  apply_capability: bool = False,
                                  capability_pattern: Optional[str] = None) -> Dict[str, Any]:
        """Run test tasks on a group of agents"""
        
        results = []
        
        for agent in agents:
            agent_results = []
            
            for task in test_tasks:
                try:
                    # Apply capability modification if needed
                    if apply_capability and capability_pattern:
                        # This would apply the capability pattern to the agent
                        # For now, just simulate the effect
                        pass
                    
                    # Run task
                    result = await agent.process_task(
                        task['description'],
                        task.get('context', {})
                    )
                    
                    success = result is not None and 'error' not in str(result).lower()
                    agent_results.append({
                        'task_id': task['id'],
                        'success': success,
                        'result': result
                    })
                    
                except Exception as e:
                    agent_results.append({
                        'task_id': task['id'],
                        'success': False,
                        'error': str(e)
                    })
            
            results.append({
                'agent_name': agent.name,
                'results': agent_results,
                'success_rate': sum(1 for r in agent_results if r['success']) / len(agent_results)
            })
        
        # Calculate group metrics
        overall_success_rate = sum(r['success_rate'] for r in results) / len(results)
        
        return {
            'agent_results': results,
            'success_rate': overall_success_rate,
            'total_tasks': len(test_tasks) * len(agents),
            'successful_tasks': sum(
                sum(1 for task in r['results'] if task['success']) 
                for r in results
            )
        }
    
    async def _generate_test_tasks(self, capability: EmergentCapability) -> List[Dict[str, Any]]:
        """Generate test tasks for capability validation"""
        base_tasks = [
            {
                'id': 'test_basic',
                'description': f"Apply {capability.name} to solve a basic problem",
                'context': {'difficulty': 'basic', 'domain': 'general'}
            },
            {
                'id': 'test_complex',
                'description': f"Use {capability.name} for a complex multi-step task",
                'context': {'difficulty': 'complex', 'domain': 'general'}
            },
            {
                'id': 'test_novel',
                'description': f"Apply {capability.name} to an unfamiliar problem type",
                'context': {'difficulty': 'medium', 'domain': 'novel'}
            }
        ]
        
        return base_tasks


class EvolutionCoordinator:
    """Coordinates the complete intelligence evolution cycle."""
    
    def __init__(self, 
                 capability_miner: CapabilityMiningEngine,
                 novelty_detector: NoveltyDetector,
                 innovation_incubator: InnovationIncubator):
        self.capability_miner = capability_miner
        self.novelty_detector = novelty_detector
        self.innovation_incubator = innovation_incubator
    
    async def run_evolution_cycle(self,
                                 agents: Dict[str, BaseAgent],
                                 orchestrator: AgentOrchestrator,
                                 discovered_capabilities: Dict[str, EmergentCapability],
                                 should_discover: bool) -> Dict[str, Any]:
        """Run complete intelligence evolution cycle."""
        evolution_results = {
            'capabilities_discovered': 0,
            'breakthroughs_detected': 0,
            'capabilities_cultivated': 0,
            'capabilities_deployed': 0
        }
        
        # 1. Discover emergent capabilities
        new_capabilities = []
        if should_discover:
            new_capabilities = await self.capability_miner.mine_emergent_capabilities(
                agents, orchestrator
            )
            evolution_results['capabilities_discovered'] = len(new_capabilities)
        
        # 2. Detect breakthrough behaviors
        breakthrough_behaviors = await self.novelty_detector.detect_breakthrough_behaviors(agents)
        evolution_results['breakthroughs_detected'] = len(breakthrough_behaviors)
        
        # 3. Cultivate promising capabilities
        cultivation_results = await self._cultivate_capabilities(
            discovered_capabilities, agents
        )
        evolution_results['capabilities_cultivated'] = len([r for r in cultivation_results if r['success']])
        
        return {
            **evolution_results,
            'new_capabilities': new_capabilities,
            'breakthrough_behaviors': breakthrough_behaviors,
            'cultivation_results': cultivation_results
        }
    
    async def _cultivate_capabilities(self, 
                                    discovered_capabilities: Dict[str, EmergentCapability],
                                    agents: Dict[str, BaseAgent]) -> List[Dict[str, Any]]:
        """Cultivate promising capabilities."""
        cultivation_results = []
        promising_capabilities = [
            cap for cap in discovered_capabilities.values()
            if cap.cultivation_status == "discovered" and cap.novelty_score >= 0.7
        ]
        
        for capability in promising_capabilities[:3]:  # Limit concurrent cultivation
            test_agents = await self._select_test_agents(capability, agents)
            
            if test_agents:
                cultivation_result = await self.innovation_incubator.cultivate_capability(
                    capability, test_agents
                )
                cultivation_results.append(cultivation_result)
        
        return cultivation_results
    
    async def _select_test_agents(self, 
                                capability: EmergentCapability,
                                agents: Dict[str, BaseAgent]) -> List[BaseAgent]:
        """Select appropriate agents for testing a capability."""
        test_agents = []
        
        # Prioritize discovery agents first
        for agent_name in capability.discovery_agents:
            if agent_name in agents:
                test_agents.append(agents[agent_name])
        
        # Add additional agents if needed
        remaining_agents = [
            agent for name, agent in agents.items()
            if name not in capability.discovery_agents
        ]
        
        # Select up to 4 total test agents
        while len(test_agents) < 4 and remaining_agents:
            test_agents.append(remaining_agents.pop(0))
        
        return test_agents


class DeploymentManager:
    """Manages deployment of validated capabilities."""
    
    async def deploy_validated_capabilities(self, 
                                          discovered_capabilities: Dict[str, EmergentCapability],
                                          agents: Dict[str, BaseAgent]) -> Dict[str, Any]:
        """Deploy validated capabilities to production agents."""
        deployment_results = {
            'deployed_count': 0,
            'deployment_details': []
        }
        
        # Find capabilities ready for deployment
        ready_capabilities = [
            cap for cap in discovered_capabilities.values()
            if cap.cultivation_status == "validated" and cap.reproducibility_score >= 0.8
        ]
        
        for capability in ready_capabilities:
            deployment_agents = await self._select_deployment_agents(capability, agents)
            
            for agent in deployment_agents:
                deployment_success = await self._deploy_to_agent(capability, agent)
                if deployment_success:
                    deployment_results['deployed_count'] += 1
                    deployment_results['deployment_details'].append({
                        'capability': capability.name,
                        'agent': agent.name,
                        'deployment_time': datetime.now().isoformat()
                    })
        
        return deployment_results
    
    async def _select_deployment_agents(self, 
                                      capability: EmergentCapability,
                                      agents: Dict[str, BaseAgent]) -> List[BaseAgent]:
        """Select agents for capability deployment."""
        # Simple selection based on compatibility
        compatible_agents = []
        
        for agent in agents.values():
            # Check if agent has prerequisite capabilities
            if self._is_compatible(capability, agent):
                compatible_agents.append(agent)
        
        return compatible_agents[:5]  # Limit to 5 agents per deployment
    
    def _is_compatible(self, capability: EmergentCapability, agent: BaseAgent) -> bool:
        """Check if agent is compatible with capability."""
        # Simple compatibility check
        return hasattr(agent, 'tools') and len(agent.tools) > 0
    
    async def _deploy_to_agent(self, capability: EmergentCapability, agent: BaseAgent) -> bool:
        """Deploy capability to specific agent."""
        try:
            # Mark capability as deployed (simplified implementation)
            if not hasattr(agent, 'deployed_capabilities'):
                agent.deployed_capabilities = set()
            
            agent.deployed_capabilities.add(capability.capability_id)
            capability.cultivation_status = "deployed"
            capability.usage_count += 1
            
            logger.info(f"Deployed capability {capability.name} to agent {agent.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy capability {capability.name} to {agent.name}: {e}")
            return False


class EmergentIntelligenceOrchestrator:
    """Main orchestrator for emergent intelligence discovery and cultivation (refactored)."""
    
    def __init__(self, 
                 safety_framework: AutonomousSafetyFramework,
                 discovery_frequency_hours: int = 6):
        self.safety_framework = safety_framework
        self.discovery_frequency = discovery_frequency_hours
        
        # Initialize components
        self.capability_miner = CapabilityMiningEngine()
        self.novelty_detector = NoveltyDetector()
        self.innovation_incubator = InnovationIncubator(safety_framework)
        
        # Initialize coordinators
        self.evolution_coordinator = EvolutionCoordinator(
            self.capability_miner, self.novelty_detector, self.innovation_incubator
        )
        self.deployment_manager = DeploymentManager()
        
        # Discovery tracking
        self.discovered_capabilities: Dict[str, EmergentCapability] = {}
        self.breakthrough_behaviors: List[BreakthroughBehavior] = []
        self.last_discovery_run = datetime.now() - timedelta(hours=discovery_frequency_hours)
        
        # Performance tracking
        self.discovery_rate = 0.0
        self.cultivation_success_rate = 0.0
        self.deployment_rate = 0.0
        
    async def orchestrate_intelligence_evolution(self,
                                               agents: Dict[str, BaseAgent],
                                               orchestrator: AgentOrchestrator) -> Dict[str, Any]:
        """Orchestrate complete intelligence evolution cycle."""
        logger.info("Starting emergent intelligence evolution orchestration")
        global_metrics.incr("emergent.orchestration.started")
        
        try:
            # Run evolution cycle
            should_discover = self._should_run_discovery()
            cycle_results = await self.evolution_coordinator.run_evolution_cycle(
                agents, orchestrator, self.discovered_capabilities, should_discover
            )
            
            # Update internal state
            self._update_internal_state(cycle_results, should_discover)
            
            # Deploy validated capabilities
            deployment_results = await self.deployment_manager.deploy_validated_capabilities(
                self.discovered_capabilities, agents
            )
            
            # Prepare final results
            evolution_results = self._prepare_evolution_results(cycle_results, deployment_results)
            
            logger.info(f"Evolution orchestration completed: {evolution_results}")
            global_metrics.incr("emergent.orchestration.completed")
            
            return evolution_results
            
        except Exception as e:
            logger.error(f"Evolution orchestration failed: {e}")
            global_metrics.incr("emergent.orchestration.failed")
            return self._create_error_results(str(e))
    
    def _update_internal_state(self, cycle_results: Dict[str, Any], should_discover: bool):
        """Update internal state with cycle results."""
        # Update capabilities
        if should_discover:
            for capability in cycle_results.get('new_capabilities', []):
                self.discovered_capabilities[capability.capability_id] = capability
            self.last_discovery_run = datetime.now()
        
        # Update breakthrough behaviors
        self.breakthrough_behaviors.extend(cycle_results.get('breakthrough_behaviors', []))
    
    def _prepare_evolution_results(self, cycle_results: Dict[str, Any], deployment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare final evolution results."""
        return {
            'capabilities_discovered': cycle_results['capabilities_discovered'],
            'breakthroughs_detected': cycle_results['breakthroughs_detected'],
            'capabilities_cultivated': cycle_results['capabilities_cultivated'],
            'capabilities_deployed': deployment_results['deployed_count'],
            'total_discoveries': len(self.discovered_capabilities),
            'evolution_metrics': self._calculate_evolution_metrics_sync()
        }
    
    def _create_error_results(self, error_message: str) -> Dict[str, Any]:
        """Create error results dictionary."""
        return {
            'error': error_message,
            'capabilities_discovered': 0,
            'breakthroughs_detected': 0,
            'capabilities_cultivated': 0,
            'capabilities_deployed': 0,
            'total_discoveries': len(self.discovered_capabilities),
            'evolution_metrics': {}
        }
    
    def _should_run_discovery(self) -> bool:
        """Check if it's time to run capability discovery"""
        time_since_last = datetime.now() - self.last_discovery_run
        return time_since_last.total_seconds() >= self.discovery_frequency * 3600
    
    # Agent selection and deployment methods moved to coordinators
    
    def _calculate_evolution_metrics_sync(self) -> Dict[str, Any]:
        """Calculate evolution metrics synchronously."""
        total_capabilities = len(self.discovered_capabilities)
        deployed_capabilities = len([
            cap for cap in self.discovered_capabilities.values()
            if cap.cultivation_status == "deployed"
        ])
        
        return {
            'total_capabilities': total_capabilities,
            'deployed_capabilities': deployed_capabilities,
            'deployment_rate': deployed_capabilities / total_capabilities if total_capabilities > 0 else 0.0,
            'breakthrough_count': len(self.breakthrough_behaviors),
            'discovery_frequency_hours': self.discovery_frequency
        }
    
    async def _calculate_evolution_metrics(self) -> Dict[str, Any]:
        """Calculate evolution performance metrics"""
        total_capabilities = len(self.discovered_capabilities)
        
        if total_capabilities == 0:
            return {
                'discovery_rate': 0.0,
                'cultivation_success_rate': 0.0,
                'deployment_rate': 0.0,
                'novelty_distribution': {},
                'emergence_type_distribution': {}
            }
        
        # Calculate rates
        cultivated_count = sum(
            1 for cap in self.discovered_capabilities.values()
            if cap.cultivation_status in ["validated", "deployed"]
        )
        deployed_count = sum(
            1 for cap in self.discovered_capabilities.values()
            if cap.cultivation_status == "deployed"
        )
        
        self.cultivation_success_rate = cultivated_count / total_capabilities
        self.deployment_rate = deployed_count / total_capabilities
        
        # Distribution analysis
        novelty_scores = [cap.novelty_score for cap in self.discovered_capabilities.values()]
        emergence_types = [cap.emergence_type.value for cap in self.discovered_capabilities.values()]
        
        return {
            'discovery_rate': self.discovery_rate,
            'cultivation_success_rate': self.cultivation_success_rate,
            'deployment_rate': self.deployment_rate,
            'total_capabilities': total_capabilities,
            'avg_novelty_score': sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0,
            'emergence_type_distribution': dict(Counter(emergence_types)),
            'breakthrough_behaviors_detected': len(self.breakthrough_behaviors)
        }
    
    def get_emergent_intelligence_metrics(self) -> Dict[str, Any]:
        """Get comprehensive emergent intelligence metrics"""
        return {
            'discovered_capabilities': len(self.discovered_capabilities),
            'breakthrough_behaviors': len(self.breakthrough_behaviors),
            'active_experiments': len(self.innovation_incubator.active_experiments),
            'cultivation_success_rate': self.cultivation_success_rate,
            'deployment_rate': self.deployment_rate,
            'last_discovery_run': self.last_discovery_run.isoformat(),
            'novelty_detector_sensitivity': self.novelty_detector.sensitivity,
            'innovation_success_rate': self.innovation_incubator.success_rate,
            'capability_status_distribution': {
                status: sum(1 for cap in self.discovered_capabilities.values() 
                           if cap.cultivation_status == status)
                for status in ["discovered", "tested", "validated", "deployed"]
            }
        }
    
    async def _select_deployment_agents(self, capability: EmergentCapability, agents: Dict[str, BaseAgent]) -> List[BaseAgent]:
        """Select agents for capability deployment"""
        deployment_agents = []
        
        # Prioritize discovery agents
        for agent_name in capability.discovery_agents[:2]:  # Up to 2 discovery agents
            if agent_name in agents:
                deployment_agents.append(agents[agent_name])
        
        # Add compatible agents based on capability type
        for agent_name, agent in agents.items():
            if len(deployment_agents) >= 3:  # Max 3 agents per deployment
                break
                
            if agent_name not in capability.discovery_agents:
                # Check compatibility based on agent's domain
                agent_capabilities = await self.capability_miner._extract_agent_capabilities(agent)
                
                # Simple compatibility check
                if capability.emergence_type.value in ['capability_synthesis', 'novel_strategy']:
                    # These can be deployed to most agents
                    deployment_agents.append(agent)
                elif any(cap in str(capability.implementation_pattern) for cap in agent_capabilities):
                    # Agent has related capabilities
                    deployment_agents.append(agent)
        
        return deployment_agents
    
    async def _apply_capability_to_agent(self, capability: EmergentCapability, agent: BaseAgent) -> bool:
        """Apply capability to agent (simplified implementation)"""
        try:
            # In a real implementation, this would:
            # 1. Modify agent's behavior/code
            # 2. Update agent's tool repertoire  
            # 3. Enhance agent's strategy selection
            # 4. Update agent's memory patterns
            
            # For now, just mark it as applied
            if not hasattr(agent, 'applied_capabilities'):
                agent.applied_capabilities = []
            
            agent.applied_capabilities.append(capability.capability_id)
            
            # Update capability usage
            capability.usage_count += 1
            capability.last_used = datetime.now()
            
            logger.info(f"Applied capability {capability.name} to agent {agent.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply capability {capability.name} to agent {agent.name}: {e}")
            return False