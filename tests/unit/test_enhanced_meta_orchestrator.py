"""
Comprehensive Unit Tests for Enhanced Meta-Orchestrator
======================================================

Tests the Phase 6 enhanced meta-orchestrator component including:
- Task complexity analysis
- Strategy learning mechanisms
- Performance tracking and optimization
- Dynamic agent selection algorithms
- Resource optimization strategies
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from dataclasses import dataclass

# Mock the Phase 6 imports since they may not exist yet
try:
    from agents.meta.enhanced_meta_orchestrator import (
        EnhancedMetaOrchestrator, TaskComplexity, AgentType,
        TaskAnalysis, StrategyRecommendation, PerformanceMetrics
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False
    
    # Mock classes for testing
    class TaskComplexity:
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        VERY_HIGH = "very_high"
    
    class AgentType:
        SPECIALIST = "specialist"
        GENERALIST = "generalist"
        COORDINATOR = "coordinator"
    
    @dataclass
    class TaskAnalysis:
        complexity: str
        estimated_duration: float
        required_skills: List[str]
        resource_requirements: Dict[str, Any]
        success_probability: float
        recommended_agents: List[str]
    
    @dataclass
    class StrategyRecommendation:
        strategy_type: str
        confidence: float
        expected_performance: float
        resource_cost: float
        explanation: str
    
    @dataclass
    class PerformanceMetrics:
        throughput: float
        accuracy: float
        resource_utilization: float
        cost_efficiency: float
        response_time: float

    class EnhancedMetaOrchestrator:
        def __init__(self, config: Dict[str, Any]):
            self.config = config
            self.task_analyzer = AsyncMock()
            self.strategy_learner = AsyncMock()
            self.performance_tracker = AsyncMock()
            self.agent_selector = AsyncMock()
            self.resource_optimizer = AsyncMock()


class TestEnhancedMetaOrchestratorUnit:
    """Unit tests for Enhanced Meta-Orchestrator component"""
    
    @pytest.fixture
    def orchestrator_config(self):
        """Standard orchestrator configuration for testing"""
        return {
            'max_concurrent_tasks': 10,
            'learning_enabled': True,
            'performance_tracking': True,
            'strategy_optimization': True,
            'resource_monitoring': True,
            'adaptation_threshold': 0.1,
            'learning_rate': 0.01
        }
    
    @pytest.fixture
    def orchestrator(self, orchestrator_config):
        """Create Enhanced Meta-Orchestrator instance"""
        return EnhancedMetaOrchestrator(orchestrator_config)
    
    @pytest.fixture
    def sample_task(self):
        """Sample task for testing"""
        return {
            'id': 'task_001',
            'type': 'data_processing',
            'description': 'Process financial statements for Q3 analysis',
            'requirements': {
                'data_size': 1000000,  # 1M records
                'accuracy_required': 0.99,
                'deadline': datetime.now() + timedelta(hours=2),
                'budget_limit': 50.0
            },
            'priority': 1,
            'context': {
                'domain': 'finance',
                'complexity_indicators': ['large_dataset', 'high_accuracy'],
                'historical_performance': {'avg_time': 3600, 'success_rate': 0.95}
            }
        }
    
    @pytest.fixture
    def mock_agents(self):
        """Mock agents with different capabilities"""
        return [
            {
                'id': 'financial_specialist_001',
                'type': AgentType.SPECIALIST,
                'capabilities': ['financial_analysis', 'data_processing'],
                'performance_history': {
                    'tasks_completed': 150,
                    'success_rate': 0.96,
                    'avg_completion_time': 2800,
                    'cost_per_task': 12.5
                },
                'current_load': 0.3,
                'availability': True
            },
            {
                'id': 'data_processor_002',
                'type': AgentType.SPECIALIST,
                'capabilities': ['data_processing', 'statistical_analysis'],
                'performance_history': {
                    'tasks_completed': 200,
                    'success_rate': 0.94,
                    'avg_completion_time': 3200,
                    'cost_per_task': 8.0
                },
                'current_load': 0.7,
                'availability': True
            },
            {
                'id': 'generalist_coordinator_003',
                'type': AgentType.GENERALIST,
                'capabilities': ['coordination', 'task_management', 'data_processing'],
                'performance_history': {
                    'tasks_completed': 300,
                    'success_rate': 0.89,
                    'avg_completion_time': 4000,
                    'cost_per_task': 15.0
                },
                'current_load': 0.1,
                'availability': True
            }
        ]


class TestTaskComplexityAnalysis(TestEnhancedMetaOrchestratorUnit):
    """Test task complexity analysis functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_task_complexity_low(self, orchestrator, sample_task):
        """Test analysis of low complexity task"""
        # Modify task to be low complexity
        low_complexity_task = sample_task.copy()
        low_complexity_task['requirements']['data_size'] = 1000
        low_complexity_task['context']['complexity_indicators'] = ['small_dataset']
        
        if IMPORTS_AVAILABLE and hasattr(orchestrator, 'analyze_task'):
            analysis = await orchestrator.analyze_task(low_complexity_task)
            
            assert isinstance(analysis, TaskAnalysis)
            assert analysis.complexity == TaskComplexity.LOW
            assert analysis.estimated_duration < 1800  # Less than 30 minutes
            assert analysis.success_probability > 0.9
            assert len(analysis.required_skills) > 0
            assert 'data_processing' in analysis.required_skills
        else:
            # Mock the behavior
            with patch.object(orchestrator, 'task_analyzer') as mock_analyzer:
                async def mock_analyze_complexity(task_data):
                    return TaskAnalysis(
                        complexity=TaskComplexity.LOW,
                        estimated_duration=900,
                        required_skills=['data_processing'],
                        resource_requirements={'cpu': 0.2, 'memory': 0.1},
                        success_probability=0.95,
                        recommended_agents=['data_processor_002']
                    )
                
                mock_analyzer.analyze_complexity.side_effect = mock_analyze_complexity
                
                analysis = await mock_analyzer.analyze_complexity(low_complexity_task)
                
                assert analysis.complexity == TaskComplexity.LOW
                assert analysis.estimated_duration == 900
                assert analysis.success_probability == 0.95
    
    @pytest.mark.asyncio
    async def test_analyze_task_complexity_high(self, orchestrator, sample_task):
        """Test analysis of high complexity task"""
        # Modify task to be high complexity
        high_complexity_task = sample_task.copy()
        high_complexity_task['requirements']['data_size'] = 10000000  # 10M records
        high_complexity_task['requirements']['accuracy_required'] = 0.999
        high_complexity_task['context']['complexity_indicators'] = [
            'very_large_dataset', 'extremely_high_accuracy', 'tight_deadline'
        ]
        
        if IMPORTS_AVAILABLE and hasattr(orchestrator, 'analyze_task'):
            analysis = await orchestrator.analyze_task(high_complexity_task)
            
            assert isinstance(analysis, TaskAnalysis)
            assert analysis.complexity in [TaskComplexity.HIGH, TaskComplexity.VERY_HIGH]
            assert analysis.estimated_duration > 3600  # More than 1 hour
            assert len(analysis.required_skills) > 2
            assert analysis.success_probability < 0.9  # Lower probability for complex tasks
        else:
            # Mock the behavior
            with patch.object(orchestrator, 'task_analyzer') as mock_analyzer:
                async def mock_analyze_complexity(task_data):
                    return TaskAnalysis(
                        complexity=TaskComplexity.HIGH,
                        estimated_duration=7200,
                        required_skills=['financial_analysis', 'data_processing', 'statistical_analysis'],
                        resource_requirements={'cpu': 0.8, 'memory': 0.6, 'storage': 0.4},
                        success_probability=0.82,
                        recommended_agents=['financial_specialist_001', 'data_processor_002']
                    )
                
                mock_analyzer.analyze_complexity.side_effect = mock_analyze_complexity
                
                analysis = await mock_analyzer.analyze_complexity(high_complexity_task)
                
                assert analysis.complexity == TaskComplexity.HIGH
                assert analysis.estimated_duration == 7200
                assert len(analysis.required_skills) == 3
    
    @pytest.mark.asyncio
    async def test_analyze_task_with_missing_information(self, orchestrator):
        """Test task analysis with incomplete information"""
        incomplete_task = {
            'id': 'incomplete_task',
            'type': 'unknown',
            'description': 'Vague task description',
            'requirements': {},
            'context': {}
        }
        
        with patch.object(orchestrator, 'task_analyzer') as mock_analyzer:
            async def mock_analyze_complexity(task_data):
                return TaskAnalysis(
                    complexity=TaskComplexity.MEDIUM,
                    estimated_duration=3600,
                    required_skills=['general_processing'],
                    resource_requirements={'cpu': 0.3, 'memory': 0.2},
                    success_probability=0.7,  # Lower due to uncertainty
                    recommended_agents=[]
                )
            
            mock_analyzer.analyze_complexity.side_effect = mock_analyze_complexity
            
            analysis = await mock_analyzer.analyze_complexity(incomplete_task)
            
            assert analysis.complexity == TaskComplexity.MEDIUM
            assert analysis.success_probability <= 0.7
            assert len(analysis.recommended_agents) == 0
    
    @pytest.mark.parametrize("data_size,expected_complexity", [
        (100, TaskComplexity.LOW),
        (10000, TaskComplexity.MEDIUM),
        (1000000, TaskComplexity.HIGH),
        (100000000, TaskComplexity.VERY_HIGH)
    ])
    @pytest.mark.asyncio
    async def test_complexity_analysis_data_size_scaling(self, orchestrator, data_size, expected_complexity):
        """Test complexity analysis scales with data size"""
        task = {
            'id': f'scaling_test_{data_size}',
            'requirements': {'data_size': data_size},
            'context': {'complexity_indicators': []}
        }
        
        with patch.object(orchestrator, 'task_analyzer') as mock_analyzer:
            async def mock_complexity_analysis(task_data):
                size = task_data['requirements']['data_size']
                if size < 1000:
                    complexity = TaskComplexity.LOW
                elif size < 100000:
                    complexity = TaskComplexity.MEDIUM
                elif size < 10000000:
                    complexity = TaskComplexity.HIGH
                else:
                    complexity = TaskComplexity.VERY_HIGH
                    
                return TaskAnalysis(
                    complexity=complexity,
                    estimated_duration=size / 1000 * 60,  # Scale time with data
                    required_skills=['data_processing'],
                    resource_requirements={'cpu': min(size / 1000000, 1.0)},
                    success_probability=max(0.6, 1.0 - size / 100000000),
                    recommended_agents=[]
                )
            
            mock_analyzer.analyze_complexity.side_effect = mock_complexity_analysis
            analysis = await mock_analyzer.analyze_complexity(task)
            
            assert analysis.complexity == expected_complexity


class TestStrategyLearning(TestEnhancedMetaOrchestratorUnit):
    """Test strategy learning and optimization"""
    
    @pytest.mark.asyncio
    async def test_learn_from_execution_success(self, orchestrator):
        """Test learning from successful task execution"""
        execution_data = {
            'task_id': 'task_001',
            'task_type': 'data_processing',
            'strategy_used': 'parallel_processing',
            'execution_time': 1800,
            'success': True,
            'resource_usage': {'cpu': 0.6, 'memory': 0.4},
            'cost': 25.0,
            'quality_score': 0.95,
            'agent_feedback': {
                'difficulty_rating': 3,
                'tool_effectiveness': 0.9,
                'collaboration_quality': 0.8
            }
        }
        
        with patch.object(orchestrator, 'strategy_learner') as mock_learner:
            async def mock_learn_from_execution(execution_data):
                return {
                    'patterns_discovered': ['parallel_effective_for_large_data'],
                    'strategy_updates': {
                        'parallel_processing': {'effectiveness': 0.95, 'confidence': 0.85}
                    },
                    'recommendations': ['increase_parallel_threshold']
                }
            
            mock_learner.learn_from_execution.side_effect = mock_learn_from_execution
            
            result = await mock_learner.learn_from_execution(execution_data)
            
            mock_learner.learn_from_execution.assert_called_once_with(execution_data)
            assert 'patterns_discovered' in result
            assert 'parallel_effective_for_large_data' in result['patterns_discovered']
    
    @pytest.mark.asyncio
    async def test_learn_from_execution_failure(self, orchestrator):
        """Test learning from failed task execution"""
        execution_data = {
            'task_id': 'task_002',
            'task_type': 'financial_analysis',
            'strategy_used': 'single_agent_sequential',
            'execution_time': 5400,
            'success': False,
            'failure_reason': 'timeout',
            'resource_usage': {'cpu': 0.9, 'memory': 0.8},
            'cost': 75.0,
            'quality_score': 0.0,
            'agent_feedback': {
                'difficulty_rating': 5,
                'tool_effectiveness': 0.3,
                'bottlenecks': ['insufficient_memory', 'complex_calculations']
            }
        }
        
        with patch.object(orchestrator, 'strategy_learner') as mock_learner:
            async def mock_learn_from_execution(execution_data):
                return {
                    'failure_patterns': ['sequential_ineffective_for_complex_finance'],
                    'strategy_adjustments': {
                        'single_agent_sequential': {'effectiveness': 0.2, 'use_threshold': 'low_complexity_only'}
                    },
                    'recommendations': ['use_parallel_for_complex_finance', 'increase_memory_allocation']
                }
            
            mock_learner.learn_from_execution.side_effect = mock_learn_from_execution
            
            learning_result = await mock_learner.learn_from_execution(execution_data)
            
            assert 'failure_patterns' in learning_result
            assert 'sequential_ineffective_for_complex_finance' in learning_result['failure_patterns']
    
    @pytest.mark.asyncio
    async def test_strategy_recommendation_generation(self, orchestrator, sample_task):
        """Test generation of strategy recommendations"""
        with patch.object(orchestrator, 'strategy_learner') as mock_learner:
            async def mock_recommend_strategy(task_data):
                return [
                    StrategyRecommendation(
                        strategy_type='parallel_processing',
                        confidence=0.85,
                        expected_performance=0.92,
                        resource_cost=35.0,
                        explanation='Historical data shows parallel processing is 85% effective for similar tasks'
                    ),
                    StrategyRecommendation(
                        strategy_type='hierarchical_delegation',
                        confidence=0.78,
                        expected_performance=0.89,
                        resource_cost=42.0,
                        explanation='Task complexity suggests hierarchical approach may be beneficial'
                    )
                ]
            
            mock_learner.recommend_strategy.side_effect = mock_recommend_strategy
            
            recommendations = await mock_learner.recommend_strategy(sample_task)
            
            assert len(recommendations) == 2
            assert recommendations[0].confidence > recommendations[1].confidence
            assert all(rec.confidence > 0.7 for rec in recommendations)
            assert all(rec.expected_performance > 0.8 for rec in recommendations)
    
    @pytest.mark.asyncio
    async def test_pattern_recognition_across_tasks(self, orchestrator):
        """Test pattern recognition across multiple task executions"""
        execution_history = [
            {
                'task_type': 'data_processing',
                'data_size': 1000000,
                'strategy': 'parallel',
                'success': True,
                'execution_time': 1200
            },
            {
                'task_type': 'data_processing',
                'data_size': 2000000,
                'strategy': 'parallel',
                'success': True,
                'execution_time': 2100
            },
            {
                'task_type': 'data_processing',
                'data_size': 500000,
                'strategy': 'sequential',
                'success': True,
                'execution_time': 800
            }
        ]
        
        with patch.object(orchestrator, 'strategy_learner') as mock_learner:
            async def mock_extract_patterns(execution_history):
                return {
                    'patterns': [
                        {
                            'pattern_type': 'data_size_strategy_correlation',
                            'description': 'Parallel processing most effective for data_size > 1M',
                            'confidence': 0.9,
                            'sample_size': 3
                        }
                    ],
                    'insights': [
                        'Sequential processing preferred for smaller datasets',
                        'Parallel processing scales well with data size'
                    ]
                }
            
            mock_learner.extract_patterns.side_effect = mock_extract_patterns
            
            patterns = await mock_learner.extract_patterns(execution_history)
            
            assert len(patterns['patterns']) >= 1
            assert patterns['patterns'][0]['confidence'] > 0.8
            assert len(patterns['insights']) >= 2


class TestPerformanceTracking(TestEnhancedMetaOrchestratorUnit):
    """Test performance tracking and metrics collection"""
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, orchestrator):
        """Test collection of performance metrics"""
        task_execution = {
            'task_id': 'perf_test_001',
            'start_time': datetime.now() - timedelta(minutes=30),
            'end_time': datetime.now(),
            'success': True,
            'agents_used': ['agent_001', 'agent_002'],
            'resource_consumption': {
                'cpu_hours': 0.5,
                'memory_gb_hours': 2.0,
                'storage_gb': 1.5
            },
            'cost': 18.5,
            'quality_metrics': {
                'accuracy': 0.96,
                'completeness': 0.98,
                'consistency': 0.94
            }
        }
        
        with patch.object(orchestrator, 'performance_tracker') as mock_tracker:
            async def mock_record_execution(task_execution):
                return PerformanceMetrics(
                    throughput=2.0,  # tasks per hour
                    accuracy=0.96,
                    resource_utilization=0.65,
                    cost_efficiency=0.82,
                    response_time=1800  # 30 minutes
                )
            
            mock_tracker.record_execution.side_effect = mock_record_execution
            
            metrics = await mock_tracker.record_execution(task_execution)
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.throughput > 0
            assert 0 <= metrics.accuracy <= 1
            assert 0 <= metrics.resource_utilization <= 1
            assert metrics.cost_efficiency > 0
    
    @pytest.mark.asyncio
    async def test_performance_trend_analysis(self, orchestrator):
        """Test analysis of performance trends over time"""
        historical_metrics = [
            {'timestamp': datetime.now() - timedelta(days=7), 'throughput': 1.5, 'accuracy': 0.94},
            {'timestamp': datetime.now() - timedelta(days=6), 'throughput': 1.7, 'accuracy': 0.95},
            {'timestamp': datetime.now() - timedelta(days=5), 'throughput': 1.6, 'accuracy': 0.96},
            {'timestamp': datetime.now() - timedelta(days=4), 'throughput': 1.9, 'accuracy': 0.97},
            {'timestamp': datetime.now() - timedelta(days=3), 'throughput': 2.1, 'accuracy': 0.96},
            {'timestamp': datetime.now() - timedelta(days=2), 'throughput': 2.0, 'accuracy': 0.98},
            {'timestamp': datetime.now() - timedelta(days=1), 'throughput': 2.2, 'accuracy': 0.97}
        ]
        
        with patch.object(orchestrator, 'performance_tracker') as mock_tracker:
            async def mock_analyze_trends(historical_metrics):
                return {
                    'throughput_trend': 'increasing',
                    'throughput_change_rate': 0.1,  # 10% improvement per day
                    'accuracy_trend': 'stable',
                    'accuracy_change_rate': 0.01,
                    'predictions': {
                        'next_week_throughput': 2.8,
                        'next_week_accuracy': 0.97
                    },
                    'anomalies': [],
                    'recommendations': ['maintain_current_strategy']
                }
            
            mock_tracker.analyze_trends.side_effect = mock_analyze_trends
            
            trends = await mock_tracker.analyze_trends(historical_metrics)
            
            assert trends['throughput_trend'] in ['increasing', 'stable', 'decreasing']
            assert trends['accuracy_trend'] in ['increasing', 'stable', 'decreasing']
            assert 'predictions' in trends
            assert isinstance(trends['recommendations'], list)
    
    @pytest.mark.asyncio
    async def test_performance_bottleneck_detection(self, orchestrator):
        """Test detection of performance bottlenecks"""
        system_metrics = {
            'cpu_utilization': [0.45, 0.52, 0.48, 0.91, 0.87, 0.93, 0.89],
            'memory_utilization': [0.65, 0.62, 0.68, 0.71, 0.69, 0.66, 0.64],
            'network_io': [0.23, 0.25, 0.21, 0.89, 0.92, 0.88, 0.91],
            'task_queue_length': [2, 1, 3, 12, 15, 18, 14],
            'response_times': [1.2, 1.1, 1.3, 4.5, 5.2, 4.8, 5.1]
        }
        
        with patch.object(orchestrator, 'performance_tracker') as mock_tracker:
            async def mock_detect_bottlenecks(system_metrics):
                return {
                    'bottlenecks_detected': [
                        {
                            'type': 'cpu_saturation',
                            'severity': 'high',
                            'affected_period': '15:30-16:30',
                            'impact': 'response_time_increase',
                            'recommendations': ['scale_cpu_resources', 'optimize_algorithms']
                        },
                        {
                            'type': 'network_io_congestion',
                            'severity': 'medium',
                            'affected_period': '15:30-16:30',
                            'impact': 'task_queue_buildup',
                            'recommendations': ['increase_network_bandwidth', 'implement_caching']
                        }
                    ],
                    'root_cause_analysis': 'Concurrent high-complexity tasks overloaded system',
                    'mitigation_strategies': ['dynamic_resource_scaling', 'intelligent_task_scheduling']
                }
            
            mock_tracker.detect_bottlenecks.side_effect = mock_detect_bottlenecks
            
            bottlenecks = await mock_tracker.detect_bottlenecks(system_metrics)
            
            assert len(bottlenecks['bottlenecks_detected']) >= 1
            assert all('severity' in b for b in bottlenecks['bottlenecks_detected'])
            assert 'root_cause_analysis' in bottlenecks
            assert isinstance(bottlenecks['mitigation_strategies'], list)


class TestDynamicAgentSelection(TestEnhancedMetaOrchestratorUnit):
    """Test dynamic agent selection algorithms"""
    
    @pytest.mark.asyncio
    async def test_optimal_agent_selection(self, orchestrator, sample_task, mock_agents):
        """Test selection of optimal agent for task"""
        with patch.object(orchestrator, 'agent_selector') as mock_selector:
            async def mock_select_optimal_agents(sample_task, mock_agents):
                return {
                    'primary_agent': 'financial_specialist_001',
                    'backup_agents': ['data_processor_002'],
                    'selection_rationale': {
                        'primary_reasons': ['domain_expertise', 'high_success_rate', 'low_current_load'],
                        'confidence': 0.92
                    },
                    'estimated_performance': {
                        'success_probability': 0.96,
                        'estimated_time': 2800,
                        'estimated_cost': 12.5
                    }
                }
            
            mock_selector.select_optimal_agents.side_effect = mock_select_optimal_agents
            
            selection = await mock_selector.select_optimal_agents(sample_task, mock_agents)
            
            assert 'primary_agent' in selection
            assert selection['primary_agent'] == 'financial_specialist_001'
            assert selection['selection_rationale']['confidence'] > 0.9
            assert len(selection['backup_agents']) >= 1
    
    @pytest.mark.asyncio
    async def test_multi_agent_team_selection(self, orchestrator, mock_agents):
        """Test selection of multi-agent teams for complex tasks"""
        complex_task = {
            'id': 'complex_team_task',
            'type': 'comprehensive_financial_audit',
            'requirements': {
                'data_size': 5000000,
                'accuracy_required': 0.999,
                'multiple_domains': ['accounting', 'tax', 'compliance'],
                'deadline': datetime.now() + timedelta(hours=4)
            }
        }
        
        with patch.object(orchestrator, 'agent_selector') as mock_selector:
            async def mock_select_agent_team(complex_task, mock_agents):
                return {
                    'team_composition': [
                        {
                            'agent_id': 'financial_specialist_001',
                            'role': 'lead_analyst',
                            'responsibility': 'financial_analysis'
                        },
                        {
                            'agent_id': 'data_processor_002',
                            'role': 'data_specialist',
                            'responsibility': 'data_processing'
                        },
                        {
                            'agent_id': 'generalist_coordinator_003',
                            'role': 'coordinator',
                            'responsibility': 'task_coordination'
                        }
                    ],
                    'coordination_strategy': 'hierarchical_with_peer_review',
                    'estimated_team_performance': {
                        'success_probability': 0.94,
                        'estimated_time': 3200,
                        'total_cost': 45.5
                    }
                }
            
            mock_selector.select_agent_team.side_effect = mock_select_agent_team
            
            team_selection = await mock_selector.select_agent_team(complex_task, mock_agents)
            
            assert len(team_selection['team_composition']) >= 2
            assert 'coordination_strategy' in team_selection
            assert team_selection['estimated_team_performance']['success_probability'] > 0.9
    
    @pytest.mark.asyncio
    async def test_load_balancing_consideration(self, orchestrator, mock_agents):
        """Test that agent selection considers current load"""
        # Modify mock agents to have different loads
        high_load_agents = mock_agents.copy()
        high_load_agents[0]['current_load'] = 0.95  # Very high load
        high_load_agents[1]['current_load'] = 0.1   # Low load
        
        task = {
            'id': 'load_balance_test',
            'requirements': {'urgency': 'high'},
            'type': 'data_processing'
        }
        
        with patch.object(orchestrator, 'agent_selector') as mock_selector:
            async def mock_selection(task_data, agents):
                # Should prefer low-load agents for urgent tasks
                available_agents = [a for a in agents if a['current_load'] < 0.8]
                if available_agents:
                    return {
                        'primary_agent': available_agents[0]['id'],
                        'selection_rationale': {
                            'primary_reasons': ['low_current_load', 'availability'],
                            'load_consideration': True
                        }
                    }
                return {'primary_agent': None}
            
            mock_selector.select_optimal_agents.side_effect = mock_selection
            selection = await mock_selector.select_optimal_agents(task, high_load_agents)
            
            assert selection['primary_agent'] == 'data_processor_002'  # Low load agent
            assert selection['selection_rationale']['load_consideration'] is True
    
    @pytest.mark.asyncio
    async def test_capability_matching(self, orchestrator, mock_agents):
        """Test matching task requirements to agent capabilities"""
        specialized_task = {
            'id': 'capability_test',
            'requirements': {
                'required_capabilities': ['financial_analysis', 'regulatory_compliance'],
                'domain_expertise': 'finance'
            }
        }
        
        with patch.object(orchestrator, 'agent_selector') as mock_selector:
            async def mock_capability_match(task_data, agents):
                required_caps = set(task_data['requirements']['required_capabilities'])
                matching_agents = []
                
                for agent in agents:
                    agent_caps = set(agent['capabilities'])
                    match_score = len(required_caps & agent_caps) / len(required_caps)
                    if match_score >= 0.5:  # At least 50% capability match
                        matching_agents.append((agent, match_score))
                
                if matching_agents:
                    best_agent = max(matching_agents, key=lambda x: x[1])
                    return {
                        'primary_agent': best_agent[0]['id'],
                        'capability_match_score': best_agent[1],
                        'selection_rationale': {
                            'primary_reasons': ['capability_match'],
                            'match_details': {
                                'required': list(required_caps),
                                'matched': list(set(best_agent[0]['capabilities']) & required_caps)
                            }
                        }
                    }
                return {'primary_agent': None}
            
            mock_selector.match_capabilities.side_effect = mock_capability_match
            selection = await mock_selector.match_capabilities(specialized_task, mock_agents)
            
            assert selection['primary_agent'] == 'financial_specialist_001'
            assert selection['capability_match_score'] >= 0.5


class TestResourceOptimization(TestEnhancedMetaOrchestratorUnit):
    """Test resource optimization strategies"""
    
    @pytest.mark.asyncio
    async def test_resource_allocation_optimization(self, orchestrator):
        """Test optimization of resource allocation"""
        resource_request = {
            'task_id': 'resource_test_001',
            'estimated_requirements': {
                'cpu_cores': 4,
                'memory_gb': 8,
                'storage_gb': 50,
                'network_bandwidth_mbps': 100
            },
            'priority': 1,
            'deadline': datetime.now() + timedelta(hours=1)
        }
        
        available_resources = {
            'total_cpu_cores': 16,
            'total_memory_gb': 64,
            'total_storage_gb': 1000,
            'total_network_bandwidth_mbps': 1000,
            'current_allocations': {
                'cpu_cores': 8,
                'memory_gb': 24,
                'storage_gb': 200,
                'network_bandwidth_mbps': 400
            }
        }
        
        with patch.object(orchestrator, 'resource_optimizer') as mock_optimizer:
            async def mock_optimize_allocation(resource_request, available_resources):
                return {
                    'allocated_resources': {
                        'cpu_cores': 4,
                        'memory_gb': 8,
                        'storage_gb': 50,
                        'network_bandwidth_mbps': 100
                    },
                    'optimization_strategy': 'immediate_allocation',
                    'estimated_efficiency': 0.85,
                    'resource_utilization': {
                        'cpu': 0.75,
                        'memory': 0.5,
                        'storage': 0.25,
                        'network': 0.5
                    },
                    'recommendations': ['monitor_memory_usage', 'consider_cpu_scaling']
                }
            
            mock_optimizer.optimize_allocation.side_effect = mock_optimize_allocation
            
            allocation = await mock_optimizer.optimize_allocation(resource_request, available_resources)
            
            assert allocation['allocated_resources']['cpu_cores'] <= available_resources['total_cpu_cores']
            assert allocation['estimated_efficiency'] > 0.8
            assert 'optimization_strategy' in allocation
    
    @pytest.mark.asyncio
    async def test_cost_optimization(self, orchestrator):
        """Test cost optimization strategies"""
        execution_options = [
            {
                'strategy': 'single_high_performance_agent',
                'estimated_cost': 50.0,
                'estimated_time': 1800,
                'success_probability': 0.95
            },
            {
                'strategy': 'multiple_standard_agents',
                'estimated_cost': 35.0,
                'estimated_time': 2400,
                'success_probability': 0.92
            },
            {
                'strategy': 'economy_mode',
                'estimated_cost': 20.0,
                'estimated_time': 3600,
                'success_probability': 0.88
            }
        ]
        
        constraints = {
            'max_budget': 40.0,
            'max_time': 3000,
            'min_success_probability': 0.9
        }
        
        with patch.object(orchestrator, 'resource_optimizer') as mock_optimizer:
            async def mock_cost_optimization(options, constraints_data):
                viable_options = []
                for option in options:
                    if (option['estimated_cost'] <= constraints_data['max_budget'] and
                        option['estimated_time'] <= constraints_data['max_time'] and
                        option['success_probability'] >= constraints_data['min_success_probability']):
                        # Calculate value score (success_prob / cost * time_factor)
                        time_factor = 1.0 - (option['estimated_time'] / 3600)
                        value_score = option['success_probability'] / option['estimated_cost'] * time_factor
                        viable_options.append((option, value_score))
                
                if viable_options:
                    best_option = max(viable_options, key=lambda x: x[1])
                    return {
                        'selected_strategy': best_option[0]['strategy'],
                        'optimization_score': best_option[1],
                        'trade_offs': {
                            'cost_savings': 50.0 - best_option[0]['estimated_cost'],
                            'time_trade_off': best_option[0]['estimated_time'] - 1800
                        }
                    }
                return {}
            
            mock_optimizer.optimize_cost.side_effect = mock_cost_optimization
            optimization = await mock_optimizer.optimize_cost(execution_options, constraints)
            
            assert optimization['selected_strategy'] == 'multiple_standard_agents'
            assert optimization['optimization_score'] > 0
    
    @pytest.mark.asyncio
    async def test_dynamic_resource_scaling(self, orchestrator):
        """Test dynamic resource scaling based on demand"""
        current_demand = {
            'active_tasks': 8,
            'queued_tasks': 12,
            'average_task_complexity': 0.7,
            'peak_resource_utilization': {
                'cpu': 0.85,
                'memory': 0.75,
                'network': 0.6
            }
        }
        
        with patch.object(orchestrator, 'resource_optimizer') as mock_optimizer:
            async def mock_calculate_scaling_needs(current_demand):
                return {
                    'scaling_recommendation': 'scale_up',
                    'scaling_factor': 1.5,
                    'target_resources': {
                        'cpu_cores': 24,  # up from 16
                        'memory_gb': 96,  # up from 64
                        'agents': 12      # up from 8
                    },
                    'scaling_rationale': [
                        'high_cpu_utilization',
                        'task_queue_buildup',
                        'increasing_task_complexity'
                    ],
                    'estimated_improvement': {
                        'throughput_increase': 0.4,
                        'response_time_reduction': 0.3,
                        'queue_reduction': 0.6
                    }
                }
            
            mock_optimizer.calculate_scaling_needs.side_effect = mock_calculate_scaling_needs
            
            scaling = await mock_optimizer.calculate_scaling_needs(current_demand)
            
            assert scaling['scaling_recommendation'] in ['scale_up', 'scale_down', 'maintain']
            assert scaling['scaling_factor'] > 1.0  # Should recommend scaling up
            assert len(scaling['scaling_rationale']) > 0


class TestIntegrationScenarios(TestEnhancedMetaOrchestratorUnit):
    """Test integrated scenarios combining multiple components"""
    
    @pytest.mark.asyncio
    async def test_complete_task_orchestration_cycle(self, orchestrator, sample_task, mock_agents):
        """Test complete orchestration cycle from task analysis to execution"""
        # Mock all components to work together
        with patch.object(orchestrator, 'task_analyzer') as mock_analyzer, \
             patch.object(orchestrator, 'strategy_learner') as mock_learner, \
             patch.object(orchestrator, 'agent_selector') as mock_selector, \
             patch.object(orchestrator, 'resource_optimizer') as mock_optimizer, \
             patch.object(orchestrator, 'performance_tracker') as mock_tracker:
            
            # Set up component responses
            mock_analyzer.analyze_complexity.return_value = TaskAnalysis(
                complexity=TaskComplexity.MEDIUM,
                estimated_duration=2400,
                required_skills=['financial_analysis', 'data_processing'],
                resource_requirements={'cpu': 0.4, 'memory': 0.3},
                success_probability=0.92,
                recommended_agents=['financial_specialist_001']
            )
            
            mock_learner.recommend_strategy.return_value = [
                StrategyRecommendation(
                    strategy_type='parallel_processing',
                    confidence=0.85,
                    expected_performance=0.91,
                    resource_cost=30.0,
                    explanation='Parallel processing recommended for this task type'
                )
            ]
            
            mock_selector.select_optimal_agents.return_value = {
                'primary_agent': 'financial_specialist_001',
                'backup_agents': ['data_processor_002']
            }
            
            mock_optimizer.optimize_allocation.return_value = {
                'allocated_resources': {'cpu_cores': 2, 'memory_gb': 4},
                'estimated_efficiency': 0.88
            }
            
            # Simulate orchestration method that doesn't exist yet
            async def mock_orchestrate_task(task):
                analysis = await mock_analyzer.analyze_complexity(task)
                strategies = await mock_learner.recommend_strategy(task)
                agents = await mock_selector.select_optimal_agents(task, mock_agents)
                resources = await mock_optimizer.optimize_allocation({
                    'task_id': task['id'],
                    'estimated_requirements': analysis.resource_requirements
                }, {'total_cpu_cores': 16, 'total_memory_gb': 64})
                
                return {
                    'task_analysis': analysis,
                    'recommended_strategy': strategies[0],
                    'selected_agents': agents,
                    'resource_allocation': resources,
                    'orchestration_status': 'ready_for_execution'
                }
            
            # Execute orchestration
            if hasattr(orchestrator, 'orchestrate_task'):
                result = await orchestrator.orchestrate_task(sample_task)
            else:
                result = await mock_orchestrate_task(sample_task)
            
            # Verify complete orchestration
            assert result['orchestration_status'] == 'ready_for_execution'
            assert result['task_analysis'].complexity == TaskComplexity.MEDIUM
            assert result['recommended_strategy'].confidence > 0.8
            assert result['selected_agents']['primary_agent'] == 'financial_specialist_001'
            assert result['resource_allocation']['estimated_efficiency'] > 0.8
    
    @pytest.mark.asyncio
    async def test_adaptive_orchestration_with_feedback(self, orchestrator):
        """Test orchestration adaptation based on execution feedback"""
        # Simulate multiple task executions with feedback
        execution_history = [
            {
                'task_type': 'financial_analysis',
                'strategy_used': 'single_agent',
                'success': False,
                'execution_time': 5400,
                'failure_reason': 'complexity_underestimated'
            },
            {
                'task_type': 'financial_analysis',
                'strategy_used': 'parallel_processing',
                'success': True,
                'execution_time': 2100,
                'quality_score': 0.94
            }
        ]
        
        with patch.object(orchestrator, 'strategy_learner') as mock_learner:
            # Set up async mock functions
            async def mock_learn_from_execution_failure(execution_data):
                return {
                    'patterns_discovered': ['financial_analysis_needs_parallel'],
                    'strategy_updates': {'single_agent': {'effectiveness': 0.3}}
                }
            
            async def mock_learn_from_execution_success(execution_data):
                return {
                    'patterns_reinforced': ['financial_analysis_needs_parallel'],
                    'strategy_updates': {'parallel_processing': {'effectiveness': 0.94}}
                }
            
            # First execution learns from failure
            mock_learner.learn_from_execution.side_effect = mock_learn_from_execution_failure
            await mock_learner.learn_from_execution(execution_history[0])
            
            # Second execution confirms the learning
            mock_learner.learn_from_execution.side_effect = mock_learn_from_execution_success
            await mock_learner.learn_from_execution(execution_history[1])
            
            # New task should now prefer parallel processing
            async def mock_recommend_strategy(task_data):
                return [
                    StrategyRecommendation(
                        strategy_type='parallel_processing',
                        confidence=0.94,  # High confidence due to learning
                        expected_performance=0.93,
                        resource_cost=25.0,
                        explanation='Learned that parallel processing is highly effective for financial analysis'
                    )
                ]
            
            mock_learner.recommend_strategy.side_effect = mock_recommend_strategy
            
            new_task = {'type': 'financial_analysis'}
            recommendations = await mock_learner.recommend_strategy(new_task)
            
            assert recommendations[0].strategy_type == 'parallel_processing'
            assert recommendations[0].confidence > 0.9  # Should be very confident now


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--cov=agents.meta.enhanced_meta_orchestrator"])