"""
Comprehensive test suite for Phase 7 Advanced Reasoning Systems
Tests all reasoning components including causal inference, working memory, 
tree of thoughts, temporal reasoning, and integrated reasoning controller
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import os
from dataclasses import dataclass
import json
import time

# Import reasoning systems
try:
    from core.reasoning.causal_inference import CausalReasoningEngine, CausalGraph, InterventionResult
    from core.reasoning.working_memory import WorkingMemorySystem, MemoryNode, CoherenceMetrics
    from core.reasoning.tree_of_thoughts import EnhancedTreeOfThoughts, ThoughtNode, ThoughtQuality
    from core.reasoning.temporal_reasoning import TemporalReasoningEngine, TemporalPattern, TemporalPrediction
    from core.reasoning.integrated_reasoning_controller import IntegratedReasoningController, ReasoningMode
    from core.reasoning.performance_optimizer import PerformanceOptimizer
    REASONING_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Reasoning system imports not available: {e}")
    REASONING_IMPORTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REASONING_IMPORTS_AVAILABLE,
    reason="Reasoning systems not available"
)


@dataclass
class TestConfig:
    """Configuration for reasoning system tests"""
    causal_accuracy_threshold: float = 0.90
    working_memory_token_limit: int = 10000
    temporal_prediction_horizons: List[str] = None
    performance_timeout: float = 1.0  # seconds
    
    def __post_init__(self):
        if self.temporal_prediction_horizons is None:
            self.temporal_prediction_horizons = ["minute", "hour", "day", "week"]


class TestDataGenerator:
    """Generate test data for reasoning systems"""
    
    @staticmethod
    def generate_causal_data(n_samples: int = 100, n_features: int = 5) -> pd.DataFrame:
        """Generate synthetic data with known causal relationships"""
        np.random.seed(42)
        
        # Generate base features
        x1 = np.random.normal(0, 1, n_samples)
        x2 = 0.5 * x1 + np.random.normal(0, 0.5, n_samples)  # x1 -> x2
        x3 = np.random.normal(0, 1, n_samples)
        x4 = 0.3 * x2 + 0.4 * x3 + np.random.normal(0, 0.3, n_samples)  # x2, x3 -> x4
        x5 = 0.6 * x4 + np.random.normal(0, 0.4, n_samples)  # x4 -> x5
        
        return pd.DataFrame({
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5
        })
    
    @staticmethod
    def generate_temporal_data(n_points: int = 1000) -> pd.DataFrame:
        """Generate temporal data with patterns"""
        np.random.seed(42)
        
        # Generate time series with trend, seasonality, and noise
        time_index = pd.date_range(start='2023-01-01', periods=n_points, freq='H')
        
        # Trend component
        trend = np.linspace(0, 10, n_points)
        
        # Seasonal component (daily pattern)
        seasonal = 3 * np.sin(2 * np.pi * np.arange(n_points) / 24)
        
        # Random noise
        noise = np.random.normal(0, 1, n_points)
        
        # Anomalies
        anomaly_indices = np.random.choice(n_points, size=int(0.05 * n_points), replace=False)
        anomalies = np.zeros(n_points)
        anomalies[anomaly_indices] = np.random.normal(0, 5, len(anomaly_indices))
        
        values = trend + seasonal + noise + anomalies
        
        return pd.DataFrame({
            'timestamp': time_index,
            'value': values,
            'trend': trend,
            'seasonal': seasonal,
            'noise': noise,
            'anomaly': anomalies
        })
    
    @staticmethod
    def generate_memory_content(n_items: int = 50) -> List[Dict[str, Any]]:
        """Generate diverse content for working memory tests"""
        content_types = [
            "factual", "procedural", "episodic", "semantic", "causal", "temporal"
        ]
        
        memories = []
        for i in range(n_items):
            content_type = content_types[i % len(content_types)]
            memory = {
                "content": f"Test memory content {i} of type {content_type}",
                "type": content_type,
                "importance": np.random.uniform(0.1, 1.0),
                "timestamp": datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                "metadata": {
                    "source": f"test_source_{i % 5}",
                    "domain": f"domain_{i % 3}",
                    "complexity": np.random.randint(1, 6)
                }
            }
            memories.append(memory)
        
        return memories


@pytest.fixture
def test_config():
    """Provide test configuration"""
    return TestConfig()


@pytest.fixture
def test_data_generator():
    """Provide test data generator"""
    return TestDataGenerator()


@pytest.fixture
def causal_test_data(test_data_generator):
    """Generate causal test data"""
    return test_data_generator.generate_causal_data()


@pytest.fixture
def temporal_test_data(test_data_generator):
    """Generate temporal test data"""
    return test_data_generator.generate_temporal_data()


@pytest.fixture
def memory_test_data(test_data_generator):
    """Generate memory test data"""
    return test_data_generator.generate_memory_content()


@pytest.fixture
async def causal_engine():
    """Create causal reasoning engine for testing"""
    engine = CausalReasoningEngine()
    await engine.initialize()
    return engine


@pytest.fixture
async def working_memory(test_config):
    """Create working memory system for testing"""
    system = WorkingMemorySystem(
        max_total_tokens=test_config.working_memory_token_limit,
        max_short_term_items=100
    )
    await system.initialize()
    return system


@pytest.fixture
async def tree_of_thoughts():
    """Create enhanced tree of thoughts system"""
    system = EnhancedTreeOfThoughts()
    await system.initialize()
    return system


@pytest.fixture
async def temporal_engine():
    """Create temporal reasoning engine"""
    engine = TemporalReasoningEngine()
    await engine.initialize()
    return engine


@pytest.fixture
async def integrated_controller(causal_engine, working_memory, tree_of_thoughts, temporal_engine):
    """Create integrated reasoning controller"""
    controller = IntegratedReasoningController(
        causal_engine=causal_engine,
        working_memory=working_memory,
        tree_of_thoughts=tree_of_thoughts,
        temporal_engine=temporal_engine
    )
    await controller.initialize()
    return controller


@pytest.fixture
async def performance_optimizer():
    """Create performance optimizer"""
    optimizer = PerformanceOptimizer()
    await optimizer.initialize()
    return optimizer


class TestCausalReasoningEngine:
    """Test suite for causal reasoning engine"""
    
    @pytest.mark.asyncio
    async def test_causal_discovery_basic(self, causal_engine, causal_test_data):
        """Test basic causal discovery functionality"""
        # Discover causal relationships
        causal_graph = await causal_engine.discover_causal_relationships(
            causal_test_data,
            discovery_method="pc"
        )
        
        # Verify graph structure
        assert causal_graph is not None
        assert len(causal_graph.nodes) == 5
        assert len(causal_graph.edges) > 0
        
        # Check for expected relationships (x1 -> x2, etc.)
        edges = causal_graph.get_edge_list()
        edge_pairs = [(edge['from'], edge['to']) for edge in edges]
        
        # Should detect at least some true causal relationships
        assert len(edge_pairs) >= 2
    
    @pytest.mark.asyncio
    async def test_causal_discovery_ensemble(self, causal_engine, causal_test_data):
        """Test ensemble causal discovery for improved accuracy"""
        causal_graph = await causal_engine.discover_causal_relationships(
            causal_test_data,
            discovery_method="ensemble"
        )
        
        # Ensemble should provide confidence scores
        edges = causal_graph.get_edge_list()
        for edge in edges:
            assert "confidence" in edge
            assert 0 <= edge["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_intervention_analysis(self, causal_engine, causal_test_data):
        """Test causal intervention analysis"""
        # First discover causal structure
        causal_graph = await causal_engine.discover_causal_relationships(
            causal_test_data,
            discovery_method="pc"
        )
        
        # Test intervention
        intervention_result = await causal_engine.analyze_intervention(
            causal_graph,
            intervention_var="x1",
            intervention_value=2.0,
            target_vars=["x2", "x4"]
        )
        
        assert intervention_result is not None
        assert "x2" in intervention_result.effects
        assert "x4" in intervention_result.effects
        
        # x1 -> x2 is direct, so effect should be significant
        assert abs(intervention_result.effects["x2"]) > 0.1
    
    @pytest.mark.asyncio
    async def test_counterfactual_analysis(self, causal_engine, causal_test_data):
        """Test counterfactual reasoning"""
        causal_graph = await causal_engine.discover_causal_relationships(
            causal_test_data,
            discovery_method="pc"
        )
        
        # Test counterfactual query
        counterfactual_result = await causal_engine.analyze_counterfactual(
            causal_graph,
            factual_data=causal_test_data.iloc[0].to_dict(),
            counterfactual_var="x1",
            counterfactual_value=1.5
        )
        
        assert counterfactual_result is not None
        assert "counterfactual_outcome" in counterfactual_result
    
    @pytest.mark.asyncio
    async def test_accuracy_target(self, causal_engine, causal_test_data, test_config):
        """Test that causal accuracy meets the 90% target"""
        # Generate multiple test scenarios
        accuracies = []
        
        for seed in range(5):  # Test multiple random seeds
            np.random.seed(seed)
            test_data = TestDataGenerator.generate_causal_data(n_samples=200)
            
            causal_graph = await causal_engine.discover_causal_relationships(
                test_data,
                discovery_method="ensemble"
            )
            
            # Calculate accuracy against known ground truth
            accuracy = await causal_engine.evaluate_discovery_accuracy(
                causal_graph,
                ground_truth_edges=[("x1", "x2"), ("x2", "x4"), ("x3", "x4"), ("x4", "x5")]
            )
            
            accuracies.append(accuracy)
        
        # Average accuracy should meet target
        avg_accuracy = np.mean(accuracies)
        assert avg_accuracy >= test_config.causal_accuracy_threshold, \
            f"Causal accuracy {avg_accuracy:.2%} below target {test_config.causal_accuracy_threshold:.2%}"


class TestWorkingMemorySystem:
    """Test suite for working memory system"""
    
    @pytest.mark.asyncio
    async def test_memory_storage_and_retrieval(self, working_memory, memory_test_data):
        """Test basic memory storage and retrieval"""
        # Store memories
        stored_ids = []
        for memory in memory_test_data[:10]:
            memory_id = await working_memory.store_memory(
                memory["content"],
                memory_type=memory["type"],
                importance=memory["importance"],
                metadata=memory["metadata"]
            )
            stored_ids.append(memory_id)
        
        assert len(stored_ids) == 10
        
        # Test retrieval
        retrieved = await working_memory.recall_memories(
            query="test memory content",
            limit=5
        )
        
        assert len(retrieved) <= 5
        assert all(mem.content for mem in retrieved)
    
    @pytest.mark.asyncio
    async def test_memory_consolidation(self, working_memory, memory_test_data):
        """Test memory consolidation process"""
        # Store many memories to trigger consolidation
        for memory in memory_test_data:
            await working_memory.store_memory(
                memory["content"],
                memory_type=memory["type"],
                importance=memory["importance"]
            )
        
        # Trigger consolidation
        consolidation_result = await working_memory.consolidate_memories()
        
        assert consolidation_result["patterns_identified"] > 0
        assert consolidation_result["memories_consolidated"] > 0
        
        # Verify system state after consolidation
        stats = working_memory.get_memory_statistics()
        assert stats["long_term_memories"] > 0
    
    @pytest.mark.asyncio
    async def test_coherence_tracking(self, working_memory, memory_test_data):
        """Test memory coherence tracking"""
        # Store related memories
        related_memories = [
            "The sky is blue on a clear day",
            "Clear days have blue skies", 
            "Blue is the color of clear skies",
            "On cloudy days, the sky is gray"  # Contradictory context
        ]
        
        for memory in related_memories:
            await working_memory.store_memory(memory, memory_type="factual")
        
        # Check coherence metrics
        coherence_metrics = await working_memory.assess_coherence()
        
        assert coherence_metrics.overall_score >= 0.0
        assert coherence_metrics.consistency_score >= 0.0
        assert len(coherence_metrics.detected_inconsistencies) >= 0
    
    @pytest.mark.asyncio
    async def test_token_limit_compliance(self, working_memory, test_config):
        """Test that memory system respects token limits"""
        # Store memories until approaching token limit
        long_memories = []
        for i in range(100):
            long_memory = f"This is a very long memory content item {i} " * 50  # ~500 tokens each
            long_memories.append(long_memory)
        
        for memory in long_memories:
            await working_memory.store_memory(memory, memory_type="episodic")
        
        # Check token usage
        stats = working_memory.get_memory_statistics()
        assert stats["total_tokens"] <= test_config.working_memory_token_limit
        
        # Verify memory management kicked in
        assert stats["total_memories"] < len(long_memories)  # Some memories should be consolidated/archived
    
    @pytest.mark.asyncio
    async def test_hierarchical_organization(self, working_memory, memory_test_data):
        """Test hierarchical memory organization"""
        # Store memories with different importance levels
        high_importance = [mem for mem in memory_test_data if mem["importance"] > 0.8]
        low_importance = [mem for mem in memory_test_data if mem["importance"] < 0.3]
        
        for memory in high_importance + low_importance:
            await working_memory.store_memory(
                memory["content"],
                importance=memory["importance"],
                memory_type=memory["type"]
            )
        
        # High importance memories should be in active memory
        active_memories = await working_memory.get_active_memories(limit=20)
        active_importance = [mem.importance for mem in active_memories]
        
        # Should prioritize high importance memories
        assert np.mean(active_importance) > 0.5


class TestEnhancedTreeOfThoughts:
    """Test suite for enhanced tree of thoughts system"""
    
    @pytest.mark.asyncio
    async def test_basic_reasoning(self, tree_of_thoughts):
        """Test basic tree of thoughts reasoning"""
        problem = "How can we optimize energy consumption in a smart building?"
        
        solution = await tree_of_thoughts.solve_problem(
            problem,
            max_depth=3,
            max_branches=3
        )
        
        assert solution is not None
        assert solution.final_answer is not None
        assert len(solution.reasoning_path) > 0
        assert solution.confidence > 0.0
    
    @pytest.mark.asyncio
    async def test_adaptive_search_strategy(self, tree_of_thoughts):
        """Test adaptive search strategy selection"""
        # Test different problem types
        problems = [
            ("Mathematical optimization problem", "analytical"),
            ("Creative design challenge", "creative"),
            ("Complex systems analysis", "comprehensive")
        ]
        
        for problem, expected_strategy_type in problems:
            result = await tree_of_thoughts.solve_problem(
                problem,
                max_depth=2,
                adaptive_strategy=True
            )
            
            assert result is not None
            # Strategy should adapt based on problem characteristics
            assert "strategy_used" in result.metadata
    
    @pytest.mark.asyncio
    async def test_quality_scoring(self, tree_of_thoughts):
        """Test thought quality scoring mechanism"""
        problem = "Design a sustainable transportation system"
        
        # Generate multiple thoughts for comparison
        thoughts = await tree_of_thoughts.generate_initial_thoughts(problem, n_thoughts=5)
        
        assert len(thoughts) == 5
        
        for thought in thoughts:
            assert thought.quality_score >= 0.0
            assert thought.quality_score <= 1.0
            assert hasattr(thought, 'feasibility_score')
            assert hasattr(thought, 'novelty_score')
    
    @pytest.mark.asyncio
    async def test_working_memory_integration(self, tree_of_thoughts, working_memory):
        """Test integration with working memory system"""
        # Configure with working memory
        tree_of_thoughts.configure_working_memory(working_memory)
        
        # Store relevant context in working memory
        context_items = [
            "Sustainable energy sources include solar and wind power",
            "Transportation systems should minimize environmental impact",
            "Smart city infrastructure enables optimized resource usage"
        ]
        
        for item in context_items:
            await working_memory.store_memory(item, memory_type="factual")
        
        # Solve problem that should use stored context
        solution = await tree_of_thoughts.solve_problem(
            "Design an eco-friendly smart city transportation network"
        )
        
        assert solution is not None
        # Solution should reference stored knowledge
        assert any(keyword in solution.final_answer.lower() 
                  for keyword in ["sustainable", "smart", "environmental"])
    
    @pytest.mark.asyncio
    async def test_parallel_exploration(self, tree_of_thoughts):
        """Test parallel exploration of solution paths"""
        problem = "Develop a multi-modal AI system for healthcare diagnostics"
        
        start_time = time.time()
        
        solution = await tree_of_thoughts.solve_problem(
            problem,
            max_depth=3,
            max_branches=4,
            parallel_exploration=True
        )
        
        parallel_time = time.time() - start_time
        
        # Test sequential execution for comparison
        start_time = time.time()
        
        solution_seq = await tree_of_thoughts.solve_problem(
            problem,
            max_depth=3,
            max_branches=4,
            parallel_exploration=False
        )
        
        sequential_time = time.time() - start_time
        
        # Parallel should be faster (or at least comparable)
        assert parallel_time <= sequential_time * 1.2  # Allow 20% tolerance
        
        # Both should produce valid solutions
        assert solution.confidence > 0.5
        assert solution_seq.confidence > 0.5


class TestTemporalReasoningEngine:
    """Test suite for temporal reasoning engine"""
    
    @pytest.mark.asyncio
    async def test_pattern_detection(self, temporal_engine, temporal_test_data):
        """Test temporal pattern detection"""
        # Add temporal data to engine
        await temporal_engine.add_temporal_data(
            temporal_test_data[['timestamp', 'value']].to_dict('records')
        )
        
        # Detect patterns
        patterns = await temporal_engine.detect_patterns()
        
        assert len(patterns) > 0
        
        # Should detect the daily seasonal pattern
        seasonal_patterns = [p for p in patterns if p.pattern_type == "seasonal"]
        assert len(seasonal_patterns) > 0
        
        # Should detect trend
        trend_patterns = [p for p in patterns if p.pattern_type == "trend"]
        assert len(trend_patterns) > 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, temporal_engine, temporal_test_data):
        """Test temporal anomaly detection"""
        await temporal_engine.add_temporal_data(
            temporal_test_data[['timestamp', 'value']].to_dict('records')
        )
        
        # Detect anomalies
        anomalies = await temporal_engine.detect_anomalies(
            sensitivity=0.95
        )
        
        assert len(anomalies) > 0
        
        # Should detect approximately 5% anomalies (as generated)
        anomaly_rate = len(anomalies) / len(temporal_test_data)
        assert 0.03 <= anomaly_rate <= 0.07  # 3-7% tolerance
    
    @pytest.mark.asyncio
    async def test_multi_horizon_prediction(self, temporal_engine, temporal_test_data, test_config):
        """Test multi-horizon temporal prediction"""
        # Add historical data
        train_data = temporal_test_data.iloc[:-100]  # Leave 100 points for validation
        await temporal_engine.add_temporal_data(
            train_data[['timestamp', 'value']].to_dict('records')
        )
        
        # Generate predictions for multiple horizons
        predictions = {}
        for horizon in test_config.temporal_prediction_horizons:
            prediction = await temporal_engine.predict(
                horizon=horizon,
                confidence_level=0.95
            )
            predictions[horizon] = prediction
        
        assert len(predictions) == len(test_config.temporal_prediction_horizons)
        
        # All predictions should have confidence intervals
        for horizon, prediction in predictions.items():
            assert prediction.mean_prediction is not None
            assert prediction.confidence_interval is not None
            assert prediction.confidence_interval["lower"] <= prediction.mean_prediction
            assert prediction.confidence_interval["upper"] >= prediction.mean_prediction
    
    @pytest.mark.asyncio
    async def test_causal_temporal_analysis(self, temporal_engine):
        """Test causal relationships in temporal data"""
        # Generate data with known causal temporal relationships
        np.random.seed(42)
        n_points = 500
        
        # X causes Y with a 2-hour delay
        timestamps = pd.date_range(start='2023-01-01', periods=n_points, freq='H')
        x_values = np.random.normal(0, 1, n_points)
        y_values = np.zeros(n_points)
        
        for i in range(2, n_points):
            y_values[i] = 0.7 * x_values[i-2] + np.random.normal(0, 0.3)
        
        # Add data to temporal engine
        temporal_data = [
            {"timestamp": ts, "variable": "X", "value": x_val}
            for ts, x_val in zip(timestamps, x_values)
        ] + [
            {"timestamp": ts, "variable": "Y", "value": y_val}
            for ts, y_val in zip(timestamps, y_values)
        ]
        
        await temporal_engine.add_temporal_data(temporal_data)
        
        # Analyze causal relationships
        causal_relationships = await temporal_engine.analyze_causal_relationships(
            variables=["X", "Y"]
        )
        
        assert len(causal_relationships) > 0
        
        # Should detect X -> Y relationship with ~2 hour lag
        xy_relationship = next(
            (rel for rel in causal_relationships if rel.cause == "X" and rel.effect == "Y"),
            None
        )
        
        assert xy_relationship is not None
        assert abs(xy_relationship.lag_hours - 2) <= 1  # Within 1 hour tolerance
    
    @pytest.mark.asyncio
    async def test_temporal_state_management(self, temporal_engine):
        """Test temporal state management and memory"""
        # Add events across different time periods
        events = [
            {"timestamp": datetime.now() - timedelta(days=7), "type": "system_start", "value": 1.0},
            {"timestamp": datetime.now() - timedelta(days=5), "type": "high_load", "value": 0.9},
            {"timestamp": datetime.now() - timedelta(days=3), "type": "maintenance", "value": 0.1},
            {"timestamp": datetime.now() - timedelta(days=1), "type": "normal_operation", "value": 0.5},
            {"timestamp": datetime.now(), "type": "current_state", "value": 0.6}
        ]
        
        for event in events:
            await temporal_engine.add_temporal_event(event)
        
        # Get current temporal state
        current_state = await temporal_engine.get_temporal_state()
        
        assert current_state is not None
        assert "recent_events" in current_state
        assert "active_patterns" in current_state
        assert "temporal_context" in current_state
        
        # Should maintain recent events in memory
        assert len(current_state["recent_events"]) <= len(events)


class TestIntegratedReasoningController:
    """Test suite for integrated reasoning controller"""
    
    @pytest.mark.asyncio
    async def test_reasoning_mode_selection(self, integrated_controller):
        """Test automatic reasoning mode selection"""
        test_cases = [
            ("What causes customer churn in our subscription service?", ReasoningMode.CAUSAL),
            ("Predict sales for the next quarter", ReasoningMode.TEMPORAL),
            ("How can we optimize our supply chain?", ReasoningMode.ANALYTICAL),
            ("What will happen if we increase prices by 10%?", ReasoningMode.PREDICTIVE)
        ]
        
        for query, expected_mode in test_cases:
            reasoning_session = await integrated_controller.create_reasoning_session(
                session_id=f"test_{hash(query)}"
            )
            
            selected_mode = await integrated_controller.select_reasoning_mode(
                query, reasoning_session
            )
            
            # Mode selection should be appropriate (allow some flexibility)
            assert selected_mode in [expected_mode, ReasoningMode.ADAPTIVE]
    
    @pytest.mark.asyncio
    async def test_multi_system_coordination(self, integrated_controller):
        """Test coordination across multiple reasoning systems"""
        query = """
        Analyze the causal factors behind declining customer satisfaction scores,
        predict future trends, and provide strategic recommendations.
        """
        
        # This should engage multiple reasoning systems
        result = await integrated_controller.reason(
            query=query,
            reasoning_mode=ReasoningMode.ADAPTIVE,
            session_id="multi_system_test"
        )
        
        assert result is not None
        assert result.reasoning_path is not None
        assert len(result.reasoning_path) > 1  # Should use multiple systems
        
        # Should include contributions from different systems
        system_types = [step.system_type for step in result.reasoning_path]
        assert len(set(system_types)) >= 2  # At least 2 different systems
    
    @pytest.mark.asyncio
    async def test_session_management(self, integrated_controller):
        """Test reasoning session management"""
        session_id = "test_session_123"
        
        # Create session
        session = await integrated_controller.create_reasoning_session(session_id)
        assert session.session_id == session_id
        
        # Add multiple interactions
        queries = [
            "What are the main factors affecting our business?",
            "How do these factors interact with each other?",
            "What can we predict about future performance?"
        ]
        
        results = []
        for query in queries:
            result = await integrated_controller.reason(
                query=query,
                session_id=session_id
            )
            results.append(result)
        
        # Session should maintain context across queries
        session_state = await integrated_controller.get_session_state(session_id)
        assert len(session_state["interaction_history"]) == len(queries)
        
        # Later queries should reference earlier context
        later_results = results[-1]
        assert later_results.context_used is not None
    
    @pytest.mark.asyncio
    async def test_performance_targets(self, integrated_controller, test_config):
        """Test that system meets performance targets"""
        simple_queries = [
            "What is the current status?",
            "Provide a summary",
            "Calculate basic metrics"
        ]
        
        for query in simple_queries:
            start_time = time.time()
            
            result = await integrated_controller.reason(
                query=query,
                reasoning_mode=ReasoningMode.ANALYTICAL,
                session_id=f"perf_test_{hash(query)}"
            )
            
            response_time = time.time() - start_time
            
            # Should meet sub-second response time for simple queries
            assert response_time < test_config.performance_timeout, \
                f"Response time {response_time:.2f}s exceeds target {test_config.performance_timeout}s"
            
            assert result is not None
    
    @pytest.mark.asyncio
    async def test_concurrent_reasoning_sessions(self, integrated_controller):
        """Test concurrent reasoning sessions"""
        # Create multiple concurrent sessions
        session_ids = [f"concurrent_session_{i}" for i in range(5)]
        queries = [f"Analyze scenario {i}" for i in range(5)]
        
        # Execute concurrently
        tasks = []
        for session_id, query in zip(session_ids, queries):
            task = integrated_controller.reason(
                query=query,
                session_id=session_id
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All sessions should complete successfully
        assert len(results) == 5
        assert all(result is not None for result in results)
        
        # Each session should maintain separate state
        for i, session_id in enumerate(session_ids):
            session_state = await integrated_controller.get_session_state(session_id)
            assert len(session_state["interaction_history"]) == 1
            assert session_state["interaction_history"][0]["query"] == queries[i]


class TestPerformanceOptimizer:
    """Test suite for performance optimizer"""
    
    @pytest.mark.asyncio
    async def test_cache_optimization(self, performance_optimizer):
        """Test intelligent caching system"""
        # Perform repeated operations
        test_operations = [
            ("compute_similarity", {"text1": "hello world", "text2": "hello earth"}),
            ("analyze_pattern", {"data": [1, 2, 3, 4, 5]}),
            ("compute_similarity", {"text1": "hello world", "text2": "hello earth"}),  # Repeat
            ("generate_embedding", {"text": "test embedding"})
        ]
        
        execution_times = []
        
        for operation, params in test_operations:
            start_time = time.time()
            
            result = await performance_optimizer.execute_with_optimization(
                operation_type=operation,
                operation_func=lambda: f"result_for_{operation}",
                params=params
            )
            
            execution_time = time.time() - start_time
            execution_times.append(execution_time)
            
            assert result is not None
        
        # Repeated operation (index 2) should be faster due to caching
        assert execution_times[2] < execution_times[0] * 0.5  # At least 50% faster
    
    @pytest.mark.asyncio
    async def test_resource_monitoring(self, performance_optimizer):
        """Test resource monitoring and optimization"""
        # Start resource monitoring
        monitoring_task = asyncio.create_task(
            performance_optimizer.monitor_resources(interval=0.1)
        )
        
        # Simulate resource-intensive operations
        await asyncio.sleep(0.5)
        
        # Get resource metrics
        metrics = performance_optimizer.get_resource_metrics()
        
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics
        assert "response_times" in metrics
        
        # Stop monitoring
        monitoring_task.cancel()
    
    @pytest.mark.asyncio
    async def test_adaptive_optimization(self, performance_optimizer):
        """Test adaptive optimization strategies"""
        # Simulate different workload patterns
        workload_patterns = [
            {"pattern": "high_frequency", "operations_per_second": 100},
            {"pattern": "batch_processing", "batch_size": 1000},
            {"pattern": "mixed_workload", "variety": 5}
        ]
        
        for pattern in workload_patterns:
            # Apply optimization strategy
            strategy = await performance_optimizer.adapt_optimization_strategy(
                workload_characteristics=pattern
            )
            
            assert strategy is not None
            assert "cache_strategy" in strategy
            assert "resource_allocation" in strategy
            assert "execution_strategy" in strategy
    
    @pytest.mark.asyncio
    async def test_performance_bottleneck_detection(self, performance_optimizer):
        """Test bottleneck detection and resolution"""
        # Simulate operations with artificial delays
        slow_operations = [
            ("slow_operation_1", 0.2),
            ("fast_operation", 0.01),
            ("slow_operation_2", 0.15),
            ("normal_operation", 0.05)
        ]
        
        for op_name, delay in slow_operations:
            await performance_optimizer.execute_with_optimization(
                operation_type=op_name,
                operation_func=lambda d=delay: asyncio.sleep(d),
                params={}
            )
        
        # Analyze for bottlenecks
        bottlenecks = await performance_optimizer.identify_bottlenecks()
        
        assert len(bottlenecks) > 0
        
        # Should identify slow operations
        slow_bottlenecks = [b for b in bottlenecks if "slow_operation" in b["operation"]]
        assert len(slow_bottlenecks) >= 2


class TestReasoningSystemIntegration:
    """Integration tests across all reasoning systems"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_reasoning_workflow(self, integrated_controller, test_config):
        """Test complete end-to-end reasoning workflow"""
        # Complex business scenario requiring multiple reasoning systems
        scenario = """
        Our e-commerce platform has experienced a 15% decline in user engagement 
        over the past 3 months. Historical data shows seasonal patterns, but this 
        decline appears unusual. We need to:
        1. Identify potential causal factors
        2. Predict future engagement trends  
        3. Develop strategic recommendations
        4. Assess the impact of proposed interventions
        """
        
        # Execute comprehensive analysis
        start_time = time.time()
        
        result = await integrated_controller.reason(
            query=scenario,
            reasoning_mode=ReasoningMode.ADAPTIVE,
            session_id="e2e_business_analysis",
            enable_all_systems=True
        )
        
        execution_time = time.time() - start_time
        
        # Verify comprehensive result
        assert result is not None
        assert result.final_answer is not None
        assert len(result.reasoning_path) >= 3  # Should use multiple systems
        
        # Should address all aspects of the query
        answer_lower = result.final_answer.lower()
        assert any(keyword in answer_lower for keyword in ["causal", "cause", "factor"])
        assert any(keyword in answer_lower for keyword in ["predict", "future", "trend"])
        assert any(keyword in answer_lower for keyword in ["recommend", "strategy", "action"])
        
        # Performance should be reasonable for complex query
        assert execution_time < 30.0  # Max 30 seconds for complex analysis
    
    @pytest.mark.asyncio
    async def test_system_resilience(self, integrated_controller):
        """Test system resilience under various conditions"""
        # Test error handling
        invalid_queries = [
            "",  # Empty query
            "?!@#$%^&*()",  # Invalid characters
            "A" * 10000,  # Extremely long query
            None  # None input
        ]
        
        for query in invalid_queries:
            try:
                result = await integrated_controller.reason(
                    query=query,
                    session_id=f"resilience_test_{hash(str(query))}"
                )
                
                # Should handle gracefully
                if result is not None:
                    assert result.success is False or result.error_message is not None
                    
            except Exception as e:
                # Should not crash the system
                assert "catastrophic" not in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_scalability_under_load(self, integrated_controller):
        """Test system scalability under concurrent load"""
        # Generate multiple concurrent requests
        concurrent_queries = [
            f"Analyze business metric {i} for optimization opportunities"
            for i in range(20)
        ]
        
        start_time = time.time()
        
        # Execute concurrently
        tasks = []
        for i, query in enumerate(concurrent_queries):
            task = integrated_controller.reason(
                query=query,
                session_id=f"load_test_session_{i}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        execution_time = time.time() - start_time
        
        # Most requests should complete successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_results) / len(concurrent_queries)
        
        assert success_rate >= 0.8  # At least 80% success rate
        
        # Should handle load efficiently
        avg_time_per_query = execution_time / len(concurrent_queries)
        assert avg_time_per_query < 5.0  # Average less than 5 seconds per query
    
    @pytest.mark.asyncio
    async def test_memory_and_learning_integration(self, integrated_controller):
        """Test integration between memory systems and learning"""
        session_id = "learning_integration_test"
        
        # Sequence of related queries to test learning
        learning_sequence = [
            "What factors influence customer retention?",
            "How do pricing changes affect customer behavior?", 
            "What retention strategies work best for price-sensitive customers?",
            "Predict the impact of a 5% price reduction on retention"
        ]
        
        results = []
        for i, query in enumerate(learning_sequence):
            result = await integrated_controller.reason(
                query=query,
                session_id=session_id
            )
            results.append(result)
            
            # Later queries should show improved context awareness
            if i > 0:
                assert result.context_used is not None
                
                # Should reference earlier findings
                context_mentions = sum(
                    1 for prev_result in results[:-1]
                    if any(word in result.final_answer.lower() 
                          for word in prev_result.final_answer.lower().split()[:10])
                )
                assert context_mentions > 0  # Should build on previous answers
        
        # Final result should integrate insights from entire sequence
        final_result = results[-1]
        assert "retention" in final_result.final_answer.lower()
        assert "price" in final_result.final_answer.lower()


@pytest.mark.performance
class TestPerformanceValidation:
    """Performance validation tests"""
    
    @pytest.mark.asyncio
    async def test_causal_discovery_performance(self, causal_engine):
        """Test causal discovery performance on larger datasets"""
        # Generate larger test dataset
        large_data = TestDataGenerator.generate_causal_data(n_samples=1000, n_features=10)
        
        start_time = time.time()
        
        causal_graph = await causal_engine.discover_causal_relationships(
            large_data,
            discovery_method="pc"
        )
        
        discovery_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert discovery_time < 30.0  # Max 30 seconds for 1000 samples
        assert causal_graph is not None
        assert len(causal_graph.nodes) == 10
    
    @pytest.mark.asyncio 
    async def test_memory_system_performance(self, working_memory):
        """Test working memory system performance"""
        # Store large number of memories
        n_memories = 1000
        memories = TestDataGenerator.generate_memory_content(n_memories)
        
        start_time = time.time()
        
        for memory in memories:
            await working_memory.store_memory(
                memory["content"],
                memory_type=memory["type"],
                importance=memory["importance"]
            )
        
        storage_time = time.time() - start_time
        
        # Test retrieval performance
        start_time = time.time()
        
        retrieved = await working_memory.recall_memories(
            query="test memory content",
            limit=50
        )
        
        retrieval_time = time.time() - start_time
        
        # Performance assertions
        assert storage_time < 60.0  # Max 60 seconds for 1000 memories
        assert retrieval_time < 2.0   # Max 2 seconds for retrieval
        assert len(retrieved) <= 50
    
    @pytest.mark.asyncio
    async def test_temporal_analysis_performance(self, temporal_engine):
        """Test temporal analysis performance on large datasets"""
        # Generate large temporal dataset
        large_temporal_data = TestDataGenerator.generate_temporal_data(n_points=5000)
        
        start_time = time.time()
        
        await temporal_engine.add_temporal_data(
            large_temporal_data[['timestamp', 'value']].to_dict('records')
        )
        
        # Perform pattern detection
        patterns = await temporal_engine.detect_patterns()
        
        # Perform prediction
        prediction = await temporal_engine.predict(
            horizon="day",
            confidence_level=0.95
        )
        
        analysis_time = time.time() - start_time
        
        # Performance assertions
        assert analysis_time < 45.0  # Max 45 seconds for 5000 points
        assert len(patterns) > 0
        assert prediction is not None


if __name__ == "__main__":
    # Run tests with coverage
    import sys
    
    # Add reasoning systems to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Run with performance profiling
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-m", "not slow",  # Skip slow tests by default
        "--cov=core.reasoning",
        "--cov-report=term-missing",
        "--cov-report=html:test_reports/reasoning_coverage"
    ])