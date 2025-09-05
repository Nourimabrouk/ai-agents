"""
Validation test suite for reasoning systems
Focuses on accuracy validation, ground truth testing, and benchmarking against known standards
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from dataclasses import dataclass
import logging

# Import reasoning systems
try:
    from core.reasoning.causal_inference import CausalReasoningEngine
    from core.reasoning.working_memory import WorkingMemorySystem
    from core.reasoning.tree_of_thoughts import EnhancedTreeOfThoughts
    from core.reasoning.temporal_reasoning import TemporalReasoningEngine
    from core.reasoning.integrated_reasoning_controller import IntegratedReasoningController
    REASONING_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Reasoning system imports not available: {e}")
    REASONING_IMPORTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not REASONING_IMPORTS_AVAILABLE,
    reason="Reasoning systems not available"
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation test"""
    test_name: str
    system_name: str
    accuracy_score: float
    precision_score: float
    recall_score: float
    f1_score: float
    execution_time: float
    memory_usage: float
    additional_metrics: Dict[str, Any]
    passed_threshold: bool


class GroundTruthDatasets:
    """Manager for ground truth datasets used in validation"""
    
    @staticmethod
    def get_causal_ground_truth() -> List[Dict[str, Any]]:
        """Generate ground truth causal datasets with known relationships"""
        datasets = []
        
        # Linear causal chain: A -> B -> C
        np.random.seed(1234)
        n = 500
        a = np.random.normal(0, 1, n)
        b = 2.0 * a + np.random.normal(0, 0.5, n)
        c = 1.5 * b + np.random.normal(0, 0.3, n)
        
        datasets.append({
            "name": "linear_chain",
            "data": pd.DataFrame({"A": a, "B": b, "C": c}),
            "ground_truth": [("A", "B"), ("B", "C")],
            "description": "Simple linear causal chain A->B->C"
        })
        
        # Fork structure: A -> B, A -> C
        a = np.random.normal(0, 1, n)
        b = 1.8 * a + np.random.normal(0, 0.4, n)
        c = 1.2 * a + np.random.normal(0, 0.6, n)
        
        datasets.append({
            "name": "fork_structure", 
            "data": pd.DataFrame({"A": a, "B": b, "C": c}),
            "ground_truth": [("A", "B"), ("A", "C")],
            "description": "Fork structure A->B and A->C"
        })
        
        # Collider structure: A -> C <- B
        a = np.random.normal(0, 1, n)
        b = np.random.normal(0, 1, n) 
        c = 0.7 * a + 0.8 * b + np.random.normal(0, 0.2, n)
        
        datasets.append({
            "name": "collider_structure",
            "data": pd.DataFrame({"A": a, "B": b, "C": c}),
            "ground_truth": [("A", "C"), ("B", "C")],
            "description": "Collider structure A->C<-B"
        })
        
        # Complex network
        a = np.random.normal(0, 1, n)
        b = 0.6 * a + np.random.normal(0, 0.4, n)
        c = np.random.normal(0, 1, n)
        d = 0.5 * b + 0.4 * c + np.random.normal(0, 0.3, n)
        e = 0.3 * a + 0.7 * d + np.random.normal(0, 0.2, n)
        
        datasets.append({
            "name": "complex_network",
            "data": pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e}),
            "ground_truth": [("A", "B"), ("B", "D"), ("C", "D"), ("A", "E"), ("D", "E")],
            "description": "Complex causal network with multiple paths"
        })
        
        return datasets
    
    @staticmethod
    def get_temporal_ground_truth() -> List[Dict[str, Any]]:
        """Generate temporal datasets with known patterns"""
        datasets = []
        
        # Pure trend
        n = 1000
        time_points = pd.date_range(start='2023-01-01', periods=n, freq='H')
        trend_values = np.linspace(10, 50, n) + np.random.normal(0, 2, n)
        
        datasets.append({
            "name": "pure_trend",
            "data": pd.DataFrame({"timestamp": time_points, "value": trend_values}),
            "ground_truth": {
                "trend": {"exists": True, "direction": "increasing", "strength": "strong"},
                "seasonality": {"exists": False},
                "anomalies": {"count": 0, "positions": []}
            },
            "description": "Pure linear trend without seasonality"
        })
        
        # Seasonal pattern (daily)
        hours = np.arange(n) % 24
        seasonal_values = 10 + 5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 1, n)
        
        datasets.append({
            "name": "daily_seasonal",
            "data": pd.DataFrame({"timestamp": time_points, "value": seasonal_values}),
            "ground_truth": {
                "trend": {"exists": False},
                "seasonality": {"exists": True, "period": 24, "amplitude": 5},
                "anomalies": {"count": 0, "positions": []}
            },
            "description": "Pure daily seasonal pattern"
        })
        
        # Trend + Seasonality + Anomalies
        trend_seasonal = np.linspace(5, 25, n) + 3 * np.sin(2 * np.pi * hours / 24)
        
        # Add specific anomalies
        anomaly_indices = [100, 300, 600, 800]
        anomaly_values = trend_seasonal.copy()
        for idx in anomaly_indices:
            if idx < len(anomaly_values):
                anomaly_values[idx] += np.random.choice([-15, 15])  # Extreme outliers
        
        anomaly_values += np.random.normal(0, 1, n)  # Background noise
        
        datasets.append({
            "name": "complex_temporal",
            "data": pd.DataFrame({"timestamp": time_points, "value": anomaly_values}),
            "ground_truth": {
                "trend": {"exists": True, "direction": "increasing", "strength": "moderate"},
                "seasonality": {"exists": True, "period": 24, "amplitude": 3},
                "anomalies": {"count": len(anomaly_indices), "positions": anomaly_indices}
            },
            "description": "Complex pattern with trend, seasonality, and anomalies"
        })
        
        return datasets
    
    @staticmethod
    def get_reasoning_ground_truth() -> List[Dict[str, Any]]:
        """Generate reasoning problems with known solutions"""
        problems = []
        
        # Mathematical optimization
        problems.append({
            "name": "linear_optimization",
            "problem": "Find the maximum value of 3x + 2y subject to x + y <= 10, x >= 0, y >= 0",
            "ground_truth": {
                "solution": {"x": 10, "y": 0},
                "optimal_value": 30,
                "solution_type": "unique"
            },
            "description": "Simple linear programming problem"
        })
        
        # Logic puzzle
        problems.append({
            "name": "logic_puzzle",
            "problem": """
            There are three people: Alice, Bob, and Charlie.
            - Alice always tells the truth
            - Bob always lies
            - Charlie sometimes tells the truth and sometimes lies
            
            Today, one of them says: "I am Charlie"
            Who said it?
            """,
            "ground_truth": {
                "solution": "Charlie",
                "reasoning": "Alice can't claim to be Charlie (would be false), Bob can't claim to be Charlie (would be true)",
                "confidence": 1.0
            },
            "description": "Classic logic puzzle about truth-tellers and liars"
        })
        
        # Causal reasoning scenario
        problems.append({
            "name": "causal_scenario",
            "problem": """
            A company notices that:
            1. When they increase advertising spend, sales go up
            2. When sales go up, customer satisfaction scores increase
            3. When customer satisfaction increases, employee morale improves
            
            What would happen if they cut advertising spend by 50%?
            """,
            "ground_truth": {
                "primary_effect": "reduced_sales",
                "secondary_effects": ["lower_satisfaction", "reduced_morale"],
                "effect_magnitude": "proportional_to_cut",
                "causal_chain": ["advertising", "sales", "satisfaction", "morale"]
            },
            "description": "Business causal reasoning scenario"
        })
        
        return problems


class ValidationSuite:
    """Comprehensive validation suite for reasoning systems"""
    
    def __init__(self):
        self.ground_truth = GroundTruthDatasets()
        self.results: List[ValidationResult] = []
    
    async def run_causal_validation(self, causal_engine: CausalReasoningEngine) -> List[ValidationResult]:
        """Validate causal reasoning engine against ground truth"""
        causal_datasets = self.ground_truth.get_causal_ground_truth()
        results = []
        
        for dataset in causal_datasets:
            logger.info(f"Validating causal discovery on: {dataset['name']}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Run causal discovery
            discovered_graph = await causal_engine.discover_causal_relationships(
                dataset["data"],
                discovery_method="ensemble"
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Evaluate against ground truth
            accuracy_metrics = self._evaluate_causal_accuracy(
                discovered_graph.get_edge_list(),
                dataset["ground_truth"]
            )
            
            result = ValidationResult(
                test_name=f"causal_{dataset['name']}",
                system_name="causal_inference",
                accuracy_score=accuracy_metrics["accuracy"],
                precision_score=accuracy_metrics["precision"],
                recall_score=accuracy_metrics["recall"],
                f1_score=accuracy_metrics["f1"],
                execution_time=execution_time,
                memory_usage=0.0,  # TODO: Add memory tracking
                logger.info('TODO item needs implementation')
                additional_metrics={
                    "dataset_size": len(dataset["data"]),
                    "true_edges": len(dataset["ground_truth"]),
                    "discovered_edges": len(discovered_graph.get_edge_list()),
                    "description": dataset["description"]
                },
                passed_threshold=accuracy_metrics["accuracy"] >= 0.90
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    async def run_temporal_validation(self, temporal_engine: TemporalReasoningEngine) -> List[ValidationResult]:
        """Validate temporal reasoning engine"""
        temporal_datasets = self.ground_truth.get_temporal_ground_truth()
        results = []
        
        for dataset in temporal_datasets:
            logger.info(f"Validating temporal analysis on: {dataset['name']}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Add data and analyze
            await temporal_engine.add_temporal_data(
                dataset["data"].to_dict('records')
            )
            
            # Detect patterns
            detected_patterns = await temporal_engine.detect_patterns()
            
            # Detect anomalies if expected
            detected_anomalies = []
            if dataset["ground_truth"].get("anomalies", {}).get("count", 0) > 0:
                detected_anomalies = await temporal_engine.detect_anomalies()
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Evaluate accuracy
            accuracy_metrics = self._evaluate_temporal_accuracy(
                detected_patterns,
                detected_anomalies,
                dataset["ground_truth"]
            )
            
            result = ValidationResult(
                test_name=f"temporal_{dataset['name']}",
                system_name="temporal_reasoning",
                accuracy_score=accuracy_metrics["accuracy"],
                precision_score=accuracy_metrics["precision"],
                recall_score=accuracy_metrics["recall"],
                f1_score=accuracy_metrics["f1"],
                execution_time=execution_time,
                memory_usage=0.0,
                additional_metrics={
                    "data_points": len(dataset["data"]),
                    "detected_patterns": len(detected_patterns),
                    "detected_anomalies": len(detected_anomalies),
                    "description": dataset["description"]
                },
                passed_threshold=accuracy_metrics["accuracy"] >= 0.85
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    async def run_reasoning_validation(self, reasoning_system: EnhancedTreeOfThoughts) -> List[ValidationResult]:
        """Validate general reasoning capabilities"""
        reasoning_problems = self.ground_truth.get_reasoning_ground_truth()
        results = []
        
        for problem in reasoning_problems:
            logger.info(f"Validating reasoning on: {problem['name']}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Solve problem
            solution = await reasoning_system.solve_problem(
                problem["problem"],
                max_depth=4,
                max_branches=3
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Evaluate solution quality
            quality_metrics = self._evaluate_reasoning_quality(
                solution,
                problem["ground_truth"]
            )
            
            result = ValidationResult(
                test_name=f"reasoning_{problem['name']}",
                system_name="tree_of_thoughts",
                accuracy_score=quality_metrics["accuracy"],
                precision_score=quality_metrics["precision"],
                recall_score=quality_metrics["recall"],
                f1_score=quality_metrics["f1"],
                execution_time=execution_time,
                memory_usage=0.0,
                additional_metrics={
                    "solution_confidence": solution.confidence,
                    "reasoning_steps": len(solution.reasoning_path),
                    "problem_complexity": problem.get("complexity", "medium"),
                    "description": problem["description"]
                },
                passed_threshold=quality_metrics["accuracy"] >= 0.80
            )
            
            results.append(result)
            self.results.append(result)
        
        return results
    
    async def run_integration_validation(self, integrated_controller: IntegratedReasoningController) -> List[ValidationResult]:
        """Validate integrated reasoning capabilities"""
        integration_scenarios = [
            {
                "name": "business_analytics",
                "query": """
                Analyze the relationship between marketing spend and revenue,
                predict next quarter performance, and recommend optimization strategies.
                """,
                "expected_systems": ["causal", "temporal", "analytical"],
                "quality_threshold": 0.75
            },
            {
                "name": "scientific_reasoning",
                "query": """
                Given experimental data showing correlation between variables A and B,
                determine if causation exists and predict outcomes of interventions.
                """,
                "expected_systems": ["causal", "analytical"],
                "quality_threshold": 0.80
            },
            {
                "name": "complex_optimization",
                "query": """
                Optimize resource allocation considering temporal constraints,
                causal dependencies, and uncertainty in demand forecasting.
                """,
                "expected_systems": ["temporal", "causal", "analytical"],
                "quality_threshold": 0.70
            }
        ]
        
        results = []
        
        for scenario in integration_scenarios:
            logger.info(f"Validating integration scenario: {scenario['name']}")
            
            start_time = asyncio.get_event_loop().time()
            
            # Execute integrated reasoning
            result = await integrated_controller.reason(
                query=scenario["query"],
                session_id=f"validation_{scenario['name']}"
            )
            
            execution_time = asyncio.get_event_loop().time() - start_time
            
            # Evaluate integration quality
            integration_metrics = self._evaluate_integration_quality(
                result,
                scenario
            )
            
            validation_result = ValidationResult(
                test_name=f"integration_{scenario['name']}",
                system_name="integrated_controller",
                accuracy_score=integration_metrics["accuracy"],
                precision_score=integration_metrics["precision"],
                recall_score=integration_metrics["recall"],
                f1_score=integration_metrics["f1"],
                execution_time=execution_time,
                memory_usage=0.0,
                additional_metrics={
                    "systems_used": len(integration_metrics.get("systems_engaged", [])),
                    "reasoning_depth": len(result.reasoning_path),
                    "context_utilization": integration_metrics.get("context_score", 0),
                    "coherence_score": integration_metrics.get("coherence", 0)
                },
                passed_threshold=integration_metrics["accuracy"] >= scenario["quality_threshold"]
            )
            
            results.append(validation_result)
            self.results.append(validation_result)
        
        return results
    
    def _evaluate_causal_accuracy(self, discovered_edges: List[Dict], ground_truth: List[Tuple]) -> Dict[str, float]:
        """Evaluate causal discovery accuracy"""
        # Convert discovered edges to set of tuples
        discovered_set = set()
        for edge in discovered_edges:
            discovered_set.add((edge['from'], edge['to']))
        
        ground_truth_set = set(ground_truth)
        
        # Calculate metrics
        true_positives = len(discovered_set.intersection(ground_truth_set))
        false_positives = len(discovered_set - ground_truth_set)
        false_negatives = len(ground_truth_set - discovered_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy as F1 score for causal discovery
        accuracy = f1
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives
        }
    
    def _evaluate_temporal_accuracy(self, patterns: List, anomalies: List, ground_truth: Dict) -> Dict[str, float]:
        """Evaluate temporal analysis accuracy"""
        scores = []
        
        # Evaluate trend detection
        trend_gt = ground_truth.get("trend", {})
        if trend_gt.get("exists", False):
            trend_patterns = [p for p in patterns if hasattr(p, 'pattern_type') and p.pattern_type == "trend"]
            trend_score = 1.0 if len(trend_patterns) > 0 else 0.0
        else:
            trend_patterns = [p for p in patterns if hasattr(p, 'pattern_type') and p.pattern_type == "trend"]
            trend_score = 1.0 if len(trend_patterns) == 0 else 0.5  # Penalize false positives
        scores.append(trend_score)
        
        # Evaluate seasonality detection
        seasonal_gt = ground_truth.get("seasonality", {})
        if seasonal_gt.get("exists", False):
            seasonal_patterns = [p for p in patterns if hasattr(p, 'pattern_type') and p.pattern_type == "seasonal"]
            seasonal_score = 1.0 if len(seasonal_patterns) > 0 else 0.0
        else:
            seasonal_patterns = [p for p in patterns if hasattr(p, 'pattern_type') and p.pattern_type == "seasonal"]
            seasonal_score = 1.0 if len(seasonal_patterns) == 0 else 0.5
        scores.append(seasonal_score)
        
        # Evaluate anomaly detection
        anomaly_gt = ground_truth.get("anomalies", {})
        expected_anomalies = anomaly_gt.get("count", 0)
        detected_count = len(anomalies)
        
        if expected_anomalies == 0:
            anomaly_score = 1.0 if detected_count == 0 else max(0, 1.0 - detected_count * 0.1)
        else:
            # Tolerance for anomaly detection (±20%)
            tolerance = max(1, expected_anomalies * 0.2)
            if abs(detected_count - expected_anomalies) <= tolerance:
                anomaly_score = 1.0
            else:
                anomaly_score = max(0, 1.0 - abs(detected_count - expected_anomalies) / expected_anomalies)
        
        scores.append(anomaly_score)
        
        # Overall accuracy
        accuracy = np.mean(scores)
        
        return {
            "accuracy": accuracy,
            "precision": accuracy,  # Simplified for temporal analysis
            "recall": accuracy,
            "f1": accuracy,
            "component_scores": {
                "trend": trend_score,
                "seasonality": seasonal_score,
                "anomalies": anomaly_score
            }
        }
    
    def _evaluate_reasoning_quality(self, solution, ground_truth: Dict) -> Dict[str, float]:
        """Evaluate general reasoning quality"""
        # This is a simplified evaluation - in practice would need more sophisticated NLP analysis
        
        solution_text = solution.final_answer.lower() if solution and solution.final_answer else ""
        
        scores = []
        
        # Check if solution contains expected elements
        if "solution" in ground_truth:
            gt_solution = str(ground_truth["solution"]).lower()
            # Simple keyword matching (in practice would use semantic similarity)
            keyword_matches = sum(1 for word in gt_solution.split() if word in solution_text)
            keyword_score = min(1.0, keyword_matches / len(gt_solution.split()))
            scores.append(keyword_score)
        
        # Check reasoning quality
        if hasattr(solution, 'reasoning_path') and solution.reasoning_path:
            reasoning_quality = min(1.0, len(solution.reasoning_path) / 3)  # Expect at least 3 reasoning steps
            scores.append(reasoning_quality)
        else:
            scores.append(0.0)
        
        # Check confidence alignment
        if hasattr(solution, 'confidence'):
            confidence_score = solution.confidence
            scores.append(confidence_score)
        else:
            scores.append(0.5)
        
        # Overall accuracy
        accuracy = np.mean(scores) if scores else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy,
            "component_scores": {
                "content_match": scores[0] if len(scores) > 0 else 0,
                "reasoning_depth": scores[1] if len(scores) > 1 else 0,
                "confidence": scores[2] if len(scores) > 2 else 0
            }
        }
    
    def _evaluate_integration_quality(self, result, scenario: Dict) -> Dict[str, float]:
        """Evaluate integrated reasoning quality"""
        scores = []
        
        # Check if expected systems were engaged
        if hasattr(result, 'reasoning_path') and result.reasoning_path:
            systems_used = set(step.system_type for step in result.reasoning_path if hasattr(step, 'system_type'))
            expected_systems = set(scenario.get("expected_systems", []))
            
            if expected_systems:
                system_coverage = len(systems_used.intersection(expected_systems)) / len(expected_systems)
            else:
                system_coverage = 1.0 if len(systems_used) > 1 else 0.5
            
            scores.append(system_coverage)
        else:
            scores.append(0.0)
        
        # Check answer quality (simplified)
        if result and result.final_answer:
            answer_length_score = min(1.0, len(result.final_answer.split()) / 50)  # Expect substantial answers
            scores.append(answer_length_score)
        else:
            scores.append(0.0)
        
        # Check coherence (simplified)
        if hasattr(result, 'reasoning_path') and len(result.reasoning_path) > 1:
            coherence_score = 0.8  # Placeholder - would need sophisticated coherence analysis
            scores.append(coherence_score)
        else:
            scores.append(0.5)
        
        accuracy = np.mean(scores)
        
        return {
            "accuracy": accuracy,
            "precision": accuracy,
            "recall": accuracy,
            "f1": accuracy,
            "systems_engaged": list(systems_used) if 'systems_used' in locals() else [],
            "context_score": scores[1] if len(scores) > 1 else 0,
            "coherence": scores[2] if len(scores) > 2 else 0
        }
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        if not self.results:
            return {"error": "No validation results available"}
        
        # Aggregate by system
        by_system = {}
        for result in self.results:
            system = result.system_name
            if system not in by_system:
                by_system[system] = []
            by_system[system].append(result)
        
        # System performance summary
        system_performance = {}
        for system, results in by_system.items():
            accuracies = [r.accuracy_score for r in results]
            execution_times = [r.execution_time for r in results]
            passed_tests = sum(1 for r in results if r.passed_threshold)
            
            system_performance[system] = {
                "total_tests": len(results),
                "passed_tests": passed_tests,
                "pass_rate": passed_tests / len(results),
                "avg_accuracy": np.mean(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "avg_execution_time": np.mean(execution_times),
                "total_execution_time": np.sum(execution_times)
            }
        
        # Overall summary
        all_accuracies = [r.accuracy_score for r in self.results]
        all_pass_rates = [1 if r.passed_threshold else 0 for r in self.results]
        
        overall_summary = {
            "total_validation_tests": len(self.results),
            "overall_pass_rate": np.mean(all_pass_rates),
            "overall_avg_accuracy": np.mean(all_accuracies),
            "systems_validated": len(by_system),
            "validation_timestamp": datetime.now().isoformat()
        }
        
        # Detailed results
        detailed_results = [
            {
                "test_name": r.test_name,
                "system": r.system_name,
                "accuracy": r.accuracy_score,
                "precision": r.precision_score,
                "recall": r.recall_score,
                "f1": r.f1_score,
                "execution_time": r.execution_time,
                "passed": r.passed_threshold,
                "additional_metrics": r.additional_metrics
            }
            for r in self.results
        ]
        
        return {
            "validation_summary": overall_summary,
            "system_performance": system_performance,
            "detailed_results": detailed_results,
            "recommendations": self._generate_recommendations(system_performance)
        }
    
    def _generate_recommendations(self, system_performance: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for system, perf in system_performance.items():
            if perf["pass_rate"] < 0.8:
                recommendations.append(
                    f"Improve {system} system reliability (pass rate: {perf['pass_rate']:.1%})"
                )
            
            if perf["avg_accuracy"] < 0.85:
                recommendations.append(
                    f"Optimize {system} system accuracy (current: {perf['avg_accuracy']:.1%})"
                )
            
            if perf["avg_execution_time"] > 5.0:
                recommendations.append(
                    f"Optimize {system} system performance (avg time: {perf['avg_execution_time']:.1f}s)"
                )
        
        if not recommendations:
            recommendations.append("All systems meeting validation thresholds - consider stress testing")
        
        return recommendations


@pytest.mark.asyncio
async def test_full_validation_suite():
    """Run complete validation suite"""
    try:
        # Initialize systems
        causal_engine = CausalReasoningEngine()
        await causal_engine.initialize()
        
        working_memory = WorkingMemorySystem()
        await working_memory.initialize()
        
        tree_of_thoughts = EnhancedTreeOfThoughts()
        await tree_of_thoughts.initialize()
        
        temporal_engine = TemporalReasoningEngine()
        await temporal_engine.initialize()
        
        integrated_controller = IntegratedReasoningController(
            causal_engine=causal_engine,
            working_memory=working_memory,
            tree_of_thoughts=tree_of_thoughts,
            temporal_engine=temporal_engine
        )
        await integrated_controller.initialize()
        
        # Run validation suite
        validation_suite = ValidationSuite()
        
        # Validate individual systems
        causal_results = await validation_suite.run_causal_validation(causal_engine)
        temporal_results = await validation_suite.run_temporal_validation(temporal_engine)
        reasoning_results = await validation_suite.run_reasoning_validation(tree_of_thoughts)
        integration_results = await validation_suite.run_integration_validation(integrated_controller)
        
        # Generate report
        report = validation_suite.generate_validation_report()
        
        # Assertions
        assert report["validation_summary"]["overall_pass_rate"] >= 0.75
        assert len(causal_results) >= 4  # Should test all causal datasets
        assert len(temporal_results) >= 3  # Should test all temporal datasets
        assert len(reasoning_results) >= 3  # Should test all reasoning problems
        assert len(integration_results) >= 3  # Should test all integration scenarios
        
        # Log results
        logger.info(f"Validation completed: {report['validation_summary']}")
        
        # Save report
        report_path = Path("test_reports/validation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
        
    except Exception as e:
        pytest.fail(f"Validation suite failed: {e}")


if __name__ == "__main__":
    # Run validation suite
    pytest.main([__file__, "-v", "--tb=short"])