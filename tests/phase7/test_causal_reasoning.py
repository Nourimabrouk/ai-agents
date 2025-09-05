"""
Phase 7 Causal Reasoning Testing Suite
Validates 90% accuracy target for causal relationship identification and intervention analysis
Tests causal inference, counterfactual reasoning, and do-calculus applications
"""

import asyncio
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import AsyncMock, Mock
from scipy import stats
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import networkx as nx
import random

# Import Phase 7 causal reasoning components
from core.reasoning.causal_inference import (
    CausalInferenceEngine, CausalRelationship, CausalDirection,
    InterventionType, CausalGraph, InterventionResult, CounterfactualAnalysis
)
from core.reasoning.temporal_reasoning import TemporalReasoningEngine, TemporalCausalModel
from core.reasoning.integrated_reasoning_controller import IntegratedReasoningController
from . import PHASE7_TEST_CONFIG


@pytest.fixture
def ground_truth_causal_data():
    """Generate ground truth causal data for testing accuracy"""
    # Create known causal relationships for testing
    # Structure: X -> Y -> Z with confounders
    np.random.seed(42)
    n_samples = 1000
    
    # Generate data with known causal structure
    confounder_c = np.random.normal(0, 1, n_samples)
    x = 0.5 * confounder_c + np.random.normal(0, 0.5, n_samples)
    y = 0.8 * x + 0.3 * confounder_c + np.random.normal(0, 0.3, n_samples) 
    z = 0.6 * y + np.random.normal(0, 0.4, n_samples)
    w = 0.7 * confounder_c + np.random.normal(0, 0.3, n_samples)  # Independent of x,y,z given c
    
    data = pd.DataFrame({
        'X': x,
        'Y': y, 
        'Z': z,
        'W': w,
        'C': confounder_c
    })
    
    # Ground truth causal relationships
    ground_truth = {
        ('X', 'Y'): {'strength': 0.8, 'direction': 'X->Y', 'exists': True},
        ('Y', 'Z'): {'strength': 0.6, 'direction': 'Y->Z', 'exists': True},
        ('C', 'X'): {'strength': 0.5, 'direction': 'C->X', 'exists': True},
        ('C', 'Y'): {'strength': 0.3, 'direction': 'C->Y', 'exists': True},
        ('C', 'W'): {'strength': 0.7, 'direction': 'C->W', 'exists': True},
        ('X', 'Z'): {'strength': 0.0, 'direction': 'none', 'exists': False},  # Mediated through Y
        ('W', 'X'): {'strength': 0.0, 'direction': 'none', 'exists': False},  # Independent
        ('W', 'Y'): {'strength': 0.0, 'direction': 'none', 'exists': False},  # Independent  
        ('W', 'Z'): {'strength': 0.0, 'direction': 'none', 'exists': False}   # Independent
    }
    
    return data, ground_truth


@pytest.fixture
def complex_causal_scenario():
    """Generate complex causal scenario with multiple pathways"""
    np.random.seed(123)
    n_samples = 1500
    
    # Complex network: Marketing -> Sales -> Revenue
    #                  Weather -> Sales
    #                  Competition -> Sales (negative)
    #                  Season -> Marketing, Weather
    
    season = np.random.choice([0, 1, 2, 3], n_samples)  # 4 seasons
    marketing = 0.6 * season + np.random.normal(0, 0.4, n_samples)
    weather = 0.4 * season + np.random.normal(0, 0.5, n_samples)
    competition = np.random.normal(0, 1, n_samples)
    
    sales = (0.7 * marketing + 0.3 * weather - 0.4 * competition + 
             np.random.normal(0, 0.3, n_samples))
    revenue = 0.9 * sales + np.random.normal(0, 0.2, n_samples)
    
    data = pd.DataFrame({
        'Season': season,
        'Marketing': marketing,
        'Weather': weather, 
        'Competition': competition,
        'Sales': sales,
        'Revenue': revenue
    })
    
    # Ground truth for complex scenario
    ground_truth = {
        ('Season', 'Marketing'): {'strength': 0.6, 'exists': True},
        ('Season', 'Weather'): {'strength': 0.4, 'exists': True},
        ('Marketing', 'Sales'): {'strength': 0.7, 'exists': True},
        ('Weather', 'Sales'): {'strength': 0.3, 'exists': True},
        ('Competition', 'Sales'): {'strength': -0.4, 'exists': True},
        ('Sales', 'Revenue'): {'strength': 0.9, 'exists': True},
        # Indirect relationships
        ('Marketing', 'Revenue'): {'strength': 0.63, 'exists': True, 'mediated': True},
        ('Season', 'Sales'): {'strength': 0.54, 'exists': True, 'mediated': True}
    }
    
    return data, ground_truth


class TestCausalRelationshipDetection:
    """Test causal relationship identification accuracy"""
    
    @pytest.fixture
    async def causal_engine(self):
        """Create causal inference engine"""
        engine = CausalInferenceEngine(
            accuracy_target=0.90,
            confidence_threshold=0.80,
            discovery_methods=['pc_algorithm', 'ges', 'fci', 'causal_discovery_toolbox'],
            validation_enabled=True
        )
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_basic_causal_discovery(self, causal_engine, ground_truth_causal_data):
        """Test basic causal relationship discovery accuracy"""
        data, ground_truth = ground_truth_causal_data
        engine = causal_engine
        
        # Discover causal relationships
        discovered_graph = await engine.discover_causal_structure(data)
        
        # Evaluate against ground truth
        accuracy_metrics = await engine.evaluate_discovery_accuracy(
            discovered_graph, ground_truth
        )
        
        target_accuracy = PHASE7_TEST_CONFIG["performance_targets"]["causal_reasoning_accuracy"]
        
        # Core accuracy assertions
        assert accuracy_metrics['relationship_accuracy'] >= target_accuracy
        assert accuracy_metrics['direction_accuracy'] >= 0.85  # Slightly lower threshold for direction
        assert accuracy_metrics['strength_correlation'] >= 0.80  # Strength estimation correlation
        
        print(f"Causal discovery accuracy: {accuracy_metrics['relationship_accuracy']:.1%}")
        print(f"Direction accuracy: {accuracy_metrics['direction_accuracy']:.1%}")
        print(f"Strength correlation: {accuracy_metrics['strength_correlation']:.3f}")
        
        # Test specific relationships
        discovered_relationships = discovered_graph.get_relationships()
        
        # Should discover X -> Y relationship
        xy_relationship = next((r for r in discovered_relationships 
                              if r.cause_variable == 'X' and r.effect_variable == 'Y'), None)
        assert xy_relationship is not None
        assert xy_relationship.confidence >= 0.8
        assert abs(xy_relationship.strength - 0.8) < 0.2  # Within reasonable range
        
        # Should discover Y -> Z relationship  
        yz_relationship = next((r for r in discovered_relationships
                              if r.cause_variable == 'Y' and r.effect_variable == 'Z'), None)
        assert yz_relationship is not None
        assert yz_relationship.confidence >= 0.8
        
        # Should NOT discover spurious X -> Z relationship (mediated through Y)
        xz_direct = any(r.cause_variable == 'X' and r.effect_variable == 'Z' 
                       and r.relationship_type == 'direct'
                       for r in discovered_relationships)
        assert xz_direct == False  # Should be identified as mediated
        
    @pytest.mark.asyncio
    async def test_complex_causal_network(self, causal_engine, complex_causal_scenario):
        """Test causal discovery in complex multi-variable scenarios"""
        data, ground_truth = complex_causal_scenario
        engine = causal_engine
        
        # Discover complex causal network
        complex_graph = await engine.discover_causal_structure(
            data, 
            max_variables=6,
            include_latent_variables=True
        )
        
        # Evaluate complex network accuracy
        complex_metrics = await engine.evaluate_discovery_accuracy(
            complex_graph, ground_truth
        )
        
        # Complex scenarios may have slightly lower accuracy
        assert complex_metrics['relationship_accuracy'] >= 0.85  # 85% for complex scenarios
        assert complex_metrics['network_structure_score'] >= 0.80
        
        # Test specific complex relationships
        relationships = complex_graph.get_relationships()
        
        # Should discover marketing -> sales -> revenue chain
        marketing_sales = next((r for r in relationships 
                               if r.cause_variable == 'Marketing' and r.effect_variable == 'Sales'), None)
        assert marketing_sales is not None
        assert marketing_sales.strength > 0.5
        
        sales_revenue = next((r for r in relationships
                            if r.cause_variable == 'Sales' and r.effect_variable == 'Revenue'), None)
        assert sales_revenue is not None
        assert sales_revenue.strength > 0.8
        
        # Should identify negative relationship: competition -> sales
        competition_sales = next((r for r in relationships
                                 if r.cause_variable == 'Competition' and r.effect_variable == 'Sales'), None)
        assert competition_sales is not None
        assert competition_sales.strength < 0  # Negative relationship
        
        print(f"Complex network accuracy: {complex_metrics['relationship_accuracy']:.1%}")
        print(f"Network structure score: {complex_metrics['network_structure_score']:.3f}")
        
    @pytest.mark.asyncio
    async def test_confounding_variable_detection(self, causal_engine, ground_truth_causal_data):
        """Test detection and handling of confounding variables"""
        data, ground_truth = ground_truth_causal_data
        engine = causal_engine
        
        # Test with and without confounding variable
        data_without_confounder = data[['X', 'Y', 'Z', 'W']].copy()  # Remove 'C'
        data_with_confounder = data.copy()
        
        # Discover without confounder (should get biased results)
        graph_biased = await engine.discover_causal_structure(data_without_confounder)
        
        # Discover with confounder (should get correct results)
        graph_unbiased = await engine.discover_causal_structure(data_with_confounder)
        
        # Compare results
        biased_metrics = await engine.evaluate_discovery_accuracy(graph_biased, ground_truth)
        unbiased_metrics = await engine.evaluate_discovery_accuracy(graph_unbiased, ground_truth)
        
        # Unbiased should be significantly more accurate
        accuracy_improvement = (unbiased_metrics['relationship_accuracy'] - 
                              biased_metrics['relationship_accuracy'])
        
        assert accuracy_improvement > 0.10, "Should see >10% accuracy improvement with confounder"
        assert unbiased_metrics['confounding_detection_score'] > 0.80
        
        # Should identify C as a confounder
        confounders = graph_unbiased.get_confounding_variables()
        assert 'C' in confounders
        
        confounding_relationships = graph_unbiased.get_confounding_relationships()
        c_confounds_xy = any(conf['confounder'] == 'C' and 
                           conf['confounded_relationship'] == ('X', 'Y')
                           for conf in confounding_relationships)
        assert c_confounds_xy == True
        
        print(f"Accuracy without confounder: {biased_metrics['relationship_accuracy']:.1%}")
        print(f"Accuracy with confounder: {unbiased_metrics['relationship_accuracy']:.1%}")
        print(f"Improvement: {accuracy_improvement:.1%}")
        
    @pytest.mark.asyncio
    async def test_temporal_causal_relationships(self, causal_engine):
        """Test identification of temporal causal relationships"""
        # Generate temporal data with known lag relationships
        n_timepoints = 500
        np.random.seed(456)
        
        # X at time t affects Y at time t+1, Y at time t affects Z at time t+2
        x_series = np.random.normal(0, 1, n_timepoints)
        y_series = np.zeros(n_timepoints)
        z_series = np.zeros(n_timepoints)
        
        for t in range(1, n_timepoints):
            y_series[t] = 0.6 * x_series[t-1] + np.random.normal(0, 0.3)
            
        for t in range(2, n_timepoints):
            z_series[t] = 0.7 * y_series[t-2] + np.random.normal(0, 0.2)
        
        temporal_data = pd.DataFrame({
            'X': x_series,
            'Y': y_series,
            'Z': z_series,
            'time': range(n_timepoints)
        })
        
        # Discover temporal causal relationships
        temporal_graph = await engine.discover_temporal_causal_structure(
            temporal_data, 
            time_column='time',
            max_lag=3
        )
        
        # Verify temporal relationships
        temporal_relationships = temporal_graph.get_temporal_relationships()
        
        # Should find X(t-1) -> Y(t)
        x_to_y = next((r for r in temporal_relationships 
                      if r.cause_variable == 'X' and r.effect_variable == 'Y' and r.lag == 1), None)
        assert x_to_y is not None
        assert x_to_y.confidence > 0.8
        assert abs(x_to_y.strength - 0.6) < 0.2
        
        # Should find Y(t-2) -> Z(t)
        y_to_z = next((r for r in temporal_relationships
                      if r.cause_variable == 'Y' and r.effect_variable == 'Z' and r.lag == 2), None)
        assert y_to_z is not None
        assert y_to_z.confidence > 0.8
        assert abs(y_to_z.strength - 0.7) < 0.2
        
        print(f"Found {len(temporal_relationships)} temporal relationships")


class TestCausalIntervention:
    """Test causal intervention analysis and do-calculus"""
    
    @pytest.fixture
    async def intervention_engine(self):
        """Create causal intervention engine"""
        engine = CausalInferenceEngine(
            intervention_methods=['do_calculus', 'backdoor_adjustment', 'front_door_adjustment'],
            counterfactual_enabled=True
        )
        await engine.initialize()
        return engine
    
    @pytest.mark.asyncio
    async def test_do_calculus_interventions(self, intervention_engine, ground_truth_causal_data):
        """Test do-calculus interventions for causal effect estimation"""
        data, ground_truth = ground_truth_causal_data
        engine = intervention_engine
        
        # First discover the causal structure
        causal_graph = await engine.discover_causal_structure(data)
        
        # Test intervention: do(X = x_value)
        intervention_result = await engine.perform_intervention(
            causal_graph=causal_graph,
            intervention_variable='X',
            intervention_value=1.0,
            target_variables=['Y', 'Z'],
            method=InterventionType.DO_INTERVENTION
        )
        
        assert intervention_result.intervention_successful == True
        assert 'Y' in intervention_result.effects
        assert 'Z' in intervention_result.effects
        
        # Verify causal effect strength
        # Expected: X=1.0 should cause Y to increase by ~0.8, Z by ~0.48 (0.8 * 0.6)
        y_effect = intervention_result.effects['Y']
        z_effect = intervention_result.effects['Z'] 
        
        assert abs(y_effect.causal_effect - 0.8) < 0.3  # Within reasonable error
        assert abs(z_effect.causal_effect - 0.48) < 0.3  # Mediated effect through Y
        
        # Test confidence intervals
        assert y_effect.confidence_interval[0] < y_effect.causal_effect < y_effect.confidence_interval[1]
        assert y_effect.p_value < 0.05  # Statistically significant
        
        print(f"Intervention do(X=1.0) effects:")
        print(f"  Y effect: {y_effect.causal_effect:.3f} ± {y_effect.standard_error:.3f}")
        print(f"  Z effect: {z_effect.causal_effect:.3f} ± {z_effect.standard_error:.3f}")
        
    @pytest.mark.asyncio
    async def test_counterfactual_analysis(self, intervention_engine, ground_truth_causal_data):
        """Test counterfactual reasoning: What if X had been different?"""
        data, ground_truth = ground_truth_causal_data
        engine = intervention_engine
        
        causal_graph = await engine.discover_causal_structure(data)
        
        # Select specific instance for counterfactual analysis
        instance_index = 100
        observed_instance = data.iloc[instance_index].to_dict()
        
        # Counterfactual: What if X had been 2.0 instead of observed value?
        counterfactual_result = await engine.perform_counterfactual_analysis(
            causal_graph=causal_graph,
            observed_instance=observed_instance,
            counterfactual_interventions={'X': 2.0},
            target_variables=['Y', 'Z']
        )
        
        assert counterfactual_result.analysis_successful == True
        
        # Compare observed vs counterfactual outcomes
        observed_y = observed_instance['Y']
        counterfactual_y = counterfactual_result.counterfactual_outcomes['Y']
        
        observed_z = observed_instance['Z']  
        counterfactual_z = counterfactual_result.counterfactual_outcomes['Z']
        
        # Verify counterfactual makes sense given causal structure
        x_change = 2.0 - observed_instance['X']
        expected_y_change = x_change * 0.8  # Direct effect coefficient
        actual_y_change = counterfactual_y - observed_y
        
        assert abs(actual_y_change - expected_y_change) < 0.5  # Reasonable accuracy
        
        # Z should change proportionally through Y
        expected_z_change = expected_y_change * 0.6
        actual_z_change = counterfactual_z - observed_z
        assert abs(actual_z_change - expected_z_change) < 0.5
        
        print(f"Counterfactual analysis for instance {instance_index}:")
        print(f"  X: {observed_instance['X']:.2f} -> 2.0 (change: {x_change:.2f})")
        print(f"  Y: {observed_y:.2f} -> {counterfactual_y:.2f} (change: {actual_y_change:.2f})")
        print(f"  Z: {observed_z:.2f} -> {counterfactual_z:.2f} (change: {actual_z_change:.2f})")
        
    @pytest.mark.asyncio
    async def test_policy_intervention_simulation(self, intervention_engine, complex_causal_scenario):
        """Test policy intervention simulation for business scenarios"""
        data, ground_truth = complex_causal_scenario
        engine = intervention_engine
        
        causal_graph = await engine.discover_causal_structure(data)
        
        # Policy intervention: Increase marketing by 20%
        current_marketing_mean = data['Marketing'].mean()
        policy_marketing_value = current_marketing_mean * 1.2
        
        policy_result = await engine.simulate_policy_intervention(
            causal_graph=causal_graph,
            policy_interventions={'Marketing': policy_marketing_value},
            target_variables=['Sales', 'Revenue'],
            simulation_samples=1000
        )
        
        assert policy_result.simulation_successful == True
        
        # Expected effects based on causal structure:
        # Marketing increase should increase Sales by ~0.7 * increase
        # Sales increase should increase Revenue by ~0.9 * sales_increase
        
        marketing_increase = policy_marketing_value - current_marketing_mean
        expected_sales_increase = marketing_increase * 0.7
        expected_revenue_increase = expected_sales_increase * 0.9
        
        actual_sales_increase = policy_result.policy_effects['Sales'].mean_effect
        actual_revenue_increase = policy_result.policy_effects['Revenue'].mean_effect
        
        # Verify policy effects within reasonable bounds
        assert abs(actual_sales_increase - expected_sales_increase) < expected_sales_increase * 0.3
        assert abs(actual_revenue_increase - expected_revenue_increase) < expected_revenue_increase * 0.3
        
        # Verify statistical significance
        assert policy_result.policy_effects['Sales'].p_value < 0.05
        assert policy_result.policy_effects['Revenue'].p_value < 0.05
        
        print(f"Policy intervention: Increase marketing by {marketing_increase:.2f}")
        print(f"  Expected sales increase: {expected_sales_increase:.2f}")
        print(f"  Actual sales increase: {actual_sales_increase:.2f} ± {policy_result.policy_effects['Sales'].std_error:.2f}")
        print(f"  Expected revenue increase: {expected_revenue_increase:.2f}")
        print(f"  Actual revenue increase: {actual_revenue_increase:.2f} ± {policy_result.policy_effects['Revenue'].std_error:.2f}")
        
    @pytest.mark.asyncio
    async def test_causal_effect_attribution(self, intervention_engine, complex_causal_scenario):
        """Test attribution of effects to different causal pathways"""
        data, ground_truth = complex_causal_scenario
        engine = intervention_engine
        
        causal_graph = await engine.discover_causal_structure(data)
        
        # Analyze what contributes to Revenue variation
        attribution_result = await engine.analyze_causal_attribution(
            causal_graph=causal_graph,
            target_variable='Revenue',
            potential_causes=['Marketing', 'Weather', 'Competition', 'Season'],
            attribution_method='shapley_values'
        )
        
        assert attribution_result.attribution_successful == True
        
        attributions = attribution_result.causal_attributions
        
        # Sales should be the primary direct cause of Revenue
        assert 'Sales' in attributions
        sales_attribution = attributions['Sales']
        assert sales_attribution.direct_effect > 0.7  # Strong direct effect
        
        # Marketing should have high indirect effect through Sales
        if 'Marketing' in attributions:
            marketing_attribution = attributions['Marketing']
            assert marketing_attribution.total_effect > marketing_attribution.direct_effect
            
        # Verify attribution sums make sense
        total_attribution = sum(attr.total_effect for attr in attributions.values())
        assert 0.8 < total_attribution < 1.2  # Should explain most of the variance
        
        print("Causal attribution to Revenue:")
        for cause, attribution in attributions.items():
            print(f"  {cause}: direct={attribution.direct_effect:.3f}, total={attribution.total_effect:.3f}")


class TestCausalReasoningIntegration:
    """Test integration of causal reasoning with other AI systems"""
    
    @pytest.fixture
    async def integrated_reasoning(self):
        """Create integrated reasoning controller with causal reasoning"""
        controller = IntegratedReasoningController(
            causal_reasoning_enabled=True,
            temporal_reasoning_enabled=True,
            working_memory_enabled=True
        )
        await controller.initialize()
        return controller
        
    @pytest.mark.asyncio
    async def test_causal_reasoning_accuracy_target(self, integrated_reasoning):
        """Test that integrated system meets 90% causal reasoning accuracy target"""
        controller = integrated_reasoning
        target_accuracy = PHASE7_TEST_CONFIG["performance_targets"]["causal_reasoning_accuracy"]
        
        # Generate multiple test scenarios
        test_scenarios = []
        for seed in range(5):  # 5 different scenarios
            np.random.seed(seed)
            scenario_data = self._generate_test_scenario(seed)
            test_scenarios.append(scenario_data)
        
        # Test causal reasoning on all scenarios
        accuracy_scores = []
        
        for scenario in test_scenarios:
            data, ground_truth = scenario
            
            # Discover causal relationships
            reasoning_result = await controller.perform_causal_analysis(data)
            
            # Evaluate accuracy
            accuracy = await controller.evaluate_causal_accuracy(
                reasoning_result.causal_graph, ground_truth
            )
            accuracy_scores.append(accuracy['overall_accuracy'])
        
        # Overall accuracy should meet target
        mean_accuracy = np.mean(accuracy_scores)
        assert mean_accuracy >= target_accuracy, f"Mean accuracy {mean_accuracy:.1%} below target {target_accuracy:.1%}"
        
        # At least 80% of scenarios should meet the target
        scenarios_meeting_target = sum(1 for acc in accuracy_scores if acc >= target_accuracy)
        assert scenarios_meeting_target >= 4, f"Only {scenarios_meeting_target}/5 scenarios met accuracy target"
        
        print(f"Causal reasoning accuracy across scenarios:")
        for i, accuracy in enumerate(accuracy_scores):
            print(f"  Scenario {i+1}: {accuracy:.1%}")
        print(f"  Mean accuracy: {mean_accuracy:.1%}")
        print(f"  Scenarios meeting target: {scenarios_meeting_target}/5")
        
    @pytest.mark.asyncio
    async def test_real_time_causal_inference(self, integrated_reasoning):
        """Test real-time causal inference performance"""
        controller = integrated_reasoning
        
        # Stream of data points for real-time analysis
        data_stream = self._generate_streaming_data(1000)
        
        processing_times = []
        accuracy_over_time = []
        
        batch_size = 50
        for i in range(0, len(data_stream), batch_size):
            batch = data_stream[i:i+batch_size]
            
            start_time = datetime.now()
            
            # Process batch in real-time
            causal_update = await controller.update_causal_model_streaming(batch)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            processing_times.append(processing_time)
            
            # Evaluate current model accuracy
            if i >= 200:  # After sufficient data
                current_accuracy = await controller.evaluate_current_causal_accuracy()
                accuracy_over_time.append(current_accuracy)
        
        # Performance requirements for real-time processing
        avg_processing_time = np.mean(processing_times)
        assert avg_processing_time < 1.0, f"Average processing time {avg_processing_time:.2f}s too slow"
        
        # Accuracy should improve and stabilize over time
        if len(accuracy_over_time) >= 3:
            final_accuracy = accuracy_over_time[-1]
            assert final_accuracy >= 0.85, f"Final streaming accuracy {final_accuracy:.1%} too low"
            
            # Should show learning/improvement
            accuracy_trend = np.polyfit(range(len(accuracy_over_time)), accuracy_over_time, 1)[0]
            assert accuracy_trend >= -0.01, "Accuracy should not degrade significantly over time"
        
        print(f"Real-time causal inference:")
        print(f"  Average processing time: {avg_processing_time*1000:.1f}ms per batch")
        print(f"  Final accuracy: {accuracy_over_time[-1]:.1%}" if accuracy_over_time else "N/A")
        
    def _generate_test_scenario(self, seed: int) -> Tuple[pd.DataFrame, Dict]:
        """Generate a test scenario with known ground truth"""
        np.random.seed(seed)
        n_samples = 800
        
        if seed == 0:
            # Simple chain: A -> B -> C
            a = np.random.normal(0, 1, n_samples)
            b = 0.7 * a + np.random.normal(0, 0.3, n_samples)
            c = 0.8 * b + np.random.normal(0, 0.2, n_samples)
            data = pd.DataFrame({'A': a, 'B': b, 'C': c})
            ground_truth = {('A', 'B'): {'strength': 0.7, 'exists': True},
                          ('B', 'C'): {'strength': 0.8, 'exists': True},
                          ('A', 'C'): {'strength': 0.0, 'exists': False}}
                          
        elif seed == 1:
            # Fork: A -> B, A -> C
            a = np.random.normal(0, 1, n_samples)
            b = 0.6 * a + np.random.normal(0, 0.4, n_samples)
            c = 0.5 * a + np.random.normal(0, 0.5, n_samples)
            data = pd.DataFrame({'A': a, 'B': b, 'C': c})
            ground_truth = {('A', 'B'): {'strength': 0.6, 'exists': True},
                          ('A', 'C'): {'strength': 0.5, 'exists': True},
                          ('B', 'C'): {'strength': 0.0, 'exists': False}}
                          
        elif seed == 2:
            # Collider: A -> C, B -> C
            a = np.random.normal(0, 1, n_samples)
            b = np.random.normal(0, 1, n_samples)
            c = 0.4 * a + 0.6 * b + np.random.normal(0, 0.3, n_samples)
            data = pd.DataFrame({'A': a, 'B': b, 'C': c})
            ground_truth = {('A', 'C'): {'strength': 0.4, 'exists': True},
                          ('B', 'C'): {'strength': 0.6, 'exists': True},
                          ('A', 'B'): {'strength': 0.0, 'exists': False}}
                          
        elif seed == 3:
            # Confounded: A <- Z -> B -> C
            z = np.random.normal(0, 1, n_samples)
            a = 0.5 * z + np.random.normal(0, 0.5, n_samples)
            b = 0.4 * z + np.random.normal(0, 0.5, n_samples)
            c = 0.7 * b + np.random.normal(0, 0.3, n_samples)
            data = pd.DataFrame({'A': a, 'B': b, 'C': c, 'Z': z})
            ground_truth = {('Z', 'A'): {'strength': 0.5, 'exists': True},
                          ('Z', 'B'): {'strength': 0.4, 'exists': True},
                          ('B', 'C'): {'strength': 0.7, 'exists': True},
                          ('A', 'B'): {'strength': 0.0, 'exists': False}}
        else:
            # Complex network
            z1 = np.random.normal(0, 1, n_samples)
            z2 = np.random.normal(0, 1, n_samples)
            a = 0.3 * z1 + np.random.normal(0, 0.4, n_samples)
            b = 0.5 * a + 0.2 * z2 + np.random.normal(0, 0.3, n_samples)
            c = 0.6 * b + 0.4 * z1 + np.random.normal(0, 0.2, n_samples)
            d = 0.3 * c + np.random.normal(0, 0.4, n_samples)
            data = pd.DataFrame({'A': a, 'B': b, 'C': c, 'D': d, 'Z1': z1, 'Z2': z2})
            ground_truth = {('Z1', 'A'): {'strength': 0.3, 'exists': True},
                          ('A', 'B'): {'strength': 0.5, 'exists': True},
                          ('Z2', 'B'): {'strength': 0.2, 'exists': True},
                          ('B', 'C'): {'strength': 0.6, 'exists': True},
                          ('Z1', 'C'): {'strength': 0.4, 'exists': True},
                          ('C', 'D'): {'strength': 0.3, 'exists': True}}
        
        return data, ground_truth
        
    def _generate_streaming_data(self, n_points: int) -> List[Dict]:
        """Generate streaming data points"""
        np.random.seed(789)
        data_points = []
        
        for i in range(n_points):
            # Simple streaming relationship that changes over time
            time_factor = i / n_points
            
            x = np.random.normal(0, 1)
            # Relationship strength changes over time
            strength = 0.3 + 0.4 * time_factor
            y = strength * x + np.random.normal(0, 0.2)
            z = 0.5 * y + np.random.normal(0, 0.3)
            
            data_points.append({
                'timestamp': i,
                'X': x,
                'Y': y, 
                'Z': z
            })
            
        return data_points


if __name__ == "__main__":
    # Run causal reasoning tests
    pytest.main([__file__, "-v", "--tb=short"])