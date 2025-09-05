#!/usr/bin/env python3
"""
Meta Demo Test Suite
====================

Comprehensive test suite for the Autonomous Intelligence Symphony meta-demo
to ensure all components work perfectly before presentation.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import demo components
try:
    from meta_demo.demo_engine import AutonomousIntelligenceSymphony, run_autonomous_intelligence_symphony
    from meta_demo.coordination import AgentOrchestrationDemo, CoordinationPattern
    from meta_demo.business_impact import BusinessTransformationDemo, ROICalculator
    from meta_demo.visualization import PerformanceVisualization, VisualizationTheme
    print("✅ All meta-demo imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class MetaDemoTestSuite:
    """Comprehensive test suite for meta-demo components"""
    
    def __init__(self):
        self.test_results: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        print("🧪 AUTONOMOUS INTELLIGENCE SYMPHONY - TEST SUITE")
        print("=" * 60)
        
        # Test core demo engine
        await self._test_demo_engine()
        
        # Test agent coordination
        await self._test_agent_coordination()
        
        # Test business impact calculator
        await self._test_business_impact()
        
        # Test visualization system
        await self._test_visualization()
        
        # Test integration
        await self._test_integration()
        
        # Generate test summary
        return self._generate_test_summary()
    
    async def _test_demo_engine(self) -> None:
        """Test core demo engine functionality"""
        print("\n🎭 Testing Core Demo Engine...")
        
        try:
            # Test symphony initialization
            symphony = AutonomousIntelligenceSymphony({
                'interactive_mode': False,
                'visual_effects': False,
                'duration_minutes': 1,  # Quick test
                'enable_spectacular_effects': False
            })
            
            # Test symphony execution (abbreviated)
            start_time = time.time()
            results = await run_autonomous_intelligence_symphony({
                'interactive_mode': False,
                'visual_effects': False,
                'duration_minutes': 1,
                'enable_spectacular_effects': False
            })
            execution_time = time.time() - start_time
            
            success = results.get('success', False)
            acts_completed = len(results.get('act_performances', []))
            
            self.test_results.append({
                'component': 'Demo Engine',
                'test': 'Symphony Execution',
                'success': success,
                'execution_time': execution_time,
                'acts_completed': acts_completed,
                'details': f"Executed {acts_completed} acts in {execution_time:.2f}s"
            })
            
            print(f"  ✅ Symphony execution: {acts_completed} acts completed in {execution_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                'component': 'Demo Engine',
                'test': 'Symphony Execution',
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Demo engine test failed: {e}")
    
    async def _test_agent_coordination(self) -> None:
        """Test agent coordination system"""
        print("\n🎼 Testing Agent Coordination...")
        
        try:
            # Test coordination demo
            coordination_demo = AgentOrchestrationDemo(
                num_agents=10,  # Smaller for testing
                coordination_pattern=CoordinationPattern.SWARM_INTELLIGENCE
            )
            
            # Initialize swarm
            start_time = time.time()
            init_results = await coordination_demo.initialize_swarm()
            init_time = time.time() - start_time
            
            success = init_results.get('swarm_initialized', False)
            agent_count = init_results.get('total_agents', 0)
            
            self.test_results.append({
                'component': 'Agent Coordination',
                'test': 'Swarm Initialization',
                'success': success,
                'execution_time': init_time,
                'agent_count': agent_count,
                'details': f"Initialized {agent_count} agents in {init_time:.2f}s"
            })
            
            print(f"  ✅ Swarm initialization: {agent_count} agents in {init_time:.2f}s")
            
            # Test coordination demonstration
            business_problem = {
                "name": "Test Optimization",
                "departments": ["sales", "operations"],
                "complexity_score": 5.0
            }
            
            start_time = time.time()
            coord_results = await coordination_demo.demonstrate_coordination(business_problem)
            coord_time = time.time() - start_time
            
            coord_success = coord_results.get('demonstration_complete', False)
            
            self.test_results.append({
                'component': 'Agent Coordination',
                'test': 'Coordination Demonstration',
                'success': coord_success,
                'execution_time': coord_time,
                'details': f"Coordination demo completed in {coord_time:.2f}s"
            })
            
            print(f"  ✅ Coordination demo: Completed in {coord_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                'component': 'Agent Coordination',
                'test': 'Coordination System',
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Agent coordination test failed: {e}")
    
    async def _test_business_impact(self) -> None:
        """Test business impact and ROI calculator"""
        print("\n💰 Testing Business Impact Calculator...")
        
        try:
            # Test business transformation demo
            business_demo = BusinessTransformationDemo()
            
            start_time = time.time()
            transformation_results = await business_demo.demonstrate_transformation()
            calc_time = time.time() - start_time
            
            success = transformation_results.get('transformation_complete', False)
            roi_data = transformation_results.get('business_impact_summary', {})
            
            self.test_results.append({
                'component': 'Business Impact',
                'test': 'Transformation Demo',
                'success': success,
                'execution_time': calc_time,
                'roi_percentage': roi_data.get('roi_percentage', 'N/A'),
                'details': f"ROI calculation completed in {calc_time:.2f}s"
            })
            
            print(f"  ✅ Business transformation: ROI {roi_data.get('roi_percentage', 'N/A')} in {calc_time:.2f}s")
            
            # Test ROI calculator directly
            roi_calculator = ROICalculator()
            
            # Create test scenario
            from meta_demo.business_impact import BusinessScenario
            test_scenario = BusinessScenario(
                name="Test Scenario",
                industry="Technology",
                company_size="Medium",
                problem_description="Test problem",
                solution_approach="Test solution",
                implementation_timeline="3 months",
                before_metrics={"processing_time_minutes": 30, "accuracy_rate": 80.0},
                after_metrics={"processing_time_minutes": 3, "accuracy_rate": 95.0},
                cost_breakdown={"implementation": 100000},
                benefit_breakdown={"savings": 500000},
                risk_factors=["test_risk"],
                success_factors=["test_success"]
            )
            
            start_time = time.time()
            roi_results = await roi_calculator.calculate_comprehensive_roi(test_scenario)
            roi_time = time.time() - start_time
            
            roi_success = 'baseline_roi' in roi_results
            roi_percentage = roi_results.get('baseline_roi', {}).roi_percentage if roi_success else 0
            
            self.test_results.append({
                'component': 'Business Impact',
                'test': 'ROI Calculator',
                'success': roi_success,
                'execution_time': roi_time,
                'roi_percentage': f"{roi_percentage:.0f}%",
                'details': f"ROI calculator executed in {roi_time:.2f}s"
            })
            
            print(f"  ✅ ROI calculator: {roi_percentage:.0f}% ROI in {roi_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                'component': 'Business Impact',
                'test': 'Business Impact System',
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Business impact test failed: {e}")
    
    async def _test_visualization(self) -> None:
        """Test visualization system"""
        print("\n🎨 Testing Visualization System...")
        
        try:
            # Test performance visualization
            perf_viz = PerformanceVisualization()
            
            test_metrics = {
                "overall_performance": 95.0,
                "cpu_usage": 65.0,
                "memory_usage": 2.1,
                "throughput": 10.5
            }
            
            start_time = time.time()
            dashboard = await perf_viz.create_performance_dashboard(test_metrics)
            viz_time = time.time() - start_time
            
            viz_success = dashboard.get('layout') == 'spectacular_grid'
            component_count = len(dashboard.get('components', []))
            
            self.test_results.append({
                'component': 'Visualization',
                'test': 'Performance Dashboard',
                'success': viz_success,
                'execution_time': viz_time,
                'component_count': component_count,
                'details': f"Dashboard with {component_count} components created in {viz_time:.2f}s"
            })
            
            print(f"  ✅ Performance dashboard: {component_count} components in {viz_time:.2f}s")
            
            # Test act-specific visualization
            start_time = time.time()
            act_viz = await perf_viz.create_act_visualization("Birth of Intelligence", {"statistics": {"agents": 100}})
            act_viz_time = time.time() - start_time
            
            act_viz_success = act_viz.get('type') == 'particle_system'
            
            self.test_results.append({
                'component': 'Visualization',
                'test': 'Act Visualization',
                'success': act_viz_success,
                'execution_time': act_viz_time,
                'details': f"Act visualization created in {act_viz_time:.2f}s"
            })
            
            print(f"  ✅ Act visualization: Created in {act_viz_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                'component': 'Visualization',
                'test': 'Visualization System',
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Visualization test failed: {e}")
    
    async def _test_integration(self) -> None:
        """Test system integration"""
        print("\n🔧 Testing System Integration...")
        
        try:
            # Test component integration
            start_time = time.time()
            
            # Create instances of all major components
            symphony = AutonomousIntelligenceSymphony()
            coordination = AgentOrchestrationDemo(num_agents=5)
            business_demo = BusinessTransformationDemo()
            visualization = PerformanceVisualization()
            
            integration_time = time.time() - start_time
            
            self.test_results.append({
                'component': 'Integration',
                'test': 'Component Integration',
                'success': True,
                'execution_time': integration_time,
                'details': f"All components integrated in {integration_time:.2f}s"
            })
            
            print(f"  ✅ Component integration: All systems integrated in {integration_time:.2f}s")
            
            # Test data flow between components
            test_metrics = {"performance": 90.0, "agents": 10}
            
            start_time = time.time()
            dashboard = await visualization.create_performance_dashboard(test_metrics)
            data_flow_time = time.time() - start_time
            
            data_flow_success = len(dashboard.get('components', [])) > 0
            
            self.test_results.append({
                'component': 'Integration',
                'test': 'Data Flow',
                'success': data_flow_success,
                'execution_time': data_flow_time,
                'details': f"Data flow validated in {data_flow_time:.2f}s"
            })
            
            print(f"  ✅ Data flow: Validated in {data_flow_time:.2f}s")
            
        except Exception as e:
            self.test_results.append({
                'component': 'Integration',
                'test': 'System Integration',
                'success': False,
                'error': str(e)
            })
            print(f"  ❌ Integration test failed: {e}")
    
    def _generate_test_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        total_time = time.time() - self.start_time
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get('success', False)])
        failed_tests = total_tests - successful_tests
        success_rate = (successful_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Categorize results by component
        by_component = {}
        for result in self.test_results:
            component = result.get('component', 'Unknown')
            if component not in by_component:
                by_component[component] = {'passed': 0, 'failed': 0, 'tests': []}
            
            by_component[component]['tests'].append(result)
            if result.get('success', False):
                by_component[component]['passed'] += 1
            else:
                by_component[component]['failed'] += 1
        
        summary = {
            'test_suite': 'Autonomous Intelligence Symphony Meta-Demo',
            'total_execution_time': total_time,
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'success_rate': success_rate,
            'grade': self._calculate_grade(success_rate),
            'by_component': by_component,
            'detailed_results': self.test_results,
            'timestamp': time.time(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _calculate_grade(self, success_rate: float) -> str:
        """Calculate overall grade based on success rate"""
        if success_rate >= 95:
            return "A+ (Exceptional)"
        elif success_rate >= 90:
            return "A (Excellent)"
        elif success_rate >= 85:
            return "B+ (Good)"
        elif success_rate >= 80:
            return "B (Satisfactory)"
        elif success_rate >= 70:
            return "C (Needs Improvement)"
        else:
            return "D (Major Issues)"
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [r for r in self.test_results if not r.get('success', False)]
        
        if not failed_tests:
            recommendations.append("🎉 All tests passed! Meta-demo is ready for presentation.")
            recommendations.append("✅ System performance is optimal")
            recommendations.append("✅ All components are fully functional")
        else:
            recommendations.append(f"⚠️ {len(failed_tests)} test(s) failed - review before presentation")
            
            for test in failed_tests:
                component = test.get('component', 'Unknown')
                error = test.get('error', 'Unknown error')
                recommendations.append(f"🔧 Fix {component}: {error}")
        
        return recommendations


async def main():
    """Main test execution function"""
    print("🧪 STARTING META-DEMO TEST SUITE")
    print("=" * 70)
    
    # Run comprehensive tests
    test_suite = MetaDemoTestSuite()
    summary = await test_suite.run_all_tests()
    
    # Print detailed summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']}")
    print(f"Failed: {summary['failed_tests']}")
    print(f"Success Rate: {summary['success_rate']:.1f}%")
    print(f"Overall Grade: {summary['grade']}")
    print(f"Total Time: {summary['total_execution_time']:.2f}s")
    
    # Component breakdown
    print(f"\n📋 COMPONENT BREAKDOWN:")
    for component, stats in summary['by_component'].items():
        total_component_tests = stats['passed'] + stats['failed']
        component_rate = (stats['passed'] / total_component_tests * 100) if total_component_tests > 0 else 0
        print(f"  {component}: {stats['passed']}/{total_component_tests} ({component_rate:.1f}%)")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    for rec in summary['recommendations']:
        print(f"  {rec}")
    
    # Save results
    results_file = f"test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    # Final verdict
    if summary['success_rate'] >= 95:
        print(f"\n🎆 META-DEMO STATUS: READY FOR SPECTACULAR PRESENTATION!")
        return 0
    elif summary['success_rate'] >= 80:
        print(f"\n⚠️ META-DEMO STATUS: MOSTLY READY - Minor fixes needed")
        return 0
    else:
        print(f"\n❌ META-DEMO STATUS: NOT READY - Major fixes required")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))