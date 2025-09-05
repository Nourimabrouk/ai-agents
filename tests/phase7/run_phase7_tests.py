"""
Phase 7 Autonomous Intelligence Ecosystem Test Runner
Comprehensive validation of all Phase 7 capabilities and targets
"""

import asyncio
import pytest
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass, asdict
import subprocess
import psutil
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from . import PHASE7_TEST_CONFIG


@dataclass
class TestSuiteResult:
    """Results from running a test suite"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    execution_time: float
    success_rate: float
    critical_failures: List[str]
    performance_metrics: Dict[str, Any]
    timestamp: datetime


@dataclass
class Phase7ValidationReport:
    """Complete Phase 7 validation report"""
    validation_timestamp: datetime
    overall_success: bool
    target_achievements: Dict[str, bool]
    suite_results: Dict[str, TestSuiteResult]
    performance_summary: Dict[str, Any]
    critical_issues: List[str]
    recommendations: List[str]
    system_metrics: Dict[str, Any]


class Phase7TestRunner:
    """Comprehensive test runner for Phase 7 validation"""
    
    def __init__(self, config_override: Dict[str, Any] = None):
        self.config = PHASE7_TEST_CONFIG.copy()
        if config_override:
            self.config.update(config_override)
        
        self.test_suites = {
            'autonomous_orchestration': 'tests/phase7/test_autonomous_orchestration.py',
            'performance': 'tests/phase7/test_performance.py', 
            'security': 'tests/phase7/test_security.py',
            'causal_reasoning': 'tests/phase7/test_causal_reasoning.py',
            'self_modification': 'tests/phase7/test_self_modification.py',
            'integration': 'tests/phase7/test_integration.py'
        }
        
        self.results = {}
        self.start_time = None
        self.system_metrics = {}
        
    async def run_comprehensive_validation(self) -> Phase7ValidationReport:
        """Run complete Phase 7 validation suite"""
        logger.info("🚀 Starting Phase 7 Autonomous Intelligence Ecosystem Validation")
        logger.info(f"Target Performance Metrics:")
        for metric, target in self.config['performance_targets'].items():
            logger.info(f"  {metric}: {target}")
        
        self.start_time = datetime.now()
        
        try:
            # Pre-validation system check
            await self._pre_validation_checks()
            
            # Run all test suites
            await self._run_all_test_suites()
            
            # Post-validation analysis
            validation_report = await self._generate_validation_report()
            
            # Save results
            await self._save_validation_results(validation_report)
            
            # Display summary
            self._display_validation_summary(validation_report)
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            raise
            
    async def _pre_validation_checks(self):
        """Perform pre-validation system checks"""
        logger.info("🔍 Performing pre-validation system checks...")
        
        # Check system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        logger.info(f"System Resources: {memory_gb:.1f}GB RAM, {cpu_count} CPU cores")
        
        # Verify test dependencies
        missing_deps = await self._check_dependencies()
        if missing_deps:
            logger.warning(f"Missing dependencies: {missing_deps}")
            
        # Initialize system metrics collection
        self.system_metrics['pre_validation'] = {
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'disk_usage_gb': psutil.disk_usage('.').used / (1024**3)
        }
        
    async def _check_dependencies(self) -> List[str]:
        """Check for missing test dependencies"""
        required_packages = [
            'numpy', 'pandas', 'scipy', 'scikit-learn', 
            'networkx', 'pytest', 'pytest-asyncio'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
                
        return missing
        
    async def _run_all_test_suites(self):
        """Run all Phase 7 test suites"""
        logger.info("🧪 Running Phase 7 test suites...")
        
        # Run test suites in logical order
        execution_order = [
            'autonomous_orchestration',
            'causal_reasoning', 
            'self_modification',
            'security',
            'performance',
            'integration'
        ]
        
        for suite_name in execution_order:
            if suite_name in self.test_suites:
                await self._run_test_suite(suite_name)
            
    async def _run_test_suite(self, suite_name: str):
        """Run individual test suite"""
        suite_path = self.test_suites[suite_name]
        logger.info(f"  Running {suite_name} tests...")
        
        start_time = time.perf_counter()
        
        try:
            # Run pytest on the test suite
            result = await self._execute_pytest(suite_path)
            
            execution_time = time.perf_counter() - start_time
            
            # Parse results
            suite_result = TestSuiteResult(
                suite_name=suite_name,
                total_tests=result.get('total', 0),
                passed_tests=result.get('passed', 0),
                failed_tests=result.get('failed', 0),
                skipped_tests=result.get('skipped', 0),
                execution_time=execution_time,
                success_rate=result.get('success_rate', 0.0),
                critical_failures=result.get('critical_failures', []),
                performance_metrics=result.get('performance_metrics', {}),
                timestamp=datetime.now()
            )
            
            self.results[suite_name] = suite_result
            
            logger.info(f"    {suite_name}: {suite_result.passed_tests}/{suite_result.total_tests} passed ({suite_result.success_rate:.1%}) in {execution_time:.2f}s")
            
            if suite_result.critical_failures:
                logger.warning(f"    Critical failures: {len(suite_result.critical_failures)}")
                
        except Exception as e:
            logger.error(f"    Failed to run {suite_name}: {e}")
            
            # Create failure result
            self.results[suite_name] = TestSuiteResult(
                suite_name=suite_name,
                total_tests=0,
                passed_tests=0,
                failed_tests=1,
                skipped_tests=0,
                execution_time=time.perf_counter() - start_time,
                success_rate=0.0,
                critical_failures=[f"Suite execution failed: {e}"],
                performance_metrics={},
                timestamp=datetime.now()
            )
            
    async def _execute_pytest(self, test_path: str) -> Dict[str, Any]:
        """Execute pytest and parse results"""
        try:
            # Run pytest with JSON report
            cmd = [
                sys.executable, '-m', 'pytest', 
                test_path,
                '-v',
                '--tb=short',
                '--json-report',
                '--json-report-file=temp_test_report.json'
            ]
            
            # Execute pytest
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=Path(__file__).parent.parent.parent
            )
            
            stdout, stderr = await process.communicate()
            
            # Parse JSON report if available
            report_path = Path(__file__).parent.parent.parent / 'temp_test_report.json'
            if report_path.exists():
                with open(report_path, 'r') as f:
                    report_data = json.load(f)
                    
                # Clean up temp file
                report_path.unlink()
                
                # Extract metrics
                summary = report_data.get('summary', {})
                
                return {
                    'total': summary.get('total', 0),
                    'passed': summary.get('passed', 0),
                    'failed': summary.get('failed', 0),
                    'skipped': summary.get('skipped', 0),
                    'success_rate': summary.get('passed', 0) / max(summary.get('total', 1), 1),
                    'critical_failures': self._extract_critical_failures(report_data),
                    'performance_metrics': self._extract_performance_metrics(report_data)
                }
            else:
                # Fallback parsing from stdout/stderr
                return self._parse_pytest_output(stdout.decode(), stderr.decode())
                
        except Exception as e:
            logger.error(f"Error executing pytest: {e}")
            return {
                'total': 0, 'passed': 0, 'failed': 1, 'skipped': 0,
                'success_rate': 0.0, 'critical_failures': [str(e)], 'performance_metrics': {}
            }
            
    def _extract_critical_failures(self, report_data: Dict) -> List[str]:
        """Extract critical failures from test report"""
        critical_failures = []
        
        for test in report_data.get('tests', []):
            if test.get('outcome') == 'failed':
                # Check if this is a critical failure
                test_name = test.get('nodeid', '')
                failure_message = test.get('call', {}).get('longrepr', '')
                
                # Define critical failure patterns
                critical_patterns = [
                    'performance_targets',
                    'autonomous_improvement',
                    'causal_reasoning_accuracy', 
                    'complex_task_success',
                    'security_violation',
                    'safety_violation'
                ]
                
                if any(pattern in test_name.lower() or pattern in failure_message.lower() 
                      for pattern in critical_patterns):
                    critical_failures.append(f"{test_name}: {failure_message}")
                    
        return critical_failures
        
    def _extract_performance_metrics(self, report_data: Dict) -> Dict[str, Any]:
        """Extract performance metrics from test report"""
        performance_metrics = {}
        
        for test in report_data.get('tests', []):
            test_name = test.get('nodeid', '')
            
            # Extract performance-related metrics
            if 'performance' in test_name.lower():
                duration = test.get('duration', 0)
                performance_metrics[test_name] = {
                    'duration': duration,
                    'outcome': test.get('outcome')
                }
                
        return performance_metrics
        
    def _parse_pytest_output(self, stdout: str, stderr: str) -> Dict[str, Any]:
        """Fallback parsing of pytest output"""
        # Basic parsing of pytest summary
        lines = stdout.split('\n')
        
        total = passed = failed = skipped = 0
        
        for line in lines:
            if 'passed' in line and 'failed' in line:
                # Try to extract numbers from summary line
                words = line.split()
                for i, word in enumerate(words):
                    if word == 'passed' and i > 0:
                        try:
                            passed = int(words[i-1])
                        except (ValueError, IndexError):
        logger.info(f'Method {function_name} called')
        return {}
                    elif word == 'failed' and i > 0:
                        try:
                            failed = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                    elif word == 'skipped' and i > 0:
                        try:
                            skipped = int(words[i-1])
                        except (ValueError, IndexError):
                            pass
                            
        total = passed + failed + skipped
        
        return {
            'total': total,
            'passed': passed, 
            'failed': failed,
            'skipped': skipped,
            'success_rate': passed / max(total, 1),
            'critical_failures': [],
            'performance_metrics': {}
        }
        
    async def _generate_validation_report(self) -> Phase7ValidationReport:
        """Generate comprehensive validation report"""
        logger.info("📊 Generating validation report...")
        
        # Calculate overall metrics
        total_tests = sum(r.total_tests for r in self.results.values())
        total_passed = sum(r.passed_tests for r in self.results.values())
        total_failed = sum(r.failed_tests for r in self.results.values())
        overall_success_rate = total_passed / max(total_tests, 1)
        
        # Check target achievements
        target_achievements = await self._evaluate_target_achievements()
        
        # Collect critical issues
        critical_issues = []
        for result in self.results.values():
            critical_issues.extend(result.critical_failures)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(target_achievements, critical_issues)
        
        # Collect system metrics
        self.system_metrics['post_validation'] = {
            'memory_usage_gb': psutil.virtual_memory().used / (1024**3),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'disk_usage_gb': psutil.disk_usage('.').used / (1024**3),
            'total_execution_time': (datetime.now() - self.start_time).total_seconds()
        }
        
        # Overall success determination
        overall_success = (
            overall_success_rate >= 0.90 and  # 90% of tests pass
            sum(target_achievements.values()) >= len(target_achievements) * 0.8 and  # 80% of targets met
            len(critical_issues) == 0  # No critical failures
        )
        
        return Phase7ValidationReport(
            validation_timestamp=self.start_time,
            overall_success=overall_success,
            target_achievements=target_achievements,
            suite_results=self.results,
            performance_summary={
                'total_tests': total_tests,
                'passed_tests': total_passed,
                'failed_tests': total_failed,
                'success_rate': overall_success_rate,
                'execution_time': self.system_metrics['post_validation']['total_execution_time']
            },
            critical_issues=critical_issues,
            recommendations=recommendations,
            system_metrics=self.system_metrics
        )
        
    async def _evaluate_target_achievements(self) -> Dict[str, bool]:
        """Evaluate whether performance targets were achieved"""
        target_achievements = {}
        
        # Define target evaluation logic
        targets = self.config['performance_targets']
        
        # Causal reasoning accuracy (90% target)
        causal_suite = self.results.get('causal_reasoning')
        if causal_suite:
            target_achievements['causal_reasoning_accuracy'] = (
                causal_suite.success_rate >= targets['causal_reasoning_accuracy']
            )
        else:
            target_achievements['causal_reasoning_accuracy'] = False
            
        # Autonomous improvement (15% target) 
        self_mod_suite = self.results.get('self_modification')
        if self_mod_suite:
            target_achievements['autonomous_improvement'] = (
                self_mod_suite.success_rate >= 0.80  # Proxy: 80% of improvement tests pass
            )
        else:
            target_achievements['autonomous_improvement'] = False
            
        # Complex task success (95% target)
        integration_suite = self.results.get('integration')
        if integration_suite:
            target_achievements['complex_task_success'] = (
                integration_suite.success_rate >= targets['complex_task_success']
            )
        else:
            target_achievements['complex_task_success'] = False
            
        # Performance targets (concurrent agents, response time)
        performance_suite = self.results.get('performance')
        if performance_suite:
            target_achievements['performance_scalability'] = (
                performance_suite.success_rate >= 0.85  # 85% of performance tests pass
            )
        else:
            target_achievements['performance_scalability'] = False
            
        # Security and safety
        security_suite = self.results.get('security')
        if security_suite:
            target_achievements['security_safety'] = (
                security_suite.success_rate >= 0.90  # 90% of security tests pass
            )
        else:
            target_achievements['security_safety'] = False
            
        # Autonomous orchestration
        orchestration_suite = self.results.get('autonomous_orchestration')
        if orchestration_suite:
            target_achievements['autonomous_orchestration'] = (
                orchestration_suite.success_rate >= 0.85
            )
        else:
            target_achievements['autonomous_orchestration'] = False
            
        return target_achievements
        
    async def _generate_recommendations(self, targets: Dict[str, bool], critical_issues: List[str]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Target-specific recommendations
        if not targets.get('causal_reasoning_accuracy', False):
            recommendations.append("Improve causal reasoning algorithms and training data quality")
            
        if not targets.get('autonomous_improvement', False):
            recommendations.append("Enhance self-modification safety validation and improvement measurement")
            
        if not targets.get('complex_task_success', False):
            recommendations.append("Optimize task decomposition and agent coordination for complex scenarios")
            
        if not targets.get('performance_scalability', False):
            recommendations.append("Implement performance optimizations for high-concurrency scenarios")
            
        if not targets.get('security_safety', False):
            recommendations.append("Strengthen security monitoring and anomaly detection systems")
            
        # Critical issue recommendations
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical failures before production deployment")
            
        # System-level recommendations
        memory_usage = self.system_metrics.get('post_validation', {}).get('memory_usage_gb', 0)
        if memory_usage > 8:  # High memory usage
            recommendations.append("Optimize memory usage - consider memory pooling and garbage collection tuning")
            
        return recommendations
        
    async def _save_validation_results(self, report: Phase7ValidationReport):
        """Save validation results to files"""
        timestamp = report.validation_timestamp.strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        report_path = Path(f"tests/phase7/validation_report_{timestamp}.json")
        with open(report_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
            
        # Save summary report
        summary_path = Path(f"tests/phase7/validation_summary_{timestamp}.md")
        await self._generate_markdown_summary(report, summary_path)
        
        logger.info(f"Validation results saved to: {report_path} and {summary_path}")
        
    async def _generate_markdown_summary(self, report: Phase7ValidationReport, path: Path):
        """Generate markdown summary report"""
        with open(path, 'w') as f:
            f.write(f"# Phase 7 Autonomous Intelligence Ecosystem Validation Report\n\n")
            f.write(f"**Validation Date:** {report.validation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Overall Success:** {'✅ PASSED' if report.overall_success else '❌ FAILED'}\n\n")
            
            f.write("## Performance Target Achievements\n\n")
            for target, achieved in report.target_achievements.items():
                status = "✅ ACHIEVED" if achieved else "❌ NOT ACHIEVED" 
                f.write(f"- **{target.replace('_', ' ').title()}:** {status}\n")
            f.write("\n")
            
            f.write("## Test Suite Results\n\n")
            f.write("| Suite | Tests | Passed | Failed | Success Rate | Time |\n")
            f.write("|-------|-------|--------|--------|--------------|------|\n")
            
            for suite_name, result in report.suite_results.items():
                f.write(f"| {suite_name} | {result.total_tests} | {result.passed_tests} | "
                       f"{result.failed_tests} | {result.success_rate:.1%} | {result.execution_time:.2f}s |\n")
            f.write("\n")
            
            f.write("## Performance Summary\n\n")
            f.write(f"- **Total Tests:** {report.performance_summary['total_tests']}\n")
            f.write(f"- **Tests Passed:** {report.performance_summary['passed_tests']}\n")
            f.write(f"- **Tests Failed:** {report.performance_summary['failed_tests']}\n")
            f.write(f"- **Overall Success Rate:** {report.performance_summary['success_rate']:.1%}\n")
            f.write(f"- **Total Execution Time:** {report.performance_summary['execution_time']:.2f}s\n\n")
            
            if report.critical_issues:
                f.write("## Critical Issues\n\n")
                for issue in report.critical_issues:
                    f.write(f"- {issue}\n")
                f.write("\n")
                
            if report.recommendations:
                f.write("## Recommendations\n\n")
                for rec in report.recommendations:
                    f.write(f"- {rec}\n")
                f.write("\n")
                
            f.write("## System Metrics\n\n")
            f.write("### Pre-Validation\n")
            pre_metrics = report.system_metrics.get('pre_validation', {})
            for key, value in pre_metrics.items():
                f.write(f"- **{key}:** {value}\n")
                
            f.write("\n### Post-Validation\n")
            post_metrics = report.system_metrics.get('post_validation', {})
            for key, value in post_metrics.items():
                f.write(f"- **{key}:** {value}\n")
                
    def _display_validation_summary(self, report: Phase7ValidationReport):
        """Display validation summary to console"""
        print("\n" + "="*80)
        print("🎯 PHASE 7 AUTONOMOUS INTELLIGENCE ECOSYSTEM VALIDATION COMPLETE")
        print("="*80)
        
        print(f"\n📊 OVERALL RESULT: {'✅ SUCCESS' if report.overall_success else '❌ FAILED'}")
        print(f"   Validation completed in {report.performance_summary['execution_time']:.2f} seconds")
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Total Tests: {report.performance_summary['total_tests']}")
        print(f"   Passed: {report.performance_summary['passed_tests']}")
        print(f"   Failed: {report.performance_summary['failed_tests']}")
        print(f"   Success Rate: {report.performance_summary['success_rate']:.1%}")
        
        print(f"\n🎯 TARGET ACHIEVEMENTS:")
        for target, achieved in report.target_achievements.items():
            status = "✅" if achieved else "❌"
            print(f"   {status} {target.replace('_', ' ').title()}")
            
        print(f"\n🧪 TEST SUITE BREAKDOWN:")
        for suite_name, result in report.suite_results.items():
            status = "✅" if result.success_rate >= 0.80 else "❌"
            print(f"   {status} {suite_name}: {result.passed_tests}/{result.total_tests} ({result.success_rate:.1%})")
            
        if report.critical_issues:
            print(f"\n⚠️  CRITICAL ISSUES ({len(report.critical_issues)}):")
            for issue in report.critical_issues[:3]:  # Show first 3
                print(f"   • {issue[:100]}...")
                
        if report.recommendations:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for rec in report.recommendations[:3]:  # Show first 3
                print(f"   • {rec}")
                
        print("\n" + "="*80)
        
        if report.overall_success:
            print("🎉 Phase 7 Autonomous Intelligence Ecosystem is READY FOR PRODUCTION!")
        else:
            print("⚠️  Phase 7 requires additional development before production deployment")
            
        print("="*80 + "\n")


# CLI Interface
async def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Phase 7 Autonomous Intelligence Test Runner')
    parser.add_argument('--suite', help='Run specific test suite only')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (reduced test counts)')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Configuration overrides
    config_override = {}
    if args.quick:
        config_override.update({
            'test_modes': {'unit': True, 'integration': False, 'performance': False}
        })
    
    # Create and run test runner
    runner = Phase7TestRunner(config_override)
    
    try:
        if args.suite:
            # Run specific suite
            await runner._run_test_suite(args.suite)
            result = runner.results.get(args.suite)
            if result:
                print(f"\n{args.suite} Results: {result.passed_tests}/{result.total_tests} passed ({result.success_rate:.1%})")
        else:
            # Run full validation
            report = await runner.run_comprehensive_validation()
            
            # Exit with appropriate code
            sys.exit(0 if report.overall_success else 1)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())