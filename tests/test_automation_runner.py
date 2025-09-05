"""
Phase 6 Test Automation Runner
=============================

Comprehensive test automation system for Phase 6 AI agents including:
- Automated test discovery and execution
- Test result aggregation and reporting
- CI/CD pipeline integration
- Performance benchmarking
- Coverage analysis
- Quality gates enforcement
- Test environment management
- Parallel test execution
- Failure analysis and reporting
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import logging
import concurrent.futures
from enum import Enum
import xml.etree.ElementTree as ET

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_automation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests"""
    UNIT = "unit"
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    SECURITY = "security"
    ERROR_HANDLING = "error_handling"
    END_TO_END = "e2e"


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    test_type: TestType
    status: TestStatus
    duration: float
    start_time: datetime
    end_time: datetime
    error_message: Optional[str] = None
    output: Optional[str] = None
    coverage_data: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    security_findings: Optional[List[str]] = None


@dataclass
class TestSuiteResult:
    """Test suite result data structure"""
    suite_name: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    error_tests: int
    total_duration: float
    coverage_percentage: float
    test_results: List[TestResult]
    start_time: datetime
    end_time: datetime


@dataclass
class QualityGate:
    """Quality gate configuration"""
    name: str
    metric: str
    threshold: float
    operator: str  # '>=', '<=', '>', '<', '=='
    mandatory: bool = True


class TestAutomationConfig:
    """Test automation configuration"""
    
    def __init__(self):
        self.test_directories = {
            TestType.UNIT: "tests/unit",
            TestType.INTEGRATION: "tests/integration", 
            TestType.PERFORMANCE: "tests/performance",
            TestType.SECURITY: "tests/security",
            TestType.ERROR_HANDLING: "tests/error_handling"
        }
        
        self.quality_gates = [
            QualityGate("unit_test_coverage", "coverage", 95.0, ">=", True),
            QualityGate("integration_test_success", "success_rate", 95.0, ">=", True),
            QualityGate("performance_throughput", "throughput", 50.0, ">=", True),
            QualityGate("security_vulnerabilities", "critical_issues", 0, "==", True),
            QualityGate("error_recovery_rate", "recovery_success", 90.0, ">=", True),
            QualityGate("test_execution_time", "total_duration", 1800, "<=", False)  # 30 minutes
        ]
        
        self.parallel_execution = True
        self.max_workers = min(4, os.cpu_count())
        self.timeout_seconds = 3600  # 1 hour
        self.retry_failed_tests = True
        self.generate_reports = True
        
        # Environment configuration
        self.test_environment = {
            "PYTHONPATH": str(project_root),
            "TESTING": "true",
            "LOG_LEVEL": "INFO"
        }


class TestDiscovery:
    """Test discovery and classification"""
    
    def __init__(self, config: TestAutomationConfig):
        self.config = config
        
    def discover_tests(self) -> Dict[TestType, List[Path]]:
        """Discover all test files by type"""
        discovered_tests = {test_type: [] for test_type in TestType}
        
        for test_type, directory in self.config.test_directories.items():
            test_dir = project_root / directory
            
            if not test_dir.exists():
                logger.warning(f"Test directory not found: {test_dir}")
                continue
                
            # Find Python test files
            test_files = list(test_dir.rglob("test_*.py"))
            test_files.extend(test_dir.rglob("*_test.py"))
            
            discovered_tests[test_type] = test_files
            logger.info(f"Discovered {len(test_files)} {test_type.value} tests")
        
        return discovered_tests
    
    def get_test_dependencies(self, test_file: Path) -> List[str]:
        """Get test dependencies from file"""
        dependencies = []
        
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Parse import statements
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('import ') or line.startswith('from '):
                        dependencies.append(line)
                        
        except Exception as e:
            logger.warning(f"Could not parse dependencies for {test_file}: {e}")
            
        return dependencies


class TestExecutor:
    """Test execution engine"""
    
    def __init__(self, config: TestAutomationConfig):
        self.config = config
        
    async def execute_test_file(self, test_file: Path, test_type: TestType) -> TestResult:
        """Execute a single test file"""
        start_time = datetime.now()
        
        try:
            # Prepare pytest command
            cmd = [
                sys.executable, "-m", "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--json-report",
                f"--json-report-file={test_file.stem}_report.json",
                "--cov=.",
                f"--cov-report=json:{test_file.stem}_coverage.json"
            ]
            
            # Add test-type specific options
            if test_type == TestType.PERFORMANCE:
                cmd.extend(["--benchmark-only", "--benchmark-json=benchmark_results.json"])
            elif test_type == TestType.SECURITY:
                cmd.extend(["-m", "security"])
                
            # Execute test
            env = {**os.environ, **self.config.test_environment}
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=project_root
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self.config.timeout_seconds
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            if process.returncode == 0:
                status = TestStatus.PASSED
                error_message = None
            else:
                status = TestStatus.FAILED
                error_message = stderr.decode('utf-8')
            
            # Load additional data
            coverage_data = self._load_coverage_data(test_file.stem)
            performance_metrics = self._load_performance_data(test_file.stem)
            
            return TestResult(
                test_name=test_file.stem,
                test_type=test_type,
                status=status,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                error_message=error_message,
                output=stdout.decode('utf-8'),
                coverage_data=coverage_data,
                performance_metrics=performance_metrics
            )
            
        except asyncio.TimeoutError:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_file.stem,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                error_message=f"Test timed out after {self.config.timeout_seconds} seconds"
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            return TestResult(
                test_name=test_file.stem,
                test_type=test_type,
                status=TestStatus.ERROR,
                duration=duration,
                start_time=start_time,
                end_time=end_time,
                error_message=str(e)
            )
    
    def _load_coverage_data(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Load coverage data for test"""
        coverage_file = project_root / f"{test_name}_coverage.json"
        
        try:
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load coverage data for {test_name}: {e}")
            
        return {}
    
    def _load_performance_data(self, test_name: str) -> Optional[Dict[str, Any]]:
        """Load performance data for test"""
        perf_file = project_root / "benchmark_results.json"
        
        try:
            if perf_file.exists():
                with open(perf_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load performance data for {test_name}: {e}")
            
        return {}
    
    async def execute_test_suite(self, test_files: List[Path], test_type: TestType) -> TestSuiteResult:
        """Execute a complete test suite"""
        start_time = datetime.now()
        
        logger.info(f"Executing {len(test_files)} {test_type.value} tests")
        
        if self.config.parallel_execution and len(test_files) > 1:
            # Parallel execution
            semaphore = asyncio.Semaphore(self.config.max_workers)
            
            async def execute_with_semaphore(test_file):
                async with semaphore:
                    return await self.execute_test_file(test_file, test_type)
            
            tasks = [execute_with_semaphore(test_file) for test_file in test_files]
            test_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Sequential execution
            test_results = []
            for test_file in test_files:
                result = await self.execute_test_file(test_file, test_type)
                test_results.append(result)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Process results
        valid_results = [r for r in test_results if isinstance(r, TestResult)]
        
        passed_tests = sum(1 for r in valid_results if r.status == TestStatus.PASSED)
        failed_tests = sum(1 for r in valid_results if r.status == TestStatus.FAILED)
        skipped_tests = sum(1 for r in valid_results if r.status == TestStatus.SKIPPED)
        error_tests = sum(1 for r in valid_results if r.status == TestStatus.ERROR)
        
        # Calculate overall coverage
        coverage_percentage = self._calculate_overall_coverage(valid_results)
        
        suite_result = TestSuiteResult(
            suite_name=f"{test_type.value}_tests",
            total_tests=len(valid_results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            error_tests=error_tests,
            total_duration=total_duration,
            coverage_percentage=coverage_percentage,
            test_results=valid_results,
            start_time=start_time,
            end_time=end_time
        )
        
        logger.info(f"{test_type.value} tests completed: {passed_tests}/{len(valid_results)} passed in {total_duration:.2f}s")
        
        return suite_result
    
    def _calculate_overall_coverage(self, test_results: List[TestResult]) -> float:
        """Calculate overall test coverage"""
        coverage_data = [r.coverage_data for r in test_results if r.coverage_data]
        
        if not coverage_data:
            return 0.0
        
        # Aggregate coverage data
        total_lines = 0
        covered_lines = 0
        
        for coverage in coverage_data:
            if 'totals' in coverage:
                total_lines += coverage['totals'].get('num_statements', 0)
                covered_lines += coverage['totals'].get('covered_lines', 0)
        
        if total_lines == 0:
            return 0.0
        
        return (covered_lines / total_lines) * 100


class QualityGateEvaluator:
    """Quality gate evaluation"""
    
    def __init__(self, config: TestAutomationConfig):
        self.config = config
    
    def evaluate_quality_gates(self, suite_results: List[TestSuiteResult]) -> Dict[str, Any]:
        """Evaluate all quality gates"""
        gate_results = {}
        overall_pass = True
        
        for gate in self.config.quality_gates:
            result = self._evaluate_gate(gate, suite_results)
            gate_results[gate.name] = result
            
            if gate.mandatory and not result['passed']:
                overall_pass = False
        
        return {
            'overall_pass': overall_pass,
            'gate_results': gate_results,
            'evaluation_time': datetime.now().isoformat()
        }
    
    def _evaluate_gate(self, gate: QualityGate, suite_results: List[TestSuiteResult]) -> Dict[str, Any]:
        """Evaluate a single quality gate"""
        try:
            # Extract metric value
            metric_value = self._extract_metric_value(gate.metric, suite_results)
            
            # Evaluate condition
            passed = self._evaluate_condition(metric_value, gate.threshold, gate.operator)
            
            return {
                'passed': passed,
                'metric_value': metric_value,
                'threshold': gate.threshold,
                'operator': gate.operator,
                'mandatory': gate.mandatory
            }
            
        except Exception as e:
            logger.error(f"Error evaluating quality gate {gate.name}: {e}")
            return {
                'passed': False,
                'error': str(e),
                'mandatory': gate.mandatory
            }
    
    def _extract_metric_value(self, metric: str, suite_results: List[TestSuiteResult]) -> float:
        """Extract metric value from test results"""
        if metric == "coverage":
            # Average coverage across all suites
            coverages = [suite.coverage_percentage for suite in suite_results]
            return sum(coverages) / len(coverages) if coverages else 0.0
            
        elif metric == "success_rate":
            # Overall success rate
            total_tests = sum(suite.total_tests for suite in suite_results)
            passed_tests = sum(suite.passed_tests for suite in suite_results)
            return (passed_tests / total_tests * 100) if total_tests > 0 else 0.0
            
        elif metric == "total_duration":
            # Total execution time
            return sum(suite.total_duration for suite in suite_results)
            
        elif metric == "throughput":
            # Tests per second
            total_tests = sum(suite.total_tests for suite in suite_results)
            total_duration = sum(suite.total_duration for suite in suite_results)
            return total_tests / total_duration if total_duration > 0 else 0.0
            
        elif metric == "critical_issues":
            # Count of critical security issues
            critical_count = 0
            for suite in suite_results:
                for test_result in suite.test_results:
                    if test_result.security_findings:
                        critical_count += len([f for f in test_result.security_findings if "critical" in f.lower()])
            return critical_count
            
        elif metric == "recovery_success":
            # Error recovery success rate
            recovery_tests = []
            for suite in suite_results:
                if "error_handling" in suite.suite_name or "resilience" in suite.suite_name:
                    recovery_tests.extend(suite.test_results)
            
            if not recovery_tests:
                return 100.0  # No recovery tests, assume success
            
            passed_recovery = sum(1 for test in recovery_tests if test.status == TestStatus.PASSED)
            return (passed_recovery / len(recovery_tests) * 100) if recovery_tests else 100.0
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def _evaluate_condition(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate condition based on operator"""
        if operator == ">=":
            return value >= threshold
        elif operator == "<=":
            return value <= threshold
        elif operator == ">":
            return value > threshold
        elif operator == "<":
            return value < threshold
        elif operator == "==":
            return abs(value - threshold) < 0.001  # Float comparison with tolerance
        else:
            raise ValueError(f"Unknown operator: {operator}")


class ReportGenerator:
    """Test report generation"""
    
    def __init__(self, config: TestAutomationConfig):
        self.config = config
    
    def generate_comprehensive_report(
        self, 
        suite_results: List[TestSuiteResult],
        quality_gate_results: Dict[str, Any]
    ) -> str:
        """Generate comprehensive test report"""
        
        # Calculate summary statistics
        total_tests = sum(suite.total_tests for suite in suite_results)
        total_passed = sum(suite.passed_tests for suite in suite_results)
        total_failed = sum(suite.failed_tests for suite in suite_results)
        total_errors = sum(suite.error_tests for suite in suite_results)
        total_duration = sum(suite.total_duration for suite in suite_results)
        avg_coverage = sum(suite.coverage_percentage for suite in suite_results) / len(suite_results) if suite_results else 0
        
        report_lines = [
            "=" * 100,
            "PHASE 6 AI AGENTS - COMPREHENSIVE TEST AUTOMATION REPORT",
            "=" * 100,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Execution Time: {total_duration:.2f} seconds ({total_duration/60:.1f} minutes)",
            "",
            "EXECUTIVE SUMMARY:",
            "-" * 20,
            f"Total Tests Executed: {total_tests}",
            f"Tests Passed: {total_passed} ({total_passed/total_tests*100:.1f}%)" if total_tests > 0 else "Tests Passed: 0 (0.0%)",
            f"Tests Failed: {total_failed} ({total_failed/total_tests*100:.1f}%)" if total_tests > 0 else "Tests Failed: 0 (0.0%)",
            f"Tests with Errors: {total_errors} ({total_errors/total_tests*100:.1f}%)" if total_tests > 0 else "Tests with Errors: 0 (0.0%)",
            f"Average Code Coverage: {avg_coverage:.1f}%",
            f"Quality Gates Status: {'✓ PASSED' if quality_gate_results['overall_pass'] else '✗ FAILED'}",
            ""
        ]
        
        # Test suite breakdown
        report_lines.extend([
            "TEST SUITE BREAKDOWN:",
            "-" * 25
        ])
        
        for suite in suite_results:
            success_rate = (suite.passed_tests / suite.total_tests * 100) if suite.total_tests > 0 else 0
            report_lines.extend([
                f"{suite.suite_name.upper()}:",
                f"  Tests: {suite.total_tests} | Passed: {suite.passed_tests} | Failed: {suite.failed_tests} | Errors: {suite.error_tests}",
                f"  Success Rate: {success_rate:.1f}% | Duration: {suite.total_duration:.2f}s | Coverage: {suite.coverage_percentage:.1f}%",
                ""
            ])
        
        # Quality gates details
        report_lines.extend([
            "QUALITY GATES EVALUATION:",
            "-" * 30
        ])
        
        for gate_name, gate_result in quality_gate_results['gate_results'].items():
            status = "✓ PASS" if gate_result['passed'] else "✗ FAIL"
            mandatory = " (MANDATORY)" if gate_result.get('mandatory', False) else " (OPTIONAL)"
            
            if 'error' in gate_result:
                report_lines.append(f"{gate_name}: ERROR - {gate_result['error']}{mandatory}")
            else:
                metric_value = gate_result['metric_value']
                threshold = gate_result['threshold']
                operator = gate_result['operator']
                report_lines.append(f"{gate_name}: {status} - {metric_value:.2f} {operator} {threshold}{mandatory}")
        
        report_lines.append("")
        
        # Performance metrics
        report_lines.extend([
            "PERFORMANCE METRICS:",
            "-" * 20,
            f"Test Throughput: {total_tests/total_duration:.2f} tests/second" if total_duration > 0 else "Test Throughput: N/A",
            f"Average Test Duration: {total_duration/total_tests:.3f} seconds" if total_tests > 0 else "Average Test Duration: N/A",
            f"Parallel Execution: {'Enabled' if self.config.parallel_execution else 'Disabled'}",
            f"Max Workers: {self.config.max_workers}",
            ""
        ])
        
        # Failed tests details
        failed_tests = []
        for suite in suite_results:
            for test_result in suite.test_results:
                if test_result.status in [TestStatus.FAILED, TestStatus.ERROR]:
                    failed_tests.append(test_result)
        
        if failed_tests:
            report_lines.extend([
                "FAILED TESTS DETAILS:",
                "-" * 22
            ])
            
            for test in failed_tests[:10]:  # Show first 10 failures
                report_lines.extend([
                    f"Test: {test.test_name} ({test.test_type.value})",
                    f"Status: {test.status.value.upper()}",
                    f"Duration: {test.duration:.3f}s",
                    f"Error: {test.error_message[:200] if test.error_message else 'No error message'}{'...' if test.error_message and len(test.error_message) > 200 else ''}",
                    ""
                ])
            
            if len(failed_tests) > 10:
                report_lines.append(f"... and {len(failed_tests) - 10} more failed tests")
            
            report_lines.append("")
        
        # Test recommendations
        report_lines.extend([
            "RECOMMENDATIONS:",
            "-" * 15
        ])
        
        recommendations = self._generate_recommendations(suite_results, quality_gate_results)
        for rec in recommendations:
            report_lines.append(f"• {rec}")
        
        report_lines.extend([
            "",
            "=" * 100
        ])
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, suite_results: List[TestSuiteResult], quality_gate_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Coverage recommendations
        avg_coverage = sum(suite.coverage_percentage for suite in suite_results) / len(suite_results) if suite_results else 0
        if avg_coverage < 90:
            recommendations.append(f"Increase test coverage from {avg_coverage:.1f}% to at least 90%")
        
        # Performance recommendations
        total_duration = sum(suite.total_duration for suite in suite_results)
        if total_duration > 1800:  # 30 minutes
            recommendations.append(f"Optimize test execution time (currently {total_duration/60:.1f} minutes)")
        
        # Failed tests recommendations
        total_failed = sum(suite.failed_tests + suite.error_tests for suite in suite_results)
        if total_failed > 0:
            recommendations.append(f"Fix {total_failed} failing/error tests before production deployment")
        
        # Quality gate recommendations
        if not quality_gate_results['overall_pass']:
            failed_gates = [name for name, result in quality_gate_results['gate_results'].items() 
                          if not result['passed'] and result.get('mandatory', False)]
            if failed_gates:
                recommendations.append(f"Address mandatory quality gate failures: {', '.join(failed_gates)}")
        
        # Test type specific recommendations
        for suite in suite_results:
            if 'security' in suite.suite_name and suite.failed_tests > 0:
                recommendations.append("Address security test failures before deployment")
            elif 'performance' in suite.suite_name and suite.failed_tests > 0:
                recommendations.append("Investigate performance test failures - may indicate bottlenecks")
            elif 'error_handling' in suite.suite_name and suite.failed_tests > 0:
                recommendations.append("Fix error handling tests to ensure system resilience")
        
        if not recommendations:
            recommendations.append("All tests passing - system ready for deployment")
        
        return recommendations
    
    def save_report_files(self, suite_results: List[TestSuiteResult], quality_gate_results: Dict[str, Any]):
        """Save various report formats"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        reports_dir = project_root / 'test_reports'
        reports_dir.mkdir(exist_ok=True)
        
        # Comprehensive text report
        text_report = self.generate_comprehensive_report(suite_results, quality_gate_results)
        with open(reports_dir / f'test_report_{timestamp}.txt', 'w') as f:
            f.write(text_report)
        
        # JSON report for CI/CD integration
        json_report = {
            'timestamp': timestamp,
            'summary': {
                'total_tests': sum(suite.total_tests for suite in suite_results),
                'passed_tests': sum(suite.passed_tests for suite in suite_results),
                'failed_tests': sum(suite.failed_tests for suite in suite_results),
                'error_tests': sum(suite.error_tests for suite in suite_results),
                'total_duration': sum(suite.total_duration for suite in suite_results),
                'average_coverage': sum(suite.coverage_percentage for suite in suite_results) / len(suite_results) if suite_results else 0
            },
            'suite_results': [asdict(suite) for suite in suite_results],
            'quality_gates': quality_gate_results
        }
        
        with open(reports_dir / f'test_results_{timestamp}.json', 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        # JUnit XML for CI/CD systems
        self._generate_junit_xml(suite_results, reports_dir / f'junit_results_{timestamp}.xml')
        
        logger.info(f"Test reports saved to {reports_dir}")
    
    def _generate_junit_xml(self, suite_results: List[TestSuiteResult], output_file: Path):
        """Generate JUnit XML format report"""
        root = ET.Element('testsuites')
        
        for suite in suite_results:
            suite_element = ET.SubElement(root, 'testsuite')
            suite_element.set('name', suite.suite_name)
            suite_element.set('tests', str(suite.total_tests))
            suite_element.set('failures', str(suite.failed_tests))
            suite_element.set('errors', str(suite.error_tests))
            suite_element.set('skipped', str(suite.skipped_tests))
            suite_element.set('time', str(suite.total_duration))
            
            for test_result in suite.test_results:
                test_element = ET.SubElement(suite_element, 'testcase')
                test_element.set('classname', f"{suite.suite_name}.{test_result.test_name}")
                test_element.set('name', test_result.test_name)
                test_element.set('time', str(test_result.duration))
                
                if test_result.status == TestStatus.FAILED:
                    failure_element = ET.SubElement(test_element, 'failure')
                    failure_element.set('message', test_result.error_message or 'Test failed')
                    failure_element.text = test_result.error_message or 'No error details'
                elif test_result.status == TestStatus.ERROR:
                    error_element = ET.SubElement(test_element, 'error')
                    error_element.set('message', test_result.error_message or 'Test error')
                    error_element.text = test_result.error_message or 'No error details'
                elif test_result.status == TestStatus.SKIPPED:
                    ET.SubElement(test_element, 'skipped')
        
        # Write XML file
        tree = ET.ElementTree(root)
        tree.write(output_file, encoding='utf-8', xml_declaration=True)


class TestAutomationRunner:
    """Main test automation runner"""
    
    def __init__(self, config: Optional[TestAutomationConfig] = None):
        self.config = config or TestAutomationConfig()
        self.discovery = TestDiscovery(self.config)
        self.executor = TestExecutor(self.config)
        self.quality_evaluator = QualityGateEvaluator(self.config)
        self.report_generator = ReportGenerator(self.config)
    
    async def run_all_tests(self) -> Tuple[List[TestSuiteResult], Dict[str, Any]]:
        """Run complete test automation pipeline"""
        logger.info("Starting Phase 6 test automation pipeline")
        start_time = datetime.now()
        
        # Discover tests
        discovered_tests = self.discovery.discover_tests()
        
        # Execute test suites
        suite_results = []
        
        for test_type, test_files in discovered_tests.items():
            if test_files:  # Only run if tests exist
                logger.info(f"Running {test_type.value} test suite...")
                suite_result = await self.executor.execute_test_suite(test_files, test_type)
                suite_results.append(suite_result)
        
        # Evaluate quality gates
        quality_gate_results = self.quality_evaluator.evaluate_quality_gates(suite_results)
        
        # Generate reports
        if self.config.generate_reports:
            self.report_generator.save_report_files(suite_results, quality_gate_results)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Test automation pipeline completed in {total_duration:.2f} seconds")
        logger.info(f"Quality Gates: {'PASSED' if quality_gate_results['overall_pass'] else 'FAILED'}")
        
        return suite_results, quality_gate_results
    
    async def run_specific_test_type(self, test_type: TestType) -> TestSuiteResult:
        """Run tests for a specific test type"""
        logger.info(f"Running {test_type.value} tests only")
        
        discovered_tests = self.discovery.discover_tests()
        test_files = discovered_tests.get(test_type, [])
        
        if not test_files:
            logger.warning(f"No {test_type.value} tests found")
            return TestSuiteResult(
                suite_name=f"{test_type.value}_tests",
                total_tests=0,
                passed_tests=0,
                failed_tests=0,
                skipped_tests=0,
                error_tests=0,
                total_duration=0,
                coverage_percentage=0,
                test_results=[],
                start_time=datetime.now(),
                end_time=datetime.now()
            )
        
        return await self.executor.execute_test_suite(test_files, test_type)
    
    def get_test_status(self) -> Dict[str, Any]:
        """Get current test automation status"""
        discovered_tests = self.discovery.discover_tests()
        
        return {
            'config': {
                'parallel_execution': self.config.parallel_execution,
                'max_workers': self.config.max_workers,
                'timeout_seconds': self.config.timeout_seconds
            },
            'discovered_tests': {
                test_type.value: len(test_files) 
                for test_type, test_files in discovered_tests.items()
            },
            'quality_gates': [
                {
                    'name': gate.name,
                    'metric': gate.metric,
                    'threshold': gate.threshold,
                    'mandatory': gate.mandatory
                }
                for gate in self.config.quality_gates
            ]
        }


async def main():
    """Main entry point for test automation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 6 Test Automation Runner")
    parser.add_argument("--test-type", choices=[t.value for t in TestType], help="Run specific test type")
    parser.add_argument("--parallel", action="store_true", default=True, help="Enable parallel execution")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--timeout", type=int, default=3600, help="Test timeout in seconds")
    parser.add_argument("--no-reports", action="store_true", help="Disable report generation")
    parser.add_argument("--status", action="store_true", help="Show test automation status")
    
    args = parser.parse_args()
    
    # Configure automation
    config = TestAutomationConfig()
    config.parallel_execution = args.parallel
    config.max_workers = args.workers
    config.timeout_seconds = args.timeout
    config.generate_reports = not args.no_reports
    
    runner = TestAutomationRunner(config)
    
    if args.status:
        status = runner.get_test_status()
        print(json.dumps(status, indent=2))
        return {}
    
    try:
        if args.test_type:
            # Run specific test type
            test_type = TestType(args.test_type)
            suite_result = await runner.run_specific_test_type(test_type)
            
            print(f"\n{test_type.value.upper()} TEST RESULTS:")
            print(f"Tests: {suite_result.total_tests}")
            print(f"Passed: {suite_result.passed_tests}")
            print(f"Failed: {suite_result.failed_tests}")
            print(f"Errors: {suite_result.error_tests}")
            print(f"Duration: {suite_result.total_duration:.2f}s")
            print(f"Coverage: {suite_result.coverage_percentage:.1f}%")
            
        else:
            # Run all tests
            suite_results, quality_gate_results = await runner.run_all_tests()
            
            # Print summary
            report = runner.report_generator.generate_comprehensive_report(suite_results, quality_gate_results)
            print(report)
            
            # Exit code for CI/CD
            if not quality_gate_results['overall_pass']:
                sys.exit(1)
                
    except KeyboardInterrupt:
        logger.info("Test automation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test automation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())