"""
Comprehensive test runner for Phase 7 Advanced Reasoning Systems
Orchestrates all reasoning system tests including unit tests, validation, and benchmarks
"""

import asyncio
import sys
import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_reports/reasoning_test_execution.log')
    ]
)
logger = logging.getLogger(__name__)

# Ensure test reports directory exists
Path("test_reports").mkdir(exist_ok=True)


class ReasoningTestRunner:
    """Orchestrates execution of all reasoning system tests"""
    
    def __init__(self):
        self.test_results = {}
        self.execution_start_time = None
        self.execution_end_time = None
        
        # Test configuration
        self.test_suites = {
            "unit_tests": {
                "file": "test_reasoning_systems.py",
                "description": "Comprehensive unit and integration tests",
                "markers": ["not slow", "not performance"],
                "required": True
            },
            "validation_tests": {
                "file": "test_reasoning_validation.py", 
                "description": "Accuracy validation against ground truth",
                "markers": [],
                "required": True
            },
            "performance_benchmarks": {
                "file": "test_reasoning_benchmarks.py",
                "description": "Performance benchmarking suite",
                "markers": ["performance"],
                "required": False
            }
        }
        
        # Target thresholds
        self.targets = {
            "causal_accuracy": 0.90,
            "working_memory_tokens": 10000,
            "response_time_simple": 1.0,  # seconds
            "overall_success_rate": 0.85,
            "reliability_threshold": 0.95
        }
    
    def parse_arguments(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description="Run comprehensive reasoning systems test suite"
        )
        
        parser.add_argument(
            "--suite", 
            choices=["all", "unit", "validation", "performance"],
            default="all",
            help="Test suite to run"
        )
        
        parser.add_argument(
            "--quick",
            action="store_true",
            help="Run quick test suite (skip slow and performance tests)"
        )
        
        parser.add_argument(
            "--coverage",
            action="store_true",
            help="Generate test coverage reports"
        )
        
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Verbose output"
        )
        
        parser.add_argument(
            "--parallel",
            type=int,
            default=1,
            help="Number of parallel test workers"
        )
        
        parser.add_argument(
            "--report-format",
            choices=["json", "html", "both"],
            default="both",
            help="Test report format"
        )
        
        parser.add_argument(
            "--fail-fast",
            action="store_true", 
            help="Stop on first test failure"
        )
        
        return parser.parse_args()
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are available"""
        logger.info("Checking test prerequisites...")
        
        try:
            # Check if reasoning systems are importable
            import core.reasoning.causal_inference
            import core.reasoning.working_memory
            import core.reasoning.tree_of_thoughts
            import core.reasoning.temporal_reasoning
            import core.reasoning.integrated_reasoning_controller
            import core.reasoning.performance_optimizer
            
            logger.info("✓ All reasoning systems available")
            return True
            
        except ImportError as e:
            logger.error(f"✗ Missing reasoning system dependencies: {e}")
            return False
    
    async def run_test_suite(self, suite_name: str, args: argparse.Namespace) -> Dict[str, Any]:
        """Run a specific test suite"""
        if suite_name not in self.test_suites:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        suite_config = self.test_suites[suite_name]
        logger.info(f"Running {suite_name}: {suite_config['description']}")
        
        # Build pytest command
        pytest_cmd = [
            sys.executable, "-m", "pytest",
            suite_config["file"],
            "--tb=short",
            "--disable-warnings"
        ]
        
        # Add verbosity
        if args.verbose:
            pytest_cmd.extend(["-v", "-s"])
        else:
            pytest_cmd.append("-q")
        
        # Add markers
        if suite_config["markers"]:
            marker_expr = " and ".join(suite_config["markers"])
            pytest_cmd.extend(["-m", marker_expr])
        elif args.quick and suite_name == "unit_tests":
            pytest_cmd.extend(["-m", "not slow and not performance"])
        
        # Add parallel execution
        if args.parallel > 1:
            pytest_cmd.extend(["-n", str(args.parallel)])
        
        # Add coverage
        if args.coverage:
            pytest_cmd.extend([
                "--cov=core.reasoning",
                "--cov-report=term-missing",
                f"--cov-report=html:test_reports/{suite_name}_coverage"
            ])
        
        # Add fail-fast
        if args.fail_fast:
            pytest_cmd.append("-x")
        
        # Add JSON report
        json_report_path = f"test_reports/{suite_name}_results.json"
        pytest_cmd.extend(["--json-report", f"--json-report-file={json_report_path}"])
        
        # Execute test suite
        start_time = time.time()
        
        try:
            result = subprocess.run(
                pytest_cmd,
                cwd=Path(__file__).parent,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results
            test_result = {
                "suite_name": suite_name,
                "exit_code": result.returncode,
                "execution_time": execution_time,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": " ".join(pytest_cmd)
            }
            
            # Try to load JSON report for detailed metrics
            try:
                if os.path.exists(json_report_path):
                    with open(json_report_path, 'r') as f:
                        json_report = json.load(f)
                        test_result.update({
                            "total_tests": json_report["summary"]["total"],
                            "passed_tests": json_report["summary"].get("passed", 0),
                            "failed_tests": json_report["summary"].get("failed", 0),
                            "skipped_tests": json_report["summary"].get("skipped", 0),
                            "error_tests": json_report["summary"].get("error", 0),
                            "success_rate": json_report["summary"].get("passed", 0) / max(1, json_report["summary"]["total"]),
                            "detailed_results": json_report.get("tests", [])
                        })
            except Exception as e:
                logger.warning(f"Could not parse JSON report for {suite_name}: {e}")
            
            # Log results
            if result.returncode == 0:
                logger.info(f"✓ {suite_name} completed successfully in {execution_time:.1f}s")
            else:
                logger.error(f"✗ {suite_name} failed with exit code {result.returncode}")
                if result.stderr:
                    logger.error(f"Error output: {result.stderr[:500]}")
            
            return test_result
            
        except subprocess.TimeoutExpired:
            logger.error(f"✗ {suite_name} timed out after 30 minutes")
            return {
                "suite_name": suite_name,
                "exit_code": -1,
                "execution_time": 1800,
                "error": "Test suite timed out",
                "success_rate": 0.0
            }
        
        except Exception as e:
            logger.error(f"✗ {suite_name} failed with exception: {e}")
            return {
                "suite_name": suite_name,
                "exit_code": -1,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "success_rate": 0.0
            }
    
    async def run_all_tests(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Run all selected test suites"""
        self.execution_start_time = datetime.now()
        logger.info(f"Starting comprehensive reasoning systems test execution at {self.execution_start_time}")
        
        # Determine which suites to run
        if args.suite == "all":
            suites_to_run = list(self.test_suites.keys())
        elif args.suite == "unit":
            suites_to_run = ["unit_tests"]
        elif args.suite == "validation":
            suites_to_run = ["validation_tests"]
        elif args.suite == "performance":
            suites_to_run = ["performance_benchmarks"]
        
        # Filter out non-required suites if quick mode
        if args.quick:
            suites_to_run = [s for s in suites_to_run if self.test_suites[s]["required"]]
        
        logger.info(f"Selected test suites: {', '.join(suites_to_run)}")
        
        # Run test suites sequentially (to avoid resource conflicts)
        results = {}
        overall_success = True
        
        for suite_name in suites_to_run:
            try:
                result = await self.run_test_suite(suite_name, args)
                results[suite_name] = result
                
                # Check if this suite failed
                if result["exit_code"] != 0:
                    overall_success = False
                    if args.fail_fast:
                        logger.error("Stopping due to test failure (--fail-fast enabled)")
                        break
                        
            except Exception as e:
                logger.error(f"Failed to run {suite_name}: {e}")
                results[suite_name] = {
                    "suite_name": suite_name,
                    "exit_code": -1,
                    "error": str(e),
                    "success_rate": 0.0
                }
                overall_success = False
                
                if args.fail_fast:
                    break
        
        self.execution_end_time = datetime.now()
        total_execution_time = (self.execution_end_time - self.execution_start_time).total_seconds()
        
        # Compile overall results
        overall_results = {
            "execution_summary": {
                "start_time": self.execution_start_time.isoformat(),
                "end_time": self.execution_end_time.isoformat(),
                "total_execution_time": total_execution_time,
                "overall_success": overall_success,
                "suites_run": len(results),
                "suites_passed": sum(1 for r in results.values() if r["exit_code"] == 0),
                "suites_failed": sum(1 for r in results.values() if r["exit_code"] != 0)
            },
            "suite_results": results,
            "target_assessment": self.assess_targets(results),
            "recommendations": self.generate_recommendations(results)
        }
        
        self.test_results = overall_results
        return overall_results
    
    def assess_targets(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess performance against targets"""
        assessment = {}
        
        # Overall success rate assessment
        total_tests = sum(r.get("total_tests", 0) for r in results.values())
        total_passed = sum(r.get("passed_tests", 0) for r in results.values())
        
        if total_tests > 0:
            overall_success_rate = total_passed / total_tests
            assessment["overall_success_rate"] = {
                "target": self.targets["overall_success_rate"],
                "achieved": overall_success_rate,
                "passed": overall_success_rate >= self.targets["overall_success_rate"]
            }
        
        # Suite-specific assessments
        for suite_name, result in results.items():
            if "success_rate" in result:
                assessment[f"{suite_name}_success"] = {
                    "target": 0.95 if suite_name == "performance_benchmarks" else 0.90,
                    "achieved": result["success_rate"],
                    "passed": result["success_rate"] >= (0.95 if suite_name == "performance_benchmarks" else 0.90)
                }
        
        # Calculate overall target achievement
        passed_targets = sum(1 for target in assessment.values() if target.get("passed", False))
        total_targets = len(assessment)
        
        assessment["summary"] = {
            "targets_passed": passed_targets,
            "total_targets": total_targets,
            "target_pass_rate": passed_targets / max(1, total_targets)
        }
        
        return assessment
    
    def generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        # Check each test suite
        for suite_name, result in results.items():
            if result["exit_code"] != 0:
                recommendations.append(f"Address failures in {suite_name} test suite")
            
            if result.get("success_rate", 0) < 0.9:
                recommendations.append(f"Improve {suite_name} success rate (current: {result.get('success_rate', 0):.1%})")
            
            # Performance-specific recommendations
            if suite_name == "performance_benchmarks":
                if result.get("execution_time", 0) > 600:  # 10 minutes
                    recommendations.append("Optimize performance test execution time")
        
        # Overall recommendations
        total_execution_time = sum(r.get("execution_time", 0) for r in results.values())
        if total_execution_time > 3600:  # 1 hour
            recommendations.append("Consider parallelizing test execution to reduce total time")
        
        failed_suites = [name for name, result in results.items() if result["exit_code"] != 0]
        if len(failed_suites) > 1:
            recommendations.append("Multiple test suite failures indicate systematic issues")
        
        if not recommendations:
            recommendations.append("All test suites passed successfully - system ready for production")
        
        return recommendations
    
    def generate_test_report(self, args: argparse.Namespace) -> None:
        """Generate comprehensive test report"""
        if not self.test_results:
            logger.error("No test results available for report generation")
            return {}
        
        # Generate JSON report
        json_report_path = "test_reports/comprehensive_reasoning_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        logger.info(f"JSON test report saved to: {json_report_path}")
        
        # Generate HTML report if requested
        if args.report_format in ["html", "both"]:
            self.generate_html_report()
        
        # Print summary to console
        self.print_test_summary()
    
    def generate_html_report(self) -> None:
        """Generate HTML test report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reasoning Systems Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .success {{ color: green; font-weight: bold; }}
                .failure {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .metric {{ margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .passed {{ background-color: #d4edda; }}
                .failed {{ background-color: #f8d7da; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Phase 7 Reasoning Systems Test Report</h1>
                <p>Generated: {datetime.now().isoformat()}</p>
                <p>Total Execution Time: {self.test_results['execution_summary']['total_execution_time']:.1f} seconds</p>
            </div>
            
            <h2>Execution Summary</h2>
            <div class="metric">
                <strong>Overall Success:</strong> 
                <span class="{'success' if self.test_results['execution_summary']['overall_success'] else 'failure'}">
                    {'✓ PASSED' if self.test_results['execution_summary']['overall_success'] else '✗ FAILED'}
                </span>
            </div>
            
            <div class="metric">
                <strong>Suites Executed:</strong> {self.test_results['execution_summary']['suites_run']}<br>
                <strong>Suites Passed:</strong> {self.test_results['execution_summary']['suites_passed']}<br>
                <strong>Suites Failed:</strong> {self.test_results['execution_summary']['suites_failed']}
            </div>
            
            <h2>Test Suite Results</h2>
            <table>
                <tr>
                    <th>Test Suite</th>
                    <th>Status</th>
                    <th>Success Rate</th>
                    <th>Execution Time</th>
                    <th>Total Tests</th>
                    <th>Passed</th>
                    <th>Failed</th>
                </tr>
        """
        
        for suite_name, result in self.test_results['suite_results'].items():
            status_class = "passed" if result["exit_code"] == 0 else "failed"
            status_text = "✓ PASSED" if result["exit_code"] == 0 else "✗ FAILED"
            
            html_content += f"""
                <tr class="{status_class}">
                    <td>{suite_name}</td>
                    <td>{status_text}</td>
                    <td>{result.get('success_rate', 0):.1%}</td>
                    <td>{result.get('execution_time', 0):.1f}s</td>
                    <td>{result.get('total_tests', 'N/A')}</td>
                    <td>{result.get('passed_tests', 'N/A')}</td>
                    <td>{result.get('failed_tests', 'N/A')}</td>
                </tr>
            """
        
        html_content += """
            </table>
            
            <h2>Target Assessment</h2>
        """
        
        if "target_assessment" in self.test_results:
            for target_name, assessment in self.test_results['target_assessment'].items():
                if target_name != "summary":
                    passed_class = "success" if assessment.get("passed", False) else "failure"
                    html_content += f"""
                        <div class="metric">
                            <strong>{target_name}:</strong>
                            <span class="{passed_class}">
                                {assessment.get('achieved', 0):.1%} 
                                (target: {assessment.get('target', 0):.1%})
                                {'✓' if assessment.get('passed', False) else '✗'}
                            </span>
                        </div>
                    """
        
        html_content += """
            <h2>Recommendations</h2>
            <ul>
        """
        
        for recommendation in self.test_results.get('recommendations', []):
            html_content += f"<li>{recommendation}</li>"
        
        html_content += """
            </ul>
        </body>
        </html>
        """
        
        html_report_path = "test_reports/comprehensive_reasoning_test_report.html"
        with open(html_report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML test report saved to: {html_report_path}")
    
    def print_test_summary(self) -> None:
        """Print test summary to console"""
        print("\n" + "="*80)
        print("PHASE 7 REASONING SYSTEMS TEST SUMMARY")
        print("="*80)
        
        summary = self.test_results['execution_summary']
        print(f"Execution Time: {summary['total_execution_time']:.1f} seconds")
        print(f"Overall Result: {'✓ PASSED' if summary['overall_success'] else '✗ FAILED'}")
        print(f"Suites Run: {summary['suites_run']}")
        print(f"Suites Passed: {summary['suites_passed']}")
        print(f"Suites Failed: {summary['suites_failed']}")
        
        print("\nDETAILED RESULTS:")
        print("-" * 80)
        
        for suite_name, result in self.test_results['suite_results'].items():
            status = "✓ PASSED" if result["exit_code"] == 0 else "✗ FAILED"
            print(f"{suite_name:25} | {status:8} | {result.get('success_rate', 0):6.1%} | {result.get('execution_time', 0):6.1f}s")
        
        if "target_assessment" in self.test_results and "summary" in self.test_results["target_assessment"]:
            target_summary = self.test_results["target_assessment"]["summary"]
            print(f"\nTARGET ACHIEVEMENT: {target_summary['targets_passed']}/{target_summary['total_targets']} ({target_summary['target_pass_rate']:.1%})")
        
        if self.test_results.get('recommendations'):
            print("\nRECOMMENDATIONS:")
            for i, rec in enumerate(self.test_results['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("="*80)


async def main():
    """Main execution function"""
    runner = ReasoningTestRunner()
    args = runner.parse_arguments()
    
    # Check prerequisites
    if not runner.check_prerequisites():
        logger.error("Prerequisites not met. Cannot run tests.")
        return 1
    
    try:
        # Run tests
        results = await runner.run_all_tests(args)
        
        # Generate reports
        runner.generate_test_report(args)
        
        # Return appropriate exit code
        if results['execution_summary']['overall_success']:
            logger.info("All tests completed successfully!")
            return 0
        else:
            logger.error("Some tests failed. Check the detailed report.")
            return 1
            
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        return 1
        
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))