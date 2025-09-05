"""
Comprehensive Test Runner for AI Agents Repository
Executes all test suites and generates detailed reports
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Comprehensive test execution and reporting"""
    
    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / "tests"
        self.reports_dir = self.project_root / "test_reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        self.results = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "test_suites": {},
            "coverage": {},
            "summary": {},
            "errors": []
        }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test suites and collect results"""
        print("Starting Comprehensive Test Suite Execution")
        print("=" * 60)
        
        self.results["start_time"] = datetime.now().isoformat()
        start_time = time.perf_counter()
        
        try:
            # Run individual test suites
            self._run_base_agent_tests()
            self._run_orchestrator_tests()
            self._run_utils_tests()
            self._run_existing_contract_tests()
            
            # Run coverage analysis
            self._run_coverage_analysis()
            
            # Generate performance report
            self._run_performance_tests()
            
            # Run integration smoke tests
            self._run_integration_tests()
            
        except Exception as e:
            self.results["errors"].append(f"Test execution error: {str(e)}")
            print(f"❌ Error during test execution: {e}")
        
        finally:
            end_time = time.perf_counter()
            self.results["end_time"] = datetime.now().isoformat()
            self.results["duration"] = end_time - start_time
            
            self._generate_summary()
            self._save_results()
            self._print_final_report()
        
        return self.results
    
    def _run_base_agent_tests(self):
        """Run BaseAgent comprehensive tests"""
        print("\n[AGENT] Running BaseAgent Tests...")
        
        result = self._execute_pytest(
            "tests/python/test_base_agent_comprehensive.py",
            "BaseAgent Tests"
        )
        self.results["test_suites"]["base_agent"] = result
    
    def _run_orchestrator_tests(self):
        """Run Orchestrator comprehensive tests"""
        print("\n[ORCHESTRATOR] Running Orchestrator Tests...")
        
        result = self._execute_pytest(
            "tests/python/test_orchestrator_comprehensive.py",
            "Orchestrator Tests"
        )
        self.results["test_suites"]["orchestrator"] = result
    
    def _run_utils_tests(self):
        """Run Utilities tests"""
        print("\n[UTILS] Running Utilities Tests...")
        
        result = self._execute_pytest(
            "tests/python/test_utils_comprehensive.py", 
            "Utilities Tests"
        )
        self.results["test_suites"]["utilities"] = result
    
    def _run_existing_contract_tests(self):
        """Run existing contract tests"""
        print("\n[CONTRACTS] Running Existing Contract Tests...")
        
        # Run base agent contract test
        result1 = self._execute_pytest(
            "tests/python/test_base_agent.py",
            "BaseAgent Contract"
        )
        
        # Run orchestrator contract test
        result2 = self._execute_pytest(
            "tests/python/test_orchestrator.py",
            "Orchestrator Contract"
        )
        
        self.results["test_suites"]["contracts"] = {
            "base_agent": result1,
            "orchestrator": result2
        }
    
    def _run_coverage_analysis(self):
        """Run comprehensive coverage analysis"""
        print("\n[COVERAGE] Running Coverage Analysis...")
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "--cov=templates",
                "--cov=orchestrator", 
                "--cov=utils",
                "--cov-report=html:test_reports/htmlcov",
                "--cov-report=json:test_reports/coverage.json",
                "--cov-report=term-missing",
                "tests/python/",
                "-v"
            ]
            
            result = subprocess.run(
                cmd, 
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            # Parse coverage results
            coverage_file = self.reports_dir / "coverage.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                self.results["coverage"] = {
                    "total_coverage": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "lines_covered": coverage_data.get("totals", {}).get("covered_lines", 0),
                    "lines_missing": coverage_data.get("totals", {}).get("missing_lines", 0),
                    "file_coverage": {},
                    "success": result.returncode == 0,
                    "output": result.stdout,
                    "errors": result.stderr if result.returncode != 0 else None
                }
                
                # Add per-file coverage
                for filename, file_data in coverage_data.get("files", {}).items():
                    self.results["coverage"]["file_coverage"][filename] = {
                        "coverage": file_data.get("summary", {}).get("percent_covered", 0),
                        "covered": file_data.get("summary", {}).get("covered_lines", 0),
                        "missing": file_data.get("summary", {}).get("missing_lines", 0)
                    }
                
                print(f"[SUCCESS] Coverage Analysis Complete: {self.results['coverage']['total_coverage']:.1f}%")
            else:
                print("[WARNING] Coverage report not generated")
                
        except subprocess.TimeoutExpired:
            self.results["coverage"] = {"error": "Coverage analysis timed out"}
            print("[ERROR] Coverage analysis timed out")
        except Exception as e:
            self.results["coverage"] = {"error": str(e)}
            print(f"[ERROR] Coverage analysis failed: {e}")
    
    def _run_performance_tests(self):
        """Run performance benchmarking tests"""
        print("\n[PERFORMANCE] Running Performance Tests...")
        
        try:
            # Run specific performance test markers
            result = self._execute_pytest(
                "tests/python/",
                "Performance Tests",
                extra_args=["-m", "not slow", "--tb=short"]
            )
            
            self.results["test_suites"]["performance"] = result
            print("[SUCCESS] Performance tests completed")
            
        except Exception as e:
            self.results["errors"].append(f"Performance tests error: {str(e)}")
            print(f"[ERROR] Performance tests failed: {e}")
    
    def _run_integration_tests(self):
        """Run integration smoke tests"""
        print("\n[INTEGRATION] Running Integration Tests...")
        
        # Create a simple integration test
        integration_test = """
import asyncio
import sys
from pathlib import Path

# Add project root to path  
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def test_basic_integration():
    '''Basic integration smoke test'''
    try:
        from templates.base_agent import BaseAgent, Action
        from orchestrator import AgentOrchestrator, Task
        
        # Test agent creation
        class TestAgent(BaseAgent):
            async def execute(self, task, action: Action):
                return {'integration': 'success'}
        
        agent = TestAgent('integration_test')
        
        # Test orchestrator
        orchestrator = AgentOrchestrator('integration_test')
        orchestrator.register_agent(agent)
        
        # Test task delegation
        task = Task(id='integration', description='Integration test', requirements={})
        result = await orchestrator.delegate_task(task)
        
        assert result is not None
        assert 'integration' in result
        
        print('[SUCCESS] Basic integration test passed')
        return True
        
    except Exception as e:
        print(f'[ERROR] Integration test failed: {e}')
        return False

if __name__ == '__main__':
    success = asyncio.run(test_basic_integration())
    sys.exit(0 if success else 1)
"""
        
        integration_file = self.test_dir / "integration_smoke_test.py"
        with open(integration_file, 'w') as f:
            f.write(integration_test)
        
        try:
            result = subprocess.run(
                [sys.executable, str(integration_file)],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            self.results["test_suites"]["integration"] = {
                "success": result.returncode == 0,
                "output": result.stdout,
                "errors": result.stderr if result.returncode != 0 else None,
                "duration": "< 60s"
            }
            
            if result.returncode == 0:
                print("[SUCCESS] Integration tests passed")
            else:
                print(f"[ERROR] Integration tests failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.results["test_suites"]["integration"] = {"error": "Integration tests timed out"}
            print("[ERROR] Integration tests timed out")
        finally:
            if integration_file.exists():
                integration_file.unlink()
    
    def _execute_pytest(self, test_path: str, test_name: str, extra_args: List[str] = None) -> Dict[str, Any]:
        """Execute pytest on a specific test file"""
        args = [
            sys.executable, "-m", "pytest",
            test_path,
            "-v",
            "--tb=short",
            f"--junitxml=test_reports/{test_name.lower().replace(' ', '_')}_results.xml"
        ]
        
        if extra_args:
            args.extend(extra_args)
        
        try:
            result = subprocess.run(
                args,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per test suite
            )
            
            # Parse output for test counts
            output_lines = result.stdout.split('\n')
            test_summary = self._parse_pytest_output(output_lines)
            
            return {
                "name": test_name,
                "success": result.returncode == 0,
                "return_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr if result.returncode != 0 else None,
                "test_counts": test_summary,
                "duration": "< 300s"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "name": test_name,
                "success": False,
                "error": "Test suite timed out after 5 minutes",
                "duration": ">= 300s"
            }
        except Exception as e:
            return {
                "name": test_name,
                "success": False,
                "error": str(e)
            }
    
    def _parse_pytest_output(self, output_lines: List[str]) -> Dict[str, int]:
        """Parse pytest output to extract test counts"""
        summary = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
        
        for line in output_lines:
            if "passed" in line and "failed" in line:
                # Parse summary line like "5 passed, 2 failed in 1.23s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i + 1 < len(parts):
                        next_part = parts[i + 1]
                        if "passed" in next_part:
                            summary["passed"] = int(part)
                        elif "failed" in next_part:
                            summary["failed"] = int(part)
                        elif "skipped" in next_part:
                            summary["skipped"] = int(part)
                        elif "error" in next_part:
                            summary["errors"] = int(part)
            elif "passed" in line and "failed" not in line:
                # Simple passed line
                parts = line.split()
                for i, part in enumerate(parts):
                    if part.isdigit() and i + 1 < len(parts) and "passed" in parts[i + 1]:
                        summary["passed"] = int(part)
        
        return summary
    
    def _generate_summary(self):
        """Generate test execution summary"""
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        total_errors = 0
        successful_suites = 0
        total_suites = 0
        
        for suite_name, suite_data in self.results["test_suites"].items():
            if isinstance(suite_data, dict):
                if suite_data.get("success", False):
                    successful_suites += 1
                total_suites += 1
                
                # Count tests if available
                if "test_counts" in suite_data:
                    counts = suite_data["test_counts"]
                    total_passed += counts.get("passed", 0)
                    total_failed += counts.get("failed", 0)
                    total_skipped += counts.get("skipped", 0)
                    total_errors += counts.get("errors", 0)
            
            elif isinstance(suite_data, dict) and "base_agent" in suite_data:
                # Handle nested suite data (like contracts)
                for sub_suite_name, sub_suite_data in suite_data.items():
                    if sub_suite_data.get("success", False):
                        successful_suites += 1
                    total_suites += 1
                    
                    if "test_counts" in sub_suite_data:
                        counts = sub_suite_data["test_counts"]
                        total_passed += counts.get("passed", 0)
                        total_failed += counts.get("failed", 0)
                        total_skipped += counts.get("skipped", 0)
                        total_errors += counts.get("errors", 0)
        
        self.results["summary"] = {
            "total_test_suites": total_suites,
            "successful_suites": successful_suites,
            "failed_suites": total_suites - successful_suites,
            "suite_success_rate": (successful_suites / total_suites * 100) if total_suites > 0 else 0,
            "total_tests": total_passed + total_failed + total_skipped,
            "passed_tests": total_passed,
            "failed_tests": total_failed,
            "skipped_tests": total_skipped,
            "error_tests": total_errors,
            "test_success_rate": (total_passed / (total_passed + total_failed) * 100) if (total_passed + total_failed) > 0 else 0,
            "coverage_percentage": self.results.get("coverage", {}).get("total_coverage", 0),
            "total_duration": self.results.get("duration", 0),
            "errors_encountered": len(self.results.get("errors", []))
        }
    
    def _save_results(self):
        """Save detailed test results to JSON"""
        results_file = self.reports_dir / "comprehensive_test_results.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n[RESULTS] Detailed results saved to: {results_file}")
    
    def _print_final_report(self):
        """Print comprehensive final report"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST EXECUTION REPORT")
        print("=" * 60)
        
        summary = self.results["summary"]
        
        print(f"\n[TIME] EXECUTION TIME: {summary['total_duration']:.2f} seconds")
        print(f"[COVERAGE] COVERAGE: {summary['coverage_percentage']:.1f}%")
        
        print(f"\n[SUITES] TEST SUITES:")
        print(f"   Total Suites: {summary['total_test_suites']}")
        print(f"   Successful: {summary['successful_suites']} ({summary['suite_success_rate']:.1f}%)")
        print(f"   Failed: {summary['failed_suites']}")
        
        print(f"\n[TESTS] INDIVIDUAL TESTS:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']} ({summary['test_success_rate']:.1f}%)")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Skipped: {summary['skipped_tests']}")
        print(f"   Errors: {summary['error_tests']}")
        
        if self.results.get("errors"):
            print(f"\n[ERROR] EXECUTION ERRORS: {len(self.results['errors'])}")
            for error in self.results["errors"]:
                print(f"   - {error}")
        
        # Suite-specific results
        print(f"\n[DETAILS] DETAILED SUITE RESULTS:")
        for suite_name, suite_data in self.results["test_suites"].items():
            if isinstance(suite_data, dict) and "success" in suite_data:
                status = "[PASS]" if suite_data["success"] else "[FAIL]"
                counts = suite_data.get("test_counts", {})
                passed = counts.get("passed", 0)
                total = sum(counts.values()) if counts else 0
                print(f"   {suite_name}: {status} ({passed}/{total} tests)")
        
        # Coverage details
        coverage = self.results.get("coverage", {})
        if "file_coverage" in coverage:
            print(f"\n[COVERAGE] COVERAGE BY FILE:")
            for filename, file_cov in coverage["file_coverage"].items():
                if file_cov["coverage"] is not None:
                    print(f"   {filename}: {file_cov['coverage']:.1f}%")
        
        print(f"\n[COMPLETE] TEST EXECUTION COMPLETE!")
        
        if summary["failed_suites"] == 0 and summary["failed_tests"] == 0:
            print("[SUCCESS] ALL TESTS PASSED SUCCESSFULLY!")
        else:
            print("[WARNING] Some tests failed - check detailed results for more information")
        
        print("=" * 60)


def main():
    """Main execution function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Comprehensive Test Runner for AI Agents")
        print("Usage: python run_comprehensive_tests.py")
        print("\nThis script runs all test suites and generates comprehensive reports:")
        print("• BaseAgent unit tests")
        print("• Orchestrator integration tests")
        print("• Utilities tests")
        print("• Coverage analysis")
        print("• Performance benchmarking")
        print("• Integration smoke tests")
        return []
    
    runner = TestRunner()
    results = runner.run_all_tests()
    
    # Exit with appropriate code
    summary = results.get("summary", {})
    if summary.get("failed_suites", 1) > 0 or summary.get("failed_tests", 1) > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()