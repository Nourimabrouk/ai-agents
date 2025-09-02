#!/usr/bin/env python3
"""
Performance Testing and Optimization Script
Tests system performance and validates optimizations
"""

import time
import asyncio
import statistics
import psutil
import sys
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import json

class PerformanceTestRunner:
    """Comprehensive performance testing suite"""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get system performance information"""
        return {
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
            "memory_percent": psutil.virtual_memory().percent,
            "platform": sys.platform,
            "python_version": sys.version.split()[0]
        }
    
    def test_import_performance(self) -> Dict[str, float]:
        """Test module import performance"""
        print("[PERF] Testing import performance...")
        
        modules_to_test = [
            "orchestrator",
            "agents.accountancy.invoice_processor", 
            "master_platform",
            "demo.launch_ultimate_demo"
        ]
        
        import_times = {}
        for module in modules_to_test:
            start_time = time.time()
            try:
                __import__(module)
                import_time = time.time() - start_time
                import_times[module] = import_time
                print(f"   {module}: {import_time:.3f}s")
            except Exception as e:
                print(f"   {module}: FAILED ({e})")
                import_times[module] = None
        
        avg_import_time = statistics.mean([t for t in import_times.values() if t is not None])
        print(f"   Average import time: {avg_import_time:.3f}s")
        
        return import_times
    
    async def test_async_performance(self) -> Dict[str, float]:
        """Test async operation performance"""
        print("[PERF] Testing async performance...")
        
        # Test concurrent task execution
        async def dummy_task(task_id: int, duration: float = 0.1):
            await asyncio.sleep(duration)
            return f"task_{task_id}"
        
        # Test different concurrency levels
        concurrency_tests = [5, 10, 20, 50]
        async_results = {}
        
        for concurrency in concurrency_tests:
            start_time = time.time()
            tasks = [dummy_task(i) for i in range(concurrency)]
            results = await asyncio.gather(*tasks)
            elapsed_time = time.time() - start_time
            
            async_results[f"concurrency_{concurrency}"] = elapsed_time
            print(f"   Concurrency {concurrency}: {elapsed_time:.3f}s")
        
        return async_results
    
    def test_file_operations(self) -> Dict[str, float]:
        """Test file operation performance"""
        print("[PERF] Testing file operations...")
        
        test_data = "x" * 10000  # 10KB test data
        test_files = []
        file_ops = {}
        
        # Test file writing
        start_time = time.time()
        for i in range(100):
            test_file = Path(f"temp_test_{i}.txt")
            test_files.append(test_file)
            test_file.write_text(test_data)
        write_time = time.time() - start_time
        file_ops["write_100_files"] = write_time
        print(f"   Write 100 files: {write_time:.3f}s")
        
        # Test file reading
        start_time = time.time()
        for test_file in test_files:
            content = test_file.read_text()
        read_time = time.time() - start_time
        file_ops["read_100_files"] = read_time
        print(f"   Read 100 files: {read_time:.3f}s")
        
        # Cleanup
        start_time = time.time()
        for test_file in test_files:
            if test_file.exists():
                test_file.unlink()
        cleanup_time = time.time() - start_time
        file_ops["cleanup_100_files"] = cleanup_time
        print(f"   Cleanup 100 files: {cleanup_time:.3f}s")
        
        return file_ops
    
    def test_cpu_intensive_operations(self) -> Dict[str, float]:
        """Test CPU-intensive operations"""
        print("[PERF] Testing CPU-intensive operations...")
        
        def fibonacci(n):
            if n <= 1:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        def cpu_task(iterations=1000000):
            total = 0
            for i in range(iterations):
                total += i * i
            return total
        
        cpu_results = {}
        
        # Test single-threaded CPU task
        start_time = time.time()
        result = cpu_task()
        single_thread_time = time.time() - start_time
        cpu_results["single_thread_cpu"] = single_thread_time
        print(f"   Single-thread CPU task: {single_thread_time:.3f}s")
        
        # Test multi-threaded CPU task
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_task, 250000) for _ in range(4)]
            results = [f.result() for f in futures]
        multi_thread_time = time.time() - start_time
        cpu_results["multi_thread_cpu"] = multi_thread_time
        print(f"   Multi-thread CPU task: {multi_thread_time:.3f}s")
        
        # Test small recursive task (Fibonacci)
        start_time = time.time()
        fib_result = fibonacci(30)
        fibonacci_time = time.time() - start_time
        cpu_results["fibonacci_30"] = fibonacci_time
        print(f"   Fibonacci(30): {fibonacci_time:.3f}s")
        
        return cpu_results
    
    def test_memory_usage(self) -> Dict[str, Any]:
        """Test memory usage patterns"""
        print("[PERF] Testing memory usage...")
        
        initial_memory = psutil.virtual_memory().percent
        
        # Test memory allocation
        large_lists = []
        start_time = time.time()
        
        for i in range(100):
            large_list = [j for j in range(10000)]
            large_lists.append(large_list)
        
        allocation_time = time.time() - start_time
        peak_memory = psutil.virtual_memory().percent
        
        # Cleanup
        del large_lists
        final_memory = psutil.virtual_memory().percent
        
        memory_results = {
            "initial_memory_percent": initial_memory,
            "peak_memory_percent": peak_memory,
            "final_memory_percent": final_memory,
            "memory_increase_percent": peak_memory - initial_memory,
            "allocation_time_seconds": allocation_time
        }
        
        print(f"   Initial memory: {initial_memory:.1f}%")
        print(f"   Peak memory: {peak_memory:.1f}%")
        print(f"   Final memory: {final_memory:.1f}%")
        print(f"   Memory increase: {peak_memory - initial_memory:.1f}%")
        
        return memory_results
    
    def analyze_performance(self) -> Dict[str, str]:
        """Analyze performance results and provide recommendations"""
        analysis = {
            "overall_status": "GOOD",
            "recommendations": [],
            "warnings": []
        }
        
        # Analyze import performance
        if "import_performance" in self.results:
            import_times = [t for t in self.results["import_performance"].values() if t is not None]
            if import_times and statistics.mean(import_times) > 2.0:
                analysis["warnings"].append("Import times are high - consider optimizing module imports")
                analysis["overall_status"] = "MODERATE"
        
        # Analyze memory usage
        if "memory_usage" in self.results:
            memory_increase = self.results["memory_usage"].get("memory_increase_percent", 0)
            if memory_increase > 20:
                analysis["warnings"].append("High memory usage detected - monitor memory consumption")
                analysis["overall_status"] = "MODERATE"
        
        # General recommendations
        analysis["recommendations"].extend([
            "Use async operations for I/O-bound tasks",
            "Implement connection pooling for database operations", 
            "Cache frequently accessed data",
            "Monitor system resources during production use"
        ])
        
        return analysis
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all performance tests"""
        print("=" * 60)
        print("[PERFORMANCE] AI Agents Platform Performance Testing")
        print("=" * 60)
        print()
        
        # System information
        self.results["system_info"] = self.get_system_info()
        print("[INFO] System Information:")
        for key, value in self.results["system_info"].items():
            print(f"   {key}: {value}")
        print()
        
        # Run tests
        self.results["import_performance"] = self.test_import_performance()
        print()
        
        self.results["async_performance"] = await self.test_async_performance()
        print()
        
        self.results["file_operations"] = self.test_file_operations()
        print()
        
        self.results["cpu_operations"] = self.test_cpu_intensive_operations()
        print()
        
        self.results["memory_usage"] = self.test_memory_usage()
        print()
        
        # Analysis
        self.results["analysis"] = self.analyze_performance()
        
        # Total test time
        total_time = time.time() - self.start_time
        self.results["total_test_time"] = total_time
        
        return self.results
    
    def save_results(self, filename: str = "performance_test_results.json"):
        """Save performance test results to file"""
        try:
            with open(filename, "w") as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"[SAVE] Results saved to {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to save results: {e}")
    
    def print_summary(self):
        """Print performance test summary"""
        print("=" * 60)
        print("[SUMMARY] Performance Test Summary")
        print("=" * 60)
        
        analysis = self.results.get("analysis", {})
        status = analysis.get("overall_status", "UNKNOWN")
        
        print(f"[STATUS] Overall Performance: {status}")
        print(f"[TIME] Total Test Time: {self.results.get('total_test_time', 0):.1f}s")
        print()
        
        # Warnings
        warnings = analysis.get("warnings", [])
        if warnings:
            print("[WARNINGS]")
            for warning in warnings:
                print(f"   - {warning}")
            print()
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            print("[RECOMMENDATIONS]")
            for rec in recommendations:
                print(f"   - {rec}")
            print()
        
        # Key metrics
        print("[KEY METRICS]")
        if "import_performance" in self.results:
            import_times = [t for t in self.results["import_performance"].values() if t is not None]
            if import_times:
                print(f"   Average import time: {statistics.mean(import_times):.3f}s")
        
        if "memory_usage" in self.results:
            memory_increase = self.results["memory_usage"].get("memory_increase_percent", 0)
            print(f"   Memory usage increase: {memory_increase:.1f}%")
        
        if "cpu_operations" in self.results:
            single_thread = self.results["cpu_operations"].get("single_thread_cpu", 0)
            multi_thread = self.results["cpu_operations"].get("multi_thread_cpu", 0)
            if single_thread > 0 and multi_thread > 0:
                speedup = single_thread / multi_thread
                print(f"   Multi-threading speedup: {speedup:.2f}x")
        
        print("=" * 60)

async def main():
    """Main performance testing function"""
    runner = PerformanceTestRunner()
    
    try:
        # Run all tests
        results = await runner.run_all_tests()
        
        # Save and summarize
        runner.save_results()
        runner.print_summary()
        
        # Return status
        status = results.get("analysis", {}).get("overall_status", "UNKNOWN")
        if status == "GOOD":
            return 0
        elif status == "MODERATE":
            return 0  # Still acceptable
        else:
            return 1
            
    except Exception as e:
        print(f"[ERROR] Performance testing failed: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)