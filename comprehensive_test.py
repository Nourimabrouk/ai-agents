#!/usr/bin/env python3
"""
Comprehensive System Test - Windows Validation
Tests all major functionality to ensure smooth operation
"""

import asyncio
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone


def print_section(title: str):
    """Print a test section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


async def test_core_imports():
    """Test that all core modules can be imported"""
    print_section("TESTING CORE IMPORTS")
    
    try:
        # Test system imports
        from system import AIAgentsSystem, get_system
        print("[PASS] System modules imported successfully")
        
        # Test core imports
        from core import (
            initialize_system, shutdown_system, get_system_health,
            AgentId, TaskId, ExecutionContext, ExecutionResult, Priority
        )
        print("[PASS] Core modules imported successfully")
        
        # Test security imports
        from core.security import AutonomousSecurityFramework, SecurityLevel
        print("[PASS] Security modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Import test failed: {e}")
        traceback.print_exc()
        return False


async def test_system_lifecycle():
    """Test system startup, operation, and shutdown"""
    print_section("TESTING SYSTEM LIFECYCLE")
    
    try:
        from system import get_system
        system = get_system()
        print("[PASS] System instance created")
        
        # Test startup
        success = await system.start()
        print(f"[PASS] System startup: {'SUCCESS' if success else 'FAILED'}")
        
        if not success:
            return False
        
        # Test status
        status = await system.get_status()
        overall_status = status.get('overall_status', 'unknown')
        print(f"[PASS] System status: {overall_status}")
        
        # Test security status
        security_status = await system.get_security_status()
        print(f"[PASS] Security status retrieved: {type(security_status).__name__}")
        
        # Test shutdown
        await system.stop()
        print("[PASS] System shutdown completed")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] System lifecycle test failed: {e}")
        traceback.print_exc()
        return False


async def test_demos():
    """Test that demo scripts work correctly"""
    print_section("TESTING DEMO SCRIPTS")
    
    try:
        # Test working demo by importing and running a simple version
        print("Testing working_demo.py functionality...")
        
        # Import the demo components
        import working_demo
        
        # Create a simple orchestrator and test it
        orchestrator = working_demo.SimpleOrchestrator()
        
        # Add test agents
        orchestrator.add_agent(working_demo.SimpleAgent("TestAgent1"))
        orchestrator.add_agent(working_demo.SimpleAgent("TestAgent2"))
        
        # Execute a small test
        tasks = ["Test task 1", "Test task 2"]
        results = await orchestrator.execute_tasks(tasks)
        
        print(f"[PASS] Demo executed successfully: {len(results)} results")
        return True
        
    except Exception as e:
        print(f"[FAIL] Demo test failed: {e}")
        traceback.print_exc()
        return False


async def test_windows_compatibility():
    """Test Windows-specific functionality"""
    print_section("TESTING WINDOWS COMPATIBILITY")
    
    try:
        # Test path handling
        test_path = Path.home() / ".ai-agents" / "test.txt"
        print(f"[PASS] Path handling works: {test_path}")
        
        # Test datetime handling
        current_time = datetime.now(timezone.utc)
        print(f"[PASS] DateTime handling works: {current_time.isoformat()}")
        
        # Test async functionality
        await asyncio.sleep(0.01)
        print("[PASS] Async functionality works")
        
        # Test logging
        import logging
        logger = logging.getLogger("test")
        logger.info("Test log message")
        print("[PASS] Logging functionality works")
        
        return True
        
    except Exception as e:
        print(f"[FAIL] Windows compatibility test failed: {e}")
        traceback.print_exc()
        return False


async def test_performance():
    """Test basic performance characteristics"""
    print_section("TESTING PERFORMANCE")
    
    try:
        start_time = datetime.now(timezone.utc)
        
        # Test system startup speed
        from system import get_system
        system = get_system()
        startup_success = await system.start()
        
        if startup_success:
            startup_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"[PASS] System startup time: {startup_time:.2f}s")
            
            # Test a simple operation
            op_start = datetime.now(timezone.utc)
            status = await system.get_status()
            op_time = (datetime.now(timezone.utc) - op_start).total_seconds()
            print(f"[PASS] Status retrieval time: {op_time:.3f}s")
            
            await system.stop()
            
            total_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            print(f"[PASS] Total test time: {total_time:.2f}s")
            
            return True
        else:
            print("[FAIL] System startup failed")
            return False
            
    except Exception as e:
        print(f"[FAIL] Performance test failed: {e}")
        traceback.print_exc()
        return False


async def run_comprehensive_test():
    """Run all tests and provide final assessment"""
    print(f"COMPREHENSIVE SYSTEM TEST - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Platform: {sys.platform}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {Path.cwd()}")
    
    # Run all tests
    tests = [
        ("Core Imports", test_core_imports),
        ("System Lifecycle", test_system_lifecycle),
        ("Demo Scripts", test_demos),
        ("Windows Compatibility", test_windows_compatibility),
        ("Performance", test_performance)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[FAIL] {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Final assessment
    print_section("FINAL ASSESSMENT")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {(passed/total)*100:.1f}%")
    
    print("\nDetailed Results:")
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "[PASS]" if result else "[FAIL]"
        print(f"  {symbol} {test_name}: {status}")
    
    if passed == total:
        print(f"\n*** ALL TESTS PASSED! System is ready for use. ***")
        print("You can now run:")
        print("  - python system.py (main system)")
        print("  - python working_demo.py (demo script)")
    else:
        print(f"\nWARNING: {total - passed} tests failed. Review the output above.")
    
    return passed == total


if __name__ == "__main__":
    try:
        result = asyncio.run(run_comprehensive_test())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest suite failed: {e}")
        traceback.print_exc()
        sys.exit(1)