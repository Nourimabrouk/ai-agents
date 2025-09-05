"""
Test package for Phase 7 Advanced Reasoning Systems
Comprehensive test suite including unit tests, validation, and benchmarking
"""

from pathlib import Path

# Test configuration
TEST_DIR = Path(__file__).parent
REPORTS_DIR = TEST_DIR.parent / "test_reports"
REASONING_SYSTEMS_DIR = TEST_DIR.parent.parent / "core" / "reasoning"

# Ensure required directories exist
REPORTS_DIR.mkdir(exist_ok=True)

# Test suite information
TEST_SUITES = {
    "unit_tests": "Comprehensive unit and integration tests for all reasoning components",
    "validation_tests": "Accuracy validation against ground truth datasets", 
    "performance_benchmarks": "Performance testing under various load conditions"
}

# Performance targets
PERFORMANCE_TARGETS = {
    "causal_accuracy": 0.90,
    "working_memory_tokens": 10000,
    "response_time_simple": 1.0,  # seconds
    "overall_success_rate": 0.85,
    "reliability_threshold": 0.95
}

__version__ = "1.0.0"
__author__ = "Claude Code - Phase 7 Implementation"