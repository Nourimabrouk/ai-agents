"""
Phase 7 Test Configuration and Fixtures
Common test fixtures and configuration for all Phase 7 tests
"""

import pytest
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, Mock
import logging

# Configure test logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Phase 7 components for fixtures
from core.autonomous.orchestrator import AutonomousMetaOrchestrator, AutonomyLevel
from core.autonomous.safety import AutonomousSafetyFramework, SafetyLevel
from core.security.autonomous_security import AutonomousSecurityFramework, SecurityLevel
from core.reasoning.integrated_reasoning_controller import IntegratedReasoningController
from templates.base_agent import BaseAgent


# Test configuration
pytest_plugins = ['pytest_asyncio']


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def temp_directory():
    """Create temporary directory for test files"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def safety_framework():
    """Create safety framework for testing"""
    framework = AutonomousSafetyFramework(safety_level=SafetyLevel.MODERATE)
    await framework.initialize()
    return framework


@pytest.fixture 
async def security_framework():
    """Create security framework for testing"""
    framework = AutonomousSecurityFramework(security_level=SecurityLevel.MEDIUM)
    await framework.initialize()
    return framework


@pytest.fixture
async def autonomous_orchestrator(safety_framework):
    """Create autonomous orchestrator for testing"""
    orchestrator = AutonomousMetaOrchestrator(
        autonomy_level=AutonomyLevel.SEMI_AUTONOMOUS,
        safety_framework=safety_framework
    )
    await orchestrator.initialize()
    return orchestrator


@pytest.fixture
async def reasoning_controller():
    """Create integrated reasoning controller for testing"""
    controller = IntegratedReasoningController(
        causal_reasoning_enabled=True,
        temporal_reasoning_enabled=True,
        working_memory_enabled=True
    )
    await controller.initialize()
    return controller


@pytest.fixture
def mock_test_agent():
    """Create mock test agent"""
    
    class MockTestAgent(BaseAgent):
        def __init__(self, agent_id: str = "test_agent"):
            super().__init__(agent_id)
            self.execution_history = []
            self.performance_metrics = {
                'success_rate': 0.85,
                'avg_response_time': 1.2,
                'quality_score': 0.80
            }
            
        async def execute_task(self, task):
            result = {
                'task_id': task.get('task_id', 'unknown'),
                'success': True,
                'execution_time': 1.0,
                'timestamp': datetime.now()
            }
            self.execution_history.append(result)
            return result
            
        async def get_performance_metrics(self):
            return self.performance_metrics.copy()
            
    return MockTestAgent()


@pytest.fixture
def sample_test_data():
    """Sample test data for various test scenarios"""
    return {
        'causal_data': {
            'variables': ['X', 'Y', 'Z'],
            'relationships': [('X', 'Y', 0.8), ('Y', 'Z', 0.6)],
            'sample_size': 1000
        },
        'performance_data': {
            'target_response_time': 1.0,
            'target_throughput': 100,
            'target_success_rate': 0.95
        },
        'security_scenarios': [
            {'type': 'privilege_escalation', 'severity': 'high'},
            {'type': 'data_exfiltration', 'severity': 'critical'},
            {'type': 'code_injection', 'severity': 'high'}
        ]
    }


@pytest.fixture
async def test_database():
    """Create test database for integration tests"""
    # Mock database for testing
    class TestDatabase:
        def __init__(self):
            self.data = {}
            
        async def store(self, key: str, value: Any):
            self.data[key] = value
            
        async def retrieve(self, key: str):
            return self.data.get(key)
            
        async def clear(self):
            self.data.clear()
            
    db = TestDatabase()
    yield db
    await db.clear()


# Test markers
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and scalability tests"
    )
    config.addinivalue_line(
        "markers", "security: Security and safety validation tests"
    )
    config.addinivalue_line(
        "markers", "e2e: End-to-end workflow tests"
    )
    config.addinivalue_line(
        "markers", "slow: Slow-running tests (>5 seconds)"
    )


# Test utilities
class TestMetrics:
    """Test metrics collection utility"""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
        
    def start_timer(self, metric_name: str):
        self.start_times[metric_name] = datetime.now()
        
    def end_timer(self, metric_name: str):
        if metric_name in self.start_times:
            elapsed = datetime.now() - self.start_times[metric_name]
            self.metrics[metric_name] = elapsed.total_seconds()
            del self.start_times[metric_name]
            
    def record_metric(self, metric_name: str, value: Any):
        self.metrics[metric_name] = value
        
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()


@pytest.fixture
def test_metrics():
    """Test metrics collection fixture"""
    return TestMetrics()


# Async test helpers
async def wait_for_condition(condition_func, timeout: float = 5.0, interval: float = 0.1):
    """Wait for a condition to become true"""
    import time
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        if await condition_func() if asyncio.iscoroutinefunction(condition_func) else condition_func():
            return True
        await asyncio.sleep(interval)
        
    return False


@pytest.fixture
def wait_for():
    """Wait for condition fixture"""
    return wait_for_condition


# Performance test helpers
class PerformanceMonitor:
    """Monitor system performance during tests"""
    
    def __init__(self):
        import psutil
        self.process = psutil.Process()
        self.snapshots = []
        
    def take_snapshot(self, label: str = ""):
        """Take a performance snapshot"""
        snapshot = {
            'label': label,
            'timestamp': datetime.now(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'cpu_percent': self.process.cpu_percent(),
            'open_files': len(self.process.open_files()) if hasattr(self.process, 'open_files') else 0
        }
        self.snapshots.append(snapshot)
        return snapshot
        
    def get_memory_growth(self) -> float:
        """Calculate memory growth between first and last snapshot"""
        if len(self.snapshots) < 2:
            return 0.0
        return self.snapshots[-1]['memory_mb'] - self.snapshots[0]['memory_mb']
        
    def get_peak_memory(self) -> float:
        """Get peak memory usage"""
        if not self.snapshots:
            return 0.0
        return max(s['memory_mb'] for s in self.snapshots)


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture"""
    monitor = PerformanceMonitor()
    monitor.take_snapshot("test_start")
    yield monitor
    monitor.take_snapshot("test_end")


# Error injection helpers for testing robustness
class ErrorInjector:
    """Inject controlled errors for robustness testing"""
    
    def __init__(self):
        self.error_patterns = []
        
    def add_error_pattern(self, condition_func, error_type: Exception, probability: float = 1.0):
        """Add an error pattern that triggers under certain conditions"""
        self.error_patterns.append({
            'condition': condition_func,
            'error_type': error_type,
            'probability': probability
        })
        
    async def check_and_inject_error(self, context: Dict[str, Any]):
        """Check conditions and inject errors if patterns match"""
        import random
        
        for pattern in self.error_patterns:
            condition_met = (
                await pattern['condition'](context) 
                if asyncio.iscoroutinefunction(pattern['condition']) 
                else pattern['condition'](context)
            )
            
            if condition_met and random.random() < pattern['probability']:
                raise pattern['error_type']("Injected error for testing")


@pytest.fixture
def error_injector():
    """Error injection fixture"""
    return ErrorInjector()


# Test data generators
def generate_causal_test_data(n_samples: int = 1000, seed: int = 42):
    """Generate synthetic causal test data"""
    import numpy as np
    import pandas as pd
    
    np.random.seed(seed)
    
    # Simple causal structure: X -> Y -> Z
    x = np.random.normal(0, 1, n_samples)
    y = 0.7 * x + np.random.normal(0, 0.3, n_samples)
    z = 0.6 * y + np.random.normal(0, 0.4, n_samples)
    
    return pd.DataFrame({'X': x, 'Y': y, 'Z': z})


def generate_performance_test_tasks(num_tasks: int = 100, complexity_range: tuple = (0.1, 0.9)):
    """Generate performance test tasks"""
    import random
    
    tasks = []
    for i in range(num_tasks):
        complexity = random.uniform(*complexity_range)
        task = {
            'task_id': f'perf_task_{i}',
            'complexity': complexity,
            'expected_time': complexity * 2.0,
            'requirements': random.choice(['analysis', 'optimization', 'coordination'])
        }
        tasks.append(task)
        
    return tasks


@pytest.fixture
def test_data_generators():
    """Test data generation utilities"""
    return {
        'causal_data': generate_causal_test_data,
        'performance_tasks': generate_performance_test_tasks
    }


# Cleanup helpers
@pytest.fixture(autouse=True)
async def cleanup_after_test():
    """Automatic cleanup after each test"""
    yield
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Clear any global state (if needed)
    # This would be customized based on actual global state in the system