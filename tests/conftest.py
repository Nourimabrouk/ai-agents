"""
Test configuration and fixtures for AI Agents test suite
Sets up proper imports and shared test fixtures
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import asyncio
import tempfile
from typing import Any, Dict

# Import modules to verify they're available
try:
    from templates.base_agent import BaseAgent, Action, Observation, Thought
    from orchestrator import AgentOrchestrator, Task
    from utils.observability.logging import get_logger
    from utils.observability.metrics import MetricsCollector
    from utils.persistence.memory_store import SqliteMemoryStore
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Provide a temporary database path for tests"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.db') as f:
        db_path = f.name
    yield db_path
    try:
        os.unlink(db_path)
    except OSError:
        pass


@pytest.fixture
def mock_agent():
    """Create a simple mock agent for testing"""
    
    class MockAgent(BaseAgent):
        async def execute(self, task: Any, action: Action) -> Dict[str, Any]:
            return {
                "agent": self.name,
                "task": str(task),
                "success": True,
                "mock": True
            }
    
    return MockAgent("mock_agent")


@pytest.fixture
def sample_task():
    """Create a sample task for testing"""
    return Task(
        id="test_task",
        description="Sample test task",
        requirements={"test": True}
    )


@pytest.fixture
def test_orchestrator():
    """Create an orchestrator for testing"""
    return AgentOrchestrator("test_orchestrator")


@pytest.fixture
def test_logger():
    """Create a test logger"""
    return get_logger("test_logger")


@pytest.fixture
def test_metrics():
    """Create test metrics collector"""
    return MetricsCollector("test_metrics")


@pytest.fixture  
def memory_store(temp_db_path):
    """Create a memory store for testing"""
    return SqliteMemoryStore(temp_db_path)


# Configure asyncio for Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items during collection"""
    for item in items:
        # Mark async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Add markers based on test names
        if "performance" in item.name.lower():
            item.add_marker(pytest.mark.performance)
        if "integration" in item.name.lower():
            item.add_marker(pytest.mark.integration)
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)