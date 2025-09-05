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
from typing import Any, Dict, Generator, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

# Import modules to verify they're available
try:
    from templates.base_agent import BaseAgent, Action, Observation, Thought
    from orchestrator import AgentOrchestrator, Task
    from utils.observability.logging import get_logger
    from utils.observability.metrics import MetricsCollector
    from utils.persistence.memory_store import SqliteMemoryStore
except ImportError as e:
    print(f"Warning: Could not import all modules: {e}")

# API testing imports
try:
    import pytest_asyncio
    from fastapi.testclient import TestClient
    from httpx import AsyncClient
    from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    
    # API imports
    from api.main import app
    from api.models.database_models import Base, User, Organization
    from api.database.session import get_database
    from api.config import get_settings, TestingSettings
    API_IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: API testing modules not available: {e}")
    API_IMPORTS_AVAILABLE = False


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
    logger.info(f'Method {function_name} called')
    return {}


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


# =============================================================================
# API TESTING FIXTURES
# =============================================================================

if API_IMPORTS_AVAILABLE:
    # Test settings
    test_settings = TestingSettings()

    @pytest.fixture(scope="session")
    def test_database_url():
        """Test database URL"""
        return "sqlite+aiosqlite:///:memory:"

    @pytest.fixture(scope="session")
    async def test_engine(test_database_url):
        """Create test database engine"""
        engine = create_async_engine(
            test_database_url,
            poolclass=StaticPool,
            connect_args={"check_same_thread": False},
            echo=False
        )
        
        # Create all tables
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        yield engine
        
        # Cleanup
        await engine.dispose()

    @pytest.fixture
    async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
        """Create test database session"""
        async_session = async_sessionmaker(
            test_engine, class_=AsyncSession, expire_on_commit=False
        )
        
        async with async_session() as session:
            yield session
            await session.rollback()

    @pytest.fixture
    def client() -> Generator[TestClient, None, None]:
        """Create test client with mocked dependencies"""
        
        # Mock database dependency
        async def mock_get_database():
            mock_session = AsyncMock(spec=AsyncSession)
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()
            yield mock_session
        
        app.dependency_overrides[get_database] = mock_get_database
        
        # Mock other services
        with patch('api.services.processing_service.ProcessingService') as mock_processing, \
             patch('api.services.monitoring_service.MonitoringService') as mock_monitoring, \
             patch('api.services.webhook_service.WebhookService') as mock_webhook, \
             patch('api.auth.auth_manager.AuthManager') as mock_auth:
            
            # Setup service mocks
            app.state.processing_service = mock_processing.return_value
            app.state.monitoring_service = mock_monitoring.return_value
            app.state.webhook_service = mock_webhook.return_value
            app.state.auth_manager = mock_auth.return_value
            
            with TestClient(app) as test_client:
                yield test_client
        
        # Clean up overrides
        app.dependency_overrides.clear()

    @pytest.fixture
    async def async_client() -> AsyncGenerator[AsyncClient, None]:
        """Create async test client"""
        
        # Mock database dependency
        async def mock_get_database():
            mock_session = AsyncMock(spec=AsyncSession)
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()
            yield mock_session
        
        app.dependency_overrides[get_database] = mock_get_database
        
        # Mock services
        with patch('api.services.processing_service.ProcessingService') as mock_processing, \
             patch('api.services.monitoring_service.MonitoringService') as mock_monitoring, \
             patch('api.services.webhook_service.WebhookService') as mock_webhook, \
             patch('api.auth.auth_manager.AuthManager') as mock_auth:
            
            app.state.processing_service = mock_processing.return_value
            app.state.monitoring_service = mock_monitoring.return_value
            app.state.webhook_service = mock_webhook.return_value
            app.state.auth_manager = mock_auth.return_value
            
            async with AsyncClient(app=app, base_url="http://test") as async_test_client:
                yield async_test_client
        
        app.dependency_overrides.clear()

    @pytest.fixture
    def auth_headers() -> Dict[str, str]:
        """Mock authentication headers"""
        return {
            "Authorization": "Bearer test_token_12345",
            "Content-Type": "application/json"
        }

    @pytest.fixture
    def test_user():
        """Test user data"""
        return User(
            id="user_123",
            username="testuser",
            email="test@example.com",
            password_hash="hashed_password",
            organization_id="org_123",
            full_name="Test User",
            is_active=True,
            is_verified=True,
            roles=["user"],
            permissions=["documents.process", "documents.read"]
        )

    @pytest.fixture
    def test_organization():
        """Test organization data"""
        return Organization(
            id="org_123",
            name="Test Organization",
            display_name="Test Org",
            subscription_tier="professional",
            monthly_document_limit=10000,
            monthly_cost_limit=500.00,
            is_active=True
        )

    @pytest.fixture(scope="session", autouse=True)
    def setup_test_environment():
        """Setup test environment variables"""
        os.environ["ENVIRONMENT"] = "testing"
        os.environ["SKIP_CONFIG_VALIDATION"] = "true"
        os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
        
        yield
        
        # Cleanup
        for key in ["ENVIRONMENT", "SKIP_CONFIG_VALIDATION", "DATABASE_URL"]:
            os.environ.pop(key, None)