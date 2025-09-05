"""
Database Session Management
Async SQLAlchemy session factory with connection pooling and health checks
"""

import asyncio
from pathlib import Path
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession, 
    AsyncEngine, 
    create_async_engine, 
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DisconnectionError, OperationalError

from api.config import get_settings
from api.models.database_models import Base
from utils.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


async def create_database_engine() -> AsyncEngine:
    """
    Create async database engine with optimized configuration
    
    Features:
    - Connection pooling with health checks
    - Automatic reconnection on failures
    - Query logging and performance monitoring
    - Connection lifecycle events
    """
    database_url = settings.get_database_url(async_driver=True)
    
    # Engine configuration
    engine_kwargs = {
        "url": database_url,
        "echo": settings.database.database_echo_sql,
    }
    
    # SQLite configuration (uses file-based database)
    if "sqlite" in database_url.lower():
        engine_kwargs.update({
            "poolclass": NullPool,  # SQLite doesn't support connection pooling
            "connect_args": {"check_same_thread": False}  # Allow SQLite to be used across threads
        })
    else:
        # PostgreSQL configuration
        engine_kwargs.update({
            "pool_size": settings.database.database_pool_size,
            "max_overflow": settings.database.database_pool_max_overflow,
            "pool_timeout": settings.database.database_pool_timeout,
            "pool_recycle": 3600,  # Recycle connections every hour
            "pool_pre_ping": True,  # Enable connection health checks
            "poolclass": QueuePool
        })
    
    # Create engine
    engine = create_async_engine(**engine_kwargs)
    
    # Setup connection event listeners
    setup_connection_events(engine.sync_engine)
    
    logger.info(f"Created database engine: {database_url}")
    return engine


def setup_connection_events(engine: Engine) -> None:
    """Setup SQLAlchemy event listeners for connection monitoring"""
    
    @event.listens_for(engine, "connect")
    def receive_connect(dbapi_connection, connection_record):
        """Log new database connections"""
        logger.debug("New database connection established")
    
    @event.listens_for(engine, "checkout")
    def receive_checkout(dbapi_connection, connection_record, connection_proxy):
        """Log connection checkout from pool"""
        logger.debug("Database connection checked out from pool")
    
    @event.listens_for(engine, "checkin")
    def receive_checkin(dbapi_connection, connection_record):
        """Log connection checkin to pool"""
        logger.debug("Database connection returned to pool")
    
    @event.listens_for(engine, "invalidate")
    def receive_invalidate(dbapi_connection, connection_record, exception):
        """Log connection invalidation"""
        logger.warning(f"Database connection invalidated: {exception}")
    
    @event.listens_for(engine, "soft_invalidate")
    def receive_soft_invalidate(dbapi_connection, connection_record, exception):
        """Log soft connection invalidation"""
        logger.warning(f"Database connection soft invalidated: {exception}")


async def init_database() -> None:
    """
    Initialize database engine and create tables
    
    This should be called once during application startup
    """
    global _engine, _session_factory
    
    try:
        # Create engine
        _engine = await create_database_engine()
        
        # Create session factory
        _session_factory = async_sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=True,
            autocommit=False
        )
        
        # Create tables if they don't exist
        if not settings.is_testing:  # Skip in testing - migrations handle this
            async with _engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        
        # Test connection
        await test_database_connection()
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


async def close_database() -> None:
    """
    Close database connections and cleanup resources
    
    This should be called during application shutdown
    """
    global _engine, _session_factory
    
    try:
        if _engine:
            await _engine.dispose()
            _engine = None
            
        _session_factory = None
        
        logger.info("Database connections closed successfully")
        
    except Exception as e:
        logger.error(f"Database cleanup failed: {e}")


async def get_database() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency to get database session
    
    Provides automatic session management with proper cleanup:
    - Creates new session for each request
    - Handles transactions and rollbacks
    - Ensures proper cleanup on completion
    
    Usage:
        @app.get(str(Path("/api/endpoint").resolve()))
        async def endpoint(db: AsyncSession = Depends(get_database)):
            # Use db for database operations
            logger.info(f'Method {function_name} called')
            return {}
    """
    if not _session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        await session.rollback()
        raise
    finally:
        await session.close()


@asynccontextmanager
async def get_database_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions outside FastAPI dependencies
    
    Usage:
        async with get_database_session() as db:
            # Use db for database operations
    return {}
    """
    if not _session_factory:
        raise RuntimeError("Database not initialized. Call init_database() first.")
    
    session = _session_factory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        logger.error(f"Database session error: {e}")
        await session.rollback()
        raise
    finally:
        await session.close()


async def test_database_connection() -> bool:
    """
    Test database connectivity and basic operations
    
    Returns True if database is healthy, False otherwise
    """
    try:
        if not _engine:
            return False
        
        async with _engine.begin() as conn:
            # Test basic query
            result = await conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            
            if row and row.test == 1:
                logger.info("Database connection test successful")
                return True
            else:
                logger.error("Database connection test failed: unexpected result")
                return False
                
    except (DisconnectionError, OperationalError) as e:
        logger.error(f"Database connection test failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Database connection test error: {e}")
        return False


async def get_database_health() -> dict:
    """
    Get comprehensive database health information
    
    Returns dictionary with connection pool status and performance metrics
    """
    try:
        if not _engine:
            return {
                "status": "disconnected",
                "error": "Database engine not initialized"
            }
        
        # Get pool status
        pool = _engine.pool
        pool_status = {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }
        
        # Test connection
        connection_healthy = await test_database_connection()
        
        # Calculate pool utilization
        total_connections = pool_status["checked_in"] + pool_status["checked_out"]
        max_connections = settings.database.database_pool_size + settings.database.database_pool_max_overflow
        utilization = (total_connections / max_connections) * 100 if max_connections > 0 else 0
        
        return {
            "status": "healthy" if connection_healthy else "unhealthy",
            "connection_test": connection_healthy,
            "pool_status": pool_status,
            "pool_utilization_percent": round(utilization, 2),
            "database_url": settings.database.database_url.replace(
                settings.database.database_url.password, "***"
            ) if settings.database.database_url.password else str(settings.database.database_url)
        }
        
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


async def execute_raw_query(query: str, parameters: dict = None) -> list:
    """
    Execute raw SQL query with parameters
    
    Use this for complex queries that are difficult to express with SQLAlchemy ORM
    
    Args:
        query: SQL query string with named parameters
        parameters: Dictionary of parameter values
    
    Returns:
        List of result rows as dictionaries
    """
    if not _engine:
        raise RuntimeError("Database not initialized")
    
    try:
        async with _engine.begin() as conn:
            result = await conn.execute(text(query), parameters or {})
            
            # Convert result to list of dictionaries
            columns = result.keys()
            rows = []
            for row in result.fetchall():
                rows.append(dict(zip(columns, row)))
            
            return rows
            
    except Exception as e:
        logger.error(f"Raw query execution failed: {query[:100]}... Error: {e}")
        raise


class DatabaseManager:
    """
    Higher-level database management class
    
    Provides utilities for:
    - Transaction management
    - Bulk operations
    - Query optimization
    - Connection monitoring
    """
    
    def __init__(self):
        self.engine = _engine
        self.session_factory = _session_factory
    
    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Managed transaction context
        
        Automatically handles commit/rollback and ensures proper cleanup
        
        Usage:
            db_manager = DatabaseManager()
            async with db_manager.transaction() as session:
                # All operations in this block are in a single transaction
                user = User(name="test")
                session.add(user)
                # Automatic commit on success, rollback on exception
        """
        if not self.session_factory:
            raise RuntimeError("Database not initialized")
        
        session = self.session_factory()
        try:
            async with session.begin():
                yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    async def bulk_insert(self, model_class, data: list[dict]) -> int:
        """
        High-performance bulk insert operation
        
        Args:
            model_class: SQLAlchemy model class
            data: List of dictionaries with model data
        
        Returns:
            Number of inserted records
        """
        if not data:
            return 0
        
        try:
            async with self.transaction() as session:
                objects = [model_class(**item) for item in data]
                session.add_all(objects)
                await session.flush()
                return len(objects)
                
        except Exception as e:
            logger.error(f"Bulk insert failed for {model_class.__name__}: {e}")
            raise
    
    async def bulk_update(self, model_class, updates: list[dict]) -> int:
        """
        High-performance bulk update operation
        
        Args:
            model_class: SQLAlchemy model class
            updates: List of dictionaries with id and update data
        
        Returns:
            Number of updated records
        """
        if not updates:
            return 0
        
        try:
            async with self.transaction() as session:
                updated_count = 0
                for update_data in updates:
                    stmt = session.query(model_class).filter(
                        model_class.id == update_data['id']
                    ).update(update_data, synchronize_session=False)
                    updated_count += stmt.rowcount
                
                return updated_count
                
        except Exception as e:
            logger.error(f"Bulk update failed for {model_class.__name__}: {e}")
            raise
    
    async def get_connection_stats(self) -> dict:
        """Get detailed connection pool statistics"""
        if not self.engine:
            return {}
        
        pool = self.engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in_connections": pool.checkedin(),
            "checked_out_connections": pool.checkedout(),
            "overflow_connections": pool.overflow(),
            "invalid_connections": pool.invalid(),
            "total_connections": pool.checkedin() + pool.checkedout(),
            "max_connections": settings.database.database_pool_size + settings.database.database_pool_max_overflow
        }


# Singleton database manager instance
db_manager = DatabaseManager()


# Migration support utilities
async def check_migration_status() -> dict:
    """
    Check database migration status
    
    Returns information about current schema version and pending migrations
    """
    try:
        # This would integrate with Alembic for production use
        async with get_database_session() as session:
            # Check if alembic_version table exists
            result = await session.execute(text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'alembic_version'
                );
            """))
            
            has_alembic = result.scalar()
            
            if has_alembic:
                # Get current version
                version_result = await session.execute(text("SELECT version_num FROM alembic_version"))
                current_version = version_result.scalar()
                
                return {
                    "has_migrations": True,
                    "current_version": current_version,
                    "status": "managed"
                }
            else:
                return {
                    "has_migrations": False,
                    "current_version": None,
                    "status": "unmanaged"
                }
                
    except Exception as e:
        logger.error(f"Migration status check failed: {e}")
        return {
            "has_migrations": False,
            "current_version": None,
            "status": "error",
            "error": str(e)
        }


# Export key components
__all__ = [
    "init_database",
    "close_database", 
    "get_database",
    "get_database_session",
    "test_database_connection",
    "get_database_health",
    "execute_raw_query",
    "DatabaseManager",
    "db_manager",
    "check_migration_status"
]