"""
Enterprise API Configuration System
Environment-based configuration with validation and type safety
"""

import os
from functools import lru_cache
from typing import List, Optional, Dict, Any
from pathlib import Path

from pydantic import Field, validator, PostgresDsn, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    """Database configuration"""
    database_url: PostgresDsn = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/enterprise_docs",
        description="Async PostgreSQL database URL"
    )
    database_pool_size: int = Field(default=20, ge=1, le=100)
    database_pool_max_overflow: int = Field(default=30, ge=0, le=100)
    database_pool_timeout: int = Field(default=30, ge=1, le=300)
    database_echo_sql: bool = Field(default=False, description="Log SQL queries")
    
    model_config = SettingsConfigDict(env_prefix="DATABASE_")


class RedisSettings(BaseSettings):
    """Redis configuration for caching and queuing"""
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    redis_password: Optional[str] = None
    redis_max_connections: int = Field(default=50, ge=1, le=200)
    redis_socket_timeout: int = Field(default=30, ge=1, le=300)
    redis_socket_connect_timeout: int = Field(default=30, ge=1, le=300)
    
    # Cache settings
    cache_default_ttl: int = Field(default=3600, ge=60, le=86400)  # 1 hour default
    cache_max_entries: int = Field(default=10000, ge=100, le=100000)
    
    model_config = SettingsConfigDict(env_prefix="REDIS_")


class SecuritySettings(BaseSettings):
    """Security and authentication configuration"""
    secret_key: str = Field(
        default="your-super-secret-key-change-in-production",
        description="Secret key for JWT token signing"
    )
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=5, le=1440)
    refresh_token_expire_days: int = Field(default=7, ge=1, le=30)
    
    # Password hashing
    password_hash_schemes: List[str] = Field(default=["bcrypt"], description="Password hashing schemes")
    password_bcrypt_rounds: int = Field(default=12, ge=4, le=16)
    
    # API Keys
    api_key_length: int = Field(default=32, ge=16, le=64)
    api_key_prefix: str = Field(default="eapi_", description="API key prefix")
    
    # Rate limiting
    rate_limit_per_minute: int = Field(default=100, ge=1, le=10000)
    rate_limit_burst: int = Field(default=200, ge=1, le=20000)
    
    # CORS
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins"
    )
    cors_credentials: bool = Field(default=True)
    cors_methods: List[str] = Field(default=["*"])
    cors_headers: List[str] = Field(default=["*"])
    
    # Trusted hosts
    allowed_hosts: List[str] = Field(
        default=["localhost", "127.0.0.1", "*.yourdomain.com"],
        description="Allowed host headers"
    )
    
    @validator('secret_key')
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError('Secret key must be at least 32 characters long')
        return v
    
    model_config = SettingsConfigDict(env_prefix="SECURITY_")


class ProcessingSettings(BaseSettings):
    """Document processing configuration"""
    max_file_size_mb: int = Field(default=50, ge=1, le=500)
    max_batch_size: int = Field(default=100, ge=1, le=1000)
    max_concurrent_processes: int = Field(default=10, ge=1, le=50)
    
    # Processing timeouts
    default_processing_timeout: int = Field(default=30, ge=5, le=300)
    max_processing_timeout: int = Field(default=300, ge=30, le=1800)
    
    # Quality thresholds
    default_confidence_threshold: float = Field(default=0.95, ge=0.5, le=1.0)
    min_confidence_threshold: float = Field(default=0.7, ge=0.3, le=0.95)
    
    # Cost controls
    max_cost_per_document: float = Field(default=0.05, ge=0.001, le=1.0)
    daily_cost_limit: float = Field(default=100.0, ge=1.0, le=10000.0)
    
    # Storage paths
    upload_directory: Path = Field(default=Path("data/uploads"))
    processing_directory: Path = Field(default=Path("data/processing"))
    archive_directory: Path = Field(default=Path("data/archive"))
    
    # Supported file types
    supported_file_types: List[str] = Field(
        default=[".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".txt", ".docx", ".xlsx"],
        description="Supported file extensions"
    )
    
    @validator('upload_directory', 'processing_directory', 'archive_directory')
    def ensure_directories_exist(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    model_config = SettingsConfigDict(env_prefix="PROCESSING_")


class IntegrationSettings(BaseSettings):
    """Enterprise integration configuration"""
    # QuickBooks
    quickbooks_client_id: Optional[str] = None
    quickbooks_client_secret: Optional[str] = None
    quickbooks_sandbox_mode: bool = Field(default=True)
    quickbooks_webhook_token: Optional[str] = None
    
    # SAP
    sap_client_id: Optional[str] = None
    sap_client_secret: Optional[str] = None
    sap_server_url: Optional[HttpUrl] = None
    sap_system_id: Optional[str] = None
    
    # NetSuite
    netsuite_account_id: Optional[str] = None
    netsuite_consumer_key: Optional[str] = None
    netsuite_consumer_secret: Optional[str] = None
    netsuite_token_id: Optional[str] = None
    netsuite_token_secret: Optional[str] = None
    
    # Xero
    xero_client_id: Optional[str] = None
    xero_client_secret: Optional[str] = None
    xero_redirect_uri: Optional[HttpUrl] = None
    
    # Webhook settings
    webhook_timeout: int = Field(default=30, ge=5, le=300)
    webhook_max_retries: int = Field(default=3, ge=0, le=10)
    webhook_retry_delay: int = Field(default=60, ge=1, le=3600)
    
    model_config = SettingsConfigDict(env_prefix="INTEGRATION_")


class MonitoringSettings(BaseSettings):
    """Monitoring and observability configuration"""
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="structured", description="Log format (structured/plain)")
    log_file_path: Optional[Path] = None
    log_rotation_size: str = Field(default="100MB", description="Log file rotation size")
    log_retention_days: int = Field(default=30, ge=1, le=365)
    
    # Metrics
    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    metrics_path: str = Field(default="/metrics")
    
    # Health checks
    health_check_interval: int = Field(default=30, ge=5, le=300)
    health_check_timeout: int = Field(default=10, ge=1, le=60)
    
    # Performance monitoring
    enable_tracing: bool = Field(default=True)
    trace_sampling_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    enable_profiling: bool = Field(default=False)
    
    # Alerting
    enable_alerts: bool = Field(default=True)
    alert_webhook_url: Optional[HttpUrl] = None
    alert_email_recipients: List[str] = Field(default_factory=list)
    
    model_config = SettingsConfigDict(env_prefix="MONITORING_")


class EmailSettings(BaseSettings):
    """Email configuration for notifications"""
    smtp_server: str = Field(default="smtp.gmail.com")
    smtp_port: int = Field(default=587, ge=1, le=65535)
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = Field(default=True)
    
    # Email templates
    from_email: str = Field(default="noreply@yourdomain.com")
    from_name: str = Field(default="Enterprise Document Processing")
    
    # Rate limiting
    email_rate_limit_per_hour: int = Field(default=100, ge=1, le=1000)
    
    model_config = SettingsConfigDict(env_prefix="EMAIL_")


class CelerySettings(BaseSettings):
    """Celery configuration for background tasks"""
    celery_broker_url: str = Field(
        default="redis://localhost:6379/1",
        description="Celery message broker URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL"
    )
    
    # Task settings
    task_serializer: str = Field(default="json")
    result_serializer: str = Field(default="json")
    accept_content: List[str] = Field(default=["json"])
    result_expires: int = Field(default=3600, ge=300, le=86400)
    
    # Worker settings
    worker_concurrency: int = Field(default=4, ge=1, le=32)
    worker_max_tasks_per_child: int = Field(default=1000, ge=100, le=10000)
    worker_prefetch_multiplier: int = Field(default=1, ge=1, le=10)
    
    # Queue settings
    default_queue: str = Field(default="default")
    processing_queue: str = Field(default="processing")
    integration_queue: str = Field(default="integrations")
    notification_queue: str = Field(default="notifications")
    
    model_config = SettingsConfigDict(env_prefix="CELERY_")


class Settings(BaseSettings):
    """Main settings class combining all configuration sections"""
    
    # Environment
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    testing: bool = Field(default=False, description="Testing mode")
    
    # Application info
    app_name: str = Field(default="Enterprise Document Processing API")
    app_version: str = Field(default="1.0.0")
    app_description: str = Field(
        default="Production-ready API for multi-domain document processing"
    )
    
    # API settings
    api_v1_prefix: str = Field(default="/api/v1")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)
    api_workers: int = Field(default=4, ge=1, le=32)
    
    # Component settings
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    integrations: IntegrationSettings = Field(default_factory=IntegrationSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    email: EmailSettings = Field(default_factory=EmailSettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed_envs = ["development", "staging", "production", "testing"]
        if v not in allowed_envs:
            raise ValueError(f"Environment must be one of: {allowed_envs}")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment"""
        return self.testing or self.environment == "testing"
    
    @property
    def cors_origins(self) -> List[str]:
        """Get CORS origins based on environment"""
        if self.is_production:
            return ["https://yourdomain.com", "https://app.yourdomain.com"]
        return self.security.cors_origins
    
    @property
    def allowed_hosts(self) -> List[str]:
        """Get allowed hosts based on environment"""
        if self.is_production:
            return ["yourdomain.com", "app.yourdomain.com", "api.yourdomain.com"]
        return self.security.allowed_hosts
    
    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL with appropriate driver"""
        url = str(self.database.database_url)
        if not async_driver:
            url = url.replace("postgresql+asyncpg://", "postgresql://")
        return url
    
    def get_redis_config(self) -> Dict[str, Any]:
        """Get Redis configuration dictionary"""
        return {
            "url": self.redis.redis_url,
            "password": self.redis.redis_password,
            "max_connections": self.redis.redis_max_connections,
            "socket_timeout": self.redis.redis_socket_timeout,
            "socket_connect_timeout": self.redis.redis_socket_connect_timeout,
        }
    
    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration dictionary"""
        return {
            "broker_url": self.celery.celery_broker_url,
            "result_backend": self.celery.celery_result_backend,
            "task_serializer": self.celery.task_serializer,
            "result_serializer": self.celery.result_serializer,
            "accept_content": self.celery.accept_content,
            "result_expires": self.celery.result_expires,
            "worker_concurrency": self.celery.worker_concurrency,
            "worker_max_tasks_per_child": self.celery.worker_max_tasks_per_child,
            "worker_prefetch_multiplier": self.celery.worker_prefetch_multiplier,
        }
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )


# Configuration factory with caching
@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    
    Settings are cached to avoid re-reading configuration on every request.
    Use `get_settings.cache_clear()` to force refresh.
    """
    return Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings"""
    environment: str = "development"
    debug: bool = True
    
    model_config = SettingsConfigDict(env_file=".env.development")


class ProductionSettings(Settings):
    """Production environment settings"""
    environment: str = "production"
    debug: bool = False
    
    model_config = SettingsConfigDict(env_file=".env.production")


class TestingSettings(Settings):
    """Testing environment settings"""
    environment: str = "testing"
    testing: bool = True
    
    # Override for testing
    database: DatabaseSettings = Field(default_factory=lambda: DatabaseSettings(
        database_url="postgresql+asyncpg://test:test@localhost:5432/test_enterprise_docs"
    ))
    
    model_config = SettingsConfigDict(env_file=".env.testing")


# Factory function for environment-specific settings
def get_settings_for_environment(env: str) -> Settings:
    """Get settings instance for specific environment"""
    if env == "development":
        return DevelopmentSettings()
    elif env == "production":
        return ProductionSettings()
    elif env == "testing":
        return TestingSettings()
    else:
        return Settings(environment=env)


# Validate critical configuration on import
def validate_critical_config():
    """Validate critical configuration that could prevent startup"""
    try:
        settings = get_settings()
        
        # Validate database URL
        if not settings.database.database_url:
            raise ValueError("Database URL is required")
        
        # Validate secret key in production
        if settings.is_production and settings.security.secret_key == "your-super-secret-key-change-in-production":
            raise ValueError("Production secret key must be changed from default")
        
        # Validate required directories exist
        for directory in [settings.processing.upload_directory, 
                         settings.processing.processing_directory,
                         settings.processing.archive_directory]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
        
    except Exception as e:
        import logging
        logging.error(f"Configuration validation failed: {e}")
        raise


# Validate on import (optional - can be disabled for testing)
if os.getenv("SKIP_CONFIG_VALIDATION") != "true":
    validate_critical_config()