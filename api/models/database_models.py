"""
Database Models for Enterprise Document Processing System
SQLAlchemy models with multi-tenant architecture and comprehensive relationships
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Any, Optional

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Text, JSON,
    ForeignKey, UniqueConstraint, Index, CheckConstraint, DECIMAL
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.sql import func

# Base class for all models
Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)


class UUIDMixin:
    """Mixin for UUID primary key"""
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, nullable=False)


# =============================================================================
# ORGANIZATION AND USER MANAGEMENT
# =============================================================================

class Organization(Base, UUIDMixin, TimestampMixin):
    """
    Multi-tenant organization model
    
    Each organization has isolated data and billing
    """
    __tablename__ = "organizations"
    
    name = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)
    domain = Column(String(255), nullable=True, unique=True)
    
    # Contact information
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    
    # Address information
    address_line1 = Column(String(255), nullable=True)
    address_line2 = Column(String(255), nullable=True)
    city = Column(String(100), nullable=True)
    state = Column(String(100), nullable=True)
    postal_code = Column(String(20), nullable=True)
    country = Column(String(100), nullable=True)
    
    # Subscription and billing
    subscription_tier = Column(String(50), nullable=False, default="starter")
    billing_email = Column(String(255), nullable=True)
    monthly_document_limit = Column(Integer, nullable=False, default=1000)
    monthly_cost_limit = Column(DECIMAL(10, 2), nullable=False, default=Decimal("100.00"))
    
    # Status and metadata
    is_active = Column(Boolean, nullable=False, default=True)
    extra_metadata = Column(JSON, nullable=True)
    
    # Relationships
    users = relationship("User", back_populates="organization", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="organization", cascade="all, delete-orphan")
    integrations = relationship("Integration", back_populates="organization", cascade="all, delete-orphan")
    webhooks = relationship("Webhook", back_populates="organization", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_organization_domain", "domain"),
        Index("idx_organization_active", "is_active"),
    )
    
    def __repr__(self):
        return f"<Organization(id='{self.id}', name='{self.name}')>"


class User(Base, UUIDMixin, TimestampMixin):
    """
    User model with role-based access control
    """
    __tablename__ = "users"
    
    # Basic user information
    username = Column(String(100), nullable=False, unique=True)
    email = Column(String(255), nullable=False, unique=True)
    password_hash = Column(String(255), nullable=False)
    
    # Profile information
    full_name = Column(String(255), nullable=True)
    job_title = Column(String(100), nullable=True)
    department = Column(String(100), nullable=True)
    phone = Column(String(50), nullable=True)
    
    # Organization relationship
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Access control
    roles = Column(ARRAY(String), nullable=False, default=list)
    permissions = Column(ARRAY(String), nullable=False, default=list)
    
    # Account status
    is_active = Column(Boolean, nullable=False, default=True)
    is_verified = Column(Boolean, nullable=False, default=False)
    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    
    # Preferences and settings
    preferences = Column(JSON, nullable=True)
    timezone = Column(String(50), nullable=False, default="UTC")
    locale = Column(String(10), nullable=False, default="en-US")
    
    # Relationships
    organization = relationship("Organization", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    user_sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="created_by")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_email", "email"),
        Index("idx_user_username", "username"),
        Index("idx_user_organization", "organization_id"),
        Index("idx_user_active", "is_active"),
    )
    
    @validates('email')
    def validate_email(self, key, email):
        """Basic email validation"""
        if '@' not in email:
            raise ValueError("Invalid email address")
        return email.lower()
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role"""
        return role in (self.roles or [])
    
    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission"""
        return permission in (self.permissions or [])
    
    def __repr__(self):
        return f"<User(id='{self.id}', username='{self.username}')>"


class APIKey(Base, UUIDMixin, TimestampMixin):
    """
    API key model for programmatic access
    """
    __tablename__ = "api_keys"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String(255), nullable=False)
    key_hash = Column(String(255), nullable=False, unique=True)
    
    # Access control
    permissions = Column(ARRAY(String), nullable=False, default=list)
    rate_limit = Column(Integer, nullable=True)  # Requests per minute
    
    # Usage tracking
    usage_count = Column(Integer, nullable=False, default=0)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    last_used_ip = Column(String(45), nullable=True)
    
    # Status and expiration
    is_active = Column(Boolean, nullable=False, default=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index("idx_api_key_hash", "key_hash"),
        Index("idx_api_key_user", "user_id"),
        Index("idx_api_key_active", "is_active"),
    )
    
    def __repr__(self):
        return f"<APIKey(id='{self.id}', name='{self.name}')>"


class UserSession(Base, UUIDMixin, TimestampMixin):
    """
    User session tracking for security and analytics
    """
    __tablename__ = "user_sessions"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    token_hash = Column(String(255), nullable=False, unique=True)
    
    # Session metadata
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    device_type = Column(String(50), nullable=True)
    browser = Column(String(100), nullable=True)
    os = Column(String(100), nullable=True)
    
    # Session status
    is_active = Column(Boolean, nullable=False, default=True)
    last_activity = Column(DateTime(timezone=True), nullable=False, default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="user_sessions")
    
    # Indexes
    __table_args__ = (
        Index("idx_user_session_token", "token_hash"),
        Index("idx_user_session_user", "user_id"),
        Index("idx_user_session_active", "is_active"),
    )


# =============================================================================
# DOCUMENT PROCESSING MODELS
# =============================================================================

class Document(Base, UUIDMixin, TimestampMixin):
    """
    Core document model with processing results
    """
    __tablename__ = "documents"
    
    # Organization and user
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Document metadata
    filename = Column(String(255), nullable=True)
    original_filename = Column(String(255), nullable=True)
    content_type = Column(String(100), nullable=True)
    file_size_bytes = Column(Integer, nullable=True)
    file_hash = Column(String(255), nullable=True)  # SHA-256 hash for deduplication
    
    # Storage information
    storage_path = Column(String(500), nullable=True)
    storage_bucket = Column(String(255), nullable=True)
    
    # Document classification
    document_type = Column(String(50), nullable=True)
    classification_confidence = Column(Float, nullable=True)
    classification_method = Column(String(100), nullable=True)
    
    # Processing results
    processing_status = Column(String(50), nullable=False, default="pending")
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_duration_ms = Column(Integer, nullable=True)
    
    # Extracted data
    extracted_data = Column(JSON, nullable=True)
    confidence_score = Column(Float, nullable=True)
    validation_errors = Column(JSON, nullable=True)
    
    # Competitive processing results
    competitive_results = Column(JSON, nullable=True)
    processing_method = Column(String(100), nullable=True)
    
    # Cost tracking
    processing_cost = Column(DECIMAL(10, 4), nullable=True)
    cost_breakdown = Column(JSON, nullable=True)
    
    # Integration results
    integration_status = Column(String(50), nullable=True)
    integration_results = Column(JSON, nullable=True)
    external_references = Column(JSON, nullable=True)  # References in external systems
    
    # Metadata and tags
    tags = Column(ARRAY(String), nullable=True, default=list)
    custom_fields = Column(JSON, nullable=True)
    source = Column(String(100), nullable=True)  # upload, email, api, etc.
    
    # Relationships
    organization = relationship("Organization", back_populates="documents")
    created_by = relationship("User", back_populates="documents")
    processing_logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_document_organization", "organization_id"),
        Index("idx_document_user", "created_by_id"),
        Index("idx_document_status", "processing_status"),
        Index("idx_document_type", "document_type"),
        Index("idx_document_created", "created_at"),
        Index("idx_document_hash", "file_hash"),
        CheckConstraint("confidence_score >= 0 AND confidence_score <= 1", name="check_confidence_score"),
        CheckConstraint("processing_cost >= 0", name="check_processing_cost_positive"),
    )
    
    def __repr__(self):
        return f"<Document(id='{self.id}', filename='{self.filename}')>"


class ProcessingLog(Base, UUIDMixin, TimestampMixin):
    """
    Detailed processing logs for debugging and analytics
    """
    __tablename__ = "processing_logs"
    
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    
    # Log details
    stage = Column(String(100), nullable=False)  # classification, extraction, validation, etc.
    level = Column(String(20), nullable=False, default="INFO")  # DEBUG, INFO, WARN, ERROR
    message = Column(Text, nullable=False)
    
    # Processing metadata
    processor_name = Column(String(100), nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    memory_usage_mb = Column(Integer, nullable=True)
    
    # Additional context
    extra_metadata = Column(JSON, nullable=True)
    stack_trace = Column(Text, nullable=True)  # For errors
    
    # Relationships
    document = relationship("Document", back_populates="processing_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_processing_log_document", "document_id"),
        Index("idx_processing_log_level", "level"),
        Index("idx_processing_log_stage", "stage"),
        Index("idx_processing_log_created", "created_at"),
    )


class BatchProcessing(Base, UUIDMixin, TimestampMixin):
    """
    Batch processing job tracking
    """
    __tablename__ = "batch_processing"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Batch metadata
    batch_name = Column(String(255), nullable=True)
    total_documents = Column(Integer, nullable=False)
    processed_documents = Column(Integer, nullable=False, default=0)
    failed_documents = Column(Integer, nullable=False, default=0)
    
    # Status and timing
    status = Column(String(50), nullable=False, default="pending")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    estimated_completion_at = Column(DateTime(timezone=True), nullable=True)
    
    # Configuration
    processing_config = Column(JSON, nullable=True)
    webhook_url = Column(String(500), nullable=True)
    
    # Results aggregation
    total_cost = Column(DECIMAL(10, 4), nullable=True, default=Decimal("0.0000"))
    average_confidence = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)
    
    # Document references
    document_ids = Column(ARRAY(UUID), nullable=False, default=list)
    
    # Indexes
    __table_args__ = (
        Index("idx_batch_processing_organization", "organization_id"),
        Index("idx_batch_processing_user", "created_by_id"),
        Index("idx_batch_processing_status", "status"),
        Index("idx_batch_processing_created", "created_at"),
    )


# =============================================================================
# INTEGRATION MODELS
# =============================================================================

class Integration(Base, UUIDMixin, TimestampMixin):
    """
    Enterprise system integration configurations
    """
    __tablename__ = "integrations"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    
    # Integration details
    integration_type = Column(String(50), nullable=False)  # quickbooks, sap, netsuite, xero
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    
    # Connection configuration
    endpoint_url = Column(String(500), nullable=True)
    api_version = Column(String(20), nullable=True)
    
    # Credentials (encrypted)
    credentials = Column(JSON, nullable=True)  # Encrypted credential storage
    oauth_tokens = Column(JSON, nullable=True)  # OAuth token storage
    
    # Mapping and configuration
    field_mappings = Column(JSON, nullable=True)
    processing_rules = Column(JSON, nullable=True)
    auto_post_enabled = Column(Boolean, nullable=False, default=False)
    
    # Status and health
    is_active = Column(Boolean, nullable=False, default=True)
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    last_error = Column(Text, nullable=True)
    health_status = Column(String(50), nullable=False, default="unknown")
    
    # Usage statistics
    total_documents_posted = Column(Integer, nullable=False, default=0)
    successful_posts = Column(Integer, nullable=False, default=0)
    failed_posts = Column(Integer, nullable=False, default=0)
    
    # Relationships
    organization = relationship("Organization", back_populates="integrations")
    
    # Indexes
    __table_args__ = (
        Index("idx_integration_organization", "organization_id"),
        Index("idx_integration_type", "integration_type"),
        Index("idx_integration_active", "is_active"),
        UniqueConstraint("organization_id", "integration_type", "name", name="uq_org_integration_name"),
    )


class Webhook(Base, UUIDMixin, TimestampMixin):
    """
    Webhook configuration for async notifications
    """
    __tablename__ = "webhooks"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=False)
    created_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    
    # Webhook configuration
    name = Column(String(255), nullable=False)
    url = Column(String(500), nullable=False)
    secret = Column(String(255), nullable=True)  # For HMAC verification
    
    # Event configuration
    events = Column(ARRAY(String), nullable=False, default=list)
    headers = Column(JSON, nullable=True)
    
    # Delivery settings
    timeout_seconds = Column(Integer, nullable=False, default=30)
    retry_attempts = Column(Integer, nullable=False, default=3)
    retry_delay_seconds = Column(Integer, nullable=False, default=60)
    
    # Status
    is_active = Column(Boolean, nullable=False, default=True)
    
    # Statistics
    total_deliveries = Column(Integer, nullable=False, default=0)
    successful_deliveries = Column(Integer, nullable=False, default=0)
    failed_deliveries = Column(Integer, nullable=False, default=0)
    last_delivery_at = Column(DateTime(timezone=True), nullable=True)
    last_success_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    organization = relationship("Organization", back_populates="webhooks")
    webhook_deliveries = relationship("WebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_webhook_organization", "organization_id"),
        Index("idx_webhook_active", "is_active"),
        Index("idx_webhook_events", "events", postgresql_using="gin"),
    )


class WebhookDelivery(Base, UUIDMixin, TimestampMixin):
    """
    Webhook delivery tracking and retry management
    """
    __tablename__ = "webhook_deliveries"
    
    webhook_id = Column(UUID(as_uuid=True), ForeignKey("webhooks.id"), nullable=False)
    
    # Delivery details
    event_type = Column(String(100), nullable=False)
    event_id = Column(String(255), nullable=False)
    payload = Column(JSON, nullable=False)
    
    # Delivery status
    status = Column(String(50), nullable=False, default="pending")  # pending, delivered, failed, cancelled
    attempt_count = Column(Integer, nullable=False, default=0)
    max_attempts = Column(Integer, nullable=False, default=3)
    
    # Response details
    response_status_code = Column(Integer, nullable=True)
    response_body = Column(Text, nullable=True)
    response_headers = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Timing
    scheduled_at = Column(DateTime(timezone=True), nullable=False, default=func.now())
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    duration_ms = Column(Integer, nullable=True)
    
    # Relationships
    webhook = relationship("Webhook", back_populates="webhook_deliveries")
    
    # Indexes
    __table_args__ = (
        Index("idx_webhook_delivery_webhook", "webhook_id"),
        Index("idx_webhook_delivery_status", "status"),
        Index("idx_webhook_delivery_scheduled", "scheduled_at"),
        Index("idx_webhook_delivery_event", "event_type", "event_id"),
    )


# =============================================================================
# AUDIT AND MONITORING MODELS
# =============================================================================

class AuditLog(Base, UUIDMixin, TimestampMixin):
    """
    Comprehensive audit logging for security and compliance
    """
    __tablename__ = "audit_logs"
    
    organization_id = Column(UUID(as_uuid=True), ForeignKey("organizations.id"), nullable=True)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    
    # Event details
    event_type = Column(String(100), nullable=False)  # login, logout, document_process, etc.
    resource_type = Column(String(100), nullable=True)  # document, user, organization, etc.
    resource_id = Column(String(255), nullable=True)
    action = Column(String(100), nullable=False)  # create, read, update, delete, process
    
    # Request details
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    request_id = Column(String(255), nullable=True)
    session_id = Column(String(255), nullable=True)
    
    # Event outcome
    status = Column(String(50), nullable=False)  # success, failure, warning
    error_code = Column(String(100), nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Additional context
    extra_metadata = Column(JSON, nullable=True)
    changes = Column(JSON, nullable=True)  # Before/after values for updates
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index("idx_audit_log_organization", "organization_id"),
        Index("idx_audit_log_user", "user_id"),
        Index("idx_audit_log_event_type", "event_type"),
        Index("idx_audit_log_resource", "resource_type", "resource_id"),
        Index("idx_audit_log_created", "created_at"),
        Index("idx_audit_log_status", "status"),
    )


class SystemMetrics(Base, UUIDMixin):
    """
    System performance and usage metrics
    """
    __tablename__ = "system_metrics"
    
    # Timestamp (using custom field instead of mixin for specific indexing)
    timestamp = Column(DateTime(timezone=True), nullable=False, default=func.now())
    metric_date = Column(DateTime(timezone=True), nullable=False, default=func.now())
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    
    # Metric values
    value = Column(Float, nullable=False)
    count = Column(Integer, nullable=True)
    min_value = Column(Float, nullable=True)
    max_value = Column(Float, nullable=True)
    avg_value = Column(Float, nullable=True)
    
    # Dimensions
    organization_id = Column(UUID(as_uuid=True), nullable=True)
    user_id = Column(UUID(as_uuid=True), nullable=True)
    document_type = Column(String(50), nullable=True)
    processing_method = Column(String(100), nullable=True)
    
    # Additional tags
    tags = Column(JSON, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index("idx_system_metrics_timestamp", "timestamp"),
        Index("idx_system_metrics_date", "metric_date"),
        Index("idx_system_metrics_name", "metric_name"),
        Index("idx_system_metrics_org_date", "organization_id", "metric_date"),
        Index("idx_system_metrics_name_date", "metric_name", "metric_date"),
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)


def drop_all_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(bind=engine)


# Export all models for easy importing
__all__ = [
    # Base classes
    "Base", "TimestampMixin", "UUIDMixin",
    
    # Organization and User models
    "Organization", "User", "APIKey", "UserSession",
    
    # Document processing models
    "Document", "ProcessingLog", "BatchProcessing",
    
    # Integration models
    "Integration", "Webhook", "WebhookDelivery",
    
    # Audit and monitoring models
    "AuditLog", "SystemMetrics",
    
    # Utilities
    "create_all_tables", "drop_all_tables"
]