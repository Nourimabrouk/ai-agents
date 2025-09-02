"""
API Models for Enterprise Document Processing System
Pydantic models for request/response validation, serialization, and documentation
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal

from pydantic import BaseModel, Field, validator, HttpUrl
from pydantic.types import FilePath, PositiveInt


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class DocumentType(str, Enum):
    """Supported document types"""
    INVOICE = "invoice"
    PURCHASE_ORDER = "purchase_order"
    RECEIPT = "receipt"
    BANK_STATEMENT = "bank_statement"
    CONTRACT = "contract"
    EXPENSE_REPORT = "expense_report"
    TAX_DOCUMENT = "tax_document"
    INSURANCE_CLAIM = "insurance_claim"
    UNKNOWN = "unknown"


class ProcessingStrategy(str, Enum):
    """Processing strategies"""
    SPEED_OPTIMIZED = "speed_optimized"
    ACCURACY_OPTIMIZED = "accuracy_optimized"
    COMPETITIVE = "competitive"
    CONSENSUS = "consensus"
    COST_OPTIMIZED = "cost_optimized"


class ProcessingStatus(str, Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IntegrationType(str, Enum):
    """Enterprise integration types"""
    QUICKBOOKS = "quickbooks"
    SAP = "sap"
    NETSUITE = "netsuite"
    XERO = "xero"
    CUSTOM = "custom"


class WebhookEvent(str, Enum):
    """Webhook event types"""
    DOCUMENT_PROCESSED = "document.processed"
    DOCUMENT_FAILED = "document.failed"
    BATCH_COMPLETED = "batch.completed"
    THRESHOLD_EXCEEDED = "threshold.exceeded"
    INTEGRATION_ERROR = "integration.error"


# =============================================================================
# BASE MODELS
# =============================================================================

class BaseResponse(BaseModel):
    """Base response model with common fields"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request correlation ID")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None,
            Decimal: lambda v: float(v) if v else None
        }


class ErrorResponse(BaseResponse):
    """Standard error response"""
    error_code: str = Field(..., description="Machine-readable error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    suggestions: Optional[List[str]] = Field(None, description="Suggested solutions")


class SuccessResponse(BaseResponse):
    """Standard success response"""
    success: bool = Field(True)
    message: Optional[str] = Field(None, description="Success message")
    data: Optional[Dict[str, Any]] = Field(None, description="Response data")


# =============================================================================
# PROCESSING MODELS
# =============================================================================

class ProcessingConfig(BaseModel):
    """Processing configuration options"""
    strategy: ProcessingStrategy = ProcessingStrategy.COMPETITIVE
    accuracy_threshold: float = Field(0.95, ge=0.0, le=1.0)
    max_processing_time_seconds: int = Field(30, ge=1, le=300)
    max_cost_per_document: Decimal = Field(Decimal("0.05"), ge=0)
    enable_parallel_processing: bool = True
    enable_validation: bool = True
    save_intermediate_results: bool = False
    
    @validator('accuracy_threshold')
    def validate_accuracy_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('Accuracy threshold must be between 0.0 and 1.0')
        return v


class DocumentMetadata(BaseModel):
    """Document metadata information"""
    filename: Optional[str] = None
    file_size_bytes: Optional[int] = None
    content_type: Optional[str] = None
    upload_timestamp: Optional[datetime] = None
    source: Optional[str] = Field(None, description="Document source (upload, email, API, etc.)")
    tags: Optional[List[str]] = Field(default_factory=list)
    custom_fields: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ProcessingRequest(BaseModel):
    """Request for document processing"""
    # Input options (one required)
    text_content: Optional[str] = Field(None, description="Raw text content to process")
    file_path: Optional[str] = Field(None, description="Path to file for processing")
    file_url: Optional[HttpUrl] = Field(None, description="URL to download file")
    base64_content: Optional[str] = Field(None, description="Base64 encoded file content")
    
    # Processing options
    processing_config: Optional[ProcessingConfig] = Field(default_factory=ProcessingConfig)
    document_type_hint: Optional[DocumentType] = Field(None, description="Optional document type hint")
    
    # Integration options
    auto_post_to_accounting: bool = Field(False, description="Automatically post to accounting system")
    accounting_integration: Optional[IntegrationType] = None
    
    # Notification options
    webhook_url: Optional[HttpUrl] = Field(None, description="Webhook URL for completion notification")
    email_notification: Optional[str] = Field(None, description="Email for completion notification")
    
    # Metadata
    metadata: Optional[DocumentMetadata] = Field(default_factory=DocumentMetadata)
    
    @validator('text_content', 'file_path', 'file_url', 'base64_content', pre=True, always=True)
    def validate_input_source(cls, v, values):
        # At least one input method must be provided
        input_fields = ['text_content', 'file_path', 'file_url', 'base64_content']
        provided_inputs = sum(1 for field in input_fields if values.get(field))
        
        if provided_inputs == 0:
            raise ValueError('At least one input method must be provided')
        elif provided_inputs > 1:
            raise ValueError('Only one input method should be provided')
        
        return v


class ClassificationRequest(BaseModel):
    """Request for document classification only"""
    text_content: Optional[str] = None
    file_path: Optional[str] = None
    file_url: Optional[HttpUrl] = None
    base64_content: Optional[str] = None
    metadata: Optional[DocumentMetadata] = Field(default_factory=DocumentMetadata)


class BatchProcessingRequest(BaseModel):
    """Request for batch document processing"""
    documents: List[ProcessingRequest] = Field(..., min_items=1, max_items=100)
    batch_name: Optional[str] = Field(None, description="Optional batch identifier")
    processing_config: Optional[ProcessingConfig] = Field(default_factory=ProcessingConfig)
    max_concurrent_documents: int = Field(5, ge=1, le=10)
    webhook_url: Optional[HttpUrl] = Field(None, description="Webhook for batch completion")
    
    @validator('documents')
    def validate_documents(cls, v):
        if len(v) > 100:
            raise ValueError('Maximum 100 documents per batch')
        return v


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class CompetitiveResult(BaseModel):
    """Result from competitive processing method"""
    processor_name: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int = Field(..., ge=0)
    cost_estimate: Decimal = Field(..., ge=0)
    success: bool
    error_message: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None


class ProcessingResult(BaseModel):
    """Detailed processing result"""
    document_type: DocumentType
    extracted_data: Dict[str, Any]
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_method: str
    validation_errors: List[str] = Field(default_factory=list)
    processing_time_ms: int = Field(..., ge=0)
    cost_breakdown: Dict[str, Decimal]
    competitive_results: List[CompetitiveResult] = Field(default_factory=list)


class ProcessingResponse(BaseResponse):
    """Response from document processing"""
    document_id: str = Field(..., description="Unique document identifier")
    success: bool
    classification: Dict[str, Any] = Field(..., description="Document classification result")
    extracted_data: Dict[str, Any] = Field(..., description="Extracted structured data")
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: int = Field(..., ge=0)
    validation_errors: List[str] = Field(default_factory=list)
    cost_breakdown: Dict[str, Decimal]
    processing_method: str
    competitive_results: List[CompetitiveResult] = Field(default_factory=list)
    
    # Integration results (if enabled)
    accounting_integration_result: Optional[Dict[str, Any]] = None
    
    # File information
    original_filename: Optional[str] = None
    file_size_bytes: Optional[int] = None


class BatchProcessingResponse(BaseResponse):
    """Response from batch processing request"""
    batch_id: str
    status: ProcessingStatus
    total_documents: int
    completed_documents: int = 0
    failed_documents: int = 0
    estimated_completion_time: Optional[datetime] = None
    results: List[ProcessingResponse] = Field(default_factory=list)


class ProcessingStatusResponse(BaseResponse):
    """Document processing status response"""
    document_id: str
    status: ProcessingStatus
    progress_percentage: float = Field(..., ge=0.0, le=100.0)
    current_stage: str
    estimated_completion_time: Optional[datetime] = None
    error_message: Optional[str] = None
    partial_results: Optional[Dict[str, Any]] = None


# =============================================================================
# INTEGRATION MODELS
# =============================================================================

class IntegrationConfig(BaseModel):
    """Enterprise integration configuration"""
    integration_type: IntegrationType
    connection_string: Optional[str] = None
    api_endpoint: Optional[HttpUrl] = None
    credentials: Dict[str, str] = Field(default_factory=dict)
    custom_mapping: Optional[Dict[str, str]] = None
    auto_post_enabled: bool = False
    validation_rules: Optional[Dict[str, Any]] = None


class IntegrationResult(BaseModel):
    """Result from enterprise integration"""
    integration_type: IntegrationType
    success: bool
    transaction_id: Optional[str] = None
    posted_amount: Optional[Decimal] = None
    posting_date: Optional[datetime] = None
    error_message: Optional[str] = None
    external_reference: Optional[str] = None


# =============================================================================
# WEBHOOK MODELS
# =============================================================================

class WebhookConfig(BaseModel):
    """Webhook configuration"""
    url: HttpUrl
    events: List[WebhookEvent]
    secret: Optional[str] = Field(None, description="Secret for HMAC verification")
    headers: Optional[Dict[str, str]] = Field(default_factory=dict)
    retry_attempts: int = Field(3, ge=0, le=10)
    timeout_seconds: int = Field(30, ge=1, le=300)
    active: bool = True


class WebhookPayload(BaseModel):
    """Standard webhook payload"""
    event_type: WebhookEvent
    event_id: str
    timestamp: datetime
    data: Dict[str, Any]
    organization_id: str
    user_id: Optional[str] = None
    
    # HMAC signature will be added in headers
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }


# =============================================================================
# SYSTEM MODELS
# =============================================================================

class HealthResponse(BaseResponse):
    """System health status response"""
    status: str = Field(..., description="Overall system status")
    version: str
    uptime_seconds: int = Field(..., ge=0)
    database_connected: bool
    processing_service_active: bool
    external_services: Dict[str, bool] = Field(default_factory=dict)
    performance_metrics: Dict[str, Union[int, float]] = Field(default_factory=dict)


class MetricsResponse(BaseResponse):
    """Comprehensive system metrics"""
    processing_metrics: Dict[str, Any] = Field(default_factory=dict)
    performance_metrics: Dict[str, Any] = Field(default_factory=dict)
    usage_metrics: Dict[str, Any] = Field(default_factory=dict)
    cost_metrics: Dict[str, Any] = Field(default_factory=dict)
    error_metrics: Dict[str, Any] = Field(default_factory=dict)
    integration_metrics: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# AUTHENTICATION MODELS
# =============================================================================

class UserInfo(BaseModel):
    """User information"""
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None
    organization_id: str
    organization_name: str
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    last_login: Optional[datetime] = None


class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: Optional[str] = None
    user_info: UserInfo


class LoginRequest(BaseModel):
    """Login request"""
    username: Optional[str] = None
    email: Optional[str] = None
    password: str
    api_key: Optional[str] = None
    organization_id: Optional[str] = None
    
    @validator('username', 'email', pre=True, always=True)
    def validate_identifier(cls, v, values):
        if not values.get('api_key') and not values.get('username') and not values.get('email'):
            raise ValueError('Username, email, or API key required')
        return v


# =============================================================================
# ANALYTICS MODELS
# =============================================================================

class ProcessingAnalytics(BaseModel):
    """Processing analytics data"""
    total_documents_processed: int = Field(..., ge=0)
    successful_processing_count: int = Field(..., ge=0)
    failed_processing_count: int = Field(..., ge=0)
    average_confidence_score: float = Field(..., ge=0.0, le=1.0)
    average_processing_time_ms: float = Field(..., ge=0.0)
    total_cost: Decimal = Field(..., ge=0)
    
    # By document type
    document_type_breakdown: Dict[DocumentType, Dict[str, Any]] = Field(default_factory=dict)
    
    # By time period
    daily_volume: List[Dict[str, Any]] = Field(default_factory=list)
    hourly_distribution: Dict[int, int] = Field(default_factory=dict)
    
    # Top errors
    top_errors: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Performance trends
    accuracy_trend: List[float] = Field(default_factory=list)
    cost_trend: List[Decimal] = Field(default_factory=list)
    speed_trend: List[float] = Field(default_factory=list)


class CostAnalysis(BaseModel):
    """Cost analysis and optimization recommendations"""
    total_cost: Decimal = Field(..., ge=0)
    cost_per_document: Decimal = Field(..., ge=0)
    cost_by_document_type: Dict[DocumentType, Decimal] = Field(default_factory=dict)
    cost_by_processing_method: Dict[str, Decimal] = Field(default_factory=dict)
    
    # Budget tracking
    monthly_budget: Optional[Decimal] = None
    budget_used_percentage: Optional[float] = None
    projected_monthly_cost: Optional[Decimal] = None
    
    # Optimization recommendations
    optimization_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    potential_savings: Optional[Decimal] = None


# =============================================================================
# VALIDATION UTILITIES
# =============================================================================

def validate_document_data(document_type: DocumentType, extracted_data: Dict[str, Any]) -> List[str]:
    """Validate extracted data based on document type"""
    errors = []
    
    if document_type == DocumentType.INVOICE:
        required_fields = ['invoice_number', 'total_amount', 'vendor_name', 'invoice_date']
        for field in required_fields:
            if not extracted_data.get(field):
                errors.append(f"Missing required invoice field: {field}")
    
    elif document_type == DocumentType.PURCHASE_ORDER:
        required_fields = ['po_number', 'vendor_name', 'total_amount']
        for field in required_fields:
            if not extracted_data.get(field):
                errors.append(f"Missing required PO field: {field}")
    
    elif document_type == DocumentType.RECEIPT:
        required_fields = ['merchant_name', 'total_amount', 'transaction_date']
        for field in required_fields:
            if not extracted_data.get(field):
                errors.append(f"Missing required receipt field: {field}")
    
    # Add more document type validations as needed
    
    return errors


# =============================================================================
# MODEL REGISTRY
# =============================================================================

# Export all models for easy importing
__all__ = [
    # Enums
    'DocumentType', 'ProcessingStrategy', 'ProcessingStatus', 
    'IntegrationType', 'WebhookEvent',
    
    # Base Models
    'BaseResponse', 'ErrorResponse', 'SuccessResponse',
    
    # Processing Models
    'ProcessingConfig', 'DocumentMetadata', 'ProcessingRequest',
    'ClassificationRequest', 'BatchProcessingRequest',
    
    # Response Models
    'CompetitiveResult', 'ProcessingResult', 'ProcessingResponse',
    'BatchProcessingResponse', 'ProcessingStatusResponse',
    
    # Integration Models
    'IntegrationConfig', 'IntegrationResult',
    
    # Webhook Models
    'WebhookConfig', 'WebhookPayload',
    
    # System Models
    'HealthResponse', 'MetricsResponse',
    
    # Authentication Models
    'UserInfo', 'TokenResponse', 'LoginRequest',
    
    # Analytics Models
    'ProcessingAnalytics', 'CostAnalysis',
    
    # Utilities
    'validate_document_data'
]