"""
Base Enterprise Integration Framework
Common interface and utilities for all accounting system integrations
"""

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.models.api_models import ProcessingResult, IntegrationType
from api.models.database_models import Integration, Document
from api.config import get_settings
from utils.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class IntegrationStatus(str, Enum):
    """Integration connection status"""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    ERROR = "error"
    AUTHENTICATING = "authenticating"
    EXPIRED = "expired"


class PostingStatus(str, Enum):
    """Document posting status"""
    PENDING = "pending"
    POSTED = "posted"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IntegrationCredentials(BaseModel):
    """Base credentials model"""
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    api_key: Optional[str] = None
    server_url: Optional[str] = None
    tenant_id: Optional[str] = None
    
    class Config:
        extra = "allow"  # Allow additional fields for system-specific credentials


class IntegrationResult(BaseModel):
    """Result from integration operation"""
    success: bool
    transaction_id: Optional[str] = None
    external_reference: Optional[str] = None
    posted_amount: Optional[Decimal] = None
    posting_date: Optional[datetime] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentMapping(BaseModel):
    """Document field mapping configuration"""
    source_field: str
    target_field: str
    transformation: Optional[str] = None  # Optional transformation function name
    required: bool = False
    default_value: Optional[Any] = None


class ValidationRule(BaseModel):
    """Business rule for document validation"""
    field_name: str
    rule_type: str  # required, range, format, custom
    parameters: Dict[str, Any] = Field(default_factory=dict)
    error_message: str = "Validation failed"


class BaseIntegration(ABC):
    """
    Abstract base class for all enterprise integrations
    
    Provides common functionality:
    - Authentication management
    - Error handling and retry logic
    - Rate limiting and throttling
    - Audit logging and monitoring
    - Data mapping and transformation
    - Validation and business rules
    """
    
    def __init__(
        self,
        integration_type: IntegrationType,
        organization_id: str,
        integration_config: Dict[str, Any]
    ):
        self.integration_type = integration_type
        self.organization_id = organization_id
        self.config = integration_config
        
        # Connection state
        self.credentials: Optional[IntegrationCredentials] = None
        self.status = IntegrationStatus.DISCONNECTED
        self.last_sync_time: Optional[datetime] = None
        self.last_error: Optional[str] = None
        
        # Performance metrics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
        
        # HTTP session for connection reuse
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Field mappings and validation rules
        self.field_mappings: List[DocumentMapping] = []
        self.validation_rules: List[ValidationRule] = []
        
        logger.info(f"Initialized {integration_type.value} integration for org {organization_id}")
    
    async def initialize(self):
        """Initialize integration with credentials and configuration"""
        try:
            # Load credentials from database
            await self._load_credentials()
            
            # Load field mappings and validation rules
            await self._load_configuration()
            
            # Create HTTP session
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=settings.integrations.webhook_timeout)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)
            
            # Test connection
            await self.test_connection()
            
            logger.info(f"Integration {self.integration_type.value} initialized successfully")
            
        except Exception as e:
            logger.error(f"Integration initialization failed: {e}")
            self.status = IntegrationStatus.ERROR
            self.last_error = str(e)
            raise
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    @abstractmethod
    async def authenticate(self, credentials: IntegrationCredentials) -> bool:
        """
        Authenticate with the external system
        
        Args:
            credentials: Integration-specific credentials
            
        Returns:
            True if authentication successful, False otherwise
        """
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test connection to external system
        
        Returns:
            True if connection successful, False otherwise
        """
        logger.info(f'Method {function_name} called')
        return {}
    
    @abstractmethod
    async def post_document(
        self,
        document_data: Dict[str, Any],
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Post document to external system
        
        Args:
            document_data: Extracted document data
            document_type: Type of document (invoice, purchase_order, etc.)
            metadata: Additional metadata
            
        Returns:
            Integration result with transaction details
        """
        pass
    
    @abstractmethod
    async def get_posting_status(self, transaction_id: str) -> PostingStatus:
        """
        Get status of previously posted document
        
        Args:
            transaction_id: Transaction ID from posting operation
            
        Returns:
            Current posting status
        """
        return {}
    
    @abstractmethod
    async def cancel_posting(self, transaction_id: str) -> bool:
        """
        Cancel a pending posting operation
        
        Args:
            transaction_id: Transaction ID to cancel
            
        Returns:
            True if cancelled successfully, False otherwise
        """
        logger.info(f'Method {function_name} called')
        return {}
    
    async def post_document_safe(
        self,
        document_data: Dict[str, Any],
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Post document with comprehensive error handling and retry logic
        """
        max_retries = 3
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                # Validate document data
                validation_result = await self.validate_document(document_data, document_type)
                if not validation_result["valid"]:
                    return IntegrationResult(
                        success=False,
                        error_message=f"Validation failed: {validation_result['errors']}",
                        error_code="VALIDATION_ERROR"
                    )
                
                # Map document fields
                mapped_data = await self.map_document_fields(document_data, document_type)
                
                # Apply business rules
                rules_result = await self.apply_business_rules(mapped_data, document_type)
                if not rules_result["valid"]:
                    return IntegrationResult(
                        success=False,
                        error_message=f"Business rules failed: {rules_result['errors']}",
                        error_code="BUSINESS_RULES_ERROR"
                    )
                
                # Perform posting
                start_time = datetime.utcnow()
                result = await self.post_document(mapped_data, document_type, metadata)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                # Update metrics
                self.total_requests += 1
                if result.success:
                    self.successful_requests += 1
                else:
                    self.failed_requests += 1
                
                self._update_avg_response_time(duration)
                
                # Log operation
                await self._log_integration_operation(
                    "post_document", 
                    result.success, 
                    result.error_message,
                    {"document_type": document_type, "attempt": attempt + 1}
                )
                
                return result
                
            except aiohttp.ClientTimeout:
                logger.warning(f"Integration timeout (attempt {attempt + 1}/{max_retries})")
                if attempt == max_retries - 1:
                    return IntegrationResult(
                        success=False,
                        error_message="Request timeout",
                        error_code="TIMEOUT_ERROR"
                    )
                
            except aiohttp.ClientError as e:
                logger.warning(f"Integration client error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return IntegrationResult(
                        success=False,
                        error_message=f"Client error: {str(e)}",
                        error_code="CLIENT_ERROR"
                    )
                
            except Exception as e:
                logger.error(f"Integration error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    return IntegrationResult(
                        success=False,
                        error_message=f"Unexpected error: {str(e)}",
                        error_code="UNEXPECTED_ERROR"
                    )
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
        
        # Should never reach here, but just in case
        return IntegrationResult(
            success=False,
            error_message="Max retries exceeded",
            error_code="MAX_RETRIES_EXCEEDED"
        )
    
    async def validate_document(self, document_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """
        Validate document data against integration requirements
        
        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        errors = []
        
        # Check required fields based on document type
        required_fields = self._get_required_fields(document_type)
        for field in required_fields:
            if not document_data.get(field):
                errors.append(f"Required field missing: {field}")
        
        # Apply validation rules
        for rule in self.validation_rules:
            field_value = document_data.get(rule.field_name)
            
            if rule.rule_type == "required" and not field_value:
                errors.append(f"Required field: {rule.field_name}")
                
            elif rule.rule_type == "range" and field_value is not None:
                min_val = rule.parameters.get("min")
                max_val = rule.parameters.get("max")
                
                if min_val is not None and float(field_value) < min_val:
                    errors.append(f"{rule.field_name} below minimum: {min_val}")
                if max_val is not None and float(field_value) > max_val:
                    errors.append(f"{rule.field_name} above maximum: {max_val}")
                    
            elif rule.rule_type == "format" and field_value is not None:
                import re
                pattern = rule.parameters.get("pattern")
                if pattern and not re.match(pattern, str(field_value)):
                    errors.append(f"{rule.field_name} format invalid: {rule.error_message}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def map_document_fields(self, document_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """
        Map document fields from source to target system format
        
        Args:
            document_data: Source document data
            document_type: Type of document
            
        Returns:
            Mapped document data
        """
        mapped_data = {}
        
        # Apply field mappings
        for mapping in self.field_mappings:
            source_value = document_data.get(mapping.source_field)
            
            if source_value is not None:
                # Apply transformation if specified
                if mapping.transformation:
                    source_value = await self._apply_transformation(source_value, mapping.transformation)
                
                mapped_data[mapping.target_field] = source_value
                
            elif mapping.required:
                # Use default value if available
                if mapping.default_value is not None:
                    mapped_data[mapping.target_field] = mapping.default_value
                else:
                    logger.warning(f"Required field {mapping.source_field} missing and no default value")
        
        return mapped_data
    
    async def apply_business_rules(self, document_data: Dict[str, Any], document_type: str) -> Dict[str, Any]:
        """
        Apply business rules and transformations
        
        Returns:
            Dictionary with 'valid' boolean and 'errors' list
        """
        errors = []
        
        # Document type specific business rules
        if document_type == "invoice":
            errors.extend(await self._validate_invoice_rules(document_data))
        elif document_type == "purchase_order":
            errors.extend(await self._validate_purchase_order_rules(document_data))
        elif document_type == "receipt":
            errors.extend(await self._validate_receipt_rules(document_data))
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def get_integration_health(self) -> Dict[str, Any]:
        """Get comprehensive integration health status"""
        try:
            # Test connection
            connection_healthy = await self.test_connection()
            
            # Calculate success rate
            success_rate = 0.0
            if self.total_requests > 0:
                success_rate = (self.successful_requests / self.total_requests) * 100
            
            return {
                "integration_type": self.integration_type.value,
                "status": self.status.value,
                "connection_healthy": connection_healthy,
                "last_sync": self.last_sync_time.isoformat() if self.last_sync_time else None,
                "last_error": self.last_error,
                "metrics": {
                    "total_requests": self.total_requests,
                    "successful_requests": self.successful_requests,
                    "failed_requests": self.failed_requests,
                    "success_rate_percent": round(success_rate, 2),
                    "avg_response_time_seconds": round(self.avg_response_time, 3)
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "integration_type": self.integration_type.value,
                "status": "error",
                "connection_healthy": False,
                "last_error": str(e),
                "metrics": {}
            }
    
    # Protected helper methods
    
    async def _load_credentials(self):
        """Load credentials from database"""
        # This would load encrypted credentials from the Integration model
        # For now, we'll load from config
        creds_dict = self.config.get("credentials", {})
        self.credentials = IntegrationCredentials(**creds_dict)
    
    async def _load_configuration(self):
        """Load field mappings and validation rules"""
        # Load field mappings
        mappings_config = self.config.get("field_mappings", {})
        for source_field, config in mappings_config.items():
            if isinstance(config, str):
                # Simple mapping
                self.field_mappings.append(DocumentMapping(
                    source_field=source_field,
                    target_field=config
                ))
            else:
                # Complex mapping
                self.field_mappings.append(DocumentMapping(**config, source_field=source_field))
        
        # Load validation rules
        rules_config = self.config.get("validation_rules", [])
        for rule_config in rules_config:
            self.validation_rules.append(ValidationRule(**rule_config))
    
    def _get_required_fields(self, document_type: str) -> List[str]:
        """Get required fields for document type"""
        required_fields_map = {
            "invoice": ["invoice_number", "vendor_name", "total_amount", "invoice_date"],
            "purchase_order": ["po_number", "vendor_name", "total_amount"],
            "receipt": ["merchant_name", "total_amount", "transaction_date"],
            "bank_statement": ["account_number", "statement_date"]
        }
        
        return required_fields_map.get(document_type, [])
    
    async def _apply_transformation(self, value: Any, transformation: str) -> Any:
        """Apply transformation function to field value"""
        if transformation == "uppercase":
            return str(value).upper()
        elif transformation == "lowercase":
            return str(value).lower()
        elif transformation == "strip":
            return str(value).strip()
        elif transformation == "to_decimal":
            return Decimal(str(value))
        elif transformation == "to_date":
            if isinstance(value, str):
                return datetime.fromisoformat(value)
            return value
        else:
            logger.warning(f"Unknown transformation: {transformation}")
            return value
    
    async def _validate_invoice_rules(self, data: Dict[str, Any]) -> List[str]:
        """Validate invoice-specific business rules"""
        errors = []
        
        # Invoice amount should be positive
        total_amount = data.get("total_amount")
        if total_amount and float(total_amount) <= 0:
            errors.append("Invoice amount must be positive")
        
        # Invoice date should not be in the future
        invoice_date = data.get("invoice_date")
        if invoice_date:
            if isinstance(invoice_date, str):
                try:
                    invoice_date = datetime.fromisoformat(invoice_date)
                except ValueError:
                    errors.append("Invalid invoice date format")
                    return errors
            
            if invoice_date > datetime.utcnow():
                errors.append("Invoice date cannot be in the future")
        
        return errors
    
    async def _validate_purchase_order_rules(self, data: Dict[str, Any]) -> List[str]:
        """Validate purchase order specific business rules"""
        errors = []
        
        # PO amount should be positive
        total_amount = data.get("total_amount")
        if total_amount and float(total_amount) <= 0:
            errors.append("Purchase order amount must be positive")
        
        return errors
    
    async def _validate_receipt_rules(self, data: Dict[str, Any]) -> List[str]:
        """Validate receipt specific business rules"""
        errors = []
        
        # Receipt amount should be positive
        total_amount = data.get("total_amount")
        if total_amount and float(total_amount) <= 0:
            errors.append("Receipt amount must be positive")
        
        return errors
    
    def _update_avg_response_time(self, duration: float):
        """Update average response time with exponential moving average"""
        alpha = 0.1
        if self.avg_response_time == 0:
            self.avg_response_time = duration
        else:
            self.avg_response_time = (1 - alpha) * self.avg_response_time + alpha * duration
    
    async def _log_integration_operation(
        self,
        operation: str,
        success: bool,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log integration operation for audit trail"""
        log_data = {
            "integration_type": self.integration_type.value,
            "organization_id": self.organization_id,
            "operation": operation,
            "success": success,
            "error_message": error_message,
            "metadata": metadata or {}
        }
        
        if success:
            logger.info(f"Integration operation successful: {operation}", extra=log_data)
        else:
            logger.error(f"Integration operation failed: {operation}", extra=log_data)


# Integration factory
class IntegrationFactory:
    """Factory for creating integration instances"""
    
    @staticmethod
    def create_integration(
        integration_type: IntegrationType,
        organization_id: str,
        config: Dict[str, Any]
    ) -> BaseIntegration:
        """Create integration instance based on type"""
        
        if integration_type == IntegrationType.QUICKBOOKS:
            from api.integrations.quickbooks_integration import QuickBooksIntegration
            return QuickBooksIntegration(organization_id, config)
            
        elif integration_type == IntegrationType.SAP:
            from api.integrations.sap_integration import SAPIntegration
            return SAPIntegration(organization_id, config)
            
        elif integration_type == IntegrationType.NETSUITE:
            from api.integrations.netsuite_integration import NetSuiteIntegration
            return NetSuiteIntegration(organization_id, config)
            
        elif integration_type == IntegrationType.XERO:
            from api.integrations.xero_integration import XeroIntegration
            return XeroIntegration(organization_id, config)
            
        else:
            raise ValueError(f"Unsupported integration type: {integration_type}")


# Export classes
__all__ = [
    "BaseIntegration",
    "IntegrationCredentials", 
    "IntegrationResult",
    "DocumentMapping",
    "ValidationRule",
    "IntegrationStatus",
    "PostingStatus",
    "IntegrationFactory"
]