"""
Comprehensive API Tests
Test suite for the main FastAPI application
"""

import asyncio
from pathlib import Path
import json
import pytest
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import AsyncMock, patch, MagicMock

import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from api.main import app
from api.models.api_models import ProcessingRequest, ProcessingResponse, DocumentType
from api.models.database_models import User, Organization, Document
from api.config import get_settings
from tests.conftest import test_settings


class TestHealthEndpoints:
    """Test system health and monitoring endpoints"""
    
    def test_health_check_success(self, client: TestClient):
        """Test successful health check"""
        response = client.get(str(Path("/health").resolve()))
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        
    def test_health_check_detailed(self, client: TestClient):
        """Test detailed health check response"""
        with patch('api.services.monitoring_service.MonitoringService.get_health_status') as mock_health:
            mock_health.return_value = {
                "overall_healthy": True,
                "database_connected": True,
                "processing_service_active": True,
                "external_services": {"quickbooks": True},
                "avg_response_time_ms": 150.5,
                "active_connections": 5,
                "memory_usage_mb": 512.0,
                "cpu_usage_percent": 25.5,
                "uptime_seconds": 3600
            }
            
            response = client.get(str(Path("/health").resolve()))
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert data["database_connected"] == True
            assert data["processing_service_active"] == True
            assert "performance_metrics" in data
    
    def test_health_check_unhealthy(self, client: TestClient):
        """Test health check when system is unhealthy"""
        with patch('api.services.monitoring_service.MonitoringService.get_health_status') as mock_health:
            mock_health.return_value = {
                "overall_healthy": False,
                "database_connected": False,
                "processing_service_active": True,
                "external_services": {},
                "avg_response_time_ms": 0,
                "active_connections": 0,
                "memory_usage_mb": 0,
                "cpu_usage_percent": 0,
                "uptime_seconds": 100
            }
            
            response = client.get(str(Path("/health").resolve()))
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["database_connected"] == False


class TestAuthenticationEndpoints:
    """Test authentication and authorization endpoints"""
    
    def test_login_success(self, client: TestClient):
        """Test successful login"""
        with patch('api.auth.auth_manager.AuthManager.authenticate') as mock_auth:
            mock_auth.return_value = {
                "access_token": "test_token_123",
                "token_type": "bearer",
                "expires_in": 1800,
                "refresh_token": "refresh_token_123",
                "user_info": {
                    "user_id": "user_123",
                    "username": "testuser",
                    "email": "test@example.com",
                    "organization_id": "org_123"
                }
            }
            
            response = client.post(str(Path("/auth/login").resolve()), json={
                "username": "testuser",
                "password": "testpass123"
            })
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["access_token"] == "test_token_123"
            assert data["token_type"] == "bearer"
            assert "user_info" in data
    
    def test_login_invalid_credentials(self, client: TestClient):
        """Test login with invalid credentials"""
        with patch('api.auth.auth_manager.AuthManager.authenticate') as mock_auth:
            mock_auth.side_effect = Exception("Invalid credentials")
            
            response = client.post(str(Path("/auth/login").resolve()), json={
                "username": "invalid",
                "password": "wrongpass"
            })
            
            assert response.status_code == 401
            assert "Invalid credentials" in response.json()["detail"]
    
    def test_token_refresh(self, client: TestClient):
        """Test token refresh"""
        with patch('api.auth.auth_manager.AuthManager.refresh_token') as mock_refresh:
            mock_refresh.return_value = {
                "access_token": "new_token_456",
                "token_type": "bearer",
                "expires_in": 1800
            }
            
            response = client.post(str(Path("/auth/refresh").resolve()), json={
                "refresh_token": "refresh_token_123"
            })
            
            assert response.status_code == 200
            data = response.json()
            assert data["access_token"] == "new_token_456"
    
    def test_logout(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test user logout"""
        with patch('api.auth.auth_manager.AuthManager.logout') as mock_logout:
            mock_logout.return_value = None
            
            response = client.post(str(Path("/auth/logout").resolve()), headers=auth_headers)
            
            assert response.status_code == 200
            assert response.json()["message"] == "Successfully logged out"


class TestDocumentProcessingEndpoints:
    """Test document processing API endpoints"""
    
    def test_process_document_text_success(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test successful document processing from text"""
        with patch('api.services.processing_service.ProcessingService.process_document') as mock_process:
            mock_process.return_value = {
                "document_id": "doc_123",
                "success": True,
                "classification": {
                    "document_type": "invoice",
                    "confidence": 0.95
                },
                "extracted_data": {
                    "invoice_number": "INV-001",
                    "total_amount": "1000.00",
                    "vendor_name": "ACME Corp"
                },
                "confidence_score": 0.95,
                "processing_time_ms": 1500,
                "validation_errors": [],
                "cost_breakdown": {
                    "classification_cost": Decimal("0.01"),
                    "processing_cost": Decimal("0.02"),
                    "total_cost": Decimal("0.03")
                },
                "processing_method": "competitive_multi_domain"
            }
            
            request_data = {
                "text_content": "INVOICE #INV-001\nACME Corp\nTotal: $1,000.00",
                "processing_config": {
                    "strategy": "competitive",
                    "accuracy_threshold": 0.95
                }
            }
            
            response = client.post(
                str(Path("/api/v1/documents/process").resolve()),
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["document_id"] == "doc_123"
            assert data["classification"]["document_type"] == "invoice"
            assert data["extracted_data"]["invoice_number"] == "INV-001"
            assert data["confidence_score"] == 0.95
    
    def test_process_document_validation_error(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test document processing with validation error"""
        # Test with no input provided
        response = client.post(
            str(Path("/api/v1/documents/process").resolve()),
            json={},
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_process_document_service_error(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test document processing with service error"""
        with patch('api.services.processing_service.ProcessingService.process_document') as mock_process:
            mock_process.side_effect = Exception("Processing service unavailable")
            
            request_data = {
                "text_content": "Test document content"
            }
            
            response = client.post(
                str(Path("/api/v1/documents/process").resolve()),
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 500
            assert "Processing service unavailable" in response.json()["detail"]
    
    def test_classify_document_success(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test successful document classification"""
        with patch('api.services.processing_service.ProcessingService.classify_document') as mock_classify:
            mock_classify.return_value = {
                "document_type": "invoice",
                "confidence_score": 0.92,
                "processing_time_ms": 500,
                "cost": Decimal("0.01"),
                "supported_fields": [
                    "invoice_number", "vendor_name", "total_amount", 
                    "invoice_date", "due_date"
                ]
            }
            
            request_data = {
                "text_content": "INVOICE #12345\nVendor: Test Company\nAmount: $500.00"
            }
            
            response = client.post(
                str(Path("/api/v1/documents/classify").resolve()),
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["document_type"] == "invoice"
            assert data["confidence_score"] == 0.92
            assert "supported_fields" in data
    
    def test_get_document_status(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test getting document processing status"""
        with patch('api.services.processing_service.ProcessingService.get_document_status') as mock_status:
            mock_status.return_value = {
                "document_id": "doc_123",
                "status": "completed",
                "progress_percentage": 100.0,
                "current_stage": "completed",
                "estimated_completion_time": None,
                "error_message": None,
                "partial_results": {
                    "classification": {"document_type": "invoice"}
                }
            }
            
            response = client.get(
                str(Path("/api/v1/documents/doc_123/status").resolve()),
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["document_id"] == "doc_123"
            assert data["status"] == "completed"
            assert data["progress_percentage"] == 100.0
    
    def test_get_document_result(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test getting document processing result"""
        with patch('api.services.processing_service.ProcessingService.get_document_result') as mock_result:
            mock_result.return_value = {
                "document_id": "doc_123",
                "filename": "test_invoice.pdf",
                "document_type": "invoice",
                "extracted_data": {
                    "invoice_number": "INV-001",
                    "total_amount": "1500.00"
                },
                "confidence_score": 0.96,
                "processing_time_ms": 2000,
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:30:02Z"
            }
            
            response = client.get(
                str(Path("/api/v1/documents/doc_123/result").resolve()),
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["document_id"] == "doc_123"
            assert data["document_type"] == "invoice"
            assert data["extracted_data"]["invoice_number"] == "INV-001"


class TestBatchProcessingEndpoints:
    """Test batch document processing endpoints"""
    
    def test_batch_process_documents(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test batch document processing"""
        with patch('api.services.processing_service.ProcessingService.batch_process_documents') as mock_batch:
            mock_batch.return_value = {
                "batch_id": "batch_123",
                "status": "processing",
                "total_documents": 3,
                "estimated_completion_time": "2024-01-15T10:35:00Z"
            }
            
            request_data = {
                "documents": [
                    {"text_content": "Invoice #1 content"},
                    {"text_content": "Invoice #2 content"},
                    {"text_content": "Receipt content"}
                ],
                "batch_name": "test_batch",
                "max_concurrent_documents": 2
            }
            
            response = client.post(
                str(Path("/api/v1/documents/batch").resolve()),
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["batch_id"] == "batch_123"
            assert data["total_documents"] == 3
            assert data["status"] == "processing"


class TestFileUploadEndpoints:
    """Test file upload endpoints"""
    
    def test_upload_document_success(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test successful file upload and processing"""
        with patch('api.services.processing_service.ProcessingService.process_uploaded_file') as mock_upload:
            mock_upload.return_value = {
                "document_id": "doc_456",
                "success": True,
                "filename": "test.pdf",
                "classification": {"document_type": "invoice"},
                "extracted_data": {"invoice_number": "TEST-001"}
            }
            
            # Create test file
            test_file = ("test.pdf", b"fake pdf content", "application/pdf")
            
            response = client.post(
                str(Path("/api/v1/documents/upload").resolve()),
                files={"file": test_file},
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["success"] == True
            assert data["document_id"] == "doc_456"
    
    def test_upload_file_too_large(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test file upload with size limit exceeded"""
        with patch('api.services.processing_service.ProcessingService.process_uploaded_file') as mock_upload:
            mock_upload.side_effect = Exception("File size exceeds maximum limit of 50MB")
            
            # Create large test file
            large_file = ("large.pdf", b"x" * (51 * 1024 * 1024), "application/pdf")
            
            response = client.post(
                str(Path("/api/v1/documents/upload").resolve()),
                files={"file": large_file},
                headers=auth_headers
            )
            
            assert response.status_code == 500
            assert "File size exceeds maximum limit" in response.json()["detail"]


class TestWebhookEndpoints:
    """Test webhook management endpoints"""
    
    def test_create_webhook_success(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test successful webhook creation"""
        with patch('api.services.webhook_service.WebhookService.create_webhook') as mock_create:
            mock_create.return_value = {
                "webhook_id": "webhook_123",
                "status": "active",
                "events": ["document.processed", "document.failed"],
                "url": "https://example.com/webhook",
                "created_at": "2024-01-15T10:30:00Z"
            }
            
            request_data = {
                "url": "https://example.com/webhook",
                "events": ["document.processed", "document.failed"],
                "secret": "webhook_secret_123",
                "retry_attempts": 3,
                "timeout_seconds": 30,
                "active": True
            }
            
            response = client.post(
                str(Path("/api/v1/webhooks").resolve()),
                json=request_data,
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["webhook_id"] == "webhook_123"
            assert data["status"] == "active"
    
    def test_list_webhooks(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test listing webhooks"""
        with patch('api.services.webhook_service.WebhookService.list_webhooks') as mock_list:
            mock_list.return_value = [
                {
                    "id": "webhook_123",
                    "name": "webhook_1",
                    "url": "https://example.com/webhook1",
                    "events": ["document.processed"],
                    "is_active": True,
                    "total_deliveries": 10,
                    "successful_deliveries": 9,
                    "failed_deliveries": 1
                }
            ]
            
            response = client.get(
                str(Path("/api/v1/webhooks").resolve()),
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "webhooks" in data
            assert len(data["webhooks"]) == 1
            assert data["webhooks"][0]["id"] == "webhook_123"


class TestMetricsEndpoints:
    """Test monitoring and metrics endpoints"""
    
    def test_get_system_metrics(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test getting system metrics"""
        with patch('api.services.monitoring_service.MonitoringService.get_comprehensive_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "timestamp": "2024-01-15T10:30:00Z",
                "processing": {
                    "total_documents_24h": 100,
                    "success_rate_percent": 95.0
                },
                "performance": {
                    "avg_response_time_ms": 250.5,
                    "p95_response_time_ms": 500.0
                },
                "usage": {
                    "active_users_24h": 25
                },
                "cost": {
                    "total_cost_24h": 15.75
                }
            }
            
            response = client.get(
                str(Path("/api/v1/metrics").resolve()),
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "processing" in data
            assert "performance" in data
            assert data["processing"]["total_documents_24h"] == 100
    
    def test_get_processing_analytics(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test getting processing analytics"""
        with patch('api.services.monitoring_service.MonitoringService.get_processing_analytics') as mock_analytics:
            mock_analytics.return_value = {
                "period": {
                    "start_date": "2024-01-01T00:00:00Z",
                    "end_date": "2024-01-15T23:59:59Z"
                },
                "summary": {
                    "total_documents": 500,
                    "success_rate_percent": 96.2,
                    "average_confidence_score": 0.94
                },
                "document_type_breakdown": {
                    "invoice": {"count": 300, "successful": 290},
                    "receipt": {"count": 200, "successful": 191}
                }
            }
            
            response = client.get(
                str(Path("/api/v1/analytics/processing").resolve()),
                params={
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-15"
                },
                headers=auth_headers
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert "summary" in data
            assert data["summary"]["total_documents"] == 500
            assert "document_type_breakdown" in data


class TestIntegrationEndpoints:
    """Test enterprise integration endpoints"""
    
    def test_get_integration_status(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test getting integration status"""
        response = client.get(
            str(Path("/api/v1/integrations/status").resolve()),
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should return status for all integrations
        assert "quickbooks" in data
        assert "sap" in data
        assert "netsuite" in data
        assert "xero" in data


class TestErrorHandling:
    """Test API error handling"""
    
    def test_404_not_found(self, client: TestClient):
        """Test 404 error handling"""
        response = client.get(str(Path("/nonexistent/endpoint").resolve()))
        
        assert response.status_code == 404
    
    def test_unauthorized_access(self, client: TestClient):
        """Test unauthorized access"""
        response = client.post(str(Path("/api/v1/documents/process").resolve()), json={})
        
        assert response.status_code == 401
    
    def test_invalid_json(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test invalid JSON handling"""
        response = client.post(
            str(Path("/api/v1/documents/process").resolve()),
            data="invalid json",
            headers={**auth_headers, "content-type": "application/json"}
        )
        
        assert response.status_code == 422


class TestAPIValidation:
    """Test API input validation"""
    
    def test_processing_request_validation(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test processing request validation"""
        # Test missing required fields
        response = client.post(
            str(Path("/api/v1/documents/process").resolve()),
            json={"processing_config": {"strategy": "invalid_strategy"}},
            headers=auth_headers
        )
        
        assert response.status_code == 422
        
        # Test invalid accuracy threshold
        response = client.post(
            str(Path("/api/v1/documents/process").resolve()),
            json={
                "text_content": "test",
                "processing_config": {"accuracy_threshold": 1.5}  # > 1.0
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422
    
    def test_webhook_validation(self, client: TestClient, auth_headers: Dict[str, str]):
        """Test webhook configuration validation"""
        # Test invalid URL
        response = client.post(
            str(Path("/api/v1/webhooks").resolve()),
            json={
                "url": "invalid_url",
                "events": ["document.processed"]
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422


@pytest.mark.asyncio
class TestAsyncEndpoints:
    """Test async-specific functionality"""
    
    async def test_concurrent_processing_requests(self, async_client: AsyncClient, auth_headers: Dict[str, str]):
        """Test handling concurrent processing requests"""
        with patch('api.services.processing_service.ProcessingService.process_document') as mock_process:
            mock_process.return_value = {
                "document_id": f"doc_{asyncio.current_task().get_name()}",
                "success": True,
                "classification": {"document_type": "invoice"},
                "extracted_data": {},
                "confidence_score": 0.95,
                "processing_time_ms": 1000
            }
            
            # Send multiple requests concurrently
            requests = []
            for i in range(5):
                request = async_client.post(
                    str(Path("/api/v1/documents/process").resolve()),
                    json={"text_content": f"Document {i}"},
                    headers=auth_headers
                )
                requests.append(request)
            
            responses = await asyncio.gather(*requests)
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
                data = response.json()
                assert data["success"] == True


# Integration test fixtures and utilities
@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "invoice": {
            "text_content": """
            INVOICE #INV-2024-001
            Date: January 15, 2024
            
            From: ACME Corporation
            123 Business Street
            Business City, BC 12345
            
            To: Client Company
            456 Client Avenue
            Client City, CC 67890
            
            Description: Professional Services
            Amount: $2,500.00
            Tax (10%): $250.00
            Total: $2,750.00
            
            Payment Due: February 15, 2024
            """,
            "expected_data": {
                "invoice_number": "INV-2024-001",
                "vendor_name": "ACME Corporation", 
                "total_amount": "2750.00",
                "invoice_date": "2024-01-15",
                "due_date": "2024-02-15"
            }
        },
        "receipt": {
            "text_content": """
            RECEIPT
            Store: Tech Electronics
            Date: 2024-01-15 14:30
            Transaction: TXN-789123
            
            Items:
            - USB Cable: $15.99
            - Phone Case: $24.95
            - Screen Protector: $12.99
            
            Subtotal: $53.93
            Tax: $5.39
            Total: $59.32
            
            Payment: Credit Card ****1234
            """,
            "expected_data": {
                "merchant_name": "Tech Electronics",
                "transaction_amount": "59.32",
                "transaction_date": "2024-01-15"
            }
        }
    }


@pytest.fixture
def mock_processing_service():
    """Mock processing service for testing"""
    with patch('api.services.processing_service.ProcessingService') as mock:
        service_instance = mock.return_value
        service_instance.process_document = AsyncMock()
        service_instance.classify_document = AsyncMock()
        service_instance.get_document_status = AsyncMock()
        service_instance.get_document_result = AsyncMock()
        yield service_instance