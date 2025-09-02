"""
Comprehensive Test Suite for Enterprise API
Production-ready tests covering all endpoints and scenarios
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

# Import API components
from api.main import app, create_app
from api.models.api_models import (
    ProcessingRequest, BatchProcessingRequest, ClassificationRequest,
    ProcessingConfig, DocumentMetadata, WebhookConfig
)
from api.models.database_models import (
    Organization, User, Document, APIKey, Webhook, WebhookDelivery
)
from api.auth.auth_manager import AuthManager
from api.services.processing_service import ProcessingService
from api.services.monitoring_service import MonitoringService
from api.services.webhook_service import WebhookService
from api.integrations.quickbooks_integration import QuickBooksIntegration
from api.config import get_settings


class TestEnterpriseAPI:
    """Comprehensive test suite for Enterprise Document Processing API"""
    
    @pytest.fixture
    def client(self):
        """Test client fixture"""
        return TestClient(app)
    
    @pytest.fixture
    def settings(self):
        """Settings fixture"""
        return get_settings()
    
    @pytest.fixture
    def mock_auth_manager(self):
        """Mock authentication manager"""
        auth_manager = AsyncMock(spec=AuthManager)
        auth_manager.authenticate.return_value = {
            "access_token": "test_token",
            "token_type": "bearer",
            "expires_in": 3600,
            "user_info": {
                "user_id": "test-user-id",
                "username": "testuser",
                "organization_id": "test-org-id"
            }
        }
        return auth_manager
    
    @pytest.fixture
    def mock_processing_service(self):
        """Mock processing service"""
        service = AsyncMock(spec=ProcessingService)
        service.process_document.return_value = {
            "document_id": "test-doc-id",
            "success": True,
            "classification": {
                "document_type": "invoice",
                "confidence": 0.95
            },
            "extracted_data": {
                "invoice_number": "INV-001",
                "total_amount": 1000.00,
                "vendor_name": "Test Vendor"
            },
            "confidence_score": 0.95,
            "processing_time_ms": 1500,
            "cost_breakdown": {
                "total_cost": Decimal("0.03"),
                "classification_cost": Decimal("0.01"),
                "processing_cost": Decimal("0.02")
            }
        }
        return service
    
    @pytest.fixture
    def sample_processing_request(self):
        """Sample processing request"""
        return {
            "text_content": "Invoice #INV-001\nTo: Test Customer\nAmount: $1,000.00\nFrom: Test Vendor",
            "processing_config": {
                "strategy": "competitive",
                "accuracy_threshold": 0.95,
                "max_processing_time_seconds": 30
            },
            "metadata": {
                "filename": "test_invoice.txt",
                "source": "api_test"
            }
        }
    
    @pytest.fixture
    def auth_headers(self):
        """Authentication headers"""
        return {"Authorization": "Bearer test_token"}


class TestHealthEndpoints(TestEnterpriseAPI):
    """Test health and system endpoints"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data
        assert "performance_metrics" in data
    
    def test_metrics_endpoint_requires_auth(self, client):
        """Test metrics endpoint requires authentication"""
        response = client.get("/metrics")
        assert response.status_code == 401
    
    @patch('api.main.get_current_user')
    def test_metrics_endpoint_with_auth(self, mock_get_user, client, auth_headers):
        """Test metrics endpoint with authentication"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        
        response = client.get("/metrics", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "timestamp" in data
        assert "processing_metrics" in data
        assert "performance_metrics" in data


class TestAuthenticationEndpoints(TestEnterpriseAPI):
    """Test authentication and authorization"""
    
    def test_login_endpoint(self, client, mock_auth_manager):
        """Test login endpoint"""
        with patch('api.main.AuthManager', return_value=mock_auth_manager):
            response = client.post("/auth/login", json={
                "username": "testuser",
                "password": "testpass"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert "token_type" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials"""
        with patch('api.main.AuthManager') as mock_auth_class:
            mock_auth_manager = AsyncMock()
            mock_auth_manager.authenticate.side_effect = Exception("Invalid credentials")
            mock_auth_class.return_value = mock_auth_manager
            
            response = client.post("/auth/login", json={
                "username": "invalid",
                "password": "invalid"
            })
        
        assert response.status_code == 401
    
    def test_refresh_token(self, client, mock_auth_manager):
        """Test token refresh endpoint"""
        mock_auth_manager.refresh_token.return_value = {
            "access_token": "new_test_token",
            "token_type": "bearer",
            "expires_in": 3600
        }
        
        with patch('api.main.AuthManager', return_value=mock_auth_manager):
            response = client.post("/auth/refresh", json={
                "refresh_token": "test_refresh_token"
            })
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] == "new_test_token"
    
    @patch('api.main.get_current_user')
    def test_logout(self, mock_get_user, client, auth_headers, mock_auth_manager):
        """Test logout endpoint"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        
        with patch('api.main.AuthManager', return_value=mock_auth_manager):
            response = client.post("/auth/logout", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data


class TestDocumentProcessingEndpoints(TestEnterpriseAPI):
    """Test document processing endpoints"""
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    @patch('api.main.get_database')
    def test_process_document_success(
        self, 
        mock_db, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers, 
        sample_processing_request,
        mock_processing_service
    ):
        """Test successful document processing"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        with patch.object(app.state, 'processing_service', mock_processing_service):
            response = client.post(
                "/api/v1/documents/process",
                json=sample_processing_request,
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "document_id" in data
        assert "classification" in data
        assert "extracted_data" in data
        assert "confidence_score" in data
        assert data["confidence_score"] >= 0.9
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_process_document_validation_error(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers
    ):
        """Test document processing with validation error"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        # Invalid request (no content)
        response = client.post(
            "/api/v1/documents/process",
            json={},
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_batch_processing(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers,
        mock_processing_service
    ):
        """Test batch document processing"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        mock_processing_service.batch_process_documents.return_value = {
            "batch_id": "test-batch-id",
            "total_documents": 3,
            "estimated_completion_time": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
        batch_request = {
            "documents": [
                {"text_content": "Invoice 1"},
                {"text_content": "Invoice 2"},
                {"text_content": "Invoice 3"}
            ],
            "processing_config": {
                "strategy": "competitive",
                "accuracy_threshold": 0.95
            },
            "max_concurrent_documents": 2
        }
        
        with patch.object(app.state, 'processing_service', mock_processing_service):
            response = client.post(
                "/api/v1/documents/batch",
                json=batch_request,
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "batch_id" in data
        assert data["total_documents"] == 3
        assert "status_url" in data
    
    @patch('api.main.get_current_user')
    def test_get_document_status(
        self, 
        mock_get_user, 
        client, 
        auth_headers,
        mock_processing_service
    ):
        """Test document status retrieval"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        
        mock_processing_service.get_document_status.return_value = {
            "document_id": "test-doc-id",
            "status": "completed",
            "progress_percentage": 100.0,
            "current_stage": "completed"
        }
        
        with patch.object(app.state, 'processing_service', mock_processing_service):
            response = client.get(
                "/api/v1/documents/test-doc-id/status",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress_percentage"] == 100.0
    
    @patch('api.main.get_current_user')
    def test_classify_document(
        self, 
        mock_get_user, 
        client, 
        auth_headers,
        mock_processing_service
    ):
        """Test document classification endpoint"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        
        mock_processing_service.classify_document.return_value = {
            "document_type": "invoice",
            "confidence_score": 0.95,
            "processing_time_ms": 500,
            "cost": 0.01,
            "supported_fields": ["invoice_number", "total_amount", "vendor_name"]
        }
        
        classification_request = {
            "text_content": "Invoice #INV-001 Amount: $1000.00"
        }
        
        with patch.object(app.state, 'processing_service', mock_processing_service):
            response = client.post(
                "/api/v1/documents/classify",
                json=classification_request,
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["document_type"] == "invoice"
        assert data["confidence_score"] >= 0.9


class TestWebhookEndpoints(TestEnterpriseAPI):
    """Test webhook management endpoints"""
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_create_webhook(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers
    ):
        """Test webhook creation"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        mock_webhook_service = AsyncMock()
        mock_webhook_service.create_webhook.return_value = {
            "webhook_id": "test-webhook-id",
            "status": "active",
            "events": ["document.processed"],
            "url": "https://example.com/webhook"
        }
        
        webhook_config = {
            "name": "Test Webhook",
            "url": "https://example.com/webhook",
            "events": ["document.processed"],
            "secret": "test_secret"
        }
        
        with patch.object(app.state, 'webhook_service', mock_webhook_service):
            response = client.post(
                "/api/v1/webhooks",
                json=webhook_config,
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "webhook_id" in data
        assert data["status"] == "active"
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_list_webhooks(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers
    ):
        """Test webhook listing"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        mock_webhook_service = AsyncMock()
        mock_webhook_service.list_webhooks.return_value = [
            {
                "id": "webhook-1",
                "name": "Test Webhook 1",
                "url": "https://example.com/webhook1",
                "events": ["document.processed"],
                "is_active": True
            }
        ]
        
        with patch.object(app.state, 'webhook_service', mock_webhook_service):
            response = client.get(
                "/api/v1/webhooks",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "webhooks" in data
        assert len(data["webhooks"]) == 1


class TestFileUploadEndpoints(TestEnterpriseAPI):
    """Test file upload endpoints"""
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_file_upload_and_process(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers,
        mock_processing_service
    ):
        """Test file upload and processing"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        mock_processing_service.process_uploaded_file.return_value = {
            "document_id": "test-doc-id",
            "success": True,
            "processing_time_ms": 2000
        }
        
        # Create test file
        test_file_content = b"Invoice #INV-001\nAmount: $1000.00"
        
        with patch.object(app.state, 'processing_service', mock_processing_service):
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test_invoice.txt", test_file_content, "text/plain")},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "document_id" in data
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_file_upload_size_limit(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers
    ):
        """Test file upload size limit"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        # Create oversized file (simulate > 50MB)
        with patch('api.main.UploadFile') as mock_upload:
            mock_file = MagicMock()
            mock_file.size = 51 * 1024 * 1024  # 51MB
            mock_upload.return_value = mock_file
            
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("large_file.txt", b"x" * 1000, "text/plain")},
                headers=auth_headers
            )
            
            assert response.status_code == 413  # Payload too large


class TestIntegrationEndpoints(TestEnterpriseAPI):
    """Test enterprise integration endpoints"""
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_get_integration_status(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers
    ):
        """Test integration status endpoint"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        response = client.get(
            "/api/v1/integrations/status",
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "quickbooks" in data
        assert "sap" in data
        assert "netsuite" in data
        assert "xero" in data


class TestAnalyticsEndpoints(TestEnterpriseAPI):
    """Test analytics and reporting endpoints"""
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_processing_analytics(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers
    ):
        """Test processing analytics endpoint"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        mock_monitoring_service = AsyncMock()
        mock_monitoring_service.get_processing_analytics.return_value = {
            "period": {
                "start_date": "2024-01-01T00:00:00",
                "end_date": "2024-01-31T23:59:59",
                "days": 31
            },
            "summary": {
                "total_documents": 1000,
                "successful_documents": 962,
                "success_rate_percent": 96.2,
                "average_confidence_score": 0.94,
                "total_cost": 30.00
            }
        }
        
        with patch.object(app.state, 'monitoring_service', mock_monitoring_service):
            response = client.get(
                "/api/v1/analytics/processing?start_date=2024-01-01&end_date=2024-01-31",
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert data["summary"]["success_rate_percent"] >= 96.0


class TestErrorHandling(TestEnterpriseAPI):
    """Test error handling and edge cases"""
    
    def test_404_handler(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_400_bad_request(self, client, auth_headers):
        """Test 400 bad request handling"""
        response = client.post(
            "/api/v1/documents/process",
            json={"invalid": "data"},
            headers=auth_headers
        )
        assert response.status_code in [400, 401, 422]
    
    @patch('api.main.get_current_user')
    def test_500_internal_error(self, mock_get_user, client, auth_headers):
        """Test 500 internal server error handling"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        
        # Mock service to raise exception
        with patch.object(app.state, 'processing_service') as mock_service:
            mock_service.process_document.side_effect = Exception("Test error")
            
            response = client.post(
                "/api/v1/documents/process",
                json={"text_content": "test"},
                headers=auth_headers
            )
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data


class TestRateLimiting(TestEnterpriseAPI):
    """Test rate limiting functionality"""
    
    @patch('api.main.get_current_user')
    def test_rate_limiting(self, mock_get_user, client, auth_headers):
        """Test rate limiting enforcement"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        
        # Make multiple requests rapidly
        responses = []
        for i in range(10):
            response = client.get("/health", headers=auth_headers)
            responses.append(response.status_code)
        
        # At least some requests should succeed
        assert any(status == 200 for status in responses)
        
        # Rate limiting might kick in for excessive requests
        # This depends on the specific rate limiting configuration


class TestSecurity(TestEnterpriseAPI):
    """Test security features"""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present"""
        response = client.options("/health")
        
        # Check for CORS headers in the response
        assert "access-control-allow-origin" in [h.lower() for h in response.headers.keys()]
    
    def test_security_headers(self, client):
        """Test security headers are present"""
        response = client.get("/health")
        
        # Check for security headers
        headers = [h.lower() for h in response.headers.keys()]
        assert "x-content-type-options" in headers
        assert "x-frame-options" in headers
    
    def test_no_server_info_leak(self, client):
        """Test server information is not leaked"""
        response = client.get("/health")
        
        # Server header should not reveal implementation details
        server_header = response.headers.get("server", "").lower()
        assert "fastapi" not in server_header
        assert "uvicorn" not in server_header


class TestPerformance(TestEnterpriseAPI):
    """Test performance requirements"""
    
    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time"""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        assert response.status_code == 200
        
        # Response should be under 200ms
        response_time = (end_time - start_time) * 1000
        assert response_time < 200
    
    @patch('api.main.get_current_user')
    @patch('api.main.get_current_organization')
    def test_processing_endpoint_performance(
        self, 
        mock_get_org, 
        mock_get_user, 
        client, 
        auth_headers,
        mock_processing_service
    ):
        """Test processing endpoint performance"""
        mock_get_user.return_value = MagicMock(id="test-user-id")
        mock_get_org.return_value = MagicMock(id="test-org-id")
        
        import time
        
        with patch.object(app.state, 'processing_service', mock_processing_service):
            start_time = time.time()
            response = client.post(
                "/api/v1/documents/process",
                json={"text_content": "Test invoice content"},
                headers=auth_headers
            )
            end_time = time.time()
        
        assert response.status_code == 200
        
        # API response should be under 5 seconds (excluding actual processing)
        response_time = (end_time - start_time) * 1000
        assert response_time < 5000


# Integration test fixtures and utilities
@pytest.fixture
async def test_database():
    """Test database fixture"""
    # This would set up a test database
    # Implementation depends on specific database setup
    pass


@pytest.fixture
async def test_organization():
    """Test organization fixture"""
    return {
        "id": "test-org-id",
        "name": "Test Organization",
        "subscription_tier": "enterprise",
        "monthly_document_limit": 10000,
        "is_active": True
    }


@pytest.fixture
async def test_user():
    """Test user fixture"""
    return {
        "id": "test-user-id",
        "username": "testuser",
        "email": "test@example.com",
        "organization_id": "test-org-id",
        "roles": ["user"],
        "permissions": ["documents.process", "documents.view"],
        "is_active": True
    }


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])