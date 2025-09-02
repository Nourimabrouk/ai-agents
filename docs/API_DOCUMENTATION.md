# Enterprise Document Processing API Documentation

## Overview

The Enterprise Document Processing API is a production-ready, multi-tenant document processing platform that transforms your document processing system into a scalable, secure enterprise solution. Built on FastAPI with advanced orchestration capabilities, it provides RESTful access to multi-domain document processing with 96.2% accuracy at $0.03 per document.

### Key Features

- **Multi-Domain Processing**: Supports 7+ document types with competitive processing
- **Enterprise Authentication**: JWT, API keys, OAuth 2.0 integration
- **Multi-Tenant Architecture**: Complete organization-based data isolation
- **Real-Time Monitoring**: Comprehensive analytics and performance metrics
- **Enterprise Integrations**: Direct integration with QuickBooks, SAP, NetSuite, Xero
- **Webhook Support**: Async notifications and event-driven architecture
- **Advanced Security**: Rate limiting, audit logging, GDPR compliance

## Quick Start

### 1. Authentication

First, obtain an access token:

```bash
curl -X POST "https://api.yourdomain.com/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "username": "your_username",
       "password": "your_password"
     }'
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 1800,
  "refresh_token": "refresh_token_here",
  "user_info": {
    "user_id": "user_123",
    "username": "your_username",
    "email": "user@company.com",
    "organization_id": "org_123"
  }
}
```

### 2. Process Your First Document

```bash
curl -X POST "https://api.yourdomain.com/api/v1/documents/process" \
     -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "text_content": "INVOICE #INV-001\nACME Corp\nTotal: $1,500.00",
       "processing_config": {
         "strategy": "competitive",
         "accuracy_threshold": 0.95
       }
     }'
```

Response:
```json
{
  "document_id": "doc_abc123",
  "success": true,
  "classification": {
    "document_type": "invoice",
    "confidence": 0.97
  },
  "extracted_data": {
    "invoice_number": "INV-001",
    "vendor_name": "ACME Corp",
    "total_amount": "1500.00"
  },
  "confidence_score": 0.97,
  "processing_time_ms": 1245,
  "cost_breakdown": {
    "total_cost": 0.03
  }
}
```

## API Reference

### Base URL
- Production: `https://api.yourdomain.com`
- Staging: `https://staging-api.yourdomain.com`

### Authentication

All API requests require authentication via:
- **Bearer Token**: `Authorization: Bearer <token>`
- **API Key**: `X-API-Key: <api_key>`

### Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | System health check |
| POST | `/auth/login` | Authenticate user |
| POST | `/auth/refresh` | Refresh access token |
| POST | `/api/v1/documents/process` | Process single document |
| POST | `/api/v1/documents/batch` | Batch process documents |
| GET | `/api/v1/documents/{id}/status` | Get processing status |
| GET | `/api/v1/documents/{id}/result` | Get processing result |
| POST | `/api/v1/documents/classify` | Classify document type |
| POST | `/api/v1/documents/upload` | Upload and process file |
| GET | `/api/v1/metrics` | System metrics |
| GET | `/api/v1/analytics/processing` | Processing analytics |
| POST | `/api/v1/webhooks` | Create webhook |
| GET | `/api/v1/webhooks` | List webhooks |

---

## Document Processing

### Single Document Processing

Process a single document with automatic classification and data extraction.

**Endpoint:** `POST /api/v1/documents/process`

**Request Body:**
```json
{
  "text_content": "Document text content",
  "processing_config": {
    "strategy": "competitive",
    "accuracy_threshold": 0.95,
    "max_processing_time_seconds": 30,
    "max_cost_per_document": 0.05
  },
  "auto_post_to_accounting": false,
  "webhook_url": "https://your-app.com/webhook",
  "metadata": {
    "filename": "invoice.pdf",
    "source": "email"
  }
}
```

**Processing Strategies:**
- `competitive`: Multiple processors compete for best result
- `accuracy_optimized`: Use most accurate processor
- `speed_optimized`: Use fastest processor
- `cost_optimized`: Use cheapest processor first
- `consensus`: Multiple processors with consensus voting

**Response:**
```json
{
  "document_id": "doc_unique_id",
  "success": true,
  "classification": {
    "document_type": "invoice",
    "confidence": 0.96
  },
  "extracted_data": {
    "invoice_number": "INV-2024-001",
    "vendor_name": "Tech Solutions Inc",
    "vendor_address": "123 Business St, City, ST 12345",
    "customer_name": "Your Company",
    "invoice_date": "2024-01-15",
    "due_date": "2024-02-15",
    "total_amount": "2750.00",
    "tax_amount": "250.00",
    "subtotal": "2500.00",
    "line_items": [
      {
        "description": "Professional Services",
        "quantity": 50,
        "unit_price": "50.00",
        "amount": "2500.00"
      }
    ]
  },
  "confidence_score": 0.96,
  "processing_time_ms": 1847,
  "validation_errors": [],
  "cost_breakdown": {
    "classification_cost": 0.01,
    "processing_cost": 0.02,
    "total_cost": 0.03
  },
  "processing_method": "competitive_multi_domain",
  "competitive_results": [
    {
      "processor_name": "specialized_invoice_v2",
      "confidence_score": 0.96,
      "processing_time_ms": 1200,
      "success": true
    }
  ]
}
```

### Document Classification Only

Quickly classify document type without full processing.

**Endpoint:** `POST /api/v1/documents/classify`

**Request:**
```json
{
  "text_content": "INVOICE #12345...",
  "metadata": {
    "filename": "document.pdf"
  }
}
```

**Response:**
```json
{
  "document_type": "invoice",
  "confidence_score": 0.94,
  "processing_time_ms": 245,
  "cost": 0.01,
  "supported_fields": [
    "invoice_number",
    "vendor_name", 
    "total_amount",
    "invoice_date",
    "due_date",
    "line_items"
  ]
}
```

### Batch Processing

Process multiple documents efficiently with parallel processing.

**Endpoint:** `POST /api/v1/documents/batch`

**Request:**
```json
{
  "documents": [
    {
      "text_content": "INVOICE #001...",
      "metadata": {"filename": "invoice1.pdf"}
    },
    {
      "text_content": "RECEIPT from Store...",
      "metadata": {"filename": "receipt1.jpg"}
    }
  ],
  "batch_name": "monthly_processing",
  "max_concurrent_documents": 5,
  "processing_config": {
    "strategy": "competitive",
    "accuracy_threshold": 0.95
  },
  "webhook_url": "https://your-app.com/batch-complete"
}
```

**Response:**
```json
{
  "batch_id": "batch_xyz789",
  "status": "processing",
  "total_documents": 2,
  "estimated_completion_time": "2024-01-15T10:05:00Z",
  "status_url": "/api/v1/documents/batch/batch_xyz789/status"
}
```

### File Upload Processing

Upload and process files directly.

**Endpoint:** `POST /api/v1/documents/upload`

**Request:**
```bash
curl -X POST "https://api.yourdomain.com/api/v1/documents/upload" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@invoice.pdf" \
     -F 'processing_options={"auto_post_to_accounting": true}'
```

**Supported File Types:**
- PDF documents (`.pdf`)
- Images (`.jpg`, `.jpeg`, `.png`, `.tiff`)
- Excel files (`.xlsx`, `.xls`)
- Text files (`.txt`)
- Word documents (`.docx`)

**File Size Limit:** 50MB per file

---

## Status and Results

### Get Processing Status

**Endpoint:** `GET /api/v1/documents/{document_id}/status`

**Response:**
```json
{
  "document_id": "doc_abc123",
  "status": "processing",
  "progress_percentage": 75.0,
  "current_stage": "data_extraction",
  "estimated_completion_time": "2024-01-15T10:02:30Z",
  "error_message": null,
  "partial_results": {
    "classification": {
      "document_type": "invoice",
      "confidence": 0.95
    }
  }
}
```

**Status Values:**
- `pending`: Queued for processing
- `processing`: Currently being processed
- `completed`: Successfully completed
- `failed`: Processing failed
- `cancelled`: Processing was cancelled

### Get Processing Results

**Endpoint:** `GET /api/v1/documents/{document_id}/result`

**Query Parameters:**
- `format`: Response format (`json`, `xml`, `csv`, `excel`)

**Response:** Complete processing results with all extracted data, metadata, and performance metrics.

---

## Enterprise Integrations

### QuickBooks Integration

Automatically post processed invoices and expenses to QuickBooks Online.

**Setup:**
1. Configure QuickBooks credentials in your organization settings
2. Enable auto-posting in processing requests
3. Map document fields to QuickBooks accounts

**Example Request:**
```json
{
  "text_content": "INVOICE #123...",
  "auto_post_to_accounting": true,
  "accounting_integration": "quickbooks",
  "processing_config": {
    "strategy": "accuracy_optimized"
  }
}
```

**Integration Response:**
```json
{
  "integration_result": {
    "success": true,
    "transaction_id": "qb_bill_456",
    "external_reference": "Bill #123",
    "posted_amount": 1500.00,
    "posting_date": "2024-01-15T10:30:00Z"
  }
}
```

### SAP Integration

Direct integration with SAP ERP systems via RFC/REST connections.

**Supported SAP Modules:**
- Financial Accounting (FI)
- Accounts Payable (AP) 
- Accounts Receivable (AR)
- Materials Management (MM)

### NetSuite Integration

RESTlet-based integration for real-time financial data synchronization.

### Xero Integration

Real-time bookkeeping with automatic transaction categorization.

**Integration Status:**
```bash
GET /api/v1/integrations/status
```

**Response:**
```json
{
  "quickbooks": {
    "status": "connected",
    "last_sync": "2024-01-15T10:30:00Z",
    "health": "healthy"
  },
  "sap": {
    "status": "disconnected",
    "last_sync": null,
    "health": "unknown"
  },
  "netsuite": {
    "status": "connected", 
    "last_sync": "2024-01-15T09:15:00Z",
    "health": "healthy"
  },
  "xero": {
    "status": "connected",
    "last_sync": "2024-01-15T11:00:00Z", 
    "health": "healthy"
  }
}
```

---

## Webhooks

Configure webhooks to receive real-time notifications about document processing events.

### Create Webhook

**Endpoint:** `POST /api/v1/webhooks`

**Request:**
```json
{
  "url": "https://your-app.com/webhooks/documents",
  "events": [
    "document.processed",
    "document.failed",
    "batch.completed",
    "threshold.exceeded"
  ],
  "secret": "your-webhook-secret",
  "headers": {
    "X-Custom-Header": "value"
  },
  "retry_attempts": 3,
  "timeout_seconds": 30,
  "active": true
}
```

### Webhook Events

| Event | Description | Payload |
|-------|-------------|---------|
| `document.processed` | Document processing completed | Full processing result |
| `document.failed` | Document processing failed | Error details |
| `batch.completed` | Batch processing finished | Batch summary |
| `threshold.exceeded` | Cost/usage threshold exceeded | Threshold details |
| `integration.error` | External integration error | Error information |

### Webhook Payload Format

```json
{
  "event_type": "document.processed",
  "event_id": "evt_unique_id",
  "timestamp": "2024-01-15T10:30:00Z",
  "organization_id": "org_123",
  "user_id": "user_456",
  "data": {
    "document_id": "doc_abc123",
    "processing_result": {
      // Complete processing result
    }
  }
}
```

### Webhook Security

All webhook payloads include HMAC-SHA256 signatures in the `X-Webhook-Signature` header:

```python
import hmac
import hashlib

def verify_webhook(payload, signature, secret):
    expected = hmac.new(
        secret.encode('utf-8'),
        payload.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

## Monitoring and Analytics

### System Metrics

**Endpoint:** `GET /api/v1/metrics`

**Response:**
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "processing": {
    "total_documents_24h": 1250,
    "successful_documents_24h": 1203,
    "failed_documents_24h": 47,
    "success_rate_percent": 96.24,
    "documents_per_hour": 52.1
  },
  "performance": {
    "avg_response_time_ms": 245.5,
    "p95_response_time_ms": 850.0,
    "p99_response_time_ms": 1500.0,
    "memory_usage_mb": 512.3,
    "cpu_usage_percent": 23.7
  },
  "usage": {
    "active_users_24h": 45,
    "api_calls_24h": 2840,
    "peak_concurrent_users": 12
  },
  "cost": {
    "total_cost_24h": 37.50,
    "cost_per_document": 0.03,
    "estimated_monthly_cost": 1125.00
  }
}
```

### Processing Analytics

**Endpoint:** `GET /api/v1/analytics/processing`

**Query Parameters:**
- `start_date`: Start date (ISO 8601)
- `end_date`: End date (ISO 8601) 
- `document_type`: Filter by document type

**Response:**
```json
{
  "period": {
    "start_date": "2024-01-01T00:00:00Z",
    "end_date": "2024-01-15T23:59:59Z",
    "days": 15
  },
  "summary": {
    "total_documents": 5420,
    "successful_documents": 5214,
    "failed_documents": 206,
    "success_rate_percent": 96.20,
    "average_confidence_score": 0.943,
    "average_processing_time_ms": 1847.2,
    "total_cost": 162.60,
    "cost_per_document": 0.03
  },
  "document_type_breakdown": {
    "invoice": {
      "count": 3250,
      "successful": 3138,
      "avg_confidence": 0.951
    },
    "receipt": {
      "count": 1580,
      "successful": 1523,
      "avg_confidence": 0.932
    },
    "purchase_order": {
      "count": 590,
      "successful": 553,
      "avg_confidence": 0.941
    }
  },
  "daily_volume": [
    {"date": "2024-01-01", "count": 340},
    {"date": "2024-01-02", "count": 387}
  ],
  "performance_trends": {
    "accuracy_trend": [0.94, 0.95, 0.94, 0.96],
    "speed_trend": [1650, 1720, 1580, 1847]
  }
}
```

---

## Error Handling

### Standard Error Response Format

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "Request validation failed",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123",
  "details": {
    "field_errors": [
      {
        "field": "accuracy_threshold",
        "message": "Value must be between 0.0 and 1.0"
      }
    ]
  }
}
```

### Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_FAILED` | 401 | Invalid credentials |
| `TOKEN_EXPIRED` | 401 | Access token has expired |
| `INSUFFICIENT_PERMISSIONS` | 403 | User lacks required permissions |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `DOCUMENT_NOT_FOUND` | 404 | Document does not exist |
| `PROCESSING_FAILED` | 500 | Document processing error |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `QUOTA_EXCEEDED` | 429 | Monthly quota exceeded |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

### Rate Limiting

API requests are subject to rate limits:

- **Global**: 10,000 requests per minute
- **Per User**: 1,000 requests per minute
- **Processing**: 100 documents per minute

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset time (Unix timestamp)

---

## SDK and Code Examples

### Python SDK

```python
from enterprise_doc_api import DocumentProcessor

# Initialize client
client = DocumentProcessor(
    api_key="your-api-key",
    base_url="https://api.yourdomain.com"
)

# Process document
result = client.process_document(
    text="INVOICE #123...",
    strategy="competitive",
    accuracy_threshold=0.95
)

print(f"Document type: {result.classification.document_type}")
print(f"Confidence: {result.confidence_score}")
print(f"Extracted data: {result.extracted_data}")
```

### JavaScript/Node.js

```javascript
const { DocumentProcessor } = require('enterprise-doc-api');

const client = new DocumentProcessor({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.yourdomain.com'
});

async function processDocument() {
  try {
    const result = await client.processDocument({
      textContent: 'INVOICE #123...',
      processingConfig: {
        strategy: 'competitive',
        accuracyThreshold: 0.95
      }
    });
    
    console.log('Document type:', result.classification.documentType);
    console.log('Confidence:', result.confidenceScore);
  } catch (error) {
    console.error('Processing failed:', error.message);
  }
}
```

### cURL Examples

**Process Document:**
```bash
curl -X POST "https://api.yourdomain.com/api/v1/documents/process" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text_content": "INVOICE #123\nVendor: ACME Corp\nTotal: $1,500.00",
    "processing_config": {
      "strategy": "competitive"
    }
  }'
```

**Upload File:**
```bash
curl -X POST "https://api.yourdomain.com/api/v1/documents/upload" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@invoice.pdf" \
  -F 'processing_options={"auto_post_to_accounting": true}'
```

**Get Status:**
```bash
curl -X GET "https://api.yourdomain.com/api/v1/documents/doc_123/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Best Practices

### Performance Optimization

1. **Batch Processing**: Use batch endpoints for multiple documents
2. **Appropriate Strategy**: Choose processing strategy based on needs
3. **Webhook Usage**: Use webhooks for long-running operations
4. **File Size**: Optimize file sizes before upload
5. **Caching**: Cache frequently accessed results

### Security Best Practices

1. **Token Management**: Rotate tokens regularly
2. **API Key Security**: Store API keys securely
3. **Webhook Verification**: Always verify webhook signatures
4. **HTTPS Only**: Use HTTPS for all API calls
5. **Input Validation**: Validate all input data

### Error Handling

1. **Retry Logic**: Implement exponential backoff
2. **Status Monitoring**: Check processing status for long operations
3. **Error Logging**: Log all API errors for troubleshooting
4. **Graceful Degradation**: Handle service unavailability

### Cost Optimization

1. **Strategy Selection**: Use appropriate processing strategies
2. **Confidence Thresholds**: Set reasonable thresholds
3. **Document Pre-filtering**: Filter documents before processing
4. **Monitoring**: Track usage and costs regularly

---

## Support and Resources

### Getting Help

- **Documentation**: [https://docs.yourdomain.com](https://docs.yourdomain.com)
- **API Reference**: [https://api.yourdomain.com/docs](https://api.yourdomain.com/docs)
- **Status Page**: [https://status.yourdomain.com](https://status.yourdomain.com)
- **Support Email**: support@yourdomain.com

### Community

- **GitHub**: [https://github.com/yourdomain/enterprise-doc-api](https://github.com/yourdomain/enterprise-doc-api)
- **Discord**: [https://discord.gg/yourdomain](https://discord.gg/yourdomain)
- **Stack Overflow**: Tag `enterprise-doc-api`

### Enterprise Support

For enterprise customers:
- **Dedicated Support**: Available 24/7
- **Custom Integrations**: Tailored integration development
- **On-Premise Deployment**: Private cloud deployment options
- **Training**: Team training and onboarding sessions

---

*This documentation is regularly updated. For the latest information, visit our [documentation portal](https://docs.yourdomain.com).*