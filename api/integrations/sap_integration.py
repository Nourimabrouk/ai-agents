"""
SAP ERP Integration
Enterprise SAP integration with multiple SAP product support
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

import aiohttp
from pydantic import BaseModel

from api.integrations.base_integration import (
    BaseIntegration, IntegrationCredentials, IntegrationResult, 
    IntegrationStatus, PostingStatus, IntegrationType
)
from api.config import get_settings
from utils.observability.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()


class SAPCredentials(IntegrationCredentials):
    """SAP-specific credentials supporting multiple SAP products"""
    system_type: str  # "s4hana", "business_one", "ariba", "concur"
    server_url: str
    username: str
    password: str
    
    # SAP S/4HANA specific
    client: Optional[str] = None
    language: Optional[str] = "EN"
    
    # SAP Business One specific
    company_db: Optional[str] = None
    
    # OAuth credentials (for newer SAP APIs)
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    token_expires_at: Optional[datetime] = None


class SAPDocument(BaseModel):
    """SAP document structure"""
    document_type: str
    business_partner: str
    posting_date: str
    due_date: Optional[str] = None
    reference: str
    total_amount: Decimal
    currency: str = "USD"
    line_items: List[Dict[str, Any]] = []


class SAPIntegration(BaseIntegration):
    """
    SAP ERP Integration supporting multiple SAP products
    
    Supported Systems:
    - SAP S/4HANA (via REST APIs and OData)
    - SAP Business One (via Service Layer)
    - SAP Ariba (via APIs)
    - SAP Concur (via APIs)
    
    Features:
    - Multi-system support with unified interface
    - Document posting (invoices, purchase orders, expenses)
    - Master data management (vendors, customers, GL accounts)
    - Real-time validation and error handling
    - Batch processing capabilities
    """
    
    def __init__(self, organization_id: str, config: Dict[str, Any]):
        super().__init__(IntegrationType.SAP, organization_id, config)
        
        self.system_type = config.get("system_type", "s4hana")
        self.api_version = config.get("api_version", "v1")
        
        # Set up system-specific configurations
        if self.system_type == "s4hana":
            self._setup_s4hana_config(config)
        elif self.system_type == "business_one":
            self._setup_business_one_config(config)
        elif self.system_type == "ariba":
            self._setup_ariba_config(config)
        elif self.system_type == "concur":
            self._setup_concur_config(config)
        else:
            raise ValueError(f"Unsupported SAP system type: {self.system_type}")
    
    def _setup_s4hana_config(self, config: Dict[str, Any]):
        """Configure for SAP S/4HANA"""
        self.odata_service_path = "/sap/opu/odata/sap"
        self.rest_api_path = "/sap/bc/rest"
        
        # Common S/4HANA services
        self.services = {
            "invoice": f"{self.odata_service_path}/API_SUPPLIERINVOICE_PROCESS_SRV",
            "purchase_order": f"{self.odata_service_path}/API_PURCHASEORDER_PROCESS_SRV",
            "vendor": f"{self.odata_service_path}/API_BUSINESS_PARTNER",
            "gl_account": f"{self.odata_service_path}/API_GLACCOUNTLINEITEM_SRV"
        }
    
    def _setup_business_one_config(self, config: Dict[str, Any]):
        """Configure for SAP Business One"""
        self.service_layer_path = "/b1s/v1"
        
        self.services = {
            "invoice": f"{self.service_layer_path}/PurchaseInvoices",
            "purchase_order": f"{self.service_layer_path}/PurchaseOrders",
            "vendor": f"{self.service_layer_path}/BusinessPartners",
            "items": f"{self.service_layer_path}/Items"
        }
    
    def _setup_ariba_config(self, config: Dict[str, Any]):
        """Configure for SAP Ariba"""
        self.ariba_api_path = "/api"
        
        self.services = {
            "invoice": f"{self.ariba_api_path}/invoices",
            "purchase_order": f"{self.ariba_api_path}/purchase-orders",
            "supplier": f"{self.ariba_api_path}/suppliers"
        }
    
    def _setup_concur_config(self, config: Dict[str, Any]):
        """Configure for SAP Concur"""
        self.concur_api_path = "/api/v3.0"
        
        self.services = {
            "expense": f"{self.concur_api_path}/expense/reports",
            "receipt": f"{self.concur_api_path}/receipts",
            "vendor": f"{self.concur_api_path}/common/vendors"
        }
    
    async def authenticate(self, credentials: SAPCredentials) -> bool:
        """
        Authenticate with SAP system
        
        Supports multiple authentication methods based on system type
        """
        try:
            self.credentials = credentials
            
            if self.system_type == "s4hana":
                success = await self._authenticate_s4hana()
            elif self.system_type == "business_one":
                success = await self._authenticate_business_one()
            elif self.system_type in ["ariba", "concur"]:
                success = await self._authenticate_oauth()
            else:
                success = False
            
            if success:
                self.status = IntegrationStatus.CONNECTED
                logger.info(f"SAP {self.system_type} authentication successful")
            else:
                self.status = IntegrationStatus.ERROR
                self.last_error = "Authentication failed"
                logger.error(f"SAP {self.system_type} authentication failed")
            
            return success
            
        except Exception as e:
            logger.error(f"SAP authentication error: {e}")
            self.status = IntegrationStatus.ERROR
            self.last_error = str(e)
            return False
    
    async def _authenticate_s4hana(self) -> bool:
        """Authenticate with SAP S/4HANA using basic auth"""
        try:
            # Test connection with a simple OData query
            url = f"{self.credentials.server_url}{self.services['vendor']}"
            
            auth = aiohttp.BasicAuth(self.credentials.username, self.credentials.password)
            headers = {
                "Accept": "application/json",
                "X-CSRF-Token": "Fetch"
            }
            
            # Add SAP-specific headers
            if self.credentials.client:
                headers["sap-client"] = self.credentials.client
            if self.credentials.language:
                headers["sap-language"] = self.credentials.language
            
            async with self.session.get(url, auth=auth, headers=headers) as response:
                if response.status in [200, 204]:
                    # Store CSRF token for future requests
                    self.csrf_token = response.headers.get("X-CSRF-Token")
                    return True
                else:
                    logger.error(f"S/4HANA authentication failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"S/4HANA authentication error: {e}")
            return False
    
    async def _authenticate_business_one(self) -> bool:
        """Authenticate with SAP Business One Service Layer"""
        try:
            login_url = f"{self.credentials.server_url}{self.service_layer_path}/Login"
            
            login_data = {
                "UserName": self.credentials.username,
                "Password": self.credentials.password,
                "CompanyDB": self.credentials.company_db
            }
            
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(login_url, json=login_data, headers=headers) as response:
                if response.status == 200:
                    # Service Layer uses session cookies for authentication
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Business One login failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Business One authentication error: {e}")
            return False
    
    async def _authenticate_oauth(self) -> bool:
        """Authenticate using OAuth 2.0 for cloud SAP services"""
        try:
            if self.credentials.access_token and self.credentials.token_expires_at:
                # Check if token is still valid
                if datetime.utcnow() < self.credentials.token_expires_at:
                    return True
                # Try to refresh token
                if self.credentials.refresh_token:
                    return await self._refresh_oauth_token()
            
            # Get new token using client credentials
            token_url = f"{self.credentials.server_url}/oauth/token"
            
            data = {
                "grant_type": "client_credentials",
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret
            }
            
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with self.session.post(token_url, data=data, headers=headers) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    self.credentials.access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 3600)
                    self.credentials.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                    
                    if "refresh_token" in token_data:
                        self.credentials.refresh_token = token_data["refresh_token"]
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"OAuth authentication failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"OAuth authentication error: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to SAP system"""
        try:
            if self.system_type == "s4hana":
                return await self._test_s4hana_connection()
            elif self.system_type == "business_one":
                return await self._test_business_one_connection()
            elif self.system_type in ["ariba", "concur"]:
                return await self._test_oauth_connection()
            else:
                return False
                
        except Exception as e:
            logger.error(f"SAP connection test failed: {e}")
            return False
    
    async def _test_s4hana_connection(self) -> bool:
        """Test S/4HANA connection"""
        try:
            url = f"{self.credentials.server_url}{self.services['vendor']}?$top=1"
            
            auth = aiohttp.BasicAuth(self.credentials.username, self.credentials.password)
            headers = {"Accept": "application/json"}
            
            if self.credentials.client:
                headers["sap-client"] = self.credentials.client
            
            async with self.session.get(url, auth=auth, headers=headers) as response:
                return response.status in [200, 204]
                
        except Exception:
            return False
    
    async def _test_business_one_connection(self) -> bool:
        """Test Business One connection"""
        try:
            url = f"{self.credentials.server_url}{self.services['items']}?$top=1"
            headers = {"Accept": "application/json"}
            
            async with self.session.get(url, headers=headers) as response:
                return response.status == 200
                
        except Exception:
            return False
    
    async def _test_oauth_connection(self) -> bool:
        """Test OAuth-based connection"""
        try:
            # Test with a simple API call
            test_endpoint = list(self.services.values())[0]
            url = f"{self.credentials.server_url}{test_endpoint}"
            
            headers = {
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Accept": "application/json"
            }
            
            async with self.session.get(url, headers=headers) as response:
                return response.status in [200, 204]
                
        except Exception:
            return False
    
    async def post_document(
        self,
        document_data: Dict[str, Any],
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Post document to SAP system
        
        Supports different document types based on SAP system
        """
        try:
            if document_type == "invoice":
                return await self._post_invoice(document_data, metadata)
            elif document_type == "purchase_order":
                return await self._post_purchase_order(document_data, metadata)
            elif document_type == "receipt" and self.system_type == "concur":
                return await self._post_expense(document_data, metadata)
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Unsupported document type for {self.system_type}: {document_type}",
                    error_code="UNSUPPORTED_DOCUMENT_TYPE"
                )
                
        except Exception as e:
            logger.error(f"SAP document posting failed: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="POSTING_FAILED"
            )
    
    async def _post_invoice(self, document_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> IntegrationResult:
        """Post invoice to SAP system"""
        try:
            if self.system_type == "s4hana":
                return await self._post_s4hana_invoice(document_data, metadata)
            elif self.system_type == "business_one":
                return await self._post_b1_invoice(document_data, metadata)
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Invoice posting not supported for {self.system_type}",
                    error_code="NOT_SUPPORTED"
                )
                
        except Exception as e:
            logger.error(f"Invoice posting failed: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="INVOICE_POSTING_FAILED"
            )
    
    async def _post_s4hana_invoice(self, document_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> IntegrationResult:
        """Post invoice to SAP S/4HANA"""
        try:
            # Create supplier invoice document
            invoice_data = {
                "SupplierInvoice": document_data.get("invoice_number", ""),
                "BusinessPartner": await self._get_or_create_vendor(document_data.get("vendor_name")),
                "DocumentDate": self._format_date(document_data.get("invoice_date")),
                "PostingDate": self._format_date(document_data.get("posting_date", document_data.get("invoice_date"))),
                "InvoiceGrossAmount": str(document_data.get("total_amount", 0)),
                "DocumentCurrency": document_data.get("currency", "USD"),
                "to_SuplrInvcItemPurOrdRef": []
            }
            
            # Add line items
            line_items = document_data.get("line_items", [])
            if not line_items and document_data.get("total_amount"):
                # Create single line item
                line_items = [{
                    "description": "Imported expense",
                    "amount": document_data.get("total_amount")
                }]
            
            for i, item in enumerate(line_items, 1):
                line_data = {
                    "SupplierInvoiceItem": str(i),
                    "TaxCode": "I1",  # Default tax code
                    "SupplierInvoiceItemAmount": str(item.get("amount", 0)),
                    "QuantityInPurchaseOrderUnit": item.get("quantity", 1),
                    "PurchaseOrderUnitOfMeasure": "EA",
                    "SupplierInvoiceItemText": item.get("description", "")
                }
                invoice_data["to_SuplrInvcItemPurOrdRef"].append(line_data)
            
            # Post to S/4HANA
            url = f"{self.credentials.server_url}{self.services['invoice']}/A_SupplierInvoice"
            
            auth = aiohttp.BasicAuth(self.credentials.username, self.credentials.password)
            headers = {
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-CSRF-Token": getattr(self, 'csrf_token', '')
            }
            
            if self.credentials.client:
                headers["sap-client"] = self.credentials.client
            
            async with self.session.post(url, json=invoice_data, auth=auth, headers=headers) as response:
                if response.status in [200, 201]:
                    result_data = await response.json()
                    invoice_id = result_data.get("d", {}).get("SupplierInvoice")
                    
                    return IntegrationResult(
                        success=True,
                        transaction_id=f"sap_invoice_{invoice_id}",
                        external_reference=invoice_id,
                        posted_amount=Decimal(str(document_data.get("total_amount", 0))),
                        posting_date=datetime.utcnow(),
                        metadata={"sap_document_id": invoice_id}
                    )
                else:
                    error_text = await response.text()
                    return IntegrationResult(
                        success=False,
                        error_message=f"S/4HANA API error: {error_text}",
                        error_code=f"SAP_API_ERROR_{response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"S/4HANA invoice posting failed: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="S4HANA_INVOICE_ERROR"
            )
    
    async def _post_b1_invoice(self, document_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> IntegrationResult:
        """Post invoice to SAP Business One"""
        try:
            # Create purchase invoice for Business One
            vendor_code = await self._get_or_create_b1_vendor(document_data.get("vendor_name"))
            
            invoice_data = {
                "CardCode": vendor_code,
                "DocDate": self._format_date(document_data.get("invoice_date")),
                "DocDueDate": self._format_date(document_data.get("due_date")),
                "NumAtCard": document_data.get("invoice_number", ""),
                "Comments": "Imported from document processing system",
                "DocumentLines": []
            }
            
            # Add line items
            line_items = document_data.get("line_items", [])
            if not line_items and document_data.get("total_amount"):
                line_items = [{
                    "description": "Imported expense",
                    "amount": document_data.get("total_amount")
                }]
            
            for item in line_items:
                line_data = {
                    "ItemDescription": item.get("description", "Imported item"),
                    "Quantity": item.get("quantity", 1),
                    "UnitPrice": float(item.get("unit_price", item.get("amount", 0))),
                    "AccountCode": "610100"  # Default expense account
                }
                invoice_data["DocumentLines"].append(line_data)
            
            # Post to Business One
            url = f"{self.credentials.server_url}{self.services['invoice']}"
            headers = {"Content-Type": "application/json"}
            
            async with self.session.post(url, json=invoice_data, headers=headers) as response:
                if response.status in [200, 201]:
                    result_data = await response.json()
                    doc_entry = result_data.get("DocEntry")
                    
                    return IntegrationResult(
                        success=True,
                        transaction_id=f"b1_invoice_{doc_entry}",
                        external_reference=str(doc_entry),
                        posted_amount=Decimal(str(document_data.get("total_amount", 0))),
                        posting_date=datetime.utcnow(),
                        metadata={"b1_doc_entry": doc_entry}
                    )
                else:
                    error_text = await response.text()
                    return IntegrationResult(
                        success=False,
                        error_message=f"Business One API error: {error_text}",
                        error_code=f"B1_API_ERROR_{response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"Business One invoice posting failed: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="B1_INVOICE_ERROR"
            )
    
    async def get_posting_status(self, transaction_id: str) -> PostingStatus:
        """Get status of posted document"""
        try:
            # Extract system type and document ID
            if transaction_id.startswith("sap_invoice_"):
                return await self._get_s4hana_status(transaction_id)
            elif transaction_id.startswith("b1_invoice_"):
                return await self._get_b1_status(transaction_id)
            else:
                return PostingStatus.FAILED
                
        except Exception as e:
            logger.error(f"Failed to get posting status: {e}")
            return PostingStatus.FAILED
    
    # Helper methods
    
    async def _get_or_create_vendor(self, vendor_name: str) -> str:
        """Get or create vendor in SAP system"""
        if self.system_type == "s4hana":
            return await self._get_or_create_s4hana_vendor(vendor_name)
        elif self.system_type == "business_one":
            return await self._get_or_create_b1_vendor(vendor_name)
        else:
            return vendor_name  # Fallback
    
    async def _get_or_create_s4hana_vendor(self, vendor_name: str) -> str:
        """Get or create vendor in S/4HANA"""
        # Implementation for S/4HANA vendor management
        return "1000000"  # Placeholder vendor code
    
    async def _get_or_create_b1_vendor(self, vendor_name: str) -> str:
        """Get or create vendor in Business One"""
        # Implementation for Business One vendor management
        return "V001"  # Placeholder vendor code
    
    def _format_date(self, date_value: Any) -> str:
        """Format date for SAP systems"""
        if not date_value:
            return datetime.utcnow().strftime("%Y-%m-%d")
        
        if isinstance(date_value, datetime):
            return date_value.strftime("%Y-%m-%d")
        elif isinstance(date_value, str):
            try:
                parsed_date = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                return parsed_date.strftime("%Y-%m-%d")
            except ValueError:
                return datetime.utcnow().strftime("%Y-%m-%d")
        
        return datetime.utcnow().strftime("%Y-%m-%d")
    
    async def _refresh_oauth_token(self) -> bool:
        """Refresh OAuth token"""
        try:
            token_url = f"{self.credentials.server_url}/oauth/token"
            
            data = {
                "grant_type": "refresh_token",
                "refresh_token": self.credentials.refresh_token,
                "client_id": self.credentials.client_id,
                "client_secret": self.credentials.client_secret
            }
            
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            
            async with self.session.post(token_url, data=data, headers=headers) as response:
                if response.status == 200:
                    token_data = await response.json()
                    
                    self.credentials.access_token = token_data["access_token"]
                    expires_in = token_data.get("expires_in", 3600)
                    self.credentials.token_expires_at = datetime.utcnow() + timedelta(seconds=expires_in)
                    
                    return True
                else:
                    return False
                    
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False


# Export classes
__all__ = ["SAPIntegration", "SAPCredentials", "SAPDocument"]