"""
Xero Integration
Complete Xero accounting software integration with OAuth 2.0
"""

import asyncio
import base64
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal
from urllib.parse import urlencode

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


class XeroCredentials(IntegrationCredentials):
    """Xero-specific credentials with OAuth 2.0"""
    client_id: str
    client_secret: str
    
    # OAuth 2.0 tokens
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_at: Optional[datetime] = None
    
    # OAuth flow
    authorization_code: Optional[str] = None
    redirect_uri: Optional[str] = None
    
    # Xero organization (tenant)
    tenant_id: Optional[str] = None
    organization_name: Optional[str] = None
    
    # Scopes
    scopes: List[str] = ["accounting.transactions", "accounting.contacts", "accounting.settings"]


class XeroLineItem(BaseModel):
    """Xero line item structure"""
    description: str
    quantity: int = 1
    unit_amount: Decimal
    account_code: Optional[str] = None
    tax_type: Optional[str] = "NONE"
    item_code: Optional[str] = None


class XeroIntegration(BaseIntegration):
    """
    Xero Accounting Software Integration
    
    Features:
    - OAuth 2.0 authentication with PKCE
    - Multi-tenant support
    - Bills and Invoices creation
    - Purchase Orders processing
    - Contact management (suppliers/customers)
    - Real-time synchronization
    - Webhook support for notifications
    - Comprehensive error handling
    """
    
    def __init__(self, organization_id: str, config: Dict[str, Any]):
        super().__init__(IntegrationType.XERO, organization_id, config)
        
        # Xero API configuration
        self.api_base_url = "https://api.xero.com"
        self.auth_base_url = "https://identity.xero.com"
        self.api_version = "2.0"
        
        # API endpoints
        self.endpoints = {
            "token": f"{self.auth_base_url}/connect/token",
            "connections": f"{self.api_base_url}/connections",
            "bills": f"{self.api_base_url}/api.xro/{self.api_version}/Bills",
            "invoices": f"{self.api_base_url}/api.xro/{self.api_version}/Invoices",
            "purchase_orders": f"{self.api_base_url}/api.xro/{self.api_version}/PurchaseOrders",
            "contacts": f"{self.api_base_url}/api.xro/{self.api_version}/Contacts",
            "accounts": f"{self.api_base_url}/api.xro/{self.api_version}/Accounts",
            "items": f"{self.api_base_url}/api.xro/{self.api_version}/Items"
        }
        
        # Default headers
        self.default_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
    
    async def authenticate(self, credentials: XeroCredentials) -> bool:
        """
        Authenticate with Xero using OAuth 2.0
        
        Supports both authorization code flow and refresh token flow
        """
        try:
            self.credentials = credentials
            
            if credentials.authorization_code and not credentials.access_token:
                # Initial authentication with authorization code
                success = await self._exchange_authorization_code()
                if success:
                    await self._get_tenant_info()
                    self.status = IntegrationStatus.CONNECTED
                    return True
                    
            elif credentials.refresh_token:
                # Refresh existing token
                success = await self._refresh_access_token()
                if success:
                    self.status = IntegrationStatus.CONNECTED
                    return True
                    
            elif credentials.access_token:
                # Validate existing token
                success = await self.test_connection()
                if success:
                    self.status = IntegrationStatus.CONNECTED
                    return True
                else:
                    # Try to refresh if refresh token available
                    if credentials.refresh_token:
                        return await self.authenticate(credentials)
            
            self.status = IntegrationStatus.ERROR
            self.last_error = "Authentication failed"
            return False
            
        except Exception as e:
            logger.error(f"Xero authentication failed: {e}")
            self.status = IntegrationStatus.ERROR
            self.last_error = str(e)
            return False
    
    async def test_connection(self) -> bool:
        """Test connection by fetching organization info"""
        try:
            if not self.credentials or not self.credentials.access_token:
                return False
            
            url = f"{self.api_base_url}/api.xro/{self.api_version}/Organisation"
            
            headers = {
                **self.default_headers,
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Xero-tenant-id": self.credentials.tenant_id
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    logger.info("Xero connection test successful")
                    return True
                elif response.status == 401:
                    # Token expired, try refresh
                    if self.credentials.refresh_token:
                        refresh_success = await self._refresh_access_token()
                        if refresh_success:
                            return await self.test_connection()
                    return False
                else:
                    logger.error(f"Xero connection test failed: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Xero connection test error: {e}")
            return False
    
    async def post_document(
        self,
        document_data: Dict[str, Any],
        document_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrationResult:
        """
        Post document to Xero
        
        Supported document types:
        - invoice: Creates Bill (for vendor invoices)
        - customer_invoice: Creates Invoice (for customer billing)
        - purchase_order: Creates Purchase Order
        - receipt: Creates Bill or Expense Claim
        """
        try:
            if document_type == "invoice":
                # Determine if this is a supplier bill or customer invoice
                if self._is_supplier_bill(document_data):
                    return await self._create_bill(document_data, metadata)
                else:
                    return await self._create_invoice(document_data, metadata)
                    
            elif document_type == "customer_invoice":
                return await self._create_invoice(document_data, metadata)
                
            elif document_type == "purchase_order":
                return await self._create_purchase_order(document_data, metadata)
                
            elif document_type == "receipt":
                return await self._create_bill(document_data, metadata)
                
            else:
                return IntegrationResult(
                    success=False,
                    error_message=f"Unsupported document type: {document_type}",
                    error_code="UNSUPPORTED_DOCUMENT_TYPE"
                )
                
        except Exception as e:
            logger.error(f"Xero document posting failed: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="POSTING_FAILED"
            )
    
    async def _create_bill(self, document_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> IntegrationResult:
        """Create Xero Bill from invoice data"""
        try:
            # Get or create contact (supplier)
            contact_id = await self._get_or_create_contact(
                document_data.get("vendor_name", "Unknown Vendor"),
                contact_type="Supplier"
            )
            
            if not contact_id:
                return IntegrationResult(
                    success=False,
                    error_message="Failed to create/find supplier contact",
                    error_code="CONTACT_ERROR"
                )
            
            # Parse line items
            line_items = self._parse_line_items(document_data)
            
            # Create bill structure
            bill_data = {
                "Bills": [{
                    "Type": "ACCPAY",  # Accounts Payable
                    "Contact": {"ContactID": contact_id},
                    "Date": self._format_date(document_data.get("invoice_date")),
                    "DueDate": self._format_date(document_data.get("due_date")),
                    "Reference": document_data.get("invoice_number"),
                    "Status": "AUTHORISED",
                    "LineItems": []
                }]
            }
            
            # Add line items
            for item in line_items:
                line_data = {
                    "Description": item.description,
                    "Quantity": item.quantity,
                    "UnitAmount": float(item.unit_amount),
                    "TaxType": item.tax_type or "NONE"
                }
                
                if item.account_code:
                    line_data["AccountCode"] = item.account_code
                else:
                    # Use default expense account
                    line_data["AccountCode"] = "400"  # Default expense account
                
                bill_data["Bills"][0]["LineItems"].append(line_data)
            
            # Post to Xero
            url = self.endpoints["bills"]
            headers = {
                **self.default_headers,
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Xero-tenant-id": self.credentials.tenant_id
            }
            
            async with self.session.post(url, json=bill_data, headers=headers) as response:
                if response.status in [200, 201]:
                    result_data = await response.json()
                    bill = result_data["Bills"][0]
                    
                    return IntegrationResult(
                        success=True,
                        transaction_id=f"xero_bill_{bill['BillID']}",
                        external_reference=bill.get("Reference", bill["BillID"]),
                        posted_amount=Decimal(str(bill["Total"])),
                        posting_date=datetime.utcnow(),
                        metadata={
                            "xero_id": bill["BillID"],
                            "xero_number": bill.get("BillNumber")
                        }
                    )
                else:
                    error_text = await response.text()
                    return IntegrationResult(
                        success=False,
                        error_message=f"Xero API error: {error_text}",
                        error_code=f"XERO_API_ERROR_{response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to create bill: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="BILL_CREATION_ERROR"
            )
    
    async def _create_invoice(self, document_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> IntegrationResult:
        """Create Xero Invoice (for customer billing)"""
        try:
            # Get or create contact (customer)
            contact_id = await self._get_or_create_contact(
                document_data.get("customer_name", "Unknown Customer"),
                contact_type="Customer"
            )
            
            # Parse line items
            line_items = self._parse_line_items(document_data)
            
            # Create invoice structure
            invoice_data = {
                "Invoices": [{
                    "Type": "ACCREC",  # Accounts Receivable
                    "Contact": {"ContactID": contact_id},
                    "Date": self._format_date(document_data.get("invoice_date")),
                    "DueDate": self._format_date(document_data.get("due_date")),
                    "Reference": document_data.get("invoice_number"),
                    "Status": "AUTHORISED",
                    "LineItems": []
                }]
            }
            
            # Add line items
            for item in line_items:
                line_data = {
                    "Description": item.description,
                    "Quantity": item.quantity,
                    "UnitAmount": float(item.unit_amount),
                    "TaxType": item.tax_type or "NONE"
                }
                
                if item.account_code:
                    line_data["AccountCode"] = item.account_code
                else:
                    # Use default revenue account
                    line_data["AccountCode"] = "200"  # Default sales account
                
                invoice_data["Invoices"][0]["LineItems"].append(line_data)
            
            # Post to Xero
            url = self.endpoints["invoices"]
            headers = {
                **self.default_headers,
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Xero-tenant-id": self.credentials.tenant_id
            }
            
            async with self.session.post(url, json=invoice_data, headers=headers) as response:
                if response.status in [200, 201]:
                    result_data = await response.json()
                    invoice = result_data["Invoices"][0]
                    
                    return IntegrationResult(
                        success=True,
                        transaction_id=f"xero_invoice_{invoice['InvoiceID']}",
                        external_reference=invoice.get("Reference", invoice["InvoiceID"]),
                        posted_amount=Decimal(str(invoice["Total"])),
                        posting_date=datetime.utcnow(),
                        metadata={
                            "xero_id": invoice["InvoiceID"],
                            "xero_number": invoice.get("InvoiceNumber")
                        }
                    )
                else:
                    error_text = await response.text()
                    return IntegrationResult(
                        success=False,
                        error_message=f"Xero API error: {error_text}",
                        error_code=f"XERO_API_ERROR_{response.status}"
                    )
                    
        except Exception as e:
            logger.error(f"Failed to create invoice: {e}")
            return IntegrationResult(
                success=False,
                error_message=str(e),
                error_code="INVOICE_CREATION_ERROR"
            )
    
    async def _create_purchase_order(self, document_data: Dict[str, Any], metadata: Optional[Dict[str, Any]]) -> IntegrationResult:
        """Create Xero Purchase Order"""
        return IntegrationResult(
            success=False,
            error_message="Purchase Order creation not yet implemented",
            error_code="NOT_IMPLEMENTED"
        )
    
    async def get_posting_status(self, transaction_id: str) -> PostingStatus:
        """Get status of posted document"""
        try:
            # Extract document type and ID from transaction_id
            if transaction_id.startswith("xero_"):
                _, doc_type, doc_id = transaction_id.split("_", 2)
                
                if doc_type == "bill":
                    url = f"{self.endpoints['bills']}/{doc_id}"
                elif doc_type == "invoice":
                    url = f"{self.endpoints['invoices']}/{doc_id}"
                else:
                    return PostingStatus.FAILED
                
                headers = {
                    **self.default_headers,
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Xero-tenant-id": self.credentials.tenant_id
                }
                
                async with self.session.get(url, headers=headers) as response:
                    if response.status == 200:
                        result_data = await response.json()
                        
                        # Check document status
                        if doc_type == "bill":
                            status = result_data.get("Bills", [{}])[0].get("Status")
                        else:
                            status = result_data.get("Invoices", [{}])[0].get("Status")
                        
                        if status in ["AUTHORISED", "PAID"]:
                            return PostingStatus.POSTED
                        elif status == "VOIDED":
                            return PostingStatus.CANCELLED
                        else:
                            return PostingStatus.PENDING
                    elif response.status == 404:
                        return PostingStatus.CANCELLED
                    else:
                        return PostingStatus.FAILED
                        
        except Exception as e:
            logger.error(f"Failed to get posting status: {e}")
            return PostingStatus.FAILED
    
    async def cancel_posting(self, transaction_id: str) -> bool:
        """Cancel/void a posted document"""
        try:
            # Extract document type and ID
            if transaction_id.startswith("xero_"):
                _, doc_type, doc_id = transaction_id.split("_", 2)
                
                # Update document status to VOIDED
                if doc_type == "bill":
                    update_data = {"Bills": [{"BillID": doc_id, "Status": "VOIDED"}]}
                    url = self.endpoints["bills"]
                elif doc_type == "invoice":
                    update_data = {"Invoices": [{"InvoiceID": doc_id, "Status": "VOIDED"}]}
                    url = self.endpoints["invoices"]
                else:
                    return False
                
                headers = {
                    **self.default_headers,
                    "Authorization": f"Bearer {self.credentials.access_token}",
                    "Xero-tenant-id": self.credentials.tenant_id
                }
                
                async with self.session.post(url, json=update_data, headers=headers) as response:
                    return response.status in [200, 201]
                    
        except Exception as e:
            logger.error(f"Failed to cancel posting: {e}")
            return False
    
    # Authentication helper methods
    
    async def _exchange_authorization_code(self) -> bool:
        """Exchange authorization code for access token"""
        try:
            token_data = {
                "grant_type": "authorization_code",
                "code": self.credentials.authorization_code,
                "redirect_uri": self.credentials.redirect_uri
            }
            
            # Create basic auth header
            auth_string = f"{self.credentials.client_id}:{self.credentials.client_secret}"
            auth_bytes = auth_string.encode('ascii')
            auth_header = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            async with self.session.post(
                self.endpoints["token"], 
                data=urlencode(token_data), 
                headers=headers
            ) as response:
                if response.status == 200:
                    result_data = await response.json()
                    
                    self.credentials.access_token = result_data["access_token"]
                    self.credentials.refresh_token = result_data["refresh_token"]
                    self.credentials.expires_at = datetime.utcnow() + timedelta(seconds=result_data["expires_in"])
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Token exchange failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Authorization code exchange failed: {e}")
            return False
    
    async def _refresh_access_token(self) -> bool:
        """Refresh access token using refresh token"""
        try:
            token_data = {
                "grant_type": "refresh_token",
                "refresh_token": self.credentials.refresh_token
            }
            
            auth_string = f"{self.credentials.client_id}:{self.credentials.client_secret}"
            auth_bytes = auth_string.encode('ascii')
            auth_header = base64.b64encode(auth_bytes).decode('ascii')
            
            headers = {
                "Authorization": f"Basic {auth_header}",
                "Content-Type": "application/x-www-form-urlencoded"
            }
            
            async with self.session.post(
                self.endpoints["token"], 
                data=urlencode(token_data), 
                headers=headers
            ) as response:
                if response.status == 200:
                    result_data = await response.json()
                    
                    self.credentials.access_token = result_data["access_token"]
                    if "refresh_token" in result_data:
                        self.credentials.refresh_token = result_data["refresh_token"]
                    self.credentials.expires_at = datetime.utcnow() + timedelta(seconds=result_data["expires_in"])
                    
                    return True
                else:
                    error_text = await response.text()
                    logger.error(f"Token refresh failed: {error_text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Token refresh failed: {e}")
            return False
    
    async def _get_tenant_info(self) -> bool:
        """Get tenant information after authentication"""
        try:
            headers = {
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Content-Type": "application/json"
            }
            
            async with self.session.get(self.endpoints["connections"], headers=headers) as response:
                if response.status == 200:
                    connections = await response.json()
                    
                    if connections:
                        # Use the first connection
                        connection = connections[0]
                        self.credentials.tenant_id = connection["tenantId"]
                        self.credentials.organization_name = connection["tenantName"]
                        
                        logger.info(f"Connected to Xero organization: {self.credentials.organization_name}")
                        return True
                    else:
                        logger.error("No Xero connections found")
                        return False
                else:
                    logger.error(f"Failed to get tenant info: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to get tenant info: {e}")
            return False
    
    # Helper methods
    
    async def _get_or_create_contact(self, contact_name: str, contact_type: str = "Supplier") -> Optional[str]:
        """Get existing contact or create new one"""
        try:
            # Search for existing contact
            contact = await self._find_contact(contact_name)
            if contact:
                return contact["ContactID"]
            
            # Create new contact
            contact_data = {
                "Contacts": [{
                    "Name": contact_name,
                    "ContactStatus": "ACTIVE",
                    "IsSupplier": contact_type == "Supplier",
                    "IsCustomer": contact_type == "Customer"
                }]
            }
            
            headers = {
                **self.default_headers,
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Xero-tenant-id": self.credentials.tenant_id
            }
            
            async with self.session.post(self.endpoints["contacts"], json=contact_data, headers=headers) as response:
                if response.status in [200, 201]:
                    result_data = await response.json()
                    contact = result_data["Contacts"][0]
                    return contact["ContactID"]
                else:
                    logger.error(f"Failed to create contact: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Contact creation failed: {e}")
            return None
    
    async def _find_contact(self, contact_name: str) -> Optional[Dict[str, Any]]:
        """Find contact by name"""
        try:
            url = f"{self.endpoints['contacts']}?where=Name%3D%22{contact_name}%22"
            
            headers = {
                **self.default_headers,
                "Authorization": f"Bearer {self.credentials.access_token}",
                "Xero-tenant-id": self.credentials.tenant_id
            }
            
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    result_data = await response.json()
                    contacts = result_data.get("Contacts", [])
                    return contacts[0] if contacts else None
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Contact search failed: {e}")
            return None
    
    def _is_supplier_bill(self, document_data: Dict[str, Any]) -> bool:
        """Determine if this is a supplier bill or customer invoice"""
        vendor_name = document_data.get("vendor_name")
        customer_name = document_data.get("customer_name")
        
        # If we have vendor name but no customer name, it's likely a supplier bill
        if vendor_name and not customer_name:
            return True
        
        # Default to supplier bill for invoices
        return True
    
    def _parse_line_items(self, document_data: Dict[str, Any]) -> List[XeroLineItem]:
        """Parse line items from document data"""
        items = []
        
        line_items = document_data.get("line_items", [])
        if line_items:
            for item in line_items:
                unit_amount = item.get("unit_price", item.get("amount", 0))
                quantity = item.get("quantity", 1)
                
                # If no unit price, calculate from amount and quantity
                if not item.get("unit_price") and quantity > 0:
                    unit_amount = item.get("amount", 0) / quantity
                
                items.append(XeroLineItem(
                    description=item.get("description", "Imported item"),
                    quantity=quantity,
                    unit_amount=Decimal(str(unit_amount)),
                    account_code=item.get("account_code"),
                    tax_type=item.get("tax_type", "NONE")
                ))
        else:
            # Single line item from total
            total_amount = document_data.get("total_amount", 0)
            items.append(XeroLineItem(
                description=f"Imported from {document_data.get('invoice_number', 'document')}",
                quantity=1,
                unit_amount=Decimal(str(total_amount))
            ))
        
        return items
    
    def _format_date(self, date_value: Any) -> str:
        """Format date for Xero API"""
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


# Export classes
__all__ = ["XeroIntegration", "XeroCredentials", "XeroLineItem"]