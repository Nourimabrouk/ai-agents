"""
Specialized Document Processors
7+ specialized agents for different document types with domain-specific extraction logic
Extends proven invoice processor patterns for multi-domain intelligence
"""

import asyncio
import logging
import re
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
from pathlib import Path
import statistics

# Base agent framework
from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from agents.accountancy.invoice_processor import BudgetTracker, DocumentExtractor, InvoiceValidator
from agents.intelligence.document_classifier import DocumentType
from agents.intelligence.competitive_processor import CompetitiveProcessingEngine, ProcessingMethod

logger = get_logger(__name__)


# Data models for different document types
@dataclass
class PurchaseOrderData:
    """Purchase Order data structure"""
    po_number: Optional[str] = None
    vendor_name: Optional[str] = None
    vendor_address: Optional[str] = None
    buyer_name: Optional[str] = None
    buyer_address: Optional[str] = None
    
    # Dates
    order_date: Optional[date] = None
    delivery_date: Optional[date] = None
    expected_date: Optional[date] = None
    
    # Financial data
    subtotal: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    total_amount: Optional[Decimal] = None
    currency: str = "USD"
    
    # Line items and terms
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    payment_terms: Optional[str] = None
    delivery_terms: Optional[str] = None
    
    # Quality metrics
    confidence_score: float = 0.0
    extraction_method: str = ""
    validation_errors: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None


@dataclass
class ReceiptData:
    """Receipt data structure"""
    receipt_number: Optional[str] = None
    merchant_name: Optional[str] = None
    merchant_address: Optional[str] = None
    merchant_phone: Optional[str] = None
    
    # Transaction details
    transaction_date: Optional[date] = None
    transaction_time: Optional[str] = None
    transaction_id: Optional[str] = None
    
    # Payment information
    payment_method: Optional[str] = None
    card_last_four: Optional[str] = None
    authorization_code: Optional[str] = None
    
    # Financial data
    subtotal: Optional[Decimal] = None
    tax_amount: Optional[Decimal] = None
    tip_amount: Optional[Decimal] = None
    total_amount: Optional[Decimal] = None
    currency: str = "USD"
    
    # Items and metadata
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    confidence_score: float = 0.0
    extraction_method: str = ""
    validation_errors: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None


@dataclass
class BankStatementData:
    """Bank Statement data structure"""
    account_number: Optional[str] = None
    account_type: Optional[str] = None
    bank_name: Optional[str] = None
    statement_period: Optional[str] = None
    
    # Balances
    beginning_balance: Optional[Decimal] = None
    ending_balance: Optional[Decimal] = None
    
    # Summary data
    total_deposits: Optional[Decimal] = None
    total_withdrawals: Optional[Decimal] = None
    total_fees: Optional[Decimal] = None
    
    # Transaction data
    transactions: List[Dict[str, Any]] = field(default_factory=list)
    transaction_count: int = 0
    
    # Quality metrics
    confidence_score: float = 0.0
    extraction_method: str = ""
    validation_errors: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None


@dataclass
class ContractData:
    """Contract data structure"""
    contract_title: Optional[str] = None
    parties: List[str] = field(default_factory=list)
    effective_date: Optional[date] = None
    termination_date: Optional[date] = None
    renewal_date: Optional[date] = None
    
    # Contract terms
    key_terms: List[str] = field(default_factory=list)
    payment_terms: Optional[str] = None
    termination_conditions: List[str] = field(default_factory=list)
    
    # Financial data
    contract_value: Optional[Decimal] = None
    payment_schedule: List[Dict[str, Any]] = field(default_factory=list)
    
    # Legal information
    governing_law: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    # Quality metrics
    confidence_score: float = 0.0
    extraction_method: str = ""
    validation_errors: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None


@dataclass
class FinancialStatementData:
    """Financial Statement data structure"""
    statement_type: Optional[str] = None  # income_statement, balance_sheet, cash_flow
    company_name: Optional[str] = None
    period_ending: Optional[date] = None
    period_type: Optional[str] = None  # quarterly, annual, monthly
    
    # Income Statement data
    total_revenue: Optional[Decimal] = None
    gross_profit: Optional[Decimal] = None
    operating_income: Optional[Decimal] = None
    net_income: Optional[Decimal] = None
    
    # Balance Sheet data
    total_assets: Optional[Decimal] = None
    total_liabilities: Optional[Decimal] = None
    shareholders_equity: Optional[Decimal] = None
    
    # Cash Flow data
    operating_cash_flow: Optional[Decimal] = None
    investing_cash_flow: Optional[Decimal] = None
    financing_cash_flow: Optional[Decimal] = None
    
    # Line items
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    confidence_score: float = 0.0
    extraction_method: str = ""
    validation_errors: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None


@dataclass
class LegalDocumentData:
    """Legal Document data structure"""
    document_type: Optional[str] = None  # complaint, motion, order, etc.
    case_number: Optional[str] = None
    court: Optional[str] = None
    jurisdiction: Optional[str] = None
    
    # Parties
    plaintiff: Optional[str] = None
    defendant: Optional[str] = None
    attorneys: List[Dict[str, str]] = field(default_factory=list)
    
    # Dates
    filing_date: Optional[date] = None
    hearing_date: Optional[date] = None
    deadline_dates: List[Dict[str, Any]] = field(default_factory=list)
    
    # Document content
    claims: List[str] = field(default_factory=list)
    relief_sought: List[str] = field(default_factory=list)
    key_facts: List[str] = field(default_factory=list)
    
    # Quality metrics
    confidence_score: float = 0.0
    extraction_method: str = ""
    validation_errors: List[str] = field(default_factory=list)
    processed_at: datetime = field(default_factory=datetime.now)
    source_file: Optional[str] = None


# Base class for specialized processors
class BaseSpecializedProcessor(BaseAgent):
    """Base class for specialized document processors"""
    
    def __init__(
        self,
        name: str,
        document_type: DocumentType,
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, api_key, config=config)
        
        self.document_type = document_type
        self.budget_tracker = BudgetTracker()
        self.document_extractor = DocumentExtractor()
        self.competitive_engine = CompetitiveProcessingEngine(self.budget_tracker)
        
        # Performance metrics specific to this processor
        self.processed_documents = 0
        self.successful_extractions = 0
        self.accuracy_threshold = 0.95
        
        logger.info(f"Initialized specialized processor: {self.name} for {document_type.value}")
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute specialized document processing"""
        try:
            if isinstance(task, str) and Path(task).exists():
                return await self.process_document_file(task)
            elif isinstance(task, dict) and 'file_path' in task:
                return await self.process_document_file(task['file_path'])
            elif isinstance(task, dict) and 'text' in task:
                return await self.process_document_text(task['text'])
            else:
                return await self.process_document_text(str(task))
                
        except Exception as e:
            logger.error(f"Specialized processing failed for {self.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_data": None
            }
    
    async def process_document_file(self, file_path: str) -> Dict[str, Any]:
        """Process document from file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            logger.info(f"Processing {self.document_type.value} file: {file_path.name}")
            
            # Extract text based on file type
            text = ""
            confidence = 0.0
            
            if file_path.suffix.lower() in ['.pdf']:
                text, confidence = await self.document_extractor.extract_from_pdf(str(file_path))
            elif file_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff']:
                text, confidence = await self.document_extractor.extract_from_image(str(file_path))
            elif file_path.suffix.lower() in ['.xlsx', '.xls', '.csv']:
                text, confidence = await self.document_extractor.extract_from_excel(str(file_path))
            elif file_path.suffix.lower() in ['.txt']:
                text = file_path.read_text(encoding='utf-8', errors='ignore')
                confidence = 1.0
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
            
            if not text or confidence < 0.1:
                raise ValueError("Failed to extract readable text from document")
            
            # Process with specialized extraction
            result = await self.process_document_text(text)
            if result['success'] and result.get('document_data'):
                result['document_data']['source_file'] = str(file_path)
            
            return result
            
        except Exception as e:
            logger.error(f"File processing error for {self.name}: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_data": None
            }
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process document from text - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement process_document_text")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get processor-specific performance metrics"""
        overall_accuracy = (
            self.successful_extractions / max(1, self.processed_documents)
        )
        
        return {
            "processor_name": self.name,
            "document_type": self.document_type.value,
            "processed_documents": self.processed_documents,
            "successful_extractions": self.successful_extractions,
            "overall_accuracy": overall_accuracy,
            "target_accuracy": self.accuracy_threshold,
            "meets_target": overall_accuracy >= self.accuracy_threshold,
            "competitive_engine_performance": self.competitive_engine.get_method_performance()
        }


# Specialized Processors

class PurchaseOrderProcessor(BaseSpecializedProcessor):
    """Specialized processor for Purchase Orders"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="purchase_order_processor",
            document_type=DocumentType.PURCHASE_ORDER,
            api_key=api_key,
            config=config
        )
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process purchase order text"""
        try:
            logger.info("Processing purchase order text")
            
            # Use competitive processing engine
            extraction_result = await self.competitive_engine.get_best_result(
                text, self.document_type
            )
            
            # Convert to PurchaseOrderData
            po_data = await self._convert_to_po_data(extraction_result.extracted_data, text)
            po_data.extraction_method = extraction_result.method.value
            po_data.confidence_score = extraction_result.confidence_score
            
            # Validate data
            po_data = await self._validate_po_data(po_data)
            
            # Update metrics
            self.processed_documents += 1
            if po_data.confidence_score >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            result = {
                "success": True,
                "document_data": self._po_data_to_dict(po_data),
                "accuracy": po_data.confidence_score,
                "extraction_method": po_data.extraction_method,
                "validation_errors": po_data.validation_errors,
                "processing_time": extraction_result.processing_time,
                "cost_estimate": extraction_result.cost_estimate
            }
            
            logger.info(f"Purchase order processed, accuracy: {po_data.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Purchase order processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_data": None
            }
    
    async def _convert_to_po_data(self, extracted_data: Dict[str, Any], text: str) -> PurchaseOrderData:
        """Convert extracted data to PurchaseOrderData"""
        po_data = PurchaseOrderData()
        
        # Map extracted fields
        po_data.po_number = extracted_data.get('po_number')
        po_data.vendor_name = extracted_data.get('vendor_name')
        po_data.buyer_name = extracted_data.get('buyer_name')
        
        # Parse dates
        if extracted_data.get('order_date'):
            po_data.order_date = self._parse_date(extracted_data['order_date'])
        if extracted_data.get('delivery_date'):
            po_data.delivery_date = self._parse_date(extracted_data['delivery_date'])
        
        # Parse amounts
        if extracted_data.get('total_amount'):
            try:
                po_data.total_amount = Decimal(str(extracted_data['total_amount']))
            except (InvalidOperation, TypeError):
                pass
        
        # Extract line items if available
        if extracted_data.get('line_items'):
            po_data.line_items = extracted_data['line_items']
        else:
            # Try to extract line items from text
            po_data.line_items = await self._extract_line_items(text)
        
        return po_data
    
    async def _extract_line_items(self, text: str) -> List[Dict[str, Any]]:
        """Extract line items from purchase order text"""
        line_items = []
        lines = text.split('\n')
        
        for line in lines:
            # Look for lines with quantity, description, and price
            match = re.search(r'(\d+)\s+(.+?)\s+\$?([\d,]+(?:\.\d{2})?)', line)
            if match:
                try:
                    quantity = int(match.group(1))
                    description = match.group(2).strip()
                    unit_price = float(match.group(3).replace(',', ''))
                    
                    line_items.append({
                        'quantity': quantity,
                        'description': description,
                        'unit_price': unit_price,
                        'line_total': quantity * unit_price
                    })
                except ValueError:
                    continue
        
        return line_items
    
    async def _validate_po_data(self, po_data: PurchaseOrderData) -> PurchaseOrderData:
        """Validate purchase order data"""
        errors = []
        
        # Required field validation
        if not po_data.po_number:
            errors.append("PO number is required")
        
        if not po_data.vendor_name:
            errors.append("Vendor name is required")
        
        # Business logic validation
        if po_data.delivery_date and po_data.order_date:
            if po_data.delivery_date < po_data.order_date:
                errors.append("Delivery date cannot be before order date")
        
        # Line item validation
        if po_data.line_items and po_data.total_amount:
            calculated_total = sum(item.get('line_total', 0) for item in po_data.line_items)
            if abs(calculated_total - float(po_data.total_amount)) > 0.01:
                errors.append(f"Line item total mismatch: {calculated_total} vs {po_data.total_amount}")
        
        po_data.validation_errors = errors
        
        # Adjust confidence based on validation
        if errors:
            penalty = min(0.3, len(errors) * 0.1)
            po_data.confidence_score = max(0.0, po_data.confidence_score - penalty)
        
        return po_data
    
    def _po_data_to_dict(self, po_data: PurchaseOrderData) -> Dict[str, Any]:
        """Convert PurchaseOrderData to dictionary"""
        return {
            'po_number': po_data.po_number,
            'vendor_name': po_data.vendor_name,
            'buyer_name': po_data.buyer_name,
            'order_date': po_data.order_date.isoformat() if po_data.order_date else None,
            'delivery_date': po_data.delivery_date.isoformat() if po_data.delivery_date else None,
            'total_amount': str(po_data.total_amount) if po_data.total_amount else None,
            'line_items': po_data.line_items,
            'payment_terms': po_data.payment_terms,
            'delivery_terms': po_data.delivery_terms,
            'confidence_score': po_data.confidence_score,
            'extraction_method': po_data.extraction_method,
            'validation_errors': po_data.validation_errors,
            'processed_at': po_data.processed_at.isoformat()
        }
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string"""
        date_formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt).date()
            except ValueError:
                continue
        
        return None


class ReceiptProcessor(BaseSpecializedProcessor):
    """Specialized processor for Receipts"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="receipt_processor",
            document_type=DocumentType.RECEIPT,
            api_key=api_key,
            config=config
        )
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process receipt text"""
        try:
            logger.info("Processing receipt text")
            
            # Use competitive processing
            extraction_result = await self.competitive_engine.get_best_result(
                text, self.document_type
            )
            
            # Convert to ReceiptData
            receipt_data = await self._convert_to_receipt_data(extraction_result.extracted_data, text)
            receipt_data.extraction_method = extraction_result.method.value
            receipt_data.confidence_score = extraction_result.confidence_score
            
            # Validate data
            receipt_data = await self._validate_receipt_data(receipt_data)
            
            # Update metrics
            self.processed_documents += 1
            if receipt_data.confidence_score >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            result = {
                "success": True,
                "document_data": self._receipt_data_to_dict(receipt_data),
                "accuracy": receipt_data.confidence_score,
                "extraction_method": receipt_data.extraction_method,
                "validation_errors": receipt_data.validation_errors
            }
            
            logger.info(f"Receipt processed, accuracy: {receipt_data.confidence_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Receipt processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "document_data": None
            }
    
    async def _convert_to_receipt_data(self, extracted_data: Dict[str, Any], text: str) -> ReceiptData:
        """Convert extracted data to ReceiptData"""
        receipt_data = ReceiptData()
        
        # Map fields
        receipt_data.merchant_name = extracted_data.get('merchant_name')
        receipt_data.payment_method = extracted_data.get('payment_method')
        
        # Parse transaction date
        if extracted_data.get('transaction_date'):
            receipt_data.transaction_date = self._parse_date(extracted_data['transaction_date'])
        
        # Parse amount
        if extracted_data.get('transaction_amount'):
            try:
                receipt_data.total_amount = Decimal(str(extracted_data['transaction_amount']))
            except (InvalidOperation, TypeError):
                pass
        
        return receipt_data
    
    async def _validate_receipt_data(self, receipt_data: ReceiptData) -> ReceiptData:
        """Validate receipt data"""
        errors = []
        
        if not receipt_data.merchant_name:
            errors.append("Merchant name is required")
        
        if not receipt_data.total_amount or receipt_data.total_amount <= 0:
            errors.append("Valid transaction amount is required")
        
        receipt_data.validation_errors = errors
        
        if errors:
            penalty = min(0.3, len(errors) * 0.1)
            receipt_data.confidence_score = max(0.0, receipt_data.confidence_score - penalty)
        
        return receipt_data
    
    def _receipt_data_to_dict(self, receipt_data: ReceiptData) -> Dict[str, Any]:
        """Convert ReceiptData to dictionary"""
        return {
            'merchant_name': receipt_data.merchant_name,
            'transaction_date': receipt_data.transaction_date.isoformat() if receipt_data.transaction_date else None,
            'total_amount': str(receipt_data.total_amount) if receipt_data.total_amount else None,
            'payment_method': receipt_data.payment_method,
            'confidence_score': receipt_data.confidence_score,
            'extraction_method': receipt_data.extraction_method,
            'validation_errors': receipt_data.validation_errors,
            'processed_at': receipt_data.processed_at.isoformat()
        }
    
    def _parse_date(self, date_str: str) -> Optional[date]:
        """Parse date string"""
        date_formats = [
            "%m/%d/%Y", "%m-%d-%Y", "%Y-%m-%d", "%d/%m/%Y"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(str(date_str), fmt).date()
            except ValueError:
                continue
        
        return None


# Additional processors would follow similar patterns...
# For brevity, I'll create simplified versions of the remaining processors

class BankStatementProcessor(BaseSpecializedProcessor):
    """Specialized processor for Bank Statements"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="bank_statement_processor",
            document_type=DocumentType.BANK_STATEMENT,
            api_key=api_key,
            config=config
        )
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process bank statement text"""
        try:
            extraction_result = await self.competitive_engine.get_best_result(text, self.document_type)
            
            # Simplified processing for demonstration
            statement_data = {
                'account_number': extraction_result.extracted_data.get('account_number'),
                'statement_period': extraction_result.extracted_data.get('statement_period'),
                'beginning_balance': extraction_result.extracted_data.get('beginning_balance'),
                'ending_balance': extraction_result.extracted_data.get('ending_balance'),
                'transactions': extraction_result.extracted_data.get('transactions', []),
                'confidence_score': extraction_result.confidence_score,
                'extraction_method': extraction_result.method.value
            }
            
            self.processed_documents += 1
            if extraction_result.confidence_score >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            return {
                "success": True,
                "document_data": statement_data,
                "accuracy": extraction_result.confidence_score
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class ContractProcessor(BaseSpecializedProcessor):
    """Specialized processor for Contracts"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="contract_processor",
            document_type=DocumentType.CONTRACT,
            api_key=api_key,
            config=config
        )
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process contract text"""
        try:
            extraction_result = await self.competitive_engine.get_best_result(text, self.document_type)
            
            contract_data = {
                'parties': extraction_result.extracted_data.get('parties', []),
                'effective_date': extraction_result.extracted_data.get('effective_date'),
                'termination_date': extraction_result.extracted_data.get('termination_date'),
                'key_terms': extraction_result.extracted_data.get('key_terms', []),
                'confidence_score': extraction_result.confidence_score,
                'extraction_method': extraction_result.method.value
            }
            
            self.processed_documents += 1
            if extraction_result.confidence_score >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            return {
                "success": True,
                "document_data": contract_data,
                "accuracy": extraction_result.confidence_score
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class FinancialStatementProcessor(BaseSpecializedProcessor):
    """Specialized processor for Financial Statements"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="financial_statement_processor",
            document_type=DocumentType.FINANCIAL_STATEMENT,
            api_key=api_key,
            config=config
        )
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process financial statement text"""
        try:
            extraction_result = await self.competitive_engine.get_best_result(text, self.document_type)
            
            financial_data = {
                'statement_type': extraction_result.extracted_data.get('statement_type'),
                'period_ending': extraction_result.extracted_data.get('period_ending'),
                'total_revenue': extraction_result.extracted_data.get('total_revenue'),
                'net_income': extraction_result.extracted_data.get('net_income'),
                'total_assets': extraction_result.extracted_data.get('total_assets'),
                'confidence_score': extraction_result.confidence_score,
                'extraction_method': extraction_result.method.value
            }
            
            self.processed_documents += 1
            if extraction_result.confidence_score >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            return {
                "success": True,
                "document_data": financial_data,
                "accuracy": extraction_result.confidence_score
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class LegalDocumentProcessor(BaseSpecializedProcessor):
    """Specialized processor for Legal Documents"""
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="legal_document_processor",
            document_type=DocumentType.LEGAL_DOCUMENT,
            api_key=api_key,
            config=config
        )
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process legal document text"""
        try:
            extraction_result = await self.competitive_engine.get_best_result(text, self.document_type)
            
            legal_data = {
                'case_number': extraction_result.extracted_data.get('case_number'),
                'plaintiff': extraction_result.extracted_data.get('plaintiff'),
                'defendant': extraction_result.extracted_data.get('defendant'),
                'court': extraction_result.extracted_data.get('court'),
                'filing_date': extraction_result.extracted_data.get('filing_date'),
                'confidence_score': extraction_result.confidence_score,
                'extraction_method': extraction_result.method.value
            }
            
            self.processed_documents += 1
            if extraction_result.confidence_score >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            return {
                "success": True,
                "document_data": legal_data,
                "accuracy": extraction_result.confidence_score
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


class CustomDocumentProcessor(BaseSpecializedProcessor):
    """Flexible processor for custom document types"""
    
    def __init__(self, custom_type: str, extraction_rules: Dict[str, Any], 
                 api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name=f"custom_{custom_type}_processor",
            document_type=DocumentType.CUSTOM,
            api_key=api_key,
            config=config
        )
        
        self.custom_type = custom_type
        self.extraction_rules = extraction_rules
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process custom document text"""
        try:
            # Apply custom extraction rules
            extracted_data = {}
            
            for field, rules in self.extraction_rules.items():
                if isinstance(rules, list):  # Regex patterns
                    for pattern in rules:
                        match = re.search(pattern, text, re.IGNORECASE)
                        if match:
                            extracted_data[field] = match.group(1) if match.groups() else match.group(0)
                            break
                elif isinstance(rules, str):  # Single pattern
                    match = re.search(rules, text, re.IGNORECASE)
                    if match:
                        extracted_data[field] = match.group(1) if match.groups() else match.group(0)
            
            # Calculate confidence based on fields found
            confidence = len(extracted_data) / max(1, len(self.extraction_rules))
            
            custom_data = {
                'custom_type': self.custom_type,
                'extracted_data': extracted_data,
                'confidence_score': confidence,
                'extraction_method': 'custom_rules'
            }
            
            self.processed_documents += 1
            if confidence >= self.accuracy_threshold:
                self.successful_extractions += 1
            
            return {
                "success": True,
                "document_data": custom_data,
                "accuracy": confidence
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}


# Factory functions for easy instantiation
def create_purchase_order_processor(api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> PurchaseOrderProcessor:
    """Create purchase order processor"""
    return PurchaseOrderProcessor(api_key, config)


def create_receipt_processor(api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> ReceiptProcessor:
    """Create receipt processor"""
    return ReceiptProcessor(api_key, config)


def create_bank_statement_processor(api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> BankStatementProcessor:
    """Create bank statement processor"""
    return BankStatementProcessor(api_key, config)


def create_contract_processor(api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> ContractProcessor:
    """Create contract processor"""
    return ContractProcessor(api_key, config)


def create_financial_statement_processor(api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> FinancialStatementProcessor:
    """Create financial statement processor"""
    return FinancialStatementProcessor(api_key, config)


def create_legal_document_processor(api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> LegalDocumentProcessor:
    """Create legal document processor"""
    return LegalDocumentProcessor(api_key, config)


def create_custom_processor(custom_type: str, extraction_rules: Dict[str, Any], 
                           api_key: Optional[str] = None, config: Optional[Dict[str, Any]] = None) -> CustomDocumentProcessor:
    """Create custom document processor"""
    return CustomDocumentProcessor(custom_type, extraction_rules, api_key, config)


# Example usage
async def main():
    """Example usage of specialized processors"""
    
    # Test purchase order processor
    po_processor = create_purchase_order_processor()
    
    po_text = """
    PURCHASE ORDER #PO-2024-001
    Date: January 15, 2024
    
    Vendor: ACME Supplies Inc.
    Ship To: XYZ Corporation
    
    Delivery Date: February 1, 2024
    
    Item                   Qty    Price    Total
    Office Chairs          5      $150.00  $750.00
    Desk Lamps            10      $25.00   $250.00
    
    Total: $1,000.00
    """
    
    result = await po_processor.process_document_text(po_text)
    print(f"Purchase Order Result: {json.dumps(result, indent=2)}")
    
    # Test receipt processor
    receipt_processor = create_receipt_processor()
    
    receipt_text = """
    Target Store #1234
    123 Shopping Center
    
    Transaction Date: 01/15/2024
    Transaction Time: 14:32
    
    Items:
    Groceries              $45.67
    Household Items        $23.45
    
    Subtotal:             $69.12
    Tax:                  $5.53
    Total:                $74.65
    
    VISA ****1234
    Auth: 123456
    """
    
    result = await receipt_processor.process_document_text(receipt_text)
    print(f"Receipt Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())