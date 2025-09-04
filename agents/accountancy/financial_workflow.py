"""
Financial Workflow Automation: Phase 6 - Complete Financial Processing Pipeline
Features:
- Complete invoice-to-journal-entry pipeline
- Bank statement reconciliation
- Expense categorization with learning
- Anomaly detection in financial data
"""

import asyncio
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
from pathlib import Path
from collections import defaultdict
import re
import hashlib
from decimal import Decimal, ROUND_HALF_UP
from abc import ABC, abstractmethod

from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class TransactionType(Enum):
    """Types of financial transactions"""
    INCOME = "income"
    EXPENSE = "expense"
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    TRANSFER = "transfer"


class DocumentType(Enum):
    """Types of financial documents"""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    BANK_STATEMENT = "bank_statement"
    JOURNAL_ENTRY = "journal_entry"
    PURCHASE_ORDER = "purchase_order"
    CREDIT_NOTE = "credit_note"


class ProcessingStatus(Enum):
    """Status of document processing"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_REVIEW = "requires_review"


@dataclass
class FinancialDocument:
    """Base financial document"""
    document_id: str
    document_type: DocumentType
    raw_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    processed_at: Optional[datetime] = None
    anomalies: List[str] = field(default_factory=list)


@dataclass
class Invoice(FinancialDocument):
    """Invoice document with specific fields"""
    invoice_number: str = ""
    vendor_name: str = ""
    vendor_address: str = ""
    invoice_date: Optional[datetime] = None
    due_date: Optional[datetime] = None
    total_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    tax_amount: Decimal = field(default_factory=lambda: Decimal('0'))
    line_items: List[Dict[str, Any]] = field(default_factory=list)
    payment_terms: str = ""
    
    def __post_init__(self):
        if not self.document_type:
            self.document_type = DocumentType.INVOICE


@dataclass
class BankTransaction:
    """Bank transaction record"""
    transaction_id: str
    date: datetime
    description: str
    amount: Decimal
    balance: Decimal
    transaction_type: str  # debit/credit
    reference: str = ""
    category: str = ""
    matched_document_id: Optional[str] = None
    confidence: float = 0.0


@dataclass
class JournalEntry:
    """Double-entry journal entry"""
    entry_id: str
    date: datetime
    description: str
    reference: str
    debits: List[Tuple[str, Decimal]]  # (account, amount)
    credits: List[Tuple[str, Decimal]]  # (account, amount)
    source_document_id: Optional[str] = None
    created_by: str = "system"
    
    @property
    def total_debits(self) -> Decimal:
        return sum(amount for _, amount in self.debits)
    
    @property
    def total_credits(self) -> Decimal:
        return sum(amount for _, amount in self.credits)
    
    @property
    def is_balanced(self) -> bool:
        return self.total_debits == self.total_credits


class DocumentExtractor:
    """Extracts structured data from financial documents"""
    
    def __init__(self):
        self.extraction_patterns = {
            DocumentType.INVOICE: {
                'invoice_number': [
                    r'invoice\s*(?:number|#)?\s*:?\s*([A-Z0-9-]+)',
                    r'inv\s*(?:number|#)?\s*:?\s*([A-Z0-9-]+)',
                    r'bill\s*(?:number|#)?\s*:?\s*([A-Z0-9-]+)'
                ],
                'total_amount': [
                    r'total\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                    r'amount\s*due\s*:?\s*\$?([0-9,]+\.?[0-9]*)',
                    r'balance\s*:?\s*\$?([0-9,]+\.?[0-9]*)'
                ],
                'date': [
                    r'date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'invoice\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    r'bill\s*date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})'
                ],
                'vendor': [
                    r'from\s*:?\s*([^\n]+)',
                    r'vendor\s*:?\s*([^\n]+)',
                    r'supplier\s*:?\s*([^\n]+)'
                ]
            }
        }
        
        self.category_keywords = {
            'office_supplies': ['paper', 'pen', 'stapler', 'supplies', 'stationery'],
            'software': ['software', 'license', 'subscription', 'saas', 'cloud'],
            'travel': ['hotel', 'flight', 'travel', 'accommodation', 'transportation'],
            'utilities': ['electricity', 'gas', 'water', 'internet', 'phone'],
            'marketing': ['advertising', 'marketing', 'promotion', 'branding'],
            'consulting': ['consulting', 'advisory', 'professional services'],
            'equipment': ['computer', 'laptop', 'printer', 'equipment', 'hardware'],
            'rent': ['rent', 'lease', 'office space', 'facility']
        }
    
    async def extract_invoice_data(self, document: FinancialDocument) -> Invoice:
        """Extract structured data from invoice"""
        invoice = Invoice(
            document_id=document.document_id,
            document_type=DocumentType.INVOICE,
            raw_text=document.raw_text,
            metadata=document.metadata
        )
        
        text = document.raw_text.lower()
        
        try:
            # Extract invoice number
            for pattern in self.extraction_patterns[DocumentType.INVOICE]['invoice_number']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    invoice.invoice_number = match.group(1).upper()
                    break
            
            # Extract total amount
            for pattern in self.extraction_patterns[DocumentType.INVOICE]['total_amount']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    amount_str = match.group(1).replace(',', '')
                    invoice.total_amount = Decimal(amount_str)
                    break
            
            # Extract date
            for pattern in self.extraction_patterns[DocumentType.INVOICE]['date']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    date_str = match.group(1)
                    invoice.invoice_date = self._parse_date(date_str)
                    break
            
            # Extract vendor name
            for pattern in self.extraction_patterns[DocumentType.INVOICE]['vendor']:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    invoice.vendor_name = match.group(1).strip()
                    break
            
            # Calculate confidence score
            confidence_factors = [
                1.0 if invoice.invoice_number else 0.0,
                1.0 if invoice.total_amount > 0 else 0.0,
                1.0 if invoice.invoice_date else 0.0,
                1.0 if invoice.vendor_name else 0.0
            ]
            
            invoice.confidence_score = sum(confidence_factors) / len(confidence_factors)
            
            # Store extracted data
            invoice.extracted_data = {
                'invoice_number': invoice.invoice_number,
                'vendor_name': invoice.vendor_name,
                'total_amount': str(invoice.total_amount),
                'invoice_date': invoice.invoice_date.isoformat() if invoice.invoice_date else None,
                'confidence_score': invoice.confidence_score
            }
            
            logger.info(f"Extracted invoice data with confidence {invoice.confidence_score:.2f}")
            return invoice
            
        except Exception as e:
            logger.error(f"Failed to extract invoice data: {e}")
            invoice.confidence_score = 0.0
            invoice.processing_status = ProcessingStatus.FAILED
            return invoice
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string into datetime object"""
        date_formats = [
            '%m/%d/%Y', '%m-%d-%Y',
            '%d/%m/%Y', '%d-%m-%Y',
            '%Y-%m-%d', '%Y/%m/%d',
            '%m/%d/%y', '%m-%d-%y',
            '%d/%m/%y', '%d-%m-%y'
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        return None
    
    async def categorize_expense(self, description: str, amount: Decimal, vendor: str = "") -> Tuple[str, float]:
        """Categorize expense based on description and vendor"""
        text = f"{description} {vendor}".lower()
        
        # Check category keywords
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score / len(keywords)
        
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            confidence = category_scores[best_category]
        else:
            best_category = "general"
            confidence = 0.1
        
        # Adjust confidence based on amount (larger amounts might be more specific)
        if amount > 1000:
            confidence *= 1.2
        elif amount < 50:
            confidence *= 0.8
        
        confidence = min(1.0, confidence)
        
        return best_category, confidence


class AnomalyDetector:
    """Detects anomalies in financial data"""
    
    def __init__(self):
        self.baseline_metrics: Dict[str, Any] = {}
        self.anomaly_threshold = 2.0  # Standard deviations
        self.learning_window = 100  # Number of recent transactions to consider
        
    async def detect_invoice_anomalies(self, invoice: Invoice, historical_data: List[Invoice]) -> List[str]:
        """Detect anomalies in invoice data"""
        anomalies = []
        
        if not historical_data:
            return anomalies
        
        try:
            # Amount anomaly detection
            amounts = [float(inv.total_amount) for inv in historical_data if inv.total_amount > 0]
            if amounts:
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                
                if std_amount > 0:
                    z_score = abs(float(invoice.total_amount) - mean_amount) / std_amount
                    if z_score > self.anomaly_threshold:
                        anomalies.append(f"Unusual amount: ${invoice.total_amount} (z-score: {z_score:.2f})")
            
            # Vendor anomaly detection
            vendor_counts = defaultdict(int)
            for inv in historical_data:
                if inv.vendor_name:
                    vendor_counts[inv.vendor_name.lower()] += 1
            
            if invoice.vendor_name and invoice.vendor_name.lower() not in vendor_counts:
                anomalies.append(f"New vendor: {invoice.vendor_name}")
            
            # Date anomaly detection (weekends, holidays, etc.)
            if invoice.invoice_date:
                if invoice.invoice_date.weekday() >= 5:  # Weekend
                    anomalies.append("Invoice dated on weekend")
                
                # Future date
                if invoice.invoice_date > datetime.now() + timedelta(days=1):
                    anomalies.append("Invoice dated in the future")
            
            # Duplicate detection
            for historical_inv in historical_data:
                if (invoice.invoice_number == historical_inv.invoice_number and 
                    invoice.vendor_name.lower() == historical_inv.vendor_name.lower()):
                    anomalies.append(f"Potential duplicate: {invoice.invoice_number}")
                    break
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            anomalies.append("Anomaly detection failed")
        
        return anomalies
    
    async def detect_transaction_anomalies(self, transaction: BankTransaction, 
                                         historical_data: List[BankTransaction]) -> List[str]:
        """Detect anomalies in bank transactions"""
        anomalies = []
        
        if not historical_data:
            return anomalies
        
        try:
            # Amount anomaly detection
            amounts = [abs(float(t.amount)) for t in historical_data]
            if amounts:
                mean_amount = np.mean(amounts)
                std_amount = np.std(amounts)
                
                if std_amount > 0:
                    z_score = abs(abs(float(transaction.amount)) - mean_amount) / std_amount
                    if z_score > self.anomaly_threshold:
                        anomalies.append(f"Unusual transaction amount: ${transaction.amount} (z-score: {z_score:.2f})")
            
            # Time-based anomalies
            if transaction.date.hour < 6 or transaction.date.hour > 22:
                anomalies.append("Transaction outside business hours")
            
            if transaction.date.weekday() >= 5:
                anomalies.append("Weekend transaction")
            
            # Description anomalies (very short or very long descriptions)
            if len(transaction.description) < 3:
                anomalies.append("Unusually short transaction description")
            elif len(transaction.description) > 100:
                anomalies.append("Unusually long transaction description")
            
            # Frequency anomalies (same amount and description)
            similar_transactions = [
                t for t in historical_data[-30:]  # Last 30 transactions
                if (t.amount == transaction.amount and 
                    t.description.lower() == transaction.description.lower())
            ]
            
            if len(similar_transactions) >= 5:
                anomalies.append("High frequency of identical transactions")
            
        except Exception as e:
            logger.error(f"Transaction anomaly detection failed: {e}")
            anomalies.append("Transaction anomaly detection failed")
        
        return anomalies


class JournalEntryGenerator:
    """Generates double-entry journal entries from financial documents"""
    
    def __init__(self):
        self.account_mapping = {
            # Expense categories to accounts
            'office_supplies': '6100 - Office Supplies Expense',
            'software': '6200 - Software Expense',
            'travel': '6300 - Travel Expense',
            'utilities': '6400 - Utilities Expense',
            'marketing': '6500 - Marketing Expense',
            'consulting': '6600 - Professional Services Expense',
            'equipment': '1500 - Equipment',
            'rent': '6700 - Rent Expense',
            'general': '6000 - General Expense',
            
            # Common accounts
            'cash': '1000 - Cash',
            'accounts_payable': '2000 - Accounts Payable',
            'accounts_receivable': '1200 - Accounts Receivable',
            'sales_revenue': '4000 - Sales Revenue',
            'cost_of_goods_sold': '5000 - Cost of Goods Sold'
        }
    
    async def generate_invoice_entry(self, invoice: Invoice, category: str = None) -> JournalEntry:
        """Generate journal entry for an invoice (expense)"""
        if not category:
            extractor = DocumentExtractor()
            category, _ = await extractor.categorize_expense(
                invoice.raw_text, invoice.total_amount, invoice.vendor_name
            )
        
        # Determine expense account
        expense_account = self.account_mapping.get(category, self.account_mapping['general'])
        
        # Create journal entry
        entry = JournalEntry(
            entry_id=f"JE-{invoice.document_id}",
            date=invoice.invoice_date or datetime.now(),
            description=f"Invoice from {invoice.vendor_name} - {invoice.invoice_number}",
            reference=invoice.invoice_number,
            debits=[(expense_account, invoice.total_amount)],
            credits=[('2000 - Accounts Payable', invoice.total_amount)],
            source_document_id=invoice.document_id
        )
        
        return entry
    
    async def generate_payment_entry(self, transaction: BankTransaction, 
                                   matched_invoice: Optional[Invoice] = None) -> JournalEntry:
        """Generate journal entry for a bank transaction"""
        
        if matched_invoice:
            # Payment of invoice
            description = f"Payment to {matched_invoice.vendor_name} - {matched_invoice.invoice_number}"
            debits = [('2000 - Accounts Payable', abs(transaction.amount))]
        else:
            # General expense or income
            if transaction.amount < 0:  # Expense
                category, _ = await DocumentExtractor().categorize_expense(
                    transaction.description, abs(transaction.amount)
                )
                expense_account = self.account_mapping.get(category, self.account_mapping['general'])
                description = f"Bank expense: {transaction.description}"
                debits = [(expense_account, abs(transaction.amount))]
            else:  # Income
                description = f"Bank deposit: {transaction.description}"
                debits = []
        
        credits = [('1000 - Cash', abs(transaction.amount))] if transaction.amount < 0 else []
        if transaction.amount > 0:
            credits = [('4000 - Sales Revenue', transaction.amount)]
            debits = [('1000 - Cash', transaction.amount)]
        
        entry = JournalEntry(
            entry_id=f"JE-{transaction.transaction_id}",
            date=transaction.date,
            description=description,
            reference=transaction.reference,
            debits=debits,
            credits=credits
        )
        
        return entry
    
    def validate_journal_entry(self, entry: JournalEntry) -> Tuple[bool, List[str]]:
        """Validate journal entry for double-entry rules"""
        errors = []
        
        # Check if balanced
        if not entry.is_balanced:
            errors.append(f"Entry not balanced: Debits=${entry.total_debits}, Credits=${entry.total_credits}")
        
        # Check for empty accounts
        if not entry.debits and not entry.credits:
            errors.append("Entry has no debits or credits")
        
        # Check for negative amounts
        for account, amount in entry.debits + entry.credits:
            if amount < 0:
                errors.append(f"Negative amount in {account}: {amount}")
        
        # Check account format
        for account, amount in entry.debits + entry.credits:
            if not account or len(account.strip()) == 0:
                errors.append("Empty account name")
        
        return len(errors) == 0, errors


class ReconciliationEngine:
    """Handles bank statement reconciliation"""
    
    def __init__(self):
        self.matching_threshold = 0.8
        self.date_tolerance_days = 3
        
    async def reconcile_transactions(self, bank_transactions: List[BankTransaction],
                                   invoices: List[Invoice]) -> Dict[str, Any]:
        """Reconcile bank transactions with invoices"""
        reconciliation_results = {
            'matched_transactions': [],
            'unmatched_transactions': [],
            'unmatched_invoices': [],
            'confidence_scores': []
        }
        
        unmatched_transactions = bank_transactions.copy()
        unmatched_invoices = invoices.copy()
        
        for transaction in bank_transactions:
            best_match = None
            best_score = 0.0
            
            for invoice in unmatched_invoices:
                score = await self._calculate_match_score(transaction, invoice)
                if score > best_score and score >= self.matching_threshold:
                    best_score = score
                    best_match = invoice
            
            if best_match:
                # Found a match
                transaction.matched_document_id = best_match.document_id
                transaction.confidence = best_score
                
                reconciliation_results['matched_transactions'].append({
                    'transaction': transaction,
                    'invoice': best_match,
                    'confidence': best_score
                })
                
                unmatched_transactions.remove(transaction)
                unmatched_invoices.remove(best_match)
        
        reconciliation_results['unmatched_transactions'] = unmatched_transactions
        reconciliation_results['unmatched_invoices'] = unmatched_invoices
        
        # Calculate overall reconciliation rate
        total_transactions = len(bank_transactions)
        matched_count = len(reconciliation_results['matched_transactions'])
        reconciliation_rate = matched_count / total_transactions if total_transactions > 0 else 0
        
        reconciliation_results['reconciliation_rate'] = reconciliation_rate
        
        logger.info(f"Reconciliation completed: {matched_count}/{total_transactions} "
                   f"transactions matched ({reconciliation_rate:.2%})")
        
        return reconciliation_results
    
    async def _calculate_match_score(self, transaction: BankTransaction, invoice: Invoice) -> float:
        """Calculate matching score between transaction and invoice"""
        score = 0.0
        
        # Amount matching (exact match gets full score)
        if abs(transaction.amount) == invoice.total_amount:
            score += 0.4
        else:
            # Partial score for close amounts
            amount_diff = abs(abs(transaction.amount) - invoice.total_amount)
            max_amount = max(abs(transaction.amount), invoice.total_amount)
            if max_amount > 0:
                amount_similarity = 1.0 - (amount_diff / max_amount)
                score += 0.4 * max(0, amount_similarity)
        
        # Date matching
        if invoice.invoice_date and transaction.date:
            date_diff = abs((transaction.date - invoice.invoice_date).days)
            if date_diff <= self.date_tolerance_days:
                date_score = max(0, 1.0 - (date_diff / self.date_tolerance_days))
                score += 0.3 * date_score
        
        # Description/vendor matching
        transaction_desc = transaction.description.lower()
        vendor_name = invoice.vendor_name.lower()
        
        if vendor_name in transaction_desc or any(word in transaction_desc for word in vendor_name.split()):
            score += 0.3
        
        return min(1.0, score)


class FinancialWorkflowOrchestrator:
    """Main orchestrator for financial workflow automation"""
    
    def __init__(self):
        self.document_extractor = DocumentExtractor()
        self.anomaly_detector = AnomalyDetector()
        self.journal_entry_generator = JournalEntryGenerator()
        self.reconciliation_engine = ReconciliationEngine()
        
        # Storage
        self.processed_documents: Dict[str, FinancialDocument] = {}
        self.invoices: Dict[str, Invoice] = {}
        self.bank_transactions: Dict[str, BankTransaction] = {}
        self.journal_entries: Dict[str, JournalEntry] = {}
        
        # Analytics
        self.processing_stats = defaultdict(int)
        self.quality_metrics = defaultdict(list)
        
    async def process_invoice(self, raw_text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Complete invoice processing pipeline"""
        document_id = f"DOC-{datetime.now().strftime('%Y%m%d%H%M%S')}-{hash(raw_text) % 10000:04d}"
        
        # Create base document
        document = FinancialDocument(
            document_id=document_id,
            document_type=DocumentType.INVOICE,
            raw_text=raw_text,
            metadata=metadata or {}
        )
        
        self.processed_documents[document_id] = document
        
        try:
            # Step 1: Extract structured data
            document.processing_status = ProcessingStatus.PROCESSING
            invoice = await self.document_extractor.extract_invoice_data(document)
            self.invoices[document_id] = invoice
            
            # Step 2: Detect anomalies
            historical_invoices = list(self.invoices.values())[:-1]  # Exclude current invoice
            anomalies = await self.anomaly_detector.detect_invoice_anomalies(invoice, historical_invoices)
            invoice.anomalies = anomalies
            
            # Step 3: Generate journal entry
            category, category_confidence = await self.document_extractor.categorize_expense(
                invoice.raw_text, invoice.total_amount, invoice.vendor_name
            )
            
            journal_entry = await self.journal_entry_generator.generate_invoice_entry(invoice, category)
            
            # Validate journal entry
            is_valid, validation_errors = self.journal_entry_generator.validate_journal_entry(journal_entry)
            
            if is_valid:
                self.journal_entries[journal_entry.entry_id] = journal_entry
                invoice.processing_status = ProcessingStatus.COMPLETED
            else:
                invoice.processing_status = ProcessingStatus.REQUIRES_REVIEW
                invoice.anomalies.extend([f"Journal entry validation: {error}" for error in validation_errors])
            
            invoice.processed_at = datetime.now()
            
            # Update statistics
            self.processing_stats['invoices_processed'] += 1
            self.quality_metrics['extraction_confidence'].append(invoice.confidence_score)
            
            result = {
                'success': True,
                'document_id': document_id,
                'invoice_data': {
                    'invoice_number': invoice.invoice_number,
                    'vendor_name': invoice.vendor_name,
                    'total_amount': str(invoice.total_amount),
                    'invoice_date': invoice.invoice_date.isoformat() if invoice.invoice_date else None,
                    'category': category,
                    'category_confidence': category_confidence
                },
                'confidence_score': invoice.confidence_score,
                'anomalies': anomalies,
                'journal_entry': {
                    'entry_id': journal_entry.entry_id,
                    'debits': [(acc, str(amt)) for acc, amt in journal_entry.debits],
                    'credits': [(acc, str(amt)) for acc, amt in journal_entry.credits],
                    'is_balanced': journal_entry.is_balanced
                } if is_valid else None,
                'processing_status': invoice.processing_status.value,
                'validation_errors': validation_errors if not is_valid else []
            }
            
            logger.info(f"Invoice processed successfully: {document_id} "
                       f"(confidence: {invoice.confidence_score:.2f}, anomalies: {len(anomalies)})")
            
            return result
            
        except Exception as e:
            document.processing_status = ProcessingStatus.FAILED
            self.processing_stats['processing_failures'] += 1
            
            logger.error(f"Invoice processing failed: {e}")
            
            return {
                'success': False,
                'document_id': document_id,
                'error': str(e),
                'processing_status': ProcessingStatus.FAILED.value
            }
    
    async def process_bank_statement(self, transactions_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process bank statement transactions"""
        processed_transactions = []
        
        for tx_data in transactions_data:
            transaction = BankTransaction(
                transaction_id=tx_data.get('id', f"TX-{len(self.bank_transactions)}"),
                date=datetime.fromisoformat(tx_data['date']) if isinstance(tx_data['date'], str) else tx_data['date'],
                description=tx_data['description'],
                amount=Decimal(str(tx_data['amount'])),
                balance=Decimal(str(tx_data.get('balance', '0'))),
                transaction_type=tx_data.get('type', 'debit' if tx_data['amount'] < 0 else 'credit'),
                reference=tx_data.get('reference', '')
            )
            
            # Detect anomalies
            historical_transactions = list(self.bank_transactions.values())
            anomalies = await self.anomaly_detector.detect_transaction_anomalies(transaction, historical_transactions)
            
            # Categorize transaction
            if transaction.amount < 0:  # Expense
                category, confidence = await self.document_extractor.categorize_expense(
                    transaction.description, abs(transaction.amount)
                )
                transaction.category = category
                transaction.confidence = confidence
            
            self.bank_transactions[transaction.transaction_id] = transaction
            
            processed_transactions.append({
                'transaction_id': transaction.transaction_id,
                'date': transaction.date.isoformat(),
                'description': transaction.description,
                'amount': str(transaction.amount),
                'category': transaction.category,
                'confidence': transaction.confidence,
                'anomalies': anomalies
            })
        
        # Perform reconciliation with existing invoices
        reconciliation_results = await self.reconciliation_engine.reconcile_transactions(
            list(self.bank_transactions.values()),
            list(self.invoices.values())
        )
        
        # Generate journal entries for transactions
        journal_entries = []
        for transaction in self.bank_transactions.values():
            matched_invoice = None
            if transaction.matched_document_id:
                matched_invoice = self.invoices.get(transaction.matched_document_id)
            
            try:
                journal_entry = await self.journal_entry_generator.generate_payment_entry(
                    transaction, matched_invoice
                )
                
                is_valid, validation_errors = self.journal_entry_generator.validate_journal_entry(journal_entry)
                
                if is_valid:
                    self.journal_entries[journal_entry.entry_id] = journal_entry
                    journal_entries.append({
                        'entry_id': journal_entry.entry_id,
                        'transaction_id': transaction.transaction_id,
                        'debits': [(acc, str(amt)) for acc, amt in journal_entry.debits],
                        'credits': [(acc, str(amt)) for acc, amt in journal_entry.credits],
                        'is_balanced': journal_entry.is_balanced
                    })
            
            except Exception as e:
                logger.error(f"Failed to generate journal entry for transaction {transaction.transaction_id}: {e}")
        
        self.processing_stats['bank_statements_processed'] += 1
        
        return {
            'success': True,
            'processed_transactions': processed_transactions,
            'reconciliation': {
                'matched_count': len(reconciliation_results['matched_transactions']),
                'unmatched_transactions': len(reconciliation_results['unmatched_transactions']),
                'reconciliation_rate': reconciliation_results['reconciliation_rate']
            },
            'journal_entries': journal_entries
        }
    
    async def get_financial_summary(self) -> Dict[str, Any]:
        """Generate financial summary and analytics"""
        
        # Calculate totals
        total_invoices = len(self.invoices)
        total_invoice_amount = sum(inv.total_amount for inv in self.invoices.values())
        
        total_transactions = len(self.bank_transactions)
        total_expenses = sum(abs(tx.amount) for tx in self.bank_transactions.values() if tx.amount < 0)
        total_income = sum(tx.amount for tx in self.bank_transactions.values() if tx.amount > 0)
        
        # Quality metrics
        avg_confidence = np.mean(self.quality_metrics['extraction_confidence']) if self.quality_metrics['extraction_confidence'] else 0
        
        # Anomaly statistics
        total_anomalies = sum(len(inv.anomalies) for inv in self.invoices.values())
        
        # Category breakdown
        expense_categories = defaultdict(Decimal)
        for tx in self.bank_transactions.values():
            if tx.amount < 0 and tx.category:
                expense_categories[tx.category] += abs(tx.amount)
        
        # Monthly trends (simplified)
        monthly_data = defaultdict(lambda: {'income': Decimal('0'), 'expenses': Decimal('0')})
        for tx in self.bank_transactions.values():
            month_key = tx.date.strftime('%Y-%m')
            if tx.amount > 0:
                monthly_data[month_key]['income'] += tx.amount
            else:
                monthly_data[month_key]['expenses'] += abs(tx.amount)
        
        return {
            'summary': {
                'total_invoices': total_invoices,
                'total_invoice_amount': str(total_invoice_amount),
                'total_transactions': total_transactions,
                'total_expenses': str(total_expenses),
                'total_income': str(total_income),
                'net_income': str(total_income - total_expenses)
            },
            'quality_metrics': {
                'average_extraction_confidence': avg_confidence,
                'total_anomalies_detected': total_anomalies,
                'processing_success_rate': (
                    (self.processing_stats['invoices_processed'] - self.processing_stats['processing_failures']) /
                    max(1, self.processing_stats['invoices_processed'])
                )
            },
            'expense_categories': {k: str(v) for k, v in expense_categories.items()},
            'monthly_trends': {
                month: {'income': str(data['income']), 'expenses': str(data['expenses'])}
                for month, data in monthly_data.items()
            },
            'processing_stats': dict(self.processing_stats)
        }
    
    async def export_journal_entries(self, format: str = 'json') -> Any:
        """Export journal entries for accounting software"""
        entries_data = []
        
        for entry in self.journal_entries.values():
            entry_data = {
                'entry_id': entry.entry_id,
                'date': entry.date.isoformat(),
                'description': entry.description,
                'reference': entry.reference,
                'debits': [{'account': acc, 'amount': str(amt)} for acc, amt in entry.debits],
                'credits': [{'account': acc, 'amount': str(amt)} for acc, amt in entry.credits],
                'total_debits': str(entry.total_debits),
                'total_credits': str(entry.total_credits),
                'is_balanced': entry.is_balanced,
                'source_document_id': entry.source_document_id,
                'created_by': entry.created_by
            }
            entries_data.append(entry_data)
        
        if format.lower() == 'json':
            return json.dumps(entries_data, indent=2, default=str)
        elif format.lower() == 'csv':
            # Flatten for CSV export
            csv_data = []
            for entry in entries_data:
                for debit in entry['debits']:
                    csv_data.append({
                        'entry_id': entry['entry_id'],
                        'date': entry['date'],
                        'description': entry['description'],
                        'account': debit['account'],
                        'debit': debit['amount'],
                        'credit': '',
                        'reference': entry['reference']
                    })
                for credit in entry['credits']:
                    csv_data.append({
                        'entry_id': entry['entry_id'],
                        'date': entry['date'],
                        'description': entry['description'],
                        'account': credit['account'],
                        'debit': '',
                        'credit': credit['amount'],
                        'reference': entry['reference']
                    })
            
            return pd.DataFrame(csv_data).to_csv(index=False)
        
        return entries_data


if __name__ == "__main__":
    async def demo_financial_workflow():
        """Demonstrate financial workflow automation"""
        workflow = FinancialWorkflowOrchestrator()
        
        print("=" * 80)
        print("FINANCIAL WORKFLOW AUTOMATION DEMONSTRATION")
        print("=" * 80)
        
        # Demo invoice processing
        sample_invoices = [
            {
                'raw_text': """
                INVOICE
                From: Office Supplies Inc.
                Invoice Number: INV-2024-001
                Date: 01/15/2024
                
                Description: Office supplies and stationery
                Amount: $245.50
                """,
                'metadata': {'source': 'email_attachment'}
            },
            {
                'raw_text': """
                Software License Invoice
                Microsoft Corporation
                Invoice: MS-2024-456
                Date: 01/16/2024
                
                Office 365 Business License
                Total: $1,200.00
                """,
                'metadata': {'source': 'vendor_portal'}
            }
        ]
        
        print("PROCESSING INVOICES:")
        print("-" * 40)
        
        for i, invoice_data in enumerate(sample_invoices, 1):
            result = await workflow.process_invoice(invoice_data['raw_text'], invoice_data['metadata'])
            
            print(f"\nInvoice {i}:")
            print(f"  Success: {result['success']}")
            if result['success']:
                print(f"  Invoice #: {result['invoice_data']['invoice_number']}")
                print(f"  Vendor: {result['invoice_data']['vendor_name']}")
                print(f"  Amount: ${result['invoice_data']['total_amount']}")
                print(f"  Category: {result['invoice_data']['category']}")
                print(f"  Confidence: {result['confidence_score']:.2f}")
                print(f"  Anomalies: {len(result['anomalies'])}")
        
        # Demo bank statement processing
        print("\n" + "=" * 40)
        print("PROCESSING BANK STATEMENT:")
        print("-" * 40)
        
        sample_transactions = [
            {
                'id': 'TX-001',
                'date': '2024-01-15',
                'description': 'OFFICE SUPPLIES INC',
                'amount': -245.50,
                'balance': 5000.00,
                'reference': 'ACH-001'
            },
            {
                'id': 'TX-002',
                'date': '2024-01-16',
                'description': 'MICROSOFT CORP ONLINE',
                'amount': -1200.00,
                'balance': 3754.50,
                'reference': 'ACH-002'
            },
            {
                'id': 'TX-003',
                'date': '2024-01-17',
                'description': 'CLIENT PAYMENT - PROJECT A',
                'amount': 2500.00,
                'balance': 6254.50,
                'reference': 'WIRE-001'
            }
        ]
        
        bank_result = await workflow.process_bank_statement(sample_transactions)
        
        print(f"Processed {len(bank_result['processed_transactions'])} transactions")
        print(f"Reconciliation rate: {bank_result['reconciliation']['reconciliation_rate']:.2%}")
        print(f"Generated {len(bank_result['journal_entries'])} journal entries")
        
        # Show financial summary
        print("\n" + "=" * 40)
        print("FINANCIAL SUMMARY:")
        print("-" * 40)
        
        summary = await workflow.get_financial_summary()
        
        print(f"Total invoices: {summary['summary']['total_invoices']}")
        print(f"Total invoice amount: ${summary['summary']['total_invoice_amount']}")
        print(f"Total transactions: {summary['summary']['total_transactions']}")
        print(f"Total expenses: ${summary['summary']['total_expenses']}")
        print(f"Total income: ${summary['summary']['total_income']}")
        print(f"Net income: ${summary['summary']['net_income']}")
        print(f"Average confidence: {summary['quality_metrics']['average_extraction_confidence']:.2f}")
        print(f"Anomalies detected: {summary['quality_metrics']['total_anomalies_detected']}")
        
        print("\nExpense categories:")
        for category, amount in summary['expense_categories'].items():
            print(f"  {category}: ${amount}")
        
        # Export journal entries
        print("\n" + "=" * 40)
        print("JOURNAL ENTRIES EXPORT:")
        print("-" * 40)
        
        json_export = await workflow.export_journal_entries('json')
        print(f"Exported {len(workflow.journal_entries)} journal entries")
        print("Sample journal entry:")
        if workflow.journal_entries:
            first_entry = list(workflow.journal_entries.values())[0]
            print(f"  Entry ID: {first_entry.entry_id}")
            print(f"  Description: {first_entry.description}")
            print(f"  Debits: {first_entry.debits}")
            print(f"  Credits: {first_entry.credits}")
            print(f"  Balanced: {first_entry.is_balanced}")
    
    # Run demonstration
    asyncio.run(demo_financial_workflow())