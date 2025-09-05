"""
Competitive Processing Engine
Multiple extraction methods compete for best results with performance tracking
Extends proven invoice processor foundation for multi-domain competitive intelligence
"""

import asyncio
import logging
import re
import json
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import Enum
import statistics
import hashlib
import time

# AI service imports with fallbacks
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

# Base agent framework
from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from agents.accountancy.invoice_processor import BudgetTracker
from agents.intelligence.document_classifier import DocumentType

logger = get_logger(__name__)


class ProcessingMethod(Enum):
    """Available processing methods"""
    REGEX_EXTRACTION = "regex_extraction"
    PATTERN_MATCHING = "pattern_matching"
    CLAUDE_API = "claude_api"
    OPENAI_API = "openai_api"
    TEMPLATE_MATCHING = "template_matching"
    HYBRID_RULES = "hybrid_rules"
    STATISTICAL_EXTRACTION = "statistical_extraction"


@dataclass
class ExtractionResult:
    """Result from a single extraction method"""
    method: ProcessingMethod
    extracted_data: Dict[str, Any]
    confidence_score: float
    processing_time: float
    cost_estimate: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'method': self.method.value,
            'extracted_data': self.extracted_data,
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'cost_estimate': self.cost_estimate,
            'error_message': self.error_message,
            'metadata': self.metadata
        }


@dataclass
class MethodPerformance:
    """Performance tracking for extraction methods"""
    method: ProcessingMethod
    attempts: int = 0
    successes: int = 0
    total_processing_time: float = 0.0
    total_cost: float = 0.0
    accuracy_scores: List[float] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        return self.successes / max(1, self.attempts)
    
    @property
    def average_accuracy(self) -> float:
        """Calculate average accuracy"""
        return statistics.mean(self.accuracy_scores) if self.accuracy_scores else 0.0
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time"""
        return self.total_processing_time / max(1, self.attempts)
    
    @property
    def average_cost(self) -> float:
        """Calculate average cost"""
        return self.total_cost / max(1, self.attempts)


class RegexExtractor:
    """Regex-based extraction methods (free tier)"""
    
    # Comprehensive regex patterns for different document types
    EXTRACTION_PATTERNS = {
        DocumentType.INVOICE: {
            'invoice_number': [
                r'invoice\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)',
                r'inv\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)',
                r'bill\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)'
            ],
            'total_amount': [
                r'total\s*amount\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'total\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'amount\s*due\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'grand\s*total\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ],
            'vendor_name': [
                r'(?:from|vendor|bill\s*from)\s*:?\s*([^\n\r]+)',
                r'^([A-Z][^\n\r]{5,50})',  # First capitalized line
            ],
            'invoice_date': [
                r'(?:invoice\s*)?date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}/\d{1,2}/\d{2,4})',
                r'(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})'
            ]
        },
        DocumentType.PURCHASE_ORDER: {
            'po_number': [
                r'(?:purchase\s*order|po)\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)',
                r'po\s*#?\s*([A-Z0-9\-]+)'
            ],
            'vendor_name': [
                r'vendor\s*:?\s*([^\n\r]+)',
                r'supplier\s*:?\s*([^\n\r]+)'
            ],
            'delivery_date': [
                r'delivery\s*date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'ship\s*by\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'total_amount': [
                r'total\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'amount\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ]
        },
        DocumentType.RECEIPT: {
            'transaction_amount': [
                r'total\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'amount\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'charged\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ],
            'merchant_name': [
                r'^([A-Z][^\n\r]{5,50})',  # First line usually merchant
                r'merchant\s*:?\s*([^\n\r]+)'
            ],
            'transaction_date': [
                r'date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}/\d{1,2}/\d{2,4})'
            ],
            'payment_method': [
                r'(visa|mastercard|amex|discover)\s*[x*]*\d{4}',
                r'card\s*ending\s*in\s*(\d{4})',
                r'(cash|credit|debit)'
            ]
        },
        DocumentType.BANK_STATEMENT: {
            'account_number': [
                r'account\s*(?:number|#)\s*:?\s*([X*]*\d{4})',
                r'(\d{4}\s*\d{4}\s*\d{4}\s*\d{4})'
            ],
            'statement_period': [
                r'statement\s*period\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\s*(?:to|-)?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'period\s*:?\s*([^\n\r]+)'
            ],
            'beginning_balance': [
                r'beginning\s*balance\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'opening\s*balance\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ],
            'ending_balance': [
                r'ending\s*balance\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'closing\s*balance\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ]
        },
        DocumentType.CONTRACT: {
            'parties': [
                r'between\s*([^,\n]+)\s*(?:,\s*(?:a|an)\s*[^,\n]+,?)?\s*(?:and|&)\s*([^,\n]+)',
                r'this\s*agreement\s*is\s*between\s*([^,\n]+)\s*and\s*([^,\n]+)'
            ],
            'effective_date': [
                r'effective\s*(?:date|as\s*of)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'dated\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'termination_date': [
                r'(?:expires?|terminates?|ends?)\s*(?:on)?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'term\s*ends?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ]
        },
        DocumentType.FINANCIAL_STATEMENT: {
            'statement_type': [
                r'(income\s*statement|profit\s*and\s*loss|balance\s*sheet|cash\s*flow)',
                r'(p&l|p\s*&\s*l)'
            ],
            'period_ending': [
                r'(?:for\s*the\s*period\s*ending|period\s*ended?|as\s*of)\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
                r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})'
            ],
            'total_revenue': [
                r'total\s*revenue\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'gross\s*revenue\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ],
            'net_income': [
                r'net\s*income\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
                r'(?:profit|income)\s*\(loss\)\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
            ]
        },
        DocumentType.LEGAL_DOCUMENT: {
            'case_number': [
                r'case\s*(?:number|no\.?|#)\s*:?\s*([A-Z0-9\-]+)',
                r'docket\s*(?:number|no\.?|#)\s*:?\s*([A-Z0-9\-]+)'
            ],
            'plaintiff': [
                r'plaintiff\s*:?\s*([^,\n]+)',
                r'([^,\n]+)\s*(?:vs?\.?|versus)\s*[^,\n]+'
            ],
            'defendant': [
                r'defendant\s*:?\s*([^,\n]+)',
                r'[^,\n]+\s*(?:vs?\.?|versus)\s*([^,\n]+)'
            ],
            'court': [
                r'(?:in\s*the\s*)?([^,\n]*court[^,\n]*)',
                r'jurisdiction\s*:?\s*([^,\n]+)'
            ]
        }
    }
    
    @staticmethod
    async def extract_data(text: str, document_type: DocumentType) -> Dict[str, Any]:
        """Extract data using regex patterns"""
        if document_type not in RegexExtractor.EXTRACTION_PATTERNS:
            return {}
        
        patterns = RegexExtractor.EXTRACTION_PATTERNS[document_type]
        extracted_data = {}
        
        for field, field_patterns in patterns.items():
            for pattern in field_patterns:
                try:
                    match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                    if match:
                        if len(match.groups()) == 1:
                            extracted_data[field] = match.group(1).strip()
                        else:
                            # Multiple groups - combine or use first
                            extracted_data[field] = [g.strip() for g in match.groups() if g]
                        break  # Use first matching pattern
                except re.error as e:
                    logger.warning(f"Regex error for pattern {pattern}: {e}")
        
        # Post-process extracted data
        extracted_data = RegexExtractor._post_process_data(extracted_data, document_type)
        
        return extracted_data
    
    @staticmethod
    def _post_process_data(data: Dict[str, Any], document_type: DocumentType) -> Dict[str, Any]:
        """Post-process extracted data for cleanup and validation"""
        processed = data.copy()
        
        # Clean up amounts
        for field in ['total_amount', 'transaction_amount', 'beginning_balance', 'ending_balance', 'total_revenue', 'net_income']:
            if field in processed:
                try:
                    # Remove currency symbols and commas, convert to decimal
                    amount_str = str(processed[field]).replace('$', '').replace(',', '').strip()
                    processed[field] = float(amount_str)
                except (ValueError, TypeError):
        logger.info(f'Processing task: {locals()}')
        return {'success': True, 'message': 'Task processed'}
        
        # Clean up dates
        date_fields = ['invoice_date', 'transaction_date', 'delivery_date', 'effective_date', 'termination_date', 'period_ending']
        for field in date_fields:
            if field in processed:
                date_str = str(processed[field]).strip()
                # Normalize date format (simple approach)
                processed[field] = RegexExtractor._normalize_date(date_str)
        
        # Clean up text fields
        text_fields = ['vendor_name', 'merchant_name', 'plaintiff', 'defendant', 'court']
        for field in text_fields:
            if field in processed:
                # Clean up common artifacts
                text = str(processed[field]).strip()
                text = re.sub(r'[^\w\s&,.-]', '', text)  # Remove special chars
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                processed[field] = text[:100]  # Limit length
        
        return processed
    
    @staticmethod
    def _normalize_date(date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format"""
        # Simple date normalization - could be enhanced
        date_patterns = [
            (r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})', r'\3-\1-\2'),  # MM/DD/YYYY -> YYYY-MM-DD
            (r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})', r'\1-\2-\3'),  # YYYY/MM/DD -> YYYY-MM-DD
            (r'(\d{1,2})[-/](\d{1,2})[-/](\d{2})', r'20\3-\1-\2')  # MM/DD/YY -> 20YY-MM-DD
        ]
        
        for pattern, replacement in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                return re.sub(pattern, replacement, date_str)
        
        return date_str  # Return original if no pattern matches


class PatternMatcher:
    """Advanced pattern matching with contextual analysis"""
    
    @staticmethod
    async def extract_data(text: str, document_type: DocumentType) -> Dict[str, Any]:
        """Extract data using advanced pattern matching"""
        lines = text.split('\n')
        extracted_data = {}
        
        # Document-specific pattern matching
        if document_type == DocumentType.INVOICE:
            extracted_data = await PatternMatcher._extract_invoice_patterns(lines, text)
        elif document_type == DocumentType.PURCHASE_ORDER:
            extracted_data = await PatternMatcher._extract_po_patterns(lines, text)
        elif document_type == DocumentType.RECEIPT:
            extracted_data = await PatternMatcher._extract_receipt_patterns(lines, text)
        elif document_type == DocumentType.BANK_STATEMENT:
            extracted_data = await PatternMatcher._extract_statement_patterns(lines, text)
        elif document_type == DocumentType.CONTRACT:
            extracted_data = await PatternMatcher._extract_contract_patterns(lines, text)
        elif document_type == DocumentType.FINANCIAL_STATEMENT:
            extracted_data = await PatternMatcher._extract_financial_patterns(lines, text)
        elif document_type == DocumentType.LEGAL_DOCUMENT:
            extracted_data = await PatternMatcher._extract_legal_patterns(lines, text)
        
        return extracted_data
    
    @staticmethod
    async def _extract_invoice_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract invoice data using contextual patterns"""
        data = {}
        
        # Find vendor name (usually in first few lines)
        for i, line in enumerate(lines[:5]):
            line = line.strip()
            if (len(line) > 5 and 
                not re.match(r'^\d', line) and 
                'invoice' not in line.lower() and
                line):
                # Looks like a company name
                data['vendor_name'] = line
                break
        
        # Find amounts by looking for currency patterns
        amounts = []
        for line in lines:
            matches = re.findall(r'\$\s*([\d,]+(?:\.\d{2})?)', line)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    amounts.append(amount)
                except ValueError:
        logger.info(f'Method {function_name} called')
        return {}
        
        # Total is usually the largest amount
        if amounts:
            data['total_amount'] = max(amounts)
        
        # Look for line items (lines with descriptions and amounts)
        line_items = []
        for line in lines:
            if re.search(r'.*\$\s*[\d,]+(?:\.\d{2})?', line) and len(line.strip()) > 10:
                # Extract description and amount
                parts = re.split(r'\$\s*([\d,]+(?:\.\d{2})?)', line)
                if len(parts) >= 3:
                    description = parts[0].strip()
                    amount = parts[1].replace(',', '')
                    try:
                        line_items.append({
                            'description': description,
                            'amount': float(amount)
                        })
                    except ValueError:
                        pass
        
        if line_items:
            data['line_items'] = line_items
        
        return data
    
    @staticmethod
    async def _extract_po_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract purchase order patterns"""
        data = {}
        
        # Look for PO-specific patterns
        for line in lines:
            # PO number patterns
            po_match = re.search(r'(?:purchase\s*order|po)\s*(?:#|number|no\.?)?\s*:?\s*([A-Z0-9\-]+)', line, re.IGNORECASE)
            if po_match:
                data['po_number'] = po_match.group(1)
            
            # Delivery date patterns
            delivery_match = re.search(r'(?:deliver|ship)\s*(?:by|date)\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', line, re.IGNORECASE)
            if delivery_match:
                data['delivery_date'] = delivery_match.group(1)
        
        return data
    
    @staticmethod
    async def _extract_receipt_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract receipt patterns"""
        data = {}
        
        # Merchant name is usually first non-empty line
        for line in lines:
            line = line.strip()
            if line and not re.match(r'^\d', line):
                data['merchant_name'] = line
                break
        
        # Look for payment method indicators
        for line in lines:
            payment_match = re.search(r'(visa|mastercard|amex|discover|cash|credit|debit)', line, re.IGNORECASE)
            if payment_match:
                data['payment_method'] = payment_match.group(1).upper()
                break
        
        return data
    
    @staticmethod
    async def _extract_statement_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract bank statement patterns"""
        data = {}
        
        # Look for account number (often masked)
        for line in lines:
            account_match = re.search(r'(?:account|acct).*?([X*]*\d{4})', line, re.IGNORECASE)
            if account_match:
                data['account_number'] = account_match.group(1)
                break
        
        # Extract transactions (simplified)
        transactions = []
        for line in lines:
            # Look for lines with date and amount
            if re.search(r'\d{1,2}[-/]\d{1,2}.*\$[\d,]+(?:\.\d{2})?', line):
                transactions.append(line.strip())
        
        if transactions:
            data['transactions'] = transactions[:10]  # Limit to first 10
        
        return data
    
    @staticmethod
    async def _extract_contract_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract contract patterns"""
        data = {}
        
        # Look for agreement parties
        for line in lines:
            parties_match = re.search(r'between\s+([^,\n]+)\s+and\s+([^,\n]+)', line, re.IGNORECASE)
            if parties_match:
                data['party_1'] = parties_match.group(1).strip()
                data['party_2'] = parties_match.group(2).strip()
                break
        
        return data
    
    @staticmethod
    async def _extract_financial_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract financial statement patterns"""
        data = {}
        
        # Determine statement type
        if re.search(r'income\s*statement|profit.*loss', text, re.IGNORECASE):
            data['statement_type'] = 'income_statement'
        elif re.search(r'balance\s*sheet', text, re.IGNORECASE):
            data['statement_type'] = 'balance_sheet'
        elif re.search(r'cash\s*flow', text, re.IGNORECASE):
            data['statement_type'] = 'cash_flow'
        
        return data
    
    @staticmethod
    async def _extract_legal_patterns(lines: List[str], text: str) -> Dict[str, Any]:
        """Extract legal document patterns"""
        data = {}
        
        # Look for case information
        for line in lines:
            case_match = re.search(r'(?:case|docket)\s*(?:no\.?|#)\s*:?\s*([A-Z0-9\-]+)', line, re.IGNORECASE)
            if case_match:
                data['case_number'] = case_match.group(1)
                break
        
        return data


class AIApiExtractor:
    """AI API-based extraction (paid methods with budget consciousness)"""
    
    def __init__(self, budget_tracker: BudgetTracker):
        self.budget_tracker = budget_tracker
        self.claude_client = None
        self.openai_client = None
        
        # Initialize clients if available and within budget
        if anthropic and budget_tracker.can_use_anthropic():
            self.claude_client = anthropic.Anthropic()
        
        if openai and hasattr(budget_tracker, 'can_use_openai') and budget_tracker.can_use_openai():
            self.openai_client = openai.OpenAI()
    
    async def extract_with_claude(self, text: str, document_type: DocumentType) -> Dict[str, Any]:
        """Extract data using Claude API"""
        if not self.claude_client or not self.budget_tracker.can_use_anthropic():
            raise RuntimeError("Claude API not available or budget exceeded")
        
        # Document-type specific prompts
        prompt_templates = {
            DocumentType.INVOICE: """
Extract invoice information from this text. Return JSON with: invoice_number, vendor_name, 
total_amount (numeric), invoice_date (YYYY-MM-DD), line_items (array of {description, amount}).
""",
            DocumentType.PURCHASE_ORDER: """
Extract purchase order information from this text. Return JSON with: po_number, vendor_name,
delivery_date (YYYY-MM-DD), total_amount (numeric), line_items (array).
""",
            DocumentType.RECEIPT: """
Extract receipt information from this text. Return JSON with: merchant_name, transaction_amount (numeric),
transaction_date (YYYY-MM-DD), payment_method.
""",
            DocumentType.BANK_STATEMENT: """
Extract bank statement information from this text. Return JSON with: account_number, 
statement_period, beginning_balance (numeric), ending_balance (numeric), transactions (array).
""",
            DocumentType.CONTRACT: """
Extract contract information from this text. Return JSON with: parties (array), effective_date (YYYY-MM-DD),
termination_date (YYYY-MM-DD), key_terms (array).
""",
            DocumentType.FINANCIAL_STATEMENT: """
Extract financial statement information from this text. Return JSON with: statement_type, 
period_ending (YYYY-MM-DD), total_revenue (numeric), net_income (numeric).
""",
            DocumentType.LEGAL_DOCUMENT: """
Extract legal document information from this text. Return JSON with: case_number, plaintiff, 
defendant, court, filing_date (YYYY-MM-DD).
"""
        }
        
        prompt = prompt_templates.get(document_type, "Extract key information and return as JSON.")
        full_prompt = f"{prompt}\n\nText to analyze:\n{text[:3000]}"  # Limit text to control tokens
        
        try:
            response = self.claude_client.messages.create(
                model="claude-3-haiku-20240307",  # Use cheaper model
                max_tokens=1000,
                messages=[{"role": "user", "content": full_prompt}]
            )
            
            # Track usage
            estimated_tokens = len(full_prompt.split()) + 1000
            self.budget_tracker.add_anthropic_usage(estimated_tokens)
            
            # Parse response
            result_text = response.content[0].text
            
            # Try to extract JSON from response
            json_start = result_text.find('{')
            json_end = result_text.rfind('}')
            
            if json_start >= 0 and json_end >= 0:
                json_str = result_text[json_start:json_end + 1]
                return json.loads(json_str)
            else:
                # If no JSON found, try to parse the entire response
                return json.loads(result_text)
                
        except json.JSONDecodeError as e:
            logger.error(f"Claude API JSON parsing error: {e}")
            return {}
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            raise
    
    async def extract_with_openai(self, text: str, document_type: DocumentType) -> Dict[str, Any]:
        """Extract data using OpenAI API"""
        if not self.openai_client:
            raise RuntimeError("OpenAI API not available")
        
        # Similar implementation to Claude but for OpenAI
        # This would require OpenAI API key and budget tracking
        # For now, return empty dict as placeholder
        return {}


class CompetitiveProcessingEngine:
    """
    Orchestrates multiple extraction methods and selects best results
    Tracks performance and optimizes method selection
    """
    
    def __init__(self, budget_tracker: Optional[BudgetTracker] = None):
        self.budget_tracker = budget_tracker or BudgetTracker()
        self.regex_extractor = RegexExtractor()
        self.pattern_matcher = PatternMatcher()
        self.ai_extractor = AIApiExtractor(self.budget_tracker)
        
        # Performance tracking
        self.method_performance: Dict[ProcessingMethod, MethodPerformance] = {}
        for method in ProcessingMethod:
            self.method_performance[method] = MethodPerformance(method)
    
    async def process_competitively(
        self, 
        text: str, 
        document_type: DocumentType,
        methods: Optional[List[ProcessingMethod]] = None
    ) -> List[ExtractionResult]:
        """Run multiple extraction methods and return all results"""
        
        # Default methods if none specified
        if methods is None:
            methods = [
                ProcessingMethod.REGEX_EXTRACTION,
                ProcessingMethod.PATTERN_MATCHING,
                ProcessingMethod.HYBRID_RULES
            ]
            
            # Add AI methods if budget allows
            if self.budget_tracker.can_use_anthropic():
                methods.append(ProcessingMethod.CLAUDE_API)
        
        results = []
        
        # Run each method
        for method in methods:
            try:
                start_time = time.perf_counter()
                performance = self.method_performance[method]
                performance.attempts += 1
                
                # Execute extraction method
                if method == ProcessingMethod.REGEX_EXTRACTION:
                    extracted_data = await self.regex_extractor.extract_data(text, document_type)
                    cost = 0.0  # Free method
                elif method == ProcessingMethod.PATTERN_MATCHING:
                    extracted_data = await self.pattern_matcher.extract_data(text, document_type)
                    cost = 0.0  # Free method
                elif method == ProcessingMethod.CLAUDE_API:
                    extracted_data = await self.ai_extractor.extract_with_claude(text, document_type)
                    cost = 0.002  # Estimated cost
                elif method == ProcessingMethod.HYBRID_RULES:
                    # Combine regex and pattern matching
                    regex_data = await self.regex_extractor.extract_data(text, document_type)
                    pattern_data = await self.pattern_matcher.extract_data(text, document_type)
                    extracted_data = {**regex_data, **pattern_data}  # Merge results
                    cost = 0.0
                else:
                    logger.warning(f"Method {method} not implemented")
                    continue
                
                processing_time = time.perf_counter() - start_time
                
                # Calculate confidence score
                confidence = await self._calculate_confidence(extracted_data, document_type)
                
                result = ExtractionResult(
                    method=method,
                    extracted_data=extracted_data,
                    confidence_score=confidence,
                    processing_time=processing_time,
                    cost_estimate=cost
                )
                
                results.append(result)
                
                # Update performance tracking
                if confidence > 0.5:  # Consider success if confidence > 0.5
                    performance.successes += 1
                
                performance.total_processing_time += processing_time
                performance.total_cost += cost
                performance.accuracy_scores.append(confidence)
                
                logger.info(f"Method {method.value} completed: confidence={confidence:.2f}, time={processing_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Method {method.value} failed: {e}")
                results.append(ExtractionResult(
                    method=method,
                    extracted_data={},
                    confidence_score=0.0,
                    processing_time=0.0,
                    cost_estimate=0.0,
                    error_message=str(e)
                ))
        
        return results
    
    async def get_best_result(
        self, 
        text: str, 
        document_type: DocumentType,
        methods: Optional[List[ProcessingMethod]] = None
    ) -> ExtractionResult:
        """Get the single best result from competitive processing"""
        
        results = await self.process_competitively(text, document_type, methods)
        
        # Filter successful results
        successful_results = [r for r in results if r.confidence_score > 0.1]
        
        if not successful_results:
            # Return best failed result
            return max(results, key=lambda r: r.confidence_score) if results else ExtractionResult(
                method=ProcessingMethod.REGEX_EXTRACTION,
                extracted_data={},
                confidence_score=0.0,
                processing_time=0.0,
                cost_estimate=0.0
            )
        
        # Return result with highest confidence
        return max(successful_results, key=lambda r: r.confidence_score)
    
    async def _calculate_confidence(self, data: Dict[str, Any], document_type: DocumentType) -> float:
        """Calculate confidence score for extracted data"""
        if not data:
            return 0.0
        
        # Document-type specific confidence calculation
        required_fields = {
            DocumentType.INVOICE: ['invoice_number', 'vendor_name', 'total_amount'],
            DocumentType.PURCHASE_ORDER: ['po_number', 'vendor_name', 'delivery_date'],
            DocumentType.RECEIPT: ['merchant_name', 'transaction_amount', 'transaction_date'],
            DocumentType.BANK_STATEMENT: ['account_number', 'statement_period'],
            DocumentType.CONTRACT: ['parties', 'effective_date'],
            DocumentType.FINANCIAL_STATEMENT: ['statement_type', 'period_ending'],
            DocumentType.LEGAL_DOCUMENT: ['case_number', 'plaintiff', 'defendant']
        }
        
        document_required = required_fields.get(document_type, [])
        
        # Base confidence from field presence
        field_score = 0.0
        for field in document_required:
            if field in data and data[field]:
                field_score += 1.0
        
        # Normalize by number of required fields
        if document_required:
            field_confidence = field_score / len(document_required)
        else:
            field_confidence = 0.5  # Default for unknown types
        
        # Bonus for additional fields
        bonus = min(0.2, (len(data) - len(document_required)) * 0.05)
        
        # Quality penalties
        quality_penalty = 0.0
        for value in data.values():
            if isinstance(value, str):
                if len(value) < 2:  # Very short values
                    quality_penalty += 0.05
                if re.match(r'^[^\w]*$', value):  # Only special characters
                    quality_penalty += 0.1
        
        final_confidence = min(1.0, field_confidence + bonus - quality_penalty)
        return max(0.0, final_confidence)
    
    def get_method_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all methods"""
        performance_data = {}
        
        for method, perf in self.method_performance.items():
            performance_data[method.value] = {
                'attempts': perf.attempts,
                'successes': perf.successes,
                'success_rate': perf.success_rate,
                'average_accuracy': perf.average_accuracy,
                'average_processing_time': perf.average_processing_time,
                'average_cost': perf.average_cost,
                'total_cost': perf.total_cost
            }
        
        return performance_data
    
    def get_recommended_methods(self, document_type: DocumentType, budget_limit: float = 0.05) -> List[ProcessingMethod]:
        """Recommend best methods for a document type within budget"""
        # Sort methods by performance score (accuracy/cost ratio)
        method_scores = []
        
        for method, perf in self.method_performance.items():
            if perf.attempts > 0 and perf.average_cost <= budget_limit:
                score = perf.average_accuracy / max(0.001, perf.average_cost + 0.001)  # Accuracy/cost ratio
                method_scores.append((method, score))
        
        # Sort by score descending
        method_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top 3 methods
        return [method for method, _ in method_scores[:3]]


# Example usage and testing
async def main():
    """Example usage of competitive processing engine"""
    engine = CompetitiveProcessingEngine()
    
    # Test with sample invoice
    invoice_text = """
    ACME Corporation
    123 Business Street
    City, ST 12345
    
    INVOICE #INV-2024-001
    Date: January 15, 2024
    
    Bill To:
    XYZ Company
    456 Main Street
    
    Description                 Amount
    Consulting Services        $1,500.00
    Software License           $300.00
    
    Subtotal:                  $1,800.00
    Tax (8.5%):                $153.00
    Total Amount:              $1,953.00
    
    Due Date: February 15, 2024
    """
    
    # Process competitively
    results = await engine.process_competitively(invoice_text, DocumentType.INVOICE)
    
    print("Competitive Processing Results:")
    for result in results:
        print(f"\nMethod: {result.method.value}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Time: {result.processing_time:.3f}s")
        print(f"Data: {json.dumps(result.extracted_data, indent=2)}")
    
    # Get best result
    best_result = await engine.get_best_result(invoice_text, DocumentType.INVOICE)
    print(f"\nBest Result: {best_result.method.value} (confidence: {best_result.confidence_score:.2f})")


if __name__ == "__main__":
    asyncio.run(main())