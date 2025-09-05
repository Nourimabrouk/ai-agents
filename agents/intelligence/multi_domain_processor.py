"""
Multi-Domain Document Processing Hub
Unified interface for processing 7+ document types with specialized agents
Orchestrates competitive processing and result aggregation
"""

import asyncio
import logging
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import statistics

# Base agent framework
from templates.base_agent import BaseAgent, Action, Observation
from utils.observability.logging import get_logger
from agents.intelligence.document_classifier import DocumentClassifierAgent, DocumentType
from agents.accountancy.invoice_processor import BudgetTracker

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result from document processing with metadata"""
    document_type: DocumentType
    extracted_data: Dict[str, Any]
    confidence_score: float
    processing_method: str
    validation_errors: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    cost_estimate: float = 0.0
    competitive_results: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'document_type': self.document_type.value,
            'extracted_data': self.extracted_data,
            'confidence_score': self.confidence_score,
            'processing_method': self.processing_method,
            'validation_errors': self.validation_errors,
            'processing_time': self.processing_time,
            'cost_estimate': self.cost_estimate,
            'competitive_results': self.competitive_results,
            'timestamp': self.timestamp.isoformat()
        }


class ProcessingStrategy(Enum):
    """Processing strategies for different scenarios"""
    SPEED_OPTIMIZED = "speed_optimized"      # Fastest method only
    ACCURACY_OPTIMIZED = "accuracy_optimized"   # Best method only
    COMPETITIVE = "competitive"              # Multiple methods, best result
    CONSENSUS = "consensus"                  # Multiple methods, consensus result
    COST_OPTIMIZED = "cost_optimized"       # Cheapest method first


@dataclass
class ProcessingConfig:
    """Configuration for document processing"""
    strategy: ProcessingStrategy = ProcessingStrategy.COMPETITIVE
    accuracy_threshold: float = 0.95
    max_processing_time: float = 30.0  # seconds
    max_cost_per_document: float = 0.05  # dollars
    enable_parallel_processing: bool = True
    fallback_to_manual: bool = False
    save_intermediate_results: bool = True


class SpecializedProcessorRegistry:
    """Registry for specialized document processors"""
    
    def __init__(self):
        self.processors: Dict[DocumentType, List[BaseAgent]] = {}
        self.processor_performance: Dict[str, Dict[str, float]] = {}
    
    def register_processor(self, document_type: DocumentType, processor: BaseAgent) -> None:
        """Register a specialized processor for a document type"""
        if document_type not in self.processors:
            self.processors[document_type] = []
        
        self.processors[document_type].append(processor)
        
        # Initialize performance tracking
        processor_key = f"{document_type.value}_{processor.name}"
        if processor_key not in self.processor_performance:
            self.processor_performance[processor_key] = {
                'accuracy': 0.0,
                'speed': 0.0,
                'cost': 0.0,
                'usage_count': 0
            }
        
        logger.info(f"Registered processor {processor.name} for {document_type.value}")
    
    def get_processors(self, document_type: DocumentType) -> List[BaseAgent]:
        """Get all processors for a document type"""
        return self.processors.get(document_type, [])
    
    def get_best_processor(self, document_type: DocumentType, criterion: str = "accuracy") -> Optional[BaseAgent]:
        """Get best processor for a document type based on criterion"""
        processors = self.get_processors(document_type)
        if not processors:
            return {}
        
        best_processor = None
        best_score = -1.0
        
        for processor in processors:
            processor_key = f"{document_type.value}_{processor.name}"
            performance = self.processor_performance.get(processor_key, {})
            score = performance.get(criterion, 0.0)
            
            if score > best_score:
                best_score = score
                best_processor = processor
        
        return best_processor or processors[0]  # Fallback to first processor
    
    def update_performance(self, processor: BaseAgent, document_type: DocumentType, 
                          accuracy: float, speed: float, cost: float) -> None:
        """Update processor performance metrics"""
        processor_key = f"{document_type.value}_{processor.name}"
        
        if processor_key in self.processor_performance:
            perf = self.processor_performance[processor_key]
            usage_count = perf['usage_count']
            
            # Update with exponential moving average
            alpha = 0.1
            perf['accuracy'] = perf['accuracy'] * (1 - alpha) + accuracy * alpha
            perf['speed'] = perf['speed'] * (1 - alpha) + speed * alpha
            perf['cost'] = perf['cost'] * (1 - alpha) + cost * alpha
            perf['usage_count'] += 1


class CompetitiveProcessor:
    """Handles competitive processing with multiple methods"""
    
    def __init__(self, registry: SpecializedProcessorRegistry):
        self.registry = registry
        self.result_validator = ResultValidator()
    
    async def process_competitively(
        self, 
        document_type: DocumentType, 
        text: str, 
        file_path: Optional[str], 
        config: ProcessingConfig
    ) -> ProcessingResult:
        """Process document using competitive methods"""
        start_time = datetime.now()
        processors = self.registry.get_processors(document_type)
        
        if not processors:
            raise ValueError(f"No processors available for {document_type.value}")
        
        # Execute processing strategy
        if config.strategy == ProcessingStrategy.SPEED_OPTIMIZED:
            result = await self._process_speed_optimized(processors, text, file_path)
        elif config.strategy == ProcessingStrategy.ACCURACY_OPTIMIZED:
            result = await self._process_accuracy_optimized(processors, text, file_path)
        elif config.strategy == ProcessingStrategy.COMPETITIVE:
            result = await self._process_competitive(processors, text, file_path, config)
        elif config.strategy == ProcessingStrategy.CONSENSUS:
            result = await self._process_consensus(processors, text, file_path, config)
        elif config.strategy == ProcessingStrategy.COST_OPTIMIZED:
            result = await self._process_cost_optimized(processors, text, file_path, config)
        else:
            result = await self._process_competitive(processors, text, file_path, config)
        
        # Set processing metadata
        result.document_type = document_type
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        return result
    
    async def _process_speed_optimized(
        self, processors: List[BaseAgent], text: str, file_path: Optional[str]
    ) -> ProcessingResult:
        """Use fastest processor only"""
        best_processor = self.registry.get_best_processor(processors[0].__class__, "speed")
        
        task = {"text": text, "file_path": file_path} if file_path else {"text": text}
        result_data = await best_processor.process_task(task)
        
        return ProcessingResult(
            document_type=DocumentType.UNKNOWN,  # Will be set by caller
            extracted_data=result_data,
            confidence_score=result_data.get('accuracy', 0.0),
            processing_method=f"speed_optimized_{best_processor.name}"
        )
    
    async def _process_accuracy_optimized(
        self, processors: List[BaseAgent], text: str, file_path: Optional[str]
    ) -> ProcessingResult:
        """Use most accurate processor only"""
        best_processor = self.registry.get_best_processor(processors[0].__class__, "accuracy")
        
        task = {"text": text, "file_path": file_path} if file_path else {"text": text}
        result_data = await best_processor.process_task(task)
        
        return ProcessingResult(
            document_type=DocumentType.UNKNOWN,
            extracted_data=result_data,
            confidence_score=result_data.get('accuracy', 0.0),
            processing_method=f"accuracy_optimized_{best_processor.name}"
        )
    
    async def _process_competitive(
        self, processors: List[BaseAgent], text: str, file_path: Optional[str], config: ProcessingConfig
    ) -> ProcessingResult:
        """Run multiple processors and select best result"""
        task = {"text": text, "file_path": file_path} if file_path else {"text": text}
        
        # Run processors in parallel if enabled
        if config.enable_parallel_processing:
            results = await asyncio.gather(
                *[processor.process_task(task) for processor in processors],
                return_exceptions=True
            )
        else:
            results = []
            for processor in processors:
                try:
                    result = await processor.process_task(task)
                    results.append(result)
                except Exception as e:
                    results.append(e)
        
        # Filter successful results
        successful_results = []
        competitive_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                competitive_results.append({
                    "processor": processors[i].name,
                    "success": False,
                    "error": str(result)
                })
            else:
                successful_results.append((processors[i], result))
                competitive_results.append({
                    "processor": processors[i].name,
                    "success": True,
                    "confidence": result.get('accuracy', 0.0),
                    "data": result
                })
        
        if not successful_results:
            raise RuntimeError("All processors failed")
        
        # Select best result based on confidence score
        best_processor, best_result = max(
            successful_results, 
            key=lambda x: x[1].get('accuracy', 0.0)
        )
        
        return ProcessingResult(
            document_type=DocumentType.UNKNOWN,
            extracted_data=best_result,
            confidence_score=best_result.get('accuracy', 0.0),
            processing_method=f"competitive_{best_processor.name}",
            competitive_results=competitive_results
        )
    
    async def _process_consensus(
        self, processors: List[BaseAgent], text: str, file_path: Optional[str], config: ProcessingConfig
    ) -> ProcessingResult:
        """Run multiple processors and create consensus result"""
        competitive_result = await self._process_competitive(processors, text, file_path, config)
        
        # Analyze competitive results for consensus
        successful_results = [
            r for r in competitive_result.competitive_results 
            if r.get('success', False)
        ]
        
        if len(successful_results) < 2:
            return competitive_result
        
        # Create consensus from multiple results
        consensus_data = await self._create_consensus(successful_results)
        
        # Calculate consensus confidence
        confidences = [r.get('confidence', 0.0) for r in successful_results]
        consensus_confidence = statistics.mean(confidences) if confidences else 0.0
        
        # Boost confidence if results agree
        agreement_score = await self._calculate_agreement(successful_results)
        consensus_confidence *= (1.0 + agreement_score * 0.2)  # Up to 20% boost
        
        return ProcessingResult(
            document_type=DocumentType.UNKNOWN,
            extracted_data=consensus_data,
            confidence_score=min(1.0, consensus_confidence),
            processing_method="consensus",
            competitive_results=competitive_result.competitive_results
        )
    
    async def _process_cost_optimized(
        self, processors: List[BaseAgent], text: str, file_path: Optional[str], config: ProcessingConfig
    ) -> ProcessingResult:
        """Use cheapest processors first, escalate if needed"""
        # Sort processors by cost (assuming cost is tracked in performance)
        sorted_processors = sorted(
            processors,
            key=lambda p: self.registry.processor_performance.get(f"{p.name}", {}).get('cost', 0.0)
        )
        
        task = {"text": text, "file_path": file_path} if file_path else {"text": text}
        
        for processor in sorted_processors:
            try:
                result_data = await processor.process_task(task)
                confidence = result_data.get('accuracy', 0.0)
                
                # Use result if it meets accuracy threshold
                if confidence >= config.accuracy_threshold:
                    return ProcessingResult(
                        document_type=DocumentType.UNKNOWN,
                        extracted_data=result_data,
                        confidence_score=confidence,
                        processing_method=f"cost_optimized_{processor.name}"
                    )
            except Exception as e:
                logger.warning(f"Cost-optimized processor {processor.name} failed: {e}")
                continue
        
        # If no single processor meets threshold, fall back to competitive
        return await self._process_competitive(processors, text, file_path, config)
    
    async def _create_consensus(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create consensus data from multiple results"""
        consensus = {}
        
        # For each field, use majority vote or best confidence
        all_fields = set()
        for result in results:
            if 'data' in result and isinstance(result['data'], dict):
                all_fields.update(result['data'].keys())
        
        for field in all_fields:
            field_values = []
            for result in results:
                data = result.get('data', {})
                if field in data:
                    field_values.append({
                        'value': data[field],
                        'confidence': result.get('confidence', 0.0)
                    })
            
            if field_values:
                # Use value with highest confidence
                best_field = max(field_values, key=lambda x: x['confidence'])
                consensus[field] = best_field['value']
        
        return consensus
    
    async def _calculate_agreement(self, results: List[Dict[str, Any]]) -> float:
        """Calculate agreement score between results"""
        if len(results) < 2:
            return 1.0
        
        # Simplified agreement calculation based on common fields
        agreement_scores = []
        
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                data1 = results[i].get('data', {})
                data2 = results[j].get('data', {})
                
                common_fields = set(data1.keys()) & set(data2.keys())
                if common_fields:
                    matching_fields = sum(
                        1 for field in common_fields
                        if str(data1[field]).lower() == str(data2[field]).lower()
                    )
                    agreement = matching_fields / len(common_fields)
                    agreement_scores.append(agreement)
        
        return statistics.mean(agreement_scores) if agreement_scores else 0.0


class ResultValidator:
    """Validates and quality-checks processing results"""
    
    @staticmethod
    async def validate_result(result: ProcessingResult, document_type: DocumentType) -> ProcessingResult:
        """Validate processing result with document-specific checks"""
        errors = []
        
        # General validation
        if not result.extracted_data:
            errors.append("No data extracted")
        
        if result.confidence_score < 0.5:
            errors.append("Low confidence score")
        
        # Document-specific validation
        if document_type == DocumentType.INVOICE:
            errors.extend(await ResultValidator._validate_invoice_data(result.extracted_data))
        elif document_type == DocumentType.PURCHASE_ORDER:
            errors.extend(await ResultValidator._validate_po_data(result.extracted_data))
        elif document_type == DocumentType.RECEIPT:
            errors.extend(await ResultValidator._validate_receipt_data(result.extracted_data))
        elif document_type == DocumentType.BANK_STATEMENT:
            errors.extend(await ResultValidator._validate_statement_data(result.extracted_data))
        
        result.validation_errors = errors
        
        # Adjust confidence based on validation errors
        if errors:
            penalty = min(0.3, len(errors) * 0.1)
            result.confidence_score = max(0.0, result.confidence_score - penalty)
        
        return result
    
    @staticmethod
    async def _validate_invoice_data(data: Dict[str, Any]) -> List[str]:
        """Validate invoice-specific data"""
        errors = []
        
        required_fields = ['invoice_number', 'total_amount', 'vendor_name']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required invoice field: {field}")
        
        return errors
    
    @staticmethod
    async def _validate_po_data(data: Dict[str, Any]) -> List[str]:
        """Validate purchase order data"""
        errors = []
        
        required_fields = ['po_number', 'vendor_name', 'delivery_date']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required PO field: {field}")
        
        return errors
    
    @staticmethod
    async def _validate_receipt_data(data: Dict[str, Any]) -> List[str]:
        """Validate receipt data"""
        errors = []
        
        required_fields = ['transaction_amount', 'merchant_name', 'transaction_date']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required receipt field: {field}")
        
        return errors
    
    @staticmethod
    async def _validate_statement_data(data: Dict[str, Any]) -> List[str]:
        """Validate bank statement data"""
        errors = []
        
        required_fields = ['account_number', 'statement_period', 'transactions']
        for field in required_fields:
            if not data.get(field):
                errors.append(f"Missing required statement field: {field}")
        
        return errors


class MultiDomainProcessorAgent(BaseAgent):
    """
    Multi-Domain Document Processing Hub
    Unified interface for processing 7+ document types with competitive methods
    """
    
    def __init__(
        self,
        name: str = "multi_domain_processor",
        api_key: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name, api_key, config=config)
        
        # Core components
        self.classifier = DocumentClassifierAgent("embedded_classifier", api_key, config)
        self.processor_registry = SpecializedProcessorRegistry()
        self.competitive_processor = CompetitiveProcessor(self.processor_registry)
        self.budget_tracker = BudgetTracker()
        
        # Configuration
        self.processing_config = ProcessingConfig()
        if config and 'processing' in config:
            proc_config = config['processing']
            self.processing_config.strategy = ProcessingStrategy(
                proc_config.get('strategy', 'competitive')
            )
            self.processing_config.accuracy_threshold = proc_config.get('accuracy_threshold', 0.95)
            self.processing_config.max_cost_per_document = proc_config.get('max_cost_per_document', 0.05)
        
        # Performance metrics
        self.documents_processed = 0
        self.successful_processing = 0
        self.total_cost = 0.0
        self.average_accuracy = 0.0
        
        logger.info(f"Initialized multi-domain processor: {self.name}")
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute multi-domain document processing"""
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
            logger.error(f"Multi-domain processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_result": None
            }
    
    async def process_document_file(self, file_path: str) -> Dict[str, Any]:
        """Process document from file with full pipeline"""
        try:
            start_time = datetime.now()
            file_path = Path(file_path)
            
            logger.info(f"Processing document file: {file_path.name}")
            
            # Step 1: Classify document
            classification_result = await self.classifier.classify_document_file(str(file_path))
            
            if not classification_result['success']:
                raise RuntimeError(f"Classification failed: {classification_result.get('error')}")
            
            doc_type = DocumentType(classification_result['classification_result']['document_type'])
            
            if doc_type == DocumentType.UNKNOWN:
                raise RuntimeError("Document type could not be determined")
            
            # Step 2: Extract text (reuse from classification if available)
            text = ""  # Would extract from file or reuse from classifier
            
            # Step 3: Process with specialized processors
            processing_result = await self.competitive_processor.process_competitively(
                doc_type, text, str(file_path), self.processing_config
            )
            
            # Step 4: Validate results
            processing_result = await ResultValidator.validate_result(processing_result, doc_type)
            
            # Update metrics
            self.documents_processed += 1
            if processing_result.confidence_score >= self.processing_config.accuracy_threshold:
                self.successful_processing += 1
            
            self.total_cost += processing_result.cost_estimate
            self._update_average_accuracy(processing_result.confidence_score)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "file_path": str(file_path),
                "classification": classification_result['classification_result'],
                "processing_result": processing_result.to_dict(),
                "total_processing_time": total_time,
                "pipeline_stages": {
                    "classification_time": classification_result.get('processing_time', 0.0),
                    "processing_time": processing_result.processing_time,
                    "validation_time": 0.1  # Simplified
                },
                "cost_breakdown": {
                    "classification_cost": 0.01,  # Estimated
                    "processing_cost": processing_result.cost_estimate,
                    "total_cost": 0.01 + processing_result.cost_estimate
                }
            }
            
            logger.info(f"Document processed successfully: {file_path.name} -> {doc_type.value} "
                       f"(accuracy: {processing_result.confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": str(file_path) if 'file_path' in locals() else None
            }
    
    async def process_document_text(self, text: str) -> Dict[str, Any]:
        """Process document from text content"""
        try:
            start_time = datetime.now()
            
            logger.info("Processing document from text")
            
            # Step 1: Classify document
            classification_result = await self.classifier.classify_document_text(text)
            
            if not classification_result['success']:
                raise RuntimeError(f"Classification failed: {classification_result.get('error')}")
            
            doc_type = DocumentType(classification_result['classification_result']['document_type'])
            
            if doc_type == DocumentType.UNKNOWN:
                raise RuntimeError("Document type could not be determined")
            
            # Step 2: Process with specialized processors
            processing_result = await self.competitive_processor.process_competitively(
                doc_type, text, None, self.processing_config
            )
            
            # Step 3: Validate results
            processing_result = await ResultValidator.validate_result(processing_result, doc_type)
            
            # Update metrics
            self.documents_processed += 1
            if processing_result.confidence_score >= self.processing_config.accuracy_threshold:
                self.successful_processing += 1
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                "success": True,
                "classification": classification_result['classification_result'],
                "processing_result": processing_result.to_dict(),
                "total_processing_time": total_time
            }
            
            logger.info(f"Text processed successfully -> {doc_type.value} "
                       f"(accuracy: {processing_result.confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def register_specialized_processor(self, document_type: DocumentType, processor: BaseAgent) -> None:
        """Register a specialized processor for a document type"""
        self.processor_registry.register_processor(document_type, processor)
        logger.info(f"Registered {processor.name} for {document_type.value}")
    
    async def batch_process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Process multiple documents with optimized batching"""
        logger.info(f"Batch processing {len(file_paths)} documents")
        
        # Classify documents first to group by type
        classification_results = await self.classifier.batch_classify_documents(file_paths)
        
        # Group documents by type for optimized processing
        type_groups = {}
        for i, result in enumerate(classification_results):
            if result['success']:
                doc_type = result['classification_result']['document_type']
                if doc_type not in type_groups:
                    type_groups[doc_type] = []
                type_groups[doc_type].append((file_paths[i], result))
        
        # Process each group with specialized processors
        all_results = []
        for doc_type, files_and_results in type_groups.items():
            logger.info(f"Processing {len(files_and_results)} {doc_type} documents")
            
            # Process files of same type in parallel
            semaphore = asyncio.Semaphore(3)
            
            async def process_with_semaphore(file_path):
                async with semaphore:
                    return await self.process_document_file(file_path)
            
            batch_results = await asyncio.gather(
                *[process_with_semaphore(fp) for fp, _ in files_and_results],
                return_exceptions=True
            )
            
            all_results.extend(batch_results)
        
        return all_results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        success_rate = self.successful_processing / max(1, self.documents_processed)
        
        return {
            "documents_processed": self.documents_processed,
            "successful_processing": self.successful_processing,
            "success_rate": success_rate,
            "average_accuracy": self.average_accuracy,
            "total_cost": self.total_cost,
            "average_cost_per_document": self.total_cost / max(1, self.documents_processed),
            "target_accuracy": self.processing_config.accuracy_threshold,
            "meets_accuracy_target": self.average_accuracy >= self.processing_config.accuracy_threshold,
            "processor_performance": self.processor_registry.processor_performance,
            "registered_processors": {
                doc_type.value: len(processors) 
                for doc_type, processors in self.processor_registry.processors.items()
            }
        }
    
    def _update_average_accuracy(self, new_accuracy: float) -> None:
        """Update rolling average accuracy"""
        if self.documents_processed == 1:
            self.average_accuracy = new_accuracy
        else:
            # Exponential moving average
            alpha = 0.1
            self.average_accuracy = (1 - alpha) * self.average_accuracy + alpha * new_accuracy


# Factory function
def create_multi_domain_processor(config: Optional[Dict[str, Any]] = None) -> MultiDomainProcessorAgent:
    """Factory function to create multi-domain processor"""
    return MultiDomainProcessorAgent(config=config)


# Example usage
async def main():
    """Example usage of multi-domain processor"""
    processor = create_multi_domain_processor()
    
    # Sample invoice text
    invoice_text = """
    INVOICE #INV-001
    Date: 2024-01-15
    
    From: ACME Corp
    123 Business St
    
    To: Client Corp
    456 Main St
    
    Services: Consulting
    Amount: $2,500.00
    Tax: $250.00
    Total: $2,750.00
    """
    
    result = await processor.process_document_text(invoice_text)
    print(f"Processing result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    asyncio.run(main())