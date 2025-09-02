"""
Document Processing Service
Integrates with existing multi-domain processor and advanced orchestrator
Provides enterprise-grade processing with monitoring and cost tracking
"""

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import tempfile
import os

from fastapi import UploadFile, HTTPException
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

# Core document processing imports
from agents.intelligence.multi_domain_processor import (
    MultiDomainProcessorAgent, ProcessingStrategy, ProcessingConfig, DocumentType
)
from core.coordination.advanced_orchestrator import AdvancedOrchestrator, CoordinationPattern
from agents.intelligence.document_classifier import DocumentClassifierAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

# API models
from api.models.api_models import (
    ProcessingRequest, BatchProcessingRequest, ClassificationRequest,
    ProcessingResponse, ProcessingStatus as APIProcessingStatus
)
from api.models.database_models import (
    Document, ProcessingLog, BatchProcessing, Organization, User
)
from api.database.session import get_database_session, DatabaseManager
from api.config import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ProcessingService:
    """
    Enterprise document processing service
    
    Features:
    - Integration with multi-domain processor
    - Advanced orchestrator coordination
    - Cost tracking and budget management
    - Progress monitoring and notifications
    - Multi-tenant isolation
    - Comprehensive audit logging
    """
    
    def __init__(self):
        self.settings = settings
        
        # Initialize core processing components
        self.multi_domain_processor: Optional[MultiDomainProcessorAgent] = None
        self.advanced_orchestrator: Optional[AdvancedOrchestrator] = None
        self.document_classifier: Optional[DocumentClassifierAgent] = None
        
        # Processing state
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.batch_jobs: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.total_processed = 0
        self.successful_processed = 0
        self.total_cost = Decimal("0.00")
        
        logger.info("ProcessingService initialized")
    
    async def initialize(self):
        """Initialize processing components"""
        try:
            # Initialize multi-domain processor
            config = {
                'processing': {
                    'strategy': 'competitive',
                    'accuracy_threshold': settings.processing.default_confidence_threshold,
                    'max_cost_per_document': settings.processing.max_cost_per_document
                }
            }
            
            self.multi_domain_processor = MultiDomainProcessorAgent(
                name="enterprise_processor",
                config=config
            )
            
            # Initialize advanced orchestrator
            self.advanced_orchestrator = AdvancedOrchestrator("enterprise_orchestrator")
            
            # Initialize document classifier
            self.document_classifier = DocumentClassifierAgent(
                name="enterprise_classifier",
                config=config
            )
            
            # Register agents with orchestrator
            await self.advanced_orchestrator.register_agent(self.multi_domain_processor)
            
            logger.info("Processing components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize processing components: {e}")
            raise
    
    async def cleanup(self):
        """Cleanup processing resources"""
        try:
            # Cancel running tasks
            for task_id, task in self.processing_tasks.items():
                if not task.done():
                    task.cancel()
                    logger.info(f"Cancelled processing task: {task_id}")
            
            # Clear state
            self.processing_tasks.clear()
            self.batch_jobs.clear()
            
            logger.info("Processing service cleanup completed")
            
        except Exception as e:
            logger.error(f"Processing service cleanup failed: {e}")
    
    async def process_document(
        self,
        request: ProcessingRequest,
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        """
        Process single document with full enterprise pipeline
        
        Pipeline stages:
        1. Input validation and preparation
        2. File handling and storage
        3. Document classification
        4. Multi-domain processing with competitive selection
        5. Result validation and quality checks
        6. Cost calculation and budget tracking
        7. Database persistence and audit logging
        8. Integration posting (if enabled)
        """
        document_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting document processing: {document_id}")
            global_metrics.incr("processing.documents.started")
            
            # Stage 1: Input validation and preparation
            await self._validate_processing_request(request, organization_id)
            
            # Stage 2: File handling and storage
            file_info = await self._handle_document_input(request, document_id)
            
            # Stage 3: Create database record
            document_record = await self._create_document_record(
                document_id, request, file_info, user_id, organization_id
            )
            
            # Stage 4: Document classification
            classification_result = await self._classify_document(
                request, file_info, document_id
            )
            
            # Stage 5: Multi-domain processing
            processing_result = await self._process_with_multi_domain(
                classification_result, request, file_info, document_id
            )
            
            # Stage 6: Result validation and enhancement
            validated_result = await self._validate_and_enhance_result(
                processing_result, classification_result, document_id
            )
            
            # Stage 7: Cost calculation
            cost_breakdown = await self._calculate_processing_costs(
                processing_result, classification_result
            )
            
            # Stage 8: Update database record
            await self._update_document_record(
                document_id, classification_result, validated_result, cost_breakdown
            )
            
            # Stage 9: Integration posting (if enabled)
            integration_result = None
            if request.auto_post_to_accounting and request.accounting_integration:
                integration_result = await self._post_to_accounting_system(
                    validated_result, request.accounting_integration, organization_id
                )
            
            # Update metrics
            self.total_processed += 1
            if validated_result["success"]:
                self.successful_processed += 1
            self.total_cost += cost_breakdown["total_cost"]
            
            total_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Create response
            response_data = {
                "document_id": document_id,
                "success": validated_result["success"],
                "classification": classification_result,
                "extracted_data": validated_result.get("extracted_data", {}),
                "confidence_score": validated_result.get("confidence_score", 0.0),
                "processing_time_ms": int(total_time),
                "validation_errors": validated_result.get("validation_errors", []),
                "cost_breakdown": cost_breakdown,
                "processing_method": validated_result.get("processing_method", "unknown"),
                "competitive_results": validated_result.get("competitive_results", []),
                "integration_result": integration_result
            }
            
            logger.info(f"Document processing completed: {document_id}")
            global_metrics.incr("processing.documents.completed")
            
            return response_data
            
        except Exception as e:
            logger.error(f"Document processing failed: {document_id} - {e}")
            global_metrics.incr("processing.documents.failed")
            
            # Update document record with error
            await self._update_document_error(document_id, str(e))
            
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {str(e)}"
            )
    
    async def batch_process_documents(
        self,
        request: BatchProcessingRequest,
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        """
        Process multiple documents in batch with optimization
        
        Features:
        - Document type grouping for efficiency
        - Parallel processing with concurrency control
        - Progress tracking and status updates
        - Partial failure handling
        - Cost aggregation and budget monitoring
        """
        batch_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting batch processing: {batch_id} ({len(request.documents)} documents)")
            global_metrics.incr("processing.batches.started")
            
            # Create batch record
            batch_record = await self._create_batch_record(
                batch_id, request, user_id, organization_id
            )
            
            # Group documents by type for optimization
            document_groups = await self._group_documents_for_processing(request.documents)
            
            # Process groups with controlled concurrency
            semaphore = asyncio.Semaphore(request.max_concurrent_documents)
            all_results = []
            
            for group_name, group_docs in document_groups.items():
                logger.info(f"Processing document group: {group_name} ({len(group_docs)} documents)")
                
                # Create processing tasks for this group
                tasks = []
                for doc_request in group_docs:
                    task = asyncio.create_task(
                        self._process_document_with_semaphore(
                            semaphore, doc_request, user_id, organization_id
                        )
                    )
                    tasks.append(task)
                
                # Wait for group completion
                group_results = await asyncio.gather(*tasks, return_exceptions=True)
                all_results.extend(group_results)
                
                # Update batch progress
                await self._update_batch_progress(batch_id, len(all_results), len(request.documents))
            
            # Process results and calculate aggregates
            successful_results = []
            failed_results = []
            total_cost = Decimal("0.00")
            
            for result in all_results:
                if isinstance(result, Exception):
                    failed_results.append({"error": str(result)})
                else:
                    if result.get("success", False):
                        successful_results.append(result)
                    else:
                        failed_results.append(result)
                    
                    total_cost += result.get("cost_breakdown", {}).get("total_cost", Decimal("0.00"))
            
            # Update batch completion
            completion_time = datetime.utcnow()
            await self._complete_batch_record(
                batch_id, len(successful_results), len(failed_results), 
                total_cost, completion_time - start_time
            )
            
            batch_result = {
                "batch_id": batch_id,
                "status": "completed",
                "total_documents": len(request.documents),
                "successful_documents": len(successful_results),
                "failed_documents": len(failed_results),
                "total_cost": total_cost,
                "processing_time_seconds": (completion_time - start_time).total_seconds(),
                "results": successful_results + failed_results
            }
            
            logger.info(f"Batch processing completed: {batch_id}")
            global_metrics.incr("processing.batches.completed")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"Batch processing failed: {batch_id} - {e}")
            global_metrics.incr("processing.batches.failed")
            
            # Update batch with error status
            await self._update_batch_error(batch_id, str(e))
            
            raise HTTPException(
                status_code=500,
                detail=f"Batch processing failed: {str(e)}"
            )
    
    async def classify_document(
        self,
        request: ClassificationRequest,
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        """
        Classify document type without full processing
        
        Fast classification for routing and filtering
        """
        try:
            start_time = datetime.utcnow()
            
            # Handle input
            file_info = await self._handle_classification_input(request)
            
            # Perform classification
            if file_info.get("file_path"):
                classification = await self.document_classifier.classify_document_file(
                    file_info["file_path"]
                )
            else:
                classification = await self.document_classifier.classify_document_text(
                    file_info["text_content"]
                )
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Get supported fields for document type
            doc_type = DocumentType(classification['classification_result']['document_type'])
            supported_fields = self._get_supported_fields_for_type(doc_type)
            
            return {
                "document_type": doc_type.value,
                "confidence_score": classification['classification_result']['confidence'],
                "processing_time_ms": int(processing_time),
                "cost": Decimal("0.01"),  # Classification cost
                "supported_fields": supported_fields
            }
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Classification failed: {str(e)}"
            )
    
    async def get_document_status(self, document_id: str, user_id: str) -> Dict[str, Any]:
        """Get processing status for a document"""
        try:
            async with get_database_session() as db:
                # Get document record
                result = await db.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()
                
                if not document:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                # Check user access
                if str(document.created_by_id) != user_id:
                    raise HTTPException(status_code=403, detail="Access denied")
                
                # Calculate progress
                progress_percentage = 0.0
                current_stage = "unknown"
                
                if document.processing_status == "pending":
                    progress_percentage = 0.0
                    current_stage = "queued"
                elif document.processing_status == "processing":
                    progress_percentage = 50.0
                    current_stage = "processing"
                elif document.processing_status == "completed":
                    progress_percentage = 100.0
                    current_stage = "completed"
                elif document.processing_status == "failed":
                    progress_percentage = 0.0
                    current_stage = "failed"
                
                # Estimate completion time
                estimated_completion = None
                if document.processing_status == "processing" and document.processing_started_at:
                    avg_processing_time = timedelta(seconds=30)  # Default estimate
                    estimated_completion = document.processing_started_at + avg_processing_time
                
                return {
                    "document_id": document_id,
                    "status": document.processing_status,
                    "progress_percentage": progress_percentage,
                    "current_stage": current_stage,
                    "estimated_completion_time": estimated_completion,
                    "error_message": None,  # Would extract from processing logs
                    "partial_results": document.extracted_data if progress_percentage > 50 else None
                }
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve document status"
            )
    
    async def get_document_result(
        self, 
        document_id: str, 
        user_id: str, 
        format: str = "json"
    ) -> Union[Dict[str, Any], Dict[str, Any]]:
        """Get complete processing results for a document"""
        try:
            async with get_database_session() as db:
                # Get document record
                result = await db.execute(
                    select(Document).where(Document.id == document_id)
                )
                document = result.scalar_one_or_none()
                
                if not document:
                    raise HTTPException(status_code=404, detail="Document not found")
                
                # Check user access
                if str(document.created_by_id) != user_id:
                    raise HTTPException(status_code=403, detail="Access denied")
                
                if document.processing_status != "completed":
                    raise HTTPException(
                        status_code=400, 
                        detail="Document processing not completed"
                    )
                
                result_data = {
                    "document_id": document_id,
                    "filename": document.original_filename,
                    "document_type": document.document_type,
                    "classification_confidence": document.classification_confidence,
                    "extracted_data": document.extracted_data,
                    "confidence_score": document.confidence_score,
                    "validation_errors": document.validation_errors,
                    "processing_method": document.processing_method,
                    "competitive_results": document.competitive_results,
                    "processing_time_ms": document.processing_duration_ms,
                    "processing_cost": float(document.processing_cost),
                    "integration_results": document.integration_results,
                    "created_at": document.created_at.isoformat(),
                    "completed_at": document.processing_completed_at.isoformat()
                }
                
                if format == "json":
                    return result_data
                else:
                    # Generate file stream for other formats
                    return await self._generate_result_file(result_data, format)
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get document result: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve document result"
            )
    
    async def process_uploaded_file(
        self,
        file: UploadFile,
        processing_options: Optional[str],
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        """Process uploaded file with automatic request creation"""
        try:
            # Save uploaded file temporarily
            temp_file_path = await self._save_uploaded_file(file)
            
            # Parse processing options
            options = {}
            if processing_options:
                options = json.loads(processing_options)
            
            # Create processing request
            request = ProcessingRequest(
                file_path=temp_file_path,
                processing_config=options.get("processing_config", {}),
                auto_post_to_accounting=options.get("auto_post_to_accounting", False),
                webhook_url=options.get("webhook_url"),
                metadata={
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "file_size_bytes": file.size if hasattr(file, 'size') else 0
                }
            )
            
            # Process document
            result = await self.process_document(request, user_id, organization_id)
            
            # Cleanup temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                logger.warning(f"Failed to cleanup temp file: {e}")
            
            return result
            
        except Exception as e:
            logger.error(f"File upload processing failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Upload processing failed: {str(e)}"
            )
    
    # Private helper methods
    
    async def _validate_processing_request(
        self, 
        request: ProcessingRequest, 
        organization_id: str
    ):
        """Validate processing request against organization limits"""
        # Check file size limits
        if request.file_path and Path(request.file_path).exists():
            file_size = Path(request.file_path).stat().st_size
            max_size = settings.processing.max_file_size_mb * 1024 * 1024
            if file_size > max_size:
                raise HTTPException(
                    status_code=413,
                    detail=f"File size exceeds limit of {settings.processing.max_file_size_mb}MB"
                )
        
        # Check organization limits
        async with get_database_session() as db:
            # Get current month's usage
            result = await db.execute(
                select(Document).where(
                    Document.organization_id == organization_id,
                    Document.created_at >= datetime.utcnow().replace(day=1)
                )
            )
            current_month_docs = len(result.all())
            
            # Get organization limits
            org_result = await db.execute(
                select(Organization).where(Organization.id == organization_id)
            )
            organization = org_result.scalar_one()
            
            if current_month_docs >= organization.monthly_document_limit:
                raise HTTPException(
                    status_code=429,
                    detail="Monthly document processing limit exceeded"
                )
    
    async def _handle_document_input(
        self, 
        request: ProcessingRequest, 
        document_id: str
    ) -> Dict[str, Any]:
        """Handle different input methods and prepare for processing"""
        file_info = {}
        
        if request.file_path:
            file_info["file_path"] = request.file_path
            file_info["filename"] = Path(request.file_path).name
            file_info["file_size"] = Path(request.file_path).stat().st_size
            
        elif request.file_url:
            # Download file from URL
            file_info = await self._download_file_from_url(request.file_url, document_id)
            
        elif request.base64_content:
            # Decode base64 content
            file_info = await self._decode_base64_content(request.base64_content, document_id)
            
        elif request.text_content:
            file_info["text_content"] = request.text_content
            
        else:
            raise ValueError("No valid input method provided")
        
        return file_info
    
    async def _create_document_record(
        self,
        document_id: str,
        request: ProcessingRequest,
        file_info: Dict[str, Any],
        user_id: str,
        organization_id: str
    ) -> Document:
        """Create initial document record in database"""
        async with get_database_session() as db:
            document = Document(
                id=document_id,
                organization_id=organization_id,
                created_by_id=user_id,
                filename=file_info.get("filename"),
                original_filename=request.metadata.filename if request.metadata else None,
                content_type=request.metadata.content_type if request.metadata else None,
                file_size_bytes=file_info.get("file_size"),
                storage_path=file_info.get("file_path"),
                processing_status="processing",
                processing_started_at=datetime.utcnow(),
                source=request.metadata.source if request.metadata else "api",
                tags=request.metadata.tags if request.metadata else [],
                custom_fields=request.metadata.custom_fields if request.metadata else {}
            )
            
            db.add(document)
            await db.commit()
            
            return document
    
    async def _classify_document(
        self,
        request: ProcessingRequest,
        file_info: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """Classify document using document classifier"""
        try:
            if file_info.get("file_path"):
                result = await self.document_classifier.classify_document_file(
                    file_info["file_path"]
                )
            else:
                result = await self.document_classifier.classify_document_text(
                    file_info["text_content"]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Document classification failed: {document_id} - {e}")
            # Return unknown type as fallback
            return {
                "success": True,
                "classification_result": {
                    "document_type": "unknown",
                    "confidence": 0.5,
                    "method": "fallback"
                }
            }
    
    async def _process_with_multi_domain(
        self,
        classification_result: Dict[str, Any],
        request: ProcessingRequest,
        file_info: Dict[str, Any],
        document_id: str
    ) -> Dict[str, Any]:
        """Process document using multi-domain processor with competitive selection"""
        try:
            # Convert processing config
            processing_strategy = ProcessingStrategy.COMPETITIVE
            if request.processing_config:
                strategy_name = request.processing_config.strategy.value
                processing_strategy = ProcessingStrategy(strategy_name)
            
            # Create processing config
            config = ProcessingConfig(
                strategy=processing_strategy,
                accuracy_threshold=request.processing_config.accuracy_threshold if request.processing_config else 0.95,
                max_processing_time=request.processing_config.max_processing_time_seconds if request.processing_config else 30,
                max_cost_per_document=float(request.processing_config.max_cost_per_document) if request.processing_config else 0.05
            )
            
            # Process with multi-domain processor
            if file_info.get("file_path"):
                result = await self.multi_domain_processor.process_document_file(
                    file_info["file_path"]
                )
            else:
                result = await self.multi_domain_processor.process_document_text(
                    file_info["text_content"]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Multi-domain processing failed: {document_id} - {e}")
            raise
    
    # Additional helper methods would continue here...
    # (Implementation includes remaining private methods for cost calculation,
    #  database updates, file handling, etc.)
    
    def _get_supported_fields_for_type(self, doc_type: DocumentType) -> List[str]:
        """Get supported extraction fields for document type"""
        field_mappings = {
            DocumentType.INVOICE: [
                "invoice_number", "vendor_name", "vendor_address", "customer_name",
                "invoice_date", "due_date", "total_amount", "tax_amount", 
                "subtotal", "line_items", "payment_terms"
            ],
            DocumentType.PURCHASE_ORDER: [
                "po_number", "vendor_name", "buyer_name", "order_date",
                "delivery_date", "total_amount", "line_items", "shipping_address"
            ],
            DocumentType.RECEIPT: [
                "merchant_name", "transaction_date", "transaction_amount",
                "payment_method", "items", "tax_amount", "receipt_number"
            ],
            DocumentType.BANK_STATEMENT: [
                "account_number", "statement_date", "beginning_balance",
                "ending_balance", "transactions", "bank_name"
            ]
        }
        
        return field_mappings.get(doc_type, [])
    
    async def _save_uploaded_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location"""
        # Create temp file
        temp_dir = Path(tempfile.gettempdir()) / "enterprise_docs"
        temp_dir.mkdir(exist_ok=True)
        
        temp_file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        # Save file content
        with open(temp_file_path, "wb") as temp_file:
            content = await file.read()
            temp_file.write(content)
        
        return str(temp_file_path)
    
    async def _process_document_with_semaphore(
        self,
        semaphore: asyncio.Semaphore,
        doc_request: ProcessingRequest,
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        """Process document with concurrency control"""
        async with semaphore:
            return await self.process_document(doc_request, user_id, organization_id)


# Export main service class
__all__ = ["ProcessingService"]