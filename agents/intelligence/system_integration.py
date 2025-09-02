"""
Multi-Domain Document Processing System Integration
Ties together all components and registers with orchestrator
Production-ready integration for seamless deployment
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

# Core orchestration
from core.orchestration.orchestrator import AgentOrchestrator, Task
from templates.base_agent import BaseAgent

# System components
from agents.intelligence.document_classifier import DocumentClassifierAgent, DocumentType
from agents.intelligence.multi_domain_processor import MultiDomainProcessorAgent, ProcessingConfig, ProcessingStrategy
from agents.intelligence.competitive_processor import CompetitiveProcessingEngine
from agents.intelligence.specialized_processors import (
    PurchaseOrderProcessor,
    ReceiptProcessor,
    BankStatementProcessor,
    ContractProcessor,
    FinancialStatementProcessor,
    LegalDocumentProcessor,
    CustomDocumentProcessor,
    create_purchase_order_processor,
    create_receipt_processor,
    create_bank_statement_processor,
    create_contract_processor,
    create_financial_statement_processor,
    create_legal_document_processor
)

# Existing foundation
from agents.accountancy.invoice_processor import InvoiceProcessorAgent

logger = logging.getLogger(__name__)


class MultiDomainSystemIntegration:
    """
    Integrates all multi-domain components into unified system
    Provides single entry point for orchestrator registration
    """
    
    def __init__(self, orchestrator: AgentOrchestrator, config: Optional[Dict[str, Any]] = None):
        self.orchestrator = orchestrator
        self.config = config or {}
        
        # Core system components
        self.classifier = None
        self.multi_domain_processor = None
        self.specialized_processors = {}
        self.invoice_processor = None  # Maintain existing capability
        
        # System metrics
        self.total_documents_processed = 0
        self.system_accuracy = 0.0
        self.cost_savings = 0.0
        
        logger.info("Initializing Multi-Domain Document Processing System Integration")
    
    async def initialize_system(self) -> None:
        """Initialize and configure the complete system"""
        try:
            logger.info("🔧 Initializing Multi-Domain Processing System...")
            
            # 1. Create document classifier
            await self._create_document_classifier()
            
            # 2. Create specialized processors
            await self._create_specialized_processors()
            
            # 3. Create multi-domain processor hub
            await self._create_multi_domain_processor()
            
            # 4. Maintain existing invoice processor
            await self._initialize_invoice_processor()
            
            # 5. Register all agents with orchestrator
            await self._register_with_orchestrator()
            
            # 6. Configure system settings
            await self._configure_system()
            
            logger.info("✅ Multi-Domain Processing System initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise
    
    async def _create_document_classifier(self) -> None:
        """Create and configure document classifier"""
        logger.info("Creating document classifier...")
        
        self.classifier = DocumentClassifierAgent(
            name="production_document_classifier",
            api_key=self.config.get('api_key'),
            config=self.config.get('classifier_config', {})
        )
        
        # Register custom document types if specified
        custom_types = self.config.get('custom_document_types', {})
        for type_name, signature in custom_types.items():
            self.classifier.register_custom_document_type(type_name, signature)
        
        logger.info(f"✓ Document classifier created: {self.classifier.name}")
    
    async def _create_specialized_processors(self) -> None:
        """Create all specialized document processors"""
        logger.info("Creating specialized processors...")
        
        api_key = self.config.get('api_key')
        processor_config = self.config.get('processor_config', {})
        
        # Create processors using factory functions for consistency
        processors = {
            DocumentType.PURCHASE_ORDER: create_purchase_order_processor(api_key, processor_config),
            DocumentType.RECEIPT: create_receipt_processor(api_key, processor_config),
            DocumentType.BANK_STATEMENT: create_bank_statement_processor(api_key, processor_config),
            DocumentType.CONTRACT: create_contract_processor(api_key, processor_config),
            DocumentType.FINANCIAL_STATEMENT: create_financial_statement_processor(api_key, processor_config),
            DocumentType.LEGAL_DOCUMENT: create_legal_document_processor(api_key, processor_config)
        }
        
        self.specialized_processors = processors
        
        logger.info(f"✓ Created {len(processors)} specialized processors")
        for doc_type, processor in processors.items():
            logger.info(f"  - {doc_type.value}: {processor.name}")
    
    async def _create_multi_domain_processor(self) -> None:
        """Create multi-domain processor hub"""
        logger.info("Creating multi-domain processor hub...")
        
        # Create processor with configuration
        self.multi_domain_processor = MultiDomainProcessorAgent(
            name="production_multi_domain_processor",
            api_key=self.config.get('api_key'),
            config=self.config.get('multi_domain_config', {})
        )
        
        # Register all specialized processors
        for doc_type, processor in self.specialized_processors.items():
            self.multi_domain_processor.register_specialized_processor(doc_type, processor)
        
        # Configure processing strategy
        strategy_config = self.config.get('processing_strategy', {})
        processing_config = ProcessingConfig(
            strategy=ProcessingStrategy(strategy_config.get('strategy', 'competitive')),
            accuracy_threshold=strategy_config.get('accuracy_threshold', 0.95),
            max_cost_per_document=strategy_config.get('max_cost_per_document', 0.05),
            enable_parallel_processing=strategy_config.get('enable_parallel_processing', True)
        )
        self.multi_domain_processor.processing_config = processing_config
        
        logger.info(f"✓ Multi-domain processor created: {self.multi_domain_processor.name}")
        logger.info(f"  - Strategy: {processing_config.strategy.value}")
        logger.info(f"  - Accuracy threshold: {processing_config.accuracy_threshold:.1%}")
        logger.info(f"  - Cost limit: ${processing_config.max_cost_per_document:.4f}")
    
    async def _initialize_invoice_processor(self) -> None:
        """Initialize existing invoice processor for backward compatibility"""
        logger.info("Initializing existing invoice processor...")
        
        self.invoice_processor = InvoiceProcessorAgent(
            name="legacy_invoice_processor",
            api_key=self.config.get('api_key'),
            config=self.config.get('invoice_config', {})
        )
        
        logger.info(f"✓ Invoice processor initialized: {self.invoice_processor.name}")
        logger.info("  - Maintains proven 95%+ accuracy")
        logger.info("  - Backward compatibility preserved")
    
    async def _register_with_orchestrator(self) -> None:
        """Register all agents with orchestrator"""
        logger.info("Registering agents with orchestrator...")
        
        # Register main processing agents
        self.orchestrator.register_agent(self.classifier)
        self.orchestrator.register_agent(self.multi_domain_processor)
        self.orchestrator.register_agent(self.invoice_processor)
        
        # Register specialized processors for direct access if needed
        for doc_type, processor in self.specialized_processors.items():
            self.orchestrator.register_agent(processor)
        
        logger.info(f"✓ Registered {2 + len(self.specialized_processors)} agents with orchestrator")
    
    async def _configure_system(self) -> None:
        """Apply final system configuration"""
        logger.info("Applying system configuration...")
        
        # Performance monitoring
        monitoring_config = self.config.get('monitoring', {})
        if monitoring_config.get('enable_detailed_logging', False):
            logging.getLogger('agents.intelligence').setLevel(logging.DEBUG)
        
        # Budget tracking
        budget_config = self.config.get('budget', {})
        if budget_config:
            logger.info("Budget tracking configured")
            logger.info(f"  - Anthropic token limit: {budget_config.get('anthropic_limit', 100000)}")
            logger.info(f"  - Cost per document target: ${budget_config.get('cost_target', 0.05)}")
        
        logger.info("✓ System configuration applied")
    
    async def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process single document through complete pipeline
        High-level interface for document processing
        """
        try:
            logger.info(f"Processing document: {Path(file_path).name}")
            
            result = await self.multi_domain_processor.process_document_file(file_path)
            
            # Update system metrics
            self._update_system_metrics(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    async def process_documents_batch(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch
        Optimized for throughput and cost efficiency
        """
        try:
            logger.info(f"Batch processing {len(file_paths)} documents")
            
            results = await self.multi_domain_processor.batch_process_documents(file_paths)
            
            # Update system metrics for all results
            for result in results:
                self._update_system_metrics(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            return [{
                "success": False,
                "error": str(e),
                "file_paths": file_paths
            }]
    
    async def classify_document(self, file_path: str) -> Dict[str, Any]:
        """
        Classify document type only
        Useful for routing decisions
        """
        try:
            result = await self.classifier.classify_document_file(file_path)
            return result
            
        except Exception as e:
            logger.error(f"Document classification failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": file_path
            }
    
    async def process_with_orchestrator(self, task_description: str, file_paths: List[str]) -> Any:
        """
        Process documents via orchestrator
        Demonstrates orchestrator integration
        """
        try:
            # Create task for orchestrator
            task = Task(
                id=f"multi_domain_task_{int(asyncio.get_event_loop().time())}",
                description=task_description,
                requirements={
                    "file_paths": file_paths,
                    "processor_type": "multi_domain",
                    "accuracy_threshold": 0.95
                }
            )
            
            # Delegate to orchestrator
            result = await self.orchestrator.delegate_task(task)
            
            return result
            
        except Exception as e:
            logger.error(f"Orchestrator processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_description": task_description
            }
    
    def _update_system_metrics(self, result: Dict[str, Any]) -> None:
        """Update system-wide metrics"""
        if result.get('success'):
            self.total_documents_processed += 1
            
            # Update accuracy tracking
            processing_result = result.get('processing_result', {})
            if isinstance(processing_result, dict):
                accuracy = processing_result.get('confidence_score', 0.0)
                if isinstance(accuracy, (int, float)):
                    # Exponential moving average
                    if self.system_accuracy == 0.0:
                        self.system_accuracy = accuracy
                    else:
                        alpha = 0.1
                        self.system_accuracy = (1 - alpha) * self.system_accuracy + alpha * accuracy
            
            # Update cost savings
            cost_breakdown = result.get('cost_breakdown', {})
            if cost_breakdown:
                system_cost = cost_breakdown.get('total_cost', 0.0)
                manual_cost = 6.15  # Estimated manual processing cost
                document_savings = manual_cost - system_cost
                self.cost_savings += document_savings
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics"""
        classifier_metrics = self.classifier.get_performance_metrics() if self.classifier else {}
        processor_metrics = self.multi_domain_processor.get_performance_metrics() if self.multi_domain_processor else {}
        
        return {
            "system_overview": {
                "total_documents_processed": self.total_documents_processed,
                "system_accuracy": self.system_accuracy,
                "cost_savings_total": self.cost_savings,
                "cost_savings_per_document": self.cost_savings / max(1, self.total_documents_processed),
                "supported_document_types": len(self.specialized_processors) + 1  # +1 for invoices
            },
            "classification_metrics": classifier_metrics,
            "processing_metrics": processor_metrics,
            "specialized_processors": {
                doc_type.value: processor.get_performance_metrics()
                for doc_type, processor in self.specialized_processors.items()
            }
        }
    
    async def shutdown(self) -> None:
        """Graceful system shutdown"""
        logger.info("Shutting down Multi-Domain Processing System...")
        
        try:
            # Unregister agents from orchestrator
            if self.classifier:
                self.orchestrator.unregister_agent(self.classifier.name)
            if self.multi_domain_processor:
                self.orchestrator.unregister_agent(self.multi_domain_processor.name)
            if self.invoice_processor:
                self.orchestrator.unregister_agent(self.invoice_processor.name)
            
            for processor in self.specialized_processors.values():
                self.orchestrator.unregister_agent(processor.name)
            
            logger.info("✅ Multi-Domain Processing System shutdown complete")
            
        except Exception as e:
            logger.error(f"❌ Shutdown error: {e}")


# Factory function for easy setup
async def create_multi_domain_system(
    orchestrator: AgentOrchestrator,
    config: Optional[Dict[str, Any]] = None
) -> MultiDomainSystemIntegration:
    """
    Factory function to create and initialize complete multi-domain system
    
    Args:
        orchestrator: Agent orchestrator instance
        config: System configuration options
        
    Returns:
        Initialized MultiDomainSystemIntegration instance
    """
    system = MultiDomainSystemIntegration(orchestrator, config)
    await system.initialize_system()
    return system


# Default configuration
DEFAULT_SYSTEM_CONFIG = {
    "processing_strategy": {
        "strategy": "competitive",
        "accuracy_threshold": 0.95,
        "max_cost_per_document": 0.05,
        "enable_parallel_processing": True
    },
    "monitoring": {
        "enable_detailed_logging": False
    },
    "budget": {
        "anthropic_limit": 100000,
        "cost_target": 0.05
    },
    "custom_document_types": {
        # Add custom document types here
        # "insurance_policy": {
        #     "required_keywords": ["policy", "premium", "coverage"],
        #     "common_patterns": [r"policy\s*#\s*([A-Z0-9\-]+)"]
        # }
    }
}


# Example usage and testing
async def main():
    """Example usage of complete system integration"""
    
    # Create orchestrator
    orchestrator = AgentOrchestrator("production_orchestrator")
    
    # Create multi-domain system with configuration
    config = DEFAULT_SYSTEM_CONFIG.copy()
    system = await create_multi_domain_system(orchestrator, config)
    
    # Example document processing
    sample_invoice = """
    ACME Corp Invoice #INV-001
    Date: 2024-01-15
    Total: $1,500.00
    """
    
    # Process single document
    result = await system.multi_domain_processor.process_document_text(sample_invoice)
    print(f"Processing result: {result}")
    
    # Get system metrics
    metrics = system.get_system_metrics()
    print(f"System metrics: {metrics}")
    
    # Cleanup
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(main())