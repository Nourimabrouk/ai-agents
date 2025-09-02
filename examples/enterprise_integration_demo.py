"""
Enterprise AI Document Intelligence Platform - Integration Demo
Demonstrates the complete integration of multi-domain processing, 
advanced coordination, and enterprise features.

This example shows how to:
1. Process multiple document types using competitive agent selection
2. Apply meta-learning for optimal coordination
3. Integrate with enterprise systems (QuickBooks example)
4. Implement human-in-the-loop workflows
5. Generate business intelligence and ROI metrics

Built on Windows development environment with budget-conscious approach.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

# Import existing framework components
from core.orchestration.orchestrator import AgentOrchestrator, Task
from core.coordination.advanced_orchestrator import (
    AdvancedOrchestrator, 
    CompetitiveResult, 
    CoordinationPattern
)
from templates.base_agent import BaseAgent
from agents.accountancy.invoice_processor import InvoiceProcessorAgent
from utils.observability.logging import get_logger

# Demo-specific imports (would be actual implementations)
from examples.demo_agents.purchase_order_agent import PurchaseOrderAgent
from examples.demo_agents.contract_analyzer import ContractAnalyzerAgent
from examples.demo_agents.document_classifier import DocumentClassifierAgent
from examples.demo_integrations.quickbooks_connector import QuickBooksConnector
from examples.demo_analytics.roi_calculator import ROICalculator

logger = get_logger(__name__)

@dataclass
class ProcessingRequest:
    """Enterprise document processing request"""
    document_path: str
    organization_id: int
    user_id: str
    priority: int = 2  # 1=low, 2=medium, 3=high, 4=critical
    processing_options: Optional[Dict[str, Any]] = None
    webhook_url: Optional[str] = None
    business_context: Optional[Dict[str, Any]] = None


@dataclass
class ProcessingResult:
    """Complete processing result with enterprise metadata"""
    request_id: str
    document_type: str
    extracted_data: Dict[str, Any]
    confidence_score: float
    processing_method: str
    processing_time_ms: int
    validation_errors: List[str]
    business_impact: Dict[str, Any]
    integration_results: Dict[str, Any]
    requires_human_review: bool
    cost_estimate: float


class EnterprisePlatformDemo:
    """
    Comprehensive demo of enterprise platform capabilities
    Showcases real-world business value and technical sophistication
    """
    
    def __init__(self):
        # Initialize orchestrators
        self.basic_orchestrator = AgentOrchestrator("enterprise_basic")
        self.advanced_orchestrator = AdvancedOrchestrator("enterprise_advanced")
        
        # Initialize specialized agents
        self.document_classifier = DocumentClassifierAgent("doc_classifier")
        self.invoice_processor = InvoiceProcessorAgent("invoice_agent")
        self.po_processor = PurchaseOrderAgent("po_agent")
        self.contract_analyzer = ContractAnalyzerAgent("contract_agent")
        
        # Register agents with orchestrators
        self._register_agents()
        
        # Initialize enterprise integrations
        self.quickbooks_connector = QuickBooksConnector()
        self.roi_calculator = ROICalculator()
        
        # Performance tracking
        self.processing_metrics = {
            'total_processed': 0,
            'total_cost': 0.0,
            'total_time_saved': 0.0,
            'accuracy_scores': [],
            'processing_times': []
        }
        
        logger.info("Enterprise Platform Demo initialized")
    
    def _register_agents(self):
        """Register all agents with orchestrators"""
        agents = [
            self.document_classifier,
            self.invoice_processor, 
            self.po_processor,
            self.contract_analyzer
        ]
        
        for agent in agents:
            self.basic_orchestrator.register_agent(agent)
            self.advanced_orchestrator.register_agent(agent)
    
    async def process_enterprise_document(
        self, 
        request: ProcessingRequest
    ) -> ProcessingResult:
        """
        Complete enterprise document processing workflow
        Demonstrates all platform capabilities in realistic scenario
        """
        start_time = datetime.now()
        request_id = f"req_{int(start_time.timestamp())}"
        
        logger.info(f"Processing enterprise document: {request.document_path}")
        
        try:
            # Step 1: Document Classification
            classification_result = await self._classify_document(request)
            document_type = classification_result['document_type']
            confidence = classification_result['confidence']
            
            logger.info(f"Document classified as: {document_type} (confidence: {confidence:.2f})")
            
            # Step 2: Select Processing Strategy Based on Business Rules
            processing_strategy = await self._select_processing_strategy(
                document_type, request.priority, confidence
            )
            
            logger.info(f"Selected processing strategy: {processing_strategy}")
            
            # Step 3: Execute Processing with Selected Strategy
            processing_result = await self._execute_processing_strategy(
                request, document_type, processing_strategy
            )
            
            # Step 4: Validate and Enrich Results
            validated_result = await self._validate_and_enrich_results(
                processing_result, request.business_context
            )
            
            # Step 5: Enterprise System Integration
            integration_results = await self._integrate_with_enterprise_systems(
                validated_result, request.organization_id
            )
            
            # Step 6: Human-in-the-Loop Decision
            requires_review = await self._evaluate_human_review_requirement(
                validated_result, request.processing_options
            )
            
            # Step 7: Business Impact Analysis
            business_impact = await self._calculate_business_impact(
                validated_result, processing_result['processing_time_ms']
            )
            
            # Step 8: Generate Final Result
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            
            final_result = ProcessingResult(
                request_id=request_id,
                document_type=document_type,
                extracted_data=validated_result['extracted_data'],
                confidence_score=validated_result['confidence_score'],
                processing_method=processing_result['method_used'],
                processing_time_ms=int(total_time),
                validation_errors=validated_result.get('validation_errors', []),
                business_impact=business_impact,
                integration_results=integration_results,
                requires_human_review=requires_review,
                cost_estimate=processing_result.get('cost_estimate', 0.0)
            )
            
            # Update metrics
            await self._update_performance_metrics(final_result)
            
            # Send webhook notification if configured
            if request.webhook_url:
                await self._send_webhook_notification(request.webhook_url, final_result)
            
            logger.info(f"Document processing completed: {request_id} in {total_time:.0f}ms")
            return final_result
            
        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            # Return error result
            return ProcessingResult(
                request_id=request_id,
                document_type="unknown",
                extracted_data={},
                confidence_score=0.0,
                processing_method="error",
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                validation_errors=[str(e)],
                business_impact={},
                integration_results={},
                requires_human_review=True,
                cost_estimate=0.0
            )
    
    async def _classify_document(self, request: ProcessingRequest) -> Dict[str, Any]:
        """Step 1: Intelligent document classification"""
        
        # Create classification task
        classification_task = Task(
            id=f"classify_{int(datetime.now().timestamp())}",
            description=f"Classify document: {request.document_path}",
            requirements={
                'file_path': request.document_path,
                'organization_context': request.business_context
            }
        )
        
        # Use document classifier agent
        result = await self.document_classifier.process_task(
            classification_task.description,
            classification_task.requirements
        )
        
        return {
            'document_type': result.get('document_type', 'unknown'),
            'confidence': result.get('confidence', 0.5),
            'metadata': result.get('metadata', {})
        }
    
    async def _select_processing_strategy(
        self, 
        document_type: str, 
        priority: int, 
        confidence: float
    ) -> str:
        """Step 2: Select optimal processing strategy based on business rules"""
        
        # High priority documents get competitive processing
        if priority >= 3:
            return "competitive_agents"
        
        # Low confidence classifications get multi-agent validation
        if confidence < 0.7:
            return "consensus_validation"
        
        # Complex documents benefit from meta-learning
        if document_type in ['contract', 'legal_document']:
            return "meta_learning"
        
        # Standard documents use single best agent
        return "single_agent_optimized"
    
    async def _execute_processing_strategy(
        self, 
        request: ProcessingRequest, 
        document_type: str, 
        strategy: str
    ) -> Dict[str, Any]:
        """Step 3: Execute selected processing strategy"""
        
        # Create processing task
        processing_task = Task(
            id=f"process_{int(datetime.now().timestamp())}",
            description=f"Process {document_type} document with {strategy} strategy",
            requirements={
                'file_path': request.document_path,
                'document_type': document_type,
                'organization_id': request.organization_id,
                'priority': request.priority
            }
        )
        
        if strategy == "competitive_agents":
            # Use competitive agent selection
            candidate_agents = await self._select_agents_for_document_type(document_type)
            result = await self.advanced_orchestrator.competitive_agent_selection(
                processing_task, 
                candidate_agents,
                selection_criteria="highest_confidence"
            )
            
            return {
                'extracted_data': result.result,
                'confidence_score': result.confidence_score,
                'method_used': f"competitive_{result.agent_name}",
                'processing_time_ms': result.processing_time_ms,
                'cost_estimate': result.cost_estimate
            }
            
        elif strategy == "consensus_validation":
            # Use consensus-based processing
            result = await self.advanced_orchestrator.consensus_execution(
                await self._select_agents_for_document_type(document_type),
                processing_task
            )
            
            return {
                'extracted_data': result,
                'confidence_score': 0.8,  # Consensus typically high confidence
                'method_used': "consensus_validation",
                'processing_time_ms': 5000,  # Estimated
                'cost_estimate': 0.02
            }
            
        elif strategy == "meta_learning":
            # Use meta-learning coordinator
            result = await self.advanced_orchestrator.meta_learning_coordinator(processing_task)
            
            return {
                'extracted_data': result,
                'confidence_score': 0.85,  # Meta-learning optimized
                'method_used': "meta_learning",
                'processing_time_ms': 3000,  # Faster due to learning
                'cost_estimate': 0.01
            }
            
        else:  # single_agent_optimized
            # Use best single agent for document type
            best_agent = await self._get_best_agent_for_type(document_type)
            result = await best_agent.process_task(
                processing_task.description,
                processing_task.requirements
            )
            
            return {
                'extracted_data': result,
                'confidence_score': getattr(result, 'confidence_score', 0.75),
                'method_used': f"single_agent_{best_agent.name}",
                'processing_time_ms': 2000,  # Fastest method
                'cost_estimate': 0.005
            }
    
    async def _validate_and_enrich_results(
        self, 
        processing_result: Dict[str, Any], 
        business_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Step 4: Validate extracted data and enrich with business context"""
        
        extracted_data = processing_result['extracted_data']
        validation_errors = []
        
        # Basic validation
        if not extracted_data:
            validation_errors.append("No data extracted from document")
        
        # Business context enrichment
        if business_context:
            # Add vendor information if available
            if 'vendor_database' in business_context:
                vendor_name = extracted_data.get('vendor_name')
                if vendor_name:
                    vendor_info = business_context['vendor_database'].get(vendor_name.lower())
                    if vendor_info:
                        extracted_data['vendor_enriched'] = vendor_info
        
        # Confidence adjustment based on validation
        original_confidence = processing_result['confidence_score']
        adjusted_confidence = original_confidence
        
        if validation_errors:
            adjusted_confidence = max(0.1, original_confidence - 0.2)
        
        return {
            'extracted_data': extracted_data,
            'confidence_score': adjusted_confidence,
            'validation_errors': validation_errors,
            'enrichment_applied': bool(business_context)
        }
    
    async def _integrate_with_enterprise_systems(
        self, 
        validated_result: Dict[str, Any], 
        organization_id: int
    ) -> Dict[str, Any]:
        """Step 5: Integrate with enterprise systems (QuickBooks example)"""
        
        integration_results = {}
        extracted_data = validated_result['extracted_data']
        
        try:
            # QuickBooks integration for invoices and bills
            if 'invoice_number' in extracted_data or 'total_amount' in extracted_data:
                
                qb_result = await self.quickbooks_connector.create_bill_or_invoice(
                    organization_id=organization_id,
                    document_data=extracted_data
                )
                
                integration_results['quickbooks'] = {
                    'success': True,
                    'transaction_id': qb_result.get('id'),
                    'sync_status': 'completed'
                }
                
        except Exception as e:
            integration_results['quickbooks'] = {
                'success': False,
                'error': str(e),
                'sync_status': 'failed'
            }
        
        return integration_results
    
    async def _evaluate_human_review_requirement(
        self, 
        validated_result: Dict[str, Any], 
        processing_options: Optional[Dict[str, Any]]
    ) -> bool:
        """Step 6: Determine if human review is required"""
        
        # Force review if explicitly requested
        if processing_options and processing_options.get('force_human_review'):
            return True
        
        # Review if confidence is low
        if validated_result['confidence_score'] < 0.7:
            return True
        
        # Review if validation errors exist
        if validated_result.get('validation_errors'):
            return True
        
        # Review if high-value transaction (business rule)
        extracted_data = validated_result['extracted_data']
        if 'total_amount' in extracted_data:
            try:
                amount = float(str(extracted_data['total_amount']).replace(',', '').replace('$', ''))
                if amount > 10000:  # Review transactions over $10,000
                    return True
            except (ValueError, TypeError):
                pass
        
        return False
    
    async def _calculate_business_impact(
        self, 
        validated_result: Dict[str, Any], 
        processing_time_ms: int
    ) -> Dict[str, Any]:
        """Step 7: Calculate business impact and ROI metrics"""
        
        # Estimate manual processing time (baseline)
        manual_processing_time_minutes = 15  # Average 15 minutes manual processing
        automated_processing_time_minutes = processing_time_ms / 60000
        
        time_saved_minutes = manual_processing_time_minutes - automated_processing_time_minutes
        time_saved_hours = time_saved_minutes / 60
        
        # Calculate cost savings (assuming $25/hour for manual processing)
        hourly_rate = 25.0
        cost_savings = time_saved_hours * hourly_rate
        
        # Calculate accuracy improvement (estimated)
        manual_accuracy = 0.92  # Typical manual processing accuracy
        automated_accuracy = validated_result['confidence_score']
        accuracy_improvement = automated_accuracy - manual_accuracy
        
        return {
            'time_saved_minutes': round(time_saved_minutes, 2),
            'cost_savings_usd': round(cost_savings, 2),
            'accuracy_improvement': round(accuracy_improvement, 3),
            'efficiency_gain_percentage': round((time_saved_minutes / manual_processing_time_minutes) * 100, 1),
            'roi_calculation': {
                'manual_cost': round(manual_processing_time_minutes / 60 * hourly_rate, 2),
                'automated_cost': round(processing_time_ms / 60000 / 60 * hourly_rate + 0.01, 2),  # Include AI costs
                'net_savings': round(cost_savings - 0.01, 2)
            }
        }
    
    async def _update_performance_metrics(self, result: ProcessingResult):
        """Update platform performance metrics"""
        self.processing_metrics['total_processed'] += 1
        self.processing_metrics['total_cost'] += result.cost_estimate
        self.processing_metrics['total_time_saved'] += result.business_impact.get('time_saved_minutes', 0)
        self.processing_metrics['accuracy_scores'].append(result.confidence_score)
        self.processing_metrics['processing_times'].append(result.processing_time_ms)
    
    async def _send_webhook_notification(self, webhook_url: str, result: ProcessingResult):
        """Send webhook notification for completed processing"""
        import aiohttp
        
        webhook_payload = {
            'event': 'document.processed',
            'request_id': result.request_id,
            'document_type': result.document_type,
            'confidence_score': result.confidence_score,
            'processing_time_ms': result.processing_time_ms,
            'requires_human_review': result.requires_human_review,
            'business_impact': result.business_impact,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=webhook_payload) as response:
                    if response.status == 200:
                        logger.info(f"Webhook notification sent successfully to {webhook_url}")
                    else:
                        logger.warning(f"Webhook notification failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook notification error: {e}")
    
    # Helper methods
    async def _select_agents_for_document_type(self, document_type: str) -> List[BaseAgent]:
        """Select appropriate agents for document type"""
        if document_type == 'invoice':
            return [self.invoice_processor]
        elif document_type == 'purchase_order':
            return [self.po_processor, self.invoice_processor]  # PO processor + fallback
        elif document_type == 'contract':
            return [self.contract_analyzer]
        else:
            return [self.invoice_processor, self.po_processor]  # General processing agents
    
    async def _get_best_agent_for_type(self, document_type: str) -> BaseAgent:
        """Get single best agent for document type"""
        agents = await self._select_agents_for_document_type(document_type)
        return agents[0] if agents else self.invoice_processor
    
    async def generate_platform_analytics(self) -> Dict[str, Any]:
        """Generate comprehensive platform analytics"""
        
        metrics = self.processing_metrics
        
        # Calculate averages
        avg_accuracy = sum(metrics['accuracy_scores']) / max(len(metrics['accuracy_scores']), 1)
        avg_processing_time = sum(metrics['processing_times']) / max(len(metrics['processing_times']), 1)
        
        # ROI calculations
        total_roi = await self.roi_calculator.calculate_platform_roi(
            documents_processed=metrics['total_processed'],
            total_time_saved=metrics['total_time_saved'],
            total_cost=metrics['total_cost']
        )
        
        return {
            'performance_metrics': {
                'total_documents_processed': metrics['total_processed'],
                'average_accuracy': round(avg_accuracy, 3),
                'average_processing_time_ms': round(avg_processing_time, 0),
                'total_cost_usd': round(metrics['total_cost'], 2),
                'total_time_saved_minutes': round(metrics['total_time_saved'], 1)
            },
            'roi_analysis': total_roi,
            'efficiency_metrics': {
                'documents_per_hour': round(metrics['total_processed'] * 60 * 60000 / max(sum(metrics['processing_times']), 1), 1),
                'cost_per_document': round(metrics['total_cost'] / max(metrics['total_processed'], 1), 4),
                'accuracy_trend': 'improving' if len(metrics['accuracy_scores']) >= 2 and metrics['accuracy_scores'][-1] > metrics['accuracy_scores'][0] else 'stable'
            }
        }


# Demo Usage Example
async def run_enterprise_demo():
    """
    Demonstrate complete enterprise platform capabilities
    Process different document types and show business value
    """
    
    platform = EnterprisePlatformDemo()
    
    # Sample documents for demo (would be actual files in real usage)
    demo_requests = [
        ProcessingRequest(
            document_path="demo_documents/invoice_001.pdf",
            organization_id=1,
            user_id="demo_user",
            priority=2,
            business_context={
                'vendor_database': {
                    'acme corp': {'vendor_id': 'V001', 'payment_terms': 'Net 30'},
                    'office supplies inc': {'vendor_id': 'V002', 'payment_terms': 'Net 15'}
                }
            }
        ),
        ProcessingRequest(
            document_path="demo_documents/purchase_order_002.pdf", 
            organization_id=1,
            user_id="demo_user",
            priority=3,  # High priority - will use competitive processing
            webhook_url="https://webhook.site/demo-endpoint"
        ),
        ProcessingRequest(
            document_path="demo_documents/contract_003.pdf",
            organization_id=1, 
            user_id="demo_user",
            priority=2,
            processing_options={'force_human_review': False}
        )
    ]
    
    print("🚀 Starting Enterprise AI Document Intelligence Platform Demo")
    print("=" * 60)
    
    # Process each document
    results = []
    for i, request in enumerate(demo_requests, 1):
        print(f"\n📄 Processing Document {i}: {Path(request.document_path).name}")
        print("-" * 40)
        
        result = await platform.process_enterprise_document(request)
        results.append(result)
        
        # Display result summary
        print(f"✅ Processing completed:")
        print(f"   Document Type: {result.document_type}")
        print(f"   Confidence: {result.confidence_score:.1%}")
        print(f"   Processing Time: {result.processing_time_ms:,}ms")
        print(f"   Method: {result.processing_method}")
        print(f"   Cost: ${result.cost_estimate:.4f}")
        print(f"   Time Saved: {result.business_impact.get('time_saved_minutes', 0):.1f} minutes")
        print(f"   Cost Savings: ${result.business_impact.get('cost_savings_usd', 0):.2f}")
        print(f"   Human Review Required: {'Yes' if result.requires_human_review else 'No'}")
        
        if result.validation_errors:
            print(f"   ⚠️  Validation Issues: {len(result.validation_errors)}")
    
    # Generate platform analytics
    print(f"\n📊 Platform Analytics Summary")
    print("=" * 40)
    
    analytics = await platform.generate_platform_analytics()
    perf = analytics['performance_metrics']
    roi = analytics['roi_analysis']
    
    print(f"Total Documents Processed: {perf['total_documents_processed']}")
    print(f"Average Accuracy: {perf['average_accuracy']:.1%}")
    print(f"Average Processing Time: {perf['average_processing_time_ms']:,.0f}ms")
    print(f"Total Platform Cost: ${perf['total_cost_usd']:.2f}")
    print(f"Total Time Saved: {perf['total_time_saved_minutes']:.0f} minutes")
    print(f"ROI: {roi.get('roi_percentage', 0):.0f}%")
    print(f"Cost per Document: ${analytics['efficiency_metrics']['cost_per_document']:.4f}")
    
    print(f"\n🎯 Business Impact Summary")
    print("=" * 40)
    total_savings = sum(r.business_impact.get('cost_savings_usd', 0) for r in results)
    total_time_saved = sum(r.business_impact.get('time_saved_minutes', 0) for r in results)
    
    print(f"💰 Total Cost Savings: ${total_savings:.2f}")
    print(f"⏱️  Total Time Saved: {total_time_saved:.0f} minutes ({total_time_saved/60:.1f} hours)")
    print(f"🎯 Efficiency Improvement: {(total_time_saved / (len(results) * 15)) * 100:.0f}%")
    print(f"🤖 AI Processing Cost: ${sum(r.cost_estimate for r in results):.4f}")
    print(f"📈 ROI for Demo Batch: {((total_savings - sum(r.cost_estimate for r in results)) / max(sum(r.cost_estimate for r in results), 0.01)) * 100:.0f}%")
    
    return results, analytics


# Run the demo
if __name__ == "__main__":
    asyncio.run(run_enterprise_demo())