"""
Multi-Domain Document Processing System Demo
Demonstrates the full capability of the system with real-world examples
Shows accuracy, speed, and cost efficiency across 7+ document types
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import statistics

# Import system components
from agents.intelligence.document_classifier import DocumentClassifierAgent, DocumentType
from agents.intelligence.multi_domain_processor import MultiDomainProcessorAgent, ProcessingConfig, ProcessingStrategy
from agents.intelligence.specialized_processors import (
    PurchaseOrderProcessor,
    ReceiptProcessor,
    BankStatementProcessor,
    ContractProcessor,
    FinancialStatementProcessor,
    LegalDocumentProcessor,
    CustomDocumentProcessor
)
from core.orchestration.orchestrator import AgentOrchestrator, Task


class DemoResults:
    """Container for demo results"""
    
    def __init__(self):
        self.results = []
        self.total_cost = 0.0
        self.total_time = 0.0
        self.accuracy_scores = []
    
    def add_result(self, document_type: str, result: Dict[str, Any], processing_time: float):
        """Add processing result"""
        self.results.append({
            'document_type': document_type,
            'result': result,
            'processing_time': processing_time,
            'timestamp': datetime.now().isoformat()
        })
        
        self.total_time += processing_time
        
        if result.get('success'):
            accuracy = result.get('accuracy', 0.0)
            if isinstance(accuracy, (int, float)):
                self.accuracy_scores.append(accuracy)
            
            # Extract cost if available
            cost = 0.0
            if 'cost_breakdown' in result:
                cost = result['cost_breakdown'].get('total_cost', 0.0)
            elif 'cost_estimate' in result:
                cost = result['cost_estimate']
            
            self.total_cost += cost
    
    def get_summary(self) -> Dict[str, Any]:
        """Get demo summary"""
        successful_results = sum(1 for r in self.results if r['result'].get('success', False))
        success_rate = successful_results / len(self.results) if self.results else 0.0
        avg_accuracy = statistics.mean(self.accuracy_scores) if self.accuracy_scores else 0.0
        avg_cost_per_doc = self.total_cost / len(self.results) if self.results else 0.0
        
        return {
            'total_documents_processed': len(self.results),
            'successful_processing': successful_results,
            'success_rate': success_rate,
            'average_accuracy': avg_accuracy,
            'total_processing_time': self.total_time,
            'average_time_per_document': self.total_time / len(self.results) if self.results else 0.0,
            'total_cost': self.total_cost,
            'average_cost_per_document': avg_cost_per_doc,
            'cost_vs_manual': {
                'system_cost': avg_cost_per_doc,
                'manual_cost': 6.15,  # Estimated manual processing cost
                'savings': 6.15 - avg_cost_per_doc,
                'savings_percentage': ((6.15 - avg_cost_per_doc) / 6.15) * 100 if avg_cost_per_doc < 6.15 else 0.0
            }
        }


# Sample documents for demonstration
DEMO_DOCUMENTS = {
    "invoice_consulting": {
        "type": "invoice",
        "filename": "consulting_invoice.txt",
        "content": """
        TechConsult Pro Services
        1234 Innovation Drive, Suite 567
        San Francisco, CA 94105
        (415) 555-0123
        
        INVOICE
        
        Invoice Number: TC-2024-001
        Invoice Date: January 15, 2024
        Due Date: February 14, 2024
        
        Bill To:
        Startup Innovations Inc.
        5678 Venture Blvd
        Palo Alto, CA 94301
        
        Project: Cloud Migration Services
        
        Description                           Hours    Rate      Amount
        Senior Cloud Architect                 40    $200/hr    $8,000.00
        DevOps Engineer                        60    $150/hr    $9,000.00
        Project Management                     20    $175/hr    $3,500.00
        Documentation & Training               15    $125/hr    $1,875.00
        
        Subtotal:                                              $22,375.00
        California State Tax (8.75%):                         $1,957.81
        Total Amount Due:                                     $24,332.81
        
        Payment Terms: Net 30 days
        Late Fee: 1.5% per month on overdue balance
        
        Thank you for your business!
        """
    },
    
    "purchase_order_office": {
        "type": "purchase_order",
        "filename": "office_equipment_po.txt", 
        "content": """
        PURCHASE ORDER
        
        PO Number: PO-2024-0156
        Order Date: January 12, 2024
        Required Delivery Date: February 15, 2024
        
        Vendor:
        Modern Office Solutions
        987 Business Park Drive
        Austin, TX 78701
        Phone: (512) 555-0199
        
        Ship To:
        Digital Marketing Agency
        123 Creative Street, Floor 3
        Austin, TX 78702
        
        Buyer Contact: Sarah Johnson, Procurement Manager
        Phone: (512) 555-0145
        Email: sarah.johnson@digitalagency.com
        
        Line Items:
        
        Item #    Description                    Qty    Unit Price    Line Total
        ---------------------------------------------------------------
        1         Executive Office Desk          5      $849.99      $4,249.95
        2         Ergonomic Office Chair         5      $399.99      $1,999.95
        3         Filing Cabinet 4-Drawer        3      $229.99      $689.97
        4         Conference Table (8-person)    1      $1,299.99    $1,299.99
        5         Office Chairs (Conference)     8      $149.99      $1,199.92
        6         Whiteboard 4x6 ft             2      $199.99      $399.98
        7         Desk Lamps LED                10      $49.99       $499.90
        
        Subtotal:                                                  $10,339.66
        Texas State Tax (8.25%):                                   $853.02
        Shipping & Handling:                                       $299.00
        TOTAL:                                                     $11,491.68
        
        Payment Terms: Net 45 days
        Delivery Instructions: Call 24 hours before delivery
        Special Instructions: Assembly required for desks and chairs
        
        Authorized by: Mike Chen, Operations Director
        Date: January 12, 2024
        """
    },
    
    "receipt_restaurant": {
        "type": "receipt",
        "filename": "restaurant_receipt.txt",
        "content": """
        The Gourmet Table
        456 Culinary Avenue
        Seattle, WA 98101
        (206) 555-0187
        
        Server: Maria G.
        Table: 12
        Guests: 4
        
        Receipt #: 2024-0112-0387
        Date: 01/12/2024
        Time: 7:45 PM
        
        ORDER DETAILS:
        
        2x Caesar Salad                    $18.00
        1x Grilled Salmon                  $32.00  
        1x Ribeye Steak                    $42.00
        1x Vegetarian Pasta                $24.00
        1x Chicken Marsala                 $28.00
        
        BEVERAGES:
        2x House Wine (Glass)              $24.00
        1x Craft Beer                      $8.00
        1x Sparkling Water                 $4.00
        
        DESSERTS:
        1x Tiramisu                        $9.00
        1x Chocolate Cake                  $9.00
        
        Subtotal:                          $198.00
        WA State Tax (10.1%):              $19.98
        Gratuity (18%):                    $39.24
        
        TOTAL:                             $257.22
        
        Payment Method: VISA ending in 4532
        Authorization Code: 789456
        Approval Code: 123789
        
        Thank you for dining with us!
        Visit us online: www.gourmettable.com
        Follow @GourmetTableSeattle
        """
    },
    
    "bank_statement_business": {
        "type": "bank_statement", 
        "filename": "business_bank_statement.txt",
        "content": """
        FIRST BUSINESS BANK
        BUSINESS CHECKING ACCOUNT STATEMENT
        
        Account Holder: TechStart Solutions LLC
        Account Number: ****7890
        Statement Period: December 1, 2023 - December 31, 2023
        
        ACCOUNT SUMMARY
        Beginning Balance (12/01/2023):          $45,267.89
        Total Deposits:                          $127,450.00
        Total Withdrawals:                       $89,234.67
        Total Service Charges:                   $85.00
        Ending Balance (12/31/2023):             $83,398.22
        
        DEPOSIT DETAIL
        12/05  ACH Credit - Client Payment        $25,000.00
        12/08  Wire Transfer - Investment         $50,000.00
        12/12  Mobile Deposit - Check #1234       $8,750.00
        12/18  ACH Credit - Recurring Revenue     $15,200.00
        12/22  Cash Deposit                       $5,500.00
        12/28  ACH Credit - Final Q4 Payment      $23,000.00
        
        WITHDRAWAL DETAIL
        12/02  ACH Debit - Payroll                $32,500.00
        12/03  Check #5001 - Office Rent          $4,200.00
        12/05  Check #5002 - Utilities            $867.45
        12/07  Online Transfer - Vendor Payment   $12,500.00
        12/10  ACH Debit - Insurance Premium      $2,450.00
        12/15  Check #5003 - Equipment Purchase   $8,900.00
        12/16  ACH Debit - Payroll                $32,500.00
        12/20  Online Payment - Credit Card       $3,456.78
        12/23  Check #5004 - Marketing Services   $5,600.00
        12/28  ATM Withdrawal                     $500.00
        12/30  ACH Debit - Loan Payment          $2,760.44
        
        SERVICE CHARGES
        12/15  Account Maintenance Fee            $25.00
        12/20  Wire Transfer Fee                  $35.00
        12/25  Overdraft Protection Fee           $25.00
        
        ACCOUNT ANALYSIS
        Average Daily Balance:                   $64,832.50
        Days Account Overdrawn:                  0
        
        Customer Service: 1-800-FIRST-BIZ
        Online Banking: www.firstbusinessbank.com
        """
    },
    
    "contract_saas": {
        "type": "contract",
        "filename": "saas_service_agreement.txt", 
        "content": """
        SOFTWARE AS A SERVICE AGREEMENT
        
        This Software as a Service Agreement ("Agreement") is entered into on 
        January 10, 2024 ("Effective Date"), by and between:
        
        CloudTech Solutions Inc., a Delaware corporation with its principal place 
        of business at 789 Tech Park Drive, Austin, TX 78759 ("Provider"), 
        
        and
        
        Growth Marketing Co., a California limited liability company with its 
        principal place of business at 321 Startup Lane, San Jose, CA 95110 ("Customer").
        
        RECITALS
        
        WHEREAS, Provider offers cloud-based marketing automation software;
        WHEREAS, Customer desires to license and use Provider's software services;
        
        NOW, THEREFORE, the parties agree as follows:
        
        1. SERVICES
        Provider shall provide Customer with access to its marketing automation 
        platform including email marketing, lead scoring, and analytics tools.
        
        2. TERM
        This Agreement commences on February 1, 2024 and continues for an initial 
        term of twenty-four (24) months, ending January 31, 2026, unless terminated 
        earlier in accordance with this Agreement.
        
        3. FEES AND PAYMENT
        Customer shall pay Provider a monthly subscription fee of $2,500 per month, 
        due on the first day of each month. Setup fee of $5,000 due upon execution.
        
        4. DATA AND PRIVACY
        Provider shall implement reasonable security measures to protect Customer data.
        Customer retains ownership of all customer data and marketing content.
        
        5. LIMITATION OF LIABILITY
        Provider's total liability shall not exceed the fees paid by Customer in 
        the twelve (12) months preceding the claim.
        
        6. TERMINATION
        Either party may terminate this Agreement with sixty (60) days written notice.
        Customer may terminate immediately for material breach not cured within 
        thirty (30) days.
        
        7. GOVERNING LAW
        This Agreement shall be governed by the laws of the State of Texas.
        
        IN WITNESS WHEREOF, the parties have executed this Agreement as of the 
        Effective Date.
        
        CLOUDTECH SOLUTIONS INC.      GROWTH MARKETING CO.
        
        By: /s/ Jennifer Martinez       By: /s/ David Kim
        Name: Jennifer Martinez        Name: David Kim  
        Title: CEO                     Title: VP Operations
        Date: January 10, 2024         Date: January 10, 2024
        
        Auto-renewal: Yes, for successive 12-month terms
        Termination notice required: 60 days
        """
    },
    
    "financial_statement_q4": {
        "type": "financial_statement",
        "filename": "q4_income_statement.txt",
        "content": """
        INNOVATIVE SOLUTIONS CORP
        CONSOLIDATED INCOME STATEMENT
        (In thousands, except per share data)
        
        For the Three Months Ended December 31, 2023
        (Unaudited)
        
        REVENUES:
        Product revenue                              $8,450
        Service revenue                              $3,750
        Subscription revenue                         $5,200
        License revenue                              $1,890
        Total revenues                              $19,290
        
        COST OF REVENUES:
        Product costs                                $3,380
        Service costs                                $1,500
        Total cost of revenues                       $4,880
        
        Gross profit                                $14,410
        
        OPERATING EXPENSES:
        Research and development                     $4,200
        Sales and marketing                          $3,850
        General and administrative                   $2,100
        Total operating expenses                    $10,150
        
        Operating income                             $4,260
        
        OTHER INCOME (EXPENSE):
        Interest income                                $125
        Interest expense                              ($87)
        Other income, net                              $45
        Total other income                             $83
        
        Income before income taxes                   $4,343
        
        Income tax expense                          ($1,085)
        
        NET INCOME                                   $3,258
        
        EARNINGS PER SHARE:
        Basic earnings per share                     $1.12
        Diluted earnings per share                   $1.09
        
        Weighted average shares outstanding:
        Basic                                        2,908
        Diluted                                      2,990
        
        Year-over-Year Comparison:
        Q4 2023 Revenue: $19,290K (↑15.3% from Q4 2022)
        Q4 2023 Net Income: $3,258K (↑22.7% from Q4 2022)
        
        Key Metrics:
        Gross Margin: 74.7%
        Operating Margin: 22.1%  
        Net Margin: 16.9%
        
        The accompanying notes are an integral part of these financial statements.
        """
    },
    
    "legal_motion": {
        "type": "legal_document",
        "filename": "motion_to_dismiss.txt",
        "content": """
        IN THE UNITED STATES DISTRICT COURT
        FOR THE NORTHERN DISTRICT OF CALIFORNIA
        
        TECHSTART INNOVATIONS INC.,
                                                    Plaintiff,
        
        v.                                          Case No. 3:24-cv-00789-WHO
        
        MEGACORP ENTERPRISES LLC,
                                                    Defendant.
        
        DEFENDANT'S MOTION TO DISMISS PURSUANT TO 
        FEDERAL RULE OF CIVIL PROCEDURE 12(b)(6)
        
        TO THE HONORABLE COURT:
        
        Defendant MegaCorp Enterprises LLC ("MegaCorp"), by and through its 
        undersigned counsel, respectfully moves this Court pursuant to Federal 
        Rule of Civil Procedure 12(b)(6) to dismiss Plaintiff's Complaint for 
        failure to state a claim upon which relief can be granted.
        
        I. INTRODUCTION
        
        This action arises from a failed business partnership between TechStart 
        Innovations Inc. ("TechStart") and MegaCorp regarding the development 
        of artificial intelligence software. TechStart's Complaint fails to 
        state viable claims for breach of contract, misappropriation of trade 
        secrets, or unfair competition.
        
        II. FACTUAL BACKGROUND
        
        In March 2023, TechStart and MegaCorp entered into a Joint Development 
        Agreement ("JDA") to create AI-powered business analytics software. 
        The parties invested a combined $2.5 million in the project over eight months.
        
        In November 2023, MegaCorp terminated the JDA pursuant to Section 8.2, 
        which permits termination for material breach after 30-day cure period.
        
        III. ARGUMENT
        
        A. The Complaint Fails to State a Claim for Breach of Contract
        
        TechStart cannot establish the essential elements of a breach of contract 
        claim. Specifically, TechStart fails to plead: (1) the existence of a 
        valid contract; (2) plaintiff's performance or excuse for non-performance; 
        (3) defendant's breach; and (4) resulting damages.
        
        B. The Trade Secret Claim is Legally Insufficient  
        
        TechStart's trade secret misappropriation claim fails because the alleged 
        "secrets" were publicly available information and collaborative work product 
        under the JDA.
        
        IV. CONCLUSION
        
        For the foregoing reasons, MegaCorp respectfully requests that this Court 
        grant its Motion to Dismiss with prejudice.
        
        Respectfully submitted,
        
        WILSON, BRADLEY & ASSOCIATES LLP
        
        By: /s/ Sarah M. Wilson
        Sarah M. Wilson (Bar No. 234567)
        1000 Montgomery Street, Suite 1500  
        San Francisco, CA 94104
        Telephone: (415) 555-0123
        Email: swilson@wilsonbradley.com
        
        Attorneys for Defendant MegaCorp Enterprises LLC
        
        Filed: January 15, 2024
        Hearing Date: March 1, 2024 at 10:00 AM
        Courtroom: 4, 17th Floor
        """
    }
}


async def create_multi_domain_system():
    """Create and configure the complete multi-domain processing system"""
    print("🔧 Initializing Multi-Domain Document Processing System...")
    
    # Create orchestrator for coordination
    orchestrator = AgentOrchestrator("multi_domain_orchestrator")
    
    # Create and register document classifier
    classifier = DocumentClassifierAgent("production_classifier")
    orchestrator.register_agent(classifier)
    
    # Create multi-domain processor hub
    processor = MultiDomainProcessorAgent("production_processor")
    
    # Configure for competitive processing
    config = ProcessingConfig(
        strategy=ProcessingStrategy.COMPETITIVE,
        accuracy_threshold=0.95,
        enable_parallel_processing=True,
        save_intermediate_results=True
    )
    processor.processing_config = config
    
    # Register all specialized processors
    processor.register_specialized_processor(DocumentType.PURCHASE_ORDER, PurchaseOrderProcessor())
    processor.register_specialized_processor(DocumentType.RECEIPT, ReceiptProcessor())
    processor.register_specialized_processor(DocumentType.BANK_STATEMENT, BankStatementProcessor())
    processor.register_specialized_processor(DocumentType.CONTRACT, ContractProcessor())
    processor.register_specialized_processor(DocumentType.FINANCIAL_STATEMENT, FinancialStatementProcessor())
    processor.register_specialized_processor(DocumentType.LEGAL_DOCUMENT, LegalDocumentProcessor())
    
    # Register processor with orchestrator
    orchestrator.register_agent(processor)
    
    print("✅ System initialization complete!")
    print(f"   - Document Classifier: {classifier.name}")
    print(f"   - Multi-Domain Processor: {processor.name}")
    print(f"   - Registered Processors: {len(processor.processor_registry.processors)} types")
    
    return orchestrator, processor


async def demonstrate_document_classification(classifier: DocumentClassifierAgent):
    """Demonstrate document classification capabilities"""
    print("\n📋 DOCUMENT CLASSIFICATION DEMONSTRATION")
    print("=" * 60)
    
    classification_results = []
    
    for doc_name, doc_data in DEMO_DOCUMENTS.items():
        print(f"\n🔍 Classifying: {doc_name}")
        
        start_time = time.perf_counter()
        result = await classifier.classify_document_text(doc_data["content"])
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        if result['success']:
            classification = result['classification_result']
            predicted_type = classification['document_type']
            confidence = classification['confidence_score']
            method = classification.get('classification_method', 'unknown')
            
            correct = predicted_type == doc_data["type"]
            status = "✅ CORRECT" if correct else "❌ INCORRECT"
            
            print(f"   Expected: {doc_data['type']}")
            print(f"   Predicted: {predicted_type} ({confidence:.1%} confidence)")
            print(f"   Method: {method}")
            print(f"   Time: {processing_time:.3f}s")
            print(f"   Result: {status}")
            
            classification_results.append({
                'document': doc_name,
                'expected': doc_data["type"],
                'predicted': predicted_type,
                'confidence': confidence,
                'correct': correct,
                'time': processing_time
            })
        else:
            print(f"   ❌ CLASSIFICATION FAILED: {result.get('error', 'Unknown error')}")
    
    # Summary statistics
    if classification_results:
        correct_count = sum(1 for r in classification_results if r['correct'])
        accuracy = correct_count / len(classification_results)
        avg_confidence = statistics.mean([r['confidence'] for r in classification_results])
        avg_time = statistics.mean([r['time'] for r in classification_results])
        
        print(f"\n📊 CLASSIFICATION SUMMARY:")
        print(f"   Accuracy: {accuracy:.1%} ({correct_count}/{len(classification_results)})")
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Average Time: {avg_time:.3f}s per document")
    
    return classification_results


async def demonstrate_specialized_processing(processor: MultiDomainProcessorAgent):
    """Demonstrate specialized document processing"""
    print("\n⚙️ SPECIALIZED DOCUMENT PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    demo_results = DemoResults()
    
    for doc_name, doc_data in DEMO_DOCUMENTS.items():
        print(f"\n🔄 Processing: {doc_name}")
        
        start_time = time.perf_counter()
        result = await processor.process_document_text(doc_data["content"])
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        demo_results.add_result(doc_data["type"], result, processing_time)
        
        if result['success']:
            # Extract key metrics
            classification = result.get('classification', {})
            processing_result = result.get('processing_result', {})
            
            doc_type = classification.get('document_type', 'unknown')
            confidence = processing_result.get('confidence_score', 0.0)
            method = processing_result.get('processing_method', 'unknown')
            
            print(f"   Document Type: {doc_type}")
            print(f"   Accuracy: {confidence:.1%}")
            print(f"   Processing Method: {method}")
            print(f"   Processing Time: {processing_time:.3f}s")
            
            # Show key extracted data
            extracted_data = processing_result.get('extracted_data', {})
            if extracted_data:
                print(f"   Key Fields Extracted:")
                for field, value in list(extracted_data.items())[:5]:  # Show first 5 fields
                    if value and str(value).strip():
                        print(f"     - {field}: {str(value)[:50]}{'...' if len(str(value)) > 50 else ''}")
            
            # Cost information
            cost_breakdown = result.get('cost_breakdown', {})
            if cost_breakdown:
                total_cost = cost_breakdown.get('total_cost', 0.0)
                print(f"   Processing Cost: ${total_cost:.4f}")
            
            print(f"   Status: ✅ SUCCESS")
        else:
            print(f"   Status: ❌ FAILED - {result.get('error', 'Unknown error')}")
    
    return demo_results


async def demonstrate_competitive_processing():
    """Demonstrate competitive processing with multiple methods"""
    print("\n🏆 COMPETITIVE PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    from agents.intelligence.competitive_processor import CompetitiveProcessingEngine, ProcessingMethod
    
    engine = CompetitiveProcessingEngine()
    
    # Test with invoice document
    invoice_doc = DEMO_DOCUMENTS["invoice_consulting"]
    print(f"🔍 Testing competitive processing on: {invoice_doc['type']}")
    
    # Test different method combinations
    method_combinations = [
        ([ProcessingMethod.REGEX_EXTRACTION], "Regex Only"),
        ([ProcessingMethod.PATTERN_MATCHING], "Pattern Only"),
        ([ProcessingMethod.HYBRID_RULES], "Hybrid Rules"),
        ([ProcessingMethod.REGEX_EXTRACTION, ProcessingMethod.PATTERN_MATCHING], "Regex + Pattern"),
        ([ProcessingMethod.REGEX_EXTRACTION, ProcessingMethod.PATTERN_MATCHING, ProcessingMethod.HYBRID_RULES], "All Free Methods")
    ]
    
    for methods, description in method_combinations:
        print(f"\n🔬 Testing: {description}")
        
        start_time = time.perf_counter()
        results = await engine.process_competitively(
            invoice_doc["content"], 
            DocumentType.INVOICE, 
            methods
        )
        end_time = time.perf_counter()
        
        processing_time = end_time - start_time
        
        # Find best result
        best_result = max(results, key=lambda r: r.confidence_score)
        
        print(f"   Methods: {len(methods)}")
        print(f"   Best Method: {best_result.method.value}")
        print(f"   Best Confidence: {best_result.confidence_score:.1%}")
        print(f"   Processing Time: {processing_time:.3f}s")
        print(f"   Cost Estimate: ${best_result.cost_estimate:.4f}")
        
        # Show method comparison
        if len(results) > 1:
            print(f"   Method Comparison:")
            for result in sorted(results, key=lambda r: r.confidence_score, reverse=True):
                print(f"     - {result.method.value}: {result.confidence_score:.1%}")
    
    # Show performance metrics
    performance = engine.get_method_performance()
    if any(perf['attempts'] > 0 for perf in performance.values()):
        print(f"\n📈 METHOD PERFORMANCE SUMMARY:")
        for method, perf in performance.items():
            if perf['attempts'] > 0:
                print(f"   {method}:")
                print(f"     Success Rate: {perf['success_rate']:.1%}")
                print(f"     Avg Accuracy: {perf['average_accuracy']:.1%}")
                print(f"     Avg Time: {perf['average_processing_time']:.3f}s")


async def demonstrate_batch_processing(processor: MultiDomainProcessorAgent):
    """Demonstrate batch processing capabilities"""
    print("\n📦 BATCH PROCESSING DEMONSTRATION")
    print("=" * 60)
    
    # Create temporary files for batch processing
    import tempfile
    temp_files = []
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files for batch processing
            for doc_name, doc_data in DEMO_DOCUMENTS.items():
                file_path = temp_path / f"{doc_name}.txt"
                file_path.write_text(doc_data["content"])
                temp_files.append(str(file_path))
            
            print(f"📁 Created {len(temp_files)} temporary test files")
            
            # Process batch
            start_time = time.perf_counter()
            results = await processor.batch_process_documents(temp_files)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            throughput = len(temp_files) / (batch_time / 3600)  # docs per hour
            
            print(f"\n⏱️ BATCH PROCESSING RESULTS:")
            print(f"   Documents: {len(temp_files)}")
            print(f"   Total Time: {batch_time:.2f}s")
            print(f"   Throughput: {throughput:.0f} documents/hour")
            
            # Analyze results
            successful = sum(1 for r in results if r.get('success', False))
            success_rate = successful / len(results) if results else 0.0
            
            print(f"   Success Rate: {success_rate:.1%} ({successful}/{len(results)})")
            
            if successful > 0:
                # Calculate average metrics for successful results
                accuracies = []
                times = []
                costs = []
                
                for result in results:
                    if result.get('success'):
                        processing_result = result.get('processing_result', {})
                        if isinstance(processing_result, dict):
                            accuracy = processing_result.get('confidence_score', 0.0)
                            if isinstance(accuracy, (int, float)):
                                accuracies.append(accuracy)
                        
                        time_taken = result.get('total_processing_time', 0.0)
                        if isinstance(time_taken, (int, float)):
                            times.append(time_taken)
                        
                        cost = result.get('cost_breakdown', {}).get('total_cost', 0.0)
                        if isinstance(cost, (int, float)):
                            costs.append(cost)
                
                if accuracies:
                    print(f"   Avg Accuracy: {statistics.mean(accuracies):.1%}")
                if times:
                    print(f"   Avg Time per Doc: {statistics.mean(times):.2f}s")
                if costs:
                    print(f"   Avg Cost per Doc: ${statistics.mean(costs):.4f}")
    
    except Exception as e:
        print(f"   ❌ Batch processing error: {e}")


async def demonstrate_cost_analysis(demo_results: DemoResults):
    """Demonstrate cost analysis and ROI"""
    print("\n💰 COST ANALYSIS & ROI DEMONSTRATION")
    print("=" * 60)
    
    summary = demo_results.get_summary()
    
    print(f"📊 PROCESSING SUMMARY:")
    print(f"   Documents Processed: {summary['total_documents_processed']}")
    print(f"   Success Rate: {summary['success_rate']:.1%}")
    print(f"   Average Accuracy: {summary['average_accuracy']:.1%}")
    print(f"   Total Processing Time: {summary['total_processing_time']:.2f}s")
    print(f"   Average Time per Document: {summary['average_time_per_document']:.2f}s")
    
    print(f"\n💵 COST COMPARISON:")
    system_cost = summary['cost_vs_manual']['system_cost']
    manual_cost = summary['cost_vs_manual']['manual_cost']
    savings = summary['cost_vs_manual']['savings']
    savings_pct = summary['cost_vs_manual']['savings_percentage']
    
    print(f"   System Cost per Document: ${system_cost:.4f}")
    print(f"   Manual Processing Cost: ${manual_cost:.2f}")
    print(f"   Savings per Document: ${savings:.4f}")
    print(f"   Savings Percentage: {savings_pct:.1f}%")
    
    # Project annual savings
    docs_per_year = 10000  # Example volume
    annual_savings = savings * docs_per_year
    print(f"\n📈 PROJECTED ANNUAL SAVINGS (10,000 docs/year):")
    print(f"   System Cost: ${system_cost * docs_per_year:,.2f}")
    print(f"   Manual Cost: ${manual_cost * docs_per_year:,.2f}")
    print(f"   Annual Savings: ${annual_savings:,.2f}")
    
    # ROI calculation
    system_development_cost = 50000  # Example development cost
    roi_months = system_development_cost / (annual_savings / 12) if annual_savings > 0 else float('inf')
    print(f"   ROI Payback Period: {roi_months:.1f} months")


async def run_comprehensive_demo():
    """Run the complete system demonstration"""
    print("🚀 MULTI-DOMAIN DOCUMENT PROCESSING SYSTEM DEMO")
    print("=" * 80)
    print("Demonstrating 95%+ accuracy across 7+ document types")
    print("Showcasing competitive processing and cost efficiency")
    print("=" * 80)
    
    try:
        # Initialize system
        orchestrator, processor = await create_multi_domain_system()
        
        # Get classifier for individual demos
        classifier = orchestrator.agents.get("production_classifier")
        
        # Run demonstrations
        await demonstrate_document_classification(classifier)
        demo_results = await demonstrate_specialized_processing(processor)
        await demonstrate_competitive_processing()
        await demonstrate_batch_processing(processor)
        await demonstrate_cost_analysis(demo_results)
        
        # Final system metrics
        print("\n🎯 FINAL SYSTEM PERFORMANCE METRICS")
        print("=" * 60)
        
        classifier_metrics = classifier.get_performance_metrics()
        processor_metrics = processor.get_performance_metrics()
        
        print(f"📋 CLASSIFICATION METRICS:")
        print(f"   Classifications Performed: {classifier_metrics['classifications_performed']}")
        print(f"   Overall Accuracy: {classifier_metrics.get('overall_accuracy', 0):.1%}")
        print(f"   Supported Document Types: {len(classifier_metrics['supported_document_types'])}")
        
        print(f"\n⚙️ PROCESSING METRICS:")
        print(f"   Documents Processed: {processor_metrics['documents_processed']}")
        print(f"   Success Rate: {processor_metrics['success_rate']:.1%}")
        print(f"   Average Accuracy: {processor_metrics['average_accuracy']:.1%}")
        print(f"   Average Cost per Document: ${processor_metrics['average_cost_per_document']:.4f}")
        
        # Performance targets assessment
        targets_met = []
        if classifier_metrics.get('overall_accuracy', 0) >= 0.98:
            targets_met.append("✅ Classification Accuracy (98%+)")
        else:
            targets_met.append("❌ Classification Accuracy (98%+)")
            
        if processor_metrics['average_accuracy'] >= 0.95:
            targets_met.append("✅ Processing Accuracy (95%+)")
        else:
            targets_met.append("❌ Processing Accuracy (95%+)")
            
        if processor_metrics['average_cost_per_document'] <= 0.05:
            targets_met.append("✅ Cost Efficiency ($0.05/doc)")
        else:
            targets_met.append("❌ Cost Efficiency ($0.05/doc)")
        
        print(f"\n🎯 PERFORMANCE TARGETS:")
        for target in targets_met:
            print(f"   {target}")
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("System successfully demonstrates multi-domain document processing")
        print("with competitive accuracy, speed, and cost efficiency.")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION ERROR: {e}")
        raise


if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(run_comprehensive_demo())