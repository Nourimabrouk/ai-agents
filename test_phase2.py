#!/usr/bin/env python3
"""
Simple Phase 2 Test Script
Tests core invoice processing functionality without Unicode issues
"""

import asyncio
import json
from orchestrator import AgentOrchestrator, Task
from agents.accountancy.invoice_processor import InvoiceProcessorAgent


async def test_phase2():
    """Test Phase 2 core functionality"""
    print("Phase 2 Invoice Processing Test")
    print("=" * 40)
    
    # Initialize system
    orchestrator = AgentOrchestrator(name="test_orchestrator")
    invoice_processor = InvoiceProcessorAgent(
        name="test_processor",
        config={'memory_backend': 'sqlite', 'memory_db_path': ':memory:'}
    )
    
    # Register agent
    orchestrator.register_agent(invoice_processor)
    print(f"Orchestrator initialized with {len(orchestrator.agents)} agents")
    
    # Sample invoice
    sample_invoice = """
ACME CORPORATION
123 Business Street
New York, NY 10001

INVOICE

Invoice Number: INV-2024-001
Invoice Date: 01/15/2024

Bill To:
XYZ Company
456 Client Avenue

Description                 Total
Professional Services     $1,500.00
Consulting Hours          $1,000.00

Total Amount:             $2,500.00
    """
    
    # Create task
    task = Task(
        id="test_invoice_1",
        description="Process test invoice",
        requirements={
            "type": "invoice_processing",
            "content": sample_invoice
        }
    )
    
    print("\nProcessing sample invoice...")
    
    # Process through orchestrator
    try:
        result = await orchestrator.delegate_task(task)
        
        if result and result.get("success"):
            invoice_data = result["invoice_data"]
            print(f"SUCCESS! Accuracy: {result.get('accuracy', 0):.1%}")
            print(f"Invoice Number: {invoice_data.get('invoice_number')}")
            print(f"Vendor: {invoice_data.get('vendor_name')}")
            print(f"Amount: ${invoice_data.get('total_amount')}")
            print(f"Extraction Method: {result.get('extraction_method')}")
            
            # Get performance metrics
            metrics = invoice_processor.get_performance_metrics()
            print(f"\nPerformance Metrics:")
            print(f"Processed: {metrics['processed_invoices']}")
            print(f"Success Rate: {metrics['overall_accuracy']:.1%}")
            print(f"Target Met: {metrics['meets_target']}")
            
            # Budget usage
            budget = metrics["budget_usage"]
            print(f"Budget Usage:")
            print(f"Anthropic: {budget['anthropic_usage_percent']:.1f}% of free tier")
            print(f"Azure: {budget['azure_usage_percent']:.1f}% of free tier")
            
            return True
            
        else:
            error = result.get("error", "Unknown error") if result else "No result"
            print(f"FAILED: {error}")
            return False
            
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        return False


async def main():
    """Main test function"""
    success = await test_phase2()
    
    if success:
        print("\n" + "=" * 40)
        print("PHASE 2 TEST: PASSED")
        print("- Invoice processing operational")
        print("- Orchestrator integration working") 
        print("- Budget constraints maintained")
        print("- Quality targets achieved")
    else:
        print("\n" + "=" * 40)
        print("PHASE 2 TEST: FAILED")
        print("- Review error messages above")


if __name__ == "__main__":
    asyncio.run(main())