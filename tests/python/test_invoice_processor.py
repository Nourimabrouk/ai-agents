"""
Test suite for Invoice Processor Agent
Validates document processing, data extraction, and quality assurance
"""

import pytest
import asyncio
import tempfile
import os
from datetime import date
from decimal import Decimal

# Test the invoice processor
from agents.accountancy.invoice_processor import (
    InvoiceProcessorAgent,
    InvoiceData,
    InvoiceValidator,
    DocumentExtractor,
    BudgetTracker
)


@pytest.fixture
def sample_invoice_text():
    """Sample invoice text for testing"""
    return """
    ACME CORPORATION
    123 Business Street
    New York, NY 10001
    
    INVOICE
    
    Invoice Number: INV-2024-001
    Invoice Date: 01/15/2024
    Due Date: 02/14/2024
    
    Bill To:
    XYZ Company
    456 Client Avenue
    Los Angeles, CA 90001
    
    Description                 Quantity    Unit Price    Total
    Professional Services           10         $150.00   $1,500.00
    Consulting Hours                5          $200.00   $1,000.00
    
    Subtotal:                                           $2,500.00
    Tax (8.25%):                                         $206.25
    Total Amount:                                      $2,706.25
    
    Payment Terms: Net 30
    """


@pytest.fixture
def invoice_processor():
    """Create invoice processor for testing"""
    return InvoiceProcessorAgent(
        name="test_processor",
        config={'memory_backend': 'sqlite', 'memory_db_path': ':memory:'}
    )


@pytest.mark.asyncio
async def test_budget_tracker():
    """Test budget tracking functionality"""
    tracker = BudgetTracker()
    
    # Test initial state
    assert tracker.can_use_anthropic()
    assert tracker.can_use_azure()
    
    # Test usage tracking
    tracker.add_anthropic_usage(1000)
    assert tracker.anthropic_tokens_used == 1000
    
    tracker.add_azure_usage()
    assert tracker.azure_requests_used == 1


@pytest.mark.asyncio
async def test_document_extractor_text_confidence():
    """Test text confidence calculation"""
    # Good invoice text
    good_text = "Invoice #123 Date: 01/01/2024 Total: $100.00 Vendor: ACME Corp"
    confidence = DocumentExtractor._calculate_text_confidence(good_text)
    assert confidence > 0.7
    
    # Poor text
    poor_text = "abc def"
    confidence = DocumentExtractor._calculate_text_confidence(poor_text)
    assert confidence < 0.6
    
    # Empty text
    empty_confidence = DocumentExtractor._calculate_text_confidence("")
    assert empty_confidence == 0.0


@pytest.mark.asyncio
async def test_invoice_validator():
    """Test invoice data validation"""
    # Valid invoice data
    valid_invoice = InvoiceData(
        invoice_number="INV-2024-001",
        invoice_date=date(2024, 1, 15),
        vendor_name="ACME Corporation",
        total_amount=Decimal("2706.25"),
        subtotal=Decimal("2500.00"),
        tax_amount=Decimal("206.25")
    )
    
    validated = await InvoiceValidator.validate_invoice_data(valid_invoice)
    assert len(validated.validation_errors) == 0
    assert validated.confidence_score > 0.0
    
    # Invalid invoice data
    invalid_invoice = InvoiceData(
        invoice_number="",  # Missing required field
        vendor_name="",     # Missing required field
        total_amount=Decimal("-100")  # Invalid amount
    )
    
    validated_invalid = await InvoiceValidator.validate_invoice_data(invalid_invoice)
    assert len(validated_invalid.validation_errors) > 0


@pytest.mark.asyncio
async def test_invoice_processor_text_processing(sample_invoice_text, invoice_processor):
    """Test invoice processing from text content"""
    result = await invoice_processor.process_invoice_text(sample_invoice_text)
    
    assert result["success"] is True
    assert "invoice_data" in result
    assert result["accuracy"] > 0.0
    
    invoice_data = result["invoice_data"]
    assert invoice_data["invoice_number"] == "INV-2024-001"
    assert invoice_data["vendor_name"] == "ACME CORPORATION"
    assert invoice_data["total_amount"] == "2706.25"


@pytest.mark.asyncio
async def test_invoice_processor_task_execution(sample_invoice_text, invoice_processor):
    """Test invoice processor as a BaseAgent task"""
    from templates.base_agent import Action
    
    action = Action(
        action_type="process_invoice",
        parameters={"text": sample_invoice_text},
        tools_used=["regex_parser"],
        expected_outcome="Extracted invoice data"
    )
    
    result = await invoice_processor.execute(sample_invoice_text, action)
    
    assert result["success"] is True
    assert "invoice_data" in result
    assert result["invoice_data"]["invoice_number"] == "INV-2024-001"


@pytest.mark.asyncio
async def test_performance_metrics(invoice_processor):
    """Test performance metrics tracking"""
    initial_metrics = invoice_processor.get_performance_metrics()
    assert initial_metrics["processed_invoices"] == 0
    assert initial_metrics["overall_accuracy"] == 0.0
    
    # Process a sample invoice
    sample_text = "Invoice #123 Date: 01/01/2024 Total: $100.00 Vendor: Test Corp"
    await invoice_processor.process_invoice_text(sample_text)
    
    updated_metrics = invoice_processor.get_performance_metrics()
    assert updated_metrics["processed_invoices"] == 1
    assert "budget_usage" in updated_metrics


@pytest.mark.asyncio 
async def test_invalid_file_handling(invoice_processor):
    """Test handling of invalid file paths"""
    result = await invoice_processor.process_invoice_file("nonexistent_file.pdf")
    
    assert result["success"] is False
    assert "error" in result
    assert result["invoice_data"] is None


@pytest.mark.asyncio
async def test_orchestrator_integration():
    """Test integration with orchestrator"""
    from orchestrator import AgentOrchestrator, Task
    
    # Create orchestrator and register invoice processor
    orchestrator = AgentOrchestrator("test_orchestrator")
    processor = InvoiceProcessorAgent("invoice_agent")
    orchestrator.register_agent(processor)
    
    # Create invoice processing task
    task = Task(
        id="invoice_task_1",
        description="Process sample invoice",
        requirements={
            "type": "invoice_processing",
            "text": "Invoice #999 Date: 12/25/2023 Total: $555.00 Vendor: Test Vendor"
        }
    )
    
    # Delegate task
    result = await orchestrator.delegate_task(task)
    
    assert result is not None
    assert task.status == "completed"
    assert orchestrator.total_tasks_completed == 1


@pytest.mark.asyncio
async def test_batch_processing():
    """Test batch processing functionality"""
    processor = InvoiceProcessorAgent("batch_processor")
    
    # Create temporary test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # This test would require actual files, so we'll just test the error handling
        fake_files = [
            os.path.join(temp_dir, "invoice1.pdf"),
            os.path.join(temp_dir, "invoice2.pdf")
        ]
        
        results = await processor.batch_process_invoices(fake_files)
        
        # Should handle file not found gracefully
        assert len(results) == 2
        for result in results:
            assert result["success"] is False
            assert "error" in result


def test_invoice_data_serialization():
    """Test invoice data dictionary conversion"""
    invoice = InvoiceData(
        invoice_number="TEST-001",
        vendor_name="Test Vendor",
        total_amount=Decimal("100.00")
    )
    
    data_dict = invoice.to_dict()
    
    assert data_dict["invoice_number"] == "TEST-001"
    assert data_dict["vendor_name"] == "Test Vendor"
    assert data_dict["total_amount"] == "100.00"
    assert "processed_at" in data_dict


if __name__ == "__main__":
    # Run basic test
    async def run_basic_test():
        processor = InvoiceProcessorAgent("test")
        sample = """
        Test Company
        Invoice #ABC123
        Date: 01/01/2024
        Total: $250.00
        """
        
        result = await processor.process_invoice_text(sample)
        print("Test result:", result)
        print("Performance:", processor.get_performance_metrics())
    
    asyncio.run(run_basic_test())