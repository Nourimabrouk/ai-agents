#!/usr/bin/env python3
"""Debug regex patterns for invoice processing"""

import re

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

print("Sample Invoice Text:")
print(sample_invoice)
print("\n" + "="*50)

# Test invoice number patterns
invoice_patterns = [
    r'invoice\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)',
    r'inv\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)', 
    r'bill\s*(?:number|#|no\.?)\s*:?\s*([A-Z0-9\-]+)'
]

print("Testing Invoice Number Patterns:")
for i, pattern in enumerate(invoice_patterns):
    match = re.search(pattern, sample_invoice, re.IGNORECASE)
    print(f"Pattern {i+1}: {match.group(1) if match else 'No match'}")

# Test date patterns
date_patterns = [
    r'(?:invoice\s*)?date\s*:?\s*(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})',
    r'(?:invoice\s*)?date\s*:?\s*(\d{2,4}[-/]\d{1,2}[-/]\d{1,2})',
    r'(\d{1,2}/\d{1,2}/\d{2,4})'
]

print("\nTesting Date Patterns:")
for i, pattern in enumerate(date_patterns):
    match = re.search(pattern, sample_invoice, re.IGNORECASE)
    print(f"Pattern {i+1}: {match.group(1) if match else 'No match'}")

# Test amount patterns
amount_patterns = [
    r'total\s*amount\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
    r'total\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
    r'amount\s*due\s*:?\s*\$?([\d,]+(?:\.\d{2})?)',
    r'grand\s*total\s*:?\s*\$?([\d,]+(?:\.\d{2})?)'
]

print("\nTesting Amount Patterns:")
for i, pattern in enumerate(amount_patterns):
    match = re.search(pattern, sample_invoice, re.IGNORECASE)
    print(f"Pattern {i+1}: {match.group(1) if match else 'No match'}")

# Test vendor name extraction
print("\nTesting Vendor Name Extraction:")
lines = sample_invoice.split('\n')
for i, line in enumerate(lines[:10]):
    line = line.strip()
    print(f"Line {i}: '{line}'")
    if (len(line) > 3 and not re.match(r'^\d', line) and 
        'invoice' not in line.lower() and 
        'bill to' not in line.lower() and
        line):
        if re.search(r'[A-Z]', line) and not re.match(r'^[0-9/\-\s]+$', line):
            print(f"  -> VENDOR CANDIDATE: {line}")
            break