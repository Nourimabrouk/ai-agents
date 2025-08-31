# Getting Started with AI Agents

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.9+ installed
- Node.js 18+ installed
- Git for version control
- Code editor (VS Code recommended)

### Initial Setup

#### 1. Repository Setup
```bash
# Clone the repository
git clone <your-repository-url>
cd ai-agents

# Copy environment template
cp .env.example .env
# Edit .env file with your API keys
```

#### 2. Python Environment
```bash
# Create virtual environment
python -m venv ai-agents-env

# Activate virtual environment
# Windows:
ai-agents-env\Scripts\activate
# Linux/Mac:
source ai-agents-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 3. Node.js Setup
```bash
# Install Node.js dependencies
npm install

# Verify installation
npm run test
```

#### 4. API Keys Configuration
Edit your `.env` file and add:
- `ANTHROPIC_API_KEY` - Get from https://console.anthropic.com/
- `OPENAI_API_KEY` - Get from https://platform.openai.com/
- Other keys as needed for specific integrations

## 🎯 Your First Agent

### Hello World Claude Code Agent

Create `agents/claude-code/hello_world.py`:

```python
import asyncio
from anthropic import Anthropic

class HelloWorldAgent:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    async def greet_user(self, name: str) -> str:
        """Simple greeting agent"""
        response = await self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=100,
            messages=[{
                "role": "user", 
                "content": f"Greet {name} as a professional AI accounting assistant."
            }]
        )
        return response.content[0].text

async def main():
    # Load API key from environment
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    agent = HelloWorldAgent(os.getenv("ANTHROPIC_API_KEY"))
    greeting = await agent.greet_user("John")
    print(greeting)

if __name__ == "__main__":
    asyncio.run(main())
```

Run your first agent:
```bash
cd agents/claude-code
python hello_world.py
```

### Your First Accounting Agent

Create `agents/accountancy/invoice_parser.py`:

```python
import re
from typing import Dict, Any
from dataclasses import dataclass
from decimal import Decimal

@dataclass
class Invoice:
    invoice_number: str
    date: str
    vendor: str
    total_amount: Decimal
    line_items: list

class SimpleInvoiceParser:
    def __init__(self):
        self.patterns = {
            'invoice_number': r'(?i)invoice\s*#?\s*:?\s*([A-Z0-9\-]+)',
            'date': r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})',
            'amount': r'\$?\s*(\d+(?:,\d{3})*\.?\d{0,2})',
        }
    
    def parse_invoice_text(self, text: str) -> Dict[str, Any]:
        """Extract basic invoice information from text"""
        results = {}
        
        for field, pattern in self.patterns.items():
            match = re.search(pattern, text)
            if match:
                results[field] = match.group(1)
        
        return results
    
    def validate_invoice_data(self, data: Dict[str, Any]) -> bool:
        """Basic validation of extracted data"""
        required_fields = ['invoice_number', 'date', 'amount']
        return all(field in data for field in required_fields)

# Example usage
if __name__ == "__main__":
    parser = SimpleInvoiceParser()
    
    sample_text = """
    Invoice #: INV-2024-001
    Date: 01/15/2024
    Vendor: ABC Company
    Total: $1,250.00
    """
    
    extracted = parser.parse_invoice_text(sample_text)
    print("Extracted data:", extracted)
    print("Valid:", parser.validate_invoice_data(extracted))
```

## 🛠️ Development Workflow

### 1. Planning Phase
- Review `planning/project-ideas.md` for inspiration
- Choose a project aligned with your current skill level
- Create a project folder in `projects/`
- Document requirements and architecture

### 2. Implementation Phase
- Start with basic functionality in `agents/framework-name/`
- Build incrementally, testing each component
- Generate assets (visualizations, reports) in `assets/`
- Use shared utilities from `utils/`

### 3. Integration Phase
- Connect your agent to real data sources
- Implement error handling and logging
- Add monitoring and performance tracking
- Create user interface if needed

### 4. Documentation Phase
- Document your implementation
- Create usage examples and tutorials
- Generate visualizations of results
- Update your portfolio in `planning/`

## 📁 Directory Navigation

### Core Directories
- **`agents/`** - Your agent implementations
  - `claude-code/` - Claude Code and MCP agents
  - `microsoft/` - Microsoft AI framework agents
  - `langchain/` - LangChain and LangGraph agents
  - `accountancy/` - Domain-specific accounting agents

- **`frameworks/`** - Learning materials and templates
  - Each subdirectory contains examples and boilerplate code

- **`projects/`** - Complete end-to-end implementations
  - Start here for real-world applications

### Supporting Directories
- **`assets/`** - Generated content and outputs
- **`utils/`** - Shared utilities and integrations
- **`planning/`** - Your career roadmap and learning plans
- **`docs/`** - Documentation and guides

## 🎯 Next Steps

### Week 1-2: Environment and Basics
1. ✅ Complete setup above
2. ✅ Run hello world agent
3. ✅ Run simple invoice parser
4. 📖 Read through `frameworks/claude-code/` examples
5. 🎯 Choose your first real project from `planning/project-ideas.md`

### Week 3-4: First Real Agent
1. 🏗️ Implement your chosen project
2. 📊 Generate visualizations in `assets/visualizations/`
3. 📝 Document your implementation
4. 🧪 Add tests in `utils/testing/`

### Week 5-6: Expand and Integrate
1. 🔗 Add integrations with real systems
2. 📈 Implement monitoring and analytics
3. 🌐 Create a web interface or dashboard
4. 📚 Update your learning progress

## 🆘 Getting Help

### Common Issues
- **Import Errors**: Ensure virtual environment is activated
- **API Errors**: Check your API keys in `.env` file
- **Permission Errors**: Check file permissions and paths

### Resources
- **Documentation**: Check `docs/` directory
- **Examples**: Look in `frameworks/` for working examples
- **Community**: Join Claude Code and LangChain communities
- **Issues**: Create GitHub issues for bugs or questions

### Best Practices
- Always use virtual environments
- Keep API keys secure and never commit them
- Test your agents with small datasets first
- Document your code and decisions
- Version control everything except secrets

---

**Remember**: Start small, build incrementally, and focus on real-world problems that demonstrate business value!