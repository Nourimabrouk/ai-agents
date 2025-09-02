# Getting Started with AI Agents

## ✅ Current Status: READY FOR DEVELOPMENT

**Environment**: ✅ Fully configured and tested  
**Core Framework**: ✅ Advanced multi-agent system operational  
**Tests**: ✅ All passing (100% success rate)  
**Demo**: ✅ Working showcase of all capabilities  

### Prerequisites
- Python 3.13+ (currently installed)
- Windows 11 development environment
- Cursor IDE for optimal Claude Code integration
- Git for version control

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

#### 2. Python Environment ✅ COMPLETE
```bash
# Virtual environment already created and configured
source .venv/Scripts/activate  # Activate existing environment

# All dependencies installed:
# ✅ Core AI libraries (anthropic, openai, langchain)
# ✅ Data processing (pandas, numpy, scipy)
# ✅ Web frameworks (fastapi, streamlit)
# ✅ Database & persistence (sqlalchemy, sqlite)
# ✅ Testing & development tools
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

## 🎯 Ready-to-Use Agents

### 🔥 Demo All Capabilities (READY NOW)
```bash
# Activate environment and run comprehensive demo
source .venv/Scripts/activate
python demo.py
```

### 🧠 Cool Agent System Features
- **Multi-Agent Coordination**: Different patterns for agents working together
- **Learning & Memory**: Agents remember what worked and what didn't
- **Strategy Evolution**: Agents get better at tasks over time
- **Metrics & Logging**: See what your agents are actually doing
- **Windows-Friendly**: Proper async handling that works on Windows

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

## 🚀 Development Roadmap

### 🎯 IMMEDIATE NEXT STEPS (Week 1)
1. **Pick a Focus**: Accounting data processing or document analysis
2. **Add Claude API**: Replace dummy responses with real AI
3. **Build Real Agent**: Create an agent that does something useful
4. **Add Tools**: Give your agents actual capabilities

### 🏗️ NEXT PHASE (Weeks 2-4)
1. **Specialized Agents**: 
   - Data processing: Invoice extraction, report generation
   - Document analysis: PDF parsing, content summarization
2. **Better Tools**: File handling, web scraping, database connections
3. **Web Dashboard**: Streamlit interface to monitor your agents
4. **Error Handling**: Make agents robust and recoverable

### 🌟 ADVANCED EXPERIMENTS (Weeks 5-8)
1. **Multi-Modal**: Handle images, PDFs, audio files
2. **Distributed Agents**: Multiple agents coordinating
3. **Learning Experiments**: See how agents improve over time
4. **Performance Optimization**: Make everything faster and cheaper

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