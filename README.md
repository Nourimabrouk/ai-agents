# AI Agents Development Repository

A practical repository for developing AI agents across multiple frameworks with domain specialization capabilities.

## Purpose

- **Multi-Framework Development**: Work with Claude Code, Microsoft AI, and LangChain
- **Domain Specialization**: Focus on accountancy and other business domains
- **Agent Coordination**: Build systems where multiple agents work together
- **Practical Implementation**: Create working solutions over theoretical frameworks

## Quick Start

### Prerequisites
- Python 3.9+ 
- Node.js 18+
- Git
- Windows 11 (primary development environment)
- Cursor IDE (recommended)

### Setup
```cmd
# Clone and setup
git clone <repository-url>
cd ai-agents

# Python environment (Windows)
python -m venv ai-agents-env
ai-agents-env\Scripts\activate
pip install -r requirements.txt

# Node.js dependencies
npm install
```

### First Steps
1. Start with `agents/claude-code/` - basic Claude Code agents
2. Explore `frameworks/claude-code/` - MCP servers and tools
3. Build your first project in `projects/invoice-automation/`

## 📁 Repository Structure

- **`agents/`** - Agent implementations organized by framework and domain
- **`frameworks/`** - Learning materials, templates, and framework-specific code
- **`projects/`** - Complete end-to-end implementations
- **`assets/`** - Generated visualizations, documents, and datasets
- **`planning/`** - Career roadmaps and learning paths
- **`utils/`** - Shared utilities and integrations
- **`docs/`** - Documentation and guides

## Development Workflow

1. **Learn**: Explore `frameworks/` to understand each framework
2. **Implement**: Build agents in `agents/` using established patterns
3. **Coordinate**: Use `orchestrator.py` for multi-agent scenarios
4. **Test**: Validate with both unit and integration tests
5. **Monitor**: Track performance and behavior patterns

## Development Phases

### Phase 1: Foundation
- Master Claude Code and MCP development
- Build basic single-purpose agents
- Implement tool integration framework

### Phase 2: Coordination  
- Develop multi-agent orchestration
- Implement parallel execution patterns
- Create agent communication protocols

### Phase 3: Specialization
- Focus on domain-specific implementations
- Optimize for production workloads
- Build comprehensive monitoring

## Agent Types

### Framework Agents
- **Claude Code**: MCP servers, tool-calling agents
- **Microsoft**: Azure AI Studio, Copilot Studio workflows
- **LangChain**: LangGraph workflows, custom chains

### Domain Agents
- **Accountancy**: Invoice processing, reconciliation, compliance
- **General**: Data analysis, code generation, web scraping
- **Coordination**: Orchestrators, message brokers, supervisors

## Key Features

- **Parallel Execution**: Async/await patterns throughout
- **Windows Compatibility**: Native Windows development environment
- **Multi-Agent Coordination**: Built-in orchestration capabilities
- **Tool Integration**: Standardized tool framework
- **Performance Monitoring**: Metrics and observability built-in
- **Error Recovery**: Robust error handling and retry mechanisms

## Architecture

The repository uses a layered architecture:
- **Templates**: Base classes and patterns (`templates/`)
- **Agents**: Framework-specific implementations (`agents/`)
- **Tools**: Capability framework (`tools/`)
- **Orchestration**: Multi-agent coordination (`orchestrator.py`)
- **Projects**: End-to-end applications (`projects/`)
- **Utilities**: Shared functionality (`utils/`)

## Getting Started

1. Review `CLAUDE.md` for detailed development guidelines
2. Check `.cursorrules` for IDE-specific standards  
3. Use `templates/base_agent.py` as your starting point
4. Test with `orchestrator.py` for multi-agent scenarios
5. Follow Windows-compatible development practices

## Current Status

- [x] Repository structure and templates
- [x] Multi-agent orchestration framework
- [x] Tool integration system
- [x] Windows development environment setup
- [ ] First working agent implementation
- [ ] Integration with external systems