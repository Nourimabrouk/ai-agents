# Agents Directory

This directory contains all AI agent implementations organized by framework and domain.

## Structure

- **`claude-code/`** - Claude Code agents and MCP implementations
- **`microsoft/`** - Microsoft AI Framework agents (Copilot Studio, Azure AI)
- **`langchain/`** - LangChain and LangGraph multi-agent systems
- **`accountancy/`** - Domain-specific accounting and finance agents

## Getting Started

1. Start with `claude-code/` - most straightforward to implement
2. Move to `accountancy/` for domain-specific implementations
3. Explore `microsoft/` and `langchain/` as you expand

Each subdirectory contains:
- `README.md` - Framework-specific guidance
- Example implementations
- Configuration files
- Documentation

## Agent Development Guidelines

- Keep each agent focused on a single responsibility
- Use shared utilities from `../utils/`
- Generate assets in `../assets/`
- Follow framework-specific best practices
- Include comprehensive error handling
- Document agent capabilities and limitations