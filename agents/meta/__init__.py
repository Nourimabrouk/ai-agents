"""
Meta AI Agent System - Comprehensive agent orchestration and coordination

This package provides a meta-optimal architecture for developing AI agents with:
- Parallel planning and execution
- Specialized agent roles with clear separation of concerns
- Inter-agent communication protocol
- Knowledge management and learning
- Windows-optimized, async-first design
"""

from .meta_orchestrator import (
    MetaOrchestrator,
    DevelopmentTask,
    TaskPriority,
    AgentRole,
    AgentCapabilities
)

from .specialized_agents import (
    BaseSpecializedAgent,
    ArchitectAgent,
    DeveloperAgent,
    TesterAgent,
    ReviewerAgent,
    DocumenterAgent,
    IntegratorAgent,
    RefactorerAgent,
    DebuggerAgent
)

from .agent_protocol import (
    MessageBus,
    AgentInterface,
    ParallelCoordinator,
    Message,
    MessageType,
    MessagePriority
)

__version__ = "1.0.0"
__author__ = "AI Meta Agent System"

# Convenience imports
__all__ = [
    # Core orchestration
    "MetaOrchestrator",
    "DevelopmentTask",
    "TaskPriority",
    "AgentRole",
    "AgentCapabilities",
    
    # Specialized agents
    "BaseSpecializedAgent",
    "ArchitectAgent",
    "DeveloperAgent",
    "TesterAgent",
    "ReviewerAgent",
    "DocumenterAgent",
    "IntegratorAgent",
    "RefactorerAgent",
    "DebuggerAgent",
    
    # Communication protocol
    "MessageBus",
    "AgentInterface",
    "ParallelCoordinator",
    "Message",
    "MessageType",
    "MessagePriority"
]


def create_meta_system(config_path=None):
    """
    Factory function to create a complete meta-agent system
    
    Returns:
        MetaOrchestrator: Fully configured meta-orchestrator
    """
    return MetaOrchestrator(config_path)


async def quick_start_demo():
    """
    Quick start demonstration of the meta-agent system
    """
    from pathlib import Path
    import json
    
    # Create a sample requirement
    requirement = """
    Create a new financial analysis agent that can:
    1. Process accounting documents (PDFs, Excel files)
    2. Extract key financial metrics and ratios
    3. Generate automated insights and reports
    4. Integrate with QuickBooks API
    5. Handle multiple currencies and tax jurisdictions
    """
    
    # Create meta-orchestrator
    orchestrator = create_meta_system()
    
    print("🚀 Starting Meta-Agent System Demo")
    print("=" * 50)
    
    # Plan the development
    print("📋 Planning Development Tasks...")
    tasks = await orchestrator.plan_development(requirement, {
        "framework": "langchain",
        "target_systems": ["quickbooks", "stripe"],
        "priority": "high",
        "deadline": "2 weeks"
    })
    
    print(f"✅ Generated {len(tasks)} development tasks:")
    for i, task in enumerate(tasks[:7], 1):  # Show first 7 tasks
        print(f"   {i}. {task.description} ({task.priority.name})")
        if task.dependencies:
            print(f"      Dependencies: {', '.join(task.dependencies)}")
    
    if len(tasks) > 7:
        print(f"   ... and {len(tasks) - 7} more tasks")
    
    # Execute development (simulation)
    print("\n🔄 Executing Development...")
    results = await orchestrator.execute_development(max_parallel=3)
    
    print(f"✅ Execution Complete:")
    print(f"   • Completed: {len(results['completed'])} tasks")
    print(f"   • Failed: {len(results['failed'])} tasks")
    print(f"   • Total time: {results['total_time']:.1f} seconds")
    
    # Generate report
    print("\n📊 Generating Development Report...")
    report = await orchestrator.generate_report()
    print(report)
    
    return orchestrator, results


# Example configuration
EXAMPLE_CONFIG = {
    "max_parallel_agents": 5,
    "planning_timeout": 60,
    "execution_timeout": 300,
    "retry_attempts": 3,
    "knowledge_persistence": True,
    "windows_optimization": True,
    "async_preferred": True
}


def save_example_config(path="config.json"):
    """Save example configuration to file"""
    with open(path, 'w') as f:
        json.dump(EXAMPLE_CONFIG, f, indent=2)
    print(f"Example configuration saved to {path}")


if __name__ == "__main__":
    import asyncio
    
    # Run quick start demo
    asyncio.run(quick_start_demo())