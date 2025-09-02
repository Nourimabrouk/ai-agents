"""
Import bridge for orchestrator components
Provides access to the sophisticated orchestrator implementation
"""

# Import from the actual implementation
from core.orchestration.orchestrator import (
    AgentOrchestrator,
    Task,
    CommunicationProtocol,
    Message,
    Blackboard,
    CustomerSupportAgent,
    DataAnalystAgent,
    ClaudeCodeAgent,
    CodeReviewAgent
)

# Re-export for backward compatibility
__all__ = [
    'AgentOrchestrator',
    'Task',
    'CommunicationProtocol', 
    'Message',
    'Blackboard',
    'CustomerSupportAgent',
    'DataAnalystAgent',
    'ClaudeCodeAgent',
    'CodeReviewAgent'
]