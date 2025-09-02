#!/usr/bin/env python3
"""
AI Agents Demo Script
Demonstrates the basic functionality of the agent system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from templates.base_agent import BaseAgent, Action
from orchestrator import AgentOrchestrator, Task, CustomerSupportAgent, DataAnalystAgent
from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class ExampleAgent(BaseAgent):
    """Example agent that demonstrates basic functionality"""
    
    async def execute(self, task, action: Action):
        """Execute a simple task"""
        logger.info(f"{self.name}: Executing task '{task}' with strategy '{action.action_type}'")
        
        # Simulate some processing time
        await asyncio.sleep(0.1)
        
        # Return a result based on the task
        result = {
            "task": task,
            "strategy": action.action_type,
            "processed_by": self.name,
            "status": "completed",
            "confidence": 0.85
        }
        
        global_metrics.incr("tasks.completed")
        global_metrics.timing("task.duration", 0.1)
        
        return result


async def demo_single_agent():
    """Demonstrate single agent functionality"""
    logger.info("=== Single Agent Demo ===")
    
    # Create an agent
    agent = ExampleAgent(name="demo_agent")
    
    # Process a task
    result = await agent.process_task("Analyze customer feedback", {"priority": "high"})
    
    logger.info(f"Agent result: {result}")
    logger.info(f"Agent metrics: {agent.get_metrics()}")
    
    return result


async def demo_multi_agent_orchestration():
    """Demonstrate multi-agent orchestration"""
    logger.info("\n=== Multi-Agent Orchestration Demo ===")
    
    # Create orchestrator
    orchestrator = AgentOrchestrator("demo_orchestrator")
    
    # Register agents
    support_agent = CustomerSupportAgent("support_agent")
    analyst_agent = DataAnalystAgent("analyst_agent")
    demo_agent = ExampleAgent("helper_agent")
    
    orchestrator.register_agent(support_agent)
    orchestrator.register_agent(analyst_agent)
    orchestrator.register_agent(demo_agent)
    
    # Create a task
    task = Task(
        id="demo_task_1",
        description="Process customer inquiry and generate insights",
        requirements={"customer_id": "12345", "priority": "medium"}
    )
    
    # Delegate task
    result = await orchestrator.delegate_task(task)
    
    logger.info(f"Orchestration result: {result}")
    logger.info(f"Orchestrator metrics: {orchestrator.get_metrics()}")
    
    return result


async def demo_collaborative_execution():
    """Demonstrate collaborative agent execution"""
    logger.info("\n=== Collaborative Execution Demo ===")
    
    orchestrator = AgentOrchestrator("collaborative_demo")
    
    # Create multiple agents
    agents = [
        ExampleAgent("agent_1"),
        ExampleAgent("agent_2"),
        ExampleAgent("agent_3")
    ]
    
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Create a collaborative task
    task = Task(
        id="collab_task",
        description="collaborative problem solving task",
        requirements={"complexity": "high", "requires_consensus": True}
    )
    
    # Execute collaboratively
    result = await orchestrator._coordinate_multi_agent_task(task, agents)
    
    logger.info(f"Collaborative result: {result}")
    
    return result


async def demo_parallel_execution():
    """Demonstrate parallel agent execution"""
    logger.info("\n=== Parallel Execution Demo ===")
    
    orchestrator = AgentOrchestrator("parallel_demo")
    
    # Create agents
    agents = [
        ExampleAgent(f"parallel_agent_{i}") for i in range(3)
    ]
    
    for agent in agents:
        orchestrator.register_agent(agent)
    
    # Create a parallel task
    task = Task(
        id="parallel_task",
        description="parallel data processing task",
        requirements={"data_chunks": [1, 2, 3]}
    )
    
    # Execute in parallel
    results = await orchestrator.parallel_execution(agents, task)
    
    logger.info(f"Parallel results: {results}")
    
    return results


async def main():
    """Main demo function"""
    logger.info("Starting AI Agents Demo")
    logger.info("=" * 50)
    
    try:
        # Run demos
        await demo_single_agent()
        await demo_multi_agent_orchestration()
        await demo_collaborative_execution()
        await demo_parallel_execution()
        
        # Show global metrics
        logger.info("\n=== Global Metrics ===")
        stats = global_metrics.get_all_stats()
        for metric, data in stats.items():
            logger.info(f"{metric}: {data}")
        
        logger.info("\n=== Demo Complete ===")
        logger.info("All demos completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())