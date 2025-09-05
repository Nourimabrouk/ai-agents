#!/usr/bin/env python3
"""
Simple AI Agents Demo - Guaranteed to Work
"""

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAgent:
    """Simple demonstration agent"""
    
    def __init__(self, name: str):
        self.name = name
        self.tasks_completed = 0
    
    async def process_task(self, task: str) -> dict:
        """Process a simple task"""
        logger.info(f"Agent {self.name} processing: {task}")
        
        # Simulate work
        await asyncio.sleep(0.1)
        
        self.tasks_completed += 1
        
        return {
            'agent': self.name,
            'task': task,
            'status': 'completed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'tasks_completed': self.tasks_completed
        }


class SimpleOrchestrator:
    """Simple orchestrator for demo agents"""
    
    def __init__(self):
        self.agents = []
        self.results = []
    
    def add_agent(self, agent: SimpleAgent):
        """Add agent to orchestrator"""
        self.agents.append(agent)
        logger.info(f"Added agent: {agent.name}")
    
    async def execute_tasks(self, tasks: list) -> list:
        """Execute tasks across agents"""
        logger.info(f"Executing {len(tasks)} tasks with {len(self.agents)} agents")
        
        # Distribute tasks among agents
        task_futures = []
        for i, task in enumerate(tasks):
            agent = self.agents[i % len(self.agents)]
            future = agent.process_task(task)
            task_futures.append(future)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*task_futures)
        self.results.extend(results)
        
        return results


async def main():
    """Main demo function"""
    print("Simple AI Agents Demo")
    print("====================")
    
    # Create orchestrator
    orchestrator = SimpleOrchestrator()
    
    # Create agents
    agents = [
        SimpleAgent("DataProcessor"),
        SimpleAgent("Analyzer"),
        SimpleAgent("Validator")
    ]
    
    for agent in agents:
        orchestrator.add_agent(agent)
    
    # Define tasks
    tasks = [
        "Analyze customer data",
        "Process transaction records",
        "Validate data integrity",
        "Generate summary report",
        "Perform quality check",
        "Update database records"
    ]
    
    # Execute tasks
    start_time = datetime.now(timezone.utc)
    results = await orchestrator.execute_tasks(tasks)
    end_time = datetime.now(timezone.utc)
    
    # Display results
    print(f"\nExecution completed in {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Tasks completed: {len(results)}")
    print(f"Agents used: {len(orchestrator.agents)}")
    
    print("\nResults Summary:")
    for result in results:
        print(f"  {result['agent']}: {result['task']} -> {result['status']}")
    
    print(f"\nAgent Statistics:")
    for agent in orchestrator.agents:
        print(f"  {agent.name}: {agent.tasks_completed} tasks completed")
    
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
