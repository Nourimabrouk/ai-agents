"""
Multi-Agent Orchestration System
Coordinates multiple specialized agents working together
"""

import asyncio
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
from abc import ABC, abstractmethod
import networkx as nx
from collections import defaultdict

# Import base agent template
from templates.base_agent import BaseAgent, AgentState, Thought, Action, Observation

logger = logging.getLogger(__name__)


class CommunicationProtocol(Enum):
    """Communication protocols between agents"""
    BROADCAST = "broadcast"  # Send to all agents
    DIRECT = "direct"  # Send to specific agent
    PUBLISH_SUBSCRIBE = "publish_subscribe"  # Topic-based communication
    BLACKBOARD = "blackboard"  # Shared knowledge space


@dataclass
class Message:
    """Message structure for inter-agent communication"""
    sender: str
    recipient: str  # Can be "all" for broadcast
    content: Any
    protocol: CommunicationProtocol
    timestamp: datetime = field(default_factory=datetime.now)
    requires_response: bool = False
    priority: int = 0  # Higher number = higher priority


@dataclass
class Task:
    """Task definition for agents"""
    id: str
    description: str
    requirements: Dict[str, Any]
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    assigned_agents: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Any] = None


class Blackboard:
    """Shared knowledge space for agents"""
    
    def __init__(self):
        self.knowledge: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.history: List[Tuple[datetime, str, str, Any]] = []
    
    async def write(self, agent_id: str, key: str, value: Any) -> None:
        """Write knowledge to blackboard"""
        self.knowledge[key] = value
        self.history.append((datetime.now(), agent_id, key, value))
        
        # Notify subscribers
        if key in self.subscriptions:
            for subscriber in self.subscriptions[key]:
                logger.info(f"Notifying {subscriber} about update to {key}")
    
    async def read(self, key: str) -> Optional[Any]:
        """Read knowledge from blackboard"""
        return self.knowledge.get(key)
    
    async def subscribe(self, agent_id: str, key: str) -> None:
        """Subscribe to updates for a specific key"""
        self.subscriptions[key].add(agent_id)
    
    async def query(self, pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Query blackboard with pattern matching"""
        results = {}
        for key, value in self.knowledge.items():
            # Simple pattern matching (can be enhanced)
            if all(k in key for k in pattern.keys()):
                results[key] = value
        return results


class AgentOrchestrator:
    """
    Orchestrates multiple specialized agents
    Implements various coordination patterns
    """
    
    def __init__(self, name: str = "orchestrator"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.blackboard = Blackboard()
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.task_graph = nx.DiGraph()  # For task dependencies
        self.active_tasks: Dict[str, Task] = {}
        
        # Metrics
        self.total_tasks_completed = 0
        self.total_messages_sent = 0
        
        logger.info(f"Initialized orchestrator: {self.name}")
    
    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent with the orchestrator"""
        self.agents[agent.name] = agent
        logger.info(f"Registered agent: {agent.name}")
    
    def unregister_agent(self, agent_name: str) -> None:
        """Unregister an agent"""
        if agent_name in self.agents:
            del self.agents[agent_name]
            logger.info(f"Unregistered agent: {agent_name}")
    
    async def delegate_task(self, task: Task) -> Any:
        """
        Delegate a task to appropriate agents
        Automatically selects best agents based on capabilities
        """
        logger.info(f"Delegating task {task.id}: {task.description}")
        
        # Find suitable agents
        suitable_agents = await self._find_suitable_agents(task)
        
        if not suitable_agents:
            logger.error(f"No suitable agents found for task {task.id}")
            return None
        
        # Assign task to agents
        task.assigned_agents = [agent.name for agent in suitable_agents]
        self.active_tasks[task.id] = task
        
        # Execute task with selected pattern
        if len(suitable_agents) == 1:
            # Single agent execution
            result = await suitable_agents[0].process_task(task.description, task.requirements)
        else:
            # Multi-agent execution
            result = await self._coordinate_multi_agent_task(task, suitable_agents)
        
        # Update task status
        task.status = "completed"
        task.result = result
        self.total_tasks_completed += 1
        
        return result
    
    async def _find_suitable_agents(self, task: Task) -> List[BaseAgent]:
        """Find agents suitable for a task based on their capabilities"""
        suitable = []
        
        for agent in self.agents.values():
            # Check agent state
            if agent.state == AgentState.IDLE:
                # Check success rate threshold
                if agent.get_success_rate() > 0.5 or agent.total_tasks < 5:
                    suitable.append(agent)
        
        # Sort by success rate
        suitable.sort(key=lambda a: a.get_success_rate(), reverse=True)
        
        return suitable[:3]  # Return top 3 agents
    
    async def _coordinate_multi_agent_task(self, task: Task, agents: List[BaseAgent]) -> Any:
        """Coordinate multiple agents on a task"""
        # Choose coordination pattern based on task characteristics
        if "parallel" in task.description.lower():
            return await self.parallel_execution(agents, task)
        elif "sequential" in task.description.lower():
            return await self.sequential_execution(agents, task)
        elif "consensus" in task.description.lower():
            return await self.consensus_execution(agents, task)
        else:
            return await self.collaborative_execution(agents, task)
    
    async def hierarchical_delegation(self, task: Task) -> Any:
        """
        Top-down task decomposition and delegation
        Breaks complex tasks into subtasks
        """
        logger.info(f"Hierarchical delegation for task {task.id}")
        
        # Decompose task into subtasks
        subtasks = await self._decompose_task(task)
        
        # Create task dependency graph
        for subtask in subtasks:
            self.task_graph.add_node(subtask.id, task=subtask)
            for dep in subtask.dependencies:
                self.task_graph.add_edge(dep, subtask.id)
        
        # Execute subtasks respecting dependencies
        results = {}
        for subtask_id in nx.topological_sort(self.task_graph):
            if subtask_id in [st.id for st in subtasks]:
                subtask = next(st for st in subtasks if st.id == subtask_id)
                
                # Wait for dependencies
                for dep in subtask.dependencies:
                    if dep not in results:
                        logger.warning(f"Dependency {dep} not resolved for {subtask_id}")
                
                # Execute subtask
                result = await self.delegate_task(subtask)
                results[subtask_id] = result
        
        # Synthesize results
        return await self._synthesize_results(results)
    
    async def _decompose_task(self, task: Task) -> List[Task]:
        """Decompose a complex task into subtasks"""
        # Simple decomposition strategy (can be enhanced with AI)
        subtasks = []
        
        # Example decomposition
        subtasks.append(Task(
            id=f"{task.id}_analysis",
            description=f"Analyze requirements for {task.description}",
            requirements=task.requirements,
            dependencies=[]
        ))
        
        subtasks.append(Task(
            id=f"{task.id}_implementation",
            description=f"Implement solution for {task.description}",
            requirements=task.requirements,
            dependencies=[f"{task.id}_analysis"]
        ))
        
        subtasks.append(Task(
            id=f"{task.id}_validation",
            description=f"Validate solution for {task.description}",
            requirements=task.requirements,
            dependencies=[f"{task.id}_implementation"]
        ))
        
        return subtasks
    
    async def _synthesize_results(self, results: Dict[str, Any]) -> Any:
        """Synthesize results from multiple subtasks"""
        # Combine results (can be enhanced with AI)
        return {
            "subtask_results": results,
            "synthesis": "Combined results from all subtasks",
            "success": all(r is not None for r in results.values())
        }
    
    async def parallel_execution(self, agents: List[BaseAgent], task: Task) -> List[Any]:
        """Execute task with multiple agents in parallel"""
        logger.info(f"Parallel execution with {len(agents)} agents")
        
        tasks = [
            agent.process_task(task.description, task.requirements)
            for agent in agents
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if not isinstance(r, Exception)]
        
        return valid_results
    
    async def sequential_execution(self, agents: List[BaseAgent], task: Task) -> Any:
        """Execute task with agents in sequence, passing results forward"""
        logger.info(f"Sequential execution with {len(agents)} agents")
        
        result = None
        for agent in agents:
            context = task.requirements.copy()
            if result is not None:
                context["previous_result"] = result
            
            result = await agent.process_task(task.description, context)
        
        return result
    
    async def collaborative_execution(self, agents: List[BaseAgent], task: Task) -> Any:
        """
        Agents collaborate through discussion to solve task
        Implements a multi-agent conversation
        """
        logger.info(f"Collaborative execution with {len(agents)} agents")
        
        # Initialize conversation
        conversation_history = []
        max_rounds = 5
        consensus_threshold = 0.8
        
        for round_num in range(max_rounds):
            round_responses = []
            
            # Each agent contributes
            for agent in agents:
                context = {
                    "task": task.description,
                    "requirements": task.requirements,
                    "conversation_history": conversation_history
                }
                
                response = await agent.process_task(
                    f"Contribute to solving: {task.description}",
                    context
                )
                
                round_responses.append({
                    "agent": agent.name,
                    "response": response
                })
            
            conversation_history.extend(round_responses)
            
            # Check for consensus
            if await self._check_consensus(round_responses, consensus_threshold):
                logger.info(f"Consensus reached in round {round_num + 1}")
                break
        
        # Synthesize final answer from conversation
        return await self._synthesize_conversation(conversation_history)
    
    async def _check_consensus(self, responses: List[Dict], threshold: float) -> bool:
        """Check if agents have reached consensus"""
        # Simple consensus check (can be enhanced)
        if len(responses) < 2:
            return True
        
        # Compare similarity of responses (simplified)
        # In practice, would use embedding similarity or other metrics
        unique_responses = set(str(r["response"]) for r in responses)
        consensus_ratio = 1.0 / len(unique_responses)
        
        return consensus_ratio >= threshold
    
    async def _synthesize_conversation(self, conversation: List[Dict]) -> Any:
        """Synthesize final result from agent conversation"""
        # Extract key points from conversation
        return {
            "conversation_rounds": len(conversation),
            "participating_agents": list(set(r["agent"] for r in conversation)),
            "final_synthesis": conversation[-1] if conversation else None
        }
    
    async def consensus_execution(self, agents: List[BaseAgent], task: Task) -> Any:
        """
        Agents work independently then vote on best solution
        Implements a voting mechanism
        """
        logger.info(f"Consensus execution with {len(agents)} agents")
        
        # Get independent solutions
        solutions = await self.parallel_execution(agents, task)
        
        # Agents vote on solutions
        votes = defaultdict(int)
        
        for i, agent in enumerate(agents):
            # Each agent evaluates all solutions
            for j, solution in enumerate(solutions):
                if i != j:  # Don't vote for own solution
                    score = await agent.process_task(
                        "Evaluate solution quality (0-1)",
                        {"solution": solution, "task": task.description}
                    )
                    # Simplified: treat score as binary vote
                    if isinstance(score, (int, float)) and score > 0.5:
                        votes[j] += 1
        
        # Select solution with most votes
        if votes:
            best_solution_idx = max(votes, key=votes.get)
            return solutions[best_solution_idx]
        else:
            return solutions[0] if solutions else None
    
    async def swarm_intelligence(self, objective: str, swarm_size: int = 10) -> Any:
        """
        Implement swarm intelligence for optimization problems
        Agents act as particles in a swarm
        """
        logger.info(f"Swarm intelligence with {swarm_size} agents")
        
        # Create swarm agents if needed
        swarm = []
        for i in range(swarm_size):
            if f"swarm_{i}" not in self.agents:
                # Create simple swarm agent
                agent = BaseAgent(f"swarm_{i}")
                self.register_agent(agent)
            swarm.append(self.agents[f"swarm_{i}"])
        
        # Swarm optimization loop
        best_solution = None
        best_fitness = float('-inf')
        max_iterations = 20
        
        for iteration in range(max_iterations):
            # Each agent explores solution space
            solutions = []
            for agent in swarm:
                # Agent explores based on its position and velocity
                solution = await agent.process_task(
                    f"Explore solution for: {objective}",
                    {
                        "iteration": iteration,
                        "best_known": best_solution,
                        "local_best": agent.memory.semantic_memory.get("local_best")
                    }
                )
                solutions.append((agent, solution))
            
            # Evaluate solutions
            for agent, solution in solutions:
                # Simplified fitness evaluation
                fitness = await self._evaluate_fitness(solution, objective)
                
                # Update global best
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution
                
                # Update agent's local best
                local_best_fitness = agent.memory.semantic_memory.get("local_best_fitness", float('-inf'))
                if fitness > local_best_fitness:
                    agent.memory.semantic_memory["local_best"] = solution
                    agent.memory.semantic_memory["local_best_fitness"] = fitness
            
            # Check convergence
            if await self._check_swarm_convergence(swarm):
                logger.info(f"Swarm converged at iteration {iteration}")
                break
        
        return best_solution
    
    async def _evaluate_fitness(self, solution: Any, objective: str) -> float:
        """Evaluate fitness of a solution"""
        # Simplified fitness evaluation
        # In practice, would implement domain-specific fitness function
        return hash(str(solution)) % 100 / 100.0
    
    async def _check_swarm_convergence(self, swarm: List[BaseAgent]) -> bool:
        """Check if swarm has converged"""
        # Check if all agents have similar local bests
        local_bests = [
            agent.memory.semantic_memory.get("local_best")
            for agent in swarm
        ]
        
        # Simplified convergence check
        unique_solutions = set(str(s) for s in local_bests if s is not None)
        return len(unique_solutions) <= 2
    
    async def emergent_behavior_detection(self) -> List[Dict[str, Any]]:
        """
        Detect emergent behaviors from agent interactions
        Analyzes patterns that arise from collective behavior
        """
        logger.info("Detecting emergent behaviors")
        
        emergent_patterns = []
        
        # Analyze message patterns
        message_patterns = await self._analyze_message_patterns()
        if message_patterns:
            emergent_patterns.extend(message_patterns)
        
        # Analyze task completion patterns
        task_patterns = await self._analyze_task_patterns()
        if task_patterns:
            emergent_patterns.extend(task_patterns)
        
        # Analyze blackboard evolution
        blackboard_patterns = await self._analyze_blackboard_patterns()
        if blackboard_patterns:
            emergent_patterns.extend(blackboard_patterns)
        
        # Analyze agent specialization
        specialization_patterns = await self._analyze_specialization_patterns()
        if specialization_patterns:
            emergent_patterns.extend(specialization_patterns)
        
        return emergent_patterns
    
    async def _analyze_message_patterns(self) -> List[Dict[str, Any]]:
        """Analyze communication patterns between agents"""
        # Simplified pattern analysis
        return [{
            "type": "communication_cluster",
            "description": "Agents forming communication clusters",
            "agents_involved": list(self.agents.keys())[:3]
        }]
    
    async def _analyze_task_patterns(self) -> List[Dict[str, Any]]:
        """Analyze task completion patterns"""
        return [{
            "type": "task_specialization",
            "description": "Certain agents consistently chosen for specific task types",
            "pattern": "Agent specialization emerging"
        }]
    
    async def _analyze_blackboard_patterns(self) -> List[Dict[str, Any]]:
        """Analyze knowledge sharing patterns"""
        return [{
            "type": "knowledge_convergence",
            "description": "Shared knowledge converging to common understanding",
            "convergence_rate": 0.85
        }]
    
    async def _analyze_specialization_patterns(self) -> List[Dict[str, Any]]:
        """Analyze agent specialization patterns"""
        specializations = {}
        
        for agent_name, agent in self.agents.items():
            # Analyze agent's task history
            success_by_type = defaultdict(list)
            
            for obs in agent.memory.episodic_memory:
                task_type = obs.action.action_type
                success_by_type[task_type].append(obs.success)
            
            # Find specialization
            best_type = None
            best_rate = 0
            
            for task_type, successes in success_by_type.items():
                if successes:
                    rate = sum(successes) / len(successes)
                    if rate > best_rate:
                        best_rate = rate
                        best_type = task_type
            
            if best_type and best_rate > 0.7:
                specializations[agent_name] = best_type
        
        if specializations:
            return [{
                "type": "agent_specialization",
                "description": "Agents developing specialized capabilities",
                "specializations": specializations
            }]
        
        return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            "name": self.name,
            "registered_agents": len(self.agents),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.total_tasks_completed,
            "total_messages": self.total_messages_sent,
            "agent_metrics": {
                agent_name: agent.get_metrics()
                for agent_name, agent in self.agents.items()
            }
        }


# Example specialized agents using quickstart patterns

class CustomerSupportAgent(BaseAgent):
    """Customer support agent based on anthropic-quickstarts"""
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute customer support task"""
        # Implementation based on customer-support-agent quickstart
        return {
            "response": "Support response generated",
            "satisfaction_score": 0.9
        }


class DataAnalystAgent(BaseAgent):
    """Data analyst agent based on anthropic-quickstarts"""
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute data analysis task"""
        # Implementation based on financial-data-analyst quickstart
        return {
            "analysis": "Data insights generated",
            "visualizations": ["chart1.png", "chart2.png"]
        }


class ClaudeCodeAgent(BaseAgent):
    """Claude Code agent for development tasks"""
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute coding task"""
        return {
            "code": "Generated code implementation",
            "tests": "Unit tests created",
            "documentation": "API docs updated"
        }


class CodeReviewAgent(BaseAgent):
    """Code review agent"""
    
    async def execute(self, task: Any, action: Action) -> Any:
        """Execute code review"""
        return {
            "issues_found": ["Consider error handling", "Add type hints"],
            "suggestions": ["Refactor for clarity"],
            "approval_status": "approved_with_suggestions"
        }