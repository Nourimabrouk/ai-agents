"""
Base Agent Template - Inspired by anthropic-quickstarts patterns
Implements state-of-the-art agentic AI architecture
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
import anthropic
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    EVOLVING = "evolving"
    ERROR = "error"


@dataclass
class Thought:
    """Represents agent's reasoning process"""
    analysis: str
    strategy: str
    confidence: float
    alternatives: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Action:
    """Represents agent's action"""
    action_type: str
    parameters: Dict[str, Any]
    tools_used: List[str]
    expected_outcome: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Observation:
    """Represents agent's observation of results"""
    action: Action
    result: Any
    success: bool
    learnings: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class Memory:
    """Agent memory system with episodic and semantic storage"""
    
    def __init__(self, max_episodes: int = 1000):
        self.episodic_memory: List[Observation] = []
        self.semantic_memory: Dict[str, Any] = {}
        self.working_memory: Dict[str, Any] = {}
        self.max_episodes = max_episodes
    
    async def store_episode(self, observation: Observation) -> None:
        """Store an episodic memory"""
        self.episodic_memory.append(observation)
        if len(self.episodic_memory) > self.max_episodes:
            self.episodic_memory.pop(0)
    
    async def recall_similar(self, context: Dict[str, Any], k: int = 5) -> List[Observation]:
        """Recall similar past experiences"""
        # Implement semantic similarity search
        # For now, return most recent k episodes
        return self.episodic_memory[-k:] if self.episodic_memory else []
    
    async def extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from accumulated memories"""
        patterns = {}
        if self.episodic_memory:
            # Analyze success rates
            successes = sum(1 for obs in self.episodic_memory if obs.success)
            patterns['success_rate'] = successes / len(self.episodic_memory)
            
            # Extract common learnings
            all_learnings = []
            for obs in self.episodic_memory:
                all_learnings.extend(obs.learnings)
            patterns['common_learnings'] = list(set(all_learnings))
        
        return patterns


class LearningSystem:
    """Meta-learning system for continuous improvement"""
    
    def __init__(self):
        self.strategies: Dict[str, float] = {}  # Strategy -> Success rate
        self.adaptations: List[Dict[str, Any]] = []
    
    async def update(self, observation: Observation) -> None:
        """Update learning from observation"""
        strategy = observation.action.action_type
        if strategy not in self.strategies:
            self.strategies[strategy] = 0.0
        
        # Update success rate with exponential moving average
        alpha = 0.1  # Learning rate
        current_success = 1.0 if observation.success else 0.0
        self.strategies[strategy] = (1 - alpha) * self.strategies[strategy] + alpha * current_success
    
    async def recommend_strategy(self, available_strategies: List[str]) -> str:
        """Recommend best strategy based on past performance"""
        best_strategy = None
        best_score = -1.0
        
        for strategy in available_strategies:
            score = self.strategies.get(strategy, 0.5)  # Default to 0.5 for unknown
            if score > best_score:
                best_score = score
                best_strategy = strategy
        
        return best_strategy or available_strategies[0]


class BaseAgent(ABC):
    """
    Base Agent Class - Foundation for all AI agents
    Inspired by anthropic-quickstarts patterns with advanced agentic capabilities
    """
    
    def __init__(
        self,
        name: str,
        api_key: Optional[str] = None,
        tools: Optional[List[Callable]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.state = AgentState.IDLE
        self.config = config or {}
        
        # Core components
        self.client = anthropic.Anthropic(api_key=api_key) if api_key else None
        self.tools = tools or []
        self.memory = Memory()
        self.learning_system = LearningSystem()
        
        # Metrics tracking
        self.total_tasks = 0
        self.successful_tasks = 0
        self.total_tokens_used = 0
        
        # Sub-agents for specialized tasks
        self.sub_agents: Dict[str, 'BaseAgent'] = {}
        
        logger.info(f"Initialized agent: {self.name}")
    
    async def think(self, task: Any, context: Optional[Dict[str, Any]] = None) -> Thought:
        """
        Meta-cognitive reasoning about the task
        Analyzes the problem and formulates a strategy
        """
        self.state = AgentState.THINKING
        logger.info(f"{self.name}: Thinking about task...")
        
        # Recall similar past experiences
        past_experiences = await self.memory.recall_similar(context or {})
        
        # Get recommended strategy from learning system
        available_strategies = self._get_available_strategies()
        recommended_strategy = await self.learning_system.recommend_strategy(available_strategies)
        
        # Formulate thought
        thought = Thought(
            analysis=f"Analyzing task with {len(past_experiences)} similar past experiences",
            strategy=recommended_strategy,
            confidence=self._calculate_confidence(past_experiences),
            alternatives=available_strategies[:3]
        )
        
        logger.info(f"{self.name}: Strategy selected - {thought.strategy} (confidence: {thought.confidence:.2f})")
        return thought
    
    async def act(self, thought: Thought) -> Action:
        """
        Execute action based on reasoning
        Implements the chosen strategy using available tools
        """
        self.state = AgentState.ACTING
        logger.info(f"{self.name}: Acting on strategy: {thought.strategy}")
        
        # Select tools based on strategy
        selected_tools = self._select_tools_for_strategy(thought.strategy)
        
        # Create action
        action = Action(
            action_type=thought.strategy,
            parameters={"thought": thought.analysis},
            tools_used=[tool.__name__ for tool in selected_tools],
            expected_outcome="Task completion based on strategy"
        )
        
        # Execute with selected tools
        for tool in selected_tools:
            try:
                await self._execute_tool(tool, action.parameters)
            except Exception as e:
                logger.error(f"{self.name}: Tool execution failed - {e}")
        
        return action
    
    async def observe(self, action: Action, result: Any) -> Observation:
        """
        Observe and learn from results
        Extracts learnings and updates memory
        """
        self.state = AgentState.OBSERVING
        logger.info(f"{self.name}: Observing results...")
        
        # Evaluate success
        success = self._evaluate_success(result)
        
        # Extract learnings
        learnings = self._extract_learnings(action, result, success)
        
        # Create observation
        observation = Observation(
            action=action,
            result=result,
            success=success,
            learnings=learnings
        )
        
        # Update memory and learning system
        await self.memory.store_episode(observation)
        await self.learning_system.update(observation)
        
        # Update metrics
        self.total_tasks += 1
        if success:
            self.successful_tasks += 1
        
        logger.info(f"{self.name}: Task {'successful' if success else 'failed'} - Success rate: {self.get_success_rate():.2%}")
        
        return observation
    
    async def evolve(self) -> None:
        """
        Self-improvement based on accumulated experience
        Evolves strategies and capabilities
        """
        self.state = AgentState.EVOLVING
        logger.info(f"{self.name}: Evolving based on experience...")
        
        # Extract patterns from memory
        patterns = await self.memory.extract_patterns()
        
        # Adapt strategies based on patterns
        if patterns.get('success_rate', 0) < 0.5:
            logger.info(f"{self.name}: Low success rate detected, exploring new strategies")
            self._explore_new_strategies()
        
        # Optimize based on learnings
        if patterns.get('common_learnings'):
            logger.info(f"{self.name}: Incorporating learnings: {patterns['common_learnings'][:3]}")
            self._incorporate_learnings(patterns['common_learnings'])
    
    async def process_task(self, task: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Main task processing pipeline
        Implements the think -> act -> observe -> evolve cycle
        """
        try:
            # Think
            thought = await self.think(task, context)
            
            # Act
            action = await self.act(thought)
            
            # Execute task (to be implemented by subclasses)
            result = await self.execute(task, action)
            
            # Observe
            observation = await self.observe(action, result)
            
            # Evolve periodically
            if self.total_tasks % 10 == 0:
                await self.evolve()
            
            return result
            
        except Exception as e:
            self.state = AgentState.ERROR
            logger.error(f"{self.name}: Task processing failed - {e}")
            raise
        finally:
            self.state = AgentState.IDLE
    
    @abstractmethod
    async def execute(self, task: Any, action: Action) -> Any:
        """
        Execute the specific task - to be implemented by subclasses
        """
        pass
    
    def _get_available_strategies(self) -> List[str]:
        """Get list of available strategies"""
        return ["direct", "exploratory", "collaborative", "analytical", "creative"]
    
    def _calculate_confidence(self, past_experiences: List[Observation]) -> float:
        """Calculate confidence based on past experiences"""
        if not past_experiences:
            return 0.5
        
        successes = sum(1 for exp in past_experiences if exp.success)
        return successes / len(past_experiences)
    
    def _select_tools_for_strategy(self, strategy: str) -> List[Callable]:
        """Select appropriate tools for the chosen strategy"""
        # Map strategies to tools (simplified)
        if strategy == "analytical":
            return [tool for tool in self.tools if "analyze" in tool.__name__.lower()]
        elif strategy == "creative":
            return [tool for tool in self.tools if "generate" in tool.__name__.lower()]
        else:
            return self.tools[:3]  # Default to first 3 tools
    
    async def _execute_tool(self, tool: Callable, parameters: Dict[str, Any]) -> Any:
        """Execute a tool with given parameters"""
        if asyncio.iscoroutinefunction(tool):
            return await tool(**parameters)
        else:
            return tool(**parameters)
    
    def _evaluate_success(self, result: Any) -> bool:
        """Evaluate if the result indicates success"""
        # Override in subclasses for specific success criteria
        return result is not None
    
    def _extract_learnings(self, action: Action, result: Any, success: bool) -> List[str]:
        """Extract learnings from the action and result"""
        learnings = []
        
        if success:
            learnings.append(f"Strategy '{action.action_type}' effective for this task type")
        else:
            learnings.append(f"Strategy '{action.action_type}' needs adjustment")
        
        if action.tools_used:
            learnings.append(f"Tools used: {', '.join(action.tools_used)}")
        
        return learnings
    
    def _explore_new_strategies(self) -> None:
        """Explore new strategies when current ones aren't working"""
        # Implement strategy exploration logic
        pass
    
    def _incorporate_learnings(self, learnings: List[str]) -> None:
        """Incorporate learnings into agent behavior"""
        # Implement learning incorporation logic
        pass
    
    def get_success_rate(self) -> float:
        """Get overall success rate"""
        if self.total_tasks == 0:
            return 0.0
        return self.successful_tasks / self.total_tasks
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics"""
        return {
            "name": self.name,
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "success_rate": self.get_success_rate(),
            "total_tokens_used": self.total_tokens_used,
            "memory_size": len(self.memory.episodic_memory),
            "strategies_learned": len(self.learning_system.strategies)
        }
    
    async def spawn_sub_agent(self, name: str, specialization: str) -> 'BaseAgent':
        """Spawn a specialized sub-agent"""
        sub_agent = type(self)(
            name=f"{self.name}.{name}",
            api_key=self.config.get('api_key'),
            tools=self._get_tools_for_specialization(specialization),
            config={**self.config, 'specialization': specialization}
        )
        self.sub_agents[name] = sub_agent
        logger.info(f"{self.name}: Spawned sub-agent {sub_agent.name} for {specialization}")
        return sub_agent
    
    def _get_tools_for_specialization(self, specialization: str) -> List[Callable]:
        """Get tools specific to a specialization"""
        # Filter tools based on specialization
        return [tool for tool in self.tools if specialization.lower() in tool.__name__.lower()]
    
    async def collaborate_with(self, other_agent: 'BaseAgent', task: Any) -> Any:
        """Collaborate with another agent on a task"""
        logger.info(f"{self.name}: Collaborating with {other_agent.name}")
        
        # Share context and memories
        shared_context = {
            'my_experience': await self.memory.extract_patterns(),
            'their_experience': await other_agent.memory.extract_patterns()
        }
        
        # Both agents process the task
        my_result = await self.process_task(task, shared_context)
        their_result = await other_agent.process_task(task, shared_context)
        
        # Combine results (override in subclasses for specific combination logic)
        return self._combine_results(my_result, their_result)
    
    def _combine_results(self, result1: Any, result2: Any) -> Any:
        """Combine results from multiple agents"""
        # Default implementation - override in subclasses
        return {"agent1": result1, "agent2": result2}