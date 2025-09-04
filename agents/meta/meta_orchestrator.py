"""
Meta Orchestrator: The conductor of all AI agent development
Coordinates specialized agents for parallel planning and sequential execution
"""
import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class AgentRole(Enum):
    ARCHITECT = "architect"
    DEVELOPER = "developer"
    TESTER = "tester"
    REVIEWER = "reviewer"
    DOCUMENTER = "documenter"
    INTEGRATOR = "integrator"
    REFACTORER = "refactorer"
    DEBUGGER = "debugger"


@dataclass
class DevelopmentTask:
    """Represents a development task with all necessary context"""
    id: str
    description: str
    priority: TaskPriority
    assigned_agents: List[AgentRole] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    
    def can_execute(self, completed_tasks: set) -> bool:
        """Check if all dependencies are satisfied"""
        return all(dep in completed_tasks for dep in self.dependencies)


@dataclass
class AgentCapabilities:
    """Define what each agent type can do"""
    role: AgentRole
    skills: List[str]
    tools: List[str]
    parallel_capable: bool = True
    max_concurrent_tasks: int = 3
    specializations: Dict[str, float] = field(default_factory=dict)


class MetaOrchestrator:
    """
    The meta-conductor that orchestrates all AI agent development.
    Manages parallel planning and sequential execution of specialized agents.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        self.agents: Dict[AgentRole, Any] = {}
        self.task_queue: List[DevelopmentTask] = []
        self.completed_tasks: set = set()
        self.active_tasks: Dict[str, DevelopmentTask] = {}
        self.knowledge_base: Dict[str, Any] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.config = self._load_config(config_path)
        self._initialize_agents()
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load orchestrator configuration"""
        default_config = {
            "max_parallel_agents": 5,
            "planning_timeout": 60,
            "execution_timeout": 300,
            "retry_attempts": 3,
            "knowledge_persistence": True,
            "agent_capabilities": self._default_agent_capabilities()
        }
  
        if config_path and config_path.exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _default_agent_capabilities(self) -> Dict[str, AgentCapabilities]:
        """Define default capabilities for each agent type"""
        return {
            AgentRole.ARCHITECT: AgentCapabilities(
                role=AgentRole.ARCHITECT,
                skills=["system_design", "pattern_recognition", "api_design", "architecture_patterns"],
                tools=["mermaid", "plantuml", "draw.io", "code_analysis"],
                parallel_capable=True,
                max_concurrent_tasks=2,
                specializations={"design": 0.9, "planning": 0.8, "abstraction": 0.9}
            ),
            AgentRole.DEVELOPER: AgentCapabilities(
                role=AgentRole.DEVELOPER,
                skills=["coding", "implementation", "optimization", "refactoring"],
                tools=["vscode", "cursor", "github_copilot", "language_servers"],
                parallel_capable=True,
                max_concurrent_tasks=3,
                specializations={"python": 0.9, "javascript": 0.8, "async": 0.9}
            ),
            AgentRole.TESTER: AgentCapabilities(
                role=AgentRole.TESTER,
                skills=["unit_testing", "integration_testing", "test_design", "coverage_analysis"],
                tools=["pytest", "jest", "coverage", "mutation_testing"],
                parallel_capable=True,
                max_concurrent_tasks=4,
                specializations={"test_generation": 0.9, "edge_cases": 0.8}
            ),
            AgentRole.REVIEWER: AgentCapabilities(
                role=AgentRole.REVIEWER,
                skills=["code_review", "security_analysis", "performance_review", "best_practices"],
                tools=["sonarqube", "pylint", "eslint", "security_scanners"],
                parallel_capable=True,
                max_concurrent_tasks=2,
                specializations={"quality": 0.9, "security": 0.8, "performance": 0.7}
            ),
            AgentRole.DOCUMENTER: AgentCapabilities(
                role=AgentRole.DOCUMENTER,
                skills=["technical_writing", "api_documentation", "user_guides", "diagrams"],
                tools=["markdown", "sphinx", "mkdocs", "swagger"],
                parallel_capable=True,
                max_concurrent_tasks=2,
                specializations={"clarity": 0.9, "completeness": 0.8}
            ),
            AgentRole.INTEGRATOR: AgentCapabilities(
                role=AgentRole.INTEGRATOR,
                skills=["system_integration", "api_integration", "dependency_management"],
                tools=["docker", "kubernetes", "ci_cd", "package_managers"],
                parallel_capable=False,
                max_concurrent_tasks=1,
                specializations={"compatibility": 0.9, "deployment": 0.8}
            ),
            AgentRole.REFACTORER: AgentCapabilities(
                role=AgentRole.REFACTORER,
                skills=["code_cleanup", "pattern_extraction", "debt_reduction", "optimization"],
                tools=["ast_tools", "refactoring_tools", "complexity_analyzers"],
                parallel_capable=True,
                max_concurrent_tasks=2,
                specializations={"simplification": 0.9, "patterns": 0.8}
            ),
            AgentRole.DEBUGGER: AgentCapabilities(
                role=AgentRole.DEBUGGER,
                skills=["debugging", "error_analysis", "root_cause_analysis", "profiling"],
                tools=["debuggers", "profilers", "tracers", "log_analyzers"],
                parallel_capable=True,
                max_concurrent_tasks=2,
                specializations={"problem_solving": 0.9, "analysis": 0.9}
            )
        }
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        from .specialized_agents import (
            ArchitectAgent, DeveloperAgent, TesterAgent,
            ReviewerAgent, DocumenterAgent, IntegratorAgent,
            RefactorerAgent, DebuggerAgent
        )
        
        agent_classes = {
            AgentRole.ARCHITECT: ArchitectAgent,
            AgentRole.DEVELOPER: DeveloperAgent,
            AgentRole.TESTER: TesterAgent,
            AgentRole.REVIEWER: ReviewerAgent,
            AgentRole.DOCUMENTER: DocumenterAgent,
            AgentRole.INTEGRATOR: IntegratorAgent,
            AgentRole.REFACTORER: RefactorerAgent,
            AgentRole.DEBUGGER: DebuggerAgent
        }
        
        for role, agent_class in agent_classes.items():
            capabilities = self.config["agent_capabilities"].get(role)
            if capabilities:
                self.agents[role] = agent_class(capabilities, self.knowledge_base)
                logger.info(f"Initialized {role.value} agent")
    
    async def plan_development(self, requirement: str, context: Dict[str, Any] = None) -> List[DevelopmentTask]:
        """
        Plan development tasks in parallel using all available agents.
        Each agent contributes their perspective to the plan.
        """
        logger.info(f"Planning development for: {requirement[:100]}...")
        
        # Parallel planning phase - all agents analyze the requirement
        planning_tasks = []
        for role, agent in self.agents.items():
            if hasattr(agent, 'analyze_requirement'):
                planning_tasks.append(
                    self._agent_planning_task(agent, requirement, context)
                )
        
        # Gather all planning results in parallel
        planning_results = await asyncio.gather(*planning_tasks, return_exceptions=True)
        
        # Synthesize plans into unified task list
        unified_tasks = await self._synthesize_plans(planning_results, requirement)
        
        # Optimize task ordering and dependencies
        optimized_tasks = await self._optimize_task_order(unified_tasks)
        
        self.task_queue.extend(optimized_tasks)
        logger.info(f"Created {len(optimized_tasks)} development tasks")
        
        return optimized_tasks
    
    async def _agent_planning_task(self, agent: Any, requirement: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Individual agent planning task"""
        try:
            return await agent.analyze_requirement(requirement, context)
        except Exception as e:
            logger.error(f"Agent {agent.role} planning failed: {e}")
            return {"error": str(e), "agent": agent.role}
    
    async def _synthesize_plans(self, planning_results: List[Dict], requirement: str) -> List[DevelopmentTask]:
        """Synthesize multiple agent plans into unified task list"""
        tasks = []
        task_id_counter = 0
        
        # Extract tasks from each agent's plan
        for result in planning_results:
            if "error" not in result and "tasks" in result:
                for task_desc in result["tasks"]:
                    task = DevelopmentTask(
                        id=f"task_{task_id_counter:04d}",
                        description=task_desc["description"],
                        priority=TaskPriority[task_desc.get("priority", "MEDIUM")],
                        assigned_agents=[AgentRole[role] for role in task_desc.get("agents", [])],
                        dependencies=task_desc.get("dependencies", []),
                        context=task_desc.get("context", {})
                    )
                    tasks.append(task)
                    task_id_counter += 1
        
        # Remove duplicates and merge similar tasks
        tasks = self._merge_similar_tasks(tasks)
        
        return tasks
    
    def _merge_similar_tasks(self, tasks: List[DevelopmentTask]) -> List[DevelopmentTask]:
        """Merge similar tasks to avoid duplication"""
        merged = []
        seen_descriptions = set()
        
        for task in tasks:
            # Simple similarity check - can be enhanced with embeddings
            normalized_desc = task.description.lower().strip()
            if normalized_desc not in seen_descriptions:
                seen_descriptions.add(normalized_desc)
                merged.append(task)
            else:
                # Merge agents and context for duplicate tasks
                for existing in merged:
                    if existing.description.lower().strip() == normalized_desc:
                        existing.assigned_agents = list(set(existing.assigned_agents + task.assigned_agents))
                        existing.context.update(task.context)
                        break
        
        return merged
    
    async def _optimize_task_order(self, tasks: List[DevelopmentTask]) -> List[DevelopmentTask]:
        """Optimize task ordering for maximum parallelization"""
        # Topological sort with priority consideration
        sorted_tasks = []
        remaining = tasks.copy()
        completed_ids = set()
        
        while remaining:
            # Find tasks that can be executed
            ready_tasks = [
                task for task in remaining 
                if task.can_execute(completed_ids)
            ]
            
            if not ready_tasks:
                # Circular dependency or missing dependency
                logger.warning("Circular or missing dependencies detected")
                sorted_tasks.extend(remaining)
                break
            
            # Sort by priority
            ready_tasks.sort(key=lambda t: t.priority.value)
            
            # Add to sorted list
            for task in ready_tasks:
                sorted_tasks.append(task)
                completed_ids.add(task.id)
                remaining.remove(task)
        
        return sorted_tasks
    
    async def execute_development(self, max_parallel: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute development tasks with parallel execution where possible
        and sequential execution where required.
        """
        if max_parallel is None:
            max_parallel = self.config["max_parallel_agents"]
        
        logger.info(f"Starting development execution with max {max_parallel} parallel agents")
        
        results = {
            "completed": [],
            "failed": [],
            "skipped": [],
            "total_time": 0
        }
        
        start_time = datetime.now()
        
        while self.task_queue or self.active_tasks:
            # Find tasks ready to execute
            ready_tasks = self._get_ready_tasks(max_parallel)
            
            if ready_tasks:
                # Execute ready tasks in parallel
                execution_results = await self._execute_parallel_tasks(ready_tasks)
                
                # Process results
                for task_id, result in execution_results.items():
                    task = self.active_tasks.pop(task_id)
                    task.completed_at = datetime.now()
                    task.result = result
                    
                    if result.get("success"):
                        task.status = "completed"
                        self.completed_tasks.add(task_id)
                        results["completed"].append(task)
                        logger.info(f"Task {task_id} completed successfully")
                    else:
                        task.status = "failed"
                        results["failed"].append(task)
                        logger.error(f"Task {task_id} failed: {result.get('error')}")
                    
                    # Update knowledge base with results
                    await self._update_knowledge_base(task, result)
            
            # Small delay to prevent busy waiting
            if self.active_tasks and not ready_tasks:
                await asyncio.sleep(0.1)
        
        results["total_time"] = (datetime.now() - start_time).total_seconds()
        logger.info(f"Development execution completed in {results['total_time']:.2f} seconds")
        
        return results
    
    def _get_ready_tasks(self, max_parallel: int) -> List[DevelopmentTask]:
        """Get tasks that are ready to execute"""
        ready_tasks = []
        current_parallel = len(self.active_tasks)
        
        for task in self.task_queue[:]:
            if current_parallel >= max_parallel:
                break
            
            if task.can_execute(self.completed_tasks):
                ready_tasks.append(task)
                self.task_queue.remove(task)
                self.active_tasks[task.id] = task
                task.status = "in_progress"
                current_parallel += 1
        
        return ready_tasks
    
    async def _execute_parallel_tasks(self, tasks: List[DevelopmentTask]) -> Dict[str, Dict[str, Any]]:
        """Execute multiple tasks in parallel"""
        execution_coroutines = []
        task_ids = []
        
        for task in tasks:
            task_ids.append(task.id)
            execution_coroutines.append(self._execute_single_task(task))
        
        results = await asyncio.gather(*execution_coroutines, return_exceptions=True)
        
        return {
            task_id: {"success": True, "result": result} if not isinstance(result, Exception)
            else {"success": False, "error": str(result)}
            for task_id, result in zip(task_ids, results)
        }
    
    async def _execute_single_task(self, task: DevelopmentTask) -> Dict[str, Any]:
        """Execute a single task with the appropriate agent(s)"""
        logger.info(f"Executing task {task.id}: {task.description[:50]}...")
        
        if not task.assigned_agents:
            # Auto-assign based on task description
            task.assigned_agents = await self._auto_assign_agents(task)
        
        # Execute with primary agent
        primary_agent = self.agents.get(task.assigned_agents[0])
        if not primary_agent:
            raise ValueError(f"No agent available for role {task.assigned_agents[0]}")
        
        result = await primary_agent.execute_task(task)
        
        # If multiple agents assigned, have them collaborate
        if len(task.assigned_agents) > 1:
            for agent_role in task.assigned_agents[1:]:
                secondary_agent = self.agents.get(agent_role)
                if secondary_agent:
                    result = await secondary_agent.enhance_result(result, task)
        
        return result
    
    async def _auto_assign_agents(self, task: DevelopmentTask) -> List[AgentRole]:
        """Automatically assign agents based on task description"""
        # Simple keyword-based assignment - can be enhanced with NLP
        description_lower = task.description.lower()
        assigned = []
        
        keyword_map = {
            AgentRole.ARCHITECT: ["design", "architecture", "structure", "pattern"],
            AgentRole.DEVELOPER: ["implement", "code", "create", "build", "develop"],
            AgentRole.TESTER: ["test", "verify", "validate", "check"],
            AgentRole.REVIEWER: ["review", "analyze", "audit", "inspect"],
            AgentRole.DOCUMENTER: ["document", "describe", "explain", "write docs"],
            AgentRole.INTEGRATOR: ["integrate", "deploy", "setup", "configure"],
            AgentRole.REFACTORER: ["refactor", "cleanup", "optimize", "simplify"],
            AgentRole.DEBUGGER: ["debug", "fix", "troubleshoot", "diagnose"]
        }
        
        for role, keywords in keyword_map.items():
            if any(keyword in description_lower for keyword in keywords):
                assigned.append(role)
        
        # Default to developer if no match
        if not assigned:
            assigned = [AgentRole.DEVELOPER]
        
        return assigned
    
    async def _update_knowledge_base(self, task: DevelopmentTask, result: Dict[str, Any]):
        """Update knowledge base with task results for future learning"""
        knowledge_entry = {
            "task_id": task.id,
            "description": task.description,
            "agents": [role.value for role in task.assigned_agents],
            "duration": (task.completed_at - task.created_at).total_seconds() if task.completed_at else None,
            "success": result.get("success", False),
            "learnings": result.get("learnings", {}),
            "patterns": result.get("patterns", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # Store by category
        category = task.assigned_agents[0].value if task.assigned_agents else "general"
        if category not in self.knowledge_base:
            self.knowledge_base[category] = []
        
        self.knowledge_base[category].append(knowledge_entry)
        
        # Persist if configured
        if self.config.get("knowledge_persistence"):
            await self._persist_knowledge()
    
    async def _persist_knowledge(self):
        """Persist knowledge base to disk"""
        knowledge_path = Path("C:/Users/Nouri/Documents/GitHub/ai-agents/knowledge/meta_knowledge.json")
        knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(knowledge_path, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2, default=str)
    
    async def generate_report(self) -> str:
        """Generate a comprehensive development report"""
        report = []
        report.append("=" * 60)
        report.append("META ORCHESTRATOR DEVELOPMENT REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Task statistics
        total_tasks = len(self.completed_tasks) + len(self.active_tasks) + len(self.task_queue)
        report.append(f"Total Tasks: {total_tasks}")
        report.append(f"Completed: {len(self.completed_tasks)}")
        report.append(f"Active: {len(self.active_tasks)}")
        report.append(f"Queued: {len(self.task_queue)}")
        report.append("")
        
        # Agent utilization
        report.append("Agent Utilization:")
        for role, agent in self.agents.items():
            if hasattr(agent, 'get_statistics'):
                stats = await agent.get_statistics()
                report.append(f"  {role.value}: {stats}")
        report.append("")
        
        # Knowledge insights
        report.append("Knowledge Base Insights:")
        for category, entries in self.knowledge_base.items():
            success_rate = sum(1 for e in entries if e.get("success")) / len(entries) * 100 if entries else 0
            report.append(f"  {category}: {len(entries)} entries, {success_rate:.1f}% success rate")
        
        return "\n".join(report)


if __name__ == "__main__":
    async def demo():
        """Demonstration of meta orchestrator capabilities"""
        orchestrator = MetaOrchestrator()
        
        # Example: Develop a new AI agent
        requirement = """
        Create a new financial analysis agent that can:
        1. Process accounting documents
        2. Extract key financial metrics
        3. Generate insights and reports
        4. Integrate with existing accounting systems
        """
        
        # Plan development
        tasks = await orchestrator.plan_development(requirement, {
            "framework": "langchain",
            "target_system": "quickbooks",
            "priority": "high"
        })
        
        print(f"Generated {len(tasks)} development tasks:")
        for task in tasks[:5]:  # Show first 5
            print(f"  - {task.id}: {task.description} (Priority: {task.priority.value})")
        
        # Execute development
        results = await orchestrator.execute_development(max_parallel=3)
        
        print(f"\nExecution Results:")
        print(f"  Completed: {len(results['completed'])}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Time: {results['total_time']:.2f} seconds")
        
        # Generate report
        report = await orchestrator.generate_report()
        print("\n" + report)
    
    # Run demonstration
    asyncio.run(demo())