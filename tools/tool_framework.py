"""
Tool Integration Framework for AI Agents
Provides standardized way to add capabilities to agents
"""

from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import asyncio
import inspect
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories of tools for organization and discovery"""
    DATA_PROCESSING = "data_processing"
    API_INTEGRATION = "api_integration"
    FILE_OPERATIONS = "file_operations"
    WEB_SCRAPING = "web_scraping"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    COMMUNICATION = "communication"
    GITHUB = "github"
    DATABASE = "database"


@dataclass
class ToolMetadata:
    """Metadata for tool discovery and selection"""
    name: str
    description: str
    category: ToolCategory
    parameters: Dict[str, type]
    returns: type
    requires_auth: bool = False
    cost_estimate: float = 0.0  # Estimated cost per use (for API calls)
    reliability: float = 1.0  # 0-1 reliability score
    tags: List[str] = None


class BaseTool(ABC):
    """Base class for all tools"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metadata = self._get_metadata()
        self.execution_count = 0
        self.success_count = 0
        self.total_latency = 0.0
    
    @abstractmethod
    def _get_metadata(self) -> ToolMetadata:
        """Return tool metadata"""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters"""
        pass
    
    async def __call__(self, **kwargs) -> Any:
        """Make tool callable"""
        import time
        start_time = time.time()
        
        try:
            # Validate parameters
            self._validate_parameters(kwargs)
            
            # Execute tool
            result = await self.execute(**kwargs)
            
            # Update metrics
            self.success_count += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Tool {self.metadata.name} execution failed: {e}")
            raise
        finally:
            self.execution_count += 1
            self.total_latency += time.time() - start_time
    
    def _validate_parameters(self, params: Dict[str, Any]) -> None:
        """Validate input parameters against metadata"""
        for param_name, param_type in self.metadata.parameters.items():
            if param_name not in params:
                raise ValueError(f"Missing required parameter: {param_name}")
            
            # Type checking (simplified)
            if not isinstance(params[param_name], param_type):
                raise TypeError(f"Parameter {param_name} must be of type {param_type}")
    
    def get_success_rate(self) -> float:
        """Get tool success rate"""
        if self.execution_count == 0:
            return 0.0
        return self.success_count / self.execution_count
    
    def get_average_latency(self) -> float:
        """Get average execution latency"""
        if self.execution_count == 0:
            return 0.0
        return self.total_latency / self.execution_count


class ToolRegistry:
    """Registry for managing and discovering tools"""
    
    def __init__(self):
        self.tools: Dict[str, BaseTool] = {}
        self.categories: Dict[ToolCategory, List[str]] = {cat: [] for cat in ToolCategory}
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool in the registry"""
        name = tool.metadata.name
        if name in self.tools:
            logger.warning(f"Tool {name} already registered, overwriting")
        
        self.tools[name] = tool
        self.categories[tool.metadata.category].append(name)
        logger.info(f"Registered tool: {name} in category {tool.metadata.category.value}")
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name"""
        return self.tools.get(name)
    
    def search(
        self,
        category: Optional[ToolCategory] = None,
        tags: Optional[List[str]] = None,
        min_reliability: float = 0.0
    ) -> List[BaseTool]:
        """Search for tools based on criteria"""
        results = []
        
        for name, tool in self.tools.items():
            # Filter by category
            if category and tool.metadata.category != category:
                continue
            
            # Filter by tags
            if tags and tool.metadata.tags:
                if not any(tag in tool.metadata.tags for tag in tags):
                    continue
            
            # Filter by reliability
            if tool.metadata.reliability < min_reliability:
                continue
            
            results.append(tool)
        
        return results
    
    def get_by_category(self, category: ToolCategory) -> List[BaseTool]:
        """Get all tools in a category"""
        tool_names = self.categories.get(category, [])
        return [self.tools[name] for name in tool_names if name in self.tools]


class ToolOrchestrator:
    """Orchestrates complex tool chains and workflows"""
    
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.execution_history: List[Dict[str, Any]] = []
    
    async def execute_chain(self, tool_chain: List[Dict[str, Any]]) -> Any:
        """
        Execute a chain of tools where output of one feeds into the next
        
        tool_chain: List of dicts with 'tool_name' and 'params'
        """
        result = None
        
        for step in tool_chain:
            tool_name = step['tool_name']
            params = step.get('params', {})
            
            # Use previous result if specified
            if result is not None and step.get('use_previous_result', False):
                params['input'] = result
            
            tool = self.registry.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            result = await tool(**params)
            
            # Log execution
            self.execution_history.append({
                'tool': tool_name,
                'params': params,
                'result': result,
                'timestamp': datetime.now()
            })
        
        return result
    
    async def execute_parallel(self, tools: List[Dict[str, Any]]) -> List[Any]:
        """Execute multiple tools in parallel"""
        tasks = []
        
        for tool_spec in tools:
            tool_name = tool_spec['tool_name']
            params = tool_spec.get('params', {})
            
            tool = self.registry.get(tool_name)
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            
            tasks.append(tool(**params))
        
        return await asyncio.gather(*tasks)
    
    async def execute_conditional(
        self,
        condition_tool: str,
        condition_params: Dict[str, Any],
        true_branch: List[Dict[str, Any]],
        false_branch: List[Dict[str, Any]]
    ) -> Any:
        """Execute tools conditionally based on a condition tool's result"""
        # Evaluate condition
        condition_tool_obj = self.registry.get(condition_tool)
        if not condition_tool_obj:
            raise ValueError(f"Condition tool {condition_tool} not found")
        
        condition_result = await condition_tool_obj(**condition_params)
        
        # Execute appropriate branch
        if condition_result:
            return await self.execute_chain(true_branch)
        else:
            return await self.execute_chain(false_branch)


# Example Tool Implementations

class GitHubTool(BaseTool):
    """GitHub integration tool leveraging claude-code-action patterns"""
    
    def _get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="github_tool",
            description="Integrate with GitHub for code review, issue management, etc.",
            category=ToolCategory.GITHUB,
            parameters={
                "action": str,
                "context": dict
            },
            returns=dict,
            requires_auth=True,
            tags=["github", "code_review", "issues", "pull_requests"]
        )
    
    async def execute(self, action: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute GitHub action"""
        actions = {
            'code_review': self._code_review,
            'issue_resolution': self._issue_resolution,
            'documentation_update': self._documentation_update,
            'test_generation': self._test_generation
        }
        
        if action not in actions:
            raise ValueError(f"Unknown action: {action}")
        
        return await actions[action](context)
    
    async def _code_review(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform AI-powered code review"""
        # Implementation would integrate with GitHub API
        return {
            "status": "completed",
            "findings": ["Consider error handling", "Add type hints"],
            "suggestions": ["Refactor for clarity"]
        }
    
    async def _issue_resolution(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Help resolve GitHub issues"""
        return {
            "status": "analyzed",
            "proposed_solution": "Implementation suggestion here",
            "related_issues": []
        }
    
    async def _documentation_update(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Update documentation based on code changes"""
        return {
            "status": "updated",
            "files_modified": ["README.md", "docs/api.md"]
        }
    
    async def _test_generation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate tests for code"""
        return {
            "status": "generated",
            "test_count": 5,
            "coverage_increase": "15%"
        }


class DataAnalysisTool(BaseTool):
    """Data analysis tool for processing and analyzing data"""
    
    def _get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="data_analysis",
            description="Analyze data and extract insights",
            category=ToolCategory.ANALYSIS,
            parameters={
                "data": list,
                "analysis_type": str
            },
            returns=dict,
            tags=["analysis", "statistics", "insights"]
        )
    
    async def execute(self, data: List[Any], analysis_type: str) -> Dict[str, Any]:
        """Perform data analysis"""
        import statistics
        
        if analysis_type == "statistical":
            return {
                "mean": statistics.mean(data) if data else 0,
                "median": statistics.median(data) if data else 0,
                "std_dev": statistics.stdev(data) if len(data) > 1 else 0
            }
        elif analysis_type == "pattern":
            # Simplified pattern detection
            return {
                "patterns": ["increasing trend", "seasonal variation"],
                "anomalies": []
            }
        else:
            return {"error": "Unknown analysis type"}


class CodeGenerationTool(BaseTool):
    """Code generation tool using AI"""
    
    def _get_metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name="code_generation",
            description="Generate code based on specifications",
            category=ToolCategory.CODE_GENERATION,
            parameters={
                "specification": str,
                "language": str
            },
            returns=str,
            cost_estimate=0.01,  # Estimated API cost
            tags=["code", "generation", "ai"]
        )
    
    async def execute(self, specification: str, language: str) -> str:
        """Generate code based on specification"""
        # This would integrate with Claude or other code generation models
        template = f"""
def generated_function():
    \"\"\"
    Generated based on: {specification}
    Language: {language}
    \"\"\"
    # Implementation would go here
    pass
"""
        return template


# Tool Adapter for External Libraries

class ToolAdapter:
    """Adapts external functions/libraries to work as tools"""
    
    @staticmethod
    def from_function(
        func: Callable,
        name: str,
        description: str,
        category: ToolCategory,
        tags: Optional[List[str]] = None
    ) -> BaseTool:
        """Create a tool from a regular function"""
        
        class AdaptedTool(BaseTool):
            def _get_metadata(self) -> ToolMetadata:
                # Extract parameter types from function signature
                sig = inspect.signature(func)
                params = {
                    param.name: param.annotation if param.annotation != inspect.Parameter.empty else Any
                    for param in sig.parameters.values()
                }
                
                return ToolMetadata(
                    name=name,
                    description=description,
                    category=category,
                    parameters=params,
                    returns=sig.return_annotation if sig.return_annotation != inspect.Signature.empty else Any,
                    tags=tags or []
                )
            
            async def execute(self, **kwargs) -> Any:
                if asyncio.iscoroutinefunction(func):
                    return await func(**kwargs)
                else:
                    return func(**kwargs)
        
        return AdaptedTool()


# Initialize global registry
global_tool_registry = ToolRegistry()

# Register example tools
global_tool_registry.register(GitHubTool())
global_tool_registry.register(DataAnalysisTool())
global_tool_registry.register(CodeGenerationTool())


from datetime import datetime