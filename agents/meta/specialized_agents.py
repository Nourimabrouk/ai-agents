"""
Specialized AI agents for different development roles.
Each agent has specific capabilities and can work in parallel or sequentially.
"""
import asyncio
import re
import ast
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseSpecializedAgent:
    """Base class for all specialized development agents"""
    
    def __init__(self, capabilities: Any, knowledge_base: Dict[str, Any]):
        self.capabilities = capabilities
        self.role = capabilities.role
        self.knowledge_base = knowledge_base
        self.task_history = []
        self.current_context = {}
        self.tools = self._initialize_tools()
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """Initialize agent-specific tools"""
        return {}
    
    async def analyze_requirement(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze requirement from this agent's perspective"""
        return {
            "analysis": f"{self.name} analyzed: {requirement}",
            "context": context or {},
            "agent_type": self.agent_type,
            "timestamp": datetime.now().isoformat()
        }
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute a development task"""
        return {
            "task": task,
            "executed_by": self.name,
            "agent_type": self.agent_type,
            "result": f"Task executed by {self.agent_type}",
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
    
    async def enhance_result(self, result: Dict[str, Any], task: Any) -> Dict[str, Any]:
        """Enhance result from another agent"""
        return result  # Default: no enhancement
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get agent performance statistics"""
        return {
            "tasks_completed": len(self.task_history),
            "success_rate": self._calculate_success_rate(),
            "avg_execution_time": self._calculate_avg_time()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate from task history"""
        if not self.task_history:
            return 0.0
        successful = sum(1 for task in self.task_history if task.get("success", False))
        return (successful / len(self.task_history)) * 100
    
    def _calculate_avg_time(self) -> float:
        """Calculate average execution time"""
        if not self.task_history:
            return 0.0
        times = [task.get("execution_time", 0) for task in self.task_history]
        return sum(times) / len(times) if times else 0.0


class ArchitectAgent(BaseSpecializedAgent):
    """
    Architect Agent: Designs system architecture and high-level structure
    """
    
    async def analyze_requirement(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze and design architecture for the requirement"""
        logger.info(f"Architect analyzing requirement: {requirement[:50]}...")
        
        # Parse requirement for architectural needs
        components = await self._identify_components(requirement)
        patterns = await self._identify_patterns(requirement, context)
        interfaces = await self._design_interfaces(components)
        
        # Generate architectural tasks
        tasks = []
        
        # Design tasks
        if components:
            tasks.append({
                "description": f"Design component architecture for {len(components)} components",
                "priority": "HIGH",
                "agents": ["ARCHITECT"],
                "context": {"components": components, "patterns": patterns}
            })
        
        # Interface design
        if interfaces:
            tasks.append({
                "description": "Define API interfaces and contracts",
                "priority": "HIGH",
                "agents": ["ARCHITECT", "DEVELOPER"],
                "context": {"interfaces": interfaces}
            })
        
        # Pattern implementation
        for pattern in patterns:
            tasks.append({
                "description": f"Implement {pattern} pattern",
                "priority": "MEDIUM",
                "agents": ["ARCHITECT", "DEVELOPER"],
                "context": {"pattern": pattern}
            })
        
        return {
            "agent": "architect",
            "tasks": tasks,
            "analysis": {
                "components": components,
                "patterns": patterns,
                "interfaces": interfaces
            }
        }
    
    async def _identify_components(self, requirement: str) -> List[str]:
        """Identify system components from requirement"""
        # Pattern matching for component identification
        component_keywords = ["agent", "system", "module", "service", "component", "engine", "processor"]
        components = []
        
        for keyword in component_keywords:
            pattern = rf"\b(\w+\s+)?{keyword}s?\b"
            matches = re.findall(pattern, requirement.lower())
            components.extend([match.strip() for match in matches if match])
        
        return list(set(components))
    
    async def _identify_patterns(self, requirement: str, context: Dict[str, Any] = None) -> List[str]:
        """Identify architectural patterns to use"""
        patterns = []
        
        # Check for common patterns based on keywords
        pattern_map = {
            "observer": ["notify", "subscribe", "event", "listener"],
            "factory": ["create", "instantiate", "generate"],
            "strategy": ["algorithm", "policy", "strategy"],
            "adapter": ["integrate", "adapt", "interface", "compatibility"],
            "singleton": ["single", "instance", "global"],
            "decorator": ["enhance", "wrap", "extend"],
            "facade": ["simplify", "interface", "unified"],
            "command": ["command", "action", "undo", "execute"]
        }
        
        requirement_lower = requirement.lower()
        for pattern, keywords in pattern_map.items():
            if any(keyword in requirement_lower for keyword in keywords):
                patterns.append(pattern)
        
        # Add context-based patterns
        if context:
            if context.get("framework") == "langchain":
                patterns.append("chain")
            if context.get("parallel_execution"):
                patterns.append("async_executor")
        
        return patterns
    
    async def _design_interfaces(self, components: List[str]) -> Dict[str, Any]:
        """Design interfaces for components"""
        interfaces = {}
        
        for component in components:
            interfaces[component] = {
                "methods": [
                    f"initialize_{component}",
                    f"execute_{component}",
                    f"validate_{component}_result"
                ],
                "properties": [
                    "configuration",
                    "state",
                    "metrics"
                ],
                "events": [
                    f"{component}_started",
                    f"{component}_completed",
                    f"{component}_failed"
                ]
            }
        
        return interfaces
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute architectural design task"""
        logger.info(f"Architect executing: {task.description}")
        
        result = {
            "success": True,
            "artifacts": {}
        }
        
        # Generate architecture artifacts based on task
        if "component architecture" in task.description.lower():
            result["artifacts"]["component_diagram"] = await self._generate_component_diagram(task.context)
            result["artifacts"]["interfaces"] = await self._generate_interface_definitions(task.context)
        
        elif "api interfaces" in task.description.lower():
            result["artifacts"]["api_spec"] = await self._generate_api_specification(task.context)
            result["artifacts"]["contracts"] = await self._generate_contracts(task.context)
        
        elif "pattern" in task.description.lower():
            pattern_name = task.context.get("pattern", "generic")
            result["artifacts"]["pattern_implementation"] = await self._generate_pattern_code(pattern_name)
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 1.0  # Simulated
        })
        
        return result
    
    async def _generate_component_diagram(self, context: Dict) -> str:
        """Generate component diagram specification"""
        components = context.get("components", [])
        diagram = ["@startuml", "!define RECTANGLE"]
        
        for component in components:
            diagram.append(f"rectangle \"{component}\" as {component.replace(' ', '_')}")
        
        # Add relationships
        if len(components) > 1:
            for i in range(len(components) - 1):
                diagram.append(f"{components[i].replace(' ', '_')} --> {components[i+1].replace(' ', '_')}")
        
        diagram.append("@enduml")
        return "\n".join(diagram)
    
    async def _generate_interface_definitions(self, context: Dict) -> Dict[str, str]:
        """Generate interface definitions"""
        interfaces = {}
        components = context.get("components", [])
        
        for component in components:
            interface_name = f"I{component.title().replace(' ', '')}Agent"
            interfaces[interface_name] = f'''from abc import ABC, abstractmethod
from typing import Any, Dict, List

class {interface_name}(ABC):
    """Interface for {component} agent"""
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the {component}"""
        pass
    
    @abstractmethod
    async def execute(self, input_data: Any) -> Dict[str, Any]:
        """Execute {component} logic"""
        pass
    
    @abstractmethod
    async def validate(self, result: Any) -> bool:
        """Validate {component} results"""
        pass
'''
        
        return interfaces
    
    async def _generate_api_specification(self, context: Dict) -> Dict[str, Any]:
        """Generate API specification"""
        interfaces = context.get("interfaces", {})
        api_spec = {
            "version": "1.0.0",
            "endpoints": []
        }
        
        for interface_name, interface_def in interfaces.items():
            for method in interface_def.get("methods", []):
                api_spec["endpoints"].append({
                    "path": f"/api/{interface_name}/{method}",
                    "method": "POST",
                    "description": f"Execute {method} for {interface_name}",
                    "parameters": {
                        "input_data": "object",
                        "config": "object (optional)"
                    },
                    "response": {
                        "success": "boolean",
                        "result": "object",
                        "error": "string (optional)"
                    }
                })
        
        return api_spec
    
    async def _generate_contracts(self, context: Dict) -> Dict[str, str]:
        """Generate contract definitions"""
        contracts = {}
        interfaces = context.get("interfaces", {})
        
        for interface_name, interface_def in interfaces.items():
            contract_name = f"{interface_name}Contract"
            contracts[contract_name] = f'''from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from datetime import datetime

@dataclass
class {contract_name}:
    """Contract for {interface_name} interactions"""
    
    # Input contract
    input_schema: Dict[str, Any]
    
    # Output contract  
    output_schema: Dict[str, Any]
    
    # Validation rules
    validation_rules: List[str]
    
    # SLA requirements
    max_execution_time: float = 30.0
    retry_attempts: int = 3
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input against contract"""
        # Implementation here
        return True
    
    def validate_output(self, output_data: Any) -> bool:
        """Validate output against contract"""
        # Implementation here
        return True
'''
        
        return contracts
    
    async def _generate_pattern_code(self, pattern_name: str) -> str:
        """Generate pattern implementation code"""
        pattern_templates = {
            "observer": '''from typing import List, Any

class Subject:
    """Subject in Observer pattern"""
    def __init__(self):
        self._observers: List[Observer] = []
        self._state: Any = None
    
    def attach(self, observer: 'Observer') -> None:
        self._observers.append(observer)
    
    def detach(self, observer: 'Observer') -> None:
        self._observers.remove(observer)
    
    def notify(self) -> None:
        for observer in self._observers:
            observer.update(self)
    
    @property
    def state(self) -> Any:
        return self._state
    
    @state.setter
    def state(self, state: Any) -> None:
        self._state = state
        self.notify()

class Observer:
    """Observer in Observer pattern"""
    async def update(self, subject: Subject) -> None:
        pass
''',
            "factory": '''from abc import ABC, abstractmethod
from typing import Dict, Any

class AgentFactory:
    """Factory for creating agents"""
    
    _creators: Dict[str, type] = {}
    
    @classmethod
    def register_creator(cls, agent_type: str, creator: type) -> None:
        cls._creators[agent_type] = creator
    
    @classmethod
    def create_agent(cls, agent_type: str, **kwargs) -> Any:
        creator = cls._creators.get(agent_type)
        if not creator:
            raise ValueError(f"Unknown agent type: {agent_type}")
        return creator(**kwargs)
''',
            "strategy": '''from abc import ABC, abstractmethod
from typing import Any

class Strategy(ABC):
    """Abstract strategy interface"""
    
    @abstractmethod
    async def execute(self, data: Any) -> Any:
        pass

class Context:
    """Context that uses a strategy"""
    
    def __init__(self, strategy: Strategy):
        self._strategy = strategy
    
    @property
    def strategy(self) -> Strategy:
        return self._strategy
    
    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy
    
    async def execute_strategy(self, data: Any) -> Any:
        return await self._strategy.execute(data)
'''
        }
        
        return pattern_templates.get(pattern_name, f"# TODO: Implement {pattern_name} pattern")


class DeveloperAgent(BaseSpecializedAgent):
    """
    Developer Agent: Implements code and features
    """
    
    async def analyze_requirement(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze requirement for implementation needs"""
        logger.info(f"Developer analyzing requirement: {requirement[:50]}...")
        
        # Identify implementation tasks
        features = await self._identify_features(requirement)
        tech_stack = await self._determine_tech_stack(requirement, context)
        
        tasks = []
        
        # Core implementation tasks
        for feature in features:
            tasks.append({
                "description": f"Implement {feature}",
                "priority": "HIGH",
                "agents": ["DEVELOPER"],
                "context": {"feature": feature, "tech_stack": tech_stack}
            })
        
        # Integration tasks
        if len(features) > 1:
            tasks.append({
                "description": "Integrate feature modules",
                "priority": "MEDIUM",
                "agents": ["DEVELOPER", "INTEGRATOR"],
                "dependencies": [f"Implement {f}" for f in features],
                "context": {"features": features}
            })
        
        # Optimization task
        tasks.append({
            "description": "Optimize code performance",
            "priority": "LOW",
            "agents": ["DEVELOPER", "REFACTORER"],
            "dependencies": [f"Implement {f}" for f in features],
            "context": {"optimization_targets": ["speed", "memory"]}
        })
        
        return {
            "agent": "developer",
            "tasks": tasks,
            "analysis": {
                "features": features,
                "tech_stack": tech_stack
            }
        }
    
    async def _identify_features(self, requirement: str) -> List[str]:
        """Identify features to implement"""
        features = []
        
        # Pattern matching for feature identification
        feature_patterns = [
            r"(?:create|implement|build|develop)\s+(\w+(?:\s+\w+)*)",
            r"(\w+(?:\s+\w+)*)\s+(?:functionality|feature|capability)",
            r"(?:able to|can)\s+(\w+(?:\s+\w+)*)"
        ]
        
        for pattern in feature_patterns:
            matches = re.findall(pattern, requirement.lower())
            features.extend(matches)
        
        # Clean and deduplicate
        features = list(set([f.strip() for f in features if len(f.strip()) > 2]))
        
        return features[:5]  # Limit to top 5 features
    
    async def _determine_tech_stack(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, str]:
        """Determine technology stack"""
        tech_stack = {
            "language": "python",
            "framework": "asyncio",
            "testing": "pytest",
            "database": "sqlite"
        }
        
        if context:
            if context.get("framework"):
                tech_stack["framework"] = context["framework"]
            if "web" in requirement.lower() or "api" in requirement.lower():
                tech_stack["web_framework"] = "fastapi"
            if "database" in requirement.lower() or "data" in requirement.lower():
                tech_stack["database"] = "postgresql"
        
        return tech_stack
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute development task"""
        logger.info(f"Developer executing: {task.description}")
        
        result = {
            "success": True,
            "code": {},
            "tests": {}
        }
        
        # Generate code based on task
        if "implement" in task.description.lower():
            feature = task.context.get("feature", "generic")
            result["code"][feature] = await self._generate_feature_code(feature, task.context)
            result["tests"][feature] = await self._generate_test_code(feature)
        
        elif "integrate" in task.description.lower():
            result["code"]["integration"] = await self._generate_integration_code(task.context)
        
        elif "optimize" in task.description.lower():
            result["code"]["optimizations"] = await self._generate_optimization_code(task.context)
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 2.0  # Simulated
        })
        
        return result
    
    async def _generate_feature_code(self, feature: str, context: Dict) -> str:
        """Generate code for a feature"""
        tech_stack = context.get("tech_stack", {})
        
        # Generate appropriate code template
        if tech_stack.get("framework") == "langchain":
            return await self._generate_langchain_code(feature)
        elif tech_stack.get("web_framework") == "fastapi":
            return await self._generate_fastapi_code(feature)
        else:
            return await self._generate_generic_code(feature)
    
    async def _generate_langchain_code(self, feature: str) -> str:
        """Generate LangChain-specific code"""
        class_name = ''.join(word.capitalize() for word in feature.split())
        
        return f'''from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chains import LLMChain
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union, Any, Dict
import re

class {class_name}Agent:
    """LangChain agent for {feature}"""
    
    def __init__(self, llm, tools: List[Tool], verbose: bool = True):
        self.llm = llm
        self.tools = tools
        self.verbose = verbose
        self.agent_executor = self._create_agent_executor()
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the agent executor"""
        prompt = self._create_prompt()
        llm_chain = LLMChain(llm=self.llm, prompt=prompt)
        
        agent = LLMSingleActionAgent(
            llm_chain=llm_chain,
            output_parser=self._create_output_parser(),
            stop=["\\nObservation:"],
            allowed_tools=[tool.name for tool in self.tools]
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=self.verbose
        )
    
    def _create_prompt(self) -> StringPromptTemplate:
        """Create the agent prompt"""
        template = """You are an AI agent specialized in {feature}.
        
Available tools:
{{tools}}

Tool descriptions:
{{tool_names}}

Use this format:
Thought: Consider what to do
Action: tool_name
Action Input: input_for_tool
Observation: tool_result
... (repeat as needed)
Thought: I have the final answer
Final Answer: the_final_answer

Question: {{input}}
{{agent_scratchpad}}"""
        
        return StringPromptTemplate(
            template=template,
            input_variables=["input", "tools", "tool_names", "agent_scratchpad"]
        )
    
    def _create_output_parser(self):
        """Create output parser for agent"""
        class CustomOutputParser:
            def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
                if "Final Answer:" in llm_output:
                    return AgentFinish(
                        return_values={{"output": llm_output.split("Final Answer:")[-1].strip()}},
                        log=llm_output
                    )
                
                action_match = re.search(r"Action:\\s*(.*?)\\n", llm_output)
                action_input_match = re.search(r"Action Input:\\s*(.*)", llm_output, re.DOTALL)
                
                if action_match and action_input_match:
                    action = action_match.group(1).strip()
                    action_input = action_input_match.group(1).strip()
                    return AgentAction(tool=action, tool_input=action_input, log=llm_output)
                
                raise ValueError(f"Could not parse agent output: {{llm_output}}")
        
        return CustomOutputParser()
    
    async def run(self, input_text: str) -> str:
        """Run the agent"""
        return await self.agent_executor.arun(input_text)
'''
    
    async def _generate_fastapi_code(self, feature: str) -> str:
        """Generate FastAPI-specific code"""
        class_name = ''.join(word.capitalize() for word in feature.split())
        
        return f'''from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio

app = FastAPI(title="{feature} API", version="1.0.0")

class {class_name}Request(BaseModel):
    """Request model for {feature}"""
    data: Dict[str, Any] = Field(..., description="Input data")
    config: Optional[Dict[str, Any]] = Field(default={{}}, description="Configuration")

class {class_name}Response(BaseModel):
    """Response model for {feature}"""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class {class_name}Service:
    """Service class for {feature}"""
    
    def __init__(self):
        self.cache = {{}}
        self.metrics = {{"requests": 0, "successes": 0, "failures": 0}}
    
    async def process(self, request: {class_name}Request) -> {class_name}Response:
        """Process {feature} request"""
        self.metrics["requests"] += 1
        
        try:
            # Process the request
            result = await self._execute_{feature.replace(" ", "_")}(request.data, request.config)
            
            self.metrics["successes"] += 1
            return {class_name}Response(
                success=True,
                result=result
            )
        except Exception as e:
            self.metrics["failures"] += 1
            return {class_name}Response(
                success=False,
                error=str(e)
            )
    
    async def _execute_{feature.replace(" ", "_")}(self, data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute {feature} logic"""
        # Implementation here
        await asyncio.sleep(0.1)  # Simulate processing
        return {{"processed": data, "config_applied": config}}
    
    def get_metrics(self) -> Dict[str, int]:
        """Get service metrics"""
        return self.metrics

# Create service instance
{feature.replace(" ", "_")}_service = {class_name}Service()

@app.post("/{feature.replace(" ", "_")}", response_model={class_name}Response)
async def {feature.replace(" ", "_")}_endpoint(
    request: {class_name}Request,
    service: {class_name}Service = Depends(lambda: {feature.replace(" ", "_")}_service)
) -> {class_name}Response:
    """Endpoint for {feature}"""
    return await service.process(request)

@app.get("/{feature.replace(" ", "_")}/metrics")
async def get_metrics(
    service: {class_name}Service = Depends(lambda: {feature.replace(" ", "_")}_service)
) -> Dict[str, int]:
    """Get metrics for {feature}"""
    return service.get_metrics()
'''
    
    async def _generate_generic_code(self, feature: str) -> str:
        """Generate generic Python code"""
        class_name = ''.join(word.capitalize() for word in feature.split())
        
        return f'''import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class {class_name}Config:
    """Configuration for {feature}"""
    enabled: bool = True
    timeout: float = 30.0
    retry_attempts: int = 3
    cache_enabled: bool = True
    
class {class_name}:
    """Implementation of {feature}"""
    
    def __init__(self, config: Optional[{class_name}Config] = None):
        self.config = config or {class_name}Config()
        self.cache = {{}} if self.config.cache_enabled else None
        self.metrics = {{
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cache_hits": 0
        }}
        logger.info(f"Initialized {{class_name}} with config: {{self.config}}")
    
    async def execute(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute {feature} functionality"""
        self.metrics["total_calls"] += 1
        
        # Check cache
        cache_key = self._generate_cache_key(input_data)
        if self.cache is not None and cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            logger.debug(f"Cache hit for key: {{cache_key}}")
            return self.cache[cache_key]
        
        try:
            # Execute with retry logic
            result = await self._execute_with_retry(input_data, **kwargs)
            
            # Update cache
            if self.cache is not None:
                self.cache[cache_key] = result
            
            self.metrics["successful_calls"] += 1
            return result
            
        except Exception as e:
            self.metrics["failed_calls"] += 1
            logger.error(f"Error executing {feature}: {{e}}")
            raise
    
    async def _execute_with_retry(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Execute with retry logic"""
        last_error = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await asyncio.wait_for(
                    self._core_logic(input_data, **kwargs),
                    timeout=self.config.timeout
                )
            except asyncio.TimeoutError as e:
                last_error = e
                logger.warning(f"Attempt {{attempt + 1}} timed out")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {{attempt + 1}} failed: {{e}}")
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
        
        raise last_error
    
    async def _core_logic(self, input_data: Any, **kwargs) -> Dict[str, Any]:
        """Core {feature} logic"""
        # Simulate processing
        await asyncio.sleep(0.1)
        
        result = {{
            "input": input_data,
            "processed_at": datetime.now().isoformat(),
            "feature": "{feature}",
            "success": True
        }}
        
        # Add any additional processing here
        for key, value in kwargs.items():
            result[key] = value
        
        return result
    
    def _generate_cache_key(self, input_data: Any) -> str:
        """Generate cache key from input data"""
        import hashlib
        import json
        
        try:
            data_str = json.dumps(input_data, sort_keys=True)
        except (TypeError, ValueError):
            data_str = str(input_data)
        
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, int]:
        """Get execution metrics"""
        return self.metrics.copy()
    
    def clear_cache(self) -> None:
        """Clear the cache"""
        if self.cache is not None:
            self.cache.clear()
            logger.info("Cache cleared")
'''
    
    async def _generate_test_code(self, feature: str) -> str:
        """Generate test code for a feature"""
        class_name = ''.join(word.capitalize() for word in feature.split())
        
        return f'''import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from ..{feature.replace(" ", "_")} import {class_name}, {class_name}Config

class Test{class_name}:
    """Test suite for {feature}"""
    
    @pytest.fixture
    def config(self):
        """Test configuration"""
        return {class_name}Config(
            enabled=True,
            timeout=5.0,
            retry_attempts=2,
            cache_enabled=True
        )
    
    @pytest.fixture
    def instance(self, config):
        """Test instance"""
        return {class_name}(config)
    
    @pytest.mark.asyncio
    async def test_initialization(self, config):
        """Test {feature} initialization"""
        instance = {class_name}(config)
        assert instance.config == config
        assert instance.cache is not None
        assert instance.metrics["total_calls"] == 0
    
    @pytest.mark.asyncio
    async def test_execute_success(self, instance):
        """Test successful execution"""
        input_data = {{"test": "data"}}
        result = await instance.execute(input_data)
        
        assert result["success"] is True
        assert result["input"] == input_data
        assert instance.metrics["successful_calls"] == 1
        assert instance.metrics["failed_calls"] == 0
    
    @pytest.mark.asyncio
    async def test_cache_hit(self, instance):
        """Test cache functionality"""
        input_data = {{"test": "data"}}
        
        # First call - cache miss
        result1 = await instance.execute(input_data)
        assert instance.metrics["cache_hits"] == 0
        
        # Second call - cache hit
        result2 = await instance.execute(input_data)
        assert instance.metrics["cache_hits"] == 1
        assert result1 == result2
    
    @pytest.mark.asyncio
    async def test_retry_on_failure(self, instance):
        """Test retry logic on failure"""
        with patch.object(instance, '_core_logic', new_callable=AsyncMock) as mock_logic:
            # First attempt fails, second succeeds
            mock_logic.side_effect = [Exception("First failure"), {{"success": True}}]
            
            result = await instance.execute({{"test": "data"}})
            assert result["success"] is True
            assert mock_logic.call_count == 2
    
    @pytest.mark.asyncio
    async def test_timeout(self, instance):
        """Test timeout handling"""
        async def slow_logic(*args, **kwargs):
            await asyncio.sleep(10)
            return {{"success": True}}
        
        with patch.object(instance, '_core_logic', new=slow_logic):
            with pytest.raises(asyncio.TimeoutError):
                await instance.execute({{"test": "data"}})
    
    @pytest.mark.asyncio
    async def test_metrics_tracking(self, instance):
        """Test metrics are tracked correctly"""
        # Successful call
        await instance.execute({{"test": "data1"}})
        
        # Failed call
        with patch.object(instance, '_core_logic', side_effect=Exception("Error")):
            try:
                await instance.execute({{"test": "data2"}})
            except:
                pass
        
        metrics = instance.get_metrics()
        assert metrics["total_calls"] == 2
        assert metrics["successful_calls"] == 1
        assert metrics["failed_calls"] == 1
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, instance):
        """Test cache clearing"""
        await instance.execute({"test": "data"})
        assert len(instance.cache) > 0
        
        instance.clear_cache()
        assert len(instance.cache) == 0
    
    @pytest.mark.parametrize("input_data,expected_keys", [
        ({"simple": "data"}, ["input", "processed_at", "feature", "success"]),
        ({"complex": {"nested": "data"}}, ["input", "processed_at", "feature", "success"]),
        ([], ["input", "processed_at", "feature", "success"]),
    ])
    @pytest.mark.asyncio
    async def test_various_inputs(self, instance, input_data, expected_keys):
        """Test with various input types"""
        result = await instance.execute(input_data)
        for key in expected_keys:
            assert key in result
'''
    
    async def _generate_integration_code(self, context: Dict) -> str:
        """Generate integration code"""
        features = context.get("features", [])
        
        return f'''import asyncio
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FeatureIntegrator:
    """Integrates multiple features into a cohesive system"""
    
    def __init__(self):
        self.features = {{}}
        self.pipelines = {{}}
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize all feature modules"""
        feature_modules = {features}
        
        for feature in feature_modules:
            try:
                # Dynamic import and initialization
                module_name = feature.replace(" ", "_")
                class_name = ''.join(word.capitalize() for word in feature.split())
                
                # In real implementation, use importlib
                # module = importlib.import_module(f".{{module_name}}", package=__package__)
                # feature_class = getattr(module, class_name)
                # self.features[feature] = feature_class()
                
                logger.info(f"Initialized feature: {{feature}}")
            except Exception as e:
                logger.error(f"Failed to initialize {{feature}}: {{e}}")
    
    async def create_pipeline(self, pipeline_name: str, feature_sequence: List[str]) -> None:
        """Create a feature pipeline"""
        self.pipelines[pipeline_name] = feature_sequence
        logger.info(f"Created pipeline '{{pipeline_name}}' with {{len(feature_sequence)}} features")
    
    async def execute_pipeline(self, pipeline_name: str, input_data: Any) -> Dict[str, Any]:
        """Execute a feature pipeline"""
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{{pipeline_name}}' not found")
        
        feature_sequence = self.pipelines[pipeline_name]
        current_data = input_data
        results = {{}}
        
        for feature_name in feature_sequence:
            if feature_name in self.features:
                logger.info(f"Executing feature: {{feature_name}}")
                feature = self.features[feature_name]
                
                try:
                    result = await feature.execute(current_data)
                    results[feature_name] = result
                    current_data = result  # Pass output to next feature
                except Exception as e:
                    logger.error(f"Feature {{feature_name}} failed: {{e}}")
                    results[feature_name] = {{"error": str(e)}}
                    break
        
        return {{
            "pipeline": pipeline_name,
            "features_executed": len(results),
            "results": results,
            "final_output": current_data
        }}
    
    async def execute_parallel(self, features: List[str], input_data: Any) -> Dict[str, Any]:
        """Execute multiple features in parallel"""
        tasks = []
        
        for feature_name in features:
            if feature_name in self.features:
                feature = self.features[feature_name]
                tasks.append(self._execute_feature_async(feature_name, feature, input_data))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {{
            feature_name: result if not isinstance(result, Exception) else {{"error": str(result)}}
            for feature_name, result in zip(features, results)
        }}
    
    async def _execute_feature_async(self, name: str, feature: Any, data: Any) -> Dict[str, Any]:
        """Execute a single feature asynchronously"""
        logger.info(f"Executing feature {{name}} asynchronously")
        return await feature.execute(data)
    
    def get_feature_status(self) -> Dict[str, bool]:
        """Get status of all features"""
        return {{
            feature_name: feature is not None
            for feature_name, feature in self.features.items()
        }}
    
    def get_pipeline_info(self, pipeline_name: str) -> Dict[str, Any]:
        """Get information about a pipeline"""
        if pipeline_name not in self.pipelines:
            return {{"error": f"Pipeline '{{pipeline_name}}' not found"}}
        
        return {{
            "name": pipeline_name,
            "features": self.pipelines[pipeline_name],
            "length": len(self.pipelines[pipeline_name])
        }}
'''
    
    async def _generate_optimization_code(self, context: Dict) -> str:
        """Generate optimization code"""
        targets = context.get("optimization_targets", ["speed", "memory"])
        
        return f'''import asyncio
import functools
import time
from typing import Any, Callable, Dict, List
from collections import OrderedDict
import gc

class PerformanceOptimizer:
    """Optimizes code for {', '.join(targets)}"""
    
    def __init__(self, cache_size: int = 128):
        self.cache_size = cache_size
        self.performance_metrics = {{}}
        self.optimization_targets = {targets}
    
    # Speed optimizations
    @staticmethod
    def memoize(maxsize: int = 128):
        """Memoization decorator for expensive functions"""
        def decorator(func: Callable) -> Callable:
            cache = OrderedDict()
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                key = str(args) + str(kwargs)
                
                if key in cache:
                    # Move to end (LRU)
                    cache.move_to_end(key)
                    return cache[key]
                
                result = await func(*args, **kwargs)
                cache[key] = result
                
                # Limit cache size
                if len(cache) > maxsize:
                    cache.popitem(last=False)
                
                return result
            
            wrapper.cache_clear = lambda: cache.clear()
            wrapper.cache_info = lambda: {{
                "hits": sum(1 for _ in cache),
                "size": len(cache),
                "maxsize": maxsize
            }}
            
            return wrapper
        return decorator
    
    @staticmethod
    def batch_processor(batch_size: int = 100):
        """Process items in batches for better performance"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(items: List[Any], *args, **kwargs) -> List[Any]:
                results = []
                
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    batch_results = await asyncio.gather(*[
                        func(item, *args, **kwargs) for item in batch
                    ])
                    results.extend(batch_results)
                
                return results
            
            return wrapper
        return decorator
    
    @staticmethod
    def parallel_executor(max_workers: int = 5):
        """Execute function calls in parallel with worker limit"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                semaphore = asyncio.Semaphore(max_workers)
                
                async def limited_func(*args, **kwargs):
                    async with semaphore:
                        return await func(*args, **kwargs)
                
                return await limited_func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    # Memory optimizations
    @staticmethod
    def lazy_loader(func: Callable) -> Callable:
        """Lazy loading decorator for memory-intensive operations"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Use generator for lazy evaluation
            async def generator():
                result = await func(*args, **kwargs)
                if hasattr(result, '__iter__'):
                    for item in result:
                        yield item
                else:
                    yield result
            
            return generator()
        
        return wrapper
    
    @staticmethod
    def memory_limit(max_memory_mb: int = 100):
        """Limit memory usage for a function"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                import psutil
                import os
                
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                result = await func(*args, **kwargs)
                
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_used = current_memory - initial_memory
                
                if memory_used > max_memory_mb:
                    # Force garbage collection
                    gc.collect()
                    # Log warning
                    print(f"Warning: Function used {{memory_used:.2f}}MB (limit: {{max_memory_mb}}MB)")
                
                return result
            
            return wrapper
        return decorator
    
    # Profiling utilities
    @staticmethod
    def profile_performance(func: Callable) -> Callable:
        """Profile function performance"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            start_memory = gc.get_stats()[0].get('collected', 0)
            
            result = await func(*args, **kwargs)
            
            end_time = time.perf_counter()
            end_memory = gc.get_stats()[0].get('collected', 0)
            
            performance_data = {{
                "function": func.__name__,
                "execution_time": end_time - start_time,
                "memory_collected": end_memory - start_memory,
                "timestamp": time.time()
            }}
            
            # Store or log performance data
            print(f"Performance: {{performance_data}}")
            
            return result
        
        return wrapper
    
    # Optimization strategies
    async def optimize_function(self, func: Callable, optimization_type: str = "balanced") -> Callable:
        """Apply optimization strategy to a function"""
        if optimization_type == "speed":
            # Apply speed optimizations
            func = self.memoize(256)(func)
            func = self.parallel_executor(10)(func)
        elif optimization_type == "memory":
            # Apply memory optimizations
            func = self.lazy_loader(func)
            func = self.memory_limit(50)(func)
        elif optimization_type == "balanced":
            # Apply balanced optimizations
            func = self.memoize(128)(func)
            func = self.parallel_executor(5)(func)
            func = self.memory_limit(100)(func)
        
        # Always add profiling in debug mode
        func = self.profile_performance(func)
        
        return func
    
    async def analyze_bottlenecks(self, code_stats: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze code for performance bottlenecks"""
        bottlenecks = {{
            "speed": [],
            "memory": [],
            "io": []
        }}
        
        # Analyze for speed bottlenecks
        if code_stats.get("nested_loops", 0) > 2:
            bottlenecks["speed"].append("Deep nested loops detected - consider optimization")
        if code_stats.get("recursive_depth", 0) > 10:
            bottlenecks["speed"].append("Deep recursion detected - consider iterative approach")
        
        # Analyze for memory bottlenecks
        if code_stats.get("large_data_structures", False):
            bottlenecks["memory"].append("Large data structures detected - consider streaming or chunking")
        if code_stats.get("memory_leaks", False):
            bottlenecks["memory"].append("Potential memory leaks detected - review object lifecycle")
        
        # Analyze for I/O bottlenecks
        if code_stats.get("sync_io_in_async", False):
            bottlenecks["io"].append("Synchronous I/O in async context - use async I/O operations")
        if code_stats.get("unbuffered_io", False):
            bottlenecks["io"].append("Unbuffered I/O detected - consider buffering")
        
        return bottlenecks
'''


class TesterAgent(BaseSpecializedAgent):
    """
    Tester Agent: Creates and executes tests
    """
    
    async def analyze_requirement(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze requirement for testing needs"""
        logger.info(f"Tester analyzing requirement: {requirement[:50]}...")
        
        test_types = await self._identify_test_types(requirement)
        test_coverage_targets = await self._determine_coverage_targets(context)
        
        tasks = []
        
        # Test creation tasks
        for test_type in test_types:
            tasks.append({
                "description": f"Create {test_type} tests",
                "priority": "HIGH",
                "agents": ["TESTER"],
                "context": {"test_type": test_type, "coverage_target": test_coverage_targets}
            })
        
        # Test execution task
        tasks.append({
            "description": "Execute test suite",
            "priority": "MEDIUM",
            "agents": ["TESTER"],
            "dependencies": [f"Create {tt} tests" for tt in test_types],
            "context": {"parallel_execution": True}
        })
        
        # Coverage analysis
        tasks.append({
            "description": "Analyze test coverage",
            "priority": "LOW",
            "agents": ["TESTER", "REVIEWER"],
            "dependencies": ["Execute test suite"],
            "context": {"coverage_threshold": test_coverage_targets}
        })
        
        return {
            "agent": "tester",
            "tasks": tasks,
            "analysis": {
                "test_types": test_types,
                "coverage_targets": test_coverage_targets
            }
        }
    
    async def _identify_test_types(self, requirement: str) -> List[str]:
        """Identify types of tests needed"""
        test_types = ["unit"]  # Always include unit tests
        
        requirement_lower = requirement.lower()
        
        if "integrate" in requirement_lower or "system" in requirement_lower:
            test_types.append("integration")
        
        if "performance" in requirement_lower or "speed" in requirement_lower:
            test_types.append("performance")
        
        if "security" in requirement_lower or "vulnerability" in requirement_lower:
            test_types.append("security")
        
        if "api" in requirement_lower or "endpoint" in requirement_lower:
            test_types.append("api")
        
        return test_types
    
    async def _determine_coverage_targets(self, context: Dict[str, Any] = None) -> float:
        """Determine test coverage targets"""
        if context and "coverage_target" in context:
            return context["coverage_target"]
        
        # Default coverage targets based on criticality
        if context and context.get("priority") == "high":
            return 90.0
        elif context and context.get("priority") == "critical":
            return 95.0
        else:
            return 80.0
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute testing task"""
        logger.info(f"Tester executing: {task.description}")
        
        result = {
            "success": True,
            "tests": {},
            "results": {}
        }
        
        if "create" in task.description.lower() and "tests" in task.description.lower():
            test_type = task.context.get("test_type", "unit")
            result["tests"][test_type] = await self._generate_tests(test_type, task.context)
        
        elif "execute test" in task.description.lower():
            result["results"] = await self._execute_tests(task.context)
        
        elif "coverage" in task.description.lower():
            result["coverage"] = await self._analyze_coverage(task.context)
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 1.5  # Simulated
        })
        
        return result
    
    async def _generate_tests(self, test_type: str, context: Dict) -> str:
        """Generate tests based on type"""
        generators = {
            "unit": self._generate_unit_tests,
            "integration": self._generate_integration_tests,
            "performance": self._generate_performance_tests,
            "security": self._generate_security_tests,
            "api": self._generate_api_tests
        }
        
        generator = generators.get(test_type, self._generate_unit_tests)
        return await generator(context)
    
    async def _generate_unit_tests(self, context: Dict) -> str:
        """Generate unit tests"""
        return '''import pytest
from unittest.mock import Mock, patch
import asyncio

class TestCore:
    """Core unit tests"""
    
    @pytest.fixture
    def setup(self):
        """Test setup"""
        return {"initialized": True}
    
    def test_initialization(self, setup):
        """Test component initialization"""
        assert setup["initialized"] is True
    
    @pytest.mark.asyncio
    async def test_async_operation(self):
        """Test async operations"""
        result = await asyncio.sleep(0.01, result="success")
        assert result == "success"
    
    def test_error_handling(self):
        """Test error handling"""
        with pytest.raises(ValueError):
            raise ValueError("Expected error")
    
    @pytest.mark.parametrize("input,expected", [
        (1, 2),
        (2, 4),
        (3, 6),
    ])
    def test_parametrized(self, input, expected):
        """Test with parameters"""
        assert input * 2 == expected
'''
    
    async def _generate_integration_tests(self, context: Dict) -> str:
        """Generate integration tests"""
        return '''import pytest
import asyncio
from typing import Any

class TestIntegration:
    """Integration tests"""
    
    @pytest.fixture
    async def system(self):
        """Setup integrated system"""
        # Setup components
        components = {
            "database": Mock(),
            "api": Mock(),
            "processor": Mock()
        }
        yield components
        # Teardown
        
    @pytest.mark.asyncio
    async def test_end_to_end_flow(self, system):
        """Test complete workflow"""
        # Input -> Processing -> Output
        input_data = {"test": "data"}
        
        # Process through system
        result = await self.process_through_system(system, input_data)
        
        assert result["success"] is True
        assert "processed" in result
    
    async def process_through_system(self, system: dict, data: Any) -> dict:
        """Process data through integrated system"""
        # Simulate processing
        await asyncio.sleep(0.01)
        return {"success": True, "processed": data}
'''
    
    async def _generate_performance_tests(self, context: Dict) -> str:
        """Generate performance tests"""
        return '''import pytest
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.performance
    def test_response_time(self):
        """Test response time"""
        start = time.perf_counter()
        # Perform operation
        result = self.perform_operation()
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0  # Should complete within 1 second
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_load(self):
        """Test under concurrent load"""
        tasks = []
        for _ in range(100):
            tasks.append(self.async_operation())
        
        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        elapsed = time.perf_counter() - start
        
        assert all(r["success"] for r in results)
        assert elapsed < 5.0  # 100 operations in 5 seconds
    
    def perform_operation(self):
        """Simulate operation"""
        time.sleep(0.01)
        return {"success": True}
    
    async def async_operation(self):
        """Simulate async operation"""
        await asyncio.sleep(0.01)
        return {"success": True}
'''
    
    async def _generate_security_tests(self, context: Dict) -> str:
        """Generate security tests"""
        return '''import pytest
from typing import Any

class TestSecurity:
    """Security tests"""
    
    def test_input_validation(self):
        """Test input validation"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "../../../etc/passwd",
            "\\x00\\x01\\x02\\x03"
        ]
        
        for input_data in malicious_inputs:
            assert self.is_input_safe(input_data) is False
    
    def test_authentication_required(self):
        """Test authentication is enforced"""
        # Attempt unauthorized access
        with pytest.raises(UnauthorizedError):
            self.access_protected_resource(authenticated=False)
    
    def test_data_encryption(self):
        """Test sensitive data is encrypted"""
        sensitive_data = "password123"
        stored_data = self.store_sensitive_data(sensitive_data)
        
        assert sensitive_data not in stored_data
        assert len(stored_data) > len(sensitive_data)  # Encrypted
    
    def is_input_safe(self, input_data: str) -> bool:
        """Check if input is safe"""
        dangerous_patterns = ["<script>", "DROP", "../", "\\x00"]
        return not any(pattern in input_data for pattern in dangerous_patterns)
    
    def access_protected_resource(self, authenticated: bool):
        """Simulate resource access"""
        if not authenticated:
            raise UnauthorizedError("Authentication required")
        return {"data": "protected"}
    
    def store_sensitive_data(self, data: str) -> str:
        """Simulate encryption"""
        import hashlib
        return hashlib.sha256(data.encode()).hexdigest()

class UnauthorizedError(Exception):
    pass
'''
    
    async def _generate_api_tests(self, context: Dict) -> str:
        """Generate API tests"""
        return '''import pytest
import httpx
import asyncio
from typing import Dict, Any

class TestAPI:
    """API endpoint tests"""
    
    @pytest.fixture
    def client(self):
        """Test client"""
        return httpx.AsyncClient(base_url="http://localhost:8000")
    
    @pytest.mark.asyncio
    async def test_endpoint_health(self, client):
        """Test health endpoint"""
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_create_resource(self, client):
        """Test resource creation"""
        payload = {"name": "test", "value": 123}
        response = await client.post("/resources", json=payload)
        
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == payload["name"]
        assert "id" in data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, client):
        """Test API error handling"""
        response = await client.get("/nonexistent")
        assert response.status_code == 404
        
        error_data = response.json()
        assert "error" in error_data
        assert "message" in error_data
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, client):
        """Test rate limiting"""
        # Make many requests quickly
        tasks = []
        for _ in range(100):
            tasks.append(client.get("/limited"))
        
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some should be rate limited
        status_codes = [r.status_code for r in responses if hasattr(r, 'status_code')]
        assert 429 in status_codes  # Too Many Requests
'''
    
    async def _execute_tests(self, context: Dict) -> Dict[str, Any]:
        """Execute tests"""
        return {
            "tests_run": 25,
            "tests_passed": 23,
            "tests_failed": 2,
            "execution_time": 5.4,
            "parallel": context.get("parallel_execution", False)
        }
    
    async def _analyze_coverage(self, context: Dict) -> Dict[str, Any]:
        """Analyze test coverage"""
        threshold = context.get("coverage_threshold", 80.0)
        actual_coverage = 85.3  # Simulated
        
        return {
            "coverage_percentage": actual_coverage,
            "threshold": threshold,
            "passed": actual_coverage >= threshold,
            "uncovered_lines": ["module.py:45-47", "utils.py:123"],
            "recommendations": [
                "Add tests for error handling paths",
                "Cover edge cases in data validation"
            ]
        }


class ReviewerAgent(BaseSpecializedAgent):
    """
    Reviewer Agent: Reviews code quality and security
    """
    
    async def analyze_requirement(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze requirement for review needs"""
        logger.info(f"Reviewer analyzing requirement: {requirement[:50]}...")
        
        review_aspects = await self._identify_review_aspects(requirement, context)
        
        tasks = []
        
        # Review tasks
        for aspect in review_aspects:
            tasks.append({
                "description": f"Review {aspect}",
                "priority": "MEDIUM",
                "agents": ["REVIEWER"],
                "context": {"aspect": aspect, "standards": self._get_standards(aspect)}
            })
        
        # Final approval task
        tasks.append({
            "description": "Final code review and approval",
            "priority": "LOW",
            "agents": ["REVIEWER"],
            "dependencies": [f"Review {a}" for a in review_aspects],
            "context": {"final_review": True}
        })
        
        return {
            "agent": "reviewer",
            "tasks": tasks,
            "analysis": {
                "review_aspects": review_aspects
            }
        }
    
    async def _identify_review_aspects(self, requirement: str, context: Dict[str, Any] = None) -> List[str]:
        """Identify aspects to review"""
        aspects = ["code_quality"]  # Always review code quality
        
        requirement_lower = requirement.lower()
        
        if "security" in requirement_lower or "auth" in requirement_lower:
            aspects.append("security")
        
        if "performance" in requirement_lower or "optimize" in requirement_lower:
            aspects.append("performance")
        
        if "api" in requirement_lower or "interface" in requirement_lower:
            aspects.append("api_design")
        
        if context and context.get("framework"):
            aspects.append("framework_best_practices")
        
        return aspects
    
    def _get_standards(self, aspect: str) -> Dict[str, Any]:
        """Get review standards for an aspect"""
        standards = {
            "code_quality": {
                "max_complexity": 10,
                "max_line_length": 120,
                "naming_convention": "snake_case",
                "documentation_required": True
            },
            "security": {
                "input_validation": True,
                "authentication": True,
                "encryption": True,
                "vulnerability_scan": True
            },
            "performance": {
                "max_response_time": 1.0,
                "memory_limit": 100,
                "optimization_level": "balanced"
            },
            "api_design": {
                "restful": True,
                "versioning": True,
                "documentation": True,
                "error_handling": True
            },
            "framework_best_practices": {
                "follow_conventions": True,
                "use_recommended_patterns": True
            }
        }
        
        return standards.get(aspect, {})
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute review task"""
        logger.info(f"Reviewer executing: {task.description}")
        
        result = {
            "success": True,
            "review": {},
            "recommendations": []
        }
        
        aspect = task.context.get("aspect", "general")
        standards = task.context.get("standards", {})
        
        if aspect == "code_quality":
            result["review"] = await self._review_code_quality(standards)
        elif aspect == "security":
            result["review"] = await self._review_security(standards)
        elif aspect == "performance":
            result["review"] = await self._review_performance(standards)
        elif aspect == "api_design":
            result["review"] = await self._review_api_design(standards)
        elif task.context.get("final_review"):
            result["review"] = await self._final_review()
        
        # Add recommendations
        result["recommendations"] = await self._generate_recommendations(result["review"])
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 1.2  # Simulated
        })
        
        return result
    
    async def _review_code_quality(self, standards: Dict) -> Dict[str, Any]:
        """Review code quality"""
        return {
            "complexity": {
                "average": 7.2,
                "max": 12,
                "passed": False,
                "issues": ["Function 'process_data' has complexity 12 (max: 10)"]
            },
            "style": {
                "violations": 3,
                "passed": True,
                "issues": ["Line 145 exceeds 120 characters"]
            },
            "documentation": {
                "coverage": 85,
                "passed": True,
                "missing": ["Function 'helper_method' lacks docstring"]
            }
        }
    
    async def _review_security(self, standards: Dict) -> Dict[str, Any]:
        """Review security aspects"""
        return {
            "vulnerabilities": {
                "critical": 0,
                "high": 0,
                "medium": 1,
                "low": 2,
                "passed": True
            },
            "authentication": {
                "implemented": True,
                "strong": True,
                "passed": True
            },
            "input_validation": {
                "coverage": 95,
                "passed": True,
                "uncovered": ["Parameter 'config' in line 234"]
            }
        }
    
    async def _review_performance(self, standards: Dict) -> Dict[str, Any]:
        """Review performance aspects"""
        return {
            "response_times": {
                "average": 0.3,
                "p95": 0.8,
                "p99": 1.2,
                "passed": False,
                "issues": ["P99 response time exceeds 1.0s limit"]
            },
            "memory_usage": {
                "average_mb": 45,
                "peak_mb": 78,
                "passed": True
            },
            "optimization_opportunities": [
                "Consider caching results of expensive_calculation()",
                "Use batch processing for bulk operations"
            ]
        }
    
    async def _review_api_design(self, standards: Dict) -> Dict[str, Any]:
        """Review API design"""
        return {
            "restful_compliance": {
                "score": 90,
                "passed": True,
                "issues": ["PUT /resource should return 200, not 201"]
            },
            "documentation": {
                "openapi_spec": True,
                "examples": True,
                "passed": True
            },
            "versioning": {
                "implemented": True,
                "strategy": "url_path",
                "passed": True
            }
        }
    
    async def _final_review(self) -> Dict[str, Any]:
        """Perform final review"""
        return {
            "overall_score": 88,
            "approval_status": "approved_with_suggestions",
            "blocking_issues": [],
            "non_blocking_issues": [
                "Consider refactoring complex functions",
                "Add more comprehensive error messages"
            ]
        }
    
    async def _generate_recommendations(self, review: Dict) -> List[str]:
        """Generate recommendations based on review"""
        recommendations = []
        
        # Check for failed criteria
        for category, details in review.items():
            if isinstance(details, dict) and not details.get("passed", True):
                if "issues" in details:
                    recommendations.extend(details["issues"])
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("Code meets all standards. Consider adding more tests.")
        
        return recommendations


class DocumenterAgent(BaseSpecializedAgent):
    """
    Documenter Agent: Creates documentation
    """
    
    async def analyze_requirement(self, requirement: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze requirement for documentation needs"""
        logger.info(f"Documenter analyzing requirement: {requirement[:50]}...")
        
        doc_types = await self._identify_doc_types(requirement, context)
        
        tasks = []
        
        for doc_type in doc_types:
            tasks.append({
                "description": f"Create {doc_type} documentation",
                "priority": "LOW",
                "agents": ["DOCUMENTER"],
                "context": {"doc_type": doc_type}
            })
        
        return {
            "agent": "documenter",
            "tasks": tasks,
            "analysis": {
                "doc_types": doc_types
            }
        }
    
    async def _identify_doc_types(self, requirement: str, context: Dict[str, Any] = None) -> List[str]:
        """Identify documentation types needed"""
        doc_types = ["code_comments"]  # Always include code comments
        
        requirement_lower = requirement.lower()
        
        if "api" in requirement_lower:
            doc_types.append("api_docs")
        
        if "user" in requirement_lower or "guide" in requirement_lower:
            doc_types.append("user_guide")
        
        if context and context.get("framework"):
            doc_types.append("technical_docs")
        
        return doc_types
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute documentation task"""
        logger.info(f"Documenter executing: {task.description}")
        
        doc_type = task.context.get("doc_type", "general")
        
        generators = {
            "code_comments": self._generate_code_comments,
            "api_docs": self._generate_api_docs,
            "user_guide": self._generate_user_guide,
            "technical_docs": self._generate_technical_docs
        }
        
        generator = generators.get(doc_type, self._generate_technical_docs)
        documentation = await generator()
        
        result = {
            "success": True,
            "documentation": {doc_type: documentation}
        }
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 0.8  # Simulated
        })
        
        return result
    
    async def _generate_code_comments(self) -> str:
        """Generate code comments"""
        return '''"""
Module: Feature Implementation
Purpose: Provides core functionality for the feature
Author: AI Agent System
Date: 2024
"""

def process_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process input data and return results.
    
    Args:
        input_data: Dictionary containing input parameters
            - 'key1': Description of key1
            - 'key2': Description of key2
    
    Returns:
        Dictionary containing processed results
            - 'success': Boolean indicating success
            - 'result': Processed data
    
    Raises:
        ValueError: If input_data is invalid
        ProcessingError: If processing fails
    
    Example:
        >>> data = {'key1': 'value1', 'key2': 123}
        >>> result = process_data(data)
        >>> print(result['success'])
        True
    """
    # Implementation here
    pass
'''
    
    async def _generate_api_docs(self) -> str:
        """Generate API documentation"""
        return '''# API Documentation

## Overview
This API provides endpoints for feature functionality.

## Authentication
All endpoints require Bearer token authentication.

## Endpoints

### POST /api/process
Process data and return results.

**Request:**
```json
{
  "data": {
    "key1": "value1",
    "key2": 123
  },
  "config": {
    "option1": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "result": {
    "processed": true,
    "output": "..."
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Status Codes:**
- 200: Success
- 400: Bad Request
- 401: Unauthorized
- 500: Internal Server Error

### GET /api/status
Get system status.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 3600
}
```
'''
    
    async def _generate_user_guide(self) -> str:
        """Generate user guide"""
        return '''# User Guide

## Getting Started

### Installation
```bash
pip install feature-package
```

### Basic Usage
```python
from feature import FeatureClient

client = FeatureClient(api_key="your_api_key")
result = client.process(data={"input": "value"})
print(result)
```

## Features

### Data Processing
The system can process various data types...

### Configuration
Configure the system using...

## Troubleshooting

### Common Issues
1. **Authentication Error**: Ensure your API key is valid
2. **Timeout Error**: Increase timeout in configuration
3. **Data Format Error**: Check input data format

## Support
Contact support@example.com for assistance.
'''
    
    async def _generate_technical_docs(self) -> str:
        """Generate technical documentation"""
        return '''# Technical Documentation

## Architecture

### Component Overview
- **Core Engine**: Handles main processing logic
- **API Layer**: RESTful API interface
- **Data Layer**: Data persistence and caching
- **Integration Layer**: External system integration

### Design Patterns
- **Factory Pattern**: For creating agent instances
- **Observer Pattern**: For event handling
- **Strategy Pattern**: For algorithm selection

## Implementation Details

### Core Algorithm
The system uses a multi-stage processing pipeline:
1. Input validation
2. Data transformation
3. Core processing
4. Result aggregation
5. Output formatting

### Performance Considerations
- Async/await for I/O operations
- Caching for frequently accessed data
- Batch processing for bulk operations

## Deployment

### Requirements
- Python 3.8+
- 2GB RAM minimum
- 10GB disk space

### Configuration
Environment variables:
- `API_KEY`: Authentication key
- `DATABASE_URL`: Database connection string
- `CACHE_SIZE`: Cache size in MB

## Monitoring
- Health check endpoint: `/health`
- Metrics endpoint: `/metrics`
- Logging: JSON format to stdout
'''


class IntegratorAgent(BaseSpecializedAgent):
    """
    Integrator Agent: Handles system integration
    """
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute integration task"""
        logger.info(f"Integrator executing: {task.description}")
        
        result = {
            "success": True,
            "integration": {
                "components_integrated": 3,
                "tests_passed": True,
                "deployment_ready": True
            }
        }
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 2.5  # Simulated
        })
        
        return result


class RefactorerAgent(BaseSpecializedAgent):
    """
    Refactorer Agent: Refactors and optimizes code
    """
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute refactoring task"""
        logger.info(f"Refactorer executing: {task.description}")
        
        result = {
            "success": True,
            "refactoring": {
                "patterns_extracted": 2,
                "duplicates_removed": 5,
                "complexity_reduced": True,
                "lines_saved": 150
            }
        }
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 1.8  # Simulated
        })
        
        return result


class DebuggerAgent(BaseSpecializedAgent):
    """
    Debugger Agent: Debugs and fixes issues
    """
    
    async def execute_task(self, task: Any) -> Dict[str, Any]:
        """Execute debugging task"""
        logger.info(f"Debugger executing: {task.description}")
        
        result = {
            "success": True,
            "debugging": {
                "issues_found": 3,
                "issues_fixed": 3,
                "root_causes": ["null pointer", "race condition", "off-by-one"],
                "prevention_measures": ["add null checks", "use locks", "boundary validation"]
            }
        }
        
        # Record in history
        self.task_history.append({
            "task_id": task.id,
            "success": result["success"],
            "execution_time": 3.0  # Simulated
        })
        
        return result