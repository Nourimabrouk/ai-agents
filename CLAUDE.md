# CLAUDE.md

# AI Agents Development Environment

## Environment Context
**Operating System**: Windows 11  
**IDE**: Cursor (VS Code fork)  
**Primary Tool**: Claude Code  
**Development Focus**: Multi-framework AI agent development and coordination  
**Execution Preference**: Parallel processing and async workflows

## Core Principles
1. **Windows-First**: All commands and paths should be Windows-compatible
2. **Practical Development**: Focus on working code over theoretical concepts
3. **Agent Coordination**: Design for multi-agent and sub-agent collaboration
4. **Parallel Execution**: Prefer async/await and concurrent processing patterns
5. **Clean Code**: No emojis in code files (terminal compatibility issues on Windows)

## Testing & CI (Experimental-Friendly)
- Python: `pytest -q` (async via `pytest-asyncio`); see `tests/python/` for contract and orchestration smoke tests.
- Node: `npm test` (Jest) for JS utilities/MCP stubs; tests in `agents/__tests__/`.
- Keep tests fast and non-flaky; prefer interface/contract checks over brittle E2E.
- CI suggestion: run lint + unit/smoke tests in parallel Python/Node jobs; do not hard-fail on coverage early on.

### Command Standards
```bash
# Use Windows paths
C:\Users\Nouri\Documents\GitHub\ai-agents\

# Use Windows command syntax
dir  # not ls
cd /d  # for drive changes
python -m venv env  # virtual environments
env\Scripts\activate  # not source activate

# Use PowerShell or cmd syntax when needed
Get-ChildItem  # PowerShell equivalent of ls
```

## Repository Structure

### Directory Layout
```
ai-agents/                  # Root: AI agent development repository
├── agents/                 # Agent implementations by framework
│   ├── claude-code/       # Claude Code & MCP agents
│   ├── microsoft/         # Azure AI agents
│   ├── langchain/         # LangChain workflows
│   └── accountancy/       # Domain-specific agents
├── frameworks/             # Templates and examples
├── projects/              # Complete implementations
├── assets/                # Generated content
├── planning/              # Project roadmaps
├── utils/                 # Shared utilities
└── docs/                  # Documentation
```

### Development Workflow
- **Starting Point**: `agents/claude-code/` for MCP development
- **Iteration Process**: Build → Test → Document → Improve
- **Asset Storage**: Use `assets/` for generated files
- **Code Reuse**: Check `utils/` for existing functionality
- **Pattern Library**: Document reusable patterns

## Agent Development Standards

### Base Agent Architecture
```python
# Standard agent structure for this repository
class BaseAgent:
    """
    Base agent with coordination capabilities
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = self._init_memory()
        self.tools = self._init_tools()
        self.reflection_engine = self._init_reflection()
        self.learning_system = self._init_learning()
    
    async def think(self, input_data: Any, context: Dict = None) -> Thought:
        """Analyze task requirements"""
        return await self.reflection_engine.analyze(
            task=input_data,
            context=context,
            past_experiences=self.memory.recall_similar()
        )
    
    async def act(self, thought: Thought) -> Action:
        """Execute action based on reasoning"""
        selected_tools = await self._select_tools(thought.strategy)
        return await self._execute_with_tools(thought, selected_tools)
    
    async def observe(self, action: Action, result: Any) -> Observation:
        """Observe and learn from results"""
        observation = Observation(action, result, self._evaluate_success(result))
        await self.learning_system.update(observation)
        await self.memory.store(observation)
        return observation
    
    async def improve(self) -> None:
        """Performance optimization based on accumulated experience"""
        insights = await self.learning_system.extract_patterns()
        await self._update_strategies(insights)
        await self._optimize_tool_selection(insights)
```

### Multi-Agent Coordination Patterns
```python
# Multi-agent coordination patterns
class AgentOrchestrator:
    """Coordinates multiple specialized agents"""
    
    async def hierarchical_delegation(self, task: Task) -> Result:
        """Top-down task decomposition and delegation"""
        subtasks = await self.decompose_task(task)
        results = await asyncio.gather(*[
            self.delegate_to_specialist(subtask) 
            for subtask in subtasks
        ])
        return await self.synthesize_results(results)
    
    async def peer_collaboration(self, agents: List[Agent], task: Task) -> Result:
        """Collaborative problem-solving between equals"""
        conversation = await self.init_group_chat(agents, task)
        while not conversation.is_resolved():
            next_speaker = await self.select_next_speaker(conversation)
            response = await next_speaker.contribute(conversation)
            conversation.add_message(response)
        return conversation.final_solution()
    
    async def coordinate_agents(self, agents: AgentGroup, objective: Objective) -> Solution:
        """Coordinate multiple agents for task completion"""
        for iteration in range(self.max_iterations):
            await swarm.share_local_information()
            await swarm.update_individual_strategies()
            if await swarm.check_convergence():
                break
        return await swarm.extract_solution()

# Specialized tool-using agent
class ToolUseAgent(BaseAgent):
    """Agent for tool selection and usage"""
    
    async def plan_tool_sequence(self, goal: Goal) -> ToolPlan:
        """Plan optimal sequence of tool usage"""
        available_tools = await self.tools.get_available()
        return await self.reasoning_engine.plan_sequence(goal, available_tools)
    
    async def execute_with_adaptation(self, plan: ToolPlan) -> Result:
        """Execute plan with real-time adaptation"""
        for step in plan.steps:
            result = await self.execute_tool_step(step)
            if not result.success:
                adapted_plan = await self.adapt_plan(plan, step, result.error)
                return await self.execute_with_adaptation(adapted_plan)
        return result

# Agent with memory capabilities
class MemoryAgent(BaseAgent):
    """Manages long-term memory and pattern extraction"""
    
    async def episodic_memory(self, experience: Experience) -> None:
        """Store detailed episode with rich context"""
        await self.memory.store_episode({
            'experience': experience,
            'context': await self.extract_context(experience),
            'embeddings': await self.embed_experience(experience),
            'timestamp': datetime.now(),
            'success_metrics': await self.evaluate_success(experience)
        })
    
    async def semantic_retrieval(self, query: str, k: int = 5) -> List[Memory]:
        """Retrieve relevant memories using semantic similarity"""
        query_embedding = await self.embed_query(query)
        similar_memories = await self.memory.vector_search(query_embedding, k)
        return await self.rank_by_relevance(similar_memories, query)
```

### Prompt Engineering Standards
```python
# Standard prompt template for agents
AGENT_PROMPT_TEMPLATE = """
<instruction>
You are working in a Windows development environment with:
- Cursor IDE for code editing
- Claude Code for AI assistance
- Multi-agent coordination capabilities
- Parallel execution preferences
- Focus on practical, working solutions
</instruction>

<cognitive_approach>
1. **System 2 Reasoning**: Engage deliberate, analytical thinking
2. **Pattern Recognition**: Identify relevant patterns from experience
3. **Tool Orchestration**: Dynamically compose tool sequences
4. **Uncertainty Quantification**: Express confidence in decisions
5. **Meta-Learning**: Reflect on reasoning process for improvement
</cognitive_approach>

<task_context>
Primary Objective: {primary_objective}
Available Tools: {available_tools}
Memory Context: {relevant_memories}
Collaboration Mode: {multi_agent_context}
Learning Phase: {exploration_vs_exploitation}
</task_context>

<reasoning_framework>
Before acting, explicitly:
1. Analyze the problem space and identify key constraints
2. Consider multiple solution approaches and their trade-offs
3. Plan tool usage sequence with fallback strategies
4. Anticipate potential failure modes and recovery strategies
5. Identify opportunities for learning and pattern extraction
</reasoning_framework>

<output_specification>
Structure your response as:
1. **Reasoning**: Your step-by-step analysis
2. **Strategy**: Chosen approach with justification
3. **Actions**: Specific actions to take with tools
4. **Adaptations**: How you'll adapt based on results
5. **Learning**: What patterns this might reveal
</output_specification>
"""
```

## Domain Specialization

### Domain Agent Architecture
```python
# Domain specialization pattern
class DomainSpecialist(BaseAgent):
    """Base class for domain-specific agents"""
    
    def __init__(self, domain_knowledge: DomainKB, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_kb = domain_knowledge
        self.domain_tools = self._init_domain_tools()
        self.domain_memory = self._init_domain_memory()
    
    async def domain_reasoning(self, problem: Problem) -> DomainSolution:
        """Apply domain-specific reasoning patterns"""
        domain_context = await self.domain_kb.get_context(problem)
        reasoning_patterns = await self.domain_kb.get_patterns(problem.type)
        
        return await self.apply_domain_logic(
            problem, domain_context, reasoning_patterns
        )
    
    async def cross_domain_transfer(self, insight: Insight, target_domain: Domain) -> TransferResult:
        """Transfer insights across domains"""
        abstracted_pattern = await self.abstract_pattern(insight)
        return await self.adapt_to_domain(abstracted_pattern, target_domain)

# Example: Accounting domain
class AccountancyAgent(DomainSpecialist):
    """Specialized for financial data processing and analysis"""
    
    async def process_financial_document(self, document: Document) -> FinancialData:
        """Extract and validate financial data"""
        extracted = await self.ocr_engine.extract(document)
        validated = await self.validate_against_rules(extracted)
        categorized = await self.categorize_transactions(validated)
        return await self.create_journal_entries(categorized)
    
    async def detect_anomalies(self, transactions: List[Transaction]) -> List[Anomaly]:
        """Identify unusual patterns in financial data"""
        patterns = await self.domain_memory.get_normal_patterns()
        anomalies = []
        for transaction in transactions:
            if await self.is_anomalous(transaction, patterns):
                anomaly = await self.analyze_anomaly(transaction)
                anomalies.append(anomaly)
        return anomalies
```

### Universal Integration Patterns
```python
# System integration helper
class SystemConnector(BaseAgent):
    """Connector for external systems"""
    
    async def discover_schema(self, system_endpoint: str) -> Schema:
        """Discover system schemas"""
        raw_schema = await self.introspect_api(system_endpoint)
        return await self.normalize_schema(raw_schema)
    
    async def adaptive_authentication(self, system: System) -> AuthToken:
        """Handle multiple authentication patterns"""
        auth_patterns = await self.detect_auth_pattern(system)
        return await self.execute_auth_flow(auth_patterns)
    
    async def map_data(self, source: Schema, target: Schema) -> Mapping:
        """Map data between schemas"""
        semantic_similarity = await self.compute_semantic_similarity(source, target)
        return await self.generate_mapping_rules(semantic_similarity)
```

## Agent Capabilities

### Adaptive Systems
```python
class AdaptiveAgentSystem:
    """Agent system with improvement capabilities"""
    
    async def optimize_agents(self, population: List[Agent]) -> List[Agent]:
        """Optimize agent performance through iteration"""
        for generation in range(self.max_generations):
            # Evaluate fitness based on performance metrics
            fitness_scores = await self.evaluate_population(population)
            
            # Selection: Choose best performers
            parents = await self.selection(population, fitness_scores)
            
            # Crossover: Combine successful strategies
            offspring = await self.crossover(parents)
            
            # Mutation: Introduce random variations
            mutated = await self.mutate(offspring)
            
            population = await self.create_new_generation(parents, mutated)
            
        return population
    
    async def detect_patterns(self, agent_network: AgentNetwork) -> List[Pattern]:
        """Identify behavioral patterns"""
        interactions = await agent_network.get_interaction_history()
        patterns = await self.pattern_mining.find_emergent_patterns(interactions)
        
        novel_patterns = []
        for pattern in patterns:
            if await self.is_novel_behavior(pattern):
                emergent = await self.analyze_emergence(pattern)
                await self.incorporate_emergent_behavior(emergent)
                novel_patterns.append(emergent)
        
        return novel_patterns
    
    async def improve_learning(self, experiences: List[Experience]) -> Strategy:
        """Improve learning strategies based on experience"""
        learning_patterns = await self.extract_learning_patterns(learning_experiences)
        meta_strategies = await self.generate_meta_strategies(learning_patterns)
        optimal_strategy = await self.optimize_meta_strategy(meta_strategies)
        
        return optimal_strategy

# Learning system for agents
class LearningEngine:
    """Learning system for agent improvement"""
    
    def __init__(self):
        self.pattern_recognizer = PatternRecognizer()
        self.strategy_optimizer = StrategyOptimizer()
        self.performance_tracker = PerformanceTracker()
        self.knowledge_base = KnowledgeBase()
    
    async def learn_from_experience(self, experience: Any) -> LearningOutcome:
        """Extract insights and improve performance"""
        # Pattern extraction
        patterns = await self.pattern_recognizer.extract(experience)
        
        # Strategy optimization
        optimized_strategies = await self.strategy_optimizer.improve(patterns)
        
        # Knowledge integration
        await self.knowledge_base.integrate(patterns, optimized_strategies)
        
        # Performance evaluation
        performance_metrics = await self.performance_tracker.evaluate(experience)
        
        return LearningOutcome(
            patterns=patterns,
            strategies=optimized_strategies,
            metrics=performance_metrics
        )
```

### Advanced Reasoning Architectures
```python
# Hybrid reasoning approach
class HybridAgent(BaseAgent):
    """Agent using multiple reasoning approaches"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.neural_component = NeuralReasoningModule()
        self.symbolic_component = SymbolicReasoningEngine()
        self.integration_layer = NeuroSymbolicIntegrator()
    
    async def hybrid_reasoning(self, problem: Problem) -> Solution:
        """Combine neural intuition with symbolic logic"""
        # Neural processing for pattern recognition and intuition
        neural_insights = await self.neural_component.process(problem)
        
        # Symbolic processing for logical reasoning
        symbolic_solution = await self.symbolic_component.reason(problem)
        
        # Integration and validation
        integrated_solution = await self.integration_layer.synthesize(
            neural_insights, symbolic_solution
        )
        
        return integrated_solution
    
    async def causal_reasoning(self, observations: List[Observation]) -> CausalModel:
        """Infer causal relationships from data"""
        causal_graph = await self.symbolic_component.build_causal_graph(observations)
        causal_strengths = await self.neural_component.estimate_causal_strengths(
            observations, causal_graph
        )
        
        return CausalModel(graph=causal_graph, strengths=causal_strengths)

# Optimization agent
class OptimizationAgent(BaseAgent):
    """Agent for task optimization"""
    
    async def parallel_search(self, search_space: SearchSpace) -> Solution:
        """Explore multiple solution paths in parallel"""
        parallel_paths = await self.create_parallel_paths(search_space)
        
        # Parallel exploration with early stopping
        results = await asyncio.gather(*[
            self.explore_path(path) for path in parallel_paths
        ])
        
        # Select best solution
        best_solution = await self.select_optimal(results)
        
        return best_solution
    
    async def swarm_optimization(self, agents: List[Agent], task: Task) -> CoordinatedSolution:
        """Coordinate agents using swarm intelligence"""
        swarm = await self.initialize_swarm(agents, task)
        
        # Iterative optimization
        while not swarm.is_converged():
            await swarm.share_information()
            await swarm.update_positions()
            await swarm.evaluate_fitness()
        
        return swarm.get_best_solution()
```

## Performance Metrics

### Agent Performance Indicators
```python
INTELLIGENCE_METRICS = {
    "reasoning_capability": {
        "problem_decomposition": "measure task breakdown",
        "solution_quality": "track implementation effectiveness",
        "knowledge_transfer": "measure reusability",
        "learning_rate": "improvement over time"
    },
    "collaboration_effectiveness": {
        "agent_coordination": "measure multi-agent efficiency",
        "knowledge_sharing": "track information exchange",
        "conflict_resolution": "handle conflicting decisions",
        "pattern_detection": "identify useful behaviors"
    },
    "adaptation_metrics": {
        "environment_adaptation": "measure flexibility",
        "error_recovery": "handle and learn from failures",
        "strategy_improvement": "iterative refinement",
        "uncertainty_management": "handle incomplete information"
    },
    "exploration_metrics": {
        "exploration_efficiency": "systematic testing approach",
        "pattern_identification": "recognize useful structures",
        "hypothesis_testing": "validate assumptions",
        "experiment_design": "structured testing methodology"
    }
}
```

### Continuous Improvement Process
1. **Real-time**: Performance monitoring and adjustment during execution
2. **Per-task**: Extract patterns and optimize strategies
3. **Daily**: Aggregate learning from task completions
4. **Weekly**: Performance review and optimization
5. **Monthly**: Architecture improvements and capability expansion

## Execution Commands

### Development Commands
```bash
# OPTIMIZE: Run performance optimization on agent systems
python optimize/agent_optimizer.py --mode=performance --target=throughput

# COORDINATE: Test multi-agent coordination patterns
python coordinate/multi_agent_test.py --agents=10 --pattern=hierarchical

# ANALYZE: Performance analysis and bottleneck detection
python analyze/performance_analyzer.py --metrics=all --report=detailed

# INTEGRATE: Test system integrations
python integrate/system_connector.py --target=accounting_system --mode=test

# BENCHMARK: Run comprehensive benchmarks
python benchmark/agent_benchmark.py --suite=complete --compare=baseline
```

### Agent Development Pipeline
```bash
# 1. DESIGN: Create agent architectures for specific use cases
python design/agent_designer.py --use_case=accounting --requirements=basic

# 2. BUILD: Implement agent with proper testing
python build/agent_builder.py --template=base_agent --domain=financial

# 3. TEST: Comprehensive testing suite
python test/agent_tester.py --suite=integration --coverage=full

# 4. DEPLOY: Production deployment pipeline
python deploy/agent_deployer.py --environment=staging --monitoring=enabled

# 5. MONITOR: Production monitoring and optimization
python monitor/agent_monitor.py --metrics=all --alerts=critical
```

## Experimental Features

### AI Techniques
- **Chain-of-Thought Reasoning**: For complex accounting decisions
- **Tool Use Optimization**: Dynamic tool selection and chaining
- **Multi-Modal Processing**: Handle PDFs, images, and structured data
- **Retrieval-Augmented Generation**: Leverage accounting knowledge bases
- **Fine-Tuning**: Custom models for domain-specific tasks

### Emerging Capabilities
- **Workflow Automation**: Automated task workflows
- **Error Recovery**: Automatic error handling
- **Analytics**: Data analysis and forecasting
- **Natural Language Queries**: "Show me all invoices over $10k from last quarter"

## Learning Path

### Development Phases
1. **Foundation**: Learn Claude Code and basic automation
2. **Expansion**: Add additional framework capabilities  
3. **Specialization**: Focus on specific domain expertise
4. **Refinement**: Optimize and improve existing solutions

### Project Portfolio
- **Working Projects**: Multiple functional automation systems
- **Technical Variety**: Different framework implementations
- **Practical Impact**: Measurable efficiency improvements
- **Documentation**: Clear project documentation and examples

## Key Requirements

### Core Requirements
- **Practical Value**: Features should solve real problems
- **Code Quality**: Implement proper error handling and testing
- **Security**: Handle sensitive data appropriately
- **Maintenance**: Keep code updated and documented
- **Usability**: Design for target user expertise level

### Best Practices
- **Start Simple**: Build incrementally
- **Security First**: Prioritize data protection  
- **Code Maintenance**: Regular refactoring
- **Stay Focused**: Complete features before expanding scope
- **Collaboration**: Engage with relevant communities

---

---

---

## Instructions for Claude Code

**Operating Environment**: Windows 11 with Cursor IDE

### Core Guidelines
1. **Windows Compatibility**: Use Windows-style paths and commands (backslashes, `dir`, etc.)
2. **No Emojis in Code**: Keep code files clean - emojis cause terminal display issues
3. **Parallel Execution**: Prefer async/await patterns and concurrent processing
4. **Agent Coordination**: Design for multi-agent collaboration from the start
5. **Practical Focus**: Build working solutions over theoretical frameworks

### Development Patterns
- **File Paths**: Use `C:\Users\Nouri\Documents\GitHub\ai-agents\` format
- **Virtual Environments**: `python -m venv env` and `env\Scripts\activate`
- **Testing**: `python -m pytest` for test execution
- **Package Management**: `pip install -r requirements.txt`

### Agent Development
- Use the `templates/base_agent.py` as starting point
- Implement proper error handling and logging
- Support both individual and coordinated execution
- Include performance metrics and monitoring
- Design for extensibility and reusability

### Multi-Agent Coordination
- Use `orchestrator.py` for coordinating multiple agents
- Implement message passing between agents
- Support parallel task execution
- Include consensus and collaboration mechanisms

### Code Quality
- Type hints for all functions
- Comprehensive docstrings
- Unit tests for core functionality
- Clean separation of concerns
- Proper async/await usage

**Remember**: Focus on building functional AI agent systems for experimentation in a Windows development environment.
