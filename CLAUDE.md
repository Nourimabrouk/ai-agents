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

## 📁 REPOSITORY ARCHITECTURE

### Semantic Directory Structure
```
ai-agents/                          # Root: Multi-framework agentic AI ecosystem
├── agents/                         # CORE: Agent implementations by framework
│   ├── claude-code/               # PRIMARY: Claude Code & MCP agents (START HERE)
│   ├── microsoft/                 # SECONDARY: Azure AI, Copilot Studio agents
│   ├── langchain/                 # TERTIARY: LangChain/LangGraph workflows
│   └── accountancy/               # DOMAIN: Specialized domain agents
├── frameworks/                     # LEARNING: Templates, examples, tutorials
├── projects/                      # INTEGRATION: End-to-end implementations
├── assets/                        # GENERATED: Auto-created content
├── planning/                      # STRATEGY: Learning and exploration roadmaps
├── utils/                         # SHARED: Cross-cutting utilities
└── docs/                          # KNOWLEDGE: Documentation and guides
```

### Navigation Optimization
- **Exploration Path**: `agents/claude-code/` → `frameworks/` → `projects/`
- **Learning Loop**: Experiment → Document → Refine → Scale
- **Asset Generation**: Always create in `assets/` with timestamp and metadata
- **Utility Reuse**: Check `utils/` before implementing shared functionality
- **Pattern Discovery**: Continuously identify and abstract successful patterns

## Agent Development Standards

### Base Agent Architecture
```python
# Standard agent structure for this repository
class AgenticAgent:
    """
    Base agent with learning and coordination capabilities
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = self._init_memory()
        self.tools = self._init_tools()
        self.reflection_engine = self._init_reflection()
        self.learning_system = self._init_learning()
    
    async def think(self, input_data: Any, context: Dict = None) -> Thought:
        """Meta-cognitive reasoning about the task"""
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
    
    async def evolve(self) -> None:
        """Self-improvement based on accumulated experience"""
        insights = await self.learning_system.extract_patterns()
        await self._update_strategies(insights)
        await self._optimize_tool_selection(insights)
```

### Multi-Agent Coordination Patterns
```python
# ADVANCED: Multi-agent orchestration patterns
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
    
    async def swarm_intelligence(self, swarm: AgentSwarm, objective: Objective) -> Solution:
        """Emergent intelligence from simple agent interactions"""
        for iteration in range(self.max_iterations):
            await swarm.share_local_information()
            await swarm.update_individual_strategies()
            if await swarm.check_convergence():
                break
        return await swarm.extract_solution()

# SUB-AGENT: Specialized micro-agents for specific capabilities
class ToolUseAgent(AgenticAgent):
    """Specialized in dynamic tool selection and chaining"""
    
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

# MEMORY AGENT: Persistent learning and pattern recognition
class MemoryAgent(AgenticAgent):
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

## 🧬 DOMAIN SPECIALIZATION FRAMEWORK

### Domain Agent Architecture
```python
# EXTENSIBLE: Domain-agnostic specialization pattern
class DomainSpecialist(AgenticAgent):
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

# ACCOUNTANCY: Example domain specialization
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
# FLEXIBLE: System-agnostic integration framework
class SystemConnector(AgenticAgent):
    """Universal connector for external systems"""
    
    async def auto_discover_schema(self, system_endpoint: str) -> Schema:
        """Automatically discover and map system schemas"""
        raw_schema = await self.introspect_api(system_endpoint)
        return await self.normalize_schema(raw_schema)
    
    async def adaptive_authentication(self, system: System) -> AuthToken:
        """Handle multiple authentication patterns"""
        auth_patterns = await self.detect_auth_pattern(system)
        return await self.execute_auth_flow(auth_patterns)
    
    async def intelligent_data_mapping(self, source: Schema, target: Schema) -> Mapping:
        """AI-powered schema mapping and transformation"""
        semantic_similarity = await self.compute_semantic_similarity(source, target)
        return await self.generate_mapping_rules(semantic_similarity)
```

## 🚀 ADVANCED AGENTIC CAPABILITIES

### Self-Evolving Systems
```python
class EvolutionaryAgentSystem:
    """Self-modifying and improving agent ecosystem"""
    
    async def genetic_algorithm_optimization(self, population: List[Agent]) -> List[Agent]:
        """Evolve agent capabilities through genetic algorithms"""
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
    
    async def emergent_behavior_detection(self, agent_network: AgentNetwork) -> List[EmergentPattern]:
        """Identify unexpected emergent behaviors"""
        interactions = await agent_network.get_interaction_history()
        patterns = await self.pattern_mining.find_emergent_patterns(interactions)
        
        novel_patterns = []
        for pattern in patterns:
            if await self.is_novel_behavior(pattern):
                emergent = await self.analyze_emergence(pattern)
                await self.incorporate_emergent_behavior(emergent)
                novel_patterns.append(emergent)
        
        return novel_patterns
    
    async def meta_meta_learning(self, learning_experiences: List[LearningExperience]) -> MetaStrategy:
        """Learn how to learn better - meta^2 learning"""
        learning_patterns = await self.extract_learning_patterns(learning_experiences)
        meta_strategies = await self.generate_meta_strategies(learning_patterns)
        optimal_strategy = await self.optimize_meta_strategy(meta_strategies)
        
        return optimal_strategy

# CONSCIOUSNESS MODELING: Experimental consciousness-like behavior
class ConsciousnessEngine:
    """Experimental consciousness simulation for agents"""
    
    def __init__(self):
        self.attention_mechanism = AttentionController()
        self.working_memory = WorkingMemory(capacity=7)  # Miller's law
        self.global_workspace = GlobalWorkspace()
        self.self_model = SelfModel()
    
    async def conscious_experience(self, stimulus: Any) -> ConsciousResponse:
        """Simulate conscious processing"""
        # Attention filtering
        attended_stimulus = await self.attention_mechanism.focus(stimulus)
        
        # Working memory integration
        current_context = await self.working_memory.integrate(attended_stimulus)
        
        # Global workspace broadcasting
        conscious_content = await self.global_workspace.broadcast(current_context)
        
        # Self-reflection
        self_aware_response = await self.self_model.reflect(conscious_content)
        
        return ConsciousResponse(
            content=conscious_content,
            self_awareness=self_aware_response,
            confidence=await self.compute_confidence(conscious_content)
        )
```

### Advanced Reasoning Architectures
```python
# NEURO-SYMBOLIC: Hybrid symbolic and neural reasoning
class NeuroSymbolicAgent(AgenticAgent):
    """Combines neural networks with symbolic reasoning"""
    
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

# QUANTUM-INSPIRED: Quantum computing-inspired algorithms
class QuantumInspiredAgent(AgenticAgent):
    """Uses quantum-inspired algorithms for optimization"""
    
    async def quantum_superposition_search(self, search_space: SearchSpace) -> Solution:
        """Explore multiple solution paths simultaneously"""
        superposition_state = await self.create_superposition(search_space)
        
        # Quantum-inspired parallel exploration
        parallel_explorations = await self.explore_superposition(superposition_state)
        
        # Measurement/collapse to best solution
        measured_solution = await self.quantum_measurement(parallel_explorations)
        
        return measured_solution
    
    async def entanglement_coordination(self, agents: List[Agent], task: Task) -> CoordinatedSolution:
        """Coordinate agents using entanglement-inspired coupling"""
        entangled_system = await self.create_entanglement(agents, task)
        
        # Coordinated parallel processing
        while not entangled_system.is_converged():
            await entangled_system.parallel_update()
            await entangled_system.synchronize_states()
        
        return entangled_system.extract_solution()
```

## 📊 LEARNING AND OPTIMIZATION METRICS

### Agentic Intelligence Indicators
```python
INTELLIGENCE_METRICS = {
    "reasoning_capability": {
        "problem_decomposition_depth": "measure complexity handling",
        "solution_novelty_score": "track creative problem-solving",
        "cross_domain_transfer": "measure knowledge generalization",
        "meta_learning_rate": "speed of learning-to-learn improvement"
    },
    "collaboration_effectiveness": {
        "multi_agent_coordination": "measure swarm intelligence emergence",
        "knowledge_sharing_efficiency": "track collective learning",
        "conflict_resolution_success": "agent negotiation capabilities",
        "emergent_behavior_discovery": "novel patterns from interaction"
    },
    "adaptation_intelligence": {
        "environment_adaptation_speed": "measure plasticity",
        "failure_recovery_capability": "resilience and learning from errors",
        "strategy_evolution_rate": "continuous self-improvement",
        "uncertainty_handling": "decision-making under ambiguity"
    },
    "exploration_metrics": {
        "curiosity_driven_exploration": "autonomous learning motivation",
        "pattern_discovery_rate": "finding hidden structures",
        "hypothesis_generation": "creative theory formation",
        "experimental_design": "systematic exploration strategies"
    }
}
```

### Continuous Evolution Process
1. **Real-time**: Continuous learning and adaptation during execution
2. **Per-interaction**: Extract patterns and update strategies
3. **Daily**: Meta-learning from accumulated experiences
4. **Weekly**: Evolutionary optimization of agent populations
5. **Monthly**: Architecture-level improvements and new capabilities

## 🎯 ADVANCED EXECUTION COMMANDS

### Experimental Development Workflow
```bash
# RESEARCH: Explore cutting-edge agentic AI techniques
python research/experimental_agents.py --mode=exploration --domain=multi_modal

# EVOLUTION: Run genetic algorithm optimization on agent populations
python evolution/genetic_optimizer.py --population=50 --generations=100

# EMERGENCE: Detect and analyze emergent behaviors
python analysis/emergence_detector.py --network=agent_swarm --threshold=novelty_high

# CONSCIOUSNESS: Experiment with consciousness-like behaviors
python experimental/consciousness_engine.py --agents=metacognitive --simulation_depth=5

# NEURO_SYMBOLIC: Hybrid reasoning experiments
python reasoning/neuro_symbolic.py --problem_type=complex --integration_mode=tight
```

### Meta-Agent Development Cycle
```bash
# 1. CONCEIVE: Generate novel agent architectures
python meta/agent_architect.py --generate --constraints=domain_specific --novelty=high

# 2. EVOLVE: Evolutionary development with fitness functions
python evolution/agent_evolution.py --fitness=multi_objective --selection=tournament

# 3. EMERGE: Multi-agent emergence experiments
python emergence/swarm_experiments.py --agents=100 --interaction_rules=adaptive

# 4. CONSCIOUSNESS: Self-aware agent development
python consciousness/self_aware_agents.py --reflection_depth=recursive --awareness_level=meta

# 5. TRANSCEND: Push boundaries of agentic intelligence
python transcendence/boundary_pushing.py --explore_impossible --paradigm_shift=enabled
```

## 🧪 EXPERIMENTAL FEATURES

### Advanced AI Techniques
- **Chain-of-Thought Reasoning**: For complex accounting decisions
- **Tool Use Optimization**: Dynamic tool selection and chaining
- **Multi-Modal Processing**: Handle PDFs, images, and structured data
- **Retrieval-Augmented Generation**: Leverage accounting knowledge bases
- **Fine-Tuning**: Custom models for domain-specific tasks

### Emerging Capabilities
- **Autonomous Workflow Design**: Agents that create their own workflows
- **Self-Healing Systems**: Automatic error detection and correction
- **Predictive Analytics**: Forecast accounting issues before they occur
- **Natural Language Queries**: "Show me all invoices over $10k from last quarter"

## 🎓 LEARNING ACCELERATION

### Skill Acquisition Strategy
1. **Foundation Phase**: Master Claude Code + basic accounting automation
2. **Expansion Phase**: Add Microsoft AI and LangChain capabilities  
3. **Specialization Phase**: Deep expertise in enterprise accounting systems
4. **Innovation Phase**: Create novel solutions and thought leadership

### Portfolio Development
- **Showcase Projects**: 5+ production-ready accounting automation systems
- **Technical Depth**: Multi-framework implementations of same use case
- **Business Impact**: Quantified ROI and efficiency improvements
- **Thought Leadership**: Blog posts, conference talks, open-source contributions

## 🚨 CRITICAL SUCCESS FACTORS

### Non-Negotiable Requirements
- **Business Value First**: Every feature must solve a real accounting problem
- **Production Quality**: All code must be enterprise-ready with proper error handling
- **Security by Design**: Financial data requires maximum security and compliance
- **Continuous Learning**: Stay current with AI advances and accounting regulations
- **User-Centric Design**: Solutions must be usable by non-technical accounting professionals

### Failure Prevention
- **Over-Engineering**: Start simple, add complexity only when needed
- **Security Gaps**: Never compromise on data security or regulatory compliance  
- **Technical Debt**: Refactor regularly to maintain code quality
- **Scope Creep**: Stay focused on accountancy domain until mastery is achieved
- **Isolation**: Actively engage with accounting professionals and AI communities

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

**Remember**: Focus on building practical, working AI agent systems that can coordinate effectively in a Windows development environment.