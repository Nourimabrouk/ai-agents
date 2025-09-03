"""
Comprehensive System Integration Demo
Demonstrates all new components working together
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import logging
from typing import Any, Dict, List
from datetime import datetime, timedelta

# Import all new components
from agents.temporal.temporal_agent import TemporalAgent
from agents.temporal.temporal_engine import TemporalEvent, TimeHorizon
from agents.learning.meta_learning_agent import MetaLearningAgent
from utils.memory.vector_memory import VectorMemoryStore
from tests.advanced.behavior_validator import BehaviorValidator, BehaviorConstraint, BehaviorExpectation
from dashboard.coordination.dashboard_server import CoordinationDashboard
from core.orchestration.orchestrator import AgentOrchestrator, Task

# Existing components
from templates.base_agent import BaseAgent
from utils.observability.logging import get_logger

logger = get_logger(__name__)


class ComprehensiveSystemDemo:
    """
    Integration demo showing all components working together
    """
    
    def __init__(self):
        self.orchestrator = AgentOrchestrator("demo_orchestrator")
        self.temporal_agent = TemporalAgent("temporal_demo")
        self.learning_agent = MetaLearningAgent("learning_demo")
        self.memory_store = VectorMemoryStore("demo_memory")
        self.behavior_validator = BehaviorValidator("demo_validator")
        self.dashboard = CoordinationDashboard(port=8081)
        
        # Demo agents for testing
        self.demo_agents = []
        
        logger.info("Initialized comprehensive system demo")
    
    async def setup_system(self):
        """Set up the complete system"""
        logger.info("Setting up comprehensive demo system...")
        
        # 1. Register agents with orchestrator
        await self._register_agents()
        
        # 2. Set up behavior validation constraints
        await self._setup_behavior_validation()
        
        # 3. Initialize memory with sample data
        await self._initialize_memory_system()
        
        # 4. Configure temporal reasoning
        await self._setup_temporal_system()
        
        # 5. Initialize learning system
        await self._setup_learning_system()
        
        # 6. Attach dashboard
        self.dashboard.attach_orchestrator(self.orchestrator)
        
        logger.info("System setup completed successfully")
    
    async def _register_agents(self):
        """Register all agents with the orchestrator"""
        
        # Create some demo agents
        class DemoWorkAgent(BaseAgent):
            async def execute(self, task: Any, action) -> Any:
                await asyncio.sleep(0.1)  # Simulate work
                return {
                    "result": f"Processed task: {str(task)[:50]}",
                    "agent": self.name,
                    "timestamp": datetime.now().isoformat()
                }
        
        # Create multiple demo agents
        for i in range(3):
            agent = DemoWorkAgent(f"demo_agent_{i}")
            self.demo_agents.append(agent)
            self.orchestrator.register_agent(agent)
        
        # Register specialized agents
        self.orchestrator.register_agent(self.temporal_agent)
        self.orchestrator.register_agent(self.learning_agent)
        
        logger.info(f"Registered {len(self.orchestrator.agents)} agents")
    
    async def _setup_behavior_validation(self):
        """Set up behavior validation constraints"""
        
        # Add constraints for each agent
        for agent_name in self.orchestrator.agents.keys():
            # Response time constraint
            self.behavior_validator.add_response_time_constraint(
                agent_name, max_response_time=5.0
            )
            
            # Success rate constraint
            self.behavior_validator.add_success_rate_constraint(
                agent_name, min_success_rate=0.7
            )
            
            # Learning progression for learning agent
            if "learning" in agent_name:
                self.behavior_validator.add_learning_progression_constraint(
                    agent_name, min_improvement_rate=0.05
                )
        
        logger.info("Behavior validation constraints configured")
    
    async def _initialize_memory_system(self):
        """Initialize memory system with sample data"""
        
        # Store sample memories
        sample_memories = [
            {
                "content": "Successful task completion using parallel processing",
                "metadata": {"task_type": "processing", "success": True, "domain": "computation"},
                "tags": ["success", "parallel", "processing"]
            },
            {
                "content": "Error handling strategy for network timeouts",
                "metadata": {"task_type": "network", "success": False, "domain": "networking"},
                "tags": ["error", "network", "timeout"]
            },
            {
                "content": "Optimization technique for large dataset analysis",
                "metadata": {"task_type": "analysis", "success": True, "domain": "data_science"},
                "tags": ["optimization", "analysis", "big_data"]
            },
            {
                "content": "Collaborative problem solving with multiple agents",
                "metadata": {"task_type": "collaboration", "success": True, "domain": "coordination"},
                "tags": ["collaboration", "multi_agent", "coordination"]
            },
            {
                "content": "Learning from previous mistakes in decision making",
                "metadata": {"task_type": "learning", "success": True, "domain": "meta_learning"},
                "tags": ["learning", "mistakes", "improvement"]
            }
        ]
        
        for memory in sample_memories:
            await self.memory_store.store_memory(
                content=memory["content"],
                metadata=memory["metadata"],
                tags=memory["tags"]
            )
        
        logger.info(f"Initialized memory system with {len(sample_memories)} sample memories")
    
    async def _setup_temporal_system(self):
        """Set up temporal reasoning system"""
        
        # Add sample temporal events
        sample_events = [
            {
                "event_type": "task_completed",
                "horizon": TimeHorizon.SECOND,
                "data": {"task_id": "demo_001", "agent": "demo_agent_0"}
            },
            {
                "event_type": "performance_optimization",
                "horizon": TimeHorizon.MINUTE,
                "data": {"optimization_type": "memory_usage", "improvement": 15}
            },
            {
                "event_type": "strategic_planning",
                "horizon": TimeHorizon.HOUR,
                "data": {"planning_cycle": "daily", "objectives": ["efficiency", "learning"]}
            },
            {
                "event_type": "system_maintenance",
                "horizon": TimeHorizon.DAY,
                "data": {"maintenance_type": "routine", "components": ["memory", "logging"]}
            }
        ]
        
        for event_data in sample_events:
            event = TemporalEvent(
                timestamp=datetime.now(),
                event_type=event_data["event_type"],
                data=event_data["data"],
                confidence=0.8,
                horizon=event_data["horizon"]
            )
            
            await self.temporal_agent.temporal_engine.add_event(event)
        
        # Add some temporal objectives
        await self.temporal_agent.add_temporal_objective(
            "optimize_throughput",
            TimeHorizon.MINUTE,
            priority=1
        )
        
        await self.temporal_agent.add_temporal_objective(
            "improve_learning_efficiency",
            TimeHorizon.HOUR,
            priority=2
        )
        
        logger.info("Temporal system configured with sample events and objectives")
    
    async def _setup_learning_system(self):
        """Set up learning system with initial experiences"""
        
        # Let the learning agent process some initial tasks
        initial_tasks = [
            {"description": "analyze performance data", "domain": "analysis"},
            {"description": "optimize resource usage", "domain": "optimization"},
            {"description": "coordinate with other agents", "domain": "coordination"},
            {"description": "learn from past mistakes", "domain": "meta_learning"},
            {"description": "generate creative solutions", "domain": "creative"}
        ]
        
        for task in initial_tasks:
            await self.learning_agent.process_task(
                task["description"],
                {"domain": task["domain"]}
            )
        
        logger.info("Learning system initialized with sample experiences")
    
    async def run_comprehensive_demo(self):
        """Run comprehensive demo showcasing all components"""
        logger.info("Starting comprehensive system demo...")
        
        # Phase 1: Basic orchestration
        await self._demo_basic_orchestration()
        
        # Phase 2: Temporal reasoning
        await self._demo_temporal_reasoning()
        
        # Phase 3: Memory and learning
        await self._demo_memory_and_learning()
        
        # Phase 4: Behavior validation
        await self._demo_behavior_validation()
        
        # Phase 5: Advanced coordination
        await self._demo_advanced_coordination()
        
        logger.info("Comprehensive demo completed successfully")
    
    async def _demo_basic_orchestration(self):
        """Demonstrate basic orchestration capabilities"""
        logger.info("\n=== Phase 1: Basic Orchestration ===")
        
        # Create test tasks
        tasks = [
            Task(
                id="demo_task_1",
                description="Process customer data and generate insights",
                requirements={"data_type": "customer", "output_format": "insights"}
            ),
            Task(
                id="demo_task_2",
                description="Optimize system performance across multiple metrics",
                requirements={"metrics": ["cpu", "memory", "throughput"], "target": "optimization"}
            ),
            Task(
                id="demo_task_3",
                description="Collaborate on solving a complex problem requiring multiple perspectives",
                requirements={"complexity": "high", "collaboration": "required"}
            )
        ]
        
        # Execute tasks
        results = []
        for task in tasks:
            result = await self.orchestrator.delegate_task(task)
            results.append(result)
            logger.info(f"Completed task {task.id}: {str(result)[:100]}...")
        
        # Display orchestrator metrics
        metrics = self.orchestrator.get_metrics()
        logger.info(f"Orchestrator metrics: {metrics}")
    
    async def _demo_temporal_reasoning(self):
        """Demonstrate temporal reasoning capabilities"""
        logger.info("\n=== Phase 2: Temporal Reasoning ===")
        
        # Test prediction capabilities
        predictions = await self.temporal_agent.process_task(
            "predict future events for horizon=minute confidence=0.7"
        )
        logger.info(f"Temporal predictions: {predictions}")
        
        # Test multi-horizon optimization
        optimization = await self.temporal_agent.process_task(
            "optimize objective=maximize_efficiency across all time horizons"
        )
        logger.info(f"Multi-horizon optimization: {optimization}")
        
        # Test pattern analysis
        patterns = await self.temporal_agent.process_task(
            "analyze_patterns in temporal data"
        )
        logger.info(f"Temporal patterns: {patterns}")
        
        # Add real-time events
        current_events = [
            {"event_type": "demo_execution", "horizon": "second"},
            {"event_type": "performance_spike", "horizon": "minute"},
            {"event_type": "learning_improvement", "horizon": "hour"}
        ]
        
        for event_data in current_events:
            await self.temporal_agent.process_task(
                f"add_event event_type={event_data['event_type']} horizon={event_data['horizon']}"
            )
        
        # Get temporal insights
        insights = await self.temporal_agent.get_temporal_insights()
        logger.info(f"Temporal insights: {insights}")
    
    async def _demo_memory_and_learning(self):
        """Demonstrate memory and learning capabilities"""
        logger.info("\n=== Phase 3: Memory and Learning ===")
        
        # Test memory search
        search_results = await self.memory_store.search_similar(
            "successful task completion strategies",
            limit=3,
            similarity_threshold=0.5
        )
        
        logger.info(f"Memory search found {len(search_results)} relevant memories")
        for memory, similarity in search_results:
            logger.info(f"  - {memory.content[:50]}... (similarity: {similarity:.3f})")
        
        # Store new memory from current demo
        demo_memory_id = await self.memory_store.store_memory(
            content="Comprehensive system demo showcasing integration of temporal reasoning, learning, and coordination",
            metadata={
                "demo_phase": "memory_learning",
                "timestamp": datetime.now().isoformat(),
                "success": True,
                "domain": "system_integration"
            },
            tags=["demo", "integration", "comprehensive", "success"]
        )
        
        logger.info(f"Stored demo memory: {demo_memory_id}")
        
        # Test learning capabilities
        learning_tasks = [
            "learn from recent task execution patterns",
            "optimize strategies based on performance data",
            "transfer knowledge from successful experiences",
            "reflect on learning progress and adapt approaches"
        ]
        
        for task in learning_tasks:
            result = await self.learning_agent.process_task(task)
            logger.info(f"Learning task '{task[:30]}...': {result}")
        
        # Get learning statistics
        learning_stats = self.learning_agent.get_learning_statistics()
        logger.info(f"Learning statistics: {learning_stats}")
    
    async def _demo_behavior_validation(self):
        """Demonstrate behavior validation capabilities"""
        logger.info("\n=== Phase 4: Behavior Validation ===")
        
        # Create test tasks for validation
        validation_tasks = [
            "simple task for response time testing",
            "complex analysis requiring thorough processing",
            "error-prone task to test resilience",
            "collaborative task for multi-agent validation"
        ]
        
        # Validate each agent
        validation_results = {}
        for agent_name, agent in self.orchestrator.agents.items():
            if hasattr(agent, 'process_task'):  # Only validate processable agents
                logger.info(f"Validating agent: {agent_name}")
                
                results = await self.behavior_validator.validate_agent(
                    agent, validation_tasks
                )
                
                validation_results[agent_name] = results
                
                # Display validation summary
                passed = sum(1 for r in results if r.passed)
                total = len(results)
                logger.info(f"  {agent_name}: {passed}/{total} validations passed")
                
                for result in results:
                    status = "PASS" if result.passed else "FAIL"
                    logger.info(f"    {status}: {result.test_name} (deviation: {result.deviation:.3f})")
        
        # Generate validation summary
        summary = self.behavior_validator.get_validation_summary()
        logger.info(f"Overall validation summary: {summary}")
    
    async def _demo_advanced_coordination(self):
        """Demonstrate advanced coordination patterns"""
        logger.info("\n=== Phase 5: Advanced Coordination ===")
        
        # Test different coordination patterns
        
        # 1. Parallel execution
        parallel_task = Task(
            id="parallel_demo",
            description="parallel processing task requiring multiple agents",
            requirements={"coordination": "parallel", "agents_needed": 3}
        )
        
        parallel_result = await self.orchestrator.delegate_task(parallel_task)
        logger.info(f"Parallel coordination result: {parallel_result}")
        
        # 2. Hierarchical delegation
        hierarchical_task = Task(
            id="hierarchical_demo",
            description="complex project requiring hierarchical task breakdown",
            requirements={"complexity": "high", "structure": "hierarchical"}
        )
        
        hierarchical_result = await self.orchestrator.hierarchical_delegation(hierarchical_task)
        logger.info(f"Hierarchical delegation result: {hierarchical_result}")
        
        # 3. Emergent behavior detection
        emergent_patterns = await self.orchestrator.emergent_behavior_detection()
        logger.info(f"Detected emergent patterns: {emergent_patterns}")
        
        # 4. Test swarm intelligence
        swarm_result = await self.orchestrator.swarm_intelligence(
            "optimize system configuration for maximum efficiency",
            swarm_size=5
        )
        logger.info(f"Swarm intelligence result: {swarm_result}")
        
        # 5. Cross-system integration
        # Use temporal agent to analyze orchestration patterns
        temporal_analysis = await self.temporal_agent.process_task(
            "analyze coordination patterns from orchestrator events"
        )
        logger.info(f"Temporal analysis of coordination: {temporal_analysis}")
        
        # Use learning agent to improve coordination strategies
        learning_improvement = await self.learning_agent.process_task(
            "learn from coordination patterns to improve future orchestration"
        )
        logger.info(f"Learning-based coordination improvement: {learning_improvement}")
    
    async def run_performance_benchmarks(self):
        """Run comprehensive performance benchmarks"""
        logger.info("\n=== Performance Benchmarks ===")
        
        # Benchmark temporal reasoning
        temporal_start = datetime.now()
        for i in range(10):
            await self.temporal_agent.process_task(f"benchmark task {i}")
        temporal_duration = (datetime.now() - temporal_start).total_seconds()
        
        logger.info(f"Temporal reasoning: 10 tasks in {temporal_duration:.3f}s ({10/temporal_duration:.1f} tasks/s)")
        
        # Benchmark memory operations
        memory_start = datetime.now()
        for i in range(20):
            await self.memory_store.store_memory(
                f"benchmark memory {i}",
                metadata={"benchmark": True, "index": i}
            )
        memory_duration = (datetime.now() - memory_start).total_seconds()
        
        logger.info(f"Memory storage: 20 operations in {memory_duration:.3f}s ({20/memory_duration:.1f} ops/s)")
        
        # Benchmark orchestration
        orchestration_start = datetime.now()
        tasks = [
            Task(id=f"bench_task_{i}", description=f"benchmark task {i}", requirements={})
            for i in range(15)
        ]
        
        orchestration_tasks = [self.orchestrator.delegate_task(task) for task in tasks]
        await asyncio.gather(*orchestration_tasks)
        orchestration_duration = (datetime.now() - orchestration_start).total_seconds()
        
        logger.info(f"Orchestration: 15 tasks in {orchestration_duration:.3f}s ({15/orchestration_duration:.1f} tasks/s)")
        
        # Memory usage and statistics
        memory_stats = self.memory_store.get_statistics()
        logger.info(f"Memory system statistics: {memory_stats}")
        
        temporal_state = self.temporal_agent.temporal_engine.get_temporal_state()
        logger.info(f"Temporal system state: {temporal_state}")
        
        orchestrator_metrics = self.orchestrator.get_metrics()
        logger.info(f"Orchestrator metrics: {orchestrator_metrics}")
    
    async def cleanup_demo(self):
        """Clean up demo resources"""
        logger.info("Cleaning up demo resources...")
        
        try:
            await self.memory_store.cleanup()
            logger.info("Memory store cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up memory store: {e}")
        
        logger.info("Demo cleanup completed")


async def main():
    """Main demo function"""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    demo = ComprehensiveSystemDemo()
    
    try:
        # Set up the system
        await demo.setup_system()
        
        # Run comprehensive demo
        await demo.run_comprehensive_demo()
        
        # Run performance benchmarks
        await demo.run_performance_benchmarks()
        
        # Optional: Start dashboard (commented out to avoid blocking)
        # logger.info("Dashboard available at: http://localhost:8081")
        # await demo.dashboard.start_server()
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
        raise
    finally:
        # Cleanup
        await demo.cleanup_demo()


if __name__ == "__main__":
    asyncio.run(main())