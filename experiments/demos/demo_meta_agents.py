"""
Comprehensive demonstration of the Meta AI Agent System

This script showcases the complete meta-agent architecture with:
- Multi-agent coordination
- Parallel planning and execution  
- Inter-agent communication
- Real development scenario simulation
- Windows-optimized async patterns

Run with: python demo_meta_agents.py
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our meta-agent system
try:
    from agents.meta.enhanced_meta_orchestrator import EnhancedMetaOrchestrator
    from agents.meta.meta_orchestrator import MetaOrchestrator, AgentRole, TaskPriority
    print("✅ Using Enhanced Meta-Orchestrator")
    ENHANCED_MODE = True
except ImportError:
    from agents.meta.meta_orchestrator import MetaOrchestrator, AgentRole, TaskPriority
    print("⚠️ Using Base Meta-Orchestrator (Enhanced not available)")
    ENHANCED_MODE = False

try:
    from agents.meta import (
        MessageBus,
        AgentInterface, 
        ParallelCoordinator,
        Message,
        MessageType,
        MessagePriority,
        create_meta_system
    )
except ImportError:
    print("⚠️ Some meta-agent components not available, using simulation mode")
    ENHANCED_MODE = False


class DemoAgent(AgentInterface):
    """
    Demo agent that simulates real development work
    """
    
    def __init__(self, agent_id: str, message_bus: MessageBus, agent_type: str):
        super().__init__(agent_id, message_bus)
        self.agent_type = agent_type
        self.work_completed = 0
        self.skills = self._get_skills()
    
    def _get_skills(self) -> List[str]:
        """Get skills based on agent type"""
        skill_map = {
            "architect": ["system_design", "api_design", "patterns", "documentation"],
            "developer": ["python", "javascript", "async", "testing", "optimization"],
            "tester": ["pytest", "unittest", "integration_testing", "coverage"],
            "reviewer": ["code_review", "security", "performance", "best_practices"]
        }
        return skill_map.get(self.agent_type, ["general"])
    
    async def _handle_request(self, message: Message) -> Any:
        """Handle work request"""
        task = message.content
        logger.info(f"{self.agent_id} ({self.agent_type}) starting: {task.get('description', 'unknown task')}")
        
        # Simulate work based on task complexity
        work_time = task.get('estimated_time', 1.0)
        await asyncio.sleep(work_time * 0.1)  # Scaled for demo
        
        self.work_completed += 1
        
        # Simulate different outcomes
        success_rate = 0.9 if self.agent_type == "developer" else 0.95
        
        result = {
            "agent": self.agent_id,
            "task": task.get('description'),
            "success": success_rate > 0.1,  # High success rate for demo
            "output": f"Completed by {self.agent_type} specialist",
            "time_taken": work_time,
            "skills_used": self.skills[:2]  # Top 2 relevant skills
        }
        
        # Publish completion event
        await self.publish_event("task_completed", result)
        
        return result
    
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "skills": self.skills,
            "work_completed": self.work_completed,
            "paused": self.paused,
            "load": "light" if self.work_completed < 3 else "moderate"
        }


async def demonstrate_agent_communication():
    """Demonstrate inter-agent communication patterns"""
    print("\n🔗 Demonstrating Agent Communication")
    print("=" * 50)
    
    # Create message bus
    bus = MessageBus()
    await bus.start()
    
    # Create diverse agents
    agents = {
        "architect": DemoAgent("architect_001", bus, "architect"),
        "developer": DemoAgent("developer_001", bus, "developer"),
        "tester": DemoAgent("tester_001", bus, "tester"),
        "reviewer": DemoAgent("reviewer_001", bus, "reviewer")
    }
    
    # Register agents
    for agent_id, agent in agents.items():
        bus.register_agent(agent_id, agent)
    
    # Test 1: Direct agent-to-agent communication
    print("📞 Testing direct communication...")
    response = await agents["architect"].send_message(
        "developer_001",
        {"description": "Design review required", "priority": "high"},
        MessageType.REQUEST
    )
    print(f"   Response: {response}")
    
    # Test 2: Broadcast communication
    print("\n📢 Testing broadcast...")
    await agents["architect"].broadcast({
        "announcement": "New coding standards published",
        "url": "https://internal.wiki/standards"
    })
    
    # Test 3: Event publication
    print("\n📡 Testing event system...")
    await agents["tester"].publish_event("bug_found", {
        "severity": "medium",
        "location": "user_auth.py:142",
        "description": "Input validation missing"
    })
    
    # Test 4: Query status of all agents
    print("\n❓ Querying agent statuses...")
    status = await agents["reviewer"].query("*", "agent_status")
    for agent_id, agent_status in status.items():
        print(f"   {agent_id}: {agent_status['load']} load, {agent_status['work_completed']} tasks completed")
    
    await bus.stop()
    return agents


async def demonstrate_parallel_coordination():
    """Demonstrate parallel task coordination"""
    print("\n⚡ Demonstrating Parallel Coordination")
    print("=" * 50)
    
    bus = MessageBus()
    await bus.start()
    
    # Create agents
    agents = [
        DemoAgent(f"worker_{i:02d}", bus, "developer") for i in range(1, 6)
    ]
    
    for agent in agents:
        bus.register_agent(agent.agent_id, agent)
    
    coordinator = ParallelCoordinator(bus)
    coordinator.create_execution_pool("dev_pool", max_concurrent=3)
    
    # Test 1: Parallel execution with concurrency limits
    print("🚀 Executing parallel tasks...")
    tasks = [
        {
            "agent": f"worker_{i:02d}",
            "content": {
                "description": f"Implement feature #{i}",
                "estimated_time": i * 0.5
            }
        }
        for i in range(1, 6)
    ]
    
    start_time = datetime.now()
    results = await coordinator.execute_parallel(tasks, "dev_pool")
    duration = (datetime.now() - start_time).total_seconds()
    
    print(f"   ✅ Completed {len(results)} tasks in {duration:.2f} seconds")
    successful = [r for r in results if not isinstance(r, Exception) and r.get("success")]
    print(f"   📊 Success rate: {len(successful)}/{len(results)} ({len(successful)/len(results)*100:.1f}%)")
    
    # Test 2: Pipeline execution
    print("\n🔄 Testing pipeline execution...")
    pipeline_stages = [
        [{"agent": "worker_01", "content": {"description": "Stage 1: Analysis", "estimated_time": 1}}],
        [
            {"agent": "worker_02", "content": {"description": "Stage 2a: Implementation A", "estimated_time": 1.5}},
            {"agent": "worker_03", "content": {"description": "Stage 2b: Implementation B", "estimated_time": 1.2}}
        ],
        [{"agent": "worker_04", "content": {"description": "Stage 3: Integration", "estimated_time": 0.8}}]
    ]
    
    pipeline_results = await coordinator.pipeline(pipeline_stages, "dev_pool")
    print(f"   ✅ Pipeline completed with {len(pipeline_stages)} stages")
    
    # Test 3: Map-Reduce pattern
    print("\n🗺️ Testing Map-Reduce pattern...")
    
    map_tasks = [
        {"agent": f"worker_{i:02d}", "content": {"description": f"Process batch {i}", "estimated_time": 0.5}}
        for i in range(1, 5)
    ]
    
    async def reduce_results(results):
        total_time = sum(r.get("time_taken", 0) for r in results if isinstance(r, dict))
        return {
            "total_batches": len(results),
            "total_time": total_time,
            "average_time": total_time / len(results) if results else 0
        }
    
    map_reduce_result = await coordinator.map_reduce(map_tasks, reduce_results, "dev_pool")
    print(f"   ✅ Map-Reduce result: {map_reduce_result}")
    
    await bus.stop()
    return results


async def demonstrate_real_world_scenario():
    """Demonstrate a real-world development scenario"""
    print("\n🏢 Real-World Scenario: Building an AI Assistant")
    print("=" * 60)
    
    requirement = """
    Build an AI-powered customer service assistant that can:
    
    1. Understand natural language customer queries
    2. Access knowledge base and FAQ systems  
    3. Integrate with CRM for customer context
    4. Generate appropriate responses with sentiment awareness
    5. Escalate complex issues to human agents
    6. Learn from interaction feedback
    7. Support multiple languages
    8. Ensure data privacy and security
    9. Scale to handle 10,000+ concurrent users
    10. Provide analytics and reporting dashboard
    """
    
    context = {
        "framework": "langchain",
        "target_systems": ["salesforce", "zendesk", "slack"],
        "languages": ["python", "javascript"],
        "database": "postgresql",
        "deployment": "kubernetes",
        "timeline": "8 weeks",
        "team_size": "6 developers",
        "priority": "high",
        "compliance": ["GDPR", "CCPA"]
    }
    
    # Create meta-orchestrator with enhanced capabilities if available
    if ENHANCED_MODE:
        try:
            orchestrator = EnhancedMetaOrchestrator()
            print("🚀 Using Enhanced Meta-Orchestrator with advanced optimization")
        except Exception as e:
            print(f"⚠️ Enhanced orchestrator failed, falling back to base: {e}")
            orchestrator = MetaOrchestrator()
    else:
        try:
            orchestrator = create_meta_system()
        except:
            orchestrator = MetaOrchestrator()
    
    print("📋 Planning comprehensive development...")
    start_time = datetime.now()
    
    # Plan development with full context
    if ENHANCED_MODE and hasattr(orchestrator, 'process_request'):
        # Use enhanced meta-optimal processing
        print("🧠 Using meta-optimal request processing...")
        result = await orchestrator.process_request(requirement, context)
        planning_time = (datetime.now() - start_time).total_seconds()
        
        print(f"✅ Meta-optimal processing completed in {planning_time:.2f} seconds")
        
        if isinstance(result, dict):
            strategy = result.get('execution_strategy', 'unknown')
            score = result.get('optimization_score', 0)
            total_tasks = result.get('total_tasks', 0)
            
            print(f"🎯 Strategy Used: {strategy}")
            print(f"📊 Optimization Score: {score:.2f}")
            print(f"📋 Total Tasks: {total_tasks}")
            
            # Create mock tasks list for compatibility
            tasks = []
            for i in range(total_tasks):
                from agents.meta.meta_orchestrator import DevelopmentTask
                task = DevelopmentTask(
                    id=f"meta_task_{i:03d}",
                    description=f"Meta-optimized task {i+1}",
                    priority=TaskPriority.MEDIUM
                )
                tasks.append(task)
        else:
            tasks = []
    else:
        # Use standard planning
        tasks = await orchestrator.plan_development(requirement, context)
        planning_time = (datetime.now() - start_time).total_seconds()
    
    print(f"✅ Planning completed in {planning_time:.2f} seconds")
    print(f"📊 Generated {len(tasks)} total tasks across all agents")
    
    # Analyze task distribution
    task_distribution = {}
    for task in tasks:
        for agent_role in task.assigned_agents:
            task_distribution[agent_role.value] = task_distribution.get(agent_role.value, 0) + 1
    
    print("\n📈 Task Distribution by Agent Role:")
    for role, count in sorted(task_distribution.items()):
        print(f"   {role.capitalize()}: {count} tasks")
    
    # Show sample tasks from each category
    print("\n📝 Sample Tasks by Priority:")
    for priority in ["CRITICAL", "HIGH", "MEDIUM"]:
        priority_tasks = [t for t in tasks if t.priority.name == priority][:3]
        if priority_tasks:
            print(f"\n   {priority} Priority:")
            for task in priority_tasks:
                agents_str = ", ".join([a.value for a in task.assigned_agents])
                print(f"     • {task.description} [{agents_str}]")
                if task.dependencies:
                    print(f"       Dependencies: {len(task.dependencies)} tasks")
    
    # Execute development (simulation)
    print(f"\n🚀 Executing development with {orchestrator.config['max_parallel_agents']} parallel agents...")
    execution_start = datetime.now()
    
    results = await orchestrator.execute_development(max_parallel=4)
    execution_time = (datetime.now() - execution_start).total_seconds()
    
    print(f"✅ Development execution completed!")
    print(f"⏱️ Total execution time: {execution_time:.2f} seconds")
    print(f"📊 Results breakdown:")
    print(f"   • Successfully completed: {len(results['completed'])} tasks")
    print(f"   • Failed tasks: {len(results['failed'])} tasks")
    print(f"   • Skipped tasks: {len(results['skipped'])} tasks")
    
    if results['failed']:
        print("\n⚠️ Failed Tasks Analysis:")
        for failed_task in results['failed'][:3]:  # Show first 3 failures
            print(f"     • {failed_task.description}")
            if hasattr(failed_task, 'result') and failed_task.result:
                print(f"       Error: {failed_task.result.get('error', 'Unknown error')}")
    
    # Calculate efficiency metrics
    success_rate = len(results['completed']) / (len(results['completed']) + len(results['failed'])) * 100
    tasks_per_second = len(results['completed']) / execution_time if execution_time > 0 else 0
    
    print(f"\n📈 Performance Metrics:")
    print(f"   • Success rate: {success_rate:.1f}%")
    print(f"   • Tasks per second: {tasks_per_second:.2f}")
    print(f"   • Planning efficiency: {len(tasks)/planning_time:.1f} tasks planned per second")
    
    # Show enhanced orchestrator metrics if available
    if ENHANCED_MODE and hasattr(orchestrator, 'optimization_metrics'):
        print(f"\n🧠 Meta-Optimization Metrics:")
        metrics = orchestrator.optimization_metrics
        
        for metric, value in metrics.items():
            if isinstance(value, timedelta):
                print(f"   • {metric.replace('_', ' ').title()}: {value.total_seconds():.2f}s")
            elif isinstance(value, (int, float)):
                if 'rate' in metric or 'ratio' in metric:
                    print(f"   • {metric.replace('_', ' ').title()}: {value:.1%}")
                else:
                    print(f"   • {metric.replace('_', ' ').title()}: {value:.3f}")
            elif hasattr(value, '__len__'):
                print(f"   • {metric.replace('_', ' ').title()}: {len(value)} entries")
        
        # Generate detailed optimization report if available
        if hasattr(orchestrator, 'get_optimization_report'):
            print(f"\n📊 Generating detailed optimization report...")
            try:
                opt_report = await orchestrator.get_optimization_report()
                # Save the report to a file
                with open("optimization_report.txt", "w") as f:
                    f.write(opt_report)
                print(f"💾 Detailed report saved to optimization_report.txt")
            except Exception as e:
                print(f"⚠️ Could not generate optimization report: {e}")
    
    # Generate final report
    print("\n📋 Generating comprehensive report...")
    report = await orchestrator.generate_report()
    
    # Save results to file
    results_file = Path("demo_results.json")
    demo_results = {
        "timestamp": datetime.now().isoformat(),
        "requirement": requirement,
        "context": context,
        "planning_time": planning_time,
        "execution_time": execution_time,
        "total_tasks": len(tasks),
        "task_distribution": task_distribution,
        "success_rate": success_rate,
        "performance_metrics": {
            "tasks_per_second": tasks_per_second,
            "planning_efficiency": len(tasks)/planning_time
        }
    }
    
    with open(results_file, 'w') as f:
        json.dump(demo_results, f, indent=2)
    
    print(f"💾 Results saved to {results_file}")
    
    return orchestrator, results, demo_results


async def main():
    """Main demonstration function"""
    print("🤖 Meta AI Agent System - Comprehensive Demonstration")
    print("=" * 60)
    print("Windows-optimized, async-first AI agent coordination system")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Demo 1: Agent Communication
        agents = await demonstrate_agent_communication()
        
        # Demo 2: Parallel Coordination
        await demonstrate_parallel_coordination()
        
        # Demo 3: Real-world Scenario
        orchestrator, results, demo_results = await demonstrate_real_world_scenario()
        
        print("\n🎉 All Demonstrations Completed Successfully!")
        print("=" * 60)
        print("\n📊 Summary:")
        print(f"   • Communication patterns: ✅ Tested")
        print(f"   • Parallel coordination: ✅ Tested")
        print(f"   • Real-world scenario: ✅ Completed")
        print(f"   • Task success rate: {demo_results['success_rate']:.1f}%")
        print(f"   • Performance: {demo_results['performance_metrics']['tasks_per_second']:.2f} tasks/sec")
        
        print("\n🚀 Next Steps:")
        print("   1. Review generated code examples in specialized_agents.py")
        print("   2. Customize agent capabilities for your specific domain")
        print("   3. Integrate with your existing development tools")
        print("   4. Scale up with more specialized agents")
        print("   5. Add domain-specific knowledge bases")
        
        return True
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        return False


if __name__ == "__main__":
    # Run the comprehensive demonstration
    success = asyncio.run(main())
    
    if success:
        print(f"\n✅ Demo completed successfully at {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ Demo failed at {datetime.now().strftime('%H:%M:%S')}")
        exit(1)