"""
AI Agents System - Production Entry Point
Clean, microservice-based architecture with zero circular dependencies
"""

import asyncio
import logging
import sys
from typing import Dict, Any
from datetime import datetime, timezone

from core import (
    initialize_system,
    shutdown_system,
    get_system_health,
    AgentId,
    TaskId,
    ExecutionContext,
    ExecutionResult,
    Priority,
    get_service,
    IOrchestrator,
    ISecurityMonitor,
    AutonomousIntelligenceService,
    DeploymentManager,
    DeploymentConfiguration
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_agents.log')
    ]
)

logger = logging.getLogger(__name__)


class AIAgentsSystem:
    """
    Main system facade providing clean API for AI Agents functionality
    """
    
    def __init__(self):
        self._initialized = False
        self._running = False
    
    async def start(self, deployment_config: Dict[str, Any] = None) -> bool:
        """
        Start the AI Agents system
        """
        try:
            logger.info("Starting AI Agents System...")
            
            # Initialize core system
            await initialize_system()
            self._initialized = True
            
            # Deploy system if configuration provided
            if deployment_config:
                deployment_manager = get_service(DeploymentManager)
                config = DeploymentConfiguration(**deployment_config)
                success = await deployment_manager.deploy_system(config)
                
                if not success:
                    logger.error("Failed to deploy system")
                    return False
            
            self._running = True
            logger.info("AI Agents System started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    async def stop(self) -> bool:
        """
        Stop the AI Agents system gracefully
        """
        try:
            if not self._running:
                logger.warning("System is not running")
                return True
            
            logger.info("Stopping AI Agents System...")
            await shutdown_system()
            
            self._running = False
            self._initialized = False
            
            logger.info("AI Agents System stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping system: {e}")
            return False
    
    async def execute_task(self, task_data: Dict[str, Any], 
                         agent_namespace: str = "system",
                         priority: Priority = Priority.NORMAL) -> ExecutionResult:
        """
        Execute a task using the orchestration system
        """
        if not self._running:
            return ExecutionResult(
                success=False,
                error=Exception("System is not running")
            )
        
        try:
            orchestrator = get_service(IOrchestrator)
            
            # Create execution context
            context = ExecutionContext(
                task_id=TaskId(
                    domain="user",
                    task_type=task_data.get("type", "general"),
                    instance_id=f"{datetime.now(timezone.utc).timestamp()}"
                ),
                agent_id=AgentId(agent_namespace, "orchestrator"),
                priority=priority,
                timeout=task_data.get("timeout"),
                metadata=task_data.get("metadata", {})
            )
            
            # Execute task
            return await orchestrator.execute_task(context, task_data)
            
        except Exception as e:
            logger.error(f"Error executing task: {e}")
            return ExecutionResult(success=False, error=e)
    
    async def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        """
        if not self._initialized:
            return {
                "status": "not_initialized",
                "running": False,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        try:
            health = await get_system_health()
            health.update({
                "initialized": self._initialized,
                "running": self._running
            })
            return health
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                "status": "error",
                "error": str(e),
                "initialized": self._initialized,
                "running": self._running,
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def set_agent_autonomy(self, agent_id: str, autonomy_level: str) -> bool:
        """
        Set autonomy level for an agent
        """
        try:
            if not self._running:
                return False
            
            autonomous_service = get_service(AutonomousIntelligenceService)
            
            # Parse agent ID
            parts = agent_id.split(".")
            if len(parts) >= 2:
                agent_obj = AgentId(parts[0], parts[1])
            else:
                agent_obj = AgentId("default", agent_id)
            
            # Set autonomy level
            from core.autonomous.services import AutonomyLevel
            level = AutonomyLevel(autonomy_level)
            
            await autonomous_service.set_agent_autonomy_level(agent_obj, level)
            return True
            
        except Exception as e:
            logger.error(f"Error setting agent autonomy: {e}")
            return False
    
    async def get_security_status(self) -> Dict[str, Any]:
        """
        Get security monitoring status
        """
        try:
            if not self._running:
                return {"status": "not_running"}
            
            security_monitor = get_service(ISecurityMonitor)
            return await security_monitor.get_security_status()
            
        except Exception as e:
            logger.error(f"Error getting security status: {e}")
            return {"status": "error", "error": str(e)}
    
    def is_running(self) -> bool:
        """Check if system is running"""
        return self._running
    
    def is_initialized(self) -> bool:
        """Check if system is initialized"""
        return self._initialized


# Global system instance
_system_instance = None


def get_system() -> AIAgentsSystem:
    """Get global system instance"""
    global _system_instance
    if _system_instance is None:
        _system_instance = AIAgentsSystem()
    return _system_instance


# Convenience functions for common operations
async def start_system(config: Dict[str, Any] = None) -> bool:
    """Start the AI Agents system"""
    system = get_system()
    return await system.start(config)


async def stop_system() -> bool:
    """Stop the AI Agents system"""
    system = get_system()
    return await system.stop()


async def execute_task(task_data: Dict[str, Any]) -> ExecutionResult:
    """Execute a task using the system"""
    system = get_system()
    return await system.execute_task(task_data)


async def get_status() -> Dict[str, Any]:
    """Get system status"""
    system = get_system()
    return await system.get_status()


# Main entry point for testing and demos
async def main():
    """Main entry point for system demonstration"""
    
    # Default deployment configuration
    default_config = {
        "deployment_id": "ai-agents-demo",
        "version": "1.0.0",
        "environment": "development",
        "services": {
            "orchestrator": {"enabled": True, "instances": 1},
            "security": {"enabled": True, "instances": 1},
            "reasoning": {"enabled": True, "instances": 1},
            "autonomous": {"enabled": True, "instances": 1}
        },
        "resources": {
            "cpu_limit": "1000m",
            "memory_limit": "1Gi"
        },
        "scaling": {
            "min_instances": 1,
            "max_instances": 3,
            "target_cpu": 70
        },
        "monitoring": {
            "enabled": True,
            "interval": 30,
            "health_check_timeout": 5
        }
    }
    
    system = get_system()
    
    try:
        # Start system
        logger.info("=== AI Agents System Demonstration ===")
        success = await system.start(default_config)
        
        if not success:
            logger.error("Failed to start system")
            return {}
        
        # Get initial status
        status = await system.get_status()
        logger.info(f"System Status: {status['overall_status']}")
        
        # Execute sample task
        sample_task = {
            "type": "demo",
            "description": "Demonstrate system capabilities",
            "capability": "general",
            "data": {
                "input": "Process this demonstration task",
                "parameters": {
                    "complexity": "simple",
                    "priority": "normal"
                }
            }
        }
        
        logger.info("Executing sample task...")
        result = await system.execute_task(sample_task)
        
        if result.success:
            logger.info(f"Task completed successfully in {result.execution_time:.2f}s")
        else:
            logger.error(f"Task failed: {result.error}")
        
        # Get final status
        final_status = await system.get_status()
        logger.info(f"Final Status: {final_status}")
        
        # Wait a bit to see monitoring in action
        logger.info("Running for 10 seconds to demonstrate monitoring...")
        await asyncio.sleep(10)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in demonstration: {e}")
    finally:
        # Clean shutdown
        logger.info("Shutting down system...")
        await system.stop()
        logger.info("=== Demonstration Complete ===")


if __name__ == "__main__":
    asyncio.run(main())