"""
Predictive Coordinator for Multi-Agent Systems
Coordinates agents using predictive modeling and optimization
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from enum import Enum

from templates.base_agent import BaseAgent, Action, Observation
from .temporal_engine import TemporalEvent, TimeHorizon
from .time_series_processor import TimeSeriesProcessor, TimeSeriesForecast
from .causal_inference import CausalInferenceEngine
from utils.observability.logging import get_logger

logger = get_logger(__name__)


class CoordinationStrategy(Enum):
    """Different coordination strategies"""
    REACTIVE = "reactive"  # React to current state
    PREDICTIVE = "predictive"  # Predict and optimize
    ADAPTIVE = "adaptive"  # Adapt based on performance
    COLLABORATIVE = "collaborative"  # Multi-agent collaboration


@dataclass
class AgentPrediction:
    """Prediction about an agent's future state"""
    agent_id: str
    predicted_state: str
    confidence: float
    time_horizon: timedelta
    factors: List[str]


@dataclass
class CoordinationPlan:
    """Plan for coordinating multiple agents"""
    plan_id: str
    strategy: CoordinationStrategy
    agent_assignments: Dict[str, List[str]]  # agent_id -> tasks
    predicted_outcomes: List[AgentPrediction]
    execution_order: List[str]
    estimated_completion_time: timedelta
    confidence_score: float


class PredictiveCoordinator:
    """
    Coordinates multiple agents using predictive modeling
    Optimizes task allocation and execution scheduling
    """
    
    def __init__(self, name: str = "predictive_coordinator"):
        self.name = name
        self.agents: Dict[str, BaseAgent] = {}
        self.time_series_processor = TimeSeriesProcessor()
        self.causal_engine = CausalInferenceEngine()
        
        # Coordination state
        self.active_plans: Dict[str, CoordinationPlan] = {}
        self.agent_performance_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self.coordination_metrics: Dict[str, float] = {}
        
        logger.info(f"Initialized predictive coordinator: {name}")
    
    async def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent for coordination"""
        self.agents[agent.name] = agent
        self.agent_performance_history[agent.name] = []
        logger.info(f"Registered agent: {agent.name}")
    
    async def create_coordination_plan(self, tasks: List[str], 
                                     strategy: CoordinationStrategy = CoordinationStrategy.PREDICTIVE,
                                     deadline: Optional[datetime] = None) -> CoordinationPlan:
        """Create a coordination plan for given tasks"""
        try:
            plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Predict agent performance for different task assignments
            agent_predictions = await self._predict_agent_performance(tasks)
            
            # Optimize task allocation
            optimal_assignments = await self._optimize_task_allocation(tasks, agent_predictions)
            
            # Determine execution order
            execution_order = await self._determine_execution_order(optimal_assignments)
            
            # Estimate completion time
            completion_time = await self._estimate_completion_time(optimal_assignments)
            
            # Calculate overall confidence
            confidence_score = np.mean([pred.confidence for pred in agent_predictions]) if agent_predictions else 0.5
            
            plan = CoordinationPlan(
                plan_id=plan_id,
                strategy=strategy,
                agent_assignments=optimal_assignments,
                predicted_outcomes=agent_predictions,
                execution_order=execution_order,
                estimated_completion_time=completion_time,
                confidence_score=confidence_score
            )
            
            self.active_plans[plan_id] = plan
            logger.info(f"Created coordination plan: {plan_id}")
            
            return plan
            
        except Exception as e:
            logger.error(f"Error creating coordination plan: {e}")
            # Return fallback plan
            return CoordinationPlan(
                plan_id="fallback_plan",
                strategy=CoordinationStrategy.REACTIVE,
                agent_assignments={},
                predicted_outcomes=[],
                execution_order=[],
                estimated_completion_time=timedelta(hours=1),
                confidence_score=0.0
            )
    
    async def _predict_agent_performance(self, tasks: List[str]) -> List[AgentPrediction]:
        """Predict how each agent will perform on given tasks"""
        predictions = []
        
        try:
            for agent_name, agent in self.agents.items():
                for task in tasks:
                    # Get historical performance for similar tasks
                    performance_history = self.agent_performance_history.get(agent_name, [])
                    
                    if len(performance_history) < 3:
                        # Not enough history, use default prediction
                        confidence = 0.5
                        predicted_state = "moderate_performance"
                    else:
                        # Calculate trend in performance
                        recent_performance = [perf for _, perf in performance_history[-10:]]
                        avg_performance = np.mean(recent_performance)
                        
                        if avg_performance > 0.8:
                            predicted_state = "high_performance"
                            confidence = 0.8
                        elif avg_performance > 0.6:
                            predicted_state = "moderate_performance"
                            confidence = 0.7
                        else:
                            predicted_state = "low_performance"
                            confidence = 0.6
                    
                    # Adjust based on task complexity (simple heuristic)
                    task_complexity = len(task.split()) / 10.0  # Simple complexity measure
                    confidence *= max(0.5, 1.0 - task_complexity)
                    
                    prediction = AgentPrediction(
                        agent_id=agent_name,
                        predicted_state=predicted_state,
                        confidence=confidence,
                        time_horizon=timedelta(minutes=30),  # Default horizon
                        factors=["historical_performance", "task_complexity"]
                    )
                    
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting agent performance: {e}")
            return []
    
    async def _optimize_task_allocation(self, tasks: List[str], 
                                       predictions: List[AgentPrediction]) -> Dict[str, List[str]]:
        """Optimize allocation of tasks to agents"""
        try:
            allocation = {agent_name: [] for agent_name in self.agents.keys()}
            
            if not predictions:
                # Fallback: distribute tasks evenly
                for i, task in enumerate(tasks):
                    agent_names = list(self.agents.keys())
                    if agent_names:
                        chosen_agent = agent_names[i % len(agent_names)]
                        allocation[chosen_agent].append(task)
                return allocation
            
            # Group predictions by agent
            agent_predictions = {}
            for pred in predictions:
                if pred.agent_id not in agent_predictions:
                    agent_predictions[pred.agent_id] = []
                agent_predictions[pred.agent_id].append(pred)
            
            # Greedy allocation: assign each task to best predicted agent
            for task in tasks:
                best_agent = None
                best_score = -1.0
                
                for agent_name in self.agents.keys():
                    if agent_name in agent_predictions:
                        # Find prediction for this agent/task combination
                        relevant_predictions = agent_predictions[agent_name]
                        if relevant_predictions:
                            avg_confidence = np.mean([p.confidence for p in relevant_predictions])
                            # Prefer agents with fewer current assignments (load balancing)
                            load_factor = 1.0 / (len(allocation[agent_name]) + 1)
                            score = avg_confidence * load_factor
                            
                            if score > best_score:
                                best_score = score
                                best_agent = agent_name
                
                if best_agent:
                    allocation[best_agent].append(task)
                else:
                    # Fallback to first available agent
                    if allocation:
                        first_agent = list(allocation.keys())[0]
                        allocation[first_agent].append(task)
            
            return allocation
            
        except Exception as e:
            logger.error(f"Error optimizing task allocation: {e}")
            # Fallback allocation
            allocation = {agent_name: [] for agent_name in self.agents.keys()}
            for i, task in enumerate(tasks):
                agent_names = list(self.agents.keys())
                if agent_names:
                    chosen_agent = agent_names[i % len(agent_names)]
                    allocation[chosen_agent].append(task)
            return allocation
    
    async def _determine_execution_order(self, assignments: Dict[str, List[str]]) -> List[str]:
        """Determine optimal execution order for agents"""
        try:
            # Simple heuristic: agents with more tasks execute first
            agent_workloads = [(agent, len(tasks)) for agent, tasks in assignments.items() if tasks]
            agent_workloads.sort(key=lambda x: x[1], reverse=True)
            
            return [agent for agent, _ in agent_workloads]
            
        except Exception as e:
            logger.error(f"Error determining execution order: {e}")
            return list(assignments.keys())
    
    async def _estimate_completion_time(self, assignments: Dict[str, List[str]]) -> timedelta:
        """Estimate total completion time for the plan"""
        try:
            max_time = timedelta(0)
            
            for agent_name, tasks in assignments.items():
                if not tasks:
                    continue
                
                # Estimate time based on task complexity and agent history
                estimated_time_per_task = timedelta(minutes=10)  # Default estimate
                
                # Adjust based on historical performance
                if agent_name in self.agent_performance_history:
                    history = self.agent_performance_history[agent_name]
                    if history:
                        avg_performance = np.mean([perf for _, perf in history[-5:]])
                        # Higher performance = faster completion
                        time_factor = 1.0 / max(avg_performance, 0.1)
                        estimated_time_per_task = timedelta(minutes=10 * time_factor)
                
                agent_total_time = estimated_time_per_task * len(tasks)
                max_time = max(max_time, agent_total_time)
            
            return max_time
            
        except Exception as e:
            logger.error(f"Error estimating completion time: {e}")
            return timedelta(hours=1)  # Default fallback
    
    async def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a coordination plan"""
        try:
            if plan_id not in self.active_plans:
                raise ValueError(f"Plan {plan_id} not found")
            
            plan = self.active_plans[plan_id]
            results = {}
            
            start_time = datetime.now()
            
            # Execute tasks according to the plan
            for agent_name in plan.execution_order:
                if agent_name not in self.agents:
                    logger.warning(f"Agent {agent_name} not found, skipping")
                    continue
                
                agent = self.agents[agent_name]
                tasks = plan.agent_assignments.get(agent_name, [])
                
                if not tasks:
                    continue
                
                # Execute tasks for this agent
                agent_results = []
                for task in tasks:
                    try:
                        # Create action for the task
                        action = Action(
                            action_type="execute_task",
                            parameters={"task": task},
                            tools_used=["coordination"],
                            expected_outcome=f"Complete task: {task}"
                        )
                        
                        # Execute the task (simplified - in reality would use agent's execute method)
                        result = f"Completed task: {task}"
                        agent_results.append({
                            "task": task,
                            "result": result,
                            "success": True,
                            "execution_time": datetime.now() - start_time
                        })
                        
                        # Record performance
                        await self._record_agent_performance(agent_name, 0.8)  # Default success score
                        
                    except Exception as e:
                        logger.error(f"Error executing task {task} for agent {agent_name}: {e}")
                        agent_results.append({
                            "task": task,
                            "result": f"Error: {str(e)}",
                            "success": False,
                            "execution_time": datetime.now() - start_time
                        })
                        
                        # Record poor performance
                        await self._record_agent_performance(agent_name, 0.2)
                
                results[agent_name] = agent_results
            
            execution_time = datetime.now() - start_time
            
            # Update plan with results
            plan_summary = {
                "plan_id": plan_id,
                "execution_time": execution_time,
                "results": results,
                "total_tasks": sum(len(tasks) for tasks in plan.agent_assignments.values()),
                "successful_tasks": sum(
                    len([r for r in agent_results if r.get("success", False)])
                    for agent_results in results.values()
                ),
                "completion_rate": 0.0
            }
            
            if plan_summary["total_tasks"] > 0:
                plan_summary["completion_rate"] = plan_summary["successful_tasks"] / plan_summary["total_tasks"]
            
            logger.info(f"Completed plan {plan_id}: {plan_summary['completion_rate']:.2%} success rate")
            
            return plan_summary
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return {"error": str(e), "plan_id": plan_id}
    
    async def _record_agent_performance(self, agent_name: str, performance_score: float) -> None:
        """Record agent performance for future predictions"""
        try:
            if agent_name not in self.agent_performance_history:
                self.agent_performance_history[agent_name] = []
            
            self.agent_performance_history[agent_name].append(
                (datetime.now(), performance_score)
            )
            
            # Keep only recent history (last 100 records)
            if len(self.agent_performance_history[agent_name]) > 100:
                self.agent_performance_history[agent_name] = self.agent_performance_history[agent_name][-100:]
            
            # Update time series for prediction
            await self.time_series_processor.add_data_point(
                datetime.now(), performance_score, f"agent_performance_{agent_name}"
            )
            
            # Update causal inference
            await self.causal_engine.add_observation(
                f"agent_{agent_name}_performance", datetime.now(), performance_score
            )
            
        except Exception as e:
            logger.error(f"Error recording agent performance: {e}")
    
    async def get_coordination_metrics(self) -> Dict[str, float]:
        """Get current coordination performance metrics"""
        try:
            metrics = {}
            
            # Calculate average performance per agent
            for agent_name, history in self.agent_performance_history.items():
                if history:
                    recent_performance = [perf for _, perf in history[-10:]]
                    metrics[f"{agent_name}_avg_performance"] = np.mean(recent_performance)
                    metrics[f"{agent_name}_performance_trend"] = await self.time_series_processor.get_trend(
                        f"agent_performance_{agent_name}"
                    )
            
            # Overall coordination metrics
            all_recent_performance = []
            for history in self.agent_performance_history.values():
                if history:
                    recent = [perf for _, perf in history[-5:]]
                    all_recent_performance.extend(recent)
            
            if all_recent_performance:
                metrics["overall_performance"] = np.mean(all_recent_performance)
                metrics["performance_std"] = np.std(all_recent_performance)
            
            metrics["active_plans"] = len(self.active_plans)
            metrics["registered_agents"] = len(self.agents)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating coordination metrics: {e}")
            return {}
    
    async def adapt_coordination_strategy(self) -> CoordinationStrategy:
        """Adapt coordination strategy based on performance"""
        try:
            metrics = await self.get_coordination_metrics()
            
            overall_performance = metrics.get("overall_performance", 0.5)
            performance_std = metrics.get("performance_std", 0.5)
            
            # Simple adaptation logic
            if overall_performance > 0.8 and performance_std < 0.2:
                # High consistent performance - use predictive strategy
                return CoordinationStrategy.PREDICTIVE
            elif overall_performance > 0.6:
                # Moderate performance - use adaptive strategy
                return CoordinationStrategy.ADAPTIVE
            elif performance_std > 0.4:
                # High variance - use collaborative strategy
                return CoordinationStrategy.COLLABORATIVE
            else:
                # Low performance - fall back to reactive
                return CoordinationStrategy.REACTIVE
                
        except Exception as e:
            logger.error(f"Error adapting coordination strategy: {e}")
            return CoordinationStrategy.REACTIVE