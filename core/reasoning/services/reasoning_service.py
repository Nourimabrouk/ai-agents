"""
Reasoning Service
Orchestrates different reasoning engines and provides unified reasoning interface
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from enum import Enum

from ...shared import (
    IReasoningEngine, ReasoningMode, ReasoningRequest, ReasoningResult,
    AgentId, DomainEvent, IEventBus, get_service
)

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """Strategies for combining reasoning engines"""
    SINGLE = "single"  # Use single best engine
    PARALLEL = "parallel"  # Run multiple engines in parallel
    SEQUENTIAL = "sequential"  # Chain engines sequentially
    ENSEMBLE = "ensemble"  # Combine results from multiple engines


class ReasoningOrchestrationService:
    """
    Service for orchestrating different reasoning engines
    Provides unified interface for complex reasoning tasks
    """
    
    def __init__(self):
        self._engines: Dict[ReasoningMode, IReasoningEngine] = {}
        self._engine_performance: Dict[ReasoningMode, Dict[str, float]] = {}
        self._request_history: List[ReasoningRequest] = []
        self._result_cache: Dict[str, ReasoningResult] = {}
        self._event_bus: Optional[IEventBus] = None
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the reasoning service"""
        try:
            self._event_bus = get_service(IEventBus)
        except ValueError:
            logger.warning("EventBus not available, running without events")
        
        # Initialize performance tracking
        for mode in ReasoningMode:
            self._engine_performance[mode] = {
                "total_requests": 0,
                "successful_requests": 0,
                "average_confidence": 0.0,
                "average_response_time": 0.0
            }
    
    async def register_engine(self, mode: ReasoningMode, engine: IReasoningEngine) -> None:
        """Register a reasoning engine for specific mode"""
        async with self._lock:
            self._engines[mode] = engine
        
        logger.info(f"Registered reasoning engine for mode: {mode.value}")
        
        if self._event_bus:
            await self._event_bus.publish(DomainEvent(
                event_id=f"engine_registered_{mode.value}",
                event_type="reasoning.engine_registered",
                source=AgentId("system", "reasoning_service"),
                timestamp=datetime.utcnow(),
                data={"reasoning_mode": mode.value}
            ))
    
    async def reason(self, request: ReasoningRequest, 
                    strategy: ReasoningStrategy = ReasoningStrategy.SINGLE) -> ReasoningResult:
        """
        Execute reasoning request using specified strategy
        """
        start_time = datetime.utcnow()
        
        try:
            # Store request in history
            async with self._lock:
                self._request_history.append(request)
                if len(self._request_history) > 1000:  # Keep last 1000 requests
                    self._request_history.pop(0)
            
            # Check cache first
            cache_key = self._get_cache_key(request)
            if cache_key in self._result_cache:
                logger.debug(f"Returning cached result for reasoning request")
                return self._result_cache[cache_key]
            
            # Execute reasoning based on strategy
            if strategy == ReasoningStrategy.SINGLE:
                result = await self._reason_single(request)
            elif strategy == ReasoningStrategy.PARALLEL:
                result = await self._reason_parallel(request)
            elif strategy == ReasoningStrategy.SEQUENTIAL:
                result = await self._reason_sequential(request)
            elif strategy == ReasoningStrategy.ENSEMBLE:
                result = await self._reason_ensemble(request)
            else:
                raise ValueError(f"Unknown reasoning strategy: {strategy}")
            
            # Cache result
            self._result_cache[cache_key] = result
            
            # Update performance metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_performance_metrics(request.mode, result, execution_time)
            
            # Publish event
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id=f"reasoning_completed_{start_time.timestamp()}",
                    event_type="reasoning.completed",
                    source=AgentId("system", "reasoning_service"),
                    timestamp=datetime.utcnow(),
                    data={
                        "mode": request.mode.value,
                        "strategy": strategy.value,
                        "confidence": result.confidence,
                        "execution_time": execution_time,
                        "success": True
                    }
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reasoning service: {e}")
            
            if self._event_bus:
                await self._event_bus.publish(DomainEvent(
                    event_id=f"reasoning_failed_{start_time.timestamp()}",
                    event_type="reasoning.failed",
                    source=AgentId("system", "reasoning_service"),
                    timestamp=datetime.utcnow(),
                    data={
                        "mode": request.mode.value,
                        "strategy": strategy.value,
                        "error": str(e)
                    }
                ))
            
            raise
    
    async def _reason_single(self, request: ReasoningRequest) -> ReasoningResult:
        """Use single best engine for the request"""
        if request.mode not in self._engines:
            # Fall back to comprehensive mode if specific mode not available
            if ReasoningMode.COMPREHENSIVE in self._engines:
                engine = self._engines[ReasoningMode.COMPREHENSIVE]
            else:
                raise ValueError(f"No engine available for mode {request.mode}")
        else:
            engine = self._engines[request.mode]
        
        return await engine.reason(request)
    
    async def _reason_parallel(self, request: ReasoningRequest) -> ReasoningResult:
        """Run multiple engines in parallel and select best result"""
        # Find compatible engines
        compatible_engines = self._find_compatible_engines(request)
        
        if not compatible_engines:
            return await self._reason_single(request)
        
        # Run engines in parallel
        tasks = [engine.reason(request) for engine in compatible_engines]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Select best result (highest confidence)
        best_result = None
        for result in results:
            if isinstance(result, ReasoningResult):
                if best_result is None or result.confidence > best_result.confidence:
                    best_result = result
        
        if best_result is None:
            raise Exception("All reasoning engines failed")
        
        return best_result
    
    async def _reason_sequential(self, request: ReasoningRequest) -> ReasoningResult:
        """Chain engines sequentially for complex reasoning"""
        current_request = request
        final_result = None
        
        # Define reasoning chain based on mode
        chain = self._get_reasoning_chain(request.mode)
        
        for mode in chain:
            if mode in self._engines:
                engine = self._engines[mode]
                result = await engine.reason(current_request)
                
                if final_result is None:
                    final_result = result
                else:
                    # Combine results
                    final_result = self._combine_results(final_result, result)
                
                # Use result as input for next engine
                current_request = ReasoningRequest(
                    mode=mode,
                    problem=result.solution,
                    context={**current_request.context, "previous_result": result},
                    constraints=current_request.constraints
                )
        
        return final_result or ReasoningResult(
            solution="No reasoning engines available",
            confidence=0.0,
            reasoning_chain=["error"]
        )
    
    async def _reason_ensemble(self, request: ReasoningRequest) -> ReasoningResult:
        """Combine results from multiple engines using ensemble methods"""
        compatible_engines = self._find_compatible_engines(request)
        
        if not compatible_engines:
            return await self._reason_single(request)
        
        # Run all compatible engines
        tasks = [engine.reason(request) for engine in compatible_engines]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        valid_results = [r for r in results if isinstance(r, ReasoningResult)]
        
        if not valid_results:
            raise Exception("All reasoning engines failed")
        
        # Ensemble combination
        return self._ensemble_combine(valid_results)
    
    def _find_compatible_engines(self, request: ReasoningRequest) -> List[IReasoningEngine]:
        """Find engines compatible with the request"""
        compatible = []
        
        # Direct mode match
        if request.mode in self._engines:
            compatible.append(self._engines[request.mode])
        
        # Add comprehensive engine as fallback
        if ReasoningMode.COMPREHENSIVE in self._engines and request.mode != ReasoningMode.COMPREHENSIVE:
            compatible.append(self._engines[ReasoningMode.COMPREHENSIVE])
        
        # Add adaptive engine for complex requests
        if (ReasoningMode.ADAPTIVE in self._engines and 
            request.mode != ReasoningMode.ADAPTIVE and
            len(request.constraints) > 0):
            compatible.append(self._engines[ReasoningMode.ADAPTIVE])
        
        return compatible
    
    def _get_reasoning_chain(self, mode: ReasoningMode) -> List[ReasoningMode]:
        """Get reasoning chain for sequential strategy"""
        chains = {
            ReasoningMode.COMPREHENSIVE: [
                ReasoningMode.ANALYTICAL,
                ReasoningMode.CAUSAL,
                ReasoningMode.PREDICTIVE
            ],
            ReasoningMode.PREDICTIVE: [
                ReasoningMode.TEMPORAL,
                ReasoningMode.CAUSAL,
                ReasoningMode.PREDICTIVE
            ],
            ReasoningMode.CREATIVE: [
                ReasoningMode.CREATIVE,
                ReasoningMode.ANALYTICAL
            ]
        }
        
        return chains.get(mode, [mode])
    
    def _combine_results(self, result1: ReasoningResult, result2: ReasoningResult) -> ReasoningResult:
        """Combine two reasoning results"""
        # Weighted combination based on confidence
        w1 = result1.confidence
        w2 = result2.confidence
        total_weight = w1 + w2
        
        if total_weight == 0:
            combined_confidence = 0.5
        else:
            combined_confidence = (w1 + w2) / 2
        
        return ReasoningResult(
            solution=f"{result1.solution}\n\nAdditional insight: {result2.solution}",
            confidence=combined_confidence,
            reasoning_chain=result1.reasoning_chain + result2.reasoning_chain,
            alternative_solutions=result1.alternative_solutions + result2.alternative_solutions
        )
    
    def _ensemble_combine(self, results: List[ReasoningResult]) -> ReasoningResult:
        """Combine multiple results using ensemble methods"""
        if not results:
            raise ValueError("No results to combine")
        
        if len(results) == 1:
            return results[0]
        
        # Weighted voting based on confidence
        total_weight = sum(r.confidence for r in results)
        
        if total_weight == 0:
            # Equal weighting if no confidence information
            weights = [1.0 / len(results) for _ in results]
        else:
            weights = [r.confidence / total_weight for r in results]
        
        # Select solution with highest weighted confidence
        best_idx = max(range(len(results)), key=lambda i: results[i].confidence)
        best_result = results[best_idx]
        
        # Combine reasoning chains and alternatives
        all_chains = []
        all_alternatives = []
        
        for result in results:
            all_chains.extend(result.reasoning_chain)
            all_alternatives.extend(result.alternative_solutions)
        
        # Add other solutions as alternatives
        alternatives = [r.solution for r in results if r != best_result]
        all_alternatives.extend(alternatives)
        
        return ReasoningResult(
            solution=best_result.solution,
            confidence=sum(w * r.confidence for w, r in zip(weights, results)),
            reasoning_chain=list(set(all_chains)),  # Remove duplicates
            alternative_solutions=list(set(all_alternatives))  # Remove duplicates
        )
    
    def _get_cache_key(self, request: ReasoningRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_data = f"{request.mode.value}:{request.problem}:{str(sorted(request.context.items()))}:{':'.join(request.constraints)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _update_performance_metrics(self, mode: ReasoningMode, 
                                        result: ReasoningResult, execution_time: float) -> None:
        """Update performance metrics for reasoning mode"""
        async with self._lock:
            metrics = self._engine_performance[mode]
            
            metrics["total_requests"] += 1
            metrics["successful_requests"] += 1
            
            # Update running averages
            n = metrics["total_requests"]
            metrics["average_confidence"] = (
                (metrics["average_confidence"] * (n - 1) + result.confidence) / n
            )
            metrics["average_response_time"] = (
                (metrics["average_response_time"] * (n - 1) + execution_time) / n
            )
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all reasoning modes"""
        async with self._lock:
            return {
                "engines": {mode.value: metrics.copy() 
                           for mode, metrics in self._engine_performance.items()},
                "cache_size": len(self._result_cache),
                "request_history_size": len(self._request_history)
            }
    
    async def clear_cache(self) -> None:
        """Clear reasoning result cache"""
        async with self._lock:
            self._result_cache.clear()
        logger.info("Reasoning cache cleared")
    
    async def get_available_engines(self) -> List[ReasoningMode]:
        """Get list of available reasoning engines"""
        async with self._lock:
            return list(self._engines.keys())