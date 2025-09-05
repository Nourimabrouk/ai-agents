"""
CPU Performance Profiler for Phase 7 - Autonomous Intelligence Ecosystem
Advanced profiling with hotspot detection, call graph analysis, and performance regression detection
"""

import cProfile
import pstats
import io
import time
import asyncio
import threading
import psutil
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
import json
import logging
from contextlib import contextmanager
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """CPU profiling result with comprehensive metrics"""
    session_id: str
    function_name: str
    total_time: float
    cumulative_time: float
    call_count: int
    time_per_call: float
    percentage_total: float
    filename: str
    line_number: int
    hotspot_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'function_name': self.function_name,
            'total_time': self.total_time,
            'cumulative_time': self.cumulative_time,
            'call_count': self.call_count,
            'time_per_call': self.time_per_call,
            'percentage_total': self.percentage_total,
            'filename': self.filename,
            'line_number': self.line_number,
            'hotspot_score': self.hotspot_score
        }


@dataclass
class PerformanceRegression:
    """Performance regression detection"""
    function_name: str
    baseline_time: float
    current_time: float
    regression_percentage: float
    severity: str  # 'minor', 'moderate', 'severe', 'critical'
    detected_at: datetime
    call_count_change: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function_name': self.function_name,
            'baseline_time': self.baseline_time,
            'current_time': self.current_time,
            'regression_percentage': self.regression_percentage,
            'severity': self.severity,
            'detected_at': self.detected_at.isoformat(),
            'call_count_change': self.call_count_change
        }


class CpuProfiler:
    """
    Advanced CPU profiler with hotspot detection and regression monitoring
    Optimized for Phase 7 performance analysis
    """
    
    def __init__(self, 
                 profile_dir: str = "performance_profiles",
                 regression_threshold: float = 10.0,  # 10% performance degradation threshold
                 hotspot_threshold: float = 1.0):     # 1% of total time threshold for hotspots
        
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)
        
        self.regression_threshold = regression_threshold
        self.hotspot_threshold = hotspot_threshold
        
        # Profiling data storage
        self.active_profiles: Dict[str, cProfile.Profile] = {}
        self.profile_results: Dict[str, List[ProfileResult]] = defaultdict(list)
        self.baseline_metrics: Dict[str, ProfileResult] = {}
        self.regressions: List[PerformanceRegression] = []
        
        # System metrics tracking
        self.cpu_samples: List[Tuple[datetime, float]] = []
        self.memory_samples: List[Tuple[datetime, float]] = []
        
        # Background monitoring
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        logger.info(f"Initialized CPU Profiler with regression threshold: {regression_threshold}%")
    
    @contextmanager
    def profile_context(self, session_id: str):
        """Context manager for profiling code blocks"""
        profiler = cProfile.Profile()
        
        try:
            profiler.enable()
            start_time = time.perf_counter()
            yield profiler
        finally:
            end_time = time.perf_counter()
            profiler.disable()
            
            # Store results
            self.active_profiles[session_id] = profiler
            
            # Analyze and store results
            results = self._analyze_profile(profiler, session_id, end_time - start_time)
            self.profile_results[session_id].extend(results)
            
            logger.info(f"Profiling session '{session_id}' completed in {end_time - start_time:.4f}s")
    
    def profile_function(self, func: Callable, session_id: Optional[str] = None) -> Callable:
        """Decorator for profiling individual functions"""
        if session_id is None:
            session_id = f"{func.__name__}_{int(time.time())}"
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with self.profile_context(session_id):
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with self.profile_context(session_id):
                return await func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    async def profile_async_task(self, coro, session_id: str) -> Any:
        """Profile an async coroutine"""
        with self.profile_context(session_id):
            return await coro
    
    def start_continuous_monitoring(self, interval: float = 1.0):
        """Start continuous CPU and memory monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return {}
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system_metrics,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started continuous monitoring with {interval}s interval")
    
    def stop_continuous_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Stopped continuous monitoring")
    
    def _monitor_system_metrics(self, interval: float):
        """Monitor system CPU and memory usage"""
        while self.monitoring_active:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.cpu_samples.append((datetime.now(), cpu_percent))
                
                # Memory usage
                memory_info = psutil.virtual_memory()
                self.memory_samples.append((datetime.now(), memory_info.percent))
                
                # Keep only last 1000 samples to prevent memory growth
                if len(self.cpu_samples) > 1000:
                    self.cpu_samples = self.cpu_samples[-1000:]
                if len(self.memory_samples) > 1000:
                    self.memory_samples = self.memory_samples[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                time.sleep(interval)
    
    def _analyze_profile(self, profiler: cProfile.Profile, session_id: str, total_time: float) -> List[ProfileResult]:
        """Analyze cProfile results and extract performance metrics"""
        # Create string buffer for pstats output
        buffer = io.StringIO()
        stats = pstats.Stats(profiler, stream=buffer)
        stats.sort_stats('cumulative')
        
        results = []
        
        # Extract top functions by cumulative time
        for func_info, (call_count, reccall_count, total_time_func, cum_time) in stats.stats.items():
            filename, line_number, func_name = func_info
            
            if total_time > 0:
                percentage = (cum_time / total_time) * 100
                time_per_call = cum_time / call_count if call_count > 0 else 0
                
                # Calculate hotspot score
                hotspot_score = self._calculate_hotspot_score(
                    percentage, call_count, time_per_call, cum_time
                )
                
                result = ProfileResult(
                    session_id=session_id,
                    function_name=func_name,
                    total_time=total_time_func,
                    cumulative_time=cum_time,
                    call_count=call_count,
                    time_per_call=time_per_call,
                    percentage_total=percentage,
                    filename=filename,
                    line_number=line_number,
                    hotspot_score=hotspot_score
                )
                
                results.append(result)
        
        # Sort by hotspot score and return top performers
        results.sort(key=lambda x: x.hotspot_score, reverse=True)
        return results[:50]  # Top 50 functions
    
    def _calculate_hotspot_score(self, percentage: float, call_count: int, 
                               time_per_call: float, cum_time: float) -> float:
        """Calculate hotspot score for function prioritization"""
        # Weighted combination of factors
        score = (
            percentage * 0.4 +           # Time percentage weight
            min(100, call_count / 10) * 0.2 +  # Call frequency weight (capped)
            min(100, time_per_call * 1000) * 0.2 +  # Time per call weight
            min(100, cum_time * 100) * 0.2      # Absolute time weight
        )
        return score
    
    def identify_hotspots(self, session_id: str, top_n: int = 10) -> List[ProfileResult]:
        """Identify performance hotspots from profiling results"""
        results = self.profile_results.get(session_id, [])
        if not results:
            return []
        
        # Filter hotspots above threshold and return top N
        hotspots = [
            result for result in results 
            if result.percentage_total >= self.hotspot_threshold
        ]
        
        hotspots.sort(key=lambda x: x.hotspot_score, reverse=True)
        return hotspots[:top_n]
    
    def detect_regressions(self, session_id: str) -> List[PerformanceRegression]:
        """Detect performance regressions compared to baseline"""
        current_results = self.profile_results.get(session_id, [])
        regressions = []
        
        for result in current_results:
            baseline_key = f"{result.filename}:{result.function_name}"
            baseline = self.baseline_metrics.get(baseline_key)
            
            if baseline:
                # Calculate regression percentage
                if baseline.cumulative_time > 0:
                    regression_pct = (
                        (result.cumulative_time - baseline.cumulative_time) / 
                        baseline.cumulative_time * 100
                    )
                    
                    if regression_pct > self.regression_threshold:
                        # Determine severity
                        if regression_pct > 50:
                            severity = "critical"
                        elif regression_pct > 30:
                            severity = "severe"
                        elif regression_pct > 20:
                            severity = "moderate"
                        else:
                            severity = "minor"
                        
                        # Calculate call count change
                        call_count_change = (
                            (result.call_count - baseline.call_count) / 
                            max(1, baseline.call_count) * 100
                        )
                        
                        regression = PerformanceRegression(
                            function_name=result.function_name,
                            baseline_time=baseline.cumulative_time,
                            current_time=result.cumulative_time,
                            regression_percentage=regression_pct,
                            severity=severity,
                            detected_at=datetime.now(),
                            call_count_change=call_count_change
                        )
                        
                        regressions.append(regression)
        
        # Store regressions
        self.regressions.extend(regressions)
        
        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions")
        
        return regressions
    
    def set_baseline(self, session_id: str):
        """Set current profiling results as baseline for regression detection"""
        results = self.profile_results.get(session_id, [])
        
        for result in results:
            baseline_key = f"{result.filename}:{result.function_name}"
            self.baseline_metrics[baseline_key] = result
        
        logger.info(f"Set baseline with {len(results)} function metrics from session '{session_id}'")
    
    def generate_report(self, session_id: str) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        results = self.profile_results.get(session_id, [])
        hotspots = self.identify_hotspots(session_id)
        regressions = self.detect_regressions(session_id)
        
        # Calculate summary statistics
        if results:
            total_functions = len(results)
            avg_time_per_call = sum(r.time_per_call for r in results) / total_functions
            total_calls = sum(r.call_count for r in results)
            total_time = sum(r.total_time for r in results)
        else:
            total_functions = avg_time_per_call = total_calls = total_time = 0
        
        # System metrics summary
        cpu_avg = 0.0
        memory_avg = 0.0
        if self.cpu_samples:
            cpu_avg = sum(sample[1] for sample in self.cpu_samples[-60:]) / min(60, len(self.cpu_samples))
        if self.memory_samples:
            memory_avg = sum(sample[1] for sample in self.memory_samples[-60:]) / min(60, len(self.memory_samples))
        
        report = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_functions_profiled': total_functions,
                'total_function_calls': total_calls,
                'total_execution_time': total_time,
                'average_time_per_call': avg_time_per_call,
                'hotspots_detected': len(hotspots),
                'regressions_detected': len(regressions)
            },
            'system_metrics': {
                'avg_cpu_usage_percent': cpu_avg,
                'avg_memory_usage_percent': memory_avg,
                'samples_collected': len(self.cpu_samples)
            },
            'hotspots': [hotspot.to_dict() for hotspot in hotspots],
            'regressions': [regression.to_dict() for regression in regressions],
            'top_functions': [result.to_dict() for result in results[:20]]
        }
        
        return report
    
    def save_profile(self, session_id: str, filename: Optional[str] = None):
        """Save profiling results to file"""
        if filename is None:
            filename = f"profile_{session_id}_{int(time.time())}.prof"
        
        profile_path = self.profile_dir / filename
        
        if session_id in self.active_profiles:
            # Save binary profile for external analysis
            self.active_profiles[session_id].dump_stats(str(profile_path))
            
            # Save JSON report
            report = self.generate_report(session_id)
            json_path = profile_path.with_suffix('.json')
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved profile data to {profile_path} and report to {json_path}")
        else:
            logger.warning(f"No profile data found for session '{session_id}'")
    
    def load_baseline_from_file(self, filepath: str):
        """Load baseline metrics from saved profile report"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            if 'top_functions' in data:
                for func_data in data['top_functions']:
                    result = ProfileResult(**func_data)
                    baseline_key = f"{result.filename}:{result.function_name}"
                    self.baseline_metrics[baseline_key] = result
                
                logger.info(f"Loaded {len(data['top_functions'])} baseline metrics from {filepath}")
            else:
                logger.warning(f"No function data found in {filepath}")
                
        except Exception as e:
            logger.error(f"Error loading baseline from {filepath}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all sessions"""
        total_sessions = len(self.profile_results)
        total_functions = sum(len(results) for results in self.profile_results.values())
        total_regressions = len(self.regressions)
        
        # Find most problematic functions
        all_results = []
        for results in self.profile_results.values():
            all_results.extend(results)
        
        # Group by function and calculate averages
        function_stats = defaultdict(list)
        for result in all_results:
            key = f"{result.filename}:{result.function_name}"
            function_stats[key].append(result)
        
        problematic_functions = []
        for func_key, results in function_stats.items():
            avg_time = sum(r.cumulative_time for r in results) / len(results)
            avg_calls = sum(r.call_count for r in results) / len(results)
            avg_hotspot_score = sum(r.hotspot_score for r in results) / len(results)
            
            problematic_functions.append({
                'function': func_key,
                'avg_cumulative_time': avg_time,
                'avg_call_count': avg_calls,
                'avg_hotspot_score': avg_hotspot_score,
                'sessions_count': len(results)
            })
        
        # Sort by average hotspot score
        problematic_functions.sort(key=lambda x: x['avg_hotspot_score'], reverse=True)
        
        return {
            'total_profiling_sessions': total_sessions,
            'total_functions_analyzed': total_functions,
            'total_regressions_detected': total_regressions,
            'baseline_metrics_count': len(self.baseline_metrics),
            'system_monitoring_active': self.monitoring_active,
            'cpu_samples_collected': len(self.cpu_samples),
            'memory_samples_collected': len(self.memory_samples),
            'most_problematic_functions': problematic_functions[:10]
        }
    
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old profiling data"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        # Clean up CPU/memory samples
        self.cpu_samples = [
            sample for sample in self.cpu_samples 
            if sample[0] > cutoff_date
        ]
        self.memory_samples = [
            sample for sample in self.memory_samples 
            if sample[0] > cutoff_date
        ]
        
        # Clean up old regressions
        self.regressions = [
            regression for regression in self.regressions
            if regression.detected_at > cutoff_date
        ]
        
        logger.info(f"Cleaned up profiling data older than {days_to_keep} days")


# Global profiler instance
profiler = CpuProfiler()


# Convenience decorators
def profile(session_id: Optional[str] = None):
    """Convenience decorator for profiling functions"""
    return profiler.profile_function(session_id=session_id)


@contextmanager
def profile_block(session_id: str):
    """Convenience context manager for profiling code blocks"""
    with profiler.profile_context(session_id):
        yield
