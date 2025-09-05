"""
Temporal Reasoning Engine for Phase 7 - Autonomous Intelligence Ecosystem
Advanced time-series analysis, temporal patterns, and predictive reasoning
"""

import asyncio
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Set, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict, deque
from enum import Enum
import json
import heapq
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from utils.observability.logging import get_logger
from utils.observability.metrics import global_metrics

logger = get_logger(__name__)


class TemporalPatternType(Enum):
    """Types of temporal patterns"""
    TREND = "trend"                    # Long-term directional changes
    CYCLE = "cycle"                   # Repeating patterns
    SEASONAL = "seasonal"             # Regular seasonal variations
    ANOMALY = "anomaly"               # Unusual temporal events
    PHASE_TRANSITION = "phase_transition"  # System state changes
    CAUSAL_LAG = "causal_lag"        # Delayed causal effects
    FEEDBACK_LOOP = "feedback_loop"   # Self-reinforcing patterns
    REGIME_SHIFT = "regime_shift"     # Fundamental system changes


class TemporalScale(Enum):
    """Temporal scales for analysis"""
    MICROSECOND = "microsecond"
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class PredictionHorizon(Enum):
    """Prediction time horizons"""
    SHORT_TERM = "short_term"      # Minutes to hours
    MEDIUM_TERM = "medium_term"    # Hours to days
    LONG_TERM = "long_term"        # Days to weeks
    STRATEGIC = "strategic"        # Weeks to months


@dataclass
class TemporalPattern:
    """Represents a discovered temporal pattern"""
    pattern_id: str
    pattern_type: TemporalPatternType
    variable: str
    start_time: datetime
    end_time: datetime
    
    # Pattern characteristics
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    periodicity: Optional[timedelta] = None
    amplitude: Optional[float] = None
    phase: Optional[float] = None
    
    # Statistical properties
    statistical_significance: float = 0.0
    effect_size: float = 0.0
    correlation_coefficient: float = 0.0
    
    # Pattern parameters
    parameters: Dict[str, Any] = field(default_factory=dict)
    supporting_data: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    temporal_scale: TemporalScale = TemporalScale.HOUR
    prediction_relevance: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type.value,
            'variable': self.variable,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'strength': self.strength,
            'confidence': self.confidence,
            'periodicity_seconds': self.periodicity.total_seconds() if self.periodicity else None,
            'amplitude': self.amplitude,
            'statistical_significance': self.statistical_significance,
            'temporal_scale': self.temporal_scale.value,
            'parameters': self.parameters
        }


@dataclass
class TimeSeriesAnalysis:
    """Result of time series analysis"""
    variable: str
    analysis_period: Tuple[datetime, datetime]
    
    # Basic statistics
    mean: float
    std: float
    min_value: float
    max_value: float
    
    # Temporal characteristics
    trend_slope: float
    seasonality_strength: float
    stationarity_p_value: float
    autocorrelation_lag1: float
    
    # Pattern detection results
    detected_patterns: List[TemporalPattern] = field(default_factory=list)
    anomalies: List[Tuple[datetime, float, str]] = field(default_factory=list)
    
    # Prediction metrics
    predictability_score: float = 0.0
    forecast_accuracy: Dict[str, float] = field(default_factory=dict)
    
    # Change points
    change_points: List[Tuple[datetime, str, float]] = field(default_factory=list)


@dataclass
class TemporalPrediction:
    """Temporal prediction result"""
    variable: str
    prediction_time: datetime
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence: float
    
    # Prediction metadata
    horizon: PredictionHorizon
    method: str
    contributing_patterns: List[str] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    
    # Quality metrics
    uncertainty: float = 0.0
    reliability_score: float = 0.0
    
    created_at: datetime = field(default_factory=datetime.now)


class TimeSeriesAnalyzer:
    """Advanced time series analysis with pattern detection"""
    
    def __init__(self, significance_threshold: float = 0.05):
        self.significance_threshold = significance_threshold
        self.analysis_cache = {}
        self.pattern_detectors = {
            TemporalPatternType.TREND: self._detect_trend_patterns,
            TemporalPatternType.CYCLE: self._detect_cycle_patterns,
            TemporalPatternType.SEASONAL: self._detect_seasonal_patterns,
            TemporalPatternType.ANOMALY: self._detect_anomaly_patterns,
            TemporalPatternType.PHASE_TRANSITION: self._detect_phase_transitions,
            TemporalPatternType.FEEDBACK_LOOP: self._detect_feedback_loops
        }
    
    async def analyze_time_series(self, 
                                 variable: str, 
                                 data: List[Tuple[datetime, float]],
                                 analysis_window: Optional[timedelta] = None) -> TimeSeriesAnalysis:
        """Comprehensive time series analysis"""
        
        if not data:
            raise ValueError("No data provided for analysis")
        
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x[0])
        
        # Apply analysis window if specified
        if analysis_window:
            cutoff_time = max(t for t, v in sorted_data) - analysis_window
            sorted_data = [(t, v) for t, v in sorted_data if t >= cutoff_time]
        
        if not sorted_data:
            raise ValueError("No data in analysis window")
        
        # Extract values and timestamps
        timestamps = [t for t, v in sorted_data]
        values = np.array([v for t, v in sorted_data])
        
        # Basic statistical analysis
        analysis_period = (timestamps[0], timestamps[-1])
        basic_stats = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min_value': np.min(values),
            'max_value': np.max(values)
        }
        
        # Temporal characteristics
        temporal_chars = await self._analyze_temporal_characteristics(timestamps, values)
        
        # Pattern detection
        detected_patterns = await self._detect_all_patterns(variable, sorted_data)
        
        # Anomaly detection
        anomalies = await self._detect_anomalies(sorted_data)
        
        # Change point detection
        change_points = await self._detect_change_points(timestamps, values)
        
        # Predictability assessment
        predictability_score = await self._assess_predictability(values)
        
        analysis = TimeSeriesAnalysis(
            variable=variable,
            analysis_period=analysis_period,
            detected_patterns=detected_patterns,
            anomalies=anomalies,
            change_points=change_points,
            predictability_score=predictability_score,
            **basic_stats,
            **temporal_chars
        )
        
        # Cache analysis result
        cache_key = f"{variable}_{analysis_period[0].isoformat()}_{analysis_period[1].isoformat()}"
        self.analysis_cache[cache_key] = analysis
        
        logger.info(f"Analyzed {variable}: {len(detected_patterns)} patterns, {len(anomalies)} anomalies")
        
        return analysis
    
    async def _analyze_temporal_characteristics(self, 
                                             timestamps: List[datetime], 
                                             values: np.ndarray) -> Dict[str, float]:
        """Analyze basic temporal characteristics"""
        
        # Trend analysis (linear regression slope)
        time_numeric = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        if len(values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            trend_slope = slope
        else:
            trend_slope = 0.0
        
        # Stationarity test (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_result = adfuller(values)
            stationarity_p_value = adf_result[1]
        except ImportError:
            # Fallback: simple variance-based stationarity test
            if len(values) > 10:
                mid_point = len(values) // 2
                first_half_var = np.var(values[:mid_point])
                second_half_var = np.var(values[mid_point:])
                variance_ratio = max(first_half_var, second_half_var) / max(min(first_half_var, second_half_var), 1e-10)
                stationarity_p_value = 1.0 / variance_ratio  # Approximate
            else:
                stationarity_p_value = 0.5
        
        # Autocorrelation at lag 1
        autocorr_lag1 = 0.0
        if len(values) > 2:
            autocorr_lag1 = np.corrcoef(values[:-1], values[1:])[0, 1]
            if np.isnan(autocorr_lag1):
                autocorr_lag1 = 0.0
        
        # Seasonality strength (simplified)
        seasonality_strength = await self._estimate_seasonality_strength(values)
        
        return {
            'trend_slope': trend_slope,
            'seasonality_strength': seasonality_strength,
            'stationarity_p_value': stationarity_p_value,
            'autocorrelation_lag1': autocorr_lag1
        }
    
    async def _estimate_seasonality_strength(self, values: np.ndarray) -> float:
        """Estimate seasonality strength"""
        
        if len(values) < 10:
            return 0.0
        
        # Use FFT to detect dominant frequencies
        try:
            # Detrend the data
            detrended = signal.detrend(values)
            
            # Apply FFT
            fft_values = np.fft.fft(detrended)
            frequencies = np.fft.fftfreq(len(values))
            
            # Find dominant frequency (excluding DC component)
            power_spectrum = np.abs(fft_values[1:len(values)//2])
            if len(power_spectrum) > 0:
                max_power = np.max(power_spectrum)
                total_power = np.sum(power_spectrum)
                seasonality_strength = max_power / max(total_power, 1e-10)
            else:
                seasonality_strength = 0.0
                
        except Exception:
            # Fallback: coefficient of variation
            seasonality_strength = np.std(values) / max(np.abs(np.mean(values)), 1e-10)
            seasonality_strength = min(1.0, seasonality_strength)
        
        return min(1.0, seasonality_strength)
    
    async def _detect_all_patterns(self, 
                                 variable: str, 
                                 data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect all types of temporal patterns"""
        
        all_patterns = []
        
        # Run pattern detection in parallel
        detection_tasks = []
        for pattern_type, detector in self.pattern_detectors.items():
            task = asyncio.create_task(detector(variable, data))
            detection_tasks.append((pattern_type, task))
        
        # Collect results
        for pattern_type, task in detection_tasks:
            try:
                patterns = await task
                all_patterns.extend(patterns)
            except Exception as e:
                logger.warning(f"Pattern detection failed for {pattern_type.value}: {e}")
        
        # Sort by strength and confidence
        all_patterns.sort(key=lambda p: (p.strength * p.confidence), reverse=True)
        
        return all_patterns
    
    async def _detect_trend_patterns(self, variable: str, data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect trend patterns in time series"""
        
        patterns = []
        
        if len(data) < 3:
            return patterns
        
        timestamps = [t for t, v in data]
        values = np.array([v for t, v in data])
        time_numeric = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        
        # Linear trend detection
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        
        if p_value < self.significance_threshold and abs(r_value) > 0.3:
            trend_type = "increasing" if slope > 0 else "decreasing"
            
            pattern = TemporalPattern(
                pattern_id=f"trend_{variable}_{timestamps[0].isoformat()}",
                pattern_type=TemporalPatternType.TREND,
                variable=variable,
                start_time=timestamps[0],
                end_time=timestamps[-1],
                strength=abs(r_value),
                confidence=1.0 - p_value,
                statistical_significance=p_value,
                effect_size=abs(slope),
                correlation_coefficient=r_value,
                parameters={
                    'slope': slope,
                    'intercept': intercept,
                    'trend_type': trend_type,
                    'r_squared': r_value**2
                },
                supporting_data=data.copy()
            )
            
            patterns.append(pattern)
        
        # Polynomial trend detection (quadratic)
        if len(data) > 5:
            try:
                poly_coeffs = np.polyfit(time_numeric, values, 2)
                poly_values = np.polyval(poly_coeffs, time_numeric)
                poly_r_squared = 1 - np.sum((values - poly_values)**2) / np.sum((values - np.mean(values))**2)
                
                if poly_r_squared > 0.6 and poly_r_squared > r_value**2 + 0.1:  # Significantly better than linear
                    pattern = TemporalPattern(
                        pattern_id=f"poly_trend_{variable}_{timestamps[0].isoformat()}",
                        pattern_type=TemporalPatternType.TREND,
                        variable=variable,
                        start_time=timestamps[0],
                        end_time=timestamps[-1],
                        strength=np.sqrt(poly_r_squared),
                        confidence=min(0.95, poly_r_squared),
                        parameters={
                            'coefficients': poly_coeffs.tolist(),
                            'trend_type': 'polynomial',
                            'r_squared': poly_r_squared
                        },
                        supporting_data=data.copy()
                    )
                    
                    patterns.append(pattern)
                    
            except Exception as e:
                logger.debug(f"Polynomial trend detection failed: {e}")
        
        return patterns
    
    async def _detect_cycle_patterns(self, variable: str, data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect cyclical patterns using FFT and autocorrelation"""
        
        patterns = []
        
        if len(data) < 10:
            return patterns
        
        timestamps = [t for t, v in data]
        values = np.array([v for t, v in data])
        
        # Detrend the data
        detrended_values = signal.detrend(values)
        
        # FFT-based cycle detection
        try:
            fft_values = np.fft.fft(detrended_values)
            frequencies = np.fft.fftfreq(len(values))
            power_spectrum = np.abs(fft_values)**2
            
            # Find significant peaks in power spectrum
            peak_indices, _ = signal.find_peaks(power_spectrum[1:len(values)//2], height=np.max(power_spectrum) * 0.1)
            peak_indices += 1  # Adjust for skipping DC component
            
            for peak_idx in peak_indices[:3]:  # Top 3 peaks
                if frequencies[peak_idx] > 0:
                    period_samples = 1.0 / frequencies[peak_idx]
                    
                    # Convert to time period
                    avg_sample_interval = (timestamps[-1] - timestamps[0]) / len(timestamps)
                    period_duration = avg_sample_interval * period_samples
                    
                    # Calculate cycle strength
                    cycle_strength = power_spectrum[peak_idx] / np.sum(power_spectrum[1:len(values)//2])
                    
                    if cycle_strength > 0.05:  # Minimum threshold
                        pattern = TemporalPattern(
                            pattern_id=f"cycle_{variable}_{peak_idx}_{timestamps[0].isoformat()}",
                            pattern_type=TemporalPatternType.CYCLE,
                            variable=variable,
                            start_time=timestamps[0],
                            end_time=timestamps[-1],
                            strength=min(1.0, cycle_strength * 10),  # Scale appropriately
                            confidence=min(0.9, cycle_strength * 20),
                            periodicity=period_duration,
                            amplitude=np.std(detrended_values),
                            parameters={
                                'frequency': frequencies[peak_idx],
                                'period_samples': period_samples,
                                'peak_power': power_spectrum[peak_idx],
                                'cycle_strength': cycle_strength
                            },
                            supporting_data=data.copy()
                        )
                        
                        patterns.append(pattern)
                        
        except Exception as e:
            logger.debug(f"FFT cycle detection failed: {e}")
        
        # Autocorrelation-based cycle detection
        try:
            autocorr = np.correlate(detrended_values, detrended_values, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.3)
            peaks += 1  # Adjust for skipping lag 0
            
            for peak in peaks[:2]:  # Top 2 peaks
                lag_duration = (timestamps[-1] - timestamps[0]) * peak / len(timestamps)
                
                pattern = TemporalPattern(
                    pattern_id=f"autocorr_cycle_{variable}_{peak}_{timestamps[0].isoformat()}",
                    pattern_type=TemporalPatternType.CYCLE,
                    variable=variable,
                    start_time=timestamps[0],
                    end_time=timestamps[-1],
                    strength=autocorr[peak],
                    confidence=min(0.9, autocorr[peak]),
                    periodicity=lag_duration,
                    parameters={
                        'lag': peak,
                        'autocorr_value': autocorr[peak],
                        'detection_method': 'autocorrelation'
                    },
                    supporting_data=data.copy()
                )
                
                patterns.append(pattern)
                
        except Exception as e:
            logger.debug(f"Autocorrelation cycle detection failed: {e}")
        
        return patterns
    
    async def _detect_seasonal_patterns(self, variable: str, data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect seasonal patterns based on calendar periods"""
        
        patterns = []
        
        if len(data) < 20:  # Need sufficient data for seasonal analysis
            return patterns
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(data, columns=['timestamp', 'value'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Test for different seasonal patterns
        seasonal_periods = [
            ('hourly', 24, 'hour'),
            ('daily', 7, 'day'),
            ('weekly', 4, 'week'),
            ('monthly', 12, 'month')
        ]
        
        for season_name, period_length, groupby_unit in seasonal_periods:
            try:
                # Group by seasonal component
                if groupby_unit == 'hour':
                    seasonal_groups = df.groupby(df.index.hour)['value']
                elif groupby_unit == 'day':
                    seasonal_groups = df.groupby(df.index.dayofweek)['value']
                elif groupby_unit == 'week':
                    seasonal_groups = df.groupby(df.index.isocalendar().week % 4)['value']
                elif groupby_unit == 'month':
                    seasonal_groups = df.groupby(df.index.month)['value']
                else:
                    continue
                
                # Calculate seasonal statistics
                seasonal_means = seasonal_groups.mean()
                seasonal_stds = seasonal_groups.std()
                
                if len(seasonal_means) >= 2:
                    # Test for significant seasonal variation
                    overall_mean = df['value'].mean()
                    seasonal_variation = np.var(seasonal_means)
                    total_variation = np.var(df['value'])
                    
                    if total_variation > 0:
                        seasonal_strength = seasonal_variation / total_variation
                        
                        if seasonal_strength > 0.1:  # Minimum seasonal strength
                            # Estimate periodicity
                            if groupby_unit == 'hour':
                                periodicity = timedelta(hours=24)
                            elif groupby_unit == 'day':
                                periodicity = timedelta(days=7)
                            elif groupby_unit == 'week':
                                periodicity = timedelta(days=28)  # 4 weeks
                            elif groupby_unit == 'month':
                                periodicity = timedelta(days=365)  # 1 year
                            else:
                                periodicity = timedelta(days=1)
                            
                            pattern = TemporalPattern(
                                pattern_id=f"seasonal_{season_name}_{variable}_{df.index[0].isoformat()}",
                                pattern_type=TemporalPatternType.SEASONAL,
                                variable=variable,
                                start_time=df.index[0].to_pydatetime(),
                                end_time=df.index[-1].to_pydatetime(),
                                strength=min(1.0, seasonal_strength * 5),
                                confidence=min(0.9, seasonal_strength * 3),
                                periodicity=periodicity,
                                amplitude=np.sqrt(seasonal_variation),
                                parameters={
                                    'season_type': season_name,
                                    'period_length': period_length,
                                    'seasonal_means': seasonal_means.to_dict(),
                                    'seasonal_strength': seasonal_strength,
                                    'variation_ratio': seasonal_variation / total_variation
                                },
                                supporting_data=data.copy(),
                                temporal_scale=TemporalScale.DAY if groupby_unit in ['hour', 'day'] else TemporalScale.WEEK
                            )
                            
                            patterns.append(pattern)
                            
            except Exception as e:
                logger.debug(f"Seasonal detection failed for {season_name}: {e}")
        
        return patterns
    
    async def _detect_anomaly_patterns(self, variable: str, data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect anomalous patterns in time series"""
        
        patterns = []
        
        if len(data) < 10:
            return patterns
        
        timestamps = [t for t, v in data]
        values = np.array([v for t, v in data])
        
        # Statistical anomaly detection (Z-score based)
        z_scores = np.abs(stats.zscore(values))
        anomaly_threshold = 3.0  # Standard 3-sigma rule
        
        anomalous_indices = np.where(z_scores > anomaly_threshold)[0]
        
        if len(anomalous_indices) > 0:
            # Group consecutive anomalies
            anomaly_groups = []
            current_group = [anomalous_indices[0]]
            
            for i in range(1, len(anomalous_indices)):
                if anomalous_indices[i] - anomalous_indices[i-1] <= 2:  # Allow small gaps
                    current_group.append(anomalous_indices[i])
                else:
                    anomaly_groups.append(current_group)
                    current_group = [anomalous_indices[i]]
            
            anomaly_groups.append(current_group)
            
            # Create patterns for each anomaly group
            for group_idx, group in enumerate(anomaly_groups):
                if len(group) >= 1:  # Minimum anomaly length
                    start_idx = max(0, group[0] - 1)
                    end_idx = min(len(data) - 1, group[-1] + 1)
                    
                    anomaly_strength = np.mean(z_scores[group]) / anomaly_threshold
                    
                    pattern = TemporalPattern(
                        pattern_id=f"anomaly_{variable}_{group_idx}_{timestamps[group[0]].isoformat()}",
                        pattern_type=TemporalPatternType.ANOMALY,
                        variable=variable,
                        start_time=timestamps[start_idx],
                        end_time=timestamps[end_idx],
                        strength=min(1.0, anomaly_strength),
                        confidence=min(0.95, anomaly_strength * 0.8),
                        parameters={
                            'anomaly_indices': group,
                            'max_z_score': np.max(z_scores[group]),
                            'mean_z_score': np.mean(z_scores[group]),
                            'anomaly_values': values[group].tolist(),
                            'detection_method': 'z_score'
                        },
                        supporting_data=data[start_idx:end_idx+1]
                    )
                    
                    patterns.append(pattern)
        
        # Isolation Forest for multivariate anomaly detection (if applicable)
        try:
            from sklearn.ensemble import IsolationForest
            
            # Create feature matrix with time-based features
            time_features = []
            for t, v in data:
                features = [
                    v,  # Value
                    t.hour,  # Hour of day
                    t.weekday(),  # Day of week
                    (t - timestamps[0]).total_seconds() / 3600  # Hours since start
                ]
                time_features.append(features)
            
            if len(time_features) >= 10:
                X = np.array(time_features)
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = iso_forest.fit_predict(X)
                
                # Find anomalies (-1 indicates anomaly)
                iso_anomalies = np.where(anomaly_labels == -1)[0]
                
                if len(iso_anomalies) > 0:
                    # Group consecutive anomalies
                    iso_groups = []
                    current_group = [iso_anomalies[0]]
                    
                    for i in range(1, len(iso_anomalies)):
                        if iso_anomalies[i] - iso_anomalies[i-1] <= 3:
                            current_group.append(iso_anomalies[i])
                        else:
                            iso_groups.append(current_group)
                            current_group = [iso_anomalies[i]]
                    
                    iso_groups.append(current_group)
                    
                    # Create patterns for isolation forest anomalies
                    for group_idx, group in enumerate(iso_groups):
                        if len(group) >= 2:  # Minimum group size
                            start_idx = group[0]
                            end_idx = group[-1]
                            
                            pattern = TemporalPattern(
                                pattern_id=f"iso_anomaly_{variable}_{group_idx}_{timestamps[start_idx].isoformat()}",
                                pattern_type=TemporalPatternType.ANOMALY,
                                variable=variable,
                                start_time=timestamps[start_idx],
                                end_time=timestamps[end_idx],
                                strength=0.7,  # Fixed strength for isolation forest
                                confidence=0.8,
                                parameters={
                                    'anomaly_indices': group,
                                    'detection_method': 'isolation_forest',
                                    'group_size': len(group)
                                },
                                supporting_data=data[start_idx:end_idx+1]
                            )
                            
                            patterns.append(pattern)
                            
        except ImportError:
            logger.debug("Scikit-learn not available for isolation forest anomaly detection")
        except Exception as e:
            logger.debug(f"Isolation forest anomaly detection failed: {e}")
        
        return patterns
    
    async def _detect_phase_transitions(self, variable: str, data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect phase transitions or regime changes"""
        
        patterns = []
        
        if len(data) < 20:
            return patterns
        
        timestamps = [t for t, v in data]
        values = np.array([v for t, v in data])
        
        # Moving window variance to detect regime changes
        window_size = max(5, len(data) // 10)
        variances = []
        
        for i in range(window_size, len(values) - window_size):
            window_var = np.var(values[i-window_size:i+window_size])
            variances.append(window_var)
        
        if len(variances) > 5:
            # Detect significant changes in variance
            var_changes = np.abs(np.diff(variances))
            change_threshold = np.percentile(var_changes, 90)  # Top 10% of changes
            
            significant_changes = np.where(var_changes > change_threshold)[0]
            
            for change_idx in significant_changes:
                actual_idx = change_idx + window_size
                
                # Analyze the transition
                before_period = values[max(0, actual_idx - window_size):actual_idx]
                after_period = values[actual_idx:min(len(values), actual_idx + window_size)]
                
                if len(before_period) > 0 and len(after_period) > 0:
                    # Calculate transition strength
                    mean_change = abs(np.mean(after_period) - np.mean(before_period))
                    std_change = abs(np.std(after_period) - np.std(before_period))
                    
                    transition_strength = (mean_change + std_change) / (np.std(values) + 1e-10)
                    
                    if transition_strength > 0.5:  # Significant transition
                        pattern = TemporalPattern(
                            pattern_id=f"phase_transition_{variable}_{actual_idx}_{timestamps[actual_idx].isoformat()}",
                            pattern_type=TemporalPatternType.PHASE_TRANSITION,
                            variable=variable,
                            start_time=timestamps[max(0, actual_idx - window_size)],
                            end_time=timestamps[min(len(timestamps) - 1, actual_idx + window_size)],
                            strength=min(1.0, transition_strength),
                            confidence=min(0.9, transition_strength * 0.8),
                            parameters={
                                'transition_index': actual_idx,
                                'before_mean': np.mean(before_period),
                                'after_mean': np.mean(after_period),
                                'before_std': np.std(before_period),
                                'after_std': np.std(after_period),
                                'mean_change': mean_change,
                                'std_change': std_change,
                                'transition_strength': transition_strength
                            },
                            supporting_data=data[max(0, actual_idx - window_size):min(len(data), actual_idx + window_size)]
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_feedback_loops(self, variable: str, data: List[Tuple[datetime, float]]) -> List[TemporalPattern]:
        """Detect feedback loop patterns"""
        
        patterns = []
        
        if len(data) < 30:  # Need sufficient data for feedback detection
            return patterns
        
        timestamps = [t for t, v in data]
        values = np.array([v for t, v in data])
        
        # Look for accelerating/decelerating patterns that suggest feedback
        # Calculate second derivative (acceleration)
        first_diff = np.diff(values)
        second_diff = np.diff(first_diff)
        
        if len(second_diff) > 0:
            # Detect sustained acceleration/deceleration
            window_size = max(5, len(second_diff) // 8)
            
            for i in range(window_size, len(second_diff) - window_size):
                window = second_diff[i-window_size:i+window_size]
                
                # Check for consistent acceleration/deceleration
                if len(window) > 0:
                    mean_acceleration = np.mean(window)
                    consistency = np.sum(np.sign(window) == np.sign(mean_acceleration)) / len(window)
                    
                    if consistency > 0.7 and abs(mean_acceleration) > np.std(second_diff) * 0.5:
                        # Detected feedback pattern
                        feedback_type = "positive" if mean_acceleration > 0 else "negative"
                        
                        pattern = TemporalPattern(
                            pattern_id=f"feedback_{feedback_type}_{variable}_{i}_{timestamps[i].isoformat()}",
                            pattern_type=TemporalPatternType.FEEDBACK_LOOP,
                            variable=variable,
                            start_time=timestamps[i - window_size],
                            end_time=timestamps[i + window_size],
                            strength=min(1.0, abs(mean_acceleration) / np.std(second_diff)),
                            confidence=consistency,
                            parameters={
                                'feedback_type': feedback_type,
                                'mean_acceleration': mean_acceleration,
                                'consistency': consistency,
                                'window_size': window_size,
                                'detection_index': i
                            },
                            supporting_data=data[i-window_size:i+window_size]
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _detect_anomalies(self, data: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float, str]]:
        """Detect anomalous data points"""
        
        if len(data) < 5:
            return []
        
        timestamps = [t for t, v in data]
        values = np.array([v for t, v in data])
        
        anomalies = []
        
        # Z-score based anomaly detection
        z_scores = np.abs(stats.zscore(values))
        anomaly_indices = np.where(z_scores > 3.0)[0]
        
        for idx in anomaly_indices:
            anomalies.append((timestamps[idx], values[idx], f"statistical_outlier_z{z_scores[idx]:.2f}"))
        
        # IQR-based anomaly detection
        q75, q25 = np.percentile(values, [75, 25])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        iqr_anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
        
        for idx in iqr_anomaly_indices:
            if idx not in anomaly_indices:  # Avoid duplicates
                anomalies.append((timestamps[idx], values[idx], "iqr_outlier"))
        
        return anomalies
    
    async def _detect_change_points(self, timestamps: List[datetime], values: np.ndarray) -> List[Tuple[datetime, str, float]]:
        """Detect significant change points in the time series"""
        
        change_points = []
        
        if len(values) < 10:
            return change_points
        
        # Detect mean shifts using cumulative sum
        try:
            # Cumulative sum of deviations from mean
            mean_val = np.mean(values)
            cumsum = np.cumsum(values - mean_val)
            
            # Find points where cumulative sum changes direction significantly
            cumsum_diff = np.diff(cumsum)
            
            # Detect significant direction changes
            for i in range(1, len(cumsum_diff) - 1):
                if (cumsum_diff[i-1] * cumsum_diff[i+1] < 0 and  # Direction change
                    abs(cumsum_diff[i]) > np.std(cumsum_diff) * 2):  # Significant magnitude
                    
                    change_magnitude = abs(cumsum_diff[i]) / np.std(cumsum_diff)
                    
                    change_points.append((
                        timestamps[i + 1],
                        "mean_shift",
                        change_magnitude
                    ))
            
        except Exception as e:
            logger.debug(f"Change point detection failed: {e}")
        
        # Detect variance changes
        try:
            window_size = max(5, len(values) // 10)
            rolling_vars = []
            
            for i in range(window_size, len(values) - window_size):
                window_var = np.var(values[i-window_size:i+window_size])
                rolling_vars.append((i, window_var))
            
            if len(rolling_vars) > 2:
                var_values = [v for i, v in rolling_vars]
                var_changes = np.abs(np.diff(var_values))
                change_threshold = np.percentile(var_changes, 95)  # Top 5% of changes
                
                significant_var_changes = np.where(var_changes > change_threshold)[0]
                
                for change_idx in significant_var_changes:
                    actual_idx = rolling_vars[change_idx][0]
                    change_magnitude = var_changes[change_idx] / np.std(var_changes)
                    
                    change_points.append((
                        timestamps[actual_idx],
                        "variance_shift",
                        change_magnitude
                    ))
                    
        except Exception as e:
            logger.debug(f"Variance change detection failed: {e}")
        
        # Remove duplicate change points (within 5% of data length)
        min_distance = max(1, len(timestamps) // 20)
        filtered_change_points = []
        
        for cp in sorted(change_points, key=lambda x: x[0]):
            if not filtered_change_points or \
               any(abs((cp[0] - fcp[0]).total_seconds()) > min_distance * 60 for fcp in filtered_change_points[-3:]):
                filtered_change_points.append(cp)
        
        return filtered_change_points[:10]  # Limit to top 10 change points
    
    async def _assess_predictability(self, values: np.ndarray) -> float:
        """Assess how predictable the time series is"""
        
        if len(values) < 5:
            return 0.0
        
        predictability_factors = []
        
        # Autocorrelation-based predictability
        if len(values) > 2:
            lag1_autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
            if not np.isnan(lag1_autocorr):
                predictability_factors.append(abs(lag1_autocorr))
        
        # Trend strength
        time_array = np.arange(len(values))
        if len(values) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_array, values)
            if not np.isnan(r_value):
                predictability_factors.append(abs(r_value))
        
        # Regularity (inverse of coefficient of variation)
        if np.std(values) > 0:
            cv = np.std(values) / abs(np.mean(values) + 1e-10)
            regularity = 1.0 / (1.0 + cv)
            predictability_factors.append(regularity)
        
        # Signal-to-noise ratio
        if len(values) > 5:
            # Simple signal: moving average
            window_size = min(5, len(values) // 2)
            signal = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            
            if len(signal) > 0:
                # Noise: deviation from moving average
                aligned_values = values[:len(signal)]
                noise_std = np.std(aligned_values - signal)
                signal_std = np.std(signal)
                
                if noise_std > 0:
                    snr = signal_std / noise_std
                    snr_factor = min(1.0, snr / 5.0)  # Normalize
                    predictability_factors.append(snr_factor)
        
        # Combined predictability score
        if predictability_factors:
            return np.mean(predictability_factors)
        else:
            return 0.3  # Default moderate predictability


class TemporalPredictor:
    """Advanced temporal prediction with multiple forecasting methods"""
    
    def __init__(self, min_data_points: int = 10):
        self.min_data_points = min_data_points
        self.prediction_cache = {}
        self.model_performance = defaultdict(list)
        
    async def predict_future_values(self, 
                                  variable: str,
                                  historical_data: List[Tuple[datetime, float]],
                                  prediction_horizons: List[datetime],
                                  patterns: List[TemporalPattern] = None) -> List[TemporalPrediction]:
        """Generate predictions for multiple future time points"""
        
        if len(historical_data) < self.min_data_points:
            raise ValueError(f"Insufficient data: need at least {self.min_data_points} points")
        
        # Sort data by timestamp
        sorted_data = sorted(historical_data, key=lambda x: x[0])
        timestamps = [t for t, v in sorted_data]
        values = np.array([v for t, v in sorted_data])
        
        # Prepare prediction methods
        prediction_methods = [
            ('linear_extrapolation', self._linear_extrapolation),
            ('exponential_smoothing', self._exponential_smoothing),
            ('pattern_based', self._pattern_based_prediction),
            ('ensemble', self._ensemble_prediction)
        ]
        
        predictions = []
        
        for horizon_time in prediction_horizons:
            method_predictions = {}
            
            # Generate predictions using different methods
            for method_name, method_func in prediction_methods:
                try:
                    pred_value, confidence = await method_func(
                        timestamps, values, horizon_time, patterns or []
                    )
                    method_predictions[method_name] = (pred_value, confidence)
                except Exception as e:
                    logger.debug(f"Prediction method {method_name} failed: {e}")
            
            if method_predictions:
                # Select best prediction or ensemble
                if 'ensemble' in method_predictions:
                    best_pred, best_conf = method_predictions['ensemble']
                    best_method = 'ensemble'
                else:
                    # Choose method with highest confidence
                    best_method = max(method_predictions.keys(), key=lambda k: method_predictions[k][1])
                    best_pred, best_conf = method_predictions[best_method]
                
                # Estimate confidence interval
                pred_std = np.std(values) * (1.0 - best_conf)  # Uncertainty scales with low confidence
                confidence_interval = (
                    best_pred - 1.96 * pred_std,  # 95% CI
                    best_pred + 1.96 * pred_std
                )
                
                # Determine prediction horizon type
                time_diff = horizon_time - timestamps[-1]
                if time_diff <= timedelta(hours=6):
                    horizon_type = PredictionHorizon.SHORT_TERM
                elif time_diff <= timedelta(days=1):
                    horizon_type = PredictionHorizon.MEDIUM_TERM
                elif time_diff <= timedelta(days=7):
                    horizon_type = PredictionHorizon.LONG_TERM
                else:
                    horizon_type = PredictionHorizon.STRATEGIC
                
                prediction = TemporalPrediction(
                    variable=variable,
                    prediction_time=horizon_time,
                    predicted_value=best_pred,
                    confidence_interval=confidence_interval,
                    confidence=best_conf,
                    horizon=horizon_type,
                    method=best_method,
                    contributing_patterns=[p.pattern_id for p in (patterns or []) if p.prediction_relevance > 0.3],
                    assumptions=[
                        "Historical patterns continue",
                        "No major external disruptions",
                        "Data quality remains consistent"
                    ],
                    uncertainty=pred_std,
                    reliability_score=await self._calculate_reliability_score(best_method, time_diff)
                )
                
                predictions.append(prediction)
        
        return predictions
    
    async def _linear_extrapolation(self, timestamps: List[datetime], values: np.ndarray, 
                                  target_time: datetime, patterns: List[TemporalPattern]) -> Tuple[float, float]:
        """Linear trend extrapolation"""
        
        # Convert timestamps to numeric values
        time_numeric = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        target_numeric = (target_time - timestamps[0]).total_seconds()
        
        # Fit linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
        
        # Predict
        predicted_value = slope * target_numeric + intercept
        
        # Confidence based on R-squared and extrapolation distance
        max_time = max(time_numeric)
        extrapolation_factor = min(1.0, target_numeric / max_time) if max_time > 0 else 1.0
        confidence = abs(r_value) * (1.0 - min(0.5, extrapolation_factor - 1.0))
        
        return predicted_value, max(0.1, confidence)
    
    async def _exponential_smoothing(self, timestamps: List[datetime], values: np.ndarray,
                                   target_time: datetime, patterns: List[TemporalPattern]) -> Tuple[float, float]:
        """Exponential smoothing prediction"""
        
        # Simple exponential smoothing
        alpha = 0.3  # Smoothing parameter
        
        if len(values) == 0:
            return 0.0, 0.0
        
        # Initialize with first value
        smoothed = values[0]
        
        # Apply exponential smoothing
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        # For future prediction, assume trend continues
        if len(values) >= 2:
            recent_trend = values[-1] - values[-2]
            time_steps = (target_time - timestamps[-1]).total_seconds() / (timestamps[-1] - timestamps[-2]).total_seconds()
            predicted_value = smoothed + recent_trend * time_steps * 0.5  # Damped trend
        else:
            predicted_value = smoothed
        
        # Confidence based on recent stability
        if len(values) >= 3:
            recent_values = values[-3:]
            stability = 1.0 - (np.std(recent_values) / (np.mean(np.abs(recent_values)) + 1e-10))
            confidence = max(0.1, min(0.8, stability))
        else:
            confidence = 0.5
        
        return predicted_value, confidence
    
    async def _pattern_based_prediction(self, timestamps: List[datetime], values: np.ndarray,
                                      target_time: datetime, patterns: List[TemporalPattern]) -> Tuple[float, float]:
        """Prediction based on identified temporal patterns"""
        
        if not patterns:
            # Fallback to mean if no patterns
            return float(np.mean(values)), 0.3
        
        # Find most relevant patterns
        relevant_patterns = [p for p in patterns if p.prediction_relevance > 0.3]
        
        if not relevant_patterns:
            relevant_patterns = sorted(patterns, key=lambda p: p.strength * p.confidence, reverse=True)[:3]
        
        predictions = []
        
        for pattern in relevant_patterns:
            try:
                if pattern.pattern_type == TemporalPatternType.TREND:
                    # Use trend parameters
                    slope = pattern.parameters.get('slope', 0)
                    intercept = pattern.parameters.get('intercept', np.mean(values))
                    
                    time_elapsed = (target_time - timestamps[0]).total_seconds()
                    pred_value = slope * time_elapsed + intercept
                    
                    predictions.append((pred_value, pattern.confidence * pattern.strength))
                
                elif pattern.pattern_type == TemporalPatternType.CYCLE and pattern.periodicity:
                    # Use cyclical pattern
                    last_timestamp = timestamps[-1]
                    time_in_cycle = (target_time - last_timestamp) % pattern.periodicity
                    cycle_fraction = time_in_cycle / pattern.periodicity
                    
                    # Simple sinusoidal approximation
                    amplitude = pattern.amplitude or np.std(values)
                    phase = pattern.phase or 0
                    mean_value = np.mean(values)
                    
                    pred_value = mean_value + amplitude * np.sin(2 * np.pi * cycle_fraction + phase)
                    
                    predictions.append((pred_value, pattern.confidence * 0.8))
                
                elif pattern.pattern_type == TemporalPatternType.SEASONAL and pattern.periodicity:
                    # Use seasonal pattern
                    seasonal_means = pattern.parameters.get('seasonal_means', {})
                    
                    if seasonal_means:
                        # Determine seasonal component
                        if 'hour' in pattern.parameters.get('season_type', ''):
                            seasonal_key = target_time.hour
                        elif 'day' in pattern.parameters.get('season_type', ''):
                            seasonal_key = target_time.weekday()
                        elif 'month' in pattern.parameters.get('season_type', ''):
                            seasonal_key = target_time.month
                        else:
                            seasonal_key = 0
                        
                        pred_value = seasonal_means.get(str(seasonal_key), np.mean(values))
                        predictions.append((pred_value, pattern.confidence * 0.9))
                
            except Exception as e:
                logger.debug(f"Pattern-based prediction failed for {pattern.pattern_id}: {e}")
        
        if predictions:
            # Weight predictions by confidence
            total_weight = sum(conf for pred, conf in predictions)
            if total_weight > 0:
                weighted_pred = sum(pred * conf for pred, conf in predictions) / total_weight
                avg_confidence = total_weight / len(predictions)
                return weighted_pred, avg_confidence
        
        # Fallback
        return float(np.mean(values)), 0.3
    
    async def _ensemble_prediction(self, timestamps: List[datetime], values: np.ndarray,
                                 target_time: datetime, patterns: List[TemporalPattern]) -> Tuple[float, float]:
        """Ensemble prediction combining multiple methods"""
        
        methods = [
            ('linear', self._linear_extrapolation),
            ('exponential', self._exponential_smoothing),
            ('pattern', self._pattern_based_prediction)
        ]
        
        predictions = []
        
        for method_name, method_func in methods:
            try:
                pred_value, confidence = await method_func(timestamps, values, target_time, patterns)
                predictions.append((pred_value, confidence, method_name))
            except Exception as e:
                logger.debug(f"Ensemble method {method_name} failed: {e}")
        
        if not predictions:
            return float(np.mean(values)), 0.2
        
        # Weight by confidence
        total_weight = sum(conf for pred, conf, method in predictions)
        
        if total_weight > 0:
            ensemble_pred = sum(pred * conf for pred, conf, method in predictions) / total_weight
            
            # Ensemble confidence is typically higher due to diversification
            base_confidence = sum(conf for pred, conf, method in predictions) / len(predictions)
            ensemble_confidence = min(0.95, base_confidence * 1.2)
            
            return ensemble_pred, ensemble_confidence
        else:
            return float(np.mean(values)), 0.2
    
    async def _calculate_reliability_score(self, method: str, time_horizon: timedelta) -> float:
        """Calculate reliability score based on method and time horizon"""
        
        # Base reliability by method
        method_reliability = {
            'linear_extrapolation': 0.7,
            'exponential_smoothing': 0.8,
            'pattern_based': 0.6,
            'ensemble': 0.85
        }
        
        base_reliability = method_reliability.get(method, 0.5)
        
        # Adjust for time horizon (closer predictions are more reliable)
        horizon_hours = time_horizon.total_seconds() / 3600
        
        if horizon_hours <= 1:
            time_factor = 1.0
        elif horizon_hours <= 6:
            time_factor = 0.9
        elif horizon_hours <= 24:
            time_factor = 0.7
        elif horizon_hours <= 168:  # 1 week
            time_factor = 0.5
        else:
            time_factor = 0.3
        
        return base_reliability * time_factor


class TemporalReasoningEngine:
    """
    Comprehensive temporal reasoning engine integrating pattern detection,
    causal analysis, and predictive capabilities
    """
    
    def __init__(self, 
                 significance_threshold: float = 0.05,
                 pattern_confidence_threshold: float = 0.6,
                 prediction_horizon_hours: int = 24):
        
        self.significance_threshold = significance_threshold
        self.pattern_confidence_threshold = pattern_confidence_threshold
        self.prediction_horizon_hours = prediction_horizon_hours
        
        # Core components
        self.time_series_analyzer = TimeSeriesAnalyzer(significance_threshold)
        self.temporal_predictor = TemporalPredictor()
        
        # Data storage
        self.variable_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.discovered_patterns: Dict[str, List[TemporalPattern]] = defaultdict(list)
        self.prediction_history: List[TemporalPrediction] = []
        
        # Integration with other reasoning systems
        self.causal_engine = None  # Will be set during integration
        self.working_memory = None  # Will be set during integration
        
        # Performance metrics
        self.performance_metrics = {
            'patterns_discovered': 0,
            'predictions_made': 0,
            'prediction_accuracy': 0.0,
            'analysis_efficiency': 0.0,
            'temporal_coherence': 0.0
        }
        
        logger.info("Initialized Temporal Reasoning Engine")
    
    async def add_temporal_observation(self, variable: str, timestamp: datetime, value: float) -> None:
        """Add temporal observation for analysis"""
        
        self.variable_data[variable].append((timestamp, value))
        
        # Maintain rolling window
        max_observations = 10000
        if len(self.variable_data[variable]) > max_observations:
            self.variable_data[variable] = self.variable_data[variable][-max_observations:]
        
        global_metrics.incr("temporal_reasoning.observations_added")
    
    async def analyze_temporal_patterns(self, 
                                      variables: List[str] = None,
                                      analysis_window: Optional[timedelta] = None) -> Dict[str, TimeSeriesAnalysis]:
        """Analyze temporal patterns across variables"""
        
        if variables is None:
            variables = list(self.variable_data.keys())
        
        analysis_results = {}
        analysis_tasks = []
        
        for variable in variables:
            if variable in self.variable_data and len(self.variable_data[variable]) > 0:
                task = asyncio.create_task(
                    self.time_series_analyzer.analyze_time_series(
                        variable, self.variable_data[variable], analysis_window
                    )
                )
                analysis_tasks.append((variable, task))
        
        # Execute analyses in parallel
        for variable, task in analysis_tasks:
            try:
                analysis = await task
                analysis_results[variable] = analysis
                
                # Store discovered patterns
                self.discovered_patterns[variable] = analysis.detected_patterns
                self.performance_metrics['patterns_discovered'] += len(analysis.detected_patterns)
                
            except Exception as e:
                logger.error(f"Temporal analysis failed for {variable}: {e}")
        
        logger.info(f"Analyzed {len(analysis_results)} variables, found {sum(len(patterns) for patterns in self.discovered_patterns.values())} patterns")
        
        return analysis_results
    
    async def predict_temporal_evolution(self, 
                                       variables: List[str],
                                       prediction_horizons: List[datetime] = None,
                                       use_patterns: bool = True) -> Dict[str, List[TemporalPrediction]]:
        """Predict future evolution of temporal variables"""
        
        if prediction_horizons is None:
            # Default prediction horizons
            current_time = datetime.now()
            prediction_horizons = [
                current_time + timedelta(hours=1),
                current_time + timedelta(hours=6),
                current_time + timedelta(hours=24),
                current_time + timedelta(days=7)
            ]
        
        prediction_results = {}
        prediction_tasks = []
        
        for variable in variables:
            if variable in self.variable_data and len(self.variable_data[variable]) >= 10:
                patterns = self.discovered_patterns.get(variable, []) if use_patterns else []
                
                task = asyncio.create_task(
                    self.temporal_predictor.predict_future_values(
                        variable, self.variable_data[variable], prediction_horizons, patterns
                    )
                )
                prediction_tasks.append((variable, task))
        
        # Execute predictions in parallel
        for variable, task in prediction_tasks:
            try:
                predictions = await task
                prediction_results[variable] = predictions
                
                self.prediction_history.extend(predictions)
                self.performance_metrics['predictions_made'] += len(predictions)
                
            except Exception as e:
                logger.error(f"Temporal prediction failed for {variable}: {e}")
        
        logger.info(f"Generated predictions for {len(prediction_results)} variables")
        
        return prediction_results
    
    async def analyze_temporal_causality(self, 
                                       cause_variable: str, 
                                       effect_variable: str,
                                       max_lag: timedelta = timedelta(hours=2)) -> Optional[Dict[str, Any]]:
        """Analyze temporal causal relationships between variables"""
        
        if (cause_variable not in self.variable_data or 
            effect_variable not in self.variable_data):
            return {}
        
        cause_data = self.variable_data[cause_variable]
        effect_data = self.variable_data[effect_variable]
        
        if len(cause_data) < 10 or len(effect_data) < 10:
            return {}
        
        # Test different time lags for causal relationships
        best_lag = None
        best_correlation = 0.0
        best_p_value = 1.0
        
        # Test lags from 0 to max_lag in 5-minute increments
        current_lag = timedelta(0)
        lag_increment = timedelta(minutes=5)
        
        while current_lag <= max_lag:
            correlation, p_value = await self._test_lagged_correlation(
                cause_data, effect_data, current_lag
            )
            
            if abs(correlation) > abs(best_correlation) and p_value < 0.05:
                best_correlation = correlation
                best_p_value = p_value
                best_lag = current_lag
            
            current_lag += lag_increment
        
        if best_lag is not None and abs(best_correlation) > 0.3:
            # Perform Granger causality test
            granger_result = await self._granger_causality_test(cause_data, effect_data, best_lag)
            
            # Integrate with causal engine if available
            causal_context = {}
            if self.causal_engine:
                try:
                    # Add temporal observation to causal engine
                    for timestamp, value in cause_data[-100:]:  # Recent data
                        await self.causal_engine.add_time_series_observation(cause_variable, timestamp, value)
                    
                    for timestamp, value in effect_data[-100:]:
                        await self.causal_engine.add_time_series_observation(effect_variable, timestamp, value)
                    
                    # Get causal insights
                    causal_context = {'causal_engine_integrated': True}
                except Exception as e:
                    logger.debug(f"Could not integrate with causal engine: {e}")
            
            return {
                'cause_variable': cause_variable,
                'effect_variable': effect_variable,
                'optimal_lag': best_lag,
                'correlation': best_correlation,
                'p_value': best_p_value,
                'granger_test': granger_result,
                'causal_strength': abs(best_correlation),
                'temporal_direction': 'positive' if best_correlation > 0 else 'negative',
                'confidence': 1.0 - best_p_value,
                'causal_context': causal_context
            }
        
        return {}
    
    async def _test_lagged_correlation(self, 
                                     cause_data: List[Tuple[datetime, float]],
                                     effect_data: List[Tuple[datetime, float]],
                                     lag: timedelta) -> Tuple[float, float]:
        """Test correlation with specified lag"""
        
        # Align data with lag
        aligned_pairs = []
        
        for cause_time, cause_value in cause_data:
            target_time = cause_time + lag
            
            # Find closest effect observation
            closest_effect = None
            min_time_diff = timedelta.max
            
            for effect_time, effect_value in effect_data:
                time_diff = abs(effect_time - target_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_effect = effect_value
            
            # Include if within reasonable time window
            if min_time_diff < timedelta(minutes=10) and closest_effect is not None:
                aligned_pairs.append((cause_value, closest_effect))
        
        if len(aligned_pairs) < 5:
            return 0.0, 1.0
        
        # Calculate correlation
        cause_values = np.array([pair[0] for pair in aligned_pairs])
        effect_values = np.array([pair[1] for pair in aligned_pairs])
        
        try:
            correlation, p_value = stats.pearsonr(cause_values, effect_values)
            
            if np.isnan(correlation):
                return 0.0, 1.0
            
            return correlation, p_value
            
        except Exception:
            return 0.0, 1.0
    
    async def _granger_causality_test(self, 
                                    cause_data: List[Tuple[datetime, float]],
                                    effect_data: List[Tuple[datetime, float]],
                                    optimal_lag: timedelta) -> Dict[str, Any]:
        """Simplified Granger causality test"""
        
        # This is a simplified version - full implementation would use statsmodels
        try:
            # Align data for regression analysis
            aligned_data = []
            
            for i, (effect_time, effect_value) in enumerate(effect_data[1:], 1):
                # Find lagged cause value
                target_cause_time = effect_time - optimal_lag
                
                # Find closest cause observation
                closest_cause = None
                min_time_diff = timedelta.max
                
                for cause_time, cause_value in cause_data:
                    time_diff = abs(cause_time - target_cause_time)
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        closest_cause = cause_value
                
                # Also get previous effect value (for autoregressive term)
                prev_effect_value = effect_data[i-1][1]
                
                if closest_cause is not None and min_time_diff < timedelta(minutes=15):
                    aligned_data.append((effect_value, prev_effect_value, closest_cause))
            
            if len(aligned_data) < 10:
                return {'test_feasible': False, 'reason': 'insufficient_aligned_data'}
            
            # Simple regression: effect_t = α + β₁*effect_{t-1} + β₂*cause_{t-lag} + ε
            y = np.array([row[0] for row in aligned_data])  # Current effect
            X_ar = np.array([row[1] for row in aligned_data])  # Previous effect (autoregressive)
            X_cause = np.array([row[2] for row in aligned_data])  # Lagged cause
            
            # Fit restricted model (without cause)
            ar_slope, ar_intercept, ar_r, ar_p, ar_se = stats.linregress(X_ar, y)
            ar_predictions = ar_slope * X_ar + ar_intercept
            sse_restricted = np.sum((y - ar_predictions)**2)
            
            # Fit unrestricted model (with cause) - simplified multiple regression
            # Use correlation as approximation
            cause_correlation, cause_p = stats.pearsonr(X_cause, y)
            
            # Calculate F-statistic approximation
            n = len(aligned_data)
            if n > 3 and ar_r != 1.0:
                f_stat_approx = abs(cause_correlation)**2 / (1 - abs(cause_correlation)**2) * (n - 2)
                
                # Convert to p-value approximation
                p_value_approx = 2 * (1 - stats.norm.cdf(abs(cause_correlation) * np.sqrt(n - 2)))
                
                granger_result = {
                    'test_feasible': True,
                    'f_statistic': f_stat_approx,
                    'p_value': p_value_approx,
                    'granger_causes': p_value_approx < 0.05,
                    'cause_coefficient': cause_correlation,
                    'autoregressive_coefficient': ar_r,
                    'sample_size': n
                }
            else:
                granger_result = {'test_feasible': False, 'reason': 'insufficient_variation'}
            
            return granger_result
            
        except Exception as e:
            return {'test_feasible': False, 'reason': f'test_failed: {e}'}
    
    async def synthesize_temporal_insights(self, 
                                         analysis_results: Dict[str, TimeSeriesAnalysis],
                                         prediction_results: Dict[str, List[TemporalPrediction]] = None) -> Dict[str, Any]:
        """Synthesize insights from temporal analysis and predictions"""
        
        insights = {
            'summary': {},
            'key_patterns': [],
            'temporal_relationships': [],
            'prediction_summary': {},
            'recommendations': [],
            'risk_factors': []
        }
        
        # Summarize analysis results
        all_patterns = []
        for variable, analysis in analysis_results.items():
            all_patterns.extend(analysis.detected_patterns)
        
        # Pattern type distribution
        pattern_types = defaultdict(int)
        for pattern in all_patterns:
            pattern_types[pattern.pattern_type.value] += 1
        
        insights['summary'] = {
            'total_variables_analyzed': len(analysis_results),
            'total_patterns_found': len(all_patterns),
            'pattern_type_distribution': dict(pattern_types),
            'average_predictability': np.mean([
                analysis.predictability_score for analysis in analysis_results.values()
            ]) if analysis_results else 0.0
        }
        
        # Identify key patterns (high strength and confidence)
        key_patterns = sorted(all_patterns, key=lambda p: p.strength * p.confidence, reverse=True)[:10]
        insights['key_patterns'] = [
            {
                'pattern_id': pattern.pattern_id,
                'type': pattern.pattern_type.value,
                'variable': pattern.variable,
                'strength': pattern.strength,
                'confidence': pattern.confidence,
                'description': self._describe_pattern(pattern)
            }
            for pattern in key_patterns
        ]
        
        # Identify temporal relationships
        variables = list(analysis_results.keys())
        for i, var1 in enumerate(variables):
            for var2 in variables[i+1:]:
                causal_analysis = await self.analyze_temporal_causality(var1, var2)
                if causal_analysis and causal_analysis['causal_strength'] > 0.4:
                    insights['temporal_relationships'].append(causal_analysis)
        
        # Summarize predictions if available
        if prediction_results:
            all_predictions = []
            for predictions in prediction_results.values():
                all_predictions.extend(predictions)
            
            insights['prediction_summary'] = {
                'total_predictions': len(all_predictions),
                'average_confidence': np.mean([p.confidence for p in all_predictions]),
                'prediction_horizons': list(set(p.horizon.value for p in all_predictions)),
                'high_uncertainty_predictions': len([
                    p for p in all_predictions if p.uncertainty > np.std([pred.predicted_value for pred in all_predictions])
                ])
            }
        
        # Generate recommendations
        insights['recommendations'] = await self._generate_recommendations(analysis_results, all_patterns)
        
        # Identify risk factors
        insights['risk_factors'] = await self._identify_risk_factors(all_patterns, analysis_results)
        
        # Update performance metrics
        self.performance_metrics['temporal_coherence'] = await self._calculate_temporal_coherence(analysis_results)
        
        return insights
    
    def _describe_pattern(self, pattern: TemporalPattern) -> str:
        """Generate human-readable description of pattern"""
        
        descriptions = {
            TemporalPatternType.TREND: f"{'Increasing' if pattern.parameters.get('slope', 0) > 0 else 'Decreasing'} trend with strength {pattern.strength:.2f}",
            TemporalPatternType.CYCLE: f"Cyclical pattern with {pattern.periodicity.total_seconds()/3600:.1f}h period" if pattern.periodicity else "Cyclical pattern",
            TemporalPatternType.SEASONAL: f"Seasonal pattern ({pattern.parameters.get('season_type', 'unknown')})",
            TemporalPatternType.ANOMALY: f"Anomalous behavior detected",
            TemporalPatternType.PHASE_TRANSITION: f"Phase transition with strength {pattern.strength:.2f}",
            TemporalPatternType.FEEDBACK_LOOP: f"{'Positive' if pattern.parameters.get('feedback_type') == 'positive' else 'Negative'} feedback loop"
        }
        
        return descriptions.get(pattern.pattern_type, f"{pattern.pattern_type.value} pattern")
    
    async def _generate_recommendations(self, 
                                      analysis_results: Dict[str, TimeSeriesAnalysis],
                                      patterns: List[TemporalPattern]) -> List[str]:
        """Generate actionable recommendations based on temporal analysis"""
        
        recommendations = []
        
        # Pattern-based recommendations
        strong_trends = [p for p in patterns if p.pattern_type == TemporalPatternType.TREND and p.strength > 0.7]
        if strong_trends:
            recommendations.append(f"Monitor {len(strong_trends)} strong trends that may require intervention")
        
        anomalies = [p for p in patterns if p.pattern_type == TemporalPatternType.ANOMALY]
        if len(anomalies) > 5:
            recommendations.append(f"Investigate {len(anomalies)} anomalous patterns for root causes")
        
        feedback_loops = [p for p in patterns if p.pattern_type == TemporalPatternType.FEEDBACK_LOOP]
        if feedback_loops:
            recommendations.append("Consider interventions to manage detected feedback loops")
        
        # Predictability-based recommendations
        low_predictability_vars = [
            var for var, analysis in analysis_results.items() 
            if analysis.predictability_score < 0.3
        ]
        if low_predictability_vars:
            recommendations.append(f"Improve data collection for low-predictability variables: {', '.join(low_predictability_vars[:3])}")
        
        # Change point recommendations
        frequent_changes = [
            var for var, analysis in analysis_results.items()
            if len(analysis.change_points) > 3
        ]
        if frequent_changes:
            recommendations.append(f"Investigate frequent regime changes in: {', '.join(frequent_changes[:3])}")
        
        return recommendations
    
    async def _identify_risk_factors(self, 
                                   patterns: List[TemporalPattern],
                                   analysis_results: Dict[str, TimeSeriesAnalysis]) -> List[Dict[str, Any]]:
        """Identify potential risk factors from temporal patterns"""
        
        risk_factors = []
        
        # High volatility risk
        high_volatility_vars = []
        for var, analysis in analysis_results.items():
            cv = analysis.std / max(abs(analysis.mean), 1e-10)  # Coefficient of variation
            if cv > 1.0:  # High volatility
                high_volatility_vars.append(var)
        
        if high_volatility_vars:
            risk_factors.append({
                'type': 'high_volatility',
                'severity': 'medium',
                'variables': high_volatility_vars,
                'description': 'High volatility may indicate instability or measurement issues'
            })
        
        # Accelerating negative trends
        negative_trends = [
            p for p in patterns 
            if (p.pattern_type == TemporalPatternType.TREND and 
                p.parameters.get('slope', 0) < 0 and 
                p.strength > 0.6)
        ]
        
        if negative_trends:
            risk_factors.append({
                'type': 'declining_trends',
                'severity': 'high',
                'variables': [p.variable for p in negative_trends],
                'description': 'Strong declining trends may indicate systemic issues'
            })
        
        # Regime shifts
        regime_shifts = [p for p in patterns if p.pattern_type == TemporalPatternType.PHASE_TRANSITION]
        if len(regime_shifts) > 2:
            risk_factors.append({
                'type': 'system_instability',
                'severity': 'high',
                'variables': [p.variable for p in regime_shifts],
                'description': 'Multiple regime shifts suggest system instability'
            })
        
        # Anomaly clustering
        anomaly_vars = defaultdict(int)
        for pattern in patterns:
            if pattern.pattern_type == TemporalPatternType.ANOMALY:
                anomaly_vars[pattern.variable] += 1
        
        high_anomaly_vars = [var for var, count in anomaly_vars.items() if count > 3]
        if high_anomaly_vars:
            risk_factors.append({
                'type': 'anomaly_clustering',
                'severity': 'medium',
                'variables': high_anomaly_vars,
                'description': 'Frequent anomalies may indicate data quality or systemic issues'
            })
        
        return risk_factors
    
    async def _calculate_temporal_coherence(self, analysis_results: Dict[str, TimeSeriesAnalysis]) -> float:
        """Calculate overall temporal coherence across all variables"""
        
        if not analysis_results:
            return 0.0
        
        coherence_factors = []
        
        # Pattern consistency across variables
        all_pattern_strengths = []
        for analysis in analysis_results.values():
            pattern_strengths = [p.strength for p in analysis.detected_patterns]
            if pattern_strengths:
                all_pattern_strengths.append(np.mean(pattern_strengths))
        
        if all_pattern_strengths:
            pattern_coherence = np.mean(all_pattern_strengths)
            coherence_factors.append(pattern_coherence)
        
        # Predictability consistency
        predictability_scores = [analysis.predictability_score for analysis in analysis_results.values()]
        if predictability_scores:
            pred_coherence = 1.0 - np.std(predictability_scores)  # Lower std = higher coherence
            coherence_factors.append(max(0.0, pred_coherence))
        
        # Temporal alignment (synchronized patterns)
        # Simplified: count variables with similar pattern timing
        pattern_times = defaultdict(list)
        for analysis in analysis_results.values():
            for pattern in analysis.detected_patterns:
                pattern_times[pattern.pattern_type].append(pattern.start_time)
        
        synchronized_patterns = 0
        total_pattern_types = 0
        
        for pattern_type, times in pattern_times.items():
            if len(times) > 1:
                total_pattern_types += 1
                time_spread = max(times) - min(times)
                if time_spread < timedelta(hours=6):  # Patterns within 6 hours
                    synchronized_patterns += 1
        
        if total_pattern_types > 0:
            sync_coherence = synchronized_patterns / total_pattern_types
            coherence_factors.append(sync_coherence)
        
        # Combined coherence
        return np.mean(coherence_factors) if coherence_factors else 0.5
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        
        return {
            **self.performance_metrics,
            'total_variables': len(self.variable_data),
            'total_observations': sum(len(data) for data in self.variable_data.values()),
            'patterns_by_type': {
                pattern_type.value: sum(
                    1 for patterns in self.discovered_patterns.values()
                    for pattern in patterns
                    if pattern.pattern_type == pattern_type
                ) for pattern_type in TemporalPatternType
            },
            'recent_predictions': len([
                p for p in self.prediction_history 
                if (datetime.now() - p.created_at) < timedelta(hours=24)
            ]),
            'cache_efficiency': len(self.time_series_analyzer.analysis_cache) / max(1, len(self.variable_data))
        }
    
    def set_causal_engine(self, causal_engine) -> None:
        """Set causal reasoning engine for integration"""
        self.causal_engine = causal_engine
        logger.info("Integrated with causal reasoning engine")
    
    def set_working_memory(self, working_memory) -> None:
        """Set working memory system for integration"""
        self.working_memory = working_memory
        logger.info("Integrated with working memory system")