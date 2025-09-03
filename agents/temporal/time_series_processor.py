"""
Time Series Processing for Temporal Reasoning
Handles time-series data processing and analysis
"""

import numpy as np
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from collections import deque

from utils.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimeSeriesData:
    """Represents time series data"""
    timestamps: List[datetime]
    values: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TimeSeriesForecast:
    """Forecast result from time series analysis"""
    forecast_values: List[float]
    forecast_timestamps: List[datetime]
    confidence_intervals: List[Tuple[float, float]]
    forecast_horizon: timedelta
    accuracy_estimate: float


class TimeSeriesProcessor:
    """
    Processes time-series data for temporal reasoning
    """
    
    def __init__(self, max_data_points: int = 10000):
        self.max_data_points = max_data_points
        self.data_buffer: deque = deque(maxlen=max_data_points)
        self.seasonality_cache: Dict[str, Any] = {}
        
        logger.info("Initialized time series processor")
    
    async def add_data_point(self, timestamp: datetime, value: float, series_id: str = "default") -> None:
        """Add a data point to the time series"""
        self.data_buffer.append({
            'timestamp': timestamp,
            'value': value,
            'series_id': series_id
        })
    
    async def get_trend(self, series_id: str = "default", window_size: int = 50) -> float:
        """Calculate trend over recent data points"""
        try:
            # Get recent data for this series
            series_data = [point for point in self.data_buffer if point['series_id'] == series_id]
            
            if len(series_data) < 2:
                return 0.0
            
            # Take last window_size points
            recent_data = series_data[-window_size:]
            values = [point['value'] for point in recent_data]
            
            if len(values) < 2:
                return 0.0
            
            # Calculate linear regression slope
            x = np.arange(len(values))
            slope, _, _, _, _ = stats.linregress(x, values)
            
            return float(slope)
            
        except Exception as e:
            logger.error(f"Error calculating trend: {e}")
            return 0.0
    
    async def detect_seasonality(self, series_id: str = "default", period_hints: List[int] = None) -> Dict[str, float]:
        """Detect seasonal patterns in the data"""
        try:
            series_data = [point for point in self.data_buffer if point['series_id'] == series_id]
            
            if len(series_data) < 20:
                return {}
            
            values = np.array([point['value'] for point in series_data])
            
            # Test common periods if not provided
            if period_hints is None:
                period_hints = [7, 24, 30, 365]  # daily, hourly, monthly, yearly patterns
            
            seasonality = {}
            
            for period in period_hints:
                if len(values) < period * 2:
                    continue
                
                # Simple autocorrelation for seasonality detection
                if len(values) > period:
                    correlation = np.corrcoef(values[:-period], values[period:])[0, 1]
                    if not np.isnan(correlation):
                        seasonality[f"period_{period}"] = float(correlation)
            
            return seasonality
            
        except Exception as e:
            logger.error(f"Error detecting seasonality: {e}")
            return {}
    
    async def forecast(self, series_id: str = "default", horizon: int = 10) -> TimeSeriesForecast:
        """Generate forecasts using simple moving average with trend"""
        try:
            series_data = [point for point in self.data_buffer if point['series_id'] == series_id]
            
            if len(series_data) < 3:
                # Not enough data for meaningful forecast
                last_timestamp = datetime.now()
                return TimeSeriesForecast(
                    forecast_values=[0.0] * horizon,
                    forecast_timestamps=[last_timestamp + timedelta(minutes=i) for i in range(1, horizon+1)],
                    confidence_intervals=[(0.0, 0.0)] * horizon,
                    forecast_horizon=timedelta(minutes=horizon),
                    accuracy_estimate=0.0
                )
            
            # Get recent values and timestamps
            recent_data = series_data[-20:]  # Use last 20 points
            values = [point['value'] for point in recent_data]
            timestamps = [point['timestamp'] for point in recent_data]
            
            # Calculate moving average and trend
            window = min(5, len(values))
            moving_avg = np.mean(values[-window:])
            trend = await self.get_trend(series_id, window_size=10)
            
            # Generate forecast
            forecast_values = []
            forecast_timestamps = []
            
            last_timestamp = timestamps[-1]
            avg_interval = (timestamps[-1] - timestamps[0]) / (len(timestamps) - 1)
            
            for i in range(horizon):
                # Simple linear forecast with trend
                forecast_value = moving_avg + trend * (i + 1)
                forecast_values.append(forecast_value)
                
                forecast_timestamp = last_timestamp + avg_interval * (i + 1)
                forecast_timestamps.append(forecast_timestamp)
            
            # Estimate confidence intervals (simple approach)
            std_dev = np.std(values) if len(values) > 1 else 0.0
            confidence_intervals = [
                (val - 1.96 * std_dev, val + 1.96 * std_dev) 
                for val in forecast_values
            ]
            
            # Estimate accuracy based on recent variance
            accuracy_estimate = max(0.0, min(1.0, 1.0 - (std_dev / max(abs(moving_avg), 1.0))))
            
            return TimeSeriesForecast(
                forecast_values=forecast_values,
                forecast_timestamps=forecast_timestamps,
                confidence_intervals=confidence_intervals,
                forecast_horizon=avg_interval * horizon,
                accuracy_estimate=accuracy_estimate
            )
            
        except Exception as e:
            logger.error(f"Error generating forecast: {e}")
            # Return safe fallback
            return TimeSeriesForecast(
                forecast_values=[0.0] * horizon,
                forecast_timestamps=[datetime.now() + timedelta(minutes=i) for i in range(1, horizon+1)],
                confidence_intervals=[(0.0, 0.0)] * horizon,
                forecast_horizon=timedelta(minutes=horizon),
                accuracy_estimate=0.0
            )
    
    async def get_statistics(self, series_id: str = "default") -> Dict[str, float]:
        """Get basic statistics for a time series"""
        try:
            series_data = [point for point in self.data_buffer if point['series_id'] == series_id]
            
            if not series_data:
                return {}
            
            values = [point['value'] for point in series_data]
            
            return {
                'count': len(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'trend': await self.get_trend(series_id)
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}
    
    async def clear_series(self, series_id: str) -> None:
        """Clear all data for a specific series"""
        self.data_buffer = deque(
            [point for point in self.data_buffer if point['series_id'] != series_id],
            maxlen=self.max_data_points
        )
        logger.info(f"Cleared data for series: {series_id}")