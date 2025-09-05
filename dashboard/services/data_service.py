"""
Real-time Data Service for Enterprise Dashboard
Handles live data streaming, caching, and API integrations
"""

import asyncio
import aiohttp
import aioredis
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import websockets
import sqlite3
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RealTimeMetric:
    """Real-time metric data structure"""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    tags: Dict[str, str]

@dataclass
class ProcessingEvent:
    """Document processing event"""
    event_id: str
    timestamp: datetime
    document_id: str
    document_type: str
    processing_stage: str
    processing_time_ms: float
    accuracy_score: float
    cost: float
    status: str

class DataService:
    """Central data service for dashboard"""
    
    def __init__(self):
        self.redis_client = None
        self.db_engine = None
        self.websocket_connections = set()
        self.real_time_buffer = []
        self.cache_timeout = 300  # 5 minutes
        
        # Performance tracking
        self.metrics_buffer = []
        self.processing_events = []
        self.system_health = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'active_connections': 0,
            'processing_queue_size': 0
        }
        
        asyncio.create_task(self._initialize_connections())
    
    async def _initialize_connections(self):
        """Initialize database and cache connections"""
        try:
            # Initialize Redis for caching
            self.redis_client = aioredis.from_url(
                "redis://localhost:6379", 
                decode_responses=True
            )
            
            # Initialize database connection
            self.db_engine = create_async_engine(
                "sqlite+aiosqlite:///./dashboard_data.db"
            )
            
            # Start background tasks
            asyncio.create_task(self._real_time_data_collector())
            asyncio.create_task(self._system_health_monitor())
            
            logger.info("Data service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize data service: {e}")
    
    async def _real_time_data_collector(self):
        """Continuously collect real-time metrics"""
        while True:
            try:
                # Simulate real-time data collection
                current_time = datetime.now()
                
                # Generate processing metrics
                processing_metric = RealTimeMetric(
                    timestamp=current_time,
                    metric_name="documents_per_minute",
                    value=np.random.uniform(15, 25),
                    unit="documents",
                    tags={"source": "processing_engine"}
                )
                
                accuracy_metric = RealTimeMetric(
                    timestamp=current_time,
                    metric_name="accuracy_rate",
                    value=np.random.uniform(94, 98),
                    unit="percentage",
                    tags={"source": "validation_engine"}
                )
                
                cost_metric = RealTimeMetric(
                    timestamp=current_time,
                    metric_name="cost_per_document",
                    value=np.random.uniform(0.025, 0.035),
                    unit="usd",
                    tags={"source": "cost_calculator"}
                )
                
                # Add to buffer
                self.metrics_buffer.extend([processing_metric, accuracy_metric, cost_metric])
                
                # Keep buffer size manageable
                if len(self.metrics_buffer) > 1000:
                    self.metrics_buffer = self.metrics_buffer[-500:]
                
                # Cache metrics in Redis
                await self._cache_real_time_metrics([processing_metric, accuracy_metric, cost_metric])
                
                # Broadcast to WebSocket connections
                await self._broadcast_real_time_data({
                    'timestamp': current_time.isoformat(),
                    'metrics': {
                        'processing_rate': processing_metric.value,
                        'accuracy': accuracy_metric.value,
                        'cost_per_doc': cost_metric.value
                    }
                })
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Real-time data collection error: {e}")
                await asyncio.sleep(10)
    
    async def _system_health_monitor(self):
        """Monitor system health metrics"""
        while True:
            try:
                # Simulate system health metrics
                self.system_health.update({
                    'cpu_usage': np.random.uniform(20, 80),
                    'memory_usage': np.random.uniform(40, 90),
                    'active_connections': np.random.randint(5, 50),
                    'processing_queue_size': np.random.randint(0, 25),
                    'uptime_seconds': (datetime.now() - datetime.now().replace(hour=0, minute=0, second=0)).total_seconds()
                })
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"System health monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _cache_real_time_metrics(self, metrics: List[RealTimeMetric]):
        """Cache real-time metrics in Redis"""
        try:
            if self.redis_client:
                for metric in metrics:
                    cache_key = f"metric:{metric.metric_name}:{metric.timestamp.isoformat()}"
                    cache_data = {
                        'value': metric.value,
                        'unit': metric.unit,
                        'tags': metric.tags,
                        'timestamp': metric.timestamp.isoformat()
                    }
                    await self.redis_client.setex(
                        cache_key, 
                        self.cache_timeout, 
                        json.dumps(cache_data)
                    )
        except Exception as e:
            logger.error(f"Cache error: {e}")
    
    async def _broadcast_real_time_data(self, data: Dict):
        """Broadcast real-time data to WebSocket connections"""
        try:
            if self.websocket_connections:
                message = json.dumps(data)
                # Remove closed connections
                closed_connections = []
                for ws in self.websocket_connections:
                    try:
                        await ws.send(message)
                    except websockets.exceptions.ConnectionClosed:
                        closed_connections.append(ws)
                
                # Clean up closed connections
                for ws in closed_connections:
                    self.websocket_connections.discard(ws)
                    
        except Exception as e:
            logger.error(f"WebSocket broadcast error: {e}")
    
    async def get_dashboard_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        try:
            # Check cache first
            cache_key = f"dashboard_data:{start_date.isoformat()}:{end_date.isoformat()}"
            cached_data = await self._get_cached_data(cache_key)
            if cached_data:
                return cached_data
            
            # Generate comprehensive dashboard data
            dashboard_data = await self._generate_dashboard_data(start_date, end_date)
            
            # Cache the result
            await self._cache_dashboard_data(cache_key, dashboard_data)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to get dashboard data: {e}")
            return await self._get_fallback_data()
    
    async def _generate_dashboard_data(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        # System health data
        system_health = {
            'overall_status': 'healthy',
            'uptime_hours': self.system_health['uptime_seconds'] / 3600,
            'processing_queue_size': self.system_health['processing_queue_size'],
            'active_agents': 8,
            'success_rate': 96.2,
            'avg_response_time_ms': np.random.uniform(200, 300),
            'throughput_per_hour': np.random.randint(1000, 1300),
            'cost_per_document': 0.03,
            'cpu_usage': self.system_health['cpu_usage'],
            'memory_usage': self.system_health['memory_usage'],
            'active_connections': self.system_health['active_connections']
        }
        
        # Performance metrics
        performance_metrics = {
            'total_documents_processed': np.random.randint(40000, 50000),
            'accuracy_rate': 96.2,
            'processing_speed': system_health['throughput_per_hour'],
            'cost_savings': np.random.uniform(250000, 300000),
            'error_rate': np.random.uniform(3, 5),
            'uptime_percentage': 99.7
        }
        
        # Document type breakdown
        document_types = {
            'invoices': {
                'count': np.random.randint(15000, 20000), 
                'accuracy': np.random.uniform(96, 98), 
                'avg_processing_time': np.random.uniform(2, 3)
            },
            'receipts': {
                'count': np.random.randint(10000, 15000), 
                'accuracy': np.random.uniform(94, 97), 
                'avg_processing_time': np.random.uniform(1.5, 2.5)
            },
            'purchase_orders': {
                'count': np.random.randint(7000, 10000), 
                'accuracy': np.random.uniform(95, 98), 
                'avg_processing_time': np.random.uniform(2.5, 3.5)
            },
            'bank_statements': {
                'count': np.random.randint(2500, 4000), 
                'accuracy': np.random.uniform(92, 96), 
                'avg_processing_time': np.random.uniform(3.5, 5)
            },
            'contracts': {
                'count': np.random.randint(1500, 2500), 
                'accuracy': np.random.uniform(97, 99), 
                'avg_processing_time': np.random.uniform(7, 10)
            },
            'tax_forms': {
                'count': np.random.randint(500, 1000), 
                'accuracy': np.random.uniform(98, 99.5), 
                'avg_processing_time': np.random.uniform(10, 15)
            },
            'insurance_claims': {
                'count': np.random.randint(80, 150), 
                'accuracy': np.random.uniform(90, 95), 
                'avg_processing_time': np.random.uniform(5, 8)
            }
        }
        
        # Daily processing data
        days = (end_date - start_date).days
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        daily_processing = []
        base_docs = 1000
        
        for i, date in enumerate(dates):
            # Add some realistic patterns (weekends lower, month-end higher)
            weekday_factor = 0.7 if date.weekday() >= 5 else 1.0  # Weekend reduction
            month_end_factor = 1.3 if date.day >= 28 else 1.0    # Month-end boost
            trend_factor = 1 + (i / days) * 0.2                   # Growth trend
            
            doc_count = int(base_docs * weekday_factor * month_end_factor * trend_factor * np.random.uniform(0.8, 1.2))
            
            daily_processing.append({
                'date': date,
                'documents': doc_count,
                'accuracy': np.random.uniform(94, 98),
                'cost_savings': doc_count * 6.12,  # Savings per document
                'processing_time_avg': np.random.uniform(2, 4)
            })
        
        # Cost analysis
        cost_analysis = {
            'manual_cost_per_document': 6.15,
            'automated_cost_per_document': 0.03,
            'savings_per_document': 6.12,
            'total_savings_ytd': performance_metrics['cost_savings'],
            'roi_percentage': 99.5,
            'payback_period_months': 2.3
        }
        
        # Vendor analysis
        vendor_names = ['ABC Corp', 'TechSupply Inc', 'Office Solutions', 'DataCorp Ltd', 
                       'Global Services', 'Smart Systems', 'Enterprise Tools']
        vendor_analysis = []
        
        for vendor in vendor_names[:5]:  # Top 5 vendors
            doc_count = np.random.randint(200, 1500)
            avg_amount = np.random.uniform(50, 300)
            total_amount = doc_count * avg_amount
            
            vendor_analysis.append({
                'vendor': vendor,
                'total_amount': total_amount,
                'document_count': doc_count,
                'avg_amount': avg_amount,
                'accuracy_rate': np.random.uniform(92, 98)
            })
        
        # Anomaly detection
        anomalies = [
            {'type': 'duplicate_invoice', 'count': np.random.randint(3, 8), 'severity': 'medium'},
            {'type': 'amount_mismatch', 'count': np.random.randint(8, 15), 'severity': 'high'},
            {'type': 'missing_po_number', 'count': np.random.randint(5, 12), 'severity': 'low'},
            {'type': 'invalid_vendor', 'count': np.random.randint(1, 5), 'severity': 'critical'},
            {'type': 'date_inconsistency', 'count': np.random.randint(10, 20), 'severity': 'low'},
            {'type': 'currency_mismatch', 'count': np.random.randint(2, 6), 'severity': 'medium'}
        ]
        
        # Real-time processing events
        processing_events = []
        for i in range(50):  # Last 50 events
            event_time = datetime.now() - timedelta(minutes=i*2)
            processing_events.append({
                'timestamp': event_time,
                'document_id': f"DOC_{np.random.randint(10000, 99999)}",
                'document_type': np.random.choice(list(document_types.keys())),
                'processing_time_ms': np.random.uniform(1500, 5000),
                'accuracy_score': np.random.uniform(90, 99),
                'status': np.random.choice(['completed', 'processing', 'error'], p=[0.85, 0.10, 0.05])
            })
        
        return {
            'system_health': system_health,
            'performance_metrics': performance_metrics,
            'document_types': document_types,
            'daily_processing': daily_processing,
            'cost_analysis': cost_analysis,
            'vendor_analysis': vendor_analysis,
            'anomalies': anomalies,
            'processing_events': processing_events,
            'real_time_metrics': self.metrics_buffer[-100:],  # Last 100 metrics
            'timestamp': datetime.now()
        }
    
    async def _get_cached_data(self, cache_key: str) -> Optional[Dict]:
        """Get cached dashboard data"""
        try:
            if self.redis_client:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    return json.loads(cached)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return {}
    
    async def _cache_dashboard_data(self, cache_key: str, data: Dict):
        """Cache dashboard data"""
        try:
            if self.redis_client:
                # Convert datetime objects to strings for JSON serialization
                serializable_data = self._make_serializable(data)
                await self.redis_client.setex(
                    cache_key, 
                    self.cache_timeout, 
                    json.dumps(serializable_data, default=str)
                )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    def _make_serializable(self, obj):
        """Convert datetime objects to strings for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    async def _get_fallback_data(self) -> Dict[str, Any]:
        """Get fallback data when main data source fails"""
        return await self._generate_dashboard_data(
            datetime.now() - timedelta(days=30),
            datetime.now()
        )
    
    async def get_real_time_metrics(self) -> List[RealTimeMetric]:
        """Get current real-time metrics"""
        return self.metrics_buffer[-100:]  # Last 100 metrics
    
    async def get_processing_events(self, limit: int = 50) -> List[ProcessingEvent]:
        """Get recent processing events"""
        return self.processing_events[-limit:]
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get current system health"""
        return {
            **self.system_health,
            'timestamp': datetime.now(),
            'overall_healthy': all([
                self.system_health['cpu_usage'] < 90,
                self.system_health['memory_usage'] < 95,
                self.system_health['processing_queue_size'] < 50
            ])
        }
    
    async def websocket_handler(self, websocket, path):
        """Handle WebSocket connections for real-time updates"""
        self.websocket_connections.add(websocket)
        logger.info(f"WebSocket connection established: {websocket.remote_address}")
        
        try:
            await websocket.wait_closed()
        finally:
            self.websocket_connections.discard(websocket)
            logger.info(f"WebSocket connection closed: {websocket.remote_address}")
    
    async def start_websocket_server(self, host="localhost", port=8765):
        """Start WebSocket server for real-time data streaming"""
        try:
            await websockets.serve(self.websocket_handler, host, port)
            logger.info(f"WebSocket server started on {host}:{port}")
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
    
    async def close(self):
        """Clean up connections"""
        try:
            if self.redis_client:
                await self.redis_client.close()
            
            for ws in self.websocket_connections:
                await ws.close()
            
            logger.info("Data service connections closed")
        except Exception as e:
            logger.error(f"Error closing data service: {e}")

# Singleton instance
_data_service_instance = None

def get_data_service() -> DataService:
    """Get singleton data service instance"""
    global _data_service_instance
    if _data_service_instance is None:
        _data_service_instance = DataService()
    return _data_service_instance