
"""
Minimal Test Dashboard Server
Simple coordination dashboard for testing
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime
from pathlib import Path

class MinimalDashboardServer:
    """Minimal dashboard server for testing"""
    
    def __init__(self, port: int = 8502):
        self.port = port
        self.agents_data = {}
        self.metrics_data = []
        self.connections = set()
    
    async def start(self):
        """Start the minimal dashboard"""
        print(f"[TEST] Minimal dashboard ready on port {self.port}")
        
        # Generate some test data
        await self.generate_test_data()
        
        return True
    
    async def generate_test_data(self):
        """Generate test data for dashboard"""
        self.agents_data = {
            "temporal_agent": {
                "status": "active",
                "tasks_completed": 25,
                "performance": 0.85,
                "last_active": datetime.now().isoformat()
            },
            "memory_agent": {
                "status": "active", 
                "tasks_completed": 18,
                "performance": 0.92,
                "last_active": datetime.now().isoformat()
            },
            "coordination_agent": {
                "status": "active",
                "tasks_completed": 12,
                "performance": 0.78,
                "last_active": datetime.now().isoformat()
            }
        }
        
        self.metrics_data = [
            {"timestamp": datetime.now().isoformat(), "metric": "throughput", "value": 45.2},
            {"timestamp": datetime.now().isoformat(), "metric": "accuracy", "value": 92.1},
            {"timestamp": datetime.now().isoformat(), "metric": "response_time", "value": 1.8}
        ]
    
    async def get_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        return {
            "status": "operational",
            "active_agents": len(self.agents_data),
            "total_metrics": len(self.metrics_data),
            "uptime": "test_mode"
        }
    
    async def get_agents(self) -> Dict[str, Any]:
        """Get agent data"""
        return self.agents_data
    
    async def get_metrics(self) -> List[Dict[str, Any]]:
        """Get metrics data"""
        return self.metrics_data

# Global test dashboard instance
test_dashboard = MinimalDashboardServer()
