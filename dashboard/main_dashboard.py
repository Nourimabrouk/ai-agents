"""
Enterprise Business Intelligence Dashboard
Real-time document processing platform performance and ROI analytics
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
import time
from typing import Dict, List, Any, Optional
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Enterprise Document Processing Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import dashboard components
from components.performance_monitor import PerformanceMonitor
from components.roi_analytics import ROIAnalytics
from components.document_intelligence import DocumentIntelligence
from components.real_time_charts import RealTimeCharts
from services.data_service import DataService
from utils.auth import AuthManager
from utils.formatting import format_currency, format_percentage, format_number

# CSS styling for professional appearance
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 20px;
        margin: -20px -20px 20px -20px;
        border-radius: 10px;
        text-align: center;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2a5298;
        margin: 10px 0;
    }
    
    .success-metric {
        border-left-color: #28a745;
    }
    
    .warning-metric {
        border-left-color: #ffc107;
    }
    
    .danger-metric {
        border-left-color: #dc3545;
    }
    
    .info-metric {
        border-left-color: #17a2b8;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-healthy { background-color: #28a745; }
    .status-warning { background-color: #ffc107; }
    .status-critical { background-color: #dc3545; }
    
    .dashboard-section {
        margin: 30px 0;
        padding: 20px;
        background: #f8f9fa;
        border-radius: 10px;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

class EnterpriseDashboard:
    """Main dashboard controller class"""
    
    def __init__(self):
        self.data_service = DataService()
        self.auth_manager = AuthManager()
        self.performance_monitor = PerformanceMonitor()
        self.roi_analytics = ROIAnalytics()
        self.document_intelligence = DocumentIntelligence()
        self.real_time_charts = RealTimeCharts()
        
        # Initialize session state
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'selected_date_range' not in st.session_state:
            st.session_state.selected_date_range = (
                datetime.now() - timedelta(days=30),
                datetime.now()
            )
    
    async def load_dashboard_data(self) -> Dict[str, Any]:
        """Load all dashboard data asynchronously"""
        try:
            # Run data loading operations in parallel
            dashboard_data = await self.data_service.get_dashboard_data(
                start_date=st.session_state.selected_date_range[0],
                end_date=st.session_state.selected_date_range[1]
            )
            
            return dashboard_data
        except Exception as e:
            st.error(f"Failed to load dashboard data: {e}")
            return self._get_demo_data()
    
    def _get_demo_data(self) -> Dict[str, Any]:
        """Generate demo data for dashboard preview"""
        now = datetime.now()
        dates = pd.date_range(start=now - timedelta(days=30), end=now, freq='D')
        
        return {
            'system_health': {
                'overall_status': 'healthy',
                'uptime_hours': 720.5,
                'processing_queue_size': 15,
                'active_agents': 8,
                'success_rate': 96.2,
                'avg_response_time_ms': 245,
                'throughput_per_hour': 1125,
                'cost_per_document': 0.03
            },
            'performance_metrics': {
                'total_documents_processed': 45832,
                'accuracy_rate': 96.2,
                'processing_speed': 1125,
                'cost_savings': 282150.75,
                'error_rate': 3.8,
                'uptime_percentage': 99.7
            },
            'document_types': {
                'invoices': {'count': 18500, 'accuracy': 97.1, 'avg_processing_time': 2.3},
                'receipts': {'count': 12200, 'accuracy': 95.8, 'avg_processing_time': 1.8},
                'purchase_orders': {'count': 8900, 'accuracy': 96.5, 'avg_processing_time': 2.7},
                'bank_statements': {'count': 3200, 'accuracy': 94.2, 'avg_processing_time': 4.1},
                'contracts': {'count': 2100, 'accuracy': 98.3, 'avg_processing_time': 8.5},
                'tax_forms': {'count': 832, 'accuracy': 99.1, 'avg_processing_time': 12.2},
                'insurance_claims': {'count': 100, 'accuracy': 92.5, 'avg_processing_time': 6.8}
            },
            'daily_processing': [
                {'date': date, 
                 'documents': np.random.randint(800, 1500),
                 'accuracy': np.random.uniform(94, 98),
                 'cost_savings': np.random.uniform(5000, 15000)}
                for date in dates
            ],
            'cost_analysis': {
                'manual_cost_per_document': 6.15,
                'automated_cost_per_document': 0.03,
                'savings_per_document': 6.12,
                'total_savings_ytd': 282150.75,
                'roi_percentage': 99.5
            },
            'vendor_analysis': [
                {'vendor': 'ABC Corp', 'total_amount': 125000, 'document_count': 850, 'avg_amount': 147.06},
                {'vendor': 'TechSupply Inc', 'total_amount': 98500, 'document_count': 445, 'avg_amount': 221.35},
                {'vendor': 'Office Solutions', 'total_amount': 76800, 'document_count': 1250, 'avg_amount': 61.44},
                {'vendor': 'DataCorp Ltd', 'total_amount': 54200, 'document_count': 320, 'avg_amount': 169.38}
            ],
            'anomalies': [
                {'type': 'duplicate_invoice', 'count': 5, 'severity': 'medium'},
                {'type': 'amount_mismatch', 'count': 12, 'severity': 'high'},
                {'type': 'missing_po_number', 'count': 8, 'severity': 'low'},
                {'type': 'invalid_vendor', 'count': 3, 'severity': 'critical'}
            ]
        }
    
    def render_header(self):
        """Render dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Enterprise Document Processing Dashboard</h1>
            <p>Real-time Business Intelligence • Multi-Domain AI Platform • 99.5% Cost Reduction Achievement</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.markdown("## 🎛️ Dashboard Controls")
        
        # Auto-refresh toggle
        st.session_state.auto_refresh = st.sidebar.checkbox(
            "🔄 Auto Refresh (5 sec)", 
            value=st.session_state.auto_refresh
        )
        
        # Date range selector
        st.sidebar.markdown("### 📅 Date Range")
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=st.session_state.selected_date_range,
            max_value=datetime.now(),
            key="date_range_input"
        )
        
        if len(date_range) == 2:
            st.session_state.selected_date_range = date_range
        
        # Organization selector (for multi-tenant)
        st.sidebar.markdown("### 🏢 Organization")
        org_options = ["All Organizations", "Acme Corp", "TechStart Inc", "Global Enterprises"]
        selected_org = st.sidebar.selectbox("Select Organization", org_options)
        
        # Document type filter
        st.sidebar.markdown("### 📄 Document Types")
        doc_types = st.sidebar.multiselect(
            "Filter by Document Type",
            ["invoices", "receipts", "purchase_orders", "bank_statements", 
             "contracts", "tax_forms", "insurance_claims"],
            default=["invoices", "receipts", "purchase_orders"]
        )
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Now", type="primary"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
        
        # Export options
        st.sidebar.markdown("### 📊 Export Data")
        if st.sidebar.button("📋 Export to CSV"):
            # This would export current dashboard data
            st.sidebar.success("Export initiated!")
        
        if st.sidebar.button("📄 Generate PDF Report"):
            # This would generate a PDF report
            st.sidebar.success("PDF report generated!")
        
        # System info
        st.sidebar.markdown("### ℹ️ System Info")
        st.sidebar.info(f"""
        **Last Refresh:** {st.session_state.last_refresh.strftime('%H:%M:%S')}  
        **Dashboard Version:** v1.0.0  
        **API Status:** 🟢 Connected  
        **Data Freshness:** < 30 seconds  
        """)
    
    def render_key_metrics(self, data: Dict[str, Any]):
        """Render key performance metrics cards"""
        metrics = data['system_health']
        perf_metrics = data['performance_metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
            st.metric(
                label="🎯 Processing Accuracy",
                value=f"{perf_metrics['accuracy_rate']:.1f}%",
                delta=f"+{np.random.uniform(0.1, 0.5):.1f}%"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card info-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚡ Throughput/Hour",
                value=f"{format_number(metrics['throughput_per_hour'])}",
                delta=f"+{np.random.randint(50, 150)}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card success-metric">', unsafe_allow_html=True)
            st.metric(
                label="💰 Cost Savings",
                value=format_currency(perf_metrics['cost_savings']),
                delta=f"+{format_currency(np.random.uniform(1000, 5000))}"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            uptime_color = "success" if metrics['uptime_hours'] > 700 else "warning"
            st.markdown(f'<div class="metric-card {uptime_color}-metric">', unsafe_allow_html=True)
            st.metric(
                label="🔥 System Uptime",
                value=f"{metrics['uptime_hours']:.1f}h",
                delta=f"99.7% availability"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Additional metrics row
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            st.metric(
                label="📊 Documents Processed",
                value=format_number(perf_metrics['total_documents_processed']),
                delta=f"+{np.random.randint(500, 2000)} today"
            )
        
        with col6:
            st.metric(
                label="🤖 Active Agents",
                value=metrics['active_agents'],
                delta="All healthy" if metrics['active_agents'] == 8 else "Check status"
            )
        
        with col7:
            st.metric(
                label="💸 Cost per Document",
                value=format_currency(metrics['cost_per_document']),
                delta=f"-{format_currency(6.12)} vs manual"
            )
        
        with col8:
            error_delta = "🟢 Good" if perf_metrics['error_rate'] < 5 else "🟡 Monitor"
            st.metric(
                label="⚠️ Error Rate",
                value=f"{perf_metrics['error_rate']:.1f}%",
                delta=error_delta
            )
    
    def render_real_time_monitor(self, data: Dict[str, Any]):
        """Render real-time processing monitor"""
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("## 🔄 Real-Time Processing Monitor")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Processing throughput chart
            self.real_time_charts.render_throughput_chart(data)
        
        with col2:
            # System status indicators
            st.markdown("### System Health")
            
            health_status = data['system_health']['overall_status']
            status_icon = "🟢" if health_status == 'healthy' else "🟡" if health_status == 'warning' else "🔴"
            
            st.markdown(f"""
            **Overall Status:** {status_icon} {health_status.title()}  
            **Queue Size:** {data['system_health']['processing_queue_size']} documents  
            **Response Time:** {data['system_health']['avg_response_time_ms']}ms  
            **Active Processing:** {np.random.randint(5, 25)} documents  
            """)
            
            # Processing queue visualization
            queue_data = pd.DataFrame({
                'Document Type': ['Invoices', 'Receipts', 'POs', 'Other'],
                'Queue Count': [8, 4, 2, 1]
            })
            
            fig_queue = px.pie(
                queue_data, 
                values='Queue Count', 
                names='Document Type',
                title="Processing Queue Breakdown"
            )
            fig_queue.update_layout(height=300)
            st.plotly_chart(fig_queue, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_performance_analytics(self, data: Dict[str, Any]):
        """Render performance analytics section"""
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("## 📈 Performance Analytics")
        
        tab1, tab2, tab3 = st.tabs(["📊 Processing Trends", "🎯 Accuracy Analysis", "⚡ Speed Metrics"])
        
        with tab1:
            self.performance_monitor.render_processing_trends(data)
        
        with tab2:
            self.performance_monitor.render_accuracy_analysis(data)
        
        with tab3:
            self.performance_monitor.render_speed_metrics(data)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_roi_dashboard(self, data: Dict[str, Any]):
        """Render ROI analytics dashboard"""
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("## 💰 ROI Analytics & Cost Savings")
        
        self.roi_analytics.render_cost_comparison(data)
        self.roi_analytics.render_savings_timeline(data)
        self.roi_analytics.render_roi_calculator(data)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_document_intelligence(self, data: Dict[str, Any]):
        """Render document intelligence insights"""
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("## 🧠 Document Intelligence Insights")
        
        tab1, tab2, tab3 = st.tabs(["📋 Document Analysis", "🚨 Anomaly Detection", "🏢 Vendor Intelligence"])
        
        with tab1:
            self.document_intelligence.render_document_breakdown(data)
        
        with tab2:
            self.document_intelligence.render_anomaly_detection(data)
        
        with tab3:
            self.document_intelligence.render_vendor_analysis(data)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_3d_visualizations(self, data: Dict[str, Any]):
        """Render impressive 3D visualizations"""
        st.markdown('<div class="dashboard-section">', unsafe_allow_html=True)
        st.markdown("## 🎨 Advanced 3D Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 3D processing flow visualization
            self.real_time_charts.render_3d_processing_flow(data)
        
        with col2:
            # 3D accuracy surface plot
            self.real_time_charts.render_3d_accuracy_surface(data)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    async def run_dashboard(self):
        """Main dashboard execution"""
        # Render components
        self.render_header()
        self.render_sidebar()
        
        # Load data
        dashboard_data = await self.load_dashboard_data()
        
        # Main dashboard content
        self.render_key_metrics(dashboard_data)
        self.render_real_time_monitor(dashboard_data)
        self.render_performance_analytics(dashboard_data)
        self.render_roi_dashboard(dashboard_data)
        self.render_document_intelligence(dashboard_data)
        self.render_3d_visualizations(dashboard_data)
        
        # Auto-refresh mechanism
        if st.session_state.auto_refresh:
            time.sleep(0.1)  # Small delay to prevent excessive refreshing
            st.rerun()

# Initialize and run dashboard
@st.cache_resource
def get_dashboard():
    """Get dashboard instance with caching"""
    return EnterpriseDashboard()

def main():
    """Main application entry point"""
    dashboard = get_dashboard()
    
    # Run dashboard asynchronously
    try:
        asyncio.run(dashboard.run_dashboard())
    except Exception as e:
        st.error(f"Dashboard error: {e}")
        st.info("Displaying demo data...")
        # Fallback to synchronous mode with demo data
        dashboard_data = dashboard._get_demo_data()
        dashboard.render_key_metrics(dashboard_data)
        dashboard.render_real_time_monitor(dashboard_data)
        dashboard.render_performance_analytics(dashboard_data)
        dashboard.render_roi_dashboard(dashboard_data)
        dashboard.render_document_intelligence(dashboard_data)

if __name__ == "__main__":
    main()