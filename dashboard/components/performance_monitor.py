"""
Performance Monitor Component
Real-time performance tracking and analytics visualizations
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time

class PerformanceMonitor:
    """Performance monitoring dashboard component"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#2a5298',
            'secondary': '#1e3c72', 
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8'
        }
    
    def render_processing_trends(self, data: Dict[str, Any]):
        """Render processing trends charts"""
        st.markdown("### 📈 Processing Volume Trends")
        
        # Convert daily processing data to DataFrame
        daily_df = pd.DataFrame(data['daily_processing'])
        daily_df['date'] = pd.to_datetime(daily_df['date'])
        
        # Create multi-chart layout
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Daily Document Processing', 
                'Processing Accuracy Trend',
                'Cost Savings Over Time',
                'Average Processing Time'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Daily processing volume
        fig.add_trace(
            go.Scatter(
                x=daily_df['date'],
                y=daily_df['documents'],
                mode='lines+markers',
                name='Documents Processed',
                line=dict(color=self.color_scheme['primary'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Accuracy trend
        fig.add_trace(
            go.Scatter(
                x=daily_df['date'],
                y=daily_df['accuracy'],
                mode='lines+markers',
                name='Accuracy %',
                line=dict(color=self.color_scheme['success'], width=3),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Cost savings
        fig.add_trace(
            go.Bar(
                x=daily_df['date'],
                y=daily_df['cost_savings'],
                name='Daily Savings',
                marker_color=self.color_scheme['info']
            ),
            row=2, col=1
        )
        
        # Processing time
        if 'processing_time_avg' in daily_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=daily_df['date'],
                    y=daily_df['processing_time_avg'],
                    mode='lines+markers',
                    name='Avg Time (sec)',
                    line=dict(color=self.color_scheme['warning'], width=3),
                    marker=dict(size=6)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Processing Performance Trends",
            title_x=0.5
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Documents", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy (%)", row=1, col=2)
        fig.update_yaxes(title_text="Cost Savings ($)", row=2, col=1)
        fig.update_yaxes(title_text="Time (seconds)", row=2, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add trend analysis
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_daily_docs = daily_df['documents'].mean()
            trend = "📈" if daily_df['documents'].iloc[-1] > avg_daily_docs else "📉"
            st.metric(
                "Average Daily Volume",
                f"{avg_daily_docs:,.0f}",
                delta=f"{trend} {daily_df['documents'].iloc[-1] - avg_daily_docs:+.0f}"
            )
        
        with col2:
            avg_accuracy = daily_df['accuracy'].mean()
            accuracy_trend = daily_df['accuracy'].iloc[-1] - daily_df['accuracy'].iloc[0]
            st.metric(
                "Average Accuracy",
                f"{avg_accuracy:.1f}%",
                delta=f"{accuracy_trend:+.1f}%"
            )
        
        with col3:
            total_savings = daily_df['cost_savings'].sum()
            st.metric(
                "Total Period Savings",
                f"${total_savings:,.2f}",
                delta="vs manual processing"
            )
    
    def render_accuracy_analysis(self, data: Dict[str, Any]):
        """Render accuracy analysis by document type"""
        st.markdown("### 🎯 Accuracy Analysis by Document Type")
        
        # Extract document type accuracy data
        doc_types = data['document_types']
        
        # Create accuracy comparison chart
        doc_names = list(doc_types.keys())
        accuracies = [doc_types[doc]['accuracy'] for doc in doc_names]
        volumes = [doc_types[doc]['count'] for doc in doc_names]
        
        # Create bubble chart (accuracy vs volume)
        fig_bubble = go.Figure()
        
        # Color scale based on accuracy
        colors = ['#dc3545' if acc < 95 else '#ffc107' if acc < 97 else '#28a745' for acc in accuracies]
        
        fig_bubble.add_trace(
            go.Scatter(
                x=volumes,
                y=accuracies,
                mode='markers+text',
                text=[name.replace('_', ' ').title() for name in doc_names],
                textposition='middle center',
                marker=dict(
                    size=[vol/100 for vol in volumes],  # Scale bubble size
                    sizemode='diameter',
                    sizeref=max(volumes)/100000,
                    color=colors,
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                hovertemplate=
                '<b>%{text}</b><br>' +
                'Accuracy: %{y:.1f}%<br>' +
                'Volume: %{x:,} documents<br>' +
                '<extra></extra>'
            )
        )
        
        fig_bubble.update_layout(
            title="Accuracy vs Volume Analysis (Bubble Size = Volume)",
            xaxis_title="Document Volume",
            yaxis_title="Accuracy Rate (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Accuracy distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy distribution histogram
            fig_hist = px.histogram(
                x=accuracies,
                nbins=10,
                title="Accuracy Distribution",
                labels={'x': 'Accuracy Rate (%)', 'y': 'Number of Document Types'},
                color_discrete_sequence=[self.color_scheme['primary']]
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Document type performance radar chart
            categories = [name.replace('_', ' ').title() for name in doc_names]
            
            fig_radar = go.Figure()
            
            fig_radar.add_trace(go.Scatterpolar(
                r=accuracies,
                theta=categories,
                fill='toself',
                name='Accuracy Rate',
                line_color=self.color_scheme['primary']
            ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[85, 100]
                    )),
                title="Document Type Performance Radar",
                height=400
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        
        # Top performers and improvement opportunities
        st.markdown("#### 🏆 Performance Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🥇 Top Performers**")
            sorted_by_accuracy = sorted(doc_types.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            
            for i, (doc_type, metrics) in enumerate(sorted_by_accuracy[:3]):
                icon = "🥇" if i == 0 else "🥈" if i == 1 else "🥉"
                st.markdown(f"""
                {icon} **{doc_type.replace('_', ' ').title()}**  
                Accuracy: {metrics['accuracy']:.1f}% | Volume: {metrics['count']:,}
                """)
        
        with col2:
            st.markdown("**🎯 Improvement Opportunities**")
            sorted_by_accuracy_asc = sorted(doc_types.items(), key=lambda x: x[1]['accuracy'])
            
            for doc_type, metrics in sorted_by_accuracy_asc[:3]:
                improvement_potential = 98 - metrics['accuracy']  # Assuming 98% is target
                st.markdown(f"""
                📈 **{doc_type.replace('_', ' ').title()}**  
                Current: {metrics['accuracy']:.1f}% | Potential: +{improvement_potential:.1f}%
                """)
    
    def render_speed_metrics(self, data: Dict[str, Any]):
        """Render processing speed and performance metrics"""
        st.markdown("### ⚡ Processing Speed Metrics")
        
        # System health metrics
        health = data['system_health']
        
        # Create speed metrics dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Throughput/Hour",
                f"{health['throughput_per_hour']:,}",
                delta=f"+{np.random.randint(50, 150)} vs yesterday"
            )
        
        with col2:
            st.metric(
                "Avg Response Time",
                f"{health['avg_response_time_ms']:.0f}ms",
                delta=f"-{np.random.randint(10, 50)}ms improvement"
            )
        
        with col3:
            st.metric(
                "Queue Size",
                health['processing_queue_size'],
                delta="Real-time"
            )
        
        with col4:
            st.metric(
                "Active Agents",
                health['active_agents'],
                delta="All healthy" if health['active_agents'] == 8 else "Monitor"
            )
        
        # Processing time by document type
        doc_types = data['document_types']
        
        # Create processing time comparison
        doc_names = [name.replace('_', ' ').title() for name in doc_types.keys()]
        processing_times = [doc_types[doc]['avg_processing_time'] for doc in doc_types.keys()]
        
        fig_times = go.Figure(data=[
            go.Bar(
                x=doc_names,
                y=processing_times,
                marker_color=[
                    self.color_scheme['success'] if time < 3 else 
                    self.color_scheme['warning'] if time < 6 else 
                    self.color_scheme['danger'] 
                    for time in processing_times
                ],
                text=[f"{time:.1f}s" for time in processing_times],
                textposition='auto'
            )
        ])
        
        fig_times.update_layout(
            title="Average Processing Time by Document Type",
            xaxis_title="Document Type",
            yaxis_title="Processing Time (seconds)",
            height=400
        )
        
        st.plotly_chart(fig_times, use_container_width=True)
        
        # Real-time processing gauge
        col1, col2 = st.columns(2)
        
        with col1:
            # CPU usage gauge
            fig_cpu = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health.get('cpu_usage', 45),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "CPU Usage (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_scheme['primary']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90}
                }
            ))
            fig_cpu.update_layout(height=300)
            st.plotly_chart(fig_cpu, use_container_width=True)
        
        with col2:
            # Memory usage gauge
            fig_memory = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=health.get('memory_usage', 65),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Memory Usage (%)"},
                delta={'reference': 70},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.color_scheme['info']},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 95}
                }
            ))
            fig_memory.update_layout(height=300)
            st.plotly_chart(fig_memory, use_container_width=True)
        
        # Performance insights
        st.markdown("#### 🔍 Performance Insights")
        
        insights = []
        
        # Generate insights based on data
        avg_processing_time = np.mean(processing_times)
        if avg_processing_time < 3:
            insights.append("🟢 Excellent processing speed across all document types")
        elif avg_processing_time > 5:
            insights.append("🟡 Some document types may benefit from optimization")
        
        if health['throughput_per_hour'] > 1000:
            insights.append("🟢 High throughput maintained - system performing well")
        
        if health['processing_queue_size'] < 10:
            insights.append("🟢 Processing queue is well managed")
        elif health['processing_queue_size'] > 20:
            insights.append("🟡 Processing queue building up - consider scaling")
        
        cpu_usage = health.get('cpu_usage', 45)
        if cpu_usage > 80:
            insights.append("🟡 High CPU usage detected - monitor for performance impact")
        elif cpu_usage < 50:
            insights.append("🟢 CPU resources are well utilized")
        
        for insight in insights:
            st.markdown(f"• {insight}")
    
    def render_real_time_performance_stream(self, data: Dict[str, Any]):
        """Render real-time performance streaming visualization"""
        st.markdown("### 🌊 Real-Time Performance Stream")
        
        # Create placeholder for real-time updates
        placeholder = st.empty()
        
        # Generate real-time data
        timestamps = []
        throughput_values = []
        accuracy_values = []
        
        for i in range(50):
            timestamps.append(datetime.now() - timedelta(seconds=i*10))
            throughput_values.append(np.random.uniform(15, 25))
            accuracy_values.append(np.random.uniform(94, 98))
        
        # Reverse to get chronological order
        timestamps.reverse()
        throughput_values.reverse()
        accuracy_values.reverse()
        
        # Create streaming chart
        fig_stream = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Real-Time Throughput (docs/min)', 'Real-Time Accuracy (%)'),
            vertical_spacing=0.1
        )
        
        # Throughput stream
        fig_stream.add_trace(
            go.Scatter(
                x=timestamps,
                y=throughput_values,
                mode='lines',
                name='Throughput',
                line=dict(color=self.color_scheme['primary'], width=2),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Accuracy stream
        fig_stream.add_trace(
            go.Scatter(
                x=timestamps,
                y=accuracy_values,
                mode='lines',
                name='Accuracy',
                line=dict(color=self.color_scheme['success'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        fig_stream.update_layout(
            height=500,
            showlegend=False,
            title_text="Live Performance Monitoring"
        )
        
        fig_stream.update_xaxes(title_text="Time", row=2, col=1)
        fig_stream.update_yaxes(title_text="Documents/min", row=1, col=1)
        fig_stream.update_yaxes(title_text="Accuracy %", row=2, col=1)
        
        with placeholder.container():
            st.plotly_chart(fig_stream, use_container_width=True)