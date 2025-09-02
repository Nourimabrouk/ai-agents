"""
ROI Analytics Component
Advanced ROI calculations, cost analysis, and business value visualization
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import math

class ROIAnalytics:
    """ROI analytics and business value component"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#2a5298',
            'secondary': '#1e3c72',
            'success': '#28a745',
            'warning': '#ffc107', 
            'danger': '#dc3545',
            'info': '#17a2b8',
            'money': '#28a745',
            'savings': '#20c997'
        }
        
        # Default cost assumptions
        self.cost_assumptions = {
            'manual_hourly_rate': 25.0,
            'manual_processing_time_minutes': 15,
            'automated_cost_per_document': 0.03,
            'infrastructure_monthly_cost': 5000,
            'setup_cost': 50000
        }
    
    def render_cost_comparison(self, data: Dict[str, Any]):
        """Render cost comparison visualization"""
        st.markdown("### 💰 Cost Analysis & Savings Breakdown")
        
        cost_data = data['cost_analysis']
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Manual Cost/Doc",
                f"${cost_data['manual_cost_per_document']:.2f}",
                delta=f"+{cost_data['manual_cost_per_document']:.2f} vs automated"
            )
        
        with col2:
            st.metric(
                "Automated Cost/Doc", 
                f"${cost_data['automated_cost_per_document']:.3f}",
                delta=f"-{cost_data['savings_per_document']:.2f} savings"
            )
        
        with col3:
            st.metric(
                "Savings Per Document",
                f"${cost_data['savings_per_document']:.2f}",
                delta=f"{cost_data['roi_percentage']:.1f}% reduction"
            )
        
        with col4:
            st.metric(
                "Total YTD Savings",
                f"${cost_data['total_savings_ytd']:,.2f}",
                delta="vs manual processing"
            )
        
        # Cost breakdown visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Cost comparison bar chart
            categories = ['Manual Processing', 'Automated Processing']
            costs = [cost_data['manual_cost_per_document'], cost_data['automated_cost_per_document']]
            
            fig_cost_comparison = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=costs,
                    marker_color=[self.color_scheme['danger'], self.color_scheme['success']],
                    text=[f"${cost:.3f}" for cost in costs],
                    textposition='auto'
                )
            ])
            
            fig_cost_comparison.update_layout(
                title="Cost Per Document Comparison",
                yaxis_title="Cost ($)",
                height=400
            )
            
            st.plotly_chart(fig_cost_comparison, use_container_width=True)
        
        with col2:
            # Cost breakdown pie chart
            monthly_docs = data['performance_metrics']['total_documents_processed'] / 12  # Assuming yearly data
            
            manual_monthly_cost = monthly_docs * cost_data['manual_cost_per_document']
            automated_monthly_cost = monthly_docs * cost_data['automated_cost_per_document']
            infrastructure_cost = self.cost_assumptions['infrastructure_monthly_cost']
            
            cost_breakdown = {
                'Manual Labor (Avoided)': manual_monthly_cost,
                'AI Processing': automated_monthly_cost,
                'Infrastructure': infrastructure_cost,
                'Net Savings': manual_monthly_cost - automated_monthly_cost - infrastructure_cost
            }
            
            fig_breakdown = px.pie(
                values=list(cost_breakdown.values()),
                names=list(cost_breakdown.keys()),
                title="Monthly Cost Breakdown",
                color_discrete_sequence=[self.color_scheme['danger'], self.color_scheme['primary'], 
                                       self.color_scheme['warning'], self.color_scheme['success']]
            )
            fig_breakdown.update_layout(height=400)
            st.plotly_chart(fig_breakdown, use_container_width=True)
    
    def render_savings_timeline(self, data: Dict[str, Any]):
        """Render savings timeline and projections"""
        st.markdown("### 📈 Savings Timeline & ROI Projection")
        
        # Generate historical and projected data
        months = 24  # 2 years
        start_date = datetime.now() - timedelta(days=365)  # Start 1 year ago
        
        timeline_data = []
        cumulative_savings = 0
        monthly_volume = data['performance_metrics']['total_documents_processed'] / 12
        monthly_savings = monthly_volume * data['cost_analysis']['savings_per_document']
        
        for i in range(months):
            month_date = start_date + timedelta(days=30*i)
            
            # Add some realistic growth and seasonality
            growth_factor = 1 + (i * 0.05)  # 5% monthly growth
            seasonal_factor = 1 + 0.2 * math.sin(2 * math.pi * i / 12)  # Seasonal variation
            
            monthly_docs = monthly_volume * growth_factor * seasonal_factor
            monthly_saving = monthly_docs * data['cost_analysis']['savings_per_document']
            cumulative_savings += monthly_saving
            
            # Infrastructure costs
            infrastructure_cost = self.cost_assumptions['infrastructure_monthly_cost']
            net_monthly_savings = monthly_saving - infrastructure_cost
            
            timeline_data.append({
                'date': month_date,
                'monthly_documents': int(monthly_docs),
                'monthly_savings': monthly_saving,
                'cumulative_savings': cumulative_savings,
                'infrastructure_cost': infrastructure_cost,
                'net_savings': net_monthly_savings,
                'roi_percentage': (cumulative_savings / self.cost_assumptions['setup_cost']) * 100 if cumulative_savings > 0 else 0
            })
        
        timeline_df = pd.DataFrame(timeline_data)
        
        # Create timeline visualization
        fig_timeline = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Monthly Savings Trend',
                'Cumulative Savings Growth', 
                'ROI Percentage Over Time',
                'Volume vs Savings Correlation'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # Monthly savings
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['monthly_savings'],
                mode='lines+markers',
                name='Monthly Savings',
                line=dict(color=self.color_scheme['success'], width=3),
                fill='tonexty'
            ),
            row=1, col=1
        )
        
        # Cumulative savings
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['cumulative_savings'],
                mode='lines+markers',
                name='Cumulative Savings',
                line=dict(color=self.color_scheme['money'], width=3)
            ),
            row=1, col=2
        )
        
        # ROI percentage
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df['date'],
                y=timeline_df['roi_percentage'],
                mode='lines+markers',
                name='ROI %',
                line=dict(color=self.color_scheme['primary'], width=3)
            ),
            row=2, col=1
        )
        
        # Volume vs Savings scatter
        fig_timeline.add_trace(
            go.Scatter(
                x=timeline_df['monthly_documents'],
                y=timeline_df['monthly_savings'],
                mode='markers',
                name='Volume/Savings',
                marker=dict(
                    color=timeline_df['roi_percentage'],
                    colorscale='Viridis',
                    size=8,
                    colorbar=dict(title="ROI %")
                )
            ),
            row=2, col=2
        )
        
        fig_timeline.update_layout(
            height=700,
            showlegend=False,
            title_text="Comprehensive ROI Analysis"
        )
        
        # Update axes
        fig_timeline.update_xaxes(title_text="Date", row=2, col=1)
        fig_timeline.update_xaxes(title_text="Monthly Documents", row=2, col=2)
        fig_timeline.update_yaxes(title_text="Monthly Savings ($)", row=1, col=1)
        fig_timeline.update_yaxes(title_text="Cumulative Savings ($)", row=1, col=2)
        fig_timeline.update_yaxes(title_text="ROI (%)", row=2, col=1)
        fig_timeline.update_yaxes(title_text="Monthly Savings ($)", row=2, col=2)
        
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Key insights
        st.markdown("#### 🎯 Key ROI Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            payback_months = self.cost_assumptions['setup_cost'] / (monthly_savings - self.cost_assumptions['infrastructure_monthly_cost'])
            st.metric(
                "Payback Period",
                f"{payback_months:.1f} months",
                delta="Initial investment recovered"
            )
        
        with col2:
            annual_savings = (monthly_savings - self.cost_assumptions['infrastructure_monthly_cost']) * 12
            st.metric(
                "Annual Net Savings",
                f"${annual_savings:,.2f}",
                delta="After infrastructure costs"
            )
        
        with col3:
            five_year_savings = annual_savings * 5
            st.metric(
                "5-Year Projection",
                f"${five_year_savings:,.2f}",
                delta="Projected total savings"
            )
    
    def render_roi_calculator(self, data: Dict[str, Any]):
        """Interactive ROI calculator"""
        st.markdown("### 🧮 Interactive ROI Calculator")
        
        with st.expander("🎛️ Adjust Parameters", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                manual_rate = st.slider(
                    "Manual Hourly Rate ($)",
                    min_value=15.0,
                    max_value=50.0,
                    value=self.cost_assumptions['manual_hourly_rate'],
                    step=1.0
                )
                
                processing_time = st.slider(
                    "Manual Processing Time (min)",
                    min_value=5,
                    max_value=30,
                    value=self.cost_assumptions['manual_processing_time_minutes'],
                    step=1
                )
                
                monthly_volume = st.number_input(
                    "Monthly Document Volume",
                    min_value=1000,
                    max_value=100000,
                    value=int(data['performance_metrics']['total_documents_processed'] / 12),
                    step=1000
                )
            
            with col2:
                ai_cost = st.slider(
                    "AI Cost Per Document ($)",
                    min_value=0.01,
                    max_value=0.10,
                    value=self.cost_assumptions['automated_cost_per_document'],
                    step=0.005,
                    format="%.3f"
                )
                
                infrastructure_cost = st.number_input(
                    "Monthly Infrastructure Cost ($)",
                    min_value=1000,
                    max_value=20000,
                    value=self.cost_assumptions['infrastructure_monthly_cost'],
                    step=500
                )
                
                setup_cost = st.number_input(
                    "Initial Setup Cost ($)",
                    min_value=10000,
                    max_value=200000,
                    value=self.cost_assumptions['setup_cost'],
                    step=5000
                )
        
        # Calculate ROI with custom parameters
        manual_cost_per_doc = (manual_rate / 60) * processing_time
        savings_per_doc = manual_cost_per_doc - ai_cost
        monthly_savings = monthly_volume * savings_per_doc
        net_monthly_savings = monthly_savings - infrastructure_cost
        annual_net_savings = net_monthly_savings * 12
        payback_months = setup_cost / net_monthly_savings if net_monthly_savings > 0 else float('inf')
        roi_percentage = ((savings_per_doc / manual_cost_per_doc) * 100) if manual_cost_per_doc > 0 else 0
        
        # Display calculated results
        st.markdown("#### 📊 Calculated ROI Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Cost Savings/Doc",
                f"${savings_per_doc:.3f}",
                delta=f"{roi_percentage:.1f}% reduction"
            )
        
        with col2:
            st.metric(
                "Monthly Net Savings",
                f"${net_monthly_savings:,.2f}",
                delta="After all costs"
            )
        
        with col3:
            st.metric(
                "Payback Period",
                f"{payback_months:.1f} months" if payback_months != float('inf') else "Never",
                delta="Investment recovery"
            )
        
        with col4:
            st.metric(
                "Annual ROI",
                f"{(annual_net_savings/setup_cost)*100:.0f}%" if setup_cost > 0 else "N/A",
                delta="Return on investment"
            )
        
        # ROI visualization with custom parameters
        years = 5
        roi_projection = []
        cumulative_savings = 0
        
        for year in range(1, years + 1):
            annual_savings = net_monthly_savings * 12
            cumulative_savings += annual_savings
            
            roi_projection.append({
                'year': year,
                'annual_savings': annual_savings,
                'cumulative_savings': cumulative_savings,
                'cumulative_roi': (cumulative_savings / setup_cost) * 100 if setup_cost > 0 else 0
            })
        
        roi_df = pd.DataFrame(roi_projection)
        
        # Create projection chart
        fig_projection = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Cumulative Savings Projection', 'ROI Growth Over Time')
        )
        
        # Cumulative savings
        fig_projection.add_trace(
            go.Bar(
                x=roi_df['year'],
                y=roi_df['cumulative_savings'],
                name='Cumulative Savings',
                marker_color=self.color_scheme['success'],
                text=[f"${savings:,.0f}" for savings in roi_df['cumulative_savings']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # ROI percentage
        fig_projection.add_trace(
            go.Scatter(
                x=roi_df['year'],
                y=roi_df['cumulative_roi'],
                mode='lines+markers',
                name='Cumulative ROI %',
                line=dict(color=self.color_scheme['primary'], width=4),
                marker=dict(size=10)
            ),
            row=1, col=2
        )
        
        fig_projection.update_layout(
            height=400,
            showlegend=False
        )
        
        fig_projection.update_xaxes(title_text="Year", row=1, col=1)
        fig_projection.update_xaxes(title_text="Year", row=1, col=2)
        fig_projection.update_yaxes(title_text="Cumulative Savings ($)", row=1, col=1)
        fig_projection.update_yaxes(title_text="ROI (%)", row=1, col=2)
        
        st.plotly_chart(fig_projection, use_container_width=True)
    
    def render_business_value_analysis(self, data: Dict[str, Any]):
        """Render comprehensive business value analysis"""
        st.markdown("### 🎯 Business Value Analysis")
        
        # Efficiency improvements
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚡ Efficiency Gains")
            
            # Time savings calculation
            manual_hours_saved = data['performance_metrics']['total_documents_processed'] * (self.cost_assumptions['manual_processing_time_minutes'] / 60)
            automated_hours_used = data['performance_metrics']['total_documents_processed'] * 0.05  # 3 minutes for AI processing
            
            efficiency_gains = {
                'Time Saved (Hours)': manual_hours_saved - automated_hours_used,
                'FTE Equivalent': (manual_hours_saved - automated_hours_used) / (40 * 52),  # 40 hrs/week, 52 weeks/year
                'Speed Improvement': (manual_hours_saved / automated_hours_used) if automated_hours_used > 0 else 0,
                'Error Reduction': f"{100 - data['performance_metrics']['error_rate']:.1f}%"
            }
            
            for metric, value in efficiency_gains.items():
                if isinstance(value, float):
                    if metric == 'Speed Improvement':
                        st.metric(metric, f"{value:.1f}x faster")
                    elif 'FTE' in metric:
                        st.metric(metric, f"{value:.1f} people")
                    else:
                        st.metric(metric, f"{value:,.0f}")
                else:
                    st.metric(metric, value)
        
        with col2:
            st.markdown("#### 💼 Business Impact")
            
            # Business impact metrics
            business_impact = {
                'Process Automation': f"{data['performance_metrics']['accuracy_rate']:.1f}% automated",
                'Quality Improvement': f"{data['performance_metrics']['accuracy_rate']:.1f}% accuracy",
                'Cost Reduction': f"{data['cost_analysis']['roi_percentage']:.1f}% savings",
                'Scalability Factor': "10x capacity increase"
            }
            
            for metric, value in business_impact.items():
                st.metric(metric, value)
        
        # Competitive advantage analysis
        st.markdown("#### 🏆 Competitive Advantage")
        
        advantages = [
            "🚀 **Speed**: 99.5% faster than manual processing",
            "🎯 **Accuracy**: 96.2% accuracy rate with continuous improvement", 
            "💰 **Cost**: 99.5% cost reduction per document processed",
            "📊 **Scale**: Process 1,125+ documents per hour capacity",
            "🔄 **Consistency**: 24/7 processing with minimal human intervention",
            "📈 **Growth**: Scales automatically with business growth",
            "🛡️ **Compliance**: Automated audit trails and compliance reporting",
            "🧠 **Intelligence**: Continuous learning and improvement capabilities"
        ]
        
        col1, col2 = st.columns(2)
        
        for i, advantage in enumerate(advantages):
            col = col1 if i % 2 == 0 else col2
            col.markdown(advantage)
    
    def render_industry_benchmarking(self, data: Dict[str, Any]):
        """Render industry benchmarking analysis"""
        st.markdown("### 📊 Industry Benchmarking")
        
        # Industry comparison data (simulated)
        industry_data = {
            'Metrics': ['Accuracy Rate', 'Processing Speed', 'Cost Efficiency', 'Error Rate', 'Uptime'],
            'Our Performance': [96.2, 95, 99, 4, 99.7],  # Percentile scores
            'Industry Average': [85, 60, 70, 15, 95],
            'Best in Class': [98, 90, 95, 2, 99.9]
        }
        
        benchmark_df = pd.DataFrame(industry_data)
        
        # Radar chart for benchmarking
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=benchmark_df['Our Performance'],
            theta=benchmark_df['Metrics'],
            fill='toself',
            name='Our Performance',
            line_color=self.color_scheme['primary']
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=benchmark_df['Industry Average'],
            theta=benchmark_df['Metrics'],
            fill='toself',
            name='Industry Average',
            line_color=self.color_scheme['warning']
        ))
        
        fig_radar.add_trace(go.Scatterpolar(
            r=benchmark_df['Best in Class'],
            theta=benchmark_df['Metrics'],
            fill='toself',
            name='Best in Class',
            line_color=self.color_scheme['success']
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Industry Performance Benchmarking",
            height=500
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Performance ranking
        st.markdown("#### 🏅 Performance Rankings")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🥇 Leading Areas**")
            leading_areas = ["Cost Efficiency", "Processing Speed", "Accuracy Rate"]
            for area in leading_areas:
                st.markdown(f"• {area}")
        
        with col2:
            st.markdown("**🎯 On Target**")
            target_areas = ["Uptime", "Scalability"]
            for area in target_areas:
                st.markdown(f"• {area}")
        
        with col3:
            st.markdown("**📈 Improvement Opportunities**")
            improvement_areas = ["Error Rate Reduction", "Advanced Analytics"]
            for area in improvement_areas:
                st.markdown(f"• {area}")