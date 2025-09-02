"""
Document Intelligence Component
Advanced document analysis, anomaly detection, and vendor intelligence
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import seaborn as sns
import matplotlib.pyplot as plt

class DocumentIntelligence:
    """Document intelligence and analysis component"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#2a5298',
            'secondary': '#1e3c72',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'critical': '#8B0000',
            'medium': '#FF8C00',
            'low': '#32CD32'
        }
        
        # Severity color mapping
        self.severity_colors = {
            'critical': self.color_scheme['critical'],
            'high': self.color_scheme['danger'],
            'medium': self.color_scheme['warning'], 
            'low': self.color_scheme['success']
        }
    
    def render_document_breakdown(self, data: Dict[str, Any]):
        """Render comprehensive document type analysis"""
        st.markdown("### 📋 Document Type Intelligence")
        
        doc_types = data['document_types']
        
        # Convert to DataFrame for easier manipulation
        doc_df = pd.DataFrame.from_dict(doc_types, orient='index')
        doc_df['document_type'] = doc_df.index
        doc_df['document_type_clean'] = doc_df['document_type'].str.replace('_', ' ').str.title()
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_docs = doc_df['count'].sum()
            st.metric("Total Documents", f"{total_docs:,}")
        
        with col2:
            avg_accuracy = doc_df['accuracy'].mean()
            st.metric("Average Accuracy", f"{avg_accuracy:.1f}%")
        
        with col3:
            fastest_type = doc_df.loc[doc_df['avg_processing_time'].idxmin(), 'document_type_clean']
            fastest_time = doc_df['avg_processing_time'].min()
            st.metric("Fastest Processing", f"{fastest_type} ({fastest_time:.1f}s)")
        
        with col4:
            most_accurate = doc_df.loc[doc_df['accuracy'].idxmax(), 'document_type_clean']
            max_accuracy = doc_df['accuracy'].max()
            st.metric("Most Accurate", f"{most_accurate} ({max_accuracy:.1f}%)")
        
        # Document volume and accuracy visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Document volume treemap
            fig_treemap = px.treemap(
                doc_df,
                values='count',
                names='document_type_clean',
                title='Document Volume Distribution',
                color='accuracy',
                color_continuous_scale='RdYlGn',
                color_continuous_midpoint=95
            )
            
            fig_treemap.update_layout(height=400)
            st.plotly_chart(fig_treemap, use_container_width=True)
        
        with col2:
            # Accuracy vs Volume scatter plot
            fig_scatter = px.scatter(
                doc_df,
                x='count',
                y='accuracy',
                size='avg_processing_time',
                color='document_type_clean',
                title='Accuracy vs Volume Analysis',
                labels={
                    'count': 'Document Volume',
                    'accuracy': 'Accuracy (%)',
                    'avg_processing_time': 'Avg Processing Time (s)'
                },
                hover_data=['avg_processing_time']
            )
            
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Processing performance matrix
        st.markdown("#### 📊 Performance Matrix Analysis")
        
        # Create performance matrix
        fig_matrix = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Processing Time by Document Type',
                'Accuracy Distribution',
                'Volume vs Accuracy Correlation',
                'Processing Efficiency Score'
            )
        )
        
        # Processing time bar chart
        fig_matrix.add_trace(
            go.Bar(
                x=doc_df['document_type_clean'],
                y=doc_df['avg_processing_time'],
                name='Processing Time',
                marker_color=self.color_scheme['info'],
                text=[f"{time:.1f}s" for time in doc_df['avg_processing_time']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Accuracy distribution
        fig_matrix.add_trace(
            go.Box(
                y=doc_df['accuracy'],
                name='Accuracy Distribution',
                marker_color=self.color_scheme['success']
            ),
            row=1, col=2
        )
        
        # Volume vs Accuracy correlation
        fig_matrix.add_trace(
            go.Scatter(
                x=doc_df['count'],
                y=doc_df['accuracy'],
                mode='markers+text',
                text=doc_df['document_type_clean'],
                textposition='top center',
                name='Volume/Accuracy',
                marker=dict(size=10, color=self.color_scheme['primary'])
            ),
            row=2, col=1
        )
        
        # Efficiency score (accuracy / processing_time)
        efficiency_score = doc_df['accuracy'] / doc_df['avg_processing_time']
        fig_matrix.add_trace(
            go.Bar(
                x=doc_df['document_type_clean'],
                y=efficiency_score,
                name='Efficiency Score',
                marker_color=self.color_scheme['warning'],
                text=[f"{score:.1f}" for score in efficiency_score],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        fig_matrix.update_layout(height=700, showlegend=False)
        fig_matrix.update_xaxes(tickangle=45, row=1, col=1)
        fig_matrix.update_xaxes(tickangle=45, row=2, col=2)
        
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Document type insights
        st.markdown("#### 🔍 Document Type Insights")
        
        insights = []
        
        # High volume types
        high_volume = doc_df.nlargest(3, 'count')
        insights.append(f"📊 **High Volume Types**: {', '.join(high_volume['document_type_clean'].values)}")
        
        # High accuracy types
        high_accuracy = doc_df.nlargest(3, 'accuracy')
        insights.append(f"🎯 **Most Accurate**: {', '.join(high_accuracy['document_type_clean'].values)}")
        
        # Fast processing types
        fast_processing = doc_df.nsmallest(3, 'avg_processing_time')
        insights.append(f"⚡ **Fastest Processing**: {', '.join(fast_processing['document_type_clean'].values)}")
        
        for insight in insights:
            st.markdown(insight)
    
    def render_anomaly_detection(self, data: Dict[str, Any]):
        """Render anomaly detection and analysis"""
        st.markdown("### 🚨 Anomaly Detection & Analysis")
        
        anomalies = data['anomalies']
        anomaly_df = pd.DataFrame(anomalies)
        
        # Anomaly summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_anomalies = anomaly_df['count'].sum()
            st.metric("Total Anomalies", total_anomalies)
        
        with col2:
            critical_anomalies = anomaly_df[anomaly_df['severity'] == 'critical']['count'].sum()
            st.metric("Critical Issues", critical_anomalies, delta="Requires immediate attention")
        
        with col3:
            high_anomalies = anomaly_df[anomaly_df['severity'] == 'high']['count'].sum()
            st.metric("High Priority", high_anomalies, delta="Review within 24h")
        
        with col4:
            anomaly_rate = (total_anomalies / data['performance_metrics']['total_documents_processed']) * 100
            st.metric("Anomaly Rate", f"{anomaly_rate:.2f}%", delta="Of total documents")
        
        # Anomaly visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Anomaly count by type
            fig_anomaly_count = px.bar(
                anomaly_df,
                x='type',
                y='count',
                color='severity',
                color_discrete_map=self.severity_colors,
                title='Anomalies by Type and Severity',
                labels={'type': 'Anomaly Type', 'count': 'Count'}
            )
            
            fig_anomaly_count.update_xaxes(tickangle=45)
            fig_anomaly_count.update_layout(height=400)
            st.plotly_chart(fig_anomaly_count, use_container_width=True)
        
        with col2:
            # Severity distribution
            severity_counts = anomaly_df.groupby('severity')['count'].sum().reset_index()
            
            fig_severity = px.pie(
                severity_counts,
                values='count',
                names='severity',
                title='Anomaly Severity Distribution',
                color='severity',
                color_discrete_map=self.severity_colors
            )
            
            fig_severity.update_layout(height=400)
            st.plotly_chart(fig_severity, use_container_width=True)
        
        # Anomaly trend analysis
        st.markdown("#### 📈 Anomaly Trend Analysis")
        
        # Generate trend data (simulated)
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        anomaly_trends = []
        
        for date in dates:
            for _, anomaly in anomaly_df.iterrows():
                daily_count = np.random.poisson(anomaly['count'] / 30)  # Distribute monthly count across days
                anomaly_trends.append({
                    'date': date,
                    'anomaly_type': anomaly['type'],
                    'severity': anomaly['severity'],
                    'count': daily_count
                })
        
        trend_df = pd.DataFrame(anomaly_trends)
        daily_trend = trend_df.groupby(['date', 'severity'])['count'].sum().reset_index()
        
        # Stacked area chart for anomaly trends
        fig_trend = px.area(
            daily_trend,
            x='date',
            y='count',
            color='severity',
            color_discrete_map=self.severity_colors,
            title='Daily Anomaly Trends by Severity'
        )
        
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Detailed anomaly analysis
        st.markdown("#### 🔍 Detailed Anomaly Analysis")
        
        # Anomaly details table
        anomaly_details = []
        for _, anomaly in anomaly_df.iterrows():
            impact_level = {
                'critical': 'Business Critical - Immediate Action Required',
                'high': 'High Impact - Review Within 24 Hours', 
                'medium': 'Medium Impact - Weekly Review',
                'low': 'Low Impact - Monthly Review'
            }
            
            recommendations = {
                'duplicate_invoice': 'Implement duplicate detection rules',
                'amount_mismatch': 'Enhance validation algorithms',
                'missing_po_number': 'Add mandatory field validation',
                'invalid_vendor': 'Update vendor master data',
                'date_inconsistency': 'Implement date format standardization',
                'currency_mismatch': 'Add currency validation checks'
            }
            
            anomaly_details.append({
                'Anomaly Type': anomaly['type'].replace('_', ' ').title(),
                'Count': anomaly['count'],
                'Severity': anomaly['severity'].title(),
                'Impact': impact_level.get(anomaly['severity'], 'Unknown'),
                'Recommendation': recommendations.get(anomaly['type'], 'Review and investigate')
            })
        
        anomaly_details_df = pd.DataFrame(anomaly_details)
        st.dataframe(anomaly_details_df, use_container_width=True)
        
        # Anomaly resolution tracking
        st.markdown("#### ⚡ Resolution Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚨 Immediate Actions (Critical)**")
            critical_anomalies_list = anomaly_df[anomaly_df['severity'] == 'critical']
            for _, anomaly in critical_anomalies_list.iterrows():
                st.markdown(f"• {anomaly['type'].replace('_', ' ').title()}: {anomaly['count']} occurrences")
        
        with col2:
            st.markdown("**📋 Scheduled Reviews (High/Medium)**")
            other_anomalies = anomaly_df[anomaly_df['severity'].isin(['high', 'medium'])]
            for _, anomaly in other_anomalies.iterrows():
                st.markdown(f"• {anomaly['type'].replace('_', ' ').title()}: {anomaly['count']} occurrences ({anomaly['severity']})")
    
    def render_vendor_analysis(self, data: Dict[str, Any]):
        """Render vendor intelligence and spending analysis"""
        st.markdown("### 🏢 Vendor Intelligence & Spending Analysis")
        
        vendor_data = data['vendor_analysis']
        vendor_df = pd.DataFrame(vendor_data)
        
        # Vendor summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_vendors = len(vendor_df)
            st.metric("Active Vendors", total_vendors)
        
        with col2:
            total_spending = vendor_df['total_amount'].sum()
            st.metric("Total Spending", f"${total_spending:,.2f}")
        
        with col3:
            total_documents = vendor_df['document_count'].sum()
            st.metric("Total Transactions", f"{total_documents:,}")
        
        with col4:
            avg_transaction = total_spending / total_documents if total_documents > 0 else 0
            st.metric("Avg Transaction", f"${avg_transaction:.2f}")
        
        # Vendor spending visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Spending by vendor
            fig_spending = px.bar(
                vendor_df,
                x='vendor',
                y='total_amount',
                title='Total Spending by Vendor',
                color='total_amount',
                color_continuous_scale='Blues'
            )
            
            fig_spending.update_xaxes(tickangle=45)
            fig_spending.update_layout(height=400)
            st.plotly_chart(fig_spending, use_container_width=True)
        
        with col2:
            # Transaction volume vs spending
            fig_volume_spend = px.scatter(
                vendor_df,
                x='document_count',
                y='total_amount',
                size='avg_amount',
                color='vendor',
                title='Transaction Volume vs Total Spending',
                labels={
                    'document_count': 'Number of Transactions',
                    'total_amount': 'Total Spending ($)',
                    'avg_amount': 'Average Amount'
                }
            )
            
            fig_volume_spend.update_layout(height=400)
            st.plotly_chart(fig_volume_spend, use_container_width=True)
        
        # Vendor performance analysis
        st.markdown("#### 📊 Vendor Performance Analysis")
        
        # Add performance metrics to vendor data
        vendor_df['spending_rank'] = vendor_df['total_amount'].rank(ascending=False)
        vendor_df['volume_rank'] = vendor_df['document_count'].rank(ascending=False)
        vendor_df['efficiency_score'] = vendor_df['total_amount'] / vendor_df['document_count']
        
        # Performance matrix
        fig_performance = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Average Transaction Amount',
                'Processing Accuracy by Vendor',
                'Spending Distribution',
                'Transaction Frequency'
            )
        )
        
        # Average transaction amount
        fig_performance.add_trace(
            go.Bar(
                x=vendor_df['vendor'],
                y=vendor_df['avg_amount'],
                name='Avg Amount',
                marker_color=self.color_scheme['primary'],
                text=[f"${amt:.2f}" for amt in vendor_df['avg_amount']],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Processing accuracy (if available)
        if 'accuracy_rate' in vendor_df.columns:
            fig_performance.add_trace(
                go.Scatter(
                    x=vendor_df['vendor'],
                    y=vendor_df['accuracy_rate'],
                    mode='markers+lines',
                    name='Accuracy Rate',
                    line=dict(color=self.color_scheme['success'])
                ),
                row=1, col=2
            )
        else:
            # Simulate accuracy data
            vendor_df['accuracy_rate'] = np.random.uniform(92, 98, len(vendor_df))
            fig_performance.add_trace(
                go.Scatter(
                    x=vendor_df['vendor'],
                    y=vendor_df['accuracy_rate'],
                    mode='markers+lines',
                    name='Accuracy Rate',
                    line=dict(color=self.color_scheme['success'])
                ),
                row=1, col=2
            )
        
        # Spending distribution pie
        fig_performance.add_trace(
            go.Pie(
                labels=vendor_df['vendor'],
                values=vendor_df['total_amount'],
                name="Spending Share"
            ),
            row=2, col=1
        )
        
        # Transaction frequency
        fig_performance.add_trace(
            go.Bar(
                x=vendor_df['vendor'],
                y=vendor_df['document_count'],
                name='Transaction Count',
                marker_color=self.color_scheme['info'],
                text=vendor_df['document_count'],
                textposition='auto'
            ),
            row=2, col=2
        )
        
        fig_performance.update_layout(height=700, showlegend=False)
        fig_performance.update_xaxes(tickangle=45, row=1, col=1)
        fig_performance.update_xaxes(tickangle=45, row=1, col=2)
        fig_performance.update_xaxes(tickangle=45, row=2, col=2)
        
        st.plotly_chart(fig_performance, use_container_width=True)
        
        # Vendor insights and recommendations
        st.markdown("#### 💡 Vendor Insights & Recommendations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Top Performers**")
            
            # Top spending vendors
            top_spending = vendor_df.nlargest(3, 'total_amount')
            st.markdown("*By Total Spending:*")
            for _, vendor in top_spending.iterrows():
                st.markdown(f"• **{vendor['vendor']}**: ${vendor['total_amount']:,.2f}")
            
            # High volume vendors
            high_volume = vendor_df.nlargest(3, 'document_count')
            st.markdown("*By Transaction Volume:*")
            for _, vendor in high_volume.iterrows():
                st.markdown(f"• **{vendor['vendor']}**: {vendor['document_count']:,} transactions")
        
        with col2:
            st.markdown("**🎯 Optimization Opportunities**")
            
            # High average transaction vendors
            high_avg = vendor_df.nlargest(3, 'avg_amount')
            st.markdown("*High-Value Relationships:*")
            for _, vendor in high_avg.iterrows():
                st.markdown(f"• **{vendor['vendor']}**: ${vendor['avg_amount']:.2f} avg")
            
            # Processing efficiency opportunities
            low_accuracy = vendor_df.nsmallest(2, 'accuracy_rate')
            st.markdown("*Accuracy Improvement Needed:*")
            for _, vendor in low_accuracy.iterrows():
                st.markdown(f"• **{vendor['vendor']}**: {vendor['accuracy_rate']:.1f}% accuracy")
        
        # Vendor relationship matrix
        st.markdown("#### 🤝 Vendor Relationship Matrix")
        
        # Create relationship matrix based on spending and volume
        vendor_df['relationship_score'] = (
            (vendor_df['total_amount'] / vendor_df['total_amount'].max() * 50) +
            (vendor_df['document_count'] / vendor_df['document_count'].max() * 50)
        )
        
        vendor_df['relationship_category'] = pd.cut(
            vendor_df['relationship_score'],
            bins=[0, 25, 50, 75, 100],
            labels=['Transactional', 'Operational', 'Strategic', 'Critical']
        )
        
        # Relationship matrix visualization
        fig_matrix = px.scatter(
            vendor_df,
            x='document_count',
            y='total_amount',
            color='relationship_category',
            size='avg_amount',
            hover_name='vendor',
            title='Vendor Relationship Matrix',
            labels={
                'document_count': 'Transaction Volume',
                'total_amount': 'Total Spending ($)'
            }
        )
        
        fig_matrix.update_layout(height=500)
        st.plotly_chart(fig_matrix, use_container_width=True)
        
        # Vendor summary table
        st.markdown("#### 📋 Vendor Performance Summary")
        
        summary_df = vendor_df[['vendor', 'total_amount', 'document_count', 'avg_amount', 'accuracy_rate', 'relationship_category']].copy()
        summary_df.columns = ['Vendor', 'Total Spending', 'Transactions', 'Avg Amount', 'Accuracy %', 'Relationship Type']
        summary_df = summary_df.sort_values('Total Spending', ascending=False)
        
        # Format the summary table
        summary_df['Total Spending'] = summary_df['Total Spending'].apply(lambda x: f"${x:,.2f}")
        summary_df['Transactions'] = summary_df['Transactions'].apply(lambda x: f"{x:,}")
        summary_df['Avg Amount'] = summary_df['Avg Amount'].apply(lambda x: f"${x:.2f}")
        summary_df['Accuracy %'] = summary_df['Accuracy %'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(summary_df, use_container_width=True)