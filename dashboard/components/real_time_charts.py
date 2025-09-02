"""
Real-Time Charts Component
Impressive real-time visualizations including 3D charts, animations, and live updates
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time
import math

class RealTimeCharts:
    """Real-time visualization and 3D chart component"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#2a5298',
            'secondary': '#1e3c72',
            'success': '#28a745',
            'warning': '#ffc107',
            'danger': '#dc3545',
            'info': '#17a2b8',
            'gradient_start': '#667eea',
            'gradient_end': '#764ba2'
        }
        
        # 3D color schemes
        self.color_scales = {
            'performance': ['#FFF7FB', '#ECE7F2', '#D0D1E6', '#A6BDDB', '#74A9CF', '#3690C0', '#0570B0', '#045A8D', '#023858'],
            'accuracy': ['#FFFFE5', '#F7FCB9', '#D9F0A3', '#ADDD8E', '#78C679', '#41AB5D', '#238443', '#006837', '#004529'],
            'volume': ['#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
        }
    
    def render_throughput_chart(self, data: Dict[str, Any]):
        """Render real-time throughput visualization"""
        st.markdown("#### ⚡ Real-Time Processing Throughput")
        
        # Generate real-time throughput data
        current_time = datetime.now()
        time_points = []
        throughput_values = []
        queue_sizes = []
        
        # Generate last 50 data points (5-minute intervals)
        for i in range(50):
            time_point = current_time - timedelta(seconds=i*6)  # 6-second intervals
            
            # Add some realistic patterns
            base_throughput = 20
            time_factor = math.sin(i * 0.1) * 3  # Oscillation
            random_factor = np.random.uniform(-2, 2)
            
            throughput = max(0, base_throughput + time_factor + random_factor)
            queue_size = max(0, data['system_health']['processing_queue_size'] + np.random.randint(-5, 5))
            
            time_points.append(time_point)
            throughput_values.append(throughput)
            queue_sizes.append(queue_size)
        
        # Reverse to get chronological order
        time_points.reverse()
        throughput_values.reverse()
        queue_sizes.reverse()
        
        # Create dual-axis chart
        fig_throughput = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=("Real-Time Processing Monitor",)
        )
        
        # Throughput line with gradient fill
        fig_throughput.add_trace(
            go.Scatter(
                x=time_points,
                y=throughput_values,
                mode='lines',
                name='Documents/Min',
                line=dict(color=self.color_scheme['primary'], width=3),
                fill='tonexty',
                fillcolor=f"rgba(42, 82, 152, 0.1)",
                hovertemplate='<b>Throughput</b><br>%{y:.1f} docs/min<br>%{x}<extra></extra>'
            ),
            secondary_y=False
        )
        
        # Queue size as bars
        fig_throughput.add_trace(
            go.Bar(
                x=time_points,
                y=queue_sizes,
                name='Queue Size',
                marker_color=self.color_scheme['warning'],
                opacity=0.6,
                yaxis='y2',
                hovertemplate='<b>Queue Size</b><br>%{y} documents<br>%{x}<extra></extra>'
            ),
            secondary_y=True
        )
        
        # Add trend line
        z = np.polyfit(range(len(throughput_values)), throughput_values, 1)
        p = np.poly1d(z)
        trend_line = p(range(len(throughput_values)))
        
        fig_throughput.add_trace(
            go.Scatter(
                x=time_points,
                y=trend_line,
                mode='lines',
                name='Trend',
                line=dict(color=self.color_scheme['danger'], width=2, dash='dash'),
                hovertemplate='<b>Trend</b><br>%{y:.1f} docs/min<br><extra></extra>'
            ),
            secondary_y=False
        )
        
        # Update layout
        fig_throughput.update_layout(
            height=400,
            hovermode='x unified',
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Set y-axes titles
        fig_throughput.update_yaxes(title_text="Documents/Minute", secondary_y=False)
        fig_throughput.update_yaxes(title_text="Queue Size", secondary_y=True)
        fig_throughput.update_xaxes(title_text="Time")
        
        st.plotly_chart(fig_throughput, use_container_width=True)
        
        # Real-time metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            current_throughput = throughput_values[-1]
            avg_throughput = np.mean(throughput_values[-10:])  # Last 10 points
            delta_throughput = current_throughput - avg_throughput
            st.metric(
                "Current Throughput",
                f"{current_throughput:.1f}/min",
                delta=f"{delta_throughput:+.1f}"
            )
        
        with col2:
            current_queue = queue_sizes[-1]
            avg_queue = np.mean(queue_sizes[-10:])
            delta_queue = current_queue - avg_queue
            st.metric(
                "Current Queue",
                f"{current_queue}",
                delta=f"{delta_queue:+.0f}"
            )
        
        with col3:
            trend_direction = "📈" if z[0] > 0 else "📉" if z[0] < 0 else "➡️"
            st.metric(
                "Trend",
                f"{trend_direction}",
                delta=f"{z[0]*10:.2f}/min per 10 intervals"
            )
    
    def render_3d_processing_flow(self, data: Dict[str, Any]):
        """Render 3D processing flow visualization"""
        st.markdown("#### 🌊 3D Processing Flow Visualization")
        
        # Generate 3D flow data
        doc_types = list(data['document_types'].keys())
        
        # Create 3D surface data
        x = np.linspace(0, 10, 50)  # Time axis
        y = np.linspace(0, len(doc_types), len(doc_types))  # Document types
        
        # Generate processing flow surface
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        
        for i, doc_type in enumerate(doc_types):
            doc_data = data['document_types'][doc_type]
            base_volume = doc_data['count'] / 1000  # Scale down
            
            # Create wave pattern for this document type
            for j, time_point in enumerate(x):
                wave = base_volume * (1 + 0.3 * np.sin(time_point + i))
                noise = np.random.uniform(-0.1, 0.1) * base_volume
                Z[i, j] = max(0, wave + noise)
        
        # Create 3D surface plot
        fig_3d_flow = go.Figure(data=[
            go.Surface(
                z=Z,
                x=x,
                y=[doc_type.replace('_', ' ').title() for doc_type in doc_types],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Processing Volume", titleside="right"),
                hovertemplate=
                '<b>Document Type:</b> %{y}<br>' +
                '<b>Time:</b> %{x:.1f}<br>' +
                '<b>Volume:</b> %{z:.2f}<br>' +
                '<extra></extra>'
            )
        ])
        
        fig_3d_flow.update_layout(
            title='3D Document Processing Flow',
            scene=dict(
                xaxis_title='Time Progression',
                yaxis_title='Document Types', 
                zaxis_title='Processing Volume',
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=1.2)
                ),
                bgcolor='rgba(0,0,0,0)'
            ),
            height=500
        )
        
        st.plotly_chart(fig_3d_flow, use_container_width=True)
        
        # 3D scatter plot of processing metrics
        st.markdown("#### 📊 3D Performance Metrics")
        
        # Prepare 3D scatter data
        scatter_data = []
        for doc_type, metrics in data['document_types'].items():
            scatter_data.append({
                'document_type': doc_type.replace('_', ' ').title(),
                'volume': metrics['count'],
                'accuracy': metrics['accuracy'],
                'speed': 1 / metrics['avg_processing_time'],  # Inverse for "speed"
                'efficiency': metrics['accuracy'] / metrics['avg_processing_time']
            })
        
        scatter_df = pd.DataFrame(scatter_data)
        
        fig_3d_scatter = go.Figure(data=[
            go.Scatter3d(
                x=scatter_df['volume'],
                y=scatter_df['accuracy'],
                z=scatter_df['speed'],
                mode='markers+text',
                marker=dict(
                    size=scatter_df['efficiency'] * 2,
                    color=scatter_df['efficiency'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Efficiency Score"),
                    opacity=0.8
                ),
                text=scatter_df['document_type'],
                textposition="middle center",
                hovertemplate=
                '<b>%{text}</b><br>' +
                '<b>Volume:</b> %{x:,}<br>' +
                '<b>Accuracy:</b> %{y:.1f}%<br>' +
                '<b>Speed:</b> %{z:.2f}<br>' +
                '<extra></extra>'
            )
        ])
        
        fig_3d_scatter.update_layout(
            title='3D Performance Analysis (Volume × Accuracy × Speed)',
            scene=dict(
                xaxis_title='Document Volume',
                yaxis_title='Accuracy (%)',
                zaxis_title='Processing Speed',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            height=500
        )
        
        st.plotly_chart(fig_3d_scatter, use_container_width=True)
    
    def render_3d_accuracy_surface(self, data: Dict[str, Any]):
        """Render 3D accuracy surface visualization"""
        st.markdown("#### 🎯 3D Accuracy Landscape")
        
        # Generate accuracy surface data
        doc_types = list(data['document_types'].keys())
        time_periods = 24  # Hours
        
        # Create meshgrid
        x = np.linspace(0, 23, time_periods)  # Hours of day
        y = np.linspace(0, len(doc_types)-1, len(doc_types))  # Document type indices
        X, Y = np.meshgrid(x, y)
        
        # Generate accuracy surface with patterns
        Z = np.zeros_like(X)
        
        for i, doc_type in enumerate(doc_types):
            base_accuracy = data['document_types'][doc_type]['accuracy']
            
            for j, hour in enumerate(x):
                # Add hourly patterns (lower accuracy during night hours)
                hour_factor = 0.95 + 0.05 * np.cos(2 * np.pi * hour / 24)  # Peak at noon
                
                # Add some document-type specific patterns
                type_pattern = 1 + 0.02 * np.sin(i * np.pi / len(doc_types))
                
                # Random variation
                noise = np.random.uniform(-0.5, 0.5)
                
                Z[i, j] = base_accuracy * hour_factor * type_pattern + noise
        
        # Create 3D surface
        fig_accuracy_surface = go.Figure(data=[
            go.Surface(
                z=Z,
                x=x,
                y=[doc_type.replace('_', ' ').title() for doc_type in doc_types],
                colorscale='RdYlGn',
                colorbar=dict(title="Accuracy (%)", titleside="right"),
                hovertemplate=
                '<b>Hour:</b> %{x:.0f}:00<br>' +
                '<b>Document Type:</b> %{y}<br>' +
                '<b>Accuracy:</b> %{z:.1f}%<br>' +
                '<extra></extra>'
            )
        ])
        
        fig_accuracy_surface.update_layout(
            title='24-Hour Accuracy Landscape by Document Type',
            scene=dict(
                xaxis_title='Hour of Day',
                yaxis_title='Document Types',
                zaxis_title='Accuracy (%)',
                camera=dict(
                    eye=dict(x=1.3, y=1.3, z=1.0)
                )
            ),
            height=500
        )
        
        st.plotly_chart(fig_accuracy_surface, use_container_width=True)
    
    def render_animated_timeline(self, data: Dict[str, Any]):
        """Render animated timeline visualization"""
        st.markdown("#### 🎬 Animated Processing Timeline")
        
        # Generate timeline data
        daily_data = data['daily_processing']
        df = pd.DataFrame(daily_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Create animated scatter plot
        fig_animated = px.scatter(
            df,
            x='date',
            y='documents',
            size='cost_savings',
            color='accuracy',
            animation_frame=df.index,
            animation_group='date',
            hover_name='date',
            size_max=30,
            color_continuous_scale='Viridis',
            title='Animated Daily Processing Performance',
            labels={
                'documents': 'Documents Processed',
                'accuracy': 'Accuracy (%)',
                'cost_savings': 'Cost Savings ($)'
            }
        )
        
        fig_animated.update_layout(height=500)
        
        # Note: Streamlit doesn't support Plotly animations well, so we'll show static version
        # In a full implementation, this could be enhanced with custom JavaScript
        st.plotly_chart(fig_animated, use_container_width=True)
        
        st.info("💡 **Note**: In a production environment, this chart would show real-time animated updates as new data arrives.")
    
    def render_heatmap_analysis(self, data: Dict[str, Any]):
        """Render advanced heatmap analysis"""
        st.markdown("#### 🔥 Performance Heatmap Analysis")
        
        # Create performance matrix
        doc_types = list(data['document_types'].keys())
        metrics = ['Volume', 'Accuracy', 'Speed', 'Cost Efficiency']
        
        # Normalize data to 0-100 scale for heatmap
        heatmap_data = []
        
        for doc_type in doc_types:
            doc_data = data['document_types'][doc_type]
            
            # Normalize each metric
            volume_score = min(100, (doc_data['count'] / 20000) * 100)  # Assuming 20k is max
            accuracy_score = doc_data['accuracy']
            speed_score = max(0, 100 - (doc_data['avg_processing_time'] * 10))  # Inverse scoring
            cost_score = max(0, 100 - (doc_data['avg_processing_time'] * 15))  # Cost efficiency
            
            heatmap_data.append([volume_score, accuracy_score, speed_score, cost_score])
        
        # Create heatmap
        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=metrics,
            y=[doc_type.replace('_', ' ').title() for doc_type in doc_types],
            colorscale='RdYlGn',
            text=[[f'{val:.1f}' for val in row] for row in heatmap_data],
            texttemplate='%{text}',
            textfont={"size": 12},
            hovertemplate=
            '<b>Document Type:</b> %{y}<br>' +
            '<b>Metric:</b> %{x}<br>' +
            '<b>Score:</b> %{z:.1f}<br>' +
            '<extra></extra>'
        ))
        
        fig_heatmap.update_layout(
            title='Document Type Performance Heatmap',
            xaxis_title='Performance Metrics',
            yaxis_title='Document Types',
            height=400
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Performance radar charts for top performers
        col1, col2 = st.columns(2)
        
        # Get top 2 performers by volume
        sorted_types = sorted(data['document_types'].items(), key=lambda x: x[1]['count'], reverse=True)
        
        for i, (doc_type, metrics_data) in enumerate(sorted_types[:2]):
            col = col1 if i == 0 else col2
            
            with col:
                # Radar chart for this document type
                categories = ['Volume', 'Accuracy', 'Speed', 'Efficiency']
                values = heatmap_data[doc_types.index(doc_type)]
                
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=doc_type.replace('_', ' ').title(),
                    line_color=self.color_scheme['primary'] if i == 0 else self.color_scheme['info']
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 100]
                        )),
                    showlegend=True,
                    title=f"{doc_type.replace('_', ' ').title()} Performance",
                    height=350
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    
    def render_geographic_analysis(self, data: Dict[str, Any]):
        """Render geographic document processing analysis"""
        st.markdown("#### 🌍 Geographic Processing Distribution")
        
        # Simulate geographic data
        locations = [
            {'city': 'New York', 'lat': 40.7128, 'lon': -74.0060, 'documents': 12500, 'accuracy': 96.8},
            {'city': 'London', 'lat': 51.5074, 'lon': -0.1278, 'documents': 8900, 'accuracy': 95.2},
            {'city': 'Tokyo', 'lat': 35.6762, 'lon': 139.6503, 'documents': 6700, 'accuracy': 97.1},
            {'city': 'Sydney', 'lat': -33.8688, 'lon': 151.2093, 'documents': 4200, 'accuracy': 94.8},
            {'city': 'Toronto', 'lat': 43.6532, 'lon': -79.3832, 'documents': 3800, 'accuracy': 96.3},
            {'city': 'Frankfurt', 'lat': 50.1109, 'lon': 8.6821, 'documents': 2900, 'accuracy': 95.7}
        ]
        
        geo_df = pd.DataFrame(locations)
        
        # Create geographic visualization
        fig_geo = go.Figure()
        
        # Add markers for each location
        fig_geo.add_trace(go.Scattergeo(
            lon=geo_df['lon'],
            lat=geo_df['lat'],
            text=geo_df['city'],
            mode='markers+text',
            marker=dict(
                size=geo_df['documents'] / 200,  # Scale marker size
                color=geo_df['accuracy'],
                colorscale='RdYlGn',
                cmin=90,
                cmax=100,
                colorbar=dict(title="Accuracy (%)"),
                line=dict(width=1, color='white')
            ),
            textposition="bottom center",
            hovertemplate=
            '<b>%{text}</b><br>' +
            'Documents: %{marker.size}<br>' +
            'Accuracy: %{marker.color:.1f}%<br>' +
            '<extra></extra>'
        ))
        
        fig_geo.update_layout(
            title='Global Document Processing Centers',
            geo=dict(
                projection_type='natural earth',
                showland=True,
                landcolor='lightgray',
                showocean=True,
                oceancolor='lightblue'
            ),
            height=500
        )
        
        st.plotly_chart(fig_geo, use_container_width=True)
        
        # Processing volume by region
        col1, col2 = st.columns(2)
        
        with col1:
            # Regional volume chart
            regions = ['North America', 'Europe', 'Asia Pacific', 'Others']
            regional_volumes = [16300, 11800, 10900, 7000]  # Sum from locations
            
            fig_regional = px.pie(
                values=regional_volumes,
                names=regions,
                title='Processing Volume by Region',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            
            st.plotly_chart(fig_regional, use_container_width=True)
        
        with col2:
            # Accuracy by region
            regional_accuracy = [96.5, 95.4, 96.0, 94.2]
            
            fig_accuracy = go.Figure(data=[
                go.Bar(
                    x=regions,
                    y=regional_accuracy,
                    marker_color=self.color_scheme['success'],
                    text=[f'{acc:.1f}%' for acc in regional_accuracy],
                    textposition='auto'
                )
            ])
            
            fig_accuracy.update_layout(
                title='Average Accuracy by Region',
                yaxis_title='Accuracy (%)',
                height=400
            )
            
            st.plotly_chart(fig_accuracy, use_container_width=True)