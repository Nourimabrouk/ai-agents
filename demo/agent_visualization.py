"""
Agent Visualization System for AI Document Intelligence Platform
3D visualizations of agent networks, swarm behavior, and processing flows
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import time
import random
import math
from typing import Dict, List, Any, Tuple
import json

class AgentVisualizationSystem:
    """Advanced 3D visualization system for AI agents"""
    
    def __init__(self):
        self.agent_network = self._initialize_agent_network()
        self.swarm_state = self._initialize_swarm_state()
        self.processing_flows = self._initialize_processing_flows()
        self.performance_history = self._generate_performance_history()
        
    def _initialize_agent_network(self) -> Dict[str, Any]:
        """Initialize agent network structure"""
        agents = []
        
        # Different agent types
        agent_types = [
            {"type": "Classifier", "count": 8, "color": "#667eea", "specialty": "Document Classification"},
            {"type": "OCR", "count": 12, "color": "#ff6b35", "specialty": "Text Extraction"},
            {"type": "Extractor", "count": 15, "color": "#74b9ff", "specialty": "Data Field Extraction"},
            {"type": "Validator", "count": 6, "color": "#00b894", "specialty": "Data Validation"},
            {"type": "Integrator", "count": 4, "color": "#e17055", "specialty": "ERP Integration"},
            {"type": "Coordinator", "count": 3, "color": "#a29bfe", "specialty": "Task Coordination"}
        ]
        
        agent_id = 0
        for agent_type in agent_types:
            for i in range(agent_type["count"]):
                agent = {
                    "id": agent_id,
                    "name": f"{agent_type['type']}_{i+1:02d}",
                    "type": agent_type["type"],
                    "color": agent_type["color"],
                    "specialty": agent_type["specialty"],
                    "performance": random.uniform(0.85, 0.99),
                    "load": random.uniform(0.3, 0.9),
                    "status": random.choice(["idle", "processing", "competing", "collaborating"]),
                    "position": {
                        "x": random.uniform(-10, 10),
                        "y": random.uniform(-10, 10),
                        "z": random.uniform(-5, 5)
                    },
                    "connections": [],
                    "tasks_completed": random.randint(150, 2500),
                    "success_rate": random.uniform(0.92, 0.99),
                    "specialization_score": random.uniform(0.7, 0.95)
                }
                agents.append(agent)
                agent_id += 1
        
        # Create connections between agents
        for agent in agents:
            # Connect to agents of complementary types
            if agent["type"] == "Classifier":
                # Connect to OCR agents
                ocr_agents = [a for a in agents if a["type"] == "OCR"]
                agent["connections"] = random.sample(ocr_agents, min(4, len(ocr_agents)))
            elif agent["type"] == "OCR":
                # Connect to Extractor agents
                extractor_agents = [a for a in agents if a["type"] == "Extractor"]
                agent["connections"] = random.sample(extractor_agents, min(3, len(extractor_agents)))
            elif agent["type"] == "Extractor":
                # Connect to Validator agents
                validator_agents = [a for a in agents if a["type"] == "Validator"]
                agent["connections"] = random.sample(validator_agents, min(2, len(validator_agents)))
            elif agent["type"] == "Validator":
                # Connect to Integrator agents
                integrator_agents = [a for a in agents if a["type"] == "Integrator"]
                agent["connections"] = random.sample(integrator_agents, min(2, len(integrator_agents)))
            
            # Add some random cross-connections for collaboration
            other_agents = [a for a in agents if a["id"] != agent["id"]]
            additional_connections = random.sample(other_agents, random.randint(1, 3))
            agent["connections"].extend(additional_connections)
            agent["connections"] = list({a["id"]: a for a in agent["connections"]}.values())  # Remove duplicates
        
        return {"agents": agents}
    
    def _initialize_swarm_state(self) -> Dict[str, Any]:
        """Initialize swarm intelligence state"""
        return {
            "swarm_size": 20,
            "convergence_threshold": 0.1,
            "learning_rate": 0.05,
            "collaboration_strength": 0.8,
            "competition_intensity": 0.6,
            "emergence_events": [],
            "collective_intelligence": 0.87,
            "swarm_efficiency": 0.94
        }
    
    def _initialize_processing_flows(self) -> Dict[str, Any]:
        """Initialize document processing flow data"""
        return {
            "pipeline_stages": [
                {"name": "Intake", "throughput": 1000, "queue_size": 234, "avg_time": 0.5},
                {"name": "Classification", "throughput": 950, "queue_size": 12, "avg_time": 2.1},
                {"name": "OCR", "throughput": 900, "queue_size": 45, "avg_time": 5.3},
                {"name": "Extraction", "throughput": 850, "queue_size": 67, "avg_time": 3.8},
                {"name": "Validation", "throughput": 800, "queue_size": 23, "avg_time": 2.7},
                {"name": "Integration", "throughput": 750, "queue_size": 8, "avg_time": 8.2},
                {"name": "Completion", "throughput": 750, "queue_size": 0, "avg_time": 0.1}
            ],
            "flow_rate": 750,  # documents per hour
            "error_rate": 0.038,
            "retry_rate": 0.012,
            "success_rate": 0.962
        }
    
    def _generate_performance_history(self) -> List[Dict[str, Any]]:
        """Generate agent performance history"""
        history = []
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30 * 24):  # 30 days of hourly data
            timestamp = base_date + timedelta(hours=i)
            
            # Simulate performance metrics with trends and variance
            base_performance = 0.85 + (i * 0.0001)  # Gradual improvement
            daily_cycle = 0.05 * math.sin(2 * math.pi * i / 24)  # Daily performance cycle
            weekly_cycle = 0.02 * math.sin(2 * math.pi * i / (24 * 7))  # Weekly cycle
            noise = random.uniform(-0.02, 0.02)
            
            performance = base_performance + daily_cycle + weekly_cycle + noise
            performance = max(0.8, min(0.99, performance))  # Clamp values
            
            history.append({
                "timestamp": timestamp,
                "avg_performance": performance,
                "throughput": 150 + random.randint(-20, 30),
                "accuracy": 96.2 + random.uniform(-1.5, 1.5),
                "agent_count": 48 + random.randint(-2, 3),
                "collaboration_events": random.randint(5, 25),
                "competition_events": random.randint(2, 15),
                "emergence_events": random.randint(0, 3)
            })
        
        return history

    def create_3d_agent_network(self) -> go.Figure:
        """Create 3D visualization of agent network"""
        agents = self.agent_network["agents"]
        
        fig = go.Figure()
        
        # Create edges (connections between agents)
        edge_x, edge_y, edge_z = [], [], []
        
        for agent in agents:
            for connected_agent in agent["connections"]:
                edge_x.extend([agent["position"]["x"], connected_agent["position"]["x"], None])
                edge_y.extend([agent["position"]["y"], connected_agent["position"]["y"], None])
                edge_z.extend([agent["position"]["z"], connected_agent["position"]["z"], None])
        
        # Add connection lines
        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode='lines',
            line=dict(color='rgba(125, 125, 125, 0.3)', width=2),
            hoverinfo='none',
            name='Agent Connections',
            showlegend=False
        ))
        
        # Group agents by type for color coding
        agent_types = {}
        for agent in agents:
            agent_type = agent["type"]
            if agent_type not in agent_types:
                agent_types[agent_type] = {
                    "x": [], "y": [], "z": [], "text": [], "colors": [], "sizes": []
                }
            
            agent_types[agent_type]["x"].append(agent["position"]["x"])
            agent_types[agent_type]["y"].append(agent["position"]["y"])
            agent_types[agent_type]["z"].append(agent["position"]["z"])
            
            hover_text = (f"<b>{agent['name']}</b><br>"
                         f"Type: {agent['type']}<br>"
                         f"Performance: {agent['performance']:.1%}<br>"
                         f"Load: {agent['load']:.1%}<br>"
                         f"Status: {agent['status']}<br>"
                         f"Success Rate: {agent['success_rate']:.1%}<br>"
                         f"Tasks Completed: {agent['tasks_completed']}")
            
            agent_types[agent_type]["text"].append(hover_text)
            agent_types[agent_type]["colors"].append(agent["color"])
            agent_types[agent_type]["sizes"].append(agent["performance"] * 20 + 5)
        
        # Add agent nodes by type
        for agent_type, data in agent_types.items():
            fig.add_trace(go.Scatter3d(
                x=data["x"], y=data["y"], z=data["z"],
                mode='markers',
                marker=dict(
                    size=data["sizes"],
                    color=data["colors"][0],  # Use consistent color per type
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                text=data["text"],
                hovertemplate='%{text}<extra></extra>',
                name=f'{agent_type} Agents'
            ))
        
        fig.update_layout(
            title="AI Agent Network - 3D Visualization",
            scene=dict(
                xaxis_title="Network Space X",
                yaxis_title="Network Space Y",
                zaxis_title="Network Space Z",
                bgcolor="rgba(240, 240, 240, 0.1)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='cube'
            ),
            height=700,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.2)",
                borderwidth=1
            )
        )
        
        return fig

    def create_swarm_intelligence_animation(self) -> go.Figure:
        """Create animated swarm intelligence visualization"""
        n_particles = 25
        n_frames = 50
        
        # Initialize swarm positions
        swarm_data = []
        
        # Objective function (simple 2D landscape)
        def objective_function(x, y):
            return -(x**2 + y**2) + 5 * np.sin(x) + 3 * np.cos(y)
        
        # Create frames for animation
        frames = []
        
        # Initialize particle positions
        particles = []
        for i in range(n_particles):
            particles.append({
                "x": random.uniform(-5, 5),
                "y": random.uniform(-5, 5),
                "vx": random.uniform(-0.1, 0.1),
                "vy": random.uniform(-0.1, 0.1),
                "best_x": 0,
                "best_y": 0,
                "best_fitness": float('-inf')
            })
        
        global_best_x, global_best_y = 0, 0
        global_best_fitness = float('-inf')
        
        for frame in range(n_frames):
            frame_x, frame_y, frame_fitness = [], [], []
            
            # Update particles
            for particle in particles:
                # Evaluate current position
                fitness = objective_function(particle["x"], particle["y"])
                
                # Update personal best
                if fitness > particle["best_fitness"]:
                    particle["best_fitness"] = fitness
                    particle["best_x"] = particle["x"]
                    particle["best_y"] = particle["y"]
                
                # Update global best
                if fitness > global_best_fitness:
                    global_best_fitness = fitness
                    global_best_x = particle["x"]
                    global_best_y = particle["y"]
                
                # Update velocity (simplified PSO)
                w = 0.7  # inertia weight
                c1, c2 = 1.5, 1.5  # acceleration coefficients
                r1, r2 = random.random(), random.random()
                
                particle["vx"] = (w * particle["vx"] + 
                                 c1 * r1 * (particle["best_x"] - particle["x"]) +
                                 c2 * r2 * (global_best_x - particle["x"]))
                particle["vy"] = (w * particle["vy"] + 
                                 c1 * r1 * (particle["best_y"] - particle["y"]) +
                                 c2 * r2 * (global_best_y - particle["y"]))
                
                # Update position
                particle["x"] += particle["vx"]
                particle["y"] += particle["vy"]
                
                # Keep within bounds
                particle["x"] = max(-5, min(5, particle["x"]))
                particle["y"] = max(-5, min(5, particle["y"]))
                
                frame_x.append(particle["x"])
                frame_y.append(particle["y"])
                frame_fitness.append(fitness)
            
            # Create frame data
            frames.append(go.Frame(
                data=[
                    go.Scatter(
                        x=frame_x,
                        y=frame_y,
                        mode='markers',
                        marker=dict(
                            size=[max(5, min(20, f + 10)) for f in frame_fitness],
                            color=frame_fitness,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Fitness Score")
                        ),
                        text=[f"Agent {i+1}<br>Fitness: {f:.2f}" for i, f in enumerate(frame_fitness)],
                        hovertemplate='%{text}<extra></extra>',
                        name='Swarm Agents'
                    ),
                    go.Scatter(
                        x=[global_best_x],
                        y=[global_best_y],
                        mode='markers',
                        marker=dict(
                            size=25,
                            color='red',
                            symbol='star',
                            line=dict(width=2, color='white')
                        ),
                        name='Global Best',
                        hovertemplate=f'Global Best<br>Fitness: {global_best_fitness:.2f}<extra></extra>'
                    )
                ],
                name=f'Frame {frame}'
            ))
        
        # Create the figure
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=[p["x"] for p in particles],
                    y=[p["y"] for p in particles],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='blue',
                        opacity=0.6
                    ),
                    name='Swarm Agents'
                ),
                go.Scatter(
                    x=[global_best_x],
                    y=[global_best_y],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color='red',
                        symbol='star'
                    ),
                    name='Global Best'
                )
            ],
            frames=frames
        )
        
        # Add animation controls
        fig.update_layout(
            title="Swarm Intelligence Optimization Animation",
            xaxis=dict(range=[-5, 5], title="Solution Space X"),
            yaxis=dict(range=[-5, 5], title="Solution Space Y"),
            updatemenus=[{
                "buttons": [
                    {"args": [None, {"frame": {"duration": 200, "redraw": True},
                                    "fromcurrent": True}], "label": "Play", "method": "animate"},
                    {"args": [[None], {"frame": {"duration": 0, "redraw": True},
                                      "mode": "immediate", "transition": {"duration": 0}}],
                     "label": "Pause", "method": "animate"}
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": 0,
                "yanchor": "top"
            }],
            sliders=[{
                "steps": [
                    {"args": [[f"Frame {k}"], {"frame": {"duration": 0, "redraw": True},
                                              "mode": "immediate"}], "label": f"{k}", "method": "animate"}
                    for k in range(n_frames)
                ],
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {"font": {"size": 20}, "prefix": "Frame:", "visible": True, "xanchor": "right"},
                "transition": {"duration": 300, "easing": "cubic-in-out"},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
            }],
            height=600
        )
        
        return fig

    def create_processing_flow_sankey(self) -> go.Figure:
        """Create advanced Sankey diagram for document processing flow"""
        flow_data = self.processing_flows
        stages = flow_data["pipeline_stages"]
        
        # Create nodes
        node_labels = []
        node_colors = []
        
        # Add pipeline stages
        stage_colors = ['#667eea', '#ff6b35', '#74b9ff', '#00b894', '#e17055', '#a29bfe', '#00cec9']
        for i, stage in enumerate(stages):
            node_labels.append(f"{stage['name']}<br>{stage['throughput']}/hr")
            node_colors.append(stage_colors[i % len(stage_colors)])
        
        # Add error handling nodes
        node_labels.extend(["Errors", "Retries", "Success"])
        node_colors.extend(['#d63031', '#fdcb6e', '#00b894'])
        
        # Create links
        source_indices = []
        target_indices = []
        values = []
        link_colors = []
        
        # Main processing flow
        for i in range(len(stages) - 1):
            source_indices.append(i)
            target_indices.append(i + 1)
            values.append(stages[i + 1]["throughput"])
            link_colors.append('rgba(102, 126, 234, 0.4)')
        
        # Error flows
        total_docs = stages[0]["throughput"]
        errors = int(total_docs * flow_data["error_rate"])
        retries = int(total_docs * flow_data["retry_rate"])
        success = int(total_docs * flow_data["success_rate"])
        
        # Errors from various stages
        error_stages = [1, 2, 3, 4, 5]  # Classification, OCR, Extraction, Validation, Integration
        for stage_idx in error_stages:
            stage_errors = errors // len(error_stages)
            source_indices.append(stage_idx)
            target_indices.append(len(stages))  # Errors node
            values.append(stage_errors)
            link_colors.append('rgba(214, 48, 49, 0.6)')
        
        # Retries flow
        source_indices.append(len(stages))  # From Errors
        target_indices.append(len(stages) + 1)  # To Retries
        values.append(retries)
        link_colors.append('rgba(253, 203, 110, 0.6)')
        
        # Retry back to processing
        source_indices.append(len(stages) + 1)  # From Retries
        target_indices.append(2)  # Back to OCR
        values.append(retries)
        link_colors.append('rgba(253, 203, 110, 0.4)')
        
        # Final success
        source_indices.append(len(stages) - 1)  # From Completion
        target_indices.append(len(stages) + 2)  # To Success
        values.append(success)
        link_colors.append('rgba(0, 184, 148, 0.6)')
        
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=20,
                thickness=25,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=node_colors,
                x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 0.85, 0.85],  # X positions
                y=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 0.8]   # Y positions
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values,
                color=link_colors,
                hovertemplate='%{source.label} → %{target.label}<br>Volume: %{value} docs/hr<extra></extra>'
            )
        )])
        
        fig.update_layout(
            title="Advanced Document Processing Pipeline Flow",
            font_size=10,
            height=500
        )
        
        return fig

    def create_agent_performance_heatmap(self) -> go.Figure:
        """Create agent performance heatmap"""
        agents = self.agent_network["agents"]
        
        # Create performance matrix by agent type and time
        agent_types = ["Classifier", "OCR", "Extractor", "Validator", "Integrator", "Coordinator"]
        hours = list(range(24))
        
        # Generate performance data for each hour and agent type
        performance_matrix = []
        for agent_type in agent_types:
            hourly_performance = []
            type_agents = [a for a in agents if a["type"] == agent_type]
            base_performance = np.mean([a["performance"] for a in type_agents])
            
            for hour in hours:
                # Add daily cycle and some noise
                daily_factor = 0.9 + 0.1 * np.sin(2 * np.pi * (hour - 6) / 24)  # Peak at 12 PM
                hourly_perf = base_performance * daily_factor + random.uniform(-0.02, 0.02)
                hourly_performance.append(max(0.8, min(1.0, hourly_perf)))
            
            performance_matrix.append(hourly_performance)
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=[f"{h:02d}:00" for h in hours],
            y=agent_types,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Performance Score"),
            hovertemplate='Agent Type: %{y}<br>Time: %{x}<br>Performance: %{z:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Agent Performance Heatmap (24-Hour Cycle)",
            xaxis_title="Hour of Day",
            yaxis_title="Agent Type",
            height=400
        )
        
        return fig

    def create_real_time_metrics_dashboard(self) -> go.Figure:
        """Create real-time metrics dashboard"""
        # Generate recent performance data
        recent_data = self.performance_history[-24:]  # Last 24 hours
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Throughput Trends', 'Accuracy Trends', 'Agent Activity', 'Collaboration Events'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"type": "indicator"}, {"type": "bar"}]]
        )
        
        # Throughput trends
        times = [d["timestamp"] for d in recent_data]
        throughput = [d["throughput"] for d in recent_data]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=throughput,
                mode='lines+markers',
                name='Throughput',
                line=dict(color='#667eea', width=2)
            ),
            row=1, col=1
        )
        
        # Accuracy trends
        accuracy = [d["accuracy"] for d in recent_data]
        
        fig.add_trace(
            go.Scatter(
                x=times,
                y=accuracy,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='#ff6b35', width=2)
            ),
            row=1, col=2
        )
        
        # Agent activity gauge
        current_agents = recent_data[-1]["agent_count"]
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_agents,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Active Agents"},
                delta={'reference': 48},
                gauge={
                    'axis': {'range': [None, 60]},
                    'bar': {'color': "#00b894"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 45], 'color': "yellow"},
                        {'range': [45, 60], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ),
            row=2, col=1
        )
        
        # Collaboration events
        event_types = ['Collaboration', 'Competition', 'Emergence']
        event_counts = [
            recent_data[-1]["collaboration_events"],
            recent_data[-1]["competition_events"],
            recent_data[-1]["emergence_events"]
        ]
        
        fig.add_trace(
            go.Bar(
                x=event_types,
                y=event_counts,
                marker_color=['#74b9ff', '#e17055', '#a29bfe'],
                name='Events'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Real-Time Agent System Metrics"
        )
        
        return fig

    def create_emergent_behavior_timeline(self) -> go.Figure:
        """Create timeline visualization of emergent behaviors"""
        # Generate emergent behavior events
        events = [
            {
                "timestamp": datetime.now() - timedelta(hours=23),
                "type": "Auto Load Balancing",
                "description": "OCR agents self-organized to handle spike in image documents",
                "impact": "high",
                "agents_involved": 8
            },
            {
                "timestamp": datetime.now() - timedelta(hours=19),
                "type": "Specialization Discovery",
                "description": "Classifier_03 developed expertise in insurance forms",
                "impact": "medium",
                "agents_involved": 1
            },
            {
                "timestamp": datetime.now() - timedelta(hours=15),
                "type": "Collaborative Optimization",
                "description": "Extractor agents sharing pattern recognition improvements",
                "impact": "high",
                "agents_involved": 6
            },
            {
                "timestamp": datetime.now() - timedelta(hours=11),
                "type": "Error Recovery Swarm",
                "description": "Multiple agents coordinated to handle processing backlog",
                "impact": "high",
                "agents_involved": 12
            },
            {
                "timestamp": datetime.now() - timedelta(hours=7),
                "type": "Meta-Learning Event",
                "description": "System adapted processing strategy based on document patterns",
                "impact": "very_high",
                "agents_involved": 25
            },
            {
                "timestamp": datetime.now() - timedelta(hours=3),
                "type": "Competitive Selection",
                "description": "Best performing agents automatically promoted to handle complex documents",
                "impact": "high",
                "agents_involved": 4
            }
        ]
        
        # Create timeline visualization
        fig = go.Figure()
        
        impact_colors = {
            "low": "#74b9ff",
            "medium": "#fdcb6e", 
            "high": "#ff6b35",
            "very_high": "#d63031"
        }
        
        impact_sizes = {
            "low": 8,
            "medium": 12,
            "high": 16,
            "very_high": 24
        }
        
        for event in events:
            fig.add_trace(go.Scatter(
                x=[event["timestamp"]],
                y=[event["type"]],
                mode='markers',
                marker=dict(
                    size=impact_sizes[event["impact"]],
                    color=impact_colors[event["impact"]],
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                text=event["description"],
                hovertemplate=f'<b>%{{y}}</b><br>%{{text}}<br>Agents: {event["agents_involved"]}<br>Impact: {event["impact"]}<extra></extra>',
                name=event["type"],
                showlegend=False
            ))
        
        fig.update_layout(
            title="Emergent Behavior Timeline (Last 24 Hours)",
            xaxis_title="Time",
            yaxis_title="Behavior Type",
            height=500,
            hovermode='closest'
        )
        
        return fig

def create_agent_visualization_dashboard():
    """Create the comprehensive agent visualization dashboard"""
    st.set_page_config(
        page_title="AI Agent Network - 3D Visualization",
        page_icon="🤖",
        layout="wide"
    )
    
    # Custom CSS for agent visualization
    st.markdown("""
    <style>
    .agent-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-metric {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-active { background-color: #00b894; }
    .status-processing { background-color: #fdcb6e; }
    .status-competing { background-color: #ff6b35; }
    .status-collaborating { background-color: #74b9ff; }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="agent-header">
        <h1>🤖 AI Agent Network Visualization</h1>
        <h3>Real-Time 3D Agent Swarm Intelligence</h3>
        <p>Interactive visualization of multi-agent coordination and emergent behaviors</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize visualization system
    viz_system = AgentVisualizationSystem()
    
    # Real-time status metrics
    st.subheader("🔥 Real-Time Agent Status")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    agents = viz_system.agent_network["agents"]
    total_agents = len(agents)
    avg_performance = np.mean([a["performance"] for a in agents])
    active_agents = len([a for a in agents if a["status"] != "idle"])
    
    with col1:
        st.metric("Total Agents", total_agents)
    with col2:
        st.metric("Active Agents", active_agents, delta=f"{active_agents/total_agents:.1%}")
    with col3:
        st.metric("Avg Performance", f"{avg_performance:.1%}", delta="+2.3pp")
    with col4:
        st.metric("Swarm Efficiency", "94.7%", delta="+1.2pp")
    with col5:
        st.metric("Emergence Events", "12", delta="+5")
    
    # Visualization selector
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "3D Agent Network",
            "Swarm Intelligence Animation", 
            "Processing Flow Analysis",
            "Performance Analytics",
            "Emergent Behavior Timeline"
        ],
        key="viz_type_selector"
    )
    
    if viz_type == "3D Agent Network":
        show_3d_agent_network(viz_system)
    elif viz_type == "Swarm Intelligence Animation":
        show_swarm_intelligence(viz_system)
    elif viz_type == "Processing Flow Analysis":
        show_processing_flow(viz_system)
    elif viz_type == "Performance Analytics":
        show_performance_analytics(viz_system)
    else:  # Emergent Behavior Timeline
        show_emergent_behavior(viz_system)

def show_3d_agent_network(viz_system: AgentVisualizationSystem):
    """Show 3D agent network visualization"""
    st.header("🌐 3D Agent Network Visualization")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main 3D network visualization
        network_fig = viz_system.create_3d_agent_network()
        st.plotly_chart(network_fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Network Statistics")
        
        agents = viz_system.agent_network["agents"]
        
        # Agent type distribution
        agent_types = {}
        for agent in agents:
            agent_type = agent["type"]
            if agent_type not in agent_types:
                agent_types[agent_type] = {"count": 0, "avg_performance": 0}
            agent_types[agent_type]["count"] += 1
            agent_types[agent_type]["avg_performance"] += agent["performance"]
        
        # Calculate averages
        for agent_type in agent_types:
            agent_types[agent_type]["avg_performance"] /= agent_types[agent_type]["count"]
        
        st.markdown("**Agent Distribution:**")
        for agent_type, stats in agent_types.items():
            st.text(f"{agent_type}: {stats['count']} agents ({stats['avg_performance']:.1%})")
        
        st.markdown("**Network Properties:**")
        total_connections = sum(len(a["connections"]) for a in agents)
        avg_connections = total_connections / len(agents)
        st.text(f"Total Connections: {total_connections}")
        st.text(f"Avg Connections/Agent: {avg_connections:.1f}")
        st.text(f"Network Density: {total_connections / (len(agents) * (len(agents) - 1)):.3f}")
        
        # Status indicators
        st.subheader("🚦 Agent Status")
        
        status_counts = {}
        for agent in agents:
            status = agent["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        status_colors = {
            "idle": "status-active",
            "processing": "status-processing", 
            "competing": "status-competing",
            "collaborating": "status-collaborating"
        }
        
        for status, count in status_counts.items():
            percentage = count / len(agents) * 100
            st.markdown(f'<span class="status-indicator {status_colors.get(status, "status-active")}"></span>{status.title()}: {count} ({percentage:.1f}%)', 
                       unsafe_allow_html=True)

def show_swarm_intelligence(viz_system: AgentVisualizationSystem):
    """Show swarm intelligence animation"""
    st.header("🐝 Swarm Intelligence Optimization")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Swarm animation
        swarm_fig = viz_system.create_swarm_intelligence_animation()
        st.plotly_chart(swarm_fig, use_container_width=True)
        
        st.info("🎮 **Interactive Controls**: Use the Play/Pause buttons and slider below the chart to control the animation")
    
    with col2:
        st.subheader("🧠 Swarm Intelligence Metrics")
        
        swarm_state = viz_system.swarm_state
        
        st.metric("Swarm Size", swarm_state["swarm_size"])
        st.metric("Collective Intelligence", f"{swarm_state['collective_intelligence']:.1%}")
        st.metric("Swarm Efficiency", f"{swarm_state['swarm_efficiency']:.1%}")
        st.metric("Collaboration Strength", f"{swarm_state['collaboration_strength']:.1%}")
        st.metric("Competition Intensity", f"{swarm_state['competition_intensity']:.1%}")
        
        st.subheader("⚡ Swarm Behaviors")
        
        behaviors = [
            "🎯 **Global Optimization**: Finding optimal solutions through collective search",
            "🤝 **Information Sharing**: Agents share discoveries with the swarm",
            "🏃 **Adaptive Movement**: Dynamic position updates based on fitness landscape",
            "📈 **Convergence**: Gradual focusing on promising solution regions",
            "🔄 **Exploration vs Exploitation**: Balancing search breadth and depth",
            "🧬 **Emergent Intelligence**: Complex behaviors from simple agent rules"
        ]
        
        for behavior in behaviors:
            st.markdown(behavior)
        
        # Real-time swarm updates
        if st.button("🔄 Update Swarm State", key="update_swarm"):
            st.success("Swarm state updated! New optimization cycle initiated.")
            time.sleep(1)
            st.experimental_rerun()

def show_processing_flow(viz_system: AgentVisualizationSystem):
    """Show processing flow analysis"""
    st.header("🔄 Document Processing Flow Analysis")
    
    # Advanced Sankey diagram
    flow_fig = viz_system.create_processing_flow_sankey()
    st.plotly_chart(flow_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Pipeline Performance")
        
        flow_data = viz_system.processing_flows
        stages = flow_data["pipeline_stages"]
        
        # Create performance table
        performance_data = []
        for i, stage in enumerate(stages):
            if i < len(stages) - 1:
                throughput_loss = stages[i]["throughput"] - stages[i + 1]["throughput"]
                efficiency = (stages[i + 1]["throughput"] / stages[i]["throughput"]) * 100 if stages[i]["throughput"] > 0 else 100
            else:
                throughput_loss = 0
                efficiency = 100
            
            performance_data.append({
                "Stage": stage["name"],
                "Throughput": f"{stage['throughput']}/hr",
                "Queue": stage["queue_size"],
                "Avg Time": f"{stage['avg_time']:.1f}s",
                "Efficiency": f"{efficiency:.1f}%",
                "Loss": throughput_loss
            })
        
        perf_df = pd.DataFrame(performance_data)
        st.dataframe(perf_df, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Flow Optimization")
        
        # Bottleneck analysis
        bottlenecks = []
        for i, stage in enumerate(stages[:-1]):
            queue_ratio = stage["queue_size"] / stage["throughput"] if stage["throughput"] > 0 else 0
            if queue_ratio > 0.05:  # 5% threshold
                bottlenecks.append(stage["name"])
        
        if bottlenecks:
            st.markdown("**🚨 Detected Bottlenecks:**")
            for bottleneck in bottlenecks:
                st.markdown(f"- {bottleneck}")
        else:
            st.success("✅ No significant bottlenecks detected")
        
        # Flow metrics
        st.markdown("**📈 Flow Metrics:**")
        st.metric("Overall Throughput", f"{flow_data['flow_rate']} docs/hr")
        st.metric("Success Rate", f"{flow_data['success_rate']:.1%}")
        st.metric("Error Rate", f"{flow_data['error_rate']:.1%}")
        st.metric("Retry Rate", f"{flow_data['retry_rate']:.1%}")
        
        # Optimization suggestions
        st.markdown("**💡 Optimization Suggestions:**")
        suggestions = [
            "🔧 Scale OCR agents during peak hours",
            "⚡ Implement parallel validation processing",
            "🎯 Add intelligent document routing",
            "📊 Optimize queue management algorithms",
            "🤖 Deploy adaptive load balancing"
        ]
        
        for suggestion in suggestions:
            st.markdown(suggestion)

def show_performance_analytics(viz_system: AgentVisualizationSystem):
    """Show performance analytics dashboard"""
    st.header("📊 Agent Performance Analytics")
    
    # Real-time metrics dashboard
    metrics_fig = viz_system.create_real_time_metrics_dashboard()
    st.plotly_chart(metrics_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Performance heatmap
        heatmap_fig = viz_system.create_agent_performance_heatmap()
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Performance Insights")
        
        # Top performing agents
        agents = viz_system.agent_network["agents"]
        top_performers = sorted(agents, key=lambda x: x["performance"], reverse=True)[:5]
        
        st.markdown("**🏆 Top Performing Agents:**")
        for i, agent in enumerate(top_performers):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i]
            st.text(f"{medal} {agent['name']}: {agent['performance']:.1%}")
        
        # Performance distribution
        performances = [a["performance"] for a in agents]
        performance_stats = {
            "Mean": np.mean(performances),
            "Median": np.median(performances),
            "Std Dev": np.std(performances),
            "Min": np.min(performances),
            "Max": np.max(performances)
        }
        
        st.markdown("**📈 Performance Statistics:**")
        for stat, value in performance_stats.items():
            if stat == "Std Dev":
                st.text(f"{stat}: ±{value:.3f}")
            else:
                st.text(f"{stat}: {value:.1%}")
        
        # Specialization analysis
        st.subheader("🎨 Agent Specialization")
        
        specialization_scores = [a["specialization_score"] for a in agents]
        avg_specialization = np.mean(specialization_scores)
        
        st.metric("Avg Specialization", f"{avg_specialization:.1%}")
        
        # Most specialized agents by type
        for agent_type in ["Classifier", "OCR", "Extractor", "Validator"]:
            type_agents = [a for a in agents if a["type"] == agent_type]
            if type_agents:
                most_specialized = max(type_agents, key=lambda x: x["specialization_score"])
                st.text(f"{agent_type}: {most_specialized['name']} ({most_specialized['specialization_score']:.1%})")

def show_emergent_behavior(viz_system: AgentVisualizationSystem):
    """Show emergent behavior analysis"""
    st.header("🧬 Emergent Behavior Analysis")
    
    # Emergent behavior timeline
    timeline_fig = viz_system.create_emergent_behavior_timeline()
    st.plotly_chart(timeline_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔬 Behavior Classification")
        
        behavior_types = {
            "Auto Load Balancing": {
                "frequency": "High",
                "impact": "Very High", 
                "description": "Agents automatically redistribute workload based on capacity"
            },
            "Specialization Discovery": {
                "frequency": "Medium",
                "impact": "High",
                "description": "Individual agents develop expertise in specific document types"
            },
            "Collaborative Optimization": {
                "frequency": "High",
                "impact": "High",
                "description": "Agents share learning and optimization strategies"
            },
            "Error Recovery Swarm": {
                "frequency": "Low",
                "impact": "Very High",
                "description": "Coordinated response to system errors or overload"
            },
            "Meta-Learning Event": {
                "frequency": "Low",
                "impact": "Very High",
                "description": "System-wide adaptation of processing strategies"
            },
            "Competitive Selection": {
                "frequency": "Medium",
                "impact": "High",
                "description": "Best performers automatically handle complex tasks"
            }
        }
        
        for behavior, details in behavior_types.items():
            st.markdown(f"**{behavior}**")
            st.markdown(f"- Frequency: {details['frequency']}")
            st.markdown(f"- Impact: {details['impact']}")
            st.markdown(f"- Description: {details['description']}")
            st.markdown("")
    
    with col2:
        st.subheader("📈 Emergence Metrics")
        
        # Simulate emergence metrics
        emergence_metrics = {
            "Collective Intelligence": 0.87,
            "System Adaptability": 0.92,
            "Emergent Efficiency": 0.89,
            "Behavioral Diversity": 0.76,
            "Learning Velocity": 0.83,
            "Self-Organization": 0.91
        }
        
        for metric, value in emergence_metrics.items():
            st.metric(metric, f"{value:.1%}")
        
        st.subheader("🎯 Emergence Indicators")
        
        indicators = [
            "🌟 **Novel Solutions**: System discovers unexpected optimization approaches",
            "🔄 **Self-Repair**: Automatic recovery from failures without intervention",
            "📚 **Collective Memory**: Shared learning across all agent instances",
            "⚡ **Adaptive Routing**: Dynamic workflow optimization based on conditions",
            "🧠 **Pattern Evolution**: Recognition patterns improve through use",
            "🤝 **Collaborative Networks**: Spontaneous agent partnership formation"
        ]
        
        for indicator in indicators:
            st.markdown(indicator)
        
        # Real-time emergence detection
        if st.button("🔍 Scan for New Emergent Behaviors", key="scan_emergence"):
            with st.spinner("Analyzing agent interactions..."):
                time.sleep(2)
            
            new_behaviors = [
                "🆕 Cross-type collaboration pattern detected",
                "🆕 Self-healing workflow adaptation observed", 
                "🆕 Novel error correction strategy emerged"
            ]
            
            for behavior in new_behaviors:
                st.success(behavior)

if __name__ == "__main__":
    create_agent_visualization_dashboard()