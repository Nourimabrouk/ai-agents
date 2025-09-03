import React, { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
  ScatterChart,
  Scatter,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Heatmap as RechartsHeatmap
} from 'recharts';
import { format } from 'date-fns';
import * as d3 from 'd3';
import { Agent, PerformanceMetric, AgentNetwork } from '@types/index';
import styled from 'styled-components';

const DashboardContainer = styled.div`
  display: grid;
  grid-template-columns: 1fr 1fr;
  grid-template-rows: auto auto auto;
  gap: 20px;
  padding: 20px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  min-height: 100vh;
  color: white;
`;

const PanelContainer = styled(motion.div)`
  background: rgba(255, 255, 255, 0.1);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 20px;
  border: 1px solid rgba(255, 255, 255, 0.2);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
`;

const MetricCard = styled(motion.div)`
  background: rgba(255, 255, 255, 0.15);
  border-radius: 10px;
  padding: 15px;
  text-align: center;
  backdrop-filter: blur(5px);
  border: 1px solid rgba(255, 255, 255, 0.1);
`;

interface PerformanceDashboardProps {
  network: AgentNetwork;
  metrics: PerformanceMetric[];
  timeRange: { start: Date; end: Date };
}

const PerformanceDashboard: React.FC<PerformanceDashboardProps> = ({
  network,
  metrics,
  timeRange
}) => {
  const [selectedMetric, setSelectedMetric] = useState<string>('throughput');
  const [viewMode, setViewMode] = useState<'overview' | 'detailed' | '3d'>('overview');

  // Process metrics for visualization
  const timeSeriesData = useMemo(() => {
    const grouped = metrics.reduce((acc, metric) => {
      const timeKey = format(metric.timestamp, 'HH:mm:ss');
      if (!acc[timeKey]) {
        acc[timeKey] = { time: timeKey, timestamp: metric.timestamp };
      }
      acc[timeKey][metric.metric] = metric.value;
      return acc;
    }, {} as Record<string, any>);

    return Object.values(grouped).sort((a, b) => 
      new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );
  }, [metrics]);

  // Agent performance trajectories for 3D visualization
  const agentTrajectories = useMemo(() => {
    return network.agents.map(agent => {
      const agentMetrics = metrics.filter(m => m.agentId === agent.id);
      const trajectory = agentMetrics.map(m => ({
        x: m.timestamp.getTime(),
        y: m.value,
        z: agent.performance.utilization,
        agent: agent.name,
        metric: m.metric
      }));
      return { agent: agent.name, data: trajectory };
    });
  }, [network.agents, metrics]);

  // Heatmap data for strategy effectiveness
  const strategyHeatmapData = useMemo(() => {
    const strategies = ['momentum', 'reversal', 'breakout', 'support', 'resistance'];
    const timeSlots = Array.from({ length: 24 }, (_, i) => i);
    
    return timeSlots.map(hour => 
      strategies.map(strategy => ({
        hour,
        strategy,
        effectiveness: Math.random() * 100,
        volume: Math.random() * 1000
      }))
    ).flat();
  }, []);

  // Real-time anomaly detection
  const anomalies = useMemo(() => {
    return metrics.filter(m => {
      const threshold = 2; // Standard deviations
      const agentMetrics = metrics.filter(metric => 
        metric.agentId === m.agentId && metric.metric === m.metric
      );
      const mean = agentMetrics.reduce((sum, metric) => sum + metric.value, 0) / agentMetrics.length;
      const stdDev = Math.sqrt(
        agentMetrics.reduce((sum, metric) => sum + Math.pow(metric.value - mean, 2), 0) / agentMetrics.length
      );
      return Math.abs(m.value - mean) > threshold * stdDev;
    });
  }, [metrics]);

  // Performance projections using linear regression
  const performanceProjections = useMemo(() => {
    const now = new Date();
    const futurePoints = 10;
    const projections: any[] = [];

    network.agents.forEach(agent => {
      const agentMetrics = metrics.filter(m => m.agentId === agent.id);
      if (agentMetrics.length < 2) return;

      const x = agentMetrics.map((_, i) => i);
      const y = agentMetrics.map(m => m.value);

      // Simple linear regression
      const n = x.length;
      const sumX = x.reduce((a, b) => a + b, 0);
      const sumY = y.reduce((a, b) => a + b, 0);
      const sumXY = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
      const sumXX = x.reduce((sum, xi) => sum + xi * xi, 0);

      const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
      const intercept = (sumY - slope * sumX) / n;

      for (let i = 1; i <= futurePoints; i++) {
        const futureX = n + i;
        const projectedValue = slope * futureX + intercept;
        projections.push({
          time: format(new Date(now.getTime() + i * 60000), 'HH:mm:ss'),
          agent: agent.name,
          projected: Math.max(0, projectedValue),
          confidence: Math.max(0, 1 - (i / futurePoints))
        });
      }
    });

    return projections;
  }, [network.agents, metrics]);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{
          background: 'rgba(0,0,0,0.8)',
          padding: '10px',
          borderRadius: '5px',
          border: '1px solid rgba(255,255,255,0.2)'
        }}>
          <p>{`Time: ${label}`}</p>
          {payload.map((entry: any, index: number) => (
            <p key={index} style={{ color: entry.color }}>
              {`${entry.dataKey}: ${entry.value?.toFixed(2)}`}
            </p>
          ))}
        </div>
      );
    }
    return null;
  };

  return (
    <DashboardContainer>
      {/* Real-time metrics overview */}
      <PanelContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h3>Real-Time Performance Metrics</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px' }}>
          <MetricCard whileHover={{ scale: 1.05 }}>
            <div style={{ fontSize: '2em', fontWeight: 'bold' }}>
              {network.metrics.networkThroughput.toFixed(1)}
            </div>
            <div>Throughput/sec</div>
          </MetricCard>
          <MetricCard whileHover={{ scale: 1.05 }}>
            <div style={{ fontSize: '2em', fontWeight: 'bold' }}>
              {(network.metrics.overallAccuracy * 100).toFixed(1)}%
            </div>
            <div>Accuracy</div>
          </MetricCard>
          <MetricCard whileHover={{ scale: 1.05 }}>
            <div style={{ fontSize: '2em', fontWeight: 'bold' }}>
              {network.metrics.averageLatency.toFixed(0)}ms
            </div>
            <div>Latency</div>
          </MetricCard>
          <MetricCard whileHover={{ scale: 1.05 }}>
            <div style={{ fontSize: '2em', fontWeight: 'bold', color: anomalies.length > 0 ? '#ff6b6b' : '#51cf66' }}>
              {anomalies.length}
            </div>
            <div>Anomalies</div>
          </MetricCard>
        </div>
      </PanelContainer>

      {/* Agent Performance Trajectories (3D-style) */}
      <PanelContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <h3>Agent Performance Trajectories</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={timeSeriesData}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
            <XAxis 
              dataKey="time" 
              stroke="rgba(255,255,255,0.7)"
              tick={{ fontSize: 12 }}
            />
            <YAxis stroke="rgba(255,255,255,0.7)" tick={{ fontSize: 12 }} />
            <Tooltip content={<CustomTooltip />} />
            <Legend />
            {network.agents.map((agent, index) => (
              <Line
                key={agent.id}
                type="monotone"
                dataKey={`throughput_${agent.id}`}
                stroke={`hsl(${index * 360 / network.agents.length}, 70%, 60%)`}
                strokeWidth={2}
                dot={{ r: 4 }}
                connectNulls={false}
                name={agent.name}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </PanelContainer>

      {/* Strategy Effectiveness Heatmap */}
      <PanelContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.0 }}
      >
        <h3>Strategy Effectiveness Heatmap</h3>
        <HeatmapVisualization data={strategyHeatmapData} />
      </PanelContainer>

      {/* Predictive Performance Projections */}
      <PanelContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.2 }}
      >
        <h3>Performance Projections</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={performanceProjections.slice(0, 20)}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.2)" />
            <XAxis dataKey="time" stroke="rgba(255,255,255,0.7)" />
            <YAxis stroke="rgba(255,255,255,0.7)" />
            <Tooltip content={<CustomTooltip />} />
            <Area
              type="monotone"
              dataKey="projected"
              stroke="#8884d8"
              fill="url(#projectionGradient)"
              strokeWidth={2}
            />
            <defs>
              <linearGradient id="projectionGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#8884d8" stopOpacity={0.2} />
              </linearGradient>
            </defs>
          </AreaChart>
        </ResponsiveContainer>
      </PanelContainer>

      {/* Multi-dimensional Performance Analysis */}
      <PanelContainer
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1.4 }}
        style={{ gridColumn: '1 / -1' }}
      >
        <h3>Multi-Dimensional Performance Analysis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {/* Radar Chart for Agent Comparison */}
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={network.agents.map(agent => ({
              agent: agent.name,
              throughput: agent.performance.throughput / 100,
              accuracy: agent.performance.accuracy / 100,
              responseTime: (1000 - agent.performance.responseTime) / 1000,
              utilization: agent.performance.utilization / 100
            }))}>
              <PolarGrid stroke="rgba(255,255,255,0.2)" />
              <PolarAngleAxis dataKey="agent" tick={{ fontSize: 10, fill: 'white' }} />
              <PolarRadiusAxis tick={{ fontSize: 10, fill: 'white' }} />
              <Radar
                name="Performance"
                dataKey="throughput"
                stroke="#8884d8"
                fill="#8884d8"
                fillOpacity={0.3}
              />
              <Radar
                name="Accuracy"
                dataKey="accuracy"
                stroke="#82ca9d"
                fill="#82ca9d"
                fillOpacity={0.3}
              />
              <Legend />
            </RadarChart>
          </ResponsiveContainer>

          {/* Scatter plot for Cost-Benefit Analysis */}
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid stroke="rgba(255,255,255,0.2)" />
              <XAxis 
                type="number" 
                dataKey="cost" 
                name="Cost" 
                stroke="rgba(255,255,255,0.7)"
                tick={{ fontSize: 10 }}
              />
              <YAxis 
                type="number" 
                dataKey="benefit" 
                name="Benefit"
                stroke="rgba(255,255,255,0.7)"
                tick={{ fontSize: 10 }}
              />
              <Tooltip 
                cursor={{ strokeDasharray: '3 3' }}
                content={<CustomTooltip />}
              />
              <Scatter
                name="Agents"
                data={network.agents.map(agent => ({
                  cost: agent.performance.responseTime,
                  benefit: agent.performance.throughput * agent.performance.accuracy,
                  name: agent.name,
                  size: agent.performance.utilization
                }))}
                fill="#8884d8"
              />
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </PanelContainer>
    </DashboardContainer>
  );
};

// Custom Heatmap Component using D3
const HeatmapVisualization: React.FC<{ data: any[] }> = ({ data }) => {
  const svgRef = React.useRef<SVGSVGElement>(null);

  React.useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 70 };
    const width = 400 - margin.left - margin.right;
    const height = 200 - margin.bottom - margin.top;

    const strategies = [...new Set(data.map(d => d.strategy))];
    const hours = [...new Set(data.map(d => d.hour))].sort((a, b) => a - b);

    const xScale = d3.scaleBand()
      .domain(hours.map(String))
      .range([0, width])
      .padding(0.05);

    const yScale = d3.scaleBand()
      .domain(strategies)
      .range([height, 0])
      .padding(0.05);

    const colorScale = d3.scaleSequential(d3.interpolateViridis)
      .domain([0, 100]);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Add heatmap rectangles
    g.selectAll('rect')
      .data(data)
      .enter()
      .append('rect')
      .attr('x', d => xScale(String(d.hour)) || 0)
      .attr('y', d => yScale(d.strategy) || 0)
      .attr('width', xScale.bandwidth())
      .attr('height', yScale.bandwidth())
      .attr('fill', d => colorScale(d.effectiveness))
      .style('opacity', 0)
      .transition()
      .duration(1000)
      .style('opacity', 1);

    // Add axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))
      .selectAll('text')
      .style('fill', 'white')
      .style('font-size', '10px');

    g.append('g')
      .call(d3.axisLeft(yScale))
      .selectAll('text')
      .style('fill', 'white')
      .style('font-size', '10px');

  }, [data]);

  return (
    <div style={{ display: 'flex', justifyContent: 'center' }}>
      <svg
        ref={svgRef}
        width={400}
        height={240}
        style={{ background: 'rgba(255,255,255,0.05)', borderRadius: '5px' }}
      />
    </div>
  );
};

export default PerformanceDashboard;