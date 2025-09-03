export interface Agent {
  id: string;
  name: string;
  type: 'coordinating' | 'processing' | 'analysis' | 'integration';
  status: 'active' | 'idle' | 'error' | 'offline';
  position: { x: number; y: number; z: number };
  performance: {
    throughput: number;
    accuracy: number;
    responseTime: number;
    utilization: number;
  };
  connections: string[];
  lastActivity: Date;
  metadata: {
    specialization?: string;
    version: string;
    capabilities: string[];
  };
}

export interface AgentNetwork {
  agents: Agent[];
  connections: Connection[];
  metrics: NetworkMetrics;
  timestamp: Date;
}

export interface Connection {
  id: string;
  sourceId: string;
  targetId: string;
  type: 'data_flow' | 'coordination' | 'feedback' | 'error';
  strength: number;
  latency: number;
  bandwidth: number;
  messages: number;
  direction: 'bidirectional' | 'source_to_target' | 'target_to_source';
}

export interface NetworkMetrics {
  totalAgents: number;
  activeAgents: number;
  totalConnections: number;
  averageLatency: number;
  networkThroughput: number;
  overallAccuracy: number;
  systemHealth: number;
  loadDistribution: number[];
}

export interface PerformanceMetric {
  timestamp: Date;
  agentId: string;
  metric: string;
  value: number;
  unit: string;
  category: 'performance' | 'accuracy' | 'efficiency' | 'health';
}

export interface TradingEvent {
  id: string;
  timestamp: Date;
  type: 'buy' | 'sell' | 'analysis' | 'decision' | 'execution';
  agentId: string;
  symbol?: string;
  quantity?: number;
  price?: number;
  confidence: number;
  reasoning: string;
  outcome?: 'success' | 'failure' | 'pending';
  profit?: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  volume: number;
  timestamp: Date;
  indicators: {
    rsi: number;
    macd: number;
    bollinger: { upper: number; lower: number; middle: number };
    sentiment: number;
  };
}

export interface VisualizationState {
  selectedAgent: string | null;
  timeRange: { start: Date; end: Date };
  viewMode: '2d' | '3d' | 'immersive';
  filters: {
    agentTypes: string[];
    performanceRange: [number, number];
    connectionTypes: string[];
  };
  playback: {
    isPlaying: boolean;
    speed: number;
    currentTime: Date;
  };
}

export interface DashboardConfig {
  refreshInterval: number;
  maxDataPoints: number;
  animationDuration: number;
  colorScheme: 'light' | 'dark' | 'auto';
  layout: 'grid' | 'flow' | 'custom';
  enabledPanels: string[];
}