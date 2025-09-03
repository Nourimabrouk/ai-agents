import React, { useState, useEffect, useMemo } from 'react';
import styled from 'styled-components';
import { motion, AnimatePresence } from 'framer-motion';
import { Routes, Route, useNavigate, useLocation } from 'react-router-dom';
import NetworkVisualization3D from '@components/NetworkVisualization3D';
import PerformanceDashboard from '@components/PerformanceDashboard';
import TradingFloorSimulation from '@components/TradingFloorSimulation';
import { useWebSocket } from '@hooks/useWebSocket';
import { Agent, AgentNetwork, PerformanceMetric, TradingEvent, MarketData } from '@types/index';

const AppContainer = styled.div`
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
  overflow: hidden;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
`;

const NavigationBar = styled(motion.nav)`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: 70px;
  background: rgba(0, 0, 0, 0.9);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(255, 255, 255, 0.1);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 30px;
  z-index: 1000;
`;

const NavButton = styled(motion.button)<{ active?: boolean }>`
  background: ${props => props.active ? 'rgba(64, 158, 255, 0.2)' : 'transparent'};
  border: 1px solid ${props => props.active ? '#409eff' : 'rgba(255, 255, 255, 0.2)'};
  color: ${props => props.active ? '#409eff' : 'white'};
  padding: 10px 20px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 500;
  transition: all 0.3s ease;

  &:hover {
    background: rgba(64, 158, 255, 0.1);
    border-color: #409eff;
  }
`;

const StatusIndicator = styled(motion.div)<{ connected: boolean }>`
  display: flex;
  align-items: center;
  gap: 8px;
  color: ${props => props.connected ? '#4caf50' : '#f44336'};
  font-size: 14px;
`;

const StatusDot = styled.div<{ connected: boolean }>`
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: ${props => props.connected ? '#4caf50' : '#f44336'};
  animation: ${props => props.connected ? 'pulse 2s infinite' : 'none'};

  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.5; }
    100% { opacity: 1; }
  }
`;

const MainContent = styled.div`
  margin-top: 70px;
  height: calc(100vh - 70px);
  position: relative;
`;

const LoadingOverlay = styled(motion.div)`
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  color: white;
  z-index: 999;
`;

const ErrorOverlay = styled(motion.div)`
  position: fixed;
  top: 100px;
  right: 30px;
  background: rgba(244, 67, 54, 0.9);
  color: white;
  padding: 15px 20px;
  border-radius: 8px;
  border: 1px solid rgba(244, 67, 54, 0.5);
  z-index: 1001;
  max-width: 300px;
`;

// Demo data generator for fallback
const generateDemoData = (): {
  network: AgentNetwork;
  metrics: PerformanceMetric[];
  tradingEvents: TradingEvent[];
  marketData: MarketData[];
} => {
  const agents: Agent[] = Array.from({ length: 8 }, (_, i) => ({
    id: `agent-${i}`,
    name: `Agent ${i + 1}`,
    type: ['coordinating', 'processing', 'analysis', 'integration'][i % 4] as any,
    status: Math.random() > 0.2 ? 'active' : 'idle' as any,
    position: {
      x: (Math.random() - 0.5) * 20,
      y: Math.random() * 5,
      z: (Math.random() - 0.5) * 20
    },
    performance: {
      throughput: Math.random() * 100,
      accuracy: 85 + Math.random() * 15,
      responseTime: 100 + Math.random() * 300,
      utilization: Math.random() * 100
    },
    connections: [],
    lastActivity: new Date(),
    metadata: {
      version: '1.0.0',
      capabilities: ['trading', 'analysis', 'coordination']
    }
  }));

  const connections = agents.flatMap(agent =>
    agents
      .filter(other => other.id !== agent.id && Math.random() > 0.7)
      .map(other => ({
        id: `${agent.id}-${other.id}`,
        sourceId: agent.id,
        targetId: other.id,
        type: ['data_flow', 'coordination', 'feedback'][Math.floor(Math.random() * 3)] as any,
        strength: Math.random(),
        latency: Math.random() * 100,
        bandwidth: Math.random() * 1000,
        messages: Math.floor(Math.random() * 100),
        direction: 'bidirectional' as any
      }))
  );

  const network: AgentNetwork = {
    agents,
    connections,
    metrics: {
      totalAgents: agents.length,
      activeAgents: agents.filter(a => a.status === 'active').length,
      totalConnections: connections.length,
      averageLatency: connections.reduce((sum, c) => sum + c.latency, 0) / connections.length,
      networkThroughput: agents.reduce((sum, a) => sum + a.performance.throughput, 0),
      overallAccuracy: agents.reduce((sum, a) => sum + a.performance.accuracy, 0) / agents.length / 100,
      systemHealth: Math.random() * 0.2 + 0.8,
      loadDistribution: agents.map(a => a.performance.utilization)
    },
    timestamp: new Date()
  };

  const metrics: PerformanceMetric[] = Array.from({ length: 100 }, (_, i) => ({
    timestamp: new Date(Date.now() - (100 - i) * 1000),
    agentId: agents[Math.floor(Math.random() * agents.length)].id,
    metric: ['throughput', 'accuracy', 'latency', 'utilization'][Math.floor(Math.random() * 4)],
    value: Math.random() * 100,
    unit: 'percentage',
    category: 'performance' as any
  }));

  const tradingEvents: TradingEvent[] = Array.from({ length: 50 }, (_, i) => ({
    id: `event-${i}`,
    timestamp: new Date(Date.now() - (50 - i) * 2000),
    type: ['buy', 'sell', 'analysis', 'decision'][Math.floor(Math.random() * 4)] as any,
    agentId: agents[Math.floor(Math.random() * agents.length)].id,
    symbol: ['AAPL', 'GOOGL', 'TSLA', 'MSFT'][Math.floor(Math.random() * 4)],
    quantity: Math.floor(Math.random() * 1000),
    price: Math.random() * 500 + 100,
    confidence: Math.random(),
    reasoning: 'AI analysis indicates favorable conditions',
    outcome: Math.random() > 0.3 ? 'success' : 'pending' as any,
    profit: (Math.random() - 0.5) * 10000
  }));

  const marketData: MarketData[] = [
    'AAPL', 'GOOGL', 'TSLA', 'MSFT', 'AMZN', 'NVDA', 'META', 'BRK.A'
  ].map(symbol => ({
    symbol,
    price: Math.random() * 500 + 100,
    change: (Math.random() - 0.5) * 10,
    volume: Math.floor(Math.random() * 1000000),
    timestamp: new Date(),
    indicators: {
      rsi: Math.random() * 100,
      macd: (Math.random() - 0.5) * 10,
      bollinger: {
        upper: Math.random() * 600 + 200,
        lower: Math.random() * 200 + 50,
        middle: Math.random() * 400 + 100
      },
      sentiment: Math.random() * 2 - 1
    }
  }));

  return { network, metrics, tradingEvents, marketData };
};

const App: React.FC = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [timelineProgress, setTimelineProgress] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [selectedEvent, setSelectedEvent] = useState<TradingEvent | null>(null);

  // WebSocket connection
  const {
    agentNetwork,
    performanceMetrics,
    tradingEvents,
    isConnected,
    connectionError,
    reconnectAttempts
  } = useWebSocket();

  // Fallback to demo data
  const demoData = useMemo(() => generateDemoData(), []);

  const currentData = useMemo(() => ({
    network: agentNetwork || demoData.network,
    metrics: performanceMetrics.length > 0 ? performanceMetrics : demoData.metrics,
    tradingEvents: tradingEvents.length > 0 ? tradingEvents : demoData.tradingEvents,
    marketData: demoData.marketData
  }), [agentNetwork, performanceMetrics, tradingEvents, demoData]);

  const currentView = location.pathname.slice(1) || 'network';

  // Auto-play timeline
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setTimelineProgress(prev => (prev + 1) % 100);
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying]);

  const handleNavigate = (view: string) => {
    navigate(`/${view}`);
  };

  const handleEventSelect = (event: TradingEvent) => {
    setSelectedEvent(event);
    console.log('Event selected:', event);
  };

  return (
    <AppContainer>
      <NavigationBar
        initial={{ y: -70 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: "easeOut" }}
      >
        <motion.div 
          style={{ display: 'flex', alignItems: 'center', gap: '30px' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <h2 style={{ color: 'white', margin: 0, fontSize: '18px', fontWeight: '600' }}>
            AI Agents Visualization Platform
          </h2>
          
          <div style={{ display: 'flex', gap: '15px' }}>
            <NavButton
              active={currentView === 'network'}
              onClick={() => handleNavigate('network')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              🌐 Network 3D
            </NavButton>
            
            <NavButton
              active={currentView === 'dashboard'}
              onClick={() => handleNavigate('dashboard')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              📊 Performance
            </NavButton>
            
            <NavButton
              active={currentView === 'trading'}
              onClick={() => handleNavigate('trading')}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              🏢 Trading Floor
            </NavButton>
          </div>
        </motion.div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
          <StatusIndicator connected={isConnected}>
            <StatusDot connected={isConnected} />
            {isConnected ? 'Live Data' : `Reconnecting... (${reconnectAttempts})`}
          </StatusIndicator>

          <motion.button
            onClick={() => setIsPlaying(!isPlaying)}
            style={{
              background: 'transparent',
              border: '1px solid rgba(255,255,255,0.2)',
              color: 'white',
              padding: '8px 16px',
              borderRadius: '6px',
              cursor: 'pointer',
              fontSize: '14px'
            }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {isPlaying ? '⏸️ Pause' : '▶️ Play'}
          </motion.button>
        </div>
      </NavigationBar>

      <MainContent>
        <AnimatePresence mode="wait">
          <Routes>
            <Route
              path="/"
              element={
                <motion.div
                  key="network"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.1 }}
                  transition={{ duration: 0.5 }}
                  style={{ width: '100%', height: '100%' }}
                >
                  <NetworkVisualization3D
                    network={currentData.network}
                    selectedAgent={selectedAgent}
                    onAgentSelect={setSelectedAgent}
                    timelineProgress={timelineProgress}
                  />
                </motion.div>
              }
            />
            
            <Route
              path="/network"
              element={
                <motion.div
                  key="network"
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 1.1 }}
                  transition={{ duration: 0.5 }}
                  style={{ width: '100%', height: '100%' }}
                >
                  <NetworkVisualization3D
                    network={currentData.network}
                    selectedAgent={selectedAgent}
                    onAgentSelect={setSelectedAgent}
                    timelineProgress={timelineProgress}
                  />
                </motion.div>
              }
            />
            
            <Route
              path="/dashboard"
              element={
                <motion.div
                  key="dashboard"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.5 }}
                  style={{ width: '100%', height: '100%', overflow: 'auto' }}
                >
                  <PerformanceDashboard
                    network={currentData.network}
                    metrics={currentData.metrics}
                    timeRange={{ start: new Date(Date.now() - 3600000), end: new Date() }}
                  />
                </motion.div>
              }
            />
            
            <Route
              path="/trading"
              element={
                <motion.div
                  key="trading"
                  initial={{ opacity: 0, rotateY: 90 }}
                  animate={{ opacity: 1, rotateY: 0 }}
                  exit={{ opacity: 0, rotateY: -90 }}
                  transition={{ duration: 0.6 }}
                  style={{ width: '100%', height: '100%' }}
                >
                  <TradingFloorSimulation
                    agents={currentData.network.agents}
                    tradingEvents={currentData.tradingEvents}
                    marketData={currentData.marketData}
                    isPlaying={isPlaying}
                    onEventSelect={handleEventSelect}
                  />
                </motion.div>
              }
            />
          </Routes>
        </AnimatePresence>
      </MainContent>

      {/* Loading overlay */}
      <AnimatePresence>
        {!isConnected && reconnectAttempts === 0 && (
          <LoadingOverlay
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <div style={{ fontSize: '18px', marginBottom: '20px' }}>
              Connecting to Agent Network...
            </div>
            <div style={{ 
              width: '50px', 
              height: '50px', 
              border: '3px solid rgba(255,255,255,0.3)',
              borderTop: '3px solid white',
              borderRadius: '50%',
              animation: 'spin 1s linear infinite'
            }} />
          </LoadingOverlay>
        )}
      </AnimatePresence>

      {/* Error overlay */}
      <AnimatePresence>
        {connectionError && reconnectAttempts > 3 && (
          <ErrorOverlay
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
          >
            <div style={{ fontWeight: 'bold', marginBottom: '10px' }}>
              Connection Error
            </div>
            <div style={{ fontSize: '14px' }}>
              {connectionError}
            </div>
            <div style={{ fontSize: '12px', marginTop: '10px', opacity: 0.8 }}>
              Using demo data for visualization
            </div>
          </ErrorOverlay>
        )}
      </AnimatePresence>

      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </AppContainer>
  );
};

export default App;