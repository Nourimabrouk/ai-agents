import React, { useRef, useState, useMemo, useCallback } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import {
  OrbitControls,
  Environment,
  Text,
  Box,
  Plane,
  Sphere,
  Cylinder,
  Html,
  PerspectiveCamera,
  useTexture
} from '@react-three/drei';
import * as THREE from 'three';
import { motion, AnimatePresence } from 'framer-motion';
import { Agent, TradingEvent, MarketData } from '@types/index';
import { animated, useSpring, config } from '@react-spring/three';
import { format } from 'date-fns';

interface TradingFloorSimulationProps {
  agents: Agent[];
  tradingEvents: TradingEvent[];
  marketData: MarketData[];
  isPlaying: boolean;
  onEventSelect: (event: TradingEvent) => void;
}

// AI Agent Avatar Component
const AgentAvatar: React.FC<{
  agent: Agent;
  position: [number, number, number];
  isActive: boolean;
  currentAction?: string;
}> = ({ agent, position, isActive, currentAction }) => {
  const meshRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  const { scale, color, emissive } = useSpring({
    scale: isActive ? [1.2, 1.2, 1.2] : [1, 1, 1],
    color: getAgentAvatarColor(agent.type),
    emissive: isActive ? '#ff6b6b' : '#000000',
    config: config.wobbly
  });

  useFrame((state) => {
    if (meshRef.current && isActive) {
      // Subtle floating animation when active
      meshRef.current.position.y = position[1] + Math.sin(state.clock.elapsedTime * 2) * 0.1;
      meshRef.current.rotation.y += 0.01;
    }
  });

  const getAvatarGeometry = () => {
    switch (agent.type) {
      case 'coordinating':
        return (
          <group>
            <Cylinder args={[0.3, 0.5, 1.5, 8]} position={[0, 0.75, 0]}>
              <meshStandardMaterial color={color} />
            </Cylinder>
            <Sphere args={[0.25]} position={[0, 1.7, 0]}>
              <meshStandardMaterial color={color} />
            </Sphere>
          </group>
        );
      case 'processing':
        return (
          <group>
            <Box args={[0.6, 1.5, 0.4]} position={[0, 0.75, 0]}>
              <meshStandardMaterial color={color} />
            </Box>
            <Sphere args={[0.25]} position={[0, 1.7, 0]}>
              <meshStandardMaterial color={color} />
            </Sphere>
          </group>
        );
      case 'analysis':
        return (
          <group>
            <Cylinder args={[0.25, 0.4, 1.5, 6]} position={[0, 0.75, 0]}>
              <meshStandardMaterial color={color} />
            </Cylinder>
            <Sphere args={[0.25]} position={[0, 1.7, 0]}>
              <meshStandardMaterial color={color} />
            </Sphere>
          </group>
        );
      default:
        return (
          <group>
            <Cylinder args={[0.3, 0.4, 1.5, 8]} position={[0, 0.75, 0]}>
              <meshStandardMaterial color={color} />
            </Cylinder>
            <Sphere args={[0.25]} position={[0, 1.7, 0]}>
              <meshStandardMaterial color={color} />
            </Sphere>
          </group>
        );
    }
  };

  return (
    <animated.group
      ref={meshRef}
      position={position}
      scale={scale}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      {getAvatarGeometry()}
      
      {/* Status indicator above head */}
      <Sphere args={[0.05]} position={[0, 2.2, 0]}>
        <animated.meshBasicMaterial color={emissive} />
      </Sphere>

      {/* Agent name label */}
      <Text
        position={[0, 2.5, 0]}
        fontSize={0.15}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {agent.name}
      </Text>

      {/* Activity indicator */}
      {currentAction && (
        <Html position={[0, -0.5, 0]} style={{ pointerEvents: 'none' }}>
          <div style={{
            background: 'rgba(0,0,0,0.8)',
            color: 'white',
            padding: '5px 10px',
            borderRadius: '15px',
            fontSize: '12px',
            textAlign: 'center',
            minWidth: '80px'
          }}>
            {currentAction}
          </div>
        </Html>
      )}

      {/* Detailed info on hover */}
      {hovered && (
        <Html position={[1, 1, 0]} style={{ pointerEvents: 'none' }}>
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            style={{
              background: 'rgba(0,0,0,0.9)',
              color: 'white',
              padding: '10px',
              borderRadius: '8px',
              fontSize: '12px',
              minWidth: '150px',
              border: '1px solid rgba(255,255,255,0.2)'
            }}
          >
            <div><strong>{agent.name}</strong></div>
            <div>Type: {agent.type}</div>
            <div>Performance: {agent.performance.accuracy.toFixed(1)}%</div>
            <div>Utilization: {agent.performance.utilization.toFixed(1)}%</div>
          </motion.div>
        </Html>
      )}
    </animated.group>
  );
};

// Trading Event Visualization
const TradingEventEffect: React.FC<{
  event: TradingEvent;
  position: [number, number, number];
  onComplete: () => void;
}> = ({ event, position, onComplete }) => {
  const meshRef = useRef<THREE.Group>(null);

  const { scale, opacity } = useSpring({
    from: { scale: [0, 0, 0], opacity: 1 },
    to: async (next) => {
      await next({ scale: [2, 2, 2], opacity: 0.8 });
      await next({ scale: [4, 4, 4], opacity: 0 });
      onComplete();
    },
    config: { tension: 200, friction: 15 }
  });

  const getEventColor = () => {
    switch (event.type) {
      case 'buy': return '#4caf50';
      case 'sell': return '#f44336';
      case 'analysis': return '#2196f3';
      case 'decision': return '#ff9800';
      default: return '#9c27b0';
    }
  };

  return (
    <animated.group ref={meshRef} position={position} scale={scale}>
      <Sphere args={[0.1]}>
        <animated.meshBasicMaterial 
          color={getEventColor()} 
          transparent 
          opacity={opacity}
        />
      </Sphere>
      <Html>
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          style={{
            background: getEventColor(),
            color: 'white',
            padding: '2px 6px',
            borderRadius: '10px',
            fontSize: '10px',
            fontWeight: 'bold'
          }}
        >
          {event.type.toUpperCase()}
        </motion.div>
      </Html>
    </animated.group>
  );
};

// Market Data Display Screens
const MarketScreen: React.FC<{
  marketData: MarketData[];
  position: [number, number, number];
  rotation: [number, number, number];
}> = ({ marketData, position, rotation }) => {
  const screenRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (screenRef.current) {
      // Subtle screen glow effect
      const material = screenRef.current.material as THREE.MeshStandardMaterial;
      material.emissiveIntensity = 0.3 + 0.1 * Math.sin(state.clock.elapsedTime * 2);
    }
  });

  return (
    <group position={position} rotation={rotation}>
      {/* Screen frame */}
      <Box args={[3, 2, 0.1]} ref={screenRef}>
        <meshStandardMaterial 
          color="#1a1a1a" 
          emissive="#0066cc" 
          emissiveIntensity={0.3}
        />
      </Box>

      {/* Market data overlay */}
      <Html
        position={[0, 0, 0.1]}
        transform
        occlude
        style={{
          width: '280px',
          height: '180px',
          background: 'rgba(0,10,30,0.9)',
          border: '2px solid #0066cc',
          borderRadius: '10px',
          padding: '10px',
          color: '#00ff00',
          fontFamily: 'monospace',
          fontSize: '12px',
          overflow: 'hidden'
        }}
      >
        <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div style={{ borderBottom: '1px solid #0066cc', paddingBottom: '5px', marginBottom: '10px' }}>
            <strong>LIVE MARKET DATA</strong>
          </div>
          {marketData.slice(0, 6).map((data, index) => (
            <div key={index} style={{ 
              display: 'flex', 
              justifyContent: 'space-between',
              marginBottom: '3px',
              color: data.change >= 0 ? '#00ff00' : '#ff4444'
            }}>
              <span>{data.symbol}</span>
              <span>${data.price.toFixed(2)}</span>
              <span>{data.change >= 0 ? '+' : ''}{data.change.toFixed(2)}%</span>
            </div>
          ))}
        </div>
      </Html>
    </group>
  );
};

// Profit/Loss Visualization
const ProfitLossVisualization: React.FC<{
  tradingEvents: TradingEvent[];
  position: [number, number, number];
}> = ({ tradingEvents, position }) => {
  const profitEvents = tradingEvents.filter(e => e.profit !== undefined);
  const totalProfit = profitEvents.reduce((sum, e) => sum + (e.profit || 0), 0);

  const { height, color } = useSpring({
    height: Math.abs(totalProfit) / 1000,
    color: totalProfit >= 0 ? '#4caf50' : '#f44336',
    config: config.slow
  });

  return (
    <animated.group position={position}>
      <animated.mesh position={[0, height.to(h => h / 2), 0]}>
        <boxGeometry args={[0.5, height, 0.5]} />
        <animated.meshStandardMaterial color={color} />
      </animated.mesh>
      
      <Html position={[0, height.to(h => h + 0.5), 0]}>
        <div style={{
          background: 'rgba(0,0,0,0.8)',
          color: totalProfit >= 0 ? '#4caf50' : '#f44336',
          padding: '5px 10px',
          borderRadius: '5px',
          textAlign: 'center',
          fontWeight: 'bold',
          fontSize: '14px'
        }}>
          ${totalProfit.toFixed(0)}
        </div>
      </Html>
    </animated.group>
  );
};

// Trading Floor Environment
const TradingFloorEnvironment: React.FC = () => {
  return (
    <>
      {/* Floor */}
      <Plane args={[50, 50]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
        <meshStandardMaterial color="#2a2a2a" />
      </Plane>

      {/* Grid pattern on floor */}
      <Plane args={[50, 50]} rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.01, 0]}>
        <meshBasicMaterial color="#444444" transparent opacity={0.3} />
      </Plane>

      {/* Ceiling */}
      <Plane args={[50, 50]} rotation={[Math.PI / 2, 0, 0]} position={[0, 10, 0]}>
        <meshStandardMaterial color="#1a1a1a" />
      </Plane>

      {/* Walls */}
      <Plane args={[50, 10]} position={[0, 5, -25]}>
        <meshStandardMaterial color="#2a2a2a" />
      </Plane>
      <Plane args={[50, 10]} rotation={[0, Math.PI, 0]} position={[0, 5, 25]}>
        <meshStandardMaterial color="#2a2a2a" />
      </Plane>
      <Plane args={[50, 10]} rotation={[0, Math.PI / 2, 0]} position={[-25, 5, 0]}>
        <meshStandardMaterial color="#2a2a2a" />
      </Plane>
      <Plane args={[50, 10]} rotation={[0, -Math.PI / 2, 0]} position={[25, 5, 0]}>
        <meshStandardMaterial color="#2a2a2a" />
      </Plane>

      {/* Atmospheric lighting */}
      <ambientLight intensity={0.2} />
      <pointLight position={[10, 8, 10]} intensity={1} color="#ffffff" />
      <pointLight position={[-10, 8, -10]} intensity={0.8} color="#0066cc" />
      <spotLight
        position={[0, 15, 0]}
        angle={0.5}
        penumbra={1}
        intensity={1.5}
        castShadow
      />
    </>
  );
};

// Main Trading Floor Simulation Component
const TradingFloorSimulation: React.FC<TradingFloorSimulationProps> = ({
  agents,
  tradingEvents,
  marketData,
  isPlaying,
  onEventSelect
}) => {
  const [selectedEvent, setSelectedEvent] = useState<TradingEvent | null>(null);
  const [activeEvents, setActiveEvents] = useState<TradingEvent[]>([]);
  const [cameraPosition, setCameraPosition] = useState<[number, number, number]>([15, 10, 15]);

  // Generate agent positions in a realistic trading floor layout
  const agentPositions = useMemo(() => {
    const positions: Record<string, [number, number, number]> = {};
    const gridSize = Math.ceil(Math.sqrt(agents.length));
    
    agents.forEach((agent, index) => {
      const row = Math.floor(index / gridSize);
      const col = index % gridSize;
      const x = (col - gridSize / 2) * 3;
      const z = (row - gridSize / 2) * 3;
      positions[agent.id] = [x, 0, z];
    });
    
    return positions;
  }, [agents]);

  // Simulate real-time trading events
  const handleEventEffect = useCallback((event: TradingEvent) => {
    setActiveEvents(prev => [...prev, event]);
    setTimeout(() => {
      setActiveEvents(prev => prev.filter(e => e.id !== event.id));
    }, 3000);
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        shadows
        style={{ background: 'linear-gradient(to bottom, #0a0a0a, #1a1a2e)' }}
      >
        <PerspectiveCamera makeDefault position={cameraPosition} fov={75} />
        
        {/* Environment and lighting */}
        <TradingFloorEnvironment />
        <Environment preset="night" />

        {/* AI Agent Avatars */}
        {agents.map(agent => (
          <AgentAvatar
            key={agent.id}
            agent={agent}
            position={agentPositions[agent.id]}
            isActive={agent.status === 'active'}
            currentAction={tradingEvents
              .filter(e => e.agentId === agent.id)
              .slice(-1)[0]?.type}
          />
        ))}

        {/* Market Data Screens */}
        <MarketScreen
          marketData={marketData}
          position={[0, 3, -12]}
          rotation={[0, 0, 0]}
        />
        <MarketScreen
          marketData={marketData.slice(3)}
          position={[12, 3, 0]}
          rotation={[0, -Math.PI / 2, 0]}
        />

        {/* Trading Event Effects */}
        <AnimatePresence>
          {activeEvents.map(event => (
            <TradingEventEffect
              key={event.id}
              event={event}
              position={agentPositions[event.agentId] || [0, 2, 0]}
              onComplete={() => {}}
            />
          ))}
        </AnimatePresence>

        {/* Profit/Loss Towers */}
        {agents.map((agent, index) => (
          <ProfitLossVisualization
            key={agent.id}
            tradingEvents={tradingEvents.filter(e => e.agentId === agent.id)}
            position={[agentPositions[agent.id][0] + 1, 0, agentPositions[agent.id][2]]}
          />
        ))}

        {/* Camera controls */}
        <OrbitControls
          target={[0, 2, 0]}
          maxDistance={30}
          minDistance={5}
          maxPolarAngle={Math.PI / 2.2}
          enablePan={true}
          autoRotate={isPlaying}
          autoRotateSpeed={0.5}
        />
      </Canvas>

      {/* Control Panel */}
      <motion.div
        initial={{ opacity: 0, y: 100 }}
        animate={{ opacity: 1, y: 0 }}
        style={{
          position: 'absolute',
          bottom: 20,
          left: 20,
          right: 20,
          background: 'rgba(0,0,0,0.9)',
          padding: '20px',
          borderRadius: '15px',
          border: '1px solid rgba(255,255,255,0.2)',
          color: 'white'
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '20px' }}>
          {/* Real-time metrics */}
          <div>
            <h4>Market Status</h4>
            <div>Active Agents: {agents.filter(a => a.status === 'active').length}</div>
            <div>Recent Events: {tradingEvents.slice(-5).length}</div>
            <div>Total P&L: ${tradingEvents.reduce((sum, e) => sum + (e.profit || 0), 0).toFixed(0)}</div>
          </div>

          {/* Recent events */}
          <div>
            <h4>Recent Events</h4>
            <div style={{ maxHeight: '60px', overflowY: 'auto' }}>
              {tradingEvents.slice(-3).map(event => (
                <div 
                  key={event.id}
                  style={{ 
                    fontSize: '12px', 
                    marginBottom: '5px',
                    cursor: 'pointer',
                    padding: '3px',
                    borderRadius: '3px',
                    background: selectedEvent?.id === event.id ? 'rgba(255,255,255,0.1)' : 'transparent'
                  }}
                  onClick={() => {
                    setSelectedEvent(event);
                    onEventSelect(event);
                  }}
                >
                  {format(event.timestamp, 'HH:mm:ss')} - {event.type} by {event.agentId}
                </div>
              ))}
            </div>
          </div>

          {/* Camera controls */}
          <div>
            <h4>View Controls</h4>
            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
              <button
                onClick={() => setCameraPosition([15, 10, 15])}
                style={{ padding: '5px 10px', borderRadius: '5px', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white' }}
              >
                Overview
              </button>
              <button
                onClick={() => setCameraPosition([0, 20, 0])}
                style={{ padding: '5px 10px', borderRadius: '5px', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white' }}
              >
                Top Down
              </button>
              <button
                onClick={() => setCameraPosition([25, 3, 0])}
                style={{ padding: '5px 10px', borderRadius: '5px', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white' }}
              >
                Side View
              </button>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

// Helper function to get agent avatar color
const getAgentAvatarColor = (type: string): string => {
  switch (type) {
    case 'coordinating': return '#e74c3c';
    case 'processing': return '#3498db';
    case 'analysis': return '#2ecc71';
    case 'integration': return '#f39c12';
    default: return '#95a5a6';
  }
};

export default TradingFloorSimulation;