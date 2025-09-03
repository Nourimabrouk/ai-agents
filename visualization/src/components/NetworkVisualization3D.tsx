import React, { useMemo, useRef, useCallback, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { 
  OrbitControls, 
  Text, 
  Sphere, 
  Box, 
  Line,
  Html,
  Environment,
  PerspectiveCamera
} from '@react-three/drei';
import * as THREE from 'three';
import { Agent, AgentNetwork, Connection } from '@types/index';
import { animated, useSpring } from '@react-spring/three';
import { motion } from 'framer-motion';

interface NetworkVisualization3DProps {
  network: AgentNetwork;
  selectedAgent: string | null;
  onAgentSelect: (agentId: string) => void;
  timelineProgress: number;
}

const AgentNode: React.FC<{
  agent: Agent;
  isSelected: boolean;
  onClick: () => void;
  scale: number;
}> = ({ agent, isSelected, onClick, scale }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  const { position, color, pulseScale } = useSpring({
    position: [agent.position.x, agent.position.y, agent.position.z] as [number, number, number],
    color: isSelected ? '#ff6b6b' : getAgentColor(agent),
    pulseScale: isSelected ? 1.2 : hovered ? 1.1 : 1,
    config: { tension: 300, friction: 30 }
  });

  useFrame((state) => {
    if (meshRef.current) {
      // Gentle floating animation
      meshRef.current.position.y += Math.sin(state.clock.elapsedTime * 2) * 0.01;
      
      // Performance-based glow intensity
      const intensity = agent.performance.utilization / 100;
      (meshRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity = intensity * 0.5;
    }
  });

  const getGeometry = () => {
    switch (agent.type) {
      case 'coordinating': return <Box args={[scale, scale, scale]} />;
      case 'processing': return <Sphere args={[scale * 0.7, 16, 16]} />;
      case 'analysis': return <Box args={[scale * 0.8, scale * 0.5, scale * 1.2]} />;
      case 'integration': return <Sphere args={[scale * 0.6, 8, 8]} />;
      default: return <Sphere args={[scale * 0.7, 16, 16]} />;
    }
  };

  return (
    <animated.group position={position} scale={pulseScale}>
      <animated.mesh
        ref={meshRef}
        onClick={onClick}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
      >
        {getGeometry()}
        <animated.meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.2}
          roughness={0.3}
          metalness={0.7}
          transparent
          opacity={agent.status === 'active' ? 1.0 : 0.6}
        />
      </animated.mesh>
      
      {/* Agent label */}
      <Text
        position={[0, scale + 0.5, 0]}
        fontSize={0.3}
        color="white"
        anchorX="center"
        anchorY="middle"
      >
        {agent.name}
      </Text>
      
      {/* Status indicator */}
      <Sphere args={[0.1]} position={[scale * 0.7, scale * 0.7, 0]}>
        <meshBasicMaterial color={getStatusColor(agent.status)} />
      </Sphere>
      
      {/* Performance indicators */}
      {isSelected && (
        <Html position={[scale + 1, 0, 0]} style={{ pointerEvents: 'none' }}>
          <div style={{
            background: 'rgba(0,0,0,0.8)',
            color: 'white',
            padding: '10px',
            borderRadius: '5px',
            fontSize: '12px',
            minWidth: '200px'
          }}>
            <div><strong>{agent.name}</strong></div>
            <div>Type: {agent.type}</div>
            <div>Status: {agent.status}</div>
            <div>Throughput: {agent.performance.throughput.toFixed(1)}/s</div>
            <div>Accuracy: {agent.performance.accuracy.toFixed(1)}%</div>
            <div>Response: {agent.performance.responseTime.toFixed(0)}ms</div>
            <div>Utilization: {agent.performance.utilization.toFixed(1)}%</div>
          </div>
        </Html>
      )}
    </animated.group>
  );
};

const ConnectionLine: React.FC<{
  connection: Connection;
  agents: Agent[];
  animate: boolean;
}> = ({ connection, agents, animate }) => {
  const lineRef = useRef<THREE.Line>(null);
  
  const sourceAgent = agents.find(a => a.id === connection.sourceId);
  const targetAgent = agents.find(a => a.id === connection.targetId);
  
  if (!sourceAgent || !targetAgent) return null;

  const points = [
    new THREE.Vector3(sourceAgent.position.x, sourceAgent.position.y, sourceAgent.position.z),
    new THREE.Vector3(targetAgent.position.x, targetAgent.position.y, targetAgent.position.z)
  ];

  const { opacity, scale } = useSpring({
    opacity: connection.strength * 0.8,
    scale: Math.max(0.5, connection.bandwidth / 100),
    config: { tension: 200, friction: 25 }
  });

  useFrame((state) => {
    if (lineRef.current && animate) {
      // Animated data flow effect
      const material = lineRef.current.material as THREE.LineBasicMaterial;
      material.opacity = 0.3 + 0.4 * Math.sin(state.clock.elapsedTime * 3 + connection.id.length);
    }
  });

  const getConnectionColor = () => {
    switch (connection.type) {
      case 'data_flow': return '#4CAF50';
      case 'coordination': return '#2196F3';
      case 'feedback': return '#FF9800';
      case 'error': return '#F44336';
      default: return '#9E9E9E';
    }
  };

  return (
    <Line
      ref={lineRef}
      points={points}
      color={getConnectionColor()}
      lineWidth={scale.get()}
      transparent
      opacity={opacity.get()}
    />
  );
};

const NetworkBackground: React.FC = () => {
  const { scene } = useThree();

  useMemo(() => {
    // Add grid helper
    const gridHelper = new THREE.GridHelper(50, 50, '#333333', '#333333');
    gridHelper.position.y = -10;
    scene.add(gridHelper);

    // Add ambient particles for atmosphere
    const particleGeometry = new THREE.BufferGeometry();
    const particleCount = 1000;
    const positions = new Float32Array(particleCount * 3);
    
    for (let i = 0; i < particleCount * 3; i += 3) {
      positions[i] = (Math.random() - 0.5) * 100;
      positions[i + 1] = (Math.random() - 0.5) * 100;
      positions[i + 2] = (Math.random() - 0.5) * 100;
    }
    
    particleGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    
    const particleMaterial = new THREE.PointsMaterial({
      color: '#888888',
      size: 0.1,
      transparent: true,
      opacity: 0.3
    });
    
    const particles = new THREE.Points(particleGeometry, particleMaterial);
    scene.add(particles);

    return () => {
      scene.remove(gridHelper);
      scene.remove(particles);
    };
  }, [scene]);

  return null;
};

const NetworkVisualization3D: React.FC<NetworkVisualization3DProps> = ({
  network,
  selectedAgent,
  onAgentSelect,
  timelineProgress
}) => {
  const [animateConnections, setAnimateConnections] = useState(true);

  const nodeScale = useMemo(() => {
    const maxThroughput = Math.max(...network.agents.map(a => a.performance.throughput));
    return (agent: Agent) => 0.5 + (agent.performance.throughput / maxThroughput) * 1.5;
  }, [network.agents]);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas style={{ background: 'linear-gradient(to bottom, #0a0a0a, #1a1a1a)' }}>
        <PerspectiveCamera makeDefault position={[20, 20, 20]} fov={60} />
        
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <pointLight position={[10, 10, 10]} intensity={1} color="#ffffff" />
        <pointLight position={[-10, -10, -10]} intensity={0.5} color="#4a90e2" />
        <spotLight
          position={[0, 50, 0]}
          angle={0.3}
          penumbra={1}
          intensity={2}
          castShadow
          color="#ffffff"
        />

        {/* Environment */}
        <Environment preset="night" />
        <NetworkBackground />

        {/* Agent nodes */}
        {network.agents.map(agent => (
          <AgentNode
            key={agent.id}
            agent={agent}
            isSelected={selectedAgent === agent.id}
            onClick={() => onAgentSelect(agent.id)}
            scale={nodeScale(agent)}
          />
        ))}

        {/* Connections */}
        {network.connections.map(connection => (
          <ConnectionLine
            key={connection.id}
            connection={connection}
            agents={network.agents}
            animate={animateConnections}
          />
        ))}

        {/* Controls */}
        <OrbitControls
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          maxDistance={100}
          minDistance={5}
          autoRotate={false}
          autoRotateSpeed={0.5}
        />
      </Canvas>

      {/* Control panel */}
      <motion.div
        initial={{ opacity: 0, x: -100 }}
        animate={{ opacity: 1, x: 0 }}
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          background: 'rgba(0,0,0,0.8)',
          padding: '15px',
          borderRadius: '10px',
          color: 'white',
          minWidth: '200px'
        }}
      >
        <h3>Network Overview</h3>
        <div>Total Agents: {network.agents.length}</div>
        <div>Active: {network.agents.filter(a => a.status === 'active').length}</div>
        <div>Connections: {network.connections.length}</div>
        <div>Health: {(network.metrics.systemHealth * 100).toFixed(1)}%</div>
        
        <div style={{ marginTop: '10px' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <input
              type="checkbox"
              checked={animateConnections}
              onChange={(e) => setAnimateConnections(e.target.checked)}
            />
            Animate Connections
          </label>
        </div>
      </motion.div>

      {/* Performance metrics overlay */}
      {selectedAgent && (
        <motion.div
          initial={{ opacity: 0, y: 100 }}
          animate={{ opacity: 1, y: 0 }}
          style={{
            position: 'absolute',
            bottom: 20,
            right: 20,
            background: 'rgba(0,0,0,0.9)',
            padding: '20px',
            borderRadius: '10px',
            color: 'white',
            minWidth: '300px'
          }}
        >
          {(() => {
            const agent = network.agents.find(a => a.id === selectedAgent);
            if (!agent) return null;
            
            return (
              <>
                <h3>Agent Details: {agent.name}</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div>
                    <div>Type: {agent.type}</div>
                    <div>Status: {agent.status}</div>
                    <div>Specialization: {agent.metadata.specialization || 'General'}</div>
                  </div>
                  <div>
                    <div>Throughput: {agent.performance.throughput.toFixed(1)}/s</div>
                    <div>Accuracy: {agent.performance.accuracy.toFixed(1)}%</div>
                    <div>Response: {agent.performance.responseTime.toFixed(0)}ms</div>
                  </div>
                </div>
                <div style={{ marginTop: '10px' }}>
                  <div style={{ 
                    background: '#333', 
                    height: '10px', 
                    borderRadius: '5px',
                    overflow: 'hidden'
                  }}>
                    <div style={{ 
                      background: agent.performance.utilization > 80 ? '#f44336' : 
                                 agent.performance.utilization > 60 ? '#ff9800' : '#4caf50',
                      width: `${agent.performance.utilization}%`,
                      height: '100%',
                      transition: 'width 0.3s ease'
                    }} />
                  </div>
                  <div style={{ fontSize: '12px', marginTop: '5px' }}>
                    Utilization: {agent.performance.utilization.toFixed(1)}%
                  </div>
                </div>
              </>
            );
          })()}
        </motion.div>
      )}
    </div>
  );
};

// Helper functions
const getAgentColor = (agent: Agent): string => {
  switch (agent.type) {
    case 'coordinating': return '#e74c3c';
    case 'processing': return '#3498db';
    case 'analysis': return '#2ecc71';
    case 'integration': return '#f39c12';
    default: return '#95a5a6';
  }
};

const getStatusColor = (status: string): string => {
  switch (status) {
    case 'active': return '#2ecc71';
    case 'idle': return '#f39c12';
    case 'error': return '#e74c3c';
    case 'offline': return '#95a5a6';
    default: return '#95a5a6';
  }
};

export default NetworkVisualization3D;