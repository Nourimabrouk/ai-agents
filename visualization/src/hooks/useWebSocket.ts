import { useEffect, useState, useCallback, useRef } from 'react';
import io, { Socket } from 'socket.io-client';
import { Agent, AgentNetwork, PerformanceMetric, TradingEvent } from '@types/index';

interface WebSocketData {
  agentNetwork: AgentNetwork | null;
  performanceMetrics: PerformanceMetric[];
  tradingEvents: TradingEvent[];
  isConnected: boolean;
  connectionError: string | null;
}

export const useWebSocket = (url: string = 'ws://localhost:8000') => {
  const [data, setData] = useState<WebSocketData>({
    agentNetwork: null,
    performanceMetrics: [],
    tradingEvents: [],
    isConnected: false,
    connectionError: null
  });

  const socketRef = useRef<Socket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState(0);
  const maxReconnectAttempts = 10;

  const connect = useCallback(() => {
    if (socketRef.current?.connected) {
      return;
    }

    try {
      socketRef.current = io(url, {
        transports: ['websocket', 'polling'],
        timeout: 20000,
        reconnection: true,
        reconnectionDelay: 1000,
        reconnectionAttempts: maxReconnectAttempts
      });

      socketRef.current.on('connect', () => {
        console.log('WebSocket connected');
        setData(prev => ({ ...prev, isConnected: true, connectionError: null }));
        setReconnectAttempts(0);
      });

      socketRef.current.on('disconnect', (reason) => {
        console.log('WebSocket disconnected:', reason);
        setData(prev => ({ ...prev, isConnected: false }));
      });

      socketRef.current.on('connect_error', (error) => {
        console.error('WebSocket connection error:', error);
        setData(prev => ({ 
          ...prev, 
          isConnected: false, 
          connectionError: error.message 
        }));
        
        setReconnectAttempts(prev => prev + 1);
        if (reconnectAttempts < maxReconnectAttempts) {
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, Math.min(1000 * Math.pow(2, reconnectAttempts), 30000));
        }
      });

      // Agent network updates
      socketRef.current.on('agent_network_update', (network: AgentNetwork) => {
        setData(prev => ({ ...prev, agentNetwork: network }));
      });

      // Performance metrics stream
      socketRef.current.on('performance_metrics', (metrics: PerformanceMetric[]) => {
        setData(prev => ({
          ...prev,
          performanceMetrics: [...prev.performanceMetrics, ...metrics]
            .slice(-1000) // Keep last 1000 metrics
            .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
        }));
      });

      // Trading events stream
      socketRef.current.on('trading_event', (event: TradingEvent) => {
        setData(prev => ({
          ...prev,
          tradingEvents: [event, ...prev.tradingEvents].slice(0, 100) // Keep last 100 events
        }));
      });

      // System health updates
      socketRef.current.on('system_health', (health: any) => {
        console.log('System health update:', health);
      });

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      setData(prev => ({ 
        ...prev, 
        connectionError: error instanceof Error ? error.message : 'Unknown error' 
      }));
    }
  }, [url, reconnectAttempts]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (socketRef.current) {
      socketRef.current.disconnect();
      socketRef.current = null;
    }
    
    setData(prev => ({ ...prev, isConnected: false }));
  }, []);

  const sendMessage = useCallback((event: string, data: any) => {
    if (socketRef.current?.connected) {
      socketRef.current.emit(event, data);
    } else {
      console.warn('WebSocket not connected, message not sent:', event, data);
    }
  }, []);

  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    ...data,
    connect,
    disconnect,
    sendMessage,
    reconnectAttempts
  };
};