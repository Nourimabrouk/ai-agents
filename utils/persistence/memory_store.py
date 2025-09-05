"""
Memory persistence for AI agents
Provides durable storage for agent memories and learning
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from pathlib import Path
import pickle
import logging

logger = logging.getLogger(__name__)


class SqliteMemoryStore:
    """SQLite-based memory persistence"""
    
    def __init__(self, db_path: str = "agent_memory.db"):
        """Initialize SQLite memory store"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Episodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    result TEXT,
                    success BOOLEAN,
                    learnings TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Semantic memory table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS semantic_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(agent_name, key)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_episodes_agent ON episodes(agent_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_semantic_agent ON semantic_memory(agent_name)")
            
            conn.commit()
    
    def save_episode(self, agent_name: str, observation: Any) -> None:
        """Save an episodic memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Serialize complex objects
                action_data = {
                    "action_type": observation.action.action_type,
                    "parameters": observation.action.parameters,
                    "tools_used": observation.action.tools_used,
                    "expected_outcome": observation.action.expected_outcome
                }
                
                cursor.execute("""
                    INSERT INTO episodes (agent_name, timestamp, action, result, success, learnings)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    agent_name,
                    observation.timestamp.isoformat(),
                    json.dumps(action_data),
                    json.dumps(observation.result) if observation.result else None,
                    observation.success,
                    json.dumps(observation.learnings)
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save episode: {e}")
    
    def recent_episodes(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodes for an agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT timestamp, action, result, success, learnings
                    FROM episodes
                    WHERE agent_name = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (agent_name, limit))
                
                episodes = []
                for row in cursor.fetchall():
                    episodes.append({
                        "timestamp": row[0],
                        "action": json.loads(row[1]),
                        "result": json.loads(row[2]) if row[2] else None,
                        "success": bool(row[3]),
                        "learnings": json.loads(row[4]) if row[4] else []
                    })
                
                return episodes
        except Exception as e:
            logger.error(f"Failed to retrieve episodes: {e}")
            return []
    
    def save_semantic(self, agent_name: str, key: str, value: Any) -> None:
        """Save semantic memory"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO semantic_memory (agent_name, key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (
                    agent_name,
                    key,
                    json.dumps(value),
                    datetime.now(timezone.utc).isoformat()
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save semantic memory: {e}")
    
    def get_semantic(self, agent_name: str, key: str) -> Optional[Any]:
        """Get semantic memory value"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT value FROM semantic_memory
                    WHERE agent_name = ? AND key = ?
                """, (agent_name, key))
                
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
                return {}
        except Exception as e:
            logger.error(f"Failed to get semantic memory: {e}")
            return {}
    
    def all_semantic(self, agent_name: str) -> Dict[str, Any]:
        """Get all semantic memory for an agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT key, value FROM semantic_memory
                    WHERE agent_name = ?
                """, (agent_name,))
                
                memory = {}
                for row in cursor.fetchall():
                    memory[row[0]] = json.loads(row[1])
                
                return memory
        except Exception as e:
            logger.error(f"Failed to get all semantic memory: {e}")
            return {}
    
    def clear_agent_memory(self, agent_name: str) -> None:
        """Clear all memory for an agent"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM episodes WHERE agent_name = ?", (agent_name,))
                cursor.execute("DELETE FROM semantic_memory WHERE agent_name = ?", (agent_name,))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to clear agent memory: {e}")


class SemanticProxy(dict):
    """Proxy for semantic memory that auto-persists changes"""
    
    def __init__(self, agent_name: str, backend: SqliteMemoryStore, initial_data: Dict[str, Any]):
        super().__init__(initial_data)
        self.agent_name = agent_name
        self.backend = backend
    
    def __setitem__(self, key: str, value: Any):
        """Override to persist on write"""
        super().__setitem__(key, value)
        if self.backend:
            try:
                self.backend.save_semantic(self.agent_name, key, value)
            except Exception as e:
                logger.error(f"Failed to persist semantic memory: {e}")
    
    def __delitem__(self, key: str):
        """Override to handle deletion"""
        super().__delitem__(key)
        # Could add deletion from backend if needed
    
    def update(self, *args, **kwargs):
        """Override to persist batch updates"""
        super().update(*args, **kwargs)
        if self.backend:
            for key, value in self.items():
                try:
                    self.backend.save_semantic(self.agent_name, key, value)
                except Exception as e:
                    logger.error(f"Failed to persist semantic memory update: {e}")