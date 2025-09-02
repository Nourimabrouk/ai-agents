"""
Logging utilities for AI agents
Provides structured logging with context tracking
"""

import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime, timezone


class AgentFormatter(logging.Formatter):
    """Custom formatter for agent logs"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        if hasattr(record, 'agent_name'):
            log_obj['agent_name'] = record.agent_name
        if hasattr(record, 'task_id'):
            log_obj['task_id'] = record.task_id
        if hasattr(record, 'metrics'):
            log_obj['metrics'] = record.metrics
            
        # For development, use human-readable format
        if record.levelname in ['ERROR', 'CRITICAL']:
            return f"[{log_obj['timestamp']}] {log_obj['level']} - {log_obj['logger']}: {log_obj['message']}"
        else:
            return f"{log_obj['level']}: {log_obj['message']}"


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """
    Get a configured logger for an agent or module
    
    Args:
        name: Logger name (usually __name__)
        level: Optional logging level override
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if not already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AgentFormatter())
        logger.addHandler(handler)
        
        # Set level from environment or default
        if level is not None:
            logger.setLevel(level)
        else:
            logger.setLevel(logging.INFO)
    
    return logger


class LogContext:
    """Context manager for structured logging"""
    
    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.context = kwargs
        self.old_context = {}
    
    def __enter__(self):
        # Store and set context
        for key, value in self.context.items():
            if hasattr(self.logger, key):
                self.old_context[key] = getattr(self.logger, key)
            setattr(self.logger, key, value)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore old context
        for key in self.context:
            if key in self.old_context:
                setattr(self.logger, key, self.old_context[key])
            else:
                delattr(self.logger, key)