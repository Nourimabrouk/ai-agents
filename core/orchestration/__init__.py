"""
Core Orchestration Module
Multi-agent coordination and task management
"""

# Ensure imports work from new location
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from .orchestrator import *
