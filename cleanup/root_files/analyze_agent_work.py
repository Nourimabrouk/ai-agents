#!/usr/bin/env python3
"""
Analyze what each agent implemented
"""
import os
import glob
from pathlib import Path
from datetime import datetime, timedelta

def analyze_recent_files(hours_back=3):
    """Find files modified in the last N hours"""
    cutoff_time = datetime.now() - timedelta(hours=hours_back)
    recent_files = []
    
    for root, dirs, files in os.walk('.'):
        # Skip .git and __pycache__ directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        
        for file in files:
            if file.endswith(('.py', '.tsx', '.ts', '.js', '.md', '.txt')):
                filepath = os.path.join(root, file)
                try:
                    mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                    if mtime > cutoff_time:
                        size = os.path.getsize(filepath)
                        recent_files.append((filepath, mtime, size))
                except:
    logger.info(f'Method {function_name} called')
    return None
    
    return sorted(recent_files, key=lambda x: x[1], reverse=True)

def analyze_by_directory():
    """Analyze files by directory structure"""
    directories = {
        'data/': 'meta-orchestrator (Synthetic Data Pipeline)',
        'visualization/': 'system-architect (Advanced Visualization)',
        'rl/': 'pattern-researcher (RL Framework)', 
        'demos/': 'code-synthesizer (Demo Scenarios)',
        'core/infrastructure/': 'system-architect (Implementation Gaps)'
    }
    
    for dir_path, agent in directories.items():
        if os.path.exists(dir_path):
            print(f"\n{'='*60}")
            print(f"[AGENT] {agent}")
            print(f"{'='*60}")
            
            files = []
            for root, dirs, filenames in os.walk(dir_path):
                for filename in filenames:
                    if filename.endswith(('.py', '.tsx', '.ts', '.js', '.md')):
                        filepath = os.path.join(root, filename)
                        size = os.path.getsize(filepath)
                        files.append((filepath, size))
            
            files.sort(key=lambda x: x[1], reverse=True)  # Sort by size
            
            total_size = sum(size for _, size in files)
            total_lines = 0
            
            for filepath, size in files:
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        print(f"  [FILE] {filepath:<50} {size:>8,} bytes  {lines:>6} lines")
                except:
                    print(f"  [FILE] {filepath:<50} {size:>8,} bytes  (binary)")
            
            print(f"\n  [TOTAL] {len(files)} files, {total_size:,} bytes, {total_lines:,} lines")

def count_implementations():
    """Count key implementations"""
    patterns = {
        'Classes': r'class\s+\w+',
        'Async Functions': r'async\s+def\s+\w+',
        'API Endpoints': r'@app\.(get|post|put|delete)',
        'React Components': r'const\s+\w+.*=.*\(\)',
        'WebSocket Handlers': r'websocket|WebSocket',
        'RL Algorithms': r'PPO|SAC|MADDPG|QMIX'
    }
    
    print(f"\n{'='*60}")
    print("[IMPLEMENTATION ANALYSIS]")
    print(f"{'='*60}")
    
    for pattern_name, pattern in patterns.items():
        count = 0
        files = glob.glob('**/*.py', recursive=True) + \
                glob.glob('**/*.ts', recursive=True) + \
                glob.glob('**/*.tsx', recursive=True)
        
        for filepath in files:
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    import re
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    count += len(matches)
            except:
    return 0
        
        print(f"  {pattern_name:<20}: {count:>5} found")

if __name__ == "__main__":
    print("[AGENT WORK ANALYSIS]")
    print("=" * 60)
    
    # Analyze recent files
    recent = analyze_recent_files(hours_back=6)
    if recent:
        print(f"\n[RECENT] FILES MODIFIED IN LAST 6 HOURS: {len(recent)}")
        print("-" * 60)
        for filepath, mtime, size in recent[:20]:  # Top 20
            print(f"  {mtime.strftime('%H:%M')} | {filepath:<50} | {size:>8,} bytes")
    
    # Analyze by directory (agent)
    analyze_by_directory()
    
    # Count implementations
    count_implementations()
    
    print(f"\n{'='*60}")
    print("[COMPLETE] ANALYSIS COMPLETE")
    print(f"{'='*60}")