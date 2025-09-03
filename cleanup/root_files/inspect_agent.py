#!/usr/bin/env python3
"""
Deep dive into specific agent implementations
"""
import os
import sys

def inspect_meta_orchestrator():
    """Inspect meta-orchestrator work (synthetic data pipeline)"""
    print("🤖 META-ORCHESTRATOR: Synthetic Data Pipeline Implementation")
    print("=" * 70)
    
    files = [
        'data/market/financial_data_pipeline.py',
        'data/synthetic/business_data_generator.py',
        'data/integration/data_agent_integration.py',
        'requirements-data-pipeline.txt'
    ]
    
    for filepath in files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = len(f.readlines())
                    first_10_lines = f.seek(0) or f.read().split('\n')[:10]
                
                print(f"\n📄 {filepath}")
                print(f"   Size: {size:,} bytes | Lines: {lines:,}")
                print("   Preview:")
                for i, line in enumerate(first_10_lines[:10], 1):
                    print(f"   {i:2d}: {line[:80]}")
                    
            except Exception as e:
                print(f"   Error reading: {e}")
        else:
            print(f"❌ Missing: {filepath}")

def inspect_system_architect():
    """Inspect system-architect work (visualization + infrastructure)"""
    print("\n🤖 SYSTEM-ARCHITECT: Visualization & Infrastructure")
    print("=" * 70)
    
    viz_files = [
        'visualization/src/components/NetworkVisualization3D.tsx',
        'visualization/src/components/PerformanceDashboard.tsx', 
        'visualization/src/components/TradingFloorSimulation.tsx',
        'backend/visualization_server.py'
    ]
    
    infra_files = [
        'core/infrastructure/real_time_websocket_coordinator.py',
        'core/infrastructure/distributed_agent_communication_protocol.py',
        'core/infrastructure/production_monitoring_system.py'
    ]
    
    for section, files in [("Visualization", viz_files), ("Infrastructure", infra_files)]:
        print(f"\n📊 {section} Files:")
        for filepath in files:
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        lines = len(content.split('\n'))
                        # Look for key patterns
                        patterns = ['class ', 'async def', 'function ', '@app.', 'const ']
                        implementations = sum(content.count(p) for p in patterns)
                    
                    print(f"   📄 {os.path.basename(filepath):<40} {size:>8,}b {lines:>5}L {implementations:>3} impl")
                except:
                    print(f"   📄 {os.path.basename(filepath):<40} {size:>8,}b (binary)")
            else:
                print(f"   ❌ {filepath}")

def inspect_pattern_researcher():
    """Inspect pattern-researcher work (RL framework)"""
    print("\n🤖 PATTERN-RESEARCHER: Reinforcement Learning Framework")
    print("=" * 70)
    
    rl_files = [
        'rl/environments/trading_environment.py',
        'rl/environments/supply_chain_environment.py',
        'rl/algorithms/ppo_agent.py',
        'rl/algorithms/networks.py',
        'rl/training/curriculum.py',
        'rl/training/training_pipeline.py',
        'test_rl_system.py',
        'rl_training_demo.py'
    ]
    
    print("📊 RL Implementation Files:")
    for filepath in rl_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    
                    # Count RL-specific implementations
                    rl_patterns = {
                        'Classes': content.count('class '),
                        'Async Methods': content.count('async def'),
                        'Environments': content.count('Environment'),
                        'Agents': content.count('Agent'),
                        'Networks': content.count('Network')
                    }
                
                pattern_str = " | ".join(f"{k}:{v}" for k,v in rl_patterns.items())
                print(f"   📄 {os.path.basename(filepath):<35} {size:>8,}b {lines:>5}L | {pattern_str}")
            except:
                print(f"   📄 {os.path.basename(filepath):<35} {size:>8,}b (error)")
        else:
            print(f"   ❌ {filepath}")

def inspect_code_synthesizer():
    """Inspect code-synthesizer work (demo scenarios)"""
    print("\n🤖 CODE-SYNTHESIZER: Impressive Demo Scenarios")
    print("=" * 70)
    
    demo_files = [
        'demos/phase4_trading_tournament.py',
        'demos/phase4_supply_chain_crisis.py', 
        'demos/phase4_autonomous_research.py',
        'demos/phase4_demo_launcher.py',
        'setup_phase4_demos.py'
    ]
    
    print("📊 Demo Implementation Files:")
    total_demo_size = 0
    for filepath in demo_files:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            total_demo_size += size
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    lines = len(content.split('\n'))
                    
                    # Count demo-specific features
                    demo_features = {
                        'Streamlit': content.count('st.'),
                        'Plotly': content.count('plotly') + content.count('px.') + content.count('go.'),
                        'AI Agents': content.count('Agent'),
                        'Async Ops': content.count('async def'),
                        'Real-time': content.count('real_time') + content.count('realtime')
                    }
                
                features_str = " | ".join(f"{k}:{v}" for k,v in demo_features.items() if v > 0)
                print(f"   📄 {os.path.basename(filepath):<35} {size:>8,}b {lines:>5}L")
                print(f"       Features: {features_str}")
            except:
                print(f"   📄 {os.path.basename(filepath):<35} {size:>8,}b (error)")
    
    print(f"\n   📊 Total Demo Implementation: {total_demo_size:,} bytes")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        agent = sys.argv[1].lower()
        if agent == "meta":
            inspect_meta_orchestrator()
        elif agent == "system":
            inspect_system_architect()
        elif agent == "pattern":
            inspect_pattern_researcher()
        elif agent == "code":
            inspect_code_synthesizer()
        else:
            print("Usage: python inspect_agent.py [meta|system|pattern|code]")
    else:
        # Show all
        inspect_meta_orchestrator()
        inspect_system_architect() 
        inspect_pattern_researcher()
        inspect_code_synthesizer()
        
        print(f"\n{'='*70}")
        print("✅ DEEP INSPECTION COMPLETE")
        print("Usage: python inspect_agent.py [meta|system|pattern|code] for specific agent")
        print(f"{'='*70}")