"""
RL Framework Setup and Verification Script
Installs dependencies and verifies the reinforcement learning framework
"""

import subprocess
import sys
import os
import asyncio
from pathlib import Path
import importlib.util

def install_requirements():
    """Install required packages"""
    print("Installing RL framework requirements...")
    
    requirements_file = Path(__file__).parent / "rl" / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"Requirements file not found: {requirements_file}")
        return False
    
    try:
        # Install requirements
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Requirements installed successfully!")
            return True
        else:
            print(f"❌ Failed to install requirements: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required_packages = [
        "torch",
        "numpy", 
        "pandas",
        "matplotlib",
        "gymnasium",
        "networkx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {missing_packages}")
        print("Run: pip install -r rl/requirements.txt")
        return False
    
    print("✅ All dependencies available!")
    return True

def check_gpu_support():
    """Check GPU support"""
    print("Checking GPU support...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ CUDA available with {gpu_count} GPU(s)")
            print(f"✅ Primary GPU: {gpu_name}")
            return True
        else:
            print("⚠️  CUDA not available - using CPU only")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU support: {e}")
        return False

def verify_framework_structure():
    """Verify the RL framework structure"""
    print("Verifying RL framework structure...")
    
    base_dir = Path(__file__).parent
    required_files = [
        "rl/__init__.py",
        "rl/environments/__init__.py",
        "rl/environments/base_environment.py",
        "rl/environments/trading_environment.py",
        "rl/algorithms/__init__.py", 
        "rl/algorithms/ppo_agent.py",
        "rl/algorithms/networks.py",
        "rl/training/__init__.py",
        "rl/training/curriculum.py",
        "rl/training/training_pipeline.py",
        "rl_training_demo.py",
        "test_rl_system.py",
        "data/synthetic_data_generator.py"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        full_path = base_dir / file_path
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n❌ Missing files: {missing_files}")
        return False
    
    print("✅ All framework files present!")
    return True

async def run_basic_tests():
    """Run basic functionality tests"""
    print("Running basic functionality tests...")
    
    try:
        # Test synthetic data generation
        print("Testing synthetic data generation...")
        from data.synthetic_data_generator import SyntheticDataGenerator
        
        generator = SyntheticDataGenerator(seed=42)
        data = generator.generate_agent_network(num_agents=3)
        
        if len(data['agents']) == 3:
            print("✅ Synthetic data generation works")
        else:
            print("❌ Synthetic data generation failed")
            return False
        
        # Test environment creation
        print("Testing environment creation...")
        from rl.environments.trading_environment import TradingEnvironment
        from rl.environments.base_environment import AgentConfig, AgentRole
        
        agent_configs = [
            AgentConfig(agent_id="test_agent", role=AgentRole.EXECUTOR, initial_balance=10000)
        ]
        
        env = TradingEnvironment(
            agent_configs=agent_configs,
            symbols=['AAPL'],
            max_steps=10,
            seed=42
        )
        
        obs, _ = env.reset()
        if obs and len(obs) == 1:
            print("✅ Environment creation works")
        else:
            print("❌ Environment creation failed")
            return False
        
        # Test PPO agent
        print("Testing PPO agent...")
        from rl.algorithms.ppo_agent import PPOAgent, PPOConfig
        
        config = PPOConfig(hidden_dims=[16, 16], batch_size=32)
        agent = PPOAgent(
            agent_id="test_agent",
            state_dim=env._get_observation_dimension(),
            action_dim=env._get_action_dimension(),
            config=config
        )
        
        state = list(obs.values())[0]
        action = agent.select_action(state, evaluation=True)
        
        if action is not None and len(action) == env._get_action_dimension():
            print("✅ PPO agent works")
        else:
            print("❌ PPO agent failed")
            return False
        
        print("✅ Basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Basic tests failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_quick_demo():
    """Create a quick demo script for immediate testing"""
    demo_script = '''"""
Quick RL Framework Demo
A minimal example to verify the framework is working
"""

import asyncio
import numpy as np
from rl.environments.trading_environment import TradingEnvironment
from rl.environments.base_environment import AgentConfig, AgentRole
from rl.algorithms.ppo_agent import PPOAgent, PPOConfig

async def quick_demo():
    print("Running quick RL framework demo...")
    
    # Create environment
    agent_configs = [
        AgentConfig(agent_id="demo_agent", role=AgentRole.EXECUTOR, initial_balance=10000)
    ]
    
    env = TradingEnvironment(
        agent_configs=agent_configs,
        symbols=['DEMO'],
        max_steps=50,
        seed=42
    )
    
    # Create agent
    config = PPOConfig(hidden_dims=[32, 32], batch_size=32)
    agent = PPOAgent(
        agent_id="demo_agent",
        state_dim=env._get_observation_dimension(),
        action_dim=env._get_action_dimension(),
        config=config
    )
    
    # Run a few episodes
    for episode in range(3):
        print(f"Episode {episode + 1}/3")
        
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        for step in range(50):
            action = agent.select_action(obs["demo_agent"], evaluation=True)
            next_obs, rewards, terminated, truncated, _ = env.step({"demo_agent": action})
            
            episode_reward += rewards.get("demo_agent", 0)
            steps += 1
            
            if terminated.get("demo_agent") or truncated.get("demo_agent"):
                break
            
            obs = next_obs
        
        print(f"  Reward: {episode_reward:.3f}, Steps: {steps}")
    
    print("✅ Quick demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(quick_demo())
'''
    
    demo_file = Path(__file__).parent / "quick_rl_demo.py"
    with open(demo_file, 'w') as f:
        f.write(demo_script)
    
    print(f"✅ Created quick demo script: {demo_file}")
    return demo_file

async def main():
    """Main setup and verification function"""
    print("="*60)
    print("RL FRAMEWORK SETUP AND VERIFICATION")
    print("="*60)
    
    success = True
    
    # Step 1: Check framework structure
    if not verify_framework_structure():
        print("❌ Framework structure verification failed")
        success = False
    
    # Step 2: Install requirements
    if success:
        if not install_requirements():
            print("❌ Requirements installation failed")
            success = False
    
    # Step 3: Check dependencies
    if success:
        if not check_dependencies():
            print("❌ Dependency check failed")
            success = False
    
    # Step 4: Check GPU support
    if success:
        check_gpu_support()  # Non-critical
    
    # Step 5: Run basic tests
    if success:
        if not await run_basic_tests():
            print("❌ Basic functionality tests failed")
            success = False
    
    # Step 6: Create quick demo
    if success:
        demo_file = create_quick_demo()
        print(f"\nTo test the framework, run: python {demo_file.name}")
    
    print("\n" + "="*60)
    if success:
        print("✅ RL FRAMEWORK SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nFramework is ready to use! Next steps:")
        print("1. Run quick demo: python quick_rl_demo.py")
        print("2. Run full tests: python test_rl_system.py") 
        print("3. Run training demo: python rl_training_demo.py --demo_type trading")
        print("4. Check documentation: rl/README.md")
        
        # Performance tips
        print("\n🚀 Performance Tips:")
        print("- Use GPU if available for faster training")
        print("- Adjust batch_size based on your memory capacity")
        print("- Start with smaller environments for initial testing")
        print("- Use curriculum learning for better sample efficiency")
        
        # Business applications
        print("\n💼 Business Applications:")
        print("- Algorithmic trading with multi-agent strategies")
        print("- Supply chain optimization and risk management") 
        print("- Dynamic resource allocation and scheduling")
        print("- Portfolio management and risk assessment")
        
    else:
        print("❌ RL FRAMEWORK SETUP FAILED")
        print("="*60)
        print("Please check the error messages above and resolve issues.")
        print("Common solutions:")
        print("- Ensure Python 3.8+ is installed")
        print("- Run: pip install --upgrade pip")
        print("- Install missing packages manually")
        print("- Check internet connection for package downloads")

if __name__ == "__main__":
    asyncio.run(main())