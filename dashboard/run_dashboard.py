"""
Quick Start Runner for Enterprise Dashboard
Alternative entry point for running the dashboard directly with Python
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add the dashboard directory to the Python path
dashboard_dir = Path(__file__).parent
sys.path.insert(0, str(dashboard_dir))

def setup_environment():
    """Setup environment variables and configuration"""
    
    # Set Streamlit configuration
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_THEME_PRIMARY_COLOR'] = '#2a5298'
    os.environ['STREAMLIT_THEME_BACKGROUND_COLOR'] = '#ffffff'
    os.environ['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR'] = '#f0f2f6'
    
    # Dashboard-specific environment
    os.environ['DASHBOARD_ENVIRONMENT'] = 'development'
    os.environ['DASHBOARD_DEBUG'] = 'true'
    
    # Create logs directory if it doesn't exist
    logs_dir = dashboard_dir / 'logs'
    logs_dir.mkdir(exist_ok=True)
    
    print("🚀 Enterprise Dashboard - Quick Start")
    print("=" * 50)
    print(f"📁 Dashboard Directory: {dashboard_dir}")
    print(f"🌐 Server Port: {os.environ['STREAMLIT_SERVER_PORT']}")
    print(f"🔧 Environment: {os.environ['DASHBOARD_ENVIRONMENT']}")
    print("=" * 50)

def check_dependencies():
    """Check if required dependencies are installed"""
    
    required_packages = [
        'streamlit',
        'plotly', 
        'pandas',
        'numpy',
        'redis'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements-dashboard.txt")
        print("\n   Or run the deployment script:")
        print("   .\\deploy.ps1 -Install")
        return False
    
    print("✅ All required packages are installed")
    return True

def run_dashboard():
    """Run the Streamlit dashboard"""
    
    try:
        main_dashboard = dashboard_dir / 'main_dashboard.py'
        
        if not main_dashboard.exists():
            print(f"❌ Dashboard file not found: {main_dashboard}")
            return False
        
        print("🚀 Starting Enterprise Dashboard...")
        print("   Access the dashboard at: http://localhost:8501")
        print("   Use Ctrl+C to stop the server")
        print("-" * 50)
        
        # Run Streamlit
        cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            str(main_dashboard),
            '--server.port', os.environ['STREAMLIT_SERVER_PORT'],
            '--server.address', os.environ['STREAMLIT_SERVER_ADDRESS'],
            '--browser.gatherUsageStats', 'false',
            '--theme.primaryColor', os.environ['STREAMLIT_THEME_PRIMARY_COLOR'],
            '--theme.backgroundColor', os.environ['STREAMLIT_THEME_BACKGROUND_COLOR'],
            '--theme.secondaryBackgroundColor', os.environ['STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR']
        ]
        
        subprocess.run(cmd)
        return True
        
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return False

def main():
    """Main entry point"""
    
    setup_environment()
    
    if not check_dependencies():
        sys.exit(1)
    
    if not run_dashboard():
        sys.exit(1)
    
    print("✨ Dashboard session completed")

if __name__ == "__main__":
    main()