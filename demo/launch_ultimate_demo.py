"""
Master Demo Launcher for AI Document Intelligence Platform
Launches all demo components with proper configuration and monitoring
"""

import subprocess
import sys
import time
import json
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional
import threading
import signal
import os
from datetime import datetime

class DemoLauncher:
    """Master controller for launching and managing all demo components"""
    
    def __init__(self):
        self.demo_root = Path(__file__).parent
        self.processes: List[subprocess.Popen] = []
        self.demo_configs = {
            "ultimate_demo": {
                "file": "ultimate_demo.py",
                "port": 8501,
                "name": "Ultimate Demo - Main Showcase",
                "audience": "All audiences - comprehensive overview",
                "priority": 1,
                "auto_open": True
            },
            "interactive_presentation": {
                "file": "interactive_presentation.py", 
                "port": 8502,
                "name": "Interactive Presentation - Professional Slides",
                "audience": "Board meetings, investor presentations",
                "priority": 2,
                "auto_open": False
            },
            "benchmark_showcase": {
                "file": "benchmark_showcase.py",
                "port": 8503,
                "name": "Benchmark Showcase - Performance Analysis", 
                "audience": "Technical evaluations, RFP responses",
                "priority": 3,
                "auto_open": False
            },
            "agent_visualization": {
                "file": "agent_visualization.py",
                "port": 8504,
                "name": "Agent Visualization - 3D Networks & Swarm Intelligence",
                "audience": "Technical teams, AI enthusiasts",
                "priority": 4,
                "auto_open": False
            },
            "business_calculator": {
                "file": "business_calculator.py",
                "port": 8505,
                "name": "Business Calculator - ROI & Financial Analysis",
                "audience": "CFO, finance teams, procurement",
                "priority": 5,
                "auto_open": False
            }
        }
        
        self.running_demos: Dict[str, subprocess.Popen] = {}
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        required_packages = [
            ("streamlit", "streamlit"), 
            ("plotly", "plotly"), 
            ("pandas", "pandas"), 
            ("numpy", "numpy"), 
            ("networkx", "networkx"), 
            ("PIL", "pillow"), 
            ("dateutil", "python-dateutil")
        ]
        
        missing_packages = []
        for import_name, package_name in required_packages:
            try:
                __import__(import_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            print(f"[ERROR] Missing required packages: {', '.join(missing_packages)}")
            print("[INSTALL] Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("[OK] All dependencies satisfied")
        return True
    
    def check_port_availability(self, port: int) -> bool:
        """Check if a port is available"""
        import socket
        
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('localhost', port))
                return True
            except OSError:
                return False
    
    def find_available_port(self, start_port: int) -> int:
        """Find next available port starting from given port"""
        port = start_port
        while not self.check_port_availability(port):
            port += 1
            if port > start_port + 100:  # Prevent infinite loop
                raise Exception(f"Could not find available port starting from {start_port}")
        return port
    
    def launch_demo(self, demo_name: str, custom_port: Optional[int] = None) -> bool:
        """Launch a specific demo component"""
        if demo_name not in self.demo_configs:
            print(f"[ERROR] Unknown demo: {demo_name}")
            return False
        
        config = self.demo_configs[demo_name]
        demo_file = self.demo_root / config["file"]
        
        if not demo_file.exists():
            print(f"[ERROR] Demo file not found: {demo_file}")
            return False
        
        # Determine port
        port = custom_port or config["port"]
        if not self.check_port_availability(port):
            print(f"[WARNING] Port {port} in use, finding alternative...")
            port = self.find_available_port(port)
            print(f"[INFO] Using port {port} instead")
        
        # Launch command
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(demo_file),
            "--server.port", str(port),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--server.fileWatcherType", "none"
        ]
        
        try:
            print(f"[LAUNCH] Starting {config['name']} on port {port}...")
            process = subprocess.Popen(
                cmd,
                cwd=self.demo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            # Store process
            self.running_demos[demo_name] = process
            self.processes.append(process)
            
            # Wait for startup
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is None:
                print(f"[SUCCESS] {config['name']} launched successfully!")
                print(f"[URL] http://localhost:{port}")
                
                # Auto-open browser if configured
                if config.get("auto_open", False):
                    webbrowser.open(f"http://localhost:{port}")
                
                return True
            else:
                print(f"[ERROR] Failed to launch {config['name']}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Error launching {demo_name}: {str(e)}")
            return False
    
    def launch_all_demos(self) -> Dict[str, bool]:
        """Launch all demo components"""
        print("[DEMO SUITE] Launching Complete Demo Suite...")
        print("=" * 60)
        
        results = {}
        
        # Sort by priority
        sorted_demos = sorted(
            self.demo_configs.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for demo_name, config in sorted_demos:
            results[demo_name] = self.launch_demo(demo_name)
            time.sleep(2)  # Stagger launches
        
        return results
    
    def launch_custom_configuration(self, demo_selection: List[str]) -> Dict[str, bool]:
        """Launch selected demo components"""
        print(f"[CUSTOM] Launching Custom Demo Configuration...")
        print("=" * 50)
        
        results = {}
        for demo_name in demo_selection:
            results[demo_name] = self.launch_demo(demo_name)
            time.sleep(2)
        
        return results
    
    def show_demo_status(self):
        """Show status of all running demos"""
        print("\n[STATUS] Demo Status Dashboard")
        print("=" * 50)
        
        if not self.running_demos:
            print("No demos currently running")
            return
        
        for demo_name, process in self.running_demos.items():
            config = self.demo_configs[demo_name]
            status = "[RUNNING]" if process.poll() is None else "[STOPPED]"
            print(f"{status} | {config['name']} | Port: {config['port']}")
            print(f"         URL: http://localhost:{config['port']}")
            print(f"         Audience: {config['audience']}")
            print()
    
    def stop_demo(self, demo_name: str) -> bool:
        """Stop a specific demo"""
        if demo_name not in self.running_demos:
            print(f"[ERROR] Demo {demo_name} is not running")
            return False
        
        process = self.running_demos[demo_name]
        try:
            process.terminate()
            process.wait(timeout=5)
            del self.running_demos[demo_name]
            self.processes.remove(process)
            print(f"[STOPPED] {self.demo_configs[demo_name]['name']}")
            return True
        except subprocess.TimeoutExpired:
            process.kill()
            del self.running_demos[demo_name]
            self.processes.remove(process)
            print(f"[KILLED] Force-killed {self.demo_configs[demo_name]['name']}")
            return True
        except Exception as e:
            print(f"[ERROR] Error stopping {demo_name}: {str(e)}")
            return False
    
    def stop_all_demos(self):
        """Stop all running demos"""
        print("[SHUTDOWN] Stopping all demos...")
        
        for demo_name in list(self.running_demos.keys()):
            self.stop_demo(demo_name)
        
        print("[COMPLETE] All demos stopped")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for clean shutdown"""
        def signal_handler(signum, frame):
            print("\n[INTERRUPT] Received interrupt signal, shutting down demos...")
            self.stop_all_demos()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
    
    def create_demo_report(self) -> str:
        """Create a demo session report"""
        report = {
            "session_start": datetime.now().isoformat(),
            "demos_launched": list(self.running_demos.keys()),
            "urls": {
                name: f"http://localhost:{config['port']}" 
                for name, config in self.demo_configs.items() 
                if name in self.running_demos
            },
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "demo_root": str(self.demo_root)
            }
        }
        
        report_file = self.demo_root / f"demo_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        return str(report_file)

def print_banner():
    """Print welcome banner"""
    banner = """
    ================================================================
    |                                                              |
    |        AI DOCUMENT INTELLIGENCE PLATFORM DEMO               |
    |                                                              |
    |              Ultimate Demo Launcher System                   |
    |                                                              |
    |     Transform Document Processing with AI Excellence         |
    |                                                              |
    ================================================================
    """
    print(banner)

def show_menu():
    """Show main menu options"""
    print("\n[MENU] Demo Launch Options:")
    print("=" * 40)
    print("1. [MAIN] Launch Ultimate Demo (Recommended)")
    print("2. [CALC] Launch Business Calculator") 
    print("3. [VIZ] Launch Agent Visualization")
    print("4. [BENCH] Launch Benchmark Showcase")
    print("5. [PRES] Launch Interactive Presentation")
    print("6. [ALL] Launch ALL Demos")
    print("7. [STATUS] Show Demo Status")
    print("8. [STOP] Stop All Demos")
    print("9. [HELP] Help & Information")
    print("0. [EXIT] Exit")
    print()

def show_help():
    """Show help information"""
    help_text = """
    [GUIDE] AI Document Intelligence Platform Demo Guide
    ===================================================
    
    [COMPONENTS] Demo Components:
    
    [MAIN] Ultimate Demo (Port 8501) - RECOMMENDED START
    * Comprehensive system overview
    * Live document processing
    * Multi-stakeholder dashboards
    * Real-time ROI calculations
    * Best for: All audiences, first demonstrations
    
    [CALC] Business Calculator (Port 8505)
    * Interactive ROI calculator
    * Scenario modeling and comparison
    * Sensitivity analysis
    * Financial projections
    * Best for: CFOs, finance teams, budget committees
    
    [VIZ] Agent Visualization (Port 8504)
    * 3D agent network visualization
    * Swarm intelligence animations
    * Real-time performance metrics
    * Emergent behavior analysis
    * Best for: Technical teams, AI enthusiasts
    
    [BENCH] Benchmark Showcase (Port 8503)
    * Performance comparisons
    * Competitive analysis
    * Technical deep dives
    * Historical trends
    * Best for: Technical evaluations, RFPs
    
    [PRES] Interactive Presentation (Port 8502)
    * Professional slide presentation
    * Board-ready format
    * Audience-specific views
    * Navigation controls
    * Best for: Board meetings, investor presentations
    
    [TIPS] Tips for Success:
    * Start with Ultimate Demo for comprehensive overview
    * Use Business Calculator for financial stakeholders
    * Multi-screen setup: Main demo + specialized views
    * Have stable internet connection
    * Use Chrome browser for best performance
    
    [AUDIENCE] Audience Recommendations:
    * CEOs/Executives: Ultimate Demo -> Market Position
    * CFOs/Finance: Business Calculator -> Custom ROI
    * CTOs/Technical: Agent Visualization -> Benchmarks
    * COOs/Operations: Ultimate Demo -> Live Processing
    
    [SUPPORT] Support:
    * Check demo_setup_guide.md for detailed instructions
    * Review presentation scripts in scripts/ directory
    * Use sample documents in samples/ directory
    """
    print(help_text)

def main():
    """Main demo launcher interface"""
    print_banner()
    
    launcher = DemoLauncher()
    launcher.setup_signal_handlers()
    
    # Check dependencies
    if not launcher.check_dependencies():
        print("\n[ERROR] Please install missing dependencies and try again")
        sys.exit(1)
    
    print("[OK] Demo launcher initialized successfully!")
    
    # Interactive menu loop
    while True:
        show_menu()
        
        try:
            choice = input("Select option (0-9): ").strip()
            
            if choice == "1":
                print("\n[LAUNCH] Starting Ultimate Demo...")
                launcher.launch_demo("ultimate_demo")
                
            elif choice == "2":
                print("\n[LAUNCH] Starting Business Calculator...")
                launcher.launch_demo("business_calculator")
                
            elif choice == "3":
                print("\n[LAUNCH] Starting Agent Visualization...")
                launcher.launch_demo("agent_visualization")
                
            elif choice == "4":
                print("\n[LAUNCH] Starting Benchmark Showcase...")
                launcher.launch_demo("benchmark_showcase")
                
            elif choice == "5":
                print("\n[LAUNCH] Starting Interactive Presentation...")
                launcher.launch_demo("interactive_presentation")
                
            elif choice == "6":
                print("\n[LAUNCH ALL] Starting ALL Demo Components...")
                results = launcher.launch_all_demos()
                
                print("\n[SUMMARY] Launch Summary:")
                for demo, success in results.items():
                    status = "[SUCCESS]" if success else "[FAILED] "
                    print(f"{status} | {launcher.demo_configs[demo]['name']}")
                
                if any(results.values()):
                    print("\n[DEMO URLS]")
                    for demo_name, success in results.items():
                        if success:
                            config = launcher.demo_configs[demo_name]
                            print(f"  * {config['name']}: http://localhost:{config['port']}")
                
            elif choice == "7":
                launcher.show_demo_status()
                
            elif choice == "8":
                launcher.stop_all_demos()
                
            elif choice == "9":
                show_help()
                
            elif choice == "0":
                print("\n[SHUTDOWN] Shutting down...")
                launcher.stop_all_demos()
                print("[GOODBYE] Thank you for using the AI Document Intelligence Demo!")
                break
                
            else:
                print("[ERROR] Invalid option. Please select 0-9.")
                
            # Add separator
            print("\n" + "="*60)
            
        except KeyboardInterrupt:
            print("\n\n[INTERRUPT] Interrupted by user")
            launcher.stop_all_demos()
            break
        except Exception as e:
            print(f"[ERROR] {str(e)}")

if __name__ == "__main__":
    main()