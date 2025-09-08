#!/usr/bin/env python3
"""
AI Therapy System - Simple Startup Script
Provides easy commands to run, test, and manage the system
"""

import sys
import os
import subprocess
import argparse
import time
import requests
from pathlib import Path
from dotenv import load_dotenv



# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

load_dotenv()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required. Current version:", sys.version)
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'transformers', 'torch', 'numpy', 'requests'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing.append(package)
            print(f"❌ {package}")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def check_ollama():
    """Check if Ollama is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"✅ Ollama ({len(models)} models available)")
            return True
        else:
            print("⚠️  Ollama server error")
            return False
    except requests.exceptions.RequestException:
        print("❌ Ollama not available")
        return False

def check_groq():
    """Check if Groq API key is configured"""
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        print("✅ Groq API key configured")
        return True
    else:
        print("❌ Groq API key not set")
        return False

def run_quick_test():
    """Run quick system test"""
    print("\n🧪 Running quick test...")
    try:
        from test_system import quick_test
        return quick_test()
    except Exception as e:
        print(f"❌ Quick test failed: {e}")
        return False

def run_full_tests():
    """Run comprehensive test suite"""
    print("\n🧪 Running full test suite...")
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_ollama():
    """Guide user through Ollama setup"""
    print("\n🤖 Ollama Setup Guide")
    print("-" * 30)
    
    # Check if Ollama is installed
    try:
        result = subprocess.run(["ollama", "--version"], 
                              capture_output=True, text=True)
        print("✅ Ollama is installed")
    except FileNotFoundError:
        print("❌ Ollama not found. Please install:")
        print("  macOS: brew install ollama")
        print("  Linux: curl -fsSL https://ollama.ai/install.sh | sh")
        print("  Windows: Download from https://ollama.ai/download")
        return False
    
    # Check if Ollama is running
    if not check_ollama():
        print("\n🚀 Starting Ollama...")
        print("Run in another terminal: ollama serve")
        
        # Wait for user to start Ollama
        input("Press Enter after starting Ollama...")
        
        if not check_ollama():
            print("❌ Ollama still not accessible")
            return False
    
    # Check for models
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = response.json().get('models', [])
        
        if not models:
            print("\n📥 No models found. Pulling llama2...")
            subprocess.run(["ollama", "pull", "llama2"], check=True)
            print("✅ llama2 model installed")
        else:
            print(f"✅ Found {len(models)} models")
            for model in models[:3]:  # Show first 3
                print(f"  - {model.get('name', 'unknown')}")
    
    except Exception as e:
        print(f"⚠️  Could not check models: {e}")
    
    return True

def start_server(host="127.0.0.1", port=5000, debug=False):
    """Start the Flask server"""
    print(f"\n🚀 Starting AI Therapy System...")
    print(f"Server: http://{host}:{port}")
    print("Press Ctrl+C to stop")
    print("-" * 40)
    
    # Set environment variables
    os.environ['HOST'] = host
    os.environ['PORT'] = str(port)
    if debug:
        os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Import and run the app
        from app import app, initialize_components
        
        # Initialize components first
        print("Initializing AI components...")
        initialize_components()
        
        # Start the server
        app.run(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        return False
    
    return True

def system_status():
    """Show comprehensive system status"""
    print("\n📊 AI Therapy System Status")
    print("=" * 40)
    
    # Python version
    print("\n🐍 Python Environment:")
    check_python_version()
    
    # Dependencies
    print("\n📦 Dependencies:")
    deps_ok = check_dependencies()
    
    # LLM Providers
    print("\n🤖 LLM Providers:")
    ollama_ok = check_ollama()
    groq_ok = check_groq()
    
    # System health
    print("\n🏥 System Health:")
    try:
        response = requests.get("http://127.0.0.1:5000/health", timeout=3)
        if response.status_code == 200:
            print("✅ Server responding")
            health_data = response.json()
            print(f"✅ Components: {health_data.get('components', {})}")
        else:
            print("⚠️  Server error")
    except requests.exceptions.RequestException:
        print("❌ Server not running")
    
    # Overall status
    print("\n🎯 Overall Status:")
    if deps_ok and (ollama_ok or groq_ok):
        print("✅ System ready for use")
    elif deps_ok:
        print("⚠️  System functional (fallback mode only)")
    else:
        print("❌ System needs setup")

def create_config():
    """Create configuration file"""
    print("\n⚙️  Creating configuration file...")
    try:
        from config import create_example_config_file
        create_example_config_file()
        print("✅ Configuration file created: config.json")
        print("Edit config.json to customize settings")
    except Exception as e:
        print(f"❌ Failed to create config: {e}")

def main():
    """Main command-line interface"""
    parser = argparse.ArgumentParser(
        description="AI Therapy System - Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                    # Show system status
  python run.py --start            # Start the server
  python run.py --test             # Run quick test
  python run.py --setup            # Setup system
  python run.py --start --debug    # Start in debug mode
        """
    )
    
    parser.add_argument('--start', action='store_true', 
                       help='Start the web server')
    parser.add_argument('--test', action='store_true',
                       help='Run quick system test')
    parser.add_argument('--test-full', action='store_true',
                       help='Run comprehensive test suite')
    parser.add_argument('--setup', action='store_true',
                       help='Setup system (install deps, configure)')
    parser.add_argument('--install', action='store_true',
                       help='Install dependencies only')
    parser.add_argument('--config', action='store_true',
                       help='Create configuration file')
    parser.add_argument('--ollama', action='store_true',
                       help='Setup Ollama')
    
    # Server options
    parser.add_argument('--host', default='127.0.0.1',
                       help='Server host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Server port (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Run server in debug mode')
    
    args = parser.parse_args()
    
    print("🧠 AI Therapy Assistant - Phase 1")
    print("=" * 40)
    
    # Handle commands
    if args.install:
        success = install_dependencies()
        sys.exit(0 if success else 1)
    
    elif args.config:
        create_config()
    
    elif args.ollama:
        success = setup_ollama()
        sys.exit(0 if success else 1)
    
    elif args.setup:
        print("🔧 Setting up system...")
        
        # Check Python
        if not check_python_version():
            sys.exit(1)
        
        # Install dependencies
        if not install_dependencies():
            sys.exit(1)
        
        # Create config
        create_config()
        
        # Setup Ollama (optional)
        print("\n🤖 Would you like to setup Ollama for local LLM? (y/n):")
        if input().lower().startswith('y'):
            setup_ollama()
        
        # Run test
        if run_quick_test():
            print("\n🎉 Setup complete! System is ready to use.")
            print("Run: python run.py --start")
        else:
            print("\n⚠️  Setup completed with issues. Check the test output above.")
    
    elif args.test:
        success = run_quick_test()
        sys.exit(0 if success else 1)
    
    elif args.test_full:
        success = run_full_tests()
        sys.exit(0 if success else 1)
    
    elif args.start:
        start_server(args.host, args.port, args.debug)
    
    else:
        # Default: show status
        system_status()
        print("\nUse --help for command options")


if __name__ == "__main__":
    main()
