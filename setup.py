#!/usr/bin/env python3
"""
Setup script for agentZERO-ollama
Handles installation, model downloading, and initial configuration
"""

import os
import sys
import subprocess
import json
import time
import requests
from pathlib import Path

class AgentZeroSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.ollama_url = "http://localhost:11434"
        self.required_models = [
            "smollm:135m",
            "smollm:360m", 
            "smollm:1.7b"
        ]
    
    def check_python_version(self):
        """Check if Python version is compatible"""
        print("🐍 Checking Python version...")
        if sys.version_info < (3, 8):
            print("❌ Python 3.8 or higher is required")
            return False
        print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
        return True
    
    def check_docker(self):
        """Check if Docker is available"""
        print("🐳 Checking Docker...")
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {result.stdout.strip()}")
                return True
        except FileNotFoundError:
            pass
        
        print("❌ Docker not found")
        return False
    
    def check_ollama(self):
        """Check if Ollama is installed and running"""
        print("🤖 Checking Ollama...")
        
        # Check if ollama command exists
        try:
            result = subprocess.run(["ollama", "--version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                print("❌ Ollama not installed")
                return False
        except FileNotFoundError:
            print("❌ Ollama not found")
            return False
        
        # Check if Ollama server is running
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama server is running")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print("⚠️ Ollama installed but server not running")
        return "not_running"
    
    def start_ollama(self):
        """Start Ollama server"""
        print("🚀 Starting Ollama server...")
        try:
            # Start ollama serve in background
            subprocess.Popen(["ollama", "serve"], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            
            # Wait for server to be ready
            for i in range(30):  # Wait up to 30 seconds
                try:
                    response = requests.get(f"{self.ollama_url}/api/tags", timeout=2)
                    if response.status_code == 200:
                        print("✅ Ollama server started successfully")
                        return True
                except requests.exceptions.RequestException:
                    pass
                time.sleep(1)
            
            print("❌ Failed to start Ollama server")
            return False
            
        except Exception as e:
            print(f"❌ Error starting Ollama: {e}")
            return False
    
    def install_ollama(self):
        """Install Ollama"""
        print("📦 Installing Ollama...")
        
        if sys.platform.startswith('linux'):
            # Linux installation
            try:
                subprocess.run([
                    "curl", "-fsSL", "https://ollama.ai/install.sh"
                ], check=True, stdout=subprocess.PIPE)
                subprocess.run(["sh"], input=b"", check=True)
                print("✅ Ollama installed successfully")
                return True
            except subprocess.CalledProcessError:
                print("❌ Failed to install Ollama")
                return False
        
        elif sys.platform == 'darwin':
            # macOS - suggest manual installation
            print("🍎 For macOS, please install Ollama manually:")
            print("   Visit: https://ollama.ai/download")
            print("   Or use: brew install ollama")
            return False
        
        elif sys.platform.startswith('win'):
            # Windows - suggest manual installation
            print("🪟 For Windows, please install Ollama manually:")
            print("   Visit: https://ollama.ai/download")
            return False
        
        else:
            print(f"❌ Unsupported platform: {sys.platform}")
            return False
    
    def check_models(self):
        """Check if required models are available"""
        print("🧠 Checking models...")
        
        try:
            response = requests.get(f"{self.ollama_url}/api/tags")
            if response.status_code != 200:
                print("❌ Cannot connect to Ollama server")
                return False
            
            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]
            
            missing_models = []
            for model in self.required_models:
                if model not in available_models:
                    missing_models.append(model)
                else:
                    print(f"✅ {model} available")
            
            if missing_models:
                print(f"⚠️ Missing models: {', '.join(missing_models)}")
                return missing_models
            
            print("✅ All required models available")
            return True
            
        except Exception as e:
            print(f"❌ Error checking models: {e}")
            return False
    
    def download_models(self, models):
        """Download required models"""
        print("📥 Downloading models...")
        
        for model in models:
            print(f"📥 Downloading {model}...")
            try:
                # Use ollama pull command
                result = subprocess.run(
                    ["ollama", "pull", model],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print(f"✅ {model} downloaded successfully")
                else:
                    print(f"❌ Failed to download {model}: {result.stderr}")
                    return False
                    
            except Exception as e:
                print(f"❌ Error downloading {model}: {e}")
                return False
        
        return True
    
    def install_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
            ], check=True)
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directories...")
        
        directories = ["data", "logs", "cache"]
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            print(f"✅ Created {directory}/")
        
        return True
    
    def create_config(self):
        """Create default configuration"""
        print("⚙️ Creating configuration...")
        
        config = {
            "ollama": {
                "url": self.ollama_url,
                "models": self.required_models
            },
            "ui": {
                "host": "0.0.0.0",
                "port": 8501
            },
            "logging": {
                "level": "INFO",
                "file": "logs/agentzero.log"
            }
        }
        
        config_file = self.project_root / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Configuration created")
        return True
    
    def test_installation(self):
        """Test the installation"""
        print("🧪 Testing installation...")
        
        # Test Ollama connection
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                print("❌ Ollama server not responding")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            return False
        
        # Test model availability
        models_check = self.check_models()
        if models_check != True:
            print("❌ Models not properly installed")
            return False
        
        # Test Python imports
        try:
            sys.path.insert(0, str(self.project_root))
            from core.ollama_client import OllamaClient
            from agents.planner import PlannerAgent
            print("✅ Python modules import successfully")
        except ImportError as e:
            print(f"❌ Import error: {e}")
            return False
        
        print("✅ Installation test passed")
        return True
    
    def run_setup(self):
        """Run the complete setup process"""
        print("🚀 agentZERO-ollama Setup")
        print("=" * 50)
        
        # Check Python version
        if not self.check_python_version():
            return False
        
        # Check and install Ollama
        ollama_status = self.check_ollama()
        if ollama_status == False:
            if not self.install_ollama():
                return False
            ollama_status = self.check_ollama()
        
        if ollama_status == "not_running":
            if not self.start_ollama():
                return False
        
        # Install Python dependencies
        if not self.install_dependencies():
            return False
        
        # Create directories
        if not self.create_directories():
            return False
        
        # Create configuration
        if not self.create_config():
            return False
        
        # Check and download models
        models_check = self.check_models()
        if models_check != True:
            if isinstance(models_check, list):
                if not self.download_models(models_check):
                    return False
            else:
                return False
        
        # Test installation
        if not self.test_installation():
            return False
        
        print("\n🎉 Setup completed successfully!")
        print("\nTo start agentZERO-ollama:")
        print("  python -m streamlit run ui/app.py")
        print("\nOr using Docker:")
        print("  docker-compose up")
        print("\nWeb interface will be available at: http://localhost:8501")
        
        return True

def main():
    """Main setup function"""
    setup = AgentZeroSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "check":
            # Just check status
            setup.check_python_version()
            setup.check_docker()
            setup.check_ollama()
            setup.check_models()
            
        elif command == "models":
            # Only download models
            models_check = setup.check_models()
            if isinstance(models_check, list):
                setup.download_models(models_check)
            
        elif command == "test":
            # Only test installation
            setup.test_installation()
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: check, models, test")
    
    else:
        # Run full setup
        success = setup.run_setup()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

