#!/usr/bin/env python3
"""
Backend Server Launcher Script (Cross-platform Python)
Handles virtual environment setup and starts the FastAPI backend server
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def print_header():
    """Print the application header"""
    print("=" * 70)
    print("HE Team LLM Assistant - Backend")
    print("=" * 70)
    print()


def check_env_file():
    """Check if .env file exists"""
    if not Path(".env").exists():
        print("❌ Error: .env file not found")
        print("Please create a .env file and configure it with your settings")
        print()
        print("Required environment variables:")
        print("  - OLLAMA_HOST (e.g., http://localhost:11434)")
        print("  - OLLAMA_MODEL (e.g., llama2)")
        print("  - SERVER_HOST (e.g., 0.0.0.0)")
        print("  - SERVER_PORT (e.g., 8000)")
        return False
    return True


def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    return True


def get_venv_python():
    """Get the path to the virtual environment Python executable"""
    venv_dir = Path("venv")
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    else:
        return venv_dir / "bin" / "python"


def get_venv_pip():
    """Get the path to the virtual environment pip executable"""
    venv_dir = Path("venv")
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "pip.exe"
    else:
        return venv_dir / "bin" / "pip"


def create_venv():
    """Create virtual environment if it doesn't exist"""
    venv_dir = Path("venv")
    
    if not venv_dir.exists():
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created successfully")
            print()
        except subprocess.CalledProcessError as e:
            print(f"❌ Error creating virtual environment: {e}")
            return False
    else:
        print("✅ Virtual environment already exists")
        print()
    
    return True


def install_dependencies():
    """Install dependencies from requirements.txt"""
    print("📦 Installing dependencies...")
    
    pip_path = get_venv_pip()
    
    if not Path("requirements.txt").exists():
        print("⚠️ Warning: requirements.txt not found, skipping dependency installation")
        print()
        return True
    
    try:
        # Upgrade pip first
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            check=True,
            capture_output=True
        )
        
        # Install dependencies
        subprocess.run(
            [str(pip_path), "install", "-r", "requirements.txt"],
            check=True
        )
        print("✅ Dependencies installed successfully")
        print()
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True


def run_server():
    """Run the backend server"""
    print("🚀 Starting backend server...")
    print()
    
    venv_python = get_venv_python()
    
    try:
        # Run server.py using the virtual environment Python
        subprocess.run([str(venv_python), "server.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running server: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
        return True
    
    return True


def main():
    """Main function to orchestrate backend startup"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check .env file
    if not check_env_file():
        sys.exit(1)
    
    print("✅ Environment file found")
    print()
    
    # Create virtual environment
    if not create_venv():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Run server
    if not run_server():
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

