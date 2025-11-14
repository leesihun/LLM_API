#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Server Launcher Script (Direct - No Virtual Environment)
Starts the FastAPI backend server using system Python
"""

import os
import sys
import subprocess
from pathlib import Path

# Set UTF-8 encoding for Windows console to handle emojis
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def print_header():
    """Print the application header"""
    print("=" * 70)
    print("HE Team LLM Assistant - Backend")
    print("=" * 70)
    print()


def check_env_file():
    """Check if .env file exists, create if missing"""
    if not Path(".env").exists():
        print("⚠️ Warning: .env file not found")
        print("Creating .env file with default settings...")
        # The settings.py will auto-create it
        print("✅ .env file will be created on first run")
        print()
    return True


def check_python_version():
    """Check if Python version is sufficient"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False

    print(f"✅ Python version: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_dependencies():
    """Check if critical dependencies are installed"""
    print("📦 Checking dependencies...")

    required_packages = [
        'fastapi',
        'uvicorn',
        'langchain',
        'langgraph'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print()
        print("Please install dependencies with:")
        print(f"  pip install -r requirements.txt")
        print()
        return False

    print("✅ All critical dependencies installed")
    return True


def run_server():
    """Run the backend server"""
    print("🚀 Starting backend server...")
    print()

    try:
        # Run server.py using the current Python interpreter
        subprocess.run([sys.executable, "server.py"], check=True)
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

    print()

    # Check .env file
    if not check_env_file():
        sys.exit(1)

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    print()

    # Run server
    if not run_server():
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()

