#!/bin/bash
# Frontend Server Launcher for Linux/Mac
# Run: chmod +x run_frontend.sh && ./run_frontend.sh

set -e

echo "Starting Frontend Server..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Run the frontend server
python3 run_frontend.py
