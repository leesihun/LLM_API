#!/bin/bash

# Start Both Backend and Frontend (Linux/Mac)

echo "=========================================="
echo "HE Team LLM Assistant - Full Stack"
echo "=========================================="

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found"
    echo "Please copy .env.example to .env and configure it"
    exit 1
fi

# Start backend in background
echo "Starting backend server..."
./run_backend.sh &
BACKEND_PID=$!

# Wait for backend to start
sleep 5

# Start frontend
echo ""
echo "Starting frontend server..."
./run_frontend.sh

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
