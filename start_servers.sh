#!/bin/bash
# Start both the main server and tools server

echo "Starting LLM API Servers..."
echo ""

# Start tools server in background
echo "Starting Tools API on port 10006..."
python tools_server.py &
TOOLS_PID=$!

# Wait a moment for tools server to start
sleep 2

# Start main server in background
echo "Starting Main API on port 10007..."
python server.py &
MAIN_PID=$!

echo ""
echo "Both servers started!"
echo "  - Tools API (PID $TOOLS_PID): http://localhost:10006"
echo "  - Main API (PID $MAIN_PID):  http://localhost:10007"
echo ""
echo "Press Ctrl+C to stop both servers..."

# Wait for Ctrl+C
trap "kill $TOOLS_PID $MAIN_PID; exit" INT
wait
