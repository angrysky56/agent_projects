#!/bin/bash

# Function to cleanup background processes on exit
cleanup() {
    echo "Stopping COMPASS..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID
    fi
    if [ -n "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID
    fi
    exit
}

# Trap SIGINT (Ctrl+C)
trap cleanup SIGINT

echo "Starting COMPASS Unified Cognitive System..."

# Kill any existing processes on ports 3000 and 8000
fuser -k 3000/tcp > /dev/null 2>&1
fuser -k 8000/tcp > /dev/null 2>&1

# Start Backend
echo "Starting Backend API Server..."
python3 -m backend.api_server &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 2

# Start Frontend
echo "Starting Web UI..."
cd web-ui
npm run dev &
FRONTEND_PID=$!
cd ..

echo "COMPASS is running!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "Press Ctrl+C to stop."

# Keep script running to maintain background processes
wait
