#!/bin/bash

# Debug script to run worker in foreground and see errors

set -e

echo "🔍 Debug Worker Script"
echo "====================="
echo

# Activate venv
source venv/bin/activate

# Kill any existing processes
echo "Cleaning up existing processes..."
pkill -9 -f "python3 -m" 2>/dev/null || true
sleep 2

# Start coordinator in background
echo "Starting coordinator..."
python3 -m coordinator.main --host 0.0.0.0 --port 50051 > logs/debug-coordinator.log 2>&1 &
COORD_PID=$!
echo "Coordinator PID: $COORD_PID"
sleep 5

# Check if coordinator is running
if ! ps -p $COORD_PID > /dev/null; then
    echo "❌ Coordinator failed to start!"
    cat logs/debug-coordinator.log
    exit 1
fi

echo "✅ Coordinator running"
echo

# Run worker in foreground with verbose output
echo "Starting worker in foreground..."
echo "Press Ctrl+C to stop"
echo "---"

python3 -m worker.main \
    --worker-id debug-worker \
    --coordinator localhost:50051 \
    --model-path ~/models/Llama-3.2-3B-Instruct-4bit \
    --port 50052

# Cleanup on exit
trap "kill $COORD_PID 2>/dev/null" EXIT
