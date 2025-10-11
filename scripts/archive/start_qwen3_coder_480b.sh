#!/bin/bash

# Start qwen3-coder-480b-mlx model on both workers
# Run this script on mac-studio-1 (coordinator node)

echo "Starting qwen3-coder-480b-mlx distributed inference..."

# Kill any existing processes
echo "Stopping existing services..."
pkill -f "python -m coordinator.main" || true
pkill -f "python -m worker.main" || true
pkill -f "python -m api.main" || true

sleep 2

# Start coordinator
echo "Starting coordinator..."
nohup python -m coordinator.main > logs/coordinator.log 2>&1 &
COORDINATOR_PID=$!
echo "Coordinator started with PID: $COORDINATOR_PID"

sleep 5

# Start worker on mac-studio-1
echo "Starting worker on mac-studio-1..."
nohup python -m worker.main \
    --worker-id worker-studio1 \
    --port 8080 \
    --coordinator localhost:50051 \
    --model-path "/Volumes/thunderbay/models/qwen3-coder-480b-mlx" \
    --model-name "qwen3-coder-480b-mlx" \
    > logs/worker-studio1.log 2>&1 &
WORKER1_PID=$!
echo "Worker-studio1 started with PID: $WORKER1_PID"

sleep 3

# Start API server
echo "Starting API server..."
nohup python -m api.main > logs/api.log 2>&1 &
API_PID=$!
echo "API server started with PID: $API_PID"

echo ""
echo "Services started:"
echo "- Coordinator PID: $COORDINATOR_PID"
echo "- Worker-studio1 PID: $WORKER1_PID" 
echo "- API Server PID: $API_PID"
echo ""
echo "Now start worker on mac-studio-2 using:"
echo "ssh 10.201.20.12"
echo "cd ~/mlx-distributed-inference"
echo "source venv/bin/activate"
echo "python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.10:50051 --model-path \"/Volumes/models/mlx/qwen3-coder-480b-mlx\" --model-name \"qwen3-coder-480b-mlx\""
echo ""
echo "Monitor logs with:"
echo "tail -f logs/coordinator.log"
echo "tail -f logs/worker-studio1.log"
echo "tail -f logs/api.log"