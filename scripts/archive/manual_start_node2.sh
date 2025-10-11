#!/bin/bash

# Manual startup script for node 2 worker
# Run this script ON node 2 (mac-studio-2)

echo "🚀 Starting Worker on Node 2"
echo "============================="

# Activate virtual environment
source ~/mlx-distributed-inference/venv/bin/activate

# Set environment variables
export COORDINATOR_HOST="10.201.20.11:50051"
export WORKER_ID="worker-studio2"
export WORKER_PORT="8080"
export MODEL_NAME="/Volumes/thunderbay/models/qwen3-480b-streaming-q4"

# Create logs directory
mkdir -p ~/mlx-distributed-inference/logs

# Stop any existing worker
pkill -f "python -m worker.main" || true
sleep 2

echo "Starting worker with:"
echo "  Worker ID: $WORKER_ID"
echo "  Port: $WORKER_PORT"
echo "  Coordinator: $COORDINATOR_HOST"
echo "  Model: $MODEL_NAME"
echo ""

# Start the worker
cd ~/mlx-distributed-inference
python -m worker.main \
    --worker-id "$WORKER_ID" \
    --port "$WORKER_PORT" \
    --coordinator "$COORDINATOR_HOST" \
    --model-name "$MODEL_NAME" \
    2>&1 | tee logs/worker-studio2.log