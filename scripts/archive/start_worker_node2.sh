#!/bin/bash

# Start worker on node 2 with proper environment setup
# Usage: ./scripts/start_worker_node2.sh

set -e

STUDIO2_IP="10.201.20.12"
COORDINATOR_IP="10.201.20.11"  # Studio 1 IP
MODEL_NAME="Qwen/Qwen2.5-Coder-32B-Instruct"

echo "🚀 Starting Worker on Node 2 (Studio 2)"
echo "========================================"
echo "Node 2 IP: $STUDIO2_IP"
echo "Coordinator IP: $COORDINATOR_IP"
echo "Model: $MODEL_NAME"
echo ""

# Check SSH connectivity
echo "1. Testing SSH connectivity..."
if ! ssh -o ConnectTimeout=5 "$STUDIO2_IP" "echo 'SSH OK'" >/dev/null 2>&1; then
    echo "❌ SSH connection failed to $STUDIO2_IP"
    exit 1
fi
echo "✅ SSH connection working"

# Check if project exists
echo "2. Checking project directory..."
if ! ssh "$STUDIO2_IP" "test -d ~/mlx-distributed-inference" 2>/dev/null; then
    echo "❌ Project directory not found on node 2"
    echo "Please clone the project to ~/mlx-distributed-inference/ on node 2"
    exit 1
fi
echo "✅ Project directory exists"

# Check virtual environment
echo "3. Checking virtual environment..."
if ! ssh "$STUDIO2_IP" "test -f ~/mlx-distributed-inference/venv/bin/activate" 2>/dev/null; then
    echo "❌ Virtual environment not found on node 2"
    echo "Please set up virtual environment on node 2:"
    echo "  ssh $STUDIO2_IP"
    echo "  cd ~/mlx-distributed-inference"
    echo "  python -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo "✅ Virtual environment exists"

# Stop any existing worker
echo "4. Stopping any existing worker..."
ssh "$STUDIO2_IP" "pkill -f 'python -m worker.main' || true" 2>/dev/null
sleep 2
echo "✅ Existing worker stopped"

# Create logs directory if needed
echo "5. Setting up logs directory..."
ssh "$STUDIO2_IP" "mkdir -p ~/mlx-distributed-inference/logs" 2>/dev/null
echo "✅ Logs directory ready"

# Start the worker
echo "6. Starting worker on node 2..."
ssh "$STUDIO2_IP" "cd ~/mlx-distributed-inference && source venv/bin/activate && nohup python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator $COORDINATOR_IP:50051 --model-name '$MODEL_NAME' > logs/worker-studio2.log 2>&1 &" 

# Wait a moment for startup
sleep 3

# Check if worker started successfully
echo "7. Verifying worker startup..."
WORKER_RUNNING=$(ssh "$STUDIO2_IP" "ps aux | grep 'python -m worker.main' | grep -v grep | wc -l" 2>/dev/null || echo "0")
if [ "$WORKER_RUNNING" -gt 0 ]; then
    echo "✅ Worker is running on node 2"
    ssh "$STUDIO2_IP" "ps aux | grep 'python -m worker.main' | grep -v grep"
else
    echo "❌ Worker failed to start. Checking logs..."
    ssh "$STUDIO2_IP" "tail -10 ~/mlx-distributed-inference/logs/worker-studio2.log" 2>/dev/null || echo "No log output"
    exit 1
fi

echo ""
echo "🎉 Worker successfully started on node 2!"
echo ""
echo "Monitor the worker:"
echo "  ssh $STUDIO2_IP \"tail -f ~/mlx-distributed-inference/logs/worker-studio2.log\""
echo ""
echo "Check worker status:"
echo "  ssh $STUDIO2_IP \"ps aux | grep 'python -m worker.main' | grep -v grep\""