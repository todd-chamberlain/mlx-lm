#!/bin/bash

# Debug script to check what's happening on node 2
# Usage: ./scripts/debug_node2.sh

set -e

STUDIO2_IP="10.201.20.12"

echo "=== Debugging Node 2 (Studio 2) ==="
echo "IP: $STUDIO2_IP"
echo ""

# Check SSH connectivity
echo "1. Testing SSH connectivity..."
if ssh -o ConnectTimeout=5 "$STUDIO2_IP" "echo 'SSH connection successful'" 2>/dev/null; then
    echo "✓ SSH connection working"
else
    echo "✗ SSH connection failed"
    exit 1
fi
echo ""

# Check if the project directory exists
echo "2. Checking project directory..."
ssh "$STUDIO2_IP" "ls -la ~/mlx-distributed-inference/" 2>/dev/null || {
    echo "✗ Project directory not found on node 2"
    echo "Please ensure the project is cloned to ~/mlx-distributed-inference/ on node 2"
    exit 1
}
echo "✓ Project directory exists"
echo ""

# Check Python environment
echo "3. Checking Python environment..."
ssh "$STUDIO2_IP" "cd ~/mlx-distributed-inference && python --version && which python" 2>/dev/null || {
    echo "✗ Python not available or not in correct directory"
}
echo ""

# Check if virtual environment is activated
echo "4. Checking virtual environment..."
ssh "$STUDIO2_IP" "cd ~/mlx-distributed-inference && python -c 'import sys; print(sys.prefix)'" 2>/dev/null
echo ""

# Check MLX installation
echo "5. Checking MLX installation..."
ssh "$STUDIO2_IP" "cd ~/mlx-distributed-inference && python -c 'import mlx.core as mx; print(f\"MLX version: {mx.__version__}\")'" 2>/dev/null || {
    echo "✗ MLX not installed or not working on node 2"
}
echo ""

# Check worker process
echo "6. Checking worker process..."
WORKER_RUNNING=$(ssh "$STUDIO2_IP" "ps aux | grep 'python -m worker.main' | grep -v grep | wc -l" 2>/dev/null || echo "0")
if [ "$WORKER_RUNNING" -gt 0 ]; then
    echo "✓ Worker process is running"
    ssh "$STUDIO2_IP" "ps aux | grep 'python -m worker.main' | grep -v grep"
else
    echo "✗ Worker process is NOT running"
fi
echo ""

# Check log file
echo "7. Checking log file..."
if ssh "$STUDIO2_IP" "test -f ~/mlx-distributed-inference/logs/worker-studio2.log" 2>/dev/null; then
    echo "✓ Log file exists"
    echo "Last 10 lines of log:"
    ssh "$STUDIO2_IP" "tail -10 ~/mlx-distributed-inference/logs/worker-studio2.log" 2>/dev/null
else
    echo "✗ Log file does not exist"
    echo "Checking if logs directory exists..."
    ssh "$STUDIO2_IP" "ls -la ~/mlx-distributed-inference/logs/" 2>/dev/null || {
        echo "✗ Logs directory does not exist"
    }
fi
echo ""

# Check HuggingFace cache
echo "8. Checking HuggingFace cache..."
ssh "$STUDIO2_IP" "ls -la ~/.cache/huggingface/hub/ | grep -i qwen || echo 'No Qwen models in cache'" 2>/dev/null
echo ""

# Check disk space
echo "9. Checking disk space..."
ssh "$STUDIO2_IP" "df -h | grep -E '(Filesystem|/System/Volumes/Data)'" 2>/dev/null
echo ""

# Check network connectivity to coordinator
echo "10. Checking network connectivity to coordinator..."
COORDINATOR_IP=$(hostname -I | awk '{print $1}' || echo "localhost")
ssh "$STUDIO2_IP" "nc -z $COORDINATOR_IP 50051 && echo 'Can connect to coordinator' || echo 'Cannot connect to coordinator'" 2>/dev/null
echo ""

# Check if worker config exists
echo "11. Checking worker configuration..."
if ssh "$STUDIO2_IP" "test -f ~/mlx-distributed-inference/config/worker_config.yaml" 2>/dev/null; then
    echo "✓ Worker config exists"
    echo "Worker config content:"
    ssh "$STUDIO2_IP" "cat ~/mlx-distributed-inference/config/worker_config.yaml" 2>/dev/null
else
    echo "✗ Worker config file missing"
fi
echo ""

echo "=== Debug Summary ==="
echo "If the worker is not running, try starting it manually on node 2:"
echo "ssh $STUDIO2_IP"
echo "cd ~/mlx-distributed-inference"
echo "source venv/bin/activate  # if using virtual environment"
echo "python -m worker.main --model-name Qwen/Qwen2.5-Coder-32B-Instruct --coordinator $COORDINATOR_IP:50051 --worker-id studio2"