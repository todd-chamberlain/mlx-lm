#!/bin/bash

# Direct fix for node 2 worker startup
echo "🔧 Fixing node 2 worker startup..."

# Stop any existing worker on node 2
ssh 10.201.20.12 "pkill -f 'python -m worker.main' || true"
sleep 2

# Start worker with correct parameters
echo "Starting worker on node 2 with correct coordinator IP..."
ssh 10.201.20.12 "cd ~/mlx-distributed-inference && source venv/bin/activate && nohup python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.10:50051 --model-path '/Volumes/thunderbay/models/qwen3-480b-streaming-q4' --model-name 'qwen3-480b-streaming-q4' > logs/worker-studio2.log 2>&1 &"

sleep 3

# Check if it's running
echo "Checking worker status..."
WORKER_COUNT=$(ssh 10.201.20.12 "ps aux | grep 'python -m worker.main' | grep -v grep | wc -l" 2>/dev/null || echo "0")

if [ "$WORKER_COUNT" -gt 0 ]; then
    echo "✅ Worker is running on node 2"
    ssh 10.201.20.12 "ps aux | grep 'python -m worker.main' | grep -v grep"
    echo ""
    echo "Monitor logs: ssh 10.201.20.12 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'"
else
    echo "❌ Worker failed to start. Checking logs..."
    ssh 10.201.20.12 "tail -10 ~/mlx-distributed-inference/logs/worker-studio2.log" 2>/dev/null || echo "No log output"
fi