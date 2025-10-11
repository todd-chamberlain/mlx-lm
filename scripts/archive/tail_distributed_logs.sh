#!/bin/bash

# Tail logs from both Mac Studios
# Usage: ./scripts/tail_distributed_logs.sh

set -e

STUDIO2_IP="10.201.20.12"
LOG_DIR="logs"

echo "=== Tailing logs from both Mac Studios ==="
echo "Studio 1 (local): $LOG_DIR/worker-studio1.log"
echo "Studio 2 (remote): $STUDIO2_IP:~/mlx-distributed-inference/logs/worker-studio2.log"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Function to tail local log
tail_local() {
    if [ -f "$LOG_DIR/worker-studio1.log" ]; then
        tail -f "$LOG_DIR/worker-studio1.log" | sed 's/^/[Studio1] /'
    else
        echo "[Studio1] Log file not found: $LOG_DIR/worker-studio1.log"
        sleep 1
        tail_local
    fi
}

# Function to tail remote log
tail_remote() {
    ssh "$STUDIO2_IP" "tail -f ~/mlx-distributed-inference/logs/worker-studio2.log" 2>/dev/null | sed 's/^/[Studio2] /' || {
        echo "[Studio2] SSH connection failed or log file not found"
        sleep 5
        tail_remote
    }
}

# Start both tails in background
tail_local &
LOCAL_PID=$!

tail_remote &
REMOTE_PID=$!

# Wait for interrupt
trap 'kill $LOCAL_PID $REMOTE_PID 2>/dev/null; exit' INT

wait