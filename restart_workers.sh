#!/bin/bash

# Restart workers only (keep coordinator and API running)

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

STUDIO2_HOST="mac-studio-2"
MODEL_NAME="Llama-3.2-3B-Instruct-4bit"
STUDIO1_IP="10.201.20.10"

echo -e "${CYAN}Restarting Workers${NC}"
echo

# Stop existing workers
echo -e "${YELLOW}[1/3]${NC} Stopping existing workers..."
pkill -f "worker.main.*worker-studio1" 2>/dev/null || true
ssh ${STUDIO2_HOST} "pkill -f 'worker.main.*worker-studio2'" 2>/dev/null || true
sleep 2
echo -e "${GREEN}✓${NC} Workers stopped"
echo

# Create logs directory
mkdir -p logs

# Activate venv
source venv/bin/activate

# Start worker on Studio 1
echo -e "${YELLOW}[2/3]${NC} Starting worker on Mac Studio 1..."
nohup python3 -m worker.main \
    --worker-id worker-studio1 \
    --coordinator localhost:50051 \
    --model-path ~/models/${MODEL_NAME} \
    --port 50052 \
    > logs/worker-studio1.log 2>&1 &
WORKER1_PID=$!
echo -e "${GREEN}✓${NC} Worker 1 started (PID: ${WORKER1_PID})"
echo

# Start worker on Studio 2
echo -e "${YELLOW}[3/3]${NC} Starting worker on Mac Studio 2..."
ssh ${STUDIO2_HOST} "cd ~/mlx-distributed-inference && \
    source venv/bin/activate && \
    nohup python3 -m worker.main \
        --worker-id worker-studio2 \
        --coordinator ${STUDIO1_IP}:50051 \
        --model-path ~/models/${MODEL_NAME} \
        --port 50052 \
    > logs/worker-studio2.log 2>&1 &"
sleep 2

if ssh ${STUDIO2_HOST} "pgrep -f 'worker.main.*worker-studio2'" >/dev/null; then
    WORKER2_PID=$(ssh ${STUDIO2_HOST} "pgrep -f 'worker.main.*worker-studio2'")
    echo -e "${GREEN}✓${NC} Worker 2 started (PID: ${WORKER2_PID})"
else
    echo -e "${RED}✗${NC} Failed to start worker 2"
fi
echo

# Wait for registration
echo "Waiting for workers to register (15 seconds)..."
sleep 15

# Check status
echo
echo "Checking registration status..."
python3 test_worker_registration.py 2>&1 | grep -A 10 "Checking registered workers"

echo
echo -e "${GREEN}✓ Workers restarted${NC}"
echo
echo "Check logs:"
echo "  tail -f logs/worker-studio1.log"
echo "  ssh ${STUDIO2_HOST} 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'"
