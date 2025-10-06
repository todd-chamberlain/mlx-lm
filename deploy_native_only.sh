#!/bin/bash

# Native-only deployment (no Docker) for quick testing
# Runs API, coordinator, and workers all natively

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
STUDIO2_HOST="mac-studio-2"
PROJECT_DIR="$(pwd)"
MODEL_NAME="Llama-3.2-3B-Instruct-4bit"
MODEL_PATH="~/models/${MODEL_NAME}"
STUDIO1_IP=$(ipconfig getifaddr en0)

echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  MLX Distributed Inference - Native Deployment             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo "Mode: Native (no Docker)"
echo "Studio 1 IP: $STUDIO1_IP"
echo

# Stop any existing processes
echo -e "${YELLOW}[1/5]${NC} Stopping existing processes..."
pkill -f "python3 -m api.main" 2>/dev/null || true
pkill -f "python3 -m coordinator.main" 2>/dev/null || true
pkill -f "python3 -m worker.main" 2>/dev/null || true
ssh ${STUDIO2_HOST} "pkill -f 'python3 -m worker.main'" 2>/dev/null || true
sleep 2
echo -e "${GREEN}✓${NC} Cleanup complete"
echo

# Create logs directory
mkdir -p logs

# Activate venv
source venv/bin/activate

# Start coordinator on Studio 1
echo -e "${YELLOW}[2/5]${NC} Starting coordinator on Mac Studio 1..."
python3 -m coordinator.main \
    --host 0.0.0.0 \
    --port 50051 \
    > logs/coordinator.log 2>&1 &
COORDINATOR_PID=$!
sleep 3

if ps -p $COORDINATOR_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} Coordinator started (PID: ${COORDINATOR_PID})"
else
    echo -e "${RED}✗${NC} Failed to start coordinator"
    cat logs/coordinator.log
    exit 1
fi
echo

# Start API server on Studio 1
echo -e "${YELLOW}[3/5]${NC} Starting API server on Mac Studio 1..."
export COORDINATOR_HOST="localhost:50051"
python3 -m api.main \
    --host 0.0.0.0 \
    --port 8000 \
    > logs/api.log 2>&1 &
API_PID=$!
sleep 3

if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} API server started (PID: ${API_PID})"
else
    echo -e "${RED}✗${NC} Failed to start API server"
    cat logs/api.log
    exit 1
fi
echo

# Start worker on Studio 1
echo -e "${YELLOW}[4/5]${NC} Starting worker on Mac Studio 1..."
python3 -m worker.main \
    --worker-id worker-studio1 \
    --coordinator localhost:50051 \
    --model-path ~/models/${MODEL_NAME} \
    --port 50052 \
    > logs/worker-studio1.log 2>&1 &
WORKER1_PID=$!
sleep 5

if ps -p $WORKER1_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} Worker 1 started (PID: ${WORKER1_PID})"
else
    echo -e "${RED}✗${NC} Failed to start worker 1"
    cat logs/worker-studio1.log
    exit 1
fi
echo

# Start worker on Studio 2
echo -e "${YELLOW}[5/5]${NC} Starting worker on Mac Studio 2..."
ssh ${STUDIO2_HOST} "cd ~/mlx-distributed-inference && \
    source venv/bin/activate && \
    nohup python3 -m worker.main \
        --worker-id worker-studio2 \
        --coordinator ${STUDIO1_IP}:50051 \
        --model-path ~/models/${MODEL_NAME} \
        --port 50052 \
    > logs/worker-studio2.log 2>&1 &"
sleep 5

if ssh ${STUDIO2_HOST} "pgrep -f 'python3 -m worker.main'" >/dev/null; then
    WORKER2_PID=$(ssh ${STUDIO2_HOST} "pgrep -f 'python3 -m worker.main'")
    echo -e "${GREEN}✓${NC} Worker 2 started (PID: ${WORKER2_PID})"
else
    echo -e "${RED}✗${NC} Failed to start worker 2"
    ssh ${STUDIO2_HOST} "cat ~/mlx-distributed-inference/logs/worker-studio2.log"
    exit 1
fi
echo

# Wait for workers to register
echo "Waiting for workers to register..."
sleep 10

# Check health
echo "Checking system health..."
HEALTH=$(curl -s http://localhost:8000/health 2>/dev/null || echo "{}")
WORKER_COUNT=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workers', 0))" 2>/dev/null || echo "0")

echo
echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Deployment Complete!                                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo
echo -e "${GREEN}✓ Mac Studio 1:${NC}"
echo "  • IP: ${STUDIO1_IP}"
echo "  • Coordinator: localhost:50051 (PID: ${COORDINATOR_PID})"
echo "  • API Server: http://localhost:8000 (PID: ${API_PID})"
echo "  • Worker: worker-studio1 (PID: ${WORKER1_PID})"
echo
echo -e "${GREEN}✓ Mac Studio 2:${NC}"
echo "  • Host: ${STUDIO2_HOST}"
echo "  • Worker: worker-studio2 (PID: ${WORKER2_PID})"
echo
echo -e "${GREEN}✓ System Status:${NC}"
echo "  • Active Workers: ${WORKER_COUNT}"
echo "  • Model: ${MODEL_NAME}"
echo
echo -e "${CYAN}Test Commands:${NC}"
echo "  • Health: curl http://localhost:8000/health"
echo "  • Test: ./test_deployment.sh"
echo "  • Status: ./check_status.sh"
echo "  • Stop: pkill -f 'python3 -m'"
echo
echo -e "${GREEN}🎉 System is running!${NC}"
echo

# Save PIDs
cat > deployment_pids.txt << EOF
Coordinator PID: ${COORDINATOR_PID}
API PID: ${API_PID}
Worker 1 PID: ${WORKER1_PID}
Worker 2 PID: ${WORKER2_PID}
EOF
