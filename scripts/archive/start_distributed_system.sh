#!/bin/bash
# Orchestration script to start the entire distributed MLX inference system
# Run this on mac-studio-1

set -e

# Configuration
STUDIO1_IP="10.201.20.10"
STUDIO2_IP="10.201.20.12"
COORDINATOR_PORT="50051"
API_PORT="8000"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "🚀 Starting MLX Distributed Inference System"
echo "=============================================="
echo ""

# Step 1: Start Coordinator on studio-1
echo -e "${GREEN}[1/5] Starting Coordinator on mac-studio-1...${NC}"
if pgrep -f "coordinator.main" > /dev/null; then
    echo "  ⚠️  Coordinator already running, restarting..."
    pkill -f "coordinator.main"
    sleep 2
fi

nohup ./venv/bin/python -m coordinator.main > logs/coordinator.log 2>&1 &
COORDINATOR_PID=$!
echo "  ✅ Coordinator started (PID: $COORDINATOR_PID)"
sleep 3

# Verify coordinator is running
if ! pgrep -f "coordinator.main" > /dev/null; then
    echo -e "${RED}  ❌ Coordinator failed to start!${NC}"
    echo "  Check logs: tail -f logs/coordinator.log"
    exit 1
fi

# Step 2: Start API Server on studio-1
echo -e "${GREEN}[2/5] Starting API Server on mac-studio-1...${NC}"
if pgrep -f "api.main" > /dev/null; then
    echo "  ⚠️  API Server already running, restarting..."
    pkill -f "api.main"
    sleep 2
fi

nohup ./venv/bin/python -m api.main > logs/api.log 2>&1 &
API_PID=$!
echo "  ✅ API Server started (PID: $API_PID)"
sleep 3

# Verify API is running
if ! curl -s "http://localhost:${API_PORT}/health" > /dev/null; then
    echo -e "${YELLOW}  ⚠️  API health check failed (may still be starting)${NC}"
fi

# Step 3: Start Worker on studio-1
echo -e "${GREEN}[3/5] Starting Worker on mac-studio-1...${NC}"
if pgrep -f "worker.main.*worker-studio1" > /dev/null; then
    echo "  ⚠️  Worker already running, restarting..."
    pkill -f "worker.main.*worker-studio1"
    sleep 2
fi

nohup ./venv/bin/python -m worker.main \
    --worker-id worker-studio1 \
    --port 8080 \
    --coordinator localhost:${COORDINATOR_PORT} \
    > logs/worker-studio1.log 2>&1 &
WORKER1_PID=$!
echo "  ✅ Worker-studio1 started (PID: $WORKER1_PID)"
sleep 5

# Step 4: Start Worker on studio-2 via SSH
echo -e "${GREEN}[4/5] Starting Worker on mac-studio-2...${NC}"
echo "  Checking SSH connectivity to ${STUDIO2_IP}..."

if ssh -o ConnectTimeout=5 ${STUDIO2_IP} "echo 'SSH OK'" > /dev/null 2>&1; then
    echo "  ✅ SSH connection successful"

    # Kill existing worker on studio-2
    echo "  Stopping existing worker on studio-2..."
    ssh ${STUDIO2_IP} "pkill -f 'worker.main.*worker-studio2' || true" 2>/dev/null

    # Start worker on studio-2
    echo "  Starting worker-studio2..."
    ssh ${STUDIO2_IP} "cd ~/mlx-distributed-inference && \
        source venv/bin/activate && \
        nohup python -m worker.main \
            --worker-id worker-studio2 \
            --port 8080 \
            --coordinator ${STUDIO1_IP}:${COORDINATOR_PORT} \
            > logs/worker-studio2.log 2>&1 &"

    echo "  ✅ Worker-studio2 started on ${STUDIO2_IP}"
else
    echo -e "${YELLOW}  ⚠️  Cannot connect to ${STUDIO2_IP} via SSH${NC}"
    echo "  You'll need to manually start worker on studio-2:"
    echo ""
    echo "    ssh ${STUDIO2_IP}"
    echo "    cd ~/mlx-distributed-inference"
    echo "    source venv/bin/activate"
    echo "    python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator ${STUDIO1_IP}:${COORDINATOR_PORT}"
    echo ""
fi

# Step 5: Verify system health
echo -e "${GREEN}[5/5] Verifying system health...${NC}"
sleep 5

HEALTH_CHECK=$(curl -s "http://localhost:${API_PORT}/health" || echo '{"status":"error"}')
STATUS=$(echo $HEALTH_CHECK | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
WORKERS=$(echo $HEALTH_CHECK | grep -o '"total_workers":[0-9]*' | cut -d':' -f2)

echo ""
echo "System Status:"
echo "=============="
echo "  Status: $STATUS"
echo "  Total Workers: ${WORKERS:-0}"
echo ""

if [ "$STATUS" = "healthy" ]; then
    echo -e "${GREEN}✅ System is healthy and ready!${NC}"
else
    echo -e "${YELLOW}⚠️  System status: $STATUS${NC}"
fi

echo ""
echo "Useful Commands:"
echo "================"
echo "  Health check:        curl http://localhost:${API_PORT}/health | jq ."
echo "  List models:         curl http://localhost:${API_PORT}/models | jq ."
echo "  View coordinator:    tail -f logs/coordinator.log"
echo "  View API:            tail -f logs/api.log"
echo "  View worker 1:       tail -f logs/worker-studio1.log"
echo "  View worker 2:       ssh ${STUDIO2_IP} 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'"
echo ""
echo "To load Qwen3-Coder-480B with sharding:"
echo "  python scripts/download_qwen3_with_sharding.py"
echo ""
