#!/bin/bash
# Hybrid Deployment Orchestration
# Starts Docker services (coordinator, API, monitoring) + Native MLX workers

set -e

# Configuration
STUDIO1_IP="10.201.20.10"
STUDIO2_IP="10.201.20.12"
COORDINATOR_HOST="localhost"  # Docker coordinator accessible on localhost
COORDINATOR_PORT="50051"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "🚀 Starting MLX Hybrid Distributed System"
echo "=========================================="
echo ""
echo "Architecture:"
echo "  🐳 Docker Services: Coordinator, API, Monitoring"
echo "  ⚡ Native Workers: MLX on Apple Silicon"
echo ""

# Step 1: Start Docker services
echo -e "${BLUE}[1/4] Starting Docker services...${NC}"
echo "  Starting: Redis, Coordinator, API, Prometheus, Grafana"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}  ❌ Docker is not running!${NC}"
    echo "  Please start Docker Desktop and try again."
    exit 1
fi

# Start Docker Compose services
docker-compose -f docker-compose.hybrid.yml up -d

echo "  ✅ Docker services started"
echo "  Waiting for services to be healthy..."
sleep 10

# Wait for coordinator to be ready
MAX_WAIT=30
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if docker exec mlx-coordinator python -c "import grpc; channel = grpc.insecure_channel('localhost:50051'); grpc.channel_ready_future(channel).result(timeout=5); channel.close()" 2>/dev/null; then
        echo -e "  ${GREEN}✅ Coordinator is ready${NC}"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}  ❌ Coordinator failed to start${NC}"
    echo "  Check logs: docker logs mlx-coordinator"
    exit 1
fi

# Wait for API to be ready
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf "http://localhost:8000/health" > /dev/null; then
        echo -e "  ${GREEN}✅ API is ready${NC}"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}  ❌ API failed to start${NC}"
    echo "  Check logs: docker logs mlx-api"
    exit 1
fi

# Step 2: Start native worker on studio-1
echo -e "${BLUE}[2/4] Starting native worker on mac-studio-1...${NC}"

if pgrep -f "worker.main.*worker-studio1" > /dev/null; then
    echo "  ⚠️  Worker already running, restarting..."
    pkill -f "worker.main.*worker-studio1"
    sleep 2
fi

nohup ./venv/bin/python -m worker.main \
    --worker-id worker-studio1 \
    --port 8080 \
    --coordinator ${COORDINATOR_HOST}:${COORDINATOR_PORT} \
    > logs/worker-studio1.log 2>&1 &

WORKER1_PID=$!
echo "  ✅ Worker-studio1 started (PID: $WORKER1_PID)"
sleep 5

# Step 3: Start native worker on studio-2
echo -e "${BLUE}[3/4] Starting native worker on mac-studio-2...${NC}"
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

# Step 4: Verify system health
echo -e "${BLUE}[4/4] Verifying system health...${NC}"
sleep 5

HEALTH_CHECK=$(curl -s "http://localhost:8000/health" || echo '{"status":"error"}')
STATUS=$(echo $HEALTH_CHECK | grep -o '"status":"[^"]*"' | cut -d'"' -f4)
WORKERS=$(echo $HEALTH_CHECK | grep -o '"total_workers":[0-9]*' | cut -d':' -f2)

echo ""
echo "System Status:"
echo "=============="
echo "  Status: $STATUS"
echo "  Total Workers: ${WORKERS:-0}"
echo ""

if [ "$STATUS" = "healthy" ]; then
    echo -e "${GREEN}✅ Hybrid system is healthy and ready!${NC}"
else
    echo -e "${YELLOW}⚠️  System status: $STATUS${NC}"
fi

echo ""
echo "Service URLs:"
echo "============="
echo "  API:          http://localhost:8000"
echo "  Coordinator:  localhost:50051 (gRPC)"
echo "  Prometheus:   http://localhost:9090"
echo "  Grafana:      http://localhost:3000 (admin/admin)"
echo ""
echo "Useful Commands:"
echo "================"
echo "  Health check:         curl http://localhost:8000/health | jq ."
echo "  List models:          curl http://localhost:8000/models | jq ."
echo "  Docker logs:          docker logs mlx-coordinator"
echo "  Docker logs:          docker logs mlx-api"
echo "  Worker 1 logs:        tail -f logs/worker-studio1.log"
echo "  Worker 2 logs:        ssh ${STUDIO2_IP} 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'"
echo "  Stop system:          ./scripts/stop_hybrid_system.sh"
echo ""
echo "To load Qwen3-Coder-480B with sharding:"
echo "  python scripts/download_qwen3_with_sharding.py"
echo ""
