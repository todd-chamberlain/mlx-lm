#!/bin/bash

# Automated deployment script for both Mac Studios
# Uses passwordless SSH to deploy to mac-studio-2

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
MODEL_PATH="/Users/Shared/models/${MODEL_NAME}"

echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  MLX Distributed Inference - Dual Studio Deployment        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Get Studio 1 IP
echo -e "${YELLOW}[1/8]${NC} Getting Mac Studio 1 IP address..."
STUDIO1_IP=$(ipconfig getifaddr en0)
if [ -z "$STUDIO1_IP" ]; then
    echo -e "${RED}✗ Could not determine IP address${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Mac Studio 1 IP: ${STUDIO1_IP}"
echo

# Check SSH connectivity to Studio 2
echo -e "${YELLOW}[2/8]${NC} Testing SSH connection to Mac Studio 2..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes ${STUDIO2_HOST} "echo 'SSH OK'" >/dev/null 2>&1; then
    echo -e "${GREEN}✓${NC} SSH connection to ${STUDIO2_HOST} successful"
else
    echo -e "${RED}✗ Cannot connect to ${STUDIO2_HOST} via SSH${NC}"
    echo "  Make sure passwordless SSH is configured"
    exit 1
fi
echo

# Check if model exists locally
echo -e "${YELLOW}[3/8]${NC} Checking for model on Mac Studio 1..."
if [ -d "$MODEL_PATH" ]; then
    echo -e "${GREEN}✓${NC} Model found: ${MODEL_PATH}"
else
    echo -e "${YELLOW}⚠${NC} Model not found locally"
    echo "  Downloading ${MODEL_NAME}..."
    mkdir -p /Users/Shared/models
    cd /Users/Shared/models
    huggingface-cli download mlx-community/${MODEL_NAME} --local-dir ${MODEL_NAME}
    cd "$PROJECT_DIR"
    echo -e "${GREEN}✓${NC} Model downloaded"
fi
echo

# Sync code to Studio 2
echo -e "${YELLOW}[4/8]${NC} Syncing code to Mac Studio 2..."
ssh ${STUDIO2_HOST} "mkdir -p ~/mlx-distributed-inference"
rsync -az --exclude 'venv' --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    "${PROJECT_DIR}/" ${STUDIO2_HOST}:~/mlx-distributed-inference/
echo -e "${GREEN}✓${NC} Code synced to Mac Studio 2"
echo

# Check/sync model to Studio 2
echo -e "${YELLOW}[5/8]${NC} Checking model on Mac Studio 2..."
if ssh ${STUDIO2_HOST} "[ -d '${MODEL_PATH}' ]"; then
    echo -e "${GREEN}✓${NC} Model already exists on Mac Studio 2"
else
    echo -e "${YELLOW}⚠${NC} Model not found on Mac Studio 2"
    echo "  Syncing model (this may take a while)..."
    ssh ${STUDIO2_HOST} "mkdir -p /Users/Shared/models"
    rsync -az --progress "${MODEL_PATH}/" ${STUDIO2_HOST}:${MODEL_PATH}/
    echo -e "${GREEN}✓${NC} Model synced to Mac Studio 2"
fi
echo

# Deploy on Studio 1 (containerized services + worker)
echo -e "${YELLOW}[6/8]${NC} Deploying on Mac Studio 1..."
echo "  Starting containerized services..."

# Stop any existing services
docker-compose down 2>/dev/null || true
pkill -f "python3 -m worker.main" 2>/dev/null || true
sleep 2

# Start containerized services
docker-compose up -d

# Wait for services to be ready
echo "  Waiting for services to start..."
sleep 15

# Check services
if docker-compose ps | grep -q "Up"; then
    echo -e "${GREEN}✓${NC} Containerized services running"
else
    echo -e "${RED}✗ Failed to start containerized services${NC}"
    docker-compose logs
    exit 1
fi

# Start local worker
echo "  Starting worker on Mac Studio 1..."
python3 -m worker.main \
    --worker-id worker-studio1 \
    --coordinator localhost:50051 \
    --model-path "${MODEL_PATH}" \
    --port 50052 > logs/worker-studio1.log 2>&1 &

WORKER1_PID=$!
sleep 5

if ps -p $WORKER1_PID > /dev/null; then
    echo -e "${GREEN}✓${NC} Worker started on Mac Studio 1 (PID: ${WORKER1_PID})"
else
    echo -e "${RED}✗ Failed to start worker on Mac Studio 1${NC}"
    cat logs/worker-studio1.log
    exit 1
fi
echo

# Deploy on Studio 2 (worker only)
echo -e "${YELLOW}[7/8]${NC} Deploying on Mac Studio 2..."

# Stop any existing worker
ssh ${STUDIO2_HOST} "pkill -f 'python3 -m worker.main' 2>/dev/null || true"
sleep 2

# Start worker on Studio 2
echo "  Starting worker on Mac Studio 2..."
ssh ${STUDIO2_HOST} "cd ~/mlx-distributed-inference && \
    nohup python3 -m worker.main \
        --worker-id worker-studio2 \
        --coordinator ${STUDIO1_IP}:50051 \
        --model-path '${MODEL_PATH}' \
        --port 50052 \
    > logs/worker-studio2.log 2>&1 &"

sleep 5

# Check if worker is running on Studio 2
if ssh ${STUDIO2_HOST} "pgrep -f 'python3 -m worker.main'" >/dev/null; then
    WORKER2_PID=$(ssh ${STUDIO2_HOST} "pgrep -f 'python3 -m worker.main'")
    echo -e "${GREEN}✓${NC} Worker started on Mac Studio 2 (PID: ${WORKER2_PID})"
else
    echo -e "${RED}✗ Failed to start worker on Mac Studio 2${NC}"
    ssh ${STUDIO2_HOST} "cat ~/mlx-distributed-inference/logs/worker-studio2.log"
    exit 1
fi
echo

# Verify deployment
echo -e "${YELLOW}[8/8]${NC} Verifying deployment..."
sleep 5

# Check API health
echo "  Checking API health..."
HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
WORKER_COUNT=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workers', 0))" 2>/dev/null || echo "0")

if [ "$WORKER_COUNT" -eq 2 ]; then
    echo -e "${GREEN}✓${NC} API reports 2 workers active"
else
    echo -e "${YELLOW}⚠${NC} API reports ${WORKER_COUNT} workers (expected 2)"
    echo "  Workers may still be registering, checking again in 5 seconds..."
    sleep 5
    HEALTH_RESPONSE=$(curl -s http://localhost:8000/health)
    WORKER_COUNT=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workers', 0))" 2>/dev/null || echo "0")
    if [ "$WORKER_COUNT" -eq 2 ]; then
        echo -e "${GREEN}✓${NC} API now reports 2 workers active"
    else
        echo -e "${YELLOW}⚠${NC} Still only ${WORKER_COUNT} workers registered"
    fi
fi
echo

# Test inference
echo "  Testing inference..."
TEST_RESPONSE=$(curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer your-secret-api-key-here" \
    -d '{
        "messages": [{"role": "user", "content": "Say hello"}],
        "max_tokens": 10
    }')

if echo "$TEST_RESPONSE" | grep -q "choices"; then
    echo -e "${GREEN}✓${NC} Inference test successful"
else
    echo -e "${YELLOW}⚠${NC} Inference test returned unexpected response"
fi
echo

# Deployment summary
echo -e "${CYAN}${BOLD}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Deployment Complete!                                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo
echo -e "${GREEN}✓ Mac Studio 1:${NC}"
echo "  • IP: ${STUDIO1_IP}"
echo "  • API Server: http://localhost:8000"
echo "  • Coordinator: localhost:50051"
echo "  • Worker: worker-studio1 (PID: ${WORKER1_PID})"
echo "  • Grafana: http://localhost:3000"
echo "  • Prometheus: http://localhost:9090"
echo
echo -e "${GREEN}✓ Mac Studio 2:${NC}"
echo "  • Host: ${STUDIO2_HOST}"
echo "  • Worker: worker-studio2 (PID: ${WORKER2_PID})"
echo "  • Connected to: ${STUDIO1_IP}:50051"
echo
echo -e "${GREEN}✓ System Status:${NC}"
echo "  • Active Workers: ${WORKER_COUNT}"
echo "  • Model: ${MODEL_NAME}"
echo "  • Status: Ready for inference"
echo
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Run tests:        ./test_deployment.sh"
echo "  2. View health:      curl http://localhost:8000/health"
echo "  3. Open Grafana:     open http://localhost:3000"
echo "  4. View logs:        tail -f logs/*.log"
echo "  5. Stop workers:     ./stop_both_studios.sh"
echo
echo -e "${GREEN}🎉 Deployment successful!${NC}"
echo

# Save deployment info
cat > deployment_info.txt << EOF
Deployment Information
=====================
Date: $(date)
Studio 1 IP: ${STUDIO1_IP}
Studio 2 Host: ${STUDIO2_HOST}
Worker 1 PID: ${WORKER1_PID}
Worker 2 PID: ${WORKER2_PID}
Model: ${MODEL_NAME}
Active Workers: ${WORKER_COUNT}

Management Commands:
- Health check: curl http://localhost:8000/health
- View logs: tail -f logs/*.log
- Stop all: ./stop_both_studios.sh
- Restart: ./deploy_both_studios.sh
EOF

echo "Deployment info saved to: deployment_info.txt"
