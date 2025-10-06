#!/bin/bash

# Check status of both Mac Studios

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

STUDIO2_HOST="mac-studio-2"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  MLX Distributed Inference - System Status${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo

# Mac Studio 1
echo -e "${YELLOW}Mac Studio 1:${NC}"

# Check Docker containers
echo -n "  Containerized Services: "
if docker-compose ps 2>/dev/null | grep -q "Up"; then
    RUNNING=$(docker-compose ps 2>/dev/null | grep "Up" | wc -l | tr -d ' ')
    echo -e "${GREEN}✓ ${RUNNING} services running${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check local worker
echo -n "  Worker: "
if pgrep -f "python3 -m worker.main" >/dev/null; then
    PID=$(pgrep -f "python3 -m worker.main")
    echo -e "${GREEN}✓ Running (PID: ${PID})${NC}"
else
    echo -e "${RED}✗ Not running${NC}"
fi

# Check API
echo -n "  API Health: "
if curl -s http://localhost:8000/health >/dev/null 2>&1; then
    HEALTH=$(curl -s http://localhost:8000/health)
    WORKERS=$(echo "$HEALTH" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workers', 0))" 2>/dev/null || echo "?")
    echo -e "${GREEN}✓ Responding (${WORKERS} workers)${NC}"
else
    echo -e "${RED}✗ Not responding${NC}"
fi

echo

# Mac Studio 2
echo -e "${YELLOW}Mac Studio 2:${NC}"

# Check SSH
echo -n "  SSH Connection: "
if ssh -o ConnectTimeout=5 -o BatchMode=yes ${STUDIO2_HOST} "echo 'OK'" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ Connected${NC}"
    
    # Check worker on Studio 2
    echo -n "  Worker: "
    if ssh ${STUDIO2_HOST} "pgrep -f 'python3 -m worker.main'" >/dev/null 2>&1; then
        PID=$(ssh ${STUDIO2_HOST} "pgrep -f 'python3 -m worker.main'")
        echo -e "${GREEN}✓ Running (PID: ${PID})${NC}"
    else
        echo -e "${RED}✗ Not running${NC}"
    fi
else
    echo -e "${RED}✗ Cannot connect${NC}"
fi

echo
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"

# Quick test
echo
echo "Quick API Test:"
curl -s http://localhost:8000/health 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "API not responding"
