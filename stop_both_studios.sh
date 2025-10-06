#!/bin/bash

# Stop all services on both Mac Studios

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

STUDIO2_HOST="mac-studio-2"

echo -e "${YELLOW}Stopping MLX Distributed Inference System...${NC}"
echo

# Stop Studio 1
echo "Stopping services on Mac Studio 1..."
echo "  • Stopping worker..."
pkill -f "python3 -m worker.main" 2>/dev/null || echo "    No worker running"

echo "  • Stopping containerized services..."
docker-compose down 2>/dev/null || echo "    No containers running"

echo -e "${GREEN}✓${NC} Mac Studio 1 stopped"
echo

# Stop Studio 2
echo "Stopping services on Mac Studio 2..."
if ssh -o ConnectTimeout=5 ${STUDIO2_HOST} "pkill -f 'python3 -m worker.main' 2>/dev/null"; then
    echo -e "${GREEN}✓${NC} Mac Studio 2 worker stopped"
else
    echo "  No worker running on Mac Studio 2"
fi
echo

echo -e "${GREEN}✓ All services stopped${NC}"
