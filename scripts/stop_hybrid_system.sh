#!/bin/bash
# Stop the hybrid distributed system

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

STUDIO2_IP="10.201.20.12"

echo -e "${BLUE}Stopping MLX Hybrid Distributed System...${NC}"
echo ""

# Stop native workers
echo -e "${YELLOW}[1/2] Stopping native workers...${NC}"

# Stop worker-studio1
if pgrep -f "worker.main.*worker-studio1" > /dev/null; then
    pkill -f "worker.main.*worker-studio1"
    echo "  ✅ Stopped worker-studio1"
else
    echo "  ⚠️  Worker-studio1 not running"
fi

# Stop worker-studio2
echo "  Stopping worker-studio2 on ${STUDIO2_IP}..."
ssh ${STUDIO2_IP} "pkill -f 'worker.main.*worker-studio2'" 2>/dev/null && echo "  ✅ Stopped worker-studio2" || echo "  ⚠️  Worker-studio2 not running"

# Stop Docker services
echo -e "${YELLOW}[2/2] Stopping Docker services...${NC}"
docker-compose -f docker-compose.hybrid.yml down

echo ""
echo -e "${GREEN}✅ Hybrid system stopped${NC}"
echo ""
echo "To remove all data volumes, run:"
echo "  docker-compose -f docker-compose.hybrid.yml down -v"
echo ""
