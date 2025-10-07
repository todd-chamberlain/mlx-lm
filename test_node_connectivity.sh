#!/bin/bash

# Simple network connectivity test between Mac Studios

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     Mac Studio Network Connectivity Test                          ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration
STUDIO2_HOST="${STUDIO2_HOST:-mac-studio-2}"
STUDIO2_IP="${STUDIO2_IP:-10.201.20.12}"
STUDIO1_IP="${STUDIO1_IP:-10.201.20.10}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Studio 1 IP: $STUDIO1_IP"
echo "  Studio 2 Host: $STUDIO2_HOST"
echo "  Studio 2 IP: $STUDIO2_IP"
echo

# Test 1: SSH Connectivity
echo -e "${YELLOW}[1/5]${NC} Testing SSH connectivity to Studio 2..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes ${STUDIO2_HOST} "echo 'SSH OK'" >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} SSH connection successful"
else
    echo -e "  ${RED}✗${NC} SSH connection failed"
    echo "  Try: ssh ${STUDIO2_HOST}"
fi
echo

# Test 2: Ping Test (Thunderbolt Network)
echo -e "${YELLOW}[2/5]${NC} Testing Thunderbolt network connectivity..."
if ping -c 3 -W 2 ${STUDIO2_IP} >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Can ping Studio 2 at ${STUDIO2_IP}"
    
    # Get ping stats
    PING_STATS=$(ping -c 10 ${STUDIO2_IP} 2>&1 | tail -1)
    echo "  ${CYAN}${PING_STATS}${NC}"
else
    echo -e "  ${RED}✗${NC} Cannot ping Studio 2 at ${STUDIO2_IP}"
    echo "  Check Thunderbolt connection"
fi
echo

# Test 3: Reverse Connectivity (Studio 2 -> Studio 1)
echo -e "${YELLOW}[3/5]${NC} Testing reverse connectivity (Studio 2 -> Studio 1)..."
if ssh ${STUDIO2_HOST} "ping -c 3 -W 2 ${STUDIO1_IP}" >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Studio 2 can reach Studio 1 at ${STUDIO1_IP}"
else
    echo -e "  ${RED}✗${NC} Studio 2 cannot reach Studio 1"
fi
echo

# Test 4: Port Connectivity (gRPC ports)
echo -e "${YELLOW}[4/5]${NC} Testing gRPC port connectivity..."

# Test port 50051 (Coordinator)
echo -n "  Port 50051 (Coordinator): "
if nc -z -w 2 localhost 50051 2>/dev/null; then
    echo -e "${GREEN}✓ Open${NC}"
else
    echo -e "${RED}✗ Closed${NC}"
fi

# Test port 50052 (Worker on Studio 1)
echo -n "  Port 50052 (Worker Studio 1): "
if nc -z -w 2 localhost 50052 2>/dev/null; then
    echo -e "${GREEN}✓ Open${NC}"
else
    echo -e "${RED}✗ Closed${NC}"
fi

# Test port 50052 on Studio 2
echo -n "  Port 50052 (Worker Studio 2): "
if nc -z -w 2 ${STUDIO2_HOST} 50052 2>/dev/null; then
    echo -e "${GREEN}✓ Open${NC}"
else
    echo -e "${RED}✗ Closed${NC}"
fi
echo

# Test 5: Network Bandwidth Test
echo -e "${YELLOW}[5/5]${NC} Testing network bandwidth (Thunderbolt)..."
echo "  Creating 100MB test file..."
dd if=/dev/zero of=/tmp/test_100mb bs=1m count=100 2>/dev/null

echo "  Transferring to Studio 2..."
START_TIME=$(date +%s)
scp -q /tmp/test_100mb ${STUDIO2_HOST}:/tmp/test_100mb_received 2>/dev/null
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

if [ $DURATION -gt 0 ]; then
    SPEED=$((100 / DURATION))
    echo -e "  ${GREEN}✓${NC} Transfer completed in ${DURATION}s (~${SPEED} MB/s)"
    
    # Cleanup
    rm -f /tmp/test_100mb
    ssh ${STUDIO2_HOST} "rm -f /tmp/test_100mb_received" 2>/dev/null
else
    echo -e "  ${RED}✗${NC} Transfer failed or too fast to measure"
fi
echo

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Summary                                                           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo

echo -e "${CYAN}Network Status:${NC}"
echo "  • SSH connectivity: Check above"
echo "  • Thunderbolt network: Check ping results"
echo "  • gRPC ports: Check port status"
echo "  • Bandwidth: Check transfer speed"
echo

echo -e "${CYAN}Next Steps:${NC}"
echo "  1. If network is OK, start services:"
echo "     ${GREEN}./deploy_both_studios.sh${NC}"
echo
echo "  2. Then run full validation:"
echo "     ${GREEN}./validate_nodes.sh${NC}"
echo
echo "  3. Or check service status:"
echo "     ${GREEN}./check_status.sh${NC}"
echo
