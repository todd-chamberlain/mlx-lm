#!/bin/bash

# Quick validation script for distributed layer inference

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     MLX Distributed Layer Inference - Quick Validation            ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration
NODE1="${NODE1:-10.201.20.10:50052}"
NODE2="${NODE2:-10.201.20.12:50052}"
COORDINATOR="${COORDINATOR:-10.201.20.10:50051}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Node 1: $NODE1"
echo "  Node 2: $NODE2"
echo "  Coordinator: $COORDINATOR"
echo

# Check if Python script exists
if [ ! -f "validate_distributed_layers.py" ]; then
    echo -e "${RED}Error: validate_distributed_layers.py not found${NC}"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${BLUE}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Run validation
echo -e "${BLUE}Running validation tests...${NC}"
echo

python3 validate_distributed_layers.py \
    --node1 "$NODE1" \
    --node2 "$NODE2" \
    --coordinator "$COORDINATOR"

EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Validation completed successfully${NC}"
else
    echo -e "${RED}✗ Validation failed${NC}"
fi

exit $EXIT_CODE
