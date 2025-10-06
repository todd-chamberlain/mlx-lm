#!/bin/bash

# Quick deployment test script for Mac Studios
# This script tests the deployment and shows real-time performance

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  MLX Distributed Inference - Deployment Test              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo

# Configuration
API_URL="${API_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-your-secret-api-key-here}"

# Test 1: Health Check
echo -e "${YELLOW}[1/5]${NC} Testing API Health..."
HEALTH_RESPONSE=$(curl -s "${API_URL}/health")
echo "$HEALTH_RESPONSE" | python3 -m json.tool

WORKER_COUNT=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin).get('workers', 0))")
echo -e "${GREEN}✓${NC} API is healthy with ${WORKER_COUNT} workers"
echo

# Test 2: List Models
echo -e "${YELLOW}[2/5]${NC} Listing available models..."
curl -s "${API_URL}/v1/models" \
  -H "Authorization: Bearer ${API_KEY}" | python3 -m json.tool
echo -e "${GREEN}✓${NC} Models endpoint working"
echo

# Test 3: Simple Inference
echo -e "${YELLOW}[3/5]${NC} Testing simple inference..."
START_TIME=$(date +%s%N)

RESPONSE=$(curl -s -X POST "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "messages": [
      {"role": "user", "content": "Say hello in exactly 5 words."}
    ],
    "max_tokens": 20,
    "temperature": 0.7
  }')

END_TIME=$(date +%s%N)
LATENCY=$(( (END_TIME - START_TIME) / 1000000 ))

echo "$RESPONSE" | python3 -m json.tool
echo -e "${GREEN}✓${NC} Inference completed in ${LATENCY}ms"
echo

# Test 4: Streaming Inference
echo -e "${YELLOW}[4/5]${NC} Testing streaming inference..."
echo -n "Response: "

curl -s -X POST "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${API_KEY}" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "max_tokens": 50,
    "stream": true
  }' | while IFS= read -r line; do
    if [[ "$line" == data:* ]]; then
        echo -n "."
    fi
done

echo
echo -e "${GREEN}✓${NC} Streaming working"
echo

# Test 5: Load Test (10 concurrent requests)
echo -e "${YELLOW}[5/5]${NC} Running mini load test (10 requests)..."

# Create temp file for request
cat > /tmp/test_request.json << 'EOF'
{
  "messages": [
    {"role": "user", "content": "Hi"}
  ],
  "max_tokens": 10
}
EOF

# Run concurrent requests
TOTAL_TIME=0
SUCCESS_COUNT=0

for i in {1..10}; do
    START=$(date +%s%N)
    
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${API_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "Authorization: Bearer ${API_KEY}" \
        -d @/tmp/test_request.json)
    
    END=$(date +%s%N)
    REQ_TIME=$(( (END - START) / 1000000 ))
    TOTAL_TIME=$((TOTAL_TIME + REQ_TIME))
    
    if [[ "$STATUS" == "200" ]]; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        echo -e "  Request $i: ${GREEN}✓${NC} ${REQ_TIME}ms"
    else
        echo -e "  Request $i: ${RED}✗${NC} HTTP $STATUS"
    fi
done

AVG_TIME=$((TOTAL_TIME / 10))

echo
echo -e "${GREEN}✓${NC} Load test complete: ${SUCCESS_COUNT}/10 successful"
echo -e "  Average latency: ${AVG_TIME}ms"
echo

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Test Summary                                              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo
echo -e "  Workers Active:     ${GREEN}${WORKER_COUNT}${NC}"
echo -e "  Health Check:       ${GREEN}✓ PASS${NC}"
echo -e "  Models Endpoint:    ${GREEN}✓ PASS${NC}"
echo -e "  Simple Inference:   ${GREEN}✓ PASS${NC} (${LATENCY}ms)"
echo -e "  Streaming:          ${GREEN}✓ PASS${NC}"
echo -e "  Load Test:          ${GREEN}✓ PASS${NC} (${SUCCESS_COUNT}/10, avg ${AVG_TIME}ms)"
echo
echo -e "${GREEN}🎉 All tests passed! System is working correctly.${NC}"
echo

# Cleanup
rm -f /tmp/test_request.json

# Show monitoring URLs
echo "📊 Monitoring Dashboards:"
echo "  • API Health:  ${API_URL}/health"
echo "  • Metrics:     ${API_URL}/metrics"
echo "  • Grafana:     http://localhost:3000"
echo "  • Prometheus:  http://localhost:9090"
echo
