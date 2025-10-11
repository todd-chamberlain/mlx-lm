#!/bin/bash

# Monitor memory usage across both Mac Studios during distributed model loading
# Usage: ./scripts/monitor_distributed_loading.sh

set -e

# Configuration
STUDIO1_IP="localhost"  # Current machine
STUDIO2_IP="10.201.20.12"
LOG_DIR="logs"
MONITOR_INTERVAL=2  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Distributed Model Loading Monitor ===${NC}"
echo "Monitoring memory usage across Mac Studios during model loading"
echo "Studio 1 (local): $STUDIO1_IP"
echo "Studio 2 (remote): $STUDIO2_IP"
echo ""

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

# Function to get memory usage
get_memory_usage() {
    local host=$1
    local label=$2
    
    if [ "$host" = "localhost" ]; then
        # Local memory usage
        local mem_info=$(vm_stat | grep -E "(Pages free|Pages active|Pages inactive|Pages speculative|Pages wired down)")
        local page_size=$(vm_stat | head -1 | grep -o '[0-9]*')
        
        # Calculate memory in GB
        local free_pages=$(echo "$mem_info" | grep "Pages free" | awk '{print $3}' | tr -d '.')
        local active_pages=$(echo "$mem_info" | grep "Pages active" | awk '{print $3}' | tr -d '.')
        local inactive_pages=$(echo "$mem_info" | grep "Pages inactive" | awk '{print $3}' | tr -d '.')
        local wired_pages=$(echo "$mem_info" | grep "Pages wired down" | awk '{print $4}' | tr -d '.')
        
        local total_used=$((active_pages + inactive_pages + wired_pages))
        local used_gb=$(echo "scale=2; $total_used * $page_size / 1024 / 1024 / 1024" | bc -l)
        local free_gb=$(echo "scale=2; $free_pages * $page_size / 1024 / 1024 / 1024" | bc -l)
        
        echo -e "${label}: Used: ${used_gb}GB, Free: ${free_gb}GB"
    else
        # Remote memory usage via SSH
        local remote_cmd="vm_stat | grep -E '(Pages free|Pages active|Pages inactive|Pages speculative|Pages wired down)'"
        local mem_info=$(ssh "$host" "$remote_cmd" 2>/dev/null || echo "SSH_ERROR")
        
        if [ "$mem_info" = "SSH_ERROR" ]; then
            echo -e "${label}: ${RED}SSH connection failed${NC}"
        else
            local page_size=$(ssh "$host" "vm_stat | head -1 | grep -o '[0-9]*'" 2>/dev/null || echo "4096")
            
            local free_pages=$(echo "$mem_info" | grep "Pages free" | awk '{print $3}' | tr -d '.' || echo "0")
            local active_pages=$(echo "$mem_info" | grep "Pages active" | awk '{print $3}' | tr -d '.' || echo "0")
            local inactive_pages=$(echo "$mem_info" | grep "Pages inactive" | awk '{print $3}' | tr -d '.' || echo "0")
            local wired_pages=$(echo "$mem_info" | grep "Pages wired down" | awk '{print $4}' | tr -d '.' || echo "0")
            
            local total_used=$((active_pages + inactive_pages + wired_pages))
            local used_gb=$(echo "scale=2; $total_used * $page_size / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0.00")
            local free_gb=$(echo "scale=2; $free_pages * $page_size / 1024 / 1024 / 1024" | bc -l 2>/dev/null || echo "0.00")
            
            echo -e "${label}: Used: ${used_gb}GB, Free: ${free_gb}GB"
        fi
    fi
}

# Function to check worker processes
check_worker_processes() {
    echo -e "\n${YELLOW}=== Worker Process Status ===${NC}"
    
    # Local worker
    local local_worker=$(ps aux | grep "python -m worker.main" | grep -v grep | wc -l)
    echo -e "Studio 1 Worker: $([ $local_worker -gt 0 ] && echo -e "${GREEN}Running${NC}" || echo -e "${RED}Not Running${NC}")"
    
    # Remote worker
    local remote_worker=$(ssh "$STUDIO2_IP" "ps aux | grep 'python -m worker.main' | grep -v grep | wc -l" 2>/dev/null || echo "0")
    echo -e "Studio 2 Worker: $([ $remote_worker -gt 0 ] && echo -e "${GREEN}Running${NC}" || echo -e "${RED}Not Running${NC}")"
}

# Function to monitor logs
monitor_logs() {
    echo -e "\n${YELLOW}=== Recent Log Activity ===${NC}"
    
    # Local logs
    if [ -f "$LOG_DIR/worker-studio1.log" ]; then
        echo -e "${BLUE}Studio 1 (last 3 lines):${NC}"
        tail -3 "$LOG_DIR/worker-studio1.log" | sed 's/^/  /'
    fi
    
    # Remote logs
    echo -e "${BLUE}Studio 2 (last 3 lines):${NC}"
    ssh "$STUDIO2_IP" "tail -3 ~/mlx-distributed-inference/logs/worker-studio2.log 2>/dev/null || echo '  Log file not found'" | sed 's/^/  /'
}

# Main monitoring loop
echo -e "${GREEN}Starting continuous monitoring (Ctrl+C to stop)...${NC}"
echo ""

while true; do
    clear
    echo -e "${BLUE}=== Distributed Model Loading Monitor - $(date) ===${NC}"
    echo ""
    
    # Memory usage
    echo -e "${YELLOW}=== Memory Usage ===${NC}"
    get_memory_usage "localhost" "${GREEN}Studio 1 (Local)${NC}"
    get_memory_usage "$STUDIO2_IP" "${GREEN}Studio 2 (Remote)${NC}"
    
    # Worker processes
    check_worker_processes
    
    # Log monitoring
    monitor_logs
    
    echo ""
    echo -e "${BLUE}Refreshing in $MONITOR_INTERVAL seconds... (Ctrl+C to stop)${NC}"
    sleep $MONITOR_INTERVAL
done