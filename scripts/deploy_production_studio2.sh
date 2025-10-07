#!/bin/bash

# Production Deployment Script for Mac Studio 2
# This script deploys and validates the MLX worker on Mac Studio 2

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKER_ID="worker-studio2"
STUDIO1_HOST="${STUDIO1_HOST:-mac-studio-1}"
STUDIO1_IP="${STUDIO1_IP:-10.201.20.10}"
COORDINATOR_HOST="${COORDINATOR_HOST:-${STUDIO1_IP}:50051}"
MODEL_PATH="${MODEL_PATH:-/Users/Shared/models}"
MODEL_NAME="${MODEL_NAME:-Llama-3.2-3B-Instruct-4bit}"
WORKER_PORT="${WORKER_PORT:-50052}"
LOG_DIR="${PROJECT_ROOT}/logs"
VALIDATION_LOG="${LOG_DIR}/deployment_validation_studio2.log"

# Deployment results
declare -A validation_results
overall_status="SUCCESS"

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$VALIDATION_LOG"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$VALIDATION_LOG"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$VALIDATION_LOG"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$VALIDATION_LOG"
}

# Print header
print_header() {
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  MLX Distributed Inference - Mac Studio 2 Production Deployment   ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo "Timestamp: $(date)"
    echo "Worker ID: $WORKER_ID"
    echo "Model: $MODEL_NAME"
    echo "Coordinator: $COORDINATOR_HOST"
    echo
}

# Check network connectivity to Studio 1
check_network_connectivity() {
    log_info "Checking network connectivity to Mac Studio 1..."
    
    local status="PASS"
    local details=""
    
    # Ping Studio 1
    if ping -c 3 -W 2 "$STUDIO1_IP" >/dev/null 2>&1; then
        local latency=$(ping -c 3 "$STUDIO1_IP" 2>&1 | grep 'avg' | awk -F'/' '{print $5}')
        details="Ping successful, latency: ${latency}ms. "
        log_info "Network connectivity to Studio 1: OK (${latency}ms)"
    else
        log_error "Cannot ping Mac Studio 1 at $STUDIO1_IP"
        status="FAIL"
        details="Cannot ping $STUDIO1_IP. "
        overall_status="FAIL"
    fi
    
    # Check coordinator port
    if command -v nc >/dev/null 2>&1; then
        if nc -z -w 5 "$STUDIO1_IP" 50051 2>/dev/null; then
            details+="Coordinator port accessible. "
            log_info "Coordinator port 50051 is accessible"
        else
            log_error "Cannot connect to coordinator at ${STUDIO1_IP}:50051"
            status="FAIL"
            details+="Coordinator port not accessible. "
            overall_status="FAIL"
        fi
    fi
    
    validation_results["network_connectivity"]="$status|$details"
    
    if [[ "$status" == "FAIL" ]]; then
        return 1
    fi
    
    return 0
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local status="PASS"
    local details=""
    
    # Check macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This script requires macOS"
        status="FAIL"
        details="Not running on macOS"
        validation_results["prerequisites"]="$status|$details"
        return 1
    fi
    
    # Check Apple Silicon
    if [[ "$(uname -m)" == "arm64" ]]; then
        details+="Apple Silicon detected. "
    else
        log_warn "Not running on Apple Silicon - performance may be limited"
        details+="Not Apple Silicon. "
    fi
    
    # Check Python
    if command -v python3 >/dev/null 2>&1; then
        local python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
        details+="Python $python_version. "
        log_info "Python version: $python_version"
    else
        log_error "Python 3 not found"
        status="FAIL"
        details+="Python 3 missing. "
    fi
    
    # Check MLX
    if python3 -c "import mlx.core" 2>/dev/null; then
        local mlx_version=$(python3 -c "import mlx; print(mlx.__version__)" 2>/dev/null || echo "unknown")
        details+="MLX $mlx_version. "
        log_info "MLX version: $mlx_version"
    else
        log_error "MLX not installed"
        status="FAIL"
        details+="MLX missing. "
    fi
    
    # Check virtual environment
    if [[ -d "$PROJECT_ROOT/venv" ]]; then
        details+="venv exists. "
        log_info "Virtual environment found"
    else
        log_warn "Virtual environment not found at $PROJECT_ROOT/venv"
        details+="venv missing. "
    fi
    
    # Check model directory
    if [[ -d "$MODEL_PATH" ]]; then
        details+="Model path exists. "
        log_info "Model directory found: $MODEL_PATH"
    else
        log_error "Model directory not found: $MODEL_PATH"
        status="FAIL"
        details+="Model path missing. "
    fi
    
    # Check specific model
    if [[ -d "$MODEL_PATH/$MODEL_NAME" ]]; then
        local model_size=$(du -sh "$MODEL_PATH/$MODEL_NAME" 2>/dev/null | cut -f1)
        details+="Model $MODEL_NAME ($model_size). "
        log_info "Model found: $MODEL_NAME ($model_size)"
    else
        log_error "Model not found: $MODEL_PATH/$MODEL_NAME"
        status="FAIL"
        details+="Model $MODEL_NAME missing. "
    fi
    
    validation_results["prerequisites"]="$status|$details"
    
    if [[ "$status" == "FAIL" ]]; then
        overall_status="FAIL"
        return 1
    fi
    
    log_success "Prerequisites check passed"
    return 0
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    local status="PASS"
    local details=""
    
    # Get system info
    local model=$(sysctl -n hw.model 2>/dev/null || echo "unknown")
    local cpu_count=$(sysctl -n hw.ncpu 2>/dev/null || echo "unknown")
    local memory_bytes=$(sysctl -n hw.memsize 2>/dev/null || echo "0")
    local memory_gb=$((memory_bytes / 1024 / 1024 / 1024))
    
    details+="Model: $model, CPUs: $cpu_count, Memory: ${memory_gb}GB. "
    
    log_info "System: $model"
    log_info "CPUs: $cpu_count"
    log_info "Memory: ${memory_gb}GB"
    
    # Check available disk space
    local disk_avail=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $4}')
    details+="Disk available: $disk_avail. "
    log_info "Disk available: $disk_avail"
    
    # Check memory pressure
    if command -v vm_stat >/dev/null 2>&1; then
        local page_size=$(vm_stat | grep "page size" | awk '{print $8}')
        local pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        
        if [[ -n "$page_size" && -n "$pages_free" ]]; then
            local free_mb=$((pages_free * page_size / 1024 / 1024))
            details+="Free memory: ${free_mb}MB. "
            log_info "Free memory: ${free_mb}MB"
            
            if [[ $free_mb -lt 4096 ]]; then
                log_warn "Low free memory: ${free_mb}MB"
                status="WARN"
            fi
        fi
    fi
    
    validation_results["system_resources"]="$status|$details"
    
    if [[ "$status" == "FAIL" ]]; then
        overall_status="FAIL"
    fi
    
    log_success "System resources check completed"
    return 0
}

# Start worker
start_worker() {
    log_info "Starting worker: $WORKER_ID..."
    
    local status="PASS"
    local details=""
    
    # Activate virtual environment if it exists
    if [[ -d "$PROJECT_ROOT/venv" ]]; then
        source "$PROJECT_ROOT/venv/bin/activate"
        log_info "Activated virtual environment"
    fi
    
    # Check if worker is already running
    if pgrep -f "worker.main.*$WORKER_ID" >/dev/null; then
        log_warn "Worker $WORKER_ID is already running"
        local pid=$(pgrep -f "worker.main.*$WORKER_ID")
        details="Already running (PID: $pid)"
        validation_results["worker_start"]="SKIP|$details"
        return 0
    fi
    
    # Start worker using the startup script
    log_info "Executing worker startup script..."
    
    "$SCRIPT_DIR/start_worker_macos.sh" \
        --worker-id "$WORKER_ID" \
        --coordinator "$COORDINATOR_HOST" \
        --model-path "$MODEL_PATH/$MODEL_NAME" \
        --log-level INFO \
        2>&1 | tee -a "$VALIDATION_LOG" &
    
    local start_pid=$!
    
    # Wait for worker to start
    log_info "Waiting for worker to start..."
    sleep 10
    
    # Check if worker process is running
    if pgrep -f "worker.main.*$WORKER_ID" >/dev/null; then
        local worker_pid=$(pgrep -f "worker.main.*$WORKER_ID")
        details="Started successfully (PID: $worker_pid)"
        log_success "Worker started with PID: $worker_pid"
    else
        log_error "Worker failed to start"
        status="FAIL"
        details="Failed to start"
        overall_status="FAIL"
    fi
    
    validation_results["worker_start"]="$status|$details"
    
    if [[ "$status" == "FAIL" ]]; then
        return 1
    fi
    
    return 0
}

# Verify worker registration
verify_worker_registration() {
    log_info "Verifying worker registration with coordinator..."
    
    local status="PASS"
    local details=""
    local max_attempts=30
    local attempt=0
    
    # Wait for worker to register
    while [[ $attempt -lt $max_attempts ]]; do
        sleep 2
        attempt=$((attempt + 1))
        
        # Check worker logs for registration success
        if grep -q "Successfully registered with coordinator" "$LOG_DIR/worker-${WORKER_ID}.log" 2>/dev/null; then
            details="Registered successfully after ${attempt} attempts"
            log_success "Worker registered with coordinator"
            validation_results["worker_registration"]="$status|$details"
            return 0
        fi
        
        # Check for registration errors
        if grep -q "Failed to register" "$LOG_DIR/worker-${WORKER_ID}.log" 2>/dev/null; then
            log_error "Worker registration failed"
            status="FAIL"
            details="Registration failed"
            overall_status="FAIL"
            validation_results["worker_registration"]="$status|$details"
            return 1
        fi
        
        log_info "Waiting for registration... (attempt $attempt/$max_attempts)"
    done
    
    log_warn "Worker registration could not be verified within timeout"
    status="WARN"
    details="Registration timeout after ${max_attempts} attempts"
    validation_results["worker_registration"]="$status|$details"
    
    return 0
}

# Test model loading
test_model_loading() {
    log_info "Testing model loading..."
    
    local status="PASS"
    local details=""
    
    # Create test script
    local test_script=$(mktemp)
    cat > "$test_script" << 'EOF'
import sys
import time
from mlx_lm import load

try:
    model_path = sys.argv[1]
    print(f"Loading model from: {model_path}", file=sys.stderr)
    
    start_time = time.time()
    model, tokenizer = load(model_path)
    load_time = time.time() - start_time
    
    print(f"SUCCESS|Model loaded in {load_time:.2f}s")
    sys.exit(0)
except Exception as e:
    print(f"FAIL|{str(e)}")
    sys.exit(1)
EOF
    
    # Run test
    log_info "Loading model: $MODEL_PATH/$MODEL_NAME"
    local result
    result=$(python3 "$test_script" "$MODEL_PATH/$MODEL_NAME" 2>&1 || echo "FAIL|Test script failed")
    
    rm -f "$test_script"
    
    status=$(echo "$result" | cut -d'|' -f1)
    details=$(echo "$result" | cut -d'|' -f2-)
    
    if [[ "$status" == "SUCCESS" ]]; then
        log_success "Model loading test passed: $details"
    else
        log_error "Model loading test failed: $details"
        overall_status="FAIL"
    fi
    
    validation_results["model_loading"]="$status|$details"
    
    if [[ "$status" == "FAIL" ]]; then
        return 1
    fi
    
    return 0
}

# Test local inference
test_local_inference() {
    log_info "Testing local inference..."
    
    local status="PASS"
    local details=""
    
    # Create test script
    local test_script=$(mktemp)
    cat > "$test_script" << 'EOF'
import sys
import time
from mlx_lm import load, generate

try:
    model_path = sys.argv[1]
    
    # Load model
    model, tokenizer = load(model_path)
    
    # Test inference
    prompt = "Hello, world!"
    start_time = time.time()
    
    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=10,
        verbose=False
    )
    
    inference_time = time.time() - start_time
    
    if response and len(response) > 0:
        print(f"SUCCESS|Inference completed in {inference_time:.2f}s, generated {len(response)} chars")
    else:
        print(f"FAIL|No response generated")
    
    sys.exit(0)
except Exception as e:
    print(f"FAIL|{str(e)}")
    sys.exit(1)
EOF
    
    # Run test
    log_info "Running inference test..."
    local result
    result=$(python3 "$test_script" "$MODEL_PATH/$MODEL_NAME" 2>&1 || echo "FAIL|Test script failed")
    
    rm -f "$test_script"
    
    status=$(echo "$result" | cut -d'|' -f1)
    details=$(echo "$result" | cut -d'|' -f2-)
    
    if [[ "$status" == "SUCCESS" ]]; then
        log_success "Local inference test passed: $details"
    else
        log_error "Local inference test failed: $details"
        overall_status="FAIL"
    fi
    
    validation_results["local_inference"]="$status|$details"
    
    if [[ "$status" == "FAIL" ]]; then
        return 1
    fi
    
    return 0
}

# Check memory usage
check_memory_usage() {
    log_info "Checking memory usage..."
    
    local status="PASS"
    local details=""
    
    # Get worker PID
    local worker_pid
    if pgrep -f "worker.main.*$WORKER_ID" >/dev/null; then
        worker_pid=$(pgrep -f "worker.main.*$WORKER_ID")
    else
        log_error "Worker process not found"
        status="FAIL"
        details="Worker process not running"
        validation_results["memory_usage"]="$status|$details"
        return 1
    fi
    
    # Get memory usage
    local mem_info
    mem_info=$(ps -p "$worker_pid" -o rss,vsz,pmem 2>/dev/null || echo "0 0 0")
    
    local rss_kb=$(echo "$mem_info" | tail -1 | awk '{print $1}')
    local vsz_kb=$(echo "$mem_info" | tail -1 | awk '{print $2}')
    local mem_percent=$(echo "$mem_info" | tail -1 | awk '{print $3}')
    
    local rss_mb=$((rss_kb / 1024))
    local vsz_mb=$((vsz_kb / 1024))
    
    details="RSS: ${rss_mb}MB, VSZ: ${vsz_mb}MB, %MEM: ${mem_percent}%"
    
    log_info "Worker memory usage: $details"
    
    # Check for excessive memory usage
    if [[ $rss_mb -gt 16384 ]]; then
        log_warn "High memory usage: ${rss_mb}MB"
        status="WARN"
    fi
    
    validation_results["memory_usage"]="$status|$details"
    
    return 0
}

# Check performance metrics
check_performance_metrics() {
    log_info "Checking performance metrics..."
    
    local status="PASS"
    local details=""
    
    # Get worker PID
    local worker_pid
    if pgrep -f "worker.main.*$WORKER_ID" >/dev/null; then
        worker_pid=$(pgrep -f "worker.main.*$WORKER_ID")
    else
        log_error "Worker process not found"
        status="FAIL"
        details="Worker process not running"
        validation_results["performance_metrics"]="$status|$details"
        return 1
    fi
    
    # Get CPU usage
    local cpu_percent
    cpu_percent=$(ps -p "$worker_pid" -o %cpu 2>/dev/null | tail -1 | tr -d ' ')
    
    # Get process uptime
    local etime
    etime=$(ps -p "$worker_pid" -o etime 2>/dev/null | tail -1 | tr -d ' ')
    
    details="CPU: ${cpu_percent}%, Uptime: $etime"
    
    log_info "Worker performance: $details"
    
    # Check worker metrics endpoint if available
    if command -v curl >/dev/null 2>&1; then
        local metrics_url="http://localhost:${WORKER_PORT}/metrics"
        if curl -s --connect-timeout 5 "$metrics_url" >/dev/null 2>&1; then
            details+=", Metrics endpoint: available"
            log_info "Metrics endpoint is accessible"
        else
            details+=", Metrics endpoint: unavailable"
            log_warn "Metrics endpoint not accessible"
        fi
    fi
    
    validation_results["performance_metrics"]="$status|$details"
    
    return 0
}

# Generate deployment report
generate_deployment_report() {
    local report_file="$LOG_DIR/deployment_report_studio2_$(date +%Y%m%d_%H%M%S).txt"
    
    {
        echo "╔════════════════════════════════════════════════════════════════════╗"
        echo "║  Mac Studio 2 Production Deployment Report                        ║"
        echo "╚════════════════════════════════════════════════════════════════════╝"
        echo
        echo "Timestamp: $(date)"
        echo "Worker ID: $WORKER_ID"
        echo "Model: $MODEL_NAME"
        echo "Coordinator: $COORDINATOR_HOST"
        echo "Overall Status: $overall_status"
        echo
        echo "Validation Results:"
        echo "==================="
        
        for check_name in $(printf '%s\n' "${!validation_results[@]}" | sort); do
            local result="${validation_results[$check_name]}"
            local status=$(echo "$result" | cut -d'|' -f1)
            local details=$(echo "$result" | cut -d'|' -f2-)
            
            printf "%-30s: [%-4s] %s\n" "$check_name" "$status" "$details"
        done
        
        echo
        echo "Log Files:"
        echo "=========="
        echo "Validation log: $VALIDATION_LOG"
        echo "Worker log: $LOG_DIR/worker-${WORKER_ID}.log"
        echo
        
        if [[ "$overall_status" == "SUCCESS" ]]; then
            echo "✓ Deployment completed successfully"
        else
            echo "✗ Deployment completed with issues"
        fi
        
    } | tee "$report_file"
    
    log_info "Deployment report saved to: $report_file"
}

# Print summary
print_summary() {
    echo
    echo -e "${CYAN}${BOLD}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║  Deployment Summary                                                ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    
    for check_name in $(printf '%s\n' "${!validation_results[@]}" | sort); do
        local result="${validation_results[$check_name]}"
        local status=$(echo "$result" | cut -d'|' -f1)
        local details=$(echo "$result" | cut -d'|' -f2-)
        
        local status_color
        case "$status" in
            "PASS"|"SUCCESS") status_color="$GREEN" ;;
            "WARN") status_color="$YELLOW" ;;
            "FAIL") status_color="$RED" ;;
            "SKIP") status_color="$BLUE" ;;
            *) status_color="$NC" ;;
        esac
        
        printf "${status_color}[%-7s]${NC} %-25s: %s\n" "$status" "$check_name" "$details"
    done
    
    echo
    if [[ "$overall_status" == "SUCCESS" ]]; then
        echo -e "${GREEN}${BOLD}✓ Mac Studio 2 deployment completed successfully!${NC}"
    else
        echo -e "${RED}${BOLD}✗ Mac Studio 2 deployment completed with issues${NC}"
    fi
    echo
}

# Main function
main() {
    # Create log directory
    mkdir -p "$LOG_DIR"
    
    # Clear validation log
    > "$VALIDATION_LOG"
    
    # Print header
    print_header
    
    # Run deployment steps
    check_network_connectivity || exit 1
    check_prerequisites || exit 1
    check_system_resources
    start_worker || exit 1
    verify_worker_registration
    test_model_loading || log_warn "Model loading test failed, but continuing..."
    test_local_inference || log_warn "Local inference test failed, but continuing..."
    check_memory_usage
    check_performance_metrics
    
    # Generate report
    generate_deployment_report
    
    # Print summary
    print_summary
    
    # Exit with appropriate code
    if [[ "$overall_status" == "SUCCESS" ]]; then
        exit 0
    else
        exit 1
    fi
}

# Run main function
main "$@"
