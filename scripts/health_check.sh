#!/bin/bash

# MLX System Health Check Script
# Comprehensive health monitoring for MLX distributed inference system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKER_PID_DIR="${PROJECT_ROOT}/run"
WORKER_LOG_DIR="${PROJECT_ROOT}/logs/worker"
CONFIG_DIR="${PROJECT_ROOT}/config"

# Default configuration
WORKER_ID="${WORKER_ID:-}"
CHECK_ALL="${CHECK_ALL:-false}"
VERBOSE="${VERBOSE:-false}"
OUTPUT_FORMAT="${OUTPUT_FORMAT:-text}"
TIMEOUT="${TIMEOUT:-10}"
COORDINATOR_HOST="${COORDINATOR_HOST:-coordinator.local:50051}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Health check results
declare -A health_results
health_summary=""
overall_status="HEALTHY"

# Logging functions
log_info() {
    if [[ "$OUTPUT_FORMAT" == "text" ]]; then
        echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_warn() {
    if [[ "$OUTPUT_FORMAT" == "text" ]]; then
        echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

log_error() {
    if [[ "$OUTPUT_FORMAT" == "text" ]]; then
        echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
    fi
}

log_debug() {
    if [[ "$VERBOSE" == "true" && "$OUTPUT_FORMAT" == "text" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Function to check system resources
check_system_resources() {
    log_debug "Checking system resources..."
    
    local status="HEALTHY"
    local details=""
    
    # Check CPU usage
    local cpu_usage
    if command -v top >/dev/null 2>&1; then
        cpu_usage=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//')
        if [[ -n "$cpu_usage" ]]; then
            details+="CPU: ${cpu_usage}% "
            if (( $(echo "$cpu_usage > 80" | bc -l) )); then
                status="WARNING"
            fi
        fi
    fi
    
    # Check memory usage
    local memory_info
    if command -v vm_stat >/dev/null 2>&1; then
        local page_size=$(vm_stat | grep "page size" | awk '{print $8}')
        local pages_free=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\.//')
        local pages_active=$(vm_stat | grep "Pages active" | awk '{print $3}' | sed 's/\.//')
        local pages_inactive=$(vm_stat | grep "Pages inactive" | awk '{print $3}' | sed 's/\.//')
        local pages_wired=$(vm_stat | grep "Pages wired down" | awk '{print $4}' | sed 's/\.//')
        
        if [[ -n "$page_size" && -n "$pages_free" ]]; then
            local total_pages=$((pages_free + pages_active + pages_inactive + pages_wired))
            local used_pages=$((total_pages - pages_free))
            local memory_usage=$((used_pages * 100 / total_pages))
            
            details+="Memory: ${memory_usage}% "
            if [[ $memory_usage -gt 85 ]]; then
                status="WARNING"
            fi
        fi
    fi
    
    # Check disk space
    local disk_usage
    disk_usage=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')
    if [[ -n "$disk_usage" ]]; then
        details+="Disk: ${disk_usage}% "
        if [[ $disk_usage -gt 90 ]]; then
            status="CRITICAL"
        elif [[ $disk_usage -gt 80 ]]; then
            status="WARNING"
        fi
    fi
    
    # Check load average
    local load_avg
    load_avg=$(uptime | awk -F'load averages:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    if [[ -n "$load_avg" ]]; then
        details+="Load: ${load_avg} "
    fi
    
    health_results["system_resources"]="$status|$details"
    
    if [[ "$status" != "HEALTHY" ]]; then
        overall_status="$status"
    fi
    
    log_debug "System resources check completed: $status"
}

# Function to check worker process health
check_worker_process() {
    local worker_id="$1"
    local pid_file="$WORKER_PID_DIR/worker-${worker_id}.pid"
    
    log_debug "Checking worker process: $worker_id"
    
    local status="HEALTHY"
    local details=""
    
    if [[ ! -f "$pid_file" ]]; then
        status="CRITICAL"
        details="PID file not found"
        health_results["worker_${worker_id}_process"]="$status|$details"
        overall_status="CRITICAL"
        return
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        status="CRITICAL"
        details="Process not running (PID: $pid)"
        health_results["worker_${worker_id}_process"]="$status|$details"
        overall_status="CRITICAL"
        return
    fi
    
    # Check process details
    if command -v ps >/dev/null 2>&1; then
        local process_info
        process_info=$(ps -p "$pid" -o pid,ppid,pcpu,pmem,etime,command 2>/dev/null || echo "")
        if [[ -n "$process_info" ]]; then
            local cpu_percent=$(echo "$process_info" | tail -1 | awk '{print $3}')
            local mem_percent=$(echo "$process_info" | tail -1 | awk '{print $4}')
            local elapsed_time=$(echo "$process_info" | tail -1 | awk '{print $5}')
            
            details="PID: $pid, CPU: ${cpu_percent}%, Memory: ${mem_percent}%, Uptime: $elapsed_time"
            
            # Check for high resource usage
            if (( $(echo "$cpu_percent > 90" | bc -l) )); then
                status="WARNING"
            fi
            if (( $(echo "$mem_percent > 80" | bc -l) )); then
                status="WARNING"
            fi
        fi
    fi
    
    health_results["worker_${worker_id}_process"]="$status|$details"
    
    if [[ "$status" != "HEALTHY" && "$overall_status" == "HEALTHY" ]]; then
        overall_status="$status"
    fi
    
    log_debug "Worker process check completed: $worker_id - $status"
}

# Function to check worker HTTP endpoint
check_worker_endpoint() {
    local worker_id="$1"
    local worker_port="${2:-8080}"
    
    log_debug "Checking worker HTTP endpoint: $worker_id:$worker_port"
    
    local status="HEALTHY"
    local details=""
    
    # Check health endpoint
    local health_url="http://localhost:${worker_port}/health"
    local response_code
    local response_time
    
    if command -v curl >/dev/null 2>&1; then
        local start_time=$(date +%s.%N)
        response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout "$TIMEOUT" "$health_url" 2>/dev/null || echo "000")
        local end_time=$(date +%s.%N)
        response_time=$(echo "$end_time - $start_time" | bc -l | awk '{printf "%.3f", $0}')
        
        if [[ "$response_code" == "200" ]]; then
            details="HTTP 200, Response time: ${response_time}s"
            
            # Check response time
            if (( $(echo "$response_time > 5.0" | bc -l) )); then
                status="WARNING"
                details+=" (slow response)"
            fi
        elif [[ "$response_code" == "000" ]]; then
            status="CRITICAL"
            details="Connection failed"
        else
            status="WARNING"
            details="HTTP $response_code, Response time: ${response_time}s"
        fi
    else
        # Fallback to nc if curl is not available
        if command -v nc >/dev/null 2>&1; then
            if nc -z localhost "$worker_port" 2>/dev/null; then
                details="Port $worker_port is open"
            else
                status="CRITICAL"
                details="Port $worker_port is not accessible"
            fi
        else
            status="UNKNOWN"
            details="No tools available to check endpoint"
        fi
    fi
    
    health_results["worker_${worker_id}_endpoint"]="$status|$details"
    
    if [[ "$status" == "CRITICAL" ]]; then
        overall_status="CRITICAL"
    elif [[ "$status" != "HEALTHY" && "$overall_status" == "HEALTHY" ]]; then
        overall_status="$status"
    fi
    
    log_debug "Worker endpoint check completed: $worker_id - $status"
}

# Function to check coordinator connectivity
check_coordinator_connectivity() {
    log_debug "Checking coordinator connectivity: $COORDINATOR_HOST"
    
    local status="HEALTHY"
    local details=""
    
    local coordinator_host=$(echo "$COORDINATOR_HOST" | cut -d':' -f1)
    local coordinator_port=$(echo "$COORDINATOR_HOST" | cut -d':' -f2)
    
    if command -v nc >/dev/null 2>&1; then
        if nc -z "$coordinator_host" "$coordinator_port" 2>/dev/null; then
            details="Connection successful to $COORDINATOR_HOST"
        else
            status="CRITICAL"
            details="Cannot connect to $COORDINATOR_HOST"
        fi
    else
        status="UNKNOWN"
        details="netcat not available for connectivity check"
    fi
    
    health_results["coordinator_connectivity"]="$status|$details"
    
    if [[ "$status" == "CRITICAL" ]]; then
        overall_status="CRITICAL"
    elif [[ "$status" != "HEALTHY" && "$overall_status" == "HEALTHY" ]]; then
        overall_status="$status"
    fi
    
    log_debug "Coordinator connectivity check completed: $status"
}

# Function to check log files
check_log_files() {
    local worker_id="$1"
    
    log_debug "Checking log files for worker: $worker_id"
    
    local status="HEALTHY"
    local details=""
    
    local log_file="$WORKER_LOG_DIR/worker-${worker_id}.log"
    
    if [[ -f "$log_file" ]]; then
        local log_size=$(stat -f%z "$log_file" 2>/dev/null || echo "0")
        local log_size_mb=$((log_size / 1024 / 1024))
        
        details="Log size: ${log_size_mb}MB"
        
        # Check for recent errors
        if command -v tail >/dev/null 2>&1 && command -v grep >/dev/null 2>&1; then
            local recent_errors
            recent_errors=$(tail -100 "$log_file" | grep -i "error\|exception\|critical" | wc -l | tr -d ' ')
            
            if [[ "$recent_errors" -gt 0 ]]; then
                status="WARNING"
                details+=", Recent errors: $recent_errors"
            fi
        fi
        
        # Check log file size
        if [[ $log_size_mb -gt 500 ]]; then
            status="WARNING"
            details+=", Large log file"
        fi
    else
        status="WARNING"
        details="Log file not found: $log_file"
    fi
    
    health_results["worker_${worker_id}_logs"]="$status|$details"
    
    if [[ "$status" != "HEALTHY" && "$overall_status" == "HEALTHY" ]]; then
        overall_status="$status"
    fi
    
    log_debug "Log files check completed: $worker_id - $status"
}

# Function to check configuration files
check_configuration() {
    log_debug "Checking configuration files..."
    
    local status="HEALTHY"
    local details=""
    
    # Check main config files
    local config_files=("$CONFIG_DIR/worker.yaml" "$CONFIG_DIR/api.yaml" "$CONFIG_DIR/coordinator.yaml")
    local missing_configs=()
    
    for config_file in "${config_files[@]}"; do
        if [[ ! -f "$config_file" ]]; then
            missing_configs+=("$(basename "$config_file")")
        fi
    done
    
    if [[ ${#missing_configs[@]} -gt 0 ]]; then
        status="WARNING"
        details="Missing configs: ${missing_configs[*]}"
    else
        details="All configuration files present"
    fi
    
    health_results["configuration"]="$status|$details"
    
    if [[ "$status" != "HEALTHY" && "$overall_status" == "HEALTHY" ]]; then
        overall_status="$status"
    fi
    
    log_debug "Configuration check completed: $status"
}

# Function to get worker port from config
get_worker_port() {
    local worker_id="$1"
    local config_file="$CONFIG_DIR/worker.yaml"
    
    # Default port
    echo "8080"
    
    # Try to extract port from config file if it exists
    if [[ -f "$config_file" ]] && command -v grep >/dev/null 2>&1; then
        local port
        port=$(grep "^port:" "$config_file" | awk '{print $2}' | head -1)
        if [[ -n "$port" && "$port" =~ ^[0-9]+$ ]]; then
            echo "$port"
        fi
    fi
}

# Function to perform comprehensive health check
perform_health_check() {
    local worker_id="$1"
    
    log_info "Performing health check for worker: $worker_id"
    
    # System-wide checks
    check_system_resources
    check_coordinator_connectivity
    check_configuration
    
    # Worker-specific checks
    check_worker_process "$worker_id"
    
    local worker_port
    worker_port=$(get_worker_port "$worker_id")
    check_worker_endpoint "$worker_id" "$worker_port"
    check_log_files "$worker_id"
}

# Function to check all workers
check_all_workers() {
    log_info "Performing health check for all workers..."
    
    # System-wide checks
    check_system_resources
    check_coordinator_connectivity
    check_configuration
    
    # Find all worker PID files
    local workers_found=false
    
    for pid_file in "$WORKER_PID_DIR"/worker-*.pid; do
        if [[ -f "$pid_file" ]]; then
            local worker_id=$(basename "$pid_file" .pid | sed 's/worker-//')
            workers_found=true
            
            check_worker_process "$worker_id"
            
            local worker_port
            worker_port=$(get_worker_port "$worker_id")
            check_worker_endpoint "$worker_id" "$worker_port"
            check_log_files "$worker_id"
        fi
    done
    
    if [[ "$workers_found" == "false" ]]; then
        log_warn "No running workers found"
        health_results["workers"]="WARNING|No running workers found"
        if [[ "$overall_status" == "HEALTHY" ]]; then
            overall_status="WARNING"
        fi
    fi
}

# Function to output results in text format
output_text_results() {
    echo
    echo -e "${CYAN}=== MLX System Health Check Results ===${NC}"
    echo -e "Timestamp: $(date)"
    echo -e "Overall Status: $(get_status_color "$overall_status")$overall_status${NC}"
    echo
    
    for check_name in $(printf '%s\n' "${!health_results[@]}" | sort); do
        local result="${health_results[$check_name]}"
        local status=$(echo "$result" | cut -d'|' -f1)
        local details=$(echo "$result" | cut -d'|' -f2)
        
        echo -e "$(get_status_color "$status")[$status]${NC} $check_name: $details"
    done
    
    echo
}

# Function to output results in JSON format
output_json_results() {
    echo "{"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\","
    echo "  \"overall_status\": \"$overall_status\","
    echo "  \"checks\": {"
    
    local first=true
    for check_name in $(printf '%s\n' "${!health_results[@]}" | sort); do
        local result="${health_results[$check_name]}"
        local status=$(echo "$result" | cut -d'|' -f1)
        local details=$(echo "$result" | cut -d'|' -f2)
        
        if [[ "$first" == "true" ]]; then
            first=false
        else
            echo ","
        fi
        
        echo -n "    \"$check_name\": {\"status\": \"$status\", \"details\": \"$details\"}"
    done
    
    echo
    echo "  }"
    echo "}"
}

# Function to get status color
get_status_color() {
    case "$1" in
        "HEALTHY") echo "$GREEN" ;;
        "WARNING") echo "$YELLOW" ;;
        "CRITICAL") echo "$RED" ;;
        "UNKNOWN") echo "$BLUE" ;;
        *) echo "$NC" ;;
    esac
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Perform comprehensive health checks on MLX distributed inference system.

OPTIONS:
    -h, --help              Show this help message
    -i, --worker-id ID      Check specific worker by ID
    -a, --all               Check all running workers
    -v, --verbose           Enable verbose output
    -f, --format FORMAT     Output format: text, json (default: text)
    -t, --timeout SECONDS   HTTP request timeout (default: $TIMEOUT)
    --coordinator HOST:PORT Coordinator address (default: $COORDINATOR_HOST)

EXAMPLES:
    # Check specific worker
    $0 --worker-id my-worker

    # Check all workers
    $0 --all

    # Check with verbose output
    $0 --all --verbose

    # Output in JSON format
    $0 --all --format json

    # Check with custom coordinator
    $0 --all --coordinator localhost:50051

EOF
}

# Main function
main() {
    local action=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -i|--worker-id)
                WORKER_ID="$2"
                action="check_single"
                shift 2
                ;;
            -a|--all)
                CHECK_ALL="true"
                action="check_all"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            --coordinator)
                COORDINATOR_HOST="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate output format
    if [[ "$OUTPUT_FORMAT" != "text" && "$OUTPUT_FORMAT" != "json" ]]; then
        log_error "Invalid output format: $OUTPUT_FORMAT. Use 'text' or 'json'"
        exit 1
    fi
    
    # Create necessary directories
    mkdir -p "$WORKER_PID_DIR"
    mkdir -p "$WORKER_LOG_DIR"
    
    # Execute health checks
    case "$action" in
        "check_single")
            if [[ -z "$WORKER_ID" ]]; then
                log_error "Worker ID is required for single worker check"
                exit 1
            fi
            perform_health_check "$WORKER_ID"
            ;;
        "check_all")
            check_all_workers
            ;;
        "")
            if [[ -n "${WORKER_ID:-}" ]]; then
                perform_health_check "$WORKER_ID"
            else
                check_all_workers
            fi
            ;;
        *)
            log_error "Invalid action: $action"
            exit 1
            ;;
    esac
    
    # Output results
    case "$OUTPUT_FORMAT" in
        "text")
            output_text_results
            ;;
        "json")
            output_json_results
            ;;
    esac
    
    # Exit with appropriate code
    case "$overall_status" in
        "HEALTHY") exit 0 ;;
        "WARNING") exit 1 ;;
        "CRITICAL") exit 2 ;;
        *) exit 3 ;;
    esac
}

# Run main function
main "$@"