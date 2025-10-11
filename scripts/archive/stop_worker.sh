#!/bin/bash

# MLX Worker Stop Script
# This script stops a running MLX worker gracefully

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKER_PID_DIR="${PROJECT_ROOT}/run"
WORKER_LOG_DIR="${PROJECT_ROOT}/logs/worker"

# Default configuration
WORKER_ID="${WORKER_ID:-$(hostname)-worker}"
FORCE_KILL="${FORCE_KILL:-false}"
TIMEOUT="${TIMEOUT:-30}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1" >&2
}

# Function to stop a specific worker
stop_worker() {
    local worker_id="$1"
    local pid_file="$WORKER_PID_DIR/worker-${worker_id}.pid"
    
    if [[ ! -f "$pid_file" ]]; then
        log_error "PID file not found for worker: $worker_id"
        log_error "Worker may not be running or was started manually"
        return 1
    fi
    
    local pid=$(cat "$pid_file")
    
    if ! kill -0 "$pid" 2>/dev/null; then
        log_warn "Worker process not found (PID: $pid), cleaning up PID file"
        rm -f "$pid_file"
        return 0
    fi
    
    log_info "Stopping worker: $worker_id (PID: $pid)"
    
    if [[ "$FORCE_KILL" == "true" ]]; then
        log_warn "Force killing worker process..."
        kill -KILL "$pid"
        rm -f "$pid_file"
        log_info "Worker force killed"
        return 0
    fi
    
    # Send SIGTERM for graceful shutdown
    log_info "Sending graceful shutdown signal..."
    kill -TERM "$pid"
    
    # Wait for graceful shutdown
    local count=0
    while kill -0 "$pid" 2>/dev/null && [[ $count -lt $TIMEOUT ]]; do
        sleep 1
        ((count++))
        if [[ $((count % 5)) -eq 0 ]]; then
            log_info "Waiting for graceful shutdown... ($count/${TIMEOUT}s)"
        fi
    done
    
    # Check if process is still running
    if kill -0 "$pid" 2>/dev/null; then
        log_warn "Worker did not shutdown gracefully within ${TIMEOUT}s"
        log_warn "Force killing worker process..."
        kill -KILL "$pid"
        
        # Wait a moment for force kill to take effect
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Failed to kill worker process"
            return 1
        fi
    fi
    
    # Clean up PID file
    rm -f "$pid_file"
    log_info "Worker stopped successfully: $worker_id"
    
    return 0
}

# Function to stop all workers
stop_all_workers() {
    log_info "Stopping all workers..."
    
    local stopped_count=0
    local failed_count=0
    
    for pid_file in "$WORKER_PID_DIR"/worker-*.pid; do
        if [[ -f "$pid_file" ]]; then
            local worker_id=$(basename "$pid_file" .pid | sed 's/worker-//')
            if stop_worker "$worker_id"; then
                ((stopped_count++))
            else
                ((failed_count++))
            fi
        fi
    done
    
    if [[ $stopped_count -eq 0 && $failed_count -eq 0 ]]; then
        log_info "No running workers found"
    else
        log_info "Stopped $stopped_count workers"
        if [[ $failed_count -gt 0 ]]; then
            log_warn "Failed to stop $failed_count workers"
        fi
    fi
}

# Function to list running workers
list_workers() {
    log_info "Listing running workers..."
    
    local found_workers=false
    
    for pid_file in "$WORKER_PID_DIR"/worker-*.pid; do
        if [[ -f "$pid_file" ]]; then
            local worker_id=$(basename "$pid_file" .pid | sed 's/worker-//')
            local pid=$(cat "$pid_file")
            
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Worker ID: $worker_id (PID: $pid)"
                found_workers=true
            else
                echo "  Worker ID: $worker_id (PID: $pid) - STALE"
            fi
        fi
    done
    
    if [[ "$found_workers" == "false" ]]; then
        echo "  No running workers found"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Stop MLX worker processes gracefully or forcefully.

OPTIONS:
    -h, --help              Show this help message
    -i, --worker-id ID      Stop specific worker by ID
    -a, --all               Stop all running workers
    -f, --force             Force kill workers (SIGKILL)
    -t, --timeout SECONDS   Graceful shutdown timeout (default: $TIMEOUT)
    -l, --list              List running workers
    --cleanup               Clean up stale PID files

EXAMPLES:
    # Stop specific worker
    $0 --worker-id my-worker

    # Stop all workers
    $0 --all

    # Force kill specific worker
    $0 --worker-id my-worker --force

    # Stop with custom timeout
    $0 --worker-id my-worker --timeout 60

    # List running workers
    $0 --list

    # Clean up stale PID files
    $0 --cleanup

EOF
}

# Function to cleanup stale PID files
cleanup_stale_pids() {
    log_info "Cleaning up stale PID files..."
    
    local cleaned_count=0
    
    for pid_file in "$WORKER_PID_DIR"/worker-*.pid; do
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if ! kill -0 "$pid" 2>/dev/null; then
                local worker_id=$(basename "$pid_file" .pid | sed 's/worker-//')
                log_info "Removing stale PID file for worker: $worker_id"
                rm -f "$pid_file"
                ((cleaned_count++))
            fi
        fi
    done
    
    if [[ $cleaned_count -eq 0 ]]; then
        log_info "No stale PID files found"
    else
        log_info "Cleaned up $cleaned_count stale PID files"
    fi
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
                action="stop_single"
                shift 2
                ;;
            -a|--all)
                action="stop_all"
                shift
                ;;
            -f|--force)
                FORCE_KILL="true"
                shift
                ;;
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -l|--list)
                action="list"
                shift
                ;;
            --cleanup)
                action="cleanup"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Create PID directory if it doesn't exist
    mkdir -p "$WORKER_PID_DIR"
    
    # Execute action
    case "$action" in
        "stop_single")
            stop_worker "$WORKER_ID"
            ;;
        "stop_all")
            stop_all_workers
            ;;
        "list")
            list_workers
            ;;
        "cleanup")
            cleanup_stale_pids
            ;;
        "")
            if [[ -n "${WORKER_ID:-}" ]]; then
                stop_worker "$WORKER_ID"
            else
                log_error "No action specified. Use --help for usage information."
                exit 1
            fi
            ;;
        *)
            log_error "Invalid action: $action"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"