#!/bin/bash

# MLX Worker Startup Script for macOS
# This script starts a native MLX worker with proper configuration and monitoring

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKER_CONFIG_DIR="${PROJECT_ROOT}/config"
WORKER_LOG_DIR="${PROJECT_ROOT}/logs/worker"
WORKER_PID_DIR="${PROJECT_ROOT}/run"
WORKER_DATA_DIR="${PROJECT_ROOT}/data/worker"

# Default configuration
WORKER_ID="${WORKER_ID:-$(hostname)-worker}"
CONFIG_FILE="${CONFIG_FILE:-${WORKER_CONFIG_DIR}/worker.yaml}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
COORDINATOR_HOST="${COORDINATOR_HOST:-coordinator.local:50051}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models}"
PYTHON_ENV="${PYTHON_ENV:-${PROJECT_ROOT}/venv}"

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

log_debug() {
    if [[ "${DEBUG:-false}" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Function to check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check if running on macOS
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This script is designed for macOS only"
        exit 1
    fi
    
    # Check for Apple Silicon
    if [[ "$(uname -m)" != "arm64" ]]; then
        log_warn "This worker is optimized for Apple Silicon (M1/M2/M3)"
    fi
    
    # Check Python environment
    if [[ -d "$PYTHON_ENV" ]]; then
        source "$PYTHON_ENV/bin/activate"
        log_info "Activated Python virtual environment: $PYTHON_ENV"
    else
        log_warn "Python virtual environment not found at $PYTHON_ENV"
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check MLX installation
    if ! python3 -c "import mlx.core" 2>/dev/null; then
        log_error "MLX not installed. Please install MLX: pip install mlx"
        exit 1
    fi
    
    log_info "System requirements check passed"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    mkdir -p "$WORKER_LOG_DIR"
    mkdir -p "$WORKER_PID_DIR"
    mkdir -p "$WORKER_DATA_DIR"
    mkdir -p "$(dirname "$CONFIG_FILE")"
    
    log_info "Directories created successfully"
}

# Function to validate configuration
validate_configuration() {
    log_info "Validating worker configuration..."
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_warn "Configuration file not found: $CONFIG_FILE"
        log_info "Creating default configuration..."
        create_default_config
    fi
    
    # Validate model path
    if [[ ! -d "$MODEL_PATH" ]]; then
        log_error "Model path does not exist: $MODEL_PATH"
        exit 1
    fi
    
    # Test coordinator connectivity
    if command -v nc >/dev/null 2>&1; then
        coordinator_host=$(echo "$COORDINATOR_HOST" | cut -d':' -f1)
        coordinator_port=$(echo "$COORDINATOR_HOST" | cut -d':' -f2)
        
        if ! nc -z "$coordinator_host" "$coordinator_port" 2>/dev/null; then
            log_warn "Cannot connect to coordinator at $COORDINATOR_HOST"
            log_warn "Worker will attempt to connect on startup"
        else
            log_info "Coordinator connectivity verified"
        fi
    fi
    
    log_info "Configuration validation completed"
}

# Function to create default configuration
create_default_config() {
    cat > "$CONFIG_FILE" << EOF
# MLX Worker Configuration
worker_id: "$WORKER_ID"
port: 8080

model:
  path: "$MODEL_PATH"
  name: "default-model"
  cache_size: 2
  validation_enabled: true

device:
  memory_limit_gb: null
  gpu_memory_fraction: 0.8
  auto_detect: true

performance:
  max_concurrent: 4
  request_timeout: 300
  cleanup_interval: 60
  metrics_history_size: 100

coordinator:
  host: "$COORDINATOR_HOST"
  auto_discover: true
  heartbeat_interval: 10
  connection_timeout: 30
  max_retries: 3
  retry_delay: 1.0

health:
  check_interval: 30.0
  report_interval: 60
  metrics_enabled: true

logging:
  level: "$LOG_LEVEL"
  file: "$WORKER_LOG_DIR/worker.log"
  max_size: "100MB"
  backup_count: 5

monitoring:
  metrics_port: 9090
  jaeger_endpoint: "http://localhost:14268/api/traces"
EOF
    
    log_info "Default configuration created at $CONFIG_FILE"
}

# Function to check if worker is already running
check_existing_worker() {
    local pid_file="$WORKER_PID_DIR/worker-${WORKER_ID}.pid"
    
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            log_error "Worker $WORKER_ID is already running (PID: $pid)"
            exit 1
        else
            log_warn "Stale PID file found, removing..."
            rm -f "$pid_file"
        fi
    fi
}

# Function to setup signal handlers
setup_signal_handlers() {
    local pid_file="$WORKER_PID_DIR/worker-${WORKER_ID}.pid"
    
    cleanup() {
        log_info "Received shutdown signal, stopping worker..."
        if [[ -f "$pid_file" ]]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid"
                
                # Wait for graceful shutdown
                local count=0
                while kill -0 "$pid" 2>/dev/null && [[ $count -lt 30 ]]; do
                    sleep 1
                    ((count++))
                done
                
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    log_warn "Worker did not shutdown gracefully, force killing..."
                    kill -KILL "$pid"
                fi
            fi
            rm -f "$pid_file"
        fi
        exit 0
    }
    
    trap cleanup SIGTERM SIGINT
}

# Function to start the worker
start_worker() {
    log_info "Starting MLX worker: $WORKER_ID"
    
    local pid_file="$WORKER_PID_DIR/worker-${WORKER_ID}.pid"
    local log_file="$WORKER_LOG_DIR/worker-${WORKER_ID}.log"
    
    # Export environment variables
    export WORKER_ID="$WORKER_ID"
    export CONFIG_FILE="$CONFIG_FILE"
    export LOG_LEVEL="$LOG_LEVEL"
    
    # Start worker process
    cd "$PROJECT_ROOT"
    
    if [[ "${FOREGROUND:-false}" == "true" ]]; then
        # Run in foreground
        log_info "Starting worker in foreground mode..."
        python3 -m worker.main --config "$CONFIG_FILE" 2>&1 | tee "$log_file"
    else
        # Run in background
        log_info "Starting worker in background mode..."
        nohup python3 -m worker.main --config "$CONFIG_FILE" > "$log_file" 2>&1 &
        local worker_pid=$!
        
        # Save PID
        echo "$worker_pid" > "$pid_file"
        
        # Wait a moment to check if process started successfully
        sleep 2
        if ! kill -0 "$worker_pid" 2>/dev/null; then
            log_error "Worker failed to start. Check logs: $log_file"
            rm -f "$pid_file"
            exit 1
        fi
        
        log_info "Worker started successfully (PID: $worker_pid)"
        log_info "Log file: $log_file"
        log_info "PID file: $pid_file"
    fi
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Start MLX worker on macOS with proper configuration and monitoring.

OPTIONS:
    -h, --help              Show this help message
    -c, --config FILE       Configuration file path (default: $CONFIG_FILE)
    -i, --worker-id ID      Worker ID (default: $WORKER_ID)
    -m, --model-path PATH   Model directory path (default: $MODEL_PATH)
    -l, --log-level LEVEL   Log level (default: $LOG_LEVEL)
    -f, --foreground        Run in foreground mode
    -d, --debug             Enable debug logging
    --coordinator HOST:PORT Coordinator address (default: $COORDINATOR_HOST)

ENVIRONMENT VARIABLES:
    WORKER_ID               Worker identifier
    CONFIG_FILE             Configuration file path
    LOG_LEVEL               Logging level
    COORDINATOR_HOST        Coordinator address
    MODEL_PATH              Model directory path
    PYTHON_ENV              Python virtual environment path
    DEBUG                   Enable debug mode (true/false)
    FOREGROUND              Run in foreground (true/false)

EXAMPLES:
    # Start worker with default configuration
    $0

    # Start worker with custom configuration
    $0 --config /path/to/worker.yaml --worker-id my-worker

    # Start worker in foreground with debug logging
    $0 --foreground --debug --log-level DEBUG

    # Start worker with custom model path
    $0 --model-path /path/to/models --coordinator localhost:50051

EOF
}

# Main function
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -i|--worker-id)
                WORKER_ID="$2"
                shift 2
                ;;
            -m|--model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            -l|--log-level)
                LOG_LEVEL="$2"
                shift 2
                ;;
            -f|--foreground)
                FOREGROUND="true"
                shift
                ;;
            -d|--debug)
                DEBUG="true"
                shift
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
    
    log_info "Starting MLX Worker deployment script"
    log_info "Worker ID: $WORKER_ID"
    log_info "Configuration: $CONFIG_FILE"
    log_info "Model path: $MODEL_PATH"
    log_info "Coordinator: $COORDINATOR_HOST"
    
    # Execute startup sequence
    check_system_requirements
    create_directories
    validate_configuration
    check_existing_worker
    setup_signal_handlers
    start_worker
    
    if [[ "${FOREGROUND:-false}" != "true" ]]; then
        log_info "Worker startup completed successfully"
        log_info "Use 'scripts/stop_worker.sh --worker-id $WORKER_ID' to stop the worker"
        log_info "Use 'scripts/health_check.sh --worker-id $WORKER_ID' to check worker health"
    fi
}

# Run main function
main "$@"