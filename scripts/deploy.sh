#!/bin/bash

# MLX Distributed Inference Deployment Script
# Master deployment automation script for production deployment

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default configuration
DEPLOYMENT_MODE="${DEPLOYMENT_MODE:-production}"
SKIP_CHECKS="${SKIP_CHECKS:-false}"
SKIP_BACKUP="${SKIP_BACKUP:-false}"
AUTO_START="${AUTO_START:-true}"
COORDINATOR_HOST="${COORDINATOR_HOST:-coordinator.local:50051}"
WORKER_COUNT="${WORKER_COUNT:-1}"
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Deployment statistics
deployment_start_time=""
deployment_end_time=""
services_deployed=0
workers_started=0

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

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_step() {
    echo -e "${CYAN}${BOLD}[STEP]${NC} $1"
}

# Function to show deployment banner
show_banner() {
    echo -e "${CYAN}${BOLD}"
    echo "=================================================================="
    echo "    MLX Distributed Inference System Deployment"
    echo "=================================================================="
    echo -e "${NC}"
    echo "Deployment Mode: $DEPLOYMENT_MODE"
    echo "Project Root: $PROJECT_ROOT"
    echo "Coordinator: $COORDINATOR_HOST"
    echo "Worker Count: $WORKER_COUNT"
    echo "Model Path: $MODEL_PATH"
    echo "Timestamp: $(date)"
    echo
}

# Function to check system prerequisites
check_prerequisites() {
    log_step "Checking system prerequisites..."
    
    local errors=0
    
    # Check operating system
    if [[ "$(uname)" != "Darwin" ]]; then
        log_error "This deployment script is designed for macOS"
        ((errors++))
    fi
    
    # Check for required commands
    local required_commands=("python3" "docker" "docker-compose" "git")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" >/dev/null 2>&1; then
            log_error "Required command not found: $cmd"
            ((errors++))
        else
            log_info "Found: $cmd"
        fi
    done
    
    # Check Python version
    local python_version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    log_info "Python version: $python_version"
    
    # Check Docker
    if docker --version >/dev/null 2>&1; then
        log_info "Docker: $(docker --version)"
    else
        log_error "Docker is not running or not accessible"
        ((errors++))
    fi
    
    # Check Docker Compose
    if docker-compose --version >/dev/null 2>&1; then
        log_info "Docker Compose: $(docker-compose --version)"
    else
        log_error "Docker Compose not available"
        ((errors++))
    fi
    
    # Check MLX installation
    if python3 -c "import mlx.core" 2>/dev/null; then
        log_info "MLX installation verified"
    else
        log_error "MLX not installed. Please install: pip install mlx"
        ((errors++))
    fi
    
    # Check model directory
    if [[ -d "$MODEL_PATH" ]]; then
        local model_count=$(find "$MODEL_PATH" -name "*.bin" -o -name "*.safetensors" -o -name "*.gguf" | wc -l | tr -d ' ')
        log_info "Model directory found with $model_count model files"
    else
        log_warn "Model directory not found: $MODEL_PATH"
        log_warn "You may need to download models before starting workers"
    fi
    
    if [[ $errors -gt 0 ]]; then
        log_error "Prerequisites check failed with $errors errors"
        return 1
    fi
    
    log_success "Prerequisites check passed"
    return 0
}

# Function to setup project structure
setup_project_structure() {
    log_step "Setting up project structure..."
    
    # Create necessary directories
    local directories=(
        "logs/api"
        "logs/coordinator" 
        "logs/worker"
        "logs/system"
        "run"
        "data/api"
        "data/coordinator"
        "data/worker"
        "backups"
        "models"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$PROJECT_ROOT/$dir"
        log_info "Created directory: $dir"
    done
    
    # Set proper permissions
    chmod 755 "$PROJECT_ROOT/run"
    chmod 755 "$PROJECT_ROOT/logs"
    chmod 755 "$PROJECT_ROOT/data"
    
    log_success "Project structure setup completed"
}

# Function to validate configuration files
validate_configuration() {
    log_step "Validating configuration files..."
    
    local config_files=(
        "config/api.yaml"
        "config/coordinator.yaml"
        "config/worker.yaml"
    )
    
    for config_file in "${config_files[@]}"; do
        local full_path="$PROJECT_ROOT/$config_file"
        if [[ -f "$full_path" ]]; then
            log_info "Configuration file found: $config_file"
            
            # Validate YAML syntax
            if python3 -c "import yaml; yaml.safe_load(open('$full_path'))" 2>/dev/null; then
                log_info "YAML syntax valid: $config_file"
            else
                log_error "Invalid YAML syntax: $config_file"
                return 1
            fi
        else
            log_warn "Configuration file not found: $config_file"
            log_info "Default configuration will be used"
        fi
    done
    
    log_success "Configuration validation completed"
}

# Function to create backup before deployment
create_deployment_backup() {
    if [[ "$SKIP_BACKUP" == "true" ]]; then
        log_info "Skipping backup (disabled)"
        return 0
    fi
    
    log_step "Creating pre-deployment backup..."
    
    if [[ -f "$SCRIPT_DIR/backup_recovery.sh" ]]; then
        if "$SCRIPT_DIR/backup_recovery.sh" --type full --exclude-logs backup; then
            log_success "Pre-deployment backup created"
        else
            log_warn "Backup creation failed, continuing deployment"
        fi
    else
        log_warn "Backup script not found, skipping backup"
    fi
}

# Function to deploy containerized services
deploy_containerized_services() {
    log_step "Deploying containerized services..."
    
    cd "$PROJECT_ROOT"
    
    # Choose compose file based on deployment mode
    local compose_files=("-f" "docker-compose.yml")
    
    case "$DEPLOYMENT_MODE" in
        "development")
            compose_files+=("-f" "docker-compose.dev.yml")
            ;;
        "production")
            compose_files+=("-f" "docker-compose.prod.yml")
            ;;
    esac
    
    # Pull latest images
    log_info "Pulling Docker images..."
    if docker-compose "${compose_files[@]}" pull; then
        log_info "Docker images pulled successfully"
    else
        log_warn "Failed to pull some images, continuing with local images"
    fi
    
    # Start services
    log_info "Starting containerized services..."
    if docker-compose "${compose_files[@]}" up -d; then
        log_success "Containerized services started"
        ((services_deployed++))
    else
        log_error "Failed to start containerized services"
        return 1
    fi
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 10
    
    # Check service health
    local services=("api" "coordinator" "redis" "prometheus" "grafana")
    for service in "${services[@]}"; do
        if docker-compose "${compose_files[@]}" ps "$service" | grep -q "Up"; then
            log_info "Service healthy: $service"
        else
            log_warn "Service may not be healthy: $service"
        fi
    done
    
    log_success "Containerized services deployment completed"
}

# Function to deploy native workers
deploy_native_workers() {
    log_step "Deploying native MLX workers..."
    
    if [[ ! -f "$SCRIPT_DIR/start_worker_macos.sh" ]]; then
        log_error "Worker startup script not found"
        return 1
    fi
    
    # Deploy multiple workers if requested
    for ((i=1; i<=WORKER_COUNT; i++)); do
        local worker_id="worker-$(hostname)-${i}"
        
        log_info "Starting worker: $worker_id"
        
        # Set worker-specific environment
        export WORKER_ID="$worker_id"
        export COORDINATOR_HOST="$COORDINATOR_HOST"
        export MODEL_PATH="$MODEL_PATH"
        
        # Start worker
        if "$SCRIPT_DIR/start_worker_macos.sh" --worker-id "$worker_id" --coordinator "$COORDINATOR_HOST" --model-path "$MODEL_PATH"; then
            log_success "Worker started: $worker_id"
            ((workers_started++))
        else
            log_error "Failed to start worker: $worker_id"
        fi
        
        # Brief delay between worker starts
        if [[ $i -lt $WORKER_COUNT ]]; then
            sleep 5
        fi
    done
    
    log_success "Native workers deployment completed ($workers_started/$WORKER_COUNT started)"
}

# Function to verify deployment
verify_deployment() {
    log_step "Verifying deployment..."
    
    # Check containerized services
    log_info "Checking containerized services..."
    
    # API health check
    local api_health="http://localhost:8000/health"
    if curl -s "$api_health" >/dev/null 2>&1; then
        log_info "API service is responding"
    else
        log_warn "API service health check failed"
    fi
    
    # Coordinator health (if accessible via HTTP)
    # This would depend on coordinator exposing HTTP health endpoint
    
    # Check native workers
    log_info "Checking native workers..."
    
    if [[ -f "$SCRIPT_DIR/health_check.sh" ]]; then
        if "$SCRIPT_DIR/health_check.sh" --all --format text; then
            log_info "Worker health checks passed"
        else
            log_warn "Some worker health checks failed"
        fi
    fi
    
    # Check system resources
    log_info "Checking system resources..."
    
    # Memory usage
    local memory_info
    if command -v vm_stat >/dev/null 2>&1; then
        memory_info=$(vm_stat | head -5)
        log_info "Memory status checked"
    fi
    
    # Disk space
    local disk_usage
    disk_usage=$(df -h "$PROJECT_ROOT" | tail -1 | awk '{print $5}' | sed 's/%//')
    log_info "Disk usage: ${disk_usage}%"
    
    if [[ $disk_usage -gt 90 ]]; then
        log_warn "Disk usage is high: ${disk_usage}%"
    fi
    
    log_success "Deployment verification completed"
}

# Function to setup monitoring and logging
setup_monitoring() {
    log_step "Setting up monitoring and logging..."
    
    # Setup log rotation
    if [[ -f "$SCRIPT_DIR/log_rotation.sh" ]]; then
        log_info "Creating log rotation configuration..."
        "$SCRIPT_DIR/log_rotation.sh" --create-config
        log_info "Log rotation configured"
    fi
    
    # Create monitoring cron jobs
    log_info "Setting up monitoring cron jobs..."
    
    # Create cron job for health checks
    local cron_health="*/5 * * * * $SCRIPT_DIR/health_check.sh --all --format json > $PROJECT_ROOT/logs/system/health_check.log 2>&1"
    
    # Create cron job for log rotation
    local cron_logrotate="0 2 * * * $SCRIPT_DIR/log_rotation.sh > $PROJECT_ROOT/logs/system/log_rotation.log 2>&1"
    
    # Create cron job for backup
    local cron_backup="0 3 * * 0 $SCRIPT_DIR/backup_recovery.sh backup > $PROJECT_ROOT/logs/system/backup.log 2>&1"
    
    # Note: In a real deployment, these would be added to crontab
    # For now, we just create a cron configuration file
    cat > "$PROJECT_ROOT/cron_jobs.txt" << EOF
# MLX System Monitoring Cron Jobs
# Add these to your crontab with: crontab cron_jobs.txt

# Health checks every 5 minutes
$cron_health

# Log rotation daily at 2 AM
$cron_logrotate

# Weekly backup on Sunday at 3 AM
$cron_backup
EOF
    
    log_info "Cron job configuration created: cron_jobs.txt"
    log_info "Add to crontab with: crontab cron_jobs.txt"
    
    log_success "Monitoring and logging setup completed"
}

# Function to show deployment summary
show_deployment_summary() {
    deployment_end_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo
    echo -e "${CYAN}${BOLD}=================================================================="
    echo "    Deployment Summary"
    echo -e "==================================================================${NC}"
    echo
    echo "Deployment Mode: $DEPLOYMENT_MODE"
    echo "Start Time: $deployment_start_time"
    echo "End Time: $deployment_end_time"
    echo "Services Deployed: $services_deployed"
    echo "Workers Started: $workers_started"
    echo
    echo -e "${GREEN}${BOLD}Deployment Status: SUCCESS${NC}"
    echo
    echo "Next Steps:"
    echo "1. Verify all services are running: docker-compose ps"
    echo "2. Check worker health: $SCRIPT_DIR/health_check.sh --all"
    echo "3. Test API endpoint: curl http://localhost:8000/health"
    echo "4. Monitor logs: tail -f logs/*/*.log"
    echo "5. Add cron jobs: crontab cron_jobs.txt"
    echo
    echo "Management Commands:"
    echo "- Stop workers: $SCRIPT_DIR/stop_worker.sh --all"
    echo "- Health check: $SCRIPT_DIR/health_check.sh --all"
    echo "- Log rotation: $SCRIPT_DIR/log_rotation.sh"
    echo "- Backup: $SCRIPT_DIR/backup_recovery.sh backup"
    echo
}

# Function to cleanup on failure
cleanup_on_failure() {
    log_error "Deployment failed, performing cleanup..."
    
    # Stop any started workers
    if [[ -f "$SCRIPT_DIR/stop_worker.sh" ]]; then
        "$SCRIPT_DIR/stop_worker.sh" --all || true
    fi
    
    # Stop containerized services
    cd "$PROJECT_ROOT"
    docker-compose down || true
    
    log_error "Cleanup completed"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy MLX distributed inference system with containerized services and native workers.

OPTIONS:
    -h, --help              Show this help message
    -m, --mode MODE         Deployment mode: development, production (default: $DEPLOYMENT_MODE)
    -w, --workers COUNT     Number of workers to start (default: $WORKER_COUNT)
    -c, --coordinator HOST  Coordinator address (default: $COORDINATOR_HOST)
    --model-path PATH       Model directory path (default: $MODEL_PATH)
    --skip-checks           Skip prerequisite checks
    --skip-backup           Skip pre-deployment backup
    --no-auto-start         Don't automatically start services
    --cleanup               Cleanup existing deployment before starting

EXAMPLES:
    # Production deployment with 2 workers
    $0 --mode production --workers 2

    # Development deployment
    $0 --mode development --workers 1

    # Deployment with custom coordinator
    $0 --coordinator localhost:50051 --workers 3

    # Cleanup and redeploy
    $0 --cleanup --mode production

ENVIRONMENT VARIABLES:
    DEPLOYMENT_MODE         Deployment mode (development/production)
    WORKER_COUNT            Number of workers to start
    COORDINATOR_HOST        Coordinator address
    MODEL_PATH              Model directory path
    SKIP_CHECKS             Skip prerequisite checks (true/false)
    SKIP_BACKUP             Skip backup creation (true/false)
    AUTO_START              Auto-start services (true/false)

EOF
}

# Main function
main() {
    local cleanup_first=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -m|--mode)
                DEPLOYMENT_MODE="$2"
                shift 2
                ;;
            -w|--workers)
                WORKER_COUNT="$2"
                shift 2
                ;;
            -c|--coordinator)
                COORDINATOR_HOST="$2"
                shift 2
                ;;
            --model-path)
                MODEL_PATH="$2"
                shift 2
                ;;
            --skip-checks)
                SKIP_CHECKS="true"
                shift
                ;;
            --skip-backup)
                SKIP_BACKUP="true"
                shift
                ;;
            --no-auto-start)
                AUTO_START="false"
                shift
                ;;
            --cleanup)
                cleanup_first=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate deployment mode
    if [[ "$DEPLOYMENT_MODE" != "development" && "$DEPLOYMENT_MODE" != "production" ]]; then
        log_error "Invalid deployment mode: $DEPLOYMENT_MODE"
        exit 1
    fi
    
    # Validate worker count
    if ! [[ "$WORKER_COUNT" =~ ^[0-9]+$ ]] || [[ $WORKER_COUNT -lt 1 ]]; then
        log_error "Invalid worker count: $WORKER_COUNT"
        exit 1
    fi
    
    # Set up error handling
    trap cleanup_on_failure ERR
    
    deployment_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Show banner
    show_banner
    
    # Cleanup if requested
    if [[ "$cleanup_first" == "true" ]]; then
        log_step "Cleaning up existing deployment..."
        cleanup_on_failure
        sleep 5
    fi
    
    # Execute deployment steps
    if [[ "$SKIP_CHECKS" != "true" ]]; then
        check_prerequisites
    fi
    
    setup_project_structure
    validate_configuration
    create_deployment_backup
    
    if [[ "$AUTO_START" == "true" ]]; then
        deploy_containerized_services
        deploy_native_workers
        verify_deployment
        setup_monitoring
    fi
    
    # Show summary
    show_deployment_summary
    
    log_success "MLX distributed inference system deployed successfully!"
}

# Run main function
main "$@"