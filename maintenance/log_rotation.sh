#!/bin/bash

# MLX Log Rotation and Cleanup Script
# Manages log files for all MLX services with configurable retention policies

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LOG_BASE_DIR="${PROJECT_ROOT}/logs"

# Default configuration
MAX_LOG_SIZE="${MAX_LOG_SIZE:-100M}"
MAX_LOG_FILES="${MAX_LOG_FILES:-10}"
MAX_AGE_DAYS="${MAX_AGE_DAYS:-30}"
COMPRESS_LOGS="${COMPRESS_LOGS:-true}"
DRY_RUN="${DRY_RUN:-false}"
VERBOSE="${VERBOSE:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Statistics
total_files_processed=0
total_files_rotated=0
total_files_compressed=0
total_files_deleted=0
total_space_freed=0

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
    if [[ "$VERBOSE" == "true" ]]; then
        echo -e "${BLUE}[DEBUG]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
    fi
}

# Function to convert size string to bytes
size_to_bytes() {
    local size_str="$1"
    local size_num=$(echo "$size_str" | sed 's/[^0-9.]//g')
    local size_unit=$(echo "$size_str" | sed 's/[0-9.]//g' | tr '[:lower:]' '[:upper:]')
    
    case "$size_unit" in
        "B"|"") echo "$size_num" | cut -d'.' -f1 ;;
        "K"|"KB") echo "$size_num * 1024" | bc | cut -d'.' -f1 ;;
        "M"|"MB") echo "$size_num * 1024 * 1024" | bc | cut -d'.' -f1 ;;
        "G"|"GB") echo "$size_num * 1024 * 1024 * 1024" | bc | cut -d'.' -f1 ;;
        *) echo "0" ;;
    esac
}

# Function to format bytes to human readable
bytes_to_human() {
    local bytes="$1"
    
    if [[ $bytes -lt 1024 ]]; then
        echo "${bytes}B"
    elif [[ $bytes -lt $((1024 * 1024)) ]]; then
        echo "$(echo "scale=1; $bytes / 1024" | bc)K"
    elif [[ $bytes -lt $((1024 * 1024 * 1024)) ]]; then
        echo "$(echo "scale=1; $bytes / 1024 / 1024" | bc)M"
    else
        echo "$(echo "scale=1; $bytes / 1024 / 1024 / 1024" | bc)G"
    fi
}

# Function to get file size in bytes
get_file_size() {
    local file="$1"
    if [[ -f "$file" ]]; then
        stat -f%z "$file" 2>/dev/null || echo "0"
    else
        echo "0"
    fi
}

# Function to rotate a single log file
rotate_log_file() {
    local log_file="$1"
    local max_size_bytes="$2"
    local max_files="$3"
    
    log_debug "Processing log file: $log_file"
    
    if [[ ! -f "$log_file" ]]; then
        log_debug "Log file does not exist: $log_file"
        return 0
    fi
    
    ((total_files_processed++))
    
    local file_size
    file_size=$(get_file_size "$log_file")
    
    log_debug "File size: $(bytes_to_human "$file_size"), Max size: $(bytes_to_human "$max_size_bytes")"
    
    # Check if rotation is needed
    if [[ $file_size -lt $max_size_bytes ]]; then
        log_debug "File size is within limits, no rotation needed"
        return 0
    fi
    
    log_info "Rotating log file: $log_file ($(bytes_to_human "$file_size"))"
    
    # Shift existing rotated files
    for ((i = max_files - 1; i >= 1; i--)); do
        local old_file="${log_file}.${i}"
        local new_file="${log_file}.$((i + 1))"
        
        if [[ -f "$old_file" ]]; then
            if [[ "$DRY_RUN" == "true" ]]; then
                log_debug "[DRY RUN] Would move: $old_file -> $new_file"
            else
                mv "$old_file" "$new_file"
                log_debug "Moved: $old_file -> $new_file"
            fi
        fi
    done
    
    # Move current log file to .1
    local rotated_file="${log_file}.1"
    if [[ "$DRY_RUN" == "true" ]]; then
        log_debug "[DRY RUN] Would move: $log_file -> $rotated_file"
    else
        cp "$log_file" "$rotated_file"
        > "$log_file"  # Truncate original file
        log_debug "Rotated: $log_file -> $rotated_file"
    fi
    
    ((total_files_rotated++))
    
    # Compress rotated file if enabled
    if [[ "$COMPRESS_LOGS" == "true" ]]; then
        compress_log_file "$rotated_file"
    fi
    
    # Remove excess files
    local excess_file="${log_file}.$((max_files + 1))"
    if [[ -f "$excess_file" ]]; then
        local excess_size
        excess_size=$(get_file_size "$excess_file")
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "[DRY RUN] Would delete excess file: $excess_file"
        else
            rm -f "$excess_file"
            log_debug "Deleted excess file: $excess_file"
            ((total_files_deleted++))
            total_space_freed=$((total_space_freed + excess_size))
        fi
    fi
    
    # Check for compressed excess files
    local excess_compressed="${excess_file}.gz"
    if [[ -f "$excess_compressed" ]]; then
        local excess_size
        excess_size=$(get_file_size "$excess_compressed")
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "[DRY RUN] Would delete excess compressed file: $excess_compressed"
        else
            rm -f "$excess_compressed"
            log_debug "Deleted excess compressed file: $excess_compressed"
            ((total_files_deleted++))
            total_space_freed=$((total_space_freed + excess_size))
        fi
    fi
}

# Function to compress a log file
compress_log_file() {
    local log_file="$1"
    
    if [[ ! -f "$log_file" ]]; then
        return 0
    fi
    
    # Skip if already compressed
    if [[ "$log_file" == *.gz ]]; then
        return 0
    fi
    
    log_debug "Compressing log file: $log_file"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_debug "[DRY RUN] Would compress: $log_file"
    else
        if command -v gzip >/dev/null 2>&1; then
            gzip "$log_file"
            log_debug "Compressed: $log_file -> ${log_file}.gz"
            ((total_files_compressed++))
        else
            log_warn "gzip not available, skipping compression of $log_file"
        fi
    fi
}

# Function to clean old log files by age
clean_old_logs() {
    local log_dir="$1"
    local max_age_days="$2"
    
    if [[ ! -d "$log_dir" ]]; then
        return 0
    fi
    
    log_debug "Cleaning old logs in: $log_dir (older than $max_age_days days)"
    
    # Find and remove old log files
    while IFS= read -r -d '' old_file; do
        local file_size
        file_size=$(get_file_size "$old_file")
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "[DRY RUN] Would delete old file: $old_file ($(bytes_to_human "$file_size"))"
        else
            rm -f "$old_file"
            log_debug "Deleted old file: $old_file ($(bytes_to_human "$file_size"))"
            ((total_files_deleted++))
            total_space_freed=$((total_space_freed + file_size))
        fi
    done < <(find "$log_dir" -name "*.log*" -type f -mtime "+$max_age_days" -print0 2>/dev/null || true)
}

# Function to process worker logs
process_worker_logs() {
    local worker_log_dir="$LOG_BASE_DIR/worker"
    
    if [[ ! -d "$worker_log_dir" ]]; then
        log_debug "Worker log directory does not exist: $worker_log_dir"
        return 0
    fi
    
    log_info "Processing worker logs in: $worker_log_dir"
    
    local max_size_bytes
    max_size_bytes=$(size_to_bytes "$MAX_LOG_SIZE")
    
    # Process all worker log files
    for log_file in "$worker_log_dir"/*.log; do
        if [[ -f "$log_file" ]]; then
            rotate_log_file "$log_file" "$max_size_bytes" "$MAX_LOG_FILES"
        fi
    done
    
    # Clean old logs
    clean_old_logs "$worker_log_dir" "$MAX_AGE_DAYS"
}

# Function to process API logs
process_api_logs() {
    local api_log_dir="$LOG_BASE_DIR/api"
    
    if [[ ! -d "$api_log_dir" ]]; then
        log_debug "API log directory does not exist: $api_log_dir"
        return 0
    fi
    
    log_info "Processing API logs in: $api_log_dir"
    
    local max_size_bytes
    max_size_bytes=$(size_to_bytes "$MAX_LOG_SIZE")
    
    # Process API log files
    for log_file in "$api_log_dir"/*.log; do
        if [[ -f "$log_file" ]]; then
            rotate_log_file "$log_file" "$max_size_bytes" "$MAX_LOG_FILES"
        fi
    done
    
    # Clean old logs
    clean_old_logs "$api_log_dir" "$MAX_AGE_DAYS"
}

# Function to process coordinator logs
process_coordinator_logs() {
    local coordinator_log_dir="$LOG_BASE_DIR/coordinator"
    
    if [[ ! -d "$coordinator_log_dir" ]]; then
        log_debug "Coordinator log directory does not exist: $coordinator_log_dir"
        return 0
    fi
    
    log_info "Processing coordinator logs in: $coordinator_log_dir"
    
    local max_size_bytes
    max_size_bytes=$(size_to_bytes "$MAX_LOG_SIZE")
    
    # Process coordinator log files
    for log_file in "$coordinator_log_dir"/*.log; do
        if [[ -f "$log_file" ]]; then
            rotate_log_file "$log_file" "$max_size_bytes" "$MAX_LOG_FILES"
        fi
    done
    
    # Clean old logs
    clean_old_logs "$coordinator_log_dir" "$MAX_AGE_DAYS"
}

# Function to process system logs
process_system_logs() {
    local system_log_dir="$LOG_BASE_DIR/system"
    
    if [[ ! -d "$system_log_dir" ]]; then
        log_debug "System log directory does not exist: $system_log_dir"
        return 0
    fi
    
    log_info "Processing system logs in: $system_log_dir"
    
    local max_size_bytes
    max_size_bytes=$(size_to_bytes "$MAX_LOG_SIZE")
    
    # Process system log files
    for log_file in "$system_log_dir"/*.log; do
        if [[ -f "$log_file" ]]; then
            rotate_log_file "$log_file" "$max_size_bytes" "$MAX_LOG_FILES"
        fi
    done
    
    # Clean old logs
    clean_old_logs "$system_log_dir" "$MAX_AGE_DAYS"
}

# Function to show disk usage summary
show_disk_usage() {
    log_info "Disk usage summary for log directories:"
    
    if command -v du >/dev/null 2>&1; then
        for log_dir in "$LOG_BASE_DIR"/*; do
            if [[ -d "$log_dir" ]]; then
                local usage
                usage=$(du -sh "$log_dir" 2>/dev/null | cut -f1)
                echo "  $(basename "$log_dir"): $usage"
            fi
        done
        
        local total_usage
        total_usage=$(du -sh "$LOG_BASE_DIR" 2>/dev/null | cut -f1)
        echo "  Total: $total_usage"
    fi
}

# Function to create log rotation configuration
create_logrotate_config() {
    local config_file="$PROJECT_ROOT/logrotate.conf"
    
    log_info "Creating logrotate configuration: $config_file"
    
    cat > "$config_file" << EOF
# MLX Log Rotation Configuration
# Generated by log_rotation.sh

# Global settings
compress
delaycompress
missingok
notifempty
create 644 $(whoami) $(id -gn)

# Worker logs
$LOG_BASE_DIR/worker/*.log {
    size $MAX_LOG_SIZE
    rotate $MAX_LOG_FILES
    copytruncate
    postrotate
        # Send HUP signal to workers to reopen log files
        pkill -HUP -f "worker.main" || true
    endscript
}

# API logs
$LOG_BASE_DIR/api/*.log {
    size $MAX_LOG_SIZE
    rotate $MAX_LOG_FILES
    copytruncate
    postrotate
        # Send HUP signal to API server to reopen log files
        pkill -HUP -f "api.main" || true
    endscript
}

# Coordinator logs
$LOG_BASE_DIR/coordinator/*.log {
    size $MAX_LOG_SIZE
    rotate $MAX_LOG_FILES
    copytruncate
    postrotate
        # Send HUP signal to coordinator to reopen log files
        pkill -HUP -f "coordinator.main" || true
    endscript
}

# System logs
$LOG_BASE_DIR/system/*.log {
    size $MAX_LOG_SIZE
    rotate $MAX_LOG_FILES
    copytruncate
}
EOF
    
    log_info "Logrotate configuration created. Use with: logrotate -f $config_file"
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Rotate and clean up log files for MLX distributed inference system.

OPTIONS:
    -h, --help              Show this help message
    -s, --max-size SIZE     Maximum log file size (default: $MAX_LOG_SIZE)
    -f, --max-files NUM     Maximum number of rotated files (default: $MAX_LOG_FILES)
    -a, --max-age DAYS      Maximum age in days for log files (default: $MAX_AGE_DAYS)
    -c, --compress          Compress rotated logs (default: $COMPRESS_LOGS)
    --no-compress           Disable log compression
    -n, --dry-run           Show what would be done without making changes
    -v, --verbose           Enable verbose output
    --create-config         Create logrotate configuration file
    --disk-usage            Show disk usage summary
    --workers-only          Process only worker logs
    --api-only              Process only API logs
    --coordinator-only      Process only coordinator logs
    --system-only           Process only system logs

EXAMPLES:
    # Rotate all logs with default settings
    $0

    # Dry run to see what would be done
    $0 --dry-run --verbose

    # Rotate with custom settings
    $0 --max-size 50M --max-files 5 --max-age 14

    # Process only worker logs
    $0 --workers-only

    # Create logrotate configuration
    $0 --create-config

    # Show disk usage
    $0 --disk-usage

ENVIRONMENT VARIABLES:
    MAX_LOG_SIZE            Maximum log file size (e.g., 100M, 1G)
    MAX_LOG_FILES           Maximum number of rotated files
    MAX_AGE_DAYS            Maximum age in days for log files
    COMPRESS_LOGS           Enable/disable log compression (true/false)
    DRY_RUN                 Enable dry run mode (true/false)
    VERBOSE                 Enable verbose output (true/false)

EOF
}

# Main function
main() {
    local process_workers=true
    local process_api=true
    local process_coordinator=true
    local process_system=true
    local show_disk_usage_only=false
    local create_config_only=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -s|--max-size)
                MAX_LOG_SIZE="$2"
                shift 2
                ;;
            -f|--max-files)
                MAX_LOG_FILES="$2"
                shift 2
                ;;
            -a|--max-age)
                MAX_AGE_DAYS="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESS_LOGS="true"
                shift
                ;;
            --no-compress)
                COMPRESS_LOGS="false"
                shift
                ;;
            -n|--dry-run)
                DRY_RUN="true"
                shift
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            --create-config)
                create_config_only=true
                shift
                ;;
            --disk-usage)
                show_disk_usage_only=true
                shift
                ;;
            --workers-only)
                process_api=false
                process_coordinator=false
                process_system=false
                shift
                ;;
            --api-only)
                process_workers=false
                process_coordinator=false
                process_system=false
                shift
                ;;
            --coordinator-only)
                process_workers=false
                process_api=false
                process_system=false
                shift
                ;;
            --system-only)
                process_workers=false
                process_api=false
                process_coordinator=false
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate numeric parameters
    if ! [[ "$MAX_LOG_FILES" =~ ^[0-9]+$ ]] || [[ $MAX_LOG_FILES -lt 1 ]]; then
        log_error "Invalid max-files value: $MAX_LOG_FILES"
        exit 1
    fi
    
    if ! [[ "$MAX_AGE_DAYS" =~ ^[0-9]+$ ]] || [[ $MAX_AGE_DAYS -lt 1 ]]; then
        log_error "Invalid max-age value: $MAX_AGE_DAYS"
        exit 1
    fi
    
    # Create log base directory if it doesn't exist
    mkdir -p "$LOG_BASE_DIR"
    
    # Handle special actions
    if [[ "$create_config_only" == "true" ]]; then
        create_logrotate_config
        exit 0
    fi
    
    if [[ "$show_disk_usage_only" == "true" ]]; then
        show_disk_usage
        exit 0
    fi
    
    # Show configuration
    log_info "Starting log rotation with configuration:"
    log_info "  Max log size: $MAX_LOG_SIZE"
    log_info "  Max log files: $MAX_LOG_FILES"
    log_info "  Max age: $MAX_AGE_DAYS days"
    log_info "  Compress logs: $COMPRESS_LOGS"
    log_info "  Dry run: $DRY_RUN"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warn "DRY RUN MODE - No changes will be made"
    fi
    
    # Process logs
    if [[ "$process_workers" == "true" ]]; then
        process_worker_logs
    fi
    
    if [[ "$process_api" == "true" ]]; then
        process_api_logs
    fi
    
    if [[ "$process_coordinator" == "true" ]]; then
        process_coordinator_logs
    fi
    
    if [[ "$process_system" == "true" ]]; then
        process_system_logs
    fi
    
    # Show summary
    echo
    log_info "Log rotation completed successfully"
    log_info "Statistics:"
    log_info "  Files processed: $total_files_processed"
    log_info "  Files rotated: $total_files_rotated"
    log_info "  Files compressed: $total_files_compressed"
    log_info "  Files deleted: $total_files_deleted"
    log_info "  Space freed: $(bytes_to_human "$total_space_freed")"
    
    # Show disk usage after rotation
    if [[ "$VERBOSE" == "true" ]]; then
        echo
        show_disk_usage
    fi
}

# Run main function
main "$@"