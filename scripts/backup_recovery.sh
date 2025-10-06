#!/bin/bash

# MLX Backup and Recovery Script
# Comprehensive backup and recovery solution for MLX distributed inference system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_BASE_DIR="${BACKUP_BASE_DIR:-${PROJECT_ROOT}/backups}"
CONFIG_DIR="${PROJECT_ROOT}/config"
DATA_DIR="${PROJECT_ROOT}/data"
LOG_DIR="${PROJECT_ROOT}/logs"
MODELS_DIR="${PROJECT_ROOT}/models"

# Default configuration
BACKUP_TYPE="${BACKUP_TYPE:-full}"
RETENTION_DAYS="${RETENTION_DAYS:-30}"
COMPRESSION="${COMPRESSION:-true}"
ENCRYPTION="${ENCRYPTION:-false}"
ENCRYPTION_KEY="${ENCRYPTION_KEY:-}"
REMOTE_BACKUP="${REMOTE_BACKUP:-false}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_PATH="${REMOTE_PATH:-}"
VERIFY_BACKUP="${VERIFY_BACKUP:-true}"
EXCLUDE_LOGS="${EXCLUDE_LOGS:-false}"
EXCLUDE_MODELS="${EXCLUDE_MODELS:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Backup statistics
backup_start_time=""
backup_end_time=""
backup_size=0
files_backed_up=0

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

# Function to generate backup filename
generate_backup_filename() {
    local backup_type="$1"
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local hostname=$(hostname -s)
    
    echo "mlx_${backup_type}_${hostname}_${timestamp}"
}

# Function to create backup manifest
create_backup_manifest() {
    local backup_dir="$1"
    local manifest_file="$backup_dir/backup_manifest.json"
    
    log_debug "Creating backup manifest: $manifest_file"
    
    cat > "$manifest_file" << EOF
{
    "backup_info": {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "hostname": "$(hostname)",
        "backup_type": "$BACKUP_TYPE",
        "version": "1.0",
        "script_version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    },
    "system_info": {
        "os": "$(uname -s)",
        "arch": "$(uname -m)",
        "kernel": "$(uname -r)",
        "python_version": "$(python3 --version 2>&1 | cut -d' ' -f2)"
    },
    "backup_settings": {
        "compression": $COMPRESSION,
        "encryption": $ENCRYPTION,
        "exclude_logs": $EXCLUDE_LOGS,
        "exclude_models": $EXCLUDE_MODELS,
        "retention_days": $RETENTION_DAYS
    },
    "directories": {
        "config": "$(realpath "$CONFIG_DIR" 2>/dev/null || echo "$CONFIG_DIR")",
        "data": "$(realpath "$DATA_DIR" 2>/dev/null || echo "$DATA_DIR")",
        "logs": "$(realpath "$LOG_DIR" 2>/dev/null || echo "$LOG_DIR")",
        "models": "$(realpath "$MODELS_DIR" 2>/dev/null || echo "$MODELS_DIR")"
    }
}
EOF
}

# Function to backup configuration files
backup_configuration() {
    local backup_dir="$1"
    local config_backup_dir="$backup_dir/config"
    
    log_info "Backing up configuration files..."
    
    if [[ ! -d "$CONFIG_DIR" ]]; then
        log_warn "Configuration directory not found: $CONFIG_DIR"
        return 0
    fi
    
    mkdir -p "$config_backup_dir"
    
    # Copy all configuration files
    if cp -r "$CONFIG_DIR"/* "$config_backup_dir/" 2>/dev/null; then
        local file_count=$(find "$config_backup_dir" -type f | wc -l | tr -d ' ')
        log_info "Backed up $file_count configuration files"
        files_backed_up=$((files_backed_up + file_count))
    else
        log_warn "No configuration files found to backup"
    fi
}

# Function to backup data files
backup_data() {
    local backup_dir="$1"
    local data_backup_dir="$backup_dir/data"
    
    log_info "Backing up data files..."
    
    if [[ ! -d "$DATA_DIR" ]]; then
        log_debug "Data directory not found: $DATA_DIR"
        return 0
    fi
    
    mkdir -p "$data_backup_dir"
    
    # Copy data files
    if cp -r "$DATA_DIR"/* "$data_backup_dir/" 2>/dev/null; then
        local file_count=$(find "$data_backup_dir" -type f | wc -l | tr -d ' ')
        log_info "Backed up $file_count data files"
        files_backed_up=$((files_backed_up + file_count))
    else
        log_debug "No data files found to backup"
    fi
}

# Function to backup log files
backup_logs() {
    local backup_dir="$1"
    local logs_backup_dir="$backup_dir/logs"
    
    if [[ "$EXCLUDE_LOGS" == "true" ]]; then
        log_info "Skipping log files backup (excluded)"
        return 0
    fi
    
    log_info "Backing up log files..."
    
    if [[ ! -d "$LOG_DIR" ]]; then
        log_debug "Log directory not found: $LOG_DIR"
        return 0
    fi
    
    mkdir -p "$logs_backup_dir"
    
    # Copy only recent log files (last 7 days)
    find "$LOG_DIR" -name "*.log*" -mtime -7 -type f -exec cp {} "$logs_backup_dir/" \; 2>/dev/null || true
    
    local file_count=$(find "$logs_backup_dir" -type f | wc -l | tr -d ' ')
    if [[ $file_count -gt 0 ]]; then
        log_info "Backed up $file_count recent log files"
        files_backed_up=$((files_backed_up + file_count))
    else
        log_debug "No recent log files found to backup"
    fi
}

# Function to backup model files
backup_models() {
    local backup_dir="$1"
    local models_backup_dir="$backup_dir/models"
    
    if [[ "$EXCLUDE_MODELS" == "true" ]]; then
        log_info "Skipping model files backup (excluded)"
        return 0
    fi
    
    log_info "Backing up model files..."
    
    if [[ ! -d "$MODELS_DIR" ]]; then
        log_warn "Models directory not found: $MODELS_DIR"
        return 0
    fi
    
    mkdir -p "$models_backup_dir"
    
    # Create symlinks to model files instead of copying (models can be very large)
    find "$MODELS_DIR" -type f -name "*.bin" -o -name "*.safetensors" -o -name "*.gguf" | while read -r model_file; do
        local relative_path=$(realpath --relative-to="$MODELS_DIR" "$model_file")
        local target_dir="$models_backup_dir/$(dirname "$relative_path")"
        mkdir -p "$target_dir"
        
        # Create a reference file instead of copying the large model
        echo "Model file: $model_file" > "$models_backup_dir/$relative_path.ref"
        echo "Size: $(stat -f%z "$model_file" 2>/dev/null || echo 'unknown')" >> "$models_backup_dir/$relative_path.ref"
        echo "Modified: $(stat -f%Sm "$model_file" 2>/dev/null || echo 'unknown')" >> "$models_backup_dir/$relative_path.ref"
        echo "Checksum: $(shasum -a 256 "$model_file" 2>/dev/null | cut -d' ' -f1 || echo 'unknown')" >> "$models_backup_dir/$relative_path.ref"
    done
    
    local ref_count=$(find "$models_backup_dir" -name "*.ref" | wc -l | tr -d ' ')
    if [[ $ref_count -gt 0 ]]; then
        log_info "Created references for $ref_count model files"
        files_backed_up=$((files_backed_up + ref_count))
    else
        log_debug "No model files found to reference"
    fi
}

# Function to backup database files
backup_databases() {
    local backup_dir="$1"
    local db_backup_dir="$backup_dir/databases"
    
    log_info "Backing up database files..."
    
    mkdir -p "$db_backup_dir"
    
    # Look for SQLite databases
    find "$PROJECT_ROOT" -name "*.db" -o -name "*.sqlite" -o -name "*.sqlite3" | while read -r db_file; do
        if [[ -f "$db_file" ]]; then
            local db_name=$(basename "$db_file")
            log_debug "Backing up database: $db_name"
            cp "$db_file" "$db_backup_dir/"
            files_backed_up=$((files_backed_up + 1))
        fi
    done
    
    # Export any running database data if needed
    # This would be extended for specific database types
}

# Function to compress backup
compress_backup() {
    local backup_dir="$1"
    local compressed_file="${backup_dir}.tar.gz"
    
    if [[ "$COMPRESSION" != "true" ]]; then
        log_debug "Compression disabled, skipping"
        return 0
    fi
    
    log_info "Compressing backup..."
    
    local original_size=$(du -sb "$backup_dir" | cut -f1)
    
    if tar -czf "$compressed_file" -C "$(dirname "$backup_dir")" "$(basename "$backup_dir")"; then
        local compressed_size=$(stat -f%z "$compressed_file" 2>/dev/null || echo "0")
        local compression_ratio=$(echo "scale=1; $compressed_size * 100 / $original_size" | bc)
        
        log_info "Backup compressed successfully"
        log_info "Original size: $(bytes_to_human "$original_size")"
        log_info "Compressed size: $(bytes_to_human "$compressed_size") (${compression_ratio}%)"
        
        # Remove uncompressed directory
        rm -rf "$backup_dir"
        
        backup_size=$compressed_size
        echo "$compressed_file"
    else
        log_error "Failed to compress backup"
        backup_size=$original_size
        echo "$backup_dir"
    fi
}

# Function to encrypt backup
encrypt_backup() {
    local backup_file="$1"
    local encrypted_file="${backup_file}.enc"
    
    if [[ "$ENCRYPTION" != "true" ]]; then
        log_debug "Encryption disabled, skipping"
        echo "$backup_file"
        return 0
    fi
    
    if [[ -z "$ENCRYPTION_KEY" ]]; then
        log_error "Encryption enabled but no encryption key provided"
        echo "$backup_file"
        return 1
    fi
    
    log_info "Encrypting backup..."
    
    if command -v openssl >/dev/null 2>&1; then
        if openssl enc -aes-256-cbc -salt -in "$backup_file" -out "$encrypted_file" -k "$ENCRYPTION_KEY"; then
            log_info "Backup encrypted successfully"
            rm -f "$backup_file"
            echo "$encrypted_file"
        else
            log_error "Failed to encrypt backup"
            echo "$backup_file"
        fi
    else
        log_error "OpenSSL not available for encryption"
        echo "$backup_file"
    fi
}

# Function to verify backup integrity
verify_backup() {
    local backup_file="$1"
    
    if [[ "$VERIFY_BACKUP" != "true" ]]; then
        log_debug "Backup verification disabled, skipping"
        return 0
    fi
    
    log_info "Verifying backup integrity..."
    
    # Check if file exists and is readable
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    if [[ ! -r "$backup_file" ]]; then
        log_error "Backup file not readable: $backup_file"
        return 1
    fi
    
    # Verify compressed file integrity
    if [[ "$backup_file" == *.tar.gz ]]; then
        if tar -tzf "$backup_file" >/dev/null 2>&1; then
            log_info "Backup archive integrity verified"
        else
            log_error "Backup archive is corrupted"
            return 1
        fi
    fi
    
    # Generate and store checksum
    local checksum_file="${backup_file}.sha256"
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$backup_file" > "$checksum_file"
        log_info "Backup checksum saved: $checksum_file"
    fi
    
    return 0
}

# Function to upload backup to remote location
upload_backup() {
    local backup_file="$1"
    
    if [[ "$REMOTE_BACKUP" != "true" ]]; then
        log_debug "Remote backup disabled, skipping"
        return 0
    fi
    
    if [[ -z "$REMOTE_HOST" || -z "$REMOTE_PATH" ]]; then
        log_error "Remote backup enabled but host or path not configured"
        return 1
    fi
    
    log_info "Uploading backup to remote location..."
    log_info "Remote: $REMOTE_HOST:$REMOTE_PATH"
    
    local remote_file="$REMOTE_PATH/$(basename "$backup_file")"
    
    if command -v rsync >/dev/null 2>&1; then
        if rsync -avz --progress "$backup_file" "$REMOTE_HOST:$remote_file"; then
            log_info "Backup uploaded successfully"
            
            # Upload checksum file if it exists
            local checksum_file="${backup_file}.sha256"
            if [[ -f "$checksum_file" ]]; then
                rsync -avz "$checksum_file" "$REMOTE_HOST:${remote_file}.sha256"
            fi
        else
            log_error "Failed to upload backup"
            return 1
        fi
    elif command -v scp >/dev/null 2>&1; then
        if scp "$backup_file" "$REMOTE_HOST:$remote_file"; then
            log_info "Backup uploaded successfully"
            
            # Upload checksum file if it exists
            local checksum_file="${backup_file}.sha256"
            if [[ -f "$checksum_file" ]]; then
                scp "$checksum_file" "$REMOTE_HOST:${remote_file}.sha256"
            fi
        else
            log_error "Failed to upload backup"
            return 1
        fi
    else
        log_error "No suitable tool found for remote backup (rsync or scp required)"
        return 1
    fi
}

# Function to clean old backups
clean_old_backups() {
    log_info "Cleaning old backups (older than $RETENTION_DAYS days)..."
    
    local deleted_count=0
    local freed_space=0
    
    # Clean local backups
    find "$BACKUP_BASE_DIR" -name "mlx_*" -type f -mtime "+$RETENTION_DAYS" | while read -r old_backup; do
        local file_size=$(stat -f%z "$old_backup" 2>/dev/null || echo "0")
        log_debug "Deleting old backup: $old_backup ($(bytes_to_human "$file_size"))"
        rm -f "$old_backup"
        rm -f "${old_backup}.sha256"  # Remove checksum file too
        deleted_count=$((deleted_count + 1))
        freed_space=$((freed_space + file_size))
    done
    
    if [[ $deleted_count -gt 0 ]]; then
        log_info "Deleted $deleted_count old backups, freed $(bytes_to_human "$freed_space")"
    else
        log_info "No old backups found to delete"
    fi
}

# Function to list available backups
list_backups() {
    log_info "Available backups in $BACKUP_BASE_DIR:"
    
    if [[ ! -d "$BACKUP_BASE_DIR" ]]; then
        log_warn "Backup directory does not exist: $BACKUP_BASE_DIR"
        return 0
    fi
    
    local backup_count=0
    
    for backup_file in "$BACKUP_BASE_DIR"/mlx_*; do
        if [[ -f "$backup_file" ]]; then
            local file_size=$(stat -f%z "$backup_file" 2>/dev/null || echo "0")
            local file_date=$(stat -f%Sm "$backup_file" 2>/dev/null || echo "unknown")
            
            echo "  $(basename "$backup_file") - $(bytes_to_human "$file_size") - $file_date"
            backup_count=$((backup_count + 1))
        fi
    done
    
    if [[ $backup_count -eq 0 ]]; then
        echo "  No backups found"
    else
        echo "  Total: $backup_count backups"
    fi
}

# Function to restore from backup
restore_backup() {
    local backup_file="$1"
    local restore_dir="${2:-$PROJECT_ROOT}"
    
    log_info "Restoring from backup: $backup_file"
    log_info "Restore location: $restore_dir"
    
    if [[ ! -f "$backup_file" ]]; then
        log_error "Backup file not found: $backup_file"
        return 1
    fi
    
    # Verify backup before restore
    if ! verify_backup "$backup_file"; then
        log_error "Backup verification failed, aborting restore"
        return 1
    fi
    
    # Create restore directory
    mkdir -p "$restore_dir"
    
    # Handle encrypted backups
    local working_file="$backup_file"
    if [[ "$backup_file" == *.enc ]]; then
        if [[ -z "$ENCRYPTION_KEY" ]]; then
            log_error "Encrypted backup but no decryption key provided"
            return 1
        fi
        
        log_info "Decrypting backup..."
        local decrypted_file="${backup_file%.enc}"
        
        if openssl enc -aes-256-cbc -d -in "$backup_file" -out "$decrypted_file" -k "$ENCRYPTION_KEY"; then
            working_file="$decrypted_file"
        else
            log_error "Failed to decrypt backup"
            return 1
        fi
    fi
    
    # Extract backup
    if [[ "$working_file" == *.tar.gz ]]; then
        log_info "Extracting compressed backup..."
        if tar -xzf "$working_file" -C "$restore_dir" --strip-components=1; then
            log_info "Backup extracted successfully"
        else
            log_error "Failed to extract backup"
            return 1
        fi
    else
        log_info "Copying backup files..."
        if cp -r "$working_file"/* "$restore_dir/"; then
            log_info "Backup files copied successfully"
        else
            log_error "Failed to copy backup files"
            return 1
        fi
    fi
    
    # Clean up temporary decrypted file
    if [[ "$working_file" != "$backup_file" ]]; then
        rm -f "$working_file"
    fi
    
    log_info "Restore completed successfully"
    log_warn "Please restart all MLX services to use restored configuration"
}

# Function to create full backup
create_full_backup() {
    local backup_name
    backup_name=$(generate_backup_filename "full")
    local backup_dir="$BACKUP_BASE_DIR/$backup_name"
    
    log_info "Creating full backup: $backup_name"
    
    backup_start_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Create backup directory
    mkdir -p "$backup_dir"
    
    # Create manifest
    create_backup_manifest "$backup_dir"
    
    # Backup components
    backup_configuration "$backup_dir"
    backup_data "$backup_dir"
    backup_logs "$backup_dir"
    backup_models "$backup_dir"
    backup_databases "$backup_dir"
    
    backup_end_time=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Compress backup
    local final_backup
    final_backup=$(compress_backup "$backup_dir")
    
    # Encrypt if enabled
    final_backup=$(encrypt_backup "$final_backup")
    
    # Verify backup
    if ! verify_backup "$final_backup"; then
        log_error "Backup verification failed"
        return 1
    fi
    
    # Upload to remote if enabled
    upload_backup "$final_backup"
    
    # Update statistics
    backup_size=$(stat -f%z "$final_backup" 2>/dev/null || echo "0")
    
    log_info "Full backup completed successfully"
    log_info "Backup file: $final_backup"
    log_info "Backup size: $(bytes_to_human "$backup_size")"
    log_info "Files backed up: $files_backed_up"
    log_info "Start time: $backup_start_time"
    log_info "End time: $backup_end_time"
    
    return 0
}

# Function to create incremental backup
create_incremental_backup() {
    log_warn "Incremental backup not yet implemented"
    log_info "Falling back to full backup"
    create_full_backup
}

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] COMMAND

Backup and recovery solution for MLX distributed inference system.

COMMANDS:
    backup              Create a backup (default: full)
    restore FILE        Restore from backup file
    list                List available backups
    clean               Clean old backups
    verify FILE         Verify backup integrity

OPTIONS:
    -h, --help              Show this help message
    -t, --type TYPE         Backup type: full, incremental (default: full)
    -d, --dir DIR           Backup directory (default: $BACKUP_BASE_DIR)
    -r, --retention DAYS    Retention period in days (default: $RETENTION_DAYS)
    -c, --compress          Enable compression (default: $COMPRESSION)
    --no-compress           Disable compression
    -e, --encrypt           Enable encryption
    --encryption-key KEY    Encryption key for backup
    --exclude-logs          Exclude log files from backup
    --exclude-models        Exclude model files from backup
    --remote-host HOST      Remote host for backup upload
    --remote-path PATH      Remote path for backup upload
    --no-verify             Skip backup verification
    --debug                 Enable debug output

EXAMPLES:
    # Create full backup
    $0 backup

    # Create backup with compression and encryption
    $0 --compress --encrypt --encryption-key "mykey123" backup

    # Create backup excluding logs and models
    $0 --exclude-logs --exclude-models backup

    # List available backups
    $0 list

    # Restore from backup
    $0 restore /path/to/backup.tar.gz

    # Clean old backups
    $0 clean

    # Verify backup integrity
    $0 verify /path/to/backup.tar.gz

ENVIRONMENT VARIABLES:
    BACKUP_BASE_DIR         Base directory for backups
    RETENTION_DAYS          Backup retention period
    COMPRESSION             Enable/disable compression
    ENCRYPTION              Enable/disable encryption
    ENCRYPTION_KEY          Encryption key
    REMOTE_BACKUP           Enable/disable remote backup
    REMOTE_HOST             Remote backup host
    REMOTE_PATH             Remote backup path

EOF
}

# Main function
main() {
    local command=""
    local restore_file=""
    local verify_file=""
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_usage
                exit 0
                ;;
            -t|--type)
                BACKUP_TYPE="$2"
                shift 2
                ;;
            -d|--dir)
                BACKUP_BASE_DIR="$2"
                shift 2
                ;;
            -r|--retention)
                RETENTION_DAYS="$2"
                shift 2
                ;;
            -c|--compress)
                COMPRESSION="true"
                shift
                ;;
            --no-compress)
                COMPRESSION="false"
                shift
                ;;
            -e|--encrypt)
                ENCRYPTION="true"
                shift
                ;;
            --encryption-key)
                ENCRYPTION_KEY="$2"
                ENCRYPTION="true"
                shift 2
                ;;
            --exclude-logs)
                EXCLUDE_LOGS="true"
                shift
                ;;
            --exclude-models)
                EXCLUDE_MODELS="true"
                shift
                ;;
            --remote-host)
                REMOTE_HOST="$2"
                REMOTE_BACKUP="true"
                shift 2
                ;;
            --remote-path)
                REMOTE_PATH="$2"
                REMOTE_BACKUP="true"
                shift 2
                ;;
            --no-verify)
                VERIFY_BACKUP="false"
                shift
                ;;
            --debug)
                DEBUG="true"
                shift
                ;;
            backup)
                command="backup"
                shift
                ;;
            restore)
                command="restore"
                restore_file="$2"
                shift 2
                ;;
            list)
                command="list"
                shift
                ;;
            clean)
                command="clean"
                shift
                ;;
            verify)
                command="verify"
                verify_file="$2"
                shift 2
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Default command
    if [[ -z "$command" ]]; then
        command="backup"
    fi
    
    # Validate backup type
    if [[ "$BACKUP_TYPE" != "full" && "$BACKUP_TYPE" != "incremental" ]]; then
        log_error "Invalid backup type: $BACKUP_TYPE"
        exit 1
    fi
    
    # Create backup base directory
    mkdir -p "$BACKUP_BASE_DIR"
    
    # Execute command
    case "$command" in
        "backup")
            case "$BACKUP_TYPE" in
                "full")
                    create_full_backup
                    ;;
                "incremental")
                    create_incremental_backup
                    ;;
            esac
            clean_old_backups
            ;;
        "restore")
            if [[ -z "$restore_file" ]]; then
                log_error "Restore file not specified"
                exit 1
            fi
            restore_backup "$restore_file"
            ;;
        "list")
            list_backups
            ;;
        "clean")
            clean_old_backups
            ;;
        "verify")
            if [[ -z "$verify_file" ]]; then
                log_error "Verify file not specified"
                exit 1
            fi
            verify_backup "$verify_file"
            ;;
        *)
            log_error "Invalid command: $command"
            exit 1
            ;;
    esac
}

# Run main function
main "$@"