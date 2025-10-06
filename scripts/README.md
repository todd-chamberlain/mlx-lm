# MLX Deployment Scripts

This directory contains comprehensive deployment automation scripts for the MLX distributed inference system. These scripts provide production-ready deployment, monitoring, and maintenance capabilities for both containerized services and native MLX workers on macOS.

## Scripts Overview

### Core Deployment Scripts

#### `deploy.sh` - Master Deployment Script
The main deployment automation script that orchestrates the entire system deployment.

**Features:**
- Complete system deployment with prerequisite checks
- Support for development and production modes
- Automated containerized services deployment
- Native MLX worker deployment on macOS
- Configuration validation and backup creation
- Health verification and monitoring setup

**Usage:**
```bash
# Production deployment with 2 workers
./scripts/deploy.sh --mode production --workers 2

# Development deployment
./scripts/deploy.sh --mode development --workers 1

# Custom coordinator deployment
./scripts/deploy.sh --coordinator localhost:50051 --workers 3
```

#### `start_worker_macos.sh` - Native Worker Startup
Starts native MLX workers on macOS with proper configuration and monitoring.

**Features:**
- System requirements validation (Apple Silicon optimization)
- Automatic configuration generation
- Process management with PID files
- Signal handling for graceful shutdown
- Comprehensive logging and error handling
- Virtual environment support

**Usage:**
```bash
# Start worker with default configuration
./scripts/start_worker_macos.sh

# Start worker with custom configuration
./scripts/start_worker_macos.sh --config /path/to/worker.yaml --worker-id my-worker

# Start worker in foreground with debug logging
./scripts/start_worker_macos.sh --foreground --debug --log-level DEBUG
```

#### `stop_worker.sh` - Worker Management
Gracefully stops MLX workers with proper cleanup.

**Features:**
- Graceful shutdown with SIGTERM
- Force kill option for unresponsive workers
- Bulk operations (stop all workers)
- PID file cleanup
- Configurable timeout handling

**Usage:**
```bash
# Stop specific worker
./scripts/stop_worker.sh --worker-id my-worker

# Stop all workers
./scripts/stop_worker.sh --all

# Force kill worker
./scripts/stop_worker.sh --worker-id my-worker --force
```

### Monitoring and Maintenance Scripts

#### `health_check.sh` - System Health Monitoring
Comprehensive health monitoring for all system components.

**Features:**
- System resource monitoring (CPU, memory, disk)
- Worker process health checks
- HTTP endpoint validation
- Coordinator connectivity testing
- Log file analysis
- JSON and text output formats

**Usage:**
```bash
# Check all workers
./scripts/health_check.sh --all

# Check specific worker
./scripts/health_check.sh --worker-id my-worker

# JSON output for monitoring systems
./scripts/health_check.sh --all --format json
```

#### `log_rotation.sh` - Log Management
Automated log rotation and cleanup with configurable retention policies.

**Features:**
- Size-based log rotation
- Age-based cleanup
- Compression support
- Service-specific configuration
- Disk usage monitoring
- Logrotate integration

**Usage:**
```bash
# Rotate all logs with default settings
./scripts/log_rotation.sh

# Custom rotation settings
./scripts/log_rotation.sh --max-size 50M --max-files 5 --max-age 14

# Dry run to preview changes
./scripts/log_rotation.sh --dry-run --verbose
```

#### `backup_recovery.sh` - Backup and Recovery
Comprehensive backup and recovery solution for system data and configuration.

**Features:**
- Full and incremental backups
- Compression and encryption support
- Remote backup capabilities
- Backup verification and integrity checks
- Automated retention management
- Model file reference handling (to avoid copying large files)

**Usage:**
```bash
# Create full backup
./scripts/backup_recovery.sh backup

# Encrypted backup
./scripts/backup_recovery.sh --encrypt --encryption-key "mykey123" backup

# Restore from backup
./scripts/backup_recovery.sh restore /path/to/backup.tar.gz

# List available backups
./scripts/backup_recovery.sh list
```

## Directory Structure

After deployment, the following directory structure is created:

```
project_root/
├── scripts/                 # Deployment and management scripts
├── config/                  # Configuration files
├── logs/                    # Log files
│   ├── api/                # API server logs
│   ├── coordinator/        # Coordinator service logs
│   ├── worker/             # Worker logs
│   └── system/             # System and monitoring logs
├── run/                     # PID files for running processes
├── data/                    # Application data
│   ├── api/               # API server data
│   ├── coordinator/       # Coordinator data
│   └── worker/            # Worker data
├── backups/                # System backups
└── models/                 # ML model files
```

## Environment Variables

The scripts support various environment variables for configuration:

### Global Variables
- `DEPLOYMENT_MODE`: Deployment mode (development/production)
- `PROJECT_ROOT`: Project root directory
- `LOG_LEVEL`: Logging level (DEBUG/INFO/WARN/ERROR)
- `DEBUG`: Enable debug output (true/false)

### Worker Configuration
- `WORKER_ID`: Worker identifier
- `COORDINATOR_HOST`: Coordinator address (host:port)
- `MODEL_PATH`: Model directory path
- `PYTHON_ENV`: Python virtual environment path

### Backup Configuration
- `BACKUP_BASE_DIR`: Base directory for backups
- `RETENTION_DAYS`: Backup retention period
- `COMPRESSION`: Enable/disable compression
- `ENCRYPTION`: Enable/disable encryption
- `REMOTE_BACKUP`: Enable/disable remote backup

### Log Rotation Configuration
- `MAX_LOG_SIZE`: Maximum log file size
- `MAX_LOG_FILES`: Maximum number of rotated files
- `MAX_AGE_DAYS`: Maximum age for log files
- `COMPRESS_LOGS`: Enable/disable log compression

## Production Deployment Guide

### Prerequisites
1. macOS with Apple Silicon (M1/M2/M3) for optimal performance
2. Python 3.11+ with MLX installed
3. Docker and Docker Compose
4. Sufficient disk space for models and logs

### Step-by-Step Deployment

1. **Prepare the environment:**
   ```bash
   # Clone the repository
   git clone <repository-url>
   cd mlx-distributed-inference
   
   # Install Python dependencies
   pip install -r requirements.txt
   ```

2. **Configure the system:**
   ```bash
   # Copy example configuration
   cp .env.example .env
   
   # Edit configuration files
   vim config/api.yaml
   vim config/coordinator.yaml
   vim config/worker.yaml
   ```

3. **Deploy the system:**
   ```bash
   # Production deployment with 2 workers
   ./scripts/deploy.sh --mode production --workers 2
   ```

4. **Verify deployment:**
   ```bash
   # Check all services
   ./scripts/health_check.sh --all
   
   # Test API endpoint
   curl http://localhost:8000/health
   ```

5. **Setup monitoring:**
   ```bash
   # Add cron jobs for automated monitoring
   crontab cron_jobs.txt
   ```

### Monitoring and Maintenance

#### Daily Operations
- Monitor system health: `./scripts/health_check.sh --all`
- Check logs: `tail -f logs/*/*.log`
- Monitor disk usage: `df -h`

#### Weekly Maintenance
- Rotate logs: `./scripts/log_rotation.sh`
- Create backup: `./scripts/backup_recovery.sh backup`
- Clean old backups: `./scripts/backup_recovery.sh clean`

#### Troubleshooting
- Check worker status: `./scripts/health_check.sh --worker-id <worker-id>`
- Restart worker: `./scripts/stop_worker.sh --worker-id <worker-id> && ./scripts/start_worker_macos.sh --worker-id <worker-id>`
- View detailed logs: `tail -f logs/worker/worker-<worker-id>.log`

## Security Considerations

1. **File Permissions**: Scripts automatically set appropriate permissions for log and data directories
2. **Process Isolation**: Workers run as separate processes with PID file management
3. **Configuration Security**: Sensitive configuration can be encrypted in backups
4. **Network Security**: Coordinator communication can be secured with TLS (configure in coordinator settings)

## Integration with CI/CD

The deployment scripts are designed to integrate with CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
deploy:
  runs-on: macos-latest
  steps:
    - uses: actions/checkout@v2
    - name: Deploy MLX System
      run: |
        ./scripts/deploy.sh --mode production --workers 2 --skip-backup
    - name: Verify Deployment
      run: |
        ./scripts/health_check.sh --all --format json
```

## Customization

The scripts are designed to be easily customizable:

1. **Configuration Templates**: Modify default configuration generation in `start_worker_macos.sh`
2. **Health Checks**: Add custom health checks in `health_check.sh`
3. **Backup Policies**: Customize backup retention and compression in `backup_recovery.sh`
4. **Monitoring**: Extend monitoring capabilities in `log_rotation.sh`

## Support and Troubleshooting

### Common Issues

1. **Worker fails to start**: Check MLX installation and model path
2. **Coordinator connection failed**: Verify coordinator is running and accessible
3. **High memory usage**: Adjust `gpu_memory_fraction` in worker configuration
4. **Log files growing too large**: Configure log rotation with smaller size limits

### Getting Help

- Check script help: `./scripts/<script-name>.sh --help`
- Enable debug logging: `DEBUG=true ./scripts/<script-name>.sh`
- Review logs: `tail -f logs/system/*.log`

For additional support, refer to the main project documentation and issue tracker.