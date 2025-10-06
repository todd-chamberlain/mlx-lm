# System Health and Recovery Documentation

This document describes the system health monitoring, resource management, and recovery mechanisms implemented for the MLX distributed inference system.

## Overview

The recovery system provides:
- **Automatic health monitoring** with configurable checks and thresholds
- **Resource monitoring** for CPU, memory, and disk usage
- **Automatic recovery actions** triggered by health issues
- **Data consistency checking** and repair capabilities
- **File integrity verification** using checksums

## Components

### 1. ServiceHealthMonitor

Monitors service health and triggers recovery actions when issues are detected.

#### Features
- Configurable health checks
- State machine with 5 states: healthy, degraded, unhealthy, recovering, failed
- Automatic state transitions based on consecutive failures/successes
- Priority-based recovery actions with retry limits
- Complete health history tracking

#### Usage Example

```python
from common.service_recovery import ServiceHealthMonitor, HealthCheckResult, ServiceState

# Create monitor
monitor = ServiceHealthMonitor(
    service_name="api-server",
    check_interval=30.0,
    unhealthy_threshold=3,
    recovery_threshold=2
)

# Add health check
def check_api_health():
    # Your health check logic
    is_healthy = check_api_endpoint()
    return HealthCheckResult(
        is_healthy=is_healthy,
        state=ServiceState.HEALTHY if is_healthy else ServiceState.UNHEALTHY,
        message="API endpoint check"
    )

monitor.add_health_check(check_api_health)

# Add recovery action
def restart_api_server():
    # Your recovery logic
    return restart_service("api-server")

monitor.add_recovery_action(
    name="restart-api",
    action=restart_api_server,
    priority=10,
    max_retries=3
)

# Start monitoring
await monitor.start_monitoring()
```

### 2. ResourceMonitor

Monitors system resources and triggers cleanup when thresholds are exceeded.

#### Features
- Real-time CPU, memory, and disk monitoring
- Configurable thresholds for each resource
- Automatic cleanup action triggers
- Resource usage history tracking
- Detailed resource status reporting

#### Usage Example

```python
from common.resource_monitor import ResourceMonitor, ResourceThresholds

# Create monitor with custom thresholds
thresholds = ResourceThresholds(
    cpu_percent=80.0,
    memory_percent=85.0,
    disk_percent=90.0,
    min_disk_free_gb=5.0
)

monitor = ResourceMonitor(
    thresholds=thresholds,
    check_interval=60.0
)

# Add cleanup action
def cleanup_old_data(exceeded):
    if 'disk' in exceeded:
        # Clean up old logs, cache, etc.
        cleanup_logs()
        cleanup_cache()

monitor.add_cleanup_action(cleanup_old_data)

# Start monitoring
await monitor.start_monitoring()

# Get current status
status = monitor.get_resource_status()
print(f"CPU: {status['cpu_percent']}%")
print(f"Memory: {status['memory_percent']}%")
print(f"Disk: {status['disk_percent']}%")
```

### 3. DataConsistencyChecker

Checks and repairs data consistency issues.

#### Features
- Pluggable consistency checks
- Issue categorization by severity (low, medium, high, critical)
- Automatic and manual repair options
- Repair history tracking
- Comprehensive consistency reports

#### Usage Example

```python
from common.data_consistency import DataConsistencyChecker, ConsistencyIssue
from datetime import datetime

# Create checker
checker = DataConsistencyChecker(data_dir="./data")

# Add consistency check
def check_database_integrity():
    issues = []
    # Your consistency check logic
    if database_has_orphaned_records():
        issues.append(ConsistencyIssue(
            issue_type="orphaned_records",
            severity="medium",
            description="Found orphaned database records",
            affected_data="database",
            timestamp=datetime.now(),
            auto_repairable=True
        ))
    return issues

checker.add_consistency_check(check_database_integrity)

# Add repair action
def repair_orphaned_records(issue):
    # Your repair logic
    return cleanup_orphaned_records()

checker.add_repair_action("orphaned_records", repair_orphaned_records)

# Run checks and repairs
issues = await checker.run_consistency_checks()
print(f"Found {len(issues)} issues")

# Repair auto-repairable issues
stats = await checker.repair_all_issues(auto_only=True)
print(f"Repaired: {stats['repaired']}, Failed: {stats['failed']}")

# Get report
report = checker.get_consistency_report()
```

### 4. FileIntegrityChecker

Verifies file integrity using SHA256 checksums.

#### Features
- SHA256 checksum calculation and verification
- JSON manifest persistence
- Batch file verification
- Corruption detection
- Missing file detection

#### Usage Example

```python
from common.data_consistency import FileIntegrityChecker
from pathlib import Path

# Create checker
checker = FileIntegrityChecker(manifest_file=Path("data/manifest.json"))

# Add files to manifest
important_files = [
    Path("config/api.yaml"),
    Path("config/coordinator.yaml"),
    Path("data/models/model.bin")
]

for file_path in important_files:
    checker.add_file(file_path)

# Save manifest
checker.save_manifest()

# Later, verify files
corrupted = checker.verify_all_files()
if corrupted:
    print(f"Corrupted files: {corrupted}")
else:
    print("All files verified successfully")
```

### 5. CleanupManager

Manages cleanup operations for logs, cache, and temporary files.

#### Features
- Age-based log file cleanup
- Cache directory cleanup
- Temporary file cleanup
- Configurable retention policies

#### Usage Example

```python
from common.resource_monitor import CleanupManager
from pathlib import Path

# Create cleanup manager
cleanup = CleanupManager(
    log_dir=Path("logs"),
    cache_dir=Path("cache")
)

# Clean up old logs (older than 7 days)
count = cleanup.cleanup_old_logs(max_age_days=7)
print(f"Deleted {count} old log files")

# Clean up cache
count = cleanup.cleanup_cache()
print(f"Deleted {count} cache files")

# Clean up temp files
count = cleanup.cleanup_temp_files()
print(f"Deleted {count} temp files")
```

## Integration Example

Here's how to integrate all components in a service:

```python
import asyncio
from common.service_recovery import ServiceHealthMonitor
from common.resource_monitor import ResourceMonitor, CleanupManager
from common.data_consistency import DataConsistencyChecker

async def setup_recovery_system():
    # Health monitoring
    health_monitor = ServiceHealthMonitor("my-service")
    health_monitor.add_health_check(check_service_health)
    health_monitor.add_recovery_action("restart", restart_service)
    
    # Resource monitoring
    resource_monitor = ResourceMonitor()
    cleanup_manager = CleanupManager()
    
    def cleanup_resources(exceeded):
        if 'disk' in exceeded:
            cleanup_manager.cleanup_old_logs()
            cleanup_manager.cleanup_cache()
    
    resource_monitor.add_cleanup_action(cleanup_resources)
    
    # Data consistency
    consistency_checker = DataConsistencyChecker()
    consistency_checker.add_consistency_check(check_data_integrity)
    consistency_checker.add_repair_action("corruption", repair_corruption)
    
    # Start monitoring
    await health_monitor.start_monitoring()
    await resource_monitor.start_monitoring()
    
    # Periodic consistency checks
    async def periodic_consistency_check():
        while True:
            await asyncio.sleep(3600)  # Every hour
            issues = await consistency_checker.run_consistency_checks()
            if issues:
                await consistency_checker.repair_all_issues(auto_only=True)
    
    asyncio.create_task(periodic_consistency_check())

# Run in your service
asyncio.run(setup_recovery_system())
```

## State Machine

The ServiceHealthMonitor uses a state machine:

```
HEALTHY → DEGRADED → UNHEALTHY → RECOVERING → HEALTHY
   ↑                      ↓
   └──────────────────────┘
```

- **HEALTHY**: All checks passing
- **DEGRADED**: Some checks failing, but below unhealthy threshold
- **UNHEALTHY**: Consecutive failures exceed threshold, triggers recovery
- **RECOVERING**: Recovery actions executed, waiting for health to stabilize
- **FAILED**: Recovery actions exhausted, manual intervention needed

## Configuration

### Health Monitor Configuration

```python
ServiceHealthMonitor(
    service_name="my-service",
    check_interval=30.0,          # Seconds between checks
    unhealthy_threshold=3,        # Failures before unhealthy
    recovery_threshold=2          # Successes before recovered
)
```

### Resource Monitor Configuration

```python
ResourceThresholds(
    cpu_percent=80.0,             # CPU usage threshold
    memory_percent=85.0,          # Memory usage threshold
    disk_percent=90.0,            # Disk usage threshold
    min_disk_free_gb=5.0          # Minimum free disk space
)
```

## Best Practices

1. **Health Checks**: Keep health checks fast (<1 second) and focused
2. **Recovery Actions**: Order by priority, with quick fixes first
3. **Thresholds**: Set conservative thresholds to avoid false positives
4. **Cleanup**: Run cleanup actions during low-traffic periods
5. **Monitoring**: Monitor the monitors - track their own health
6. **Testing**: Test recovery actions in staging before production
7. **Logging**: Log all state transitions and recovery attempts
8. **Alerts**: Set up alerts for repeated recovery failures

## Troubleshooting

### Health Monitor Not Triggering Recovery

- Check if health checks are actually failing
- Verify unhealthy_threshold is set appropriately
- Check if recovery actions are registered
- Review logs for exceptions in health checks

### Resource Monitor Not Cleaning Up

- Verify thresholds are being exceeded
- Check if cleanup actions are registered
- Ensure cleanup actions have proper permissions
- Review logs for cleanup action failures

### Data Consistency Issues Not Repaired

- Check if repair actions are registered for issue types
- Verify auto_repairable flag is set correctly
- Check repair action return values
- Review repair history for failure patterns

## Performance Considerations

- Health checks run in the monitoring loop - keep them fast
- Resource monitoring uses psutil - minimal overhead
- File integrity checking is I/O intensive - run during off-peak
- Cleanup operations can be disk-intensive - schedule appropriately

## Security Considerations

- Health check functions should not expose sensitive data
- Recovery actions should validate inputs
- File integrity manifest should be protected
- Cleanup operations should validate paths to prevent deletion of critical files
