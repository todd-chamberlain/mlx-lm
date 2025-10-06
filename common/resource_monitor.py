"""
System Resource Monitoring and Cleanup

Monitors system resources (CPU, memory, disk) and performs cleanup when needed.
"""

import asyncio
import logging
import psutil
import shutil
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


logger = logging.getLogger(__name__)


@dataclass
class ResourceThresholds:
    """Resource usage thresholds."""
    cpu_percent: float = 80.0
    memory_percent: float = 85.0
    disk_percent: float = 90.0
    min_disk_free_gb: float = 5.0


@dataclass
class ResourceUsage:
    """Current resource usage."""
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    disk_percent: float
    disk_free_gb: float
    timestamp: datetime


class ResourceMonitor:
    """
    Monitors system resources and triggers cleanup actions.
    """
    
    def __init__(
        self,
        thresholds: Optional[ResourceThresholds] = None,
        check_interval: float = 60.0
    ):
        self.thresholds = thresholds or ResourceThresholds()
        self.check_interval = check_interval
        
        self.cleanup_actions: List[Callable] = []
        self.usage_history: List[ResourceUsage] = []
        self.max_history = 100
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None

    
    def add_cleanup_action(self, action: Callable) -> None:
        """Add a cleanup action to be triggered when resources are low."""
        self.cleanup_actions.append(action)
        logger.info(f"Added cleanup action: {action.__name__}")
    
    def get_current_usage(self) -> ResourceUsage:
        """Get current resource usage."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024 ** 3)
        memory_available_gb = memory.available / (1024 ** 3)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = disk.percent
        disk_free_gb = disk.free / (1024 ** 3)
        
        return ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_available_gb=memory_available_gb,
            disk_percent=disk_percent,
            disk_free_gb=disk_free_gb,
            timestamp=datetime.now()
        )
    
    def check_thresholds(self, usage: ResourceUsage) -> Dict[str, bool]:
        """Check if any thresholds are exceeded."""
        exceeded = {}
        
        if usage.cpu_percent > self.thresholds.cpu_percent:
            exceeded['cpu'] = True
            logger.warning(
                f"CPU usage ({usage.cpu_percent:.1f}%) exceeds threshold "
                f"({self.thresholds.cpu_percent}%)"
            )
        
        if usage.memory_percent > self.thresholds.memory_percent:
            exceeded['memory'] = True
            logger.warning(
                f"Memory usage ({usage.memory_percent:.1f}%) exceeds threshold "
                f"({self.thresholds.memory_percent}%)"
            )
        
        if usage.disk_percent > self.thresholds.disk_percent:
            exceeded['disk'] = True
            logger.warning(
                f"Disk usage ({usage.disk_percent:.1f}%) exceeds threshold "
                f"({self.thresholds.disk_percent}%)"
            )
        
        if usage.disk_free_gb < self.thresholds.min_disk_free_gb:
            exceeded['disk_free'] = True
            logger.warning(
                f"Free disk space ({usage.disk_free_gb:.1f}GB) below minimum "
                f"({self.thresholds.min_disk_free_gb}GB)"
            )
        
        return exceeded

    
    async def trigger_cleanup(self, exceeded: Dict[str, bool]) -> None:
        """Trigger cleanup actions when thresholds are exceeded."""
        if not self.cleanup_actions:
            logger.warning("No cleanup actions configured")
            return
        
        logger.info(f"Triggering cleanup actions for: {list(exceeded.keys())}")
        
        for action in self.cleanup_actions:
            try:
                if asyncio.iscoroutinefunction(action):
                    await action(exceeded)
                else:
                    action(exceeded)
                logger.info(f"Cleanup action {action.__name__} completed")
            except Exception as e:
                logger.error(f"Cleanup action {action.__name__} failed: {e}")
    
    async def monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting resource monitoring")
        
        while self._monitoring:
            try:
                usage = self.get_current_usage()
                
                # Store in history
                self.usage_history.append(usage)
                if len(self.usage_history) > self.max_history:
                    self.usage_history = self.usage_history[-self.max_history:]
                
                # Check thresholds
                exceeded = self.check_thresholds(usage)
                
                if exceeded:
                    await self.trigger_cleanup(exceeded)
                
                logger.debug(
                    f"Resource usage - CPU: {usage.cpu_percent:.1f}%, "
                    f"Memory: {usage.memory_percent:.1f}%, "
                    f"Disk: {usage.disk_percent:.1f}%"
                )
                
            except Exception as e:
                logger.error(f"Error in resource monitoring loop: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            logger.warning("Resource monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self.monitor_loop())
        logger.info("Started resource monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped resource monitoring")
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        if not self.usage_history:
            return {"status": "no data"}
        
        current = self.usage_history[-1]
        
        return {
            "cpu_percent": current.cpu_percent,
            "memory_percent": current.memory_percent,
            "memory_used_gb": current.memory_used_gb,
            "memory_available_gb": current.memory_available_gb,
            "disk_percent": current.disk_percent,
            "disk_free_gb": current.disk_free_gb,
            "timestamp": current.timestamp.isoformat(),
            "history_size": len(self.usage_history)
        }


class CleanupManager:
    """
    Manages cleanup operations for various system resources.
    """
    
    def __init__(self, log_dir: Optional[Path] = None, cache_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs")
        self.cache_dir = cache_dir or Path("cache")
    
    def cleanup_old_logs(self, max_age_days: int = 7) -> int:
        """Clean up log files older than specified days."""
        if not self.log_dir.exists():
            return 0
        
        count = 0
        cutoff_time = datetime.now().timestamp() - (max_age_days * 86400)
        
        for log_file in self.log_dir.rglob("*.log*"):
            try:
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    count += 1
                    logger.info(f"Deleted old log file: {log_file}")
            except Exception as e:
                logger.error(f"Failed to delete {log_file}: {e}")
        
        return count
    
    def cleanup_cache(self) -> int:
        """Clean up cache directory."""
        if not self.cache_dir.exists():
            return 0
        
        count = 0
        for cache_file in self.cache_dir.rglob("*"):
            if cache_file.is_file():
                try:
                    cache_file.unlink()
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info(f"Cleaned up {count} cache files")
        return count
    
    def cleanup_temp_files(self, temp_dir: Optional[Path] = None) -> int:
        """Clean up temporary files."""
        temp_dir = temp_dir or Path("/tmp")
        count = 0
        
        # Clean up old temp files (older than 1 day)
        cutoff_time = datetime.now().timestamp() - 86400
        
        try:
            for temp_file in temp_dir.glob("mlx_*"):
                if temp_file.stat().st_mtime < cutoff_time:
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                        elif temp_file.is_dir():
                            shutil.rmtree(temp_file)
                        count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete temp file {temp_file}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning temp files: {e}")
        
        logger.info(f"Cleaned up {count} temp files")
        return count
