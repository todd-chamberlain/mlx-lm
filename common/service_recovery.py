"""
Service Recovery and Health Management System

Provides automatic service restart, health monitoring, and recovery mechanisms
for the MLX distributed inference system.
"""

import asyncio
import logging
import time
import psutil
import signal
import subprocess
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading


logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    RECOVERING = "recovering"
    FAILED = "failed"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    is_healthy: bool
    state: ServiceState
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryAction:
    """Recovery action to be executed."""
    name: str
    action: Callable
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0


class ServiceHealthMonitor:
    """
    Monitors service health and triggers recovery actions.
    """

    
    def __init__(
        self,
        service_name: str,
        check_interval: float = 30.0,
        unhealthy_threshold: int = 3,
        recovery_threshold: int = 2
    ):
        self.service_name = service_name
        self.check_interval = check_interval
        self.unhealthy_threshold = unhealthy_threshold
        self.recovery_threshold = recovery_threshold
        
        self.current_state = ServiceState.HEALTHY
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self.last_check_time: Optional[datetime] = None
        self.health_history: List[HealthCheckResult] = []
        self.max_history = 100
        
        self.health_checks: List[Callable] = []
        self.recovery_actions: List[RecoveryAction] = []
        
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    def add_health_check(self, check_func: Callable) -> None:
        """Add a health check function."""
        self.health_checks.append(check_func)
        logger.info(f"Added health check: {check_func.__name__}")
    
    def add_recovery_action(
        self,
        name: str,
        action: Callable,
        priority: int = 0,
        max_retries: int = 3
    ) -> None:
        """Add a recovery action."""
        recovery_action = RecoveryAction(
            name=name,
            action=action,
            priority=priority,
            max_retries=max_retries
        )
        self.recovery_actions.append(recovery_action)
        # Sort by priority (higher priority first)
        self.recovery_actions.sort(key=lambda x: x.priority, reverse=True)
        logger.info(f"Added recovery action: {name} (priority: {priority})")

    
    async def perform_health_check(self) -> HealthCheckResult:
        """Perform all health checks and aggregate results."""
        if not self.health_checks:
            return HealthCheckResult(
                is_healthy=True,
                state=ServiceState.HEALTHY,
                message="No health checks configured"
            )
        
        failed_checks = []
        all_metrics = {}
        
        for check_func in self.health_checks:
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                if isinstance(result, HealthCheckResult):
                    if not result.is_healthy:
                        failed_checks.append(result.message)
                    all_metrics.update(result.metrics)
                elif not result:
                    failed_checks.append(f"{check_func.__name__} failed")
                    
            except Exception as e:
                logger.error(f"Health check {check_func.__name__} raised exception: {e}")
                failed_checks.append(f"{check_func.__name__}: {str(e)}")
        
        is_healthy = len(failed_checks) == 0
        
        if is_healthy:
            state = ServiceState.HEALTHY
            message = "All health checks passed"
        else:
            state = ServiceState.UNHEALTHY
            message = f"Failed checks: {', '.join(failed_checks)}"
        
        return HealthCheckResult(
            is_healthy=is_healthy,
            state=state,
            message=message,
            metrics=all_metrics
        )

    
    async def update_state(self, check_result: HealthCheckResult) -> None:
        """Update service state based on health check result."""
        async with self._lock:
            self.last_check_time = datetime.now()
            self.health_history.append(check_result)
            
            # Trim history
            if len(self.health_history) > self.max_history:
                self.health_history = self.health_history[-self.max_history:]
            
            if check_result.is_healthy:
                self.consecutive_failures = 0
                self.consecutive_successes += 1
                
                # Recover from unhealthy state
                if self.current_state in [ServiceState.UNHEALTHY, ServiceState.RECOVERING]:
                    if self.consecutive_successes >= self.recovery_threshold:
                        logger.info(f"{self.service_name}: Recovered to healthy state")
                        self.current_state = ServiceState.HEALTHY
                        self.consecutive_successes = 0
                    else:
                        self.current_state = ServiceState.RECOVERING
            else:
                self.consecutive_successes = 0
                self.consecutive_failures += 1
                
                # Transition to unhealthy state
                if self.consecutive_failures >= self.unhealthy_threshold:
                    if self.current_state != ServiceState.UNHEALTHY:
                        logger.warning(
                            f"{self.service_name}: Transitioned to unhealthy state "
                            f"after {self.consecutive_failures} failures"
                        )
                        self.current_state = ServiceState.UNHEALTHY
                        await self.trigger_recovery()
                else:
                    self.current_state = ServiceState.DEGRADED

    
    async def trigger_recovery(self) -> bool:
        """Trigger recovery actions."""
        if not self.recovery_actions:
            logger.warning(f"{self.service_name}: No recovery actions configured")
            return False
        
        logger.info(f"{self.service_name}: Triggering recovery actions")
        
        for action in self.recovery_actions:
            if action.retry_count >= action.max_retries:
                logger.error(
                    f"Recovery action {action.name} exceeded max retries "
                    f"({action.max_retries})"
                )
                continue
            
            try:
                logger.info(f"Executing recovery action: {action.name}")
                
                if asyncio.iscoroutinefunction(action.action):
                    result = await action.action()
                else:
                    result = action.action()
                
                if result:
                    logger.info(f"Recovery action {action.name} succeeded")
                    action.retry_count = 0
                    return True
                else:
                    action.retry_count += 1
                    logger.warning(
                        f"Recovery action {action.name} failed "
                        f"(attempt {action.retry_count}/{action.max_retries})"
                    )
                    
            except Exception as e:
                action.retry_count += 1
                logger.error(
                    f"Recovery action {action.name} raised exception: {e} "
                    f"(attempt {action.retry_count}/{action.max_retries})"
                )
        
        return False

    
    async def monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info(f"Starting health monitoring for {self.service_name}")
        
        while self._monitoring:
            try:
                check_result = await self.perform_health_check()
                await self.update_state(check_result)
                
                logger.debug(
                    f"{self.service_name} health check: {check_result.state.value} - "
                    f"{check_result.message}"
                )
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            await asyncio.sleep(self.check_interval)
    
    async def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._monitoring:
            logger.warning(f"{self.service_name}: Monitoring already started")
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self.monitor_loop())
        logger.info(f"Started monitoring for {self.service_name}")
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self._monitoring:
            return
        
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped monitoring for {self.service_name}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        return {
            "service": self.service_name,
            "state": self.current_state.value,
            "consecutive_failures": self.consecutive_failures,
            "consecutive_successes": self.consecutive_successes,
            "last_check": self.last_check_time.isoformat() if self.last_check_time else None,
            "recent_checks": len(self.health_history)
        }
