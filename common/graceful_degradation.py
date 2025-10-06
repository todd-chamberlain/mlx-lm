"""
Graceful Degradation Implementation

Provides graceful degradation patterns for handling partial failures
and maintaining service availability under adverse conditions.
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, List, Union, Protocol
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ServiceHealth(Enum):
    """Service health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class DegradationLevel(Enum):
    """Degradation levels."""
    NONE = 0
    MINIMAL = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


@dataclass
class ServiceStatus:
    """Service status information."""
    name: str
    health: ServiceHealth = ServiceHealth.UNKNOWN
    last_check: float = field(default_factory=time.time)
    error_count: int = 0
    success_count: int = 0
    response_time: float = 0.0
    degradation_level: DegradationLevel = DegradationLevel.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def is_available(self) -> bool:
        """Check if service is available for requests."""
        return self.health in [ServiceHealth.HEALTHY, ServiceHealth.DEGRADED]


class FallbackStrategy(Protocol):
    """Protocol for fallback strategies."""
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute fallback strategy."""
        ...


class CacheFallback:
    """Fallback to cached data."""
    
    def __init__(self, cache: Dict[str, Any], ttl: float = 300.0):
        self.cache = cache
        self.ttl = ttl
        self.cache_timestamps: Dict[str, float] = {}
    
    def execute(self, key: str, default: Any = None) -> Any:
        """Get cached value if available and not expired."""
        if key in self.cache:
            timestamp = self.cache_timestamps.get(key, 0)
            if time.time() - timestamp < self.ttl:
                logger.debug(f"Using cached fallback for key: {key}")
                return self.cache[key]
        
        logger.debug(f"No valid cache for key: {key}, using default")
        return default
    
    def update_cache(self, key: str, value: Any):
        """Update cache with new value."""
        self.cache[key] = value
        self.cache_timestamps[key] = time.time()


class DefaultValueFallback:
    """Fallback to default values."""
    
    def __init__(self, default_values: Dict[str, Any]):
        self.default_values = default_values
    
    def execute(self, key: str, *args, **kwargs) -> Any:
        """Return default value for key."""
        default = self.default_values.get(key)
        logger.debug(f"Using default fallback for key: {key}, value: {default}")
        return default


class AlternativeServiceFallback:
    """Fallback to alternative service."""
    
    def __init__(self, alternative_service: Callable):
        self.alternative_service = alternative_service
    
    def execute(self, *args, **kwargs) -> Any:
        """Call alternative service."""
        logger.debug("Using alternative service fallback")
        return self.alternative_service(*args, **kwargs)


class ReducedFunctionalityFallback:
    """Fallback with reduced functionality."""
    
    def __init__(self, reduced_function: Callable):
        self.reduced_function = reduced_function
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute with reduced functionality."""
        logger.debug("Using reduced functionality fallback")
        return self.reduced_function(*args, **kwargs)


@dataclass
class DegradationConfig:
    """Configuration for graceful degradation."""
    health_check_interval: float = 30.0
    error_threshold: int = 5
    success_threshold: int = 3
    response_time_threshold: float = 5.0
    degradation_timeout: float = 300.0
    recovery_timeout: float = 60.0
    enable_fallbacks: bool = True
    enable_circuit_breaker: bool = True


class GracefulDegradationManager:
    """Manages graceful degradation for services."""
    
    def __init__(self, config: DegradationConfig):
        self.config = config
        self.services: Dict[str, ServiceStatus] = {}
        self.fallback_strategies: Dict[str, List[FallbackStrategy]] = {}
        self.degradation_callbacks: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._monitoring_thread: Optional[threading.Thread] = None
        self._shutdown = False
        
        # Start health monitoring
        self._start_monitoring()
    
    def register_service(
        self,
        name: str,
        health_check: Optional[Callable] = None,
        fallback_strategies: Optional[List[FallbackStrategy]] = None
    ):
        """
        Register a service for degradation management.
        
        Args:
            name: Service name
            health_check: Function to check service health
            fallback_strategies: List of fallback strategies
        """
        with self._lock:
            self.services[name] = ServiceStatus(name=name)
            
            if fallback_strategies:
                self.fallback_strategies[name] = fallback_strategies
            
            if health_check:
                self.degradation_callbacks[name] = [health_check]
        
        logger.info(f"Registered service for degradation management: {name}")
    
    def add_fallback_strategy(self, service_name: str, strategy: FallbackStrategy):
        """Add fallback strategy for service."""
        with self._lock:
            if service_name not in self.fallback_strategies:
                self.fallback_strategies[service_name] = []
            self.fallback_strategies[service_name].append(strategy)
        
        logger.debug(f"Added fallback strategy for service: {service_name}")
    
    def call_with_degradation(
        self,
        service_name: str,
        func: Callable,
        *args,
        fallback_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Call function with graceful degradation support.
        
        Args:
            service_name: Name of the service
            func: Function to call
            *args: Function arguments
            fallback_key: Key for fallback strategies
            **kwargs: Function keyword arguments
            
        Returns:
            Function result or fallback result
        """
        service_status = self.get_service_status(service_name)
        
        # Check if service is available
        if not service_status.is_available:
            logger.warning(f"Service {service_name} is not available, using fallback")
            return self._execute_fallback(service_name, fallback_key, *args, **kwargs)
        
        # Try to execute the function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(service_name, execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._record_failure(service_name, execution_time, e)
            
            # Try fallback
            logger.warning(f"Service {service_name} failed: {e}, using fallback")
            return self._execute_fallback(service_name, fallback_key, *args, **kwargs)
    
    async def async_call_with_degradation(
        self,
        service_name: str,
        func: Callable,
        *args,
        fallback_key: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Async version of call_with_degradation."""
        service_status = self.get_service_status(service_name)
        
        # Check if service is available
        if not service_status.is_available:
            logger.warning(f"Service {service_name} is not available, using fallback")
            return await self._async_execute_fallback(service_name, fallback_key, *args, **kwargs)
        
        # Try to execute the function
        start_time = time.time()
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Record success
            execution_time = time.time() - start_time
            self._record_success(service_name, execution_time)
            
            return result
            
        except Exception as e:
            # Record failure
            execution_time = time.time() - start_time
            self._record_failure(service_name, execution_time, e)
            
            # Try fallback
            logger.warning(f"Service {service_name} failed: {e}, using fallback")
            return await self._async_execute_fallback(service_name, fallback_key, *args, **kwargs)
    
    def _execute_fallback(self, service_name: str, fallback_key: Optional[str], *args, **kwargs) -> Any:
        """Execute fallback strategies."""
        if not self.config.enable_fallbacks:
            raise RuntimeError(f"Service {service_name} unavailable and fallbacks disabled")
        
        strategies = self.fallback_strategies.get(service_name, [])
        
        for strategy in strategies:
            try:
                if fallback_key:
                    return strategy.execute(fallback_key, *args, **kwargs)
                else:
                    return strategy.execute(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback strategy failed for {service_name}: {e}")
                continue
        
        # No fallback strategies worked
        raise RuntimeError(f"All fallback strategies failed for service: {service_name}")
    
    async def _async_execute_fallback(self, service_name: str, fallback_key: Optional[str], *args, **kwargs) -> Any:
        """Execute fallback strategies (async version)."""
        if not self.config.enable_fallbacks:
            raise RuntimeError(f"Service {service_name} unavailable and fallbacks disabled")
        
        strategies = self.fallback_strategies.get(service_name, [])
        
        for strategy in strategies:
            try:
                if fallback_key:
                    result = strategy.execute(fallback_key, *args, **kwargs)
                else:
                    result = strategy.execute(*args, **kwargs)
                
                # Handle async results
                if asyncio.iscoroutine(result):
                    return await result
                else:
                    return result
                    
            except Exception as e:
                logger.warning(f"Async fallback strategy failed for {service_name}: {e}")
                continue
        
        # No fallback strategies worked
        raise RuntimeError(f"All async fallback strategies failed for service: {service_name}")
    
    def _record_success(self, service_name: str, execution_time: float):
        """Record successful service call."""
        with self._lock:
            if service_name in self.services:
                status = self.services[service_name]
                status.success_count += 1
                status.response_time = execution_time
                status.last_check = time.time()
                
                # Update health based on success
                if status.success_count >= self.config.success_threshold:
                    if status.health == ServiceHealth.UNHEALTHY:
                        status.health = ServiceHealth.DEGRADED
                    elif status.health == ServiceHealth.DEGRADED:
                        status.health = ServiceHealth.HEALTHY
                    
                    # Reset error count on recovery
                    status.error_count = 0
                    status.degradation_level = DegradationLevel.NONE
    
    def _record_failure(self, service_name: str, execution_time: float, exception: Exception):
        """Record failed service call."""
        with self._lock:
            if service_name in self.services:
                status = self.services[service_name]
                status.error_count += 1
                status.response_time = execution_time
                status.last_check = time.time()
                
                # Update health based on failures
                if status.error_count >= self.config.error_threshold:
                    if status.health == ServiceHealth.HEALTHY:
                        status.health = ServiceHealth.DEGRADED
                        status.degradation_level = DegradationLevel.MINIMAL
                    elif status.health == ServiceHealth.DEGRADED:
                        status.health = ServiceHealth.UNHEALTHY
                        status.degradation_level = DegradationLevel.SEVERE
                
                # Store error information
                status.metadata["last_error"] = str(exception)
                status.metadata["last_error_time"] = time.time()
    
    def get_service_status(self, service_name: str) -> ServiceStatus:
        """Get current service status."""
        with self._lock:
            return self.services.get(service_name, ServiceStatus(name=service_name))
    
    def get_all_service_status(self) -> Dict[str, ServiceStatus]:
        """Get status of all services."""
        with self._lock:
            return self.services.copy()
    
    def force_service_health(self, service_name: str, health: ServiceHealth):
        """Force service health state."""
        with self._lock:
            if service_name in self.services:
                self.services[service_name].health = health
                logger.info(f"Forced service {service_name} health to {health.value}")
    
    def _start_monitoring(self):
        """Start background health monitoring."""
        def monitoring_loop():
            while not self._shutdown:
                try:
                    self._check_service_health()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Error in health monitoring: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_thread.start()
        logger.info("Started graceful degradation monitoring")
    
    def _check_service_health(self):
        """Check health of all registered services."""
        current_time = time.time()
        
        with self._lock:
            services_to_check = list(self.services.keys())
        
        for service_name in services_to_check:
            callbacks = self.degradation_callbacks.get(service_name, [])
            
            for callback in callbacks:
                try:
                    # Execute health check
                    is_healthy = callback()
                    
                    with self._lock:
                        status = self.services[service_name]
                        status.last_check = current_time
                        
                        if is_healthy:
                            self._record_success(service_name, 0.0)
                        else:
                            self._record_failure(service_name, 0.0, Exception("Health check failed"))
                
                except Exception as e:
                    logger.warning(f"Health check failed for {service_name}: {e}")
                    with self._lock:
                        if service_name in self.services:
                            self._record_failure(service_name, 0.0, e)
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get overall system health summary."""
        with self._lock:
            total_services = len(self.services)
            healthy_services = sum(1 for s in self.services.values() if s.health == ServiceHealth.HEALTHY)
            degraded_services = sum(1 for s in self.services.values() if s.health == ServiceHealth.DEGRADED)
            unhealthy_services = sum(1 for s in self.services.values() if s.health == ServiceHealth.UNHEALTHY)
            
            overall_health = ServiceHealth.HEALTHY
            if unhealthy_services > 0:
                overall_health = ServiceHealth.UNHEALTHY
            elif degraded_services > 0:
                overall_health = ServiceHealth.DEGRADED
            
            return {
                "overall_health": overall_health.value,
                "total_services": total_services,
                "healthy_services": healthy_services,
                "degraded_services": degraded_services,
                "unhealthy_services": unhealthy_services,
                "health_percentage": (healthy_services / total_services * 100) if total_services > 0 else 0,
                "services": {
                    name: {
                        "health": status.health.value,
                        "success_rate": status.success_rate,
                        "response_time": status.response_time,
                        "degradation_level": status.degradation_level.value
                    }
                    for name, status in self.services.items()
                }
            }
    
    def shutdown(self):
        """Shutdown graceful degradation manager."""
        self._shutdown = True
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Graceful degradation manager shutdown")


# Global degradation manager
_global_degradation_manager: Optional[GracefulDegradationManager] = None


def get_degradation_manager(config: Optional[DegradationConfig] = None) -> GracefulDegradationManager:
    """Get global degradation manager."""
    global _global_degradation_manager
    
    if _global_degradation_manager is None:
        if config is None:
            config = DegradationConfig()
        _global_degradation_manager = GracefulDegradationManager(config)
    
    return _global_degradation_manager


def with_degradation(service_name: str, fallback_key: Optional[str] = None):
    """
    Decorator for adding graceful degradation to functions.
    
    Args:
        service_name: Name of the service
        fallback_key: Key for fallback strategies
        
    Example:
        @with_degradation("external_api", "user_data")
        def get_user_data(user_id):
            # Function that might fail
            pass
    """
    def decorator(func):
        degradation_manager = get_degradation_manager()
        
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                return await degradation_manager.async_call_with_degradation(
                    service_name, func, *args, fallback_key=fallback_key, **kwargs
                )
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                return degradation_manager.call_with_degradation(
                    service_name, func, *args, fallback_key=fallback_key, **kwargs
                )
            return sync_wrapper
    
    return decorator