"""
Worker pool management for the coordinator service.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Set, Callable, Any
from collections import defaultdict
import uuid

from .models import WorkerInfo, WorkerStatus
from .worker_selector import LoadBalancer, SelectionConfig
from .health_monitor import HealthMonitor, HealthThresholds, ClusterHealth
from proto.errors import MLXError, MLXErrorCode


logger = logging.getLogger(__name__)


class WorkerPool:
    """Thread-safe worker pool management."""
    
    def __init__(
        self,
        heartbeat_timeout: float = 30.0,
        cleanup_interval: float = 60.0,
        max_workers: int = 100,
        circuit_breaker_failure_threshold: int = 10,  # Increased from 5 to 10
        circuit_breaker_recovery_timeout: float = 15.0,  # Reduced from 60 to 15 seconds
        selection_config: Optional[SelectionConfig] = None,
        health_thresholds: Optional[HealthThresholds] = None,
        health_check_interval: float = 10.0
    ):
        self.heartbeat_timeout = heartbeat_timeout
        self.cleanup_interval = cleanup_interval
        self.max_workers = max_workers
        self.circuit_breaker_failure_threshold = circuit_breaker_failure_threshold
        self.circuit_breaker_recovery_timeout = circuit_breaker_recovery_timeout
        
        # Worker storage
        self._workers: Dict[str, WorkerInfo] = {}
        self._workers_by_model: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # Load balancer for worker selection
        self.load_balancer = LoadBalancer(selection_config)
        
        # Health monitor
        self.health_monitor = HealthMonitor(
            thresholds=health_thresholds,
            check_interval=health_check_interval
        )
        
        # Event callbacks
        self._worker_registered_callbacks: List[Callable[[WorkerInfo], None]] = []
        self._worker_deregistered_callbacks: List[Callable[[str], None]] = []
        self._worker_status_changed_callbacks: List[Callable[[WorkerInfo, WorkerStatus], None]] = []
        
        logger.info(f"WorkerPool initialized with {max_workers} max workers")
    
    async def start(self) -> None:
        """Start background tasks."""
        logger.info("Starting worker pool background tasks")
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        await self.health_monitor.start()
    
    async def stop(self) -> None:
        """Stop background tasks and cleanup."""
        logger.info("Stopping worker pool")
        self._shutdown_event.set()
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Stop health monitor
        await self.health_monitor.stop()
        
        # Clear all workers
        with self._lock:
            self._workers.clear()
            self._workers_by_model.clear()
        
        logger.info("Worker pool stopped")
    
    def register_worker(self, registration) -> Dict[str, Any]:
        """Register a new worker or update existing one."""
        worker_id = registration.worker_id
        
        if not worker_id:
            raise MLXError(
                message="Worker ID is required",
                error_code=MLXErrorCode.INVALID_REQUEST
            )
        
        if not registration.address:
            raise MLXError(
                message="Worker address is required",
                error_code=MLXErrorCode.INVALID_REQUEST
            )
        
        if registration.max_concurrent <= 0:
            raise MLXError(
                message="max_concurrent must be positive",
                error_code=MLXErrorCode.INVALID_REQUEST
            )
        
        with self._lock:
            # Check worker limit
            if worker_id not in self._workers and len(self._workers) >= self.max_workers:
                raise MLXError(
                    message=f"Maximum number of workers ({self.max_workers}) reached",
                    error_code=MLXErrorCode.RESOURCE_EXHAUSTED
                )
            
            # Check for duplicate addresses (different worker IDs)
            for existing_id, existing_worker in self._workers.items():
                if (existing_id != worker_id and 
                    existing_worker.address == registration.address):
                    raise MLXError(
                        message=f"Address {registration.address} already registered by worker {existing_id}",
                        error_code=MLXErrorCode.ALREADY_EXISTS
                    )
            
            is_new_worker = worker_id not in self._workers
            
            if is_new_worker:
                # Create new worker
                worker = WorkerInfo.from_registration(registration)
                self._workers[worker_id] = worker
                
                # Update model mappings
                for model in worker.supported_models:
                    self._workers_by_model[model].add(worker_id)
                
                logger.info(f"Registered new worker: {worker_id} at {worker.address}")
                
                # Trigger callbacks
                for callback in self._worker_registered_callbacks:
                    try:
                        callback(worker)
                    except Exception as e:
                        logger.error(f"Error in worker registered callback: {e}")
            
            else:
                # Update existing worker
                worker = self._workers[worker_id]
                old_models = set(worker.supported_models)
                
                # Update worker info
                worker.address = registration.address
                worker.device_capabilities = worker.device_capabilities.from_proto(registration.device_info)
                worker.max_concurrent = registration.max_concurrent
                worker.supported_models = list(registration.supported_models)
                worker.last_heartbeat = time.time()
                worker.status = WorkerStatus.HEALTHY
                worker.consecutive_failures = 0
                worker.circuit_breaker_open = False
                
                # Update model mappings
                new_models = set(registration.supported_models)
                
                # Remove from old models
                for model in old_models - new_models:
                    self._workers_by_model[model].discard(worker_id)
                
                # Add to new models
                for model in new_models - old_models:
                    self._workers_by_model[model].add(worker_id)
                
                logger.info(f"Updated existing worker: {worker_id} at {worker.address}")
            
            return {
                'success': True,
                'message': f"Worker {worker_id} {'registered' if is_new_worker else 'updated'} successfully",
                'coordinator_id': self._generate_coordinator_id(),
                'is_new': is_new_worker
            }
    
    def deregister_worker(self, worker_id: str, reason: str = "manual") -> bool:
        """Deregister a worker."""
        with self._lock:
            if worker_id not in self._workers:
                logger.warning(f"Attempted to deregister unknown worker: {worker_id}")
                return False
            
            worker = self._workers[worker_id]
            
            # Remove from model mappings
            for model in worker.supported_models:
                self._workers_by_model[model].discard(worker_id)
            
            # Remove worker
            del self._workers[worker_id]
            
            logger.info(f"Deregistered worker {worker_id} (reason: {reason})")
            
            # Trigger callbacks
            for callback in self._worker_deregistered_callbacks:
                try:
                    callback(worker_id)
                except Exception as e:
                    logger.error(f"Error in worker deregistered callback: {e}")
            
            return True
    
    def update_worker_heartbeat(self, worker_id: str, heartbeat_request) -> Dict[str, Any]:
        """Update worker heartbeat and status."""
        with self._lock:
            if worker_id not in self._workers:
                raise MLXError(
                    message=f"Worker {worker_id} not registered",
                    error_code=MLXErrorCode.WORKER_NOT_FOUND
                )
            
            worker = self._workers[worker_id]
            old_status = worker.status
            
            # Update heartbeat
            worker.update_heartbeat(heartbeat_request)
            
            # Update circuit breaker
            worker.update_circuit_breaker(
                self.circuit_breaker_failure_threshold,
                self.circuit_breaker_recovery_timeout
            )
            
            # Check for status changes
            if worker.status != old_status:
                logger.info(f"Worker {worker_id} status changed: {old_status.value} -> {worker.status.value}")
                
                # Trigger callbacks
                for callback in self._worker_status_changed_callbacks:
                    try:
                        callback(worker, old_status)
                    except Exception as e:
                        logger.error(f"Error in worker status changed callback: {e}")
            
            return {
                'acknowledged': True,
                'message': 'Heartbeat received',
                'heartbeat_interval': 10  # seconds
            }
    
    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get worker by ID."""
        with self._lock:
            return self._workers.get(worker_id)
    
    def get_all_workers(self) -> List[WorkerInfo]:
        """Get all workers."""
        with self._lock:
            return list(self._workers.values())
    
    def get_healthy_workers(self) -> List[WorkerInfo]:
        """Get all healthy workers."""
        with self._lock:
            return [worker for worker in self._workers.values() if worker.is_healthy(self.heartbeat_timeout)]
    
    def get_available_workers(self) -> List[WorkerInfo]:
        """Get workers available for new requests."""
        with self._lock:
            return [worker for worker in self._workers.values() if worker.is_available_for_request()]
    
    def get_workers_for_model(self, model: str) -> List[WorkerInfo]:
        """Get workers that support a specific model."""
        with self._lock:
            worker_ids = self._workers_by_model.get(model, set())
            logger.info(f"[ROUTING] _workers_by_model['{model}'] = {worker_ids}")
            logger.info(f"[ROUTING] All models in _workers_by_model: {list(self._workers_by_model.keys())}")
            return [self._workers[worker_id] for worker_id in worker_ids if worker_id in self._workers]
    
    def get_available_workers_for_model(self, model: str) -> List[WorkerInfo]:
        """Get available workers that support a specific model."""
        workers = self.get_workers_for_model(model)
        logger.info(f"[ROUTING] get_workers_for_model('{model}') returned {len(workers)} workers")
        available = [worker for worker in workers if worker.is_available_for_request()]
        logger.info(f"[ROUTING] {len(available)} of {len(workers)} workers are available for requests")
        return available
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self._lock:
            total_workers = len(self._workers)
            healthy_workers = len(self.get_healthy_workers())
            available_workers = len(self.get_available_workers())
            
            total_capacity = sum(worker.max_concurrent for worker in self._workers.values())
            current_load = sum(worker.current_load for worker in self._workers.values())
            
            # Model statistics
            model_stats = {}
            for model, worker_ids in self._workers_by_model.items():
                available_for_model = len([
                    worker_id for worker_id in worker_ids 
                    if worker_id in self._workers and self._workers[worker_id].is_available_for_request()
                ])
                model_stats[model] = {
                    'total_workers': len(worker_ids),
                    'available_workers': available_for_model
                }
            
            return {
                'total_workers': total_workers,
                'healthy_workers': healthy_workers,
                'available_workers': available_workers,
                'total_capacity': total_capacity,
                'current_load': current_load,
                'utilization': current_load / max(total_capacity, 1),
                'models': model_stats,
                'max_workers': self.max_workers
            }
    
    def record_request_start(self, worker_id: str) -> bool:
        """Record that a request has started on a worker."""
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            worker.record_request_start()
            return True
    
    def record_request_complete(self, worker_id: str, success: bool = True, latency_ms: float = 0.0) -> bool:
        """Record that a request has completed on a worker."""
        with self._lock:
            if worker_id not in self._workers:
                return False
            
            worker = self._workers[worker_id]
            worker.record_request_complete(success, latency_ms)
            
            # Update circuit breaker
            worker.update_circuit_breaker(
                self.circuit_breaker_failure_threshold,
                self.circuit_breaker_recovery_timeout
            )
            
            return True
    
    async def _cleanup_loop(self) -> None:
        """Background task to cleanup unhealthy workers and update health."""
        logger.info("Starting worker cleanup loop")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                # Update cluster health
                self.check_cluster_health()
                
                # Cleanup unhealthy workers
                await self._cleanup_unhealthy_workers()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
        
        logger.info("Worker cleanup loop stopped")
    
    async def _cleanup_unhealthy_workers(self) -> None:
        """Remove workers that haven't sent heartbeats."""
        current_time = time.time()
        workers_to_remove = []
        
        with self._lock:
            for worker_id, worker in self._workers.items():
                time_since_heartbeat = current_time - worker.last_heartbeat
                
                if time_since_heartbeat > self.heartbeat_timeout * 2:  # 2x timeout for removal
                    workers_to_remove.append((worker_id, time_since_heartbeat))
        
        # Remove unhealthy workers
        for worker_id, time_since_heartbeat in workers_to_remove:
            logger.warning(f"Removing unhealthy worker {worker_id} (no heartbeat for {time_since_heartbeat:.1f}s)")
            self.deregister_worker(worker_id, reason="heartbeat_timeout")
    
    def _generate_coordinator_id(self) -> str:
        """Generate a unique coordinator ID."""
        return f"coordinator-{uuid.uuid4().hex[:8]}"
    
    # Event callback management
    def add_worker_registered_callback(self, callback: Callable[[WorkerInfo], None]) -> None:
        """Add callback for worker registration events."""
        self._worker_registered_callbacks.append(callback)
    
    def add_worker_deregistered_callback(self, callback: Callable[[str], None]) -> None:
        """Add callback for worker deregistration events."""
        self._worker_deregistered_callbacks.append(callback)
    
    def add_worker_status_changed_callback(self, callback: Callable[[WorkerInfo, WorkerStatus], None]) -> None:
        """Add callback for worker status change events."""
        self._worker_status_changed_callbacks.append(callback)
    
    def remove_worker_registered_callback(self, callback: Callable[[WorkerInfo], None]) -> None:
        """Remove worker registration callback."""
        if callback in self._worker_registered_callbacks:
            self._worker_registered_callbacks.remove(callback)
    
    def remove_worker_deregistered_callback(self, callback: Callable[[str], None]) -> None:
        """Remove worker deregistration callback."""
        if callback in self._worker_deregistered_callbacks:
            self._worker_deregistered_callbacks.remove(callback)
    
    def remove_worker_status_changed_callback(self, callback: Callable[[WorkerInfo, WorkerStatus], None]) -> None:
        """Remove worker status change callback."""
        if callback in self._worker_status_changed_callbacks:
            self._worker_status_changed_callbacks.remove(callback)
    
    # Worker selection methods
    def select_worker_for_request(
        self,
        model: Optional[str] = None,
        request_requirements: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkerInfo]:
        """Select the best available worker for a request."""
        available_workers = self.get_available_workers()
        
        if not available_workers:
            logger.warning("No available workers for request")
            return None
        
        selected_worker = self.load_balancer.select_worker_for_request(
            available_workers=available_workers,
            model=model,
            request_requirements=request_requirements,
            request_context=request_context
        )
        
        if selected_worker:
            logger.debug(f"Selected worker {selected_worker.worker_id} for request")
        else:
            logger.warning("Failed to select worker for request")
        
        return selected_worker
    
    def select_worker_with_fallback(
        self,
        model: Optional[str] = None,
        request_requirements: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None,
        exclude_workers: Optional[Set[str]] = None
    ) -> Optional[WorkerInfo]:
        """Select worker with fallback options for retries."""
        available_workers = self.get_available_workers()
        
        # Exclude workers that have already been tried
        if exclude_workers:
            available_workers = [
                w for w in available_workers 
                if w.worker_id not in exclude_workers
            ]
        
        if not available_workers:
            logger.warning("No available workers for fallback selection")
            return None
        
        # Try with requirements first
        selected_worker = self.load_balancer.select_worker_for_request(
            available_workers=available_workers,
            model=model,
            request_requirements=request_requirements,
            request_context=request_context
        )
        
        # If no worker found and we have requirements, try without requirements
        if not selected_worker and request_requirements:
            logger.debug("Retrying worker selection without requirements")
            selected_worker = self.load_balancer.select_worker_for_request(
                available_workers=available_workers,
                model=model,
                request_requirements=None,
                request_context=request_context
            )
        
        return selected_worker
    
    def get_worker_selection_stats(self) -> Dict[str, Any]:
        """Get worker selection statistics."""
        return self.load_balancer.get_stats()
    
    def update_selection_config(self, config: SelectionConfig) -> None:
        """Update worker selection configuration."""
        self.load_balancer = LoadBalancer(config)
        logger.info(f"Updated worker selection config: {config.strategy.value}")
    
    # Health monitoring methods
    def check_cluster_health(self) -> ClusterHealth:
        """Check and update cluster health status."""
        with self._lock:
            workers = list(self._workers.values())
        
        return self.health_monitor.check_cluster_health(workers)
    
    def get_cluster_health(self) -> Optional[ClusterHealth]:
        """Get current cluster health."""
        return self.health_monitor.get_cluster_health()
    
    def get_worker_health(self, worker_id: str):
        """Get health info for specific worker."""
        return self.health_monitor.get_worker_health(worker_id)
    
    def get_all_worker_health(self):
        """Get health info for all workers."""
        return self.health_monitor.get_all_worker_health()
    
    def get_unhealthy_workers(self) -> List[str]:
        """Get list of unhealthy worker IDs."""
        return self.health_monitor.get_unhealthy_workers()
    
    def is_worker_healthy(self, worker_id: str) -> bool:
        """Check if specific worker is healthy."""
        return self.health_monitor.is_worker_healthy(worker_id)
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get comprehensive health statistics."""
        return self.health_monitor.get_monitoring_stats()
    
    def add_health_callback(self, callback_type: str, callback: Callable) -> None:
        """Add health monitoring callback."""
        if callback_type == 'health_changed':
            self.health_monitor.add_health_changed_callback(callback)
        elif callback_type == 'worker_unhealthy':
            self.health_monitor.add_worker_unhealthy_callback(callback)
        elif callback_type == 'worker_recovered':
            self.health_monitor.add_worker_recovered_callback(callback)
        elif callback_type == 'critical_alert':
            self.health_monitor.add_critical_alert_callback(callback)
        else:
            raise ValueError(f"Unknown callback type: {callback_type}")