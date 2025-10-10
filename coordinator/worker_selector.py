"""
Intelligent worker selection algorithms for the coordinator service.
"""

import logging
import random
import time
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

from .models import WorkerInfo, WorkerStatus


logger = logging.getLogger(__name__)


class SelectionStrategy(Enum):
    """Worker selection strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED_SCORE = "weighted_score"
    RANDOM = "random"
    FASTEST_RESPONSE = "fastest_response"


@dataclass
class SelectionWeights:
    """Weights for worker selection scoring."""
    load_weight: float = 0.5
    latency_weight: float = 0.3
    reliability_weight: float = 0.2
    
    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.load_weight + self.latency_weight + self.reliability_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")


@dataclass
class SelectionConfig:
    """Configuration for worker selection."""
    strategy: SelectionStrategy = SelectionStrategy.WEIGHTED_SCORE
    weights: SelectionWeights = None
    enable_fallback: bool = True
    fallback_strategy: SelectionStrategy = SelectionStrategy.LEAST_LOADED
    min_workers_for_strategy: int = 2
    prefer_local_workers: bool = False
    local_preference_bonus: float = 0.1
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = SelectionWeights()


class WorkerSelector:
    """Intelligent worker selection with multiple strategies."""
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        self.config = config or SelectionConfig()
        self._round_robin_index = 0
        self._selection_history: Dict[str, int] = {}
        
        logger.info(f"WorkerSelector initialized with strategy: {self.config.strategy.value}")
    
    def select_worker(
        self,
        available_workers: List[WorkerInfo],
        model: Optional[str] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkerInfo]:
        """Select the best worker for a request."""
        if not available_workers:
            logger.warning("No available workers for selection")
            return None
        
        # Filter workers by model if specified
        if model:
            model_workers = [w for w in available_workers if model in w.supported_models]
            if not model_workers:
                logger.warning(f"No workers support model: {model}")
                return None
            available_workers = model_workers
        
        # Apply health filtering
        healthy_workers = [w for w in available_workers if w.is_available_for_request()]
        if not healthy_workers:
            logger.warning("No healthy workers available")
            return None
        
        # Use fallback strategy if not enough workers
        strategy = self.config.strategy
        if (len(healthy_workers) < self.config.min_workers_for_strategy and 
            self.config.enable_fallback):
            strategy = self.config.fallback_strategy
            logger.debug(f"Using fallback strategy: {strategy.value}")
        
        # Select worker based on strategy
        try:
            selected_worker = self._select_by_strategy(healthy_workers, strategy, request_context)
            
            if selected_worker:
                # Update selection history
                self._selection_history[selected_worker.worker_id] = (
                    self._selection_history.get(selected_worker.worker_id, 0) + 1
                )
                
                logger.debug(f"Selected worker: {selected_worker.worker_id} using {strategy.value}")
                return selected_worker
            else:
                logger.warning(f"Strategy {strategy.value} failed to select worker")
                return None
                
        except Exception as e:
            logger.error(f"Error in worker selection: {e}")
            # Fallback to random selection
            return random.choice(healthy_workers) if healthy_workers else None
    
    def _select_by_strategy(
        self,
        workers: List[WorkerInfo],
        strategy: SelectionStrategy,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkerInfo]:
        """Select worker using specific strategy."""
        if strategy == SelectionStrategy.ROUND_ROBIN:
            return self._select_round_robin(workers)
        elif strategy == SelectionStrategy.LEAST_LOADED:
            return self._select_least_loaded(workers)
        elif strategy == SelectionStrategy.WEIGHTED_SCORE:
            return self._select_weighted_score(workers, request_context)
        elif strategy == SelectionStrategy.RANDOM:
            return self._select_random(workers)
        elif strategy == SelectionStrategy.FASTEST_RESPONSE:
            return self._select_fastest_response(workers)
        else:
            logger.error(f"Unknown selection strategy: {strategy}")
            return self._select_random(workers)
    
    def _select_round_robin(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """Round-robin selection."""
        if not workers:
            return None
        
        # Sort workers by ID for consistent ordering
        sorted_workers = sorted(workers, key=lambda w: w.worker_id)
        
        selected = sorted_workers[self._round_robin_index % len(sorted_workers)]
        self._round_robin_index += 1
        
        return selected
    
    def _select_least_loaded(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """Select worker with lowest current load."""
        if not workers:
            return None
        
        # Sort by load (ascending), then by utilization ratio
        def load_key(worker: WorkerInfo) -> tuple:
            utilization = worker.current_load / max(worker.max_concurrent, 1)
            return (worker.current_load, utilization, worker.worker_id)
        
        return min(workers, key=load_key)
    
    def _select_weighted_score(
        self,
        workers: List[WorkerInfo],
        request_context: Optional[Dict[str, Any]] = None
    ) -> WorkerInfo:
        """Select worker based on weighted scoring."""
        if not workers:
            return None
        
        scored_workers = []
        
        for worker in workers:
            score = self._calculate_worker_score(worker, request_context)
            scored_workers.append((worker, score))
        
        # Sort by score (ascending - lower is better)
        scored_workers.sort(key=lambda x: x[1])
        
        # Log top candidates for debugging
        if logger.isEnabledFor(logging.DEBUG):
            top_3 = scored_workers[:3]
            logger.debug(f"Top worker candidates: {[(w.worker_id, s) for w, s in top_3]}")
        
        return scored_workers[0][0]
    
    def _select_random(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """Random selection."""
        return random.choice(workers) if workers else None
    
    def _select_fastest_response(self, workers: List[WorkerInfo]) -> WorkerInfo:
        """Select worker with fastest average response time."""
        if not workers:
            return None
        
        # Sort by average latency (ascending)
        def latency_key(worker: WorkerInfo) -> tuple:
            # Prefer workers with actual latency data
            if worker.metrics.total_requests == 0:
                return (float('inf'), worker.worker_id)
            return (worker.metrics.avg_latency_ms, worker.worker_id)
        
        return min(workers, key=latency_key)
    
    def _calculate_worker_score(
        self,
        worker: WorkerInfo,
        request_context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate weighted score for worker selection."""
        weights = self.config.weights
        
        # Load factor (0.0 to 1.0+)
        load_factor = worker.current_load / max(worker.max_concurrent, 1)
        
        # Latency factor (normalized to 0-1 range, assuming max 2000ms)
        latency_factor = min(worker.metrics.avg_latency_ms / 2000.0, 1.0)
        
        # If no requests processed yet, use a neutral latency score
        if worker.metrics.total_requests == 0:
            latency_factor = 0.5
        
        # Reliability factor (failure rate)
        total_requests = max(worker.metrics.total_requests, 1)
        reliability_factor = worker.metrics.failed_requests / total_requests
        
        # Base weighted score
        score = (
            load_factor * weights.load_weight +
            latency_factor * weights.latency_weight +
            reliability_factor * weights.reliability_weight
        )
        
        # Apply local preference bonus if configured
        if self.config.prefer_local_workers and request_context:
            client_ip = request_context.get('client_ip')
            if client_ip and self._is_local_worker(worker, client_ip):
                score -= self.config.local_preference_bonus
        
        # Apply penalties for problematic workers
        score += self._calculate_penalties(worker)
        
        return max(0.0, score)  # Ensure non-negative score
    
    def _calculate_penalties(self, worker: WorkerInfo) -> float:
        """Calculate penalty factors for worker selection."""
        penalty = 0.0
        
        # Circuit breaker penalty
        if worker.circuit_breaker_open:
            penalty += 1.0  # Heavy penalty for circuit breaker
        
        # Recent failure penalty
        if worker.consecutive_failures > 0:
            penalty += worker.consecutive_failures * 0.1
        
        # Stale heartbeat penalty
        time_since_heartbeat = time.time() - worker.last_heartbeat
        if time_since_heartbeat > 15.0:  # More than 15 seconds
            penalty += (time_since_heartbeat - 15.0) / 30.0  # Gradual penalty
        
        return penalty
    
    def _is_local_worker(self, worker: WorkerInfo, client_ip: str) -> bool:
        """Check if worker is local to the client (simplified implementation)."""
        # This is a simplified implementation
        # In production, you might want more sophisticated network topology detection
        try:
            worker_ip = worker.address.split(':')[0]
            client_subnet = '.'.join(client_ip.split('.')[:3])
            worker_subnet = '.'.join(worker_ip.split('.')[:3])
            return client_subnet == worker_subnet
        except Exception:
            return False
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get worker selection statistics."""
        total_selections = sum(self._selection_history.values())
        
        return {
            'total_selections': total_selections,
            'worker_distribution': dict(self._selection_history),
            'round_robin_index': self._round_robin_index,
            'current_strategy': self.config.strategy.value,
            'fallback_enabled': self.config.enable_fallback
        }
    
    def reset_stats(self) -> None:
        """Reset selection statistics."""
        self._selection_history.clear()
        self._round_robin_index = 0
        logger.info("Worker selection statistics reset")


class WorkerFilter:
    """Advanced worker filtering capabilities."""
    
    @staticmethod
    def filter_by_model(workers: List[WorkerInfo], model: str) -> List[WorkerInfo]:
        """Filter workers that support a specific model."""
        logger.info(f"[FILTER] Filtering {len(workers)} workers for model '{model}'")
        for w in workers:
            logger.info(f"[FILTER] Worker {w.worker_id} supports: {w.supported_models}")
        filtered = [w for w in workers if model in w.supported_models]
        logger.info(f"[FILTER] {len(filtered)} workers support model '{model}'")
        return filtered
    
    @staticmethod
    def filter_by_capacity(workers: List[WorkerInfo], min_capacity: int = 1) -> List[WorkerInfo]:
        """Filter workers with available capacity."""
        return [w for w in workers if (w.max_concurrent - w.current_load) >= min_capacity]
    
    @staticmethod
    def filter_by_device_type(workers: List[WorkerInfo], device_model: str) -> List[WorkerInfo]:
        """Filter workers by device model."""
        return [w for w in workers if device_model.lower() in w.device_capabilities.model.lower()]
    
    @staticmethod
    def filter_by_memory(workers: List[WorkerInfo], min_memory_gb: int) -> List[WorkerInfo]:
        """Filter workers with minimum memory."""
        return [w for w in workers if w.device_capabilities.memory_gb >= min_memory_gb]
    
    @staticmethod
    def filter_by_performance(
        workers: List[WorkerInfo],
        min_tokens_per_second: float = 0.0,
        max_latency_ms: float = float('inf')
    ) -> List[WorkerInfo]:
        """Filter workers by performance metrics."""
        filtered = []
        for worker in workers:
            if worker.metrics.total_requests == 0:
                # Include workers with no history (benefit of doubt)
                filtered.append(worker)
                continue
            
            if (worker.metrics.tokens_per_second >= min_tokens_per_second and
                worker.metrics.avg_latency_ms <= max_latency_ms):
                filtered.append(worker)
        
        return filtered
    
    @staticmethod
    def filter_by_reliability(
        workers: List[WorkerInfo],
        max_failure_rate: float = 0.1,
        min_requests: int = 10
    ) -> List[WorkerInfo]:
        """Filter workers by reliability metrics."""
        filtered = []
        for worker in workers:
            if worker.metrics.total_requests < min_requests:
                # Include workers with insufficient history
                filtered.append(worker)
                continue
            
            failure_rate = worker.metrics.failed_requests / worker.metrics.total_requests
            if failure_rate <= max_failure_rate:
                filtered.append(worker)
        
        return filtered


class LoadBalancer:
    """High-level load balancer combining selection and filtering."""
    
    def __init__(self, selector_config: Optional[SelectionConfig] = None):
        self.selector = WorkerSelector(selector_config)
        self.filter = WorkerFilter()
        
        logger.info("LoadBalancer initialized")
    
    def select_worker_for_request(
        self,
        available_workers: List[WorkerInfo],
        model: Optional[str] = None,
        request_requirements: Optional[Dict[str, Any]] = None,
        request_context: Optional[Dict[str, Any]] = None
    ) -> Optional[WorkerInfo]:
        """Select the best worker for a request with advanced filtering."""
        if not available_workers:
            return None
        
        # Start with all available workers
        candidates = available_workers.copy()
        
        # Apply model filtering
        if model:
            candidates = self.filter.filter_by_model(candidates, model)
            if not candidates:
                logger.warning(f"No workers support model: {model}")
                return None
        
        # Apply requirement-based filtering
        if request_requirements:
            # Memory requirements
            if 'min_memory_gb' in request_requirements:
                candidates = self.filter.filter_by_memory(
                    candidates, request_requirements['min_memory_gb']
                )
            
            # Performance requirements
            if 'min_tokens_per_second' in request_requirements:
                candidates = self.filter.filter_by_performance(
                    candidates,
                    min_tokens_per_second=request_requirements['min_tokens_per_second'],
                    max_latency_ms=request_requirements.get('max_latency_ms', float('inf'))
                )
            
            # Reliability requirements
            if 'max_failure_rate' in request_requirements:
                candidates = self.filter.filter_by_reliability(
                    candidates,
                    max_failure_rate=request_requirements['max_failure_rate']
                )
            
            # Device type requirements
            if 'device_model' in request_requirements:
                candidates = self.filter.filter_by_device_type(
                    candidates, request_requirements['device_model']
                )
        
        # Select from filtered candidates
        return self.selector.select_worker(candidates, model, request_context)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        return {
            'selector_stats': self.selector.get_selection_stats(),
            'config': {
                'strategy': self.selector.config.strategy.value,
                'weights': {
                    'load': self.selector.config.weights.load_weight,
                    'latency': self.selector.config.weights.latency_weight,
                    'reliability': self.selector.config.weights.reliability_weight
                }
            }
        }