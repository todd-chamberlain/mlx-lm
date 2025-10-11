"""
MLX Inference Engine

This module handles inference processing with native MLX integration,
including token counting, performance metrics, and generation parameters.
"""

import time
import logging
import asyncio
from typing import Dict, Any, Optional, Iterator, Tuple
from dataclasses import dataclass
from threading import Lock
import uuid

import mlx.core as mx
import mlx_lm
from transformers import PreTrainedTokenizer

from .model_manager import ModelManager
from common.gc_optimizer import GCOptimizer, initialize_gc_optimizer
from common.tensor_lifecycle import TensorLifecycleManager, get_lifecycle_manager

logger = logging.getLogger(__name__)


@dataclass
class InferenceRequest:
    """Inference request data"""
    request_id: str
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    seed: Optional[int] = None
    model: str = "default"
    use_cache: bool = True


@dataclass
class InferenceResponse:
    """Inference response data"""
    request_id: str
    text: str
    tokens: list
    prompt_tokens: int
    completion_tokens: int
    tokens_per_second: float
    worker_id: str
    latency_ms: float


@dataclass
class StreamChunk:
    """Streaming response chunk"""
    request_id: str
    text: str
    token: int
    is_final: bool
    worker_id: str


class InferenceMetrics:
    """Track inference performance metrics"""
    
    def __init__(self):
        self.total_requests = 0
        self.failed_requests = 0
        self.active_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.lock = Lock()
    
    def start_request(self):
        """Mark request as started"""
        with self.lock:
            self.total_requests += 1
            self.active_requests += 1
    
    def end_request(self, success: bool, tokens: int, latency: float):
        """Mark request as completed"""
        with self.lock:
            self.active_requests -= 1
            if not success:
                self.failed_requests += 1
            else:
                self.total_tokens += tokens
                self.total_latency += latency
    
    def get_stats(self) -> Dict[str, float]:
        """Get current metrics"""
        with self.lock:
            successful_requests = self.total_requests - self.failed_requests
            avg_latency = (self.total_latency / successful_requests 
                          if successful_requests > 0 else 0.0)
            avg_tokens_per_sec = (self.total_tokens / (self.total_latency / 1000) 
                                 if self.total_latency > 0 else 0.0)
            
            return {
                'total_requests': self.total_requests,
                'failed_requests': self.failed_requests,
                'active_requests': self.active_requests,
                'avg_latency_ms': avg_latency,
                'tokens_per_second': avg_tokens_per_sec,
                'success_rate': (successful_requests / self.total_requests 
                               if self.total_requests > 0 else 1.0)
            }


class InferenceEngine:
    """MLX inference engine with performance tracking"""
    
    def __init__(self, model_manager: ModelManager, worker_id: str, 
                 enable_gc_optimization: bool = True,
                 enable_lifecycle_tracking: bool = True):
        self.model_manager = model_manager
        self.worker_id = worker_id
        self.metrics = InferenceMetrics()
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.request_lock = Lock()
        
        # Initialize GC optimizer
        self.gc_optimizer: Optional[GCOptimizer] = None
        if enable_gc_optimization:
            # Create callback to record GC events to Prometheus
            def gc_event_callback(event):
                try:
                    from .metrics import worker_metrics
                    if worker_metrics:
                        worker_metrics.record_gc_event(
                            generation=event.generation,
                            collected=event.collected,
                            uncollectable=event.uncollectable,
                            duration=event.duration_ms / 1000.0,  # Convert to seconds
                            manual=event.manual
                        )
                except Exception as e:
                    logger.error(f"Error recording GC event to metrics: {e}")
            
            self.gc_optimizer = initialize_gc_optimizer(
                gc_threshold=(10000, 20, 20),
                manual_gc_enabled=True,
                track_metrics=True,
                event_callback=gc_event_callback
            )
            logger.info(f"Initialized GC optimizer for worker {worker_id}")
        
        # Initialize tensor lifecycle manager
        self.lifecycle_manager: Optional[TensorLifecycleManager] = None
        self.enable_lifecycle_tracking = enable_lifecycle_tracking
        if enable_lifecycle_tracking:
            self.lifecycle_manager = get_lifecycle_manager()
            logger.info(f"Initialized tensor lifecycle manager for worker {worker_id}")
        
        logger.info(f"Initialized InferenceEngine for worker {worker_id}")
    
    def _validate_request(self, request: InferenceRequest) -> Optional[str]:
        """Validate inference request parameters"""
        if not request.prompt:
            return "Prompt cannot be empty"
        
        if request.max_tokens <= 0 or request.max_tokens > 4096:
            return "max_tokens must be between 1 and 4096"
        
        if request.temperature < 0.0 or request.temperature > 2.0:
            return "temperature must be between 0.0 and 2.0"
        
        if request.top_p < 0.0 or request.top_p > 1.0:
            return "top_p must be between 0.0 and 1.0"
        
        # Check if model is loaded
        if not self.model_manager.get_model(request.model):
            return f"Model {request.model} is not loaded"
        
        return None
    
    def _count_tokens(self, tokenizer: PreTrainedTokenizer, text: str) -> int:
        """Count tokens in text"""
        try:
            tokens = tokenizer.encode(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Rough estimation: ~4 characters per token
            return len(text) // 4
    
    def _prepare_generation_params(self, request: InferenceRequest) -> Dict[str, Any]:
        """Prepare parameters for MLX generation"""
        params = {
            'max_tokens': request.max_tokens,
            'temperature': request.temperature,
            'top_p': request.top_p,
        }
        
        if request.seed is not None:
            params['seed'] = request.seed
        
        return params
    
    async def process_inference(self, request: InferenceRequest) -> InferenceResponse:
        """Process a single inference request"""
        start_time = time.time()
        
        # Track request start
        self.metrics.start_request()
        with self.request_lock:
            self.active_requests[request.request_id] = start_time
        
        try:
            # Validate request
            validation_error = self._validate_request(request)
            if validation_error:
                raise ValueError(validation_error)
            
            # ============================================================
            # SHARDING DECISION POINT: Inference Routing
            # ============================================================
            # This is where we decide whether to use standard inference or
            # distributed sharded inference based on the model's metadata.
            #
            # Decision criteria:
            # - If model_info['is_sharded'] == True: Use distributed pipeline
            # - If model_info['is_sharded'] == False: Use standard inference
            #
            # The 'is_sharded' flag is set by ModelManager when loading a model
            # with shard_metadata (see ModelManager.load_sharded_model).
            # ============================================================
            
            model_info = self.model_manager.get_model_info_dict(request.model)
            is_sharded = model_info and model_info.get('is_sharded', False)
            
            logger.info(
                f"Routing inference request",
                extra={
                    "request_id": request.request_id,
                    "model": request.model,
                    "worker_id": self.worker_id,
                    "is_sharded": is_sharded,
                    "routing_decision": "sharded_pipeline" if is_sharded else "standard_inference",
                    "operation": "inference_routing"
                }
            )
            
            if is_sharded:
                # Route to distributed inference pipeline
                # This will use DistributedForwardPass to execute across multiple workers
                return await self._process_sharded_inference(request, model_info)
            
            # Standard (non-sharded) inference path
            # Get model and tokenizer
            model_tuple = self.model_manager.get_model(request.model)
            if not model_tuple:
                raise RuntimeError(f"Model {request.model} not available")
            
            model, tokenizer = model_tuple
            
            # Count prompt tokens
            prompt_tokens = self._count_tokens(tokenizer, request.prompt)
            
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(request)
            
            logger.debug(f"Processing inference request {request.request_id} "
                        f"with {prompt_tokens} prompt tokens")
            
            # Run inference
            generation_start = time.time()
            
            # Track tensors if lifecycle tracking is enabled
            tensor_ids = []
            if self.lifecycle_manager:
                # Track input tensors
                input_tensor_id = f"{request.request_id}_input"
                try:
                    # Get input tokens as tensor
                    input_ids = tokenizer.encode(request.prompt)
                    input_tensor = mx.array(input_ids)
                    self.lifecycle_manager.track_tensor(
                        input_tensor_id,
                        input_tensor,
                        context="inference_input"
                    )
                    tensor_ids.append(input_tensor_id)
                except Exception as e:
                    logger.debug(f"Could not track input tensor: {e}")
            
            # Use MLX-LM generate function
            response_text = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                **gen_params
            )
            
            generation_time = time.time() - generation_start
            
            # Track output tensors if lifecycle tracking is enabled
            if self.lifecycle_manager:
                output_tensor_id = f"{request.request_id}_output"
                try:
                    # Get output tokens as tensor
                    output_ids = tokenizer.encode(response_text)
                    output_tensor = mx.array(output_ids)
                    self.lifecycle_manager.track_tensor(
                        output_tensor_id,
                        output_tensor,
                        context="inference_output"
                    )
                    tensor_ids.append(output_tensor_id)
                except Exception as e:
                    logger.debug(f"Could not track output tensor: {e}")
            
            # Count completion tokens
            completion_tokens = self._count_tokens(tokenizer, response_text)
            
            # Calculate metrics
            total_time = time.time() - start_time
            latency_ms = total_time * 1000
            tokens_per_second = completion_tokens / generation_time if generation_time > 0 else 0
            
            # Create response
            response = InferenceResponse(
                request_id=request.request_id,
                text=response_text,
                tokens=tokenizer.encode(response_text),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                tokens_per_second=tokens_per_second,
                worker_id=self.worker_id,
                latency_ms=latency_ms
            )
            
            # Track successful completion
            self.metrics.end_request(True, completion_tokens, latency_ms)
            
            logger.info(f"Completed inference {request.request_id} in {latency_ms:.2f}ms "
                       f"({tokens_per_second:.2f} tokens/sec)")
            
            return response
            
        except Exception as e:
            # Track failed request
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.end_request(False, 0, latency_ms)
            
            logger.error(f"Inference failed for request {request.request_id}: {e}")
            raise
        
        finally:
            # Clean up request tracking
            with self.request_lock:
                self.active_requests.pop(request.request_id, None)
            
            # Release tracked tensors for eager deallocation
            if self.lifecycle_manager:
                for tensor_id in tensor_ids:
                    self.lifecycle_manager.release_tensor(tensor_id)
            
            # Trigger manual GC after request completion
            if self.gc_optimizer:
                self.gc_optimizer.manual_gc_cycle(generation=0)
    
    async def stream_inference(self, request: InferenceRequest) -> Iterator[StreamChunk]:
        """Process streaming inference request"""
        start_time = time.time()
        
        # Track request start
        self.metrics.start_request()
        with self.request_lock:
            self.active_requests[request.request_id] = start_time
        
        try:
            # Validate request
            validation_error = self._validate_request(request)
            if validation_error:
                raise ValueError(validation_error)
            
            # ============================================================
            # SHARDING DECISION POINT: Streaming Inference Routing
            # ============================================================
            # Similar to standard inference, we route streaming requests based
            # on whether the model is sharded.
            #
            # Decision criteria:
            # - If model_info['is_sharded'] == True: Use distributed streaming
            # - If model_info['is_sharded'] == False: Use standard streaming
            # ============================================================
            
            model_info = self.model_manager.get_model_info_dict(request.model)
            is_sharded = model_info and model_info.get('is_sharded', False)
            
            logger.info(
                f"Routing streaming inference request",
                extra={
                    "request_id": request.request_id,
                    "model": request.model,
                    "worker_id": self.worker_id,
                    "is_sharded": is_sharded,
                    "routing_decision": "sharded_streaming_pipeline" if is_sharded else "standard_streaming",
                    "operation": "streaming_inference_routing"
                }
            )
            
            if is_sharded:
                # Route to distributed streaming inference pipeline
                # This will use DistributedForwardPass for streaming generation
                async for chunk in self._stream_sharded_inference(request, model_info):
                    yield chunk
                return
            
            # Standard (non-sharded) streaming inference path
            # Get model and tokenizer
            model_tuple = self.model_manager.get_model(request.model)
            if not model_tuple:
                raise RuntimeError(f"Model {request.model} not available")
            
            model, tokenizer = model_tuple
            
            # Count prompt tokens
            prompt_tokens = self._count_tokens(tokenizer, request.prompt)
            
            # Prepare generation parameters
            gen_params = self._prepare_generation_params(request)
            
            logger.debug(f"Starting streaming inference {request.request_id} "
                        f"with {prompt_tokens} prompt tokens")
            
            # Track tensors if lifecycle tracking is enabled
            tensor_ids = []
            if self.lifecycle_manager:
                # Track input tensors
                input_tensor_id = f"{request.request_id}_stream_input"
                try:
                    input_ids = tokenizer.encode(request.prompt)
                    input_tensor = mx.array(input_ids)
                    self.lifecycle_manager.track_tensor(
                        input_tensor_id,
                        input_tensor,
                        context="streaming_input"
                    )
                    tensor_ids.append(input_tensor_id)
                except Exception as e:
                    logger.debug(f"Could not track input tensor: {e}")
            
            # Use MLX-LM stream_generate function
            token_count = 0
            generation_start = time.time()
            
            for token_text in mlx_lm.stream_generate(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                **gen_params
            ):
                token_count += 1
                
                # Encode the token text to get token ID
                try:
                    token_ids = tokenizer.encode(token_text)
                    token_id = token_ids[-1] if token_ids else 0
                except:
                    token_id = 0
                
                # Create stream chunk
                chunk = StreamChunk(
                    request_id=request.request_id,
                    text=token_text,
                    token=token_id,
                    is_final=False,
                    worker_id=self.worker_id
                )
                
                yield chunk
                
                # Allow other coroutines to run
                await asyncio.sleep(0)
            
            # Send final chunk
            generation_time = time.time() - generation_start
            final_chunk = StreamChunk(
                request_id=request.request_id,
                text="",
                token=0,
                is_final=True,
                worker_id=self.worker_id
            )
            yield final_chunk
            
            # Track successful completion
            total_time = time.time() - start_time
            latency_ms = total_time * 1000
            self.metrics.end_request(True, token_count, latency_ms)
            
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            logger.info(f"Completed streaming inference {request.request_id} "
                       f"with {token_count} tokens in {latency_ms:.2f}ms "
                       f"({tokens_per_second:.2f} tokens/sec)")
            
        except Exception as e:
            # Track failed request
            latency_ms = (time.time() - start_time) * 1000
            self.metrics.end_request(False, 0, latency_ms)
            
            logger.error(f"Streaming inference failed for request {request.request_id}: {e}")
            raise
        
        finally:
            # Clean up request tracking
            with self.request_lock:
                self.active_requests.pop(request.request_id, None)
            
            # Release tracked tensors for eager deallocation
            if self.lifecycle_manager:
                for tensor_id in tensor_ids:
                    self.lifecycle_manager.release_tensor(tensor_id)
            
            # Trigger manual GC after streaming request completion
            if self.gc_optimizer:
                self.gc_optimizer.manual_gc_cycle(generation=0)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current inference metrics"""
        stats = self.metrics.get_stats()
        
        # Add current active requests info
        with self.request_lock:
            current_time = time.time()
            active_request_times = [
                current_time - start_time 
                for start_time in self.active_requests.values()
            ]
        
        stats.update({
            'active_requests': len(active_request_times),
            'longest_active_request_seconds': max(active_request_times) if active_request_times else 0.0
        })
        
        # Add GC metrics if optimizer is enabled
        if self.gc_optimizer:
            stats['gc_stats'] = self.gc_optimizer.get_gc_stats()
        
        # Add tensor lifecycle metrics if enabled
        if self.lifecycle_manager:
            lifecycle_stats = self.lifecycle_manager.get_stats()
            stats['tensor_lifecycle'] = {
                'total_allocations': lifecycle_stats.total_allocations,
                'total_deallocations': lifecycle_stats.total_deallocations,
                'current_tracked': lifecycle_stats.current_tracked,
                'current_bytes_tracked': lifecycle_stats.current_bytes_tracked,
                'eager_deallocations': lifecycle_stats.eager_deallocations,
                'allocation_rate': lifecycle_stats.get_allocation_rate(),
                'memory_efficiency': lifecycle_stats.get_memory_efficiency()
            }
        
        return stats
    
    def get_active_request_count(self) -> int:
        """Get number of currently active requests"""
        with self.request_lock:
            return len(self.active_requests)
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel an active request (best effort)"""
        with self.request_lock:
            if request_id in self.active_requests:
                # Note: MLX doesn't support cancellation mid-generation
                # This is mainly for tracking purposes
                logger.warning(f"Request {request_id} marked for cancellation "
                             f"(MLX doesn't support mid-generation cancellation)")
                return True
            return False
    
    def cleanup_stale_requests(self, max_age_seconds: float = 300):
        """Clean up requests that have been active too long"""
        current_time = time.time()
        stale_requests = []
        
        with self.request_lock:
            for request_id, start_time in self.active_requests.items():
                if current_time - start_time > max_age_seconds:
                    stale_requests.append(request_id)
            
            for request_id in stale_requests:
                self.active_requests.pop(request_id, None)
                logger.warning(f"Cleaned up stale request: {request_id}")
        
        return len(stale_requests)
    
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """
        Detect potential memory leaks in tensor lifecycle.
        
        Returns:
            Dictionary with leak detection results
        """
        if not self.lifecycle_manager:
            return {'enabled': False}
        
        leak_report = self.lifecycle_manager.detect_leaks()
        
        return {
            'enabled': True,
            'leaked_tensor_count': len(leak_report.leaked_tensors),
            'total_leaked_bytes': leak_report.total_leaked_bytes,
            'leak_contexts': leak_report.leak_contexts,
            'oldest_leak_age_seconds': leak_report.oldest_leak_age_seconds
        }
    
    def cleanup_leaked_tensors(self) -> int:
        """
        Clean up detected leaked tensors.
        
        Returns:
            Number of tensors cleaned up
        """
        if not self.lifecycle_manager:
            return 0
        
        cleaned = self.lifecycle_manager.cleanup_leaked_tensors()
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} leaked tensors")
        
        return cleaned
    
    def get_memory_profile(self) -> Dict[str, Any]:
        """
        Get detailed memory profiling information.
        
        Returns:
            Dictionary with memory profiling data
        """
        if not self.lifecycle_manager:
            return {'enabled': False}
        
        profile = self.lifecycle_manager.get_memory_profile()
        profile['enabled'] = True
        
        return profile
    
    async def _process_sharded_inference(
        self,
        request: InferenceRequest,
        model_info: Dict[str, Any]
    ) -> InferenceResponse:
        """
        Process inference request for sharded model using distributed pipeline.
        
        Args:
            request: Inference request
            model_info: Model information including shard metadata
            
        Returns:
            InferenceResponse with results from distributed pipeline
        """
        from .distributed_forward import DistributedForwardPass
        
        start_time = time.time()
        # ============================================================
        # SHARD METADATA USAGE: Pipeline Validation
        # ============================================================
        # Shard metadata contains information about this worker's role in the
        # distributed pipeline. Key fields:
        # - is_first_worker: Must be True to initiate inference
        # - start_layer/end_layer: Layer range this worker handles
        # - next_worker_id/next_worker_address: Next worker in pipeline
        #
        # Only the first worker in the pipeline can initiate inference.
        # Subsequent workers receive activations from previous workers.
        # ============================================================
        
        shard_metadata = model_info.get('shard_metadata', {})
        
        # Check if this is the first worker in the pipeline
        if not shard_metadata.get('is_first_worker', False):
            raise RuntimeError(
                f"Sharded inference must be initiated on first worker in pipeline. "
                f"This worker handles layers {shard_metadata.get('start_layer')}-"
                f"{shard_metadata.get('end_layer')}"
            )
        
        logger.info(
            f"Processing sharded inference request through distributed pipeline",
            extra={
                "request_id": request.request_id,
                "model": request.model,
                "worker_id": self.worker_id,
                "start_layer": shard_metadata.get('start_layer'),
                "end_layer": shard_metadata.get('end_layer'),
                "layer_range": f"{shard_metadata.get('start_layer')}-{shard_metadata.get('end_layer')}",
                "next_worker": shard_metadata.get('next_worker_id'),
                "operation": "sharded_inference_start"
            }
        )
        
        # Get model and tokenizer
        model_tuple = self.model_manager.get_model(request.model)
        if not model_tuple:
            raise RuntimeError(f"Model {request.model} not available")
        
        model, tokenizer = model_tuple
        
        # Tokenize prompt
        prompt_tokens = self._count_tokens(tokenizer, request.prompt)
        input_ids = tokenizer.encode(request.prompt)
        input_tensor = mx.array(input_ids).reshape(1, -1)
        
        # Build pipeline configuration from shard metadata
        node_assignments = {
            self.worker_id: list(range(
                shard_metadata['start_layer'],
                shard_metadata['end_layer']
            ))
        }
        
        node_addresses = {}
        
        # Add next worker if present
        if shard_metadata.get('next_worker_id'):
            next_worker_id = shard_metadata['next_worker_id']
            next_worker_address = shard_metadata['next_worker_address']
            
            # Estimate next worker's layer range
            # This should come from coordinator in production
            total_layers = model_info.get('num_layers', 120)
            next_start = shard_metadata['end_layer']
            next_end = total_layers
            
            node_assignments[next_worker_id] = list(range(next_start, next_end))
            node_addresses[next_worker_id] = next_worker_address
        
        # Create distributed forward pass
        forward_pass = DistributedForwardPass(
            pipeline_id=request.request_id,
            node_assignments=node_assignments,
            node_addresses=node_addresses,
            compression="zstd",
            timeout=30.0
        )
        
        try:
            # Execute distributed forward pass
            output_logits = await forward_pass.forward(
                input_ids=input_tensor,
                request_id=request.request_id
            )
            
            # Sample token from logits (greedy sampling for simplicity)
            next_token_id = int(mx.argmax(output_logits[0, -1, :]).item())
            
            # Generate remaining tokens
            generated_tokens = [next_token_id]
            for _ in range(request.max_tokens - 1):
                next_input = mx.array([[next_token_id]])
                output_logits = await forward_pass.forward(
                    input_ids=next_input,
                    request_id=request.request_id
                )
                next_token_id = int(mx.argmax(output_logits[0, -1, :]).item())
                generated_tokens.append(next_token_id)
                
                # Check for EOS
                if next_token_id == tokenizer.eos_token_id:
                    break
            
            # Decode response
            response_text = tokenizer.decode(generated_tokens)
            
            # Calculate metrics
            total_time = time.time() - start_time
            latency_ms = total_time * 1000
            tokens_per_second = len(generated_tokens) / total_time if total_time > 0 else 0
            
            logger.info(
                f"Completed sharded inference request",
                extra={
                    "request_id": request.request_id,
                    "model": request.model,
                    "worker_id": self.worker_id,
                    "latency_ms": latency_ms,
                    "tokens_generated": len(generated_tokens),
                    "tokens_per_second": tokens_per_second,
                    "operation": "sharded_inference_complete"
                }
            )
            
            return InferenceResponse(
                request_id=request.request_id,
                text=response_text,
                tokens=generated_tokens,
                prompt_tokens=prompt_tokens,
                completion_tokens=len(generated_tokens),
                tokens_per_second=tokens_per_second,
                worker_id=self.worker_id,
                latency_ms=latency_ms
            )
            
        except Exception as e:
            logger.error(
                f"Sharded inference failed",
                extra={
                    "request_id": request.request_id,
                    "model": request.model,
                    "worker_id": self.worker_id,
                    "error": str(e),
                    "operation": "sharded_inference_error"
                }
            )
            raise RuntimeError(f"Sharded inference failed: {str(e)}") from e
            
        finally:
            await forward_pass.close()
    
    async def _stream_sharded_inference(
        self,
        request: InferenceRequest,
        model_info: Dict[str, Any]
    ) -> Iterator[StreamChunk]:
        """
        Process streaming inference request for sharded model using distributed pipeline.
        
        Args:
            request: Inference request
            model_info: Model information including shard metadata
            
        Yields:
            StreamChunk objects with generated tokens
        """
        from .distributed_forward import DistributedForwardPass
        
        start_time = time.time()
        shard_metadata = model_info.get('shard_metadata', {})
        
        # Check if this is the first worker in the pipeline
        if not shard_metadata.get('is_first_worker', False):
            raise RuntimeError(
                f"Sharded streaming inference must be initiated on first worker. "
                f"This worker handles layers {shard_metadata.get('start_layer')}-"
                f"{shard_metadata.get('end_layer')}"
            )
        
        logger.info(
            f"Processing sharded streaming inference request {request.request_id} "
            f"through distributed pipeline"
        )
        
        # Get model and tokenizer
        model_tuple = self.model_manager.get_model(request.model)
        if not model_tuple:
            raise RuntimeError(f"Model {request.model} not available")
        
        model, tokenizer = model_tuple
        
        # Tokenize prompt
        input_ids = tokenizer.encode(request.prompt)
        input_tensor = mx.array(input_ids).reshape(1, -1)
        
        # Build pipeline configuration
        node_assignments = {
            self.worker_id: list(range(
                shard_metadata['start_layer'],
                shard_metadata['end_layer']
            ))
        }
        
        node_addresses = {}
        
        # Add next worker if present
        if shard_metadata.get('next_worker_id'):
            next_worker_id = shard_metadata['next_worker_id']
            next_worker_address = shard_metadata['next_worker_address']
            
            total_layers = model_info.get('num_layers', 120)
            next_start = shard_metadata['end_layer']
            next_end = total_layers
            
            node_assignments[next_worker_id] = list(range(next_start, next_end))
            node_addresses[next_worker_id] = next_worker_address
        
        # Create distributed forward pass
        forward_pass = DistributedForwardPass(
            pipeline_id=request.request_id,
            node_assignments=node_assignments,
            node_addresses=node_addresses,
            compression="zstd",
            timeout=30.0
        )
        
        try:
            # Execute distributed forward pass for prompt
            output_logits = await forward_pass.forward(
                input_ids=input_tensor,
                request_id=request.request_id
            )
            
            # Sample first token
            next_token_id = int(mx.argmax(output_logits[0, -1, :]).item())
            
            # Stream tokens
            token_count = 0
            for _ in range(request.max_tokens):
                # Decode token
                token_text = tokenizer.decode([next_token_id])
                
                # Create and yield chunk
                chunk = StreamChunk(
                    request_id=request.request_id,
                    text=token_text,
                    token=next_token_id,
                    is_final=False,
                    worker_id=self.worker_id
                )
                yield chunk
                token_count += 1
                
                # Check for EOS
                if next_token_id == tokenizer.eos_token_id:
                    break
                
                # Generate next token
                next_input = mx.array([[next_token_id]])
                output_logits = await forward_pass.forward(
                    input_ids=next_input,
                    request_id=request.request_id
                )
                next_token_id = int(mx.argmax(output_logits[0, -1, :]).item())
                
                # Allow other coroutines to run
                await asyncio.sleep(0)
            
            # Send final chunk
            final_chunk = StreamChunk(
                request_id=request.request_id,
                text="",
                token=0,
                is_final=True,
                worker_id=self.worker_id
            )
            yield final_chunk
            
            total_time = time.time() - start_time
            tokens_per_second = token_count / total_time if total_time > 0 else 0
            
            logger.info(
                f"Completed sharded streaming inference {request.request_id} "
                f"with {token_count} tokens in {total_time:.2f}s "
                f"({tokens_per_second:.2f} tokens/sec)"
            )
            
        except Exception as e:
            logger.error(f"Sharded streaming inference failed for {request.request_id}: {e}")
            raise RuntimeError(f"Sharded streaming inference failed: {str(e)}") from e
            
        finally:
            await forward_pass.close()
