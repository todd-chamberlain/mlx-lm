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
    
    def __init__(self, model_manager: ModelManager, worker_id: str):
        self.model_manager = model_manager
        self.worker_id = worker_id
        self.metrics = InferenceMetrics()
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.request_lock = Lock()
        
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
            'temp': request.temperature,
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
            
            # Use MLX-LM generate function
            response_text = mlx_lm.generate(
                model=model,
                tokenizer=tokenizer,
                prompt=request.prompt,
                **gen_params
            )
            
            generation_time = time.time() - generation_start
            
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