"""
Distributed Token Generation

Implements iterative token generation through a distributed pipeline,
managing KV cache across nodes and coordinating generation stopping criteria.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, AsyncIterator, Tuple
from dataclasses import dataclass, field
import mlx.core as mx
import mlx.nn as nn

from .distributed_forward import DistributedForwardPass
from .activation_transfer import ActivationTransfer
from proto.distributed_pb2 import (
    ActivationData,
    GenerationConfig,
    GenerationMetrics,
    StopReason
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """State for distributed generation process."""
    
    # Generation parameters
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    stop_sequences: List[str] = field(default_factory=list)
    
    # Current state
    generated_tokens: List[int] = field(default_factory=list)
    current_position: int = 0
    finished: bool = False
    stop_reason: Optional[str] = None
    
    # KV cache per node
    kv_caches: Dict[str, Any] = field(default_factory=dict)
    
    # Metrics
    start_time: float = field(default_factory=time.time)
    tokens_per_second: float = 0.0
    total_latency: float = 0.0


@dataclass
class GenerationResult:
    """Result of distributed generation."""
    
    tokens: List[int]
    text: str
    stop_reason: str
    metrics: Dict[str, Any]


class DistributedGenerator:
    """
    Manages distributed token generation through a pipeline.
    
    Coordinates iterative generation across multiple nodes, managing
    KV cache state and applying generation stopping criteria.
    """
    
    def __init__(
        self,
        forward_pass: DistributedForwardPass,
        activation_transfer: ActivationTransfer,
        tokenizer: Any,
        node_id: str
    ):
        """
        Initialize distributed generator.
        
        Args:
            forward_pass: Distributed forward pass handler
            activation_transfer: Activation transfer protocol
            tokenizer: Tokenizer for text encoding/decoding
            node_id: ID of this node in the pipeline
        """
        self.forward_pass = forward_pass
        self.activation_transfer = activation_transfer
        self.tokenizer = tokenizer
        self.node_id = node_id
        
        # Generation state
        self.current_state: Optional[GenerationState] = None
        
        # Metrics
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0
        
        logger.info(f"Initialized DistributedGenerator for node {node_id}")
    
    async def generate(
        self,
        prompt_tokens: List[int],
        max_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False
    ) -> AsyncIterator[GenerationResult]:
        """
        Generate tokens through distributed pipeline.
        
        Args:
            prompt_tokens: Input prompt tokens
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty factor
            stop_sequences: Sequences that stop generation
            stream: Whether to stream tokens as generated
            
        Yields:
            GenerationResult objects (one per token if streaming, final result if not)
        """
        # Initialize generation state
        state = GenerationState(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            stop_sequences=stop_sequences or []
        )
        self.current_state = state
        
        try:
            # Process prompt through pipeline
            logger.info(f"Starting generation with {len(prompt_tokens)} prompt tokens")
            
            # Initial forward pass with prompt
            prompt_logits = await self._process_prompt(prompt_tokens, state)
            
            # Sample first token
            first_token = self._sample_token(prompt_logits, state)
            state.generated_tokens.append(first_token)
            state.current_position = len(prompt_tokens)
            
            if stream:
                yield self._create_result(state, partial=True)
            
            # Iterative generation
            while not state.finished and len(state.generated_tokens) < max_tokens:
                # Generate next token
                next_token = await self._generate_next_token(state)
                
                if next_token is None:
                    break
                
                state.generated_tokens.append(next_token)
                state.current_position += 1
                
                # Check stopping criteria
                if self._should_stop(state):
                    break
                
                if stream:
                    yield self._create_result(state, partial=True)
            
            # Finalize generation
            state.finished = True
            if state.stop_reason is None:
                state.stop_reason = "max_tokens" if len(state.generated_tokens) >= max_tokens else "complete"
            
            # Calculate final metrics
            state.total_latency = time.time() - state.start_time
            if state.total_latency > 0:
                state.tokens_per_second = len(state.generated_tokens) / state.total_latency
            
            # Update global metrics
            self.total_tokens_generated += len(state.generated_tokens)
            self.total_generation_time += state.total_latency
            
            logger.info(
                f"Generation complete: {len(state.generated_tokens)} tokens, "
                f"{state.tokens_per_second:.2f} tokens/sec, "
                f"stop_reason={state.stop_reason}"
            )
            
            yield self._create_result(state, partial=False)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            state.finished = True
            state.stop_reason = "error"
            raise
        
        finally:
            self.current_state = None
    
    async def _process_prompt(
        self,
        prompt_tokens: List[int],
        state: GenerationState
    ) -> mx.array:
        """
        Process prompt through pipeline to get initial logits.
        
        Args:
            prompt_tokens: Input prompt tokens
            state: Generation state
            
        Returns:
            Logits for next token prediction
        """
        # Convert tokens to MLX array
        input_ids = mx.array(prompt_tokens).reshape(1, -1)
        
        # Forward pass through pipeline
        logits = await self.forward_pass.forward(
            input_ids,
            cache=state.kv_caches,
            return_cache=True
        )
        
        # Return logits for last position
        return logits[:, -1, :]
    
    async def _generate_next_token(
        self,
        state: GenerationState
    ) -> Optional[int]:
        """
        Generate next token through pipeline.
        
        Args:
            state: Current generation state
            
        Returns:
            Next token ID or None if generation should stop
        """
        try:
            # Get last generated token
            last_token = state.generated_tokens[-1]
            input_ids = mx.array([[last_token]])
            
            # Forward pass with KV cache
            logits = await self.forward_pass.forward(
                input_ids,
                cache=state.kv_caches,
                return_cache=True
            )
            
            # Sample next token
            next_token = self._sample_token(logits[:, -1, :], state)
            
            return next_token
            
        except Exception as e:
            logger.error(f"Failed to generate next token: {e}")
            state.stop_reason = "error"
            return None
    
    def _sample_token(
        self,
        logits: mx.array,
        state: GenerationState
    ) -> int:
        """
        Sample next token from logits using generation parameters.
        
        Args:
            logits: Model logits for next token
            state: Generation state with sampling parameters
            
        Returns:
            Sampled token ID
        """
        # Apply temperature
        if state.temperature != 1.0:
            logits = logits / state.temperature
        
        # Apply repetition penalty
        if state.repetition_penalty != 1.0 and state.generated_tokens:
            for token_id in set(state.generated_tokens):
                logits[0, token_id] /= state.repetition_penalty
        
        # Convert to probabilities
        probs = mx.softmax(logits, axis=-1)
        
        # Apply top-k filtering
        if state.top_k > 0:
            top_k_probs, top_k_indices = mx.topk(probs, state.top_k, axis=-1)
            # Create mask for top-k
            mask = mx.zeros_like(probs)
            mask = mx.scatter(mask, top_k_indices, mx.ones_like(top_k_probs), axis=-1)
            probs = probs * mask
            probs = probs / mx.sum(probs, axis=-1, keepdims=True)
        
        # Apply top-p (nucleus) filtering
        if state.top_p < 1.0:
            sorted_probs, sorted_indices = mx.sort(probs, axis=-1)
            sorted_probs = sorted_probs[:, ::-1]
            sorted_indices = sorted_indices[:, ::-1]
            
            cumsum_probs = mx.cumsum(sorted_probs, axis=-1)
            mask = cumsum_probs <= state.top_p
            
            # Keep at least one token
            mask[:, 0] = True
            
            # Apply mask
            filtered_probs = mx.where(mask, sorted_probs, mx.zeros_like(sorted_probs))
            filtered_probs = filtered_probs / mx.sum(filtered_probs, axis=-1, keepdims=True)
            
            # Sample from filtered distribution
            token_idx = mx.random.categorical(mx.log(filtered_probs))
            token_id = sorted_indices[0, token_idx.item()]
        else:
            # Sample from full distribution
            token_id = mx.random.categorical(mx.log(probs))
        
        return int(token_id.item())
    
    def _should_stop(self, state: GenerationState) -> bool:
        """
        Check if generation should stop based on stopping criteria.
        
        Args:
            state: Current generation state
            
        Returns:
            True if generation should stop
        """
        # Check max tokens
        if len(state.generated_tokens) >= state.max_tokens:
            state.stop_reason = "max_tokens"
            return True
        
        # Check for EOS token
        if state.generated_tokens and state.generated_tokens[-1] == self.tokenizer.eos_token_id:
            state.stop_reason = "eos_token"
            return True
        
        # Check stop sequences
        if state.stop_sequences:
            # Decode recent tokens to check for stop sequences
            recent_text = self.tokenizer.decode(state.generated_tokens[-20:])
            for stop_seq in state.stop_sequences:
                if stop_seq in recent_text:
                    state.stop_reason = f"stop_sequence:{stop_seq}"
                    return True
        
        return False
    
    def _create_result(
        self,
        state: GenerationState,
        partial: bool = False
    ) -> GenerationResult:
        """
        Create generation result from current state.
        
        Args:
            state: Current generation state
            partial: Whether this is a partial result (streaming)
            
        Returns:
            GenerationResult object
        """
        # Decode tokens to text
        text = self.tokenizer.decode(state.generated_tokens)
        
        # Calculate metrics
        elapsed = time.time() - state.start_time
        tokens_per_sec = len(state.generated_tokens) / elapsed if elapsed > 0 else 0.0
        
        metrics = {
            "tokens_generated": len(state.generated_tokens),
            "tokens_per_second": tokens_per_sec,
            "total_latency": elapsed,
            "partial": partial,
            "position": state.current_position
        }
        
        return GenerationResult(
            tokens=state.generated_tokens.copy(),
            text=text,
            stop_reason=state.stop_reason or "in_progress",
            metrics=metrics
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get generation metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_tokens_per_sec = 0.0
        if self.total_generation_time > 0:
            avg_tokens_per_sec = self.total_tokens_generated / self.total_generation_time
        
        return {
            "total_tokens_generated": self.total_tokens_generated,
            "total_generation_time": self.total_generation_time,
            "average_tokens_per_second": avg_tokens_per_sec,
            "active_generation": self.current_state is not None
        }
    
    async def cleanup(self):
        """Clean up generator resources."""
        if self.current_state:
            self.current_state.kv_caches.clear()
            self.current_state = None
        
        logger.info("DistributedGenerator cleaned up")


class KVCacheManager:
    """
    Manages KV cache across distributed pipeline nodes.
    
    Handles cache initialization, updates, and synchronization
    between pipeline stages.
    """
    
    def __init__(self, node_id: str, num_layers: int):
        """
        Initialize KV cache manager.
        
        Args:
            node_id: ID of this node
            num_layers: Number of layers on this node
        """
        self.node_id = node_id
        self.num_layers = num_layers
        self.caches: Dict[str, List[Tuple[mx.array, mx.array]]] = {}
        
        logger.info(f"Initialized KVCacheManager for {num_layers} layers")
    
    def initialize_cache(
        self,
        request_id: str,
        batch_size: int = 1,
        max_length: int = 2048
    ):
        """
        Initialize KV cache for a request.
        
        Args:
            request_id: Unique request identifier
            batch_size: Batch size
            max_length: Maximum sequence length
        """
        # Initialize empty cache for each layer
        self.caches[request_id] = []
        
        logger.debug(f"Initialized KV cache for request {request_id}")
    
    def get_cache(self, request_id: str) -> Optional[List[Tuple[mx.array, mx.array]]]:
        """
        Get KV cache for a request.
        
        Args:
            request_id: Request identifier
            
        Returns:
            KV cache or None if not found
        """
        return self.caches.get(request_id)
    
    def update_cache(
        self,
        request_id: str,
        layer_idx: int,
        key: mx.array,
        value: mx.array
    ):
        """
        Update KV cache for a specific layer.
        
        Args:
            request_id: Request identifier
            layer_idx: Layer index
            key: Key tensor
            value: Value tensor
        """
        if request_id not in self.caches:
            self.initialize_cache(request_id)
        
        cache = self.caches[request_id]
        
        # Extend cache list if needed
        while len(cache) <= layer_idx:
            cache.append((None, None))
        
        # Update cache for this layer
        cache[layer_idx] = (key, value)
    
    def clear_cache(self, request_id: str):
        """
        Clear KV cache for a request.
        
        Args:
            request_id: Request identifier
        """
        if request_id in self.caches:
            del self.caches[request_id]
            logger.debug(f"Cleared KV cache for request {request_id}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics for caches.
        
        Returns:
            Dictionary of memory usage metrics
        """
        total_size = 0
        num_caches = len(self.caches)
        
        for cache in self.caches.values():
            for k, v in cache:
                if k is not None:
                    total_size += k.nbytes
                if v is not None:
                    total_size += v.nbytes
        
        return {
            "num_active_caches": num_caches,
            "total_memory_bytes": total_size,
            "total_memory_mb": total_size / (1024 * 1024)
        }
