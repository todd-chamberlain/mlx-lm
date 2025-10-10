"""
Layer Worker for Distributed Inference

Loads and processes specific layers of a model in a distributed pipeline.
"""

import asyncio
import logging
import time
from typing import Optional, Tuple
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from .model_partitioner import LayerAssignment, PipelineConfig
from .model_adapter import ModelAdapter

logger = logging.getLogger(__name__)


class LayerWorker:
    """Worker that processes specific model layers"""
    
    def __init__(
        self,
        node_id: str,
        model_path: str,
        assignment: LayerAssignment
    ):
        self.node_id = node_id
        self.model_path = model_path
        self.assignment = assignment
        
        self.model = None
        self.tokenizer = None
        self.adapter = None
        self.layers = None
        self.embedding = None
        self.norm = None
        
        logger.info(f"Initialized LayerWorker for {node_id}")
        logger.info(f"  Assigned layers: {assignment.start_layer}-{assignment.end_layer}")
    
    async def load_layers(self) -> bool:
        """
        Load only the assigned layers into memory.
        
        Uses ShardedModelLoader for memory-efficient layer loading.
        """
        try:
            logger.info(f"🚀 Loading SHARDED model from {self.model_path}")
            logger.info(f"  📊 Assigned layers: {self.assignment.start_layer} to {self.assignment.end_layer-1}")
            
            num_layers = self.assignment.end_layer - self.assignment.start_layer
            # Note: total_layers should be determined from model config, not hardcoded
            # This is a temporary value for logging purposes
            total_layers = self.assignment.total_layers if hasattr(self.assignment, 'total_layers') else 80
            percent = (num_layers / total_layers) * 100
            
            logger.info(f"  💾 Loading {num_layers}/{total_layers} layers ({percent:.0f}% of model)")
            logger.info(f"  ⚡ Expected RAM usage: ~{percent:.0f}% of full model")
            
            start_time = time.time()
            
            # Load full model for now - ShardedModelLoader will be integrated in Task 3
            # after bug fixes are applied
            self.model, self.tokenizer = load(self.model_path)
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f}s")
            logger.info(f"  Note: Currently loading full model - selective loading will be enabled in Task 3")
            
            # Create adapter for universal model handling
            self.adapter = ModelAdapter(self.model)
            model_info = self.adapter.get_model_info()
            logger.info(f"Model type: {model_info['model_type']}")
            logger.info(f"Model has {model_info['num_layers']} layers")
            logger.info(f"Uses tied embeddings: {model_info['uses_tied_embeddings']}")
            
            # Extract layers using adapter
            all_layers = self.adapter.get_layers()
            total_layers = len(all_layers)
            
            # Extract assigned layers
            self.layers = all_layers[self.assignment.start_layer:self.assignment.end_layer]
            logger.info(f"Extracted {len(self.layers)} layers ({self.assignment.start_layer}-{self.assignment.end_layer})")
            
            # Store embedding if first node
            if self.assignment.start_layer == 0:
                self.embedding = self.adapter.get_embedding()
                logger.info("Stored embedding layer (first node)")
            
            # Store norm if last node
            if self.assignment.end_layer >= total_layers:
                self.norm = self.adapter.get_norm()
                if self.norm:
                    logger.info("Stored final norm (last node)")
                logger.info("This is the last node in pipeline")
            
            return True
                
        except Exception as e:
            logger.error(f"Failed to load layers: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def forward(self, hidden_states: mx.array, cache=None) -> Tuple[mx.array, Optional[any]]:
        """
        Process hidden states through assigned layers.
        
        Args:
            hidden_states: Input activations
            cache: Optional cache for attention
            
        Returns:
            Tuple of (output_hidden_states, cache)
        """
        if self.layers is None:
            raise RuntimeError("Layers not loaded")
        
        logger.debug(f"Processing through {len(self.layers)} layers")
        logger.debug(f"Input shape: {hidden_states.shape}")
        
        # Process through each layer
        for i, layer in enumerate(self.layers):
            layer_idx = self.assignment.start_layer + i
            logger.debug(f"Processing layer {layer_idx}")
            
            # Forward through layer
            hidden_states = layer(hidden_states, cache=cache)
            
            # Handle tuple return (some layers return (hidden_states, cache))
            if isinstance(hidden_states, tuple):
                hidden_states, cache = hidden_states
        
        logger.debug(f"Output shape: {hidden_states.shape}")
        
        return hidden_states, cache
    
    def embed_tokens(self, input_ids: mx.array) -> mx.array:
        """
        Embed input tokens (first node only).
        
        Args:
            input_ids: Token IDs
            
        Returns:
            Embedded tokens
        """
        if self.embedding is None:
            raise RuntimeError("Embedding layer not available (not first node)")
        
        logger.debug(f"Embedding {len(input_ids)} tokens")
        return self.embedding(input_ids)
    
    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        """
        Compute output logits (last node only).
        
        Uses the model adapter for universal compatibility.
        
        Args:
            hidden_states: Final hidden states
            
        Returns:
            Logits
        """
        if self.adapter is None:
            raise RuntimeError("Model not loaded")
        
        if self.norm is None and self.assignment.end_layer < len(self.adapter.get_layers()):
            raise RuntimeError("Output computation not available (not last node)")
        
        logger.debug(f"Computing logits from hidden states shape: {hidden_states.shape}")
        
        # Use adapter to compute logits (handles all model types)
        logits = self.adapter.compute_logits(hidden_states)
        
        return logits
    
    def get_memory_usage(self) -> dict:
        """Get memory usage statistics"""
        if self.model is None:
            return {'loaded': False}
        
        # Get MLX memory stats
        memory_stats = mx.metal.get_active_memory() / 1024 / 1024 / 1024  # GB
        
        return {
            'loaded': True,
            'num_layers': len(self.layers) if self.layers else 0,
            'layer_range': f"{self.assignment.start_layer}-{self.assignment.end_layer}",
            'memory_gb': memory_stats,
            'has_embedding': self.embedding is not None,
            'has_norm': self.norm is not None,
            'model_type': self.adapter.model_type if self.adapter else 'unknown'
        }
    
    def __repr__(self) -> str:
        return f"LayerWorker(node={self.node_id}, layers={self.assignment.start_layer}-{self.assignment.end_layer})"
