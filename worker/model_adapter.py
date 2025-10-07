"""
Model Adapter for Different Architectures

Handles differences between model architectures (Llama, Mistral, Qwen, Phi, etc.)
"""

import logging
from typing import Optional, Tuple
import mlx.core as mx

logger = logging.getLogger(__name__)


class ModelAdapter:
    """Adapts different model architectures for distributed inference"""
    
    def __init__(self, model, model_type: Optional[str] = None):
        self.model = model
        self.model_type = model_type or self._detect_model_type()
        logger.info(f"Initialized ModelAdapter for {self.model_type}")
    
    def _detect_model_type(self) -> str:
        """Detect model type from structure"""
        if hasattr(self.model, 'model_type'):
            return self.model.model_type
        
        # Check for common model structures
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'layers'):
                # Most transformer models
                return 'transformer'
        
        return 'unknown'
    
    def get_layers(self):
        """Get model layers"""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        else:
            raise ValueError("Cannot find model layers")
    
    def get_embedding(self):
        """Get embedding layer"""
        # Try common locations
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens
            elif hasattr(self.model.model, 'wte'):  # GPT-style
                return self.model.model.wte
            elif hasattr(self.model.model, 'token_embedding'):
                return self.model.model.token_embedding
        
        if hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens
        
        raise ValueError("Cannot find embedding layer")
    
    def get_norm(self):
        """Get final layer norm"""
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'norm'):
                return self.model.model.norm
            elif hasattr(self.model.model, 'ln_f'):  # GPT-style
                return self.model.model.ln_f
            elif hasattr(self.model.model, 'final_layernorm'):
                return self.model.model.final_layernorm
        
        if hasattr(self.model, 'norm'):
            return self.model.norm
        
        return None  # Some models don't have final norm
    
    def get_output_projection(self):
        """Get output projection layer (lm_head)"""
        # Try common locations
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head
        elif hasattr(self.model, 'output'):
            return self.model.output
        elif hasattr(self.model.model, 'lm_head'):
            return self.model.model.lm_head
        
        # Many models use tied embeddings (no separate lm_head)
        return None
    
    def compute_logits(self, hidden_states: mx.array) -> mx.array:
        """
        Compute output logits from hidden states.
        
        Handles different model architectures:
        - Models with separate lm_head
        - Models with tied embeddings
        - Quantized models
        """
        # Apply final norm if available
        norm = self.get_norm()
        if norm is not None:
            hidden_states = norm(hidden_states)
            logger.debug(f"Applied final norm, shape: {hidden_states.shape}")
        
        # Try to get output projection
        output_proj = self.get_output_projection()
        
        if output_proj is not None:
            # Model has separate output projection
            logger.debug("Using separate output projection")
            logits = output_proj(hidden_states)
        else:
            # Use tied embeddings
            logger.debug("Using tied embeddings for output projection")
            embed_layer = self.get_embedding()
            
            if hasattr(embed_layer, 'as_linear'):
                # Quantized embedding - use as_linear method
                logits = embed_layer.as_linear(hidden_states)
            elif hasattr(embed_layer, 'weight'):
                # Regular embedding - matrix multiply with transpose
                logits = hidden_states @ embed_layer.weight.T
            else:
                raise ValueError("Cannot compute logits: no output projection or embedding weights")
        
        logger.debug(f"Logits shape: {logits.shape}")
        return logits
    
    def get_model_info(self) -> dict:
        """Get model information"""
        layers = self.get_layers()
        
        return {
            'model_type': self.model_type,
            'num_layers': len(layers),
            'has_embedding': self.get_embedding() is not None,
            'has_norm': self.get_norm() is not None,
            'has_output_proj': self.get_output_projection() is not None,
            'uses_tied_embeddings': self.get_output_projection() is None
        }


# Model-specific configurations
MODEL_CONFIGS = {
    'llama': {
        'layers_path': 'model.layers',
        'embedding_path': 'model.embed_tokens',
        'norm_path': 'model.norm',
        'output_path': None,  # Uses tied embeddings
        'tied_embeddings': True
    },
    'mistral': {
        'layers_path': 'model.layers',
        'embedding_path': 'model.embed_tokens',
        'norm_path': 'model.norm',
        'output_path': None,  # Uses tied embeddings
        'tied_embeddings': True
    },
    'qwen': {
        'layers_path': 'model.layers',
        'embedding_path': 'model.embed_tokens',
        'norm_path': 'model.norm',
        'output_path': 'lm_head',
        'tied_embeddings': False
    },
    'phi': {
        'layers_path': 'model.layers',
        'embedding_path': 'model.embed_tokens',
        'norm_path': 'model.final_layernorm',
        'output_path': 'lm_head',
        'tied_embeddings': False
    },
    'gpt': {
        'layers_path': 'model.h',
        'embedding_path': 'model.wte',
        'norm_path': 'model.ln_f',
        'output_path': 'lm_head',
        'tied_embeddings': False
    }
}


def get_model_config(model_type: str) -> dict:
    """Get configuration for a specific model type"""
    model_type_lower = model_type.lower()
    
    for key in MODEL_CONFIGS:
        if key in model_type_lower:
            return MODEL_CONFIGS[key]
    
    # Default to llama-style
    logger.warning(f"Unknown model type {model_type}, using llama-style config")
    return MODEL_CONFIGS['llama']
