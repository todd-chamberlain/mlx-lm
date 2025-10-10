"""
MLX Model Loading and Management

This module handles MLX model loading, caching, validation, and device detection
for the distributed inference worker.
"""

import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from threading import Lock

import mlx.core as mx
import mlx_lm
from transformers import AutoTokenizer
import psutil

from .model_loader import ModelLoader, LayerAssignment

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Device information for capability reporting"""
    model: str
    memory_gb: int
    gpu_cores: int
    os_version: str
    mlx_version: str


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_path: str
    model_name: str
    model: Any
    tokenizer: Any
    loaded_at: float
    memory_usage_gb: float
    max_tokens: int = 4096
    context_length: int = 4096
    shard_metadata: Optional[Dict[str, Any]] = None
    
    @property
    def is_sharded(self) -> bool:
        """Check if this model is sharded"""
        return self.shard_metadata is not None and len(self.shard_metadata) > 0


class ModelManager:
    """Manages MLX model loading, caching, and validation"""
    
    def __init__(self, model_cache_size: int = 2, enable_fallback: bool = True):
        self.model_cache_size = model_cache_size
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_lock = Lock()
        self.device_info = self._detect_device_info()
        self.model_loader = ModelLoader()
        self.enable_fallback = enable_fallback
        
        # Fallback metrics
        self.fallback_count = 0
        self.sharded_load_failures = 0
        self.sharded_load_successes = 0
        
        logger.info(f"Initialized ModelManager with device: {self.device_info.model}")
        logger.info(f"Available memory: {self.device_info.memory_gb}GB")
        logger.info(f"GPU cores: {self.device_info.gpu_cores}")
        logger.info(f"Fallback enabled: {self.enable_fallback}")
    
    def _detect_device_info(self) -> DeviceInfo:
        """Detect Apple Silicon device capabilities"""
        try:
            # Get device model
            model_output = subprocess.check_output(['sysctl', '-n', 'hw.model'], text=True).strip()
            
            # Get memory in bytes and convert to GB
            memory_bytes = int(subprocess.check_output(['sysctl', '-n', 'hw.memsize'], text=True).strip())
            memory_gb = int(memory_bytes / (1024**3))
            
            # Get OS version
            os_version = subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip()
            
            # Get MLX version
            mlx_version = mx.__version__
            
            # Estimate GPU cores based on chip type
            gpu_cores = self._estimate_gpu_cores(model_output)
            
            return DeviceInfo(
                model=model_output,
                memory_gb=memory_gb,
                gpu_cores=gpu_cores,
                os_version=os_version,
                mlx_version=mlx_version
            )
            
        except Exception as e:
            logger.error(f"Failed to detect device info: {e}")
            # Fallback values
            return DeviceInfo(
                model="Unknown",
                memory_gb=16,
                gpu_cores=8,
                os_version="Unknown",
                mlx_version=mx.__version__
            )
    
    def _estimate_gpu_cores(self, model: str) -> int:
        """Estimate GPU cores based on Apple Silicon chip type"""
        model_lower = model.lower()
        
        # M1 series
        if 'm1' in model_lower:
            if 'max' in model_lower:
                return 32  # M1 Max
            elif 'pro' in model_lower:
                return 16  # M1 Pro
            else:
                return 8   # M1
        
        # M2 series
        elif 'm2' in model_lower:
            if 'ultra' in model_lower:
                return 76  # M2 Ultra
            elif 'max' in model_lower:
                return 38  # M2 Max
            elif 'pro' in model_lower:
                return 19  # M2 Pro
            else:
                return 10  # M2
        
        # M3 series
        elif 'm3' in model_lower:
            if 'max' in model_lower:
                return 40  # M3 Max
            elif 'pro' in model_lower:
                return 18  # M3 Pro
            else:
                return 10  # M3
        
        # Default fallback
        return 8
    
    def validate_model_path(self, model_path: str) -> bool:
        """Validate that a model path exists and contains required files"""
        try:
            path = Path(model_path)
            if not path.exists():
                logger.error(f"Model path does not exist: {model_path}")
                return False
            
            # Check for required MLX model files
            required_files = ['config.json']
            optional_files = ['tokenizer.json', 'tokenizer_config.json', 'weights.npz']
            
            existing_files = [f.name for f in path.iterdir() if f.is_file()]
            
            # Check required files
            for required_file in required_files:
                if required_file not in existing_files:
                    logger.error(f"Missing required file: {required_file} in {model_path}")
                    return False
            
            # Check for at least some model weights
            weight_files = [f for f in existing_files if f.endswith(('.npz', '.safetensors'))]
            if not weight_files:
                logger.error(f"No weight files found in {model_path}")
                return False
            
            logger.info(f"Model validation passed for: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error validating model path {model_path}: {e}")
            return False
    
    def get_model_info(self, model_path: str) -> Optional[Dict[str, Any]]:
        """Get model configuration information"""
        try:
            config_path = Path(model_path) / 'config.json'
            if not config_path.exists():
                return None
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            return {
                'model_type': config.get('model_type', 'unknown'),
                'vocab_size': config.get('vocab_size', 0),
                'hidden_size': config.get('hidden_size', 0),
                'num_layers': config.get('num_hidden_layers', 0),
                'max_position_embeddings': config.get('max_position_embeddings', 2048),
                'architectures': config.get('architectures', [])
            }
            
        except Exception as e:
            logger.error(f"Error reading model config: {e}")
            return None
    
    def estimate_model_memory(self, model_path: str) -> float:
        """Estimate model memory usage in GB"""
        try:
            # Get model info
            model_info = self.get_model_info(model_path)
            if not model_info:
                return 4.0  # Default estimate
            
            # Rough estimation based on parameters
            vocab_size = model_info.get('vocab_size', 32000)
            hidden_size = model_info.get('hidden_size', 4096)
            num_layers = model_info.get('num_layers', 32)
            
            # Estimate parameters (very rough)
            embedding_params = vocab_size * hidden_size
            layer_params = num_layers * (hidden_size * hidden_size * 4)  # Simplified
            total_params = embedding_params + layer_params
            
            # Assume 4-bit quantization (0.5 bytes per parameter) + overhead
            memory_gb = (total_params * 0.5) / (1024**3) * 1.5  # 50% overhead
            
            return max(memory_gb, 1.0)  # Minimum 1GB
            
        except Exception as e:
            logger.error(f"Error estimating model memory: {e}")
            return 4.0  # Default fallback
    
    def load_model(
        self,
        model_path: str,
        model_name: Optional[str] = None,
        layer_assignment: Optional[LayerAssignment] = None
    ) -> bool:
        """
        Load an MLX model with caching and validation.
        
        Args:
            model_path: Path or HuggingFace model name
            model_name: Optional name for the model
            layer_assignment: Optional layer assignment for sharded loading
            
        Returns:
            True if successful, False otherwise
        """
        if model_name is None:
            model_name = Path(model_path).name
        
        # If layer assignment provided, use sharded loading
        if layer_assignment is not None:
            return self.load_sharded_model(model_path, layer_assignment, model_name)
        
        with self.model_lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Validate model path if it's a local path
            if os.path.exists(model_path) and not self.validate_model_path(model_path):
                return False
            
            # Check memory requirements
            if os.path.exists(model_path):
                estimated_memory = self.estimate_model_memory(model_path)
            else:
                # For HuggingFace models, use a default estimate
                estimated_memory = 4.0
            
            available_memory = self._get_available_memory()
            
            if estimated_memory > available_memory:
                logger.error(f"Insufficient memory for model {model_name}: "
                           f"need {estimated_memory:.1f}GB, have {available_memory:.1f}GB")
                
                # Try to free up space by unloading old models
                if not self._free_memory_for_model(estimated_memory):
                    return False
            
            try:
                logger.info(f"Loading model: {model_name} from {model_path}")
                start_time = time.time()
                
                # Load model and tokenizer using MLX-LM
                model, tokenizer = mlx_lm.load(model_path)
                
                load_time = time.time() - start_time
                actual_memory = self._measure_model_memory()
                
                # Get model configuration
                if os.path.exists(model_path):
                    model_info_dict = self.get_model_info(model_path) or {}
                else:
                    model_info_dict = {}
                max_tokens = model_info_dict.get('max_position_embeddings', 4096)
                
                # Store model info
                model_info = ModelInfo(
                    model_path=model_path,
                    model_name=model_name,
                    model=model,
                    tokenizer=tokenizer,
                    loaded_at=time.time(),
                    memory_usage_gb=actual_memory,
                    max_tokens=max_tokens,
                    context_length=max_tokens
                )
                
                self.loaded_models[model_name] = model_info
                
                logger.info(f"Successfully loaded model {model_name} in {load_time:.2f}s")
                logger.info(f"Model memory usage: {actual_memory:.2f}GB")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return False
    
    def load_sharded_model(
        self,
        model_name: str,
        layer_assignment: LayerAssignment,
        model_alias: Optional[str] = None
    ) -> bool:
        """
        Load a sharded model with only assigned layers.
        
        This method coordinates the download and loading of a model shard,
        ensuring proper metadata is populated and validation is performed.
        
        Args:
            model_name: HuggingFace model name or path
            layer_assignment: Layer assignment for this worker
            model_alias: Optional alias for the model (defaults to model_name)
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            ValueError: If layer assignment is invalid
        """
        if model_alias is None:
            model_alias = model_name
        
        with self.model_lock:
            # Check if already loaded
            if model_alias in self.loaded_models:
                logger.info(f"Sharded model {model_alias} already loaded")
                return True
            
            try:
                # Validate layer assignment
                if layer_assignment.start_layer < 0:
                    raise ValueError(f"Invalid start_layer: {layer_assignment.start_layer} (must be >= 0)")
                
                if layer_assignment.end_layer <= layer_assignment.start_layer:
                    raise ValueError(
                        f"Invalid layer range: {layer_assignment.start_layer}-{layer_assignment.end_layer} "
                        f"(end_layer must be > start_layer)"
                    )
                
                logger.info(
                    f"Loading sharded model: {model_name}, "
                    f"layers {layer_assignment.start_layer}-{layer_assignment.end_layer}",
                    extra={
                        "model_name": model_name,
                        "model_alias": model_alias,
                        "worker_id": layer_assignment.worker_id,
                        "start_layer": layer_assignment.start_layer,
                        "end_layer": layer_assignment.end_layer,
                        "operation": "sharded_model_load_start"
                    }
                )
                start_time = time.time()
                
                # Download shard if needed
                download_success = self.model_loader.download_shard(
                    model_name,
                    layer_assignment
                )
                
                if not download_success:
                    error_msg = f"Failed to download shard for {model_name}"
                    logger.error(error_msg, extra={
                        "model_name": model_name,
                        "worker_id": layer_assignment.worker_id,
                        "operation": "shard_download_failed"
                    })
                    return False
                
                # Load partial model using LayerWorker
                from .layer_worker import LayerWorker
                
                layer_worker = LayerWorker(
                    node_id=layer_assignment.worker_id,
                    model_path=model_name,
                    assignment=layer_assignment
                )
                
                # Load layers asynchronously (but wait for completion)
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                load_success = loop.run_until_complete(layer_worker.load_layers())
                loop.close()
                
                if not load_success:
                    error_msg = f"Failed to load layers for {model_name}"
                    logger.error(error_msg, extra={
                        "model_name": model_name,
                        "worker_id": layer_assignment.worker_id,
                        "operation": "layer_load_failed"
                    })
                    return False
                
                load_time = time.time() - start_time
                actual_memory = self._measure_model_memory()
                
                # ============================================================
                # SHARD METADATA CREATION: Model Info Population
                # ============================================================
                # This is where shard metadata is created and attached to the
                # ModelInfo object. This metadata is used by:
                #
                # 1. InferenceEngine to determine routing (is_sharded flag)
                # 2. DistributedForwardPass to build pipeline configuration
                # 3. Coordinator to track shard status
                #
                # Key fields:
                # - start_layer/end_layer: Layer range this worker handles
                # - is_first_worker: True if start_layer == 0 (can initiate inference)
                # - is_last_worker: True if no next worker (returns final output)
                # - next_worker_id/address: Next worker in pipeline
                # ============================================================
                
                # Create shard metadata with proper validation
                shard_metadata = {
                    'start_layer': layer_assignment.start_layer,
                    'end_layer': layer_assignment.end_layer,
                    'worker_id': layer_assignment.worker_id,
                    'next_worker_id': layer_assignment.next_worker_id,
                    'next_worker_address': layer_assignment.next_worker_address,
                    'is_first_worker': layer_assignment.start_layer == 0,
                    'is_last_worker': layer_assignment.next_worker_id is None or layer_assignment.next_worker_id == "",
                    'total_layers': layer_assignment.end_layer - layer_assignment.start_layer,
                }
                
                # Validate shard metadata
                if not self._validate_shard_metadata(shard_metadata):
                    error_msg = f"Invalid shard metadata for {model_name}"
                    logger.error(error_msg, extra={
                        "model_name": model_name,
                        "shard_metadata": shard_metadata,
                        "operation": "shard_metadata_validation_failed"
                    })
                    return False
                
                # Store model info with shard metadata
                model_info = ModelInfo(
                    model_path=model_name,
                    model_name=model_alias,
                    model=layer_worker.model,
                    tokenizer=layer_worker.tokenizer,
                    loaded_at=time.time(),
                    memory_usage_gb=actual_memory,
                    max_tokens=4096,  # Default, can be updated
                    context_length=4096,
                    shard_metadata=shard_metadata
                )
                
                self.loaded_models[model_alias] = model_info
                
                logger.info(
                    f"Successfully loaded sharded model {model_alias} in {load_time:.2f}s",
                    extra={
                        "model_name": model_name,
                        "model_alias": model_alias,
                        "worker_id": layer_assignment.worker_id,
                        "load_time_seconds": load_time,
                        "memory_usage_gb": actual_memory,
                        "start_layer": layer_assignment.start_layer,
                        "end_layer": layer_assignment.end_layer,
                        "operation": "sharded_model_load_complete"
                    }
                )
                
                self.sharded_load_successes += 1
                return True
                
            except ValueError as e:
                self.sharded_load_failures += 1
                logger.error(
                    f"Validation error loading sharded model {model_name}: {e}",
                    extra={
                        "model_name": model_name,
                        "error": str(e),
                        "operation": "sharded_model_load_validation_error"
                    }
                )
                
                # Attempt fallback if enabled
                if self.enable_fallback:
                    return self._fallback_to_full_model(model_name, model_alias, "validation_error", str(e))
                return False
                
            except Exception as e:
                self.sharded_load_failures += 1
                logger.error(
                    f"Failed to load sharded model {model_name}: {e}",
                    extra={
                        "model_name": model_name,
                        "error": str(e),
                        "operation": "sharded_model_load_failed"
                    }
                )
                import traceback
                traceback.print_exc()
                
                # Attempt fallback if enabled
                if self.enable_fallback:
                    return self._fallback_to_full_model(model_name, model_alias, "load_error", str(e))
                return False
    
    def _fallback_to_full_model(
        self,
        model_name: str,
        model_alias: str,
        failure_reason: str,
        error_message: str
    ) -> bool:
        """
        Fallback to loading the full model when sharded loading fails.
        
        Args:
            model_name: HuggingFace model name or path
            model_alias: Alias for the model
            failure_reason: Reason for fallback (e.g., "validation_error", "load_error")
            error_message: Original error message
            
        Returns:
            True if fallback successful, False otherwise
        """
        self.fallback_count += 1
        
        logger.warning(
            f"Attempting fallback to full model loading for {model_name}",
            extra={
                "model_name": model_name,
                "model_alias": model_alias,
                "failure_reason": failure_reason,
                "original_error": error_message,
                "fallback_count": self.fallback_count,
                "operation": "fallback_to_full_model_start"
            }
        )
        
        try:
            # Attempt to load full model without layer assignment
            success = self.load_model(model_name, model_alias, layer_assignment=None)
            
            if success:
                logger.warning(
                    f"Successfully loaded full model {model_alias} as fallback",
                    extra={
                        "model_name": model_name,
                        "model_alias": model_alias,
                        "failure_reason": failure_reason,
                        "fallback_count": self.fallback_count,
                        "operation": "fallback_to_full_model_success"
                    }
                )
                return True
            else:
                logger.error(
                    f"Fallback to full model failed for {model_name}",
                    extra={
                        "model_name": model_name,
                        "model_alias": model_alias,
                        "failure_reason": failure_reason,
                        "operation": "fallback_to_full_model_failed"
                    }
                )
                return False
                
        except Exception as e:
            logger.error(
                f"Exception during fallback to full model for {model_name}: {e}",
                extra={
                    "model_name": model_name,
                    "model_alias": model_alias,
                    "failure_reason": failure_reason,
                    "fallback_error": str(e),
                    "operation": "fallback_to_full_model_exception"
                }
            )
            return False
    
    def get_fallback_metrics(self) -> Dict[str, int]:
        """
        Get metrics about fallback usage.
        
        Returns:
            Dictionary with fallback metrics
        """
        return {
            'fallback_count': self.fallback_count,
            'sharded_load_failures': self.sharded_load_failures,
            'sharded_load_successes': self.sharded_load_successes,
            'fallback_rate': (
                self.fallback_count / max(1, self.sharded_load_failures + self.sharded_load_successes)
            ) if (self.sharded_load_failures + self.sharded_load_successes) > 0 else 0.0
        }
    
    def _validate_shard_metadata(self, shard_metadata: Dict[str, Any]) -> bool:
        """
        Validate shard metadata for correctness.
        
        Args:
            shard_metadata: Shard metadata dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        required_fields = [
            'start_layer', 'end_layer', 'worker_id',
            'is_first_worker', 'is_last_worker', 'total_layers'
        ]
        
        # Check required fields exist
        for field in required_fields:
            if field not in shard_metadata:
                logger.error(f"Missing required field in shard_metadata: {field}")
                return False
        
        # Validate layer range
        if shard_metadata['start_layer'] < 0:
            logger.error(f"Invalid start_layer: {shard_metadata['start_layer']}")
            return False
        
        if shard_metadata['end_layer'] <= shard_metadata['start_layer']:
            logger.error(
                f"Invalid layer range: {shard_metadata['start_layer']}-{shard_metadata['end_layer']}"
            )
            return False
        
        # Validate total_layers matches range
        expected_total = shard_metadata['end_layer'] - shard_metadata['start_layer']
        if shard_metadata['total_layers'] != expected_total:
            logger.error(
                f"total_layers mismatch: expected {expected_total}, got {shard_metadata['total_layers']}"
            )
            return False
        
        # Validate worker_id is not empty
        if not shard_metadata['worker_id']:
            logger.error("worker_id cannot be empty")
            return False
        
        # Validate pipeline consistency
        if not shard_metadata['is_last_worker']:
            if not shard_metadata.get('next_worker_id'):
                logger.error("Non-last worker must have next_worker_id")
                return False
            if not shard_metadata.get('next_worker_address'):
                logger.error("Non-last worker must have next_worker_address")
                return False
        
        return True
    
    def get_model(self, model_name: str) -> Optional[Tuple[Any, Any]]:
        """Get a loaded model and tokenizer"""
        with self.model_lock:
            if model_name not in self.loaded_models:
                return None
            
            model_info = self.loaded_models[model_name]
            return model_info.model, model_info.tokenizer
    
    def get_model_info_dict(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information as dictionary"""
        with self.model_lock:
            if model_name not in self.loaded_models:
                return None
            
            model_info = self.loaded_models[model_name]
            info_dict = {
                'model_name': model_info.model_name,
                'model_path': model_info.model_path,
                'loaded_at': model_info.loaded_at,
                'memory_usage_gb': model_info.memory_usage_gb,
                'max_tokens': model_info.max_tokens,
                'context_length': model_info.context_length,
                'is_sharded': model_info.is_sharded
            }
            
            # Include shard metadata if present
            if model_info.shard_metadata:
                info_dict['shard_metadata'] = model_info.shard_metadata
            
            return info_dict
    
    def unload_model(self, model_name: str) -> bool:
        """
        Unload a model to free memory.
        Handles both regular and sharded models.
        
        Args:
            model_name: Name of the model to unload
            
        Returns:
            True if successful, False otherwise
        """
        with self.model_lock:
            if model_name not in self.loaded_models:
                logger.warning(f"Model {model_name} not loaded, nothing to unload")
                return False
            
            try:
                model_info = self.loaded_models[model_name]
                is_sharded = model_info.is_sharded
                
                if is_sharded:
                    logger.info(
                        f"Unloading sharded model {model_name} "
                        f"(layers {model_info.shard_metadata['start_layer']}-"
                        f"{model_info.shard_metadata['end_layer']})"
                    )
                    
                    # Clean up LayerWorker resources
                    # The model reference in model_info.model is the LayerWorker's model
                    # Deleting it will trigger cleanup
                    if hasattr(model_info.model, 'cleanup'):
                        try:
                            model_info.model.cleanup()
                        except Exception as e:
                            logger.warning(f"Error calling cleanup on model: {e}")
                else:
                    logger.info(f"Unloading model: {model_name}")
                
                # Remove from loaded models
                del self.loaded_models[model_name]
                
                # Force garbage collection to free memory
                import gc
                gc.collect()
                
                logger.info(f"Successfully unloaded model: {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded models with their info"""
        with self.model_lock:
            logger.info(f"[DEBUG] list_loaded_models called. loaded_models keys: {list(self.loaded_models.keys())}")
            return {
                name: self.get_model_info_dict(name)
                for name in self.loaded_models.keys()
            }
    
    def set_shard_metadata(
        self,
        model_name: str,
        shard_metadata: Dict[str, Any]
    ) -> bool:
        """
        Set shard metadata for a loaded model.
        
        Args:
            model_name: Name of the model
            shard_metadata: Shard metadata dictionary containing:
                - start_layer: Starting layer index
                - end_layer: Ending layer index (exclusive)
                - next_worker_id: ID of next worker in pipeline
                - next_worker_address: Address of next worker
                
        Returns:
            True if metadata was set, False if model not found
        """
        with self.model_lock:
            if model_name not in self.loaded_models:
                logger.warning(f"Cannot set shard metadata: model {model_name} not loaded")
                return False
            
            model_info = self.loaded_models[model_name]
            model_info.shard_metadata = shard_metadata
            
            logger.info(
                f"Set shard metadata for {model_name}: "
                f"layers {shard_metadata.get('start_layer')}-{shard_metadata.get('end_layer')}"
            )
            return True
    
    def _get_available_memory(self) -> float:
        """Get available system memory in GB"""
        try:
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            return available_gb
        except Exception:
            return 8.0  # Conservative fallback
    
    def _measure_model_memory(self) -> float:
        """Measure current memory usage in GB"""
        try:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024**2)
            return memory_mb / 1024
        except Exception:
            return 2.0  # Fallback estimate
    
    def _free_memory_for_model(self, required_memory_gb: float) -> bool:
        """Free memory by unloading old models"""
        if len(self.loaded_models) == 0:
            return False
        
        # Sort models by last access time (oldest first)
        models_by_age = sorted(
            self.loaded_models.items(),
            key=lambda x: x[1].loaded_at
        )
        
        freed_memory = 0.0
        models_to_unload = []
        
        for model_name, model_info in models_by_age:
            if len(self.loaded_models) - len(models_to_unload) <= 1:
                # Keep at least one model loaded
                break
            
            models_to_unload.append(model_name)
            freed_memory += model_info.memory_usage_gb
            
            if freed_memory >= required_memory_gb:
                break
        
        # Unload selected models
        for model_name in models_to_unload:
            self.unload_model(model_name)
        
        return freed_memory >= required_memory_gb
    
    def get_device_info(self) -> DeviceInfo:
        """Get device information"""
        return self.device_info
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_gb': memory.used / (1024**3),
                'percent': memory.percent
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {
                'total_gb': 16.0,
                'available_gb': 8.0,
                'used_gb': 8.0,
                'percent': 50.0
            }