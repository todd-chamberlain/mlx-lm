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


class ModelManager:
    """Manages MLX model loading, caching, and validation"""
    
    def __init__(self, model_cache_size: int = 2):
        self.model_cache_size = model_cache_size
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.model_lock = Lock()
        self.device_info = self._detect_device_info()
        
        logger.info(f"Initialized ModelManager with device: {self.device_info.model}")
        logger.info(f"Available memory: {self.device_info.memory_gb}GB")
        logger.info(f"GPU cores: {self.device_info.gpu_cores}")
    
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
    
    def load_model(self, model_path: str, model_name: Optional[str] = None) -> bool:
        """Load an MLX model with caching and validation"""
        if model_name is None:
            model_name = Path(model_path).name
        
        with self.model_lock:
            # Check if already loaded
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return True
            
            # Validate model path
            if not self.validate_model_path(model_path):
                return False
            
            # Check memory requirements
            estimated_memory = self.estimate_model_memory(model_path)
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
                model_info_dict = self.get_model_info(model_path) or {}
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
            return {
                'model_name': model_info.model_name,
                'model_path': model_info.model_path,
                'loaded_at': model_info.loaded_at,
                'memory_usage_gb': model_info.memory_usage_gb,
                'max_tokens': model_info.max_tokens,
                'context_length': model_info.context_length
            }
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        with self.model_lock:
            if model_name not in self.loaded_models:
                return False
            
            try:
                del self.loaded_models[model_name]
                # Force garbage collection
                import gc
                gc.collect()
                
                logger.info(f"Unloaded model: {model_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error unloading model {model_name}: {e}")
                return False
    
    def list_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded models with their info"""
        with self.model_lock:
            return {
                name: self.get_model_info_dict(name)
                for name in self.loaded_models.keys()
            }
    
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