"""
Model Loader for Distributed Inference

This module handles downloading and loading models for distributed inference,
supporting both full model loading and partial layer loading (sharding).
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass

import mlx.core as mx
from huggingface_hub import hf_hub_download, model_info, list_repo_files
from transformers import AutoTokenizer

from common.logging_system import get_logger

logger = get_logger(__name__)


@dataclass
class LayerAssignment:
    """
    Layer assignment for a worker in the distributed inference pipeline.
    
    Attributes:
        worker_id: Unique identifier for this worker
        start_layer: Starting layer index (inclusive)
        end_layer: Ending layer index (exclusive)
        next_worker_id: ID of the next worker in the pipeline (None if last)
        next_worker_address: gRPC address of the next worker (None if last)
    """
    worker_id: str
    start_layer: int
    end_layer: int
    next_worker_id: Optional[str] = None
    next_worker_address: Optional[str] = None


class ModelLoader:
    """
    Loads models for distributed inference with support for layer-level sharding.
    
    This class handles:
    - Downloading model files from HuggingFace Hub
    - Loading full models or partial models with specific layer assignments
    - Progress tracking and structured logging
    - Memory-efficient loading for distributed inference
    
    When a LayerAssignment is provided, only the assigned layers are loaded,
    reducing memory usage per worker. Without a LayerAssignment, the full
    model is loaded.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model loader.
        
        Args:
            cache_dir: Directory for caching downloaded files.
                      Defaults to HF_HOME environment variable or ~/.cache/huggingface
        """
        self.cache_dir = cache_dir or os.getenv('HF_HOME', os.path.expanduser('~/.cache/huggingface'))
        logger.info(f"Initialized ModelLoader with cache_dir: {self.cache_dir}")
    
    def get_layer_count(self, model_name: str) -> int:
        """
        Get the total number of layers in a model by reading config.json.
        
        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            
        Returns:
            Number of layers in the model
            
        Raises:
            ValueError: If config cannot be read or layer count not found
        """
        try:
            logger.info(f"Fetching layer count for model: {model_name}")
            
            # Download config.json
            config_path = hf_hub_download(
                repo_id=model_name,
                filename="config.json",
                cache_dir=self.cache_dir
            )
            
            # Read config
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Try different config keys for layer count
            layer_count = (
                config.get('num_hidden_layers') or
                config.get('n_layer') or
                config.get('num_layers')
            )
            
            if layer_count is None:
                raise ValueError(f"Could not find layer count in config for {model_name}")
            
            logger.info(f"Model {model_name} has {layer_count} layers")
            return layer_count
            
        except Exception as e:
            logger.error(f"Failed to get layer count for {model_name}: {e}")
            raise ValueError(f"Failed to get layer count: {e}") from e
    
    def _get_layer_weight_files(self, model_name: str, layer_assignment: LayerAssignment) -> List[str]:
        """
        Determine which weight files contain the assigned layers.
        
        Args:
            model_name: HuggingFace model name
            layer_assignment: Layer assignment for this worker
            
        Returns:
            List of weight filenames to download
        """
        try:
            # List all files in the repository
            all_files = list_repo_files(repo_id=model_name)
            
            # Filter for weight files
            weight_files = [
                f for f in all_files
                if f.endswith(('.safetensors', '.bin', '.npz'))
            ]
            
            logger.info(f"Found {len(weight_files)} weight files in {model_name}")
            
            # Filter for layer-specific files if possible
            layer_specific_files = []
            for wf in weight_files:
                # Check if filename contains layer numbers
                filename_lower = wf.lower()
                
                # Try to extract layer numbers from filename
                # Common patterns: "layer.0", "layers.0", "model.layers.0"
                for layer_idx in range(layer_assignment.start_layer, layer_assignment.end_layer):
                    if f"layer.{layer_idx}" in filename_lower or f"layers.{layer_idx}" in filename_lower:
                        layer_specific_files.append(wf)
                        break
            
            # If we found layer-specific files, use those
            if layer_specific_files:
                logger.info(f"Found {len(layer_specific_files)} layer-specific weight files")
                return layer_specific_files
            
            # Otherwise, return all weight files (we'll filter during loading)
            # Note: This is not optimal but necessary for models without layer-specific files
            logger.info(f"No layer-specific files found, will download all {len(weight_files)} weight files")
            return weight_files
            
        except Exception as e:
            logger.error(f"Failed to list weight files for {model_name}: {e}")
            raise
    
    def _get_required_files(self, model_name: str, layer_assignment: LayerAssignment) -> List[str]:
        """
        Get list of all required files for loading a sharded model.
        
        Args:
            model_name: HuggingFace model name
            layer_assignment: Layer assignment for this worker
            
        Returns:
            List of filenames to download
        """
        required_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "tokenizer.model",  # For some models
        ]
        
        # Add weight files for assigned layers
        weight_files = self._get_layer_weight_files(model_name, layer_assignment)
        required_files.extend(weight_files)
        
        return required_files
    
    def download_shard(
        self,
        model_name: str,
        layer_assignment: LayerAssignment,
        progress_callback: Optional[callable] = None
    ) -> bool:
        """
        Download the files needed for assigned layers.
        
        Downloads model configuration, tokenizer files, and weight files needed
        for the specified layer range. Provides progress tracking via callback.
        
        Args:
            model_name: HuggingFace model name (e.g., "meta-llama/Llama-2-7b-hf")
            layer_assignment: Layer assignment specifying which layers to download
            progress_callback: Optional callback function(filename, progress_pct) for progress updates
            
        Returns:
            True if download successful, False otherwise
            
        Raises:
            Exception: If critical files cannot be downloaded
        """
        try:
            logger.info(
                f"Starting shard download for {model_name}",
                extra={
                    "model_name": model_name,
                    "worker_id": layer_assignment.worker_id,
                    "start_layer": layer_assignment.start_layer,
                    "end_layer": layer_assignment.end_layer,
                    "num_layers": layer_assignment.end_layer - layer_assignment.start_layer,
                    "layer_range": f"{layer_assignment.start_layer}-{layer_assignment.end_layer}",
                    "operation": "shard_download_start"
                }
            )
            
            # Get list of required files
            required_files = self._get_required_files(model_name, layer_assignment)
            logger.info(f"Need to download {len(required_files)} files")
            
            # Download each file
            downloaded_count = 0
            for filename in required_files:
                try:
                    logger.debug(f"Downloading {filename}...")
                    
                    # Download file
                    file_path = hf_hub_download(
                        repo_id=model_name,
                        filename=filename,
                        cache_dir=self.cache_dir
                    )
                    
                    downloaded_count += 1
                    progress_pct = (downloaded_count / len(required_files)) * 100
                    
                    logger.debug(f"Downloaded {filename} ({progress_pct:.1f}% complete)")
                    
                    # Call progress callback if provided
                    if progress_callback:
                        progress_callback(filename, progress_pct)
                        
                except Exception as e:
                    # Some files might not exist (e.g., tokenizer.model for some models)
                    # Log but continue
                    logger.warning(f"Could not download {filename}: {e}")
                    continue
            
            logger.info(
                f"Shard download complete for {model_name}",
                extra={
                    "model_name": model_name,
                    "worker_id": layer_assignment.worker_id,
                    "start_layer": layer_assignment.start_layer,
                    "end_layer": layer_assignment.end_layer,
                    "layer_range": f"{layer_assignment.start_layer}-{layer_assignment.end_layer}",
                    "files_downloaded": downloaded_count,
                    "total_files": len(required_files),
                    "operation": "shard_download_complete"
                }
            )
            return True
            
        except Exception as e:
            logger.error(
                f"Failed to download shard for {model_name}: {e}",
                extra={
                    "model_name": model_name,
                    "worker_id": layer_assignment.worker_id,
                    "error": str(e),
                    "operation": "shard_download_failed"
                }
            )
            return False
    
    def load_partial_model(
        self,
        model_name: str,
        layer_assignment: LayerAssignment
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Load a partial model with only assigned layers.
        
        This method loads the full model using mlx_lm and returns it along with
        shard metadata. The actual layer extraction is handled by LayerWorker.
        
        Note: Currently loads the full model. Future optimization could load
        only required weight files for better memory efficiency during loading.
        
        Args:
            model_name: HuggingFace model name or local path
            layer_assignment: Layer assignment specifying which layers this worker handles
            
        Returns:
            Tuple of (model, tokenizer, shard_metadata) where:
                - model: The loaded MLX model (full model, to be filtered by LayerWorker)
                - tokenizer: The model's tokenizer
                - shard_metadata: Dictionary containing layer assignment and pipeline info
            
        Raises:
            RuntimeError: If model loading fails
        """
        try:
            logger.info(
                f"Loading partial model {model_name}, "
                f"layers {layer_assignment.start_layer}-{layer_assignment.end_layer}",
                extra={
                    "model_name": model_name,
                    "worker_id": layer_assignment.worker_id,
                    "start_layer": layer_assignment.start_layer,
                    "end_layer": layer_assignment.end_layer,
                    "operation": "partial_model_load_start"
                }
            )
            
            # Load the full model using mlx_lm
            # The LayerWorker will extract only the assigned layers
            from mlx_lm import load
            
            model, tokenizer = load(model_name)
            
            # Create shard metadata with correct field names
            shard_metadata = {
                'start_layer': layer_assignment.start_layer,
                'end_layer': layer_assignment.end_layer,
                'worker_id': layer_assignment.worker_id,
                'next_worker_id': layer_assignment.next_worker_id,
                'next_worker_address': layer_assignment.next_worker_address,
                'is_first_worker': layer_assignment.start_layer == 0,
                'is_last_worker': layer_assignment.next_worker_id is None or layer_assignment.next_worker_id == "",
            }
            
            logger.info(
                f"Successfully loaded partial model {model_name}",
                extra={
                    "model_name": model_name,
                    "worker_id": layer_assignment.worker_id,
                    "shard_metadata": shard_metadata,
                    "operation": "partial_model_load_complete"
                }
            )
            
            return model, tokenizer, shard_metadata
            
        except Exception as e:
            logger.error(
                f"Failed to load partial model {model_name}: {e}",
                extra={
                    "model_name": model_name,
                    "worker_id": layer_assignment.worker_id,
                    "error": str(e),
                    "operation": "partial_model_load_failed"
                }
            )
            raise RuntimeError(f"Failed to load partial model: {e}") from e
    
    def get_download_progress(self, model_name: str) -> Dict[str, Any]:
        """
        Get download progress for a model.
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            Dictionary with progress information
        """
        # Check if model exists in cache
        model_cache_path = Path(self.cache_dir) / "hub" / f"models--{model_name.replace('/', '--')}"
        
        if not model_cache_path.exists():
            return {
                'status': 'not_started',
                'progress_percent': 0.0,
                'downloaded_files': 0,
                'total_files': 0
            }
        
        # Count downloaded files
        downloaded_files = list(model_cache_path.rglob('*'))
        downloaded_files = [f for f in downloaded_files if f.is_file()]
        
        return {
            'status': 'complete' if len(downloaded_files) > 0 else 'not_started',
            'progress_percent': 100.0 if len(downloaded_files) > 0 else 0.0,
            'downloaded_files': len(downloaded_files),
            'total_files': len(downloaded_files)  # We don't know total until we start
        }
