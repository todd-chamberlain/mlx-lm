"""
Model Registry - Maps model names to local filesystem paths per worker.

This allows workers to know where models are stored on their local filesystem
when the coordinator requests them to load a specific model.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry that maps model names to local paths for each worker."""

    def __init__(self, worker_id: str, config_path: Optional[str] = None):
        """
        Initialize ModelRegistry.

        Args:
            worker_id: ID of this worker (e.g., "worker-studio1")
            config_path: Path to worker_models.yaml config file
        """
        self.worker_id = worker_id
        self.config_path = config_path or self._find_config_file()
        self.models: Dict[str, str] = {}

        if self.config_path:
            self._load_config()
        else:
            logger.warning(f"No model registry config found for worker {worker_id}")

    def _find_config_file(self) -> Optional[str]:
        """Find the worker_models.yaml config file."""
        # Try common locations
        possible_paths = [
            "config/worker_models.yaml",
            "../config/worker_models.yaml",
            "/etc/mlx-distributed/worker_models.yaml",
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path

        return None

    def _load_config(self):
        """Load model paths from config file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if 'models' not in config:
                logger.warning(f"No 'models' section in config file: {self.config_path}")
                return

            # Extract paths for this worker
            for model_name, worker_paths in config['models'].items():
                if isinstance(worker_paths, dict) and self.worker_id in worker_paths:
                    model_path = worker_paths[self.worker_id]
                    self.models[model_name] = model_path
                    logger.info(f"Registered model '{model_name}' -> {model_path}")

            logger.info(f"Loaded {len(self.models)} model paths for worker {self.worker_id}")

        except Exception as e:
            logger.error(f"Error loading model registry config: {e}")

    def get_model_path(self, model_name: str) -> Optional[str]:
        """
        Get the local filesystem path for a model.

        Args:
            model_name: Name of the model

        Returns:
            Local path to the model, or None if not found
        """
        path = self.models.get(model_name)

        if path:
            logger.debug(f"Resolved model '{model_name}' to path: {path}")
            return path
        else:
            logger.warning(f"Model '{model_name}' not found in registry for worker {self.worker_id}")
            return None

    def has_model(self, model_name: str) -> bool:
        """Check if worker has a registered path for this model."""
        return model_name in self.models

    def list_models(self) -> Dict[str, str]:
        """Get all registered models for this worker."""
        return self.models.copy()
