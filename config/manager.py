"""
Configuration Manager with hot-reloading and validation capabilities.
"""

import os
import asyncio
import logging
import threading
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Type, TypeVar
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import yaml

from .base import BaseConfig
from .api_config import APIConfig
from .coordinator_config import CoordinatorConfig
from .worker_config import WorkerConfig

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseConfig)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration file changes."""
    
    def __init__(self, config_manager: 'ConfigManager'):
        self.config_manager = config_manager
        self.debounce_delay = 1.0  # seconds
        self._reload_timer: Optional[threading.Timer] = None
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml'] and file_path.name in self.config_manager._watched_files:
            logger.debug(f"Configuration file modified: {file_path}")
            self._debounced_reload(file_path)
    
    def _debounced_reload(self, file_path: Path):
        """Debounced configuration reload to avoid multiple rapid reloads."""
        if self._reload_timer:
            self._reload_timer.cancel()
        
        self._reload_timer = threading.Timer(
            self.debounce_delay,
            lambda: asyncio.create_task(self.config_manager._reload_config(file_path))
        )
        self._reload_timer.start()


class ConfigManager:
    """
    Configuration manager with hot-reloading, validation, and environment override support.
    """
    
    def __init__(self):
        self._configs: Dict[str, BaseConfig] = {}
        self._config_paths: Dict[str, Path] = {}
        self._config_classes: Dict[str, Type[BaseConfig]] = {}
        self._reload_callbacks: Dict[str, list] = {}
        self._watched_files: Dict[str, str] = {}  # filename -> config_name
        self._observer: Optional[Observer] = None
        self._lock = asyncio.Lock()
        
        # Register default config classes
        self.register_config_class('api', APIConfig)
        self.register_config_class('coordinator', CoordinatorConfig)
        self.register_config_class('worker', WorkerConfig)
    
    def register_config_class(self, name: str, config_class: Type[T]):
        """Register a configuration class."""
        self._config_classes[name] = config_class
        self._reload_callbacks[name] = []
    
    def load_config(self, name: str, config_path: Optional[str] = None, 
                   config_class: Optional[Type[T]] = None) -> T:
        """
        Load configuration from file or environment variables.
        
        Args:
            name: Configuration name (e.g., 'api', 'coordinator', 'worker')
            config_path: Path to configuration file (optional)
            config_class: Configuration class (optional, uses registered class if not provided)
            
        Returns:
            Configuration instance
        """
        if config_class is None:
            config_class = self._config_classes.get(name)
            if config_class is None:
                raise ValueError(f"No configuration class registered for '{name}'")
        
        # Determine config file path
        if config_path is None:
            config_path = self._get_default_config_path(name)
        
        config_path = Path(config_path)
        
        # Load configuration
        if config_path.exists():
            logger.info(f"Loading {name} configuration from {config_path}")
            config = self._load_from_file(config_path, config_class)
        else:
            logger.info(f"Configuration file not found: {config_path}, using environment variables and defaults")
            config = config_class()
        
        # Store configuration
        self._configs[name] = config
        self._config_paths[name] = config_path
        self._watched_files[config_path.name] = name
        
        return config
    
    def get_config(self, name: str) -> Optional[BaseConfig]:
        """Get loaded configuration by name."""
        return self._configs.get(name)
    
    def update_config(self, name: str, updates: Dict[str, Any]) -> bool:
        """
        Update configuration with new values.
        
        Args:
            name: Configuration name
            updates: Dictionary of updates to apply
            
        Returns:
            True if update was successful, False otherwise
        """
        config = self._configs.get(name)
        if config is None:
            logger.error(f"Configuration '{name}' not loaded")
            return False
        
        try:
            # Create updated configuration
            config_dict = config.model_dump()
            self._deep_update(config_dict, updates)
            
            # Validate new configuration
            config_class = self._config_classes[name]
            new_config = config_class(**config_dict)
            
            # Update stored configuration
            self._configs[name] = new_config
            
            # Trigger callbacks
            asyncio.create_task(self._trigger_callbacks(name, new_config))
            
            logger.info(f"Configuration '{name}' updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update configuration '{name}': {e}")
            return False
    
    def add_reload_callback(self, name: str, callback: Callable[[BaseConfig], None]):
        """Add callback to be called when configuration is reloaded."""
        if name not in self._reload_callbacks:
            self._reload_callbacks[name] = []
        self._reload_callbacks[name].append(callback)
    
    def start_watching(self):
        """Start watching configuration files for changes."""
        if self._observer is not None:
            return
        
        self._observer = Observer()
        handler = ConfigFileHandler(self)
        
        # Watch config directories
        config_dirs = set()
        for config_path in self._config_paths.values():
            config_dirs.add(config_path.parent)
        
        for config_dir in config_dirs:
            if config_dir.exists():
                self._observer.schedule(handler, str(config_dir), recursive=False)
                logger.info(f"Watching configuration directory: {config_dir}")
        
        self._observer.start()
        logger.info("Configuration file watching started")
    
    def stop_watching(self):
        """Stop watching configuration files."""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Configuration file watching stopped")
    
    def validate_config(self, name: str) -> tuple[bool, Optional[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        config = self._configs.get(name)
        if config is None:
            return False, f"Configuration '{name}' not loaded"
        
        try:
            # Re-validate the configuration
            config_class = self._config_classes[name]
            config_class(**config.model_dump())
            return True, None
        except Exception as e:
            return False, str(e)
    
    def export_config(self, name: str, output_path: str, format: str = 'yaml') -> bool:
        """
        Export configuration to file.
        
        Args:
            name: Configuration name
            output_path: Output file path
            format: Output format ('yaml' or 'json')
            
        Returns:
            True if export was successful, False otherwise
        """
        config = self._configs.get(name)
        if config is None:
            logger.error(f"Configuration '{name}' not loaded")
            return False
        
        try:
            output_path = Path(output_path)
            config_dict = config.model_dump()
            
            if format.lower() == 'yaml':
                with open(output_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                import json
                with open(output_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration '{name}' exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export configuration '{name}': {e}")
            return False
    
    def _get_default_config_path(self, name: str) -> str:
        """Get default configuration file path for a service."""
        config_dir = Path("config")
        return str(config_dir / f"{name}.yaml")
    
    def _load_from_file(self, config_path: Path, config_class: Type[T]) -> T:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            if config_data is None:
                config_data = {}
            
            # Flatten nested configuration for environment variable override
            flat_config = {}
            self._flatten_dict(config_data, flat_config)
            
            # Override with environment variables
            original_env = {}
            for key, value in flat_config.items():
                env_key = key.upper()
                if env_key not in os.environ:
                    original_env[env_key] = os.environ.get(env_key)
                    os.environ[env_key] = str(value)
            
            try:
                config = config_class()
            finally:
                # Restore original environment
                for key, original_value in original_env.items():
                    if original_value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = original_value
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            raise
    
    def _flatten_dict(self, d: Dict[str, Any], result: Dict[str, Any], prefix: str = ""):
        """Flatten nested dictionary for environment variable mapping."""
        for key, value in d.items():
            env_key = f"{prefix}{key}".upper()
            if isinstance(value, dict):
                self._flatten_dict(value, result, f"{prefix}{key}_")
            else:
                result[env_key] = value
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """Deep update dictionary."""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    async def _reload_config(self, file_path: Path):
        """Reload configuration from file."""
        async with self._lock:
            config_name = self._watched_files.get(file_path.name)
            if config_name is None:
                return
            
            try:
                logger.info(f"Reloading configuration '{config_name}' from {file_path}")
                
                config_class = self._config_classes[config_name]
                new_config = self._load_from_file(file_path, config_class)
                
                # Update stored configuration
                old_config = self._configs.get(config_name)
                self._configs[config_name] = new_config
                
                # Trigger callbacks
                await self._trigger_callbacks(config_name, new_config)
                
                logger.info(f"Configuration '{config_name}' reloaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to reload configuration '{config_name}': {e}")
    
    async def _trigger_callbacks(self, name: str, config: BaseConfig):
        """Trigger reload callbacks for a configuration."""
        callbacks = self._reload_callbacks.get(name, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(config)
                else:
                    callback(config)
            except Exception as e:
                logger.error(f"Error in reload callback for '{name}': {e}")


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    return config_manager