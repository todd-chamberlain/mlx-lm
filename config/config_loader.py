"""
Configuration Loader for Distributed Inference

Loads configuration from YAML files and environment variables.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, List


class Config:
    """Configuration manager"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._find_config()
        self.config = self._load_config()
    
    def _find_config(self) -> str:
        """Find configuration file"""
        # Check environment variable
        if 'MLX_CONFIG' in os.environ:
            return os.environ['MLX_CONFIG']
        
        # Check common locations
        locations = [
            'config/distributed_config.yaml',
            'distributed_config.yaml',
            '~/.mlx/config.yaml',
            '/etc/mlx/config.yaml'
        ]
        
        for loc in locations:
            path = Path(loc).expanduser()
            if path.exists():
                return str(path)
        
        # Return default
        return 'config/distributed_config.yaml'
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        path = Path(self.config_path).expanduser()
        
        if not path.exists():
            # Return defaults if no config file
            return self._get_defaults()
        
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override with environment variables
        config = self._apply_env_overrides(config)
        
        return config
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'server': {
                'host': '0.0.0.0',
                'port': 8080
            },
            'nodes': [
                {'id': 'node-1', 'address': '10.201.20.10:50052'},
                {'id': 'node-2', 'address': '10.201.20.12:50052'}
            ],
            'model': {
                'default': 'mlx-community/Llama-3.2-3B-Instruct-4bit',
                'cache_dir': '~/.cache/huggingface'
            },
            'pipeline': {
                'strategy': 'balanced'
            },
            'performance': {
                'max_concurrent_requests': 10,
                'drain_timeout': 30
            },
            'logging': {
                'level': 'INFO',
                'file': '/tmp/distributed_server.log'
            }
        }
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Server
        if 'MLX_HOST' in os.environ:
            config['server']['host'] = os.environ['MLX_HOST']
        if 'MLX_PORT' in os.environ:
            config['server']['port'] = int(os.environ['MLX_PORT'])
        
        # Model
        if 'MLX_MODEL' in os.environ:
            config['model']['default'] = os.environ['MLX_MODEL']
        if 'MLX_CACHE_DIR' in os.environ:
            config['model']['cache_dir'] = os.environ['MLX_CACHE_DIR']
        
        # Nodes
        if 'MLX_NODE1_ADDRESS' in os.environ:
            config['nodes'][0]['address'] = os.environ['MLX_NODE1_ADDRESS']
        if 'MLX_NODE2_ADDRESS' in os.environ:
            if len(config['nodes']) > 1:
                config['nodes'][1]['address'] = os.environ['MLX_NODE2_ADDRESS']
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.config.get('server', {})
    
    def get_nodes(self) -> List[Dict[str, str]]:
        """Get node configuration"""
        return self.config.get('nodes', [])
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self.config.get('model', {})
    
    def get_pipeline_config(self) -> Dict[str, Any]:
        """Get pipeline configuration"""
        return self.config.get('pipeline', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self.config.get('performance', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        return self.config.get('logging', {})


# Global config instance
_config = None


def get_config(config_path: str = None) -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None or config_path:
        _config = Config(config_path)
    return _config
