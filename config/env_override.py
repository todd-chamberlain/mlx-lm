"""
Environment variable override utilities for configuration management.
"""

import os
import logging
from typing import Dict, Any, Optional, Type, TypeVar
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class EnvironmentOverride:
    """Utility class for handling environment variable overrides."""
    
    @staticmethod
    def apply_overrides(config_dict: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration dictionary.
        
        Args:
            config_dict: Configuration dictionary to override
            prefix: Environment variable prefix
            
        Returns:
            Updated configuration dictionary
        """
        result = config_dict.copy()
        
        # Flatten the config dict to match environment variable patterns
        flat_config = {}
        EnvironmentOverride._flatten_dict(result, flat_config, prefix)
        
        # Apply environment overrides
        for env_key, config_path in flat_config.items():
            env_value = os.getenv(env_key)
            if env_value is not None:
                EnvironmentOverride._set_nested_value(result, config_path, env_value)
                logger.debug(f"Applied environment override: {env_key}={env_value}")
        
        return result
    
    @staticmethod
    def _flatten_dict(d: Dict[str, Any], result: Dict[str, str], 
                     prefix: str = "", path: list = None):
        """
        Flatten nested dictionary to create environment variable mappings.
        
        Args:
            d: Dictionary to flatten
            result: Result dictionary mapping env vars to config paths
            prefix: Current prefix for environment variables
            path: Current path in the configuration
        """
        if path is None:
            path = []
        
        for key, value in d.items():
            current_path = path + [key]
            env_key = f"{prefix}{key}".upper()
            
            if isinstance(value, dict):
                EnvironmentOverride._flatten_dict(
                    value, result, f"{prefix}{key}_", current_path
                )
            else:
                result[env_key] = current_path
    
    @staticmethod
    def _set_nested_value(d: Dict[str, Any], path: list, value: str):
        """
        Set a nested value in a dictionary using a path.
        
        Args:
            d: Dictionary to update
            path: Path to the value (list of keys)
            value: Value to set (will be converted to appropriate type)
        """
        current = d
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert string value to appropriate type
        final_key = path[-1]
        current[final_key] = EnvironmentOverride._convert_value(value)
    
    @staticmethod
    def _convert_value(value: str) -> Any:
        """
        Convert string value to appropriate Python type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Handle list values (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    @staticmethod
    def get_env_var_name(config_path: str, prefix: str = "") -> str:
        """
        Generate environment variable name from configuration path.
        
        Args:
            config_path: Dot-separated configuration path (e.g., "server.port")
            prefix: Optional prefix for environment variable
            
        Returns:
            Environment variable name
        """
        env_name = config_path.replace('.', '_').upper()
        if prefix:
            env_name = f"{prefix.upper()}_{env_name}"
        return env_name
    
    @staticmethod
    def generate_env_template(config_dict: Dict[str, Any], 
                            prefix: str = "", 
                            output_file: Optional[str] = None) -> str:
        """
        Generate environment variable template from configuration.
        
        Args:
            config_dict: Configuration dictionary
            prefix: Environment variable prefix
            output_file: Optional output file path
            
        Returns:
            Environment variable template as string
        """
        lines = []
        lines.append(f"# Environment variables for configuration")
        lines.append(f"# Generated automatically - modify as needed")
        lines.append("")
        
        # Flatten config and generate template
        flat_config = {}
        EnvironmentOverride._flatten_dict(config_dict, flat_config, prefix)
        
        for env_key, config_path in sorted(flat_config.items()):
            # Get current value from config
            current_value = EnvironmentOverride._get_nested_value(config_dict, config_path)
            
            # Format the line
            comment = f"# Configuration path: {'.'.join(config_path)}"
            env_line = f"{env_key}={current_value}"
            
            lines.append(comment)
            lines.append(env_line)
            lines.append("")
        
        template = "\n".join(lines)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(template)
            logger.info(f"Environment template written to {output_file}")
        
        return template
    
    @staticmethod
    def _get_nested_value(d: Dict[str, Any], path: list) -> Any:
        """Get nested value from dictionary using path."""
        current = d
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current


def load_dotenv_file(env_file: str = ".env") -> Dict[str, str]:
    """
    Load environment variables from .env file.
    
    Args:
        env_file: Path to .env file
        
    Returns:
        Dictionary of environment variables
    """
    env_vars = {}
    env_path = Path(env_file)
    
    if not env_path.exists():
        logger.debug(f"Environment file not found: {env_file}")
        return env_vars
    
    try:
        with open(env_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse key=value pairs
                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    
                    env_vars[key] = value
                    
                    # Set in os.environ if not already set
                    if key not in os.environ:
                        os.environ[key] = value
                else:
                    logger.warning(f"Invalid line in {env_file}:{line_num}: {line}")
        
        logger.info(f"Loaded {len(env_vars)} environment variables from {env_file}")
        
    except Exception as e:
        logger.error(f"Failed to load environment file {env_file}: {e}")
    
    return env_vars


def create_env_example(config_dict: Dict[str, Any], 
                      output_file: str = ".env.example",
                      prefix: str = "") -> bool:
    """
    Create .env.example file from configuration.
    
    Args:
        config_dict: Configuration dictionary
        output_file: Output file path
        prefix: Environment variable prefix
        
    Returns:
        True if successful, False otherwise
    """
    try:
        template = EnvironmentOverride.generate_env_template(
            config_dict, prefix, output_file
        )
        return True
    except Exception as e:
        logger.error(f"Failed to create environment example file: {e}")
        return False