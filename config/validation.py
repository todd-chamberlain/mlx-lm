"""
Configuration validation utilities and custom validators.
"""

import re
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_port(port: int) -> bool:
        """Validate port number."""
        return 1 <= port <= 65535
    
    @staticmethod
    def validate_host(host: str) -> bool:
        """Validate hostname or IP address."""
        if not host:
            return False
        
        # Check if it's a valid IP address
        try:
            socket.inet_aton(host)
            return True
        except socket.error:
            pass
        
        # Check if it's a valid hostname
        hostname_pattern = re.compile(
            r'^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$'
        )
        
        if '.' in host:
            # FQDN
            parts = host.split('.')
            return all(hostname_pattern.match(part) for part in parts)
        else:
            # Simple hostname
            return hostname_pattern.match(host) is not None
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False
    
    @staticmethod
    def validate_file_path(path: str, must_exist: bool = True) -> bool:
        """Validate file path."""
        try:
            file_path = Path(path)
            if must_exist:
                return file_path.exists() and file_path.is_file()
            else:
                # Check if parent directory exists
                return file_path.parent.exists()
        except Exception:
            return False
    
    @staticmethod
    def validate_directory_path(path: str, must_exist: bool = True) -> bool:
        """Validate directory path."""
        try:
            dir_path = Path(path)
            if must_exist:
                return dir_path.exists() and dir_path.is_dir()
            else:
                # Check if parent directory exists or path is absolute
                return dir_path.parent.exists() or dir_path.is_absolute()
        except Exception:
            return False
    
    @staticmethod
    def validate_log_level(level: str) -> bool:
        """Validate logging level."""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        return level.upper() in valid_levels
    
    @staticmethod
    def validate_memory_size(size_str: str) -> bool:
        """Validate memory size string (e.g., '1GB', '512MB')."""
        pattern = re.compile(r'^\d+(\.\d+)?\s*(B|KB|MB|GB|TB)$', re.IGNORECASE)
        return pattern.match(size_str.strip()) is not None
    
    @staticmethod
    def validate_duration(duration_str: str) -> bool:
        """Validate duration string (e.g., '30s', '5m', '1h')."""
        pattern = re.compile(r'^\d+(\.\d+)?\s*(s|m|h|d)$', re.IGNORECASE)
        return pattern.match(duration_str.strip()) is not None
    
    @staticmethod
    def validate_percentage(value: Union[int, float]) -> bool:
        """Validate percentage value (0-100)."""
        try:
            num_value = float(value)
            return 0 <= num_value <= 100
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_ratio(value: Union[int, float]) -> bool:
        """Validate ratio value (0.0-1.0)."""
        try:
            num_value = float(value)
            return 0.0 <= num_value <= 1.0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_positive_number(value: Union[int, float]) -> bool:
        """Validate positive number."""
        try:
            num_value = float(value)
            return num_value > 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_non_negative_number(value: Union[int, float]) -> bool:
        """Validate non-negative number."""
        try:
            num_value = float(value)
            return num_value >= 0
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def validate_api_key(api_key: str, min_length: int = 32) -> bool:
        """Validate API key format and length."""
        if not api_key or not isinstance(api_key, str):
            return False
        
        # Check minimum length
        if len(api_key) < min_length:
            return False
        
        # Check for valid characters (alphanumeric and some special chars)
        pattern = re.compile(r'^[A-Za-z0-9_\-\.]+$')
        return pattern.match(api_key) is not None
    
    @staticmethod
    def validate_worker_id(worker_id: str) -> bool:
        """Validate worker ID format."""
        if not worker_id or not isinstance(worker_id, str):
            return False
        
        # Worker ID should be alphanumeric with hyphens and underscores
        pattern = re.compile(r'^[A-Za-z0-9_\-]+$')
        return pattern.match(worker_id) is not None and len(worker_id) <= 64


class ConfigValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, field: str, value: Any, message: str):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(f"Validation error for field '{field}': {message}")


class ConfigValidationResult:
    """Configuration validation result."""
    
    def __init__(self):
        self.is_valid = True
        self.errors: List[ConfigValidationError] = []
        self.warnings: List[str] = []
    
    def add_error(self, field: str, value: Any, message: str):
        """Add validation error."""
        self.is_valid = False
        self.errors.append(ConfigValidationError(field, value, message))
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def get_error_summary(self) -> str:
        """Get summary of validation errors."""
        if not self.errors:
            return "No validation errors"
        
        lines = ["Configuration validation errors:"]
        for error in self.errors:
            lines.append(f"  - {error.field}: {error.message}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        return "\n".join(lines)


def validate_config_dict(config_dict: Dict[str, Any], 
                        validation_rules: Dict[str, Any]) -> ConfigValidationResult:
    """
    Validate configuration dictionary against validation rules.
    
    Args:
        config_dict: Configuration dictionary to validate
        validation_rules: Dictionary of validation rules
        
    Returns:
        ConfigValidationResult with validation results
    """
    result = ConfigValidationResult()
    
    def validate_nested(data: Dict[str, Any], rules: Dict[str, Any], path: str = ""):
        """Recursively validate nested configuration."""
        for key, rule in rules.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in data:
                if rule.get('required', False):
                    result.add_error(current_path, None, "Required field is missing")
                continue
            
            value = data[key]
            
            # Handle nested objects
            if isinstance(rule, dict) and 'type' not in rule:
                if isinstance(value, dict):
                    validate_nested(value, rule, current_path)
                else:
                    result.add_error(current_path, value, "Expected nested object")
                continue
            
            # Validate field type
            expected_type = rule.get('type')
            if expected_type and not isinstance(value, expected_type):
                result.add_error(
                    current_path, value, 
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )
                continue
            
            # Apply custom validators
            validators = rule.get('validators', [])
            for validator_func in validators:
                try:
                    if not validator_func(value):
                        validator_name = getattr(validator_func, '__name__', 'validator')
                        result.add_error(
                            current_path, value, 
                            f"Failed validation: {validator_name}"
                        )
                except Exception as e:
                    result.add_error(
                        current_path, value, 
                        f"Validator error: {str(e)}"
                    )
            
            # Check value constraints
            min_val = rule.get('min')
            max_val = rule.get('max')
            
            if min_val is not None and value < min_val:
                result.add_error(
                    current_path, value, 
                    f"Value must be at least {min_val}"
                )
            
            if max_val is not None and value > max_val:
                result.add_error(
                    current_path, value, 
                    f"Value must be at most {max_val}"
                )
            
            # Check string length constraints
            if isinstance(value, str):
                min_len = rule.get('min_length')
                max_len = rule.get('max_length')
                
                if min_len is not None and len(value) < min_len:
                    result.add_error(
                        current_path, value, 
                        f"String must be at least {min_len} characters"
                    )
                
                if max_len is not None and len(value) > max_len:
                    result.add_error(
                        current_path, value, 
                        f"String must be at most {max_len} characters"
                    )
            
            # Check allowed values
            allowed_values = rule.get('allowed')
            if allowed_values and value not in allowed_values:
                result.add_error(
                    current_path, value, 
                    f"Value must be one of: {allowed_values}"
                )
    
    validate_nested(config_dict, validation_rules)
    return result


# Common validation rules for different configuration types
COMMON_VALIDATION_RULES = {
    'server': {
        'host': {
            'type': str,
            'required': True,
            'validators': [ConfigValidator.validate_host]
        },
        'port': {
            'type': int,
            'required': True,
            'validators': [ConfigValidator.validate_port],
            'min': 1,
            'max': 65535
        }
    },
    'logging': {
        'level': {
            'type': str,
            'validators': [ConfigValidator.validate_log_level],
            'allowed': ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        },
        'file': {
            'type': str,
            'validators': [lambda x: ConfigValidator.validate_file_path(x, must_exist=False)]
        }
    },
    'monitoring': {
        'metrics_port': {
            'type': int,
            'validators': [ConfigValidator.validate_port],
            'min': 1,
            'max': 65535
        },
        'jaeger_endpoint': {
            'type': str,
            'validators': [ConfigValidator.validate_url]
        }
    }
}