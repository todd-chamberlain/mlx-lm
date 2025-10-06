"""
Audit logging and request tracking utilities.
"""

import json
import logging
import time
from typing import Optional, Dict, Any
from fastapi import Request
from fastapi.security import HTTPAuthorizationCredentials

from .auth import get_client_id


# Setup audit logger
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# Create file handler for audit logs
audit_handler = logging.FileHandler("audit.log")
audit_handler.setLevel(logging.INFO)

# Create formatter
audit_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
audit_handler.setFormatter(audit_formatter)
audit_logger.addHandler(audit_handler)


class AuditLogger:
    """Handles audit logging for API requests."""
    
    @staticmethod
    def log_request(
        request: Request,
        credentials: Optional[HTTPAuthorizationCredentials] = None,
        request_data: Optional[Dict[str, Any]] = None,
        response_status: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
        duration_ms: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log API request for audit trail.
        
        Args:
            request: FastAPI request object
            credentials: Authentication credentials
            request_data: Request payload data
            response_status: HTTP response status code
            response_data: Response payload data (summary only)
            duration_ms: Request duration in milliseconds
            error: Error message if any
        """
        client_id = get_client_id(credentials)
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        audit_entry = {
            "timestamp": time.time(),
            "client_id": client_id,
            "client_ip": client_ip,
            "user_agent": user_agent,
            "method": request.method,
            "path": str(request.url.path),
            "query_params": dict(request.query_params),
            "authenticated": credentials is not None,
            "request_size": len(str(request_data)) if request_data else 0,
            "response_status": response_status,
            "duration_ms": duration_ms,
            "error": error
        }
        
        # Add request-specific data (sanitized)
        if request_data:
            sanitized_request = AuditLogger._sanitize_request_data(request_data)
            audit_entry["request_summary"] = sanitized_request
        
        # Add response summary
        if response_data:
            audit_entry["response_summary"] = AuditLogger._sanitize_response_data(response_data)
        
        audit_logger.info(json.dumps(audit_entry))
    
    @staticmethod
    def log_authentication_attempt(
        request: Request,
        success: bool,
        client_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log authentication attempts.
        
        Args:
            request: FastAPI request object
            success: Whether authentication was successful
            client_id: Client identifier if successful
            error: Error message if failed
        """
        client_ip = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        auth_entry = {
            "timestamp": time.time(),
            "event_type": "authentication",
            "client_ip": client_ip,
            "user_agent": user_agent,
            "success": success,
            "client_id": client_id,
            "error": error
        }
        
        if success:
            audit_logger.info(f"AUTH_SUCCESS: {json.dumps(auth_entry)}")
        else:
            audit_logger.warning(f"AUTH_FAILURE: {json.dumps(auth_entry)}")
    
    @staticmethod
    def log_rate_limit_exceeded(
        request: Request,
        client_id: str,
        limit: str
    ) -> None:
        """
        Log rate limit violations.
        
        Args:
            request: FastAPI request object
            client_id: Client identifier
            limit: Rate limit that was exceeded
        """
        client_ip = request.client.host if request.client else "unknown"
        
        rate_limit_entry = {
            "timestamp": time.time(),
            "event_type": "rate_limit_exceeded",
            "client_id": client_id,
            "client_ip": client_ip,
            "path": str(request.url.path),
            "limit": limit
        }
        
        audit_logger.warning(f"RATE_LIMIT: {json.dumps(rate_limit_entry)}")
    
    @staticmethod
    def _sanitize_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize request data for logging (remove sensitive information).
        
        Args:
            data: Request data dictionary
            
        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        
        for key, value in data.items():
            if key.lower() in ["password", "token", "key", "secret"]:
                sanitized[key] = "[REDACTED]"
            elif key == "messages" and isinstance(value, list):
                # Summarize messages without full content
                sanitized[key] = {
                    "count": len(value),
                    "roles": [msg.get("role", "unknown") for msg in value if isinstance(msg, dict)]
                }
            elif isinstance(value, str) and len(value) > 100:
                # Truncate long strings
                sanitized[key] = value[:100] + "..."
            else:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def _sanitize_response_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize response data for logging.
        
        Args:
            data: Response data dictionary
            
        Returns:
            Sanitized data dictionary
        """
        sanitized = {}
        
        # Only log summary information for responses
        if "choices" in data:
            choices = data["choices"]
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict) and "message" in first_choice:
                    message = first_choice["message"]
                    if isinstance(message, dict) and "content" in message:
                        content = message["content"]
                        sanitized["response_length"] = len(content) if isinstance(content, str) else 0
        
        if "usage" in data:
            sanitized["usage"] = data["usage"]
        
        if "model" in data:
            sanitized["model"] = data["model"]
        
        return sanitized


# Global audit logger instance
audit = AuditLogger()