"""
Authentication and authorization utilities.
"""

import os
from typing import Optional
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPAuthorizationCredentials


def verify_api_key(
    credentials: Optional[HTTPAuthorizationCredentials],
    request: Optional[Request] = None
) -> None:
    """
    Verify API key from Bearer token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if authentication is enabled
    if os.getenv("ENABLE_AUTH", "false").lower() != "true":
        return
    
    # Check if credentials are provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Verify Bearer scheme
    if credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme. Expected Bearer token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Get valid API keys from environment
    api_keys_str = os.getenv("API_KEYS", "")
    if not api_keys_str:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API keys not configured"
        )
    
    valid_keys = [key.strip() for key in api_keys_str.split(",") if key.strip()]
    if not valid_keys:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No valid API keys configured"
        )
    
    # Verify the provided token
    token = credentials.credentials
    if not token or len(token) < 32:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key format",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if token not in valid_keys:
        # Log failed authentication attempt
        if request:
            from .audit import audit
            audit.log_authentication_attempt(
                request, 
                success=False, 
                error="Invalid API key"
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    # Log successful authentication
    if request:
        from .audit import audit
        audit.log_authentication_attempt(
            request,
            success=True,
            client_id=get_client_id(credentials)
        )


def get_client_id(credentials: Optional[HTTPAuthorizationCredentials]) -> str:
    """
    Get client ID for rate limiting and logging.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Client identifier string
    """
    if credentials and credentials.credentials:
        # Use first 8 characters of API key as client ID
        return credentials.credentials[:8]
    return "anonymous"