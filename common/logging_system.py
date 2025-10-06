"""
Comprehensive structured logging and error tracking system.

Provides:
- Structured JSON logging with context
- Request correlation and tracing
- Error aggregation and tracking
- Performance metrics integration
- Log level management
"""

import logging
import json
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, List
from contextlib import contextmanager
from threading import local
import uuid


# Thread-local storage for request context
_request_context = local()


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add request context if available
        if hasattr(_request_context, 'trace_id'):
            log_data["trace_id"] = _request_context.trace_id
        if hasattr(_request_context, 'request_id'):
            log_data["request_id"] = _request_context.request_id
        if hasattr(_request_context, 'user_id'):
            log_data["user_id"] = _request_context.user_id
        
        # Add extra fields from record
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        return json.dumps(log_data)


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context to all log messages."""
    
    def __init__(self, logger: logging.Logger, extra: Optional[Dict[str, Any]] = None):
        super().__init__(logger, extra or {})
    
    def process(self, msg: str, kwargs: Dict[str, Any]) -> tuple:
        """Add extra context to log messages."""
        # Merge extra fields
        extra_fields = dict(self.extra)
        if 'extra' in kwargs:
            extra_fields.update(kwargs.pop('extra'))
        
        # Create a custom LogRecord attribute for extra fields
        if extra_fields:
            if 'extra' not in kwargs:
                kwargs['extra'] = {}
            kwargs['extra']['extra_fields'] = extra_fields
        
        return msg, kwargs


class ErrorTracker:
    """Track and aggregate errors for monitoring."""
    
    def __init__(self, max_errors: int = 1000):
        self.max_errors = max_errors
        self.errors: List[Dict[str, Any]] = []
        self.error_counts: Dict[str, int] = {}
        self.error_rates: Dict[str, List[float]] = {}
    
    def track_error(
        self,
        error_type: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ):
        """Track an error occurrence."""
        error_data = {
            "timestamp": time.time(),
            "error_type": error_type,
            "message": message,
            "context": context or {},
        }
        
        if exception:
            error_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "traceback": traceback.format_exc()
            }
        
        # Add to error list
        self.errors.append(error_data)
        if len(self.errors) > self.max_errors:
            self.errors.pop(0)
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Track error rate (errors per minute)
        if error_type not in self.error_rates:
            self.error_rates[error_type] = []
        self.error_rates[error_type].append(time.time())
        
        # Clean old timestamps (keep last 60 seconds)
        cutoff = time.time() - 60
        self.error_rates[error_type] = [
            t for t in self.error_rates[error_type] if t > cutoff
        ]
    
    def get_error_rate(self, error_type: str) -> float:
        """Get error rate per minute for a specific error type."""
        if error_type not in self.error_rates:
            return 0.0
        return len(self.error_rates[error_type])
    
    def get_total_error_rate(self) -> float:
        """Get total error rate per minute across all types."""
        return sum(len(rates) for rates in self.error_rates.values())
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked errors."""
        return {
            "total_errors": len(self.errors),
            "error_counts": dict(self.error_counts),
            "error_rates": {
                error_type: self.get_error_rate(error_type)
                for error_type in self.error_rates
            },
            "total_error_rate": self.get_total_error_rate(),
        }
    
    def get_recent_errors(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent errors."""
        return self.errors[-limit:]


# Global error tracker instance
_error_tracker = ErrorTracker()


def setup_logging(
    service_name: str,
    log_level: str = "INFO",
    json_format: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logging for a service.
    
    Args:
        service_name: Name of the service
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON formatting for logs
        log_file: Optional file path for log output
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(service_name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if json_format:
        console_handler.setFormatter(StructuredFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        if json_format:
            file_handler.setFormatter(StructuredFormatter())
        else:
            file_handler.setFormatter(
                logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
            )
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str, **context) -> ContextLogger:
    """
    Get a logger with optional context.
    
    Args:
        name: Logger name
        **context: Additional context to include in all log messages
    
    Returns:
        ContextLogger instance
    """
    logger = logging.getLogger(name)
    return ContextLogger(logger, context)


@contextmanager
def request_context(
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    user_id: Optional[str] = None,
    **extra
):
    """
    Context manager for request-scoped logging context.
    
    Args:
        request_id: Unique request identifier
        trace_id: Distributed trace identifier
        user_id: User identifier
        **extra: Additional context fields
    """
    # Generate IDs if not provided
    if request_id is None:
        request_id = str(uuid.uuid4())
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    
    # Store in thread-local storage
    _request_context.request_id = request_id
    _request_context.trace_id = trace_id
    if user_id:
        _request_context.user_id = user_id
    
    # Store extra fields
    for key, value in extra.items():
        setattr(_request_context, key, value)
    
    try:
        yield {
            "request_id": request_id,
            "trace_id": trace_id,
            "user_id": user_id,
            **extra
        }
    finally:
        # Clean up context
        if hasattr(_request_context, 'request_id'):
            delattr(_request_context, 'request_id')
        if hasattr(_request_context, 'trace_id'):
            delattr(_request_context, 'trace_id')
        if hasattr(_request_context, 'user_id'):
            delattr(_request_context, 'user_id')
        for key in extra:
            if hasattr(_request_context, key):
                delattr(_request_context, key)


def log_error(
    logger: logging.Logger,
    error_type: str,
    message: str,
    exception: Optional[Exception] = None,
    **context
):
    """
    Log an error and track it for monitoring.
    
    Args:
        logger: Logger instance
        error_type: Type/category of error
        message: Error message
        exception: Optional exception object
        **context: Additional context
    """
    # Track error
    _error_tracker.track_error(error_type, message, context, exception)
    
    # Log error
    if exception:
        logger.error(
            f"{error_type}: {message}",
            exc_info=exception,
            extra=context
        )
    else:
        logger.error(
            f"{error_type}: {message}",
            extra=context
        )


def get_error_tracker() -> ErrorTracker:
    """Get the global error tracker instance."""
    return _error_tracker


def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    **context
):
    """
    Log performance metrics.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Duration in milliseconds
        **context: Additional context
    """
    logger.info(
        f"Performance: {operation} completed in {duration_ms:.2f}ms",
        extra={
            "operation": operation,
            "duration_ms": duration_ms,
            **context
        }
    )


@contextmanager
def log_operation(
    logger: logging.Logger,
    operation: str,
    log_performance_metrics: bool = True,
    **context
):
    """
    Context manager for logging operation start/end and performance.
    
    Args:
        logger: Logger instance
        operation: Operation name
        log_performance_metrics: Whether to log performance metrics
        **context: Additional context
    """
    start_time = time.time()
    logger.info(f"Starting: {operation}", extra=context)
    
    try:
        yield
        duration_ms = (time.time() - start_time) * 1000
        
        if log_performance_metrics:
            log_performance(logger, operation, duration_ms, **context)
        else:
            logger.info(f"Completed: {operation}", extra=context)
    
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        log_error(
            logger,
            "operation_failed",
            f"Operation {operation} failed after {duration_ms:.2f}ms",
            exception=e,
            operation=operation,
            duration_ms=duration_ms,
            **context
        )
        raise
