# Comprehensive Logging and Error Tracking System

Complete structured logging system with request correlation, error tracking, and performance monitoring for the MLX distributed inference system.

## Features

- **Structured JSON Logging**: Machine-readable logs with consistent format
- **Request Correlation**: Track requests across distributed services with trace IDs
- **Error Aggregation**: Automatic error tracking and rate monitoring
- **Performance Metrics**: Built-in performance logging and timing
- **Context Management**: Thread-safe request context for correlation
- **Error Rate Monitoring**: Real-time error rate calculation and alerting

## Quick Start

### 1. Setup Logging for a Service

```python
from common.logging_system import setup_logging

# Initialize logging for your service
logger = setup_logging(
    service_name="api_server",
    log_level="INFO",
    json_format=True,
    log_file="/var/log/api_server.log"
)

logger.info("Service started successfully")
```

### 2. Use Request Context for Correlation

```python
from common.logging_system import request_context, get_logger

logger = get_logger("api_server")

# Create request context with correlation IDs
with request_context(
    request_id="req-123",
    trace_id="trace-456",
    user_id="user-789"
) as ctx:
    logger.info("Processing request")
    # All logs within this context will include the IDs
```

### 3. Log Operations with Performance Tracking

```python
from common.logging_system import log_operation

with log_operation(logger, "database_query", table="users"):
    # Your operation here
    result = db.query("SELECT * FROM users")
    # Automatically logs start, end, and duration
```

### 4. Track and Log Errors

```python
from common.logging_system import log_error

try:
    risky_operation()
except Exception as e:
    log_error(
        logger,
        "operation_failed",
        "Failed to complete operation",
        exception=e,
        user_id="user-123",
        operation="risky_operation"
    )
```

### 5. Monitor Error Rates

```python
from common.logging_system import get_error_tracker

tracker = get_error_tracker()

# Get error summary
summary = tracker.get_error_summary()
print(f"Total errors: {summary['total_errors']}")
print(f"Error rate: {summary['total_error_rate']} per minute")

# Get recent errors
recent = tracker.get_recent_errors(limit=10)
for error in recent:
    print(f"{error['error_type']}: {error['message']}")
```

## Log Format

### JSON Format (Structured)

```json
{
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "level": "INFO",
  "logger": "api_server",
  "message": "Request processed successfully",
  "module": "api",
  "function": "handle_request",
  "line": 42,
  "trace_id": "trace-abc-123",
  "request_id": "req-456",
  "user_id": "user-789",
  "duration_ms": 125.5,
  "status": 200
}
```

### Plain Text Format

```
2024-01-15 10:30:45,123 - api_server - INFO - Request processed successfully
```

## API Reference

### setup_logging()

Initialize logging for a service.

```python
logger = setup_logging(
    service_name: str,          # Name of the service
    log_level: str = "INFO",    # DEBUG, INFO, WARNING, ERROR, CRITICAL
    json_format: bool = True,   # Use JSON formatting
    log_file: Optional[str] = None  # Optional log file path
) -> logging.Logger
```

### get_logger()

Get a logger with optional context.

```python
logger = get_logger(
    name: str,      # Logger name
    **context       # Additional context fields
) -> ContextLogger
```

### request_context()

Context manager for request-scoped logging.

```python
with request_context(
    request_id: Optional[str] = None,  # Auto-generated if not provided
    trace_id: Optional[str] = None,    # Auto-generated if not provided
    user_id: Optional[str] = None,
    **extra                             # Additional context fields
) as ctx:
    # All logs here will include the context
    pass
```

### log_error()

Log an error and track it for monitoring.

```python
log_error(
    logger: logging.Logger,
    error_type: str,                    # Error category/type
    message: str,                       # Error message
    exception: Optional[Exception] = None,  # Exception object
    **context                           # Additional context
)
```

### log_performance()

Log performance metrics.

```python
log_performance(
    logger: logging.Logger,
    operation: str,         # Operation name
    duration_ms: float,     # Duration in milliseconds
    **context               # Additional context
)
```

### log_operation()

Context manager for operation logging with automatic timing.

```python
with log_operation(
    logger: logging.Logger,
    operation: str,                     # Operation name
    log_performance_metrics: bool = True,  # Log performance
    **context                           # Additional context
):
    # Your operation here
    pass
```

### get_error_tracker()

Get the global error tracker instance.

```python
tracker = get_error_tracker()

# Get error summary
summary = tracker.get_error_summary()

# Get error rate for specific type
rate = tracker.get_error_rate("validation_error")

# Get recent errors
recent = tracker.get_recent_errors(limit=10)
```

## Integration Examples

### API Server

```python
from common.logging_system import setup_logging, request_context, log_operation

logger = setup_logging("api_server", log_level="INFO", json_format=True)

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    with request_context(
        request_id=request.id,
        user_id=request.user_id,
        endpoint="/v1/chat/completions"
    ):
        with log_operation(logger, "process_chat_request"):
            result = await process_request(request)
            return result
```

### Worker Service

```python
from common.logging_system import get_logger, log_operation, log_error

logger = get_logger("mlx_worker", worker_id="worker-1")

def process_inference(request):
    try:
        with log_operation(logger, "inference", model=request.model):
            result = model.generate(request.prompt)
            return result
    except Exception as e:
        log_error(
            logger,
            "inference_error",
            f"Inference failed: {str(e)}",
            exception=e,
            model=request.model
        )
        raise
```

### Coordinator Service

```python
from common.logging_system import get_logger, request_context, log_error

logger = get_logger("coordinator")

def route_request(request):
    with request_context(trace_id=request.trace_id, request_id=request.id):
        try:
            worker = select_worker()
            logger.info(f"Routing to worker: {worker.id}")
            return forward_to_worker(worker, request)
        except WorkerUnavailableError as e:
            log_error(
                logger,
                "worker_unavailable",
                "No workers available",
                exception=e
            )
            raise
```

## Distributed Tracing

Track requests across multiple services using trace IDs:

```python
# API Server
with request_context(trace_id="trace-123") as ctx:
    api_logger.info("Received request")
    
    # Pass trace_id to coordinator
    coordinator_request = {
        "trace_id": ctx["trace_id"],
        "data": request_data
    }
    
    # Coordinator
    with request_context(trace_id=coordinator_request["trace_id"]):
        coordinator_logger.info("Routing request")
        
        # Pass to worker
        worker_request = {
            "trace_id": coordinator_request["trace_id"],
            "data": coordinator_request["data"]
        }
        
        # Worker
        with request_context(trace_id=worker_request["trace_id"]):
            worker_logger.info("Processing inference")
            # All logs share the same trace_id
```

## Error Monitoring and Alerting

```python
from common.logging_system import get_error_tracker

tracker = get_error_tracker()

# Check error rates
def monitor_errors():
    summary = tracker.get_error_summary()
    
    # Alert on high error rate
    if summary["total_error_rate"] > 10:
        alert("High error rate detected", summary)
    
    # Alert on specific error types
    for error_type, rate in summary["error_rates"].items():
        if rate > 5:
            alert(f"High rate for {error_type}", rate)
```

## Best Practices

1. **Always use request context** for API requests to enable correlation
2. **Use log_operation** for timing critical operations
3. **Use log_error** instead of logger.error for trackable errors
4. **Include relevant context** in all log messages
5. **Use structured logging (JSON)** in production
6. **Monitor error rates** regularly for system health
7. **Propagate trace IDs** across service boundaries
8. **Set appropriate log levels** (DEBUG for development, INFO for production)

## Configuration

### Environment Variables

```bash
# Log level
export LOG_LEVEL=INFO

# Log format
export LOG_FORMAT=json  # or 'plain'

# Log file path
export LOG_FILE=/var/log/service.log

# Service name
export SERVICE_NAME=api_server
```

### Configuration File

```yaml
logging:
  level: INFO
  format: json
  file: /var/log/service.log
  service_name: api_server
```

## Performance Considerations

- JSON formatting adds ~10-20% overhead vs plain text
- Error tracking keeps last 1000 errors in memory by default
- Request context uses thread-local storage (thread-safe)
- Error rate calculation uses sliding 60-second window

## Troubleshooting

### Logs not appearing

Check log level configuration and ensure handlers are properly set up.

### Missing trace IDs

Ensure request_context is used and trace IDs are propagated across services.

### High memory usage

Reduce ErrorTracker max_errors limit or implement log rotation.

### Performance impact

Use plain text format in development, JSON in production. Disable performance logging for high-frequency operations.

## See Also

- `logging_examples.py` - Complete usage examples
- `test_logging_system.py` - Comprehensive test suite
- Prometheus metrics integration for error rate monitoring
- Grafana dashboards for log visualization
