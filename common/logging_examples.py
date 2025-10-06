"""
Examples of using the comprehensive logging system.

This module demonstrates how to integrate the logging system
into various services and components.
"""

from common.logging_system import (
    setup_logging,
    get_logger,
    request_context,
    log_error,
    log_performance,
    log_operation,
    get_error_tracker,
)


# Example 1: Basic Service Setup
def setup_api_service():
    """Example: Setting up logging for API service."""
    # Set up structured JSON logging
    logger = setup_logging(
        service_name="api_server",
        log_level="INFO",
        json_format=True,
        log_file="/var/log/api_server.log"
    )
    
    logger.info("API server starting up")
    return logger


# Example 2: Request Handling with Context
def handle_api_request(request_data):
    """Example: Handling an API request with full context."""
    logger = get_logger("api_server")
    
    # Create request context for correlation
    with request_context(
        request_id=request_data.get("id"),
        user_id=request_data.get("user_id"),
        endpoint=request_data.get("endpoint"),
        method=request_data.get("method")
    ) as ctx:
        logger.info(f"Received request: {ctx['request_id']}")
        
        try:
            # Log the operation with automatic performance tracking
            with log_operation(logger, "process_request"):
                # Process request
                result = process_request(request_data)
                logger.info("Request processed successfully")
                return result
        
        except Exception as e:
            # Log error with full context
            log_error(
                logger,
                "request_processing_error",
                f"Failed to process request: {str(e)}",
                exception=e,
                request_data=request_data
            )
            raise


# Example 3: Worker Service with Performance Tracking
def worker_inference_task(model_name, prompt):
    """Example: Worker performing inference with logging."""
    logger = get_logger("mlx_worker", worker_id="worker-1", model=model_name)
    
    with request_context(trace_id="trace-123"):
        logger.info(f"Starting inference for model: {model_name}")
        
        try:
            # Track model loading
            with log_operation(logger, "load_model", model=model_name):
                model = load_model(model_name)
            
            # Track inference
            with log_operation(logger, "run_inference", prompt_length=len(prompt)):
                result = model.generate(prompt)
            
            # Log performance metrics
            log_performance(
                logger,
                "inference_complete",
                duration_ms=150.5,
                tokens_generated=50,
                model=model_name
            )
            
            return result
        
        except Exception as e:
            log_error(
                logger,
                "inference_error",
                f"Inference failed for model {model_name}",
                exception=e,
                model=model_name,
                prompt_length=len(prompt)
            )
            raise


# Example 4: Coordinator Service with Error Tracking
def coordinator_route_request(request):
    """Example: Coordinator routing with error tracking."""
    logger = get_logger("coordinator", service="coordinator")
    
    with request_context(request_id=request.id):
        logger.info("Routing inference request")
        
        try:
            # Select worker
            with log_operation(logger, "select_worker"):
                worker = select_best_worker()
            
            # Forward request
            with log_operation(logger, "forward_request", worker_id=worker.id):
                response = forward_to_worker(worker, request)
            
            logger.info(f"Request routed successfully to {worker.id}")
            return response
        
        except WorkerUnavailableError as e:
            log_error(
                logger,
                "worker_unavailable",
                "No workers available for request",
                exception=e,
                request_id=request.id
            )
            # Try fallback
            return handle_fallback(request)
        
        except Exception as e:
            log_error(
                logger,
                "routing_error",
                "Failed to route request",
                exception=e,
                request_id=request.id
            )
            raise


# Example 5: Monitoring Error Rates
def monitor_system_health():
    """Example: Monitoring system health and error rates."""
    logger = get_logger("monitoring")
    tracker = get_error_tracker()
    
    # Get error summary
    summary = tracker.get_error_summary()
    
    logger.info(
        "System health check",
        extra={
            "total_errors": summary["total_errors"],
            "error_rate": summary["total_error_rate"],
            "error_counts": summary["error_counts"]
        }
    )
    
    # Check for high error rates
    if summary["total_error_rate"] > 10:  # More than 10 errors per minute
        logger.warning(
            "High error rate detected",
            extra={"error_rate": summary["total_error_rate"]}
        )
        
        # Get recent errors for analysis
        recent_errors = tracker.get_recent_errors(limit=5)
        for error in recent_errors:
            logger.warning(
                f"Recent error: {error['error_type']} - {error['message']}",
                extra={"error_data": error}
            )
    
    return summary


# Example 6: Distributed Tracing Across Services
def distributed_request_flow():
    """Example: Request flow across multiple services with tracing."""
    # API Server receives request
    api_logger = get_logger("api_server")
    
    with request_context(trace_id="trace-abc-123") as ctx:
        api_logger.info("API: Received chat completion request")
        
        # Forward to coordinator
        coordinator_logger = get_logger("coordinator")
        with request_context(
            trace_id=ctx["trace_id"],  # Propagate trace ID
            request_id="req-456"
        ):
            coordinator_logger.info("Coordinator: Routing request")
            
            # Forward to worker
            worker_logger = get_logger("worker")
            with request_context(
                trace_id=ctx["trace_id"],  # Same trace ID
                request_id="req-456",
                worker_id="worker-1"
            ):
                worker_logger.info("Worker: Processing inference")
                
                # All logs will have the same trace_id for correlation
                with log_operation(worker_logger, "inference"):
                    result = perform_inference()
                
                worker_logger.info("Worker: Inference complete")
            
            coordinator_logger.info("Coordinator: Response received")
        
        api_logger.info("API: Returning response to client")


# Example 7: Error Aggregation and Alerting
def check_error_thresholds():
    """Example: Check error thresholds and trigger alerts."""
    logger = get_logger("alerting")
    tracker = get_error_tracker()
    
    # Define thresholds
    thresholds = {
        "worker_unavailable": 5,  # Alert if > 5 per minute
        "inference_error": 3,
        "routing_error": 2,
    }
    
    for error_type, threshold in thresholds.items():
        rate = tracker.get_error_rate(error_type)
        
        if rate > threshold:
            logger.error(
                f"Alert: {error_type} rate exceeded threshold",
                extra={
                    "error_type": error_type,
                    "current_rate": rate,
                    "threshold": threshold,
                    "severity": "high"
                }
            )
            
            # Trigger alert (e.g., send to monitoring system)
            trigger_alert(error_type, rate, threshold)


# Dummy functions for examples
def load_model(name):
    """Dummy model loading."""
    return type('Model', (), {'generate': lambda self, p: f"Generated from {p}"})()

def process_request(data):
    """Dummy request processing."""
    return {"status": "success"}

def select_best_worker():
    """Dummy worker selection."""
    return type('Worker', (), {'id': 'worker-1'})()

def forward_to_worker(worker, request):
    """Dummy request forwarding."""
    return {"result": "success"}

def handle_fallback(request):
    """Dummy fallback handler."""
    return {"result": "fallback"}

def perform_inference():
    """Dummy inference."""
    return "inference result"

def trigger_alert(error_type, rate, threshold):
    """Dummy alert triggering."""
    pass

class WorkerUnavailableError(Exception):
    """Dummy exception."""
    pass


if __name__ == "__main__":
    # Run examples
    print("Example 1: Service Setup")
    setup_api_service()
    
    print("\nExample 5: Monitor System Health")
    monitor_system_health()
    
    print("\nLogging system examples completed!")
