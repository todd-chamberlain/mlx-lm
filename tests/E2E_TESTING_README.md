# End-to-End Integration Testing Guide

## Overview

This document describes the comprehensive end-to-end integration test suite for the MLX distributed inference system. These tests validate the complete request flow from API to worker, multi-worker load distribution, failure scenarios, and streaming response handling.

## Test Structure

### 1. Complete Request Flow Tests (`TestCompleteRequestFlow`)

Tests the full request lifecycle from API server through coordinator to worker and back.

**Key Tests:**
- `test_api_to_worker_request_flow` - Validates complete request flow with OpenAI-compatible API
- `test_request_routing_through_coordinator` - Tests coordinator routing logic
- `test_worker_selection_and_load_balancing` - Verifies worker selection algorithm
- `test_response_aggregation_and_formatting` - Tests OpenAI format conversion

**What's Tested:**
- API endpoint receives and validates requests
- Coordinator routes requests to appropriate workers
- Worker selection based on load, latency, and health
- Response aggregation and OpenAI-compatible formatting
- Request ID tracking and correlation

### 2. Multi-Worker Load Distribution (`TestMultiWorkerLoadDistribution`)

Tests load distribution across multiple workers with various balancing strategies.

**Key Tests:**
- `test_requests_distributed_across_workers` - Validates even distribution
- `test_load_balancing_algorithms` - Tests different balancing strategies
- `test_worker_health_based_routing` - Verifies health-aware routing
- `test_concurrent_request_handling` - Tests concurrent request processing

**What's Tested:**
- Round-robin load balancing
- Weighted load balancing based on worker metrics
- Least-latency worker selection
- Health-based worker filtering
- Concurrent request handling across worker pool

### 3. Failure Scenario Testing (`TestFailureScenarios`)

Tests system resilience under various failure conditions.

**Key Tests:**
- `test_worker_failure_detection` - Tests heartbeat-based failure detection
- `test_worker_failover` - Validates automatic failover
- `test_coordinator_resilience_to_worker_failures` - Tests coordinator resilience
- `test_api_server_handling_coordinator_failures` - Tests API error handling
- `test_graceful_degradation_under_partial_failures` - Tests partial failure handling

**What's Tested:**
- Worker failure detection via missed heartbeats
- Automatic failover to healthy workers
- Coordinator continues operating with reduced capacity
- API returns appropriate errors when coordinator unavailable
- System maintains functionality with partial worker availability

### 4. Streaming Response Validation (`TestStreamingResponseValidation`)

Tests streaming inference capabilities and error handling.

**Key Tests:**
- `test_streaming_chat_completions` - Tests SSE streaming format
- `test_streaming_token_by_token_delivery` - Validates token streaming
- `test_streaming_error_handling` - Tests error handling during streaming
- `test_client_disconnection_handling` - Tests disconnection handling

**What's Tested:**
- Server-Sent Events (SSE) format compliance
- Token-by-token streaming delivery
- Error handling mid-stream
- Client disconnection detection and cleanup
- Streaming timeout handling

### 5. System Integration Tests (`TestSystemIntegration`)

Tests complete system integration and performance characteristics.

**Key Tests:**
- `test_full_system_startup_and_communication` - Tests system initialization
- `test_end_to_end_latency_measurement` - Measures request latency
- `test_request_tracing_across_services` - Tests distributed tracing
- `test_system_under_load` - Tests system under concurrent load

**What's Tested:**
- All services can be initialized and communicate
- End-to-end latency is within acceptable bounds
- Request tracing works across service boundaries
- System handles concurrent requests correctly

## Running the Tests

### Run All End-to-End Tests

```bash
pytest tests/test_end_to_end_integration.py -v
```

### Run Specific Test Class

```bash
# Test complete request flow
pytest tests/test_end_to_end_integration.py::TestCompleteRequestFlow -v

# Test load distribution
pytest tests/test_end_to_end_integration.py::TestMultiWorkerLoadDistribution -v

# Test failure scenarios
pytest tests/test_end_to_end_integration.py::TestFailureScenarios -v

# Test streaming
pytest tests/test_end_to_end_integration.py::TestStreamingResponseValidation -v

# Test system integration
pytest tests/test_end_to_end_integration.py::TestSystemIntegration -v
```

### Run Specific Test

```bash
pytest tests/test_end_to_end_integration.py::TestCompleteRequestFlow::test_api_to_worker_request_flow -v
```

### Run with Coverage

```bash
pytest tests/test_end_to_end_integration.py --cov=. --cov-report=html
```

## Test Fixtures and Mocking

### Mock Workers

Tests create mock gRPC worker servers that simulate real worker behavior:

```python
def create_mock_worker(self, worker_id: str, port: int):
    """Create a mock worker gRPC server."""
    # Creates gRPC server with InferenceService implementation
    # Tracks request counts and simulates responses
```

### Mock Coordinator

Tests use mock coordinator services or patch coordinator clients:

```python
with patch('api.coordinator_client.CoordinatorClient') as mock_coord:
    mock_coord.return_value.route_inference = AsyncMock(return_value=mock_response)
```

### Request Tracking

Tests track requests across workers to verify distribution:

```python
self.request_counts = {}  # Track requests per worker
# Verify even distribution across workers
```

## Test Scenarios

### Normal Operation

1. Client sends request to API
2. API validates and forwards to coordinator
3. Coordinator selects optimal worker
4. Worker processes inference
5. Response flows back through coordinator to API
6. Client receives OpenAI-compatible response

### Worker Failure

1. Worker stops sending heartbeats
2. Health monitor detects failure
3. Worker marked as unhealthy
4. New requests routed to healthy workers
5. System continues operating with reduced capacity

### Streaming Operation

1. Client requests streaming response
2. API establishes SSE connection
3. Coordinator routes to worker
4. Worker streams tokens back
5. API forwards tokens as SSE events
6. Client receives token-by-token updates

### Load Distribution

1. Multiple concurrent requests arrive
2. Coordinator distributes across workers
3. Load balancing algorithm selects workers
4. Requests processed in parallel
5. Responses aggregated and returned

## Performance Expectations

### Latency Targets

- **End-to-end latency**: < 1000ms for typical requests
- **Processing latency**: < 500ms for worker processing
- **Streaming first token**: < 200ms
- **Coordinator routing**: < 50ms

### Throughput Targets

- **Concurrent requests**: 50+ simultaneous requests
- **Worker utilization**: 80%+ under load
- **Request distribution**: Even across healthy workers

## Debugging Failed Tests

### Check Test Logs

```bash
pytest tests/test_end_to_end_integration.py -v -s
```

### Enable Debug Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Inspect Mock Calls

```python
# Check if mock was called
assert mock_coord.route_inference.called

# Check call arguments
mock_coord.route_inference.assert_called_with(expected_request)
```

### Verify gRPC Connections

```bash
# Check if gRPC server is listening
netstat -an | grep 50051
```

## Integration with CI/CD

These tests should be run as part of the CI/CD pipeline:

```yaml
# Example GitHub Actions workflow
- name: Run End-to-End Tests
  run: |
    pytest tests/test_end_to_end_integration.py -v --cov=. --cov-report=xml
    
- name: Upload Coverage
  uses: codecov/codecov-action@v3
  with:
    files: ./coverage.xml
```

## Future Enhancements

1. **Real Service Testing**: Add tests that start actual services instead of mocks
2. **Performance Benchmarking**: Add detailed performance measurement and regression detection
3. **Chaos Testing**: Add random failure injection to test resilience
4. **Load Testing**: Add sustained load tests with realistic traffic patterns
5. **Security Testing**: Add authentication and authorization tests

## Related Documentation

- [API Server Documentation](../api/README.md)
- [Coordinator Service Documentation](../coordinator/README.md)
- [Worker Implementation](../worker/README.md)
- [Deployment Guide](../DOCKER_DEPLOYMENT.md)
- [Monitoring Guide](../config/grafana/README.md)
