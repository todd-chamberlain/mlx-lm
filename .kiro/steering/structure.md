# Project Structure

## Top-Level Organization

```
mlx-distributed-inference/
├── api/              # FastAPI REST API server
├── coordinator/      # Coordinator service (worker management, routing)
├── worker/           # MLX inference workers
├── proto/            # gRPC protocol definitions
├── config/           # Configuration files and loaders
├── common/           # Shared utilities (logging, caching, memory, recovery)
├── docker/           # Docker build files and scripts
├── scripts/          # Deployment and utility scripts
├── tests/            # Test suite
├── docs/             # Documentation
└── logs/             # Runtime logs
```

## Service Directories

### api/
FastAPI server providing OpenAI-compatible endpoints.

Key files:
- `main.py` - FastAPI app, endpoints, lifespan management
- `models.py` - Pydantic request/response models
- `coordinator_client.py` - gRPC client to coordinator
- `auth.py` - Authentication and authorization
- `middleware.py` - CORS, rate limiting, request logging
- `streaming.py` - SSE streaming response handling
- `metrics.py` - Prometheus metrics
- `tracing.py` - OpenTelemetry distributed tracing

### coordinator/
Central coordination service managing workers and routing requests.

Key files:
- `main.py` - Service entry point
- `service.py` - gRPC server implementation
- `worker_pool.py` - Worker registration and health tracking
- `request_router.py` - Load balancing and worker selection
- `batch_manager.py` / `batch_processor.py` - Batch processing
- `cache_coordinator.py` - Distributed cache coordination
- `health_monitor.py` - Worker health checks
- `metrics.py` - Metrics collection

### worker/
MLX inference engines running model partitions.

Key files:
- `main.py` - Worker entry point and lifecycle
- `model_manager.py` - Model loading, caching, memory management
- `inference_engine.py` - MLX inference execution
- `model_partitioner.py` - Layer distribution logic
- `grpc_service.py` - gRPC server for inference requests
- `coordinator_client.py` - Registration and heartbeat
- `distributed_forward.py` - Distributed layer execution
- `distributed_generation.py` - Token generation across workers
- `activation_transfer.py` - Inter-worker activation passing

### proto/
Protocol Buffer definitions for gRPC communication.

- `distributed.proto` - Worker registration, health, coordination
- `inference.proto` - Inference requests and responses
- `*_pb2.py` / `*_pb2_grpc.py` - Generated Python stubs
- `errors.py` - Custom exception hierarchy
- `validation.py` - Request validation utilities

### config/
Configuration management and YAML files.

- `base.py` - Base configuration classes
- `config_loader.py` - YAML loading with env overrides
- `api_config.py` / `coordinator_config.py` / `worker_config.py` - Service configs
- `*.yaml` - Configuration files for each service
- `nginx/` / `redis/` / `prometheus/` / `grafana/` - Service-specific configs

### common/
Shared utilities used across services.

Key modules:
- `logging_system.py` - Structured logging with correlation IDs
- `kv_cache.py` / `activation_cache.py` - Caching implementations
- `memory_pool.py` - Tensor memory pooling
- `gc_optimizer.py` - Garbage collection tuning
- `circuit_breaker.py` - Circuit breaker pattern
- `retry.py` / `timeout.py` - Resilience utilities
- `error_recovery.py` / `service_recovery.py` - Recovery strategies
- `resource_monitor.py` - System resource monitoring

### tests/
Comprehensive test suite organized by type.

Test categories (use pytest markers):
- `test_*.py` - Unit tests (fast, isolated)
- `test_*_integration.py` - Integration tests
- `test_performance_*.py` - Performance benchmarks
- `smoke/` - Deployment smoke tests

## Configuration Files

### Docker
- `docker-compose.yml` - Development stack
- `docker-compose.prod.yml` - Production overrides
- `docker/Dockerfile.*` - Service-specific images

### Build & Test
- `Makefile` - Build automation
- `pytest.ini` - Test configuration
- `requirements.txt` - Shared dependencies
- `*/requirements.txt` - Service-specific dependencies

### Deployment
- `deploy_*.sh` - Deployment scripts for different scenarios
- `scripts/deploy_production_*.sh` - Production deployment per node

## Naming Conventions

- **Services**: Lowercase with underscores (`coordinator_client.py`)
- **Classes**: PascalCase (`ModelManager`, `InferenceEngine`)
- **Functions**: snake_case (`load_model`, `route_inference`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_WORKERS`, `DEFAULT_PORT`)
- **Proto messages**: PascalCase (`InferenceRequest`, `WorkerInfo`)
- **Config files**: lowercase with hyphens (`docker-compose.yml`)

## Import Patterns

```python
# Standard library
import os
import asyncio
from typing import Optional, Dict

# Third-party
import grpc
from fastapi import FastAPI
import mlx.core as mx

# Local absolute imports
from config.base import load_config
from common.logging_system import get_logger
from proto.inference_pb2 import InferenceRequest
```

## Logging Conventions

Use structured logging with context:
```python
logger.info("Inference completed", extra={
    "request_id": request_id,
    "worker_id": worker_id,
    "latency_ms": latency,
    "tokens": token_count
})
```
