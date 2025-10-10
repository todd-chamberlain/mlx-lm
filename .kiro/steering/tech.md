# Technology Stack

## Core Technologies

- **Python 3.9+** - Primary language
- **MLX** - Apple's ML framework for Apple Silicon (Metal GPU acceleration)
- **gRPC** - Inter-service communication (coordinator ↔ workers)
- **FastAPI** - REST API server with async support
- **Protocol Buffers** - Service definitions and serialization

## Key Libraries

### ML & Inference
- `mlx` (≥0.15.0) - Apple Silicon ML framework
- `mlx-lm` (≥0.8.0) - Language model utilities
- `transformers` - HuggingFace model support
- `tokenizers` - Fast tokenization

### API & Web
- `fastapi` - Async web framework
- `uvicorn` - ASGI server
- `httpx` / `aiohttp` - Async HTTP clients
- `python-jose` - JWT authentication
- `slowapi` - Rate limiting

### Distributed Systems
- `grpcio` / `grpcio-tools` - gRPC framework
- `redis` / `aioredis` - Caching and state management
- `celery` - Background task processing (optional)

### Observability
- `prometheus-client` - Metrics collection
- `opentelemetry-*` - Distributed tracing
- `structlog` - Structured logging

### Configuration & Utilities
- `pydantic` - Data validation and settings
- `pyyaml` - Configuration files
- `python-dotenv` - Environment variables
- `psutil` - System monitoring

## Build System

### Makefile Targets

```bash
make setup          # Install all dependencies
make proto          # Generate gRPC stubs from .proto files
make test           # Run pytest test suite
make docker-build   # Build Docker images
make docker-up      # Start all services
make docker-down    # Stop all services
make clean          # Remove generated files and caches
```

### Docker Compose

Services defined in `docker-compose.yml`:
- `redis` - State persistence
- `coordinator` - Worker coordination
- `api` - REST API server
- `nginx` - Load balancer
- `monitoring` - Prometheus + Grafana
- `jaeger` - Distributed tracing

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test categories
pytest -m unit          # Fast unit tests
pytest -m integration   # Integration tests
pytest -m performance   # Performance benchmarks
pytest -m smoke         # Deployment smoke tests

# With coverage
pytest --cov=. --cov-report=html
```

### Protocol Buffers

Generate gRPC stubs:
```bash
python scripts/generate_proto.py
# or
make proto
```

Proto files in `proto/`:
- `distributed.proto` - Coordinator ↔ Worker communication
- `inference.proto` - Inference request/response definitions

## Development Commands

### Start Services Locally

```bash
# Coordinator
python -m coordinator.main

# Worker
python -m worker.main --model-name <model> --coordinator localhost:50051

# API Server
python -m api.main
# or with reload
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables

Key variables (see `.env.example`):
- `COORDINATOR_HOST` - Coordinator address
- `WORKER_ID` - Unique worker identifier
- `MODEL_PATH` / `MODEL_NAME` - Model configuration
- `HF_HOME` - HuggingFace cache location
- `LOG_LEVEL` - Logging verbosity
- `ENABLE_AUTH` - API authentication toggle
- `API_KEYS` - Comma-separated API keys

## Code Style

- **Async/await** - Prefer async for I/O operations
- **Type hints** - Use Pydantic models and Python type annotations
- **Logging** - Structured logging with context (request_id, worker_id)
- **Error handling** - Custom exception hierarchy in `proto/errors.py`
- **Configuration** - YAML files + environment variable overrides
