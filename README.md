# MLX Distributed Inference Server

A production-ready hybrid MLX distributed inference server that combines containerized orchestration services with native MLX workers running directly on Apple Silicon hardware.

## Architecture

The system uses a hybrid architecture:
- **Containerized Services**: API server, coordinator, and monitoring stack run in Docker containers
- **Native Workers**: MLX inference workers run natively on macOS hosts for maximum GPU performance
- **OpenAI Compatibility**: Provides OpenAI-compatible API endpoints for seamless integration

## Project Structure

```
mlx-distributed-inference/
├── api/                    # FastAPI server (containerized)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── __init__.py
├── coordinator/            # Coordinator service (containerized)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── __init__.py
├── worker/                 # MLX workers (native)
│   ├── requirements.txt
│   └── __init__.py
├── proto/                  # Protocol buffer definitions
│   ├── inference.proto
│   ├── validation.py
│   └── __init__.py
├── config/                 # Configuration files
│   ├── base.py
│   ├── nginx/
│   ├── redis/
│   └── prometheus/
├── scripts/                # Utility scripts
│   └── generate_proto.py
├── docker-compose.yml      # Container orchestration
├── requirements.txt        # Shared dependencies
└── Makefile               # Development commands
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- Apple Silicon Mac (for MLX workers)

### Setup

1. **Clone and setup the project:**
   ```bash
   git clone <repository-url>
   cd mlx-distributed-inference
   make setup
   ```

2. **Generate protocol buffer stubs:**
   ```bash
   make proto
   ```

3. **Start containerized services:**
   ```bash
   make docker-up
   ```

4. **Configure and start MLX workers** (on each Mac):
   ```bash
   # Install MLX dependencies
   pip install -r worker/requirements.txt
   
   # Configure worker (edit config/worker.yaml)
   # Start worker
   make dev-worker
   ```

### Development

- **Start API server in development mode:**
  ```bash
  make dev-api
  ```

- **Start coordinator in development mode:**
  ```bash
  make dev-coordinator
  ```

- **View logs:**
  ```bash
  make logs
  ```

- **Run tests:**
  ```bash
  make test
  ```

## Configuration

### Environment Variables

Key environment variables for configuration:

- `ENABLE_AUTH`: Enable API key authentication
- `API_KEYS`: Comma-separated list of valid API keys
- `COORDINATOR_HOST`: Coordinator service address
- `REDIS_URL`: Redis connection URL
- `ENABLE_METRICS`: Enable Prometheus metrics
- `JAEGER_ENDPOINT`: Jaeger tracing endpoint

### Configuration Files

- `config/nginx/nginx.conf`: Load balancer configuration
- `config/redis/redis.conf`: Redis cache configuration
- `config/prometheus/prometheus.yml`: Metrics collection configuration

## API Endpoints

The server provides OpenAI-compatible endpoints:

- `POST /v1/chat/completions`: Chat completion requests
- `GET /health`: System health status
- `GET /metrics`: Prometheus metrics
- `GET /models`: Available models

## Monitoring

Access monitoring dashboards:

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

## Development Commands

```bash
make help          # Show all available commands
make setup         # Install dependencies
make proto         # Generate gRPC stubs
make build         # Build Docker images
make test          # Run tests
make clean         # Clean generated files
make docker-up     # Start all services
make docker-down   # Stop all services
make logs          # View service logs
make monitor       # Open monitoring dashboards
```

## Next Steps

This project structure provides the foundation for implementing the MLX distributed inference server. The next tasks involve:

1. Implementing the gRPC communication layer
2. Building the coordinator service with worker management
3. Creating the native MLX worker implementation
4. Developing the FastAPI server with OpenAI compatibility
5. Setting up monitoring and observability
6. Adding comprehensive error handling and resilience

See the `tasks.md` file in the spec directory for the complete implementation plan.

## License

[Add your license here]