# MLX Distributed Inference Server Makefile

.PHONY: help setup proto build test clean docker-build docker-up docker-down logs

# Default target
help:
	@echo "MLX Distributed Inference Server"
	@echo ""
	@echo "Available targets:"
	@echo "  setup        - Install dependencies and set up development environment"
	@echo "  proto        - Generate gRPC protocol buffer stubs"
	@echo "  build        - Build all Docker images"
	@echo "  test         - Run all tests"
	@echo "  clean        - Clean generated files and caches"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start all services with Docker Compose"
	@echo "  docker-down  - Stop all services"
	@echo "  logs         - Show logs from all services"

# Development setup
setup:
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pip install -r api/requirements.txt
	pip install -r coordinator/requirements.txt
	pip install -r worker/requirements.txt
	@echo "Development environment ready!"

# Generate protocol buffer stubs
proto:
	@echo "Generating gRPC protocol buffer stubs..."
	python scripts/generate_proto.py
	@echo "Protocol buffer stubs generated!"

# Build Docker images
build: proto
	@echo "Building Docker images..."
	docker-compose build
	@echo "Docker images built!"

# Run tests
test: proto
	@echo "Running tests..."
	python -m pytest tests/ -v
	@echo "Tests completed!"

# Clean generated files
clean:
	@echo "Cleaning generated files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*_pb2.py" -delete
	find . -type f -name "*_pb2_grpc.py" -delete
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	@echo "Cleanup completed!"

# Docker operations
docker-build: proto
	@echo "Building Docker images..."
	./docker/build.sh

docker-build-prod: proto
	@echo "Building production Docker images..."
	./docker/build.sh --version $(shell git describe --tags --always) --push

docker-up: proto
	@echo "Starting services..."
	docker-compose up -d
	@echo "Services started! Check status with 'make logs'"

docker-up-prod: proto
	@echo "Starting production services..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
	@echo "Production services started!"

docker-down:
	@echo "Stopping services..."
	docker-compose down
	@echo "Services stopped!"

docker-down-prod:
	@echo "Stopping production services..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml down
	@echo "Production services stopped!"

docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "Docker cleanup completed!"

logs:
	@echo "Showing logs from all services..."
	docker-compose logs -f

logs-prod:
	@echo "Showing production logs..."
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml logs -f

# Development helpers
dev-api:
	@echo "Starting API server in development mode..."
	cd api && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-coordinator:
	@echo "Starting coordinator in development mode..."
	cd coordinator && python main.py

dev-worker:
	@echo "Starting MLX worker in development mode..."
	cd worker && python main.py

# Linting and formatting
lint:
	@echo "Running linting..."
	flake8 api/ coordinator/ worker/ proto/ config/
	black --check api/ coordinator/ worker/ proto/ config/

format:
	@echo "Formatting code..."
	black api/ coordinator/ worker/ proto/ config/
	isort api/ coordinator/ worker/ proto/ config/

# Database operations (if needed)
db-reset:
	@echo "Resetting Redis data..."
	docker-compose exec redis redis-cli FLUSHALL

# Monitoring
monitor:
	@echo "Opening monitoring dashboards..."
	@echo "Grafana: http://localhost:3000 (admin/admin)"
	@echo "Prometheus: http://localhost:9090"
	@echo "Jaeger: http://localhost:16686"

# Health check
health:
	@echo "Checking service health..."
	curl -f http://localhost/health || echo "API server not responding"
	curl -f http://localhost:9090/-/healthy || echo "Prometheus not responding"
	curl -f http://localhost:3000/api/health || echo "Grafana not responding"