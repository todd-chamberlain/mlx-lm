#!/bin/bash
set -e

# MLX Coordinator Service startup script

echo "Starting MLX Coordinator Service..."

# Set default environment variables
export GRPC_PORT=${GRPC_PORT:-50051}
export METRICS_PORT=${METRICS_PORT:-9090}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export MAX_WORKERS=${MAX_WORKERS:-100}
export HEALTH_CHECK_INTERVAL=${HEALTH_CHECK_INTERVAL:-30}
export WORKER_TIMEOUT=${WORKER_TIMEOUT:-60}

# Redis configuration for worker state persistence
export REDIS_HOST=${REDIS_HOST:-redis}
export REDIS_PORT=${REDIS_PORT:-6379}
export REDIS_DB=${REDIS_DB:-0}

# Enable tracing if Jaeger endpoint is provided
if [ -n "$JAEGER_ENDPOINT" ]; then
    export ENABLE_TRACING=true
    echo "Distributed tracing enabled: $JAEGER_ENDPOINT"
else
    export ENABLE_TRACING=false
    echo "Distributed tracing disabled"
fi

# Wait for Redis to be ready if configured
if [ "$REDIS_HOST" != "none" ]; then
    echo "Waiting for Redis at $REDIS_HOST:$REDIS_PORT..."
    timeout=60
    while [ $timeout -gt 0 ]; do
        if python -c "
import redis
import sys
try:
    r = redis.Redis(host='$REDIS_HOST', port=$REDIS_PORT, db=$REDIS_DB, socket_timeout=5)
    r.ping()
    print('Redis is ready')
    sys.exit(0)
except:
    sys.exit(1)
        "; then
            break
        fi
        echo "Redis not ready, waiting..."
        sleep 2
        timeout=$((timeout - 2))
    done
    
    if [ $timeout -le 0 ]; then
        echo "Warning: Timeout waiting for Redis, continuing without persistence"
    fi
fi

# Generate protocol buffers if needed
if [ ! -f "proto/inference_pb2.py" ]; then
    echo "Generating protocol buffers..."
    python -m grpc_tools.protoc \
        --proto_path=proto \
        --python_out=proto \
        --grpc_python_out=proto \
        proto/inference.proto
fi

# Create necessary directories
mkdir -p /tmp/coordinator/logs

# Start the coordinator service
echo "Starting coordinator service on port $GRPC_PORT"
echo "Metrics server on port $METRICS_PORT"
echo "Max workers: $MAX_WORKERS"
echo "Health check interval: ${HEALTH_CHECK_INTERVAL}s"
echo "Worker timeout: ${WORKER_TIMEOUT}s"

exec python coordinator/main.py