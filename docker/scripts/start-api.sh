#!/bin/bash
set -e

# MLX API Server startup script

echo "Starting MLX API Server..."

# Set default environment variables if not provided
export PORT=${PORT:-8000}
export HOST=${HOST:-0.0.0.0}
export WORKERS=${WORKERS:-1}
export LOG_LEVEL=${LOG_LEVEL:-info}
export COORDINATOR_HOST=${COORDINATOR_HOST:-coordinator:50051}

# Enable authentication if API keys are provided
if [ -n "$API_KEYS" ]; then
    export ENABLE_AUTH=true
    echo "Authentication enabled"
else
    export ENABLE_AUTH=false
    echo "Authentication disabled"
fi

# Enable tracing if Jaeger endpoint is provided
if [ -n "$JAEGER_ENDPOINT" ]; then
    export ENABLE_TRACING=true
    echo "Distributed tracing enabled: $JAEGER_ENDPOINT"
else
    export ENABLE_TRACING=false
    echo "Distributed tracing disabled"
fi

# Wait for coordinator to be ready
echo "Waiting for coordinator at $COORDINATOR_HOST..."
timeout=60
while [ $timeout -gt 0 ]; do
    if python -c "
import grpc
import sys
try:
    channel = grpc.insecure_channel('$COORDINATOR_HOST')
    grpc.channel_ready_future(channel).result(timeout=5)
    channel.close()
    print('Coordinator is ready')
    sys.exit(0)
except:
    sys.exit(1)
    "; then
        break
    fi
    echo "Coordinator not ready, waiting..."
    sleep 2
    timeout=$((timeout - 2))
done

if [ $timeout -le 0 ]; then
    echo "Timeout waiting for coordinator"
    exit 1
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

# Start the API server
echo "Starting API server on $HOST:$PORT with $WORKERS workers"
exec uvicorn api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level "$LOG_LEVEL" \
    --access-log \
    --loop uvloop \
    --http httptools