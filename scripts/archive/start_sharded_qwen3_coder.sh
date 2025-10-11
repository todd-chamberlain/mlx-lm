#!/bin/bash

# Start qwen3-coder-480b-mlx with automatic sharding enabled
echo "🚀 Starting qwen3-coder-480b-mlx with automatic sharding..."

# Kill any existing processes
echo "🛑 Stopping existing services..."
pkill -f "python -m coordinator.main" || true
pkill -f "python -m worker.main" || true
pkill -f "python -m api.main" || true

sleep 3

# Set sharding environment variables
export ENABLE_AUTO_SHARDING=true
export SHARD_THRESHOLD_GB=100.0
export COORDINATOR_HOST="localhost:50051"

echo "⚙️  Sharding configuration:"
echo "   - Auto sharding: $ENABLE_AUTO_SHARDING"
echo "   - Threshold: $SHARD_THRESHOLD_GB GB"
echo ""

# Start coordinator with sharding enabled
echo "🎯 Starting coordinator with sharding enabled..."
ENABLE_AUTO_SHARDING=true SHARD_THRESHOLD_GB=100.0 nohup python -m coordinator.main > logs/coordinator.log 2>&1 &
COORDINATOR_PID=$!
echo "   Coordinator started with PID: $COORDINATOR_PID"

sleep 5

# Start API server
echo "🌐 Starting API server..."
COORDINATOR_HOST="localhost:50051" nohup python -m api.main > logs/api.log 2>&1 &
API_PID=$!
echo "   API server started with PID: $API_PID"

sleep 3

echo ""
echo "✅ Services started:"
echo "   - Coordinator PID: $COORDINATOR_PID"
echo "   - API Server PID: $API_PID"
echo ""
echo "📋 Next steps:"
echo "1. Wait for services to initialize (check logs)"
echo "2. Use the API to download the model with sharding enabled:"
echo ""
echo "   curl -X POST http://localhost:8000/models/download \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"model_name\": \"qwen3-coder-480b-mlx\", \"enable_sharding\": true}'"
echo ""
echo "3. Start worker on mac-studio-2:"
echo "   ssh 10.201.20.12"
echo "   cd ~/mlx-distributed-inference"
echo "   source venv/bin/activate"
echo "   python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.10:50051"
echo ""
echo "4. Start worker on mac-studio-1:"
echo "   python -m worker.main --worker-id worker-studio1 --port 8081 --coordinator localhost:50051"
echo ""
echo "📊 Monitor logs:"
echo "   tail -f logs/coordinator.log"
echo "   tail -f logs/api.log"