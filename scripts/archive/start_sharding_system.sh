#!/bin/bash
# Start the distributed sharding system for Qwen3-480B

echo "🚀 Starting Distributed Sharding System for Qwen3-480B"
echo "=================================================="
echo

# Check if coordinator is running
if pgrep -f "coordinator.main" > /dev/null; then
    echo "✅ Coordinator already running"
else
    echo "❌ Coordinator not running. Start it first:"
    echo "   python -m coordinator.main"
    exit 1
fi

echo "📍 Model locations:"
echo "   mac-studio-1: /Volumes/thunderbay/qwen3-480b-streaming-q4 (not available yet)"
echo "   mac-studio-2: /Volumes/Models/qwen3-480b-streaming-q4 ✅"
echo

echo "🖥️  Starting workers..."
echo

echo "1. Starting worker on mac-studio-1..."
python -m worker.main --worker-id worker-studio1 --port 8080 --coordinator localhost:50051 &
WORKER1_PID=$!
echo "   Worker-studio1 started (PID: $WORKER1_PID)"

echo "2. Starting worker on mac-studio-2..."
echo "   Run this command on mac-studio-2:"
echo "   ssh 10.201.20.12 'cd ~/mlx-distributed-inference && python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.11:50051'"

echo
echo "3. Once both workers are running, load the model:"
echo "   python -c \"
import grpc
from proto import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
request = inference_pb2.ModelDownloadRequest(model_name='/Volumes/Models/qwen3-480b-streaming-q4')
response = stub.DownloadModel(request, timeout=600)
print(f'Success: {response.success}, Message: {response.message}')
channel.close()
\""

echo
echo "🎯 This will automatically shard the 480B model across both Mac Studios!"