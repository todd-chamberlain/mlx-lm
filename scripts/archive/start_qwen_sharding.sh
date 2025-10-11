#!/bin/bash
# Start Qwen3-480B with sharding across both Mac Studios

echo "🚀 Starting Qwen3-480B Distributed Sharding System"
echo "=================================================="

# Check coordinator
if ! pgrep -f "coordinator.main" > /dev/null; then
    echo "❌ Coordinator not running. Start it first:"
    echo "   python -m coordinator.main"
    exit 1
fi
echo "✅ Coordinator running"

# Start worker on mac-studio-1
echo "🖥️  Starting worker-studio1..."
python -m worker.main --worker-id worker-studio1 --port 8080 --coordinator localhost:50051 > logs/worker-studio1.log 2>&1 &
WORKER1_PID=$!
echo "✅ Worker-studio1 started (PID: $WORKER1_PID)"

# Start worker on mac-studio-2
echo "🖥️  Starting worker-studio2..."
ssh 10.201.20.12 "cd ~/mlx-distributed-inference && source venv/bin/activate && python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.10:50051 --model-path '/Volumes/thunderbay/models/qwen3-480b-streaming-q4' --model-name 'qwen3-480b-streaming-q4' > logs/worker-studio2.log 2>&1 &" &
WORKER2_PID=$!
echo "✅ Worker-studio2 started (PID: $WORKER2_PID)"

# Wait for workers to register
echo "⏳ Waiting for workers to register..."
sleep 5

# Load the model with sharding
echo "📡 Loading Qwen3-480B with automatic sharding..."
python -c "
import grpc
from proto import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel('localhost:50051')
stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
request = inference_pb2.ModelDownloadRequest(model_name='/Volumes/thunderbay/models/qwen3-480b-streaming-q4')
response = stub.DownloadModel(request, timeout=600)

print(f'✅ Success: {response.success}')
print(f'📝 Message: {response.message}')

if response.success:
    print()
    print('🎯 Qwen3-480B loading with automatic sharding!')
    print('⏳ Loading 241 safetensors files (~460GB)...')
    print('📊 Model will be split across both Mac Studios')
    print('🔥 Each worker will load ~50% of the layers')
    print()
    print('Monitor progress:')
    print('  Mac-studio-1: tail -f logs/worker-studio1.log')
    print('  Mac-studio-2: ssh 10.201.20.12 \"tail -f ~/mlx-distributed-inference/logs/worker-studio2.log\"')
else:
    print('❌ Model loading failed')

channel.close()
"

echo
echo "🎉 Sharding system started!"
echo "Monitor memory usage to see the model loading across both machines."