#!/bin/bash
# Start worker on mac-studio-2 with access to the Qwen3-480B model

echo "🚀 Starting worker on mac-studio-2..."
echo "📍 Model path: /Volumes/Models/qwen3-480b-streaming-q4"
echo "🔗 Coordinator: 10.201.20.11:50051"
echo

ssh 10.201.20.12 "cd ~/mlx-distributed-inference && python -m worker.main \
  --worker-id worker-studio2 \
  --port 8080 \
  --coordinator 10.201.20.11:50051 \
  --model-path /Volumes/Models/qwen3-480b-streaming-q4 \
  > logs/worker-studio2.log 2>&1 &"

echo "✅ Worker started on mac-studio-2"
echo "📊 Monitor with: ssh 10.201.20.12 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'"