#!/bin/bash
# Manual script to start worker on mac-studio-2
# Run this in your terminal:

echo "To start worker on mac-studio-2, run these commands:"
echo
echo "ssh 10.201.20.12"
echo "cd ~/mlx-distributed-inference"
echo "/opt/homebrew/bin/python3 -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.11:50051"
echo
echo "This will:"
echo "- Connect worker-studio2 to coordinator on mac-studio-1"
echo "- Give access to /Volumes/Models/qwen3-480b-streaming-q4"
echo "- Enable sharding across both Mac Studios"