#!/usr/bin/env python3
"""
Simple script to load Qwen model with manual sharding
"""

import subprocess
import time

print("🚀 Loading Qwen3-Coder-480B with manual sharding...")
print("=" * 60)

# Stop current workers
print("📍 Stopping current workers...")
subprocess.run(["pkill", "-f", "worker.main"], check=False)
subprocess.run(["ssh", "10.201.20.12", "pkill -f 'worker.main'"], check=False)
time.sleep(3)

# Start worker-1 with first half of layers (0-30)
print("📍 Starting worker-studio1 with layers 0-30...")
subprocess.Popen([
    "./venv/bin/python", "-m", "worker.main",
    "--worker-id", "worker-studio1",
    "--port", "8080",
    "--coordinator", "localhost:50051",
    "--model-path", "/Volumes/thunderbay/models/qwen3-coder-480b-mlx",
    "--model-name", "qwen3-coder-480b-mlx",
    "--start-layer", "0",
    "--end-layer", "30"
], stdout=open("logs/worker-studio1.log", "w"), stderr=subprocess.STDOUT)

time.sleep(3)

# Start worker-2 with second half of layers (31-61)
print("📍 Starting worker-studio2 with layers 31-61...")
subprocess.run([
    "ssh", "10.201.20.12",
    "cd ~/mlx-distributed-inference && source venv/bin/activate && "
    "nohup python -m worker.main "
    "--worker-id worker-studio2 "
    "--port 8080 "
    "--coordinator 10.201.20.10:50051 "
    "--model-path '/Volumes/models/mlx/qwen3-coder-480b-mlx' "
    "--model-name 'qwen3-coder-480b-mlx' "
    "--start-layer 31 "
    "--end-layer 61 "
    "> logs/worker-studio2.log 2>&1 &"
], shell=True)

print("\n✅ Workers started with sharding!")
print("   - Worker-studio1: layers 0-30")
print("   - Worker-studio2: layers 31-61")
print("\nMonitor with:")
print("  tail -f logs/worker-studio1.log")
print("  ssh 10.201.20.12 'tail -f ~/mlx-distributed-inference/logs/worker-studio2.log'")
