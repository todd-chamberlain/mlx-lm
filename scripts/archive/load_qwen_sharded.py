#!/usr/bin/env python3
"""
Load Qwen3-480B with automatic sharding across both Mac Studios
"""

import grpc
from proto import inference_pb2, inference_pb2_grpc

def load_model_with_sharding():
    print("🚀 Loading Qwen3-480B with Automatic Sharding")
    print("=" * 50)
    print()
    print("📍 Model locations:")
    print("   mac-studio-1: /Volumes/thunderbay/models/qwen3-480b-streaming-q4")
    print("   mac-studio-2: /Volumes/models/qwen3-480b-streaming-q4")
    print()
    
    # Connect to coordinator
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2_grpc.CoordinatorServiceStub(channel)
    
    # Try loading from mac-studio-1 first
    model_path = '/Volumes/thunderbay/models/qwen3-480b-streaming-q4'
    print(f"📡 Requesting model load: {model_path}")
    
    try:
        request = inference_pb2.ModelDownloadRequest(model_name=model_path)
        response = stub.DownloadModel(request, timeout=600)
        
        print(f"✅ Success: {response.success}")
        print(f"📝 Message: {response.message}")
        
        if response.success:
            print()
            print("🎯 Qwen3-480B loading with automatic sharding!")
            print("⏳ Loading 241 safetensors files (~460GB)...")
            print("📊 Model will be split across both Mac Studios")
            print("🔥 Each worker will load ~50% of the layers")
            print()
            print("Monitor memory usage:")
            print("  Mac-studio-1: Activity Monitor")
            print("  Mac-studio-2: ssh 10.201.20.12 'top -pid $(pgrep python)'")
        else:
            print("❌ Model loading failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    channel.close()

if __name__ == "__main__":
    load_model_with_sharding()