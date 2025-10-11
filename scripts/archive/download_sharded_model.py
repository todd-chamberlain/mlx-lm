#!/usr/bin/env python3
"""
Download qwen3-coder-480b-mlx model with automatic sharding enabled
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def check_api_status():
    """Check if API is responding"""
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def download_model_with_sharding():
    """Download model with sharding enabled"""
    print("📥 Downloading qwen3-coder-480b-mlx with automatic sharding...")
    
    payload = {
        "model_name": "qwen3-coder-480b-mlx",
        "enable_sharding": True
    }
    
    try:
        response = requests.post(f"{API_BASE}/admin/download_model", json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Download request successful!")
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            return True
        else:
            print(f"❌ Download failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False

def check_models():
    """Check available models"""
    try:
        response = requests.get(f"{API_BASE}/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return models.get("data", [])
        return []
    except:
        return []

def main():
    print("🚀 Starting sharded model download...")
    print("=" * 60)
    
    # Check API status
    if not check_api_status():
        print("❌ API server is not responding. Make sure it's running.")
        sys.exit(1)
    
    print("✅ API server is responding")
    
    # Check current models
    models = check_models()
    model_names = [m.get("id", "") for m in models]
    
    if "qwen3-coder-480b-mlx" in model_names:
        print("ℹ️  qwen3-coder-480b-mlx is already available")
        print("📦 Current models:", model_names)
    else:
        print("📦 Current models:", model_names)
        print("🔄 Model not found, initiating download...")
        
        # Download with sharding
        if download_model_with_sharding():
            print("🎉 Download initiated successfully!")
            print("📋 Note: The model will be automatically sharded across available workers")
            print("📋 Make sure to start workers on both mac-studio-1 and mac-studio-2")
        else:
            print("❌ Download failed")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📋 Next steps:")
    print("1. Start worker on mac-studio-1:")
    print("   python -m worker.main --worker-id worker-studio1 --port 8081 --coordinator localhost:50051")
    print("")
    print("2. Start worker on mac-studio-2:")
    print("   ssh 10.201.20.12")
    print("   cd ~/mlx-distributed-inference")
    print("   source venv/bin/activate")
    print("   python -m worker.main --worker-id worker-studio2 --port 8080 --coordinator 10.201.20.10:50051")
    print("")
    print("3. Test the model:")
    print("   python scripts/test_qwen3_coder_480b.py")

if __name__ == "__main__":
    main()