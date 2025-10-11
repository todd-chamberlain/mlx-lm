#!/usr/bin/env python3
"""
Download qwen3-coder-480b-mlx model with automatic sharding
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def check_workers():
    """Check how many workers are available"""
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            health = response.json()
            worker_count = health.get("total_workers", 0)
            healthy_count = health.get("healthy_workers", 0)
            print(f"✅ Available workers: {worker_count} (healthy: {healthy_count})")
            return healthy_count
        else:
            print(f"❌ Failed to get health status: {response.status_code}")
            return 0
    except Exception as e:
        print(f"❌ Error checking workers: {e}")
        return 0

def download_model_with_sharding():
    """Download model with sharding enabled"""
    print("📥 Downloading qwen3-coder-480b-mlx with automatic sharding...")
    
    # First check if we have enough workers
    worker_count = check_workers()
    if worker_count < 2:
        print("❌ Need at least 2 workers for sharding. Please start worker on mac-studio-2")
        return False
    
    payload = {
        "model_name": "qwen3-coder-480b-mlx",
        "enable_sharding": True,
        "force_redownload": False,
        "num_layers": 62,  # From config.json: "num_hidden_layers": 62
        "model_size_gb": 250.0  # Approximate size based on 62 files of ~4GB each
    }
    
    try:
        print("📤 Sending download request with sharding enabled...")
        response = requests.post(f"{API_BASE}/admin/download_model", json=payload, timeout=300)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Download request successful!")
            print(f"📋 Response: {json.dumps(result, indent=2)}")
            
            # Check if sharding was applied
            if result.get("sharded", False):
                print("🎯 Model will be sharded across workers!")
                shard_plan = result.get("shard_plan", {})
                for worker_id, assignment in shard_plan.items():
                    print(f"   - {worker_id}: layers {assignment.get('start_layer', 0)}-{assignment.get('end_layer', 0)}")
            else:
                print("⚠️  Model was not sharded (may be below threshold or insufficient workers)")
            
            return True
        else:
            print(f"❌ Download failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during download: {e}")
        return False

def monitor_download_progress():
    """Monitor download progress"""
    print("\n🔍 Monitoring download progress...")
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        try:
            # Check model status
            response = requests.get(f"{API_BASE}/models")
            if response.status_code == 200:
                models = response.json()
                model_ids = [m.get("id", "") for m in models.get("data", [])]
                
                if "qwen3-coder-480b-mlx" in model_ids:
                    print("🎉 Model is loaded and available!")
                    
                    # Get shard status
                    try:
                        shard_response = requests.get(f"{API_BASE}/admin/models/qwen3-coder-480b-mlx/shards")
                        if shard_response.status_code == 200:
                            shard_info = shard_response.json()
                            print("📊 Shard status:")
                            for worker_id, shard in shard_info.get("shards", {}).items():
                                status = shard.get("status", "unknown")
                                layers = f"{shard.get('start_layer', 0)}-{shard.get('end_layer', 0)}"
                                print(f"   - {worker_id}: layers {layers} ({status})")
                    except:
                        pass
                    
                    return True
                else:
                    print(f"⏳ Still loading... (elapsed: {elapsed:.1f}s)")
            
        except Exception as e:
            print(f"❌ Error checking status: {e}")
        
        time.sleep(10)
        
        # Timeout after 10 minutes
        if elapsed > 600:
            print("⏰ Timeout reached. Download may have failed.")
            return False

def main():
    print("🚀 Starting qwen3-coder-480b-mlx download with sharding...")
    print("=" * 60)
    
    # Download with sharding
    if not download_model_with_sharding():
        print("❌ Download failed")
        sys.exit(1)
    
    # Monitor progress
    if monitor_download_progress():
        print("\n🎉 Model successfully loaded with sharding!")
        print("Ready to test inference!")
    else:
        print("\n❌ Download monitoring failed")
        sys.exit(1)

if __name__ == "__main__":
    main()