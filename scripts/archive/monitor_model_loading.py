#!/usr/bin/env python3
"""
Monitor model loading progress
"""

import time
import requests
import subprocess
import sys

def check_worker_log():
    """Check worker log for loading progress"""
    try:
        result = subprocess.run(['tail', '-n', '5', 'logs/worker-studio1.log'], 
                              capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error reading log: {e}"

def check_api_status():
    """Check if API is responding"""
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_models():
    """Check available models"""
    try:
        response = requests.get("http://localhost:8000/models", timeout=5)
        if response.status_code == 200:
            models = response.json()
            return [m.get("id", "") for m in models.get("data", [])]
        return []
    except:
        return []

def main():
    print("🔍 Monitoring model loading progress...")
    print("=" * 60)
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Elapsed time: {elapsed:.1f}s")
        
        # Check API status
        api_ok = check_api_status()
        print(f"🌐 API Status: {'✅ OK' if api_ok else '❌ Not responding'}")
        
        # Check available models
        models = check_models()
        if models:
            print(f"📦 Available models: {models}")
            if "qwen3-coder-480b-mlx" in models:
                print("🎉 qwen3-coder-480b-mlx is loaded and ready!")
                break
        else:
            print("📦 No models available yet")
        
        # Check worker log
        log_tail = check_worker_log()
        print(f"📋 Worker log (last 5 lines):")
        for line in log_tail.strip().split('\n')[-3:]:  # Show last 3 lines
            if line.strip():
                print(f"   {line}")
        
        print("-" * 60)
        time.sleep(10)  # Check every 10 seconds
        
        # Timeout after 10 minutes
        if elapsed > 600:
            print("⏰ Timeout reached (10 minutes). Model loading may have failed.")
            sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")
        sys.exit(0)