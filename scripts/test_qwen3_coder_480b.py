#!/usr/bin/env python3
"""
Test script for qwen3-coder-480b-mlx model
"""

import requests
import json
import time
import sys

API_BASE = "http://localhost:8000"

def test_model_loading():
    """Test if the model is loaded and available"""
    print("🔍 Checking model availability...")
    
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            models = response.json()
            print(f"✅ Available models: {models}")
            
            if "qwen3-coder-480b-mlx" in [m.get("id", "") for m in models.get("data", [])]:
                print("✅ qwen3-coder-480b-mlx model is loaded!")
                return True
            else:
                print("❌ qwen3-coder-480b-mlx model not found in loaded models")
                return False
        else:
            print(f"❌ Failed to get models: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        return False

def test_simple_inference():
    """Test basic inference functionality"""
    print("\n🧪 Testing simple inference...")
    
    payload = {
        "model": "qwen3-coder-480b-mlx",
        "messages": [
            {"role": "user", "content": "Write a simple Python function to add two numbers"}
        ],
        "max_tokens": 150,
        "temperature": 0.1
    }
    
    try:
        print("📤 Sending inference request...")
        response = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("✅ Inference successful!")
            print(f"📝 Response: {content[:200]}...")
            return True
        else:
            print(f"❌ Inference failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return False

def test_streaming_inference():
    """Test streaming inference functionality"""
    print("\n🌊 Testing streaming inference...")
    
    payload = {
        "model": "qwen3-coder-480b-mlx",
        "messages": [
            {"role": "user", "content": "Explain what a binary search algorithm does"}
        ],
        "max_tokens": 100,
        "temperature": 0.1,
        "stream": True
    }
    
    try:
        print("📤 Sending streaming request...")
        response = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, stream=True, timeout=60)
        
        if response.status_code == 200:
            print("✅ Streaming started!")
            tokens_received = 0
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # Remove 'data: ' prefix
                        if data.strip() == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data)
                            if 'choices' in chunk and len(chunk['choices']) > 0:
                                delta = chunk['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    print(delta['content'], end='', flush=True)
                                    tokens_received += 1
                        except json.JSONDecodeError:
                            continue
            
            print(f"\n✅ Streaming completed! Received {tokens_received} token chunks")
            return True
        else:
            print(f"❌ Streaming failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error during streaming: {e}")
        return False

def test_code_generation():
    """Test code generation capabilities"""
    print("\n💻 Testing code generation...")
    
    payload = {
        "model": "qwen3-coder-480b-mlx",
        "messages": [
            {"role": "user", "content": "Write a Python class for a simple calculator with add, subtract, multiply, and divide methods"}
        ],
        "max_tokens": 300,
        "temperature": 0.1
    }
    
    try:
        print("📤 Sending code generation request...")
        response = requests.post(f"{API_BASE}/v1/chat/completions", json=payload, timeout=90)
        
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            print("✅ Code generation successful!")
            print(f"📝 Generated code:\n{content}")
            return True
        else:
            print(f"❌ Code generation failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error during code generation: {e}")
        return False

def main():
    print("🚀 Testing qwen3-coder-480b-mlx model...")
    print("=" * 50)
    
    # Wait for services to be ready
    print("⏳ Waiting for services to be ready...")
    time.sleep(5)
    
    success_count = 0
    total_tests = 4
    
    # Test 1: Model loading
    if test_model_loading():
        success_count += 1
    
    # Test 2: Simple inference
    if test_simple_inference():
        success_count += 1
    
    # Test 3: Streaming inference
    if test_streaming_inference():
        success_count += 1
    
    # Test 4: Code generation
    if test_code_generation():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! qwen3-coder-480b-mlx is working correctly!")
        sys.exit(0)
    else:
        print("⚠️  Some tests failed. Check the logs for details.")
        sys.exit(1)

if __name__ == "__main__":
    main()