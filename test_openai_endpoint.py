#!/usr/bin/env python3
"""
Test OpenAI-Compatible API Endpoint with Distributed Layers

This tests the full stack:
1. Load model with distributed layers
2. Serve via OpenAI-compatible API
3. Test chat completions endpoint
"""

import asyncio
import sys
import time
import requests
import json

sys.path.insert(0, '.')


class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")


def print_test(name: str, status: str, details: str = ""):
    if status == "PASS":
        symbol = f"{Colors.GREEN}✓{Colors.NC}"
        status_text = f"{Colors.GREEN}PASS{Colors.NC}"
    elif status == "FAIL":
        symbol = f"{Colors.RED}✗{Colors.NC}"
        status_text = f"{Colors.RED}FAIL{Colors.NC}"
    elif status == "WARN":
        symbol = f"{Colors.YELLOW}⚠{Colors.NC}"
        status_text = f"{Colors.YELLOW}WARN{Colors.NC}"
    else:
        symbol = f"{Colors.CYAN}ℹ{Colors.NC}"
        status_text = f"{Colors.CYAN}INFO{Colors.NC}"
    
    print(f"  {symbol} {name:.<50} {status_text}")
    if details:
        print(f"     {Colors.CYAN}{details}{Colors.NC}")


def test_health_endpoint(base_url: str):
    """Test health endpoint"""
    print_header("TEST 1: Health Endpoint")
    
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        data = response.json()
        
        print_test("Health Endpoint", "PASS", f"Status: {response.status_code}")
        print_test("API Status", "INFO", data.get('status', 'unknown'))
        print_test("Workers", "INFO", f"{data.get('healthy_workers', 0)}/{data.get('total_workers', 0)}")
        
        return data.get('status') == 'healthy'
        
    except Exception as e:
        print_test("Health Endpoint", "FAIL", str(e))
        return False


def test_models_endpoint(base_url: str, api_key: str):
    """Test models endpoint"""
    print_header("TEST 2: Models Endpoint")
    
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=5)
        data = response.json()
        
        print_test("Models Endpoint", "PASS", f"Status: {response.status_code}")
        
        if 'data' in data:
            models = data['data']
            print_test("Available Models", "INFO", f"{len(models)} models")
            for model in models:
                print(f"     • {model.get('id', 'unknown')}")
        
        return response.status_code == 200
        
    except Exception as e:
        print_test("Models Endpoint", "FAIL", str(e))
        return False


def test_chat_completion(base_url: str, api_key: str, prompt: str):
    """Test chat completion endpoint"""
    print_header("TEST 3: Chat Completion (Non-Streaming)")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "default",  # Use default model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        print_test("Request", "INFO", f'Prompt: "{prompt}"')
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            
            print_test("Response", "PASS", f"Status: {response.status_code}")
            print_test("Latency", "INFO", f"{latency:.2f}ms")
            
            if 'choices' in data and len(data['choices']) > 0:
                content = data['choices'][0]['message']['content']
                print_test("Generated Text", "PASS", f'"{content[:100]}..."')
                
                if 'usage' in data:
                    usage = data['usage']
                    print_test("Token Usage", "INFO", 
                              f"Prompt: {usage.get('prompt_tokens', 0)}, "
                              f"Completion: {usage.get('completion_tokens', 0)}, "
                              f"Total: {usage.get('total_tokens', 0)}")
                
                return True
            else:
                print_test("Generated Text", "FAIL", "No choices in response")
                return False
        else:
            print_test("Response", "FAIL", f"Status: {response.status_code}")
            print(f"     Error: {response.text}")
            return False
            
    except Exception as e:
        print_test("Chat Completion", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_streaming_completion(base_url: str, api_key: str, prompt: str):
    """Test streaming chat completion"""
    print_header("TEST 4: Chat Completion (Streaming)")
    
    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "default",  # Use default model
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 30,
            "temperature": 0.7,
            "stream": True
        }
        
        print_test("Request", "INFO", f'Prompt: "{prompt}"')
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        )
        
        if response.status_code == 200:
            print_test("Response", "PASS", f"Status: {response.status_code}")
            
            print(f"\n     {Colors.CYAN}Streaming output:{Colors.NC} ", end="", flush=True)
            
            chunks = 0
            full_text = ""
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data_str = line[6:]
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    print(content, end="", flush=True)
                                    full_text += content
                                    chunks += 1
                        except json.JSONDecodeError:
                            pass
            
            latency = (time.time() - start_time) * 1000
            
            print(f"\n")
            print_test("Chunks Received", "INFO", f"{chunks} chunks")
            print_test("Total Latency", "INFO", f"{latency:.2f}ms")
            print_test("Generated Text", "PASS", f'"{full_text}"')
            
            return True
        else:
            print_test("Response", "FAIL", f"Status: {response.status_code}")
            return False
            
    except Exception as e:
        print_test("Streaming Completion", "FAIL", str(e))
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_requests(base_url: str, api_key: str, num_requests: int = 5):
    """Test concurrent requests"""
    print_header(f"TEST 5: Concurrent Requests ({num_requests} requests)")
    
    import concurrent.futures
    
    def make_request(i):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "default",  # Use default model
            "messages": [
                {"role": "user", "content": f"Count to {i+1}"}
            ],
            "max_tokens": 20,
            "temperature": 0.7
        }
        
        start = time.time()
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            latency = (time.time() - start) * 1000
            
            if response.status_code == 200:
                return {'success': True, 'latency': latency, 'request': i+1}
            else:
                return {'success': False, 'latency': latency, 'request': i+1, 'error': response.status_code}
        except Exception as e:
            return {'success': False, 'latency': (time.time() - start) * 1000, 'request': i+1, 'error': str(e)}
    
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        
        print_test("Total Requests", "INFO", f"{num_requests}")
        print_test("Successful", "PASS" if len(successful) == num_requests else "WARN", 
                   f"{len(successful)}/{num_requests}")
        
        if successful:
            avg_latency = sum(r['latency'] for r in successful) / len(successful)
            min_latency = min(r['latency'] for r in successful)
            max_latency = max(r['latency'] for r in successful)
            
            print_test("Avg Latency", "INFO", f"{avg_latency:.2f}ms")
            print_test("Min/Max Latency", "INFO", f"{min_latency:.2f}ms / {max_latency:.2f}ms")
        
        if failed:
            print_test("Failed Requests", "WARN", f"{len(failed)} failed")
            for r in failed:
                print(f"     Request {r['request']}: {r.get('error', 'unknown error')}")
        
        return len(successful) == num_requests
        
    except Exception as e:
        print_test("Concurrent Requests", "FAIL", str(e))
        return False


def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test OpenAI-compatible API endpoint')
    parser.add_argument('--url', default='http://localhost:8000', help='API base URL')
    parser.add_argument('--api-key', default='your-secret-api-key-here', help='API key')
    parser.add_argument('--prompt', default='The capital of France is', help='Test prompt')
    parser.add_argument('--concurrent', type=int, default=5, help='Number of concurrent requests')
    
    args = parser.parse_args()
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     OpenAI-Compatible API Endpoint Test                           ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    print(f"API URL: {args.url}")
    print(f"Test Prompt: {args.prompt}")
    
    # Run tests
    results = []
    
    results.append(("Health", test_health_endpoint(args.url)))
    results.append(("Models", test_models_endpoint(args.url, args.api_key)))
    results.append(("Chat Completion", test_chat_completion(args.url, args.api_key, args.prompt)))
    results.append(("Streaming", test_streaming_completion(args.url, args.api_key, "Hello")))
    results.append(("Concurrent", test_concurrent_requests(args.url, args.api_key, args.concurrent)))
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.NC}" if result else f"{Colors.RED}✗ FAIL{Colors.NC}"
        print(f"  {name:.<50} {status}")
    
    print()
    if passed == total:
        print(f"{Colors.GREEN}✅ All tests passed! ({passed}/{total}){Colors.NC}")
        return 0
    else:
        print(f"{Colors.YELLOW}⚠ Some tests failed ({passed}/{total}){Colors.NC}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
