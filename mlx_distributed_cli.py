#!/usr/bin/env python3
"""
MLX Distributed Inference CLI

Unified command-line interface for managing distributed inference.
"""

import sys
import argparse
import requests
import json
import subprocess
import time
from pathlib import Path


class Colors:
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    BOLD = '\033[1m'
    NC = '\033[0m'


class DistributedCLI:
    """CLI for distributed inference management"""
    
    def __init__(self, api_url: str = "http://localhost:8080"):
        self.api_url = api_url
    
    def _print(self, msg: str, color: str = Colors.NC):
        """Print colored message"""
        print(f"{color}{msg}{Colors.NC}")
    
    def _api_call(self, method: str, endpoint: str, **kwargs):
        """Make API call"""
        url = f"{self.api_url}{endpoint}"
        try:
            if method == "GET":
                response = requests.get(url, **kwargs)
            elif method == "POST":
                response = requests.post(url, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self._print(f"✗ API Error: {e}", Colors.RED)
            return None
    
    def health(self):
        """Check server health"""
        self._print("Checking server health...", Colors.CYAN)
        data = self._api_call("GET", "/health")
        
        if data:
            status = data.get('status', 'unknown')
            color = Colors.GREEN if status == 'healthy' else Colors.YELLOW
            
            self._print(f"\nStatus: {status}", color)
            self._print(f"Model: {data.get('model', 'none')}")
            self._print(f"Nodes: {data.get('nodes', 0)}")
            self._print(f"Active Requests: {data.get('active_requests', 0)}")
            self._print(f"Loading: {data.get('is_loading', False)}")
    
    def list_models(self):
        """List available models"""
        self._print("Available Models:", Colors.CYAN)
        data = self._api_call("GET", "/admin/models")
        
        if data:
            self._print(f"\nCurrent: {data.get('current_model', 'none')}", Colors.GREEN)
            self._print("\nSuggested Models:", Colors.BOLD)
            
            for model in data.get('suggested_models', []):
                self._print(f"\n  • {model['name']}")
                self._print(f"    Size: {model['size']}, {model['description']}")
    
    def download(self, model_name: str):
        """Download model in background"""
        self._print(f"Starting background download: {model_name}", Colors.CYAN)
        
        data = self._api_call("POST", f"/admin/download_model?model_name={model_name}")
        
        if data:
            if data.get('status') == 'started':
                self._print(f"\n✓ Download started", Colors.GREEN)
                self._print(f"Task ID: {data.get('task_id')}")
                self._print(f"\nCheck progress: mlx-cli download-status {model_name}")
            else:
                self._print(f"\n⚠ {data.get('status')}: {data.get('message', '')}", Colors.YELLOW)
    
    def download_status(self, model_name: str = None):
        """Check download status"""
        if model_name:
            self._print(f"Download status for: {model_name}", Colors.CYAN)
            endpoint = f"/admin/download_status?model_name={model_name}"
        else:
            self._print("All downloads:", Colors.CYAN)
            endpoint = "/admin/download_status"
        
        data = self._api_call("GET", endpoint)
        
        if data:
            print(json.dumps(data, indent=2))
    
    def load_model(self, model_name: str, drain_timeout: int = 30):
        """Load/swap to a different model"""
        self._print(f"Loading model: {model_name}", Colors.CYAN)
        self._print(f"Drain timeout: {drain_timeout}s")
        
        endpoint = f"/admin/load_model?model_name={model_name}&drain_timeout={drain_timeout}"
        data = self._api_call("POST", endpoint)
        
        if data:
            if data.get('success'):
                self._print(f"\n✓ Model loaded successfully", Colors.GREEN)
                self._print(f"Old: {data.get('old_model')}")
                self._print(f"New: {data.get('new_model')}")
                self._print(f"Nodes: {data.get('nodes')}")
            else:
                self._print(f"\n✗ Load failed", Colors.RED)
    
    def test(self, prompt: str = "Hello", max_tokens: int = 20):
        """Test inference"""
        self._print(f"Testing inference...", Colors.CYAN)
        self._print(f"Prompt: {prompt}")
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            data = response.json()
            
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            
            self._print(f"\n✓ Response:", Colors.GREEN)
            self._print(content)
            self._print(f"\nTokens: {usage.get('total_tokens', 0)}")
            
        except Exception as e:
            self._print(f"\n✗ Test failed: {e}", Colors.RED)
    
    def start_server(self, model: str, port: int = 8080, background: bool = False):
        """Start the distributed inference server"""
        self._print(f"Starting server...", Colors.CYAN)
        self._print(f"Model: {model}")
        self._print(f"Port: {port}")
        
        cmd = [
            "python3", "run_distributed_inference.py",
            "--model", model,
            "--port", str(port)
        ]
        
        if background:
            self._print("Starting in background...", Colors.YELLOW)
            subprocess.Popen(
                cmd,
                stdout=open('/tmp/distributed_server.log', 'w'),
                stderr=subprocess.STDOUT
            )
            self._print(f"\n✓ Server started in background", Colors.GREEN)
            self._print(f"Log: tail -f /tmp/distributed_server.log")
        else:
            self._print("Starting in foreground...", Colors.YELLOW)
            subprocess.run(cmd)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description='MLX Distributed Inference CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mlx-cli health                          # Check server health
  mlx-cli list                            # List available models
  mlx-cli download MODEL_NAME             # Download model in background
  mlx-cli download-status [MODEL_NAME]    # Check download status
  mlx-cli load MODEL_NAME                 # Load/swap model
  mlx-cli test "Your prompt"              # Test inference
  mlx-cli start MODEL_NAME --background   # Start server
        """
    )
    
    parser.add_argument('--api-url', default='http://localhost:8080', help='API URL')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Health command
    subparsers.add_parser('health', help='Check server health')
    
    # List models command
    subparsers.add_parser('list', help='List available models')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download model in background')
    download_parser.add_argument('model', help='Model name to download')
    
    # Download status command
    status_parser = subparsers.add_parser('download-status', help='Check download status')
    status_parser.add_argument('model', nargs='?', help='Model name (optional)')
    
    # Load model command
    load_parser = subparsers.add_parser('load', help='Load/swap model')
    load_parser.add_argument('model', help='Model name to load')
    load_parser.add_argument('--drain-timeout', type=int, default=30, help='Drain timeout in seconds')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test inference')
    test_parser.add_argument('--prompt', default='Hello', help='Test prompt')
    test_parser.add_argument('--max-tokens', type=int, default=20, help='Max tokens')
    
    # Start server command
    start_parser = subparsers.add_parser('start', help='Start server')
    start_parser.add_argument('model', help='Model to load')
    start_parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    start_parser.add_argument('--background', action='store_true', help='Run in background')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    cli = DistributedCLI(args.api_url)
    
    # Execute command
    if args.command == 'health':
        cli.health()
    elif args.command == 'list':
        cli.list_models()
    elif args.command == 'download':
        cli.download(args.model)
    elif args.command == 'download-status':
        cli.download_status(args.model)
    elif args.command == 'load':
        cli.load_model(args.model, args.drain_timeout)
    elif args.command == 'test':
        cli.test(args.prompt, args.max_tokens)
    elif args.command == 'start':
        cli.start_server(args.model, args.port, args.background)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
