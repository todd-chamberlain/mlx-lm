#!/usr/bin/env python3
"""
Hybrid Deployment Orchestrator for MLX Distributed Inference

Manages N worker nodes over network/thunderbolt with Docker services.
Reads configuration from config/nodes.yaml.
"""

import os
import sys
import yaml
import subprocess
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional

# Colors for output
class Colors:
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'

class NodeOrchestrator:
    """Orchestrates distributed MLX workers across N nodes"""

    def __init__(self, config_path: str = "config/nodes.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.coordinator = self.config.get('coordinator', {})
        self.workers = self.config.get('workers', [])

    def _load_config(self) -> dict:
        """Load node configuration"""
        if not os.path.exists(self.config_path):
            print(f"{Colors.RED}❌ Config file not found: {self.config_path}{Colors.NC}")
            sys.exit(1)

        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def start_docker_services(self):
        """Start Docker services (coordinator, API, monitoring)"""
        print(f"{Colors.BLUE}[1/3] Starting Docker services...{Colors.NC}")
        print("  Starting: Redis, Coordinator, API, Prometheus, Grafana")

        # Check if Docker is running
        try:
            subprocess.run(["docker", "info"], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            print(f"{Colors.RED}  ❌ Docker is not running!{Colors.NC}")
            print("  Please start Docker Desktop and try again.")
            return False

        # Start Docker Compose
        try:
            subprocess.run(
                ["docker-compose", "-f", "docker-compose.hybrid.yml", "up", "-d"],
                check=True
            )
            print("  ✅ Docker services started")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}  ❌ Failed to start Docker services{Colors.NC}")
            return False

        # Wait for services to be healthy
        print("  Waiting for services to be healthy...")
        time.sleep(10)

        # Check coordinator health
        if not self._wait_for_coordinator():
            return False

        # Check API health
        if not self._wait_for_api():
            return False

        return True

    def _wait_for_coordinator(self, max_wait: int = 30) -> bool:
        """Wait for coordinator to be ready"""
        waited = 0
        while waited < max_wait:
            try:
                result = subprocess.run(
                    ["docker", "exec", "mlx-coordinator", "python", "-c",
                     "import grpc; channel = grpc.insecure_channel('localhost:50051'); "
                     "grpc.channel_ready_future(channel).result(timeout=5); channel.close()"],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode == 0:
                    print(f"  {Colors.GREEN}✅ Coordinator is ready{Colors.NC}")
                    return True
            except:
                pass
            time.sleep(2)
            waited += 2

        print(f"{Colors.RED}  ❌ Coordinator failed to start{Colors.NC}")
        print("  Check logs: docker logs mlx-coordinator")
        return False

    def _wait_for_api(self, max_wait: int = 30) -> bool:
        """Wait for API to be ready"""
        waited = 0
        while waited < max_wait:
            try:
                response = requests.get("http://localhost:8000/health", timeout=5)
                if response.status_code == 200:
                    print(f"  {Colors.GREEN}✅ API is ready{Colors.NC}")
                    return True
            except:
                pass
            time.sleep(2)
            waited += 2

        print(f"{Colors.RED}  ❌ API failed to start{Colors.NC}")
        print("  Check logs: docker logs mlx-api")
        return False

    def start_workers(self):
        """Start all configured worker nodes"""
        print(f"{Colors.BLUE}[2/3] Starting {len(self.workers)} worker node(s)...{Colors.NC}")

        coordinator_host = self.coordinator.get('host', 'localhost')
        coordinator_port = self.coordinator.get('port', 50051)

        started_count = 0
        for worker in self.workers:
            worker_id = worker['id']
            host = worker['host']
            port = worker['port']
            connection_type = worker.get('connection_type', 'ssh')

            print(f"\n  Starting {worker_id} on {host}...")

            if connection_type == 'local':
                success = self._start_local_worker(worker, coordinator_host, coordinator_port)
            elif connection_type in ['ssh', 'thunderbolt']:
                success = self._start_remote_worker(worker, coordinator_host, coordinator_port)
            else:
                print(f"    {Colors.YELLOW}⚠️  Unknown connection type: {connection_type}{Colors.NC}")
                continue

            if success:
                started_count += 1
                print(f"    {Colors.GREEN}✅ {worker_id} started{Colors.NC}")
            else:
                print(f"    {Colors.RED}❌ Failed to start {worker_id}{Colors.NC}")

        print(f"\n  Started {started_count}/{len(self.workers)} workers")
        return started_count > 0

    def _start_local_worker(self, worker: dict, coordinator_host: str, coordinator_port: int) -> bool:
        """Start a local worker"""
        worker_id = worker['id']
        port = worker['port']

        # Kill existing worker
        subprocess.run(["pkill", "-f", f"worker.main.*{worker_id}"], capture_output=True)
        time.sleep(2)

        # Start worker
        try:
            with open(f"logs/{worker_id}.log", "w") as log_file:
                subprocess.Popen(
                    [
                        "./venv/bin/python", "-m", "worker.main",
                        "--worker-id", worker_id,
                        "--port", str(port),
                        "--coordinator", f"{coordinator_host}:{coordinator_port}"
                    ],
                    stdout=log_file,
                    stderr=subprocess.STDOUT
                )
            time.sleep(3)
            return True
        except Exception as e:
            print(f"      Error: {e}")
            return False

    def _start_remote_worker(self, worker: dict, coordinator_host: str, coordinator_port: int) -> bool:
        """Start a remote worker via SSH"""
        worker_id = worker['id']
        host = worker['host']
        port = worker['port']
        ssh_user = worker.get('ssh_user', os.getenv('USER'))
        work_dir = worker.get('work_dir', '~/mlx-distributed-inference')

        # For remote workers, coordinator should use the host's IP, not localhost
        if coordinator_host == 'localhost':
            # Get local IP
            result = subprocess.run(
                ["hostname", "-I"],
                capture_output=True,
                text=True
            )
            coordinator_host = result.stdout.strip().split()[0] if result.stdout else "10.201.20.10"

        # Kill existing worker
        subprocess.run(
            ["ssh", f"{ssh_user}@{host}" if ssh_user else host,
             f"pkill -f 'worker.main.*{worker_id}' || true"],
            capture_output=True
        )
        time.sleep(2)

        # Start worker
        cmd = [
            "ssh", f"{ssh_user}@{host}" if ssh_user else host,
            f"cd {work_dir} && "
            f"source venv/bin/activate && "
            f"nohup python -m worker.main "
            f"--worker-id {worker_id} "
            f"--port {port} "
            f"--coordinator {coordinator_host}:{coordinator_port} "
            f"> logs/{worker_id}.log 2>&1 &"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0
        except Exception as e:
            print(f"      Error: {e}")
            return False

    def verify_health(self):
        """Verify system health"""
        print(f"{Colors.BLUE}[3/3] Verifying system health...{Colors.NC}")
        time.sleep(5)

        try:
            response = requests.get("http://localhost:8000/health", timeout=10)
            health = response.json()

            status = health.get('status', 'unknown')
            total_workers = health.get('total_workers', 0)
            healthy_workers = health.get('healthy_workers', 0)

            print(f"\nSystem Status:")
            print(f"==============")
            print(f"  Status: {status}")
            print(f"  Total Workers: {total_workers}")
            print(f"  Healthy Workers: {healthy_workers}")
            print()

            if status == 'healthy' and healthy_workers >= len(self.workers):
                print(f"{Colors.GREEN}✅ Hybrid system is healthy and ready!{Colors.NC}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠️  System partially healthy ({healthy_workers}/{len(self.workers)} workers){Colors.NC}")
                return False

        except Exception as e:
            print(f"{Colors.RED}❌ Failed to check health: {e}{Colors.NC}")
            return False

    def print_info(self):
        """Print system information"""
        print()
        print("Service URLs:")
        print("=============")
        print("  API:          http://localhost:8000")
        print("  Coordinator:  localhost:50051 (gRPC)")
        print("  Prometheus:   http://localhost:9090")
        print("  Grafana:      http://localhost:3000 (admin/admin)")
        print()
        print("Worker Nodes:")
        print("=============")
        for worker in self.workers:
            print(f"  {worker['id']:20s} {worker['host']}:{worker['port']}")
        print()
        print("Useful Commands:")
        print("================")
        print("  Health:       curl http://localhost:8000/health | jq .")
        print("  Models:       curl http://localhost:8000/models | jq .")
        print("  Docker logs:  docker logs mlx-coordinator")
        print("  Docker logs:  docker logs mlx-api")
        for worker in self.workers:
            if worker.get('connection_type') == 'local':
                print(f"  Worker logs:  tail -f logs/{worker['id']}.log")
            else:
                host = worker['host']
                work_dir = worker.get('work_dir', '~/mlx-distributed-inference')
                print(f"  Worker logs:  ssh {host} 'tail -f {work_dir}/logs/{worker['id']}.log'")
        print()

    def start(self):
        """Start the entire hybrid system"""
        print("🚀 Starting MLX Hybrid Distributed System")
        print("==========================================")
        print(f"Configuration: {self.config_path}")
        print(f"Workers: {len(self.workers)} node(s)")
        print()

        # Start Docker services
        if not self.start_docker_services():
            return False

        # Start workers
        if not self.start_workers():
            return False

        # Verify health
        self.verify_health()

        # Print info
        self.print_info()

        return True


def main():
    """Main entry point"""
    orchestrator = NodeOrchestrator()
    success = orchestrator.start()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
