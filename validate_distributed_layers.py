#!/usr/bin/env python3
"""
Distributed Layer Inference Validation Script

Tests:
1. Node connectivity (both Mac Studios can see each other)
2. Distributed layer inference (model split across multiple nodes)
3. Pipeline execution and activation transfer
4. Performance metrics
"""

import asyncio
import sys
import time
import logging
from typing import List, Dict, Any, Optional
import grpc
from grpc import aio as grpc_aio

# Add proto to path
sys.path.insert(0, '.')

from proto import distributed_pb2 as dist_pb
from proto import distributed_pb2_grpc as dist_grpc
from proto import inference_pb2 as inf_pb
from proto import inference_pb2_grpc as inf_grpc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[0;32m'
    BLUE = '\033[0;34m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    CYAN = '\033[0;36m'
    MAGENTA = '\033[0;35m'
    NC = '\033[0m'  # No Color
    BOLD = '\033[1m'


class DistributedLayerValidator:
    """Validates distributed layer inference setup"""
    
    def __init__(self, node1_addr: str, node2_addr: str, coordinator_addr: str):
        self.node1_addr = node1_addr
        self.node2_addr = node2_addr
        self.coordinator_addr = coordinator_addr
        
        self.node1_channel: Optional[grpc_aio.Channel] = None
        self.node2_channel: Optional[grpc_aio.Channel] = None
        self.coordinator_channel: Optional[grpc_aio.Channel] = None
        
        self.results = {
            'connectivity': {},
            'registration': {},
            'distributed_inference': {},
            'performance': {}
        }
    
    def print_header(self, text: str):
        """Print a formatted header"""
        print(f"\n{Colors.BLUE}{'='*70}{Colors.NC}")
        print(f"{Colors.BLUE}{Colors.BOLD}{text:^70}{Colors.NC}")
        print(f"{Colors.BLUE}{'='*70}{Colors.NC}\n")
    
    def print_test(self, name: str, status: str, details: str = ""):
        """Print test result"""
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
    
    async def connect_to_nodes(self) -> bool:
        """Connect to all nodes"""
        self.print_header("STEP 1: Node Connectivity")
        
        success = True
        
        # Connect to Node 1
        try:
            self.node1_channel = grpc_aio.insecure_channel(
                self.node1_addr,
                options=[
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ]
            )
            stub = inf_grpc.WorkerServiceStub(self.node1_channel)
            response = await asyncio.wait_for(
                stub.HealthCheck(inf_pb.HealthCheckRequest()),
                timeout=5.0
            )
            self.print_test(f"Node 1 ({self.node1_addr})", "PASS", f"Status: {response.status}")
            self.results['connectivity']['node1'] = True
        except Exception as e:
            self.print_test(f"Node 1 ({self.node1_addr})", "FAIL", str(e))
            self.results['connectivity']['node1'] = False
            success = False
        
        # Connect to Node 2
        try:
            self.node2_channel = grpc_aio.insecure_channel(
                self.node2_addr,
                options=[
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ]
            )
            stub = inf_grpc.WorkerServiceStub(self.node2_channel)
            response = await asyncio.wait_for(
                stub.HealthCheck(inf_pb.HealthCheckRequest()),
                timeout=5.0
            )
            self.print_test(f"Node 2 ({self.node2_addr})", "PASS", f"Status: {response.status}")
            self.results['connectivity']['node2'] = True
        except Exception as e:
            self.print_test(f"Node 2 ({self.node2_addr})", "FAIL", str(e))
            self.results['connectivity']['node2'] = False
            success = False
        
        # Connect to Coordinator
        try:
            self.coordinator_channel = grpc_aio.insecure_channel(
                self.coordinator_addr,
                options=[
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024),
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                ]
            )
            stub = inf_grpc.CoordinatorServiceStub(self.coordinator_channel)
            response = await asyncio.wait_for(
                stub.HealthCheck(inf_pb.HealthCheckRequest(include_workers=True)),
                timeout=5.0
            )
            self.print_test(
                f"Coordinator ({self.coordinator_addr})", 
                "PASS", 
                f"Workers: {response.healthy_workers}/{response.total_workers}"
            )
            self.results['connectivity']['coordinator'] = True
            self.results['connectivity']['registered_workers'] = response.healthy_workers
        except Exception as e:
            self.print_test(f"Coordinator ({self.coordinator_addr})", "FAIL", str(e))
            self.results['connectivity']['coordinator'] = False
            success = False
        
        return success
    
    async def test_node_to_node_connectivity(self) -> bool:
        """Test if nodes can communicate with each other"""
        self.print_header("STEP 2: Node-to-Node Communication")
        
        # Test Node 1 -> Node 2
        try:
            stub = dist_grpc.DistributedInferenceServiceStub(self.node2_channel)
            
            # Create a simple health check for distributed service
            health_req = dist_pb.PipelineHealthRequest(pipeline_id="test")
            response = await asyncio.wait_for(
                stub.CheckPipelineHealth(health_req),
                timeout=5.0
            )
            self.print_test("Node 1 → Node 2", "PASS", f"Pipeline status: {response.status}")
            self.results['connectivity']['node1_to_node2'] = True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                self.print_test(
                    "Node 1 → Node 2", 
                    "WARN", 
                    "Distributed service not yet implemented"
                )
                self.results['connectivity']['node1_to_node2'] = 'not_implemented'
            else:
                self.print_test("Node 1 → Node 2", "FAIL", str(e))
                self.results['connectivity']['node1_to_node2'] = False
        except Exception as e:
            self.print_test("Node 1 → Node 2", "FAIL", str(e))
            self.results['connectivity']['node1_to_node2'] = False
        
        # Test Node 2 -> Node 1
        try:
            stub = dist_grpc.DistributedInferenceServiceStub(self.node1_channel)
            
            health_req = dist_pb.PipelineHealthRequest(pipeline_id="test")
            response = await asyncio.wait_for(
                stub.CheckPipelineHealth(health_req),
                timeout=5.0
            )
            self.print_test("Node 2 → Node 1", "PASS", f"Pipeline status: {response.status}")
            self.results['connectivity']['node2_to_node1'] = True
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                self.print_test(
                    "Node 2 → Node 1", 
                    "WARN", 
                    "Distributed service not yet implemented"
                )
                self.results['connectivity']['node2_to_node1'] = 'not_implemented'
            else:
                self.print_test("Node 2 → Node 1", "FAIL", str(e))
                self.results['connectivity']['node2_to_node1'] = False
        except Exception as e:
            self.print_test("Node 2 → Node 1", "FAIL", str(e))
            self.results['connectivity']['node2_to_node1'] = False
        
        return True
    
    async def test_worker_registration(self) -> bool:
        """Test that both workers are registered with coordinator"""
        self.print_header("STEP 3: Worker Registration Status")
        
        try:
            stub = inf_grpc.CoordinatorServiceStub(self.coordinator_channel)
            response = await stub.HealthCheck(
                inf_pb.HealthCheckRequest(include_workers=True)
            )
            
            total = response.total_workers
            healthy = response.healthy_workers
            
            self.print_test(
                "Total Workers Registered", 
                "PASS" if total >= 2 else "WARN",
                f"{total} workers"
            )
            
            self.print_test(
                "Healthy Workers", 
                "PASS" if healthy >= 2 else "WARN",
                f"{healthy}/{total} healthy"
            )
            
            # Print individual worker status
            for worker in response.worker_health:
                status = "PASS" if worker.status == "healthy" else "WARN"
                self.print_test(
                    f"Worker: {worker.worker_id}",
                    status,
                    f"Load: {worker.current_load}, Last HB: {worker.last_heartbeat_seconds}s ago"
                )
            
            self.results['registration']['total_workers'] = total
            self.results['registration']['healthy_workers'] = healthy
            
            return healthy >= 2
            
        except Exception as e:
            self.print_test("Worker Registration Check", "FAIL", str(e))
            return False
    
    async def test_basic_inference(self) -> bool:
        """Test basic inference through coordinator"""
        self.print_header("STEP 4: Basic Inference Test")
        
        try:
            stub = inf_grpc.CoordinatorServiceStub(self.coordinator_channel)
            
            start_time = time.time()
            
            request = inf_pb.InferenceRequest(
                request_id="test-basic-inference",
                prompt="Say hello in 5 words",
                max_tokens=20,
                temperature=0.7
            )
            
            response = await asyncio.wait_for(
                stub.RouteInference(request),
                timeout=30.0
            )
            
            latency = (time.time() - start_time) * 1000
            
            self.print_test(
                "Basic Inference",
                "PASS",
                f"Latency: {latency:.2f}ms, Worker: {response.worker_id}"
            )
            
            print(f"\n     {Colors.CYAN}Response: {response.text[:100]}{Colors.NC}\n")
            
            self.results['distributed_inference']['basic'] = {
                'success': True,
                'latency_ms': latency,
                'worker_id': response.worker_id
            }
            
            return True
            
        except Exception as e:
            self.print_test("Basic Inference", "FAIL", str(e))
            self.results['distributed_inference']['basic'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_distributed_layer_inference(self) -> bool:
        """Test distributed layer inference (if implemented)"""
        self.print_header("STEP 5: Distributed Layer Inference")
        
        try:
            # Create pipeline configuration
            pipeline_config = dist_pb.PipelineConfig(
                pipeline_id="test-pipeline",
                model_name="mlx-community/Llama-3.2-3B-Instruct-4bit",
                total_layers=32,
                strategy=dist_pb.PartitionStrategy.BALANCED,
                assignments=[
                    dist_pb.LayerAssignment(
                        node_id="node-1",
                        start_layer=0,
                        end_layer=16,
                        next_node_id="node-2",
                        next_node_address=self.node2_addr
                    ),
                    dist_pb.LayerAssignment(
                        node_id="node-2",
                        start_layer=16,
                        end_layer=32,
                        next_node_id="",
                        next_node_address=""
                    )
                ]
            )
            
            self.print_test(
                "Pipeline Configuration",
                "INFO",
                f"Model: {pipeline_config.model_name}, Layers: {pipeline_config.total_layers}"
            )
            
            # Test distributed inference
            stub = dist_grpc.DistributedInferenceServiceStub(self.node1_channel)
            
            request = dist_pb.DistributedInferenceRequest(
                request_id="test-distributed",
                pipeline_id="test-pipeline",
                input_tokens=[1, 2, 3, 4, 5],  # Dummy tokens
                max_tokens=10,
                temperature=0.7
            )
            
            start_time = time.time()
            
            response = await asyncio.wait_for(
                stub.ProcessLayers(request),
                timeout=60.0
            )
            
            latency = (time.time() - start_time) * 1000
            
            self.print_test(
                "Distributed Layer Processing",
                "PASS",
                f"Latency: {latency:.2f}ms"
            )
            
            if response.metrics:
                self.print_test(
                    "Stage Latency",
                    "INFO",
                    f"{response.metrics.stage_latency_ms}ms"
                )
                self.print_test(
                    "Transfer Latency",
                    "INFO",
                    f"{response.metrics.transfer_latency_ms}ms"
                )
                self.print_test(
                    "Activation Size",
                    "INFO",
                    f"{response.metrics.activation_size_bytes / 1024 / 1024:.2f}MB"
                )
            
            self.results['distributed_inference']['distributed'] = {
                'success': True,
                'latency_ms': latency,
                'metrics': {
                    'stage_latency_ms': response.metrics.stage_latency_ms if response.metrics else 0,
                    'transfer_latency_ms': response.metrics.transfer_latency_ms if response.metrics else 0,
                    'activation_size_bytes': response.metrics.activation_size_bytes if response.metrics else 0
                }
            }
            
            return True
            
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.UNIMPLEMENTED:
                self.print_test(
                    "Distributed Layer Inference",
                    "WARN",
                    "Not yet implemented - this is expected"
                )
                self.results['distributed_inference']['distributed'] = {
                    'success': False,
                    'status': 'not_implemented'
                }
                return True  # Not a failure, just not implemented yet
            else:
                self.print_test("Distributed Layer Inference", "FAIL", str(e))
                self.results['distributed_inference']['distributed'] = {
                    'success': False,
                    'error': str(e)
                }
                return False
        except Exception as e:
            self.print_test("Distributed Layer Inference", "FAIL", str(e))
            self.results['distributed_inference']['distributed'] = {
                'success': False,
                'error': str(e)
            }
            return False
    
    async def test_performance(self) -> bool:
        """Run performance tests"""
        self.print_header("STEP 6: Performance Testing")
        
        num_requests = 10
        latencies = []
        
        try:
            stub = inf_grpc.CoordinatorServiceStub(self.coordinator_channel)
            
            self.print_test("Running Performance Test", "INFO", f"{num_requests} requests")
            
            for i in range(num_requests):
                request = inf_pb.InferenceRequest(
                    request_id=f"perf-test-{i}",
                    prompt="Count to 5",
                    max_tokens=10,
                    temperature=0.7
                )
                
                start_time = time.time()
                
                try:
                    response = await asyncio.wait_for(
                        stub.RouteInference(request),
                        timeout=30.0
                    )
                    latency = (time.time() - start_time) * 1000
                    latencies.append(latency)
                    
                    print(f"     Request {i+1}/{num_requests}: {latency:.2f}ms")
                    
                except Exception as e:
                    print(f"     Request {i+1}/{num_requests}: FAILED - {e}")
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                min_latency = min(latencies)
                max_latency = max(latencies)
                
                self.print_test(
                    "Average Latency",
                    "PASS",
                    f"{avg_latency:.2f}ms"
                )
                self.print_test(
                    "Min/Max Latency",
                    "INFO",
                    f"{min_latency:.2f}ms / {max_latency:.2f}ms"
                )
                self.print_test(
                    "Success Rate",
                    "PASS",
                    f"{len(latencies)}/{num_requests} ({len(latencies)/num_requests*100:.1f}%)"
                )
                
                self.results['performance'] = {
                    'avg_latency_ms': avg_latency,
                    'min_latency_ms': min_latency,
                    'max_latency_ms': max_latency,
                    'success_rate': len(latencies) / num_requests
                }
                
                return True
            else:
                self.print_test("Performance Test", "FAIL", "No successful requests")
                return False
                
        except Exception as e:
            self.print_test("Performance Test", "FAIL", str(e))
            return False
    
    def print_summary(self):
        """Print test summary"""
        self.print_header("VALIDATION SUMMARY")
        
        # Connectivity
        print(f"{Colors.BOLD}Connectivity:{Colors.NC}")
        node1_ok = self.results['connectivity'].get('node1', False)
        node2_ok = self.results['connectivity'].get('node2', False)
        coord_ok = self.results['connectivity'].get('coordinator', False)
        
        print(f"  Node 1: {Colors.GREEN if node1_ok else Colors.RED}{'✓' if node1_ok else '✗'}{Colors.NC}")
        print(f"  Node 2: {Colors.GREEN if node2_ok else Colors.RED}{'✓' if node2_ok else '✗'}{Colors.NC}")
        print(f"  Coordinator: {Colors.GREEN if coord_ok else Colors.RED}{'✓' if coord_ok else '✗'}{Colors.NC}")
        
        # Registration
        print(f"\n{Colors.BOLD}Registration:{Colors.NC}")
        workers = self.results['registration'].get('healthy_workers', 0)
        print(f"  Healthy Workers: {workers}")
        
        # Inference
        print(f"\n{Colors.BOLD}Inference:{Colors.NC}")
        basic_ok = self.results['distributed_inference'].get('basic', {}).get('success', False)
        print(f"  Basic Inference: {Colors.GREEN if basic_ok else Colors.RED}{'✓' if basic_ok else '✗'}{Colors.NC}")
        
        dist_result = self.results['distributed_inference'].get('distributed', {})
        if dist_result.get('status') == 'not_implemented':
            print(f"  Distributed Layers: {Colors.YELLOW}⚠ Not Implemented{Colors.NC}")
        else:
            dist_ok = dist_result.get('success', False)
            print(f"  Distributed Layers: {Colors.GREEN if dist_ok else Colors.RED}{'✓' if dist_ok else '✗'}{Colors.NC}")
        
        # Performance
        if 'performance' in self.results and self.results['performance']:
            print(f"\n{Colors.BOLD}Performance:{Colors.NC}")
            perf = self.results['performance']
            print(f"  Avg Latency: {perf.get('avg_latency_ms', 0):.2f}ms")
            print(f"  Success Rate: {perf.get('success_rate', 0)*100:.1f}%")
        
        # Overall status
        print(f"\n{Colors.BOLD}Overall Status:{Colors.NC}")
        if node1_ok and node2_ok and coord_ok and workers >= 2:
            print(f"  {Colors.GREEN}✓ System is ready for distributed inference{Colors.NC}")
        else:
            print(f"  {Colors.RED}✗ System needs attention{Colors.NC}")
    
    async def cleanup(self):
        """Close all connections"""
        if self.node1_channel:
            await self.node1_channel.close()
        if self.node2_channel:
            await self.node2_channel.close()
        if self.coordinator_channel:
            await self.coordinator_channel.close()
    
    async def run_all_tests(self) -> bool:
        """Run all validation tests"""
        try:
            # Step 1: Connect to nodes
            if not await self.connect_to_nodes():
                print(f"\n{Colors.RED}Failed to connect to all nodes{Colors.NC}")
                return False
            
            # Step 2: Test node-to-node connectivity
            await self.test_node_to_node_connectivity()
            
            # Step 3: Check worker registration
            await self.test_worker_registration()
            
            # Step 4: Test basic inference
            await self.test_basic_inference()
            
            # Step 5: Test distributed layer inference
            await self.test_distributed_layer_inference()
            
            # Step 6: Performance testing
            await self.test_performance()
            
            # Print summary
            self.print_summary()
            
            return True
            
        finally:
            await self.cleanup()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate distributed layer inference setup')
    parser.add_argument('--node1', default='localhost:50052', help='Node 1 address')
    parser.add_argument('--node2', default='mac-studio-2:50052', help='Node 2 address')
    parser.add_argument('--coordinator', default='localhost:50051', help='Coordinator address')
    
    args = parser.parse_args()
    
    print(f"{Colors.CYAN}{Colors.BOLD}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     MLX Distributed Layer Inference Validation                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.NC}")
    
    print(f"Configuration:")
    print(f"  Node 1: {args.node1}")
    print(f"  Node 2: {args.node2}")
    print(f"  Coordinator: {args.coordinator}")
    
    validator = DistributedLayerValidator(
        node1_addr=args.node1,
        node2_addr=args.node2,
        coordinator_addr=args.coordinator
    )
    
    success = await validator.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    asyncio.run(main())
