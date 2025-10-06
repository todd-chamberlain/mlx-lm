"""
Coordinator Client for MLX Worker

This module handles worker registration, heartbeat sending, graceful shutdown,
and network discovery for connecting to the coordinator service.
"""

import asyncio
import logging
import time
import socket
from typing import Optional, Dict, Any
import grpc
from grpc import aio as grpc_aio

# Import generated gRPC code
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from proto.inference_pb2_grpc import CoordinatorServiceStub
# Use async stub for async operations
from proto import inference_pb2_grpc
from proto import inference_pb2 as pb
from proto.grpc_base import GRPCClientBase as GrpcClient

from .model_manager import ModelManager
from .inference_engine import InferenceEngine

logger = logging.getLogger(__name__)


class CoordinatorClient:
    """Client for communicating with the coordinator service"""
    
    def __init__(self, worker_id: str, model_manager: ModelManager, 
                 inference_engine: InferenceEngine, worker_port: int,
                 coordinator_host: str = "localhost:50051",
                 heartbeat_interval: int = 10):
        self.worker_id = worker_id
        self.model_manager = model_manager
        self.inference_engine = inference_engine
        self.worker_port = worker_port
        self.coordinator_host = coordinator_host
        self.heartbeat_interval = heartbeat_interval
        
        self.channel: Optional[grpc_aio.Channel] = None
        self.stub: Optional[CoordinatorServiceStub] = None
        self.registered = False
        self.heartbeat_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Connection retry settings
        self.max_retries = 5
        self.retry_delay = 5.0
        self.connection_timeout = 10.0
        
        logger.info(f"Initialized CoordinatorClient for worker {worker_id}")
        logger.info(f"Coordinator: {coordinator_host}, Worker port: {worker_port}")
    
    async def connect(self) -> bool:
        """Connect to the coordinator service"""
        try:
            # Create gRPC channel with lenient keepalive options to avoid too_many_pings
            options = [
                # Disable keepalive to avoid too_many_pings error
                # ('grpc.keepalive_time_ms', 60000),  # Disabled
                # ('grpc.keepalive_timeout_ms', 10000),  # Disabled
                # ('grpc.keepalive_permit_without_calls', False),  # Disabled
                ('grpc.http2.max_pings_without_data', 0),
                ('grpc.http2.min_time_between_pings_ms', 120000),  # 2 minutes
                ('grpc.max_receive_message_length', 4 * 1024 * 1024),
                ('grpc.max_send_message_length', 4 * 1024 * 1024),
            ]
            
            self.channel = grpc_aio.insecure_channel(self.coordinator_host, options=options)
            # Use async stub for async operations
            self.stub = inference_pb2_grpc.CoordinatorServiceStub(self.channel)
            
            # Test connection with a simple health check
            try:
                await asyncio.wait_for(
                    self.stub.HealthCheck(pb.HealthCheckRequest()),
                    timeout=self.connection_timeout
                )
                logger.info(f"Successfully connected to coordinator at {self.coordinator_host}")
                return True
                
            except asyncio.TimeoutError:
                logger.error(f"Connection timeout to coordinator at {self.coordinator_host}")
                await self.disconnect()
                return False
            
        except Exception as e:
            logger.error(f"Failed to connect to coordinator: {e}")
            await self.disconnect()
            return False
    
    async def disconnect(self):
        """Disconnect from the coordinator service"""
        if self.channel:
            await self.channel.close()
            self.channel = None
            self.stub = None
            logger.info("Disconnected from coordinator")
    
    def _get_worker_address(self) -> str:
        """Get the worker's network address"""
        try:
            # Get local IP address
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                # Connect to a remote address to determine local IP
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            
            return f"{local_ip}:{self.worker_port}"
            
        except Exception as e:
            logger.warning(f"Failed to determine local IP: {e}")
            return f"localhost:{self.worker_port}"
    
    def _create_device_info(self) -> pb.DeviceInfo:
        """Create device info message"""
        device_info = self.model_manager.get_device_info()
        
        return pb.DeviceInfo(
            model=device_info.model,
            memory_gb=device_info.memory_gb,
            gpu_cores=device_info.gpu_cores,
            os_version=device_info.os_version,
            mlx_version=device_info.mlx_version
        )
    
    def _get_supported_models(self) -> list:
        """Get list of supported/loaded models"""
        loaded_models = self.model_manager.list_loaded_models()
        return list(loaded_models.keys())
    
    async def register(self, max_concurrent: int = 4) -> bool:
        """Register worker with the coordinator"""
        if not self.stub:
            logger.error("Not connected to coordinator")
            return False
        
        try:
            # Create registration request
            registration = pb.WorkerRegistration(
                worker_id=self.worker_id,
                address=self._get_worker_address(),
                device_info=self._create_device_info(),
                max_concurrent=max_concurrent,
                supported_models=self._get_supported_models()
            )
            
            logger.info(f"Registering worker {self.worker_id} with coordinator...")
            
            # Send registration request (sync stub, run in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, self.stub.RegisterWorker, registration)
            
            if response.success:
                self.registered = True
                logger.info(f"Successfully registered with coordinator: {response.message}")
                logger.info(f"Coordinator ID: {response.coordinator_id}")
                return True
            else:
                logger.error(f"Registration failed: {response.message}")
                return False
                
        except grpc.RpcError as e:
            logger.error(f"gRPC error during registration: {e.code()}: {e.details()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during registration: {e}")
            return False
    
    async def start_heartbeat(self):
        """Start the heartbeat task"""
        if self.heartbeat_task and not self.heartbeat_task.done():
            logger.warning("Heartbeat task already running")
            return
        
        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(f"Started heartbeat task with {self.heartbeat_interval}s interval")
    
    async def stop_heartbeat(self):
        """Stop the heartbeat task"""
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
            try:
                await self.heartbeat_task
            except asyncio.CancelledError:
                pass
            self.heartbeat_task = None
            logger.info("Stopped heartbeat task")
    
    async def _heartbeat_loop(self):
        """Main heartbeat loop"""
        consecutive_failures = 0
        max_failures = 3
        
        while not self.shutdown_event.is_set():
            try:
                if not self.registered:
                    logger.warning("Worker not registered, skipping heartbeat")
                    await asyncio.sleep(self.heartbeat_interval)
                    continue
                
                # Send heartbeat
                success = await self._send_heartbeat()
                
                if success:
                    consecutive_failures = 0
                else:
                    consecutive_failures += 1
                    logger.warning(f"Heartbeat failed ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        logger.error("Too many consecutive heartbeat failures, attempting re-registration")
                        self.registered = False
                        
                        # Try to reconnect and re-register
                        if await self.connect():
                            await self.register()
                        
                        consecutive_failures = 0
                
                # Wait for next heartbeat
                await asyncio.sleep(self.heartbeat_interval)
                
            except asyncio.CancelledError:
                logger.info("Heartbeat loop cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in heartbeat loop: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _send_heartbeat(self) -> bool:
        """Send a single heartbeat to the coordinator"""
        if not self.stub:
            return False
        
        try:
            # Get current metrics
            metrics = self.inference_engine.get_metrics()
            memory_usage = self.model_manager.get_memory_usage()
            
            # Create status message
            status = pb.WorkerStatus(
                status="healthy",  # TODO: Add health determination logic
                current_load=metrics['active_requests'],
                memory_usage_percent=memory_usage['percent'],
                uptime_seconds=int(time.time() - self.inference_engine.metrics.total_requests)  # Approximate
            )
            
            # Create metrics message
            worker_metrics = pb.WorkerMetrics(
                avg_latency_ms=metrics['avg_latency_ms'],
                tokens_per_second=metrics['tokens_per_second'],
                total_requests=metrics['total_requests'],
                failed_requests=metrics['failed_requests'],
                active_requests=metrics['active_requests'],
                memory_usage_gb=memory_usage['used_gb']
            )
            
            # Create heartbeat request
            heartbeat_request = pb.HeartbeatRequest(
                worker_id=self.worker_id,
                status=status,
                metrics=worker_metrics
            )
            
            # Send heartbeat
            response = await asyncio.wait_for(
                self.stub.Heartbeat(heartbeat_request),
                timeout=5.0
            )
            
            if response.acknowledged:
                logger.debug(f"Heartbeat acknowledged: {response.message}")
                
                # Update heartbeat interval if coordinator suggests a different one
                if response.heartbeat_interval > 0:
                    self.heartbeat_interval = response.heartbeat_interval
                
                return True
            else:
                logger.warning(f"Heartbeat not acknowledged: {response.message}")
                return False
                
        except asyncio.TimeoutError:
            logger.warning("Heartbeat timeout")
            return False
        except grpc.RpcError as e:
            logger.warning(f"gRPC error during heartbeat: {e.code()}: {e.details()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during heartbeat: {e}")
            return False
    
    async def deregister(self) -> bool:
        """Deregister worker from coordinator (graceful shutdown)"""
        if not self.registered or not self.stub:
            return True
        
        try:
            # Send a final heartbeat with "draining" status
            status = pb.WorkerStatus(
                status="draining",
                current_load=self.inference_engine.get_active_request_count(),
                memory_usage_percent=self.model_manager.get_memory_usage()['percent'],
                uptime_seconds=int(time.time())
            )
            
            heartbeat_request = pb.HeartbeatRequest(
                worker_id=self.worker_id,
                status=status,
                metrics=pb.WorkerMetrics()
            )
            
            await self.stub.Heartbeat(heartbeat_request)
            logger.info("Sent deregistration heartbeat")
            
            self.registered = False
            return True
            
        except Exception as e:
            logger.error(f"Error during deregistration: {e}")
            return False
    
    async def start(self, max_concurrent: int = 4) -> bool:
        """Start the coordinator client (connect, register, start heartbeat)"""
        try:
            # Connect to coordinator
            if not await self.connect():
                return False
            
            # Register worker
            if not await self.register(max_concurrent):
                return False
            
            # Start heartbeat
            await self.start_heartbeat()
            
            logger.info("CoordinatorClient started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start CoordinatorClient: {e}")
            return False
    
    async def stop(self):
        """Stop the coordinator client (deregister, stop heartbeat, disconnect)"""
        try:
            logger.info("Stopping CoordinatorClient...")
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Stop heartbeat
            await self.stop_heartbeat()
            
            # Deregister
            await self.deregister()
            
            # Disconnect
            await self.disconnect()
            
            logger.info("CoordinatorClient stopped")
            
        except Exception as e:
            logger.error(f"Error stopping CoordinatorClient: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to coordinator"""
        return self.channel is not None and not self.channel.get_state() == grpc.ChannelConnectivity.SHUTDOWN
    
    def is_registered(self) -> bool:
        """Check if registered with coordinator"""
        return self.registered
    
    def get_status(self) -> Dict[str, Any]:
        """Get client status information"""
        return {
            'worker_id': self.worker_id,
            'coordinator_host': self.coordinator_host,
            'connected': self.is_connected(),
            'registered': self.is_registered(),
            'heartbeat_interval': self.heartbeat_interval,
            'heartbeat_running': self.heartbeat_task is not None and not self.heartbeat_task.done()
        }


class NetworkDiscovery:
    """Helper class for discovering coordinator services on the network"""
    
    @staticmethod
    async def discover_coordinator(timeout: float = 5.0) -> Optional[str]:
        """Discover coordinator service using mDNS or network scanning"""
        # This is a simplified implementation
        # In a real deployment, you might use mDNS, Consul, or other service discovery
        
        common_ports = [50051, 50052, 50053]
        common_hosts = ['coordinator', 'coordinator.local', 'localhost']
        
        for host in common_hosts:
            for port in common_ports:
                address = f"{host}:{port}"
                if await NetworkDiscovery._test_coordinator_connection(address, timeout):
                    logger.info(f"Discovered coordinator at {address}")
                    return address
        
        logger.warning("No coordinator discovered")
        return None
    
    @staticmethod
    async def _test_coordinator_connection(address: str, timeout: float) -> bool:
        """Test if a coordinator is available at the given address"""
        try:
            channel = grpc_aio.insecure_channel(address)
            stub = CoordinatorServiceStub(channel)
            
            # Try a simple health check
            await asyncio.wait_for(
                stub.HealthCheck(pb.HealthCheckRequest()),
                timeout=timeout
            )
            
            await channel.close()
            return True
            
        except Exception:
            return False